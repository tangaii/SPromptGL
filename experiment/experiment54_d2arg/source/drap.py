"""Experiment 51.1 DC-SRA-GFR.

This module keeps the audited E1 four-branch GFR as the main path and adds a
small, optional branch that compares graph and self-loop views of each modality.
The branch uses CORR-conditioned prototypes and patient-specific gates, while
private candidates remain outside the cross-modal geometry loss.  All graph
learners and the E1 GCNs are the isolated historical implementations.
"""
from __future__ import annotations
import math
from typing import Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F

PHENO, ANAT, FUNC, CORR = 0, 1, 2, 3
DIMS = [48, 6, 10, 256]
AUX = [PHENO, ANAT, FUNC]


def rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-8))


class PLNNoPEN(nn.Module):
    """PEN-free CAM with the historical parameter names and shape."""
    def __init__(self, dims, hp, layers_module):
        super().__init__()
        self.input_data_dims = dims
        self.d_q = self.d_k = self.d_v = hp.n_hidden
        self.n_head = hp.n_head
        self.dropout = hp.dropout
        self.n_layer = hp.nlayer
        self.modal_num = hp.nmodal
        self.n_class = hp.nclass
        self.fusion_weight = nn.Parameter(torch.ones([6, 1]))
        self.InputLayer = layers_module.VariLengthInputLayer(dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []
        for i in range(self.n_layer):
            encoder = layers_module.EncodeLayer(self.d_k*self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            self.add_module(f"encode_{i}", encoder); self.Encoder.append(encoder)
            feedforward = layers_module.FeedForwardLayer(self.d_v*self.n_head, self.d_v*self.n_head, dropout=self.dropout)
            self.add_module(f"feed_{i}", feedforward); self.FeedForward.append(feedforward)
        self.Outputlayer = layers_module.OutputLayer(self.d_v*self.n_head*self.modal_num, self.d_v*self.n_head, self.n_class, self.modal_num, self.dropout)

    def forward(self, x):
        bs = x.size(0)
        x, input_attention = self.InputLayer(x)
        attn = input_attention.mean(dim=1)
        for enc, ff in zip(self.Encoder, self.FeedForward):
            x, attn_layer = enc(q=x, k=x, v=x, modal_num=self.modal_num)
            attn = attn_layer.mean(dim=1)
            x = ff(x)
        return self.Outputlayer(x.view(bs, -1), attn.view(bs, -1)) + ([],)


class LegacyFixedGFR(nn.Module):
    """E1 four-way aggregation with the historical PHENO overwrite repaired."""
    def __init__(self, graphlearn_cls, gcn_cls, utils_module, th=0.9):
        super().__init__()
        self.utils = utils_module
        self.graphs = nn.ModuleList([graphlearn_cls(d, th=th, mode='weighted-cosine') for d in DIMS])
        self.gcn = nn.ModuleList([gcn_cls(sum(DIMS), sum(DIMS)) for _ in DIMS])
        self.fusion_weight = nn.Parameter(torch.ones(4, 1))
        self.last: Dict = {}

    def forward(self, parts):
        raw = torch.cat(parts, 1)
        fused = raw
        adjs, msgs = [], []
        for m, x in enumerate(parts):
            adj = self.graphs[m](x)
            eye = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
            norm = self.utils.normalize_adj(adj + eye)
            msg = self.gcn[m](raw, norm)
            fused = fused + torch.sigmoid(self.fusion_weight[m]) * msg
            adjs.append(adj); msgs.append(msg)
        self.last = {'adjs': adjs, 'msgs': msgs, 'fused': fused, 'raw': raw}
        return fused, adjs, self.last


class ModalGCN(nn.Module):
    def __init__(self, layers_module, hidden=16):
        super().__init__()
        self.gcn1 = layers_module.GraphConv(hidden, hidden)
        self.gcn2 = layers_module.GraphConv(hidden, hidden)

    def forward(self, x, adj):
        return self.gcn2(F.relu(self.gcn1(x, adj)), adj)


class ModalBranch(nn.Module):
    """One modality's shared graph/self and private patient-only features."""
    def __init__(self, layers_module, dim, hidden=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
        self.gcn = ModalGCN(layers_module, hidden)
        self.s_proj = nn.Linear(hidden, 8)
        self.c_proj = nn.Linear(hidden, 8)

    def forward(self, x, adj, adj_perm=None):
        u = self.encoder(x)
        eye = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
        norm = adj if adj_perm is None else adj_perm
        hg = self.gcn(u, norm)
        hi = self.gcn(u, eye)
        sg = self.s_proj(hg)
        si = self.s_proj(hi)
        c = self.c_proj(u)
        return {'u': u, 'hg': hg, 'hi': hi, 'sg': sg, 'si': si, 'c': c}


class DCSRA(nn.Module):
    """E1 plus disease-conditioned selective relational alignment."""
    def __init__(self, graphlearn_cls, gcn_cls, layers_module, utils_module,
                 th=0.9, mode='E7', hidden=16):
        super().__init__()
        self.mode = mode
        self.utils = utils_module
        self.base = LegacyFixedGFR(graphlearn_cls, gcn_cls, utils_module, th=th)
        self.branches = nn.ModuleList([ModalBranch(layers_module, d, hidden) for d in DIMS])
        # 8 shared + 8 private -> original modality dimensions.
        self.write = nn.ModuleList([nn.Linear(16, d) for d in DIMS])
        # [S_self, Delta_S, prototype probabilities, ANAT/FUNC condition] = 34.
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(34, 16), nn.GELU(), nn.Linear(16, 1)) for _ in DIMS
        ])
        for g in self.gates:
            nn.init.zeros_(g[-1].bias)
        self.prototypes = nn.Parameter(torch.randn(2, 8) * 0.05)
        # 0.08 * sigmoid(-log 3) = 0.02 at initialization.
        self.rho_raw = nn.Parameter(torch.full((4,), -math.log(3.0)))
        self.last: Dict = {}

    def prototype_prob(self, corr_self):
        z = F.normalize(corr_self, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        return F.softmax(z @ p.t() / 0.2, dim=-1)

    @staticmethod
    def _normalize_adj(utils_module, adj):
        eye = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
        return utils_module.normalize_adj(adj + eye)

    def forward(self, parts, align_perm=None, new_relation_perm=None,
                force_new_self=False, intervention=None):
        base_fused, adjs, base_details = self.base(parts)
        raw = torch.cat(parts, 1)
        branch = []
        # New relation permutation is applied only to new graph branches.
        for m, x in enumerate(parts):
            adj = self._normalize_adj(self.utils, adjs[m])
            adj_perm = None
            if not force_new_self and new_relation_perm is not None:
                p = new_relation_perm.long()
                adj_perm = self._normalize_adj(self.utils, adjs[m][p][:, p])
            if force_new_self:
                adj_perm = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
            branch.append(self.branches[m](x, adj, adj_perm=adj_perm))
        # Prototype is always predicted from CORR self representation.  Groups
        # without a prototype use a zero placeholder but keep the same shape.
        proto = self.prototype_prob(branch[CORR]['si'])
        if self.mode in {'E2_SELF', 'E3_GRAPH', 'E4_FIXED', 'E5_GATE'}:
            proto_gate = torch.zeros_like(proto)
        else:
            proto_gate = proto
        q = torch.cat([parts[ANAT], parts[FUNC]], dim=1)
        gates = []
        s_used, c_used, residuals = [], [], []
        for m, b in enumerate(branch):
            delta = b['sg'] - b['si']
            if self.mode == 'E2_SELF':
                gate = torch.zeros((raw.shape[0], 1), device=raw.device, dtype=raw.dtype)
                su = b['si']
            elif self.mode == 'E3_GRAPH':
                gate = torch.ones((raw.shape[0], 1), device=raw.device, dtype=raw.dtype)
                su = b['sg']
            elif self.mode == 'E4_FIXED':
                gate = torch.full((raw.shape[0], 1), 0.5, device=raw.device, dtype=raw.dtype)
                su = b['si'] + gate * delta
            else:
                gate_input = torch.cat([b['si'], delta, proto_gate, q], dim=1)
                gate = torch.sigmoid(self.gates[m](gate_input))
                su = b['si'] + gate * delta
            if intervention == 'gate_zero':
                gate = torch.zeros_like(gate); su = b['si']
            if intervention == 'graph_off':
                gate = torch.zeros_like(gate); su = b['si']
            c = b['c']
            if self.mode == 'E11_NO_PRIVATE' or intervention == 'private_off':
                c = torch.zeros_like(c)
            r = self.write[m](torch.cat([su, c], 1))
            scale = 0.08 * torch.sigmoid(self.rho_raw[m])
            residual = scale * rms(x).detach() * torch.tanh(r)
            s_used.append(su); c_used.append(c); residuals.append(residual); gates.append(gate)
        fused = base_fused + torch.cat(residuals, dim=1)
        if intervention == 'base_only':
            fused = base_fused
        self.last = {
            'base_fused': base_fused, 'adjs': adjs, 'base_msgs': base_details['msgs'],
            'branches': branch, 's_used': s_used, 'c_used': c_used,
            'gates': torch.cat(gates, dim=1), 'proto': proto, 'proto_gate': proto_gate,
            'residuals': residuals, 'fused': fused,
            'rho': (0.08 * torch.sigmoid(self.rho_raw)).detach(),
            'new_relation_perm': new_relation_perm,
        }
        return fused, adjs, self.last

    def auxiliary_losses(self, train_idx, align_idx, align_perm=None,
                         lambda_proto=0.02, lambda_align=0.02,
                         lambda_dec=0.001, lambda_var=0.001,
                         enable_proto=False, enable_align=False,
                         uniform_align=False, labels=None):
        z = self.last
        idx = train_idx.long()
        y = labels.long()
        proto_loss = F.cross_entropy(
            torch.log(z['proto'][idx].clamp_min(1e-8)), y[idx]
        ) if enable_proto else z['fused'].new_zeros(())
        # Weak covariance penalty only separates S_self and private C.
        dec_loss = z['fused'].new_zeros(())
        var_loss = z['fused'].new_zeros(())
        for b in z['branches']:
            s = b['si'][idx] - b['si'][idx].mean(0, keepdim=True)
            c = b['c'][idx] - b['c'][idx].mean(0, keepdim=True)
            cov = (s.t() @ c / max(len(idx), 1))
            dec_loss = dec_loss + cov.pow(2).mean()
            var_loss = var_loss + F.relu(0.05 - b['si'][idx].std(0, unbiased=False)).mean()
            var_loss = var_loss + F.relu(0.05 - b['c'][idx].std(0, unbiased=False)).mean()
        dec_loss = dec_loss / len(z['branches'])
        var_loss = var_loss / (2 * len(z['branches']))
        align_raw = z['fused'].new_zeros(())
        w_rows = []
        if enable_align:
            ss = [b['sg'] for b in z['branches']]
            corr = ss[CORR]
            # Correspondence permutations apply only to auxiliary representations.
            if align_perm is None:
                ap = [torch.arange(corr.shape[0], device=corr.device) for _ in range(4)]
            else:
                ap = [align_perm[m].long() for m in range(4)]
            dc = torch.cdist(corr[align_idx], corr[align_idx], p=2)
            dc = dc / (dc.norm() + 1e-8)
            weights = []
            for m in AUX:
                gm = z['gates'][:, m]
                gc = z['gates'][:, CORR]
                w = (gm[align_idx] * gc[align_idx]).mean().detach()
                weights.append(w)
            if uniform_align:
                mean_w = torch.stack(weights).mean()
                weights = [mean_w for _ in weights]
            for k, m in enumerate(AUX):
                ds = ss[m][ap[m][align_idx]]
                dm = torch.cdist(ds, ds, p=2)
                dm = dm / (dm.norm() + 1e-8)
                align_raw = align_raw + weights[k] * (dm - dc).pow(2).mean()
                w_rows.append(weights[k].detach())
            align_raw = align_raw / 3.0
        return (lambda_proto * proto_loss, lambda_align * align_raw,
                lambda_dec * dec_loss, lambda_var * var_loss,
                {'proto_raw': proto_loss.detach(), 'align_raw': align_raw.detach(),
                 'dec_raw': dec_loss.detach(), 'var_raw': var_loss.detach(),
                 'align_weights': [float(v.detach().cpu()) for v in w_rows]})


# Explicit checkpoint initialization helpers.
def load_e1_anchor(model: nn.Module, anchor_gfr: Dict):
    model.base.fusion_weight.data.copy_(anchor_gfr['fusion_weight'])
    for m in range(4):
        gsrc = {k[len(f'GraphConstruct{m}.'):]: v for k, v in anchor_gfr.items() if k.startswith(f'GraphConstruct{m}.')}
        msrc = {k[len(f'MessagePassing{m}.'):]: v for k, v in anchor_gfr.items() if k.startswith(f'MessagePassing{m}.')}
        model.base.graphs[m].load_state_dict(gsrc, strict=True)
        model.base.gcn[m].load_state_dict(msrc, strict=True)


def load_legacy_checkpoint(model: LegacyFixedGFR, state: Dict):
    model.load_state_dict(state, strict=True)


# ---------------------------------------------------------------------------
# Experiment 52: Disease-conditioned Relation-Adversarial Propagation (DRAP)
# ---------------------------------------------------------------------------

class _GradientReversal(torch.autograd.Function):
    """Identity in the forward pass and -scale in the relation backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float):
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


def grad_reverse(x: torch.Tensor, scale: float = 0.03) -> torch.Tensor:
    return _GradientReversal.apply(x, scale)


class RelationEncoder(nn.Module):
    """A modality-local, deterministic 32-D patient relation representation."""

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelationScorer(nn.Module):
    """Shared architecture with independent parameters for each modality.

    The input is |z_i-z_j|, z_i*z_j, and cos(z_i,z_j), so swapping the
    endpoints leaves every feature unchanged.  The scalar logit is converted
    to q_ij only after the scorer, preserving a differentiable graph edge path.
    """

    def __init__(self, hidden: int = 32, pair_hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden * 2 + 1, pair_hidden),
            nn.GELU(),
            nn.Linear(pair_hidden, 1),
        )

    @staticmethod
    def features(z: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        zn = F.normalize(z, dim=-1)
        zi, zj = zn[i], zn[j]
        diff = (zi - zj).abs()
        prod = zi * zj
        cos = (zi * zj).sum(dim=-1, keepdim=True)
        return torch.cat((diff, prod, cos), dim=-1)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        return self.net(pair_features).squeeze(-1)


class ConditionalModalityDiscriminator(nn.Module):
    """Predict the source modality while receiving the pair class condition."""

    def __init__(self, pair_dim: int = 65, hidden: int = 64):
        super().__init__()
        # 65 relation features + a two-way same/different condition.
        self.net = nn.Sequential(
            nn.Linear(pair_dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, pair_features: torch.Tensor, pair_y: torch.Tensor) -> torch.Tensor:
        cond = F.one_hot(pair_y.long(), num_classes=2).to(pair_features.dtype)
        return self.net(torch.cat((pair_features, cond), dim=-1))


class DRAPGFR(nn.Module):
    """E1's four original GCNs with differentiable disease relation edge weights.

    ``base`` is the already-constructed ``LegacyFixedGFR``.  It is passed in
    rather than constructed here so that E1 shares the exact E51 construction
    order, parameter shapes, and initial values.  DRAP never adds a feature
    residual: the only new path to the classifier is the original GCN message
    computed from ``normalize_adj(A_drap + I)``.
    """

    def __init__(self, base: LegacyFixedGFR, dims=None, hidden: int = 32,
                 grl_scale: float = 0.03):
        super().__init__()
        self.base = base
        self.dims = list(DIMS if dims is None else dims)
        self.hidden = int(hidden)
        self.grl_scale = float(grl_scale)
        self.relation_encoders = nn.ModuleList(
            [RelationEncoder(d, hidden) for d in self.dims]
        )
        self.relation_scorers = nn.ModuleList(
            [RelationScorer(hidden) for _ in self.dims]
        )
        self.discriminator = ConditionalModalityDiscriminator(hidden * 2 + 1)
        self.last: Dict = {}

    @staticmethod
    def _edge_index(adj: torch.Tensor):
        n = int(adj.shape[0])
        support = (adj > 0)
        support = support & (~torch.eye(n, device=adj.device, dtype=torch.bool))
        # One score is computed per unordered pair and mirrored, enforcing
        # q_ij=q_ji exactly even if a graph has tiny numerical asymmetry.
        return torch.nonzero(torch.triu(support, diagonal=1), as_tuple=True)

    @staticmethod
    def _weight_from_q(q: torch.Tensor) -> torch.Tensor:
        confidence = 2.0 * (q - 0.5).abs()
        return torch.clamp(
            1.0 + 0.5 * (2.0 * q - 1.0) * confidence,
            min=0.5, max=1.5,
        )

    def _reweight(self, adj: torch.Tensor, z: torch.Tensor, scorer: RelationScorer):
        n = int(adj.shape[0])
        ei, ej = self._edge_index(adj)
        # Diagonal entries are kept at one.  The original GraphLearn graph
        # already contains diagonal cosine self-similarity, and the separately
        # added I in normalize_adj must remain the same self-loop convention.
        if ei.numel() == 0:
            q = adj.new_empty((0,))
            w = adj.new_empty((0,))
            out = adj
        else:
            feat = scorer.features(z, ei, ej)
            logits = scorer(feat)
            q = torch.sigmoid(logits)
            w = self._weight_from_q(q)
            wm = torch.ones_like(adj)
            wm[ei, ej] = w
            wm[ej, ei] = w
            out = adj * wm
            # Preserve a mathematically symmetric finite nonnegative matrix.
            out = 0.5 * (out + out.transpose(0, 1))
            eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
            out = out * (1.0 - eye) + torch.diag(torch.diagonal(adj))
        qmat = torch.full_like(adj, 0.5)
        if ei.numel() != 0:
            qmat[ei, ej] = q
            qmat[ej, ei] = q
        qmat.fill_diagonal_(0.5)
        return out, qmat, w, (ei, ej)

    def _pair_outputs(self, z_list, pair_i, pair_j, pair_y):
        if pair_i is None or pair_i.numel() == 0:
            empty = self.base.fusion_weight.new_empty((0, 4))
            labels = self.base.fusion_weight.new_empty((0,), dtype=torch.long)
            return [], [], self.base.fusion_weight.new_zeros(()), self.base.fusion_weight.new_zeros(()), empty, labels
        feats, logits = [], []
        for z, scorer in zip(z_list, self.relation_scorers):
            pf = scorer.features(z, pair_i, pair_j)
            feats.append(pf)
            logits.append(scorer(pf))
        pair_loss = torch.stack([
            F.binary_cross_entropy_with_logits(lg, pair_y.to(lg.dtype))
            for lg in logits
        ]).mean()
        # The relation representation alone receives GRL.  The condition is
        # concatenated after reversal, so discriminator gradients are normal.
        all_rel = torch.cat([grad_reverse(pf, self.grl_scale) for pf in feats], dim=0)
        all_y = pair_y.long().repeat(len(feats))
        all_mod = torch.arange(len(feats), device=all_rel.device).repeat_interleave(pair_i.numel())
        adv_logits = self.discriminator(all_rel, all_y)
        adv_loss = F.cross_entropy(adv_logits, all_mod)
        return feats, logits, pair_loss, adv_loss, adv_logits, all_mod

    def forward(self, parts, pair_i=None, pair_j=None, pair_y=None,
                use_reweight=True, force_self=False):
        raw = torch.cat(parts, dim=1)
        fused = raw
        adjs, drap_adjs, norm_adjs, msgs = [], [], [], []
        z_list, q_list, weight_list, edge_list = [], [], [], []
        for m, x in enumerate(parts):
            adj = self.base.graphs[m](x)
            z = self.relation_encoders[m](x)
            drap_adj, qmat, edge_w, edge_idx = self._reweight(
                adj, z, self.relation_scorers[m]
            )
            if force_self:
                used = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
            elif use_reweight:
                used = drap_adj
            else:
                used = adj
            eye = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
            normalized = self.base.utils.normalize_adj(used + eye)
            msg = self.base.gcn[m](raw, normalized)
            fused = fused + torch.sigmoid(self.base.fusion_weight[m]) * msg
            adjs.append(adj); drap_adjs.append(drap_adj); norm_adjs.append(normalized)
            msgs.append(msg); z_list.append(z); q_list.append(qmat)
            weight_list.append(edge_w); edge_list.append(edge_idx)
        if pair_y is None:
            pair_y = raw.new_zeros((0,), dtype=torch.long)
        pair_feats, pair_logits, pair_loss, adv_loss, adv_logits, adv_mod = self._pair_outputs(
            z_list, pair_i, pair_j, pair_y
        )
        self.last = {
            'raw': raw, 'fused': fused, 'adjs': adjs, 'drap_adjs': drap_adjs,
            'norm_adjs': norm_adjs, 'msgs': msgs, 'z': z_list, 'q': q_list,
            'edge_weights': weight_list, 'edge_indices': edge_list,
            'pair_features': pair_feats, 'pair_logits': pair_logits,
            'pair_loss': pair_loss, 'adv_loss': adv_loss,
            'adv_logits': adv_logits, 'adv_modal': adv_mod,
            'use_reweight': bool(use_reweight), 'force_self': bool(force_self),
        }
        return fused, adjs, self.last


def parameter_norms(module: nn.Module) -> Dict[str, float]:
    return {name: float(p.detach().norm().cpu()) for name, p in module.named_parameters()}


def grad_norms(module: nn.Module) -> Dict[str, float]:
    out = {}
    for name, p in module.named_parameters():
        out[name] = float(p.grad.detach().norm().cpu()) if p.grad is not None else 0.0
    return out
