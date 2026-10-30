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
