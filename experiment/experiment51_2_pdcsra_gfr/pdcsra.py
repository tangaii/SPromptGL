"""Experiment 51.2 PDC-SRA-GFR.

The module keeps the audited Experiment 51 E1 four-branch GFR as the main
path and adds a patient-pair selector. The selector uses symmetric pair
statistics, is supported only on the original GraphLearn edges, and the same
selected weights drive the added graph message and the CORR-anchored geometry
loss. C is a private/possibly complementary channel and is never aligned.
"""
from __future__ import annotations
import math
from typing import Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F
from dcsra_base import PLNNoPEN, LegacyFixedGFR

PHENO, ANAT, FUNC, CORR = 0, 1, 2, 3
DIMS = [48, 6, 10, 256]
AUX = [PHENO, ANAT, FUNC]

def rms(x):
    return torch.sqrt(x.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-8))

class ModalGCN(nn.Module):
    def __init__(self, layers_module, hidden=16):
        super().__init__()
        self.gcn1 = layers_module.GraphConv(hidden, hidden)
        self.gcn2 = layers_module.GraphConv(hidden, hidden)
    def forward(self, x, adj):
        return self.gcn2(F.relu(self.gcn1(x, adj)), adj)

class PairSelector(nn.Module):
    """Symmetric patient-pair score; output is an unbounded logit matrix."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 8), nn.GELU(), nn.Linear(8, 1))
    def pair_features(self, sm, cm, sc, cc, p, q):
        ns, nc, nsc, ncc = [F.normalize(z, dim=1) for z in (sm, cm, sc, cc)]
        fs = ns @ ns.t(); fc = nsc @ nsc.t(); fcp = nc @ ncc.t()
        fp = p @ p.t()
        pd = (p[:, None, :] - p[None, :, :]).abs().mean(-1)
        qd = (q[:, None, :] - q[None, :, :]).abs().mean(-1)
        return torch.stack((fs, fc, fcp, fp, pd, qd), dim=-1)
    def forward(self, sm, cm, sc, cc, p, q, support):
        feats = self.pair_features(sm, cm, sc, cc, p.detach(), q)
        logits = self.net(feats.reshape(-1, 6)).reshape(feats.shape[0], feats.shape[1])
        logits = 0.5 * (logits + logits.t())
        g = torch.sigmoid(logits / 0.5)
        # The selector cannot create edges outside the original GraphLearn support.
        g = g * support.detach().clamp_min(0)
        g.fill_diagonal_(0.0)
        return logits, g

class PDCModel(nn.Module):
    """E1 base plus pairwise disease-conditioned selective graph alignment."""
    def __init__(self, graphlearn_cls, gcn_cls, layers_module, utils_module,
                 th=0.9, mode='E6', hidden=16, base=None):
        super().__init__()
        self.mode = str(mode)
        self.utils = utils_module
        self.base = base if base is not None else LegacyFixedGFR(graphlearn_cls, gcn_cls, utils_module, th=th)
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(d, hidden), nn.GELU()) for d in DIMS])
        self.gcn = nn.ModuleList([ModalGCN(layers_module, hidden) for _ in DIMS])
        self.s_proj = nn.ModuleList([nn.Linear(hidden, 8) for _ in DIMS])
        self.c_proj = nn.ModuleList([nn.Linear(hidden, 8) for _ in DIMS])
        self.selectors = nn.ModuleList([PairSelector() for _ in DIMS])
        self.write = nn.ModuleList([nn.Linear(16, d) for d in DIMS])
        # rho=0.12*sigmoid(-1.0986)=0.03 at initialization, nonzero and bounded.
        self.rho_raw = nn.Parameter(torch.full((4,), -math.log(3.0)))
        self.last: Dict = {}

    @staticmethod
    def norm_adj(utils, a):
        n = a.shape[0]; eye = torch.eye(n, device=a.device, dtype=a.dtype)
        return utils.normalize_adj(a + eye)

    def _mode_flags(self):
        return self.mode in {'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10'}

    def forward(self, parts, p_cond, precomputed=None, relation_perm=None,
                align_perm=None, intervention=None):
        if precomputed is None:
            base_fused, adjs, base_details = self.base(parts)
        else:
            base_fused, adjs, base_details = precomputed
        q = torch.cat([parts[ANAT], parts[FUNC]], dim=1)
        # CORR self view is the anchor representation for pair statistics.
        u = [self.encoders[m](parts[m]) for m in range(4)]
        eye = torch.eye(parts[0].shape[0], device=parts[0].device, dtype=parts[0].dtype)
        self_states, graph_states, s_self, s_graph, c_self = [], [], [], [], []
        logits_all, g_all, used_adj = [], [], []
        for m in range(4):
            adj = adjs[m]
            ai = eye
            hi = self.gcn[m](u[m], ai)
            si = self.s_proj[m](hi)
            ci = self.c_proj[m](u[m])
            self_states.append(hi); s_self.append(si); c_self.append(ci)
        sc, cc = s_self[CORR], c_self[CORR]
        for m in range(4):
            adj = adjs[m]
            if self.mode == 'E2':
                logits = torch.zeros_like(adj); g = torch.zeros_like(adj); an = eye
            elif self.mode == 'E3':
                logits = torch.zeros_like(adj); g = (adj.detach().clamp_min(0)); g.fill_diagonal_(0.0); an = self.norm_adj(self.utils, adj)
            elif self.mode == 'E7':
                logits = torch.zeros_like(adj); g = (adj.detach() > 0).to(adj.dtype); g.fill_diagonal_(0.0); an = self.norm_adj(self.utils, g)
            else:
                logits, g = self.selectors[m](s_self[m], c_self[m], sc, cc, p_cond, q, adj)
                if intervention == 'selector_off':
                    g = (adj.detach() > 0).to(adj.dtype); g.fill_diagonal_(0.0)
                an = self.norm_adj(self.utils, g)
            if relation_perm is not None:
                an = self.norm_adj(self.utils, g[relation_perm][:, relation_perm])
            hg = self.gcn[m](u[m], an)
            sg = self.s_proj[m](hg)
            logits_all.append(logits); g_all.append(g); used_adj.append(an); graph_states.append(hg); s_graph.append(sg)
        deltas = [s_graph[m] - s_self[m] for m in range(4)]
        residuals=[]; c_used=[]
        for m in range(4):
            c = c_self[m]
            if self.mode == 'E10' or intervention == 'private_off':
                c = torch.zeros_like(c)
            delta = deltas[m]
            if self.mode == 'E2': delta = torch.zeros_like(delta)
            if intervention == 'graph_off': delta = torch.zeros_like(delta)
            v = self.write[m](torch.cat([delta, c], dim=1))
            scale = 0.12 * torch.sigmoid(self.rho_raw[m])
            residuals.append(scale * rms(parts[m]).detach() * torch.tanh(v / rms(v).detach().clamp_min(1e-3)))
            c_used.append(c)
        fused = base_fused + torch.cat(residuals, dim=1)
        if intervention == 'base_only': fused = base_fused
        self.last = {'base_fused':base_fused,'adjs':adjs,'base_details':base_details,
                     'u':u,'self_states':self_states,'graph_states':graph_states,
                     's_self':s_self,'s_graph':s_graph,'c':c_used,
                     'deltas':deltas,'logits':logits_all,'g':g_all,'used_adj':used_adj,
                     'residuals':residuals,'fused':fused,'p_cond':p_cond.detach(),
                     'rho':(0.12*torch.sigmoid(self.rho_raw)).detach()}
        return fused, adjs, self.last

    def pair_loss(self, pair_i, pair_j, pair_y, enabled=True):
        if not enabled or pair_i.numel()==0: return self.last['fused'].new_zeros(())
        losses=[]
        for logits in self.last['logits']:
            losses.append(F.binary_cross_entropy_with_logits(logits[pair_i, pair_j], pair_y.float()))
        return torch.stack(losses).mean()

    def align_loss(self, align_idx, align_pairs, align_perm=None, enabled=True):
        if not enabled or align_pairs.numel()==0: return self.last['fused'].new_zeros(()), []
        idx = align_idx.long(); pi = align_pairs[:,0].long(); pj = align_pairs[:,1].long()
        out=[]; raw=[]; weights=[]
        # anchor pairs are compared using the same selected g used for propagation.
        for m in AUX:
            if align_perm is None:
                ii, jj = pi, pj
                sm = self.last['s_graph'][m]
                gm = self.last['g'][m]
            else:
                pm = align_perm[m].long()
                ii, jj = pm[pi], pm[pj]
                sm = self.last['s_graph'][m]
                gm = self.last['g'][m]
            ic, jc = pi, pj
            sc = self.last['s_graph'][CORR]; gc = self.last['g'][CORR]
            support = (self.last['adjs'][m][ii,jj] > 0) & (self.last['adjs'][CORR][ic,jc] > 0)
            if not bool(support.any()): continue
            ii,jj,ic,jc = ii[support],jj[support],ic[support],jc[support]
            dm = (sm[ii]-sm[jj]).pow(2).sum(-1).clamp_min(1e-12).sqrt()
            dc = (sc[ic]-sc[jc]).pow(2).sum(-1).clamp_min(1e-12).sqrt()
            dm = dm / dm.mean().detach().clamp_min(1e-6); dc = dc / dc.mean().detach().clamp_min(1e-6)
            if align_perm is None:
                w = (gm[ii,jj] * gc[ic,jc]).detach()
            else:
                w = (gm[ii,jj] * gc[ic,jc]).detach()
            val = (w*(dm-dc).pow(2)).sum()/(w.sum()+1e-8)
            raw.append(val); weights.append(float(w.mean().detach().cpu())); out.append(val)
        if not out: return self.last['fused'].new_zeros(()), []
        return torch.stack(out).mean(), weights


def load_e1_anchor(model, anchor_gfr):
    model.base.fusion_weight.data.copy_(anchor_gfr['fusion_weight'])
    for m in range(4):
        gsrc={k[len(f'GraphConstruct{m}.'):]:v for k,v in anchor_gfr.items() if k.startswith(f'GraphConstruct{m}.')}
        msrc={k[len(f'MessagePassing{m}.'):]:v for k,v in anchor_gfr.items() if k.startswith(f'MessagePassing{m}.')}
        model.base.graphs[m].load_state_dict(gsrc, strict=True)
        model.base.gcn[m].load_state_dict(msrc, strict=True)
