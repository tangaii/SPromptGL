"""DUGSA model and fold-local disease-utility helpers.

The implementation is isolated from the project root.  It keeps the audited
GraphLearn/normalization operators, but gives each modality its own 32-D
encoder and GCN.  Shared (S), complementary (C), and nuisance (N) heads are
computed from the same graph/self parameters; only S receives utility-gated
cross-patient increments.  C remains available to the classifier and N is
regularized but never written to the CAM input.
"""
from __future__ import annotations
from itertools import combinations
from typing import Dict, Optional
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

PHENO, ANAT, FUNC, CORR = 0, 1, 2, 3
DIMS = [48, 6, 10, 256]
PAIR_INDEX = [(i, j) for i in range(4) for j in range(i + 1, 4)]


def _rms(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return torch.sqrt(x.pow(2).mean(dim=dim, keepdim=True).clamp_min(1e-8))


class PLNNoPEN(nn.Module):
    """PEN-free CAM copied into the experiment's private source tree."""
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


class ModalGCN(nn.Module):
    def __init__(self, layers_module, hidden=32):
        super().__init__()
        self.gcn1 = layers_module.GraphConv(hidden, hidden)
        self.gcn2 = layers_module.GraphConv(hidden, hidden)

    def forward(self, x, adj):
        return self.gcn2(F.relu(self.gcn1(x, adj)), adj)


class ModalityEncoder(nn.Module):
    def __init__(self, d, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.LayerNorm(hidden))

    def forward(self, x):
        return self.net(x)


class DisentangleHead(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.s = nn.Linear(hidden, 12)
        self.c = nn.Linear(hidden, 12)
        self.n = nn.Linear(hidden, 8)

    def forward(self, x):
        z = self.proj(x)
        return self.s(z), self.c(z), self.n(z)


class DUGSAGFR(nn.Module):
    """DUGSA GFR replacement with fixed output order 48/6/10/256."""
    def __init__(self, graphlearn_cls, layers_module, utils_module, th=0.9,
                 mode='E5', hidden=32, rho_max=0.15):
        super().__init__()
        self.mode = mode
        self.th = th
        self.utils = utils_module
        self.graphs = nn.ModuleList([graphlearn_cls(d, th=th, mode='weighted-cosine') for d in DIMS])
        self.encoders = nn.ModuleList([ModalityEncoder(d, hidden) for d in DIMS])
        self.gcn = nn.ModuleList([ModalGCN(layers_module, hidden) for _ in DIMS])
        self.decomp = nn.ModuleList([DisentangleHead(hidden) for _ in DIMS])
        self.mod_proj = nn.ModuleList([nn.Linear(24, d) for d in DIMS])
        self.simple_proj = nn.ModuleList([nn.Linear(hidden, d) for d in DIMS])
        # nonzero bounded initialization: rho=0.15*sigmoid(-0.693)=0.05
        self.rho_raw = nn.Parameter(torch.full((4,), -math.log(2.0)))
        self.rho_max = float(rho_max)
        self.last = {}

    def _prop_adj(self, adj, perm=None, self_only=False):
        n = adj.shape[0]
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
        if self_only:
            return eye
        if perm is not None:
            adj = adj[perm][:, perm]
        return self.utils.normalize_adj(adj + eye)

    def forward(self, parts, utility_g, relation_perm=None, force_self=False):
        raw = torch.cat(parts, dim=1)
        adjs, graph_states, self_states = [], [], []
        s_graph, c_graph, n_graph, s_self, c_self, n_self = [], [], [], [], [], []
        fused_parts = []
        for m, x in enumerate(parts):
            adj = self.graphs[m](x)
            adjs.append(adj)
            u = self.encoders[m](x)
            ag = self._prop_adj(adj, relation_perm, self_only=force_self)
            ai = self._prop_adj(adj, None, self_only=True)
            hg = self.gcn[m](u, ag)
            hi = self.gcn[m](u, ai)
            sg, cg, ng = self.decomp[m](hg)
            si, ci, ni = self.decomp[m](hi)
            graph_states.append(hg); self_states.append(hi)
            s_graph.append(sg); c_graph.append(cg); n_graph.append(ng)
            s_self.append(si); c_self.append(ci); n_self.append(ni)
        # Utility matrix is fold-local and fixed during optimization.
        g = utility_g.to(raw.device, raw.dtype)
        if self.mode == 'E7_UNIFORM':
            mean_g = g[torch.triu(torch.ones_like(g), diagonal=1) > 0].mean()
            g_use = torch.full_like(g, mean_g)
            g_use.fill_diagonal_(0.0)
        elif self.mode == 'E8_WRONG_UTILITY':
            perm = torch.tensor([1, 2, 3, 4, 5, 0], device=raw.device)
            g_use = torch.zeros_like(g)
            for k, (i, j) in enumerate(PAIR_INDEX):
                src = PAIR_INDEX[int(perm[k])]
                g_use[i, j] = g_use[j, i] = g[src[0], src[1]]
        elif self.mode in {'E2_MODAL', 'E3_SHARED_PRIVATE'}:
            g_use = torch.zeros_like(g)
        elif self.mode in {'E4_DECOMP'}:
            g_use = torch.full_like(g, 0.5); g_use.fill_diagonal_(0.0)
        else:
            g_use = g
        q = torch.stack([g_use[m].sum() / 3.0 for m in range(4)])
        if self.mode == 'E10_SELF':
            # Keep utility metadata but force graph=self for the actual message.
            q_for_output = q
        else:
            q_for_output = q
        for m, x in enumerate(parts):
            if self.mode == 'E2_MODAL':
                v = self.simple_proj[m](graph_states[m])
            elif self.mode == 'E3_SHARED_PRIVATE':
                v = self.mod_proj[m](torch.cat([s_graph[m], c_graph[m]], 1))
            else:
                su = s_self[m] + q_for_output[m] * (s_graph[m] - s_self[m])
                cu = c_graph[m]
                if self.mode == 'E11_NO_COMPLEMENT':
                    cu = torch.zeros_like(cu)
                v = self.mod_proj[m](torch.cat([su, cu], 1))
            scale = self.rho_max * torch.sigmoid(self.rho_raw[m])
            residual = scale * _rms(x).detach() * torch.tanh(F.layer_norm(v, (v.shape[1],)))
            fused_parts.append(x + residual)
        fused = torch.cat(fused_parts, dim=1)
        self.last = {
            'adjs': adjs, 'graph_states': graph_states, 'self_states': self_states,
            's_graph': s_graph, 'c_graph': c_graph, 'n_graph': n_graph,
            's_self': s_self, 'c_self': c_self, 'n_self': n_self,
            'g_used': g_use.detach(), 'q': q.detach(), 'rho': (self.rho_max*torch.sigmoid(self.rho_raw)).detach(),
            'fused': fused, 'parts': fused_parts, 'relation_perm': relation_perm,
        }
        return fused, adjs, self.last

    def auxiliary_losses(self, train_idx, align_idx, utility_g, lambda_align=0.02,
                         lambda_n=0.005, lambda_v=0.005, enable_align=True):
        z = self.last
        idx = train_idx
        # N is never sent to the CAM; its covariance with labels is weakly penalized.
        n_loss = z['fused'].new_zeros(())
        y = self._labels_for_loss
        yy = y[idx].float(); yy = yy - yy.mean()
        for n in z['n_graph']:
            nnv = n[idx] - n[idx].mean(0, keepdim=True)
            cov = (nnv * yy[:, None]).mean(0)
            n_loss = n_loss + cov.pow(2).mean()
        n_loss = n_loss / len(z['n_graph'])
        # Variance floor prevents constant S/C/N heads without forcing equality.
        v_loss = z['fused'].new_zeros(())
        for arr in z['s_graph'] + z['c_graph'] + z['n_graph']:
            sd = arr[idx].std(0, unbiased=False)
            v_loss = v_loss + F.relu(0.05 - sd).mean()
        v_loss = v_loss / (len(z['s_graph']) + len(z['c_graph']) + len(z['n_graph']))
        a_loss = z['fused'].new_zeros(())
        if enable_align and lambda_align > 0:
            ss = z['s_graph']
            for m, n in PAIR_INDEX:
                w = utility_g[m, n].to(a_loss.device, a_loss.dtype)
                if float(w.detach()) == 0.0:
                    continue
                dm = torch.cdist(ss[m][align_idx], ss[m][align_idx], p=2)
                dn = torch.cdist(ss[n][align_idx], ss[n][align_idx], p=2)
                dm = dm / dm.mean().clamp_min(1e-6)
                dn = dn / dn.mean().clamp_min(1e-6)
                a_loss = a_loss + w * (dm - dn).pow(2).mean()
        return lambda_align*a_loss, lambda_n*n_loss, lambda_v*v_loss, {'align_raw':a_loss.detach(), 'n_raw':n_loss.detach(), 'v_raw':v_loss.detach()}


class LegacyFixedGFR(nn.Module):
    """E1: original full-320 four graph path with PHENO overwrite repaired."""
    def __init__(self, graphlearn_cls, gcn_cls, utils_module, th=0.9):
        super().__init__()
        self.utils = utils_module
        self.graphs = nn.ModuleList([graphlearn_cls(d, th=th, mode='weighted-cosine') for d in DIMS])
        self.gcn = nn.ModuleList([gcn_cls(sum(DIMS), sum(DIMS)) for _ in DIMS])
        self.fusion_weight = nn.Parameter(torch.ones(4, 1))
        self.last = {}

    def forward(self, parts, utility_g=None, relation_perm=None, force_self=False):
        raw = torch.cat(parts, 1); fused = raw; adjs=[]; msgs=[]
        for m, x in enumerate(parts):
            adj=self.graphs[m](x); adjs.append(adj)
            n=adj.shape[0]; eye=torch.eye(n,device=adj.device,dtype=adj.dtype)
            if force_self: an=eye
            elif relation_perm is not None: an=self.utils.normalize_adj(adj[relation_perm][:,relation_perm]+eye)
            else: an=self.utils.normalize_adj(adj+eye)
            msg=self.gcn[m](raw,an); msgs.append(msg)
            fused = fused + torch.sigmoid(self.fusion_weight[m])*msg
        self.last={'adjs':adjs,'msgs':msgs,'fused':fused}
        return fused,adjs,self.last
