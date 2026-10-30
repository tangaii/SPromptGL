"""Experiment 51.3 EC-SRGL.

The audited Experiment 51 E1 GFR is kept intact as the main path.  This
module adds a separate, small branch that only uses edges supported by the
four original GraphLearn matrices.  The same selected edge weights are used
for the added neighbourhood message and for the CORR anchored geometry loss.
The private branch never enters the relational losses.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from dcsra_base import LegacyFixedGFR, PLNNoPEN, DIMS, PHENO, ANAT, FUNC, CORR

AUX = (PHENO, ANAT, FUNC)


def rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-8))


class PairSelector(nn.Module):
    """A symmetric pair scorer whose output is restricted to graph support."""

    def __init__(self, in_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 16), nn.GELU(), nn.Linear(16, 1))

    @staticmethod
    def features(sm: torch.Tensor, sc: torch.Tensor, p: torch.Tensor,
                 q: torch.Tensor) -> torch.Tensor:
        nsm = F.normalize(sm, dim=1)
        nsc = F.normalize(sc, dim=1)
        nq = F.normalize(q, dim=1)
        fp = p @ p.t()
        fq = nq @ nq.t()
        pd = (p[:, None, :] - p[None, :, :]).abs().mean(-1)
        qd = (q[:, None, :] - q[None, :, :]).abs().mean(-1)
        fs = nsm @ nsm.t()
        fc = nsc @ nsc.t()
        return torch.stack((fs, fc, fq, fp, qd, pd), dim=-1)

    def forward(self, sm: torch.Tensor, sc: torch.Tensor, p: torch.Tensor,
                q: torch.Tensor, support: torch.Tensor,
                tau: float = 0.5):
        feat = self.features(sm, sc, p.detach(), q)
        logits = self.net(feat.reshape(-1, feat.shape[-1])).reshape(feat.shape[:2])
        logits = 0.5 * (logits + logits.t())
        g = torch.sigmoid(logits / tau)
        g = g * (support > 0).to(g.dtype).detach()
        g = g - torch.diag_embed(torch.diagonal(g))
        return logits, g


class ECSRGModel(nn.Module):
    """E1 plus supported-edge disease evidence selection and neighbourhood loss."""

    def __init__(self, graphlearn_cls, gcn_cls, layers_module, utils_module,
                 th: float = 0.9, mode: str = "E7", hidden: int = 16,
                 s_dim: int = 8, c_dim: int = 8, base: Optional[nn.Module] = None):
        super().__init__()
        self.mode = str(mode)
        self.utils = utils_module
        self.base = base if base is not None else LegacyFixedGFR(
            graphlearn_cls, gcn_cls, utils_module, th=th
        )
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.GELU()) for d in DIMS
        ])
        self.s_proj = nn.ModuleList([nn.Linear(hidden, s_dim) for _ in DIMS])
        self.c_proj = nn.ModuleList([nn.Linear(hidden, c_dim) for _ in DIMS])
        self.value_proj = nn.ModuleList([
            nn.Linear(s_dim, s_dim, bias=False) for _ in DIMS
        ])
        self.selectors = nn.ModuleList([PairSelector() for _ in DIMS])
        self.graph_write = nn.ModuleList([
            nn.Linear(s_dim, d, bias=False) for d in DIMS
        ])
        self.private_write = nn.ModuleList([
            nn.Linear(c_dim, d, bias=False) for d in DIMS
        ])
        self.neighbor_head = nn.Sequential(
            nn.Linear(4 * s_dim, 16), nn.GELU(), nn.Linear(16, 2)
        )
        # max=0.10 and initial value=0.025; both branches are nonzero at init.
        self.rho_graph_raw = nn.Parameter(torch.full((4,), -math.log(3.0)))
        self.rho_private_raw = nn.Parameter(torch.full((4,), -math.log(3.0)))
        self.last: Dict = {}

    @property
    def use_selector(self) -> bool:
        return self.mode in {"E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12"}

    @property
    def use_neighbor(self) -> bool:
        return self.mode in {"E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12"}

    @property
    def use_pair(self) -> bool:
        return self.mode in {"E6", "E7", "E8", "E9", "E10", "E11", "E12"}

    @property
    def use_align(self) -> bool:
        return self.mode in {"E7", "E8", "E9", "E10", "E11", "E12"}

    @staticmethod
    def _zero_diag(x: torch.Tensor) -> torch.Tensor:
        return x - torch.diag_embed(torch.diagonal(x))

    def _selected_edges(self, m: int, adj: torch.Tensor, s: torch.Tensor,
                        sc: torch.Tensor, p: torch.Tensor, q: torch.Tensor):
        if self.mode in {"E2"}:
            z = torch.zeros_like(adj)
            return torch.zeros_like(adj), z, z
        if self.mode in {"E3", "E4"}:
            z = self._zero_diag(adj)
            return torch.zeros_like(adj), z, z
        logits, g = self.selectors[m](s, sc, p, q, adj)
        b = adj * g
        if self.mode == "E8":
            den = adj.detach().sum().clamp_min(1e-8)
            alpha = b.sum() / den
            b = adj * alpha
        return logits, g, self._zero_diag(b)

    def forward(self, parts, p_cond: torch.Tensor,
                precomputed=None, relation_perm: Optional[torch.Tensor] = None,
                intervention: Optional[str] = None):
        if precomputed is None:
            base_fused, adjs, base_details = self.base(parts)
        else:
            base_fused, adjs, base_details = precomputed
        q = torch.cat((parts[ANAT], parts[FUNC]), dim=1)
        u = [self.encoders[m](parts[m]) for m in range(4)]
        s = [self.s_proj[m](u[m]) for m in range(4)]
        c = [self.c_proj[m](u[m]) for m in range(4)]
        corr = s[CORR]
        logits_all, select_all, b_all, b_used_all = [], [], [], []
        msg_all, d_all = [], []
        for m in range(4):
            logits, g, b = self._selected_edges(m, adjs[m], s[m], corr, p_cond, q)
            b_used = b
            if self.mode == "E10" and relation_perm is not None:
                pidx = relation_perm.long()
                b_used = b[pidx][:, pidx]
            value = self.value_proj[m](s[m])
            den = b_used.sum(dim=1, keepdim=True)
            msg = b_used @ value / den.clamp_min(1e-8)
            has = den > 1e-12
            msg = torch.where(has, msg, torch.zeros_like(msg))
            d = torch.where(has, msg - value, torch.zeros_like(msg))
            if intervention == "graph_off":
                d = torch.zeros_like(d)
            logits_all.append(logits)
            select_all.append(g)
            b_all.append(b)
            b_used_all.append(b_used)
            msg_all.append(msg)
            d_all.append(d)
        neighbor = torch.cat(msg_all, dim=1)
        neighbor_logits = self.neighbor_head(neighbor)
        residuals_graph, residuals_private = [], []
        rho_g = 0.10 * torch.sigmoid(self.rho_graph_raw)
        rho_c = 0.10 * torch.sigmoid(self.rho_private_raw)
        for m in range(4):
            dm = d_all[m]
            dscale = rms(dm).detach().clamp_min(1e-3)
            rg = rho_g[m] * rms(parts[m]).detach() * torch.tanh(self.graph_write[m](dm / dscale))
            cscale = rms(c[m]).detach().clamp_min(1e-3)
            rc = rho_c[m] * rms(parts[m]).detach() * torch.tanh(self.private_write[m](c[m] / cscale))
            if self.mode == "E12" or intervention == "private_off":
                rc = torch.zeros_like(rc)
            residuals_graph.append(rg)
            residuals_private.append(rc)
        fused = base_fused + torch.cat(
            [residuals_graph[m] + residuals_private[m] for m in range(4)], dim=1
        )
        if intervention == "base_only":
            fused = base_fused
        self.last = {
            "base_fused": base_fused, "adjs": adjs, "base_details": base_details,
            "u": u, "s": s, "c": c, "logits": logits_all,
            "g": select_all, "b": b_all, "b_used": b_used_all,
            "msg": msg_all, "d": d_all, "neighbor": neighbor,
            "neighbor_logits": neighbor_logits, "residuals_graph": residuals_graph,
            "residuals_private": residuals_private, "fused": fused,
            "p_cond": p_cond.detach(), "rho_graph": rho_g.detach(),
            "rho_private": rho_c.detach(),
        }
        return fused, adjs, self.last

    def pair_loss(self, pair_i: torch.Tensor, pair_j: torch.Tensor,
                  pair_y: torch.Tensor, pair_valid: torch.Tensor,
                  enabled: bool, tau: float = 0.5):
        if not enabled:
            return self.last["fused"].new_zeros(()), []
        vals, counts = [], []
        for m, logits in enumerate(self.last["logits"]):
            if logits.numel() == 0:
                continue
            i, j = pair_i[m], pair_j[m]
            valid = pair_valid[m] & (i >= 0) & (j >= 0)
            if bool(valid.any()):
                # A changing GraphLearn support cannot make an outside-support edge valid.
                valid = valid & (self.last["adjs"][m][i.clamp_min(0), j.clamp_min(0)] > 0)
            if bool(valid.any()):
                vals.append(F.binary_cross_entropy_with_logits(
                    logits[i[valid], j[valid]] / tau, pair_y[m][valid].float()
                ))
                counts.append(int(valid.sum()))
        if not vals:
            return self.last["fused"].new_zeros(()), counts
        return torch.stack(vals).mean(), counts

    def neighbor_loss(self, train_idx: torch.Tensor, labels: torch.Tensor, enabled: bool):
        if not enabled:
            return self.last["fused"].new_zeros(())
        return F.cross_entropy(self.last["neighbor_logits"][train_idx], labels[train_idx])

    def align_loss(self, align_i: torch.Tensor, align_j: torch.Tensor,
                   align_valid: torch.Tensor, align_order: torch.Tensor,
                   enabled: bool, wrong_correspondence: bool = False):
        if not enabled:
            return self.last["fused"].new_zeros(()), [], []
        vals, weights, counts = [], [], []
        for k, m in enumerate(AUX):
            i, j = align_i[k], align_j[k]
            valid = align_valid[k] & (i >= 0) & (j >= 0)
            if bool(valid.any()):
                valid = valid & (self.last["adjs"][m][i.clamp_min(0), j.clamp_min(0)] > 0)
                valid = valid & (self.last["adjs"][CORR][i.clamp_min(0), j.clamp_min(0)] > 0)
            if not bool(valid.any()):
                continue
            ii, jj = i[valid], j[valid]
            sm, sc = self.last["s"][m], self.last["s"][CORR]
            dm = (sm[ii] - sm[jj]).pow(2).sum(-1).clamp_min(1e-12).sqrt()
            dc = (sc[ii] - sc[jj]).pow(2).sum(-1).clamp_min(1e-12).sqrt()
            if wrong_correspondence:
                order = align_order[k][valid].long()
                dm = (sm[i[order]] - sm[j[order]]).pow(2).sum(-1).clamp_min(1e-12).sqrt()
            dm = dm / dm.mean().detach().clamp_min(1e-6)
            dc = dc / dc.mean().detach().clamp_min(1e-6)
            w = (self.last["b"][m][ii, jj] * self.last["b"][CORR][ii, jj]).detach()
            val = (w * (dm - dc).pow(2)).sum() / (w.sum() + 1e-8)
            vals.append(val); weights.append(float(w.mean().detach().cpu())); counts.append(int(valid.sum()))
        if not vals:
            return self.last["fused"].new_zeros(()), weights, counts
        return torch.stack(vals).mean(), weights, counts


def load_e1_anchor(model: nn.Module, anchor_gfr: Dict):
    base = model.base if hasattr(model, "base") else model
    base.fusion_weight.data.copy_(anchor_gfr["fusion_weight"])
    for m in range(4):
        gsrc = {k[len(f"GraphConstruct{m}."):]: v for k, v in anchor_gfr.items()
                if k.startswith(f"GraphConstruct{m}.")}
        msrc = {k[len(f"MessagePassing{m}."):]: v for k, v in anchor_gfr.items()
                if k.startswith(f"MessagePassing{m}.")}
        base.graphs[m].load_state_dict(gsrc, strict=True)
        base.gcn[m].load_state_dict(msrc, strict=True)

