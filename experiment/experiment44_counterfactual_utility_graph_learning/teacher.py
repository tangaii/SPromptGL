"""Cross-fitted, label-isolated edge-utility teachers for CU-GL."""
from __future__ import annotations

import copy
import hashlib
import random
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from integration import make_hp, make_pln
from cugl import CUGLBranch, CUConfig


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TeacherNet(nn.Module):
    """Lightweight directed candidate graph followed by the original CAM head."""
    def __init__(self, layers, hp=None):
        super().__init__()
        hp = hp or make_hp()
        self.branch = CUGLBranch(CUConfig(k=16, use_cf=False, symmetric=False, use_null=True))
        self.cam = make_pln(layers, hp)

    def forward(self, x: torch.Tensor, candidates: torch.Tensor):
        x_new, info = self.branch(x, candidates)
        out = self.cam(x_new)[0]
        return out, x_new, info

    def precompute(self, x: torch.Tensor, candidates: torch.Tensor):
        """Cache all unchanged two-layer quantities for exact target edits."""
        rel_cache = []
        deltas = []
        for m, rel in enumerate(self.branch.relations):
            c = candidates[:, m, :]
            w, null, logits = rel.weights(x, c, symmetric=False, force_no_null=False)
            safe = c.clamp_min(0)
            d1 = rel.down1(x)
            h1 = F.relu(d1 + (w.unsqueeze(-1) * d1[safe]).sum(1))
            d2 = rel.down2(h1)
            h2_self = d2
            h2_full = F.relu(h2_self + (w.unsqueeze(-1) * d2[safe]).sum(1))
            base = F.relu(h2_self)
            delta = rel.alpha * torch.tanh(rel.up(h2_full - base))
            rel_cache.append((w, safe, d1, h1, d2, logits, null))
            deltas.append(delta)
        x_new = x + torch.stack(deltas, dim=1).sum(1)
        return rel_cache, torch.stack(deltas, dim=1), x_new

    def target_without(self, x: torch.Tensor, candidates: torch.Tensor,
                       cache, deltas: torch.Tensor, x_new: torch.Tensor,
                       target: int, relation: int, slot: int):
        """Exact local recomputation after removing only sender->receiver."""
        rel = self.branch.relations[relation]
        c = candidates[:, relation, :]
        w, safe, d1, h1, d2, logits, null = cache[relation]
        wt = w[target].clone(); wt[slot] = 0.0
        h1_t = F.relu(d1[target] + (wt.unsqueeze(-1) * d1[safe[target]]).sum(0))
        d2_t = rel.down2(h1_t)
        full_t = F.relu(d2_t + (wt.unsqueeze(-1) * d2[safe[target]]).sum(0))
        base_t = F.relu(d2_t)
        d_t = rel.alpha * torch.tanh(rel.up(full_t - base_t))
        target_x = x_new[target] - deltas[target, relation] + d_t
        return target_x


def train_teacher(layers, x: torch.Tensor, y: torch.Tensor,
                  train_idx: np.ndarray, candidates: torch.Tensor,
                  seed: int, epochs: int = 35):
    seed_all(seed)
    hp = make_hp()
    net = TeacherNet(layers, hp).to(x.device)
    # Teacher training uses only inner-train labels.  Held-out labels are not
    # touched until the frozen model is used to measure edge utility.
    opt = torch.optim.Adam(net.parameters(), lr=hp.lr, weight_decay=hp.reg)
    idx = torch.as_tensor(train_idx, dtype=torch.long, device=x.device)
    class_count = torch.bincount(y[idx], minlength=2).float().clamp_min(1)
    weight = (len(idx) / (2 * class_count)).to(x.device)
    trace = []
    net.train()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        out, _, _ = net(x, candidates)
        loss = F.nll_loss(out[idx], y[idx], weight=weight)
        if not torch.isfinite(loss):
            raise FloatingPointError('teacher loss is not finite')
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        if epoch in (0, epochs - 1) or epoch % 10 == 9:
            trace.append({'epoch': epoch, 'loss': float(loss.detach())})
    net.eval()
    return net, trace


@torch.no_grad()
def edge_utilities(net: TeacherNet, x: torch.Tensor, y: torch.Tensor,
                   candidates: torch.Tensor, heldout_idx: np.ndarray):
    """Compute OOF U(i<-j,m) for each held-out receiver.

    The only use of the held-out label is selecting log p(y_i).  It does not
    enter teacher optimisation or candidate construction.
    """
    net.eval()
    cache, deltas, x_new = net.precompute(x, candidates)
    baseline_logp = net.cam(x_new)[0]
    n, _, k = candidates.shape
    utilities = np.full((n, 4, k), np.nan, dtype=np.float32)
    valid = np.zeros((n, 4, k), dtype=np.uint8)
    for i in np.asarray(heldout_idx, dtype=np.int64):
        yi = int(y[i].item())
        base = float(baseline_logp[i, yi].item())
        variants=[]; locations=[]
        for m in range(4):
            for slot in range(k):
                if int(candidates[i, m, slot]) < 0: continue
                variants.append(net.target_without(x, candidates, cache, deltas, x_new, int(i), m, slot))
                locations.append((m,slot)); valid[i,m,slot]=1
        # One CAM call per receiver (rather than one GPU launch per edge).
        if variants:
            alt=net.cam(torch.stack(variants,0))[0][:,yi]
            for value,(m,slot) in zip(alt.detach().cpu().numpy(),locations):
                utilities[i,m,slot]=np.float32(base-float(value))
    return utilities, valid, {
        'heldout_count': int(len(heldout_idx)),
        'valid_edges': int(valid.sum()),
        'finite_count': int(np.isfinite(utilities).sum()),
        'mean_abs': float(np.nanmean(np.abs(utilities))) if np.isfinite(utilities).any() else 0.0,
        'positive_fraction': float(np.nanmean(utilities > 0)) if np.isfinite(utilities).any() else 0.0,
        'negative_fraction': float(np.nanmean(utilities < 0)) if np.isfinite(utilities).any() else 0.0,
        'baseline_sha256': hashlib.sha256(x_new.detach().cpu().numpy().tobytes()).hexdigest(),
    }

