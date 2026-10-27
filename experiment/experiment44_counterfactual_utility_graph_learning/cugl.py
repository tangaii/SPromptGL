"""CU-GL: a small, directed, rejectable patient-graph branch.

The module is deliberately self contained.  It consumes a frozen candidate
pool (receiver rows, sender columns) and returns the original 320-dimensional
patient representation plus a controlled neighbour residual.  No label or
patient identifier enters the scorer at inference time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CUConfig:
    k: int = 16
    use_cf: bool = False
    symmetric: bool = False
    use_null: bool = True
    utility_weight: float = 0.10
    rank_weight: float = 0.02


def _inv_sigmoid(p: float) -> float:
    return math.log(p / (1.0 - p))


class DirectedRelation(nn.Module):
    """One receiver-conditioned relation with an explicit NULL action."""

    def __init__(self, dim: int, full_dim: int = 320, hidden: int = 32):
        super().__init__()
        self.dim = dim
        self.rx = nn.Linear(dim, hidden, bias=False)
        self.sx = nn.Linear(dim, hidden, bias=False)
        self.score = nn.Sequential(
            nn.Linear(hidden * 4 + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, 1, bias=False)
        )
        # Two lightweight message layers.  They use the complete patient
        # feature, while the candidate set is still modality specific.
        self.down1 = nn.Linear(full_dim, hidden, bias=False)
        self.down2 = nn.Linear(hidden, hidden, bias=False)
        self.up = nn.Linear(hidden, full_dim, bias=False)
        self.null_logit = nn.Parameter(torch.zeros(()))
        self.alpha_logit = nn.Parameter(torch.tensor(_inv_sigmoid(0.4)))

    @property
    def alpha(self) -> torch.Tensor:
        return 0.25 * torch.sigmoid(self.alpha_logit)

    def raw_pair_score(self, receiver: torch.Tensor, sender: torch.Tensor,
                       similarity: torch.Tensor) -> torch.Tensor:
        r = self.rx(receiver)
        s = self.sx(sender)
        z = torch.cat((r, s, r - s, r * s, similarity.unsqueeze(-1)), dim=-1)
        return self.score(z).squeeze(-1)

    def weights(self, x: torch.Tensor, cand: torch.Tensor,
                symmetric: bool = False, force_no_null: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return candidate weights, NULL weight and raw logits.

        ``cand`` is [N,K] with -1 for an absent sender.  The row is receiver
        i and every non-negative entry is a sender j (j -> i).
        """
        n, k = cand.shape
        safe = cand.clamp_min(0)
        receiver = x[:, None, :].expand(n, k, -1)[..., :self.dim]
        sender = x[safe][..., :self.dim]
        # Cosine is only an input feature; the learned pair score remains
        # directional unless the declared symmetric ablation is selected.
        sim = F.cosine_similarity(receiver, sender, dim=-1, eps=1e-6)
        logits = self.raw_pair_score(receiver.reshape(-1, self.dim),
                                     sender.reshape(-1, self.dim),
                                     sim.reshape(-1)).reshape(n, k)
        valid = cand.ge(0)
        logits = logits.masked_fill(~valid, -1e9)
        if symmetric:
            rev_receiver = sender
            rev_sender = receiver
            rev = self.raw_pair_score(rev_receiver.reshape(-1, self.dim),
                                       rev_sender.reshape(-1, self.dim),
                                       sim.reshape(-1)).reshape(n, k)
            logits = 0.5 * (logits + rev.masked_fill(~valid, -1e9))
            logits = logits.masked_fill(~valid, -1e9)
        if force_no_null:
            probs = torch.softmax(logits, dim=1)
            null = torch.zeros(n, device=x.device, dtype=x.dtype)
        else:
            all_logits = torch.cat((logits, self.null_logit.expand(n, 1)), dim=1)
            probs_all = torch.softmax(all_logits, dim=1)
            probs, null = probs_all[:, :k], probs_all[:, k]
        return probs * valid.to(x.dtype), null, logits

    def message(self, x: torch.Tensor, cand: torch.Tensor,
                symmetric: bool, use_null: bool,
                remove: Optional[Tuple[int, int]] = None):
        weights, null, logits = self.weights(
            x, cand, symmetric=symmetric, force_no_null=not use_null)
        if remove is not None:
            ri, slot = remove
            if 0 <= ri < weights.shape[0] and 0 <= slot < weights.shape[1]:
                weights = weights.clone()
                weights[ri, slot] = 0.0
        safe = cand.clamp_min(0)
        # Layer 1: self representation plus incoming messages.
        h = self.down1(x)
        sender_h = self.down1(x)[safe]
        h1_msg = (weights.unsqueeze(-1) * sender_h).sum(1)
        h1 = F.relu(h + h1_msg)
        # Layer 2 has the same directed receiver <- sender convention.
        h2_self = self.down2(h1)
        sender_h2 = self.down2(h1)[safe]
        h2_msg = (weights.unsqueeze(-1) * sender_h2).sum(1)
        full2 = F.relu(h2_self + h2_msg)
        base2 = F.relu(h2_self)
        delta = self.up(full2 - base2)
        # The sigmoid coefficient is bounded by 0.25 and starts at 0.10.
        delta = self.alpha * torch.tanh(delta)
        if remove is not None:
            # The caller needs the exact edge-removed target response.  The
            # non-target rows are unchanged and are still returned for audits.
            pass
        return delta, weights, null, logits


class CUGLBranch(nn.Module):
    """Four relation branches and the common utility objective."""

    dims = (48, 6, 10, 256)

    def __init__(self, cfg: CUConfig):
        super().__init__()
        self.cfg = cfg
        self.relations = nn.ModuleList([DirectedRelation(d) for d in self.dims])

    def forward(self, x: torch.Tensor, candidates: torch.Tensor,
                mode: Optional[CUConfig] = None):
        cfg = mode or self.cfg
        deltas, masses, nulls, logits = [], [], [], []
        for m, rel in enumerate(self.relations):
            d, w, q, l = rel.message(x, candidates[:, m, :],
                                     symmetric=cfg.symmetric,
                                     use_null=cfg.use_null)
            deltas.append(d); masses.append(w.sum(1)); nulls.append(q); logits.append(l)
        residual = torch.stack(deltas, dim=1).sum(1)
        return x + residual, {
            'delta': torch.stack(deltas, dim=1),
            'candidate_mass': torch.stack(masses, dim=1),
            'null_mass': torch.stack(nulls, dim=1),
            'logits': logits,
        }

    def utility_loss(self, info: Dict, targets: torch.Tensor,
                     mask: torch.Tensor, cfg: Optional[CUConfig] = None):
        """Robust magnitude regression and within-receiver ranking.

        Targets are model-level edge utilities, not clinical causal labels.
        Unknown candidates are masked.  Near-zero teacher responses can be
        downweighted by the supplied mask.
        """
        cfg = cfg or self.cfg
        losses = []
        rank_losses = []
        for m, logits in enumerate(info['logits']):
            t = targets[:, m, :].to(logits.dtype)
            valid = mask[:, m, :].to(torch.bool) & torch.isfinite(t)
            if valid.any():
                # Utility responses are small; preserving their sign and
                # magnitude is more informative than a binary edge target.
                losses.append(F.smooth_l1_loss(logits[valid], t[valid], reduction='mean'))
                # Vectorised within-receiver ranking: largest utility should
                # outrank smallest utility.  The previous Python row loop
                # dominated runtime for 871 receivers and is unnecessary.
                enough = valid.sum(1) >= 2
                if enough.any():
                    hi_t = t.masked_fill(~valid, -float('inf')).argmax(1)
                    lo_t = t.masked_fill(~valid, float('inf')).argmin(1)
                    rows = torch.where(enough)[0]
                    hi = logits[rows, hi_t[rows]]; lo = logits[rows, lo_t[rows]]
                    margin = (t[rows, hi_t[rows]] - t[rows, lo_t[rows]]).detach().abs().clamp_min(1e-3)
                    rank_losses.append(F.softplus(-(hi - lo) * margin).mean())
        reg = torch.stack(losses).mean() if losses else sum(p.sum() * 0 for p in self.parameters())
        rank = torch.stack(rank_losses).mean() if rank_losses else reg * 0
        return cfg.utility_weight * reg + cfg.rank_weight * rank, {
            'utility_reg': float(reg.detach()), 'utility_rank': float(rank.detach()),
            'utility_valid': int(mask.sum().item())
        }

