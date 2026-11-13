"""Common, deterministic CAM144 memory-fusion utilities for Experiment 79.2.

The module deliberately keeps the original DARG/CAM/OutputLayer outside this
file.  Memory branches receive detached CAM144 and use a frozen eval-mode
OutputLayer only for their own loss; no branch gradient can reach the base.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


CAM_DIM = 144
MEMORY_DIM = 64
TEMPERATURE = 0.2
DELTA_CAP = 0.075


def output_layer_logp(output_layer: nn.Module, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a frozen eval OutputLayer while preserving gradients to ``h``."""
    from torch.func import functional_call
    was_training = bool(output_layer.training)
    output_layer.eval()
    params = {n: p.detach() for n, p in output_layer.named_parameters()}
    buffers = {n: b.detach() for n, b in output_layer.named_buffers()}
    result = functional_call(output_layer, (params, buffers), (h, None))
    if was_training:
        output_layer.train()
    return result


def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=eps)


def masked_cosine(q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pairwise masked cosine for ``q[N,D]`` and ``k[M,D]`` with masks ``M,M,D``."""
    qv = q[:, None, :] - 0.0
    kv = k[None, :, :] - 0.0
    # The caller supplies a broadcastable mask.  No label is used here.
    num = (qv * kv * mask[None, :, :]).sum(-1)
    qn = (qv.square() * mask[None, :, :]).sum(-1).sqrt().clamp_min(1e-8)
    kn = (kv.square() * mask[None, :, :]).sum(-1).sqrt().clamp_min(1e-8)
    return num / (qn * kn)


def stable_softmax(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if scores.shape[-1] == 0:
        return scores
    masked = scores.masked_fill(~valid, -1e9)
    alpha = torch.softmax(masked, dim=-1)
    return alpha * valid.to(scores.dtype)


def _zero_linear(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class MemoryFusionBase(nn.Module):
    """Common memory-conditioned CAM144 enhancer.

    The only path producing ``delta_h`` starts from the two memory reads.  The
    current CAM and base probabilities enter the query and gate only; they do
    not have a direct path to the 144-dimensional correction.
    """

    def __init__(self, cam_dim: int = CAM_DIM, memory_dim: int = MEMORY_DIM,
                 temperature: float = TEMPERATURE, delta_cap: float = DELTA_CAP):
        super().__init__()
        self.cam_dim = int(cam_dim)
        self.memory_dim = int(memory_dim)
        self.temperature = float(temperature)
        self.delta_cap = float(delta_cap)
        self.q_proj = nn.Linear(self.cam_dim, self.memory_dim, bias=False)
        self.key_norm = nn.LayerNorm(self.cam_dim)
        self.query_norm = nn.LayerNorm(self.cam_dim)
        self.v_proj = nn.Linear(self.cam_dim, self.memory_dim, bias=False)
        self.read_to_delta = nn.Sequential(
            nn.Linear(2 * self.memory_dim, self.memory_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.memory_dim, self.cam_dim, bias=False),
        )
        # base log-probabilities (2), two similarity/entropy pairs (4), and
        # two availability indicators (2).
        self.gate = nn.Linear(8, 1, bias=True)
        nn.init.constant_(self.gate.bias, -3.0)
        # Keep the first projection trainable at initialization.  Zeroing both
        # projections would create a dead gradient path; the prescribed zero
        # initialization is applied only to the final 64->144 projection.
        _zero_linear(self.read_to_delta[2])

    def query(self, h: torch.Tensor) -> torch.Tensor:
        return normalize_rows(self.q_proj(self.query_norm(h)))

    def key(self, h: torch.Tensor) -> torch.Tensor:
        return normalize_rows(self.q_proj(self.key_norm(h)))

    def finish(self, h: torch.Tensor, base_logp: torch.Tensor,
               read_a: torch.Tensor, read_b: torch.Tensor,
               sim_a: torch.Tensor, ent_a: torch.Tensor,
               sim_b: torch.Tensor, ent_b: torch.Tensor,
               avail_a: torch.Tensor, avail_b: torch.Tensor,
               retrieval: Optional[List[dict]] = None,
               extra: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
        n = h.shape[0]
        availability = torch.stack([avail_a, avail_b], dim=1).to(h.dtype)
        sim_stats = torch.stack([sim_a, ent_a, sim_b, ent_b], dim=1)
        gate_input = torch.cat([base_logp.detach(), sim_stats, availability], dim=1)
        gate = torch.sigmoid(self.gate(gate_input)).squeeze(1)
        raw = self.read_to_delta(torch.cat([read_a, read_b], dim=1))
        delta = gate[:, None] * raw
        hnorm = h.norm(dim=1, keepdim=True)
        dnorm = delta.norm(dim=1, keepdim=True)
        cap = self.delta_cap * hnorm
        scale = torch.minimum(torch.ones_like(dnorm), cap / dnorm.clamp_min(1e-12))
        delta = delta * scale
        empty = availability.sum(1).eq(0)
        if bool(empty.any()):
            delta = torch.where(empty[:, None], torch.zeros_like(delta), delta)
        out = h + delta
        diagnostics = {
            "delta_h": delta,
            "raw_delta": raw,
            "gate": gate,
            "gate_input": gate_input,
            "availability": availability,
            "similarity_stats": sim_stats,
            "delta_ratio": delta.norm(dim=1) / hnorm.squeeze(1).clamp_min(1e-12),
            "cap_fraction": (scale.squeeze(1) < 0.999999).to(h.dtype),
            "empty_fraction": empty.to(h.dtype),
            "retrieval": retrieval or [],
        }
        if extra:
            diagnostics.update(extra)
        return out, diagnostics

    def empty_forward(self, h: torch.Tensor, base_logp: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        z = h.new_zeros((h.shape[0], self.memory_dim))
        zero = h.new_zeros((h.shape[0],))
        return self.finish(h, base_logp, z, z, zero, zero, zero, zero, zero, zero)


def memory_loss(logp_memory: torch.Tensor, base_logp: torch.Tensor,
                labels: torch.Tensor, ids: Sequence[int], class_weights: torch.Tensor,
                delta: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """Shared formal branch loss with base-error weighting 2 and base-correct 1."""
    idx = torch.as_tensor(list(ids), dtype=torch.long, device=logp_memory.device)
    y = labels[idx].long()
    base_pred = base_logp.detach().argmax(1)
    hard = base_pred.ne(y)
    sample_weight = torch.where(hard, torch.full_like(y, 2.0, dtype=torch.float32),
                                torch.full_like(y, 1.0, dtype=torch.float32))
    nll = F.nll_loss(logp_memory, y, weight=class_weights.to(logp_memory.device).float(), reduction="none")
    cls = (nll * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
    correct = ~hard
    preserve = ((logp_memory - base_logp.detach()).square().sum(1)[correct].mean()
                if bool(correct.any()) else logp_memory.new_zeros(()))
    ratio = delta.norm(dim=1) / h.norm(dim=1).clamp_min(1e-8)
    reg = ratio.square().mean()
    total = cls + 0.20 * preserve + 0.02 * reg
    return total, {
        "mem_cls": float(cls.detach()),
        "correct_preserve": float(preserve.detach()),
        "delta_reg": float(reg.detach()),
        "base_error_count": int(hard.sum()),
        "gate_mean": float("nan"),
        "delta_ratio_mean": float(ratio.detach().mean()),
    }


def detach_memory(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: detach_memory(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [detach_memory(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(detach_memory(v) for v in obj)
    return copy.deepcopy(obj)


def as_device(x, device, dtype=None):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype or x.dtype)
    return x

