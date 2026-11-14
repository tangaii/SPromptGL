"""Shared competitive-memory fusion for Experiment 79.3.

Only detached CAM144 is supplied by the DARG base.  The complete feature
correction is generated from the two disease-specific memory reads; CAM and
the base probabilities can affect querying and gate strength, but cannot
generate a bypass correction path.
"""
from __future__ import annotations
import copy
import math
from typing import Optional, Sequence
import torch
from torch import nn
import torch.nn.functional as F

CAM_DIM = 144
MEMORY_DIM = 64
TEMPERATURE = 0.2
DELTA_CAP = 0.075

def normalize_rows(x, eps=1e-8):
    return F.normalize(x, dim=-1, eps=eps)

def output_layer_logp(output_layer: nn.Module, h: torch.Tensor):
    from torch.func import functional_call
    was_training = bool(output_layer.training)
    output_layer.eval()
    params = {n: p.detach() for n, p in output_layer.named_parameters()}
    buffers = {n: b.detach() for n, b in output_layer.named_buffers()}
    out = functional_call(output_layer, (params, buffers), (h, None))
    if was_training:
        output_layer.train()
    return out[0] if isinstance(out, (tuple, list)) else out

def stable_softmax(scores, valid):
    if scores.shape[-1] == 0:
        return scores
    masked = scores.masked_fill(~valid, -1e9)
    return torch.softmax(masked, dim=-1) * valid.to(scores.dtype)

def _zero(linear):
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)

class MemoryFusionBase(nn.Module):
    def __init__(self, cam_dim=CAM_DIM, memory_dim=MEMORY_DIM,
                 temperature=TEMPERATURE, delta_cap=DELTA_CAP, gate_bias=-1.5):
        super().__init__()
        self.cam_dim, self.memory_dim = int(cam_dim), int(memory_dim)
        self.temperature, self.delta_cap = float(temperature), float(delta_cap)
        # Shared query/key map: retrieval and attention live in one normalized space.
        self.q_proj = nn.Linear(self.cam_dim, self.memory_dim, bias=False)
        self.query_norm = nn.LayerNorm(self.cam_dim)
        self.key_norm = nn.LayerNorm(self.cam_dim)
        self.v_proj = nn.Linear(self.cam_dim, self.memory_dim, bias=False)
        self.read_to_delta = nn.Sequential(
            nn.Linear(3 * self.memory_dim, self.memory_dim, bias=False),
            nn.GELU(), nn.Linear(self.memory_dim, self.cam_dim, bias=False))
        # base p(2), memory p(2), p diff, reliability, entropy(2), availability(2)
        self.gate = nn.Linear(10, 1, bias=True)
        nn.init.constant_(self.gate.bias, float(gate_bias))
        _zero(self.read_to_delta[2])

    def query(self, h):
        return normalize_rows(self.q_proj(self.query_norm(h)))

    def key(self, h):
        return normalize_rows(self.q_proj(self.key_norm(h)))

    def finish(self, h, base_logp, read_a, read_b, score_a, score_b,
               rel, ent_a, ent_b, avail_a, avail_b, retrieval=None, extra=None):
        # The two scores share a scale and compete in one softmax.
        evidence_logits = torch.stack([score_a, score_b], dim=1)
        finite = torch.isfinite(evidence_logits).all(1)
        evidence_logits = torch.where(finite[:, None], evidence_logits,
                                      torch.zeros_like(evidence_logits))
        mem_prob = torch.softmax(evidence_logits, dim=1)
        avail = torch.stack([avail_a, avail_b], dim=1).to(h.dtype)
        diff = mem_prob[:, 0] - mem_prob[:, 1]
        rel = rel.to(h.dtype).clamp(0, 1)
        gate_x = torch.cat([base_logp.detach(), mem_prob.detach(), diff[:, None],
                            rel[:, None].detach(), ent_a[:, None], ent_b[:, None], avail], 1)
        gate = torch.sigmoid(self.gate(gate_x)).squeeze(1)
        raw = self.read_to_delta(torch.cat([read_a, read_b, read_a - read_b], 1))
        delta = gate[:, None] * raw
        hnorm = h.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cap = self.delta_cap * hnorm
        dnorm = delta.norm(dim=1, keepdim=True)
        scale = torch.minimum(torch.ones_like(dnorm), cap / dnorm.clamp_min(1e-12))
        delta = delta * scale
        empty = avail.sum(1).eq(0)
        if bool(empty.any()):
            delta = torch.where(empty[:, None], torch.zeros_like(delta), delta)
        diag = {
            "delta_h": delta, "raw_delta": raw, "gate": gate,
            "availability": avail, "similarity_stats": torch.stack([score_a, score_b, ent_a, ent_b], 1),
            "delta_ratio": delta.norm(dim=1) / hnorm.squeeze(1),
            "cap_fraction": (scale.squeeze(1) < 0.999999).to(h.dtype),
            "empty_fraction": empty.to(h.dtype), "retrieval": retrieval or [],
            "evidence_logits": evidence_logits, "memory_probability": mem_prob,
            "memory_reliability": rel,
        }
        if extra: diag.update(extra)
        return h + delta, diag

    def empty_forward(self, h, base_logp):
        z = h.new_zeros((h.shape[0], self.memory_dim)); zero = h.new_zeros((h.shape[0],))
        return self.finish(h, base_logp, z, z, zero, zero, zero, zero, zero, zero, zero)

def memory_loss(logp_memory, base_logp, labels, ids, class_weights, delta, h, evidence_logits=None):
    idx = torch.as_tensor(list(ids), dtype=torch.long, device=logp_memory.device)
    y = labels[idx].long()
    base_pred = base_logp.detach().argmax(1)
    hard = base_pred.ne(y)
    w = torch.where(hard, torch.full_like(y, 3., dtype=torch.float32),
                    torch.full_like(y, 1., dtype=torch.float32))
    nll_each = F.nll_loss(logp_memory, y, weight=class_weights.to(logp_memory.device).float(), reduction="none")
    cls = (nll_each * w).sum() / w.sum().clamp_min(1.)
    true = logp_memory.gather(1, y[:, None]).squeeze(1)
    wrong = logp_memory.gather(1, (1-y)[:, None]).squeeze(1)
    wrong_margin = F.softplus(0.2 - (true - wrong))[hard].mean() if bool(hard.any()) else logp_memory.new_zeros(())
    preserve = ((logp_memory - base_logp.detach()).square().sum(1)[~hard].mean()
                if bool((~hard).any()) else logp_memory.new_zeros(()))
    evidence = (F.cross_entropy(evidence_logits, y) if evidence_logits is not None else logp_memory.new_zeros(()))
    ratio = delta.norm(dim=1) / h.norm(dim=1).clamp_min(1e-8)
    reg = ratio.square().mean()
    total = cls + .20 * wrong_margin + .10 * preserve + .10 * evidence + .01 * reg
    return total, {"mem_cls": float(cls.detach()), "wrong_margin": float(wrong_margin.detach()),
        "correct_preserve": float(preserve.detach()), "evidence": float(evidence.detach()),
        "delta_reg": float(reg.detach()), "base_error_count": int(hard.sum()),
        "delta_ratio_mean": float(ratio.detach().mean())}

def detach_memory(obj):
    if torch.is_tensor(obj): return obj.detach().cpu().clone()
    if isinstance(obj, dict): return {k: detach_memory(v) for k,v in obj.items()}
    if isinstance(obj, list): return [detach_memory(v) for v in obj]
    if isinstance(obj, tuple): return tuple(detach_memory(v) for v in obj)
    return copy.deepcopy(obj)
