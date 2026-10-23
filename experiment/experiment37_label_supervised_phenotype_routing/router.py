"""Low-capacity phenotype-conditioned modulation of existing patient edges."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class PCPRRouter(nn.Module):
    def __init__(self, input_dim: int = 768, key_dim: int = 64,
                 beta_max: float = 0.25, beta_initial: float = 0.10):
        super().__init__()
        self.query = nn.Linear(input_dim, key_dim, bias=False)
        self.key = nn.Linear(input_dim, key_dim, bias=False)
        self.beta_raw = nn.Parameter(torch.empty(()))
        self.beta_max = float(beta_max)
        initial = min(max(beta_initial / beta_max, 1e-5), 1 - 1e-5)
        nn.init.constant_(self.beta_raw, math.log(initial / (1 - initial)))
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)

    @property
    def beta(self):
        return self.beta_max * torch.sigmoid(self.beta_raw)

    def forward(self, raw_adj: torch.Tensor, condition: torch.Tensor,
                directed: bool):
        if raw_adj.ndim != 2 or raw_adj.shape[0] != raw_adj.shape[1]:
            raise ValueError("raw adjacency must be square")
        if condition.shape != (raw_adj.shape[0], self.query.in_features):
            raise ValueError("condition rows/dimension do not match the graph")
        n = raw_adj.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=raw_adj.device)
        support = (raw_adj > 0) & ~eye
        # Text vectors are cached unit vectors; apply the same scale convention
        # to the numeric-condition controls before the shared Q/K projections.
        condition = F.normalize(condition, p=2, dim=-1)
        constant_condition = torch.equal(condition, condition[:1].expand_as(condition))
        q = self.query(condition)
        k = self.key(condition)
        scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(q.shape[-1])
        degree = support.sum(dim=1, keepdim=True).clamp_min(1)
        row_center = (scores * support.to(scores.dtype)).sum(dim=1, keepdim=True) / degree
        centered = scores - row_center
        if not directed:
            centered = (centered + centered.transpose(0, 1)) * 0.5
        # Identical participant conditions imply a constant supported-edge
        # score. Make the mathematical identity exact in floating point.
        if constant_condition:
            centered = torch.zeros_like(scores)
        pre_tanh_centered = centered
        centered = torch.tanh(centered)
        gate = torch.exp(self.beta * centered)
        offdiag = torch.where(support, raw_adj * gate, torch.zeros_like(raw_adj))
        conditioned = offdiag + torch.diag(torch.diagonal(raw_adj))
        if not directed and not constant_condition:
            conditioned = (conditioned + conditioned.transpose(0, 1)) * 0.5
        conditioned = torch.where(support | torch.diag(eye), conditioned, torch.zeros_like(conditioned))
        with_self = conditioned + torch.eye(n, dtype=raw_adj.dtype, device=raw_adj.device)
        row_sum = with_self.sum(dim=1, keepdim=True).clamp_min(torch.finfo(with_self.dtype).eps)
        propagation = with_self / row_sum
        diagnostics = {
            "edge_support": support,
            "scores": scores,
            "pre_tanh_centered_scores": pre_tanh_centered,
            "centered_scores": centered,
            "raw_adj": raw_adj,
            "conditioned_adj": conditioned,
            "propagation": propagation,
            "beta": self.beta.detach(),
        }
        return propagation, diagnostics


def validate_router():
    torch.manual_seed(3601)
    n = 9
    x = torch.randn(n, 768)
    a = torch.rand(n, n)
    a = (a + a.T) * 0.5
    a[a < 0.65] = 0
    a.fill_diagonal_(0)
    router = PCPRRouter()
    p_sym, ds = router(a, x, directed=False)
    assert torch.allclose(ds["conditioned_adj"], ds["conditioned_adj"].T, atol=1e-7)
    assert torch.allclose(p_sym.sum(1), torch.ones(n), atol=1e-7)
    assert torch.equal(ds["conditioned_adj"][a == 0], torch.zeros_like(a[a == 0]))
    assert torch.isfinite(p_sym).all()
    p_sym.sum().backward()
    assert router.query.weight.grad is not None and router.query.weight.grad.norm() > 0
    assert router.key.weight.grad is not None and router.key.weight.grad.norm() > 0
    assert router.beta_raw.grad is not None and torch.isfinite(router.beta_raw.grad)
    router.zero_grad(set_to_none=True)
    p_dir, dd = router(a, x, directed=True)
    assert torch.allclose(p_dir.sum(1), torch.ones(n), atol=1e-7)
    assert torch.isfinite(p_dir).all()
    constant = torch.ones_like(x)
    p_constant, dc = router(a, constant, directed=False)
    baseline = (a + torch.eye(n, dtype=a.dtype, device=a.device))
    baseline = baseline / baseline.sum(1, keepdim=True)
    assert torch.equal(dc["pre_tanh_centered_scores"], torch.zeros_like(ds["scores"]))
    assert torch.equal(p_constant, baseline), "constant-condition routing is not an exact identity"
    return {
        "symmetric_raw_max_error": float((ds["conditioned_adj"] - ds["conditioned_adj"].T).abs().max()),
        "directed_asymmetry_l1": float((dd["conditioned_adj"] - dd["conditioned_adj"].T).abs().sum()),
        "beta_initial": float(router.beta.detach()),
        "supported_edges": int((a > 0).sum()),
        "constant_condition_identity": "BITWISE_EQUAL",
    }
