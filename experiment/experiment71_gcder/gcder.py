"""Graph-conditioned dual-disease evidence retrieval.

The module is deliberately independent of the historical SPromptGL sources.  It
receives the four 36-dimensional CAM positions from one CAM forward pass and
returns a pair of shared-score disease evidence values.  The only decision
fusion is the bounded scalar beta applied to the original log probabilities.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class GCDER(nn.Module):
    """Two disease-conditioned queries read four CAM modality positions."""

    def __init__(self, text_dim: int):
        super().__init__()
        if int(text_dim) <= 0:
            raise ValueError("text_dim must be positive")
        # The two projections are shared by the ASD and TD query rows.  Biases
        # are excluded here so the text and patient graph terms are exactly the
        # two terms in the registered equation q = LN(W_t t + W_f f).
        self.text_proj = nn.Linear(int(text_dim), 36, bias=False)
        self.graph_proj = nn.Linear(320, 36, bias=False)
        self.query_norm = nn.LayerNorm(36)
        self.key = nn.Linear(36, 36, bias=False)
        self.value = nn.Linear(36, 36, bias=False)
        self.score_norm = nn.LayerNorm(36)
        self.score_hidden = nn.Linear(36, 36, bias=True)
        self.score_out = nn.Linear(36, 1, bias=True)
        self.beta_raw = nn.Parameter(torch.zeros(()))

    @property
    def beta(self) -> torch.Tensor:
        return 0.5 * torch.tanh(self.beta_raw)

    def forward(
        self,
        cam_positions: torch.Tensor,
        graph_features: torch.Tensor,
        text_bank: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if cam_positions.ndim != 3 or tuple(cam_positions.shape[1:]) != (4, 36):
            raise ValueError(f"CAM positions must be [N,4,36], got {tuple(cam_positions.shape)}")
        if graph_features.ndim != 2 or tuple(graph_features.shape) != (cam_positions.shape[0], 320):
            raise ValueError(f"GFR features must be [N,320], got {tuple(graph_features.shape)}")
        if text_bank.ndim != 2 or text_bank.shape[0] != 2 or text_bank.shape[1] != self.text_proj.in_features:
            raise ValueError(
                f"text bank must be [2,{self.text_proj.in_features}] ordered [ASD,TD], got {tuple(text_bank.shape)}"
            )

        # [N,2,36].  The text rows are broadcast over patients and the same
        # graph projection is added to both disease queries.
        q = self.query_norm(
            self.text_proj(text_bank).unsqueeze(0) + self.graph_proj(graph_features).unsqueeze(1)
        )
        k = self.key(cam_positions)
        v = self.value(cam_positions)
        attention = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(36.0), dim=-1)
        retrieved = torch.matmul(attention, v)
        score = self.score_out(F.gelu(self.score_hidden(self.score_norm(retrieved)))).squeeze(-1)
        return {
            "query": q,
            "key": k,
            "value": v,
            "attention": attention,
            "retrieved": retrieved,
            "scores": score,
            "delta_score": score[:, 0] - score[:, 1],
            "beta": self.beta,
            "beta_delta_score": self.beta * (score[:, 0] - score[:, 1]),
        }


def fuse_log_probabilities(logp_base: torch.Tensor, scores: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Correctly renormalise base log probabilities after score fusion."""
    if logp_base.shape != scores.shape or logp_base.shape[-1] != 2:
        raise ValueError(f"base logp and scores must both be [N,2], got {tuple(logp_base.shape)} and {tuple(scores.shape)}")
    return torch.log_softmax(logp_base + beta * scores, dim=-1)


def assert_shapes(out: Dict[str, torch.Tensor], n: int, text_dim: int) -> None:
    expected = {
        "query": (n, 2, 36),
        "key": (n, 4, 36),
        "value": (n, 4, 36),
        "attention": (n, 2, 4),
        "retrieved": (n, 2, 36),
        "scores": (n, 2),
        "delta_score": (n,),
    }
    for key, shape in expected.items():
        if tuple(out[key].shape) != shape:
            raise AssertionError(f"{key} shape {tuple(out[key].shape)} != {shape}")
    if out["query"].dtype != out["key"].dtype:
        raise AssertionError("query/key dtype mismatch")
    sums = out["attention"].sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise AssertionError("attention does not sum to one over the four CAM modalities")


def parameter_count(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())
