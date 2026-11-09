"""DCER-v2 feature residual applied to the 144-D CAM input of OutputLayer."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class DCERV2(nn.Module):
    def __init__(self, text_bank: torch.Tensor, semantic_dim: int = 32, temperature: float = 0.2):
        super().__init__()
        if tuple(text_bank.shape) != (2, 1024):
            raise ValueError(f"text_bank must be [2,1024] in [ASD,TD] order, got {tuple(text_bank.shape)}")
        self.register_buffer("text_bank", text_bank.detach().float().clone())
        self.text_proj = nn.Linear(1024, semantic_dim, bias=False)
        self.context_film = nn.Linear(36, 4 * semantic_dim)
        self.query_norm = nn.LayerNorm(semantic_dim)
        self.key_proj = nn.Linear(9, semantic_dim, bias=False)
        self.evidence_proj = nn.Linear(18, 9, bias=False)
        self.validity = nn.Sequential(
            nn.Linear(46, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.gamma_raw = nn.Parameter(torch.zeros(()))
        self.temperature = float(temperature)

    @property
    def gamma(self) -> torch.Tensor:
        return 0.15 * torch.tanh(self.gamma_raw)

    @staticmethod
    def _global_context(h: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        return F.avg_pool1d(h.reshape(n * 4, 1, 36), kernel_size=4, stride=4).reshape(n, 4, 9)

    def contrast_scores(self, tokens: torch.Tensor, global_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text = self.text_proj(self.text_bank)
        film = self.context_film(global_flat).reshape(-1, 2, 2, text.shape[-1])
        scale = 0.5 * torch.tanh(film[:, :, 0])
        shift = 0.5 * torch.tanh(film[:, :, 1])
        query = self.query_norm(text.unsqueeze(0) * (1.0 + scale) + shift)
        query = F.normalize(query, dim=-1, eps=1e-8)
        keys = F.normalize(self.key_proj(tokens), dim=-1, eps=1e-8)
        scores = torch.einsum("ncd,ntd->nct", query, keys)
        difference = torch.tanh((scores[:, 0] - scores[:, 1]) / (2.0 * self.temperature))
        return query, keys, scores, difference

    def forward(
        self,
        h: torch.Tensor,
        *,
        common_suppression: bool,
        validity_enabled: bool,
        validity_gate_enabled: bool = True,
        gamma_enabled: bool = True,
        semantic_neutral: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if h.ndim != 2 or tuple(h.shape[1:]) != (144,):
            raise ValueError(f"CAM input must be [N,144], got {tuple(h.shape)}")
        n = h.shape[0]
        h4 = h.reshape(n, 4, 36)
        tokens = h4.reshape(n, 16, 9)
        global_tokens = self._global_context(h)
        global_flat = global_tokens.reshape(n, 36)

        query, keys, scores, difference = self.contrast_scores(tokens, global_flat)
        attention = torch.softmax(scores / self.temperature, dim=-1)
        if semantic_neutral:
            difference = torch.zeros_like(difference)

        evidence_asd = F.relu(difference).unsqueeze(-1) * tokens
        evidence_td = F.relu(-difference).unsqueeze(-1) * tokens
        raw_residual = self.evidence_proj(torch.cat((evidence_asd, evidence_td), dim=-1))

        if common_suppression:
            common = (16.0 * F.relu(torch.minimum(attention[:, 0], attention[:, 1]) - 1.0 / 16.0)).clamp(0.0, 1.0)
            common_factor = 1.0 - common
        else:
            common = torch.zeros_like(difference)
            common_factor = torch.ones_like(difference)

        if validity_enabled:
            validity_input = torch.cat((tokens, global_flat[:, None, :].expand(-1, 16, -1), difference.unsqueeze(-1)), dim=-1)
            validity = torch.sigmoid(self.validity(validity_input).squeeze(-1))
        else:
            validity = torch.ones_like(difference)
        if not validity_gate_enabled:
            validity_factor = torch.ones_like(validity)
        else:
            validity_factor = validity

        gated = raw_residual * (common_factor * validity_factor).unsqueeze(-1)
        token_rms = torch.sqrt(tokens.square().mean(dim=-1, keepdim=True).clamp_min(1e-8))
        bounded = token_rms * torch.tanh(gated / (token_rms + 1e-8))
        gamma = self.gamma if gamma_enabled else self.gamma * 0.0
        h_prime = h + gamma * bounded.reshape(n, 144)

        return {
            "h_prime": h_prime,
            "tokens": tokens,
            "global_tokens": global_tokens,
            "query": query,
            "keys": keys,
            "scores": scores,
            "attention": attention,
            "difference": difference,
            "common": common,
            "common_factor": common_factor,
            "validity": validity,
            "raw_residual": raw_residual,
            "bounded_residual": bounded,
            "residual": (gamma * bounded).reshape(n, 144),
            "gamma": gamma,
        }


def tensor_sha(module: nn.Module, include_buffers: bool = False) -> str:
    import hashlib

    digest = hashlib.sha256()
    state = module.state_dict() if include_buffers else dict(module.named_parameters())
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
