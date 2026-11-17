"""BP-CPSDC counterfactual dynamic residual module.

The module has no stochastic layers.  Real and patient-mean counterfactual
windows pass through the same parameters and the candidate residual is a
bias-free projection of their difference.  Therefore an exact equality of the
two inputs produces an exact zero residual (up to the arithmetic performed on
the same tensors).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation,
                              padding=dilation)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + F.gelu(y))


class BPCPSDC(nn.Module):
    """Shared real/counterfactual temporal evidence and conservative CAM delta."""

    def __init__(
        self,
        text_embeddings: torch.Tensor,
        mean_center: torch.Tensor,
        mean_scale: torch.Tensor,
        residual_scale: torch.Tensor,
        residual_cap: float = 0.05,
    ):
        super().__init__()
        if tuple(text_embeddings.shape[:1]) != (8,):
            raise ValueError("BP-CPSDC requires eight frozen prompt embeddings")
        for value, name in ((mean_center, "mean_center"), (mean_scale, "mean_scale"),
                            (residual_scale, "residual_scale")):
            if tuple(value.shape) != (252,):
                raise ValueError(f"{name} must have shape [252]")
        self.register_buffer("text_embeddings", F.normalize(text_embeddings.float(), dim=-1))
        self.register_buffer("mean_center", mean_center.float())
        self.register_buffer("mean_scale", mean_scale.float().clamp_min(1e-6))
        self.register_buffer("residual_scale", residual_scale.float().clamp_min(1e-6))
        self.register_buffer("dynamic_applicability", torch.tensor([0., 0., 0., 0., 0., 1., 0., .25]))
        self.residual_cap = float(residual_cap)

        # Shared temporal encoder.  No dropout or stochastic positional encoding.
        self.input_projection = nn.Linear(252, 64)
        self.input_norm = nn.LayerNorm(64)
        self.tcn1 = TemporalBlock(64, 1)
        self.tcn2 = TemporalBlock(64, 2)
        self.text_project = nn.Linear(self.text_embeddings.shape[1], 64, bias=False)
        self.dynamic_key = nn.Linear(64, 64, bias=False)
        self.dynamic_value = nn.Linear(64, 64, bias=False)

        # The final candidate has no bias: u=0 => h=0 exactly.
        self.output_projection = nn.Linear(64, 144, bias=False)
        # Gate uses only detached-CAM/confidence and the dynamic difference strength.
        self.reliability = nn.Sequential(
            nn.Linear(144 + 64 + 3, 32), nn.GELU(), nn.Linear(32, 1)
        )
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _valid_mean(self, windows: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        valid = torch.arange(windows.shape[1], device=windows.device).unsqueeze(0) < counts.unsqueeze(1)
        return windows.masked_fill(~valid.unsqueeze(-1), 0).sum(1) / counts.to(windows.dtype).unsqueeze(1)

    def _encode(self, windows: torch.Tensor, counts: torch.Tensor,
                patient_mean: torch.Tensor, counterfactual: bool) -> torch.Tensor:
        n, max_w, _ = windows.shape
        out = windows.new_zeros((n, max_w, 64))
        for length_value in torch.unique(counts, sorted=True).tolist():
            length = int(length_value)
            rows = torch.nonzero(counts == length, as_tuple=False).flatten()
            seq = windows.index_select(0, rows)[:, :length]
            means = patient_mean.index_select(0, rows)
            if counterfactual:
                seq = means[:, None, :].expand(-1, length, -1)
            # The same scaler and the same encoder are used in both paths.
            seq = (seq - self.mean_center) / self.mean_scale
            h = self.input_norm(self.input_projection(seq))
            h = self.tcn1(h)
            h = self.tcn2(h)
            out[rows, :length] = h
        return out

    @staticmethod
    def _attention(query: torch.Tensor, tokens: torch.Tensor, counts: torch.Tensor):
        # query [N,8,64], tokens [N,W,64]
        scores = torch.einsum("nkd,nld->nkl", query, tokens) / (tokens.shape[-1] ** 0.5)
        valid = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0) < counts.unsqueeze(1)
        scores = scores.masked_fill(~valid[:, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        evidence = torch.einsum("nkl,nld->nkd", weights, tokens)
        return evidence, weights

    def forward(
        self,
        cam144: torch.Tensor,
        base_logp: torch.Tensor,
        windows: torch.Tensor,
        counts: torch.Tensor,
        *,
        prompt_mask: Optional[torch.Tensor] = None,
        force_zero: bool = False,
    ):
        if cam144.ndim != 2 or cam144.shape[1] != 144:
            raise ValueError(f"CAM must be [N,144], got {tuple(cam144.shape)}")
        if windows.ndim != 3 or windows.shape[0] != cam144.shape[0] or windows.shape[2] != 252:
            raise ValueError("windows must be [N,W,252] and aligned with CAM rows")
        if counts.shape != (cam144.shape[0],) or torch.any(counts < 1) or torch.any(counts > windows.shape[1]):
            raise ValueError("invalid dynamic window counts")
        if base_logp.shape != (cam144.shape[0], 2):
            raise ValueError("base_logp must be [N,2]")

        mean = self._valid_mean(windows, counts)
        real_tokens = self._encode(windows, counts, mean, counterfactual=False)
        cf_tokens = self._encode(windows, counts, mean, counterfactual=True)
        query = self.text_project(self.text_embeddings).unsqueeze(0).expand(cam144.shape[0], -1, -1)
        real_evidence, real_attention = self._attention(query, self.dynamic_value(self.dynamic_key(real_tokens)), counts)
        cf_evidence, cf_attention = self._attention(query, self.dynamic_value(self.dynamic_key(cf_tokens)), counts)
        difference = real_evidence - cf_evidence
        # PPU kernels can be non-bitwise deterministic even for equal matrix
        # inputs.  Detect the exact counterfactual identity in input space and
        # force its semantic difference to the mathematical zero tensor.
        valid = torch.arange(windows.shape[1], device=windows.device).unsqueeze(0) < counts.unsqueeze(1)
        # Equality to the patient-mean replacement is equivalent to all valid
        # windows being exactly identical; comparing to the first valid window
        # avoids a floating-point reduction of the mean reintroducing a tiny
        # rounding residual.
        static_identity = ((windows - windows[:, :1, :]).masked_fill(~valid.unsqueeze(-1), 0).abs().sum(dim=(1, 2)) == 0)
        difference = torch.where(static_identity[:, None, None], torch.zeros_like(difference), difference)

        if prompt_mask is None:
            prompt_mask = torch.ones(8, dtype=cam144.dtype, device=cam144.device)
        else:
            prompt_mask = prompt_mask.to(device=cam144.device, dtype=cam144.dtype).reshape(8)
        applicability = self.dynamic_applicability.to(cam144).view(1, 8, 1) * prompt_mask.view(1, 8, 1)
        denom = applicability.sum().clamp_min(1.0)
        u = (difference * applicability).sum(dim=1) / denom
        h = self.output_projection(u)

        base_prob = base_logp.detach().exp()
        confidence = base_prob.max(dim=1, keepdim=True).values
        dynamic_strength = difference.square().sum(dim=(1, 2), keepdim=False).sqrt().unsqueeze(1)
        u_strength = u.square().sum(dim=1, keepdim=True).sqrt()
        cam_norm = cam144.detach().norm(dim=1, keepdim=True).clamp_min(1e-6)
        cam_normalized = cam144.detach() / cam_norm
        gate_input = torch.cat((cam_normalized, u, confidence, dynamic_strength, u_strength), dim=1)
        gate = torch.sigmoid(self.reliability(gate_input))

        # ReZero-style zero scalar and explicit dependence on counterfactual r.
        delta_raw = torch.tanh(self.alpha) * gate * h
        if force_zero:
            delta_raw = torch.zeros_like(delta_raw)
        raw_ratio = delta_raw.norm(dim=1, keepdim=True) / cam_norm
        ratio_safe = raw_ratio.clamp_min(1e-8)
        bounded_scale = self.residual_cap * torch.tanh(raw_ratio / self.residual_cap) / ratio_safe
        # Use the continuous r->0 limit (scale=1) at the zero-initialized
        # residual.  A hard zero branch here would also zero alpha's first
        # gradient and make the ReZero parameter impossible to train.
        scale = torch.where(raw_ratio > 0, bounded_scale, torch.ones_like(raw_ratio))
        delta = delta_raw * scale
        diagnostics = {
            "real_evidence": real_evidence,
            "cf_evidence": cf_evidence,
            "difference": difference,
            "u": u,
            "candidate": h,
            "gate": gate,
            "alpha": self.alpha.detach().clone(),
            "raw_delta": delta_raw,
            "delta": delta,
            "relative_residual": delta.norm(dim=1) / cam_norm.squeeze(1),
            "raw_relative_residual": raw_ratio.squeeze(1),
            "residual_limit_hit": (raw_ratio.squeeze(1) > self.residual_cap).float(),
            "real_attention": real_attention,
            "cf_attention": cf_attention,
            "mean_feature": mean,
        }
        return delta, diagnostics


def fold_dynamic_scaler(windows: torch.Tensor, counts: torch.Tensor, train_indices: torch.Tensor):
    """Fit fold-local mean/residual scales from training patients only."""
    valid = torch.arange(windows.shape[1], device=windows.device).unsqueeze(0) < counts.unsqueeze(1)
    means = windows.masked_fill(~valid.unsqueeze(-1), 0).sum(1) / counts.to(windows.dtype).unsqueeze(1)
    rows = train_indices.to(device=windows.device, dtype=torch.long)
    mean_center = means.index_select(0, rows).mean(0)
    mean_scale = means.index_select(0, rows).std(0, unbiased=False).clamp_min(1e-5)
    residual = windows.index_select(0, rows) - means.index_select(0, rows).unsqueeze(1)
    mask = valid.index_select(0, rows)
    residual_scale = residual[mask].std(0, unbiased=False).clamp_min(1e-4)
    return means, mean_center, mean_scale, residual_scale
