"""PSDC temporal encoder, frozen-text queries, and bounded CAM residual."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=padding)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(F.gelu(y))
        return self.norm(x + y)


class PSDC(nn.Module):
    def __init__(
        self,
        text_embeddings: torch.Tensor,
        mean_center: torch.Tensor,
        mean_scale: torch.Tensor,
        residual_scale: torch.Tensor,
        residual_cap: float = 0.1,
    ):
        super().__init__()
        if text_embeddings.ndim != 2 or text_embeddings.shape[0] != 8:
            raise ValueError("PSDC requires exactly eight frozen text embeddings")
        if mean_center.shape != (252,) or mean_scale.shape != (252,) or residual_scale.shape != (252,):
            raise ValueError("fold-local dynamic scaler shape must be 252")
        text = F.normalize(text_embeddings.float(), dim=-1)
        self.register_buffer("text_embeddings", text)
        self.register_buffer("mean_center", mean_center.float())
        self.register_buffer("mean_scale", mean_scale.float().clamp_min(1e-6))
        self.register_buffer("residual_scale", residual_scale.float().clamp_min(1e-6))
        self.register_buffer("dynamic_applicability", torch.tensor([0, 0, 0, 0, 0, 1.0, 0, 0.25]))
        self.residual_cap = float(residual_cap)

        self.static_tokenizer = nn.Linear(144, 8 * 64)
        self.static_token_norm = nn.LayerNorm(64)
        self.mean_projection = nn.Linear(252, 64)
        self.residual_projection = nn.Linear(252, 64)
        self.temporal_input_norm = nn.LayerNorm(64)
        self.tcn1 = TemporalBlock(64, dilation=1)
        self.tcn2 = TemporalBlock(64, dilation=2)
        self.text_project = nn.Linear(text.shape[1], 64, bias=False)
        self.static_key = nn.Linear(64, 64, bias=False)
        self.static_value = nn.Linear(64, 64, bias=False)
        self.dynamic_key = nn.Linear(64, 64, bias=False)
        self.dynamic_value = nn.Linear(64, 64, bias=False)
        self.evidence = nn.Sequential(nn.Linear(64 * 3, 64), nn.GELU(), nn.Linear(64, 64))
        self.gate = nn.Sequential(nn.Linear(64 * 3, 64), nn.GELU(), nn.Linear(64, 64))
        self.output_projection = nn.Linear(64, 144)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output_projection.bias)
        nn.init.constant_(self.gate[-1].bias, -1.1)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _temporal_tokens(
        self,
        windows: torch.Tensor,
        counts: torch.Tensor,
        mean: torch.Tensor,
        surrogate: bool,
        temporal_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        n, max_windows, _ = windows.shape
        output = windows.new_zeros((n, max_windows, 64))
        mean_z = (mean - self.mean_center) / self.mean_scale
        lengths = torch.unique(counts, sorted=True).tolist()
        for length_value in lengths:
            length = int(length_value)
            if length < 1:
                continue
            indices = torch.nonzero(counts == length, as_tuple=False).flatten()
            sequence = windows.index_select(0, indices)[:, :length]
            if temporal_indices is not None:
                order = temporal_indices.index_select(0, indices)[:, :length]
                sequence = torch.gather(sequence, 1, order.unsqueeze(-1).expand(-1, -1, 252))
            patient_mean = mean.index_select(0, indices)
            if surrogate:
                sequence_mean = patient_mean[:, None, :].expand(-1, length, -1)
                residual = torch.zeros_like(sequence_mean)
            else:
                sequence_mean = sequence
                residual = sequence - patient_mean[:, None, :]
            mean_feature = mean_z.index_select(0, indices)[:, None, :]
            residual_feature = residual / self.residual_scale
            h = self.temporal_input_norm(
                self.mean_projection(mean_feature) + self.residual_projection(residual_feature)
            )
            h = self.tcn1(h)
            h = self.tcn2(h)
            output[indices, :length] = h
        return output

    @staticmethod
    def _cross_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.einsum("nkd,nld->nkl", query, key) / (query.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("nkl,nld->nkd", weights, value), weights

    def forward(
        self,
        cam144: torch.Tensor,
        windows: torch.Tensor,
        counts: torch.Tensor,
        *,
        surrogate: bool = False,
        dynamic_off: bool = False,
        temporal_indices: torch.Tensor | None = None,
        prompt_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if cam144.ndim != 2 or cam144.shape[1] != 144:
            raise ValueError(f"expected CAM144 [N,144], received {tuple(cam144.shape)}")
        if windows.ndim != 3 or windows.shape[0] != cam144.shape[0] or windows.shape[2] != 252:
            raise ValueError("window data must be [N,W,252] and aligned to CAM patient rows")
        if counts.shape != (cam144.shape[0],) or torch.any(counts < 1) or torch.any(counts > windows.shape[1]):
            raise ValueError("invalid window counts/mask")
        if dynamic_off:
            zero = cam144.new_zeros(cam144.shape)
            return zero, {"delta_raw": zero, "static_evidence": zero, "dynamic_evidence": zero}

        n = cam144.shape[0]
        static_tokens = self.static_tokenizer(cam144).reshape(n, 8, 64)
        static_tokens = self.static_token_norm(static_tokens)
        valid_mask = torch.arange(windows.shape[1], device=windows.device).unsqueeze(0) < counts.unsqueeze(1)
        mean = windows.masked_fill(~valid_mask.unsqueeze(-1), 0).sum(dim=1) / counts.unsqueeze(1)
        temporal = self._temporal_tokens(windows, counts, mean, surrogate, temporal_indices)
        if prompt_indices is None:
            text = self.text_embeddings
        else:
            text = self.text_embeddings.index_select(0, prompt_indices.to(self.text_embeddings.device))
        query = self.text_project(text).unsqueeze(0).expand(n, -1, -1)
        static_evidence, static_weights = self._cross_attention(
            query, self.static_key(static_tokens), self.static_value(static_tokens)
        )

        temporal_by_group = temporal.new_zeros((n, 8, 64))
        temporal_weights = temporal.new_zeros((n, 8, windows.shape[1]))
        for length_value in torch.unique(counts, sorted=True).tolist():
            length = int(length_value)
            indices = torch.nonzero(counts == length, as_tuple=False).flatten()
            valid = temporal.index_select(0, indices)[:, :length]
            q = query.index_select(0, indices)
            selected, weights = self._cross_attention(
                q, self.dynamic_key(valid), self.dynamic_value(valid)
            )
            temporal_by_group[indices] = selected
            temporal_weights[indices, :, :length] = weights

        difference = temporal_by_group - static_evidence
        joint = torch.cat((static_evidence, temporal_by_group, difference), dim=-1)
        candidate = self.evidence(joint)
        gate = torch.sigmoid(self.gate(joint))
        applicability = self.dynamic_applicability.view(1, 8, 1)
        complementary = (candidate * gate * applicability).sum(dim=1) / applicability.sum().clamp_min(1.0)
        delta_raw = self.output_projection(complementary)
        raw_ratio = delta_raw.norm(dim=-1, keepdim=True) / cam144.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = self.residual_cap * torch.tanh(raw_ratio / self.residual_cap) / raw_ratio.clamp_min(1e-8)
        delta = delta_raw * scale
        diagnostics = {
            "delta_raw": delta_raw,
            "static_evidence": static_evidence,
            "dynamic_evidence": temporal_by_group,
            "static_attention": static_weights,
            "temporal_attention": temporal_weights,
            "gate": gate,
            "candidate": candidate,
            "relative_residual": delta.norm(dim=-1) / cam144.norm(dim=-1).clamp_min(1e-6),
            "residual_limit_hit": (raw_ratio > self.residual_cap).float(),
            "mean_feature": mean,
        }
        return delta, diagnostics


def fold_dynamic_scaler(windows: torch.Tensor, counts: torch.Tensor, train_indices: torch.Tensor):
    """Fit cross-patient feature scales using only outer-fold training patients."""
    valid_mask = torch.arange(windows.shape[1], device=windows.device).unsqueeze(0) < counts.unsqueeze(1)
    means = windows.masked_fill(~valid_mask.unsqueeze(-1), 0).sum(dim=1) / counts.unsqueeze(1)
    train_rows = train_indices.to(device=windows.device, dtype=torch.long)
    mean_center = means.index_select(0, train_rows).mean(dim=0)
    mean_scale = means.index_select(0, train_rows).std(dim=0, unbiased=False).clamp_min(1e-5)
    train_residual = windows.index_select(0, train_rows) - means.index_select(0, train_rows).unsqueeze(1)
    train_mask = valid_mask.index_select(0, train_rows)
    residual_values = train_residual[train_mask]
    residual_scale = residual_values.std(dim=0, unbiased=False).clamp_min(1e-4)
    return means, mean_center, mean_scale, residual_scale
