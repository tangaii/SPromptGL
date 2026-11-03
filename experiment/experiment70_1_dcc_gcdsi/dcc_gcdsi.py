from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DCCGCDSI(nn.Module):
    """Disease-contrastive, graph-conditioned, per-modality bounded injection."""

    def __init__(self, text_bank: torch.Tensor, residual_bound: float = 0.15, eps: float = 1e-8):
        super().__init__()
        if text_bank.ndim != 2 or text_bank.shape[0] != 2:
            raise ValueError(f"text_bank must be [2,D] ordered [ASD,TD], got {tuple(text_bank.shape)}")
        self.register_buffer("text_bank", text_bank.detach().float().clone(), persistent=True)
        self.residual_bound = float(residual_bound)
        self.eps = float(eps)

        self.query_h = nn.Linear(36, 36, bias=False)
        self.query_f = nn.Linear(320, 36, bias=False)
        self.q_cls = nn.Parameter(torch.empty(36))
        self.query_norm = nn.LayerNorm(36)
        self.w_delta = nn.Linear(text_bank.shape[1], 36, bias=False)
        self.w_center = nn.Linear(text_bank.shape[1], 36, bias=False)
        self.w_out = nn.Linear(36, 36, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(108, 36),
            nn.GELU(),
            nn.Linear(36, 36),
        )
        self.lambda_raw = nn.Parameter(torch.zeros(()))
        self.reset_parameters()

    @property
    def residual_scale(self) -> torch.Tensor:
        return self.residual_bound * torch.tanh(self.lambda_raw)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.q_cls.unsqueeze(0), gain=1.0)
        nn.init.zeros_(self.lambda_raw)

    def forward(
        self,
        h: torch.Tensor,
        graph_features: torch.Tensor,
        *,
        text_bank: torch.Tensor | None = None,
        disable_graph: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if h.ndim != 3 or tuple(h.shape[1:]) != (4, 36):
            raise ValueError(f"CAM representation must be [N,4,36], got {tuple(h.shape)}")
        if tuple(graph_features.shape) != (h.shape[0], 320):
            raise ValueError(f"GFR representation must be [N,320], got {tuple(graph_features.shape)}")
        bank = self.text_bank if text_bank is None else text_bank
        if bank.ndim != 2 or tuple(bank.shape) != (2, self.text_bank.shape[1]):
            raise ValueError(f"text bank must be [2,{self.text_bank.shape[1]}], got {tuple(bank.shape)}")

        tokens = F.normalize(bank, dim=-1, eps=self.eps)
        center_raw = (tokens[0] + tokens[1]) * 0.5
        center = F.normalize(center_raw, dim=0, eps=self.eps)
        delta_raw = tokens[0] - tokens[1]
        delta = delta_raw / (delta_raw.norm() + self.eps)
        key_delta = F.normalize(self.w_delta(delta), dim=0, eps=self.eps)

        graph_query_input = torch.zeros_like(graph_features) if disable_graph else graph_features
        q = self.query_norm(
            self.query_h(h)
            + self.query_f(graph_query_input).unsqueeze(1)
            + self.q_cls.view(1, 1, 36)
        )
        z = torch.einsum("nmd,d->nm", q, key_delta)
        attention = torch.softmax(torch.stack((z, -z), dim=-1), dim=-1)
        signed = attention[..., 0] - attention[..., 1]
        tanh_signed = torch.tanh(z)
        if not torch.allclose(signed, tanh_signed, atol=2e-6, rtol=2e-6):
            raise FloatingPointError("softmax([z,-z]) difference is inconsistent with tanh(z)")

        modulated_center = F.normalize(self.w_center(center), dim=0, eps=self.eps)
        center_by_patient = modulated_center.view(1, 1, 36).expand_as(h)
        candidate = torch.tanh(self.w_out(h * center_by_patient))
        gate = torch.sigmoid(self.gate(torch.cat((h, q, center_by_patient), dim=-1)))
        direction = gate * candidate

        cam_rms = h.square().mean(dim=-1, keepdim=True).sqrt()
        direction_rms = direction.square().mean(dim=-1, keepdim=True).sqrt()
        normalized_direction = direction / (direction_rms + self.eps)
        scale = self.residual_scale
        residual = scale * signed.unsqueeze(-1) * cam_rms * normalized_direction
        residual_ratio = residual.square().mean(dim=-1, keepdim=True).sqrt() / (cam_rms + self.eps)
        if not torch.isfinite(residual).all() or not torch.isfinite(residual_ratio).all():
            raise FloatingPointError("DCC-GCDSI produced a non-finite residual")
        if bool((residual_ratio > self.residual_bound + 1e-6).any()):
            raise AssertionError("per-patient/per-modality residual bound was exceeded")

        h_prime = h + residual
        diagnostics = {
            "query": q,
            "z": z,
            "attention": attention,
            "signed_strength": signed,
            "tanh_signed_strength": tanh_signed,
            "center": center,
            "delta": delta,
            "delta_norm": delta_raw.norm(),
            "text_cosine": F.cosine_similarity(tokens[0], tokens[1], dim=0),
            "key_delta": key_delta,
            "modulated_center": modulated_center,
            "gate": gate,
            "candidate": candidate,
            "residual": residual,
            "cam_rms": cam_rms.squeeze(-1),
            "candidate_rms": direction_rms.squeeze(-1),
            "residual_ratio": residual_ratio.squeeze(-1),
            "lambda": scale,
        }
        return h_prime, diagnostics
