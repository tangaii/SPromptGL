from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class GCDSI(nn.Module):
    """Shared two-token disease semantic injection for four CAM modalities."""

    def __init__(self, text_width: int):
        super().__init__()
        self.query_h = nn.Linear(36, 36)
        self.query_f = nn.Linear(320, 36, bias=False)
        self.q_cls = nn.Parameter(torch.zeros(36))
        self.query_norm = nn.LayerNorm(36)
        self.key = nn.Linear(text_width, 36)
        self.value = nn.Linear(text_width, 36)
        self.output = nn.Linear(36, 36, bias=False)
        self.gate = nn.Sequential(nn.Linear(108, 36), nn.GELU(), nn.Linear(36, 36))
        self.gamma_raw = nn.Parameter(torch.zeros(()))
        nn.init.xavier_uniform_(self.output.weight, gain=0.01)

    @property
    def gamma(self) -> torch.Tensor:
        return 0.5 * torch.tanh(self.gamma_raw)

    def forward(self, h: torch.Tensor, graph_features: torch.Tensor, text_bank: torch.Tensor):
        if h.ndim != 3 or tuple(h.shape[1:]) != (4, 36):
            raise ValueError(f"CAM representation must be [N,4,36], got {tuple(h.shape)}")
        if tuple(graph_features.shape) != (h.shape[0], 320):
            raise ValueError(f"GFR representation must be [N,320], got {tuple(graph_features.shape)}")
        if text_bank.ndim != 2 or text_bank.shape[0] != 2:
            raise ValueError(f"Text bank must be [2,D] ordered [ASD,TD], got {tuple(text_bank.shape)}")
        q = self.query_norm(self.query_h(h) + self.query_f(graph_features).unsqueeze(1) + self.q_cls)
        k = self.key(text_bank)
        v = self.value(text_bank)
        attention = torch.softmax(torch.matmul(q, k.transpose(0, 1)) / math.sqrt(36.0), dim=-1)
        retrieved = torch.matmul(attention, v)
        gate = torch.sigmoid(self.gate(torch.cat((h, q, retrieved), dim=-1)))
        residual = self.gamma * gate * self.output(retrieved)
        return h + residual, {
            "query": q,
            "attention": attention,
            "gate": gate,
            "residual": residual,
            "gamma": self.gamma,
            "cam_rms": h.square().mean(dim=-1).sqrt(),
            "graph_rms": graph_features.square().mean(dim=-1).sqrt(),
        }


def install_output_hook(cam: nn.Module, module: GCDSI, text_bank: torch.Tensor) -> None:
    """Route the existing 4x36 CAM features through GCDSI before OutputLayer."""
    import types

    output_layer = cam.Outputlayer
    original_forward = output_layer.forward
    cam.gcdsi = module
    cam._gcdsi_text_bank = text_bank
    cam._gcdsi_graph_features = None
    cam._gcdsi_diagnostics = None

    def forward_with_gcdsi(self, x: torch.Tensor, attn_embedding: torch.Tensor):
        if getattr(cam, "_gcdsi_active", False):
            graph = cam._gcdsi_graph_features
            if graph is None:
                raise RuntimeError("GCDSI requires the current GFR output before CAM classification")
            h = x.reshape(x.shape[0], 4, 36)
            h_prime, diagnostic = module(h, graph, cam._gcdsi_text_bank)
            cam._gcdsi_diagnostics = diagnostic
            x = h_prime.reshape(x.shape[0], 144)
        else:
            cam._gcdsi_diagnostics = None
        return original_forward(x, attn_embedding)

    output_layer.forward = types.MethodType(forward_with_gcdsi, output_layer)
