"""Prompt-conditioned patient-internal local relation composition."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class PRCL(nn.Module):
    def __init__(self, anchors: torch.Tensor, use_text: bool, hidden: int = 32):
        super().__init__()
        if tuple(anchors.shape) != (8, hidden):
            raise ValueError(f"expected 8 frozen {hidden}-D anchors, got {tuple(anchors.shape)}")
        self.hidden = int(hidden)
        self.use_text = bool(use_text)
        self.local_proj = nn.Linear(9, hidden)
        self.local_norm = nn.LayerNorm(hidden)
        self.position = nn.Parameter(torch.empty(16, hidden))
        self.key_proj = nn.Linear(64, hidden)
        self.key_norm = nn.LayerNorm(hidden)
        self.value_proj = nn.Linear(96, hidden)
        self.query_proj = nn.Linear(hidden, hidden)
        self.query_norm = nn.LayerNorm(hidden)
        self.null_key = nn.Parameter(torch.zeros(hidden))
        self.fusion = nn.Linear(3 * hidden, hidden)
        self.output = nn.Linear(hidden, 9)
        self.alpha_gate = nn.Linear(144, 1)
        self.gamma_raw = nn.Parameter(torch.tensor(math.log(0.4 / 0.6)))

        head_identity = torch.zeros(8, hidden)
        head_identity[:, :8] = torch.eye(8)
        self.register_buffer("head_identity", head_identity, persistent=True)
        self.register_buffer("text_anchors", anchors.detach().float().clone(), persistent=True)
        src, dst = [], []
        for p in range(16):
            for q in range(16):
                if p != q:
                    src.append(p)
                    dst.append(q)
        candidate_index_map = torch.full((16, 16), -1, dtype=torch.long)
        for p in range(16):
            candidate_index_map[p, torch.arange(16) != p] = torch.arange(15)
        self.register_buffer("edge_src", torch.tensor(src, dtype=torch.long), persistent=True)
        self.register_buffer("edge_dst", torch.tensor(dst, dtype=torch.long), persistent=True)
        self.register_buffer("candidate_index_map", candidate_index_map, persistent=True)
        self.register_buffer("undirected_pairs", torch.triu_indices(16, 16, offset=1), persistent=True)

        nn.init.normal_(self.position, mean=0.0, std=0.02)
        nn.init.zeros_(self.alpha_gate.weight)
        nn.init.zeros_(self.alpha_gate.bias)

    @property
    def gamma(self):
        return 0.5 * torch.sigmoid(self.gamma_raw)

    def forward(self, z_cam: torch.Tensor, intervention: str = "full",
                text_override: torch.Tensor | None = None):
        if z_cam.ndim != 2 or z_cam.shape[1] != 144:
            raise ValueError(f"PRCL expects CAM144 [N,144], got {tuple(z_cam.shape)}")
        n = z_cam.shape[0]
        # These are latent CAM coordinates, not anatomical parcels.
        h = z_cam.reshape(n, 4, 4, 9).reshape(n, 16, 9)
        u = self.local_norm(self.local_proj(h) + self.position.unsqueeze(0))

        up = u[:, self.edge_src, :]
        uq = u[:, self.edge_dst, :]
        key_features = torch.cat((up * uq, (up - uq).abs()), dim=-1)
        key = self.key_norm(self.key_proj(key_features)).reshape(n, 16, 15, self.hidden)
        value_features = torch.cat((uq, up * uq, uq - up), dim=-1)
        value = self.value_proj(value_features).reshape(n, 16, 15, self.hidden)

        visual_query = self.query_proj(u)
        anchors = self.text_anchors if text_override is None else text_override
        if tuple(anchors.shape) != (8, self.hidden):
            raise ValueError(f"prompt override must have shape [8,{self.hidden}]")
        if not self.use_text or intervention in ("text_off", "nonsemantic"):
            text_term = torch.zeros_like(anchors)
        else:
            text_term = anchors.to(device=z_cam.device, dtype=z_cam.dtype)
        if intervention == "swap_groups":
            text_term = torch.cat((text_term[4:], text_term[:4]), dim=0)
        gamma = self.gamma * 0.0 if intervention == "gamma_zero" else self.gamma
        if intervention == "within_group_permute":
            order = torch.tensor([2, 0, 3, 1, 6, 4, 7, 5], device=text_term.device)
            text_term = text_term[order]
            head_identity = self.head_identity[order]
        else:
            head_identity = self.head_identity
        query = self.query_norm(
            visual_query[:, None, :, :]
            + head_identity[None, :, None, :]
            + gamma * math.sqrt(self.hidden) * text_term[None, :, None, :]
        )
        scores = torch.einsum("nkp d,npq d->nkpq", query, key) / math.sqrt(self.hidden)
        null_score = torch.einsum("nkpd,d->nkp", query, self.null_key) / math.sqrt(self.hidden)
        logits = torch.cat((scores, null_score.unsqueeze(-1)), dim=-1)
        weights = torch.softmax(logits, dim=-1)
        if intervention == "uniform":
            weights = torch.full_like(weights, 1.0 / 16.0)
        elif intervention == "null_only":
            weights = torch.zeros_like(weights)
            weights[..., -1] = 1.0
        messages = torch.einsum("nkpq,npqd->nkpd", weights[..., :15], value)

        normal_message = messages[:, :4].mean(dim=1)
        abnormal_message = messages[:, 4:].mean(dim=1)
        if intervention == "swap_groups":
            normal_message, abnormal_message = abnormal_message, normal_message
        relation_message = self.fusion(torch.cat((
            normal_message, abnormal_message, abnormal_message - normal_message
        ), dim=-1))
        delta_h = self.output(relation_message).reshape(n, 144)
        raw_delta = torch.tanh(delta_h)
        alpha = 0.1 * torch.sigmoid(self.alpha_gate(z_cam))
        unbounded_delta = alpha * raw_delta
        z_norm = torch.linalg.vector_norm(z_cam, dim=-1, keepdim=True)
        delta_norm = torch.linalg.vector_norm(unbounded_delta, dim=-1, keepdim=True)
        scale = (0.1 * z_norm / (delta_norm + 1e-12)).clamp(max=1.0)
        delta = unbounded_delta * scale
        if intervention in ("prcl_off", "writeback_off"):
            delta = torch.zeros_like(delta)
        z_prcl = z_cam + delta

        details = {
            "tokens": u,
            "query": query,
            "key": key,
            "value": value,
            "attention": weights,
            "candidate_map": self.candidate_index_map,
            "messages": messages,
            "normal_message": normal_message,
            "abnormal_message": abnormal_message,
            "relation_message": relation_message,
            "alpha": alpha,
            "gamma": gamma,
            "text_term_norm": text_term.norm(dim=-1),
            "visual_query_norm": visual_query.norm(dim=-1).mean(dim=-1),
            "raw_delta": raw_delta,
            "delta": delta,
            "delta_ratio": torch.linalg.vector_norm(delta, dim=-1) / (z_norm.squeeze(-1) + 1e-12),
            "cap_scale": scale.squeeze(-1),
            "intervention": intervention,
        }
        return z_prcl, details


def initialize(anchors: torch.Tensor, use_text: bool, seed: int, device):
    """Initialize all trainable PRCL values without consuming the public RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        module = PRCL(anchors.cpu(), use_text=use_text)
    return module.to(device)


def residual_penalty(delta: torch.Tensor, z_cam: torch.Tensor, indices: torch.Tensor):
    ids = indices.to(device=delta.device, dtype=torch.long)
    numerator = delta[ids].square().sum(dim=-1)
    denominator = z_cam[ids].square().sum(dim=-1).clamp_min(1e-12)
    return (numerator / denominator).mean()

