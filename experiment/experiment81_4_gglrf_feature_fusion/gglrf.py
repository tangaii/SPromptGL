"""E81.4: global-guided local relational feature fusion, independent of DARG.

Only Key/Value feature construction differs between the two formal groups.
All 16 latent CAM coordinate slices survive the residual write-back.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

GROUP_UNARY = "unary"
GROUP_PAIRWISE = "pairwise"


class GGLRF(nn.Module):
    def __init__(self, eta=0.1, rho=0.10, eps=1e-12):
        super().__init__()
        self.eta, self.rho, self.eps = float(eta), float(rho), float(eps)
        self.global_proj = nn.Linear(144, 32)
        self.global_norm = nn.LayerNorm(32)
        self.local_proj = nn.Linear(9, 32)
        self.local_norm = nn.LayerNorm(32)
        self.position = nn.Parameter(torch.empty(16, 32))
        self.query_proj = nn.Linear(64, 32)
        self.query_norm = nn.LayerNorm(32)
        self.key_proj = nn.Linear(64, 32)
        self.value_proj = nn.Linear(96, 32)
        self.null_key = nn.Parameter(torch.zeros(32))
        self.register_buffer("null_value", torch.zeros(32))
        self.read_key = nn.Linear(64, 32)
        self.read_query = nn.Linear(32, 32)
        self.output = nn.Linear(32, 9, bias=False)
        # Same deterministic eight head directions as the audited E81.3;
        # each 32-D head uses Q_p + its fixed one-hot direction.
        head_identity = torch.zeros(8, 32)
        head_identity[:, :8] = torch.eye(8)
        self.register_buffer("head_identity", head_identity)
        self.register_buffer("neighbors", torch.tensor(
            [[q for q in range(16) if q != p] for p in range(16)], dtype=torch.long))
        nn.init.normal_(self.position, 0.0, 0.02)
        nn.init.zeros_(self.output.weight)

    @classmethod
    def initialized(cls, salt, device, eta=0.1, rho=0.10):
        # CPU-only private stream; leaves public CPU and device streams intact.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(salt))
            module = cls(eta=eta, rho=rho)
        return module.to(device)

    @staticmethod
    def construct_features(receiver, sender, group):
        if group == GROUP_UNARY:
            nonlinear = sender.tanh()
            return (torch.cat((sender, nonlinear), -1),
                    torch.cat((sender, nonlinear, sender.square()), -1))
        if group == GROUP_PAIRWISE:
            product, difference = receiver * sender, sender - receiver
            return (torch.cat((product, difference), -1),
                    torch.cat((sender, product, difference), -1))
        raise ValueError(group)

    def encode(self, z_rel):
        h = z_rel.reshape(-1, 16, 9)
        u = self.local_norm(self.local_proj(h) + self.position)
        g = F.gelu(self.global_norm(self.global_proj(z_rel)))
        return h, u, g

    def attend(self, u, g, group, null_only=False):
        n = u.shape[0]
        receiver = u.unsqueeze(2).expand(n, 16, 15, 32)
        sender = u[:, self.neighbors, :]
        key_input, value_input = self.construct_features(receiver, sender, group)
        key, value = self.key_proj(key_input), self.value_proj(value_input)
        q_base = self.query_norm(self.query_proj(torch.cat(
            (u, g.unsqueeze(1).expand(n, 16, 32)), -1)))
        query = q_base.unsqueeze(1) + self.head_identity[None, :, None, :]
        all_keys = torch.cat((key, self.null_key.view(1, 1, 1, 32).expand(n, 16, 1, 32)), 2)
        all_values = torch.cat((value, self.null_value.view(1, 1, 1, 32).expand(n, 16, 1, 32)), 2)
        scores = torch.einsum("nhpd,npqd->nhpq", query, all_keys) / math.sqrt(32.0)
        if null_only:
            scores = scores.clone()
            scores[..., :15] = -torch.inf
        attention = scores.softmax(-1)
        messages = torch.einsum("nhpq,npqd->nhpd", attention, all_values)
        return {
            "key": key, "value": value, "query": query, "attention": attention,
            "messages": messages, "m_local": messages.mean(1),
        }

    def readout(self, u, m_local, g):
        k_read = self.read_key(torch.cat((u, m_local), -1))
        q_read = self.read_query(g)
        score = (q_read.unsqueeze(1) * k_read).sum(-1) / math.sqrt(32.0)
        beta = score.softmax(1)
        selected = (16.0 * beta.unsqueeze(-1)) * m_local
        raw_local = self.output(selected)
        return {"k_read": k_read, "q_read": q_read, "beta": beta,
                "m_selected": selected, "raw_delta_local": raw_local,
                "raw_delta": raw_local.reshape(-1, 144)}

    def forward(self, z_cam, group, intervention="full", blocked_rows=(), eta=None):
        if z_cam.ndim != 2 or z_cam.shape[1] != 144:
            raise ValueError(f"expected CAM144, got {tuple(z_cam.shape)}")
        route_eta = self.eta if eta is None else float(eta)
        z_rel = z_cam.detach() + route_eta * (z_cam - z_cam.detach())
        h, u, g = self.encode(z_rel)
        selection_g = torch.zeros_like(g) if intervention == "global_off" else g
        feature_mode = GROUP_UNARY if intervention == "unary" else group
        details = self.attend(u, selection_g, feature_mode, null_only=intervention == "null")
        details.update(self.readout(u, details["m_local"], selection_g))
        raw = details["raw_delta"]
        reference_norm = torch.linalg.vector_norm(z_cam.detach(), dim=-1, keepdim=True)
        bound = self.rho * reference_norm
        delta = bound * raw / torch.sqrt(bound.square() + raw.square().sum(-1, keepdim=True) + self.eps)
        if intervention == "off":
            delta = delta * 0.0
        legal = torch.ones(z_cam.shape[0], dtype=torch.bool, device=z_cam.device)
        if blocked_rows:
            legal[list(blocked_rows)] = False
            delta = delta * legal.unsqueeze(-1)
        enhanced = z_cam + delta
        details.update({"h": h, "u": u, "g": g, "z_rel": z_rel, "z_cam": z_cam,
                        "bound": bound, "reference_norm": reference_norm, "delta": delta,
                        "z_enhanced": enhanced, "legal_rows": legal,
                        "delta_ratio": torch.linalg.vector_norm(delta, dim=-1) /
                        (reference_norm.squeeze(-1) + self.eps)})
        return enhanced, details


def residual_penalty(details, train_indices):
    indices = torch.as_tensor(train_indices, dtype=torch.long, device=details["delta"].device)
    indices = indices[details["legal_rows"][indices]]
    if indices.numel() == 0:
        raise RuntimeError("no legal training rows for residual regularizer")
    return details["delta_ratio"][indices].square().mean()
