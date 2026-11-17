"""E81.3 pairwise relational correction branches.

The module is deliberately independent of the historical DARG/CAM code.  It
receives CAM144 immediately before the original OutputLayer and returns an
exactly stable log-probability correction.  G1 and G2 share every parameter
shape and initialization; only the construction of sender features differs.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


GROUP_UNARY = "unary"
GROUP_PAIRWISE = "pairwise"


class PairwiseRelationalCorrection(nn.Module):
    """Patient-conditional, null-aware local relation correction."""

    def __init__(self, hidden: int = 32, heads: int = 8, eta: float = 0.1):
        super().__init__()
        if hidden != 32 or heads != 8:
            raise ValueError("E81.3 fixes hidden=32 and heads=8")
        self.hidden = hidden
        self.heads = heads
        self.eta = float(eta)

        self.local_proj = nn.Linear(9, hidden)
        self.local_norm = nn.LayerNorm(hidden)
        self.position = nn.Parameter(torch.empty(16, hidden))

        # Both groups use the same projection shapes.  Their input feature
        # construction is the only formal architectural difference.
        self.key_proj = nn.Linear(64, hidden)
        self.key_norm = nn.LayerNorm(hidden)
        self.value_proj = nn.Linear(96, hidden)
        self.query_proj = nn.Linear(hidden, hidden)
        self.query_norm = nn.LayerNorm(hidden)

        self.null_key = nn.Parameter(torch.zeros(hidden))
        self.register_buffer("null_value", torch.zeros(hidden))

        # Patient-conditioned local readout: one score per receiving local.
        self.readout_hidden = nn.Linear(64, hidden)
        self.readout_score = nn.Linear(hidden, 1)

        # Four detached summary statistics accompany the 32-D relation summary:
        # signed margin, absolute margin, mean null mass, normalized entropy.
        self.gate = nn.Sequential(
            nn.Linear(36, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )
        self.correction_head = nn.Linear(hidden, 1, bias=False)

        # A fixed head identity gives the eight query heads distinct directions
        # without introducing another trainable or random operation.
        head_identity = torch.zeros(heads, hidden)
        head_identity[:, :heads] = torch.eye(heads)
        self.register_buffer("head_identity", head_identity)

        receivers, senders = [], []
        candidate_map = torch.full((16, 16), -1, dtype=torch.long)
        for p in range(16):
            slot = 0
            for q in range(16):
                if q == p:
                    continue
                receivers.append(p)
                senders.append(q)
                candidate_map[p, q] = slot
                slot += 1
        self.register_buffer("receiver_index", torch.tensor(receivers, dtype=torch.long))
        self.register_buffer("sender_index", torch.tensor(senders, dtype=torch.long))
        self.register_buffer("candidate_map", candidate_map)
        self.register_buffer("undirected_pairs", torch.triu_indices(16, 16, offset=1))

        nn.init.normal_(self.position, mean=0.0, std=0.02)
        nn.init.zeros_(self.correction_head.weight)
        # Keep the initial learned gate conservative without suppressing the
        # branch by a hand-coded uncertainty rule.
        nn.init.constant_(self.gate[2].bias, -1.0)

    @classmethod
    def initialized(cls, seed: int, device, eta: float = 0.1):
        # fork_rng isolates all module initialization from the historical
        # public PyTorch RNG stream.  The device is intentionally not included
        # because initialization happens on a private CPU stream first.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            module = cls(eta=eta)
        return module.to(device)

    @staticmethod
    def _safe_entropy(attention: torch.Tensor) -> torch.Tensor:
        entropy = -(attention * attention.clamp_min(1e-12).log()).sum(dim=-1)
        return entropy.mean(dim=(1, 2)) / math.log(16.0)

    def _construct_features(self, receiver: torch.Tensor, sender: torch.Tensor,
                            group: str):
        if group == GROUP_UNARY:
            sender_tanh = torch.tanh(sender)
            key_features = torch.cat((sender, sender_tanh), dim=-1)
            value_features = torch.cat((sender, sender_tanh, sender.square()), dim=-1)
        elif group == GROUP_PAIRWISE:
            difference = sender - receiver
            product = receiver * sender
            key_features = torch.cat((product, difference), dim=-1)
            value_features = torch.cat((sender, product, difference), dim=-1)
        else:
            raise ValueError(f"unknown relation group {group!r}")
        return key_features, value_features

    def forward(self, z_cam: torch.Tensor, logp_anchor: torch.Tensor, group: str,
                training: bool, overlap_rows: Iterable[int] = (),
                intervention: str = "full", eta: float | None = None):
        if z_cam.ndim != 2 or tuple(z_cam.shape[1:]) != (144,):
            raise ValueError(f"expected CAM144 [N,144], got {tuple(z_cam.shape)}")
        if tuple(logp_anchor.shape) != (z_cam.shape[0], 2):
            raise ValueError("logp anchor must have shape [N,2]")
        if group not in (GROUP_UNARY, GROUP_PAIRWISE):
            raise ValueError(f"unknown relation group {group!r}")

        n = z_cam.shape[0]
        eta_used = self.eta if eta is None else float(eta)
        # Exact forward identity with a controlled backward scale eta_used.
        z_rel = z_cam.detach() + eta_used * (z_cam - z_cam.detach())
        tokens = self.local_norm(
            self.local_proj(z_rel.reshape(n, 16, 9)) + self.position.unsqueeze(0)
        )
        receiver = tokens[:, self.receiver_index].reshape(n, 16, 15, self.hidden)
        sender = tokens[:, self.sender_index].reshape(n, 16, 15, self.hidden)
        key_features, value_features = self._construct_features(receiver, sender, group)
        keys = self.key_norm(self.key_proj(key_features))
        values = self.value_proj(value_features)

        query = self.query_proj(tokens)[:, None, :, :] + self.head_identity[None, :, None, :]
        query = self.query_norm(query)
        scores = torch.einsum("n k p d, n p q d -> n k p q", query, keys) / math.sqrt(self.hidden)
        null_score = torch.einsum("n k p d, d -> n k p", query, self.null_key) / math.sqrt(self.hidden)
        attention = torch.softmax(torch.cat((scores, null_score.unsqueeze(-1)), dim=-1), dim=-1)
        if intervention == "null_only":
            attention = torch.zeros_like(attention)
            attention[..., -1] = 1.0
        elif intervention != "full":
            raise ValueError(f"unknown relation intervention {intervention!r}")

        messages = torch.einsum("n k p q, n p q d -> n k p d", attention[..., :15], values)
        m_local = messages.mean(dim=1)
        readout_input = torch.cat((tokens, m_local), dim=-1)
        readout_score = self.readout_score(torch.tanh(self.readout_hidden(readout_input))).squeeze(-1)
        beta = torch.softmax(readout_score, dim=-1)
        relation_summary = torch.sum(beta.unsqueeze(-1) * m_local, dim=1)

        null_weight = attention[..., -1].mean(dim=(1, 2))
        nonnull_mass = 1.0 - null_weight
        entropy = self._safe_entropy(attention)
        relation_statistics = torch.stack((nonnull_mass, entropy), dim=-1)

        base_margin_signed = (logp_anchor[:, 1] - logp_anchor[:, 0]).detach()
        base_margin_abs = base_margin_signed.abs()
        gate_features = torch.cat((
            relation_summary,
            base_margin_signed.unsqueeze(-1),
            base_margin_abs.unsqueeze(-1),
            null_weight.unsqueeze(-1),
            entropy.unsqueeze(-1),
        ), dim=-1)
        gate = self.gate(gate_features).squeeze(-1)

        raw_margin = self.correction_head(relation_summary).squeeze(-1)
        delta_margin = torch.tanh(raw_margin) * gate
        if training and overlap_rows:
            row_index = torch.as_tensor(tuple(overlap_rows), dtype=torch.long, device=z_cam.device)
            delta_margin = delta_margin.index_fill(0, row_index, 0.0)

        base_td_margin = logp_anchor[:, 1] - logp_anchor[:, 0]
        final_td_margin = base_td_margin + delta_margin
        # Stable and exactly identity-preserving log-probability correction.
        delta_logp_asd = F.softplus(base_td_margin) - F.softplus(final_td_margin)
        delta_logp_td = F.softplus(-base_td_margin) - F.softplus(-final_td_margin)
        logp_final = logp_anchor + torch.stack((delta_logp_asd, delta_logp_td), dim=-1)

        return logp_final, {
            "tokens": tokens,
            "z_rel": z_rel,
            "query": query,
            "keys": keys,
            "values": values,
            "attention": attention,
            "messages": messages,
            "m_local": m_local,
            "readout_score": readout_score,
            "beta": beta,
            "relation_summary": relation_summary,
            "relation_statistics": relation_statistics,
            "null_weight": null_weight,
            "nonnull_mass": nonnull_mass,
            "entropy": entropy,
            "gate_features": gate_features,
            "gate": gate,
            "raw_margin": raw_margin,
            "delta_margin": delta_margin,
            "logp_anchor": logp_anchor,
            "logp_final": logp_final,
            "relation_mode": group,
            "intervention": intervention,
            "eta": eta_used,
        }
