"""Self-local and cross-local CAM144-to-logit correction branches."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class LocalCorrection(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.hidden = hidden
        self.local_proj = nn.Linear(9, hidden)
        self.local_norm = nn.LayerNorm(hidden)
        self.position = nn.Parameter(torch.empty(16, hidden))
        self.key_proj = nn.Linear(64, hidden)
        self.key_norm = nn.LayerNorm(hidden)
        self.value_proj = nn.Linear(96, hidden)
        self.query_proj = nn.Linear(hidden, hidden)
        self.query_norm = nn.LayerNorm(hidden)
        self.null_key = nn.Parameter(torch.zeros(hidden))
        self.gate = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.correction_head = nn.Linear(hidden, 1, bias=False)

        head_identity = torch.zeros(8, hidden)
        head_identity[:, :8] = torch.eye(8)
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

    @classmethod
    def initialized(cls, seed: int, device):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            module = cls()
        return module.to(device)

    def forward(self, z_cam: torch.Tensor, logp_base: torch.Tensor, group: str,
                training: bool, overlap_rows=(), intervention: str = "full"):
        if z_cam.ndim != 2 or tuple(z_cam.shape[1:]) != (144,):
            raise ValueError(f"expected CAM144 [N,144], got {tuple(z_cam.shape)}")
        if group not in ("self", "cross"):
            raise ValueError(f"unknown relation mode: {group}")
        n = z_cam.shape[0]
        tokens = self.local_norm(
            self.local_proj(z_cam.reshape(n, 16, 9)) + self.position.unsqueeze(0)
        )
        receiver = tokens[:, self.receiver_index].reshape(n, 16, 15, self.hidden)
        if group == "self":
            sender = receiver
        else:
            sender = tokens[:, self.sender_index].reshape(n, 16, 15, self.hidden)

        difference = sender - receiver
        key_features = torch.cat((receiver * sender, difference), dim=-1)
        keys = self.key_norm(self.key_proj(key_features))
        value_features = torch.cat((sender, receiver * sender, difference), dim=-1)
        values = self.value_proj(value_features)

        query = self.query_proj(tokens)[:, None, :, :] + self.head_identity[None, :, None, :]
        query = self.query_norm(query)
        scores = torch.einsum("nkpd,npqd->nkpq", query, keys) / math.sqrt(self.hidden)
        null_score = torch.einsum("nkpd,d->nkp", query, self.null_key) / math.sqrt(self.hidden)
        attention = torch.softmax(torch.cat((scores, null_score.unsqueeze(-1)), dim=-1), dim=-1)
        if intervention == "null_only":
            attention = torch.zeros_like(attention)
            attention[..., -1] = 1.0
        messages = torch.einsum("nkpq,npqd->nkpd", attention[..., :15], values)

        relation_summary = messages.mean(dim=(1, 2))
        nonnull_mass = attention[..., :15].sum(dim=-1).mean(dim=(1, 2))
        entropy = -(attention * attention.clamp_min(1e-12).log()).sum(dim=-1).mean(dim=(1, 2))
        entropy = entropy / math.log(16.0)
        relation_stats = torch.stack((nonnull_mass, entropy), dim=-1)
        g_rel = torch.sigmoid(self.gate(torch.cat((relation_summary, relation_stats), dim=-1))).squeeze(-1)

        base_margin = (logp_base[:, 1] - logp_base[:, 0]).abs()
        g_unc = torch.sigmoid((0.5 - base_margin.detach()) / 0.3)
        raw_margin = self.correction_head(relation_summary).squeeze(-1)
        delta_margin = 0.75 * torch.tanh(raw_margin) * g_unc * g_rel
        if training and overlap_rows:
            row_index = torch.as_tensor(overlap_rows, dtype=torch.long, device=z_cam.device)
            delta_margin = delta_margin.index_fill(0, row_index, 0.0)

        base_td_margin = logp_base[:, 1] - logp_base[:, 0]
        final_td_margin = base_td_margin + delta_margin
        delta_logp_asd = F.softplus(base_td_margin) - F.softplus(final_td_margin)
        delta_logp_td = F.softplus(-base_td_margin) - F.softplus(-final_td_margin)
        logp_final = logp_base + torch.stack((delta_logp_asd, delta_logp_td), dim=-1)
        return logp_final, {
            "tokens": tokens,
            "query": query,
            "keys": keys,
            "values": values,
            "attention": attention,
            "messages": messages,
            "relation_summary": relation_summary,
            "relation_statistics": relation_stats,
            "g_unc": g_unc,
            "g_rel": g_rel,
            "raw_margin": raw_margin,
            "delta_margin": delta_margin,
            "logp_base": logp_base,
            "logp_final": logp_final,
            "relation_mode": group,
            "intervention": intervention,
        }
