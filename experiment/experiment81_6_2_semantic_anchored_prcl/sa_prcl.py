"""Semantic-anchored patient-conditioned pairwise relations for E81.6.2."""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


PROMPTS = [
    "Nromal brain volume",
    "Uniform cortical thickness",
    "Symmetrical brain structure",
    "Normal ventricular morphology",
    "Altered brain volume",
    "Abnormal functional connectivity",
    "White matter integrity issues",
    "Different activation patterns",
]
MEDICAL = ((0, 1, 2, 3), (4, 5, 6, 7))
SCRAMBLED = ((0, 1, 4, 5), (2, 3, 6, 7))


class SAPRCL(nn.Module):
    """Shared pairwise relation mapper with a frozen 8-D semantic geometry."""

    def __init__(self, semantic_basis: torch.Tensor, rho: float = 0.1, eps: float = 1e-12):
        super().__init__()
        if tuple(semantic_basis.shape) != (8, 8):
            raise ValueError(f"expected fixed semantic basis [8,8], got {tuple(semantic_basis.shape)}")
        self.rho, self.eps = float(rho), float(eps)
        self.local_proj = nn.Linear(9, 32)
        self.position = nn.Parameter(torch.empty(16, 32))
        self.local_norm = nn.LayerNorm(32)
        self.query_proj = nn.Linear(32, 32)
        self.query_norm = nn.LayerNorm(32)
        self.visual_key_proj = nn.Linear(64, 32)
        self.semantic_key_proj = nn.Linear(64, 8, bias=False)
        self.value_proj = nn.Linear(96, 32)
        self.null_key = nn.Parameter(torch.zeros(32))
        self.output = nn.Linear(32, 9, bias=False)
        self.register_buffer("semantic_basis", semantic_basis.detach().clone().float())
        self.register_buffer("null_value", torch.zeros(32))
        self.register_buffer(
            "neighbors",
            torch.tensor([[q for q in range(16) if q != p] for p in range(16)], dtype=torch.long),
        )
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.weight)

    @classmethod
    def initialized(cls, salt: int, device, semantic_basis: torch.Tensor, rho: float = 0.1):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(salt))
            module = cls(semantic_basis, rho=rho)
        return module.to(device)

    @staticmethod
    def global_pool(z_rel: torch.Tensor) -> torch.Tensor:
        n = z_rel.shape[0]
        return F.avg_pool1d(z_rel.reshape(n * 4, 1, 36), 4, 4).reshape(n, 4, 9)

    def forward(
        self,
        z_cam: torch.Tensor,
        *,
        grouping: str = "medical",
        eta: float = 0.0,
        intervention: str = "full",
        blocked_rows=(),
    ):
        if z_cam.ndim != 2 or z_cam.shape[1] != 144:
            raise ValueError(f"expected CAM144, got {tuple(z_cam.shape)}")
        if grouping not in ("medical", "scrambled"):
            raise ValueError(f"unknown prompt grouping: {grouping}")
        if intervention not in ("full", "prompt_off", "null_only", "global_off", "residual_off") and not intervention.startswith("delete_prompt_"):
            raise ValueError(f"unknown intervention: {intervention}")

        n = z_cam.shape[0]
        z_rel = z_cam.detach() + float(eta) * (z_cam - z_cam.detach())
        slots = z_rel.reshape(n, 4, 4, 9)
        local = slots.reshape(n, 16, 9)
        u = self.local_norm(self.local_proj(local) + self.position)

        receiver = u.unsqueeze(2).expand(n, 16, 15, 32)
        sender = u[:, self.neighbors, :]
        product = receiver * sender
        difference = sender - receiver
        pair_key_input = torch.cat((product, difference.abs()), dim=-1)
        pair_value_input = torch.cat((sender, product, difference), dim=-1)
        key_visual = self.visual_key_proj(pair_key_input)
        key_semantic = F.normalize(self.semantic_key_proj(pair_key_input), dim=-1)
        value = self.value_proj(pair_value_input)
        query = self.query_norm(self.query_proj(u))

        key_with_null = torch.cat(
            (key_visual, self.null_key.view(1, 1, 1, 32).expand(n, 16, 1, 32)), dim=2
        )
        value_with_null = torch.cat(
            (value, self.null_value.view(1, 1, 1, 32).expand(n, 16, 1, 32)), dim=2
        )
        score_visual = torch.einsum("npd,npqd->npq", query, key_with_null) / math.sqrt(32.0)
        score_semantic = torch.einsum("kd,npqd->nkpq", self.semantic_basis, key_semantic)
        if intervention == "prompt_off":
            score_semantic = torch.zeros_like(score_semantic)
        elif intervention.startswith("delete_prompt_"):
            index = int(intervention.rsplit("_", 1)[1])
            if index not in range(8):
                raise ValueError(f"invalid prompt index: {index}")
            score_semantic = score_semantic.clone()
            score_semantic[:, index] = 0.0

        score_prompt = torch.cat(
            (
                score_visual[:, None, :, :15] + score_semantic,
                score_visual[:, None, :, 15:].expand(n, 8, 16, 1),
            ),
            dim=-1,
        )
        if intervention == "null_only":
            score_visual = score_visual.clone()
            score_prompt = score_prompt.clone()
            score_visual[..., :15] = -torch.inf
            score_prompt[..., :15] = -torch.inf

        attention_visual = score_visual.softmax(dim=-1)
        attention_prompt = score_prompt.softmax(dim=-1)
        all_attention = torch.cat((attention_visual[:, None], attention_prompt), dim=1)
        all_messages = torch.einsum("nkpq,npqd->nkpd", all_attention, value_with_null)
        message_visual, messages = all_messages[:, 0], all_messages[:, 1:]
        delta_messages = messages - message_visual[:, None]

        evidence = torch.logsumexp(score_prompt[..., :15], dim=-1) - torch.logsumexp(
            score_visual[..., :15], dim=-1
        ).unsqueeze(1)
        if intervention == "null_only":
            evidence = torch.zeros_like(evidence)
        groups = MEDICAL if grouping == "medical" else SCRAMBLED
        normal_indices = list(groups[0])
        abnormal_indices = list(groups[1])
        beta_normal = evidence[:, normal_indices].softmax(dim=1)
        beta_abnormal = evidence[:, abnormal_indices].softmax(dim=1)
        message_normal = (beta_normal[..., None] * delta_messages[:, normal_indices]).sum(dim=1)
        message_abnormal = (beta_abnormal[..., None] * delta_messages[:, abnormal_indices]).sum(dim=1)
        relation_contrast = message_abnormal - message_normal

        d = self.output(relation_contrast).reshape(n, 4, 4, 9)
        global_features = self.global_pool(z_rel)
        matching_global = torch.zeros_like(global_features) if intervention == "global_off" else global_features
        matching = (d * matching_global.unsqueeze(2)).softmax(dim=-1)
        weighted = matching * d
        raw_delta = weighted.reshape(n, 144)
        reference_norm = z_cam.detach().norm(dim=-1, keepdim=True)
        bound = self.rho * reference_norm
        delta = bound * raw_delta / torch.sqrt(bound.square() + raw_delta.square().sum(-1, keepdim=True) + self.eps)

        legal = torch.ones(n, dtype=torch.bool, device=z_cam.device)
        if blocked_rows:
            legal[list(blocked_rows)] = False
            delta = delta * legal[:, None]
        if intervention == "residual_off" or intervention == "prompt_off" or intervention == "null_only":
            delta = delta * 0.0
        z_enhanced = z_cam + delta
        details = {
            "z_cam": z_cam,
            "z_rel": z_rel,
            "slots": slots,
            "local": local,
            "u": u,
            "global": global_features,
            "query": query,
            "key_visual": key_visual,
            "key_semantic": key_semantic,
            "value": value,
            "score_visual": score_visual,
            "score_semantic": score_semantic,
            "score_prompt": score_prompt,
            "attention_visual": attention_visual,
            "attention_prompt": attention_prompt,
            "message_visual": message_visual,
            "messages": messages,
            "delta_messages": delta_messages,
            "evidence": evidence,
            "beta_normal": beta_normal,
            "beta_abnormal": beta_abnormal,
            "message_normal": message_normal,
            "message_abnormal": message_abnormal,
            "relation_contrast": relation_contrast,
            "d": d,
            "matching": matching,
            "weighted": weighted,
            "raw_delta": raw_delta,
            "delta": delta,
            "raw_ratio": raw_delta.norm(dim=-1) / (reference_norm[:, 0] + self.eps),
            "delta_ratio": delta.norm(dim=-1) / (reference_norm[:, 0] + self.eps),
            "legal": legal,
            "z_enhanced": z_enhanced,
            "normal_indices": normal_indices,
            "abnormal_indices": abnormal_indices,
        }
        return z_enhanced, details


def residual_penalty(details, train_indices):
    ids = torch.as_tensor(train_indices, device=details["raw_delta"].device, dtype=torch.long)
    ids = ids[details["legal"][ids]]
    if not ids.numel():
        raise RuntimeError("no legal training patients for residual regularization")
    return details["raw_ratio"][ids].square().mean()
