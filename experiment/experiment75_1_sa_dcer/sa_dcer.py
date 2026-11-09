"""Semantic-anchored, signed disease evidence refinement at the CAM feature boundary."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SADecer(nn.Module):
    def __init__(self, text_bank: torch.Tensor, projection: torch.Tensor):
        super().__init__()
        if tuple(text_bank.shape) != (8, 1024):
            raise ValueError(f"text_bank must be [8,1024], got {tuple(text_bank.shape)}")
        if tuple(projection.shape) != (1024, 128):
            raise ValueError(f"projection must be [1024,128], got {tuple(projection.shape)}")
        self.register_buffer("text_bank", text_bank.detach().float().clone())
        self.register_buffer("projection", projection.detach().float().clone())
        self.context_proj = nn.Linear(36, 128)
        self.key_proj = nn.Linear(9, 128, bias=False)
        self.evidence_proj = nn.Linear(18, 9, bias=False)
        self.confidence_net = nn.Sequential(
            nn.Linear(46, 32), nn.GELU(), nn.Linear(32, 1)
        )
        self.gamma_raw = nn.Parameter(torch.tensor(-0.6931471805599453))

    @property
    def gamma(self) -> torch.Tensor:
        return 0.15 * torch.sigmoid(self.gamma_raw)

    @staticmethod
    def global_context(h: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        return F.avg_pool1d(h.reshape(n * 4, 1, 36), kernel_size=4, stride=4).reshape(n, 4, 9).reshape(n, 36)

    def _semantic_bank(self, intervention: str) -> torch.Tensor:
        bank = self.text_bank
        if intervention == "prompt_class_swap":
            return torch.cat((bank[4:8], bank[0:4]), dim=0)
        if intervention == "prompt_within_class_shuffle":
            return torch.cat((bank[2:4], bank[0:2], bank[6:8], bank[4:6]), dim=0)
        return bank

    def contrastive_scores(self, tokens: torch.Tensor, context: torch.Tensor, *, intervention: str = "full") -> dict[str, torch.Tensor]:
        text = F.normalize(self._semantic_bank(intervention) @ self.projection, dim=-1)
        query = F.normalize(text.unsqueeze(0) + 0.1 * torch.tanh(self.context_proj(context)).unsqueeze(1), dim=-1)
        key = F.normalize(self.key_proj(tokens), dim=-1)
        scores = torch.einsum("nkd,nqd->nkq", key, query) / 0.2
        score_td = torch.logsumexp(scores[..., :4], dim=-1) - torch.log(scores.new_tensor(4.0))
        score_asd = torch.logsumexp(scores[..., 4:], dim=-1) - torch.log(scores.new_tensor(4.0))
        difference = torch.tanh(0.5 * (score_asd - score_td))
        prompt_weights = torch.cat((torch.softmax(scores[..., :4], dim=-1), torch.softmax(scores[..., 4:], dim=-1)), dim=-1)
        return {"query": query, "key": key, "scores": scores, "score_asd": score_asd,
                "score_td": score_td, "difference": difference, "prompt_weights": prompt_weights}

    def forward(self, h: torch.Tensor, *, group: str, intervention: str = "full") -> dict[str, torch.Tensor]:
        if tuple(h.shape[1:]) != (144,):
            raise ValueError(f"SA-DCER expects [N,144], got {tuple(h.shape)}")
        if intervention not in {"full", "dcer_off", "confidence_neutral", "prompt_class_swap", "prompt_within_class_shuffle"}:
            raise ValueError(f"unknown SA-DCER intervention: {intervention}")
        tokens = h.reshape(h.shape[0], 16, 9)
        context = self.global_context(h)
        confidence_enabled = group in ("G2-Faithful", "G3-Anonymous") and intervention != "confidence_neutral"
        if intervention == "dcer_off":
            zeros = torch.zeros_like(tokens)
            return {
                "h_prime": h, "tokens": tokens, "global_context": context,
                "difference": zeros[..., 0], "scores": h.new_zeros((h.shape[0], 16, 8)),
                "prompt_weights": h.new_zeros((h.shape[0], 16, 8)),
                "confidence": h.new_zeros((h.shape[0], 16)), "confidence_factor": h.new_ones((h.shape[0], 16)),
                "raw_residual": zeros, "bounded_residual": zeros, "residual": h.new_zeros(h.shape),
                "gamma": self.gamma.detach() * 0.0,
            }

        contrast = self.contrastive_scores(tokens, context, intervention=intervention)
        query, key, scores = contrast["query"], contrast["key"], contrast["scores"]
        score_asd, score_td, difference = contrast["score_asd"], contrast["score_td"], contrast["difference"]
        evidence_asd = F.relu(difference).unsqueeze(-1) * tokens
        evidence_td = F.relu(-difference).unsqueeze(-1) * tokens
        raw_residual = self.evidence_proj(torch.cat((evidence_asd, evidence_td), dim=-1))
        confidence_input = torch.cat((tokens, context[:, None, :].expand(-1, 16, -1), difference.unsqueeze(-1)), dim=-1)
        confidence = torch.sigmoid(self.confidence_net(confidence_input)).squeeze(-1)
        confidence_factor = 0.75 + 0.5 * confidence if confidence_enabled else torch.ones_like(confidence)
        raw_gated = raw_residual * confidence_factor.unsqueeze(-1)
        rms = tokens.square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        bounded = rms * torch.tanh(raw_gated / rms)
        residual = self.gamma * bounded
        h_prime = h + residual.reshape_as(h)
        prompt_weights = contrast["prompt_weights"]
        return {
            "h_prime": h_prime, "tokens": tokens, "global_context": context,
            "query": query, "key": key, "scores": scores, "prompt_weights": prompt_weights,
            "difference": difference, "score_asd": score_asd, "score_td": score_td,
            "confidence": confidence, "confidence_factor": confidence_factor,
            "raw_residual": raw_residual, "bounded_residual": bounded,
            "residual": residual.reshape_as(h), "gamma": self.gamma,
        }


def signed_margin_teacher(output_layer: nn.Module, h: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Training-index-only signed sensitivity to halving each CAM token."""
    with torch.no_grad():
        selected = h.index_select(0, indices)
        tokens = selected.reshape(selected.shape[0], 16, 9)
        n = tokens.shape[0]
        baseline = _deterministic_logits(output_layer, selected)
        base_margin = baseline[:, 0] - baseline[:, 1]
        factors = torch.ones((16, 16), dtype=h.dtype, device=h.device)
        factors.diagonal().fill_(0.5)
        attenuated = (tokens[:, None, :, :] * factors[None, :, :, None]).reshape(n * 16, 144)
        margins = _deterministic_logits(output_layer, attenuated).reshape(n, 16, 2)
        delta = base_margin[:, None] - (margins[..., 0] - margins[..., 1])
        scale = torch.quantile(delta.abs(), 0.75, dim=1, keepdim=True).clamp_min(0.05)
        return torch.tanh(delta / scale).detach()


def _deterministic_logits(output_layer: nn.Module, h: torch.Tensor) -> torch.Tensor:
    z = output_layer.mlp_head[0](h)
    z = output_layer.mlp_head[1](z)
    z = output_layer.mlp_head[2](z)
    return output_layer.classifier(z)

