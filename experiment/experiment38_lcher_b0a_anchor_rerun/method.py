"""LCHER hypothesis competition and edge-index-constrained patient evidence routing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def build_td_reference(x: np.ndarray, y: np.ndarray, train: np.ndarray, fold: int,
                       edge_indices: np.ndarray, alpha: float = 10.0,
                       sigma_floor: float = 1e-4, z_clip: float = 3.0) -> dict:
    """Train-fold-only TD conditional reference; train TD gets deterministic OOF predictions."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    train = np.asarray(train, dtype=np.int64)
    td = train[y[train] == 1]
    assert len(td) >= 5
    corr = x[:, 64 + np.asarray(edge_indices, dtype=np.int64)]
    # PHENO is [age one-hot 26, site one-hot 20, sex one-hot 2]. Site is a nuisance covariate.
    cov = np.concatenate([x[:, :26], x[:, 26:46], x[:, 46:48]], axis=1)
    mu = np.zeros((len(x), len(edge_indices)), dtype=np.float64)
    residual_oof = np.zeros((len(td), len(edge_indices)), dtype=np.float64)
    kf = KFold(n_splits=5, shuffle=True, random_state=38000 + int(fold))
    for fit_pos, held_pos in kf.split(td):
        fit_ids, held_ids = td[fit_pos], td[held_pos]
        model = Ridge(alpha=alpha, fit_intercept=True, solver="svd")
        model.fit(cov[fit_ids], corr[fit_ids])
        pred = model.predict(cov[held_ids])
        mu[held_ids] = pred
        residual_oof[held_pos] = corr[held_ids] - pred
    # Robust fold-level scale comes only from OOF outer-train TD residuals.
    center = np.median(residual_oof, axis=0)
    sigma = 1.4826 * np.median(np.abs(residual_oof - center), axis=0)
    sigma = np.maximum(sigma, sigma_floor)
    full = Ridge(alpha=alpha, fit_intercept=True, solver="svd")
    full.fit(cov[td], corr[td])
    non_td_train = train[y[train] == 0]
    heldout = np.setdiff1d(np.arange(len(x)), td, assume_unique=False)
    mu[non_td_train] = full.predict(cov[non_td_train])
    mu[heldout] = full.predict(cov[heldout])
    z = np.clip((corr - mu) / sigma, -z_clip, z_clip).astype(np.float32)
    assert np.isfinite(z).all() and np.isfinite(mu).all() and np.isfinite(sigma).all()
    return {
        "mu": mu.astype(np.float32), "sigma": sigma.astype(np.float32), "z": z,
        "train_td": td, "train": train, "fold": int(fold),
        "oof_residual_rms": float(np.sqrt(np.mean(residual_oof ** 2))),
        "sigma_min": float(sigma.min()), "sigma_median": float(np.median(sigma)),
        "fit_subjects": td.tolist(),
    }


class LCHER(nn.Module):
    """One shared, small competitive router; each hypothesis sees only its mapped FC indices."""

    def __init__(self, text_embeddings: torch.Tensor, edge_to_hypothesis: list[list[int]],
                 direction: torch.Tensor, use_aux: bool = True, use_rival: bool = True,
                 top_k: int = 8, seed: int = 0):
        super().__init__()
        assert text_embeddings.shape == (13, 2, 1024)
        assert len(edge_to_hypothesis) == 13 and sum(map(len, edge_to_hypothesis)) == 39
        self.register_buffer("text_embeddings", F.normalize(text_embeddings.float(), dim=-1))
        self.register_buffer("edge_direction", direction.float())
        self.edge_to_hypothesis = [list(map(int, e)) for e in edge_to_hypothesis]
        self.use_aux, self.use_rival, self.top_k = use_aux, use_rival, int(top_k)
        self.query = nn.Linear(1024, 64, bias=False)
        self.context = nn.Linear(28, 64, bias=False)
        self.key = nn.Linear(4, 64, bias=True)
        self.summary = nn.Sequential(nn.LayerNorm(39), nn.Linear(39, 144, bias=False))
        self.alpha = nn.Parameter(torch.zeros(()))
        self.soft_temperature = 0.25
        self.null_logit = 0.0
        self.register_buffer("last_scores", torch.zeros(871, 39), persistent=False)
        self.register_buffer("last_signed", torch.zeros(871, 39), persistent=False)
        self.register_buffer("last_mask", torch.zeros(871, 39), persistent=False)
        self.register_buffer("last_gate", torch.zeros(871, 39), persistent=False)
        self._last_diagnostics = {}
        # Explicit generator prevents this module's initialization from changing shared RNG state.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            for layer in (self.query, self.context, self.key):
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * 0.02)
                if layer.bias is not None:
                    layer.bias.zero_()
            self.summary[1].weight.copy_(torch.randn(self.summary[1].weight.shape, generator=generator) * 0.01)

    def _selector(self, score: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        hard = torch.zeros_like(score)
        idx = torch.topk(score, self.top_k, dim=-1, sorted=True).indices
        hard.scatter_(1, idx, 1.0)
        if not training:
            return hard, hard
        threshold = score.topk(self.top_k, dim=-1).values[:, -1:].detach()
        soft = torch.sigmoid((score - threshold) / self.soft_temperature)
        soft = soft * (self.top_k / soft.sum(-1, keepdim=True).clamp_min(1e-6))
        st = hard + soft - soft.detach()
        return st, hard

    def forward(self, cam_flat: torch.Tensor, raw_x: torch.Tensor, z: torch.Tensor,
                qc_reliability: torch.Tensor, mask_override: torch.Tensor | None = None,
                inference_hard: bool = False, residual_enabled: bool = True) -> tuple[torch.Tensor, dict]:
        n = raw_x.shape[0]
        corr = raw_x[:, 64 + torch.as_tensor(
            [edge for group in self.edge_to_hypothesis for edge in group], device=raw_x.device
        )]
        cov_context = torch.cat([raw_x[:, :26], raw_x[:, 46:48]], dim=-1)
        if self.use_aux:
            context = self.context(cov_context)
            gate_subject = qc_reliability.reshape(n, 1).clamp(0.05, 1.0)
        else:
            context = torch.zeros((n, 64), device=raw_x.device, dtype=raw_x.dtype)
            gate_subject = torch.ones((n, 1), device=raw_x.device, dtype=raw_x.dtype)
        zflat = z.reshape(n, 39)
        prior = self.edge_direction.to(raw_x).reshape(1, 39).expand(n, -1)
        rel = gate_subject.expand(-1, 39)
        key_input = torch.stack([corr, zflat, prior, rel], dim=-1)
        keys = F.normalize(self.key(key_input), dim=-1)
        texts = self.text_embeddings.to(device=raw_x.device, dtype=raw_x.dtype)
        q = F.normalize(self.query(texts), dim=-1)
        q_asd = q[:, 0].unsqueeze(0) + context.unsqueeze(1)
        q_td = q[:, 1].unsqueeze(0) + context.unsqueeze(1)
        q_asd = F.normalize(q_asd, dim=-1)
        q_td = F.normalize(q_td, dim=-1)
        scores_asd = torch.zeros((n, 39), device=raw_x.device, dtype=raw_x.dtype)
        scores_td = torch.zeros_like(scores_asd)
        cursor = 0
        for h, edge_ids in enumerate(self.edge_to_hypothesis):
            count = len(edge_ids)
            k_h = keys[:, cursor:cursor + count]
            scores_asd[:, cursor:cursor + count] = (q_asd[:, h:h + 1] * k_h).sum(-1)
            scores_td[:, cursor:cursor + count] = (q_td[:, h:h + 1] * k_h).sum(-1)
            cursor += count
        if not self.use_rival:
            generic = 0.5 * (scores_asd + scores_td)
            probs = torch.softmax(torch.stack([generic, generic, torch.zeros_like(generic)], dim=-1), dim=-1)
            signed = torch.sigmoid(generic) * torch.tanh(zflat)
            generic_score = generic.abs() * zflat.abs() * rel
            score = generic_score
        else:
            probs = torch.softmax(torch.stack([scores_asd, scores_td,
                                               torch.zeros_like(scores_asd) + self.null_logit], dim=-1), dim=-1)
            signed = (probs[..., 0] - probs[..., 1]) * torch.tanh(zflat)
            score = (probs[..., 0] + probs[..., 1]) * zflat.abs() * rel + (probs[..., 0] - probs[..., 1]).abs()
        if mask_override is None:
            selected, hard = self._selector(score, self.training and not inference_hard)
        else:
            selected = mask_override.to(score)
            hard = (selected.detach() >= 0.5).to(score)
        signed_selected = signed * selected
        summaries = []
        cursor = 0
        for edge_ids in self.edge_to_hypothesis:
            count = len(edge_ids)
            p = probs[:, cursor:cursor + count]
            sv = signed_selected[:, cursor:cursor + count]
            summaries.extend([
                sv.sum(-1, keepdim=True) / max(count, 1),
                (p[..., 0] - p[..., 1]).mean(-1, keepdim=True),
                p[..., 2].mean(-1, keepdim=True),
            ])
            cursor += count
        summary = torch.cat(summaries, dim=-1)
        delta = self.summary(summary)
        # Per-patient residual is hard-clamped to 5% of the CAM representation norm.
        raw_norm = delta.norm(dim=-1, keepdim=True)
        cap = 0.05 * cam_flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        delta = delta * torch.clamp(cap / raw_norm.clamp_min(1e-8), max=1.0)
        delta = torch.tanh(self.alpha) * delta if residual_enabled else torch.zeros_like(delta)
        output = cam_flat + delta
        self.last_scores = score.detach()
        self.last_signed = signed.detach()
        self.last_mask = hard.detach()
        self.last_gate = rel.detach()
        self._last_diagnostics = {
            "signed": signed, "score": score, "selected": selected,
            "hard_selected": hard, "probs": probs, "delta": delta,
            "cam_norm": cam_flat.norm(dim=-1),
            "residual_ratio": (delta.norm(dim=-1) / cam_flat.norm(dim=-1).clamp_min(1e-8)),
            "null_fraction": probs[..., 2].mean(),
        }
        return output, self._last_diagnostics


def make_wrong_edge_map(hypotheses: list[dict], seed: int = 38002) -> list[list[int]]:
    """Whole-group derangement within exact group-size strata; source edge sets are disjoint."""
    rng = np.random.default_rng(seed)
    result = [None] * len(hypotheses)
    sizes: dict[int, list[int]] = {}
    for i, item in enumerate(hypotheses):
        sizes.setdefault(len(item["edges"]), []).append(i)
    for size, ids in sizes.items():
        if len(ids) < 2:
            raise ValueError(f"cannot derange size-{size} hypothesis groups")
        perm = np.asarray(ids)
        while True:
            perm = rng.permutation(ids)
            if np.all(perm != ids):
                break
        for dst, src in zip(ids, perm.tolist()):
            result[dst] = list(hypotheses[src]["edges"])
    assert all(set(result[i]).isdisjoint(hypotheses[i]["edges"]) for i in range(len(result)))
    return result


def candidate_direction(value: str) -> float:
    return 1.0 if value == "hyper_candidate" else -1.0 if value == "hypo_candidate" else 0.0


def load_prompts(path: str | Path) -> tuple[list[dict], list[list[int]], torch.Tensor]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    main = [h for h in obj["hypotheses"] if h["candidate"]]
    assert len(main) == 13 and len({edge for h in main for edge in h["edges"]}) == 39
    mapping = [list(h["edges"]) for h in main]
    direction = torch.tensor([candidate_direction(h["direction"]) for h in main for _ in h["edges"]])
    return main, mapping, direction
