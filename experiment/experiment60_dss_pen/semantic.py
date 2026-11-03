from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@contextmanager
def isolated_torch_rng(seed: int):
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if devices:
            torch.cuda.manual_seed_all(int(seed))
        yield


def project_nonnegative_l1_ball(values: torch.Tensor, force_sum_one: bool = False) -> torch.Tensor:
    positive = values.clamp_min(0)
    sorted_values, _ = torch.sort(values, dim=-1, descending=True)
    cssv = sorted_values.cumsum(dim=-1) - 1
    steps = torch.arange(1, values.shape[-1] + 1, device=values.device, dtype=values.dtype)
    condition = sorted_values - cssv / steps > 0
    rho = condition.sum(dim=-1, keepdim=True).clamp_min(1) - 1
    theta = cssv.gather(-1, rho) / (rho.to(values.dtype) + 1)
    projected = (values - theta).clamp_min(0)
    if force_sum_one:
        return projected
    return torch.where(positive.sum(dim=-1, keepdim=True) <= 1, positive, projected)


def constrained_projection(
    z: torch.Tensor,
    prototypes: torch.Tensor,
    mode: str = "partial_simplex",
    iterations: int = 128,
    epsilon: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Solve one class projection with a fixed unrolled FISTA path."""
    if mode == "softmax_attention":
        coefficients = torch.softmax(z @ prototypes.T, dim=-1)
        projected = coefficients @ prototypes
        distance = (z - projected).square().sum(dim=-1)
        return coefficients, projected, {"distance": distance, "iterations": torch.zeros((), device=z.device)}

    gram = prototypes @ prototypes.T
    cross = z @ prototypes.T
    lipschitz = 2.0 * (prototypes.square().sum() + epsilon)
    coefficients = torch.zeros_like(cross)
    accelerated = coefficients
    momentum = 1.0
    for _ in range(int(iterations)):
        gradient = 2.0 * (accelerated @ gram - cross + epsilon * accelerated)
        candidate = accelerated - gradient / lipschitz
        if mode == "partial_simplex":
            updated = project_nonnegative_l1_ball(candidate)
        elif mode == "simplex_sum_eq_one":
            updated = project_nonnegative_l1_ball(candidate, force_sum_one=True)
        elif mode == "nonnegative_box":
            updated = candidate.clamp(0.0, 1.0)
        else:
            raise ValueError(f"unknown projection mode: {mode}")
        next_momentum = (1.0 + (1.0 + 4.0 * momentum * momentum) ** 0.5) / 2.0
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - coefficients)
        coefficients, momentum = updated, next_momentum

    projected = coefficients @ prototypes
    distance = (z - projected).square().sum(dim=-1)
    return coefficients, projected, {
        "distance": distance,
        "iterations": torch.tensor(float(iterations), device=z.device),
        "lipschitz": lipschitz.detach(),
    }


def constrained_projection_batch(
    z: torch.Tensor,
    prototypes: torch.Tensor,
    mode: str = "partial_simplex",
    iterations: int = 128,
    epsilon: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate the same per-class solver for all classes in one tensor batch."""
    if prototypes.ndim != 3:
        raise ValueError("batched prototypes must have shape [classes, concepts, dimension]")
    if mode == "softmax_attention":
        coefficients = torch.softmax(torch.einsum("nd,ckd->cnk", z, prototypes), dim=-1)
        projected = torch.einsum("cnk,ckd->cnd", coefficients, prototypes)
        distance = (z.unsqueeze(0) - projected).square().sum(dim=-1)
        return coefficients, projected, {"distance": distance, "iterations": torch.zeros((), device=z.device)}

    gram = prototypes @ prototypes.transpose(-1, -2)
    cross = torch.einsum("nd,ckd->cnk", z, prototypes)
    lipschitz = 2.0 * (prototypes.square().sum(dim=(-1, -2)) + epsilon).view(-1, 1, 1)
    coefficients = torch.zeros_like(cross)
    accelerated = coefficients
    momentum = 1.0
    for _ in range(int(iterations)):
        gradient = 2.0 * (torch.matmul(accelerated, gram) - cross + epsilon * accelerated)
        candidate = accelerated - gradient / lipschitz
        if mode == "partial_simplex":
            updated = project_nonnegative_l1_ball(candidate)
        elif mode == "simplex_sum_eq_one":
            updated = project_nonnegative_l1_ball(candidate, force_sum_one=True)
        elif mode == "nonnegative_box":
            updated = candidate.clamp(0.0, 1.0)
        else:
            raise ValueError(f"unknown projection mode: {mode}")
        next_momentum = (1.0 + (1.0 + 4.0 * momentum * momentum) ** 0.5) / 2.0
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - coefficients)
        coefficients, momentum = updated, next_momentum
    projected = torch.einsum("cnk,ckd->cnd", coefficients, prototypes)
    distance = (z.unsqueeze(0) - projected).square().sum(dim=-1)
    return coefficients, projected, {
        "distance": distance,
        "iterations": torch.tensor(float(iterations), device=z.device),
        "lipschitz": lipschitz.detach().reshape(-1),
    }


class DSSPen(nn.Module):
    def __init__(
        self,
        text_bank: torch.Tensor,
        projection_mode: str = "partial_simplex",
        input_slot_mask: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ):
        super().__init__()
        if text_bank.ndim != 3 or text_bank.shape[0] != 2:
            raise ValueError("DSS text bank must have shape [2, concepts, text_dim]")
        self.register_buffer("text_bank", F.normalize(text_bank.float(), dim=-1))
        self.register_buffer("input_slot_mask", torch.tensor(input_slot_mask, dtype=torch.float32))
        self.projection_mode = projection_mode
        self.patient_projector = nn.Sequential(
            nn.Linear(144, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )
        self.text_adapter = nn.Linear(text_bank.shape[-1], 64, bias=False)
        self.correction = nn.Linear(64, 144, bias=False)
        nn.init.zeros_(self.correction.weight)

    def project(self, h: torch.Tensor) -> dict[str, Any]:
        n = h.shape[0]
        query_h = h * self.input_slot_mask.to(device=h.device, dtype=h.dtype).view(1, 4, 1)
        z = F.normalize(self.patient_projector(query_h.reshape(n, 144)), dim=-1)
        prototypes = F.normalize(self.text_adapter(self.text_bank), dim=-1)
        coeffs, projected, details = constrained_projection_batch(z, prototypes, self.projection_mode)
        distances = details["distance"].permute(1, 0)
        info = [
            {key: value[index] if torch.is_tensor(value) and value.ndim > 0 and key == "distance" else value for key, value in details.items()}
            for index in range(2)
        ]
        return {
            "z": z,
            "prototypes": prototypes,
            "coefficients": coeffs.permute(1, 0, 2),
            "projected": projected.permute(1, 0, 2),
            "distances": distances,
            "solver": info,
        }

    def losses(
        self,
        output: dict[str, Any],
        labels: torch.Tensor,
        train_idx: torch.Tensor,
        set_loss_enabled: bool,
        geometry_loss_enabled: bool = True,
        set_margin: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        distances = output["distances"][train_idx]
        y = labels[train_idx].long()
        set_loss = distances.sum() * 0.0
        if set_loss_enabled:
            signed = distances.gather(1, y[:, None]).squeeze(1) - distances.gather(1, (1 - y)[:, None]).squeeze(1)
            per_subject = F.relu(float(set_margin) + signed)
            set_loss = torch.stack([per_subject[y == cls].mean() for cls in range(2)]).mean()
        geometry = distances.sum() * 0.0
        if geometry_loss_enabled:
            raw = F.normalize(self.text_bank.flatten(0, 1), dim=-1)
            adapted = F.normalize(self.text_adapter(self.text_bank).flatten(0, 1), dim=-1)
            raw_gram = raw @ raw.T
            adapted_gram = adapted @ adapted.T
            geometry = (raw_gram - adapted_gram).square().mean()
        return {"set": set_loss, "geometry": geometry}

    def correct(
        self,
        h: torch.Tensor,
        output: dict[str, Any],
        epoch: int,
        residual_enabled: bool = True,
        slot_mask: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        residual_cap: float = 0.1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        d = output["distances"]
        class_mix = torch.softmax(-d / 0.2, dim=-1)
        semantic_direction = (class_mix[..., None] * output["projected"]).sum(dim=1) - output["z"]
        match_gate = (1.0 - d.min(dim=-1).values).clamp(0.0, 1.0)
        raw = self.correction(semantic_direction).view(-1, 4, 36)
        masks = torch.tensor(slot_mask, dtype=h.dtype, device=h.device).view(1, 4, 1)
        raw = raw * masks
        h_parts = h
        raw_norm = raw.norm(dim=-1, keepdim=True)
        h_norm = h_parts.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = (float(residual_cap) * h_norm / raw_norm.clamp_min(1e-8)).clamp(max=1.0)
        bounded = raw * scale
        schedule = min(0.1, max(0.0, (int(epoch) - 10) / 10.0 * 0.1))
        if not residual_enabled:
            schedule = 0.0
        correction = schedule * match_gate[:, None, None] * bounded
        return h + correction, {
            "match_gate": match_gate,
            "class_mix": class_mix,
            "semantic_direction": semantic_direction,
            "raw_correction": raw,
            "bounded_correction": bounded,
            "applied_correction": correction,
            "residual_schedule": torch.tensor(schedule, device=h.device),
        }


class BrainPromptModule(nn.Module):
    def __init__(self, disease_text: torch.Tensor):
        super().__init__()
        self.register_buffer("disease_text", F.normalize(disease_text.float(), dim=-1))
        self.patient_projector = nn.Sequential(nn.Linear(144, 128), nn.GELU(), nn.Linear(128, 64))
        self.text_adapter = nn.Linear(disease_text.shape[-1], 64, bias=False)

    def logits(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = F.normalize(self.patient_projector(h.reshape(h.shape[0], 144)), dim=-1)
        text = F.normalize(self.text_adapter(self.disease_text), dim=-1)
        return z @ text.T, z


class SaberModule(nn.Module):
    def __init__(self, disease_text: torch.Tensor, retrieval_text: torch.Tensor):
        super().__init__()
        self.register_buffer("disease_text", F.normalize(disease_text.float(), dim=-1))
        self.register_buffer("retrieval_text", retrieval_text.float())
        self.patient_projector = nn.Sequential(nn.Linear(144, 128), nn.GELU(), nn.Linear(128, 64))
        self.text_adapter = nn.Linear(retrieval_text.shape[-1], 64, bias=False)
        self.q_cls = nn.Parameter(torch.zeros(64))
        self.query = nn.Linear(64, 64, bias=False)
        self.key = nn.Linear(64, 64, bias=False)
        self.value = nn.Linear(64, 64, bias=False)
        self.gate = nn.Linear(64, 64)
        self.correction = nn.Linear(64, 144, bias=False)
        nn.init.zeros_(self.correction.weight)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        base = self.patient_projector(h.reshape(h.shape[0], 144))
        query = self.query(base + self.q_cls)
        tokens = self.text_adapter(self.retrieval_text)
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0).expand(h.shape[0], -1, -1)
        elif tokens.ndim == 3 and tokens.shape[0] != h.shape[0]:
            raise ValueError("patient-specific retrieval tokens have the wrong subject count")
        scores = torch.einsum("nd,nkd->nk", query, self.key(tokens)) / (64.0 ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        fused = torch.einsum("nk,nkd->nd", weights, self.value(tokens))
        gated = torch.sigmoid(self.gate(fused)) * fused
        semantic = F.normalize(base + gated, dim=-1)
        disease = F.normalize(self.text_adapter(self.disease_text), dim=-1)
        disease_logits = semantic @ disease.T / 0.07
        raw = self.correction(gated).view(-1, 4, 36)
        raw_norm = raw.norm(dim=-1, keepdim=True)
        h_norm = h.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        raw = raw * (0.1 * h_norm / raw_norm.clamp_min(1e-8)).clamp(max=1.0)
        return {
            "disease_logits": disease_logits,
            "semantic": semantic,
            "weights": weights,
            "fused": fused,
            "raw_correction": raw,
        }


class OriginalSPromptPEN(nn.Module):
    def __init__(self, original_text: torch.Tensor):
        super().__init__()
        self.register_buffer("original_text", original_text.float())
        self.mlp = nn.Sequential(
            nn.Linear(4096, 128), nn.ReLU(), nn.Linear(128, 9), nn.ReLU()
        )
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        n = h.shape[0]
        grouped = torch.stack(
            (self.original_text[:4].reshape(-1), self.original_text[4:].reshape(-1)), dim=0
        )
        text = F.normalize(self.mlp(grouped), dim=-1)
        parts = h.view(n, 4, 4, 9)
        part_norm = F.normalize(parts, p=2, dim=-1)
        similarities = torch.einsum("nmqd,cd->nmqc", part_norm, text)
        weights = similarities.softmax(dim=-1).amax(dim=-1)
        weighted_parts = parts * weights.unsqueeze(-1)
        global_x = self.pool(h.reshape(n * 4, 1, 36)).view(n, 4, 9)
        residual_parts = []
        for position in range(4):
            part = weighted_parts[:, :, position, :]
            residual_parts.append(torch.softmax(part * global_x, dim=-1) * part)
        residual = torch.cat(residual_parts, dim=2).view(n, 4, 36)
        return h + residual, {
            "weights": weights,
            "text_prototypes": text,
            "residual": residual,
        }


def module_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
