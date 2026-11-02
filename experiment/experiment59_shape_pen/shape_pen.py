"""Shared disease-loss, partial-alignment and evidence utilities for E59."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiseaseProjection(nn.Module):
    """One shared BrainPrompt-style map for both frozen or learned anchors."""

    def __init__(self, prototype_bank: torch.Tensor, learnable: bool = False):
        super().__init__()
        if prototype_bank.ndim != 2 or prototype_bank.shape[0] != 2:
            raise ValueError("Disease prototype bank must have shape [2, D].")
        self.label_transform = nn.Linear(int(prototype_bank.shape[1]), 144, bias=True)
        if learnable:
            self.label_embeddings = nn.Parameter(prototype_bank.detach().clone())
        else:
            self.register_buffer("label_embeddings", prototype_bank.detach().clone())

    def prototypes(self) -> torch.Tensor:
        return F.normalize(self.label_transform(self.label_embeddings), p=2, dim=-1)


def attach_position_capture(cam: nn.Module) -> None:
    """Capture the final CAM positions without replacing its original forward."""
    if getattr(cam, "_e59_capture_handle", None) is not None:
        return

    def capture(_module, _inputs, output):
        cam._e59_positions = output

    cam._e59_positions = None
    cam._e59_capture_handle = cam.FeedForward[-1].register_forward_hook(capture)


def deterministic_head_logits(cam: nn.Module, representation: torch.Tensor) -> torch.Tensor:
    """Run the original OutputLayer with only its Dropout disabled."""
    head = cam.Outputlayer
    norm, linear, activation, _dropout = head.mlp_head
    values = F.layer_norm(
        representation,
        norm.normalized_shape,
        norm.weight,
        norm.bias,
        norm.eps,
    )
    values = activation(linear(values))
    return head.classifier(values)


def true_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.long().reshape(-1)
    other = 1 - labels
    return logits.gather(1, labels[:, None]).squeeze(1) - logits.gather(1, other[:, None]).squeeze(1)


@torch.no_grad()
def evidence_weights(
    cam: nn.Module,
    positions: torch.Tensor,
    target_indices: torch.Tensor,
    target_labels: torch.Tensor,
    reference_indices: torch.Tensor,
    *,
    exclude_self: bool,
    rho_min: float = 0.2,
    scale_floor: float = 0.1,
    scale_quantile: float = 0.75,
    scale_override: float | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Deterministic training-fold position substitution and detached rho."""
    if len(reference_indices) < 2:
        raise ValueError("At least two outer-training references are required.")
    target = positions[target_indices].detach()
    refs = positions[reference_indices].detach()
    reference_sum = refs.sum(dim=0)
    if exclude_self:
        if not torch.equal(target_indices, reference_indices):
            raise ValueError("Leave-one-out evidence requires identical target/reference indices.")
        replacements = (reference_sum.unsqueeze(0) - target) / float(len(reference_indices) - 1)
    else:
        replacements = reference_sum.unsqueeze(0).expand(len(target_indices), -1, -1) / float(len(reference_indices))

    labels = target_labels.long().reshape(-1)
    base_logits = deterministic_head_logits(cam, target.reshape(len(target), -1))
    base_margin = true_class_margin(base_logits, labels)

    counterfactual = target[:, None, :, :].expand(-1, target.shape[1], -1, -1).clone()
    position_count = target.shape[1]
    for modality in range(position_count):
        counterfactual[:, modality, modality, :] = replacements[:, modality, :]
    counterfactual_logits = deterministic_head_logits(
        cam,
        counterfactual.reshape(-1, target.shape[1] * target.shape[2]),
    ).reshape(len(target), position_count, 2)
    counterfactual_margin = true_class_margin(
        counterfactual_logits.reshape(-1, 2),
        labels[:, None].expand(-1, position_count).reshape(-1),
    ).reshape(len(target), position_count)
    delta = base_margin[:, None] - counterfactual_margin

    positive = delta.clamp_min(0.0)
    absolute = delta.abs()
    magnitude = absolute.sum(dim=1)
    positive_fraction = positive.sum(dim=1) / magnitude.clamp_min(1e-8)
    if scale_override is None:
        scale = torch.quantile(magnitude, scale_quantile).clamp_min(float(scale_floor))
    else:
        scale = torch.as_tensor(scale_override, dtype=magnitude.dtype, device=magnitude.device).clamp_min(float(scale_floor))
    rho = float(rho_min) + (1.0 - float(rho_min)) * positive_fraction * torch.tanh(magnitude / scale)
    return {
        "rho": rho.detach().clamp(min=float(rho_min), max=1.0),
        "delta": delta.detach(),
        "scale": scale.detach(),
        "positive_fraction": positive_fraction.detach(),
        "magnitude": magnitude.detach(),
    }


def project_disease_direction(
    representation: torch.Tensor,
    prototypes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prototypes.shape != (2, representation.shape[-1]):
        raise ValueError("Projected prototypes and patient representation dimensions disagree.")
    direction = prototypes[0] - prototypes[1]
    direction_norm = direction.norm()
    if not torch.isfinite(direction_norm) or float(direction_norm.detach()) <= 1e-8:
        raise ValueError("Disease prototype difference is zero or non-finite.")
    direction = direction / direction_norm
    disease = (representation * direction).sum(dim=-1, keepdim=True) * direction
    individual_residual = representation - disease
    return disease, individual_residual, direction


def cosine_class_loss(
    representation: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = F.normalize(representation, p=2, dim=-1) @ F.normalize(prototypes, p=2, dim=-1).T
    logits = logits / float(temperature)
    per_patient = F.cross_entropy(logits, labels.long().reshape(-1), reduction="none")
    return per_patient.mean(), logits, per_patient


def projected_dot_class_loss(
    representation: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use prototype dot products so a 1-D disease projection keeps its scale gradient."""
    logits = representation @ F.normalize(prototypes, p=2, dim=-1).T
    logits = logits / float(temperature)
    per_patient = F.cross_entropy(logits, labels.long().reshape(-1), reduction="none")
    return per_patient.mean(), logits, per_patient


def class_balanced_weighted_mean(
    per_patient: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    labels = labels.long().reshape(-1)
    weights = weights.detach().to(per_patient).reshape(-1)
    class_losses = []
    for label in (0, 1):
        mask = labels == label
        if not bool(mask.any()):
            raise ValueError(f"Training fold is missing class {label}.")
        denominator = weights[mask].sum().clamp_min(1e-8)
        class_losses.append((per_patient[mask] * weights[mask]).sum() / denominator)
    return torch.stack(class_losses).mean()


def supervised_contrastive_loss(
    representation: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if representation.shape[0] < 3:
        raise ValueError("Supervised contrastive loss needs at least three training patients.")
    labels = labels.long().reshape(-1)
    features = F.normalize(representation, p=2, dim=-1)
    logits = features @ features.T / float(temperature)
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    candidates = ~eye
    positives = labels[:, None].eq(labels[None, :]) & candidates
    if not bool(positives.any(dim=1).all()):
        raise ValueError("Each contrastive anchor needs a same-class training positive.")
    log_denominator = torch.logsumexp(logits.masked_fill(~candidates, -torch.inf), dim=1, keepdim=True)
    log_probability = logits - log_denominator
    per_anchor = -(log_probability.masked_fill(~positives, 0.0).sum(dim=1) / positives.sum(dim=1))
    return torch.stack([per_anchor[labels == label].mean() for label in (0, 1)]).mean()


def make_geometry_matched_anonymous_bank(
    reference: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    if reference.shape[0] != 2:
        raise ValueError("The official ABIDE bank must contain exactly two class vectors.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    vectors = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    first = F.normalize(vectors[0], dim=0)
    second = vectors[1] - torch.dot(vectors[1], first) * first
    second = F.normalize(second, dim=0)
    target_cosine = F.cosine_similarity(reference[0:1], reference[1:2]).item()
    vectors[0] = first * reference[0].norm()
    vectors[1] = (target_cosine * first + (1.0 - target_cosine**2) ** 0.5 * second) * reference[1].norm()
    return vectors
