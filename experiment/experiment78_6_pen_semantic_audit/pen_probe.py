"""Frozen, dependency-light probes for the original SPromptGL PEN.

The functions below reproduce the algebra in the checked-in network.py without
loading a checkpoint.  They are used for numerical unit tests and positive
controls when historical PEN weights are unavailable.  They deliberately do
not claim to be patient-level results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

PROMPTS = [
    "Nromal brain volume", "Uniform cortical thickness", "Symmetrical brain structure",
    "Normal ventricular morphology", "Altered brain volume", "Abnormal functional connectivity",
    "White matter integrity issues", "Different activation patterns",
]


def spgs_weight(similarity: torch.Tensor) -> torch.Tensor:
    """Exact original operation: softmax over two classes then max."""
    return torch.softmax(similarity, dim=-1).max(dim=-1).values


def sigmoid_abs_margin(similarity: torch.Tensor) -> torch.Tensor:
    """Equivalent closed form for a two-class softmax maximum."""
    if similarity.shape[-1] != 2:
        raise ValueError("the identity is only for two classes")
    return torch.sigmoid((similarity[..., 0] - similarity[..., 1]).abs())


def original_pen(cam: torch.Tensor, text_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Reproduce original SPGS + RES for arbitrary N (shape N,4,4,9).

    The official code uses x.view(871,4,4,9); N is generalized here only so a
    small deterministic positive-control can run without a patient checkpoint.
    """
    if cam.ndim != 4 or tuple(cam.shape[1:]) != (4, 4, 9):
        raise ValueError(f"expected [N,4,4,9], got {tuple(cam.shape)}")
    if tuple(text_vectors.shape) != (2, 9):
        raise ValueError(f"expected [2,9] text vectors, got {tuple(text_vectors.shape)}")
    part = cam
    part_norm = F.normalize(part, p=2, dim=-1)
    text_norm = F.normalize(text_vectors, p=2, dim=-1)
    sim = torch.einsum("nmid,cd->nmic", part_norm, text_norm)
    weights = spgs_weight(sim)
    scaled = weights.unsqueeze(-1) * part
    global_x = F.avg_pool1d(cam.reshape(cam.shape[0] * 4, 1, 36), kernel_size=4, stride=4)
    global_x = global_x.reshape(cam.shape[0], 4, 9)
    parts = []
    v_sims = []
    for i in range(4):
        v = F.softmax(scaled[:, :, i, :] * global_x, dim=-1)
        parts.append(v * scaled[:, :, i, :])
        v_sims.append(v)
    residual = torch.cat(parts, dim=2)
    enhanced = cam.reshape(cam.shape[0], 4, 36) + residual
    enhanced = enhanced.reshape(cam.shape[0], 4, 4, 9)
    return {
        "cam": cam,
        "similarity": sim,
        "class_softmax": torch.softmax(sim, dim=-1),
        "spgs_weight": weights,
        "scaled_part": scaled,
        "global_x": global_x,
        "v_sim": torch.stack(v_sims, dim=2),
        "residual": residual,
        "enhanced": enhanced,
    }


def flatten_feature(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(t.shape[0], -1)


def probe(seed: int = 786, n: int = 23) -> Dict[str, object]:
    """Run deterministic synthetic source/operation controls only."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cam = torch.randn((n, 4, 4, 9), generator=g)
    text = torch.randn((2, 9), generator=g)
    classifier = torch.randn((2, 144), generator=g)
    bias = torch.randn((2,), generator=g)
    labels = torch.randint(0, 2, (n,), generator=g)

    base = original_pen(cam, text)
    swapped = original_pen(cam, text.flip(0))
    equal_text = text.mean(0, keepdim=True).repeat(2, 1)
    neutral = original_pen(cam, equal_text)
    numeric = original_pen(cam, torch.arange(18, dtype=cam.dtype).reshape(2, 9) - 8.5)
    wrong = original_pen(cam, text[[1, 0]])

    def logits(effect):
        return flatten_feature(effect["enhanced"]) @ classifier.T + bias

    l0 = logits(base)
    l_swap = logits(swapped)
    l_neutral = logits(neutral)
    l_numeric = logits(numeric)
    l_wrong = logits(wrong)
    p0 = l0.softmax(-1)[:, 0]

    # Positive controls: both a classifier perturbation and a PEN feature perturbation
    classifier_perturbed = classifier.clone(); classifier_perturbed[0, 0] += 1.e-3
    l_classifier_perturbed = flatten_feature(base["enhanced"]) @ classifier_perturbed.T + bias
    cam_perturbed = cam.clone(); cam_perturbed[:, 0, 0, 0] += 1.e-3
    l_feature_perturbed = logits(original_pen(cam_perturbed, text))

    return {
        "seed": seed,
        "n": n,
        "labels_checksum": hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
        "spgs_math_max_abs_error": float((base["spgs_weight"] - sigmoid_abs_margin(base["similarity"])).abs().max()),
        "swap_weight_max_abs": float((base["spgs_weight"] - swapped["spgs_weight"]).abs().max()),
        "swap_enhanced_max_abs": float((base["enhanced"] - swapped["enhanced"]).abs().max()),
        "swap_logits_max_abs": float((l0 - l_swap).abs().max()),
        "swap_prediction_equal": bool(torch.equal(l0.argmax(1), l_swap.argmax(1))),
        "neutral_logits_mean_abs": float((l0 - l_neutral).abs().mean()),
        "numeric_logits_mean_abs": float((l0 - l_numeric).abs().mean()),
        "wrong_text_logits_mean_abs": float((l0 - l_wrong).abs().mean()),
        "text_probability_mean_abs": float((p0 - l_numeric.softmax(-1)[:, 0]).abs().mean()),
        "positive_classifier_logit_mean_abs": float((l0 - l_classifier_perturbed).abs().mean()),
        "positive_feature_logit_mean_abs": float((l0 - l_feature_perturbed).abs().mean()),
        "spgs_weight_min": float(base["spgs_weight"].min()),
        "spgs_weight_max": float(base["spgs_weight"].max()),
        "spgs_weight_mean": float(base["spgs_weight"].mean()),
        "spgs_weight_std": float(base["spgs_weight"].std(unbiased=False)),
        "signed_margin_mean": float((base["similarity"][..., 0] - base["similarity"][..., 1]).mean()),
        "signed_margin_std": float((base["similarity"][..., 0] - base["similarity"][..., 1]).std(unbiased=False)),
        "absolute_margin_mean": float((base["similarity"][..., 0] - base["similarity"][..., 1]).abs().mean()),
        "residual_relative_norm_mean": float((flatten_feature(base["residual"]).norm(dim=1) / flatten_feature(base["cam"]).norm(dim=1).clamp_min(1.e-12)).mean()),
        "enhanced_input_cosine_mean": float(F.cosine_similarity(flatten_feature(base["cam"]), flatten_feature(base["enhanced"]), dim=1).mean()),
    }


def source_evidence(source_path: str | Path) -> Dict[str, object]:
    """Locate exact source constructs and line numbers for the report."""
    path = Path(source_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    patterns = {
        "prompt_list": "Nromal brain volume",
        "clip_load": "clip.load('RN50'",
        "text_buffer": "register_buffer('prompt_text_features'",
        "text_concat": "torch.cat([text_features[0:4]",
        "mlp_4096_128_9": "MLP(input_dim = 4096,hidden_dim = 128, output_dim = 9)",
        "part_shape": "part_x = x.view(871, 4, 4, 9)",
        "normalize_part": "part_x_norm = F.normalize(part_x, p=2, dim=-1)",
        "normalize_text": "text_features_norm = F.normalize(text_features, p=2, dim=-1)",
        "similarity": "similarity_matrix = torch.matmul",
        "softmax_classes": "torch.softmax(similarity_matrix, dim=-1)",
        "max_class": "similarity_max, _ = torch.max(similarity_matrix, dim=-1)",
        "spgs_scale": "part_x = similarity_max.unsqueeze(-1) * part_x",
        "avg_pool": "self.avg_pool = nn.AvgPool1d(kernel_size=4, stride=4)",
        "res_vsim": "Vsim = F.softmax(part*global_x, dim=-1)",
        "res_add": "x = x + torch.cat(combine_part,dim=2)",
        "output": "output, hidden = self.Outputlayer(x, attn_embedding)",
    }
    evidence = {}
    for name, needle in patterns.items():
        hits = [i + 1 for i, line in enumerate(lines) if needle in line]
        evidence[name] = {"needle": needle, "lines": hits, "found": bool(hits)}
    evidence["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence["line_count"] = len(lines)
    return evidence


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2, sort_keys=True))
