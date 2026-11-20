"""Deterministic native-head VJP and fold-local, cross-fitted EMA moments."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F


def frozen_head(head, cam):
    norm, linear, gelu, dropout = head.mlp_head
    if not isinstance(dropout, nn.Dropout):
        raise TypeError("Historical classification head changed")
    hidden = F.layer_norm(cam, norm.normalized_shape, norm.weight.detach(), norm.bias.detach(), norm.eps)
    hidden = F.linear(hidden, linear.weight.detach(), linear.bias.detach())
    hidden = F.gelu(hidden, approximate=gelu.approximate)
    return F.linear(hidden, head.classifier.weight.detach(), head.classifier.bias.detach())


def margin(logits, labels):
    return (logits[:, 0] - logits[:, 1]) * (1 - 2 * labels)


def head_jacobian(head, cam):
    """One VJP of the label-free ASD-minus-TD logit gap, with no higher-order graph."""
    with torch.enable_grad():
        leaf = cam.detach().requires_grad_(True)
        logits = frozen_head(head, leaf)
        gradient, = torch.autograd.grad((logits[:, 0] - logits[:, 1]).sum(), leaf)
    return logits.detach(), gradient.detach()


def balanced_mean(values, labels):
    return 0.5 * (values[labels == 0].mean() + values[labels == 1].mean())


def fit_reference(z, labels, legal, fold, cfg):
    """Build full and five held-out moment banks on already legal E61 rows."""
    if z.dtype != np.float64 or z.shape != (len(legal), 16):
        raise ValueError("Expected float64 E61 spectrum for legal training rows")
    seed = cfg["seed_base"] + fold
    splitter = StratifiedKFold(n_splits=cfg["inner_splits"], shuffle=True, random_state=np.random.RandomState(seed))
    group = np.zeros(len(legal), dtype=np.int64)
    for index, (_, hold) in enumerate(splitter.split(z, labels), 1):
        group[hold] = index
    masks = np.stack([np.ones(len(legal), bool), *(group != g for g in range(1, cfg["inner_splits"] + 1))])
    masks = masks[:, None, :] & (labels[None, None, :] == np.array([0, 1])[None, :, None])
    counts = masks.sum(axis=-1)
    if not (counts > 16).all() or (group == 0).any():
        raise ValueError("A class EMA bank has insufficient legal rows")
    means = np.stack([z[mask].mean(axis=0) for mask in masks.reshape(-1, len(z))]).reshape(*counts.shape, 16)
    variances = np.stack([z[mask].var(axis=0) for mask in masks.reshape(-1, len(z))]).reshape(*counts.shape, 16)
    if not np.isfinite(means).all() or not np.isfinite(variances).all():
        raise ValueError("Nonfinite E61 class moments")
    return {"legal": legal.copy(), "labels": labels.copy(), "seed": seed, "group": group,
            "counts": counts, "mean": means, "variance": variances}


class ClassMoments(nn.Module):
    def __init__(self, reference, subjects, variance, cfg):
        super().__init__()
        self.cfg = cfg
        legal = reference["legal"]
        labels = reference["labels"]
        bank = np.zeros(subjects, dtype=np.int64)
        bank[legal] = reference["group"]
        counts = reference["counts"]
        class_mask = labels[None, None] == np.array([0, 1])[None, :, None]
        include = np.arange(len(counts))[:, None] != reference["group"][None, :]
        weights = (class_mask & include[:, None]) / counts[..., None]
        self.register_buffer("summary_weights", torch.from_numpy(weights.astype(np.float32)), persistent=False)
        self.register_buffer("legal", torch.from_numpy(legal.copy()))
        self.register_buffer("bank", torch.from_numpy(bank))
        self.register_buffer("counts", torch.from_numpy(counts.copy()))
        self.register_buffer("mean", torch.tensor(reference["mean"], dtype=torch.float32))
        self.register_buffer("variance", torch.tensor(reference["variance"], dtype=torch.float32))
        self.register_buffer("anchor", variance.detach().clone())
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def advance(self, z):
        values = z[self.legal]
        weights = self.summary_weights
        batch_mean = (weights.flatten(0, 1) @ values).reshape_as(self.mean)
        deviations = values - batch_mean.unsqueeze(-2)
        batch_variance = (weights.unsqueeze(-1) * deviations.square()).sum(dim=-2)
        rate = self.cfg["ema_alpha"]
        difference = batch_mean - self.mean
        self.variance.lerp_(batch_variance, rate).addcmul_(difference, difference, value=rate * (1 - rate))
        self.mean.lerp_(batch_mean, rate)
        self.updates.add_(1)

    def evidence(self, z, training, reliability_off=False):
        bank = self.bank if training else slice(0, 1)
        mean, var, count = self.mean[bank], self.variance[bank], self.counts[bank]
        between = mean[:, 0] - mean[:, 1]
        shrink = self.cfg["reference_shrinkage"]
        pool = (1 - shrink) * var.mean(dim=1) + shrink * self.anchor
        variance_floor = self.cfg["relative_variance_floor"] * self.anchor + self.cfg["absolute_variance_floor"]
        scale = torch.maximum(pool, variance_floor).sqrt()
        contrast = between / scale
        patient_position = (z - mean.mean(dim=1)) / scale
        signal = contrast.square()
        uncertainty = (var / count.unsqueeze(-1)).sum(dim=1) / scale.square()
        class_reliability = signal / (signal + uncertainty + self.cfg["effect_prior"] ** 2)
        support = torch.rsqrt(1 + (patient_position / self.cfg["support_radius"]).square())
        reliability = class_reliability * support
        if reliability_off:
            reliability = torch.ones_like(reliability)
        evidence = (patient_position * contrast).tanh() * reliability
        return {"effect": contrast.expand_as(z), "reliability": reliability, "class_evidence": evidence,
                "class_support": support, "stat_scale": scale}
