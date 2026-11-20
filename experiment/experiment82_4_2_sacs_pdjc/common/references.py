"""Deterministic original-head Jacobians and legal cross-fitted class moments."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F


def frozen_head(head, cam):
    norm, linear, gelu, dropout = head.mlp_head
    if not isinstance(dropout, nn.Dropout):
        raise TypeError("Historical OutputLayer dropout position changed")
    hidden = F.layer_norm(cam, norm.normalized_shape, norm.weight.detach(), norm.bias.detach(), norm.eps)
    hidden = F.linear(hidden, linear.weight.detach(), linear.bias.detach())
    hidden = F.gelu(hidden, approximate=gelu.approximate)
    return F.linear(hidden, head.classifier.weight.detach(), head.classifier.bias.detach())


def margin(logits, labels):
    return (logits[:, 0] - logits[:, 1]) * (1 - 2 * labels)


def head_jacobian(head, cam):
    """One batched VJP of the ASD-minus-TD logit difference, without a higher-order graph."""
    with torch.enable_grad():
        leaf = cam.detach().requires_grad_(True)
        logits = frozen_head(head, leaf)
        gradient, = torch.autograd.grad((logits[:, 0] - logits[:, 1]).sum(), leaf)
    return logits.detach(), gradient.detach()


def balanced_mean(values, labels):
    return 0.5 * (values[labels == 0].mean() + values[labels == 1].mean())


def fit_reference(z, labels, legal, fold, cfg):
    if z.dtype != np.float64 or z.shape != (len(legal), 16):
        raise ValueError("Expected float64 Original O coordinates from legal E61 rows")
    seed = cfg["seed_base"] + fold
    splitter = StratifiedKFold(n_splits=cfg["inner_splits"], shuffle=True,
                               random_state=np.random.RandomState(seed))
    group = np.zeros(len(legal), dtype=np.int64)
    for index, (_, held) in enumerate(splitter.split(z, labels), 1):
        group[held] = index
    masks = np.stack([np.ones(len(legal), bool), *(group != value for value in range(1, cfg["inner_splits"] + 1))])
    masks = masks[:, None, :] & (labels[None, None, :] == np.array([0, 1])[None, :, None])
    counts = masks.sum(axis=-1)
    if not (counts > 16).all() or (group == 0).any():
        raise ValueError("Insufficient legal class rows for cross-fitted moments")
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
        legal, labels, counts = reference["legal"], reference["labels"], reference["counts"]
        bank = np.zeros(subjects, dtype=np.int64)
        bank[legal] = reference["group"]
        class_mask = labels[None, None] == np.array([0, 1])[None, :, None]
        include = np.arange(len(counts))[:, None] != reference["group"][None, :]
        weights = (class_mask & include[:, None]) / counts[..., None]
        self.register_buffer("summary_weights", torch.from_numpy(weights.astype(np.float32)), persistent=False)
        self.register_buffer("legal", torch.from_numpy(legal.copy()))
        self.register_buffer("bank", torch.from_numpy(bank))
        self.register_buffer("counts", torch.from_numpy(counts.copy()))
        self.register_buffer("mean", torch.tensor(reference["mean"], dtype=torch.float32))
        self.register_buffer("variance", torch.tensor(reference["variance"], dtype=torch.float32))
        self.register_buffer("anchor", variance.detach().float().clone())
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def advance(self, coordinates):
        values = coordinates[self.legal]
        weights = self.summary_weights
        batch_mean = (weights.flatten(0, 1) @ values).reshape_as(self.mean)
        deviations = values - batch_mean.unsqueeze(-2)
        batch_variance = (weights.unsqueeze(-1) * deviations.square()).sum(dim=-2)
        rate = self.cfg["ema_alpha"]
        difference = batch_mean - self.mean
        self.variance.lerp_(batch_variance, rate).addcmul_(difference, difference, value=rate * (1 - rate))
        self.mean.lerp_(batch_mean, rate)
        self.updates.add_(1)

    def evidence(self, coordinates, training, reliability_off=False):
        bank = self.bank if training else slice(0, 1)
        means, variances, counts = self.mean[bank], self.variance[bank], self.counts[bank]
        effect = means[:, 0] - means[:, 1]
        shrink = self.cfg["reference_shrinkage"]
        pooled = (1 - shrink) * variances.mean(dim=1) + shrink * self.anchor
        floor = self.cfg["relative_variance_floor"] * self.anchor + self.cfg["absolute_variance_floor"]
        scale = torch.maximum(pooled, floor).sqrt()
        standardized_effect = effect / scale
        position = (coordinates - means.mean(dim=1)) / scale
        signal = standardized_effect.square()
        uncertainty = (variances / counts.unsqueeze(-1)).sum(dim=1) / scale.square()
        class_reliability = signal / (signal + uncertainty + self.cfg["effect_prior"] ** 2)
        support = torch.rsqrt(1 + (position / self.cfg["support_radius"]).square())
        reliability = class_reliability * support
        if reliability_off:
            reliability = torch.ones_like(reliability)
        evidence = (position * standardized_effect).tanh() * reliability
        return {"effect": standardized_effect.expand_as(coordinates), "reliability": reliability,
                "class_evidence": evidence, "class_support": support, "stat_scale": scale}
