import math
from itertools import combinations

import torch
import torch.nn.functional as F


PAIR_INDICES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
]

PAIR_NAMES = [
    "PHENO_ANAT",
    "PHENO_FUNC",
    "PHENO_Correlation",
    "ANAT_FUNC",
    "ANAT_Correlation",
    "FUNC_Correlation",
]


def normalize_tokens(tokens, eps=1e-8):
    return F.normalize(tokens, p=2, dim=-1, eps=eps)


def _off_diagonal_mean(matrix):
    n = matrix.size(0)
    if n <= 1:
        return matrix.new_tensor(0.0)
    return (matrix.sum() - torch.diagonal(matrix).sum()) / float(n * (n - 1))


def _multi_rbf_kernel(x, y, sigmas=(0.25, 0.5, 1.0, 2.0)):
    dist2 = torch.cdist(x, y, p=2).pow(2)
    kernels = [torch.exp(-dist2 / (2.0 * sigma * sigma)) for sigma in sigmas]
    return torch.stack(kernels, dim=0).mean(dim=0)


def _mmd2_biased(x, y):
    k_xx = _multi_rbf_kernel(x, x)
    k_yy = _multi_rbf_kernel(y, y)
    k_xy = _multi_rbf_kernel(x, y)
    return torch.clamp(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean(), min=0.0)


def pair_info_nce(tokens, m1, m2, temperature=0.10):
    """Subject matched InfoNCE for one modality pair."""
    z = normalize_tokens(tokens)
    z1 = z[:, m1, :]
    z2 = z[:, m2, :]
    batch_size = z1.size(0)
    if batch_size < 2:
        return z.sum() * 0.0

    logits = (z1 @ z2.transpose(0, 1)) / temperature
    target = torch.arange(batch_size, device=tokens.device)
    loss12 = F.cross_entropy(logits, target)
    loss21 = F.cross_entropy(logits.transpose(0, 1), target)
    normalizer = max(math.log(float(batch_size)), 1.0)
    return 0.5 * (loss12 + loss21) / normalizer


def pair_class_conditional_mmd(tokens, labels, m1, m2):
    z = normalize_tokens(tokens)
    losses = []
    for cls in torch.unique(labels):
        mask = labels == cls
        if int(mask.sum().item()) < 2:
            continue
        losses.append(_mmd2_biased(z[mask, m1, :], z[mask, m2, :]))
    if not losses:
        return z.sum() * 0.0
    return torch.stack(losses).mean()


def compute_pairwise_hybrid_losses(tokens, labels, temperature=0.10):
    pair_losses = []
    pair_parts = []
    for m1, m2 in PAIR_INDICES:
        nce = pair_info_nce(tokens, m1, m2, temperature)
        cmmd = pair_class_conditional_mmd(tokens, labels, m1, m2)
        hybrid = 0.5 * nce + 0.5 * cmmd
        pair_losses.append(hybrid)
        pair_parts.append({"nce": nce, "cmmd": cmmd, "hybrid": hybrid})
    return pair_losses, pair_parts


def weighted_pair_alignment_loss(pair_losses, weights):
    stacked = torch.stack(pair_losses)
    weights = weights.to(device=stacked.device, dtype=stacked.dtype).detach()
    if weights.numel() != stacked.numel():
        raise ValueError("pair weight count mismatch")
    weights = weights / weights.mean().clamp_min(1e-8)
    return (weights * stacked).mean()


def paired_info_nce(tokens, temperature=0.10):
    """The Experiment 3 global NCE reference, kept for integrity checks."""
    z = normalize_tokens(tokens)
    batch_size, modal_num, _ = z.shape
    if batch_size < 2 or modal_num < 2:
        return z.sum() * 0.0
    target = torch.arange(batch_size, device=z.device)
    normalizer = max(math.log(float(batch_size)), 1.0)
    losses = []
    for m1, m2 in combinations(range(modal_num), 2):
        logits = (z[:, m1, :] @ z[:, m2, :].transpose(0, 1)) / temperature
        losses.append(0.5 * (
            F.cross_entropy(logits, target)
            + F.cross_entropy(logits.transpose(0, 1), target)
        ) / normalizer)
    return torch.stack(losses).mean()


def class_conditional_mmd(tokens, labels):
    """The Experiment 3 global CMMD reference, kept for integrity checks."""
    z = normalize_tokens(tokens)
    batch_size, modal_num, _ = z.shape
    if batch_size < 2 or modal_num < 2:
        return z.sum() * 0.0
    losses = []
    for cls in torch.unique(labels):
        mask = labels == cls
        if int(mask.sum().item()) < 2:
            continue
        for m1, m2 in combinations(range(modal_num), 2):
            losses.append(_mmd2_biased(z[mask, m1, :], z[mask, m2, :]))
    if not losses:
        return z.sum() * 0.0
    return torch.stack(losses).mean()


def compute_alignment_loss(tokens, labels, mode="none", temperature=0.10):
    zero = tokens.sum() * 0.0
    if mode == "none":
        return zero, {"nce": 0.0, "cmmd": 0.0, "total": 0.0}
    if mode == "pair_nce":
        nce = paired_info_nce(tokens, temperature)
        return nce, {"nce": float(nce.detach()), "cmmd": 0.0, "total": float(nce.detach())}
    if mode == "cmmd":
        cmmd = class_conditional_mmd(tokens, labels)
        return cmmd, {"nce": 0.0, "cmmd": float(cmmd.detach()), "total": float(cmmd.detach())}
    if mode == "hybrid":
        nce = paired_info_nce(tokens, temperature)
        cmmd = class_conditional_mmd(tokens, labels)
        total = 0.5 * nce + 0.5 * cmmd
        return total, {
            "nce": float(nce.detach()),
            "cmmd": float(cmmd.detach()),
            "total": float(total.detach()),
        }
    raise ValueError("Unknown alignment mode: {}".format(mode))


@torch.no_grad()
def compute_alignment_metrics(tokens, labels):
    z = normalize_tokens(tokens)
    batch_size, modal_num, _ = z.shape
    pos_cos, neg_cos, retrieval = [], [], []
    if batch_size >= 2 and modal_num >= 2:
        target = torch.arange(batch_size, device=z.device)
        for m1, m2 in combinations(range(modal_num), 2):
            sim = z[:, m1, :] @ z[:, m2, :].transpose(0, 1)
            pos_cos.append(torch.diagonal(sim).mean())
            neg_cos.append(_off_diagonal_mean(sim))
            retrieval.append(0.5 * (
                (sim.argmax(dim=1) == target).float().mean()
                + (sim.argmax(dim=0) == target).float().mean()
            ))
    if pos_cos:
        pos = torch.stack(pos_cos).mean()
        neg = torch.stack(neg_cos).mean()
        r1 = torch.stack(retrieval).mean()
    else:
        pos = z.new_tensor(0.0)
        neg = z.new_tensor(0.0)
        r1 = z.new_tensor(0.0)
    return {
        "pos_cos": float(pos),
        "neg_cos": float(neg),
        "cos_gap": float(pos - neg),
        "retrieval_r1": float(r1),
        "cmmd": float(class_conditional_mmd(z, labels)),
    }


@torch.no_grad()
def compute_pair_alignment_metrics(tokens, labels):
    z = normalize_tokens(tokens)
    output = []
    batch_size = z.size(0)
    target = torch.arange(batch_size, device=z.device)
    for pair_id, (m1, m2) in enumerate(PAIR_INDICES):
        if batch_size >= 2:
            sim = z[:, m1, :] @ z[:, m2, :].transpose(0, 1)
            pos = torch.diagonal(sim).mean()
            neg = _off_diagonal_mean(sim)
            r1 = 0.5 * (
                (sim.argmax(dim=1) == target).float().mean()
                + (sim.argmax(dim=0) == target).float().mean()
            )
        else:
            pos = z.new_tensor(0.0)
            neg = z.new_tensor(0.0)
            r1 = z.new_tensor(0.0)
        output.append({
            "pair_id": pair_id,
            "pair_name": PAIR_NAMES[pair_id],
            "pos_cos": float(pos),
            "neg_cos": float(neg),
            "cos_gap": float(pos - neg),
            "retrieval_r1": float(r1),
            "cmmd": float(pair_class_conditional_mmd(tokens, labels, m1, m2)),
        })
    return output
