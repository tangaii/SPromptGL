"""Leakage-controlled lightweight probes for frozen Experiment 75.4 features."""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_VALUES = (0.01, 0.1, 1.0)


def asd_target(y_original):
    return 1 - np.asarray(y_original, dtype=np.int64)


def binary_metrics(y_asd, probability):
    y = np.asarray(y_asd, dtype=np.int64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, 1 - 1e-8)
    pred = (p >= 0.5).astype(np.int64)
    auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
    return {
        "acc": float(accuracy_score(y, pred)),
        "hard_auc": float(balanced_accuracy_score(y, pred)),
        "prob_auc": auc,
        "sen": float(((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())),
        "spe": float(((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum())),
        "asd_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "probability": p,
        "prediction": pred,
    }


def _pipeline(c):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=float(c), solver="lbfgs", max_iter=3000, tol=1e-9,
        random_state=0, class_weight=None,
    ))


def nested_logistic(train_x, train_y, eval_x, seed, c_values=C_VALUES):
    """Select C on outer-train-only five-fold log-loss, then refit once."""
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.int64)
    z = np.asarray(eval_x, dtype=np.float64)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    scores = {}
    for c in c_values:
        losses = []
        for fit_idx, val_idx in inner.split(x, y):
            model = _pipeline(c)
            model.fit(x[fit_idx], y[fit_idx])
            p = model.predict_proba(x[val_idx])[:, 1]
            losses.append(log_loss(y[val_idx], np.clip(p, 1e-8, 1 - 1e-8), labels=[0, 1]))
        scores[float(c)] = float(np.mean(losses))
    selected = min(scores, key=lambda c: (scores[c], c))
    model = _pipeline(selected)
    model.fit(x, y)
    probability = model.predict_proba(z)[:, 1]
    return probability, selected, scores, model


def fixed_projection(seed, token_dim=9, text_dim=1024, out_dim=128):
    rng = np.random.default_rng(int(seed))
    wt = rng.standard_normal((token_dim, out_dim), dtype=np.float64) / math.sqrt(out_dim)
    wp = rng.standard_normal((text_dim, out_dim), dtype=np.float64) / math.sqrt(out_dim)
    return wt.astype(np.float32), wp.astype(np.float32)


def b1_features(tokens, text_vectors, classes, token_projection, text_projection):
    """Fixed label-independent projections followed by cosine class contrasts."""
    t = np.asarray(tokens, dtype=np.float32)
    e = np.asarray(text_vectors, dtype=np.float32)
    k = t @ np.asarray(token_projection, dtype=np.float32)
    k /= np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1e-8)
    q = e @ np.asarray(text_projection, dtype=np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    return b1_features_projected(k, q, classes)


def b1_features_projected(projected_tokens, projected_text, classes):
    k = np.asarray(projected_tokens, dtype=np.float32)
    q = np.asarray(projected_text, dtype=np.float32)
    sims = np.einsum("ntd,pd->ntp", k, q, optimize=True)
    cls = np.asarray(classes)
    asd = np.flatnonzero(cls == "ASD")
    td = np.flatnonzero(cls == "TD")
    if not len(asd) or not len(td):
        raise ValueError("prompt bank must contain ASD and TD prompts")
    contrast = sims[:, :, asd].mean(axis=-1) - sims[:, :, td].mean(axis=-1)
    return contrast.astype(np.float32), sims.astype(np.float32), q.astype(np.float32)


def anonymous_gram_matched(projected_text, seed):
    """Rotate in the actual 128-D use space, preserving its Gram matrix."""
    e = np.asarray(projected_text, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    a = rng.standard_normal((e.shape[1], e.shape[1]))
    q, r = np.linalg.qr(a)
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    q = q * signs[None, :]
    anon = e @ q
    err = float(np.max(np.abs(e @ e.T - anon @ anon.T)))
    return anon.astype(np.float32), err


def random_unit_text(seed, count=8, dim=1024):
    rng = np.random.default_rng(int(seed))
    x = rng.standard_normal((count, dim))
    x /= np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    return x.astype(np.float32)


def trainable_alignment_scores(tokens, train_y, projected_text, classes, seed, c,
                               base_odds=None, random_anchor=None,
                               steps=60, learning_rate=0.03):
    """Train a shared 9->128 token alignment and a linear diagnostic head."""
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
    x = torch.as_tensor(np.asarray(tokens), dtype=torch.float32)
    q = torch.as_tensor(np.asarray(projected_text), dtype=torch.float32)
    q = F.normalize(q, dim=-1)
    labels = np.asarray(classes)
    asd_idx = torch.as_tensor(np.flatnonzero(labels == "ASD"), dtype=torch.long)
    td_idx = torch.as_tensor(np.flatnonzero(labels == "TD"), dtype=torch.long)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    w = (torch.randn((9, 128), generator=g, dtype=torch.float32) * 0.04).requires_grad_()
    use_base = base_odds is not None
    width = 17 if use_base else 16
    head = (torch.randn((width,), generator=g, dtype=torch.float32) * 0.01).requires_grad_()
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    anchor = None
    if random_anchor is not None:
        anchor = F.normalize(torch.as_tensor(random_anchor, dtype=torch.float32), dim=0)
    b = None if not use_base else torch.as_tensor(np.asarray(base_odds), dtype=torch.float32).reshape(-1, 1)
    y = torch.as_tensor(np.asarray(train_y, dtype=np.float32))
    optimizer = torch.optim.Adam((w, head, bias), lr=float(learning_rate))
    n = max(1, x.shape[0])
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        key = F.normalize(x @ w, dim=-1, eps=1e-8)
        if anchor is not None:
            contrast = torch.einsum("ntd,d->nt", key, anchor)
        else:
            sim = torch.einsum("ntd,pd->ntp", key, q)
            contrast = sim[:, :, asd_idx].mean(dim=-1) - sim[:, :, td_idx].mean(dim=-1)
        mean = contrast.mean(dim=0, keepdim=True)
        std = contrast.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
        z = (contrast - mean) / std
        if use_base:
            bm = b.mean(dim=0, keepdim=True)
            bs = b.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
            feats = torch.cat(((b - bm) / bs, z), dim=-1)
        else:
            feats = z
        logits = feats @ head + bias
        loss = F.binary_cross_entropy_with_logits(logits, y)
        penalty = (w.square().sum() + head.square().sum()) / (2.0 * float(c) * n)
        (loss + penalty).backward()
        optimizer.step()

    with torch.no_grad():
        key = F.normalize(x @ w, dim=-1, eps=1e-8)
        if anchor is not None:
            contrast = torch.einsum("ntd,d->nt", key, anchor)
        else:
            sim = torch.einsum("ntd,pd->ntp", key, q)
            contrast = sim[:, :, asd_idx].mean(dim=-1) - sim[:, :, td_idx].mean(dim=-1)
        mean = contrast.mean(dim=0, keepdim=True)
        std = contrast.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    return {
        "weight": w.detach(), "head": head.detach(), "bias": bias.detach(),
        "contrast_mean": mean.detach(), "contrast_std": std.detach(),
        "base_mean": None if not use_base else bm.detach(),
        "base_std": None if not use_base else bs.detach(),
        "anchor": anchor, "c": float(c),
    }


def alignment_predict(model, tokens, projected_text, classes, base_odds=None):
    import torch
    import torch.nn.functional as F
    x = torch.as_tensor(np.asarray(tokens), dtype=torch.float32)
    q = F.normalize(torch.as_tensor(np.asarray(projected_text), dtype=torch.float32), dim=-1)
    tags = np.asarray(classes)
    ai = torch.as_tensor(np.flatnonzero(tags == "ASD"), dtype=torch.long)
    ti = torch.as_tensor(np.flatnonzero(tags == "TD"), dtype=torch.long)
    with torch.no_grad():
        key = F.normalize(x @ model["weight"], dim=-1, eps=1e-8)
        if model["anchor"] is not None:
            contrast = torch.einsum("ntd,d->nt", key, model["anchor"])
        else:
            sim = torch.einsum("ntd,pd->ntp", key, q)
            contrast = sim[:, :, ai].mean(dim=-1) - sim[:, :, ti].mean(dim=-1)
        z = (contrast - model["contrast_mean"]) / model["contrast_std"]
        if base_odds is not None:
            b = torch.as_tensor(np.asarray(base_odds), dtype=torch.float32).reshape(-1, 1)
            b = (b - model["base_mean"]) / model["base_std"]
            z = torch.cat((b, z), dim=-1)
        logits = z @ model["head"] + model["bias"]
        return torch.sigmoid(logits).cpu().numpy(), contrast.cpu().numpy()


def nested_alignment(train_tokens, train_y, eval_tokens, text_projected, classes,
                     seed, train_base=None, eval_base=None, random_anchor=None,
                     c_values=C_VALUES, steps=60):
    """Nested five-fold C choice for the constrained learned-alignment probe."""
    import torch
    y = np.asarray(train_y, dtype=np.int64)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    scores = {}
    for c in c_values:
        losses = []
        for inner_id, (fit_idx, val_idx) in enumerate(inner.split(np.zeros(len(y)), y)):
            model = trainable_alignment_scores(
                train_tokens[fit_idx], y[fit_idx], text_projected, classes,
                seed=int(seed) + 1000 * inner_id + 17, c=c,
                base_odds=None if train_base is None else train_base[fit_idx],
                random_anchor=random_anchor, steps=steps,
            )
            p, _ = alignment_predict(
                model, train_tokens[val_idx], text_projected, classes,
                base_odds=None if train_base is None else train_base[val_idx],
            )
            losses.append(log_loss(y[val_idx], np.clip(p, 1e-8, 1 - 1e-8), labels=[0, 1]))
        scores[float(c)] = float(np.mean(losses))
    selected = min(scores, key=lambda c: (scores[c], c))
    final = trainable_alignment_scores(
        train_tokens, y, text_projected, classes, seed=int(seed) + 90000,
        c=selected, base_odds=train_base, random_anchor=random_anchor, steps=steps,
    )
    p, contrast = alignment_predict(final, eval_tokens, text_projected, classes, base_odds=eval_base)
    return p, contrast, selected, scores, final


def exact_signflip_p(differences):
    d = np.asarray(differences, dtype=np.float64)
    observed = abs(float(d.mean()))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(d)):
        values.append(abs(float(np.mean(d * np.asarray(signs)))))
    return float(np.mean(np.asarray(values) >= observed - 1e-15))


def bootstrap_mean_ci(differences, seed, resamples=10000):
    d = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(d, size=(int(resamples), len(d)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        out[idx] = running
    return out.tolist()
