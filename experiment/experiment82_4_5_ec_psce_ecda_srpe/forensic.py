"""Independent numerical formulas used by preflight and the post-training audit."""
from __future__ import annotations

import math

from common import history as h

import numpy as np
import torch
from scipy.special import expit, ndtr, softmax
from sklearn.model_selection import StratifiedKFold

CFG = h.CFG
EPS = CFG["spectrum"]["eps"]
R, B, E = (CFG[key] for key in ("residual", "boundary", "evidence"))
U32 = np.finfo(np.float32).eps / 2
U64 = np.finfo(np.float64).eps / 2


def array(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def f64(x):
    return array(x).astype(np.float64, copy=False)


def equal(actual, expected, name):
    np.testing.assert_array_equal(array(actual), array(expected), err_msg=name)


def gamma(ops, unit=U32):
    return ops * unit / (1 - ops * unit)


def bounded(actual, expected, operations, name, scale=1.0, unit=U32):
    """Test a specified forward-error envelope and return its utilization."""
    d = np.abs(f64(actual) - f64(expected))
    bound = gamma(operations, unit) * np.asarray(scale) + 2 * unit
    ratio = float(np.max(d / bound))
    h.require(np.isfinite(ratio) and ratio <= 1, f"{name} exceeds numerical bound by {ratio:.3g}×")
    return {"max_abs": float(d.max()), "bound_utilization": ratio}


def dot(actual, x, y, name, additional=0):
    x, y = f64(x), f64(y)
    return bounded(actual, x @ y, 2 * x.shape[-1] + additional, name, np.abs(x) @ np.abs(y))


def independent_head(head, cam):
    """Float64 LayerNorm/GELU and analytic derivative of the ASD-minus-TD logit."""
    norm, first, gelu, dropout = head.mlp_head
    h.require(gelu.approximate == "none" and isinstance(dropout, torch.nn.Dropout), "Original GELU/Dropout differs")
    c = f64(cam)
    centered = c - c.mean(-1, keepdims=True)
    sd = np.sqrt((centered * centered).mean(-1, keepdims=True) + norm.eps)
    standardized = centered / sd
    inputs = standardized * f64(norm.weight) + f64(norm.bias)
    w, b = f64(first.weight), f64(first.bias)
    wc, bc = f64(head.classifier.weight), f64(head.classifier.bias)
    x = inputs @ w.T + b
    phi = ndtr(x)
    logits = (x * phi) @ wc.T + bc
    gelu_prime = phi + x * np.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    diff = wc[0] - wc[1]
    linear = ((gelu_prime * diff) @ w) * f64(norm.weight)
    jac = (linear - linear.mean(-1, keepdims=True) -
           standardized * (standardized * linear).mean(-1, keepdims=True)) / sd
    # Absolute scale terms follow both dense products, normalization, and subtractive derivative.
    wsum = np.abs(inputs) @ np.abs(w.T) + np.abs(b)
    logits_scale = 1 + wsum @ np.abs(wc.T) + np.abs(bc)
    sensitivity = (1.25 * np.abs(diff) @ np.abs(w)) * np.abs(f64(norm.weight))
    jac_scale = (sensitivity + sensitivity.mean() + np.abs(standardized) *
                 (np.abs(standardized) * sensitivity).mean(-1, keepdims=True)) / sd
    return logits, jac, logits_scale, 1 + jac_scale


def check_original_o(ctx):
    ref = ctx["original"]
    legal = ctx["members"]["legal"]
    cam = f64(ctx["warm_cam"])[legal]
    labels = array(ctx["labels"])
    axes = f64(ref["U"])
    n = cam.shape[1]
    mean = cam.mean(0)
    cov = np.cov(cam, rowvar=False, ddof=1)
    ca, ct = (np.cov(cam[labels == y], rowvar=False, ddof=1) for y in (0, 1))
    v, u = np.linalg.eigh(cov)
    projector = np.eye(n) - u[:, -8:] @ u[:, -8:].T
    dv, du = np.linalg.eigh(projector @ (ca - ct) @ projector)
    ix = np.argsort(np.abs(dv))[::-1][:8]
    expected = np.concatenate((u[:, -8:][:, ::-1], du[:, ix]), axis=1)
    # Eigenvectors are identifiable up to sign. Compare eigen-equation residuals
    # and both projectors using float64 backward error and the spectrum gap.
    gap = min(v[-8] - v[-9], np.abs(dv[ix[-1]]) - np.sort(np.abs(dv))[-9])
    error_budget = gamma(20 * n, U64) * (np.linalg.norm(cov, 2) + np.linalg.norm(ca - ct, 2))
    max_projection = 0.0
    for start in (0, 8):
        target = expected[:, start:start + 8]
        observed = axes[:, start:start + 8]
        err = np.linalg.norm(observed @ observed.T - target @ target.T, 2)
        h.require(err <= 2 * error_budget / gap, "Independent Original O eigenspace differs")
        max_projection = max(max_projection, float(err))
    h.require(np.linalg.norm(axes.T @ axes - np.eye(16), 2) <= gamma(20 * n, U64),
              "Original O axes are not orthonormal")
    bounded(ref["mu"], mean, 2 * len(cam), "Original O legal CAM center",
            1 + np.abs(cam).mean(0), U64)
    bounded(ref["lambda"], np.einsum("ik,ij,jk->k", axes, cov, axes), 14 * n,
            "Original O legal variances", 1 + np.linalg.norm(cov), U64)
    return {"projector_max_2norm": max_projection, "eigengap": float(gap), "backward_budget": float(error_budget)}


def class_banks(ctx):
    ref = ctx["reference"]
    group = np.zeros(len(ref["labels"]), dtype=np.int64)
    splitter = StratifiedKFold(n_splits=E["inner_folds"], shuffle=True, random_state=ref["seed"])
    for j, (_, held) in enumerate(splitter.split(f64(ref["h"]), ref["labels"]), 1):
        group[held] = j
    equal(group, ref["group"], "Stratified inner assignment")
    legal = ctx["members"]["legal"]
    h.require(not np.intersect1d(legal, ctx["members"]["test"]).size, "Held-out row in Gaussian banks")
    centered = f64(ctx["warm_cam"])[legal] - f64(ctx["original"]["mu"])
    points = (centered @ f64(ctx["original"]["U"])) / np.sqrt(f64(ctx["original"]["lambda"]) + EPS)
    mask = np.stack([np.stack([(group != i) & (ref["labels"] == k) for k in range(2)]) for i in range(6)])
    equal(mask, ref["mask"], "Cross-fit class indicator")
    equal(mask.sum(-1), ref["counts"], "Cross-fit class counts")
    h.require(np.all(~mask[group, :, np.arange(len(legal))]), "Cross-fitted row uses own Gaussian bank")
    means = np.array([[points[m].mean(0) for m in bank] for bank in mask])
    second = np.array([[(points[m] ** 2).mean(0) for m in bank] for bank in mask])
    stable = stable_variance(means, second, mask.sum(-1))
    for key, target in (("mean", means), ("second", second), ("stable", stable), ("h", points)):
        bounded(ref[key], target, 4 * len(points), f"Independent initial bank {key}",
                1 + np.abs(target), U64)
    llr = gaussian(points, means[group], stable[group])[1]
    label = ref["labels"]
    signed = np.where(label[:, None] == 0, llr, -llr)
    skill = (np.tanh(signed[label == 0] / 2).mean(0) + np.tanh(signed[label == 1] / 2).mean(0)) / 2
    bounded(ref["calibration"], (1 + skill) / 2, 12 * len(points), "Independent OOF calibration", unit=U64)
    return {"counts": mask.sum(-1).tolist(), "min_oof_score": float(skill.min()), "max_oof_score": float(skill.max())}


def stable_variance(mean, second, count):
    n = count[:, :, None]
    v = np.maximum(0, second - mean ** 2) * (n / (n - 1))
    pooled = (v * n).sum(1) / n.sum(1)
    return 0.9 * v + 0.1 * pooled[:, None, :] + np.maximum(0.01 * pooled[:, None, :], 1e-4)


def gaussian(points, mean, var):
    density = -0.5 * (np.log(var) + (points[:, None, :] - mean) ** 2 / var)
    llr = density[:, 0, :] - density[:, 1, :]
    return llr, E["llr_bound"] * np.tanh(llr / E["llr_bound"])


def independent_observation(ctx, cam):
    ref = ctx["reference"]
    z = (f64(cam)[ref["legal"]] - f64(ctx["original"]["mu"])) @ f64(ctx["original"]["U"])
    z /= np.sqrt(f64(ctx["original"]["lambda"]) + EPS)
    split = [[z[group] for group in row] for row in ref["mask"]]
    obs_mean = [[points.mean(0) for points in row] for row in split]
    obs_second = [[(points ** 2).mean(0) for points in row] for row in split]
    return np.array([obs_mean, obs_second])


def normalize(values):
    return values / np.maximum(values.sum(-1, keepdims=True), EPS)


def vector_units(x, floor=R["jacobian_floor"]):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), floor)


def linear(layer, x):
    z = x @ f64(layer.weight).T
    if layer.bias is not None:
        z = z + f64(layer.bias)
    return z


def controller(module, x, extra_error=0):
    w1, w2 = f64(module[0].weight), f64(module[-1].weight)
    width = w1.shape[-1]
    v = linear(module[0], x)
    bias_scale = lambda layer: abs(f64(layer.bias)) if layer.bias is not None else 0
    magnitude = abs(x) @ abs(w1.T) + bias_scale(module[0])
    first_error = gamma(2 * width + 8) * (magnitude + 1) + extra_error * abs(w1.T).sum(0)
    result = linear(module[-1], v * ndtr(v))
    error = 1.25 * first_error @ abs(w2.T)
    error += gamma(2 * w2.shape[-1] + 8) * (1 + abs(v) @ abs(w2.T) + bias_scale(module[-1]))
    return result, error


def from_error(actual, expected, limit, name):
    d = abs(f64(actual) - expected)
    usage = np.max(d / np.maximum(limit, 2 * U32))
    h.require(np.isfinite(usage) and usage <= 1, f"{name} exceeds propagated error bound by {usage:.3g}×")
    return {"max_abs": float(d.max()), "bound_utilization": float(usage)}


def branch_formulas(ctx, cap, training=False, intervention="full"):
    mod = ctx["module"]
    d = cap["details"]
    c = f64(cap["cam"])
    head = ctx["state"]["cam"].Outputlayer
    axes = f64(mod.spectrum.basis)
    equal(d["branch"], cap["cam"], "Branch forward CAM")
    equal(mod.spectrum.basis, ctx["original"]["U"], "Frozen Original O basis")
    semantic_off = intervention == "semantic_off"

    white = f64(d["white"])
    desk = np.stack((abs(white), np.broadcast_to(f64(mod.spectrum.logvar.float()), white.shape),
                     np.broadcast_to(f64(mod.spectrum.disease_covariance.float()), white.shape)), -1)
    equal(d["descriptors"], desk, "Original O descriptors")
    token_x = np.stack((white, *np.moveaxis(desk, -1, 0)), axis=-1)
    embedding = linear(mod.spectrum.direction, axes.T)[None] + linear(mod.spectrum.descriptor, token_x)
    centered = embedding - embedding.mean(-1, keepdims=True)
    embed_sd = ((centered ** 2).mean(-1, keepdims=True) + mod.spectrum.norm.eps) ** .5
    tokens = centered / embed_sd * f64(mod.spectrum.norm.weight) + f64(mod.spectrum.norm.bias)
    j0 = dot(d["z"], c - f64(mod.spectrum.center), axes, "Original O projection", 2)
    bounded(white, f64(d["z"]) / np.sqrt(f64(mod.spectrum.variance) + EPS), 12, "Original O whitening", 1 + abs(white))
    bounded(d["tokens"], tokens, 350, "Original O prompt token embedding",
            1 + abs(tokens) + abs(embedding).mean(-1, keepdims=True) / embed_sd)
    spec = f64(d["tokens"])
    query = mod.spectrum.query_anonymous if intervention == "anonymous_query" else mod.spectrum.query_medical
    pp, nk = f64(query), f64(mod.spectrum.null)
    pp = pp / np.linalg.norm(pp, axis=-1, keepdims=True)
    nk = nk / np.linalg.norm(nk)
    keys = spec @ f64(mod.spectrum.key.weight).T
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    scores = np.concatenate((keys @ pp.T, np.broadcast_to(pp @ nk, (len(c), 1, 8))), axis=1).transpose(0, 2, 1)
    attention = softmax(scores / f64(d["temperature"]), axis=-1)
    h.require(attention.shape == (871, 8, 17), "Prompt attention does not use exactly 17 keys")
    errors = {"projection": j0, "medical_attention": bounded(d["attention"], attention, 160, "RN50 plus Null attention")}
    semantic = np.full((len(c), 8, 16), 1 / 17) if semantic_off else f64(d["attention"])[:, :, :16]
    bounded(d["semantic_attention"], semantic, 6, "Effective semantic attention")
    semantic = f64(d["semantic_attention"])
    coverage = 1 - np.prod(1 - semantic, axis=1)
    errors["coverage"] = bounded(d["coverage"], coverage, 50, "Coverage with 16 spectral directions and Null")
    bounded(d["null_mass"], 1 - semantic.sum(-1).mean(-1), 50, "Null mass")
    if semantic_off:
        equal(d["context"], np.zeros_like(array(d["context"])), "Query content reached semantic-off context")
    else:
        expected = (semantic.transpose(0, 2, 1) @ pp) / np.maximum(semantic.sum(1)[..., None], EPS)
        bounded(d["context"], expected, 90, "Query content at every direction", 1 + abs(expected))

    logits, jac_analytic, logit_scale, jac_scale = independent_head(head, c)
    m = f64(d["margin"])
    errors["deterministic_margin"] = bounded(m, logits[:, 0] - logits[:, 1], 390, "Native margin", logit_scale.sum(-1))
    jac = f64(d["jacobian"])
    errors["analytic_jacobian"] = bounded(jac, jac_analytic, 450, "Analytic LayerNorm-GELU Jacobian", jac_scale)
    jaxis = f64(d["jacobian_axis"])
    dot(jaxis, jac, axes, "Jacobian coordinate projection")
    jp = jaxis @ axes.T
    cnorm = np.linalg.norm(c, axis=-1)
    limit = R["fraction"] * cnorm
    capacity = limit * np.linalg.norm(jp if ctx["group"] == "G1" else jac, axis=-1)
    feasibility = expit((f64(d["capacity"]) - abs(m)) / (B["b_fraction"] * f64(d["capacity"]) + B["b_offset"]))
    bounded(d["capacity"], capacity, 300, "First-order margin capacity", 1 + np.abs(capacity))
    bounded(d["feasibility"], feasibility, 380, "Continuous capacity feature")
    cover = f64(d["coverage"])
    null = f64(d["null_mass"])
    tokens = f64(d["tokens"])
    context = f64(d["context"])
    w = f64(d["importance"])
    action = f64(d["action"])
    h.require((abs(action) <= 1).all(), "Signed action is outside [-1,1]")

    if ctx["group"] == "G1":
        sem = normalize(cover)
        stat_logits, stat_error = controller(mod.spectral_weights, desk)
        sp = softmax(stat_logits[..., 0], axis=-1)
        bounded(d["w_sem"], sem, 55, "G1 semantic weight")
        from_error(d["w_sp"], sp, stat_error.squeeze(-1).max(axis=1, keepdims=True) + gamma(50), "G1 statistical weight")
        entropy = -(sem * np.log(np.maximum(sem, EPS))).sum(-1) / math.log(16)
        beta = (1 - null) * (0.20 + 0.60 * (1 - entropy))
        if semantic_off:
            beta[:] = 0
        bounded(d["beta"], beta, 100, "G1 semantic interpolation")
        wp, ws = f64(d["w_sp"]), f64(d["w_sem"])
        b = f64(d["beta"])
        bounded(w, wp + b[:, None] * (ws - wp), 16, "G1 fused weights")
        contrast = semantic[:, 4:].mean(1) - semantic[:, :4].mean(1)
        if intervention == "prompt_contrast_off":
            contrast[:] = 0
        bounded(d["prompt_contrast"], contrast, 25, "Medical abnormal-minus-normal contrast")
        calibrated = np.tanh(f64(d["prompt_contrast"]) / CFG["prompt"]["contrast_temperature"])
        bounded(d["calibrated_contrast"], calibrated, 15, "Temperature-calibrated Prompt contrast")
        sem_signal = np.zeros_like(ws) if semantic_off else 16 * ws - 1
        scalars = np.stack((np.tanh(white), sem_signal, 16 * wp - 1, 16 * w - 1, vector_units(jaxis),
                            np.broadcast_to(np.tanh(m)[:, None], white.shape),
                            np.broadcast_to(b[:, None], white.shape), f64(d["calibrated_contrast"])), -1)
        token_context = context * cover[..., None]
        act_args = np.concatenate((tokens * (16 * w[..., None]), token_context, scalars, np.tanh(desk)), -1)
        act_logits, act_bound = controller(mod.action, act_args, gamma(60) * (1 + abs(act_args).max(-1, keepdims=True)))
        learned = np.tanh(act_logits[..., 0])
        if intervention == "signed_action_off":
            learned = abs(learned)
        from_error(action, learned, act_bound[..., 0] + gamma(20), "G1 signed controller")
        gate_features = (np.tanh(m), np.log1p(f64(d["capacity"])), f64(d["feasibility"]), b, null,
                         np.log1p(abs(white).mean(-1)), cover.mean(-1), (w * f64(d["calibrated_contrast"])).sum(-1))
        gate_args = np.concatenate(((tokens * w[..., None]).sum(1), (token_context * w[..., None]).sum(1),
                                    np.stack(gate_features, -1)), -1)
        direction = R["spectral_action_multiplier"] * w * action
        bounded(d["direction_coordinates"], direction, 8, "G1 scaled signed spectral coordinates")
        raw_p = (limit * f64(d["gate"]))[:, None] * (direction @ axes.T)
        raw_q = np.zeros_like(raw_p)
    else:
        ref = ctx["reference"]
        bank = f64(ref["bank"]).astype(int) if training else np.zeros(len(c), dtype=int)
        stats = mod.statistics.snapshot()
        mu = f64(stats["mean"].float())[bank]
        var = f64(stats["stable"].float())[bank]
        counts = f64(stats["counts"])[bank]
        raw, _ = gaussian(white, mu, var)
        density = -0.5 * (np.log(var) + (white[:, None, :] - mu) ** 2 / var)
        bounded(d["raw_llr"], raw, 34, "Signed Gaussian log density difference", 1 + abs(density).sum(1))
        clip = E["llr_bound"]
        bounded(d["llr"], clip * np.tanh(f64(d["raw_llr"]) / clip), 16, "Bounded Gaussian LLR", 1 + abs(f64(d["llr"])))
        pool = var.mean(1)
        signal = (mu[:, 0] - mu[:, 1]) ** 2 / pool + .5 * (np.log(var[:, 0] / var[:, 1])) ** 2
        uncertainty = (var / counts[..., None]).sum(1) / pool + (2 / (counts - E["ddof"])).sum(1)[:, None]
        trust = signal / (signal + uncertainty + E["effect_prior"] ** 2)
        distance = (white - mu.mean(1)) / np.sqrt(pool) / E["support_radius"]
        reliability = trust * (1 + distance ** 2) ** -.5 * f64(ref["calibration"].float())
        bounded(d["reliability"], reliability, 150, "G2 Gaussian evidence reliability")
        cn, ca = (1 - np.prod(1 - att, axis=1) for att in (semantic[:, :4], semantic[:, 4:]))
        bounded(d["normal"], cn, 30, "Normal Query coverage")
        bounded(d["abnormal"], ca, 30, "Abnormal Query coverage")
        llr, re = f64(d["llr"]), f64(d["reliability"])
        pref = expit(llr / E["prompt_evidence_temperature"])
        bounded(w, normalize(cn * (1 - pref) + ca * pref), 85, "G2 evidence weights")
        support = (w * re).sum(-1)
        evidence = (w * re * llr).sum(-1)
        bounded(d["support"], support, 55, "G2 support")
        bounded(d["evidence"], evidence, 55, "G2 evidence", 1 + (w * re * abs(llr)).sum(-1))
        e_dir = np.tanh(f64(d["evidence"]) / E["evidence_temperature"])
        bounded(d["direction"], e_dir, 16, "ASD-positive evidence")
        bounded(d["conflict"], expit(-m * e_dir / E["conflict_temperature"]), 90, "Evidence conflict feature")
        prior = E["signed_prior_scale"] * e_dir
        if intervention == "evidence_prior_off":
            prior[:] = 0
        bounded(d["signed_prior"], prior, 28, "Additive evidence direction prior")
        p, q = f64(d["parallel"]), f64(d["perp"])
        dot(p, jaxis, axes.T, "G2 projected Jacobian")
        bounded(q, jac - p, 4, "G2 complementary Jacobian", abs(jac) + abs(p))
        bounded(p + q, jac, 4, "G2 Jacobian reconstruction", 1 + abs(p) + abs(q))
        cross = (p * q).sum(-1)
        error = gamma(2 * 144 + 2 * 16 + 8) * (p ** 2 + q ** 2).sum(-1)
        h.require(np.all(np.abs(cross) <= error + 2 * U32), "G2 Jacobian subspaces are not orthogonal")
        errors["orthogonality_max_dot"] = float(np.max(np.abs(cross)))
        extras = np.stack((np.tanh(white), np.tanh(llr / 2), re, f64(d["normal"]), f64(d["abnormal"]), cover), -1)
        enriched = tokens + np.tanh(linear(mod.evidence_tokens, extras))
        gate_features = (np.tanh(m), np.log1p(f64(d["capacity"])), f64(d["feasibility"]), f64(d["conflict"]),
                         f64(d["support"]), f64(d["direction"]), null, np.log1p(np.linalg.norm(p, axis=-1)),
                         np.log1p(np.linalg.norm(q, axis=-1)), np.log1p(abs(white).mean(-1)), np.tanh(white).mean(-1))
        gate_args = np.concatenate(((enriched * w[..., None]).sum(1), (context * w[..., None]).sum(1),
                                    np.stack(gate_features, -1)), -1)
        expected_logits, error = [], []
        for worker in (mod.parallel_action, mod.complement_action):
            logits_a, bound_a = controller(worker, gate_args, gamma(160) * (1 + abs(gate_args).max(-1, keepdims=True)))
            expected_logits.append(logits_a)
            error.append(bound_a)
        from_error(d["action_logits"], np.concatenate(expected_logits, axis=-1), np.concatenate(error, axis=-1),
                   "Independently parameterized G2 action logits")
        observed_logits = f64(d["action_logits"])
        bounded(action[:, 0], np.tanh(observed_logits[:, 0] + prior), 30, "Spectral signed action with LLR prior")
        bounded(action[:, 1], np.tanh(observed_logits[:, 1]), 20, "Complementary signed action without LLR prior")
        size = limit * f64(d["gate"])
        raw_p = size[:, None] * action[:, :1] * vector_units(p)
        raw_q = size[:, None] * action[:, 1:] * vector_units(q)
        if intervention == "spectral_only":
            raw_q[:] = 0
        if intervention == "complement_only":
            raw_p[:] = 0

    gate_logits, gate_error = controller(mod.gate, gate_args, gamma(160) * (1 + abs(gate_args).max(-1, keepdims=True)))
    from_error(d["gate"], expit(gate_logits[:, 0]), 0.25 * gate_error[:, 0] + gamma(24), "Single unmultiplied gate")
    allowed = array(mod.spectrum.allowed)
    raw_p[~allowed] = raw_q[~allowed] = 0
    if intervention == "residual_off":
        raw_p[:] = raw_q[:] = 0
    if ctx["group"] == "G1":
        scale_p = limit[:, None] * abs(direction) @ abs(axes.T)
        scale_q = np.zeros_like(raw_q)
    else:
        scale_p, scale_q = abs(raw_p), abs(raw_q)
    bounded(d["raw_parallel"], raw_p, 350, "Spectral CAM residual before bound", abs(raw_p) + scale_p)
    bounded(d["raw_perp"], raw_q, 350, "Complementary CAM residual before bound", abs(raw_q) + scale_q)
    raw = f64(d["raw_parallel"]) + f64(d["raw_perp"])
    relative = np.linalg.norm(raw, axis=-1) / np.maximum(limit, EPS)
    squeeze = np.divide(np.tanh(relative), relative, out=np.ones_like(relative), where=relative != 0)
    shrink = squeeze * limit / np.maximum(limit, EPS)
    delta = f64(d["delta"])
    bounded(delta, raw * shrink[:, None], 325, "Smooth 8 percent feature bound", 1 + abs(raw))
    equal(d["delta"][~allowed], np.zeros_like(array(d["delta"])[~allowed]), "Fold 2 residual exactly zero")
    equal(cap["enhanced"], (c + delta).astype(np.float32), "Residual is applied at CAM144")
    dn = np.linalg.norm(delta, axis=-1)
    h.require(np.all(dn <= limit * (1 + gamma(2 * 144 + 10))), "CAM residual exceeded norm bound")
    if ctx["group"] == "G1":
        outside = delta - (delta @ axes) @ axes.T
        h.require(np.all(np.linalg.norm(outside, axis=-1) <= gamma(2 * 144) * dn),
                  "G1 realized residual leaves U16")
    if not training:
        expected_head, _, scale_head, _ = independent_head(head, cap["enhanced"])
        errors["real_head_logits"] = bounded(cap["logits"], expected_head, 390, "Original output head", scale_head)
    errors["max_relative"] = float(np.max(dn / np.maximum(cnorm, EPS)))
    return errors


def legal_memory(ctx):
    head = ctx["state"]["cam"].Outputlayer
    legal = ctx["members"]["legal"]
    c = ctx["warm_cam"][legal]
    labels = array(ctx["labels"])
    logits, _, scale, _ = independent_head(head, c)
    unit_margin = (logits[:, 0] - logits[:, 1]) * (1 - 2 * labels)
    temp = CFG["boundary"]["error_temperature"]
    expected = expit(-unit_margin / temp)
    limit = 0.25 * gamma(390) * scale.sum(-1) / temp + gamma(12)
    check = from_error(ctx["hard_initial"], expected, limit, "Legal epoch 10 difficulty")
    h.require(len(expected) == len(legal) and set(legal).isdisjoint(ctx["members"]["test"]),
              "Epoch 10 difficulty contains a held-out label")
    return {"rows": len(legal), "legal_index_sha": h.sha(legal), "memory_sha": h.sha(ctx["hard_initial"].detach().cpu()),
            "mean": float(expected.mean()), **check}


@torch.no_grad()
def amplitude_probe(ctx, cam):
    """Hold controllers' inputs fixed and scale their realized actions, including zero."""
    mod = ctx["module"]
    head = ctx["state"]["cam"].Outputlayer
    _, start = mod(cam, head, False)
    if ctx["group"] == "G1":
        controllers = [(mod.action, 0.)]
        parts = ("raw_parallel",)
    else:
        controllers = [(mod.parallel_action, start["signed_prior"][:, None]), (mod.complement_action, 0.)]
        parts = ("raw_parallel", "raw_perp")
    expected_norms = {part: float(start[part].norm()) for part in parts}
    h.require(all(value > 0 for value in expected_norms.values()), "Actual correction requires a nonzero action carrier")
    observed = {}
    for value in (0.5, 0.0):
        def scale(prior):
            return lambda _mod, _args, out: torch.atanh(value * torch.tanh(out + prior)) - prior

        handles = [controller.register_forward_hook(scale(prior)) for controller, prior in controllers]
        try:
            c, details = mod(cam, head, False)
        finally:
            for token in handles:
                token.remove()
        equal(details["gate"], start["gate"], "Action scaling changed the patient gate")
        for part in parts:
            # The second action logit can cancel the explicit prior. Its absolute rounding scale matters.
            raw_scale = abs(f64(start[part])) + f64(start["cam_norm"])[:, None] * R["fraction"]
            bounded(details[part], value * f64(start[part]), 18, "CAM effect follows action amplitude", raw_scale)
        if value == 0:
            equal(c, cam, "Zero actions must preserve original CAM")
            equal(details["delta"], torch.zeros_like(details["delta"]), "Zero actions produce zero residual")
        elif value == .5:
            h.require(float(details["delta"].norm()) < float(start["delta"].norm()),
                      "Halving all actions did not decrease the actual residual")
        observed[str(value)] = {part: float(details[part].norm()) for part in parts}
    with torch.no_grad():
        _, zero = mod(torch.zeros_like(cam), head, False)
        equal(zero["delta"], torch.zeros_like(zero["delta"]), "Zero CAM has zero residual")
    for carrier in (0., CFG["residual"]["jacobian_floor"] * 1e-5):
        x = torch.full((2, 144), carrier / 12, device=cam.device, dtype=cam.dtype)
        from common.spectral import unit
        bounded(unit(x), f64(x) / CFG["residual"]["jacobian_floor"], 4, "Near-zero Jacobian safe unit",
                abs(f64(x)) / CFG["residual"]["jacobian_floor"])
    return {"full": expected_norms, "scales": observed, "zero_cam": True, "near_zero_carrier": True}
