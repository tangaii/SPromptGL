"""Independent numerical formulas used by preflight and the post-training audit."""
from __future__ import annotations

import math
from copy import deepcopy

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
    cm, cv = comparison_moments(ctx, {k: ref[k] for k in ("mean", "second", "counts")})
    for i in range(1, 6):
        for j in range(1, 6):
            both = mask[i] & mask[j]
            h.require(not both[:, group == i].any(), "Evidence disagreement used an in-fold row")
            m = np.stack([points[x].mean(0) for x in both])
            s = np.stack([(points[x] ** 2).mean(0) for x in both])
            v = stable_variance(m[None], s[None], both.sum(-1)[None])[0]
            bounded(cm[i, j-1], m, 6 * len(points), "Legal comparison mean from explicit held-out mask",
                    1 + abs(m), U64)
            bounded(cv[i, j-1], v, 8 * len(points), "Legal comparison variance from explicit held-out mask",
                    1 + abs(v), U64)
    return {"counts": mask.sum(-1).tolist(), "min_oof_score": float(skill.min()), "max_oof_score": float(skill.max()),
            "disagreement_comparison_banks_exclude_target_inner_fold": True}


def comparison_moments(ctx, stats):
    ref = ctx["reference"]
    masks = f64(ref["mask"]).astype(bool)
    means, second, n = (f64(stats[key]) for key in ("mean", "second", "counts"))
    mu = np.empty((6, 5, 2, 16), dtype=np.float64)
    va = np.empty_like(mu)
    for i in range(6):
        for j in range(1, 6):
            count = (masks[i] & masks[j]).sum(-1)
            if i == 0 or i == j:
                m, s = means[j], second[j]
            else:
                coeff = np.zeros(6)
                coeff[i], coeff[j], coeff[0] = 1, 1, -1
                n_sum = coeff @ n
                equal(n_sum, count, "Comparison bank membership")
                m, s = ((x * (coeff[:, None] * n)[:, :, None]).sum(0) / count[:, None] for x in (means, second))
            mu[i, j-1] = m
            va[i, j-1] = stable_variance(m[None], s[None], count[None])[0]
    return mu, va


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

    def attend(queries):
        scores = np.concatenate((keys @ queries.T, np.broadcast_to(queries @ nk, (len(c), 1, 8))), axis=1)
        return softmax(scores.transpose(0, 2, 1) / f64(d["temperature"]), axis=-1)

    orig_attention = attend(pp)
    h.require(orig_attention.shape == (871, 8, 17), "Prompt attention size differs")
    errors = {"projection": j0}
    content = orig_attention[:, :, :16].transpose(0, 2, 1) @ pp
    att = orig_attention
    if ctx["group"] == "G1":
        cap_eta = CFG["prompt"]["common_suppression_cap"]
        eta = 0. if intervention == "differential_prompt_off" else cap_eta * expit(f64(mod.spectrum.suppression_logit))
        bounded(d["common_suppression"], eta, 16, "Bounded common query suppression")
        common = pp.mean(0)
        common /= np.linalg.norm(common)
        qdiff = pp - eta * (pp @ common)[:, None] * common
        qdiff /= np.linalg.norm(qdiff, axis=-1, keepdims=True)
        bounded(d["differential_query"], qdiff, 270, "Differential Prompt projection")
        ad = attend(qdiff)
        bounded(d["original_attention"], orig_attention, 160, "Original prompt attention")
        bounded(d["differential_attention"], ad, 320, "Differential prompt attention")
        att = (orig_attention + ad) / 2
        content = (content + ad[:, :, :16].transpose(0, 2, 1) @ qdiff) / 2
    errors["medical_attention"] = bounded(d["attention"], att, 350, "Original and differential 8 × 17 attention")
    equal(d["semantic_attention"], d["attention"][:, :, :16], "Semantic attention omits only Null")
    semantic = f64(d["semantic_attention"])
    coverage = 1 - np.prod(1 - semantic, axis=1)
    errors["coverage"] = bounded(d["coverage"], coverage, 50, "Coverage")
    null = f64(d["null_mass"])
    bounded(null, 1 - semantic.sum(-1).mean(-1), 60, "Null mass")
    expected = content / np.maximum(att[:, :, :16].sum(1)[..., None], EPS)
    bounded(d["context"], expected, 400, "Original and differential Query content", 1 + abs(expected))
    cover, context, tokens = (f64(d[key]) for key in ("coverage", "context", "tokens"))

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
    bounded(d["capacity"], capacity, 310, "First-order margin capacity", 1 + np.abs(capacity))
    w = f64(d["importance"])
    action = f64(d["action"])
    white0 = np.tanh(white)
    if ctx["group"] == "G1":
        sem = normalize(cover)
        stat_logits, stat_error = controller(mod.spectral_weights, desk)
        sp = softmax(stat_logits[..., 0], axis=-1)
        bounded(d["w_sem"], sem, 55, "Semantic weight")
        from_error(d["w_sp"], sp, stat_error[..., 0].max(axis=1, keepdims=True) + gamma(50), "Statistical weight")
        entropy = -(sem * np.log(np.maximum(sem, EPS))).sum(-1) / math.log(16)
        b = f64(d["beta"])
        bounded(b, (1 - null) * (0.20 + 0.60 * (1 - entropy)), 100, "Semantic interpolation")
        wp, ws = f64(d["w_sp"]), f64(d["w_sem"])
        bounded(w, wp + b[:, None] * (ws - wp), 16, "Fused weights")
        contrast = semantic[:, 4:].mean(1) - semantic[:, :4].mean(1)
        bounded(d["prompt_contrast"], contrast, 25, "Abnormal-minus-normal Prompt contrast")
        calibrated = f64(d["calibrated_contrast"])
        bounded(calibrated, np.tanh(f64(d["prompt_contrast"]) / CFG["prompt"]["contrast_temperature"]),
                18, "Calibrated Prompt contrast")
        scalars = np.stack((white0, 16 * ws - 1, 16 * wp - 1, 16 * w - 1, vector_units(jaxis),
                            np.broadcast_to(np.tanh(m)[:, None], white.shape),
                            np.broadcast_to(b[:, None], white.shape), calibrated), -1)
        token_context = context * cover[..., None]
        inputs = np.concatenate((tokens * (16 * w[..., None]), token_context, scalars, np.tanh(desk)), -1)
        val, err = controller(mod.direction_controller, inputs, gamma(60) * (1 + abs(inputs).max(-1, keepdims=True)))
        expected_action = np.tanh(val[..., 0])
        if intervention == "signed_action_off":
            expected_action = abs(expected_action)
        from_error(action, expected_action, err[..., 0] + gamma(20), "Signed correction direction")
        direction = R["spectral_action_multiplier"] * w * action
        bounded(d["direction_coordinates"], direction, 8, "Unnormalized signed spectral coordinates")
        prompt = (token_context * w[..., None]).sum(1)
        rot = CFG["rotation"]
        ri = np.concatenate((white0, calibrated, 16 * w - 1, prompt), -1)
        val, err = controller(mod.rotation_controller, ri, gamma(100) * (1 + abs(ri).max(-1, keepdims=True)))
        angles = f64(d["rotation_angles"])
        expected_angles = rot["angle_cap"] * np.tanh(val)
        if intervention == "patient_rotation_off":
            expected_angles[:] = 0
        errors["rotation_angle"] = from_error(angles, expected_angles, rot["angle_cap"] * err + gamma(20), "Rotation angle")
        h.require((abs(angles) <= rot["angle_cap"] + U32).all(), "Rotation angle exceeds registered limit")
        rmatrix = np.broadcast_to(np.eye(16), (len(c), 16, 16)).copy()
        for t, (a, b0) in enumerate(rot["planes"]):
            cc, ss = np.cos(angles[:, t]), np.sin(angles[:, t])
            rmatrix[:, a, a] = rmatrix[:, b0, b0] = cc
            rmatrix[:, a, b0], rmatrix[:, b0, a] = -ss, ss
        ident = rmatrix.transpose(0, 2, 1) @ rmatrix - np.eye(16)
        h.require(abs(ident).max() <= gamma(64, U64), "Patient rotation is not orthogonal")
        errors["rotation_orthogonality_max"] = float(abs(ident).max())
        rotated = f64(d["rotated_coordinates"])
        bounded(rotated, (rmatrix @ direction[..., None])[..., 0], 35, "Actual within-U16 patient rotation")
        alignment = (vector_units(rotated) * vector_units(jaxis)).sum(-1)
        bounded(d["alignment"], alignment, 70, "Patient rotation/Jacobian alignment")
        features = (np.tanh(m), np.log1p(abs(m)), np.log1p(f64(d["capacity"])), b, null, cover.mean(-1),
                    abs(calibrated).mean(-1), f64(d["alignment"]))
        amplitude_input = np.concatenate(((tokens * w[..., None]).sum(1), prompt, np.stack(features, -1)), -1)
        val, err = controller(mod.amplitude_controller, amplitude_input, gamma(150) * (1 + abs(amplitude_input).max(-1, keepdims=True)))
        amp = f64(d["amplitude"])
        from_error(amp, np.tanh(val[:, 0]), err[:, 0] + gamma(20), "Learned signed patient selection")
        bounded(d["gate"], abs(amp), 3, "Absolute signed amplitude")
        raw_p = (limit * amp)[:, None] * (rotated @ axes.T)
        raw_q = np.zeros_like(raw_p)
    else:
        ref = ctx["reference"]
        bank = array(ref["bank"]) if training else np.zeros(len(c), dtype=np.int64)
        stats = mod.statistics.snapshot()
        mu = f64(stats["mean"].float())[bank]
        var = f64(stats["stable"].float())[bank]
        counts = f64(stats["counts"])[bank]
        raw, _ = gaussian(white, mu, var)
        density = -0.5 * (np.log(var) + (white[:, None, :] - mu) ** 2 / var)
        bounded(d["raw_llr"], raw, 34, "Signed Gaussian log density difference", 1 + abs(density).sum(1))
        clip = E["llr_bound"]
        bounded(d["llr"], clip * np.tanh(f64(d["raw_llr"]) / clip), 16, "Bounded Gaussian LLR")
        pool = var.mean(1)
        signal = (mu[:, 0] - mu[:, 1]) ** 2 / pool + .5 * (np.log(var[:, 0] / var[:, 1])) ** 2
        uncertainty = (var / counts[..., None]).sum(1) / pool + (2 / (counts - E["ddof"])).sum(1)[:, None]
        trust = signal / (signal + uncertainty + E["effect_prior"] ** 2)
        distance = (white - mu.mean(1)) / np.sqrt(pool) / E["support_radius"]
        reliability = trust * (1 + distance ** 2) ** -.5 * f64(ref["calibration"].float())
        bounded(d["reliability"], reliability, 150, "Gaussian evidence reliability")
        cmu, cvar = comparison_moments(ctx, stats)
        bounded(mod.statistics.comparison_mean, cmu, 2, "Leave-two comparison mean after cast", 1 + abs(cmu))
        bounded(mod.statistics.comparison_variance, cvar, 2, "Leave-two comparison variance after cast", 1 + abs(cvar))
        cmu = f64(mod.statistics.comparison_mean)[bank]
        cvar = f64(mod.statistics.comparison_variance)[bank]
        cdensity = -0.5 * (np.log(cvar) + (white[:, None, None, :] - cmu) ** 2 / cvar)
        cllr = clip * np.tanh((cdensity[:, :, 0] - cdensity[:, :, 1]) / clip)
        ca_observed = f64(d["disagreement_axis"])
        sd_error = gamma(65) * np.sqrt(np.mean((1 + abs(cdensity).sum(2)) ** 2, axis=1))
        from_error(ca_observed, np.sqrt(cllr.var(1) + EPS), sd_error, "Leave-two bounded LLR disagreement")
        cn, ca = (1 - np.prod(1 - att, axis=1) for att in (semantic[:, :4], semantic[:, 4:]))
        bounded(d["normal"], cn, 30, "Normal Query coverage")
        bounded(d["abnormal"], ca, 30, "Abnormal Query coverage")
        llr, re = f64(d["llr"]), f64(d["reliability"])
        pref = expit(llr / E["prompt_evidence_temperature"])
        bounded(w, normalize(cn * (1 - pref) + ca * pref), 85, "Prompt evidence weights")
        bounded(d["support"], (w * re).sum(-1), 55, "Weighted class evidence support")
        evidence = (w * re * llr).sum(-1)
        bounded(d["evidence"], evidence, 55, "Gaussian evidence", 1 + (w * re * abs(llr)).sum(-1))
        edir = np.tanh(f64(d["evidence"]) / E["evidence_temperature"])
        bounded(d["direction"], edir, 20, "ASD-positive evidence sign")
        conflict = expit(-m * f64(d["direction"]) / E["conflict_temperature"])
        if intervention == "evidence_conflict_off":
            conflict[:] = .5
        bounded(d["conflict"], conflict, 80, "Evidence conflict")
        bounded(d["disagreement"], (w * ca_observed).sum(-1), 60, "Weighted bank disagreement")
        p, q = f64(d["parallel"]), f64(d["perp"])
        dot(p, jaxis, axes.T, "Projected Jacobian")
        bounded(q, jac - p, 4, "Complementary Jacobian", abs(jac) + abs(p))
        bounded(p + q, jac, 4, "Jacobian reconstruction", 1 + abs(p) + abs(q))
        cross = (p * q).sum(-1)
        p2, q2 = (p ** 2).sum(-1), (q ** 2).sum(-1)
        h.require((abs(cross) <= gamma(330) * (p2 + q2) + 2 * U32).all(), "Jacobian subspaces are not orthogonal")
        errors["orthogonality_max_dot"] = float(abs(cross).max())
        contrast = f64(d["prompt_contrast"])
        bounded(contrast, f64(d["abnormal"]) - f64(d["normal"]), 4, "Normal/abnormal Gaussian Prompt contrast")
        extras = np.stack((white0, np.tanh(llr / 2), re, f64(d["normal"]), f64(d["abnormal"]), cover, contrast,
                           np.log1p(ca_observed)), -1)
        enriched = tokens + np.tanh(linear(mod.evidence_tokens, extras))
        features = (np.tanh(m), np.log1p(abs(m)), f64(d["conflict"]), f64(d["support"]), f64(d["direction"]), null,
                    np.log1p(limit * np.sqrt(p2)), np.log1p(limit * np.sqrt(q2)), white0.mean(-1),
                    np.log1p(f64(d["disagreement"])), (w * contrast).sum(-1), np.tanh((w * llr).sum(-1)))
        inputs = np.concatenate(((enriched * w[..., None]).sum(1), (context * w[..., None]).sum(1), np.stack(features, -1)), -1)
        logits_ca, err = controller(mod.calibrator, inputs, gamma(210) * (1 + abs(inputs).max(-1, keepdims=True)))
        observed_logits = f64(d["calibration_logits"])
        from_error(observed_logits, logits_ca, err, "Independent desired-margin calibrator")
        conf = CFG["margin_calibration"]
        shift = conf["shift_cap"] * np.tanh(observed_logits[:, 0])
        if intervention == "calibrated_evidence_off":
            shift = abs(shift) * f64(d["direction"])
        bounded(d["shift"], shift, 25, "Bounded target shift")
        bounded(d["target_margin"], m + shift, 34, "True decision target margin", abs(m) + abs(shift))
        r, a = expit(observed_logits[:, 1]), expit(observed_logits[:, 2])
        bounded(d["correction_reliability"], r, 16, "Correction reliability")
        bounded(d["allocation"], a, 16, "Spectral allocation")
        desired = f64(d["desired_change"])
        bounded(desired, r * shift, 34, "Shared desired margin change")
        bounded(desired, r * (f64(d["target_margin"]) - m), 34, "Desired change r(target−m)", abs(m) + abs(shift))
        expected_s = np.column_stack((a * desired, (1 - a) * desired))
        bounded(d["action"], expected_s, 25, "Joint signed candidate budgets")
        raw_p = expected_s[:, :1] * p / (p2 + conf["jacobian_ridge"])[:, None]
        raw_q = expected_s[:, 1:] * q / (q2 + conf["jacobian_ridge"])[:, None]
        if intervention == "spectral_only":
            raw_q[:] = 0
        if intervention == "complement_only":
            raw_p[:] = 0
        linear_p = (raw_p * jac).sum(-1)
        linear_q = (raw_q * jac).sum(-1)
        h.require(not ((linear_p * linear_q < 0) & (abs(linear_p) > 2 * U32) &
                       (abs(linear_q) > 2 * U32)).any(), "Independent candidate Jacobians oppose the common target")
    allowed = array(mod.spectrum.allowed)
    raw_p[~allowed] = raw_q[~allowed] = 0
    if intervention == "residual_off":
        raw_p[:] = raw_q[:] = 0
    if ctx["group"] == "G1":
        scale_p = limit[:, None] * abs(rotated) @ abs(axes.T)
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
    """Check actual residual scaling while holding the other controls fixed."""
    mod = ctx["module"]
    head = ctx["state"]["cam"].Outputlayer
    _, start = mod(cam, head, False)
    g1 = ctx["group"] == "G1"
    parts = ("raw_parallel",) if g1 else ("raw_parallel", "raw_perp")
    expected = {part: float(start[part].norm()) for part in parts}
    h.require(all(v > 0 for v in expected.values()), "Missing full active residual component")
    observed = {}
    for value in (0.5, 0.0):
        def scale(_worker, _args, logits):
            if g1:
                return torch.atanh(value * logits.tanh())
            return torch.cat((torch.atanh(value * logits[:, :1].tanh()), logits[:, 1:]), -1)

        target = mod.amplitude_controller if g1 else mod.calibrator
        with target.register_forward_hook(scale):
            c, details = mod(cam, head, False)
        if g1:
            equal(details["rotation_angles"], start["rotation_angles"], "Scaling amplitude changed angle")
        else:
            for key in ("correction_reliability", "allocation"):
                equal(details[key], start[key], "Scaling target shift changed another control")
        for part in parts:
            raw_scale = abs(f64(start[part])) + f64(start["cam_norm"])[:, None] * R["fraction"]
            bounded(details[part], value * f64(start[part]), 20, "Actual CAM shift follows action amplitude", raw_scale)
        if value == 0:
            equal(c, cam, "Zero correction preserves original CAM")
            equal(details["delta"], torch.zeros_like(details["delta"]), "Zero action gives zero residual")
        else:
            h.require(float(details["delta"].norm()) < float(start["delta"].norm()),
                      "Halving the action did not decrease the realized residual")
        observed[str(value)] = {part: float(details[part].norm()) for part in parts}
    if g1:
        def scale_direction(_model, _args, logits):
            return torch.atanh(0.5 * logits.tanh())

        def fix_amplitude(_model, _args, logits):
            return start["amplitude_logit"][:, None]

        with mod.direction_controller.register_forward_hook(scale_direction), mod.amplitude_controller.register_forward_hook(fix_amplitude):
            _, d = mod(cam, head, False)
        bounded(d["raw_parallel"], 0.5 * f64(start["raw_parallel"]), 60,
                "Unnormalized direction scaling with amplitude held fixed", 1 + abs(f64(start["raw_parallel"])))
    _, zero = mod(torch.zeros_like(cam), head, False)
    equal(zero["delta"], torch.zeros_like(zero["delta"]), "Zero CAM has zero residual")
    if not g1:
        for factor in (0., 1e-10):
            near_head = deepcopy(head)
            near_head.classifier.weight.mul_(factor)
            _, values = mod(cam, near_head, False)
            h.require(bool(values["delta"].isfinite().all() & (values["ratio"] <= R["fraction"]).all()),
                      "Ridge failed to stabilize near-zero Jacobian")
            if factor == 0:
                equal(values["delta"], torch.zeros_like(values["delta"]), "Zero Jacobian has zero residual")
    return {"full": expected, "scales": observed, "zero_cam": True, "near_zero_carrier": not g1,
            "direction_scaling_with_fixed_amplitude": g1}
