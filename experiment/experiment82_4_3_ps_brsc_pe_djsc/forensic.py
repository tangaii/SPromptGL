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
    v = np.maximum(0, second - mean ** 2) * n / (n - 1)
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


def vector_units(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), R["jacobian_floor"])


def branch_formulas(ctx, cap, training=False, intervention="full"):
    mod = ctx["module"]
    d = cap["details"]
    c = f64(cap["cam"])
    head = ctx["state"]["cam"].Outputlayer
    axes = f64(mod.spectrum.basis)
    equal(d["branch"], cap["cam"], "Branch forward CAM")
    equal(mod.spectrum.basis, ctx["original"]["U"], "Frozen Original O basis")
    exact_semantics = intervention == "semantic_off"
    spec = f64(d["tokens"])
    query = mod.spectrum.query_anonymous if intervention == "anonymous_query" else mod.spectrum.query_medical
    pp, nk = f64(query), f64(mod.spectrum.null)
    pp = pp / np.linalg.norm(pp, axis=-1, keepdims=True)
    nk = nk / np.linalg.norm(nk)
    keys = spec @ f64(mod.spectrum.key.weight).T
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    scores = np.concatenate((keys @ pp.T, np.broadcast_to(pp @ nk, (len(c), 1, 8))), axis=1).transpose(0, 2, 1)
    attention = softmax(scores / array(d["temperature"]), axis=-1)
    h.require(attention.shape == (871, 8, 17), "Prompt attention does not use exactly 17 keys")
    errors = {
        "projection": dot(d["z"], c - f64(mod.spectrum.center), axes, "Original O projection", 2),
        "medical_attention": bounded(d["attention"], attention, 160, "RN50 plus Null attention"),
    }
    semantic = np.full((len(c), 8, 16), 1 / 17) if exact_semantics else attention[:, :, :16]
    coverage = 1 - np.prod(1 - semantic, axis=1)
    errors["coverage"] = bounded(d["coverage"], coverage, 192, "Coverage with 16 spectral directions and Null")
    if exact_semantics:
        equal(d["context"], np.zeros_like(array(d["context"])), "Query content reached semantic-off context")
    logits, jac_analytic, logit_scale, jac_scale = independent_head(head, c)
    errors["deterministic_margin"] = bounded(d["margin"], logits[:, 0] - logits[:, 1], 390, "Native margin",
                                              logit_scale.sum(-1))
    jac = f64(d["jacobian"])
    errors["analytic_jacobian"] = bounded(jac, jac_analytic, 450, "Analytic LayerNorm-GELU Jacobian", jac_scale)
    jaxis = f64(d["jacobian_axis"])
    dot(jaxis, jac, axes, "Jacobian coordinate projection")
    jp = jaxis @ axes.T
    jq = jac - jp
    cnorm = np.linalg.norm(c, axis=-1)
    limit = R["fraction"] * cnorm
    capacity = limit * np.linalg.norm(jp if ctx["group"] == "G1" else jac, axis=-1)
    feasibility = expit((capacity - abs(f64(d["margin"]))) / (B["b_fraction"] * capacity + B["b_offset"]))
    bounded(d["capacity"], capacity, 300, "First-order margin capacity", 1 + np.abs(capacity))
    bounded(d["feasibility"], feasibility, 380, "Continuous capacity gate")
    if ctx["group"] == "G1":
        sem = normalize(f64(d["coverage"]))
        statistical_logits = (f64(d["descriptors"]) @ f64(mod.spectral_weights[0].weight).T +
                              f64(mod.spectral_weights[0].bias))
        statistical_logits *= ndtr(statistical_logits)
        sp = softmax((statistical_logits @ f64(mod.spectral_weights[-1].weight).T +
                      f64(mod.spectral_weights[-1].bias))[..., 0], axis=-1)
        null = f64(d["null_mass"])
        beta = (1 - null) * (0.20 + 0.60 * (1 + (sem * np.log(sem)).sum(-1) / np.log(16)))
        if exact_semantics:
            beta[:] = 0
        w = f64(d["importance"])
        bounded(d["w_sem"], sem, 128, "G1 semantic importance normalization")
        bounded(d["w_sp"], sp, 150, "G1 learned spectral statistics importance")
        bounded(d["beta"], beta, 200, "G1 semantic mixing")
        bounded(w, (1 - beta[:, None]) * f64(d["w_sp"]) + beta[:, None] * sem, 200, "G1 mixed importance")
        action = f64(d["action"])
        h.require((abs(action) <= 1).all(), "G1 signed action exceeds unit amplitude")
        gate = f64(d["learned_gate"])
        if intervention != "correctability_gate_off":
            gate = gate * feasibility
        bounded(d["gate"], gate, 390, "G1 feasibility is applied")
        raw_p = (R["branch_scale"] * cnorm * f64(d["gate"]))[:, None] * ((vector_units(w) * action) @ axes.T)
        raw_q = np.zeros_like(raw_p)
    else:
        source = mod.statistics
        bank = array(source.bank) if training else np.zeros(len(c), dtype=int)
        state = source.snapshot()
        mu, var, num = (f64(state[k])[bank] for k in ("mean", "stable", "counts"))
        white = f64(d["white"])
        llr0, llr = gaussian(white, mu, var)
        likelihood_scale = 1 + (np.abs(np.log(var)) + (white[:, None] - mu) ** 2 / var).sum(1)
        errors["gaussian_llr"] = bounded(d["raw_llr"], llr0, 40, "ASD-minus-TD Gaussian LLR", likelihood_scale)
        bounded(d["llr"], llr, 72, "Bounded Gaussian evidence", 1 + abs(llr))
        pooled = var.mean(1)
        signal = (mu[:, 0] - mu[:, 1]) ** 2 / pooled + (np.log(var[:, 0] / var[:, 1]) ** 2) / 2
        noise = (var / num[:, :, None]).sum(1) / pooled + (2 / (num - 1)).sum(1, keepdims=True)
        distance = (white - mu.mean(1)) / (np.sqrt(pooled) * E["support_radius"])
        rel = f64(source.calibration) * (signal / (signal + noise + E["effect_prior"] ** 2)) / np.sqrt(1 + distance ** 2)
        if intervention == "reliability_off":
            rel[:] = 1
        bounded(d["reliability"], rel, 120, "G2 class reliability")
        cn = 1 - np.prod(1 - f64(d["semantic_attention"])[:, :4], axis=1)
        ca = 1 - np.prod(1 - f64(d["semantic_attention"])[:, 4:], axis=1)
        preference = expit(f64(d["llr"]) / E["prompt_evidence_temperature"])
        weight = normalize(cn * (1 - preference) + ca * preference)
        support = (weight * rel).sum(-1)
        evidence = (weight * rel * f64(d["llr"])).sum(-1)
        direction = np.tanh(evidence / E["evidence_temperature"])
        conflict = expit(-f64(d["margin"]) * direction / E["conflict_temperature"])
        if intervention == "evidence_direction_off":
            direction = np.abs(direction)
        bounded(d["importance"], weight, 120, "G2 Prompt-conditioned disease weighting")
        bounded(d["support"], support, 180, "G2 aggregate reliability")
        bounded(d["evidence"], evidence, 200, "G2 continuous disease evidence", 1 + abs(evidence))
        bounded(d["direction"], direction, 320, "G2 evidence direction")
        bounded(d["conflict"], conflict, 320, "G2 disagreement gate")
        p, q = f64(d["parallel"]), f64(d["perp"])
        dot(p, jaxis, axes.T, "G2 spectral Jacobian")
        bounded(q, jac - p, 4, "G2 complementary Jacobian", 1 + abs(jac - p))
        bounded(p + q, jac, 4, "G2 Jacobian reconstruction", 1 + abs(p) + abs(q))
        cross = (p * q).sum(-1)
        error = gamma(2 * 144 + 2 * 16 + 8) * (p ** 2 + q ** 2).sum(-1)
        h.require(np.all(np.abs(cross) <= error + 2 * U32), "G2 Jacobian subspaces are not orthogonal")
        errors["orthogonality_max_dot"] = float(np.max(np.abs(cross)))
        gate = (f64(d["feasibility"]) * f64(d["conflict"]) * f64(d["learned_gate"]) *
                (E["reliability_gate_floor"] + (1 - E["reliability_gate_floor"]) * f64(d["support"])))
        bounded(d["gate"], gate, 12, "G2 applied reliability/conflict/feasibility")
        for key in ("strength_parallel", "strength_perp"):
            v = f64(d[key])
            h.require(((v > 0) & (v < 1)).all(), "G2 strengths exceed sigmoid range")
        amplitude = R["branch_scale"] * cnorm * f64(d["gate"]) * f64(d["direction"])
        raw_p = (amplitude * f64(d["strength_parallel"]))[:, None] * vector_units(p)
        raw_q = (amplitude * f64(d["strength_perp"]))[:, None] * vector_units(q)
        if intervention == "spectral_only":
            raw_q[:] = 0
        if intervention == "complement_only":
            raw_p[:] = 0
    allowed = array(mod.spectrum.allowed)
    raw_p[~allowed] = 0
    raw_q[~allowed] = 0
    if intervention == "residual_off":
        raw_p[:] = 0
        raw_q[:] = 0
    bounded(d["raw_parallel"], raw_p, 350, "Spectral CAM residual before bound", 1 + abs(raw_p))
    bounded(d["raw_perp"], raw_q, 350, "Complementary CAM residual before bound", 1 + abs(raw_q))
    raw = f64(d["raw_parallel"]) + f64(d["raw_perp"])
    relative = np.linalg.norm(raw, axis=-1) / np.maximum(limit, EPS)
    scale = np.divide(np.tanh(relative), relative, out=np.ones_like(relative), where=relative != 0)
    delta = f64(d["delta"])
    bounded(delta, raw * (scale * limit / np.maximum(limit, EPS))[:, None],
            325, "Smooth 8 percent feature bound", 1 + abs(raw))
    equal(d["delta"][~allowed], np.zeros_like(array(d["delta"])[~allowed]), "Fold 2 residual exactly zero")
    equal(cap["enhanced"], cap["cam"] + d["delta"], "Residual is applied at CAM144")
    dn = np.linalg.norm(delta, axis=-1)
    h.require(np.all(dn <= limit * (1 + gamma(2 * 144 + 10))), "CAM residual exceeded norm bound")
    if ctx["group"] == "G1":
        outside = delta - (delta @ axes) @ axes.T
        h.require(np.all(np.linalg.norm(outside, axis=-1) <= gamma(2 * 144) * dn),
                  "G1 realized residual leaves U16")
    elif intervention != "residual_off":
        directed = (delta * jac).sum(-1) * f64(d["direction"])
        h.require((directed >= -gamma(2 * 144) * np.linalg.norm(jac, axis=-1) * dn).all(),
                  "G2 real residual opposes continuous evidence")
    if not training:
        expected_head, _, scale_head, _ = independent_head(head, cap["enhanced"])
        errors["real_head_logits"] = bounded(cap["logits"], expected_head, 390, "Original output head",
                                              scale_head)
    errors["max_relative"] = float(np.max(dn / np.maximum(cnorm, EPS)))
    return errors
