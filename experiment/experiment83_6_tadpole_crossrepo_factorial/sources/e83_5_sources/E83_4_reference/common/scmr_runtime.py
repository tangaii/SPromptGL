"""Expose the archived SCMR computation with TADPOLE class and shape bindings."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold

from scmr_optimization import HISTORICAL_ROOT, SOURCE_HASHES, require
from source_runtime import archived_definitions, replace_exact, verified_text


@lru_cache(maxsize=1)
def source_config():
    return json.loads(verified_text(HISTORICAL_ROOT / "config.json", SOURCE_HASHES["config.json"]))


def class_pair(logits):
    """Keep binary 0-vs-1; orient the native leading pair by diagnosis in TADPOLE."""
    if logits.shape[-1] == 2:
        return torch.arange(2, device=logits.device).expand(len(logits), 2)
    require(logits.shape[-1] == 3, "SCMR requires the registered binary or TADPOLE class order")
    top_two = logits.detach().argsort(dim=-1, descending=True, stable=True)[:, :2]
    return top_two.sort(dim=-1, descending=True).values


CLASS_PAIRS = ((0, 1), (0, 2), (1, 2))


def _fixed_logits(head, cam):
    norm, linear, gelu, _ = head.mlp_head
    value = F.layer_norm(cam, norm.normalized_shape, norm.weight.detach(), norm.bias.detach(), norm.eps)
    value = F.linear(value, linear.weight.detach(), linear.bias.detach())
    value = F.gelu(value, approximate=gelu.approximate)
    return F.linear(value, head.classifier.weight.detach(), head.classifier.bias.detach())


def _differentiable_logits(head, cam):
    norm, linear, gelu, _ = head.mlp_head
    value = F.layer_norm(cam, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
    value = F.linear(value, linear.weight, linear.bias)
    value = F.gelu(value, approximate=gelu.approximate)
    return F.linear(value, head.classifier.weight, head.classifier.bias)


def _pair_head_jacobian(head, cam, directed_pair):
    with torch.enable_grad():
        leaf = cam.detach().requires_grad_()
        logits = _fixed_logits(head, leaf).gather(1, directed_pair)
        margin = logits[:, 0] - logits[:, 1]
        jacobian, = torch.autograd.grad(margin.sum(), leaf)
    value = margin.detach() + ((cam - cam.detach()) * jacobian).sum(-1)
    return value, jacobian.detach()


def _normalized_weights(value):
    return value / value.sum(-1, keepdim=True).clamp_min(source_config()["spectrum"]["eps"])


def _pair_forward(module, cam, head, training, intervention, base, mild, severe, pair_weight):
    query_indices = torch.tensor(
        [*range(mild * 4, mild * 4 + 4), *range(severe * 4, severe * 4 + 4)],
        dtype=torch.long, device=cam.device,
    )
    attention = base["attention"][:, query_indices]
    semantic = attention[:, :, :base["tokens"].shape[1]]
    normal_attention, abnormal_attention = semantic[:, :4], semantic[:, 4:]
    normal = 1 - (1 - normal_attention).prod(1)
    abnormal = 1 - (1 - abnormal_attention).prod(1)
    contrast = abnormal - normal

    directed_pair = torch.tensor([severe, mild], dtype=torch.long, device=cam.device).expand(len(cam), 2)
    raw_llr, llr, reliability = module.statistics(base["white"], training, directed_pair)
    disagreement_axis = module.statistics.disagreement(base["white"], training, directed_pair)
    evidence_weights = _normalized_weights(torch.lerp(
        normal, abnormal, (llr / source_config()["evidence"]["prompt_evidence_temperature"]).sigmoid()
    ))
    support = (evidence_weights * reliability).sum(-1)
    evidence = (evidence_weights * reliability * llr).sum(-1)
    disagreement = (evidence_weights * disagreement_axis).sum(-1)
    evidence_direction = (evidence / source_config()["evidence"]["evidence_temperature"]).tanh()

    query = module.spectrum.query_anonymous if intervention == "anonymous_query" else module.spectrum.query_medical
    pair_query = query[query_indices]
    context_numerator = torch.einsum("nqk,qd->nkd", semantic, pair_query)
    context = context_numerator / semantic.sum(1).clamp_min(1e-8).unsqueeze(-1)
    coverage = 1 - (1 - semantic).prod(1)
    null_mass = 1 - semantic.sum(-1).mean(-1)

    native_margin, jacobian = _pair_head_jacobian(head, cam, directed_pair)
    basis = module.spectrum.basis.to(cam)
    axis = jacobian @ basis
    parallel = axis @ basis.T
    perp = jacobian - parallel
    parallel_sq = parallel.square().sum(-1)
    perp_sq = perp.square().sum(-1)
    parallel_norm = parallel.norm(dim=-1)
    perp_norm = perp.norm(dim=-1)
    cfg = source_config()
    conflict = (-native_margin * evidence_direction / cfg["evidence"]["conflict_temperature"]).sigmoid()
    if intervention == "evidence_conflict_off":
        conflict = torch.full_like(conflict, .5)
    white = base["white"].tanh()
    descriptors = torch.stack((white, (llr / 2).tanh(), reliability, normal, abnormal,
                               coverage, contrast, disagreement_axis.log1p()), -1)
    tokens = base["tokens"] + module.evidence_tokens(descriptors).tanh()
    spectrum_summary = (tokens * evidence_weights.unsqueeze(-1)).sum(1)
    prompt_summary = (context * evidence_weights.unsqueeze(-1)).sum(1)
    cap = cfg["residual"]["fraction"] * base["cam_norm"]
    scalars = torch.stack((native_margin.tanh(), native_margin.abs().log1p(), conflict, support,
                           evidence_direction, null_mass, (cap * parallel_norm).log1p(),
                           (cap * perp_norm).log1p(), white.mean(-1), disagreement.log1p(),
                           (evidence_weights * contrast).sum(-1), (evidence_weights * llr).sum(-1).tanh()), -1)
    calibrator_logits = module.calibrator(torch.cat((spectrum_summary, prompt_summary, scalars), -1))
    shift_cfg = cfg["margin_calibration"]
    shift = shift_cfg["shift_cap"] * calibrator_logits[:, 0].tanh()
    if intervention == "calibrated_evidence_off":
        shift = shift.abs() * evidence_direction
    trust, allocation = calibrator_logits[:, 1:].sigmoid().unbind(-1)
    desired = trust * shift
    parallel_strength = desired * allocation
    perp_strength = desired * (1 - allocation)
    raw_parallel = (parallel_strength / (parallel_sq + shift_cfg["jacobian_ridge"]))[:, None] * parallel
    raw_perp = (perp_strength / (perp_sq + shift_cfg["jacobian_ridge"]))[:, None] * perp
    if intervention == "spectral_only":
        raw_perp = torch.zeros_like(raw_perp)
    if intervention == "complement_only":
        raw_parallel = torch.zeros_like(raw_parallel)

    details = {
        "class_pair": directed_pair, "pair_weight": pair_weight, "raw_llr": raw_llr, "llr": llr,
        "reliability": reliability, "importance": evidence_weights, "support": support,
        "normal": normal, "abnormal": abnormal, "prompt_contrast": contrast,
        "evidence": evidence, "direction": evidence_direction,
        "disagreement_axis": disagreement_axis, "disagreement": disagreement,
        "calibration_logits": calibrator_logits, "shift": shift,
        "target_margin": native_margin + shift, "desired_change": desired,
        "correction_reliability": trust, "allocation": allocation,
        "action": torch.stack((parallel_strength, perp_strength), -1),
        "margin": native_margin, "jacobian": jacobian, "jacobian_axis": axis,
        "parallel": parallel, "perp": perp, "capacity": cap * jacobian.norm(dim=-1),
        "conflict": conflict, "gate": trust, "strength_parallel": parallel_strength,
        "strength_perp": perp_strength, "tokens": tokens, "context": context,
        "coverage": coverage, "null_mass": null_mass,
        "raw_parallel": raw_parallel, "raw_perp": raw_perp,
    }
    return details


def _all_pair_forward(self, cam, head, training, intervention="full"):
    base = self.spectrum.encode(cam, intervention)
    native_probabilities = _fixed_logits(head, cam.detach()).softmax(-1).detach()
    pair_weights = torch.stack([
        (native_probabilities[:, mild] + native_probabilities[:, severe]) * .5
        for mild, severe in CLASS_PAIRS
    ], dim=1)
    pair_outputs = []
    raw_parallel, raw_perp = [], []
    for pair_index, (mild, severe) in enumerate(CLASS_PAIRS):
        detail = _pair_forward(self, cam, head, training, intervention, base, mild, severe,
                               pair_weights[:, pair_index])
        pair_outputs.append(detail)
        raw_parallel.append(detail["raw_parallel"] * pair_weights[:, pair_index, None])
        raw_perp.append(detail["raw_perp"] * pair_weights[:, pair_index, None])
    aggregate_parallel = torch.stack(raw_parallel).sum(0)
    aggregate_perp = torch.stack(raw_perp).sum(0)

    aggregate_keys = (
        "raw_llr", "llr", "reliability", "importance", "support", "normal", "abnormal",
        "prompt_contrast", "evidence", "direction", "disagreement_axis", "disagreement",
        "calibration_logits", "shift", "target_margin", "desired_change", "correction_reliability",
        "allocation", "action", "margin", "jacobian", "jacobian_axis", "parallel", "perp",
        "capacity", "conflict", "gate", "strength_parallel", "strength_perp", "tokens", "context",
        "coverage", "null_mass",
    )
    details = dict(base)
    details["class_pairs"] = torch.tensor([[severe, mild] for mild, severe in CLASS_PAIRS], device=cam.device)
    details["pair_weights"] = pair_weights
    details["pair_outputs"] = pair_outputs
    for key in aggregate_keys:
        details[key] = sum(pair_weights[:, index].reshape(-1, *([1] * (pair_outputs[index][key].ndim - 1)))
                           * pair_outputs[index][key] for index in range(len(pair_outputs)))
    details["raw_parallel"] = aggregate_parallel
    details["raw_perp"] = aggregate_perp
    corrected, result = self.spectrum.correct(cam, aggregate_parallel, aggregate_perp, details, intervention)
    return corrected, result


def _macro_class_mean(values, labels):
    return torch.stack([values[labels == class_id].mean() for class_id in labels.unique(sorted=True)]).mean()


def _pair_balanced(values, labels, mild, severe):
    return .5 * (values[labels == mild].mean() + values[labels == severe].mean())


def _pair_signed_margin(logits, labels, mild, severe):
    sign = torch.where(labels == severe, 1., -1.).to(logits.dtype)
    return sign * (logits[:, severe] - logits[:, mild]), sign


def _l2_bound(raw, cam_norm, fraction, eps):
    cap = fraction * cam_norm
    safe_cap = cap.clamp_min(eps)
    ratio = raw.norm(dim=-1) / safe_cap
    small = ratio < 1e-4
    local = torch.where(small, ratio, 0.0)
    regular = torch.where(small, 1.0, ratio)
    shrink = torch.where(small, 1 - local.square() / 3 + 2 * local.pow(4) / 15,
                         regular.tanh() / regular)
    return shrink * (cap / safe_cap)


def _all_pair_objectives(self, head, capture, labels, rows):
    cfg = source_config()
    details = capture["details"]
    rows = rows.long()
    y = labels[rows]
    cam = capture["cam"][rows]
    full_logits = _differentiable_logits(head, capture["enhanced"][rows])
    native_logits = _differentiable_logits(head, cam)
    full_ce = _macro_class_mean(F.cross_entropy(full_logits, y, reduction="none"), y)
    off_ce = _macro_class_mean(F.cross_entropy(native_logits, y, reduction="none"), y)
    true_margin_full = true_class_margin(full_logits, y)
    true_margin_native = true_class_margin(native_logits, y)
    gain_from_final = true_margin_full - true_margin_native

    raw_candidates = torch.stack((details["raw_parallel"][rows], details["raw_perp"][rows]))
    shrink = _l2_bound(raw_candidates, details["cam_norm"][rows], cfg["residual"]["fraction"],
                       cfg["spectrum"]["eps"]).unsqueeze(-1)
    candidate_logits = _differentiable_logits(head, cam.unsqueeze(0) + raw_candidates * shrink)

    auxiliary = {name: [] for name in ("gain", "guard", "guidance", "regularization")}
    diagnostic_gains = []
    for pair_index, (mild, severe) in enumerate(CLASS_PAIRS):
        selected = (y == mild) | (y == severe)
        pair_labels = y[selected]
        pair_native_logits = native_logits[selected]
        pair_full_logits = full_logits[selected]
        off, sign = _pair_signed_margin(pair_native_logits, pair_labels, mild, severe)
        pair_gain = _pair_signed_margin(pair_full_logits, pair_labels, mild, severe)[0] - off
        current = (-off / cfg["boundary"]["error_temperature"]).sigmoid()
        initial = self.hard_initial_pairs[rows[selected], pair_index]
        focus = 1 - (1 - current) * (1 - cfg["boundary"]["initial_memory_mix"] * initial)
        guard = (1 - current) * (-off.abs() / cfg["boundary"]["protect_scale"]).exp()
        tau = cfg["boundary"]["gain_temperature"]
        shortfall = F.softplus(-pair_gain / tau)

        pair_candidates = candidate_logits[:, selected]
        candidate_margins = []
        for candidate in pair_candidates:
            candidate_margins.append(_pair_signed_margin(candidate, pair_labels, mild, severe)[0] - off)
        dual_weights = cfg["dual_gain_weights"]
        gain_cost = (dual_weights["joint"] * shortfall
                     + dual_weights["spectral"] * F.softplus(-candidate_margins[0] / tau)
                     + dual_weights["complement"] * F.softplus(-candidate_margins[1] / tau))

        pair_details = details["pair_outputs"][pair_index]
        support = pair_details["support"][rows[selected]]
        disagreement = pair_details["disagreement"][rows[selected]]
        uncertainty = 1 - support / ((support + .1) * (1 + disagreement))
        direction = pair_details["direction"][rows[selected]]
        need = current * ((1 + sign * direction) / 2) * (1 - uncertainty)
        target_margin = pair_details["target_margin"][rows[selected]]
        target_loss = F.softplus(-sign * target_margin)
        calibration_logits = pair_details["calibration_logits"][rows[selected], 1]
        reliability_loss = F.binary_cross_entropy_with_logits(calibration_logits, need, reduction="none")
        desired = pair_details["desired_change"][rows[selected]]
        regularization = details["ratio"][rows[selected]].square()
        regularization = regularization + cfg["margin_calibration"]["regularization_ratio"] * uncertainty * (
            desired / cfg["margin_calibration"]["shift_cap"]
        ).square()

        auxiliary["gain"].append(_pair_balanced(focus * gain_cost, pair_labels, mild, severe))
        auxiliary["guard"].append(_pair_balanced(guard * shortfall, pair_labels, mild, severe))
        auxiliary["guidance"].append(_pair_balanced(
            focus * target_loss + cfg["margin_calibration"]["reliability_aux_weight"] * reliability_loss,
            pair_labels, mild, severe,
        ))
        auxiliary["regularization"].append(_pair_balanced(regularization, pair_labels, mild, severe))
        diagnostic_gains.append(pair_gain.detach().mean())

    objectives = {name: torch.stack(values).mean() for name, values in auxiliary.items()}
    objectives.update(full=full_ce, off=off_ce, difference=full_ce - off_ce)
    return objectives, {"pair_gain_means": torch.stack(diagnostic_gains),
                        "full_ce_all_classes": full_ce.detach(),
                        "off_ce_all_classes": off_ce.detach()}


def select_classes(values, pair):
    # Class-conditioned evidence is stored with class axis one after the
    # patient batch axis. Gather the two routed classes while preserving every
    # trailing spectral/feature axis.
    if values.ndim == pair.ndim:
        values = values.unsqueeze(-1)
    values = values.expand(len(pair), *values.shape[1:])
    index = pair.reshape(len(pair), 2, *([1] * (values.ndim - 2)))
    index = index.expand(len(pair), 2, *values.shape[2:])
    return values.gather(1, index)


def select_comparison(values, pair):
    """Select routed classes on the class axis of bank-comparison tensors."""
    if values.ndim < 3:
        return values
    values = values.expand(len(pair), *values.shape[1:])
    index = pair.reshape(len(pair), 1, 2, *([1] * (values.ndim - 3)))
    index = index.expand(len(pair), values.shape[1], 2, *values.shape[3:])
    return values.gather(2, index)


def true_class_margin(logits, labels):
    index = labels.expand(*logits.shape[:-1]).unsqueeze(-1)
    native = logits.gather(-1, index).squeeze(-1)
    competitor = logits.scatter(-1, index, -torch.inf).amax(dim=-1)
    return native - competitor


@lru_cache(maxsize=None)
def scmr_components(subjects, width, *, adapted=True):
    config = source_config()
    spectral_path = HISTORICAL_ROOT / "common/spectral.py"
    spectral_text = verified_text(spectral_path, SOURCE_HASHES["common/spectral.py"])
    if adapted:
        spectral_text = replace_exact(spectral_text, "torch.ones(CFG[\"subjects\"], dtype=torch.bool)",
                                      "torch.ones(SUBJECTS, dtype=torch.bool)")
        spectral_text = replace_exact(spectral_text, "nn.Linear(144, width)", "nn.Linear(CAM_WIDTH, width)")
        spectral_text = replace_exact(spectral_text,
                                      'tuple(cam.shape) == (CFG["subjects"], 144), "Correction input is not CAM144"',
                                      'cam.ndim == 2 and cam.shape[1] == CAM_WIDTH, "Unexpected SCMR CAM width"')
        spectral_text = replace_exact(spectral_text, "allowed = self.allowed[:, None]",
                                      "allowed = (self.allowed if len(cam) == SUBJECTS else "
                                      "self.allowed.new_ones(len(cam)))[:, None]")
        spectral_text = replace_exact(spectral_text, "def head_jacobian(head, cam):",
                                      "def head_jacobian(head, cam, pair):")
        spectral_text = replace_exact(spectral_text, "value = margin(fixed_logits(head, leaf))",
                                      "value = margin(fixed_logits(head, leaf).gather(1, pair))")
    suffix = f"{subjects}_{width}_{int(adapted)}"
    spectrum = archived_definitions(
        spectral_path, ("native_logits", "fixed_logits", "margin", "head_jacobian", "split_jacobian",
                        "l2_bound", "normalized_weights", "MedicalSpectrum"),
        source=spectral_text, CFG=config, EPS=config["spectrum"]["eps"], R=config["residual"],
        P=config["prompt"], require=require, math=math, SUBJECTS=subjects, CAM_WIDTH=width,
        module_name=f"e831_scmr_spectral_{suffix}",
    )

    evidence_path = HISTORICAL_ROOT / "common/evidence.py"
    evidence_text = verified_text(evidence_path, "0c8e2e33d802cd4fd2a1996b3d298fc84297b77e56b7d24045612d7e31eac6ba")
    if adapted:
        evidence_text = replace_exact(evidence_text, "self.mask.reshape(12, -1)",
                                       "self.mask.flatten(0, 1)")
        evidence_text = replace_exact(evidence_text, "summary.reshape(2, 6, 2, 16)",
                                       'summary.reshape(2, *self.ema["mean"].shape)')
        evidence_text = replace_exact(evidence_text, "shape = (groups + 1, groups, 2, 16)",
                                       'shape = (groups + 1, groups, *reference["mean"].shape[1:])')
        evidence_text = replace_exact(evidence_text, "def forward(self, white, training):",
                                       "def forward(self, white, training, pair):")
        evidence_text = replace_exact(evidence_text, "raw, bounded = gaussian(white, mu, var)",
                                       "mu, var = select_classes(mu, pair), select_classes(var, pair)\n"
                                       "        n = select_classes(n[..., None], pair).squeeze(-1)\n"
                                       "        raw, bounded = gaussian(white, mu, var)")
        evidence_text = replace_exact(evidence_text, "def disagreement(self, white, training):",
                                       "def disagreement(self, white, training, pair):")
        evidence_text = replace_exact(evidence_text, "nll = (white[:, None, None, :] - mu).square()",
                                       "mu, variance = select_comparison(mu, pair), select_comparison(variance, pair)\n"
                                       "        nll = (white[:, None, None, :] - mu).square()")
    evidence = archived_definitions(
        evidence_path, ("stabilize", "gaussian", "ClassEvidence", "AgreementEvidence"),
        source=evidence_text, E=config["evidence"], EPS=config["spectrum"]["eps"],
        require=require, select_classes=select_classes, select_comparison=select_comparison,
        module_name=f"e831_scmr_evidence_{suffix}",
    )

    g2_path = HISTORICAL_ROOT / "g2.py"
    g2_text = verified_text(g2_path, "3ae709aacd50198602cb9510f825ee5c12c0e8b334c1dc4c55f6237982084b1a")
    if adapted:
        g2_text = replace_exact(g2_text, "class SCMRDASRPE(nn.Module):",
                                 "class SCMRDASRPE(SCMRCheckpointState, nn.Module):")
        g2_text = replace_exact(g2_text, 'raw_llr, llr, reliability = self.statistics(d["white"], training)',
                                 'pair = class_pair(fixed_logits(head, cam.detach()))\n'
                                 '        raw_llr, llr, reliability = self.statistics(d["white"], training, pair)')
        g2_text = replace_exact(g2_text, 'self.statistics.disagreement(d["white"], training)',
                                 'self.statistics.disagreement(d["white"], training, pair)')
        g2_text = replace_exact(g2_text, "head_jacobian(head, cam)", "head_jacobian(head, cam, pair)")
        g2_text = replace_exact(g2_text, "d.update(raw_llr=", "d.update(class_pair=pair, raw_llr=")
    branch = archived_definitions(
        g2_path, ("SCMRDASRPE",), source=g2_text, CFG=config, M=config["margin_calibration"],
        E=config["evidence"], R=config["residual"], P=config["prompt"],
        MedicalSpectrum=spectrum.MedicalSpectrum, AgreementEvidence=evidence.AgreementEvidence,
        head_jacobian=spectrum.head_jacobian, normalized_weights=spectrum.normalized_weights,
        split_jacobian=spectrum.split_jacobian, fixed_logits=spectrum.fixed_logits, class_pair=class_pair,
        module_name=f"e831_scmr_g2_{suffix}", SCMRCheckpointState=SCMRCheckpointState,
    )
    return branch.SCMRDASRPE, spectrum, evidence


def fit_reference(cam, labels, rows, spectrum, fold):
    cfg = source_config()
    evidence_cfg = cfg["evidence"]
    legal = rows.detach().cpu().numpy().copy()
    y = labels.detach().cpu()[rows].long()
    nw = cfg["spectrum"]["eps"]
    tensors = (cam.detach().cpu()[rows].double() - spectrum["center"]) @ spectrum["basis"]
    white = tensors / (spectrum["variance"] + nw).sqrt()
    n_groups = evidence_cfg["inner_folds"]
    seed = evidence_cfg["archived_split_seed_base"] + fold
    inner = StratifiedKFold(n_splits=n_groups, shuffle=True, random_state=seed)
    group = np.zeros(len(rows), dtype=np.int64)
    for bank, (_, held_out) in enumerate(inner.split(np.zeros(len(rows)), y.numpy()), start=1):
        group[held_out] = bank
    bank_index = torch.zeros(len(cam), dtype=torch.long)
    bank_index[rows] = torch.from_numpy(group)
    classes = torch.arange(int(y.max()) + 1)
    mask = ((torch.from_numpy(group)[None, :] != torch.arange(n_groups + 1)[:, None])[:, None, :]
            & (y[None, None, :] == classes[None, :, None]))
    counts = mask.sum(-1)
    ddof = evidence_cfg["ddof"]
    require(bool((counts > ddof).all()), "A legal reference bank lacks a class")
    samples = [white[member] for member in mask.flatten(0, 1)]
    mean = torch.stack([sample.mean(0) for sample in samples]).reshape(*counts.shape, -1)
    variance = torch.stack([sample.var(0, correction=ddof) for sample in samples]).reshape_as(mean)
    _, _, evidence = scmr_components(len(cam), cam.shape[-1])
    _, stable = evidence.stabilize(variance, counts)
    second = variance * ((counts[..., None] - ddof) / counts[..., None].double()) + mean.square()
    nll = (white[:, None, :] - mean[group]).square() / stable[group] + stable[group].log()
    own = nll[torch.arange(len(y)), y]
    competitor = nll.masked_fill(y[:, None, None] == classes[None, :, None], torch.inf).amin(1)
    bound = evidence_cfg["llr_bound"]
    true_llr = bound * ((competitor - own) / (2 * bound)).tanh()
    skill = torch.stack([(true_llr[y == c] / 2).tanh().mean(0) for c in classes]).mean(0)
    return {"legal": legal, "labels": y.numpy().copy(), "group": group, "seed": seed,
            "counts": counts, "mask": mask, "bank": bank_index, "mean": mean, "second": second,
            "stable": stable, "calibration": .5 * (1 + skill), "h": white}


def _historical_objectives(module, head, capture, labels, rows):
    return scmr_objectives()("G2", head, capture, rows, labels[rows], module.spectrum.basis.to(capture["cam"]),
                             module.hard_initial[rows])


class SCMRCheckpointState:
    @property
    def basis(self):
        return self.spectrum.basis

    @property
    def ema_updates(self):
        return self.statistics.ema["updates"]

    historical_objectives = _historical_objectives

    def get_extra_state(self):
        return self.statistics.snapshot()

    def set_extra_state(self, snapshot):
        self.statistics.restore({
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in snapshot.items()
        })

    @torch.no_grad()
    def advance_ema(self, evaluated_cam):
        self.statistics.advance(self.statistics.observation(evaluated_cam))


def make_scmr(stats, prompt_path, nclass, seed):
    original = {"U": stats["basis"], "mu": stats["center"], "lambda": stats["variance"],
                "loglambda_normalized": stats["log_variance"], "nu_normalized": stats["disease_covariance"][0]}
    reference = {**stats["reference"], "bank": stats["bank_index"]}
    require(reference["mean"].shape[1] == nclass, "SCMR classes differ between the head and reference bank")
    cls, _, _ = scmr_components(len(reference["bank"]), stats["basis"].shape[0])
    prompt = Path(prompt_path).read_bytes()
    text = torch.load(io.BytesIO(prompt), map_location="cpu", weights_only=True)
    query_shape = tuple(text["q_medical"].shape)
    require(query_shape in ((8, 32), (12, 32)), "SCMR queries must be eight historical or twelve MMGPL vectors")
    require(tuple(text["q_anonymous"].shape) == query_shape, "Medical and anonymous query shapes differ")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if nclass == 3:
            cls = type("TadpoleAllPairSCMR", (cls,), {
                "forward": _all_pair_forward,
                "historical_objectives": _all_pair_objectives,
            })
        module = cls(original, text, [], reference)
    module.register_buffer("hard_initial", stats["hard_initial"].clone())
    if nclass == 3:
        module.register_buffer("hard_initial_pairs", stats["hard_initial_pairs"].clone())
    return module


@lru_cache(maxsize=1)
def scmr_objectives():
    """Preserve the source loss, including separate bounded candidate gains."""
    config = source_config()
    path = HISTORICAL_ROOT / "common/optimization.py"
    text = verified_text(path, SOURCE_HASHES["common/optimization.py"])
    text = replace_exact(text, "return 0.5 * (x[y == 0].mean() + x[y == 1].mean())",
                         "return torch.stack([x[y == c].mean() for c in y.unique(sorted=True)]).mean()")
    text = replace_exact(text, "sign = 1 - 2 * labels",
                         'pair = d["class_pair"][rows]\n'
                         '    sign = torch.where(labels == pair[:, 0], 1, -1)\n'
                         '    routed = (labels[:, None] == pair).any(dim=1)')
    text = replace_exact(text, "sign * margin(native)", "true_margin(native, labels)")
    text = replace_exact(text, "sign * margin(full)", "true_margin(full, labels)")
    text = replace_exact(text, "sign * margin(candidates)", "true_margin(candidates, labels)")
    text = replace_exact(text, 'balanced(focus * target_loss + cal["reliability_aux_weight"] * reliability_loss, labels)',
                         'balanced(routed * (focus * target_loss + cal["reliability_aux_weight"] * reliability_loss), labels)')
    _, spectral, _ = scmr_components(0, 144)
    return archived_definitions(
        path, ("balanced", "losses"), source=text, CFG=config, B=config["boundary"],
        l2_bound=spectral.l2_bound, native_logits=spectral.native_logits, true_margin=true_class_margin,
    ).losses
