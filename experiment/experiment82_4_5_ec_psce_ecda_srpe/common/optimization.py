"""Twin-Path objectives and bounded tangent updates around the complete historical gradient."""
from __future__ import annotations

from .history import CFG, require
from .spectral import B, EPS, fixed_logits, l2_bound, margin, native_logits

import torch
from torch.nn import functional as F

AUX = CFG["auxiliary"]
CLASS = CFG["full_classification"]
ORIGINAL = ("gfr", "darg", "cam", "head")
NEW = ("prompt", "control")
ORIGINAL_LOG = ("historical", "raw", "projected", "total_applied", "raw_dot", "tangent_dot",
                "applied_dot", "dot_bound", "scale", "near_zero", "rounded_to_zero", "guard_reject",
                "update_norm", "update_dot_historical", "update_dot_extra")


def families(state, module):
    def owned(key):
        return tuple(p for group in state[key].param_groups for p in group["params"])

    head = tuple(state["cam"].Outputlayer.parameters())
    spectral = tuple(module.spectrum.parameters())
    result = {
        "gfr": owned("opt_base"), "darg": owned("opt_added"),
        "cam": tuple(p for p in owned("opt_cam") if p not in set(head)), "head": head,
        "prompt": spectral, "control": tuple(p for p in module.parameters() if p not in set(spectral)),
    }
    all_parameters = [p for v in result.values() for p in v]
    require(len(set(all_parameters)) == len(all_parameters) and all(result.values()), "Gradient family ownership differs")
    expected = set(state["model"].parameters()) | set(state["cam"].parameters()) | set(module.parameters())
    require(set(all_parameters) == expected, "Gradient family ownership is incomplete")
    return result


def balanced(x, y):
    return 0.5 * (x[y == 0].mean() + x[y == 1].mean())


@torch.no_grad()
def counterfactual(head, cam, axes, coordinates, labels):
    # Reuse current, detached native parameters; one head batch with 17 legal variants per row.
    per_axis = coordinates[:, :, None] * axes.T
    flat = per_axis.flatten(0, 1)
    scale = l2_bound(flat, cam.norm(dim=-1).repeat_interleave(axes.shape[1])).view(len(cam), -1, 1)
    masked = cam[:, None, :] - scale * per_axis
    inputs = torch.cat((cam[:, None], masked), dim=1)
    logits = fixed_logits(head, inputs.flatten(0, 1)).view(len(cam), -1, 2)
    signed = (logits[..., 0] - logits[..., 1]) * (1 - 2 * labels[:, None])
    diff = signed[:, :1] - signed[:, 1:]
    contribution = diff.clamp_min(0)
    valid = (signed[:, 0] > 0) & (contribution.amax(-1) > CFG["counterfactual"]["valid_contribution"])
    target = contribution / (contribution.sum(-1, keepdim=True) + EPS)
    return target, valid


def losses(group, head, capture, rows, labels, basis, hard_initial):
    d = capture["details"]
    cam = capture["cam"][rows]
    full = native_logits(head, capture["enhanced"][rows])
    sign = 1 - 2 * labels
    with torch.no_grad():
        off = sign * margin(fixed_logits(head, cam))
        current = (-off / B["error_temperature"]).sigmoid()
        focus = 1 - (1 - current) * (1 - B["initial_memory_mix"] * hard_initial)
        guard = (1 - current) * (-off.abs() / B["protect_scale"]).exp()
    gain = sign * margin(full) - off
    tau = B["gain_temperature"]
    shortfall = F.softplus(-gain / tau)
    gain_cost = shortfall
    diagnostics = {"focus_mean": focus.mean(), "current_focus_mean": current.mean(),
                   "initial_focus_mean": hard_initial.mean()}
    candidate_gains = {}
    if group == "G2":
        # Reuse the actual deterministic head for each independently bounded candidate.
        raw = torch.stack((d["raw_parallel"][rows], d["raw_perp"][rows]))
        shrink = l2_bound(raw, d["cam_norm"][rows]).unsqueeze(-1)
        candidates = native_logits(head, cam + raw * shrink)
        spectral_gain, complement_gain = sign * margin(candidates) - off
        z = CFG["dual_gain_weights"]
        gain_cost = (z["joint"] * shortfall + z["spectral"] * F.softplus(-spectral_gain / tau)
                     + z["complement"] * F.softplus(-complement_gain / tau))
        diagnostics.update(spectral_candidate_gain=spectral_gain.mean(), complement_candidate_gain=complement_gain.mean())
        candidate_gains = {"spectral": spectral_gain, "complement": complement_gain}

    objective = {"full": balanced(F.cross_entropy(full, labels, reduction="none"), labels),
                 "gain": balanced(focus * gain_cost, labels), "guard": balanced(guard * shortfall, labels),
                 "regularization": balanced(d["ratio"][rows].square(), labels)}
    if group == "G1":
        target, valid = counterfactual(head, cam, basis, d["z"][rows], labels)
        student = d["w_sem"][rows].clamp_min(EPS)
        row_kl = (torch.xlogy(target, target) - target * student.log()).sum(-1)
        objective["guidance"] = (row_kl * valid).sum() / valid.sum().clamp_min(1)
        diagnostics.update(cf_valid_fraction=valid.float().mean(), cf_kl=objective["guidance"])
    else:
        prior = d["direction"][rows].detach()
        support = d["support"][rows].detach()
        compatible = (sign * prior).clamp_min(0)
        objective["guidance"] = balanced(support.square() * compatible *
                                          F.softplus(-candidate_gains["spectral"] / tau), labels)
        diagnostics.update(compatible_evidence_mean=compatible.mean())
    return objective, {"gain": gain, "off": off, "focus": focus,
                       "candidate_gains": candidate_gains, "diagnostics": diagnostics}


def gradients(loss, params, *, upstream=None):
    active = [(i, p) for i, p in enumerate(params) if p.requires_grad]
    result = [None] * len(params)
    found = torch.autograd.grad(loss, [p for _, p in active], grad_outputs=upstream,
                                retain_graph=True, allow_unused=True)
    for (i, _), grad in zip(active, found):
        result[i] = grad
    return result


def split(values, groups):
    result, offset = {}, 0
    for name, params in groups.items():
        result[name] = values[offset:offset + len(params)]
        offset += len(params)
    return result


def vector(params, grads):
    return torch.cat([(p.new_zeros(p.numel()) if g is None else g.reshape(-1)) for p, g in zip(params, grads)])


def norm(grads):
    pieces = [torch.linalg.vector_norm(g.detach()) for g in grads if g is not None]
    return torch.linalg.vector_norm(torch.stack(pieces)).item() if pieces else 0.0


def inner(a, b):
    return (a * b).sum()


def cast_addend(base, displacement, cap, *, base_norm=None):
    """Reserve the parameter dtype's L2 rounding allowance before writing."""
    q = base.double()
    u = torch.finfo(base.dtype).eps / 2
    u64 = torch.finfo(q.dtype).eps / 2
    n = base.numel()
    rho = ((2 * n + 16) * u64) / (1 - (2 * n + 16) * u64)
    size = q.norm() if base_norm is None else base_norm
    room = (cap - (2 * u + 8 * rho) * (size + cap)).clamp_min(0)
    scale = (room / displacement.norm().clamp_min(1e-300)).clamp(max=1.)
    written = (q + displacement * scale).to(base.dtype)
    return written, written.double() - q, scale


def tangent_step(base, full_gradient, enabled):
    historical = base.double()
    size = historical.norm()
    safe = size > CLASS["historical_norm_floor"]
    anchor = historical / torch.where(safe, size, 1.)
    anchor = torch.where(safe, anchor, 0.)
    raw = full_gradient.double()
    raw_norm = raw.norm()
    raw_dot = inner(raw, anchor)
    tangent = raw - raw_dot * anchor
    tangent_norm = tangent.norm()
    projected_dot = inner(tangent, anchor)
    cap = CLASS["shared_cap"] * size
    written, extra, scale = cast_addend(base, tangent, cap, base_norm=size)
    added_norm = extra.norm()
    added_dot = inner(extra, anchor)

    u = torch.finfo(base.dtype).eps / 2
    u64 = torch.finfo(historical.dtype).eps / 2
    operation_count = 2 * base.numel() + 16
    rho = operation_count * u64 / (1 - operation_count * u64)
    cast_error = u * inner(anchor.abs(), historical.abs() + tangent.abs() * scale)
    dot_bound = cast_error + 8 * rho * (raw_norm + tangent_norm + size)
    good = torch.isfinite(written).all() & (added_norm <= cap) & (added_dot.abs() <= dot_bound)
    accept = safe & good & enabled
    result = torch.where(accept, written, base)
    actual = torch.where(accept, extra, 0.)
    actual_norm, actual_dot = actual.norm(), inner(actual, anchor)
    record = dict(zip(ORIGINAL_LOG[:12], (size, raw_norm, tangent_norm, actual_norm, raw_dot, projected_dot,
                    actual_dot, dot_bound, torch.where(accept, scale, 0.), (~safe).double(),
                    (safe & (tangent_norm > 0) & (actual_norm == 0) & enabled).double(),
                    (safe & ~good & enabled).double())))
    return result, record, anchor, actual


def auxiliary_terms(group, family):
    return tuple(k for k in AUX if k != "guidance" or group == "G2" or family == "prompt")


def apply_budget(group, groups, output, objectives, *, shared=True):
    all_params = [p for family in groups.values() for p in family]
    full = split(gradients(objectives["full"], all_params), groups)
    auxiliary = {}
    for term in AUX:
        affected = {name: groups[name] for name in NEW if term in auxiliary_terms(group, name)}
        params = [p for ps in affected.values() for p in ps]
        auxiliary[term] = split(gradients(AUX[term]["coefficient"] * objectives[term], params), affected)

    output["total"].backward()
    record, before, finite = {}, {}, []
    with torch.no_grad():
        for name, params in groups.items():
            full_vector = vector(params, full[name])
            if name in ORIGINAL:
                base_grads = [p.grad for p in params]
                base = vector(params, base_grads)
                result, record[name], unit_hist, added = tangent_step(base, full_vector, shared)
                parameters = vector(params, [p.detach() for p in params])
                before[name] = (parameters, unit_hist, added)
            else:
                base_grads = full[name]
                base = full_vector * CLASS["new_coefficient"]
                result = base
                magnitude = base.double().norm()
                entries = {}
                terms = auxiliary_terms(group, name)
                for term in terms:
                    raw = vector(params, auxiliary[term][name]).double()
                    limit = AUX[term]["cap"] * magnitude
                    candidate, delta, scale = cast_addend(result, raw, limit)
                    norm_added = delta.norm()
                    accept = (norm_added <= limit) & torch.isfinite(candidate).all()
                    result = torch.where(accept, candidate, result)
                    entries[term] = (raw.norm(), torch.where(accept, norm_added, 0.), torch.where(accept, scale, 0.))
                total = (result.double() - base.double()).norm()
                accepted = total <= magnitude * sum(AUX[k]["cap"] for k in terms)
                result = torch.where(accepted, result, base)
                record[name] = {"main": magnitude, "total_applied": torch.where(accepted, total, 0.),
                                **{k: tuple(torch.where(accepted, v, 0.0) if i else v for i, v in enumerate(item))
                                   for k, item in entries.items()}}
            finite.append(torch.isfinite(result).all())
            if name in ORIGINAL and not shared:
                continue
            offset = 0
            for p, old, extra in zip(params, base_grads, full[name]):
                count = p.numel()
                if p.requires_grad and (old is not None or extra is not None):
                    p.grad = result[offset:offset + count].view_as(p)
                offset += count
    require(bool(torch.stack(finite).all()), "Nonfinite applied optimizer gradients")
    return record, before


@torch.no_grad()
def optimizer_measurements(groups, budget, before):
    for family, (previous, anchor, extra) in before.items():
        ps = groups[family]
        current = vector(ps, [p.detach() for p in ps])
        step = current.double() - previous.double()
        budget[family].update(update_norm=step.norm(), update_dot_historical=inner(step, anchor),
                              update_dot_extra=inner(step, extra))
    # One transfer for all scalar budget and optimizer telemetry.
    schema, values = [], []
    for family, fields in budget.items():
        for name, value in fields.items():
            items = value if isinstance(value, tuple) else (value,)
            schema.append((family, name, len(items)))
            values.extend(items)
    packed = iter(torch.stack(values).cpu().tolist())
    converted = {k: {} for k in budget}
    for family, name, length in schema:
        converted[family][name] = next(packed) if length == 1 else dict(zip(
            ("raw", "applied", "scale"), (next(packed) for _ in range(length))))
    return converted


def inspect_paths(groups, capture, full):
    d = capture["details"]
    derivative, = torch.autograd.grad(full, d["delta"], retain_graph=True)
    residual_groups = {name: ps for name, ps in groups.items() if name != "head"}
    params = [p for ps in residual_groups.values() for p in ps]
    values = gradients(d["delta"], params + [capture["cam"]], upstream=derivative)
    trace = split(values[:-1], residual_groups)
    norms = {k: norm(items) for k, items in trace.items()}
    norms["cam144"] = norm(values[-1:])
    require(all(v > 0 for v in norms.values()) and
            all(norm([g]) > 0 for k in NEW for g in trace[k]), "Disconnected full-classification residual gradient")
    probes = [d["raw_parallel"]]
    if d["raw_perp"].requires_grad:
        probes.append(d["raw_perp"])
    raw_gradient = torch.autograd.grad(full, probes, retain_graph=True)
    directional_gradient = [float((g * x).sum(-1).norm().detach()) for g, x in zip(raw_gradient, probes)]
    require(all(x > 0 for x in directional_gradient), "Full CE does not respond to a real displacement component")
    return {"from_full_ce_through_delta": norms, "class_to_delta": norm([derivative]),
            "per_space_directional_gradient": directional_gradient,
            "min_new_parameter_residual_gradient": min(norm([g]) for k in NEW for g in trace[k])}


@torch.no_grad()
def distributions(details, rows, targets, group):
    def selected(k):
        return details[k][rows].detach().float()

    gate, ratio, action = selected("gate"), selected("ratio"), selected("action")
    q = torch.quantile(gate, torch.tensor([.10, .50, .90], device=gate.device))
    metrics = {"gate_mean": gate.mean(), "gate_p10": q[0], "gate_p50": q[1], "gate_p90": q[2],
               "gate_near_zero": (gate < CFG["near_zero_gate_threshold"]).float().mean(),
               "feasibility_mean": selected("feasibility").mean(), "residual_mean": ratio.mean(),
               "residual_max": ratio.max(), "residual_active": (ratio > CFG["active_residual_relative"]).float().mean(),
               "raw_above_cap": (selected("raw_ratio") > CFG["residual"]["fraction"]).float().mean(),
               "material_shrink": ((1 - selected("scale").squeeze(-1)) >= CFG["material_cap_attenuation"]).float().mean(),
               "action_abs_mean": action.abs().mean(), "action_positive": (action > 0).float().mean()}
    if targets:
        gain, off = targets["gain"].detach(), targets["off"].detach()
        wrong = off < 0
        metrics.update({"margin_gain": gain.mean(),
                        "correct_negative_share": (~wrong & (gain < 0)).float().sum() / (~wrong).sum().clamp_min(1),
                        "wrong_positive_share": (wrong & (gain > 0)).float().sum() / wrong.sum().clamp_min(1)})
        metrics.update({k: v.detach() for k, v in targets["diagnostics"].items()})
    if group == "G2":
        metrics.update({"evidence_mean": selected("evidence").mean(), "reliability_mean": selected("support").mean(),
                        "s_parallel_mean": action[:, 0].mean(), "s_perp_mean": action[:, 1].mean(),
                        "abs_s_parallel": action[:, 0].abs().mean(), "abs_s_perp": action[:, 1].abs().mean()})
    else:
        metrics["beta_mean"] = selected("beta").mean()
        metrics["prompt_contrast_abs_mean"] = selected("prompt_contrast").abs().mean()
    return dict(zip(metrics, torch.stack(list(metrics.values())).cpu().tolist()))
