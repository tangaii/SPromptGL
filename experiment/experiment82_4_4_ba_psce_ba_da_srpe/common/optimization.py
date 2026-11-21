"""Twin-path classifier objectives and bounded, aligned optimizer gradients."""
from __future__ import annotations

from .history import CFG, require
from .spectral import B, EPS, fixed_logits, l2_bound, margin, native_logits

import torch
from torch.nn import functional as F

AUX = CFG["auxiliary"]
CLASS = CFG["full_classification"]
ORIGINAL = ("gfr", "darg", "cam", "head")
NEW = ("prompt", "control")
ORIGINAL_LOG = ("main", "total_applied", "raw", "projected", "cos_before", "cos_projected",
                "cos_applied", "scale", "conflict", "rounded_to_zero")


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


def losses(group, head, capture, rows, labels, basis):
    d = capture["details"]
    cam = capture["cam"][rows]
    full = native_logits(head, capture["enhanced"][rows])
    with torch.no_grad():
        off = (1 - 2 * labels) * margin(fixed_logits(head, cam))
        focus = (-off / B["error_temperature"]).sigmoid()
        guard = (1 - focus) * (-off.abs() / B["protect_scale"]).exp()
    gain = (1 - 2 * labels) * margin(full) - off
    shortfall = F.softplus(-gain / B["gain_temperature"])
    objective = {"full": balanced(F.cross_entropy(full, labels, reduction="none"), labels),
                 "gain": balanced(focus * shortfall, labels),
                 "guard": balanced(guard * shortfall, labels),
                 "regularization": balanced(d["ratio"][rows].square(), labels)}
    if group == "G1":
        target, valid = counterfactual(head, cam, basis, d["z"][rows], labels)
        student = d["w_sem"][rows].clamp_min(EPS)
        row_kl = (torch.xlogy(target, target) - target * student.log()).sum(-1)
        objective["guidance"] = (row_kl * valid).sum() / valid.sum().clamp_min(1)
        diagnostic = {"cf_valid_fraction": valid.float().mean(), "cf_kl": objective["guidance"]}
    else:
        prior = d["direction"][rows].detach()
        support = d["support"][rows].detach()
        compatible = ((1 - 2 * labels) * prior).clamp_min(0)
        aligned = (d["jacobian"][rows] * d["delta"][rows]).sum(-1) * prior
        objective["guidance"] = balanced(support.square() * compatible *
                                          F.softplus(-aligned / B["gain_temperature"]), labels)
        diagnostic = {"compatible_evidence_mean": compatible.mean()}
    return objective, {"gain": gain, "off": off, **diagnostic}


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


def capped_add(base, direction, cap, anchor=None):
    """Measure the realized float32 addend with float64, including its anchor component."""
    native = base.double()
    tiny = torch.finfo(native.dtype).tiny
    direction_norm = direction.norm()
    guard = 0.99998
    initial = (guard * cap / direction_norm.clamp_min(tiny)).clamp(max=1.0)
    move = direction * initial
    if anchor is not None:
        # Reserve the rounding error in the anchor dot product. The final
        # represented gradient is also checked, including when the cap is tiny.
        u32 = torch.finfo(base.dtype).eps / 2
        reserve = 4 * u32 * inner(anchor.abs(), native.abs() + move.abs())
        move = move + (reserve - inner(move, anchor)).clamp_min(0) * anchor
    shrink = (guard * cap / move.norm().clamp_min(tiny)).clamp(max=1.0)
    candidate = (native + shrink * move).to(base.dtype)
    realized = candidate.double() - native
    accepted = (direction_norm > 0) & (cap > 0) & (realized.norm() <= cap) & torch.isfinite(candidate).all()
    if anchor is not None:
        accepted = accepted & (inner(realized, anchor) >= 0)
    combined = torch.where(accepted, candidate, base)
    extra = combined.double() - native
    return combined, extra, torch.where(accepted, initial * shrink, 0.0), ~accepted


def auxiliary_terms(group, family):
    return tuple(k for k in AUX if k != "guidance" or group == "G2" or family == "prompt")


def apply_budget(group, groups, output, objectives, *, shared=True):
    original_params = [p for name in ORIGINAL for p in groups[name]]
    added_params = [p for name in NEW for p in groups[name]]
    native_cls = split(gradients(output["cls"], original_params), {name: groups[name] for name in ORIGINAL})
    full = split(gradients(objectives["full"], original_params + added_params), groups)
    auxiliary = {}
    for term in AUX:
        affected = {name: groups[name] for name in NEW if term in auxiliary_terms(group, name)}
        params = [p for ps in affected.values() for p in ps]
        auxiliary[term] = split(gradients(AUX[term]["coefficient"] * objectives[term], params), affected)

    output["total"].backward()
    record, finite = {}, []
    with torch.no_grad():
        for name, params in groups.items():
            full_vector = vector(params, full[name])
            if name in ORIGINAL:
                base_grad = [p.grad for p in params]
                base = vector(params, base_grad)
                native = vector(params, native_cls[name]).double()
                magnitude = native.norm()
                unit_native = native / magnitude.clamp_min(torch.finfo(native.dtype).tiny)
                raw = full_vector.double() * CLASS["shared_coefficient"]
                raw_norm = raw.norm()
                component = inner(unit_native, raw)
                projected = raw - component.clamp_max(0) * unit_native
                pr_norm = projected.norm()
                cap = CLASS["shared_cap"] * magnitude if shared else magnitude * 0
                result, added, scale, rejected = capped_add(base, projected, cap, anchor=unit_native)
                added_norm = added.norm()
                safe_cos = lambda product, size: product / size.clamp_min(torch.finfo(raw.dtype).tiny)
                record[name] = dict(zip(ORIGINAL_LOG, (magnitude, added_norm, raw_norm, pr_norm,
                    safe_cos(component, raw_norm), safe_cos(inner(projected, unit_native), pr_norm),
                    safe_cos(inner(added, unit_native), added_norm), scale, (component < 0).double(),
                    ((added_norm == 0) & (cap > 0) & (pr_norm > 0)).double())))
            else:
                base_grad = full[name]
                base = full_vector * CLASS["new_coefficient"]
                result = base
                magnitude = base.double().norm()
                terms = auxiliary_terms(group, name)
                entries = {}
                for term in terms:
                    raw = vector(params, auxiliary[term][name]).double()
                    result, delta, scale, _ = capped_add(result, raw, AUX[term]["cap"] * magnitude)
                    entries[term] = (raw.norm(), delta.norm(), scale)
                total = (result.double() - base.double()).norm()
                accepted = total <= magnitude * sum(AUX[k]["cap"] for k in terms)
                result = torch.where(accepted, result, base)
                record[name] = {"main": magnitude, "total_applied": torch.where(accepted, total, 0.0),
                                **{k: tuple(torch.where(accepted, v, 0.0) if i else v for i, v in enumerate(item))
                                   for k, item in entries.items()}}
            finite.append(torch.isfinite(result).all())
            if name in ORIGINAL and not shared:
                continue
            offset = 0
            for p, old, extra in zip(params, base_grad, full[name]):
                count = p.numel()
                if p.requires_grad and (old is not None or extra is not None):
                    p.grad = result[offset:offset + count].view_as(p)
                offset += count

    require(bool(torch.stack(finite).all()), "Nonfinite applied optimizer gradients")
    # One host synchronization for all float64 budget diagnostics.
    schema, values = [], []
    for family, fields in record.items():
        for name, value in fields.items():
            items = value if isinstance(value, tuple) else (value,)
            schema.append((family, name, len(items)))
            values.extend(items)
    packed = iter(torch.stack(values).cpu().tolist())
    converted = {k: {} for k in record}
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
        metrics.update({k: v.detach() for k, v in targets.items() if k not in ("gain", "off")})
    if group == "G2":
        metrics.update({"evidence_mean": selected("evidence").mean(), "reliability_mean": selected("support").mean(),
                        "s_parallel_mean": action[:, 0].mean(), "s_perp_mean": action[:, 1].mean()})
    else:
        metrics["beta_mean"] = selected("beta").mean()
    return dict(zip(metrics, torch.stack(list(metrics.values())).cpu().tolist()))
