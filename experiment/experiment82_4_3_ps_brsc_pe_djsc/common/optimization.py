"""Legal correction objectives and bounded additions to the historical gradients."""
from __future__ import annotations

from .history import CFG, require
from .spectral import B, EPS, fixed_logits, margin, native_logits, unit

import torch
from torch.nn import functional as F

AUX = CFG["auxiliary"]
ORIGINAL = ("gfr", "darg", "cam", "head")
NEW = ("prompt", "control")


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


def smooth_positive(value, scale):
    return scale * F.softplus(value / scale)


def losses(group, head, capture, rows, labels):
    # This API receives only legal indices and a label vector of those same rows.
    d = capture["details"]
    legal_cam = capture["cam"][rows]
    native = native_logits(head, legal_cam)
    objective = {"native": F.cross_entropy(native, labels)}
    off = (1 - 2 * labels) * margin(native.detach())
    full = (1 - 2 * labels) * margin(fixed_logits(head, legal_cam.detach() + d["delta"][rows]))
    gain = full - off
    error_likelihood = (-off / B["error_temperature"]).sigmoid()
    threatened = (-off.abs() / B["protect_scale"]).exp() * (1 - error_likelihood)
    gain_weight = error_likelihood * d["feasibility"][rows].detach()
    target = B["target_gain_fraction"] * d["capacity"][rows].detach()
    objective["gain"] = balanced(gain_weight * smooth_positive(target - gain, B["gain_temperature"]), labels)
    objective["protection"] = balanced(
        threatened * smooth_positive(-gain - B["protect_allowance"], B["protect_smoothing"]), labels)
    residual_penalty = d["ratio"][rows].square()
    ysign = 1 - 2 * labels
    if group == "G1":
        proposal = (unit(d["importance"][rows]) * d["action"][rows] * unit(d["jacobian_axis"][rows])).sum(-1)
        alignment = smooth_positive(-ysign * proposal, B["direction_scale"])
        objective["alignment"] = balanced(gain_weight * alignment, labels)
    else:
        evidence = d["evidence"][rows]
        confidence = d["support"][rows]
        direction = d["direction"][rows]
        directional_gain = (d["jacobian"][rows] * d["delta"][rows]).sum(-1)
        plausible = (ysign * evidence.detach() / CFG["evidence"]["evidence_temperature"]).sigmoid()
        alignment = smooth_positive(target * direction.detach().abs() -
                                    direction.detach() * directional_gain, B["gain_temperature"])
        objective["alignment"] = balanced(gain_weight * confidence.detach() * plausible * alignment, labels)
        residual_penalty = residual_penalty * (2 - confidence.detach())
    objective["regularization"] = balanced(residual_penalty, labels)
    return objective, {"gain": gain, "off": off, "gain_weight": gain_weight, "threatened": threatened}


def norm(grads):
    magnitudes = [torch.linalg.vector_norm(g.detach()) for g in grads if g is not None]
    return torch.linalg.vector_norm(torch.stack(magnitudes)).item() if magnitudes else 0.0


def gradients(loss, params, *, upstream=None):
    active = [(i, p) for i, p in enumerate(params) if p.requires_grad]
    result = [None] * len(params)
    found = torch.autograd.grad(loss, [p for _, p in active], grad_outputs=upstream,
                                retain_graph=True, allow_unused=True)
    for (i, _), grad in zip(active, found):
        result[i] = grad
    return result


def split(values, groups):
    start = 0
    result = {}
    for name, params in groups.items():
        result[name] = values[start:start + len(params)]
        start += len(params)
    return result


def apply_budget(groups, output, objectives, check=False):
    main_params = [p for name in ORIGINAL for p in groups[name]]
    branch_params = [p for name in NEW for p in groups[name]]
    cls = split(gradients(output["cls"], main_params + branch_params), groups)
    native = gradients(AUX["native"]["coefficient"] * objectives["native"], main_params)
    terms = {"native": split(native, {name: groups[name] for name in ORIGINAL})}
    for term in AUX:
        if term != "native":
            grad = gradients(AUX[term]["coefficient"] * objectives[term], branch_params)
            terms[term] = split(grad, {name: groups[name] for name in NEW})
    output["total"].backward()
    stats = {}
    finite = []

    with torch.no_grad():
        for family, params in groups.items():
            main_norm = norm(cls[family])
            historical = [p.grad for p in params]
            family_terms = [term for term, scopes in terms.items() if family in scopes]
            combined = list(historical)
            detail = {"main": main_norm}
            total_cap = sum(AUX[term]["cap"] for term in family_terms) * main_norm
            for term in family_terms:
                raw = terms[term][family]
                raw_norm = norm(raw)
                cap = AUX[term]["cap"] * main_norm
                gain = min(1.0, 0.99998 * cap / raw_norm) if raw_norm and cap else 0.0
                # Audit the change represented by .grad after floating point addition.
                # If a tiny budget is below the current representable step, use zero.
                candidate = combined
                applied = 0.0
                for _ in range(3):
                    if gain == 0:
                        break
                    candidate = [old if extra is None else extra * gain if old is None
                                 else torch.add(old, extra, alpha=gain)
                                 for old, extra in zip(combined, raw)]
                    applied = norm([None if new is old else new if old is None else new - old
                                    for old, new in zip(combined, candidate)])
                    if applied <= cap:
                        break
                    gain *= 0.99 * cap / applied
                if applied > cap:
                    gain, candidate, applied = 0.0, combined, 0.0
                combined = candidate
                detail[term] = {"raw": raw_norm, "applied": applied, "scale": gain}
            total_applied = norm([None if new is old else new if old is None else new - old
                                  for old, new in zip(historical, combined)])
            if total_applied > total_cap:
                combined = historical
                total_applied = 0.0
                for term in family_terms:
                    detail[term]["applied"] = detail[term]["scale"] = 0.0
            detail["total_applied"] = total_applied
            stats[family] = detail
            for parameter, grad in zip(params, combined):
                parameter.grad = grad
                if grad is not None:
                    finite.append(torch.isfinite(grad).all())
    require(bool(torch.stack(finite).all()), "Nonfinite training gradient")
    if check:
        for family in NEW:
            require(all(v is not None and norm([v]) > 0 for v in cls[family]),
                    f"Actual full classifier has a disconnected {family} parameter")
        for family in ORIGINAL:
            require(stats[family]["native"]["raw"] > 0 and stats[family]["native"]["applied"] > 0,
                    f"Native loss does not update {family}")
    return stats


def inspect_paths(groups, output, capture, native):
    d = capture["details"]
    derivative, = torch.autograd.grad(output["cls"], d["delta"], retain_graph=True)
    original = {name: groups[name] for name in ORIGINAL if name != "head"}
    original_params = [p for params in original.values() for p in params]
    residual_grad = gradients(d["delta"], original_params + [capture["cam"]], upstream=derivative)
    by_family = {name: norm(grads) for name, grads in split(residual_grad[:-1], original).items()}
    by_family["cam144"] = norm(residual_grad[-1:])
    native_cam, = torch.autograd.grad(native, capture["cam"], retain_graph=True)
    branch = tuple(p for name in NEW for p in groups[name])
    residual_parameters = gradients(d["delta"], branch, upstream=derivative)
    require(min(by_family.values()) > 0 and norm([native_cam]) > 0
            and all(norm([g]) > 0 for g in residual_parameters), "Disconnected native or residual gradient path")
    return {"from_actual_class_loss_through_delta": by_family,
            "residual_class_derivative": norm([derivative]), "native_to_cam144": norm([native_cam]),
            "minimum_new_parameter_residual_gradient": min(norm([grad]) for grad in residual_parameters)}


def distributions(details, rows, targets, group):
    def selected(key):
        return details[key][rows].detach().float()

    gate = selected("gate")
    ratio = selected("ratio")
    data = [gate.mean(), gate.min(), gate.max(), selected("feasibility").mean(),
            ratio.mean(), ratio.max(), (ratio > CFG["active_residual_relative"]).float().mean()]
    keys = ["gate_mean", "gate_min", "gate_max", "feasibility_mean",
            "residual_mean", "residual_max", "residual_active"]
    action = selected("action" if group == "G1" else "direction")
    items = {"action_mean": action.mean(), "action_abs_mean": action.abs().mean(),
             "action_min": action.min(), "action_max": action.max(), "action_positive": (action > 0).float().mean()}
    if targets:
        gain, off = targets["gain"].detach(), targets["off"].detach()
        items.update({"actual_true_margin_gain": gain.mean(),
                      "correct_negative_share": ((off >= 0) & (gain < 0)).float().sum() / (off >= 0).sum().clamp_min(1),
                      "wrong_positive_share": ((off < 0) & (gain > 0)).float().sum() / (off < 0).sum().clamp_min(1)})
    if group == "G2":
        items.update({"evidence_mean": selected("evidence").mean(), "reliability_mean": selected("support").mean(),
                      "parallel_strength": selected("strength_parallel").mean(),
                      "perp_strength": selected("strength_perp").mean()})
    if group == "G1":
        items.update({"beta_mean": selected("beta").mean(), "semantic_stat_l1": (
            selected("w_sem") - selected("w_sp")).abs().sum(-1).mean()})
    return dict(zip(keys + list(items), torch.stack(data + list(items.values())).detach().cpu().tolist()))
