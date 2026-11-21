#!/usr/bin/env python3
"""Independent E81.8 checkpoint, metric, paired, and mechanism audit."""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import run

CFG = run.CFG
RESULTS = run.RESULTS
HIST_E744 = run.HIST_E744
HIST_E8172 = run.HIST_E8172
E74_GROUP = run.E74_GROUP
GROUPS = run.GROUPS
METRICS = run.METRICS
METRIC_NAMES = {
    "acc": "ACC", "hard_auc": "Hard-AUC/BA", "prob_auc": "Prob-AUC",
    "sen": "SEN-ASD", "spe": "SPE-TD", "asd_f1": "ASD-F1",
}


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _manual_metrics(logp, labels):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
    values = np.asarray(logp, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    shifted = values - values.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    prediction = values.argmax(axis=1).astype(np.int64)
    cm = confusion_matrix(labels, prediction, labels=[0, 1])
    result = {
        "acc": float(accuracy_score(labels, prediction)),
        "hard_auc": float(balanced_accuracy_score(labels, prediction)),
        "prob_auc": float(roc_auc_score((labels == 0).astype(np.int64), probability[:, 0])),
        "sen": float(recall_score(labels, prediction, pos_label=0, zero_division=0)),
        "spe": float(recall_score(labels, prediction, pos_label=1, zero_division=0)),
        "asd_f1": float(f1_score(labels, prediction, pos_label=0, zero_division=0)),
        "prediction": prediction,
        "probability": probability,
        "confusion": {"asd_asd": int(cm[0, 0]), "asd_td": int(cm[0, 1]),
                      "td_asd": int(cm[1, 0]), "td_td": int(cm[1, 1])},
    }
    return result


def _summary(values):
    import numpy as np
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "population_sd": float(array.std(ddof=0))}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _correction(source, target, labels):
    import numpy as np
    source = np.asarray(source, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    corrected = (source != labels) & (target == labels)
    harmed = (source == labels) & (target != labels)
    return {
        "corrected": int(corrected.sum()),
        "harmed": int(harmed.sum()),
        "net": int(corrected.sum() - harmed.sum()),
        "asd_corrected": int((corrected & (labels == 0)).sum()),
        "asd_harmed": int((harmed & (labels == 0)).sum()),
        "td_corrected": int((corrected & (labels == 1)).sum()),
        "td_harmed": int((harmed & (labels == 1)).sum()),
    }


def _normalize_metric_value(value):
    value = float(value)
    return value / 100.0 if abs(value) > 1.0 else value


def _historical_e8172():
    rows = [row for row in _read_csv(HIST_E8172 / "results/fold_metrics.csv") if row.get("group") == "G1"]
    if len(rows) != 10:
        raise RuntimeError("E81.7.2-G1 history must contain ten fold rows")
    return {int(row["fold"]): {metric: _normalize_metric_value(row[metric]) for metric in METRICS}
            for row in rows}


def _load_current_fold(e744, legacy, group, fold, kind):
    import torch
    state, warmup, identity = e744.prepare(fold)
    state["fold"] = fold
    state["group"] = group
    checkpoint_path = run.CKPTS / f"{group}_f{fold:02d}.{kind.lower()}.pt"
    payload = run.torch_load(checkpoint_path)
    if payload.get("schema") != "E81.8-DUA-MSR-PSRR-v1" or payload.get("kind") != kind:
        raise RuntimeError(f"invalid {kind} checkpoint schema: {checkpoint_path}")
    if payload.get("group") != group or int(payload.get("fold", -1)) != fold:
        raise RuntimeError(f"{kind} checkpoint group/fold identity mismatch: {checkpoint_path}")
    if payload.get("source_sha256") != run.source_digest():
        raise RuntimeError(f"{kind} checkpoint frozen source SHA mismatch: {checkpoint_path}")
    if payload.get("identity") != identity or payload.get("warmup_sha256") != warmup["warmup_sha256"]:
        raise RuntimeError(f"{kind} checkpoint does not match E74.4 prepare(fold={fold})")
    e744.load_history().load_bundle(state, payload["base_state"])
    module = run.make_module(group, fold, torch.device("cuda:0"), run.prompt_payload()["semantic_basis_float64"])
    module.load_state_dict(payload["relation_state"], strict=True)
    parameters = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=CFG["base"]["optimizers"]["relation"]["lr"],
                                  weight_decay=CFG["base"]["optimizers"]["relation"]["weight_decay"])
    optimizer.load_state_dict(payload["optimizer_relation"])
    relation_ids = {id(parameter) for group_state in optimizer.param_groups for parameter in group_state["params"]}
    if relation_ids != {id(parameter) for parameter in parameters}:
        raise RuntimeError(f"{kind} relation optimizer lost or gained trainable parameters")
    legacy.rng_runtime(e744).restore_rng(payload["rng"])
    return state, module, optimizer, payload, identity, checkpoint_path


def _evaluate(e744, state, module, intervention="full", *, gradients=False):
    import numpy as np
    import torch
    fold = int(state["fold"])
    hook = run.OutputPreHook(state, module)
    module.audit_edge_overrides = ()
    hook.configure(training=False, intervention=intervention)
    context = torch.enable_grad() if gradients else torch.no_grad()
    try:
        with context:
            output = e744.forward(state, E74_GROUP, training=False)
        if hook.calls != 1 or hook.capture is None:
            raise RuntimeError(f"original OutputLayer call count is not one at fold {fold}")
        test_ids = np.asarray(state["test_np"], dtype=np.int64)
        labels = state["y"].detach().cpu().numpy().astype(np.int64)[test_ids]
        logp = output["logp"].detach().cpu().numpy()[test_ids]
        result = _manual_metrics(logp, labels)
        all_logp = output["logp"]
        capture = hook.capture
        return {
            "metrics": {metric: result[metric] for metric in METRICS},
            "prediction": result["prediction"],
            "probability": result["probability"],
            "confusion": result["confusion"],
            "labels": labels,
            "test_ids": test_ids,
            "logp": all_logp,
            "capture": capture,
            "output": output,
            "hook": hook,
        }
    except BaseException:
        hook.close()
        raise


def _close_evaluation(result):
    result["hook"].close()


def _audit_darg(e744, legacy, fold):
    import numpy as np
    import torch
    np_core = np.core
    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)
    state, warmup, identity = e744.prepare(fold)
    state["fold"] = fold
    checkpoint_path = HIST_E744 / ".runtime/checkpoints" / f"{E74_GROUP}_f{fold:02d}.best.pt"
    compat = json.loads((HIST_E8172 / ".runtime/darg_best_compat.json").read_text(encoding="utf-8"))[str(fold)]
    if run.sha_file(checkpoint_path) != compat.get("checkpoint_sha256"):
        raise RuntimeError(f"historical DARG Best checkpoint SHA mismatch fold={fold}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("group") != E74_GROUP or int(payload.get("fold", -1)) != fold or payload.get("identity") != identity:
        raise RuntimeError(f"historical DARG Best checkpoint identity mismatch fold={fold}")
    e744.load_history().load_bundle(state, payload["best_state"])
    output = e744.forward(state, E74_GROUP, training=False)
    labels = state["y"].detach().cpu().numpy().astype(np.int64)[state["test_np"]]
    logp = output["logp"].detach().cpu().numpy()[state["test_np"]]
    metrics = _manual_metrics(logp, labels)
    if not np.array_equal(metrics["prediction"], np.asarray(payload["test_prediction"], dtype=np.int64)):
        raise RuntimeError(f"historical DARG prediction could not be independently reproduced fold={fold}")
    if not np.allclose(metrics["probability"], np.asarray(payload["test_probability"], dtype=np.float64), atol=1e-7, rtol=0):
        raise RuntimeError(f"historical DARG probability could not be independently reproduced fold={fold}")
    if not np.array_equal(metrics["prediction"], np.asarray(compat["test_prediction"], dtype=np.int64)):
        raise RuntimeError(f"historical DARG compatibility prediction mismatch fold={fold}")
    if not np.allclose(metrics["probability"], np.asarray(compat["test_probability"], dtype=np.float64), atol=1e-7, rtol=0):
        raise RuntimeError(f"historical DARG compatibility probability mismatch fold={fold}")
    return {
        "metrics": {metric: metrics[metric] for metric in METRICS},
        "prediction": metrics["prediction"], "probability": metrics["probability"],
        "labels": labels, "test_ids": state["test_np"].copy(),
        "confusion": metrics["confusion"],
        "checkpoint_sha256": run.sha_file(checkpoint_path),
        "best_epoch_1based": int(payload["best_epoch_1based"]),
    }


def _verify_saved_prediction(payload, evaluation):
    import numpy as np
    prediction = np.asarray(payload["prediction"], dtype=np.int64)
    probability = np.asarray(payload["probability"], dtype=np.float64)
    if not np.array_equal(prediction, evaluation["prediction"]):
        raise RuntimeError(f"saved {payload['kind']} prediction differs from independent recomputation")
    if not np.allclose(probability, evaluation["probability"], atol=1e-7, rtol=0):
        raise RuntimeError(f"saved {payload['kind']} probability differs from independent recomputation")
    for metric in METRICS:
        if abs(float(payload["current_metrics"][metric]) - evaluation["metrics"][metric]) > 2e-8:
            raise RuntimeError(f"saved {payload['kind']} {metric} differs from independent recomputation")


def _replay_latest(e744, legacy, group, fold, state, module, optimizer, payload):
    import numpy as np
    import torch
    runtime = legacy.rng_runtime(e744)
    hook = run.OutputPreHook(state, module)

    def restore():
        e744.load_history().load_bundle(state, payload["base_state"])
        state["fold"] = fold
        state["group"] = group
        module.load_state_dict(payload["relation_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_relation"])
        runtime.restore_rng(payload["rng"])

    def step():
        return run.training_step(
            e744, legacy, state, module, optimizer, hook, group,
            epoch=int(payload["current_epoch_1based"]) + 1,
            fold=fold,
            gradient_path_probe=False,
        )

    restore()
    first = step()
    first_base = e744.load_history().state_bundle(state)
    first_module = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    first_optimizer = optimizer.state_dict()
    first_rng = runtime.pack_rng()
    restore()
    second = step()
    second_base = e744.load_history().state_bundle(state)
    second_module = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    second_optimizer = optimizer.state_dict()
    second_rng = runtime.pack_rng()
    for name, left, right in (("base parameters and optimizers", first_base, second_base),
                              ("relation parameters", first_module, second_module),
                              ("relation optimizer", first_optimizer, second_optimizer),
                              ("RNG", first_rng, second_rng)):
        if name == "RNG":
            equal, where = legacy.tensor_tree_equal(left, right)
        else:
            equal, where = legacy.tensor_tree_equal(left, right)
        if not equal:
            raise RuntimeError(f"Latest replay mismatch in {name} at {where}")
    if first["total_loss"] != second["total_loss"] or first["metrics"] != second["metrics"]:
        raise RuntimeError("Latest replay changed next-step loss or metrics")
    if not np.array_equal(first["prediction"], second["prediction"]):
        raise RuntimeError("Latest replay changed next-step predictions")
    hook.close()
    return {
        "pass": True,
        "next_step_loss_exact": True,
        "next_step_metrics_exact": True,
        "base_and_relation_parameters_exact": True,
        "optimizer_states_exact": True,
        "rng_exact": True,
        "next_step_epoch_1based": int(payload["current_epoch_1based"]) + 1,
    }


def _intervention_metrics(e744, state, module, intervention, reference):
    import numpy as np
    current = _evaluate(e744, state, module, intervention)
    try:
        difference = np.abs(current["probability"][:, 0] - reference["probability"][:, 0])
        flips = int(np.sum(current["prediction"] != reference["prediction"]))
        return current, {"probability_mae": float(difference.mean()), "hard_flips": flips,
                         "acc": current["metrics"]["acc"], "ba": current["metrics"]["hard_auc"]}
    except BaseException:
        _close_evaluation(current)
        raise


def _prompt_summary(details, prefix):
    import torch
    result = {}
    for label in ("td", "asd"):
        beta = details[f"beta_{label}"].detach().float()
        entropy = -(beta.clamp_min(1e-12) * beta.clamp_min(1e-12).log()).sum(dim=-1)
        result[f"{prefix}_{label}_beta_entropy_mean"] = float(entropy.mean().cpu())
        result[f"{prefix}_{label}_beta_variance_across_receivers"] = float(beta.var(dim=1, unbiased=False).mean().cpu())
        result[f"{prefix}_{label}_beta_variance_across_prompts"] = float(beta.var(dim=-1, unbiased=False).mean().cpu())
    return result


def _relation_summary(details, prefix):
    import numpy as np
    import torch
    result = {}
    for label in ("td", "asd"):
        attention = details[f"attention_{label}"].detach().float()
        real = attention[..., :15]
        null = attention[..., 15]
        result[f"{prefix}_{label}_effective_neighbors"] = float((real > 0).sum(dim=-1).float().mean().cpu())
        result[f"{prefix}_{label}_null_mass_mean"] = float(null.mean().cpu())
        result[f"{prefix}_{label}_null_selected_rate"] = float((null >= real.max(dim=-1).values).float().mean().cpu())
    if "relation_support_difference" in details:
        result[f"{prefix}_asd_td_support_disagreement"] = float(details["relation_support_difference"].float().mean().cpu())
    residual_ratio = details["residual_ratio"].detach().float().cpu().numpy()
    result.update({
        f"{prefix}_residual_ratio_mean": float(residual_ratio.mean()),
        f"{prefix}_residual_ratio_p95": float(np.percentile(residual_ratio, 95)),
        f"{prefix}_residual_ratio_max": float(residual_ratio.max()),
    })
    return result


def _dua_proxy(e744, state, module, legal_rows):
    import numpy as np
    import torch
    hook = run.OutputPreHook(state, module)
    hook.configure(training=False, intervention="full")
    try:
        with torch.enable_grad():
            output = e744.forward(state, E74_GROUP, training=False)
            if hook.calls != 1:
                raise RuntimeError("DUA post-training utility must use one OutputLayer forward")
            labels = state["y"].long()
            legal = torch.as_tensor(legal_rows, dtype=torch.long, device=labels.device)
            margins = output["logp"].gather(1, labels[:, None]).squeeze(1) - output["logp"].gather(
                1, (1 - labels)[:, None]
            ).squeeze(1)
            gradient = torch.autograd.grad(margins[legal].sum(), hook.capture["z_final"], retain_graph=False)[0]
            slots = hook.capture["z_cam"].detach().reshape(-1, 16, 9).float()
            local_gradient = gradient.detach().reshape(-1, 16, 9).float()
            differences = slots[:, module.neighbors] - slots.unsqueeze(2)
            expanded_gradient = local_gradient.unsqueeze(2).expand_as(differences)
            dot = (expanded_gradient * differences).sum(dim=-1)
            denominator = expanded_gradient.norm(dim=-1) * differences.norm(dim=-1)
            utility = torch.where(denominator > 1e-12, dot / denominator.clamp_min(1e-12), torch.zeros_like(dot))
            utility = utility.clamp(-1.0, 1.0).detach()
            capture = hook.capture
            return {
                "utility": utility,
                "margin": margins.detach(),
                "attention_td": capture["attention_td"].detach(),
                "attention_asd": capture["attention_asd"].detach(),
                "z_cam": capture["z_cam"].detach(),
            }
    finally:
        hook.close()


def _dua_edge_deletion_audit(e744, state, module, legal_rows, full_eval):
    import numpy as np
    import torch
    proxy = _dua_proxy(e744, state, module, legal_rows)
    labels = state["y"].detach().cpu().numpy().astype(np.int64)
    utility = proxy["utility"].cpu().numpy()
    attention = {0: proxy["attention_asd"].cpu().numpy(), 1: proxy["attention_td"].cpu().numpy()}
    neighbors = module.neighbors.detach().cpu().numpy()
    buckets = {"selected_positive": [], "selected_negative": [],
               "unselected_positive": [], "unselected_negative": []}
    for patient in legal_rows:
        class_index = int(labels[patient])
        probabilities = attention[class_index][patient]
        for receiver in range(16):
            for position, sender in enumerate(neighbors[receiver]):
                value = float(utility[patient, receiver, position])
                if abs(value) < 0.15:
                    continue
                selected = probabilities[receiver, position] > 0.0
                key = ("selected_" if selected else "unselected_") + ("positive" if value > 0 else "negative")
                buckets[key].append((abs(value), int(patient), receiver, int(sender), value, selected, class_index))
    choices = []
    for key in buckets:
        choices.extend(sorted(buckets[key], reverse=True)[:2])
    records = []
    base_margin = proxy["margin"].detach().cpu().numpy()
    for _, patient, receiver, sender, value, selected, class_index in choices:
        module.audit_edge_overrides = ((patient, receiver, sender, class_index),)
        deleted = _evaluate(e744, state, module, "full")
        try:
            labels_device = state["y"].long()
            before = base_margin[patient]
            after_logs = deleted["output"]["logp"][patient]
            after = float((after_logs[labels_device[patient]] - after_logs[1 - labels_device[patient]]).detach().cpu())
            contribution = before - after
            records.append({
                "patient": patient, "receiver": receiver, "sender": sender,
                "class_index": class_index, "u": value, "selected": selected,
                "correct_margin_contribution": contribution,
                "direction_agreement": bool(value * contribution > 0),
            })
        finally:
            _close_evaluation(deleted)
    module.audit_edge_overrides = ()
    selected_effect = [row["correct_margin_contribution"] for row in records if row["selected"]]
    unselected_effect = [row["correct_margin_contribution"] for row in records if not row["selected"]]
    valid = [row for row in records if abs(row["u"]) >= 0.15 and abs(row["correct_margin_contribution"]) > 1e-12]
    correlation = float(np.corrcoef([row["u"] for row in valid],
                                    [row["correct_margin_contribution"] for row in valid])[0, 1]) if len(valid) >= 2 else float("nan")
    return {
        "representative_edges": len(records),
        "selected_edges": sum(row["selected"] for row in records),
        "unselected_edges": sum(not row["selected"] for row in records),
        "direction_consistency": (sum(row["direction_agreement"] for row in valid) / len(valid)) if valid else float("nan"),
        "u_vs_actual_margin_contribution_correlation": correlation,
        "selected_mean_actual_contribution": float(np.mean(selected_effect)) if selected_effect else float("nan"),
        "unselected_mean_actual_contribution": float(np.mean(unselected_effect)) if unselected_effect else float("nan"),
        "edge_records": records,
        "utility_mean_train": float(utility[np.asarray(legal_rows)].mean()),
        "utility_positive_fraction_train": float((utility[np.asarray(legal_rows)] > 0).mean()),
    }, proxy


def _g2_stability_audit(e744, state, module, details, legal_rows):
    import torch
    ids = torch.as_tensor(legal_rows, dtype=torch.long, device=details["z_cam"].device)
    slots = details["z_cam"].detach()[ids].reshape(-1, 16, 9).float()
    scale = slots.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    coordinates = torch.arange(slots.numel(), dtype=torch.float32, device=slots.device).reshape_as(slots)
    perturbation = 0.01 * scale.view(1, 1, 9) * torch.sin(coordinates * 12.9898 + 0.37)
    with torch.no_grad():
        perturbed_td, perturbed_asd = module.relation_distributions(
            (slots + perturbation).reshape(-1, 144), grouping="medical"
        )
    from run import _js_divergence
    baseline_td = details["attention_td"][ids].detach()
    baseline_asd = details["attention_asd"][ids].detach()
    js = 0.5 * (_js_divergence(baseline_td, perturbed_td).mean()
                + _js_divergence(baseline_asd, perturbed_asd).mean())
    return float(js.cpu()), float((perturbation.norm() / slots.norm().clamp_min(1e-12)).cpu())


def _collect_mechanism(e744, legacy, group, fold, state, module, darg, full, residual_off):
    import numpy as np
    import torch
    legal = run.legal_train_rows(state, fold)
    test_ids = full["test_ids"]
    labels = full["labels"]
    result = {"group": group, "fold": fold, **_correction(darg["prediction"], residual_off["prediction"], labels)}
    result.update({
        "A_to_B_corrected": _correction(darg["prediction"], residual_off["prediction"], labels)["corrected"],
        "A_to_B_harmed": _correction(darg["prediction"], residual_off["prediction"], labels)["harmed"],
        "A_to_B_net": _correction(darg["prediction"], residual_off["prediction"], labels)["net"],
    })
    a_b = _correction(darg["prediction"], residual_off["prediction"], labels)
    b_c = _correction(residual_off["prediction"], full["prediction"], labels)
    a_c = _correction(darg["prediction"], full["prediction"], labels)
    for prefix, counts in (("A_to_B", a_b), ("B_to_C", b_c), ("A_to_C", a_c)):
        for key, value in counts.items():
            result[f"{prefix}_{key}"] = value
    details = full["capture"]
    result.update(_prompt_summary(details, "full"))
    result.update(_relation_summary(details, "full"))

    interventions = ("dua_score_off", "uniform_beta", "prompt_off") if group == "G1" else (
        "second_order_off", "high_frequency_off", "multiscale_off", "uniform_beta", "prompt_off"
    )
    intervention_results = {}
    for intervention in interventions:
        current, summary = _intervention_metrics(e744, state, module, intervention, full)
        intervention_results[intervention] = current
        for key, value in summary.items():
            result[f"{intervention}_{key}"] = value
        result[f"{intervention}_hard_flips"] = summary["hard_flips"]
        _close_evaluation(current)

    if group == "G1":
        score_delta_td = (details["scores_td"] - details["score_base_td"]).detach().float()
        score_delta_asd = (details["scores_asd"] - details["score_base_asd"]).detach().float()
        score_off = intervention_results["dua_score_off"]["capture"]
        support_td_full = details["attention_td"][..., :15] > 0
        support_asd_full = details["attention_asd"][..., :15] > 0
        support_td_base = score_off["attention_td"][..., :15] > 0
        support_asd_base = score_off["attention_asd"][..., :15] > 0
        result.update({
            "dua_score_delta_abs_mean": float(torch.cat((score_delta_td[..., :15].flatten(),
                                                          score_delta_asd[..., :15].flatten())).abs().mean().cpu()),
            "dua_relation_support_change_rate": float(torch.cat(((support_td_full ^ support_td_base).flatten(),
                                                                  (support_asd_full ^ support_asd_base).flatten())).float().mean().cpu()),
            "dua_relation_probability_mae": float(torch.cat(((details["attention_td"] - score_off["attention_td"]).flatten(),
                                                              (details["attention_asd"] - score_off["attention_asd"]).flatten())).abs().mean().cpu()),
        })
        deletion, proxy = _dua_edge_deletion_audit(e744, state, module, legal, full)
        for key, value in deletion.items():
            if key != "edge_records":
                result[f"utility_{key}"] = value
        corrected_mask = ((residual_off["prediction"] != labels) & (full["prediction"] == labels))
        corrected_ids = test_ids[corrected_mask]
        if corrected_ids.size:
            test_proxy = _dua_proxy(e744, state, module, corrected_ids)
            utility_test = test_proxy["utility"].detach().cpu().numpy()
            attention_true = {0: test_proxy["attention_asd"].detach().cpu().numpy(),
                              1: test_proxy["attention_td"].detach().cpu().numpy()}
            patient_utility = []
            patient_positive = []
            for patient in corrected_ids:
                class_index = int(state["y"][patient].item())
                selected_mask = attention_true[class_index][patient, :, :15] > 0
                selected_values = utility_test[patient][selected_mask]
                patient_utility.append(float(selected_values.mean()) if selected_values.size else 0.0)
                patient_positive.append(bool((selected_values > 0).any()) if selected_values.size else False)
            result["corrected_patient_mean_selected_u"] = float(np.mean(patient_utility))
            result["corrected_patients_with_positive_selected_u_fraction"] = float(np.mean(patient_positive))
        else:
            result["corrected_patient_mean_selected_u"] = float("nan")
            result["corrected_patients_with_positive_selected_u_fraction"] = float("nan")
        result["corrected_patient_count"] = int(corrected_ids.size)
    else:
        adj_error = torch.cat([
            (details[f"adjacency_{label}"].sum(dim=-1) - (1.0 - details[f"null_mass_{label}"])).abs().flatten()
            for label in ("td", "asd")
        ]).max()
        result.update({
            "first_order_norm_mean": float(torch.cat((details["first_order_td"].norm(dim=-1).flatten(),
                                                       details["first_order_asd"].norm(dim=-1).flatten())).mean().cpu()),
            "second_order_norm_mean": float(torch.cat((details["second_order_td"].norm(dim=-1).flatten(),
                                                        details["second_order_asd"].norm(dim=-1).flatten())).mean().cpu()),
            "high_frequency_norm_mean": float(torch.cat((details["high_frequency_td"].norm(dim=-1).flatten(),
                                                          details["high_frequency_asd"].norm(dim=-1).flatten())).mean().cpu()),
            "second_minus_first_norm_mean": float(torch.cat(((details["second_order_td"] - details["first_order_td"]).norm(dim=-1).flatten(),
                                                              (details["second_order_asd"] - details["first_order_asd"]).norm(dim=-1).flatten())).mean().cpu()),
            "null_absorbing_row_sum_max_error": float(adj_error.cpu()),
            "asd_td_relation_l1_mean": float((details["attention_asd"] - details["attention_td"]).abs().sum(dim=-1).mean().cpu()),
        })
        stability, perturb_ratio = _g2_stability_audit(e744, state, module, details, legal)
        result["stability_js_train_mean"] = stability
        result["stability_perturbation_rms_ratio"] = perturb_ratio
        corrected_mask = ((residual_off["prediction"] != labels) & (full["prediction"] == labels))
        corrected_as_test = int(corrected_mask.sum())
        result["second_order_dependent_corrections"] = int(np.sum(
            corrected_mask & (intervention_results["second_order_off"]["prediction"] != labels)
        ))
        result["high_frequency_dependent_corrections"] = int(np.sum(
            corrected_mask & (intervention_results["high_frequency_off"]["prediction"] != labels)
        ))
        result["full_local_corrections"] = corrected_as_test
    return result


def _paired_rows(metrics_by_group, darg_metrics, e8172_metrics):
    rows = []
    comparisons = []
    reference_map = {"DARG": darg_metrics, "E81.7.2-G1": e8172_metrics}
    for group in GROUPS:
        reference_map[group] = metrics_by_group[group]
    pairs = [("G1-DARG", metrics_by_group["G1"], darg_metrics),
             ("G2-DARG", metrics_by_group["G2"], darg_metrics),
             ("G1-E81.7.2-G1", metrics_by_group["G1"], e8172_metrics),
             ("G2-E81.7.2-G1", metrics_by_group["G2"], e8172_metrics),
             ("G2-G1", metrics_by_group["G2"], metrics_by_group["G1"])]
    for name, left, right in pairs:
        for subset, folds in (("all10", list(range(1, 11))),
                              ("exclude_fold2", [fold for fold in range(1, 11) if fold != 2])):
            for metric in METRICS:
                differences = [(left[fold][metric] - right[fold][metric]) * 100.0 for fold in folds]
                summary = _summary(differences)
                wins = sum(value > 1e-12 for value in differences)
                ties = sum(abs(value) <= 1e-12 for value in differences)
                row = {
                    "comparison": name, "subset": subset, "metric": METRIC_NAMES[metric],
                    "folds": len(folds), "mean_difference_pp": summary["mean"],
                    "population_sd_difference_pp": summary["population_sd"],
                    "W_T_L": f"{wins}/{ties}/{len(folds)-wins-ties}",
                    "fold_differences_pp": json.dumps({str(fold): left[fold][metric] * 100.0 - right[fold][metric] * 100.0
                                                       for fold in folds}, sort_keys=True),
                }
                rows.append(row)
                if metric in ("acc", "hard_auc"):
                    comparisons.append(row)
    return rows, comparisons


def audit_all():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    legacy = run.load_legacy_helpers()
    gpu = legacy.verify_gpu1_device()
    e744 = run.base_runner()
    frozen_source = run.source_digest()
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("source_sha256") != frozen_source:
        raise RuntimeError("independent audit source SHA does not match frozen preflight")

    metric_rows = []
    patient_rows = []
    mechanism_rows = []
    audit_records = []
    darg_metrics = {}
    darg_outputs = {}
    group_metrics = {group: {} for group in GROUPS}
    latest_replays = 0
    checkpoint_audits = 0
    start = time.time()

    for fold in range(1, 11):
        darg = _audit_darg(e744, legacy, fold)
        darg_metrics[fold] = darg["metrics"]
        darg_outputs[fold] = darg
        for group in GROUPS:
            state = module = optimizer = None
            evaluations = {}
            payloads = {}
            paths = {}
            identity = None
            for kind in ("Best", "Latest"):
                state, module, optimizer, payload, identity, checkpoint_path = _load_current_fold(
                    e744, legacy, group, fold, kind
                )
                payloads[kind] = payload
                paths[kind] = checkpoint_path
                evaluation = _evaluate(e744, state, module, "full")
                _verify_saved_prediction(payload, evaluation)
                if kind == "Best":
                    best_evaluation = evaluation
                    group_metrics[group][fold] = evaluation["metrics"]
                    best_hash = run.sha_file(checkpoint_path)
                    metric_rows.append({
                        "group": group, "fold": fold, "checkpoint_sha256": best_hash,
                        "best_epoch_1based": payload["current_epoch_1based"],
                        "test_n": len(evaluation["labels"]),
                        **{METRIC_NAMES[metric]: value * 100.0
                           for metric, value in evaluation["metrics"].items()},
                        **evaluation["confusion"],
                    })
                    for patient, label, prediction, probability in zip(
                        evaluation["test_ids"], evaluation["labels"],
                        evaluation["prediction"], evaluation["probability"],
                    ):
                        patient_rows.append({
                            "group": group, "fold": fold, "patient_zero_based": int(patient),
                            "label": int(label), "prediction": int(prediction),
                            "prob_asd": float(probability[0]), "prob_td": float(probability[1]),
                            "correct": bool(label == prediction), "best_epoch_1based": payload["current_epoch_1based"],
                        })
                else:
                    latest_hash = run.sha_file(checkpoint_path)
                    latest_replay = _replay_latest(e744, legacy, group, fold, state, module, optimizer, payload)
                    latest_replays += 1
                    audit_records.append({
                        "group": group, "fold": fold, "kind": kind,
                        "checkpoint_sha256": latest_hash,
                        "metrics_recomputed": True,
                        "outputlayer_call_count": evaluation["hook"].calls,
                        "latest_next_step_replay": latest_replay,
                    })
                _close_evaluation(evaluation)
                checkpoint_audits += 1

            state, module, optimizer, payload, identity, checkpoint_path = _load_current_fold(
                e744, legacy, group, fold, "Best"
            )
            full = _evaluate(e744, state, module, "full")
            residual_off = _evaluate(e744, state, module, "relation_off")
            labels = full["labels"]
            if not np.array_equal(full["test_ids"], darg["test_ids"]):
                raise RuntimeError(f"historical DARG/current test row order changed fold={fold}")
            mechanism = _collect_mechanism(e744, legacy, group, fold, state, module, darg, full, residual_off)
            mechanism_rows.append(mechanism)
            a_b = _correction(darg["prediction"], residual_off["prediction"], labels)
            b_c = _correction(residual_off["prediction"], full["prediction"], labels)
            a_c = _correction(darg["prediction"], full["prediction"], labels)
            for transition, counts in (("A_to_B", a_b), ("B_to_C", b_c), ("A_to_C", a_c)):
                audit_records.append({"group": group, "fold": fold, "kind": transition, **counts})
            _close_evaluation(full)
            _close_evaluation(residual_off)

    e8172_metrics = _historical_e8172()
    paired_rows, paired_gate_rows = _paired_rows(group_metrics, darg_metrics, e8172_metrics)
    _write_csv(RESULTS / "fold_metrics.csv", metric_rows)
    _write_csv(RESULTS / "patient_predictions.csv", patient_rows)
    _write_csv(RESULTS / "paired_comparisons.csv", paired_rows)
    _write_csv(RESULTS / "mechanism.csv", mechanism_rows)
    _write_csv(RESULTS / "checkpoint_audit.csv", audit_records)

    summary = {}
    for group in GROUPS:
        summary[group] = {}
        for metric in METRICS:
            all_values = [group_metrics[group][fold][metric] for fold in range(1, 11)]
            nine_values = [group_metrics[group][fold][metric] for fold in range(1, 11) if fold != 2]
            summary[group][metric] = {
                **_summary(all_values),
                "exclude_fold2_mean": float(np.mean(nine_values)),
                "exclude_fold2_population_sd": float(np.std(nine_values, ddof=0)),
            }
    summary["DARG"] = {}
    summary["E81.7.2-G1"] = {}
    for name, source in (("DARG", darg_metrics), ("E81.7.2-G1", e8172_metrics)):
        for metric in METRICS:
            all_values = [source[fold][metric] for fold in range(1, 11)]
            nine_values = [source[fold][metric] for fold in range(1, 11) if fold != 2]
            summary[name][metric] = {
                **_summary(all_values),
                "exclude_fold2_mean": float(np.mean(nine_values)),
                "exclude_fold2_population_sd": float(np.std(nine_values, ddof=0)),
            }

    total_mechanism = {}
    for group in GROUPS:
        records = [row for row in mechanism_rows if row["group"] == group]
        total_mechanism[group] = {}
        for stage in ("A_to_B", "B_to_C", "A_to_C"):
            fields = [f"{stage}_corrected", f"{stage}_harmed", f"{stage}_net",
                      f"{stage}_asd_corrected", f"{stage}_asd_harmed",
                      f"{stage}_td_corrected", f"{stage}_td_harmed"]
            totals = {key: sum(int(row.get(key, 0)) for row in records) for key in fields}
            total_mechanism[group][stage] = totals
    group_differences = {}
    for group in GROUPS:
        group_differences[group] = {}
        for metric in METRICS:
            diffs = [(group_metrics[group][fold][metric] - darg_metrics[fold][metric]) * 100.0
                     for fold in range(1, 11)]
            group_differences[group][metric] = {
                "mean_difference_pp": float(np.mean(diffs)),
                "population_sd_difference_pp": float(np.std(diffs, ddof=0)),
                "W_T_L": next(row["W_T_L"] for row in paired_gate_rows
                              if row["comparison"] == f"{group}-DARG" and row["subset"] == "all10"
                              and row["metric"] == METRIC_NAMES[metric]),
                "exclude_fold2_mean_difference_pp": next(row["mean_difference_pp"] for row in paired_rows
                                                         if row["comparison"] == f"{group}-DARG"
                                                         and row["subset"] == "exclude_fold2"
                                                         and row["metric"] == METRIC_NAMES[metric]),
            }

    paired_summary = {}
    for comparison in ("G1-DARG", "G2-DARG", "G1-E81.7.2-G1", "G2-E81.7.2-G1", "G2-G1"):
        paired_summary[comparison] = {}
        for row in paired_rows:
            if row["comparison"] == comparison:
                paired_summary[comparison].setdefault(row["subset"], {})[row["metric"]] = {
                    "mean_difference_pp": row["mean_difference_pp"],
                    "population_sd_difference_pp": row["population_sd_difference_pp"],
                    "W_T_L": row["W_T_L"],
                    "fold_differences_pp": json.loads(row["fold_differences_pp"]),
                }

    performance = {}
    mechanism = {}
    for group in GROUPS:
        darg_acc = summary["DARG"]["acc"]["mean"]
        darg_ba = summary["DARG"]["hard_auc"]["mean"]
        sen_delta_pp = (summary[group]["sen"]["mean"] - summary["DARG"]["sen"]["mean"]) * 100.0
        exclude_acc_delta = summary[group]["acc"]["exclude_fold2_mean"] - np.mean(
            [darg_metrics[fold]["acc"] for fold in range(1, 11) if fold != 2]
        )
        exclude_ba_delta = summary[group]["hard_auc"]["exclude_fold2_mean"] - np.mean(
            [darg_metrics[fold]["hard_auc"] for fold in range(1, 11) if fold != 2]
        )
        net = total_mechanism[group]["A_to_C"]["net"]
        performance[group] = {
            "acc_exceeds_darg": summary[group]["acc"]["mean"] > darg_acc,
            "hard_auc_ba_exceeds_darg": summary[group]["hard_auc"]["mean"] > darg_ba,
            "sen_asd_not_more_than_1pp_below_darg": sen_delta_pp >= -1.0,
            "exclude_fold2_acc_direction_positive": bool(exclude_acc_delta > 0),
            "exclude_fold2_ba_direction_positive": bool(exclude_ba_delta > 0),
            "A_to_C_net_corrections_positive": net > 0,
            "exclude_fold2_acc_difference_pp": float(exclude_acc_delta * 100.0),
            "exclude_fold2_ba_difference_pp": float(exclude_ba_delta * 100.0),
            "sen_asd_difference_pp": float(sen_delta_pp),
            "pass": bool(summary[group]["acc"]["mean"] > darg_acc
                         and summary[group]["hard_auc"]["mean"] > darg_ba
                         and sen_delta_pp >= -1.0 and exclude_acc_delta > 0
                         and exclude_ba_delta > 0 and net > 0),
        }
        group_mechanisms = [row for row in mechanism_rows if row["group"] == group]
        if group == "G1":
            score_effect = float(np.mean([row["dua_score_delta_abs_mean"] for row in group_mechanisms]))
            valid_direction = [row["utility_direction_consistency"] for row in group_mechanisms
                               if math.isfinite(float(row.get("utility_direction_consistency", float("nan"))))]
            deletion_corr = [row["utility_u_vs_actual_margin_contribution_correlation"] for row in group_mechanisms
                             if math.isfinite(float(row.get("utility_u_vs_actual_margin_contribution_correlation", float("nan"))))]
            b_c_net = total_mechanism[group]["B_to_C"]["net"]
            mechanism[group] = {
                "dua_score_changed_relation": score_effect > 0,
                "mean_dua_score_delta_abs": score_effect,
                "mean_utility_direction_consistency": float(np.mean(valid_direction)) if valid_direction else float("nan"),
                "mean_u_actual_margin_correlation": float(np.mean(deletion_corr)) if deletion_corr else float("nan"),
                "positive_actual_margin_correlation_observed": bool(deletion_corr and np.mean(deletion_corr) > 0),
                "B_to_C_net_corrections": b_c_net,
                "utility_alignment_supported": bool(score_effect > 0 and valid_direction
                                                     and np.mean(valid_direction) > 0.5
                                                     and deletion_corr and np.mean(deletion_corr) > 0
                                                     and b_c_net > 0),
            }
        else:
            second_info = float(np.mean([row["second_minus_first_norm_mean"] for row in group_mechanisms]))
            stability = float(np.mean([row["stability_js_train_mean"] for row in group_mechanisms]))
            relation_difference = float(np.mean([row["asd_td_relation_l1_mean"] for row in group_mechanisms]))
            second_ablation_mae = float(np.mean([row["second_order_off_probability_mae"] for row in group_mechanisms]))
            mechanism[group] = {
                "second_order_adds_distinct_information": second_info > 0,
                "second_minus_first_relation_norm_mean": second_info,
                "train_relation_stability_js": stability,
                "asd_td_relation_l1_mean": relation_difference,
                "second_order_off_probability_mae": second_ablation_mae,
                "second_order_dependent_corrections": int(sum(row["second_order_dependent_corrections"]
                                                               for row in group_mechanisms)),
                "B_to_C_net_corrections": total_mechanism[group]["B_to_C"]["net"],
                "multiscale_mechanism_supported": bool(second_info > 0 and stability >= 0
                                                        and relation_difference > 0
                                                        and second_ablation_mae > 0
                                                        and sum(row["second_order_dependent_corrections"]
                                                                for row in group_mechanisms) > 0
                                                        and total_mechanism[group]["B_to_C"]["net"] > 0),
            }

    research_decision = {}
    for group in GROUPS:
        research_decision[group] = "GO" if performance[group]["pass"] and mechanism[group].get(
            "utility_alignment_supported" if group == "G1" else "multiscale_mechanism_supported", False
        ) else "NO-GO"
    status = {
        "status": "COMPLETE_AUDITED",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_sha256": frozen_source,
        "audit_source_sha256": run.sha_file(EXP / "audit.py"),
        "preflight_audit_source_sha256": preflight.get("audit_source_sha256"),
        "audit_source_changed_after_preflight": run.sha_file(EXP / "audit.py") != preflight.get("audit_source_sha256"),
        "gpu": gpu,
        "training_status": "20/20" if (RESULTS / "training_status.json").is_file() else "MISSING",
        "best_checkpoints_audited": checkpoint_audits // 2,
        "latest_checkpoints_audited": checkpoint_audits // 2,
        "best_latest_independent_metrics_recomputed": checkpoint_audits == 40,
        "latest_next_step_replayed": latest_replays,
        "latest_replay_all_20_pass": latest_replays == 20,
        "historical_darg_checkpoints_recomputed": 10,
        "historical_e8172_g1_folds": 10,
        "summary": summary,
        "paired_difference_vs_darg": group_differences,
        "paired_differences_all_references": paired_summary,
        "A_B_C_corrections": total_mechanism,
        "performance_gates": performance,
        "mechanism_gates": mechanism,
        "research_decision": research_decision,
        "audit_records": audit_records,
        "mechanism_records": mechanism_rows,
        "elapsed_seconds": time.time() - start,
        "protocol_limitations": [
            "historical val=test Best selection",
            "cross-validation was preceded by historical RFE and scaling",
            "transductive patient graph",
            "Fold2 retains two historical train/test-overlap rows, excluded only from new auxiliary labels and training residual",
            "single seed and fixed ten-fold history",
        ],
    }
    status = _json_safe(status)
    run.atomic_json(RESULTS / "audit_status.json", status)
    _write_report(status, paired_rows, mechanism_rows)
    return status


def _metric_text(summary, metric):
    mean = summary[metric]["mean"] * 100.0
    sd = summary[metric]["population_sd"] * 100.0
    return f"{mean:.4f} ± {sd:.4f}"


def _write_report(status, paired_rows, mechanism_rows):
    lines = [
        "# Experiment 81.8 — DUA-PSRR / MSR-PSRR",
        "",
        f"Status: `{status['status']}`. Independent Best and Latest audits: {status['best_checkpoints_audited']}/20 and {status['latest_checkpoints_audited']}/20. Latest next-step replay: {status['latest_next_step_replayed']}/20.",
        "",
        "All values are ten-fold mean ± population SD in percent. Hard-AUC is the historical hard-prediction Balanced Accuracy. Prob-AUC is ROC-AUC from continuous ASD probability.",
        "",
        "| Model | ACC | Hard-AUC / BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (("E74.4-G2 DARG", "DARG"), ("E81.7.2-G1", "E81.7.2-G1"),
                       ("G1 DUA-PSRR", "G1"), ("G2 MSR-PSRR", "G2")):
        values = status["summary"][key]
        lines.append("| " + label + " | " + " | ".join(_metric_text(values, metric)
                                                            for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")) + " |")
    lines.extend(["", "## Excluding Fold 2", "", "Nine-fold descriptive mean ± population SD in percent.",
                  "", "| Model | ACC | Hard-AUC / BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for label, key in (("E74.4-G2 DARG", "DARG"), ("E81.7.2-G1", "E81.7.2-G1"),
                       ("G1 DUA-PSRR", "G1"), ("G2 MSR-PSRR", "G2")):
        values = status["summary"][key]
        cells = []
        for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1"):
            cells.append(f"{values[metric]['exclude_fold2_mean'] * 100:.4f} ± "
                         f"{values[metric]['exclude_fold2_population_sd'] * 100:.4f}")
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines.extend(["", "## Paired Differences", "", "Difference is left minus reference in percentage points. W/T/L is computed foldwise.",
                  "", "| Comparison | Subset | ACC Δ (W/T/L) | BA Δ (W/T/L) | Prob-AUC Δ | SEN Δ | SPE Δ | ASD-F1 Δ |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for comparison in ("G1-DARG", "G2-DARG", "G1-E81.7.2-G1", "G2-E81.7.2-G1", "G2-G1"):
        for subset in ("all10", "exclude_fold2"):
            selected = [row for row in paired_rows if row["comparison"] == comparison and row["subset"] == subset]
            by_metric = {row["metric"]: row for row in selected}
            lines.append(f"| {comparison} | {subset} | {by_metric['ACC']['mean_difference_pp']:+.4f} ({by_metric['ACC']['W_T_L']}) | "
                         f"{by_metric['Hard-AUC/BA']['mean_difference_pp']:+.4f} ({by_metric['Hard-AUC/BA']['W_T_L']}) | "
                         f"{by_metric['Prob-AUC']['mean_difference_pp']:+.4f} | {by_metric['SEN-ASD']['mean_difference_pp']:+.4f} | "
                         f"{by_metric['SPE-TD']['mean_difference_pp']:+.4f} | {by_metric['ASD-F1']['mean_difference_pp']:+.4f} |")
    lines.extend(["", "## Patient Corrections", "", "A is historical DARG Best, B is the current Best checkpoint with the local residual disabled, and C is the full current checkpoint. A→B includes the changed training trajectory and Best epoch.",
                  "", "| Group | Transition | Corrected | Harmed | Net | ASD corrected/harmed | TD corrected/harmed |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for group in GROUPS:
        for transition in ("A_to_B", "B_to_C", "A_to_C"):
            counts = status["A_B_C_corrections"][group][transition]
            lines.append(f"| {group} | {transition} | {counts[f'{transition}_corrected']} | {counts[f'{transition}_harmed']} | "
                         f"{counts[f'{transition}_net']:+d} | {counts[f'{transition}_asd_corrected']}/{counts[f'{transition}_asd_harmed']} | "
                         f"{counts[f'{transition}_td_corrected']}/{counts[f'{transition}_td_harmed']} |")
    lines.extend(["", "## Mechanism", ""])
    for group in GROUPS:
        lines.append(f"### {group}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(status["mechanism_gates"][group], ensure_ascii=False, indent=2, allow_nan=False))
        lines.append("```")
        lines.append("")
    lines.extend(["## Scientific Decision", ""])
    for group in GROUPS:
        lines.append(f"- {group}: **{status['research_decision'][group]}**. Performance gate `{status['performance_gates'][group]['pass']}`. Mechanism gate `{status['mechanism_gates'][group].get('utility_alignment_supported', status['mechanism_gates'][group].get('multiscale_mechanism_supported'))}`.")
    lines.extend(["", "## Protocol Limits", ""])
    lines.extend(f"- {item}." for item in status["protocol_limitations"])
    lines.extend(["", "## Outputs", "", "- `fold_metrics.csv` contains independently recomputed Best metrics and confusion matrices.",
                  "- `patient_predictions.csv` contains fold-level Best probabilities and hard predictions.",
                  "- `paired_comparisons.csv` contains paired fold differences and W/T/L.",
                  "- `mechanism.csv` contains fold-level A/B/C and group mechanism measurements.",
                  "- `checkpoint_audit.csv` records Best/Latest load, metric, and Latest replay checks.",
                  "- `audit_status.json` contains full per-fold and gate records.", ""])
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("audit requires --run")
    status = audit_all()
    print(json.dumps({
        "status": status["status"],
        "latest_replays": status["latest_next_step_replayed"],
        "research_decision": status["research_decision"],
        "elapsed_seconds": status["elapsed_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
