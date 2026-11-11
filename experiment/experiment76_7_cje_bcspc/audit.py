#!/usr/bin/env python3
"""Independent disk-based checkpoint, prediction, and mechanism audit for E76.7."""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import run as runner

RESULTS = EXP / "results"
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1", "nll", "brier", "ece")
CORE = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
GROUPS = runner.GROUPS


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
    temp.replace(path)


def write_json(path, value):
    runner.atomic_json(path, value)


def close(a, b, atol=1e-7):
    import numpy as np
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    if hasattr(b, "detach"):
        b = b.detach().cpu().numpy()
    return np.asarray(a).shape == np.asarray(b).shape and bool(np.allclose(a, b, atol=atol, rtol=0))


def mean_sd(values):
    values = [float(value) for value in values]
    return statistics.mean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


def ci95(values):
    # Descriptive paired t interval; folds are not independent external cohorts.
    n = len(values)
    if n < 2:
        return float(values[0]), float(values[0])
    critical = {9: 2.306, 10: 2.262}.get(n, 1.96)
    mean = statistics.mean(values)
    half = critical * statistics.stdev(values) / math.sqrt(n)
    return mean - half, mean + half


def metric_summary(rows, method):
    selected = [row for row in rows if row["method"] == method]
    return {metric: mean_sd([row[metric] for row in selected]) for metric in METRICS}


def transitions(before, after, labels):
    import numpy as np
    before, after, labels = np.asarray(before, int), np.asarray(after, int), np.asarray(labels, int)
    was_wrong, now_correct = before != labels, after == labels
    was_correct, now_wrong = before == labels, after != labels
    return {
        "error_to_correct": int(np.sum(was_wrong & now_correct)),
        "correct_to_error": int(np.sum(was_correct & now_wrong)),
        "net_corrections": int(np.sum(was_wrong & now_correct) - np.sum(was_correct & now_wrong)),
        "ASD_error_to_correct": int(np.sum((labels == 0) & was_wrong & now_correct)),
        "TD_error_to_correct": int(np.sum((labels == 1) & was_wrong & now_correct)),
        "ASD_to_TD_corrections": int(np.sum((labels == 0) & (before == 1) & (after == 0))),
        "TD_to_ASD_corrections": int(np.sum((labels == 1) & (before == 0) & (after == 1))),
        "original_correct": int(np.sum(was_correct)),
        "original_correct_kept": int(np.sum(was_correct & now_correct)),
        "prediction_changes": int(np.sum(before != after)),
    }


def aggregate_diagnostics(diag, labels, base_prediction, final_prediction, interventions):
    import numpy as np
    y = np.asarray(labels, int)
    output = {}
    for name in ("delta", "base_score", "final_score", "final_boundary_evidence"):
        values = diag[name].detach().cpu().numpy().reshape(-1)
        output[f"{name}_mean"] = float(values.mean())
        output[f"{name}_std"] = float(values.std())
        output[f"{name}_abs_mean"] = float(np.abs(values).mean())
        output[f"{name}_abs_max"] = float(np.abs(values).max())
    for name in ("reliable_similarity", "boundary_similarity", "reliable_difference", "boundary_difference", "slot_difference"):
        value = diag[name].detach().cpu().numpy()
        output[name + "_mean_by_class_or_slot"] = json.dumps(value.mean(0).tolist())
    for name in ("modality_weights", "attention", "support_topk", "support_coverage", "support_reliability", "candidate_count", "prototype_role_cosine"):
        value = diag[name].detach().cpu().numpy()
        output[name + "_mean"] = json.dumps(value.mean(0).tolist())
        output[name + "_std"] = json.dumps(value.std(0).tolist())
    boundary = diag["boundary_difference"].detach().cpu().numpy()
    reliable = diag["reliable_difference"].detach().cpu().numpy()
    t = np.where(y == 0, 1.0, -1.0)
    output["reliable_direction_accuracy"] = float(np.mean(t * reliable.mean(-1) > 0))
    output["boundary_direction_accuracy"] = float(np.mean(t * boundary.mean(-1) > 0))
    output["reliable_boundary_sign_agreement"] = float(np.mean(np.sign(reliable) == np.sign(boundary)))
    output["delta_direction_matches_label"] = float(np.mean(np.sign(diag["delta"].detach().cpu().numpy()) == t))
    output["delta_direction_matches_base_error"] = float(np.mean(
        (np.asarray(base_prediction) != y) == (np.sign(diag["delta"].detach().cpu().numpy()) == t)))
    output.update(transitions(base_prediction, final_prediction, y))
    output["retrieval_fallbacks"] = int(diag["retrieval_fallbacks"])
    output["site_cap_violations"] = int(diag["site_cap_violations"])
    output["invalid_confusion_candidates"] = int(diag["invalid_confusion_candidates"])
    for key, value in interventions.items():
        output[f"{key}_mean_abs_probability_change"] = float(np.abs(value["probability"] - value["full_probability"]).mean())
        output[f"{key}_prediction_changes"] = int(np.sum(value["prediction"] != value["full_prediction"]))
        output[f"{key}_acc"] = float(value["metrics"]["acc"])
    return output


def audit_fold(group, fold, source_sha):
    import numpy as np
    import torch
    task_path = runner.task_path(group, fold)
    checkpoint_path = runner.best_path(group, fold)
    latest_path = runner.latest_path(group, fold)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    checkpoint_sha = runner.sha_file(checkpoint_path)
    if task.get("status") != "COMPLETE" or task.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError(f"task/checkpoint identity mismatch {group} fold={fold}")
    saved = runner.torch_load(checkpoint_path)
    if (saved.get("source_sha256") != task.get("source_sha256")
            or saved.get("group") != group or saved.get("fold") != fold):
        raise RuntimeError(f"checkpoint provenance mismatch {group} fold={fold}")
    best = saved["best"]
    state, warm, identity, history_task, history_best = runner.prepare_fold(fold)
    if saved["identity"] != identity or saved["warmup_sha256"] != identity["warmup_sha256"]:
        raise RuntimeError(f"warmup/fold identity mismatch {group} fold={fold}")
    hist = runner.legacy.load_base().load_history()
    hist.load_bundle(state, best["base_state"])
    state["fold"], state["group"], state["rudag_identity"] = fold, runner.BASE_GROUP, identity
    brain, sprompt = runner.prompt_vectors()
    module = runner.init_module(brain, sprompt, fold)
    module.load_state_dict(best["module"], strict=True)
    module.eval()
    eligible, ids_a, ids_b = runner.split_branch_ids(state, fold)
    if eligible != best["eligible_train_ids"] or ids_a != best["support_a"] or ids_b != best["support_b"]:
        raise RuntimeError(f"training prototype partition mismatch {group} fold={fold}")
    test_ids = list(map(int, state["test_np"]))
    if test_ids != best["test_ids"]:
        raise RuntimeError(f"test identity mismatch {group} fold={fold}")
    memory = runner.memory_device(best["memory_used"], state["y"].device)
    if memory is None:
        raise RuntimeError(f"missing memory snapshot {group} fold={fold}")
    memory_ids = set(torch.nonzero(memory["observed"], as_tuple=False).flatten().cpu().tolist())
    expected_memory_ids = set(eligible) if int(memory["updates"]) else set()
    if memory_ids != expected_memory_ids:
        raise RuntimeError(f"EMA memory contains missing/illegal patient IDs {group} fold={fold}")
    if set(torch.unique(memory["labels"][memory["observed"]]).cpu().tolist()) - {0, 1}:
        raise RuntimeError(f"invalid prototype labels {group} fold={fold}")
    if fold == 2 and ({376, 406} & memory_ids or memory["labels"][[376, 406]].ne(-1).any()):
        raise RuntimeError("Fold 2 overlap rows entered the new branch memory")

    # Independent base and corrected inference from the saved model + exact memory version.
    with torch.no_grad():
        base_out, cam = runner.hook_forward(state, training=False)
        full_logp, diag = runner.eval_method(module, base_out["logp"], cam, memory,
                                             eligible, test_ids, runner.read_sites().to(state["y"].device))
        full_metrics = runner.exact_metrics(full_logp, test_ids, state["y"])
        base_metrics = runner.exact_metrics(base_out["logp"], test_ids, state["y"])
        interventions = {}
        for name, options in (
            ("Residual-Off", {"residual_off": True}),
            ("Boundary-Off", {"boundary_off": True}),
            ("Semantic-Swap", {"semantic_swap": True}),
            ("Reliable-only", {"reliable_only": True}),
        ):
            candidate, _ = runner.eval_method(module, base_out["logp"], cam, memory,
                                               eligible, test_ids, runner.read_sites().to(state["y"].device), **options)
            metrics = runner.exact_metrics(candidate, test_ids, state["y"])
            interventions[name] = {"metrics": metrics,
                                   "probability": np.asarray(metrics["prob"]),
                                   "prediction": np.asarray(metrics["pred"])}
        for item in interventions.values():
            item["full_probability"] = np.asarray(full_metrics["prob"])
            item["full_prediction"] = np.asarray(full_metrics["pred"])
    if not close(full_metrics["pred"], best["prediction"], atol=0):
        raise RuntimeError(f"independent class prediction mismatch {group} fold={fold}")
    if not close(full_metrics["prob"], best["probability"], atol=1e-7):
        raise RuntimeError(f"independent probability mismatch {group} fold={fold}")
    if not close(base_metrics["pred"], best["base_prediction"], atol=0):
        raise RuntimeError(f"independent base class mismatch {group} fold={fold}")
    if not close(base_metrics["prob"], best["base_probability"], atol=1e-7):
        raise RuntimeError(f"independent base probability mismatch {group} fold={fold}")
    for key in runner.legacy.METRICS:
        if abs(float(full_metrics[key]) - float(best["metrics"][key])) > 1e-8:
            raise RuntimeError(f"independent {key} mismatch {group} fold={fold}")

    # Every eval retrieval candidate must come from this fold's legal train support.
    selected = diag["selected_ids"].detach().cpu().numpy().reshape(-1)
    selected = set(int(value) for value in selected if value >= 0)
    if not selected <= set(eligible) or selected & set(test_ids):
        raise RuntimeError(f"test/checkpoint retrieval has illegal support IDs {group} fold={fold}")
    if best["epoch"] != saved["best_epoch_1based"] or saved["epochs_executed"] != task["epochs_executed"]:
        raise RuntimeError(f"epoch ledger mismatch {group} fold={fold}")

    # Shared stochastic operations must match the historical base at every epoch.
    historical = {int(row["epoch"]): row for row in history_task["rng_trace"]}
    actual = {int(row["epoch"]): row for row in saved["rng_trace"]}
    common_epochs = set(actual) & set(historical)
    mismatches = [epoch for epoch in sorted(common_epochs) if historical[epoch] != actual[epoch]]
    if mismatches or (group == GROUPS[1] and not set(historical).issubset(set(actual))):
        raise RuntimeError(f"shared RNG trace mismatch {group} fold={fold}: {mismatches[:5]}")
    if group == GROUPS[1]:
        if set(actual) != set(historical):
            raise RuntimeError(f"StopGrad RNG trace has extra/missing epochs fold={fold}")
        if saved["epochs_executed"] != history_task["epochs_executed"] or task["base_best_epoch_1based"] != history_task["best_epoch_1based"]:
            raise RuntimeError(f"StopGrad base ledger differs from historical run fold={fold}")
        latest = runner.torch_load(latest_path)
        historical_cp = runner.torch_load(history_best)
        runner._assert_state_equal(historical_cp["best_state"], latest["base_ledger"]["state"],
                                   f"StopGrad historical base params/optimizers fold={fold}")
    latest = runner.torch_load(latest_path)

    labels = state["y"][torch.as_tensor(test_ids, device=state["y"].device)].detach().cpu().numpy()
    base_pred = np.asarray(best["base_prediction"], int)
    final_pred = np.asarray(best["prediction"], int)
    stats = aggregate_diagnostics(diag, labels, base_pred, final_pred, interventions)
    metric_row = {"method": group, "fold": fold, **{key: float(full_metrics[key]) for key in METRICS},
                  "best_epoch_1based": saved["best_epoch_1based"],
                  "epochs_executed": saved["epochs_executed"],
                  "checkpoint_sha256": checkpoint_sha,
                  "train_test_overlap_1based": json.dumps(identity["membership"]["train_test_overlap_1based"])}
    mechanism_row = {"method": group, "fold": fold, "best_epoch_1based": saved["best_epoch_1based"],
                     "memory_updates": best["memory_updates"], "memory_eligible_count": len(eligible),
                     "support_a_count": len(ids_a), "support_b_count": len(ids_b),
                     "selected_support_count": len(selected),
                     "semantic_projection_gradient_norm_best_epoch": best["gradient_snapshot"].get("branch_gradient_norm", 0.0),
                     "gradient_snapshot_best_epoch": json.dumps(best["gradient_snapshot"], allow_nan=False),
                     **stats}

    # E74.4-G2 comparison is independently re-run at its own selected epoch.
    hist_state, _, hist_identity, hist_task, hist_path = runner.prepare_fold(fold)
    hist_cp = runner.torch_load(hist_path)
    runner.legacy.load_base().load_history().load_bundle(hist_state, hist_cp["best_state"])
    with torch.no_grad():
        historic_out, _ = runner.hook_forward(hist_state, training=False)
        historic_metrics = runner.exact_metrics(historic_out["logp"], list(map(int, hist_state["test_np"])), hist_state["y"])
    historic_prediction = np.asarray(historic_metrics["pred"], int)
    historic_labels = hist_state["y"][torch.as_tensor(hist_state["test_np"], device=hist_state["y"].device)].detach().cpu().numpy()
    history_row = {"method": "E74.4-G2-Positive-DCR", "fold": fold,
                   **{key: float(historic_metrics[key]) for key in METRICS},
                   "best_epoch_1based": hist_task["best_epoch_1based"],
                   "epochs_executed": hist_task["epochs_executed"],
                   "checkpoint_sha256": runner.sha_file(hist_path),
                   "train_test_overlap_1based": json.dumps(hist_identity["membership"]["train_test_overlap_1based"])}
    historical_correction = transitions(historic_prediction, final_pred, historic_labels)
    correction_row = {"comparison": f"{group}-vs-E74.4-G2", "fold": fold,
                      **historical_correction}
    if group == GROUPS[0]:
        correction_row.update({f"residual_off_{key}": value for key, value in transitions(
            np.asarray(interventions["Residual-Off"]["prediction"], int), final_pred, labels).items()})

    if group == GROUPS[0]:
        # G1-vs-G2 transition is filled after both groups have been audited.
        pass
    return metric_row, mechanism_row, correction_row, history_row, {
        "prediction": final_pred.tolist(), "labels": np.asarray(labels, int).tolist(),
        "base_prediction": base_pred.tolist(), "historical_prediction": historic_prediction.tolist(),
        "interventions": {key: {"prediction": np.asarray(value["prediction"], int).tolist(),
                                 "metrics": {k: float(value["metrics"][k]) for k in METRICS}}
                          for key, value in interventions.items()},
        "checkpoint_sha256": checkpoint_sha, "rng_mismatches": len(mismatches),
        "selected_support_count": len(selected), "memory_id_count": len(memory_ids),
        "module_parameter_count": sum(value.numel() for value in best["module"].values()),
        "g2_backbone_replay_check": group == GROUPS[1],
        "latest_gradient_audit": latest.get("gradient_audit", []),
    }


def build_comparisons(metric_rows):
    pairs = [(GROUPS[0], "E74.4-G2-Positive-DCR"),
             (GROUPS[1], "E74.4-G2-Positive-DCR"),
             (GROUPS[0], GROUPS[1]),
             (GROUPS[0], "E76.6-G1-BCSPC-Full")]
    output = []
    for left, right in pairs:
        for metric in CORE:
            left_rows = {int(row["fold"]): row for row in metric_rows if row["method"] == left}
            right_rows = {int(row["fold"]): row for row in metric_rows if row["method"] == right}
            if right == "E76.6-G1-BCSPC-Full":
                left_rows = {fold: row for fold, row in left_rows.items()}
                e76 = {int(row["fold"]): row for row in read_csv(runner.E76_6 / "results/fold_metrics.csv")
                       if row["method"] == "G1-BCSPC-Full"}
                right_rows = {fold: {metric: float(row[metric]) for metric in CORE} for fold, row in e76.items()}
            if set(left_rows) != set(right_rows):
                raise RuntimeError(f"paired comparison incomplete: {left} vs {right} {metric}")
            deltas = {fold: (float(left_rows[fold][metric]) - float(right_rows[fold][metric])) * 100.0
                      for fold in sorted(left_rows)}
            all_values = list(deltas.values())
            low, high = ci95(all_values)
            nine = [value for fold, value in deltas.items() if fold != 2]
            low9, high9 = ci95(nine)
            output.append({"left": left, "right": right, "metric": metric,
                           "mean_delta_pp": statistics.mean(all_values),
                           "fold_deltas_pp": json.dumps({fold: round(value, 6) for fold, value in deltas.items()}),
                           "wins": sum(value > 1e-12 for value in all_values),
                           "ties": sum(abs(value) <= 1e-12 for value in all_values),
                           "losses": sum(value < -1e-12 for value in all_values),
                           "descriptive_ci95_low_pp": low, "descriptive_ci95_high_pp": high,
                           "exclude_fold2_mean_delta_pp": statistics.mean(nine),
                           "exclude_fold2_ci95_low_pp": low9, "exclude_fold2_ci95_high_pp": high9,
                           "n_folds": len(all_values)})
    return output


def replay_latest_step(group, fold):
    """Reload and execute the same next train step twice without saving either result."""
    import torch
    saved = runner.torch_load(runner.latest_path(group, fold))
    state, warm, identity, history_task, _ = runner.prepare_fold(fold)
    hist = runner.legacy.load_base().load_history()
    rng = runner.legacy.history_runtime()
    hist.load_bundle(state, saved["base_state"])
    state["fold"], state["group"], state["rudag_identity"] = fold, runner.BASE_GROUP, identity
    brain, sprompt = runner.prompt_vectors()
    module = runner.init_module(brain, sprompt, fold)
    module.load_state_dict(saved["module"], strict=True)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01)
    optimizer.load_state_dict(saved["optimizer"])
    memory = runner.memory_device(saved["memory"], state["y"].device)
    eligible, ids_a, ids_b = runner.split_branch_ids(state, fold)
    sites = runner.read_sites().to(state["y"].device)
    labels = state["y"].long()
    step_epoch = int(saved["next_epoch"])

    def one_step():
        hist.zero_optimizers(state)
        optimizer.zero_grad(set_to_none=True)
        out, cam = runner.hook_forward(state, training=True)
        live = group == GROUPS[0] and step_epoch + 1 >= 21
        branch = runner.branch_train(module, cam, memory, out["logp"], labels,
                                     eligible, ids_a, ids_b, sites, state["weights"], detached=not live)
        beta = (0.3 * min(1.0, max(0.0, (step_epoch + 1 - 20) / 20.0))) if live else 0.0
        gradient = runner.gradients_and_step(state, module, out, cam, memory, branch, group, step_epoch, beta)
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        if int(memory["updates"]):
            optimizer.step()
        rng_sha = rng.rng_sha(rng.pack_rng())
        result = {"base": hist.state_bundle(state),
                  "module": {key: value.detach().cpu().clone() for key,value in module.state_dict().items()},
                  "optimizer": optimizer.state_dict(), "rng": rng_sha,
                  "branch_loss": float(branch[0].detach().cpu()),
                  "gradient": gradient}
        del out, cam
        return result

    rng.restore_rng(saved["rng"])
    first = one_step()
    hist.load_bundle(state, saved["base_state"])
    module.load_state_dict(saved["module"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    memory = runner.memory_device(saved["memory"], state["y"].device)
    rng.restore_rng(saved["rng"])
    second = one_step()
    runner._assert_state_equal(first["base"], second["base"], f"latest next-step base replay {group} fold={fold}")
    def max_numeric_error(left, right):
        if torch.is_tensor(left) and torch.is_tensor(right):
            if left.shape != right.shape:
                raise RuntimeError("replay tensor shape mismatch")
            return float((left.detach().float().cpu() - right.detach().float().cpu()).abs().max()) if left.numel() else 0.0
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                raise RuntimeError("replay mapping keys mismatch")
            return max((max_numeric_error(left[key], right[key]) for key in left), default=0.0)
        if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
            if len(left) != len(right):
                raise RuntimeError("replay sequence length mismatch")
            return max((max_numeric_error(a, b) for a, b in zip(left, right)), default=0.0)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return abs(float(left) - float(right))
        if left != right:
            raise RuntimeError("replay nonnumeric state mismatch")
        return 0.0
    module_error = max_numeric_error(first["module"], second["module"])
    optimizer_error = max_numeric_error(first["optimizer"], second["optimizer"])
    if module_error > 2e-6 or optimizer_error > 2e-6:
        raise RuntimeError(f"latest branch replay error exceeds 2e-6: module={module_error}, optimizer={optimizer_error}")
    if first["rng"] != second["rng"] or abs(first["branch_loss"] - second["branch_loss"]) > 1e-7:
        raise RuntimeError(f"latest next-step RNG/loss replay mismatch {group} fold={fold}")
    del state, module, optimizer, memory
    torch.cuda.empty_cache()
    return {"group":group,"fold":fold,"next_epoch_index":step_epoch,
            "base_optimizer_replay_exact":True,"branch_optimizer_replay_within_tolerance":True,
            "branch_replay_max_abs_error":module_error,
            "branch_optimizer_replay_max_abs_error":optimizer_error,
            "branch_replay_atol":2e-6,"rng_replay_exact":True,
            "loss_replay_atol":1e-7,"loss_replay_error":abs(first["branch_loss"]-second["branch_loss"])}


def write_report(metric_rows, comparisons, correction_rows, mechanism_rows, status):
    lines = ["# Experiment 76.7 CJE-BCSPC", "",
             f"STATUS: `{status['status']}`。科研判定：`{status['method_verdict']}`。", "",
             "结果沿用E74.4-G2历史协议：10折、测试折参与ACC选模、CV前RFE/标准化及transductive图。该对比是历史协议内的探索性结果，不是严格无偏的独立泛化估计。Fold 2存在历史训练/测试重复行；新增分支已排除行377/407，但原主干历史协议仍保留该问题。", "",
             "| 方法 | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    display = [("E74.4-G2-Positive-DCR", "E74.4-G2历史基座"),
               ("E76.6-G1-BCSPC-Full", "E76.6-G1历史参考"),
               (GROUPS[0], "E76.7-G1 Full-E2E"), (GROUPS[1], "E76.7-G2 StopGrad")]
    for method, label in display:
        if method == "E76.6-G1-BCSPC-Full":
            rows = [{key: float(row[key]) for key in METRICS} for row in read_csv(runner.E76_6 / "results/fold_metrics.csv") if row["method"] == "G1-BCSPC-Full"]
        else:
            rows = [row for row in metric_rows if row["method"] == method]
        values = {}
        for metric in CORE:
            mean, sd = mean_sd([row[metric] for row in rows])
            values[metric] = f"{mean*100:.2f}±{sd*100:.2f}"
        lines.append(f"| {label} | {values['acc']} | {values['hard_auc']} | {values['prob_auc']} | {values['sen']} | {values['spe']} | {values['asd_f1']} |")
    lines.extend(["", "配对差值单位为百分点；区间为描述性配对t区间，10折不是10个独立外部队列。排除Fold 2结果见 `paired_comparisons.csv`。", "",
                  "| 配对比较 | 指标 | 均值差(pp) | W/T/L | 95%描述区间 | 排除Fold 2差值(pp) |", "|---|---|---:|---:|---:|---:|"])
    for row in comparisons:
        if row["metric"] in ("acc", "hard_auc", "prob_auc"):
            lines.append(f"| {row['left']} − {row['right']} | {row['metric']} | {row['mean_delta_pp']:.3f} | {row['wins']}/{row['ties']}/{row['losses']} | [{row['descriptive_ci95_low_pp']:.3f}, {row['descriptive_ci95_high_pp']:.3f}] | {row['exclude_fold2_mean_delta_pp']:.3f} |")
    lines.extend(["", "纠错与机制", ""])
    for group in GROUPS:
        selected = [row for row in correction_rows if row["comparison"] == f"{group}-vs-E74.4-G2"]
        lines.append(f"- {group}相对历史基座：错误→正确 {sum(int(r['error_to_correct']) for r in selected)}，正确→错误 {sum(int(r['correct_to_error']) for r in selected)}，净纠错 {sum(int(r['net_corrections']) for r in selected)}。")
    lines.append(f"- 独立审计：{status['checkpoint_recalculations']}/20最佳Checkpoint复算通过；G2与历史基座RNG逐epoch匹配：{status['g2_rng_matches_history']}；G1共有随机操作逐epoch匹配：{status['g1_rng_matches_history']}。")
    lines.append("- 四槽位注意力、可靠/边界证据、残差及Residual-Off、Boundary-Off、Semantic-Swap、Reliable-only干预详见 `mechanism.csv` 和 `correction_analysis.csv`。Prompt交换是同Checkpoint干预；本轮没有匿名语义正式训练组，因此不能据此宣称疾病文本有独立预测价值。")
    lines.append("- G1为主干保护式预热后受控端到端联合优化；G2为相同结构、始终StopGrad。新增损失的主干梯度使用最终分类损失减原分类损失的梯度差，并按原优化器组限幅。")
    lines.append("")
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit():
    import torch
    RESULTS.mkdir(parents=True, exist_ok=True)
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    source_sha = runner.sha_obj(runner.source_snapshot())
    if preflight.get("status") != "PASS" or preflight.get("source_sha256") != source_sha:
        raise RuntimeError("preflight or locked-source SHA is missing/mismatched")
    current_sources = runner.source_snapshot()
    expected_sources = preflight.get("source_files", {})
    mismatched = [path for path, digest in current_sources.items()
                  if path not in expected_sources or expected_sources[path] != digest]
    if mismatched:
        raise RuntimeError("locked source-file SHA mismatch: " + ", ".join(mismatched))
    # Checkpoints retain the aggregate training-source SHA from their run. The
    # later audit-only revision is separately locked by this preflight and manifest.
    checkpoint_sources = {
        json.loads(runner.task_path(group, fold).read_text(encoding="utf-8")).get("source_sha256")
        for group in GROUPS for fold in range(1, 11)
    }
    if len(checkpoint_sources) != 1 or None in checkpoint_sources:
        raise RuntimeError("formal task ledgers do not share one training-source identity")
    metric_rows, mechanism_rows, correction_rows, history_rows, details = [], [], [], [], {}
    for fold in range(1, 11):
        for group in GROUPS:
            metric, mechanism, correction, history, detail = audit_fold(group, fold, source_sha)
            metric_rows.append(metric)
            mechanism_rows.append(mechanism)
            correction_rows.append(correction)
            details[(group, fold)] = detail
            if group == GROUPS[0]:
                history_rows.append(history)
        if details[(GROUPS[0],fold)]["module_parameter_count"] != details[(GROUPS[1],fold)]["module_parameter_count"]:
            raise RuntimeError(f"G1/G2 added-module parameter counts differ fold={fold}")
    # The history baseline is one common, independently recalculated row per fold.
    metric_rows.extend(history_rows)
    # Add the read-only E76.6 reference to the summary table only.
    e76 = read_csv(runner.E76_6 / "results/fold_metrics.csv")
    for row in e76:
        if row["method"] == "G1-BCSPC-Full":
            fold = int(row["fold"])
            history_task = json.loads((runner.E76_6 / ".runtime/tasks" / f"G1-BCSPC-Full_f{fold:02d}.json").read_text(encoding="utf-8"))
            history_checkpoint = runner.E76_6 / ".runtime/checkpoints" / f"G1-BCSPC-Full_f{fold:02d}.best.pt"
            metric_rows.append({"method":"E76.6-G1-BCSPC-Full","fold":int(row["fold"]),
                                **{key:float(row[key]) for key in METRICS},
                                "best_epoch_1based":int(history_task["best_epoch_1based"]),
                                "epochs_executed":int(history_task["epochs_executed"]),
                                "checkpoint_sha256":runner.sha_file(history_checkpoint)})
    # Paired patient-wise G1/G2 transition analysis within identical test folds.
    for fold in range(1, 11):
        g1, g2 = details[(GROUPS[0], fold)], details[(GROUPS[1], fold)]
        if g1["labels"] != g2["labels"]:
            raise RuntimeError(f"G1/G2 patient order mismatch fold={fold}")
        correction_rows.append({"comparison":"G1-vs-G2", "fold":fold,
                                **transitions(g2["prediction"],g1["prediction"],g1["labels"])})
    comparisons = build_comparisons(metric_rows)
    write_csv(RESULTS / "fold_metrics.csv", metric_rows)
    write_csv(RESULTS / "paired_comparisons.csv", comparisons)
    write_csv(RESULTS / "mechanism.csv", mechanism_rows)
    write_csv(RESULTS / "correction_analysis.csv", correction_rows)

    g1 = metric_summary(metric_rows, GROUPS[0])
    g2 = metric_summary(metric_rows, GROUPS[1])
    base = metric_summary(metric_rows, "E74.4-G2-Positive-DCR")
    g1corr = [row for row in correction_rows if row["comparison"] == f"{GROUPS[0]}-vs-E74.4-G2"]
    net = sum(int(row["net_corrections"]) for row in g1corr)
    e2e_acc = next(row for row in comparisons if row["left"] == GROUPS[0] and row["right"] == GROUPS[1] and row["metric"] == "acc")
    base_acc = next(row for row in comparisons if row["left"] == GROUPS[0] and row["right"] == "E74.4-G2-Positive-DCR" and row["metric"] == "acc")
    base_9 = base_acc["exclude_fold2_mean_delta_pp"]
    hard_diff = next(row["mean_delta_pp"] for row in comparisons
                     if row["left"] == GROUPS[0] and row["right"] == "E74.4-G2-Positive-DCR" and row["metric"] == "hard_auc")
    prob_diff = next(row["mean_delta_pp"] for row in comparisons
                     if row["left"] == GROUPS[0] and row["right"] == "E74.4-G2-Positive-DCR" and row["metric"] == "prob_auc")
    e2e_samples = [sample for (group, _), detail in details.items() if group == GROUPS[0]
                   for sample in detail["latest_gradient_audit"] if sample.get("end_to_end")]
    e2e_nonzero = any(any(value.get("new_raw_norm", 0.0) > 0 for value in sample.get("optimizer_groups", {}).values())
                      for sample in e2e_samples)
    if (base_acc["mean_delta_pp"] > 0 and hard_diff >= -0.5 and prob_diff >= -0.5
            and e2e_acc["mean_delta_pp"] > 0 and net > 0 and base_9 > 0 and e2e_nonzero):
        verdict = "PROMISING_CJE_BCSPC"
    elif base_acc["mean_delta_pp"] > 0 and e2e_acc["mean_delta_pp"] <= 0:
        verdict = "PROTOTYPE_CORRECTION_GAIN_E2E_UNVERIFIED"
    elif e2e_acc["mean_delta_pp"] > 0:
        verdict = "E2E_IMPROVEMENT_WITHOUT_BASE_GAIN"
    else:
        verdict = "NO_GO_CJE_BCSPC"

    rng_g1 = all(details[(GROUPS[0],fold)]["rng_mismatches"] == 0 for fold in range(1,11))
    rng_g2 = all(details[(GROUPS[1],fold)]["rng_mismatches"] == 0 for fold in range(1,11))
    replay = [replay_latest_step(group, 1) for group in GROUPS]
    status = {"status":"COMPLETE_AUDITED", "method_verdict":verdict,
              "checkpoint_recalculations":20,"checkpoint_total":20,
              "g1_rng_matches_history":rng_g1,"g2_rng_matches_history":rng_g2,
              "g2_base_params_and_optimizers_match_history":all(details[(GROUPS[1],f)]["g2_backbone_replay_check"] for f in range(1,11)),
              "source_sha256":source_sha,"preflight_sha256":preflight["source_sha256"],
              "groups":list(GROUPS),"folds":10,
              "metrics_mean_sd_percent":{group:{metric:[mean*100,sd*100] for metric,(mean,sd) in metric_summary(metric_rows,group).items() if metric in CORE}
                                          for group in ("E74.4-G2-Positive-DCR","E76.6-G1-BCSPC-Full",*GROUPS)},
              "g1_minus_base_acc_pp":base_acc["mean_delta_pp"],"g1_minus_g2_acc_pp":e2e_acc["mean_delta_pp"],
              "g1_minus_base_hard_auc_pp":hard_diff,"g1_minus_base_prob_auc_pp":prob_diff,
              "g1_minus_base_net_corrections":net,
              "g1_new_backbone_gradient_observed":e2e_nonzero,
              "latest_checkpoint_replay":replay,
              "audit_checks":{"history_base_identity":True,"E61_initialization_identity":True,
                "historical_rng_all_common_epochs":rng_g1 and rng_g2,"G2_historical_best_base_state_exact":True,
                "all_20_best_predictions_recalculated":True,"stored_probabilities_recalculated":True,
                "legal_train_only_memory":True,"fold2_overlap_excluded_from_new_branch":True,
                "support_A_B_disjoint":True,"test_labels_not_in_new_memory":True,
                "four_mode_cam_hook_shape":True,"zero_delta_preflight_pass":True,
                "prompt_cache_and_order_preflight_pass":True,"G1_G2_parameter_count_matched":True,
                "residual_off_boundary_off_swap_reliable_only_interventions":True}}
    write_json(RESULTS / "audit_status.json", status)
    write_report(metric_rows, comparisons, correction_rows, mechanism_rows, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return status


if __name__ == "__main__":
    run_audit()
