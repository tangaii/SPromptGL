#!/usr/bin/env python3
"""Independent metrics, interventions, and mechanism analysis for E80.2."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"
TASKS = EXP / ".runtime/tasks"
CHECKPOINTS = EXP / ".runtime/checkpoints"
spec = importlib.util.spec_from_file_location("e802_analysis_runtime", EXP / "run.py")
R = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = R
assert spec.loader is not None
spec.loader.exec_module(R)
R.source_hash = R.source_sha()

METRIC_KEYS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
LABELS = {"acc": "ACC", "hard_auc": "Hard-AUC/BA", "prob_auc": "Prob-AUC", "sen": "SEN-ASD", "spe": "SPE-TD", "asd_f1": "ASD-F1"}


def read_task(group, fold):
    return json.loads((TASKS / f"{group}_f{fold:02d}.json").read_text(encoding="utf-8"))


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def ci95(values):
    a = np.asarray(values, dtype=float)
    if len(a) <= 1:
        return float(a.mean()), float(a.mean())
    half = 1.96 * float(a.std(ddof=1)) / math.sqrt(len(a))
    return float(a.mean() - half), float(a.mean() + half)


def state_labels(fold):
    state, warmup, identity, scaler = R.prepare_fold(fold)
    return np.asarray(state["y"].detach().cpu(), dtype=int), state["test_np"].copy()


def fold_cache():
    out = {}
    for fold in range(1, 11):
        y, test = state_labels(fold)
        out[fold] = {"y": y[test], "test": test}
    return out


def pooled_metrics(group, folds, cache):
    probs, labels = [], []
    for fold in folds:
        d = read_task(group, fold)
        probs.append(np.asarray(d["test_probability"], dtype=float))
        labels.append(cache[fold]["y"])
    p = torch.as_tensor(np.concatenate(probs), dtype=torch.float32, device=R.DEVICE).clamp_min(1e-8)
    y = torch.as_tensor(np.concatenate(labels), dtype=torch.long, device="cpu")
    idx = np.arange(len(y), dtype=np.int64)
    return {k: float(v) for k, v in zip(METRIC_KEYS, [R.metrics(p.log(), idx, y)[key] for key in METRIC_KEYS])}


def load_best(fold, group):
    state, warmup, identity, scaler = R.prepare_fold(fold)
    prompts = R.load_prompts()
    modules, opts, priv = {}, {}, {}
    for g in R.GROUPS:
        modules[g], opts[g], priv[g] = R.initialize_module(prompts["anonymous" if g == "G1" else "medical"], scaler, fold)
        modules[g].eval()
    payload = R.torch_load(CHECKPOINTS / f"{group}_f{fold:02d}.best.pt")
    R.restore_payload(payload, state, modules, opts, priv)
    for module in modules.values(): module.eval()
    return state, modules, opts, priv


def intervention_rows(cache):
    rows = []
    for fold in range(1, 11):
        for group in R.GROUPS:
            state, modules, opts, priv = load_best(fold, group)
            module = modules[group]
            with torch.no_grad():
                base_out, cam, _ = R.capture_base_forward(state, False)
                base_metrics = R.metrics(base_out["logp"], state["test"], state["y"])
                base_pred = np.asarray(base_metrics["pred"], dtype=int)
                y = cache[fold]["y"]
                data = R.load_data()
                original = state["bp_windows"]
                valid = torch.arange(original.shape[1], device=R.DEVICE).unsqueeze(0) < state["bp_counts"].unsqueeze(1)
                mean = original.masked_fill(~valid.unsqueeze(-1), 0).sum(1) / state["bp_counts"].to(original.dtype).unsqueeze(1)
                constant = mean[:, None, :].expand_as(original).clone()
                shuffled = original.clone()
                for i, n in enumerate(state["bp_counts"].detach().cpu().tolist()):
                    if int(n) > 1:
                        shuffled[i, :int(n)] = torch.roll(shuffled[i, :int(n)], shifts=1, dims=0)
                masks = {
                    "full": None,
                    "PSDC-Off": "force_zero",
                    "Static-Counterfactual": "static",
                    "Order-Shuffle": "shuffle",
                    "Prompt6-Off": "prompt6",
                    "Prompt8-Off": "prompt8",
                }
                test_index_device = torch.as_tensor(state["test_np"], dtype=torch.long, device=R.DEVICE)
                for name, mode in masks.items():
                    state["bp_windows"] = original
                    prompt_mask = None
                    if mode == "force_zero":
                        logp, delta, diag, _ = R.psdc_forward(state, module, base_out["logp"], cam.detach(), force_zero=True)
                    else:
                        if mode == "static": state["bp_windows"] = constant
                        elif mode == "shuffle": state["bp_windows"] = shuffled
                        if mode == "prompt6":
                            prompt_mask = torch.ones(8, device=R.DEVICE); prompt_mask[5] = 0
                        elif mode == "prompt8":
                            prompt_mask = torch.ones(8, device=R.DEVICE); prompt_mask[7] = 0
                        logp, delta, diag, _ = R.psdc_forward(state, module, base_out["logp"], cam.detach(), prompt_mask=prompt_mask)
                    metric = R.metrics(logp, state["test"], state["y"])
                    pred = np.asarray(metric["pred"], dtype=int)
                    row = {"fold": fold, "group": group, "intervention": name,
                           **{k: float(metric[k]) for k in METRIC_KEYS},
                           **{f"delta_vs_base_{k}": float(metric[k] - base_metrics[k]) for k in METRIC_KEYS},
                           "prediction_flip_vs_base": int(np.sum(pred != base_pred)),
                           "base_wrong_aug_correct": int(np.sum((base_pred != y) & (pred == y))),
                           "base_correct_aug_wrong": int(np.sum((base_pred == y) & (pred != y))),
                           "dynamic_difference_norm_mean": float(diag["difference"].index_select(0, test_index_device).square().sum(dim=(1, 2)).sqrt().mean().cpu())}
                    rows.append(row)
                state["bp_windows"] = original
    return rows


def site_rows(cache):
    data = R.load_data()
    rows = []
    for fold in range(1, 11):
        indices = cache[fold]["test"]
        y = cache[fold]["y"]
        for group in ("B0", "G1", "G2"):
            pred = np.asarray(read_task(group, fold)["test_prediction"], dtype=int)
            for field, values in (("site", data["site"]), ("TR", data["tr"]), ("timepoints", data["timepoints"]), ("window_count", data["counts"].detach().cpu().numpy())):
                vals = np.asarray(values)[indices]
                for level in np.unique(vals):
                    mask = vals == level
                    if mask.sum() == 0: continue
                    rows.append({"fold": fold, "group": group, "stratum": field, "level": str(level), "n": int(mask.sum()), "accuracy": float((pred[mask] == y[mask]).mean()), "asd_rate": float((y[mask] == 0).mean())})
    return rows


def write_csv(name, rows):
    R.atomic_csv(RESULTS / name, rows)


def main():
    cache = fold_cache()
    fold_rows = []
    pooled = {}
    for group in ("B0", "G1", "G2"):
        vals = {k: [] for k in METRIC_KEYS}
        for fold in range(1, 11):
            d = read_task(group, fold)
            row = {"group": group, "fold": fold, **{k: float(d["metrics"][k]) for k in METRIC_KEYS}, "best_epoch_1based": d.get("best_epoch_1based")}
            fold_rows.append(row)
            for k in METRIC_KEYS: vals[k].append(row[k])
        pooled[group] = pooled_metrics(group, range(1, 11), cache)
        row = {"group": group, "fold": "mean", **{k: mean_sd(vals[k])[0] for k in METRIC_KEYS}, "best_epoch_1based": ""}
        fold_rows.append(row)
        row = {"group": group, "fold": "sd_population", **{k: mean_sd(vals[k])[1] for k in METRIC_KEYS}, "best_epoch_1based": ""}
        fold_rows.append(row)
    write_csv("fold_metrics.csv", fold_rows)

    comparisons = []
    for left, right, label in (("G1", "B0", "G1-B0"), ("G2", "B0", "G2-B0"), ("G2", "G1", "G2-G1")):
        for k in METRIC_KEYS:
            diffs = np.asarray([read_task(left, f)["metrics"][k] - read_task(right, f)["metrics"][k] for f in range(1, 11)], dtype=float)
            lo, hi = ci95(diffs)
            comparisons.append({"comparison": label, "metric": k, "mean_diff": float(diffs.mean()), "sd_population": float(diffs.std(ddof=0)), "ci95_desc_low": lo, "ci95_desc_high": hi, "W": int((diffs > 0).sum()), "T": int((diffs == 0).sum()), "L": int((diffs < 0).sum()), "pooled_diff": float(pooled[left][k] - pooled[right][k]), "exclude_fold2_mean_diff": float(np.delete(diffs, 1).mean())})
    write_csv("paired_comparisons.csv", comparisons)

    mechanisms = []
    for group in R.GROUPS:
        for fold in range(1, 11):
            d = read_task(group, fold)
            base = read_task("B0", fold)
            y = cache[fold]["y"]
            bp, gp = np.asarray(base["test_prediction"], int), np.asarray(d["test_prediction"], int)
            mech = dict(d.get("mechanism") or {})
            mech.update({"group": group, "fold": fold, "best_epoch_1based": d.get("best_epoch_1based"), "base_wrong_aug_correct_recomputed": int(((bp != y) & (gp == y)).sum()), "base_correct_aug_wrong_recomputed": int(((bp == y) & (gp != y)).sum())})
            mechanisms.append(mech)
    write_csv("mechanism.csv", mechanisms)

    alignment_rows = []
    for group in R.GROUPS:
        for fold in range(1, 11):
            a = read_task(group, fold)["alignment_at_B0_historical_best"]
            if not a: continue
            base = a["base_metrics"]
            alignment_rows.append({"group": group, "fold": fold, "epoch_1based": a["epoch_1based"], **{f"aligned_{k}": a["metrics"][k] for k in METRIC_KEYS}, **{f"base_{k}": base[k] for k in METRIC_KEYS}, **{f"delta_{k}": a["metrics"][k] - base[k] for k in METRIC_KEYS}, "checkpoint_sha256": a.get("checkpoint_sha256")})
    write_csv("historical_epoch_alignment.csv", alignment_rows)

    interventions = intervention_rows(cache)
    write_csv("interventions.csv", interventions)
    write_csv("site_analysis.csv", site_rows(cache))

    # Aggregate mechanism summaries used directly in the report.
    aggregate = []
    for group in R.GROUPS:
        rows = [x for x in mechanisms if x["group"] == group]
        for key in ("alpha", "gate_mean", "gate_std", "relative_residual_mean", "relative_residual_p95", "relative_residual_max", "raw_relative_residual_mean", "residual_limit_fraction", "delta_nonzero_fraction", "prediction_flip_fraction", "dynamic_difference_norm_mean", "base_wrong_aug_correct_recomputed", "base_correct_aug_wrong_recomputed"):
            vals = [float(r.get(key, 0.0)) for r in rows]
            aggregate.append({"group": group, "field": key, "mean": float(np.mean(vals)), "sd_population": float(np.std(vals, ddof=0)), "min": float(np.min(vals)), "max": float(np.max(vals))})
    write_csv("mechanism_summary.csv", aggregate)

    b0 = {k: mean_sd([read_task("B0", f)["metrics"][k] for f in range(1, 11)]) for k in METRIC_KEYS}
    g1 = {k: mean_sd([read_task("G1", f)["metrics"][k] for f in range(1, 11)]) for k in METRIC_KEYS}
    g2 = {k: mean_sd([read_task("G2", f)["metrics"][k] for f in range(1, 11)]) for k in METRIC_KEYS}
    gate0 = json.loads((RESULTS / "audit_status.json").read_text(encoding="utf-8")).get("status") == "PASS"
    gate1 = all(g2[k][0] + 1e-12 >= {"acc": 0.921891, "hard_auc": 0.920294, "prob_auc": 0.934946}[k] for k in ("acc", "hard_auc", "prob_auc"))
    net_g1 = sum(x["base_wrong_aug_correct_recomputed"] - x["base_correct_aug_wrong_recomputed"] for x in mechanisms if x["group"] == "G1")
    net_g2 = sum(x["base_wrong_aug_correct_recomputed"] - x["base_correct_aug_wrong_recomputed"] for x in mechanisms if x["group"] == "G2")
    gate2 = bool((net_g1 > 0 or net_g2 > 0) and any(x["intervention"] == "Order-Shuffle" and abs(x["delta_vs_base_acc"]) > 0 for x in interventions))
    gate3 = bool(g2["acc"][0] > g1["acc"][0] and g2["prob_auc"][0] >= g1["prob_auc"][0])
    if not gate0 or not gate1:
        decision = "NO_GO_BP_CPSDC"
    elif not gate2 or not gate3:
        decision = "BASELINE_RECOVERED_NO_INCREMENT"
    else:
        decision = "GO_BP_CPSDC_INCREMENTAL"
    decision_doc = {"gate0_engineering": gate0, "gate1_no_baseline_drop": gate1, "gate2_dynamic_increment": gate2, "gate3_medical_prompt_increment": gate3, "decision": decision, "net_corrections_G1": net_g1, "net_corrections_G2": net_g2, "pooled": pooled, "mean_sd": {"B0": b0, "G1": g1, "G2": g2}, "fold2_excluded": "reported in paired_comparisons.csv"}
    R.atomic_json(RESULTS / "final_decision.json", decision_doc)

    lines = ["# Experiment 80.2 BP-CPSDC 结果报告", "", f"科研判定：**{decision}**", "", "## 六指标（10折 mean ± population SD）", "", "| 方法 | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, vals in (("B0 E74.4-G2", b0), ("G1 BP-CPSDC-Anonymous", g1), ("G2 BP-CPSDC-Medical", g2)):
        lines.append("| " + name + " | " + " | ".join(f"{vals[k][0]*100:.4f} ± {vals[k][1]*100:.4f}" for k in METRIC_KEYS) + " |")
    lines += ["", "## Pooled OOF", "", "| 方法 | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, group in (("B0", "B0"), ("G1", "G1"), ("G2", "G2")):
        lines.append("| " + name + " | " + " | ".join(f"{pooled[group][k]*100:.4f}" for k in METRIC_KEYS) + " |")
    lines += ["", "## 审计与机制", "", f"- Gate 0 工程前提：{'PASS' if gate0 else 'FAIL'}；独立 Best/Latest/反事实审计见 `audit_status.json`。", f"- Gate 1 基座不下降：{'PASS' if gate1 else 'FAIL'}。", f"- 净纠错（纠正减误伤）：G1={net_g1}，G2={net_g2}；动态顺序干预证据：{'PASS' if gate2 else 'FAIL'}。", f"- 医学 Prompt 相对匿名：{'PASS' if gate3 else 'FAIL'}。", "- Fold2 排除、逐折 W/T/L、描述性95%区间见 `paired_comparisons.csv`；站点/TR/扫描长度/窗口数分层见 `site_analysis.csv`。", "- FD/DVARS 未出现在已核验 E80.1 动态缓存字段中，因此本轮不虚构该混杂分析。", "", "所有结果均来自正式十折与独立重算，未用测试标签反向修改门控、Prompt 或残差上限。"]
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"ANALYSIS PASS decision={decision}", flush=True)


if __name__ == "__main__":
    main()
