"""Post-training analysis for Experiment 79.3.

This script consumes only the independently replayed audit artifact and the
read-only E74.4-G2 fold table. It does not select epochs or alter weights.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BASE_RESULTS = Path("/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr/results")
AUDIT_PATH = RESULTS / "audit_status.json"
BASE_FOLD_PATH = BASE_RESULTS / "fold_metrics.csv"
GROUPS = ["E79.3-G1-CLDCM", "E79.3-G2-CPDEM"]
METRICS = ["acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1"]
METRIC_LABELS = {"acc": "ACC", "hard_auc": "Hard-AUC/BA", "prob_auc": "Prob-AUC", "sen": "SEN-ASD", "spe": "SPE-TD", "asd_f1": "ASD-F1"}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def mean_sd(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def tcrit(n: int) -> float:
    return {10: 2.2621571627, 9: 2.3060041350}.get(n, 2.2621571627)


def paired_summary(diffs: list[float]) -> tuple[float, float, float, float, int, int, int]:
    a = np.asarray(diffs, dtype=float); n = len(a)
    mu = float(a.mean()); sd = float(a.std(ddof=1)) if n > 1 else 0.0
    half = tcrit(n) * sd / math.sqrt(n) if n else float("nan")
    wins = int(np.sum(a > 1e-12)); losses = int(np.sum(a < -1e-12)); ties = n - wins - losses
    return mu, sd, mu - half, mu + half, wins, ties, losses


def load_base() -> dict[int, dict[str, float]]:
    out = {}
    with BASE_FOLD_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("group") == "E74.4-G2-Positive-DCR":
                out[int(row["fold"])] = {m: float(row[m]) for m in METRICS}
    if len(out) != 10: raise RuntimeError(f"expected 10 historical G2 rows, found {len(out)}")
    return out


def load_audit() -> dict:
    data = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if data.get("status") != "PASS" or data.get("best_checkpoints_audited") != 40 or data.get("latest_audited") != 10:
        raise RuntimeError(f"incomplete independent audit: {data.get('status')}")
    return data


def metric_value(row: dict, metric: str) -> float:
    return float(row["replay_metrics"][metric])


def write_fold_metrics(audit: dict, base: dict[int, dict[str, float]]) -> list[dict]:
    rows = []
    for fold in range(1, 11):
        rows.append({"method": "E74.4-G2", "ledger": "historical", "group": "E74.4-G2-Positive-DCR", "fold": fold, "epoch": "historical", **{m: base[fold][m] * 100 for m in METRICS}})
    for r in audit["best_rows"]:
        tr = r["transition"]
        rows.append({"method": r["group"], "ledger": r["ledger"], "group": r["group"], "fold": int(r["fold"]), "epoch": int(r["epoch"]),
                     **{m: metric_value(r, m) * 100 for m in METRICS}, "base_errors": tr["base_errors"], "method_errors": tr["method_errors"],
                     "error_to_correct": tr["error_to_correct"], "correct_to_error": tr["correct_to_error"], "net_correction": tr["net_correction"],
                     "hard_prediction_changes": tr["hard_prediction_changes"], "gate_mean": r["gate_mean"], "delta_ratio_mean": r["delta_ratio_mean"],
                     "cap_fraction": r["cap_fraction"], "memory_count": r["memory_count"], "memory_ids_count": r["memory_ids_count"]})
    fields = ["method", "ledger", "group", "fold", "epoch"] + METRICS + ["base_errors", "method_errors", "error_to_correct", "correct_to_error", "net_correction", "hard_prediction_changes", "gate_mean", "delta_ratio_mean", "cap_fraction", "memory_count", "memory_ids_count"]
    write_csv(RESULTS / "fold_metrics.csv", rows, fields); return rows


def write_summary(audit: dict, base: dict[int, dict[str, float]]) -> list[dict]:
    specs = [("E74.4-G2", "historical", None), ("G1 Base-Anchored", "base_anchored", GROUPS[0]), ("G2 Base-Anchored", "base_anchored", GROUPS[1]), ("G1 Method-Selected", "method", GROUPS[0]), ("G2 Method-Selected", "method", GROUPS[1])]
    rows = []
    for name, ledger, group in specs:
        vals = {m: ([base[f][m] for f in range(1, 11)] if group is None else [metric_value(r, m) for r in audit["best_rows"] if r["ledger"] == ledger and r["group"] == group]) for m in METRICS}
        row = {"method": name, "ledger": ledger, "group": group or "E74.4-G2-Positive-DCR", "n_folds": 10}
        for m in METRICS:
            mu, sd = mean_sd(vals[m]); row[f"{m}_mean_pct"] = mu * 100; row[f"{m}_sd_pct"] = sd * 100
        rows.append(row)
    fields = ["method", "ledger", "group", "n_folds"] + [x for m in METRICS for x in (f"{m}_mean_pct", f"{m}_sd_pct")]
    write_csv(RESULTS / "summary.csv", rows, fields); return rows


def write_paired(audit: dict, base: dict[int, dict[str, float]]) -> list[dict]:
    specs = [("G1-base", GROUPS[0], "base"), ("G2-base", GROUPS[1], "base"), ("G2-G1", GROUPS[1], GROUPS[0])]
    rows = []
    for ledger in ("base_anchored", "method"):
        for subset, folds in (("all10", list(range(1, 11))), ("exclude_fold2", [f for f in range(1, 11) if f != 2])):
            for comp, lhs, rhs in specs:
                for metric in METRICS:
                    diffs = []
                    for f in folds:
                        lr = next(r for r in audit["best_rows"] if r["ledger"] == ledger and r["group"] == lhs and int(r["fold"]) == f)
                        lv = metric_value(lr, metric); rv = base[f][metric] if rhs == "base" else metric_value(next(r for r in audit["best_rows"] if r["ledger"] == ledger and r["group"] == rhs and int(r["fold"]) == f), metric)
                        diffs.append((lv - rv) * 100)
                    mu, sd, lo, hi, w, t, l = paired_summary(diffs)
                    rows.append({"ledger": ledger, "subset": subset, "comparison": comp, "metric": METRIC_LABELS[metric], "n": len(diffs), "mean_diff_pp": mu, "sd_diff_pp": sd, "ci95_low_pp": lo, "ci95_high_pp": hi, "wins": w, "ties": t, "losses": l, "fold_diffs_pp": json.dumps([round(x, 8) for x in diffs])})
    fields = ["ledger", "subset", "comparison", "metric", "n", "mean_diff_pp", "sd_diff_pp", "ci95_low_pp", "ci95_high_pp", "wins", "ties", "losses", "fold_diffs_pp"]
    write_csv(RESULTS / "paired_comparisons.csv", rows, fields); return rows


def write_corrections(audit: dict) -> list[dict]:
    rows = []
    keys = ("base_errors", "method_errors", "error_to_correct", "correct_to_error", "net_correction", "hard_prediction_changes", "asd_error_corrected", "td_error_corrected")
    for r in audit["best_rows"]:
        rows.append({"ledger": r["ledger"], "group": r["group"], "fold": int(r["fold"]), "epoch": int(r["epoch"]), **r["transition"]})
    for ledger in ("base_anchored", "method"):
        for group in GROUPS:
            rr = [x for x in rows if x["ledger"] == ledger and x["group"] == group]
            for label, sub in (("ALL10", rr), ("EXCLUDE_FOLD2", [x for x in rr if int(x["fold"]) != 2])):
                out = {"ledger": ledger, "group": group, "fold": label, "epoch": ""}
                out.update({k: sum(float(x[k]) for x in sub) for k in keys})
                out["correct_retention"] = float(np.mean([x["correct_retention"] for x in sub])); out["mean_abs_prob_delta"] = float(np.mean([x["mean_abs_prob_delta"] for x in sub]))
                rows.append(out)
    fields = ["ledger", "group", "fold", "epoch", *keys, "correct_retention", "mean_abs_prob_delta"]
    write_csv(RESULTS / "correction_analysis.csv", rows, fields); return rows


def memory_summary(audit: dict) -> list[dict]:
    rows = []
    for r in audit["best_rows"]:
        group, ledger, fold = r["group"], r["ledger"], int(r["fold"]); idx = GROUPS.index(group)
        p = ROOT / ".runtime" / "checkpoints" / f"{group}_f{fold:02d}.{ledger}.pt"; rec = torch.load(p, map_location="cpu", weights_only=False); m = rec["memory_used"][idx]
        row = {"ledger": ledger, "group": group, "fold": fold, "epoch": int(r["epoch"]), "version": int(m.get("version", 0)) if m else 0, "entry_count": 0, "unique_patient_count": 0, "source_count": len(m.get("source_ids", [])) if m else 0, "mean_reliability": "", "mean_support": "", "mean_similarity": "", "mean_separation": "", "candidate_count": "", "rejected_count": "", "mean_evidence_dims": "", "mean_joint_drop": "", "component_count": "", "mean_component_members": "", "mean_variance": "", "mean_prior": "", "fold2_excluded_present": False}
        if idx == 0:
            ids = list(m.get("ids", [])) if m else []; flat = [int(x) for pair in ids for x in pair]
            row.update(entry_count=len(ids), unique_patient_count=len(set(flat)), mean_reliability=float(torch.as_tensor(m["reliability"]).mean()) if ids else "", mean_support=float(np.mean([len(x) for x in m.get("support_ids", [])])) if ids else "", mean_similarity=float(torch.as_tensor(m["similarity"]).mean()) if ids else "", mean_separation=float(torch.as_tensor(m["local_separation"]).mean()) if ids else "", fold2_excluded_present=bool(set(flat) & {376, 406}) if fold == 2 else False)
        elif idx == 1:
            ids = list(m.get("ids", [])) if m else []; mask = torch.as_tensor(m.get("mask", []), dtype=torch.float32)
            row.update(entry_count=len(ids), unique_patient_count=len(set(int(x) for x in ids)), mean_reliability=float(torch.as_tensor(m["reliability"]).mean()) if ids else "", candidate_count=int(m.get("candidate_count", len(ids))), rejected_count=int(m.get("rejected_count", 0)), mean_evidence_dims=float(mask.sum(1).mean()) if ids else "", mean_joint_drop=float(torch.as_tensor(m["joint_drop"]).mean()) if ids else "", fold2_excluded_present=bool(set(map(int, ids)) & {376, 406}) if fold == 2 else False)
        else:
            raise RuntimeError("unexpected memory index")
        rows.append(row)
    write_csv(RESULTS / "memory_analysis.csv", rows, list(rows[0].keys())); return rows


def write_interventions(audit: dict) -> list[dict]:
    full = {(r["ledger"], r["group"], int(r["fold"])): r for r in audit["best_rows"]}; rows = []
    for r in audit["interventions"]:
        f = full[(r["ledger"], r["group"], int(r["fold"]))]["replay_metrics"]; im = r["metrics"]
        rows.append({"fold": int(r["fold"]), "ledger": r["ledger"], "group": r["group"], "epoch": int(r["epoch"]), "intervention": r["intervention"], "full_acc_pct": f["acc"] * 100, "intervention_acc_pct": im["acc"] * 100, "full_prob_auc_pct": f["prob_auc"] * 100, "intervention_prob_auc_pct": im["prob_auc"] * 100, "acc_delta_pp": r["acc_delta_full_minus_intervention"] * 100, "hard_auc_delta_pp": r["hard_auc_delta"] * 100, "prob_auc_delta_pp": r["prob_auc_delta"] * 100, "hard_prediction_flips": r["hard_prediction_flips"], "prob_abs_delta_mean": r.get("prob_abs_delta_mean", ""), "delta_ratio_mean": r["delta_ratio_mean"]})
    write_csv(RESULTS / "intervention_analysis.csv", rows, list(rows[0].keys())); return rows


def write_mechanism(audit: dict) -> list[dict]:
    rows = []; ints = audit["interventions"]
    for ledger in ("base_anchored", "method"):
        for group in GROUPS:
            br = [r for r in audit["best_rows"] if r["ledger"] == ledger and r["group"] == group]; ir = [r for r in ints if r["ledger"] == ledger and r["group"] == group]
            row = {"ledger": ledger, "group": group, "n_folds": len(br), "mean_gate": float(np.mean([r["gate_mean"] for r in br])), "mean_delta_ratio": float(np.mean([r["delta_ratio_mean"] for r in br])), "p90_delta_ratio": float(np.quantile([r["delta_ratio_mean"] for r in br], .9)), "mean_cap_fraction": float(np.mean([r["cap_fraction"] for r in br])), "mean_memory_count": float(np.mean([r["memory_count"] for r in br])), "net_correction": sum(r["transition"]["net_correction"] for r in br), "error_to_correct": sum(r["transition"]["error_to_correct"] for r in br), "correct_to_error": sum(r["transition"]["correct_to_error"] for r in br), "hard_flips": sum(r["transition"]["hard_prediction_changes"] for r in br)}
            for name in sorted(set(x["intervention"] for x in ir)):
                rr = [x for x in ir if x["intervention"] == name]; key = name.lower().replace("-", "_").replace(" ", "_")
                row[f"{key}_prob_auc_delta_pp"] = float(np.mean([x["prob_auc_delta"] for x in rr]) * 100); row[f"{key}_hard_flips"] = int(sum(x["hard_prediction_flips"] for x in rr))
            rows.append(row)
    write_csv(RESULTS / "mechanism.csv", rows, sorted({k for r in rows for k in r})); return rows


def main() -> None:
    audit = load_audit(); base = load_base(); fold_rows = write_fold_metrics(audit, base); summary = write_summary(audit, base); paired = write_paired(audit, base); corrections = write_corrections(audit); memories = memory_summary(audit); interventions = write_interventions(audit); mechanism = write_mechanism(audit)
    payload = {"status": "PASS", "audit_status": audit["status"], "summary_rows": len(summary), "fold_rows": len(fold_rows), "paired_rows": len(paired), "correction_rows": len(corrections), "memory_rows": len(memories), "intervention_rows": len(interventions), "mechanism_rows": len(mechanism)}
    (RESULTS / "analysis_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
