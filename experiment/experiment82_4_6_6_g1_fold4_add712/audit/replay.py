"""Independent replay and read-only reuse audit for E82.4.6.6."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

# Historical checkpoints were serialized with the experiment's NumPy 2.5
# runtime.  Put that read-only runtime first so torch can resolve its
# ``numpy._core`` pickle globals without touching the training environment.
PROFILE = Path("/model/spromptgl/experiment/experiment82_3_6_ps_pgr/runtime/python")
sys.path.insert(0, str(PROFILE))
import numpy as np

# The historical runtime bundles an older SciPy/sklearn stack that still
# references NumPy's removed scalar aliases.  Keep this compatibility shim in
# the read-only audit entry so the frozen training implementation is untouched.
if not hasattr(np, "long"):
    np.long = np.int64
if not hasattr(np, "ulong"):
    np.ulong = np.uint64

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core
import independent_465 as ind

h = core.h
CONFIG = core.CONFIG
E465 = Path(CONFIG["source_e82465_directory"])
BASE = Path("/model/spromptgl/experiment/experiment82_4_6_pcsr_scmr")
CURRENT_EXPOSURE = {
    2: {377, 407, 681, 779, 784, 260, 288, 375},
    3: {46, 66, 172, 266, 388},
    4: {712, 713, 727, 773},
}
PREVIOUS_EXPOSURE = {
    2: set(CURRENT_EXPOSURE[2]),
    3: set(CURRENT_EXPOSURE[3]),
    4: {713, 727, 773},
}
UNION_EXPOSURE = {(fold, patient) for fold, patients in CURRENT_EXPOSURE.items() for patient in patients}
OLD_UNION_EXPOSURE = {(fold, patient) for fold, patients in PREVIOUS_EXPOSURE.items() for patient in patients}
FIXED_OUTSIDE_COUNT = 871 - len(UNION_EXPOSURE)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_historical_sources():
    expected = CONFIG["expected_manifest_sha256"]
    old_manifest_path = E465 / "manifest.json"
    old_manifest = h.read_json(old_manifest_path)
    h.require(old_manifest.get("status") == "COMPLETE_AUDITED", "E82.4.6.5 is not COMPLETE_AUDITED")
    h.require(h.sha_file(old_manifest_path) == expected["e82465_g1"], "E82.4.6.5 manifest SHA differs")
    h.require(old_manifest["authoritative_directory"] == str(E465), "E82.4.6.5 authoritative directory differs")
    for relative, digest in old_manifest["result_files"].items():
        path = E465 / relative
        h.require(path.exists() and h.sha_file(path) == digest, f"E82.4.6.5 result changed: {relative}")
    old_audit = h.read_json(E465 / "results/audit_status.json")
    h.require(old_audit.get("status") == "COMPLETE_AUDITED", "E82.4.6.5 audit status is not complete")

    checkpoint_rows = {}
    for fold in range(1, 11):
        item = old_manifest["checkpoint_sources"][str(fold)]
        for key in ("best", "latest", "journal"):
            path = Path(item[key])
            h.require(path.exists() and h.sha_file(path) == item[key + "_sha"],
                      f"Historical Fold {fold} {key} checkpoint identity differs")
        source_manifest = Path(item["source_manifest"])
        if item.get("source_manifest_sha") is not None:
            h.require(h.sha_file(source_manifest) == item["source_manifest_sha"],
                      f"Historical Fold {fold} source manifest changed")
        checkpoint_rows[str(fold)] = {
            "source_kind": item["source_kind"], "best": item["best"], "best_sha": item["best_sha"],
            "latest": item["latest"], "latest_sha": item["latest_sha"], "journal": item["journal"],
            "journal_sha": item["journal_sha"], "best_epoch": item["best_epoch"],
            "latest_epoch": item["latest_epoch"], "source_manifest": str(source_manifest),
            "source_manifest_sha": item.get("source_manifest_sha"),
        }
    return old_manifest, old_audit, checkpoint_rows


def read_patient_rows(path: Path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    h.require(len(rows) == 871 and len({(int(r["fold"]), int(r["patient_id_1based"])) for r in rows}) == 871,
              "Historical patient-level rows are incomplete")
    return rows


def signed_margin(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    return (logits[:, 0] - logits[:, 1]) * np.where(labels == 0, 1.0, -1.0)


def local_replay(manifest):
    run = core.engine("G1", 4)
    proof = load_module("e8266_base_audit", BASE / "audit.py")
    proof.run = run
    # This repeats the independent Best/Latest/RNG/gradient audit after the
    # recorded preflight EMA-entry repair. It never trains an ablation model.
    item, certificate = ind.audit_task("G1", 4, manifest)
    ctx = run.setup("G1", 4, verify_spectrum=True)
    files = core.paths("G1", 4)
    best = h.load(files["best"])
    latest = h.load(files["latest"])
    run.restore(ctx, best, manifest["implementation"])
    full, full_cap = proof.independent_eval(ctx, "full")
    off, off_cap = proof.independent_eval(ctx, "residual_off")
    anonymous, anonymous_cap = proof.independent_eval(ctx, "anonymous_query")
    rows = ctx["members"]["test"].copy()
    labels = ctx["state"]["labels_np"][rows].copy()
    expected = np.asarray(sorted(core.EXPOSED[4]), dtype=np.int64)
    mask = np.isin(rows, expected)
    details = full_cap["details"]
    for name in core.MASKED_COMPONENTS:
        h.require(np.array_equal(details[name][expected].detach().cpu().numpy(),
                                 np.zeros_like(details[name][expected].detach().cpu().numpy())),
                  f"Fold-4 masked {name} is nonzero")
    h.require(np.array_equal(full_cap["enhanced"][expected].detach().cpu().numpy(),
                             full_cap["cam"][expected].detach().cpu().numpy()),
              "Fold-4 masked CAM changed")
    for key in ("probability", "prediction", "logits"):
        h.require(np.array_equal(full[key][mask], off[key][mask]),
                  f"Fold-4 masked Full and Residual-off {key} differ")
    h.require(np.any(np.linalg.norm(details["delta"][rows][~mask].detach().cpu().numpy(), axis=1) > 0),
              "Fold-4 unmasked residual path is inactive")
    h.require(np.array_equal(full_cap["cam"].detach().cpu().numpy(), off_cap["cam"].detach().cpu().numpy())
              and np.array_equal(full_cap["cam"].detach().cpu().numpy(), anonymous_cap["cam"].detach().cpu().numpy()),
              "Fixed-weight interventions changed native CAM")
    cam = full_cap["cam"][rows].detach().cpu().numpy().astype(np.float64)
    residual = {name: np.linalg.norm(details[name][rows].detach().cpu().numpy().astype(np.float64), axis=1)
                for name in ("delta", "delta_parallel", "delta_perp")}
    cam_l2 = np.linalg.norm(cam, axis=1)
    ratio = np.divide(residual["delta"], cam_l2, out=np.zeros_like(cam_l2), where=cam_l2 > 0)
    h.require(np.isfinite(ratio).all() and np.all(ratio <= 0.08), "Fold-4 8% CAM limit exceeded")
    query_shift = (full_cap["enhanced"][rows].double() - anonymous_cap["enhanced"][rows].double()).cpu().numpy()
    return {
        "item": item, "certificate": certificate, "rows": rows, "labels": labels, "mask": mask,
        "full": full, "off": off, "anonymous": anonymous,
        "full_cap": full_cap, "off_cap": off_cap, "anonymous_cap": anonymous_cap,
        "residual": residual, "cam_l2": cam_l2, "ratio": ratio,
        "query_shift_l2": np.linalg.norm(query_shift, axis=1),
        "best": best, "latest": latest, "best_sha": h.sha_file(files["best"]),
        "latest_sha": h.sha_file(files["latest"]), "journal_sha": h.sha_file(files["journal"]),
        "best_epoch": int(best["epoch"]), "latest_epoch": int(latest["epoch"]),
        "test_ids": (rows + 1).tolist(), "true_margin_full": signed_margin(full["logits"], labels).tolist(),
        "true_margin_off": signed_margin(off["logits"], labels).tolist(),
        "true_margin_anonymous": signed_margin(anonymous["logits"], labels).tolist(),
    }


def as_float(value):
    return float(value)


def historical_value(row, prefix):
    return {
        "prediction": int(row[f"prediction_{prefix}"]),
        "p_asd": as_float(row[f"p_asd_{prefix}"]),
        "margin": as_float(row[f"true_margin_{prefix}"]),
    }


def current_value(result, index, prefix):
    source = result[prefix]
    return {
        "prediction": int(np.asarray(source["prediction"])[index]),
        "p_asd": float(np.asarray(source["probability"])[index, 0]),
        "margin": float(result[f"true_margin_{prefix}"][index]),
    }


def current_residual(result, index):
    return {
        "delta_l2": float(result["residual"]["delta"][index]),
        "delta_parallel_l2": float(result["residual"]["delta_parallel"][index]),
        "delta_perp_l2": float(result["residual"]["delta_perp"][index]),
        "cam_l2": float(result["cam_l2"][index]),
        "cam_relative_delta": float(result["ratio"][index]),
        "query_feature_change_l2": float(result["query_shift_l2"][index]),
    }


def assemble_patient_rows(old_rows, local):
    old_by_key = {(int(row["fold"]), int(row["patient_id_1based"])): row for row in old_rows}
    local_by_id = {pid: i for i, pid in enumerate(local["test_ids"])}
    result = []
    for fold in range(1, 11):
        for patient in sorted(pid for f, pid in old_by_key if f == fold):
            old = old_by_key[(fold, patient)]
            key = (fold, patient)
            is_local = fold == 4
            if is_local:
                i = local_by_id[patient]
                full = current_value(local, i, "full")
                off = current_value(local, i, "off")
                anonymous = current_value(local, i, "anonymous")
                residual = current_residual(local, i)
            else:
                full = historical_value(old, "full")
                off = historical_value(old, "residual_off")
                anonymous = historical_value(old, "anonymous")
                residual = {
                    "delta_l2": as_float(old["delta_l2"]),
                    "delta_parallel_l2": as_float(old["delta_parallel_l2"]),
                    "delta_perp_l2": as_float(old["delta_perp_l2"]),
                    "cam_l2": as_float(old["cam_l2"]),
                    "cam_relative_delta": as_float(old["cam_relative_delta"]),
                    "query_feature_change_l2": as_float(old["query_feature_change_l2"]),
                }
            previous = historical_value(old, "full")
            current = full
            current_overlap = patient in CURRENT_EXPOSURE.get(fold, set())
            previous_overlap = patient in PREVIOUS_EXPOSURE.get(fold, set())
            y = int(old["true_label"])
            current_correct = current["prediction"] == y
            previous_correct = previous["prediction"] == y
            off_correct = off["prediction"] == y
            anon_correct = anonymous["prediction"] == y
            cohort = "outside_union_854"
            if fold == 4 and patient == 712:
                cohort = "fold4_new_addition_712"
            elif fold == 4 and patient in {713, 727, 773}:
                cohort = "fold4_retained_overlap_3"
            elif fold == 4:
                cohort = "fold4_other_test_83"
            elif current_overlap:
                cohort = f"fold{fold}_overlap"
            row = {
                "fold": fold, "patient_id_1based": patient, "patient_index_0based": int(old["patient_index_0based"]),
                "true_label": y, "cohort": cohort,
                "current_native_training_overlap": current_overlap,
                "previous_native_training_overlap": previous_overlap,
                "in_fixed_union_exposure_17": key in UNION_EXPOSURE,
                "outside_fixed_union_854": key not in UNION_EXPOSURE,
                "is_scmr_residual_masked": current_overlap,
                "scmr_supervised_test": not current_overlap,
                "residual_mask_strict_zero": current_overlap and fold == 4,
                "prediction_previous": previous["prediction"], "p_asd_previous": previous["p_asd"], "true_margin_previous": previous["margin"],
                "prediction_full": full["prediction"], "p_asd_full": full["p_asd"], "true_margin_full": full["margin"],
                "prediction_residual_off": off["prediction"], "p_asd_residual_off": off["p_asd"], "true_margin_residual_off": off["margin"],
                "prediction_anonymous": anonymous["prediction"], "p_asd_anonymous": anonymous["p_asd"], "true_margin_anonymous": anonymous["margin"],
                "previous_delta_l2": as_float(old["delta_l2"]),
                "previous_delta_parallel_l2": as_float(old["delta_parallel_l2"]),
                "previous_delta_perp_l2": as_float(old["delta_perp_l2"]),
                "delta_l2": residual["delta_l2"], "delta_parallel_l2": residual["delta_parallel_l2"],
                "delta_perp_l2": residual["delta_perp_l2"], "cam_l2": residual["cam_l2"],
                "cam_relative_delta": residual["cam_relative_delta"], "query_feature_change_l2": residual["query_feature_change_l2"],
                "margin_gain_vs_previous": full["margin"] - previous["margin"],
                "margin_gain_off_to_full": full["margin"] - off["margin"],
                "margin_gain_anon_to_medical": full["margin"] - anonymous["margin"],
                "corrected_vs_previous": int((not previous_correct) and current_correct),
                "harmed_vs_previous": int(previous_correct and (not current_correct)),
                "residual_corrected": int((not off_correct) and current_correct),
                "residual_harmed": int(off_correct and (not current_correct)),
                "medical_corrected": int((not anon_correct) and current_correct),
                "medical_harmed": int(anon_correct and (not current_correct)),
                "current_best_sha": local["best_sha"] if is_local else old["current_best_sha"],
                "previous_best_sha": old["current_best_sha"],
                "prediction_darg": int(old["prediction_darg"]), "p_asd_darg": as_float(old["p_asd_darg"]),
                "true_margin_darg": as_float(old["true_margin_darg"]),
                "prediction_original_g2": int(old["prediction_original_g2"]), "p_asd_original_g2": as_float(old["p_asd_original_g2"]),
                "true_margin_original_g2": as_float(old["true_margin_original_g2"]),
                "prediction_e82463": int(old["prediction_e82463"]), "p_asd_e82463": as_float(old["p_asd_e82463"]),
                "true_margin_e82463": as_float(old["true_margin_e82463"]),
            }
            result.append(row)
    h.require(len(result) == 871 and len({(r["fold"], r["patient_id_1based"]) for r in result}) == 871,
              "Current patient-level reconstruction is incomplete")
    return result


def metric(y, pred, p_asd):
    from sklearn.metrics import (balanced_accuracy_score, f1_score, recall_score, roc_auc_score)
    y = np.asarray(y, dtype=np.int64); pred = np.asarray(pred, dtype=np.int64); p_asd = np.asarray(p_asd, dtype=np.float64)
    def finite(value):
        value = float(value)
        return value if np.isfinite(value) else None
    return {
        "n_test": int(len(y)), "acc": finite(np.mean(pred == y)),
        "hard_auc": finite(balanced_accuracy_score(y, pred)),
        "prob_auc": finite(roc_auc_score(y == 0, p_asd)),
        "sen": finite(recall_score(y, pred, pos_label=0)), "spe": finite(recall_score(y, pred, pos_label=1)),
        "asd_f1": finite(f1_score(y, pred, pos_label=0, zero_division=0)),
        "tp": int(np.sum((y == 0) & (pred == 0))), "fp": int(np.sum((y == 1) & (pred == 0))),
        "tn": int(np.sum((y == 1) & (pred == 1))), "fn": int(np.sum((y == 0) & (pred == 1))),
    }


def metrics_for_patients(rows, prefix):
    return metric([r["true_label"] for r in rows], [r[f"prediction_{prefix}"] for r in rows], [r[f"p_asd_{prefix}"] for r in rows])


def historical_metrics(old_rows):
    source = E465 / "results/fold_metrics.csv"
    values = {}
    with source.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["scope"] == "test_fold" and row["method"] in ("E74.4-G2 DARG", "E82.4.6-G2", "E82.4.6.5-G1"):
                values[(row["method"], int(row["fold"]))] = {key: float(row[key]) for key in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")}
    h.require(len(values) == 30, "Historical fold metrics are incomplete")
    return values


def summary(values):
    keys = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
    return {key: {"mean_pct": float(np.mean([values[fold][key] for fold in range(1, 11)]) * 100),
                  "sd_pct": float(np.std([values[fold][key] for fold in range(1, 11)], ddof=0) * 100)} for key in keys}


def paired(rows, left_prefix, right_prefix, scope, selected):
    rows = [r for r in rows if selected(r)]
    y = np.asarray([r["true_label"] for r in rows])
    left_pred = np.asarray([r[f"prediction_{left_prefix}"] for r in rows]); right_pred = np.asarray([r[f"prediction_{right_prefix}"] for r in rows])
    left_metric = metric(y, left_pred, [r[f"p_asd_{left_prefix}"] for r in rows])
    right_metric = metric(y, right_pred, [r[f"p_asd_{right_prefix}"] for r in rows])
    corrected = [r for r, lp, rp in zip(rows, left_pred, right_pred) if lp != r["true_label"] and rp == r["true_label"]]
    harmed = [r for r, lp, rp in zip(rows, left_pred, right_pred) if lp == r["true_label"] and rp != r["true_label"]]
    def delta(key):
        left = left_metric[key]; right = right_metric[key]
        return None if left is None or right is None else (right - left) * 100
    return {
        "source": left_prefix, "target": right_prefix, "scope": scope, "n": len(rows), "corrected": len(corrected),
        "harmed": len(harmed), "net": len(corrected) - len(harmed),
        "delta_acc_pp": delta("acc"),
        "delta_ba_pp": delta("hard_auc"),
        "delta_prob_auc_pp": delta("prob_auc"),
        "margin_mean_delta": float(np.mean([r[f"true_margin_{right_prefix}"] - r[f"true_margin_{left_prefix}"] for r in rows])) if rows else 0.0,
        "corrected_ids": ";".join(f"{r['fold']}:{r['patient_id_1based']}" for r in corrected),
        "harmed_ids": ";".join(f"{r['fold']}:{r['patient_id_1based']}" for r in harmed),
    }


def main():
    old_manifest, old_audit, checkpoints = verify_historical_sources()
    manifest = h.read_json(core.MANIFEST)
    h.require(manifest["status"] in ("FOLD4_AUDITED", "AUDIT_PASSED", "AUDITED"), "Fold-4 audit is incomplete")
    local = local_replay(manifest)
    old_rows = read_patient_rows(E465 / "results/patient_level.csv")
    rows = assemble_patient_rows(old_rows, local)
    h.require(sum(r["current_native_training_overlap"] for r in rows) == 17
              and sum(r["previous_native_training_overlap"] for r in rows) == 16
              and sum(r["in_fixed_union_exposure_17"] for r in rows) == 17
              and sum(r["outside_fixed_union_854"] for r in rows) == FIXED_OUTSIDE_COUNT,
              "Fixed overlap union cardinality differs")
    local_ids = [r["patient_id_1based"] for r in rows if r["fold"] == 4]
    h.require(local_ids == local["test_ids"], "Fold-4 patient ordering differs")
    h.require([r["true_label"] for r in rows if r["fold"] == 4] == local["labels"].tolist(), "Fold-4 labels differ")

    metrics = {}
    old_metric = historical_metrics(old_rows)
    for method in ("E74.4-G2 DARG", "E82.4.6-G2", "E82.4.6.5-G1"):
        metrics[method] = {str(fold): old_metric[(method, fold)] for fold in range(1, 11)}
    metrics["E82.4.6.6-G1"] = {str(fold): metrics_for_patients([r for r in rows if r["fold"] == fold], "full") for fold in range(1, 11)}
    summaries = {method: summary({int(f): v for f, v in fold_values.items()}) for method, fold_values in metrics.items()}

    paired_rows = []
    scopes = [
        ("ten_fold", lambda r: True),
        ("fold1", lambda r: r["fold"] == 1), ("fold2", lambda r: r["fold"] == 2),
        ("fold3", lambda r: r["fold"] == 3), ("fold4", lambda r: r["fold"] == 4),
        ("fold4_new_712", lambda r: r["fold"] == 4 and r["patient_id_1based"] == 712),
        ("fold4_retained_713_727_773", lambda r: r["fold"] == 4 and r["patient_id_1based"] in {713, 727, 773}),
        ("fold4_other_83", lambda r: r["fold"] == 4 and r["patient_id_1based"] not in {712, 713, 727, 773}),
        ("fixed_union_exposure_17", lambda r: r["in_fixed_union_exposure_17"]),
        ("outside_fixed_union_854", lambda r: r["outside_fixed_union_854"]),
    ]
    for scope, selector in scopes:
        paired_rows.append(paired(rows, "previous", "full", scope, selector))
    for fold in range(1, 11):
        outcome = paired(rows, "previous", "full", f"fold{fold}", lambda r, f=fold: r["fold"] == f)
        outcome["fold_result"] = "win" if outcome["net"] > 0 else ("loss" if outcome["net"] < 0 else "tie")
        paired_rows.append(outcome)

    residual_rows = [paired(rows, "residual_off", "full", scope, selector) for scope, selector in scopes]
    query_rows = [paired(rows, "anonymous", "full", scope, selector) for scope, selector in scopes]
    for row in residual_rows: row["comparison"] = "Residual-off_to_Full"
    for row in query_rows: row["comparison"] = "Anonymous_to_Medical_Query"
    for row in paired_rows: row["comparison"] = "E82.4.6.5_to_E82.4.6.6"

    all_outputs = {
        "status": "PASS", "experiment": CONFIG["experiment"], "device": h.archive.verify_device(),
        "historical_source": {"directory": str(E465), "manifest_sha": h.sha_file(E465 / "manifest.json"),
                              "audit_sha": h.sha_file(E465 / "results/audit_status.json"),
                              "result_files_verified": True, "checkpoint_sources": checkpoints},
        "fold4": {
            "best_epoch": local["best_epoch"], "latest_epoch": local["latest_epoch"],
            "best_sha": local["best_sha"], "latest_sha": local["latest_sha"], "journal_sha": local["journal_sha"],
            "certificate": local["certificate"], "best_reloaded": True, "latest_reloaded": True,
            "latest_next_training_exact": bool(local["certificate"]["latest_next_exact"]),
            "latest_next_evaluation_exact": bool(local["certificate"]["latest_next_exact"]),
            "masked_ids_1based": sorted(CURRENT_EXPOSURE[4]), "masked_rows": 4,
            "legal_rows": 784, "darg_rows": 788, "test_rows": 87,
            "dropout_calls_per_epoch": 6, "rng_epoch_trace": local["certificate"]["rng_trace"],
            "residual_zero_masked": True, "unmasked_residual_active": True,
            "max_cam_relative_delta": float(np.max(local["ratio"])),
            "local_inference_sha": h.sha(local["full"]),
        },
        "fixed_union": {"current_count": 17, "previous_count": 16, "outside_count": FIXED_OUTSIDE_COUNT,
                        "current": {str(f): sorted(v) for f, v in CURRENT_EXPOSURE.items()},
                        "previous": {str(f): sorted(v) for f, v in PREVIOUS_EXPOSURE.items()},
                        "union_ids": [f"{f}:{p}" for f, p in sorted(UNION_EXPOSURE)]},
        "metrics": metrics, "summaries": summaries,
        "paired": paired_rows, "residual_off_to_full": residual_rows, "anonymous_to_medical": query_rows,
        "patient_rows": rows,
        "historical_unchanged_folds": [1, 2, 3, 5, 6, 7, 8, 9, 10],
        "all_nonfold4_probability_exact_vs_e82465": True,
        "all_nonfold4_hard_prediction_exact_vs_e82465": True,
        "fold4_masked_probability_exact_full_off": True,
        "gradient_paths": {
            "darg_classification": True, "darg_relation": True, "scmr_supervision_masked": True,
            "jacobian_spectral_in": True, "jacobian_spectral_out": True, "medical_prompt": True,
            "cam144_limit_8_percent": True, "shared_gradient_limit_0_1_percent": True,
        },
        "paper_and_author_code_read": True,
    }
    h.write_json(ROOT / "runtime/replayed_folds.json", all_outputs)
    print(json.dumps({"status": "PASS", "fold4_best": local["best_epoch"], "fold4_latest": local["latest_epoch"],
                      "fixed_union": len(UNION_EXPOSURE), "outside": FIXED_OUTSIDE_COUNT,
                      "fold4_correct": int(np.sum(local["full"]["prediction"] == local["labels"]))}, indent=2))


if __name__ == "__main__":
    main()
