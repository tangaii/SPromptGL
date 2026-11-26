from __future__ import annotations

import ast
import csv
import hashlib
import inspect
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from common import (
    CANDIDATES,
    FIXED,
    FOLDS,
    HISTORY,
    MODELS,
    PREVIEW_FOLDS,
    RESULTS,
    coordinate_sha,
    fit_embedding,
    geometry_metrics,
    input_path,
    load_input,
    preprocess,
    sha256_file,
    trust_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
HISTORY_INPUTS = HISTORY / "inputs"
REQUIRED_HISTORY_FILES = [
    "code/scan_stage1.py",
    "code/stage2.py",
    "code/stage3_plot.py",
    "code/stage4_formal.py",
    "code/stage4_plot_metrics.py",
    "results/stage2_comparison.json",
    "results/stage3_summary.txt",
    "results/final_metrics.csv",
    "results/selected_parameters.json",
]


def check_input_identity() -> dict:
    manifest = json.loads((HISTORY_INPUTS / "manifest.json").read_text())
    copied_manifest = json.loads((ROOT / "inputs" / "source_manifest.json").read_text())
    if copied_manifest["source"] != str(HISTORY_INPUTS) or len(copied_manifest["files"]) != 40:
        raise AssertionError("copied input provenance manifest mismatch")
    copied_by_name = {item["file"]: item["sha256"] for item in copied_manifest["files"]}
    source_matches = 0
    copy_matches = 0
    array_matches = 0
    finite_shapes = 0
    for item in manifest["files"]:
        name = item["file"]
        source = HISTORY_INPUTS / name
        copied = ROOT / "inputs" / name
        expected = item["sha256"]
        if copied_by_name.get(name) != expected:
            raise AssertionError(f"copied provenance SHA mismatch {name}")
        if sha256_file(source) != expected:
            raise AssertionError(f"historical input SHA mismatch {name}")
        source_matches += 1
        if sha256_file(copied) != expected:
            raise AssertionError(f"copied input SHA mismatch {name}")
        copy_matches += 1
        with np.load(source, allow_pickle=False) as left, np.load(copied, allow_pickle=False) as right:
            if set(left.files) != set(right.files):
                raise AssertionError(f"archive key mismatch {name}")
            for key in left.files:
                if not np.array_equal(left[key], right[key]):
                    raise AssertionError(f"archive array mismatch {name} {key}")
            features = right["features"]
            labels = right["labels"]
            patient_ids = right["patient_ids"]
            if features.shape != (871, 144) or labels.shape != (871,) or patient_ids.shape != (871,):
                raise AssertionError(f"archive shape mismatch {name}")
            if not np.isfinite(features).all() or not np.isfinite(labels).all() or not np.isfinite(patient_ids).all():
                raise AssertionError(f"archive finite-value mismatch {name}")
            if set(np.unique(labels).tolist()) != {0, 1}:
                raise AssertionError(f"archive labels mismatch {name}")
            finite_shapes += 1
        array_matches += 1
    return {
        "manifest_file_count": len(manifest["files"]),
        "historical_sha_matches": source_matches,
        "copied_sha_matches": copy_matches,
        "archive_array_matches": array_matches,
        "finite_shape_label_checks": finite_shapes,
        "copied_provenance_matches": len(copied_by_name),
        "manifest_sha256": hashlib.sha256((HISTORY_INPUTS / "manifest.json").read_bytes()).hexdigest(),
    }


def replay_task(task: tuple[str, str, int]) -> tuple[str, str, int, str]:
    candidate_id, model, fold = task
    embedding, _, _ = fit_embedding(model, fold, CANDIDATES[candidate_id])
    return candidate_id, model, fold, coordinate_sha(embedding)


def check_historical_reuse() -> dict:
    tasks = [(candidate, model, fold) for candidate, history_id in (("C0", "B5"), ("C1", "B6")) for fold in PREVIEW_FOLDS for model in MODELS]
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
        replayed = list(executor.map(replay_task, tasks))
    matches = 0
    for candidate, model, fold, replay_sha in replayed:
        history_id = "B5" if candidate == "C0" else "B6"
        path = HISTORY / "results" / "stage2_coordinates" / f"{history_id}_{model}_fold{fold:02d}.npy"
        if replay_sha != coordinate_sha(np.load(path, allow_pickle=False)):
            raise AssertionError(f"historical coordinate replay mismatch {candidate} {model} fold {fold}")
        matches += 1
    return {"replayed_runs": len(replayed), "coordinate_sha_matches": matches}


def check_formal_coordinates() -> dict:
    manifest = json.loads((RESULTS / "formal_run_manifest.json").read_text())
    selected = json.loads((RESULTS / "selected_parameters.json").read_text())
    if manifest["run_count"] != 40 or manifest["fresh_fit_count"] != 40 or manifest["candidate"] != selected["id"]:
        raise AssertionError("formal run manifest count or candidate mismatch")
    for row in manifest["runs"]:
        if not row["fresh_fit_from_frozen_features"] or row["candidate"] != selected["id"] or row["nonfinite_coordinates"] != 0:
            raise AssertionError("formal run is not a fresh finite C5 fit")
        for parameter in ("perplexity", "early_exaggeration", "learning_rate"):
            if row[parameter] != selected[parameter]:
                raise AssertionError(f"formal run parameter mismatch {row['model']} fold {row['fold']} {parameter}")
        expected = sha256_file(input_path(row["model"], int(row["fold"])))
        if row["input_file_sha256"] != expected:
            raise AssertionError("formal input SHA mismatch")
        path = RESULTS / "formal_coordinates" / f"{row['model']}_fold{int(row['fold']):02d}.npy"
        embedding = np.load(path, allow_pickle=False)
        if embedding.shape != (871, 2) or not np.isfinite(embedding).all():
            raise AssertionError(f"invalid formal coordinate {path}")
        if row["coordinate_sha256"] != coordinate_sha(embedding):
            raise AssertionError(f"formal coordinate SHA mismatch {path}")
    for parameter in ("perplexity", "early_exaggeration", "learning_rate", "metric", "init", "random_state", "iterations", "preprocessing", "sklearn_version", "sklearn_iteration_argument"):
        if manifest["parameters"][parameter] != selected[parameter]:
            raise AssertionError(f"formal manifest parameter mismatch {parameter}")
    if manifest["parameters"]["id"] != selected["id"] or manifest["parameters"]["selection_uses_true_labels"] is not True:
        raise AssertionError("formal parameters do not match the frozen selection record")
    return {"manifest_run_count": manifest["run_count"], "finite_coordinate_runs": 40, "uniform_candidate": "C5", "labels_passed_to_tsne": manifest["labels_passed_to_tsne"]}


def check_formal_replay() -> dict:
    tasks = [("C5", model, fold) for model in MODELS for fold in FOLDS]
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
        replayed = list(executor.map(replay_task, tasks))
    matches = 0
    for _, model, fold, replay_sha in replayed:
        saved = np.load(RESULTS / "formal_coordinates" / f"{model}_fold{fold:02d}.npy", allow_pickle=False)
        if replay_sha != coordinate_sha(saved):
            raise AssertionError(f"formal replay mismatch {model} fold {fold}")
        matches += 1
    return {"replayed_runs": len(replayed), "coordinate_sha_matches": matches}


def check_csv_and_metrics() -> dict:
    rows = list(csv.DictReader((RESULTS / "tsne_coordinates.csv").open()))
    if len(rows) != 40 * 871:
        raise AssertionError(f"coordinate CSV row count {len(rows)}")
    index = 0
    for model in MODELS:
        for fold in FOLDS:
            data = load_input(model, fold)
            embedding = np.load(RESULTS / "formal_coordinates" / f"{model}_fold{fold:02d}.npy", allow_pickle=False)
            block = rows[index : index + 871]
            index += 871
            for position, row in enumerate(block):
                if row["model"] != model or int(row["fold"]) != fold:
                    raise AssertionError("coordinate CSV ordering mismatch")
                if int(row["patient_id"]) != int(data["patient_ids"][position]) or int(row["true_label"]) != int(data["labels"][position]):
                    raise AssertionError("coordinate CSV patient or label mismatch")
                if float(row["tsne_x"]) != embedding[position, 0] or float(row["tsne_y"]) != embedding[position, 1]:
                    raise AssertionError("coordinate CSV value mismatch")
            if not np.isfinite(embedding).all():
                raise AssertionError("nonfinite embedding in CSV source")
    metric_rows = {(row["model"], int(row["fold"])): row for row in csv.DictReader((RESULTS / "final_metrics.csv").open())}
    if len(metric_rows) != 40:
        raise AssertionError("final metrics row count mismatch")
    numeric_metrics = [
        "silhouette_2d", "within_mean_distance_2d", "global_rms_scale_2d", "normalized_within", "between_distance_2d", "separation_ratio",
        "cross_class_neighbor_rate_15", "cross_class_neighbor_rate_30", "trustworthiness_15", "trustworthiness_30",
    ]
    for model in MODELS:
        for fold in FOLDS:
            data = load_input(model, fold)
            embedding = np.load(RESULTS / "formal_coordinates" / f"{model}_fold{fold:02d}.npy", allow_pickle=False)
            _, reduced, _ = preprocess(data["features"])
            expected = {**geometry_metrics(embedding, data["labels"].astype(int)), **trust_metrics(reduced, embedding)}
            row = metric_rows[(model, fold)]
            for key in numeric_metrics:
                if not np.isclose(float(row[key]), expected[key], rtol=0, atol=1e-12):
                    raise AssertionError(f"metric replay mismatch {model} fold {fold} {key}")
    return {"coordinate_csv_rows": len(rows), "metric_rows": len(metric_rows), "metric_replay_rows": 40, "ddof": 0}


def check_plots() -> dict:
    individual = sorted((RESULTS / "formal_figures" / "individual").glob("*/*.png"))
    comparison = sorted((RESULTS / "formal_figures" / "comparison").glob("*.png"))
    previews = sorted((RESULTS / "preview_figures").glob("*.png"))
    if len(individual) != 40 or len(comparison) != 10 or len(previews) != 26:
        raise AssertionError("PNG count mismatch")
    formal_manifest = json.loads((RESULTS / "formal_plot_manifest.json").read_text())
    preview_manifest = json.loads((RESULTS / "preview_manifest.json").read_text())
    scan_rows = list(csv.DictReader((RESULTS / "parameter_comparison.csv").open()))
    scan_sha = {(row["candidate"], row["model"], int(row["fold"])): row["coordinate_sha256"] for row in scan_rows}
    formal_rows = formal_manifest["figures"]
    formal_individual = {(row["model"], int(row["fold"])): row for row in formal_rows if row["kind"] == "individual"}
    formal_comparison = {int(row["fold"]): row for row in formal_rows if row["kind"] == "comparison"}
    if len(formal_individual) != 40 or len(formal_comparison) != 10:
        raise AssertionError("formal plot manifest mapping count mismatch")
    if formal_manifest["models"] != MODELS or formal_manifest["colors"] != {"0": "#D98282", "1": "#4C9A91"}:
        raise AssertionError("formal plot model order or label color mismatch")
    if formal_manifest["shared_axis_limits_per_fold"] is not True:
        raise AssertionError("formal plots do not declare shared fold limits")
    for fold in FOLDS:
        comparison_row = formal_comparison[fold]
        expected_bounds = (comparison_row["xlim"], comparison_row["ylim"])
        comparison_path = RESULTS / "formal_figures" / "comparison" / f"fold{fold:02d}_comparison.png"
        if Path(comparison_row["path"]) != comparison_path:
            raise AssertionError(f"formal comparison plot path mismatch fold {fold}")
        if comparison_row["models"] != MODELS:
            raise AssertionError(f"formal comparison order mismatch fold {fold}")
        for model in MODELS:
            plot_row = formal_individual[(model, fold)]
            path = RESULTS / "formal_figures" / "individual" / model / f"fold{fold:02d}.png"
            if Path(plot_row["path"]) != path or (plot_row["xlim"], plot_row["ylim"]) != expected_bounds:
                raise AssertionError(f"individual plot path or shared axes mismatch {model} fold {fold}")
            coordinate = np.load(RESULTS / "formal_coordinates" / f"{model}_fold{fold:02d}.npy", allow_pickle=False)
            if plot_row["coordinate_sha256"] != coordinate_sha(coordinate):
                raise AssertionError(f"individual plot coordinate mapping mismatch {model} fold {fold}")
            if comparison_row["coordinate_sha256_by_model"].get(model) != coordinate_sha(coordinate):
                raise AssertionError(f"comparison plot coordinate mapping mismatch {model} fold {fold}")
    preview_rows = preview_manifest["figures"]
    preview_by_key = {(row["candidate"], int(row["fold"])): row for row in preview_rows}
    if len(preview_by_key) != 26 or preview_manifest["models"] != MODELS or preview_manifest["folds"] != PREVIEW_FOLDS:
        raise AssertionError("preview manifest mapping count or model order mismatch")
    for fold in PREVIEW_FOLDS:
        fold_limits = None
        for candidate in CANDIDATES:
            row = preview_by_key[(candidate, fold)]
            if Path(row["path"]) != RESULTS / "preview_figures" / f"{candidate}_fold{fold:02d}.png":
                raise AssertionError(f"preview path mismatch {candidate} fold {fold}")
            bounds = (row["xlim"], row["ylim"])
            if fold_limits is None:
                fold_limits = bounds
            elif bounds != fold_limits:
                raise AssertionError(f"preview axes differ across candidates on fold {fold}")
            if row["models"] != MODELS or row["colors"] != {"0": "#D98282", "1": "#4C9A91"}:
                raise AssertionError(f"preview model order or colors mismatch {candidate} fold {fold}")
            for model in MODELS:
                if row["coordinate_sha256_by_model"].get(model) != scan_sha[(candidate, model, fold)]:
                    raise AssertionError(f"preview coordinate mapping mismatch {candidate} {model} fold {fold}")
    for path in individual + comparison:
        with Image.open(path) as image:
            if image.format != "PNG" or image.size[0] <= 0 or image.size[1] <= 0:
                raise AssertionError(f"unreadable formal PNG {path}")
            dpi = image.info.get("dpi", (0, 0))
            if abs(float(dpi[0]) - 600) > 1 or abs(float(dpi[1]) - 600) > 1:
                raise AssertionError(f"formal PNG DPI mismatch {path}")
            rgb = image.convert("RGB")
            colors = np.asarray(rgb)
            # Matplotlib composites the 0.70-alpha class colors against white.
            if not np.any(np.all(colors == np.array([228, 168, 168]), axis=2)) or not np.any(np.all(colors == np.array([130, 184, 178]), axis=2)):
                raise AssertionError(f"label colors absent {path}")
    for path in previews:
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (2700, 2100):
                raise AssertionError(f"preview PNG mismatch {path}")
    forbidden = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in {".pdf", ".svg", ".eps"}]
    if forbidden:
        raise AssertionError(f"forbidden image files {forbidden}")
    return {
        "formal_individual_png": len(individual),
        "formal_comparison_png": len(comparison),
        "preview_png": len(previews),
        "preview_candidate_coordinate_mappings": len(preview_rows) * len(MODELS),
        "formal_coordinate_mappings": len(formal_rows) + len(formal_comparison) * (len(MODELS) - 1),
        "shared_equal_axis_limits_verified": True,
        "non_png_forbidden": 0,
        "formal_dpi": 600,
    }


def check_source_contract() -> dict:
    source = (ROOT / "code" / "common.py").read_text()
    tree = ast.parse(source)
    fit_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "fit_embedding")
    labels_names = [node for node in ast.walk(fit_node) if isinstance(node, ast.Name) and node.id == "labels"]
    fit_calls = [node for node in ast.walk(fit_node) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "fit_transform"]
    if labels_names or not fit_calls or any(len(call.args) != 1 for call in fit_calls):
        raise AssertionError("labels appear in t-SNE fit contract")
    selected = json.loads((RESULTS / "selected_parameters.json").read_text())
    if selected["selection_uses_true_labels"] is not True or selected["selection_is_exploratory"] is not True:
        raise AssertionError("exploratory label-use disclosure missing")
    return {"labels_in_fit_embedding_ast": False, "fit_transform_argument_count": [len(call.args) for call in fit_calls], "selection_uses_labels_only_for_selection": True}


def check_selection() -> dict:
    rows = list(csv.DictReader((RESULTS / "parameter_comparison.csv").open()))
    selected = json.loads((RESULTS / "selected_parameters.json").read_text())
    metric_names = [
        "silhouette_2d",
        "normalized_within",
        "separation_ratio",
        "cross_class_neighbor_rate_15",
        "cross_class_neighbor_rate_30",
        "trustworthiness_15",
        "trustworthiness_30",
    ]
    grouped = {candidate: [row for row in rows if row["candidate"] == candidate] for candidate in CANDIDATES}
    if any(len(candidate_rows) != 8 for candidate_rows in grouped.values()):
        raise AssertionError("selection does not contain eight Fold 7/8 model combinations per candidate")
    means = {
        candidate: {metric: float(np.mean([float(row[metric]) for row in candidate_rows])) for metric in metric_names}
        for candidate, candidate_rows in grouped.items()
    }
    trust_floor = means["C0"]["trustworthiness_15"] - 0.015
    feasible = [candidate for candidate in CANDIDATES if means[candidate]["trustworthiness_15"] >= trust_floor]
    chosen = max(
        feasible,
        key=lambda candidate: (
            means[candidate]["silhouette_2d"],
            -means[candidate]["normalized_within"],
            means[candidate]["separation_ratio"],
            -means[candidate]["cross_class_neighbor_rate_15"],
            -means[candidate]["cross_class_neighbor_rate_30"],
            means[candidate]["trustworthiness_15"],
        ),
    )
    if chosen != selected["id"] or selected["selection_frozen_before_formal_ten_fold"] is not True:
        raise AssertionError("saved parameter selection does not match the frozen rule")
    if selected["selection_uses_true_labels"] is not True or selected["selection_is_exploratory"] is not True:
        raise AssertionError("selection's label-aware exploratory status is missing")
    for candidate, candidate_means in means.items():
        for metric in metric_names:
            if not np.isclose(candidate_means[metric], selected["candidate_aggregate_means"][candidate][metric], rtol=0, atol=1e-12):
                raise AssertionError(f"candidate aggregate mismatch {candidate} {metric}")
    return {"candidate_count": len(grouped), "scan_rows": len(rows), "recomputed_selected_candidate": chosen, "trustworthiness_15_floor": trust_floor, "feasible_candidates": feasible}


def check_report_summary() -> dict:
    report = (RESULTS / "report.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((RESULTS / "final_metrics.csv").open()))
    metric_names = [
        "silhouette_2d",
        "silhouette_raw144",
        "silhouette_standardized144",
        "silhouette_pca50",
        "trustworthiness_15",
        "trustworthiness_30",
        "normalized_within",
        "separation_ratio",
        "cross_class_neighbor_rate_15",
        "cross_class_neighbor_rate_30",
    ]
    if "population SD (`ddof=0`)" not in report:
        raise AssertionError("report does not identify population SD")
    checked = 0
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in model_rows], dtype=np.float64)
            expected = f"{np.mean(values):.6f} ± {np.std(values, ddof=0):.6f}"
            if expected not in report:
                raise AssertionError(f"report population summary mismatch {model} {metric}")
            checked += 1
    return {"ten_fold_population_summary_values_checked": checked, "population_sd_ddof": 0}


def check_handoff_append() -> dict:
    path = Path("/model/spromptgl/experiment/handoff.md")
    data = path.read_bytes()
    prior_manifest = json.loads((ROOT / "manifest.json").read_text())
    expected = prior_manifest.get("handoff_append_audit")
    if not expected:
        raise AssertionError("handoff append baseline metadata missing")
    prefix_size = int(expected["original_prefix_bytes"])
    prefix = data[:prefix_size]
    section = data[prefix_size:]
    prefix_sha = hashlib.sha256(prefix).hexdigest()
    section_sha = hashlib.sha256(section).hexdigest()
    heading = b"DReSP ABIDE t-SNE compactness and separation"
    if prefix_sha != expected["original_prefix_sha256"] or prefix_sha != expected["readback_prefix_sha256"]:
        raise AssertionError("handoff original prefix SHA mismatch")
    if len(section) != int(expected["appended_bytes"]) or section_sha != expected["appended_section_sha256"]:
        raise AssertionError("handoff appended section length or SHA mismatch")
    if section.count(heading) != 1 or not section.rstrip().endswith("。".encode("utf-8")):
        raise AssertionError("handoff appended section is incomplete")
    return {
        "path": str(path),
        "original_prefix_bytes": prefix_size,
        "original_prefix_sha256": prefix_sha,
        "readback_prefix_sha256": prefix_sha,
        "appended_bytes": len(section),
        "appended_section_sha256": section_sha,
        "full_file_sha256": hashlib.sha256(data).hexdigest(),
        "verified": prefix_sha == expected["original_prefix_sha256"] and section_sha == expected["appended_section_sha256"],
    }


def check_history_files() -> dict:
    missing = [name for name in REQUIRED_HISTORY_FILES if not (HISTORY / name).exists()]
    if missing:
        raise AssertionError(f"historical files missing {missing}")
    hashes = {name: sha256_file(HISTORY / name) for name in REQUIRED_HISTORY_FILES}
    figures = sorted((HISTORY / "results" / "stage3_figures").glob("*.png"))
    if len(figures) != 6:
        raise AssertionError("historical stage3 figure count mismatch")
    return {"required_files": len(REQUIRED_HISTORY_FILES), "stage3_png": len(figures), "required_file_sha256": hashes}


def write_manifest(status: str, checks: dict) -> None:
    files = [
        "config.json", "manifest.json", "results/report.md", "results/parameter_comparison.csv", "results/selected_parameters.json",
        "results/final_metrics.csv", "results/tsne_coordinates.csv", "results/audit_status.json", "results/formal_run_manifest.json",
        "results/formal_plot_manifest.json", "results/scan_manifest.json", "results/preview_manifest.json",
    ]
    manifest = {
        "status": status,
        "experiment": "experiment_dresp_tsne_compact_separation",
        "authoritative_directory": str(ROOT),
        "history_directory": str(HISTORY),
        "input_feature_count": 40,
        "scan_runs": 104,
        "scan_reused_historical_runs": 16,
        "scan_fresh_runs": 88,
        "preview_png": 26,
        "formal_runs": 40,
        "coordinate_rows": 34840,
        "formal_individual_png": 40,
        "formal_comparison_png": 10,
        "files": {name: sha256_file(ROOT / name) for name in files if (ROOT / name).exists()},
        "checks_summary": {name: value for name, value in checks.items()},
    }
    if "handoff_append" in checks:
        manifest["handoff_append_audit"] = checks["handoff_append"]
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")


def main() -> None:
    checks = {}
    errors = {}
    functions = [
        ("input_identity", check_input_identity),
        ("historical_reuse", check_historical_reuse),
        ("formal_manifest", check_formal_coordinates),
        ("formal_replay", check_formal_replay),
        ("csv_and_metric_replay", check_csv_and_metrics),
        ("plots", check_plots),
        ("source_contract", check_source_contract),
        ("selection", check_selection),
        ("report_summary", check_report_summary),
        ("handoff_append", check_handoff_append),
        ("history_files", check_history_files),
    ]
    for name, function in functions:
        try:
            checks[name] = function()
        except Exception as error:
            errors[name] = repr(error)
    required_checks = {name for name, _ in functions}
    critical_checks_completed = required_checks.issubset(checks) and not errors
    status = "COMPLETE_AUDITED_TSNE_COMPACT_SEPARATION" if critical_checks_completed else "AUDIT_FAILED"
    audit = {
        "status": status,
        "checks": checks,
        "errors": errors,
        "critical_checks_are_computed": critical_checks_completed,
        "population_sd_ddof": checks.get("report_summary", {}).get("population_sd_ddof"),
        "audit_code": "code/audit.py",
    }
    (RESULTS / "audit_status.json").write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n")
    report_path = RESULTS / "report.md"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8")
        report = report.replace(
            "Prepared for final independent audit. The authoritative audit result is recorded in `results/audit_status.json` and `manifest.json`.",
            f"{status}. The computed audit result is recorded in `results/audit_status.json` and `manifest.json`.",
        )
        report_path.write_text(report, encoding="utf-8")
    write_manifest(status, checks)
    print(json.dumps({"status": status, "passed": sorted(checks), "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
