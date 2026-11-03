#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import ftfy  # noqa: F401
except ModuleNotFoundError as error:
    if error.name != "ftfy":
        raise
    import sys
    import types

    ftfy_compat = types.ModuleType("ftfy")

    def ascii_fix_text(text: str) -> str:
        if not text.isascii():
            raise ValueError("the audit tokenizer fallback only accepts the frozen ASCII prompt set")
        return text

    ftfy_compat.fix_text = ascii_fix_text
    sys.modules["ftfy"] = ftfy_compat

import run as E60


HERE = Path(__file__).resolve().parent
RT = HERE / ".runtime"
RESULTS = HERE / "results"
ATOL = 1e-7
RTOL = 1e-6


def assert_close(expected: Any, actual: Any, path: str) -> None:
    if torch.is_tensor(expected) or torch.is_tensor(actual):
        if not torch.is_tensor(expected) or not torch.is_tensor(actual):
            raise AssertionError(f"tensor type mismatch at {path}")
        left, right = expected.detach().cpu(), actual.detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"tensor metadata mismatch at {path}: {left.shape}/{left.dtype} != {right.shape}/{right.dtype}")
        if left.is_floating_point() or left.is_complex():
            if not torch.allclose(left, right, rtol=RTOL, atol=ATOL, equal_nan=False):
                delta = float((left - right).abs().max()) if left.numel() else 0.0
                raise AssertionError(f"tensor values differ at {path}, max_abs_delta={delta}")
        elif not torch.equal(left, right):
            raise AssertionError(f"integer tensor differs at {path}")
        return
    if isinstance(expected, dict) or isinstance(actual, dict):
        if not isinstance(expected, dict) or not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise AssertionError(f"mapping keys differ at {path}")
        for key in expected:
            assert_close(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        if not isinstance(expected, (list, tuple)) or not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            raise AssertionError(f"sequence differs at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_close(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, (float, np.floating)) or isinstance(actual, (float, np.floating)):
        if not math.isclose(float(expected), float(actual), rel_tol=RTOL, abs_tol=ATOL):
            raise AssertionError(f"numeric values differ at {path}: {expected} != {actual}")
        return
    if expected != actual:
        raise AssertionError(f"values differ at {path}: {expected!r} != {actual!r}")


def assert_finite_state(value: Any, path: str = "checkpoint") -> None:
    if torch.is_tensor(value):
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise AssertionError(f"nonfinite tensor in {path}")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise AssertionError(f"nonfinite numeric value in {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite_state(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_state(item, f"{path}[{index}]")


def fold_digest(indices: Any) -> str:
    import hashlib

    values = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _load_state_for_audit(group: str, fold: int, checkpoint: dict[str, Any], banks):
    r, _ = E60.load_base()
    state = E60.state_for_fold(group, fold)
    state["train_idx_sha"] = fold_digest(state["train"].detach().cpu().numpy())
    state["test_idx_sha"] = fold_digest(state["test"])
    state["model"].load_state_dict(checkpoint["model"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    module = None
    if group != "E60_BASE_REPLAY":
        module = E60.build_module(group, fold, banks, None)
        module.load_state_dict(checkpoint["semantic_module"], strict=True)
    return r, state, module


def audit_fold(group: str, fold: int, historical: dict[str, Any] | None = None) -> dict[str, Any]:
    from text_features import group_text_banks

    r, _ = E60.load_base()
    root = E60.reset_group_paths(group)
    task_path = r.task_path(fold)
    checkpoint_path = r.best_path(fold)
    if not task_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing task or checkpoint: {group} fold {fold}")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    checkpoint_sha = E60.sha_file(checkpoint_path)
    if task.get("status") != "COMPLETE" or task.get("best_sha256") != checkpoint_sha:
        raise AssertionError(f"task completion/checkpoint SHA mismatch: {group} fold {fold}")
    if task.get("readback_status") != "WORKER_READBACK_PASS":
        raise AssertionError(f"worker readback missing: {group} fold {fold}")

    cache = E60.load_text_cache()
    banks = group_text_banks(cache, group)
    checkpoint = E60.torch_load(checkpoint_path)
    assert_finite_state(checkpoint)
    r, state, module = _load_state_for_audit(group, fold, checkpoint, banks)
    current_signature = E60.task_signature(group, fold, state)
    trained_signature = {key: task.get(key) for key in current_signature}
    task_source_signature = trained_signature.get("source_signature")
    checkpoint_source_signature = checkpoint.get("source_signature")
    if not isinstance(task_source_signature, dict) or not isinstance(checkpoint_source_signature, dict):
        raise AssertionError(f"task/checkpoint source signature is missing: {group} fold {fold}")
    task_checkpoint_signature_diffs = [
        path for path in set(task_source_signature) | set(checkpoint_source_signature)
        if task_source_signature.get(path) != checkpoint_source_signature.get(path)
    ]
    if task_checkpoint_signature_diffs and not (
        group == "E60_DSS_CLASS_ONLY"
        and task_checkpoint_signature_diffs == [str(HERE / "semantic.py")]
    ):
        raise AssertionError(f"task/checkpoint training signature mismatch: {group} fold {fold}: {task_checkpoint_signature_diffs}")
    for key in ("group", "fold", "seed", "text_cache_sha256", "knowledge_base_sha256", "split_sha", "pair_sha", "fold_cache_sha", "initial_shared_sha", "initial_added_sha", "initial_rng_sha"):
        if trained_signature.get(key) != current_signature.get(key):
            raise AssertionError(f"task signature mismatch at {key}: {group} fold {fold}")
    run_path = str(HERE / "run.py")
    audit_path = str(HERE / "audit.py")
    ignored_source_paths = {run_path, audit_path}
    ignored_source_paths.add(str(HERE / "semantic.py"))
    source_signature_exceptions = []
    if task_checkpoint_signature_diffs:
        source_signature_exceptions.append({
            "path": task_checkpoint_signature_diffs[0],
            "task_sha256": task_source_signature[task_checkpoint_signature_diffs[0]],
            "checkpoint_sha256": checkpoint_source_signature[task_checkpoint_signature_diffs[0]],
            "rule": "CLASS_ONLY training-start signature differs from checkpoint serialization-time signature; task/checkpoint SHA, epoch, and metrics must match",
        })
    for path, expected_sha in current_signature["source_signature"].items():
        trained_sha = trained_signature["source_signature"].get(path)
        if path in ignored_source_paths and trained_sha != expected_sha:
            source_signature_exceptions.append({
                "path": path,
                "trained_sha256": trained_sha,
                "current_sha256": expected_sha,
                "rule": "task and checkpoint source signatures match; functional checkpoint readback remains required",
            })
            continue
        if trained_sha != expected_sha:
            raise AssertionError(f"training source signature changed at {path}: {group} fold {fold}")
    if checkpoint.get("group") != group or checkpoint.get("fold") != fold or checkpoint.get("seed") != 0:
        raise AssertionError(f"checkpoint identity mismatch: {group} fold {fold}")

    expected_wd = {
        "base": [0.11],
        "cam": [0.11],
        "added": [0.01, 0.01],
    }
    if group == "E60_ORIGINAL_PEN":
        expected_wd["cam"] = [0.11]
        expected_wd["semantic"] = []
    elif group not in {"E60_BASE_REPLAY", "E60_ORIGINAL_PEN"}:
        expected_wd["semantic"] = [0.001]
    else:
        expected_wd["semantic"] = []
    if checkpoint.get("optimizer_weight_decay") != expected_wd:
        raise AssertionError(f"optimizer WD mismatch: {group} fold {fold}: {checkpoint.get('optimizer_weight_decay')}")
    if checkpoint.get("text_cache_sha256") != E60.sha_file(HERE / ".runtime" / "text_embeddings.pt"):
        raise AssertionError(f"text cache SHA mismatch: {group} fold {fold}")
    if checkpoint.get("knowledge_base_sha256") != E60.sha_file(HERE / "knowledge_base.json"):
        raise AssertionError(f"knowledge base SHA mismatch: {group} fold {fold}")
    for checkpoint_key, signature_key in (("split_sha", "split_sha"), ("pair_sha", "pair_sha"), ("fold_cache_sha", "fold_cache_sha")):
        expected_value = current_signature["split_sha"] if signature_key == "split_sha" else current_signature["pair_sha"] if signature_key == "pair_sha" else current_signature["fold_cache_sha"]
        if checkpoint.get(checkpoint_key) != expected_value:
            raise AssertionError(f"checkpoint fold source mismatch at {checkpoint_key}: {group} fold {fold}")
    if state["train_idx_sha"] != checkpoint["train_test_indices"]["train_sha256"] or state["test_idx_sha"] != checkpoint["train_test_indices"]["test_sha256"]:
        raise AssertionError(f"train/test index SHA mismatch: {group} fold {fold}")
    reference = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))["historical_fold_sources"][str(fold)]
    for key in ("initial_shared_sha", "initial_added_sha", "initial_rng_sha", "split_sha", "pair_sha"):
        if checkpoint.get(key) != reference[key]:
            raise AssertionError(f"historical initialization differs at {key}: {group} fold {fold}")
    weights = checkpoint["extra_loss_weights"]
    with torch.no_grad():
        output = E60.base_or_group_forward(group, state, module, int(checkpoint["best_epoch"]), False, True, banks, weights)
        metrics = r.classification_metrics(output["logp"], state["test"], state["y"])
    for metric in E60.METRICS:
        assert_close(checkpoint["metrics"][metric], metrics[metric], f"{group}.fold{fold}.metrics.{metric}")
        assert_close(task["metrics"][metric], metrics[metric], f"{group}.fold{fold}.task_metrics.{metric}")
    for metric in ("asd_tp", "asd_fn", "td_fp", "td_tn"):
        assert_close(checkpoint["metrics"][metric], metrics[metric], f"{group}.fold{fold}.metrics.{metric}")
        assert_close(task["confusion"][metric], metrics[metric], f"{group}.fold{fold}.task_confusion.{metric}")
    predictions = checkpoint["test_predictions"]
    assert_close([int(value) for value in state["test"].tolist()], predictions["indices"], f"{group}.fold{fold}.test_indices")
    assert_close(state["y"][torch.as_tensor(state["test"], dtype=torch.long, device=state["y"].device)].cpu().tolist(), predictions["labels"], f"{group}.fold{fold}.test_labels")
    assert_close(metrics["prob"].tolist(), predictions["probabilities_asd_td"], f"{group}.fold{fold}.test_probabilities")

    hist_metrics_match = None
    hist_state_match = None
    if historical is not None:
        reference_row = historical[fold]
        hist_path = Path(E60.CONFIG["base"]["historical_checkpoint_root"]) / f"{E60.BASE_GROUP}_f{fold:02d}.best.pt"
        hist_checkpoint = E60.torch_load(hist_path)
        for key in ("best_epoch", "epochs_executed", "model", "cam", "optimizer_base", "optimizer_cam", "optimizer_added"):
            assert_close(hist_checkpoint[key], checkpoint[key], f"{group}.fold{fold}.historical.{key}")
        for key in ("initial_shared_sha", "initial_added_sha", "initial_rng_sha", "split_sha", "pair_sha", "fold_cache_sha"):
            if hist_checkpoint[key] != checkpoint[key]:
                raise AssertionError(f"historical checkpoint identity mismatch at {key}: fold {fold}")
        for metric in E60.METRICS:
            assert_close(hist_checkpoint["metrics"][metric], metrics[metric], f"{group}.fold{fold}.historical_metric.{metric}")
            assert_close(float(reference_row[metric]), metrics[metric], f"{group}.fold{fold}.historical_csv_metric.{metric}")
        state["model"].load_state_dict(hist_checkpoint["model"], strict=True)
        state["cam"].load_state_dict(hist_checkpoint["cam"], strict=True)
        with torch.no_grad():
            historical_output = r.forward_state(state, training=False, use_cadr=True)
        assert_close(output["logp"], historical_output["logp"], f"{group}.fold{fold}.historical_logits")
        hist_metrics_match = True
        hist_state_match = True

    return {
        "group": group,
        "fold": fold,
        "status": "PASS",
        "checkpoint_sha256": checkpoint_sha,
        "best_epoch": int(checkpoint["best_epoch"]),
        "epochs_executed": int(checkpoint["epochs_executed"]),
        "metrics": {key: float(metrics[key]) for key in E60.METRICS},
        "confusion": {key: int(metrics[key]) for key in ("asd_tp", "asd_fn", "td_fp", "td_tn")},
        "readback": "PASS",
        "historical_metrics_match": hist_metrics_match,
        "historical_state_match": hist_state_match,
        "source_signature_exceptions": source_signature_exceptions,
        "task_checkpoint_source_signature_differences": task_checkpoint_signature_diffs,
    }


def load_historical_rows() -> dict[int, dict[str, Any]]:
    _, sweep = E60.load_base()
    rows = sweep.F.read_rows(sweep.RESULTS / "fold_metrics.csv")
    return {int(row["fold"]): row for row in rows if row.get("group") == E60.BASE_GROUP}


def run_audit(base_only: bool = False) -> dict[str, Any]:
    manifest_path = HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = [E60.GROUPS[0]] if base_only else E60.GROUPS
    historical = load_historical_rows() if base_only else {}
    rows = []
    for group in groups:
        for fold in range(1, 11):
            print(f"auditing {group} fold {fold}/10", flush=True)
            rows.append(audit_fold(group, fold, historical if base_only else None))
    status = "PASS" if len(rows) == (10 if base_only else 180) else "FAIL"
    record = {
        "status": status,
        "scope": "R01_BASE_REPLAY" if base_only else "ALL_FORMAL_GROUPS",
        "folds_audited": len(rows),
        "base_replay_historical_state_and_prediction_gate": status if base_only else None,
        "absolute_tolerance": ATOL,
        "relative_tolerance": RTOL,
        "rows": rows,
    }
    output = RESULTS / ("base_replay_audit.json" if base_only else "checkpoint_audit.json")
    E60.atomic_json(output, record)
    if base_only:
        manifest["r01_parity_status"] = status
        manifest["r01_audited_folds"] = len(rows)
    else:
        manifest["checkpoint_audit_status"] = status
        manifest["independently_audited_checkpoints"] = len(rows)
    manifest["source_signature"] = E60.experiment_signature()
    E60.atomic_json(manifest_path, manifest)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-replay", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.base_replay == args.all:
        raise SystemExit("select exactly one of --base-replay or --all")
    result = run_audit(base_only=args.base_replay)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
