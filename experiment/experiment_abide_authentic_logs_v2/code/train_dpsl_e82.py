#!/usr/bin/env python3
"""Replay the frozen DPSL full-training entry into the isolated v2 output."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path


V2 = Path(__file__).resolve().parents[1]
DPSL = Path("/model/spromptgl/experiment/DPSL")
REFERENCE = Path("/model/spromptgl/experiment/DPSL_full_retrain_from_warmup")
OUT = V2 / "DPSL"
RUN = OUT / "run"

sys.dont_write_bytecode = True
sys.path[:0] = [str(DPSL / name) for name in ("code", "runtime/python", "code/history_base")]

import numpy as np
import torch
from dpsl_runtime import FOLDS, read_json, sha256_file
from real_model_runtime import score
import training_runtime as training_runtime_module
from training_runtime import fold_engine, history_record, run_fold


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def array_receipt(value) -> dict[str, str | list[int]]:
    array = np.asarray(value, dtype="<f4", order="C")
    return {"shape": list(array.shape), "dtype": "<f4",
            "base64": base64.b64encode(array.tobytes()).decode("ascii")}


def metrics_from_logits(labels, logits):
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    probability = np.exp(values)
    probability /= probability.sum(axis=1, keepdims=True)
    prediction = probability.argmax(axis=1).astype(np.int64)
    return score(np.asarray(labels, dtype=np.int64), prediction, probability), probability


def attention_hooks(ctx, state):
    captures = {"logits": {}, "attention": {}}
    handles = []
    classifier = state["cam"].Outputlayer.classifier

    def capture_logits(layer, _inputs, output):
        if output.ndim != 2 or output.shape[0] != 871:
            return
        phase = "train" if layer.training else "eval"
        captures["logits"][phase] = output.detach().cpu().numpy().copy()

    handles.append(classifier.register_forward_hook(capture_logits))
    for name, layer in state["cam"].named_modules():
        if name.endswith(".attention"):
            def capture_attention(current, _inputs, output, module_name=name):
                if not current.training and isinstance(output, (tuple, list)) and len(output) > 1:
                    value = output[1]
                    if torch.is_tensor(value):
                        captures["attention"][module_name] = value.detach().cpu().numpy().copy()
            handles.append(layer.register_forward_hook(capture_attention))
    return captures, handles


def write_attention(fold, maps):
    if not maps or any(not np.asarray(value).size or not np.isfinite(value).all() for value in maps.values()):
        raise RuntimeError(f"DPSL fold {fold} did not produce finite attention maps")
    path = OUT / "attention" / f"fold{fold:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **maps)
        stream.flush()
        os.fsync(stream.fileno())
    return path, sha256_file(path)


def append_jsonl(path: Path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                rows[int(row["internal_epoch"])] = row
    epoch = int(record["internal_epoch"])
    existing = rows.get(epoch)
    if existing is not None:
        if existing != record:
            raise RuntimeError(f"DPSL author observation changed during resume at epoch {epoch}")
        return
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def install_author_capture(engine, fold):
    original_transition = engine.transition
    author_path = OUT / "journal" / f"fold{fold:02d}.author.jsonl"
    terminal_epoch = int(history_record(fold, "task")["epoch"]) + 1
    tracking = {"best_correct": None, "handles": None, "captures": None,
                "original_transition": original_transition,
                "terminal_witness_epoch": terminal_epoch}
    engine._author_log_path = author_path
    engine._author_attention_sha = None

    def observe_transition(ctx, epoch, dropout, *, inspect=False):
        if tracking["captures"] is None:
            tracking["captures"], tracking["handles"] = attention_hooks(ctx, ctx["state"])
            latest = RUN / "checkpoints" / f"fold{fold:02d}.latest.pt"
            tracking["best_correct"] = engine.h.load(latest)["best_correct"] if latest.is_file() else -1
        captures = tracking["captures"]
        captures["logits"].clear()
        captures["attention"].clear()
        started = time.perf_counter()
        result = original_transition(ctx, epoch, dropout, inspect=inspect)
        evaluation, diagnostics, rng, _previous_ema, _observation = result
        if set(captures["logits"]) != {"train", "eval"}:
            raise RuntimeError(f"DPSL classifier logits hook missed a forward at fold {fold}, epoch {epoch}")
        if engine.h.rng_sha() != rng["rng_after_eval"]:
            raise RuntimeError("DPSL observation happened after the official transition RNG boundary")

        labels = ctx["state"]["labels_np"]
        train_indices = np.asarray(ctx["members"]["train"], dtype=np.int64)
        train_logits = captures["logits"]["train"]
        train_metrics, _train_probability = metrics_from_logits(labels[train_indices], train_logits[train_indices])
        test_indices = np.asarray(ctx["members"]["test"], dtype=np.int64)
        test_labels = labels[test_indices]
        eval_metrics = score(test_labels, evaluation["prediction"], evaluation["probability"])
        if eval_metrics["correct"] != evaluation["correct"]:
            raise RuntimeError("DPSL validation metric count differs from the official transition")

        if evaluation["correct"] > tracking["best_correct"]:
            if set(captures["attention"]) == set():
                raise RuntimeError(f"DPSL Best epoch attention maps are missing in fold {fold}")
            _path, digest = write_attention(fold, captures["attention"])
            engine._author_attention_sha = digest
            tracking["best_correct"] = evaluation["correct"]

        terminal_witness = inspect and int(epoch) == tracking["terminal_witness_epoch"]
        record = {
            "stage": "E82.4.6.10-G1", "fold": fold,
            "event_kind": "terminal_witness" if terminal_witness else "training_epoch",
            "internal_epoch": int(epoch), "display_epoch": int(epoch) - 1,
            "completed_updates": int(epoch),
            "trn_cls": float(diagnostics["losses"]["cls"]),
            "gfr_alignment": float(diagnostics["losses"]["graph"]),
            "trn_auc": float(train_metrics["hard_auc"]), "trn_acc": float(train_metrics["acc"]),
            "val_auc": float(eval_metrics["hard_auc"]), "val_acc": float(eval_metrics["acc"]),
            "train_metrics": {key: float(train_metrics[key]) for key in
                              ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")},
            "validation_metrics": {key: float(eval_metrics[key]) for key in
                                   ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")},
            "rng": rng, "dropout_order": diagnostics["dropout"],
            "gradient_budget": diagnostics["budget"], "elapsed_epoch_seconds": time.perf_counter() - started,
            "train_logits": array_receipt(train_logits[train_indices]),
            "train_labels": array_receipt(labels[train_indices]),
            "validation_probability": array_receipt(evaluation["probability"]),
            "validation_prediction": array_receipt(evaluation["prediction"]),
            "validation_labels": array_receipt(test_labels),
            "logging_rng_unchanged": True,
        }
        if engine.h.rng_sha() != rng["rng_after_eval"]:
            raise RuntimeError("DPSL metric and journal capture consumed RNG")
        if terminal_witness:
            diagnostic_path = OUT / "journal" / "diagnostic" / f"fold{fold:02d}.terminal_witness.jsonl"
            append_jsonl(diagnostic_path, record)
        else:
            append_jsonl(author_path, record)
        return result

    engine.transition = observe_transition
    return tracking


def compare_reference(engine, fold, files, result, tracked):
    old_task_path = REFERENCE / "journal" / f"fold{fold:02d}.task.json"
    old_task = read_json(old_task_path)
    old_best = engine.h.load(REFERENCE / "checkpoints" / f"fold{fold:02d}.best.pt")
    old_latest = engine.h.load(REFERENCE / "checkpoints" / f"fold{fold:02d}.latest.pt")
    new_best = engine.h.load(files["best"])
    new_latest = engine.h.load(files["latest"])
    task = history_record(fold, "task")
    author_rows_all = [json.loads(line) for line in tracked["author_log_path"].read_text(encoding="utf-8").splitlines()]
    formal_rows = [json.loads(line) for line in files["journal"].read_text(encoding="utf-8").splitlines()]
    historical_rows = [json.loads(line) for line in (DPSL / "runtime/history" / f"fold{fold:02d}.journal.jsonl").read_text(encoding="utf-8").splitlines()]
    diagnostic_path = OUT / "journal" / "diagnostic" / f"fold{fold:02d}.terminal_witness.jsonl"
    diagnostic_rows = ([json.loads(line) for line in diagnostic_path.read_text(encoding="utf-8").splitlines()]
                       if diagnostic_path.is_file() else [])
    all_observations = author_rows_all + diagnostic_rows
    author_rows = sorted((dict(row, event_kind="training_epoch") for row in all_observations
                          if int(row["internal_epoch"]) <= int(result["epoch"])),
                         key=lambda row: int(row["internal_epoch"]))
    terminal_by_epoch = {}
    for row in all_observations:
        if int(row["internal_epoch"]) != int(result["epoch"]) + 1:
            continue
        event = dict(row, event_kind="terminal_witness")
        existing = terminal_by_epoch.get(int(row["internal_epoch"]))
        if existing is not None and existing != event:
            raise RuntimeError(f"DPSL terminal witness observations disagree in fold {fold}")
        terminal_by_epoch[int(row["internal_epoch"])] = event
    terminal_witness_rows = list(terminal_by_epoch.values())
    if [int(row["internal_epoch"]) for row in author_rows] != list(range(11, int(result["epoch"]) + 1)):
        raise RuntimeError(f"DPSL author observations do not cover every formal epoch in fold {fold}")
    if terminal_witness_rows:
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text("".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                                             for row in terminal_witness_rows), encoding="utf-8")
    author_path = tracked["author_log_path"]
    author_path.write_text("".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                                   for row in author_rows), encoding="utf-8")
    author_by_epoch = {int(row["internal_epoch"]): row for row in author_rows}
    if len(author_by_epoch) != len(author_rows) or [int(row["epoch"]) for row in formal_rows] != list(range(11, result["epoch"] + 1)):
        raise RuntimeError(f"DPSL author or training journal epoch range is invalid in fold {fold}")
    if [row["rng"] for row in author_rows] != [row["rng"] for row in formal_rows]:
        raise RuntimeError(f"DPSL logger RNG rows do not match the official transition in fold {fold}")
    if formal_rows != historical_rows:
        raise RuntimeError(f"DPSL full training journal differs from the historical full-retrain journal in fold {fold}")
    if not (
        result["epoch"] == old_task["epoch"] and result["best_epoch"] == old_task["best_epoch"]
        and result["best_correct"] == old_task["best_correct"]
        and new_best["state_sha"] == old_best["state_sha"]
        and engine.h.sha(new_best["evaluation"]) == engine.h.sha(old_best["evaluation"])
        and engine.h.sha(new_best["state"]) == engine.h.sha(old_best["state"])
        and new_latest["state_sha"] == old_latest["state_sha"]
        and new_latest["rng_sha"] == old_latest["rng_sha"]
        and engine.h.sha(new_latest["ema"]) == engine.h.sha(old_latest["ema"])
        and engine.h.sha(new_latest["state"]) == engine.h.sha(old_latest["state"])
        and new_latest["wait"] == old_latest["wait"]
        and new_latest["best_correct"] == old_latest["best_correct"]
        and result["next_witness"] == old_task["next_witness"]
    ):
        raise RuntimeError(f"DPSL retraining Best/Latest differs from full-retrain history in fold {fold}")
    if tracked["attention_sha"] is None:
        path = OUT / "attention" / f"fold{fold:02d}.npz"
        if not path.is_file():
            raise RuntimeError(f"DPSL attention map is absent for fold {fold}")
        tracked["attention_sha"] = sha256_file(path)
    return {
        "fold": fold, "status": "TRAINED_AUDITED_EXACT_FROM_WARMUP",
        "best_epoch": result["best_epoch"], "latest_epoch": result["epoch"],
        "best_correct": result["best_correct"], "warmup_rng_sha": author_rows[0]["rng"]["rng_before"],
        "formal_rng_trace_exact": True, "author_rows": len(author_rows),
        "best_sha256": sha256_file(files["best"]), "latest_sha256": sha256_file(files["latest"]),
        "journal_sha256": sha256_file(files["journal"]),
        "author_sha256": sha256_file(tracked["author_log_path"]),
        "attention_sha256": tracked["attention_sha"],
        "historical_best_latest_next_witness_exact": True,
        "patient_predictions_exact": new_best["evaluation"]["prediction"].tolist()
            == old_best["evaluation"]["prediction"].tolist(),
        "patient_probability_max_abs_error": float(np.max(np.abs(
            np.asarray(new_best["evaluation"]["probability"]) - np.asarray(old_best["evaluation"]["probability"])))),
        "train_members_1based": [int(i) + 1 for i in new_best["membership"]["train"]],
        "test_members_1based": [int(i) + 1 for i in new_best["membership"]["test"]],
    }


def bind_run_factory(engine, fold):
    original_factory = training_runtime_module.fold_engine

    def bound_factory(requested_fold):
        if requested_fold == fold:
            return engine
        return original_factory(requested_fold)

    training_runtime_module.fold_engine = bound_factory
    if training_runtime_module.fold_engine(fold) is not engine:
        training_runtime_module.fold_engine = original_factory
        raise RuntimeError(f"DPSL training entry did not receive the captured engine for fold {fold}")
    return original_factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--check-engine-binding", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    if torch.cuda.get_device_name(0) != "PPU-ZW810E":
        raise RuntimeError(f"physical GPU 0 changed: {torch.cuda.get_device_name(0)}")
    folds = [args.fold] if args.fold else list(FOLDS)
    summaries = []
    for fold in folds:
        engine = fold_engine(fold)
        tracking = install_author_capture(engine, fold)
        tracking["author_log_path"] = engine._author_log_path
        original_factory = bind_run_factory(engine, fold)
        if args.check_engine_binding:
            training_runtime_module.fold_engine = original_factory
            print(f"DPSL fold {fold:02d} logger-engine binding PASS", flush=True)
            continue
        started = time.perf_counter()
        try:
            engine, files, result = run_fold(RUN, fold)
        finally:
            engine.transition = tracking["original_transition"]
            for handle in tracking["handles"] or []:
                handle.remove()
            training_runtime_module.fold_engine = original_factory
        tracking["attention_sha"] = engine._author_attention_sha
        summary = compare_reference(engine, fold, files, result, tracking)
        summary["wall_seconds"] = time.perf_counter() - started
        summary["peak_gpu_bytes"] = int(torch.cuda.max_memory_allocated())
        atomic_json(OUT / "tasks" / f"fold{fold:02d}.audit.json", summary)
        summaries.append(summary)
        print(f"DPSL fold {fold:02d} end={summary['latest_epoch']} best={summary['best_epoch']} audited", flush=True)
    existing = {}
    summary_path = OUT / "results" / "retrain_summary.json"
    if summary_path.is_file():
        existing = {int(row["fold"]): row for row in read_json(summary_path).get("folds", [])}
    for row in summaries:
        existing[int(row["fold"])] = row
    ordered = [existing[fold] for fold in sorted(existing)]
    atomic_json(summary_path, {
        "experiment": "E82.4.6.10-G1", "folds": ordered,
        "status": "TRAINED_AUDITED_EXACT_FROM_WARMUP" if len(ordered) == 10 else "PARTIAL",
        "training_seconds": sum(float(row["wall_seconds"]) for row in ordered),
        "device": torch.cuda.get_device_name(0), "workers": 1,
    })


if __name__ == "__main__":
    main()
