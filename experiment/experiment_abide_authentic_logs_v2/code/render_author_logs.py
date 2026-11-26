#!/usr/bin/env python3
"""Render the author-format logs solely from this run's training records."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


V2 = Path(__file__).resolve().parents[1]
DPSL = Path("/model/spromptgl/experiment/DPSL")
AUTHOR_LOG = Path("/model/spromptgl/experiment/experiment_darg_jsl/author_log.py")
OUT = V2
sys.path[:0] = [str(DPSL / name) for name in ("code", "runtime/python", "code/history_base")]
import numpy as np


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_author_module():
    spec = importlib.util.spec_from_file_location("original_author_log_format", AUTHOR_LOG)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fold_data(stage, fold, author):
    endpoint_file = OUT / "Warmup/results" / f"fold{fold:02d}.endpoint.npz"
    with np.load(endpoint_file) as data:
        labels = data["labels"].astype(np.int64)
        warmup_train = data["train_indices"].astype(np.int64)
        warmup_test = data["test_indices"].astype(np.int64)
        if stage == "Warmup":
            train_indices = warmup_train
            metrics = read_json(OUT / "Warmup/results/endpoint_evaluation.json")["folds"][fold - 1]["test_metrics"]
            prediction = data["prediction"].astype(np.int64)
            probability = data["probability"]
            records = [dict(row, trn_cls=row["classification_ce"],
                             gfr_alignment=row["gfr_alignment"],
                             trn_auc=row["training_metrics"]["hard_auc"],
                             trn_acc=row["training_metrics"]["acc"],
                             val_auc=row["validation_metrics"]["hard_auc"],
                             val_acc=row["validation_metrics"]["acc"],
                             epoch=row["display_epoch"])
                       for row in read_jsonl(OUT / "Warmup/journal" / f"fold{fold:02d}.jsonl")]
            best_author = read_json(OUT / "Warmup/results/endpoint_evaluation.json")["folds"][fold - 1]
            elapsed = read_json(OUT / "Warmup/tasks" / f"fold{fold:02d}.json")["training_seconds"]
        elif stage == "DARG":
            audit = read_json(OUT / "DARG/tasks" / f"fold{fold:02d}.audit.json")
            train_indices = np.asarray(audit["train_members_1based"], dtype=np.int64) - 1
            rows = read_jsonl(OUT / "DARG/journal" / f"fold{fold:02d}.author.jsonl")
            records = [dict(row, epoch=row["epoch"]) for row in author] + rows
            checkpoint = OUT / "DARG/runtime/checkpoints" / f"E74.4-G2-Positive-DCR_f{fold:02d}.best.pt"
            import torch
            best = torch.load(checkpoint, map_location="cpu", weights_only=False)
            metrics = best["metrics"]
            prediction = np.asarray(best["test_prediction"], dtype=np.int64)
            probability = np.asarray(best["test_probability"])
            test_indices = np.asarray(audit["test_members_1based"], dtype=np.int64) - 1
            elapsed = (float(read_json(OUT / "Warmup/tasks" / f"fold{fold:02d}.json")["training_seconds"])
                       + float(audit["seconds"]))
            best_author = audit
        else:
            audit = read_json(OUT / "DPSL/tasks" / f"fold{fold:02d}.audit.json")
            train_indices = np.asarray(audit["train_members_1based"], dtype=np.int64) - 1
            rows = read_jsonl(OUT / "DPSL/journal" / f"fold{fold:02d}.author.jsonl")
            records = [dict(row, epoch=row["display_epoch"]) for row in author] + [
                dict(row, epoch=row["display_epoch"], trn_cls=row["trn_cls"]) for row in rows
            ]
            import torch
            checkpoint = OUT / "DPSL/run/checkpoints" / f"fold{fold:02d}.best.pt"
            best = torch.load(checkpoint, map_location="cpu", weights_only=False)
            test_indices = np.asarray(best["membership"]["test"], dtype=np.int64)
            test_labels = labels[test_indices]
            probability = np.asarray(best["evaluation"]["probability"])
            prediction = np.asarray(best["evaluation"]["prediction"], dtype=np.int64)
            from real_model_runtime import score
            metrics = score(test_labels, prediction, probability)
            run_task = read_json(OUT / "DPSL/run/journal" / f"fold{fold:02d}.task.json")
            elapsed = (float(read_json(OUT / "Warmup/tasks" / f"fold{fold:02d}.json")["training_seconds"])
                       + float(run_task["wall_seconds"]))
            best_author = audit
    train_labels = labels[train_indices]
    if stage == "Warmup":
        test_indices = warmup_test
    elif stage == "DPSL":
        test_indices = np.asarray(best["membership"]["test"], dtype=np.int64)
    test_labels = labels[test_indices]
    return {
        "fold": fold, "labels": test_labels, "train_counter": Counter(train_labels.tolist()),
        "records": records, "prediction": prediction, "probability": probability,
        "acc": float(metrics["acc"]), "hard_auc": float(metrics["hard_auc"]),
        "prob_auc": float(metrics["prob_auc"]), "sen": float(metrics["sen"]),
        "spe": float(metrics["spe"]), "asd_f1": float(metrics["asd_f1"]),
        "elapsed_sec": elapsed, "summary_source": best_author,
    }


def render(stage, author):
    endpoint_summary = read_json(OUT / "Warmup/results/endpoint_evaluation.json")
    if endpoint_summary.get("status") != "PASS" or len(endpoint_summary.get("folds", [])) != 10:
        raise RuntimeError("Warmup endpoint evaluation is incomplete")
    fold_summaries = []
    for fold in range(1, 11):
        warmup_records = read_jsonl(OUT / "Warmup/journal" / f"fold{fold:02d}.jsonl")
        if len(warmup_records) != 10:
            raise RuntimeError(f"Warmup fold {fold} does not have exactly ten real epochs")
        author_rows = [dict(row, epoch=row["display_epoch"], trn_cls=row["classification_ce"],
                            gfr_alignment=row["gfr_alignment"], trn_auc=row["training_metrics"]["hard_auc"],
                            trn_acc=row["training_metrics"]["acc"], val_auc=row["validation_metrics"]["hard_auc"],
                            val_acc=row["validation_metrics"]["acc"])
                       for row in warmup_records]
        additional = [] if stage == "Warmup" else author_rows
        if stage == "Warmup":
            author_rows = []
        current = fold_data(stage, fold, author_rows)
        fold_summaries.append({**current, "records": current["records"]})

    lines = []
    all_sensitivity, all_specificity, accuracies, hard_aucs = [], [], [], []
    for summary in fold_summaries:
        fold = summary["fold"]
        lines.append(str(summary["train_counter"]))
        for record in summary["records"]:
            lines.append(author.format_epoch(record, 450))
        prediction = np.asarray(summary["prediction"], dtype=np.int64)
        labels = np.asarray(summary["labels"], dtype=np.int64)
        sensitivity, specificity = author._sensitivity_specificity(labels, prediction)
        all_sensitivity.append(sensitivity)
        all_specificity.append(specificity)
        accuracies.append(summary["acc"])
        hard_aucs.append(summary["hard_auc"])
        lines.append(f"time: {summary['elapsed_sec']:.4f} sec.")
        lines.append(f"tst-auc: {summary['hard_auc']:.4f}  acc: {summary['acc']:.4f}")
        if stage == "Warmup":
            attention_path = OUT / "Warmup/attention" / f"fold{fold:02d}.npz"
        elif stage == "DARG":
            attention_path = OUT / "DARG/attention" / f"fold{fold:02d}.npz"
        else:
            attention_path = OUT / "DPSL/attention" / f"fold{fold:02d}.npz"
        if not attention_path.is_file() or not sha_file(attention_path):
            raise RuntimeError(f"attention maps are absent for {stage} fold {fold}")
        with np.load(attention_path) as maps:
            if not maps.files or any(not maps[key].size for key in maps.files):
                raise RuntimeError(f"attention maps are empty for {stage} fold {fold}")
        lines.append("attention maps have been saved.")
        lines.append(f"sens: {all_sensitivity}")
        lines.append(f"spes: {all_specificity}")

    accuracy = np.asarray(accuracies, dtype=np.float64)
    hard_auc = np.asarray(hard_aucs, dtype=np.float64)
    sensitivity_array = np.asarray(all_sensitivity, dtype=np.float64)
    specificity_array = np.asarray(all_specificity, dtype=np.float64)
    lines.append(f"SEN_mean: {sensitivity_array.mean(axis=0)} SEN_std: {sensitivity_array.std(axis=0)}")
    lines.append(f"SPE_mean: {specificity_array.mean(axis=0)} SPE_std: {specificity_array.std(axis=0)}")
    lines.append(f"val={accuracy.mean() * 100:.2f}% tst_acc={accuracy.mean() * 100:.2f}% tst_auc={hard_auc.mean() * 100:.2f}%")
    lines.append(f"tst_acc_std={accuracy.std(ddof=0):.4f} tst_auc_std={hard_auc.std(ddof=0):.4f}")
    lines.append(f"({float(accuracy.mean())}, {float(accuracy.mean())})")
    text = "\n".join(lines) + "\n"
    output = OUT / stage / "result_ABIDE.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)
    author._validate_emitted(text, 10)
    expected_rows = sum(len(item["records"]) for item in fold_summaries)
    if sum(line.lstrip().split("/", 1)[0].isdigit() and "/450 trn-loss-MF:" in line for line in lines) != expected_rows:
        raise RuntimeError(f"author row count mismatch for {stage}")
    return {"path": str(output), "lines": len(lines), "epoch_rows": expected_rows,
            "sha256": sha_file(output), "bytes": output.stat().st_size,
            "fields_and_spacing": "PASS", "attention_maps_verified": 10,
            "folds": [{key: item[key] for key in ("fold", "acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1", "elapsed_sec")}
                      for item in fold_summaries]}


def main():
    author = load_author_module()
    template = author.check_template()
    stage_rows = {}
    for stage in ("Warmup", "DARG", "DPSL"):
        if stage == "DARG" and not (OUT / "DARG/results/retrain_summary.json").is_file():
            continue
        if stage == "DPSL" and not (OUT / "DPSL/results/retrain_summary.json").is_file():
            continue
        stage_rows[stage] = render(stage, author)
    atomic = {"source_template": template, "stages": stage_rows}
    (OUT / "results/author_log_render.json").write_text(json.dumps(atomic, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(atomic, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
