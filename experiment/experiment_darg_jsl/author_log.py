"""Emit and validate the original SPromptGL ABIDE console format."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "inputs/log_template/result_ABIDE.template.txt"
REFERENCE = Path("/data/adodas/SPromptGL-main/result_ABIDE.txt")


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _epoch_rows(text):
    return [line for line in text.splitlines() if re.match(r"^\s*\d+/\d+ trn-loss-MF:", line)]


def _canonical_epoch(line):
    line = re.sub(r"^\s*\d+/\d+", "<epoch>", line)
    return re.sub(r"(?<=: )[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", "<value>", line)


def _canonical_summary(line):
    return re.sub(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", "<value>", line)


def check_template():
    if not TEMPLATE.is_file() or not REFERENCE.is_file():
        raise RuntimeError("original ABIDE log template or source is missing")
    if TEMPLATE.read_bytes() != REFERENCE.read_bytes():
        raise RuntimeError("isolated author-log template differs from the read-only source log")
    sample = _epoch_rows(TEMPLATE.read_text(encoding="utf-8"))[0]
    expected_fields = ["trn-loss-MF", "trn-loss-MF", "trn-loss-MF", "trn-auc", "acc", "val-auc", "acc"]
    if re.findall(r"[A-Za-z-]+(?=:)", sample) != expected_fields:
        raise RuntimeError("author epoch field count/order changed")
    generated = format_epoch({"epoch": 0, "trn_cls": 1.0, "gfr_alignment": 2.0,
                              "trn_auc": 0.5, "trn_acc": 0.5, "val_auc": 0.5, "val_acc": 0.5}, 450)
    if _canonical_epoch(sample) != _canonical_epoch(generated):
        raise RuntimeError("author epoch spacing or field order differs from the source template")
    expected_summary_prefixes = (
        "time:", "tst-auc:", "attention maps have been saved.", "sens:", "spes:",
        "SEN_mean:", "SPE_mean:", "val=", "tst_acc_std=", "("
    )
    for prefix in expected_summary_prefixes:
        if prefix not in TEMPLATE.read_text(encoding="utf-8"):
            raise RuntimeError(f"source author log is missing its {prefix!r} record")
    return {"sha256": sha_file(TEMPLATE), "reference_sha256": sha_file(REFERENCE),
            "epoch_fields": expected_fields, "epoch_spacing": "PASS", "summary_fields": "PASS"}


def format_epoch(record, max_epochs):
    cls = float(record["trn_cls"])
    return (
        f"{int(record['epoch']):3d}/{int(max_epochs)} trn-loss-MF: {cls:.4f}  "
        f"trn-loss-MF: {cls:.4f} trn-loss-MF: {float(record['gfr_alignment']):.4f} "
        f"trn-auc: {float(record['trn_auc']):.4f}  acc: {float(record['trn_acc']):.4f} "
        f"val-auc: {float(record['val_auc']):.4f}  acc: {float(record['val_acc']):.4f}"
    )


def _sensitivity_specificity(labels, prediction):
    matrix = np.zeros((2, 2), dtype=np.int64)
    np.add.at(matrix, (labels.astype(np.int64), prediction.astype(np.int64)), 1)
    sensitivity = []
    specificity = []
    for label in range(2):
        tp = matrix[label, label]
        fn = matrix[label, :].sum() - tp
        fp = matrix[:, label].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        sensitivity.append(float(tp / (tp + fn + 1e-8)))
        specificity.append(float(tn / (tn + fp + 1e-8)))
    return sensitivity, specificity


def _read_jsonl(path):
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def render_result(fold_summaries, labels, max_epochs):
    check_template()
    lines = [str(Counter(np.asarray(labels, dtype=np.int64).tolist()))]
    all_sensitivity = []
    all_specificity = []
    accuracies = []
    hard_aucs = []

    for summary in fold_summaries:
        fold = int(summary["fold"])
        records = _read_jsonl(ROOT / "runtime/journal" / f"fold{fold:02d}.jsonl")
        for row in records:
            lines.append(format_epoch(row["author_log"], max_epochs))

        labels_fold = np.asarray(summary["labels"], dtype=np.int64)
        prediction = np.asarray(summary["prediction"], dtype=np.int64)
        sensitivity, specificity = _sensitivity_specificity(labels_fold, prediction)
        all_sensitivity.append(sensitivity)
        all_specificity.append(specificity)
        accuracies.append(float(summary["acc"]))
        hard_aucs.append(float(summary["hard_auc"]))
        lines.append(f"time: {float(summary['elapsed_sec']):.4f} sec.")
        lines.append(f"tst-auc: {float(summary['hard_auc']):.4f}  acc: {float(summary['acc']):.4f}")
        if not (ROOT / "artifacts/attention" / f"fold{fold:02d}.npy").is_file():
            raise RuntimeError(f"fold {fold} attention map was not saved")
        if not (ROOT / "artifacts/graph" / f"fold{fold:02d}.npz").is_file():
            raise RuntimeError(f"fold {fold} graph features were not saved")
        lines.append("attention maps have been saved.")
        lines.append(f"sens: {all_sensitivity}")
        lines.append(f"spes: {all_specificity}")

    acc = np.asarray(accuracies, dtype=np.float64)
    auc = np.asarray(hard_aucs, dtype=np.float64)
    lines.append(f"SEN_mean: {np.asarray(all_sensitivity).mean(axis=0)} SEN_std: {np.asarray(all_sensitivity).std(axis=0)}")
    lines.append(f"SPE_mean: {np.asarray(all_specificity).mean(axis=0)} SPE_std: {np.asarray(all_specificity).std(axis=0)}")
    lines.append(f"val={acc.mean() * 100:.2f}% tst_acc={acc.mean() * 100:.2f}% tst_auc={auc.mean() * 100:.2f}%")
    lines.append(f"tst_acc_std={acc.std(ddof=0):.4f} tst_auc_std={auc.std(ddof=0):.4f}")
    lines.append(f"({float(acc.mean())}, {float(acc.mean())})")
    result = "\n".join(lines) + "\n"
    _validate_emitted(result, len(fold_summaries))
    temporary = ROOT / "result_ABIDE.txt.tmp"
    temporary.write_text(result, encoding="utf-8")
    temporary.replace(ROOT / "result_ABIDE.txt")
    return {"sha256": sha_file(ROOT / "result_ABIDE.txt"), "lines": len(lines),
            "epoch_rows": len(_epoch_rows(result)), "folds": len(fold_summaries)}


def _validate_emitted(text, fold_count):
    template = TEMPLATE.read_text(encoding="utf-8")
    expected_epoch = _canonical_epoch(_epoch_rows(template)[0])
    emitted_epochs = _epoch_rows(text)
    if any(_canonical_epoch(line) != expected_epoch for line in emitted_epochs):
        raise RuntimeError("generated author epoch rows changed field ordering or spacing")
    if text.count("attention maps have been saved.") != fold_count:
        raise RuntimeError("generated author log attention-map record count differs from fold count")
    for prefix, expected in (("time:", fold_count), ("tst-auc:", fold_count), ("sens:", fold_count), ("spes:", fold_count)):
        if sum(line.startswith(prefix) for line in text.splitlines()) != expected:
            raise RuntimeError(f"generated author log {prefix} record count differs from fold count")
    for summary in ("SEN_mean:", "SPE_mean:", "val=", "tst_acc_std=", "("):
        if sum(line.startswith(summary) for line in text.splitlines()) != 1:
            raise RuntimeError(f"generated author log {summary} summary count differs")
