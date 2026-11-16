#!/usr/bin/env python3
"""Independent checkpoint, prediction, metric, and resume audit for E80.1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
spec = importlib.util.spec_from_file_location("experiment80_1_audited_runner", EXP / "run.py")
run = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run
spec.loader.exec_module(run)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_metrics(labels, probabilities):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(y), 2) or not np.isfinite(p).all():
        raise ValueError("invalid ASD/TD probability matrix")
    pred = p.argmax(axis=1)
    tn_cm = confusion_matrix(y, pred, labels=[0, 1])
    tp, fn, fp, tn = (int(x) for x in tn_cm.ravel())
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    return {
        "acc": float(np.mean(pred == y)),
        "hard_auc": float(roc_auc_score(np.eye(2)[y], np.eye(2)[pred])),
        "prob_auc": float(roc_auc_score(y, p[:, 1])),
        "sen": float(sen),
        "spe": float(spe),
        "ba": float((sen + spe) / 2),
        "asd_f1": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "asd_tp": tp,
        "asd_fn": fn,
        "td_fp": fp,
        "td_tn": tn,
        "n_test": int(len(y)),
        "pred": pred,
    }


def _same_metrics(left, right, atol=1e-10):
    keys = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
    return all(abs(float(left[key]) - float(right[key])) <= atol for key in keys)


def _task_payload(group, fold):
    task = json.loads(run.task_path(group, fold).read_text(encoding="utf-8"))
    best_path = run.checkpoint_path(group, fold, "best")
    latest_path = run.checkpoint_path(group, fold, "latest")
    if task.get("status") != "COMPLETE" or task.get("group") != group or int(task.get("fold", -1)) != fold:
        raise RuntimeError(f"task identity/status failed: {group} fold {fold}")
    if task.get("source_sha256") != run.source_sha():
        raise RuntimeError(f"task source hash differs from audited source: {group} fold {fold}")
    if task.get("checkpoint_sha256") != sha_file(best_path):
        raise RuntimeError(f"Best Checkpoint SHA mismatch: {group} fold {fold}")
    if task.get("latest_sha256") != sha_file(latest_path):
        raise RuntimeError(f"Latest Checkpoint SHA mismatch: {group} fold {fold}")
    best = run.load_history().torch_load(best_path)
    latest = run.load_history().torch_load(latest_path)
    for payload, kind in ((best, "BEST"), (latest, "LATEST")):
        if payload.get("group") != group or int(payload.get("fold", -1)) != fold:
            raise RuntimeError(f"{kind} checkpoint task identity failed: {group} fold {fold}")
        if payload.get("source_sha256") != task["source_sha256"] or payload.get("identity") != task["identity"]:
            raise RuntimeError(f"{kind} checkpoint provenance differs from task record: {group} fold {fold}")
    if best.get("checkpoint_kind") != "BEST" or latest.get("checkpoint_kind") != "LATEST":
        raise RuntimeError(f"checkpoint kind metadata mismatch: {group} fold {fold}")
    if int(best["best_epoch"]) != int(task["best_epoch_zero_based"]):
        raise RuntimeError(f"best epoch mismatch: {group} fold {fold}")
    return task, best, latest, best_path, latest_path


def _restore_best(group, fold, task, best):
    state, warmup, identity, module, optimizer, private_rng, scaler_sha, prompts = run.build_task_state(group, fold)
    if identity != task["identity"] or identity["warmup_sha256"] != task["warmup_sha256"]:
        raise RuntimeError(f"warmup/split identity changed during independent restore: {group} fold {fold}")
    run.load_history().load_history().load_bundle(state, best["best"]["state"])
    module.load_state_dict(best["best"]["psdc"], strict=True)
    base = run.load_historical_e61().load_base()
    base.restore_rng(best["best"]["rng_shared"])
    state["cam"].eval()
    state["model"].eval()
    module.eval()
    with torch.no_grad():
        output = run.forward_psdc(state, module, group, training=False)
    indices = np.asarray(task["test_indices"], dtype=np.int64)
    metrics = run.metrics_for(output["logp"], indices, state["y"])
    prob = np.asarray(metrics["prob"], dtype=np.float64)
    independently_recomputed = independent_metrics(np.asarray(state["y"].detach().cpu())[indices], prob)
    if not _same_metrics(metrics, independently_recomputed):
        raise RuntimeError(f"independent metric calculation differs: {group} fold {fold}")
    if not np.array_equal(metrics["pred"], np.asarray(best["best"]["prediction"], dtype=np.int64)):
        raise RuntimeError(f"independently restored predictions differ from Best Checkpoint: {group} fold {fold}")
    if not np.allclose(prob, np.asarray(best["best"]["probability"], dtype=np.float64), rtol=0, atol=1e-7):
        raise RuntimeError(f"independently restored probabilities differ from Best Checkpoint: {group} fold {fold}")
    if not _same_metrics(independently_recomputed, task["metrics"], atol=1e-10):
        raise RuntimeError(f"task metrics do not match independent prediction metrics: {group} fold {fold}")
    return state, module, optimizer, private_rng, scaler_sha, prompts, metrics


def _replay_latest(group, fold, state, module, optimizer, private_rng, latest):
    history = run.load_history().load_history()
    base = run.load_historical_e61().load_base()

    def restore():
        history.load_bundle(state, latest["current"])
        module.load_state_dict(latest["current_psdc"], strict=True)
        optimizer.load_state_dict(latest["current_opt_psdc"])
        private_rng.load_state_dict(latest["private_rng"])
        base.restore_rng(latest["shared_rng"])

    restore()
    first = run.optimizer_step(state, module, optimizer, group, training=True)
    first_base = history.state_bundle(state)
    first_psdc = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    first_opt = optimizer.state_dict()
    first_private = private_rng.state_dict()
    first_shared = base.pack_rng()
    first_loss = float(first["total"].detach().cpu())

    restore()
    replay = run.optimizer_step(state, module, optimizer, group, training=True)
    replay_base = history.state_bundle(state)
    replay_shared = base.pack_rng()
    if (run.object_sha(first_base) != run.object_sha(replay_base) or
            run.object_sha(first_psdc) != run.object_sha(module.state_dict()) or
            run.object_sha(first_opt) != run.object_sha(optimizer.state_dict()) or
            run.object_sha(first_private) != run.object_sha(private_rng.state_dict()) or
            base.rng_sha(first_shared) != base.rng_sha(replay_shared) or
            first_loss != float(replay["total"].detach().cpu())):
        raise RuntimeError(f"Latest Checkpoint one-step replay differs: {group} fold {fold}")
    return float(latest["next_epoch"]), first_loss


def audit_all():
    result_rows = []
    for fold in range(1, 11):
        for group in run.GROUPS:
            task, best, latest, best_path, latest_path = _task_payload(group, fold)
            state, module, optimizer, private_rng, scaler_sha, prompts, metrics = _restore_best(group, fold, task, best)
            next_epoch, replay_loss = _replay_latest(group, fold, state, module, optimizer, private_rng, latest)
            result_rows.append({
                "group": group, "fold": fold, "best_checkpoint_sha256": sha_file(best_path),
                "latest_checkpoint_sha256": sha_file(latest_path), "best_epoch_zero_based": task["best_epoch_zero_based"],
                "best_epoch_1based": task["best_epoch_1based"], "n_test": int(metrics["n_test"]),
                "metrics_recomputed": True, "predictions_recomputed": True,
                "latest_next_epoch": int(next_epoch), "latest_one_step_replay": "PASS",
                "warmup_sha256": task["warmup_sha256"], "split_sha": task["identity"]["split_sha"],
                "pair_sha": task["identity"]["pair_sha"], "scaler_sha": scaler_sha,
                "prompt_sha": task["identity_doc"]["prompt_text_sha"],
                "rng_parity_exact_epochs": task["rng_parity"]["exact_triplets"],
                "rng_parity_mismatches": task["rng_parity"]["mismatches"],
                "latest_replay_loss": replay_loss,
            })
            print(f"AUDITED {group} fold={fold} best/latest PASS", flush=True)
    output = run.RESULTS / "checkpoint_audit.csv"
    run.atomic_csv(output, result_rows)
    status = {
        "status": "PASS_30_OF_30",
        "audited_tasks": len(result_rows),
        "best_checkpoints_independently_restored": len(result_rows),
        "latest_checkpoints_one_step_replayed_twice": len(result_rows),
        "source_sha256": run.source_sha(),
        "checkpoint_audit_csv_sha256": sha_file(output),
        "all_fold2_overlap_rows_retained": True,
        "historical_selection_protocol": "val=test; retained unchanged and disclosed",
    }
    run.atomic_json(run.RESULTS / "audit_status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(audit_all(), indent=2))


if __name__ == "__main__":
    main()
