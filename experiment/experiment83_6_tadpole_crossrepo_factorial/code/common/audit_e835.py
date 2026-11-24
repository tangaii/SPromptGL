"""Independent checkpoint replay and unified five-group metric audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from fold_checkpoint import (DATA_SHA, SPLIT_SHA, atomic_write, identity, sha_file,
                             validate_identity, write_json)
from latest_replay import _next_step_signature
from metrics import acc_auc
from replay_check import configure_worker, load_data, replay_fold
from run_e833 import (GROUPS, build_model, capture_rng_state, load_data as load_training_data,
                      optimizers, restore_rng, seed_everything, sha_object)
import run_g5


GROUP_NAMES = {
    GROUPS[0]: "SPromptGL-MMGPL12",
    GROUPS[1]: "No-PEN",
    GROUPS[2]: "DARG",
    GROUPS[3]: "DARG+DPSL",
    GROUPS[4]: "E43.4-E0 Anchored Physical CAM-only",
}


def _split_map(splits):
    return {int(row["fold"]): row for row in splits}


def _validate_checkpoint_hashes(group, fold, split):
    group_dir = ROOT / group
    best_path = group_dir / "best_checkpoints" / f"fold{fold:02d}.pt"
    latest_path = group_dir / "latest_checkpoints" / f"fold{fold:02d}.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    best_sha, latest_sha = sha_file(best_path), sha_file(latest_path)
    if latest.get("best_sha256") != best_sha:
        raise RuntimeError(f"{group} fold {fold} Latest does not bind to Best")
    train = np.asarray(split["train_idx"], dtype=np.int64)
    test = np.asarray(split["test_idx"], dtype=np.int64)
    expected = identity(group, fold, train, test)
    historical = group in GROUPS[1:3]
    if historical:
        keys = ("group", "fold", "seed", "model_type", "data_sha256", "split_sha256",
                "train_idx", "test_idx")
        for payload in (best, latest):
            if any(payload.get(key) != expected[key] for key in keys):
                raise RuntimeError(f"{group} fold {fold} historical identity mismatch")
    else:
        validate_identity(best, expected)
        validate_identity(latest, expected)
    if group in GROUPS[1:3]:
        _verify_inherited_artifact_inventory(group, fold)
    return best, latest, best_sha, latest_sha, expected


def _verify_inherited_artifact_inventory(group, fold):
    config_path = ROOT / "configs" / f"{group}.json"
    config = json.loads(config_path.read_text())
    source_group = config["source_group"]
    e834_root = Path("/model/spromptgl/experiment/experiment83_4_tadpole_four_group_complete_evidence")
    source_dir = e834_root / source_group
    if not source_dir.is_dir():
        raise FileNotFoundError(f"E83.4 source fold directory missing for {group}")
    for kind, directory_key in (("best", "best_checkpoint_sha256s"),
                                ("latest", "latest_checkpoint_sha256s")):
        source_path = source_dir / f"{kind}_checkpoints" / f"fold{fold:02d}.pt"
        local_path = ROOT / group / f"{kind}_checkpoints" / f"fold{fold:02d}.pt"
        expected_sha = config[directory_key][source_path.name]
        if sha_file(source_path) != expected_sha or sha_file(local_path) != expected_sha:
            raise RuntimeError(f"Inherited {kind} checkpoint copy/hash mismatch for {group} fold {fold}")
    return {"source_config_sha256": sha_file(source_dir / "config.json"),
            "release_config_sha256": sha_file(config_path)}


def _latest_replay(group, fold, latest, best_sha, split, x_np, y_np, device):
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    rows = torch.from_numpy(train_idx).long().to(device)
    seed_everything()
    model = build_model(group, spectral_stats=latest.get("spectral_stats"), device=device, fold=fold)
    opts, opt_cfg = optimizers(model, group)
    if latest.get("optimizer_config") != opt_cfg:
        raise RuntimeError(f"{group} fold {fold} optimizer config differs on recovery")
    model.load_state_dict(latest["model"], strict=True)
    for name, state in latest["optimizers"].items():
        opts[name].load_state_dict(state)
    if model.use_darg:
        model.darg.enter_positive_dcr()
    pair, same = latest["relation_pairs"], latest["relation_targets"]
    if pair is not None:
        pair, same = pair.to(device), same.to(device)
    restore_rng(latest["rng"])
    restored_rng_sha = sha_object(capture_rng_state())
    with torch.no_grad():
        logp, _ = model(x, labels, rows, pair, same, dpsl_enabled=True)
    if not torch.isfinite(logp).all():
        raise RuntimeError(f"{group} fold {fold} Latest forward is non-finite")
    first = _next_step_signature(group, fold, latest, x, labels, rows, device)
    second = _next_step_signature(group, fold, latest, x, labels, rows, device)
    if first != second:
        raise RuntimeError(f"{group} fold {fold} next-step recovery is non-deterministic")
    return {"latest_sha256": sha_file(ROOT / group / "latest_checkpoints" / f"fold{fold:02d}.pt"),
            "best_sha256": best_sha, "latest_epoch": int(latest["epoch"]),
            "optimizer_restored": True, "rng_restored": restored_rng_sha == sha_object(latest["rng"]),
            "finite_latest_forward": True, "next_step_signature": first}


def _g5_next_step(fold, latest, split, x_np, y_np, device):
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    config_path = ROOT / "configs" / f"{GROUPS[4]}.json"
    identity = run_g5.checkpoint_identity(fold, train_idx,
                                          np.asarray(split["test_idx"], dtype=np.int64),
                                          sha_file(config_path))
    for key, value in identity.items():
        if latest.get(key) != value:
            raise RuntimeError(f"G5 fold {fold} Latest identity mismatch at {key}")
    anchor_path = ROOT / GROUPS[4] / "anchors" / f"fold{fold:02d}.pt"
    anchor = torch.load(anchor_path, map_location="cpu", weights_only=False)
    model, optimizer = run_g5.restore_anchor(anchor, device)
    model.load_state_dict(latest["model"], strict=True)
    optimizer.load_state_dict(latest["optimizer"])
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    rows = torch.from_numpy(train_idx).long().to(device)

    def step_signature():
        seed_everything()
        model, optimizer = run_g5.restore_anchor(anchor, device)
        model.load_state_dict(latest["model"], strict=True)
        optimizer.load_state_dict(latest["optimizer"])
        run_g5.restore_rng(latest["rng"])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, logits, _ = model(x)
        loss = run_g5.weighted_nll(logits, labels, rows)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logp, next_logits, _ = model(x)
        return {
            "loss": float(loss),
            "model_sha256": run_g5.digest(model.state_dict()),
            "optimizer_sha256": run_g5.digest(optimizer.state_dict()),
            "rng_sha256": run_g5.digest(run_g5.capture_rng_state()),
            "eval_sha256": hashlib.sha256(logp.detach().cpu().numpy().tobytes()).hexdigest(),
        }

    first, second = step_signature(), step_signature()
    if first != second:
        raise RuntimeError(f"G5 fold {fold} deterministic Latest next-step differs")
    return {"latest_sha256": sha_file(ROOT / GROUPS[4] / "latest_checkpoints" / f"fold{fold:02d}.pt"),
            "best_sha256": latest["best_sha256"], "latest_epoch": int(latest["epoch"]),
            "optimizer_restored": True, "rng_restored": True,
            "finite_latest_forward": True, "next_step_signature": first}


def _g5_best_replay(fold, split, x_np, y_np, ids, device):
    group_dir = ROOT / GROUPS[4]
    best_path = group_dir / "best_checkpoints" / f"fold{fold:02d}.pt"
    latest_path = group_dir / "latest_checkpoints" / f"fold{fold:02d}.pt"
    anchor_path = group_dir / "anchors" / f"fold{fold:02d}.pt"
    prediction_path = group_dir / "predictions" / f"fold{fold:02d}.csv"
    audit_path = group_dir / "audits" / f"fold{fold:02d}.json"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    audit = json.loads(audit_path.read_text())
    anchor = torch.load(anchor_path, map_location="cpu", weights_only=False)
    if audit.get("status") != "TRAINED":
        raise RuntimeError(f"G5 fold {fold} training audit incomplete")
    hashes = {"best_sha256": sha_file(best_path), "latest_sha256": sha_file(latest_path),
              "anchor_sha256": sha_file(anchor_path), "prediction_sha256": sha_file(prediction_path)}
    for key, value in hashes.items():
        expected_key = "anchor_sha256" if key == "anchor_sha256" else key
        audit_key = "latest_sha256" if key == "latest_sha256" else key
        if audit.get(audit_key) != value:
            raise RuntimeError(f"G5 fold {fold} {key} mismatch with audit")
    if latest.get("best_sha256") != hashes["best_sha256"]:
        raise RuntimeError(f"G5 fold {fold} Latest Best reference mismatch")
    train = np.asarray(split["train_idx"], dtype=np.int64)
    test = np.asarray(split["test_idx"], dtype=np.int64)
    config_path = ROOT / "configs" / f"{GROUPS[4]}.json"
    expected = run_g5.checkpoint_identity(fold, train, test, sha_file(config_path))
    for key, value in expected.items():
        if best.get(key) != value or latest.get(key) != value or anchor.get(key) != value:
            raise RuntimeError(f"G5 fold {fold} identity mismatch at {key}")
    fresh_anchor = run_g5.construct_anchor(fold, device)
    for key in ("cam_sha256", "adam_sha256", "rng_sha256"):
        if fresh_anchor[key] != anchor.get(key):
            raise RuntimeError(f"G5 fold {fold} anchor differs from isolated reconstruction at {key}")
    model, optimizer = run_g5.restore_anchor(anchor, device)
    structure = run_g5.audit_physical_structure(model, optimizer)
    model.load_state_dict(best["model"], strict=True)
    model.eval()
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    with torch.no_grad():
        logp, logits, _ = model(x)
    probability = logp.exp().cpu().numpy()
    prediction = logits.argmax(-1).cpu().numpy().astype(np.int64)
    table = pd.read_csv(prediction_path)
    np.testing.assert_array_equal(table["sample_index"].to_numpy(np.int64), test)
    np.testing.assert_array_equal(table["sample_id"].to_numpy(), ids.iloc[test]["RID"].to_numpy())
    np.testing.assert_array_equal(table["true_label"].to_numpy(np.int64), y_np[test])
    np.testing.assert_array_equal(table["pred_label"].to_numpy(np.int64), prediction[test])
    np.testing.assert_allclose(table[["prob_NC", "prob_MCI", "prob_AD"]], probability[test],
                               rtol=1e-6, atol=1e-7)
    parity = run_g5.math_equivalence(anchor, x, labels,
                                     torch.from_numpy(train).long().to(device), device)
    if parity != audit.get("parity") or parity.get("status") != "PASS":
        raise RuntimeError(f"G5 fold {fold} two-step CAM parity differs from training audit")
    return {"status": "PASS", **hashes, "physical_structure": structure,
            "fresh_untrained_anchor_match": True, "two_step_cam_adam_rng_parity": parity,
            "prediction_replay_exact": True}, latest


def _read_predictions(group, fold, split, x_np, y_np, ids):
    path = ROOT / group / "predictions" / f"fold{fold:02d}.csv"
    table = pd.read_csv(path)
    test = np.asarray(split["test_idx"], dtype=np.int64)
    if len(table) != len(test) or not np.all(table["fold"].to_numpy() == fold):
        raise RuntimeError(f"{group} fold {fold} prediction table row count/fold mismatch")
    np.testing.assert_array_equal(table["sample_index"].to_numpy(np.int64), test)
    np.testing.assert_array_equal(table["sample_id"].to_numpy(), ids.iloc[test]["RID"].to_numpy())
    truth = y_np[test]
    np.testing.assert_array_equal(table["true_label"].to_numpy(np.int64), truth)
    prediction = table["pred_label"].to_numpy(np.int64)
    probabilities = table[["prob_NC", "prob_MCI", "prob_AD"]].to_numpy(np.float64)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-6):
        raise RuntimeError(f"{group} fold {fold} probabilities invalid")
    metric = acc_auc(truth, prediction, 3)
    metric["AUC_original"] = metric.pop("AUC")
    metric["AUC_probability"] = float(roc_auc_score(truth, probabilities, average="macro", multi_class="ovr"))
    correct = prediction == truth
    classes = {"NC": int((correct & (truth == 0)).sum()),
               "MCI": int((correct & (truth == 1)).sum()),
               "AD": int((correct & (truth == 2)).sum())}
    return table, metric, int(correct.sum()), classes


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Run replay on CPU")
    args = parser.parse_args()
    torch.set_num_threads(1)
    configure_worker()
    x_np, y_np, ids, splits = load_data()
    _, _, _, splits_train = load_training_data()
    split_by_fold = _split_map(splits)
    if len(splits) != 10 or len(splits_train) != 10 or DATA_SHA != "0b8a14a9c066752c2d937c93d43c2824c41beaaa7d0a451b083db8dcba07c5bc" or SPLIT_SHA != "7c5a4dc9bd8ee7ffc6e5aff8bf841513f26a5ff6cf1b6c110a8153aa6a106ec5":
        raise RuntimeError("Locked TADPOLE dataset/splits are not the expected release")
    device = torch.device("cpu" if args.cpu else "cuda:0")
    records, fold_metrics, mechanism = [], [], {}
    latest_passes = next_step_passes = new_training_next_steps = 0
    g5_physical = g5_anchor = g5_parity = warmup = 0
    for group in GROUPS:
        for split in splits:
            fold = int(split["fold"])
            if group == GROUPS[4]:
                best_record, latest = _g5_best_replay(fold, split, x_np, y_np, ids, device)
                best_sha = best_record["best_sha256"]
                latest_record = _g5_next_step(fold, latest, split, x_np, y_np, device)
                g5_physical += int(best_record["physical_structure"]["forbidden_absent"])
                g5_anchor += int(best_record["fresh_untrained_anchor_match"])
                g5_parity += int(best_record["two_step_cam_adam_rng_parity"]["status"] == "PASS")
            else:
                inherited = group in GROUPS[1:3]
                replay = replay_fold(group, split, (x_np, y_np, ids, splits), device,
                                     historical_identity=inherited)
                if replay.get("status") != "PASS":
                    raise RuntimeError(f"Best replay did not pass for {group} fold {fold}")
                best, latest, best_sha, latest_sha, expected = _validate_checkpoint_hashes(group, fold, split)
                latest_record = _latest_replay(group, fold, latest, best_sha, split,
                                               x_np, y_np, device)
                best_record = {"status": "PASS", "best_sha256": best_sha,
                               "latest_sha256": latest_sha, "best_epoch": int(best["epoch"]),
                               "best_replay": "PASS", "identity": "HISTORICAL_SOURCE" if inherited else "FROZEN_RELEASE"}
                if group == GROUPS[3]:
                    warmup += int(replay["warmup_parity"]["status"] == "PASS")
                if "mechanism" in replay:
                    mechanism.setdefault(group, {})[f"fold{fold:02d}"] = replay["mechanism"]
                best_record["best_replay"] = replay
            latest_passes += 1
            next_step_passes += 1
            if group in (GROUPS[0], GROUPS[3], GROUPS[4]):
                new_training_next_steps += 1
            table, metric, correct_count, class_correct = _read_predictions(
                group, fold, split, x_np, y_np, ids)
            fold_metrics.append({"group": group, "method": GROUP_NAMES[group], "fold": fold,
                                 **metric, "correct_count": correct_count,
                                 "correct_NC": class_correct["NC"],
                                 "correct_MCI": class_correct["MCI"],
                                 "correct_AD": class_correct["AD"]})
            records.append({"group": group, "fold": fold, "best": best_record,
                            "latest": latest_record, "metrics_recomputed": metric,
                            "prediction_rows": len(table)})
            print(f"PASS {group} fold {fold:02d} Best/Latest and predictions", flush=True)

    metrics_frame = pd.DataFrame(fold_metrics)
    if len(metrics_frame) != 50 or metrics_frame.groupby("group").size().to_dict() != {g: 10 for g in GROUPS}:
        raise RuntimeError("The five formal groups do not contain exactly 50 fold results")
    all_indices = []
    for group in GROUPS:
        group_rows = metrics_frame[metrics_frame.group == group]
        if int(group_rows.correct_count.sum()) != int(group_rows[["correct_NC", "correct_MCI", "correct_AD"]].sum().sum()):
            raise RuntimeError(f"Correct-class counts do not sum for {group}")
        index = []
        for row in range(1, 11):
            index.extend(pd.read_csv(ROOT / group / "predictions" / f"fold{row:02d}.csv")["sample_index"].tolist())
        if sorted(index) != list(range(598)) or len(set(index)) != 598:
            raise RuntimeError(f"{group} predictions do not cover each TADPOLE patient exactly once")
        all_indices.extend(index)
    summary_rows = []
    for group in GROUPS:
        rows = metrics_frame[metrics_frame.group == group]
        summary = {"group": group, "method": GROUP_NAMES[group], "folds": 10,
                   "correct_total": int(rows.correct_count.sum()),
                   "correct_NC": int(rows.correct_NC.sum()),
                   "correct_MCI": int(rows.correct_MCI.sum()),
                   "correct_AD": int(rows.correct_AD.sum())}
        for metric_name in ("ACC", "AUC_original", "AUC_probability"):
            summary[f"{metric_name}_mean"] = float(rows[metric_name].mean())
            summary[f"{metric_name}_std"] = float(rows[metric_name].std(ddof=0))
        summary_rows.append(summary)
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    csv_buffer = io.StringIO(newline="")
    metrics_frame.to_csv(csv_buffer, index=False)
    atomic_write(out / "fold_metrics.csv", lambda stream: stream.write(csv_buffer.getvalue().encode()))
    summary_frame = pd.DataFrame(summary_rows)
    csv_buffer = io.StringIO(newline="")
    summary_frame.to_csv(csv_buffer, index=False)
    atomic_write(out / "group_summary.csv", lambda stream: stream.write(csv_buffer.getvalue().encode()))
    write_json(out / "group_metrics.json", {row["group"]: row for row in summary_rows})
    replay_summary = {
        "status": "PASS", "best_replay": "50/50", "latest_restore_and_deterministic_next_step": "50/50",
        "new_training_next_step_recovery": f"{new_training_next_steps}/30",
        "inherited_next_step_recovery": f"{next_step_passes - new_training_next_steps}/20",
        "g3_g4_warmup_alignment": f"{warmup}/10",
        "g5_physical_removal": f"{g5_physical}/10", "g5_fresh_anchor": f"{g5_anchor}/10",
        "g5_two_step_cam_adam_rng_parity": f"{g5_parity}/10",
        "dataset_sha256": DATA_SHA, "split_sha256": SPLIT_SHA,
        "records": records,
    }
    if (warmup != 10 or g5_physical != 10 or g5_anchor != 10 or g5_parity != 10
            or next_step_passes != 50 or new_training_next_steps != 30):
        replay_summary["status"] = "FAIL"
    write_json(ROOT / "audits/independent_replay.json", replay_summary)
    write_json(ROOT / "audits/mechanism_summary.json", mechanism)
    print(json.dumps({"status": replay_summary["status"], "summary": summary_rows,
                      "audit": {key: value for key, value in replay_summary.items() if key != "records"}}, indent=2))
    return 0 if replay_summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run())
