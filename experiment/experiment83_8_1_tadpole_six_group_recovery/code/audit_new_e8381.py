"""Independently replay all E83.8.1 Best and Latest fold artifacts."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import run_e8381 as runner
from metrics import acc_auc


GROUPS = [f"X{number}" for number in range(1, 7)]
HISTORY_ROOT = Path("/model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha(value) -> str:
    return runner.sha_object(value)


def hard_metrics(y_true, prediction, probabilities):
    metric = acc_auc(y_true, prediction, 3)
    metric["AUC_probability"] = float(roc_auc_score(
        np.eye(3, dtype=np.float64)[y_true], probabilities,
        average="macro", multi_class="ovr"))
    return metric


def checkpoint(group, fold, kind):
    path = runner.ROOT / "groups" / group / f"{kind}_checkpoints" / f"fold{fold:02d}.pt"
    return path, torch.load(path, map_location="cpu", weights_only=False)


def validate_identity(group, fold, split, payload, seal):
    config = runner.group_config(group)
    expected = runner.identity(group, fold, np.asarray(split["train_idx"], dtype=np.int64),
                               np.asarray(split["test_idx"], dtype=np.int64), config)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"{group} fold {fold} sealed identity differs at {key}")
    if payload.get("implementation_sha256") != seal["implementation_sha256"]:
        raise RuntimeError(f"{group} fold {fold} implementation seal differs")


@torch.no_grad()
def replay_best(group, fold, split, x, y, y_np, ids, seal):
    best_path, best = checkpoint(group, fold, "best")
    latest_path, latest = checkpoint(group, fold, "latest")
    validate_identity(group, fold, split, best, seal)
    validate_identity(group, fold, split, latest, seal)
    if latest.get("best_sha256") != file_sha(best_path):
        raise RuntimeError(f"{group} fold {fold} Latest does not reference its Best checkpoint")
    audit_path = runner.ROOT / "groups" / group / "audits" / f"fold{fold:02d}.json"
    audit = json.loads(audit_path.read_text())
    if (audit.get("status") != "TRAINED" or audit.get("checkpoint_sha256") != file_sha(best_path)
            or audit.get("latest_sha256") != file_sha(latest_path)):
        raise RuntimeError(f"{group} fold {fold} training audit and checkpoints differ")
    config = runner.group_config(group)
    model = runner.build_model(config, fold, x.device, spectral_stats=best.get("spectral_stats"))
    if model.use_darg:
        model.darg.enter_positive_dcr()
    model.load_state_dict(best["model"], strict=True)
    model.eval()
    train = torch.as_tensor(split["train_idx"], device=x.device).long()
    test = torch.as_tensor(split["test_idx"], device=x.device).long()
    pair, same = best.get("relation_pairs"), best.get("relation_targets")
    if pair is not None:
        pair, same = pair.to(x.device), same.to(x.device)
    output, _ = model(x, y, train, pair, same, dpsl_enabled=True)
    prediction = output[test].argmax(1).cpu().numpy().astype(np.int64)
    probabilities = output[test].exp().cpu().numpy()
    test_np = test.cpu().numpy()
    metric = hard_metrics(y_np[test_np], prediction, probabilities)
    if (metric["ACC"] != audit["ACC"]
            or abs(metric["AUC"] - audit["AUC_original"]) > 1e-12
            or abs(metric["AUC_probability"] - audit["AUC_probability"]) > 1e-12):
        raise RuntimeError(f"{group} fold {fold} independently replayed Best metrics differ")
    table_path = runner.ROOT / "groups" / group / "predictions" / f"fold{fold:02d}.csv"
    table = pd.read_csv(table_path)
    np.testing.assert_array_equal(table.sample_index.to_numpy(np.int64), test_np)
    np.testing.assert_array_equal(table.sample_id.to_numpy(), ids.iloc[test_np].RID.to_numpy())
    np.testing.assert_array_equal(table.true_label.to_numpy(np.int64), y_np[test_np])
    np.testing.assert_array_equal(table.pred_label.to_numpy(np.int64), prediction)
    np.testing.assert_allclose(table[["prob_NC", "prob_MCI", "prob_AD"]],
                               probabilities, rtol=1e-6, atol=1e-7)
    if hashlib.sha256(prediction.tobytes()).hexdigest() != audit["prediction_sha256"]:
        raise RuntimeError(f"{group} fold {fold} independent prediction hash differs")
    if np.intersect1d(split["train_idx"], split["test_idx"]).size:
        raise RuntimeError(f"{group} fold {fold} train/test indices overlap")
    return {
        "group": group, "fold": fold, "best_status": "PASS", "best_path": str(best_path),
        "best_sha256": file_sha(best_path), "latest_path": str(latest_path),
        "latest_sha256": file_sha(latest_path), "best_epoch": int(best["epoch"]),
        "ACC": metric["ACC"], "AUC_original": metric["AUC"],
        "AUC_probability": metric["AUC_probability"],
        "correct": int((prediction == y_np[test_np]).sum()),
        "supervision_kind": audit.get("supervision_kind"),
        "graph_supervision_final": audit.get("graph_supervision_final"),
        "dpsl_residual_max": audit.get("dpsl_residual_max"),
        "dpsl_shared_cap_configured": audit.get("dpsl_shared_cap_configured"),
        "dpsl_shared_gradient_ratio_max": audit.get("dpsl_shared_gradient_ratio_max"),
    }


def next_step(group, fold, latest, x, y, train, pair, same, config):
    model = runner.build_model(config, fold, x.device, spectral_stats=latest.get("spectral_stats"))
    if model.use_darg:
        model.darg.enter_positive_dcr()
    optimizers, optimizer_config = runner.make_optimizers(model, config, formal=True)
    if optimizer_config != latest["optimizer_config"]:
        raise RuntimeError(f"{group} fold {fold} Latest optimizer contract differs")
    model.load_state_dict(latest["model"], strict=True)
    for name, state in latest["optimizers"].items():
        optimizers[name].load_state_dict(copy.deepcopy(state))
    runner.historical.restore_rng(latest["rng"])
    model.train()
    runner.zero_optimizers(optimizers)
    _, aux = model(x, y, train, pair, same, dpsl_enabled=True)
    epoch = int(latest["epoch"]) + 1
    kind = runner.active_supervision_kind(config, epoch)
    _, total, _, _ = runner.historical_loss(aux, y, train, model, kind)
    if model.dpsl is not None:
        runner.dpsl_optimizer_step(model, optimizers, total, aux["scmr_objectives"],
                                   float(config.get("dpsl_shared_cap", 0.001)))
    else:
        total.backward()
        runner.step_optimizers(optimizers)
    model.eval()
    with torch.no_grad():
        output, eval_aux = model(x, y, train, pair, same, dpsl_enabled=True)
    if model.dpsl is not None:
        model.dpsl.advance_ema(eval_aux["cam"])
    return {
        "next_epoch": epoch, "next_supervision_kind": kind,
        "loss": float(total.detach()), "model_sha256": object_sha(model.state_dict()),
        "optimizer_sha256": object_sha({key: value.state_dict() for key, value in optimizers.items()}),
        "rng_sha256": object_sha(runner.historical.capture_rng_state()),
        "output_sha256": hashlib.sha256(output.detach().cpu().numpy().tobytes()).hexdigest(),
        "ema_updates": int(model.dpsl.ema_updates) if model.dpsl is not None else 0,
    }


def replay_latest(group, fold, split, x, y):
    path, latest = checkpoint(group, fold, "latest")
    config = runner.group_config(group)
    train = torch.as_tensor(split["train_idx"], device=x.device).long()
    pair, same = latest.get("relation_pairs"), latest.get("relation_targets")
    if pair is not None:
        pair, same = pair.to(x.device), same.to(x.device)
    runner.historical.restore_rng(latest["rng"])
    exact_rng = object_sha(runner.historical.capture_rng_state()) == object_sha(latest["rng"])
    runner.historical.restore_rng(latest["rng"])
    first = next_step(group, fold, latest, x, y, train, pair, same, config)
    runner.historical.restore_rng(latest["rng"])
    second = next_step(group, fold, latest, x, y, train, pair, same, config)
    if not exact_rng or first != second:
        raise RuntimeError(f"{group} fold {fold} Latest next-step replay is not deterministic")
    return {"latest_status": "PASS", "latest_sha256": file_sha(path),
            "rng_restored_exact": exact_rng, "optimizer_state_restored": True,
            "next_step_deterministic": True, "next_step": first}


def audit_warmup_pairing():
    rows = []
    splits = {int(item["fold"]): item for item in runner.load_data()[3]}
    y_np = runner.load_data()[1]
    for fold in range(1, 11):
        configs = {group: runner.group_config(group) for group in ("X3", "X4", "X5", "X6")}
        payloads = {group: checkpoint(group, fold, "best")[1] for group in configs}
        warmups = {group: payload["warmup"] for group, payload in payloads.items()}
        def exact_fields(left, right):
            fields = ("initial_model_sha256", "initial_optimizer_sha256", "initial_rng_sha256",
                      "base_state_sha256", "optimizer_state_sha256", "rng_state_sha256",
                      "pair_sha256", "train_index_sha256", "train_labels_sha256")
            differences = [field for field in fields if left.get(field) != right.get(field)]
            if differences or left.get("history") != right.get("history"):
                raise RuntimeError(f"Fold {fold} paired warmups differ at {differences or ['history']}")
        def shared_fields(left, right):
            fields = ("initial_shared_darg_cam_gfr_sha256", "initial_shared_optimizer_sha256",
                      "initial_rng_sha256", "shared_darg_cam_gfr_state_sha256",
                      "shared_optimizer_state_sha256", "rng_state_sha256", "pair_sha256",
                      "train_index_sha256", "train_labels_sha256")
            differences = [field for field in fields if left.get(field) != right.get(field)]
            if differences or left.get("history") != right.get("history"):
                raise RuntimeError(f"Fold {fold} shared DARG/CAM/GFR warmups differ at {differences or ['history']}")

        shared_fields(warmups["X3"], warmups["X4"])
        exact_fields(warmups["X5"], warmups["X6"])
        for group in ("X3", "X4"):
            values = [float(item["graph_supervision"]) for item in warmups[group]["history"]]
            if len(values) != 10 or any(value <= 0 for value in values):
                raise RuntimeError(f"{group} fold {fold} ALoss was not active throughout warmup")
        for group in ("X5", "X6"):
            values = [float(item.get("graph_supervision", 0.0)) for item in warmups[group]["history"]]
            if len(values) != 10 or any(value != 0 for value in values):
                raise RuntimeError(f"{group} fold {fold} pair-mean was active during warmup")
        train_idx = np.asarray(splits[fold]["train_idx"], dtype=np.int64)
        h4_match = {}
        for group in ("X5", "X6"):
            h4_match[group] = runner.compare_warmup(
                group, fold, configs[group], warmups[group], payloads[group]["relation_pairs"],
                payloads[group]["relation_targets"], train_idx, y_np)
            if h4_match[group]["status"] == "FAIL":
                raise RuntimeError(f"{group} fold {fold} does not match archived H4 warmup")
        rows.append({"fold": fold, "X3_X4_exact": True, "X5_X6_exact": True,
                     "X3_X4_aloss_nonzero_epochs": 10, "X5_X6_pairmean_nonzero_warmup_epochs": 0,
                     "H4_match_status": {group: result["status"] for group, result in h4_match.items()},
                     "X3_initial_sha256": warmups["X3"]["initial_model_sha256"],
                     "X4_initial_sha256": warmups["X4"]["initial_model_sha256"],
                     "X5_initial_sha256": warmups["X5"]["initial_model_sha256"],
                     "X6_initial_sha256": warmups["X6"]["initial_model_sha256"]})
    return {"status": "PASS", "folds": len(rows), "paired_passes": len(rows), "records": rows,
            "historical_initial_state_note": "E83.5 H4 does not store initial model/optimizer/RNG hashes; archived warmup matches are PARTIAL_MATCH where reported."}


def main():
    runner.configure_worker()
    x_np, y_np, ids, splits = runner.load_data()
    x, y = torch.as_tensor(x_np, device="cuda:0"), torch.as_tensor(y_np, device="cuda:0").long()
    seal_path = runner.ROOT / "audits/training_seal.json"
    seal = json.loads(seal_path.read_text())
    records = []
    for group in GROUPS:
        for split in splits:
            fold = int(split["fold"])
            best = replay_best(group, fold, split, x, y, y_np, ids, seal)
            latest = replay_latest(group, fold, split, x, y)
            records.append({**best, **latest})
            print(f"REPLAY {group} fold {fold:02d} Best=PASS Latest=PASS", flush=True)
    warmup = audit_warmup_pairing()
    value = {
        "experiment": "Experiment 83.8.1", "status": "PASS" if len(records) == 60 else "INCOMPLETE",
        "folds": len(records), "best_passes": sum(row["best_status"] == "PASS" for row in records),
        "latest_passes": sum(row["latest_status"] == "PASS" for row in records),
        "next_step_passes": sum(row["next_step_deterministic"] for row in records),
        "groups": {group: sum(row["group"] == group for row in records) for group in GROUPS},
        "warmup_pairing": warmup, "records": records,
    }
    runner.write_json(runner.ROOT / "audits/checkpoint_replay.json", value)
    runner.write_json(runner.ROOT / "audits/warmup_pairing.json", warmup)
    print(json.dumps({key: value[key] for key in
                      ("status", "folds", "best_passes", "latest_passes", "next_step_passes")}, indent=2))
    if value["status"] != "PASS" or warmup["paired_passes"] != 10:
        raise RuntimeError("E83.8.1 independent checkpoint or warmup replay is incomplete")


if __name__ == "__main__":
    main()
