"""Independently replay all E83.7 Best and Latest fold artifacts."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import run_e837 as runner
from fold_checkpoint import write_json
from metrics import acc_auc


GROUPS = [f"T{number}" for number in range(1, 7)]


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha(value) -> str:
    import io
    stream = io.BytesIO()
    torch.save(value, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def capture_rng():
    return runner.historical.capture_rng_state()


def restore_rng(state):
    runner.historical.restore_rng(state)


def hard_metrics(y_true, prediction, probabilities):
    metric = acc_auc(y_true, prediction, 3)
    metric["AUC_probability"] = float(roc_auc_score(
        np.eye(3, dtype=np.float64)[y_true], probabilities,
        average="macro", multi_class="ovr"))
    return metric


def load_checkpoint(group, fold, kind):
    path = runner.ROOT / "groups" / group / f"{kind}_checkpoints" / f"fold{fold:02d}.pt"
    return path, torch.load(path, map_location="cpu", weights_only=False)


def validate_identity(group, fold, split, payload, seal):
    config = runner.group_config(group)
    expected = runner.identity(group, fold, np.asarray(split["train_idx"], dtype=np.int64),
                               np.asarray(split["test_idx"], dtype=np.int64), config)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"{group} fold {fold} {key} differs from the sealed identity")
    if payload.get("implementation_sha256") != seal["implementation_sha256"]:
        raise RuntimeError(f"{group} fold {fold} implementation identity differs from seal")


@torch.no_grad()
def replay_best(group, fold, split, x, y, y_np, ids, seal):
    path, saved = load_checkpoint(group, fold, "best")
    latest_path, latest = load_checkpoint(group, fold, "latest")
    validate_identity(group, fold, split, saved, seal)
    validate_identity(group, fold, split, latest, seal)
    if latest.get("best_sha256") != file_sha(path):
        raise RuntimeError(f"{group} fold {fold} Latest does not point to Best")
    audit_path = runner.ROOT / "groups" / group / "audits" / f"fold{fold:02d}.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "TRAINED" or audit.get("checkpoint_sha256") != file_sha(path):
        raise RuntimeError(f"{group} fold {fold} audit/checkpoint identity differs")
    config = runner.group_config(group)
    model = runner.build_model(config, fold, x.device, spectral_stats=saved.get("spectral_stats"))
    if model.use_darg:
        model.darg.enter_positive_dcr()
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    train = torch.as_tensor(split["train_idx"], device=x.device).long()
    test = torch.as_tensor(split["test_idx"], device=x.device).long()
    pair, same = saved.get("relation_pairs"), saved.get("relation_targets")
    if pair is not None:
        pair, same = pair.to(x.device), same.to(x.device)
    output, aux = model(x, y, train, pair, same, dpsl_enabled=True)
    prediction = output[test].argmax(1).cpu().numpy().astype(np.int64)
    probabilities = output[test].exp().cpu().numpy()
    test_np = test.cpu().numpy()
    metric = hard_metrics(y_np[test_np], prediction, probabilities)
    if metric["ACC"] != audit["ACC"] or abs(metric["AUC"] - audit["AUC_original"]) > 1e-12:
        raise RuntimeError(f"{group} fold {fold} Best metrics differ from training audit")
    prediction_path = runner.ROOT / "groups" / group / "predictions" / f"fold{fold:02d}.csv"
    table = pd.read_csv(prediction_path)
    np.testing.assert_array_equal(table["sample_index"].to_numpy(np.int64), test_np)
    np.testing.assert_array_equal(table["sample_id"].to_numpy(), ids.iloc[test_np]["RID"].to_numpy())
    np.testing.assert_array_equal(table["true_label"].to_numpy(np.int64), y_np[test_np])
    np.testing.assert_array_equal(table["pred_label"].to_numpy(np.int64), prediction)
    np.testing.assert_allclose(table[["prob_NC", "prob_MCI", "prob_AD"]], probabilities,
                               rtol=1e-6, atol=1e-7)
    if hashlib.sha256(prediction.tobytes()).hexdigest() != audit["prediction_sha256"]:
        raise RuntimeError(f"{group} fold {fold} prediction array hash differs")
    if config.get("graph_supervision_schedule") == "warmup_only":
        if audit.get("graph_supervision_final") != 0.0 or audit.get("formal_graph_supervision_enabled"):
            raise RuntimeError(f"{group} fold {fold} graph supervision remained active after warmup")
    return {
        "group": group, "fold": fold, "best_status": "PASS", "best_sha256": file_sha(path),
        "latest_sha256": file_sha(latest_path), "best_epoch": int(saved["epoch"]),
        "ACC": metric["ACC"], "AUC_original": metric["AUC"],
        "AUC_probability": metric["AUC_probability"],
        "correct": int((prediction == y_np[test_np]).sum()),
        "graph_supervision_final": audit.get("graph_supervision_final"),
        "dpsl_shared_cap_configured": audit.get("dpsl_shared_cap_configured"),
        "dpsl_shared_gradient_ratio_max": audit.get("dpsl_shared_gradient_ratio_max"),
        "weight_decay_schedule": audit.get("weight_decay_schedule"),
    }


def next_step(group, fold, latest, x, y, train, pair, same, config):
    model = runner.build_model(config, fold, x.device, spectral_stats=latest.get("spectral_stats"))
    if model.use_darg:
        model.darg.enter_positive_dcr()
    optimizers, optimizer_config = runner.make_optimizers(model, config, formal=True)
    if optimizer_config != latest["optimizer_config"]:
        raise RuntimeError(f"{group} fold {fold} Latest optimizer config differs")
    model.load_state_dict(latest["model"], strict=True)
    for name, state in latest["optimizers"].items():
        optimizers[name].load_state_dict(copy.deepcopy(state))
    restore_rng(latest["rng"])
    model.train()
    runner.zero_optimizers(optimizers)
    _, aux = model(x, y, train, pair, same, dpsl_enabled=True)
    formal_graph = (bool(config.get("pair_mean_graph_supervision"))
                    and config.get("graph_supervision_schedule", "none") != "warmup_only")
    _, historical_total, _, _ = runner.historical_loss(aux, y, train, model, formal_graph)
    if model.dpsl is not None:
        runner.dpsl_optimizer_step(model, optimizers, historical_total, aux["scmr_objectives"],
                                   float(config.get("dpsl_shared_cap", 0.001)))
    else:
        historical_total.backward()
        runner.step_optimizers(optimizers)
    model.eval()
    with torch.no_grad():
        output, eval_aux = model(x, y, train, pair, same, dpsl_enabled=True)
    if model.dpsl is not None:
        model.dpsl.advance_ema(eval_aux["cam"])
    return {
        "loss": float(historical_total.detach()), "model_sha256": object_sha(model.state_dict()),
        "optimizer_sha256": object_sha({key: value.state_dict() for key, value in optimizers.items()}),
        "rng_sha256": object_sha(capture_rng()),
        "output_sha256": hashlib.sha256(output.detach().cpu().numpy().tobytes()).hexdigest(),
        "ema_updates": int(model.dpsl.ema_updates) if model.dpsl is not None else 0,
    }


def replay_latest(group, fold, split, x, y):
    path, latest = load_checkpoint(group, fold, "latest")
    config = runner.group_config(group)
    train = torch.as_tensor(split["train_idx"], device=x.device).long()
    pair, same = latest.get("relation_pairs"), latest.get("relation_targets")
    if pair is not None:
        pair, same = pair.to(x.device), same.to(x.device)
    restore_rng(latest["rng"])
    exact_rng = object_sha(capture_rng()) == object_sha(latest["rng"])
    first = next_step(group, fold, latest, x, y, train, pair, same, config)
    second = next_step(group, fold, latest, x, y, train, pair, same, config)
    if not exact_rng or first != second:
        raise RuntimeError(f"{group} fold {fold} Latest next-step replay differs")
    return {"latest_status": "PASS", "latest_sha256": file_sha(path),
            "rng_restored_exact": exact_rng, "optimizer_state_restored": True,
            "next_step_deterministic": True, "next_step": first}


def main():
    runner.configure_worker()
    x_np, y_np, ids, splits = runner.load_data()
    x, y = torch.from_numpy(x_np).cuda(0), torch.from_numpy(y_np).long().cuda(0)
    seal = json.loads((runner.ROOT / "audits/training_seal.json").read_text())
    records = []
    for group in GROUPS:
        for split in splits:
            fold = int(split["fold"])
            best = replay_best(group, fold, split, x, y, y_np, ids, seal)
            latest = replay_latest(group, fold, split, x, y)
            records.append({**best, **latest})
            print(f"REPLAY {group} fold {fold:02d} Best=PASS Latest=PASS", flush=True)
    value = {
        "experiment": "Experiment 83.7", "status": "PASS" if len(records) == 60 else "FAIL",
        "folds": len(records),
        "best_passes": sum(item["best_status"] == "PASS" for item in records),
        "latest_passes": sum(item["latest_status"] == "PASS" for item in records),
        "next_step_passes": sum(item["next_step_deterministic"] for item in records),
        "records": records,
    }
    write_json(runner.ROOT / "audits/new_checkpoint_replay.json", value)
    print(json.dumps({key: value[key] for key in
                      ("status", "folds", "best_passes", "latest_passes", "next_step_passes")}, indent=2))
    if value["status"] != "PASS":
        raise RuntimeError("E83.7 new-group checkpoint replay failed")


if __name__ == "__main__":
    main()
