"""Independently replay E83.5 Best and Latest artifacts from the server."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import run_e837 as runner
from fidelity_model import FidelityModel
from fold_checkpoint import write_json
from metrics import acc_auc
from run_g5 import PhysicalCAMOnly, weighted_nll


HISTORICAL_ROOT = Path("/model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity")
GROUPS = {
    "H1": ("G1_E28_E0", "G1_E28_E0", runner.OLD_PROMPT_RN50),
    "H2": ("G2_E43_1_E00", "G2_E43_1_E00", runner.PROMPT8),
    "H3": ("G3_E74_4_G2", "G3_E74_4_G2", runner.PROMPT8),
    "H4": ("G4_E82_4_6_10_G1", "G4_E82_4_6_10_G1", runner.OLD_PROMPT_SCMR),
    "H5": ("G5_E43_4_E0_ANCHORED_PHYSICAL_CAM", "G5_E43_4_E0_ANCHORED_PHYSICAL_CAM", None),
}


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_object(value) -> str:
    import io
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_cpu = state.get("torch_cpu", state.get("cpu"))
    torch_cuda = state.get("torch_cuda", state.get("cuda", []))
    torch.set_rng_state(torch_cpu.cpu())
    if torch_cuda:
        torch.cuda.set_rng_state_all([item.cpu() for item in torch_cuda])


def capture_rng():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [item.clone() for item in torch.cuda.get_rng_state_all()],
    }


def build_historical_model(logical, payload, fold, device):
    source_group, base, prompt_path = GROUPS[logical]
    if logical == "H5":
        return PhysicalCAMOnly().to(device)
    runner.historical.seed_everything(0)
    model = FidelityModel(
        runner.historical.DIMS, base, 3, str(prompt_path), payload.get("spectral_stats"),
        824602 * 100 + fold, fold=fold, device=device,
    ).to(device)
    if model.use_darg:
        model.darg.enter_positive_dcr()
    model.load_state_dict(payload["model"], strict=True)
    return model


@torch.no_grad()
def replay_best(logical, fold, split, x, y, ids, y_np, device):
    source_group, _, _ = GROUPS[logical]
    directory = HISTORICAL_ROOT / source_group
    best_path = directory / "best_checkpoints" / f"fold{fold:02d}.pt"
    latest_path = directory / "latest_checkpoints" / f"fold{fold:02d}.pt"
    audit_path = directory / "audits" / f"fold{fold:02d}.json"
    prediction_path = directory / "predictions" / f"fold{fold:02d}.csv"
    saved = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    audit = json.loads(audit_path.read_text())
    train = np.asarray(split["train_idx"], dtype=np.int64)
    test = np.asarray(split["test_idx"], dtype=np.int64)
    for payload in (saved, latest):
        if (payload.get("fold") != fold or payload.get("train_idx") != train.tolist()
                or payload.get("test_idx") != test.tolist()):
            raise RuntimeError(f"Historical fold identity differs for {logical} fold {fold}")
    if latest.get("best_sha256") != sha_file(best_path):
        raise RuntimeError(f"Historical Latest does not point to Best for {logical} fold {fold}")
    if audit.get("status") != "TRAINED" or audit.get("best_epoch") != saved.get("epoch"):
        raise RuntimeError(f"Historical fold audit differs for {logical} fold {fold}")

    model = build_historical_model(logical, saved, fold, device)
    if logical == "H5":
        model.load_state_dict(saved["model"], strict=True)
    model.eval()
    if logical == "H5":
        logits = model(x)[1]
        logp = torch.log_softmax(logits, dim=-1)
    else:
        pair, same = saved.get("relation_pairs"), saved.get("relation_targets")
        if pair is not None:
            pair, same = pair.to(device), same.to(device)
            if not np.isin(pair.cpu().numpy(), train).all():
                raise RuntimeError(f"Historical relation pair includes non-train patients in {logical} fold {fold}")
            np.testing.assert_array_equal(same.cpu().numpy(), y_np[pair[:, 0].cpu().numpy()] ==
                                          y_np[pair[:, 1].cpu().numpy()])
        else:
            same = None
        rows = torch.as_tensor(train, device=device).long()
        logp, _ = model(x, y, rows, pair, same, dpsl_enabled=True)
    prediction = logp[test].argmax(1).cpu().numpy().astype(np.int64)
    probabilities = logp[test].exp().cpu().numpy()
    metrics = acc_auc(y_np[test], prediction, 3)
    probability_auc = float(roc_auc_score(
        np.eye(3, dtype=np.float64)[y_np[test]], probabilities, average="macro", multi_class="ovr"))
    pred_sha = hashlib.sha256(prediction.tobytes()).hexdigest()
    expected = pd.read_csv(prediction_path)
    np.testing.assert_array_equal(expected["sample_index"].to_numpy(np.int64), test)
    np.testing.assert_array_equal(expected["sample_id"].to_numpy(), ids.iloc[test]["RID"].to_numpy())
    np.testing.assert_array_equal(expected["true_label"].to_numpy(np.int64), y_np[test])
    np.testing.assert_array_equal(expected["pred_label"].to_numpy(np.int64), prediction)
    np.testing.assert_allclose(expected[["prob_NC", "prob_MCI", "prob_AD"]],
                               probabilities, rtol=1e-6, atol=1e-7)
    if logical == "H5":
        if sha_file(prediction_path) != audit.get("prediction_sha256"):
            raise RuntimeError(f"Historical H5 prediction artifact hash differs in fold {fold}")
    elif pred_sha != (saved.get("prediction_sha256") or audit.get("prediction_sha256")):
        raise RuntimeError(f"Historical prediction hash differs for {logical} fold {fold}")
    if metrics["ACC"] != audit["ACC"] or abs(metrics["AUC"] - audit.get("AUC", audit.get("AUC_original", metrics["AUC"]))) > 1e-12:
        raise RuntimeError(f"Historical hard-label metrics differ for {logical} fold {fold}")
    return {
        "group": logical, "fold": fold, "best_status": "PASS",
        "best_path": str(best_path), "best_sha256": sha_file(best_path),
        "latest_path": str(latest_path), "latest_sha256": sha_file(latest_path),
        "source_experiment": saved.get("experiment"), "best_epoch": int(saved["epoch"]),
        "ACC": metrics["ACC"], "AUC_original": metrics["AUC"],
        "AUC_probability": probability_auc, "correct": int((prediction == y_np[test]).sum()),
        "prediction_sha256": pred_sha, "prediction_and_probability_match_source": True,
    }


def next_step_historical(logical, latest, fold, x, y, train_idx, device):
    source_group, base, prompt_path = GROUPS[logical]
    runner.historical.seed_everything(0)
    train = torch.as_tensor(train_idx, device=device).long()
    if logical == "H5":
        model = PhysicalCAMOnly().to(device)
        model.load_state_dict(latest["model"], strict=True)
        optimizer = torch.optim.Adam(model.cam.parameters(), lr=0.012, weight_decay=0.0036)
        optimizer.load_state_dict(copy.deepcopy(latest["optimizer"]))
        restore_rng(latest["rng"])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, logits, _ = model(x)
        loss = weighted_nll(logits, y, train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model(x)[0]
        return {
            "loss": float(loss), "model_sha256": sha_object(model.state_dict()),
            "optimizer_sha256": sha_object(optimizer.state_dict()),
            "rng_sha256": sha_object(capture_rng()),
            "output_sha256": hashlib.sha256(output.cpu().numpy().tobytes()).hexdigest(),
        }

    model = FidelityModel(runner.historical.DIMS, base, 3, str(prompt_path),
                          latest.get("spectral_stats"), 824602 * 100 + fold,
                          fold=fold, device=device).to(device)
    if model.use_darg:
        model.darg.enter_positive_dcr()
    model.load_state_dict(latest["model"], strict=True)
    optimizers, config = runner.make_optimizers(model, {"low_base_weight_decay": False})
    if config != latest["optimizer_config"]:
        raise RuntimeError(f"Historical optimizer identity mismatch for {logical} fold {fold}")
    for name, state in latest["optimizers"].items():
        optimizers[name].load_state_dict(copy.deepcopy(state))
    pair, same = latest.get("relation_pairs"), latest.get("relation_targets")
    if pair is not None:
        pair, same = pair.to(device), same.to(device)
    restore_rng(latest["rng"])
    model.train()
    runner.zero_optimizers(optimizers)
    _, aux = model(x, y, train, pair, same, dpsl_enabled=True)
    _, total = runner.historical.historical_batch_loss(aux, y, train, model.use_darg)
    if model.dpsl is not None:
        runner.historical.scmr_optimizer_step(model, optimizers, total, aux["scmr_objectives"])
    else:
        total.backward()
        runner.step_optimizers(optimizers)
    model.eval()
    with torch.no_grad():
        output, eval_aux = model(x, y, train, pair, same, dpsl_enabled=True)
    if model.dpsl is not None:
        model.dpsl.advance_ema(eval_aux["cam"])
    return {
        "loss": float(total), "model_sha256": sha_object(model.state_dict()),
        "optimizer_sha256": sha_object({name: value.state_dict() for name, value in optimizers.items()}),
        "rng_sha256": sha_object(capture_rng()),
        "output_sha256": hashlib.sha256(output.detach().cpu().numpy().tobytes()).hexdigest(),
        "dpsl_ema_updates": int(model.dpsl.ema_updates) if model.dpsl is not None else 0,
    }


def replay_latest(logical, fold, split, x, y, device):
    source_group, _, _ = GROUPS[logical]
    latest_path = HISTORICAL_ROOT / source_group / "latest_checkpoints" / f"fold{fold:02d}.pt"
    best_path = HISTORICAL_ROOT / source_group / "best_checkpoints" / f"fold{fold:02d}.pt"
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    train = np.asarray(split["train_idx"], dtype=np.int64)
    test = np.asarray(split["test_idx"], dtype=np.int64)
    if (latest.get("best_sha256") != sha_file(best_path) or latest.get("fold") != fold
            or latest.get("train_idx") != train.tolist() or latest.get("test_idx") != test.tolist()):
        raise RuntimeError(f"Historical Latest identity mismatch for {logical} fold {fold}")
    if logical != "H5":
        model = build_historical_model(logical, latest, fold, device)
        optimizer, optimizer_config = runner.make_optimizers(model, {"low_base_weight_decay": False})
        if optimizer_config != latest["optimizer_config"]:
            raise RuntimeError(f"Historical optimizer config differs for {logical} fold {fold}")
        for name, state in latest["optimizers"].items():
            optimizer[name].load_state_dict(copy.deepcopy(state))
    else:
        model = PhysicalCAMOnly().to(device)
        model.load_state_dict(latest["model"], strict=True)
        optimizer = torch.optim.Adam(model.cam.parameters(), lr=0.012, weight_decay=0.0036)
        optimizer.load_state_dict(copy.deepcopy(latest["optimizer"]))
    restore_rng(latest["rng"])
    rng_restored = sha_object(capture_rng()) == sha_object(latest["rng"])
    first = next_step_historical(logical, latest, fold, x, y, train, device)
    second = next_step_historical(logical, latest, fold, x, y, train, device)
    if not rng_restored or first != second:
        raise RuntimeError(f"Latest deterministic next-step replay failed for {logical} fold {fold}")
    return {"group": logical, "fold": fold, "latest_status": "PASS",
            "rng_restored_exact": rng_restored, "optimizer_state_restored": True,
            "next_step_deterministic": True, "next_step": first}


def main():
    runner.configure_worker()
    x_np, y_np, ids, splits = runner.load_data()
    x = torch.from_numpy(x_np).cuda(0)
    y = torch.from_numpy(y_np).long().cuda(0)
    records = []
    for logical in GROUPS:
        for split in splits:
            fold = int(split["fold"])
            best = replay_best(logical, fold, split, x, y, ids, y_np, x.device)
            latest = replay_latest(logical, fold, split, x, y, x.device)
            records.append({**best, **latest})
            print(f"REPLAY {logical} fold {fold:02d} Best=PASS Latest=PASS", flush=True)
    value = {"status": "PASS" if len(records) == 50 else "FAIL", "folds": len(records),
             "best_passes": sum(row["best_status"] == "PASS" for row in records),
             "latest_passes": sum(row["latest_status"] == "PASS" for row in records),
             "next_step_passes": sum(row["next_step_deterministic"] for row in records),
             "source_checkpoint_root": str(HISTORICAL_ROOT), "records": records}
    write_json(runner.ROOT / "audits/base_checkpoint_replay.json", value)
    print(json.dumps({key: value[key] for key in ("status", "folds", "best_passes", "latest_passes",
                                                 "next_step_passes")}, indent=2))


if __name__ == "__main__":
    main()
