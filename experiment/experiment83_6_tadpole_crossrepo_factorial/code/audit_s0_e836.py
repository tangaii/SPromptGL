"""Independent replay of the tangaii/sp positive-control checkpoints."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SP_ROOT = ROOT / "groups/S0/sp"
SP_CODE = ROOT / "code/sp_runner"
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code/common"))
os.environ.setdefault("E836_SP_RUN_ROOT", str(SP_ROOT))
sys.path.insert(0, str(SP_CODE))
import run_e836
import run as sp_run
from config import DEFAULTS, VARIANTS, MMGPL_CLASS_PROMPTS
from network import GFR, PLN, prompt_features
from utils import ClsLoss, GraphConstructLoss


VARIANT = "mmgpl_class_graph_sup"
VARIANT_ID = "mmgpl_class_prompts:pair_mean"
RUN_ROOT = SP_ROOT / "runs" / VARIANT


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha(value):
    import io
    stream = io.BytesIO()
    torch.save(value, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def capture_rng():
    return sp_run.capture_rng_state()


def restore_rng(state):
    return sp_run.restore_rng_state(state)


def metrics(y_true, prediction, probabilities):
    hard = roc_auc_score(np.eye(3)[y_true], np.eye(3)[prediction], average="macro", multi_class="ovr")
    probability = roc_auc_score(np.eye(3)[y_true], probabilities, average="macro", multi_class="ovr")
    return {"ACC": float(np.mean(y_true == prediction)), "AUC_original": float(hard),
            "AUC_probability": float(probability)}


def make_models(device, features, variant):
    config = SimpleNamespace(**(DEFAULTS | VARIANTS[variant]))
    prompts = MMGPL_CLASS_PROMPTS
    gfr = GFR((145, 148, 3, 36, 24, 9), config.th, config.fusion, config.alpha).to(device)
    pln = PLN((145, 148, 3, 36, 24, 9), config, prompts, device).to(device)
    pln.apply(sp_run.my_weight_init)
    return config, prompts, gfr, pln


def load_data(device):
    config = SimpleNamespace(**(DEFAULTS | VARIANTS[VARIANT]))
    features, labels, dims, info = sp_run.load_data(config, device)
    return config, features, labels, dims, info


def next_step(latest, fold, features, labels, train, device):
    config, prompts, gfr, pln = make_models(device, features, VARIANT)
    gfr.load_state_dict(latest["gfr"], strict=True)
    pln.load_state_dict(latest["pln"], strict=True)
    opt_gfr = torch.optim.Adam(gfr.parameters(), lr=config.lr, weight_decay=config.reg)
    opt_pln = torch.optim.Adam(pln.parameters(), lr=config.lr, weight_decay=config.reg)
    opt_gfr.load_state_dict(copy.deepcopy(latest["optimizer_gfr"]))
    opt_pln.load_state_dict(copy.deepcopy(latest["optimizer_pln"]))
    restore_rng(latest["rng"])
    train_tensor = torch.as_tensor(train, device=device).long()
    labels_tensor = torch.as_tensor(labels, device=device).long()
    target_edges = labels_tensor[train_tensor][:, None].eq(labels_tensor[train_tensor][None, :]).float()
    modalities = list(features.split((145, 148, 3, 36, 24, 9), dim=1))
    raw = torch.cat(modalities, dim=1)
    eye = torch.eye(len(features), device=device, dtype=features.dtype)
    gfr.train(); pln.train(); opt_gfr.zero_grad(set_to_none=True); opt_pln.zero_grad(set_to_none=True)
    fused, graphs, scores = gfr(raw, modalities, eye)
    logp = pln(fused)
    counts = np.bincount(labels_tensor[train_tensor].cpu().numpy(), minlength=3).astype("float32")
    weights = torch.tensor(len(train) / (np.maximum(counts, 1) * 3), dtype=logp.dtype, device=device)
    classification = ClsLoss(logp, labels_tensor, train_tensor, weights)
    structure = sum(GraphConstructLoss(part, adj, config.theta_smooth, config.theta_degree,
                                        config.theta_sparsity)
                     for part, adj in zip(modalities, graphs)) / len(graphs)
    supervision = torch.stack([
        (score[train_tensor[:, None], train_tensor[None, :]] - target_edges).square().mean()
        for score in scores]).mean()
    total = classification + config.lambda_graph * (structure + supervision)
    total.backward(); opt_gfr.step(); opt_pln.step()
    gfr.eval(); pln.eval()
    with torch.no_grad():
        fused, _, _ = gfr(raw, modalities, eye)
        output = pln(fused)
    return {"loss": float(total.detach()), "model_sha256": object_sha({"gfr": gfr.state_dict(), "pln": pln.state_dict()}),
            "optimizer_sha256": object_sha({"gfr": opt_gfr.state_dict(), "pln": opt_pln.state_dict()}),
            "rng_sha256": object_sha(capture_rng()),
            "output_sha256": hashlib.sha256(output.detach().cpu().numpy().tobytes()).hexdigest()}


def main():
    device = torch.device("cuda:0")
    config, features, labels, dims, info = load_data(device)
    labels_np = labels.copy()
    split_path = RUN_ROOT / "splits.json"
    splits = json.loads(split_path.read_text())
    records = []
    prompt_texts = [text for group in MMGPL_CLASS_PROMPTS.values() for text in group]
    prompt_features(prompt_texts, device, "frozen_pretrained")
    for split in splits:
        fold = int(split["fold"])
        fold_dir = RUN_ROOT / f"fold_{fold:02d}"
        best_path, latest_path = fold_dir / "best.pt", fold_dir / "latest.pt"
        pred_path, metric_path = fold_dir / "predictions.csv", fold_dir / "metrics.json"
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        metric_source = json.loads(metric_path.read_text())
        train = np.asarray(split["train"], dtype=np.int64)
        test = np.asarray(split["test"], dtype=np.int64)
        if (best.get("fold") != fold or latest.get("fold") != fold
                or latest.get("variant") != VARIANT_ID
                or latest.get("best_checkpoint_state", {}).get("epoch") != best.get("epoch")):
            raise RuntimeError(f"S0 fold {fold} checkpoint identity differs")
        _, _, gfr, pln = make_models(device, features, VARIANT)
        gfr.load_state_dict(best["gfr"], strict=True); pln.load_state_dict(best["pln"], strict=True)
        modalities = list(features.split(tuple(dims), dim=1)); raw = torch.cat(modalities, dim=1)
        eye = torch.eye(len(features), device=device, dtype=features.dtype)
        gfr.eval(); pln.eval()
        with torch.no_grad():
            output = pln(gfr(raw, modalities, eye)[0])
        prediction = output[test].argmax(1).cpu().numpy().astype(np.int64)
        probabilities = output[test].exp().cpu().numpy()
        observed = metrics(labels_np[test], prediction, probabilities)
        if observed["ACC"] != metric_source["acc"] or abs(observed["AUC_original"] - metric_source["auc_hard"]) > 1e-12:
            raise RuntimeError(f"S0 fold {fold} Best metrics differ")
        table = pd.read_csv(pred_path)
        np.testing.assert_array_equal(table["row_index"].to_numpy(np.int64), test)
        np.testing.assert_array_equal(table["label"].to_numpy(np.int64), labels_np[test])
        np.testing.assert_array_equal(table["prediction"].to_numpy(np.int64), prediction)
        np.testing.assert_allclose(table[["p_NC", "p_MCI", "p_AD"]], probabilities, rtol=1e-6, atol=1e-7)
        restore_rng(latest["rng"])
        first = next_step(latest, fold, features, labels, train, device)
        second = next_step(latest, fold, features, labels, train, device)
        if first != second:
            raise RuntimeError(f"S0 fold {fold} Latest next-step replay differs")
        records.append({"group": "S0", "fold": fold, "best_status": "PASS", "latest_status": "PASS",
                        "best_sha256": file_sha(best_path), "latest_sha256": file_sha(latest_path),
                        "best_epoch_zero_based": int(best["epoch"]), "ACC": observed["ACC"],
                        "AUC_original": observed["AUC_original"], "AUC_probability": observed["AUC_probability"],
                        "correct": int((prediction == labels_np[test]).sum()),
                        "rng_restored_exact": True, "optimizer_state_restored": True,
                        "next_step_deterministic": True, "next_step": first})
        print(f"REPLAY S0 fold {fold:02d} Best=PASS Latest=PASS", flush=True)
    value = {"status": "PASS" if len(records) == 10 else "FAIL", "folds": len(records),
             "best_passes": sum(r["best_status"] == "PASS" for r in records),
             "latest_passes": sum(r["latest_status"] == "PASS" for r in records),
             "next_step_passes": sum(r["next_step_deterministic"] for r in records),
             "records": records, "source_commit": "f4534f8d6cc9288dd1d08f85470c4eb78e64789c"}
    (ROOT / "audits/s0_checkpoint_replay.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value[key] for key in ("status", "folds", "best_passes", "latest_passes",
                                                 "next_step_passes")}, indent=2))


if __name__ == "__main__":
    main()
