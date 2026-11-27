#!/usr/bin/env python3
"""Rebuild E43.4-E0 from the verified E43.1 anchors and extract CAM144.

This module deliberately keeps the original PhysicalCAMOnly implementation in
inputs/cam_only_source. It does not load a complete SPromptGL model or use a
logit, probability, attention map, or classification feature as CAM144.
"""
from __future__ import annotations

import csv
import copy
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

ROOT = Path("/data/adodas/SPromptGL-main")
EXP = Path("/model/spromptgl/experiment/experiment_dresp_tsne_five_models_label_guided")
SOURCE = EXP / "inputs/cam_only_source"
OUT = EXP / "inputs/cam_only_features"
AUDIT = EXP / "results/cam_only_rebuild"
ANCHORS = Path(
    "/model/spromptgl/experiment/experiment_darg_jsl/code/vendor/repo/"
    "experiment/experiment72_mgdss/.runtime/vendor/root/experiment/"
    "experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
)
EXPECTED = SOURCE / "historical_fold_metrics.csv"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def state_sha(value, torch_module) -> str:
    h = hashlib.sha256()

    def add(item):
        if isinstance(item, torch_module.Tensor):
            arr = item.detach().cpu().contiguous().numpy()
            h.update(repr((str(arr.dtype), arr.shape)).encode())
            h.update(arr.tobytes())
        elif isinstance(item, np.ndarray):
            h.update(repr((str(item.dtype), item.shape)).encode())
            h.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                h.update(repr(key).encode())
                add(item[key])
        elif isinstance(item, (list, tuple)):
            for part in item:
                add(part)
        else:
            h.update(repr(item).encode())

    add(value)
    return h.hexdigest()


def rng_snapshot(torch_module):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch_module.get_rng_state(),
        "cuda": torch_module.cuda.get_rng_state_all(),
    }


def restore_rng(snapshot, torch_module):
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch_module.set_rng_state(snapshot["cpu"].cpu())
    torch_module.cuda.set_rng_state_all([state.cpu() for state in snapshot["cuda"]])


def rng_sha(snapshot) -> str:
    h = hashlib.sha256(pickle.dumps((snapshot["python"], snapshot["numpy"]), protocol=4))
    for state in [snapshot["cpu"], *snapshot["cuda"]]:
        h.update(state.cpu().numpy().tobytes())
    return h.hexdigest()


def load_context():
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CAM_ONLY_CHECKPOINT: CUDA is required by the archived VariLengthInputLayer")
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sys.path.insert(0, str(SOURCE))
    import layers
    from integration import build

    return torch, layers, build


def load_data():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    frame = pd.read_csv(ROOT / "data/ABIDE/processed_standard_data.csv")
    x = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = frame.iloc[:, -1].to_numpy(dtype=np.int64) - 1
    if x.shape != (871, 320) or not np.isfinite(x).all():
        raise ValueError(f"unexpected CAM-only source data shape or finiteness: {x.shape}")
    if np.bincount(y).tolist() != [403, 468]:
        raise ValueError("unexpected ASD/TD counts")
    folds = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y))
    return x, y, folds


def split_sha(train, test) -> str:
    return sha256_bytes(train.astype(np.int64).tobytes() + test.astype(np.int64).tobytes())


def load_expected():
    with EXPECTED.open(newline="") as handle:
        return {
            int(row["fold"]): row
            for row in csv.DictReader(handle)
            if row.get("group") in {"0", "E0_ANCHORED_PHYSICAL_CAM"}
            or row.get("group_id") == "0"
        }


def load_anchor(fold: int, torch_module):
    path = ANCHORS / f"fold{fold:02}.pt"
    if not path.is_file():
        raise RuntimeError(f"BLOCKED_CAM_ONLY_CHECKPOINT: missing anchor {path}")
    return path, torch_module.load(path, map_location="cpu", weights_only=False)


def make_agent(fold, anchor, x, y, folds, torch_module, layers, build):
    train_idx, test_idx = folds[fold - 1]
    hp = SimpleNamespace(
        n_hidden=18,
        n_head=2,
        dropout=0.35,
        nlayer=1,
        nmodal=4,
        nclass=2,
    )
    model = build(layers, hp).to("cuda")
    if sum(parameter.numel() for parameter in model.parameters()) != 49544:
        raise AssertionError("CAM-only parameter count is not 49,544")
    features = torch_module.from_numpy(x).float().cuda()
    targets = torch_module.from_numpy(y).long().cuda()
    counts = np.bincount(y[train_idx], minlength=2).astype(np.float32)
    weights = torch_module.from_numpy((len(train_idx) / counts / 2).astype(np.float32)).cuda()
    optimizer = torch_module.optim.Adam(model.parameters(), lr=0.0038, weight_decay=0.11)
    model.load_state_dict(anchor["model"]["pln"])
    optimizer.load_state_dict(copy.deepcopy(anchor["mf"]))
    restore_rng(anchor["rng"], torch_module)
    model.train()
    return SimpleNamespace(
        model=model,
        optimizer=optimizer,
        features=features,
        targets=targets,
        train_idx=np.asarray(train_idx),
        test_idx=np.asarray(test_idx),
        weights=weights,
    )


def forward(agent):
    return agent.model(agent.features)[0]


def train_step(agent, torch_module):
    import torch.nn.functional as functional

    agent.model.train()
    agent.optimizer.zero_grad()
    output = forward(agent)
    loss = functional.nll_loss(output[agent.train_idx], agent.targets[agent.train_idx], agent.weights)
    if not bool(torch_module.isfinite(output).all()) or not bool(torch_module.isfinite(loss)):
        raise FloatingPointError("non-finite CAM-only training state")
    loss.backward()
    agent.optimizer.step()
    return float(loss.detach().cpu())


def evaluate(agent, torch_module):
    agent.model.eval()
    with torch_module.no_grad():
        return forward(agent)


def test_metrics(output, agent, torch_module):
    from sklearn.metrics import confusion_matrix, roc_auc_score

    predictions = output[agent.test_idx].detach().cpu().numpy().argmax(axis=1)
    targets = agent.targets[agent.test_idx].detach().cpu().numpy()
    tn_or_tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    tp, fn, fp, tn = map(int, tn_or_tp)
    return {
        "acc": float((predictions == targets).mean()),
        "hard_auc": float(roc_auc_score(np.eye(2)[targets], np.eye(2)[predictions])),
        "spe": float(tn / (tn + fp)),
        "sen": float(tp / (tp + fn)),
        "asd_tp": tp,
        "asd_fn": fn,
        "td_fp": fp,
        "td_tn": tn,
        "test_predictions": predictions.astype(np.int64),
    }


def capture_cam144(agent, torch_module):
    captured = []

    def pre_hook(_module, args):
        if not args:
            raise AssertionError("OutputLayer hook received no input")
        captured.append(args[0].detach().cpu().clone())

    handle = agent.model.Outputlayer.register_forward_pre_hook(pre_hook)
    with torch_module.no_grad():
        output, hidden, _attention = agent.model(agent.features)
    handle.remove()
    if len(captured) != 1 or tuple(captured[0].shape) != (871, 144):
        raise AssertionError(f"pre-OutputLayer CAM representation has shape {getattr(captured[0], 'shape', None)}")
    cam144 = captured[0].numpy().astype(np.float32, copy=True)
    if not np.isfinite(cam144).all():
        raise FloatingPointError("CAM144 contains NaN or Inf")
    return output, hidden, cam144


def rebuild_fold(fold, x, y, folds, expected, torch_module, layers, build):
    anchor_path, anchor = load_anchor(fold, torch_module)
    train_idx, test_idx = folds[fold - 1]
    if anchor["fold"] != fold or anchor["split_sha"] != split_sha(train_idx, test_idx):
        raise AssertionError(f"anchor or split identity mismatch for fold {fold}")
    agent = make_agent(fold, anchor, x, y, folds, torch_module, layers, build)
    if any(token in name.lower() for name in list(dict(agent.model.named_parameters())) + list(agent.model.state_dict()) for token in ("gfr", "graph", "gcn", "pen", "spgs", "prompt")):
        raise AssertionError("forbidden GFR/PEN/graph parameter found in CAM-only network")

    best_acc = 0.0
    best_epoch = -1
    wait = 0
    best_state = None
    started = time.monotonic()
    trace = []
    epochs_executed = 0
    for epoch in range(450):
        loss = train_step(agent, torch_module)
        output = evaluate(agent, torch_module).detach()
        acc = float(output[agent.test_idx].argmax(1).eq(agent.targets[agent.test_idx]).double().mean().item())
        epochs_executed = epoch + 1
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            wait = 0
            best_state = {
                "model": {name: value.detach().cpu().clone() for name, value in agent.model.state_dict().items()},
                "optimizer": copy.deepcopy(agent.optimizer.state_dict()),
                "rng": copy.deepcopy(rng_snapshot(torch_module)),
            }
        else:
            wait += 1
        if epoch % 10 == 0 or epoch == 449:
            trace.append({"epoch": epoch, "loss": loss, "acc": acc, "best_acc": best_acc})
        if wait > 200:
            break
    if best_state is None:
        raise RuntimeError(f"no CAM-only best state for fold {fold}")

    agent.model.load_state_dict(best_state["model"])
    output, hidden, cam144 = capture_cam144(agent, torch_module)
    without_hook = evaluate(agent, torch_module).detach().cpu().numpy()
    with_hook = output.detach().cpu().numpy()
    if not np.array_equal(with_hook, without_hook):
        raise AssertionError(f"feature hook changed model output for fold {fold}")
    if tuple(hidden.shape) != (871, 36):
        raise AssertionError(f"unexpected hidden shape {tuple(hidden.shape)}")
    metrics = test_metrics(output, agent, torch_module)
    test_pred_sha = sha256_bytes(metrics["test_predictions"].tobytes())
    expected_row = expected[fold]
    checks = {}
    for key in ("acc", "hard_auc", "spe", "sen"):
        checks[key] = abs(metrics[key] - float(expected_row[key])) <= 1e-12
    for key in ("asd_tp", "asd_fn", "td_fp", "td_tn"):
        checks[key] = metrics[key] == int(expected_row[key])
    checks["best_epoch"] = best_epoch == int(expected_row["best_epoch"])
    checks["epochs_executed"] = epochs_executed == int(expected_row["epochs_executed"])
    checks["test_prediction_sha256"] = test_pred_sha == expected_row["prediction_sha256"]
    if not all(checks.values()):
        raise RuntimeError(f"BLOCKED_CAM_ONLY_CHECKPOINT: E0 rebuild mismatch fold {fold}: {checks}")

    patient_ids = np.arange(871, dtype=np.int64)
    output_predictions = output.detach().cpu().numpy().argmax(axis=1).astype(np.int64)
    out_path = OUT / f"CAM_only_fold{fold:02}_features.npz"
    np.savez_compressed(
        out_path,
        features=cam144,
        labels=y.astype(np.int64),
        predictions=output_predictions,
        patient_ids=patient_ids,
        test_indices=np.asarray(test_idx, dtype=np.int64),
        cam=cam144,
    )
    row = {
        "fold": fold,
        "status": "PASS",
        "anchor_path": str(anchor_path),
        "anchor_sha256": sha256_file(anchor_path),
        "anchor_signature": anchor["signature"],
        "split_sha256": split_sha(train_idx, test_idx),
        "model_parameter_count": sum(parameter.numel() for parameter in agent.model.parameters()),
        "cam_shape": list(cam144.shape),
        "hidden_shape": list(hidden.shape),
        "output_shape": list(with_hook.shape),
        "cam_array_sha256": sha256_bytes(cam144.tobytes()),
        "feature_archive_sha256": sha256_file(out_path),
        "test_prediction_sha256": test_pred_sha,
        "full_prediction_sha256": sha256_bytes(output_predictions.tobytes()),
        "best_epoch": best_epoch,
        "epochs_executed": epochs_executed,
        "metrics": {key: value for key, value in metrics.items() if key != "test_predictions"},
        "historical_identity_checks": checks,
        "hook_output_equal_without_hook": True,
        "trace": trace,
        "wall_seconds": time.monotonic() - started,
    }
    (AUDIT / f"fold{fold:02}_identity.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(f"CAM_ONLY_FOLD_PASS fold={fold:02} epoch={best_epoch} archive_sha={row['feature_archive_sha256']}", flush=True)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    torch_module, layers, build = load_context()
    x, y, folds = load_data()
    expected = load_expected()
    rows = []
    for fold in range(1, 11):
        rows.append(rebuild_fold(fold, x, y, folds, expected, torch_module, layers, build))
    summary = {
        "status": "PASS_10_OF_10",
        "checkpoint_identity": "REBUILT_FROM_VERIFIED_E43.1_ANCHORS_AND_MATCHED_E43.4_E0_HISTORY",
        "source_commit": "396512c7789d4b27c0651ea85cee60fbc05e0c6a",
        "source_files": {path.name: sha256_file(path) for path in SOURCE.iterdir() if path.is_file()},
        "anchor_directory": str(ANCHORS),
        "anchor_hashes": {str(fold): sha256_file(ANCHORS / f"fold{fold:02}.pt") for fold in range(1, 11)},
        "feature_archives": rows,
        "model_parameter_count": 49544,
        "feature_definition": "pre-OutputLayer x.view(batch_size,-1), captured by forward pre-hook",
        "labels_and_patient_ids": "copied from the original 871-row ABIDE order, ASD=0 and TD=1",
    }
    (AUDIT / "cam_only_rebuild_status.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("CAM_ONLY_REBUILD_PASS 10/10", flush=True)


if __name__ == "__main__":
    main()
