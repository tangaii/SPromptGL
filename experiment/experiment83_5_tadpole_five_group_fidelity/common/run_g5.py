"""Anchored, physically instantiated TADPOLE CAM-only formal folds."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cam_only import PhysicalCAMOnly, weighted_nll
from fold_checkpoint import (DATA_SHA, SPLIT_SHA, atomic_write, fingerprint, lock,
                             publish_epoch, recover_best, sha_file, write_checkpoint,
                             write_json)
from run_e833 import (PROMPT8, SEED, capture_rng_state, load_data, restore_rng,
                      seed_everything, configure_worker)
from fidelity_model import FidelityModel
from metrics import acc_auc

ROOT = Path(__file__).resolve().parents[1]
G5 = "G5_E43_4_E0_ANCHORED_PHYSICAL_CAM"


def digest(value) -> str:
    hasher = hashlib.sha256()

    def add(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            hasher.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            hasher.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            hasher.update(str((item.dtype, item.shape)).encode())
            hasher.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                hasher.update(repr(key).encode())
                add(item[key])
        elif isinstance(item, (list, tuple)):
            for value in item:
                add(value)
        else:
            hasher.update(repr(item).encode())

    add(value)
    return hasher.hexdigest()


def construct_anchor(fold: int, device: torch.device):
    """Rebuild E83.4 G2's fresh TADPOLE CAM constructor state in isolation."""
    outside = capture_rng_state()
    try:
        seed_everything(SEED)
        reference = FidelityModel((145, 148, 3, 36, 24, 9), "G2_E43_1_E00", 3,
                                  PROMPT8, fold=fold, device=device).to(device)
        historical_optimizer = torch.optim.Adam(reference.cam.parameters(), lr=0.012,
                                                weight_decay=0.0036)
        anchor = {
            "fold": fold,
            "group": G5,
            "source": "E83.4 G2 untrained TADPOLE CAM constructor only",
            "cam_state": copy.deepcopy(reference.cam.state_dict()),
            "adam_state": copy.deepcopy(historical_optimizer.state_dict()),
            "rng": capture_rng_state(),
        }
        anchor["cam_sha256"] = digest(anchor["cam_state"])
        anchor["adam_sha256"] = digest(anchor["adam_state"])
        anchor["rng_sha256"] = digest(anchor["rng"])
        return anchor
    finally:
        restore_rng(outside)


def restore_anchor(anchor, device: torch.device):
    model = PhysicalCAMOnly().to(device)
    model.cam.load_state_dict(anchor["cam_state"], strict=True)
    optimizer = torch.optim.Adam(model.cam.parameters(), lr=0.012, weight_decay=0.0036)
    optimizer.load_state_dict(copy.deepcopy(anchor["adam_state"]))
    restore_rng(anchor["rng"])
    if digest(model.cam.state_dict()) != anchor["cam_sha256"]:
        raise RuntimeError("Physical G5 CAM differs from its untrained anchor")
    if digest(optimizer.state_dict()) != anchor["adam_sha256"]:
        raise RuntimeError("Physical G5 Adam differs from its untrained anchor")
    if digest(capture_rng_state()) != anchor["rng_sha256"]:
        raise RuntimeError("Physical G5 RNG differs from its post-construction anchor")
    audit_physical_structure(model, optimizer)
    return model, optimizer


def audit_physical_structure(model, optimizer):
    forbidden = ("gfr", "graphlearn", "gcn", "messagepassing", "pen", "spgs",
                 "res", "clip", "prompt", "graphconstructloss")
    names = [name.lower() for name, _ in model.named_parameters()]
    keys = [name.lower() for name in model.state_dict()]
    module_names = [type(module).__name__.lower() for module in model.modules()]
    dropout_probabilities = [float(module.p) for module in model.modules()
                             if isinstance(module, torch.nn.Dropout)]
    if any(probability != 0.0 for probability in dropout_probabilities):
        raise RuntimeError("G5 dropout differs from the registered zero-dropout protocol")
    if any(term in name for name in names + keys + module_names for term in forbidden):
        raise RuntimeError("Forbidden module or state key in physical G5 model")
    optimized = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    model_parameters = [id(parameter) for parameter in model.parameters()]
    if len(optimizer.param_groups) != 1 or set(optimized) != set(model_parameters):
        raise RuntimeError("G5 must have one optimizer over CAM parameters only")
    if not all(name.startswith("cam.") for name in names + keys):
        raise RuntimeError("G5 registered state outside the CAM")
    return {"parameters": sum(parameter.numel() for parameter in model.parameters()),
            "named_parameters": names, "state_keys": keys,
            "module_types": module_names,
            "dropout_probabilities": dropout_probabilities,
            "optimizer_count": len(optimizer.param_groups), "forbidden_absent": True}


def math_equivalence(anchor, x, labels, rows, device):
    outside = capture_rng_state()
    try:
        reference, reference_optimizer = restore_anchor(anchor, device)
        physical, physical_optimizer = restore_anchor(anchor, device)
        step_records = []
        for step in range(2):
            start_rng = capture_rng_state()
            reference_optimizer.zero_grad(set_to_none=True)
            physical_optimizer.zero_grad(set_to_none=True)
            ref_logp, ref_details = reference.cam(x)
            ref_logits = ref_details["logits"]
            phy_logp, phy_logits, _ = physical(x)
            torch.testing.assert_close(phy_logits, ref_logits, rtol=0, atol=0)
            ref_loss = weighted_nll(ref_logits, labels, rows)
            phy_loss = weighted_nll(phy_logits, labels, rows)
            torch.testing.assert_close(phy_loss, ref_loss, rtol=0, atol=0)
            ref_loss.backward()
            phy_loss.backward()
            ref_grad = {name: (None if parameter.grad is None else parameter.grad.detach().clone())
                        for name, parameter in reference.named_parameters()}
            phy_grad = {name: (None if parameter.grad is None else parameter.grad.detach().clone())
                        for name, parameter in physical.named_parameters()}
            if ref_grad.keys() != phy_grad.keys():
                raise RuntimeError("G5 reference and physical parameter sets differ")
            for name in ref_grad:
                if ref_grad[name] is None or phy_grad[name] is None:
                    if ref_grad[name] is not phy_grad[name]:
                        raise RuntimeError(f"G5 gradient connectivity differs for {name}")
                else:
                    torch.testing.assert_close(phy_grad[name], ref_grad[name], rtol=0, atol=0)
            reference_optimizer.step()
            physical_optimizer.step()
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(physical.state_dict()[name], value, rtol=0, atol=0)
            if digest(reference_optimizer.state_dict()) != digest(physical_optimizer.state_dict()):
                raise RuntimeError("G5 Adam states differ after matched update")
            if digest(capture_rng_state()) != digest(start_rng):
                raise RuntimeError("Unexpected RNG change in deterministic CAM-only parity step")
            step_records.append({"step": step + 1, "logits_exact": True, "loss_exact": True,
                                 "all_gradients_exact": True, "parameters_exact": True,
                                 "adam_state_exact": True, "rng_exact": True})
        return {"status": "PASS", "steps": step_records}
    finally:
        restore_rng(outside)


def checkpoint_identity(fold, train_idx, test_idx, config_hash):
    prompt_manifest = json.loads((ROOT / "prompt_manifest.json").read_text())
    return {
        "experiment": "Experiment 83.5", "group": G5, "fold": fold, "seed": SEED,
        "model_type": "PhysicalCAMOnly", "implementation_sha256": fingerprint(),
        "data_sha256": DATA_SHA, "split_sha256": SPLIT_SHA,
        "protocol_sha256": sha_file(ROOT / "protocol.json"),
        "source_manifest_sha256": sha_file(ROOT / "sources/source_manifest.json"),
        "prompt_manifest_sha256": sha_file(ROOT / "prompt_manifest.json"),
        "prompt_vector_sha256": prompt_manifest["vector_sha256"],
        "prompt_query_sha256": prompt_manifest["query_sha256"],
        "group_config_sha256": config_hash,
        "train_idx": np.asarray(train_idx, dtype=np.int64).tolist(),
        "test_idx": np.asarray(test_idx, dtype=np.int64).tolist(),
    }


def _prediction_table(fold, test_idx, ids, truth, prediction, probability):
    import pandas as pd

    return pd.DataFrame({
        "fold": fold, "seed": SEED, "sample_index": test_idx,
        "sample_id": ids.iloc[test_idx]["RID"].to_numpy(), "true_label": truth,
        "pred_label": prediction, "prob_NC": probability[:, 0],
        "prob_MCI": probability[:, 1], "prob_AD": probability[:, 2],
    })


def _verify_completed_fold(fold, audit_path, best_path, latest_path, anchor_path, prediction_path,
                           identity):
    import pandas as pd

    audit = json.loads(audit_path.read_text())
    for key, value in identity.items():
        if audit.get(key) != value:
            raise RuntimeError(f"Completed G5 fold {fold} has a different {key}")
    if audit.get("status") != "TRAINED":
        raise RuntimeError(f"G5 fold {fold} has an incomplete audit")
    if (audit.get("best_sha256") != sha_file(best_path)
            or audit.get("latest_sha256") != sha_file(latest_path)
            or audit.get("anchor_sha256") != sha_file(anchor_path)
            or audit.get("prediction_sha256") != sha_file(prediction_path)):
        raise RuntimeError(f"Completed G5 fold {fold} artifact hash mismatch")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    if best.get("epoch") != audit.get("best_epoch"):
        raise RuntimeError(f"G5 fold {fold} Best epoch differs from its audit")
    for key, value in identity.items():
        if best.get(key) != value or latest.get(key) != value:
            raise RuntimeError(f"G5 fold {fold} checkpoint differs from its {key} identity")
    if latest.get("best_sha256") != audit["best_sha256"]:
        raise RuntimeError(f"G5 fold {fold} Latest points to a different Best")
    if latest.get("epoch") != audit.get("latest_epoch"):
        raise RuntimeError(f"G5 fold {fold} Latest epoch differs from its audit")
    table = pd.read_csv(prediction_path)
    if len(table) != audit["test_count"] or int(table["fold"].iloc[0]) != fold:
        raise RuntimeError(f"G5 fold {fold} prediction table is incomplete")
    return {"group": G5, "fold": fold, "status": "VERIFIED_PREVIOUS",
            "best_sha256": audit["best_sha256"], "latest_sha256": audit["latest_sha256"]}


def train_fold(fold: int, workers: int = 1):
    device = torch.device("cuda:0")
    (x_np, y_np, ids, splits) = load_data()
    split = next(item for item in splits if int(item["fold"]) == fold)
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    group_dir = ROOT / G5
    for name in ("anchors", "best_checkpoints", "latest_checkpoints", "predictions", "audits"):
        (group_dir / name).mkdir(parents=True, exist_ok=True)
    anchor_path = group_dir / "anchors" / f"fold{fold:02d}.pt"
    best_path = group_dir / "best_checkpoints" / f"fold{fold:02d}.pt"
    latest_path = group_dir / "latest_checkpoints" / f"fold{fold:02d}.pt"
    audit_path = group_dir / "audits" / f"fold{fold:02d}.json"
    prediction_path = group_dir / "predictions" / f"fold{fold:02d}.csv"
    config_path = ROOT / "configs" / f"{G5}.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Frozen G5 config is missing: {config_path}")
    signature = fingerprint()
    identity = checkpoint_identity(fold, train_idx, test_idx, sha_file(config_path))
    if audit_path.is_file():
        return _verify_completed_fold(fold, audit_path, best_path, latest_path, anchor_path,
                                      prediction_path, identity)
    if best_path.exists() and not latest_path.exists():
        raise RuntimeError(f"Unsealed G5 Best without Latest at {best_path}")
    if not anchor_path.exists():
        anchor = construct_anchor(fold, device)
        anchor.update(identity)
        anchor.update({"anchor_type": "untrained_CAM_and_initial_Adam_and_post_construction_RNG"})
        write_checkpoint(anchor_path, anchor)
    anchor = torch.load(anchor_path, map_location="cpu", weights_only=False)
    expected_identity = identity
    if any(anchor.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError("G5 initialization anchor provenance mismatch")
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    rows = torch.from_numpy(train_idx).long().to(device)
    test = torch.from_numpy(test_idx).long().to(device)
    model, optimizer = restore_anchor(anchor, device)
    predictions_initial = model(x)[1]
    if not torch.isfinite(predictions_initial).all():
        raise FloatingPointError("G5 initial logits are non-finite")
    parity = math_equivalence(anchor, x, labels, rows, device)
    saved = torch.load(latest_path, map_location="cpu", weights_only=False) if latest_path.exists() else None
    if latest_path.exists():
        for key, value in identity.items():
            if saved.get(key) != value:
                raise RuntimeError(f"G5 Latest differs from frozen release field {key}")
        recover_best(best_path, saved.get("best_sha256"))
        if saved.get("anchor_sha256") != sha_file(anchor_path):
            raise RuntimeError("G5 Latest points to a different CAM anchor")
        if saved.get("optimizer_config") != {"class": "Adam", "lr": 0.012, "weight_decay": 0.0036}:
            raise RuntimeError("G5 Latest optimizer configuration differs from protocol")
        if saved.get("parity") != parity:
            raise RuntimeError("G5 resumed math-parity result differs from its anchor")
        if any(saved.get(key) != value for key, value in identity.items()):
            raise RuntimeError("G5 Latest does not match the frozen training seal")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        best_acc, best_epoch, wait = (saved[key] for key in ("best_acc", "best_epoch", "wait"))
        start_epoch = int(saved["epoch"]) + 1
        history = saved["history"]
    else:
        best_acc, best_epoch, wait, start_epoch, history = -1.0, -1, 0, 1, []
    optimizer_config = {"class": "Adam", "lr": 0.012, "weight_decay": 0.0036}
    best_sha = saved.get("best_sha256") if saved else None
    max_epochs, patience = 450, 200
    started = time.monotonic()
    with (group_dir / "train.log").open("a", buffering=1) as log:
        for epoch in range(start_epoch, max_epochs + 1):
            if wait > patience:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            _, logits, _ = model(x)
            loss = weighted_nll(logits, labels, rows)
            if not torch.isfinite(loss):
                raise FloatingPointError("G5 loss is non-finite")
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                eval_logits = model(x)[1]
                prediction = eval_logits.argmax(1)
                val_acc = float((prediction[test] == labels[test]).float().mean())
            history.append({"epoch": epoch, "loss": float(loss.detach()), "val_acc": val_acc})
            best_payload = None
            if val_acc > best_acc:
                best_acc, best_epoch, wait = val_acc, epoch, 0
                best_payload = {**identity, "epoch": epoch, "best_acc": best_acc,
                                "model": copy.deepcopy(model.state_dict()),
                                "anchor_sha256": sha_file(anchor_path),
                                "anchor_cam_sha256": anchor["cam_sha256"],
                                "anchor_adam_sha256": anchor["adam_sha256"],
                                "anchor_rng_sha256": anchor["rng_sha256"],
                                "optimizer_config": optimizer_config,
                                "parity": parity}
            else:
                wait += 1
            latest = {**identity, "epoch": epoch, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "optimizer_config": optimizer_config,
                      "rng": capture_rng_state(), "best_acc": best_acc,
                      "best_epoch": best_epoch, "best_sha256": best_sha, "wait": wait,
                      "history": history, "parity": parity,
                      "anchor_sha256": sha_file(anchor_path)}
            best_sha = publish_epoch(latest_path, best_path, latest, best_payload)
            if epoch == start_epoch or epoch % 10 == 0:
                log.write(f"G5 fold={fold:02d} epoch={epoch:03d} loss={float(loss):.6f} acc={val_acc:.6f}\n")
        if not best_path.exists():
            raise RuntimeError("G5 training produced no Best checkpoint")
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best["model"], strict=True)
        model.eval()
        with torch.no_grad():
            logp, logits, _ = model(x)
        probability = logp.exp().cpu().numpy()
        prediction = logits.argmax(1).cpu().numpy()
        from metrics import acc_auc
        metric = acc_auc(y_np[test_idx], prediction[test_idx], 3)
        from sklearn.metrics import roc_auc_score
        metric["AUC_probability"] = float(roc_auc_score(
            y_np[test_idx], probability[test_idx], average="macro", multi_class="ovr"))
        table = _prediction_table(fold, test_idx, ids, y_np[test_idx],
                                  prediction[test_idx], probability[test_idx])
        atomic_write(prediction_path,
                     lambda stream: stream.write(table.to_csv(index=False).encode()))
        audit = {**identity, "status": "TRAINED", "workers": workers,
                 "best_epoch": int(best["epoch"]), "ACC": metric["ACC"],
                 "AUC_original": metric["AUC"], "AUC_probability": metric["AUC_probability"],
                 "correct_count": int((prediction[test_idx] == y_np[test_idx]).sum()),
                 "class_correct_counts": {str(class_id): int(((prediction[test_idx] == y_np[test_idx])
                                                  & (y_np[test_idx] == class_id)).sum())
                                           for class_id in range(3)},
                 "train_count": len(train_idx), "test_count": len(test_idx),
                 "anchor_sha256": sha_file(anchor_path),
                 "anchor_cam_sha256": anchor["cam_sha256"], "anchor_adam_sha256": anchor["adam_sha256"],
                 "anchor_rng_sha256": anchor["rng_sha256"], "parity": parity,
                 "physical_structure": audit_physical_structure(model, optimizer),
                 "best_sha256": sha_file(best_path), "latest_sha256": sha_file(latest_path),
                 "prediction_sha256": sha_file(prediction_path),
                 "latest_epoch": int(latest["epoch"]), "stop_wait": int(latest["wait"]),
                 "peak_reserved_mib": torch.cuda.max_memory_reserved(device)/2**20,
                 "train_seconds": time.monotonic()-started}
        write_json(audit_path, audit)
        return audit


def _run_fold(fold, workers):
    from run_e833 import configure_worker

    configure_worker()
    directory = ROOT / G5
    (directory / "latest_checkpoints").mkdir(parents=True, exist_ok=True)
    with lock(directory / "latest_checkpoints" / f"fold{fold:02d}.lock"):
        return train_fold(fold, workers)


def _write_group_summary(workers):
    directory = ROOT / G5
    audits = [json.loads(path.read_text()) for path in
              sorted((directory / "audits").glob("fold*.json"))]
    if len(audits) != 10 or any(row.get("status") != "TRAINED" for row in audits):
        raise RuntimeError(f"G5 has {len(audits)} audited folds, expected ten")
    rows = [{"group": G5, "fold": row["fold"], "ACC": row["ACC"],
             "AUC_original": row["AUC_original"], "AUC_probability": row["AUC_probability"],
             "correct_count": row["correct_count"], "best_epoch": row["best_epoch"]}
            for row in audits]
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(directory / "fold_metrics.csv", lambda stream: stream.write(csv_buffer.getvalue().encode()))
    summary = {"group": G5, "folds": 10, "workers": workers,
               "data_sha256": DATA_SHA, "split_sha256": SPLIT_SHA,
               "correct_total": int(sum(row["correct_count"] for row in audits))}
    for metric in ("ACC", "AUC_original", "AUC_probability"):
        values = np.asarray([row[metric] for row in audits], dtype=np.float64)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    write_json(directory / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--folds", default="1,2,3,4,5,6,7,8,9,10")
    args = parser.parse_args()
    if args.workers not in (1, 2, 4, 8, 10):
        parser.error("--workers must be one of the measured settings 1, 2, 4, 8, or 10")
    folds = sorted({int(value) for value in args.folds.split(",") if value})
    if not folds or not set(folds) <= set(range(1, 11)):
        parser.error("--folds must contain fold IDs from 1 to 10")
    from run_e833 import require_formal_training_gate

    require_formal_training_gate()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Formal G5 execution requires the assigned single GPU")
    output = ROOT / G5
    (output / "anchors").mkdir(parents=True, exist_ok=True)
    failures = []
    with lock(ROOT / ".training.lock"):
        with ProcessPoolExecutor(args.workers, mp_context=__import__("multiprocessing").get_context("spawn")) as pool:
            tasks = {pool.submit(_run_fold, fold, args.workers): fold for fold in folds}
            for future in as_completed(tasks):
                fold = tasks[future]
                try:
                    result = future.result()
                    print(f"{result['status']} {G5} fold {fold:02d}", flush=True)
                except Exception as exc:
                    failures.append(fold)
                    import traceback
                    traceback.print_exception(type(exc), exc, exc.__traceback__)
                    print(f"ERROR {G5} fold {fold:02d}: {exc!r}", file=sys.stderr, flush=True)
    if not failures and set(folds) == set(range(1, 11)):
        _write_group_summary(args.workers)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
