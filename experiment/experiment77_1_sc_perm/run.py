#!/usr/bin/env python3
"""Train SC-PERM groups from exact E74.4-G2 fold warmup states."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from memory import SC_PERM, TemporalErrorTracker
from model import build_sc_perm, forward_sc_perm, prepare_text_cache


HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
BASE = Path(CFG["base"]["path"])
TASKS = HERE / "runtime" / "tasks"
LOGS = HERE / "runtime" / "logs"
CHECKPOINTS = HERE / "checkpoints"
RESULTS = HERE / "results"
GROUPS = ("E77.1-G1", "E77.1-G2", "E77.1-G3")
BASE_GROUP = CFG["base"]["group"]


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E74 = _load_file("e77_locked_e744_runner", BASE / "run.py")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_value(value) -> str:
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"T" + str(tensor.dtype).encode() + repr(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            arr = np.ascontiguousarray(item)
            digest.update(b"A" + str(arr.dtype).encode() + repr(arr.shape).encode() + arr.tobytes())
        elif isinstance(item, dict):
            digest.update(b"{")
            for key in sorted(item, key=lambda x: str(x)):
                visit(str(key)); visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            digest.update(repr(item).encode("utf-8") + b"\0")

    visit(value)
    return digest.hexdigest()


def module_sha(module: torch.nn.Module, parameters_only: bool = False) -> str:
    values = dict(module.named_parameters()) if parameters_only else module.state_dict()
    return digest_value(values)


def atomic_torch(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def task_path(group: str, fold: int) -> Path:
    return TASKS / f"{group}_f{fold:02d}.json"


def latest_path(group: str, fold: int) -> Path:
    return CHECKPOINTS / f"{group}_f{fold:02d}.latest.pt"


def best_path(group: str, fold: int) -> Path:
    return CHECKPOINTS / f"{group}_f{fold:02d}.best.pt"


def log_path(group: str, fold: int) -> Path:
    return LOGS / f"{group}_f{fold:02d}.jsonl"


def append_log(record: dict) -> None:
    path = HERE / "train.log"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_manifest() -> dict:
    files = [
        HERE / "config.json", HERE / "run.py", HERE / "model.py", HERE / "memory.py", HERE / "audit.py",
        BASE / "config.json", BASE / "run.py", BASE / "soft_positive_dcr.py", BASE / "manifest.json",
        BASE / "results" / "fold_metrics.csv", BASE / "results" / "summary.csv",
        BASE / "results" / "audit_status.json",
        HERE / "runtime" / "text_embeddings.pt",
        Path("/model/spromptgl/experiment/experiment60_dss_pen/manifest.json"),
        Path("/data/adodas/SPromptGL-main/main.py"), Path("/data/adodas/SPromptGL-main/model.py"),
        Path("/data/adodas/SPromptGL-main/network.py"), Path("/data/adodas/SPromptGL-main/layers.py"),
        Path("/data/adodas/SPromptGL-main/utils.py"),
        Path("/data/adodas/SPromptGL-main/ABIDE-simple-2-concat-weighted-cosine.sh"),
        Path("/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv"),
        Path("/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy"),
    ]
    for historical in ("experiment75_dcer_v2", "experiment75_1_sa_dcer", "experiment77_ecpsm"):
        root = Path("/model/spromptgl/experiment") / historical
        for relative in ("config.json", "run.py", "dcer_v2.py", "sa_dcer.py", "memory.py", "model.py",
                         "manifest.json", "results/report.md", "results/fold_metrics.csv",
                         "results/interventions.csv", "results/memory_audit.json"):
            candidate = root / relative
            if candidate.is_file():
                files.append(candidate)
    files.extend(sorted(Path(CFG["base"]["warmup_path"]).glob("fold*.pt")))
    files.extend(sorted((BASE / ".runtime/tasks").glob("E74.4-G2-Positive-DCR_f*.json")))
    result = {}
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"protected read-only dependency is missing: {path}")
        result[str(path.resolve())] = sha_file(path)
    return result


def source_sha() -> str:
    return digest_value(source_manifest())


def cpu_state(module: torch.nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def optimizer_old_group_state(optimizer: torch.optim.Optimizer, old_group_count: int) -> dict:
    state = optimizer.state_dict()
    groups = state["param_groups"][:old_group_count]
    ids = {parameter_id for group in groups for parameter_id in group["params"]}
    return {"param_groups": groups, "state": {key: value for key, value in state["state"].items() if key in ids}}


def public_state_sha(state: dict, old_added_groups: int) -> str:
    hist = E74.load_history()
    public = {
        "model": cpu_state(state["model"]),
        "cam": cpu_state(state["cam"]),
        "opt_base": state["opt_base"].state_dict(),
        "opt_cam": state["opt_cam"].state_dict(),
        "opt_added": optimizer_old_group_state(state["opt_added"], old_added_groups),
    }
    return digest_value(public)


def allowed_train_ids(state: dict) -> np.ndarray:
    train = np.asarray(state["train_np"], dtype=np.int64)
    test = np.asarray(state["test_np"], dtype=np.int64)
    return np.setdiff1d(train, test, assume_unique=False)


def initialize_group(fold: int, group: str):
    state, warmup, identity = E74.prepare(fold)
    state["fold"] = fold
    state["group"] = BASE_GROUP
    state["rudag_identity"] = identity
    runtime = E74.load_history().load_e69().load_e61().load_base()
    if runtime.rng_sha(runtime.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold} did not restore the exact E61 epoch-10 RNG")
    base_task = json.loads((BASE / ".runtime/tasks" / f"E74.4-G2-Positive-DCR_f{fold:02d}.json").read_text(encoding="utf-8"))
    if base_task.get("status") != "COMPLETE" or base_task.get("identity") != identity:
        raise RuntimeError(f"fold {fold} does not match the audited E74.4-G2 identity")
    rng_before_module = runtime.pack_rng()
    ecpsm = build_sc_perm(fold, group, state["x"].device)
    if runtime.rng_sha(runtime.pack_rng()) != runtime.rng_sha(rng_before_module):
        raise RuntimeError("SC-PERM module initialization changed an inherited public RNG")
    old_added_groups = len(state["opt_added"].param_groups)
    added_optimizer_cfg = CFG["base"]["optimizer"]["added"]
    state["opt_added"].add_param_group({
        "params": list(ecpsm.parameters()),
        "lr": added_optimizer_cfg["lr"],
        "weight_decay": added_optimizer_cfg["weight_decay"],
    })
    tracker = TemporalErrorTracker(CFG["method"]["hard_error_window"]).to(state["x"].device)
    return state, warmup, identity, base_task, ecpsm, tracker, old_added_groups, runtime


def _finite_gradients(modules) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
               for module in modules for parameter in module.parameters())


def _snapshot(state: dict, ecpsm: SC_PERM, tracker: TemporalErrorTracker) -> dict:
    return {
        "base_state": copy.deepcopy(E74.load_history().state_bundle(state)),
        "ecpsm_state": cpu_state(ecpsm),
        "tracker_state": cpu_state(tracker),
    }


def _restore(state: dict, ecpsm: SC_PERM, tracker: TemporalErrorTracker, saved: dict) -> None:
    E74.load_history().load_bundle(state, saved["base_state"])
    ecpsm.load_state_dict(saved["ecpsm_state"], strict=True)
    tracker.load_state_dict(saved["tracker_state"], strict=True)


def _metric_payload(metrics: dict) -> dict:
    return {key: float(value) for key, value in metrics.items() if key not in ("pred", "prob")}


def _signature(state: dict, ecpsm: SC_PERM, old_added_groups: int,
               tracker: TemporalErrorTracker, output: dict, rng_trace: dict) -> dict:
    return {
        "logp_sha256": digest_value(output["logp"].detach()),
        "loss_sha256": digest_value(output["total"].detach()),
        "shared_state_sha256": public_state_sha(state, old_added_groups),
        "ecpsm_parameters_sha256": module_sha(ecpsm, parameters_only=True),
        "class_memory_sha256": digest_value({
            key: value for key, value in ecpsm.state_dict().items()
            if key.startswith("class_")
        }),
        "hard_memory_sha256": digest_value({
            key: value for key, value in ecpsm.state_dict().items() if key.startswith("hard_")
        }),
        "tracker_sha256": digest_value(tracker.state_dict()),
        "rng": rng_trace,
    }


def _initial_memory(state: dict, ecpsm: SC_PERM, group: str) -> dict:
    allowed = allowed_train_ids(state)
    capture: dict = {}
    with torch.no_grad():
        output, _ = forward_sc_perm(
            E74, state, ecpsm, group, training=False, local_epoch=0,
            prototype_off=True, capture=capture,
        )
    features = capture["cam_features"]
    allowed_t = torch.as_tensor(allowed, dtype=torch.long, device=features.device)
    ecpsm.update_class_memory(features, state["y"], allowed_t)
    return {"logp_sha256": digest_value(output["logp"].detach()), "class_memory_initialized": True,
            "allowed_memory_ids": int(len(allowed)), "explicit_test_overlap_excluded": int(len(state["train_np"]) - len(allowed))}


def training_epoch(state: dict, ecpsm: SC_PERM, tracker: TemporalErrorTracker,
                   group: str, local_epoch: int, allowed: np.ndarray,
                   allowed_t: torch.Tensor, runtime) -> dict:
    memory_epoch = int(local_epoch)
    rng_before = runtime.rng_sha(runtime.pack_rng())
    E74.load_history().zero_optimizers(state)
    out, train_hook = forward_sc_perm(E74, state, ecpsm, group,
                                      training=True, local_epoch=memory_epoch)
    info = train_hook["info"]
    train_ids = torch.as_tensor(allowed, dtype=torch.long, device=state["x"].device)
    proto_loss = ecpsm.prototype_aux_loss(info["class_scores"], state["y"], train_ids)
    hard_loss = out["total"] * 0.0
    if CFG["method"]["groups"][group]["hard_memory"] and memory_epoch >= CFG["method"]["hard_memory_warmup_epochs"]:
        hard_loss = ecpsm.hard_ranking_loss(
            info, state["y"], torch.nonzero(tracker.stable_fp_mask, as_tuple=False).flatten(),
            torch.nonzero(tracker.stable_fn_mask, as_tuple=False).flatten(),
        )
    out["prototype_aux"] = proto_loss
    out["hard_ranking"] = hard_loss
    out["total"] = out["total"] + CFG["method"]["prototype_aux_weight"] * proto_loss \
        + CFG["method"]["hard_ranking_weight"] * hard_loss
    if not torch.isfinite(out["total"]):
        raise FloatingPointError(f"non-finite SC-PERM loss at fold={state['fold']} group={group} epoch={memory_epoch}")
    out["total"].backward()
    if not _finite_gradients((state["model"], state["cam"], ecpsm)):
        raise FloatingPointError(f"non-finite SC-PERM gradient at fold={state['fold']} group={group} epoch={memory_epoch}")
    module_grad = [float(p.grad.detach().norm().cpu()) for p in ecpsm.parameters() if p.grad is not None]
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    rng_after_train = runtime.rng_sha(runtime.pack_rng())

    capture: dict = {}
    with torch.no_grad():
        evaluated, _ = forward_sc_perm(E74, state, ecpsm, group,
                                       training=False, local_epoch=memory_epoch, capture=capture)
        metrics = runtime.classification_metrics(evaluated["logp"], state["test"], state["y"])
    rng_after_eval = runtime.rng_sha(runtime.pack_rng())
    evaluation_memory = cpu_state(ecpsm)

    error_state = tracker.update(evaluated["logp"], state["y"], allowed_t,
                                 min_votes=CFG["method"]["hard_error_min_votes"])
    memory_update = {"pool_fp": 0, "pool_fn": 0}
    if CFG["method"]["groups"][group]["hard_memory"] and memory_epoch >= CFG["method"]["hard_memory_warmup_epochs"]:
        memory_update = ecpsm.update_hard_memory(
            capture["cam_features"], state["y"], error_state["vote_counts"],
            error_state["prediction_confidence"],
            allowed_t, memory_epoch,
        )
    ecpsm.update_class_memory(capture["cam_features"], state["y"], allowed_t)
    rng_after_memory = runtime.rng_sha(runtime.pack_rng())
    if rng_after_memory != rng_after_eval:
        raise RuntimeError("SC-PERM memory update advanced the inherited public RNG")
    if not torch.isfinite(ecpsm.class_centers).all() or not torch.isfinite(ecpsm.hard_centers).all():
        raise FloatingPointError(f"non-finite SC-PERM memory state at fold={state['fold']} epoch={memory_epoch}")

    hard_diagnostic = {
        "epoch": memory_epoch, "fp_true": int(error_state["fp_ids"].numel()),
        "fn_true": int(error_state["fn_ids"].numel()),
        "near_fp": int(error_state["near_fp_ids"].numel()),
        "near_fn": int(error_state["near_fn_ids"].numel()),
        "fp_confidence": float(error_state["fp_confidence"]),
        "fn_confidence": float(error_state["fn_confidence"]),
        "fp_age_mean": float(error_state["fp_age_mean"]),
        "fn_age_mean": float(error_state["fn_age_mean"]),
        "fp_prototypes": int((ecpsm.hard_counts[0] > 0).sum().item()),
        "fn_prototypes": int((ecpsm.hard_counts[1] > 0).sum().item()),
        "unique_stable_fp_patients": int(error_state["unique_stable_fp_patients"]),
        "unique_stable_fn_patients": int(error_state["unique_stable_fn_patients"]),
        "hard_fp_occupancy": int((ecpsm.hard_subject_ids[0] >= 0).sum().item()),
        "hard_fn_occupancy": int((ecpsm.hard_subject_ids[1] >= 0).sum().item()),
        "pool_fp": int(memory_update["pool_fp"]), "pool_fn": int(memory_update["pool_fn"]),
    }
    rng_trace = {"epoch": int(CFG["base"]["start_epoch_zero_based"]) + memory_epoch + 1,
                 "rng_before": rng_before, "rng_after_train": rng_after_train,
                 "rng_after_eval": rng_after_eval, "rng_after_memory": rng_after_memory}
    return {"out": out, "evaluated": evaluated, "capture": capture, "metrics": metrics,
            "error_state": error_state, "hard_diagnostic": hard_diagnostic,
            "evaluation_memory": evaluation_memory, "rng": rng_trace,
            "module_grad_norms": module_grad}


def train_group(fold: int, group: str, expected_source_sha: str) -> dict:
    if group not in GROUPS or fold not in range(1, 11):
        raise ValueError(f"invalid formal group/fold: {group}/{fold}")
    task_file, best_file, latest_file = task_path(group, fold), best_path(group, fold), latest_path(group, fold)
    if task_file.is_file():
        existing = json.loads(task_file.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE" and existing.get("source_sha256") == expected_source_sha and existing.get("checkpoint_sha256") == sha_file(best_file):
            return existing
        raise RuntimeError(f"refusing to overwrite incompatible E77 task record {task_file}")

    state, warmup, identity, base_task, ecpsm, tracker, old_added_groups, runtime = initialize_group(fold, group)
    allowed = allowed_train_ids(state)
    allowed_t = torch.as_tensor(allowed, dtype=torch.long, device=state["x"].device)
    overlap = sorted(set(np.asarray(state["train_np"], dtype=np.int64).tolist()) &
                     set(np.asarray(state["test_np"], dtype=np.int64).tolist()))
    if sorted(overlap) != ([376, 406] if fold == 2 else []):
        raise RuntimeError(f"unexpected fold membership overlap in fold {fold}: {overlap}")

    start_epoch = int(CFG["base"]["start_epoch_zero_based"])
    wait, best_acc, best_epoch = 0, -1.0, -1
    trace, signatures, hard_diagnostics = [], [], []
    rng_before_memory = runtime.pack_rng()
    initial_memory = _initial_memory(state, ecpsm, group)
    if runtime.rng_sha(runtime.pack_rng()) != runtime.rng_sha(rng_before_memory):
        raise RuntimeError("initial class memory construction changed an inherited public RNG")
    if latest_file.is_file():
        saved = torch.load(latest_file, map_location=state["x"].device, weights_only=False)
        expected_fields = {
            "fold": fold, "group": group, "source_sha256": expected_source_sha,
            "identity": identity, "warmup_sha256": identity["warmup_sha256"],
        }
        for key, expected in expected_fields.items():
            if saved.get(key) != expected:
                raise RuntimeError(f"latest checkpoint identity mismatch for {key}: fold {fold}")
        _restore(state, ecpsm, tracker, saved)
        runtime.restore_rng(saved["rng"])
        start_epoch = int(saved["next_epoch"])
        wait = int(saved["wait"])
        best_acc = float(saved["best_acc"])
        best_epoch = int(saved["best_epoch"])
        trace = list(saved["rng_trace"])
        signatures = list(saved["warmup_signatures"])
        hard_diagnostics = list(saved["hard_diagnostics"])
        if not best_file.is_file():
            raise RuntimeError("latest recovery state exists without its best checkpoint")
    elif runtime.rng_sha(runtime.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold}/{group} initial RNG changed before epoch 11")

    if start_epoch < 10 or start_epoch > CFG["base"]["max_epochs"]:
        raise RuntimeError(f"invalid E77 recovery epoch {start_epoch}")
    started = time.time()
    executed = start_epoch
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, CFG["base"]["max_epochs"]):
        memory_epoch = epoch - CFG["base"]["start_epoch_zero_based"]
        step = training_epoch(state, ecpsm, tracker, group, memory_epoch, allowed, allowed_t, runtime)
        out, evaluated, metrics = step["out"], step["evaluated"], step["metrics"]
        eval_capture, evaluation_memory = step["capture"], step["evaluation_memory"]
        epoch_rng = step["rng"]
        trace.append(epoch_rng)
        executed = epoch + 1
        rng_after_train = epoch_rng["rng_after_train"]
        rng_after_eval = epoch_rng["rng_after_eval"]
        improved = metrics["acc"] > best_acc
        if improved:
            best_acc, best_epoch, wait = float(metrics["acc"]), epoch, 0
        else:
            wait += 1

        hard_diagnostics.append(step["hard_diagnostic"])

        if memory_epoch < CFG["method"]["hard_memory_warmup_epochs"]:
            signatures.append(_signature(state, ecpsm, old_added_groups, tracker, evaluated, epoch_rng))

        if improved:
            best_payload = {
                "experiment": CFG["experiment"], "group": group, "fold": fold, "seed": 0,
                "source_sha256": expected_source_sha, "base_identity": identity,
                "warmup_sha256": identity["warmup_sha256"], "warmup_rng_sha": identity["warmup_rng_sha"],
                "best_epoch_1based": epoch + 1, "epochs_executed_at_selection": epoch + 1,
                "metrics": _metric_payload(metrics),
                "test_probability": metrics["prob"].copy(), "test_prediction": metrics["pred"].copy(),
                "test_indices": np.asarray(state["test_np"], dtype=np.int64).copy(),
                "base_state": copy.deepcopy(E74.load_history().state_bundle(state)),
                "ecpsm_state": evaluation_memory,
                "post_epoch_ecpsm_state": cpu_state(ecpsm),
                "tracker_state": cpu_state(tracker),
                "memory_update_epoch": epoch,
                "rng": runtime.pack_rng(),
                "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval,
                "rng_trace": list(trace),
                "initial_memory": initial_memory,
                "hard_diagnostics": list(hard_diagnostics),
                "scheduler_state": None,
            }
            atomic_torch(best_file, best_payload)

        if executed % 10 == 0 or executed == CFG["base"]["max_epochs"] or wait > CFG["base"]["early_stop_wait_gt"]:
            atomic_torch(latest_file, {
                "experiment": CFG["experiment"], "group": group, "fold": fold,
                "source_sha256": expected_source_sha, "identity": identity,
                "warmup_sha256": identity["warmup_sha256"],
                "base_state": copy.deepcopy(E74.load_history().state_bundle(state)),
                "ecpsm_state": cpu_state(ecpsm), "tracker_state": cpu_state(tracker),
                "rng": runtime.pack_rng(), "next_epoch": epoch + 1, "wait": wait,
                "best_acc": best_acc, "best_epoch": best_epoch,
                "rng_trace": list(trace), "warmup_signatures": list(signatures),
                "hard_diagnostics": list(hard_diagnostics), "initial_memory": initial_memory,
            })
            record = {
                "group": group, "fold": fold, "epoch": epoch + 1,
                "best_epoch_1based": best_epoch + 1, "best_acc": best_acc,
                "metrics": _metric_payload(metrics), "loss": float(out["total"].detach().cpu()),
                "rng": epoch_rng, "hard_memory": step["hard_diagnostic"],
                "class_memory_active": int((ecpsm.class_counts > 0).sum().item()),
                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            }
            with log_path(group, fold).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
                stream.flush(); os.fsync(stream.fileno())
            append_log(record)
            print(f"{group} fold={fold} epoch={epoch + 1} best={best_acc:.6f}@{best_epoch + 1} wait={wait}", flush=True)

        del out, evaluated
        if wait > CFG["base"]["early_stop_wait_gt"]:
            break

    if best_epoch < 0 or not best_file.is_file():
        raise RuntimeError(f"no ACC-selected E77 checkpoint for {group} fold {fold}")
    best = torch.load(best_file, map_location="cpu", weights_only=False)
    task = {
        "status": "COMPLETE", "group": group, "fold": fold, "seed": 0,
        "source_sha256": expected_source_sha,
        "checkpoint_sha256": sha_file(best_file),
        "base_identity": identity,
        "best_epoch_1based": best_epoch + 1, "epochs_executed": executed,
        "metrics": best["metrics"], "rng_trace": trace,
        "warmup_signatures": signatures, "hard_diagnostics": hard_diagnostics,
        "memory_summary": ecpsm.diagnostics(),
        "initial_memory": initial_memory,
        "allowed_memory_ids": int(len(allowed)),
        "test_ids_excluded_from_memory": int(len(state["train_np"]) - len(allowed)),
        "train_test_overlap_1based": [i + 1 for i in overlap],
        "wall_seconds": time.time() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "device": torch.cuda.get_device_name(0),
    }
    atomic_json(task_file, task)
    print(f"COMPLETE {group} fold={fold} best={best_acc:.6f}@{best_epoch + 1}", flush=True)
    del best, state, ecpsm, tracker
    torch.cuda.empty_cache()
    return task


def _same_warmup_signatures(g2: list, g3: list) -> dict:
    keys = ("logp_sha256", "loss_sha256", "shared_state_sha256", "ecpsm_parameters_sha256",
            "class_memory_sha256", "hard_memory_sha256", "tracker_sha256")
    if len(g2) != 10 or len(g3) != 10:
        return {"status": "FAIL", "reason": f"expected 10 signatures each, got {len(g2)}/{len(g3)}"}
    mismatches = []
    rng_checks = 0
    for index, (left, right) in enumerate(zip(g2, g3), start=1):
        for key in keys:
            if left.get(key) != right.get(key):
                mismatches.append({"epoch": index, "field": key, "g2": left.get(key), "g3": right.get(key)})
        if left.get("rng") != right.get("rng"):
            mismatches.append({"epoch": index, "field": "rng", "g2": left.get("rng"), "g3": right.get("rng")})
        else:
            rng_checks += 1
    return {"status": "PASS" if not mismatches else "FAIL", "epochs": 10,
            "matching_rng_epochs": rng_checks, "mismatches": mismatches}


def train_fold(fold: int, expected_source_sha: str) -> dict:
    group_tasks = {}
    for group in GROUPS:
        group_tasks[group] = train_group(fold, group, expected_source_sha)
    parity = _same_warmup_signatures(group_tasks["E77.1-G2"]["warmup_signatures"],
                                     group_tasks["E77.1-G3"]["warmup_signatures"])
    if parity["status"] != "PASS":
        raise RuntimeError(f"G2/G3 common 10-epoch warmup parity failed at fold {fold}: {parity}")
    atomic_json(RESULTS / f"fold{fold:02d}_warmup_parity.json", parity)
    return {"fold": fold, "groups": group_tasks, "warmup_parity": parity}


def _run_all(workers: int) -> None:
    if workers < 1 or workers > 10:
        raise ValueError("E77.1 worker count must be in [1,10]")
    prepare_text_cache()
    preflight = subprocess.run([sys.executable, str(HERE / "audit.py"), "preflight"], cwd=HERE)
    if preflight.returncode:
        raise RuntimeError("E77.1 preflight failed; formal training was not started")
    preflight_record = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight_record.get("status") != "PASS":
        raise RuntimeError("E77.1 preflight record is not PASS")
    expected_source_sha = preflight_record["source_sha256"]
    pending = [fold for fold in range(1, 11)
               if any(not task_path(group, fold).is_file() for group in GROUPS)]
    attempts = {fold: 0 for fold in pending}
    failures = []
    while pending or any(item["process"].poll() is None for item in locals().get("active", [])):
        active = []
        while pending and len(active) < workers:
            fold = pending.pop(0)
            attempts[fold] += 1
            logfile = LOGS / f"fold{fold:02d}.stdout.log"
            logfile.parent.mkdir(parents=True, exist_ok=True)
            output = logfile.open("a", encoding="utf-8")
            env = os.environ.copy()
            env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "NUMEXPR_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
            process = subprocess.Popen([sys.executable, str(HERE / "run.py"), "--worker", "--fold", str(fold),
                                        "--source-sha", expected_source_sha],
                                       cwd=HERE, env=env, stdout=output, stderr=subprocess.STDOUT)
            active.append({"process": process, "fold": fold, "output": output})
        while active:
            pending_active = []
            for item in active:
                code = item["process"].poll()
                if code is None:
                    pending_active.append(item)
                    continue
                item["output"].close()
                fold = item["fold"]
                if code == 0 and all(task_path(group, fold).is_file() for group in GROUPS):
                    print(f"E77.1 fold {fold}/10 complete (attempt {attempts[fold]})", flush=True)
                elif attempts[fold] < 2:
                    failures.append({"fold": fold, "attempt": attempts[fold], "exit_code": code})
                    pending.append(fold)
                else:
                    raise RuntimeError(f"E77.1 fold worker failed after retry: fold={fold} exit={code}")
            if not pending_active:
                break
            active = pending_active
            time.sleep(1)
    manifest = {
        "worker_limit": workers, "fold_attempts": attempts, "failed_attempts": failures,
        "completed_group_folds": sum(task_path(group, fold).is_file() for group in GROUPS for fold in range(1, 11)),
    }
    atomic_json(RESULTS / "training_execution.json", manifest)
    if manifest["completed_group_folds"] != 30:
        raise RuntimeError(f"expected 30 completed group-fold tasks, found {manifest['completed_group_folds']}")
    subprocess.run([sys.executable, str(HERE / "audit.py"), "finalize"], cwd=HERE, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=CFG["execution"]["workers"])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--source-sha")
    args = parser.parse_args()
    for directory in (TASKS, LOGS, CHECKPOINTS, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if args.fold is None or not args.source_sha:
            raise ValueError("--worker requires --fold and --source-sha")
        source = source_sha()
        if source != args.source_sha:
            raise RuntimeError(f"worker source SHA changed after preflight: {source} != {args.source_sha}")
        result = train_fold(args.fold, source)
        print(json.dumps({"fold": args.fold, "warmup_parity": result["warmup_parity"]}, ensure_ascii=True), flush=True)
    elif args.all:
        _run_all(args.workers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
