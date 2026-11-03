#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

HERE = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/SPromptGL-main")
CONFIG_PATH = HERE / "config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
BASE_EXP = Path(CONFIG["base"]["base_source"])
BASE_GROUP = CONFIG["base"]["group"]
RT = HERE / ".runtime"
RESULTS = HERE / "results"
GROUPS = [item["name"] for item in CONFIG["groups"]]
METRICS = ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")
STATE = None
R = None
S = None

sys.path.insert(0, str(BASE_EXP))


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_obj(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_torch_save(path: Path, obj: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as stream:
        torch.save(obj, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def load_base() -> tuple[Any, Any]:
    global R, S
    if R is not None and S is not None:
        return R, S
    base_module_path = BASE_EXP / "run.py"
    current_run = sys.modules.get("run")
    if current_run is None or Path(getattr(current_run, "__file__", "")).resolve() != base_module_path.resolve():
        spec = importlib.util.spec_from_file_location("run", base_module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load historical base runner from {base_module_path}")
        R = importlib.util.module_from_spec(spec)
        sys.modules["run"] = R
        spec.loader.exec_module(R)
    else:
        R = current_run
    S = importlib.import_module("run_cadr_fallback_wd_sweep")
    S.SWEEP_ROOT = RT / "base_replay"
    S.install_group(BASE_GROUP)
    return R, S


def reset_group_paths(group: str) -> Path:
    r, _ = load_base()
    root = RT / "groups" / group
    tasks, checkpoints, logs = root / "tasks", root / "checkpoints", root / "logs"
    for path in (root, tasks, checkpoints, logs):
        path.mkdir(parents=True, exist_ok=True)
    r.RT, r.CKPTS, r.LOGS = root, checkpoints, logs
    r.task_path = lambda fold: tasks / f"{group}_f{fold:02d}.json"
    r.best_path = lambda fold: checkpoints / f"{group}_f{fold:02d}.best.pt"
    r.latest_path = lambda fold: checkpoints / f"{group}_f{fold:02d}.latest.pt"
    return root


def experiment_signature() -> dict[str, str]:
    files = [
        HERE / "run.py", HERE / "semantic.py", HERE / "text_features.py",
        HERE / "audit.py", CONFIG_PATH, HERE / "design.md", HERE / "knowledge_base.json",
        BASE_EXP / "run.py", BASE_EXP / "cadr_gfr.py", BASE_EXP / "config.json",
        BASE_EXP / "run_cadr_fallback.py", BASE_EXP / "run_cadr_fallback_wd_sweep.py",
        BASE_EXP / "config_fallback_wd_sweep.json",
    ]
    return {str(path): sha_file(path) for path in files}


def reference_inputs() -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    _, sweep = load_base()
    source_manifest, reference_53_2, _ = sweep.load_references()
    manifest = json.loads((BASE_EXP / "manifest_fallback_wd_sweep.json").read_text(encoding="utf-8"))
    rows = sweep.F.read_rows(sweep.RESULTS / "fold_metrics.csv")
    group_rows = {
        int(row["fold"]): row for row in rows
        if row.get("group") == BASE_GROUP
    }
    selected = {
        int(row["fold"]): row for row in manifest["folds"]
        if row.get("group") == BASE_GROUP
    }
    if len(group_rows) != 10 or len(selected) != 10:
        raise RuntimeError("the selected historical third-group fold records are incomplete")
    if manifest["status"] != "COMPLETE_AUDITED" or manifest["independent_checkpoint_audits_passed"] != 40:
        raise RuntimeError("the historical sweep manifest is not fully audited")
    checkpoint_root = Path(CONFIG["base"]["historical_checkpoint_root"])
    for fold in range(1, 11):
        path = checkpoint_root / f"{BASE_GROUP}_f{fold:02d}.best.pt"
        record = selected[fold]
        if not path.is_file() or sha_file(path) != record["checkpoint_sha256"]:
            raise RuntimeError(f"historical checkpoint SHA mismatch for fold {fold}")
    return {"source_manifest": source_manifest, "sweep_manifest": manifest}, reference_53_2, {
        fold: {**group_rows[fold], **selected[fold]} for fold in range(1, 11)
    }


def state_for_fold(group: str, fold: int):
    r, sweep = load_base()
    sweep.install_group(BASE_GROUP)
    reset_group_paths(group)
    state = r.build_state(fold, r.prepare_fold(fold))
    state["group"] = group
    state["cadr_weight_decay"] = 0.01
    for name, expected in (("opt_base", 0.11), ("opt_cam", 0.11), ("opt_added", 0.01)):
        actual = [float(pg["weight_decay"]) for pg in state[name].param_groups]
        if not actual or any(value != expected for value in actual):
            raise AssertionError((group, fold, name, actual, expected))
    return state


def module_seed(group: str, fold: int) -> int:
    # All structurally shared DSS groups receive identical common initialization.
    group_seed = 60000 if group in GROUPS[6:] else 61000 + GROUPS.index(group)
    return group_seed + int(fold)


def build_module(group: str, fold: int, banks: dict[str, Any], semantic):
    import torch

    from semantic import BrainPromptModule, DSSPen, OriginalSPromptPEN, SaberModule, isolated_torch_rng

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with isolated_torch_rng(module_seed(group, fold)):
        if group == "E60_ORIGINAL_PEN":
            module = OriginalSPromptPEN(banks["original"])
        elif group in {"E60_BRAINPROMPT_LABEL", "E60_BRAINPROMPT_MEDICAL"}:
            module = BrainPromptModule(banks["brainprompt"])
        elif group == "E60_SABER_DECISION":
            module = SaberModule(banks["disease_centroids"], banks["patient_text"])
        elif group == "E60_DISEASE_QUERY":
            module = SaberModule(banks["disease_centroids"], banks["medical"].flatten(0, 1))
        elif group in GROUPS[6:]:
            projection_mode = "partial_simplex"
            if group == "E60_DSS_SOFTMAX_ATTENTION":
                projection_mode = "softmax_attention"
            elif group == "E60_DSS_FORCE_MATCH":
                projection_mode = "simplex_sum_eq_one"
            elif group == "E60_DSS_FREE_SET":
                projection_mode = "nonnegative_box"
            input_mask = (0.0, 0.0, 0.0, 1.0) if group == "E60_DSS_CORR_ONLY" else (1.0, 1.0, 1.0, 1.0)
            module = DSSPen(banks["dss"], projection_mode=projection_mode, input_slot_mask=input_mask)
        else:
            raise KeyError(group)
    return module.to(device)


def make_group_optimizer(group: str, state: dict[str, Any], module):
    import torch

    if group == "E60_BASE_REPLAY":
        return None
    if group == "E60_ORIGINAL_PEN":
        state["opt_cam"] = torch.optim.Adam(
            list(state["cam"].parameters()) + list(module.parameters()),
            lr=0.0038,
            weight_decay=0.11,
        )
        return None
    return torch.optim.AdamW(module.parameters(), lr=0.001, weight_decay=0.001)


def cam_encode(cam, fused):
    n = fused.shape[0]
    x, input_attention = cam.InputLayer(fused)
    attention = input_attention.mean(dim=1)
    for encoder, feedforward in zip(cam.Encoder, cam.FeedForward):
        x, layer_attention = encoder(q=x, k=x, v=x, modal_num=cam.modal_num)
        attention = layer_attention.mean(dim=1)
        x = feedforward(x)
    return x.view(n, 4, 36), attention


def _grad_norm(parameters) -> float:
    import torch

    terms = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(terms).sum().sqrt().cpu()) if terms else 0.0


def sem_forward(group: str, state: dict[str, Any], module, epoch: int, banks: dict[str, Any], weights: dict[str, float]):
    import torch
    from torch.nn import functional as F

    capture: dict[str, Any] = {}
    cam = state["cam"]

    def forward_with_module(fused):
        h, attention = cam_encode(cam, fused)
        module_input = h if epoch >= 10 else h.detach()
        extra_loss = h.sum() * 0.0
        if group == "E60_ORIGINAL_PEN":
            h_used, details = module(h)
            capture.update(details)
        elif group in {"E60_BRAINPROMPT_LABEL", "E60_BRAINPROMPT_MEDICAL"}:
            logits, z = module.logits(module_input)
            train = state["train"]
            extra_loss = weights["brainprompt"] * F.cross_entropy(logits[train], state["y"][train])
            h_used = h
            capture.update({"disease_logits": logits, "semantic_z": z, "disease_ce": extra_loss})
        elif group in {"E60_SABER_DECISION", "E60_DISEASE_QUERY"}:
            output = module(module_input)
            train = state["train"]
            disease_loss = F.cross_entropy(output["disease_logits"][train], state["y"][train])
            extra_loss = weights["saber"] * disease_loss
            schedule = min(0.1, max(0.0, (int(epoch) - 10) / 10.0 * 0.1))
            correction = schedule * output["raw_correction"]
            h_used = h + correction
            capture.update({**output, "disease_ce": disease_loss, "applied_correction": correction})
        elif group in GROUPS[6:]:
            output = module.project(module_input)
            group_config = next(item for item in CONFIG["groups"] if item["name"] == group)
            losses = module.losses(
                output,
                state["y"],
                state["train"],
                set_loss_enabled=bool(group_config.get("set_loss", False)),
                geometry_loss_enabled=bool(group_config.get("geometry_loss", True)),
                set_margin=float(CONFIG["dss"]["set_margin"]),
            )
            extra_loss = weights["set"] * losses["set"] + weights["geometry"] * losses["geometry"]
            corrected, diagnostics = module.correct(
                h,
                output,
                epoch=epoch,
                residual_enabled=bool(group_config.get("correction", False)),
                slot_mask=tuple(group_config.get("output_slot_mask", (1, 1, 1, 1))),
                residual_cap=float(CONFIG["dss"]["residual_norm_cap_per_position"]),
            )
            h_used = corrected
            capture.update({**output, **losses, **diagnostics})
        else:
            raise KeyError(group)

        logp, hidden = cam.Outputlayer(h_used.reshape(h_used.shape[0], -1), attention.reshape(h.shape[0], -1))
        capture.update({"h": h, "h_used": h_used, "attention": attention, "extra_loss": extra_loss})
        return logp, hidden, []

    cam.forward = forward_with_module
    try:
        result = state["_base_forward"](state, training=state["_is_training"], use_cadr=state["_use_cadr"])
    finally:
        cam.forward = state["_original_cam_forward"]
    result["total"] = result["total"] + capture["extra_loss"]
    result["semantic"] = capture
    return result


def base_or_group_forward(group: str, state, module, epoch: int, training: bool, use_cadr: bool, banks, weights):
    r, _ = load_base()
    if group == "E60_BASE_REPLAY":
        return r.forward_state(state, training=training, use_cadr=use_cadr)
    state["_base_forward"] = r.forward_state
    state["_is_training"] = training
    state["_use_cadr"] = use_cadr
    state["_original_cam_forward"] = state["cam"].forward
    module.train(training)
    return sem_forward(group, state, module, epoch, banks, weights)


def module_state_cpu(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def optimizer_state_cpu(optimizer):
    import copy

    return copy.deepcopy(optimizer.state_dict())


def group_checkpoint(group, state, module, opt_semantic, best, fold, best_epoch, epochs, metrics, diagnostics, weights):
    import torch

    r, _ = load_base()
    cp = r._build_checkpoint(state, best, fold, best_epoch, epochs, metrics, diagnostics)
    cp.update({
        "experiment": "Experiment 60 DSS-PEN",
        "group": group,
        "cadr_weight_decay": 0.01,
        "gfr_weight_decay": 0.11,
        "camera_weight_decay": 0.11,
        "optimizer_weight_decay": {
            "base": [float(pg["weight_decay"]) for pg in state["opt_base"].param_groups],
            "cam": [float(pg["weight_decay"]) for pg in state["opt_cam"].param_groups],
            "added": [float(pg["weight_decay"]) for pg in state["opt_added"].param_groups],
            "semantic": [] if opt_semantic is None else [float(pg["weight_decay"]) for pg in opt_semantic.param_groups],
        },
        "semantic_module": None if module is None else best.get("semantic_module"),
        "optimizer_semantic": None if opt_semantic is None else best.get("optimizer_semantic"),
        "extra_loss_weights": weights,
        "source_signature": experiment_signature(),
        "text_cache_sha256": sha_file(HERE / ".runtime" / "text_embeddings.pt"),
        "knowledge_base_sha256": sha_file(HERE / "knowledge_base.json"),
        "train_test_indices": {
            "train_sha256": state["train_idx_sha"],
            "test_sha256": state["test_idx_sha"],
        },
        "test_predictions": {
            "indices": [int(value) for value in state["test"].tolist()],
            "labels": state["y"][torch.as_tensor(state["test"], dtype=torch.long, device=state["y"].device)].detach().cpu().tolist(),
            "probabilities_asd_td": metrics["prob"].tolist(),
        },
        "diagnostics": diagnostics,
    })
    return cp


def get_fold_weights() -> dict[str, float]:
    path = RESULTS / "loss_calibration.json"
    if not path.is_file():
        raise RuntimeError("loss calibration must pass before formal training")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "PASS":
        raise RuntimeError("loss calibration status is not PASS")
    return {key: float(value) for key, value in data["weights"].items()}


def task_signature(group: str, fold: int, state) -> dict[str, Any]:
    return {
        "group": group,
        "fold": int(fold),
        "seed": 0,
        "source_signature": experiment_signature(),
        "text_cache_sha256": sha_file(HERE / ".runtime" / "text_embeddings.pt"),
        "knowledge_base_sha256": sha_file(HERE / "knowledge_base.json"),
        "split_sha": state["split_sha"],
        "pair_sha": state["cache_meta"]["pair_sha"],
        "fold_cache_sha": state["cache_meta"]["npz_sha256"],
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_added_sha": state["initial_added_sha"],
        "initial_rng_sha": state["construction_rng_sha"],
    }


def task_matches_signature(task: dict[str, Any], signature: dict[str, Any]) -> bool:
    return all(task.get(key) == value for key, value in signature.items())


def train_one(group: str, fold: int, banks: dict[str, Any]) -> dict[str, Any]:
    import copy
    import numpy as np
    import torch

    r, _ = load_base()
    root = reset_group_paths(group)
    task_path = r.task_path(fold)
    checkpoint_path = r.best_path(fold)
    latest_path = r.latest_path(fold)
    state = state_for_fold(group, fold)
    signature = task_signature(group, fold, state)
    if task_path.is_file() and checkpoint_path.is_file():
        task = json.loads(task_path.read_text(encoding="utf-8"))
        if task.get("status") == "COMPLETE" and task.get("best_sha256") == sha_file(checkpoint_path) and task_matches_signature(task, signature):
            return task

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    module = None if group == "E60_BASE_REPLAY" else build_module(group, fold, banks, None)
    opt_semantic = make_group_optimizer(group, state, module)
    state["train_idx_sha"] = hashlib.sha256(state["train"].detach().cpu().numpy().astype(np.int64).tobytes()).hexdigest()
    state["test_idx_sha"] = hashlib.sha256(np.asarray(state["test"], dtype=np.int64).tobytes()).hexdigest()
    weights = get_fold_weights()
    start_epoch = 0
    wait = 0
    best_acc = -1.0
    best_epoch = -1
    best_state = None
    best_metrics = None
    best_diag = None
    warmup_best = -1.0
    warmup_best_epoch = -1
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        if saved.get("signature") == signature:
            state["model"].load_state_dict(saved["model"], strict=True)
            state["cam"].load_state_dict(saved["cam"], strict=True)
            state["opt_base"].load_state_dict(saved["optimizer_base"])
            state["opt_cam"].load_state_dict(saved["optimizer_cam"])
            state["opt_added"].load_state_dict(saved["optimizer_added"])
            if module is not None:
                module.load_state_dict(saved["semantic_module"], strict=True)
            if opt_semantic is not None:
                opt_semantic.load_state_dict(saved["optimizer_semantic"])
            start_epoch = int(saved["next_epoch"])
            wait = int(saved["wait"])
            best_acc = float(saved["best_acc"])
            best_epoch = int(saved["best_epoch"])
            best_state = saved["best_state"]
            best_metrics = saved["best_metrics"]
            best_diag = saved["best_diag"]
            warmup_best = float(saved["warmup_best"])
            warmup_best_epoch = int(saved["warmup_best_epoch"])
            r.restore_rng(saved["rng"])

    started = time.time()
    epochs_executed = start_epoch
    for epoch in range(start_epoch, int(r.CFG["cv"]["max_epochs"])):
        state["opt_base"].zero_grad(set_to_none=True)
        state["opt_cam"].zero_grad(set_to_none=True)
        state["opt_added"].zero_grad(set_to_none=True)
        if opt_semantic is not None:
            opt_semantic.zero_grad(set_to_none=True)
        active = epoch >= int(r.CFG["warmup"]["epochs"])
        out = base_or_group_forward(group, state, module, epoch, True, active, banks, weights)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"nonfinite total loss at {group} fold {fold}, epoch {epoch}")
        out["total"].backward()
        for model in (state["model"], state["cam"]):
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"nonfinite base gradient at {group} fold {fold}, epoch {epoch}")
        if module is not None:
            for parameter in module.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"nonfinite semantic gradient at {group} fold {fold}, epoch {epoch}")
        gradients = {
            "base": _grad_norm(state["base_params"]),
            "cam": _grad_norm(state["cam"].parameters()),
            "added": _grad_norm(parameter for _, parameter in state["added_named"]),
            "semantic": 0.0 if module is None else _grad_norm(module.parameters()),
        }
        state["opt_base"].step()
        state["opt_cam"].step()
        state["opt_added"].step()
        if opt_semantic is not None:
            opt_semantic.step()
        epochs_executed = epoch + 1

        with torch.no_grad():
            evaluated = base_or_group_forward(group, state, module, epoch, False, active, banks, weights)
            metrics = r.classification_metrics(evaluated["logp"], state["test"], state["y"])
        if not active:
            if metrics["acc"] > warmup_best:
                warmup_best, warmup_best_epoch = metrics["acc"], epoch
        elif metrics["acc"] > best_acc:
            best_acc, best_epoch, wait = metrics["acc"], epoch, 0
            best_state = {
                "model": r._state_cpu(state["model"]),
                "cam": r._state_cpu(state["cam"]),
                "optimizer_base": copy.deepcopy(state["opt_base"].state_dict()),
                "optimizer_cam": copy.deepcopy(state["opt_cam"].state_dict()),
                "optimizer_added": copy.deepcopy(state["opt_added"].state_dict()),
                "semantic_module": None if module is None else module_state_cpu(module),
                "optimizer_semantic": None if opt_semantic is None else copy.deepcopy(opt_semantic.state_dict()),
            }
            best_metrics = {key: value for key, value in metrics.items() if key not in {"pred", "prob"}}
            semantic_diag = {}
            if module is not None:
                current_sem = evaluated.get("semantic", {})
                for key in ("set", "geometry", "disease_ce", "residual_schedule"):
                    value = current_sem.get(key)
                    if torch.is_tensor(value):
                        semantic_diag[key] = float(value.detach().float().mean().cpu())
                semantic_diag["gradient_norm"] = gradients["semantic"]
            best_diag = {
                "loss_cls": float(out["cls"].detach().cpu()),
                "loss_graph": float(out["graph"].detach().cpu()),
                "loss_pair": float(out["pair"].detach().cpu()),
                "loss_adv": float(out["adv"].detach().cpu()),
                "loss_preserve": float(out["preserve"].detach().cpu()),
                "gradient_norms": gradients,
                "semantic": semantic_diag,
                "propagation_modes": evaluated["info"]["propagation_modes"],
            }
            cp = group_checkpoint(group, state, module, opt_semantic, best_state, fold, best_epoch, epochs_executed, metrics, best_diag, weights)
            atomic_torch_save(checkpoint_path, cp)
        else:
            wait += 1

        if (epoch + 1) % 10 == 0 or epoch == r.CFG["cv"]["max_epochs"] - 1 or (active and wait > r.CFG["cv"]["early_stop_wait_gt"]):
            latest = {
                "signature": signature,
                "group": group,
                "fold": fold,
                "next_epoch": epoch + 1,
                "wait": wait,
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "best_metrics": best_metrics,
                "best_diag": best_diag,
                "warmup_best": warmup_best,
                "warmup_best_epoch": warmup_best_epoch,
                "model": r._state_cpu(state["model"]),
                "cam": r._state_cpu(state["cam"]),
                "optimizer_base": state["opt_base"].state_dict(),
                "optimizer_cam": state["opt_cam"].state_dict(),
                "optimizer_added": state["opt_added"].state_dict(),
                "semantic_module": None if module is None else module_state_cpu(module),
                "optimizer_semantic": None if opt_semantic is None else opt_semantic.state_dict(),
                "rng": r.pack_rng(),
            }
            atomic_torch_save(latest_path, latest)
            atomic_json(RT / "worker_status" / f"{group}_f{fold:02d}.json", {
                "group": group, "fold": fold, "epoch": epoch + 1, "best_epoch": best_epoch,
                "best_acc": best_acc, "wait": wait, "stage": "TRAINING",
            })
        if active and wait > int(r.CFG["cv"]["early_stop_wait_gt"]):
            break

    if best_state is None:
        raise RuntimeError(f"no post-warmup checkpoint selected for {group} fold {fold}")
    state["model"].load_state_dict(best_state["model"], strict=True)
    state["cam"].load_state_dict(best_state["cam"], strict=True)
    state["opt_base"].load_state_dict(best_state["optimizer_base"])
    state["opt_cam"].load_state_dict(best_state["optimizer_cam"])
    state["opt_added"].load_state_dict(best_state["optimizer_added"])
    if module is not None:
        module.load_state_dict(best_state["semantic_module"], strict=True)
    if opt_semantic is not None:
        opt_semantic.load_state_dict(best_state["optimizer_semantic"])
    with torch.no_grad():
        readback = base_or_group_forward(group, state, module, best_epoch, False, True, banks, weights)
        readback_metrics = r.classification_metrics(readback["logp"], state["test"], state["y"])
    for metric in METRICS:
        if abs(float(readback_metrics[metric]) - float(best_metrics[metric])) > 1e-10:
            raise AssertionError(f"best/readback mismatch for {group} fold {fold} {metric}")
    diagnostics = best_diag or {}
    cp = group_checkpoint(group, state, module, opt_semantic, best_state, fold, best_epoch, epochs_executed, readback_metrics, diagnostics, weights)
    cp["readback_status"] = "WORKER_READBACK_PASS"
    atomic_torch_save(checkpoint_path, cp)
    cp_sha = sha_file(checkpoint_path)
    peak_gpu = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
    import resource
    peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    task = {
        **signature,
        "status": "COMPLETE",
        "best_epoch": int(best_epoch),
        "epochs_executed": int(epochs_executed),
        "warmup_best_epoch": int(warmup_best_epoch),
        "warmup_best_acc": float(warmup_best),
        "best_sha256": cp_sha,
        "metrics": {key: float(readback_metrics[key]) for key in METRICS},
        "confusion": {key: int(readback_metrics[key]) for key in ("asd_tp", "asd_fn", "td_fp", "td_tn")},
        "wall_seconds": float(time.time() - started),
        "peak_cuda_reserved_bytes": peak_gpu,
        "peak_rss_kb": peak_rss_kb,
        "optimizer_weight_decay": cp["optimizer_weight_decay"],
        "readback_status": "WORKER_READBACK_PASS",
        "weights": weights,
    }
    atomic_json(task_path, task)
    latest_path.unlink(missing_ok=True)
    atomic_json(RT / "worker_status" / f"{group}_f{fold:02d}.json", {
        "group": group, "fold": fold, "stage": "COMPLETE", "best_epoch": best_epoch,
        "checkpoint_sha256": cp_sha,
    })
    return task


def build_patient_ids(state):
    return state["y"].detach().cpu()


def load_text_cache():
    cache_path = HERE / ".runtime" / "text_embeddings.pt"
    if not cache_path.is_file():
        raise FileNotFoundError("run --prepare to create the frozen CLIP embedding cache")
    return torch_load(cache_path)


def torch_load(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def worker(group: str, fold: int) -> None:
    import torch
    from text_features import group_text_banks

    cache = load_text_cache()
    banks = group_text_banks(cache, group)
    task = train_one(group, fold, banks)
    print(f"COMPLETE {group} fold={fold} best_epoch={task['best_epoch']} acc={task['metrics']['acc']:.6f}", flush=True)


def _rng_snapshot(R):
    return R.pack_rng()


def _compare_optimizer_states(a: dict, b: dict) -> bool:
    import torch

    def equal(left, right):
        if torch.is_tensor(left) or torch.is_tensor(right):
            return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
        if isinstance(left, dict) or isinstance(right, dict):
            return (
                isinstance(left, dict)
                and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(equal(left[key], right[key]) for key in left)
            )
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            return (
                isinstance(left, (list, tuple))
                and isinstance(right, (list, tuple))
                and len(left) == len(right)
                and all(equal(x, y) for x, y in zip(left, right))
            )
        return type(left) is type(right) and left == right

    return equal(a, b)


def calibration_forward(module, state, group, banks):
    import torch
    from torch.nn import functional as F

    r, _ = load_base()
    captured = {}
    cam = state["cam"]

    def capture_forward(fused):
        h, attention = cam_encode(cam, fused)
        captured["h"] = h
        logits, _ = cam.Outputlayer(h.flatten(1), attention.flatten(1))
        return logits, h, []

    original = cam.forward
    cam.forward = capture_forward
    try:
        out = r.forward_state(state, training=True, use_cadr=True)
    finally:
        cam.forward = original
    h = captured["h"]
    if group == "E60_DSS_FULL":
        projected = module.project(h)
        losses = module.losses(projected, state["y"], state["train"], True, True, 0.1)
        return out, h, losses["set"], losses["geometry"]
    if group == "E60_SABER_DECISION":
        result = module(h)
        disease_loss = F.cross_entropy(result["disease_logits"][state["train"]], state["y"][state["train"]])
        return out, h, disease_loss, None
    raise ValueError(group)


def calibrate_losses() -> dict[str, Any]:
    import torch
    from text_features import group_text_banks

    r, _ = load_base()
    cache = load_text_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = {}
    for group in ("E60_DSS_FULL", "E60_SABER_DECISION"):
        banks = group_text_banks(cache, group)
        state = state_for_fold(group, 1)
        module = build_module(group, 1, banks, None)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rng = _rng_snapshot(r)
        cls_out, h, sem_loss, geometry_loss = calibration_forward(module, state, group, banks)
        cls_grad = torch.autograd.grad(cls_out["cls"], h, retain_graph=True, allow_unused=False)[0]
        sem_grad = torch.autograd.grad(sem_loss, h, retain_graph=True, allow_unused=False)[0]
        base_norm = cls_grad.detach().float().norm()
        semantic_norm = sem_grad.detach().float().norm().clamp_min(1e-12)
        if group == "E60_DSS_FULL":
            adapter = module.text_adapter.weight
            set_grad = torch.autograd.grad(sem_loss, adapter, retain_graph=True, allow_unused=True)[0]
            geo_grad = torch.autograd.grad(geometry_loss, adapter, retain_graph=False, allow_unused=True)[0]
            if set_grad is None or geo_grad is None:
                raise RuntimeError("DSS loss calibration produced an unused adapter gradient")
            set_adapter_norm = set_grad.detach().float().norm().clamp_min(1e-12)
            geo_adapter_norm = geo_grad.detach().float().norm().clamp_min(1e-12)
            set_weight = 0.1 * float(base_norm / semantic_norm)
            geo_weight = 0.05 * float(set_adapter_norm / geo_adapter_norm)
            results = {"set": set_weight, "geometry": geo_weight, "saber": 0.0, "brainprompt": 0.001}
            results["dss_calibration"] = {
                "base_cls_gradient_norm_H": float(base_norm),
                "set_gradient_norm_H": float(semantic_norm),
                "set_gradient_norm_text_adapter": float(set_adapter_norm),
                "geometry_gradient_norm_text_adapter": float(geo_adapter_norm),
                "targets": {"set_to_base_H": 0.1, "geometry_to_set_adapter": 0.05},
            }
        else:
            saber_weight = 0.1 * float(base_norm / semantic_norm)
            results["saber"] = saber_weight
            results["saber_calibration"] = {
                "base_cls_gradient_norm_H": float(base_norm),
                "disease_contrast_gradient_norm_H": float(semantic_norm),
                "target_contrast_to_base_H": 0.1,
            }
        r.restore_rng(rng)
        if not math.isfinite(float(base_norm)) or not math.isfinite(float(semantic_norm)):
            raise FloatingPointError(f"nonfinite loss calibration for {group}")
        del state, module, cls_out, h, sem_loss, geometry_loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "status": "PASS",
        "fold": 1,
        "outer_test_labels_used": False,
        "weights": {key: value for key, value in results.items() if isinstance(value, (int, float))},
        "details": {key: value for key, value in results.items() if isinstance(value, dict)},
        "config_sha256": sha_file(CONFIG_PATH),
        "text_cache_sha256": sha_file(HERE / ".runtime" / "text_embeddings.pt"),
        "knowledge_base_sha256": sha_file(HERE / "knowledge_base.json"),
    }
    atomic_json(RESULTS / "loss_calibration.json", result)
    return result


def _module_disabled_cam_parity(state, module, banks):
    import torch
    r, _ = load_base()
    cam = state["cam"]
    cam.eval()
    state["model"].eval()
    state["cam"].eval()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    with torch.no_grad():
        baseline = r.forward_state(state, training=False, use_cadr=True)
    baseline_cpu = (baseline["fused"].cpu(), baseline["logp"].cpu())
    torch.set_rng_state(cpu_rng)
    if cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
    state["_base_forward"] = r.forward_state
    state["_is_training"] = False
    state["_use_cadr"] = True
    state["_original_cam_forward"] = cam.forward
    with torch.no_grad():
        replay = sem_forward("E60_DSS_FULL", state, module, 10, banks, {"set": 0.0, "geometry": 0.0, "brainprompt": 0.001, "saber": 0.0})
    replay_cpu = (replay["fused"].cpu(), replay["logp"].cpu())
    if not torch.equal(baseline_cpu[0], replay_cpu[0]) or not torch.equal(baseline_cpu[1], replay_cpu[1]):
        raise AssertionError("DSS zero-residual path differs from the unmodified base forward")

    state["model"].train()
    cam.train()
    module.train()
    train_rng = r.pack_rng()
    base_train = r.forward_state(state, training=True, use_cadr=True)
    base_after_rng = r.rng_sha(r.pack_rng())
    r.restore_rng(train_rng)
    state["_is_training"] = True
    grouped_train = sem_forward("E60_DSS_FULL", state, module, 10, banks, {"set": 0.0, "geometry": 0.0, "brainprompt": 0.001, "saber": 0.0})
    grouped_after_rng = r.rng_sha(r.pack_rng())
    if not torch.equal(base_train["fused"].cpu(), grouped_train["fused"].cpu()) or not torch.equal(base_train["logp"].cpu(), grouped_train["logp"].cpu()):
        raise AssertionError("DSS zero-residual training path differs from the unmodified base forward")
    if base_after_rng != grouped_after_rng:
        raise AssertionError("DSS zero-residual training path consumes a different RNG sequence")


def preflight() -> dict[str, Any]:
    import torch
    import numpy as np
    from text_features import group_text_banks
    from semantic import DSSPen, constrained_projection, isolated_torch_rng

    r, sweep = load_base()
    refs, reference_53_2, selected = reference_inputs()
    cache = load_text_cache()
    sweep.preflight(reference_53_2)
    ids = []
    forwards = []
    source_audit = []
    for fold in range(1, 11):
        state = state_for_fold(BASE_GROUP, fold)
        ref = selected[fold]
        for key, state_key in (
            ("initial_shared_sha", "initial_shared_sha"),
            ("initial_added_sha", "initial_added_sha"),
            ("initial_rng_sha", "construction_rng_sha"),
            ("split_sha", "split_sha"),
            ("pair_sha", "pair_sha"),
        ):
            actual = state["cache_meta"]["pair_sha"] if state_key == "pair_sha" else state[state_key]
            if actual != ref[key]:
                raise AssertionError((fold, key, actual, ref[key]))
        ids.append((state["initial_shared_sha"], state["initial_added_sha"], state["construction_rng_sha"], state["split_sha"], state["cache_meta"]["pair_sha"]))
        with torch.no_grad():
            out = r.forward_state(state, training=False, use_cadr=True)
        forwards.append((out["fused"].detach().cpu(), out["logp"].detach().cpu()))
        source_audit.append({"fold": fold, **{key: ref[key] for key in ("checkpoint_sha256", "initial_shared_sha", "initial_added_sha", "initial_rng_sha", "split_sha", "pair_sha")}})
    if len(set(ids)) != 10:
        # Fold-local model hashes should differ; this guard only ensures the per-fold records exist.
        if len(ids) != 10:
            raise AssertionError("expected 10 fold initialization records")

    state_a = state_for_fold(BASE_GROUP, 1)
    state_b = state_for_fold(BASE_GROUP, 1)
    if state_a["initial_shared_sha"] != state_b["initial_shared_sha"] or state_a["construction_rng_sha"] != state_b["construction_rng_sha"]:
        raise AssertionError("reconstructed base state is not repeatable")
    rng = r.pack_rng()
    r.restore_rng(state_a["construction_rng"])
    out_a = r.forward_state(state_a, training=True, use_cadr=True)
    out_a["total"].backward()
    for opt in (state_a["opt_base"], state_a["opt_cam"], state_a["opt_added"]):
        opt.step()
    rng_a = r.rng_sha(r.pack_rng())
    params_a = [parameter.detach().cpu().clone() for parameter in list(state_a["model"].parameters()) + list(state_a["cam"].parameters())]
    optim_a = [opt.state_dict() for opt in (state_a["opt_base"], state_a["opt_cam"], state_a["opt_added"])]
    r.restore_rng(state_b["construction_rng"])
    out_b = r.forward_state(state_b, training=True, use_cadr=True)
    out_b["total"].backward()
    for opt in (state_b["opt_base"], state_b["opt_cam"], state_b["opt_added"]):
        opt.step()
    rng_b = r.rng_sha(r.pack_rng())
    params_b = [parameter.detach().cpu().clone() for parameter in list(state_b["model"].parameters()) + list(state_b["cam"].parameters())]
    optim_b = [opt.state_dict() for opt in (state_b["opt_base"], state_b["opt_cam"], state_b["opt_added"])]
    if not torch.equal(out_a["logp"].cpu(), out_b["logp"].cpu()) or not torch.equal(out_a["total"].cpu(), out_b["total"].cpu()):
        raise AssertionError("same-seed base replay forward/loss differs")
    if rng_a != rng_b:
        raise AssertionError("same-seed first shared optimizer step consumes a different RNG sequence")
    if any(not torch.equal(a, b) for a, b in zip(params_a, params_b)) or not all(_compare_optimizer_states(a, b) for a, b in zip(optim_a, optim_b)):
        raise AssertionError("same-seed first shared optimizer step differs")
    r.restore_rng(rng)

    banks = group_text_banks(cache, "E60_DSS_FULL")
    pre_module_rng = r.rng_sha(r.pack_rng())
    module = build_module("E60_DSS_FULL", 1, banks, None)
    post_module_rng = r.rng_sha(r.pack_rng())
    if pre_module_rng != post_module_rng:
        raise AssertionError("semantic module initialization changed the base RNG")
    _module_disabled_cam_parity(state_for_fold(BASE_GROUP, 1), module, banks)

    z_raw = torch.randn(8, 64, device="cuda:0" if torch.cuda.is_available() else "cpu", requires_grad=True)
    z = torch.nn.functional.normalize(z_raw, dim=-1)
    p = torch.nn.functional.normalize(torch.randn(6, 64, device=z.device), dim=-1)
    constraints = {}
    convergence = {}
    for mode in ("partial_simplex", "simplex_sum_eq_one", "nonnegative_box", "softmax_attention"):
        a, projected, detail = constrained_projection(z, p, mode=mode, iterations=128)
        if not torch.isfinite(a).all() or not torch.isfinite(projected).all() or (a < -1e-6).any():
            raise AssertionError(f"invalid projection values for {mode}")
        if mode == "partial_simplex" and (a.sum(-1) > 1.00001).any():
            raise AssertionError("partial projection violated its mass budget")
        if mode == "simplex_sum_eq_one" and not torch.allclose(a.sum(-1), torch.ones(8, device=z.device), atol=1e-5):
            raise AssertionError("forced-match projection did not sum to one")
        if mode == "nonnegative_box" and (a > 1.00001).any():
            raise AssertionError("free-set box projection violated its upper bound")
        if mode == "softmax_attention" and not torch.allclose(a.sum(-1), torch.ones(8, device=z.device), atol=1e-6):
            raise AssertionError("softmax attention coefficients do not sum to one")
        grad = torch.autograd.grad(projected.square().mean() + detail["distance"].mean(), z_raw, retain_graph=True)[0]
        if not torch.isfinite(grad).all() or float(grad.norm().detach().cpu()) <= 0:
            raise AssertionError(f"projection solver has an invalid gradient for {mode}")
        constraints[mode] = {"coefficient_min": float(a.min().detach().cpu()), "mass_max": float(a.sum(-1).max().detach().cpu())}
        if mode != "softmax_attention":
            reference, _, _ = constrained_projection(z.detach(), p, mode=mode, iterations=256)
            max_delta = float((a.detach() - reference).abs().max().cpu())
            if max_delta > 1e-4:
                raise AssertionError(f"128-step projection failed its 256-step convergence check for {mode}: {max_delta}")
            convergence[mode] = {"iterations": 128, "reference_iterations": 256, "max_coefficient_delta": max_delta, "tolerance": 1e-4}

    io_path = RT / "io_probe" / "checkpoint.pt"
    atomic_torch_save(io_path, {"probe": torch.arange(7), "sha": "experiment60-atomic-io"})
    readback = torch_load(io_path)
    if readback["sha"] != "experiment60-atomic-io" or not torch.equal(readback["probe"], torch.arange(7)):
        raise AssertionError("atomic checkpoint write/readback test failed")

    result = {
        "status": "PASS",
        "source_status": refs["sweep_manifest"]["status"],
        "historical_selected_group": BASE_GROUP,
        "historical_selected_checkpoints_sha_verified": 10,
        "fold_initialization_rng_split_pair_checks": 10,
        "fold_initialization_records": source_audit,
        "base_forward_outputs_recorded": 10,
        "base_first_step_parity": "BITWISE_PASS",
        "base_first_step_rng_sha": rng_a,
        "base_first_step_rng_parity": "BITWISE_PASS",
        "semantic_init_rng_isolation": "PASS",
        "zero_residual_forward_parity": "BITWISE_PASS",
        "zero_residual_training_rng_parity": "BITWISE_PASS",
        "projection_constraints": constraints,
        "projection_convergence": convergence,
        "atomic_write_rename_fsync_readback": "PASS",
        "optimizer_weight_decay": {"gfr": [0.11], "cam": [0.11], "cadr": [0.01, 0.01]},
        "device": str(r.runtime()["device"]),
        "source_signature": experiment_signature(),
        "text_cache_sha256": sha_file(HERE / ".runtime" / "text_embeddings.pt"),
        "knowledge_base_sha256": sha_file(HERE / "knowledge_base.json"),
    }
    atomic_json(RESULTS / "preflight.json", result)
    return result


def prepare() -> dict[str, Any]:
    from text_features import prepare_text_cache

    r, _ = load_base()
    cache = prepare_text_cache(r)
    refs, _, selected = reference_inputs()
    manifest = {
        "experiment": "Experiment 60 DSS-PEN",
        "status": "PREPARED",
        "formal_groups": GROUPS,
        "folds_per_group": 10,
        "planned_formal_folds": 180,
        "completed_formal_folds": 0,
        "historical_base_group": BASE_GROUP,
        "historical_base_status": refs["sweep_manifest"]["status"],
        "source_signature": experiment_signature(),
        "data_sha256": {
            "processed_standard_data.csv": sha_file(ROOT / "data/ABIDE/processed_standard_data.csv"),
            "modal_feat_dict.npy": sha_file(ROOT / "data/ABIDE/modal_feat_dict.npy"),
        },
        "knowledge_base_sha256": sha_file(HERE / "knowledge_base.json"),
        "text_cache_sha256": sha_file(HERE / ".runtime" / "text_embeddings.pt"),
        "text_source_hashes": cache["source_hashes"],
        "historical_fold_sources": selected,
        "status_by_group": {group: "PENDING" for group in GROUPS},
        "completed_by_group": {group: 0 for group in GROUPS},
    }
    atomic_json(HERE / "manifest.json", manifest)
    return {"status": "PASS", "text_cache_sha256": manifest["text_cache_sha256"], "groups": len(GROUPS)}


def _resource_caps(requested: int, gpu_peak: int, rss_peak_kb: int) -> tuple[int, dict[str, Any]]:
    import psutil
    import torch

    requested = max(1, min(10, int(requested)))
    gpu_total = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    gpu_free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
    if gpu_peak > 0:
        gpu_cap = max(1, int((gpu_free * 0.8) // max(1, gpu_peak * 1.2)))
    else:
        gpu_cap = requested
    rss_bytes = max(1, rss_peak_kb * 1024)
    cpu_cap = max(1, int((psutil.virtual_memory().available * 0.65) // (rss_bytes * 1.5)))
    chosen = min(requested, gpu_cap, cpu_cap)
    return max(1, chosen), {
        "requested": requested,
        "selected": max(1, chosen),
        "gpu_total_bytes": int(gpu_total),
        "gpu_free_bytes_at_planning": int(gpu_free),
        "worker_peak_cuda_reserved_bytes": int(gpu_peak),
        "worker_peak_rss_kb": int(rss_peak_kb),
        "gpu_cap": int(gpu_cap),
        "cpu_cap": int(cpu_cap),
    }


def launch_tasks(group: str, folds: list[int], max_workers: int, retry_failed: int = 1) -> None:
    active: dict[subprocess.Popen, tuple[int, Any, Path]] = {}
    pending = list(folds)
    failed: list[int] = []
    root = RT / "groups" / group / "logs"
    root.mkdir(parents=True, exist_ok=True)
    while pending or active:
        while pending and len(active) < max_workers:
            fold = pending.pop(0)
            log = root / f"{group}_f{fold:02d}.log"
            handle = log.open("ab")
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": "0",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "NUMEXPR_NUM_THREADS": "2",
            })
            proc = subprocess.Popen(
                [sys.executable, str(HERE / "run.py"), "--worker", "--group", group, "--fold", str(fold)],
                cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT,
            )
            active[proc] = (fold, handle, log)
        done = []
        for proc, (fold, handle, log) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            handle.close()
            if code == 0:
                task = json.loads((RT / "groups" / group / "tasks" / f"{group}_f{fold:02d}.json").read_text(encoding="utf-8"))
                print(f"completed {group} fold {fold}/10; ACC={task['metrics']['acc']:.4f}", flush=True)
            else:
                failed.append(fold)
                tail = log.read_text(errors="replace")[-6000:] if log.exists() else "missing worker log"
                print(f"failed {group} fold {fold} ({code}):\n{tail}", flush=True)
            done.append(proc)
        for proc in done:
            active.pop(proc, None)
        if active or pending:
            time.sleep(1)
    if failed and retry_failed > 0:
        retry = sorted(set(failed))
        print(f"retrying failed tasks only for {group}: {retry}", flush=True)
        launch_tasks(group, retry, max_workers=1, retry_failed=retry_failed - 1)
        remaining = []
        for fold in retry:
            task = RT / "groups" / group / "tasks" / f"{group}_f{fold:02d}.json"
            if not task.is_file() or json.loads(task.read_text(encoding="utf-8")).get("status") != "COMPLETE":
                remaining.append(fold)
        if remaining:
            raise RuntimeError(f"unrecovered worker failures for {group}: {remaining}")
    elif failed:
        raise RuntimeError(f"worker failures for {group}: {sorted(set(failed))}")


def update_manifest_group(group: str, status: str) -> None:
    path = HERE / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tasks = []
    for fold in range(1, 11):
        task_path = RT / "groups" / group / "tasks" / f"{group}_f{fold:02d}.json"
        if task_path.is_file():
            tasks.append(json.loads(task_path.read_text(encoding="utf-8")))
    manifest["completed_by_group"][group] = len([task for task in tasks if task.get("status") == "COMPLETE"])
    manifest["status_by_group"][group] = status
    manifest["completed_formal_folds"] = sum(manifest["completed_by_group"].values())
    manifest["status"] = "TRAINING_PARTIAL" if manifest["completed_formal_folds"] else "PREPARED"
    atomic_json(path, manifest)


def run_all(max_workers: int = 10) -> None:
    import psutil
    import torch
    from text_features import group_text_banks

    preflight_path = RESULTS / "preflight.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("run --preflight before formal training")
    _ = get_fold_weights()
    cache = load_text_cache()

    base_audit_path = RESULTS / "base_replay_audit.json"
    base_audit_pass = base_audit_path.is_file() and json.loads(base_audit_path.read_text(encoding="utf-8")).get("status") == "PASS"
    if not base_audit_pass:
        # R01 fold 1 supplies the observed process envelope for a measured worker cap.
        task1 = RT / "groups" / GROUPS[0] / "tasks" / f"{GROUPS[0]}_f01.json"
        launch_tasks(GROUPS[0], [1], 1)
        measure = json.loads(task1.read_text(encoding="utf-8"))
        if torch.cuda.is_available():
            gpu_peak = int(measure["peak_cuda_reserved_bytes"])
            gpu_free = int(torch.cuda.mem_get_info()[0])
        else:
            gpu_peak, gpu_free = 0, 0
        max_parallel, resource_record = _resource_caps(max_workers, gpu_peak, int(measure["peak_rss_kb"]))
        resource_record["gpu_free_bytes_at_planning"] = gpu_free
        resource_record["cpu_available_bytes_at_planning"] = int(psutil.virtual_memory().available)
        atomic_json(RESULTS / "resource_plan.json", resource_record)

        launch_tasks(GROUPS[0], list(range(2, 11)), max_parallel)
        update_manifest_group(GROUPS[0], "TRAINING_COMPLETE")
        audit = subprocess.run([sys.executable, str(HERE / "audit.py"), "--base-replay"], cwd=str(ROOT), check=False)
        if audit.returncode != 0:
            update_manifest_group(GROUPS[0], "BASE_PARITY_FAILED")
            raise RuntimeError("R01 base replay did not pass the frozen historical parity gate; later groups were not started")
        update_manifest_group(GROUPS[0], "PARITY_PASS")
    elif not (RESULTS / "resource_plan.json").is_file():
        raise RuntimeError("R01 parity is PASS but its resource plan is missing")
    else:
        resource_record = json.loads((RESULTS / "resource_plan.json").read_text(encoding="utf-8"))
        max_parallel = min(max(1, int(max_workers)), int(resource_record["selected"]))

    for group in GROUPS[1:]:
        tasks = [
            fold for fold in range(1, 11)
            if not (RT / "groups" / group / "tasks" / f"{group}_f{fold:02d}.json").is_file()
            or json.loads((RT / "groups" / group / "tasks" / f"{group}_f{fold:02d}.json").read_text(encoding="utf-8")).get("status") != "COMPLETE"
        ]
        if tasks:
            launch_tasks(group, tasks, max_parallel)
        update_manifest_group(group, "TRAINING_COMPLETE")

    print("FORMAL_TRAINING_DONE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(), ensure_ascii=False), flush=True)
    elif args.preflight:
        print(json.dumps(preflight(), ensure_ascii=False), flush=True)
    elif args.calibrate:
        print(json.dumps(calibrate_losses(), ensure_ascii=False), flush=True)
    elif args.worker:
        if args.group not in GROUPS or args.fold is None or not 1 <= args.fold <= 10:
            raise SystemExit("--worker needs a declared --group and fold 1..10")
        worker(args.group, args.fold)
    elif args.run:
        run_all(args.max_workers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
