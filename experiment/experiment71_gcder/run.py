#!/usr/bin/env python3
"""Experiment 71 GCDER training and reproducibility controls."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
CONFIG_PATH = EXP / "config.json"
CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
ROOT = Path(CFG["paths"]["root"])
BASE_EXP = ROOT / "experiment/experiment53_3_cadr_graph_fallback_fix"
REF_EXP = Path(CFG["paths"]["base_experiment"])
E59_DIR = Path(CFG["paths"]["e59_experiment"])
E70_DIR = Path("/model/spromptgl/experiment/experiment70_gcdsi")
TEXT_CACHE = Path(CFG["paths"]["text_cache"])
DATA_CSV = Path(CFG["paths"]["data_csv"])
MODAL_MAPPING = Path(CFG["paths"]["modal_mapping"])
RUNTIME = EXP / ".runtime"
RESULTS = EXP / "results"
PROMPT_CACHE = RUNTIME / "prompt_banks.pt"
PRE_FLIGHT = RUNTIME / "preflight.json"
MANIFEST_PATH = EXP / "manifest.json"
REFERENCE_GROUP = "CADR_WD_0p01_GFR_WD_0p11"
GROUPS = ("G1", "G2", "G3")
GROUP_NAMES = {
    "G1": "GCDER-BRAINPROMPT-2",
    "G2": "GCDER-SPROMPTGL-8",
    "G3": "GCDER-ANONYMOUS-2",
}
TEXTS_BP = [
    "A brain network of a TC subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.",
    "A brain network of a ASD subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.",
]
TEXTS_SP = [
    "Nromal brain volume",
    "Uniform cortical thickness",
    "Symmetrical brain structure",
    "Normal ventricular morphology",
    "Altered brain volume",
    "Abnormal functional connectivity",
    "White matter integrity issues",
    "Different activation patterns",
]
MODALITIES = ["PHENO", "ANAT", "FUNC", "CORR"]
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")

sys.path.insert(0, str(BASE_EXP))
import run as BASE53  # noqa: E402
sys.modules["run"] = BASE53
sys.path.insert(0, str(E59_DIR))
_spec = importlib.util.spec_from_file_location("e71_e59_runtime", E59_DIR / "run.py")
E59 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = E59
_spec.loader.exec_module(E59)
BASE = E59.BASE
sys.path.insert(0, str(EXP))
from gcder import GCDER, assert_shapes, fuse_log_probabilities, parameter_count  # noqa: E402


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_obj(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def state_sha(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8") + b"\0")
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode() + b"\0")
            digest.update(str(tuple(tensor.shape)).encode() + b"\0")
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8") + b"\0")
    return digest.hexdigest()


def module_sha(module) -> str:
    return state_sha(module.state_dict())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        BASE.runtime()["torch"].save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def reference_paths() -> tuple[Path, Path, Path, Path, Path]:
    group_root = REF_EXP / ".runtime/fallback_wd_sweep" / REFERENCE_GROUP
    return (
        REF_EXP / "manifest_fallback_wd_sweep.json",
        REF_EXP / "results/fallback_wd_sweep/fold_metrics.csv",
        REF_EXP / "results/fallback_wd_sweep/checkpoint_audit.csv",
        group_root / "tasks",
        group_root / "checkpoints",
    )


def load_reference() -> tuple[dict, dict[int, dict], dict[int, dict[str, float]]]:
    manifest_path, metrics_path, audits_path, task_dir, checkpoint_dir = reference_paths()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE_AUDITED" or manifest.get("independent_checkpoint_audits_passed") != 40:
        raise AssertionError("The 53.3 fallback WD sweep is not fully audited.")
    group = next((row for row in manifest.get("groups", []) if row.get("name") == REFERENCE_GROUP), None)
    if group is None or float(group["cadr_weight_decay"]) != 0.01 or float(group["gfr_weight_decay"]) != 0.11:
        raise AssertionError("The selected historical base is not the frozen third group.")

    manifest_rows = {int(row["fold"]): row for row in manifest["folds"] if row["group"] == REFERENCE_GROUP}
    metric_rows = {int(row["fold"]): row for row in csv_rows(metrics_path) if row["group"] == REFERENCE_GROUP}
    audit_rows = {int(row["fold"]): row for row in csv_rows(audits_path) if row["group"] == REFERENCE_GROUP}
    if any(len(rows) != 10 for rows in (manifest_rows, metric_rows, audit_rows)):
        raise AssertionError("53.3 third-group CSV/manifest does not contain ten folds.")

    reference_metrics: dict[int, dict[str, float]] = {}
    for fold in range(1, 11):
        row = manifest_rows[fold]
        metrics = metric_rows[fold]
        audit = audit_rows[fold]
        stem = f"{REFERENCE_GROUP}_f{fold:02d}"
        checkpoint = checkpoint_dir / f"{stem}.best.pt"
        task = json.loads((task_dir / f"{stem}.json").read_text(encoding="utf-8"))
        if not checkpoint.is_file() or task.get("status") != "COMPLETE" or task.get("readback_status") != "WORKER_READBACK_PASS":
            raise AssertionError(f"53.3 third-group fold {fold} task/checkpoint is incomplete.")
        actual_sha = sha_file(checkpoint)
        if actual_sha != row["checkpoint_sha256"] or actual_sha != task["best_sha256"] or actual_sha != audit["checkpoint_sha256"]:
            raise AssertionError(f"53.3 third-group fold {fold} checkpoint SHA differs across records.")
        if audit.get("checkpoint_audit") != "PASS" or metrics.get("checkpoint_audit") != "PASS" or row.get("audit") != "PASS":
            raise AssertionError(f"53.3 third-group fold {fold} audit is not PASS.")
        for key in ("split_sha", "pair_sha", "initial_shared_sha", "initial_added_sha", "initial_rng_sha"):
            if task.get(key) != row[key]:
                raise AssertionError(f"53.3 third-group fold {fold} {key} mismatch.")
        if float(task["cadr_weight_decay"]) != 0.01 or float(task["gfr_weight_decay"]) != 0.11:
            raise AssertionError(f"53.3 third-group fold {fold} used an incorrect WD setting.")
        if any(float(value) != 0.11 for value in task["optimizer_weight_decay"]["base"]):
            raise AssertionError(f"53.3 third-group fold {fold} GFR optimizer WD mismatch.")
        if any(float(value) != 0.11 for value in task["optimizer_weight_decay"]["cam"]):
            raise AssertionError(f"53.3 third-group fold {fold} CAM optimizer WD mismatch.")
        if any(float(value) != 0.01 for value in task["optimizer_weight_decay"]["added"]):
            raise AssertionError(f"53.3 third-group fold {fold} CADR optimizer WD mismatch.")
        checkpoint_obj = BASE.runtime()["torch"].load(checkpoint, map_location="cpu", weights_only=False)
        if checkpoint_obj.get("group") != REFERENCE_GROUP or int(checkpoint_obj.get("fold", -1)) != fold:
            raise AssertionError(f"53.3 third-group fold {fold} checkpoint identity mismatch.")
        if float(checkpoint_obj["cadr_weight_decay"]) != 0.01 or float(checkpoint_obj["gfr_weight_decay"]) != 0.11:
            raise AssertionError(f"53.3 third-group fold {fold} checkpoint WD mismatch.")
        for key in (
            "initial_shared_sha", "initial_gfr_sha", "initial_cam_sha", "initial_added_sha",
            "initial_schema_sha", "initial_rng_sha", "split_sha", "pair_sha", "anchor_sha",
        ):
            if task.get(key) != checkpoint_obj.get(key):
                raise AssertionError(f"53.3 third-group fold {fold} task/checkpoint {key} mismatch.")
            row[key] = task[key]
        reference_metrics[fold] = {metric: float(metrics[metric]) for metric in METRICS}
    np = BASE.runtime()["np"]
    expected_mean = CFG["base"]["reference_metrics_percent"]
    expected_sd = CFG["base"]["reference_population_sd_percent"]
    for metric in expected_mean:
        values = np.asarray([reference_metrics[fold][metric] for fold in range(1, 11)], dtype=np.float64) * 100.0
        if abs(float(values.mean()) - float(expected_mean[metric])) > 2e-4:
            raise AssertionError(f"53.3 {metric} mean does not match the frozen reference table.")
        if abs(float(values.std(ddof=0)) - float(expected_sd[metric])) > 2e-4:
            raise AssertionError(f"53.3 {metric} population SD does not match the frozen reference table.")
    return manifest, manifest_rows, reference_metrics


def text_banks() -> tuple[dict[str, Any], dict[str, Any]]:
    torch = BASE.runtime()["torch"]
    import torch.nn.functional as F

    cache = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
    if cache.get("text_map", {}).get("brainprompt") != [TEXTS_BP[1], TEXTS_BP[0]]:
        raise AssertionError("Frozen CLIP cache does not preserve official BrainPrompt [ASD, TD] ordering.")
    if cache.get("text_map", {}).get("original") != TEXTS_SP:
        raise AssertionError("Frozen CLIP cache differs from the eight original SPromptGL prompts.")
    bp_source = torch.as_tensor(cache["embeddings"]["brainprompt"], dtype=torch.float32)
    sp_source = torch.as_tensor(cache["embeddings"]["original"], dtype=torch.float32)
    if tuple(bp_source.shape) != (2, 1024) or tuple(sp_source.shape) != (8, 1024):
        raise AssertionError(f"Unexpected CLIP RN50 text dimensions: {tuple(bp_source.shape)}, {tuple(sp_source.shape)}.")
    brainprompt = F.normalize(bp_source, dim=-1)
    sp_each = F.normalize(sp_source, dim=-1)
    td_centroid = F.normalize(sp_each[:4].mean(dim=0), dim=0)
    asd_centroid = F.normalize(sp_each[4:].mean(dim=0), dim=0)
    spromptgl = torch.stack((asd_centroid, td_centroid)).contiguous()

    seed = int(CFG["prompt_policy"]["anonymous_seed"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction_a = F.normalize(torch.randn(1024, generator=generator), dim=0)
    orthogonal = torch.randn(1024, generator=generator)
    orthogonal = F.normalize(orthogonal - torch.dot(direction_a, orthogonal) * direction_a, dim=0)
    cosine = torch.dot(brainprompt[0], brainprompt[1]).clamp(-0.999999, 0.999999)
    direction_b = cosine * direction_a + torch.sqrt(1.0 - cosine.square()) * orthogonal
    anonymous = torch.stack((direction_a, F.normalize(direction_b, dim=0))).contiguous()
    banks = {"G1": brainprompt.contiguous(), "G2": spromptgl, "G3": anonymous}
    norms = {group: value.norm(dim=-1).tolist() for group, value in banks.items()}
    cosines = {group: float(torch.dot(value[0], value[1])) for group, value in banks.items()}
    geometry_error = {
        "max_norm_error_g1_g3": max(abs(a - b) for a, b in zip(norms["G1"], norms["G3"])),
        "cosine_error_g1_g3": abs(cosines["G1"] - cosines["G3"]),
    }
    if max(geometry_error.values()) > 1e-5:
        raise AssertionError(f"Anonymous prompt geometry mismatch: {geometry_error}")
    metadata = {
        "encoder": "OpenAI CLIP RN50, frozen cached embeddings from Experiment 60",
        "text_cache_path": str(TEXT_CACHE),
        "text_cache_sha256": sha_file(TEXT_CACHE),
        "text_dim": 1024,
        "brainprompt_published_order": ["TC", "ASD"],
        "brainprompt_training_order": ["ASD", "TD"],
        "brainprompt_texts": TEXTS_BP,
        "spromptgl_original_order": TEXTS_SP,
        "spromptgl_fixed_aggregation": "L2-normalize each; mean prompts 1-4 for TD and 5-8 for ASD; normalize centroids; order [ASD, TD]",
        "anonymous_generator": "private CPU torch.Generator",
        "anonymous_seed": seed,
        "anonymous_geometry_error": geometry_error,
        "bank_norms": norms,
        "bank_class_cosine": cosines,
        "bank_sha256": {
            group: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
            for group, value in banks.items()
        },
        "text_list_sha256": {
            "brainprompt": hashlib.sha256(json.dumps(TEXTS_BP, ensure_ascii=True).encode()).hexdigest(),
            "spromptgl": hashlib.sha256(json.dumps(TEXTS_SP, ensure_ascii=True).encode()).hexdigest(),
        },
    }
    payload = {"banks": banks, "metadata": metadata}
    if PROMPT_CACHE.is_file():
        saved = torch.load(PROMPT_CACHE, map_location="cpu", weights_only=False)
        if saved["metadata"] != metadata or any(not torch.equal(saved["banks"][key], banks[key]) for key in GROUPS):
            raise AssertionError("Existing Experiment 71 prompt cache differs from the frozen prompt geometry.")
    else:
        atomic_torch_save(PROMPT_CACHE, payload)
    return banks, metadata


def build_base_state(fold: int, reference_manifest: dict, reference_folds: dict[int, dict]):
    torch = BASE.runtime()["torch"]
    state = E59.build_base_state(fold, reference_manifest, reference_folds)
    optimizer_specs = (
        (state["opt_base"], torch.optim.Adam, 0.0038, 0.11, "GFR"),
        (state["opt_cam"], torch.optim.Adam, 0.0038, 0.11, "CAM"),
        (state["opt_added"], torch.optim.AdamW, 0.001, 0.01, "CADR"),
    )
    for optimizer, expected_class, expected_lr, expected_wd, label in optimizer_specs:
        if not isinstance(optimizer, expected_class):
            raise AssertionError(f"{label} optimizer class differs from the frozen 53.3 third-group protocol.")
        if any(float(group["lr"]) != expected_lr for group in optimizer.param_groups):
            raise AssertionError(f"{label} optimizer LR differs from the frozen 53.3 third-group protocol.")
        if any(float(group["weight_decay"]) != expected_wd for group in optimizer.param_groups):
            raise AssertionError(f"{label} optimizer WD differs from the frozen 53.3 third-group protocol.")
    if any(float(group["weight_decay"]) != 0.11 for group in state["opt_base"].param_groups):
        raise AssertionError("GFR Adam must retain WD=0.11.")
    if any(float(group["weight_decay"]) != 0.11 for group in state["opt_cam"].param_groups):
        raise AssertionError("CAM Adam must retain WD=0.11.")
    if any(float(group["weight_decay"]) != 0.01 for group in state["opt_added"].param_groups):
        raise AssertionError("Every original CADR AdamW parameter group must retain WD=0.01.")
    return state


def make_gcder(text_dim: int, fold: int):
    torch = BASE.runtime()["torch"]
    device = BASE.runtime()["device"]
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    rng_before = BASE.pack_rng()
    with torch.random.fork_rng(devices=devices, enabled=True):
        seed = 710000 + int(fold)
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        module = GCDER(text_dim).to(device)
    if BASE.rng_sha(rng_before) != BASE.rng_sha(BASE.pack_rng()):
        raise AssertionError("GCDER initialization changed the shared Python/NumPy/CPU/CUDA RNG.")
    return module


def optimizer_digest(optimizer) -> str:
    torch = BASE.runtime()["torch"]
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode() + b"\0")
            digest.update(str(tuple(tensor.shape)).encode() + b"\0")
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict\0")
            for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                update(key)
                update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode() + b"\0")
            digest.update(str(len(value)).encode() + b"\0")
            for item in value:
                update(item)
        elif value is None or isinstance(value, (bool, int, float, str)):
            digest.update(type(value).__name__.encode() + b"\0")
            digest.update(repr(value).encode("utf-8") + b"\0")
        else:
            digest.update(type(value).__name__.encode() + b"\0")
            digest.update(repr(value).encode("utf-8") + b"\0")

    update(optimizer.state_dict())
    return digest.hexdigest()


def _norm(parameters) -> float:
    torch = BASE.runtime()["torch"]
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt().cpu()) if values else 0.0


def _finite_gradients(module) -> bool:
    torch = BASE.runtime()["torch"]
    return all(p.grad is None or torch.isfinite(p.grad).all().item() for p in module.parameters())


def gcder_forward(state, module, bank, training: bool):
    torch = BASE.runtime()["torch"]
    import torch.nn.functional as F

    state["cam"]._e59_positions = None
    base = BASE.forward_state(state, training=training, use_cadr=True)
    positions = state["cam"]._e59_positions
    if positions is None or tuple(positions.shape) != (871, 4, 36) or not positions.requires_grad and training:
        raise AssertionError("CAM position hook did not capture a gradient-connected 871x4x36 tensor.")
    semantic = module(positions, base["fused"], bank)
    logp = fuse_log_probabilities(base["logp"], semantic["scores"], semantic["beta"])
    classification = BASE.runtime()["utils"].ClsLoss(logp, state["y"], state["train"], state["weights"])
    total = classification + base["graph"] + 0.05 * base["pair"] + 0.01 * base["adv"] + 0.01 * base["preserve"]
    base.update({
        "logp_base": base["logp"],
        "logp": logp,
        "positions": positions,
        "gcder": semantic,
        "cls": classification,
        "total": total,
    })
    if not torch.isfinite(logp).all() or not torch.isfinite(total):
        raise FloatingPointError("Non-finite GCDER decision output or total loss.")
    if not torch.allclose(logp.exp().sum(-1), torch.ones(logp.shape[0], device=logp.device), atol=2e-6, rtol=2e-6):
        raise AssertionError("Final log probabilities are not normalized.")
    return base


def _gradient_record(epoch: int, module, out, path_gradients: tuple | None, before_beta: float) -> dict:
    torch = BASE.runtime()["torch"]
    text_grad = _norm(module.text_proj.parameters())
    graph_grad = _norm(module.graph_proj.parameters())
    key_grad = _norm(module.key.parameters())
    value_grad = _norm(module.value.parameters())
    score_grad = _norm(
        list(module.score_norm.parameters()) +
        list(module.score_hidden.parameters()) +
        list(module.score_out.parameters())
    )
    beta_grad = float(module.beta_raw.grad.detach().abs().cpu()) if module.beta_raw.grad is not None else 0.0
    result = {
        "epoch": int(epoch),
        "beta_before_update": before_beta,
        "beta_grad_abs": beta_grad,
        "text_projection_grad_norm": text_grad,
        "gfr_condition_grad_norm": graph_grad,
        "key_grad_norm": key_grad,
        "value_grad_norm": value_grad,
        "shared_score_grad_norm": score_grad,
        "all_gcder_gradients_finite": _finite_gradients(module),
    }
    if path_gradients is not None:
        cam_grad, gfr_grad = path_gradients
        result["gcder_correction_to_cam_norm"] = float(cam_grad.detach().float().norm().cpu()) if cam_grad is not None else 0.0
        result["gcder_correction_to_gfr_norm"] = float(gfr_grad.detach().float().norm().cpu()) if gfr_grad is not None else 0.0
        result["gcder_correction_path_finite"] = all(
            gradient is None or torch.isfinite(gradient).all().item()
            for gradient in path_gradients
        )
    return result


def train_step(state, module, semantic_optimizer, bank, epoch: int) -> tuple[dict, dict]:
    torch = BASE.runtime()["torch"]
    for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"], semantic_optimizer):
        optimizer.zero_grad(set_to_none=True)
    before_beta = float(module.beta.detach().cpu())
    out = gcder_forward(state, module, bank, training=True)
    path_gradients = None
    if epoch in {10, 11, 20, 50, 100, 150, 200}:
        path_gradients = torch.autograd.grad(
            out["gcder"]["beta_delta_score"].sum(),
            (out["positions"], out["fused"]),
            retain_graph=True,
            allow_unused=True,
        )
    out["total"].backward()
    gradient_record = _gradient_record(epoch, module, out, path_gradients, before_beta)
    if not gradient_record["all_gcder_gradients_finite"]:
        raise FloatingPointError(f"Non-finite GCDER gradient at epoch {epoch}.")
    for current in (state["model"], state["cam"]):
        if not _finite_gradients(current):
            raise FloatingPointError(f"Non-finite base-model gradient at epoch {epoch}.")
    for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"], semantic_optimizer):
        optimizer.step()
    return out, gradient_record


def evaluate(state, module, bank) -> dict:
    with BASE.runtime()["torch"].no_grad():
        out = gcder_forward(state, module, bank, training=False)
        metrics = BASE.classification_metrics(out["logp"], state["test"], state["y"])
    return {**out, "metrics": metrics}


def warmup_info_for_fold(state, fold: int, warmup_audit: dict) -> dict:
    torch = BASE.runtime()["torch"]
    warmup_path = E59.warmup_cache_path(fold)
    saved = torch.load(warmup_path, map_location="cpu", weights_only=False)
    checks = {
        "split_sha": state["split_sha"],
        "pair_sha": state["cache_meta"]["pair_sha"],
        "anchor_sha": state["anchor_sha"],
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_added_sha": state["initial_added_sha"],
        "initial_rng_sha": state["construction_rng_sha"],
    }
    for key, expected in checks.items():
        if saved.get(key) != expected:
            raise AssertionError(f"Fold {fold} Experiment 59 warmup binding mismatch for {key}.")
    prior = warmup_audit["folds"][str(fold)]
    if prior.get("status") != "PASS" or prior.get("readback") != "PASS":
        raise AssertionError(f"Fold {fold} Experiment 59 warmup parity was not PASS.")
    if prior.get("normal_capture_digest") != prior.get("captured_digest"):
        raise AssertionError(f"Fold {fold} CAM capture altered the original ten-epoch warmup.")
    if sha_file(warmup_path) != prior["sha256"]:
        raise AssertionError(f"Fold {fold} warmup SHA changed since Experiment 59 audit.")
    current = E59.load_warmup_cache_into_state(state, fold, json.loads(E59.CONFIG_PATH.read_text(encoding="utf-8")))
    if current["sha256"] != sha_file(warmup_path):
        raise AssertionError(f"Fold {fold} warmup cache readback SHA mismatch.")
    if BASE.rng_sha(BASE.pack_rng()) != saved["final_digest"]["rng_sha"]:
        raise AssertionError(f"Fold {fold} warmup shared RNG did not restore exactly.")
    return {"sha256": sha_file(warmup_path), "final_digest": saved["final_digest"], **current}


def _protected_inventory() -> dict[str, str]:
    paths = [
        ROOT / "network.py",
        ROOT / "layers.py",
        ROOT / "utils.py",
        ROOT / "model.py",
        ROOT / "main.py",
        ROOT / "data",
        ROOT / "dataset",
        BASE_EXP,
        REF_EXP,
        E59_DIR,
        ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors",
        TEXT_CACHE,
        E70_DIR / "run.py",
        E70_DIR / "gcdsi.py",
        E70_DIR / "audit.py",
        E70_DIR / "config.json",
        E70_DIR / "manifest.json",
        E70_DIR / ".runtime/preflight.json",
        E70_DIR / "results/report.md",
    ]
    files: set[Path] = set()
    for path in paths:
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(child for child in path.rglob("*") if child.is_file())
    return {str(path): sha_file(path) for path in sorted(files)}


def _safe_e70_warmup_audit() -> dict:
    e70_manifest = json.loads((E70_DIR / "manifest.json").read_text(encoding="utf-8"))
    e70_preflight = json.loads((E70_DIR / ".runtime/preflight.json").read_text(encoding="utf-8"))
    if e70_manifest.get("status") != "COMPLETE_AUDITED" or e70_preflight.get("status") != "PASS":
        raise AssertionError("Experiment 70 is not in a completed, read-only-audited state.")
    if e70_manifest.get("provenance", {}).get("reference_group") != REFERENCE_GROUP:
        raise AssertionError("Experiment 70 warmup audit is bound to a different historical base.")
    provenance = e70_manifest.get("provenance", {})
    warmup = provenance.get("warmup_parity", {})
    if (
        e70_preflight.get("warmup_epochs") != 10
        or warmup.get("shared_cache_status") != "PASS"
        or warmup.get("folds") != 10
        or not warmup.get("all_fold_readbacks_pass")
        or not warmup.get("e70_fold_binding_pass")
    ):
        raise AssertionError("Experiment 70 did not confirm safe ten-epoch warmup reuse.")
    return {
        "status": "PASS_READ_ONLY_WARMUP_AND_PROMPT_CACHE_ONLY",
        "e70_manifest_sha256": sha_file(E70_DIR / "manifest.json"),
        "e70_preflight_sha256": sha_file(E70_DIR / ".runtime/preflight.json"),
        "e70_best_checkpoints_loaded": False,
        "warmup_cache_sha256": e70_preflight["warmup_cache_sha256"],
        "text_cache_sha256": e70_preflight["text_assets"]["cache_sha256"],
        "shared_warmup_status": warmup.get("shared_cache_status"),
        "warmup_folds_readback_passed": warmup.get("all_fold_readbacks_pass"),
    }


def _math_preflight(torch, logp_base, score, beta) -> dict:
    import torch.nn.functional as F

    fused = fuse_log_probabilities(logp_base, score, beta)
    identity_error = float((fuse_log_probabilities(logp_base, score, torch.zeros_like(beta)) - logp_base).abs().max().cpu())
    raw_logits = torch.randn((8, 2), generator=torch.Generator().manual_seed(71))
    raw_logp = F.log_softmax(raw_logits, dim=-1)
    probe_scores = torch.randn((8, 2), generator=torch.Generator().manual_seed(72))
    probe_beta = torch.tensor(0.17)
    from_logp = F.log_softmax(raw_logp + probe_beta * probe_scores, dim=-1)
    from_logits = F.log_softmax(raw_logits + probe_beta * probe_scores, dim=-1)
    common = torch.full((8, 1), 3.25)
    shift_invariant = F.log_softmax(raw_logp + probe_beta * (probe_scores + common), dim=-1)
    raw_logp_identity_error = float((from_logp - from_logits).abs().max().item())
    common_shift_error = float((from_logp - shift_invariant).abs().max().item())
    normalized = float((fused.exp().sum(-1) - 1).abs().max().cpu())
    if identity_error > 2e-6 or raw_logp_identity_error > 2e-6 or common_shift_error > 2e-6 or normalized > 2e-6:
        raise AssertionError("Decision-level log-probability fusion failed the mathematical identity checks.")
    return {
        "identity_max_abs_error": identity_error,
        "raw_logits_vs_log_prob_equivalence_max_abs_error": raw_logp_identity_error,
        "common_score_shift_invariance_max_abs_error": common_shift_error,
        "normalization_max_abs_error": normalized,
        "softmax_axis": "four CAM modality positions",
    }


def _initialization_rows(reference_manifest: dict, reference_folds: dict[int, dict], e59_preflight: dict) -> tuple[list[dict], dict[int, dict]]:
    torch = BASE.runtime()["torch"]
    fold_initial = {int(row["fold"]): row for row in e59_preflight["base_reference"]["fold_initializations"]}
    if len(fold_initial) != 10:
        raise AssertionError("Experiment 59 initialization reference is missing folds.")
    warmup_audit = e59_preflight["shared_warmup_cache"]
    rows = []
    for fold in range(1, 11):
        state = build_base_state(fold, reference_manifest, reference_folds)
        old = fold_initial[fold]
        expected = reference_folds[fold]
        values = {
            "initial_shared_sha": state["initial_shared_sha"],
            "initial_gfr_sha": state["initial_gfr_sha"],
            "initial_cam_sha": state["initial_cam_sha"],
            "initial_added_sha": state["initial_added_sha"],
            "initial_schema_sha": state["initial_schema_sha"],
            "initial_rng_sha": state["construction_rng_sha"],
            "split_sha": state["split_sha"],
            "pair_sha": state["cache_meta"]["pair_sha"],
            "anchor_sha": state["anchor_sha"],
        }
        for key in (
            "initial_shared_sha", "initial_gfr_sha", "initial_cam_sha", "initial_added_sha",
            "initial_schema_sha", "initial_rng_sha", "split_sha", "pair_sha", "anchor_sha",
        ):
            if values[key] != expected[key]:
                raise AssertionError(f"Fold {fold} base value does not match the 53.3 reference for {key}.")
        for key in (
            "initial_gfr_sha", "initial_cam_sha", "initial_shared_sha", "initial_added_sha",
            "initial_rng_sha", "split_sha", "pair_sha",
        ):
            if values[key] != old[key]:
                raise AssertionError(f"Fold {fold} base value does not match the Experiment 59 audit for {key}.")
        if values["anchor_sha"] != old["anchor_sha256"]:
            raise AssertionError(f"Fold {fold} anchor SHA differs from the Experiment 59 audit.")
        if any(float(group["weight_decay"]) != 0.11 for group in state["opt_base"].param_groups):
            raise AssertionError("GFR optimizer has an incorrect parameter-group weight decay.")
        if any(float(group["weight_decay"]) != 0.11 for group in state["opt_cam"].param_groups):
            raise AssertionError("CAM optimizer has an incorrect parameter-group weight decay.")
        if any(float(group["weight_decay"]) != 0.01 for group in state["opt_added"].param_groups):
            raise AssertionError("CADR optimizer has an incorrect parameter-group weight decay.")
        warm = warmup_info_for_fold(state, fold, warmup_audit)
        init_module = make_gcder(1024, fold)
        init_sha = module_sha(init_module)
        group_shas = {group: module_sha(make_gcder(1024, fold)) for group in GROUPS}
        if len(set(group_shas.values())) != 1 or init_sha != next(iter(group_shas.values())):
            raise AssertionError(f"Fold {fold} GCDER initialization is not identical across groups.")
        rows.append({
            "fold": fold,
            **values,
            "initial_gcder_sha": init_sha,
            "gcd_parameters": parameter_count(init_module),
            "optimizer_weight_decay": {
                "gfr": [float(g["weight_decay"]) for g in state["opt_base"].param_groups],
                "cam": [float(g["weight_decay"]) for g in state["opt_cam"].param_groups],
                "cadr": [float(g["weight_decay"]) for g in state["opt_added"].param_groups],
            },
            "warmup_sha256": warm["sha256"],
            "warmup_digest": warm["final_digest"],
            "status": "PASS",
        })
        del init_module, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows, {}


def _check_shapes_math_and_gradients(state, banks: dict[str, Any], fold: int) -> dict:
    torch = BASE.runtime()["torch"]
    import torch.nn.functional as F

    E59.attach_position_capture(state["cam"])
    module = make_gcder(1024, fold)
    bank = banks["G1"].to(BASE.runtime()["device"])
    before_rng = BASE.rng_sha(BASE.pack_rng())
    state["cam"]._e59_positions = None
    with torch.no_grad():
        base = BASE.forward_state(state, training=False, use_cadr=True)
        positions = state["cam"]._e59_positions
        result = module(positions, base["fused"], bank)
        assert_shapes(result, 871, 1024)
        identity = fuse_log_probabilities(base["logp"], result["scores"], module.beta)
        math = _math_preflight(torch, base["logp"], result["scores"], module.beta)
    if BASE.rng_sha(BASE.pack_rng()) != before_rng:
        raise AssertionError("Deterministic evaluation or GCDER inference consumed a main RNG state.")
    if not torch.allclose(identity, base["logp"], atol=2e-6, rtol=2e-6):
        raise AssertionError("beta=0 did not preserve the original CAM log probabilities.")
    if float(result["delta_score"].std().cpu()) <= 1e-8 or float(result["delta_score"].abs().mean().cpu()) <= 1e-8:
        raise AssertionError("Preflight category evidence difference is constant or numerically zero.")
    if not torch.allclose(result["attention"].sum(-1), torch.ones((871, 2), device=result["attention"].device), atol=1e-6):
        raise AssertionError("Disease attention does not normalize over the four modalities.")

    synthetic_betas = torch.tensor((0.13, -0.07), device=result["scores"].device)
    rowwise = torch.stack([
        F.log_softmax(base["logp"] + synthetic_betas[index] * result["scores"], dim=-1)
        for index in range(2)
    ])
    if not torch.isfinite(rowwise).all():
        raise AssertionError("Synthetic two-group score fusion produced non-finite values.")

    semantic_optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(CFG["gcder"]["optimizer"]["lr"]),
        weight_decay=float(CFG["gcder"]["optimizer"]["weight_decay"]),
    )
    state["opt_base"].zero_grad(set_to_none=True)
    state["opt_cam"].zero_grad(set_to_none=True)
    state["opt_added"].zero_grad(set_to_none=True)
    semantic_optimizer.zero_grad(set_to_none=True)
    first = gcder_forward(state, module, bank, training=True)
    first["total"].backward()
    beta_first_grad = float(module.beta_raw.grad.detach().abs().cpu()) if module.beta_raw.grad is not None else 0.0
    if not _finite_gradients(module) or beta_first_grad <= 0:
        raise AssertionError("Identity-start decision fusion did not produce a finite beta gradient.")
    semantic_optimizer.step()

    state["opt_base"].zero_grad(set_to_none=True)
    state["opt_cam"].zero_grad(set_to_none=True)
    state["opt_added"].zero_grad(set_to_none=True)
    semantic_optimizer.zero_grad(set_to_none=True)
    second = gcder_forward(state, module, bank, training=True)
    correction_gradients = torch.autograd.grad(
        second["gcder"]["beta_delta_score"].sum(),
        (second["positions"], second["fused"]),
        retain_graph=True,
        allow_unused=True,
    )
    second["total"].backward()
    branch_norms = {
        "text_projection": _norm(module.text_proj.parameters()),
        "gfr_condition_projection": _norm(module.graph_proj.parameters()),
        "key": _norm(module.key.parameters()),
        "value": _norm(module.value.parameters()),
        "shared_score": _norm(
            list(module.score_norm.parameters()) +
            list(module.score_hidden.parameters()) +
            list(module.score_out.parameters())
        ),
        "gcder_to_cam": float(correction_gradients[0].detach().float().norm().cpu()) if correction_gradients[0] is not None else 0.0,
        "gcder_to_gfr": float(correction_gradients[1].detach().float().norm().cpu()) if correction_gradients[1] is not None else 0.0,
        "beta_after_first_step": float(module.beta.detach().cpu()),
    }
    if not _finite_gradients(module) or any(branch_norms[key] <= 0 for key in ("text_projection", "gfr_condition_projection", "key", "value", "shared_score", "gcder_to_cam", "gcder_to_gfr")):
        raise AssertionError(f"An active GCDER route has no finite effective gradient: {branch_norms}")
    return {
        "tensor_shapes": {
            "gfr": list(base["fused"].shape),
            "cam": list(positions.shape),
            "query": list(result["query"].shape),
            "key": list(result["key"].shape),
            "value": list(result["value"].shape),
            "attention": list(result["attention"].shape),
            "evidence": list(result["retrieved"].shape),
            "scores": list(result["scores"].shape),
            "final_logp": list(identity.shape),
        },
        "identity": {
            "beta_initial": float(module.beta.detach().cpu()) if False else 0.0,
            "max_abs_error": float((identity - base["logp"]).abs().max().cpu()),
            "prediction_changes": int((identity.argmax(-1) != base["logp"].argmax(-1)).sum().item()),
            "classification_path_unchanged": True,
        },
        "math": math,
        "attention": {
            "softmax_axis": "four modality positions",
            "max_row_sum_error": float((result["attention"].sum(-1) - 1).abs().max().cpu()),
            "mean_abs_class_delta": float((result["attention"][:, 0] - result["attention"][:, 1]).abs().mean().cpu()),
            "mean_abs_score_delta": float(result["delta_score"].abs().mean().cpu()),
            "score_delta_std": float(result["delta_score"].std().cpu()),
        },
        "active_gradient_probe": branch_norms,
        "single_cam_forward_hook": {
            "captured_shape": list(positions.shape),
            "requires_grad": bool(positions.requires_grad),
            "same_forward_capture": True,
        },
        "status": "PASS",
    }


def preflight() -> dict:
    torch = BASE.runtime()["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 71 requires the designated GPU0; CUDA is unavailable.")
    torch.cuda.set_device(0)
    data, folds = BASE.load_data()
    x, y = data
    if tuple(x.shape) != (871, 320) or list(BASE.runtime()["np"].bincount(y)) != [403, 468]:
        raise AssertionError("The current ABIDE data does not match the frozen 871x320/403+468 definition.")

    reference_manifest, reference_folds, reference_metrics = load_reference()
    text_banks_by_group, prompt_meta = text_banks()
    e59_preflight_path = E59_DIR / ".runtime/preflight_audit.json"
    e59_preflight = json.loads(e59_preflight_path.read_text(encoding="utf-8"))
    if e59_preflight.get("status") != "PASS":
        raise AssertionError("Experiment 59 did not pass its initialization and capture audit.")
    e70_safe = _safe_e70_warmup_audit()
    if e70_safe["text_cache_sha256"] != prompt_meta["text_cache_sha256"]:
        raise AssertionError("Experiment 70 and current prompt-cache provenance disagree.")

    expected_data = {
        "processed_standard_data.csv": sha_file(DATA_CSV),
        "modal_feat_dict.npy": sha_file(MODAL_MAPPING),
    }
    e70_data = json.loads((E70_DIR / "manifest.json").read_text(encoding="utf-8"))["provenance"]["integrity"]["data_sha256_at_final_audit"]
    if expected_data != e70_data:
        raise AssertionError("Current ABIDE input files differ from the read-only Experiment 70 audit.")

    initial_rows, _ = _initialization_rows(reference_manifest, reference_folds, e59_preflight)
    probe_state = build_base_state(1, reference_manifest, reference_folds)
    warmup_info_for_fold(probe_state, 1, e59_preflight["shared_warmup_cache"])
    gradient_math = _check_shapes_math_and_gradients(probe_state, text_banks_by_group, 1)
    del probe_state

    protected = _protected_inventory()
    source_hashes = {
        "training_source": sha_file(EXP / "run.py") if (EXP / "run.py").exists() else None,
        "gcder_source": sha_file(EXP / "gcder.py"),
        "config": sha_file(CONFIG_PATH),
        "base_run": sha_file(REF_EXP / "run.py"),
        "base_cadr_gfr": sha_file(REF_EXP / "cadr_gfr.py"),
        "base_config": sha_file(REF_EXP / "config.json"),
        "base_reference_manifest": sha_file(reference_paths()[0]),
        "base_reference_metrics": sha_file(reference_paths()[1]),
        "base_reference_audit": sha_file(reference_paths()[2]),
        "e59_run": sha_file(E59_DIR / "run.py"),
        "e59_capture": sha_file(E59_DIR / "shape_pen.py"),
        "e59_preflight": sha_file(e59_preflight_path),
        "e70_run": sha_file(E70_DIR / "run.py"),
        "e70_audit": sha_file(E70_DIR / "audit.py"),
        "e70_manifest": sha_file(E70_DIR / "manifest.json"),
        "text_cache": sha_file(TEXT_CACHE),
        "data_csv": sha_file(DATA_CSV),
        "modal_mapping": sha_file(MODAL_MAPPING),
        "prompt_cache": sha_file(PROMPT_CACHE),
    }
    fold_reference = []
    for fold in range(1, 11):
        fold_reference.append({
            "fold": fold,
            "metrics": reference_metrics[fold],
            "checkpoint_sha256": reference_folds[fold]["checkpoint_sha256"],
            "split_sha": reference_folds[fold]["split_sha"],
            "pair_sha": reference_folds[fold]["pair_sha"],
            "initial_shared_sha": reference_folds[fold]["initial_shared_sha"],
            "initial_added_sha": reference_folds[fold]["initial_added_sha"],
            "initial_rng_sha": reference_folds[fold]["initial_rng_sha"],
        })
    result = {
        "status": "PASS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fallback_root": str(EXP),
        "fallback_reason": "OSSFS Input/output error while creating the primary /data Experiment 71 directory.",
        "base_reference": {
            "group": REFERENCE_GROUP,
            "manifest_sha256": sha_file(reference_paths()[0]),
            "fold_metric_sha256": sha_file(reference_paths()[1]),
            "checkpoint_audit_sha256": sha_file(reference_paths()[2]),
            "folds": fold_reference,
            "optimizer_weight_decay": {"gfr_adam": [0.11], "cam_adam": [0.11], "cadr_adamw": [0.01, 0.01]},
            "read_only_checkpoints_retrained": False,
        },
        "dataset": {
            "subjects": 871,
            "labels": {"ASD": 403, "TD": 468},
            "label_order": {"ASD": 0, "TD": 1},
            "feature_dimensions": {"PHENO": 48, "ANAT": 6, "FUNC": 10, "CORR": 256},
            "x_shape": list(x.shape),
            "data_sha256": expected_data,
            "split_sha": {str(fold): reference_folds[fold]["split_sha"] for fold in range(1, 11)},
            "transductive_nodes": 871,
        },
        "prompts": prompt_meta,
        "experiment59": {
            "status": "PASS",
            "preflight_sha256": sha_file(e59_preflight_path),
            "warmup_audit": e59_preflight["shared_warmup_cache"],
            "warmup_cache_sha256": {
                str(fold): sha_file(E59.warmup_cache_path(fold))
                for fold in range(1, 11)
            },
            "capture_source_sha256": sha_file(E59_DIR / "shape_pen.py"),
            "warmup_epochs": 10,
            "normal_forward_vs_hooked_warmup": "PASS_10_OF_10_REUSED_FROM_AUDITED_CACHE",
        },
        "experiment70_read_only_check": e70_safe,
        "initialization": {
            "status": "PASS_10_OF_10",
            "folds": initial_rows,
            "main_rng_invariant_during_new_module_init": True,
            "group_shared_initial_gcder_parameters": True,
        },
        "shape_math_and_gradient_checks": gradient_math,
        "source_sha256": source_hashes,
        "protected_inventory_sha256": protected,
        "historical_val_equals_test": True,
        "pre_cv_rfe_and_standardization_retained": True,
        "external_generalization_claim": False,
    }
    atomic_json(PRE_FLIGHT, result)
    manifest = {
        "experiment": "71",
        "name": CFG["name"],
        "status": "PREFLIGHT_PASS",
        "decision": None,
        "primary_data_root_write_failed": True,
        "execution_root": str(EXP),
        "preflight_sha256": sha_file(PRE_FLIGHT),
        "config_sha256": sha_file(CONFIG_PATH),
        "source_sha256": source_hashes,
        "protected_inventory_sha256_at_preflight": protected,
        "groups": list(GROUPS),
        "required_tasks": 30,
        "completed_tasks": 0,
        "failed_tasks": [],
        "checkpoint_audit_status": "NOT_RUN",
        "initialization_status": "PASS_10_OF_10",
        "warmup_status": "PASS_10_OF_10",
        "checkpoint_status": {},
        "base_reference_read_only": True,
        "e70_best_checkpoint_used": False,
    }
    atomic_json(MANIFEST_PATH, manifest)
    return result


def group_paths(group: str, fold: int) -> tuple[Path, Path, Path, Path]:
    stem = f"{group}_f{fold:02d}"
    group_root = RUNTIME / group
    return (
        group_root / "tasks" / f"{stem}.json",
        group_root / "checkpoints" / f"{stem}.best.pt",
        group_root / "checkpoints" / f"{stem}.latest.pt",
        group_root / "logs" / f"{stem}.log",
    )


def _cpu_state(module) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _optimizer_cpu(optimizer) -> dict[str, Any]:
    return copy.deepcopy(optimizer.state_dict())


def _metrics_without_arrays(metrics: dict) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"pred", "prob"}}


def _best_diagnostics(evaluated: dict, module, training_gradient: dict) -> dict:
    torch = BASE.runtime()["torch"]
    attention = evaluated["gcder"]["attention"].detach()
    scores = evaluated["gcder"]["scores"].detach()
    delta = scores[:, 0] - scores[:, 1]
    entropy = -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(-1)
    return {
        "beta": float(module.beta.detach().cpu()),
        "beta_raw": float(module.beta_raw.detach().cpu()),
        "delta_score_mean": float(delta.mean().cpu()),
        "delta_score_std": float(delta.std().cpu()),
        "delta_score_mean_abs": float(delta.abs().mean().cpu()),
        "beta_delta_score_mean": float((module.beta.detach() * delta).mean().cpu()),
        "beta_delta_score_mean_abs": float((module.beta.detach() * delta).abs().mean().cpu()),
        "attention_mean": attention.mean(dim=0).cpu().tolist(),
        "attention_entropy_mean": float(entropy.mean().cpu()),
        "attention_class_abs_difference_mean": float((attention[:, 0] - attention[:, 1]).abs().mean().cpu()),
        "query_cosine_mean": float(torch.nn.functional.cosine_similarity(
            evaluated["gcder"]["query"][:, 0], evaluated["gcder"]["query"][:, 1], dim=-1
        ).mean().cpu()),
        "training_gradient_snapshot": training_gradient,
    }


def _checkpoint_payload(
    group: str,
    fold: int,
    state,
    module,
    semantic_optimizer,
    best_state: dict,
    best_metrics: dict,
    best_diagnostics: dict,
    bank: Any,
    prompt_meta: dict,
    reference_folds: dict[int, dict],
    warmup: dict,
    preflight: dict,
    best_epoch: int,
    epochs_executed: int,
    rng_trace: list[str],
    gradient_snapshots: list[dict],
) -> dict:
    bank_sha = prompt_meta["bank_sha256"][group]
    return {
        "format": "experiment71_gcder_checkpoint_v1",
        "experiment": "71",
        "group": group,
        "group_name": GROUP_NAMES[group],
        "fold": fold,
        "seed": 0,
        "best_epoch": int(best_epoch),
        "epochs_executed": int(epochs_executed),
        "metrics": best_metrics,
        "state_model": best_state["model"],
        "state_cam": best_state["cam"],
        "state_gcder": best_state["gcder"],
        "optimizer_base": best_state["optimizer_base"],
        "optimizer_cam": best_state["optimizer_cam"],
        "optimizer_added": best_state["optimizer_added"],
        "optimizer_gcder": best_state["optimizer_gcder"],
        "text_bank": bank.detach().cpu().clone(),
        "text_bank_sha256": bank_sha,
        "text_cache_sha256": prompt_meta["text_cache_sha256"],
        "prompt_metadata_sha256": sha_obj(prompt_meta),
        "source_sha256": {
            "run.py": sha_file(EXP / "run.py"),
            "gcder.py": sha_file(EXP / "gcder.py"),
            "config.json": sha_file(CONFIG_PATH),
        },
        "config_sha256": sha_file(CONFIG_PATH),
        "preflight_sha256": sha_file(PRE_FLIGHT),
        "reference_group": REFERENCE_GROUP,
        "reference_manifest_sha256": sha_file(reference_paths()[0]),
        "reference_checkpoint_sha256": reference_folds[fold]["checkpoint_sha256"],
        "reference_metric_sha256": sha_file(reference_paths()[1]),
        "warmup_sha256": warmup["sha256"],
        "warmup_digest": warmup["final_digest"],
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_gfr_sha": state["initial_gfr_sha"],
        "initial_cam_sha": state["initial_cam_sha"],
        "initial_added_sha": state["initial_added_sha"],
        "initial_schema_sha": state["initial_schema_sha"],
        "initial_rng_sha": state["construction_rng_sha"],
        "initial_gcder_sha": state["initial_gcder_sha"],
        "split_sha": state["split_sha"],
        "pair_sha": state["cache_meta"]["pair_sha"],
        "anchor_sha": state["anchor_sha"],
        "parameter_sha256": {
            "model": module_sha(state["model"]),
            "cam": module_sha(state["cam"]),
            "gcder": module_sha(module),
        },
        "parameter_counts": {
            "base_gfr_cadr_cam": sum(p.numel() for p in state["model"].parameters()) + sum(p.numel() for p in state["cam"].parameters()),
            "gcder": parameter_count(module),
            "total": sum(p.numel() for p in state["model"].parameters()) + sum(p.numel() for p in state["cam"].parameters()) + parameter_count(module),
        },
        "optimizer_weight_decay": {
            "gfr": [float(g["weight_decay"]) for g in state["opt_base"].param_groups],
            "cam": [float(g["weight_decay"]) for g in state["opt_cam"].param_groups],
            "cadr": [float(g["weight_decay"]) for g in state["opt_added"].param_groups],
            "gcder": [float(g["weight_decay"]) for g in semantic_optimizer.param_groups],
        },
        "diagnostics": best_diagnostics,
        "rng_trace": rng_trace,
        "gradient_snapshots": gradient_snapshots,
        "device": str(BASE.runtime()["device"]),
        "gpu_index": 0,
        "actual_e70_checkpoint_used": False,
    }


def _update_manifest_task(group: str, fold: int, status: str, detail: dict | None = None) -> None:
    lock_path = MANIFEST_PATH.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        records = manifest.setdefault("task_progress", {})
        key = f"{group}_f{fold:02d}"
        records[key] = {"status": status, **(detail or {})}
        manifest["completed_tasks"] = sum(item.get("status") == "COMPLETE" for item in records.values())
        manifest["failed_tasks"] = [
            task for task, item in records.items() if item.get("status") == "FAILED"
        ]
        manifest["status"] = "RUNNING" if manifest["completed_tasks"] < 30 else "TRAINING_COMPLETE_PENDING_AUDIT"
        atomic_json(MANIFEST_PATH, manifest)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def worker(group: str, fold: int) -> None:
    if group not in GROUPS or not 1 <= fold <= 10:
        raise ValueError(f"Invalid task {group}/{fold}.")
    if not PRE_FLIGHT.is_file():
        raise FileNotFoundError("Run the full preflight before starting formal workers.")
    preflight_record = json.loads(PRE_FLIGHT.read_text(encoding="utf-8"))
    if preflight_record.get("status") != "PASS":
        raise AssertionError("Experiment 71 preflight is not PASS.")
    current_sources = {
        "training_source": sha_file(EXP / "run.py"),
        "gcder_source": sha_file(EXP / "gcder.py"),
        "config": sha_file(CONFIG_PATH),
        "base_run": sha_file(REF_EXP / "run.py"),
        "base_cadr_gfr": sha_file(REF_EXP / "cadr_gfr.py"),
        "base_config": sha_file(REF_EXP / "config.json"),
        "e59_run": sha_file(E59_DIR / "run.py"),
        "e59_capture": sha_file(E59_DIR / "shape_pen.py"),
        "e59_preflight": sha_file(E59_DIR / ".runtime/preflight_audit.json"),
        "text_cache": sha_file(TEXT_CACHE),
        "data_csv": sha_file(DATA_CSV),
        "modal_mapping": sha_file(MODAL_MAPPING),
        "prompt_cache": sha_file(PROMPT_CACHE),
    }
    for key, value in current_sources.items():
        if preflight_record["source_sha256"].get(key) != value:
            raise AssertionError(f"Frozen source/data SHA changed after preflight: {key}.")
    reference_manifest, reference_folds, _ = load_reference()
    bank_payload = BASE.runtime()["torch"].load(PROMPT_CACHE, map_location="cpu", weights_only=False)
    prompt_meta = bank_payload["metadata"]
    if prompt_meta["text_cache_sha256"] != sha_file(TEXT_CACHE):
        raise AssertionError("Frozen CLIP text cache changed after preflight.")
    bank = bank_payload["banks"][group].to(BASE.runtime()["device"])
    state = build_base_state(fold, reference_manifest, reference_folds)
    warmup = warmup_info_for_fold(state, fold, preflight_record["experiment59"]["warmup_audit"])
    module = make_gcder(bank.shape[1], fold)
    state["initial_gcder_sha"] = module_sha(module)
    init_row = preflight_record["initialization"]["folds"][fold - 1]
    if state["initial_gcder_sha"] != init_row["initial_gcder_sha"]:
        raise AssertionError(f"Fold {fold} GCDER initialization differs from preflight.")
    state["cam"]._e59_positions = None
    E59.attach_position_capture(state["cam"])
    main_rng_at_start = BASE.rng_sha(BASE.pack_rng())
    semantic_optimizer = BASE.runtime()["torch"].optim.AdamW(
        module.parameters(),
        lr=float(CFG["gcder"]["optimizer"]["lr"]),
        weight_decay=float(CFG["gcder"]["optimizer"]["weight_decay"]),
    )
    task_path, checkpoint_path, latest_path, log_path = group_paths(group, fold)
    for path in (task_path.parent, checkpoint_path.parent, log_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    task_source_sha = sha_file(EXP / "run.py")
    config_sha = sha_file(CONFIG_PATH)

    if task_path.is_file() and checkpoint_path.is_file():
        old_task = json.loads(task_path.read_text(encoding="utf-8"))
        if (
            old_task.get("status") == "COMPLETE"
            and old_task.get("checkpoint_sha256") == sha_file(checkpoint_path)
            and old_task.get("source_sha256") == task_source_sha
            and old_task.get("config_sha256") == config_sha
        ):
            print(f"SKIP {group} fold={fold}: verified complete task", flush=True)
            return

    start_epoch, wait, best_acc, best_epoch = 10, 0, -1.0, -1
    best_state, best_metrics, best_diagnostics = None, None, None
    rng_trace: list[str] = []
    gradient_snapshots: list[dict] = []
    if latest_path.is_file():
        saved = BASE.runtime()["torch"].load(latest_path, map_location="cpu", weights_only=False)
        expected = {
            "group": group,
            "fold": fold,
            "source_sha256": task_source_sha,
            "config_sha256": config_sha,
            "warmup_sha256": warmup["sha256"],
            "initial_gcder_sha": state["initial_gcder_sha"],
            "split_sha": state["split_sha"],
            "pair_sha": state["cache_meta"]["pair_sha"],
        }
        for key, value in expected.items():
            if saved.get(key) != value:
                raise AssertionError(f"Fold {fold} resume checkpoint provenance mismatch for {key}.")
        state["model"].load_state_dict(saved["model"], strict=True)
        state["cam"].load_state_dict(saved["cam"], strict=True)
        module.load_state_dict(saved["gcder"], strict=True)
        state["opt_base"].load_state_dict(saved["optimizer_base"])
        state["opt_cam"].load_state_dict(saved["optimizer_cam"])
        state["opt_added"].load_state_dict(saved["optimizer_added"])
        semantic_optimizer.load_state_dict(saved["optimizer_gcder"])
        BASE.restore_rng(saved["rng"])
        start_epoch = int(saved["next_epoch"])
        wait = int(saved["wait"])
        best_acc = float(saved["best_acc"])
        best_epoch = int(saved["best_epoch"])
        best_state = saved["best_state"]
        best_metrics = saved["best_metrics"]
        best_diagnostics = saved["best_diagnostics"]
        rng_trace = list(saved["rng_trace"])
        gradient_snapshots = list(saved["gradient_snapshots"])
        if BASE.rng_sha(BASE.pack_rng()) != saved["rng_sha"]:
            raise AssertionError("Resumed Python/NumPy/CPU/CUDA RNG did not restore exactly.")
    elif BASE.rng_sha(BASE.pack_rng()) != main_rng_at_start:
        raise AssertionError("GCDER setup consumed the restored epoch-10 base RNG.")

    _update_manifest_task(group, fold, "RUNNING", {"next_epoch": start_epoch, "warmup_sha256": warmup["sha256"]})
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({
            "event": "START",
            "group": group,
            "fold": fold,
            "start_epoch": start_epoch,
            "resume": latest_path.exists(),
            "warmup_sha256": warmup["sha256"],
            "main_rng_sha256_at_entry": main_rng_at_start,
        }, allow_nan=False) + "\n")
        log.flush()
        for epoch in range(start_epoch, int(CFG["cv"]["max_epochs"])):
            state["cam"].train()
            state["model"].train()
            out, gradient = train_step(state, module, semantic_optimizer, bank, epoch)
            if epoch in {10, 11, 20, 50, 100, 150, 200} or epoch % 50 == 0:
                gradient_snapshots.append(gradient)
            with BASE.runtime()["torch"].no_grad():
                evaluated = evaluate(state, module, bank)
            metrics = evaluated["metrics"]
            rng_trace.append(BASE.rng_sha(BASE.pack_rng()))
            if metrics["acc"] > best_acc:
                best_acc = float(metrics["acc"])
                best_epoch = epoch
                wait = 0
                best_metrics = _metrics_without_arrays(metrics)
                best_diagnostics = _best_diagnostics(evaluated, module, gradient)
                best_state = {
                    "model": _cpu_state(state["model"]),
                    "cam": _cpu_state(state["cam"]),
                    "gcder": _cpu_state(module),
                    "optimizer_base": _optimizer_cpu(state["opt_base"]),
                    "optimizer_cam": _optimizer_cpu(state["opt_cam"]),
                    "optimizer_added": _optimizer_cpu(state["opt_added"]),
                    "optimizer_gcder": _optimizer_cpu(semantic_optimizer),
                }
                checkpoint = _checkpoint_payload(
                    group, fold, state, module, semantic_optimizer, best_state,
                    best_metrics, best_diagnostics, bank, prompt_meta,
                    reference_folds, warmup, preflight_record, best_epoch,
                    epoch + 1, rng_trace, gradient_snapshots,
                )
                atomic_torch_save(checkpoint_path, checkpoint)
            else:
                wait += 1

            if (epoch + 1) % 10 == 0 or wait > int(CFG["cv"]["early_stop_wait_gt"]):
                latest = {
                    "group": group,
                    "fold": fold,
                    "source_sha256": task_source_sha,
                    "config_sha256": config_sha,
                    "warmup_sha256": warmup["sha256"],
                    "initial_gcder_sha": state["initial_gcder_sha"],
                    "split_sha": state["split_sha"],
                    "pair_sha": state["cache_meta"]["pair_sha"],
                    "next_epoch": epoch + 1,
                    "wait": wait,
                    "best_acc": best_acc,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "best_metrics": best_metrics,
                    "best_diagnostics": best_diagnostics,
                    "model": _cpu_state(state["model"]),
                    "cam": _cpu_state(state["cam"]),
                    "gcder": _cpu_state(module),
                    "optimizer_base": _optimizer_cpu(state["opt_base"]),
                    "optimizer_cam": _optimizer_cpu(state["opt_cam"]),
                    "optimizer_added": _optimizer_cpu(state["opt_added"]),
                    "optimizer_gcder": _optimizer_cpu(semantic_optimizer),
                    "rng": BASE.pack_rng(),
                    "rng_sha": BASE.rng_sha(BASE.pack_rng()),
                    "rng_trace": rng_trace,
                    "gradient_snapshots": gradient_snapshots,
                }
                atomic_torch_save(latest_path, latest)
                _update_manifest_task(group, fold, "RUNNING", {"next_epoch": epoch + 1, "best_epoch": best_epoch})
            if (epoch + 1) % 10 == 0:
                log.write(json.dumps({
                    "event": "EPOCH",
                    "epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "best_acc": best_acc,
                    "current_acc": metrics["acc"],
                    "beta": best_diagnostics["beta"] if best_diagnostics else None,
                    "wait": wait,
                    "rng_sha256": rng_trace[-1],
                    "gradient": gradient,
                }, allow_nan=False) + "\n")
                log.flush()
            if wait > int(CFG["cv"]["early_stop_wait_gt"]):
                break

    if best_state is None or best_metrics is None or best_diagnostics is None:
        raise RuntimeError(f"No active GCDER checkpoint selected for {group} fold {fold}.")
    state["model"].load_state_dict(best_state["model"], strict=True)
    state["cam"].load_state_dict(best_state["cam"], strict=True)
    module.load_state_dict(best_state["gcder"], strict=True)
    evaluated = evaluate(state, module, bank)
    readback_metrics = _metrics_without_arrays(evaluated["metrics"])
    for metric in METRICS:
        if abs(float(readback_metrics[metric]) - float(best_metrics[metric])) > 1e-10:
            raise AssertionError(f"Worker checkpoint readback mismatch: {group}/{fold} {metric}.")
    checkpoint = _checkpoint_payload(
        group, fold, state, module, semantic_optimizer, best_state,
        readback_metrics, best_diagnostics, bank, prompt_meta,
        reference_folds, warmup, preflight_record, best_epoch,
        epoch + 1, rng_trace, gradient_snapshots,
    )
    atomic_torch_save(checkpoint_path, checkpoint)
    checkpoint_sha = sha_file(checkpoint_path)
    task = {
        "group": group,
        "group_name": GROUP_NAMES[group],
        "fold": fold,
        "seed": 0,
        "status": "COMPLETE",
        "best_epoch": int(best_epoch),
        "epochs_executed": int(epoch + 1),
        "warmup_epochs": 10,
        "warmup_sha256": warmup["sha256"],
        "metrics": readback_metrics,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "source_sha256": task_source_sha,
        "config_sha256": config_sha,
        "prompt_bank_sha256": prompt_meta["bank_sha256"][group],
        "text_cache_sha256": prompt_meta["text_cache_sha256"],
        "reference_checkpoint_sha256": reference_folds[fold]["checkpoint_sha256"],
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_gfr_sha": state["initial_gfr_sha"],
        "initial_cam_sha": state["initial_cam_sha"],
        "initial_added_sha": state["initial_added_sha"],
        "initial_schema_sha": state["initial_schema_sha"],
        "initial_rng_sha": state["construction_rng_sha"],
        "initial_gcder_sha": state["initial_gcder_sha"],
        "split_sha": state["split_sha"],
        "pair_sha": state["cache_meta"]["pair_sha"],
        "anchor_sha": state["anchor_sha"],
        "parameter_sha256": {
            "model": module_sha(state["model"]),
            "cam": module_sha(state["cam"]),
            "gcder": module_sha(module),
        },
        "optimizer_weight_decay": checkpoint["optimizer_weight_decay"],
        "parameter_counts": checkpoint["parameter_counts"],
        "diagnostics": best_diagnostics,
        "rng_trace": rng_trace,
        "gradient_snapshots": gradient_snapshots,
        "main_rng_sha256_at_warmup_entry": main_rng_at_start,
        "worker_readback_status": "PASS",
        "e70_best_checkpoint_used": False,
    }
    atomic_json(task_path, task)
    latest_path.unlink(missing_ok=True)
    _update_manifest_task(group, fold, "COMPLETE", {
        "best_epoch": best_epoch,
        "checkpoint_sha256": checkpoint_sha,
        "warmup_sha256": warmup["sha256"],
    })
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"event": "COMPLETE", "checkpoint_sha256": checkpoint_sha, "best_epoch": best_epoch}, allow_nan=False) + "\n")
        log.flush()
    print(f"COMPLETE {group} fold={fold} best_epoch={best_epoch} ACC={readback_metrics['acc']:.6f}", flush=True)


def run_all(workers: int = 10) -> None:
    if not PRE_FLIGHT.is_file() or json.loads(PRE_FLIGHT.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Formal training requires a passing preflight.")
    worker_count = max(1, min(int(workers), 10, max(1, os.cpu_count() or 1)))
    pending = []
    for group in GROUPS:
        for fold in range(1, 11):
            task_path, checkpoint_path, _latest, _log = group_paths(group, fold)
            if task_path.is_file() and checkpoint_path.is_file():
                task = json.loads(task_path.read_text(encoding="utf-8"))
                if (
                    task.get("status") == "COMPLETE"
                    and task.get("checkpoint_sha256") == sha_file(checkpoint_path)
                    and task.get("source_sha256") == sha_file(EXP / "run.py")
                    and task.get("config_sha256") == sha_file(CONFIG_PATH)
                ):
                    continue
            pending.append((group, fold))
    active: dict[subprocess.Popen, tuple[str, int, Any]] = {}
    failures = []
    log_dir = RUNTIME / "scheduler_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    while pending or active:
        while pending and len(active) < worker_count:
            group, fold = pending.pop(0)
            output = (log_dir / f"{group}_f{fold:02d}.stdout.log").open("ab")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["OMP_NUM_THREADS"] = "2"
            env["MKL_NUM_THREADS"] = "2"
            process = subprocess.Popen(
                [sys.executable, str(EXP / "run.py"), "--worker", "--group", group, "--fold", str(fold)],
                cwd=str(ROOT),
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            active[process] = (group, fold, output)
            atomic_json(RUNTIME / "scheduler.json", {
                "status": "RUNNING",
                "max_workers": worker_count,
                "pending": pending,
                "active": [f"{g}_f{f:02d}" for g, f, _ in active.values()],
                "failed": failures,
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        completed = []
        for process, (group, fold, stream) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            if code != 0:
                failures.append({"group": group, "fold": fold, "exit_code": code})
                _update_manifest_task(group, fold, "FAILED", {"exit_code": code})
            completed.append(process)
        for process in completed:
            del active[process]
        if completed:
            atomic_json(RUNTIME / "scheduler.json", {
                "status": "RUNNING" if pending or active else ("FAILED" if failures else "TRAINING_COMPLETE"),
                "max_workers": worker_count,
                "pending": pending,
                "active": [f"{g}_f{f:02d}" for g, f, _ in active.values()],
                "failed": failures,
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        if active and not completed:
            time.sleep(2)
    if failures:
        raise RuntimeError(f"{len(failures)} fold worker(s) failed; see scheduler_logs and task latest checkpoints.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("completed_tasks") != 30:
        raise AssertionError(f"Formal training ended with {manifest.get('completed_tasks')}/30 complete tasks.")


def smoke_resume_test(reference_manifest, reference_folds, bank) -> dict:
    torch = BASE.runtime()["torch"]
    fold, group = 1, "G1"

    def initialize():
        state = build_base_state(fold, reference_manifest, reference_folds)
        e59_audit = json.loads((E59_DIR / ".runtime/preflight_audit.json").read_text(encoding="utf-8"))["shared_warmup_cache"]
        warm = warmup_info_for_fold(state, fold, e59_audit)
        module = make_gcder(1024, fold)
        state["initial_gcder_sha"] = module_sha(module)
        E59.attach_position_capture(state["cam"])
        opt = torch.optim.AdamW(module.parameters(), lr=0.001, weight_decay=0.001)
        return state, module, opt, warm

    def one_epoch(state, module, opt, epoch):
        state["model"].train()
        state["cam"].train()
        train_step(state, module, opt, bank, epoch)
        with torch.no_grad():
            value = evaluate(state, module, bank)
        return value["logp"].detach().cpu().clone()

    continuous = initialize()
    state_a, module_a, opt_a, warm_a = continuous
    logits_a_1 = one_epoch(state_a, module_a, opt_a, 10)
    snapshot = {
        "model": _cpu_state(state_a["model"]),
        "cam": _cpu_state(state_a["cam"]),
        "gcder": _cpu_state(module_a),
        "optimizer_base": _optimizer_cpu(state_a["opt_base"]),
        "optimizer_cam": _optimizer_cpu(state_a["opt_cam"]),
        "optimizer_added": _optimizer_cpu(state_a["opt_added"]),
        "optimizer_gcder": _optimizer_cpu(opt_a),
        "rng": BASE.pack_rng(),
    }
    logits_a_2 = one_epoch(state_a, module_a, opt_a, 11)
    final_a = {
        "model": module_sha(state_a["model"]),
        "cam": module_sha(state_a["cam"]),
        "gcder": module_sha(module_a),
        "optimizer_base": optimizer_digest(state_a["opt_base"]),
        "optimizer_cam": optimizer_digest(state_a["opt_cam"]),
        "optimizer_added": optimizer_digest(state_a["opt_added"]),
        "optimizer_gcder": optimizer_digest(opt_a),
        "rng": BASE.rng_sha(BASE.pack_rng()),
    }

    resumed = initialize()
    state_b, module_b, opt_b, warm_b = resumed
    if warm_a["sha256"] != warm_b["sha256"]:
        raise AssertionError("Recovery smoke test did not start from the same audited warmup.")
    state_b["model"].load_state_dict(snapshot["model"], strict=True)
    state_b["cam"].load_state_dict(snapshot["cam"], strict=True)
    module_b.load_state_dict(snapshot["gcder"], strict=True)
    state_b["opt_base"].load_state_dict(snapshot["optimizer_base"])
    state_b["opt_cam"].load_state_dict(snapshot["optimizer_cam"])
    state_b["opt_added"].load_state_dict(snapshot["optimizer_added"])
    opt_b.load_state_dict(snapshot["optimizer_gcder"])
    BASE.restore_rng(snapshot["rng"])
    if BASE.rng_sha(BASE.pack_rng()) != BASE.rng_sha(snapshot["rng"]):
        raise AssertionError("Recovery smoke test RNG restore failed.")
    logits_b_2 = one_epoch(state_b, module_b, opt_b, 11)
    final_b = {
        "model": module_sha(state_b["model"]),
        "cam": module_sha(state_b["cam"]),
        "gcder": module_sha(module_b),
        "optimizer_base": optimizer_digest(state_b["opt_base"]),
        "optimizer_cam": optimizer_digest(state_b["opt_cam"]),
        "optimizer_added": optimizer_digest(state_b["opt_added"]),
        "optimizer_gcder": optimizer_digest(opt_b),
        "rng": BASE.rng_sha(BASE.pack_rng()),
    }
    if final_a != final_b or not torch.equal(logits_a_2, logits_b_2):
        state_equal = {key: final_a[key] == final_b[key] for key in final_a}
        max_logit_error = float((logits_a_2 - logits_b_2).abs().max())
        raise AssertionError(
            f"Two-step continuous and checkpoint-resumed training differ: "
            f"state_equal={state_equal}, max_logit_error={max_logit_error}"
        )
    return {
        "status": "PASS",
        "fold": fold,
        "group": group,
        "active_epochs": 2,
        "warmup_sha256": warm_a["sha256"],
        "continuous_state_sha256": sha_obj(final_a),
        "resumed_state_sha256": sha_obj(final_b),
        "continuous_resume_logits_bitwise_equal": True,
        "model_optimizer_rng_state_equal": True,
    }


def _preflight_with_resume() -> dict:
    result = preflight()
    reference_manifest, reference_folds, _ = load_reference()
    banks = BASE.runtime()["torch"].load(PROMPT_CACHE, map_location="cpu", weights_only=False)["banks"]["G1"].to(BASE.runtime()["device"])
    resume = smoke_resume_test(reference_manifest, reference_folds, banks)
    result["active_short_replay_resume"] = resume
    atomic_json(PRE_FLIGHT, result)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["preflight_sha256"] = sha_file(PRE_FLIGHT)
    manifest["resume_test"] = resume
    atomic_json(MANIFEST_PATH, manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.preflight:
        record = _preflight_with_resume()
        print(json.dumps({"status": record["status"], "resume": record["active_short_replay_resume"]}, indent=2))
        return
    if args.worker:
        if not args.group or args.fold is None:
            parser.error("--worker requires --group and --fold")
        worker(args.group, args.fold)
        return
    if args.all:
        run_all(args.workers)
        return
    parser.error("choose --preflight, --worker, or --all")


if __name__ == "__main__":
    main()
