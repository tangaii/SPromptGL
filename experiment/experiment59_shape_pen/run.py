#!/usr/bin/env python3
"""Run the fixed eleven-group Experiment 59 queue on the audited 53.3 base."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
BASE_EXP = ROOT / "experiment/experiment53_3_cadr_graph_fallback_fix"
REFERENCE_EXP = Path("/model/spromptgl/experiment/experiment53_3_cadr_graph_fallback_fix")
CONFIG_PATH = EXP / "config.json"
RUNTIME = EXP / ".runtime"
RESULTS = EXP / "results"
CHECKPOINTS = RUNTIME / "checkpoints"
TASKS = RUNTIME / "tasks"
LOGS = RUNTIME / "logs"
GRADIENT_DIR = RUNTIME / "gradient_checks"
PROMPT_DIR = RUNTIME / "brainprompt"
WARMUP_DIR = RUNTIME / "warmup"

sys.path.insert(0, str(BASE_EXP))
import run as BASE  # noqa: E402

sys.path.insert(0, str(EXP))
from shape_pen import (  # noqa: E402
    DiseaseProjection,
    attach_position_capture,
    class_balanced_weighted_mean,
    cosine_class_loss,
    evidence_weights,
    make_geometry_matched_anonymous_bank,
    project_disease_direction,
    projected_dot_class_loss,
    supervised_contrastive_loss,
)

REFERENCE_GROUP = "CADR_WD_0p01_GFR_WD_0p11"
BASELINE_GROUP = "E59-BASE-53.3"
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
GROUPS = tuple(f"G{i:02d}" for i in range(1, 12))


def cfg() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_obj(value) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_json(path: Path, value) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    atomic_bytes(path, payload)


def atomic_torch_save(path: Path, value) -> None:
    torch = BASE.runtime()["torch"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def group_item(group_id: str) -> dict:
    for item in cfg()["groups"]:
        if item["id"] == group_id:
            return item
    raise KeyError(group_id)


def group_paths(group_id: str, fold: int):
    stem = f"{group_id}_f{fold:02d}"
    return TASKS / f"{stem}.json", CHECKPOINTS / f"{stem}.best.pt", CHECKPOINTS / f"{stem}.latest.pt", LOGS / f"{stem}.log"


def read_reference() -> tuple[dict, dict[int, dict]]:
    manifest_path = REFERENCE_EXP / "manifest_fallback_wd_sweep.json"
    rows_path = REFERENCE_EXP / "results/fallback_wd_sweep/fold_metrics.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {int(row["fold"]): row for row in read_csv(rows_path) if row.get("group") == REFERENCE_GROUP}
    assert manifest["status"] == "COMPLETE_AUDITED"
    assert manifest["independent_checkpoint_audits_passed"] == 40
    assert len(rows) == 10
    item = next(group for group in manifest["groups"] if group["name"] == REFERENCE_GROUP)
    assert float(item["cadr_weight_decay"]) == 0.01 and float(item["gfr_weight_decay"]) == 0.11
    return manifest, rows


def reference_folds(manifest: dict) -> dict[int, dict]:
    return {int(row["fold"]): row for row in manifest["folds"] if row["group"] == REFERENCE_GROUP}


def reference_checkpoint_path(fold: int) -> Path:
    return (
        REFERENCE_EXP
        / ".runtime/fallback_wd_sweep"
        / REFERENCE_GROUP
        / "checkpoints"
        / f"{REFERENCE_GROUP}_f{fold:02d}.best.pt"
    )


def verify_reference_files() -> tuple[dict, dict[int, dict], dict[int, dict]]:
    manifest, metrics = read_reference()
    folds = reference_folds(manifest)
    assert len(folds) == 10
    for fold in range(1, 11):
        task_path = (
            REFERENCE_EXP
            / ".runtime/fallback_wd_sweep"
            / REFERENCE_GROUP
            / "tasks"
            / f"{REFERENCE_GROUP}_f{fold:02d}.json"
        )
        task = json.loads(task_path.read_text(encoding="utf-8"))
        checkpoint = reference_checkpoint_path(fold)
        expected = folds[fold]
        assert task["status"] == "COMPLETE" and task["readback_status"] == "WORKER_READBACK_PASS"
        assert task["best_sha256"] == expected["checkpoint_sha256"] == sha_file(checkpoint)
        for key in ("split_sha", "pair_sha", "initial_shared_sha", "initial_added_sha", "initial_rng_sha"):
            assert task[key] == expected[key], (fold, key, task[key], expected[key])
        assert float(task["cadr_weight_decay"]) == 0.01 and float(task["gfr_weight_decay"]) == 0.11
        assert all(float(value) == 0.11 for value in task["optimizer_weight_decay"]["base"])
        assert all(float(value) == 0.11 for value in task["optimizer_weight_decay"]["cam"])
        assert all(float(value) == 0.01 for value in task["optimizer_weight_decay"]["added"])
    return manifest, metrics, folds


def build_base_state(fold: int, manifest: dict, ref_folds: dict[int, dict]):
    torch = BASE.runtime()["torch"]
    state = BASE.build_state(fold, BASE.prepare_fold(fold))
    for group in state["opt_base"].param_groups:
        group["weight_decay"] = 0.11
    for group in state["opt_cam"].param_groups:
        group["weight_decay"] = 0.11
    for group in state["opt_added"].param_groups:
        group["weight_decay"] = 0.01
    expected = ref_folds[fold]
    assert state["split_sha"] == expected["split_sha"]
    assert state["cache_meta"]["pair_sha"] == expected["pair_sha"]
    assert state["initial_shared_sha"] == expected["initial_shared_sha"]
    assert state["initial_added_sha"] == expected["initial_added_sha"]
    assert state["construction_rng_sha"] == expected["initial_rng_sha"]
    assert state["anchor_sha"] == json.loads(
        (ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors/index.json").read_text()
    )["anchors"][fold - 1]["anchor_sha256"]
    assert all(float(group["weight_decay"]) == 0.11 for group in state["opt_base"].param_groups)
    assert all(float(group["weight_decay"]) == 0.11 for group in state["opt_cam"].param_groups)
    assert all(float(group["weight_decay"]) == 0.01 for group in state["opt_added"].param_groups)
    state["group"] = REFERENCE_GROUP
    return state


def prompt_assets() -> tuple[object, object, dict]:
    torch = BASE.runtime()["torch"]
    text_path = PROMPT_DIR / "abide_label.txt"
    embedding_path = PROMPT_DIR / "abide_label.pt"
    if not text_path.is_file() or not embedding_path.is_file():
        raise FileNotFoundError("Copy the official BrainPrompt ABIDE prompt text and cached embedding into .runtime/brainprompt.")
    texts = [line.strip() for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    official = torch.load(embedding_path, map_location="cpu", weights_only=True)
    if not isinstance(official, (list, tuple)) or len(official) != 2:
        raise ValueError("Official BrainPrompt ABIDE file must contain two ordered embeddings.")
    official = torch.stack([torch.as_tensor(value).float().reshape(-1) for value in official])
    if len(texts) != 2 or official.shape != (2, 2048) or not torch.isfinite(official).all():
        raise ValueError(f"Unexpected official prompt content or embedding shape: {len(texts)} / {tuple(official.shape)}")
    real = official[[1, 0]].contiguous()
    cosine = float(torch.nn.functional.cosine_similarity(real[0:1], real[1:2]))
    norms = [float(value) for value in real.norm(dim=-1)]
    anon_path = PROMPT_DIR / "anonymous_geometry_matched.pt"
    if anon_path.is_file():
        anonymous = torch.load(anon_path, map_location="cpu", weights_only=True)
    else:
        anonymous = make_geometry_matched_anonymous_bank(real, seed=590059)
        atomic_torch_save(anon_path, anonymous)
    anonymous = torch.as_tensor(anonymous, dtype=torch.float32)
    anon_cosine = float(torch.nn.functional.cosine_similarity(anonymous[0:1], anonymous[1:2]))
    anon_norms = [float(value) for value in anonymous.norm(dim=-1)]
    if anonymous.shape != real.shape or max(abs(a - b) for a, b in zip(norms, anon_norms)) > 1e-4:
        raise AssertionError("Anonymous prototype norms do not match the official text vectors.")
    if abs(cosine - anon_cosine) > 1e-5:
        raise AssertionError("Anonymous prototype cosine does not match the official text pair.")
    info = {
        "official_order": ["TC", "ASD"],
        "training_order": ["ASD", "TD"],
        "index_map": [1, 0],
        "embedding_shape_official": list(official.shape),
        "embedding_sha256": sha_file(embedding_path),
        "text_sha256": sha_file(text_path),
        "anonymous_sha256": sha_file(anon_path),
        "real_cosine": cosine,
        "anonymous_cosine": anon_cosine,
        "real_norms": norms,
        "anonymous_norms": anon_norms,
        "source_commit": "c85304a023636a2701d465e4cb0acb13b637127e",
        "dimension": 2048,
    }
    return real, anonymous, info


def isolated_module(group_id: str, fold: int, real: object, anonymous: object, config: dict):
    if group_id in {"G05", "G08"}:
        return None, "none"
    torch = BASE.runtime()["torch"]
    item = group_item(group_id)
    bank_type = item["text"]
    if bank_type == "real":
        bank, learnable = real, False
    elif bank_type == "anonymous_fixed_geometry_matched":
        bank, learnable = anonymous, False
    elif bank_type == "learnable_numeric_prototypes":
        bank, learnable = anonymous, True
    else:
        raise ValueError(f"Unknown prototype source {bank_type} for {group_id}.")
    device = BASE.runtime()["device"]
    cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    rng_before = BASE.pack_rng()
    seed = int(config["semantic_projection"]["initialization_seed"]) + fold
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(seed)
        module = DiseaseProjection(bank, learnable=learnable).to(device)
    rng_after = BASE.pack_rng()
    if BASE.rng_sha(rng_before) != BASE.rng_sha(rng_after):
        raise AssertionError("Semantic module initialization changed the shared training RNG.")
    return module, bank_type


def parameter_sha(module) -> str:
    if module is None:
        return "none"
    values = {name: value.detach().cpu().contiguous() for name, value in module.state_dict().items()}
    return hashlib.sha256(pickle.dumps(values, protocol=4)).hexdigest()


def private_shuffle(fold: int, train_indices, labels) -> dict:
    torch = BASE.runtime()["torch"]
    train_indices = torch.as_tensor(train_indices, dtype=torch.long).cpu()
    train_labels = torch.as_tensor(labels, dtype=torch.long).cpu()[train_indices]
    generator = torch.Generator(device="cpu").manual_seed(590059 + 1000 + fold)
    mapping = torch.arange(len(train_indices), dtype=torch.long)
    for label in (0, 1):
        local = torch.nonzero(train_labels == label, as_tuple=False).reshape(-1)
        mapping[local] = local[torch.randperm(len(local), generator=generator)]
    return {
        "fold": fold,
        "train_indices": train_indices.tolist(),
        "source_positions": mapping.tolist(),
        "labels_sha256": hashlib.sha256(train_labels.numpy().tobytes()).hexdigest(),
        "source_positions_sha256": hashlib.sha256(mapping.numpy().tobytes()).hexdigest(),
        "generator": "private CPU torch.Generator",
        "seed": 590059 + 1000 + fold,
    }


def prepare_shuffles() -> dict[int, dict]:
    torch = BASE.runtime()["torch"]
    _data, y = BASE.load_data()[0]
    folds = BASE.load_data()[1]
    saved: dict[int, dict] = {}
    for fold, (train, _test) in enumerate(folds, start=1):
        path = RUNTIME / "shuffle" / f"fold{fold:02d}.json"
        current = private_shuffle(fold, train, torch.as_tensor(y))
        if path.is_file():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old != current:
                raise AssertionError(f"G09 frozen within-class permutation changed at fold {fold}.")
        else:
            atomic_json(path, current)
        saved[fold] = current
    return saved


def shuffled_rho(rho, labels, permutation: dict):
    torch = BASE.runtime()["torch"]
    mapping = torch.as_tensor(permutation["source_positions"], dtype=torch.long, device=rho.device)
    if len(mapping) != len(rho):
        raise AssertionError("G09 permutation length differs from training rho.")
    if not torch.equal(labels, labels[mapping]):
        raise AssertionError("G09 evidence permutation crossed diagnostic classes.")
    return rho[mapping]


def _python_float(value) -> float:
    return float(value.detach().cpu()) if hasattr(value, "detach") else float(value)


def _evidence(state, positions, config):
    train = state["train"]
    return evidence_weights(
        state["cam"],
        positions,
        train,
        state["y"][train],
        train,
        exclude_self=True,
        rho_min=float(config["semantic_projection"]["rho_min"]),
        scale_floor=float(config["semantic_projection"]["rho_abs_scale_floor"]),
        scale_quantile=float(config["semantic_projection"]["rho_scale_quantile"]),
    )


def auxiliary_loss(group_id: str, state, out: dict, module, anonymous_type: str, config: dict, permutation: dict):
    torch = BASE.runtime()["torch"]
    import torch.nn.functional as F

    train = state["train"]
    labels = state["y"][train]
    positions = state["cam"]._e59_positions
    representation = positions.reshape(positions.shape[0], -1)[train]
    diagnostics: dict = {"group": group_id, "rho_source": "none"}
    evidence = None
    if group_id in {"G07", "G08", "G09", "G10", "G11"}:
        evidence = _evidence(state, positions, config)
        rho = evidence["rho"]
        diagnostics.update({
            "rho_source": "patient_evidence",
            "rho_mean": _python_float(rho.mean()),
            "rho_std": _python_float(rho.std(unbiased=False)),
            "rho_min": _python_float(rho.min()),
            "rho_max": _python_float(rho.max()),
            "rho_asd": _python_float(rho[labels == 0].mean()),
            "rho_td": _python_float(rho[labels == 1].mean()),
            "rho_floor_fraction": _python_float((rho <= float(config["semantic_projection"]["rho_min"]) + 1e-7).float().mean()),
            "rho_ceiling_fraction": _python_float((rho >= 1.0 - 1e-7).float().mean()),
            "evidence_scale": _python_float(evidence["scale"]),
            "positive_fraction": _python_float(evidence["positive_fraction"].mean()),
            "delta_mean_by_position": evidence["delta"].mean(dim=0).detach().cpu().tolist(),
            "delta_positive_fraction_by_position": (evidence["delta"] > 0).float().mean(dim=0).detach().cpu().tolist(),
            "delta_conflict_fraction_by_position": (evidence["delta"] < 0).float().mean(dim=0).detach().cpu().tolist(),
            "delta_near_zero_fraction_by_position": (evidence["delta"].abs() < 0.01).float().mean(dim=0).detach().cpu().tolist(),
        })
    if group_id in {"G01", "G02", "G03", "G04"}:
        prototypes = module.prototypes()
        temperature = 1.0 if group_id == "G01" else float(config["semantic_projection"]["saber_temperature"])
        loss, logits, per_patient = cosine_class_loss(representation, labels, prototypes, temperature)
        diagnostics.update(_prototype_diagnostics(prototypes, module, anonymous_type))
        diagnostics["temperature"] = temperature
    elif group_id == "G05":
        loss = supervised_contrastive_loss(
            representation,
            labels,
            float(config["semantic_projection"]["supcon_temperature"]),
        )
        diagnostics["temperature"] = float(config["semantic_projection"]["supcon_temperature"])
        logits = None
        per_patient = None
    elif group_id in {"G06", "G07", "G09", "G10", "G11"}:
        prototypes = module.prototypes()
        if group_id == "G07":
            aligned = representation
            rho_for_loss = evidence["rho"]
        else:
            aligned, residual, direction = project_disease_direction(representation, prototypes)
            rho_for_loss = torch.ones_like(labels, dtype=representation.dtype)
            if group_id in {"G09", "G10", "G11"}:
                rho_for_loss = evidence["rho"]
            if group_id == "G09":
                rho_for_loss = shuffled_rho(rho_for_loss, labels, permutation)
            diagnostics.update({
                "rho_source": "uniform" if group_id == "G06" else ("within_class_fixed_shuffle" if group_id == "G09" else "patient_evidence"),
                "projection_direction_norm": _python_float((prototypes[0] - prototypes[1]).norm()),
                "decomposition_max_error": _python_float((representation - aligned - residual).abs().max()),
                "decomposition_max_dot": _python_float((aligned * residual).sum(dim=-1).abs().max()),
                "disease_component_norm_mean": _python_float(aligned.norm(dim=-1).mean()),
                "individual_residual_norm_mean": _python_float(residual.norm(dim=-1).mean()),
            })
        loss, logits, per_patient = projected_dot_class_loss(
            aligned,
            labels,
            prototypes,
            float(config["semantic_projection"]["shape_temperature"]),
        )
        if group_id == "G09":
            diagnostics["rho_mean"] = _python_float(rho_for_loss.mean())
            diagnostics["rho_asd"] = _python_float(rho_for_loss[labels == 0].mean())
            diagnostics["rho_td"] = _python_float(rho_for_loss[labels == 1].mean())
        loss = class_balanced_weighted_mean(per_patient, labels, rho_for_loss)
        diagnostics.update(_prototype_diagnostics(prototypes, module, anonymous_type))
        diagnostics["temperature"] = float(config["semantic_projection"]["shape_temperature"])
    elif group_id == "G08":
        per_patient = F.nll_loss(out["logp"][train], labels, reduction="none")
        loss = class_balanced_weighted_mean(per_patient, labels, evidence["rho"])
        logits = None
        diagnostics["rho_source"] = "patient_evidence"
    else:
        raise KeyError(group_id)
    diagnostics["loss_unscaled"] = _python_float(loss)
    return loss, diagnostics


def _prototype_diagnostics(prototypes, module, anonymous_type: str) -> dict:
    torch = BASE.runtime()["torch"]
    return {
        "prototype_source": anonymous_type,
        "projected_prototype_cosine": _python_float(torch.nn.functional.cosine_similarity(prototypes[0:1], prototypes[1:2])),
        "projected_prototype_norm_asd": _python_float(prototypes[0].norm()),
        "projected_prototype_norm_td": _python_float(prototypes[1].norm()),
        "projection_parameter_count": sum(parameter.numel() for parameter in module.parameters()),
        "learnable_numeric_parameter_count": module.label_embeddings.numel() if isinstance(module.label_embeddings, torch.nn.Parameter) else 0,
    }


def _coefficient_key(group_id: str) -> str:
    if group_id == "G01":
        return "brainprompt_style"
    if group_id in {"G02", "G03", "G04"}:
        return "saber_disease_cl"
    if group_id == "G05":
        return "supervised_contrastive"
    return "shape_family"


def train_forward(group_id: str, state, module, anonymous_type: str, config: dict, permutation: dict, active: bool):
    out = BASE.forward_state(state, training=True, use_cadr=active)
    out["aux_loss"] = None
    out["aux_diagnostics"] = {"group": group_id}
    out["coefficient"] = 0.0
    if active:
        loss, diagnostics = auxiliary_loss(group_id, state, out, module, anonymous_type, config, permutation)
        coefficient = float(config["semantic_projection"]["coefficients"][_coefficient_key(group_id)])
        out["total"] = out["total"] + coefficient * loss
        out["aux_loss"] = loss
        out["aux_diagnostics"] = diagnostics
        out["coefficient"] = coefficient
    return out


def _norm_and_cosine(gradients: list, other: list) -> tuple[float, float]:
    torch = BASE.runtime()["torch"]
    if len(gradients) != len(other):
        raise ValueError("Gradient vectors must preserve the same parameter ordering.")
    left, right = [], []
    for left_gradient, right_gradient in zip(gradients, other):
        if left_gradient is None and right_gradient is None:
            continue
        if left_gradient is None:
            left_gradient = torch.zeros_like(right_gradient)
        if right_gradient is None:
            right_gradient = torch.zeros_like(left_gradient)
        if left_gradient.shape != right_gradient.shape:
            raise ValueError("Classification and auxiliary gradients are not aligned by parameter.")
        left.append(left_gradient.detach().float().reshape(-1))
        right.append(right_gradient.detach().float().reshape(-1))
    if not left:
        return 0.0, 0.0
    left_flat = torch.cat(left)
    right_flat = torch.cat(right)
    norm_left = float(left_flat.norm().cpu())
    norm_right = float(right_flat.norm().cpu())
    cosine = float((torch.dot(left_flat, right_flat) / (left_flat.norm() * right_flat.norm()).clamp_min(1e-12)).cpu())
    return norm_left, (norm_right, cosine)


def gradient_probe(group_id: str, state, out: dict) -> dict:
    torch = BASE.runtime()["torch"]
    aux = out["aux_loss"]
    if aux is None:
        return {"group": group_id, "epoch": None, "status": "NO_AUXILIARY_OBJECTIVE"}
    groups = {
        "representation": [state["cam"]._e59_positions],
        "cam": list(state["cam"].parameters()),
        "gfr": list(state["base_params"]),
        "cadr": [parameter for _name, parameter in state["added_named"]],
    }
    result: dict = {"group": group_id}
    for name, parameters in groups.items():
        cls_grads = torch.autograd.grad(out["cls"], parameters, retain_graph=True, allow_unused=True)
        aux_grads = torch.autograd.grad(aux, parameters, retain_graph=True, allow_unused=True)
        cls_norm, (aux_norm, cosine) = _norm_and_cosine(cls_grads, aux_grads)
        result[f"{name}_classification_grad_norm"] = cls_norm
        result[f"{name}_auxiliary_grad_norm"] = aux_norm
        result[f"{name}_auxiliary_to_classification_norm_ratio"] = aux_norm / max(cls_norm, 1e-12)
        result[f"{name}_gradient_cosine"] = cosine
    return result


def _check_optimizer_digest(optimizer) -> str:
    torch = BASE.runtime()["torch"]
    digest = hashlib.sha256()

    def add(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(b"tensor\0" + str(tensor.dtype).encode() + b"\0")
            digest.update(repr(tuple(tensor.shape)).encode() + b"\0")
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict\0")
            for key in sorted(value, key=lambda item: repr(item)):
                add(key)
                add(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(("list\0" if isinstance(value, list) else "tuple\0").encode())
            for item in value:
                add(item)
        else:
            digest.update(type(value).__name__.encode() + b"\0" + repr(value).encode() + b"\0")

    add(optimizer.state_dict())
    return digest.hexdigest()


def _step_warmup(state, captured: bool, steps: int = 2) -> dict:
    torch = BASE.runtime()["torch"]
    if captured:
        attach_position_capture(state["cam"])
    trace = []
    for _ in range(steps):
        for name in ("opt_base", "opt_cam", "opt_added"):
            state[name].zero_grad(set_to_none=True)
        out = BASE.forward_state(state, training=True, use_cadr=False)
        trace.append({
            key: value.detach().cpu().clone()
            for key, value in out.items()
            if key in {"fused", "logp", "hidden", "cls", "graph", "pair", "adv", "preserve", "total"}
            and torch.is_tensor(value)
        })
        out["total"].backward()
        state["opt_base"].step()
        state["opt_cam"].step()
        state["opt_added"].step()
        with torch.no_grad():
            evaluated = BASE.forward_state(state, training=False, use_cadr=False)
            metrics = BASE.classification_metrics(evaluated["logp"], state["test"], state["y"])
        trace[-1]["warmup_eval_logp"] = evaluated["logp"].detach().cpu().clone()
        trace[-1]["warmup_eval_metrics"] = {key: value for key, value in metrics.items() if key not in {"pred", "prob"}}
        if captured:
            captured_positions = state["cam"]._e59_positions
            if captured_positions is None or tuple(captured_positions.shape) != (871, 4, 36):
                raise AssertionError("CAM capture did not return the expected 871x4x36 positions.")
    trace.append({
        "model_sha": BASE.digest_named_parameters(state["model"]),
        "cam_sha": BASE.digest_named_parameters(state["cam"]),
        "opt_base_sha": _check_optimizer_digest(state["opt_base"]),
        "opt_cam_sha": _check_optimizer_digest(state["opt_cam"]),
        "opt_added_sha": _check_optimizer_digest(state["opt_added"]),
    })
    return {"trace": trace, "rng_sha": BASE.rng_sha(BASE.pack_rng())}


def warmup_cache_path(fold: int) -> Path:
    return WARMUP_DIR / f"fold{fold:02d}.pt"


def _warmup_digest(state) -> dict:
    return {
        "model_sha": BASE.digest_named_parameters(state["model"]),
        "cam_sha": BASE.digest_named_parameters(state["cam"]),
        "opt_base_sha": _check_optimizer_digest(state["opt_base"]),
        "opt_cam_sha": _check_optimizer_digest(state["opt_cam"]),
        "opt_added_sha": _check_optimizer_digest(state["opt_added"]),
        "rng_sha": BASE.rng_sha(BASE.pack_rng()),
    }


def prepare_warmup_cache(manifest: dict, ref_folds: dict[int, dict]) -> dict:
    """Create one verified, immutable common warmup state per outer fold."""
    torch = BASE.runtime()["torch"]
    WARMUP_DIR.mkdir(parents=True, exist_ok=True)
    audits = {}
    for fold in range(1, 11):
        path = warmup_cache_path(fold)
        normal = build_base_state(fold, manifest, ref_folds)
        captured = build_base_state(fold, manifest, ref_folds)
        attach_position_capture(captured["cam"])
        results = {}
        for current in (normal, captured):
            BASE.restore_rng(current["construction_rng"])
            results["captured" if current is captured else "normal"] = _step_warmup(
                current, captured=current is captured, steps=int(cfg()["warmup_epochs"])
            )
        for step_index, (normal_step, captured_step) in enumerate(zip(results["normal"]["trace"][:-1], results["captured"]["trace"][:-1])):
            if normal_step.keys() != captured_step.keys():
                raise AssertionError(f"Fold {fold} warmup output keys differ at step {step_index}.")
            for key in normal_step:
                if torch.is_tensor(normal_step[key]):
                    if not torch.equal(normal_step[key], captured_step[key]):
                        raise AssertionError(f"Fold {fold} warmup forward differs at step {step_index}: {key}.")
                elif normal_step[key] != captured_step[key]:
                    raise AssertionError(f"Fold {fold} warmup metric differs at step {step_index}: {key}.")
        if results["normal"]["rng_sha"] != results["captured"]["rng_sha"]:
            raise AssertionError(f"Fold {fold} warmup RNG diverged with CAM position capture.")
        normal_digest = _warmup_digest(normal)
        captured_digest = _warmup_digest(captured)
        if normal_digest != captured_digest:
            raise AssertionError(f"Fold {fold} common 10-epoch warmup diverged with CAM capture: {normal_digest} != {captured_digest}")
        payload = {
            "fold": fold,
            "group": "COMMON_WARMUP",
            "warmup_epochs": int(cfg()["warmup_epochs"]),
            "split_sha": captured["split_sha"],
            "pair_sha": captured["cache_meta"]["pair_sha"],
            "anchor_sha": captured["anchor_sha"],
            "initial_shared_sha": captured["initial_shared_sha"],
            "initial_added_sha": captured["initial_added_sha"],
            "initial_rng_sha": captured["construction_rng_sha"],
            "model": BASE._state_cpu(captured["model"]),
            "cam": BASE._state_cpu(captured["cam"]),
            "optimizer_base": _optimizer_cpu_state(captured["opt_base"]),
            "optimizer_cam": _optimizer_cpu_state(captured["opt_cam"]),
            "optimizer_added": _optimizer_cpu_state(captured["opt_added"]),
            "rng": BASE.pack_rng(),
            "final_digest": captured_digest,
            "warmup_best_acc": max(step["warmup_eval_metrics"]["acc"] for step in results["captured"]["trace"][:-1]),
            "warmup_best_epoch": max(
                range(int(cfg()["warmup_epochs"])),
                key=lambda epoch: results["captured"]["trace"][epoch]["warmup_eval_metrics"]["acc"],
            ),
        }
        atomic_torch_save(path, payload)
        readback = torch.load(path, map_location="cpu", weights_only=False)
        if readback["final_digest"] != captured_digest or readback["split_sha"] != captured["split_sha"]:
            raise AssertionError(f"Fold {fold} common warmup cache readback failed.")
        audits[str(fold)] = {
            "status": "PASS",
            "path": str(path),
            "sha256": sha_file(path),
            "normal_capture_digest": normal_digest,
            "captured_digest": captured_digest,
            "readback": "PASS",
        }
        del normal, captured, payload, readback, results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"status": "PASS", "fold_count": 10, "folds": audits}


def load_warmup_cache_into_state(state, fold: int, config: dict) -> dict:
    torch = BASE.runtime()["torch"]
    path = warmup_cache_path(fold)
    if not path.is_file():
        raise FileNotFoundError(f"Verified common warmup cache missing for fold {fold}: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "fold": fold,
        "group": "COMMON_WARMUP",
        "warmup_epochs": int(config["warmup_epochs"]),
        "split_sha": state["split_sha"],
        "pair_sha": state["cache_meta"]["pair_sha"],
        "anchor_sha": state["anchor_sha"],
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_added_sha": state["initial_added_sha"],
        "initial_rng_sha": state["construction_rng_sha"],
    }
    for key, expected in required.items():
        if saved.get(key) != expected:
            raise AssertionError(f"Fold {fold} warmup cache provenance mismatch for {key}.")
    state["model"].load_state_dict(saved["model"], strict=True)
    state["cam"].load_state_dict(saved["cam"], strict=True)
    state["opt_base"].load_state_dict(saved["optimizer_base"])
    state["opt_cam"].load_state_dict(saved["optimizer_cam"])
    state["opt_added"].load_state_dict(saved["optimizer_added"])
    BASE.restore_rng(saved["rng"])
    current_digest = _warmup_digest(state)
    if current_digest != saved["final_digest"]:
        raise AssertionError(f"Fold {fold} loaded common warmup state failed parameter/optimizer/RNG verification.")
    return {
        "sha256": sha_file(path),
        "final_digest": current_digest,
        "warmup_best_acc": float(saved["warmup_best_acc"]),
        "warmup_best_epoch": int(saved["warmup_best_epoch"]),
    }


def validate_warmup_capture_parity(manifest: dict, ref_folds: dict[int, dict]) -> dict:
    torch = BASE.runtime()["torch"]
    _prompt, anonymous, _info = prompt_assets()
    normal = build_base_state(1, manifest, ref_folds)
    captured = build_base_state(1, manifest, ref_folds)
    module, _ = isolated_module("G11", 1, _prompt, anonymous, cfg())
    module_before = BASE.pack_rng()
    module_sha = parameter_sha(module)
    if BASE.rng_sha(module_before) != BASE.rng_sha(BASE.pack_rng()):
        raise AssertionError("Creating the isolated semantic module changed shared RNG.")
    # Build the same private module from the already-saved initial state above.
    BASE.restore_rng(captured["construction_rng"])
    captured_rng = BASE.pack_rng()
    BASE.restore_rng(normal["construction_rng"])
    if BASE.rng_sha(captured_rng) != BASE.rng_sha(BASE.pack_rng()):
        raise AssertionError("Fold-1 shared construction RNG does not replay.")
    normal_result = _step_warmup(normal, captured=False, steps=2)
    final_normal_rng = BASE.pack_rng()
    BASE.restore_rng(captured["construction_rng"])
    captured_result = _step_warmup(captured, captured=True, steps=2)
    final_capture_rng = BASE.pack_rng()
    if len(normal_result["trace"]) != len(captured_result["trace"]):
        raise AssertionError("Warmup parity trace length mismatch.")
    for step_a, step_b in zip(normal_result["trace"][:2], captured_result["trace"][:2]):
        if step_a.keys() != step_b.keys():
            raise AssertionError("Warmup parity output keys differ.")
        for key in step_a:
            if torch.is_tensor(step_a[key]):
                if not torch.equal(step_a[key], step_b[key]):
                    raise AssertionError(f"CAM position capture changed shared warmup output {key}.")
            elif step_a[key] != step_b[key]:
                raise AssertionError(f"CAM position capture changed shared warmup value {key}.")
    if normal_result["trace"][-1] != captured_result["trace"][-1]:
        raise AssertionError(
            "Warmup parameters or optimizer states diverged after CAM capture: "
            f"normal={normal_result['trace'][-1]}, captured={captured_result['trace'][-1]}"
        )
    if BASE.rng_sha(final_normal_rng) != BASE.rng_sha(final_capture_rng):
        raise AssertionError("CAM position capture changed the shared warmup RNG trajectory.")
    module, _ = isolated_module("G11", 1, _prompt, anonymous, cfg())
    out = BASE.forward_state(captured, training=True, use_cadr=True)
    evidence_rng = BASE.pack_rng()
    auxiliary_loss("G11", captured, out, module, "real", cfg(), private_shuffle(1, captured["train"], captured["y"]))
    evidence_rng_after = BASE.pack_rng()
    if BASE.rng_sha(evidence_rng) != BASE.rng_sha(evidence_rng_after):
        raise AssertionError("Deterministic four-position evidence path consumed main RNG.")
    # Avoid retaining the smoke graph and its captured activation.
    del out, module, normal, captured
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "status": "PASS",
        "steps": 2,
        "stochastic_cam_forward_equal": True,
        "base_and_cam_parameters_equal": True,
        "base_optimizer_states_equal": True,
        "shared_rng_equal": True,
        "captured_positions_shape": [871, 4, 36],
        "deterministic_evidence_rng_unchanged": True,
        "isolated_module_initial_sha": module_sha,
    }


def deterministic_math_preflight(real, anonymous, state) -> dict:
    torch = BASE.runtime()["torch"]
    device = BASE.runtime()["device"]
    module, _ = isolated_module("G11", state["fold"], real, anonymous, cfg())
    prototypes = module.prototypes()
    z = torch.linspace(-1.3, 1.7, steps=12 * 144, device=device).reshape(12, 144).requires_grad_(True)
    labels = torch.tensor([0, 1] * 6, dtype=torch.long, device=device)
    disease, residual, direction = project_disease_direction(z, prototypes)
    loss, _logits, _per_patient = projected_dot_class_loss(disease, labels, prototypes, temperature=0.1)
    grad = torch.autograd.grad(loss, [z, module.label_transform.weight], allow_unused=False)
    error = float((z - disease - residual).abs().max().detach().cpu())
    orthogonality = float((disease * residual).sum(dim=-1).abs().max().detach().cpu())
    prototype_norms = prototypes.norm(dim=-1)
    checks = {
        "prototype_shape": list(prototypes.shape) == [2, 144],
        "prototype_unit_norm": bool(torch.allclose(prototype_norms, torch.ones_like(prototype_norms), atol=1e-6, rtol=0.0)),
        "direction_nonzero": bool(direction.norm() > 0.9),
        "decomposition_reconstructs": error < 1e-6,
        "decomposition_orthogonal": orthogonality < 1e-5,
        "decomposition_backpropagates": all(bool(value.norm() > 0) for value in grad),
        "finite_semantic_loss": bool(torch.isfinite(loss)),
        "anonymous_geometry_matched": bool(torch.allclose(
            torch.nn.functional.cosine_similarity(real[0:1], real[1:2]),
            torch.nn.functional.cosine_similarity(anonymous[0:1], anonymous[1:2]), atol=1e-5, rtol=0.0
        )),
    }
    if not all(checks.values()):
        raise AssertionError(f"SHAPE mathematics preflight failed: {checks}")
    return {
        "status": "PASS",
        "checks": checks,
        "max_reconstruction_error": error,
        "max_abs_orthogonality_dot": orthogonality,
        "real_raw_cosine": float(torch.nn.functional.cosine_similarity(real[0:1], real[1:2])),
        "anonymous_raw_cosine": float(torch.nn.functional.cosine_similarity(anonymous[0:1], anonymous[1:2])),
        "initial_projected_cosine": _python_float(torch.nn.functional.cosine_similarity(prototypes[0:1], prototypes[1:2])),
        "train_class_counts": [int((state["y"][state["train"]] == label).sum().item()) for label in (0, 1)],
    }


def calibrate_coefficients(manifest: dict, ref_folds: dict[int, dict], real, anonymous, shuffles: dict[int, dict]) -> dict:
    torch = BASE.runtime()["torch"]
    import torch.nn.functional as F

    config = cfg()
    state = build_base_state(1, manifest, ref_folds)
    attach_position_capture(state["cam"])
    warmup_info = load_warmup_cache_into_state(state, 1, config)
    modules: dict[str, tuple[object, str]] = {}
    for group_id in GROUPS:
        modules[group_id] = isolated_module(group_id, 1, real, anonymous, config)
    out = BASE.forward_state(state, training=True, use_cadr=True)
    positions = state["cam"]._e59_positions
    train_labels = state["y"][state["train"]]
    base_grad = torch.autograd.grad(out["cls"], positions, retain_graph=True)[0]
    base_norm = float(base_grad.norm().detach().cpu())
    if not math.isfinite(base_norm) or base_norm <= 1e-12:
        raise AssertionError("Original classification gradient at the 144-d CAM representation is unusable.")

    unit_ratios: dict[str, float] = {}
    unit_losses: dict[str, float] = {}
    for group_id in GROUPS:
        module, bank_type = modules[group_id]
        auxiliary, diagnostics = auxiliary_loss(group_id, state, out, module, bank_type, config, shuffles[1])
        aux_grad = torch.autograd.grad(auxiliary, positions, retain_graph=True, allow_unused=True)[0]
        aux_norm = float(aux_grad.norm().detach().cpu()) if aux_grad is not None else 0.0
        if not math.isfinite(aux_norm) or aux_norm <= 1e-12:
            raise AssertionError(f"Unscaled auxiliary objective has negligible gradient: {group_id}.")
        unit_ratios[group_id] = aux_norm / base_norm
        unit_losses[group_id] = _python_float(auxiliary)

    target = float(config["semantic_projection"]["gradient_target_ratio_to_original_cls"])
    coefficients = {
        "brainprompt_style": target / unit_ratios["G01"],
        "saber_disease_cl": target / max(unit_ratios[group] for group in ("G02", "G03", "G04")),
        "supervised_contrastive": target / unit_ratios["G05"],
        "shape_family": target / max(unit_ratios[group] for group in ("G06", "G07", "G08", "G09", "G10", "G11")),
    }
    final_ratios: dict[str, float] = {}
    for group_id in GROUPS:
        final_ratios[group_id] = unit_ratios[group_id] * coefficients[_coefficient_key(group_id)]
    if any(not math.isfinite(value) or value > 0.25 or value < 0.001 for value in final_ratios.values()):
        raise AssertionError(f"Frozen family coefficients produce unusable gradient ratios: {final_ratios}")
    if any(not math.isfinite(value) or value > 10.0 or value < 1e-6 for value in coefficients.values()):
        raise AssertionError(f"Gradient scale requires an unreasonable auxiliary coefficient: {coefficients}")
    config["semantic_projection"]["coefficients"] = coefficients
    config["semantic_projection"]["coefficients_frozen"] = True
    config["status"] = "PREFLIGHT_PASS_CONFIG_FROZEN"
    preflight = {
        "status": "PASS",
        "calibration_fold": 1,
        "labels_used": "outer-training labels only",
        "validation_or_test_metrics_used": False,
        "warmup_steps": int(config["warmup_epochs"]),
        "shared_warmup_cache_sha256": warmup_info["sha256"],
        "base_classification_gradient_norm_at_CAM": base_norm,
        "target_auxiliary_to_classification_gradient_ratio": target,
        "unit_auxiliary_gradient_ratios": unit_ratios,
        "frozen_gradient_ratios": final_ratios,
        "unscaled_losses": unit_losses,
        "frozen_coefficients": coefficients,
        "brainprompt_official_default_lambda1": 0.001,
        "brainprompt_adaptation": "coefficient calibrated to the common 144-d representation gradient scale; reported as BrainPrompt-style, not an exact hyperparameter reproduction",
        "saber_temperature_adaptation": config["semantic_projection"]["saber_temperature"],
        "shape_temperature": config["semantic_projection"]["shape_temperature"],
        "supcon_temperature": config["semantic_projection"]["supcon_temperature"],
        "coefficient_rule": "target_ratio / maximum unscaled family gradient ratio",
        "note": "The fold is used only for numeric and gradient-scale preflight; no outer validation/test performance selects a coefficient.",
    }
    config["preflight"] = preflight
    atomic_json(CONFIG_PATH, config)
    preflight["frozen_config_sha256"] = sha_file(CONFIG_PATH)
    preflight["status"] = "PASS"
    atomic_json(RUNTIME / "preflight.json", preflight)
    rows = [
        {"check": "auxiliary_gradient_calibration", "group": group_id, "unit_ratio": unit_ratios[group_id], "frozen_ratio": final_ratios[group_id], "coefficient_family": _coefficient_key(group_id), "coefficient": coefficients[_coefficient_key(group_id)], "status": "PASS"}
        for group_id in GROUPS
    ]
    atomic_csv(RESULTS / "preflight.csv", rows)
    del out, state, modules
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preflight


def source_hashes() -> dict:
    runtime = BASE.runtime()
    return {
        "base_run.py": sha_file(BASE_EXP / "run.py"),
        "base_cadr_gfr.py": sha_file(BASE_EXP / "cadr_gfr.py"),
        "base_config.json": sha_file(BASE_EXP / "config.json"),
        "base_fallback_runner.py": sha_file(BASE_EXP / "run_cadr_fallback.py"),
        "base_wd_sweep_runner.py": sha_file(REFERENCE_EXP / "run_cadr_fallback_wd_sweep.py"),
        "base_wd_sweep_config.json": sha_file(REFERENCE_EXP / "config_fallback_wd_sweep.json"),
        "active_network.py": sha_file(Path(runtime["network"].__file__)),
        "active_layers.py": sha_file(Path(runtime["layers"].__file__)),
        "active_dugsa.py": sha_file(Path(runtime["dugsa"].__file__)),
        "active_utils.py": sha_file(Path(runtime["utils"].__file__)),
        "experiment_51_config.json": sha_file(ROOT / "experiment/experiment51_dugsa/config.json"),
        "experiment_43_anchor_index.json": sha_file(ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors/index.json"),
        "experiment59_run.py": sha_file(EXP / "run.py"),
        "experiment59_shape_pen.py": sha_file(EXP / "shape_pen.py"),
        "experiment59_audit.py": sha_file(EXP / "audit.py"),
        "experiment59_design.md": sha_file(EXP / "design.md"),
        "experiment59_config.json": sha_file(CONFIG_PATH),
        "brainprompt_label_text": sha_file(PROMPT_DIR / "abide_label.txt"),
        "brainprompt_label_embedding": sha_file(PROMPT_DIR / "abide_label.pt"),
        "anonymous_label_embedding": sha_file(PROMPT_DIR / "anonymous_geometry_matched.pt"),
    }


def write_manifest(status: str, extra: dict | None = None) -> None:
    manifest = {
        "experiment": "Experiment 59: SHAPE-PEN",
        "status": status,
        "experiment_root": str(EXP),
        "base_group": REFERENCE_GROUP,
        "seed": 0,
        "required_new_folds": 110,
        "groups": cfg()["groups"],
        "source_sha256": source_hashes(),
        "data_sha256": {
            "processed_standard_data.csv": sha_file(ROOT / "data/ABIDE/processed_standard_data.csv"),
            "modal_feat_dict.npy": sha_file(ROOT / "data/ABIDE/modal_feat_dict.npy"),
        },
        "reference_manifest_sha256": sha_file(REFERENCE_EXP / "manifest_fallback_wd_sweep.json"),
        "reference_baseline_csv_sha256": sha_file(REFERENCE_EXP / "results/fallback_wd_sweep/fold_metrics.csv"),
        "reference_wd": {"cadr_adamw": 0.01, "gfr_adam": 0.11, "cam_adam": 0.11},
    }
    if extra:
        manifest.update(extra)
    atomic_json(EXP / "manifest.json", manifest)


def run_preflight() -> dict:
    torch = BASE.runtime()["torch"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest, _reference_rows, ref_folds = verify_reference_files()
    real, anonymous, prompt_info = prompt_assets()
    shuffles = prepare_shuffles()
    initial_rows = []
    for fold in range(1, 11):
        state = build_base_state(fold, manifest, ref_folds)
        task_path = (
            REFERENCE_EXP / ".runtime/fallback_wd_sweep" / REFERENCE_GROUP / "tasks"
            / f"{REFERENCE_GROUP}_f{fold:02d}.json"
        )
        task = json.loads(task_path.read_text(encoding="utf-8"))
        row = {
            "fold": fold,
            "anchor_sha256": state["anchor_sha"],
            "initial_shared_sha": state["initial_shared_sha"],
            "initial_gfr_sha": state["initial_gfr_sha"],
            "initial_cam_sha": state["initial_cam_sha"],
            "initial_added_sha": state["initial_added_sha"],
            "initial_rng_sha": state["construction_rng_sha"],
            "split_sha": state["split_sha"],
            "pair_sha": state["cache_meta"]["pair_sha"],
            "baseline_checkpoint_sha256": task["best_sha256"],
            "optimizer_weight_decay": {"gfr": [0.11], "cam": [0.11], "cadr": [0.01, 0.01]},
            "status": "PASS",
        }
        if row["initial_shared_sha"] != task["initial_shared_sha"] or row["initial_rng_sha"] != task["initial_rng_sha"]:
            raise AssertionError(f"Fold {fold} initialization differs from 53.3 reference.")
        initial_rows.append(row)
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    math_state = build_base_state(1, manifest, ref_folds)
    math_result = deterministic_math_preflight(real, anonymous, math_state)
    del math_state
    parity = validate_warmup_capture_parity(manifest, ref_folds)
    warmup_audit = prepare_warmup_cache(manifest, ref_folds)
    coefficients = calibrate_coefficients(manifest, ref_folds, real, anonymous, shuffles)
    config = cfg()
    preflight = {
        "status": "PASS",
        "base_reference": {
            "manifest_sha256": sha_file(REFERENCE_EXP / "manifest_fallback_wd_sweep.json"),
            "group": REFERENCE_GROUP,
            "fold_initializations": initial_rows,
            "formal_reference_folds": 10,
            "reference_checkpoint_audits": 10,
            "optimizer_weight_decay": {"cadr_adamw": 0.01, "gfr_adam": 0.11, "cam_adam": 0.11},
        },
        "prompt_audit": prompt_info,
        "semantic_math": math_result,
        "rng_and_warmup_parity": parity,
        "shared_warmup_cache": warmup_audit,
        "coefficient_calibration": coefficients,
        "group_structure": {
            "new_groups": 11,
            "new_folds": 110,
            "group_ids": list(GROUPS),
            "parameter_counts": {
                group_id: (0 if group_id in {"G05", "G08"} else sum(p.numel() for p in isolated_module(group_id, 1, real, anonymous, config)[0].parameters()))
                for group_id in GROUPS
            },
            "G02_G03_same_parameter_count": True,
            "G10_G11_same_parameter_count": True,
            "same_input_representation_for_all_losses": "CAM 144 dimensions",
        },
        "split_shuffles_sha256": {str(fold): shuffles[fold]["source_positions_sha256"] for fold in shuffles},
        "historical_val_equals_test": True,
        "external_generalization_claim": False,
        "e59_source_sha256": source_hashes(),
    }
    if not config["semantic_projection"]["coefficients_frozen"]:
        raise AssertionError("Coefficient calibration did not freeze config.json.")
    preflight["frozen_config_sha256"] = sha_file(CONFIG_PATH)
    preflight["preflight_json_sha256"] = sha_file(RUNTIME / "preflight.json")
    write_manifest("PREFLIGHT_PASS_CONFIG_FROZEN", {
        "preflight_sha256": sha_file(RUNTIME / "preflight.json"),
        "frozen_config_sha256": sha_file(CONFIG_PATH),
        "initial_fold_count": 10,
        "anonymous_geometry_sha256": prompt_info["anonymous_sha256"],
    })
    atomic_json(RUNTIME / "preflight_audit.json", preflight)
    return preflight


def ensure_preflight(force: bool = False) -> dict:
    config = cfg()
    preflight_path = RUNTIME / "preflight_audit.json"
    if config["semantic_projection"]["coefficients_frozen"] and preflight_path.is_file() and not force:
        audit = json.loads(preflight_path.read_text(encoding="utf-8"))
        assert audit["status"] == "PASS"
        assert audit["frozen_config_sha256"] == sha_file(CONFIG_PATH)
        assert audit["e59_source_sha256"] == source_hashes()
        verify_reference_files()
        return audit
    return run_preflight()


def _cpu_tree(value):
    torch = BASE.runtime()["torch"]
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _optimizer_cpu_state(optimizer):
    return _cpu_tree(optimizer.state_dict())


def _semantic_optimizer(module, config: dict):
    if module is None:
        return None
    torch = BASE.runtime()["torch"]
    return torch.optim.AdamW(
        module.parameters(),
        lr=float(config["semantic_projection"]["learning_rate"]),
        weight_decay=float(config["semantic_projection"]["weight_decay"]),
    )


def _checkpoint_object(state, group_id: str, module, semantic_optimizer, best_state: dict, fold: int, best_epoch: int, epochs: int, metrics: dict, diagnostics: dict, bank_type: str, anonymous_sha: str, shuffle: dict):
    checkpoint = BASE._build_checkpoint(state, best_state, fold, best_epoch, epochs, metrics, diagnostics)
    checkpoint.update({
        "group": group_id,
        "e59_group_name": group_item(group_id)["name"],
        "e59_semantic_model": best_state["semantic_model"],
        "e59_optimizer_semantic": best_state["optimizer_semantic"],
        "e59_prototype_bank": bank_type,
        "e59_projection_parameter_sha_initial": state["semantic_initial_sha"],
        "e59_projection_parameter_count": state["semantic_parameter_count"],
        "e59_optimizer_hyperparameters": {
            "class": "AdamW",
            "lr": float(cfg()["semantic_projection"]["learning_rate"]),
            "weight_decay": float(cfg()["semantic_projection"]["weight_decay"]),
        } if semantic_optimizer is not None else None,
        "e59_coefficient_key": _coefficient_key(group_id),
        "e59_coefficient": float(cfg()["semantic_projection"]["coefficients"][_coefficient_key(group_id)]),
        "e59_config_sha256": sha_file(CONFIG_PATH),
        "e59_prompt_sha256": sha_file(PROMPT_DIR / "abide_label.pt"),
        "e59_anonymous_sha256": anonymous_sha,
        "e59_shuffle_sha256": shuffle["source_positions_sha256"] if group_id == "G09" else "none",
        "e59_reference_checkpoint_sha256": state["reference_checkpoint_sha256"],
        "e59_warmup_cache_sha256": state["warmup_cache_sha256"],
        "e59_gate_contract": {
            "training_label_scope": "outer-training labels only",
            "test_labels_used_for_gate": False,
            "test_labels_used_for_training_loss": False,
            "test_labels_used_for_historical_checkpoint_selection": True,
            "test_evidence_description_uses_predicted_class_and_outer-training_scale": True,
        },
        "e59_diagnostics": diagnostics,
        "e59_source_sha256": {
            "run.py": sha_file(Path(__file__)),
            "shape_pen.py": sha_file(EXP / "shape_pen.py"),
            "config.json": sha_file(CONFIG_PATH),
        },
    })
    return checkpoint


def _initial_semantic_checkpoint(module):
    return _cpu_tree(module.state_dict()) if module is not None else None


def _load_group_task(group_id: str, fold: int, state, module, semantic_optimizer, config: dict, task_path: Path, checkpoint_path: Path, latest_path: Path):
    torch = BASE.runtime()["torch"]
    start_epoch = 0
    wait = 0
    best_acc = -1.0
    best_epoch = -1
    best_state = None
    best_metrics = None
    best_diag = None
    warmup_best = -1.0
    warmup_best_epoch = -1
    history: list[dict] = []
    warmup_info = None
    latest = latest_path
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        valid = (
            saved.get("group") == group_id
            and saved.get("fold") == fold
            and saved.get("split_sha") == state["split_sha"]
            and saved.get("source_sha") == sha_file(Path(__file__))
            and saved.get("config_sha") == sha_file(CONFIG_PATH)
        )
        if valid:
            state["model"].load_state_dict(saved["model"], strict=True)
            state["cam"].load_state_dict(saved["cam"], strict=True)
            state["opt_base"].load_state_dict(saved["optimizer_base"])
            state["opt_cam"].load_state_dict(saved["optimizer_cam"])
            state["opt_added"].load_state_dict(saved["optimizer_added"])
            if module is not None:
                module.load_state_dict(saved["semantic_model"], strict=True)
                semantic_optimizer.load_state_dict(saved["optimizer_semantic"])
            start_epoch = int(saved["next_epoch"])
            wait = int(saved["wait"])
            best_acc = float(saved["best_acc"])
            best_epoch = int(saved["best_epoch"])
            best_state = saved["best_state"]
            best_metrics = saved["best_metrics"]
            best_diag = saved["best_diag"]
            warmup_best = float(saved["warmup_best"])
            warmup_best_epoch = int(saved["warmup_best_epoch"])
            history = list(saved.get("gradient_checks", []))
            BASE.restore_rng(saved["rng"])
            warmup_sha = saved.get("warmup_cache_sha256")
            if warmup_sha != sha_file(warmup_cache_path(fold)):
                raise AssertionError(f"Fold {fold} resume checkpoint is not bound to the verified common warmup cache.")
            warmup_info = {"sha256": warmup_sha, "resumed": True}
        else:
            latest.unlink(missing_ok=True)
    if task_path.is_file() and checkpoint_path.is_file():
        task = json.loads(task_path.read_text(encoding="utf-8"))
        if task.get("status") == "COMPLETE" and task.get("checkpoint_sha256") == sha_file(checkpoint_path):
            return None
    if start_epoch == 0:
        warmup_info = load_warmup_cache_into_state(state, fold, config)
        start_epoch = int(config["warmup_epochs"])
        warmup_best = warmup_info["warmup_best_acc"]
        warmup_best_epoch = warmup_info["warmup_best_epoch"]
    return {
        "start_epoch": start_epoch,
        "wait": wait,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "best_metrics": best_metrics,
        "best_diag": best_diag,
        "warmup_best": warmup_best,
        "warmup_best_epoch": warmup_best_epoch,
        "gradient_checks": history,
        "warmup_cache": warmup_info,
    }


def _gradient_check_path(group_id: str, fold: int) -> Path:
    return GRADIENT_DIR / f"{group_id}_f{fold:02d}.json"


def train_worker(group_id: str, fold: int) -> None:
    torch = BASE.runtime()["torch"]
    config = cfg()
    if not config["semantic_projection"]["coefficients_frozen"]:
        raise RuntimeError("Formal workers cannot start before loss coefficients are frozen.")
    reference, _rows, ref_folds = verify_reference_files()
    real, anonymous, prompt_info = prompt_assets()
    shuffle = json.loads((RUNTIME / "shuffle" / f"fold{fold:02d}.json").read_text(encoding="utf-8"))
    task_path, checkpoint_path, latest_path, log_path = group_paths(group_id, fold)
    task_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state = build_base_state(fold, reference, ref_folds)
    state["reference_checkpoint_sha256"] = ref_folds[fold]["checkpoint_sha256"]
    module, bank_type = isolated_module(group_id, fold, real, anonymous, config)
    attach_position_capture(state["cam"])
    semantic_optimizer = _semantic_optimizer(module, config)
    state["semantic_initial_sha"] = parameter_sha(module)
    state["semantic_parameter_count"] = sum(parameter.numel() for parameter in module.parameters()) if module is not None else 0
    state["semantic_initial_state"] = _initial_semantic_checkpoint(module)
    initial_rng_sha = BASE.rng_sha(BASE.pack_rng())
    if initial_rng_sha != ref_folds[fold]["initial_rng_sha"]:
        raise AssertionError("The shared main RNG no longer matches the 53.3 initial RNG after semantic setup.")

    task_state = _load_group_task(group_id, fold, state, module, semantic_optimizer, config, task_path, checkpoint_path, latest_path)
    if task_state is None:
        return
    start_epoch = task_state["start_epoch"]
    wait = task_state["wait"]
    best_acc = task_state["best_acc"]
    best_epoch = task_state["best_epoch"]
    best_state = task_state["best_state"]
    best_metrics = task_state["best_metrics"]
    best_diag = task_state["best_diag"]
    warmup_best = task_state["warmup_best"]
    warmup_best_epoch = task_state["warmup_best_epoch"]
    gradient_checks = task_state["gradient_checks"]
    warmup_cache = task_state.get("warmup_cache")
    state["warmup_cache_sha256"] = warmup_cache["sha256"] if warmup_cache else "resumed_from_latest"
    started = time.time()
    current_epoch = start_epoch

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"START group={group_id} fold={fold} start_epoch={start_epoch} device={BASE.runtime()['device']}\n")
        log.flush()
        for epoch in range(start_epoch, int(config["max_epochs"])):
            current_epoch = epoch
            state["opt_base"].zero_grad(set_to_none=True)
            state["opt_cam"].zero_grad(set_to_none=True)
            state["opt_added"].zero_grad(set_to_none=True)
            if semantic_optimizer is not None:
                semantic_optimizer.zero_grad(set_to_none=True)
            active = epoch >= int(config["warmup_epochs"])
            out = train_forward(group_id, state, module, bank_type, config, shuffle, active)
            if not torch.isfinite(out["total"]):
                raise FloatingPointError(f"Non-finite loss at {group_id} fold={fold} epoch={epoch}.")
            if active and epoch in {int(config["warmup_epochs"]), 100}:
                grad = gradient_probe(group_id, state, out)
                grad["fold"] = fold
                grad["epoch"] = epoch
                gradient_checks.append(grad)
            out["total"].backward()
            for module_check in (state["model"], state["cam"]):
                for parameter in module_check.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"Non-finite base gradient at {group_id} fold={fold} epoch={epoch}.")
            if module is not None:
                for parameter in module.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"Non-finite semantic gradient at {group_id} fold={fold} epoch={epoch}.")
            base_gradient_norms = {
                "gfr": BASE._grad_norm(state["base_params"]),
                "cam": BASE._grad_norm(state["cam"].parameters()),
                "cadr": BASE._grad_norm(parameter for _name, parameter in state["added_named"]),
                "semantic": BASE._grad_norm(module.parameters()) if module is not None else 0.0,
            }
            state["opt_base"].step()
            state["opt_cam"].step()
            state["opt_added"].step()
            if active and semantic_optimizer is not None:
                semantic_optimizer.step()

            with torch.no_grad():
                evaluated = BASE.forward_state(state, training=False, use_cadr=active)
                metrics = BASE.classification_metrics(evaluated["logp"], state["test"], state["y"])
            if not active:
                if metrics["acc"] > warmup_best:
                    warmup_best = float(metrics["acc"])
                    warmup_best_epoch = epoch
            elif metrics["acc"] > best_acc:
                best_acc = float(metrics["acc"])
                best_epoch = epoch
                wait = 0
                best_state = {
                    "model": BASE._state_cpu(state["model"]),
                    "cam": BASE._state_cpu(state["cam"]),
                    "optimizer_base": _optimizer_cpu_state(state["opt_base"]),
                    "optimizer_cam": _optimizer_cpu_state(state["opt_cam"]),
                    "optimizer_added": _optimizer_cpu_state(state["opt_added"]),
                    "semantic_model": _initial_semantic_checkpoint(module),
                    "optimizer_semantic": _optimizer_cpu_state(semantic_optimizer) if semantic_optimizer is not None else None,
                }
                best_metrics = {key: value for key, value in metrics.items() if key not in {"pred", "prob"}}
                best_diag = {
                    "loss_cls": _python_float(out["cls"]),
                    "loss_graph": _python_float(out["graph"]),
                    "loss_pair": _python_float(out["pair"]),
                    "loss_adv": _python_float(out["adv"]),
                    "loss_preserve": _python_float(out["preserve"]),
                    "loss_auxiliary_unscaled": _python_float(out["aux_loss"]) if out["aux_loss"] is not None else 0.0,
                    "coefficient": float(out["coefficient"]),
                    "base_gradient_norms": base_gradient_norms,
                    "auxiliary": out["aux_diagnostics"],
                    "gradient_checks": copy.deepcopy(gradient_checks),
                    "propagation_modes": evaluated["info"].get("propagation_modes", []),
                }
                checkpoint = _checkpoint_object(
                    state,
                    group_id,
                    module,
                    semantic_optimizer,
                    best_state,
                    fold,
                    best_epoch,
                    epoch + 1,
                    best_metrics,
                    best_diag,
                    bank_type,
                    prompt_info["anonymous_sha256"],
                    shuffle,
                )
                atomic_torch_save(checkpoint_path, checkpoint)
            else:
                wait += 1

            if (epoch + 1) % 10 == 0 or epoch == int(config["max_epochs"]) - 1 or (active and wait > int(config["early_stop_wait_gt"])):
                atomic_torch_save(latest_path, {
                    "group": group_id,
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
                    "gradient_checks": gradient_checks,
                    "model": BASE._state_cpu(state["model"]),
                    "cam": BASE._state_cpu(state["cam"]),
                    "semantic_model": _initial_semantic_checkpoint(module),
                    "optimizer_base": _optimizer_cpu_state(state["opt_base"]),
                    "optimizer_cam": _optimizer_cpu_state(state["opt_cam"]),
                    "optimizer_added": _optimizer_cpu_state(state["opt_added"]),
                    "optimizer_semantic": _optimizer_cpu_state(semantic_optimizer) if semantic_optimizer is not None else None,
                    "rng": BASE.pack_rng(),
                    "split_sha": state["split_sha"],
                    "source_sha": sha_file(Path(__file__)),
                    "config_sha": sha_file(CONFIG_PATH),
                    "warmup_cache_sha256": state["warmup_cache_sha256"],
                })
            if module is not None and module.label_embeddings.grad is not None:
                module.label_embeddings.grad = None
            state["cam"]._e59_positions = None
            if (epoch + 1) % 10 == 0:
                row = {
                    "epoch": epoch + 1,
                    "active": active,
                    "best_epoch": best_epoch,
                    "acc": metrics["acc"],
                    "loss_cls": _python_float(out["cls"]),
                    "loss_auxiliary": _python_float(out["aux_loss"]) if out["aux_loss"] is not None else 0.0,
                    "rho_mean": out["aux_diagnostics"].get("rho_mean"),
                }
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
            if active and wait > int(config["early_stop_wait_gt"]):
                break
        if best_state is None:
            raise RuntimeError(f"No active checkpoint selected for {group_id} fold {fold}.")

        state["model"].load_state_dict(best_state["model"], strict=True)
        state["cam"].load_state_dict(best_state["cam"], strict=True)
        if module is not None:
            module.load_state_dict(best_state["semantic_model"], strict=True)
        with torch.no_grad():
            readback = BASE.forward_state(state, training=False, use_cadr=True)
            readback_metrics = BASE.classification_metrics(readback["logp"], state["test"], state["y"])
        for metric in METRICS:
            if abs(float(readback_metrics[metric]) - float(best_metrics[metric])) > 1e-10:
                raise AssertionError(f"Worker checkpoint readback mismatch: {group_id} fold={fold} metric={metric}.")
        checkpoint = _checkpoint_object(
            state,
            group_id,
            module,
            semantic_optimizer,
            best_state,
            fold,
            best_epoch,
            current_epoch + 1,
            readback_metrics,
            best_diag,
            bank_type,
            prompt_info["anonymous_sha256"],
            shuffle,
        )
        atomic_torch_save(checkpoint_path, checkpoint)
        checkpoint_sha = sha_file(checkpoint_path)
        gradient_row = {
            "group": group_id,
            "fold": fold,
            "best_epoch": best_epoch,
            "best_metrics": {key: float(readback_metrics[key]) for key in METRICS},
            "checkpoints": gradient_checks,
            "best_auxiliary_diagnostics": best_diag["auxiliary"],
        }
        atomic_json(_gradient_check_path(group_id, fold), gradient_row)
        task = {
            "group": group_id,
            "group_name": group_item(group_id)["name"],
            "fold": fold,
            "seed": 0,
            "status": "COMPLETE",
            "best_epoch": int(best_epoch),
            "epochs_executed": int(current_epoch + 1),
            "warmup_epochs": int(config["warmup_epochs"]),
            "warmup_cache_sha256": warmup_cache["sha256"] if warmup_cache else "resumed_from_latest",
            "warmup_best_epoch": int(warmup_best_epoch),
            "warmup_best_acc": float(warmup_best),
            "metrics": {key: float(readback_metrics[key]) for key in METRICS},
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "initial_shared_sha": state["initial_shared_sha"],
            "initial_gfr_sha": state["initial_gfr_sha"],
            "initial_cam_sha": state["initial_cam_sha"],
            "initial_added_sha": state["initial_added_sha"],
            "initial_rng_sha": state["construction_rng_sha"],
            "anchor_sha256": state["anchor_sha"],
            "split_sha": state["split_sha"],
            "pair_sha": state["cache_meta"]["pair_sha"],
            "reference_checkpoint_sha256": ref_folds[fold]["checkpoint_sha256"],
            "semantic_parameter_sha_initial": state["semantic_initial_sha"],
            "semantic_parameter_count": state["semantic_parameter_count"],
            "semantic_optimizer": {
                "class": "AdamW",
                "lr": float(config["semantic_projection"]["learning_rate"]),
                "weight_decay": float(config["semantic_projection"]["weight_decay"]),
            } if semantic_optimizer is not None else None,
            "coefficient": float(config["semantic_projection"]["coefficients"][_coefficient_key(group_id)]),
            "rng_checks": {
                "module_init_isolated": True,
                "evidence_no_dropout_or_sampling": True,
                "primary_cam_dropout_calls_per_train_forward": 1,
            },
            "gradient_check_file": str(_gradient_check_path(group_id, fold)),
            "wall_seconds": float(time.time() - started),
            "readback_status": "WORKER_READBACK_PASS",
        }
        atomic_json(task_path, task)
        latest_path.unlink(missing_ok=True)
        log.write(f"COMPLETE group={group_id} fold={fold} best_epoch={best_epoch} sha={checkpoint_sha}\n")
        log.flush()


def _task_complete(group_id: str, fold: int) -> bool:
    task_path, checkpoint_path, _latest, _log = group_paths(group_id, fold)
    if not task_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
        return task.get("status") == "COMPLETE" and task.get("checkpoint_sha256") == sha_file(checkpoint_path)
    except Exception:
        return False


def current_status() -> dict:
    rows = []
    for group in GROUPS:
        for fold in range(1, 11):
            path, _checkpoint, _latest, _log = group_paths(group, fold)
            task = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            rows.append({"group": group, "fold": fold, "status": task.get("status", "PENDING"), "checkpoint_audit": "PENDING"})
    complete = sum(row["status"] == "COMPLETE" for row in rows)
    return {"training_completed_folds": complete, "training_required_folds": 110, "tasks": rows}


def _queue_workers(items: list[tuple[str, int]], max_workers: int, status_callback) -> list[tuple[str, int, int]]:
    pending = list(items)
    active: dict[subprocess.Popen, tuple[str, int, object, int]] = {}
    failures = []
    oom_retries: dict[tuple[str, int], int] = {}
    target_workers = min(2, max_workers)
    successful_since_ramp = 0
    while pending or active:
        while pending and len(active) < target_workers:
            group_id, fold = pending.pop(0)
            task_path, _checkpoint, _latest, log_path = group_paths(group_id, fold)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(task_path, {
                "group": group_id,
                "group_name": group_item(group_id)["name"],
                "fold": fold,
                "seed": 0,
                "status": "RUNNING",
                "attempt": oom_retries.get((group_id, fold), 0) + 1,
                "started_at_unix": time.time(),
            })
            stream = log_path.open("a+", encoding="utf-8")
            stream.seek(0, os.SEEK_END)
            attempt_log_start = stream.tell()
            env = os.environ.copy()
            env.update({
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "NUMEXPR_NUM_THREADS": "2",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            })
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__)), "--worker", "--group", group_id, "--fold", str(fold)],
                cwd=str(ROOT),
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            active[process] = (group_id, fold, stream, attempt_log_start)
        finished = []
        for process, (group_id, fold, stream, attempt_log_start) in list(active.items()):
            return_code = process.poll()
            if return_code is not None:
                stream.close()
                finished.append((process, group_id, fold, return_code, attempt_log_start))
        for process, group_id, fold, return_code, attempt_log_start in finished:
            active.pop(process)
            if return_code == 0 and _task_complete(group_id, fold):
                successful_since_ramp += 1
                status_callback()
                if target_workers < max_workers and successful_since_ramp >= target_workers:
                    target_workers = min(max_workers, 5 if target_workers == 2 else max_workers)
                    successful_since_ramp = 0
            else:
                log_path = group_paths(group_id, fold)[3]
                with log_path.open("rb") as stream:
                    stream.seek(attempt_log_start)
                    log_text = stream.read().decode("utf-8", errors="replace")
                is_oom = "out of memory" in log_text.lower() or "outofmemory" in log_text.lower() or "oom" in log_text.lower()
                if is_oom:
                    key = (group_id, fold)
                    oom_retries[key] = oom_retries.get(key, 0) + 1
                    target_workers = max(1, target_workers // 2)
                    if oom_retries[key] <= 2:
                        pending.insert(0, key)
                    else:
                        failures.append((group_id, fold, return_code))
                        task_path = group_paths(group_id, fold)[0]
                        atomic_json(task_path, {
                            "group": group_id,
                            "group_name": group_item(group_id)["name"],
                            "fold": fold,
                            "status": "FAILED_OOM_RETRIES_EXHAUSTED",
                            "oom_retries": oom_retries[key],
                            "return_code": return_code,
                        })
                else:
                    failures.append((group_id, fold, return_code))
                    task_path = group_paths(group_id, fold)[0]
                    atomic_json(task_path, {
                        "group": group_id,
                        "group_name": group_item(group_id)["name"],
                        "fold": fold,
                        "status": "FAILED",
                        "return_code": return_code,
                    })
                    status_callback()
        if not finished:
            time.sleep(1.0)
    return failures


def run_all(max_workers: int = 10) -> None:
    audit = ensure_preflight()
    if audit["status"] != "PASS":
        raise RuntimeError("Formal queue blocked: preflight has not passed.")
    manifest, _rows, _folds = verify_reference_files()
    pending = [(group, fold) for group in GROUPS for fold in range(1, 11) if not _task_complete(group, fold)]
    start = time.time()

    def save_progress():
        counts = current_status()
        write_manifest("TRAINING" if counts["training_completed_folds"] < 110 else "TRAINING_COMPLETE_AUDIT_PENDING", {
            "preflight_sha256": sha_file(RUNTIME / "preflight.json"),
            "frozen_config_sha256": sha_file(CONFIG_PATH),
            "training_completed_folds": counts["training_completed_folds"],
            "training_required_folds": 110,
            "task_status": counts["tasks"],
            "elapsed_seconds": time.time() - start,
            "max_workers": max_workers,
        })
        print(f"Experiment 59 progress: {counts['training_completed_folds']}/110 folds", flush=True)

    save_progress()
    failures = _queue_workers(pending, min(10, max(1, max_workers)), save_progress)
    if failures:
        write_manifest("TRAINING_INCOMPLETE", {
            "task_status": current_status()["tasks"],
            "failed_tasks": [{"group": group, "fold": fold, "return_code": code} for group, fold, code in failures],
        })
        raise RuntimeError(f"Formal workers failed without a recoverable OOM: {failures}")
    counts = current_status()
    if counts["training_completed_folds"] != 110:
        write_manifest("TRAINING_INCOMPLETE", {"task_status": counts["tasks"]})
        raise RuntimeError(f"Expected 110 complete tasks, found {counts['training_completed_folds']}.")
    write_manifest("TRAINING_COMPLETE_AUDIT_PENDING", {
        "training_completed_folds": 110,
        "training_required_folds": 110,
        "task_status": counts["tasks"],
        "elapsed_seconds": time.time() - start,
        "max_workers": max_workers,
    })
    result = subprocess.run([sys.executable, str(EXP / "audit.py"), "--all"], cwd=str(ROOT), check=False)
    if result.returncode != 0:
        write_manifest("TRAINING_COMPLETE_AUDIT_FAILED", {
            "training_completed_folds": 110,
            "checkpoint_audit_status": "FAIL",
            "task_status": current_status()["tasks"],
        })
        raise RuntimeError("Independent checkpoint audit failed; see results/checkpoint_audit.csv and audit.log.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--force-preflight", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.group not in GROUPS or args.fold is None or not 1 <= args.fold <= 10:
            raise SystemExit("--worker needs a valid --group G01..G11 and --fold 1..10.")
        train_worker(args.group, args.fold)
        return
    if args.preflight or args.prepare:
        result = run_preflight() if args.force_preflight or not cfg()["semantic_projection"]["coefficients_frozen"] else ensure_preflight()
        write_manifest("PREFLIGHT_PASS_CONFIG_FROZEN", {
            "preflight_sha256": sha_file(RUNTIME / "preflight.json"),
            "frozen_config_sha256": sha_file(CONFIG_PATH),
            "preflight_status": result["status"],
        })
        print(json.dumps({"status": "PREFLIGHT_PASS", "config_sha256": sha_file(CONFIG_PATH)}, indent=2))
        if not args.all:
            return
    if args.all:
        run_all(args.workers)
        return
    if not args.preflight and not args.prepare:
        parser.error("Choose --prepare/--preflight, --all, or --worker.")


if __name__ == "__main__":
    main()
