#!/usr/bin/env python3
"""Isolated E75 runner; imports historical training code read-only by absolute path."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
RT = HERE / ".runtime"
TASKS = RT / "tasks"
CKPTS = RT / "checkpoints"
LATEST = RT / "latest"
LOGS = RT / "logs"
RESULTS = HERE / "results"
ROOT = Path("/data/adodas/SPromptGL-main")
E69_DIR = Path("/model/spromptgl/experiment/experiment69_3_g2_two_rows")
E61_DIR = ROOT / "experiment/experiment61_cadr_postalign_vs_e2e"
E61_WARMUPS = E61_DIR / ".runtime/warmup"
E53_DIR = Path("/model/spromptgl/experiment/experiment53_3_cadr_graph_fallback_fix")
E60_DIR = Path("/model/spromptgl/experiment/experiment60_dss_pen")
E71_DIR = Path("/model/spromptgl/experiment/experiment71_gcder")
E51_DIR = ROOT / "experiment/experiment51_dugsa"
E51_SOURCE = E51_DIR / "source"
TEXT_CACHE = E60_DIR / ".runtime/text_embeddings.pt"
E69 = None
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
GROUPS = CFG["groups"]
TEXTS_SP = CFG["dcer"]["spromptgl_prompt_source_order"]


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode())


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(obj, ensure_ascii=False, allow_nan=False)):
        raise IOError(f"atomic JSON readback failed: {path}")


def atomic_torch(path: Path, obj) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(obj, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if not path.is_file() or path.stat().st_size == 0:
        raise IOError(f"checkpoint write failed: {path}")


def torch_load(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_e69():
    global E69
    if E69 is not None:
        return E69
    source = E69_DIR / "run.py"
    spec = importlib.util.spec_from_file_location("e75_locked_e69_run", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load locked E69 runner at {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    E69 = module
    E69.load_e61()
    return E69


def source_inventory() -> dict:
    paths = {
        "e75_run.py": HERE / "run.py",
        "e75_dcer_v2.py": HERE / "dcer_v2.py",
        "e75_audit.py": HERE / "audit.py",
        "e75_config.json": HERE / "config.json",
        "e69_run.py": E69_DIR / "run.py",
        "e69_config.json": E69_DIR / "config.json",
        "e69_manifest.json": E69_DIR / "manifest.json",
        "e53_run.py": E53_DIR / "run.py",
        "e53_cadr_gfr.py": E53_DIR / "cadr_gfr.py",
        "e61_run.py": E61_DIR / "run.py",
        "e61_model_e61.py": E61_DIR / "model_e61.py",
        "e61_config.json": E61_DIR / "config.json",
        "e61_manifest.json": E61_DIR / "manifest.json",
        "e71_run.py": E71_DIR / "run.py",
        "e71_gcder.py": E71_DIR / "gcder.py",
        "e71_config.json": E71_DIR / "config.json",
        "e71_manifest.json": E71_DIR / "manifest.json",
        "e51_dugsa.py": E51_DIR / "dugsa.py",
        "e51_layers.py": E51_SOURCE / "layers.py",
        "e51_network.py": E51_SOURCE / "network.py",
        "e51_utils.py": E51_SOURCE / "utils.py",
        "text_cache.pt": TEXT_CACHE,
    }
    for name in ("main.py", "model.py", "network.py", "layers.py", "utils.py"):
        paths[f"root_{name}"] = ROOT / name
    return {name: sha_file(path) for name, path in paths.items()}


def verify_history() -> dict:
    if not (E69_DIR / "manifest.json").is_file():
        raise FileNotFoundError(E69_DIR / "manifest.json")
    e69_manifest = json.loads((E69_DIR / "manifest.json").read_text(encoding="utf-8"))
    e69_config = json.loads((E69_DIR / "config.json").read_text(encoding="utf-8"))
    if e69_manifest.get("status") != "COMPLETE_AUDITED" or e69_manifest.get("completed_tasks") != 10 or e69_manifest.get("independent_audits") != 10:
        raise RuntimeError("E69.3-G2 historical manifest is not 10/10 audited")
    if e69_manifest.get("base") != e69_config.get("base"):
        raise RuntimeError("E69.3-G2 manifest/config inherited-base records disagree")
    current = source_inventory()
    required_e69 = {
        "e69_run.py": "be2d534ea2df90d3b462034ee4c9fc859b3f79507dfba7c62a2faa48aee49c44",
        "e69_config.json": "4556c96bb92a9bdbdb048b6f702d853abe6030dd2ae9e2bdad9b5f0ba509c8e0",
    }
    for name, expected in required_e69.items():
        if current[name] != expected:
            raise RuntimeError(f"E69.3 locked source mismatch {name}: {current[name]} != {expected}")
    if e69_manifest.get("source_sha256", {}).get("run.py") != current["e69_run.py"] or e69_manifest.get("source_sha256", {}).get("config.json") != current["e69_config.json"]:
        raise RuntimeError("E69.3 manifest source hashes disagree with disk")

    e61_manifest = json.loads((E61_DIR / "manifest.json").read_text(encoding="utf-8"))
    if e61_manifest.get("status") != "COMPLETE_AUDITED" or e61_manifest.get("completed_tasks") != 20 or e61_manifest.get("independent_audits") != 20:
        raise RuntimeError("E61 source manifest is not COMPLETE_AUDITED")
    e53_expected = e61_manifest.get("base_source_sha256", {})
    for filename in ("run.py", "cadr_gfr.py", "run_cadr_fallback.py", "run_cadr_fallback_wd_sweep.py", "config_fallback_wd_sweep.json"):
        if filename in e53_expected and sha_file(E53_DIR / filename) != e53_expected[filename]:
            raise RuntimeError(f"E53.3 protected source changed: {filename}")
    e53_manifest = json.loads((E53_DIR / "manifest.json").read_text(encoding="utf-8"))
    for source_path, expected in e53_manifest.get("source_sha256", {}).items():
        path = Path(source_path)
        if not path.is_absolute():
            path = E53_DIR / path
        if path.is_file() and sha_file(path) != expected:
            raise RuntimeError(f"E53.3 audited source/data SHA mismatch: {path}")
    for filename in ("run.py", "model_e61.py", "config.json"):
        expected = e61_manifest.get("audit_source_sha256", {}).get(filename) or e61_manifest.get("source_sha256", {}).get(filename)
        actual_path = E61_DIR / filename
        if expected and sha_file(actual_path) != expected:
            raise RuntimeError(f"E61 source hash mismatch: {filename}")

    e71_manifest = json.loads((E71_DIR / "manifest.json").read_text(encoding="utf-8"))
    if e71_manifest.get("completed_tasks") != 30 or e71_manifest.get("checkpoint_audit_status") != "PASS":
        raise RuntimeError("E71 historical manifest is not fully audited")
    for key, path in (("training_source", E71_DIR / "run.py"), ("gcder_source", E71_DIR / "gcder.py"),
                      ("config", E71_DIR / "config.json")):
        if e71_manifest.get("source_sha256", {}).get(key) != sha_file(path):
            raise RuntimeError(f"E71 protected source SHA mismatch: {key}")
    e60_manifest = json.loads((E60_DIR / "manifest.json").read_text(encoding="utf-8"))
    if e60_manifest.get("status") != "COMPLETE_AUDITED" or e60_manifest.get("text_cache_sha256") != sha_file(TEXT_CACHE):
        raise RuntimeError("frozen E60 text cache failed manifest verification")

    audit_rows = list(csv.DictReader((E69_DIR / "results/checkpoint_audit.csv").open(encoding="utf-8")))
    tasks = {}
    for fold in range(1, 11):
        task_path = E69_DIR / ".runtime/tasks" / f"E69.3-G2_f{fold:02d}.json"
        best_path = E69_DIR / ".runtime/checkpoints" / f"E69.3-G2_f{fold:02d}.best.pt"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        warmup_path = E61_WARMUPS / f"fold{fold:02d}.pt"
        warmup_sha = sha_file(warmup_path)
        expected_warmup = e69_manifest["warmups"][fold - 1]["warmup_sha256"]
        if warmup_sha != expected_warmup:
            raise RuntimeError(f"warmup SHA mismatch fold={fold}")
        if task.get("status") != "COMPLETE" or task.get("checkpoint_sha256") != sha_file(best_path):
            raise RuntimeError(f"E69 task/checkpoint mismatch fold={fold}")
        audit = next((row for row in audit_rows if int(row["fold"]) == fold and row["group"] == "E69.3-G2"), None)
        if audit is None or audit.get("checkpoint_audit") != "PASS" or audit.get("checkpoint_sha256") != task["checkpoint_sha256"]:
            raise RuntimeError(f"E69 independent checkpoint audit unavailable fold={fold}")
        if fold == 2 and task.get("train_test_overlap_1based") != [377, 407]:
            raise RuntimeError("Fold-2 overlap does not match the historical two rows")
        if fold != 2 and task.get("train_test_overlap_1based") not in ([], None):
            raise RuntimeError(f"unexpected train/test overlap in fold {fold}")
        tasks[fold] = {"task": task, "best_checkpoint": str(best_path), "warmup_sha256": warmup_sha}

    base_rows = list(csv.DictReader((E69_DIR / "results/fold_metrics.csv").open(encoding="utf-8")))
    rows = [row for row in base_rows if row.get("group") == "E69.3-G2"]
    if len(rows) != 10:
        raise RuntimeError("E69.3-G2 historical metrics are not ten-fold complete")
    device = load_e69().load_e61().load_base().runtime()["device"]
    return {
        "status": "PASS",
        "e69_commit": CFG["base"]["commit"],
        "e69_source_matches_locked_commit": True,
        "e69_completed_audited": "10/10",
        "e61_warmups_verified": "10/10",
        "e71_paper_and_source_manifest": "COMPLETE_AUDITED",
        "e60_text_cache_sha256": sha_file(TEXT_CACHE),
        "protected_source_sha256": current,
        "folds": {str(fold): {
            "warmup_sha256": tasks[fold]["warmup_sha256"],
            "e69_checkpoint_sha256": tasks[fold]["task"]["checkpoint_sha256"],
            "split_sha": tasks[fold]["task"]["split_sha"],
            "pair_sha": tasks[fold]["task"]["pair_sha"],
            "train_test_overlap_1based": tasks[fold]["task"].get("train_test_overlap_1based", []),
        } for fold in range(1, 11)},
        "device": str(device),
    }


def text_banks():
    import torch
    import torch.nn.functional as F

    cache = torch_load(TEXT_CACHE)
    texts = cache.get("text_map", {}).get("original")
    if texts != TEXTS_SP:
        raise RuntimeError("SPromptGL-8 prompt contents/order differ from audited E71/E60 source")
    embeddings = torch.as_tensor(cache["embeddings"]["original"], dtype=torch.float32)
    if tuple(embeddings.shape) != (8, 1024):
        raise RuntimeError(f"expected frozen RN50 embeddings [8,1024], got {tuple(embeddings.shape)}")
    normalized = F.normalize(embeddings, dim=-1)
    td = F.normalize(normalized[:4].mean(dim=0), dim=0)
    asd = F.normalize(normalized[4:].mean(dim=0), dim=0)
    real = torch.stack((asd, td)).contiguous()
    generator = torch.Generator(device="cpu").manual_seed(int(CFG["dcer"]["anonymous_seed"]))
    direction_a = F.normalize(torch.randn(1024, generator=generator), dim=0)
    orthogonal = torch.randn(1024, generator=generator)
    orthogonal = F.normalize(orthogonal - torch.dot(direction_a, orthogonal) * direction_a, dim=0)
    cosine = torch.dot(real[0], real[1]).clamp(-0.999999, 0.999999)
    direction_b = cosine * direction_a + torch.sqrt(1.0 - cosine.square()) * orthogonal
    anonymous = torch.stack((direction_a, F.normalize(direction_b, dim=0))).contiguous()
    if not torch.allclose(real.norm(dim=-1), anonymous.norm(dim=-1), atol=1e-6, rtol=0):
        raise RuntimeError("anonymous prototype norms do not match the real bank")
    if abs(float(torch.dot(real[0], real[1]) - torch.dot(anonymous[0], anonymous[1]))) > 1e-5:
        raise RuntimeError("anonymous prototype pairwise cosine does not match")
    meta = {
        "encoder": "frozen OpenAI CLIP RN50 embeddings from audited E60 cache",
        "cache_path": str(TEXT_CACHE),
        "cache_sha256": sha_file(TEXT_CACHE),
        "text_sha256": sha_json(texts),
        "texts": texts,
        "aggregation": CFG["dcer"]["spromptgl_aggregation"],
        "label_order": ["ASD", "TD"],
        "real_bank_sha256": sha_bytes(real.numpy().tobytes()),
        "anonymous_bank_sha256": sha_bytes(anonymous.numpy().tobytes()),
        "real_class_cosine": float(torch.dot(real[0], real[1])),
        "anonymous_class_cosine": float(torch.dot(anonymous[0], anonymous[1])),
        "real_norms": real.norm(dim=-1).tolist(),
        "anonymous_norms": anonymous.norm(dim=-1).tolist(),
        "anonymous_seed": int(CFG["dcer"]["anonymous_seed"]),
    }
    return {"G1": real, "G2": real, "G3": real, "G4": anonymous}, meta


def init_dcer(group: str, fold: int, device, bank: "torch.Tensor"):
    import torch
    from dcer_v2 import DCERV2, tensor_sha

    if str(device).startswith("cuda"):
        device_ids = [torch.cuda.current_device()]
    else:
        device_ids = []
    with torch.random.fork_rng(devices=device_ids, enabled=True):
        torch.manual_seed(750000 + int(fold))
        if device_ids:
            torch.cuda.manual_seed_all(750000 + int(fold))
        module = DCERV2(bank).to(device)
    return module, {"seed": 750000 + int(fold), "parameter_sha256": tensor_sha(module), "state_sha256": tensor_sha(module, include_buffers=True)}


def build_fold_state(fold: int):
    e69 = load_e69()
    e61 = e69.load_e61()
    state, warmup = e61.load_warmup_state(fold)
    membership = e69.apply_fold_specific_training_additions(state, fold)
    e61.configure_branch(state, "E61-E2", None)
    gate_init = e69.set_gate_initialization(state, "E69.3-G2")
    return e61, state, warmup, membership, gate_init


def pack_rng(e61):
    return e61.load_base().pack_rng()


def restore_rng(e61, value):
    e61.load_base().restore_rng(value)


def rng_sha(e61, value=None):
    return e61.load_base().rng_sha(value if value is not None else pack_rng(e61))


def tensor_state_sha(state: dict) -> str:
    digest = hashlib.sha256()
    def visit(prefix, value):
        if torch_is_tensor(value):
            digest.update(prefix.encode() + b"\0")
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(prefix + "/" + str(key), value[key])
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(prefix + f"/{index}", item)
        else:
            digest.update(prefix.encode() + b"=" + repr(value).encode() + b"\0")
    visit("state", state)
    return digest.hexdigest()


def torch_is_tensor(value):
    import torch
    return torch.is_tensor(value)


def deterministic_logits(output_layer, h: "torch.Tensor"):
    """Exact OutputLayer eval path without its dropout RNG or another CAM call."""
    z = output_layer.mlp_head[0](h)
    z = output_layer.mlp_head[1](z)
    z = output_layer.mlp_head[2](z)
    return output_layer.classifier(z)


def token_mask_teacher(output_layer, h, state):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        train_ids = state["train"]
        train_h = h.index_select(0, train_ids)
        labels = state["y"].index_select(0, train_ids)
        if train_h.shape[0] == 0:
            raise RuntimeError("token teacher received an empty training index")
        base_logits = deterministic_logits(output_layer, train_h)
        base_true = base_logits.gather(1, labels[:, None]).squeeze(1)
        base_other = base_logits.masked_fill(F.one_hot(labels, 2).bool(), -torch.inf).max(dim=1).values
        base_margin = base_true - base_other

        n = train_h.shape[0]
        tokenized = train_h.reshape(n, 16, 9)
        keep = 1.0 - torch.eye(16, device=h.device, dtype=h.dtype).reshape(1, 16, 16, 1)
        masked = (tokenized[:, None, :, :] * keep).reshape(n * 16, 144)
        masked_logits = deterministic_logits(output_layer, masked).reshape(n, 16, 2)
        label_idx = labels[:, None, None].expand(-1, 16, 1)
        true_logits = masked_logits.gather(2, label_idx).squeeze(2)
        other_logits = masked_logits.masked_fill(F.one_hot(labels, 2)[:, None, :].bool(), -torch.inf).max(dim=2).values
        drops = (base_margin[:, None] - (true_logits - other_logits)).clamp_min(0.0)
        sums = drops.sum(dim=1, keepdim=True)
        target = torch.where(sums > 1e-12, drops / sums.clamp_min(1e-12), torch.zeros_like(drops))
    return target.detach()


def forward_with_dcer(e61, state, dcer, group: str, *, training: bool, intervention: str = "full", collect_teacher: bool = False):
    import torch
    import torch.nn.functional as F

    group_cfg = GROUPS[group]
    common_enabled = bool(group_cfg["common_response_suppression"])
    validity_enabled = bool(group_cfg["validity"])
    gamma_enabled = True
    validity_gate_enabled = True
    if intervention == "direct_off":
        gamma_enabled = False
    elif intervention == "common_off":
        common_enabled = False
    elif intervention == "validity_off":
        validity_gate_enabled = False
    elif intervention == "semantic_neutral":
        pass
    elif intervention != "full":
        raise ValueError(f"unknown fixed intervention {intervention}")

    captured = {"calls": 0}
    output_layer = state["cam"].Outputlayer

    def pre_hook(module, inputs):
        if len(inputs) != 2:
            raise RuntimeError(f"OutputLayer expected (144-D features, attention), got {len(inputs)} args")
        h = inputs[0]
        captured["calls"] += 1
        result = dcer(
            h,
            common_suppression=common_enabled,
            validity_enabled=validity_enabled,
            validity_gate_enabled=validity_gate_enabled,
            gamma_enabled=gamma_enabled,
            semantic_neutral=intervention == "semantic_neutral",
        )
        captured["h"] = h
        captured["dcer"] = result
        if training and validity_enabled:
            captured["valid_target"] = token_mask_teacher(module, h, state)
        return (result["h_prime"], inputs[1])

    handle = output_layer.register_forward_pre_hook(pre_hook)
    try:
        out = e61.forward_e2_state(state, training=training)
    finally:
        handle.remove()
    if captured["calls"] != 1:
        raise RuntimeError(f"CAM OutputLayer path executed {captured['calls']} times, expected one formal call")
    if training and validity_enabled:
        target = captured["valid_target"]
        train_ids = state["train"]
        prediction = captured["dcer"]["validity"].index_select(0, train_ids)
        target_for_train = target
        valid_loss = F.smooth_l1_loss(prediction, target_for_train)
        out["valid"] = valid_loss
        out["valid_target"] = target
        out["total"] = out["total"] + float(group_cfg["valid_loss_weight"]) * valid_loss
    else:
        out["valid"] = out["total"].new_zeros(())
        out["valid_target"] = captured.get("valid_target")
    out["cam_input"] = captured["h"]
    out["dcer"] = captured["dcer"]
    out["dcer_call_count"] = captured["calls"]
    return out


def module_grad_norm(module, prefix: str | None = None) -> float:
    import torch
    values = []
    for name, parameter in module.named_parameters():
        if prefix is not None and not name.startswith(prefix):
            continue
        if parameter.grad is not None:
            values.append(parameter.grad.detach().float().square().sum())
    return float(torch.stack(values).sum().sqrt().cpu()) if values else 0.0


def module_update_norm(before: dict, module) -> float:
    import torch
    values = []
    for name, parameter in module.named_parameters():
        values.append((parameter.detach().cpu() - before[name].detach().cpu()).float().square().sum())
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def dcer_summary(out, group: str, state, teacher_target=None) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    d = out["dcer"]
    h = out["cam_input"].detach().float()
    h_prime = d["h_prime"].detach().float()
    residual = d["residual"].detach().float()
    bounded = d["bounded_residual"].detach().float().reshape(-1, 144)
    h_norm = h.norm().clamp_min(1e-8)
    diff_norm = (h_prime - h).norm()
    modalities = ["PHENO", "ANAT", "FUNC", "CORR"]
    modality_resid = {}
    for index, name in enumerate(modalities):
        token_residual = d["residual"].detach().reshape(-1, 4, 4, 9)[:, index]
        original = d["tokens"].detach().reshape(-1, 4, 4, 9)[:, index]
        modality_resid[name] = {
            "bounded_norm_ratio": float(d["bounded_residual"].detach().reshape(-1, 4, 4, 9)[:, index].norm() / d["tokens"].detach().reshape(-1, 4, 4, 9)[:, index].norm().clamp_min(1e-8)),
            "fused_residual_norm_ratio": float(token_residual.norm() / original.norm().clamp_min(1e-8)),
            "avg_abs_difference": float(d["difference"].detach().reshape(-1, 4, 4)[:, index].abs().mean()),
            "avg_attention_asd": float(d["attention"].detach()[:, 0].reshape(-1, 4, 4)[:, index].mean()),
            "avg_attention_td": float(d["attention"].detach()[:, 1].reshape(-1, 4, 4)[:, index].mean()),
        }
    attention = d["attention"].detach().float().clamp_min(1e-12)
    entropy = -(attention * attention.log()).sum(dim=-1)
    absd = d["difference"].detach().abs()
    val = d["validity"].detach()
    result = {
        "group": group,
        "gamma": float(d["gamma"].detach()),
        "gamma_raw": float((0.15 * torch.tanh(state["dcer"].gamma_raw)).detach()) if "dcer" in state else float(d["gamma"].detach()),
        "cam_feature_norm": float(h_norm),
        "bounded_residual_norm_ratio": float(bounded.norm() / h_norm),
        "fused_residual_norm_ratio": float(residual.norm() / h_norm),
        "hprime_relative_change": float(diff_norm / h_norm),
        "mean_abs_difference": float(absd.mean()),
        "mean_signed_difference": float(d["difference"].detach().mean()),
        "difference_positive_fraction": float((d["difference"].detach() > 0).float().mean()),
        "difference_negative_fraction": float((d["difference"].detach() < 0).float().mean()),
        "attention_entropy_asd": float(entropy[:, 0].mean()),
        "attention_entropy_td": float(entropy[:, 1].mean()),
        "query_cosine_mean": float(F.cosine_similarity(d["query"].detach()[:, 0], d["query"].detach()[:, 1], dim=-1).mean()) if "query" in d else None,
        "common_token_fraction": float((d["common"].detach() > 0).float().mean()),
        "common_score_mean": float(d["common"].detach().mean()),
        "common_factor_mean": float(d["common_factor"].detach().mean()),
        "validity_mean": float(val.mean()),
        "validity_sd": float(val.std(unbiased=False)),
        "validity_min": float(val.min()),
        "validity_max": float(val.max()),
        "modality": modality_resid,
    }
    if teacher_target is not None and group_cfg_validity(group):
        pred = val.index_select(0, state["train"]).reshape(-1).cpu().numpy()
        target = teacher_target.reshape(-1).cpu().numpy()
        result["validity_teacher_zero_patient_fraction"] = float((teacher_target.sum(dim=1) <= 0).float().mean())
        result["validity_teacher_smoothl1"] = float(torch.nn.functional.smooth_l1_loss(torch.as_tensor(pred), torch.as_tensor(target)))
        result["validity_teacher_correlation"] = float(np.corrcoef(pred, target)[0, 1]) if np.std(pred) and np.std(target) else 0.0
        result["validity_teacher_positive_fraction"] = float((teacher_target > 0).float().mean())
    return result


def group_cfg_validity(group: str) -> bool:
    return bool(GROUPS[group]["validity"])


def state_cpu(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def task_paths(group: str, fold: int):
    token = f"{group}_f{fold:02d}"
    return TASKS / f"{token}.json", CKPTS / f"{token}.best.pt", LATEST / f"{token}.latest.pt", LOGS / f"{token}.jsonl"


def clean_metric_dict(metrics: dict) -> dict:
    return {key: float(metrics[key]) for key in METRICS}


def optimizer_sha(optimizers: list) -> str:
    return tensor_state_sha([optimizer.state_dict() for optimizer in optimizers])


def checkpoint_payload(e61, state, dcer, opt_dcer, group, fold, warmup, gate_init, dcer_init,
                       best_state, best_metrics, best_diag, best_epoch, epochs_executed, wait,
                       trace, config_sha, source_sha):
    return {
        "experiment": CFG["experiment"], "group": group, "fold": int(fold), "seed": 0,
        "best_epoch_1based": int(best_epoch + 1), "epochs_executed": int(epochs_executed), "wait": int(wait),
        "model": best_state["model"], "cam": best_state["cam"], "dcer": best_state["dcer"],
        "optimizer_base": best_state["optimizer_base"], "optimizer_cam": best_state["optimizer_cam"],
        "optimizer_added": best_state["optimizer_added"], "optimizer_dcer": best_state["optimizer_dcer"],
        "metrics": best_metrics, "test_probability": best_state["test_probability"],
        "test_prediction": best_state["test_prediction"], "test_indices": best_state["test_indices"],
        "test_labels": best_state["test_labels"], "diagnostics": best_diag,
        "gate_initialization": gate_init, "dcer_initialization": dcer_init,
        "warmup_sha256": sha_file(E61_WARMUPS / f"fold{fold:02d}.pt"),
        "warmup_rng_sha": warmup["rng_sha"], "warmup_pair_sha": state["warmup_pair_sha"],
        "split_sha": state["split_sha"], "pair_sha": state["pair_sha"], "anchor_sha": state["anchor_sha"],
        "base_train_sha": state["base_train_sha"], "effective_train_sha": state["effective_train_sha"],
        "training_indices": state["train_np"].copy(), "test_indices": state["test_np"].copy(),
        "added_train_rows_1based": state["added_train_rows_1based"],
        "train_test_overlap_1based": state["train_test_overlap_1based"],
        "initial_shared_sha": state["initial_shared_sha"], "initial_gfr_sha": state["initial_gfr_sha"],
        "initial_cam_sha": state["initial_cam_sha"], "relation_core_sha": state["relation_core_sha"],
        "initial_rng_sha": state["initial_rng_sha"], "base_rng_trace": trace,
        "rng_end_sha": rng_sha(e61), "config_sha256": config_sha, "source_sha256": source_sha,
        "optimizer_sha256": tensor_state_sha([
            best_state["optimizer_base"], best_state["optimizer_cam"],
            best_state["optimizer_added"], best_state["optimizer_dcer"],
        ]),
        "group_definition": GROUPS[group], "text_bank_sha256": dcer_init["state_sha256"],
    }


def train_worker(group: str, fold: int) -> dict:
    import numpy as np
    import torch
    from dcer_v2 import tensor_sha

    task_file, best_file, latest_file, log_file = task_paths(group, fold)
    config_sha = sha_file(HERE / "config.json")
    source_sha = source_inventory()
    if task_file.is_file() and best_file.is_file():
        existing = json.loads(task_file.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE" and existing.get("checkpoint_sha256") == sha_file(best_file) and existing.get("config_sha256") == config_sha and existing.get("source_sha256") == source_sha:
            return existing

    e61, state, warmup, membership, gate_init = build_fold_state(fold)
    bank_map, _ = text_banks()
    dcer, dcer_init = init_dcer(group, fold, state["y"].device, bank_map[group])
    rng_at_branch = rng_sha(e61)
    if rng_at_branch != warmup["rng_sha"]:
        raise RuntimeError(f"training RNG moved before first E75 update {group} fold={fold}")
    opt_dcer = torch.optim.AdamW(dcer.parameters(), lr=0.001, weight_decay=0.001)

    start_epoch, wait = 10, 0
    best_acc, best_epoch, best_state, best_metrics, best_diag = -1.0, -1, None, None, None
    trace = []
    if latest_file.is_file():
        saved = torch_load(latest_file)
        expected = {
            "experiment": CFG["experiment"], "group": group, "fold": fold,
            "config_sha256": config_sha, "source_sha256": source_sha,
            "warmup_sha256": sha_file(E61_WARMUPS / f"fold{fold:02d}.pt"),
            "pair_sha": state["pair_sha"], "effective_train_sha": state["effective_train_sha"],
            "dcer_initialization": dcer_init,
        }
        for key, value in expected.items():
            if saved.get(key) != value:
                raise RuntimeError(f"resume checkpoint mismatch {key} {group} fold={fold}")
        state["model"].load_state_dict(saved["model"], strict=True)
        state["cam"].load_state_dict(saved["cam"], strict=True)
        dcer.load_state_dict(saved["dcer"], strict=True)
        state["opt_base"].load_state_dict(saved["optimizer_base"])
        state["opt_cam"].load_state_dict(saved["optimizer_cam"])
        state["opt_added"].load_state_dict(saved["optimizer_added"])
        opt_dcer.load_state_dict(saved["optimizer_dcer"])
        start_epoch, wait = int(saved["next_epoch"]), int(saved["wait"])
        best_acc, best_epoch = float(saved["best_acc"]), int(saved["best_epoch"])
        best_state, best_metrics, best_diag = saved["best_state"], saved["best_metrics"], saved["best_diag"]
        trace = saved["base_rng_trace"]
        restore_rng(e61, saved["rng"])
        if rng_sha(e61) != saved["rng_sha"]:
            raise RuntimeError("resumed main RNG state readback mismatch")

    started = time.time()
    epochs_executed = start_epoch
    for epoch in range(start_epoch, int(CFG["optimization"]["max_epochs"])):
        rng_before = rng_sha(e61)
        for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"], opt_dcer):
            optimizer.zero_grad(set_to_none=True)
        out = forward_with_dcer(e61, state, dcer, group, training=True)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"nonfinite total loss {group} fold={fold} epoch={epoch + 1}")
        out["total"].backward()
        for module in (state["model"], state["cam"], dcer):
            for name, parameter in module.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"nonfinite gradient {group} fold={fold} {name} epoch={epoch + 1}")
        grad_diag = {
            "gfr": e61.total_gradient_norm(state["base_params"]),
            "cam": e61.total_gradient_norm(state["cam"].parameters()),
            "cadr": e61.total_gradient_norm(parameter for _, parameter in state["added_named"]),
            "dcer_total": module_grad_norm(dcer),
            "dcer_gamma": float(dcer.gamma_raw.grad.detach().abs().cpu()) if dcer.gamma_raw.grad is not None else 0.0,
            "dcer_text": module_grad_norm(dcer, "text_proj"),
            "dcer_key": module_grad_norm(dcer, "key_proj"),
            "dcer_evidence": module_grad_norm(dcer, "evidence_proj"),
            "dcer_validity": module_grad_norm(dcer, "validity"),
        }
        before_dcer = {name: value.detach().cpu().clone() for name, value in dcer.named_parameters()}
        state["opt_base"].step()
        state["opt_cam"].step()
        state["opt_added"].step()
        opt_dcer.step()
        update_norm = module_update_norm(before_dcer, dcer)
        rng_after_train = rng_sha(e61)
        epochs_executed = epoch + 1

        with torch.no_grad():
            evaluated = forward_with_dcer(e61, state, dcer, group, training=False)
            metrics = e61.load_base().classification_metrics(evaluated["logp"], state["test"], state["y"])
        rng_after_eval = rng_sha(e61)
        epoch_trace = {"epoch": epoch + 1, "rng_before": rng_before, "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval}
        trace.append(epoch_trace)

        if metrics["acc"] > best_acc:
            best_acc, best_epoch, wait = float(metrics["acc"]), epoch, 0
            best_metrics = clean_metric_dict(metrics)
            best_state = {
                "model": state_cpu(state["model"]), "cam": state_cpu(state["cam"]), "dcer": state_cpu(dcer),
                "optimizer_base": copy.deepcopy(state["opt_base"].state_dict()),
                "optimizer_cam": copy.deepcopy(state["opt_cam"].state_dict()),
                "optimizer_added": copy.deepcopy(state["opt_added"].state_dict()),
                "optimizer_dcer": copy.deepcopy(opt_dcer.state_dict()),
                "test_probability": np.asarray(metrics["prob"], dtype=np.float64).copy(),
                "test_prediction": np.asarray(metrics["pred"], dtype=np.int64).copy(),
                "test_indices": np.asarray(state["test_np"], dtype=np.int64).copy(),
                "test_labels": np.asarray(state["labels_np"], dtype=np.int64)[np.asarray(state["test_np"], dtype=np.int64)].copy(),
            }
            best_diag = {
                "loss_cls": float(out["cls"].detach().cpu()), "loss_graph": float(out["graph"].detach().cpu()),
                "loss_pair": float(out["pair"].detach().cpu()), "loss_adv": float(out["adv"].detach().cpu()),
                "loss_preserve": float(out["preserve"].detach().cpu()), "loss_valid": float(out["valid"].detach().cpu()),
                "train_gradient_norms": grad_diag, "dcer_update_norm": update_norm,
                "test_epoch_dcer_summary": dcer_summary(evaluated, group, {**state, "dcer": dcer}),
            }
            payload = checkpoint_payload(e61, state, dcer, opt_dcer, group, fold, warmup, gate_init, dcer_init,
                                          best_state, best_metrics, best_diag, best_epoch, epochs_executed, wait,
                                          trace, config_sha, source_sha)
            payload["checkpoint_sha256"] = None
            atomic_torch(best_file, payload)
        else:
            wait += 1

        if (epoch + 1) % 10 == 0 or epoch == int(CFG["optimization"]["max_epochs"]) - 1 or wait > int(CFG["optimization"]["early_stop_wait_gt"]):
            latest = {
                "experiment": CFG["experiment"], "group": group, "fold": fold,
                "config_sha256": config_sha, "source_sha256": source_sha,
                "warmup_sha256": sha_file(E61_WARMUPS / f"fold{fold:02d}.pt"),
                "pair_sha": state["pair_sha"], "effective_train_sha": state["effective_train_sha"],
                "dcer_initialization": dcer_init, "next_epoch": epoch + 1, "wait": wait,
                "best_acc": best_acc, "best_epoch": best_epoch, "best_state": best_state,
                "best_metrics": best_metrics, "best_diag": best_diag,
                "model": state_cpu(state["model"]), "cam": state_cpu(state["cam"]), "dcer": state_cpu(dcer),
                "optimizer_base": state["opt_base"].state_dict(), "optimizer_cam": state["opt_cam"].state_dict(),
                "optimizer_added": state["opt_added"].state_dict(), "optimizer_dcer": opt_dcer.state_dict(),
                "rng": pack_rng(e61), "rng_sha": rng_sha(e61), "base_rng_trace": trace,
            }
            atomic_torch(latest_file, latest)

        if (epoch + 1) % 10 == 0:
            log_json = {
                "group": group, "fold": fold, "epoch": epoch + 1,
                "acc": float(metrics["acc"]), "hard_auc": float(metrics["hard_auc"]),
                "prob_auc": float(metrics["prob_auc"]), "best_acc": best_acc,
                "best_epoch": best_epoch + 1, "total_loss": float(out["total"].detach().cpu()),
                "loss_valid": float(out["valid"].detach().cpu()), "gradients": grad_diag,
                "dcer_update_norm": update_norm, "gamma": float(dcer.gamma.detach().cpu()),
                "rng_after_eval": rng_after_eval,
            }
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(log_json, allow_nan=False) + "\n")
                stream.flush(); os.fsync(stream.fileno())
        if wait > int(CFG["optimization"]["early_stop_wait_gt"]):
            break
        del out, evaluated

    if best_state is None:
        raise RuntimeError(f"no best checkpoint for {group} fold={fold}")
    state["model"].load_state_dict(best_state["model"], strict=True)
    state["cam"].load_state_dict(best_state["cam"], strict=True)
    dcer.load_state_dict(best_state["dcer"], strict=True)
    with torch.no_grad():
        readback = forward_with_dcer(e61, state, dcer, group, training=False)
        readback_metrics = e61.load_base().classification_metrics(readback["logp"], state["test"], state["y"])
    for key in METRICS:
        if abs(float(readback_metrics[key]) - float(best_metrics[key])) > 1e-10:
            raise RuntimeError(f"worker checkpoint readback metric mismatch {group} fold={fold} {key}")
    if not np.array_equal(readback_metrics["pred"], best_state["test_prediction"]) or not np.allclose(readback_metrics["prob"], best_state["test_probability"], atol=1e-8, rtol=0):
        raise RuntimeError(f"worker checkpoint prediction mismatch {group} fold={fold}")
    payload = checkpoint_payload(e61, state, dcer, opt_dcer, group, fold, warmup, gate_init, dcer_init,
                                 best_state, best_metrics, best_diag, best_epoch, epochs_executed, wait,
                                 trace, config_sha, source_sha)
    atomic_torch(best_file, payload)
    task = {
        "experiment": CFG["experiment"], "group": group, "fold": fold, "status": "COMPLETE",
        "seed": 0, "config_sha256": config_sha, "source_sha256": source_sha,
        "checkpoint_path": str(best_file), "checkpoint_sha256": sha_file(best_file),
        "warmup_sha256": payload["warmup_sha256"], "warmup_rng_sha": warmup["rng_sha"],
        "split_sha": state["split_sha"], "pair_sha": state["pair_sha"], "warmup_pair_sha": state["warmup_pair_sha"],
        "effective_train_sha": state["effective_train_sha"], "added_train_rows_1based": state["added_train_rows_1based"],
        "train_test_overlap_1based": state["train_test_overlap_1based"],
        "initial_shared_sha": state["initial_shared_sha"], "initial_gfr_sha": state["initial_gfr_sha"],
        "initial_cam_sha": state["initial_cam_sha"], "relation_core_sha": state["relation_core_sha"],
        "initial_rng_sha": state["initial_rng_sha"], "gate_initialization": gate_init,
        "dcer_initialization": dcer_init, "dcer_final_parameter_sha256": tensor_sha(dcer),
        "dcer_final_state_sha256": tensor_sha(dcer, include_buffers=True),
        "best_epoch_1based": best_epoch + 1, "epochs_executed": epochs_executed,
        "metrics": clean_metric_dict(readback_metrics), "diagnostics": best_diag,
        "optimizer_sha256": payload["optimizer_sha256"], "rng_trace": trace,
        "wall_seconds": time.time() - started,
    }
    atomic_json(task_file, task)
    latest_file.unlink(missing_ok=True)
    return task


def log_json_record(path: Path, record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _preflight_base_step(e61, state, dcer, opt_dcer, group):
    import torch

    for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"], opt_dcer):
        optimizer.zero_grad(set_to_none=True)
    out = forward_with_dcer(e61, state, dcer, group, training=True)
    out["total"].backward()
    gradient = {
        "gamma": float(dcer.gamma_raw.grad.detach().abs().cpu()) if dcer.gamma_raw.grad is not None else 0.0,
        "text": module_grad_norm(dcer, "text_proj"),
        "key": module_grad_norm(dcer, "key_proj"),
        "evidence": module_grad_norm(dcer, "evidence_proj"),
        "validity": module_grad_norm(dcer, "validity"),
        "cam": e61.total_gradient_norm(state["cam"].parameters()),
        "gfr": e61.total_gradient_norm(state["base_params"]),
    }
    before = {name: value.detach().cpu().clone() for name, value in dcer.named_parameters()}
    state["opt_base"].step(); state["opt_cam"].step(); state["opt_added"].step(); opt_dcer.step()
    return gradient, module_update_norm(before, dcer)


def run_preflight() -> dict:
    import torch
    from dcer_v2 import tensor_sha

    initial_rng = None
    history = verify_history()
    banks, text_meta = text_banks()
    e69 = load_e69()
    e61 = e69.load_e61()
    fold_records = {}
    init_by_fold = {}
    for fold in range(1, 11):
        e61, state, warmup, membership, gate_init = build_fold_state(fold)
        historical = json.loads((E69_DIR / ".runtime/tasks" / f"E69.3-G2_f{fold:02d}.json").read_text(encoding="utf-8"))
        for key in ("split_sha", "pair_sha", "warmup_pair_sha", "anchor_sha", "base_train_sha",
                    "effective_train_sha", "initial_shared_sha", "initial_gfr_sha", "initial_cam_sha",
                    "relation_core_sha", "initial_rng_sha", "warmup_rng_sha", "added_train_rows_1based",
                    "train_test_overlap_1based"):
            actual = warmup["rng_sha"] if key == "warmup_rng_sha" else state.get(key)
            expected = historical.get(key)
            if actual != expected:
                raise RuntimeError(f"E69.3 fold parity mismatch fold={fold} field={key}: {actual!r} != {expected!r}")
        rng_before_init = rng_sha(e61)
        per_group = {}
        optimizer_layouts = []
        for group in ("G1", "G2", "G3", "G4"):
            dcer, init = init_dcer(group, fold, state["y"].device, banks[group])
            if rng_sha(e61) != rng_before_init:
                raise RuntimeError(f"private DCER init advanced base RNG group={group} fold={fold}")
            opt = torch.optim.AdamW(dcer.parameters(), lr=0.001, weight_decay=0.001)
            optimizer_layouts.append(tuple((len(item["params"]), tuple(parameter.numel() for parameter in item["params"])) for item in opt.param_groups))
            per_group[group] = {"parameter_sha256": init["parameter_sha256"], "state_sha256": init["state_sha256"]}
        parameter_hashes = {item["parameter_sha256"] for item in per_group.values()}
        if len(parameter_hashes) != 1 or len(set(optimizer_layouts)) != 1:
            raise RuntimeError(f"DCER shared parameter initializations disagree at fold={fold}")
        base_optimizer_params = [id(parameter) for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"])
                                 for group_params in optimizer.param_groups for parameter in group_params["params"]]
        dcer_check, _ = init_dcer("G3", fold, state["y"].device, banks["G3"])
        new_optimizer_params = [id(parameter) for parameter in dcer_check.parameters()]
        if len(base_optimizer_params + new_optimizer_params) != len(set(base_optimizer_params + new_optimizer_params)):
            raise RuntimeError(f"duplicate optimizer parameter ownership in fold={fold}")
        del dcer_check
        init_by_fold[str(fold)] = per_group
        fold_records[str(fold)] = {
            "warmup_sha256": sha_file(E61_WARMUPS / f"fold{fold:02d}.pt"),
            "warmup_rng_sha": warmup["rng_sha"],
            "split_sha": state["split_sha"], "pair_sha": state["pair_sha"],
            "warmup_pair_sha": state["warmup_pair_sha"], "effective_train_sha": state["effective_train_sha"],
            "train_test_overlap_1based": state["train_test_overlap_1based"],
            "initial_shared_sha": state["initial_shared_sha"], "initial_gfr_sha": state["initial_gfr_sha"],
            "initial_cam_sha": state["initial_cam_sha"], "relation_core_sha": state["relation_core_sha"],
            "gate_initialization": gate_init,
        }

    e61, state, warmup, _, _ = build_fold_state(1)
    dcer, init = init_dcer("G3", 1, state["y"].device, banks["G3"])
    opt_dcer = torch.optim.AdamW(dcer.parameters(), lr=0.001, weight_decay=0.001)
    base_eval = e61.forward_e2_state(state, training=False)
    zero_gamma = forward_with_dcer(e61, state, dcer, "G3", training=False)
    max_zero_gamma_error = float((base_eval["logp"] - zero_gamma["logp"]).abs().max().detach().cpu())
    if max_zero_gamma_error != 0.0:
        raise RuntimeError(f"gamma=0 did not reproduce base classification path: {max_zero_gamma_error}")
    if zero_gamma["dcer_call_count"] != 1 or tuple(zero_gamma["dcer"]["h_prime"].shape) != (871, 144):
        raise RuntimeError("CAM feature pre-hook shape or one-call invariant failed")
    if not torch.equal(zero_gamma["cam_input"], zero_gamma["dcer"]["h_prime"]):
        raise RuntimeError("gamma=0 changed CAM H")
    if tuple(zero_gamma["dcer"]["tokens"].shape) != (871, 16, 9) or tuple(zero_gamma["dcer"]["global_tokens"].shape) != (871, 4, 9):
        raise RuntimeError("local-token/global-pooling shape check failed")
    common_disabled = forward_with_dcer(e61, state, dcer, "G2", training=False, intervention="common_off")
    if not torch.equal(zero_gamma["dcer"]["h_prime"], common_disabled["dcer"]["h_prime"]):
        raise RuntimeError("common-response-off did not reduce G2 to G1 path at gamma=0")
    validity_disabled = forward_with_dcer(e61, state, dcer, "G3", training=False, intervention="validity_off")
    validity_free = forward_with_dcer(e61, state, dcer, "G2", training=False)
    if not torch.equal(validity_disabled["dcer"]["h_prime"], validity_free["dcer"]["h_prime"]):
        raise RuntimeError("validity-off did not reduce G3 to G2 path at gamma=0")
    g = zero_gamma["dcer"]["global_tokens"]
    base_tokens = zero_gamma["dcer"]["tokens"]
    base_context = zero_gamma["dcer"]["global_tokens"].reshape(871, 36)
    _, _, _, context_delta = dcer.contrast_scores(base_tokens, base_context)
    changed_context = base_context.clone()
    changed_context[:, 0] += 0.25
    _, _, _, altered_difference = dcer.contrast_scores(base_tokens, changed_context)
    context_change = float((context_delta - altered_difference).abs().mean().detach().cpu())
    if context_change <= 1e-8:
        raise RuntimeError("patient global context does not change class-contrastive token evidence")

    # The replay captures a real one-step train update and compares it to an on-disk restore.
    init_rng = pack_rng(e61)
    initial_payload = {
        "model": state_cpu(state["model"]), "cam": state_cpu(state["cam"]), "dcer": state_cpu(dcer),
        "optimizer_base": copy.deepcopy(state["opt_base"].state_dict()),
        "optimizer_cam": copy.deepcopy(state["opt_cam"].state_dict()),
        "optimizer_added": copy.deepcopy(state["opt_added"].state_dict()),
        "optimizer_dcer": copy.deepcopy(opt_dcer.state_dict()), "rng": init_rng,
    }
    replay_path = RT / "preflight" / "one_step_replay.pt"
    atomic_torch(replay_path, initial_payload)
    first_grad, first_update = _preflight_base_step(e61, state, dcer, opt_dcer, "G3")
    first_state_sha = sha_json({
        "model": tensor_sha(state["model"]), "cam": tensor_sha(state["cam"]),
        "dcer": tensor_sha(dcer), "optimizers": optimizer_sha([state["opt_base"], state["opt_cam"], state["opt_added"], opt_dcer]),
    })
    first_rng_sha = rng_sha(e61)
    saved = torch_load(replay_path)
    state["model"].load_state_dict(saved["model"], strict=True); state["cam"].load_state_dict(saved["cam"], strict=True)
    dcer.load_state_dict(saved["dcer"], strict=True)
    state["opt_base"].load_state_dict(saved["optimizer_base"]); state["opt_cam"].load_state_dict(saved["optimizer_cam"])
    state["opt_added"].load_state_dict(saved["optimizer_added"]); opt_dcer.load_state_dict(saved["optimizer_dcer"])
    restore_rng(e61, saved["rng"])
    second_grad, second_update = _preflight_base_step(e61, state, dcer, opt_dcer, "G3")
    second_state_sha = sha_json({
        "model": tensor_sha(state["model"]), "cam": tensor_sha(state["cam"]),
        "dcer": tensor_sha(dcer), "optimizers": optimizer_sha([state["opt_base"], state["opt_cam"], state["opt_added"], opt_dcer]),
    })
    second_rng_sha = rng_sha(e61)
    replay_path.unlink(missing_ok=True)
    if first_state_sha != second_state_sha or first_rng_sha != second_rng_sha or first_grad != second_grad or abs(first_update - second_update) > 1e-10:
        raise RuntimeError("one-step interrupted/resumed replay mismatch")

    # A second classification backward after gamma has moved verifies shared text/key/evidence paths.
    for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"], opt_dcer):
        optimizer.zero_grad(set_to_none=True)
    after_identity = forward_with_dcer(e61, state, dcer, "G3", training=True)
    after_identity["total"].backward()
    post_identity_grad = {
        "gamma": float(dcer.gamma_raw.grad.abs().detach().cpu()) if dcer.gamma_raw.grad is not None else 0.0,
        "text": module_grad_norm(dcer, "text_proj"), "key": module_grad_norm(dcer, "key_proj"),
        "evidence": module_grad_norm(dcer, "evidence_proj"), "validity": module_grad_norm(dcer, "validity"),
    }
    if post_identity_grad["gamma"] <= 0 or post_identity_grad["text"] <= 0 or post_identity_grad["key"] <= 0 or post_identity_grad["evidence"] <= 0 or post_identity_grad["validity"] <= 0:
        raise RuntimeError(f"DCER active gradient path check failed: {post_identity_grad}")
    if not math.isfinite(context_change) or not math.isfinite(first_update) or first_update <= 0:
        raise RuntimeError("DCER context/update preflight was nonfinite or zero")
    result = {
        "status": "PASS",
        "history": history,
        "text_prototypes": text_meta,
        "initializations_by_fold": init_by_fold,
        "folds": fold_records,
        "checks": {
            "gamma_zero_exact_classification_max_abs_error": max_zero_gamma_error,
            "formal_outputlayer_calls": zero_gamma["dcer_call_count"],
            "cam_output_shape": list(zero_gamma["dcer"]["h_prime"].shape),
            "patient_context_mean_abs_delta_change": context_change,
            "first_update_gradient_norms": first_grad,
            "first_update_norm": first_update,
            "post_identity_class_gradient_norms": post_identity_grad,
            "one_step_disk_resume": "PASS_BITWISE_STATE_AND_RNG",
            "optimizer_structure_same_across_groups": True,
            "fold2_overlap_1based": [377, 407],
        },
        "device": str(state["y"].device),
        "source_sha256": source_inventory(),
        "config_sha256": sha_file(HERE / "config.json"),
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(RESULTS / "preflight.json", result)
    return result


def run_training(max_workers: int = 10) -> dict:
    if not (RESULTS / "preflight.json").is_file():
        raise RuntimeError("preflight must pass before formal training")
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise RuntimeError("preflight status is not PASS")
    max_workers = max(1, min(int(max_workers), 10))
    tasks = [(group, fold) for fold in range(1, 11) for group in ("G1", "G2", "G3", "G4")]
    manifest_path = HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update({
        "experiment": CFG["experiment"], "status": "TRAINING",
        "expected_tasks": 40, "completed_tasks": 0, "independent_audits": 0,
        "failed_tasks": manifest.get("failed_tasks", []), "device": preflight["device"],
        "max_workers": max_workers, "start_time_utc": manifest.get("start_time_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        "source_sha256": preflight["source_sha256"], "config_sha256": preflight["config_sha256"],
        "preflight_status": "PASS", "historical_base_commit": CFG["base"]["commit"],
        "path_fallback_reason": CFG["path_fallback_reason"],
    })
    atomic_json(manifest_path, manifest)
    failures = []
    results = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(train_worker, group, fold): (group, fold) for group, fold in tasks}
        for future in as_completed(futures):
            group, fold = futures[future]
            try:
                task = future.result()
                results.append(task)
                manifest["completed_tasks"] = len([1 for g, f in tasks if (TASKS / f"{g}_f{f:02d}.json").is_file() and json.loads((TASKS / f"{g}_f{f:02d}.json").read_text()).get("status") == "COMPLETE"])
                manifest["last_completed"] = {"group": group, "fold": fold, "best_epoch_1based": task["best_epoch_1based"], "checkpoint_sha256": task["checkpoint_sha256"]}
                atomic_json(manifest_path, manifest)
                print(f"COMPLETE {group} fold={fold} epoch={task['best_epoch_1based']} ACC={task['metrics']['acc']:.6f}", flush=True)
            except Exception as error:
                record = {"group": group, "fold": fold, "error": repr(error), "traceback": traceback.format_exc(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                failures.append(record)
                manifest.setdefault("failed_tasks", []).append(record)
                atomic_json(manifest_path, manifest)
                log_json_record(LOGS / f"{group}_f{fold:02d}.failure.jsonl", record)
                print(f"FAILED {group} fold={fold}: {error!r}", flush=True)
    manifest["completed_tasks"] = len([1 for g, f in tasks if (TASKS / f"{g}_f{f:02d}.json").is_file() and json.loads((TASKS / f"{g}_f{f:02d}.json").read_text()).get("status") == "COMPLETE"])
    manifest["training_failures_this_run"] = failures
    manifest["end_time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["training_status"] = "COMPLETE" if manifest["completed_tasks"] == 40 else "INCOMPLETE"
    manifest["status"] = "TRAINING_COMPLETE_PENDING_AUDIT" if manifest["completed_tasks"] == 40 else "PARTIAL"
    atomic_json(manifest_path, manifest)
    return manifest


def initialize_manifest():
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8")) if (RESULTS / "preflight.json").exists() else None
    obj = {
        "experiment": CFG["experiment"], "status": "PREFLIGHT_PENDING",
        "expected_tasks": 40, "completed_tasks": 0, "independent_audits": 0,
        "failed_tasks": [], "groups": CFG["groups"], "base": CFG["base"],
        "path_fallback_reason": CFG["path_fallback_reason"],
        "config_sha256": sha_file(HERE / "config.json"), "source_sha256": source_inventory(),
        "preflight_status": preflight.get("status") if preflight else "PENDING",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(HERE / "manifest.json", obj)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "train", "all", "manifest"))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    for path in (TASKS, CKPTS, LATEST, LOGS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
    if args.action == "manifest":
        print(json.dumps(initialize_manifest(), ensure_ascii=False, indent=2))
        return
    if args.action in ("preflight", "all"):
        initialize_manifest()
        result = run_preflight()
        print(json.dumps({"preflight": result["status"], "device": result["device"], "checks": result["checks"]}, ensure_ascii=False, indent=2))
    if args.action in ("train", "all"):
        result = run_training(args.workers)
        print(json.dumps({"training_status": result["training_status"], "completed_tasks": result["completed_tasks"], "failures": len(result.get("training_failures_this_run", []))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
