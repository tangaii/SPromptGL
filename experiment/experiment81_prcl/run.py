#!/usr/bin/env python3
"""Audited Experiment 81 runner: E74.4 DARG + CAM144 PRCL."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
RESULTS = HERE / "results"
RUNTIME = HERE / "runtime"
CHECKPOINTS = RUNTIME / "checkpoints"
TASKS = RUNTIME / "tasks"
LOGS = RUNTIME / "logs"
BASE_GROUP = "E74.4-G2-Positive-DCR"
GROUPS = tuple(CFG["groups"])
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
BASE = None
PATH_ADAPTER = None
PRCL_IMPL = None


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_value(value) -> str:
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"T" + str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"N" + str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode() + b"\0")
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode() + b"\0")

    visit(value)
    return digest.hexdigest()


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp.write_text(text, encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if path.read_text(encoding="utf-8") != text:
        raise IOError(f"JSON readback failed: {path}")


def atomic_torch(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if not path.is_file() or path.stat().st_size == 0:
        raise IOError(f"Checkpoint readback failed: {path}")


def torch_load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
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


def log_jsonl(path: Path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def runtime():
    global BASE, PATH_ADAPTER, PRCL_IMPL
    if BASE is None:
        adapter_path = HERE / "path_adapter.py"
        spec = importlib.util.spec_from_file_location("e81_readonly_path_adapter", adapter_path)
        PATH_ADAPTER = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = PATH_ADAPTER
        spec.loader.exec_module(PATH_ADAPTER)
        BASE = PATH_ADAPTER.load_base_runner()
        model_spec = importlib.util.spec_from_file_location("experiment81_prcl_impl", HERE / "prcl.py")
        PRCL_IMPL = importlib.util.module_from_spec(model_spec)
        sys.modules[model_spec.name] = PRCL_IMPL
        model_spec.loader.exec_module(PRCL_IMPL)
    return BASE


def base_runtime():
    hist = runtime().load_history()
    return hist.load_e69().load_e61().load_base()


def text_payload():
    path = RUNTIME / "text_embeddings.pt"
    payload = torch_load(path)
    if payload.get("cache_sha256") is None:
        raise RuntimeError("frozen text cache has no cache SHA")
    if payload.get("weights_sha256") != CFG["text"]["weights_sha256"]:
        raise RuntimeError("RN50 cache weights differ from frozen config")
    return payload


def source_map():
    base = runtime()
    files = [HERE / name for name in ("config.json", "design.md", "prcl.py", "run.py", "audit.py", "make_text_cache.py", "path_adapter.py", "runtime/text_embeddings.pt")]
    files += [Path("/data/adodas/SPromptGL-main") / name for name in (
        "main.py", "model.py", "network.py", "layers.py", "utils.py",
        "data/ABIDE/processed_standard_data.csv", "data/ABIDE/modal_feat_dict.npy",
        "CLIP/clip/clip.py", "CLIP/clip/simple_tokenizer.py", "CLIP/clip/bpe_simple_vocab_16e6.txt.gz",
    )]
    files += [Path(CFG["text"]["weights"])]
    files += [Path("/data/adodas/SPromptGL-main/experiment/experiment61_cadr_postalign_vs_e2e") / ".runtime/warmup" / f"fold{fold:02d}.pt" for fold in range(1, 11)]
    e744_root = Path("/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr")
    files += [e744_root / name for name in (
        "run.py", "soft_positive_dcr.py", "audit.py", "config.json", "manifest.json",
        "results/report.md", "results/fold_metrics.csv", "results/audit_status.json",
    )]
    for fold in range(1, 11):
        token = f"{BASE_GROUP}_f{fold:02d}"
        files += [e744_root / ".runtime/tasks" / f"{token}.json",
                  e744_root / ".runtime/checkpoints" / f"{token}.best.pt",
                  e744_root / ".runtime/checkpoints" / f"{token}.latest.pt"]
    hist = base.load_history()
    e741 = hist
    files += [e741.EXP / name for name in ("run.py", "rudag.py", "audit.py", "config.json", "manifest.json", "results/fold_metrics.csv")]
    files += [Path("/data/adodas/SPromptGL-main/experiment/experiment61_cadr_postalign_vs_e2e") / name for name in ("run.py", "model_e61.py", "config.json", "manifest.json")]
    files += [Path("/data/adodas/SPromptGL-main/experiment/experiment53_3_cadr_graph_fallback_fix") / name for name in ("run.py", "cadr_gfr.py", "config.json", "manifest.json")]
    files += [Path("/data/adodas/SPromptGL-main/experiment/experiment72_mgdss/.runtime/vendor/root/experiment/experiment51_dugsa") / name for name in ("dugsa.py", "source/layers.py", "source/network.py", "source/utils.py")]
    files += sorted(Path("/data/adodas/SPromptGL-main/experiment/experiment72_mgdss/.runtime/vendor/root/experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors").glob("*"))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("frozen provenance files missing: " + ", ".join(missing[:8]))
    return {str(path.resolve()): sha_file(path) for path in sorted(set(files))}


def frozen_source_sha():
    expected = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    current = source_map()
    digest = sha_value(current)
    if digest != expected.get("source_sha256") or current != expected.get("source_map"):
        raise RuntimeError("E81 source bytes differ from the frozen preflight manifest")
    return digest


def prepare(fold: int):
    base = runtime()
    state, warmup, identity = base.prepare(fold)
    state["fold"] = fold
    expected_overlap = [377, 407] if fold == 2 else []
    if identity["membership"]["train_test_overlap_1based"] != expected_overlap:
        raise RuntimeError(f"historical fold overlap differs at fold {fold}: {identity['membership']}")
    if tuple(state["train_pairs_np"].shape) != (4096, 2):
        raise RuntimeError(f"expected 4096 E74.4 relation pairs at fold {fold}")
    if not np.isin(state["train_pairs_np"], state["train_np"]).all():
        raise RuntimeError(f"non-training patient in historical relation pairs at fold {fold}")
    return state, warmup, identity


def paths(group: str, fold: int):
    token = f"{group}_f{fold:02d}"
    return (TASKS / f"{token}.json", CHECKPOINTS / f"{token}.best.pt",
            CHECKPOINTS / f"{token}.latest.pt", LOGS / f"{token}.jsonl")


def make_module(fold: int, group: str, device):
    payload = text_payload()
    anchors = payload["medical_anchors"]
    use_text = group != GROUPS[0]
    if group == GROUPS[2]:
        anchors = payload["anonymous_anchors"]
    module = PRCL_IMPL.initialize(anchors, use_text=use_text,
                                  seed=CFG["prcl"]["new_module_init_seed_base"] + fold,
                                  device=device)
    return module


def hook_forward(state, module, group: str, training=False, intervention="full", text_override=None, passthrough=False):
    base = runtime()
    if module is not None:
        module.train(training)
    calls = {"count": 0, "details": None, "z_cam": None}
    layer = state["cam"].Outputlayer

    def replace_input(_layer, args):
        calls["count"] += 1
        if calls["count"] != 1 or len(args) < 2 or tuple(args[0].shape) != (871, 144):
            raise RuntimeError(f"OutputLayer pre-hook expected one CAM144 [871,144], got {[tuple(x.shape) for x in args if torch.is_tensor(x)]}")
        calls["z_cam"] = args[0]
        if passthrough:
            return None
        z_prcl, details = module(args[0], intervention=intervention, text_override=text_override)
        calls["details"] = details
        return (z_prcl, *args[1:])

    handle = layer.register_forward_pre_hook(replace_input)
    try:
        out = base.forward(state, BASE_GROUP, training=training)
    finally:
        handle.remove()
    if calls["count"] != 1:
        raise RuntimeError(f"OutputLayer hook called {calls['count']} times")
    if not passthrough and calls["details"] is None:
        raise RuntimeError("PRCL hook did not return diagnostics")
    return out, calls


def initial_identity(state, warmup, identity, module, optimizer, group):
    rt = base_runtime()
    if rt.rng_sha(rt.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {state['fold']} did not restore E61 epoch-10 public RNG")
    return {
        "base_identity": identity,
        "warmup_sha256": sha_file(runtime().load_history().load_e69().load_e61().warmup_path(state["fold"])),
        "warmup_rng_sha": warmup["rng_sha"],
        "initial_rng_sha": rt.rng_sha(rt.pack_rng()),
        "initial_base_model_cam_sha": runtime().load_history().state_model_sha(state),
        "original_optimizer_sha": sha_value({key: state[key].state_dict() for key in ("opt_base", "opt_cam", "opt_added")}),
        "module_init_sha256": sha_value({key: value.detach().cpu() for key, value in module.named_parameters()}),
        "module_state_sha256": sha_value({key: value.detach().cpu() for key, value in module.state_dict().items()}),
        "new_optimizer_sha256": sha_value(optimizer.state_dict()),
        "group": group,
        "train_pair_count": int(state["train_pairs_np"].shape[0]),
        "fold2_overlap_1based": identity["membership"]["train_test_overlap_1based"],
    }


def state_bundle(state, module, optimizer):
    hist = runtime().load_history()
    return {
        "base_state": hist.state_bundle(state),
        "prcl": {key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
        "optimizer_prcl": copy.deepcopy(optimizer.state_dict()),
    }


def load_bundle(state, module, optimizer, bundle):
    runtime().load_history().load_bundle(state, bundle["base_state"])
    module.load_state_dict(bundle["prcl"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(bundle["optimizer_prcl"]))


def gradient_l2(parameters):
    pieces = [parameter.grad.detach().double().square().sum()
              for parameter in parameters if parameter.grad is not None]
    return float(torch.sqrt(torch.stack(pieces).sum()).cpu()) if pieces else 0.0


def compare_tree(left, right, where="root"):
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            raise RuntimeError(f"tensor mismatch at {where}")
    elif isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
            raise RuntimeError(f"array mismatch at {where}")
    elif isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise RuntimeError(f"key mismatch at {where}")
        for key in left:
            compare_tree(left[key], right[key], f"{where}.{key}")
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise RuntimeError(f"length mismatch at {where}")
        for index, (a, b) in enumerate(zip(left, right)):
            compare_tree(a, b, f"{where}[{index}]")
    elif left != right:
        raise RuntimeError(f"value mismatch at {where}: {left!r} != {right!r}")


def base_off_update(fold: int, use_hook: bool):
    state, _, _ = prepare(fold)
    base = runtime()
    hist = base.load_history()
    rt = base_runtime()
    hist.zero_optimizers(state)
    if use_hook:
        out, _ = hook_forward(state, None, GROUPS[0], training=True, passthrough=True)
    else:
        out = base.forward(state, BASE_GROUP, training=True)
    result = {
        "logp": out["logp"].detach().cpu().clone(),
        "losses": {key: float(out[key].detach().cpu()) for key in ("cls", "graph", "pair", "adv", "preserve", "total")},
    }
    out["total"].backward()
    result["gradients"] = {
        key: {name: None if param.grad is None else param.grad.detach().cpu().clone()
              for name, param in state[key].named_parameters()}
        for key in ("model", "cam")
    }
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    result["bundle"] = hist.state_bundle(state)
    result["rng_sha"] = rt.rng_sha(rt.pack_rng())
    return result


def tensor_preflight():
    runtime()
    payload = text_payload()
    med, anon = payload["medical_anchors"], payload["anonymous_anchors"]
    expected_prompts = [
        "Nromal brain volume", "Uniform cortical thickness", "Symmetrical brain structure",
        "Normal ventricular morphology", "Altered brain volume", "Abnormal functional connectivity",
        "White matter integrity issues", "Different activation patterns",
    ]
    if payload["medical_texts"] != expected_prompts:
        raise RuntimeError("frozen medical Prompt text/order differs from root network.py specification")
    expected_anonymous = CFG["text"]["anonymous_texts"]
    if payload["anonymous_texts"] != expected_anonymous:
        raise RuntimeError("frozen anonymous Prompt text/order differs from config")
    banned = ("brain", "asd", "autism", "connectivity", "volume", "cortical", "white matter")
    if any(any(term in text.lower() for term in banned) for text in expected_anonymous):
        raise RuntimeError("anonymous Prompt contains prohibited medical language")
    if tuple(med.shape) != (8, 32) or tuple(anon.shape) != (8, 32):
        raise RuntimeError("frozen text anchors must be [8,32]")
    if torch.equal(med, anon):
        raise RuntimeError("medical and anonymous RN50 anchors unexpectedly match")
    if payload["gram_max_abs_error"] > 2e-10:
        raise RuntimeError("common text projection did not preserve the joint Gram matrix")
    n = 3
    torch.manual_seed(991)
    z = torch.randn(n, 144)
    module = PRCL_IMPL.initialize(med, True, 810991, torch.device("cpu"))
    z_out, details = module(z)
    if tuple(details["tokens"].shape) != (n, 16, 32):
        raise RuntimeError("local token shape mismatch")
    if tuple(details["key"].shape) != (n, 16, 15, 32) or tuple(details["value"].shape) != (n, 16, 15, 32):
        raise RuntimeError("directed relation tensor shape mismatch")
    if module.undirected_pairs.shape[1] != 120 or module.edge_src.numel() != 240:
        raise RuntimeError("local relation count must be 120 undirected/240 directed")
    if bool((module.edge_src == module.edge_dst).any()) or module.edge_src.numel() != torch.unique(torch.stack((module.edge_src, module.edge_dst), 1), dim=0).shape[0]:
        raise RuntimeError("directed local adjacency contains a self edge or duplicate")
    if tuple(details["attention"].shape) != (n, 8, 16, 16) or tuple(z_out.shape) != (n, 144):
        raise RuntimeError("query head, Null softmax, or restored CAM144 shape mismatch")
    if not torch.allclose(details["attention"].sum(-1), torch.ones(n, 8, 16), atol=1e-6):
        raise RuntimeError("Null-inclusive local softmax does not sum to one")
    if not bool((details["delta_ratio"] <= 0.100001).all()):
        raise RuntimeError("per-patient relative residual cap failed")
    # A patient output must be independent of other patient rows in the batch.
    changed = z.clone()
    changed[1] += 10.0
    with torch.no_grad():
        a = module(z)[0][0]
        b = module(changed)[0][0]
    if not torch.equal(a, b):
        raise RuntimeError("patient relation operation mixed rows across patients")
    return {
        "status": "PASS", "input": [n, 144], "tokens": [n, 16, 9],
        "encoded_tokens": [n, 16, 32], "undirected_candidate_pairs": 120,
        "directed_message_positions": 240, "self_edges": 0,
        "attention": [n, 8, 16, 16], "output": [n, 144],
        "text_cache_sha256": sha_file(RUNTIME / "text_embeddings.pt"),
        "text_effective_rank": int(payload["effective_rank"]),
        "joint_gram_max_abs_error": float(payload["gram_max_abs_error"]),
        "medical_prompt_text_order_exact": True,
        "anonymous_prompt_text_order_exact": True,
        "medical_anonymous_gram_max_abs_difference": float((payload["gram_final"][:8, :8] - payload["gram_final"][8:, 8:]).abs().max()),
        "text_cache_internal_sha256": payload["cache_sha256"],
        "cross_patient_mixing": False,
    }


def local_relation_diagnostic(fold=1):
    """Frozen E74.4-best features; every diagnostic label belongs to outer train."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler

    state, _, _ = prepare(fold)
    state["fold"] = fold
    hist = runtime().load_history()
    base_best = Path("/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr/.runtime/checkpoints") / f"{BASE_GROUP}_f{fold:02d}.best.pt"
    historical = torch_load(base_best)
    hist.load_bundle(state, historical["best_state"])
    rt = base_runtime()
    rng_saved = rt.pack_rng()
    state["model"].eval()
    state["cam"].eval()
    try:
        with torch.no_grad():
            _, capture = hook_forward(state, None, GROUPS[0], training=False, passthrough=True)
            tokens = capture["z_cam"].reshape(871, 16, 9).detach().cpu().numpy().astype(np.float64)
    finally:
        rt.restore_rng(rng_saved)

    train_ids = np.setdiff1d(state["train_np"], state["test_np"]).astype(np.int64)
    y_all = state["y"].detach().cpu().numpy()
    labels = y_all[train_ids].astype(np.int64)
    if len(np.unique(labels)) != 2:
        raise RuntimeError("outer-training-only relation diagnostic requires both classes")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=8181)
    inner_train, inner_test = next(split.split(np.zeros((len(labels), 1)), labels))

    pair_p, pair_q = np.triu_indices(16, k=1)
    pair_product = tokens[:, pair_p] * tokens[:, pair_q]
    pair_difference = np.abs(tokens[:, pair_p] - tokens[:, pair_q])
    normalized = tokens / np.maximum(np.linalg.norm(tokens, axis=-1, keepdims=True), 1e-12)
    cosine = np.einsum("nip,njp->nij", normalized, normalized)
    rng = np.random.default_rng(8181)

    def features_for_partition(group_ids):
        local = np.stack([tokens[:, group_ids == group].mean(axis=1) for group in range(4)], axis=1).reshape(len(tokens), 36)
        same = group_ids[pair_p] == group_ids[pair_q]
        relation_parts = []
        for mask in (same, ~same):
            if not bool(mask.any()):
                raise RuntimeError("coordinate partition created an empty relation class")
            relation_parts.extend((pair_product[:, mask].mean(axis=1), pair_difference[:, mask].mean(axis=1)))
        return local, np.concatenate(relation_parts, axis=-1)

    true_groups = np.repeat(np.arange(4), 4)
    local_real, relation_real = features_for_partition(true_groups)
    random_groups = np.repeat(np.arange(4), 4)
    rng.shuffle(random_groups)
    local_random, relation_random = features_for_partition(random_groups)
    feature_sets = {
        "local_mean_9d": tokens.mean(axis=1),
        "slot_pooled_local_36d": local_real,
        "slot_pair_relation_36d": relation_real,
        "random_group_local_36d": local_random,
        "random_group_pair_relation_36d": relation_random,
        "original_cam144_reference": tokens.reshape(len(tokens), 144),
    }
    diagnostic = {}
    for name, features in feature_sets.items():
        x_train = features[train_ids][inner_train]
        x_test = features[train_ids][inner_test]
        scaler = StandardScaler().fit(x_train)
        classifier = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=0)
        classifier.fit(scaler.transform(x_train), labels[inner_train])
        prediction = classifier.predict(scaler.transform(x_test))
        score = classifier.decision_function(scaler.transform(x_test))
        diagnostic[name] = {
            "feature_dim": int(features.shape[1]),
            "accuracy": float(accuracy_score(labels[inner_test], prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(labels[inner_test], prediction)),
            "roc_auc_positive_label_1": float(roc_auc_score(labels[inner_test], score)),
        }

    real_within = np.repeat(np.arange(4), 4)
    pair_cos = cosine[:, pair_p, pair_q]
    within_cos = pair_cos[:, real_within[pair_p] == real_within[pair_q]].mean()
    cross_cos = pair_cos[:, real_within[pair_p] != real_within[pair_q]].mean()
    random_gaps = []
    for _ in range(100):
        partition = np.repeat(np.arange(4), 4)
        rng.shuffle(partition)
        within_mask = partition[pair_p] == partition[pair_q]
        random_gaps.append(float(pair_cos[:, within_mask].mean() - pair_cos[:, ~within_mask].mean()))
    return {
        "status": "PASS", "fold": fold, "historical_checkpoint_sha256": sha_file(base_best),
        "outer_train_patients": int(len(train_ids)), "inner_train_patients": int(len(inner_train)),
        "inner_validation_patients": int(len(inner_test)), "uses_outer_test_labels": False,
        "random_group_seed": 8181, "random_partition_repeats_for_structure": 100,
        "real_slot_within_cosine_mean": float(within_cos), "real_slot_cross_cosine_mean": float(cross_cos),
        "real_slot_within_minus_cross_cosine": float(within_cos - cross_cos),
        "random_slot_within_minus_cross_cosine_p50": float(np.median(random_gaps)),
        "random_slot_within_minus_cross_cosine_p95": float(np.quantile(random_gaps, 0.95)),
        "dimension_matched_features": ["slot_pooled_local_36d", "slot_pair_relation_36d", "random_group_local_36d", "random_group_pair_relation_36d"],
        "classifier": "StandardScaler + LogisticRegression(C=1.0, solver=liblinear)",
        "results": diagnostic,
        "interpretation_limit": "A single deterministic inner split is a capacity-controlled feasibility probe, not a model-selection result; frozen features inherit the historical transductive graph.",
    }


def grad_preflight(fold: int):
    state, warmup, identity = prepare(fold)
    rt = base_runtime()
    payload = text_payload()
    records = {}
    initialized = {}
    for group in GROUPS:
        module = make_module(fold, group, state["parts"][0].device)
        initialized[group] = module
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01)
        rt.restore_rng(warmup["rng"])
        before_rng = rt.rng_sha(rt.pack_rng())
        out, captured = hook_forward(state, module, group, training=True)
        after_rng = rt.rng_sha(rt.pack_rng())
        if before_rng != warmup["rng_sha"]:
            raise RuntimeError("PRCL init changed the shared RNG before classification forward")
        reference_task = Path("/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr/.runtime/tasks") / f"{BASE_GROUP}_f{fold:02d}.json"
        reference_trace = next(item for item in json.loads(reference_task.read_text(encoding="utf-8"))["rng_trace"] if int(item["epoch"]) == 11)
        if after_rng != reference_trace["rng_after_train"]:
            raise RuntimeError(f"PRCL changed the E74.4 shared training RNG trajectory in fold {fold}/{group}")
        module.eval()
        with torch.no_grad():
            _, eval_capture = hook_forward(state, module, group, training=False)
        after_eval_rng = rt.rng_sha(rt.pack_rng())
        if after_eval_rng != reference_trace["rng_after_eval"]:
            raise RuntimeError(f"PRCL changed the E74.4 shared evaluation RNG trajectory in fold {fold}/{group}")
        grad_params = {
            "local_proj": tuple(module.local_proj.parameters()),
            "key_proj": tuple(module.key_proj.parameters()) + tuple(module.key_norm.parameters()),
            "value_proj": tuple(module.value_proj.parameters()),
            "visual_query": tuple(module.query_proj.parameters()) + tuple(module.query_norm.parameters()),
            "gamma": (module.gamma_raw,),
            "fusion": tuple(module.fusion.parameters()),
            "residual_writeback": tuple(module.output.parameters()) + tuple(module.alpha_gate.parameters()),
        }
        norms = {}
        for label, params in grad_params.items():
            grads = torch.autograd.grad(out["cls"], params, retain_graph=True, allow_unused=True)
            vals = [grad.detach().double().square().sum() for grad in grads if grad is not None]
            norms[label] = float(torch.sqrt(torch.stack(vals).sum()).cpu()) if vals else 0.0
        darg_params = tuple(parameter for parameter in state["model"].parameters() if parameter.requires_grad)
        cam_params = tuple(parameter for parameter in state["cam"].parameters() if parameter.requires_grad)
        darg_grads = torch.autograd.grad(out["cls"], darg_params, retain_graph=True, allow_unused=True)
        cam_grads = torch.autograd.grad(out["cls"], cam_params, retain_graph=True, allow_unused=True)
        norm_group = lambda grads: float(torch.sqrt(torch.stack([g.detach().double().square().sum() for g in grads if g is not None]).sum()).cpu()) if any(g is not None for g in grads) else 0.0
        norms["darg_gfr"] = norm_group(darg_grads)
        norms["cam"] = norm_group(cam_grads)
        output_params = tuple(parameter for parameter in state["cam"].Outputlayer.parameters() if parameter.requires_grad)
        norms["output_layer"] = norm_group(torch.autograd.grad(out["cls"], output_params, retain_graph=True, allow_unused=True))
        for name, value in norms.items():
            if name in ("gamma",) and group == GROUPS[0]:
                continue
            if value <= 0 or not math.isfinite(value):
                raise RuntimeError(f"classification gradient missing/nonfinite for {group}/{name}: {value}")
        if module.text_anchors.grad is not None or module.head_identity.grad is not None:
            raise RuntimeError("fixed text/head buffers unexpectedly received gradients")
        records[group] = {
            "gradient_norms": norms,
            "logp_cls_loss_is_original_weighted_nll": True,
            "text_buffers_frozen": True,
            "rng_before_forward": before_rng,
            "rng_after_train_forward": after_rng,
            "rng_after_eval_forward": after_eval_rng,
            "gamma_initial": float(module.gamma.detach().cpu()),
            "text_anchor_sha": sha_value(module.text_anchors),
        }
        del out, captured, eval_capture, optimizer, darg_grads, cam_grads
        hist = runtime().load_history()
        hist.zero_optimizers(state)

    for left, right in ((GROUPS[1], GROUPS[2]), (GROUPS[0], GROUPS[1])):
        a = dict(initialized[left].named_parameters())
        b = dict(initialized[right].named_parameters())
        if a.keys() != b.keys() or any(not torch.equal(a[name], b[name]) for name in a):
            raise RuntimeError(f"group parameter initializations differ: {left} vs {right}")
    probe = torch.randn(11, 144, device=state["parts"][0].device)
    medical_module = initialized[GROUPS[1]]
    anonymous = text_payload()["anonymous_anchors"].to(probe.device)
    with torch.no_grad():
        _, med_detail = medical_module(probe)
        _, anon_detail = medical_module(probe, text_override=anonymous)
    if torch.equal(med_detail["query"], anon_detail["query"]) or torch.equal(med_detail["attention"], anon_detail["attention"]):
        raise RuntimeError("frozen Prompt replacement did not alter PRCL Query and relation weights")
    if torch.equal(payload["medical_anchors"], payload["anonymous_anchors"]):
        raise RuntimeError("G2/G3 text banks are identical")
    return {"fold": fold, "status": "PASS", "groups": records,
            "g2_g3_trainable_init_equal": True,
            "g1_visual_init_equal": True,
            "medical_to_anonymous_replacement_changes_query": True,
            "medical_to_anonymous_replacement_changes_attention": True,
            "medical_anonymous_anchor_values_differ": True,
            "public_rng_unchanged_by_private_module_initialization": True,
            "fold2_overlap_1based": identity["membership"]["train_test_overlap_1based"]}


def preflight():
    if not torch.cuda.is_available():
        raise RuntimeError("E81 formal runs require the configured CUDA device")
    if not (RUNTIME / "text_embeddings.pt").is_file():
        raise FileNotFoundError("run make_text_cache.py before preflight")
    start = time.time()
    tensor_record = tensor_preflight()
    parity = []
    for fold in range(1, 11):
        reference = base_off_update(fold, use_hook=False)
        candidate = base_off_update(fold, use_hook=True)
        compare_tree(reference["logp"], candidate["logp"], f"fold{fold}.logp")
        compare_tree(reference["losses"], candidate["losses"], f"fold{fold}.losses")
        compare_tree(reference["gradients"], candidate["gradients"], f"fold{fold}.gradients")
        compare_tree(reference["bundle"], candidate["bundle"], f"fold{fold}.bundle")
        if reference["rng_sha"] != candidate["rng_sha"]:
            raise RuntimeError(f"base-off public RNG mismatch in fold {fold}")
        parity.append({"fold": fold, "status": "PASS", "logp_exact": True, "all_losses_exact": True,
                       "original_gradients_exact": True, "all_three_optimizer_updates_exact": True, "rng_exact": True})
        del reference, candidate
        torch.cuda.empty_cache()
    gradients = grad_preflight(1)
    relation_diagnostic = local_relation_diagnostic(1)
    provenance = source_map()
    source_digest = sha_value(provenance)
    preflight_result = {
        "status": "PASS", "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - start,
        "device": torch.cuda.get_device_name(0),
        "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "tensor_preflight": tensor_record,
        "base_off_parity": parity,
        "base_off_parity_pass_count": len(parity),
        "classification_gradient_audit": gradients,
        "frozen_local_relation_diagnostic": relation_diagnostic,
        "source_sha256": source_digest,
        "source_map": provenance,
        "limitations": [
            "Historical validation equals test and test ACC selects the checkpoint.",
            "CV preprocessing/RFE and the transductive 871-patient graph are inherited.",
            "Fold 2 historical training/test overlap rows 377 and 407 are retained in original DARG training; PRCL adds no test-label input.",
            "CAM latent coordinates are not anatomical ROIs or independently identified concepts.",
        ],
    }
    atomic_json(RESULTS / "preflight.json", preflight_result)
    previous_manifest_path = HERE / "manifest.json"
    previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8")) if previous_manifest_path.is_file() else {}
    training_attempts = list(previous_manifest.get("training_attempts", []))
    if previous_manifest.get("formal_training_started"):
        previous_failures = previous_manifest.get("failed_jobs", [])
        failure_tail = previous_failures[0].get("tail", "") if previous_failures else ""
        training_attempts.append({
            "source_sha256": previous_manifest.get("source_sha256"),
            "worker_limit": previous_manifest.get("worker_limit"),
            "elapsed_seconds": float(previous_manifest.get("training_wall_seconds", 0.0)),
            "completed_tasks": int(previous_manifest.get("completed_tasks", 0)),
            "failed_task_count": len(previous_failures),
            "failure_detail": failure_tail[-1200:],
            "worker_logs": [str(LOGS / f"dispatch_{item.get('group')}_f{int(item.get('fold', 0)):02d}.log")
                            for item in previous_failures],
            "checkpoints_written": len(list(CHECKPOINTS.glob("*.best.pt"))),
            "optimizer_updates_completed": int(previous_manifest.get("completed_optimizer_updates", 0)),
        })
    manifest = {
        "experiment": CFG["experiment"], "status": "PREFLIGHT_PASS",
        "experiment_root": str(HERE), "preferred_root": CFG["preferred_root"],
        "root_fallback_reason": CFG["root_fallback_reason"], "expected_tasks": 30,
        "completed_tasks": 0, "independent_audits": 0, "source_sha256": source_digest,
        "source_map": provenance, "preflight": {k: v for k, v in preflight_result.items() if k not in ("source_map",)},
        "worker_limit": 10, "formal_training_started": False,
        "training_attempts": training_attempts,
    }
    atomic_json(HERE / "manifest.json", manifest)
    print(f"PREFLIGHT PASS base_off=10/10 gpu={preflight_result['device']} source={source_digest}", flush=True)
    return preflight_result


def checkpoint_payload(group, fold, source_hash, identity, best, executed, trace, init_record, elapsed, peak):
    return {
        "experiment": CFG["experiment"], "group": group, "fold": fold, "seed": 0,
        "source_sha256": source_hash, "identity": identity, "initialization": init_record,
        "best_epoch_1based": int(best["epoch"]), "epochs_executed": int(executed),
        "metrics": best["metrics"], "test_probability": best["probability"],
        "test_prediction": best["prediction"], "test_indices": best["test_indices"],
        "test_logp": best["test_logp"], "best_losses": best["losses"],
        "rng_trace": trace, "best_state": best["state"],
        "best_rng": best["rng"], "elapsed_seconds": elapsed, "peak_gpu_bytes": int(peak),
    }


def train_worker(group: str, fold: int):
    if group not in GROUPS or fold not in range(1, 11):
        raise ValueError(f"invalid E81 task: {group}/{fold}")
    source_hash = frozen_source_sha()
    state, warmup, identity = prepare(fold)
    state["group"] = group
    rt = base_runtime()
    module = make_module(fold, group, state["parts"][0].device)
    optimizer_prcl = torch.optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01)
    init_record = initial_identity(state, warmup, identity, module, optimizer_prcl, group)
    task_file, best_file, latest_file, log_file = paths(group, fold)
    if task_file.is_file():
        task = json.loads(task_file.read_text(encoding="utf-8"))
        if task.get("status") == "COMPLETE" and task.get("source_sha256") == source_hash and task.get("checkpoint_sha256") == sha_file(best_file):
            print(f"SKIP complete {group} fold={fold}", flush=True)
            return task
        raise RuntimeError(f"refusing to overwrite incompatible task record {task_file}")
    start_epoch, wait = 10, 0
    best_acc, best, trace = -1.0, None, []
    if latest_file.is_file():
        saved = torch_load(latest_file)
        for key, expected in (("group", group), ("fold", fold), ("source_sha256", source_hash),
                              ("identity", identity), ("warmup_sha256", identity["warmup_sha256"])):
            if saved.get(key) != expected:
                raise RuntimeError(f"latest state identity mismatch {key} for {group}/{fold}")
        load_bundle(state, module, optimizer_prcl, saved["current"])
        rt.restore_rng(saved["rng"])
        start_epoch, wait = int(saved["next_epoch"]), int(saved["wait"])
        best_acc, best, trace = float(saved["best_acc"]), saved["best"], saved["rng_trace"]
    elif rt.rng_sha(rt.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold} did not begin at the restored E61 epoch-10 RNG")
    if not 10 <= start_epoch <= CFG["base"]["max_epochs"]:
        raise RuntimeError(f"invalid resume epoch {start_epoch}")

    hist = runtime().load_history()
    residual_train = np.setdiff1d(state["train_np"], state["test_np"]).astype(np.int64)
    residual_ids = torch.as_tensor(residual_train, dtype=torch.long, device=state["parts"][0].device)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    executed = start_epoch
    for epoch in range(start_epoch, CFG["base"]["max_epochs"]):
        rng_before = rt.rng_sha(rt.pack_rng())
        hist.zero_optimizers(state)
        optimizer_prcl.zero_grad(set_to_none=True)
        out, captured = hook_forward(state, module, group, training=True)
        details = captured["details"]
        l_res = PRCL_IMPL.residual_penalty(details["delta"], captured["z_cam"], residual_ids)
        total = out["total"] + CFG["prcl"]["residual_penalty_weight"] * l_res
        if not torch.isfinite(total) or not torch.isfinite(details["delta"]).all():
            raise FloatingPointError(f"nonfinite PRCL loss/residual at {group} fold={fold} epoch={epoch + 1}")
        total.backward()
        for module_check, label in ((state["model"], "DARG"), (state["cam"], "CAM"), (module, "PRCL")):
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in module_check.parameters()):
                raise FloatingPointError(f"nonfinite {label} gradient at {group} fold={fold} epoch={epoch + 1}")
        grad_norms = {
            "prcl": gradient_l2(module.parameters()),
            "cam": gradient_l2(state["cam"].parameters()),
            "darg_gfr": gradient_l2(state["model"].parameters()),
        }
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        optimizer_prcl.step()
        rng_after_train = rt.rng_sha(rt.pack_rng())
        executed = epoch + 1
        train_stats = {
            "losses": {key: float(out[key].detach().cpu()) for key in ("cls", "graph", "pair", "adv", "preserve", "total")},
            "l_res": float(l_res.detach().cpu()), "total_with_residual": float(total.detach().cpu()),
            "gamma": float(details["gamma"].detach().cpu()),
            "text_norm_mean": float(details["text_term_norm"].mean().detach().cpu()),
            "visual_query_norm_mean": float(details["visual_query_norm"].mean().detach().cpu()),
            "text_visual_norm_ratio": float((details["text_term_norm"].mean() * math.sqrt(32) * details["gamma"] / (details["visual_query_norm"].mean() + 1e-12)).detach().cpu()),
            "alpha_mean": float(details["alpha"].detach().mean().cpu()),
            "delta_ratio_mean": float(details["delta_ratio"].detach().mean().cpu()),
            "delta_ratio_p50": float(torch.quantile(details["delta_ratio"].detach(), 0.50).cpu()),
            "delta_ratio_p95": float(torch.quantile(details["delta_ratio"].detach(), 0.95).cpu()),
            "cap_trigger_rate": float((details["cap_scale"].detach() < 0.999999).float().mean().cpu()),
            "gradient_norms": grad_norms,
        }
        with torch.no_grad():
            eval_out, eval_capture = hook_forward(state, module, group, training=False)
            metrics = rt.classification_metrics(eval_out["logp"], state["test"], state["y"])
            eval_details = eval_capture["details"]
        rng_after_eval = rt.rng_sha(rt.pack_rng())
        rng_epoch = {"epoch": executed, "rng_before": rng_before,
                     "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval}
        trace.append(rng_epoch)
        row = {
            "epoch": executed, "metrics": {key: float(metrics[key]) for key in METRICS},
            "losses": train_stats["losses"], "l_res": train_stats["l_res"],
            "prcl": {key: value for key, value in train_stats.items() if key not in ("losses", "l_res", "total_with_residual", "gradient_norms")},
            "gradient_norms": train_stats["gradient_norms"], "rng": rng_epoch,
        }
        log_jsonl(log_file, row)
        if metrics["acc"] > best_acc:
            best_acc = float(metrics["acc"])
            wait = 0
            best = {
                "epoch": executed,
                "metrics": {key: float(metrics[key]) for key in METRICS},
                "confusion": {key: int(metrics[key]) for key in ("asd_tp", "asd_fn", "td_fp", "td_tn")},
                "probability": np.asarray(metrics["prob"]).copy(),
                "prediction": np.asarray(metrics["pred"]).copy(),
                "test_indices": np.asarray(state["test_np"], dtype=np.int64).copy(),
                "test_logp": eval_out["logp"].detach().cpu().clone(),
                "state": state_bundle(state, module, optimizer_prcl),
                "rng": {**rng_epoch, "packed": rt.pack_rng()},
                "losses": train_stats,
                "diagnostics": {key: value.detach().cpu().clone() for key, value in eval_details.items()
                                if torch.is_tensor(value) and key in ("attention", "query", "key", "value", "tokens", "normal_message", "abnormal_message", "relation_message", "alpha", "delta", "delta_ratio", "cap_scale", "raw_delta", "visual_query_norm", "text_term_norm")},
            }
            atomic_torch(best_file, checkpoint_payload(group, fold, source_hash, identity, best,
                                                        executed, trace, init_record,
                                                        time.time() - started, torch.cuda.max_memory_allocated()))
        else:
            wait += 1
        if executed % 10 == 0 or executed == CFG["base"]["max_epochs"] or wait > CFG["base"]["early_stop_wait_gt"]:
            atomic_torch(latest_file, {
                "group": group, "fold": fold, "source_sha256": source_hash, "identity": identity,
                "warmup_sha256": identity["warmup_sha256"], "current": state_bundle(state, module, optimizer_prcl),
                "rng": rt.pack_rng(), "next_epoch": executed, "wait": wait,
                "best_acc": best_acc, "best": best, "rng_trace": trace,
            })
            print(f"{group} fold={fold} epoch={executed} best={best_acc:.6f}@{best['epoch'] if best else 0} wait={wait}", flush=True)
        del out, captured, details, eval_out, eval_capture, eval_details, total, l_res
        if wait > CFG["base"]["early_stop_wait_gt"]:
            break
    if best is None:
        raise RuntimeError(f"no strict-ACC Checkpoint selected for {group} fold={fold}")
    elapsed = time.time() - started
    peak = int(torch.cuda.max_memory_allocated())
    atomic_torch(best_file, checkpoint_payload(group, fold, source_hash, identity, best,
                                                executed, trace, init_record, elapsed, peak))
    atomic_torch(latest_file, {
        "group": group, "fold": fold, "source_sha256": source_hash, "identity": identity,
        "warmup_sha256": identity["warmup_sha256"], "current": state_bundle(state, module, optimizer_prcl),
        "rng": rt.pack_rng(), "next_epoch": executed, "wait": wait,
        "best_acc": best_acc, "best": best, "rng_trace": trace,
    })
    task = {
        "status": "COMPLETE", "group": group, "fold": fold, "seed": 0,
        "source_sha256": source_hash, "checkpoint_sha256": sha_file(best_file),
        "latest_sha256": sha_file(latest_file), "identity": identity,
        "initialization": init_record, "best_epoch_1based": best["epoch"],
        "epochs_executed": executed, "metrics": best["metrics"],
        "confusion": best["confusion"], "rng_trace": trace,
        "wall_seconds": elapsed, "peak_gpu_bytes": peak,
        "device": torch.cuda.get_device_name(0), "warmup_sha256": init_record["warmup_sha256"],
    }
    atomic_json(task_file, task)
    print(f"COMPLETE {group} fold={fold} best={best_acc:.6f}@{best['epoch']}", flush=True)
    return task


def restore_step_replay(fold=1):
    """Compare two real DARG+CAM+PRCL updates with a step-one disk restore."""
    rt = base_runtime()
    history = runtime().load_history()
    cache_path = RUNTIME / "recovery_replay.pt"

    def fresh():
        state, warmup, identity = prepare(fold)
        state["fold"] = fold
        module = make_module(fold, GROUPS[1], state["parts"][0].device)
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01)
        return state, warmup, identity, module, optimizer

    def update(state, module, optimizer):
        history.zero_optimizers(state)
        optimizer.zero_grad(set_to_none=True)
        out, captured = hook_forward(state, module, GROUPS[1], training=True)
        details = captured["details"]
        train_ids = np.setdiff1d(state["train_np"], state["test_np"]).astype(np.int64)
        ids = torch.as_tensor(train_ids, dtype=torch.long, device=details["delta"].device)
        residual = PRCL_IMPL.residual_penalty(details["delta"], captured["z_cam"], ids)
        total = out["total"] + CFG["prcl"]["residual_penalty_weight"] * residual
        total.backward()
        gradients = {"prcl": gradient_l2(module.parameters()),
                     "cam": gradient_l2(state["cam"].parameters()),
                     "darg_gfr": gradient_l2(state["model"].parameters())}
        if any(value <= 0 or not math.isfinite(value) for value in gradients.values()):
            raise RuntimeError(f"recovery replay found a missing/nonfinite gradient group: {gradients}")
        train_logp = out["logp"].detach().cpu().clone()
        train_attention = details["attention"].detach().cpu().clone()
        for name in ("opt_base", "opt_cam", "opt_added"):
            state[name].step()
        optimizer.step()
        rng_after_train = rt.pack_rng()
        with torch.no_grad():
            eval_out, eval_capture = hook_forward(state, module, GROUPS[1], training=False)
            eval_logp = eval_out["logp"].detach().cpu().clone()
            eval_attention = eval_capture["details"]["attention"].detach().cpu().clone()
            eval_delta = eval_capture["details"]["delta"].detach().cpu().clone()
        record = {
            "loss": float(total.detach().cpu()),
            "loss_parts": {key: float(out[key].detach().cpu()) for key in ("cls", "graph", "pair", "adv", "preserve", "total")},
            "residual_loss": float(residual.detach().cpu()),
            "gradient_norms": gradients,
            "train_logp": train_logp,
            "train_attention": train_attention,
            "logp": eval_logp,
            "probability": torch.softmax(eval_out["logp"].detach(), dim=-1).cpu().clone(),
            "attention": eval_attention,
            "delta": eval_delta,
            "rng_after_train": rng_after_train,
            "rng": rt.pack_rng(),
            "rng_sha": rt.rng_sha(rt.pack_rng()),
        }
        del out, captured, details, residual, total, eval_out, eval_capture
        return record

    state_a, warmup_a, _, module_a, opt_a = fresh()
    start_bundle = state_bundle(state_a, module_a, opt_a)
    start_rng = rt.pack_rng()
    if rt.rng_sha(start_rng) != warmup_a["rng_sha"]:
        raise RuntimeError("recovery replay did not begin from the E61 epoch-10 RNG")
    rt.restore_rng(start_rng)
    a1 = update(state_a, module_a, opt_a)
    a2 = update(state_a, module_a, opt_a)
    expected_bundle = state_bundle(state_a, module_a, opt_a)
    del state_a, module_a, opt_a
    torch.cuda.empty_cache()

    state_b, warmup_b, _, module_b, opt_b = fresh()
    load_bundle(state_b, module_b, opt_b, start_bundle)
    rt.restore_rng(start_rng)
    b1 = update(state_b, module_b, opt_b)
    compare_tree(a1["loss"], b1["loss"], "recovery.step1.loss")
    compare_tree(a1["loss_parts"], b1["loss_parts"], "recovery.step1.loss_parts")
    compare_tree(a1["gradient_norms"], b1["gradient_norms"], "recovery.step1.gradient_norms")
    compare_tree(a1["train_logp"], b1["train_logp"], "recovery.step1.train_logp")
    compare_tree(a1["train_attention"], b1["train_attention"], "recovery.step1.train_attention")
    compare_tree(a1["logp"], b1["logp"], "recovery.step1.logp")
    compare_tree(a1["attention"], b1["attention"], "recovery.step1.attention")
    compare_tree(a1["rng_after_train"], b1["rng_after_train"], "recovery.step1.rng_after_train")
    compare_tree(a1["rng"], b1["rng"], "recovery.step1.rng")
    atomic_torch(cache_path, {
        "bundle": state_bundle(state_b, module_b, opt_b), "rng": b1["rng"],
        "step1": {key: value for key, value in b1.items() if key != "rng"},
        "identity": warmup_b["rng_sha"],
    })
    del state_b, module_b, opt_b
    torch.cuda.empty_cache()

    state_c, _, _, module_c, opt_c = fresh()
    saved = torch_load(cache_path)
    load_bundle(state_c, module_c, opt_c, saved["bundle"])
    rt.restore_rng(saved["rng"])
    c2 = update(state_c, module_c, opt_c)
    resumed_bundle = state_bundle(state_c, module_c, opt_c)
    compare_tree(a2["loss"], c2["loss"], "recovery.step2.loss")
    compare_tree(a2["loss_parts"], c2["loss_parts"], "recovery.step2.loss_parts")
    compare_tree(a2["gradient_norms"], c2["gradient_norms"], "recovery.step2.gradient_norms")
    compare_tree(a2["train_logp"], c2["train_logp"], "recovery.step2.train_logp")
    compare_tree(a2["train_attention"], c2["train_attention"], "recovery.step2.train_attention")
    compare_tree(a2["logp"], c2["logp"], "recovery.step2.logp")
    compare_tree(a2["probability"], c2["probability"], "recovery.step2.probability")
    compare_tree(a2["attention"], c2["attention"], "recovery.step2.attention")
    compare_tree(a2["delta"], c2["delta"], "recovery.step2.delta")
    compare_tree(expected_bundle, resumed_bundle, "recovery.final_bundle")
    compare_tree(a2["rng"], c2["rng"], "recovery.final_rng")
    cache_path.unlink(missing_ok=True)
    return {"status": "PASS", "steps": 2, "real_darg_cam_prcl_forward": True,
            "base_model_and_cam": "PASS", "all_three_inherited_optimizers": "PASS",
            "prcl_and_fourth_optimizer": "PASS", "rng": "PASS", "loss": "PASS",
            "probability": "PASS", "logp": "PASS", "attention": "PASS",
            "residual": "PASS", "disk_checkpoint_restore": "PASS"}


def full_preflight():
    pre = preflight()
    recovery = restore_step_replay()
    pre["two_step_recovery_replay"] = recovery
    atomic_json(RESULTS / "preflight.json", pre)
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    manifest["preflight"]["two_step_recovery_replay"] = recovery
    atomic_json(HERE / "manifest.json", manifest)
    return pre


def worker_process(group, fold):
    cmd = [sys.executable, str(HERE / "run.py"), "--worker", group, str(fold)]
    return subprocess.run(cmd, text=True, capture_output=True, env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})


def dispatch(workers=10):
    source_hash = frozen_source_sha()
    jobs = [(group, fold) for fold in range(1, 11) for group in GROUPS]
    start = time.time()
    failures = []
    with ThreadPoolExecutor(max_workers=min(10, max(1, workers))) as pool:
        futures = {pool.submit(worker_process, group, fold): (group, fold) for group, fold in jobs}
        for future in as_completed(futures):
            group, fold = futures[future]
            result = future.result()
            log_path = LOGS / f"dispatch_{group}_f{fold:02d}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr, encoding="utf-8")
            print(f"JOB {group} fold={fold} exit={result.returncode}", flush=True)
            if result.returncode != 0:
                failures.append({"group": group, "fold": fold, "returncode": result.returncode,
                                 "tail": (result.stdout + "\n" + result.stderr)[-4000:]})
    tasks = []
    for group, fold in jobs:
        task_file, best_file, latest_file, _ = paths(group, fold)
        if task_file.is_file():
            task = json.loads(task_file.read_text(encoding="utf-8"))
            if task.get("status") == "COMPLETE" and task.get("source_sha256") == source_hash and task.get("checkpoint_sha256") == sha_file(best_file):
                tasks.append(task)
    wall = time.time() - start
    manifest_path = HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attempt = {
        "source_sha256": source_hash, "worker_limit": min(10, workers),
        "elapsed_seconds": wall, "completed_tasks": len(tasks),
        "failed_task_count": len(failures),
        "failed_task_ids": [f"{item['group']}_f{int(item['fold']):02d}" for item in failures],
        "failure_detail": failures[0]["tail"][-1200:] if failures else None,
        "best_checkpoints_written": len(list(CHECKPOINTS.glob("*.best.pt"))),
    }
    attempts = list(manifest.get("training_attempts", []))
    attempts.append(attempt)
    manifest.update({"status": "TRAINING_COMPLETE" if len(tasks) == 30 and not failures else "PARTIAL",
                     "completed_tasks": len(tasks), "training_wall_seconds": wall,
                     "worker_limit": min(10, workers), "failed_jobs": failures,
                     "formal_training_started": True, "training_attempts": attempts})
    atomic_json(manifest_path, manifest)
    if failures or len(tasks) != 30:
        raise RuntimeError(f"formal training incomplete: {len(tasks)}/30 tasks, failures={len(failures)}")
    print(f"TRAINING COMPLETE 30/30 wall_seconds={wall:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--worker", nargs=2, metavar=("GROUP", "FOLD"))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    for directory in (RESULTS, CHECKPOINTS, TASKS, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    if args.worker:
        train_worker(args.worker[0], int(args.worker[1]))
    elif args.preflight:
        full_preflight()
    elif args.train:
        dispatch(args.workers)
    else:
        parser.error("choose --preflight, --train or --worker")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

