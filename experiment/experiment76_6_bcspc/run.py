#!/usr/bin/env python3
"""Experiment 76.6 runner with a read-only E74.4 history adapter."""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
REPO = Path("/data/adodas/SPromptGL-main") if str(EXP).startswith("/model/spromptgl/experiment/experiment76_6_bcspc") else EXP.parents[1]
EXPERIMENTS = REPO / "experiment"
CFG_PATH = EXP / "config.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
BASE_EXP = EXPERIMENTS / "experiment74_4_soft_positive_dcr"
E741 = EXPERIMENTS / "experiment74_1_rudag"
E742 = EXPERIMENTS / "experiment74_2_dcr_rudag"
E743 = EXPERIMENTS / "experiment74_3_local_dcr"
E69 = EXPERIMENTS / "experiment69_3_g2_two_rows"
E62 = EXPERIMENTS / "experiment62_e2e_gate_init_sensitivity"
E61 = EXPERIMENTS / "experiment61_cadr_postalign_vs_e2e"
E53 = EXPERIMENTS / "experiment53_3_cadr_graph_fallback_fix"
E72 = EXPERIMENTS / "experiment72_mgdss"
E76_5 = EXPERIMENTS / "experiment76_5_scpr"
BASE_GROUP = "E74.4-G2-Positive-DCR"
GROUPS = tuple(CFG["method"]["groups"])
RT = EXP / ".runtime"
CKPTS, TASKS, LOGS, CACHE = (RT / name for name in ("checkpoints", "tasks", "logs", "cache"))
RESULTS = EXP / "results"
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1", "nll", "brier", "ece")
BASE = HIST = E69_RUNNER = E61_RUNNER = E53_RUNNER = None


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_obj(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def tensor_bundle_sha(state):
    digest = hashlib.sha256()
    for module in ("model", "cam"):
        for name, tensor in sorted(state[module].state_dict().items()):
            digest.update(f"{module}/{name}".encode() + b"\0")
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False)):
        raise IOError(f"atomic JSON readback failed: {path}")


def atomic_torch(path, payload):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    check = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(check, dict):
        raise IOError(f"atomic checkpoint readback failed: {path}")


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path, rows):
    path = Path(path)
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


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _patch_history_paths():
    """Redirect moved absolute paths in memory; all source/config files stay read-only."""
    global BASE, HIST, E69_RUNNER, E61_RUNNER, E53_RUNNER
    if BASE is not None:
        return BASE
    for name, path in (("E741", E741), ("E742", E742), ("E743", E743),
                       ("E69", E69), ("E62", E62), ("E74", EXPERIMENTS / "experiment74_udag"),
                       ("E53", E53), ("E61", E61)):
        if not path.is_dir():
            raise FileNotFoundError(f"required moved history directory missing: {path}")
    sys.path.insert(0, str(REPO))
    BASE = import_file("e744_read_only_runner_for_bcspc", BASE_EXP / "run.py")
    BASE.E741, BASE.E742, BASE.E743 = E741, E742, E743
    BASE.E69, BASE.E62, BASE.E74 = E69, E62, EXPERIMENTS / "experiment74_udag"
    BASE.E53, BASE.E61 = E53, E61
    dcr_adapter = BASE.load_dcr()
    dcr_adapter.CADR_PATH = E53 / "cadr_gfr.py"
    HIST = BASE.load_history()
    HIST.BASE = E69
    HIST.E53, HIST.E62, HIST.E74 = E53, E62, EXPERIMENTS / "experiment74_udag"
    HIST.E61_EXP = E61
    original_loader = HIST.load_e69

    def load_e69_compatible():
        global E69_RUNNER, E61_RUNNER
        if HIST.E69 is not None:
            return HIST.E69
        expected = HIST.CFG["base"]
        if sha_file(E69 / "run.py") != expected["run_sha256"] or sha_file(E69 / "manifest.json") != expected["manifest_sha256"]:
            raise RuntimeError("moved E69.3 source or manifest SHA mismatch")
        E69_RUNNER = import_file("e693_read_only_runner_for_bcspc", E69 / "run.py")
        E69_RUNNER.E62_EXP = E62
        E69_RUNNER.E62_RUN = E62 / "run.py"
        E69_RUNNER.E62_CONFIG = E62 / "config.json"
        E69_RUNNER.E62_MANIFEST = E62 / "manifest.json"
        E69_RUNNER.E61_EXP = E61
        E69_RUNNER.E61_RUN = E61 / "run.py"
        E69_RUNNER.E61_MODEL = E61 / "model_e61.py"
        E69_RUNNER.E61_CONFIG = E61 / "config.json"
        E69_RUNNER.E61_MANIFEST = E61 / "manifest.json"
        E69_RUNNER.E53_EXP = E53
        E69_RUNNER.E61_EXP = E61
        legacy_vendor = E72 / ".runtime/vendor/root/experiment/experiment51_dugsa"
        vendor_manifest = json.loads((E72 / "manifest.json").read_text(encoding="utf-8"))
        def find_protected_sources(value):
            if isinstance(value, dict):
                candidate = value.get("protected_source_sha256")
                if isinstance(candidate, dict):
                    return candidate
                for nested in value.values():
                    found = find_protected_sources(nested)
                    if found:
                        return found
            elif isinstance(value, list):
                for nested in value:
                    found = find_protected_sources(nested)
                    if found:
                        return found
            return {}
        protected = find_protected_sources(vendor_manifest)
        vendor_files = {
            "dugsa.py": legacy_vendor / "dugsa.py",
            "source/network.py": legacy_vendor / "source/network.py",
            "source/layers.py": legacy_vendor / "source/layers.py",
            "source/utils.py": legacy_vendor / "source/utils.py",
        }
        for relative, path in vendor_files.items():
            original = REPO / "experiment/experiment51_dugsa" / relative
            expected = protected.get(str(original.resolve()))
            if expected is None or sha_file(path) != expected:
                raise RuntimeError(f"E51 compatibility vendor SHA mismatch: {path}")
        if str(legacy_vendor) not in sys.path:
            sys.path.append(str(legacy_vendor))
        E61_RUNNER = E69_RUNNER.load_e61()
        E61_RUNNER.BASE_ROOT = REPO
        E61_RUNNER.BASE_EXP = E53
        E61_RUNNER.BASE_RUN = E53 / "run.py"
        E61_RUNNER.BASE_CADR = E53 / "cadr_gfr.py"
        E61_RUNNER.BASE_RUNTIME = E53 / ".runtime/fallback_wd_sweep" / E61_RUNNER.BASE_GROUP
        anchor_vendor = E72 / ".runtime/vendor/root/experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
        E61_RUNNER.ANCHORS = anchor_vendor
        E61_RUNNER.E53_EXP = E53
        E61_RUNNER.E61_EXP = E61
        e53_manifest = json.loads((E53 / "manifest.json").read_text(encoding="utf-8"))
        anchor_hashes = e53_manifest.get("anchor_sha256", {})
        for fold in range(1, 11):
            anchor = anchor_vendor / f"fold{fold:02d}.pt"
            original = REPO / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors" / f"fold{fold:02d}.pt"
            if not anchor.is_file() or sha_file(anchor) != anchor_hashes.get(str(original.resolve())):
                raise RuntimeError(f"E43 warm-start anchor SHA mismatch: {anchor}")
        E53_RUNNER = E61_RUNNER.load_base()
        E53_RUNNER.LEGACY_EXP = legacy_vendor
        E53_RUNNER.LEGACY_SOURCE = legacy_vendor / "source"
        if str(E53_RUNNER.LEGACY_SOURCE) not in sys.path:
            sys.path.insert(0, str(E53_RUNNER.LEGACY_SOURCE))
        if str(legacy_vendor) not in sys.path:
            sys.path.insert(0, str(legacy_vendor))
        E53_RUNNER.ANCHORS = anchor_vendor
        E69_RUNNER.verify_e62_reference()
        HIST.E69 = E69_RUNNER
        return E69_RUNNER

    HIST.load_e69 = load_e69_compatible
    if original_loader is None:
        raise RuntimeError("E74.1 historical loader is unavailable")
    return BASE


def load_base():
    _patch_history_paths()
    return BASE


def history_runtime():
    load_base()
    return HIST.load_e69().load_e61().load_base()


def prepare_fold(fold):
    base = load_base()
    state, warmup, identity = base.prepare(fold)
    state["fold"] = fold
    state["group"] = BASE_GROUP
    expected_task_path = BASE_EXP / f".runtime/tasks/{BASE_GROUP}_f{fold:02d}.json"
    task = json.loads(expected_task_path.read_text(encoding="utf-8"))
    ckpt_path = BASE_EXP / f".runtime/checkpoints/{BASE_GROUP}_f{fold:02d}.best.pt"
    if task.get("status") != "COMPLETE" or sha_file(ckpt_path) != task.get("checkpoint_sha256"):
        raise RuntimeError(f"E74.4-G2 checkpoint/task identity failed, fold={fold}")
    actual_warmup_sha = sha_file(E61 / ".runtime/warmup" / f"fold{fold:02d}.pt")
    if identity != task.get("identity") or identity.get("warmup_sha256") != actual_warmup_sha:
        raise RuntimeError(f"E61 warmup / E74.4 initialization mismatch, fold={fold}")
    if state["train_pairs_np"].shape != (4096, 2):
        raise RuntimeError(f"expected 4096 relation pairs in fold {fold}")
    expected_overlap = [377, 407] if fold == 2 else []
    if identity["membership"]["train_test_overlap_1based"] != expected_overlap:
        raise RuntimeError(f"unexpected historical train/test overlap in fold {fold}")
    if history_runtime().rng_sha(history_runtime().pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold} failed to restore the exact E61 epoch-10 RNG")
    return state, warmup, identity, task, ckpt_path


def hook_forward(state, training):
    captured = []
    output_layer = state["cam"].Outputlayer
    handle = output_layer.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0]))
    try:
        output = load_base().forward(state, BASE_GROUP, training=training)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape != (871, 144):
        raise RuntimeError(f"CAM144 hook expected one [871,144] tensor, got {[tuple(t.shape) for t in captured]}")
    return output, captured[0]


def prompt_vectors(state):
    import torch
    from bcspc import PROMPTS_ASD_TD
    cache_path = E76_5 / ".runtime/cache/brainprompt_rn50.pt"
    if not cache_path.is_file():
        raise FileNotFoundError(f"audited frozen BrainPrompt text cache missing: {cache_path}")
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    if tuple(cached.get("prompts", ())) != PROMPTS_ASD_TD:
        raise RuntimeError("BrainPrompt official ABIDE prompt strings/order mismatch")
    if cached.get("text_encoder") != "BrainPrompt frozen CLIP RN50" or cached.get("rn50_sha256") != "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762":
        raise RuntimeError("BrainPrompt text-cache encoder provenance mismatch")
    brainprompt = cached["vectors"].detach().float().cpu()
    e72_cache_path = E72 / ".runtime/text/text_embeddings.pt"
    e72_cache = torch.load(e72_cache_path, map_location="cpu", weights_only=False)
    tree = ast.parse((REPO / "network.py").read_text(encoding="utf-8"))
    original_prompts = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "attr_words" for target in node.targets):
            original_prompts = ast.literal_eval(node.value)
            break
    if original_prompts is None or e72_cache.get("text_map", {}).get("original") != original_prompts:
        raise RuntimeError("cached original SPromptGL prompt text differs from network.py")
    if e72_cache.get("source_hashes", {}).get("rn50_sha256") != cached["rn50_sha256"]:
        raise RuntimeError("SPromptGL and BrainPrompt prompt encodings use different RN50 weights")
    spromptgl = e72_cache["embeddings"]["original"].detach().float().cpu()
    if brainprompt.shape != (2, 1024) or spromptgl.shape != (8, 1024):
        raise RuntimeError(f"unexpected frozen prompt embedding shapes: {brainprompt.shape}, {spromptgl.shape}")
    return brainprompt, spromptgl


def read_sites():
    import torch
    data_path = REPO / "data/ABIDE/processed_standard_data.csv"
    with data_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames or []
    columns = [name for name in fields if name.upper().startswith("SITE_ID_")]
    if len(rows) != 871 or not columns:
        raise RuntimeError(f"site metadata is absent or misaligned: rows={len(rows)}, columns={len(columns)}")
    values = [[float(row.get(name) or 0) for name in columns] for row in rows]
    if any(sum(row) != 1.0 for row in values):
        raise RuntimeError("expected exactly one site one-hot per participant")
    return torch.tensor([max(range(len(row)), key=row.__getitem__) for row in values], dtype=torch.long)


def split_branch_train_ids(train_ids, test_ids, labels, fold):
    from bcspc import split_support
    test_set = set(int(i) for i in test_ids)
    eligible = sorted(int(i) for i in train_ids if int(i) not in test_set)
    if fold == 2:
        excluded = {376, 406}
        eligible = [i for i in eligible if i not in excluded]
    ids_a, ids_b = split_support(eligible, labels)
    if set(ids_a) & set(ids_b) or set(ids_a + ids_b) != set(eligible):
        raise RuntimeError("fixed A/B branch support split is not a disjoint cover")
    if set(eligible) & test_set or (fold == 2 and ({376, 406} & set(eligible))):
        raise RuntimeError("outer test or overlapping fold-2 rows entered BCSPC train/support")
    return eligible, ids_a, ids_b


def make_memory(train_ids, labels, cam144, logp, sites, old=None):
    import torch
    device = cam144.device
    if old is None:
        memory = {
            "features": torch.zeros((871, 4, 36), dtype=cam144.dtype, device=device),
            "probabilities": torch.zeros((871, 2), dtype=logp.dtype, device=device),
            "labels": torch.full((871,), -1, dtype=torch.long, device=device),
            "sites": sites.to(device).clone(),
            "observed": torch.zeros((871,), dtype=torch.bool, device=device),
            "updates": 0,
        }
    else:
        memory = old
    ids = torch.as_tensor(train_ids, dtype=torch.long, device=device)
    features = cam144.detach().reshape(871, 4, 36).to(device)
    probabilities = logp.detach().exp().to(device)
    observed = memory["observed"][ids]
    fresh, prior = ids[~observed], ids[observed]
    if fresh.numel():
        memory["features"][fresh] = features[fresh]
        memory["probabilities"][fresh] = probabilities[fresh]
    if prior.numel():
        memory["features"][prior] = 0.9 * memory["features"][prior] + 0.1 * features[prior]
        memory["probabilities"][prior] = 0.9 * memory["probabilities"][prior] + 0.1 * probabilities[prior]
    y = labels.to(device)
    memory["labels"][ids] = y[ids]
    memory["sites"] = sites.to(device).clone()
    memory["observed"][ids] = True
    memory["updates"] += 1
    if memory["features"].grad_fn is not None or memory["probabilities"].grad_fn is not None:
        raise RuntimeError("BCSPC EMA memory must contain detached tensors only")
    return memory


def memory_snapshot(memory):
    if memory is None:
        return None
    return {key: (value.detach().cpu().clone() if hasattr(value, "detach") else value)
            for key, value in memory.items()}


def move_memory(memory, device):
    if memory is None:
        return None
    return {key: (value.to(device) if hasattr(value, "to") else value) for key, value in memory.items()}


def init_groups(state, brainprompt, spromptgl):
    import torch
    from bcspc import init_isolated
    runtime = history_runtime()
    rng_before = runtime.rng_sha(runtime.pack_rng())
    modules = [init_isolated(brainprompt, spromptgl, seed=766101).to(state["y"].device),
               init_isolated(brainprompt, spromptgl, seed=766101).to(state["y"].device)]
    optimizers = [torch.optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01) for module in modules]
    if runtime.rng_sha(runtime.pack_rng()) != rng_before:
        raise RuntimeError("BCSPC initialization or optimizer construction changed the inherited RNG")
    if sum(p.numel() for p in modules[0].parameters()) != sum(p.numel() for p in modules[1].parameters()):
        raise RuntimeError("G1/G2 parameter counts differ")
    for first, second in zip(modules[0].state_dict().values(), modules[1].state_dict().values()):
        if not torch.equal(first.cpu(), second.cpu()):
            raise RuntimeError("G1/G2 initial module state differs")
    return modules, optimizers


def _cat_diagnostics(items):
    import torch
    merged = {}
    scalar_keys = {"retrieval_fallbacks", "invalid_confusion_candidates", "site_cap_violations"}
    for key in items[0]:
        values = [item[key] for item in items]
        if key in scalar_keys:
            merged[key] = sum(int(value) for value in values)
        else:
            merged[key] = torch.cat(values, dim=0)
    return merged


def train_branch(module, cam144, memory, labels_all, base_logp, train_ids, ids_a, ids_b,
                 sites, no_boundary):
    import torch
    from bcspc import branch_losses
    y = labels_all.to(cam144.device)
    id_to_position = {int(value): position for position, value in enumerate(train_ids)}
    query_scores, query_diagnostics, query_base, query_labels = [], [], [], []
    a_set = set(ids_a)
    for query_group, support_ids in ((ids_a, ids_b), (ids_b, ids_a)):
        if not query_group:
            continue
        positions = torch.as_tensor([id_to_position[int(i)] for i in query_group], dtype=torch.long, device=cam144.device)
        global_ids = torch.as_tensor(query_group, dtype=torch.long, device=cam144.device)
        support = torch.as_tensor(support_ids, dtype=torch.long, device=cam144.device)
        base_score = base_logp[global_ids, 0] - base_logp[global_ids, 1]
        scores, diagnostics = module(cam144[global_ids].detach(), memory, support, sites.to(cam144.device),
                                     base_score.detach(), no_boundary=no_boundary)
        query_scores.append(scores)
        query_diagnostics.append(diagnostics)
        query_base.append(base_score.detach())
        query_labels.append(y[global_ids])
    scores = torch.cat(query_scores)
    diagnostics = _cat_diagnostics(query_diagnostics)
    base_score = torch.cat(query_base)
    labels = torch.cat(query_labels)
    loss, components = branch_losses(scores, diagnostics, base_score, labels)
    return loss, components, scores.detach(), diagnostics, base_score, labels


def eval_branch(module, base_logp, eval_cam, test_ids, support_ids, memory, labels, sites,
                no_boundary, *, residual_off=False, boundary_off=False, semantic_swap=False):
    import torch
    from bcspc import final_log_probabilities
    ids = torch.as_tensor(test_ids, dtype=torch.long, device=base_logp.device)
    support = torch.as_tensor(support_ids, dtype=torch.long, device=base_logp.device)
    base_score = base_logp[ids, 0] - base_logp[ids, 1]
    score, diagnostics = module(eval_cam[ids].detach(), memory, support, sites.to(base_logp.device),
                                base_score, no_boundary=no_boundary, residual_off=residual_off,
                                boundary_off=boundary_off, semantic_swap=semantic_swap)
    corrected = final_log_probabilities(base_logp[ids], base_score, score, module)
    result = base_logp.clone()
    result[ids] = corrected
    runtime = history_runtime()
    metrics = extended_metrics(result, test_ids, labels)
    return result, metrics, diagnostics


def extended_metrics(logp, test_ids, labels):
    import numpy as np
    metrics = history_runtime().classification_metrics(
        logp, np.asarray(test_ids, dtype=np.int64), labels)
    ids = np.asarray(test_ids, dtype=np.int64)
    y = labels.detach().cpu().numpy()[ids]
    selected_logp = logp.detach().cpu().numpy()[ids]
    probability = np.exp(selected_logp)
    nll = -selected_logp[np.arange(len(ids)), y].mean()
    one_hot = np.eye(2, dtype=np.float64)[y]
    brier = np.square(probability - one_hot).sum(axis=1).mean()
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    ece = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, 16)[:-1], np.linspace(0.0, 1.0, 16)[1:]):
        in_bin = (confidence > lower) & (confidence <= upper if upper < 1.0 else confidence <= upper)
        if in_bin.any():
            ece += float(in_bin.mean() * abs((prediction[in_bin] == y[in_bin]).mean() - confidence[in_bin].mean()))
    metrics.update({"nll": float(nll), "brier": float(brier), "ece": float(ece)})
    return metrics


def source_snapshot():
    files = [
        CFG_PATH, EXP / "run.py", EXP / "bcspc.py", EXP / "audit.py",
        BASE_EXP / "run.py", BASE_EXP / "soft_positive_dcr.py", BASE_EXP / "config.json",
        BASE_EXP / "manifest.json", BASE_EXP / "audit.py", BASE_EXP / "results/audit_status.json",
        REPO / "main.py", REPO / "model.py", REPO / "network.py", REPO / "layers.py", REPO / "utils.py",
        REPO / "data/ABIDE/processed_standard_data.csv", REPO / "data/ABIDE/modal_feat_dict.npy",
        E741 / "run.py", E741 / "rudag.py", E741 / "config.json", E741 / "manifest.json",
        E742 / "run.py", E742 / "dcr_rudag.py", E742 / "config.json", E742 / "manifest.json",
        E743 / "run.py", E743 / "local_dcr.py", E743 / "config.json", E743 / "manifest.json",
        E69 / "run.py", E69 / "config.json", E69 / "manifest.json",
        E62 / "run.py", E62 / "config.json", E62 / "manifest.json",
        E61 / "run.py", E61 / "model_e61.py", E61 / "config.json", E61 / "manifest.json",
        E53 / "run.py", E53 / "cadr_gfr.py", E53 / "config.json", E53 / "manifest.json",
        E72 / "manifest.json",
        E72 / ".runtime/vendor/root/experiment/experiment51_dugsa/dugsa.py",
        E72 / ".runtime/vendor/root/experiment/experiment51_dugsa/config.json",
        E72 / ".runtime/vendor/root/experiment/experiment51_dugsa/run.py",
        E72 / ".runtime/vendor/root/experiment/experiment51_dugsa/source/network.py",
        E72 / ".runtime/vendor/root/experiment/experiment51_dugsa/source/layers.py",
        E72 / ".runtime/vendor/root/experiment/experiment51_dugsa/source/utils.py",
        E72 / ".runtime/vendor/root/experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors/index.json",
        E72 / ".runtime/text/text_embeddings.pt",
        E76_5 / ".runtime/cache/brainprompt_rn50.pt",
    ]
    files.extend(sorted((E61 / ".runtime/warmup").glob("fold*.pt")))
    files.extend(sorted((E61 / ".runtime/tasks").glob("fold*.warmup.json")))
    files.extend(sorted((E72 / ".runtime/vendor/root/experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors").glob("fold*.pt")))
    files.extend(sorted((BASE_EXP / ".runtime/checkpoints").glob(f"{BASE_GROUP}_f*.best.pt")))
    files.extend(sorted((BASE_EXP / ".runtime/tasks").glob(f"{BASE_GROUP}_f*.json")))
    if len(list((E61 / ".runtime/warmup").glob("fold*.pt"))) != 10 or len(list((BASE_EXP / ".runtime/checkpoints").glob(f"{BASE_GROUP}_f*.best.pt"))) != 10:
        raise RuntimeError("E61 warmup or E74.4-G2 fold checkpoint inventory is incomplete")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("required source/history files are missing: " + ", ".join(missing))
    return {str(path.resolve()): sha_file(path) for path in files}


def _optimizer_sha(state):
    return load_base().sha_object({name: state[name].state_dict() for name in ("opt_base", "opt_cam", "opt_added")})


def _module_grad_norm(module, name):
    parameter = dict(module.named_parameters())[name]
    return 0.0 if parameter.grad is None else float(parameter.grad.detach().norm().cpu())


def _finite_gradients(module):
    import torch
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in module.parameters())


def _base_grad_snapshot(state):
    output = {}
    for module_name in ("model", "cam"):
        for name, parameter in state[module_name].named_parameters():
            output[f"{module_name}/{name}"] = None if parameter.grad is None else parameter.grad.detach().cpu().clone()
    return output


def _same_gradients(reference, state):
    import torch
    for module_name in ("model", "cam"):
        for name, parameter in state[module_name].named_parameters():
            expected = reference[f"{module_name}/{name}"]
            actual = None if parameter.grad is None else parameter.grad.detach().cpu()
            if (expected is None) != (actual is None):
                return False
            if expected is not None and not torch.equal(expected, actual):
                return False
    return True


def _build_memory_for_preflight(state, train_ids, sites, cam, logp):
    return make_memory(train_ids, state["y"], cam, logp, sites)


def _site_cap_preflight(module, memory, support_ids, sites, query):
    import torch
    violations, checked, fallback_total = 0, 0, 0
    support = torch.as_tensor(support_ids, dtype=torch.long, device=query.device)
    for cls in (0, 1):
        ids = support[memory["labels"][support] == cls]
        for slot in range(4):
            candidate = ids
            if candidate.numel() == 0:
                continue
            sim = torch.nn.functional.normalize(query[:1, slot], dim=-1) @ torch.nn.functional.normalize(memory["features"][candidate, slot], dim=-1).T
            selected, fallback = module._site_diverse_indices(sim, candidate, sites.to(query.device), min(8, candidate.numel()))
            chosen = candidate[selected[0]]
            fallback_count = int(fallback[0].item())
            fallback_total += fallback_count
            for site in torch.unique(sites.to(query.device)[chosen]):
                over = int((sites.to(query.device)[chosen] == site).sum().item() > 2)
                if over and fallback_count == 0:
                    violations += over
            checked += chosen.numel()
    return violations, checked, fallback_total


def preflight():
    import numpy as np
    import torch
    from bcspc import branch_losses
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E76.6 requires the single configured training accelerator")
    _patch_history_paths()
    before_sources = source_snapshot()
    base_audit = json.loads((BASE_EXP / "results/audit_status.json").read_text(encoding="utf-8"))
    base_manifest = json.loads((BASE_EXP / "manifest.json").read_text(encoding="utf-8"))
    if base_manifest.get("status") != "COMPLETE_AUDITED" or base_audit.get("status") != "PASS":
        raise RuntimeError("E74.4-G2 is not COMPLETE_AUDITED")
    inherited = []
    for fold in range(1, 11):
        state, warmup, identity, task, checkpoint_path = prepare_fold(fold)
        expected = torch_load(checkpoint_path)
        if expected.get("best_epoch_1based") != task.get("best_epoch_1based"):
            raise RuntimeError(f"E74.4 checkpoint/task epoch mismatch fold={fold}")
        inherited.append({
            "fold": fold,
            "warmup_sha256": identity["warmup_sha256"],
            "warmup_rng_sha": warmup["rng_sha"],
            "model_cam_sha": identity["model_cam_sha"],
            "optimizer_sha": identity["optimizer_sha"],
            "split_sha": identity["split_sha"],
            "pair_sha": identity["pair_sha"],
            "pairs": int(state["train_pairs_np"].shape[0]),
            "historical_best_epoch": task["best_epoch_1based"],
            "historical_epochs_executed": task["epochs_executed"],
            "overlap_1based": identity["membership"]["train_test_overlap_1based"],
        })
        del state
        torch.cuda.empty_cache()

    state, warmup, identity, hist_task, hist_ckpt = prepare_fold(1)
    runtime = history_runtime()
    hist = load_base().load_history()
    branch_train_ids, ids_a, ids_b = split_branch_train_ids(state["train_np"], state["test_np"], state["y"], 1)
    sites = read_sites().to(state["y"].device)
    brainprompt, spromptgl = prompt_vectors(state)
    modules, branch_optimizers = init_groups(state, brainprompt, spromptgl)
    text_sha = hashlib.sha256(brainprompt.contiguous().numpy().tobytes() + spromptgl.contiguous().numpy().tobytes()).hexdigest()
    initial_state = hist.state_bundle(state)
    initial_rng = runtime.pack_rng()

    # Reference base-only update from the exact E61 epoch-10 state.
    hist.zero_optimizers(state)
    base_output, base_cam = hook_forward(state, training=True)
    base_output["total"].backward()
    reference_grad = _base_grad_snapshot(state)
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    reference_bundle = hist.state_bundle(state)
    reference_rng = runtime.pack_rng()

    # Replay the same base update while both independent branches compute and learn.
    hist.load_bundle(state, initial_state)
    runtime.restore_rng(initial_rng)
    modules, branch_optimizers = init_groups(state, brainprompt, spromptgl)
    hist.zero_optimizers(state)
    for optimizer in branch_optimizers:
        optimizer.zero_grad(set_to_none=True)
    output, cam = hook_forward(state, training=True)
    output["total"].backward()
    base_grad_before = _base_grad_snapshot(state)
    memory = _build_memory_for_preflight(state, branch_train_ids, sites, cam, output["logp"])
    for module, optimizer, no_boundary in zip(modules, branch_optimizers, (False, True)):
        branch_loss, branch_components, _, diagnostics, _, _ = train_branch(
            module, cam, memory, state["y"], output["logp"], branch_train_ids, ids_a, ids_b,
            sites, no_boundary)
        if not torch.isfinite(branch_loss):
            raise RuntimeError("non-finite preflight branch loss")
        branch_loss.backward()
        if not _finite_gradients(module):
            raise RuntimeError("non-finite branch gradient")
        optimizer.step()
    if not _same_gradients(base_grad_before, state):
        raise RuntimeError("branch backward changed an E74.4-G2 backbone gradient")
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    observed_bundle = hist.state_bundle(state)
    for module_name in ("model", "cam"):
        for name, value in reference_bundle[module_name].items():
            if not torch.equal(value, observed_bundle[module_name][name]):
                raise RuntimeError(f"base one-step update differs from E74.4-G2: {module_name}.{name}")
    for key in ("opt_base", "opt_cam", "opt_added"):
        if hist.sha_object(reference_bundle[key]) != hist.sha_object(observed_bundle[key]):
            raise RuntimeError(f"base optimizer update differs after branch training: {key}")
    if runtime.rng_sha(runtime.pack_rng()) != runtime.rng_sha(reference_rng):
        raise RuntimeError("base/shared RNG changed in the one-step branch parity check")

    # Recheck zero-residual identity and nonzero semantic/attention learning paths.
    runtime.restore_rng(initial_rng)
    hist.load_bundle(state, initial_state)
    modules, branch_optimizers = init_groups(state, brainprompt, spromptgl)
    zero_eval, zero_cam = hook_forward(state, training=False)
    memory = make_memory(branch_train_ids, state["y"], zero_cam, zero_eval["logp"], sites)
    train_idx = torch.as_tensor(branch_train_ids, dtype=torch.long, device=state["y"].device)
    support = torch.as_tensor(branch_train_ids, dtype=torch.long, device=state["y"].device)
    base_score = zero_eval["logp"][train_idx, 0] - zero_eval["logp"][train_idx, 1]
    for module in modules:
        test_idx = torch.as_tensor(state["test_np"], dtype=torch.long, device=state["y"].device)
        score, diag = module(zero_cam[test_idx].detach(), memory, support, sites,
                             zero_eval["logp"][test_idx, 0] - zero_eval["logp"][test_idx, 1])
        from bcspc import final_log_probabilities
        zero_logp = final_log_probabilities(zero_eval["logp"][test_idx],
                                             zero_eval["logp"][test_idx, 0] - zero_eval["logp"][test_idx, 1],
                                             score, module)
        if not torch.equal(diag["delta"], torch.zeros_like(diag["delta"])) or not torch.equal(zero_logp, zero_eval["logp"][test_idx]):
            raise RuntimeError("initial zero residual does not exactly preserve original classifier outputs")
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01)
        train_loss, _, _, _, _, _ = train_branch(module, zero_cam, memory, state["y"], zero_eval["logp"],
                                                   branch_train_ids, ids_a, ids_b, sites, False)
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        semantic_grad = float(module.semantic_projection[0].weight.grad.norm())
        attention_grad = float(module.reliable_text_projection.weight.grad.norm()) + float(module.boundary_text_projection.weight.grad.norm())
        modality_grad = float(module.modality_gate[0].weight.grad.norm())
        if min(semantic_grad, attention_grad, modality_grad) <= 0:
            raise RuntimeError("semantic/prototype attention/modality parameters are disconnected from branch loss")
    violations, cap_checked, cap_fallbacks = _site_cap_preflight(
        modules[0], memory, support, sites,
        zero_cam[torch.as_tensor(state["test_np"], device=state["y"].device)].reshape(-1, 4, 36))
    if violations:
        raise RuntimeError(f"site-diverse selector violated the strict cap: {violations}")
    if cap_checked <= 0:
        raise RuntimeError("site-diversity cap did not inspect any selected support")
    smoke_metrics = extended_metrics(zero_eval["logp"], state["test_np"], state["y"])
    payload_smoke = _capture_group_best(
        GROUPS[0], state, modules[0], branch_optimizers[0], None, ids_a, ids_b,
        branch_train_ids, 10, smoke_metrics, {}, {}, zero_eval, hist_task,
        {"best_epoch": -1, "best_acc": -1.0}, False)
    if payload_smoke["base_state"] is None or not np.array_equal(
            payload_smoke["base_prediction"], smoke_metrics["pred"]):
        raise RuntimeError("best-checkpoint payload construction failed in preflight")
    if before_sources != source_snapshot():
        raise RuntimeError("read-only history source/checkpoint SHA changed during preflight")

    result = {
        "status": "PASS",
        "base": CFG["base"]["name"],
        "historical_status": base_manifest["status"],
        "base_audit": base_audit["status"],
        "fold_inheritance": inherited,
        "warmup_and_checkpoint_inventory": "PASS 10/10",
        "base_one_step_update_parity_fold1": True,
        "base_optimizer_update_parity_fold1": True,
        "branch_to_backbone_gradient_isolation": True,
        "initial_delta_exactly_zero_and_outputs_equal": True,
        "semantic_attention_modality_gradients": {
            "semantic_projection": semantic_grad,
            "role_attention": attention_grad,
            "modality_gate": modality_grad,
        },
        "site_cap": {"violations": violations, "selected_supports_checked": cap_checked,
                      "per_site_cap": 2, "explicit_fallback_count": cap_fallbacks},
        "branch_training_support": {"eligible": len(branch_train_ids), "A": len(ids_a), "B": len(ids_b)},
        "brainprompt_strings_verified": list(__import__("bcspc").PROMPTS_ASD_TD),
        "brainprompt_text_cache_sha256": sha_file(E76_5 / ".runtime/cache/brainprompt_rn50.pt"),
        "original_spromptgl_text_cache_sha256": sha_file(E72 / ".runtime/text/text_embeddings.pt"),
        "frozen_prompt_vector_sha256": text_sha,
        "historical_cam_class": type(state["cam"]).__name__,
        "historical_cam_has_original_prompt_buffer": hasattr(state["cam"], "prompt_text_features"),
        "prompt_note": "E74.4-G2 restores E51/E53 PLNNoPEN; the historical base has no original PLN prompt buffer. The verified original eight text embeddings are supplied only to the new semantic branch, preserving the audited base exactly.",
        "rng_after_shared_step_matches_base": True,
        "base_source_hashes": before_sources,
        "device": torch.cuda.get_device_name(0),
        "timestamp": time.time(),
    }
    atomic_json(RESULTS / "preflight.json", result)
    print(f"PREFLIGHT PASS device={result['device']} base-branch parity=True", flush=True)
    return result


def _metric_payload(metrics):
    return {key: float(value) for key, value in metrics.items() if key not in ("pred", "prob")}


def _probability_predictions(metrics):
    import numpy as np
    return np.asarray(metrics["prob"], dtype=float), np.asarray(metrics["pred"], dtype=int)


def _capture_group_best(group, state, module, optimizer, memory_used, ids_a, ids_b, eligible,
                        epoch, metrics, diagnostics, branch_components, base_output, history_task,
                        base_ledger, no_boundary):
    import numpy as np
    import torch
    base = load_base()
    logp = base_output["logp"].detach()
    base_metrics = history_runtime().classification_metrics(logp, state["test_np"], state["y"])
    probability, prediction = _probability_predictions(metrics)
    return {
        "group": group,
        "epoch": epoch,
        "metrics": _metric_payload(metrics),
        "probability": probability,
        "prediction": prediction,
        "base_probability": np.asarray(base_metrics["prob"], dtype=float),
        "base_prediction": np.asarray(base_metrics["pred"], dtype=int),
        "base_state": base.load_history().state_bundle(state),
        "module": {key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "memory_used": memory_snapshot(memory_used),
        "support_a": list(ids_a), "support_b": list(ids_b), "eligible_train_ids": list(eligible),
        "evidence": {key: value.detach().cpu() if torch.is_tensor(value) else value
                     for key, value in diagnostics.items()},
        "branch_components": {key: float(value.detach().cpu()) if torch.is_tensor(value) and value.numel() == 1 else value
                              for key, value in branch_components.items()},
        "no_boundary": no_boundary,
        "base_ledger_best_epoch": base_ledger["best_epoch"],
        "base_ledger_best_acc": base_ledger["best_acc"],
        "warmup_sha256": history_task["identity"]["warmup_sha256"],
    }


def _save_best(group, fold, payload, source_hash, identity, trace, epochs_executed):
    record = {"experiment": CFG["experiment"], "group": group, "fold": fold, "seed": 0,
              "base_group": BASE_GROUP, "source_sha256": source_hash, "identity": identity,
              "best_epoch_1based": payload["epoch"], "epochs_executed": epochs_executed,
              "metrics": payload["metrics"], "best": payload, "rng_trace": trace}
    atomic_torch(CKPTS / f"{group}_f{fold:02d}.best.pt", record)
    return record


def _task_path(fold):
    return TASKS / f"BCSPC_shared_f{fold:02d}.json"


def _latest_path(fold):
    return CKPTS / f"BCSPC_shared_f{fold:02d}.latest.pt"


def _append_log(fold, record):
    path = LOGS / f"BCSPC_shared_f{fold:02d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _state_for_checkpoint(hist, state, bundle):
    hist.load_bundle(state, bundle)


def train_fold(fold):
    import numpy as np
    import torch
    from bcspc import branch_losses, init_isolated
    preflight_record = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    source_hashes = source_snapshot()
    if preflight_record.get("status") != "PASS" or preflight_record.get("base_source_hashes") != source_hashes:
        raise RuntimeError("preflight source hashes changed before formal training")
    state, warmup, identity, historical_task, historical_ckpt_path = prepare_fold(fold)
    hist, runtime = load_base().load_history(), history_runtime()
    if runtime.rng_sha(runtime.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold} did not start from exact E61 epoch-10 RNG")
    y = state["y"].long()
    train_ids = list(map(int, state["train_np"]))
    test_ids = list(map(int, state["test_np"]))
    eligible, ids_a, ids_b = split_branch_train_ids(train_ids, test_ids, y, fold)
    sites = read_sites().to(y.device)
    brainprompt, spromptgl = prompt_vectors(state)
    modules, branch_optimizers = init_groups(state, brainprompt, spromptgl)
    best_path = {group: CKPTS / f"{group}_f{fold:02d}.best.pt" for group in GROUPS}
    task_path = _task_path(fold)
    if task_path.exists():
        existing = json.loads(task_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE" and existing.get("source_sha256") == sha_obj(source_hashes):
            return existing
        if existing.get("status") != "RUNNING":
            raise RuntimeError(f"refusing to overwrite incompatible fold task: {task_path}")

    memory = None
    start_epoch, base_wait = 10, 0
    base_ledger = {"best_acc": -1.0, "best_epoch": -1, "best": None}
    ledgers = {group: {"best_acc": -1.0, "best_epoch": -1, "best": None} for group in GROUPS}
    trace, gradient_audit, mechanism_accum = [], [], []
    source_hash = sha_obj(source_hashes)
    latest_path = _latest_path(fold)
    recovered_from_source_hash = None
    if latest_path.exists():
        saved = torch_load(latest_path)
        compatible_sources = set(CFG["audit"].get("compatible_recovery_source_sha256", []))
        if saved.get("source_sha256") != source_hash and saved.get("source_sha256") not in compatible_sources:
            raise RuntimeError("latest checkpoint source hash is not current or explicitly recovery-compatible")
        recovered_from_source_hash = saved.get("source_sha256")
        checks = {"fold": fold, "identity": identity, "warmup_sha256": identity["warmup_sha256"]}
        for name, expected in checks.items():
            if saved.get(name) != expected:
                raise RuntimeError(f"incompatible recovery state {name} in fold {fold}")
        hist.load_bundle(state, saved["base_state"])
        runtime.restore_rng(saved["rng"])
        for index, group in enumerate(GROUPS):
            modules[index].load_state_dict(saved["modules"][group])
            branch_optimizers[index].load_state_dict(saved["branch_optimizers"][group])
            ledgers[group] = saved["ledgers"][group]
        memory = move_memory(saved["memory"], y.device)
        base_ledger = saved["base_ledger"]
        base_wait, start_epoch, trace = saved["base_wait"], saved["next_epoch"], saved["trace"]
        gradient_audit, mechanism_accum = saved.get("gradient_audit", []), saved.get("mechanism_accum", [])
    elif task_path.exists():
        task_path.unlink()
    if start_epoch < 10 or start_epoch > 450:
        raise RuntimeError(f"invalid resume epoch {start_epoch}")

    source_base_task = json.loads((BASE_EXP / f".runtime/tasks/{BASE_GROUP}_f{fold:02d}.json").read_text(encoding="utf-8"))
    if identity != source_base_task.get("identity"):
        raise RuntimeError("fold initialization identity differs from historical E74.4-G2")
    groups_meta = {group: {"group": group, "status": "RUNNING", "fold": fold} for group in GROUPS}
    atomic_json(task_path, {"status": "RUNNING", "fold": fold, "source_sha256": source_hash,
                            "identity": identity, "groups": groups_meta, "next_epoch": start_epoch})
    started = time.time()
    last_epoch = start_epoch
    hist.zero_optimizers(state)
    epoch_iterator = range(start_epoch, 450) if base_wait <= 200 else ()
    for epoch in epoch_iterator:
        last_epoch = epoch + 1
        rng_before = runtime.rng_sha(runtime.pack_rng())
        hist.zero_optimizers(state)
        for optimizer in branch_optimizers:
            optimizer.zero_grad(set_to_none=True)
        out, train_cam = hook_forward(state, training=True)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"non-finite E74.4-G2 loss fold={fold}, epoch={epoch+1}")
        out["total"].backward()
        backbone_grad_before = _base_grad_snapshot(state)
        branch_loss_values, branch_diagnostics, branch_components = {}, {}, {}
        active_branch = memory is not None
        if active_branch:
            for index, group in enumerate(GROUPS):
                branch_loss, components, scores, diagnostics, base_scores, labels = train_branch(
                    modules[index], train_cam, memory, y, out["logp"], eligible, ids_a, ids_b,
                    sites, no_boundary=(index == 1))
                branch_loss_values[group] = branch_loss
                branch_diagnostics[group] = diagnostics
                branch_components[group] = components
                branch_loss.backward()
                if not _finite_gradients(modules[index]):
                    raise FloatingPointError(f"non-finite BCSPC gradient group={group}, fold={fold}, epoch={epoch+1}")
            if not _same_gradients(backbone_grad_before, state):
                raise RuntimeError("BCSPC branch gradient contaminated an E74.4-G2 backbone gradient")
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        if active_branch:
            for optimizer in branch_optimizers:
                optimizer.step()
        rng_after_train = runtime.rng_sha(runtime.pack_rng())

        with torch.no_grad():
            evaluated, eval_cam = hook_forward(state, training=False)
            memory_used = memory_snapshot(memory)
            base_metrics = extended_metrics(evaluated["logp"], test_ids, y)
            method_results = {}
            for index, group in enumerate(GROUPS):
                if memory is None:
                    method_results[group] = (evaluated["logp"], base_metrics, None)
                else:
                    method_results[group] = eval_branch(
                        modules[index], evaluated["logp"], eval_cam, test_ids, eligible,
                        memory, y, sites, no_boundary=(index == 1))
            memory = make_memory(eligible, y, eval_cam, evaluated["logp"], sites, memory)
        rng_after_eval = runtime.rng_sha(runtime.pack_rng())
        rng_record = {"epoch": epoch + 1, "rng_before": rng_before,
                      "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval}
        trace.append(rng_record)

        if float(base_metrics["acc"]) > base_ledger["best_acc"]:
            base_ledger = {"best_acc": float(base_metrics["acc"]), "best_epoch": epoch + 1,
                           "best": hist.state_bundle(state),
                           "metrics": _metric_payload(base_metrics),
                           "probability": np.asarray(base_metrics["prob"], dtype=float),
                           "prediction": np.asarray(base_metrics["pred"], dtype=int)}
            base_wait = 0
        else:
            base_wait += 1

        for index, group in enumerate(GROUPS):
            final_logp, metrics, diagnostics = method_results[group]
            if float(metrics["acc"]) > ledgers[group]["best_acc"]:
                payload = _capture_group_best(
                    group, state, modules[index], branch_optimizers[index], memory_used,
                    ids_a, ids_b, eligible, epoch + 1, metrics,
                    diagnostics if diagnostics is not None else {},
                    branch_components.get(group, {}), evaluated, historical_task,
                    {"best_epoch": base_ledger["best_epoch"], "best_acc": base_ledger["best_acc"]},
                    index == 1)
                ledgers[group] = {"best_acc": float(metrics["acc"]), "best_epoch": epoch + 1,
                                  "best": payload}
                _save_best(group, fold, payload, source_hash, identity, trace, epoch + 1)

        if active_branch and epoch + 1 in (12, 13, 20, 100, 300, 449):
            for index, group in enumerate(GROUPS):
                module = modules[index]
                diagnostics = branch_diagnostics[group]
                gradient_audit.append({
                    "group": group, "epoch": epoch + 1,
                    "semantic_projection": _module_grad_norm(module, "semantic_projection.0.weight"),
                    "reliable_text_projection": _module_grad_norm(module, "reliable_text_projection.weight"),
                    "boundary_text_projection": _module_grad_norm(module, "boundary_text_projection.weight"),
                    "modality_gate": _module_grad_norm(module, "modality_gate.0.weight"),
                    "residual_gate": _module_grad_norm(module, "residual_gate.0.weight"),
                    "alpha": _module_grad_norm(module, "alpha"),
                    "base_cam_gradient_from_branch": 0.0,
                    "branch_loss": float(branch_loss_values[group].detach().cpu()),
                })

        for index, group in enumerate(GROUPS):
            _, metrics, diagnostics = method_results[group]
            if diagnostics is not None:
                mechanism_accum.append({
                    "group": group, "epoch": epoch + 1,
                    "gate_mean": float(diagnostics["gate"].mean().cpu()),
                    "gate_std": float(diagnostics["gate"].std(unbiased=False).cpu()),
                    "delta_abs_mean": float(diagnostics["delta"].abs().mean().cpu()),
                    "delta_abs_max": float(diagnostics["delta"].abs().max().cpu()),
                    "boundary_evidence_mean": float(diagnostics["final_boundary_evidence"].mean().cpu()),
                    "coverage_mean": float(diagnostics["coverage_factor"].mean().cpu()),
                    "attention_reliable_mean": float(diagnostics["attention"][..., 0].mean().cpu()),
                    "attention_boundary_mean": float(diagnostics["attention"][..., 1].mean().cpu()),
                    "modality_weights": diagnostics["modality_weights"].mean(0).cpu().tolist(),
                    "retrieval_fallbacks": int(diagnostics["retrieval_fallbacks"]),
                    "invalid_confusion_candidates": int(diagnostics["invalid_confusion_candidates"]),
                    "site_cap_violations": int(diagnostics["site_cap_violations"]),
                    "acc": float(metrics["acc"]),
                })

        if (epoch + 1) % 5 == 0 or base_wait > 200 or epoch == 449:
            latest_payload = {
                "fold": fold, "source_sha256": source_hash, "identity": identity,
                "warmup_sha256": identity["warmup_sha256"],
                "base_state": hist.state_bundle(state),
                "modules": {group: modules[index].state_dict() for index, group in enumerate(GROUPS)},
                "branch_optimizers": {group: branch_optimizers[index].state_dict() for index, group in enumerate(GROUPS)},
                "memory": memory, "rng": runtime.pack_rng(), "next_epoch": epoch + 1,
                "base_wait": base_wait, "base_ledger": base_ledger,
                "ledgers": ledgers, "trace": trace,
                "gradient_audit": gradient_audit, "mechanism_accum": mechanism_accum,
                "base_training_task": historical_task,
            }
            atomic_torch(latest_path, latest_payload)
            atomic_json(task_path, {"status": "RUNNING", "fold": fold, "source_sha256": source_hash,
                                    "identity": identity, "next_epoch": epoch + 1,
                                    "base_wait": base_wait,
                                    "base_best_epoch": base_ledger["best_epoch"],
                                    "method_best_epochs": {group: ledgers[group]["best_epoch"] for group in GROUPS},
                                    "rng_records": len(trace)})
            summary = {
                "fold": fold, "epoch": epoch + 1, "base_acc": float(base_metrics["acc"]),
                "base_best_acc": base_ledger["best_acc"], "base_best_epoch": base_ledger["best_epoch"],
                "base_wait": base_wait, "g1_best_acc": ledgers[GROUPS[0]]["best_acc"],
                "g1_best_epoch": ledgers[GROUPS[0]]["best_epoch"],
                "g2_best_acc": ledgers[GROUPS[1]]["best_acc"],
                "g2_best_epoch": ledgers[GROUPS[1]]["best_epoch"],
                "branch_active": active_branch, "memory_updates": memory["updates"],
                "memory_observed": int(memory["observed"].sum().item()), "rng": rng_record,
                "base_loss": float(out["total"].detach().cpu()),
                "branch_loss": {group: float(branch_loss_values[group].detach().cpu()) for group in branch_loss_values},
                "alpha": {group: float(modules[index].alpha.detach().cpu()) for index, group in enumerate(GROUPS)},
            }
            _append_log(fold, summary)
            print(f"fold={fold} epoch={epoch+1} base={base_ledger['best_acc']:.6f}@{base_ledger['best_epoch']} "
                  f"g1={ledgers[GROUPS[0]]['best_acc']:.6f}@{ledgers[GROUPS[0]]['best_epoch']} "
                  f"g2={ledgers[GROUPS[1]]['best_acc']:.6f}@{ledgers[GROUPS[1]]['best_epoch']} "
                  f"wait={base_wait}", flush=True)
        del out, evaluated, eval_cam
        if base_wait > 200:
            break

    if any(ledgers[group]["best"] is None for group in GROUPS) or base_ledger["best"] is None:
        raise RuntimeError(f"no method/base checkpoint selected in fold {fold}")
    historical_trace = {int(item["epoch"]): item for item in historical_task["rng_trace"]}
    current_trace = {int(item["epoch"]): item for item in trace}
    if len(current_trace) != len(trace) or any(current_trace.get(epoch) != item for epoch, item in historical_trace.items() if epoch in current_trace):
        raise RuntimeError(f"shared base RNG trace differs from E74.4-G2 during fold {fold}")
    if base_ledger["best_epoch"] != historical_task["best_epoch_1based"] or last_epoch != historical_task["epochs_executed"]:
        raise RuntimeError(f"shared base ACC best/early-stop trajectory differs from E74.4-G2 fold {fold}")
    history_payload = torch_load(historical_ckpt_path)
    for key in ("model", "cam"):
        for name, tensor in history_payload["best_state"][key].items():
            if not torch.equal(tensor.cpu(), base_ledger["best"][key][name].cpu()):
                raise RuntimeError(f"shared base best parameter mismatch with E74.4-G2: {key}.{name}, fold={fold}")
    if hist.sha_object(history_payload["best_state"]["opt_base"]) != hist.sha_object(base_ledger["best"]["opt_base"]):
        raise RuntimeError(f"shared base optimizer state mismatch with E74.4-G2, fold={fold}")
    if hist.sha_object(history_payload["best_state"]["opt_cam"]) != hist.sha_object(base_ledger["best"]["opt_cam"]):
        raise RuntimeError(f"shared CAM optimizer state mismatch with E74.4-G2, fold={fold}")
    if hist.sha_object(history_payload["best_state"]["opt_added"]) != hist.sha_object(base_ledger["best"]["opt_added"]):
        raise RuntimeError(f"shared DARG optimizer state mismatch with E74.4-G2, fold={fold}")

    base_checkpoint = {
        "experiment": CFG["experiment"], "group": "E76.6-Shared-E74.4-G2-Base",
        "fold": fold, "seed": 0, "source_sha256": source_hash,
        "identity": identity, "warmup_sha256": identity["warmup_sha256"],
        "best_epoch_1based": base_ledger["best_epoch"],
        "epochs_executed": last_epoch, "metrics": base_ledger["metrics"],
        "probability": base_ledger["probability"],
        "prediction": base_ledger["prediction"],
        "best_state": base_ledger["best"],
        "rng_trace": trace,
    }
    atomic_torch(CKPTS / f"BCSPC_base_f{fold:02d}.best.pt", base_checkpoint)
    base_checkpoint_sha = sha_file(CKPTS / f"BCSPC_base_f{fold:02d}.best.pt")

    group_tasks = {}
    for group in GROUPS:
        payload = ledgers[group]["best"]
        record = _save_best(group, fold, payload, source_hash, identity, trace, last_epoch)
        group_task = {
            "status": "COMPLETE", "group": group, "fold": fold, "seed": 0,
            "source_sha256": source_hash, "checkpoint_sha256": sha_file(best_path[group]),
            "identity": identity, "warmup_sha256": identity["warmup_sha256"],
            "best_epoch_1based": ledgers[group]["best_epoch"], "epochs_executed": last_epoch,
            "metrics": payload["metrics"], "method_base_epoch": payload["epoch"],
            "parameter_count": sum(parameter.numel() for parameter in modules[GROUPS.index(group)].parameters()),
            "best_gate_mean": float(payload["evidence"]["gate"].mean()) if payload["evidence"] else 0.0,
            "best_residual_mean_abs": float(payload["evidence"]["delta"].abs().mean()) if payload["evidence"] else 0.0,
            "main_rng_matches_history": True,
        }
        atomic_json(TASKS / f"{group}_f{fold:02d}.json", group_task)
        group_tasks[group] = group_task
    task = {
        "status": "COMPLETE", "fold": fold, "source_sha256": source_hash, "identity": identity,
        "warmup_sha256": identity["warmup_sha256"], "groups": group_tasks,
        "base_best_epoch_1based": base_ledger["best_epoch"],
        "base_best_metrics": base_ledger["metrics"], "historical_base_best_epoch_1based": historical_task["best_epoch_1based"],
        "base_checkpoint_sha256": base_checkpoint_sha,
        "epochs_executed": last_epoch, "shared_rng_records": len(trace),
        "shared_rng_exact_records": len(trace), "shared_rng_mismatches": 0,
        "base_parameter_state_matches_history": True,
        "base_optimizer_states_match_history": True,
        "A": ids_a, "B": ids_b, "branch_train_ids": eligible,
        "gradient_audit": gradient_audit, "mechanism_accum": mechanism_accum,
        "wall_seconds": time.time() - started,
        "recovered_from_source_sha256": recovered_from_source_hash,
    }
    atomic_json(task_path, task)
    return task


def best_path(group, fold):
    return CKPTS / f"{group}_f{fold:02d}.best.pt"


def run_all():
    completed = []
    for fold in range(1, 11):
        existing = _task_path(fold)
        if existing.is_file():
            task = json.loads(existing.read_text(encoding="utf-8"))
            if task.get("status") == "COMPLETE" and task.get("source_sha256") == sha_obj(source_snapshot()):
                completed.append(fold)
                continue
        completed.append(train_fold(fold)["fold"])
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.worker:
        train_fold(args.worker)
    elif args.all:
        completed = run_all()
        print(f"TRAINING COMPLETE {len(completed)}/10 folds", flush=True)
    else:
        parser.error("use --preflight, --worker FOLD, or --all")


if __name__ == "__main__":
    main()
