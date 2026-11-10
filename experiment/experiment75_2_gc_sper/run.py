#!/usr/bin/env python3
"""E74.4-based formal runner for Experiment 75.2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
RT = EXP / ".runtime"
TASKS, CKPTS, LATEST, LOGS, RESULTS = (RT / "tasks", RT / "checkpoints", RT / "latest", RT / "logs", EXP / "results")
BASE_DIR = Path(CFG["base"]["path"])
BASE_GROUP = CFG["base"]["name"]
E75_DIR = Path(CFG["references"]["e75_1_path"])
E60_DIR = Path(CFG["references"]["e60_path"])
TEXT_CACHE = E60_DIR / ".runtime/text_embeddings.pt"
HANDOFF = Path("/model/spromptgl/experiment/handoff.md")
GROUPS = tuple(CFG["method"]["groups"])
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
BASE, API, GCS = None, None, None


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_object(value):
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def sha_state(module):
    digest = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        digest.update(key.encode() + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_torch(path, payload):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def load_base():
    global BASE
    if BASE is None:
        spec = importlib.util.spec_from_file_location("e744_locked_runner_for_gc_sper", BASE_DIR / "run.py")
        BASE = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = BASE
        spec.loader.exec_module(BASE)
    return BASE


def load_gc():
    global GCS
    if GCS is None:
        spec = importlib.util.spec_from_file_location("gc_sper_752_isolated", EXP / "gc_sper.py")
        GCS = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = GCS
        spec.loader.exec_module(GCS)
    return GCS


def load_text_cache():
    global API
    if API is None:
        spec = importlib.util.spec_from_file_location("sa_dcer_751_text_loader", E75_DIR / "run.py")
        API = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = API
        spec.loader.exec_module(API)
    embeddings = API.text_bank().detach().cpu().float()
    if tuple(embeddings.shape) != (8, 1024) or not torch_isfinite(embeddings):
        raise RuntimeError("frozen original SPromptGL-8 text cache invalid")
    return embeddings


def torch_isfinite(value):
    import torch
    return bool(torch.isfinite(value).all())


def source_inventory():
    base = load_base()
    e75_files = ["sa_dcer.py", "run.py", "audit.py", "config.json", "manifest.json",
                 "results/report.md", "results/summary.csv", "results/paired_comparisons.csv",
                 "results/mechanism.csv", "results/interventions.csv", "results/audit_status.json"]
    current = {
        "config.json": sha_file(EXP / "config.json"),
        "gc_sper.py": sha_file(EXP / "gc_sper.py"),
        "run.py": sha_file(EXP / "run.py"),
        "audit.py": sha_file(EXP / "audit.py"),
        "e74_4_manifest_sha256": sha_file(BASE_DIR / "manifest.json"),
        "e74_4_source_manifest_sha256": base.source_sha(),
        "e74_4_audit_status_sha256": sha_file(BASE_DIR / "results/audit_status.json"),
        "e74_4_fold_metrics_sha256": sha_file(BASE_DIR / "results/fold_metrics.csv"),
        "e74_4_checkpoint_audit_sha256": sha_file(BASE_DIR / "results/checkpoint_audit.csv"),
        "e60_text_cache_sha256": sha_file(TEXT_CACHE),
    }
    for relative in e75_files:
        path = E75_DIR / relative
        if not path.is_file():
            raise FileNotFoundError(f"required E75.1 reference missing: {path}")
        current[f"e75.1/{relative}"] = sha_file(path)
    return current


def source_digest():
    return hashlib.sha256(json.dumps(source_inventory(), sort_keys=True).encode()).hexdigest()


def rng_runtime():
    base = load_base()
    return base.load_history().load_e69().load_e61().load_base()


def prepare_verified(fold):
    import torch
    base = load_base()
    state, warmup, identity = base.prepare(fold)
    state["fold"] = fold
    state["group"] = BASE_GROUP
    state["rudag_identity"] = identity
    task_path = BASE_DIR / ".runtime/tasks" / f"{BASE_GROUP}_f{fold:02d}.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    audit_rows = list(csv.DictReader((BASE_DIR / "results/checkpoint_audit.csv").open(encoding="utf-8")))
    audit = next((row for row in audit_rows if row["group"] == BASE_GROUP and int(row["fold"]) == fold), None)
    checkpoint = BASE_DIR / ".runtime/checkpoints" / f"{BASE_GROUP}_f{fold:02d}.best.pt"
    if task.get("status") != "COMPLETE" or task.get("identity") != identity:
        raise RuntimeError(f"E74.4-G2 base task/prepare identity mismatch in fold {fold}")
    if not audit or audit.get("status") != "INDEPENDENT_AUDIT_PASS" or audit.get("checkpoint_sha256") != task.get("checkpoint_sha256"):
        raise RuntimeError(f"E74.4-G2 independent audit unavailable in fold {fold}")
    if not checkpoint.is_file() or sha_file(checkpoint) != task.get("checkpoint_sha256"):
        raise RuntimeError(f"E74.4-G2 checkpoint SHA mismatch in fold {fold}")
    expected_overlap = [377, 407] if fold == 2 else []
    if identity["membership"]["train_test_overlap_1based"] != expected_overlap:
        raise RuntimeError(f"E74.4-G2 fold membership mismatch in fold {fold}")
    if state["train_pairs_np"].shape != (4096, 2) or identity.get("train_pair_scope_pass") is not True:
        raise RuntimeError(f"E74.4-G2 pair count/scope mismatch in fold {fold}")
    runtime = rng_runtime()
    if runtime.rng_sha(runtime.pack_rng()) != warmup.get("rng_sha"):
        raise RuntimeError(f"fold {fold} did not restore the E61 epoch-10 RNG state")
    state["gc_sper_identity"] = identity
    state["base_task"] = task
    return state, warmup, identity, task


def group_module(text, group, fold, device):
    import torch
    gc = load_gc()
    anonymous = group == "G3-Anonymous"
    module = gc.GCSper(text, projection_seed=752100 + fold, init_seed=752000 + fold,
                       anonymous=anonymous, anonymous_seed=752900 + fold).to(device)
    return module


def serialize_bank(bank):
    return {key: getattr(bank, key).detach().cpu() if hasattr(getattr(bank, key), "detach") else getattr(bank, key)
            for key in ("full", "crossfit", "assignment", "source_indices", "class_counts", "partition_counts", "source_sha256", "bank_sha256")}


def deserialize_bank(value):
    gc = load_gc()
    return gc.PrototypeBank(**value)


def make_capture(state, module, bank, group, intervention="full", training=False):
    import torch
    import torch.nn.functional as F
    captured = {"cam_calls": 0, "outputlayer_calls": 0}
    cam = state["cam"]
    train_ids = torch.as_tensor(state["train"], dtype=torch.long, device=next(cam.parameters()).device)
    aux_ids = bank.source_indices.to(train_ids.device)
    labels = state["y"]
    def cam_pre(_mod, inputs):
        if len(inputs) != 1 or tuple(inputs[0].shape) != (871, 320):
            raise RuntimeError("E74.4 CAM input hook did not receive the expected F[871,320]")
        captured["f320"] = inputs[0]
    def output_pre(_mod, inputs):
        if not inputs or tuple(inputs[0].shape) != (871, 144):
            raise RuntimeError("E74.4 Outputlayer hook did not receive H[871,144]")
        h = inputs[0]
        result = module.evidence(h, captured["f320"], bank, train_ids, group, intervention)
        captured["raw_h"] = h
        captured["result"] = result
        captured["outputlayer_calls"] += 1
        if training:
            lp, la = module.auxiliary_losses(result, labels, aux_ids, group)
            captured["l_proto"], captured["l_align"] = lp, la
        return (result["h_prime"], *inputs[1:])
    h1 = cam.register_forward_pre_hook(cam_pre)
    h2 = cam.Outputlayer.register_forward_pre_hook(output_pre)
    try:
        out = load_base().forward(state, BASE_GROUP, training=training, collect=False)
    finally:
        h1.remove(); h2.remove()
    captured["cam_calls"] = 1
    if captured["outputlayer_calls"] != 1:
        raise RuntimeError(f"formal Outputlayer called {captured['outputlayer_calls']} times")
    if training:
        out["l_proto"] = captured["l_proto"]
        out["l_align"] = captured["l_align"]
        out["total"] = out["total"] + 0.01 * out["l_proto"] + (0.01 * out["l_align"] if group != "G1-Proto" else 0.0)
    else:
        out["l_proto"] = out["total"].new_zeros(())
        out["l_align"] = out["total"].new_zeros(())
    return out, captured


def initialize_bank(state, group, module):
    import torch
    base = load_base(); runtime = rng_runtime()
    rng_before = runtime.rng_sha(runtime.pack_rng())
    module.eval(); state["model"].eval(); state["cam"].eval()
    with torch.no_grad():
        out, capture = make_capture(state, module, _temporary_bank(state), group, "module_off", training=False)
    # The temporary bank is used only to capture raw CAM features; replace it with a real training-only bank.
    test = torch.as_tensor(state["test"], dtype=torch.long, device=state["y"].device)
    bank = load_gc().build_prototype_bank(
        capture["raw_h"].reshape(871, 16, 9), state["y"],
        torch.as_tensor(state["train"], dtype=torch.long, device=state["y"].device), test, 5
    )
    if rng_before != runtime.rng_sha(runtime.pack_rng()):
        raise RuntimeError("prototype initialization forward changed the inherited main RNG")
    return bank, capture


def _temporary_bank(state):
    import torch
    # Valid shape-only neutral bank; it cannot include any label-derived values.
    full = torch.zeros((2, 16, 9), device=state["y"].device, dtype=torch.float32)
    full[:, :, 0] = 1
    crossfit = full.unsqueeze(0).repeat(5, 1, 1, 1)
    assignment = torch.full((871,), -1, device=full.device, dtype=torch.long)
    empty = torch.empty((0,), device=full.device, dtype=torch.long)
    counts = torch.ones((2,), device=full.device, dtype=torch.long)
    parts = torch.ones((5, 2), device=full.device, dtype=torch.long)
    return load_gc().PrototypeBank(full, crossfit, assignment, empty, counts, parts, "temporary", "temporary")


def finite_module(module):
    import torch
    return all(torch.isfinite(p).all().item() for p in module.parameters())


def preflight():
    import numpy as np
    import torch
    for path in (TASKS, CKPTS, LATEST, LOGS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one PPU device; available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
    base_manifest = json.loads((BASE_DIR / "manifest.json").read_text(encoding="utf-8"))
    if (base_manifest.get("status"), base_manifest.get("completed_tasks"), base_manifest.get("independent_audits"),
        base_manifest.get("checkpoint_audit")) != ("COMPLETE_AUDITED", 30, 30, "PASS 30/30"):
        raise RuntimeError("E74.4 base is not audited 30/30")
    if base_manifest.get("source_sha256") != CFG["base"]["manifest_source_sha256"]:
        raise RuntimeError("E74.4 source manifest SHA differs from the fixed reference")
    e75_manifest = json.loads((E75_DIR / "manifest.json").read_text(encoding="utf-8"))
    if (e75_manifest.get("status"), e75_manifest.get("completed_tasks"), e75_manifest.get("independent_audits")) != ("COMPLETE_AUDITED", 30, 30):
        raise RuntimeError("E75.1 required historical run is not COMPLETE_AUDITED 30/30")
    text = load_text_cache()
    runtime = rng_runtime()
    warmup_checks, init_checks = [], []
    for fold in range(1, 11):
        state, warmup, identity, task = prepare_verified(fold)
        warmup_checks.append({"fold": fold, "warmup_sha256": identity["warmup_sha256"], "rng_sha": warmup["rng_sha"],
                              "checkpoint_sha256": task["checkpoint_sha256"], "train_pair_sha256": identity["pair_sha"]})
        device = next(state["cam"].parameters()).device
        rng_before = runtime.rng_sha(runtime.pack_rng())
        module_hashes = []
        for group in GROUPS:
            module = group_module(text, group, fold, device)
            module_hashes.append({"group": group, "trainable_state_sha256": sha_state(module),
                                  "semantic_sha256": sha_state(module)})
            del module
        if len({row["trainable_state_sha256"] for row in module_hashes}) != 1:
            # Buffers differ in G3, but the trainable parameters must be byte-identical.
            def trainable_hash(mod):
                digest = hashlib.sha256()
                for name, param in sorted(mod.named_parameters()):
                    digest.update(name.encode() + b"\0"); digest.update(param.detach().cpu().contiguous().numpy().tobytes())
                return digest.hexdigest()
            hashes = []
            for group in GROUPS:
                mod = group_module(text, group, fold, device); hashes.append(trainable_hash(mod)); del mod
            if len(set(hashes)) != 1:
                raise RuntimeError(f"new module trainable initialization differs across groups, fold {fold}")
        module = group_module(text, GROUPS[0], fold, device)
        # Compare historical base and explicit zero-residual path in eval mode.
        state["model"].eval(); state["cam"].eval(); module.eval()
        start_rng = runtime.pack_rng(); start_sha = runtime.rng_sha(start_rng)
        with torch.no_grad():
            plain = load_base().forward(state, BASE_GROUP, training=False)
            hooked, hook_capture = make_capture(state, module, _temporary_bank(state), GROUPS[0], "module_off", False)
        if not torch.equal(plain["logp"], hooked["logp"]) or not torch.equal(hook_capture["result"]["h_prime"], hook_capture["raw_h"]):
            raise RuntimeError("module_off does not exactly reproduce E74.4 Outputlayer inputs and predictions")
        bank = load_gc().build_prototype_bank(hook_capture["raw_h"].reshape(871, 16, 9), state["y"], state["train"], state["test"], 5)
        test_ids = torch.as_tensor(state["test"], dtype=torch.long, device=state["y"].device)
        labels_swapped = state["y"].clone()
        labels_swapped[test_ids] = 1 - labels_swapped[test_ids]
        bank_swapped = load_gc().build_prototype_bank(hook_capture["raw_h"].reshape(871, 16, 9), labels_swapped, state["train"], state["test"], 5)
        if bank.bank_sha256 != bank_swapped.bank_sha256:
            raise RuntimeError("test-label permutation changed the new prototype bank")
        if fold == 2:
            if bool(torch.isin(torch.tensor([376, 406], device=bank.source_indices.device), bank.source_indices).any()):
                raise RuntimeError("Fold 2 exposed rows entered the auxiliary prototype source")
        if group_module(text, "G2-GC-SPER", fold, device).semantic_embeddings.shape != (8, 128):
            raise RuntimeError("semantic embedding shape mismatch")
        g2 = group_module(text, "G2-GC-SPER", fold, device)
        g3 = group_module(text, "G3-Anonymous", fold, device)
        gram_error = float((g2.semantic_embeddings @ g2.semantic_embeddings.T - g3.semantic_embeddings @ g3.semantic_embeddings.T).abs().max())
        if gram_error >= 1e-5:
            raise RuntimeError(f"G2/G3 projected semantic Gram mismatch {gram_error}")
        if abs(float(g2.alpha.detach()) - 0.2) > 1e-6 or abs(float(g2.gamma.detach()) - 0.1) > 1e-6:
            raise RuntimeError("alpha/gamma initial values differ from locked configuration")
        if not torch.isfinite(hook_capture["result"]["h_prime"]).all() or not torch.isfinite(hook_capture["result"]["reliability"]).all():
            raise RuntimeError("GC-SPER outputs are non-finite")
        if float(hook_capture["result"]["reliability"].min()) < 0 or float(hook_capture["result"]["reliability"].max()) > 1:
            raise RuntimeError("reliability escaped [0,1]")
        if runtime.rng_sha(runtime.pack_rng()) != start_sha:
            raise RuntimeError("GC-SPER initialization or no-grad baseline preflight changed main RNG")
        init_checks.append({"fold": fold, "module_parameter_sha256": module_hashes[0]["trainable_state_sha256"],
                            "projection_sha256": sha_state(g2), "anonymous_gram_max_abs_error": gram_error,
                            "prototype_source_sha256": bank.source_sha256, "prototype_bank_sha256": bank.bank_sha256,
                            "prototype_source_count": int(bank.source_indices.numel()), "module_off_exact": True,
                            "main_rng_unchanged": True})
        del state, module, g2, g3, plain, hooked, hook_capture, bank, bank_swapped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # A single isolated gradient/update replay checks the full state bundle without running a pilot experiment.
    state, warmup, identity, _ = prepare_verified(1)
    device = next(state["cam"].parameters()).device
    module = group_module(text, "G2-GC-SPER", 1, device)
    bank, _ = initialize_bank(state, "G2-GC-SPER", module)
    opt_new = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=0.001, weight_decay=0.001)
    hist = load_base().load_history(); rt = rng_runtime()
    initial_bundle = hist.state_bundle(state)
    initial_module = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    initial_optimizer = pickle.loads(pickle.dumps(opt_new.state_dict(), protocol=4))
    initial_bank = serialize_bank(bank)
    initial_rng = rt.pack_rng()
    def max_difference(left, right):
        if torch.is_tensor(left) and torch.is_tensor(right):
            if left.shape != right.shape:
                return float("inf")
            if not left.numel():
                return 0.0
            return float((left.detach().cpu().double() - right.detach().cpu().double()).abs().max())
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                return float("inf")
            return max((max_difference(left[key], right[key]) for key in left), default=0.0)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                return float("inf")
            return max((max_difference(a, b) for a, b in zip(left, right)), default=0.0)
        try:
            return 0.0 if left == right else float("inf")
        except Exception:
            return float("inf")
    def one_step():
        hist.zero_optimizers(state); opt_new.zero_grad(set_to_none=True)
        out, capture = make_capture(state, module, bank, "G2-GC-SPER", training=True)
        logp_before_update = out["logp"].detach().cpu().clone()
        out["total"].backward()
        required = (module.graph_mlp[0].weight, module.scale_asd.weight, module.key.weight,
                    module.alpha_raw, module.gamma_raw)
        if any(p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0 for p in required):
            raise RuntimeError("G2 classification/auxiliary path has a missing or zero required gradient")
        if not torch.isfinite(out["total"]):
            raise RuntimeError("non-finite preflight loss")
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        opt_new.step()
        return {"base": hist.state_bundle(state), "module": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
                "optimizer": pickle.loads(pickle.dumps(opt_new.state_dict(), protocol=4)),
                "rng": rt.pack_rng(), "loss": float(out["total"].detach().cpu()), "logp": logp_before_update,
                "gradients": {"context": float(module.graph_mlp[0].weight.grad.norm().cpu()),
                              "film": float(module.scale_asd.weight.grad.norm().cpu()),
                              "key": float(module.key.weight.grad.norm().cpu()),
                              "alpha": float(module.alpha_raw.grad.abs().cpu()),
                              "gamma": float(module.gamma_raw.grad.abs().cpu())}}
    first = one_step()
    hist.load_bundle(state, initial_bundle); module.load_state_dict(initial_module); opt_new.load_state_dict(initial_optimizer)
    bank = deserialize_bank(initial_bank); rt.restore_rng(initial_rng)
    second = one_step()
    replay_checks = {"base_max_abs_diff": max_difference(first["base"], second["base"]),
                     "module_max_abs_diff": max_difference(first["module"], second["module"]),
                     "optimizer_max_abs_diff": max_difference(first["optimizer"], second["optimizer"]),
                     "output_max_abs_diff": max_difference(first["logp"], second["logp"]),
                     "rng_equal": rt.rng_sha(first["rng"]) == rt.rng_sha(second["rng"]),
                     "loss_first": first["loss"], "loss_second": second["loss"],
                     "loss_abs_diff": abs(first["loss"] - second["loss"])}
    if any(replay_checks[key] > 1e-6 for key in ("base_max_abs_diff", "module_max_abs_diff", "optimizer_max_abs_diff", "output_max_abs_diff")) or not replay_checks["rng_equal"] or replay_checks["loss_abs_diff"] > 1e-8:
        raise RuntimeError(f"full base+GC-SPER checkpoint/update/RNG replay did not match: {replay_checks}")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    result = {"status": "PASS", "device": torch.cuda.get_device_name(0), "device_count": 1,
              "gpu_memory_free_bytes_at_preflight": int(free_bytes), "gpu_memory_total_bytes": int(total_bytes),
              "base_status": base_manifest["status"], "base_completed": base_manifest["completed_tasks"],
              "base_audited": base_manifest["independent_audits"], "e75_1_status": e75_manifest["status"],
              "e75_1_completed": e75_manifest["completed_tasks"], "e75_1_audited": e75_manifest["independent_audits"],
              "text_cache_sha256": sha_file(TEXT_CACHE), "warmup_identity_checks": warmup_checks,
              "same_fold_trainable_init_checks": init_checks, "single_step_replay": {**replay_checks, "passed": True, "gradient_norms": first["gradients"]},
              "training_start_epoch_zero_based": 10, "groups": list(GROUPS), "expected_tasks": 30}
    atomic_json(RESULTS / "preflight.json", result)
    source_map = source_inventory()
    atomic_json(EXP / "manifest.json", {"experiment": CFG["experiment"], "status": "PREFLIGHT_PASS",
                 "experiment_root": str(EXP), "preferred_root": CFG["preferred_root"],
                 "root_fallback_reason": CFG["root_fallback_reason"], "config": CFG,
                 "source_inventory": source_map, "source_sha256": source_digest(),
                 "expected_tasks": 30, "completed_tasks": 0, "independent_audits": 0,
                 "preflight": result, "training_failures": [], "training_retries": [], "oom_events": 0})
    return result


def task_paths(group, fold):
    stem = f"{group}_f{fold:02d}"
    return TASKS / f"{stem}.json", CKPTS / f"{stem}.best.pt", LATEST / f"{stem}.latest.pt", LOGS / f"{stem}.jsonl"


def task_complete(group, fold, source_hash):
    task_path, checkpoint_path, _, _ = task_paths(group, fold)
    if not task_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
        return task.get("status") == "COMPLETE" and task.get("source_sha256") == source_hash and task.get("checkpoint_sha256") == sha_file(checkpoint_path)
    except Exception:
        return False


def _cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _checkpoint(group, fold, identity, base_task, source_hash, best, executed, trace, init_meta):
    return {"experiment": CFG["experiment"], "group": group, "fold": fold, "seed": 0,
            "source_sha256": source_hash, "identity": identity,
            "base_task_checkpoint_sha256": base_task["checkpoint_sha256"], "base_best_checkpoint_used_for_training": False,
            "best_epoch_1based": best["epoch"], "epochs_executed": executed,
            "base_state": best["base_state"], "module_state": best["module_state"],
            "base_optimizer_state_in_bundle": True, "module_optimizer_state": best["optimizer_state"],
            "prototype_bank": best["prototype_bank"], "prototype_metadata": best["prototype_metadata"],
            "test_prediction": best["prediction"], "test_probability": best["probability"],
            "metrics": best["metrics"], "losses": best["losses"],
            "best_rng_after_train": best["rng_after_train"], "best_rng_after_eval": best["rng_after_eval"],
            "rng_trace": trace, "init_meta": init_meta}


def train_worker(group, fold):
    import numpy as np
    import torch
    if group not in GROUPS or fold not in range(1, 11):
        raise ValueError(f"invalid task {group} fold={fold}")
    source_hash = source_digest()
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    manifest = json.loads((EXP / "manifest.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or manifest.get("source_sha256") != source_hash:
        raise RuntimeError("worker source SHA differs from the passing preflight")
    state, warmup, identity, base_task = prepare_verified(fold)
    state["fold"] = fold; state["group"] = group; state["rudag_identity"] = identity
    task_file, best_file, latest_file, log_file = task_paths(group, fold)
    if task_complete(group, fold, source_hash):
        return json.loads(task_file.read_text(encoding="utf-8"))
    if task_file.exists() and not latest_file.exists():
        raise RuntimeError(f"refusing to overwrite incompatible task record {task_file}")
    hist = load_base().load_history(); rt = rng_runtime(); gc = load_gc()
    device = next(state["cam"].parameters()).device
    text = load_text_cache()
    module = group_module(text, group, fold, device)
    init_sha = sha_state(module)
    start_rng_sha = rt.rng_sha(rt.pack_rng())
    bank, _ = initialize_bank(state, group, module)
    if rt.rng_sha(rt.pack_rng()) != start_rng_sha:
        raise RuntimeError("initial prototype bank changed inherited RNG")
    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=0.001, weight_decay=0.001)
    init_meta = {"parameter_sha256": init_sha, "projection_sha256": sha_state(module),
                 "text_cache_sha256": sha_file(TEXT_CACHE), "prototype_bank_sha256": bank.bank_sha256,
                 "prototype_source_sha256": bank.source_sha256, "anonymous_gram_error": float(module.gram_error.cpu()),
                 "main_rng_sha_before_module_init": start_rng_sha, "main_rng_sha_after_module_init": start_rng_sha,
                 "warmup_sha256": identity["warmup_sha256"]}
    start_epoch, wait_count = 10, 0
    best_acc, best_epoch, best, trace = -1.0, -1, None, []
    if latest_file.is_file():
        saved = torch_load(latest_file)
        for key, expected in (("group", group), ("fold", fold), ("source_sha256", source_hash), ("identity", identity)):
            if saved.get(key) != expected:
                raise RuntimeError(f"latest recovery mismatch for {key}")
        hist.load_bundle(state, saved["base_state"]); module.load_state_dict(saved["module_state"])
        optimizer.load_state_dict(saved["module_optimizer_state"]); bank = deserialize_bank(saved["next_prototype_bank"])
        rt.restore_rng(saved["rng"])
        start_epoch, wait_count = int(saved["next_epoch"]), int(saved["wait_count"])
        best_acc, best_epoch, best, trace = float(saved["best_acc"]), int(saved["best_epoch"]), saved["best"], saved["rng_trace"]
    started = time.time(); executed = start_epoch
    torch.cuda.reset_peak_memory_stats()
    aux_ids = bank.source_indices.to(device)
    for epoch in range(start_epoch, int(CFG["training"]["max_epochs"])):
        rng_before = rt.rng_sha(rt.pack_rng())
        hist.zero_optimizers(state); optimizer.zero_grad(set_to_none=True)
        out, cap = make_capture(state, module, bank, group, training=True)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"non-finite training loss in {group} fold {fold} epoch {epoch+1}")
        out["total"].backward()
        if not finite_module(module) or not load_base()._finite_grads(state["model"]) or not load_base()._finite_grads(state["cam"]):
            raise FloatingPointError("non-finite gradient in formal worker")
        grad_snapshot = {"graph_context": float(module.graph_mlp[0].weight.grad.norm().detach().cpu()) if module.graph_mlp[0].weight.grad is not None else 0.0,
                         "film_asd": float(module.scale_asd.weight.grad.norm().detach().cpu()) if module.scale_asd.weight.grad is not None else 0.0,
                         "key": float(module.key.weight.grad.norm().detach().cpu()) if module.key.weight.grad is not None else 0.0,
                         "alpha": float(module.alpha_raw.grad.abs().detach().cpu()) if module.alpha_raw.grad is not None else 0.0,
                         "gamma": float(module.gamma_raw.grad.abs().detach().cpu()) if module.gamma_raw.grad is not None else 0.0}
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        optimizer.step()
        rng_after_train = rt.rng_sha(rt.pack_rng())
        executed = epoch + 1
        with torch.no_grad():
            evaluated, eval_cap = make_capture(state, module, bank, group, training=False)
            metrics = rt.classification_metrics(evaluated["logp"], state["test"], state["y"])
        rng_after_eval = rt.rng_sha(rt.pack_rng())
        trace_item = {"epoch": executed, "rng_before_train": rng_before,
                      "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval}
        trace.append(trace_item)
        next_bank = gc.build_prototype_bank(eval_cap["raw_h"].detach().reshape(871, 16, 9), state["y"], state["train"], state["test"], 5)
        loss_values = {key: float(out[key].detach().cpu()) for key in ("cls", "graph", "pair", "adv", "preserve", "l_proto", "l_align", "total")}
        if float(metrics["acc"]) > best_acc:
            best_acc, best_epoch, wait_count = float(metrics["acc"]), epoch, 0
            best = {"epoch": epoch + 1, "base_state": hist.state_bundle(state), "module_state": _cpu_state(module),
                    "optimizer_state": optimizer.state_dict(), "prototype_bank": serialize_bank(bank),
                    "prototype_metadata": {"source_sha256": bank.source_sha256, "bank_sha256": bank.bank_sha256,
                                           "source_indices": bank.source_indices.detach().cpu(),
                                           "class_counts": bank.class_counts.detach().cpu(),
                                           "partition_counts": bank.partition_counts.detach().cpu(),
                                           "assignment": bank.assignment.detach().cpu()},
                    "metrics": {key: float(value) for key, value in metrics.items() if key not in ("pred", "prob")},
                    "probability": np.asarray(metrics["prob"], dtype=np.float64).copy(),
                    "prediction": np.asarray(metrics["pred"], dtype=np.int64).copy(),
                    "losses": loss_values, "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval,
                    "gamma": float(module.gamma.detach().cpu()), "alpha": float(module.alpha.detach().cpu()),
                    "residual_cam_ratio": float(eval_cap["result"]["residual"].norm() / eval_cap["raw_h"].norm().clamp_min(1e-12)),
                    "prototype_bank_next_sha256": next_bank.bank_sha256,
                    "reliability_mean": float(eval_cap["result"]["reliability"].mean().cpu()),
                    "semantic_agreement_mean": float(eval_cap["result"]["r_agree"].mean().cpu()),
                    "gradient_snapshot": grad_snapshot}
            atomic_torch(best_file, _checkpoint(group, fold, identity, base_task, source_hash, best, executed, trace, init_meta))
        else:
            wait_count += 1
        bank = next_bank
        if executed % 10 == 0 or executed == int(CFG["training"]["max_epochs"]) or wait_count > int(CFG["training"]["early_stop_wait_gt"]):
            latest = {"group": group, "fold": fold, "source_sha256": source_hash, "identity": identity,
                      "base_state": hist.state_bundle(state), "module_state": _cpu_state(module),
                      "module_optimizer_state": optimizer.state_dict(), "next_prototype_bank": serialize_bank(bank),
                      "rng": rt.pack_rng(), "next_epoch": executed, "wait_count": wait_count,
                      "best_acc": best_acc, "best_epoch": best_epoch, "best": best, "rng_trace": trace}
            atomic_torch(latest_file, latest)
            if executed % 10 == 0 or wait_count > int(CFG["training"]["early_stop_wait_gt"]):
                with log_file.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"group": group, "fold": fold, "epoch": executed,
                                "best_epoch": best_epoch + 1, "best_acc": best_acc, "losses": loss_values,
                                "alpha": float(module.alpha.detach().cpu()), "gamma": float(module.gamma.detach().cpu()),
                                "gradients": grad_snapshot, "rng": trace_item,
                                "prototype_bank_sha256": bank.bank_sha256,
                                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated())}, allow_nan=False) + "\n")
                    stream.flush(); os.fsync(stream.fileno())
                print(f"{group} fold={fold} epoch={executed} best={best_acc:.6f}@{best_epoch+1}", flush=True)
        del out, evaluated, cap, eval_cap
        if wait_count > int(CFG["training"]["early_stop_wait_gt"]):
            break
    if best is None:
        raise RuntimeError(f"no test-ACC-selected checkpoint for {group} fold {fold}")
    checkpoint = _checkpoint(group, fold, identity, base_task, source_hash, best, executed, trace, init_meta)
    atomic_torch(best_file, checkpoint)
    task = {"status": "COMPLETE", "group": group, "fold": fold, "seed": 0,
            "source_sha256": source_hash, "identity": identity, "checkpoint_sha256": sha_file(best_file),
            "base_task_checkpoint_sha256": base_task["checkpoint_sha256"], "best_epoch_1based": best_epoch + 1,
            "epochs_executed": executed, "metrics": best["metrics"], "rng_trace": trace,
            "wall_seconds": time.time() - started, "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "device": torch.cuda.get_device_name(0), "init_meta": init_meta, "losses": best["losses"]}
    atomic_json(task_file, task)
    return task


def run_training(workers=10):
    from multiprocessing import get_context
    workers = max(1, min(10, int(workers)))
    source_hash = source_digest()
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or json.loads((EXP / "manifest.json").read_text())["source_sha256"] != source_hash:
        raise RuntimeError("training source/config changed after preflight")
    jobs = [(group, fold) for fold in range(1, 11) for group in GROUPS]
    pending = [(group, fold) for group, fold in jobs if not task_complete(group, fold, source_hash)]
    manifest = json.loads((EXP / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({"status": "TRAINING", "training_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "worker_limit": workers, "completed_tasks": sum(task_complete(g, f, source_hash) for g, f in jobs)})
    atomic_json(EXP / "manifest.json", manifest)
    started = time.time(); retries, failures, peak_workers, oom = [], [], 0, 0
    # ProcessPoolExecutor spawn keeps every task's model, optimizers and RNG private.
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        future_map = {pool.submit(train_worker, g, f): (g, f, 1) for g, f in pending}
        peak_workers = min(workers, len(future_map))
        while future_map:
            for future in as_completed(tuple(future_map)):
                g, f, attempt = future_map.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    record = {"group": g, "fold": f, "attempt": attempt, "error": repr(exc),
                              "traceback": "".join(traceback.format_exception(exc))[-5000:],
                              "oom": "out of memory" in repr(exc).lower()}
                    if record["oom"]:
                        oom += 1
                    if attempt < 2:
                        retries.append(record); future_map[pool.submit(train_worker, g, f)] = (g, f, attempt + 1)
                    else:
                        failures.append(record)
                completed = sum(task_complete(group, fold, source_hash) for group, fold in jobs)
                manifest = json.loads((EXP / "manifest.json").read_text(encoding="utf-8"))
                manifest.update({"completed_tasks": completed, "training_execution": {
                    "worker_limit": workers, "max_workers_observed": peak_workers, "wall_seconds": time.time() - started,
                    "retries": retries, "failures": failures, "oom_events": oom}})
                atomic_json(EXP / "manifest.json", manifest)
                break
    completed = sum(task_complete(group, fold, source_hash) for group, fold in jobs)
    manifest = json.loads((EXP / "manifest.json").read_text(encoding="utf-8"))
    manifest["completed_tasks"] = completed
    manifest["training_execution"] = {"worker_limit": workers, "max_workers_observed": peak_workers,
        "wall_seconds": time.time() - started, "retries": retries, "failures": failures, "oom_events": oom}
    manifest["training_end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["status"] = "TRAINED" if completed == 30 else "PARTIAL"
    atomic_json(EXP / "manifest.json", manifest)
    if completed != 30:
        raise RuntimeError(f"formal training completed {completed}/30")
    return manifest["training_execution"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    for path in (TASKS, CKPTS, LATEST, LOGS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        print(json.dumps(preflight(), indent=2))
    elif args.train:
        print(json.dumps(run_training(args.workers), indent=2))
    elif args.worker:
        train_worker(args.group, args.fold)
    else:
        parser.error("choose --preflight, --train, or --worker")


if __name__ == "__main__":
    main()
