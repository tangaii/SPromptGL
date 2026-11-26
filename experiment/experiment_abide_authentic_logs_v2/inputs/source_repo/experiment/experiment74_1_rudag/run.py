#!/usr/bin/env python3
"""Experiment 74.1 runner; all training state is isolated under this directory."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
RT = EXP / ".runtime"
TASKS = RT / "tasks"
CKPTS = RT / "checkpoints"
LOGS = RT / "logs"
RESULTS = EXP / "results"
BASE = Path(CFG["base"]["path"])
E53 = Path("/model/spromptgl/experiment/experiment53_3_cadr_graph_fallback_fix")
E61_EXP = Path("/data/adodas/SPromptGL-main/experiment/experiment61_cadr_postalign_vs_e2e")
E62 = Path("/model/spromptgl/experiment/experiment62_e2e_gate_init_sensitivity")
E74 = Path("/model/spromptgl/experiment/experiment74_udag")
GROUPS = tuple(CFG["rudag"]["groups"])
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
E69 = None
RUDAG = None


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_object(value):
    import numpy as np
    import torch
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            digest.update(b"T" + str(item.dtype).encode() + str(tuple(item.shape)).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b"N" + str(item.dtype).encode() + str(item.shape).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode() + b"\0")
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode() + b"\0")

    visit(value)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
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
    if not path.is_file() or path.stat().st_size == 0:
        raise IOError(f"checkpoint write/readback failed: {path}")


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path, rows):
    path = Path(path)
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


def load_e69():
    global E69
    if E69 is not None:
        return E69
    expected = CFG["base"]
    if sha_file(BASE / "run.py") != expected["run_sha256"] or sha_file(BASE / "manifest.json") != expected["manifest_sha256"]:
        raise RuntimeError("E69.3-G2 source or manifest SHA changed")
    manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status"), manifest.get("completed_tasks"), manifest.get("independent_audits")) != ("COMPLETE_AUDITED", 10, 10):
        raise RuntimeError("E69.3-G2 historical reference is not fully audited")
    if manifest.get("base", {}).get("group") != "E62-E2":
        raise RuntimeError("E69.3-G2 lineage does not point to E62-E2")
    spec = importlib.util.spec_from_file_location("rudag_readonly_e693_runner", BASE / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.load_e61()
    module.verify_e62_reference()
    E69 = module
    return module


def load_rudag():
    global RUDAG
    if RUDAG is not None:
        return RUDAG
    name = "rudag_741_isolated_model"
    spec = importlib.util.spec_from_file_location(name, EXP / "rudag.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    RUDAG = module
    return module


def source_files():
    paths = [
        EXP / "config.json", EXP / "run.py", EXP / "rudag.py", EXP / "audit.py",
        Path("/data/adodas/SPromptGL-main/main.py"), Path("/data/adodas/SPromptGL-main/model.py"),
        Path("/data/adodas/SPromptGL-main/network.py"), Path("/data/adodas/SPromptGL-main/layers.py"),
        Path("/data/adodas/SPromptGL-main/utils.py"),
        Path("/data/adodas/SPromptGL-main/ABIDE-simple-2-concat-weighted-cosine.sh"),
        Path("/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv"),
        Path("/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy"),
        E53 / "run.py", E53 / "cadr_gfr.py", E53 / "config.json", E53 / "manifest.json",
        E61_EXP / "run.py", E61_EXP / "model_e61.py", E61_EXP / "config.json", E61_EXP / "manifest.json",
        E62 / "run.py", E62 / "config.json", E62 / "manifest.json", E62 / "results/fold_metrics.csv",
        BASE / "run.py", BASE / "config.json", BASE / "manifest.json", BASE / "results/fold_metrics.csv",
        E74 / "run.py", E74 / "udag.py", E74 / "config.json", E74 / "manifest.json", E74 / "results/fold_metrics.csv",
    ]
    output = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required provenance file missing: {path}")
        output[str(path)] = sha_file(path)
    e61 = load_e69().load_e61()
    for fold in range(1, 11):
        path = e61.warmup_path(fold)
        output[str(path)] = sha_file(path)
    return output


def state_model_sha(state):
    digest = hashlib.sha256()
    for key in ("model", "cam"):
        for name, tensor in sorted(state[key].state_dict().items()):
            digest.update((key + "/" + name).encode() + b"\0")
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_bundle(state):
    import torch
    return {
        "model": {k: v.detach().cpu().clone() for k, v in state["model"].state_dict().items()},
        "cam": {k: v.detach().cpu().clone() for k, v in state["cam"].state_dict().items()},
        "opt_base": copy.deepcopy(state["opt_base"].state_dict()),
        "opt_cam": copy.deepcopy(state["opt_cam"].state_dict()),
        "opt_added": copy.deepcopy(state["opt_added"].state_dict()),
    }


def load_bundle(state, bundle):
    state["model"].load_state_dict(bundle["model"], strict=True)
    state["cam"].load_state_dict(bundle["cam"], strict=True)
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].load_state_dict(copy.deepcopy(bundle[key]))


def zero_optimizers(state):
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].zero_grad(set_to_none=True)


def prepare(fold):
    import numpy as np
    import torch

    g = load_e69()
    e61 = g.load_e61()
    base_runtime = e61.load_base()
    state, warmup = e61.load_warmup_state(fold)
    membership = g.apply_fold_specific_training_additions(state, fold)
    e61.configure_branch(state, "E61-E2", None)

    model = state["model"]
    gate_slots_before = g.optimizer_gate_state(state)
    for name, slot in gate_slots_before.items():
        if not slot["in_optimizer"]:
            raise RuntimeError(f"gate is missing from the inherited DARG optimizer: {name}")
    if any(slot["present"] for slot in gate_slots_before.values()):
        raise RuntimeError(f"warmup unexpectedly has optimizer moments for compatibility gates: {gate_slots_before}")

    theta = torch.tensor(
        [math.log(0.1 / 0.9)] * 3 + [math.log(0.99 / 0.01)],
        dtype=model.adaptive_gate_logits.dtype,
        device=model.adaptive_gate_logits.device,
    )
    rng_before = base_runtime.rng_sha(base_runtime.pack_rng())
    with torch.no_grad():
        model.adaptive_gate_logits.copy_(theta)
    model.edge_gate_logits.requires_grad_(False)
    model.edge_gate_logits.grad = None
    model.adaptive_gate_logits.requires_grad_(True)
    model.adaptive_gate_logits.grad = None
    # Reparameterized adaptive logits start from the declared values, so their
    # optimizer moments must be empty. The other inherited optimizer state stays intact.
    adaptive_slot_cleared = model.adaptive_gate_logits in state["opt_added"].state
    state["opt_added"].state.pop(model.adaptive_gate_logits, None)
    rng_after = base_runtime.rng_sha(base_runtime.pack_rng())
    if rng_before != rng_after or rng_after != warmup["rng_sha"]:
        raise RuntimeError("deterministic gate initialization changed the inherited RNG state")

    if not np.isin(state["train_pairs_np"], state["train_np"]).all():
        raise RuntimeError("training relation pairs include a patient outside effective training")
    if state["train_pairs_np"].shape != (4096, 2):
        raise RuntimeError(f"expected exactly 4096 training patient pairs, got {state['train_pairs_np'].shape}")
    if not np.array_equal(
        state["train_same_np"],
        state["labels_np"][state["train_pairs_np"][:, 0]] == state["labels_np"][state["train_pairs_np"][:, 1]],
    ):
        raise RuntimeError("training pair targets differ from the effective training labels")
    expected_overlap = [377, 407] if fold == 2 else []
    if membership["train_test_overlap_1based"] != expected_overlap:
        raise RuntimeError(f"unexpected train/test label exposure in fold {fold}: {membership}")

    task = json.loads((BASE / ".runtime/tasks" / f"E69.3-G2_f{fold:02d}.json").read_text(encoding="utf-8"))
    for key in ("initial_shared_sha", "initial_gfr_sha", "initial_cam_sha", "relation_core_sha", "initial_rng_sha", "split_sha", "pair_sha", "warmup_pair_sha", "effective_train_sha", "anchor_sha"):
        if state.get(key) != task.get(key):
            raise RuntimeError(f"E69.3-G2 initialization identity differs for fold {fold}: {key}")
    if sha_file(e61.warmup_path(fold)) != task["warmup_sha256"] or warmup["rng_sha"] != task["warmup_rng_sha"]:
        raise RuntimeError(f"warmup provenance mismatch in fold {fold}")

    expected_wd_lr = (("opt_base", 0.11, 0.0038), ("opt_cam", 0.11, 0.0038), ("opt_added", 0.01, 0.001))
    for key, weight_decay, lr in expected_wd_lr:
        for group in state[key].param_groups:
            if abs(group["weight_decay"] - weight_decay) > 1e-12 or abs(group["lr"] - lr) > 1e-12:
                raise RuntimeError(f"inherited optimizer settings differ for {key}: {group}")
    if model.edge_gate_logits.requires_grad or not model.adaptive_gate_logits.requires_grad:
        raise RuntimeError("compatibility/effective gate trainability is incorrect")

    initial = {
        "model_cam_sha": state_model_sha(state),
        "optimizer_sha": sha_object({key: state[key].state_dict() for key in ("opt_base", "opt_cam", "opt_added")}),
        "warmup_sha256": sha_file(e61.warmup_path(fold)),
        "warmup_rng_sha": warmup["rng_sha"],
        "initial_rng_sha": base_runtime.rng_sha(base_runtime.pack_rng()),
        "edge_gate_logits_initial": model.edge_gate_logits.detach().cpu().tolist(),
        "edge_gate_values_initial": torch.sigmoid(model.edge_gate_logits.detach()).cpu().tolist(),
        "adaptive_gate_theta_initial": model.adaptive_gate_logits.detach().cpu().tolist(),
        "gate_scales": [0.01, 0.01, 0.01, 1.0],
        "effective_gate_values_initial": [
            float(0.01 * torch.sigmoid(model.adaptive_gate_logits[i]).item()) if i < 3
            else float(torch.sigmoid(model.adaptive_gate_logits[i]).item()) for i in range(4)
        ],
        "adaptive_optimizer_state_was_cleared": bool(adaptive_slot_cleared),
        "optimizer_gate_slots_at_warmup": gate_slots_before,
        "membership": membership,
        "split_sha": state["split_sha"],
        "pair_sha": state["pair_sha"],
        "effective_train_sha": state["effective_train_sha"],
        "train_pair_scope_pass": True,
    }
    state["rudag_identity"] = initial
    state["fold"] = fold
    return state, warmup, initial


def task_path(group, fold):
    return TASKS / f"{group}_f{fold:02d}.json"


def best_path(group, fold):
    return CKPTS / f"{group}_f{fold:02d}.best.pt"


def latest_path(group, fold):
    return CKPTS / f"{group}_f{fold:02d}.latest.pt"


def log_path(group, fold):
    return LOGS / f"{group}_f{fold:02d}.jsonl"


def forward(state, group, training=False, intervention="full", fold=None, collect=False):
    return load_rudag().forward_state(
        state, load_e69().load_e61(), group, training=training,
        intervention=intervention, fold=state["fold"] if fold is None else fold,
        collect_diagnostics=collect,
        permutation_seed_base=CFG["rudag"]["score_permutation_seed_base"],
    )


def parameter_gradients(state):
    e61 = load_e69().load_e61()
    model = state["model"]
    return {
        "adaptive_gate_abs": model.adaptive_gate_logits.grad.detach().abs().cpu().tolist() if model.adaptive_gate_logits.grad is not None else [0.0] * 4,
        "edge_gate_grad_is_none": model.edge_gate_logits.grad is None,
        "scorer_norms": [e61.total_gradient_norm(module.parameters()) for module in model.scorers],
        "encoder_norms": [e61.total_gradient_norm(module.parameters()) for module in model.encoders],
        "discriminator_norm": e61.total_gradient_norm(model.discriminator.parameters()),
        "graphlearn_norms": [e61.total_gradient_norm(module.parameters()) for module in model.graphs],
        "gcn_norms": [e61.total_gradient_norm(module.parameters()) for module in model.gcn],
    }


def graph_preflight_tests():
    import torch
    from rudag import unified_adjacency

    base = torch.tensor([
        [0.7, 0.8, 0.0, 0.0],
        [0.8, 0.6, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.2],
        [0.0, 0.0, 0.2, 0.0],
    ], dtype=torch.float64)
    rows = torch.tensor([0, 1, 2])
    cols = torch.tensor([1, 2, 3])
    q = torch.tensor([0.9, 0.5, 0.1], dtype=torch.float64, requires_grad=True)
    corr_gate = torch.tensor(0.99, dtype=torch.float64, requires_grad=True)
    corr, _ = unified_adjacency(base, rows, cols, q, corr_gate, 3)
    expected = torch.tensor([0.8 * (1 + 0.8 * 0.99 * 0.8), 0.5, 0.2 * (1 - 0.8 * 0.99 * 0.8)], dtype=torch.float64)
    if not torch.allclose(corr[rows, cols], expected, atol=1e-12, rtol=0):
        raise RuntimeError("CORR all-edge continuous reweighting failed")
    if not torch.equal(corr.diag(), base.diag()) or not torch.equal(corr, corr.T) or corr.min() < 0:
        raise RuntimeError("CORR diagonal/symmetry/nonnegativity invariant failed")
    if torch.any((base == 0) & (corr != 0)):
        raise RuntimeError("unified graph added an edge outside GraphLearn support")
    neutral, _ = unified_adjacency(base, rows, cols, torch.full_like(q, 0.5), corr_gate, 3)
    zero_gate, _ = unified_adjacency(base, rows, cols, q, corr_gate.new_zeros(()), 3)
    if not torch.equal(neutral, base) or not torch.equal(zero_gate, base):
        raise RuntimeError("q=.5 or g=0 did not recover the base graph")

    weak_gate = torch.tensor(0.001, dtype=torch.float64, requires_grad=True)
    weak, _ = unified_adjacency(base, rows[:2], cols[:2], q[:2], weak_gate, 0)
    if not torch.equal(weak.diag(), base.diag()) or not torch.equal(weak, weak.T):
        raise RuntimeError("non-CORR graph symmetry/diagonal failed")
    if not torch.equal(weak[1, 2], base[1, 2]):
        raise RuntimeError("non-CORR changed an edge outside its selected candidate set")
    if torch.any((base == 0) & (weak != 0)):
        raise RuntimeError("non-CORR added a fallback/nonexistent edge")
    if not torch.isfinite(load_e69().load_e61().load_base().runtime()["utils"].normalize_adj(corr + torch.eye(4, dtype=corr.dtype))).all():
        raise RuntimeError("A+I graph normalization is not finite")
    zero_node = torch.zeros((4, 4), dtype=torch.float64)
    norm_zero = load_e69().load_e61().load_base().runtime()["utils"].normalize_adj(zero_node + torch.eye(4, dtype=zero_node.dtype))
    if not torch.isfinite(norm_zero).all():
        raise RuntimeError("zero-degree graph normalization produced non-finite values")
    (corr.square().sum() + weak.square().sum()).backward()
    if q.grad is None or q.grad.abs().sum() == 0 or corr_gate.grad is None or corr_gate.grad.abs() == 0 or weak_gate.grad is None or weak_gate.grad.abs() == 0:
        raise RuntimeError("unified adjacency edge/gate gradient check failed")
    return {
        "status": "PASS", "corr_scored_all_nonzero_offdiag": True,
        "noncorr_selected_only": True, "support_preserved": True,
        "symmetric_nonnegative": True, "diagonal_unchanged": True,
        "q_half_neutral": True, "corr_gate_zero_neutral": True,
        "normalization_adds_identity_once": True, "zero_degree_finite": True,
        "edge_and_gate_gradients": True,
    }


def _finite_module_grads(module):
    import torch
    return all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in module.parameters())


def preflight():
    import numpy as np
    import torch

    for path in (TASKS, CKPTS, LOGS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
    e69 = load_e69()
    e61 = e69.load_e61()
    base_runtime = e61.load_base()
    source_sha = source_files()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("the historical PPU-ZW810E torch.cuda backend is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible accelerator, found {torch.cuda.device_count()}")
    memory_free, memory_total = torch.cuda.mem_get_info()

    inheritance = []
    for fold in range(1, 11):
        s1, w1, i1 = prepare(fold)
        s2, w2, i2 = prepare(fold)
        if i1["model_cam_sha"] != i2["model_cam_sha"] or i1["optimizer_sha"] != i2["optimizer_sha"]:
            raise RuntimeError(f"G1/G2 initial shared state differs in fold {fold}")
        if w1["rng_sha"] != w2["rng_sha"] or base_runtime.rng_sha(base_runtime.pack_rng()) != w1["rng_sha"]:
            raise RuntimeError(f"warmup RNG inheritance differs in fold {fold}")
        if i1["train_pair_scope_pass"] is not True or i2["train_pair_scope_pass"] is not True:
            raise RuntimeError(f"training pair scope failed in fold {fold}")
        inheritance.append({"fold": fold, "group_initial_model_sha_equal": True,
                            "optimizer_sha_equal": True, "warmup_sha256": i1["warmup_sha256"],
                            "warmup_rng_sha": w1["rng_sha"], "effective_train_sha": i1["effective_train_sha"],
                            "pair_sha": i1["pair_sha"], "overlap_1based": i1["membership"]["train_test_overlap_1based"],
                            "gate_values": i1["effective_gate_values_initial"],
                            "edge_gate_frozen": not s1["model"].edge_gate_logits.requires_grad,
                            "adaptive_gate_reparameterization_recorded": True})
        del s1, s2, w1, w2
        torch.cuda.empty_cache()

    tests = graph_preflight_tests()
    state, warmup, identity = prepare(1)
    initial = state_bundle(state)
    start_rng = base_runtime.pack_rng()
    call_counts = [0, 0, 0, 0]
    cam_count = [0]
    hooks = [module.register_forward_hook(lambda _m, _i, _o, index=i: call_counts.__setitem__(index, call_counts[index] + 1)) for i, module in enumerate(state["model"].gcn)]
    hooks.append(state["cam"].register_forward_hook(lambda *_: cam_count.__setitem__(0, cam_count[0] + 1)))
    zero_optimizers(state)
    out = forward(state, "G2-RUDAG-Full", training=False)
    out["cls"].backward()
    natural_grad = parameter_gradients(state)
    for hook in hooks:
        hook.remove()
    if call_counts != [1, 1, 1, 1] or cam_count[0] != 1:
        raise RuntimeError(f"formal GCN/CAM call count mismatch: {call_counts}, {cam_count}")
    if not _finite_module_grads(state["model"]) or not _finite_module_grads(state["cam"]):
        raise RuntimeError("natural classification gradients contain NaN/Inf")
    if not state["model"].edge_gate_logits.grad is None:
        raise RuntimeError("frozen edge_gate_logits received a gradient")
    del out

    zero_optimizers(state)
    controlled_hooks = []
    for scorer in state["model"].scorers:
        controlled_hooks.append(scorer.register_forward_hook(
            lambda _m, _i, value: value + (torch.full_like(value, float(torch.logit(torch.tensor(0.9, device=value.device)))) - value.detach())
        ))
    controlled = forward(state, "G2-RUDAG-Full", training=False)
    controlled["cls"].backward()
    controlled_grad = parameter_gradients(state)
    for hook in controlled_hooks:
        hook.remove()
    if not all(math.isfinite(v) and v > 0 for v in controlled_grad["adaptive_gate_abs"]):
        raise RuntimeError(f"one or more effective gates have no controlled classification gradient: {controlled_grad}")
    if not all(math.isfinite(v) and v > 0 for v in controlled_grad["scorer_norms"]):
        raise RuntimeError(f"one or more disease scorers have no controlled classification gradient: {controlled_grad}")

    # Bounded direct differences must remain within each raw feature block's RMS.
    from cadr_gfr import _bounded_block_delta
    bounded_ok = True
    for raw_block, delta_block in zip(state["parts"], controlled["info"]["neighbor_differences"]):
        rms = torch.sqrt(raw_block.square().mean(1, keepdim=True).clamp_min(1e-8))
        bound = _bounded_block_delta(delta_block, raw_block)
        bounded_ok = bounded_ok and bool((bound.abs() <= rms + 1e-6).all())
    if not bounded_ok:
        raise RuntimeError("bounded neighbor differences exceed the RMS-tanh range")
    if any(identity["effective_gate_values_initial"][i] > 0.01 for i in range(3)):
        raise RuntimeError("non-CORR gate exceeds its structural 0.01 cap")
    del controlled

    load_bundle(state, initial)
    base_runtime.restore_rng(start_rng)
    zero_optimizers(state)
    torch.cuda.reset_peak_memory_stats()
    first = forward(state, "G1-RUDAG-U", training=True)
    first_logp = first["logp"].detach().cpu().clone()
    first_loss = float(first["total"].detach())
    first["total"].backward()
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    after_one = state_bundle(state)
    rng_g1_train = base_runtime.rng_sha(base_runtime.pack_rng())
    with torch.no_grad():
        eval_g1 = forward(state, "G1-RUDAG-U", training=False)
    rng_g1_eval = base_runtime.rng_sha(base_runtime.pack_rng())
    del first
    load_bundle(state, initial)
    base_runtime.restore_rng(start_rng)
    zero_optimizers(state)
    second = forward(state, "G1-RUDAG-U", training=True)
    second_logp = second["logp"].detach().cpu()
    second_loss = float(second["total"].detach())
    second["total"].backward()
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    after_two = state_bundle(state)
    rng_g1_replay = base_runtime.rng_sha(base_runtime.pack_rng())
    if not torch.equal(first_logp, second_logp) or first_loss != second_loss:
        raise RuntimeError("same-group one-step forward replay failed")
    if sha_object(after_one) != sha_object(after_two) or rng_g1_train != rng_g1_replay:
        raise RuntimeError("same-group first optimizer update or RNG replay failed")
    load_bundle(state, initial)
    base_runtime.restore_rng(start_rng)
    zero_optimizers(state)
    g2_step = forward(state, "G2-RUDAG-Full", training=True)
    g2_step["total"].backward()
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    rng_g2_train = base_runtime.rng_sha(base_runtime.pack_rng())
    with torch.no_grad():
        _ = forward(state, "G2-RUDAG-Full", training=False)
    rng_g2_eval = base_runtime.rng_sha(base_runtime.pack_rng())
    if rng_g1_train != rng_g2_train or rng_g1_eval != rng_g2_eval:
        raise RuntimeError("G1/G2 shared-epoch RNG trajectories differ")
    peak_bytes = int(torch.cuda.max_memory_allocated())
    next_rng = base_runtime.pack_rng()

    # Save/load the entire recoverable state and confirm the next update is identical.
    resume_path = RT / "preflight_resume.pt"
    atomic_torch(resume_path, {"state": initial, "rng": start_rng})
    loaded = torch_load(resume_path)
    load_bundle(state, loaded["state"])
    base_runtime.restore_rng(loaded["rng"])
    zero_optimizers(state)
    third = forward(state, "G1-RUDAG-U", training=True)
    third_logp = third["logp"].detach().cpu()
    third_loss = float(third["total"].detach())
    third["total"].backward()
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    after_three = state_bundle(state)
    if not torch.equal(first_logp, third_logp) or first_loss != third_loss or sha_object(after_one) != sha_object(after_three):
        raise RuntimeError("save/reload optimizer/model next-update parity failed")
    if base_runtime.rng_sha(base_runtime.pack_rng()) != rng_g1_train:
        raise RuntimeError("save/reload next-step RNG parity failed")
    resume_path.unlink()
    base_runtime.restore_rng(next_rng)

    result = {
        "status": "PASS", "device": device_name, "device_count": torch.cuda.device_count(),
        "memory_free_bytes_before": int(memory_free), "memory_total_bytes": int(memory_total),
        "measured_preflight_peak_allocated_bytes": peak_bytes,
        "graph_tests": tests, "all_fold_inheritance": inheritance,
        "natural_classification_gradients": natural_grad,
        "controlled_classification_gradients": controlled_grad,
        "formal_gcn_calls_per_modality": call_counts, "cam_calls": cam_count[0],
        "direct_difference_bound_pass": bounded_ok,
        "noncorr_gate_cap_pass": True,
        "same_group_update_and_g1_g2_rng_parity": True,
        "checkpoint_optimizer_rng_resume_parity": True,
        "rng_state_restored_after_preflight_checks": base_runtime.rng_sha(base_runtime.pack_rng()) == base_runtime.rng_sha(next_rng),
        "source_sha256": source_sha,
    }
    atomic_json(RESULTS / "preflight.json", result)
    print(f"PREFLIGHT PASS device={device_name} peak={peak_bytes} bytes", flush=True)
    return result


def _checkpoint_payload(state, group, fold, best, epoch, executed, source_sha, trace):
    return {
        "experiment": CFG["experiment"], "group": group, "fold": fold, "seed": 0,
        "config": CFG, "source_sha256": source_sha,
        "identity": state["rudag_identity"],
        "best_epoch_1based": epoch + 1, "epochs_executed": executed,
        "metrics": best["metrics"], "test_probability": best["probability"],
        "test_prediction": best["prediction"], "best_state": best["state"],
        "best_losses": best["losses"], "best_training_rng_sha": best["rng_after_train"],
        "best_eval_rng_sha": best["rng_after_eval"], "rng_trace": trace,
        "initial_gate_values": state["rudag_identity"]["effective_gate_values_initial"],
        "initial_gate_theta": state["rudag_identity"]["adaptive_gate_theta_initial"],
        "edge_gate_values_initial": state["rudag_identity"]["edge_gate_values_initial"],
        "adaptive_optimizer_state_was_cleared": state["rudag_identity"]["adaptive_optimizer_state_was_cleared"],
    }


def train_worker(group, fold):
    import numpy as np
    import torch

    if group not in GROUPS or fold not in range(1, 11):
        raise ValueError(f"invalid formal task: {group} fold {fold}")
    source_sha = source_files()
    state, warmup, identity = prepare(fold)
    task_file, best_file, latest_file = task_path(group, fold), best_path(group, fold), latest_path(group, fold)
    if task_file.exists():
        existing = json.loads(task_file.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE" and existing.get("source_sha256") == source_sha and existing.get("checkpoint_sha256") == sha_file(best_file):
            print(f"SKIP already complete {group} fold={fold}", flush=True)
            return existing
        raise RuntimeError(f"refusing to overwrite incompatible task record {task_file}")

    e61 = load_e69().load_e61()
    base_runtime = e61.load_base()
    if base_runtime.rng_sha(base_runtime.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError("formal task did not restore the exact shared warmup RNG")

    start_epoch, wait = 10, 0
    best_acc, best_epoch, best = -1.0, -1, None
    trace = []
    if latest_file.exists():
        saved = torch_load(latest_file)
        checks = {
            "group": group, "fold": fold, "source_sha256": source_sha,
            "identity": identity, "warmup_sha256": identity["warmup_sha256"],
        }
        for key, value in checks.items():
            if saved.get(key) != value:
                raise RuntimeError(f"incompatible recovery state field {key} for {group} fold={fold}")
        load_bundle(state, saved["current"])
        base_runtime.restore_rng(saved["rng"])
        start_epoch, wait = int(saved["next_epoch"]), int(saved["wait"])
        best_acc, best_epoch, best, trace = float(saved["best_acc"]), int(saved["best_epoch"]), saved["best"], saved["trace"]
    if start_epoch < 10 or start_epoch > CFG["training"]["max_epochs"]:
        raise RuntimeError(f"invalid recovery epoch {start_epoch}")

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    executed = start_epoch
    for epoch in range(start_epoch, CFG["training"]["max_epochs"]):
        rng_before = base_runtime.rng_sha(base_runtime.pack_rng())
        zero_optimizers(state)
        out = forward(state, group, training=True)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"nonfinite total loss {group} fold={fold} epoch={epoch + 1}")
        out["total"].backward()
        if not all(_finite_module_grads(module) for module in (state["model"], state["cam"])):
            raise FloatingPointError(f"nonfinite gradient {group} fold={fold} epoch={epoch + 1}")
        gradients = parameter_gradients(state)
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        rng_after_train = base_runtime.rng_sha(base_runtime.pack_rng())
        executed = epoch + 1
        with torch.no_grad():
            evaluated = forward(state, group, training=False)
            metrics = base_runtime.classification_metrics(evaluated["logp"], state["test"], state["y"])
        rng_after_eval = base_runtime.rng_sha(base_runtime.pack_rng())
        record_rng = {"epoch": epoch + 1, "rng_before": rng_before,
                      "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval}
        trace.append(record_rng)

        if metrics["acc"] > best_acc:
            best_acc = float(metrics["acc"])
            best_epoch = epoch
            wait = 0
            best = {
                "state": state_bundle(state),
                "metrics": {key: float(value) for key, value in metrics.items() if key not in ("pred", "prob")},
                "probability": metrics["prob"].copy(),
                "prediction": metrics["pred"].copy(),
                "losses": {key: float(out[key].detach().cpu()) for key in ("cls", "graph", "pair", "adv", "preserve", "total")},
                "rng_after_train": rng_after_train,
                "rng_after_eval": rng_after_eval,
                "gradient": gradients,
            }
            atomic_torch(best_file, _checkpoint_payload(state, group, fold, best, best_epoch, executed, source_sha, trace))
        else:
            wait += 1

        if (executed % 10 == 0 or executed == CFG["training"]["max_epochs"] or wait > CFG["training"]["early_stop_wait_gt"]):
            atomic_torch(latest_file, {
                "group": group, "fold": fold, "source_sha256": source_sha,
                "identity": identity, "warmup_sha256": identity["warmup_sha256"],
                "current": state_bundle(state), "rng": base_runtime.pack_rng(),
                "next_epoch": executed, "wait": wait,
                "best_acc": best_acc, "best_epoch": best_epoch, "best": best, "trace": trace,
            })
            effective_gate_values = torch.sigmoid(state["model"].adaptive_gate_logits.detach()).cpu().tolist()
            effective_gate_values[:3] = [0.01 * value for value in effective_gate_values[:3]]
            log_record = {
                "group": group, "fold": fold, "epoch": executed,
                "best_epoch_1based": best_epoch + 1, "best_acc": best_acc,
                "metrics": {key: float(metrics[key]) for key in METRICS},
                "loss": float(out["total"].detach().cpu()),
                "gate_values": [float(value) for value in effective_gate_values],
                "gate_grad_abs": gradients["adaptive_gate_abs"],
                "rng": record_rng,
                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            }
            with log_path(group, fold).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(log_record, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(f"{group} fold={fold} epoch={executed} best={best_acc:.6f}@{best_epoch + 1} wait={wait}", flush=True)
        del out, evaluated
        if wait > CFG["training"]["early_stop_wait_gt"]:
            break

    if best is None:
        raise RuntimeError(f"no checkpoint selected for {group} fold={fold}")
    checkpoint = _checkpoint_payload(state, group, fold, best, best_epoch, executed, source_sha, trace)
    atomic_torch(best_file, checkpoint)
    task = {
        "status": "COMPLETE", "group": group, "fold": fold, "seed": 0,
        "source_sha256": source_sha, "identity": identity,
        "checkpoint_sha256": sha_file(best_file),
        "best_epoch_1based": best_epoch + 1, "epochs_executed": executed,
        "metrics": best["metrics"], "rng_trace": trace,
        "wall_seconds": time.time() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "device": torch.cuda.get_device_name(0),
    }
    atomic_json(task_file, task)
    print(f"COMPLETE {group} fold={fold} best={best_acc:.6f}@{best_epoch + 1}", flush=True)
    return task


def _task_complete(group, fold, source_sha):
    path = task_path(group, fold)
    checkpoint = best_path(group, fold)
    if not path.is_file() or not checkpoint.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return record.get("status") == "COMPLETE" and record.get("source_sha256") == source_sha and record.get("checkpoint_sha256") == sha_file(checkpoint)
    except Exception:
        return False


def run_all(workers=10):
    workers = max(1, min(10, int(workers)))
    preflight_record = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight_record.get("status") != "PASS":
        raise RuntimeError("formal training requires a passing preflight")
    expected_sources = preflight_record["source_sha256"]
    if source_files() != expected_sources:
        raise RuntimeError("a protected code/data/reference source changed after preflight")

    import torch
    free, total = torch.cuda.mem_get_info()
    peak = max(1, int(preflight_record["measured_preflight_peak_allocated_bytes"]))
    estimated_worker_need = int(peak * 1.4)
    memory_cap = max(1, int((free * 0.80) / estimated_worker_need))
    worker_limit = min(workers, memory_cap)
    if worker_limit < 1:
        worker_limit = 1
    pending = [(group, fold) for fold in range(1, 11) for group in GROUPS]
    pending = [(g, f) for g, f in pending if not _task_complete(g, f, expected_sources)]
    controller_path = LOGS / "controller.log"
    failed, retries = [], []
    active = {}
    attempts = {(g, f): 0 for g, f in pending}
    maximum_observed = 0
    start_time = time.time()
    with controller_path.open("a", encoding="utf-8") as controller:
        controller.write(json.dumps({"event": "START", "pending": len(pending), "worker_limit": worker_limit,
                                     "memory_free": int(free), "memory_total": int(total),
                                     "preflight_peak": peak, "device": torch.cuda.get_device_name(0)}) + "\n")
        controller.flush()
        while pending or active:
            while pending and len(active) < worker_limit:
                group, fold = pending.pop(0)
                attempts[(group, fold)] += 1
                logfile = LOGS / f"{group}_f{fold:02d}.stdout.log"
                stream = logfile.open("a", encoding="utf-8")
                env = os.environ.copy()
                env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                            "NUMEXPR_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_VISIBLE_DEVICES": "0"})
                proc = subprocess.Popen(
                    [sys.executable, str(EXP / "run.py"), "--worker", "--group", group, "--fold", str(fold)],
                    cwd=str(EXP), env=env, stdout=stream, stderr=subprocess.STDOUT,
                )
                active[proc] = (group, fold, stream, logfile)
                maximum_observed = max(maximum_observed, len(active))
            time.sleep(2)
            for proc, (group, fold, stream, logfile) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                stream.close()
                del active[proc]
                if code == 0 and _task_complete(group, fold, expected_sources):
                    event = {"event": "COMPLETE", "group": group, "fold": fold, "attempt": attempts[(group, fold)]}
                else:
                    try:
                        tail = logfile.read_text(encoding="utf-8", errors="replace")[-8000:]
                    except Exception:
                        tail = "unable to read task log"
                    oom = any(term in tail.lower() for term in ("out of memory", "memory allocation", "resource exhausted", "oom"))
                    retry_limit = 2 if oom else 1
                    if attempts[(group, fold)] <= retry_limit:
                        retries.append({"group": group, "fold": fold, "attempt": attempts[(group, fold)],
                                        "exit_code": code, "oom_detected": oom, "tail": tail})
                        pending.append((group, fold))
                        if oom:
                            worker_limit = max(1, worker_limit // 2)
                    else:
                        failed.append({"group": group, "fold": fold, "exit_code": code,
                                       "attempts": attempts[(group, fold)], "oom_detected": oom, "tail": tail})
                    event = {"event": "FAIL_OR_RETRY", "group": group, "fold": fold,
                             "attempt": attempts[(group, fold)], "exit_code": code, "oom_detected": oom}
                controller.write(json.dumps(event) + "\n")
                controller.flush()
            status = {
                "status": "TRAINING" if not failed else "PARTIAL",
                "expected_tasks": 20,
                "completed_tasks": sum(_task_complete(g, f, expected_sources) for g in GROUPS for f in range(1, 11)),
                "audited_tasks": 0, "active_tasks": [f"{v[0]}_f{v[1]:02d}" for v in active.values()],
                "failed_attempt_records": len(retries), "failed_tasks": failed,
                "retries": retries,
                "max_workers_observed": maximum_observed, "worker_limit": worker_limit,
                "device": torch.cuda.get_device_name(0), "oom_events": sum(x["oom_detected"] for x in retries + failed),
            }
            atomic_json(EXP / "manifest.json", status)
        controller.write(json.dumps({"event": "TRAINING_STOP", "wall_seconds": time.time() - start_time,
                                     "completed": sum(_task_complete(g, f, expected_sources) for g in GROUPS for f in range(1, 11)),
                                     "failed": failed, "retries": retries, "max_workers_observed": maximum_observed}) + "\n")
        controller.flush()
    if failed:
        raise RuntimeError(f"formal training failures remain: {failed}")
    if sum(_task_complete(g, f, expected_sources) for g in GROUPS for f in range(1, 11)) != 20:
        raise RuntimeError("formal training did not produce all 20 complete checkpoints")
    print(f"TRAINING COMPLETE 20/20 max_workers={maximum_observed} retries={len(retries)}", flush=True)
    training_execution = {"max_workers_observed": maximum_observed, "worker_limit": worker_limit,
                          "retries": retries, "failures": failed,
                          "oom_events": sum(x["oom_detected"] for x in retries + failed),
                          "wall_seconds": time.time() - start_time}
    atomic_json(EXP / "manifest.json", {
        "status": "TRAINED_AWAITING_AUDIT", "expected_tasks": 20, "completed_tasks": 20,
        "audited_tasks": 0, "device": torch.cuda.get_device_name(0),
        "worker_limit": worker_limit, "max_workers_observed": maximum_observed,
        "failed_attempt_records": len(retries), "failed_tasks": failed,
        "retries": retries, "oom_events": training_execution["oom_events"],
        "source_sha256": expected_sources,
    })
    name = "rudag_741_independent_audit"
    spec = importlib.util.spec_from_file_location(name, EXP / "audit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    final_manifest = module.finalize()
    return {"training_execution": training_execution, "manifest": final_manifest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    for path in (TASKS, CKPTS, LOGS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        preflight()
    elif args.worker:
        train_worker(args.group, args.fold)
    elif args.audit:
        name = "rudag_741_independent_audit"
        spec = importlib.util.spec_from_file_location(name, EXP / "audit.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module.finalize()
    else:
        run_all(args.workers)


if __name__ == "__main__":
    main()
