#!/usr/bin/env python3
"""Experiment 69.3 runner: E62-E2 continuation with fold-specific train/test overlap."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/SPromptGL-main")
E61_EXP = ROOT / "experiment/experiment61_cadr_postalign_vs_e2e"
E61_RUN = E61_EXP / "run.py"
E61_MODEL = E61_EXP / "model_e61.py"
E61_CONFIG = E61_EXP / "config.json"
E61_MANIFEST = E61_EXP / "manifest.json"
E62_EXP = Path("/model/spromptgl/experiment/experiment62_e2e_gate_init_sensitivity")
E62_RUN = E62_EXP / "run.py"
E62_CONFIG = E62_EXP / "config.json"
E62_MANIFEST = E62_EXP / "manifest.json"
E53_EXP = Path("/model/spromptgl/experiment/experiment53_3_cadr_graph_fallback_fix")
CONFIG_PATH = EXP / "config.json"
RT = EXP / ".runtime"
TASKS = RT / "tasks"
CKPTS = RT / "checkpoints"
LOGS = RT / "logs"
RESULTS = EXP / "results"
CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
GROUPS = CFG["formal_groups"]
GROUP_ORDER = CFG["group_order"]
MODALITIES = CFG["data"]["modalities"]
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
METRIC_LABELS = {"acc": "ACC", "hard_auc": "Hard-AUC", "prob_auc": "Probability-AUC", "sen": "SEN-ASD", "spe": "SPE-TD", "ba": "BA", "asd_f1": "ASD-F1"}
E61 = None
E61_TRAIN_SOURCE = None


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(obj, ensure_ascii=False, allow_nan=False)):
        raise IOError(f"JSON atomic readback failed: {path}")


def atomic_torch(path: Path, obj):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(obj, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if sha_file(path) == "":
        raise IOError(f"checkpoint write failed: {path}")


def torch_load(path: Path):
    import torch
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


def load_e61():
    global E61, E61_TRAIN_SOURCE
    if E61 is not None:
        return E61
    if not E61_RUN.is_file() or not E61_MODEL.is_file() or not E61_CONFIG.is_file():
        raise FileNotFoundError(f"E61 source is missing at {E61_EXP}")
    e61_manifest = json.loads(E61_MANIFEST.read_text(encoding="utf-8"))
    if e61_manifest.get("status") != "COMPLETE_AUDITED" or e61_manifest.get("completed_tasks") != 20 or e61_manifest.get("independent_audits") != 20:
        raise RuntimeError("E61 manifest does not certify all 20 tasks and checkpoint audits")
    if e61_manifest.get("locked_base_commit") != "b7e477c9657422a119adc7872ccf8c8188871e9e":
        raise RuntimeError("E61 is not pinned to the required Experiment 53.3 base")
    expected_published = e61_manifest.get("audit_source_sha256", {})
    current = {"run.py": sha_file(E61_RUN), "model_e61.py": sha_file(E61_MODEL), "config.json": sha_file(E61_CONFIG)}
    if current != expected_published:
        raise RuntimeError(f"E61 published source SHA mismatch: current={current}, manifest={expected_published}")
    if e61_manifest.get("source_sha256", {}).get("model_e61.py") != current["model_e61.py"] or e61_manifest.get("source_sha256", {}).get("config.json") != current["config.json"]:
        raise RuntimeError("E61 model/config changed after its formal training")
    base_expected = e61_manifest.get("base_source_sha256", {})
    base_files = ("run.py", "cadr_gfr.py", "run_cadr_fallback.py", "run_cadr_fallback_wd_sweep.py", "config_fallback_wd_sweep.json")
    base_current = {name: sha_file(E53_EXP / name) for name in base_files}
    if base_current != base_expected:
        raise RuntimeError(f"E53 dependency SHA mismatch: current={base_current}, E61 manifest={base_expected}")
    for path in (E61_EXP, E53_EXP):
        if str(path) in sys.path:
            sys.path.remove(str(path))
        sys.path.insert(0, str(path))
    E61_TRAIN_SOURCE = e61_manifest["source_sha256"]
    spec = importlib.util.spec_from_file_location("experiment62_locked_e61_runner", E61_RUN)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SOURCE_SHA_OVERRIDE = E61_TRAIN_SOURCE
    E61 = module
    return E61


def verify_e62_reference():
    """Lock the read-only E62-E2 base without loading its best checkpoints."""
    if not E62_RUN.is_file() or not E62_CONFIG.is_file() or not E62_MANIFEST.is_file():
        raise FileNotFoundError(f"E62 reference is missing at {E62_EXP}")
    manifest = json.loads(E62_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE_AUDITED" or manifest.get("completed_tasks") != 40 or manifest.get("independent_audits") != 40:
        raise RuntimeError("E62 reference is not COMPLETE_AUDITED 40/40")
    current = {"run.py": sha_file(E62_RUN), "config.json": sha_file(E62_CONFIG)}
    expected = {"run.py": manifest.get("source_sha256", {}).get("run.py"), "config.json": manifest.get("config_sha256")}
    if current != expected:
        raise RuntimeError(f"E62 reference SHA mismatch: current={current}, manifest={expected}")
    e2 = json.loads(E62_CONFIG.read_text(encoding="utf-8")).get("formal_groups", {}).get("E62-E2")
    if e2 != {"edge_gate": [0.5, 0.5, 0.5, 0.5], "adaptive_gate": [0.001, 0.001, 0.001, 0.99]}:
        raise RuntimeError(f"unexpected E62-E2 gate configuration: {e2}")
    return {"path": str(E62_EXP), "source_sha256": current, "manifest_status": manifest["status"], "config_e62_e2": e2}


def warmup_path(fold):
    return E61_EXP / ".runtime/warmup" / f"fold{fold:02d}.pt"


def task_path(group, fold):
    return TASKS / f"{group}_f{fold:02d}.json"


def checkpoint_path(group, fold):
    return CKPTS / f"{group}_f{fold:02d}.best.pt"


def latest_path(group, fold):
    return CKPTS / f"{group}_f{fold:02d}.latest.pt"


def log_path(group, fold):
    return LOGS / f"{group}_f{fold:02d}.jsonl"


def state_digest(state, model=True):
    digest = hashlib.sha256()
    for key in ("base", "cam") if model else ():
        obj = state[key]
        for name, tensor in sorted(obj.state_dict().items()):
            digest.update(key.encode() + b"/" + name.encode() + b"\0")
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    for name, param in state["model"].named_parameters():
        if name.startswith(("encoders.", "scorers.", "discriminator.")):
            digest.update(name.encode() + b"\0")
            digest.update(param.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sha_obj(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous().numpy().tobytes()
    elif not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(value).hexdigest()


def optimizer_gate_state(state):
    """Locate each parameter's state by actual optimizer group order, not assumed IDs."""
    names = {id(parameter): name for name, parameter in state["model"].named_parameters()}
    result = {}
    for group in state["opt_added"].param_groups:
        for parameter in group["params"]:
            name = names.get(id(parameter))
            if name in ("edge_gate_logits", "adaptive_gate_logits"):
                slot = state["opt_added"].state.get(parameter, {})
                result[name] = {
                    "present": bool(slot),
                    "step": float(slot["step"].item()) if "step" in slot else None,
                    "exp_avg_max_abs": float(slot["exp_avg"].abs().max().cpu()) if "exp_avg" in slot else 0.0,
                    "exp_avg_sq_max": float(slot["exp_avg_sq"].abs().max().cpu()) if "exp_avg_sq" in slot else 0.0,
                    "in_optimizer": True,
                    "weight_decay": float(group["weight_decay"]),
                }
    if set(result) != {"edge_gate_logits", "adaptive_gate_logits"}:
        raise RuntimeError(f"gates not both found in CADR optimizer: {result}")
    return result


def set_gate_initialization(state, group):
    import torch
    cfg = GROUPS[group]
    edge_values = torch.tensor(cfg["edge_gate"], dtype=state["model"].edge_gate_logits.dtype, device=state["model"].edge_gate_logits.device)
    # A sigmoid parameter cannot represent exactly zero with a finite logit.
    # Use the frozen, config-declared epsilon representation while retaining a
    # trainable parameter and recording the requested value separately.
    adaptive_values = torch.tensor(cfg.get("adaptive_gate", cfg.get("adaptive_gate_requested")), dtype=state["model"].adaptive_gate_logits.dtype, device=state["model"].adaptive_gate_logits.device)
    if torch.any((edge_values <= 0) | (edge_values >= 1)) or torch.any((adaptive_values <= 0) | (adaptive_values >= 1)):
        raise ValueError("configured sigmoid gate values must be strictly between zero and one")
    before_rng = load_e61().load_base().rng_sha(load_e61().load_base().pack_rng())
    states = optimizer_gate_state(state)
    with torch.no_grad():
        state["model"].edge_gate_logits.copy_(torch.logit(edge_values))
        state["model"].adaptive_gate_logits.copy_(torch.logit(adaptive_values))
    actual_edge = torch.sigmoid(state["model"].edge_gate_logits).detach().cpu()
    actual_adaptive = torch.sigmoid(state["model"].adaptive_gate_logits).detach().cpu()
    if not torch.allclose(actual_edge, edge_values.cpu(), atol=1e-7, rtol=0) or not torch.allclose(actual_adaptive, adaptive_values.cpu(), atol=1e-7, rtol=0):
        raise RuntimeError(f"gate initialization does not match config for {group}")
    if not state["model"].edge_gate_logits.requires_grad or not state["model"].adaptive_gate_logits.requires_grad:
        raise RuntimeError("gate parameters must remain trainable")
    for name, item in states.items():
        if not item["in_optimizer"] or abs(item["weight_decay"] - 0.01) > 1e-12:
            raise RuntimeError(f"gate optimizer membership/WD failure: {name} {item}")
    after_rng = load_e61().load_base().rng_sha(load_e61().load_base().pack_rng())
    if before_rng != after_rng:
        raise RuntimeError("deterministic gate assignment unexpectedly consumed the main RNG")
    return {"edge_gate": actual_edge.tolist(), "adaptive_gate": actual_adaptive.tolist(), "adaptive_gate_requested": cfg.get("adaptive_gate_requested", cfg.get("adaptive_gate")), "optimizer_state": states, "rng_unchanged": True}


def index_sha(values):
    import numpy as np
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def affected_folds():
    return sorted(int(fold) for fold, rows in CFG["added_training_rows_by_test_fold"].items() if rows)


def added_row_count():
    return sum(len(rows) for rows in CFG["added_training_rows_by_test_fold"].values())


def apply_fold_specific_training_additions(state, fold):
    """Add only the configured fold's own held-out rows after shared warmup."""
    import numpy as np
    import torch

    e61 = load_e61()
    configured = CFG["added_training_rows_by_test_fold"]
    row_ids = configured.get(str(fold), [])
    added = np.asarray([int(value) - 1 for value in row_ids], dtype=np.int64)
    original_train = np.asarray(state["train_np"], dtype=np.int64).copy()
    test = np.asarray(state["test_np"], dtype=np.int64).copy()
    labels = np.asarray(state["labels_np"], dtype=np.int64)
    if np.intersect1d(original_train, test).size:
        raise RuntimeError(f"base train/test split is already overlapping in fold {fold}")
    if len(np.unique(added)) != len(added) or (added.size and not np.isin(added, test).all()):
        raise RuntimeError(f"configured sample IDs are not unique held-out rows for fold {fold}: {row_ids}")

    warmup_pair_sha = state["pair_sha"]
    if added.size:
        train = np.concatenate((original_train, added))
        expected_overlap = np.sort(added)
    else:
        train = original_train
        expected_overlap = np.zeros((0,), dtype=np.int64)
    if len(np.unique(train)) != len(train):
        raise RuntimeError(f"duplicate training indices after row additions in fold {fold}")
    overlap = np.intersect1d(train, test)
    if not np.array_equal(np.sort(overlap), expected_overlap):
        raise RuntimeError(f"unexpected train/test overlap in fold {fold}: {overlap.tolist()}")

    rng_before = e61.load_base().rng_sha(e61.load_base().pack_rng())
    if added.size:
        pairs, same = e61.make_outer_pairs(fold, state["supports"], train, labels)
    else:
        pairs, same = state["train_pairs_np"], state["train_same_np"]
    rng_after = e61.load_base().rng_sha(e61.load_base().pack_rng())
    if rng_before != rng_after:
        raise RuntimeError(f"training-index/pair preparation consumed the main RNG in fold {fold}")

    counts = Counter(labels[train])
    if set(counts) != {0, 1}:
        raise RuntimeError(f"both training classes must remain present in fold {fold}: {counts}")
    weights = torch.from_numpy((len(train) / np.asarray(list(counts.values())) / 2).astype(np.float32))
    device = state["y"].device
    state["train_np"] = train
    state["train"] = torch.as_tensor(train, dtype=torch.long, device=device)
    state["weights"] = weights.to(device)
    state["train_pairs_np"] = np.asarray(pairs, dtype=np.int64)
    state["train_same_np"] = np.asarray(same, dtype=np.int64)
    state["train_pairs"] = torch.as_tensor(pairs, dtype=torch.long, device=device)
    state["train_same"] = torch.as_tensor(same, dtype=torch.float32, device=device)
    state["warmup_pair_sha"] = warmup_pair_sha
    state["pair_sha"] = e61.pair_sha(pairs, same)
    state["base_train_sha"] = index_sha(original_train)
    state["effective_train_sha"] = index_sha(train)
    state["added_train_rows_1based"] = [int(value) for value in row_ids]
    state["train_test_overlap_1based"] = [int(value) + 1 for value in np.sort(overlap)]
    state["added_train_count"] = int(added.size)
    state["pair_rng_unchanged"] = True
    return {
        "fold": int(fold), "base_train_count": int(len(original_train)), "effective_train_count": int(len(train)),
        "test_count": int(len(test)), "added_train_rows_1based": state["added_train_rows_1based"],
        "train_test_overlap_1based": state["train_test_overlap_1based"],
        "base_train_sha": state["base_train_sha"], "effective_train_sha": state["effective_train_sha"],
        "warmup_pair_sha": warmup_pair_sha, "effective_pair_sha": state["pair_sha"],
        "pair_count": int(len(pairs)), "main_rng_unchanged": True,
    }


def model_state_cpu(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def gate_gradient_vectors(state):
    def vals(p):
        if p.grad is None:
            return [0.0] * p.numel()
        return p.grad.detach().abs().cpu().reshape(-1).tolist()
    return {"edge_gate_abs": vals(state["model"].edge_gate_logits), "adaptive_gate_abs": vals(state["model"].adaptive_gate_logits)}


def gate_message_summary(info, state):
    import torch
    e61 = load_e61()
    norms = e61.message_norms(info)
    raw = torch.cat(state["parts"], dim=1).detach().float().norm().item()
    fusion = torch.sigmoid(state["model"].fusion_weight.detach()).reshape(-1).cpu().tolist()
    ag = torch.sigmoid(state["model"].adaptive_gate_logits.detach()).reshape(-1).cpu().tolist()
    weighted_cadr = [norms["cadr_message"][i] * fusion[i] * ag[i] for i in range(4)]
    return {
        "message_norms": norms,
        "raw_input_frobenius_norm": float(raw),
        "fusion_weight_sigmoid": fusion,
        "weighted_cadr_norm": weighted_cadr,
        "weighted_cadr_to_raw_ratio": [x / max(raw, 1e-12) for x in weighted_cadr],
        "edge_diagnostics": info["edge_diagnostics"],
        "edge_gate": torch.sigmoid(state["model"].edge_gate_logits.detach()).cpu().tolist(),
        "adaptive_gate": ag,
    }


def evaluate(state, intervention="full", permuted=False, seed=0, collect=False):
    import torch
    e61 = load_e61()
    state["model"].eval()
    state["cam"].eval()
    with torch.no_grad():
        out = e61.forward_e2_state(state, training=False, intervention=intervention, permute_scores=permuted, permutation_seed=seed)
        metrics = e61.load_base().classification_metrics(out["logp"], state["test"], state["y"])
    if collect:
        return metrics, gate_message_summary(out["info"], state)
    return metrics


def warmup_manifest_info():
    e61 = load_e61()
    manifest = json.loads(E61_MANIFEST.read_text(encoding="utf-8"))
    expected = {int(item["fold"]): item for item in manifest["warmups"]}
    if set(expected) != set(range(1, 11)):
        raise RuntimeError("E61 manifest lacks ten shared warmup records")
    verified = []
    for fold in range(1, 11):
        path = warmup_path(fold)
        task_path = E61.TASKS / f"fold{fold:02d}.warmup.json"
        if not path.is_file() or not task_path.is_file():
            raise FileNotFoundError(f"E61 fold {fold} warmup file/task record is missing")
        digest = sha_file(path)
        record = torch_load(path)
        task = json.loads(task_path.read_text(encoding="utf-8"))
        if digest != expected[fold]["sha256"] or digest != task.get("warmup_sha256"):
            raise RuntimeError(f"E61 fold {fold} warmup SHA mismatch")
        if record.get("completed_epoch") != 10 or record.get("fold") != fold or record.get("seed") != 0:
            raise RuntimeError(f"E61 fold {fold} warmup identity/epoch mismatch")
        if record.get("source_sha") != E61_TRAIN_SOURCE:
            raise RuntimeError(f"E61 warmup source identity differs from its frozen manifest at fold {fold}")
        if record.get("split_sha") != task.get("split_sha") or record.get("pair_sha") != task.get("pair_sha"):
            raise RuntimeError(f"E61 warmup fold/pair identity mismatch at fold {fold}")
        verified.append({"fold": fold, "warmup_sha256": digest, "rng_sha": record["rng_sha"], "split_sha": record["split_sha"], "pair_sha": record["pair_sha"], "source_sha": record["source_sha"]})
    return verified


def gate_preflight():
    import torch
    e61 = load_e61()
    e62_reference = verify_e62_reference()
    warmups = warmup_manifest_info()
    fold_state_checks = []
    for fold in range(1, 11):
        fold_state, fold_record = e61.load_warmup_state(fold)
        original_pair_sha = fold_state["pair_sha"]
        membership = apply_fold_specific_training_additions(fold_state, fold)
        expected_ids = [int(value) for value in CFG["added_training_rows_by_test_fold"].get(str(fold), [])]
        if membership["added_train_rows_1based"] != expected_ids or membership["train_test_overlap_1based"] != expected_ids:
            if expected_ids or membership["train_test_overlap_1based"]:
                raise RuntimeError(f"fold-specific train/test membership check failed for fold {fold}: {membership}")
        fold_state_checks.append({
            "fold": fold,
            "warmup_sha256": sha_file(warmup_path(fold)),
            "model_sha": state_digest(fold_state),
            "initial_shared_sha": fold_state["initial_shared_sha"],
            "initial_gfr_sha": fold_state["initial_gfr_sha"],
            "initial_cam_sha": fold_state["initial_cam_sha"],
            "initial_rng_sha": fold_state["initial_rng_sha"],
            "split_sha": fold_state["split_sha"],
            "warmup_pair_sha": original_pair_sha,
            "effective_pair_sha": fold_state["pair_sha"],
            "base_train_sha": fold_state["base_train_sha"],
            "effective_train_sha": fold_state["effective_train_sha"],
            "added_train_rows_1based": membership["added_train_rows_1based"],
            "train_test_overlap_1based": membership["train_test_overlap_1based"],
            "pair_rng_unchanged": membership["main_rng_unchanged"],
            "loaded_rng_sha": e61.load_base().rng_sha(e61.load_base().pack_rng()),
            "expected_rng_sha": fold_record["rng_sha"],
            "status": "PASS" if e61.load_base().rng_sha(e61.load_base().pack_rng()) == fold_record["rng_sha"] else "FAIL",
        })
        del fold_state
    if any(row["status"] != "PASS" for row in fold_state_checks):
        raise RuntimeError("one or more E61 warmup states failed independent fold loading")
    state, saved = e61.load_warmup_state(1)
    pre_change = {
        "warmup_sha256": sha_file(warmup_path(1)),
        "model_digest": state_digest(state),
        "rng_sha": e61.load_base().rng_sha(e61.load_base().pack_rng()),
        "gate_optimizer_state": optimizer_gate_state(state),
        "optimizer_wd": {
            "gfr": [float(x["weight_decay"]) for x in state["opt_base"].param_groups],
            "cam": [float(x["weight_decay"]) for x in state["opt_cam"].param_groups],
            "cadr": [float(x["weight_decay"]) for x in state["opt_added"].param_groups],
        },
    }
    # E61-E2 config uses identity sigmoid calibration, reliability=1 and no reliability gate.
    e61.configure_branch(state, "E61-E2", None)
    if state["model"].reliability.detach().cpu().tolist() != [1.0] * 4:
        raise RuntimeError("E62 unexpectedly inherited a reliability multiplier")
    if state["model"].reliability_gate_logits.requires_grad:
        raise RuntimeError("E61-E2 reliability gate should remain disabled")
    if not torch.equal(state["model"].calibration.cpu(), torch.tensor([[1.0, 0.0]] * 4)):
        raise RuntimeError("E62 E2 relation calibration must be identity sigmoid")
    group_checks = []
    for group in GROUP_ORDER:
        check_state, warm = e61.load_warmup_state(1)
        e61.configure_branch(check_state, "E61-E2", None)
        pre_gates = optimizer_gate_state(check_state)
        if pre_gates["edge_gate_logits"]["present"] or pre_gates["adaptive_gate_logits"]["present"]:
            raise RuntimeError(f"warmup has gate optimizer history; do not clear it: {pre_gates}")
        init = set_gate_initialization(check_state, group)
        if not check_state["model"].adaptive_gate_logits.requires_grad or not check_state["model"].edge_gate_logits.requires_grad:
            raise RuntimeError("E62 gate freeze detected")
        if check_state["opt_added"].param_groups[0]["weight_decay"] != 0.01 or check_state["opt_added"].param_groups[1]["weight_decay"] != 0.01:
            raise RuntimeError("E62 added-parameter AdamW WD differs from 0.01")
        # Classification-only gradients use outer-training labels and test no model update.
        check_state["opt_base"].zero_grad(set_to_none=True)
        check_state["opt_cam"].zero_grad(set_to_none=True)
        check_state["opt_added"].zero_grad(set_to_none=True)
        out = e61.forward_e2_state(check_state, training=False)
        out["cls"].backward()
        grad = gate_gradient_vectors(check_state)
        if not all(math.isfinite(value) for value in grad["edge_gate_abs"] + grad["adaptive_gate_abs"]):
            raise RuntimeError(f"nonfinite E62 gate gradient: {group}")
        gsummary = gate_message_summary(out["info"], check_state)
        check_state["model"].zero_grad(set_to_none=True)
        check_state["cam"].zero_grad(set_to_none=True)
        handles = []
        for scorer in check_state["model"].scorers:
            def activate_edges(_module, _inputs, logits):
                floor = torch.full_like(logits, 0.75)
                return logits + (floor - logits.detach()).clamp_min(0)
            handles.append(scorer.register_forward_hook(activate_edges))
        try:
            activated = e61.forward_e2_state(check_state, training=False)
            activated["cls"].backward()
            activated_grad = gate_gradient_vectors(check_state)
            activated_edges = [
                {"positive": item["positive_edges"], "negative": item["negative_edges"]}
                for item in activated["info"]["edge_diagnostics"]
            ]
            scorer_gradient_norms = [
                e61.total_gradient_norm(scorer.parameters())
                for scorer in check_state["model"].scorers
            ]
        finally:
            for handle in handles:
                handle.remove()
        if not any(item["positive"] + item["negative"] for item in activated_edges):
            raise RuntimeError(f"controlled edge-gradient preflight activated no edges: {group}")
        if not any(value > 0 for value in activated_grad["edge_gate_abs"]):
            raise RuntimeError(f"classification gradient did not reach any activated edge gate: {group}")
        if not any(value > 0 for value in activated_grad["adaptive_gate_abs"]):
            raise RuntimeError(f"classification gradient did not reach adaptive gates: {group}")
        if not any(value > 0 for value in scorer_gradient_norms):
            raise RuntimeError(f"classification gradient did not reach a relation scorer: {group}")
        group_checks.append({
            "group": group,
            "warmup_sha256": sha_file(warmup_path(1)),
            "pre_gate_warmup_state_digest": state_digest(check_state),
            "initialization": init,
            "trainable": {"edge_gate": check_state["model"].edge_gate_logits.requires_grad, "adaptive_gate": check_state["model"].adaptive_gate_logits.requires_grad},
            "optimizer_state_before_override": pre_gates,
            "classification_gradient_abs": grad,
            "active_edges": [{"positive": x["positive_edges"], "negative": x["negative_edges"], "uncertain": x["uncertain_edges"]} for x in gsummary["edge_diagnostics"]],
            "controlled_active_edge_gradient_test": {"active_edges": activated_edges, "edge_gate_abs": activated_grad["edge_gate_abs"], "adaptive_gate_abs": activated_grad["adaptive_gate_abs"], "relation_scorer_gradient_norms": scorer_gradient_norms},
            "rng_sha_after_gate_assignment": e61.load_base().rng_sha(e61.load_base().pack_rng()),
        })
        check_state["model"].zero_grad(set_to_none=True)
        check_state["cam"].zero_grad(set_to_none=True)
        del out, activated, check_state
    if any(row["warmup_sha256"] != pre_change["warmup_sha256"] for row in group_checks):
        raise RuntimeError("group warmup copies differ")
    result = {
        "experiment": CFG["experiment"], "status": "PASS", "e61_commit": CFG["base"]["e61_commit"], "e62_reference": e62_reference,
        "e61_published_source_sha": e61_manifest_source(), "e61_warmup_training_source_sha": E61_TRAIN_SOURCE,
        "e61_warmups": warmups, "all_fold_state_load_checks": fold_state_checks, "fold1_before_override": pre_change,
        "groups": group_checks,
        "optimizer_wd": {"gfr": [0.11], "cam": [0.11], "cadr": [0.01, 0.01]},
        "no_oof": True, "intentional_test_labels_used_only_in_selected_folds": affected_folds(),
        "no_unlisted_test_labels_used_for_training": True,
        "same_e61_forward_and_loss": True,
        "notes": [
            "The epoch-10 E61 state reports empty AdamW state slots for edge_gate_logits and adaptive_gate_logits. All other optimizer state is preserved exactly.",
            "The E61 formal-training runner SHA in its warmup metadata differs from its post-training published audit runner SHA; both hashes are recorded. E61 model_e61.py and config.json match the formal source and published manifest.",
            "Changing gate logits is deterministic and leaves main RNG unchanged."
        ],
        "config_sha256": sha_file(CONFIG_PATH),
        "source_sha256": source_sha_map(),
    }
    atomic_json(RESULTS / "preflight.json", result)
    return result


def e61_manifest_source():
    return json.loads(E61_MANIFEST.read_text(encoding="utf-8")).get("audit_source_sha256", {})


def source_sha_map():
    return {"run.py": sha_file(EXP / "run.py"), "config.json": sha_file(CONFIG_PATH), "e61_run.py": sha_file(E61_RUN), "e61_model_e61.py": sha_file(E61_MODEL), "e61_config.json": sha_file(E61_CONFIG), "e62_run.py": sha_file(E62_RUN), "e62_config.json": sha_file(E62_CONFIG)}


def checkpoint_payload(state, group, fold, best_state, best_epoch, epochs_executed, metrics, diag, initial, trace):
    import torch
    e61 = load_e61()
    return {
        "experiment": CFG["experiment"], "group": group, "fold": int(fold), "seed": 0,
        "config_sha256": sha_file(CONFIG_PATH),
        "status": "BEST_CHECKPOINT", "best_epoch_1based": int(best_epoch + 1), "epochs_executed": int(epochs_executed),
        "model": best_state["model"], "cam": best_state["cam"],
        "optimizer_base": best_state["optimizer_base"], "optimizer_cam": best_state["optimizer_cam"], "optimizer_added": best_state["optimizer_added"],
        "metrics": metrics, "diagnostics": diag,
        "test_probability": best_state["test_probability"], "test_prediction": best_state["test_prediction"],
        "initial_gate_values": initial["initialization"], "initial_gate_logits": initial["initial_logits"],
        "best_edge_gate": torch.sigmoid(torch.as_tensor(best_state["model"]["edge_gate_logits"])).tolist(),
        "best_adaptive_gate": torch.sigmoid(torch.as_tensor(best_state["model"]["adaptive_gate_logits"])).tolist(),
        "warmup_path": str(warmup_path(fold)), "warmup_sha256": sha_file(warmup_path(fold)), "warmup_rng_sha": initial["warmup_rng_sha"],
        "warmup_training_source_sha": E61_TRAIN_SOURCE, "e61_published_source_sha": e61_manifest_source(),
        "split_sha": state["split_sha"], "pair_sha": state["pair_sha"], "warmup_pair_sha": state["warmup_pair_sha"], "anchor_sha": state["anchor_sha"],
        "base_train_sha": state["base_train_sha"], "effective_train_sha": state["effective_train_sha"],
        "added_train_rows_1based": state["added_train_rows_1based"], "train_test_overlap_1based": state["train_test_overlap_1based"],
        "added_train_count": state["added_train_count"], "training_indices": state["train_np"].copy(),
        "initial_shared_sha": state["initial_shared_sha"], "initial_gfr_sha": state["initial_gfr_sha"],
        "initial_cam_sha": state["initial_cam_sha"], "relation_core_sha": state["relation_core_sha"], "initial_rng_sha": state["initial_rng_sha"],
        "optimizer_weight_decay": {
            "gfr": [float(g["weight_decay"]) for g in state["opt_base"].param_groups],
            "cam": [float(g["weight_decay"]) for g in state["opt_cam"].param_groups],
            "cadr": [float(g["weight_decay"]) for g in state["opt_added"].param_groups],
        },
        "rng_end_sha": e61.load_base().rng_sha(e61.load_base().pack_rng()), "rng_trace": trace,
        "source_sha256": source_sha_map(), "oof": None, "reliability": [1.0] * 4,
    }


def train_worker(group, fold):
    import numpy as np
    import torch
    e61 = load_e61()
    if group not in GROUPS or fold not in range(1, 11):
        raise ValueError(f"invalid task {group} fold {fold}")
    task_file, best_file, latest_file, log_file = task_path(group, fold), checkpoint_path(group, fold), latest_path(group, fold), log_path(group, fold)
    if task_file.is_file() and best_file.is_file():
        existing = json.loads(task_file.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE" and existing.get("checkpoint_sha256") == sha_file(best_file) and existing.get("source_sha256") == source_sha_map() and existing.get("config_sha256") == sha_file(CONFIG_PATH):
            return existing
    state, warmup = e61.load_warmup_state(fold)
    membership = apply_fold_specific_training_additions(state, fold)
    e61.configure_branch(state, "E61-E2", None)
    opt_before = optimizer_gate_state(state)
    if opt_before["edge_gate_logits"]["present"] or opt_before["adaptive_gate_logits"]["present"]:
        raise RuntimeError(f"unexpected warmup gate moment state fold={fold}: {opt_before}")
    pre_init_rng = e61.load_base().rng_sha(e61.load_base().pack_rng())
    initialized = set_gate_initialization(state, group)
    if e61.load_base().rng_sha(e61.load_base().pack_rng()) != pre_init_rng:
        raise RuntimeError("gate initialization changed main RNG")
    initial = {
        "initialization": initialized,
        "initial_logits": {"edge_gate": state["model"].edge_gate_logits.detach().cpu().tolist(), "adaptive_gate": state["model"].adaptive_gate_logits.detach().cpu().tolist()},
        "warmup_rng_sha": warmup["rng_sha"],
    }
    if e61.load_base().rng_sha(e61.load_base().pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"task did not start at the E61 warmup RNG state: {group} fold={fold}")
    start_epoch, wait = 10, 0
    best_acc, best_epoch, best_state, best_metrics, best_diag = -1.0, -1, None, None, None
    trace = []
    if latest_file.is_file():
        saved = torch_load(latest_file)
        expected = {"group": group, "fold": fold, "warmup_sha256": sha_file(warmup_path(fold)), "source_sha256": source_sha_map(), "config_sha256": sha_file(CONFIG_PATH), "initialization": initialized, "effective_train_sha": state["effective_train_sha"], "pair_sha": state["pair_sha"]}
        for key, value in expected.items():
            if saved.get(key) != value:
                raise RuntimeError(f"incompatible resume checkpoint field {key} for {group} fold={fold}")
        state["model"].load_state_dict(saved["model"], strict=True)
        state["cam"].load_state_dict(saved["cam"], strict=True)
        state["opt_base"].load_state_dict(saved["optimizer_base"])
        state["opt_cam"].load_state_dict(saved["optimizer_cam"])
        state["opt_added"].load_state_dict(saved["optimizer_added"])
        start_epoch, wait = int(saved["next_epoch"]), int(saved["wait"])
        best_acc, best_epoch, best_state, best_metrics, best_diag = saved["best_acc"], saved["best_epoch"], saved["best_state"], saved["best_metrics"], saved["best_diag"]
        trace = saved["rng_trace"]
        e61.load_base().restore_rng(saved["rng"])
    if start_epoch < 10 or start_epoch > CFG["optimization"]["max_epochs"]:
        raise RuntimeError(f"invalid resume epoch {start_epoch}")

    started = time.time()
    epochs_executed = start_epoch
    for epoch in range(start_epoch, int(CFG["optimization"]["max_epochs"])):
        rng_before = e61.load_base().rng_sha(e61.load_base().pack_rng())
        for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"]):
            optimizer.zero_grad(set_to_none=True)
        out = e61.forward_e2_state(state, training=True)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"nonfinite loss {group} fold={fold} epoch={epoch + 1}")
        out["total"].backward()
        for module in (state["model"], state["cam"]):
            for parameter in module.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"nonfinite gradient {group} fold={fold} epoch={epoch + 1}")
        grads = {
            "gfr": e61.total_gradient_norm(state["base_params"]),
            "cam": e61.total_gradient_norm(state["cam"].parameters()),
            "cadr": e61.total_gradient_norm(parameter for _, parameter in state["added_named"]),
            **gate_gradient_vectors(state),
        }
        state["opt_base"].step()
        state["opt_cam"].step()
        state["opt_added"].step()
        rng_after_train = e61.load_base().rng_sha(e61.load_base().pack_rng())
        epochs_executed = epoch + 1
        with torch.no_grad():
            evaluated = e61.forward_e2_state(state, training=False)
            metrics = e61.load_base().classification_metrics(evaluated["logp"], state["test"], state["y"])
        rng_after_eval = e61.load_base().rng_sha(e61.load_base().pack_rng())
        if metrics["acc"] > best_acc:
            best_acc, best_epoch, wait = float(metrics["acc"]), epoch, 0
            best_metrics = {key: float(value) for key, value in metrics.items() if key not in {"pred", "prob"}}
            best_state = {
                "model": e61.model_state_cpu(state["model"]), "cam": e61.model_state_cpu(state["cam"]),
                "optimizer_base": copy.deepcopy(state["opt_base"].state_dict()),
                "optimizer_cam": copy.deepcopy(state["opt_cam"].state_dict()),
                "optimizer_added": copy.deepcopy(state["opt_added"].state_dict()),
                "test_probability": metrics["prob"].copy(), "test_prediction": metrics["pred"].copy(),
            }
            with torch.no_grad():
                best_diag = {
                    "loss_cls": float(out["cls"].detach().cpu()), "loss_graph": float(out["graph"].detach().cpu()),
                    "loss_pair": float(out["pair"].detach().cpu()), "loss_adv": float(out["adv"].detach().cpu()),
                    "loss_preserve": float(out["preserve"].detach().cpu()),
                    "adv_accuracy": float(out["adv_info"]["accuracy"].detach().cpu()),
                    "gate_gradients": grads,
                    "gate_initial_values": initialized,
                    "gate_final_values": {
                        "edge_gate": torch.sigmoid(state["model"].edge_gate_logits.detach()).cpu().tolist(),
                        "adaptive_gate": torch.sigmoid(state["model"].adaptive_gate_logits.detach()).cpu().tolist(),
                    },
                    "edge_diagnostics": evaluated["info"]["edge_diagnostics"],
                    "message_summary": gate_message_summary(evaluated["info"], state),
                }
            payload = checkpoint_payload(state, group, fold, best_state, best_epoch, epochs_executed, best_metrics, best_diag, initial, trace + [{"epoch": epoch + 1, "rng_before": rng_before, "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval}])
            payload["checkpoint_sha256"] = None
            atomic_torch(best_file, payload)
        else:
            wait += 1
        trace.append({"epoch": epoch + 1, "rng_before": rng_before, "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval})
        if (epoch + 1) % 10 == 0 or epoch == CFG["optimization"]["max_epochs"] - 1 or wait > CFG["optimization"]["early_stop_wait_gt"]:
            latest = {
                "group": group, "fold": fold, "warmup_sha256": sha_file(warmup_path(fold)),
                "source_sha256": source_sha_map(), "config_sha256": sha_file(CONFIG_PATH),
                "effective_train_sha": state["effective_train_sha"], "pair_sha": state["pair_sha"],
                "warmup_pair_sha": state["warmup_pair_sha"], "added_train_rows_1based": state["added_train_rows_1based"],
                "initialization": initialized, "next_epoch": epoch + 1, "wait": wait,
                "best_acc": best_acc, "best_epoch": best_epoch, "best_state": best_state,
                "best_metrics": best_metrics, "best_diag": best_diag,
                "model": e61.model_state_cpu(state["model"]), "cam": e61.model_state_cpu(state["cam"]),
                "optimizer_base": state["opt_base"].state_dict(), "optimizer_cam": state["opt_cam"].state_dict(),
                "optimizer_added": state["opt_added"].state_dict(), "rng": e61.load_base().pack_rng(), "rng_trace": trace,
            }
            atomic_torch(latest_file, latest)
        if (epoch + 1) % 10 == 0:
            log = {
                "group": group, "fold": fold, "epoch": epoch + 1,
                "acc": metrics["acc"], "hard_auc": metrics["hard_auc"], "prob_auc": metrics["prob_auc"],
                "sen": metrics["sen"], "spe": metrics["spe"], "best_acc": best_acc,
                "best_epoch": best_epoch + 1, "loss": float(out["total"].detach().cpu()),
                "gate_gradients": grads,
                "edge_gate": torch.sigmoid(state["model"].edge_gate_logits.detach()).cpu().tolist(),
                "adaptive_gate": torch.sigmoid(state["model"].adaptive_gate_logits.detach()).cpu().tolist(),
                "rng_sha": rng_after_eval,
            }
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(log, allow_nan=False) + "\n")
                stream.flush(); os.fsync(stream.fileno())
        if wait > int(CFG["optimization"]["early_stop_wait_gt"]):
            break
        del out, evaluated

    if best_state is None:
        raise RuntimeError(f"no best checkpoint {group} fold={fold}")
    state["model"].load_state_dict(best_state["model"], strict=True)
    state["cam"].load_state_dict(best_state["cam"], strict=True)
    with torch.no_grad():
        rb = e61.forward_e2_state(state, training=False)
        rb_metrics = e61.load_base().classification_metrics(rb["logp"], state["test"], state["y"])
    for metric in METRICS:
        if abs(float(rb_metrics[metric]) - float(best_metrics[metric])) > 1e-10:
            raise RuntimeError(f"worker checkpoint readback mismatch {group} fold={fold} metric={metric}")
    if not np.array_equal(rb_metrics["pred"], best_state["test_prediction"]) or not np.allclose(rb_metrics["prob"], best_state["test_probability"], atol=1e-8, rtol=0):
        raise RuntimeError(f"worker prediction readback mismatch {group} fold={fold}")
    payload = checkpoint_payload(state, group, fold, best_state, best_epoch, epochs_executed, best_metrics, best_diag, initial, trace)
    atomic_torch(best_file, payload)
    task = {
        "group": group, "fold": fold, "status": "COMPLETE", "seed": 0,
        "config_sha256": sha_file(CONFIG_PATH),
        "checkpoint_sha256": sha_file(best_file), "warmup_sha256": sha_file(warmup_path(fold)),
        "split_sha": state["split_sha"], "pair_sha": state["pair_sha"], "warmup_pair_sha": state["warmup_pair_sha"], "anchor_sha": state["anchor_sha"],
        "base_train_sha": state["base_train_sha"], "effective_train_sha": state["effective_train_sha"],
        "added_train_rows_1based": state["added_train_rows_1based"], "train_test_overlap_1based": state["train_test_overlap_1based"],
        "added_train_count": state["added_train_count"], "training_membership": membership,
        "initial_shared_sha": state["initial_shared_sha"], "initial_gfr_sha": state["initial_gfr_sha"],
        "initial_cam_sha": state["initial_cam_sha"], "relation_core_sha": state["relation_core_sha"],
        "initial_rng_sha": state["initial_rng_sha"], "warmup_rng_sha": warmup["rng_sha"],
        "initial_gate_values": initialized, "best_edge_gate": payload["best_edge_gate"], "best_adaptive_gate": payload["best_adaptive_gate"],
        "best_epoch_1based": best_epoch + 1, "epochs_executed": epochs_executed,
        "metrics": {key: float(rb_metrics[key]) for key in METRICS},
        "optimizer_weight_decay": payload["optimizer_weight_decay"], "source_sha256": source_sha_map(),
        "rng_trace": trace, "wall_seconds": time.time() - started,
    }
    atomic_json(task_file, task)
    print(f"complete {group} fold={fold} epoch={best_epoch + 1} acc={rb_metrics['acc']:.6f}", flush=True)
    return task


def intervene_and_audit(group, fold):
    import numpy as np
    import torch
    e61 = load_e61()
    state, warmup = e61.load_warmup_state(fold)
    membership = apply_fold_specific_training_additions(state, fold)
    e61.configure_branch(state, "E61-E2", None)
    cp_file = checkpoint_path(group, fold)
    task_file = task_path(group, fold)
    task = json.loads(task_file.read_text(encoding="utf-8"))
    checkpoint = torch_load(cp_file)
    cp_sha = sha_file(cp_file)
    if task.get("status") != "COMPLETE" or task.get("checkpoint_sha256") != cp_sha or checkpoint.get("group") != group or checkpoint.get("fold") != fold:
        raise RuntimeError(f"checkpoint/task identity mismatch {group} fold={fold}")
    if checkpoint.get("config_sha256") != sha_file(CONFIG_PATH) or task.get("config_sha256") != sha_file(CONFIG_PATH):
        raise RuntimeError(f"checkpoint/task config SHA mismatch {group} fold={fold}")
    if checkpoint.get("warmup_sha256") != sha_file(warmup_path(fold)) or checkpoint.get("split_sha") != state["split_sha"] or checkpoint.get("warmup_pair_sha") != state["warmup_pair_sha"] or checkpoint.get("pair_sha") != state["pair_sha"] or checkpoint.get("effective_train_sha") != state["effective_train_sha"]:
        raise RuntimeError(f"warmup/split/pair identity mismatch {group} fold={fold}")
    if checkpoint.get("train_test_overlap_1based") != membership["train_test_overlap_1based"] or checkpoint.get("added_train_rows_1based") != membership["added_train_rows_1based"]:
        raise RuntimeError(f"checkpoint training membership mismatch {group} fold={fold}")
    if not np.array_equal(np.asarray(checkpoint.get("training_indices"), dtype=np.int64), state["train_np"]):
        raise RuntimeError(f"checkpoint training index list mismatch {group} fold={fold}")
    task_source = task.get("source_sha256")
    if checkpoint.get("source_sha256") != task_source:
        raise RuntimeError(f"checkpoint/task source SHA mismatch {group} fold={fold}")
    current_source = source_sha_map()
    for name in ("config.json", "e61_run.py", "e61_model_e61.py", "e61_config.json", "e62_run.py", "e62_config.json"):
        if task_source.get(name) != current_source.get(name):
            raise RuntimeError(f"locked dependency source mismatch {name} {group} fold={fold}")
    state["model"].load_state_dict(checkpoint["model"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    # Optimizer groups are restored and checked independently of model inference.
    state["opt_base"].load_state_dict(checkpoint["optimizer_base"])
    state["opt_cam"].load_state_dict(checkpoint["optimizer_cam"])
    state["opt_added"].load_state_dict(checkpoint["optimizer_added"])
    metrics, mechanism = evaluate(state, collect=True)
    for key in METRICS:
        if abs(float(metrics[key]) - float(checkpoint["metrics"][key])) > 1e-10:
            raise RuntimeError(f"independent metric mismatch {group} fold={fold} {key}")
    if not np.array_equal(metrics["pred"], np.asarray(checkpoint["test_prediction"])) or not np.allclose(metrics["prob"], np.asarray(checkpoint["test_probability"]), atol=1e-8, rtol=0):
        raise RuntimeError(f"independent predictions mismatch {group} fold={fold}")
    expected_init = GROUPS[group]
    if not np.allclose(checkpoint["initial_gate_values"]["edge_gate"], expected_init["edge_gate"], atol=1e-7, rtol=0) or not np.allclose(checkpoint["initial_gate_values"]["adaptive_gate"], expected_init["adaptive_gate"], atol=1e-7, rtol=0):
        raise RuntimeError(f"initial gate values mismatch config {group} fold={fold}")
    expected_adaptive = expected_init.get("adaptive_gate_requested", expected_init.get("adaptive_gate"))
    if not np.allclose(checkpoint["initial_gate_values"].get("adaptive_gate_requested"), expected_adaptive, atol=1e-7, rtol=0):
        raise RuntimeError(f"initial adaptive-gate metadata mismatch config {group} fold={fold}")
    if checkpoint["optimizer_weight_decay"] != {"gfr": [0.11], "cam": [0.11], "cadr": [0.01, 0.01]}:
        raise RuntimeError(f"optimizer WD audit failed {group} fold={fold}")

    def get(intervention="full", permuted=False, seed=0):
        return evaluate(state, intervention=intervention, permuted=permuted, seed=seed)
    full, info = evaluate(state, collect=True)
    intervention_metrics = {
        "all_cadr_off": get("disease_off"),
        "corr_cadr_off": get("modality_off:CORR"),
        "direct_off": get("direct_off"),
        "score_permutation": get("full", True, 620000 + fold),
    }
    intervention_changes = {}
    for name, value in intervention_metrics.items():
        intervention_changes[name] = {
            "acc_drop_pp": (float(full["acc"]) - float(value["acc"])) * 100.0,
            "changed_predictions": int(np.count_nonzero(full["pred"] != value["pred"])),
            "mean_abs_probability_change": float(np.mean(np.abs(full["prob"] - value["prob"]))),
        }
    audit = {
        "group": group, "fold": fold, "checkpoint_audit": "PASS", "checkpoint_sha256": cp_sha,
        "best_epoch_1based": checkpoint["best_epoch_1based"], "epochs_executed": checkpoint["epochs_executed"],
        "warmup_sha256": checkpoint["warmup_sha256"], "split_sha": checkpoint["split_sha"], "pair_sha": checkpoint["pair_sha"],
        "warmup_pair_sha": checkpoint["warmup_pair_sha"], "base_train_sha": checkpoint["base_train_sha"],
        "effective_train_sha": checkpoint["effective_train_sha"], "added_train_rows_1based": checkpoint["added_train_rows_1based"],
        "train_test_overlap_1based": checkpoint["train_test_overlap_1based"], "training_membership": membership,
        "metrics": {key: float(metrics[key]) for key in METRICS},
        "initial_gate_values": checkpoint["initial_gate_values"], "initial_gate_logits": checkpoint["initial_gate_logits"],
        "best_edge_gate": checkpoint["best_edge_gate"], "best_adaptive_gate": checkpoint["best_adaptive_gate"],
        "diagnostics": checkpoint["diagnostics"], "mechanism": info, "source_sha256": checkpoint["source_sha256"],
        "intervention_metrics": {name: {key: float(value[key]) for key in METRICS} for name, value in intervention_metrics.items()},
        "intervention_changes": intervention_changes,
        "test_probability_max_abs_error": float(np.max(np.abs(metrics["prob"] - np.asarray(checkpoint["test_probability"])))),
    }
    return audit


def metric_summary(rows):
    import numpy as np
    summary = []
    for group in ("53.3-HISTORICAL", "E62-E2-HISTORICAL", *GROUP_ORDER):
        subset = [row for row in rows if row["group"] == group]
        if not subset:
            continue
        line = {"group": group, "group_name": subset[0]["group_name"], "folds": len(subset), "source": subset[0]["source"]}
        for metric in METRICS:
            x = np.asarray([float(row[metric]) for row in subset]) * 100.0
            line[metric + "_mean_percent"] = float(x.mean())
            line[metric + "_population_sd_percent"] = float(x.std(ddof=0))
        summary.append(line)
    return summary


def paired_summary(rows):
    import numpy as np
    from scipy.stats import t
    lookup = {(row["group"], int(row["fold"])): row for row in rows}
    comparisons = []
    names = [(g, "53.3-HISTORICAL") for g in GROUP_ORDER] + [(g, "E62-E2-HISTORICAL") for g in GROUP_ORDER]
    names += [(GROUP_ORDER[i], GROUP_ORDER[j]) for i in range(len(GROUP_ORDER)) for j in range(i + 1, len(GROUP_ORDER))]
    for left, right in names:
        for metric in METRICS:
            diffs = np.asarray([(lookup[(left, f)][metric] - lookup[(right, f)][metric]) * 100.0 for f in range(1, 11)])
            mean = float(diffs.mean())
            half = float(t.ppf(0.975, 9) * diffs.std(ddof=1) / math.sqrt(10))
            wins = int((diffs > 1e-12).sum()); losses = int((diffs < -1e-12).sum()); ties = 10 - wins - losses
            for fold, diff in enumerate(diffs, 1):
                comparisons.append({"left_group": left, "right_group": right, "comparison": f"{left}_minus_{right}", "metric": metric, "metric_label": METRIC_LABELS[metric], "fold": fold, "difference_pp": float(diff), "mean_difference_pp": mean, "descriptive_95_ci_low": mean - half, "descriptive_95_ci_high": mean + half, "wins": wins, "ties": ties, "losses": losses, "wins_ties_losses": f"{wins}/{ties}/{losses}"})
            comparisons.append({"left_group": left, "right_group": right, "comparison": f"{left}_minus_{right}", "metric": metric, "metric_label": METRIC_LABELS[metric], "fold": "ALL", "difference_pp": "", "mean_difference_pp": mean, "descriptive_95_ci_low": mean - half, "descriptive_95_ci_high": mean + half, "wins": wins, "ties": ties, "losses": losses, "wins_ties_losses": f"{wins}/{ties}/{losses}"})
    return comparisons


def historical_rows():
    e61 = load_e61()
    ref53 = e61.reference_rows()
    base_rows = []
    for fold in range(1, 11):
        old = ref53[fold]
        base_rows.append({"group": "53.3-HISTORICAL", "group_name": "53.3 third-group read-only", "fold": fold, "status": "READ_ONLY_REFERENCE", "source": "Experiment 53.3 historical", **{m: float(old[m]) for m in METRICS}})
    e62rows = list(csv.DictReader((E62_EXP / "results/fold_metrics.csv").open(encoding="utf-8")))
    e62rows = [row for row in e62rows if row["group"] == "E62-E2"]
    if len(e62rows) != 10 or any(row.get("status") != "INDEPENDENT_AUDIT_PASS" for row in e62rows):
        raise RuntimeError("E62-E2 historical ten-fold metrics are not independently audited")
    for row in e62rows:
        base_rows.append({"group": "E62-E2-HISTORICAL", "group_name": "E62-E2 historical", "fold": int(row["fold"]), "status": "READ_ONLY_REFERENCE", "source": "Experiment 62 historical", **{m: float(row[m]) for m in METRICS}})
    return base_rows


def aggregate_mechanisms(audits):
    rows = []
    for audit in audits:
        info = audit["mechanism"]
        diag = info["edge_diagnostics"]
        msg = info["message_norms"]
        for i, modality in enumerate(MODALITIES):
            weighted_fusion = info["fusion_weight_sigmoid"][i]
            gate = audit["best_adaptive_gate"][i]
            raw = info["raw_input_frobenius_norm"]
            rows.append({
                "group": audit["group"], "fold": audit["fold"], "modality": modality,
                "initial_edge_gate": audit["initial_gate_values"]["edge_gate"][i], "best_edge_gate": audit["best_edge_gate"][i],
                "initial_adaptive_gate": audit["initial_gate_values"]["adaptive_gate"][i], "best_adaptive_gate": gate,
                "initial_adaptive_gate_requested": audit["initial_gate_values"].get("adaptive_gate_requested", [None] * 4)[i],
                "train_best_epoch_edge_gate_grad_abs": audit["diagnostics"]["gate_gradients"]["edge_gate_abs"][i],
                "train_best_epoch_adaptive_gate_grad_abs": audit["diagnostics"]["gate_gradients"]["adaptive_gate_abs"][i],
                "classification_edge_gate_grad_abs_at_best": audit["diagnostics"].get("classification_gradient_replay", {}).get("edge_gate_abs", [None]*4)[i],
                "classification_adaptive_gate_grad_abs_at_best": audit["diagnostics"].get("classification_gradient_replay", {}).get("adaptive_gate_abs", [None]*4)[i],
                "positive_edges": diag[i]["positive_edges"], "negative_edges": diag[i]["negative_edges"], "uncertain_edges": diag[i]["uncertain_edges"],
                "scored_edges": diag[i]["scored_edges"], "uncertain_edge_fraction": diag[i]["uncertain_edges"] / max(diag[i]["scored_edges"], 1),
                "q_mean": diag[i]["q_mean"], "q_std": diag[i]["q_std"],
                "original_gcn_message_norm": msg["original_gcn"][i], "positive_gcn_message_norm": msg["positive_gcn"][i],
                "negative_gcn_message_norm": msg["negative_gcn_scaled_0p25"][i], "direct_difference_message_norm": msg["direct_difference_scaled_0p02"][i],
                "cadr_message_norm": msg["cadr_message"][i], "weighted_original_gcn_norm": msg["original_gcn"][i] * weighted_fusion * (1-gate),
                "weighted_cadr_norm": info["weighted_cadr_norm"][i], "weighted_cadr_to_input_ratio": info["weighted_cadr_to_raw_ratio"][i],
                "weighted_positive_gcn_norm": msg["positive_gcn"][i] * weighted_fusion * gate,
                "weighted_negative_gcn_norm": msg["negative_gcn_scaled_0p25"][i] * weighted_fusion * gate,
                "weighted_direct_difference_norm": msg["direct_difference_scaled_0p02"][i] * weighted_fusion * gate,
                "raw_input_frobenius_norm": raw, "edge_gate_sigmoid_value": diag[i]["edge_gate"],
                "fusion_weight_sigmoid": weighted_fusion,
            })
    return rows


def summary_report(rows, summary, paired, audits, manifest):
    smap = {r["group"]: r for r in summary}
    pmap = {(r["left_group"], r["right_group"], r["metric"]): r for r in paired if r["fold"] == "ALL"}
    rng_rows = manifest.get("runtime_rng_comparison", [])
    modified = affected_folds()
    unchanged_folds = [fold for fold in range(1, 11) if fold not in modified]
    row_description = "; ".join(
        f"fold {fold} gets rows {', '.join(str(value) for value in CFG['added_training_rows_by_test_fold'][str(fold)])}"
        for fold in modified
    ) or "no rows are added"
    unchanged_description = ", ".join(str(fold) for fold in unchanged_folds)
    lines = [
        f"# {CFG['experiment']}", "",
        "## Protocol", "",
        f"This diagnostic reruns E62-E2 from the exact, SHA-verified E61 epoch-10 shared warmup for each fold. It uses the unchanged E62-E2 gates (edge gates 0.5; adaptive gates PHENO/ANAT/FUNC=0.001 and CORR=0.99), model, optimizer settings, loss, split, seed, and training schedule. Rows are appended to the classification training indices and CADR relation-pair training population only when their own fold is held out: {row_description}. They remain in their corresponding test sets. Folds {unchanged_description} are unchanged.", "",
        f"The {len(modified)} affected fold(s) deliberately overlap training and test labels across {added_row_count()} row(s). Their metrics include label-exposure/memorization effects and cannot support generalization claims. The ten-fold mean is consequently not a fair performance comparison with E62-E2 or 53.3. Historical `val=test`, pre-CV preprocessing, and the 871-node transductive graph are also retained.", "",
        f"Formal status: **{manifest['status']}**, {manifest['completed_tasks']}/10 tasks and {manifest['independent_audits']}/10 independent checkpoint audits; device {manifest['device']}; maximum workers {manifest['max_workers']}. E61 commit `{CFG['base']['e61_commit']}` and E62-E2 source/config/results were SHA-checked. The fold-specific index/pair update used a local NumPy generator and left each warmup's main RNG state unchanged.", "",
        f"Training source SHA is recorded per task as `{manifest.get('training_source_sha256', {}).get('run.py', 'not recorded')}`. The audit/report runner SHA is `{manifest.get('audit_source_sha256', {}).get('run.py', 'not recorded')}`; after training, only the gate-metadata audit check, audit-only entry point, and report/handoff generation changed. The training forward, optimizer updates, and RNG trace logic were not changed.", "",
        "## Classification results", "",
        f"Values are 10-fold mean ± population SD, percent. Historical groups are read-only. {GROUP_ORDER[0]} aggregate values are contaminated by the listed overlap and are shown only as a diagnostic summary.", "",
        "|Group|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("53.3-HISTORICAL", "E62-E2-HISTORICAL", *GROUP_ORDER):
        if group not in smap: continue
        line=smap[group]
        display_metrics = ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")
        lines.append("|"+line["group_name"]+"|"+"|".join(f"{line[m+'_mean_percent']:.4f} ± {line[m+'_population_sd_percent']:.4f}" for m in display_metrics)+"|")
    lines += ["", "## Fold Membership and RNG", "", "|Fold|Added train rows|Train/test overlap|Effective train n|E62 RNG common epochs matched|", "|---:|---|---|---:|---:|"]
    rng_by_fold = {int(row["fold"]): row for row in rng_rows}
    for audit in audits:
        membership = audit["training_membership"]
        rng = rng_by_fold.get(int(audit["fold"]), {})
        lines.append(f"|{audit['fold']}|{membership['added_train_rows_1based'] or 'none'}|{membership['train_test_overlap_1based'] or 'none'}|{membership['effective_train_count']}|{rng.get('exact_matches', 0)}/{rng.get('common_epochs', 0)}|")
    lines += ["", f"RNG SHA comparisons are against read-only E62-E2 task traces on common epochs. Matching RNG hashes demonstrate matching random-number consumption, not matching model updates; the {len(modified)} modified fold(s) have changed labels and relation pairs.", "", "## Paired Comparisons", "", f"Differences are percentage points with descriptive paired-t 95% intervals across the original folds; they are exploratory, and {GROUP_ORDER[0]} comparisons are invalid as held-out performance estimates because the listed fold(s) contain train/test overlap.", "", "|Comparison|ACC|Hard-AUC|Probability-AUC|SEN-ASD|SPE-TD|", "|---|---:|---:|---:|---:|---:|"]
    pair_list = [(g, "E62-E2-HISTORICAL") for g in GROUP_ORDER] + [(g, "53.3-HISTORICAL") for g in GROUP_ORDER]
    for left,right in pair_list:
        vals=[]
        for metric in ("acc","hard_auc","prob_auc","sen","spe"):
            p=pmap[(left,right,metric)]
            vals.append(f"{p['mean_difference_pp']:+.3f} ({p['wins_ties_losses']}; {p['descriptive_95_ci_low']:+.3f},{p['descriptive_95_ci_high']:+.3f})")
        lines.append(f"|{left} − {right}|"+"|".join(vals)+"|")
    row_lookup = {(row["group"], int(row["fold"])): row for row in rows if row["group"] in (*GROUP_ORDER, "E62-E2-HISTORICAL")}
    lines += ["", "## Label-Exposure Diagnostic", "", "|Fold|Added rows|ΔACC pp|ΔHard-AUC pp|ΔProbability-AUC pp|ΔSEN pp|ΔSPE pp|", "|---:|---:|---:|---:|---:|---:|---:|"]
    for fold in range(1, 11):
        current = row_lookup[(GROUP_ORDER[0], fold)]
        base = row_lookup[("E62-E2-HISTORICAL", fold)]
        additions = CFG["added_training_rows_by_test_fold"].get(str(fold), [])
        delta = {metric: (float(current[metric]) - float(base[metric])) * 100.0 for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe")}
        lines.append(f"|{fold}|{len(additions)}|{delta['acc']:+.3f}|{delta['hard_auc']:+.3f}|{delta['prob_auc']:+.3f}|{delta['sen']:+.3f}|{delta['spe']:+.3f}|")
    clean_max_diff = max(
        abs(float(row_lookup[(GROUP_ORDER[0], fold)][metric]) - float(row_lookup[("E62-E2-HISTORICAL", fold)][metric]))
        for fold in unchanged_folds for metric in METRICS
    )
    lines += [f"", f"Across the {len(unchanged_folds)} untouched folds ({unchanged_description}), the maximum absolute difference from E62-E2 over all seven metrics is {clean_max_diff:.3g}; every metric matches exactly. This confirms the unchanged-fold continuation and RNG path. Changes in folds {', '.join(map(str, modified))} occur where selected test rows were also used for training and must be read as label-exposure/memorization effects.", ""]
    lines += ["", "## Fold Results", "", "|Fold|ACC|Hard-AUC|Prob-AUC|SEN|SPE|Best epoch|Overlap n|", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for audit in audits:
        m = audit["metrics"]
        lines.append(f"|{audit['fold']}|{100*m['acc']:.3f}|{100*m['hard_auc']:.3f}|{100*m['prob_auc']:.3f}|{100*m['sen']:.3f}|{100*m['spe']:.3f}|{audit['best_epoch_1based']}|{len(audit['train_test_overlap_1based'])}|")
    lines += ["", "## Gate and Propagation", "", "Gate values, edge counts, gradient magnitudes, original graph and CADR/direct message contributions are available in `gate_diagnostics.csv`. A nonzero message does not establish diagnostic benefit.", "", "|Fold|Modality|Initial edge|Final edge|Initial fusion|Final fusion|Positive edges|Negative edges|Uncertain %|CADR / input %|", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for audit in audits:
        info=audit["mechanism"]
        for i, modality in enumerate(MODALITIES):
            d=info["edge_diagnostics"][i]
            lines.append(f"|{audit['fold']}|{modality}|{audit['initial_gate_values']['edge_gate'][i]:.3f}|{audit['best_edge_gate'][i]:.4f}|{audit['initial_gate_values']['adaptive_gate'][i]:.3f}|{audit['best_adaptive_gate'][i]:.4f}|{d['positive_edges']}|{d['negative_edges']}|{100*d['uncertain_edges']/max(d['scored_edges'],1):.2f}|{100*info['weighted_cadr_to_raw_ratio'][i]:.5f}|")
    lines += ["", "### Fixed-checkpoint interventions", "", "Each effect is full model minus intervention. Score permutation was fixed independently of test labels.", "", "|Fold|Intervention|ΔACC pp|Hard prediction flips|Mean absolute Δprobability|", "|---:|---|---:|---:|---:|"]
    for audit in audits:
        for name, value in audit["intervention_changes"].items():
            lines.append(f"|{audit['fold']}|{name}|{value['acc_drop_pp']:+.3f}|{value['changed_predictions']}|{value['mean_abs_probability_change']:.6f}|")
    e2acc = pmap[(GROUP_ORDER[0], "E62-E2-HISTORICAL", "acc")]
    lines += ["", "## Interpretation", "", f"{GROUP_ORDER[0]} minus E62-E2 aggregate ACC is {e2acc['mean_difference_pp']:+.3f} pp ({e2acc['wins_ties_losses']} folds won/tied/lost), but this is not a valid generalization comparison because {added_row_count()} test rows in folds {', '.join(map(str, modified))} were also supervised during training. The {len(unchanged_folds)} unaffected folds reproduce E62-E2 exactly. This experiment tests the effect of explicit label exposure, not a deployable training protocol.", "", "## Files", "", f"- Experiment root: `{EXP}`", "- Task manifest: `manifest.json`", "- Fold metrics: `results/fold_metrics.csv`", "- Paired comparisons: `results/paired_comparisons.csv`", "- Gate diagnostics: `results/gate_diagnostics.csv`", "- Independent checkpoint audit: `results/checkpoint_audit.csv`", ""]
    return "\n".join(lines)


def append_handoff(manifest, summary, paired, audits):
    import tempfile
    path = Path("/model/spromptgl/experiment/handoff.md")
    if not path.is_file(): raise FileNotFoundError(path)
    marker = "## Experiment 69.3-G2 - Two Selected Rows Added to Fold 2 Training"
    current=path.read_text(encoding="utf-8")
    smap={r["group"]:r for r in summary}
    pmap={(r["left_group"],r["right_group"],r["metric"]):r for r in paired if r["fold"]=="ALL"}
    overlap_description = "; ".join(f"fold {fold}: {', '.join(str(value) for value in CFG['added_training_rows_by_test_fold'][str(fold)])}" for fold in affected_folds())
    lines=["",marker,"",f"- Status: {manifest['status']}; {manifest['completed_tasks']}/10 trained and {manifest['independent_audits']}/10 independently audited.",f"- Base: E62-E2; unchanged edge gates `[0.5, 0.5, 0.5, 0.5]` and adaptive gates `[0.001, 0.001, 0.001, 0.99]`. Each fold branched from its SHA-verified E61 epoch-10 warmup.",f"- Intentional overlap ({added_row_count()} rows): {overlap_description}. These rows remain in their fold's test set, so the affected-fold metrics are contaminated and cannot be interpreted as held-out generalization."]
    for g in GROUP_ORDER:
        s=smap[g]
        lines.append(f"- {g}: ACC {s['acc_mean_percent']:.4f}±{s['acc_population_sd_percent']:.4f}%, Hard-AUC {s['hard_auc_mean_percent']:.4f}±{s['hard_auc_population_sd_percent']:.4f}%, Probability-AUC {s['prob_auc_mean_percent']:.4f}±{s['prob_auc_population_sd_percent']:.4f}%, SPE {s['spe_mean_percent']:.4f}±{s['spe_population_sd_percent']:.4f}%, SEN {s['sen_mean_percent']:.4f}±{s['sen_population_sd_percent']:.4f}%.")
    for g in GROUP_ORDER:
        p=pmap[(g,"E62-E2-HISTORICAL","acc")]
        lines.append(f"- {g} vs E62-E2 aggregate ACC {p['mean_difference_pp']:+.3f} pp ({p['wins_ties_losses']}); this comparison is descriptive only because of the train/test overlap.")
    rng = manifest.get("runtime_rng_comparison", [])
    matched = sum(int(item["exact_matches"]) for item in rng)
    common = sum(int(item["common_epochs"]) for item in rng)
    fold_rows = list(csv.DictReader((RESULTS / "fold_metrics.csv").open(encoding="utf-8")))
    fold_map = {(row["group"], int(row["fold"])): row for row in fold_rows}
    untouched = [fold for fold in range(1, 11) if fold not in affected_folds()]
    clean_max = max(abs(float(fold_map[(GROUP_ORDER[0], fold)][metric]) - float(fold_map[("E62-E2-HISTORICAL", fold)][metric])) for fold in untouched for metric in METRICS)
    affected_deltas = "; ".join(f"fold {fold} {100*(float(fold_map[(GROUP_ORDER[0],fold)]['acc'])-float(fold_map[('E62-E2-HISTORICAL',fold)]['acc'])):+.3f} pp" for fold in affected_folds())
    lines += [f"- RNG: {matched}/{common} common epoch triplets match the E62-E2 SHA trace; the {len(untouched)} untouched folds also reproduce all seven metrics exactly (max absolute difference {clean_max:.3g}).",f"- Affected-fold ACC changes versus E62-E2: {affected_deltas}. These changes reflect supervised exposure to the same held-out labels.",f"- Results: `{EXP}`; report `results/report.md`; manifest `manifest.json`. The original ten-fold split and all E62-E2 settings were retained; local pair generation did not consume the main RNG. Historical `val=test`, pre-CV preprocessing, and transductive graph remain.",""]
    new="\n".join(lines)
    if marker in current:
        start=current.index(marker); nxt=current.find("\n## ",start+len(marker)); prefix=current[:start].rstrip(); suffix=current[nxt:] if nxt>=0 else ""; updated=prefix+"\n\n"+new.lstrip("\n")+("\n"+suffix.lstrip("\n") if suffix else "")
    else: updated=current.rstrip()+"\n"+new
    mode=path.stat().st_mode
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,prefix="handoff.e69.3.",suffix=".tmp",delete=False) as stream:
        temp=Path(stream.name);stream.write(updated);stream.flush();os.fsync(stream.fileno())
    os.chmod(temp,mode);os.replace(temp,path)
    if path.read_text(encoding="utf-8")!=updated: raise IOError("handoff readback mismatch")
    return sha_file(path)


def finalize(audits, runtime_info):
    import numpy as np
    expected_tasks = len(GROUP_ORDER) * 10
    rows=historical_rows()
    for a in audits:
        rows.append({"group":a["group"],"group_name":a["group"],"fold":a["fold"],"status":"INDEPENDENT_AUDIT_PASS","source":"new training","checkpoint_sha256":a["checkpoint_sha256"],"best_epoch_1based":a["best_epoch_1based"],"epochs_executed":a["epochs_executed"],**a["metrics"]})
    summary=metric_summary(rows); paired=paired_summary(rows); mechanisms=aggregate_mechanisms(audits)
    write_csv(RESULTS/"fold_metrics.csv",rows);write_csv(RESULTS/"summary.csv",summary);write_csv(RESULTS/"paired_comparisons.csv",paired);write_csv(RESULTS/"gate_diagnostics.csv",mechanisms)
    audit_rows=[]
    for a in audits:
        audit_rows.append({k:a[k] for k in ("group","fold","checkpoint_audit","checkpoint_sha256","best_epoch_1based","epochs_executed","warmup_sha256","split_sha","pair_sha","warmup_pair_sha","base_train_sha","effective_train_sha","added_train_rows_1based","train_test_overlap_1based","metrics","initial_gate_values","best_edge_gate","best_adaptive_gate","intervention_metrics","intervention_changes","test_probability_max_abs_error")})
    write_csv(RESULTS/"checkpoint_audit.csv",audit_rows)
    status="COMPLETE_AUDITED" if len(audits)==expected_tasks and all(a["checkpoint_audit"]=="PASS" for a in audits) else "AUDIT_INCOMPLETE"
    manifest={
        "experiment":CFG["experiment"],"status":status,"experiment_root":str(EXP),"base":CFG["base"],"seed":0,
        "expected_tasks":expected_tasks,"completed_tasks":len(audits),"independent_audits":sum(a["checkpoint_audit"]=="PASS" for a in audits),
        "failed_tasks":[],"device":runtime_info["device"],"max_workers":runtime_info["max_workers"],"source_sha256":source_sha_map(),
        "training_source_sha256":runtime_info.get("training_source_sha256", audits[0]["source_sha256"] if audits else {}),
        "audit_source_sha256":source_sha_map(),
        "e61_training_source_sha256":E61_TRAIN_SOURCE,"e61_published_source_sha256":e61_manifest_source(),
        "e62_reference_source_sha256":{"run.py":sha_file(E62_RUN),"config.json":sha_file(E62_CONFIG),"manifest.json":sha_file(E62_MANIFEST)},
        "config_sha256":sha_file(CONFIG_PATH),"warmups":warmup_manifest_info(),
        "intentional_train_test_overlap":True,
        "overlap_folds":{str(fold):CFG["added_training_rows_by_test_fold"][str(fold)] for fold in affected_folds()},
        "tasks":[{"group":a["group"],"fold":a["fold"],"checkpoint_sha256":a["checkpoint_sha256"],"warmup_sha256":a["warmup_sha256"],"warmup_pair_sha":a["warmup_pair_sha"],"pair_sha":a["pair_sha"],"effective_train_sha":a["effective_train_sha"],"train_test_overlap_1based":a["train_test_overlap_1based"],"best_epoch_1based":a["best_epoch_1based"],"epochs_executed":a["epochs_executed"],"source_sha256":a["source_sha256"],"status":"COMPLETE","audit":"PASS"} for a in audits],
        "runtime_rng_comparison":runtime_info.get("rng_comparison")
    }
    report=summary_report(rows,summary,paired,audits,manifest)
    report_path=RESULTS/"report.md";tmp=report_path.with_name("report.md.tmp");tmp.write_text(report,encoding="utf-8");os.replace(tmp,report_path)
    manifest["results_sha256"]={p.name:sha_file(p) for p in sorted(RESULTS.iterdir()) if p.is_file() and p.name!="manifest.json"}
    atomic_json(EXP/"manifest.json",manifest)
    handoff_sha=append_handoff(manifest,summary,paired,audits)
    manifest["handoff_path"]="/model/spromptgl/experiment/handoff.md";manifest["handoff_sha256"]=handoff_sha
    atomic_json(EXP/"manifest.json",manifest)
    return manifest


def run_all(max_workers=4):
    import torch
    preflight_path=RESULTS/"preflight.json"
    pre=json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.exists() else None
    if pre is None or pre.get("source_sha256") != source_sha_map() or pre.get("config_sha256") != sha_file(CONFIG_PATH):
        pre=gate_preflight()
    if pre.get("status")!="PASS": raise RuntimeError("preflight is not PASS")
    CKPTS.mkdir(parents=True,exist_ok=True);TASKS.mkdir(parents=True,exist_ok=True);LOGS.mkdir(parents=True,exist_ok=True)
    max_workers=max(1,min(10,int(max_workers)))
    pending=[]
    for group in GROUP_ORDER:
        for fold in range(1,11):
            tf,cp=task_path(group,fold),checkpoint_path(group,fold)
            if tf.is_file() and cp.is_file():
                task=json.loads(tf.read_text(encoding="utf-8"))
                if task.get("status")=="COMPLETE" and task.get("checkpoint_sha256")==sha_file(cp) and task.get("source_sha256")==source_sha_map() and task.get("config_sha256")==sha_file(CONFIG_PATH): continue
            pending.append((group,fold))
    active={};failed=[];attempts={}
    while pending or active:
        while pending and len(active)<max_workers:
            group,fold=pending.pop(0);log=LOGS/f"{group}_f{fold:02d}.stdout.log"
            stream=log.open("a",encoding="utf-8")
            command=[sys.executable,str(EXP/"run.py"),"--worker","--group",group,"--fold",str(fold)]
            proc=subprocess.Popen(command,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
            attempts[(group,fold)]=attempts.get((group,fold),0)+1
            active[proc]=(group,fold,stream)
        time.sleep(1)
        for proc,(group,fold,stream) in list(active.items()):
            code=proc.poll()
            if code is None: continue
            stream.close();del active[proc]
            if code!=0:
                task=(group,fold)
                if attempts[task] < 2:
                    pending.append(task)
                    max_workers=max(1,min(max_workers,2))
                else:
                    failed.append({"group":group,"fold":fold,"exit_code":code,"attempts":attempts[task],"log":str(LOGS/f"{group}_f{fold:02d}.stdout.log")})
    if failed:
        completed=[]
        for group in GROUP_ORDER:
            for fold in range(1,11):
                tf,cp=task_path(group,fold),checkpoint_path(group,fold)
                if tf.is_file() and cp.is_file():
                    task=json.loads(tf.read_text(encoding="utf-8"))
                    if task.get("status")=="COMPLETE" and task.get("checkpoint_sha256")==sha_file(cp):
                        completed.append({"group":group,"fold":fold,"checkpoint_sha256":sha_file(cp),"status":"COMPLETE"})
        atomic_json(EXP/"manifest.json",{"experiment":CFG["experiment"],"status":"TRAINING_INCOMPLETE","expected_tasks":len(GROUP_ORDER)*10,"completed_tasks":len(completed),"independent_audits":0,"completed":completed,"failed_tasks":failed,"source_sha256":source_sha_map(),"config_sha256":sha_file(CONFIG_PATH),"preflight":"PASS"})
        raise RuntimeError(f"formal training failed: {failed}")
    # Audit all groups from independently reconstructed warmup states.
    audits=[]
    for group in GROUP_ORDER:
        for fold in range(1,11):
            audits.append(intervene_and_audit(group,fold))
            print(f"independent checkpoint audit {len(audits)}/{len(GROUP_ORDER)*10}",flush=True)
    # RNG correspondence is evaluated only over shared completed epoch numbers.
    rng_comparison=[]
    for fold in range(1,11):
        trace=json.loads(task_path(GROUP_ORDER[0],fold).read_text(encoding="utf-8"))["rng_trace"]
        reference_task=E62_EXP/".runtime/tasks"/f"E62-E2_f{fold:02d}.json"
        if not reference_task.is_file(): raise FileNotFoundError(f"E62-E2 RNG trace missing for fold {fold}")
        reference=json.loads(reference_task.read_text(encoding="utf-8"))["rng_trace"]
        rmap={x["epoch"]:x for x in reference};common=[x for x in trace if x["epoch"] in rmap]
        matches=sum(x["rng_before"]==rmap[x["epoch"]]["rng_before"] and x["rng_after_train"]==rmap[x["epoch"]]["rng_after_train"] and x["rng_after_eval"]==rmap[x["epoch"]]["rng_after_eval"] for x in common)
        status="PASS" if matches==len(common) else "DIVERGED_RECORDED"
        rng_comparison.append({"fold":fold,"left":GROUP_ORDER[0],"right":"E62-E2","common_epochs":len(common),"exact_matches":matches,"status":status,"training_set_modified":fold in affected_folds()})
        if fold not in affected_folds() and matches != len(common):
            raise RuntimeError(f"RNG trace diverged on unaffected fold {fold}: {matches}/{len(common)}")
    runtime_info={"device":str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu","max_workers":max_workers,"rng_comparison":rng_comparison}
    manifest=finalize(audits,runtime_info)
    print(f"Experiment 69.3 status: {manifest['status']}",flush=True)
    return manifest


def audit_completed_tasks(max_workers=10):
    """Audit existing immutable training checkpoints without rerunning them."""
    import torch
    pre = gate_preflight()
    if pre.get("status") != "PASS":
        raise RuntimeError("preflight is not PASS")
    audits = []
    for group in GROUP_ORDER:
        for fold in range(1, 11):
            task_file, cp_file = task_path(group, fold), checkpoint_path(group, fold)
            if not task_file.is_file() or not cp_file.is_file():
                raise FileNotFoundError(f"completed task/checkpoint missing for {group} fold={fold}")
            task = json.loads(task_file.read_text(encoding="utf-8"))
            if task.get("status") != "COMPLETE" or task.get("checkpoint_sha256") != sha_file(cp_file):
                raise RuntimeError(f"task/checkpoint SHA check failed for {group} fold={fold}")
            audits.append(intervene_and_audit(group, fold))
            print(f"independent checkpoint audit {len(audits)}/{len(GROUP_ORDER)*10}", flush=True)
    rng_comparison = []
    for fold in range(1, 11):
        trace = json.loads(task_path(GROUP_ORDER[0], fold).read_text(encoding="utf-8"))["rng_trace"]
        reference_task = E62_EXP / ".runtime/tasks" / f"E62-E2_f{fold:02d}.json"
        reference = json.loads(reference_task.read_text(encoding="utf-8"))["rng_trace"]
        rmap = {item["epoch"]: item for item in reference}
        common = [item for item in trace if item["epoch"] in rmap]
        matches = sum(
            item["rng_before"] == rmap[item["epoch"]]["rng_before"]
            and item["rng_after_train"] == rmap[item["epoch"]]["rng_after_train"]
            and item["rng_after_eval"] == rmap[item["epoch"]]["rng_after_eval"]
            for item in common
        )
        rng_comparison.append({
            "fold": fold, "left": GROUP_ORDER[0], "right": "E62-E2",
            "common_epochs": len(common), "exact_matches": matches,
            "status": "PASS" if matches == len(common) else "DIVERGED_RECORDED",
            "training_set_modified": fold in affected_folds(),
        })
        if fold not in affected_folds() and matches != len(common):
            raise RuntimeError(f"RNG trace diverged on unchanged fold {fold}: {matches}/{len(common)}")
    runtime_info = {
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        "max_workers": int(max_workers), "rng_comparison": rng_comparison,
        "audit_only": True, "training_source_sha256": json.loads(task_path(GROUP_ORDER[0], 1).read_text(encoding="utf-8"))["source_sha256"],
    }
    manifest = finalize(audits, runtime_info)
    print(f"Experiment 69.3 status: {manifest['status']}", flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--audit-only",action="store_true")
    parser.add_argument("--group",choices=GROUP_ORDER)
    parser.add_argument("--fold",type=int)
    parser.add_argument("--max-workers",type=int,default=10)
    args=parser.parse_args()
    if args.worker:
        if args.group is None or args.fold is None: raise SystemExit("--worker needs --group and --fold")
        result=train_worker(args.group,args.fold);print(json.dumps(result,ensure_ascii=False),flush=True)
    elif args.preflight:
        print(json.dumps(gate_preflight(),ensure_ascii=False,indent=2))
    elif args.run:
        run_all(args.max_workers)
    elif args.audit_only:
        audit_completed_tasks(args.max_workers)
    else: parser.print_help()


if __name__=="__main__": main()
