#!/usr/bin/env python3
"""Experiment 64 runner: refine E62-E2 CORR adaptive fusion initialization."""
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
E62_EXP = Path(CFG["e62_reference"]["path"])
E63_EXP = Path(CFG["e63_reference"]["path"])
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
    spec = importlib.util.spec_from_file_location("experiment63_locked_e61_runner", E61_RUN)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SOURCE_SHA_OVERRIDE = E61_TRAIN_SOURCE
    E61 = module
    return E61


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
    adaptive_values = torch.tensor(cfg["adaptive_gate"], dtype=state["model"].adaptive_gate_logits.dtype, device=state["model"].adaptive_gate_logits.device)
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
    return {"edge_gate": actual_edge.tolist(), "adaptive_gate": actual_adaptive.tolist(), "optimizer_state": states, "rng_unchanged": True}


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


def verify_e62_reference():
    ref = CFG["e62_reference"]
    manifest_path = E62_EXP / "manifest.json"
    config_path = E62_EXP / "config.json"
    runner_path = E62_EXP / "run.py"
    metrics_path = E62_EXP / "results/fold_metrics.csv"
    expected_hashes = {
        manifest_path: ref["manifest_sha256"],
        config_path: ref["config_sha256"],
        runner_path: ref["run_sha256"],
        metrics_path: ref["fold_metrics_sha256"],
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or sha_file(path) != expected:
            raise RuntimeError(f"E62 read-only reference SHA mismatch: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE_AUDITED" or manifest.get("completed_tasks") != 40 or manifest.get("independent_audits") != 40:
        raise RuntimeError("E62 reference does not certify all 40 tasks and checkpoint audits")
    if manifest.get("source_sha256", {}).get("run.py") != ref["run_sha256"] or manifest.get("config_sha256") != ref["config_sha256"]:
        raise RuntimeError("E62 reference manifest source identity mismatch")
    if manifest.get("results_sha256", {}).get("fold_metrics.csv") != ref["fold_metrics_sha256"]:
        raise RuntimeError("E62 reference result manifest SHA mismatch")
    tasks = {(item["group"], int(item["fold"])): item for item in manifest.get("tasks", [])}
    checked = []
    for fold in range(1, 11):
        task = tasks.get((ref["group"], fold))
        checkpoint = E62_EXP / ".runtime/checkpoints" / f"{ref['group']}_f{fold:02d}.best.pt"
        if task is None or task.get("status") != "COMPLETE" or task.get("audit") != "PASS":
            raise RuntimeError(f"E62-E2 read-only task/audit missing at fold {fold}")
        if not checkpoint.is_file() or sha_file(checkpoint) != task.get("checkpoint_sha256"):
            raise RuntimeError(f"E62-E2 checkpoint SHA mismatch at fold {fold}")
        checked.append({"fold": fold, "checkpoint_sha256": task["checkpoint_sha256"]})
    return {
        "status": "PASS", "group": ref["group"], "config_sha256": ref["config_sha256"],
        "runner_sha256": ref["run_sha256"], "manifest_sha256": ref["manifest_sha256"],
        "fold_metrics_sha256": ref["fold_metrics_sha256"], "checked_checkpoints": checked,
    }


def verify_e63_reference():
    ref = CFG["e63_reference"]
    manifest_path = E63_EXP / "manifest.json"
    config_path = E63_EXP / "config.json"
    runner_path = E63_EXP / "run.py"
    metrics_path = E63_EXP / "results/fold_metrics.csv"
    report_path = E63_EXP / "results/report.md"
    expected_hashes = {
        manifest_path: ref["manifest_sha256"],
        config_path: ref["config_sha256"],
        runner_path: ref["run_sha256"],
        metrics_path: ref["fold_metrics_sha256"],
        report_path: ref["report_sha256"],
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or sha_file(path) != expected:
            raise RuntimeError(f"E63 read-only reference SHA mismatch: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE_AUDITED" or manifest.get("completed_tasks") != 30 or manifest.get("independent_audits") != 30:
        raise RuntimeError("E63 reference does not certify all 30 tasks and checkpoint audits")
    if manifest.get("source_sha256", {}).get("run.py") != ref["run_sha256"] or manifest.get("config_sha256") != ref["config_sha256"]:
        raise RuntimeError("E63 reference manifest source identity mismatch")
    for name, expected in (("fold_metrics.csv", ref["fold_metrics_sha256"]), ("report.md", ref["report_sha256"])):
        if manifest.get("results_sha256", {}).get(name) != expected:
            raise RuntimeError(f"E63 reference result manifest SHA mismatch: {name}")
    tasks = {(item["group"], int(item["fold"])): item for item in manifest.get("tasks", [])}
    checked = []
    for fold in range(1, 11):
        task = tasks.get((ref["group"], fold))
        checkpoint = E63_EXP / ".runtime/checkpoints" / f"{ref['group']}_f{fold:02d}.best.pt"
        if task is None or task.get("status") != "COMPLETE" or task.get("audit") != "PASS":
            raise RuntimeError(f"E63-G1 read-only task/audit missing at fold {fold}")
        if not checkpoint.is_file() or sha_file(checkpoint) != task.get("checkpoint_sha256"):
            raise RuntimeError(f"E63-G1 checkpoint SHA mismatch at fold {fold}")
        checked.append({"fold": fold, "checkpoint_sha256": task["checkpoint_sha256"]})
    return {
        "status": "PASS", "group": ref["group"], "config_sha256": ref["config_sha256"],
        "runner_sha256": ref["run_sha256"], "manifest_sha256": ref["manifest_sha256"],
        "fold_metrics_sha256": ref["fold_metrics_sha256"], "report_sha256": ref["report_sha256"],
        "checked_checkpoints": checked,
    }


def gate_preflight():
    import torch
    e61 = load_e61()
    warmups = warmup_manifest_info()
    e62_reference = verify_e62_reference()
    e63_reference = verify_e63_reference()
    fold_state_checks = []
    for fold in range(1, 11):
        fold_state, fold_record = e61.load_warmup_state(fold)
        fold_state_checks.append({
            "fold": fold,
            "warmup_sha256": sha_file(warmup_path(fold)),
            "model_sha": state_digest(fold_state),
            "initial_shared_sha": fold_state["initial_shared_sha"],
            "initial_gfr_sha": fold_state["initial_gfr_sha"],
            "initial_cam_sha": fold_state["initial_cam_sha"],
            "initial_rng_sha": fold_state["initial_rng_sha"],
            "split_sha": fold_state["split_sha"],
            "pair_sha": fold_state["pair_sha"],
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
        raise RuntimeError("E64 unexpectedly inherited a reliability multiplier")
    if state["model"].reliability_gate_logits.requires_grad:
        raise RuntimeError("E61-E2 reliability gate should remain disabled")
    if not torch.equal(state["model"].calibration.cpu(), torch.tensor([[1.0, 0.0]] * 4)):
        raise RuntimeError("E64 E2 relation calibration must be identity sigmoid")
    group_checks = []
    for group in GROUP_ORDER:
        check_state, warm = e61.load_warmup_state(1)
        e61.configure_branch(check_state, "E61-E2", None)
        pre_gates = optimizer_gate_state(check_state)
        if pre_gates["edge_gate_logits"]["present"] or pre_gates["adaptive_gate_logits"]["present"]:
            raise RuntimeError(f"warmup has gate optimizer history; do not clear it: {pre_gates}")
        init = set_gate_initialization(check_state, group)
        if not check_state["model"].adaptive_gate_logits.requires_grad or not check_state["model"].edge_gate_logits.requires_grad:
            raise RuntimeError("E64 gate freeze detected")
        if check_state["opt_added"].param_groups[0]["weight_decay"] != 0.01 or check_state["opt_added"].param_groups[1]["weight_decay"] != 0.01:
            raise RuntimeError("E64 added-parameter AdamW WD differs from 0.01")
        # Classification-only gradients use outer-training labels and test no model update.
        check_state["opt_base"].zero_grad(set_to_none=True)
        check_state["opt_cam"].zero_grad(set_to_none=True)
        check_state["opt_added"].zero_grad(set_to_none=True)
        out = e61.forward_e2_state(check_state, training=False)
        out["cls"].backward()
        grad = gate_gradient_vectors(check_state)
        if not all(math.isfinite(value) for value in grad["edge_gate_abs"] + grad["adaptive_gate_abs"]):
            raise RuntimeError(f"nonfinite E64 gate gradient: {group}")
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
        "experiment": CFG["experiment"], "status": "PASS", "e61_commit": CFG["base"]["commit"],
        "e61_published_source_sha": e61_manifest_source(), "e61_warmup_training_source_sha": E61_TRAIN_SOURCE,
        "e61_warmups": warmups, "all_fold_state_load_checks": fold_state_checks, "fold1_before_override": pre_change,
        "e62_e2_read_only_reference": e62_reference,
        "e63_g1_read_only_reference": e63_reference,
        "groups": group_checks,
        "optimizer_wd": {"gfr": [0.11], "cam": [0.11], "cadr": [0.01, 0.01]},
        "no_oof": True, "no_test_labels_in_graph_or_relation_training": True,
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
    return {"run.py": sha_file(EXP / "run.py"), "config.json": sha_file(CONFIG_PATH), "e61_run.py": sha_file(E61_RUN), "e61_model_e61.py": sha_file(E61_MODEL), "e61_config.json": sha_file(E61_CONFIG)}


def checkpoint_payload(state, group, fold, best_state, best_epoch, epochs_executed, metrics, diag, initial, trace):
    import torch
    e61 = load_e61()
    return {
        "experiment": CFG["experiment"], "group": group, "fold": int(fold), "seed": 0,
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
        "split_sha": state["split_sha"], "pair_sha": state["pair_sha"], "anchor_sha": state["anchor_sha"],
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
        if existing.get("status") == "COMPLETE" and existing.get("checkpoint_sha256") == sha_file(best_file):
            return existing
    state, warmup = e61.load_warmup_state(fold)
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
        expected = {"group": group, "fold": fold, "warmup_sha256": sha_file(warmup_path(fold)), "source_sha256": source_sha_map(), "config_sha256": sha_file(CONFIG_PATH), "initialization": initialized}
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
        "checkpoint_sha256": sha_file(best_file), "warmup_sha256": sha_file(warmup_path(fold)),
        "split_sha": state["split_sha"], "pair_sha": state["pair_sha"], "anchor_sha": state["anchor_sha"],
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
    e61.configure_branch(state, "E61-E2", None)
    cp_file = checkpoint_path(group, fold)
    task_file = task_path(group, fold)
    task = json.loads(task_file.read_text(encoding="utf-8"))
    checkpoint = torch_load(cp_file)
    cp_sha = sha_file(cp_file)
    if task.get("status") != "COMPLETE" or task.get("checkpoint_sha256") != cp_sha or checkpoint.get("group") != group or checkpoint.get("fold") != fold:
        raise RuntimeError(f"checkpoint/task identity mismatch {group} fold={fold}")
    if checkpoint.get("warmup_sha256") != sha_file(warmup_path(fold)) or checkpoint.get("split_sha") != state["split_sha"] or checkpoint.get("pair_sha") != state["pair_sha"]:
        raise RuntimeError(f"warmup/split/pair identity mismatch {group} fold={fold}")
    task_source = task.get("source_sha256")
    if checkpoint.get("source_sha256") != task_source:
        raise RuntimeError(f"checkpoint/task source SHA mismatch {group} fold={fold}")
    current_source = source_sha_map()
    for name in ("config.json", "e61_run.py", "e61_model_e61.py", "e61_config.json"):
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
    if checkpoint["optimizer_weight_decay"] != {"gfr": [0.11], "cam": [0.11], "cadr": [0.01, 0.01]}:
        raise RuntimeError(f"optimizer WD audit failed {group} fold={fold}")

    def get(intervention="full", permuted=False, seed=0):
        return evaluate(state, intervention=intervention, permuted=permuted, seed=seed)
    full, info = evaluate(state, collect=True)
    intervention_metrics = {
        "all_cadr_off": get("disease_off"),
        "corr_cadr_off": get("modality_off:CORR"),
        "direct_off": get("direct_off"),
        "score_permutation": get("full", True, 640000 + fold),
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
    for group in ("53.3-HISTORICAL", "E61-E2-HISTORICAL", "E62-E2-HISTORICAL", "E63-G1-HISTORICAL", *GROUP_ORDER):
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
    historical_groups = ("53.3-HISTORICAL", "E61-E2-HISTORICAL", "E62-E2-HISTORICAL", "E63-G1-HISTORICAL")
    names = [(g, ref) for g in GROUP_ORDER for ref in historical_groups]
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
    e61rows = list(csv.DictReader((E61_EXP / "results/fold_metrics.csv").open(encoding="utf-8")))
    e61rows = [row for row in e61rows if row["group"] == "E61-E2"]
    if len(e61rows) != 10 or any(row.get("status") != "INDEPENDENT_AUDIT_PASS" for row in e61rows):
        raise RuntimeError("E61-E2 historical ten-fold metrics are not independently audited")
    for row in e61rows:
        base_rows.append({"group": "E61-E2-HISTORICAL", "group_name": "E61-E2 historical", "fold": int(row["fold"]), "status": "READ_ONLY_REFERENCE", "source": "Experiment 61 historical", **{m: float(row[m]) for m in METRICS}})
    verify_e62_reference()
    e62rows = list(csv.DictReader((E62_EXP / "results/fold_metrics.csv").open(encoding="utf-8")))
    e62rows = [row for row in e62rows if row["group"] == "E62-E2"]
    if len(e62rows) != 10 or any(row.get("status") != "INDEPENDENT_AUDIT_PASS" for row in e62rows):
        raise RuntimeError("E62-E2 historical ten-fold metrics are not independently audited")
    if {int(row["fold"]) for row in e62rows} != set(range(1, 11)):
        raise RuntimeError("E62-E2 historical reference does not have exactly one row per fold")
    for row in e62rows:
        base_rows.append({"group": "E62-E2-HISTORICAL", "group_name": "E62-E2 historical (read-only)", "fold": int(row["fold"]), "status": "READ_ONLY_REFERENCE", "source": "Experiment 62 audited fold metrics", **{m: float(row[m]) for m in METRICS}})
    verify_e63_reference()
    e63rows = list(csv.DictReader((E63_EXP / "results/fold_metrics.csv").open(encoding="utf-8")))
    e63rows = [row for row in e63rows if row["group"] == "E63-G1"]
    if len(e63rows) != 10 or any(row.get("status") != "INDEPENDENT_AUDIT_PASS" for row in e63rows):
        raise RuntimeError("E63-G1 historical ten-fold metrics are not independently audited")
    if {int(row["fold"]) for row in e63rows} != set(range(1, 11)):
        raise RuntimeError("E63-G1 historical reference does not have exactly one row per fold")
    for row in e63rows:
        base_rows.append({"group": "E63-G1-HISTORICAL", "group_name": "E63-G1 (CORR 0.999, read-only)", "fold": int(row["fold"]), "status": "READ_ONLY_REFERENCE", "source": "Experiment 63 audited fold metrics", **{m: float(row[m]) for m in METRICS}})
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
    import numpy as np
    smap = {r["group"]: r for r in summary}
    pmap = {(r["left_group"], r["right_group"], r["metric"]): r for r in paired if r["fold"] == "ALL"}
    modalities = MODALITIES
    lines = [
        "# Experiment 64 — CORR Adaptive Fusion Gate Initialization Sensitivity", "",
        "## Protocol", "",
        "Three groups vary only the sigmoid-space initial value of the CORR adaptive fusion gate from E62-E2: 0.995, 0.992, and 0.993. Edge gates stay at 0.5 and PHENO/ANAT/FUNC fusion gates stay at 0.001. All groups start from each exact E61 shared epoch-10 checkpoint used by E62-E2, including GFR/CAM/CADR weights, all three optimizer states, and Python/NumPy/Torch CPU/CUDA RNG; configured gate logits are set immediately before epoch 11. E62-E2 and E63-G1 best checkpoints are read-only references, not fine-tuning starts. The E61-E2 forward, loss, data, pair cache, fold split, epoch budget, early stopping, and checkpoint selection are reused. No OOF or reliability multiplier is used.", "",
        f"E61 is pinned to commit `{CFG['base']['commit']}`. Its published source files and historical manifest were SHA-checked. Warmup files were individually checked against E61 manifest/task SHA, split and pair identity. The E61 training runner SHA recorded inside warmup is preserved separately from the published post-training audit runner SHA; model and config SHA match. Both gate parameters had empty optimizer-state entries after warmup, were already in the CADR AdamW parameter groups at WD=0.01, and their state was not reset.", "",
        "History `val=test` checkpoint selection, pre-CV RFE/standardization and the transductive 871-node graph remain. Results are exploratory, not unbiased external generalization estimates.", "",
        "The actual E61 epoch-10 relation scores were all between the 0.45/0.55 thresholds in fold 1, so the initial disease graphs were empty and edge-gate classification gradients were zero at that exact checkpoint. A separate deterministic controlled-edge test verified nonzero classification gradients through activated edge gates and relation scorers. Formal training determines whether the learned relation scores create active edges.", "",
        f"Formal status: **{manifest['status']}**, {manifest['completed_tasks']}/{CFG['formal_task_count']} tasks and {manifest['independent_audits']}/{CFG['formal_task_count']} independent readbacks; device {manifest['device']}, maximum workers {manifest['max_workers']}.", "",
        "## Classification results", "",
        "Values are ten-fold mean ± population SD, percent. Historical groups are read-only.", "",
        "|Group|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("53.3-HISTORICAL", "E61-E2-HISTORICAL", "E62-E2-HISTORICAL", "E63-G1-HISTORICAL", *GROUP_ORDER):
        if group not in smap: continue
        line=smap[group]
        display_metrics = ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")
        lines.append("|"+line["group_name"]+"|"+"|".join(f"{line[m+'_mean_percent']:.4f} ± {line[m+'_population_sd_percent']:.4f}" for m in display_metrics)+"|")
    lines += ["", "## Paired comparisons", "", "Differences are percentage points with descriptive paired-t 95% intervals over ten overlapping CV folds; they are not confirmatory confidence claims.", "", "|Comparison|ACC|Hard-AUC|Probability-AUC|SEN-ASD|SPE-TD|", "|---|---:|---:|---:|---:|---:|"]
    historical_groups = ("53.3-HISTORICAL", "E61-E2-HISTORICAL", "E62-E2-HISTORICAL", "E63-G1-HISTORICAL")
    pair_list = [(g, ref) for g in GROUP_ORDER for ref in historical_groups]
    pair_list += [(GROUP_ORDER[i], GROUP_ORDER[j]) for i in range(len(GROUP_ORDER)) for j in range(i+1,len(GROUP_ORDER))]
    for left,right in pair_list:
        vals=[]
        for metric in ("acc","hard_auc","prob_auc","sen","spe"):
            p=pmap[(left,right,metric)]
            vals.append(f"{p['mean_difference_pp']:+.3f} ({p['wins_ties_losses']}; {p['descriptive_95_ci_low']:+.3f},{p['descriptive_95_ci_high']:+.3f})")
        lines.append(f"|{left} − {right}|"+"|".join(vals)+"|")
    lines += ["", "## Gate and propagation mechanisms", "", "Initial and best-checkpoint gate values, edge counts, gradient magnitudes, individual GCN/direct messages, and normalized contribution are in `gate_diagnostics.csv`. A nonzero or larger message norm does not establish diagnostic benefit.", "", "|Group|Modality|Initial edge|Final edge|Initial fusion|Final fusion|Positive edges|Negative edges|Uncertain %|CADR / input %|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for audit in audits:
        info=audit["mechanism"]
        for i, modality in enumerate(modalities):
            d=info["edge_diagnostics"][i]
            lines.append(f"|{audit['group']}|{modality}|{audit['initial_gate_values']['edge_gate'][i]:.3f}|{audit['best_edge_gate'][i]:.4f}|{audit['initial_gate_values']['adaptive_gate'][i]:.3f}|{audit['best_adaptive_gate'][i]:.4f}|{d['positive_edges']}|{d['negative_edges']}|{100*d['uncertain_edges']/max(d['scored_edges'],1):.2f}|{100*info['weighted_cadr_to_raw_ratio'][i]:.5f}|")
    lines += ["", "### Fixed-checkpoint interventions", "", "Each effect is full model minus intervention. Permutation is fixed independently of test labels.", "", "|Group|Intervention|ΔACC pp|Hard prediction flips|Mean absolute Δprobability|", "|---|---|---:|---:|---:|"]
    for audit in audits:
        for name, value in audit["intervention_changes"].items():
            lines.append(f"|{audit['group']}|{name}|{value['acc_drop_pp']:+.3f}|{value['changed_predictions']}|{value['mean_abs_probability_change']:.6f}|")
    lines += ["", "## Scientific judgment", "", "These groups test CORR fusion-gate initialization sensitivity, not a new architecture. They hold edge gates at 0.5 and non-CORR fusion gates at 0.001 while varying only the CORR start around E62-E2's 0.99; E63-G1 at 0.999 supplies a read-only upper reference. Read differences jointly with ASD sensitivity and fixed-checkpoint interventions; the folds overlap in training patients and use one seed, so results remain exploratory.", "", "## Files", "", f"- Experiment root: `{EXP}`", "- Task manifest: `manifest.json`", "- Fold metrics: `results/fold_metrics.csv`", "- Paired comparisons: `results/paired_comparisons.csv`", "- Gate diagnostics: `results/gate_diagnostics.csv`", "- Independent checkpoint audit: `results/checkpoint_audit.csv`", ""]
    return "\n".join(lines)


def append_handoff(manifest, summary, paired, audits):
    import tempfile
    path = Path("/model/spromptgl/experiment/handoff.md")
    if not path.is_file(): raise FileNotFoundError(path)
    marker = "## Experiment 64 — CORR Adaptive Fusion Gate Initialization Sensitivity"
    current=path.read_text(encoding="utf-8")
    smap={r["group"]:r for r in summary}
    pmap={(r["left_group"],r["right_group"],r["metric"]):r for r in paired if r["fold"]=="ALL"}
    lines=["",marker,"",f"- Status: {manifest['status']}; {manifest['completed_tasks']}/{CFG['formal_task_count']} trained and {manifest['independent_audits']}/{CFG['formal_task_count']} independently audited.",f"- Base: E62-E2 configuration on E61-E2 source commit `{CFG['base']['commit']}`; all groups branched from the same ten SHA-verified E61 epoch-10 warmups used by E62-E2. E62-E2 and E63-G1 best checkpoints, E61-E2, and 53.3 results are read-only references.","- Sigmoid-space gate initialization in `[PHENO, ANAT, FUNC, CORR]` order: E64-G1 edge `[.50,.50,.50,.50]`, fusion `[.001,.001,.001,.995]`; E64-G2 edge `[.50,.50,.50,.50]`, fusion `[.001,.001,.001,.992]`; E64-G3 edge `[.50,.50,.50,.50]`, fusion `[.001,.001,.001,.993]`. Only the CORR fusion gate initial value differs between groups."]
    for g in GROUP_ORDER:
        s=smap[g]
        lines.append(f"- {g}: ACC {s['acc_mean_percent']:.4f}±{s['acc_population_sd_percent']:.4f}%, Hard-AUC {s['hard_auc_mean_percent']:.4f}±{s['hard_auc_population_sd_percent']:.4f}%, Probability-AUC {s['prob_auc_mean_percent']:.4f}±{s['prob_auc_population_sd_percent']:.4f}%, SPE {s['spe_mean_percent']:.4f}±{s['spe_population_sd_percent']:.4f}%, SEN {s['sen_mean_percent']:.4f}±{s['sen_population_sd_percent']:.4f}%.")
    for g in GROUP_ORDER:
        p=pmap[(g,"E62-E2-HISTORICAL","acc")]
        lines.append(f"- {g} vs E62-E2 historical ACC {p['mean_difference_pp']:+.3f} pp ({p['wins_ties_losses']}); full paired metrics and checkpoint interventions are in the report.")
    lines += [f"- E62-E2 and E63-G1 read-only source/config/manifest/fold-metric hashes were checked before training; manifest records the exact hashes. Results: `{EXP}`; report `results/report.md`; manifest `manifest.json`. No OOF was run. Historical `val=test`, pre-CV preprocessing, and transductive graph remain.",""]
    new="\n".join(lines)
    if marker in current:
        start=current.index(marker); nxt=current.find("\n## ",start+len(marker)); prefix=current[:start].rstrip(); suffix=current[nxt:] if nxt>=0 else ""; updated=prefix+"\n\n"+new.lstrip("\n")+("\n"+suffix.lstrip("\n") if suffix else "")
    else: updated=current.rstrip()+"\n"+new
    mode=path.stat().st_mode
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,prefix="handoff.e64.",suffix=".tmp",delete=False) as stream:
        temp=Path(stream.name);stream.write(updated);stream.flush();os.fsync(stream.fileno())
    os.chmod(temp,mode);os.replace(temp,path)
    if path.read_text(encoding="utf-8")!=updated: raise IOError("handoff readback mismatch")
    return sha_file(path)


def finalize(audits, runtime_info):
    import numpy as np
    rows=historical_rows()
    for a in audits:
        rows.append({"group":a["group"],"group_name":a["group"],"fold":a["fold"],"status":"INDEPENDENT_AUDIT_PASS","source":"new training","checkpoint_sha256":a["checkpoint_sha256"],"best_epoch_1based":a["best_epoch_1based"],"epochs_executed":a["epochs_executed"],**a["metrics"]})
    summary=metric_summary(rows); paired=paired_summary(rows); mechanisms=aggregate_mechanisms(audits)
    write_csv(RESULTS/"fold_metrics.csv",rows);write_csv(RESULTS/"summary.csv",summary);write_csv(RESULTS/"paired_comparisons.csv",paired);write_csv(RESULTS/"gate_diagnostics.csv",mechanisms)
    audit_rows=[]
    for a in audits:
        audit_rows.append({k:a[k] for k in ("group","fold","checkpoint_audit","checkpoint_sha256","best_epoch_1based","epochs_executed","warmup_sha256","split_sha","pair_sha","metrics","initial_gate_values","best_edge_gate","best_adaptive_gate","intervention_metrics","intervention_changes","test_probability_max_abs_error")})
    write_csv(RESULTS/"checkpoint_audit.csv",audit_rows)
    status="COMPLETE_AUDITED" if len(audits)==CFG["formal_task_count"] and all(a["checkpoint_audit"]=="PASS" for a in audits) else "AUDIT_INCOMPLETE"
    manifest={"experiment":CFG["experiment"],"status":status,"experiment_root":str(EXP),"base":CFG["base"],"e62_reference":verify_e62_reference(),"e63_reference":verify_e63_reference(),"seed":0,"expected_tasks":CFG["formal_task_count"],"completed_tasks":len(audits),"independent_audits":sum(a["checkpoint_audit"]=="PASS" for a in audits),"failed_tasks":[],"device":runtime_info["device"],"max_workers":runtime_info["max_workers"],"source_sha256":source_sha_map(),"e61_training_source_sha256":E61_TRAIN_SOURCE,"e61_published_source_sha256":e61_manifest_source(),"config_sha256":sha_file(CONFIG_PATH),"warmups":warmup_manifest_info(),"tasks":[{"group":a["group"],"fold":a["fold"],"checkpoint_sha256":a["checkpoint_sha256"],"warmup_sha256":a["warmup_sha256"],"best_epoch_1based":a["best_epoch_1based"],"epochs_executed":a["epochs_executed"],"source_sha256":a["source_sha256"],"status":"COMPLETE","audit":"PASS"} for a in audits],"runtime_rng_comparison":runtime_info.get("rng_comparison")}
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
                if task.get("status")=="COMPLETE" and task.get("checkpoint_sha256")==sha_file(cp): continue
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
        atomic_json(EXP/"manifest.json",{"experiment":CFG["experiment"],"status":"TRAINING_INCOMPLETE","expected_tasks":CFG["formal_task_count"],"completed_tasks":len(completed),"independent_audits":0,"completed":completed,"failed_tasks":failed,"source_sha256":source_sha_map(),"config_sha256":sha_file(CONFIG_PATH),"preflight":"PASS","e62_reference":verify_e62_reference(),"e63_reference":verify_e63_reference()})
        raise RuntimeError(f"formal training failed: {failed}")
    # Audit all groups from independently reconstructed warmup states.
    audits=[]
    for group in GROUP_ORDER:
        for fold in range(1,11):
            audits.append(intervene_and_audit(group,fold))
            print(f"independent checkpoint audit {len(audits)}/{CFG['formal_task_count']}",flush=True)
    # RNG correspondence is evaluated only over shared completed epoch numbers.
    rng_comparison=[]
    for fold in range(1,11):
        traces={g:json.loads(task_path(g,fold).read_text(encoding="utf-8"))["rng_trace"] for g in GROUP_ORDER}
        per=[]
        for i,left in enumerate(GROUP_ORDER):
            for right in GROUP_ORDER[i+1:]:
                rmap={x["epoch"]:x for x in traces[right]};common=[x for x in traces[left] if x["epoch"] in rmap]
                matches=sum(x["rng_before"]==rmap[x["epoch"]]["rng_before"] and x["rng_after_train"]==rmap[x["epoch"]]["rng_after_train"] and x["rng_after_eval"]==rmap[x["epoch"]]["rng_after_eval"] for x in common)
                per.append({"fold":fold,"left":left,"right":right,"common_epochs":len(common),"exact_matches":matches,"status":"PASS" if matches==len(common) else "DIVERGED_RECORDED"})
        rng_comparison.extend(per)
    runtime_info={"device":str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu","max_workers":max_workers,"rng_comparison":rng_comparison}
    manifest=finalize(audits,runtime_info)
    print(f"Experiment 64 status: {manifest['status']}",flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--group",choices=GROUP_ORDER)
    parser.add_argument("--fold",type=int)
    parser.add_argument("--max-workers",type=int,default=4)
    args=parser.parse_args()
    if args.worker:
        if args.group is None or args.fold is None: raise SystemExit("--worker needs --group and --fold")
        result=train_worker(args.group,args.fold);print(json.dumps(result,ensure_ascii=False),flush=True)
    elif args.preflight:
        print(json.dumps(gate_preflight(),ensure_ascii=False,indent=2))
    elif args.run:
        run_all(args.max_workers)
    else: parser.print_help()


if __name__=="__main__": main()
