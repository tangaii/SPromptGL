#!/usr/bin/env python3
"""Isolated formal runner for Experiment 74.2 DCR-RUDAG."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
RT = EXP / ".runtime"
TASKS, CKPTS, LOGS = RT / "tasks", RT / "checkpoints", RT / "logs"
RESULTS = EXP / "results"
E741 = Path(CFG["reference"]["e741_path"])
E69 = Path(CFG["historical_references"]["e693"])
E62 = Path(CFG["historical_references"]["e62"])
E74 = Path(CFG["historical_references"]["e74"])
E53 = Path("/model/spromptgl/experiment/experiment53_3_cadr_graph_fallback_fix")
E61 = Path("/data/adodas/SPromptGL-main/experiment/experiment61_cadr_postalign_vs_e2e")
REPO = Path("/data/adodas/SPromptGL-main")
GROUP = CFG["method"]["groups"][0]
GROUPS = (GROUP,)
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
HIST, DCR = None, None


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_object(value):
    return load_history().sha_object(value)


def atomic_json(path, payload):
    return load_history().atomic_json(path, payload)


def atomic_torch(path, payload):
    return load_history().atomic_torch(path, payload)


def torch_load(path):
    return load_history().torch_load(path)


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


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_history():
    global HIST
    if HIST is not None:
        return HIST
    expected = CFG["reference"]
    fixed = {
        "rudag.py": "e741_rudag_sha256",
        "run.py": "e741_run_sha256",
        "audit.py": "e741_audit_sha256",
        "config.json": "e741_config_sha256",
        "manifest.json": "e741_manifest_sha256",
        "results/report.md": "e741_report_sha256",
        "results/mechanism.csv": "e741_mechanism_sha256",
        "results/interventions.csv": "e741_interventions_sha256",
        "results/fold_metrics.csv": "e741_fold_metrics_sha256",
    }
    for relative, key in fixed.items():
        actual = sha_file(E741 / relative)
        if actual != expected[key]:
            raise RuntimeError(f"E74.1 reference SHA mismatch for {relative}: {actual}")
    manifest = json.loads((E741 / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("status"), manifest.get("completed_tasks"), manifest.get("independent_audits")) != ("COMPLETE_AUDITED", 20, 20):
        raise RuntimeError("E74.1 is not a complete independently audited reference")
    HIST = _load_file("rudag_741_locked_runner_for_dcr742", E741 / "run.py")
    return HIST


def load_dcr():
    global DCR
    if DCR is None:
        DCR = _load_file("dcr_rudag_742_isolated_model", EXP / "dcr_rudag.py")
    return DCR


def source_manifest():
    files = [
        EXP / "config.json", EXP / "run.py", EXP / "dcr_rudag.py", EXP / "audit.py",
        REPO / "main.py", REPO / "model.py", REPO / "network.py", REPO / "layers.py", REPO / "utils.py",
        REPO / "ABIDE-simple-2-concat-weighted-cosine.sh",
        REPO / "data/ABIDE/processed_standard_data.csv", REPO / "data/ABIDE/modal_feat_dict.npy",
        E53 / "run.py", E53 / "cadr_gfr.py", E53 / "config.json", E53 / "manifest.json",
        E61 / "run.py", E61 / "model_e61.py", E61 / "config.json", E61 / "manifest.json",
        E69 / "run.py", E69 / "config.json", E69 / "manifest.json", E69 / "results/fold_metrics.csv", E69 / "results/summary.csv",
        E62 / "run.py", E62 / "config.json", E62 / "manifest.json", E62 / "results/fold_metrics.csv", E62 / "results/summary.csv",
        E74 / "run.py", E74 / "udag.py", E74 / "config.json", E74 / "manifest.json", E74 / "results/fold_metrics.csv", E74 / "results/summary.csv",
        E741 / "run.py", E741 / "rudag.py", E741 / "audit.py", E741 / "config.json", E741 / "manifest.json",
        E741 / "results/fold_metrics.csv", E741 / "results/summary.csv", E741 / "results/mechanism.csv", E741 / "results/interventions.csv",
    ]
    files.extend(sorted((E61 / ".runtime/warmup").glob("fold*.pt")))
    files.extend(sorted((E69 / ".runtime/tasks").glob("E69.3-G2_f*.json")))
    files.extend(sorted((E62 / ".runtime/tasks").glob("E62-E2_f*.json")))
    files.extend(sorted((E74 / ".runtime/tasks").glob("E74-UDAG_f*.json")))
    files.extend(sorted((E741 / ".runtime/tasks").glob("G1-RUDAG-U_f*.json")))
    result = {}
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"required source/reference is missing: {path}")
        result[str(path.resolve())] = sha_file(path)
    return result


def source_sha():
    return sha_object(source_manifest())


def prepare(fold):
    hist = load_history()
    state, warmup, identity = hist.prepare(fold)
    task_path = E741 / ".runtime/tasks" / f"G1-RUDAG-U_f{fold:02d}.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if task.get("status") != "COMPLETE" or task.get("identity") != identity:
        raise RuntimeError(f"E74.1-G1 initialization identity mismatch in fold {fold}")
    expected_overlap = [377, 407] if fold == 2 else []
    if identity["membership"]["train_test_overlap_1based"] != expected_overlap:
        raise RuntimeError(f"unexpected training/test overlap in fold {fold}")
    if identity["train_pair_scope_pass"] is not True or state["train_pairs_np"].shape != (4096, 2):
        raise RuntimeError(f"training relation-pair scope or count failed in fold {fold}")
    return state, warmup, identity


def task_path(fold):
    return TASKS / f"{GROUP}_f{fold:02d}.json"


def best_path(fold):
    return CKPTS / f"{GROUP}_f{fold:02d}.best.pt"


def latest_path(fold):
    return CKPTS / f"{GROUP}_f{fold:02d}.latest.pt"


def log_path(fold):
    return LOGS / f"{GROUP}_f{fold:02d}.jsonl"


def forward(state, training=False, intervention="full", collect=False):
    hist = load_history()
    return load_dcr().forward_state(
        state, hist.load_e69().load_e61(), GROUP, training=training,
        intervention=intervention, fold=state["fold"], collect_diagnostics=collect,
        permutation_seed_base=CFG["method"]["score_permutation_seed_base"],
    )


def _finite_grads(module):
    import torch
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in module.parameters())


def _classification_gradients(state, output):
    import torch
    model = state["model"]
    gate = torch.autograd.grad(output["cls"], model.adaptive_gate_logits, retain_graph=True, allow_unused=True)[0]
    scorer_norms = []
    for i, scorer in enumerate(model.scorers):
        gradients = torch.autograd.grad(output["cls"], tuple(scorer.parameters()),
                                        retain_graph=i < len(model.scorers) - 1, allow_unused=True)
        pieces = [g.detach().square().sum() for g in gradients if g is not None]
        scorer_norms.append(float(torch.sqrt(torch.stack(pieces).sum()).cpu()) if pieces else 0.0)
    return gate.detach().cpu().tolist() if gate is not None else [0.0] * 4, scorer_norms


def _total_loss_gradient_snapshot(state):
    import torch
    model = state["model"]
    gate = model.adaptive_gate_logits.grad
    gate_values = gate.detach().abs().cpu().tolist() if gate is not None else [0.0] * 4
    scorer_norms = []
    for scorer in model.scorers:
        pieces = [parameter.grad.detach().square().sum() for parameter in scorer.parameters() if parameter.grad is not None]
        scorer_norms.append(float(torch.sqrt(torch.stack(pieces).sum()).cpu()) if pieces else 0.0)
    return gate_values, scorer_norms


def graph_preflight_tests():
    import torch
    dcr = load_dcr()
    base = torch.tensor([[0.2, 0.8, 0.0, 0.0], [0.8, 0.3, 0.4, 0.0],
                         [0.0, 0.4, 0.1, 0.7], [0.0, 0.0, 0.7, 0.5]], dtype=torch.float64)
    rows, cols = torch.triu(base > 0, diagonal=1).nonzero(as_tuple=True)
    q = torch.tensor([0.8, 0.2, 0.7], dtype=torch.float64, requires_grad=True)
    corr_gate = torch.tensor(0.99, dtype=torch.float64, requires_grad=True)
    corr, corr_info = dcr.unified_adjacency(base, rows, cols, q, corr_gate, 3)
    if corr_info["row"].numel() != 3 or not torch.equal(corr, corr.T):
        raise RuntimeError("CORR must score all original-positive undirected edges symmetrically")
    if (corr < 0).any() or not torch.equal(corr.diag(), base.diag()):
        raise RuntimeError("unified adjacency must stay nonnegative and preserve its diagonal")
    if ((base == 0) & (corr != 0)).any():
        raise RuntimeError("unified adjacency added an unsupported edge")

    half = torch.full_like(q, 0.5)
    neutral, _ = dcr.unified_adjacency(base, rows, cols, half, corr_gate, 3)
    if not torch.equal(neutral, base) or not torch.equal(neutral - base, torch.zeros_like(base)):
        raise RuntimeError("q=0.5 must produce an exactly zero disease edge increment")
    zero_gate, _ = dcr.unified_adjacency(base, rows, cols, q, corr_gate.new_zeros(()), 3)
    if not torch.equal(zero_gate, base):
        raise RuntimeError("zero CORR gate must recover the base graph")

    # The non-CORR branch must leave valid but unselected edges untouched.
    selected_rows, selected_cols = torch.tensor([0]), torch.tensor([1])
    weak, _ = dcr.unified_adjacency(base, selected_rows, selected_cols,
                                    torch.tensor([0.8], dtype=torch.float64),
                                    torch.tensor(0.001, dtype=torch.float64), 0)
    if weak[1, 2] != base[1, 2] or weak[2, 3] != base[2, 3]:
        raise RuntimeError("non-CORR reweighting escaped its selected effective candidate edges")
    if ((base == 0) & (weak != 0)).any() or (weak < 0).any():
        raise RuntimeError("non-CORR graph changed support or became negative")
    if torch.equal(weak, base):
        raise RuntimeError("selected non-CORR edge did not receive its declared weak update")

    theta = torch.tensor(1e6, dtype=torch.float64)
    if 0.01 * torch.sigmoid(theta) > 0.01:
        raise RuntimeError("non-CORR gate exceeded its structural cap")

    x = torch.zeros((4, 320), dtype=torch.float64)
    x[1, 0] = 3.0
    s = torch.zeros((4, 4), dtype=torch.float64)
    s[0, 1] = s[1, 0] = 2.0
    delta = torch.zeros_like(s)
    delta[0, 1] = delta[1, 0] = 1.0
    positive, bounded_positive = dcr._bounded_disease_residual(s, delta, x)
    delta[0, 1] = delta[1, 0] = -1.0
    negative, _ = dcr._bounded_disease_residual(s, delta, x)
    if positive[0, 0] != 1.5 or negative[0, 0] != -1.5 or positive[0, 0] != -negative[0, 0]:
        raise RuntimeError("signed disease residual direction or base-degree denominator is incorrect")
    if (torch.diagonal(delta) != 0).any():
        raise RuntimeError("disease edge increment diagonal must be zero")
    for start, width in zip((0, 48, 54, 64), (48, 6, 10, 256)):
        block = bounded_positive[:, start:start + width]
        raw = x[:, start:start + width]
        rms = torch.sqrt(raw.square().mean(1, keepdim=True).clamp_min(1e-8))
        if (block.abs() > rms + 1e-8).any():
            raise RuntimeError("bounded DCR residual exceeds the block RMS-tanh limit")

    (corr.square().sum() + weak.square().sum()).backward()
    if q.grad is None or not torch.isfinite(q.grad).all() or q.grad.abs().sum() == 0:
        raise RuntimeError("unified edge weights are disconnected from disease scores")
    if corr_gate.grad is None or corr_gate.grad.abs() == 0:
        raise RuntimeError("CORR adaptive gate has no graph gradient")
    zero = torch.zeros((4, 4), dtype=torch.float64)
    norm = load_history().load_e69().load_e61().load_base().runtime()["utils"].normalize_adj(zero + torch.eye(4, dtype=torch.float64))
    if not torch.isfinite(norm).all():
        raise RuntimeError("zero-degree normalization produced NaN/Inf")
    return {
        "corr_all_valid_edges_scored": True,
        "noncorr_selected_effective_only": True,
        "support_preserved": True,
        "symmetric_nonnegative_and_diagonal_preserved": True,
        "neutral_and_zero_gate_exact": True,
        "signed_residual_and_base_degree": True,
        "bounded_residual_pass": True,
        "zero_degree_normalization_finite": True,
        "edge_score_gate_gradients": True,
    }


def _one_update(state, base_runtime):
    import torch
    hist = load_history()
    hist.zero_optimizers(state)
    out = forward(state, training=True)
    logp = out["logp"].detach().cpu().clone()
    loss = float(out["total"].detach().cpu())
    out["total"].backward()
    if not _finite_grads(state["model"]) or not _finite_grads(state["cam"]):
        raise RuntimeError("one-step preflight update produced non-finite gradients")
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    result = (logp, loss, hist.state_bundle(state), base_runtime.rng_sha(base_runtime.pack_rng()))
    del out
    return result


def preflight():
    import torch
    for path in (TASKS, CKPTS, LOGS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one historical PPU-ZW810E torch.cuda device; available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
    hist = load_history()
    e61 = hist.load_e69().load_e61()
    base_runtime = e61.load_base()
    sources = source_manifest()
    device_name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()

    inheritance = []
    for fold in range(1, 11):
        state, warmup, identity = prepare(fold)
        e741_task = json.loads((E741 / ".runtime/tasks" / f"G1-RUDAG-U_f{fold:02d}.json").read_text(encoding="utf-8"))
        if identity != e741_task["identity"]:
            raise RuntimeError(f"DCR initialization differs from E74.1-G1 in fold {fold}")
        if base_runtime.rng_sha(base_runtime.pack_rng()) != warmup["rng_sha"]:
            raise RuntimeError(f"fold {fold} did not restore its exact E61 epoch-10 RNG")
        optimizers = (("opt_base", 0.11, 0.0038), ("opt_cam", 0.11, 0.0038), ("opt_added", 0.01, 0.001))
        for key, wd, lr in optimizers:
            if any(abs(float(group["weight_decay"]) - wd) > 1e-12 or abs(float(group["lr"]) - lr) > 1e-12 for group in state[key].param_groups):
                raise RuntimeError(f"optimizer settings mismatch in fold {fold}: {key}")
        inheritance.append({
            "fold": fold, "warmup_sha256": identity["warmup_sha256"],
            "warmup_rng_sha": warmup["rng_sha"], "initial_model_cam_sha": identity["model_cam_sha"],
            "initial_optimizer_sha": identity["optimizer_sha"], "split_sha": identity["split_sha"],
            "pair_sha": identity["pair_sha"], "effective_train_sha": identity["effective_train_sha"],
            "overlap_1based": identity["membership"]["train_test_overlap_1based"],
            "e741_g1_initialization_match": True,
        })
        del state
        torch.cuda.empty_cache()

    graph_tests = graph_preflight_tests()
    state, warmup, identity = prepare(1)
    initial = hist.state_bundle(state)
    start_rng = base_runtime.pack_rng()

    # With the residual off, the inherited E74.1 G1 forward must be identical.
    with torch.no_grad():
        legacy = hist.forward(state, "G1-RUDAG-U", training=False)
        dcr_off = forward(state, training=False, intervention="direct_off")
        if not torch.equal(legacy["fused"], dcr_off["fused"]) or not torch.equal(legacy["logp"], dcr_off["logp"]):
            raise RuntimeError("zero-residual DCR path differs from the E74.1-G1 forward")

        neutral = forward(state, training=False, intervention="disease_neutral", collect=True)
        if any(not torch.equal(a, b) for a, b in zip(neutral["unified_adjs"], neutral["original_adjs"])):
            raise RuntimeError("q=0.5 intervention did not exactly recover each base graph")
        if any(torch.count_nonzero(x).item() for x in neutral["info"]["disease_delta_adjs"] + neutral["info"]["disease_residuals_raw"] + neutral["info"]["disease_residuals_bounded"] + neutral["info"]["direct_messages"]):
            raise RuntimeError("q=0.5 intervention did not make DeltaA and DCR residual exactly zero")
        if tuple(neutral["fused"].shape) != (871, 320) or tuple(neutral["logp"].shape) != (871, 2):
            raise RuntimeError("CAM/classifier pathway dimensions changed")

    gcn_calls, cam_calls = [0, 0, 0, 0], [0]
    hooks = [module.register_forward_hook(lambda _m, _i, _o, idx=i: gcn_calls.__setitem__(idx, gcn_calls[idx] + 1)) for i, module in enumerate(state["model"].gcn)]
    hooks.append(state["cam"].register_forward_hook(lambda *_: cam_calls.__setitem__(0, cam_calls[0] + 1)))
    out = forward(state, training=False, collect=True)
    for hook in hooks:
        hook.remove()
    if gcn_calls != [1, 1, 1, 1] or cam_calls != [1]:
        raise RuntimeError(f"DCR formal forward call counts differ: GCN={gcn_calls}, CAM={cam_calls}")

    gate_grad, scorer_grad = _classification_gradients(state, out)
    if not all(math.isfinite(v) for v in gate_grad + scorer_grad) or scorer_grad[3] <= 0:
        raise RuntimeError(f"CORR disease residual has no finite classification gradient to its scorer: {scorer_grad}")
    controlled_hooks = []
    for scorer in state["model"].scorers:
        controlled_hooks.append(scorer.register_forward_hook(
            lambda _module, _inputs, value: value + (torch.full_like(value, 2.1972245773362196) - value.detach())
        ))
    controlled_full = forward(state, training=False)
    controlled_off = forward(state, training=False, intervention="direct_off")
    for hook in controlled_hooks:
        hook.remove()
    controlled_delta_loss = controlled_full["cls"] - controlled_off["cls"]
    controlled_dcr_grads = []
    for index, scorer in enumerate(state["model"].scorers):
        grads = torch.autograd.grad(controlled_delta_loss, tuple(scorer.parameters()),
                                    retain_graph=index < len(state["model"].scorers) - 1, allow_unused=True)
        pieces = [g.detach().square().sum() for g in grads if g is not None]
        controlled_dcr_grads.append(float(torch.sqrt(torch.stack(pieces).sum()).cpu()) if pieces else 0.0)
    if not all(math.isfinite(v) and v > 0 for v in controlled_dcr_grads):
        raise RuntimeError(f"DCR-only controlled classification gradient is missing for a relation scorer: {controlled_dcr_grads}")
    for residual in out["info"]["disease_residuals_bounded"]:
        raw = state["x"]
        cursor = 0
        for width in (48, 6, 10, 256):
            block, raw_block = residual[:, cursor:cursor + width], raw[:, cursor:cursor + width]
            rms = torch.sqrt(raw_block.square().mean(1, keepdim=True).clamp_min(1e-8))
            if (block.abs() > rms + 1e-6).any():
                raise RuntimeError("bounded DCR residual exceeds its modality block RMS")
            cursor += width
    if any(float(identity["effective_gate_values_initial"][i]) > 0.010000001 for i in range(3)):
        raise RuntimeError("a non-CORR gate exceeds the fixed 0.01 cap")
    del out, legacy, dcr_off, neutral, controlled_full, controlled_off, controlled_delta_loss

    first = _one_update(state, base_runtime)
    hist.load_bundle(state, initial)
    base_runtime.restore_rng(start_rng)
    second = _one_update(state, base_runtime)
    if not torch.equal(first[0], second[0]) or first[1] != second[1] or sha_object(first[2]) != sha_object(second[2]) or first[3] != second[3]:
        raise RuntimeError("DCR one-step update replay failed")

    hist.load_bundle(state, initial)
    base_runtime.restore_rng(start_rng)
    resume_file = RT / "preflight_resume.pt"
    atomic_torch(resume_file, {"state": initial, "rng": start_rng})
    saved = torch_load(resume_file)
    hist.load_bundle(state, saved["state"])
    base_runtime.restore_rng(saved["rng"])
    resumed = _one_update(state, base_runtime)
    resume_file.unlink()
    if not torch.equal(first[0], resumed[0]) or first[1] != resumed[1] or sha_object(first[2]) != sha_object(resumed[2]) or first[3] != resumed[3]:
        raise RuntimeError("DCR checkpoint/RNG restore did not reproduce the next optimizer update")

    peak = int(torch.cuda.max_memory_allocated())
    result = {
        "status": "PASS", "device": device_name, "device_count": 1,
        "memory_free_bytes_before": int(free_bytes), "memory_total_bytes": int(total_bytes),
        "measured_preflight_peak_allocated_bytes": peak,
        "graph_tests": graph_tests, "all_fold_warmup_and_e741_g1_identity": inheritance,
        "e741_g1_direct_off_forward_exact": True, "q_half_zero_graph_and_residual_exact": True,
        "classification_gradient_gate_abs": gate_grad, "classification_gradient_scorer_norms": scorer_grad,
        "controlled_DCR_only_classification_scorer_gradient_norms": controlled_dcr_grads,
        "formal_gcn_calls_per_modality": gcn_calls, "cam_calls": cam_calls[0],
        "bounded_residual_pass": True, "noncorr_gate_cap_pass": True,
        "one_step_update_replay": True, "checkpoint_optimizer_rng_restore_replay": True,
        "source_manifest": sources, "source_sha256": sha_object(sources),
    }
    atomic_json(RESULTS / "preflight.json", result)
    previous_manifest_path = EXP / "manifest.json"
    prior_failure_ledger = RESULTS / "prior_training_failures.json"
    if previous_manifest_path.is_file():
        previous = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") == "PARTIAL" and previous.get("failed_tasks"):
            failures = previous["failed_tasks"]
            prior_worker_invocations = sum(int(item.get("attempts", 1)) for item in failures)
            atomic_json(prior_failure_ledger, {
                "failed_worker_invocations": prior_worker_invocations,
                "automatic_retry_failures": previous.get("retries", []),
                "terminal_fold_failures": [{
                    "fold": item.get("fold"), "attempts": item.get("attempts"),
                    "oom": item.get("oom", item.get("oom_detected", False)),
                    "exception": item.get("tail", "").split("Traceback", 1)[-1][-1200:],
                } for item in failures],
                "completed_optimizer_updates": 0,
                "checkpoints_written": 0,
                "repair": "Removed per-epoch autograd.grad over a graph already consumed by total.backward; training now reads the existing .grad buffers.",
            })
    if prior_failure_ledger.is_file():
        prior_ledger = json.loads(prior_failure_ledger.read_text(encoding="utf-8"))
    else:
        prior_ledger = {"failed_worker_invocations": 0, "automatic_retry_failures": [], "terminal_fold_failures": []}
    atomic_json(EXP / "manifest.json", {
        "experiment": CFG["experiment"], "status": "PREFLIGHT_PASS",
        "experiment_root": str(EXP), "preferred_root": CFG["preferred_root"],
        "root_fallback_reason": CFG["root_fallback_reason"], "expected_tasks": 10,
        "completed_tasks": 0, "independent_audits": 0, "preflight": result,
        "prior_training_failures": prior_ledger,
    })
    print(f"PREFLIGHT PASS device={device_name} peak={peak} bytes", flush=True)
    return result


def _checkpoint_payload(state, best, epoch, executed, source_hash, trace):
    return {
        "experiment": CFG["experiment"], "group": GROUP, "fold": state["fold"], "seed": 0,
        "config": CFG, "source_sha256": source_hash, "identity": state["rudag_identity"],
        "best_epoch_1based": epoch + 1, "epochs_executed": executed,
        "metrics": best["metrics"], "test_probability": best["probability"],
        "test_prediction": best["prediction"], "best_state": best["state"],
        "best_losses": best["losses"], "best_training_rng_sha": best["rng_after_train"],
        "best_eval_rng_sha": best["rng_after_eval"], "rng_trace": trace,
        "initial_gate_values": state["rudag_identity"]["effective_gate_values_initial"],
        "initial_gate_theta": state["rudag_identity"]["adaptive_gate_theta_initial"],
    }


def _task_complete(fold, source_hash):
    path, checkpoint = task_path(fold), best_path(fold)
    if not path.is_file() or not checkpoint.is_file():
        return False
    try:
        task = json.loads(path.read_text(encoding="utf-8"))
        return task.get("status") == "COMPLETE" and task.get("source_sha256") == source_hash and task.get("checkpoint_sha256") == sha_file(checkpoint)
    except Exception:
        return False


def train_worker(fold):
    import numpy as np
    import torch
    if fold not in range(1, 11):
        raise ValueError(f"invalid formal fold {fold}")
    sources = source_manifest()
    source_hash = sha_object(sources)
    preflight_record = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight_record.get("status") != "PASS" or preflight_record.get("source_sha256") != source_hash:
        raise RuntimeError("formal worker source hashes differ from passing preflight")
    state, warmup, identity = prepare(fold)
    state["fold"] = fold
    state["rudag_identity"] = identity
    task_file, best_file, latest_file = task_path(fold), best_path(fold), latest_path(fold)
    if task_file.exists():
        existing = json.loads(task_file.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE" and existing.get("source_sha256") == source_hash and existing.get("checkpoint_sha256") == sha_file(best_file):
            print(f"SKIP already complete fold={fold}", flush=True)
            return existing
        raise RuntimeError(f"refusing to overwrite incompatible task record: {task_file}")

    hist = load_history()
    base_runtime = hist.load_e69().load_e61().load_base()
    if base_runtime.rng_sha(base_runtime.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold} did not start from the E61 epoch-10 RNG")

    start_epoch, wait = 10, 0
    best_acc, best_epoch, best, trace = -1.0, -1, None, []
    if latest_file.exists():
        saved = torch_load(latest_file)
        for key, expected in (("fold", fold), ("group", GROUP), ("source_sha256", source_hash),
                              ("identity", identity), ("warmup_sha256", identity["warmup_sha256"])):
            if saved.get(key) != expected:
                raise RuntimeError(f"incompatible recovery state {key} for fold {fold}")
        hist.load_bundle(state, saved["current"])
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
        hist.zero_optimizers(state)
        out = forward(state, training=True)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"non-finite DCR loss at fold={fold} epoch={epoch + 1}")
        out["total"].backward()
        if not _finite_grads(state["model"]) or not _finite_grads(state["cam"]):
            raise FloatingPointError(f"non-finite DCR gradient at fold={fold} epoch={epoch + 1}")
        gate_grad, scorer_grad = _total_loss_gradient_snapshot(state)
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        rng_after_train = base_runtime.rng_sha(base_runtime.pack_rng())
        executed = epoch + 1
        with torch.no_grad():
            evaluated = forward(state, training=False)
            metrics = base_runtime.classification_metrics(evaluated["logp"], state["test"], state["y"])
        rng_after_eval = base_runtime.rng_sha(base_runtime.pack_rng())
        epoch_rng = {"epoch": epoch + 1, "rng_before": rng_before,
                     "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval}
        trace.append(epoch_rng)

        if metrics["acc"] > best_acc:
            best_acc, best_epoch, wait = float(metrics["acc"]), epoch, 0
            best = {
                "state": hist.state_bundle(state),
                "metrics": {key: float(value) for key, value in metrics.items() if key not in ("pred", "prob")},
                "probability": metrics["prob"].copy(), "prediction": metrics["pred"].copy(),
                "losses": {key: float(out[key].detach().cpu()) for key in ("cls", "graph", "pair", "adv", "preserve", "total")},
                "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval,
                "total_loss_gate_grad": gate_grad, "total_loss_scorer_grad": scorer_grad,
            }
            atomic_torch(best_file, _checkpoint_payload(state, best, best_epoch, executed, source_hash, trace))
        else:
            wait += 1

        if executed % 10 == 0 or executed == CFG["training"]["max_epochs"] or wait > CFG["training"]["early_stop_wait_gt"]:
            atomic_torch(latest_file, {
                "group": GROUP, "fold": fold, "source_sha256": source_hash, "identity": identity,
                "warmup_sha256": identity["warmup_sha256"], "current": hist.state_bundle(state),
                "rng": base_runtime.pack_rng(), "next_epoch": executed, "wait": wait,
                "best_acc": best_acc, "best_epoch": best_epoch, "best": best, "trace": trace,
            })
            gate_values = [float(0.01 * torch.sigmoid(state["model"].adaptive_gate_logits[i]).detach().cpu()) if i < 3
                           else float(torch.sigmoid(state["model"].adaptive_gate_logits[i]).detach().cpu()) for i in range(4)]
            record = {
                "group": GROUP, "fold": fold, "epoch": executed, "best_epoch_1based": best_epoch + 1,
                "best_acc": best_acc, "metrics": {key: float(metrics[key]) for key in METRICS},
                "loss": float(out["total"].detach().cpu()), "gate_values": gate_values,
                "total_loss_gate_grad_abs": [abs(float(v)) for v in gate_grad],
                "total_loss_scorer_grad_norms": scorer_grad, "rng": epoch_rng,
                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            }
            with log_path(fold).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(f"{GROUP} fold={fold} epoch={executed} best={best_acc:.6f}@{best_epoch + 1} wait={wait}", flush=True)
        del out, evaluated
        if wait > CFG["training"]["early_stop_wait_gt"]:
            break
    if best is None:
        raise RuntimeError(f"no ACC-selected best checkpoint for fold {fold}")
    checkpoint = _checkpoint_payload(state, best, best_epoch, executed, source_hash, trace)
    atomic_torch(best_file, checkpoint)
    task = {
        "status": "COMPLETE", "group": GROUP, "fold": fold, "seed": 0,
        "source_sha256": source_hash, "identity": identity,
        "checkpoint_sha256": sha_file(best_file), "best_epoch_1based": best_epoch + 1,
        "epochs_executed": executed, "metrics": best["metrics"], "rng_trace": trace,
        "wall_seconds": time.time() - started, "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "device": torch.cuda.get_device_name(0),
    }
    atomic_json(task_file, task)
    print(f"COMPLETE {GROUP} fold={fold} best={best_acc:.6f}@{best_epoch + 1}", flush=True)
    return task


def run_all(workers=10):
    import torch
    workers = max(1, min(10, int(workers)))
    preflight_record = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight_record.get("status") != "PASS" or source_sha() != preflight_record.get("source_sha256"):
        raise RuntimeError("formal training requires the exact passing preflight sources")
    prior_failure_path = RESULTS / "prior_training_failures.json"
    prior_failures = json.loads(prior_failure_path.read_text(encoding="utf-8")) if prior_failure_path.is_file() else {
        "failed_worker_invocations": 0, "automatic_retry_failures": [], "terminal_fold_failures": []
    }
    free, total = torch.cuda.mem_get_info()
    peak = max(1, int(preflight_record["measured_preflight_peak_allocated_bytes"]))
    per_worker = int(peak * 1.4)
    memory_cap = max(1, int(free * 0.80 / per_worker))
    worker_limit = min(workers, memory_cap)
    pending = [fold for fold in range(1, 11) if not _task_complete(fold, preflight_record["source_sha256"])]
    retries, failures, active = [], [], {}
    attempts = {fold: 0 for fold in pending}
    maximum = 0
    started = time.time()
    controller = LOGS / "controller.jsonl"
    with controller.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "START", "pending": pending, "worker_limit": worker_limit,
                                "memory_free": int(free), "memory_total": int(total), "preflight_peak": peak,
                                "device": torch.cuda.get_device_name(0)}) + "\n")
        stream.flush()
        while pending or active:
            while pending and len(active) < worker_limit:
                fold = pending.pop(0)
                attempts[fold] += 1
                logfile = LOGS / f"{GROUP}_f{fold:02d}.stdout.log"
                output = logfile.open("a", encoding="utf-8")
                env = os.environ.copy()
                env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                            "NUMEXPR_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_VISIBLE_DEVICES": "0"})
                proc = subprocess.Popen([sys.executable, str(EXP / "run.py"), "--worker", "--fold", str(fold)],
                                        cwd=str(EXP), env=env, stdout=output, stderr=subprocess.STDOUT)
                active[proc] = (fold, output, logfile)
                maximum = max(maximum, len(active))
            time.sleep(2)
            for proc, (fold, output, logfile) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                output.close()
                del active[proc]
                if code == 0 and _task_complete(fold, preflight_record["source_sha256"]):
                    event = {"event": "COMPLETE", "fold": fold, "attempt": attempts[fold]}
                else:
                    try:
                        tail = logfile.read_text(encoding="utf-8", errors="replace")[-8000:]
                    except Exception:
                        tail = "unable to read worker log"
                    oom = any(term in tail.lower() for term in ("out of memory", "memory allocation", "resource exhausted", "oom"))
                    retry_limit = 2 if oom else 1
                    if attempts[fold] <= retry_limit:
                        retries.append({"fold": fold, "attempt": attempts[fold], "exit_code": code, "oom": oom, "tail": tail})
                        pending.append(fold)
                        if oom:
                            worker_limit = max(1, worker_limit // 2)
                    else:
                        failures.append({"fold": fold, "attempts": attempts[fold], "exit_code": code, "oom": oom, "tail": tail})
                    event = {"event": "FAIL_OR_RETRY", "fold": fold, "attempt": attempts[fold], "exit_code": code, "oom": oom}
                stream.write(json.dumps(event) + "\n")
                stream.flush()
            manifest = {
                "experiment": CFG["experiment"], "status": "TRAINING" if not failures else "PARTIAL",
                "experiment_root": str(EXP), "preferred_root": CFG["preferred_root"],
                "root_fallback_reason": CFG["root_fallback_reason"], "expected_tasks": 10,
                "completed_tasks": sum(_task_complete(f, preflight_record["source_sha256"]) for f in range(1, 11)),
                "independent_audits": 0, "active_tasks": [],
                "failed_attempt_records": len(retries), "failed_tasks": failures, "retries": retries,
                "prior_training_failures": prior_failures,
                "max_workers_observed": maximum, "worker_limit": worker_limit,
                "device": torch.cuda.get_device_name(0), "oom_events": sum(item["oom"] for item in retries + failures),
            }
            manifest["active_tasks"] = [str(value[0]) for value in active.values()]
            atomic_json(EXP / "manifest.json", manifest)
        stream.write(json.dumps({"event": "STOP", "wall_seconds": time.time() - started,
                                 "completed": sum(_task_complete(f, preflight_record["source_sha256"]) for f in range(1, 11)),
                                 "failures": failures, "retries": retries, "max_workers": maximum}) + "\n")
        stream.flush()
    if failures:
        raise RuntimeError(f"formal training failures remain: {failures}")
    completed = sum(_task_complete(f, preflight_record["source_sha256"]) for f in range(1, 11))
    if completed != 10:
        raise RuntimeError(f"formal training produced only {completed}/10 complete folds")
    training = {
        "max_workers_observed": maximum, "worker_limit": worker_limit,
        "retry_count": len(retries), "retries": retries, "failures": failures,
        "prior_failure_batches": [prior_failures] if prior_failures.get("failed_worker_invocations", 0) else [],
        "prior_failed_worker_invocations": int(prior_failures.get("failed_worker_invocations", 0)),
        "oom_events": sum(item["oom"] for item in retries + failures), "wall_seconds": time.time() - started,
    }
    atomic_json(EXP / "manifest.json", {
        "experiment": CFG["experiment"], "status": "TRAINED_AWAITING_AUDIT",
        "experiment_root": str(EXP), "preferred_root": CFG["preferred_root"],
        "root_fallback_reason": CFG["root_fallback_reason"], "expected_tasks": 10,
        "completed_tasks": 10, "independent_audits": 0, "device": torch.cuda.get_device_name(0),
        "worker_limit": worker_limit, "max_workers_observed": maximum,
        "failed_attempt_records": len(retries) + int(prior_failures.get("failed_worker_invocations", 0)), "failed_tasks": failures,
        "prior_training_failures": prior_failures,
        "training_execution": training, "warmup_retrained": False,
        "source_sha256": preflight_record["source_sha256"],
    })
    module = _load_file("dcr_rudag_742_independent_audit", EXP / "audit.py")
    return module.finalize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    for path in (TASKS, CKPTS, LOGS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        preflight()
    elif args.worker:
        train_worker(args.fold)
    elif args.audit:
        module = _load_file("dcr_rudag_742_independent_audit", EXP / "audit.py")
        module.finalize()
    else:
        run_all(args.workers)


if __name__ == "__main__":
    main()
