#!/usr/bin/env python3
"""Pre-training parity checks and independent Experiment 77 checkpoint audits."""
from __future__ import annotations

import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import run as runner
from memory import ECPSM
from model import cam_boundary_hook, forward_with_memory


HERE = Path(__file__).resolve().parent
BASE = Path(runner.CFG["base"]["experiment_root"])
RESULTS = HERE / "results"
CHECKPOINTS = HERE / "checkpoints"
GROUPS = runner.GROUPS
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
PRIMARY = ("acc", "hard_auc", "prob_auc", "sen", "spe")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _mean_sd(values) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def _percent_metrics(metrics: dict) -> dict:
    return {key: float(metrics[key]) * 100.0 for key in METRICS if key in metrics}


def _metrics_from_output(runtime, output, state) -> dict:
    return runtime.classification_metrics(output["logp"], state["test"], state["y"])


def _step(state: dict, ecpsm: ECPSM, tracker, group: str, memory_epoch: int,
          runtime, allowed: torch.Tensor) -> tuple[dict, dict, dict]:
    before = runtime.rng_sha(runtime.pack_rng())
    runner.E74.load_history().zero_optimizers(state)
    output, hook = forward_with_memory(runner.E74, state, ecpsm, group,
                                       training=True, memory_epoch=memory_epoch)
    _check(tuple(output["logp"].shape) == (871, 2) and tuple(output["hidden"].shape) == (871, 36),
           "inherited CAM/classifier output dimensions changed in E77 dry-run")
    info = hook.get("info", {})
    _check(tuple(info.get("residual", torch.empty(0)).shape) == (871, 144),
           "EC-PSM residual did not preserve the 144-D classifier input")
    _check(bool(torch.isfinite(info["residual"]).all()) and float(info["residual"].detach().abs().max()) <= 0.050001,
           "EC-PSM residual is non-finite or exceeds its configured bound")
    _check(bool(torch.isfinite(output["total"])), "non-finite dry-run loss")
    output["total"].backward()
    _check(runner._finite_gradients((state["model"], state["cam"], ecpsm)),
           "non-finite dry-run gradient")
    _check(any(parameter.grad is not None for parameter in ecpsm.parameters()),
           "EC-PSM received no gradient in dry-run")
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    after_train = runtime.rng_sha(runtime.pack_rng())
    capture: dict = {}
    with torch.no_grad():
        evaluated, _ = forward_with_memory(runner.E74, state, ecpsm, group,
                                           training=False, memory_epoch=memory_epoch,
                                           capture=capture)
    after_eval = runtime.rng_sha(runtime.pack_rng())
    if group == "E77-G2":
        error_state = tracker.update(
            evaluated["logp"], state["y"], allowed,
            min_votes=runner.CFG["method"]["hard_memory"]["stable_error_votes"],
        )
        if memory_epoch >= runner.CFG["method"]["hard_memory"]["warmup_ecpsm_epochs"]:
            ecpsm.update_hard_memory(capture["cam_features"], error_state["fp_ids"],
                                     error_state["fn_ids"], allowed)
    ecpsm.update_class_memory(capture["cam_features"], state["y"], allowed)
    after_memory = runtime.rng_sha(runtime.pack_rng())
    _check(after_memory == after_eval, "memory update advanced the inherited public RNG")
    return output, evaluated, {
        "rng_before": before, "rng_after_train": after_train,
        "rng_after_eval": after_eval, "rng_after_memory": after_memory,
    }


def _run_warmup(fold: int, group: str, *, check_resume: bool = False) -> dict:
    state, warmup, identity, base_task, ecpsm, tracker, old_groups, runtime = runner.initialize_group(fold, group)
    allowed_np = runner.allowed_train_ids(state)
    allowed = torch.as_tensor(allowed_np, dtype=torch.long, device=state["x"].device)
    initial_memory = runner._initial_memory(state, ecpsm, group)
    initial_rng = runtime.rng_sha(runtime.pack_rng())
    _check(initial_rng == warmup["rng_sha"], f"fold {fold} {group} changed restored E74 RNG during setup")
    class_valid = ecpsm.class_valid.cpu().numpy()
    allowed_mask = np.zeros(runner.CFG["data"]["subjects"], dtype=bool)
    allowed_mask[allowed_np] = True
    _check(np.array_equal(class_valid, allowed_mask), "class memory contains IDs outside effective train-minus-test")
    test_ids = np.asarray(state["test_np"], dtype=np.int64)
    _check(not bool(ecpsm.class_valid[test_ids].any()), "explicit test ID entered the class prototype bank")

    reference_trace = {item["epoch"]: item for item in base_task["rng_trace"]}
    signatures = []
    resume_record = None
    for local_epoch in range(10):
        if check_resume and local_epoch == 1 and resume_record is not None:
            continue
        output, evaluated, trace = _step(state, ecpsm, tracker, group, local_epoch, runtime, allowed)
        epoch_number = int(runner.CFG["base"]["start_epoch_zero_based"]) + local_epoch + 1
        trace["epoch"] = epoch_number
        expected = reference_trace.get(epoch_number)
        _check(expected is not None, f"missing E74 reference RNG trace at epoch {epoch_number}")
        for key in ("rng_before", "rng_after_train", "rng_after_eval"):
            _check(trace[key] == expected[key],
                   f"fold {fold} {group} RNG mismatch at base epoch {epoch_number} ({key})")
        signature = runner._signature(state, ecpsm, old_groups, evaluated, trace)
        signatures.append(signature)

        if check_resume and local_epoch == 0:
            saved = {
                "base_state": runner.E74.load_history().state_bundle(state),
                "ecpsm_state": runner.cpu_state(ecpsm),
                "tracker_state": runner.cpu_state(tracker),
                "rng": runtime.pack_rng(),
                "next_epoch": local_epoch + 1,
                "fold": fold,
                "group": group,
                "warmup_sha256": identity["warmup_sha256"],
            }
            resume_path = HERE / "runtime" / "preflight_resume.pt"
            runner.atomic_torch(resume_path, saved)
            decoded = torch.load(resume_path, map_location=state["x"].device, weights_only=False)
            next_original = _step(state, ecpsm, tracker, group, 1, runtime, allowed)
            next_epoch_number = int(runner.CFG["base"]["start_epoch_zero_based"]) + 2
            next_original[2]["epoch"] = next_epoch_number
            next_signature = runner._signature(state, ecpsm, old_groups, next_original[1], next_original[2])
            expected_next = reference_trace.get(next_epoch_number)
            _check(expected_next is not None and all(next_original[2][key] == expected_next[key]
                   for key in ("rng_before", "rng_after_train", "rng_after_eval")),
                   f"fold {fold} resume replay did not follow E74 RNG trace at epoch {next_epoch_number}")
            runner.E74.load_history().load_bundle(state, decoded["base_state"])
            ecpsm.load_state_dict(decoded["ecpsm_state"], strict=True)
            tracker.load_state_dict(decoded["tracker_state"], strict=True)
            runtime.restore_rng(decoded["rng"])
            next_resumed = _step(state, ecpsm, tracker, group, 1, runtime, allowed)
            next_resumed[2]["epoch"] = next_epoch_number
            resumed_signature = runner._signature(state, ecpsm, old_groups, next_resumed[1], next_resumed[2])
            _check(next_signature == resumed_signature,
                   "serialized E77 base/optimizer/memory/tracker/RNG recovery did not reproduce the next epoch")
            signatures.append(resumed_signature)
            resume_record = {
                "status": "PASS", "checkpoint_sha256": runner.sha_file(resume_path),
                "next_epoch_signature_sha256": runner.digest_value(next_signature),
                "restored_next_epoch": 2,
            }
            del next_original, next_resumed, next_signature, resumed_signature, decoded
            gc.collect()

    summary = {
        "fold": fold, "group": group, "warmup_signatures": signatures,
        "initial_memory": initial_memory,
        "memory_allowed_ids": int(len(allowed_np)),
        "test_ids_excluded": int(len(state["train_np"]) - len(allowed_np)),
        "rng_restored_sha256": initial_rng,
        "resume_audit": resume_record,
    }
    del state, ecpsm, tracker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _base_disabled_parity() -> dict:
    base_state, warmup, identity = runner.E74.prepare(1)
    base_state["fold"] = 1
    base_state["group"] = runner.BASE_GROUP
    runtime = runner.E74.load_history().load_e69().load_e61().load_base()
    start_rng = runtime.pack_rng()
    _check(runtime.rng_sha(start_rng) == warmup["rng_sha"], "fold 1 base parity did not start from audited E61 RNG")
    base_group_count = len(base_state["opt_added"].param_groups)
    runner.E74.load_history().zero_optimizers(base_state)
    base_output = runner.E74.forward(base_state, runner.BASE_GROUP, training=True, collect=False)
    base_output["total"].backward()
    for key in ("opt_base", "opt_cam", "opt_added"):
        base_state[key].step()
    base_rng_after = runtime.rng_sha(runtime.pack_rng())
    base_public_sha = runner.public_state_sha(base_state, base_group_count)

    e77_state, _, _, _, ecpsm, _, old_groups, e77_runtime = runner.initialize_group(1, "E77-G1")
    e77_runtime.restore_rng(start_rng)
    rng_before = e77_runtime.rng_sha(e77_runtime.pack_rng())
    runner.E74.load_history().zero_optimizers(e77_state)
    query_ids = torch.arange(runner.CFG["data"]["subjects"], device=e77_state["x"].device)
    with cam_boundary_hook(e77_state["cam"], ecpsm, query_ids, enabled=False):
        e77_output = runner.E74.forward(e77_state, runner.BASE_GROUP, training=True, collect=False)
    e77_output["total"].backward()
    _check(runner._finite_gradients((e77_state["model"], e77_state["cam"], ecpsm)),
           "disabled-hook parity had non-finite gradients")
    for key in ("opt_base", "opt_cam", "opt_added"):
        e77_state[key].step()
    e77_rng_after = e77_runtime.rng_sha(e77_runtime.pack_rng())
    e77_public_sha = runner.public_state_sha(e77_state, old_groups)
    result = {
        "fold": 1,
        "initial_identity": identity,
        "same_initial_rng": rng_before == warmup["rng_sha"],
        "logp_equal": torch.equal(base_output["logp"], e77_output["logp"]),
        "loss_equal": torch.equal(base_output["total"], e77_output["total"]),
        "public_parameters_optimizer_equal": base_public_sha == e77_public_sha,
        "rng_after_train_equal": base_rng_after == e77_rng_after,
        "base_rng_after_train": base_rng_after,
        "e77_rng_after_train": e77_rng_after,
        "public_state_sha256": base_public_sha,
        "new_parameters_unchanged": all(parameter.grad is None for parameter in ecpsm.parameters()),
    }
    _check(all(result[key] for key in (
        "same_initial_rng", "logp_equal", "loss_equal", "public_parameters_optimizer_equal",
        "rng_after_train_equal", "new_parameters_unchanged")),
        f"base-disabled E77 parity failed: {result}")
    del base_state, e77_state, ecpsm
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _loo_formula_test() -> dict:
    module = ECPSM(torch.zeros(8, 1024), n_subjects=871)
    vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    features = torch.zeros(871, 2)
    features[:4] = vectors
    valid = torch.zeros(871, dtype=torch.bool)
    valid[:4] = True
    assignments = torch.full((871,), -1, dtype=torch.long)
    assignments[:4] = torch.tensor([0, 0, 1, 1])
    sums = torch.stack((vectors[:2].sum(0), vectors[2:].sum(0)))
    counts = torch.tensor([2, 2])
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(770077)
        module = ECPSM(torch.zeros(8, 1024), n_subjects=871)
    result = module._retrieve(vectors[:1], torch.tensor([0]), features, valid, assignments, sums, counts)
    expected_centers = torch.nn.functional.normalize(torch.stack((vectors[1], vectors[2:].mean(0))), dim=-1)
    actual_sim = torch.nn.functional.normalize(vectors[:1], dim=-1) @ expected_centers.T
    weights = torch.softmax(actual_sim / module.temperature, dim=-1)
    expected = (weights[:, :, None] * expected_centers[None]).sum(dim=1)
    _check(torch.allclose(result[0], expected, atol=1e-6, rtol=0),
           "training-query retrieval did not subtract the query subject's actual cluster contribution")
    del module
    return {"status": "PASS", "shared_cluster_subjects": 2, "self_vector_removed": True}


def preflight() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    source_hashes = runner.source_manifest()
    source_hash = runner.digest_value(source_hashes)
    base = runner.CFG["base"]
    _check(runner.sha_file(BASE / "run.py") == base["runner_sha256"], "E74.4 runner SHA mismatch")
    _check(runner.sha_file(BASE / base["model_file"]) == base["model_sha256"], "E74.4 model SHA mismatch")
    _check(runner.sha_file(BASE / "config.json") == base["config_sha256"], "E74.4 config SHA mismatch")
    _check(runner.sha_file(BASE / "manifest.json") == base["manifest_sha256"], "E74.4 manifest SHA mismatch")
    device_name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)

    # Every fold must resolve to the archived E74.4 task and the matching E61 warmup identity.
    inheritance = []
    for fold in range(1, 11):
        state, warmup, identity, task, ecpsm, tracker, _, runtime = runner.initialize_group(fold, "E77-G1")
        expected_overlap = [376, 406] if fold == 2 else []
        actual_overlap = sorted(set(np.asarray(state["train_np"], dtype=np.int64).tolist()) &
                                set(np.asarray(state["test_np"], dtype=np.int64).tolist()))
        _check(actual_overlap == expected_overlap, f"fold {fold} overlap differs from the archived E74.4 audit")
        _check(identity.get("membership", {}).get("train_test_overlap_1based") ==
               ([377, 407] if fold == 2 else []), f"fold {fold} overlap audit identity mismatch")
        _check(runtime.rng_sha(runtime.pack_rng()) == warmup["rng_sha"],
               f"fold {fold} did not restore exact E61 warmup RNG")
        _check(task.get("status") == "COMPLETE" and task.get("identity") == identity,
               f"fold {fold} E74.4 task is not the audited G2 run")
        inheritance.append({
            "fold": fold, "base_task_status": task["status"],
            "warmup_sha256": identity["warmup_sha256"],
            "warmup_rng_sha256": identity["warmup_rng_sha"],
            "initial_model_cam_sha256": identity["model_cam_sha"],
            "initial_optimizer_sha256": identity["optimizer_sha"],
            "split_sha256": identity["split_sha"],
            "pair_sha256": identity["pair_sha"],
            "allowed_train_ids": int(len(runner.allowed_train_ids(state))),
            "test_train_overlap_1based": [int(i + 1) for i in actual_overlap],
            "memory_init_preserved_rng": runtime.rng_sha(runtime.pack_rng()) == warmup["rng_sha"],
        })
        del state, ecpsm, tracker
        gc.collect()
        torch.cuda.empty_cache()

    disabled = _base_disabled_parity()
    loo = _loo_formula_test()
    warmup_audits = []
    for fold in range(1, 11):
        g1 = _run_warmup(fold, "E77-G1", check_resume=(fold == 1))
        g2 = _run_warmup(fold, "E77-G2")
        parity = runner._same_warmup_signatures(g1["warmup_signatures"], g2["warmup_signatures"])
        _check(parity["status"] == "PASS", f"G1/G2 common warmup parity failed in fold {fold}: {parity}")
        warmup_audits.append({
            "fold": fold, "status": "PASS", "epochs": 10,
            "g1_rng_reference_matches": 10, "g2_rng_reference_matches": 10,
            "g1_g2_signature_match": True,
            "g1_initial_memory_sha256": g1["initial_memory"]["logp_sha256"],
            "g2_initial_memory_sha256": g2["initial_memory"]["logp_sha256"],
            "resume_audit": g1["resume_audit"] if fold == 1 else None,
        })
        _check(g1["initial_memory"]["logp_sha256"] == g2["initial_memory"]["logp_sha256"],
               f"fold {fold} G1/G2 initial class memory forward differs")

    result = {
        "status": "PASS", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_sha256": source_hash, "source_hashes": source_hashes,
        "device": device_name, "device_count": torch.cuda.device_count(),
        "memory_free_bytes_before": int(free_bytes), "memory_total_bytes": int(total_bytes),
        "protected_sources_unchanged_baseline": source_hashes,
        "fold_inheritance": inheritance,
        "base_disabled_one_step_parity": disabled,
        "leave_one_subject_out_cluster_test": loo,
        "g1_g2_warmup_parity": warmup_audits,
        "training_gate": "PASS; formal training authorized by audited RNG/forward/optimizer parity",
    }
    runner.atomic_json(RESULTS / "preflight.json", result)
    print(json.dumps({"status": "PASS", "source_sha256": source_hash,
                      "folds": len(inheritance), "warmup_parity": len(warmup_audits),
                      "device": device_name}, ensure_ascii=True), flush=True)
    return result


def _read_base_metrics() -> dict[int, dict]:
    path = BASE / "results" / "fold_metrics.csv"
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row["group"] == runner.BASE_GROUP]
    _check(len(selected) == 10, f"expected ten audited E74.4-G2 rows in {path}, got {len(selected)}")
    return {int(row["fold"]): {metric: float(row[metric]) for metric in METRICS} for row in selected}


def _load_checkpoint(fold: int, group: str) -> tuple[dict, ECPSM, object, object]:
    path = CHECKPOINTS / f"{group}_f{fold:02d}.best.pt"
    task_path = runner.task_path(group, fold)
    _check(path.is_file() and task_path.is_file(), f"missing formal checkpoint/task for {group} fold {fold}")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    checkpoint_sha = runner.sha_file(path)
    _check(task.get("status") == "COMPLETE" and task.get("checkpoint_sha256") == checkpoint_sha,
           f"checkpoint/task hash or status mismatch for {group} fold {fold}")
    _check(task.get("source_sha256") == json.loads((RESULTS / "preflight.json").read_text())["source_sha256"],
           f"source SHA mismatch for {group} fold {fold}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state, _, identity, _, ecpsm, tracker, _, runtime = runner.initialize_group(fold, group)
    _check(checkpoint.get("base_identity") == identity, f"initialization identity mismatch {group} fold {fold}")
    _check(checkpoint.get("warmup_sha256") == identity["warmup_sha256"], "warmup SHA mismatch")
    runner.E74.load_history().load_bundle(state, checkpoint["base_state"])
    ecpsm.load_state_dict(checkpoint["ecpsm_state"], strict=True)
    tracker.load_state_dict(checkpoint["tracker_state"], strict=True)
    runtime.restore_rng(checkpoint["rng"])
    state["model"].eval(); state["cam"].eval(); ecpsm.eval()
    return checkpoint, ecpsm, state, runtime


def _predict_loaded(checkpoint: dict, ecpsm: ECPSM, state: dict, runtime,
                    intervention: str = "full") -> tuple[dict, dict]:
    memory_epoch = int(checkpoint["best_epoch_1based"]) - 1 - int(runner.CFG["base"]["start_epoch_zero_based"])
    options = {
        "full": {}, "hard_off": {"hard_off": True},
        "semantic_swap": {"semantic_swap": True},
        "prototype_off": {"prototype_off": True},
    }[intervention]
    output, hook = forward_with_memory(
        runner.E74, state, ecpsm, checkpoint["group"], training=False,
        memory_epoch=memory_epoch, capture={}, **options,
    )
    metrics = _metrics_from_output(runtime, output, state)
    return metrics, hook.get("info", {})


def _independent_checkpoint_audit(group: str, fold: int, base_metrics: dict) -> tuple[dict, dict, dict]:
    checkpoint, ecpsm, state, runtime = _load_checkpoint(fold, group)
    checkpoint_ecpsm = runner.cpu_state(ecpsm)
    rng_before = runtime.rng_sha(runtime.pack_rng())
    full, info = _predict_loaded(checkpoint, ecpsm, state, runtime, "full")
    rng_after = runtime.rng_sha(runtime.pack_rng())
    _check(all(abs(float(full[key]) - float(checkpoint["metrics"][key])) <= 1e-10 for key in METRICS),
           f"independent checkpoint metric mismatch for {group} fold {fold}")
    _check(np.array_equal(full["pred"], checkpoint["test_prediction"]),
           f"independent hard prediction mismatch for {group} fold {fold}")
    _check(np.allclose(full["prob"], checkpoint["test_probability"], atol=1e-8, rtol=0),
           f"independent probability mismatch for {group} fold {fold}")
    _check(rng_after == rng_before, f"eval-only checkpoint audit consumed public RNG for {group} fold {fold}")
    _check(len(checkpoint["rng"]) > 0 and checkpoint.get("scheduler_state") is None,
           "best checkpoint lacks recoverable RNG or inherited scheduler metadata")

    if group == "E77-G2":
        interventions = {}
        for name in ("hard_off", "semantic_swap", "prototype_off"):
            ecpsm.load_state_dict(checkpoint_ecpsm, strict=True)
            state["model"].eval(); state["cam"].eval()
            changed, _ = _predict_loaded(checkpoint, ecpsm, state, runtime, name)
            interventions[name] = {
                **_percent_metrics(changed),
                "prediction_change_rate": float(np.mean(changed["pred"] != full["pred"])),
                "mean_abs_positive_probability_change": float(np.mean(np.abs(changed["prob"] - full["prob"]))),
            }
    else:
        interventions = {}

    task = json.loads(runner.task_path(group, fold).read_text(encoding="utf-8"))
    memory_summary = ecpsm.usage_summary()
    hard_rows = checkpoint.get("hard_diagnostics", [])
    hard_counts = {
        name: int(sum(int(item.get(key, 0)) for item in hard_rows))
        for name, key in (("stable_fp_epoch_observations", "fp_true"),
                          ("stable_fn_epoch_observations", "fn_true"),
                          ("near_fp_epoch_observations", "near_fp"),
                          ("near_fn_epoch_observations", "near_fn"))
    }
    hard_counts["hard_diagnostics_epochs"] = len(hard_rows)
    memory_record = {
        "group": group, "fold": fold, "selected_epoch_1based": int(checkpoint["best_epoch_1based"]),
        "class": memory_summary["class"], "hard": memory_summary["hard"],
        "error_observations": hard_counts,
        "initial_memory": checkpoint.get("initial_memory", {}),
        "task_allowed_memory_ids": int(task["allowed_memory_ids"]),
        "test_ids_excluded_from_memory": int(task["test_ids_excluded_from_memory"]),
    }
    metric_row = {"group": group, "fold": fold, **_percent_metrics(full),
                  "best_epoch_1based": int(checkpoint["best_epoch_1based"]),
                  "epochs_executed": int(task["epochs_executed"]),
                  "checkpoint_sha256": runner.sha_file(CHECKPOINTS / f"{group}_f{fold:02d}.best.pt"),
                  "train_test_overlap_1based": json.dumps(task["train_test_overlap_1based"]),
                  "base_acc": float(base_metrics[fold]["acc"]) * 100,
                  "base_hard_auc": float(base_metrics[fold]["hard_auc"]) * 100,
                  "base_prob_auc": float(base_metrics[fold]["prob_auc"]) * 100,
                  "base_sen": float(base_metrics[fold]["sen"]) * 100,
                  "base_spe": float(base_metrics[fold]["spe"]) * 100}
    for metric in PRIMARY:
        metric_row[f"delta_vs_e74_g2_{metric}_pp"] = metric_row[metric] - metric_row[f"base_{metric}"]
    audit_row = {
        "group": group, "fold": fold, "status": "PASS",
        "checkpoint_sha256": metric_row["checkpoint_sha256"],
        "best_epoch_1based": int(checkpoint["best_epoch_1based"]),
        "metrics_reproduced": True, "predictions_reproduced": True,
        "probabilities_reproduced": True, "rng_eval_unchanged": True,
        "test_count": int(len(checkpoint["test_indices"])),
        "train_test_overlap_1based": json.dumps(task["train_test_overlap_1based"]),
    }
    del state, ecpsm
    gc.collect(); torch.cuda.empty_cache()
    return metric_row, audit_row, {"memory": memory_record, "interventions": interventions}


def _summary_rows(fold_rows: list[dict], base_metrics: dict) -> tuple[list[dict], list[dict]]:
    summaries, paired = [], []
    for group in GROUPS:
        rows = [row for row in fold_rows if row["group"] == group]
        summary = {"group": group, "folds": len(rows)}
        for metric in PRIMARY:
            mean, sd = _mean_sd([row[metric] for row in rows])
            base_mean, base_sd = _mean_sd([base_metrics[fold][metric] * 100 for fold in range(1, 11)])
            rows9 = [row for row in rows if row["fold"] != 2]
            mean9, sd9 = _mean_sd([row[metric] for row in rows9])
            base9, base9sd = _mean_sd([base_metrics[fold][metric] * 100 for fold in range(1, 11) if fold != 2])
            d10, d10sd = _mean_sd([row[f"delta_vs_e74_g2_{metric}_pp"] for row in rows])
            d9, d9sd = _mean_sd([row[metric] - row[f"base_{metric}"] for row in rows9])
            summary.update({f"{metric}_mean_percent": mean, f"{metric}_population_sd_percent": sd,
                            f"{metric}_excluding_fold2_mean_percent": mean9,
                            f"{metric}_excluding_fold2_population_sd_percent": sd9,
                            f"e74_g2_{metric}_mean_percent": base_mean,
                            f"e74_g2_{metric}_population_sd_percent": base_sd,
                            f"e74_g2_{metric}_excluding_fold2_mean_percent": base9,
                            f"e74_g2_{metric}_excluding_fold2_population_sd_percent": base9sd,
                            f"delta_vs_e74_g2_{metric}_pp_mean": d10,
                            f"delta_vs_e74_g2_{metric}_pp_population_sd": d10sd,
                            f"delta_vs_e74_g2_excluding_fold2_{metric}_pp_mean": d9,
                            f"delta_vs_e74_g2_excluding_fold2_{metric}_pp_population_sd": d9sd})
            paired.append({"comparison": f"{group}_minus_E74.4_G2", "metric": metric,
                           "folds": 10, "mean_delta_pp": d10, "population_sd_delta_pp": d10sd,
                           "excluding_fold2_folds": 9, "excluding_fold2_mean_delta_pp": d9,
                           "excluding_fold2_population_sd_delta_pp": d9sd})
        summaries.append(summary)

    g1 = {row["fold"]: row for row in fold_rows if row["group"] == GROUPS[0]}
    g2 = {row["fold"]: row for row in fold_rows if row["group"] == GROUPS[1]}
    for metric in PRIMARY:
        diff = [g2[fold][metric] - g1[fold][metric] for fold in range(1, 11)]
        dmean, dsd = _mean_sd(diff)
        diff9 = [g2[fold][metric] - g1[fold][metric] for fold in range(1, 11) if fold != 2]
        d9mean, d9sd = _mean_sd(diff9)
        paired.append({"comparison": "E77-G2_minus_E77-G1", "metric": metric,
                       "folds": 10, "mean_delta_pp": dmean, "population_sd_delta_pp": dsd,
                       "excluding_fold2_folds": 9, "excluding_fold2_mean_delta_pp": d9mean,
                       "excluding_fold2_population_sd_delta_pp": d9sd})
        for fold in range(1, 11):
            g1[fold][f"delta_E77-G2_minus_E77-G1_{metric}_pp"] = g2[fold][metric] - g1[fold][metric]
            g2[fold][f"delta_E77-G2_minus_E77-G1_{metric}_pp"] = g2[fold][metric] - g1[fold][metric]
    return summaries, paired


def _write_report(preflight: dict, fold_rows: list[dict], summaries: list[dict],
                  interventions: list[dict], memory_rows: list[dict], audits: list[dict],
                  paired: list[dict], source_after: dict) -> str:
    summary_by_group = {row["group"]: row for row in summaries}
    base_values = {metric: _mean_sd([row[f"base_{metric}"] for row in fold_rows if row["group"] == GROUPS[0]])
                   for metric in PRIMARY}
    g1 = summary_by_group["E77-G1"]
    g2 = summary_by_group["E77-G2"]
    g2_delta = [g2[f"delta_vs_e74_g2_{metric}_pp_mean"] for metric in PRIMARY]
    hard_change = [row for row in interventions if row["intervention"] == "hard_off"]
    semantic_change = [row for row in interventions if row["intervention"] == "semantic_swap"]
    prototype_change = [row for row in interventions if row["intervention"] == "prototype_off"]
    hard_mean_flip = float(np.mean([row["prediction_change_rate"] for row in hard_change])) if hard_change else 0.0
    semantic_mean_flip = float(np.mean([row["prediction_change_rate"] for row in semantic_change])) if semantic_change else 0.0
    prototype_mean_flip = float(np.mean([row["prediction_change_rate"] for row in prototype_change])) if prototype_change else 0.0
    hard_active = sum(int(item["hard"][key]["active_prototypes"])
                      for item in memory_rows if item["group"] == "E77-G2" for key in item["hard"])
    hard_retrievals = sum(sum(int(v) for v in item["hard"][key]["retrieval_counts"])
                          for item in memory_rows if item["group"] == "E77-G2" for key in item["hard"])
    g1_class_banks = [item["class"] for item in memory_rows if item["group"] == "E77-G1"]
    class_slot_count = sum(len(bank["occupancy"]) for item in g1_class_banks for bank in item.values())
    class_active = sum(int(bank["active_prototypes"]) for item in g1_class_banks for bank in item.values())
    class_used_slots = sum(int(count > 0) for item in g1_class_banks for bank in item.values()
                           for count in bank["retrieval_counts"])
    class_query_coverage = float(np.mean([bank["coverage_rate"]
                                          for item in g1_class_banks for bank in item.values()]))
    class_collapse_max = max(float(bank["collapse_rate_cos_gt_0_98"])
                             for item in g1_class_banks for bank in item.values())
    g2_memory_rows = [item for item in memory_rows if item["group"] == "E77-G2"]
    fp_banks = [item["hard"][f"FP_TD_to_ASD:{modality}"]
                for item in g2_memory_rows for modality in range(4)]
    fn_banks = [item["hard"][f"FN_ASD_to_TD:{modality}"]
                for item in g2_memory_rows for modality in range(4)]
    fp_active = sum(int(bank["active_prototypes"]) for bank in fp_banks)
    fn_active = sum(int(bank["active_prototypes"]) for bank in fn_banks)
    fp_unique_subjects = sum(int(item["hard"]["FP_TD_to_ASD:0"]["true_error_subjects"])
                             for item in g2_memory_rows)
    fn_unique_subjects = sum(int(item["hard"]["FN_ASD_to_TD:0"]["true_error_subjects"])
                             for item in g2_memory_rows)
    fp_bank_coverage = sum(int(bank["query_count"] > 0) for bank in fp_banks)
    fn_bank_coverage = sum(int(bank["query_count"] > 0) for bank in fn_banks)
    fp_observations = sum(int(item["error_observations"]["stable_fp_epoch_observations"])
                          for item in g2_memory_rows)
    fn_observations = sum(int(item["error_observations"]["stable_fn_epoch_observations"])
                          for item in g2_memory_rows)
    near_fp_observations = sum(int(item["error_observations"]["near_fp_epoch_observations"])
                               for item in g2_memory_rows)
    near_fn_observations = sum(int(item["error_observations"]["near_fn_epoch_observations"])
                               for item in g2_memory_rows)
    g2_g1_paired = {row["metric"]: row for row in paired
                    if row["comparison"] == "E77-G2_minus_E77-G1"}
    g1_go = all(g1[f"delta_vs_e74_g2_{metric}_pp_mean"] > 0 for metric in ("acc", "hard_auc"))
    g2_go = all(g2[f"delta_vs_e74_g2_{metric}_pp_mean"] > 0 for metric in ("acc", "hard_auc"))
    mechanism_ok = semantic_mean_flip > 0 and prototype_mean_flip > 0
    status = "GO" if g2_go and mechanism_ok and hard_active > 0 and hard_retrievals > 0 else "NO-GO"
    if not g1_go and not g2_go:
        status = "NO-GO"

    def fmt(metric, summary):
        return f"{summary[f'{metric}_mean_percent']:.4f} +/- {summary[f'{metric}_population_sd_percent']:.4f}"

    def fmt9(metric, group):
        values = [row[metric] for row in fold_rows if row["group"] == group and row["fold"] != 2]
        mean, sd = _mean_sd(values)
        return f"{mean:.4f} +/- {sd:.4f}"

    base_fmt = {metric: f"{base_values[metric][0]:.4f} +/- {base_values[metric][1]:.4f}" for metric in PRIMARY}
    base9_values = {
        metric: _mean_sd([row[f"base_{metric}"] for row in fold_rows
                         if row["group"] == GROUPS[0] and row["fold"] != 2])
        for metric in PRIMARY
    }
    lines = [
        "# Experiment 77: EC-PSM",
        "",
        f"**Decision: {status}.** All values are percentages, mean +/- population SD across ten folds unless stated.",
        "",
        "## Audit and implementation",
        "",
        f"- Base: E74.4-G2 at `{BASE}`; runner SHA `{runner.CFG['base']['runner_sha256']}`, model SHA `{runner.CFG['base']['model_sha256']}`, config SHA `{runner.CFG['base']['config_sha256']}`, manifest SHA `{runner.CFG['base']['manifest_sha256']}`.",
        f"- Inheritance gate: preflight `{preflight['status']}`; 10/10 fold identities and E61 epoch-10 RNG snapshots matched; disabled-module one-step parity `{preflight['base_disabled_one_step_parity']['public_parameters_optimizer_equal']}`; G1/G2 first ten epochs matched in 10/10 folds; checkpoint continuation replay `{preflight['g1_g2_warmup_parity'][0]['resume_audit']['status']}`.",
        f"- E77 source SHA: `{preflight['source_sha256']}`. Protected dependency hashes were unchanged through final audit: `{runner.digest_value(source_after) == runner.digest_value(preflight['protected_sources_unchanged_baseline'])}`.",
        f"- Actual data path: DARG ordered `[PHENO, ANAT, FUNC, CORR]`, features `[871,320]` from modality dimensions `[48,6,10,256]`; inherited CAM emits four 36-D blocks at `[871,144]`; the pre-hook replaces the 144-D input to `cam.Outputlayer`, whose output is 36-D before the unchanged two-class classifier.",
        "- E74.4-G2 uses `dugsa.PLNNoPEN`: there is no active PEN or original semantic prompt computation in this audited base. EC-PSM replaces that no-PEN classifier-input boundary; therefore this comparison tests E77 against the audited DARG-CAM G2 base and does not isolate gains over an active E74.4 PEN.",
        "- E74.4 uses Adam (base and CAM: lr 0.0038, weight decay 0.11) and AdamW (added group: lr 0.001, weight decay 0.01), fixed learning rates with no scheduler, 450 maximum epochs, stopping when wait > 200, full-graph one-step-per-epoch training, and strict test-fold ACC maxima for checkpoint selection. E77 appends its parameters to the existing AdamW added group with the inherited lr/weight decay. Loss remains `L_cls + L_graph + 0.05 L_pair + 0.01 L_adv + 0.01 L_preserve`.",
        "- Fixed E77 settings: four class prototypes per class and modality, at most two FP and two FN prototypes per modality, normalized cosine retrieval with top-2 and temperature 0.2, per-subject EMA 0.9, deterministic farthest-first/Lloyd clustering (25 iterations), stable FP/FN after at least two of the last three eval-mode train predictions, common 10-epoch class-only warmup, bounded residual amplitude 0.05. No additional loss or text encoder run was added.",
        "- Frozen text source: audited E60 CLIP RN50 cache, 8 exact prompts, SHA `" + runner.CFG["method"]["semantic"]["cache_sha256"] + "`; first four prompts define the TD anchor, last four ASD. Four 9-D latent blocks are treated only as learned latent blocks, never anatomical ROIs.",
        "- Class memory updates use detached train-fold CAM features and deterministic epoch-boundary clustering; training queries subtract their own subject feature and count from their assigned cluster before retrieval. Test indices are excluded from stored class and hard memory. DARG is transductive over the full node set, so train representations may still indirectly depend on test-node features.",
        "- The archived base reports Fold-2 train/test overlap at one-based IDs 377 and 407. E77 retained it, excluded those IDs from memory, and did not change the base train split. `val=test`, test-fold ACC selection, pre-CV RFE, and full-graph transductive propagation remain inherited limitations.",
        "- Checkpoint readback: " + str(sum(row["status"] == "PASS" for row in audits)) + "/20 independent audits reproduced all metrics, hard predictions, and probabilities. Fold metrics are recomputed from each saved checkpoint.",
        "",
        "## Results",
        "",
        "| Group | ACC | Hard-AUC / BA | Prob-AUC | SEN (ASD) | SPE (TD) | Delta ACC / Hard-AUC / Prob-AUC / SEN / SPE vs base (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| E74.4-G2 reference | {base_fmt['acc']} | {base_fmt['hard_auc']} | {base_fmt['prob_auc']} | {base_fmt['sen']} | {base_fmt['spe']} | -- |",
        f"| E77-G1 | {fmt('acc', g1)} | {fmt('hard_auc', g1)} | {fmt('prob_auc', g1)} | {fmt('sen', g1)} | {fmt('spe', g1)} | " + " / ".join(f"{g1[f'delta_vs_e74_g2_{metric}_pp_mean']:+.4f}" for metric in PRIMARY) + " |",
        f"| E77-G2 | {fmt('acc', g2)} | {fmt('hard_auc', g2)} | {fmt('prob_auc', g2)} | {fmt('sen', g2)} | {fmt('spe', g2)} | " + " / ".join(f"{g2[f'delta_vs_e74_g2_{metric}_pp_mean']:+.4f}" for metric in PRIMARY) + " |",
        "",
        "Paired E77-G2 minus E77-G1 differences across the ten folds (mean +/- population SD, percentage points):",
        "",
        "| Metric | Mean delta +/- SD (pp) |",
        "|---|---:|",
        *[f"| {metric.upper()} | {g2_g1_paired[metric]['mean_delta_pp']:+.4f} +/- {g2_g1_paired[metric]['population_sd_delta_pp']:.4f} |"
          for metric in PRIMARY],
        "",
        "G1 and G2 have ten paired folds in `fold_metrics.csv`; `paired_comparisons.csv` gives foldwise mean +/- population SD differences versus E74.4-G2 and G2 minus G1. It also reports descriptive nine-fold comparisons excluding Fold 2. These remain under the historical test-selected checkpoint protocol and are not unbiased generalization estimates.",
        "",
        "Nine-fold descriptive means +/- population SD after omitting Fold 2 (no retraining):",
        "",
        "| Group | ACC | Hard-AUC / BA | Prob-AUC | SEN (ASD) | SPE (TD) |",
        "|---|---:|---:|---:|---:|---:|",
        "| E74.4-G2 reference | " + " | ".join(f"{base9_values[metric][0]:.4f} +/- {base9_values[metric][1]:.4f}" for metric in PRIMARY) + " |",
        "| E77-G1 | " + " | ".join(fmt9(metric, "E77-G1") for metric in PRIMARY) + " |",
        "| E77-G2 | " + " | ".join(fmt9(metric, "E77-G2") for metric in PRIMARY) + " |",
        "",
        "## Mechanism diagnostics",
        "",
        f"- G1 class banks have {class_active}/{class_slot_count} active prototype slots; {class_used_slots}/{class_slot_count} slots were retrieved, mean bank query coverage is {class_query_coverage:.4f}, and the maximum within-bank collapse fraction (`cos > 0.98`) is {class_collapse_max:.4f}. Occupancy and per-modality usage are in `memory_audit.json`.",
        f"- G2 selected memory represents {fp_unique_subjects} fold-summed unique training FP patients (TD predicted ASD) and {fn_unique_subjects} FN patients (ASD predicted TD); counts are training subjects only, never test labels. FP has {fp_active}/80 and FN {fn_active}/80 active prototypes, with nonzero retrieval in {fp_bank_coverage}/40 FP and {fn_bank_coverage}/40 FN modality-fold banks. Their cumulative selected-checkpoint usage counters total {hard_retrievals} prototype assignment events across training/evaluation forwards.",
        f"- Stable-error epoch observations are FP {fp_observations} and FN {fn_observations}; separately marked near-boundary observations are near-FP {near_fp_observations} and near-FN {near_fn_observations}. These are repeated epoch observations, not unique patients; confidence, age, occupancy, and direction-specific counts are in `memory_audit.json`.",
        f"- Mean test prediction change rates under fixed-checkpoint interventions: hard-memory off {hard_mean_flip:.4f}, semantic ASD/TD swap {semantic_mean_flip:.4f}, and prototype retrieval off {prototype_mean_flip:.4f}. These are inference diagnostics only, not retrained causal ablations.",
        "- FP contributes negative ASD-direction evidence and FN contributes positive ASD-direction evidence. If either bank has no selected prototypes or retrievals, the diagnostic is reported as inactive and no gain is attributed to that branch.",
        "",
        "## Interpretation and limits",
        "",
        "The result is a single-seed comparison under the inherited historical protocol. It does not establish cross-seed stability or independent clinical generalization. Validation equals test for checkpoint selection; RFE preceded outer CV; Fold 2 retains the documented overlap; and transductive full-graph propagation can indirectly expose test-node features to training representations. Excluding Fold 2 is descriptive only and does not redefine the original CV.",
        "",
        "## Core computation",
        "",
        "For patient i and modality m, the inherited classifier context is h[i,m] in R^36 and its four learned sub-blocks are z[i,m,b] in R^9 (b=1..4). DARG's 320-D output and CAM's 144-D classifier input are untouched outside the hook boundary.",
        "",
        "```text",
        "r[i,c,m] = Top2CosineRetrieval(h[i,m], leave-one-subject-out prototypes P[c,m])",
        "a[c] = normalize(W_text * frozen_text_anchor[c]) in R^9",
        "u[i,c,m] = normalize(a[c] + 0.1*tanh(W_prompt * [r[i,c,m], r[i,c,m]-r[i,1-c,m], a[c]])) in R^9",
        "alpha[i,c,m,b] = softmax_b(cos(normalize(z[i,m,b]), u[i,c,m]) / 0.2)",
        "s[i,c,m] = sum_b alpha[i,c,m,b] * z[i,m,b] in R^9",
        "e[i,m] = evidence[i,ASD,m] - evidence[i,TD,m] + 0.1*cos(h[i,m], FN[m]) - 0.1*cos(h[i,m], FP[m])",
        "Delta[i,m] = 0.05*tanh(W_e * [h[i,m], r[i,ASD,m]-r[i,TD,m], s[i,ASD,m], s[i,TD,m], e[i,m]]) in R^36",
        "h_E77[i] = h[i] + concat_m(Delta[i,m]) in R^144",
        "```",
        "",
        "Here evidence[i,c,m] is the prompt-conditioned average of four block-to-prompt cosine scores. Empty class or hard banks produce zero retrieval evidence; test IDs cannot become bank members. ASD is positive (label 0), so FN evidence raises and FP evidence lowers ASD-direction evidence. The prompt uses the frozen class text anchor, a retrieved prototype, and an explicit ASD/TD prototype difference; it is a learned feature selector, not generated natural language.",
        "",
        "SPromptGL supplies the four learned 9-D blocks per modality; they are not anatomical ROI labels. PGP-SAM uses class/instance prototype guidance for segmentation, Concentrate on Weakness mines weak segmentation regions through self-prediction, and HP²L uses hierarchical disease prompts/prototypes in a large multi-disease setting. E77 borrows only the general prototype-conditioning and error-history ideas; those works do not validate this ASD/TD transfer or E77's clinical interpretation.",
        "",
        f"No follow-on search or extra training was launched. Final decision: **{status}**."
    ]
    report = "\n".join(lines) + "\n"
    path = RESULTS / "report.md"
    temp = path.with_suffix(".md.tmp")
    temp.write_text(report, encoding="utf-8")
    os.replace(temp, path)
    return status


def finalize() -> dict:
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    _check(preflight.get("status") == "PASS", "cannot finalize without passing preflight")
    base_metrics = _read_base_metrics()
    fold_rows, audit_rows, memory_rows, intervention_rows = [], [], [], []
    for group in GROUPS:
        for fold in range(1, 11):
            metrics, audit, details = _independent_checkpoint_audit(group, fold, base_metrics)
            fold_rows.append(metrics); audit_rows.append(audit)
            memory_rows.append(details["memory"])
            if group == "E77-G2":
                for intervention, values in details["interventions"].items():
                    intervention_rows.append({"group": group, "fold": fold,
                                              "intervention": intervention, **values})

    summaries, paired = _summary_rows(fold_rows, base_metrics)
    _check(len(audit_rows) == 20 and all(row["status"] == "PASS" for row in audit_rows),
           "not all twenty checkpoints passed independent readback")
    _check(len(intervention_rows) == 30, "expected 3 fixed-checkpoint interventions for each G2 fold")
    source_after = runner.source_manifest()
    _check(runner.digest_value(source_after) == runner.digest_value(preflight["protected_sources_unchanged_baseline"]),
           "a read-only base, raw data, or historical dependency changed during E77")

    runner.E74.write_csv(RESULTS / "fold_metrics.csv", fold_rows)
    runner.E74.write_csv(RESULTS / "summary.csv", summaries)
    runner.E74.write_csv(RESULTS / "paired_comparisons.csv", paired)
    runner.E74.write_csv(RESULTS / "checkpoint_audit.csv", audit_rows)
    runner.E74.write_csv(RESULTS / "interventions.csv", intervention_rows)
    runner.atomic_json(RESULTS / "memory_audit.json", {
        "preflight_leave_one_out_test": preflight["leave_one_subject_out_cluster_test"],
        "folds": memory_rows,
    })
    status = _write_report(preflight, fold_rows, summaries, intervention_rows,
                           memory_rows, audit_rows, paired, source_after)
    manifest = {
        "experiment": "E77-ECPSM", "status": status,
        "trained_group_folds": 20, "independent_checkpoint_audits": len(audit_rows),
        "source_sha256": preflight["source_sha256"],
        "protected_sources_unchanged": True,
        "preflight": "PASS", "g1_g2_warmup_parity_folds": 10,
        "results": ["results/report.md", "results/fold_metrics.csv", "results/summary.csv",
                    "results/paired_comparisons.csv", "results/checkpoint_audit.csv",
                    "results/interventions.csv", "results/memory_audit.json"],
        "checkpoint_count": 20,
        "checkpoint_hashes": {f"{g}_f{f:02d}": runner.sha_file(CHECKPOINTS / f"{g}_f{f:02d}.best.pt")
                              for g in GROUPS for f in range(1, 11)},
        "source_hashes": preflight["source_hashes"],
        "protected_source_hashes_after": source_after,
    }
    runner.atomic_json(HERE / "manifest.json", manifest)
    print(json.dumps({"status": status, "metrics": summaries,
                      "checkpoints_audited": len(audit_rows),
                      "report": str(RESULTS / "report.md")}, ensure_ascii=True), flush=True)
    return manifest


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"preflight", "finalize"}:
        raise SystemExit("usage: audit.py preflight|finalize")
    try:
        if sys.argv[1] == "preflight":
            preflight()
        else:
            finalize()
    except Exception as error:
        failure = {
            "status": "FAIL", "phase": sys.argv[1],
            "error": f"{type(error).__name__}: {error}",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        runner.atomic_json(RESULTS / f"{sys.argv[1]}_failure.json", failure)
        print(json.dumps(failure, ensure_ascii=True), file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
