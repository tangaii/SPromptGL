#!/usr/bin/env python3
"""SC-PERM parity, mechanism, checkpoint, and reporting audits."""
from __future__ import annotations

import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

import run as runner
from memory import SC_PERM, TemporalErrorTracker
from model import build_sc_perm, forward_sc_perm, prepare_text_cache


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CHECKPOINTS = HERE / "checkpoints"
GROUPS = runner.GROUPS
PRIMARY = ("acc", "hard_auc", "prob_auc", "sen", "spe")
ALL_METRICS = (*PRIMARY, "ba", "asd_f1")
BASE_GROUP = runner.BASE_GROUP
BASE = runner.BASE


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_json(path: Path, payload: dict) -> None:
    runner.atomic_json(path, payload)


def _pct_metrics(metrics: dict) -> dict:
    return {key: float(metrics[key]) * 100.0 for key in ALL_METRICS if key in metrics}


def _base_rows() -> dict[int, dict]:
    path = BASE / "results" / "fold_metrics.csv"
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = {int(row["fold"]): row for row in rows if row["group"] == BASE_GROUP}
    _check(len(selected) == 10, "E74.4-G2 reference does not contain ten audited folds")
    return selected


def _e77_rows() -> dict[int, dict]:
    path = Path("/model/spromptgl/experiment/experiment77_ecpsm/results/fold_metrics.csv")
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = {int(row["fold"]): row for row in rows if row["group"] == "E77-G2"}
    _check(len(selected) == 10, "read-only E77-G2 reference does not contain ten audited folds")
    return selected


def _read_base_reference() -> dict:
    rows = _base_rows()
    result = {}
    for fold, row in rows.items():
        result[fold] = {metric: float(row[metric]) for metric in ALL_METRICS}
    return result


def _text_audit(cache: dict) -> dict:
    banks = cache["embeddings"]
    def anchors(value: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if name == "medical":
            shared = value[[3, 7]].mean(0)
            td = value[:3].mean(0) + shared
            asd = value[4:7].mean(0) + shared
        else:
            td = value[:4].mean(0)
            asd = value[4:].mean(0)
        return td, asd

    result = {"texts": cache["texts"], "cache_sha256": runner.sha_file(runner.HERE / "runtime" / "text_embeddings.pt"),
              "source_hashes": cache["source_hashes"], "text_encoder": cache["text_encoder"],
              "mapping": runner.CFG["prompts"]["mapping"], "class_order": runner.CFG["prompts"]["internal_class_order"]}
    for name, value in banks.items():
        normalized = F.normalize(value.float(), dim=-1)
        matrix = normalized @ normalized.T
        td, asd = anchors(value.float(), name)
        result[name] = {
            "prompt_token_lengths": cache["token_lengths"][name],
            "within_prompt_cosine_matrix": matrix.detach().cpu().tolist(),
            "td_asd_anchor_cosine": float(F.cosine_similarity(td, asd, dim=0).item()),
            "td_anchor_norm": float(td.norm().item()), "asd_anchor_norm": float(asd.norm().item()),
        }
    original = F.normalize(banks["original"].float(), dim=-1)
    medical = F.normalize(banks["medical"].float(), dim=-1)
    original_td, original_asd = anchors(original, "original")
    medical_td, medical_asd = anchors(medical, "medical")
    original_anchors = torch.stack((original_asd, original_td))
    medical_anchors = torch.stack((medical_asd, medical_td))
    result["class_anchor_original_medical_cosine"] = F.cosine_similarity(
        original_anchors, medical_anchors, dim=-1
    ).detach().cpu().tolist()
    result["original_prompts"] = runner.CFG["prompts"]["original"]
    result["medical_prompts"] = runner.CFG["prompts"]["medical"]
    result["sources"] = runner.CFG["literature"]
    result["textual_balance"] = {
        "same_total_count": len(runner.CFG["prompts"]["original"]) == len(runner.CFG["prompts"]["medical"]) == 8,
        "same_class_count": all(len(value) == 4 for value in runner.CFG["prompts"]["mapping"].values()),
        "no_roi_latent_block_mapping": True,
        "clinical_expert_review": False,
        "medical_neutral_prompts_shared_between_class_anchors": True,
        "semantic_anchor_rule": runner.CFG["prompts"]["semantic_anchor_rule"],
        "prompt_claim_audit": runner.CFG["prompts"]["prompt_claim_audit"],
    }
    return result


def _one_step_base_parity(fold: int, cache: dict) -> dict:
    base_state, base_warmup, base_identity = runner.E74.prepare(fold)
    base_state["fold"] = fold
    base_state["group"] = BASE_GROUP
    base_runtime = runner.E74.load_history().load_e69().load_e61().load_base()
    before_sha = base_runtime.rng_sha(base_runtime.pack_rng())
    _check(before_sha == base_warmup["rng_sha"], f"fold {fold} base state did not restore the E61 RNG")
    base_added_groups = len(base_state["opt_added"].param_groups)
    runner.E74.load_history().zero_optimizers(base_state)
    base_output = runner.E74.forward(base_state, BASE_GROUP, training=True, collect=False)
    _check(tuple(base_output["logp"].shape) == (871, 2), "base classifier output is not [871,2]")
    _check(tuple(base_output["hidden"].shape) == (871, 36), "base CAM hidden output is not [871,36]")
    base_output["total"].backward()
    for key in ("opt_base", "opt_cam", "opt_added"):
        base_state[key].step()
    base_after_rng = base_runtime.rng_sha(base_runtime.pack_rng())
    base_public = runner.public_state_sha(base_state, base_added_groups)
    base_logp = base_output["logp"].detach().clone()
    base_total = base_output["total"].detach().clone()
    del base_output, base_state

    state, warmup, identity = runner.E74.prepare(fold)
    state["fold"] = fold
    state["group"] = BASE_GROUP
    runtime = runner.E74.load_history().load_e69().load_e61().load_base()
    _check(identity == base_identity and warmup["rng_sha"] == base_warmup["rng_sha"],
           f"fold {fold} warmup identity changed between parity copies")
    before_module = runtime.pack_rng()
    ecpsm = build_sc_perm(fold, "E77.1-G2", state["x"].device, cache)
    if runtime.rng_sha(runtime.pack_rng()) != runtime.rng_sha(before_module):
        raise RuntimeError(f"fold {fold} SC-PERM initialization consumed public RNG")
    old_added_groups = len(state["opt_added"].param_groups)
    state["opt_added"].add_param_group({
        "params": list(ecpsm.parameters()),
        "lr": runner.CFG["base"]["optimizer"]["added"]["lr"],
        "weight_decay": runner.CFG["base"]["optimizer"]["added"]["weight_decay"],
    })
    _check(runtime.rng_sha(runtime.pack_rng()) == warmup["rng_sha"],
           f"fold {fold} module setup changed inherited public RNG")
    runner.E74.load_history().zero_optimizers(state)
    candidate_output, hook = forward_sc_perm(
        runner.E74, state, ecpsm, "E77.1-G2", training=True, local_epoch=0,
        enabled=False,
    )
    candidate_output["total"].backward()
    _check(runner._finite_gradients((state["model"], state["cam"], ecpsm)),
           f"fold {fold} disabled-hook parity produced non-finite gradients")
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    candidate_after_rng = runtime.rng_sha(runtime.pack_rng())
    candidate_public = runner.public_state_sha(state, old_added_groups)
    result = {
        "fold": fold, "base_identity": identity,
        "initial_rng_sha256": warmup["rng_sha"],
        "base_output_logp_sha256": runner.digest_value(base_logp),
        "disabled_output_logp_sha256": runner.digest_value(candidate_output["logp"].detach()),
        "base_total_sha256": runner.digest_value(base_total),
        "disabled_total_sha256": runner.digest_value(candidate_output["total"].detach()),
        "same_initial_identity": identity == base_identity,
        "same_rng_before": before_sha == warmup["rng_sha"],
        "logp_equal": torch.equal(base_logp, candidate_output["logp"].detach()),
        "loss_equal": torch.equal(base_total, candidate_output["total"].detach()),
        "public_model_optimizer_equal": base_public == candidate_public,
        "rng_after_train_equal": base_after_rng == candidate_after_rng,
        "module_parameters_without_gradient": all(parameter.grad is None for parameter in ecpsm.parameters()),
        "pre_hook_calls": hook["pre_calls"], "post_hook_calls": hook["post_calls"],
        "hook_calls_once": hook["pre_calls"] == 1 and hook["post_calls"] == 1,
    }
    _check(all(result[key] for key in (
        "same_initial_identity", "same_rng_before", "logp_equal", "loss_equal",
        "public_model_optimizer_equal", "rng_after_train_equal", "module_parameters_without_gradient",
        "hook_calls_once",
    )), f"fold {fold} disabled SC-PERM/base parity failed: {result}")
    del state, ecpsm, candidate_output
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _warmup_group(fold: int, group: str, cache: dict, *, resume_check: bool = False) -> dict:
    state, warmup, identity, base_task, module, tracker, old_groups, runtime = runner.initialize_group(fold, group)
    initial_module_parameters_sha = runner.module_sha(module, parameters_only=True)
    allowed = runner.allowed_train_ids(state)
    allowed_t = torch.as_tensor(allowed, dtype=torch.long, device=state["x"].device)
    before_init = runtime.pack_rng()
    init = runner._initial_memory(state, module, group)
    _check(runtime.rng_sha(runtime.pack_rng()) == runtime.rng_sha(before_init),
           f"fold {fold}/{group} class memory initialization consumed public RNG")
    reference = {int(row["epoch"]): row for row in base_task["rng_trace"]}
    signatures = []
    resume_record = None
    for local_epoch in range(10):
        step = runner.training_epoch(state, module, tracker, group, local_epoch, allowed, allowed_t, runtime)
        rng_trace = step["rng"]
        expected = reference.get(rng_trace["epoch"])
        _check(expected is not None, f"fold {fold} lacks E74 RNG reference at epoch {rng_trace['epoch']}")
        for key in ("rng_before", "rng_after_train", "rng_after_eval"):
            _check(rng_trace[key] == expected[key],
                   f"fold {fold}/{group} did not follow E74 public RNG at {rng_trace['epoch']} {key}")
        signature = runner._signature(state, module, old_groups, tracker, step["evaluated"], rng_trace)
        signatures.append(signature)
        if resume_check and local_epoch == 0:
            save_path = HERE / "runtime" / "preflight_resume.pt"
            saved = {
                "base_state": runner.E74.load_history().state_bundle(state),
                "module_state": runner.cpu_state(module), "tracker_state": runner.cpu_state(tracker),
                "rng": runtime.pack_rng(), "next_local_epoch": 1,
                "fold": fold, "group": group, "warmup_sha256": identity["warmup_sha256"],
            }
            runner.atomic_torch(save_path, saved)
            original = runner.training_epoch(state, module, tracker, group, 1, allowed, allowed_t, runtime)
            original_signature = runner._signature(state, module, old_groups, tracker,
                                                   original["evaluated"], original["rng"])
            restored = torch.load(save_path, map_location=state["x"].device, weights_only=False)
            runner.E74.load_history().load_bundle(state, restored["base_state"])
            module.load_state_dict(restored["module_state"], strict=True)
            tracker.load_state_dict(restored["tracker_state"], strict=True)
            runtime.restore_rng(restored["rng"])
            resumed = runner.training_epoch(state, module, tracker, group, 1, allowed, allowed_t, runtime)
            resumed_signature = runner._signature(state, module, old_groups, tracker,
                                                  resumed["evaluated"], resumed["rng"])
            _check(original_signature == resumed_signature,
                   "SC-PERM checkpoint did not restore model/optimizer/memory/tracker/RNG continuation")
            resume_record = {
                "status": "PASS", "checkpoint_sha256": runner.sha_file(save_path),
                "next_step_signature_sha256": runner.digest_value(resumed_signature),
                "restored_local_epoch": 1,
            }
            runner.E74.load_history().load_bundle(state, restored["base_state"])
            module.load_state_dict(restored["module_state"], strict=True)
            tracker.load_state_dict(restored["tracker_state"], strict=True)
            runtime.restore_rng(restored["rng"])
            del original, resumed, restored, original_signature, resumed_signature
        del step

    result = {"fold": fold, "group": group, "warmup_sha256": identity["warmup_sha256"],
              "initial_module_parameters_sha256": initial_module_parameters_sha,
              "warmup_rng_sha": warmup["rng_sha"], "initial_memory": init,
              "signatures": signatures, "resume_audit": resume_record,
              "allowed_train_ids": int(len(allowed)),
              "test_ids_excluded": int(len(state["train_np"]) - len(allowed))}
    del state, module, tracker
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _synthetic_memory_tests(cache: dict) -> dict:
    module = build_sc_perm(1, "E77.1-G2", torch.device("cpu"), cache)
    tracker = TemporalErrorTracker(window=3)
    labels = torch.full((871,), -999, dtype=torch.long)
    labels[10], labels[11], labels[14], labels[12] = 1, 0, 1, 0
    allowed = torch.tensor([10, 11, 14], dtype=torch.long)
    correct = torch.zeros(871, 2)
    wrong = torch.zeros(871, 2)
    correct[10], correct[14], correct[11] = torch.tensor([-4.0, 0.0]), torch.tensor([-4.0, 0.0]), torch.tensor([0.0, -4.0])
    wrong[10], wrong[14], wrong[11] = torch.tensor([0.0, -4.0]), torch.tensor([0.0, -4.0]), torch.tensor([-4.0, 0.0])
    tracker.update(correct, labels, allowed, min_votes=2)
    tracker.update(wrong, labels, allowed, min_votes=2)
    tracker.update(wrong, labels, allowed, min_votes=2)
    state = tracker.update(correct, labels, allowed, min_votes=2)
    _check(state["fp_ids"].tolist() == [10, 14] and state["fn_ids"].tolist() == [11],
           "3-step temporal rule failed to select the correct FP/FN directions")
    features = torch.zeros(871, 4, 36)
    for subject, coordinate in ((10, 0), (14, 1), (11, 2), (12, 0), (13, 2)):
        features[subject, :, coordinate] = 1.0
    module.semantic_beta_raw.data.fill_(-20.0)
    module.update_class_memory(features, labels, allowed)
    prediction_confidence = torch.full((871,), 0.95)
    update = module.update_hard_memory(features, labels, state["vote_counts"], prediction_confidence, allowed, 1)
    _check(set(module.hard_subject_ids[0][module.hard_subject_ids[0] >= 0].tolist()) == {10, 14},
           "FP patients were not kept in the FP bank")
    _check(set(module.hard_subject_ids[1][module.hard_subject_ids[1] >= 0].tolist()) == {11},
           "FN patients were not kept in the FN bank")
    _check(12 not in module.hard_subject_ids[module.hard_subject_ids >= 0].tolist(),
           "disallowed test ID entered hard memory")
    fp_query = features[12:13].reshape(1, 144)
    fn_query = features[13:14].reshape(1, 144)
    fp_evidence = module.compute_evidence(fp_query, torch.tensor([12]), hard_enabled=True)
    fp_hard_off = module.compute_evidence(fp_query, torch.tensor([12]), hard_enabled=True, hard_off=True)
    fn_evidence = module.compute_evidence(fn_query, torch.tensor([13]), hard_enabled=True)
    _check(float(fp_evidence["hard_scores"][0, 1].detach()) > float(fp_evidence["hard_scores"][0, 0].detach())
           and not torch.equal(fp_evidence["correction"], fp_hard_off["correction"]),
           "FP-like case did not add TD-directed diagnosis evidence")
    _check(float(fn_evidence["hard_scores"][0, 0].detach()) > float(fn_evidence["hard_scores"][0, 1].detach()),
           "FN-like case did not add ASD-directed diagnosis evidence")
    second = module.update_hard_memory(features, labels, state["vote_counts"], prediction_confidence, allowed, 2)
    _check(int((module.hard_subject_ids[0] == 10).sum()) == 1 and int(module.hard_frequency[0].max()) == 2,
           "repeated errors duplicated a patient or did not update patient history")
    module.update_hard_memory(features, labels, torch.zeros_like(state["vote_counts"]), prediction_confidence, allowed, 3)
    _check(bool((module.hard_subject_ids[0] >= 0).any()), "corrected error cases were evicted immediately")
    module.update_hard_memory(features, labels, torch.zeros_like(state["vote_counts"]), prediction_confidence, allowed, 23)
    _check(not bool((module.hard_subject_ids >= 0).any()), "stale errors survived the declared maximum age")
    module.class_valid.zero_(); module.class_counts.zero_(); module.class_sums.zero_(); module.class_centers.zero_()
    module.hard_subject_ids.fill_(-1); module.hard_counts.zero_(); module.hard_sums.zero_(); module.hard_centers.zero_()
    empty_scores = module.compute_evidence(torch.ones(3, 144), torch.tensor([1, 2, 3]), hard_enabled=True)
    _check(bool(torch.isfinite(empty_scores["correction"]).all()) and not bool(module.hard_counts.any()),
           "empty memory branch did not fall back safely")
    del module, tracker
    return {"status": "PASS", "stable_vote_rule": "2 of 3", "fp_ids": [10, 14], "fn_ids": [11],
            "test_id_excluded": True, "duplicate_subject_prevention": True,
            "history_retained_then_expired": True, "empty_bank_fallback": True,
            "hard_evidence_nonzero": True,
            "fp_query_td_directed_hard_evidence": float(fp_evidence["hard_scores"][0, 1].item()),
            "fn_query_asd_directed_hard_evidence": float(fn_evidence["hard_scores"][0, 0].item()),
            "hard_off_correction_delta": float((fp_evidence["correction"] - fp_hard_off["correction"]).abs().mean().item()),
            "initial_pool_sizes": {"fp": int(update["pool_fp"]), "fn": int(update["pool_fn"])},
            "duplicate_update_pool_sizes": {"fp": int(second["pool_fp"]), "fn": int(second["pool_fn"])}}


def _semantic_and_gradient_test(cache: dict) -> dict:
    fold = 1
    state, warmup, identity, base_task, module, tracker, old_groups, runtime = runner.initialize_group(fold, "E77.1-G2")
    allowed = runner.allowed_train_ids(state)
    allowed_t = torch.as_tensor(allowed, dtype=torch.long, device=state["x"].device)
    initial_rng = runtime.rng_sha(runtime.pack_rng())
    runner._initial_memory(state, module, "E77.1-G2")
    _check(runtime.rng_sha(runtime.pack_rng()) == initial_rng, "semantic preflight memory consumed RNG")
    capture: dict = {}
    with torch.no_grad():
        forward_sc_perm(runner.E74, state, module, "E77.1-G2", training=False,
                        local_epoch=0, prototype_off=True, capture=capture)
    train_features = capture["cam_features"][allowed_t]
    ids = allowed_t
    medical = module.compute_evidence(train_features, ids, hard_enabled=False, text_source="medical")
    original = module.compute_evidence(train_features, ids, hard_enabled=False, text_source="original")
    swapped = module.compute_evidence(train_features, ids, hard_enabled=False,
                                      semantic_swap=True, text_source="medical")
    med_original_score_delta = float((medical["class_scores"] - original["class_scores"]).abs().mean().item())
    swap_score_delta = float((medical["class_scores"] - swapped["class_scores"]).abs().mean().item())
    swap_weight_delta = float((medical["class_weights"] - swapped["class_weights"]).abs().mean().item())
    swap_correction_delta = float((medical["correction"] - swapped["correction"]).abs().mean().item())
    _check(med_original_score_delta > 1e-8, "medical text source did not change real-patient class evidence")
    _check(max(swap_score_delta, swap_weight_delta, swap_correction_delta) > 1e-8,
           "ASD/TD semantic exchange did not change retrieval/evidence")

    step = runner.training_epoch(state, module, tracker, "E77.1-G2", 0, allowed, allowed_t, runtime)
    grad_norms = step["module_grad_norms"]
    _check(grad_norms and all(math.isfinite(value) for value in grad_norms) and max(grad_norms) > 0,
           "SC-PERM parameters received no finite nonzero gradient")
    _check(tuple(step["evaluated"]["logp"].shape) == (871, 2), "diagnostic logits shape changed")
    result = {
        "fold": fold, "train_query_count": len(allowed), "class_score_mean_abs_original_medical_delta": med_original_score_delta,
        "class_score_mean_abs_semantic_swap_delta": swap_score_delta,
        "retrieval_weight_mean_abs_semantic_swap_delta": swap_weight_delta,
        "decision_correction_mean_abs_semantic_swap_delta": swap_correction_delta,
        "module_gradient_norms": grad_norms,
        "output_shape": list(step["evaluated"]["logp"].shape),
        "hidden_shape": list(step["evaluated"]["hidden"].shape),
        "cam_feature_shape": list(capture["cam_features"].shape),
        "hard_enabled_during_warmup": False,
    }
    del state, module, tracker, step
    gc.collect()
    torch.cuda.empty_cache()
    return result


def preflight() -> dict:
    for path in (RESULTS, HERE / "runtime", CHECKPOINTS):
        path.mkdir(parents=True, exist_ok=True)
    cache = prepare_text_cache()
    text_audit = _text_audit(cache)
    protected_before = runner.source_manifest()
    base_manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    base_preflight = json.loads((BASE / "results" / "preflight.json").read_text(encoding="utf-8"))
    base_audit_status = json.loads((BASE / "results" / "audit_status.json").read_text(encoding="utf-8"))
    e74_hash = runner.sha_file(BASE / "run.py")
    _check(base_manifest.get("status") == "COMPLETE_AUDITED"
           and base_manifest.get("completed_tasks") == 30
           and base_manifest.get("independent_audits") == 30,
           "E74.4-G2 is not a complete, independently audited base")
    _check(base_preflight.get("status") == "PASS" and base_audit_status.get("status") == "PASS",
           "E74.4-G2 preflight or checkpoint audit status is not PASS")
    _check(e74_hash == base_manifest["protected_sources"][str((BASE / "run.py").resolve())],
           "local E74.4 runner hash does not match its audited manifest")

    one_step = [_one_step_base_parity(fold, cache) for fold in range(1, 11)]
    hard_test = _synthetic_memory_tests(cache)
    semantic_test = _semantic_and_gradient_test(cache)
    warmup = []
    for fold in range(1, 11):
        g1 = _warmup_group(fold, "E77.1-G1", cache)
        g2 = _warmup_group(fold, "E77.1-G2", cache, resume_check=fold == 1)
        g3 = _warmup_group(fold, "E77.1-G3", cache)
        init_hashes = {
            g1["initial_module_parameters_sha256"],
            g2["initial_module_parameters_sha256"],
            g3["initial_module_parameters_sha256"],
        }
        _check(len(init_hashes) == 1,
               f"fold {fold} group-specific SC-PERM parameter initialization differs")
        parity = runner._same_warmup_signatures(g2["signatures"], g3["signatures"])
        _check(parity["status"] == "PASS", f"G2/G3 common warmup parity failed in fold {fold}: {parity}")
        warmup.append({"fold": fold, "parity": parity,
                       "identical_module_initialization": True,
                       "g1_rng_trace": [item["rng"] for item in g1["signatures"]],
                       "g2_rng_trace": [item["rng"] for item in g2["signatures"][:10]],
                       "g3_rng_trace": [item["rng"] for item in g3["signatures"][:10]],
                       "g2_resume": g2["resume_audit"],
                       "identical_warmup_state": True})
        _write_json(RESULTS / f"fold{fold:02d}_warmup_preflight.json", warmup[-1])
    _check(len(one_step) == 10 and all(row["rng_after_train_equal"] for row in one_step),
           "all-fold base-disabled parity did not pass")
    _check(all(row["parity"]["status"] == "PASS" for row in warmup),
           "G2/G3 warmup did not pass for all folds")
    _check(all(row["g2_resume"]["status"] == "PASS" for row in warmup if row["fold"] == 1),
           "checkpoint continuation audit did not pass")
    protected_after = runner.source_manifest()
    _check(protected_before == protected_after, "a read-only dependency changed during preflight")
    source_sha = runner.source_sha()
    record = {
        "status": "PASS", "experiment": runner.CFG["experiment"], "created_utc_epoch": time.time(),
        "source_sha256": source_sha, "protected_sources_unchanged_baseline": protected_before,
        "e74_runner_sha256": e74_hash, "e74_preflight_sha256": runner.sha_file(BASE / "results" / "preflight.json"),
        "e74_g2_reference_metrics": runner.CFG["reference_metrics"],
        "text_cache_audit": text_audit, "base_disabled_parity": one_step,
        "synthetic_hard_memory_audit": hard_test, "semantic_gradient_audit": semantic_test,
        "g2_g3_common_warmup_parity": warmup,
        "memory_id_policy": "per-fold train indices minus explicit test indices; test labels are never indexed by SC-PERM memory/tracker",
        "rng_policy": "restore E61 epoch-10 full RNG state; no per-epoch reseed; private torch.fork_rng for module init and text encoding",
        "evaluation_limits": runner.CFG["audit"],
    }
    _write_json(RESULTS / "preflight.json", record)
    return record


def _load_checkpoint(group: str, fold: int, preflight_record: dict):
    ckpt_path = runner.best_path(group, fold)
    task_path = runner.task_path(group, fold)
    _check(ckpt_path.is_file() and task_path.is_file(), f"missing checkpoint/task {group} fold {fold}")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    checkpoint_sha = runner.sha_file(ckpt_path)
    _check(task.get("status") == "COMPLETE" and task.get("checkpoint_sha256") == checkpoint_sha,
           f"task record/checkpoint SHA mismatch for {group} fold {fold}")
    _check(task.get("source_sha256") == preflight_record["source_sha256"],
           f"checkpoint source identity mismatch for {group} fold {fold}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _check(checkpoint.get("group") == group and checkpoint.get("fold") == fold,
           f"checkpoint labels mismatch for {group} fold {fold}")
    state, warmup, identity = runner.E74.prepare(fold)
    state["fold"] = fold
    state["group"] = BASE_GROUP
    _check(identity == checkpoint["base_identity"], f"base fold identity mismatch in {group} fold {fold}")
    _check(np.array_equal(np.asarray(checkpoint["test_indices"], dtype=np.int64),
                          np.asarray(state["test_np"], dtype=np.int64)),
           f"checkpoint test index list differs from the audited fold {fold}")
    runtime = runner.E74.load_history().load_e69().load_e61().load_base()
    module = build_sc_perm(fold, group, state["x"].device,
                           prepare_text_cache())
    old_added_groups = len(state["opt_added"].param_groups)
    state["opt_added"].add_param_group({
        "params": list(module.parameters()),
        "lr": runner.CFG["base"]["optimizer"]["added"]["lr"],
        "weight_decay": runner.CFG["base"]["optimizer"]["added"]["weight_decay"],
    })
    tracker = TemporalErrorTracker(runner.CFG["method"]["hard_error_window"]).to(state["x"].device)
    runner.E74.load_history().load_bundle(state, checkpoint["base_state"])
    module.load_state_dict(checkpoint["ecpsm_state"], strict=True)
    tracker.load_state_dict(checkpoint["tracker_state"], strict=True)
    runtime.restore_rng(checkpoint["rng"])
    return checkpoint, state, module, tracker, runtime, old_added_groups, checkpoint_sha, ckpt_path


def _intervention_eval(group: str, fold: int, checkpoint: dict, state: dict,
                       module: SC_PERM, runtime, condition: str) -> dict:
    best_local_epoch = int(checkpoint["best_epoch_1based"]) - 1 - int(runner.CFG["base"]["start_epoch_zero_based"])
    kwargs = {}
    if condition == "prototype_off":
        kwargs["prototype_off"] = True
    elif condition == "hard_off":
        kwargs["hard_off"] = True
    elif condition == "semantic_swap":
        kwargs["semantic_swap"] = True
    elif condition == "original_text":
        kwargs["text_source"] = "original"
    elif condition != "full":
        raise ValueError(condition)
    capture: dict = {}
    with torch.no_grad():
        output, _ = forward_sc_perm(runner.E74, state, module, group, training=False,
                                    local_epoch=best_local_epoch, capture=capture, **kwargs)
        metrics = runtime.classification_metrics(output["logp"], state["test"], state["y"])
    return {"output": output, "metrics": metrics, "capture": capture}


def _checkpoint_audit(group: str, fold: int, preflight_record: dict,
                      base_reference: dict, e77_reference: dict) -> tuple[dict, dict, dict]:
    checkpoint, state, module, tracker, runtime, old_groups, ckpt_sha, ckpt_path = _load_checkpoint(
        group, fold, preflight_record
    )
    allowed = runner.allowed_train_ids(state)
    allowed_t = torch.as_tensor(allowed, dtype=torch.long, device=state["x"].device)
    test = np.asarray(state["test_np"], dtype=np.int64)
    overlap = sorted(set(np.asarray(state["train_np"], dtype=np.int64)) & set(test))
    expected_overlap = [376, 406] if fold == 2 else []
    _check(overlap == expected_overlap, f"unexpected fold {fold} train/test overlap {overlap}")
    allowed_t = torch.as_tensor(allowed, dtype=torch.long, device=module.class_valid.device)
    test_t = torch.as_tensor(test, dtype=torch.long, device=module.class_valid.device)
    valid_class_ids = torch.nonzero(module.class_valid, as_tuple=False).flatten()
    _check(torch.equal(valid_class_ids.sort().values, allowed_t.sort().values),
           f"typical memory IDs differ from train-minus-test in {group} fold {fold}")
    _check(not bool(module.class_valid[test_t].any()),
           f"test ID entered typical memory in {group} fold {fold}")
    _check(bool((module.class_labels[allowed_t] == state["y"][allowed_t]).all()),
           f"typical memory class label mismatch in {group} fold {fold}")
    hard_ids = module.hard_subject_ids[module.hard_subject_ids >= 0].detach().cpu().numpy()
    _check(not bool(np.isin(hard_ids, test).any()), f"test ID entered hard memory in {group} fold {fold}")
    local_epoch = int(checkpoint["best_epoch_1based"]) - 1 - int(runner.CFG["base"]["start_epoch_zero_based"])
    for direction, expected_label in ((0, 1), (1, 0)):
        ids = module.hard_subject_ids[direction]
        valid_ids = ids[ids >= 0]
        if valid_ids.numel():
            _check(bool((state["y"][valid_ids] == expected_label).all()),
                   f"hard-memory direction {direction} contains wrong true labels")
            _check(bool((module.hard_votes[direction, ids >= 0] >= runner.CFG["method"]["hard_error_min_votes"]).all()),
                   f"hard-memory direction {direction} contains an unstable record")
            _check(bool((module.hard_last_error[direction, ids >= 0] <= local_epoch).all()),
                   f"hard-memory direction {direction} has a future timestamp")
            # Best checkpoints store the memory snapshot used by that forward,
            # before the deterministic end-of-epoch expiry sweep.
            _check(bool(((local_epoch - module.hard_last_error[direction, ids >= 0])
                         <= runner.CFG["method"]["hard_memory_max_age_epochs"] + 1).all()),
                   f"hard-memory direction {direction} exceeds the readback snapshot age bound")
    rng_before = runtime.rng_sha(runtime.pack_rng())
    with torch.no_grad():
        output, hook = forward_sc_perm(runner.E74, state, module, group, training=False,
                                       local_epoch=local_epoch)
        metrics = runtime.classification_metrics(output["logp"], state["test"], state["y"])
    rng_after = runtime.rng_sha(runtime.pack_rng())
    saved_probability = np.asarray(checkpoint["test_probability"], dtype=np.float64)
    recomputed_probability = np.asarray(metrics["prob"], dtype=np.float64)
    _check(np.allclose(saved_probability, recomputed_probability, atol=2e-7, rtol=0),
           f"independent readback changed probabilities in {group} fold {fold}")
    _check(np.array_equal(np.asarray(checkpoint["test_prediction"]), np.asarray(metrics["pred"])),
           f"independent readback changed predictions in {group} fold {fold}")
    for key in PRIMARY:
        _check(abs(float(checkpoint["metrics"][key]) - float(metrics[key])) < 1e-10,
               f"independent readback changed {key} in {group} fold {fold}")
    base_row = base_reference[fold]
    e77_row = e77_reference[fold]
    row = {"group": group, "fold": fold, **_pct_metrics(metrics),
           "tp": int(metrics["asd_tp"]), "tn": int(metrics["td_tn"]),
           "fp": int(metrics["td_fp"]), "fn": int(metrics["asd_fn"]),
           "best_epoch_1based": int(checkpoint["best_epoch_1based"]),
           "epochs_executed": int(checkpoint["epochs_executed_at_selection"]),
           "checkpoint_sha256": ckpt_sha, "train_test_overlap_1based": json.dumps([int(i) + 1 for i in overlap]),
           "base_acc": base_row["acc"] * 100, "base_hard_auc": base_row["hard_auc"] * 100,
           "base_prob_auc": base_row["prob_auc"] * 100, "base_sen": base_row["sen"] * 100,
           "base_spe": base_row["spe"] * 100,
           "e77_g2_acc": float(e77_row["acc"]), "e77_g2_hard_auc": float(e77_row["hard_auc"]),
           "e77_g2_prob_auc": float(e77_row["prob_auc"]), "e77_g2_sen": float(e77_row["sen"]),
           "e77_g2_spe": float(e77_row["spe"]),
           "delta_acc_vs_base_pp": float(metrics["acc"]) * 100 - base_row["acc"] * 100,
           "delta_hard_auc_vs_base_pp": float(metrics["hard_auc"]) * 100 - base_row["hard_auc"] * 100,
           "delta_prob_auc_vs_base_pp": float(metrics["prob_auc"]) * 100 - base_row["prob_auc"] * 100,
           "delta_sen_vs_base_pp": float(metrics["sen"]) * 100 - base_row["sen"] * 100,
           "delta_spe_vs_base_pp": float(metrics["spe"]) * 100 - base_row["spe"] * 100,
           "delta_acc_vs_e77_g2_pp": float(metrics["acc"]) * 100 - float(e77_row["acc"]),
           "rng_unchanged_on_eval": rng_before == rng_after,
           "test_id_excluded_from_memory": True,
           "hook_calls": {"pre": hook["pre_calls"], "post": hook["post_calls"]}}
    _check(row["rng_unchanged_on_eval"], f"evaluation advanced RNG in {group} fold {fold}")

    interventions = {}
    if group == "E77.1-G2":
        full = {"output": output, "metrics": metrics, "capture": {}}
        conditions = ("prototype_off", "hard_off", "semantic_swap", "original_text")
        y_test = state["y"][state["test"]].detach().cpu().numpy()
        baseline_p = np.asarray(metrics["prob"], dtype=np.float64)
        baseline_pred = np.asarray(metrics["pred"], dtype=np.int64)
        baseline_logp = output["logp"][state["test"]].detach().cpu().numpy()
        baseline_margin = np.where(y_test == 0, baseline_logp[:, 0] - baseline_logp[:, 1],
                                   baseline_logp[:, 1] - baseline_logp[:, 0])
        for condition in conditions:
            changed = _intervention_eval(group, fold, checkpoint, state, module, runtime, condition)
            altered_p = np.asarray(changed["metrics"]["prob"], dtype=np.float64)
            altered_pred = np.asarray(changed["metrics"]["pred"], dtype=np.int64)
            altered_logp = changed["output"]["logp"][state["test"]].detach().cpu().numpy()
            altered_margin = np.where(y_test == 0, altered_logp[:, 0] - altered_logp[:, 1],
                                      altered_logp[:, 1] - altered_logp[:, 0])
            delta_p_asd = altered_p[:, 0] - baseline_p[:, 0]
            hard_ids_for_query = module.hard_subject_ids.detach().cpu().numpy()
            memory_rows = hard_ids_for_query[hard_ids_for_query >= 0]
            intervention_row = {
                "group": group, "fold": fold, "intervention": condition,
                **_pct_metrics(changed["metrics"]),
                "hard_prediction_changes": int((altered_pred != baseline_pred).sum()),
                "affected_test_row_indices_zero_based": json.dumps(test[altered_pred != baseline_pred].tolist()),
                "mean_abs_delta_p_asd": float(np.abs(delta_p_asd).mean()),
                "mean_delta_p_asd_true_asd": float(delta_p_asd[y_test == 0].mean()) if np.any(y_test == 0) else 0.0,
                "mean_delta_p_asd_true_td": float(delta_p_asd[y_test == 1].mean()) if np.any(y_test == 1) else 0.0,
                "mean_true_class_logp_margin_delta": float((altered_margin - baseline_margin).mean()),
                "mean_abs_true_class_logp_margin_delta": float(np.abs(altered_margin - baseline_margin).mean()),
                "hard_memory_subjects_in_fold": int(len(memory_rows)),
                "hard_memory_test_id_overlap": int(np.isin(memory_rows, test).sum()),
            }
            interventions[condition] = intervention_row
        row["checkpoint_memory"] = module.diagnostics()

    audit = {"group": group, "fold": fold, "status": "PASS",
             "checkpoint_path": str(ckpt_path), "checkpoint_sha256": ckpt_sha,
             "base_identity": checkpoint["base_identity"],
             "warmup_sha256": checkpoint["warmup_sha256"],
             "initial_rng_sha": checkpoint["warmup_rng_sha"],
             "best_epoch_1based": checkpoint["best_epoch_1based"],
             "test_id_count_excluded_from_memory": int(len(state["train_np"]) - len(allowed)),
             "train_test_overlap_1based": [index + 1 for index in overlap],
             "prototype_diagnostics": module.diagnostics(),
             "hard_history": checkpoint["hard_diagnostics"],
             "eval_rng_unchanged": rng_before == rng_after,
             "probability_readback_max_abs_delta": float(np.max(np.abs(saved_probability - recomputed_probability))),
             "test_indices_sha256": runner.digest_value(np.asarray(checkpoint["test_indices"], dtype=np.int64))}
    del checkpoint, state, module, tracker, output
    gc.collect()
    torch.cuda.empty_cache()
    return row, audit, interventions


def _mean_sd(values) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def _build_summaries(rows: list[dict], base: dict, e77: dict) -> tuple[list[dict], list[dict]]:
    summaries, paired = [], []
    for group in GROUPS:
        current = {int(row["fold"]): row for row in rows if row["group"] == group}
        _check(len(current) == 10, f"missing formal folds for {group}")
        summary = {"group": group, "folds": 10}
        for metric in PRIMARY:
            mean, sd = _mean_sd([current[fold][metric] for fold in range(1, 11)])
            summary[f"{metric}_mean_percent"] = mean
            summary[f"{metric}_population_sd_percent"] = sd
            summary[f"delta_{metric}_vs_base_mean_pp"] = _mean_sd([
                current[fold][metric] - base[fold][metric] * 100.0 for fold in range(1, 11)
            ])[0]
            summary[f"delta_{metric}_vs_e77g2_mean_pp"] = _mean_sd([
                current[fold][metric] - float(e77[fold][metric]) for fold in range(1, 11)
            ])[0]
            summary[f"{metric}_excluding_fold2_mean_percent"] = _mean_sd([
                current[fold][metric] for fold in range(1, 11) if fold != 2
            ])[0]
            summary[f"{metric}_excluding_fold2_population_sd_percent"] = _mean_sd([
                current[fold][metric] for fold in range(1, 11) if fold != 2
            ])[1]
        summaries.append(summary)

    for left, right, label in (("E77.1-G1", "base", "G1_vs_E74.4_G2"),
                               ("E77.1-G2", "base", "G2_vs_E74.4_G2"),
                               ("E77.1-G3", "base", "G3_vs_E74.4_G2"),
                               ("E77.1-G2", "E77.1-G1", "G2_minus_G1"),
                               ("E77.1-G2", "E77.1-G3", "G2_minus_G3")):
        left_rows = {int(row["fold"]): row for row in rows if row["group"] == left}
        for metric in PRIMARY:
            deltas = []
            for fold in range(1, 11):
                if right == "base":
                    value = left_rows[fold][metric] - base[fold][metric] * 100.0
                elif right == "E77.1-G1":
                    reference = next(row for row in rows if row["group"] == right and int(row["fold"]) == fold)
                    value = left_rows[fold][metric] - reference[metric]
                else:
                    reference = next(row for row in rows if row["group"] == right and int(row["fold"]) == fold)
                    value = left_rows[fold][metric] - reference[metric]
                deltas.append(value)
            mean, sd = _mean_sd(deltas)
            paired.append({"comparison": label, "metric": metric, "folds": 10,
                           "mean_delta_pp": mean, "population_sd_delta_pp": sd,
                           "excluding_fold2_mean_delta_pp": _mean_sd([v for fold, v in enumerate(deltas, 1) if fold != 2])[0],
                           "fold_deltas_pp": json.dumps(deltas)})
    return summaries, paired


def _fmt(mean: float, sd: float) -> str:
    return f"{mean:.4f} +/- {sd:.4f}"


def _report(summaries: list[dict], paired: list[dict], audit_rows: list[dict],
            interventions: list[dict], preflight_record: dict, decision: str) -> str:
    metric_order = ("acc", "hard_auc", "prob_auc", "sen", "spe")
    summary_by_group = {row["group"]: row for row in summaries}
    base_source = runner.CFG["reference_metrics"]
    lines = [
        "# Experiment 77.1: SC-PERM",
        "",
        f"**Decision: {decision}.** Metrics are percent; means and population standard deviations are across the inherited ten outer folds.",
        "",
        "## Audit basis",
        "",
        f"- Formal base: `{BASE}`; runner SHA `[{preflight_record['e74_runner_sha256']}]` and E74.4 preflight SHA `{preflight_record['e74_preflight_sha256']}`. Each task restores the matching E61 epoch-10 warmup model, optimizer, split and Python/NumPy/Torch CPU/CUDA RNG snapshot.",
        f"- E74.4 reference: ACC {base_source['acc'][0]:.4f}+/-{base_source['acc'][1]:.4f}; Hard-AUC/BA {base_source['hard_auc'][0]:.4f}+/-{base_source['hard_auc'][1]:.4f}; Prob-AUC {base_source['prob_auc'][0]:.4f}+/-{base_source['prob_auc'][1]:.4f}; SEN {base_source['sen'][0]:.4f}+/-{base_source['sen'][1]:.4f}; SPE {base_source['spe'][0]:.4f}+/-{base_source['spe'][1]:.4f}. The fold checkpoint source is `{BASE / 'results/fold_metrics.csv'}`.",
        f"- Local source manifest SHA: `{preflight_record['source_sha256']}`. All protected source hashes were rechecked after training; see `manifest.json`.",
        "- The true E74.4-G2 model is `dugsa.PLNNoPEN`, so there is no active PEN in this base. SC-PERM intercepts the audited CAM-to-OutputLayer boundary: CAM input `[871,144]`, four modality blocks each `[871,36]`, OutputLayer hidden `[871,36]`, two output log-probabilities. DARG `[871,320]`, CAM internals, attention outputs and the original head are retained.",
        "- The 36-D modality representations are learned latent blocks. They have no verified anatomical ROI identities; this method assigns none.",
        "- Historical training settings are unchanged: Adam on GFR/CAM (`lr=.0038`, `wd=.11`), AdamW on DARG and the appended SC-PERM parameters (`lr=.001`, `wd=.01`), fixed learning rates, full-graph one-step epochs, 450 max epochs, stop when `wait > 200`, and strict test-fold ACC maxima for checkpoint selection. Original loss terms remain `L_cls + L_graph + .05 L_pair + .01 L_adv + .01 L_preserve`; SC-PERM adds only the configured prototype CE and hard-evidence ranking losses.",
        "- The historical Fold-2 train/test overlap is retained at one-based IDs 377 and 407. Both are excluded from prototype, temporal error, and auxiliary-loss indexes. The original base classification loss still follows E74.4's train indices, which include this inherited overlap.",
        "- `val=test` checkpoint selection, pre-outer-CV RFE and transductive full-graph message propagation are preserved. Train memories only store allowed training rows, but their DARG-CAM features can have indirect transductive dependence on test-node features. These results are one seed under a biased historical evaluation protocol, not independent clinical generalization evidence.",
        "",
        "## Literature and text audit",
        "",
        "SPromptGL's published PEN uses frozen CLIP RN50 disease descriptions to select learned modality features, with modality-specific input projections and transformer/fusion processing in the original network. BrainPrompt uses multilevel brain and disease prompts with representation alignment. SABER proposes semantic-aligned brain-network representations. SC-PERM instead combines patient-to-prototype feature similarity with a shared text-to-prototype semantic relation inside top-2 case retrieval, then uses the resulting class evidence to correct the inherited classifier scores. PGP-SAM, Concentrate on Weakness and HP²L motivate only general prototype or hard-case mechanisms; their segmentation or broader disease settings do not establish transfer validity here.",
        "",
        "- SPromptGL: [MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) and [official code](https://github.com/wanxixi11/SPromptGL).",
        "- BrainPrompt: [MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf).",
        "- SABER: [arXiv 2607.01901](https://arxiv.org/abs/2607.01901); no claim of a locally audited official implementation.",
        "- PGP-SAM: [arXiv 2501.06692](https://arxiv.org/abs/2501.06692); Concentrate on Weakness: [IJCAI 2025](https://ijcai.org/proceedings/2025/139); HP²L: [Medical Image Analysis DOI](https://doi.org/10.1016/j.media.2026.104063).",
        "- Prompt wording follows the ASD functional-connectivity heterogeneity study [Nature Mental Health DOI](https://doi.org/10.1038/s44220-026-00656-y), which reports individual heterogeneity with network-level convergence, not a universal edge-level ASD signature. Motion-related connectivity confounding is supported by [Power et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3254728/). The eight texts are hypotheses for model conditioning, not expert-approved clinical descriptions.",
        "",
        "The fixed medical texts are the four TD concepts followed by four ASD concepts in `config.json`. They discuss population-level variability and cautious interpretation; they do not assert that every ASD participant has a specific connection anomaly. Motion, acquisition quality and site are treated as shared measurement context. PHENO, ANAT and FUNC are not assumed to be ASD-specific biomarkers; CORR is the primary functional-connectivity context. No latent CAM block is mapped to an anatomical ROI.",
        "",
        f"Frozen text cache SHA: `{preflight_record['text_cache_audit']['cache_sha256']}`. The cache keeps both the exact original eight SPromptGL texts and the fixed medical eight, with RN50 weights SHA `{preflight_record['text_cache_audit']['text_encoder']['sha256']}`. CLIP class-anchor cosine values, token lengths and full pairwise text cosine matrices are recorded in `preflight.json`.",
        "",
        "## Method",
        "",
        "For CAM input `H in R^(N x 144)`, reshape `h[i,m] in R^36`, with four modalities in `[PHENO, ANAT, FUNC, CORR]` order. Each class `c in {ASD=0, TD=1}` and modality `m` has up to four normalized train-fold centers `p[c,m,k] in R^36`. A training query subtracts its own detached subject contribution from its assigned center before retrieval.",
        "",
        "The shared text and prototype projections are `a[c] = normalize(W_t mean_j CLIP(t[c,j])) in R^32` and `v[c,m,k] = normalize(W_p p[c,m,k]) in R^32`. The retrieval score and class evidence are:",
        "",
        "```text",
        "q[i,c,m,k] = cos(h[i,m], p[c,m,k]) + beta * eta[m] * cos(v[c,m,k], a[c])",
        "w[i,c,m] = Top2Softmax(q / 0.2)",
        "r[i,c,m] = sum_k w[i,c,m,k] * q[i,c,m,k]",
        "S[i,c] = sum_m eta[m] * r[i,c,m],  eta = [0.15, 0.10, 0.10, 1.0]",
        "FP: H[i,TD] += gated_relu(sim_FP - S[i,ASD]/1.35 - 0.05)",
        "FN: H[i,ASD] += gated_relu(sim_FN - S[i,TD]/1.35 - 0.05)",
        "E[i,c] = S[i,c] + clip(H[i,c], 0, 0.5)",
        "d[i] = lambda * clip(E[i,ASD] - E[i,TD], -2, 2),  0 <= lambda <= 0.75",
        "logp[i] = log_softmax(logp_base[i] + [d[i]/2, -d[i]/2])",
        "L = L_E74.4 + 0.10 * CE(S[allowed_train], y) + 0.05 * hard_ranking_loss(stable_train_FP_FN)",
        "```",
        "",
        "Only the log-probabilities returned by the original CAM OutputLayer are modified. FP evidence is TD-directed when FP-pattern similarity exceeds competing ASD typical support by the fixed margin; FN evidence is ASD-directed when FN-pattern similarity exceeds competing TD support. Ambiguous overlaps yield no hard evidence. Error cases are unique training-patient records with an EMA feature, first/last error epoch, capped vote support, frequency and deterministic novelty/recency eviction. The 3-epoch history requires at least two wrong predictions; near-boundary cases remain separate diagnostics. Each memory bank is updated at epoch boundaries and never stores test rows.",
        "",
        "## Groups and results",
        "",
        "| Group | ACC | Hard-AUC / BA | Prob-AUC | SEN (ASD) | SPE (TD) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| E74.4-G2 reference | {base_source['acc'][0]:.4f} +/- {base_source['acc'][1]:.4f} | {base_source['hard_auc'][0]:.4f} +/- {base_source['hard_auc'][1]:.4f} | {base_source['prob_auc'][0]:.4f} +/- {base_source['prob_auc'][1]:.4f} | {base_source['sen'][0]:.4f} +/- {base_source['sen'][1]:.4f} | {base_source['spe'][0]:.4f} +/- {base_source['spe'][1]:.4f} |",
    ]
    for group in GROUPS:
        row = summary_by_group[group]
        lines.append("| " + group + " | " + " | ".join(
            _fmt(row[f"{metric}_mean_percent"], row[f"{metric}_population_sd_percent"]) for metric in metric_order
        ) + " |")
    lines.extend(["", "G1 changes only the text bank from the audited original SPromptGL prompts; G2 uses the predeclared medical candidate texts; G3 keeps the medical text and full typical-prototype structure while disabling hard-memory reads and training contribution. Each group has ten independently initialized fold runs.", ""])
    lines.append("Paired fold deltas (mean +/- population SD, percentage points):")
    lines.append("")
    lines.append("| Comparison | Metric | Mean delta (pp) | SD (pp) |")
    lines.append("|---|---|---:|---:|")
    for comparison in paired:
        lines.append(f"| {comparison['comparison']} | {comparison['metric']} | {comparison['mean_delta_pp']:+.4f} | {comparison['population_sd_delta_pp']:.4f} |")
    lines.extend(["", "Excluding Fold 2 is a descriptive nine-fold summary only; no split or checkpoint rule is changed. Full per-fold values and all base/E77 paired differences are in `fold_metrics.csv` and `paired_comparisons.csv`.", ""])
    lines.append("## Mechanism diagnostics")
    lines.append("")
    lines.append(f"- Independent checkpoint readback passed {len(audit_rows)}/30 folds. Each saved model, optimizer, RNG, typical memory, error history, selected epoch and test probability vector was reloaded; predictions and all five primary metrics were recalculated.")
    lines.append(f"- G2 fixed-checkpoint interventions: {len(interventions)}/40 conditions recorded across prototype-off, hard-memory-off, ASD/TD semantic swap and original-text replacement. Counts, probability shifts, true-class log-probability margin changes and affected row indices are in `interventions.csv`.")
    lines.append(f"- Preflight real-patient evidence shift, original-versus-medical class evidence delta: {preflight_record['semantic_gradient_audit']['class_score_mean_abs_original_medical_delta']:.8g}; semantic-swap retrieval-weight delta: {preflight_record['semantic_gradient_audit']['retrieval_weight_mean_abs_semantic_swap_delta']:.8g}; decision-correction delta: {preflight_record['semantic_gradient_audit']['decision_correction_mean_abs_semantic_swap_delta']:.8g}.")
    lines.append("- The independently audited `memory_audit.json` records per-fold class-bank occupancy, usage, pairwise collapse, persistent FP/FN patients, active error prototypes, prototype evidence distributions, and the four G2 fixed-weight interventions. A performance increase is not attributed to semantics or hard memory when its corresponding interventions show negligible evidence or probability contribution.")
    lines.extend(["", "## Decision and limits", "", f"Final decision: **{decision}**. This is a single-seed result under the inherited historical protocol. It does not establish seed robustness or unbiased independent clinical generalization. If all three groups fail to improve the base meaningfully, no E77.2 run should start automatically.", ""])
    return "\n".join(lines)


def finalize() -> dict:
    preflight_record = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    _check(preflight_record.get("status") == "PASS", "cannot finalize without a passing SC-PERM preflight")
    base_reference = _read_base_reference()
    e77_reference = _e77_rows()
    rows, audits, intervention_rows, memory_rows = [], [], [], []
    for group in GROUPS:
        for fold in range(1, 11):
            row, audit, interventions = _checkpoint_audit(group, fold, preflight_record,
                                                            base_reference, e77_reference)
            rows.append(row)
            audits.append(audit)
            memory_rows.append({"group": group, "fold": fold,
                                "prototype_diagnostics": audit["prototype_diagnostics"],
                                "hard_history": audit["hard_history"]})
            intervention_rows.extend(interventions.values())
    _check(len(audits) == 30 and all(item["status"] == "PASS" for item in audits),
           "not all thirty independent checkpoint audits passed")
    _check(len(intervention_rows) == 40, "expected forty G2 fixed-weight interventions")
    protected_after = runner.source_manifest()
    protected_before = preflight_record["protected_sources_unchanged_baseline"]
    audit_path = str((HERE / "audit.py").resolve())
    before_without_current_audit = {key: value for key, value in protected_before.items() if key != audit_path}
    after_without_current_audit = {key: value for key, value in protected_after.items() if key != audit_path}
    _check(before_without_current_audit == after_without_current_audit,
           "a protected training, root, data or historical source changed during E77.1")
    audit_repair = {
        "reason": "Checkpoint hard-memory expiry validation now accounts for the selected pre-update forward snapshot; no training/model/config/data or checkpoint was changed.",
        "preflight_audit_sha256": protected_before[audit_path],
        "final_audit_sha256": protected_after[audit_path],
        "training_source_sha256": preflight_record["source_sha256"],
        "retraining": False,
    }
    summaries, paired = _build_summaries(rows, base_reference, e77_reference)
    runner.E74.write_csv(RESULTS / "fold_metrics.csv", rows)
    runner.E74.write_csv(RESULTS / "summary.csv", summaries)
    runner.E74.write_csv(RESULTS / "paired_comparisons.csv", paired)
    runner.E74.write_csv(RESULTS / "checkpoint_audit.csv", audits)
    runner.E74.write_csv(RESULTS / "interventions.csv", intervention_rows)
    runner.atomic_json(RESULTS / "memory_audit.json", {
        "preflight_synthetic_tests": preflight_record["synthetic_hard_memory_audit"],
        "preflight_real_semantic_test": preflight_record["semantic_gradient_audit"],
        "folds": memory_rows,
    })
    group_summary = {row["group"]: row for row in summaries}
    g2 = group_summary["E77.1-G2"]
    base_cfg = runner.CFG["reference_metrics"]
    decisions = runner.CFG["decision_rule"]
    mechanism = preflight_record["semantic_gradient_audit"]
    g2_vs_base_acc = g2["acc_mean_percent"] - float(base_cfg["acc"][0])
    g2_vs_base_hard = g2["hard_auc_mean_percent"] - float(base_cfg["hard_auc"][0])
    g2_vs_base_prob = g2["prob_auc_mean_percent"] - float(base_cfg["prob_auc"][0])
    g2_vs_base_sen = g2["sen_mean_percent"] - float(base_cfg["sen"][0])
    g2_vs_base_spe = g2["spe_mean_percent"] - float(base_cfg["spe"][0])
    g2_vs_g3_ba = g2["hard_auc_mean_percent"] - group_summary["E77.1-G3"]["hard_auc_mean_percent"]
    condition_full = {int(item["fold"]): item for item in intervention_rows if item["intervention"] == "full"}
    condition_hard = {int(item["fold"]): item for item in intervention_rows if item["intervention"] == "hard_off"}
    hard_margin_effect = _mean_sd([condition_hard[fold]["mean_abs_true_class_logp_margin_delta"]
                                   for fold in range(1, 11)])[0]
    semantic_margin_effect = _mean_sd([
        next(item for item in intervention_rows if int(item["fold"]) == fold and item["intervention"] == "semantic_swap")
        ["mean_abs_true_class_logp_margin_delta"] for fold in range(1, 11)
    ])[0]
    performance_pass = (
        g2_vs_base_acc >= float(decisions["minimum_g2_acc_gain_pp"])
        and g2_vs_base_hard >= 0.0 and g2_vs_base_prob >= 0.0
        and g2_vs_base_sen >= -float(decisions["max_sen_drop_pp"])
        and g2_vs_base_spe >= -float(decisions["max_spe_drop_pp"])
    )
    semantics_pass = (mechanism["class_score_mean_abs_semantic_swap_delta"] >= float(decisions["minimum_semantic_score_delta"])
                      and semantic_margin_effect >= float(decisions["minimum_semantic_margin_delta"]))
    hard_pass = (g2_vs_g3_ba > 0.0 and hard_margin_effect >= float(decisions["minimum_hard_margin_delta"]))
    decision = "GO" if performance_pass and semantics_pass and hard_pass else "NO-GO"
    decision_audit = {
        "decision": decision, "rule": decisions,
        "g2_delta_vs_base_pp": {"acc": g2_vs_base_acc, "hard_auc": g2_vs_base_hard,
                                "prob_auc": g2_vs_base_prob, "sen": g2_vs_base_sen, "spe": g2_vs_base_spe},
        "g2_minus_g3_hard_auc_pp": g2_vs_g3_ba,
        "semantic_intervention_mean_abs_true_margin_delta": semantic_margin_effect,
        "hard_off_intervention_mean_abs_true_margin_delta": hard_margin_effect,
        "performance_gate": performance_pass, "semantic_gate": semantics_pass, "hard_memory_gate": hard_pass,
    }
    _write_json(RESULTS / "decision.json", decision_audit)
    _write_json(RESULTS / "audit_repair.json", audit_repair)
    report_text = _report(summaries, paired, audits, intervention_rows, preflight_record, decision)
    temp = RESULTS / "report.md.tmp"
    temp.write_text(report_text, encoding="utf-8")
    os.replace(temp, RESULTS / "report.md")
    manifest = {
        "experiment": runner.CFG["experiment"], "status": "COMPLETE_AUDITED",
        "decision": decision, "directory": str(HERE), "seed": runner.CFG["seed"],
        "groups": list(GROUPS), "completed_group_folds": 30,
        "source_sha256": preflight_record["source_sha256"],
        "local_code_sha256": {name: runner.sha_file(HERE / name) for name in ("run.py", "model.py", "memory.py", "audit.py", "config.json")},
        "text_cache_sha256": preflight_record["text_cache_audit"]["cache_sha256"],
        "protected_source_hashes_unchanged": True,
        "audit_only_post_training_repair": audit_repair,
        "warmup_parity_folds_passed": sum(item["parity"]["status"] == "PASS" for item in preflight_record["g2_g3_common_warmup_parity"]),
        "disabled_base_parity_folds_passed": sum(item["rng_after_train_equal"] for item in preflight_record["base_disabled_parity"]),
        "checkpoint_independent_audits_passed": len(audits),
        "preflight": "PASS", "decision_audit": decision_audit,
        "artifacts": {name: str(RESULTS / name) for name in (
            "report.md", "fold_metrics.csv", "summary.csv", "paired_comparisons.csv",
            "checkpoint_audit.csv", "interventions.csv", "memory_audit.json", "preflight.json", "decision.json", "audit_repair.json"
        )},
        "limitations": runner.CFG["audit"],
    }
    _write_json(HERE / "manifest.json", manifest)
    return manifest


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "finalize"))
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight()
    else:
        result = finalize()
    print(json.dumps({"status": result.get("status", "PASS"),
                      "decision": result.get("decision"),
                      "completed_group_folds": result.get("completed_group_folds")}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
