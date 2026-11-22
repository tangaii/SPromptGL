"""Independently audit label scope, relation endpoints, training and replay."""
from __future__ import annotations

import csv
import json

from core import (
    CANDIDATE_IDS, CONFIG, FOLD, MASKED_COMPONENTS, MODEL_VARIANT, checkpoint_in_memory,
    engine, evaluator, ext, h, historical_pair_sampler, optimization, parent, paths, prior, registered_rows,
)
from sources import INTERVENTIONS, residual_records, score, utc_now

import numpy as np
import torch
from torch.nn import functional as F


def membership_proof(ctx):
    state, members, old = ctx["state"], ctx["members"], ctx["old_members"]
    rows = registered_rows(ctx["task_group"])
    labels, train, test, legal = state["labels_np"], members["train"], members["test"], members["legal"]
    reg = ctx["provenance"]["registration"]
    labels_path = h.REPO / "data/ABIDE/processed_standard_data.csv"
    with labels_path.open(newline="") as stream:
        reader = csv.reader(stream)
        label_column = next(reader)[-1]
        raw_labels = np.array([float(record[-1]) for record in reader], dtype=np.float64)
    candidates = [patient - 1 for patient in CANDIDATE_IDS]
    h.require(np.array_equal(raw_labels - 1, labels) and np.all(raw_labels[candidates] == 1)
              and all(i in test and i not in old["train"] for i in candidates),
              "Original label-column encoding or Fold 8 candidate scope differs")
    h.require(np.array_equal(train[:-2], old["train"]) and train[-2:].tolist() == rows
              and len(train) == len(np.unique(train)) == 786 and len(legal) == 784 and len(test) == 87
              and np.array_equal(legal, old["train"]) and np.array_equal(test, old["test"])
              and np.intersect1d(train, test).tolist() == sorted(rows)
              and torch.where(~ctx["module"].spectrum.allowed)[0].tolist() == sorted(rows)
              and np.array_equal(ctx["reference"]["legal"], legal)
              and torch.equal(ctx["labels"], state["y"][state["legal"]]),
              "Independent Fold 8 membership or mask differs")
    for key in ("train", "legal", "test", "weights", "train_pairs", "train_same"):
        value = state[key]
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        np.testing.assert_array_equal(value, members[key], err_msg=key)

    before = h.rng_sha()
    _, e61 = historical_pair_sampler()
    support = np.unique(np.sort(np.concatenate(state["supports"]), axis=1), axis=0)
    pairs, same = e61.load_base().sample_balanced_pairs(
        np.r_[ctx["warmup"]["train_indices"], rows], (raw_labels - 1).astype(np.int64), CONFIG["training"]["pair_count"],
        np.random.default_rng(CONFIG["training"]["pair_seed"]), support,
    )
    h.require(h.rng_sha() == before, "Independent relation rederivation consumed global RNG")
    np.testing.assert_array_equal(pairs, members["train_pairs"])
    np.testing.assert_array_equal(same, members["train_same"])
    np.testing.assert_array_equal(same, labels[pairs[:, 0]] == labels[pairs[:, 1]])
    h.require(pairs.shape == (4096, 2) and np.isin(pairs, train).all()
              and np.bincount(same, minlength=2).tolist() == [2048, 2048],
              "Independently rebuilt relation pairs differ")
    np.testing.assert_array_equal(members["weights"], (len(train) / (2 * np.bincount(labels[train]))).astype(np.float32))
    endpoints, times = np.unique(pairs, return_counts=True)
    missing = train[~np.isin(train, endpoints)]
    pair_stats = {}
    for row in rows:
        occurrences = int(times[endpoints == row].sum())
        pair_stats[str(row + 1)] = {
            "eligible": bool(row in train), "sampled": occurrences > 0, "endpoint_occurrences": occurrences,
            "same_occurrences": int((pairs[same == 1] == row).sum()),
            "different_occurrences": int((pairs[same == 0] == row).sum()),
        }
    h.require(len(endpoints) == reg["covered_rows"] and (missing + 1).tolist() == reg["uncovered_ids_1based"]
              and pair_stats == reg["target_pairs"]
              and h.sha(labels[test]) == reg["test_labels_sha"],
              "Independently counted relation endpoints differ")
    return {
        **reg, "raw_label_path": str(labels_path), "raw_label_column": label_column,
        "candidates": [{"patient_id_1based": i + 1, "patient_index_0based": i,
                        "raw_label": int(raw_labels[i]), "encoded_label": int(labels[i]), "in_fold8_test": True}
                       for i in candidates],
        "ordered_pairs_rederived": True, "pair_endpoints_all_darg_members": True,
        "global_rng_unchanged_during_rederive": True,
        "original_test_order_exact": True, "other_candidate_absent_from_training": True,
        "original_train_order_exact": True, "historical_features_and_supports_exact": True,
    }


def verify_target_mask(ctx, capture):
    """Independently verify the exact mask against the registered patients."""
    rows = registered_rows(ctx["task_group"])
    allowed = ctx["module"].spectrum.allowed.detach().cpu().numpy()
    expected = np.ones_like(allowed, dtype=bool)
    expected[rows] = False
    np.testing.assert_array_equal(allowed, expected)
    for key in MASKED_COMPONENTS:
        value = capture["details"][key][rows].detach().cpu().numpy()
        np.testing.assert_array_equal(value, np.zeros_like(value), err_msg=key)
    np.testing.assert_array_equal(capture["enhanced"][rows].detach().cpu().numpy(),
                                  capture["cam"][rows].detach().cpu().numpy())


def gaussian_proof(ctx, proof):
    ref, legal, state = ctx["reference"], ctx["members"]["legal"], ctx["state"]
    np.testing.assert_array_equal(ref["legal"], legal)
    np.testing.assert_array_equal(ref["labels"], state["labels_np"][legal])
    np.testing.assert_array_equal(ctx["module"].statistics.legal.cpu().numpy(), legal)
    initial = h.rng_sha()
    rebuilt = prior.rebuild_reference(FOLD, ctx["warm_cam"], ctx["original"], legal, ref["labels"].copy())
    h.require(h.sha(ref) == h.sha(rebuilt) and h.rng_sha() == initial,
              "Gaussian exact reconstruction or RNG failed")
    return {"legal_rows": len(legal), "legal_index_sha": h.sha(legal),
            "gaussian_reference_sha": h.sha(ref), "deterministic_rebuild_exact": True,
            **proof.f.class_banks(ctx)}


def gradient_scope(run, ctx, dropout):
    state, module = ctx["state"], ctx["module"]
    rows = registered_rows(ctx["task_group"])
    embeddings = []
    handles = [encoder.register_forward_hook(lambda layer, args, value: embeddings.append(value))
               for encoder in state["model"].encoders]
    before = h.rng_sha()
    try:
        with run.dropout_inventory() as calls:
            output, cap = run.forward(state, True, module)
    finally:
        for hook in handles:
            hook.remove()
    h.require(calls == dropout and len(embeddings) == 4, "Historical Dropout or DARG encoders differ")
    native_rng = h.rng_sha()
    verify_target_mask(ctx, cap)
    native = F.nll_loss(output["logp"][state["train"]], state["y"][state["train"]], state["weights"])
    h.require(torch.equal(native, output["cls"]), "DARG classification objective differs")
    logp_grad, cam_grad = torch.autograd.grad(output["cls"], (output["logp"], cap["cam"]), retain_graph=True)
    expected = state["weights"][state["y"][rows]] / state["weights"][state["y"][state["train"]]].sum()
    h.require(torch.allclose(logp_grad[rows, 0], -expected, rtol=1e-6, atol=0)
              and logp_grad[rows, 1].count_nonzero().item() == 0,
              "Expected direct DARG ASD classification gradient is absent")

    logits = tuple(output["info"]["pair_logits"])
    pairs = state["train_pairs"]
    pair_bce = torch.stack([F.binary_cross_entropy_with_logits(values, state["train_same"]) for values in logits]).mean()
    h.require(torch.equal(pair_bce, output["pair"]), "Historical relation BCE differs")
    logits_grad = torch.autograd.grad(output["pair"], logits, retain_graph=True)
    embedding_grad = torch.autograd.grad(output["pair"], embeddings, retain_graph=True)
    related_loss = 0.05 * output["pair"] + 0.01 * output["adv"] + 0.01 * output["preserve"]
    related_grad = torch.autograd.grad(related_loss, embeddings, retain_graph=True)
    outside = torch.ones_like(state["y"], dtype=torch.bool)
    outside[pairs.reshape(-1)] = False
    h.require(all(not torch.count_nonzero(grad[outside]).item() for grad in (*embedding_grad, *related_grad)),
              "Direct relation loss has gradients outside sampled endpoints")
    patients = {}
    for row in rows:
        selected = (pairs == row).any(dim=1)
        sampled = bool(selected.any().item())
        bce_logits = [float(grad[selected].double().norm()) for grad in logits_grad]
        bce_embedding = [float(grad[row].double().norm()) for grad in embedding_grad]
        weighted_embedding = [float(grad[row].double().norm()) for grad in related_grad]
        target_cam_grad = float(cam_grad[row].double().norm())
        target_logp_grad = float(logp_grad[row].double().norm())
        h.require(sampled or not any((*bce_logits, *bce_embedding, *weighted_embedding)),
                  f"Unsampled patient {row + 1} has a direct relation gradient")
        h.require(np.isfinite([target_cam_grad, target_logp_grad, *bce_embedding, *weighted_embedding]).all()
                  and target_cam_grad > 0 and target_logp_grad > 0,
                  f"Patient {row + 1} lacks a valid DARG classification gradient")
        patients[str(row + 1)] = {
            "classification_logp_grad_l2": target_logp_grad, "classification_cam144_grad_l2": target_cam_grad,
            "relation_eligible": True, "relation_sampled": sampled, "relation_sampled_pairs": int(selected.sum()),
            "relation_direct_bce_grad_l2": float(np.linalg.norm(bce_embedding)),
            "relation_direct_weighted_grad_l2": float(np.linalg.norm(weighted_embedding)),
            "relation_bce_logit_grad_l2_by_modality": bce_logits,
            "relation_bce_embedding_grad_l2_by_modality": bce_embedding,
            "relation_weighted_embedding_grad_l2_by_modality": weighted_embedding,
            "scmr_delta_grad_l2": 0.0,
        }

    auxiliary, _ = optimization.losses(MODEL_VARIANT, state["cam"].Outputlayer, cap,
                                        state["legal"], ctx["labels"],
                                        module.spectrum.basis.to(cap["cam"]), ctx["hard_initial"])
    for name, loss in auxiliary.items():
        aux_cam, = torch.autograd.grad(loss, cap["cam"], retain_graph=True)
        h.require(not aux_cam[state["test"]].count_nonzero().item(),
                  f"Held-out test CAM has a direct gradient from SCMR {name}")
    aux_delta, = torch.autograd.grad(auxiliary["full"], cap["details"]["delta"], retain_graph=True)
    trace = optimization.inspect_paths(ctx["families"], cap, auxiliary["full"])
    h.require(aux_delta[rows].count_nonzero().item() == 0 and aux_delta[state["legal"]].norm().item() > 0
              and all(v > 0 for v in trace["from_full_ce_through_delta"].values())
              and all(v > 0 for v in trace["per_space_directional_gradient"])
              and native_rng == h.rng_sha(), "Legal spectral/medical path gradient or RNG differs")
    return {
        "dropout_exact": True, "rng_before": before, "rng_after": native_rng,
        "patients": patients,
        "relation_gradient_zero_outside_endpoints": True, "modality_order": ["PHENO", "ANAT", "FUNC", "CORR"],
        "scmr_all_auxiliary_test_cam_grad_zero": True,
        "scmr_legal_delta_grad_l2": float(aux_delta[state["legal"]].norm()),
        "end_to_end_gradient": trace,
    }


def preflight(group, hashes):
    run = engine(group)
    proof = evaluator(run, f"preflight_{group}")
    gpu = h.archive.verify_device()
    torch.cuda.reset_peak_memory_stats()
    ctx = run.setup(MODEL_VARIANT, FOLD, verify_spectrum=True)
    members = membership_proof(ctx)
    geometry = proof.f.check_original_o(ctx)
    gaussian = gaussian_proof(ctx, proof)
    initial = checkpoint_in_memory(run, ctx, hashes, epoch=10)
    isolation = ext.parent_preflight.legal_label_isolation(ctx, initial)
    _, cap = run.evaluate(ctx)
    h.require(torch.equal(cap["cam"], cap["enhanced"]) and not cap["details"]["delta"].count_nonzero().item(),
              "Historical SCMR zero initialization differs")
    initial_formula = proof.f.branch_formulas(ctx, cap)
    baseline, dropout, rng = ext.parent_preflight.historic_signature(ctx, wrapped=False)
    canonical = h.read_json(parent.paths(MODEL_VARIANT, FOLD)["preflight"])
    h.require(len(dropout) == CONFIG["training"]["dropout_calls"] and dropout == canonical["dropout"]
              and rng == h.hist_trace(FOLD)[11], "First preflight Dropout or RNG differs from history")
    run.restore(ctx, initial, hashes)
    result, diagnostics, step_rng, _, _ = run.transition(ctx, 11, dropout, inspect=True)
    h.require(step_rng == rng and h.sha(h.base_bundle(ctx["state"])) == baseline["after_update"]
              and diagnostics["losses"]["difference"] == 0
              and all(diagnostics["budget"][family]["total_applied"] == 0 for family in optimization.ORIGINAL),
              "Zero-residual first transition differs from the native update")
    step_sha = h.sha((result, diagnostics, step_rng, h.sha(run.ema(ctx))))
    saved = checkpoint_in_memory(run, ctx, hashes, epoch=11)
    gradients = gradient_scope(run, ctx, dropout)
    run.restore(ctx, saved, hashes)
    next_step = run.next_witness(ctx, 12, dropout)
    run.restore(ctx, saved, hashes)
    h.require(run.next_witness(ctx, 12, dropout) == next_step and next_step["rng"] == h.hist_trace(FOLD)[12],
              "Serialized next training and evaluation do not replay exactly")
    _, cap = run.evaluate(ctx)
    proof.f.branch_formulas(ctx, cap)
    rows = registered_rows(group)
    test = ctx["members"]["test"]
    verify_target_mask(ctx, cap)
    for key in ("delta", "delta_parallel", "delta_perp"):
        h.require(torch.count_nonzero(cap["details"][key][test[~np.isin(test, rows)]].norm(dim=1)).item() == 85,
                  f"{key} is inactive for a non-target Fold 8 test patient")
    return {
        "status": "PASS", "completed_at_utc": utc_now(), "group": group, "fold": FOLD, "device": gpu,
        "implementation": hashes, "provenance": ctx["provenance"], "shared_initial": ctx["shared_initial"],
        "membership": members, "gaussian": gaussian, "original_o": geometry, "legal_label_isolation": isolation,
        "gradients": gradients, "initial_formula": initial_formula, "initial_ema": initial["ema_sha"],
        "initial_residual_exact_zero": True, "native_first_update_exact": True, "dropout": dropout,
        "first_step_sha": step_sha, "first_rng": step_rng, "next_train_eval_witness": next_step,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "process_gpu_mib": h.archive.gpu_sample()["used_mib"] - gpu["used_mib"],
    }


def checkpoint_audit(group, manifest):
    files = paths(group)
    task = h.read_json(files["task"])
    pre = h.read_json(files["preflight"])
    h.require(task["status"] == "TRAINED" and task["task_group"] == group
              and all(h.sha_file(files[k]) == task[k + "_sha"] for k in ("best", "latest", "journal"))
              and pre["status"] == "PASS" and h.sha_file(files["preflight"]) == manifest["preflight"]["groups"][group]["sha256"],
              "Training files or frozen preflight differ")
    run = engine(group)
    proof = evaluator(run, f"checkpoint_{group}")
    ctx = run.setup(MODEL_VARIANT, FOLD)
    h.require(ctx["provenance"] == pre["provenance"] and ctx["shared_initial"] == pre["shared_initial"],
              "Recreated initialized state differs")
    members = membership_proof(ctx)
    h.require(members == pre["membership"], "Membership or relation coverage differs from preflight")
    gaussian = gaussian_proof(ctx, proof)
    best, latest = (h.load(files[k]) for k in ("best", "latest"))
    journal = proof.audit_journal(ctx, task, best, latest, pre)
    h.require(journal["ema_independent"] and journal["ema_bitwise_epochs"] == journal["replayed_epochs"],
              "EMA state does not independently replay bitwise")
    records = [json.loads(line) for line in files["journal"].read_bytes().splitlines()]
    receipts = [record["routing"]["scope_receipt"] for record in records]
    h.require(all(receipt["phases"] == ["train", "eval"] and receipt["masked"]
                  and receipt["task_group"] == group and receipt["legal_rows"] == 784
                  and receipt["overlap_1based"] == members["patient_ids_1based"] for receipt in receipts),
              "Stored train/eval mask receipt differs")

    run.restore(ctx, best, manifest["implementation"])
    h.require(h.sha(run.bundle(ctx)) == best["state_sha"], "Best parameter or optimizer reload differs")
    test = ctx["members"]["test"]
    labels = ctx["state"]["labels_np"][test]
    masked = np.isin(test, members["patient_indices_0based"])
    h.require(int(masked.sum()) == 2, "The audit requires both targets in the test set")
    rng = h.rng_sha()
    outputs, captures, metrics = {}, {}, {}
    for mode, intervention in INTERVENTIONS.items():
        output, cap = proof.independent_eval(ctx, intervention)
        full_cam = captures.get("full", cap)["cam"]
        verify_target_mask(ctx, cap)
        h.require(torch.equal(cap["cam"], full_cam), "Fixed-weight intervention changed native CAM144")
        outputs[mode], captures[mode] = output, cap
        metrics[mode] = score(labels, output["prediction"], output["probability"][:, 0])
    proof.match_eval(outputs["full"], best["evaluation"], "Best independent evaluation")
    h.require(h.rng_sha() == rng, "Best evaluation changed RNG")
    for mode in ("residual_off", "anonymous"):
        h.require(all(np.array_equal(outputs["full"][k][masked], outputs[mode][k][masked])
                      for k in ("logits", "prediction", "probability")), "Target intervention predictions differ")
    full = captures["full"]
    cap_stats = {mode: proof.f.branch_formulas(ctx, captures[mode], intervention=intervention)
                 for mode, intervention in INTERVENTIONS.items()}
    residual = residual_records(captures, test)
    h.require((residual["cam_relative_delta"] <= CONFIG["training"]["cam_residual_cap"] + 2 * np.finfo(np.float32).eps).all()
              and all(np.count_nonzero(residual[k][~masked]) == 85 and not residual[k][masked].any()
                      for k in ("delta_l2", "delta_parallel_l2", "delta_perp_l2", "query_feature_change_l2")),
              "Best residual cap or patient-path scope differs")
    best_gradients = gradient_scope(run, ctx, pre["dropout"])
    fresh = run.setup(MODEL_VARIANT, FOLD)
    latest_check = proof.check_latest(fresh, latest, manifest, task, pre)
    h.require(latest_check["exact"], "Latest reload or next step differs")
    run.restore(fresh, latest, manifest["implementation"])
    current_ema = run.ema(fresh)
    run.restore_ema(fresh, latest["eval_ema"])
    latest_output, latest_cap = proof.independent_eval(fresh)
    verify_target_mask(fresh, latest_cap)
    proof.match_eval(latest_output, latest["evaluation"], "Latest independent dual-mask readback")
    run.restore_ema(fresh, current_ema)
    latest_gradients = gradient_scope(run, fresh, pre["dropout"])

    patient_records = []
    for i, subject in enumerate(test):
        rec = {"patient_id_1based": int(subject + 1), "true_label": int(labels[i])}
        for mode, out in outputs.items():
            rec.update({f"prediction_{mode}": int(out["prediction"][i]), f"p_asd_{mode}": float(out["probability"][i, 0]),
                        f"true_margin_{mode}": float(proof.signed_margin(out["logits"], labels)[i]),
                        f"p_td_{mode}": float(out["probability"][i, 1])})
            for key, source in (("delta_l2", "delta"), ("delta_parallel_l2", "delta_parallel"), ("delta_perp_l2", "delta_perp")):
                if mode != "full":
                    rec[f"{key}_{mode}"] = float(captures[mode]["details"][source][subject].double().norm())
        rec.update({key: float(values[i]) for key, values in residual.items()})
        patient_records.append(rec)

    coverage = {}
    for patient, pair in members["target_pairs"].items():
        phases = [part["patients"][patient] for part in (pre["gradients"], best_gradients, latest_gradients)]
        h.require(all(phase["relation_sampled_pairs"] == pair["endpoint_occurrences"] for phase in phases),
                  f"Patient {patient} relation counts differ across audit phases")
        confirmed = all(phase["relation_direct_bce_grad_l2"] > 0
                        and phase["relation_direct_weighted_grad_l2"] > 0 for phase in phases)
        coverage[patient] = "DIRECT_RELATION_SUPERVISION_CONFIRMED" if confirmed else "RELATION_SUPERVISION_COVERAGE_LIMITED"
    # Absence from the historical sampler is auditable when accurately recorded.
    # A sampled target lacking the expected gradient is recorded separately.
    limited = [patient for patient, verdict in coverage.items() if verdict == "RELATION_SUPERVISION_COVERAGE_LIMITED"]
    status = "COMPLETE_AUDITED" if not limited else "AUDITED_WITH_RELATION_COVERAGE_LIMITATION"
    return {
        "status": status, "completed_at_utc": utc_now(), "group": group, "fold": FOLD,
        "relation_coverage": coverage, "relation_coverage_limited_ids_1based": [int(i) for i in limited],
        "membership": members, "gaussian": gaussian, "best_reloaded": True, "latest_reloaded": True,
        "latest_next_training_exact": True, "latest_next_evaluation_exact": True, "latest_check": latest_check,
        "best_epoch": best["epoch"], "latest_epoch": latest["epoch"], "best_metrics": metrics,
        "best_gradients": best_gradients, "latest_gradients": latest_gradients,
        "best_branch_formulas": cap_stats, "journal": journal, "mask_train_eval_receipts": len(receipts),
        "target_cam144_exact_all_interventions": True, "target_logits_and_probabilities_exact_all_interventions": True,
        "latest_dual_mask_exact": True, "other_85_residuals_active": True,
        "max_residual_ratio": float(residual["cam_relative_delta"].max()),
        "artifact_sha256": {k: h.sha_file(files[k]) for k in ("best", "latest", "journal", "task", "preflight")},
        "patients": patient_records,
    }
