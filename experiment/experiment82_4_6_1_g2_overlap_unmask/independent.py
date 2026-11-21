"""Independent enrollment, Gaussian, gradient, checkpoint and prediction checks."""
from __future__ import annotations

import csv
import gc
import json

from overlap import (CFG, CONFIG, EXPOSED, FOLD, GROUP, PARENT, engine,
                     h, implementation, load_parent_module, optimization, original_manifest, parent,
                     parent_preflight, paths, rebuild_reference, scope, setup, spectral)
import forensic as f

import numpy as np
import torch
from torch.nn import functional as F

METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
SOURCE_METHOD = "82.4.6-G2 SCMR-DA-SRPE"


def untouched_scope(ctx):
    members = ctx["members"]
    held_out = members["test"][~np.isin(members["test"], EXPOSED)]
    h.require(len(held_out) == 85 and len(members["test"]) == 87 and len(members["legal"]) == 786
              and np.array_equal(np.intersect1d(members["legal"], members["test"]), EXPOSED)
              and not np.intersect1d(members["legal"], held_out).size,
              "Other held-out patients entered new labels or statistics")
    return {**ctx, "members": {**members, "test": held_out}}


def gaussian_proof(ctx):
    scoped = untouched_scope(ctx)
    ref, m = ctx["reference"], ctx["members"]
    y = ctx["state"]["labels_np"]
    h.require(np.array_equal(ref["legal"], m["train"]) and np.array_equal(ref["labels"], y[ref["legal"]]),
              "Gaussian reference membership or labels differ")
    proof = f.class_banks(scoped)
    check, oof = {}, {}
    for row in EXPOSED:
        j, = np.flatnonzero(ref["legal"] == row)
        assigned = int(ref["group"][j])
        h.require(bool(ref["mask"][0, y[row], j]) and not ref["mask"][assigned, :, j].any(),
                  "An exposed row is absent from inference or present in its own training statistics")
        oof[str(row + 1)] = {"label": int(y[row]), "inner_bank": assigned,
                              "included_in_inference_bank": True, "absent_from_own_training_bank": True}
    before = h.rng_sha()
    replay = rebuild_reference(FOLD, ctx["warm_cam"], ctx["original"], ref["legal"], ref["labels"])
    old = ctx["old_members"]["legal"]
    old_replay = rebuild_reference(FOLD, ctx["warm_cam"], ctx["original"], old, y[old].copy())
    h.require(h.rng_sha() == before and h.sha(replay) == h.sha(ref) and
              h.sha(old_replay) == ctx["original_provenance"]["reference"],
              "Gaussian reconstruction consumed RNG or did not reproduce historical/statistical math")
    check.update(proof, oof_exposed=oof, deterministic_exact=True, original_kernel_exact=True,
                 reference_sha=h.sha(ref), stratified_seed=ref["seed"], numpy_main_rng_unchanged=True)
    return check


def label_isolation(run, ctx, snapshot):
    state = ctx["state"]
    unexposed = untouched_scope(ctx)["members"]["test"]
    rows = torch.as_tensor(unexposed, device=state["y"].device)
    values = state["y"][rows].clone()
    np_values = state["labels_np"][unexposed].copy()

    def probe():
        h.base().restore_rng(snapshot["rng"])
        _, cap = run.forward(state, True, ctx["module"])
        d = cap["details"]
        b = ctx["module"].spectrum.basis.to(cap["cam"])
        loss, _ = optimization.losses(GROUP, state["cam"].Outputlayer, cap, state["legal"],
                                       ctx["labels"], b, ctx["hard_initial"])
        for term, value in loss.items():
            g, = torch.autograd.grad(value, cap["cam"], retain_graph=True)
            h.require(not g[rows].count_nonzero().item(), f"{term} has nonmember supervision")
        obs = ctx["module"].statistics.observation(cap["cam"])
        decision = [d[key] for key in ("delta", "target_margin", "llr", "reliability", "disagreement")]
        return h.sha((decision, loss, obs))

    before = probe()
    try:
        state["y"][rows] = 1 - values
        state["labels_np"][unexposed] = 1 - np_values
        h.require(probe() == before, "A remaining held-out label affected the new path or statistics")
    finally:
        state["y"][rows] = values
        state["labels_np"][unexposed] = np_values
        h.base().restore_rng(snapshot["rng"])
    return {"remaining_test_label_inversion_exact": True, "remaining_test_cam_gradients_zero": True,
            "excluded_from_new_supervision_and_banks": len(rows)}


def preflight(run, hashes):
    device = h.archive.verify_device()
    h.require(device["index"] == CONFIG["physical_gpu_index"] and device["uuid"] == CONFIG["physical_gpu_uuid"],
              "Device is not the verified physical GPU0")
    h.hist_config()
    original = original_manifest()
    other_folds = {}
    for fold in range(1, CFG["folds"] + 1):
        if fold == FOLD:
            continue
        c = parent.setup(GROUP, fold)
        archived = h.read_json(parent.paths(GROUP, fold)["preflight"])
        h.require(c["provenance"] == archived["provenance"], f"Unchanged fold {fold} statistics or identity differs")
        other_folds[str(fold)] = {k: c["provenance"][k] for k in ("reference", "membership", "legal_index_sha")}
        del c
    ctx = setup()
    state = ctx["state"]
    baseline = h.read_json(parent.paths(GROUP, FOLD)["preflight"])
    snapshot = parent_preflight.restored_payload(ctx, hashes, epoch=10, wait=0, best_epoch=0, best_correct=-1)
    old = parent_preflight.historic_signature(ctx, wrapped=False)
    h.require(old[0] == baseline["disabled_native_exact"] and old[1] == baseline["dropout"] and
              len(old[1]) == 6 and old[2] == h.hist_trace(FOLD)[11], "Direct native parity fails")
    run.restore(ctx, snapshot, hashes)
    h.require(parent_preflight.historic_signature(ctx, wrapped=True, active=True) == old,
              "Enabling the extended path changed historical native behavior")
    run.restore(ctx, snapshot, hashes)
    banks = gaussian_proof(ctx)
    difficulty = f.legal_memory(untouched_scope(ctx))
    _, initial = run.evaluate(ctx)
    d = initial["details"]
    f.equal(d["delta"], torch.zeros_like(d["delta"]), "Historical SCMR zero initialization")
    f.equal(d["target_margin"], d["margin"], "Zero initialized target shift")
    h.require(ctx["module"].spectrum.allowed.all().item(), "Expected full residual allowance for Fold 2")
    geometry = f.branch_formulas(ctx, initial)
    result, diag, rng, _, _ = run.transition(ctx, 11, old[1], inspect=True)
    h.require(diag["losses"]["difference"] == 0 and diag["losses"]["full"] == diag["losses"]["off"] and
              h.sha(h.base_bundle(state)) == old[0]["after_update"] and rng == old[2] and
              all(diag["budget"][family]["raw"] == diag["budget"][family]["total_applied"] == 0
                  for family in optimization.ORIGINAL), "Zero-init difference did not preserve the original Adam update")
    first = h.sha((result, diag, rng, h.sha(run.ema(ctx))))
    saved = parent_preflight.restored_payload(ctx, hashes, epoch=11, wait=0, best_epoch=11, best_correct=result["correct"])
    witness = run.next_witness(ctx, 12, old[1])
    run.restore(ctx, saved, hashes)
    h.require(witness == run.next_witness(ctx, 12, old[1]) and witness["rng"] == h.hist_trace(FOLD)[12],
              "Independent serialized next training and eval witness differs")
    mature = parent_preflight.restored_payload(ctx, hashes, epoch=12)
    isolate = label_isolation(run, ctx, mature)
    run.restore(ctx, mature, hashes)
    proof = proof_engine(run)
    warm_grad = proof.twin_probe(ctx, old[1], check_each=True)
    run.restore(ctx, mature, hashes)
    _, cap = run.evaluate(ctx)
    amplitude = f.amplitude_probe(ctx, cap["cam"])
    f.equal(cap["logits"], spectral.native_logits(state["cam"].Outputlayer, cap["enhanced"]),
            "Actual deterministic OutputLayer replay")
    with torch.no_grad():
        bypass, orig_cap = run.forward(state, False)
        off, off_cap = run.evaluate(ctx, "residual_off")
        f.equal(off_cap["enhanced"], orig_cap["cam"], "Residual-off native CAM")
        f.equal(off["probability"], bypass["logp"][state["test"]].exp(), "Residual-off native probabilities")
    return {"status": "PASS", "implementation": hashes, "provenance": ctx["provenance"],
            "device": device, "dropout": old[1], "scope": scope(ctx), "other_folds": other_folds,
            "native_disabled_exact": old[0], "initial_ema": h.sha(snapshot["ema"]),
            "first_step_sha": first, "first_step_diagnostics": diag, "next_step_exact": witness,
            "label_isolation": isolate, "banks": banks, "difficulty": difficulty,
            "initial_formulas": geometry, "amplitude": amplitude, "nonzero_gradient_audit": warm_grad,
            "parent_frozen_sha": original["implementation_sha"]}


def proof_engine(run):
    proof = load_parent_module("_overlap_checkpoint_audit", "audit.py")
    proof.run = run
    check_gradients = proof.twin_probe

    def overlap_gradients(ctx, expected_dropout, *, check_each=False):
        # Use the true held-out membership for the inherited zero-direct-gradient
        # assertion only. Evaluation and the stored fold still use all 87 rows.
        state = ctx["state"]
        held_out = untouched_scope(ctx)["members"]["test"]
        rows = state["test"][~np.isin(state["test"], ctx["members"]["legal"])]
        f.equal(rows, held_out, "Remaining held-out test order")
        return check_gradients({**ctx, "state": {**state, "test": rows}}, expected_dropout,
                               check_each=check_each)

    proof.twin_probe = overlap_gradients
    return proof


def metric_values(prob, pred, truth):
    """An independent confusion and pairwise ROC-AUC calculation."""
    h.require(prob.shape == (len(truth), 2) and np.isfinite(prob).all() and
              np.array_equal(pred, prob.argmax(-1)), "Unexpected prediction or probability")
    y = np.asarray(truth)
    p = prob[:, 0]
    n, k = len(y), int((y == 0).sum())
    a, t = y == 0, y == 1
    h.require(0 < k < n, "Both ASD and TD must be present in this audit scope")
    tp, fp = int((a & (pred == 0)).sum()), int((t & (pred == 0)).sum())
    tn, fn = n - k - fp, k - tp
    sen, spe = tp / k, tn / (n - k)
    comparisons = p[a, None] - p[None, t]
    rank = (2 * np.count_nonzero(comparisons > 0) + np.count_nonzero(comparisons == 0)) / (
        2 * k * (n - k))
    f1_denom = 2 * tp + fp + fn
    return {"n_test": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "acc": (tp + tn) / n,
            "hard_auc": (sen + spe) / 2, "prob_auc": rank, "sen": sen, "spe": spe,
            "asd_f1": 2 * tp / f1_denom if f1_denom else 0.}


def parent_certificates(proof, historic, guard):
    prior_status = h.read_json(PARENT / "results/audit_status.json")
    h.require(prior_status["status"] == "COMPLETE_AUDITED" and prior_status["best_reloaded"] == CFG["folds"] * 2,
              "Parent result audit is incomplete")
    with (PARENT / "results/fold_metrics.csv").open(newline="") as stream:
        reference_scores = {int(r["fold"]): r for r in csv.DictReader(stream) if r["method"] == SOURCE_METHOD}
    with (PARENT / "results/patient_margins.csv").open(newline="") as stream:
        archived_rows = {(int(r["fold"]), int(r["patient_row_1based"])): r for r in csv.DictReader(stream)
                         if r["group"] == GROUP and r["intervention"] == "full"}
    bests = {r["fold"]: r for r in historic["completed_folds"][GROUP]}
    result = {}
    for fold, identity in sorted(bests.items()):
        files = parent.paths(GROUP, fold)
        checkpoint, certificate = files["best"], files["audit"]
        for path in (checkpoint, certificate):
            h.require(h.sha_file(path) == guard[str(path)], "Historical Best or payload digest mismatch")
        h.require(identity["best_sha"] == guard[str(checkpoint)], "Historical Best is not the parent commitment")
        saved, source = h.load(checkpoint), h.read_json(certificate)
        ctx = parent.setup(GROUP, fold)
        indices = ctx["members"]["test"] + 1
        y = ctx["state"]["labels_np"][ctx["members"]["test"]]
        h.require(source["status"] == "PASS" and source["best_full_reload"] is True and
                  source["best_sha"] == identity["best_sha"], "Historical fold was not certified")
        f.equal(y, source["label"], "Read-only historical test labels")
        f.equal(indices, source["patient_rows_1based"], "Read-only historical test order")
        h.require(ctx["provenance"] == saved["provenance"], "Historical initialization differs")
        parent.restore(ctx, saved, historic["implementation"])
        now, capture = proof.independent_eval(ctx)
        off, _ = proof.independent_eval(ctx, "residual_off")
        proof.match_eval(now, saved["evaluation"], f"Original G2 Fold {fold} Best")
        f.equal(now["probability"][:, 0], source["interventions"]["full"]["probability"], "Historical ASD probabilities")
        f.equal(now["prediction"], source["interventions"]["full"]["prediction"], "Historical argmax")
        f.equal(off["probability"][:, 0], source["interventions"]["residual_off"]["probability"], "Historical Off probabilities")
        diag = capture["details"]
        if fold == FOLD:
            h.require(not diag["delta"][list(EXPOSED)].count_nonzero().item(), "Original pair was not masked")
        fm = metric_values(now["probability"], now["prediction"], y)
        for m in METRICS:
            h.require(abs(fm[m] - float(reference_scores[fold][m])) <= np.finfo(np.float64).eps,
                      f"Historical independently recalculated {m} disagrees")
        for i, index in enumerate(indices):
            archived = archived_rows[fold, int(index)]
            h.require(int(archived["label"]) == y[i] and int(archived["pred_full"]) == now["prediction"][i] and
                      float(archived["p_asd_full"]) == now["probability"][i, 0],
                      "Archived patient export differs from its Best")
        result[fold] = {"full": now, "off": off, "subjects": indices, "truth": y.copy(),
                        "best_epoch": saved["epoch"], "metrics": fm, "best_sha": identity["best_sha"],
                        "certificate_sha": guard[str(certificate)], "certificate_path": str(certificate),
                        "best_path": str(checkpoint), "inference_sha": h.sha(now),
                        "reference_sha": ctx["provenance"]["reference"],
                        "partition_sha": ctx["provenance"]["membership"]}
        del ctx, capture
    h.require(len(result) == 10 and len(archived_rows) == CFG["subjects"], "The complete ten-fold archive is required")
    return result


def actual_patient_gradients(run, ctx, expected_dropout):
    """Independently differentiate each patient's contribution to the production Full CE."""
    state = ctx["state"]
    with run.dropout_inventory() as calls:
        _, capture = run.forward(state, True, ctx["module"])
    h.require(calls == expected_dropout, "The patient gradient probe changed Dropout sequence")
    d = capture["details"]
    chosen = ctx["members"]["legal"]
    y = ctx["labels"]
    head = state["cam"].Outputlayer
    ln, linear, gelu, _ = head.mlp_head
    all_logits = head.classifier(gelu(linear(ln(capture["enhanced"][state["legal"]]))))
    values = -F.log_softmax(all_logits, dim=-1).gather(1, y[:, None]).flatten()
    loss, _ = optimization.losses(GROUP, head, capture, state["legal"], y,
                                   ctx["module"].spectrum.basis.to(capture["cam"]), ctx["hard_initial"])
    original_d, = torch.autograd.grad(loss["full"], d["delta"], retain_graph=True)
    dims = ("prompt", "control")
    groups = ctx["families"]
    params = [p for k in dims for p in groups[k]]
    result = {}
    for idx in EXPOSED:
        j, = np.flatnonzero(chosen == idx)
        class_count = int((y == y[j]).sum())
        single = values[j] / (2 * class_count)
        derivatives = torch.autograd.grad(single, (d["delta"], *params, d["raw_parallel"], d["raw_perp"]),
                                          retain_graph=True, allow_unused=True)
        derivative = derivatives[0]
        f.bounded(derivative[idx], f.f64(original_d[idx]), 30,
                  "Individual loss CAM derivative versus the original class-balanced Full CE")
        h.require(derivative.count_nonzero().item() == derivative[idx].count_nonzero().item(),
                  "The isolated patient direct CAM derivative touches another row")
        start = len(groups["prompt"]) + 1
        norms = {"full_ce_delta_l2": float(derivative[idx].norm()), "loss_contribution": float(single.detach()),
                 "prompt_l2": optimization.norm(derivatives[1:start]),
                 "control_l2": optimization.norm(derivatives[start:-2]),
                 "spectral_candidate_l2": float(derivatives[-2][idx].norm()),
                 "complement_candidate_l2": float(derivatives[-1][idx].norm())}
        h.require(all(np.isfinite(list(norms.values()))) and
                  all(norms[k] > 0 for k in ("control_l2", "full_ce_delta_l2", "spectral_candidate_l2",
                                               "complement_candidate_l2")),
                  f"The exposed patient {idx + 1} has no actual Full CE route")
        result[str(idx + 1)] = norms
    return result


def rng_against_parent():
    original_file = parent.paths(GROUP, FOLD)["journal"]
    a = {x["epoch"]: x["rng"] for x in (json.loads(line) for line in original_file.read_bytes().splitlines())}
    b = {x["epoch"]: x["rng"] for x in (json.loads(line) for line in paths()["journal"].read_bytes().splitlines())}
    intersection = a.keys() & b.keys()
    h.require(intersection and all(a[x] == b[x] for x in intersection), "Shared parent G2 RNG trace differs")
    return {"common_epochs": len(intersection), "identical_rng_phases": 3 * len(intersection)}


def audit(manifest):
    run, files = engine(), paths()
    proof = proof_engine(run)
    hashes = manifest["implementation"]
    h.require(hashes == implementation(), "Current source freeze differs")
    parent_frozen = original_manifest()
    prev = h.read_json(files["preflight"])
    task = h.read_json(files["task"])
    h.require(prev["status"] == "PASS" and task["status"] == "TRAINED" and
              prev["implementation"] == task["implementation"] == hashes, "Training certificate or preflight missing")
    best, latest = (h.load(files[k]) for k in ("best", "latest"))
    h.require(all(h.sha_file(files[k]) == task[k + "_sha"] for k in ("best", "latest", "journal")) and
              latest["best_sha"] == task["best_sha"], "Committed Fold 2 checkpoint or journal differs")
    ctx = setup()
    h.require(ctx["provenance"] == prev["provenance"], "Independent initial identity differs")
    banks = gaussian_proof(ctx)
    record = proof.audit_journal(ctx, task, best, latest, prev)
    h.require(record["ema_independent"] and record["ema_bitwise_epochs"] == record["replayed_epochs"],
              "Independent EMA trajectory differs")
    original_rng = rng_against_parent()
    first, tenth = (record[x]["paths"]["exposed_patient_gradients"] for x in ("first", "tenth"))
    h.require(all(x["control_l2"] > 0 for x in first.values()) and
              all(x["control_l2"] > 0 and x["prompt_l2"] > 0 for x in tenth.values()),
              "Recorded production patient gradients do not reach the new branch")
    run.restore(ctx, best, hashes)
    h.require(h.sha(run.bundle(ctx)) == best["state_sha"], "Best optimizer or parameters did not reload")
    full, cap = proof.independent_eval(ctx)
    off, off_cap = proof.independent_eval(ctx, "residual_off")
    proof.match_eval(full, best["evaluation"], "Overlap Best")
    f.equal(off_cap["enhanced"], cap["cam"], "Residual-off leaves original CAM intact")
    f.equal(off_cap["cam"], cap["cam"], "Shared CAM behind both Blend interventions")
    form = f.branch_formulas(ctx, cap)
    form_off = f.branch_formulas(ctx, off_cap, intervention="residual_off")
    d = cap["details"]
    t = ctx["state"]["test"]
    patients = {}
    for index in EXPOSED:
        key = str(index + 1)
        h.require(ctx["module"].spectrum.allowed[index].item(), "Exposed residual still masked")
        patients[key] = {name + "_l2": float(d[name][index].norm()) for name in ("delta", "delta_parallel", "delta_perp")}
    chosen = ctx["members"]["test"] + 1
    labels = ctx["state"]["labels_np"][ctx["members"]["test"]].copy()
    new = {"full": full, "off": off, "subjects": chosen, "truth": labels, "best_epoch": best["epoch"],
           "best_sha": task["best_sha"], "best_path": str(files["best"]),
           "metrics": metric_values(full["probability"], full["prediction"], labels),
           "residual": {k: f.f64(d[k][t]).tolist() for k in ("ratio",)},
           "inference_sha": h.sha(full), "reference_sha": ctx["provenance"]["reference"]}
    for name in ("delta", "delta_parallel", "delta_perp"):
        new["residual"][name + "_l2"] = np.linalg.norm(f.f64(d[name][t]), axis=1).tolist()
    h.require(np.all(np.asarray(new["residual"]["ratio"]) <= CFG["residual"]["fraction"] + 2 * np.finfo(np.float32).eps),
              "Excessive G2 residual")
    best_gradients = actual_patient_gradients(run, ctx, prev["dropout"])
    run.restore(ctx, best, hashes)
    best_probe = proof.twin_probe(ctx, prev["dropout"], check_each=best["epoch"] >= run.START + 2)
    del ctx, cap, off_cap
    gc.collect()
    current = setup()
    latest_result = proof.check_latest(current, latest, manifest, task, prev)
    h.require(latest_result["exact"], "Fresh Latest exact restoration failed")
    run.restore(current, latest, hashes)
    latest_gradients = actual_patient_gradients(run, current, prev["dropout"])
    del current
    historic = parent_certificates(proof, parent_frozen, manifest["protected_sha256"])
    for fold, item in historic.items():
        if fold != FOLD:
            h.require(item["reference_sha"] == prev["other_folds"][str(fold)]["reference"],
                      "Untouched fold reference differs before and after")
    result = {"status": "PASS", "best_reloaded": 1, "latest_reloaded": 1, "next_training_eval_exact": 1,
              "trained_folds": [FOLD], "unchanged_other_folds": sorted(set(historic) - {FOLD}),
              "scope": prev["scope"], "best_epoch": best["epoch"], "end_epoch": latest["epoch"],
              "rng": {**original_rng, "darg_common_epochs": record["archived_epochs"],
                      "darg_identical_phases": record["rng_phase_comparisons"], "replayed_epochs": record["replayed_epochs"]},
              "ema_exact_epochs": record["ema_bitwise_epochs"], "ledger": record,
              "warmup_and_initial": prev, "banks": banks, "geometry_full": form, "geometry_off": form_off,
              "best_training_probe": best_probe, "latest_fresh_object": latest_result,
              "affected_patients": {key: {**p, "best_supervision_gradient": best_gradients[key],
                                         "latest_supervision_gradient": latest_gradients[key],
                                         "first_step_gradient": first[key], "tenth_step_gradient": tenth[key]}
                                    for key, p in patients.items()}}
    return historic, new, result
