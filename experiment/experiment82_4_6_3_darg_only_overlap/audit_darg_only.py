"""Replay memberships, attribution boundaries, checkpoints and archived predictions."""
from __future__ import annotations

import csv
import json

from darg_only import (
    CFG, CONFIG, EXPOSED, FOLDS, FULL_PATH, GROUP, MASKED_COMPONENTS, engine, ext, f, full_audit, h,
    implementation, optimization, parent, paths, prior, source_manifests, spectral,
)

import numpy as np
import torch
from torch.nn import functional as F

METRICS, score = full_audit.METRICS, full_audit.score


def proof_engine(run):
    proof = prior.load_parent_module("_darg_only_checkpoint_audit", "audit.py")
    proof.run = run
    return proof


def scope_proof(ctx):
    fold, new, old, state = (ctx[k] for k in ("fold", "members", "old_members", "state"))
    overlap = np.asarray(EXPOSED[fold])
    legal = new["legal"]
    f.equal(np.intersect1d(new["train"], new["test"]), overlap, "Only the registered labelled overlaps")
    f.equal(np.setdiff1d(new["train"], legal), overlap, "DARG-only membership difference")
    f.equal(new["train"][:len(old["train"])], old["train"], "Ordered original training prefix")
    f.equal(new["train"][len(old["train"]):], ext.ADDED[fold], "Ordered prospective training additions")
    f.equal(legal, old["legal"], "Canonical original SCMR legal membership")
    f.equal(new["test"], old["test"], "Unchanged outer test indices")
    f.equal(np.flatnonzero(~f.array(ctx["module"].spectrum.allowed)), overlap, "Only overlap residuals disabled")
    h.require(len(new["train"]) == len(np.unique(new["train"])) == CONFIG["native_rows"]
              and len(legal) == len(np.unique(legal)) == CONFIG["new_supervision_rows"]
              and len(new["test"]) == CONFIG["test_rows"] and not np.intersect1d(legal, new["test"]).size,
              "Unexpected fold sizes or test label in the legal SCMR rows")
    for key in ("train", "legal", "test", "weights", "train_pairs", "train_same"):
        f.equal(state[key], new[key], f"Actual device {key}")
    y = state["labels_np"]
    f.equal(ctx["labels"], y[legal], "Correctly ordered legal labels")
    f.equal(state["y"], y, "Fixed device labels")
    before = h.rng_sha()
    anchors = {}
    for name, partition in (("original", old), ("extended", new)):
        pairs, same, counts = full_audit.rederive_pairs(partition["train"], y, state["supports"], 532100 + fold)
        f.equal(pairs, partition["train_pairs"], f"{name} native relation ordering")
        f.equal(same, partition["train_same"], f"{name} native relation classes")
        f.equal(np.bincount(same, minlength=2), [2048, 2048], "Balanced 4096 training pairs")
        f.equal(partition["weights"], (len(partition["train"]) /
                                     (2 * np.bincount(y[partition["train"]], minlength=2))).astype(np.float32),
                "Correct ASD/TD class weights")
        h.require(np.isin(pairs, partition["train"]).all(), "Native relation endpoints outside training")
        anchors[name] = counts
    h.require(h.rng_sha() == before, "Independent pair reconstruction changed historical RNG")
    registration = ctx["provenance"]["registration"]
    h.require(all(v > 0 for v in registration["per_overlap_pair_count"].values()),
              "An overlap does not actually participate in the native relation loss")
    return {**registration, "labels_and_test_indices_exact": True, "ordered_pairs_exact": True,
            "anchored_pairs_same_and_different": anchors, "held_out_test": 82, "scope": "DARG-only"}


def gaussian_proof(ctx):
    ref, state, module = (ctx[k] for k in ("reference", "state", "module"))
    legal = ctx["members"]["legal"]
    f.equal(ref["legal"], legal, "Exact original 784 Gaussian rows")
    f.equal(module.statistics.legal, legal, "Actual Gaussian and EMA device rows")
    f.equal(module.statistics.mask, ref["mask"], "Production OOF and inference masks")
    f.equal(ref["labels"], state["labels_np"][legal], "Gaussian training labels")
    f.equal(ref["bank"][ctx["members"]["test"]], np.zeros(CONFIG["test_rows"]), "Outer test uses inference bank")
    independent = f.class_banks(ctx)
    rng = h.rng_sha()
    rebuilt = prior.rebuild_reference(ctx["fold"], ctx["warm_cam"], ctx["original"], legal, ref["labels"].copy())
    canonical_sha = h.read_json(parent.paths(GROUP, ctx["fold"])["preflight"])["provenance"]["reference"]
    h.require(h.sha(ref) == h.sha(rebuilt) == canonical_sha and h.rng_sha() == rng,
              "Gaussian reference is not bitwise identical to the original SCMR fold")
    return {**independent, "reference_sha": canonical_sha, "exact_rebuild": True, "legal_rows": len(legal),
            "oof_self_excluded_rows": len(legal), "overlap_rows_in_statistics": 0,
            "inference_and_ema_legal_index_sha": h.sha(legal), "seed": ref["seed"]}


def mask_proof(ctx, full, off, cap, off_cap):
    fold = ctx["fold"]
    rows = list(EXPOSED[fold])
    selected = np.isin(ctx["members"]["test"], rows)
    h.require(selected.sum() == len(rows), "Missing masked test record")
    for key in MASKED_COMPONENTS:
        observed = f.array(cap["details"][key])[rows]
        f.equal(observed, np.zeros_like(observed), f"Exactly zero overlap {key}")
    f.equal(cap["cam"], off_cap["cam"], "Same original CAM under the fixed-weight counterfactual")
    f.equal(cap["enhanced"][rows], cap["cam"][rows], "Masked CAM144 exact equality")
    f.equal(off_cap["enhanced"], cap["cam"], "Residual-off CAM144 equality")
    for key in ("logits", "probability", "prediction"):
        f.equal(full[key][selected], off[key][selected], f"Fixed-weight overlap {key} equals Residual-off")
    result = {"masked_rows": len(rows), "cam_logits_probability_prediction_exact": True}
    for key in ("delta_parallel", "delta_perp", "delta"):
        value = f.f64(cap["details"][key])[ctx["members"]["test"][~selected]]
        lengths = np.linalg.norm(value, axis=1)
        h.require(np.isfinite(lengths).all() and np.any(lengths > 0), f"Unmasked {key} is inactive")
        result[key + "_active_held_out"] = int(np.count_nonzero(lengths))
        result[key + "_mean_l2_held_out"] = float(lengths.mean())
    return result


def scope_gradient(run, ctx, dropout):
    state, module = ctx["state"], ctx["module"]
    rows = list(EXPOSED[ctx["fold"]])
    with run.dropout_inventory() as calls:
        output, cap = run.forward(state, True, module)
    h.require(calls == dropout, "Scope gradient audit altered Dropout order")
    rng = h.rng_sha()
    native = F.nll_loss(output["logp"][state["train"]], state["y"][state["train"]], state["weights"])
    f.equal(native, output["cls"], "Actual native classification loss uses all 789 labels")
    logp, = torch.autograd.grad(output["cls"], output["logp"], retain_graph=True)
    expected = -state["weights"][state["y"][rows]] / state["weights"][state["y"][state["train"]]].sum()
    f.bounded(logp[rows, state["y"][rows]], expected, 2 * len(state["train"]),
              "Native NLL gradient follows the specified labels", np.abs(f.f64(expected)))
    pair_logits = tuple(output["info"]["pair_logits"])
    pair_basis = torch.stack([F.binary_cross_entropy_with_logits(logit, state["train_same"])
                              for logit in pair_logits]).mean()
    f.equal(pair_basis, output["pair"], "Actual native relation supervision uses ordered pair labels")
    pair_grad = torch.autograd.grad(output["pair"], pair_logits, retain_graph=True)
    auxiliary, _ = optimization.losses(GROUP, state["cam"].Outputlayer, cap, state["legal"], ctx["labels"],
                                       module.spectrum.basis.to(cap["cam"]), ctx["hard_initial"])
    gradients = {}
    for name, term in auxiliary.items():
        grad, = torch.autograd.grad(term, cap["cam"], retain_graph=True, allow_unused=True)
        if grad is not None:
            f.equal(grad[rows], torch.zeros_like(grad[rows]), f"No direct SCMR {name} gradient on overlap CAM")
        gradients[name] = 0.0
    delta_grad, = torch.autograd.grad(auxiliary["full"], cap["details"]["delta"], retain_graph=True)
    f.equal(delta_grad[rows], torch.zeros_like(delta_grad[rows]), "No Full CE gradient on overlap residuals")
    h.require(delta_grad[state["legal"]].count_nonzero().item() > 0, "Full CE is inactive on legal patients")
    observation = module.statistics.observation(cap["cam"])
    independent = f.independent_observation(ctx, cap["cam"])
    observation_check = f.bounded(observation, independent, 4 * len(state["legal"]),
                                  "EMA observation from 784 rows", 1 + abs(independent), f.U64)
    changed = cap["cam"].detach().clone()
    changed[rows] += 17
    f.equal(module.statistics.observation(changed), observation, "Masked features cannot contribute to EMA observation")
    patient = {}
    for row in rows:
        select = (state["train_pairs"] == row).any(dim=1)
        g = float(logp[row].norm())
        p = sum(float(part[select].norm()) for part in pair_grad)
        h.require(np.isfinite([g, p]).all() and g > 0 and p > 0, "Masked patient's native label is inactive")
        patient[str(row + 1)] = {"native_nll_logp_l2": g, "native_relation_gradient_l2_sum": p,
                                 "full_ce_delta_l2": float(delta_grad[row].norm())}
    h.require(h.rng_sha() == rng, "Scope loss/gradient diagnostics advanced historical RNG")
    return {"patients": patient, "overlap_cam_gradient_l2_by_auxiliary": gradients,
            "ema_observation": observation_check, "changing_overlap_cam_at_fixed_inputs_does_not_change_ema": True}


def initial_parity(ctx):
    fold = ctx["fold"]
    point = ext.setup(GROUP, fold)
    full_record = h.read_json(ext.paths(GROUP, fold)["preflight"])
    h.require(point["provenance"] == full_record["provenance"], "Recreated E82.4.6.2 initialization differs")
    native, extended = ctx["state"], point["state"]
    for key in ("train", "test", "weights", "train_pairs", "train_same"):
        f.equal(native[key], extended[key], "Same native " + key + " as E82.4.6.2")
    now = ext.identity_snapshot(native, ctx["module"], ctx["optimizer"])
    then = ext.identity_snapshot(extended, point["module"], point["optimizer"])
    h.require(all(now[k] == v for k, v in then.items() if k != "spectrum_and_prompts")
              and now["historical_rng"] == ctx["warmup"]["rng_sha"]
              and now["optimizer_states"] == then["optimizer_states"], "Initialization signature differs")
    for name, value in ctx["module"].state_dict().items():
        if name != "spectrum.allowed" and not name.startswith("statistics."):
            f.equal(value, point["module"].state_dict()[name], "Unchanged trainable parameter or buffer " + name)
    return {**{k: v for k, v in now.items() if k != "spectrum_and_prompts"},
            "darg_and_scmr_initial_parameters_optimizers_rng_exact": True, "native_membership_exact": True}


def preflight_fold(fold, hashes):
    run = engine(fold)
    proof = proof_engine(run)
    device = h.archive.verify_device()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ctx = run.setup(GROUP, fold, verify_spectrum=True)
    parity = initial_parity(ctx)
    membership, banks, difficulty = scope_proof(ctx), gaussian_proof(ctx), f.legal_memory(ctx)
    pre = ext.parent_preflight
    snap = pre.restored_payload(ctx, hashes, epoch=10, wait=0, best_epoch=0, best_correct=-1)
    _, start = run.evaluate(ctx)
    f.equal(start["details"]["delta"], torch.zeros_like(start["cam"]), "Zero original SCMR initialization")
    geometry = f.branch_formulas(ctx, start)
    direct, dropout, initial_rng = pre.historic_signature(ctx, wrapped=False)
    old = h.read_json(ext.paths(GROUP, fold)["preflight"])
    h.require(direct == old["extended_native_signature"] and len(dropout) == 6 and dropout == old["dropout"]
              and initial_rng == h.hist_trace(fold)[11] == old["first_rng"], "Native first-step forward/gradient/update differs")
    run.restore(ctx, snap, hashes)
    out, diag, rng, _, _ = run.transition(ctx, 11, dropout, inspect=True)
    h.require(diag["losses"]["full"] == diag["losses"]["off"] and diag["losses"]["difference"] == 0
              and h.sha(h.base_bundle(ctx["state"])) == direct["after_update"] and rng == initial_rng
              and all(diag["budget"][key]["raw"] == diag["budget"][key]["total_applied"] == 0
                      for key in optimization.ORIGINAL), "Zero-residual historical continuity failed")
    step = h.sha((out, diag, rng, h.sha(run.ema(ctx))))
    saved = pre.restored_payload(ctx, hashes, epoch=11)
    witness = run.next_witness(ctx, 12, dropout)
    run.restore(ctx, saved, hashes)
    h.require(witness == run.next_witness(ctx, 12, dropout) and witness["rng"] == h.hist_trace(fold)[12],
              "Next training and evaluation are not exactly recoverable")
    ready = pre.restored_payload(ctx, hashes, epoch=12)
    now, cap = proof.independent_eval(ctx)
    off, off_cap = proof.independent_eval(ctx, "residual_off")
    mask = mask_proof(ctx, now, off, cap, off_cap)
    routes = scope_gradient(run, ctx, dropout)
    run.restore(ctx, ready, hashes)
    budget = proof.twin_probe(ctx, dropout, check_each=True)
    return {"status": "PASS", "fold": fold, "device": device, "implementation": hashes,
            "provenance": ctx["provenance"], "dropout": dropout, "membership": membership, "gaussian": banks,
            "difficulty": difficulty, "initial_parity": parity, "native_first_step": direct, "initial_geometry": geometry,
            "scope_gradient": routes, "actual_shared_and_new_update": budget, "mature_masks": mask,
            "initial_ema": snap["ema_sha"], "first_rng": rng, "first_step_sha": step, "next_step": witness,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}


def audit_fold(fold, manifest):
    run, files = engine(fold), paths(GROUP, fold)
    proof = proof_engine(run)
    hashes = manifest["implementation"]
    preflight, task = (h.read_json(files[key]) for key in ("preflight", "task"))
    h.require(preflight["implementation"] == task["implementation"] == hashes
              and h.sha_file(files["preflight"]) == manifest["preflight"]["folds"][str(fold)]["sha"]
              and all(h.sha_file(files[key]) == task[key + "_sha"] for key in ("best", "latest", "journal")),
              "Replayed fold bytes differ from their commitments")
    best, latest = (h.load(files[key]) for key in ("best", "latest"))
    h.require(latest["best_sha"] == task["best_sha"], "Latest identifies a different Best")
    ctx = run.setup(GROUP, fold)
    parity, membership, banks = initial_parity(ctx), scope_proof(ctx), gaussian_proof(ctx)
    journal = proof.audit_journal(ctx, task, best, latest, preflight)
    h.require(journal["ema_independent"] and journal["ema_bitwise_epochs"] == journal["replayed_epochs"],
              "One or more EMA updates differ in independent replay")
    records = [json.loads(row) for row in files["journal"].read_bytes().splitlines()]
    for record in records:
        mask = record["routing"]["strict_overlap_mask"]
        h.require(mask == {"phases": ["train", "eval"], "excluded_0based": list(EXPOSED[fold]),
                           "components": list(MASKED_COMPONENTS), "cam_exact": True}, "A training mask receipt changed")
    trace = {r["epoch"]: r["rng"] for r in records}
    rng = {"DARG": {"common_epochs": journal["archived_epochs"], "exact_phases": journal["rng_phase_comparisons"]},
           "E82.4.6-G2": full_audit.archived_rng_matches(parent.paths(GROUP, fold)["journal"], trace),
           "E82.4.6.2": full_audit.archived_rng_matches(ext.paths(GROUP, fold)["journal"], trace)}
    if fold == prior.FOLD:
        rng["E82.4.6.1"] = full_audit.archived_rng_matches(prior.paths(GROUP, fold)["journal"], trace)
    run.restore(ctx, best, hashes)
    h.require(h.sha(run.bundle(ctx)) == best["state_sha"], "Best model or its four optimizers failed exact reload")
    full, cap = proof.independent_eval(ctx)
    off, off_cap = proof.independent_eval(ctx, "residual_off")
    anonymous, anon_cap = proof.independent_eval(ctx, "anonymous_query")
    proof.match_eval(full, best["evaluation"], "New Best")
    geometry = {condition: f.branch_formulas(ctx, data, intervention=condition) for condition, data in
                (("full", cap), ("residual_off", off_cap), ("anonymous_query", anon_cap))}
    mask_best = mask_proof(ctx, full, off, cap, off_cap)
    mask_proof(ctx, anonymous, off, anon_cap, off_cap)
    rows = ctx["members"]["test"]
    keep = ~np.isin(rows, EXPOSED[fold])
    delta = cap["details"]
    residual = {key + "_l2": np.linalg.norm(f.f64(delta[key][rows]), axis=1)
                for key in ("delta", "delta_parallel", "delta_perp")}
    residual["cam_ratio"] = f.f64(delta["ratio"][rows])
    query_delta = np.linalg.norm(f.f64((cap["enhanced"] - anon_cap["enhanced"])[rows]), axis=1)
    evidence = {"changed_feature_rows": int(np.count_nonzero(query_delta[keep])),
                "mean_feature_shift_l2": float(query_delta[keep].mean()),
                "changed_probability_rows": int(np.count_nonzero(full["probability"][keep, 0] !=
                                                                  anonymous["probability"][keep, 0])),
                "mean_absolute_asd_probability_change": float(np.abs(full["probability"][keep, 0] -
                                                                      anonymous["probability"][keep, 0]).mean()),
                "hard_flips": int(np.count_nonzero(full["prediction"][keep] != anonymous["prediction"][keep]))}
    item = {"full": full, "off": off, "anonymous": anonymous, "truth": ctx["state"]["labels_np"][rows].copy(),
            "subjects": rows + 1, "best_epoch": best["epoch"], "end_epoch": latest["epoch"],
            "best_sha": task["best_sha"], "best_path": str(files["best"]), "inference_sha": h.sha(full),
            "residual": residual, "query_feature_l2": query_delta}
    gradients = scope_gradient(run, ctx, preflight["dropout"])
    run.restore(ctx, best, hashes)
    budget = proof.twin_probe(ctx, preflight["dropout"], check_each=True)
    del ctx, cap, off_cap, anon_cap
    fresh = run.setup(GROUP, fold)
    recovered = proof.check_latest(fresh, latest, manifest, task, preflight)
    h.require(recovered["exact"], "Latest next step and evaluation differ from uninterrupted witness")
    run.restore(fresh, latest, hashes)
    run.restore_ema(fresh, latest["eval_ema"])
    end, end_cap = proof.independent_eval(fresh)
    off_end, off_end_cap = proof.independent_eval(fresh, "residual_off")
    mask_latest = mask_proof(fresh, end, off_end, end_cap, off_end_cap)
    return item, {"status": "PASS", "fold": fold, "best_reloaded": True, "latest_reloaded": True,
                  "latest_next_exact": True, "initial_parity": parity, "membership": membership, "gaussian": banks,
                  "geometry": geometry, "best_mask": mask_best, "latest_mask": mask_latest,
                  "all_formal_masked_epochs": len(records), "scope_gradient_best": gradients, "journal": journal,
                  "best_gradient_and_budget": budget, "latest_replay": recovered, "rng": rng, "query": evidence,
                  "first_strict_max": True, "best_epoch": best["epoch"], "end_epoch": latest["epoch"],
                  "best_sha": task["best_sha"], "latest_sha": task["latest_sha"], "journal_sha": task["journal_sha"]}


def full_path_reference(proof, canonical, full_manifest, guard):
    verified = dict(canonical)
    result_csv = FULL_PATH / "results" / "fold_metrics.csv"
    with (FULL_PATH / "results" / "patient_level.csv").open(newline="") as stream:
        patient_rows = {(int(r["fold"]), int(r["patient_id_1based"])): r for r in csv.DictReader(stream)}
    audit = h.read_json(FULL_PATH / "results" / "audit_status.json")
    h.require(audit["status"] == "COMPLETE_AUDITED", "E82.4.6.2 patient reference is not audited")
    for fold in FOLDS:
        run, files = ext.engine(fold), ext.paths(GROUP, fold)
        digest = full_audit.verify_archived_digest(files["best"], guard)
        ctx = ext.setup(GROUP, fold)
        saved = h.load(files["best"])
        run.restore(ctx, saved, full_manifest["implementation"])
        full, _ = proof.independent_eval(ctx)
        off, _ = proof.independent_eval(ctx, "residual_off")
        proof.match_eval(full, saved["evaluation"], "E82.4.6.2 immutable Best")
        subjects, y = canonical[fold]["subjects"], canonical[fold]["truth"]
        f.equal(subjects, ctx["members"]["test"] + 1, "E82.4.6.2 original test order")
        f.equal(y, ctx["state"]["labels_np"][ctx["members"]["test"]], "E82.4.6.2 true labels")
        full_audit.checked_fold_scores(result_csv, lambda r: int(r["fold"]) == fold and r["scope"] == "all"
                                       and r["method"] == "E82.4.6.2", score(full["probability"], full["prediction"], y))
        for name, prediction in (("full", full), ("off", off)):
            exported = [patient_rows[fold, int(i)] for i in subjects]
            f.equal([float(r["p_asd_current_" + name]) for r in exported], prediction["probability"][:, 0],
                    "E82.4.6.2 exact archived subject probabilities")
            f.equal([int(r["prediction_current_" + name]) for r in exported], prediction["prediction"],
                    "E82.4.6.2 exact archived subject predictions")
        verified[fold] = {"full": full, "off": off, "truth": y, "subjects": subjects, "best_epoch": saved["epoch"],
                          "best_path": str(files["best"]), "best_sha": digest, "inference_sha": h.sha(full)}
    return verified


def audit(manifest):
    h.require(implementation() == manifest["implementation"], "Training sources changed before independent audit")
    canonical_manifest, previous, full = source_manifests()
    variants, certificates = {}, {}
    for fold in FOLDS:
        variants[fold], certificates[str(fold)] = audit_fold(fold, manifest)
        print(f"AUDITED Fold {fold}: Best, Latest, next training/evaluation, RNG, EMA and scope", flush=True)
    proof = proof_engine(engine(FOLDS[0]))
    canonical = ext.prior_audit.parent_certificates(proof, canonical_manifest, manifest["protected_sha256"])
    darg = full_audit.evaluate_darg(proof, canonical, manifest)
    direct = full_audit.evaluate_direct_prior(proof, canonical, previous, manifest)
    from_full = full_path_reference(proof, canonical, full, manifest)
    h.require(all(set(ref) == set(h.FOLDS) for ref in (darg, canonical, direct, from_full)),
              "A required historical comparison is incomplete")
    return darg, canonical, direct, from_full, {**canonical, **variants}, certificates
