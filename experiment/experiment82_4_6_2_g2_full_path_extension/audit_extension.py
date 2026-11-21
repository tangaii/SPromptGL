"""Independently audit the new fold scope, training, reloads and historical comparisons."""
from __future__ import annotations

import csv
from itertools import combinations
import json

from extension import (ADDED, CFG, CONFIG, EXPOSED, FOLDS, GROUP, PARENT, PRIOR, before_context, engine, f, h,
                       implementation, optimization, parent, parent_preflight, paths, prior, prior_audit,
                       setup, source_manifests, spectral)

import numpy as np
import torch
from torch.nn import functional as F

METRICS = prior_audit.METRICS
score = prior_audit.metric_values


def held_out_scope(ctx):
    m = ctx["members"]
    fold = ctx["fold"]
    rows = m["test"][~np.isin(m["test"], EXPOSED[fold])]
    h.require(len(rows) == 82 and len(m["legal"]) == len(m["train"]) == 789 and len(m["test"]) == 87 and
              np.array_equal(np.intersect1d(m["legal"], m["test"]), EXPOSED[fold]) and
              not np.intersect1d(m["legal"], rows).size, "Unexpected held-out training membership")
    return {**ctx, "members": {**m, "test": rows}}


def rederive_pairs(indices, labels, supports, seed):
    """Reconstruct the E61 pair ordering independently from anchor pool and ordered patients."""
    n = len(labels)
    anchored = np.concatenate(tuple(supports), axis=0)
    graph_keys = np.unique(np.minimum(*anchored.T) * n + np.maximum(*anchored.T))
    rows = np.asarray(list(combinations(indices, 2)), dtype=np.int64)
    a, b = rows.T
    anchored = np.isin(np.minimum(a, b) * n + np.maximum(a, b), graph_keys)
    equivalent = labels[a] == labels[b]
    rng = np.random.Generator(np.random.PCG64(seed))
    draw, categories = [], []
    for same in (True, False):
        eligible = equivalent == same
        preferred = np.flatnonzero(eligible & anchored)
        fallback = np.flatnonzero(eligible & ~anchored)
        k = min(2048, len(preferred))
        from_graph = rng.choice(preferred, k, replace=False) if k else preferred[:0]
        from_other = rng.choice(fallback, 2048 - k, replace=False) if k < 2048 else fallback[:0]
        draw.extend((from_graph, from_other))
        categories.append(k)
    chosen = np.concatenate(draw)
    perm = rng.permutation(chosen.size)
    return rows[chosen[perm]], equivalent[chosen[perm]].astype(np.int64), categories


def scope_proof(ctx):
    scoped = held_out_scope(ctx)
    now, previous = ctx["members"], ctx["old_members"]
    state, fold = ctx["state"], ctx["fold"]
    trace = ctx["provenance"]["registration"]
    n = len(previous["train"])
    f.equal(now["train"][:n], previous["train"], "Original ordered training prefix")
    f.equal(now["train"][n:], ADDED[fold], "Only the configured appended rows")
    f.equal(now["test"], previous["test"], "Unchanged outer test order")
    f.equal(now["legal"], now["train"], "Identical Native and SCMR membership")
    h.require(len(scoped["members"]["test"]) == 82 and not np.isin(scoped["members"]["test"], now["train_pairs"]).any()
              and all(trace["per_overlap_pair_count"][str(i + 1)] > 0 for i in ADDED[fold]),
              "A held-out or appended patient's relation-pool membership differs")
    f.equal(state["train"], now["train"], "Device Native training index")
    f.equal(state["legal"], now["legal"], "Device SCMR training index")
    f.equal(state["train_pairs"], now["train_pairs"], "Device Native relation indices")
    f.equal(state["train_same"], now["train_same"], "Device Native relation classes")
    f.equal(state["weights"], now["weights"], "Device Native class weights")
    labels = state["labels_np"]
    rng = h.rng_sha()
    graph_counts = {}
    for name, m in (("original", previous), ("extended", now)):
        pairs, relation, pool = rederive_pairs(m["train"], labels, state["supports"], trace["pair_rng_seed"])
        f.equal(pairs, m["train_pairs"], f"{name} ordered pair sampling")
        f.equal(relation, m["train_same"], f"{name} pair targets")
        graph_counts[name] = pool
        actual_weights = np.asarray([len(m["train"]) / (2 * int((labels[m["train"]] == label).sum()))
                                     for label in (0, 1)], dtype=np.float32)
        f.equal(m["weights"], actual_weights, f"{name} fixed ASD/TD class weights")
    h.require(rng == h.rng_sha(), "Independent pair sampling changed global RNG")
    return {**trace, "test_and_original_training_prefix_exact": True,
            "exact_independent_pair_sampler_for_original_and_extended": True,
            "same_and_different_sampled_from_graph": graph_counts, "remaining_test_count": 82}


def gaussian_proof(ctx):
    scoped = held_out_scope(ctx)
    current, old = ctx["members"]["legal"], ctx["old_members"]["legal"]
    ref, state, fold = ctx["reference"], ctx["state"], ctx["fold"]
    y = state["labels_np"]
    f.equal(ref["legal"], current, "Gaussian legal rows")
    f.equal(ref["labels"], y[current], "Gaussian labels")
    f.equal(ref["bank"][scoped["members"]["test"]], np.zeros(82), "Only inference bank for remaining test")
    proof = f.class_banks(scoped)
    check, oof = {}, {}
    for row in EXPOSED[fold]:
        i, = np.flatnonzero(current == row)
        bank = int(ref["group"][i])
        h.require(bool(ref["mask"][0, y[row], i]) and not ref["mask"][bank, :, i].any(), "OOF self-label")
        oof[str(row + 1)] = {"label": int(y[row]), "bank": bank, "in_full_inference_bank": True, "absent_from_own_bank": True}
    rng = h.rng_sha()
    inputs = (fold, ctx["warm_cam"], ctx["original"])
    h.require(h.sha(prior.rebuild_reference(*inputs, current, y[current].copy())) == ctx["provenance"]["reference"]
              and h.sha(prior.rebuild_reference(*inputs, old, y[old].copy())) == ctx["provenance"]["source_reference_sha"]
              and h.rng_sha() == rng, "Gaussian replay differs from old or new certificate or changes RNG")
    check.update(proof, oof_exposed=oof, exact_rebuild=True, original_kernel_exact=True, main_rng_unchanged=True,
                 train_banks_omit_each_own_row=789, reference_sha=h.sha(ref), stratified_seed=ref["seed"])
    return check


def proof_engine(run):
    proof = prior.load_parent_module("_extended_checkpoint_audit", "audit.py")
    proof.run = run
    verify = proof.twin_probe

    def gradients(ctx, inventory, *, check_each=False):
        scoped = held_out_scope(ctx)
        state = ctx["state"]
        kept = state["test"][~np.isin(state["test"], ctx["members"]["train"])]
        f.equal(kept, scoped["members"]["test"], "Remaining held-out test order")
        # The old assertion checks zero classification gradient for the actual held-out rows.
        # The evaluation path still scores the original full fold.
        return verify({**ctx, "state": {**state, "test": kept}}, inventory, check_each=check_each)

    proof.twin_probe = gradients
    return proof


def label_boundary(run, ctx, checkpoint):
    state = ctx["state"]
    rows = held_out_scope(ctx)["members"]["test"]
    prev_t, prev_np = state["y"][rows].clone(), state["labels_np"][rows].copy()

    def sample():
        h.base().restore_rng(checkpoint["rng"])
        output, cap = run.forward(state, True, ctx["module"])
        d = cap["details"]
        full, _ = optimization.losses(GROUP, state["cam"].Outputlayer, cap, state["legal"], ctx["labels"],
                                     ctx["module"].spectrum.basis.to(cap["cam"]), ctx["hard_initial"])
        terms = list(full.values()) + [output["cls"]]
        for term in terms:
            g, = torch.autograd.grad(term, cap["cam"], retain_graph=True)
            h.require(not g[rows].count_nonzero().item(), "A remaining held-out label has a direct training path")
        selection = [d[k] for k in ("delta", "llr", "reliability", "desired_change", "disagreement")]
        return h.sha(([output[k] for k in h.LOSS_KEYS], full, selection,
                      ctx["module"].statistics.observation(cap["cam"])))

    expected = sample()
    try:
        state["y"][rows] = 1 - prev_t
        state["labels_np"][rows] = 1 - prev_np
        h.require(sample() == expected, "A remaining held-out label altered training or Gaussian statistics")
    finally:
        state["y"][rows] = prev_t
        state["labels_np"][rows] = prev_np
        h.base().restore_rng(checkpoint["rng"])
    return {"left_out": len(rows), "held_out_inversion_exact": True, "direct_supervision_gradients_zero": True}


def patient_gradients(run, ctx, dropout, *, expect_prompt):
    state = ctx["state"]
    with run.dropout_inventory() as calls:
        out, cap = run.forward(state, True, ctx["module"])
    h.require(calls == dropout, "Patient gradient probe did not use the six historical Dropout operations")
    head = state["cam"].Outputlayer
    layer_norm, linear, gelu, _ = head.mlp_head
    true_labels = ctx["labels"]
    inputs = cap["enhanced"][state["legal"]]
    predictor = head.classifier(gelu(linear(layer_norm(inputs))))
    row_losses = -F.log_softmax(predictor, dim=-1).gather(1, true_labels[:, None]).flatten()
    d = cap["details"]
    optimized, _ = optimization.losses(GROUP, head, cap, state["legal"], true_labels,
                                       ctx["module"].spectrum.basis.to(cap["cam"]), ctx["hard_initial"])
    actual, = torch.autograd.grad(optimized["full"], d["delta"], retain_graph=True)
    native_cls = F.nll_loss(out["logp"][state["train"]], state["y"][state["train"]], weight=state["weights"])
    f.equal(native_cls, out["cls"], "Original class-weighted native NLL on the expanded training members")
    g_native, = torch.autograd.grad(native_cls, out["logp"], retain_graph=True)
    h.require(not g_native[held_out_scope(ctx)["members"]["test"]].count_nonzero().item(),
              "Native classification trained on a remaining held-out label")
    prompt, control = (ctx["families"][name] for name in ("prompt", "control"))
    parameters = (d["delta"], *prompt, *control, d["raw_parallel"], d["raw_perp"])
    result = {}
    for index in EXPOSED[ctx["fold"]]:
        j, = np.flatnonzero(ctx["members"]["legal"] == index)
        weighted = row_losses[j] / (2 * int((true_labels == true_labels[j]).sum()))
        grads = torch.autograd.grad(weighted, parameters, retain_graph=True, allow_unused=True)
        f.bounded(grads[0][index], f.f64(actual[index]), 30, "Per-patient true Full CE contribution")
        h.require(grads[0].count_nonzero().item() == grads[0][index].count_nonzero().item(),
                  "A direct single-patient CAM loss touched another row")
        p = 1 + len(prompt)
        values = {"native_nll_logp_l2": float(g_native[index].norm()), "full_ce_delta_l2": float(grads[0][index].norm()),
                  "prompt_l2": optimization.norm(grads[1:p]), "control_l2": optimization.norm(grads[p:-2]),
                  "spectral_candidate_l2": float(grads[-2][index].norm()), "complement_candidate_l2": float(grads[-1][index].norm())}
        h.require(np.isfinite(list(values.values())).all() and all(v > 0 for k, v in values.items() if k != "prompt_l2")
                  and (not expect_prompt or values["prompt_l2"] > 0), f"Patient {index + 1} lost a classification gradient")
        result[str(index + 1)] = values
    return result


def preflight_fold(fold, hashes):
    run = engine(fold)
    device = h.archive.verify_device()
    h.require(device["uuid"] == CONFIG["physical_gpu_uuid"] and device["index"] == CONFIG["physical_gpu_index"],
              "Incorrect physical training GPU")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    source, old_preflight = before_context(fold)
    _, old_cam = run.forward(source["state"], False)
    old_head_input = h.sha(old_cam["cam"])
    original, old_drop, original_rng = parent_preflight.historic_signature(source, wrapped=False)
    original_record = old_preflight.get("native_disabled_exact", old_preflight.get("disabled_native_exact"))
    h.require(original_record == original and original_rng == h.hist_trace(fold)[11],
              "Previous native first-step certificate differs")
    del source, old_cam

    ctx = setup(GROUP, fold)
    s = ctx["state"]
    scope = scope_proof(ctx)
    banks = gaussian_proof(ctx)
    difficulty = f.legal_memory(held_out_scope(ctx))
    snapshot = parent_preflight.restored_payload(ctx, hashes, epoch=10, wait=0, best_epoch=0, best_correct=-1)
    _, initial_cap = run.evaluate(ctx)
    h.require(h.sha(initial_cap["cam"]) == old_head_input, "Epoch 10 CAM changed from membership preparation")
    f.equal(initial_cap["details"]["delta"], torch.zeros_like(initial_cap["cam"]), "Original G2's zero initial residual")
    geometry = f.branch_formulas(ctx, initial_cap)
    extended, drop, rng = parent_preflight.historic_signature(ctx, wrapped=False)
    h.require(len(drop) == 6 and drop == old_drop == old_preflight["dropout"] and rng == original_rng
              and extended["after_update"] != original["after_update"], "Extended native transition or shared RNG differs")
    run.restore(ctx, snapshot, hashes)
    h.require(parent_preflight.historic_signature(ctx, wrapped=True, active=True) == (extended, drop, rng),
              "New branch initialization changed the extended native path")
    run.restore(ctx, snapshot, hashes)
    result, diag, first_rng, _, _ = run.transition(ctx, 11, drop, inspect=True)
    h.require(diag["losses"]["full"] == diag["losses"]["off"] and diag["losses"]["difference"] == 0 and
              h.sha(h.base_bundle(s)) == extended["after_update"] and first_rng == rng and all(
                  diag["budget"][name]["raw"] == diag["budget"][name]["total_applied"] == 0
                  for name in optimization.ORIGINAL), "Zero-residual initial native parity failed")
    step = h.sha((result, diag, first_rng, h.sha(run.ema(ctx))))
    next_state = parent_preflight.restored_payload(ctx, hashes, epoch=11)
    witness = run.next_witness(ctx, 12, drop)
    run.restore(ctx, next_state, hashes)
    h.require(witness == run.next_witness(ctx, 12, drop) and witness["rng"] == h.hist_trace(fold)[12],
              "Preflight recoverable checkpoint or RNG differs")
    mature = parent_preflight.restored_payload(ctx, hashes, epoch=12)
    isolation = label_boundary(run, ctx, mature)
    run.restore(ctx, mature, hashes)
    probe = proof_engine(run).twin_probe(ctx, drop, check_each=True)
    run.restore(ctx, mature, hashes)
    routes = patient_gradients(run, ctx, drop, expect_prompt=True)
    run.restore(ctx, mature, hashes)
    _, cap = run.evaluate(ctx)
    f.equal(cap["logits"], spectral.native_logits(s["cam"].Outputlayer, cap["enhanced"]), "Same original deterministic head")
    return {"status": "PASS", "group": GROUP, "fold": fold, "implementation": hashes, "provenance": ctx["provenance"],
            "dropout": drop, "device": device, "membership": scope, "banks": banks, "difficulty": difficulty,
            "label_boundary": isolation, "geometry": geometry, "gradient_preflight": probe, "patient_gradients": routes,
            "initial_ema": h.sha(snapshot["ema"]), "initial_cam_exact": True,
            "archived_native_signature": original, "extended_native_signature": extended,
            "first_step_sha": step, "first_step_diagnostics": diag, "first_rng": first_rng, "next_step": witness,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}


def archived_rng_matches(filename, actual):
    recorded = {int(row["epoch"]): row["rng"] for row in
                (json.loads(line) for line in filename.read_bytes().splitlines())}
    common = recorded.keys() & actual.keys()
    h.require(common and all(recorded[i] == actual[i] for i in common), "Shared RNG differs from historical G2")
    return {"common_epochs": len(common), "exact_phases": 3 * len(common)}


def audit_fold(fold, manifest):
    run, files = engine(fold), paths(GROUP, fold)
    proof = proof_engine(run)
    preflight, task = (h.read_json(files[key]) for key in ("preflight", "task"))
    hashes = manifest["implementation"]
    h.require(preflight["status"] == "PASS" and task["status"] == "TRAINED" and
              preflight["implementation"] == task["implementation"] == hashes and
              all(h.sha_file(files[k]) == task[k + "_sha"] for k in ("best", "latest", "journal")),
              "Formal checkpoint, ledger, or preflight commitment differs")
    best, latest = (h.load(files[k]) for k in ("best", "latest"))
    h.require(latest["best_sha"] == task["best_sha"], "Latest points at a different Best")
    ctx = setup(GROUP, fold)
    h.require(ctx["provenance"] == preflight["provenance"], "Trained and reloaded fold scopes differ")
    membership = scope_proof(ctx)
    banks = gaussian_proof(ctx)
    journal = proof.audit_journal(ctx, task, best, latest, preflight)
    h.require(journal["ema_independent"] and journal["ema_bitwise_epochs"] == journal["replayed_epochs"],
              "Independent EMA trajectory differs")
    current_trace = {row["epoch"]: row["rng"] for row in
                     (json.loads(line) for line in files["journal"].read_bytes().splitlines())}
    rng = {"darg": {"common_epochs": journal["archived_epochs"], "exact_phases": journal["rng_phase_comparisons"]},
           "original_g2": archived_rng_matches(parent.paths(GROUP, fold)["journal"], current_trace),
           "replayed_epochs": journal["replayed_epochs"]}
    if fold == prior.FOLD:
        rng["82.4.6.1"] = archived_rng_matches(prior.paths(GROUP, fold)["journal"], current_trace)
    first, tenth = (journal[x]["paths"]["exposed_patient_gradients"] for x in ("first", "tenth"))
    ids = {str(i + 1) for i in EXPOSED[fold]}
    h.require(set(first) == set(tenth) == ids and all(v["prompt_l2"] > 0 for v in tenth.values()),
              "Recorded mature Full CE did not reach every exposed patient's Prompt route")

    run.restore(ctx, best, hashes)
    h.require(h.sha(run.bundle(ctx)) == best["state_sha"], "Best model and four optimizers did not reload")
    now, cap = proof.independent_eval(ctx)
    off, off_cap = proof.independent_eval(ctx, "residual_off")
    proof.match_eval(now, best["evaluation"], "Fold extension Best")
    geometry = f.branch_formulas(ctx, cap)
    geometry_off = f.branch_formulas(ctx, off_cap, intervention="residual_off")
    f.equal(off_cap["enhanced"], cap["cam"], "Fixed-weight Off uses the same unmodified CAM")
    d = cap["details"]
    rows = ctx["members"]["test"]
    labels = ctx["state"]["labels_np"][rows].copy()
    residual = {key + "_l2": np.linalg.norm(f.f64(d[key][rows]), axis=1)
                for key in ("delta", "delta_parallel", "delta_perp")}
    residual["cam_ratio"] = f.f64(d["ratio"][rows])
    item = {"full": now, "off": off, "subjects": rows + 1, "truth": labels, "best_epoch": best["epoch"],
            "end_epoch": latest["epoch"], "best_sha": task["best_sha"], "best_path": str(files["best"]),
            "inference_sha": h.sha(now), "residual": residual}
    best_routes = patient_gradients(run, ctx, preflight["dropout"], expect_prompt=True)
    run.restore(ctx, best, hashes)
    probe = proof.twin_probe(ctx, preflight["dropout"], check_each=True)
    del ctx, cap, off_cap
    fresh = setup(GROUP, fold)
    recovered = proof.check_latest(fresh, latest, manifest, task, preflight)
    h.require(recovered["exact"], "Latest next training and evaluation differed")
    run.restore(fresh, latest, hashes)
    latest_routes = patient_gradients(run, fresh, preflight["dropout"], expect_prompt=True)
    gradients = {i: {"first": first[i], "tenth": tenth[i], "best": best_routes[i], "latest": latest_routes[i]} for i in sorted(ids)}
    return item, {"status": "PASS", "fold": fold, "best_loaded": True, "latest_loaded": True, "latest_next_exact": True,
                  "membership": membership, "banks": banks, "label_boundary": preflight["label_boundary"],
                  "geometry": geometry, "geometry_off": geometry_off, "journal": journal, "rng": rng,
                  "gradients": gradients, "best_gradient_probe": probe, "latest_replay": recovered,
                  "best_epoch": best["epoch"], "end_epoch": latest["epoch"], "first_strict_max": True,
                  "best_sha": task["best_sha"], "latest_sha": task["latest_sha"], "journal_sha": task["journal_sha"],
                  "preflight_sha": h.sha_file(files["preflight"])}


def verify_archived_digest(path, manifest):
    actual = h.sha_file(path)
    h.require(actual == manifest["protected_sha256"][str(path)], f"Protected archived artifact differs: {path}")
    return actual


def checked_fold_scores(csv_path, match, measured):
    with csv_path.open(newline="") as stream:
        found, = [row for row in csv.DictReader(stream) if match(row)]
    h.require(all(abs(float(found[k]) - measured[k]) <= np.finfo(np.float64).eps for k in METRICS),
              "Independently recalculated historical metrics disagree")


@torch.no_grad()
def evaluate_darg(proof, historic, manifest):
    out = {}
    run = parent
    for fold, previous in historic.items():
        checkpoint, csv_path, group, scale = h.reference_paths(fold)["DARG"]
        sha = verify_archived_digest(checkpoint, manifest)
        ctx = parent.setup(GROUP, fold)
        state = ctx["state"]
        saved = h.load(checkpoint)
        h.require(scale == 1 and saved["fold"] == fold and saved["identity"] == ctx["provenance"]["identity"],
                  "DARG Best has a different history or fold")
        if "test_indices" in saved:
            f.equal(saved["test_indices"], ctx["members"]["test"], "DARG test order")
        proof.reference_modules(ctx, "DARG", saved)
        output, cap = run.forward(state, False)
        rows = ctx["members"]["test"]
        y = state["labels_np"][rows].copy()
        p = output["logp"][rows].exp().cpu().numpy()
        pred = p.argmax(-1)
        f.equal(p, saved["test_probability"], "Read-only DARG probabilities")
        f.equal(pred, saved["test_prediction"], "Read-only DARG predictions")
        f.equal(y, previous["truth"], "Read-only DARG labels")
        f.equal(rows + 1, previous["subjects"], "Read-only DARG order")
        computed = score(p, pred, y)
        checked_fold_scores(csv_path, lambda r: r["group"] == group and int(r["fold"]) == fold, computed)
        evaluation = {"probability": p, "prediction": pred, "logits": cap["logits"][rows].cpu().numpy()}
        out[fold] = {"full": evaluation, "off": evaluation, "truth": y, "subjects": rows + 1,
                     "best_epoch": saved["best_epoch_1based"], "best_path": str(checkpoint), "best_sha": sha,
                     "inference_sha": h.sha(evaluation)}
    return out


def evaluate_direct_prior(proof, historic, previous_manifest, manifest):
    fold = prior.FOLD
    files = prior.paths(GROUP, fold)
    sha = verify_archived_digest(files["best"], manifest)
    certificate = h.read_json(PRIOR / "results" / "audit_status.json")
    h.require(certificate["status"] == "COMPLETE_AUDITED" and previous_manifest["training"]["best_sha"] == sha,
              "E82.4.6.1 reference was not independently audited")
    state = prior.setup(GROUP, fold)
    saved = h.load(files["best"])
    prior.engine().restore(state, saved, previous_manifest["implementation"])
    current, _ = proof.independent_eval(state)
    off, _ = proof.independent_eval(state, "residual_off")
    proof.match_eval(current, saved["evaluation"], "E82.4.6.1 reference Best")
    subjects, y = historic[fold]["subjects"], historic[fold]["truth"]
    f.equal(state["members"]["test"] + 1, subjects, "E82.4.6.1 historical test order")
    f.equal(state["state"]["labels_np"][state["members"]["test"]], y, "E82.4.6.1 labels")
    csv_path = PRIOR / "results" / "fold_metrics.csv"
    checked_fold_scores(csv_path, lambda r: r["scope"] == "fold" and int(r["fold"]) == fold and
                        r["method"].startswith("82.4.6.1-G2"), score(current["probability"], current["prediction"], y))
    with (PRIOR / "results" / "patient_level.csv").open(newline="") as stream:
        archive = {int(r["patient_id_1based"]): r for r in csv.DictReader(stream) if int(r["fold"]) == fold}
    for k, patient in enumerate(subjects):
        item = archive[int(patient)]
        h.require(float(item["p_asd_current_full"]) == current["probability"][k, 0] and
                  float(item["p_asd_current_off"]) == off["probability"][k, 0] and
                  int(item["prediction_current_full"]) == current["prediction"][k] and int(item["true_label"]) == y[k],
                  "E82.4.6.1 committed patient result differs")
    return {**historic, fold: {"full": current, "off": off, "truth": y, "subjects": subjects, "best_epoch": saved["epoch"],
                              "best_path": str(files["best"]), "best_sha": sha, "inference_sha": h.sha(current)}}


def audit(manifest):
    h.require(implementation() == manifest["implementation"], "Frozen source differs in fresh audit")
    root, previous = source_manifests()
    variants, folds = {}, {}
    for fold in FOLDS:
        variants[fold], folds[str(fold)] = audit_fold(fold, manifest)
    proof = proof_engine(engine(FOLDS[0]))
    historic = prior_audit.parent_certificates(proof, root, manifest["protected_sha256"])
    darg = evaluate_darg(proof, historic, manifest)
    prior_eval = evaluate_direct_prior(proof, historic, previous, manifest)
    h.require(set(historic) == set(darg) == set(prior_eval) == set(h.FOLDS), "Historical comparison incomplete")
    return darg, historic, prior_eval, variants, folds
