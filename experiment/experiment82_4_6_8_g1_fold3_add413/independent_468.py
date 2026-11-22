"""Independent preflight and checkpoint audit for Experiment 82.4.6.7."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from core import (CFG, CONFIG, EXPOSED, ADDED, FOLDS, TASKS, GROUP, ROOT, MASKED_COMPONENTS,
                  engine, ext, h, implementation, parent, prior, spectral, optimization,
                  parent_preflight, paths)

def equal(a, b, message):
    if isinstance(a, torch.Tensor):
        b = torch.as_tensor(b, device=a.device)
        h.require(torch.equal(a, b), message)
    else:
        h.require(np.array_equal(np.asarray(a), np.asarray(b)), message)

def membership_proof(ctx):
    fold = ctx["fold"]; state = ctx["state"]; members = ctx["members"]; old = ctx["old_members"]
    overlap = np.asarray(EXPOSED[fold], dtype=np.int64)
    h.require(len(members["train"]) == CONFIG["native_rows"][str(fold)] and len(members["test"]) == 87
              and len(np.unique(members["train"])) == len(members["train"])
              and np.array_equal(np.sort(np.intersect1d(members["train"], members["test"])), np.sort(overlap)),
              "Extended native membership differs")
    h.require(len(members["legal"]) == 784 and len(np.unique(members["legal"])) == 784,
              "SCMR legal size differs")
    equal(members["legal"], old["legal"], "Original legal order changed")
    equal(np.sort(np.setdiff1d(members["train"], members["legal"])), np.sort(overlap),
          "Native/legal difference differs from overlap mask")
    equal(np.sort(np.flatnonzero(~ctx["module"].spectrum.allowed.cpu().numpy())), np.sort(overlap),
          "Allowed mask differs from overlap")
    for key in ("train", "legal", "test", "weights", "train_pairs", "train_same"):
        equal(state[key], members[key], "Device " + key)
    labels = state["labels_np"]
    equal(ctx["labels"], labels[members["legal"]], "Legal labels")
    h.require(np.array_equal(members["train"][:len(old["train"])], old["train"])
              and np.array_equal(members["train"][len(old["train"]):], np.asarray(ADDED[fold])),
              "Appended order differs")
    before = h.rng_sha()
    e61 = h.runner().load_history().load_e69().load_e61()
    pairs, same = e61.make_outer_pairs(fold, state["supports"], members["train"], labels)
    h.require(h.rng_sha() == before, "Independent pair audit changed global RNG")
    equal(pairs, members["train_pairs"], "Ordered relation pairs")
    equal(same, members["train_same"], "Relation labels")
    equal(np.bincount(same, minlength=2), [2048, 2048], "Balanced relation count")
    equal(members["weights"], (len(members["train"])/(2*np.bincount(labels[members["train"]], minlength=2))).astype(np.float32), "Class weights")
    reg = ctx["provenance"]["registration"]
    return {**reg, "legal_rows": len(members["legal"]), "native_rows": len(members["train"]),
            "overlap_rows": len(overlap), "pairs_rederived": True, "rng_unchanged": True}

def gaussian_proof(ctx):
    fold, ref, legal = ctx["fold"], ctx["reference"], ctx["members"]["legal"]
    labels = ctx["state"]["labels_np"]
    equal(ref["legal"], legal, "Gaussian legal rows")
    equal(ref["labels"], labels[legal], "Gaussian labels")
    equal(ctx["module"].statistics.legal, legal, "EMA legal rows")
    h.require(np.isfinite(ref["mean"]).all() and np.isfinite(ref["second"]).all()
              and np.isfinite(ref["stable"]).all(), "Gaussian moments nonfinite")
    group = np.asarray(ref["group"]); mask = np.asarray(ref["mask"]).astype(bool)
    h.require(mask.shape[2] == len(legal) and np.all(~mask[group, :, np.arange(len(legal))]),
              "OOF self-exclusion failed")
    h.require(np.all(mask.sum(-1) == np.asarray(ref["counts"])), "Gaussian counts mismatch")
    before = h.rng_sha()
    rebuilt = prior.rebuild_reference(fold, ctx["warm_cam"], ctx["original"], legal, ref["labels"].copy())
    h.require(h.sha(ref) == h.sha(rebuilt), "Gaussian deterministic rebuild differs")
    h.require(h.rng_sha() == before, "Gaussian rebuild changed global RNG")
    return {"legal_rows": len(legal), "reference_sha": h.sha(ref), "rebuild_exact": True,
            "oof_self_exclusion": True, "counts": np.asarray(ref["counts"]).tolist(),
            "ema_legal_index_sha": h.sha(legal), "test_overlap_in_bank": False}

def gradient_scope(run, ctx, dropout):
    state, module = ctx["state"], ctx["module"]
    rows = np.asarray(EXPOSED[ctx["fold"]])
    with run.dropout_inventory() as calls:
        out, cap = run.forward(state, True, module)
    h.require(calls == dropout, "Dropout sequence changed in gradient audit")
    native = F.nll_loss(out["logp"][state["train"]], state["y"][state["train"]], state["weights"])
    h.require(torch.allclose(native, out["cls"], atol=0, rtol=0), "DARG classification rows differ")
    gp, = torch.autograd.grad(out["cls"], out["logp"], retain_graph=True)
    rows_t = torch.as_tensor(rows, device=gp.device)
    h.require(float(gp[rows_t].norm()) > 0, "Overlap DARG classification gradient absent")
    pair_logits = tuple(out["info"]["pair_logits"])
    pair = torch.stack([F.binary_cross_entropy_with_logits(x, state["train_same"]) for x in pair_logits]).mean()
    h.require(torch.allclose(pair, out["pair"], atol=0, rtol=0), "DARG relation rows differ")
    aux, _ = optimization.losses("G2", state["cam"].Outputlayer, cap, state["legal"], ctx["labels"],
                                 module.spectrum.basis.to(cap["cam"]), ctx["hard_initial"])
    dg, = torch.autograd.grad(aux["full"], cap["details"]["delta"], retain_graph=True)
    h.require(torch.equal(dg[rows_t], torch.zeros_like(dg[rows_t])), "Masked overlap received SCMR CE gradient")
    legal_unmasked = np.asarray([x for x in ctx["members"]["legal"] if x not in set(rows)])
    held_t = torch.as_tensor(legal_unmasked, device=dg.device)
    h.require(float(dg[held_t].norm()) > 0, "Unmasked legal SCMR residual gradient absent")
    return {"dropout_exact": True, "native_cls_overlap_grad": float(gp[rows_t].norm()),
            "scmr_full_delta_grad_masked": float(dg[rows_t].norm()),
            "scmr_full_delta_grad_unmasked_legal": float(dg[held_t].norm()), "legal_auxiliary": list(aux)}

def preflight_task(task, fold, hashes):
    run = engine(task, fold)
    proof = prior.load_parent_module("_substitution_audit_probe", "audit.py"); proof.run = run
    device = h.archive.verify_device(); ctx = run.setup(task, fold, verify_spectrum=True)
    membership, gaussian = membership_proof(ctx), gaussian_proof(ctx)
    snap = parent_preflight.restored_payload(ctx, hashes, epoch=10, wait=0, best_epoch=0, best_correct=-1)
    _, start = run.evaluate(ctx)
    h.require(torch.equal(start["details"]["delta"], torch.zeros_like(start["cam"])), "Initial residual nonzero")
    direct, dropout, rng = parent_preflight.historic_signature(ctx, wrapped=False)
    h.require(len(dropout) == 6 and rng == h.hist_trace(fold)[11], "Historical initial RNG differs")
    run.restore(ctx, snap, hashes)
    result, diag, first_rng, _, _ = run.transition(ctx, 11, dropout, inspect=True)
    h.require(first_rng == rng and np.isfinite(list(diag["losses"].values())).all(), "First transition invalid")
    first_step_sha = h.sha((result, diag, first_rng, h.sha(run.ema(ctx))))
    step_payload = parent_preflight.restored_payload(ctx, hashes, epoch=11)
    scope = gradient_scope(run, ctx, dropout)
    run.restore(ctx, step_payload, hashes)
    witness = run.next_witness(ctx, 12, dropout)
    run.restore(ctx, step_payload, hashes)
    h.require(witness == run.next_witness(ctx, 12, dropout), "Next step replay differs")
    return {"status": "PASS", "task_group": task, "fold": fold, "implementation": hashes,
            "device": device, "provenance": ctx["provenance"], "membership": membership,
            "gaussian": gaussian, "scope_gradient": scope, "dropout": dropout,
            "first_step_sha": first_step_sha, "first_rng": first_rng, "next_step": witness,
            "initial_residual_zero": True, "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20}

def audit_task(task, fold, manifest):
    run = engine(task, fold)
    proof = prior.load_parent_module("_substitution_checkpoint_audit", "audit.py"); proof.run = run
    files = paths(task, fold); pre = h.read_json(files["preflight"]); taskmeta = h.read_json(files["task"])
    h.require(pre["status"] == "PASS" and taskmeta["status"] == "TRAINED" and pre["implementation"] == manifest["implementation"],
              "Task artifacts are not frozen")
    h.require(all(h.sha_file(files[k]) == taskmeta[k + "_sha"] for k in ("best", "latest", "journal")), "Task SHA mismatch")
    best, latest = h.load(files["best"]), h.load(files["latest"])
    ctx = run.setup(task, fold); h.require(ctx["provenance"] == pre["provenance"], "Reloaded scope differs")
    membership, gaussian = membership_proof(ctx), gaussian_proof(ctx)
    journal = proof.audit_journal(ctx, taskmeta, best, latest, pre)
    h.require(journal["ema_independent"] and journal["ema_bitwise_epochs"] == journal["replayed_epochs"], "EMA replay differs")
    records = [json.loads(x) for x in files["journal"].read_bytes().splitlines()]
    for rec in records:
        h.require(rec["routing"]["scope_receipt"]["phases"] == ["train", "eval"], "Mask receipt incomplete")
    trace = {r["epoch"]: r["rng"] for r in records}
    h.require(all(r["rng"]["rng_after_train"] == r["rng"]["rng_after_eval"] for r in records), "Evaluation consumed RNG")
    run.restore(ctx, best, manifest["implementation"])
    full, cap = proof.independent_eval(ctx); off, offcap = proof.independent_eval(ctx, "residual_off")
    anonymous, _ = proof.independent_eval(ctx, "anonymous_query")
    proof.match_eval(full, best["evaluation"], "Best")
    rows = ctx["members"]["test"]; y = ctx["state"]["labels_np"][rows].copy(); masked = np.isin(rows, EXPOSED[fold])
    details = cap["details"]
    for key in MASKED_COMPONENTS:
        masked_rows = list(EXPOSED[fold])
        h.require(torch.equal(details[key][masked_rows], torch.zeros_like(details[key][masked_rows])), "Masked residual changed")
    masked_rows = list(EXPOSED[fold])
    h.require(torch.equal(cap["enhanced"][masked_rows], cap["cam"][masked_rows]), "Masked CAM changed")
    h.require(np.array_equal(full["prediction"][masked], off["prediction"][masked]) and
              np.array_equal(full["probability"][masked], off["probability"][masked]), "Masked prediction differs")
    residual = {k: np.linalg.norm(details[k][rows].detach().cpu().numpy().astype(np.float64), axis=1)
                for k in ("delta", "delta_parallel", "delta_perp")}
    h.require(np.any(residual["delta"][~masked] > 0), "Held-out residual path inactive")
    scope = gradient_scope(run, ctx, pre["dropout"])
    fresh = run.setup(task, fold); latest_check = proof.check_latest(fresh, latest, manifest, taskmeta, pre)
    h.require(latest_check["exact"], "Latest next-step recovery differs")
    item = {"full": full, "off": off, "anonymous": anonymous, "truth": y, "subjects": rows + 1,
            "task_group": task, "fold": fold, "best_epoch": best["epoch"], "end_epoch": latest["epoch"],
            "best_sha": taskmeta["best_sha"], "best_path": str(files["best"]), "masked": rows[masked].tolist(),
            "residual": residual}
    cert = {"status": "PASS", "task_group": task, "fold": fold, "best_reloaded": True,
            "latest_reloaded": True, "latest_next_exact": True, "membership": membership,
            "gaussian": gaussian, "residual": {k: {"masked_nonzero": int(np.count_nonzero(v[masked])),
                                                       "heldout_nonzero": int(np.count_nonzero(v[~masked])),
                                                       "masked_mean": float(v[masked].mean()),
                                                       "heldout_mean": float(v[~masked].mean())} for k, v in residual.items()},
            "scope_gradient": scope, "journal": journal, "best_epoch": best["epoch"],
            "end_epoch": latest["epoch"], "best_sha": taskmeta["best_sha"], "latest_sha": taskmeta["latest_sha"],
            "journal_sha": taskmeta["journal_sha"], "rng_trace": trace}
    return item, cert
