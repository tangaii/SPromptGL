"""Recorded post-training audit-entry repair. Training implementation is unchanged."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Import the vendored historical runtime before NumPy.  It installs the
# archived dependency path used by the training and independent-audit code.
import core as _core
import independent_465 as ind
import numpy as np
import torch
if not hasattr(np, "long"):
    np.long = np.int64

def audit_task_fixed(task, fold, manifest):
    run = ind.engine(task, fold)
    proof = ind.prior.load_parent_module("_substitution_checkpoint_audit_fix", "audit.py"); proof.run = run
    files = ind.paths(task, fold); pre = ind.h.read_json(files["preflight"]); taskmeta = ind.h.read_json(files["task"])
    ind.h.require(pre["status"] == "PASS" and taskmeta["status"] == "TRAINED" and pre["implementation"] == manifest["implementation"], "Task artifacts are not frozen")
    ind.h.require(all(ind.h.sha_file(files[k]) == taskmeta[k + "_sha"] for k in ("best", "latest", "journal")), "Task SHA mismatch")
    best, latest = ind.h.load(files["best"]), ind.h.load(files["latest"])
    ctx = run.setup(task, fold); ind.h.require(ctx["provenance"] == pre["provenance"], "Reloaded scope differs")
    membership, gaussian = ind.membership_proof(ctx), ind.gaussian_proof(ctx)
    journal = proof.audit_journal(ctx, taskmeta, best, latest, pre)
    ind.h.require(journal["ema_independent"] and journal["ema_bitwise_epochs"] == journal["replayed_epochs"], "EMA replay differs")
    records = [json.loads(x) for x in files["journal"].read_bytes().splitlines()]
    for rec in records:
        ind.h.require(rec["routing"]["scope_receipt"]["phases"] == ["train", "eval"], "Mask receipt incomplete")
    trace = {r["epoch"]: r["rng"] for r in records}
    ind.h.require(all(r["rng"]["rng_after_train"] == r["rng"]["rng_after_eval"] for r in records), "Evaluation consumed RNG")
    run.restore(ctx, best, manifest["implementation"])
    full, cap = proof.independent_eval(ctx); off, offcap = proof.independent_eval(ctx, "residual_off")
    anonymous, _ = proof.independent_eval(ctx, "anonymous_query")
    proof.match_eval(full, best["evaluation"], "Best")
    rows = ctx["members"]["test"]; y = ctx["state"]["labels_np"][rows].copy(); masked = np.isin(rows, ind.EXPOSED[fold])
    details = cap["details"]; masked_rows = list(ind.EXPOSED[fold])
    for key in ind.MASKED_COMPONENTS:
        ind.h.require(torch.equal(details[key][masked_rows], torch.zeros_like(details[key][masked_rows])), "Masked residual changed")
    ind.h.require(torch.equal(cap["enhanced"][masked_rows], cap["cam"][masked_rows]), "Masked CAM changed")
    ind.h.require(np.array_equal(full["prediction"][masked], off["prediction"][masked]) and np.array_equal(full["probability"][masked], off["probability"][masked]), "Masked prediction differs")
    residual = {k: np.linalg.norm(details[k][rows].detach().cpu().numpy().astype(np.float64), axis=1) for k in ("delta", "delta_parallel", "delta_perp")}
    ind.h.require(np.any(residual["delta"][~masked] > 0), "Held-out residual path inactive")
    scope = ind.gradient_scope(run, ctx, pre["dropout"])
    fresh = run.setup(task, fold); latest_check = proof.check_latest(fresh, latest, manifest, taskmeta, pre)
    ind.h.require(latest_check["exact"], "Latest next-step recovery differs")
    item = {"full": full, "off": off, "anonymous": anonymous, "truth": y, "subjects": rows + 1, "task_group": task, "fold": fold,
            "best_epoch": best["epoch"], "end_epoch": latest["epoch"], "best_sha": taskmeta["best_sha"], "best_path": str(files["best"]),
            "masked": rows[masked].tolist(), "residual": residual}
    cert = {"status": "PASS", "task_group": task, "fold": fold, "best_reloaded": True, "latest_reloaded": True, "latest_next_exact": True,
            "membership": membership, "gaussian": gaussian,
            "residual": {k: {"masked_nonzero": int(np.count_nonzero(v[masked])), "heldout_nonzero": int(np.count_nonzero(v[~masked])),
                              "masked_mean": float(v[masked].mean()), "heldout_mean": float(v[~masked].mean())} for k, v in residual.items()},
            "scope_gradient": scope, "journal": journal, "best_epoch": best["epoch"], "end_epoch": latest["epoch"],
            "best_sha": taskmeta["best_sha"], "latest_sha": taskmeta["latest_sha"], "journal_sha": taskmeta["journal_sha"], "rng_trace": trace,
            "audit_entry_repair": "masked row indexing corrected in audit/audit_fix.py after training; training implementation unchanged"}
    return item, cert

def main():
    root = ind.ROOT; manifest = ind.h.read_json(_core.MANIFEST)
    ind.h.require(manifest["status"] in ("TRAINED", "AUDIT_FAILED", "AUDITED"), "Training is not complete")
    item, cert = audit_task_fixed("G1", 4, manifest)
    # Reuse source verification remains read-only and is repeated here.
    from execute import reuse_audit, verify_guard
    reuse = reuse_audit(manifest)
    audit = {"status": "FOLD4_AUDITED", "experiment": ind.CONFIG["experiment"], "tasks": {"G1_f04": cert}, "reused": reuse,
             "audit_entry_repair": cert["audit_entry_repair"]}
    ind.h.write_json(root / "results" / "audit_status.json", audit)
    manifest["status"] = "FOLD4_AUDITED"; manifest["audit"] = {"G1_f04": cert}; manifest["reuse"] = reuse
    manifest["audit_entry_repair"] = cert["audit_entry_repair"]
    manifest["guard_after"] = verify_guard(manifest)
    ind.h.write_json(_core.MANIFEST, manifest)
    print(json.dumps({"status": "FOLD4_AUDITED", "best_epoch": cert["best_epoch"], "end_epoch": cert["end_epoch"]}, indent=2))

if __name__ == "__main__": main()
