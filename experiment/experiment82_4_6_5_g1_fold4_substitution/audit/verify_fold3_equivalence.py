"""Independent equivalence proof for reusing audited E82.4.6.3 Fold 3."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core as local_core
import independent_465 as local_ind
import numpy as np
import torch

E463 = Path("/model/spromptgl/experiment/experiment82_4_6_3_darg_only_overlap")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()
                if key not in ("scope_receipt", "strict_overlap_mask")}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clean(item) for item in value)
    return value


def same_members(old, new):
    for key in ("train", "legal", "test", "weights", "train_pairs", "train_same"):
        local_core.h.require(np.array_equal(old["members"][key], new["members"][key]),
                             f"Fold-3 {key} differs")
        local_core.h.require(local_core.h.sha(old["state"][key]) == local_core.h.sha(new["state"][key]),
                             f"Fold-3 device {key} differs")
    local_core.h.require(np.array_equal(old["state"]["labels_np"], new["state"]["labels_np"]),
                         "Fold-3 labels differ")
    local_core.h.require(np.array_equal(old["state"]["train_np"], new["state"]["train_np"]),
                         "Fold-3 native training order differs")
    local_core.h.require(np.array_equal(old["module"].spectrum.allowed.cpu().numpy(),
                                        new["module"].spectrum.allowed.cpu().numpy()),
                         "Fold-3 residual permissions differ")
    local_core.h.require(local_core.h.sha(old["reference"]) == local_core.h.sha(new["reference"]),
                         "Fold-3 Gaussian reference differs")
    local_core.h.require(local_core.h.sha(old["original"]) == local_core.h.sha(new["original"]),
                         "Fold-3 Original O spectrum differs")
    local_core.h.require(local_core.h.sha(old["hard_initial"]) == local_core.h.sha(new["hard_initial"]),
                         "Fold-3 epoch-10 difficulty memory differs")
    for key in ("identity", "optimizers", "warm_cam", "spectrum", "basis", "prompt",
                "native_rng", "legal_index_sha", "memory_sha", "reference", "initial_module",
                "parameters"):
        local_core.h.require(old["provenance"].get(key) == new["provenance"].get(key),
                             f"Fold-3 initialization provenance differs: {key}")
    local_core.h.require(local_core.h.sha(old["run"].bundle(old)) ==
                         local_core.h.sha(new["run"].bundle(new)),
                         "Fold-3 initial model or optimizer bundle differs")


def main():
    old_adapter = load("e463_adapter_equivalence", E463 / "darg_only.py")
    old_run = old_adapter.engine(3)
    new_run = local_ind.engine("G1", 3)
    h = local_core.h
    old_ctx = old_run.setup("G2", 3, verify_spectrum=True)
    old_identity = local_core.ext.identity_snapshot(old_ctx["state"], old_ctx["module"], old_ctx["optimizer"])
    new_ctx = new_run.setup("G1", 3, verify_spectrum=True)
    new_identity = local_core.ext.identity_snapshot(new_ctx["state"], new_ctx["module"], new_ctx["optimizer"])
    h.require(old_identity == new_identity, "Fold-3 model, optimizer, RNG or graph inputs differ")
    old_ctx["run"] = old_run
    new_ctx["run"] = new_run
    same_members(old_ctx, new_ctx)

    old_manifest = h.read_json(E463 / "manifest.json")
    new_manifest = h.read_json(ROOT / "manifest.json")
    old_hashes, new_hashes = old_adapter.implementation(), local_ind.implementation()
    h.require(old_manifest["status"] == "COMPLETE_AUDITED" and old_manifest["implementation"] == old_hashes
              and new_manifest["implementation"] == new_hashes and old_run.CFG == new_run.CFG,
              "Fold-3 training configuration, early stopping or code freeze differs")
    old_preflight_path = E463 / "runtime/G2_f03.preflight.json"
    old_preflight = h.read_json(old_preflight_path)
    h.require(h.sha_file(old_preflight_path) == old_manifest["preflight"]["folds"]["3"]["sha"]
              and old_ctx["provenance"] == old_preflight["provenance"], "Fold-3 source preflight differs")
    membership = local_ind.membership_proof(new_ctx)
    gaussian = local_ind.gaussian_proof(new_ctx)
    withdrawn = np.asarray([674, 697, 778, 781]) - 1
    h.require(np.isin(withdrawn, new_ctx["members"]["test"]).all()
              and not np.isin(withdrawn, new_ctx["members"]["train"]).any()
              and not np.isin(withdrawn, new_ctx["members"]["legal"]).any()
              and not np.isin(withdrawn, new_ctx["members"]["train_pairs"]).any()
              and new_ctx["module"].spectrum.allowed[withdrawn.tolist()].all().item(),
              "A withdrawn patient did not return to ordinary test inference")
    old_snap = local_ind.parent_preflight.restored_payload(
        old_ctx, old_hashes, epoch=10, wait=0, best_epoch=0, best_correct=-1)
    new_snap = local_ind.parent_preflight.restored_payload(
        new_ctx, new_hashes, epoch=10, wait=0, best_epoch=0, best_correct=-1)
    h.require(old_snap["state_sha"] == new_snap["state_sha"] and old_snap["ema_sha"] == new_snap["ema_sha"]
              and old_snap["rng_sha"] == new_snap["rng_sha"] == h.hist_trace(3)[11]["rng_before"],
              "Fold-3 epoch-10 state, EMA, or RNG differs")
    old_run.restore(old_ctx, old_snap, old_hashes)
    old_native, old_dropout, old_native_rng = local_ind.parent_preflight.historic_signature(old_ctx, wrapped=False)
    new_run.restore(new_ctx, new_snap, new_hashes)
    new_native, new_dropout, new_native_rng = local_ind.parent_preflight.historic_signature(new_ctx, wrapped=False)
    h.require(old_native == new_native == old_preflight["native_first_step"]
              and old_dropout == new_dropout == old_preflight["dropout"] and len(old_dropout) == 6
              and old_native_rng == new_native_rng == h.hist_trace(3)[11], "Fold-3 native first step differs")
    old_run.restore(old_ctx, old_snap, old_hashes)
    old_out, old_diag, old_rng, old_previous, old_observed = old_run.transition(
        old_ctx, 11, old_dropout, inspect=True)
    old_step = {
        "result": local_core.h.sha(old_out),
        "losses": local_core.h.sha(old_diag["losses"]),
        "diagnostics_without_scope_receipt": local_core.h.sha(clean(old_diag)),
        "rng": old_rng,
        "previous_ema": local_core.h.sha(old_previous),
        "observed": local_core.h.sha(old_observed),
        "post_update": local_core.h.sha(old_run.bundle(old_ctx)),
        "post_ema": local_core.h.sha(old_run.ema(old_ctx)),
    }

    new_run.restore(new_ctx, new_snap, new_hashes)
    new_out, new_diag, new_rng, new_previous, new_observed = new_run.transition(
        new_ctx, 11, new_dropout, inspect=True)
    new_step = {
        "result": local_core.h.sha(new_out),
        "losses": local_core.h.sha(new_diag["losses"]),
        "diagnostics_without_scope_receipt": local_core.h.sha(clean(new_diag)),
        "rng": new_rng,
        "previous_ema": local_core.h.sha(new_previous),
        "observed": local_core.h.sha(new_observed),
        "post_update": local_core.h.sha(new_run.bundle(new_ctx)),
        "post_ema": local_core.h.sha(new_run.ema(new_ctx)),
    }
    local_core.h.require(old_step == new_step,
                         "Fold-3 epoch-11 transition differs from the E82.4.6.3 archived design")
    h.require(h.sha((old_out, old_diag, old_rng, old_step["post_ema"])) == old_preflight["first_step_sha"],
              "Fold-3 first transition differs from archived preflight")
    old_task = old_manifest["completed_folds"]["3"]
    old_best = E463 / "checkpoints/G2_f03.best.pt"
    old_latest = E463 / "checkpoints/G2_f03.latest.pt"
    old_journal = E463 / "runtime/G2_f03.jsonl"
    h.require(all(h.sha_file(path) == old_task[key + "_sha"] for key, path in
                  (("best", old_best), ("latest", old_latest), ("journal", old_journal))),
              "E82.4.6.3 Fold-3 checkpoint or journal identity differs")
    with old_journal.open() as journal:
        epoch11 = json.loads(journal.readline())
    h.require(epoch11["epoch"] == 11 and epoch11["checkpoint_state_sha"] == old_step["post_update"]
              and epoch11["probability_sha"] == h.sha(old_out["probability"])
              and epoch11["prediction_bits"] == np.packbits(old_out["prediction"]).tobytes().hex()
              and epoch11["rng"] == old_step["rng"] and epoch11["ema_before"] == old_step["previous_ema"]
              and epoch11["ema_after"] == old_step["post_ema"], "Fold-3 first formal journal entry differs")
    proof = {
        "status": "PASS",
        "source": str(E463),
        "source_manifest_sha256": local_core.h.sha_file(E463 / "manifest.json"),
        "source_best_sha256": local_core.h.sha_file(old_best),
        "source_latest_sha256": old_task["latest_sha"],
        "source_journal_sha256": old_task["journal_sha"],
        "source_audit_sha256": local_core.h.sha_file(E463 / "results/audit_status.json"),
        "local_implementation_sha256": local_core.h.sha(local_ind.implementation()),
        "train_legal_test_weights_relations_equal": True,
        "gaussian_original_o_prompt_hard_memory_equal": True,
        "model_module_and_optimizer_initial_state_equal": True,
        "E61_rng_and_six_dropout_calls_equal": True,
        "epoch11_native_and_scmr_transition_equal": True,
        "epoch11_transition_components": old_step,
        "epoch10_identities": old_identity,
        "native_first_step": old_native,
        "membership": membership,
        "gaussian": gaussian,
        "withdrawn_unmasked_1based": (withdrawn + 1).tolist(),
        "best_epoch": old_task["best_epoch"],
        "latest_epoch": old_task["epoch"],
    }
    target = ROOT / "runtime/fold03_reuse_equivalence.json"
    target.write_text(json.dumps(proof, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"Fold 3 exact equivalence PASS | Best {proof['best_epoch']} | Latest {proof['latest_epoch']}")


if __name__ == "__main__":
    main()
