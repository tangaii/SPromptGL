"""Read-only Fold 3 initialization, relation gradient and sampling audit."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core
from replay import E463, load_module
from independent_468 import gaussian_proof, membership_proof
import numpy as np
import torch
from torch.nn import functional as F


def main():
    h = core.h
    run = core.engine("G1", 3)
    old_run = load_module("historical_fold3_adapter", E463 / "darg_only.py").engine(3)
    ctx = run.setup("G1", 3)
    old = old_run.setup("G2", 3)
    keys = ("train", "legal", "test")
    for key in keys:
        current = ctx["members"][key]
        expected = old["members"][key]
        if key == "train":
            current = current[:-1]
        h.require(np.array_equal(current, expected), f"Historical Fold 3 ordered {key} differs")
    identity = core.ext.identity_snapshot(ctx["state"], ctx["module"], ctx["optimizer"])
    old_identity = core.ext.identity_snapshot(old["state"], old["module"], old["optimizer"])
    differences = {k for k in identity if identity[k] != old_identity[k]}
    h.require(differences == {"spectrum_and_prompts"}, f"Unexpected initial state differences {differences}")
    old["module"].spectrum.allowed.copy_(ctx["module"].spectrum.allowed)
    h.require(h.sha(old["module"].spectrum.state_dict()) == identity["spectrum_and_prompts"],
              "Spectrum or Prompt initialization differs after aligning only the permitted-row mask")
    h.require(h.sha(run.ema(ctx)) == h.sha(old_run.ema(old)), "Initial EMA differs")
    h.require(run.CFG == old_run.CFG, "Historical training configuration differs")
    pre = h.read_json(ROOT / "runtime/G1_f03.preflight.json")
    h.require(h.rng_sha() == pre["first_rng"]["rng_before"], "Initial RNG differs")
    membership = membership_proof(ctx)
    gaussian_proof(ctx)
    state = ctx["state"]
    h.require(int(state["labels_np"][412]) == 0 and int(ctx["members"]["train"][-1]) == 412
              and 412 in ctx["members"]["test"], "Patient 413 label or membership differs")
    training = h.read_json(ROOT / "runtime/G1_f03.training.json")
    best = h.load(ROOT / "checkpoints/G1_f03.best.pt")
    run.restore(ctx, best, training["implementation"])
    with run.dropout_inventory() as calls:
        output, cap = run.forward(state, True, ctx["module"])
    h.require(calls == pre["dropout"] and len(calls) == 6, "Historical dropout calls differ")
    rng = h.rng_sha()
    pair_logits = tuple(output["info"]["pair_logits"])
    pair = torch.stack([F.binary_cross_entropy_with_logits(logit, state["train_same"])
                        for logit in pair_logits]).mean()
    h.require(torch.equal(pair, output["pair"]), "Relation loss differs")
    pair_gradient = torch.autograd.grad(output["pair"], pair_logits, retain_graph=True)
    classification_gradient, = torch.autograd.grad(output["cls"], output["logp"], retain_graph=True)
    h.require(torch.count_nonzero(classification_gradient.norm(dim=1)).item() == 790,
              "Classification loss did not reach all 790 patients")
    endpoint_ids = np.unique(ctx["members"]["train_pairs"])
    unsampled = np.setdiff1d(ctx["members"]["train"], endpoint_ids) + 1
    per_patient = {}
    for row in core.EXPOSED[3]:
        select = (state["train_pairs"] == row).any(dim=1)
        per_patient[str(row + 1)] = {
            "pair_rows": int(select.sum()),
            "relation_logit_grad_l2_sum": sum(float(g[select].norm()) for g in pair_gradient),
            "classification_logp_grad_l2": float(classification_gradient[row].norm()),
        }
    h.require(per_patient["413"]["relation_logit_grad_l2_sum"] > 0
              and all(g.norm().item() > 0 for g in pair_gradient)
              and h.rng_sha() == rng, "Pair gradient or audit RNG differs")
    result = {
        "status": "PASS", "epoch10_identity": identity,
        "historical_e463_membership_prefix_exact": True,
        "historical_e463_initial_ema_and_training_config_exact": True,
        "dropout": calls, "classification_rows_with_gradient": 790,
        "relation_eligible_rows": len(ctx["members"]["train"]),
        "relation_sampled_unique_endpoint_rows": len(endpoint_ids),
        "relation_unsampled_ids_1based": unsampled.tolist(),
        "overlap_gradient": per_patient, "membership": membership,
        "diagnostics_rng_unchanged": True,
    }
    h.write_json(ROOT / "runtime/fold3_scope_verification.json", result)
    print("Fold 3 exact initial state and gradients PASS. Pair endpoint coverage %s/790." % len(endpoint_ids))


if __name__ == "__main__":
    main()
