"""Independently evaluate the historical Fold 2 checkpoint in read-only mode."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core
from replay import E464, BASE, load_module, signed_margin
import numpy as np
import torch


def main():
    h = core.h
    adapter = load_module("fold2_source_adapter", E464 / "core.py")
    run = adapter.engine("G1", 2)
    proof = load_module("fold2_source_audit", BASE / "audit.py")
    proof.run = run
    ctx = run.setup("G1", 2)
    manifest = h.read_json(E464 / "manifest.json")
    checkpoint_path = E464 / "checkpoints/G1_f02.best.pt"
    best = h.load(checkpoint_path)
    run.restore(ctx, best, manifest["implementation"])
    rng = h.rng_sha()
    full, capture = proof.independent_eval(ctx, "full")
    off, _ = proof.independent_eval(ctx, "residual_off")
    anonymous, _ = proof.independent_eval(ctx, "anonymous_query")
    proof.match_eval(full, best["evaluation"], "Historical Fold 2 Best")
    h.require(h.rng_sha() == rng, "Fold 2 evaluation consumed RNG")

    members, state = ctx["members"], ctx["state"]
    test = members["test"]
    labels = state["labels_np"][test]
    overlap = np.intersect1d(members["train"], test)
    expected = [259, 287, 374, 376, 406, 680, 778, 783]
    related = np.asarray(members["train_pairs"])
    allowed = ctx["module"].spectrum.allowed
    h.require(overlap.tolist() == expected and len(members["train"]) == 792
              and len(np.unique(members["train"])) == 792 and len(test) == 87
              and len(members["legal"]) == 784
              and np.array_equal(ctx["reference"]["legal"], members["legal"])
              and np.array_equal(ctx["module"].statistics.legal.cpu().numpy(), members["legal"])
              and torch.where(~allowed)[0].tolist() == expected
              and 539 in test and 539 not in members["train"] and 539 not in related
              and allowed[539].item(), "Restored Fold 2 membership differs")
    details = capture["details"]
    for key in core.MASKED_COMPONENTS:
        h.require(torch.count_nonzero(details[key][expected]).item() == 0, "Fold 2 mask differs")
    h.require(torch.equal(capture["enhanced"][expected], capture["cam"][expected])
              and torch.count_nonzero(details["delta"][539]).item() > 0,
              "Fold 2 feature scope differs")

    values = {
        "status": "PASS", "fold": 2, "checkpoint": str(checkpoint_path),
        "checkpoint_sha": h.sha_file(checkpoint_path), "epoch": int(best["epoch"]),
        "source_manifest_sha": h.sha_file(E464 / "manifest.json"),
        "test_indices_0based": test.tolist(), "labels": labels.tolist(),
        "darg_train_count": len(members["train"]), "legal_count": len(members["legal"]),
        "train_sha": h.sha(members["train"]), "legal_sha": h.sha(members["legal"]),
        "relation_pair_sha": h.sha(related), "overlap_ids": (overlap + 1).tolist(),
        "patient540_in_train_or_relation_pairs": False, "patient540_residual_allowed": True,
        "patient540_joint_residual_l2": float(details["delta"][539].double().norm()),
        "masked_residual_cam_exact": True, "rng_unchanged": True,
    }
    for key, output in (("full", full), ("off", off), ("anonymous", anonymous)):
        values[key] = {k: np.asarray(v).tolist() for k, v in output.items()}
        values["margin_" + key] = signed_margin(output["logits"], labels).tolist()
    h.write_json(ROOT / "runtime/independent_fold_02.json", values)
    print(f"Fold 2 Best {values['epoch']} read-only replay PASS. Correct {full['correct']}/87.")


if __name__ == "__main__":
    main()
