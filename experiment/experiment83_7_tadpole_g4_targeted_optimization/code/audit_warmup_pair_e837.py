"""Verify the paired T3/T4 graph-supervision warmups fold by fold."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import run_e837 as runner
from fold_checkpoint import write_json


def main():
    records = []
    for fold in range(1, 11):
        paths = {
            group: runner.ROOT / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt"
            for group in ("T3", "T4")
        }
        payloads = {group: torch.load(path, map_location="cpu", weights_only=False)
                    for group, path in paths.items()}
        left, right = payloads["T3"], payloads["T4"]
        left_warm, right_warm = left["warmup"], right["warmup"]
        compared = ("initial_model_sha256", "initial_optimizer_sha256", "initial_rng_sha256",
                    "base_state_sha256", "optimizer_state_sha256", "rng_state_sha256",
                    "pair_sha256", "train_index_sha256", "train_labels_sha256")
        differences = [key for key in compared if left_warm.get(key) != right_warm.get(key)]
        if left.get("train_idx") != right.get("train_idx") or left.get("test_idx") != right.get("test_idx"):
            differences.append("fold_indices")
        if not np.array_equal(left["relation_pairs"].cpu(), right["relation_pairs"].cpu()):
            differences.append("relation_pairs")
        if not torch.equal(left["relation_targets"].cpu(), right["relation_targets"].cpu()):
            differences.append("relation_targets")
        left_history = left_warm.get("history", [])
        right_history = right_warm.get("history", [])
        if left_history != right_history:
            differences.append("history")
        if not all(float(item.get("graph_supervision", 0.0)) > 0 for item in left_history):
            differences.append("T3_graph_supervision_nonzero")
        if not all(float(item.get("graph_supervision", 0.0)) > 0 for item in right_history):
            differences.append("T4_graph_supervision_nonzero")
        for group, payload in payloads.items():
            audit = json.loads((runner.ROOT / "groups" / group / "audits" / f"fold{fold:02d}.json").read_text())
            if audit.get("formal_graph_supervision_enabled") is not False or audit.get("graph_supervision_final") != 0.0:
                differences.append(f"{group}_formal_graph_supervision")
        records.append({"fold": fold, "status": "PASS" if not differences else "FAIL",
                        "field_differences": differences,
                        "initial_model_sha256": left_warm["initial_model_sha256"],
                        "post_warmup_model_sha256": left_warm["base_state_sha256"],
                        "post_warmup_optimizer_sha256": left_warm["optimizer_state_sha256"],
                        "post_warmup_rng_sha256": left_warm["rng_state_sha256"],
                        "graph_supervision_epochs": len(left_history)})
    value = {"experiment": "Experiment 83.7", "status": "PASS" if all(
        item["status"] == "PASS" for item in records) else "FAIL", "folds": len(records),
             "paired_passes": sum(item["status"] == "PASS" for item in records),
             "records": records}
    write_json(runner.ROOT / "audits/warmup_pairing.json", value)
    print(json.dumps({key: value[key] for key in ("status", "folds", "paired_passes")}, indent=2))
    if value["status"] != "PASS":
        raise RuntimeError("T3/T4 paired warmup verification failed")


if __name__ == "__main__":
    main()
