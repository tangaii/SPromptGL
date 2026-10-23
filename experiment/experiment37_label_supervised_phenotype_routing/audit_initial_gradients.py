"""Recreate the first auxiliary-loss gradient check for every supervised fold.

This audit performs one disposable optimizer update per supervised group/fold.
It does not modify formal task checkpoints or fold metrics.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=[0, 1], required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    import numpy as np
    import torch
    import run as exp

    root = args.root.resolve()
    data = exp.data_context(root)
    network, _, _ = exp.load_modules(root)
    from integration import parameter_sha256
    splits = exp.fold_splits(data["y"])
    entries = exp.load_rng_entries()
    reference = {int(x["fold"]): x for x in exp.load_task("G00_T0_REPLAY")["metrics"]}
    rows = []
    for group in args.groups:
        if group not in exp.GROUPS:
            raise ValueError(f"unknown group: {group}")
        meta = exp.GROUPS[group]
        if meta["relation"] == "none":
            raise ValueError(f"group has no auxiliary relation objective: {group}")
        for fold, (train, test) in enumerate(splits, start=1):
            cache = exp.load_condition_cache(fold) if meta["condition"] != "none" else {}
            condition, permutation, _ = exp.fold_condition(group, cache, data, train, test, fold)
            aux_labels, _ = exp.auxiliary_train_labels(data, train, group, fold)
            exp.restore_rng(entries[fold - 1])
            agent = exp.make_agent(root, data, group, fold, train, test, condition, aux_labels)
            init_hash = parameter_sha256(agent, baseline_common_only=True)
            if init_hash != reference[fold]["initial_parameter_sha256"]:
                raise RuntimeError(f"common initialization mismatch: {group}, fold {fold}")
            step = agent.run_epoch("simple-2")
            rel = step["relation_meta"]
            rows.append({
                "config": group, "seed": 0, "fold": fold, "physical_gpu": args.gpu,
                "relation_mode": meta["relation"], "relation_loss": step["relation_loss"],
                "aux_pairs": rel.get("pairs", 0), "same_pairs": rel.get("same_pairs", 0),
                "different_pairs": rel.get("different_pairs", 0),
                "aux_query_grad_norm": rel.get("aux_query_grad_norm", 0.0),
                "aux_key_grad_norm": rel.get("aux_key_grad_norm", 0.0),
                "aux_graph_p_grad_norms": rel.get("aux_graph_p_grad_norms", []),
                "total_query_grad_norm": rel.get("query_total_grad_norm", 0.0),
                "total_key_grad_norm": rel.get("key_total_grad_norm", 0.0),
                "test_pair_count": rel.get("aux_test_pair_count", 0),
                "initial_parameter_sha256": init_hash,
                "initial_rng_sha256": exp.rng_sha(entries[fold - 1]),
            })
            for name in ("MF_sav", "GCMP_sav"):
                stream = getattr(agent, name, None)
                if stream is not None:
                    stream.close()
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["aux_graph_p_grad_norms"] = str(row["aux_graph_p_grad_norms"])
            writer.writerow(row)
    print(f"INITIAL_GRADIENT_AUDIT rows={len(rows)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
