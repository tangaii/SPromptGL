#!/usr/bin/env python3
"""Read-only site/label enrichment audit for the learned sender-restricted graphs."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run


def main():
    c = run.load_context(); x, y, folds = run.load_data(); frame = pd.read_csv(run.DATA_CSV)
    mapping = np.load(run.DATA_MAP, allow_pickle=True).item()
    site_idx = [i for i, name in enumerate(mapping["PHENO"]) if "SITE_ID_" in str(name)]
    site = x[:, site_idx].argmax(1)
    rows = []
    for gid in (2, 5, 6):
        for fold, (train, test) in enumerate(folds, 1):
            if gid == 0:
                continue
            task = json.loads((run.RT / "tasks" / f"g{gid}_f{fold:02d}.json").read_text())
            best = run.load_torch(run.RT / "checkpoints" / f"g{gid}_f{fold:02d}.best.pt")
            anchor = run.load_torch(run.RT / "anchors" / f"fold{fold:02d}.pt")
            a = run.new_agent(x, y, train, test, fold, gid, anchor)
            run.load_bundle(a, best["model"])
            with c["torch"].no_grad(): out = c["method"].evaluate(a, gid, best["best_epoch"], c["utils"])
            for i, info in enumerate(out["meta"]):
                sender = info["sender"].cpu().numpy()
                receiver = np.arange(len(sender))[:, None]
                edges = sender.reshape(-1)
                recv = np.repeat(np.arange(len(sender)), sender.shape[1])
                same_site = float(np.mean(site[recv] == site[edges]))
                same_label = float(np.mean(y[recv] == y[edges]))
                train_sender = np.isin(edges, train)
                cross_site = float(np.mean(site[recv][train_sender] != site[edges][train_sender]))
                rows.append(dict(group_id=gid, group=run.GROUP_NAMES[gid], fold=fold,
                                 relation=["PHENO", "ANAT", "FUNC", "CORR"][i],
                                 edge_count=len(edges), same_site=same_site,
                                 same_label=same_label, cross_site=cross_site,
                                 self_edges=int(np.sum(recv == edges))))
    out = HERE / "results/site_diagnostics.csv"
    fields = list(rows[0])
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print("SITE_DIAGNOSTICS", len(rows), out)


if __name__ == "__main__": main()
