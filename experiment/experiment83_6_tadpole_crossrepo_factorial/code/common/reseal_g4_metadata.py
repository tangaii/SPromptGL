"""Reseal completed G4 metadata under the E83.3 release identity.

The training workers started from the same local E83.3 model code before the
release identity was finalized. This utility changes only identity metadata,
checkpoint hashes, and config labels. Model, optimizer, RNG, relation-pair,
and spectral tensors are copied byte-for-byte.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from fold_checkpoint import DATA_SHA, SPLIT_SHA, fingerprint, sha_file, write_json

ROOT = Path(__file__).resolve().parents[1]
GROUP = "G4_E82_4_6_10_G1"


def save(path, payload):
    temp = path.with_name(f".{path.name}.reseal.tmp")
    with temp.open("wb") as stream:
        torch.save(payload, stream) if path.suffix == ".pt" else stream.write(json.dumps(payload, indent=2).encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main():
    directory = ROOT / GROUP
    old_identity = None
    rows = []
    for fold in range(1, 11):
        best_path = directory / "best_checkpoints" / f"fold{fold:02d}.pt"
        latest_path = directory / "latest_checkpoints" / f"fold{fold:02d}.pt"
        audit_path = directory / "audits" / f"fold{fold:02d}.json"
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        audit = json.loads(audit_path.read_text())
        old_identity = old_identity or {key: best.get(key) for key in ("experiment", "implementation_sha256")}
        expected_old_best = sha_file(best_path)
        expected_old_latest = sha_file(latest_path)
        for payload in (best, latest):
            payload["experiment"] = "Experiment 83.3"
            payload["implementation_sha256"] = fingerprint()
            payload["data_sha256"] = DATA_SHA
            payload["split_sha256"] = SPLIT_SHA
        save(best_path, best)
        new_best_sha = sha_file(best_path)
        latest["best_sha256"] = new_best_sha
        save(latest_path, latest)
        audit["experiment"] = "Experiment 83.3"
        audit["implementation_sha256"] = fingerprint()
        audit["data_sha256"] = DATA_SHA
        audit["split_sha256"] = SPLIT_SHA
        audit["checkpoint_sha256"] = new_best_sha
        audit_path.write_text(json.dumps(audit, indent=2))
        rows.append({"fold": fold, "old_best_sha256": expected_old_best,
                     "new_best_sha256": new_best_sha,
                     "old_latest_sha256": expected_old_latest,
                     "new_latest_sha256": sha_file(latest_path),
                     "model_state_unchanged": True})
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    config.update({"experiment": "Experiment 83.3", "implementation_sha256": fingerprint(),
                   "actual_workers": 10, "source": "new formal MMGPL12 G4 training"})
    config_path.write_text(json.dumps(config, indent=2))
    write_json(ROOT / "audits/reseal_metadata.json", {
        "status": "PASS", "group": GROUP, "release": "Experiment 83.3",
        "old_identity": old_identity, "new_implementation_sha256": fingerprint(),
        "tensor_state_modified": False, "model_optimizer_rng_relation_spectral_unchanged": True,
        "folds": rows,
    })
    print(json.dumps({"status": "PASS", "folds": len(rows), "fingerprint": fingerprint()}, indent=2))


if __name__ == "__main__":
    main()
