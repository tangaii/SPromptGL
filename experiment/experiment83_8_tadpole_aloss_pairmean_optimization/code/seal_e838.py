"""Freeze E83.8 implementation, inputs, and configurations before training."""
from __future__ import annotations

import json

import run_e838 as runner


def main():
    path = runner.ROOT / "audits/training_seal.json"
    if path.exists():
        raise RuntimeError("An E83.8 source seal already exists")
    checkpoint_files = [file for group in (runner.ROOT / "groups").iterdir()
                        for kind in ("best_checkpoints", "latest_checkpoints")
                        for file in (group / kind).glob("*.pt")]
    if checkpoint_files:
        raise RuntimeError("Cannot freeze E83.8 after a formal checkpoint has been written")
    hashes = runner.current_implementation_hashes()
    value = {
        "status": "SEALED_BEFORE_TRAINING", "experiment": "Experiment 83.8",
        "files": hashes, "implementation_sha256": runner.implementation_fingerprint(hashes),
        "file_count": len(hashes), "data_sha256": runner.DATA_SHA,
        "split_sha256": runner.SPLIT_SHA,
        "groups": [f"X{number}" for number in range(1, 7)],
        "registered_formal_folds": 60,
    }
    runner.write_json(path, value)
    print(json.dumps({key: value[key] for key in
                      ("status", "file_count", "implementation_sha256", "registered_formal_folds")},
                     indent=2))


if __name__ == "__main__":
    main()
