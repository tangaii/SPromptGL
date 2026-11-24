"""Create the immutable pre-training source and input fingerprint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_e836 import ROOT, current_implementation_hashes, implementation_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace-before-training", action="store_true")
    args = parser.parse_args()
    path = ROOT / "audits/training_seal.json"
    groups = ROOT / "groups"
    if path.exists() and not args.replace_before_training:
        raise RuntimeError("Training seal already exists. Use the explicit pre-training replacement flag only before any fold output exists.")
    if args.replace_before_training:
        existing = list(groups.glob("*/best_checkpoints/fold*.pt"))
        if existing:
            raise RuntimeError("Cannot replace a training seal after formal Best checkpoints exist")
    hashes = current_implementation_hashes()
    value = {
        "status": "SEALED_BEFORE_TRAINING",
        "experiment": "Experiment 83.6",
        "files": hashes,
        "implementation_sha256": implementation_fingerprint(hashes),
        "file_count": len(hashes),
        "replacement": bool(args.replace_before_training),
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": value["status"], "file_count": len(hashes),
                      "implementation_sha256": value["implementation_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
