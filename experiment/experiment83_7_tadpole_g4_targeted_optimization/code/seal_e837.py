from __future__ import annotations

import json
from pathlib import Path

import run_e837 as runner


def main():
    path = runner.ROOT / "audits/training_seal.json"
    if path.exists():
        raise RuntimeError("E83.7 training seal already exists")
    groups = list((runner.ROOT / "groups").glob("*/best_checkpoints/fold*.pt"))
    if groups:
        raise RuntimeError("Cannot seal after formal checkpoint output exists")
    hashes = runner.current_implementation_hashes()
    value = {
        "status": "SEALED_BEFORE_TRAINING",
        "experiment": "Experiment 83.7",
        "files": hashes,
        "implementation_sha256": runner.implementation_fingerprint(hashes),
        "file_count": len(hashes),
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": value["status"], "file_count": len(hashes),
                      "implementation_sha256": value["implementation_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
