"""Release identity and atomic fold checkpoints."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os

import torch

from source_runtime import ROOT


DATA_SHA = "0b8a14a9c066752c2d937c93d43c2824c41beaaa7d0a451b083db8dcba07c5bc"
SPLIT_SHA = "7c5a4dc9bd8ee7ffc6e5aff8bf841513f26a5ff6cf1b6c110a8153aa6a106ec5"
TRAINING_MODULES = (
    "fidelity_model.py", "fold_checkpoint.py", "metrics.py", "run_e833.py",
    "cam_only.py", "run_g5.py", "scmr_optimization.py", "scmr_runtime.py", "source_runtime.py",
)


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_hashes():
    paths = tuple(sorted((ROOT / "common").glob("*.py")))
    return {str(path.relative_to(ROOT)): sha_file(path) for path in paths}


def fingerprint():
    source_hashes = implementation_hashes()
    training_hashes = {name: source_hashes[f"common/{name}"] for name in TRAINING_MODULES}
    return hashlib.sha256(json.dumps(training_hashes, sort_keys=True).encode()).hexdigest()


@contextmanager
def lock(path):
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def atomic_write(path, callback):
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("wb") as stream:
            callback(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, lambda stream: stream.write(json.dumps(value, indent=2, allow_nan=False).encode()))


def write_checkpoint(path, value):
    atomic_write(path, lambda stream: torch.save(value, stream))


def identity(group, fold, train_rows, test_rows):
    prompt_path = (ROOT / "inputs/prompts/mmGPL12_scmr.pt" if group in ("G1_E28_E0", "G4_E82_4_6_10_G1")
                   else ROOT / "sources/E82_4_6_G2_SCMR_DA_SRPE/inputs/prompts/prompts.pt")
    protocol_path = ROOT / "protocol.json"
    config_path = ROOT / "configs" / f"{group}.json"
    return {
        "experiment": "Experiment 83.5", "group": group, "fold": fold, "seed": 0,
        "model_type": "FidelityModel", "implementation_sha256": fingerprint(),
        "data_sha256": DATA_SHA, "split_sha256": SPLIT_SHA,
        "protocol_sha256": sha_file(protocol_path), "prompt_sha256": sha_file(prompt_path),
        "source_manifest_sha256": sha_file(ROOT / "sources/source_manifest.json"),
        "prompt_manifest_sha256": sha_file(ROOT / "prompt_manifest.json"),
        "group_config_sha256": sha_file(config_path),
        "train_idx": train_rows.tolist(), "test_idx": test_rows.tolist(),
    }


def validate_identity(payload, expected):
    differences = [key for key, value in expected.items() if payload.get(key) != value]
    if differences:
        raise RuntimeError(f"Checkpoint has a different fold release: {', '.join(differences)}")


def recover_best(path, expected_sha):
    """Finish a committed Best promotion left between two atomic renames."""
    previous = path.with_suffix(".pending")
    if path.is_file() and sha_file(path) == expected_sha:
        previous.unlink(missing_ok=True)
    elif previous.is_file() and sha_file(previous) == expected_sha:
        os.replace(previous, path)
    elif expected_sha is not None or path.exists():
        raise RuntimeError(f"No Best checkpoint matches Latest for {path.name}")
    else:
        previous.unlink(missing_ok=True)


def publish_epoch(latest, best_path, state, new_best):
    if new_best is not None:
        pending = best_path.with_suffix(".pending")
        write_checkpoint(pending, new_best)
        state["best_sha256"] = sha_file(pending)
    write_checkpoint(latest, state)
    if new_best is not None:
        os.replace(pending, best_path)
    return state["best_sha256"]
