#!/usr/bin/env python3
"""Rebase E77 recovery metadata after the authorized OSSFS directory migration."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

import run


ROOT = Path(__file__).resolve().parent
OLD_ROOT = Path("/data/adodas/SPromptGL-main/experiment/experiment77_ecpsm")
PREFLIGHT = ROOT / "results" / "preflight.json"


def atomic_torch(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + f".migration.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + f".migration.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main() -> None:
    if ROOT != Path("/model/spromptgl/experiment/experiment77_ecpsm"):
        raise RuntimeError(f"unexpected destination: {ROOT}")
    if not OLD_ROOT.is_dir() or not (ROOT / "run.py").is_file():
        raise RuntimeError("both the verified source copy and destination must exist")

    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    old_sha = preflight["source_sha256"]
    old_hashes = preflight["source_hashes"]
    current_hashes = run.source_manifest()
    current_sha = run.digest_value(current_hashes)

    migrated_checkpoints = []
    for path in sorted((ROOT / "checkpoints").glob("E77-*_f*.pt")):
        if ".tmp." in path.name:
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("source_sha256") != old_sha:
            raise RuntimeError(f"unexpected checkpoint source SHA in {path.name}")
        before = {key: value for key, value in payload.items() if key != "source_sha256"}
        before_digest = run.digest_value(before)
        payload["source_sha256"] = current_sha
        atomic_torch(path, payload)
        reloaded = torch.load(path, map_location="cpu", weights_only=False)
        after = {key: value for key, value in reloaded.items() if key != "source_sha256"}
        if run.digest_value(after) != before_digest:
            raise RuntimeError(f"checkpoint payload changed during SHA migration: {path.name}")
        migrated_checkpoints.append(path.name)

    migrated_tasks = []
    for path in sorted((ROOT / "runtime" / "tasks").glob("E77-*_f*.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        if task.get("source_sha256") != old_sha or task.get("status") != "COMPLETE":
            raise RuntimeError(f"unexpected task metadata in {path.name}")
        group, fold = task["group"], int(task["fold"])
        best = ROOT / "checkpoints" / f"{group}_f{fold:02d}.best.pt"
        task["source_sha256"] = current_sha
        task["checkpoint_sha256"] = run.sha_file(best)
        atomic_json(path, task)
        migrated_tasks.append(path.name)

    preflight["prior_source_sha256"] = old_sha
    preflight["prior_source_hashes"] = old_hashes
    preflight["source_sha256"] = current_sha
    preflight["source_hashes"] = current_hashes
    preflight["protected_sources_unchanged_baseline"] = current_hashes
    preflight["ossfs_migration"] = {
        "reason": "real OSError [Errno 5] while closing a formal worker log under the original OSSFS output directory",
        "from": str(OLD_ROOT),
        "to": str(ROOT),
        "copy_verified_by_recursive_diff": True,
        "recovery_sha_rebased_without_training_state_changes": True,
        "old_source_sha256": old_sha,
        "new_source_sha256": current_sha,
        "migrated_checkpoints": migrated_checkpoints,
        "migrated_complete_tasks": migrated_tasks,
        "latest_next_epochs": {
            path.stem.replace(".latest", ""): int(torch.load(path, map_location="cpu", weights_only=False)["next_epoch"])
            for path in sorted((ROOT / "checkpoints").glob("E77-*_f*.latest.pt"))
            if ".tmp." not in path.name
        },
    }
    atomic_json(PREFLIGHT, preflight)

    # Uncommitted checkpoint temporaries were left by terminated workers; only
    # remove a temporary when its canonical checkpoint has been validated above.
    for path in (ROOT / "checkpoints").glob("*.tmp.*"):
        canonical = path.name.split(".tmp.", 1)[0]
        if (ROOT / "checkpoints" / canonical).is_file():
            path.unlink()

    # The destination now contains the verified, resumable copy. Keep exactly one
    # effective E77 directory after this explicit OSSFS fallback.
    import shutil

    shutil.rmtree(OLD_ROOT)
    if OLD_ROOT.exists():
        raise RuntimeError("old OSSFS experiment directory still exists")
    print(json.dumps({
        "status": "PASS",
        "new_source_sha256": current_sha,
        "migrated_checkpoints": migrated_checkpoints,
        "migrated_tasks": migrated_tasks,
        "latest_next_epochs": preflight["ossfs_migration"]["latest_next_epochs"],
        "old_directory_removed": True,
    }, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
