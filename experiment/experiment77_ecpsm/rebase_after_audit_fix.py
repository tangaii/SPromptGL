#!/usr/bin/env python3
"""Rebase metadata after a report-only final audit script correction."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

import run


ROOT = Path(__file__).resolve().parent
PREFLIGHT = ROOT / "results" / "preflight.json"


def atomic_torch(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + f".rebase.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + f".rebase.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main() -> None:
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    old_sha = preflight["source_sha256"]
    old_hashes = preflight["source_hashes"]
    hashes = run.source_manifest()
    source_sha = run.digest_value(hashes)
    checkpoints = sorted(path for path in (ROOT / "checkpoints").glob("E77-*_f*.pt")
                        if path.name.endswith((".best.pt", ".latest.pt")))
    if len([p for p in checkpoints if p.name.endswith(".best.pt")]) != 20:
        raise RuntimeError("expected twenty final best checkpoints")
    if len([p for p in checkpoints if p.name.endswith(".latest.pt")]) != 20:
        raise RuntimeError("expected twenty recovery checkpoints")

    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("source_sha256") != old_sha:
            raise RuntimeError(f"unexpected pre-fix source SHA in {path.name}")
        preserved = {key: value for key, value in payload.items() if key != "source_sha256"}
        preserved_sha = run.digest_value(preserved)
        payload["source_sha256"] = source_sha
        atomic_torch(path, payload)
        reread = torch.load(path, map_location="cpu", weights_only=False)
        reread_preserved = {key: value for key, value in reread.items() if key != "source_sha256"}
        if run.digest_value(reread_preserved) != preserved_sha:
            raise RuntimeError(f"non-metadata checkpoint state changed: {path.name}")

    tasks = sorted((ROOT / "runtime" / "tasks").glob("E77-*_f*.json"))
    if len(tasks) != 20:
        raise RuntimeError(f"expected twenty complete task records, found {len(tasks)}")
    for path in tasks:
        task = json.loads(path.read_text(encoding="utf-8"))
        if task.get("status") != "COMPLETE" or task.get("source_sha256") != old_sha:
            raise RuntimeError(f"unexpected pre-fix task metadata: {path.name}")
        best = ROOT / "checkpoints" / f"{task['group']}_f{int(task['fold']):02d}.best.pt"
        task["source_sha256"] = source_sha
        task["checkpoint_sha256"] = run.sha_file(best)
        atomic_json(path, task)

    preflight["source_sha_before_report_fix"] = old_sha
    preflight["source_hashes_before_report_fix"] = old_hashes
    preflight["source_sha256"] = source_sha
    preflight["source_hashes"] = hashes
    preflight["protected_sources_unchanged_baseline"] = hashes
    preflight["post_training_audit_fix"] = {
        "change": "report-only fixes for nine-fold base reference, paired group deltas, and memory audit counts",
        "old_source_sha256": old_sha,
        "new_source_sha256": source_sha,
        "checkpoint_tensor_optimizer_memory_rng_payloads_unchanged": True,
        "rebased_best_and_latest_checkpoints": len(checkpoints),
        "rebased_complete_tasks": len(tasks),
        "training_repeated": False,
    }
    atomic_json(PREFLIGHT, preflight)
    print(json.dumps({"status": "PASS", "old_source_sha256": old_sha,
                      "new_source_sha256": source_sha,
                      "rebased_checkpoints": len(checkpoints), "rebased_tasks": len(tasks)},
                     ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
