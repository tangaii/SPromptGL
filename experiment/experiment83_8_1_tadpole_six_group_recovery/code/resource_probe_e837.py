"""Probe the registered eight-worker GPU concurrency before formal folds."""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))
import run_e837 as runner


def probe_worker(worker_id: int) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    runner.configure_worker()
    x_np, y_np, _, splits = runner.load_data()
    split = splits[worker_id]
    config = runner.group_config("T3")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    runner.historical.seed_everything(0)
    x = torch.from_numpy(x_np).to(device)
    labels = torch.as_tensor(y_np, device=device).long()
    rows = torch.as_tensor(split["train_idx"], device=device).long()
    model = runner.build_model(config, int(split["fold"]), device)
    optimizers, _ = runner.make_optimizers(model, config)
    start = time.monotonic()
    runner.zero_optimizers(optimizers)
    _, aux = model(x, labels, rows, darg_warmup=True)
    _, total, graph_sup, _ = runner.historical_loss(aux, labels, rows, model, True)
    if not torch.isfinite(total) or float(graph_sup.detach()) <= 0:
        raise RuntimeError("Eight-worker probe received an invalid graph-supervision loss")
    total.backward()
    runner.step_optimizers(optimizers)
    torch.cuda.synchronize(device)
    return {
        "worker_id": worker_id, "fold": int(split["fold"]),
        "loss": float(total.detach()), "graph_supervision": float(graph_sup.detach()),
        "seconds": time.monotonic() - start,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "device": torch.cuda.get_device_name(device), "finite": True,
    }


def main() -> None:
    seal_path = ROOT / "audits/training_seal.json"
    preflight_path = ROOT / "audits/preflight_start.json"
    if not seal_path.is_file() or not preflight_path.is_file():
        raise RuntimeError("The source seal and method preflight must pass before the resource probe")
    preflight = json.loads(preflight_path.read_text())
    if (preflight.get("status") != "PASS" or preflight.get("formal_training_allowed") is not True
            or preflight.get("training_seal_sha256") != runner.sha_file(seal_path)):
        raise RuntimeError("Resource probe does not match the passing source preflight")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Resource probe requires exactly one assigned GPU")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    records = []
    started = time.monotonic()
    with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn")) as pool:
        tasks = [pool.submit(probe_worker, worker_id) for worker_id in range(8)]
        for future in as_completed(tasks):
            record = future.result()
            records.append(record)
            print(f"RESOURCE worker={record['worker_id']} fold={record['fold']:02d} "
                  f"seconds={record['seconds']:.2f} peak_reserved={record['peak_reserved_mib']:.1f}MiB",
                  flush=True)
    records.sort(key=lambda row: row["worker_id"])
    value = {
        "experiment": "Experiment 83.7", "status": "PASS" if len(records) == 8 else "FAIL",
        "worker_count": len(records), "global_worker_limit": 8, "device": records[0]["device"],
        "training_seal_sha256": runner.sha_file(seal_path),
        "concurrent_probe_seconds": time.monotonic() - started,
        "peak_reserved_mib_max": max(row["peak_reserved_mib"] for row in records),
        "peak_allocated_mib_max": max(row["peak_allocated_mib"] for row in records),
        "records": records,
    }
    (ROOT / "audits/resource_probe.json").write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: value[key] for key in ("status", "worker_count", "device",
                                                 "concurrent_probe_seconds", "peak_reserved_mib_max")}, indent=2))
    if value["status"] != "PASS":
        raise RuntimeError("The registered global eight-worker GPU resource probe failed")


if __name__ == "__main__":
    main()
