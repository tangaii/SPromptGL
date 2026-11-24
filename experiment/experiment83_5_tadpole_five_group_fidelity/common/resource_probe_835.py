"""Compare 1, 2, 4, 8, and 10 concurrent G4 training-step workers."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import time

import torch

from fold_checkpoint import ROOT, atomic_write


def worker_probe():
    from preflight import gpu_runtime_probe, load_data

    torch.set_num_threads(1)
    x, y, splits = load_data()
    train = torch.as_tensor(splits[0]["train_idx"], dtype=torch.long)
    return gpu_runtime_probe(x, y, train)


def measure(workers: int):
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    started = time.perf_counter()
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(worker_probe) for _ in range(workers)]
        values = [future.result() for future in futures]
    seconds = time.perf_counter() - started
    peaks = [value["peak_reserved_mib"] for value in values]
    return {"workers": workers, "status": "PASS", "wall_seconds": seconds,
            "aggregate_steps_per_second": workers / seconds,
            "peak_reserved_mib_each": peaks, "summed_peak_reserved_mib": sum(peaks),
            "device": values[0]["device"], "steps_finite": all(
                value["all_pair_training_objectives_finite"] for value in values)}


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E83.5 resource probe requires the assigned single GPU")
    total_memory = torch.cuda.get_device_properties(0).total_memory / 2**20
    measurements = []
    for workers in (1, 2, 4, 8, 10):
        try:
            item = measure(workers)
            item["memory_safe"] = item["summed_peak_reserved_mib"] < 0.8 * total_memory
            measurements.append(item)
            print(json.dumps(item), flush=True)
        except Exception as exc:
            measurements.append({"workers": workers, "status": "FAIL",
                                 "memory_safe": False, "error": repr(exc)})
            print(json.dumps(measurements[-1]), flush=True)
            torch.cuda.empty_cache()
    safe = [item for item in measurements if item["status"] == "PASS" and item["memory_safe"]]
    if not safe:
        raise RuntimeError("No concurrency setting passed throughput and memory checks")
    selected = max(safe, key=lambda item: item["aggregate_steps_per_second"])
    one = next(item for item in measurements if item["workers"] == 1)
    two = next(item for item in measurements if item["workers"] == 2)
    result = {
        "experiment": "Experiment 83.5", "status": "PASS",
        "device": selected["device"], "device_total_memory_mib": total_memory,
        "worker_processes_tested": [1, 2, 4, 8, 10], "measurements": measurements,
        "selected_gpu_workers": selected["workers"],
        "selection_rule": "Highest measured aggregate full G4 training-step throughput among safe successful settings. Safe requires summed peak reserved memory below 80 percent of total GPU memory.",
        "single_process_wall_seconds": one["wall_seconds"],
        "two_process_wall_seconds": two.get("wall_seconds"),
        "single_process_steps_per_second": one["aggregate_steps_per_second"],
        "two_process_steps_per_second": two.get("aggregate_steps_per_second"),
        "two_vs_one_throughput_ratio": (two["aggregate_steps_per_second"] / one["aggregate_steps_per_second"]
                                         if two.get("status") == "PASS" else None),
        "single_process_peak_reserved_mib": one["peak_reserved_mib_each"][0],
        "two_process_peak_reserved_mib_each": two.get("peak_reserved_mib_each", []),
        "two_process_summed_peak_reserved_mib": two.get("summed_peak_reserved_mib"),
        "selected_aggregate_steps_per_second": selected["aggregate_steps_per_second"],
        "formal_training_folds_touched": 0,
    }
    path = ROOT / "audits/resource_probe.json"
    atomic_write(path, lambda stream: stream.write(json.dumps(result, indent=2, allow_nan=False).encode()))
    print(json.dumps({"status": "PASS", "selected_gpu_workers": result["selected_gpu_workers"],
                      "selected_aggregate_steps_per_second": result["selected_aggregate_steps_per_second"],
                      "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
