"""Measure one- and two-process GPU training-step throughput before release."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fold_checkpoint import ROOT, atomic_write


CHILD = """
import json
import torch
from preflight import gpu_runtime_probe, load_data
x, y, splits = load_data()
result = gpu_runtime_probe(x, y, torch.tensor(splits[0]['train_idx'], dtype=torch.long))
print(json.dumps(result, allow_nan=False))
"""


def run_probe():
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    command = [sys.executable, "-B", "-c", CHILD]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT / "common", env=env, capture_output=True,
                               text=True, check=False)
    seconds = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f"Single-process GPU probe failed: {completed.stderr[-4000:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1]), seconds


def run_parallel():
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    command = [sys.executable, "-B", "-c", CHILD]
    processes = [subprocess.Popen(command, cwd=ROOT / "common", env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for _ in range(2)]
    started = time.perf_counter()
    results = []
    for process in processes:
        stdout, stderr = process.communicate()
        if process.returncode:
            raise RuntimeError(f"Concurrent GPU probe failed: {stderr[-4000:]}")
        results.append(json.loads(stdout.strip().splitlines()[-1]))
    seconds = time.perf_counter() - started
    return results, seconds


def main():
    single, single_wall = run_probe()
    concurrent, concurrent_wall = run_parallel()
    single_throughput = 1.0 / single_wall
    concurrent_throughput = 2.0 / concurrent_wall
    throughput_ratio = concurrent_throughput / single_throughput
    total_memory_mib = 96 * 1024
    summed_peak_mib = sum(item["peak_reserved_mib"] for item in concurrent)
    selected_workers = 2 if throughput_ratio > 1.05 and summed_peak_mib < .8 * total_memory_mib else 1
    result = {
        "status": "PASS", "device": single["device"], "device_total_memory_mib": total_memory_mib,
        "worker_processes_tested": [1, 2], "single_process_wall_seconds": single_wall,
        "two_process_wall_seconds": concurrent_wall,
        "single_process_steps_per_second": single_throughput,
        "two_process_steps_per_second": concurrent_throughput,
        "two_vs_one_throughput_ratio": throughput_ratio,
        "single_process_peak_reserved_mib": single["peak_reserved_mib"],
        "two_process_peak_reserved_mib_each": [item["peak_reserved_mib"] for item in concurrent],
        "two_process_summed_peak_reserved_mib": summed_peak_mib,
        "selected_gpu_workers": selected_workers,
        "selection_rule": "Use two workers only when measured aggregate throughput improves by more than 5 percent and summed peak reservation stays below 80 percent of device memory.",
        "single_probe": single, "parallel_probes": concurrent,
        "formal_training_folds_touched": 0,
    }
    path = ROOT / "audits/resource_probe.json"
    atomic_write(path, lambda stream: stream.write(json.dumps(result, indent=2, allow_nan=False).encode()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
