"""Measure one registered step for each method family under eight workers."""
from __future__ import annotations

import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

import run_e8381 as runner


def probe_worker(worker_id: int, group: str) -> dict:
    runner.configure_worker()
    x_np, y_np, _, splits = runner.load_data()
    split = splits[worker_id]
    fold = int(split["fold"])
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    runner.historical.seed_everything(runner.SEED)
    x = torch.as_tensor(x_np, device=device)
    labels = torch.as_tensor(y_np, device=device).long()
    train_np = split["train_idx"]
    train = torch.as_tensor(train_np, device=device).long()
    config = runner.group_config(group)
    model = runner.build_model(config, fold, device)
    optimizers, _ = runner.make_optimizers(model, config)
    started = time.monotonic()
    pair = same = None
    if model.use_darg:
        supports = runner.historical.anchor_pair_support(model, x)
        pair_np, same_np = runner.historical.make_pairs(train_np, y_np, fold, supports)
        pair = torch.as_tensor(pair_np, device=device).long()
        same = torch.as_tensor(same_np, device=device).float()
        model.darg.enter_positive_dcr()
    if model.use_dpsl:
        model.eval()
        with torch.no_grad():
            _, warm_aux = model(x, labels, train, pair, same, dpsl_enabled=False)
        stats = runner.fit_spectral_statistics(warm_aux["cam"], labels, train, fold=fold,
                                                warm_logits=warm_aux["native_logits"])
        stats["warmup_cam"] = warm_aux["cam"].detach().cpu()
        stats["warmup_logits"] = warm_aux["native_logits"].detach().cpu()
        model.attach_dpsl(stats, str(runner.prompt_path_for(config, runner.BASE_GROUPS["H4"])),
                          824602 * 100 + fold)
        optimizers["dpsl"] = torch.optim.AdamW(model.dpsl.parameters(), lr=0.0005, weight_decay=0.01)
    runner.zero_optimizers(optimizers)
    model.train()
    _, aux = model(x, labels, train, pair, same, dpsl_enabled=True)
    kind = runner.active_supervision_kind(config, 11)
    _, historical_total, supervision, _ = runner.historical_loss(aux, labels, train, model, kind)
    total = historical_total + aux["dpsl_loss"]
    if model.use_dpsl:
        budget = runner.dpsl_optimizer_step(model, optimizers, historical_total,
                                             aux["scmr_objectives"],
                                             float(config["dpsl_shared_cap"]))
    else:
        historical_total.backward()
        runner.step_optimizers(optimizers)
        budget = {"shared_cap": 0.0}
    if not torch.isfinite(total) or not torch.isfinite(supervision):
        raise RuntimeError(f"{group} resource probe received a non-finite loss")
    torch.cuda.synchronize(device)
    return {
        "worker_id": worker_id, "group": group, "fold": fold, "loss": float(total.detach()),
        "supervision_kind": kind, "supervision": float(supervision.detach()),
        "shared_cap": float(budget["shared_cap"]),
        "seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "device": torch.cuda.get_device_name(device), "finite": True,
    }


def main() -> None:
    seal_path = runner.ROOT / "audits/training_seal.json"
    preflight_path = runner.ROOT / "audits/preflight_start.json"
    if not seal_path.is_file() or not preflight_path.is_file():
        raise RuntimeError("The E83.8 source seal and method preflight must exist first")
    preflight = json.loads(preflight_path.read_text())
    if (preflight.get("shared_status") != "PASS"
            or preflight.get("training_seal_sha256") != runner.sha_file(seal_path)):
        raise RuntimeError("The resource probe does not match the sealed shared preflight")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Resource probe requires the assigned single GPU")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    started = time.monotonic()
    records = []
    groups = ["X1", "X2", "X3", "X4", "X5", "X6", "X4", "X6"]
    with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn")) as pool:
        tasks = {pool.submit(probe_worker, worker_id, group): (worker_id, group)
                 for worker_id, group in enumerate(groups)}
        for future in as_completed(tasks):
            worker_id, group = tasks[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {"worker_id": worker_id, "group": group,
                          "fold": int(worker_id + 1), "finite": False,
                          "failure_reason": repr(exc)}
            records.append(record)
            if record["finite"]:
                print(f"RESOURCE {record['group']} worker={record['worker_id']} fold={record['fold']:02d} "
                      f"seconds={record['seconds']:.2f} peak_reserved={record['peak_reserved_mib']:.1f}MiB",
                      flush=True)
            else:
                print(f"RESOURCE_FAIL {group} worker={worker_id}: {record['failure_reason']}",
                      flush=True)
    records.sort(key=lambda row: row["worker_id"])
    group_statuses = {}
    for group in sorted(set(groups)):
        group_statuses[group] = {
            "status": "PASS" if all(row["finite"] for row in records if row["group"] == group) else "FAIL",
            "probe_count": sum(row["group"] == group for row in records),
        }
    for group in (f"X{index}" for index in range(1, 7)):
        if group not in group_statuses:
            group_statuses[group] = {"status": "NOT_PROBED", "probe_count": 0}
    successful_records = [row for row in records if row["finite"]]
    value = {
        "experiment": "Experiment 83.8.1",
        "status": "PASS" if len(successful_records) == 8 else "PARTIAL",
        "groups": group_statuses,
        "worker_count": len(records), "global_worker_limit": 8,
        "device": (successful_records[0]["device"] if successful_records
                   else torch.cuda.get_device_name(0)),
        "training_seal_sha256": runner.sha_file(seal_path),
        "concurrent_probe_seconds": time.monotonic() - started,
        "peak_reserved_mib_max": max((row.get("peak_reserved_mib", 0.0) for row in records), default=0.0),
        "peak_allocated_mib_max": max((row.get("peak_allocated_mib", 0.0) for row in records), default=0.0),
        "records": records,
    }
    runner.write_json(runner.ROOT / "audits/resource_probe.json", value)
    print(json.dumps({key: value[key] for key in
                      ("status", "worker_count", "device", "concurrent_probe_seconds",
                       "peak_reserved_mib_max", "groups")}, indent=2))
    if len(records) != 8:
        raise RuntimeError("The eight-worker PPU resource probe did not complete all tasks")


if __name__ == "__main__":
    main()
