#!/usr/bin/env python3
"""Independent Best/Latest audit with explicit PPU float32 replay tolerance."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import copy
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment80_1_backend_audit_base", EXP / "audit.py")
base_audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base_audit
spec.loader.exec_module(base_audit)
run = base_audit.run

# PPU float32 reductions can differ by a few ulps after an otherwise exact restore.
PARAM_ATOL = 1e-5
PARAM_RTOL = 1e-5


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def capture(state, module, optimizer, private_rng, base_runtime):
    history = run.load_history().load_history()
    return {
        "base": copy.deepcopy(history.state_bundle(state)),
        "psdc": {key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "private_rng": copy.deepcopy(private_rng.state_dict()),
        "shared_rng": copy.deepcopy(base_runtime.pack_rng()),
    }


def restore(state, module, optimizer, private_rng, base_runtime, latest):
    history = run.load_history().load_history()
    history.load_bundle(state, latest["current"])
    module.load_state_dict(latest["current_psdc"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(latest["current_opt_psdc"]))
    private_rng.load_state_dict(latest["private_rng"])
    base_runtime.restore_rng(latest["shared_rng"])
    return {
        "base": run.object_sha(history.state_bundle(state)) == run.object_sha(latest["current"]),
        "psdc": run.object_sha(module.state_dict()) == run.object_sha(latest["current_psdc"]),
        "optimizer": run.object_sha(optimizer.state_dict()) == run.object_sha(latest["current_opt_psdc"]),
        "private_rng": run.object_sha(private_rng.state_dict()) == run.object_sha(latest["private_rng"]),
        "shared_rng": base_runtime.rng_sha(base_runtime.pack_rng()) == base_runtime.rng_sha(latest["shared_rng"]),
    }


def parameter_difference(left, right):
    maximum = 0.0
    squared = 0.0
    count = 0
    failed = []
    for name in left:
        a, b = left[name], right[name]
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            if a != b:
                failed.append(name)
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            failed.append(name)
            continue
        difference = (a.float() - b.float()).abs()
        maximum = max(maximum, float(difference.max().item()) if difference.numel() else 0.0)
        squared += float(difference.square().sum().item())
        count += difference.numel()
        if not torch.allclose(a, b, atol=PARAM_ATOL, rtol=PARAM_RTOL, equal_nan=False):
            failed.append(name)
    return maximum, (squared / max(count, 1)) ** 0.5, failed


def tree_difference(left, right, path="optimizer"):
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or left.dtype != right.dtype:
            return float("inf"), [], [path]
        if left.is_floating_point() or left.is_complex():
            difference = (left.float() - right.float()).abs()
            maximum = float(difference.max().item()) if difference.numel() else 0.0
            rms = float(difference.square().mean().sqrt().item()) if difference.numel() else 0.0
            failed = [] if torch.allclose(left, right, atol=PARAM_ATOL, rtol=PARAM_RTOL, equal_nan=False) else [path]
            return maximum, [rms], failed
        return (0.0, [], []) if torch.equal(left, right) else (float("inf"), [], [path])
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return float("inf"), [], [path + ".keys"]
        results = [tree_difference(left[key], right[key], f"{path}.{key}") for key in left]
    elif isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        if len(left) != len(right):
            return float("inf"), [], [path + ".length"]
        results = [tree_difference(a, b, f"{path}[{i}]") for i, (a, b) in enumerate(zip(left, right))]
    else:
        return (0.0, [], []) if left == right else (float("inf"), [], [path])
    maximum = max((item[0] for item in results), default=0.0)
    rms_values = [value for item in results for value in item[1]]
    failed = [value for item in results for value in item[2]]
    return maximum, rms_values, failed


def audit_latest(group, fold, state, module, optimizer, private_rng, latest):
    base_runtime = run.load_historical_e61().load_base()
    restored_first = restore(state, module, optimizer, private_rng, base_runtime, latest)
    first_output = run.optimizer_step(state, module, optimizer, group, training=True)
    first = capture(state, module, optimizer, private_rng, base_runtime)
    first_loss = float(first_output["total"].detach().cpu())

    restored_second = restore(state, module, optimizer, private_rng, base_runtime, latest)
    replay_output = run.optimizer_step(state, module, optimizer, group, training=True)
    replay = capture(state, module, optimizer, private_rng, base_runtime)
    replay_loss = float(replay_output["total"].detach().cpu())

    base_exact = run.object_sha(first["base"]) == run.object_sha(replay["base"])
    optimizer_exact = run.object_sha(first["optimizer"]) == run.object_sha(replay["optimizer"])
    private_exact = run.object_sha(first["private_rng"]) == run.object_sha(replay["private_rng"])
    shared_exact = base_runtime.rng_sha(first["shared_rng"]) == base_runtime.rng_sha(replay["shared_rng"])
    loss_exact = first_loss == replay_loss
    max_abs, rms, failed = parameter_difference(first["psdc"], replay["psdc"])
    optimizer_max, optimizer_rms_parts, optimizer_failed = tree_difference(first["optimizer"], replay["optimizer"])
    optimizer_rms = (sum(value * value for value in optimizer_rms_parts) / max(len(optimizer_rms_parts), 1)) ** 0.5
    restore_exact = all(restored_first.values()) and all(restored_second.values())
    if not restore_exact or not all((base_exact, private_exact, shared_exact, loss_exact)) or failed or optimizer_failed:
        raise RuntimeError(
            f"Latest replay audit failed {group} fold {fold}: base={base_exact}, "
            f"private_rng={private_exact}, shared_rng={shared_exact}, loss={loss_exact}, "
            f"restore={restored_first}/{restored_second}, max_abs={max_abs}, optimizer_max_abs={optimizer_max}, "
            f"parameter_failures={failed}, optimizer_failures={optimizer_failed}"
        )
    return {
        "latest_update_bitwise_exact": max_abs == 0.0,
        "latest_update_parameters_within_tolerance": True,
        "latest_checkpoint_restore_bitwise_exact": restore_exact,
        "psdc_parameter_max_abs_difference": max_abs,
        "psdc_parameter_rms_difference": rms,
        "optimizer_update_bitwise_exact": optimizer_exact,
        "optimizer_state_max_abs_difference": optimizer_max,
        "optimizer_state_rms_difference": optimizer_rms,
        "psdc_parameter_atol": PARAM_ATOL,
        "psdc_parameter_rtol": PARAM_RTOL,
        "base_update_exact": base_exact,
        "optimizer_update_exact": optimizer_exact,
        "private_rng_exact": private_exact,
        "shared_rng_exact": shared_exact,
        "loss_exact": loss_exact,
        "replay_loss": replay_loss,
    }


def audit_all():
    rows = []
    bitwise_count = 0
    optimizer_bitwise_count = 0
    maximum_drift = 0.0
    maximum_optimizer_drift = 0.0
    for fold in range(1, 11):
        for group in run.GROUPS:
            task, best, latest, best_path, latest_path = base_audit._task_payload(group, fold)
            state, module, optimizer, private_rng, scaler_sha, prompts, metrics = base_audit._restore_best(
                group, fold, task, best
            )
            replay = audit_latest(group, fold, state, module, optimizer, private_rng, latest)
            bitwise_count += int(replay["latest_update_bitwise_exact"])
            optimizer_bitwise_count += int(replay["optimizer_update_bitwise_exact"])
            maximum_drift = max(maximum_drift, replay["psdc_parameter_max_abs_difference"])
            maximum_optimizer_drift = max(maximum_optimizer_drift, replay["optimizer_state_max_abs_difference"])
            rows.append({
                "group": group,
                "fold": fold,
                "best_checkpoint_sha256": file_sha(best_path),
                "latest_checkpoint_sha256": file_sha(latest_path),
                "best_epoch_zero_based": task["best_epoch_zero_based"],
                "best_epoch_1based": task["best_epoch_1based"],
                "n_test": metrics["n_test"],
                "best_predictions_independently_recomputed": True,
                "best_six_metrics_independently_recomputed": True,
                "warmup_sha256": task["warmup_sha256"],
                "split_sha": task["identity"]["split_sha"],
                "pair_sha": task["identity"]["pair_sha"],
                "scaler_sha": scaler_sha,
                "prompt_sha": task["identity_doc"]["prompt_text_sha"],
                "rng_parity_exact_epochs": task["rng_parity"]["exact_triplets"],
                "rng_parity_mismatches": task["rng_parity"]["mismatches"],
                **replay,
            })
            print(f"AUDITED {group} fold={fold}: Best PASS; Latest parameter drift={replay['psdc_parameter_max_abs_difference']:.3g}, optimizer drift={replay['optimizer_state_max_abs_difference']:.3g}", flush=True)

    output = run.RESULTS / "checkpoint_audit.csv"
    run.atomic_csv(output, rows)
    status = {
        "status": "PASS_WITH_FLOAT32_BACKEND_TOLERANCE" if bitwise_count < len(rows) else "PASS_30_OF_30",
        "audited_tasks": len(rows),
        "best_checkpoints_independently_restored": len(rows),
        "latest_checkpoints_independently_restored_and_replayed_twice": len(rows),
        "latest_update_bitwise_exact_count": bitwise_count,
        "latest_update_bitwise_exact": bitwise_count == len(rows),
        "optimizer_update_bitwise_exact_count": optimizer_bitwise_count,
        "optimizer_update_bitwise_exact": optimizer_bitwise_count == len(rows),
        "latest_update_parameter_atol": PARAM_ATOL,
        "latest_update_parameter_rtol": PARAM_RTOL,
        "latest_update_max_abs_psdc_parameter_drift": maximum_drift,
        "latest_update_max_abs_optimizer_state_drift": maximum_optimizer_drift,
        "base_private_rng_shared_rng_and_loss_replay_exact": True,
        "training_source_sha256": run.source_sha(),
        "audit_script_sha256": file_sha(__file__),
        "checkpoint_audit_csv_sha256": file_sha(output),
        "all_fold2_overlap_rows_retained": True,
        "historical_selection_protocol": "val=test; retained unchanged and disclosed",
    }
    run.atomic_json(run.RESULTS / "audit_status.json", status)
    return status


if __name__ == "__main__":
    print(json.dumps(audit_all(), indent=2))
