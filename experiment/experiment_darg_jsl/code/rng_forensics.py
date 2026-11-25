"""Capture same-serializer RNG and identity summaries during E61 restoration."""
from __future__ import annotations

import hashlib
import inspect
import pickle
from pathlib import Path

import torch

from common import history as h


def _digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def rng_components():
    runtime = h.base()
    packed = runtime.pack_rng()
    return {
        "combined": runtime.rng_sha(packed),
        "python": _digest_bytes(pickle.dumps(packed["python"], protocol=4)),
        "numpy": _digest_bytes(pickle.dumps(packed["numpy"], protocol=4)),
        "cpu": _digest_bytes(packed["cpu"].detach().cpu().contiguous().numpy().tobytes()),
        "device": [
            _digest_bytes(value.detach().cpu().contiguous().numpy().tobytes())
            for value in packed.get("cuda", [])
        ],
    }


def _state_summaries(state):
    if not isinstance(state, dict):
        return {}
    summary = {}
    if "model" in state and "cam" in state:
        summary["model_parameters"] = h.sha({
            "model": state["model"].state_dict(),
            "cam": state["cam"].state_dict(),
        })
    optimizer_keys = ("opt_base", "opt_cam", "opt_added")
    if all(key in state for key in optimizer_keys):
        summary["optimizers"] = {key: h.sha(state[key].state_dict()) for key in optimizer_keys}
    arrays = ("train_np", "test_np", "train_pairs_np", "train_same_np")
    if all(key in state for key in arrays):
        summary["members_and_pairs"] = h.sha({key: state[key] for key in arrays})
    return summary


def _snapshot(stage, state=None, checkpoint=None, **extra):
    row = {"stage": stage, "rng": rng_components(), "identity": _state_summaries(state)}
    if checkpoint is not None:
        row["checkpoint"] = {
            "rng_sha": checkpoint["rng_sha"],
            "rng_recomputed_sha": h.base().rng_sha(checkpoint["rng"]),
            "warmup_sha256": checkpoint.get("warmup_sha256"),
            "model_sha": h.sha({"model": checkpoint["model"], "cam": checkpoint["cam"]}),
            "optimizer_shas": {
                "opt_base": h.sha(checkpoint["optimizer_base"]),
                "opt_cam": h.sha(checkpoint["optimizer_cam"]),
                "opt_added": h.sha(checkpoint["optimizer_added"]),
            },
            "train_sha": h.sha(checkpoint["train_indices"]),
            "test_sha": h.sha(checkpoint["test_indices"]),
            "pair_sha": checkpoint.get("pair_sha"),
        }
    row.update(extra)
    return row


def capture_fold(fold, stage_callback=None):
    runner = h.runner()
    e69 = runner.load_history().load_e69()
    e61 = e69.load_e61()
    base_runtime = e61.load_base()
    target = Path(e61.warmup_path(fold)).resolve()
    stages = []
    captured_checkpoint = {}
    original_load = e61.torch_load
    original_restore = base_runtime.restore_rng

    def load_with_witness(path):
        if Path(path).resolve() != target:
            return original_load(path)
        caller = inspect.currentframe().f_back
        state = caller.f_locals.get("state")
        stages.append(_snapshot("S0", state=state, checkpoint=None,
                                path=str(target), timing="immediately_before_torch_load"))
        payload = original_load(path)
        captured_checkpoint["payload"] = payload
        stages.append(_snapshot("S1", state=state, checkpoint=payload,
                                timing="immediately_after_torch_load"))
        return payload

    def restore_with_witness(rng_state):
        caller = inspect.currentframe().f_back
        state = caller.f_locals.get("state")
        payload = captured_checkpoint.get("payload")
        if payload is None:
            return original_restore(rng_state)
        stages.append(_snapshot("S2", state=state, checkpoint=payload,
                                timing="after_model_optimizer_load_before_rng_restore"))
        original_restore(rng_state)
        stages.append(_snapshot("S3", state=state, checkpoint=payload,
                                timing="immediately_after_rng_restore"))

    def callback(stage, **values):
        row = _snapshot(stage, state=values.get("state"))
        if "members" in values:
            row["members_sha"] = h.sha(values["members"])
        if "identity" in values:
            row["warmup_identity_sha"] = h.sha(values["identity"])
        if "optimizers" in values:
            row["optimizer_identity"] = values["optimizers"]
        if "archived" in values:
            spectrum = values["archived"]["spectrum"]
            row["spectrum"] = {
                "sha": h.sha(spectrum), "basis_sha": h.sha(spectrum["U"]),
                "basis_shape": list(spectrum["U"].shape),
            }
        if "module" in values:
            row["jsl_module_sha"] = h.sha(values["module"].state_dict())
        if "optimizer" in values:
            row["jsl_optimizer_sha"] = h.sha(values["optimizer"].state_dict())
        stages.append(row)
        if stage_callback is not None:
            stage_callback(stage, row, values)

    e61.torch_load = load_with_witness
    base_runtime.restore_rng = restore_with_witness
    try:
        ctx = __import__("jsl_runtime").setup(fold, stage_callback=callback)
    finally:
        e61.torch_load = original_load
        base_runtime.restore_rng = original_restore

    if not all(stage["stage"] in {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"}
               for stage in stages):
        raise RuntimeError(f"fold {fold} RNG stage recorder emitted an unexpected stage")
    expected_order = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    actual_order = [row["stage"] for row in stages]
    if actual_order != expected_order:
        raise RuntimeError(f"fold {fold} RNG stages {actual_order} != {expected_order}")
    if stages[0]["rng"] != stages[1]["rng"] or stages[1]["rng"] != stages[2]["rng"]:
        raise RuntimeError(f"fold {fold} RNG changed while reading/restoring parameters and optimizers")
    if stages[2]["identity"] != stages[3]["identity"]:
        raise RuntimeError(f"fold {fold} model or optimizer changed during RNG restoration")
    if stages[2]["identity"].get("model_parameters") != stages[2]["checkpoint"].get("model_sha"):
        raise RuntimeError(f"fold {fold} restored model parameters differ from the E61 checkpoint")
    if stages[2]["identity"].get("optimizers") != stages[2]["checkpoint"].get("optimizer_shas"):
        raise RuntimeError(f"fold {fold} restored optimizers differ from the E61 checkpoint")
    expected_rng = ctx["warmup"]["rng_sha"]
    if stages[3]["rng"]["combined"] != expected_rng:
        raise RuntimeError(f"fold {fold} restored RNG differs from the checkpoint identity")
    for row in stages[3:]:
        if row["rng"] != stages[3]["rng"]:
            raise RuntimeError(f"fold {fold} {row['stage']} consumed a shared RNG stream")
    return ctx, stages


def add_s8(stages, ctx, notes):
    row = _snapshot("S8", state=ctx["state"],
                    timing="after_isolated_preflight_training_and_recovery_checks",
                    notes=notes)
    stages.append(row)
    return row
