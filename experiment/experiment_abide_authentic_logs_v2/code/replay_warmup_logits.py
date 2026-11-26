#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


V2 = Path(__file__).resolve().parents[1]
SOURCE_ROOT = V2 / "inputs/source_repo"
DPSL_ROOT = Path("/model/spromptgl/experiment/DPSL")
HISTORICAL_E61 = Path("/data/adodas/SPromptGL-main/experiment/experiment61_cadr_postalign_vs_e2e")
E61 = SOURCE_ROOT / "experiment/experiment61_cadr_postalign_vs_e2e"
E53 = SOURCE_ROOT / "experiment/experiment53_3_cadr_graph_fallback_fix"
E51 = SOURCE_ROOT / "experiment/experiment51_dugsa"
ANCHORS = SOURCE_ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
BASE_RUNTIME = E53 / ".runtime/fallback_wd_sweep/CADR_WD_0p01_GFR_WD_0p11"
OUT = V2 / "results/metric_recalculation/warmup_replay"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def update_structured_hash(digest, value, np, torch):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"T" + str(tensor.dtype).encode() + repr(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"A" + str(array.dtype).encode() + repr(array.shape).encode())
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"D" + str(len(value)).encode())
        for key in sorted(value, key=lambda item: repr(item)):
            update_structured_hash(digest, key, np, torch)
            update_structured_hash(digest, value[key], np, torch)
    elif isinstance(value, (list, tuple)):
        digest.update((b"L" if isinstance(value, list) else b"Q") + str(len(value)).encode())
        for item in value:
            update_structured_hash(digest, item, np, torch)
    elif isinstance(value, bytes):
        digest.update(b"B" + len(value).to_bytes(8, "big") + value)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        digest.update(type(value).__name__.encode() + b":" + repr(value).encode())
    else:
        digest.update(type(value).__qualname__.encode() + b":" + repr(value).encode())


def structured_sha(value, np, torch) -> str:
    digest = hashlib.sha256()
    update_structured_hash(digest, value, np, torch)
    return digest.hexdigest()


def exact_tree_equal(left, right, np, torch) -> bool:
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact_tree_equal(left[key], right[key], np, torch) for key in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(exact_tree_equal(a, b, np, torch) for a, b in zip(left, right))
    if isinstance(left, (float, int, str, bool, type(None), bytes)):
        return left == right
    return False


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_torch(path: Path, value, torch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_runtime():
    sys.path[:0] = [
        str(DPSL_ROOT / "runtime/python"),
        str(E61),
        str(E51 / "source"),
        str(E51),
        str(E53),
    ]
    spec = importlib.util.spec_from_file_location("e61_warmup_logits_replay_source", E61 / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.BASE_ROOT = SOURCE_ROOT
    module.BASE_EXP = E53
    module.BASE_RUN = E53 / "run.py"
    module.BASE_CADR = E53 / "cadr_gfr.py"
    module.BASE_RUNTIME = BASE_RUNTIME
    module.ANCHORS = ANCHORS
    module.RT = OUT / "runtime"
    module.WARMUP = OUT / "checkpoints"
    module.TASKS = OUT / "tasks"
    module.CKPTS = OUT / "formal_checkpoints"
    module.LOGS = OUT / "logs"
    module.RESULTS = OUT / "results"
    if module.SOURCE_SHA_OVERRIDE is not None:
        raise RuntimeError("source SHA override is active")
    return module


def base_source_sha():
    names = (
        "run.py", "cadr_gfr.py", "run_cadr_fallback.py",
        "run_cadr_fallback_wd_sweep.py", "config_fallback_wd_sweep.json",
    )
    return {name: sha_file(E53 / name) for name in names}


def historical_identity(fold: int, runtime, np, torch):
    historical_path = HISTORICAL_E61 / ".runtime/warmup" / f"fold{fold:02d}.pt"
    dpsl_copy = DPSL_ROOT / "inputs/warmup" / f"fold{fold:02d}.pt"
    if sha_file(historical_path) != sha_file(dpsl_copy):
        raise RuntimeError(f"historical warmup copies differ in fold {fold}")
    saved = runtime.torch_load(historical_path)
    task_path = HISTORICAL_E61 / ".runtime/tasks" / f"E61-E1_f{fold:02d}.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    base_task_path = BASE_RUNTIME / "tasks" / f"CADR_WD_0p01_GFR_WD_0p11_f{fold:02d}.json"
    base_task = json.loads(base_task_path.read_text(encoding="utf-8"))
    manifest = json.loads((E61 / "manifest.json").read_text(encoding="utf-8"))
    manifest_task = next(row for row in manifest["tasks"] if row["group"] == "E61-E1" and row["fold"] == fold)
    return saved, task, base_task, manifest_task, historical_path


def check_initial_state(state, historical, task, base_task, manifest_task, runtime, np, torch):
    manifest = json.loads((E61 / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "initial_shared_sha": base_task["initial_shared_sha"],
        "initial_gfr_sha": base_task["initial_gfr_sha"],
        "initial_cam_sha": base_task["initial_cam_sha"],
        "initial_rng_sha": base_task["initial_rng_sha"],
        "split_sha": base_task["split_sha"],
        "pair_sha": base_task["pair_sha"],
        "anchor_sha": base_task["anchor_sha"],
        "relation_core_sha": task["relation_core_sha"],
    }
    observed = {key: state[key] for key in expected}
    checkpoint_identity = {key: historical[key] for key in expected}
    checks = {key: observed[key] == expected[key] == checkpoint_identity[key] for key in expected}
    checks["warmup_reference_sha"] = sha_file(HISTORICAL_E61 / ".runtime/warmup" / f"fold{state['fold']:02d}.pt") == manifest_task["warmup_sha256"]
    checks["epoch1_rng_starts_at_initial_rng"] = historical["rng_trace"][0]["rng_before"] == state["initial_rng_sha"]
    checks["source_bytes_match_audit_snapshot"] = runtime.source_sha_map() == manifest["audit_source_sha256"]
    checks["base_source_bytes_match_locked_snapshot"] = base_source_sha() == manifest["base_source_sha256"]
    checks["historical_source_sha_matches_bytes"] = historical["source_sha"] == runtime.source_sha_map()
    checks["official_source_override_disabled"] = runtime.SOURCE_SHA_OVERRIDE is None
    checks["train_members_match"] = np.array_equal(state["train_np"], historical["train_indices"])
    checks["test_members_match"] = np.array_equal(state["test_np"], historical["test_indices"])
    checks["relation_pairs_match"] = np.array_equal(state["train_pairs_np"], historical["train_pairs"])
    checks["relation_labels_match"] = np.array_equal(state["train_same_np"], historical["train_same"])
    strict = [key for key in checks if key not in ("historical_source_sha_matches_bytes",)]
    return checks, all(checks[key] for key in strict), expected


def scalar_metrics(values):
    return {key: float(values[key]) for key in ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")}


def grad_norm(parameters, torch) -> float:
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt().cpu())


def save_attention(path: Path, maps: dict, np) -> str:
    if not maps or any(value.size == 0 for value in maps.values()):
        raise RuntimeError("CAM attention hooks did not capture non-empty attention maps")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **maps)
    return sha_file(path)


def state_bundle_for_compare(state, runtime):
    return {
        "model": runtime.model_state_cpu(state["model"]),
        "cam": runtime.model_state_cpu(state["cam"]),
        "optimizer_base": state["opt_base"].state_dict(),
        "optimizer_cam": state["opt_cam"].state_dict(),
        "optimizer_added": state["opt_added"].state_dict(),
    }


def replay_fold(fold: int, runtime, base, np, torch):
    journal_dir = OUT / "journal"
    checkpoint_dir = OUT / "checkpoints"
    attention_dir = OUT / "attention"
    journal_dir.mkdir(parents=True, exist_ok=True)
    journal_path = journal_dir / f"fold{fold:02d}.jsonl"
    checkpoint_path = checkpoint_dir / f"fold{fold:02d}.pt"
    summary_path = OUT / "tasks" / f"fold{fold:02d}.json"
    if journal_path.is_file() and checkpoint_path.is_file() and summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_sha = existing.get("structured_endpoint_sha256", {}).get("replay_checkpoint_file_sha256")
        if existing.get("status") == "WARMUP_EXACT_REPLAY_PASS" and expected_sha == sha_file(checkpoint_path):
            replay = runtime.torch_load(checkpoint_path)
            historical, task, base_task, manifest_task, historical_path = historical_identity(fold, runtime, np, torch)
            fields = ("model", "cam", "optimizer_base", "optimizer_cam", "optimizer_added", "rng", "rng_trace",
                      "train_indices", "test_indices", "train_pairs", "train_same", "split_sha", "pair_sha", "anchor_sha")
            current_structured = {key: replay[key] for key in fields}
            historical_structured = {key: historical[key] for key in fields}
            structured_equal = exact_tree_equal(current_structured, historical_structured, np, torch)
            existing.setdefault("comparison", {})["structured_state_sha"] = structured_equal
            endpoint = existing.setdefault("structured_endpoint_sha256", {})
            endpoint["replay_structured_state_sha256"] = structured_sha(current_structured, np, torch)
            endpoint["historical_structured_state_sha256"] = structured_sha(historical_structured, np, torch)
            endpoint["structured_state_sha_equal"] = structured_equal
            endpoint["historical_checkpoint_file_sha256"] = sha_file(historical_path)
            endpoint["historical_checkpoint_bytes_equal"] = endpoint.get("replay_checkpoint_file_sha256") == endpoint["historical_checkpoint_file_sha256"]
            atomic_json(summary_path, existing)
            return existing
        raise RuntimeError(f"existing Warmup output is incomplete or failed for fold {fold}")
    if any(path.exists() for path in (journal_path, checkpoint_path, summary_path)):
        raise FileExistsError(f"refusing to overwrite existing Warmup output for fold {fold}")

    state = runtime.build_state(fold)
    rng_after_build_state = runtime.rng_sha(base.pack_rng())
    historical, task, base_task, manifest_task, historical_path = historical_identity(fold, runtime, np, torch)
    identity_checks, identity_pass, expected = check_initial_state(state, historical, task, base_task, manifest_task, runtime, np, torch)
    identity_checks["replay_epoch1_rng_starts_at_initial_rng"] = runtime.rng_sha(base.pack_rng()) == state["initial_rng_sha"]
    identity_pass = identity_pass and identity_checks["replay_epoch1_rng_starts_at_initial_rng"]
    if not identity_pass:
        summary = {"fold": fold, "status": "BLOCKED_INITIAL_IDENTITY", "identity_checks": identity_checks,
                   "expected": expected, "actual": {key: state[key] for key in expected}}
        atomic_json(summary_path, summary)
        return summary

    dropout_order = []
    handles = []
    for name, module in state["cam"].named_modules():
        if isinstance(module, torch.nn.Dropout):
            handles.append(module.register_forward_hook(
                lambda layer, inputs, output, module_name=name: dropout_order.append(module_name) if layer.training else None
            ))
    attention_capture = {"enabled": False, "maps": {}}
    for name, module in state["cam"].named_modules():
        if module.__class__.__name__ == "Attention":
            def capture(module, inputs, output, module_name=name):
                if attention_capture["enabled"] and isinstance(output, (tuple, list)) and len(output) > 1:
                    value = output[1]
                    if torch.is_tensor(value):
                        attention_capture["maps"][module_name] = value.detach().cpu().numpy().copy()
            handles.append(module.register_forward_hook(capture))

    best_acc, best_epoch = -1.0, -1
    trace, records = [], []
    train_logits_rows, train_label_rows = [], []
    validation_logits_rows, validation_label_rows = [], []
    final_eval_logp = None
    started = time.perf_counter()
    with journal_path.open("x", encoding="utf-8") as journal:
        for epoch in range(10):
            epoch_started = time.perf_counter()
            before = runtime.rng_sha(base.pack_rng())
            for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"]):
                optimizer.zero_grad(set_to_none=True)
            dropout_order.clear()
            out = base.forward_state(state, training=True, use_cadr=False)
            if not torch.isfinite(out["total"]):
                raise FloatingPointError(f"non-finite warmup loss fold={fold} epoch={epoch + 1}")
            train_metrics = base.classification_metrics(out["logp"], state["train_np"], state["y"])
            train_logits_rows.append(out["logp"].detach().cpu().numpy()[state["train_np"]].copy())
            train_label_rows.append(state["y"][state["train_np"]].detach().cpu().numpy().copy() if torch.is_tensor(state["y"]) else np.asarray(state["y"])[state["train_np"]].copy())
            losses = {"classification_ce": float(out["cls"].detach().cpu()),
                      "gfr_alignment": float(out["graph"].detach().cpu())}
            out["total"].backward()
            for module in (state["model"], state["cam"]):
                for parameter in module.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"non-finite warmup gradient fold={fold} epoch={epoch + 1}")
            gradient_norms = {
                "model": grad_norm(state["model"].parameters(), torch),
                "cam": grad_norm(state["cam"].parameters(), torch),
            }
            if len(dropout_order) != 6:
                raise RuntimeError(f"expected six CAM dropout calls, saw {len(dropout_order)} in fold={fold} epoch={epoch + 1}: {dropout_order}")
            state["opt_base"].step()
            state["opt_cam"].step()
            state["opt_added"].step()
            after_train = runtime.rng_sha(base.pack_rng())
            with torch.no_grad():
                attention_capture["enabled"] = epoch == 9
                evaluated = base.forward_state(state, training=False, use_cadr=False)
                attention_capture["enabled"] = False
                validation_metrics = base.classification_metrics(evaluated["logp"], state["test"], state["y"])
                validation_logits_rows.append(evaluated["logp"].detach().cpu().numpy()[state["test"]].copy())
                validation_label_rows.append(state["y"][state["test"]].detach().cpu().numpy().copy() if torch.is_tensor(state["y"]) else np.asarray(state["y"])[state["test"]].copy())
                if epoch == 9:
                    final_eval_logp = evaluated["logp"].detach().cpu().clone()
            after_eval = runtime.rng_sha(base.pack_rng())
            if validation_metrics["acc"] > best_acc:
                best_acc, best_epoch = float(validation_metrics["acc"]), epoch
            epoch_rng = {"epoch": epoch + 1, "rng_before": before, "rng_after_train": after_train, "rng_after_eval": after_eval}
            trace.append(epoch_rng)
            record = {
                "stage": "E61_WARMUP", "fold": fold, "internal_epoch": epoch + 1,
                "display_epoch": epoch, "completed_updates": epoch + 1,
                "classification_ce": losses["classification_ce"],
                "gfr_alignment": losses["gfr_alignment"],
                "training_metrics": scalar_metrics(train_metrics),
                "validation_metrics": scalar_metrics(validation_metrics),
                "rng": epoch_rng, "dropout_order": list(dropout_order),
                "gradient_norms": gradient_norms,
                "elapsed_epoch_seconds": time.perf_counter() - epoch_started,
            }
            journal.write(json.dumps(record, allow_nan=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            records.append(record)
            del out, evaluated

    (OUT / "logits").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "logits" / f"fold{fold:02d}.npz",
        train_logits=np.asarray(train_logits_rows), train_labels=np.asarray(train_label_rows),
        validation_logits=np.asarray(validation_logits_rows), validation_labels=np.asarray(validation_label_rows),
        train_members=state["train_np"], test_members=state["test_np"],
        rng_before=np.asarray([r["rng_before"] for r in trace]),
        rng_after_train=np.asarray([r["rng_after_train"] for r in trace]),
        rng_after_eval=np.asarray([r["rng_after_eval"] for r in trace]))
    state["warmup_best_acc"] = best_acc
    state["warmup_best_epoch"] = best_epoch
    state["rng_trace"] = trace
    replay_bundle = state_bundle_for_compare(state, runtime)
    replay_rng = base.pack_rng()
    replay_payload = {
        "experiment": "E61 shared warmup authentic replay",
        "fold": fold,
        "seed": 0,
        "completed_epoch": 10,
        **replay_bundle,
        "rng": replay_rng,
        "rng_sha": runtime.rng_sha(replay_rng),
        "rng_trace": trace,
        "train_indices": state["train_np"].copy(),
        "test_indices": state["test_np"].copy(),
        "train_pairs": state["train_pairs_np"].copy(),
        "train_same": state["train_same_np"].copy(),
        "split_sha": state["split_sha"],
        "pair_sha": state["pair_sha"],
        "anchor_sha": state["anchor_sha"],
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_gfr_sha": state["initial_gfr_sha"],
        "initial_cam_sha": state["initial_cam_sha"],
        "relation_core_sha": state["relation_core_sha"],
        "initial_rng_sha": state["initial_rng_sha"],
        "warmup_best_acc": best_acc,
        "warmup_best_epoch": best_epoch,
        "source_sha": runtime.source_sha_map(),
        "base_source_sha": base_source_sha(),
        "author_records": records,
    }
    atomic_torch(checkpoint_path, replay_payload, torch)

    comparisons = {
        "model": exact_tree_equal(replay_bundle["model"], historical["model"], np, torch),
        "cam": exact_tree_equal(replay_bundle["cam"], historical["cam"], np, torch),
        "optimizer_base": exact_tree_equal(replay_bundle["optimizer_base"], historical["optimizer_base"], np, torch),
        "optimizer_cam": exact_tree_equal(replay_bundle["optimizer_cam"], historical["optimizer_cam"], np, torch),
        "optimizer_added": exact_tree_equal(replay_bundle["optimizer_added"], historical["optimizer_added"], np, torch),
        "rng": exact_tree_equal(replay_rng, historical["rng"], np, torch),
        "rng_trace": trace == historical["rng_trace"],
        "train_members": np.array_equal(state["train_np"], historical["train_indices"]),
        "test_members": np.array_equal(state["test_np"], historical["test_indices"]),
        "relation_pairs": np.array_equal(state["train_pairs_np"], historical["train_pairs"]),
        "relation_labels": np.array_equal(state["train_same_np"], historical["train_same"]),
        "warmup_best_epoch": best_epoch == historical["warmup_best_epoch"],
        "warmup_best_acc": best_acc == historical["warmup_best_acc"],
    }

    current_bundle = state_bundle_for_compare(state, runtime)
    rng_before_parity = runtime.rng_sha(base.pack_rng())
    state["model"].load_state_dict(historical["model"], strict=True)
    state["cam"].load_state_dict(historical["cam"], strict=True)
    with torch.no_grad():
        historical_eval = base.forward_state(state, training=False, use_cadr=False)
        historical_eval_logp = historical_eval["logp"].detach().cpu()
    comparisons["epoch10_classification_output"] = torch.equal(final_eval_logp, historical_eval_logp)
    comparisons["parity_evaluation_rng_unchanged"] = runtime.rng_sha(base.pack_rng()) == rng_before_parity
    state["model"].load_state_dict(current_bundle["model"], strict=True)
    state["cam"].load_state_dict(current_bundle["cam"], strict=True)

    attention_path = attention_dir / f"fold{fold:02d}.npz"
    attention_sha = save_attention(attention_path, attention_capture["maps"], np)
    for handle in handles:
        handle.remove()

    endpoint = {
        "model": structured_sha(replay_bundle["model"], np, torch),
        "cam": structured_sha(replay_bundle["cam"], np, torch),
        "optimizer_base": structured_sha(replay_bundle["optimizer_base"], np, torch),
        "optimizer_cam": structured_sha(replay_bundle["optimizer_cam"], np, torch),
        "optimizer_added": structured_sha(replay_bundle["optimizer_added"], np, torch),
        "rng": structured_sha(replay_rng, np, torch),
        "historical_checkpoint_file_sha256": sha_file(historical_path),
        "replay_checkpoint_file_sha256": sha_file(checkpoint_path),
        "historical_checkpoint_bytes_equal": sha_file(historical_path) == sha_file(checkpoint_path),
    }
    comparable_state = {
        **replay_bundle,
        "rng": replay_rng,
        "train_indices": state["train_np"],
        "test_indices": state["test_np"],
        "train_pairs": state["train_pairs_np"],
        "train_same": state["train_same_np"],
        "split_sha": state["split_sha"],
        "pair_sha": state["pair_sha"],
        "anchor_sha": state["anchor_sha"],
        "rng_trace": trace,
    }
    historical_state = {
        **{key: historical[key] for key in ("model", "cam", "optimizer_base", "optimizer_cam", "optimizer_added")},
        "rng": historical["rng"],
        "train_indices": historical["train_indices"],
        "test_indices": historical["test_indices"],
        "train_pairs": historical["train_pairs"],
        "train_same": historical["train_same"],
        "split_sha": historical["split_sha"],
        "pair_sha": historical["pair_sha"],
        "anchor_sha": historical["anchor_sha"],
        "rng_trace": historical["rng_trace"],
    }
    endpoint["replay_structured_state_sha256"] = structured_sha(comparable_state, np, torch)
    endpoint["historical_structured_state_sha256"] = structured_sha(historical_state, np, torch)
    endpoint["structured_state_sha_equal"] = endpoint["replay_structured_state_sha256"] == endpoint["historical_structured_state_sha256"]
    comparisons["structured_state_sha"] = endpoint["structured_state_sha_equal"]
    status = "WARMUP_EXACT_REPLAY_PASS" if all(comparisons.values()) else "WARMUP_REPLAY_MISMATCH"
    summary = {
        "fold": fold,
        "status": status,
        "initial_identity": identity_checks,
        "initial_identity_pass": identity_pass,
            "rng_stages": {
                "after_build_state": rng_after_build_state,
                "expected_initial_rng": state["initial_rng_sha"],
                "epoch1_before_train": trace[0]["rng_before"],
                "epoch10_after_evaluation": trace[-1]["rng_after_eval"],
                "first_divergence_stage": None if trace[0]["rng_before"] == state["initial_rng_sha"] else "epoch1_before_train",
            },
        "historical_source_sha": historical["source_sha"],
        "actual_source_sha": runtime.source_sha_map(),
        "historical_source_sha_discrepancy": (
            "historical_checkpoint_source_sha_differs_from_recovered_source_bytes"
            if not identity_checks["historical_source_sha_matches_bytes"] else None
        ),
        "comparison": comparisons,
        "structured_endpoint_sha256": endpoint,
        "attention": {"path": str(attention_path), "sha256": attention_sha,
                      "keys": sorted(attention_capture["maps"]),
                      "shapes": {key: list(value.shape) for key, value in attention_capture["maps"].items()}},
        "training_seconds": time.perf_counter() - started,
        "epochs": 10,
        "epoch_rng": trace,
        "dropout_order_by_epoch": [row["dropout_order"] for row in records],
        "author_records": records,
    }
    atomic_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    runtime = load_runtime()
    base = runtime.load_base()
    context = base.runtime()
    torch, np = context["torch"], context["np"]
    if not torch.cuda.is_available():
        raise RuntimeError("physical GPU0 is required")
    torch.cuda.set_device(0)
    folds = [args.fold] if args.fold else list(range(1, 11))
    rows = []
    for fold in folds:
        result = replay_fold(fold, runtime, base, np, torch)
        rows.append(result)
        print(json.dumps({"fold": fold, "status": result["status"], "seconds": result.get("training_seconds"),
                          "comparisons": result.get("comparison")}, allow_nan=False), flush=True)
    atomic_json(OUT / "results" / "replay_summary.json", {"folds": rows,
        "status": "WARMUP_EXACT_REPLAY_PASS" if len(rows) == 10 and all(r["status"] == "WARMUP_EXACT_REPLAY_PASS" for r in rows) else "INCOMPLETE_OR_MISMATCH"})


if __name__ == "__main__":
    main()
