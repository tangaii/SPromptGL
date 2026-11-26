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
DPSL = Path("/model/spromptgl/experiment/DPSL")
DARG_SOURCE = Path("/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr")
OUT = V2 / "results/metric_recalculation/darg_replay"
PLAN_PATH = V2 / "results/metric_recalculation/darg_sampling_plan.json"
GROUP = "E74.4-G2-Positive-DCR"
WARMUP_LINKAGE = {}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_tree_equal(left, right, torch, np):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact_tree_equal(left[key], right[key], torch, np) for key in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(exact_tree_equal(a, b, torch, np) for a, b in zip(left, right))
    if isinstance(left, (str, int, float, bool, type(None), bytes)):
        return left == right
    return False


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_runner():
    sys.path[:0] = [str(DPSL / "runtime/python"), str(DPSL / "code"), str(DPSL / "code/history_base")]
    spec = importlib.util.spec_from_file_location("official_e744_darg_runner", DARG_SOURCE / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    vendor = DPSL / "code/vendor/repo/experiment"
    source = V2 / "inputs/source_repo/experiment"
    module.E741 = source / "experiment74_1_rudag"
    module.E69 = source / "experiment69_3_g2_two_rows"
    module.E62 = source / "experiment62_e2e_gate_init_sensitivity"
    module.E74 = V2 / "inputs/source_repo/experiment/experiment74_udag"
    module.E742 = source / "experiment74_2_dcr_rudag"
    module.E743 = source / "experiment74_3_local_dcr"
    module.E53 = Path("/data/adodas/SPromptGL-main/experiment/experiment53_3_cadr_graph_fallback_fix")
    module.E61 = Path("/data/adodas/SPromptGL-main/experiment/experiment61_cadr_postalign_vs_e2e")
    return module


def install_history_path_adapter(module, torch, np):
    history = module.load_history()
    source = V2 / "inputs/source_repo/experiment"
    data_repo = Path("/data/adodas/SPromptGL-main")
    vendor_repo = DPSL / "code/vendor/repo"
    history.BASE = source / "experiment69_3_g2_two_rows"
    history.E53 = data_repo / "experiment/experiment53_3_cadr_graph_fallback_fix"
    history.E61_EXP = data_repo / "experiment/experiment61_cadr_postalign_vs_e2e"
    history.E62 = source / "experiment62_e2e_gate_init_sensitivity"
    history.E74 = source / "experiment74_udag"
    history.E69 = None

    def load_e69_with_recovered_paths():
        if history.E69 is not None:
            return history.E69
        expected = history.CFG["base"]
        e69_root = source / "experiment69_3_g2_two_rows"
        if sha_file(e69_root / "run.py") != expected["run_sha256"] or sha_file(e69_root / "manifest.json") != expected["manifest_sha256"]:
            raise RuntimeError("recovered E69 source differs from the locked E74.1 reference")
        manifest = json.loads((e69_root / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("status"), manifest.get("completed_tasks"), manifest.get("independent_audits")) != ("COMPLETE_AUDITED", 10, 10):
            raise RuntimeError("recovered E69 reference is not ten-fold audited")
        spec = importlib.util.spec_from_file_location("e69_path_adapted_for_darg", e69_root / "run.py")
        e69 = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = e69
        assert spec.loader is not None
        spec.loader.exec_module(e69)
        e69.E62_EXP = history.E62
        e69.E62_RUN = e69.E62_EXP / "run.py"
        e69.E62_CONFIG = e69.E62_EXP / "config.json"
        e69.E62_MANIFEST = e69.E62_EXP / "manifest.json"
        e69.E53_EXP = history.E53
        e69.E61_EXP = history.E61_EXP
        e69.E61_RUN = e69.E61_EXP / "run.py"
        e69.E61_MODEL = e69.E61_EXP / "model_e61.py"
        e69.E61_CONFIG = e69.E61_EXP / "config.json"
        e69.E61_MANIFEST = e69.E61_EXP / "manifest.json"

        def load_e61_with_verified_sources():
            if e69.E61 is not None:
                return e69.E61
            e61_manifest = json.loads(e69.E61_MANIFEST.read_text(encoding="utf-8"))
            expected_source = e61_manifest["audit_source_sha256"]
            actual_source = {name: sha_file(e69.E61_EXP / name) for name in ("run.py", "model_e61.py", "config.json")}
            if actual_source != expected_source:
                raise RuntimeError(f"E61 recovered source hashes differ from the audited source snapshot: {actual_source}")
            expected_base = e61_manifest["base_source_sha256"]
            base_files = ("run.py", "cadr_gfr.py", "run_cadr_fallback.py", "run_cadr_fallback_wd_sweep.py", "config_fallback_wd_sweep.json")
            actual_base = {name: sha_file(history.E53 / name) for name in base_files}
            if actual_base != expected_base:
                raise RuntimeError(f"E53 locked dependency differs from the E61 audit snapshot: {actual_base}")
            e51_root = vendor_repo / "experiment/experiment72_mgdss/.runtime/vendor/root/experiment/experiment51_dugsa"
            for path in (e51_root / "source", e51_root, history.E53, vendor_repo / "experiment/experiment53_3_cadr_graph_fallback_fix"):
                if str(path) in sys.path:
                    sys.path.remove(str(path))
                sys.path.insert(0, str(path))
            e61_spec = importlib.util.spec_from_file_location("e61_audited_source_for_darg", e69.E61_RUN)
            e61 = importlib.util.module_from_spec(e61_spec)
            sys.modules[e61_spec.name] = e61
            assert e61_spec.loader is not None
            e61_spec.loader.exec_module(e61)
            e61.BASE_ROOT = vendor_repo
            e61.BASE_EXP = vendor_repo / "experiment/experiment53_3_cadr_graph_fallback_fix"
            e61.BASE_RUN = e61.BASE_EXP / "run.py"
            e61.BASE_CADR = e61.BASE_EXP / "cadr_gfr.py"
            e61.BASE_RUNTIME = e61.BASE_EXP / ".runtime/fallback_wd_sweep/CADR_WD_0p01_GFR_WD_0p11"
            e61.ANCHORS = vendor_repo / "experiment/experiment72_mgdss/.runtime/vendor/root/experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
            e61.WARMUP = DPSL / "inputs/warmup"
            if e61.SOURCE_SHA_OVERRIDE is not None:
                raise RuntimeError("E61 source SHA override must remain disabled")

            def load_verified_warmup(fold):
                state = e61.build_state(fold)
                path = e61.warmup_path(fold)
                saved = e61.torch_load(path)
                replay_path = V2 / "Warmup/checkpoints" / f"fold{fold:02d}.pt"
                replay = e61.torch_load(replay_path)
                manifest_row = next(row for row in e61_manifest["warmups"] if row["fold"] == fold)
                checks = {
                    "archived_file_sha": sha_file(path) == manifest_row["sha256"],
                    "dpsl_warmup_file_sha": sha_file(path) == sha_file(DPSL / "inputs/warmup" / path.name),
                    "initial_shared_sha": saved["initial_shared_sha"] == state["initial_shared_sha"],
                    "initial_gfr_sha": saved["initial_gfr_sha"] == state["initial_gfr_sha"],
                    "initial_cam_sha": saved["initial_cam_sha"] == state["initial_cam_sha"],
                    "initial_rng_sha": saved["initial_rng_sha"] == state["initial_rng_sha"],
                    "relation_core_sha": saved["relation_core_sha"] == state["relation_core_sha"],
                    "split_sha": saved["split_sha"] == state["split_sha"],
                    "pair_sha": saved["pair_sha"] == state["pair_sha"],
                    "anchor_sha": saved["anchor_sha"] == state["anchor_sha"],
                    "replay_model": exact_tree_equal(saved["model"], replay["model"], torch, np),
                    "replay_cam": exact_tree_equal(saved["cam"], replay["cam"], torch, np),
                    "replay_optimizer_base": exact_tree_equal(saved["optimizer_base"], replay["optimizer_base"], torch, np),
                    "replay_optimizer_cam": exact_tree_equal(saved["optimizer_cam"], replay["optimizer_cam"], torch, np),
                    "replay_optimizer_added": exact_tree_equal(saved["optimizer_added"], replay["optimizer_added"], torch, np),
                    "replay_rng": exact_tree_equal(saved["rng"], replay["rng"], torch, np),
                    "replay_train_members": exact_tree_equal(saved["train_indices"], replay["train_indices"], torch, np),
                    "replay_test_members": exact_tree_equal(saved["test_indices"], replay["test_indices"], torch, np),
                    "replay_relation_pairs": exact_tree_equal(saved["train_pairs"], replay["train_pairs"], torch, np),
                    "replay_relation_labels": exact_tree_equal(saved["train_same"], replay["train_same"], torch, np),
                }
                if not all(checks.values()):
                    raise RuntimeError(f"E61 warmup-to-DARG state identity failed fold {fold}: {checks}")
                for field in ("initial_shared_sha", "initial_gfr_sha", "initial_cam_sha", "relation_core_sha", "initial_rng_sha", "split_sha", "pair_sha", "anchor_sha"):
                    if saved[field] != state[field]:
                        raise RuntimeError(f"E61 warmup identity mismatch fold={fold} field={field}")
                state["model"].load_state_dict(saved["model"], strict=True)
                state["cam"].load_state_dict(saved["cam"], strict=True)
                state["opt_base"].load_state_dict(saved["optimizer_base"])
                state["opt_cam"].load_state_dict(saved["optimizer_cam"])
                state["opt_added"].load_state_dict(saved["optimizer_added"])
                state["warmup_best_acc"] = saved["warmup_best_acc"]
                state["warmup_best_epoch"] = saved["warmup_best_epoch"]
                state["rng_trace"] = saved["rng_trace"]
                e61.load_base().restore_rng(saved["rng"])
                if e61.load_base().rng_sha(e61.load_base().pack_rng()) != saved["rng_sha"]:
                    raise RuntimeError(f"E61 archived warmup RNG readback failed fold {fold}")
                WARMUP_LINKAGE[fold] = {
                    "status": "PASS", "checks": checks,
                    "checkpoint_source_sha_recorded": saved["source_sha"],
                    "checkpoint_source_sha_computed": e61.source_sha_map(),
                    "source_sha_metadata_discrepancy": saved["source_sha"] != e61.source_sha_map(),
                    "archived_warmup_sha256": sha_file(path),
                    "replay_structured_state_sha256": replay.get("rng_sha"),
                    "rng_sha": saved["rng_sha"],
                }
                return state, saved

            e61.load_warmup_state = load_verified_warmup
            e69.E61 = e61
            e69.E61_TRAIN_SOURCE = e61_manifest["source_sha256"]
            return e61

        e69.load_e61 = load_e61_with_verified_sources
        e69.verify_e62_reference()
        history.E69 = e69
        return e69

    history.load_e69 = load_e69_with_recovered_paths
    return history


def metric_row(logp, indices, labels, base, np):
    values = base.classification_metrics(logp, indices, labels)
    return {key: float(values[key]) for key in ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")}


def save_attention(path: Path, attention, np) -> str:
    if attention is None:
        raise RuntimeError("DARG forward did not return attention maps")
    if isinstance(attention, dict):
        values = {name: np.asarray(value) for name, value in attention.items()}
    else:
        values = {"attention": attention.detach().cpu().numpy() if hasattr(attention, "detach")
                  else np.asarray(attention)}
    if not values or any(not value.size for value in values.values()):
        raise RuntimeError("DARG forward returned empty attention maps")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    return sha_file(path)


def next_step_witness(module, state, history, base, torch, np):
    dropout_order = []
    handles = []
    for name, layer in state["cam"].named_modules():
        if isinstance(layer, torch.nn.Dropout):
            handles.append(layer.register_forward_hook(
                lambda current, inputs, output, module_name=name: dropout_order.append(module_name)
                if current.training else None
            ))
    try:
        rng_before = base.rng_sha(base.pack_rng())
        history.zero_optimizers(state)
        dropout_order.clear()
        out = module.forward(state, GROUP, training=True)
        if len(dropout_order) != 6 or not torch.isfinite(out["total"]):
            raise RuntimeError(f"next-step training forward failed dropout/loss checks: {dropout_order}")
        out["total"].backward()
        if not module._finite_grads(state["model"]) or not module._finite_grads(state["cam"]):
            raise FloatingPointError("next-step DARG gradient is not finite")
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        rng_after_train = base.rng_sha(base.pack_rng())
        with torch.no_grad():
            evaluated = module.forward(state, GROUP, training=False)
            metrics = base.classification_metrics(evaluated["logp"], state["test"], state["y"])
        rng_after_eval = base.rng_sha(base.pack_rng())
        return {
            "rng": {"rng_before": rng_before, "rng_after_train": rng_after_train,
                    "rng_after_eval": rng_after_eval},
            "dropout_order": list(dropout_order),
            "losses": {key: float(out[key].detach().cpu())
                       for key in ("cls", "graph", "pair", "adv", "preserve", "total")},
            "metrics": {key: float(metrics[key]) for key in module.METRICS},
            "prediction": metrics["pred"].copy(), "probability": metrics["prob"].copy(),
            "state": history.state_bundle(state),
        }
    finally:
        for handle in handles:
            handle.remove()


def fold_paths(module, fold):
    stem = f"{GROUP}_f{fold:02d}"
    return {
        "task": module.TASKS / f"{stem}.json",
        "best": module.CKPTS / f"{stem}.best.pt",
        "latest": module.CKPTS / f"{stem}.latest.pt",
        "history": module.LOGS / f"{stem}.jsonl",
        "author": OUT / "journal" / f"fold{fold:02d}.author.jsonl",
        "attention": OUT / "attention" / f"fold{fold:02d}.npz",
    }


def train_fold(module, fold, base, np, torch):
    paths = fold_paths(module, fold)
    for key in ("task", "best", "latest", "history", "author", "attention"):
        if paths[key].exists():
            raise FileExistsError(f"refusing to overwrite DARG output {paths[key]}")

    sources = module.source_manifest()
    source_hash = module.sha_object(sources)
    formal_manifest = json.loads((DARG_SOURCE / "manifest.json").read_text(encoding="utf-8"))
    protected = formal_manifest["protected_sources"]
    mismatched = {Path(path).name: {"expected": protected.get(path), "actual": digest}
                  for path, digest in sources.items()
                  if path in protected and Path(path).name != "audit.py" and protected[path] != digest}
    if mismatched:
        raise RuntimeError(f"E74.4 training source differs from its protected source record: {mismatched}")
    state, warmup, identity = module.prepare(fold)
    state["fold"] = fold
    state["group"] = GROUP
    state["rudag_identity"] = identity
    history = module.load_history()
    if base.rng_sha(base.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold} does not begin at the archived E61 epoch-10 RNG")
    historical_task_path = DARG_SOURCE / ".runtime/tasks" / f"{GROUP}_f{fold:02d}.json"
    historical_best_path = DARG_SOURCE / ".runtime/checkpoints" / f"{GROUP}_f{fold:02d}.best.pt"
    historical_latest_path = DARG_SOURCE / ".runtime/checkpoints" / f"{GROUP}_f{fold:02d}.latest.pt"
    historical_task = json.loads(historical_task_path.read_text(encoding="utf-8"))
    if historical_task.get("status") != "COMPLETE":
        raise RuntimeError(f"historical DARG task is not complete for fold {fold}")
    if sha_file(historical_best_path) != historical_task["checkpoint_sha256"]:
        raise RuntimeError(f"historical DARG Best checkpoint SHA mismatch for fold {fold}")
    historical_trace = {int(row["epoch"]): row for row in historical_task["rng_trace"]}

    dropout_order = []
    attention_capture = {}
    handles = []
    for name, layer in state["cam"].named_modules():
        if isinstance(layer, torch.nn.Dropout):
            handles.append(layer.register_forward_hook(
                lambda module, inputs, output, module_name=name: dropout_order.append(module_name) if module.training else None
            ))
        elif name.endswith(".attention") and type(layer).__name__ == "Attention":
            def capture_attention(current, inputs, output, module_name=name):
                if not current.training and isinstance(output, (tuple, list)) and len(output) > 1:
                    attention_capture[module_name] = output[1].detach().cpu().numpy().copy()
            handles.append(layer.register_forward_hook(capture_attention))

    start_epoch, wait = 10, 0
    best_acc, best_epoch, best, trace = -1.0, -1, None, []
    best_attention_sha = None
    author_rows, history_rows = [], []
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    target_epochs = set(next(row["internal_epochs"] for row in plan["folds"] if row["fold"] == fold))
    capture_rows = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(start_epoch, int(module.CFG["training"]["max_epochs"])):
        epoch_started = time.perf_counter()
        rng_before = base.rng_sha(base.pack_rng())
        history.zero_optimizers(state)
        dropout_order.clear()
        out = module.forward(state, GROUP, training=True)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"non-finite DARG loss fold={fold} epoch={epoch + 1}")
        rng_after_forward = base.rng_sha(base.pack_rng())
        train_metrics = metric_row(out["logp"], state["train_np"], state["y"], base, np)
        if base.rng_sha(base.pack_rng()) != rng_after_forward:
            raise RuntimeError("DARG logging metrics consumed training RNG")
        train_ce = float(out["cls"].detach().cpu())
        gfr_alignment = float(out["graph"].detach().cpu())
        if len(dropout_order) != 6:
            raise RuntimeError(f"expected six CAM Dropout calls, got {dropout_order}")
        out["total"].backward()
        if not module._finite_grads(state["model"]) or not module._finite_grads(state["cam"]):
            raise FloatingPointError(f"non-finite DARG gradient fold={fold} epoch={epoch + 1}")
        gate_grad, scorer_grad = module._total_loss_gradient_snapshot(state)
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        rng_after_train = base.rng_sha(base.pack_rng())
        executed = epoch + 1
        with torch.no_grad():
            attention_capture.clear()
            evaluated = module.forward(state, GROUP, training=False)
            metrics = base.classification_metrics(evaluated["logp"], state["test"], state["y"])
        rng_after_eval = base.rng_sha(base.pack_rng())
        if executed in target_epochs:
            capture_rows.append({
                "internal_epoch": executed, "display_epoch": executed - 1,
                "train_logits": out["logp"].detach().cpu().numpy()[state["train_np"]].copy(),
                "train_labels": state["y"][state["train_np"]].detach().cpu().numpy().copy() if torch.is_tensor(state["y"]) else np.asarray(state["y"])[state["train_np"]].copy(),
                "validation_logits": evaluated["logp"].detach().cpu().numpy()[state["test"]].copy(),
                "validation_labels": state["y"][state["test"]].detach().cpu().numpy().copy() if torch.is_tensor(state["y"]) else np.asarray(state["y"])[state["test"]].copy(),
                "rng_before": rng_before, "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval,
                "state_sha": module.sha_object(history.state_bundle(state)),
            })
        if rng_after_eval != rng_after_train:
            raise RuntimeError("DARG evaluation consumed RNG")
        epoch_rng = {"epoch": executed, "rng_before": rng_before,
                     "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval}
        if epoch_rng != historical_trace.get(executed):
            raise RuntimeError(f"DARG historical RNG trajectory differs fold={fold} epoch={executed}")
        trace.append(epoch_rng)

        improved = metrics["acc"] > best_acc
        if improved:
            best_acc, best_epoch, wait = float(metrics["acc"]), epoch, 0
            best = {
                "state": history.state_bundle(state),
                "metrics": {key: float(value) for key, value in metrics.items() if key not in ("pred", "prob")},
                "probability": metrics["prob"].copy(), "prediction": metrics["pred"].copy(),
                "losses": {key: float(out[key].detach().cpu()) for key in ("cls", "graph", "pair", "adv", "preserve", "total")},
                "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval,
                "total_loss_gate_grad": gate_grad, "total_loss_scorer_grad": scorer_grad,
            }
            best_attention_sha = save_attention(paths["attention"], attention_capture, np)
            module.atomic_torch(paths["best"], module._checkpoint_payload(
                GROUP, state, best, best_epoch, executed, source_hash, trace))
        else:
            wait += 1

        author = {
            "trn_cls": train_ce, "gfr_alignment": gfr_alignment,
            "trn_auc": train_metrics["hard_auc"], "trn_acc": train_metrics["acc"],
            "val_auc": float(metrics["hard_auc"]), "val_acc": float(metrics["acc"]),
            "epoch": executed - 1, "internal_epoch": executed, "completed_updates": executed,
        }
        author_rows.append(author)
        history_row = {
            "stage": "E74.4-G2", "fold": fold, **author,
            "train_metrics": train_metrics,
            "validation_metrics": {key: float(metrics[key]) for key in module.METRICS},
            "rng": epoch_rng, "dropout_order": list(dropout_order),
            "gate_grad_abs": [abs(float(value)) for value in gate_grad],
            "scorer_grad_norms": scorer_grad,
            "gradient_finite": True, "logging_rng_unchanged": True,
            "elapsed_epoch_seconds": time.perf_counter() - epoch_started,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        }
        history_rows.append(history_row)

        if executed % 10 == 0 or executed == module.CFG["training"]["max_epochs"] or wait > module.CFG["training"]["early_stop_wait_gt"]:
            module.atomic_torch(paths["latest"], {
                "group": GROUP, "fold": fold, "source_sha256": source_hash, "identity": identity,
                "warmup_sha256": identity["warmup_sha256"], "current": history.state_bundle(state),
                "rng": base.pack_rng(), "next_epoch": executed, "wait": wait,
                "best_acc": best_acc, "best_epoch": best_epoch, "best": best, "trace": trace,
            })
            record = {
                "group": GROUP, "fold": fold, "epoch": executed, "best_epoch_1based": best_epoch + 1,
                "best_acc": best_acc, "metrics": {key: float(metrics[key]) for key in module.METRICS},
                "loss": float(out["total"].detach().cpu()), "gate_values": [
                    float(0.01 * torch.sigmoid(state["model"].adaptive_gate_logits[i]).detach().cpu()) if i < 3
                    else float(torch.sigmoid(state["model"].adaptive_gate_logits[i]).detach().cpu()) for i in range(4)],
                "total_loss_gate_grad_abs": [abs(float(value)) for value in gate_grad],
                "total_loss_scorer_grad_norms": scorer_grad, "rng": epoch_rng,
                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            }
            paths["history"].parent.mkdir(parents=True, exist_ok=True)
            with paths["history"].open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(f"{GROUP} fold={fold} epoch={executed} best={best_acc:.6f}@{best_epoch + 1} wait={wait}", flush=True)

        del out, evaluated
        if wait > module.CFG["training"]["early_stop_wait_gt"]:
            break

    if best is None:
        raise RuntimeError(f"no strict-ACC best checkpoint for DARG fold {fold}")
    module.atomic_torch(paths["best"], module._checkpoint_payload(
        GROUP, state, best, best_epoch, executed, source_hash, trace))
    if (best_epoch + 1, executed) != (historical_task["best_epoch_1based"], historical_task["epochs_executed"]):
        raise RuntimeError(f"DARG Best/Latest epochs differ from the historical task in fold {fold}")
    historical_best = module.torch_load(historical_best_path)
    if not (
        np.array_equal(best["prediction"], historical_best["test_prediction"])
        and np.array_equal(best["probability"], historical_best["test_probability"])
        and exact_tree_equal(best["state"], historical_best["best_state"], torch, np)
        and trace == historical_best["rng_trace"]
    ):
        raise RuntimeError(f"DARG Best state/patient outputs differ from history in fold {fold}")
    latest = module.torch_load(paths["latest"])
    historical_latest = module.torch_load(historical_latest_path)
    latest_checks = {
        "current_state": exact_tree_equal(latest["current"], historical_latest["current"], torch, np),
        "rng": exact_tree_equal(latest["rng"], historical_latest["rng"], torch, np),
        "trace": latest["trace"] == historical_latest["trace"],
        "epoch_wait_best": (latest["next_epoch"], latest["wait"], latest["best_epoch"], latest["best_acc"])
        == (historical_latest["next_epoch"], historical_latest["wait"],
            historical_latest["best_epoch"], historical_latest["best_acc"]),
    }
    if not all(latest_checks.values()):
        raise RuntimeError(f"DARG Latest state differs from history in fold {fold}: {latest_checks}")
    next_from_memory = next_step_witness(module, state, history, base, torch, np)
    restored, restored_warmup, restored_identity = module.prepare(fold)
    history.load_bundle(restored, latest["current"])
    base.restore_rng(latest["rng"])
    if not (restored_identity == identity and restored_warmup["rng_sha"] == warmup["rng_sha"]
            and exact_tree_equal(history.state_bundle(restored), latest["current"], torch, np)
            and base.rng_sha(base.pack_rng()) == base.rng_sha(latest["rng"])):
        raise RuntimeError(f"DARG Latest fresh-process restore failed before next step in fold {fold}")
    next_from_restore = next_step_witness(module, restored, history, base, torch, np)
    next_step_exact = all(
        exact_tree_equal(next_from_memory[key], next_from_restore[key], torch, np)
        for key in next_from_memory
    )
    if not next_step_exact:
        raise RuntimeError(f"DARG restored next training/evaluation step differs in fold {fold}")
    capture_dir = OUT / "logits"
    capture_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(capture_dir / f"fold{fold:02d}.npz",
        internal_epoch=np.asarray([r["internal_epoch"] for r in capture_rows], dtype=np.int64),
        display_epoch=np.asarray([r["display_epoch"] for r in capture_rows], dtype=np.int64),
        train_logits=np.asarray([r["train_logits"] for r in capture_rows]),
        train_labels=np.asarray([r["train_labels"] for r in capture_rows]),
        validation_logits=np.asarray([r["validation_logits"] for r in capture_rows]),
        validation_labels=np.asarray([r["validation_labels"] for r in capture_rows]),
        rng_before=np.asarray([r["rng_before"] for r in capture_rows]),
        rng_after_train=np.asarray([r["rng_after_train"] for r in capture_rows]),
        rng_after_eval=np.asarray([r["rng_after_eval"] for r in capture_rows]),
        state_sha=np.asarray([r["state_sha"] for r in capture_rows]))
    module.atomic_json(paths["task"], {
        "status": "COMPLETE", "group": GROUP, "fold": fold, "seed": 0,
        "source_sha256": source_hash, "source_files": sources, "identity": identity,
        "checkpoint_sha256": sha_file(paths["best"]), "best_epoch_1based": best_epoch + 1,
        "latest_epoch": executed, "epochs_executed": executed,
        "metrics": best["metrics"], "rng_trace": trace,
        "training_seconds": time.perf_counter() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "attention_sha256": best_attention_sha, "device": torch.cuda.get_device_name(0),
        "historical_best_sha256": historical_task["checkpoint_sha256"],
        "historical_best_tensor_and_patient_outputs_exact": True,
        "historical_latest_exact": latest_checks,
        "next_step_restore_exact": next_step_exact,
        "metric_recalculation_targets": sorted(target_epochs),
        "metric_recalculation_logits_sha256": sha_file(capture_dir / f"fold{fold:02d}.npz"),
    })
    atomic_jsonl(paths["author"], author_rows)
    atomic_jsonl(paths["history"].with_suffix(".full.jsonl"), history_rows)
    for handle in handles:
        handle.remove()

    # Verify the durable Latest can be loaded into a freshly prepared fold state.
    latest = module.torch_load(paths["latest"])
    restored, restored_warmup, restored_identity = module.prepare(fold)
    history.load_bundle(restored, latest["current"])
    base.restore_rng(latest["rng"])
    restore_state_equal = exact_tree_equal(history.state_bundle(restored), latest["current"], torch, np)
    restore_rng_equal = base.rng_sha(base.pack_rng()) == base.rng_sha(latest["rng"])
    restore_identity_equal = restored_identity == identity and restored_warmup["rng_sha"] == warmup["rng_sha"]
    if not (restore_state_equal and restore_rng_equal and restore_identity_equal):
        raise RuntimeError(f"DARG Latest restore verification failed in fold {fold}")
    summary = {
        "fold": fold, "status": "TRAINED_AUDITED", "best_epoch": best_epoch + 1,
        "latest_epoch": executed, "warmup_sha256": identity["warmup_sha256"],
        "best_sha256": sha_file(paths["best"]), "latest_sha256": sha_file(paths["latest"]),
        "history_sha256": sha_file(paths["history"].with_suffix(".full.jsonl")),
        "author_sha256": sha_file(paths["author"]), "attention_sha256": best_attention_sha,
        "restore": {"state_equal": restore_state_equal, "rng_equal": restore_rng_equal,
                    "identity_equal": restore_identity_equal},
        "seconds": time.perf_counter() - started,
        "epochs": len(author_rows), "rng_trace": trace,
        "performance": best["metrics"],
        "patient_predictions": best["prediction"].tolist(),
        "patient_probabilities": best["probability"].tolist(),
        "train_members_1based": [int(i) + 1 for i in state["train_np"]],
        "test_members_1based": [int(i) + 1 for i in state["test_np"]],
        "dropout_order": sorted(set(tuple(row["dropout_order"]) for row in history_rows)),
        "best_attention_path": str(paths["attention"]),
        "historical_best_exact": True, "historical_latest_exact": latest_checks,
        "next_step_restore_exact": next_step_exact,
    }
    module.atomic_json(OUT / "tasks" / f"fold{fold:02d}.audit.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    module = load_runner()
    module.RT = OUT / "runtime"
    module.TASKS = module.RT / "tasks"
    module.CKPTS = module.RT / "checkpoints"
    module.LOGS = module.RT / "logs"
    module.TASKS.mkdir(parents=True, exist_ok=True)
    module.CKPTS.mkdir(parents=True, exist_ok=True)
    module.LOGS.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    history = install_history_path_adapter(module, torch, np)
    base = history.load_e69().load_e61().load_base()
    torch, np = base.runtime()["torch"], base.runtime()["np"]
    torch.cuda.set_device(0)
    dcr = module.load_dcr()
    cadr_source = V2 / "inputs/source_repo/experiment/experiment53_3_cadr_graph_fallback_fix/cadr_gfr.py"
    if sha_file(cadr_source) != sha_file(module.E53 / "cadr_gfr.py"):
        raise RuntimeError("recovered E53 CADR source does not match the formal dependency")
    dcr.CADR_PATH = cadr_source
    cadr_module = dcr._cadr_module()
    if Path(cadr_module.__file__).resolve() != cadr_source.resolve():
        raise RuntimeError("DARG loaded CADR from outside the isolated source snapshot")
    folds = [args.fold] if args.fold else list(range(1, 11))
    rows = []
    for fold in folds:
        audit_path = OUT / "tasks" / f"fold{fold:02d}.audit.json"
        if audit_path.is_file():
            saved = json.loads(audit_path.read_text(encoding="utf-8"))
            if saved.get("status") != "TRAINED_AUDITED":
                raise RuntimeError(f"existing DARG fold {fold} has an incomplete audit record")
            paths = fold_paths(module, fold)
            required = {
                "best_sha256": paths["best"], "latest_sha256": paths["latest"],
                "history_sha256": paths["history"].with_suffix(".full.jsonl"),
                "author_sha256": paths["author"], "attention_sha256": paths["attention"],
            }
            if any(not path.is_file() or sha_file(path) != saved[key] for key, path in required.items()):
                raise RuntimeError(f"existing DARG fold {fold} artifacts no longer match its audit record")
            rows.append(saved)
            continue
        rows.append(train_fold(module, fold, base, np, torch))
    module.atomic_json(OUT / "results" / "retrain_summary.json", {
        "group": GROUP, "folds": rows,
        "status": "TRAINED_AUDITED" if len(rows) == 10 else "PARTIAL",
        "total_seconds": sum(row["seconds"] for row in rows),
    })


if __name__ == "__main__":
    main()
