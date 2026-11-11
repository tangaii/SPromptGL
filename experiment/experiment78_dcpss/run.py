#!/usr/bin/env python3
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.dont_write_bytecode = True
from path_adapter import HERE, BASE, REPO, load_base_runner, sha_file
from dcpss import DCPSS, deterministic_assignments, initialize

CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
GROUPS = tuple(CFG["groups"])
BASE_GROUP = CFG["base_group"]
E61 = REPO / "experiment/experiment61_cadr_postalign_vs_e2e"
RESULTS = HERE / "results"
CHECKPOINTS = HERE / "checkpoints"
TASKS = HERE / "runtime/tasks"
LOGS = HERE / "runtime/logs"
BASE_MODULE = None


def base():
    global BASE_MODULE
    if BASE_MODULE is None:
        BASE_MODULE = load_base_runner()
    return BASE_MODULE


def hist():
    return base().load_history()


def runtime():
    return hist().load_e69().load_e61().load_base()


def sha_object(value):
    return hist().sha_object(value)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def log_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_map():
    sources = {
        str((HERE / name).resolve()): sha_file(HERE / name)
        for name in ("config.json", "dcpss.py", "run.py", "audit.py", "path_adapter.py", "make_text_cache.py", "runtime/text_embeddings.pt", "results/source_review.md")
    }
    sources.update(base().source_manifest())
    for fold in range(1, 11):
        token = f"{BASE_GROUP}_f{fold:02d}"
        for relative in (
            f".runtime/tasks/{token}.json",
            f".runtime/checkpoints/{token}.best.pt",
            f".runtime/checkpoints/{token}.latest.pt",
        ):
            path = BASE / relative
            sources[str(path.resolve())] = sha_file(path)
        warmup = E61 / ".runtime/warmup" / f"fold{fold:02d}.pt"
        sources[str(warmup.resolve())] = sha_file(warmup)
    return dict(sorted(sources.items()))


def source_sha():
    return sha_object(source_map())


def forward_with_cam144(state, training=False):
    captured = {}
    output_layer = state["cam"].Outputlayer

    def capture_input(_module, args):
        captured["cam144"] = args[0]

    hook = output_layer.register_forward_pre_hook(capture_input)
    try:
        out = base().forward(state, BASE_GROUP, training=training)
    finally:
        hook.remove()
    h = captured.get("cam144")
    if h is None or tuple(h.shape) != (871, 144):
        shape = None if h is None else tuple(h.shape)
        raise RuntimeError(f"CAM OutputLayer input must be [871,144], got {shape}")
    out["cam144"] = h
    return out


def task_paths(group, fold):
    token = f"{group}_f{fold:02d}"
    return TASKS / f"{token}.json", CHECKPOINTS / f"{token}.best.pt", CHECKPOINTS / f"{token}.latest.pt", LOGS / f"{token}.jsonl"


def text_payload(group):
    payload = torch.load(HERE / "runtime/text_embeddings.pt", map_location="cpu", weights_only=False)
    if sha_file(HERE / "runtime/text_embeddings.pt") != CFG["text_cache_sha256"]:
        raise RuntimeError("CLIP text cache SHA differs from frozen config")
    spec = CFG["group_specs"][group]
    source = spec["prompt_source"]
    if source == "spromptgl":
        return payload["spromptgl"], CFG["prompts"]["spromptgl_classes"]
    if source == "brainprompt":
        return payload["brainprompt"], CFG["prompts"]["brainprompt_classes"]
    return payload["anonymous"], CFG["prompts"]["anonymous_classes"]


def warmup_features(state, before_sha):
    with torch.no_grad():
        out = forward_with_cam144(state, training=False)
    if runtime().rng_sha(runtime().pack_rng()) != before_sha:
        raise RuntimeError("warmup cluster extraction changed public RNG")
    h = out["cam144"]
    if tuple(h.shape) != (871, 144):
        raise RuntimeError(f"CAM pre-classifier representation has unexpected shape {tuple(h.shape)}")
    return h.detach()


def initial_state(fold, group):
    state, warmup, identity = base().prepare(fold)
    state["fold"] = fold
    state["group"] = group
    state["rudag_identity"] = identity
    rt = runtime()
    rng_before = rt.rng_sha(rt.pack_rng())
    if rng_before != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold} did not restore the exact epoch-10 warmup RNG")
    support = np.setdiff1d(state["train_np"], state["test_np"], assume_unique=False).astype(np.int64)
    if len(np.intersect1d(support, state["test_np"])):
        raise RuntimeError("outer-test patient entered DCPSS support set")
    h0 = warmup_features(state, rng_before)
    k = int(CFG["group_specs"][group]["k"])
    assignments = deterministic_assignments(h0, state["y"], support, k)
    embeddings, classes = text_payload(group)
    module = initialize(embeddings, classes, k, fold, state["y"].device)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=CFG["training"]["new_lr"],
        weight_decay=CFG["training"]["new_weight_decay"],
    )
    if rt.rng_sha(rt.pack_rng()) != rng_before:
        raise RuntimeError("new module initialization consumed shared RNG")
    initial = {
        "identity": identity,
        "warmup_sha256": identity["warmup_sha256"],
        "support_sha256": sha_object(support),
        "assignment_sha256": sha_object(assignments.cpu()),
        "module_init_sha256": sha_object({k: v.detach().cpu() for k, v in module.named_parameters()}),
        "base_optimizer_sha256": identity["optimizer_sha"],
        "public_rng_sha256": rng_before,
        "new_optimizer_sha256": sha_object(optimizer.state_dict()),
        "new_parameter_count": sum(p.numel() for p in module.parameters()),
    }
    return state, warmup, identity, support, assignments, module, optimizer, initial


def _bundle(state):
    return hist().state_bundle(state)


def _snapshot(state, module, optimizer):
    return {
        "base_state": _bundle(state),
        "module_state": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
        "new_optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": runtime().pack_rng(),
    }


def _restore(state, module, optimizer, payload):
    hist().load_bundle(state, payload["base_state"])
    module.load_state_dict(payload["module_state"], strict=True)
    optimizer.load_state_dict(payload["new_optimizer"])
    runtime().restore_rng(payload["rng"])


def _metric_values(logp, indices, labels):
    if torch.is_tensor(indices):
        indices = indices.detach().cpu().numpy()
    return runtime().classification_metrics(logp, indices, labels)


def _compact_metrics(metrics):
    return {k: v for k, v in metrics.items() if k not in ("prob", "pred")}


def _legal_indices(state):
    return np.setdiff1d(state["train_np"], state["test_np"], assume_unique=False)


def _train_forward(state, module, support, assignments):
    out = forward_with_cam144(state, training=True)
    qids = torch.as_tensor(_legal_indices(state), dtype=torch.long, device=out["cam144"].device)
    module.train()
    dcp = module(
        out["cam144"], out["logp"], state["y"], support, assignments, qids,
    )
    merged = out["logp"].clone()
    merged[qids] = dcp["logp"]
    cls_final = runtime().runtime()["utils"].ClsLoss(merged, state["y"], state["train"], state["weights"])
    _, aux = module.losses(dcp, out["logp"], state["y"], state["weights"])
    total = out["total"] - out["cls"] + cls_final + 0.05 * aux["sem"] + 0.02 * aux["gain"]
    return out, dcp, aux, total, qids


def _train_step(state, module, optimizer, support, assignments, collect_gradients=False):
    hist().zero_optimizers(state)
    optimizer.zero_grad(set_to_none=True)
    out, dcp, aux, total, qids = _train_forward(state, module, support, assignments)
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite E78 total loss")
    grad_record = {}
    if collect_gradients:
        targets = {
            "darg": state["model"].adaptive_gate_logits,
            "cam": state["cam"].Outputlayer.mlp_head[1].weight,
            "patient_projection": module.patient_projection.weight,
            "prompt_projection": module.text_projection.weight,
            "residual_alpha": module.alpha,
        }
        gradients = torch.autograd.grad(total, tuple(targets.values()), retain_graph=True, allow_unused=True)
        for (name, _), gradient in zip(targets.items(), gradients):
            grad_record[name] = 0.0 if gradient is None else float(gradient.detach().norm().cpu())
        if any((not np.isfinite(v)) or v <= 0 for v in grad_record.values()):
            raise RuntimeError(f"active DCPSS gradient path failed: {grad_record}")
    total.backward()
    for name in ("opt_base", "opt_cam", "opt_added"):
        state[name].step()
    optimizer.step()
    losses = {
        "base_total_before_replacement": float(out["total"].detach().cpu()),
        "base_cls": float(out["cls"].detach().cpu()),
        "final_cls_all_original_train": float((total - (out["total"] - out["cls"]) - 0.05 * aux["sem"] - 0.02 * aux["gain"]).detach().cpu()),
        "sem": float(aux["sem"].detach().cpu()),
        "gain": float(aux["gain"].detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    diagnostics = {
        "gradients": grad_record,
        "mean_abs_delta_train": float(dcp["delta"].detach().abs().mean().cpu()),
        "gamma": float(dcp["gamma"].detach().cpu()),
        "mean_prompt_entropy": float(torch.stack([
            -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(-1).mean() for a in dcp["attention"]
        ]).mean().detach().cpu()),
        "mean_beta_entropy": float((-(dcp["beta"].clamp_min(1e-12) * dcp["beta"].clamp_min(1e-12).log()).sum(-1)).mean().detach().cpu()),
        "prototype_counts_min": float(dcp["counts"].detach().min().cpu()),
        "query_count": int(len(qids)),
    }
    return losses, diagnostics


def _evaluate(state, module, support, assignments):
    module.eval()
    with torch.no_grad():
        out = forward_with_cam144(state, training=False)
        test_ids = np.asarray(state["test_np"], dtype=np.int64)
        qids = torch.as_tensor(test_ids, dtype=torch.long, device=out["cam144"].device)
        dcp = module(out["cam144"], out["logp"], state["y"], support, assignments, qids)
        base_metrics = _metric_values(out["logp"], state["test"], state["y"])
        final_full = out["logp"].clone()
        final_full[qids] = dcp["logp"]
        metrics = _metric_values(final_full, state["test"], state["y"])
        record = {
            "logp": dcp["logp"].detach().cpu().clone(),
            "base_logp": out["logp"][qids].detach().cpu().clone(),
            "metrics": metrics,
            "base_metrics": base_metrics,
            "evidence": dcp["evidence"].detach().cpu().clone(),
            "attention": [v.detach().cpu().clone() for v in dcp["attention"]],
            "beta": dcp["beta"].detach().cpu().clone(),
            "delta": dcp["delta"].detach().cpu().clone(),
            "counts": dcp["counts"].detach().cpu().clone(),
        }
    return record


def base_reference(fold):
    task_path = BASE / ".runtime/tasks" / f"{BASE_GROUP}_f{fold:02d}.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    log_path = BASE / ".runtime/logs" / f"{BASE_GROUP}_f{fold:02d}.jsonl"
    logs = {}
    with log_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            logs[int(row["epoch"])] = row
    best_path = BASE / ".runtime/checkpoints" / f"{BASE_GROUP}_f{fold:02d}.best.pt"
    latest_path = BASE / ".runtime/checkpoints" / f"{BASE_GROUP}_f{fold:02d}.latest.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    return {
        "task": task,
        "rng": {int(x["epoch"]): x for x in task["rng_trace"]},
        "logs": logs,
        "best_epoch": int(best["best_epoch_1based"]),
        "best_losses": best["best_losses"],
        "best_prob": np.asarray(best["test_probability"]),
        "state_sha": {
            int(best["best_epoch_1based"]): sha_object(best["best_state"]),
            int(latest["next_epoch"]): sha_object(latest["current"]),
        },
    }


def train_task(group, fold, expected_source_sha):
    task_path, best_path, latest_path, log_path = task_paths(group, fold)
    if task_path.is_file():
        existing = json.loads(task_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE":
            if existing.get("source_sha256") != expected_source_sha or sha_file(best_path) != existing.get("checkpoint_sha256"):
                raise RuntimeError(f"completed task identity mismatch: {task_path}")
            return existing
        raise RuntimeError(f"refusing to overwrite non-complete task record: {task_path}")

    state, warmup, identity, support, assignments, module, optimizer, initial = initial_state(fold, group)
    reference = base_reference(fold)
    start_epoch, best_acc, best_epoch, wait = 10, -1.0, -1, 0
    epochs_executed = 10
    trace, mechanism = [], []
    if latest_path.exists():
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        for key, expected in (("source_sha256", expected_source_sha), ("identity", identity), ("initial", initial)):
            if saved.get(key) != expected:
                raise RuntimeError(f"incompatible recovery state {key} for {group} fold {fold}")
        _restore(state, module, optimizer, saved)
        start_epoch, best_acc, best_epoch, wait = (saved[k] for k in ("next_epoch", "best_acc", "best_epoch", "wait"))
        epochs_executed = int(saved["next_epoch"])
        trace, mechanism = saved["trace"], saved["mechanism"]
        print(f"RESUME {group} fold={fold} epoch={start_epoch + 1}", flush=True)

    start_time = time.time()
    for epoch in range(start_epoch, CFG["max_epoch"]):
        rng_before = runtime().rng_sha(runtime().pack_rng())
        losses, diag = _train_step(state, module, optimizer, support, assignments, collect_gradients=(epoch == start_epoch))
        rng_after_train = runtime().rng_sha(runtime().pack_rng())
        evaluated = _evaluate(state, module, support, assignments)
        rng_after_eval = runtime().rng_sha(runtime().pack_rng())
        epoch_number = epoch + 1
        epochs_executed = epoch_number
        historical_rng = reference["rng"].get(epoch_number)
        rng_record = {
            "epoch": epoch_number,
            "rng_before_train": rng_before,
            "rng_after_train": rng_after_train,
            "rng_after_eval": rng_after_eval,
        }
        if historical_rng is not None:
            expected_tuple = tuple(historical_rng[k] for k in ("rng_before", "rng_after_train", "rng_after_eval"))
            actual_tuple = (rng_before, rng_after_train, rng_after_eval)
            if actual_tuple != expected_tuple:
                raise RuntimeError(f"public RNG trace differs from E74.4 fold={fold} epoch={epoch_number}")

        if epoch_number in reference["logs"] and epoch_number == 11:
            base_loss = losses["base_total_before_replacement"]
            if base_loss != reference["logs"][epoch_number]["loss"]:
                raise RuntimeError(f"initial base loss mismatch fold={fold}: {base_loss}")
        improved = float(evaluated["metrics"]["acc"]) > best_acc
        if improved:
            best_acc, best_epoch, wait = float(evaluated["metrics"]["acc"]), epoch_number, 0
            payload = _snapshot(state, module, optimizer) | {
                "group": group,
                "fold": fold,
                "source_sha256": expected_source_sha,
                "identity": identity,
                "initial": initial,
                "warmup_sha256": identity["warmup_sha256"],
                "best_epoch_1based": epoch_number,
                "metrics": _compact_metrics(evaluated["metrics"]),
                "base_metrics": _compact_metrics(evaluated["base_metrics"]),
                "test_probability": np.asarray(evaluated["metrics"]["prob"]).tolist(),
                "test_prediction": np.asarray(evaluated["metrics"]["pred"]).tolist(),
                "base_probability": np.asarray(evaluated["base_metrics"]["prob"]).tolist(),
                "base_prediction": np.asarray(evaluated["base_metrics"]["pred"]).tolist(),
                "test_ids": np.asarray(state["test_np"]).tolist(),
                "support_ids": support.tolist(),
                "assignments": assignments.detach().cpu().tolist(),
                "losses": losses,
                "mechanism": diag,
                "evidence": evaluated["evidence"].tolist(),
                "attention": [v.tolist() for v in evaluated["attention"]],
                "beta": evaluated["beta"].tolist(),
                "delta": evaluated["delta"].tolist(),
                "prototype_counts": evaluated["counts"].tolist(),
                "rng": rng_record,
                "trace": trace + [rng_record],
            }
            save_torch(best_path, payload)
        else:
            wait += 1

        trace.append(rng_record)
        mechanism_row = {
            "epoch": epoch_number,
            "base_total": losses["base_total_before_replacement"],
            "base_cls": losses["base_cls"],
            "final_cls": losses["final_cls_all_original_train"],
            "semantic_loss": losses["sem"],
            "gain_loss": losses["gain"],
            "total_loss": losses["total"],
            **diag,
            **rng_record,
        }
        mechanism.append(mechanism_row)
        log_json(log_path, {
            "group": group,
            "fold": fold,
            "epoch": epoch_number,
            "best_epoch": best_epoch,
            "best_acc": best_acc,
            "metrics": _compact_metrics(evaluated["metrics"]),
            "base_metrics": _compact_metrics(evaluated["base_metrics"]),
            "losses": losses,
            "mechanism": diag,
            "rng": rng_record,
        })

        if epoch_number % 10 == 0 or epoch_number == CFG["max_epoch"] or wait > CFG["early_stop_wait_gt"]:
            save_torch(latest_path, _snapshot(state, module, optimizer) | {
                "group": group,
                "fold": fold,
                "source_sha256": expected_source_sha,
                "identity": identity,
                "initial": initial,
                "warmup_sha256": identity["warmup_sha256"],
                "next_epoch": epoch_number,
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "wait": wait,
                "trace": trace,
                "mechanism": mechanism,
            })
            print(f"{group} fold={fold} epoch={epoch_number} best={best_acc:.6f}@{best_epoch} wait={wait}", flush=True)
        del evaluated
        if wait > CFG["early_stop_wait_gt"]:
            break

    if best_epoch < 0:
        raise RuntimeError(f"no strict-ACC checkpoint selected for {group} fold={fold}")
    checkpoint_sha = sha_file(best_path)
    task = {
        "status": "COMPLETE",
        "group": group,
        "fold": fold,
        "seed": CFG["seed"],
        "source_sha256": expected_source_sha,
        "checkpoint_sha256": checkpoint_sha,
        "identity": identity,
        "initial": initial,
        "warmup_sha256": identity["warmup_sha256"],
        "best_epoch_1based": best_epoch,
        "epochs_executed": epochs_executed,
        "metrics": torch.load(best_path, map_location="cpu", weights_only=False)["metrics"],
        "trace": trace,
        "historical_reference_epochs": len(reference["rng"]),
        "wall_seconds": time.time() - start_time,
        "device": torch.cuda.get_device_name(0),
        "checkpoint_sha256_final": checkpoint_sha,
    }
    save_json(task_path, task)
    print(f"COMPLETE {group} fold={fold} best={best_acc:.6f}@{best_epoch}", flush=True)
    return task


def worker(fold, expected_source_sha):
    try:
        results = []
        assignments = {}
        initials = {}
        traces = {}
        for group in GROUPS:
            result = train_task(group, fold, expected_source_sha)
            results.append(result)
            initials[group] = result["initial"]["module_init_sha256"]
            assignments[group] = result["initial"]["assignment_sha256"]
            traces[group] = result["trace"]
        if len(set(initials.values())) != 1:
            raise RuntimeError(f"new module parameters were not identically initialized across groups in fold {fold}")
        k3_groups = [g for g in GROUPS if CFG["group_specs"][g]["k"] == 3]
        if len({assignments[g] for g in k3_groups}) != 1:
            raise RuntimeError(f"K=3 cluster assignments differ across groups in fold {fold}")
        if len({r["initial"]["public_rng_sha256"] for r in results}) != 1:
            raise RuntimeError(f"public RNG initialization differs across groups in fold {fold}")
        trace_maps = {
            group: {int(row["epoch"]): row for row in rows}
            for group, rows in traces.items()
        }
        common_epochs = set.intersection(*(set(items) for items in trace_maps.values()))
        for epoch in common_epochs:
            digest_triplets = {
                tuple(trace_maps[group][epoch][key] for key in ("rng_before_train", "rng_after_train", "rng_after_eval"))
                for group in GROUPS
            }
            if len(digest_triplets) != 1:
                raise RuntimeError(f"public RNG common-epoch parity failed fold={fold} epoch={epoch}")
        save_json(RESULTS / f"fold_parity_f{fold:02d}.json", {
            "status": "PASS",
            "fold": fold,
            "base_initial_identity_equal": len({r["identity"]["model_cam_sha"] for r in results}) == 1,
            "base_optimizer_initial_equal": len({r["identity"]["optimizer_sha"] for r in results}) == 1,
            "new_module_init_equal": len(set(initials.values())) == 1,
            "k3_assignment_equal": len({assignments[g] for g in k3_groups}) == 1,
            "common_epoch_count": len(common_epochs),
            "public_rng_common_epoch_parity": True,
        })
    except Exception:
        save_json(HERE / "runtime/failures" / f"fold{fold:02d}.json", {"traceback": traceback.format_exc()})
        raise


def run_all(workers):
    if not 1 <= workers <= 10:
        raise ValueError("workers must be between 1 and 10")
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    expected = source_sha()
    if preflight.get("status") != "PASS" or preflight.get("source_sha256") != expected:
        raise RuntimeError("formal source identity does not match the passing preflight")
    save_json(HERE / "manifest.json", {
        "experiment": CFG["experiment"],
        "status": "FORMAL_RUNNING",
        "source_sha256": expected,
        "frozen_sources": source_map(),
        "groups": list(GROUPS),
        "formal_tasks": 50,
        "workers": workers,
        "preflight_sha256": sha_file(RESULTS / "preflight.json"),
    })
    pending = list(range(1, 11))
    active, finished = {}, []
    started = time.time()
    while pending or active:
        while pending and len(active) < workers:
            fold = pending.pop(0)
            path = LOGS / f"worker_f{fold:02d}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a", encoding="utf-8")
            proc = subprocess.Popen(
                [sys.executable, str(HERE / "run.py"), "--worker", str(fold), "--expected", expected],
                stdout=stream,
                stderr=subprocess.STDOUT,
                cwd=REPO,
            )
            active[fold] = (proc, stream)
        for fold, (proc, stream) in list(active.items()):
            if proc.poll() is not None:
                stream.close()
                del active[fold]
                finished.append({"fold": fold, "returncode": proc.returncode})
                if proc.returncode:
                    pending = []
        save_json(HERE / "runtime/scheduler.json", {
            "active_folds": list(active),
            "pending_folds": pending,
            "finished": finished,
            "elapsed_seconds": time.time() - started,
        })
        if active:
            time.sleep(3)
    if len(finished) != 10 or any(x["returncode"] for x in finished):
        raise RuntimeError(f"formal fold workers failed: {finished}")
    from audit import finalize
    finalize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected")
    args = parser.parse_args()
    if args.preflight:
        from audit import preflight
        preflight()
    elif args.all:
        run_all(args.workers)
    elif args.worker:
        worker(args.worker, args.expected)
    elif args.audit:
        from audit import finalize
        finalize()
    else:
        parser.error("choose --preflight, --all, --audit, or --worker")


if __name__ == "__main__":
    main()
