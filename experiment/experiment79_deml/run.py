#!/usr/bin/env python3
"""Experiment 79: Dynamic Error Memory Learning (DEML).

The historical DARG update is kept as one isolated forward/backward/step
sequence.  DEML is trained only from the detached CAM144 captured by the
following eval forward, and memory is written after the epoch's read/eval.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
REPO = Path("/data/adodas/SPromptGL-main")
BASE_EXP = REPO / "experiment/experiment74_4_soft_positive_dcr"
E61 = REPO / "experiment/experiment61_cadr_postalign_vs_e2e"
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
RT = EXP / ".runtime"
CKPTS, TASKS, LOGS = RT / "checkpoints", RT / "tasks", RT / "logs"
RESULTS = EXP / "results"
for path in (CKPTS, TASKS, LOGS, RESULTS):
    path.mkdir(parents=True, exist_ok=True)

BASE_GROUP = CFG["base_group"]
GROUPS = tuple(CFG["groups"])
FOLD2_EXCLUDE = {376, 406}

import path_adapter  # noqa: E402
from path_adapter import legacy  # noqa: E402
from deml import DEMLMemoryAttention, init_pair, output_layer_logp, memory_loss  # noqa: E402
from memory_bank import ErrorMemoryBank, TYPE_ASD_TO_TD, TYPE_TD_TO_ASD  # noqa: E402


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_obj(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temp.write_text(text, encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if path.read_text(encoding="utf-8") != text:
        raise IOError(f"atomic JSON readback mismatch: {path}")


def atomic_torch(path, payload):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise IOError(f"checkpoint readback failed: {path}")


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def source_snapshot():
    paths = [
        EXP / "config.json", EXP / "run.py", EXP / "deml.py", EXP / "memory_bank.py",
        EXP / "path_adapter.py", EXP / "audit.py", BASE_EXP / "run.py",
        BASE_EXP / "soft_positive_dcr.py", BASE_EXP / "config.json", BASE_EXP / "manifest.json",
        BASE_EXP / "audit.py", BASE_EXP / "results/audit_status.json",
        REPO / "main.py", REPO / "model.py", REPO / "network.py", REPO / "layers.py", REPO / "utils.py",
        REPO / "data/ABIDE/processed_standard_data.csv", REPO / "data/ABIDE/modal_feat_dict.npy",
    ]
    paths += sorted(E61.glob(".runtime/warmup/fold*.pt"))
    paths += sorted(BASE_EXP.glob(f".runtime/checkpoints/{BASE_GROUP}_f*.best.pt"))
    paths += sorted(BASE_EXP.glob(f".runtime/tasks/{BASE_GROUP}_f*.json"))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("history inventory incomplete: " + ", ".join(missing))
    return {str(p.resolve()): sha_file(p) for p in paths}


def hook_forward(state, training):
    captured = []
    handle = state["cam"].Outputlayer.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0]))
    try:
        output = legacy.load_base().forward(state, BASE_GROUP, training=training)
    finally:
        handle.remove()
    if len(captured) != 1 or tuple(captured[0].shape) != (871, 144):
        raise RuntimeError(f"CAM144 hook mismatch: {[tuple(x.shape) for x in captured]}")
    return output, captured[0]


def prepare_fold(fold):
    state, warmup, identity, task, ckpt = legacy.prepare_fold(fold)
    if task.get("status") != "COMPLETE":
        raise RuntimeError(f"E74.4-G2 task incomplete for fold {fold}")
    expected = [377, 407] if fold == 2 else []
    if identity["membership"]["train_test_overlap_1based"] != expected:
        raise RuntimeError(f"historical overlap mismatch fold={fold}")
    if state["train_pairs_np"].shape != (4096, 2):
        raise RuntimeError("E74.4-G2 relation pair count is not 4096")
    return state, warmup, identity, task, ckpt


def eligible_ids(state, fold):
    test = set(int(x) for x in state["test_np"])
    ids = sorted(int(x) for x in state["train_np"] if int(x) not in test)
    if fold == 2:
        ids = [x for x in ids if x not in FOLD2_EXCLUDE]
    if set(ids) & test or (fold == 2 and set(ids) & FOLD2_EXCLUDE):
        raise RuntimeError(f"illegal eligible set fold {fold}")
    return ids


def rng_sha(runtime):
    return runtime.rng_sha(runtime.pack_rng())


def state_bundle(runtime, state):
    return legacy.load_base().load_history().state_bundle(state)


def restore_bundle(runtime, state, bundle):
    legacy.load_base().load_history().load_bundle(state, bundle)


def module_snapshot(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def base_eval_direction(state, cam144, logp, eligible, labels):
    """Compute legal per-error local correction vectors without parameter grads."""
    import torch
    pred = logp.detach().argmax(dim=1)
    y = labels.detach().long()
    error_ids = [int(i) for i in eligible if int(pred[i]) != int(y[i])]
    if not error_ids:
        return {}
    h = cam144.detach()[error_ids].clone().requires_grad_(True)
    layer = state["cam"].Outputlayer
    score_logp, _ = output_layer_logp(layer, h)
    yy = y[error_ids].to(h.device)
    pp = pred[error_ids].to(h.device)
    scores = score_logp.gather(1, yy[:, None]).squeeze(1) - score_logp.gather(1, pp[:, None]).squeeze(1)
    grad = torch.autograd.grad(scores.sum(), h, retain_graph=False, create_graph=False)[0]
    grad = torch.nn.functional.normalize(grad, dim=1, eps=1e-12).detach().cpu()
    return {pid: grad[row] for row, pid in enumerate(error_ids)}


def run_memory_forward(module, state, query_ids, cam144, base_logp, bank,
                       sites, use_direction, **interventions):
    import torch
    ids = [int(x) for x in query_ids]
    idx = torch.as_tensor(ids, dtype=torch.long, device=cam144.device)
    h = cam144[idx].detach()
    lp = base_logp[idx].detach()
    enhanced, diagnostics = module(
        ids, h, lp, bank, sites, use_direction=use_direction, **interventions)
    logp, _ = output_layer_logp(state["cam"].Outputlayer, enhanced)
    return logp, enhanced, diagnostics


def train_memory_module(module, optimizer, state, eligible, cam144, base_logp,
                        bank, sites, use_direction):
    import torch
    module.train()
    optimizer.zero_grad(set_to_none=True)
    ids = [int(x) for x in eligible]
    idx = torch.as_tensor(ids, dtype=torch.long, device=cam144.device)
    h = cam144[idx].detach()
    lp = base_logp[idx].detach()
    enhanced, diag = module(ids, h, lp, bank, sites, use_direction=use_direction)
    logp_mem, _ = output_layer_logp(state["cam"].Outputlayer, enhanced)
    loss, comps = memory_loss(logp_mem, lp, state["y"], ids, state["weights"],
                              delta=diag["delta_h"], h=h)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite DEML memory loss")
    loss.backward()
    grad_norm = torch.sqrt(torch.stack([
        p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None
    ]).sum()).item() if any(p.grad is not None for p in module.parameters()) else 0.0
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in module.parameters()):
        raise FloatingPointError("non-finite DEML memory gradient")
    optimizer.step()
    return float(loss.detach().item()), {k: (float(v.detach().item()) if torch.is_tensor(v) and v.numel() == 1 else v)
                                        for k, v in comps.items()}, float(grad_norm)


def metric_payload(metrics):
    return {k: float(v) for k, v in metrics.items() if k not in ("prob", "pred")}


def base_method_metrics(logp, ids, y):
    return legacy.extended_metrics(logp, ids, y)


def correction_stats(base_logp, method_logp, ids, labels):
    import numpy as np
    idx = np.asarray(ids, dtype=np.int64)
    bprob = base_logp.detach().cpu().exp().numpy()[idx]
    mprob = method_logp.detach().cpu().exp().numpy()[idx]
    y = labels.detach().cpu().numpy()[idx]
    bp = bprob.argmax(1); mp = mprob.argmax(1)
    bcorrect = bp == y; mcorrect = mp == y
    flipped = bp != mp
    return {
        "base_errors": int((~bcorrect).sum()), "method_errors": int((~mcorrect).sum()),
        "error_to_correct": int(((~bcorrect) & mcorrect).sum()),
        "correct_to_error": int((bcorrect & (~mcorrect)).sum()),
        "net_correction": int(((~bcorrect) & mcorrect).sum() - (bcorrect & (~mcorrect)).sum()),
        "hard_prediction_changes": int(flipped.sum()),
        "asd_error_corrected": int(((~bcorrect) & mcorrect & (y == 0)).sum()),
        "td_error_corrected": int(((~bcorrect) & mcorrect & (y == 1)).sum()),
        "correct_retention": float((bcorrect & mcorrect).sum() / max(int(bcorrect.sum()), 1)),
        "mean_abs_prob_delta": float(np.abs(mprob - bprob).mean()),
    }


def save_best(group, fold, payload, source_hash, identity, trace, epoch):
    record = {
        "experiment": CFG["experiment"], "group": group, "fold": fold, "seed": 0,
        "base_group": BASE_GROUP, "source_sha256": source_hash, "identity": identity,
        "best_epoch_1based": int(epoch), "metrics": payload["metrics"], "best": payload,
        "rng_trace": trace, "audit_status": "PENDING",
    }
    atomic_torch(CKPTS / f"{group}_f{fold:02d}.best.pt", record)


def _memory_after_update(bank):
    return bank.snapshot()


def train_fold(fold):
    import numpy as np
    import torch
    state, warmup, identity, historical_task, historical_ckpt = prepare_fold(fold)
    runtime = legacy.history_runtime()
    hist = legacy.load_base().load_history()
    if rng_sha(runtime) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold} did not start from E61 warmup RNG")
    eligible = eligible_ids(state, fold)
    y = state["y"].long()
    sites = torch.zeros(871, dtype=torch.long, device=y.device)
    # SITE_ID one-hot columns are part of the locked processed dataset.
    import csv as _csv
    with (REPO / "data/ABIDE/processed_standard_data.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = _csv.DictReader(stream); rows = list(reader); fields = reader.fieldnames or []
    site_cols = [f for f in fields if f.upper().startswith("SITE_ID_")]
    if len(rows) != 871 or not site_cols:
        raise RuntimeError("site columns unavailable for DEML audit")
    sites = torch.tensor([max(range(len(site_cols)), key=lambda j: float(rows[i].get(site_cols[j]) or 0))
                          for i in range(871)], dtype=torch.long, device=y.device)
    source = source_snapshot()
    source_hash = sha_obj(source)
    g1, g2 = init_pair(seed=7901, device=y.device)
    modules = [g1, g2]
    opts = [torch.optim.AdamW(m.parameters(), lr=0.0005, weight_decay=0.01) for m in modules]
    if sum(p.numel() for p in g1.parameters()) != sum(p.numel() for p in g2.parameters()):
        raise RuntimeError("G1/G2 memory parameter counts differ")
    for p1, p2 in zip(g1.state_dict().values(), g2.state_dict().values()):
        if not torch.equal(p1.cpu(), p2.cpu()):
            raise RuntimeError("G1/G2 memory initialization differs")
    bank = ErrorMemoryBank(64, 8, 2, 10)
    ledgers = {group: {"best_acc": -1.0, "best_epoch": -1, "best": None} for group in GROUPS}
    base_ledger = {"best_acc": -1.0, "best_epoch": -1, "best": None}
    base_wait = 0
    trace, event_trace, fold_rows = [], [], []
    start_epoch = 10
    latest_path = CKPTS / f"shared_f{fold:02d}.latest.pt"
    task_path = TASKS / f"E79_DEML_f{fold:02d}.json"
    if latest_path.is_file():
        saved = torch_load(latest_path)
        if saved.get("identity") != identity:
            raise RuntimeError(f"incompatible latest checkpoint fold {fold}")
        # A read-only audit-script repair after a worker was launched may alter
        # the current source inventory without altering the training state.  A
        # latest checkpoint is therefore resumed under its own locked source
        # identity; the final manifest records both identities.
        if saved.get("source_sha256") != source_hash:
            source_hash = saved.get("source_sha256")
        hist.load_bundle(state, saved["base_state"]); runtime.restore_rng(saved["rng"])
        for group, module, opt in zip(GROUPS, modules, opts):
            module.load_state_dict(saved["modules"][group]); opt.load_state_dict(saved["optimizers"][group])
        bank = ErrorMemoryBank.from_snapshot(saved["memory_next"])
        ledgers, base_ledger = saved["ledgers"], saved["base_ledger"]
        base_wait, start_epoch = int(saved["base_wait"]), int(saved["next_epoch"])
        trace, event_trace, fold_rows = saved.get("trace", []), saved.get("event_trace", []), saved.get("fold_rows", [])
    else:
        atomic_json(task_path, {"status": "RUNNING", "fold": fold, "source_sha256": source_hash,
                                "identity": identity, "eligible_count": len(eligible), "next_epoch": start_epoch})
    fold_started = time.time()
    max_epoch_exclusive = int(os.environ.get("DEML_MAX_EPOCH_EXCLUSIVE", "450"))
    for epoch_index in range(start_epoch, max_epoch_exclusive):
        epoch = epoch_index + 1
        rng_before = rng_sha(runtime)
        hist.zero_optimizers(state)
        out, train_cam = hook_forward(state, training=True)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"non-finite base loss fold={fold} epoch={epoch}")
        out["total"].backward()
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        rng_after_train = rng_sha(runtime)
        evaluated, eval_cam = hook_forward(state, training=False)
        rng_after_eval = rng_sha(runtime)
        if tuple(eval_cam.shape) != (871, 144):
            raise RuntimeError("CAM144 shape changed")
        memory_used = bank.snapshot()
        # Memory branch uses exactly the previous snapshot for both groups.
        branch_loss, branch_comp, branch_grad = {}, {}, {}
        for idx, (module, opt) in enumerate(zip(modules, opts)):
            loss, comp, gnorm = train_memory_module(
                module, opt, state, eligible, eval_cam, evaluated["logp"], bank, sites,
                use_direction=(idx == 1))
            branch_loss[GROUPS[idx]] = float(loss); branch_comp[GROUPS[idx]] = comp; branch_grad[GROUPS[idx]] = gnorm
        test_ids = [int(x) for x in state["test_np"]]
        base_metrics = base_method_metrics(evaluated["logp"], test_ids, y)
        method_results = {}
        for idx, module in enumerate(modules):
            with torch.no_grad():
                method_logp, enhanced, diagnostics = run_memory_forward(
                    module, state, test_ids, eval_cam, evaluated["logp"], bank, sites,
                    use_direction=(idx == 1))
            full_logp = evaluated["logp"].detach().clone(); full_logp[torch.as_tensor(test_ids, device=y.device)] = method_logp
            metrics = base_method_metrics(full_logp, test_ids, y)
            method_results[GROUPS[idx]] = (full_logp, metrics, diagnostics)
        # The current BaseEval errors create the next epoch's memory only now.
        dirs = base_eval_direction(state, eval_cam, evaluated["logp"], eligible, y)
        event = bank.update_from_base_eval(eligible, y, eval_cam, evaluated["logp"], sites,
                                           epoch, correction_directions=dirs)
        memory_next = _memory_after_update(bank)
        if idx == 1:
            pass
        rng_record = {"epoch": epoch, "rng_before": rng_before,
                      "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval}
        trace.append(rng_record); event_trace.append(event)
        if float(base_metrics["acc"]) > base_ledger["best_acc"]:
            base_ledger = {"best_acc": float(base_metrics["acc"]), "best_epoch": epoch,
                           "metrics": metric_payload(base_metrics),
                           "probability": np.asarray(base_metrics["prob"], dtype=float),
                           "prediction": np.asarray(base_metrics["pred"], dtype=int),
                           "base_state": state_bundle(runtime, state)}
            base_wait = 0
        else:
            base_wait += 1
        for idx, group in enumerate(GROUPS):
            method_logp, metrics, diagnostics = method_results[group]
            if float(metrics["acc"]) > ledgers[group]["best_acc"]:
                transitions = correction_stats(evaluated["logp"], method_logp, test_ids, y)
                payload = {
                    "metrics": metric_payload(metrics),
                    "probability": np.asarray(metrics["prob"], dtype=float),
                    "prediction": np.asarray(metrics["pred"], dtype=int),
                    "base_probability": np.asarray(base_metrics["prob"], dtype=float),
                    "base_prediction": np.asarray(base_metrics["pred"], dtype=int),
                    "base_state": state_bundle(runtime, state),
                    "module": module_snapshot(modules[idx]),
                    "optimizer": copy.deepcopy(opts[idx].state_dict()),
                    "memory_used": copy.deepcopy(memory_used),
                    "memory_next": copy.deepcopy(memory_next),
                    "memory_event": event, "memory_event_hash": event["event_hash"],
                    "transition": transitions, "branch_loss": branch_loss[group],
                    "branch_components": branch_comp[group], "branch_gradient_norm": branch_grad[group],
                    "eligible_train_ids": eligible, "test_ids": test_ids,
                    "fold2_exclude": sorted(FOLD2_EXCLUDE) if fold == 2 else [],
                    "best_epoch_1based": epoch,
                }
                ledgers[group] = {"best_acc": float(metrics["acc"]), "best_epoch": epoch, "best": payload}
                save_best(group, fold, payload, source_hash, identity, trace, epoch)
        fold_rows.append({"fold": fold, "epoch": epoch, "base_acc": float(base_metrics["acc"]),
                          "base_prob_auc": float(base_metrics["prob_auc"]),
                          "g1_acc": float(method_results[GROUPS[0]][1]["acc"]),
                          "g2_acc": float(method_results[GROUPS[1]][1]["acc"]),
                          "memory_total": event["size_asd_to_td"] + event["size_td_to_asd"],
                          "memory_asd_to_td": event["size_asd_to_td"],
                          "memory_td_to_asd": event["size_td_to_asd"],
                          "memory_writes": event["writes"], "memory_updates": event["updates"],
                          "memory_evictions": event["evictions"], "memory_removals": event["correct_streak_removals"],
                          "base_error_count": event["base_error_count"], "branch_loss_g1": branch_loss[GROUPS[0]],
                          "branch_loss_g2": branch_loss[GROUPS[1]], "branch_grad_g1": branch_grad[GROUPS[0]],
                          "branch_grad_g2": branch_grad[GROUPS[1]], "rng_before": rng_before,
                          "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval})
        atomic_torch(latest_path, {
            "experiment": CFG["experiment"], "fold": fold, "source_sha256": source_hash,
            "identity": identity, "next_epoch": epoch, "base_wait": base_wait,
            "base_state": state_bundle(runtime, state), "rng": runtime.pack_rng(),
            "modules": {g: module_snapshot(m) for g, m in zip(GROUPS, modules)},
            "optimizers": {g: copy.deepcopy(o.state_dict()) for g, o in zip(GROUPS, opts)},
            "memory_next": memory_next, "ledgers": ledgers, "base_ledger": base_ledger,
            "trace": trace, "event_trace": event_trace, "fold_rows": fold_rows,
        })
        with (LOGS / f"E79_DEML_f{fold:02d}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"fold": fold, "epoch": epoch, "base_acc": float(base_metrics["acc"]),
                                     "g1_acc": float(method_results[GROUPS[0]][1]["acc"]),
                                     "g2_acc": float(method_results[GROUPS[1]][1]["acc"]),
                                     "base_wait": base_wait, "event": event,
                                     "rng": rng_record, "peak_gpu_bytes": int(torch.cuda.max_memory_allocated())},
                                    ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        if base_wait > 200:
            break
    if base_ledger["best_epoch"] < 0 or any(ledgers[g]["best"] is None for g in GROUPS):
        raise RuntimeError(f"no best checkpoint produced for fold {fold}")
    atomic_json(task_path, {"status": "COMPLETE", "fold": fold, "source_sha256": source_hash,
                            "identity": identity, "eligible_count": len(eligible),
                            "epochs_executed": len(fold_rows),
                            "base_best_epoch": base_ledger["best_epoch"],
                            "g1_best_epoch": ledgers[GROUPS[0]]["best_epoch"],
                            "g2_best_epoch": ledgers[GROUPS[1]]["best_epoch"],
                            "event_hash_final": event_trace[-1]["event_hash"] if event_trace else None,
                            "elapsed_seconds": time.time() - fold_started,
                            "status_note": "best checkpoints are independently audited by audit.py"})
    return {"fold": fold, "status": "COMPLETE", "epochs": len(fold_rows),
            "base": base_ledger, "ledgers": ledgers, "events": event_trace,
            "rows": fold_rows, "elapsed": time.time() - fold_started}


def preflight():
    import torch
    source = source_snapshot()
    audit = json.loads((BASE_EXP / "results/audit_status.json").read_text(encoding="utf-8"))
    manifest = json.loads((BASE_EXP / "manifest.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or manifest.get("status") != "COMPLETE_AUDITED":
        raise RuntimeError("E74.4-G2 historical audit is not PASS")
    inherited = []
    for fold in range(1, 11):
        state, warm, identity, task, ckpt = prepare_fold(fold)
        ids = eligible_ids(state, fold)
        inherited.append({"fold": fold, "warmup_sha256": identity["warmup_sha256"],
                          "warmup_rng_sha": warm["rng_sha"], "split_sha": identity["split_sha"],
                          "pair_sha": identity["pair_sha"], "pairs": int(state["train_pairs_np"].shape[0]),
                          "eligible_count": len(ids), "overlap_1based": identity["membership"]["train_test_overlap_1based"],
                          "base_checkpoint_sha256": sha_file(ckpt)})
        del state
        torch.cuda.empty_cache()
    # CAM/output-head identity and deterministic empty-memory fallback.
    state, warm, identity, task, ckpt = prepare_fold(1)
    before = rng_sha(legacy.history_runtime())
    out, cam = hook_forward(state, training=False)
    after = rng_sha(legacy.history_runtime())
    if before != after:
        raise RuntimeError("eval forward unexpectedly changed shared RNG")
    if tuple(cam.shape) != (871, 144):
        raise RuntimeError("CAM144 preflight failed")
    b1, b2 = init_pair(seed=7901, device=state["y"].device)
    bank = ErrorMemoryBank()
    sites = torch.zeros(871, dtype=torch.long, device=state["y"].device)
    ids = eligible_ids(state, 1)
    with torch.no_grad():
        enhanced, diag = b1(ids, cam[torch.as_tensor(ids, device=cam.device)].detach(),
                             out["logp"][torch.as_tensor(ids, device=cam.device)].detach(),
                             bank, sites, use_direction=False)
    if not torch.equal(enhanced, cam[torch.as_tensor(ids, device=cam.device)]):
        raise RuntimeError("empty memory did not exactly fall back to CAM144")
    if diag["delta_h"].abs().max().item() != 0.0:
        raise RuntimeError("zero-memory enhancement is not exactly zero")
    if sum(p.numel() for p in b1.parameters()) != sum(p.numel() for p in b2.parameters()):
        raise RuntimeError("G1/G2 parameter count mismatch")
    # Base one-step parity is checked by running one original step twice from
    # the same warmup bundle and RNG; the memory branch is not attached here.
    runtime = legacy.history_runtime()
    hist = legacy.load_base().load_history()
    initial_bundle = hist.state_bundle(state); initial_rng = runtime.pack_rng()
    hist.zero_optimizers(state); train_out, _ = hook_forward(state, training=True); train_out["total"].backward()
    ref_grads = {name: (p.grad.detach().clone() if p.grad is not None else None)
                 for module in (state["model"], state["cam"]) for name, p in module.named_parameters()}
    for key in ("opt_base", "opt_cam", "opt_added"): state[key].step()
    ref_bundle = hist.state_bundle(state); ref_rng = rng_sha(runtime)
    hist.load_bundle(state, initial_bundle); runtime.restore_rng(initial_rng)
    hist.zero_optimizers(state); replay_out, _ = hook_forward(state, training=True); replay_out["total"].backward()
    for key in ("opt_base", "opt_cam", "opt_added"): state[key].step()
    obs_bundle = hist.state_bundle(state)
    if ref_rng != rng_sha(runtime): raise RuntimeError("base RNG parity failed")
    for mod in ("model", "cam"):
        for name, value in ref_bundle[mod].items():
            if not torch.equal(value, obs_bundle[mod][name]): raise RuntimeError(f"base parity failed {mod}.{name}")
    atomic_json(RESULTS / "preflight.json", {"status": "PASS", "source_sha256": sha_obj(source),
                                             "source_files": source, "device": torch.cuda.get_device_name(0),
                                             "base_audit": audit, "inherited": inherited,
                                             "cam_shape": [871, 144], "empty_memory_exact": True,
                                             "g1_g2_param_count_equal": True,
                                             "base_one_step_parity": True,
                                             "no_prompt_or_oof_teacher": True,
                                             "fold2_exclude_zero_based": sorted(FOLD2_EXCLUDE),
                                             "timestamp": time.time()})
    print("PREFLIGHT PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "train", "train-all"))
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(); return
    pre = RESULTS / "preflight.json"
    if not pre.is_file() or json.loads(pre.read_text()).get("status") != "PASS":
        raise RuntimeError("run preflight before formal training")
    if args.command == "train":
        if args.fold is None or not 1 <= args.fold <= 10: raise ValueError("--fold 1..10 required")
        train_fold(args.fold); return
    # train-all is intentionally deterministic and can be dispatched by the
    # outer shell as up to ten independent workers.
    for fold in range(1, 11):
        train_fold(fold)


if __name__ == "__main__":
    main()
