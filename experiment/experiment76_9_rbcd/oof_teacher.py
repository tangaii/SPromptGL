#!/usr/bin/env python3
"""Isolated inner-5-fold OOF teachers for E76.9 RBCD.

Each outer fold is handled by an independent process.  The teacher starts from the
raw fold anchor/DARG construction and never loads the E61 epoch-10 warmup.  Only
inner-training labels enter classification, relation-pair, adversarial, pair,
prototype, weighting, or selection operations.  The inner holdout is used once,
at the final fixed epoch, only for its prediction.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
REPO = Path("/data/adodas/SPromptGL-main")
RT = EXP / ".runtime"
OOF = RT / "oof"
LOGS = RT / "logs"
BASE_GROUP = "E74.4-G2-Positive-DCR"
TEACHER_EPOCHS = 250


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_obj(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def tensor_state_sha(module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode() + b"\0")
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_torch(path: Path, payload):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as f:
        torch.save(payload, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    check = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(check, dict):
        raise IOError(f"OOF readback failed: {path}")


def load_context():
    spec_path = EXP / "path_adapter.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("e769_path_adapter_for_oof", spec_path)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    legacy = adapter.legacy
    base74 = legacy.load_base()
    history = base74.load_history()
    e69 = history.load_e69()
    e61 = e69.load_e61()
    base_runtime = e61.load_base()
    return adapter, legacy, base74, history, e69, e61, base_runtime


def raw_teacher_state(e61, outer_fold: int, inner_train, labels_np, pair_pairs, pair_same):
    """Build the raw fold anchor/DARG/CAM state without any outer-label pair operation.

    This mirrors E61 ``build_state`` only for feature/anchor/model construction.  The
    only label-dependent objects created here are the inner-train class weights and
    the supplied inner-train relation pairs.  Outer-test/inner-holdout labels remain
    in the immutable data vector solely for later audit indexing and are never read by
    the teacher forward or loss.
    """
    import math
    import numpy as np
    import torch
    from cadr_gfr import CADRGFR

    r = e61.load_base()
    (x_np, y_np), folds = r.load_data()
    train_np, test_np = folds[int(outer_fold) - 1]
    supports, fallback = r.anchor_candidate_graphs(int(outer_fold))
    core = r.build_legacy_core(int(outer_fold))
    # The constructor and raw anchor hashes are identical to E61 build_state; no
    # E61 epoch-10 parameters or optimizer state are loaded.
    rng_before = r.pack_rng()
    zeros = torch.zeros(4, dtype=torch.float32)
    calibration = torch.tensor([[1.0, 0.0]] * 4, dtype=torch.float32)
    model = CADRGFR(
        core["base"], zeros, calibration,
        torch.as_tensor(fallback, dtype=torch.bool), init_seed=532000 + int(outer_fold),
        grl_scale=0.03, edge_topk=12, positive_threshold=0.55,
        negative_threshold=0.45, negative_scale=0.25, direct_scale=0.02,
    ).to(r.runtime()["device"])
    if r.rng_sha(rng_before) != r.rng_sha(r.pack_rng()):
        raise RuntimeError("raw OOF CADR constructor changed shared RNG")
    model.adaptive_gate_logits = torch.nn.Parameter(
        torch.full((4,), math.log(0.1 / 0.9), device=r.runtime()["device"])
    )
    base_params, added_named, added_groups = r._optimizer_groups(model)
    for group in added_groups:
        group["weight_decay"] = 0.01
    opt_base = torch.optim.Adam(base_params, lr=0.0038, weight_decay=0.11)
    opt_cam = torch.optim.Adam(core["cam"].parameters(), lr=0.0038, weight_decay=0.11)
    opt_added = torch.optim.AdamW(added_groups, lr=0.001)
    inner = np.asarray(sorted(int(x) for x in inner_train), dtype=np.int64)
    inner_labels = np.asarray(y_np)[inner]
    counts = np.bincount(inner_labels, minlength=2).astype(np.float32)
    if (counts <= 0).any():
        raise RuntimeError(f"inner split has an empty class outer={outer_fold}")
    weights = torch.tensor(len(inner) / counts / 2.0, dtype=torch.float32, device=r.runtime()["device"])
    offsets = np.cumsum([0] + list(r.DIMS[:-1]))
    parts = [core["x"][:, start:start + dim] for start, dim in zip(offsets, r.DIMS)]
    base_task = json.loads((e61.BASE_RUNTIME / "tasks" / f"{e61.BASE_GROUP}_f{int(outer_fold):02d}.json").read_text(encoding="utf-8"))
    state = {
        **core, "group": None, "fold": int(outer_fold), "model": model, "cam": core["cam"],
        "opt_base": opt_base, "opt_cam": opt_cam, "opt_added": opt_added,
        "base_params": base_params, "added_named": added_named, "parts": parts,
        "weights": weights,
        "train_pairs": torch.as_tensor(pair_pairs, dtype=torch.long, device=r.runtime()["device"]),
        "train_same": torch.as_tensor(pair_same, dtype=torch.float32, device=r.runtime()["device"]),
        "train_pairs_np": np.asarray(pair_pairs, dtype=np.int64),
        "train_same_np": np.asarray(pair_same, dtype=np.int64),
        "supports": supports, "fallback": fallback,
        "initial_shared_sha": r.digest_shared(core["base"], core["cam"]),
        "initial_gfr_sha": r.module_sha(core["base"]),
        "initial_cam_sha": r.module_sha(core["cam"]),
        "relation_core_sha": e61.sha_named(model, ("encoders.", "scorers.", "discriminator.", "edge_gate_logits")),
        "initial_rng_sha": core["construction_rng_sha"],
        "initial_added_sha": r.digest_added_parameters(model),
        "pair_sha": e61.pair_sha(np.asarray(pair_pairs), np.asarray(pair_same)),
        "base_task": base_task,
        "train_np": inner,
        "test_np": np.asarray(test_np, dtype=np.int64),
        "labels_np": np.asarray(y_np, dtype=np.int64),
        "warmup_best_acc": None, "warmup_best_epoch": None,
        "rng_trace": [], "oof": None, "branch": None,
    }
    # These are feature/anchor identities only.  The historical outer pair SHA is
    # deliberately not compared because the teacher has its own inner-only pairs.
    for field in ("initial_shared_sha", "initial_gfr_sha", "initial_cam_sha",
                  "initial_rng_sha", "split_sha", "anchor_sha"):
        if state[field] != base_task[field]:
            raise RuntimeError(f"raw OOF anchor identity mismatch outer={outer_fold} field={field}")
    state["outer_train_np"] = np.asarray(train_np, dtype=np.int64)
    return state


def build_inner_pairs(base_runtime, supports, inner_train, labels_np, seed: int):
    import numpy as np
    union = np.concatenate(supports, axis=0) if any(len(x) for x in supports) else np.zeros((0, 2), dtype=np.int64)
    if len(union):
        union = np.unique(np.sort(union, axis=1), axis=0)
    return base_runtime.sample_balanced_pairs(
        np.asarray(inner_train, dtype=np.int64), labels_np, 4096,
        np.random.default_rng(int(seed)), union,
    )


def teacher_for_inner(outer_fold: int, inner_id: int, inner_train, inner_holdout,
                      labels_np, supports, e61, base74, base_runtime):
    import numpy as np
    import torch
    # All pair sampling and teacher randomness are local to this teacher.  The
    # formal E76.9 main RNG is in a different process and is never touched.
    pair_seed = 910000 + outer_fold * 100 + inner_id
    pairs, same = build_inner_pairs(base_runtime, supports, inner_train, labels_np, pair_seed)
    state = raw_teacher_state(e61, outer_fold, inner_train, labels_np, pairs, same)
    model_initial_sha = tensor_state_sha(state["model"])
    cam_initial_sha = tensor_state_sha(state["cam"])
    teacher_seed = 920000 + outer_fold * 100 + inner_id
    torch.manual_seed(teacher_seed)
    torch.cuda.manual_seed_all(teacher_seed)
    np.random.seed(teacher_seed % (2**32 - 1))
    started = time.time()
    loss_trace = []
    for epoch in range(TEACHER_EPOCHS):
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].zero_grad(set_to_none=True)
        out = base74.forward(state, BASE_GROUP, training=True)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"nonfinite OOF teacher loss outer={outer_fold} inner={inner_id} epoch={epoch+1}")
        out["total"].backward()
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        if epoch in (0, 49, 124, 249):
            loss_trace.append({"epoch": epoch + 1, "loss": float(out["total"].detach().cpu())})
        del out
    with torch.no_grad():
        evaluated = base74.forward(state, BASE_GROUP, training=False)
        probs = evaluated["logp"].exp().detach().cpu().numpy()
    holdout = np.asarray(sorted(int(x) for x in inner_holdout), dtype=np.int64)
    p_hold = probs[holdout]
    y_hold = labels_np[holdout]
    if len(set(inner_train).intersection(set(inner_holdout))):
        raise RuntimeError("inner teacher train/holdout overlap")
    record = {
        "outer_fold": int(outer_fold), "inner_fold": int(inner_id),
        "teacher_seed": int(teacher_seed), "pair_seed": int(pair_seed),
        "teacher_epochs": TEACHER_EPOCHS, "selection": "fixed_final_epoch",
        "train_ids": np.asarray(sorted(int(x) for x in inner_train), dtype=np.int64),
        "holdout_ids": holdout, "pair_sha256": sha_obj({"pairs": pairs.tolist(), "same": same.tolist()}),
        "model_initial_sha256": model_initial_sha, "cam_initial_sha256": cam_initial_sha,
        "model_final_sha256": tensor_state_sha(state["model"]),
        "cam_final_sha256": tensor_state_sha(state["cam"]),
        "holdout_probability": p_hold.astype(np.float64),
        "holdout_label": y_hold.astype(np.int64),
        "loss_trace": loss_trace, "wall_seconds": time.time() - started,
        "supervision_scope": {"classification": "inner_train_only", "graph_pair": "inner_train_only",
                              "adversary_pair": "inner_train_only", "prototype_label_support": "none",
                              "checkpoint_selection": "none", "outer_test_labels": False},
    }
    del state
    torch.cuda.empty_cache()
    return record


def fold_path(fold: int) -> Path:
    return OOF / f"fold{fold:02d}.pt"


def run_fold(fold: int):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    adapter, legacy, base74, history, e69, e61, base_runtime = load_context()
    # Read only the fixed outer indices and feature-derived anchor candidates.
    # Do not call E61 build_state here: that helper samples an outer-label relation
    # pair set before it can be narrowed to the inner split.
    base_runtime = e61.load_base()
    (_, labels_np), folds = base_runtime.load_data()
    outer_train, outer_test = folds[int(fold) - 1]
    outer_train = np.asarray(outer_train, dtype=np.int64)
    outer_test = np.asarray(outer_test, dtype=np.int64)
    eligible = np.asarray(sorted(int(i) for i in outer_train if int(i) not in set(outer_test)), dtype=np.int64)
    if fold == 2:
        eligible = np.asarray([x for x in eligible if int(x) not in {376, 406}], dtype=np.int64)
    if np.intersect1d(eligible, outer_test).size:
        raise RuntimeError(f"OOF eligible/test overlap fold={fold}")
    if fold == 2 and ({376, 406} & set(eligible.tolist())):
        raise RuntimeError("Fold 2 duplicate rows entered OOF eligible set")
    supports, _ = base_runtime.anchor_candidate_graphs(int(fold))
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000 + int(fold))
    y_eligible = labels_np[eligible]
    oof_probability = np.zeros((len(eligible), 2), dtype=np.float64)
    seen = np.zeros(len(eligible), dtype=bool)
    inner_records = []
    started = time.time()
    for inner_id, (train_pos, hold_pos) in enumerate(split.split(eligible, y_eligible), start=1):
        train_ids = eligible[train_pos]
        holdout_ids = eligible[hold_pos]
        if set(train_ids.tolist()) & set(holdout_ids.tolist()):
            raise RuntimeError("inner split overlap")
        record = teacher_for_inner(fold, inner_id, train_ids, holdout_ids,
                                   labels_np, supports, e61, base74, base_runtime)
        positions = {int(value): idx for idx, value in enumerate(eligible.tolist())}
        for patient, probability in zip(record["holdout_ids"].tolist(), record["holdout_probability"]):
            pos = positions[int(patient)]
            if seen[pos]:
                raise RuntimeError(f"duplicate OOF prediction for patient {patient}")
            oof_probability[pos] = probability
            seen[pos] = True
        inner_records.append(record)
    if not bool(seen.all()):
        raise RuntimeError(f"OOF coverage incomplete fold={fold}: {int(seen.sum())}/{len(seen)}")
    eligible_y = labels_np[eligible]
    risk_soft = 1.0 - oof_probability[np.arange(len(eligible)), eligible_y]
    risk_hard = (oof_probability.argmax(axis=1) != eligible_y).astype(np.int64)
    payload = {
        "experiment": "Experiment 76.9 RBCD", "fold": int(fold), "status": "COMPLETE",
        "outer_train_ids": outer_train, "outer_test_ids": outer_test,
        "eligible_train_ids": eligible, "eligible_labels": eligible_y.astype(np.int64),
        "oof_probability": oof_probability, "risk_soft": risk_soft.astype(np.float64),
        "risk_hard": risk_hard, "oof_error_rate": float(risk_hard.mean()),
        "inner_split": {"n_splits": 5, "shuffle": True, "random_state": 1000 + int(fold)},
        "inner_records": inner_records,
        "source_sha256": sha_obj({"oof_teacher": sha_file(EXP / "oof_teacher.py"),
                                   "rbcd": sha_file(EXP / "rbcd.py"),
                                   "base74_run": sha_file(REPO / "experiment/experiment74_4_soft_positive_dcr/run.py")}),
        "teacher_initialization": "raw_shared_anchor_DARG_construction; E61_epoch10_warmup_not_loaded",
        "test_labels_accessed": False,
        "coverage_exactly_once": True,
        "fold2_overlap_excluded": bool(fold == 2),
        "wall_seconds": time.time() - started,
    }
    atomic_torch(fold_path(fold), payload)
    (LOGS / f"oof_fold{fold:02d}.json").write_text(json.dumps({
        "fold": fold, "status": "COMPLETE", "eligible": len(eligible),
        "oof_error_rate": float(risk_hard.mean()), "wall_seconds": payload["wall_seconds"],
        "sha256": sha_file(fold_path(fold)),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fold": fold, "eligible": len(eligible), "oof_error_rate": float(risk_hard.mean()),
                      "wall_seconds": payload["wall_seconds"]}), flush=True)
    return payload


def run_all(workers: int):
    OOF.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    tasks = [fold for fold in range(1, 11) if not fold_path(fold).is_file()]
    done = [fold for fold in range(1, 11) if fold_path(fold).is_file()]
    running = {}; failures = []
    while tasks or running:
        while tasks and len(running) < max(1, min(10, int(workers))):
            fold = tasks.pop(0)
            log = (LOGS / f"worker_oof_f{fold:02d}.log").open("ab")
            proc = subprocess.Popen([sys.executable, str(EXP / "oof_teacher.py"), "--fold", str(fold)],
                                    cwd=EXP, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1",
                                         "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            running[proc] = (fold, log)
            print(f"START OOF fold={fold} pid={proc.pid} running={len(running)}", flush=True)
        time.sleep(2)
        for proc, (fold, log) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close(); del running[proc]
            if code == 0 and fold_path(fold).is_file():
                done.append(fold); print(f"DONE OOF fold={fold} {len(done)}/10", flush=True)
            else:
                failures.append({"fold": fold, "exit": code, "log": str(LOGS / f"worker_oof_f{fold:02d}.log")})
        if failures:
            for proc, (fold, log) in list(running.items()):
                proc.terminate(); log.close()
            for proc in running: proc.wait()
            raise RuntimeError(f"OOF workers failed: {failures}")
    if len(set(done)) != 10:
        raise RuntimeError(f"OOF incomplete: {sorted(set(done))}")
    return {"folds": sorted(set(done)), "workers": max(1, min(10, int(workers)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.fold is not None:
        run_fold(args.fold)
    elif args.all:
        print(json.dumps(run_all(args.workers)), flush=True)
    else:
        parser.error("choose --fold or --all")


if __name__ == "__main__":
    main()
