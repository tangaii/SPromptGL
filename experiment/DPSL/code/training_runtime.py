"""Isolated full training runs and bounded verification of the training entry."""
from __future__ import annotations

import fcntl
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from dpsl_runtime import FOLDS, ROOT, read_json, sha256_file, verify_sha, write_json
from real_model_runtime import (check_budget, fold_engine, history_record, optimizer_identity, require, restore,
                                setup, validate_evaluation)


def named_run(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name):
        raise ValueError("Run name must be a single directory name")
    path = ROOT / "runs" / name
    if path.resolve() != path:
        raise ValueError("Run directory may not traverse a symlink")
    return path


def files_for(directory, fold):
    stem = f"fold{fold:02d}"
    files = {
        "best": directory / "checkpoints" / f"{stem}.best.pt",
        "latest": directory / "checkpoints" / f"{stem}.latest.pt",
        "journal": directory / "journal" / f"{stem}.jsonl",
        "task": directory / "journal" / f"{stem}.task.json",
    }
    for path in files.values():
        require(path.resolve() == path, "Run files must remain within the named run")
    return files


def run_fold(directory, fold, stop_at=None):
    files = files_for(directory, fold)
    engine = fold_engine(fold)
    # Newly generated checkpoints bind the archived adapter and the executable
    # package. Historical checkpoints and preflight files are read-only.
    hashes = engine.h.implementation()
    code_lock = ROOT / "runtime" / "operational_sha256.json"
    if code_lock.is_file():
        hashes = {**hashes, "DPSL.runtime": sha256_file(code_lock)}
    original_implementation = engine.h.implementation
    bind = hashes if code_lock.is_file() else None
    if bind is not None:
        engine.h.implementation = lambda: bind
    try:
        preflight = {**history_record(fold, "preflight"), "implementation": hashes}
        registration = {"implementation": hashes, "preflight": {"status": preflight["status"]}}
        files["best"].parent.mkdir(parents=True, exist_ok=True)
        with redirect_stdout(sys.stderr):
            result = engine.train_fold("G2", fold, files=files, manifest=registration, preflight=preflight,
                                       stop_at=stop_at)
        return engine, files, result or read_json(files["task"])
    finally:
        engine.h.implementation = original_implementation


def train(name, folds=FOLDS, *, resume=False, from_formal_latest=False):
    directory = named_run(name)
    if from_formal_latest:
        # All archived Latest files are terminal under this exact configuration.
        # The separate resume-check command verifies the recorded one-step witness.
        results = []
        for fold in folds:
            _, engine, saved, _ = restore(fold, "latest")
            require(saved["epoch"] == engine.CFG["max_epochs"] or saved["wait"] > engine.CFG["early_stop_wait_gt"],
                    f"Fold {fold} formal Latest unexpectedly has a remaining legal epoch")
            results.append({"fold": fold, "latest_epoch": saved["epoch"], "best_epoch": saved["best_epoch"],
                            "remaining_epochs": 0})
        return {"status": "PASS", "mode": "formal Latest already at configured stopping condition", "folds": results}
    if resume and not directory.is_dir():
        raise FileNotFoundError(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".lock"
    require(lock_path.resolve() == lock_path, "Run lock may not be a symlink")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for fold in folds:
            files = files_for(directory, fold)
            if resume and not files["latest"].is_file() and (
                    len(folds) == 1 or files["best"].exists() or files["task"].exists()):
                raise FileNotFoundError(files["latest"])
            if not resume and (files["latest"].exists() or files["best"].exists()):
                raise ValueError(f"Fold {fold} already exists in run {name}; use train-resume")
        summaries = [run_fold(directory, fold)[2] for fold in folds]
    return {"status": "TRAINED", "run": str(directory.relative_to(ROOT)), "folds": summaries}


def warmup_check(folds=FOLDS):
    """Exercise the same warmup, durable save, and actual continuation code as train."""
    checks = []
    with TemporaryDirectory(prefix="warmup-audit-", dir=ROOT / "runtime") as scratch:
        for fold in folds:
            directory = Path(scratch) / f"fold{fold:02d}"
            preflight = history_record(fold, "preflight")
            engine, files, first = run_fold(directory, fold, stop_at=11)
            saved = engine.h.load(files["latest"])
            best = engine.h.load(files["best"])
            require(saved["epoch"] == best["epoch"] == first["epoch"] == 11
                    and saved["best_sha"] == sha256_file(files["best"]), "Fresh Best and Latest did not commit")
            ctx = setup(fold, engine)
            engine.restore(ctx, saved)
            validate_evaluation(saved["evaluation"], len(ctx["members"]["test"]), ctx["state"]["labels_np"][ctx["members"]["test"]])
            opts = optimizer_identity(ctx, engine)
            require(engine.h.sha(engine.bundle(ctx)) == saved["state_sha"],
                    "Freshly saved warmup optimizer or model not restorable")
            source_journal = ROOT / "runtime/history" / f"fold{fold:02d}.journal.jsonl"
            with source_journal.open("rb") as source:
                first_historical, second_historical = source.readline(), source.readline()
            require(files["journal"].read_bytes() == first_historical,
                    f"Fold {fold} first persisted epoch differs from historical journal")
            rng = json.loads(first_historical)["rng"]
            require(rng["rng_before"] == ctx["warmup"]["rng_sha"] and len(preflight["dropout"]) == 6,
                    "Warmup RNG or six Dropout calls differ")
            # Verify the next transition after the real disk restore, then run it
            # through the durable continuation entry and compare the state.
            evaluation, diag, rng_next, _, _ = engine.transition(ctx, 12, preflight["dropout"])
            expected_state = engine.h.sha(engine.bundle(ctx))
            budget = check_budget(engine, diag)
            del ctx, diag
            _, _, second = run_fold(directory, fold, stop_at=12)
            resumed = engine.h.load(files["latest"])
            require(second["epoch"] == resumed["epoch"] == 12 and resumed["state_sha"] == expected_state
                    and engine.h.sha(resumed["evaluation"]) == engine.h.sha(evaluation), "Warmup durable resume diverged")
            lines = files["journal"].read_bytes().splitlines(keepends=True)
            require(len(lines) == 2 and lines[0] == first_historical and
                    verify_sha(files["journal"], resumed["journal_sha"]) == sha256_file(files["journal"]),
                    "Warmup continuation journal is inconsistent")
            actual_second = json.loads(lines[1])
            source_second = json.loads(second_historical)
            require(actual_second == source_second and actual_second["rng"] == rng_next,
                    f"Fold {fold} resumed epoch-12 record differs from history")
            checks.append({
                "fold": fold, "status": "PASS", "inherited_epoch": 10, "warmup_epoch": 11, "resumed_epoch": 12,
                "historical_first_step_sha": preflight["first_step_sha"], "historical_journal_rows_exact": 2,
                "initial_rng_sha": rng["rng_before"], "epoch_11_rng": rng, "epoch_12_rng": rng_next,
                "dropout_calls_in_order": preflight["dropout"], "epoch_11_optimizers_after_reload": opts,
                "epoch_11_state_sha": saved["state_sha"], "epoch_12_state_sha": expected_state,
                "epoch_12_shared_gradient_fraction": budget, "epoch_11_eval_correct": saved["evaluation"]["correct"],
                "epoch_12_eval_correct": resumed["evaluation"]["correct"],
            })
            print(f"Warmup fold {fold:02d} epochs 11/12 durable save/restore and historic journal PASS",
                  file=sys.stderr, flush=True)
    return {"status": "PASS", "initial_epoch": 10, "actual_test_epochs": [11, 12], "folds": checks,
            "temporary_run_files_deleted": True, "full_historical_training_reexecuted": False}
