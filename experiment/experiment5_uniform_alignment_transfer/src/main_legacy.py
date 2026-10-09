#!/usr/bin/env python3
"""Run one original SPromptGL legacy configuration and seed."""

import argparse
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from common import (
    load_abide,
    make_hyperparameters,
    module_fingerprint,
    set_rng_seed,
    split_fingerprint,
)
from model import EvalHelper
from runner_common import fold_row, oof_rows, write_csv


CONFIGS = {
    "L0_LEGACY_BASELINE": ("none", 0.0),
    "L1_LEGACY_UNIFORM_GLOBAL": ("hybrid", 0.50),
}


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", default="/data/adodas/SPromptGL-main/data")
    p.add_argument("--config", choices=sorted(CONFIGS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--nepoch", type=int, default=450)
    p.add_argument("--early", type=int, default=200)
    p.add_argument("--align_mode", choices=["none", "hybrid"], required=True)
    p.add_argument("--align_lambda", type=float, required=True)
    p.add_argument("--align_temperature", type=float, default=0.10)
    p.add_argument("--align_warmup", type=int, default=20)
    p.add_argument("--single_fold", type=int, default=None)
    p.add_argument("--fold_results", required=True)
    p.add_argument("--oof_results", required=True)
    return p


def run(args):
    expected_mode, expected_lambda = CONFIGS[args.config]
    if args.align_mode != expected_mode or abs(args.align_lambda - expected_lambda) > 1e-12:
        raise ValueError("Config/alignment mismatch for {}".format(args.config))
    args.protocol = "legacy"

    # Match the original train_and_eval ordering and preserve sequential RNG
    # across folds.  There is deliberately no fold-level reseeding.
    set_rng_seed(args.seed)
    input_data, labels, original_labels, input_dims, modal_dict = load_abide(args.datadir)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    splits = list(skf.split(input_data, labels))
    set_rng_seed(args.seed)

    print("LEGACY_PROTOCOL=True", flush=True)
    print("CONFIG={}".format(args.config), flush=True)
    print("SEED={}".format(args.seed), flush=True)
    print("DATA_SHAPE={}".format((input_data.shape[0], input_data.shape[1] + 1)), flush=True)
    print("MODAL_DIMS={}".format(input_dims), flush=True)
    print("INTERNAL_LABEL_COUNTS={}".format(np.bincount(labels, minlength=2).tolist()), flush=True)
    print("SPLIT=StratifiedKFold(n_splits=10, random_state=10, shuffle=True)", flush=True)
    print("VAL_EQUALS_TEST=True", flush=True)
    print("CHECKPOINT_SELECTION=test_fold_acc", flush=True)
    print("NEPOCH={} EARLY={} RNG_RESEED_PER_FOLD=False".format(args.nepoch, args.early), flush=True)
    print("ALIGN_MODE={} ALIGN_LAMBDA={} ALIGN_TEMPERATURE={} ALIGN_WARMUP={}".format(
        args.align_mode, args.align_lambda, args.align_temperature, args.align_warmup
    ), flush=True)

    fold_rows, all_oof = [], []
    fold_ids = [args.single_fold] if args.single_fold is not None else list(range(10))
    for fold_id in fold_ids:
        if fold_id < 0 or fold_id >= 10:
            raise ValueError("single_fold must be 0..9")
        train_index, test_index = splits[fold_id]
        val_index = np.asarray(test_index, dtype=np.int64)
        split_sha = split_fingerprint(train_index, val_index, test_index)
        hyperpm = make_hyperparameters(args)
        # No RNG reset here: this is the original sequential legacy trajectory.
        agent = EvalHelper(
            input_dims, input_data, labels, hyperpm,
            train_index, val_index, test_index,
        )
        agent.set_base_fingerprint(module_fingerprint)
        print("=== FOLD {}/10 train={} test={} split_sha={} init_sha={} ===".format(
            fold_id + 1, len(train_index), len(test_index),
            split_sha, agent.base_init_sha256,
        ), flush=True)

        best_val_acc = -float("inf")
        best_epoch = 0
        wait_count = 0
        best_states = None
        epochs_run = 0
        started = time.perf_counter()
        for epoch in range(args.nepoch):
            agent.run_epoch(epoch)
            # Legacy val_idx is the test fold by construction.
            val_metrics = agent.validation_metrics()
            val_acc = float(val_metrics["ACC"])
            epochs_run = epoch + 1
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                wait_count = 0
                best_states = agent.state_dicts()
            else:
                wait_count += 1
                if wait_count > args.early:
                    break
            if epoch == 0 or (epoch + 1) % 25 == 0:
                print("epoch={} test_fold_acc={:.6f} best={:.6f}".format(
                    epoch + 1, val_acc, best_val_acc
                ), flush=True)
        if best_states is None:
            raise RuntimeError("No legacy checkpoint for fold {}".format(fold_id + 1))
        agent.load_state_dicts(best_states)
        result = agent.selected_test_result()
        elapsed = time.perf_counter() - started
        row = fold_row(
            "legacy", args.config, args.seed, fold_id + 1,
            train_index, train_index, val_index, test_index,
            best_epoch, best_val_acc, result, "test_fold_acc",
            agent.base_init_sha256, split_sha, epochs_run, elapsed,
        )
        fold_rows.append(row)
        all_oof.extend(oof_rows(
            "legacy", args.config, args.seed, fold_id + 1, test_index,
            original_labels, result, best_epoch, "test_fold_acc",
        ))
        print("fold={} best_epoch={} ACC={:.5f} BA={:.5f} ASD-AUC={:.5f} runtime_sec={:.2f}".format(
            fold_id + 1, best_epoch, row["ACC"], row["BA"],
            row["probability_ASD_AUC"], elapsed,
        ), flush=True)
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(args.fold_results, fold_rows)
    write_csv(args.oof_results, all_oof)
    print("WROTE_FOLDS={} WROTE_OOF={}".format(len(fold_rows), len(all_oof)))


if __name__ == "__main__":
    run(parser().parse_args())

