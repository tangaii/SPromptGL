#!/usr/bin/env python3
"""Run one strict SPromptGL Experiment 5 configuration and seed."""

import argparse
import os
import time

import numpy as np
import torch

from common import (
    load_abide,
    make_hyperparameters,
    make_strict_split,
    module_fingerprint,
    set_rng_seed,
    split_fingerprint,
)
from model import EvalHelper
from network import get_prompt_text_features
from runner_common import fold_row, oof_rows, write_csv


CONFIGS = {
    "S0_STRICT_BASELINE": ("none", 0.0),
    "S1_STRICT_UNIFORM_GLOBAL": ("hybrid", 0.50),
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
    args.protocol = "strict"

    input_data, labels, original_labels, input_dims, modal_dict = load_abide(args.datadir)
    print("STRICT_PROTOCOL=True", flush=True)
    print("CONFIG={}".format(args.config), flush=True)
    print("SEED={}".format(args.seed), flush=True)
    print("DATA_SHAPE={}".format((input_data.shape[0], input_data.shape[1] + 1)), flush=True)
    print("MODAL_DIMS={}".format(input_dims), flush=True)
    print("INTERNAL_LABEL_COUNTS={}".format(np.bincount(labels, minlength=2).tolist()), flush=True)
    print("SPLIT=outer StratifiedKFold(seed)+inner StratifiedShuffleSplit(seed+1000+fold)", flush=True)
    print("CHECKPOINT_SELECTION=inner_validation_acc", flush=True)
    print("NEPOCH={} EARLY={}".format(args.nepoch, args.early), flush=True)
    print("ALIGN_MODE={} ALIGN_LAMBDA={} ALIGN_TEMPERATURE={} ALIGN_WARMUP={}".format(
        args.align_mode, args.align_lambda, args.align_temperature, args.align_warmup
    ), flush=True)

    # Strict protocol preloads CLIP before fold-specific RNG is set.
    print("CLIP_PROMPT_CACHE=loading", flush=True)
    get_prompt_text_features()
    print("CLIP_PROMPT_CACHE=ready", flush=True)

    fold_rows, all_oof = [], []
    fold_ids = [args.single_fold] if args.single_fold is not None else list(range(10))
    for fold_id in fold_ids:
        if fold_id < 0 or fold_id >= 10:
            raise ValueError("single_fold must be 0..9")
        train_index, val_index, test_index, outer_train = make_strict_split(
            input_data, labels, args.seed, fold_id
        )
        if set(train_index) & set(val_index) or set(train_index) & set(test_index) or set(val_index) & set(test_index):
            raise RuntimeError("Strict split overlap in fold {}".format(fold_id))
        split_sha = split_fingerprint(train_index, val_index, test_index)
        fold_seed = args.seed * 100000 + fold_id * 100 + 17
        train_seed = args.seed * 100000 + fold_id * 100 + 73
        set_rng_seed(fold_seed)
        hyperpm = make_hyperparameters(args)
        agent = EvalHelper(
            input_dims, input_data, labels, hyperpm,
            train_index, val_index, test_index,
        )
        agent.set_base_fingerprint(module_fingerprint)
        set_rng_seed(train_seed)
        print("=== FOLD {}/10 train={} val={} test={} split_sha={} init_sha={} ===".format(
            fold_id + 1, len(train_index), len(val_index), len(test_index),
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
                print("epoch={} val_acc={:.6f} best={:.6f}".format(
                    epoch + 1, val_acc, best_val_acc
                ), flush=True)
        if best_states is None:
            raise RuntimeError("No strict checkpoint for fold {}".format(fold_id + 1))
        agent.load_state_dicts(best_states)
        result = agent.selected_test_result()
        elapsed = time.perf_counter() - started
        row = fold_row(
            "strict", args.config, args.seed, fold_id + 1,
            outer_train, train_index, val_index, test_index,
            best_epoch, best_val_acc, result, "inner_val_acc",
            agent.base_init_sha256, split_sha, epochs_run, elapsed,
        )
        fold_rows.append(row)
        all_oof.extend(oof_rows(
            "strict", args.config, args.seed, fold_id + 1, test_index,
            original_labels, result, best_epoch, "inner_val_acc",
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
    print("WROTE_FOLDS={} WROTE_OOF={}".format(len(fold_rows), len(all_oof)), flush=True)


if __name__ == "__main__":
    run(parser().parse_args())

