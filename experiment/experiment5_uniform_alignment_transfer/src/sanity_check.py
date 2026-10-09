#!/usr/bin/env python3
"""Mandatory pre-run checks for Experiment 5."""

import argparse
import importlib.util
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

from alignment import compute_alignment_loss
from common import (
    load_abide,
    make_legacy_splits,
    make_strict_split,
    module_fingerprint,
    set_rng_seed,
    sha256_file,
)
from network import PLN


def load_original_pln(path):
    spec = importlib.util.spec_from_file_location("spromptgl_original_network", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PLN


def hyperparameters():
    return SimpleNamespace(
        n_hidden=18, n_head=2, dropout=0.35, nlayer=1, nmodal=4,
        nclass=2,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/adodas/SPromptGL-main")
    parser.add_argument("--datadir", default="/data/adodas/SPromptGL-main/data")
    args = parser.parse_args()
    exp = os.path.join(args.root, "experiment", "experiment5_uniform_alignment_transfer")
    input_data, labels, original_labels, dims, modal_dict = load_abide(args.datadir)
    print("DATA_SHAPE={}".format((input_data.shape[0], input_data.shape[1] + 1)))
    print("MODAL_DIMS={}".format(dims))
    print("LABEL_COUNTS_INTERNAL={}".format(np.bincount(labels, minlength=2).tolist()))

    root_csv = os.path.join(args.root, "data/ABIDE/processed_standard_data.csv")
    mmgl_csv = "/data/adodas/MMGL-main/experiment/experiment4_pair_adaptive_alignment/data/ABIDE/processed_standard_data.csv"
    root_npy = os.path.join(args.root, "data/ABIDE/modal_feat_dict.npy")
    mmgl_npy = "/data/adodas/MMGL-main/experiment/experiment4_pair_adaptive_alignment/data/ABIDE/modal_feat_dict.npy"
    csv_match = sha256_file(root_csv) == sha256_file(mmgl_csv)
    npy_match = sha256_file(root_npy) == sha256_file(mmgl_npy)
    print("SPROMPTGL_MMGL_DATA_CSV_MATCH={}".format("YES" if csv_match else "NO"))
    print("SPROMPTGL_MMGL_MODAL_DICT_MATCH={}".format("YES" if npy_match else "NO"))
    if not (csv_match and npy_match):
        raise RuntimeError("Cross-model data SHA mismatch")

    source_alignment = "/data/adodas/MMGL-main/experiment/experiment4_pair_adaptive_alignment/src/alignment.py"
    destination_alignment = os.path.join(exp, "src/alignment.py")
    print("ALIGNMENT_SOURCE_SHA256={}".format(sha256_file(source_alignment)))
    print("ALIGNMENT_DEST_SHA256={}".format(sha256_file(destination_alignment)))
    if sha256_file(source_alignment) != sha256_file(destination_alignment):
        raise RuntimeError("Alignment source/destination SHA mismatch")

    # The source directory is placed first so original network.py resolves its
    # copied layers.py/utils.py, which are byte-copied from the repository.
    original_network = os.path.join(args.root, "network.py")
    OriginalPLN = load_original_pln(original_network)
    hp = hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(input_data).float().to(device)
    set_rng_seed(2025)
    old = OriginalPLN(dims, hp).to(device).eval()
    old_state = old.state_dict()
    set_rng_seed(2025)
    new = PLN(dims, hp).to(device).eval()
    new.load_state_dict(old_state)
    with torch.no_grad():
        old_output = old(x)
        new_output = new(x)
        new_full = new(x, return_modal_tokens=True)
    p_diff = float((old_output[0] - new_output[0]).abs().max())
    h_diff = float((old_output[1] - new_output[1]).abs().max())
    pre, post = new_full[3], new_full[4]
    print("FORWARD_PARITY_PROB_MAX_ABS={:.12g}".format(p_diff))
    print("FORWARD_PARITY_HIDDEN_MAX_ABS={:.12g}".format(h_diff))
    print("PRE_SHAPE={}".format(tuple(pre.shape)))
    print("POST_SHAPE={}".format(tuple(post.shape)))
    print("PROMPT_DELTA_MEAN_ABS={:.12g}".format(float((post - pre).abs().mean())))
    if p_diff >= 1e-7 or h_diff >= 1e-7:
        raise RuntimeError("Forward parity failed")
    if tuple(pre.shape) != (871, 4, 36) or tuple(post.shape) != (871, 4, 36):
        raise RuntimeError("Token shape failed")
    if float((post - pre).abs().mean()) <= 1e-8:
        raise RuntimeError("Prompt residual is zero")

    # Finite hybrid alignment and backward check.
    token_test = torch.randn(24, 4, 36, device=device, requires_grad=True)
    label_test = torch.tensor([0, 1] * 12, device=device)
    align_loss, parts = compute_alignment_loss(token_test, label_test, mode="hybrid", temperature=0.10)
    align_loss.backward()
    print("ALIGNMENT_LOSS_FINITE={}".format(bool(torch.isfinite(align_loss))))
    print("ALIGNMENT_BACKWARD_FINITE={}".format(bool(torch.isfinite(token_test.grad).all())))
    if not torch.isfinite(align_loss) or not torch.isfinite(token_test.grad).all():
        raise RuntimeError("Alignment sanity check failed")

    print("WARMUP_EPOCH_0={}".format(0.50 * min(1.0, 1.0 / 20.0)))
    print("WARMUP_EPOCH_19={}".format(0.50 * min(1.0, 20.0 / 20.0)))
    print("WARMUP_EPOCH_20={}".format(0.50 * min(1.0, 21.0 / 20.0)))

    train, val, test, outer = make_strict_split(input_data, labels, 0, 0)
    print("STRICT_SEED0_FOLD0_SIZES={}".format((len(outer), len(train), len(val), len(test))))
    print("STRICT_DISJOINT={}".format(
        not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    ))
    splits = make_legacy_splits(input_data, labels, 0)
    lt, ltest = splits[0]
    print("LEGACY_SEED0_FOLD0_SIZES={}".format((len(lt), len(ltest))))
    print("LEGACY_VAL_EQUALS_TEST=True")
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise RuntimeError("Strict split sanity failed")
    print("=== SANITY CHECK PASSED ===")


if __name__ == "__main__":
    main()

