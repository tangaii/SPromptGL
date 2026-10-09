"""Shared data, split, reproducibility and provenance helpers."""

import hashlib
import json
import os
import random
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


MODALITY_NAMES = ["PHENO", "ANAT", "FUNC", "Correlation"]
SPROMPTGL_COMMIT = "ce983b85a36d19b5d22340d1cee101add0a0a060"


def set_rng_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_hyperparameters(args):
    return SimpleNamespace(
        datname="ABIDE", nclass=2, nmodal=4,
        nepoch=int(args.nepoch), early=int(args.early),
        lr=0.0038, reg=0.11, dropout=0.35,
        nlayer=1, n_hidden=18, n_head=2, n_iter=10,
        th=0.9, GC_mode="weighted-cosine", MP_mode="GCN",
        MF_mode="concat", alpha=0.5,
        theta_smooth=1.0, theta_degree=0.5, theta_sparsity=0.0,
        mode="simple-2", seed=int(args.seed),
        align_mode=args.align_mode,
        align_lambda=float(args.align_lambda),
        align_temperature=float(args.align_temperature),
        align_warmup=int(args.align_warmup),
        protocol=args.protocol,
    )


def load_abide(datadir):
    data_dir = os.path.join(datadir, "ABIDE")
    csv_path = os.path.join(data_dir, "processed_standard_data.csv")
    dict_path = os.path.join(data_dir, "modal_feat_dict.npy")
    data = pd.read_csv(csv_path).values
    modal_feat_dict = np.load(dict_path, allow_pickle=True).item()
    input_data = data[:, :-1].astype(np.float32)
    original_labels = data[:, -1].astype(np.int64)
    labels_internal = original_labels - 1
    dims = [len(modal_feat_dict[name]) for name in MODALITY_NAMES]
    if input_data.shape != (871, 320):
        raise ValueError("Unexpected ABIDE input shape: {}".format(input_data.shape))
    if dims != [48, 6, 10, 256] or sum(dims) != input_data.shape[1]:
        raise ValueError("Unexpected ABIDE modal dimensions: {}".format(dims))
    if sorted(np.unique(original_labels).tolist()) != [1, 2]:
        raise ValueError("Expected original labels 1 and 2")
    if np.bincount(labels_internal, minlength=2).tolist() != [403, 468]:
        raise ValueError("Unexpected internal class counts")
    return input_data, labels_internal, original_labels, dims, modal_feat_dict


def make_strict_split(input_data, labels, seed, fold_id):
    outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=int(seed))
    outer_train, outer_test = list(outer.split(input_data, labels))[int(fold_id)]
    inner = StratifiedShuffleSplit(
        n_splits=1, test_size=0.10,
        random_state=int(seed) + 1000 + int(fold_id),
    )
    relative_train, relative_val = next(
        inner.split(input_data[outer_train], labels[outer_train])
    )
    return (
        outer_train[relative_train],
        outer_train[relative_val],
        outer_test,
        outer_train,
    )


def make_legacy_splits(input_data, labels, seed):
    # The original SPromptGL fixes random_state=10, independent of seed.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    return list(skf.split(input_data, labels))


def module_fingerprint(*modules):
    digest = hashlib.sha256()
    for module in modules:
        for name, value in module.state_dict().items():
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def split_fingerprint(train_index, val_index=None, test_index=None):
    digest = hashlib.sha256()
    for name, values in (("train", train_index), ("val", val_index), ("test", test_index)):
        digest.update(name.encode("utf-8"))
        if values is not None:
            arr = np.asarray(values, dtype=np.int64)
            digest.update(arr.tobytes())
    return digest.hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_commit(repo_root):
    # The supplied project directory is an exported working tree rather than a
    # git worktree.  The upload commit is retained as the source provenance.
    return SPROMPTGL_COMMIT


def parse_legacy_reference(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    match = re.search(
        r"val=([0-9.]+)%\s+tst_acc=([0-9.]+)%\s+tst_auc=([0-9.]+)%",
        text,
    )
    std = re.search(r"tst_acc_std=([0-9.]+)\s+tst_auc_std=([0-9.]+)", text)
    sen = re.search(r"SEN_mean:\s*\[([^\]]+)\]", text)
    spe = re.search(r"SPE_mean:\s*\[([^\]]+)\]", text)
    result = {"source": path}
    if match:
        result.update({"val_acc_pct": float(match.group(1)),
                       "tst_acc_pct": float(match.group(2)),
                       "tst_auc_pct": float(match.group(3))})
    if std:
        result.update({"tst_acc_std_fraction": float(std.group(1)),
                       "tst_auc_std_fraction": float(std.group(2))})
    if sen:
        result["sen_mean"] = [float(x) for x in sen.group(1).split()]
    if spe:
        result["spe_mean"] = [float(x) for x in spe.group(1).split()]
    return result


def json_dump(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)

