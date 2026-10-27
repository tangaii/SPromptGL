"""Read-only data, historical anchors, and isolated source integration."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path("/data/adodas/SPromptGL-main")
EXP = Path(__file__).resolve().parent
RT = EXP / ".runtime"
HIST = ROOT / "experiment/experiment45_pid_guided_graph_adversarial_alignment"
HIST43 = ROOT / "experiment/experiment43_2_gfr_incremental_value_audit"
DIMS = (48, 6, 10, 256)
SLICES = ((0, 48), (48, 54), (54, 64), (64, 320))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob("*")) if path.is_file()}


def write_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def read_data():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    base = ROOT / "data/ABIDE"
    frame = pd.read_csv(base / "processed_standard_data.csv")
    mapping = np.load(base / "modal_feat_dict.npy", allow_pickle=True).item()
    expected_keys = ["PHENO", "ANAT", "FUNC", "Correlation"]
    if list(mapping) != expected_keys or [len(v) for v in mapping.values()] != list(DIMS):
        raise AssertionError("modal_feat_dict does not match the frozen four-modal layout")
    feature_columns = [str(value) for key in expected_keys for value in mapping[key]]
    if frame.shape != (871, 321) or list(frame.columns[:-1]) != feature_columns:
        raise AssertionError("processed_standard_data.csv layout differs from the manifest")
    x = frame.iloc[:, :-1].to_numpy(np.float32)
    y = frame.iloc[:, -1].to_numpy(np.int64) - 1
    if x.shape != (871, 320) or not np.isfinite(x).all() or np.bincount(y).tolist() != [403, 468]:
        raise AssertionError("unexpected ABIDE shape, labels, or non-finite values")
    folds = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y))
    return x, y, folds, mapping, feature_columns, frame


def site_mapping(frame, mapping, feature_columns):
    site_columns = [column for column in feature_columns if column.startswith("SITE_ID_")]
    expected = [str(value) for value in mapping["PHENO"] if str(value).startswith("SITE_ID_")]
    if site_columns != expected or len(site_columns) != 20:
        return {"valid": False, "reason": "SITE_MAPPING_UNVERIFIED", "columns": site_columns}
    values = frame[site_columns].to_numpy(np.int64)
    counts = values.sum(axis=1)
    if not np.array_equal(np.unique(values), np.asarray([0, 1])) or not np.all(counts == 1):
        return {"valid": False, "reason": "SITE_ONE_HOT_INVALID", "columns": site_columns}
    sites = values.argmax(axis=1).astype(np.int64)
    return {
        "valid": True,
        "columns": site_columns,
        "count": len(site_columns),
        "site_ids": list(range(len(site_columns))),
        "site_names": site_columns,
        "row_one_hot_sum_unique": np.unique(counts).tolist(),
        "site_counts": np.bincount(sites, minlength=len(site_columns)).tolist(),
        "sites": sites,
    }


def fold_sha(train, test):
    return hashlib.sha256(np.asarray(train, np.int64).tobytes() +
                          np.asarray(test, np.int64).tobytes()).hexdigest()


def source_copy():
    destination = RT / "sources"
    destination.mkdir(parents=True, exist_ok=True)
    source = HIST43 / ".runtime/sources"
    for name in ("network.py", "layers.py", "utils.py"):
        shutil.copy2(source / name, destination / name)
    return {name: sha(destination / name) for name in ("network.py", "layers.py", "utils.py")}


def copy_anchors():
    source = HIST43 / ".runtime/anchors"
    destination = RT / "anchors"
    destination.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 11):
        shutil.copy2(source / f"fold{fold:02}.pt", destination / f"fold{fold:02}.pt")
    shutil.copy2(source / "source_index.json", destination / "source_index.json")
    return {str(fold): sha(destination / f"fold{fold:02}.pt") for fold in range(1, 11)}


def protected_hashes():
    names = ["main.py", "model.py", "network.py", "layers.py", "utils.py",
             "ABIDE-simple-2-concat-weighted-cosine.sh",
             "data/ABIDE/processed_standard_data.csv", "data/ABIDE/modal_feat_dict.npy"]
    return {str(ROOT / name): sha(ROOT / name) for name in names}


def make_hp(cfg):
    # These values reconstruct the read-only B0-B backbone for a forward audit.
    return SimpleNamespace(datadir=str(ROOT / "data") + "/", datname="ABIDE", cpu=False, gpu="-1",
                           nepoch=450, early=200, lr=0.0038, reg=0.11, dropout=0.35,
                           nlayer=1, n_hidden=18, n_head=2, n_iter=10, nmodal=4,
                           th=0.9, GC_mode="weighted-cosine", MP_mode="GCN", MF_mode="concat",
                           alpha=.5, theta_smooth=1.0, theta_degree=.5, theta_sparsity=0.0,
                           nclass=2, mode="simple-2", seed=0)


def historical_rows():
    import csv
    with (HIST / "results/fold_metrics.csv").open(newline="") as handle:
        return {int(row["fold"]): row for row in csv.DictReader(handle) if int(row["group_id"]) == 0}
