#!/usr/bin/env python3
"""Experiment 55 DCRAA-GFR runner. All training artifacts stay under this directory."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/SPromptGL-main")
DATA_ROOT = ROOT / "data/ABIDE"
CFG_PATH = EXP / "config.json"
CFG = json.loads(CFG_PATH.read_text())
RT = EXP / ".runtime"
CKPTS = RT / "checkpoints"
TASKS = RT / "tasks"
LOGS = RT / "logs"
FOLD_DATA = RT / "fold_data"
REL_CACHE = RT / "relation_cache"
RESULTS = EXP / "results"
HIST_E1 = Path("/tmp/SPromptGL-E51/experiment51_dugsa/.runtime/checkpoints")
ANCHORS = ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
SOURCE_NAMES = ("run.py", "dugsa.py", "network.py", "layers.py", "utils.py")
DIMS = (48, 6, 10, 256)
GROUPS = [row["id"] for row in CFG["groups"]]
CTX = None
E1 = None
DATA = None
FOLDS = None
LAYOUT = None


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def atomic_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path, value):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def torch_load(path, map_location="cpu"):
    import torch

    return torch.load(path, map_location=map_location, weights_only=False)


def runtime():
    global CTX, E1
    if CTX is not None:
        return CTX
    os.environ["DUGSA_EXP"] = str(EXP)
    sys.path.insert(0, str(EXP))
    sys.path.insert(0, str(EXP / "source"))
    spec = importlib.util.spec_from_file_location("experiment55_e1_base", EXP / "source/e1_base.py")
    E1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E1)
    CTX = E1.runtime()
    return CTX


def load_data():
    global DATA, FOLDS, LAYOUT
    if DATA is not None:
        return DATA, FOLDS
    ctx = runtime()
    DATA, FOLDS = E1.load_data()
    import numpy as np

    mapping = np.load(DATA_ROOT / "modal_feat_dict.npy", allow_pickle=True).item()
    names = [str(v) for key in ("PHENO", "ANAT", "FUNC", "Correlation") for v in mapping[key]]
    site_names = [name for name in names if name.startswith(CFG["data"]["site_prefix"])]
    assert len(site_names) == 20
    assert names[:26][0].startswith("AGE_AT_SCAN_") and names[46:48] == ["SEX_1", "SEX_2"]
    assert names[48:54] == ["anat_cnr", "anat_efc", "anat_fber", "anat_fwhm", "anat_qi1", "anat_snr"]
    assert names[54:64] == ["func_efc", "func_fber", "func_fwhm", "func_dvars", "func_outlier", "func_quality", "func_mean_fd", "func_num_fd", "func_perc_fd", "func_gsr"]
    site_cols = np.asarray([names.index(name) for name in site_names], dtype=np.int64)
    site_matrix = DATA[0][:, site_cols]
    assert np.array_equal(site_matrix.sum(axis=1), np.ones(len(site_matrix), dtype=site_matrix.dtype))
    LAYOUT = {
        "feature_names": names,
        "site_names": site_names,
        "site_cols": site_cols,
        "age_cols": np.arange(26, dtype=np.int64),
        "sex_cols": np.asarray([names.index("SEX_1"), names.index("SEX_2")], dtype=np.int64),
        "anat_cols": np.arange(48, 54, dtype=np.int64),
        "func_cols": np.arange(54, 64, dtype=np.int64),
    }
    return DATA, FOLDS


def split_sha(fold):
    import numpy as np

    tr, te = load_data()[1][fold - 1]
    return hashlib.sha256(np.asarray(tr, np.int64).tobytes() + np.asarray(te, np.int64).tobytes()).hexdigest()


def source_signatures():
    return {name: sha_file(EXP / "source" / name) for name in ("e1_base.py", "dugsa.py", "network.py", "layers.py", "utils.py")}


def config_signature():
    return sha_json(CFG)


def check_input_signatures():
    expected = {
        str(DATA_ROOT / "processed_standard_data.csv"): "b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1",
        str(DATA_ROOT / "modal_feat_dict.npy"): "8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e",
        str(ROOT / "main.py"): "dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2",
        str(ROOT / "model.py"): "254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda",
        str(ROOT / "network.py"): "484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec",
        str(ROOT / "layers.py"): "4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a",
        str(ROOT / "utils.py"): "95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67",
    }
    observed = {path: sha_file(path) for path in expected}
    mismatches = {path: (expected[path], observed[path]) for path in expected if expected[path] != observed[path]}
    if mismatches:
        raise RuntimeError(f"INPUT_SIGNATURE_MISMATCH: {mismatches}")
    for path, digest in expected.items():
        if path.endswith("network.py"):
            continue
    assert sha_file(EXP / "source/network.py") == "5542f52d598e2f4d32aaad0c46b80c214fd562f74110a1d0ffcc619b9aefb3d1"
    assert sha_file(EXP / "source/dugsa.py") == "ee59de1eb00fad670a90c732356e2798c31481b9199701ed4bccc6c6c9a76f12"
    assert sha_file(EXP / "source/e1_base.py") == "32fc05c0e8c1211947f3003eae74d705eb4087e629b8a6af0fdf1878299a8718"
    return observed


def relation_ids(y_i, y_j):
    import numpy as np

    return np.where(y_i == y_j, np.where(y_i == 0, 0, 1), 2).astype(np.int64)


def balanced_pairs(nodes, labels, sites, per_cell, rng):
    """Equal-count sampling from AA/TT/AT x same/cross-site patient pairs."""
    import numpy as np

    nodes = np.asarray(nodes, dtype=np.int64)
    if len(nodes) < 2:
        return np.empty((0, 2), np.int64), np.empty(0, np.int64), np.empty(0, np.int64), {f"{r}_{s}": 0 for r in range(3) for s in range(2)}
    ii, jj = np.triu_indices(len(nodes), 1)
    pi, pj = nodes[ii], nodes[jj]
    rel = relation_ids(labels[pi], labels[pj])
    site = (sites[pi] == sites[pj]).astype(np.int64)
    cells = {(r, s): np.flatnonzero((rel == r) & (site == s)) for r in range(3) for s in range(2)}
    counts = {f"{r}_{s}": int(len(cells[(r, s)])) for r in range(3) for s in range(2)}
    nonzero = [len(v) for v in cells.values()]
    take = min(int(per_cell), min(nonzero)) if nonzero and min(nonzero) > 0 else 0
    if take == 0:
        return np.empty((0, 2), np.int64), np.empty(0, np.int64), np.empty(0, np.int64), counts
    chosen = []
    for key in sorted(cells):
        chosen.extend(rng.choice(cells[key], size=take, replace=False).tolist())
    chosen = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(chosen)
    return np.stack((pi[chosen], pj[chosen]), axis=1), rel[chosen], site[chosen], counts


def raw_pair_features(x, pairs, cols):
    import numpy as np

    values = np.asarray(x[:, cols], dtype=np.float64)
    a, b = values[pairs[:, 0]], values[pairs[:, 1]]
    cosine = np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-8)
    return np.concatenate((np.abs(a - b), a * b, cosine[:, None]), axis=1)


def fit_pair_classifier(features, labels, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(features)
    model = LogisticRegression(C=0.1, max_iter=400, solver="lbfgs", random_state=int(seed))
    model.fit(scaler.transform(features), labels)
    return scaler, model


def pair_probabilities(scaler, model, features):
    return model.predict_proba(scaler.transform(features))


def oof_relation_evidence(x, y, sites, outer_train, cols, fold, modality):
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=551000 + fold)
    all_true, all_prob, all_site = [], [], []
    support = {f"{r}_{s}": 0 for r in range(3) for s in range(2)}
    fold_support = []
    for k, (ti, vi) in enumerate(inner.split(outer_train, y[outer_train])):
        train_nodes, valid_nodes = outer_train[ti], outer_train[vi]
        train_pairs, train_rel, train_site, train_counts = balanced_pairs(
            train_nodes, y, sites, 128, np.random.default_rng(551100 + fold * 20 + k)
        )
        valid_pairs, valid_rel, valid_site, valid_counts = balanced_pairs(
            valid_nodes, y, sites, 128, np.random.default_rng(551200 + fold * 20 + k)
        )
        fold_support.append({"train": train_counts, "valid": valid_counts})
        if len(train_pairs) < 36 or len(valid_pairs) < 36 or len(np.unique(train_rel)) != 3:
            continue
        x_train = raw_pair_features(x, train_pairs, cols)
        x_valid = raw_pair_features(x, valid_pairs, cols)
        scaler, model = fit_pair_classifier(x_train, train_rel, 551300 + fold * 20 + k)
        probs = pair_probabilities(scaler, model, x_valid)
        all_true.append(valid_rel)
        all_prob.append(probs)
        all_site.append(valid_site)
        for r, s in zip(valid_rel, valid_site):
            support[f"{int(r)}_{int(s)}"] += 1
    if not all_true:
        return {
            "auc_macro": 0.5,
            "auc_by_class": [0.5, 0.5, 0.5],
            "direction_correct": False,
            "site_support_pass": False,
            "support": support,
            "confusion": [[0, 0, 0] for _ in range(3)],
            "n_oof_pairs": 0,
            "inner_support": fold_support,
            "status": "RELATION_OOF_UNIDENTIFIABLE",
        }
    yy = np.concatenate(all_true)
    pp = np.concatenate(all_prob)
    pred = pp.argmax(axis=1)
    per_class = []
    for r in range(3):
        binary = (yy == r).astype(np.int64)
        per_class.append(float(roc_auc_score(binary, pp[:, r])) if len(np.unique(binary)) == 2 else 0.5)
    macro = float(np.mean(per_class))
    conf = confusion_matrix(yy, pred, labels=[0, 1, 2]).astype(int).tolist()
    site_support_pass = all(support[f"{r}_{s}"] >= 32 for r in range(3) for s in range(2))
    direction_correct = all(value >= 0.5 for value in per_class)
    admitted = modality == "CORR" or (macro >= 0.55 and direction_correct and site_support_pass)
    return {
        "auc_macro": macro,
        "auc_by_class": per_class,
        "direction_correct": direction_correct,
        "site_support_pass": site_support_pass,
        "support": support,
        "confusion": conf,
        "n_oof_pairs": int(len(yy)),
        "inner_support": fold_support,
        "status": "ADMITTED" if admitted else "NOT_ADMITTED",
    }


def patient_probability_proxy(x, y, train_nodes, test_nodes, corr_cols, fold):
    """Patient-disjoint CORR logistic OOF proxy and train-only temperature calibration."""
    import numpy as np
    from scipy.optimize import minimize_scalar
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    corr = np.asarray(x[:, corr_cols], dtype=np.float64)
    oof_prob = np.zeros(len(train_nodes), dtype=np.float64)
    y_train = y[train_nodes]
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=552000 + fold)
    for k, (ti, vi) in enumerate(inner.split(train_nodes, y_train)):
        fit_ids, val_ids = train_nodes[ti], train_nodes[vi]
        scaler = StandardScaler().fit(corr[fit_ids])
        model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=552100 + fold * 10 + k)
        model.fit(scaler.transform(corr[fit_ids]), y[fit_ids])
        oof_prob[vi] = model.predict_proba(scaler.transform(corr[val_ids]))[:, 0]
    eps = 1e-7
    logits = np.log(np.clip(oof_prob, eps, 1 - eps) / np.clip(1 - oof_prob, eps, 1 - eps))
    positive = (y_train == 0).astype(np.float64)

    def objective(log_temperature):
        z = logits / np.exp(log_temperature)
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return float(-np.mean(positive * np.log(np.clip(p, eps, 1)) + (1 - positive) * np.log(np.clip(1 - p, eps, 1))))

    result = minimize_scalar(objective, bounds=(math.log(0.5), math.log(3.0)), method="bounded", options={"xatol": 1e-8})
    temperature = float(np.exp(result.x))
    oof_asd = 1 / (1 + np.exp(-np.clip(logits / temperature, -30, 30)))
    full_scaler = StandardScaler().fit(corr[train_nodes])
    full_model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=552200 + fold)
    full_model.fit(full_scaler.transform(corr[train_nodes]), y[train_nodes])
    test_raw = full_model.predict_proba(full_scaler.transform(corr[test_nodes]))[:, 0]
    test_logits = np.log(np.clip(test_raw, eps, 1 - eps) / np.clip(1 - test_raw, eps, 1 - eps))
    test_asd = 1 / (1 + np.exp(-np.clip(test_logits / temperature, -30, 30)))
    probabilities = np.full((len(x), 2), 0.5, dtype=np.float64)
    probabilities[train_nodes, 0] = oof_asd
    probabilities[train_nodes, 1] = 1 - oof_asd
    probabilities[test_nodes, 0] = test_asd
    probabilities[test_nodes, 1] = 1 - test_asd
    return probabilities.astype(np.float32), {
        "inner_splits": 5,
        "temperature": temperature,
        "oof_asd_mean": float(oof_asd.mean()),
        "oof_asd_std": float(oof_asd.std()),
        "oof_brier": float(np.mean((oof_asd - positive) ** 2)),
        "calibration_rows": int(len(train_nodes)),
        "test_rows": int(len(test_nodes)),
    }


def disease_conditions(probabilities, pairs):
    import numpy as np

    pi = probabilities[pairs[:, 0]]
    pj = probabilities[pairs[:, 1]]
    result = np.stack((pi[:, 0] * pj[:, 0], pi[:, 1] * pj[:, 1], pi[:, 0] * pj[:, 1] + pi[:, 1] * pj[:, 0]), axis=1)
    assert np.allclose(result.sum(axis=1), 1.0, atol=1e-6)
    return result.astype(np.float32)


def condition_confidence(condition):
    import numpy as np

    entropy = -(condition.clip(1e-8) * np.log(condition.clip(1e-8))).sum(axis=1) / math.log(3.0)
    return np.clip(1.0 - entropy, 0.0, 1.0).astype(np.float32)


def mismatch_rows(values, strata, rng, minimum_fraction=0.25):
    import numpy as np

    n = len(values)
    best = np.arange(n, dtype=np.int64)
    best_changed = 0
    buckets = {}
    for idx, key in enumerate(strata):
        buckets.setdefault(tuple(key) if isinstance(key, (tuple, list, np.ndarray)) else key, []).append(idx)
    for _ in range(80):
        perm = np.arange(n, dtype=np.int64)
        for bucket in buckets.values():
            if len(bucket) > 1:
                shuffled = np.asarray(bucket, dtype=np.int64)
                draw = shuffled.copy()
                rng.shuffle(draw)
                if np.all(draw == shuffled):
                    draw = np.roll(shuffled, 1)
                perm[shuffled] = draw
        changed = int(np.sum(np.any(np.abs(values - values[perm]) > 1e-5, axis=1))) if values.ndim == 2 else int(np.sum(values != values[perm]))
        if changed > best_changed:
            best, best_changed = perm, changed
        if changed / max(n, 1) >= minimum_fraction:
            break
    return best, best_changed / max(n, 1)


def shuffle_site_targets(rel, condition, confidence, site_source, rng):
    import numpy as np

    bins = np.minimum((confidence * 4).astype(np.int64), 3)
    strata = [(int(r), int(np.argmax(c)), int(b)) for r, c, b in zip(rel, condition, bins)]
    perm, fraction = mismatch_rows(site_source[:, None], strata, rng, minimum_fraction=0.05)
    return site_source[perm].astype(np.int64), fraction


def site_split_groups(sites, train_nodes):
    import numpy as np

    is_train = np.zeros(len(sites), dtype=np.int64)
    is_train[np.asarray(train_nodes, dtype=np.int64)] = 1
    return [(int(sites[i]), int(is_train[i])) for i in range(len(sites))]


def degree_preserving_site_swaps(send, recv, sites, is_train, seed):
    """Directed edge swaps preserve every row/column degree and site/split strata."""
    import numpy as np

    rng = np.random.default_rng(seed)
    edges = [(int(i), int(j)) for i, j in zip(recv, send)]
    edge_set = set(edges)
    recv_group = [(int(sites[i]), int(is_train[i])) for i, _ in edges]
    send_group = [(int(sites[j]), int(is_train[j])) for _, j in edges]
    changed = 0
    attempts = max(1000, len(edges) * 24)
    for _ in range(attempts):
        a, b = rng.integers(0, len(edges), size=2)
        if a == b:
            continue
        i, j = edges[a]
        k, l = edges[b]
        if i == k or j == l or recv_group[a] != recv_group[b] or send_group[a] != send_group[b]:
            continue
        if i == l or k == j or (i, l) in edge_set or (k, j) in edge_set:
            continue
        edge_set.remove((i, j))
        edge_set.remove((k, l))
        edge_set.add((i, l))
        edge_set.add((k, j))
        edges[a] = (i, l)
        edges[b] = (k, j)
        changed += 2
    wrong = sorted(edge_set)
    wrong_recv = np.asarray([i for i, _ in wrong], dtype=np.int64)
    wrong_send = np.asarray([j for _, j in wrong], dtype=np.int64)
    orig_pairs = set((int(i), int(j)) for i, j in zip(recv, send))
    frac = 1.0 - len(orig_pairs.intersection(edge_set)) / max(len(orig_pairs), 1)
    assert len(wrong) == len(edges)
    assert np.array_equal(np.bincount(recv, minlength=len(sites)), np.bincount(wrong_recv, minlength=len(sites)))
    assert np.array_equal(np.bincount(send, minlength=len(sites)), np.bincount(wrong_send, minlength=len(sites)))
    def strata_count(r, s):
        return Counter((int(sites[i] == sites[j]), int(is_train[i]), int(is_train[j])) for i, j in zip(r, s))
    assert strata_count(recv, send) == strata_count(wrong_recv, wrong_send)
    return wrong_recv, wrong_send, frac


def corr_candidate_support(fold):
    import numpy as np
    import torch

    runtime()
    state = E1.build_state("E1", fold)
    state["model"].eval()
    with torch.no_grad():
        _, parts = state["parts"](state["feat"])
        _, adjs, _ = state["model"](parts, None)
    corr_adj = adjs[3].detach().cpu().numpy()
    n = len(corr_adj)
    recv, send, strength = [], [], []
    for i in range(n):
        candidates = np.flatnonzero((corr_adj[i] > 0) & (np.arange(n) != i))
        if len(candidates) == 0:
            continue
        order = candidates[np.argsort(-corr_adj[i, candidates], kind="stable")[: CFG["propagation"]["max_neighbors"]]]
        recv.extend([i] * len(order))
        send.extend(order.tolist())
        strength.extend(corr_adj[i, order].astype(np.float32).tolist())
    del state
    torch.cuda.empty_cache()
    return np.asarray(recv, np.int64), np.asarray(send, np.int64), np.asarray(strength, np.float32)


def cache_path(fold):
    return REL_CACHE / f"fold{fold:02d}.npz"


def cache_meta_path(fold):
    return REL_CACHE / f"fold{fold:02d}.json"


def load_fold_cache(fold):
    import numpy as np

    p = cache_path(fold)
    meta = json.loads(cache_meta_path(fold).read_text())
    if (meta["cache_sha256"] != sha_file(p) or meta["config_signature"] != config_signature()
            or meta["split_sha"] != split_sha(fold) or meta["runner_sha256"] != sha_file(EXP / "run.py")
            or meta["dcraa_sha256"] != sha_file(EXP / "dcraa.py")):
        raise RuntimeError(f"STALE_FOLD_CACHE fold={fold}")
    return {key: value for key, value in np.load(p).items()}, meta


def prepare_fold_cache(fold, force=False):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    x, y = load_data()[0]
    tr, te = load_data()[1][fold - 1]
    tr, te = np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)
    pth = cache_path(fold)
    mpth = cache_meta_path(fold)
    if not force and pth.exists() and mpth.exists():
        try:
            return load_fold_cache(fold)[1]
        except Exception:
            pass
    sites = np.argmax(x[:, LAYOUT["site_cols"]], axis=1).astype(np.int64)
    rng = np.random.default_rng(553000 + fold)
    train_pairs, train_rel, train_site, train_counts = balanced_pairs(tr, y, sites, CFG["relation"]["teacher_pair_cells"], rng)
    if len(train_pairs) == 0 or len(np.unique(train_rel)) != 3 or len(np.unique(train_site)) != 2:
        raise RuntimeError(f"RELATION_PAIR_SUPPORT_INVALID fold={fold} counts={train_counts}")
    corr_cols = np.arange(64, 320, dtype=np.int64)
    probabilities, patient_teacher_meta = patient_probability_proxy(x, y, tr, te, corr_cols, fold)
    conditions = disease_conditions(probabilities, train_pairs)
    confidences = condition_confidence(conditions)

    inner_rel, inner_probs = [], []
    evidence = {}
    modal_cols = {
        "PHENO": np.arange(48),
        "ANAT": np.arange(48, 54),
        "FUNC": np.arange(54, 64),
        "CORR": np.arange(64, 320),
    }
    for modality, cols in modal_cols.items():
        evidence[modality] = oof_relation_evidence(x, y, sites, tr, cols, fold, modality)
    admitted = [m for m in ("PHENO", "ANAT", "FUNC") if evidence[m]["status"] == "ADMITTED"]

    teacher_scaler, teacher_model = fit_pair_classifier(
        raw_pair_features(x, train_pairs, corr_cols), train_rel, 553100 + fold
    )
    relation_teacher_prob = pair_probabilities(
        teacher_scaler, teacher_model, raw_pair_features(x, train_pairs, corr_cols)
    ).astype(np.float32)

    condition_perm, condition_mismatch = mismatch_rows(
        conditions,
        [(0,)] * len(conditions),
        np.random.default_rng(553200 + fold),
        minimum_fraction=0.35,
    )
    wrong_condition = conditions[condition_perm].copy()
    wrong_site, wrong_site_fraction = shuffle_site_targets(
        train_rel,
        conditions,
        confidences,
        train_site,
        np.random.default_rng(553300 + fold),
    )

    edge_recv, edge_send, edge_strength = corr_candidate_support(fold)
    is_train = np.zeros(len(x), dtype=np.int64)
    is_train[tr] = 1
    wrong_recv, wrong_send, wrong_edge_fraction = degree_preserving_site_swaps(
        edge_send, edge_recv, sites, is_train, 553400 + fold
    )
    if not np.array_equal(edge_recv, wrong_recv):
        raise AssertionError("wrong graph changed receiver order")
    edge_condition = disease_conditions(probabilities, np.stack((edge_recv, edge_send), axis=1))
    edge_perm, edge_condition_mismatch = mismatch_rows(
        edge_condition,
        [(0,)] * len(edge_condition),
        np.random.default_rng(553500 + fold),
        minimum_fraction=0.35,
    )
    mismatched_edge_condition = edge_condition[edge_perm].copy()
    wrong_edge_condition = disease_conditions(
        probabilities, np.stack((edge_recv, wrong_send), axis=1)
    )

    quality_cols = np.concatenate((LAYOUT["anat_cols"], LAYOUT["func_cols"]))
    q_mean = x[tr][:, quality_cols].mean(axis=0)
    q_std = x[tr][:, quality_cols].std(axis=0).clip(1e-6)
    quality = ((x[:, quality_cols] - q_mean) / q_std).astype(np.float32)

    def edge_quality(rr, ss):
        if len(rr) == 0:
            return np.empty(0, dtype=np.float32)
        d2 = np.square(quality[rr] - quality[ss]).mean(axis=1)
        return np.exp(-0.5 * np.minimum(d2, 20.0)).astype(np.float32)

    true_edge_quality = edge_quality(edge_recv, edge_send)
    wrong_edge_quality = edge_quality(wrong_recv, wrong_send)
    arrays = {
        "train": tr,
        "test": te,
        "sites": sites,
        "probabilities": probabilities,
        "train_pairs": train_pairs,
        "train_rel": train_rel,
        "train_site": train_site,
        "train_condition": conditions,
        "train_confidence": confidences,
        "wrong_condition": wrong_condition,
        "wrong_site": wrong_site,
        "teacher_relation_prob": relation_teacher_prob,
        "edge_recv": edge_recv,
        "edge_send": edge_send,
        "edge_strength": edge_strength,
        "edge_quality": true_edge_quality,
        "wrong_edge_send": wrong_send,
        "wrong_edge_quality": wrong_edge_quality,
        "edge_condition": edge_condition,
        "mismatched_edge_condition": mismatched_edge_condition,
        "wrong_edge_condition": wrong_edge_condition,
    }
    tmp = pth.with_name(pth.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, pth)
    site_cell_counts = {f"{r}_{s}": int(train_counts[f"{r}_{s}"]) for r in range(3) for s in range(2)}
    effective_confidence = (conditions * confidences[:, None]).sum(axis=0)
    adversary_identifiable = all(v >= CFG["relation"]["min_adversary_support_per_cell"] for v in train_counts.values())
    confidence_supported = bool(np.all(effective_confidence >= 1e-3))
    if not adversary_identifiable:
        adv_status = "ADVERSARY_UNIDENTIFIABLE"
    elif not confidence_supported:
        adv_status = "ADVERSARY_INACTIVE"
    else:
        adv_status = "IDENTIFIABLE"
    meta = {
        "fold": fold,
        "split_sha": split_sha(fold),
        "config_signature": config_signature(),
        "runner_sha256": sha_file(EXP / "run.py"),
        "dcraa_sha256": sha_file(EXP / "dcraa.py"),
        "cache_sha256": sha_file(pth),
        "patient_teacher": patient_teacher_meta,
        "relation_evidence": evidence,
        "admitted_auxiliary_modalities": admitted,
        "train_pair_count": int(len(train_pairs)),
        "train_pair_cells_sampled": {f"{r}_{s}": int(np.sum((train_rel == r) & (train_site == s))) for r in range(3) for s in range(2)},
        "train_pair_cells_available": site_cell_counts,
        "condition_sum_max_error": float(np.max(np.abs(conditions.sum(axis=1) - 1.0))),
        "wrong_disease_condition_changed_fraction": float(condition_mismatch),
        "wrong_disease_edge_condition_changed_fraction": float(edge_condition_mismatch),
        "wrong_site_source_changed_fraction": float(wrong_site_fraction),
        "candidate_edges": int(len(edge_send)),
        "candidate_neighbors_mean": float(np.mean(np.bincount(edge_recv, minlength=len(x)))) if len(edge_recv) else 0.0,
        "candidate_neighbors_max": int(np.max(np.bincount(edge_recv, minlength=len(x)))) if len(edge_recv) else 0,
        "candidate_same_site_fraction": float(np.mean(sites[edge_recv] == sites[edge_send])) if len(edge_send) else 0.0,
        "candidate_cross_site_fraction": float(np.mean(sites[edge_recv] != sites[edge_send])) if len(edge_send) else 0.0,
        "wrong_edge_changed_fraction": float(wrong_edge_fraction),
        "effective_condition_confidence": effective_confidence.astype(float).tolist(),
        "adversary_status": adv_status,
    }
    atomic_json(mpth, meta)
    return meta


def prepare_all_caches(force=False):
    check_input_signatures()
    load_data()
    rows = []
    for fold in range(1, 11):
        rows.append(prepare_fold_cache(fold, force=force))
        print("CACHE", fold, rows[-1]["candidate_edges"], rows[-1]["admitted_auxiliary_modalities"], rows[-1]["adversary_status"], flush=True)
    return rows


def device_cache(fold):
    import numpy as np
    import torch

    arrays, meta = load_fold_cache(fold)
    on_device = {}
    for key, value in arrays.items():
        if value.dtype.kind in "iu":
            on_device[key] = torch.as_tensor(value, dtype=torch.long, device="cuda")
        else:
            on_device[key] = torch.as_tensor(value, dtype=torch.float32, device="cuda")
    return on_device, meta


def dcraa_optimizer(module):
    import torch

    decayed, no_decay = [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 or name.endswith(".bias") else decayed).append(param)
    return torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": CFG["optimizer_new"]["matrix_weight_decay"]},
            {"params": no_decay, "weight_decay": CFG["optimizer_new"]["bias_layernorm_weight_decay"]},
        ],
        lr=CFG["optimizer_new"]["lr"],
    )


def make_dcraa(fold):
    import torch
    import importlib.util

    spec = importlib.util.spec_from_file_location("experiment55_dcraa", EXP / "dcraa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Isolate all newly initialized parameters from the E1 dropout RNG stream.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(555000 + fold)
        torch.cuda.manual_seed_all(555000 + fold)
        model = module.DCRAA(DIMS, CFG["relation"]["dimension"], CFG["loss"]["grl_scale"]).cuda()
    return model, module


def build_base_state(fold):
    import torch

    runtime()
    state = E1.build_state("E1", fold)
    for group in state["opt_model"].param_groups:
        assert abs(float(group["weight_decay"]) - 0.11) < 1e-12
    for group in state["opt_cam"].param_groups:
        assert abs(float(group["weight_decay"]) - 0.11) < 1e-12
    return state


def new_state(group, fold):
    import torch

    state = build_base_state(fold)
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone()
    dcraa, module = make_dcraa(fold)
    isolated = torch.equal(cpu_before, torch.get_rng_state()) and torch.equal(cuda_before, torch.cuda.get_rng_state())
    if not isolated:
        raise RuntimeError("DCRAA_INITIALIZATION_CHANGED_E1_RNG")
    state.update(
        dcr_group=group,
        dcraa=dcraa,
        dcraa_module=module,
        opt_dcraa=dcraa_optimizer(dcraa),
        train_nodes=state["train"],
        base_group="E1",
        rng_isolation_verified=isolated,
    )
    return state


def chosen_condition(cache, group, prefix="train"):
    if group == "E6":
        return cache["wrong_condition"]
    return cache[f"{prefix}_condition"]


def site_labels(cache, group):
    return cache["wrong_site"] if group == "E7" else cache["train_site"]


def active_modalities(meta):
    return meta.get("admitted_auxiliary_modalities", [])


def forward_group(state, cache, meta, train_mode=True):
    import torch
    import torch.nn.functional as F

    group = state["dcr_group"]
    state["model"].train(train_mode)
    state["cam"].train(train_mode)
    _, parts = state["parts"](state["feat"])
    base_fused, adjs, details = state["model"](parts, None)
    graph = sum(
        runtime()["utils"].GraphConstructLoss(
            parts[i], adjs[i], CFG["base"]["theta_smooth"],
            CFG["base"]["theta_degree"], CFG["base"]["theta_sparsity"],
        ) for i in range(4)
    ) / 4.0
    dcraa = state["dcraa"]
    dcraa.train(train_mode)
    x = state["feat"]
    corr = x[:, 64:320]
    pair_ids = cache["train_pairs"]
    pair_rel = cache["train_rel"]
    pair_site = site_labels(cache, group)
    condition = chosen_condition(cache, group)
    confidence = state["dcraa_module"].confidence_from_conditions(condition)
    pair_z = dcraa.pair_representation(corr, pair_ids)
    rel_logits = dcraa.relation_logits(pair_z)

    zero = graph.new_zeros(())
    rel_loss = zero
    teacher_loss = zero
    if group not in {"E1", "E10"}:
        rel_loss = F.cross_entropy(rel_logits, pair_rel)
    if group in {"E5", "E6", "E7", "E8", "E9"}:
        teacher_loss = dcraa.protection_loss(rel_logits, cache["teacher_relation_prob"])

    adv_loss = zero
    adv_logits = None
    adv_active = meta["adversary_status"] == "IDENTIFIABLE"
    if group in {"E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"} and adv_active:
        if group == "E3":
            adv_logits = dcraa.unconditional_site_logits(pair_z)
            adv_loss = dcraa.unconditional_adv_loss(adv_logits, pair_site)
        else:
            adv_logits = dcraa.conditional_site_logits(pair_z)
            adv_loss = dcraa.conditional_adv_loss(adv_logits, pair_site, condition, confidence)

    auxiliary_loss = zero
    if group in {"E5", "E6", "E7", "E8", "E9", "E10"}:
        modalities = active_modalities(meta)
        if modalities:
            inputs = [x[:, :48], x[:, 48:54], x[:, 54:64], corr]
            aux_z = dcraa.auxiliary_pairs(inputs, pair_ids)
            selected = [aux_z[["PHENO", "ANAT", "FUNC"].index(m)] for m in modalities]
            auxiliary_loss = dcraa.auxiliary_prototype_loss(pair_z, selected, condition, confidence)

    recv = cache["edge_recv"]
    send = cache["wrong_edge_send"] if group == "E8" else cache["edge_send"]
    edge_pairs = torch.stack((recv, send), dim=1)
    if group == "E6":
        edge_condition = cache["mismatched_edge_condition"]
    elif group == "E8":
        edge_condition = cache["wrong_edge_condition"]
    else:
        edge_condition = cache["edge_condition"]
    edge_z = dcraa.pair_representation(corr, edge_pairs)
    edge_logits = dcraa.relation_logits(edge_z)
    edge_strength = cache["edge_strength"]
    edge_quality = cache["wrong_edge_quality"] if group == "E8" else cache["edge_quality"]
    edge_raw, edge_disease_score, edge_confidence = dcraa.edge_weights(
        edge_logits, edge_condition, edge_strength, edge_quality
    )
    degree = torch.zeros(len(x), dtype=x.dtype, device=x.device)
    degree.index_add_(0, recv, edge_raw)
    edge_norm = edge_raw / degree[recv].clamp_min(1e-8)
    delta = torch.zeros_like(corr)
    delta.index_add_(0, recv, edge_norm[:, None] * (corr[send] - corr[recv]))
    bounded_delta = dcraa.bounded_residual(delta, base_fused[:, 64:320])
    message_enabled = group != "E9"
    fused = dcraa.writeback(base_fused, bounded_delta, enabled=message_enabled)
    prob, hidden, attn = state["cam"](fused)
    cls = runtime()["utils"].ClsLoss(prob, state["targ"], state["train"], state["weight"])
    total = cls + graph

    loss = total
    loss = loss + CFG["loss"]["lambda_relation"] * rel_loss
    loss = loss + CFG["loss"]["lambda_adversary"] * adv_loss
    loss = loss + CFG["loss"]["lambda_teacher_anchor"] * teacher_loss
    loss = loss + CFG["loss"]["lambda_auxiliary"] * auxiliary_loss
    residual = fused[:, 64:320] - base_fused[:, 64:320]
    return {
        "parts": parts,
        "adjs": adjs,
        "details": details,
        "base_fused": base_fused,
        "e1_message": base_fused - x,
        "graph": graph,
        "align": graph.new_zeros(()),
        "nuisance": graph.new_zeros(()),
        "noncollapse": graph.new_zeros(()),
        "prob": prob,
        "hidden": hidden,
        "fused": fused,
        "cls": cls,
        "total": loss,
        "relation_logits": rel_logits,
        "relation_loss": rel_loss,
        "adversary_loss": adv_loss,
        "teacher_loss": teacher_loss,
        "auxiliary_loss": auxiliary_loss,
        "adversary_logits": adv_logits,
        "adversary_active": bool(adv_active and adv_logits is not None),
        "edge_raw": edge_raw,
        "edge_weights": edge_norm,
        "edge_disease_score": edge_disease_score,
        "edge_confidence": edge_confidence,
        "delta": delta,
        "bounded_delta": bounded_delta,
        "residual": residual,
        "message_enabled": message_enabled,
    }


def rng_pack():
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state().cpu(),
        "cuda": [v.cpu() for v in torch.cuda.get_rng_state_all()],
        "relation_numpy": None,
    }


def rng_restore(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def state_cpu(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def optimizer_cpu(optimizer):
    return copy.deepcopy(optimizer.state_dict())


def grad_norm_module(module):
    import torch

    values = [(p.grad.detach().float().square().sum()) for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt().detach().cpu()) if values else 0.0


def metrics55(log_prob, state):
    """E1 hard metrics with Probability-AUC explicitly scored as ASD probability."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    result = E1.metrics(log_prob, state)
    idx = np.asarray(state["test"], dtype=np.int64)
    y = state["targ"].detach().cpu().numpy()[idx]
    p = log_prob.detach().exp().cpu().numpy()[idx]
    result["prob_auc"] = float(roc_auc_score((y == 0).astype(np.int64), p[:, 0]))
    return result


def tensor_state_equal(left, right, atol=0.0, rtol=0.0):
    import torch

    if left.keys() != right.keys():
        return False, "keys"
    for key in left:
        a, b = left[key].detach().cpu(), right[key].detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype or not torch.allclose(a, b, atol=atol, rtol=rtol):
            return False, key
    return True, ""


def recursive_equal(left, right, path="root"):
    import torch

    if isinstance(left, torch.Tensor):
        return bool(left.shape == right.shape and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())), path
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return False, path + ".keys"
        for key in left:
            okay, where = recursive_equal(left[key], right[key], path + "." + str(key))
            if not okay:
                return False, where
        return True, ""
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return False, path + ".length"
        for index, (a, b) in enumerate(zip(left, right)):
            okay, where = recursive_equal(a, b, path + f"[{index}]")
            if not okay:
                return False, where
        return True, ""
    try:
        return bool(left == right), path
    except Exception:
        return False, path


def e0_task(fold):
    import torch

    runtime()
    x, y = load_data()[0]
    tr, _ = load_data()[1][fold - 1]
    old_path = HIST_E1 / f"E1_f{fold:02d}.best.pt"
    if not old_path.exists():
        raise FileNotFoundError(f"HISTORICAL_E1_CHECKPOINT_NOT_FOUND: {old_path}")
    old = torch_load(old_path)
    old_task_path = old_path.parents[1] / "tasks" / f"E1_f{fold:02d}.json"
    if not old_task_path.exists():
        raise FileNotFoundError(f"HISTORICAL_E1_TASK_NOT_FOUND: {old_task_path}")
    old_task = json.loads(old_task_path.read_text())
    state = build_base_state(fold)
    train = torch.as_tensor(tr, dtype=torch.long, device="cuda")
    state["group"] = "E1"
    best_acc, wait, best_epoch = -1.0, 0, -1
    best = best_cam = best_opt = best_cam_opt = best_rng = None
    epoch = -1
    start = time.time()
    for epoch in range(CFG["base"]["epochs"]):
        state["opt_model"].zero_grad()
        state["opt_cam"].zero_grad()
        out = E1.forward(state, {}, train_mode=True)
        out["total"].backward()
        state["opt_model"].step()
        state["opt_cam"].step()
        with torch.no_grad():
            ev = E1.forward(state, {}, train_mode=False)
            metrics = metrics55(ev["prob"], state)
        if metrics["acc"] > best_acc:
            best_acc, best_epoch, wait = metrics["acc"], epoch, 0
            best, best_cam = state_cpu(state["model"]), state_cpu(state["cam"])
            best_opt, best_cam_opt = optimizer_cpu(state["opt_model"]), optimizer_cpu(state["opt_cam"])
            best_rng = rng_pack()
        else:
            wait += 1
        if wait > CFG["cv"]["early_stop_wait_gt"]:
            break
    if best is None:
        raise RuntimeError(f"E0_NO_CHECKPOINT fold={fold}")
    state["model"].load_state_dict(best)
    state["cam"].load_state_dict(best_cam)
    with torch.no_grad():
        final_metrics = E1.metrics(E1.forward(state, {}, train_mode=False)["prob"], state)
    model_ok, model_where = tensor_state_equal(best, old["model"])
    cam_ok, cam_where = tensor_state_equal(best_cam, old["cam"])
    opt_model_ok, opt_model_where = recursive_equal(best_opt, old["opt_model"], "opt_model")
    opt_cam_ok, opt_cam_where = recursive_equal(best_cam_opt, old["opt_cam"], "opt_cam")
    pred = E1.forward(state, {}, train_mode=False)["prob"].detach().exp().cpu().numpy()
    old_state = build_base_state(fold)
    old_state["model"].load_state_dict(old["model"])
    old_state["cam"].load_state_dict(old["cam"])
    with torch.no_grad():
        old_pred = E1.forward(old_state, {}, train_mode=False)["prob"].detach().exp().cpu().numpy()
    pred_equal = bool((pred == old_pred).all())
    exact = (
        best_epoch == int(old["best_epoch"])
        and epoch + 1 == int(old_task["epochs_executed"])
        and all(abs(float(final_metrics[k]) - float(old["metrics"][k])) <= 1e-12 for k in final_metrics)
        and model_ok and cam_ok and opt_model_ok and opt_cam_ok and pred_equal
    )
    row = {
        "group": "E0", "fold": fold, "status": "COMPLETE" if exact else "PARITY_FAIL",
        "source": "E51_E1_exact_replay", "best_epoch": best_epoch,
        "epochs_executed": epoch + 1, "metrics": final_metrics,
        "split_sha": split_sha(fold),
        "historical_best_epoch": old["best_epoch"], "historical_epochs_executed": old["epochs_executed"],
        "model_exact": model_ok, "model_first_difference": model_where,
        "cam_exact": cam_ok, "cam_first_difference": cam_where,
        "opt_model_exact": opt_model_ok, "opt_model_first_difference": opt_model_where,
        "opt_cam_exact": opt_cam_ok, "opt_cam_first_difference": opt_cam_where,
        "prediction_exact": pred_equal, "historical_metrics": old["metrics"],
        "anchor_parameter_sha": old["anchor_parameter_sha"], "anchor_rng_sha": old["anchor_rng_sha"],
        "source_checkpoint_sha256": sha_file(old_path), "wall_seconds": time.time() - start,
        "first_forward_loss_available": False,
    }
    local_checkpoint = CKPTS / f"E0_f{fold:02d}.best.pt"
    atomic_torch_save(local_checkpoint, {
        "group": "E0", "fold": fold, "best_epoch": best_epoch,
        "epochs_executed": epoch + 1, "metrics": final_metrics,
        "model": best, "cam": best_cam, "opt_model": best_opt,
        "opt_cam": best_cam_opt, "rng": best_rng,
        "source_checkpoint_sha": sha_file(old_path), "parity": row,
    })
    row["best_sha"] = sha_file(local_checkpoint)
    atomic_json(TASKS / f"E0_f{fold:02d}.json", row)
    print("E0_PARITY", fold, "PASS" if exact else "FAIL", best_epoch, final_metrics["acc"], flush=True)
    return exact


def e0_all():
    check_input_signatures()
    results = [e0_task(fold) for fold in range(1, 11)]
    if not all(results):
        atomic_json(EXP / "parity.json", {"status": "FAIL", "folds": results})
        raise RuntimeError("E0_PARITY_FAILED; formal groups are blocked")
    atomic_json(EXP / "parity.json", {
        "status": "PASS", "folds": 10, "source": str(HIST_E1),
        "runner_sha256": sha_file(EXP / "run.py"),
        "e1_helper_sha256": sha_file(EXP / "source/e1_base.py"),
        "historical_checkpoint_sha256": {f"fold{f:02d}": sha_file(HIST_E1 / f"E1_f{f:02d}.best.pt") for f in range(1, 11)},
        "full_model_cam_optimizer_prediction_metric_parity": "10_OF_10",
        "first_forward_historical_loss": "NOT_AVAILABLE_IN_HISTORICAL_TASKS",
    })
    manifest_path = EXP / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["rng_protocol"]["parity"] = "PASS_10_OF_10"
        manifest["status"] = "E0_PARITY_PASS"
        manifest["e0_parity_sha256"] = sha_file(EXP / "parity.json")
        atomic_json(manifest_path, manifest)


def train_one(group, fold):
    import torch

    if group not in GROUPS[1:]:
        raise ValueError(f"UNSUPPORTED_NEW_GROUP {group}")
    check_input_signatures()
    if not (EXP / "parity.json").exists() or json.loads((EXP / "parity.json").read_text()).get("status") != "PASS":
        raise RuntimeError("E0_PARITY_REQUIRED_BEFORE_NEW_TRAINING")
    cache, meta = device_cache(fold)
    task = TASKS / f"{group}_f{fold:02d}.json"
    if task.exists() and json.loads(task.read_text()).get("status") == "COMPLETE":
        return
    state = new_state(group, fold)
    initial_module = state_cpu(state["dcraa"])
    init_sha = sha_json({k: v.numpy().tobytes().hex() for k, v in initial_module.items()})
    best_path = CKPTS / f"{group}_f{fold:02d}.best.pt"
    latest_path = CKPTS / f"{group}_f{fold:02d}.latest.pt"
    best_acc, wait, best_epoch = -1.0, 0, -1
    best_diag = None
    start_epoch = 0
    if latest_path.exists() and best_path.exists():
        latest = torch_load(latest_path)
        state["model"].load_state_dict(latest["model"])
        state["cam"].load_state_dict(latest["cam"])
        state["dcraa"].load_state_dict(latest["dcraa"])
        state["opt_model"].load_state_dict(latest["opt_model"])
        state["opt_cam"].load_state_dict(latest["opt_cam"])
        state["opt_dcraa"].load_state_dict(latest["opt_dcraa"])
        rng_restore(latest["rng"])
        best_state = torch_load(best_path)
        best_acc, best_epoch, wait = latest["best_acc"], latest["best_epoch"], latest["wait"]
        best_diag = best_state["diagnostics"]
        start_epoch = int(latest["epoch"]) + 1
    else:
        best_path.unlink(missing_ok=True)
        latest_path.unlink(missing_ok=True)

    train_start = time.time()
    first_grad = None
    epoch = start_epoch - 1
    for epoch in range(start_epoch, CFG["cv"]["epochs"]):
        state["opt_model"].zero_grad(set_to_none=True)
        state["opt_cam"].zero_grad(set_to_none=True)
        state["opt_dcraa"].zero_grad(set_to_none=True)
        out = forward_group(state, cache, meta, train_mode=True)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"NONFINITE_LOSS {group} fold={fold} epoch={epoch}")
        if epoch == 0:
            alpha_cls = torch.autograd.grad(out["cls"], state["dcraa"].alpha, retain_graph=True, allow_unused=True)[0]
            rel_params = [p for p in state["dcraa"].corr.parameters() if p.requires_grad]
            rel_grads = torch.autograd.grad(out["relation_loss"], rel_params, retain_graph=True, allow_unused=True) if out["relation_loss"].requires_grad else []
            cls_rel_grads = torch.autograd.grad(out["cls"], rel_params, retain_graph=True, allow_unused=True)
            adv_rel_grads = []
            adv_disc_grads = []
            if out["adversary_loss"].requires_grad and out["adversary_active"]:
                adv_rel_grads = torch.autograd.grad(out["adversary_loss"], rel_params, retain_graph=True, allow_unused=True)
                disc_params = list(state["dcraa"].site_conditional.parameters()) if group != "E3" else list(state["dcraa"].site_unconditional.parameters())
                adv_disc_grads = torch.autograd.grad(out["adversary_loss"], disc_params, retain_graph=True, allow_unused=True)
            alpha_edge_cls = torch.autograd.grad(out["cls"], (state["dcraa"].alpha, state["dcraa"].edge_scale), retain_graph=True, allow_unused=True)
            first_grad = {
                "classification_to_alpha": float(alpha_cls.detach().abs().cpu()) if alpha_cls is not None else 0.0,
                "relation_loss_to_encoder": float(torch.stack([g.detach().float().square().sum() for g in rel_grads if g is not None]).sum().sqrt().cpu()) if any(g is not None for g in rel_grads) else 0.0,
                "classification_to_relation_encoder": float(torch.stack([g.detach().float().square().sum() for g in cls_rel_grads if g is not None]).sum().sqrt().cpu()) if any(g is not None for g in cls_rel_grads) else 0.0,
                "adversarial_grl_to_relation_encoder": float(torch.stack([g.detach().float().square().sum() for g in adv_rel_grads if g is not None]).sum().sqrt().cpu()) if any(g is not None for g in adv_rel_grads) else 0.0,
                "adversarial_to_discriminator": float(torch.stack([g.detach().float().square().sum() for g in adv_disc_grads if g is not None]).sum().sqrt().cpu()) if any(g is not None for g in adv_disc_grads) else 0.0,
                "classification_to_edge_scale": float(alpha_edge_cls[1].detach().abs().cpu()) if alpha_edge_cls[1] is not None else 0.0,
                "edge_nonzero": bool(float(out["edge_raw"].detach().abs().sum()) > 0),
                "message_at_alpha_zero": float(out["residual"].detach().norm().cpu()),
            }
        out["total"].backward()
        for mod in (state["model"], state["cam"], state["dcraa"]):
            for param in mod.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"NONFINITE_GRAD {group} fold={fold} epoch={epoch}")
        grad = {
            "gfr": grad_norm_module(state["model"]),
            "cam": grad_norm_module(state["cam"]),
            "relation_encoder": grad_norm_module(state["dcraa"].corr),
            "conditional_discriminator": grad_norm_module(state["dcraa"].site_conditional),
            "unconditional_discriminator": grad_norm_module(state["dcraa"].site_unconditional),
            "alpha": float(state["dcraa"].alpha.grad.detach().abs().cpu()) if state["dcraa"].alpha.grad is not None else 0.0,
            "edge_scale": float(state["dcraa"].edge_scale.grad.detach().abs().cpu()) if state["dcraa"].edge_scale.grad is not None else 0.0,
        }
        state["opt_model"].step()
        state["opt_cam"].step()
        state["opt_dcraa"].step()
        with torch.no_grad():
            ev = forward_group(state, cache, meta, train_mode=False)
            metrics = metrics55(ev["prob"], state)
        if metrics["acc"] > best_acc:
            best_acc, best_epoch, wait = metrics["acc"], epoch, 0
            corr_nodes = state["dcraa"].corr.encode_nodes(state["feat"][:, 64:320])
            pair_z = state["dcraa"].pair_representation(state["feat"][:, 64:320], cache["train_pairs"])
            pair_prob = torch.softmax(state["dcraa"].relation_logits(pair_z), dim=1)
            rel_pred = pair_prob.argmax(1)
            relation_counts = torch.bincount(cache["train_rel"], minlength=3)
            proto = torch.stack([pair_z[cache["train_rel"] == k].mean(0) for k in range(3)])
            proto_dist = torch.cdist(torch.nn.functional.normalize(proto, dim=1), torch.nn.functional.normalize(proto, dim=1))
            best_diag = {
                "train_cls": float(ev["cls"].detach().cpu()),
                "train_graph": float(ev["graph"].detach().cpu()),
                "relation_loss": float(ev["relation_loss"].detach().cpu()),
                "adversary_loss": float(ev["adversary_loss"].detach().cpu()),
                "teacher_loss": float(ev["teacher_loss"].detach().cpu()),
                "auxiliary_loss": float(ev["auxiliary_loss"].detach().cpu()),
                "gradients": grad,
                "first_epoch_gradients": first_grad,
                "grl_encoder_grad_norm": grad["relation_encoder"],
                "relation_train_accuracy": float((rel_pred == cache["train_rel"]).float().mean().cpu()),
                "relation_train_counts": relation_counts.cpu().tolist(),
                "relation_prototype_distance": proto_dist.cpu().tolist(),
                "relation_pair_embedding_variance": float(pair_z.var(dim=0, unbiased=False).mean().cpu()),
                "node_embedding_variance": float(corr_nodes.var(dim=0, unbiased=False).mean().cpu()),
                "gfr_message_norm": float(ev["e1_message"].detach().norm().cpu()),
                "gfr_input_norm": float(state["feat"].norm().cpu()),
                "new_message_norm": float(ev["delta"].detach().norm().cpu()),
                "new_bounded_message_norm": float(ev["bounded_delta"].detach().norm().cpu()),
                "new_residual_norm": float(ev["residual"].detach().norm().cpu()),
                "new_residual_ratio": float((ev["residual"].detach().norm() / ev["fused"].detach().norm().clamp_min(1e-8)).cpu()),
                "alpha": float(state["dcraa"].alpha.detach().cpu()),
                "edge_scale": float(state["dcraa"].edge_scale.detach().cpu()),
                "edge_weight_min": float(ev["edge_weights"].min().detach().cpu()),
                "edge_weight_max": float(ev["edge_weights"].max().detach().cpu()),
                "edge_weight_entropy": float((-(ev["edge_weights"].clamp_min(1e-8) * ev["edge_weights"].clamp_min(1e-8).log()).sum() / len(state["feat"])).detach().cpu()),
                "edge_score_mean": float(ev["edge_disease_score"].mean().detach().cpu()),
                "edge_confidence_mean": float(ev["edge_confidence"].mean().detach().cpu()),
                "adversary_active": ev["adversary_active"],
                "adversary_identifiability": meta["adversary_status"],
                "candidate_edges": int(len(cache["edge_send"])),
                "wrong_edge_changed_fraction": meta["wrong_edge_changed_fraction"],
                "wrong_condition_changed_fraction": meta["wrong_disease_condition_changed_fraction"],
                "wrong_site_changed_fraction": meta["wrong_site_source_changed_fraction"],
                "auxiliary_modalities_admitted": active_modalities(meta),
            }
            checkpoint = {
                "group": group, "fold": fold, "seed": 0,
                "best_epoch": best_epoch, "epochs_executed": epoch + 1,
                "metrics": metrics,
                "model": state_cpu(state["model"]),
                "cam": state_cpu(state["cam"]),
                "dcraa": state_cpu(state["dcraa"]),
                "opt_model": optimizer_cpu(state["opt_model"]),
                "opt_cam": optimizer_cpu(state["opt_cam"]),
                "opt_dcraa": optimizer_cpu(state["opt_dcraa"]),
                "rng": rng_pack(),
                "diagnostics": best_diag,
                "split_sha": split_sha(fold),
                "fold_cache_sha": meta["cache_sha256"],
                "anchor_parameter_sha": state["anchor"]["parameter_sha"],
                "anchor_rng_sha": E1.rng_sha(state["anchor"]["rng"]),
                "source_signatures": source_signatures(),
                "config_signature": config_signature(),
                "dcraa_initial_sha": init_sha,
                "auxiliary_rng": {
                    "relation_pair_seed": 553000 + fold,
                    "condition_mismatch_seed": 553200 + fold,
                    "site_mismatch_seed": 553300 + fold,
                    "edge_swap_seed": 553400 + fold,
                    "edge_condition_seed": 553500 + fold,
                    "torch_rng_isolated": True,
                },
                "optimizer_contract": {
                    "original_model": [{"lr": g["lr"], "weight_decay": g["weight_decay"]} for g in state["opt_model"].param_groups],
                    "original_cam": [{"lr": g["lr"], "weight_decay": g["weight_decay"]} for g in state["opt_cam"].param_groups],
                    "dcraa": [{"lr": g["lr"], "weight_decay": g["weight_decay"]} for g in state["opt_dcraa"].param_groups],
                },
            }
            atomic_torch_save(best_path, checkpoint)
        else:
            wait += 1
        if epoch % 5 == 0:
            atomic_torch_save(latest_path, {
                "epoch": epoch, "wait": wait, "best_acc": best_acc, "best_epoch": best_epoch,
                "model": state_cpu(state["model"]), "cam": state_cpu(state["cam"]),
                "dcraa": state_cpu(state["dcraa"]),
                "opt_model": optimizer_cpu(state["opt_model"]),
                "opt_cam": optimizer_cpu(state["opt_cam"]),
                "opt_dcraa": optimizer_cpu(state["opt_dcraa"]),
                "rng": rng_pack(), "fold_cache_sha": meta["cache_sha256"],
            })
        if wait > CFG["cv"]["early_stop_wait_gt"]:
            break
    if not best_path.exists():
        raise RuntimeError(f"NO_BEST_CHECKPOINT {group} fold={fold}")
    best_state = torch_load(best_path)
    best_state["epochs_executed"] = epoch + 1
    atomic_torch_save(best_path, best_state)
    state["model"].load_state_dict(best_state["model"])
    state["cam"].load_state_dict(best_state["cam"])
    state["dcraa"].load_state_dict(best_state["dcraa"])
    with torch.no_grad():
        verified = forward_group(state, cache, meta, train_mode=False)
        verified_metrics = metrics55(verified["prob"], state)
    if any(abs(float(verified_metrics[k]) - float(best_state["metrics"][k])) > 1e-12 for k in verified_metrics):
        raise RuntimeError(f"BEST_CHECKPOINT_METRICS_MISMATCH {group} fold={fold}")
    row = {
        "group": group, "fold": fold, "seed": 0, "gpu": 0,
        "name": CFG["groups"][int(group[1:])]["name"], "status": "COMPLETE",
        "best_epoch": int(best_state["best_epoch"]), "epochs_executed": int(best_state["epochs_executed"]),
        "wall_seconds": time.time() - train_start,
        "best_sha": sha_file(best_path), "split_sha": split_sha(fold),
        "anchor_parameter_sha": best_state["anchor_parameter_sha"],
        "anchor_rng_sha": best_state["anchor_rng_sha"],
        "fold_cache_sha": meta["cache_sha256"],
        "dcraa_initial_sha": init_sha,
        "rng_isolation_verified": state["rng_isolation_verified"],
        **verified_metrics,
    }
    atomic_json(task, row)
    latest_path.unlink(missing_ok=True)
    print("DONE", group, fold, row["best_epoch"], row["acc"], "residual", best_state["diagnostics"]["new_residual_ratio"], flush=True)


def historical_e0_rows():
    rows = []
    for fold in range(1, 11):
        path = TASKS / f"E0_f{fold:02d}.json"
        if not path.exists():
            raise RuntimeError(f"E0_TASK_MISSING fold={fold}")
        task = json.loads(path.read_text())
        if task["status"] != "COMPLETE":
            raise RuntimeError(f"E0_PARITY_NOT_COMPLETE fold={fold}: {task['status']}")
        rows.append({
            "group": "E0", "fold": fold, "seed": 0, "gpu": "replay",
            "name": CFG["groups"][0]["name"], "status": "COMPLETE",
            "best_epoch": task["best_epoch"], "epochs_executed": task["epochs_executed"],
            "best_sha": task["best_sha"], "split_sha": split_sha(fold),
            "anchor_parameter_sha": task["anchor_parameter_sha"], "anchor_rng_sha": task["anchor_rng_sha"],
            **task["metrics"],
        })
    return rows


def write_initial_manifest():
    import numpy as np

    check_input_signatures()
    x, y = load_data()[0]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "UNAVAILABLE"
    manifest = {
        "experiment": CFG["experiment"],
        "status": "PREPARED",
        "config_sha256": sha_file(CFG_PATH),
        "config_signature": config_signature(),
        "git_commit": commit,
        "protected_input_sha256": check_input_signatures(),
        "source_sha256": source_signatures(),
        "code_sha256": {name: sha_file(EXP / name) for name in ("run.py", "dcraa.py", "tests.py", "design.md") if (EXP / name).exists()},
        "data_shape": list(x.shape),
        "label_counts": np.bincount(y).tolist(),
        "folds": {str(f): split_sha(f) for f in range(1, 11)},
        "plans": {"groups": len(GROUPS), "folds_per_group": 10, "total_group_folds": 110},
        "rng_protocol": {
            "historical_init": "Experiment51 E1 seed and fold anchor through exact copied E51 helper",
            "new_module_init": "torch.random.fork_rng, same fold seed across groups",
            "per_epoch_global_reseed": False,
            "fold_entry_restored": "E51 E1 per-fold seed/anchor via build_state",
            "parity": "PENDING",
        },
        "worker_plan": {"gpu": 0, "max_workers": 5, "cpu_threads_per_worker": 2},
        "task_status": {},
    }
    atomic_json(EXP / "manifest.json", manifest)


def prepare():
    for path in (RT, CKPTS, TASKS, LOGS, FOLD_DATA, REL_CACHE, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
    check_input_signatures()
    load_data()
    write_initial_manifest()
    print("PREPARE_PASS", config_signature(), flush=True)


def controller():
    parity_path = EXP / "parity.json"
    if not parity_path.exists() or json.loads(parity_path.read_text()).get("status") != "PASS":
        raise RuntimeError("E0_PARITY_REQUIRED_BEFORE_CONTROLLER")
    groups = GROUPS[1:]
    tasks = [(group, fold) for group in groups for fold in range(1, 11)]
    pending = []
    attempts = Counter()
    for group, fold in tasks:
        path = TASKS / f"{group}_f{fold:02d}.json"
        if path.exists() and json.loads(path.read_text()).get("status") == "COMPLETE":
            continue
        pending.append((group, fold))
    prepare_all_caches()
    workers = min(5, int(CFG["runtime"]["workers_max"]))
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.update({key: "2" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    active = {}
    started = time.time()
    failures = []
    while pending or active:
        while pending and len(active) < workers:
            group, fold = pending.pop(0)
            attempts[(group, fold)] += 1
            log_path = LOGS / f"{group}_f{fold:02d}.log"
            log_handle = log_path.open("ab")
            process = subprocess.Popen(
                [sys.executable, "-B", str(EXP / "run.py"), "worker", "--group", group, "--fold", str(fold)],
                cwd=EXP, env=env, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True,
            )
            active[(group, fold)] = (process, log_handle)
            print("START", group, fold, "attempt", attempts[(group, fold)], flush=True)
        for key, (process, log_handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log_handle.close()
            del active[key]
            group, fold = key
            task_path = TASKS / f"{group}_f{fold:02d}.json"
            valid = task_path.exists() and json.loads(task_path.read_text()).get("status") == "COMPLETE"
            if code == 0 and valid:
                print("WORKER_DONE", group, fold, flush=True)
                continue
            failure = {"group": group, "fold": fold, "exit_code": code, "attempt": attempts[key], "log": str(LOGS / f"{group}_f{fold:02d}.log")}
            if attempts[key] < 2:
                pending.append(key)
                print("RETRY", group, fold, "exit", code, flush=True)
            else:
                failures.append(failure)
                atomic_json(task_path, {**failure, "status": "FAILED"})
                print("WORKER_FAILED", group, fold, "exit", code, flush=True)
        if active:
            time.sleep(0.5)
    manifest = json.loads((EXP / "manifest.json").read_text())
    complete = sum(1 for g, f in tasks if (TASKS / f"{g}_f{f:02d}.json").exists() and json.loads((TASKS / f"{g}_f{f:02d}.json").read_text()).get("status") == "COMPLETE")
    manifest.update({"status": "TRAINING_COMPLETE" if complete == 100 else "TRAINING_PARTIAL", "completed_new_group_folds": complete, "failed_tasks": failures, "worker_wall_seconds": time.time() - started, "worker_attempts": {f"{g}_f{f:02d}": n for (g, f), n in attempts.items()}})
    atomic_json(EXP / "manifest.json", manifest)
    print("CONTROLLER_DONE", complete, "of 100", flush=True)


def aggregate_training():
    import numpy as np

    rows = historical_e0_rows()
    for group in GROUPS[1:]:
        for fold in range(1, 11):
            task = json.loads((TASKS / f"{group}_f{fold:02d}.json").read_text())
            if task.get("status") != "COMPLETE":
                raise RuntimeError(f"INCOMPLETE_TASK {group} fold={fold}")
            rows.append(task)
    atomic_csv(RESULTS / "fold_metrics.csv", rows)
    summary = []
    for group in GROUPS:
        selected = [r for r in rows if r["group"] == group]
        item = {"group": group, "name": CFG["groups"][int(group[1:])]["name"], "folds": len(selected)}
        for key in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"):
            values = np.asarray([float(r[key]) * 100.0 for r in selected])
            item[key + "_mean_pct"] = float(values.mean())
            item[key + "_sd_pct"] = float(values.std(ddof=0))
        summary.append(item)
    atomic_csv(RESULTS / "summary.csv", summary)
    by_fold = {(r["group"], int(r["fold"])): r for r in rows}
    comparisons = []
    for other in ("E0", "E1", "E2", "E3", "E4", "E6", "E7", "E8", "E9", "E10"):
        for metric in ("acc", "hard_auc", "prob_auc", "spe", "sen"):
            differences = np.asarray([(float(by_fold["E5", f][metric]) - float(by_fold[other, f][metric])) * 100.0 for f in range(1, 11)])
            comparisons.append({
                "comparison": f"E5_vs_{other}", "metric": metric,
                "delta_mean_pp": float(differences.mean()),
                "fold_deltas_pp": json.dumps(differences.tolist()),
                "wins": int((differences > 0).sum()), "ties": int((differences == 0).sum()),
                "losses": int((differences < 0).sum()),
            })
    atomic_csv(RESULTS / "paired_comparisons.csv", comparisons)
    return rows, summary, comparisons


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("e0")
    sub.add_parser("caches")
    sub.add_parser("controller")
    sub.add_parser("aggregate")
    worker = sub.add_parser("worker")
    worker.add_argument("--group", required=True)
    worker.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "e0":
        e0_all()
    elif args.command == "caches":
        if not (EXP / "parity.json").exists() or json.loads((EXP / "parity.json").read_text()).get("status") != "PASS":
            raise RuntimeError("E0_PARITY_REQUIRED_BEFORE_FOLD_CACHE")
        prepare_all_caches()
    elif args.command == "controller":
        controller()
    elif args.command == "aggregate":
        aggregate_training()
    elif args.command == "worker":
        try:
            train_one(args.group, args.fold)
        except Exception as error:
            atomic_json(TASKS / f"{args.group}_f{args.fold:02d}.json", {
                "group": args.group, "fold": args.fold, "status": "FAILED",
                "error": repr(error), "time": time.time(),
            })
            raise


if __name__ == "__main__":
    main()

