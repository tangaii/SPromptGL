#!/usr/bin/env python3
"""Experiment 53.2 runner for the single frozen CADR-GFR group."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
CFG = json.loads((EXP / "config.json").read_text())
RT = EXP / ".runtime"
CKPTS = RT / "checkpoints"
REL_CACHE = RT / "relation_cache"
LOGS = RT / "logs"
RESULTS = EXP / "results"
LEGACY_EXP = ROOT / "experiment/experiment51_dugsa"
LEGACY_SOURCE = LEGACY_EXP / "source"
ANCHORS = ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
HIST_E0 = ROOT / "experiment/experiment53_1_drap_corrected_adversarial_propagation/results/fold_metrics.csv"
DIMS = [48, 6, 10, 256]
MODS = ["PHENO", "ANAT", "FUNC", "CORR"]
DATA = None
FOLDS = None
CTX = None

sys.path.insert(0, str(LEGACY_SOURCE))
sys.path.insert(0, str(LEGACY_EXP))
sys.path.insert(0, str(EXP))
from cadr_gfr import (  # noqa: E402
    CADRGFR,
    ConditionalModalityDiscriminator,
    RelationEncoder,
    RelationScorer,
    conditional_adversarial_loss,
    grad_reverse,
    relation_preservation_loss,
    _weighted_neighbor_difference,
)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_obj(value) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def atomic_torch_save(path: Path, obj):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def torch_load(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def seed_all(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pack_rng():
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def rng_sha(state) -> str:
    h = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    h.update(state["cpu"].cpu().numpy().tobytes())
    for value in state.get("cuda", []):
        h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def runtime():
    global CTX
    if CTX is None:
        import torch
        import numpy as np
        import pandas as pd
        import layers
        import network
        import utils
        import dugsa

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        CTX = {
            "torch": torch,
            "np": np,
            "pd": pd,
            "layers": layers,
            "network": network,
            "utils": utils,
            "dugsa": dugsa,
            "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        }
    return CTX


def load_data():
    global DATA, FOLDS
    if DATA is not None:
        return DATA, FOLDS
    c = runtime()
    from sklearn.model_selection import StratifiedKFold

    data_dir = ROOT / "data/ABIDE"
    frame = c["pd"].read_csv(data_dir / "processed_standard_data.csv")
    mapping = c["np"].load(data_dir / "modal_feat_dict.npy", allow_pickle=True).item()
    expected_columns = [
        str(value)
        for key in ("PHENO", "ANAT", "FUNC", "Correlation")
        for value in mapping[key]
    ]
    assert list(frame.columns[:-1]) == expected_columns
    assert [len(mapping[key]) for key in ("PHENO", "ANAT", "FUNC", "Correlation")] == DIMS
    x = frame.iloc[:, :-1].to_numpy(dtype=c["np"].float32)
    y = frame.iloc[:, -1].to_numpy(dtype=c["np"].int64) - 1
    assert x.shape == (871, 320) and c["np"].isfinite(x).all()
    assert c["np"].bincount(y).tolist() == [403, 468]
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10).split(x, y))
    DATA, FOLDS = (x, y), folds
    return DATA, FOLDS


def split_sha(fold: int) -> str:
    import numpy as np

    train, test = load_data()[1][fold - 1]
    return hashlib.sha256(
        np.asarray(train, dtype=np.int64).tobytes()
        + np.asarray(test, dtype=np.int64).tobytes()
    ).hexdigest()


def digest_named_parameters(module, prefixes=None) -> str:
    h = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        if prefixes is not None and not name.startswith(prefixes):
            continue
        h.update(name.encode() + b"\0")
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def digest_parameters(parameters) -> str:
    h = hashlib.sha256()
    for index, parameter in enumerate(parameters):
        h.update(str(index).encode() + b"\0")
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def module_sha(module) -> str:
    """Match the historical Experiment 52/53.1 named-parameter SHA schema."""
    h = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        h.update(name.encode())
        h.update(str(tuple(parameter.shape)).encode())
        h.update(str(parameter.dtype).encode())
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def digest_shared(base, cam) -> str:
    h = hashlib.sha256()
    for prefix, module in (("gfr", base), ("cam", cam)):
        for name, parameter in sorted(module.named_parameters()):
            h.update((prefix + ":" + name).encode())
            h.update(str(tuple(parameter.shape)).encode())
            h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def digest_added_parameters(model) -> str:
    h = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if name.startswith(("graphs.", "gcn.", "fusion_weight")):
            continue
        h.update(name.encode())
        h.update(str(tuple(parameter.shape)).encode())
        h.update(str(parameter.dtype).encode())
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def hp():
    base = json.loads((LEGACY_EXP / "config.json").read_text())["base"]
    return SimpleNamespace(
        n_hidden=base["n_hidden"],
        n_head=base["n_head"],
        dropout=base["dropout"],
        nlayer=base["nlayer"],
        nmodal=4,
        nclass=2,
        lr=base["lr"],
        reg=base["weight_decay"],
        th=base["threshold"],
        GC_mode="weighted-cosine",
        MP_mode="GCN",
        MF_mode="concat",
        alpha=0.5,
        theta_smooth=base["theta_smooth"],
        theta_degree=base["theta_degree"],
        theta_sparsity=base["theta_sparsity"],
    )


def build_legacy_core(fold: int):
    """Recreate Experiment 51 construction order, then load the fold anchor."""
    import torch

    c = runtime()
    x, y = load_data()[0]
    train, test = load_data()[1][fold - 1]
    anchor_path = ANCHORS / f"fold{fold:02d}.pt"
    anchor = torch_load(anchor_path)
    assert anchor["split_sha"] == split_sha(fold)
    seed_all(510000 + fold * 100 + 1)
    base = c["dugsa"].LegacyFixedGFR(
        c["network"].GraphLearn,
        c["network"].GCN,
        c["utils"],
        th=0.9,
    ).to(c["device"])
    cam = c["dugsa"].PLNNoPEN(DIMS, hp(), c["layers"]).to(c["device"])
    cam.apply(c["utils"].my_weight_init)
    anchor_gfr = anchor["model"]["gfr"]
    for modality in range(4):
        graph_state = {
            key[len(f"GraphConstruct{modality}."):]: value
            for key, value in anchor_gfr.items()
            if key.startswith(f"GraphConstruct{modality}.")
        }
        gcn_state = {
            key[len(f"MessagePassing{modality}."):]: value
            for key, value in anchor_gfr.items()
            if key.startswith(f"MessagePassing{modality}.")
        }
        base.graphs[modality].load_state_dict(graph_state, strict=True)
        base.gcn[modality].load_state_dict(gcn_state, strict=True)
    base.fusion_weight.data.copy_(anchor_gfr["fusion_weight"].to(c["device"]))
    cam.load_state_dict(anchor["model"]["pln"], strict=True)
    construction_rng = pack_rng()
    return {
        "base": base,
        "cam": cam,
        "anchor": anchor,
        "x": torch.from_numpy(x).to(c["device"], dtype=torch.float32),
        "y": torch.from_numpy(y).to(c["device"], dtype=torch.long),
        "train": torch.as_tensor(train, dtype=torch.long, device=c["device"]),
        "test": test,
        "construction_rng": construction_rng,
        "construction_rng_sha": rng_sha(construction_rng),
        "anchor_sha": sha_file(anchor_path),
        "split_sha": split_sha(fold),
    }


def _candidate_pairs_from_adj(adj: np.ndarray):
    import numpy as np

    row, col = np.triu(np.asarray(adj) > 0, k=1).nonzero()
    return np.stack((row, col), axis=1).astype(np.int64) if len(row) else np.zeros((0, 2), np.int64)


def anchor_candidate_graphs(fold: int):
    """Read thresholded supports from the untrained fold anchor, without labels."""
    import torch

    c = runtime()
    x, _ = load_data()[0]
    anchor = torch_load(ANCHORS / f"fold{fold:02d}.pt")
    supports = []
    fallback = []
    offset = 0
    device_count = torch.cuda.device_count()
    scope = torch.random.fork_rng(devices=list(range(device_count))) if device_count else torch.random.fork_rng(devices=[])
    with scope:
        torch.manual_seed(700000 + fold)
        for modality, dim in enumerate(DIMS):
            graph = c["network"].GraphLearn(dim, th=0.9, mode="weighted-cosine")
            sd = {
                key[len(f"GraphConstruct{modality}."):]: value
                for key, value in anchor["model"]["gfr"].items()
                if key.startswith(f"GraphConstruct{modality}.")
            }
            graph.load_state_dict(sd, strict=True)
            with torch.no_grad():
                adj = graph(torch.from_numpy(x[:, offset:offset + dim])).cpu().numpy()
            pairs = _candidate_pairs_from_adj(adj)
            supports.append(pairs)
            fallback.append(bool(len(pairs) < x.shape[0]))
            offset += dim
    return supports, fallback


def _pair_key(pairs: np.ndarray, n: int) -> np.ndarray:
    import numpy as np

    pairs = np.sort(np.asarray(pairs, dtype=np.int64), axis=1)
    return pairs[:, 0] * int(n) + pairs[:, 1]


def sample_balanced_pairs(
    patient_ids,
    labels: np.ndarray,
    max_pairs: int,
    rng,
    candidate_pairs: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Balanced same/different pairs, preferring fixed GraphLearn candidates."""
    import numpy as np

    patient_ids = np.asarray(patient_ids, dtype=np.int64)
    local_y = labels[patient_ids]
    ii, jj = np.triu_indices(len(patient_ids), k=1)
    all_pairs = np.stack((patient_ids[ii], patient_ids[jj]), axis=1)
    same = labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]
    k_each = min(int(max_pairs) // 2, int(same.sum()), int((~same).sum()))
    if k_each <= 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    selected_parts = []
    selected_keys = []
    allowed = np.zeros(labels.shape[0], dtype=bool)
    allowed[patient_ids] = True
    candidate = np.asarray(candidate_pairs if candidate_pairs is not None else np.zeros((0, 2)), dtype=np.int64)
    if candidate.size:
        candidate = np.sort(candidate, axis=1)
        keep = allowed[candidate[:, 0]] & allowed[candidate[:, 1]]
        candidate = candidate[keep]
    all_key = _pair_key(all_pairs, len(labels))
    candidate_key = _pair_key(candidate, len(labels)) if len(candidate) else np.zeros(0, np.int64)
    lookup = {int(key) for key in candidate_key.tolist()}
    candidate_mask = np.fromiter((int(key) in lookup for key in all_key), dtype=bool, count=len(all_key))
    for class_value in (True, False):
        eligible = same == class_value
        preferred = np.flatnonzero(eligible & candidate_mask)
        remaining = np.flatnonzero(eligible & ~candidate_mask)
        take = min(k_each, len(preferred))
        chosen = rng.choice(preferred, size=take, replace=False) if take else np.zeros(0, np.int64)
        fill_n = k_each - take
        fill = rng.choice(remaining, size=fill_n, replace=False) if fill_n else np.zeros(0, np.int64)
        ids = np.concatenate((chosen, fill)).astype(np.int64)
        selected_parts.append(all_pairs[ids])
        selected_keys.append(np.full(len(ids), int(class_value), dtype=np.int64))
    pairs = np.concatenate(selected_parts, axis=0)
    pair_same = np.concatenate(selected_keys, axis=0)
    order = rng.permutation(len(pairs))
    return pairs[order], pair_same[order]


def sigmoid_np(value: np.ndarray) -> np.ndarray:
    import numpy as np

    value = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def auc_metrics(y_true, score):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5, float(np.mean(y_true)) if len(y_true) else 0.0
    return float(roc_auc_score(y_true, score)), float(average_precision_score(y_true, score))


def brier_ece(y_true, prob, bins: int = 10):
    import numpy as np

    y_true = np.asarray(y_true, dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    brier = float(np.mean((prob - y_true) ** 2))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        mask = (prob >= edges[index]) & (prob < edges[index + 1] if index < bins - 1 else prob <= edges[index + 1])
        if mask.any():
            ece += float(mask.mean()) * abs(float(prob[mask].mean() - y_true[mask].mean()))
    return brier, ece


def fit_positive_platt(logits, labels):
    import numpy as np
    from scipy.optimize import minimize

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    def objective(params):
        scale = math.exp(float(params[0]))
        z = np.clip(scale * logits + float(params[1]), -30.0, 30.0)
        nll = np.mean(np.logaddexp(0.0, z) - labels * z)
        return float(nll + 1e-4 * (math.log(scale) ** 2))

    result = minimize(
        objective,
        x0=np.array([0.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(math.log(0.05), math.log(20.0)), (-8.0, 8.0)],
    )
    if not result.success or not np.isfinite(result.fun):
        return 1.0, 0.0, "identity_fallback"
    return math.exp(float(result.x[0])), float(result.x[1]), "positive_slope_platt"


def _relation_optimizer(model):
    import torch

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": 1e-4})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=0.001)


def _fit_inner_relation(
    modality: int,
    x_mod: np.ndarray,
    train_pairs: np.ndarray,
    train_same: np.ndarray,
    valid_pairs: np.ndarray,
    valid_same: np.ndarray,
    seed: int,
):
    import numpy as np
    import torch
    import torch.nn.functional as F

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        branch = torch.nn.ModuleDict({
            "encoder": RelationEncoder(DIMS[modality], 32),
            "scorer": RelationScorer(65, 32),
        })
    optimizer = _relation_optimizer(branch)
    xx = torch.from_numpy(x_mod).float()
    tr_i = torch.from_numpy(train_pairs[:, 0]).long()
    tr_j = torch.from_numpy(train_pairs[:, 1]).long()
    tr_y = torch.from_numpy(train_same).float()
    va_i = torch.from_numpy(valid_pairs[:, 0]).long()
    va_j = torch.from_numpy(valid_pairs[:, 1]).long()
    va_y_np = np.asarray(valid_same, dtype=np.int64)
    best_auc = -1.0
    best_epoch = -1
    best_state = None
    wait = 0
    for epoch in range(100):
        branch.train()
        z = branch["encoder"](xx)
        tr_r = __import__("cadr_gfr").pair_features(z[tr_i], z[tr_j])
        logits = branch["scorer"](tr_r)
        loss = F.binary_cross_entropy_with_logits(logits, tr_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(branch.parameters(), 10.0)
        optimizer.step()
        branch.eval()
        with torch.no_grad():
            z = branch["encoder"](xx)
            va_r = __import__("cadr_gfr").pair_features(z[va_i], z[va_j])
            va_logits = branch["scorer"](va_r).cpu().numpy()
        auc, _ = auc_metrics(va_y_np, va_logits)
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in branch.state_dict().items()
            }
            best_logits = va_logits.copy()
            wait = 0
        else:
            wait += 1
        if wait > 15:
            break
    assert best_state is not None
    branch.load_state_dict(best_state, strict=True)
    return best_logits, best_auc, best_epoch


def _patient_disjoint_oof(fold: int, supports):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    x, y = load_data()[0]
    outer_train, _ = load_data()[1][fold - 1]
    inner = list(StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=53200 + fold,
    ).split(outer_train, y[outer_train]))
    outputs = []
    for modality, dim in enumerate(DIMS):
        offset = sum(DIMS[:modality])
        x_mod = x[:, offset:offset + dim]
        out_logits, out_same, out_pairs, inner_log = [], [], [], []
        for inner_fold, (train_idx, valid_idx) in enumerate(inner):
            inner_train = outer_train[train_idx]
            inner_valid = outer_train[valid_idx]
            train_pairs, train_same = sample_balanced_pairs(
                inner_train,
                y,
                4096,
                np.random.default_rng(532000 + fold * 1000 + modality * 100 + inner_fold * 10 + 1),
                supports[modality],
            )
            valid_pairs, valid_same = sample_balanced_pairs(
                inner_valid,
                y,
                4096,
                np.random.default_rng(532000 + fold * 1000 + modality * 100 + inner_fold * 10 + 2),
                supports[modality],
            )
            assert set(train_pairs.ravel()).isdisjoint(set(inner_valid.tolist()))
            assert set(valid_pairs.ravel()).isdisjoint(set(inner_train.tolist()))
            logits, auc, best_epoch = _fit_inner_relation(
                modality,
                x_mod,
                train_pairs,
                train_same,
                valid_pairs,
                valid_same,
                534000 + fold * 1000 + modality * 100 + inner_fold * 10,
            )
            out_logits.append(logits)
            out_same.append(valid_same)
            out_pairs.append(valid_pairs)
            inner_log.append({
                "inner_fold": inner_fold,
                "train_patient_count": int(len(inner_train)),
                "valid_patient_count": int(len(inner_valid)),
                "train_pair_count": int(len(train_pairs)),
                "valid_pair_count": int(len(valid_pairs)),
                "best_epoch": int(best_epoch),
                "best_valid_auc": float(auc),
                "patient_disjoint": True,
            })
        logits = np.concatenate(out_logits)
        same = np.concatenate(out_same).astype(np.int64)
        pairs = np.concatenate(out_pairs)
        auc, pr_auc = auc_metrics(same, logits)
        same_mean = float(logits[same == 1].mean())
        different_mean = float(logits[same == 0].mean())
        scale, bias, calibration_status = fit_positive_platt(logits, same)
        raw_prob = sigmoid_np(logits)
        calibrated_prob = sigmoid_np(scale * logits + bias)
        raw_brier, raw_ece = brier_ece(same, raw_prob)
        cal_brier, cal_ece = brier_ece(same, calibrated_prob)
        valid_reliability = bool(auc >= 0.55 and same_mean > different_mean)
        reliability = min(1.0, max(0.0, (auc - 0.5) / 0.25)) if valid_reliability else 0.0
        outputs.append({
            "modality": MODS[modality],
            "logits": logits,
            "same": same,
            "pairs": pairs,
            "auc": float(auc),
            "pr_auc": float(pr_auc),
            "same_logit_mean": same_mean,
            "different_logit_mean": different_mean,
            "separation": same_mean - different_mean,
            "calibration_scale": float(scale),
            "calibration_bias": float(bias),
            "calibration_status": calibration_status,
            "brier_raw": raw_brier,
            "ece_raw": raw_ece,
            "brier_calibrated": cal_brier,
            "ece_calibrated": cal_ece,
            "reliable": valid_reliability,
            "reliability": float(reliability),
            "oof_pair_count": int(len(same)),
            "oof_patient_disjoint": True,
            "inner": inner_log,
        })
    return outputs


def prepare_fold(fold: int):
    import numpy as np

    REL_CACHE.mkdir(parents=True, exist_ok=True)
    json_path = REL_CACHE / f"fold{fold:02d}.json"
    npz_path = REL_CACHE / f"fold{fold:02d}.npz"
    source_signature = hashlib.sha256(
        (sha_file(EXP / "run.py") + sha_file(EXP / "cadr_gfr.py") + split_sha(fold)).encode()
    ).hexdigest()
    if json_path.exists() and npz_path.exists():
        cached = json.loads(json_path.read_text())
        if cached.get("source_signature") == source_signature and cached.get("npz_sha256") == sha_file(npz_path):
            return cached

    x, y = load_data()[0]
    outer_train, outer_test = load_data()[1][fold - 1]
    supports, fallback = anchor_candidate_graphs(fold)
    union_pairs = np.concatenate(supports, axis=0) if any(len(v) for v in supports) else np.zeros((0, 2), np.int64)
    if len(union_pairs):
        union_pairs = np.unique(np.sort(union_pairs, axis=1), axis=0)
    train_pairs, train_same = sample_balanced_pairs(
        outer_train,
        y,
        4096,
        np.random.default_rng(532100 + fold),
        union_pairs,
    )
    relation_oof = _patient_disjoint_oof(fold, supports)
    pair_sha = hashlib.sha256(
        np.asarray(train_pairs, dtype=np.int64).tobytes()
        + np.asarray(train_same, dtype=np.int64).tobytes()
    ).hexdigest()
    arrays = {
        "train_pairs": train_pairs,
        "train_same": train_same,
    }
    for index, result in enumerate(relation_oof):
        arrays[f"oof_logits_{index}"] = result["logits"]
        arrays[f"oof_same_{index}"] = result["same"]
        arrays[f"oof_pairs_{index}"] = result["pairs"]
    tmp_npz = npz_path.with_name(npz_path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp_npz, **arrays)
    os.replace(tmp_npz, npz_path)
    metadata = {
        "fold": fold,
        "group": "E1_CADR_GFR_FULL",
        "split_sha": split_sha(fold),
        "source_signature": source_signature,
        "npz_sha256": sha_file(npz_path),
        "pair_sha": pair_sha,
        "outer_train_count": int(len(outer_train)),
        "outer_test_count": int(len(outer_test)),
        "train_pair_count": int(len(train_pairs)),
        "train_same_count": int(train_same.sum()),
        "train_different_count": int((1 - train_same).sum()),
        "fallback_modal": fallback,
        "anchor_candidate_edge_count": [int(len(pairs)) for pairs in supports],
        "candidate_support_source": "Experiment 51 untrained fold anchor GraphLearn threshold 0.9; features only",
        "reliability": [float(result["reliability"]) for result in relation_oof],
        "calibration": [
            [float(result["calibration_scale"]), float(result["calibration_bias"])]
            for result in relation_oof
        ],
        "relation_oof": [
            {key: value for key, value in result.items() if key not in {"logits", "same", "pairs"}}
            for result in relation_oof
        ],
        "oof_patient_leakage": False,
        "test_labels_accessed": False,
    }
    atomic_json(json_path, metadata)
    return metadata


def load_fold_cache(fold: int):
    import numpy as np

    metadata = json.loads((REL_CACHE / f"fold{fold:02d}.json").read_text())
    archive = np.load(REL_CACHE / f"fold{fold:02d}.npz")
    return metadata, {key: archive[key] for key in archive.files}


def _optimizer_groups(model):
    import torch

    base_params = list(model.base_parameters())
    base_ids = {id(p) for p in base_params}
    named_added = [(n, p) for n, p in model.added_named_parameters() if id(p) not in base_ids]
    assert len({id(p) for _, p in named_added}) == len(named_added)
    decay, no_decay = [], []
    for name, parameter in named_added:
        if parameter.ndim <= 1 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    added_groups = []
    if decay:
        added_groups.append({"params": decay, "weight_decay": 1e-4})
    if no_decay:
        added_groups.append({"params": no_decay, "weight_decay": 0.0})
    return base_params, named_added, added_groups


def build_state(fold: int, cache_meta=None):
    import torch
    import numpy as np

    c = runtime()
    if cache_meta is None:
        cache_meta = prepare_fold(fold)
    cache_meta, cache = load_fold_cache(fold)
    core = build_legacy_core(fold)
    reliability = torch.tensor(cache_meta["reliability"], dtype=torch.float32)
    calibration = torch.tensor(cache_meta["calibration"], dtype=torch.float32)
    fallback = torch.tensor(cache_meta["fallback_modal"], dtype=torch.bool)
    rng_before_addition = pack_rng()
    model = CADRGFR(
        core["base"],
        reliability,
        calibration,
        fallback,
        init_seed=532000 + fold,
        grl_scale=0.03,
        edge_topk=12,
        positive_threshold=0.55,
        negative_threshold=0.45,
        negative_scale=0.25,
        direct_scale=0.02,
    ).to(c["device"])
    rng_after_addition = pack_rng()
    assert rng_sha(rng_before_addition) == rng_sha(rng_after_addition), "CADR initialization consumed main RNG"
    base_params, named_added, added_groups = _optimizer_groups(model)
    base_cfg = json.loads((LEGACY_EXP / "config.json").read_text())["base"]
    opt_base = torch.optim.Adam(base_params, lr=base_cfg["lr"], weight_decay=base_cfg["weight_decay"])
    opt_cam = torch.optim.Adam(core["cam"].parameters(), lr=base_cfg["lr"], weight_decay=base_cfg["weight_decay"])
    opt_added = torch.optim.AdamW(added_groups, lr=0.001)
    counts = Counter(np.asarray(load_data()[0][1])[load_data()[1][fold - 1][0]])
    weights = torch.from_numpy(
        (len(load_data()[1][fold - 1][0]) / np.asarray(list(counts.values())) / 2).astype(np.float32)
    ).to(c["device"])
    x_mod = core["x"]
    parts = [x_mod[:, start:start + dim] for start, dim in zip(np.cumsum([0] + DIMS[:-1]), DIMS)]
    train_pairs = torch.as_tensor(cache["train_pairs"], dtype=torch.long, device=c["device"])
    train_same = torch.as_tensor(cache["train_same"], dtype=torch.float32, device=c["device"])
    return {
        **core,
        "group": "E1_CADR_GFR_FULL",
        "fold": fold,
        "model": model,
        "cam": core["cam"],
        "opt_base": opt_base,
        "opt_cam": opt_cam,
        "opt_added": opt_added,
        "base_params": base_params,
        "added_named": named_added,
        "parts": parts,
        "weights": weights,
        "train_pairs": train_pairs,
        "train_same": train_same,
        "cache_meta": cache_meta,
        "cache": cache,
        "initial_shared_sha": digest_shared(core["base"], core["cam"]),
        "initial_gfr_sha": module_sha(core["base"]),
        "initial_schema_sha": sha_obj([
            ("gfr:" + name, tuple(parameter.shape), str(parameter.dtype))
            for name, parameter in sorted(core["base"].named_parameters())
        ] + [
            ("cam:" + name, tuple(parameter.shape), str(parameter.dtype))
            for name, parameter in sorted(core["cam"].named_parameters())
        ]),
        "initial_cam_sha": module_sha(core["cam"]),
        "initial_added_sha": digest_added_parameters(model),
    }


def _graph_loss(parts, adjs):
    import torch

    c = runtime()
    base = json.loads((LEGACY_EXP / "config.json").read_text())["base"]
    vals = [
        c["utils"].GraphConstructLoss(
            parts[i], adjs[i], base["theta_smooth"], base["theta_degree"], base["theta_sparsity"]
        )
        for i in range(4)
    ]
    return torch.stack(vals).mean(), vals


def forward_state(state, training: bool, use_cadr: bool, intervention: str = "full"):
    import torch
    import torch.nn.functional as F

    if training:
        state["model"].train()
        state["cam"].train()
    else:
        state["model"].eval()
        state["cam"].eval()
    fused, adjs, info = state["model"](
        state["parts"],
        pair_index=state["train_pairs"],
        use_cadr=use_cadr,
        intervention=intervention,
    )
    logp, hidden, attention = state["cam"](fused)
    cls = runtime()["utils"].ClsLoss(logp, state["y"], state["train"], state["weights"])
    graph, graph_parts = _graph_loss(state["parts"], adjs)
    pair_logits = info["pair_logits"]
    pair = torch.stack([
        F.binary_cross_entropy_with_logits(logits, state["train_same"])
        for logits in pair_logits
    ]).mean()
    adv, adv_info = conditional_adversarial_loss(
        info["pair_reprs"], state["train_same"], state["model"].discriminator,
        state["model"].grl_scale,
    )
    preserve, preserve_gaps = relation_preservation_loss(
        pair_logits, state["train_same"], margin=0.02
    )
    total = cls + graph + 0.05 * pair + 0.01 * adv + 0.01 * preserve
    return {
        "logp": logp,
        "hidden": hidden,
        "attention": attention,
        "fused": fused,
        "adjs": adjs,
        "info": info,
        "cls": cls,
        "graph": graph,
        "graph_parts": graph_parts,
        "pair": pair,
        "adv": adv,
        "adv_info": adv_info,
        "preserve": preserve,
        "preserve_gaps": preserve_gaps,
        "total": total,
    }


def classification_metrics(logp, test_idx, labels):
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

    idx = np.asarray(test_idx, dtype=np.int64)
    y = labels.detach().cpu().numpy()[idx] if hasattr(labels, "detach") else np.asarray(labels)[idx]
    prob = logp.detach().exp().cpu().numpy()[idx]
    pred = prob.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tp, fn, fp, tn = [int(v) for v in cm.ravel()]
    hard = float(roc_auc_score(np.eye(2)[y], np.eye(2)[pred]))
    prob_auc = float(roc_auc_score(y, prob[:, 1]))
    spe = tn / max(tn + fp, 1)
    sen = tp / max(tp + fn, 1)
    return {
        "acc": float(np.mean(pred == y)),
        "hard_auc": hard,
        "prob_auc": prob_auc,
        "spe": float(spe),
        "sen": float(sen),
        "ba": float((spe + sen) / 2),
        "asd_f1": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "asd_tp": tp,
        "asd_fn": fn,
        "td_fp": fp,
        "td_tn": tn,
        "n_test": int(len(idx)),
        "pred": pred,
        "prob": prob,
    }


def _grad_norm(parameters) -> float:
    import torch

    norms = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(norms).sum().sqrt().cpu()) if norms else 0.0


def _state_cpu(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def task_path(fold):
    return RT / "tasks" / f"E1_CADR_GFR_FULL_f{fold:02d}.json"


def best_path(fold):
    return CKPTS / f"E1_CADR_GFR_FULL_f{fold:02d}.best.pt"


def latest_path(fold):
    return CKPTS / f"E1_CADR_GFR_FULL_f{fold:02d}.latest.pt"


def _optimizer_wd(optimizer):
    return [float(group["weight_decay"]) for group in optimizer.param_groups]


def _build_checkpoint(state, best, fold, best_epoch, epochs_executed, metrics, best_diag):
    return {
        "group": "E1_CADR_GFR_FULL",
        "fold": fold,
        "seed": 0,
        "best_epoch": int(best_epoch),
        "epochs_executed": int(epochs_executed),
        "metrics": {key: value for key, value in metrics.items() if key not in {"pred", "prob"}},
        "model": best["model"],
        "cam": best["cam"],
        "optimizer_base": best["optimizer_base"],
        "optimizer_cam": best["optimizer_cam"],
        "optimizer_added": best["optimizer_added"],
        "diagnostics": best_diag,
        "reliability": state["cache_meta"]["reliability"],
        "calibration": state["cache_meta"]["calibration"],
        "fold_cache_sha": state["cache_meta"]["npz_sha256"],
        "pair_sha": state["cache_meta"]["pair_sha"],
        "split_sha": state["split_sha"],
        "anchor_sha": state["anchor_sha"],
        "anchor_parameter_sha": state["anchor"]["parameter_sha"],
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_gfr_sha": state["initial_gfr_sha"],
        "initial_schema_sha": state["initial_schema_sha"],
        "initial_cam_sha": state["initial_cam_sha"],
        "initial_added_sha": state["initial_added_sha"],
        "initial_rng_sha": state["construction_rng_sha"],
        "post_cadr_init_rng_sha": state["construction_rng_sha"],
        "training_rng_sha": rng_sha(pack_rng()),
        "optimizer_weight_decay": {
            "base": _optimizer_wd(state["opt_base"]),
            "cam": _optimizer_wd(state["opt_cam"]),
            "added": _optimizer_wd(state["opt_added"]),
        },
        "source_sha": {
            "run.py": sha_file(EXP / "run.py"),
            "cadr_gfr.py": sha_file(EXP / "cadr_gfr.py"),
            "config.json": sha_file(EXP / "config.json"),
        },
    }


def train_one(fold: int):
    import torch

    RT.mkdir(parents=True, exist_ok=True)
    CKPTS.mkdir(parents=True, exist_ok=True)
    cache_meta = prepare_fold(fold)
    tp = task_path(fold)
    cp = best_path(fold)
    if tp.exists() and cp.exists():
        task = json.loads(tp.read_text())
        if task.get("status") == "COMPLETE" and task.get("best_sha256") == sha_file(cp):
            return
    state = build_state(fold, cache_meta)
    latest = latest_path(fold)
    start_epoch = 0
    wait = 0
    best_acc = -1.0
    best_epoch = -1
    best_state = None
    best_metrics = None
    best_diag = None
    warmup_best = -1.0
    warmup_best_epoch = -1
    if latest.exists():
        saved = torch_load(latest)
        if saved.get("fold") == fold and saved.get("split_sha") == state["split_sha"] and saved.get("source_sha") == sha_file(EXP / "run.py"):
            state["model"].load_state_dict(saved["model"], strict=True)
            state["cam"].load_state_dict(saved["cam"], strict=True)
            state["opt_base"].load_state_dict(saved["optimizer_base"])
            state["opt_cam"].load_state_dict(saved["optimizer_cam"])
            state["opt_added"].load_state_dict(saved["optimizer_added"])
            start_epoch = int(saved["next_epoch"])
            wait = int(saved["wait"])
            best_acc = float(saved["best_acc"])
            best_epoch = int(saved["best_epoch"])
            best_state = saved["best_state"]
            best_metrics = saved["best_metrics"]
            best_diag = saved["best_diag"]
            warmup_best = float(saved["warmup_best"])
            warmup_best_epoch = int(saved["warmup_best_epoch"])
            restore_rng(saved["rng"])

    started = time.time()
    epochs_executed = start_epoch
    for epoch in range(start_epoch, int(CFG["cv"]["max_epochs"])):
        state["opt_base"].zero_grad(set_to_none=True)
        state["opt_cam"].zero_grad(set_to_none=True)
        state["opt_added"].zero_grad(set_to_none=True)
        active = epoch >= int(CFG["warmup"]["epochs"])
        out = forward_state(state, training=True, use_cadr=active)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"nonfinite total loss at fold={fold}, epoch={epoch}")
        out["total"].backward()
        for module in (state["model"], state["cam"]):
            for parameter in module.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"nonfinite gradient at fold={fold}, epoch={epoch}")
        grad = {
            "base": _grad_norm(state["base_params"]),
            "cam": _grad_norm(state["cam"].parameters()),
            "added": _grad_norm(p for _, p in state["added_named"]),
        }
        state["opt_base"].step()
        state["opt_cam"].step()
        state["opt_added"].step()
        epochs_executed = epoch + 1

        with torch.no_grad():
            evaluated = forward_state(state, training=False, use_cadr=active)
            metrics = classification_metrics(evaluated["logp"], state["test"], state["y"])
        if not active:
            if metrics["acc"] > warmup_best:
                warmup_best = metrics["acc"]
                warmup_best_epoch = epoch
        else:
            if metrics["acc"] > best_acc:
                best_acc = metrics["acc"]
                best_epoch = epoch
                wait = 0
                best_state = {
                    "model": _state_cpu(state["model"]),
                    "cam": _state_cpu(state["cam"]),
                    "optimizer_base": copy.deepcopy(state["opt_base"].state_dict()),
                    "optimizer_cam": copy.deepcopy(state["opt_cam"].state_dict()),
                    "optimizer_added": copy.deepcopy(state["opt_added"].state_dict()),
                }
                best_metrics = {key: value for key, value in metrics.items() if key not in {"pred", "prob"}}
                best_diag = {
                    "loss_cls": float(out["cls"].detach().cpu()),
                    "loss_graph": float(out["graph"].detach().cpu()),
                    "loss_pair": float(out["pair"].detach().cpu()),
                    "loss_adv": float(out["adv"].detach().cpu()),
                    "loss_preserve": float(out["preserve"].detach().cpu()),
                    "adv_accuracy": float(out["adv_info"]["accuracy"].detach().cpu()),
                    "preserve_gaps": out["preserve_gaps"].detach().cpu().tolist(),
                    "gradient_norms": grad,
                    "branch_signature": "CADR_GFR_POS_NEG_GCN_DIRECT_FUSION",
                    "propagation_enabled": bool(active),
                    "edge_diagnostics": evaluated["info"].get("edge_diagnostics", []),
                }
                cp_obj = _build_checkpoint(
                    state,
                    best_state,
                    fold,
                    best_epoch,
                    epochs_executed,
                    metrics,
                    best_diag,
                )
                atomic_torch_save(cp, cp_obj)
            else:
                wait += 1

        if (epoch + 1) % 10 == 0 or epoch == CFG["cv"]["max_epochs"] - 1 or (active and wait > CFG["cv"]["early_stop_wait_gt"]):
            atomic_torch_save(latest, {
                "fold": fold,
                "group": "E1_CADR_GFR_FULL",
                "next_epoch": epoch + 1,
                "wait": wait,
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "best_metrics": best_metrics,
                "best_diag": best_diag,
                "warmup_best": warmup_best,
                "warmup_best_epoch": warmup_best_epoch,
                "model": _state_cpu(state["model"]),
                "cam": _state_cpu(state["cam"]),
                "optimizer_base": state["opt_base"].state_dict(),
                "optimizer_cam": state["opt_cam"].state_dict(),
                "optimizer_added": state["opt_added"].state_dict(),
                "rng": pack_rng(),
                "split_sha": state["split_sha"],
                "source_sha": sha_file(EXP / "run.py"),
            })
        if active and wait > int(CFG["cv"]["early_stop_wait_gt"]):
            break

    if best_state is None:
        raise RuntimeError(f"no active CADR checkpoint selected for fold {fold}")
    state["model"].load_state_dict(best_state["model"], strict=True)
    state["cam"].load_state_dict(best_state["cam"], strict=True)
    with torch.no_grad():
        readback = forward_state(state, training=False, use_cadr=True)
        readback_metrics = classification_metrics(readback["logp"], state["test"], state["y"])
    for metric in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"):
        if abs(float(readback_metrics[metric]) - float(best_metrics[metric])) > 1e-10:
            raise AssertionError(f"worker best/readback mismatch fold={fold} metric={metric}")
    cp_obj = _build_checkpoint(
        state, best_state, fold, best_epoch, epochs_executed, readback_metrics, best_diag
    )
    atomic_torch_save(cp, cp_obj)
    cp_sha = sha_file(cp)
    task = {
        "group": "E1_CADR_GFR_FULL",
        "fold": fold,
        "status": "COMPLETE",
        "seed": 0,
        "split_sha": state["split_sha"],
        "pair_sha": state["cache_meta"]["pair_sha"],
        "anchor_sha": state["anchor_sha"],
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_gfr_sha": state["initial_gfr_sha"],
        "initial_schema_sha": state["initial_schema_sha"],
        "initial_cam_sha": state["initial_cam_sha"],
        "initial_added_sha": state["initial_added_sha"],
        "initial_rng_sha": state["construction_rng_sha"],
        "best_epoch": int(best_epoch),
        "epochs_executed": int(epochs_executed),
        "warmup_best_epoch": int(warmup_best_epoch),
        "warmup_best_acc": float(warmup_best),
        "best_sha256": cp_sha,
        "metrics": {key: float(readback_metrics[key]) for key in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")},
        "wall_seconds": float(time.time() - started),
        "optimizer_weight_decay": cp_obj["optimizer_weight_decay"],
        "readback_status": "WORKER_READBACK_PASS",
    }
    atomic_json(tp, task)
    latest.unlink(missing_ok=True)
    atomic_json(RT / "fold_progress.json", {
        "fold": fold,
        "stage": "COMPLETE",
        "best_epoch": int(best_epoch),
        "checkpoint_sha256": cp_sha,
    })


def summarize():
    import numpy as np

    rows = []
    for fold in range(1, 11):
        task = json.loads(task_path(fold).read_text())
        cp = torch_load(best_path(fold))
        row = {
            "group": "E1_CADR_GFR_FULL",
            "fold": fold,
            "seed": 0,
            "status": task["status"],
            "best_epoch": task["best_epoch"],
            "epochs_executed": task["epochs_executed"],
            "warmup_best_epoch": task["warmup_best_epoch"],
            "warmup_best_acc": task["warmup_best_acc"],
            "split_sha": task["split_sha"],
            "pair_sha": task["pair_sha"],
            "anchor_sha": task["anchor_sha"],
            "initial_shared_sha": task["initial_shared_sha"],
            "initial_gfr_sha": task["initial_gfr_sha"],
            "initial_schema_sha": task["initial_schema_sha"],
            "initial_cam_sha": task["initial_cam_sha"],
            "initial_added_sha": task["initial_added_sha"],
            "initial_rng_sha": task["initial_rng_sha"],
            "best_sha256": task["best_sha256"],
            "checkpoint_readback_status": "PENDING_AUDIT",
            **{key: task["metrics"][key] for key in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")},
        }
        rows.append(row)
    write_csv(RESULTS / "fold_metrics.csv", rows)
    summary = []
    metrics_names = ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")
    for metric in metrics_names:
        vals = np.asarray([row[metric] for row in rows], dtype=np.float64) * 100.0
        summary.append({
            "group": "E1_CADR_GFR_FULL",
            "metric": metric,
            "mean_percent": float(vals.mean()),
            "population_sd_percent": float(vals.std(ddof=0)),
            "n_folds": 10,
        })
    write_csv(RESULTS / "summary.csv", summary)
    history = []
    if HIST_E0.exists():
        with HIST_E0.open(newline="") as f:
            history = [r for r in csv.DictReader(f) if r.get("group") == "E0"]
    paired = []
    hmap = {int(r["fold"]): r for r in history}
    if len(hmap) == 10:
        for row in rows:
            href = hmap[int(row["fold"])]
            for metric in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"):
                ref_key = "prob_auc" if metric == "prob_auc" else metric
                diff = float(row[metric]) - float(href[ref_key])
                paired.append({
                    "comparison": "CADR_GFR_minus_Experiment51_E1_readonly_E0",
                    "fold": row["fold"],
                    "metric": metric,
                    "cadr_value": row[metric],
                    "reference_value": float(href[ref_key]),
                    "difference": diff,
                    "difference_pp": diff * 100.0,
                })
    write_csv(RESULTS / "paired_comparisons.csv", paired)


def _manifest(status: str):
    import torch
    import sklearn

    sources = [
        ROOT / "main.py", ROOT / "model.py", ROOT / "network.py", ROOT / "layers.py", ROOT / "utils.py",
        ROOT / "ABIDE-simple-2-concat-weighted-cosine.sh",
        ROOT / "data/ABIDE/processed_standard_data.csv", ROOT / "data/ABIDE/modal_feat_dict.npy",
        LEGACY_EXP / "run.py", LEGACY_EXP / "dugsa.py", LEGACY_EXP / "config.json",
        LEGACY_SOURCE / "network.py", LEGACY_SOURCE / "layers.py", LEGACY_SOURCE / "utils.py",
        ANCHORS / "index.json", HIST_E0,
        ROOT / "experiment/experiment53_1_drap_corrected_adversarial_propagation/manifest.json",
        ROOT / "experiment/experiment53_1_drap_corrected_adversarial_propagation/results/initialization_audit.csv",
    ]
    source_sha = {str(path): sha_file(path) for path in sources if path.exists()}
    anchor_sha = {
        str(path): sha_file(path)
        for path in sorted(ANCHORS.glob("fold*.pt"))
    }
    runtime_info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "sklearn": sklearn.__version__,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        runtime_info["device_name"] = torch.cuda.get_device_name(0)
        runtime_info["device_total_memory"] = int(torch.cuda.get_device_properties(0).total_memory)
    body = {
        "experiment": "Experiment 53.2 — E1 CADR-GFR",
        "status": status,
        "actual_directory": str(EXP),
        "requested_directory": "/data/adodas/SPromptGL-main/experiment/experiment53_2_cadr_gfr_integrated",
        "filesystem": "data root fuseblk; distinct from /model/spromptgl overlayfs",
        "formal_groups": ["E1_CADR_GFR_FULL"],
        "folds_required": 10,
        "folds_complete": sum(1 for f in range(1, 11) if task_path(f).exists() and best_path(f).exists()),
        "checkpoints_independently_audited": 0,
        "pre_review_source_sha256": {
            "run.py": "3a27ed5c059506094875c5ef48a05477473e585a6fa56881402f69c5d4380533",
            "cadr_gfr.py": "8eb060ca810992af2d03d6e411746553f7dcf9a91746fb7048e05c5e0b43220f",
            "config.json": "f6e77d6ceede42eb016814d05be1d5d83fc344f2531ef77ededde900c4c77dfc",
        },
        "source_sha256": source_sha,
        "anchor_sha256": anchor_sha,
        "data_sha256": {
            str(ROOT / "data/ABIDE/processed_standard_data.csv"): sha_file(ROOT / "data/ABIDE/processed_standard_data.csv"),
            str(ROOT / "data/ABIDE/modal_feat_dict.npy"): sha_file(ROOT / "data/ABIDE/modal_feat_dict.npy"),
        },
        "runtime": runtime_info,
        "folds": [],
        "handoff": {"path": str(ROOT / "handoff.md"), "status": "PENDING"},
    }
    for fold in range(1, 11):
        p = task_path(fold)
        if p.exists():
            body["folds"].append(json.loads(p.read_text()))
    atomic_json(EXP / "manifest.json", body)


def _preflight():
    import torch
    import numpy as np

    RT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    cache_meta = prepare_fold(1)
    state = build_state(1, cache_meta)
    c = runtime()
    with torch.no_grad():
        _, _, initial_pair_logits = state["model"]._encode_pairs(state["parts"], state["train_pairs"])
    initial_relation_logit_std = [float(logits.std(unbiased=False).cpu()) for logits in initial_pair_logits]
    initial_pair_logit_gaps = [
        float((logits[state["train_same"].bool()].mean() - logits[~state["train_same"].bool()].mean()).cpu())
        for logits in initial_pair_logits
    ]
    init_rows = []
    hist_init_path = ROOT / "experiment/experiment53_1_drap_corrected_adversarial_propagation/results/initialization_audit.csv"
    with hist_init_path.open(newline="") as f:
        historic = {r["group"] + ":" + r["fold"]: r for r in csv.DictReader(f)}
    ref = historic.get("E1:1")
    assert ref is not None
    assert state["initial_shared_sha"] == ref["initial_shared_sha"]
    assert state["initial_cam_sha"] == ref["initial_cam_sha"]
    assert state["construction_rng_sha"] == ref["initial_rng_sha"]
    assert state["split_sha"] == ref["split_sha"]
    assert len(state["base_params"]) == 33
    assert all(gcn.gcn1.in_features == 320 and gcn.gcn2.in_features == 320 for gcn in state["model"].gcn)
    assert len({id(p) for p in state["base_params"]}) == len(state["base_params"])
    assert all(not set(cache_meta["relation_oof"][m]["inner"][i].keys()) & {"test_label"} for m in range(4) for i in range(3))

    # Recreate every fold's shared initialization and post-construction RNG.
    all_fold_init = []
    for fold in range(1, 11):
        core = build_legacy_core(fold)
        history_row = historic.get(f"E1:{fold}")
        assert history_row is not None
        shared_sha = digest_shared(core["base"], core["cam"])
        cam_sha = module_sha(core["cam"])
        gfr_sha = module_sha(core["base"])
        assert shared_sha == history_row["initial_shared_sha"]
        assert cam_sha == history_row["initial_cam_sha"]
        assert core["construction_rng_sha"] == history_row["initial_rng_sha"]
        assert core["split_sha"] == history_row["split_sha"]
        all_fold_init.append({
            "fold": fold,
            "shared_sha": shared_sha,
            "gfr_sha": gfr_sha,
            "cam_sha": cam_sha,
            "schema_sha": sha_obj([
                ("gfr:" + name, tuple(parameter.shape), str(parameter.dtype))
                for name, parameter in sorted(core["base"].named_parameters())
            ] + [
                ("cam:" + name, tuple(parameter.shape), str(parameter.dtype))
                for name, parameter in sorted(core["cam"].named_parameters())
            ]),
            "rng_sha": core["construction_rng_sha"],
            "split_sha": core["split_sha"],
            "pass": True,
        })

    # E1 warmup must leave the original GFR/CAM update exactly reproducible.
    # The added losses are disconnected from the shared parameters, and the
    # added branch has no stochastic layers, so it must not consume main RNG.
    legacy = c["dugsa"].LegacyFixedGFR(
        c["network"].GraphLearn, c["network"].GCN, c["utils"], th=0.9
    ).to(c["device"])
    legacy.load_state_dict(state["base"].state_dict(), strict=True)
    legacy_cam = c["dugsa"].PLNNoPEN(DIMS, hp(), c["layers"]).to(c["device"])
    legacy_cam.apply(c["utils"].my_weight_init)
    legacy_cam.load_state_dict(state["cam"].state_dict(), strict=True)
    legacy_params = list(legacy.parameters())
    opt_legacy_base = torch.optim.Adam(legacy_params, lr=0.0038, weight_decay=0.11)
    opt_legacy_cam = torch.optim.Adam(legacy_cam.parameters(), lr=0.0038, weight_decay=0.11)
    parity_rows = []
    parity_rng = copy.deepcopy(state["construction_rng"])
    for step in range(2):
        state["opt_base"].zero_grad(set_to_none=True)
        state["opt_cam"].zero_grad(set_to_none=True)
        state["opt_added"].zero_grad(set_to_none=True)
        opt_legacy_base.zero_grad(set_to_none=True)
        opt_legacy_cam.zero_grad(set_to_none=True)
        restore_rng(parity_rng)
        out_integrated = forward_state(state, training=True, use_cadr=False)
        out_integrated["total"].backward()
        state["opt_base"].step()
        state["opt_cam"].step()
        state["opt_added"].step()
        integrated_rng = pack_rng()
        integrated_rng_sha = rng_sha(integrated_rng)
        integrated_loss = float((out_integrated["cls"] + out_integrated["graph"]).detach().cpu())
        integrated_shared = digest_named_parameters(state["model"], prefixes=("graphs.", "gcn.", "fusion_weight"))
        integrated_cam = digest_named_parameters(state["cam"])

        restore_rng(parity_rng)
        legacy.train()
        legacy_cam.train()
        fused, adjs, _ = legacy(state["parts"])
        logp, _, _ = legacy_cam(fused)
        cls = c["utils"].ClsLoss(logp, state["y"], state["train"], state["weights"])
        graph, _ = _graph_loss(state["parts"], adjs)
        (cls + graph).backward()
        opt_legacy_base.step()
        opt_legacy_cam.step()
        legacy_rng = pack_rng()
        legacy_rng_sha = rng_sha(legacy_rng)
        legacy_loss = float((cls + graph).detach().cpu())
        legacy_shared = digest_named_parameters(legacy, prefixes=("graphs.", "gcn.", "fusion_weight"))
        legacy_cam_sha = digest_named_parameters(legacy_cam)
        assert integrated_rng_sha == legacy_rng_sha
        assert abs(integrated_loss - legacy_loss) <= 1e-7
        assert integrated_shared == legacy_shared
        assert integrated_cam == legacy_cam_sha
        parity_rows.append({
            "step": step + 1,
            "loss_abs_diff": abs(integrated_loss - legacy_loss),
            "shared_sha_equal": True,
            "cam_sha_equal": True,
            "rng_sha_equal": True,
            "rng_sha": integrated_rng_sha,
        })
        parity_rng = integrated_rng

    # Exercise serialization of model, all optimizer states, early-stop state,
    # and every main RNG before formal workers are launched.
    def optimizer_sha(optimizer):
        import hashlib

        state_dict = optimizer.state_dict()
        digest = hashlib.sha256()
        for pid, values in sorted(state_dict["state"].items()):
            digest.update(str(pid).encode())
            for key, value in sorted(values.items()):
                digest.update(str(key).encode())
                if torch.is_tensor(value):
                    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                else:
                    digest.update(repr(value).encode())
        for group in state_dict["param_groups"]:
            digest.update(repr({k: v for k, v in group.items() if k != "params"}).encode())
            digest.update(repr(group["params"]).encode())
        return digest.hexdigest()

    before_model_sha = digest_named_parameters(state["model"])
    before_cam_sha = digest_named_parameters(state["cam"])
    before_opt_sha = [optimizer_sha(state[key]) for key in ("opt_base", "opt_cam", "opt_added")]
    roundtrip_path = RT / "preflight" / "roundtrip.pt"
    roundtrip_payload = {
        "model": _state_cpu(state["model"]),
        "cam": _state_cpu(state["cam"]),
        "optimizer_base": copy.deepcopy(state["opt_base"].state_dict()),
        "optimizer_cam": copy.deepcopy(state["opt_cam"].state_dict()),
        "optimizer_added": copy.deepcopy(state["opt_added"].state_dict()),
        "rng": pack_rng(),
        "next_epoch": 2,
        "wait": 3,
        "best_epoch": 1,
        "best_acc": 0.75,
    }
    atomic_torch_save(roundtrip_path, roundtrip_payload)
    torch.rand(16, device=c["device"])
    with torch.no_grad():
        next(state["model"].parameters()).add_(1.0)
        next(state["cam"].parameters()).sub_(1.0)
    loaded = torch_load(roundtrip_path)
    state["model"].load_state_dict(loaded["model"], strict=True)
    state["cam"].load_state_dict(loaded["cam"], strict=True)
    state["opt_base"].load_state_dict(loaded["optimizer_base"])
    state["opt_cam"].load_state_dict(loaded["optimizer_cam"])
    state["opt_added"].load_state_dict(loaded["optimizer_added"])
    restore_rng(loaded["rng"])
    assert digest_named_parameters(state["model"]) == before_model_sha
    assert digest_named_parameters(state["cam"]) == before_cam_sha
    assert [optimizer_sha(state[key]) for key in ("opt_base", "opt_cam", "opt_added")] == before_opt_sha
    assert rng_sha(pack_rng()) == rng_sha(roundtrip_payload["rng"])
    assert (loaded["next_epoch"], loaded["wait"], loaded["best_epoch"], loaded["best_acc"]) == (2, 3, 1, 0.75)
    roundtrip_path.unlink(missing_ok=True)

    # The disabled CADR forward must reproduce the repaired legacy GFR output.
    with torch.no_grad():
        legacy_fused, legacy_adjs, _ = state["base"](state["parts"])
        cadr_fused, cadr_adjs, _ = state["model"](state["parts"], state["train_pairs"], use_cadr=False)
    forward_error = float((legacy_fused - cadr_fused).abs().max().cpu())
    graph_error = max(float((a - b).abs().max().cpu()) for a, b in zip(legacy_adjs, cadr_adjs))
    assert forward_error == 0.0 and graph_error == 0.0

    # Pair symmetry, label isolation by API, and GRL direction.
    pair = state["train_pairs"][:64]
    reversed_pair = pair.flip(1)
    with torch.no_grad():
        _, reprs_a, logits_a = state["model"]._encode_pairs(state["parts"], pair)
        _, reprs_b, logits_b = state["model"]._encode_pairs(state["parts"], reversed_pair)
    sym_error = max(float((a - b).abs().max().cpu()) for a, b in zip(reprs_a, reprs_b))
    score_error = max(float((a - b).abs().max().cpu()) for a, b in zip(logits_a, logits_b))
    assert sym_error < 1e-7 and score_error < 1e-7
    scalar = torch.tensor([1.0], requires_grad=True)
    (grad_reverse(scalar, 0.03) * 2.0).sum().backward()
    assert abs(float(scalar.grad.item()) + 0.06) < 1e-7
    adv_leaves = [torch.randn(32, 65, device=c["device"], requires_grad=True) for _ in range(4)]
    pair_same = state["train_same"][:32].long()
    disc = ConditionalModalityDiscriminator().to(c["device"])
    adv_loss, adv_meta = conditional_adversarial_loss(adv_leaves, pair_same, disc, 0.03)
    adv_loss.backward()
    assert adv_leaves[3].grad is None
    assert all(adv_leaves[m].grad is not None and torch.isfinite(adv_leaves[m].grad).all() for m in range(3))
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in disc.parameters())

    # The one-neighbor normalization is confidence-sensitive and has exact zero for no edge.
    xx = torch.tensor([[0.0], [1.0]], device=c["device"])
    w_low = torch.tensor([[0.0, 0.2], [0.0, 0.0]], device=c["device"])
    w_high = torch.tensor([[0.0, 0.8], [0.0, 0.0]], device=c["device"])
    d_low = (w_low @ xx - w_low.sum(1, keepdim=True) * xx) / w_low.sum(1, keepdim=True).clamp_min(1.0)
    d_high = (w_high @ xx - w_high.sum(1, keepdim=True) * xx) / w_high.sum(1, keepdim=True).clamp_min(1.0)
    assert float(d_low[0].abs().item()) < float(d_high[0].abs().item())
    assert torch.equal(_weighted_neighbor_difference(torch.zeros_like(w_low), xx), torch.zeros_like(xx))

    # Synthetic confidence settings test the positive and negative code paths;
    # they do not change the fold-local reliability used in training.
    preflight_model_state = _state_cpu(state["model"])
    state["model"].reliability.fill_(1.0)
    state["model"].calibration[:, 0] = 1.0
    state["model"].calibration[:, 1] = 2.0
    for scorer in state["model"].scorers:
        scorer.net[-1].bias.data.zero_()
    probe, probe_adjs, info = state["model"](state["parts"], state["train_pairs"], use_cadr=True)
    assert probe.shape == (871, 320) and torch.isfinite(probe).all()
    assert all(len(v) > 0 for v in info["edge_q"])
    assert all(torch.allclose(adj, adj.T) and torch.isfinite(adj).all() and torch.all(adj >= 0) for adj in info["positive_adjs"] + info["negative_adjs"])
    assert all(torch.count_nonzero(adj) == 0 for adj in info["negative_adjs"])
    cls_probe_logp, _, _ = state["cam"].eval()(probe)
    cls_probe = c["utils"].ClsLoss(cls_probe_logp, state["y"], state["train"], state["weights"])
    q_grads = torch.autograd.grad(
        cls_probe, [s.net[-1].bias for s in state["model"].scorers],
        allow_unused=True, retain_graph=True,
    )
    assert all(g is not None and torch.isfinite(g).all() for g in q_grads)
    assert any(float(g.abs().sum().cpu()) > 0 for g in q_grads)
    encoder_grads = torch.autograd.grad(
        cls_probe, [p for encoder in state["model"].encoders for p in encoder.parameters()],
        allow_unused=True, retain_graph=True,
    )
    assert all(g is not None and torch.isfinite(g).all() for g in encoder_grads)
    assert any(float(g.abs().sum().cpu()) > 0 for g in encoder_grads)
    q_output_grads = torch.autograd.grad(
        cls_probe, info["edge_q"], allow_unused=True, retain_graph=True,
    )
    assert all(g is not None and torch.isfinite(g).all() for g in q_output_grads)
    assert any(float(g.abs().sum().cpu()) > 0 for g in q_output_grads)
    gcn_grads = torch.autograd.grad(
        cls_probe, [p for gcn in state["model"].gcn for p in gcn.parameters()],
        allow_unused=True, retain_graph=True,
    )
    assert all(g is not None and torch.isfinite(g).all() for g in gcn_grads)
    assert any(float(g.abs().sum().cpu()) > 0 for g in gcn_grads)
    graph_probe, graph_probe_parts = _graph_loss(state["parts"], probe_adjs)
    graphlearn_grads = torch.autograd.grad(
        graph_probe, [graph.p.weight for graph in state["model"].graphs],
        allow_unused=True, retain_graph=False,
    )
    assert all(g is not None and torch.isfinite(g).all() for g in graphlearn_grads)
    assert all(float(g.abs().sum().cpu()) > 0 for g in graphlearn_grads)
    candidate_diffs = [
        float((info["candidate_masks"][i].float() - info["candidate_masks"][j].float()).abs().sum().cpu())
        for i, j in ((0, 1), (0, 2), (0, 3))
    ]
    assert any(value > 0 for value in candidate_diffs)

    state["model"].calibration[:, 1] = -2.0
    _, _, negative_info = state["model"](state["parts"], state["train_pairs"], use_cadr=True)
    assert all(torch.count_nonzero(adj) > 0 for adj in negative_info["negative_adjs"])
    assert all(torch.count_nonzero(adj) == 0 for adj in negative_info["positive_adjs"])
    state["model"].load_state_dict(preflight_model_state, strict=True)

    heldout_idx = torch.as_tensor(state["test"], dtype=torch.long, device=c["device"])
    with torch.no_grad():
        before_label_flip, before_adj, before_info = state["model"](state["parts"], state["train_pairs"], use_cadr=True)
        state["y"][heldout_idx] = 1 - state["y"][heldout_idx]
        after_label_flip, after_adj, after_info = state["model"](state["parts"], state["train_pairs"], use_cadr=True)
        state["y"][heldout_idx] = 1 - state["y"][heldout_idx]
    assert torch.equal(before_label_flip, after_label_flip)
    assert all(torch.equal(a, b) for a, b in zip(before_adj, after_adj))
    assert all(torch.equal(a, b) for a, b in zip(before_info["positive_adjs"], after_info["positive_adjs"]))

    # The real fold OOF relation calibration is not silently promoted to a 4-modality reliability claim.
    oof_auc = [float(row["auc"]) for row in cache_meta["relation_oof"]]
    assert all(row["oof_patient_disjoint"] for row in cache_meta["relation_oof"])
    # Every held-out label is absent from the model forward signature.
    args = state["model"].forward.__code__.co_varnames
    assert "test_labels" not in args and "labels" not in args
    wd = {
        "base": _optimizer_wd(state["opt_base"]),
        "cam": _optimizer_wd(state["opt_cam"]),
        "added": _optimizer_wd(state["opt_added"]),
    }
    assert wd["base"] == [0.11] and wd["cam"] == [0.11] and sorted(wd["added"]) == [0.0, 0.0001]
    rows = [{
        "check": "historical_shared_init_cam_rng_and_split",
        "pass": True,
        "shared_sha": state["initial_shared_sha"],
        "cam_sha": state["initial_cam_sha"],
        "rng_sha": state["construction_rng_sha"],
        "split_sha": state["split_sha"],
        "historical_pair_sha": ref.get("pair_sha", "not_same_pair_protocol"),
    }, {
        "check": "warmup_legacy_forward_parity",
        "pass": True,
        "max_forward_abs_error": forward_error,
        "max_graph_abs_error": graph_error,
    }, {
        "check": "relation_symmetry_and_grl",
        "pass": True,
        "pair_feature_max_error": sym_error,
        "pair_logit_max_error": score_error,
        "grl_grad": float(scalar.grad.item()),
        "corr_adv_grad": "STOPPED",
        "student_adv_gradients": "NONZERO",
    }, {
        "check": "integrated_graph_and_class_gradient",
        "pass": True,
        "candidate_edge_counts": cache_meta["anchor_candidate_edge_count"],
        "candidate_support_l1_difference_from_PHENO": candidate_diffs,
        "scorer_bias_gradients": [float(g.abs().sum().cpu()) for g in q_grads],
        "edge_q_output_gradients": [float(g.abs().sum().cpu()) for g in q_output_grads],
        "relation_encoder_gradient_norm": float(torch.stack([g.detach().float().square().sum() for g in encoder_grads]).sum().sqrt().cpu()),
        "gcn_gradient_norm": float(torch.stack([g.detach().float().square().sum() for g in gcn_grads]).sum().sqrt().cpu()),
        "graphlearn_projection_gradient_norms": [float(g.abs().sum().cpu()) for g in graphlearn_grads],
        "output_shape": list(probe.shape),
    }, {
        "check": "fold_oof_reliability",
        "pass": True,
        "oof_auc": oof_auc,
        "reliability": cache_meta["reliability"],
    }, {
        "check": "preservation_margin_initial_logit_scale",
        "pass": True,
        "frozen_margin": 0.02,
        "initial_pair_logit_std": initial_relation_logit_std,
        "initial_same_minus_different_logit_gap": initial_pair_logit_gaps,
        "pair_labels_source": "outer-train-only balanced pair sample",
    }, {
        "check": "optimizer_weight_decay",
        "pass": True,
        "weight_decay": wd,
    }, {
        "check": "all_fold_initialization_matches_53_1",
        "pass": True,
        "folds": all_fold_init,
    }, {
        "check": "two_step_warmup_shared_update_and_rng_parity",
        "pass": True,
        "steps": parity_rows,
    }, {
        "check": "checkpoint_model_optimizer_rng_resume_roundtrip",
        "pass": True,
        "model_sha_restored": before_model_sha,
        "cam_sha_restored": before_cam_sha,
        "optimizer_state_sha_restored": before_opt_sha,
        "rng_sha_restored": rng_sha(roundtrip_payload["rng"]),
        "early_stop_state_restored": True,
    }]
    write_csv(RESULTS / "preflight.csv", rows)
    atomic_json(RT / "preflight" / "two_step_parity.json", {
        "legacy_e1_vs_cadr_disabled_warmup": parity_rows,
        "all_fold_initialization": all_fold_init,
    })
    _manifest("PREFLIGHT_PASS")


def run_all(max_workers: int = 2):
    _preflight()
    pending = [f for f in range(1, 11) if not (task_path(f).exists() and best_path(f).exists() and json.loads(task_path(f).read_text()).get("status") == "COMPLETE")]
    active = {}
    max_workers = min(2, max(1, int(max_workers)))
    while pending or active:
        while pending and len(active) < max_workers:
            fold = pending.pop(0)
            log_path = LOGS / f"E1_CADR_GFR_FULL_f{fold:02d}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["CADR_EXP"] = str(EXP)
            log_handle = log_path.open("ab")
            process = subprocess.Popen(
                [sys.executable, str(EXP / "run.py"), "--worker", "--fold", str(fold)],
                cwd=str(ROOT),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active[process] = (fold, log_handle)
        completed = []
        for process, (fold, log_handle) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            log_handle.close()
            if status != 0:
                tail = (LOGS / f"E1_CADR_GFR_FULL_f{fold:02d}.log").read_text(errors="replace")[-12000:]
                raise RuntimeError(f"fold {fold} failed with exit {status}:\n{tail}")
            completed.append(process)
        for process in completed:
            active.pop(process, None)
        time.sleep(1.0)
    summarize()
    _manifest("TRAINING_COMPLETE_PENDING_INDEPENDENT_AUDIT")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    runtime()
    if args.preflight:
        _preflight()
    elif args.worker:
        if args.fold is None:
            raise SystemExit("--worker requires --fold")
        train_one(args.fold)
    elif args.run:
        run_all(args.max_workers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
