#!/usr/bin/env python3
"""Experiment 53.1 training runner; historical experiments are read-only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/SPromptGL-main")
PARENT = Path("/model/spromptgl/experiment/experiment52_drap_parametric_gcn_wd011")
LEGACY = Path("/model/spromptgl/experiment/experiment53_drap_disease_relation_propagation")
sys.path.insert(0, str(PARENT))
import importlib.util

_spec = importlib.util.spec_from_file_location("experiment52_parent_run", PARENT / "run.py")
parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parent)
sys.path.insert(0, str(EXP))
import drap531

CFG = json.loads((EXP / "config.json").read_text())
GROUPS = tuple(row["id"] for row in CFG["groups"])
GROUP_BY_ID = {row["id"]: row for row in CFG["groups"]}
MODS = ("PHENO", "ANAT", "FUNC", "CORR")
DIMS = (48, 6, 10, 256)
STARTS = (0, 48, 54, 64)
RT = EXP / ".runtime"
TASKS = RT / "tasks"
CKPTS = RT / "checkpoints"
LATEST = RT / "latest"
LOGS = RT / "logs"
REL = RT / "relation_cache"
RESULTS = EXP / "results"
E0_CKPTS = LEGACY / ".runtime/checkpoints"
E0_METRICS = LEGACY / "results/fold_metrics.csv"
CTX = None
DATA = None
FOLDS = None


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_obj(obj):
    return hashlib.sha256(pickle.dumps(obj, protocol=4)).hexdigest()


def module_sha(module):
    h = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        h.update(name.encode())
        h.update(str(tuple(parameter.shape)).encode())
        h.update(str(parameter.dtype).encode())
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float, allow_nan=False))
    os.replace(tmp, path)


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def csv_read(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def torch_save(path, obj):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        torch.save(obj, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def ensure_dirs():
    for path in (RT, TASKS, CKPTS, LATEST, LOGS, REL, RESULTS):
        path.mkdir(parents=True, exist_ok=True)


def task_path(group, fold):
    return TASKS / f"{group}_f{fold:02d}.json"


def checkpoint_path(group, fold):
    return CKPTS / f"{group}_f{fold:02d}.best.pt"


def latest_path(group, fold):
    return LATEST / f"{group}_f{fold:02d}.latest.pt"


def relation_path(fold):
    return REL / f"fold{fold:02d}.pt"


def runtime():
    global CTX
    if CTX is None:
        import numpy as np
        import pandas as pd
        import torch
        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(0)
        CTX = {"np": np, "pd": pd, "torch": torch, "device": torch.device("cuda")}
    return CTX


def configure_determinism():
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data():
    global DATA, FOLDS
    if DATA is None:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import StratifiedKFold
        base = ROOT / "data/ABIDE"
        frame = pd.read_csv(base / "processed_standard_data.csv")
        mapping = np.load(base / "modal_feat_dict.npy", allow_pickle=True).item()
        expected = [str(v) for key in ("PHENO", "ANAT", "FUNC", "Correlation") for v in mapping[key]]
        if list(frame.columns[:-1]) != expected:
            raise AssertionError("ABIDE feature mapping changed")
        x = frame.iloc[:, :-1].to_numpy(np.float32)
        y = frame.iloc[:, -1].to_numpy(np.int64) - 1
        if x.shape != (871, 320) or np.bincount(y).tolist() != [403, 468]:
            raise AssertionError("ABIDE shape or label counts changed")
        FOLDS = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y))
        DATA = (x, y)
    return DATA, FOLDS


def split_sha(fold):
    import numpy as np
    tr, te = load_data()[1][fold - 1]
    return hashlib.sha256(np.asarray(tr, np.int64).tobytes() + np.asarray(te, np.int64).tobytes()).hexdigest()


def build_parts(feat):
    import layers
    return layers.InitialDataLayer(DIMS)(feat)[1]


def state_cpu(module):
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def optimizer_cpu_state(optimizer):
    import copy
    state = copy.deepcopy(optimizer.state_dict())
    for values in state.get("state", {}).values():
        for key, value in list(values.items()):
            if hasattr(value, "detach"):
                values[key] = value.detach().cpu().clone()
    return state


def optimizer_to_cuda(optimizer):
    for values in optimizer.state.values():
        for key, value in list(values.items()):
            if hasattr(value, "to"):
                values[key] = value.cuda()


def module_grad_norm(module):
    import torch
    values = [(p.grad.detach() ** 2).sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt().cpu()) if values else 0.0


def make_relation_module(package, fold):
    import torch
    candidate = package["candidate"]
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(520000 + fold)
        core = drap531.RelationCore(hidden=CFG["relation"]["hidden"], grl_scale=CFG["adversarial"]["grl_scale"]).cuda()
    missing, unexpected = core.load_state_dict(package["relation_state"], strict=False)
    if unexpected or any(not key.startswith("discriminator.") for key in missing):
        raise RuntimeError(f"relation state schema mismatch: missing={missing}, unexpected={unexpected}")
    return drap531.DiseaseRelationPropagation(
        core,
        candidate["u_i"], candidate["u_j"], candidate["modality_idx"],
        candidate["slots"], candidate["reverse"],
        package["calibration_a"], package["calibration_b"], package["omega"],
        candidate["corr_edge_ids"], candidate["corr_permutation"],
        candidate["shuffled_pos_ids"], candidate["shuffled_neg_ids"],
        candidate["shuffled_pos_budget"], candidate["shuffled_neg_budget"],
        candidate["ordinary_pos_ids"], candidate["ordinary_neg_ids"],
        candidate["ordinary_pos_weights"], candidate["ordinary_neg_weights"],
        gamma_negative=CFG["propagation"]["gamma_negative"],
        lambda_prop=CFG["propagation"]["lambda_prop"],
        max_neighbors=CFG["relation"]["max_neighbors_per_sign"],
        thresholds=(CFG["relation"]["positive_q_threshold"], CFG["relation"]["negative_q_threshold"]),
    ).cuda()


def relation_candidate_scores(package, fold, x):
    import numpy as np
    import torch
    candidate = package["candidate"]
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(520000 + fold)
        core = drap531.RelationCore(hidden=CFG["relation"]["hidden"], grl_scale=CFG["adversarial"]["grl_scale"]).cuda()
    missing, unexpected = core.load_state_dict(package["relation_state"], strict=False)
    if unexpected or any(not key.startswith("discriminator.") for key in missing):
        raise RuntimeError("cached relation state is not compatible with corrected module")
    parts = [torch.from_numpy(x[:, start:start + dim]).float().cuda() for start, dim in zip(STARTS, DIMS)]
    i = torch.as_tensor(candidate["u_i"], device="cuda", dtype=torch.long)
    j = torch.as_tensor(candidate["u_j"], device="cuda", dtype=torch.long)
    with torch.no_grad():
        z = core.encode(parts)
        q_by_modality = torch.full((4, len(candidate["u_i"])), 0.5, device="cuda")
        for m, ids_np in enumerate(candidate["modality_idx"]):
            ids = torch.as_tensor(ids_np, device="cuda", dtype=torch.long)
            if not ids.numel():
                continue
            feat = core.scorers[m].pair_features(z[m], i[ids], j[ids])
            logits = core.scorers[m](feat)
            q_m = torch.sigmoid(float(package["calibration_a"][m]) * logits + float(package["calibration_b"][m]))
            q_by_modality[m, ids] = q_m
        omega = torch.as_tensor(package["omega"], device="cuda", dtype=torch.float32)
        usable = torch.stack([torch.isin(torch.arange(i.numel(), device="cuda"), torch.as_tensor(candidate["modality_idx"][m], device="cuda")) for m in range(4)])
        weight = usable.float() * omega[:, None]
        den = weight.sum(0)
        q = (q_by_modality * weight).sum(0) / den.clamp_min(1e-8)
        q = torch.where(den > 0, q, torch.full_like(q, 0.5))
    del core, parts
    return q.detach().cpu().numpy().astype(np.float32)


def select_disease_edges(q, candidate, max_neighbors, pos_threshold, neg_threshold, positive_budget=None, negative_budget=None):
    import numpy as np
    n_edge = len(q)
    n = len(candidate["slots"])
    by_node = [[] for _ in range(n)]
    for eid, (i, j) in enumerate(zip(candidate["u_i"], candidate["u_j"])):
        by_node[int(i)].append(eid)
        by_node[int(j)].append(eid)
    selected = {"positive": [set() for _ in range(n)], "negative": [set() for _ in range(n)]}
    for node, node_edges in enumerate(by_node):
        for sign, valid, budgets in (("positive", q >= pos_threshold, positive_budget), ("negative", q <= neg_threshold, negative_budget)):
            choices = [eid for eid in node_edges if valid[eid]]
            choices.sort(key=lambda eid: (-abs(float(q[eid]) - 0.5), eid))
            limit = max_neighbors if budgets is None else int(budgets[node])
            selected[sign][node].update(choices[:limit])
    # Mutual endpoint top-k preserves symmetry and enforces the degree budget
    # at both ends of every undirected patient relation.
    pos_ids = np.asarray([eid for eid, (i, j) in enumerate(zip(candidate["u_i"], candidate["u_j"]))
                          if eid in selected["positive"][int(i)] and eid in selected["positive"][int(j)]], dtype=np.int64)
    neg_ids = np.asarray([eid for eid, (i, j) in enumerate(zip(candidate["u_i"], candidate["u_j"]))
                          if eid in selected["negative"][int(i)] and eid in selected["negative"][int(j)]], dtype=np.int64)
    pos_w = np.maximum(2.0 * (q[pos_ids] - 0.5), 0).astype(np.float32)
    neg_w = np.maximum(2.0 * (0.5 - q[neg_ids]), 0).astype(np.float32)
    return pos_ids, pos_w, neg_ids, neg_w


def select_shuffled_control(q, candidate, positive_count, negative_count, max_neighbors, pos_threshold, neg_threshold):
    import numpy as np
    n = len(candidate["slots"])
    positive_degree = np.zeros(n, dtype=np.int64)
    negative_degree = np.zeros(n, dtype=np.int64)
    positive = []
    for eid in sorted(np.flatnonzero(q >= pos_threshold).tolist(), key=lambda k: (-float(q[k]), k)):
        i, j = int(candidate["u_i"][eid]), int(candidate["u_j"][eid])
        if positive_degree[i] < max_neighbors and positive_degree[j] < max_neighbors:
            positive.append(eid); positive_degree[i] += 1; positive_degree[j] += 1
            if len(positive) == positive_count:
                break
    positive_set = set(positive)
    negative = []
    for eid in sorted(np.flatnonzero(q <= neg_threshold).tolist(), key=lambda k: (float(q[k]), k)):
        if eid in positive_set:
            continue
        i, j = int(candidate["u_i"][eid]), int(candidate["u_j"][eid])
        if negative_degree[i] < max_neighbors and negative_degree[j] < max_neighbors:
            negative.append(eid); negative_degree[i] += 1; negative_degree[j] += 1
            if len(negative) == negative_count:
                break
    pos_ids, neg_ids = np.asarray(positive, np.int64), np.asarray(negative, np.int64)
    pos_weights = np.maximum(2 * (q[pos_ids] - 0.5), 0).astype(np.float32)
    neg_weights = np.maximum(2 * (0.5 - q[neg_ids]), 0).astype(np.float32)
    return pos_ids, pos_weights, neg_ids, neg_weights, positive_degree, negative_degree


def projected_similarity(fold, x, candidate):
    import numpy as np
    anchor = parent.fold_anchor(fold)
    gfr = anchor["model"]["gfr"]
    i, j = candidate["u_i"], candidate["u_j"]
    sim_sum = np.zeros(len(i), dtype=np.float64)
    count = np.zeros(len(i), dtype=np.float64)
    for m, (start, dim) in enumerate(zip(STARTS, DIMS)):
        ids = candidate["modality_idx"][m]
        if not len(ids):
            continue
        weight = gfr[f"GraphConstruct{m}.p.weight"].cpu().numpy()
        bias = gfr[f"GraphConstruct{m}.p.bias"].cpu().numpy()
        z = x[:, start:start + dim] @ weight.T + bias
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        sim_sum[ids] += np.sum(z[i[ids]] * z[j[ids]], axis=1)
        count[ids] += 1
    return (sim_sum / np.maximum(count, 1)).astype(np.float32)


def degree_vector(ids, candidate, n):
    import numpy as np
    degree = np.zeros(n, dtype=np.int64)
    np.add.at(degree, candidate["u_i"][ids], 1)
    np.add.at(degree, candidate["u_j"][ids], 1)
    return degree


def greedy_degree_edges(scores, candidate, budget, excluded, descending):
    import numpy as np
    remain = np.asarray(budget, dtype=np.int64).copy()
    excluded = set(int(x) for x in excluded)
    order = sorted(range(len(scores)), key=lambda eid: ((-float(scores[eid])) if descending else float(scores[eid]), eid))
    chosen = []
    for eid in order:
        if eid in excluded:
            continue
        i, j = int(candidate["u_i"][eid]), int(candidate["u_j"][eid])
        if remain[i] > 0 and remain[j] > 0:
            chosen.append(eid)
            remain[i] -= 1
            remain[j] -= 1
    return np.asarray(chosen, dtype=np.int64), remain


def match_control_weights(target_ids, target_weights, candidate_scores, source_weights):
    import numpy as np
    if not len(target_ids):
        return np.zeros((0,), np.float32)
    if not len(source_weights):
        return np.full(len(target_ids), 0.1, np.float32)
    source = np.sort(np.asarray(source_weights, np.float64))
    quantiles = (np.arange(len(target_ids), dtype=np.float64) + 0.5) / len(target_ids)
    return np.interp(quantiles, (np.arange(len(source)) + 0.5) / len(source), source).astype(np.float32)


def augment_relation_cache(package, fold, x):
    import numpy as np
    candidate = package["candidate"]
    candidate["corr_edge_ids"] = np.asarray(candidate["modality_idx"][3], dtype=np.int64)
    rng = np.random.default_rng(531700 + fold)
    candidate["corr_permutation"] = rng.permutation(len(candidate["corr_edge_ids"])).astype(np.int64)
    q = relation_candidate_scores(package, fold, x)
    p_ids, p_w, n_ids, n_w = select_disease_edges(
        q, candidate, CFG["relation"]["max_neighbors_per_sign"],
        CFG["relation"]["positive_q_threshold"], CFG["relation"]["negative_q_threshold"],
    )
    similarity = projected_similarity(fold, x, candidate)
    n_node = len(candidate["slots"])
    p_budget = degree_vector(p_ids, candidate, n_node)
    n_budget = degree_vector(n_ids, candidate, n_node)
    candidate["shuffled_pos_budget"] = p_budget.astype(np.int64)
    candidate["shuffled_neg_budget"] = n_budget.astype(np.int64)
    op_ids, op_remaining = greedy_degree_edges(similarity, candidate, p_budget, (), True)
    on_ids, on_remaining = greedy_degree_edges(similarity, candidate, n_budget, op_ids, False)
    op_weights = match_control_weights(op_ids, p_w, similarity, p_w)
    on_weights = match_control_weights(on_ids, n_w, similarity, n_w)
    e7_q = q.copy()
    corr_ids = candidate["corr_edge_ids"]
    if len(corr_ids):
        e7_q[corr_ids] = q[corr_ids[candidate["corr_permutation"]]]
    e7p, e7pw, e7n, e7nw, e7_p_degree, e7_n_degree = select_shuffled_control(
        e7_q, candidate, len(p_ids), len(n_ids), CFG["relation"]["max_neighbors_per_sign"],
        CFG["relation"]["positive_q_threshold"], CFG["relation"]["negative_q_threshold"],
    )
    candidate.update({
        "shuffled_pos_ids": e7p,
        "shuffled_neg_ids": e7n,
        "ordinary_pos_ids": op_ids,
        "ordinary_neg_ids": on_ids,
        "ordinary_pos_weights": op_weights,
        "ordinary_neg_weights": on_weights,
    })
    package["corrected_control_meta"] = {
        "q_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
        "corr_edge_count": int(len(candidate["corr_edge_ids"])),
        "e4_positive_edges": int(len(p_ids)), "e4_negative_edges": int(len(n_ids)),
        "e7_positive_edges": int(len(e7p)), "e7_negative_edges": int(len(e7n)),
        "e7_positive_unfilled_edges": int(len(p_ids) - len(e7p)),
        "e7_negative_unfilled_edges": int(len(n_ids) - len(e7n)),
        "e8_positive_edges": int(len(op_ids)), "e8_negative_edges": int(len(on_ids)),
        "e4_positive_degree_sha": hashlib.sha256(p_budget.tobytes()).hexdigest(),
        "e4_negative_degree_sha": hashlib.sha256(n_budget.tobytes()).hexdigest(),
        "e7_positive_degree_mean": float(e7_p_degree.mean()), "e7_positive_degree_max": int(e7_p_degree.max()),
        "e7_negative_degree_mean": float(e7_n_degree.mean()), "e7_negative_degree_max": int(e7_n_degree.max()),
        "e8_positive_degree_mean": float(degree_vector(op_ids, candidate, n_node).mean()),
        "e8_positive_degree_max": int(degree_vector(op_ids, candidate, n_node).max()),
        "e8_negative_degree_mean": float(degree_vector(on_ids, candidate, n_node).mean()),
        "e8_negative_degree_max": int(degree_vector(on_ids, candidate, n_node).max()),
        "e8_positive_unfilled_degree": int(op_remaining.sum()),
        "e8_negative_unfilled_degree": int(on_remaining.sum()),
        "e4_positive_weight_mean": float(np.mean(p_w)) if len(p_w) else 0.0,
        "e4_negative_weight_mean": float(np.mean(n_w)) if len(n_w) else 0.0,
        "e7_positive_weight_mean": float(np.mean(e7pw)) if len(e7pw) else 0.0,
        "e7_negative_weight_mean": float(np.mean(e7nw)) if len(e7nw) else 0.0,
        "e8_positive_weight_mean": float(np.mean(op_weights)) if len(op_weights) else 0.0,
        "e8_negative_weight_mean": float(np.mean(on_weights)) if len(on_weights) else 0.0,
        "e8_match_note": "greedy deterministic b-matching on original GraphLearn union; sign budgets target E4 node degrees; remaining degree mismatch is reported",
        "e7_match_note": "fixed permuted-q score assignment on original CORR candidate support; target E4 sign edge counts under per-node max-neighbor cap; any unfilled edge budget is reported",
        "e7_permutation_seed": 531700 + fold,
        "ordinary_similarity": "mean initial projected cosine over modalities supporting each union edge",
    }
    return package


def prepare():
    import torch
    ensure_dirs()
    (x, _), _ = load_data()
    parent.FD = parent.EXP / ".runtime/fold_data"
    missing = [f for f in range(1, 11) if not (parent.FD / f"fold{f:02d}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"read-only Experiment 52 fold data missing: {missing}")
    old_manifest = json.loads((LEGACY / "manifest.json").read_text())
    cache_hashes = old_manifest["relation_cache_sha256"]
    meta_hashes = old_manifest["relation_meta_sha256"]
    for fold in range(1, 11):
        dst = relation_path(fold)
        if dst.exists():
            package = torch_load(dst)
            package = augment_relation_cache(package, fold, x)
            torch_save(dst, package)
        else:
            src = LEGACY / ".runtime/relation_cache" / f"fold{fold:02d}.pt"
            src_meta = LEGACY / ".runtime/relation_cache" / f"fold{fold:02d}.json"
            if sha_file(src) != cache_hashes[str(fold)] or sha_file(src_meta) != meta_hashes[str(fold)]:
                raise RuntimeError(f"historical relation cache SHA mismatch fold {fold}")
            package = torch_load(src)
            if int(package["fold"]) != fold or package["split_sha"] != split_sha(fold) or not package.get("test_label_excluded"):
                raise RuntimeError(f"historical relation cache identity mismatch fold {fold}")
            package["historical_cache_sha256"] = sha_file(src)
            package = augment_relation_cache(package, fold, x)
            torch_save(dst, package)
        if int(package["fold"]) != fold or package["split_sha"] != split_sha(fold):
            raise RuntimeError(f"local relation cache mismatch fold {fold}")
        meta = package["corrected_control_meta"]
        print("CACHE_READY", fold, {k: meta[k] for k in ("corr_edge_count", "e4_positive_edges", "e4_negative_edges", "e7_positive_edges", "e7_negative_edges", "e8_positive_edges", "e8_negative_edges", "e8_positive_unfilled_degree", "e8_negative_unfilled_degree")}, flush=True)
    source_files = {
        str(path): sha_file(path)
        for path in (
            ROOT / "main.py", ROOT / "model.py", ROOT / "network.py", ROOT / "layers.py", ROOT / "utils.py",
            ROOT / "data/ABIDE/processed_standard_data.csv", ROOT / "data/ABIDE/modal_feat_dict.npy",
        )
    }
    anchor_hashes = {str(f): sha_file(parent.ANCHORS / f"fold{f:02d}.pt") for f in range(1, 11)}
    fold_hashes = {str(f): split_sha(f) for f in range(1, 11)}
    fold_data_hashes = {str(f): sha_file(parent.FD / f"fold{f:02d}.npz") for f in range(1, 11)}
    manifest = {
        "experiment": CFG["experiment"], "status": "PREPARED_PREFLIGHT_REQUIRED",
        "execution_root": str(EXP), "requested_root_unavailable": CFG["requested_root"],
        "requested_root_failure": CFG["mount_fallback"]["probe"],
        "mount_mapping": {"data_root": str(ROOT), "data_fs": "fuseblk", "execution_root": str(EXP), "execution_fs": "overlayfs", "same_mount": False},
        "formal_groups": list(GROUPS), "formal_training_folds": 80, "historical_e0_folds": 10,
        "historical_experiment53_manifest_sha256": sha_file(LEGACY / "manifest.json"),
        "historical_experiment53_source_sha256": old_manifest["final_source_sha256"],
        "historical_experiment52_manifest_sha256": sha_file(PARENT / "manifest.json"),
        "historical_e0_checkpoints_sha256": {str(f): sha_file(E0_CKPTS / f"E0_f{f:02d}.best.pt") for f in range(1, 11)},
        "historical_e0_fold_metrics_sha256": sha_file(E0_METRICS),
        "source_data_sha256": source_files,
        "anchor_sha256": anchor_hashes, "fold_sha256": fold_hashes, "fold_data_sha256": fold_data_hashes,
        "historical_relation_cache_sha256": cache_hashes,
        "local_relation_cache_sha256": {str(f): sha_file(relation_path(f)) for f in range(1, 11)},
        "pre_correction_scaffold_sha256": {
            "run.py": "c08955f2dd2b964d6277716d438bb182b6764e956a2149333097fc8c5ebd89a4",
            "audit.py": "2d2e21c8aa9cd00c59db73f6041851bf53865d06135f8721128590e78ea8a2ef",
            "drap531.py": "e480eb884da3b604313db31d325d7d646495b2f84c4a7ef9427b3a9bec81e652",
            "config.json": "84ceb0c5a5805c01d9639931bd157175842e0ebde152dde131c9b6b2470034d5",
        },
        "config_sha256": sha_file(EXP / "config.json"), "design_sha256": sha_file(EXP / "design.md"),
        "initialization": {"seed": "510000 + fold*100 + 1", "relation_seed": "520000 + fold inside fork_rng", "post_construction_rng": "Experiment52 parent builder state after E51 GFR/CAM construction"},
        "test_labels_for_drap": "forbidden; relationship and adversarial conditions draw only outer-train pair labels",
        "training_status": "NOT_STARTED", "checkpoint_audit": {"status": "NOT_STARTED", "count": 0},
        "runtime_root": str(RT), "results_root": str(RESULTS),
    }
    atomic_json(EXP / "manifest.json", manifest)
    print("PREPARE_PASS", flush=True)


def build_state(group, fold):
    import torch
    parent.runtime()
    configure_determinism()
    if group not in GROUPS:
        raise ValueError(group)
    parent.FD = parent.EXP / ".runtime/fold_data"
    state = parent.build_state("E0", fold)
    package = torch_load(relation_path(fold))
    fd = parent.load_fold_data(fold)
    state.update({
        "group": group,
        "initial_shared_sha": parent.digest_shared(state["base"], state["cam"]),
        "initial_schema_sha": parent.schema_shared(state["base"], state["cam"]),
        "initial_rng_sha": parent.rng_sha(state["initial_rng"]),
        "initial_cam_sha": module_sha(state["cam"]),
        "split_sha": split_sha(fold),
        "pair_sha": sha_file(parent.FD / f"fold{fold:02d}.npz"),
        "relation_package": package,
    })
    state["drap"] = make_relation_module(package, fold)
    state["reliable_modalities"] = list(package["reliable_modalities"])
    state["omega"] = package["omega"]
    state["adv_pair_weight"] = make_adv_pair_weights(state, fd)
    state["adv_pair_sha"] = hashlib.sha256(state["adv_pair_weight"].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    state["initial_relation_sha"] = module_sha(state["drap"].relation)
    state["opt_drap"] = torch.optim.AdamW(state["drap"].relation.optimizer_param_groups(
        lr=CFG["relation"]["lr"], matrix_decay=CFG["relation"]["matrix_weight_decay"]
    ))
    return state


def make_adv_pair_weights(state, fd):
    import torch
    core = state["drap"].relation
    parts = build_parts(state["feat"])
    pi, pj = fd["pair_i"].long(), fd["pair_j"].long()
    with torch.no_grad():
        z = core.encode(parts)[3]
        features = core.scorers[3].pair_features(z, pi, pj)
        logits = core.scorers[3](features)
        package = state["relation_package"]
        q = torch.sigmoid(float(package["calibration_a"][3]) * logits + float(package["calibration_b"][3]))
        confidence = 2.0 * (q - 0.5).abs()
        weights = torch.ones_like(confidence)
        labels = fd["pair_y"].long()
        for cls in (0, 1):
            mask = labels == cls
            if bool(mask.any()):
                mean = confidence[mask].mean()
                weights[mask] = confidence[mask] / mean.clamp_min(1e-6) if float(mean) > 1e-6 else 1.0
    return weights.detach()


def group_config(group):
    return GROUP_BY_ID[group]


def forward_state(state, fd, train_mode=True, propagation_enabled=True, override=None):
    import torch
    state["base"].train(train_mode)
    state["cam"].train(train_mode)
    state["drap"].train(train_mode)
    parts = build_parts(state["feat"])
    raw = torch.cat(parts, dim=1)
    fused_e1, adjs, base_details = state["base"](parts)
    delta, details = state["drap"](
        parts, fd["pair_i"].long(), fd["pair_j"].long(), fd["pair_y"].long(),
        state["adv_pair_weight"], group_config(state["group"]),
        propagation_enabled=propagation_enabled, override=override,
    )
    fused = fused_e1 + delta
    logp, hidden, attn = state["cam"](fused)
    utils = parent.runtime()["utils"]
    graph = sum(utils.GraphConstructLoss(
        parts[m], adjs[m], parent.CFG["base"]["theta_smooth"],
        parent.CFG["base"]["theta_degree"], parent.CFG["base"]["theta_sparsity"]
    ) for m in range(4)) / 4.0
    cls = utils.ClsLoss(logp, state["targ"], state["train"], state["weight"])
    total = cls + graph + CFG["adversarial"]["pair_loss_weight"] * details["pair_loss"] + CFG["adversarial"]["loss_weight"] * details["adv_loss"]
    return {
        "logp": logp, "hidden": hidden, "attn": attn, "fused": fused, "fused_e1": fused_e1,
        "delta": delta, "raw": raw, "parts": parts, "adjs": adjs,
        "base_details": base_details, "relation": details, "cls": cls,
        "graph": graph, "total": total,
    }


def metrics(logp, state):
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    idx = state["test"]
    if hasattr(idx, "detach"):
        idx = idx.detach().cpu().numpy()
    yy = state["targ"].detach().cpu().numpy()[idx]
    prob = logp.detach().exp().cpu().numpy()[idx]
    pred = prob.argmax(1)
    cm = confusion_matrix(yy, pred, labels=[0, 1])
    tp, fn, fp, tn = map(int, cm.ravel())
    return {
        "acc": float((pred == yy).mean()),
        "hard_auc": float(roc_auc_score(np.eye(2)[yy], np.eye(2)[pred])),
        "prob_auc": float(roc_auc_score(yy, prob[:, 1])),
        "spe": float(tn / max(tn + fp, 1)), "sen": float(tp / max(tp + fn, 1)),
        "ba": float((tn / max(tn + fp, 1) + tp / max(tp + fn, 1)) / 2),
        "asd_f1": float(f1_score(yy, pred, pos_label=0, zero_division=0)),
        "asd_tp": tp, "asd_fn": fn, "td_fp": fp, "td_tn": tn, "n_test": int(len(idx)),
    }


def gradient_audit(state, output):
    import torch
    params = list(state["drap"].relation.encoders.parameters()) + list(state["drap"].relation.scorers.parameters())
    grads = torch.autograd.grad(output["cls"], params, retain_graph=True, allow_unused=True)
    present = [g for g in grads if g is not None]
    cls_grad = float(torch.stack([(g.detach() ** 2).sum() for g in present]).sum().sqrt().cpu()) if present else 0.0
    output["total"].backward()
    finite = all(bool(torch.isfinite(p.grad).all().item()) for p in state["drap"].parameters() if p.grad is not None)
    return {
        "classification_to_relation": cls_grad,
        "relation_grad": module_grad_norm(state["drap"].relation),
        "encoder_grad": module_grad_norm(state["drap"].relation.encoders),
        "scorer_grad": module_grad_norm(state["drap"].relation.scorers),
        "discriminator_grad": module_grad_norm(state["drap"].relation.discriminator),
        "finite": finite,
    }


def optimizer_configuration(state):
    base_wd = sorted(set(float(g["weight_decay"]) for g in state["opt_base"].param_groups))
    cam_wd = sorted(set(float(g["weight_decay"]) for g in state["opt_cam"].param_groups))
    drap_wd = sorted(set(float(g["weight_decay"]) for g in state["opt_drap"].param_groups))
    return {"base_wd": base_wd, "cam_wd": cam_wd, "drap_wd": drap_wd}


def snapshot(state, epoch, wait, best_acc, best_epoch, best_states, best_diag, warmup_best):
    return {
        "group": state["group"], "fold": int(state["fold"]), "epoch": int(epoch), "wait": int(wait),
        "best_acc": float(best_acc), "best_epoch": int(best_epoch),
        "base": state_cpu(state["base"]), "cam": state_cpu(state["cam"]), "drap": state_cpu(state["drap"]),
        "opt_base": optimizer_cpu_state(state["opt_base"]), "opt_cam": optimizer_cpu_state(state["opt_cam"]),
        "opt_drap": optimizer_cpu_state(state["opt_drap"]), "rng": parent.pack_rng(),
        "best_states": best_states, "best_diag": best_diag, "warmup_best": warmup_best,
        "identity": {"split_sha": state["split_sha"], "pair_sha": state["pair_sha"],
                     "shared_sha": state["initial_shared_sha"], "rng_sha": state["initial_rng_sha"],
                     "relation_sha": state["initial_relation_sha"], "adv_pair_sha": state["adv_pair_sha"]},
    }


def train_one(group, fold, force=False):
    import torch
    ensure_dirs()
    if group not in GROUPS or fold not in range(1, 11):
        raise ValueError((group, fold))
    task = task_path(group, fold)
    best_path = checkpoint_path(group, fold)
    if not force and task.exists() and best_path.exists() and json.loads(task.read_text()).get("status") == "COMPLETE":
        print("SKIP_COMPLETE", group, fold, flush=True)
        return
    state = build_state(group, fold)
    state["fold"] = fold
    fd = parent.load_fold_data(fold)
    start, wait, best_acc, best_epoch = 0, 0, -1.0, -1
    best_states, best_diag, warmup_best = None, {}, {"acc": -1.0, "epoch": -1}
    latest = latest_path(group, fold)
    if latest.exists():
        saved = torch_load(latest)
        if saved.get("group") != group or int(saved.get("fold", -1)) != fold:
            raise RuntimeError("latest checkpoint identity mismatch")
        expected = {"split_sha": state["split_sha"], "pair_sha": state["pair_sha"],
                    "shared_sha": state["initial_shared_sha"], "rng_sha": state["initial_rng_sha"],
                    "relation_sha": state["initial_relation_sha"], "adv_pair_sha": state["adv_pair_sha"]}
        if saved.get("identity") != expected:
            raise RuntimeError("latest checkpoint initialization identity mismatch")
        state["base"].load_state_dict(saved["base"], strict=True)
        state["cam"].load_state_dict(saved["cam"], strict=True)
        state["drap"].load_state_dict(saved["drap"], strict=True)
        state["opt_base"].load_state_dict(saved["opt_base"]); parent.optimizer_to_cuda(state["opt_base"])
        state["opt_cam"].load_state_dict(saved["opt_cam"]); parent.optimizer_to_cuda(state["opt_cam"])
        state["opt_drap"].load_state_dict(saved["opt_drap"]); optimizer_to_cuda(state["opt_drap"])
        parent.restore_rng(saved["rng"])
        start, wait, best_acc, best_epoch = int(saved["epoch"]) + 1, int(saved["wait"]), float(saved["best_acc"]), int(saved["best_epoch"])
        best_states, best_diag = saved["best_states"], saved["best_diag"]
        warmup_best = saved["warmup_best"]
    t0 = time.time()
    epoch = start - 1
    try:
        for epoch in range(start, int(CFG["training"]["max_epochs"])):
            propagation_enabled = epoch >= int(CFG["propagation"]["warmup_epochs"])
            state["opt_base"].zero_grad(set_to_none=True)
            state["opt_cam"].zero_grad(set_to_none=True)
            state["opt_drap"].zero_grad(set_to_none=True)
            out = forward_state(state, fd, True, propagation_enabled=propagation_enabled)
            if not bool(torch.isfinite(out["total"]).item()):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            grads = gradient_audit(state, out)
            if not grads["finite"]:
                raise FloatingPointError(f"non-finite DRAP gradient at epoch {epoch}")
            state["opt_base"].step()
            state["opt_cam"].step()
            state["opt_drap"].step()
            with torch.no_grad():
                ev = forward_state(state, fd, False, propagation_enabled=propagation_enabled)
                met = metrics(ev["logp"], state)
            if epoch < int(CFG["propagation"]["warmup_epochs"]):
                if met["acc"] > warmup_best["acc"]:
                    warmup_best = {"acc": met["acc"], "epoch": epoch}
            elif met["acc"] > best_acc:
                best_acc, best_epoch, wait = met["acc"], epoch, 0
                best_states = {"base": state_cpu(state["base"]), "cam": state_cpu(state["cam"]), "drap": state_cpu(state["drap"])}
                best_diag = {
                    "train_cls": float(out["cls"].detach().cpu()), "train_graph": float(out["graph"].detach().cpu()),
                    "train_pair": float(out["relation"]["pair_loss"].detach().cpu()), "train_adv": float(out["relation"]["adv_loss"].detach().cpu()),
                    **grads, "propagation_enabled": propagation_enabled,
                    "positive_edges": int(out["relation"]["positive_edges"]), "negative_edges": int(out["relation"]["negative_edges"]),
                    "positive_norm": float(out["relation"]["positive_norm"]), "negative_norm": float(out["relation"]["negative_norm"]),
                    "delta_norm": float(out["relation"]["delta_norm"]),
                }
                torch_save(best_path, {
                    "group": group, "fold": fold, "seed": 0, "best_epoch": best_epoch, "epochs_executed": epoch + 1,
                    "metrics": met, **best_states, "diagnostics": best_diag,
                    "branch_signature": {key: GROUP_BY_ID[group][key] for key in ("positive_enabled", "negative_enabled", "adversarial_mode", "uncertainty_mode", "identity_mode")},
                    "best_propagation_enabled": propagation_enabled,
                    "split_sha": state["split_sha"], "pair_sha": state["pair_sha"],
                    "initial_shared_sha": state["initial_shared_sha"], "initial_schema_sha": state["initial_schema_sha"],
                    "initial_cam_sha": state["initial_cam_sha"], "initial_rng_sha": state["initial_rng_sha"],
                    "initial_relation_sha": state["initial_relation_sha"], "adv_pair_sha": state["adv_pair_sha"],
                    "reliable_modalities": state["reliable_modalities"], "omega": list(map(float, state["omega"])),
                    "optimizer_wd": optimizer_configuration(state),
                    "source_run_sha": sha_file(EXP / "run.py"), "source_drap_sha": sha_file(EXP / "drap531.py"),
                })
            else:
                wait += 1
            torch_save(latest, snapshot(state, epoch, wait, best_acc, best_epoch, best_states, best_diag, warmup_best))
            if wait > int(CFG["training"]["early_stop_wait_gt"]):
                break
        if best_states is None or best_epoch < int(CFG["propagation"]["warmup_epochs"]):
            raise RuntimeError("formal best checkpoint missing after propagation warmup")
        state["base"].load_state_dict(best_states["base"], strict=True)
        state["cam"].load_state_dict(best_states["cam"], strict=True)
        state["drap"].load_state_dict(best_states["drap"], strict=True)
        with torch.no_grad():
            readback = forward_state(state, fd, False, propagation_enabled=True)
            check = metrics(readback["logp"], state)
        if any(abs(check[key] - best_acc) > 1e-8 for key in ("acc",)):
            raise AssertionError("best checkpoint readback ACC mismatch")
        package = state["relation_package"]
        oof = package["relation_metrics"]
        row = {
            "group": group, "name": GROUP_BY_ID[group]["name"], "fold": fold, "seed": 0,
            "status": "COMPLETE", "source": "new_training", "best_epoch": best_epoch,
            "epochs_executed": epoch + 1, "warmup_best_epoch": warmup_best["epoch"],
            "warmup_best_acc": warmup_best["acc"], "warmup_active_at_best": 0,
            "wall_seconds": time.time() - t0, "checkpoint_sha256": sha_file(best_path),
            "split_sha": state["split_sha"], "pair_sha": state["pair_sha"],
            "initial_shared_sha": state["initial_shared_sha"], "initial_schema_sha": state["initial_schema_sha"],
            "initial_cam_sha": state["initial_cam_sha"], "initial_rng_sha": state["initial_rng_sha"],
            "initial_relation_sha": state["initial_relation_sha"], "adv_pair_sha": state["adv_pair_sha"],
            "reliable_modalities": json.dumps(state["reliable_modalities"]),
            "omega": json.dumps(list(map(float, state["omega"]))),
            "branch_signature": json.dumps({key: GROUP_BY_ID[group][key] for key in ("positive_enabled", "negative_enabled", "adversarial_mode", "uncertainty_mode", "identity_mode")}),
            "optimizer_wd": json.dumps(optimizer_configuration(state)),
            "pair_oof_auc_CORR": float(oof[3]["oof_auc"]),
            "pair_oof_auc_PHENO": float(oof[0]["oof_auc"]), "pair_oof_auc_ANAT": float(oof[1]["oof_auc"]),
            "pair_oof_auc_FUNC": float(oof[2]["oof_auc"]),
            "diagnostics": json.dumps(best_diag), **check,
        }
        atomic_json(task, row)
        latest.unlink(missing_ok=True)
        print("DONE", group, fold, best_epoch, row["acc"], flush=True)
    except Exception as exc:
        atomic_json(task, {"group": group, "fold": fold, "status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()})
        raise


def compare_tensors(a, b):
    import torch
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[key].detach().cpu(), b[key].detach().cpu()) for key in a)


def preflight():
    import torch
    import numpy as np
    parent.runtime()
    configure_determinism()
    ensure_dirs()
    if not all(relation_path(f).is_file() for f in range(1, 11)):
        raise RuntimeError("corrected relation/control cache is incomplete")
    parent.FD = parent.EXP / ".runtime/fold_data"
    historical_rows = [row for row in csv_read(E0_METRICS) if row.get("group") == "E0"]
    if len(historical_rows) != 10:
        raise RuntimeError("historical E0 metrics do not contain ten folds")
    e0_audit = []
    for fold in range(1, 11):
        state = parent.build_state("E0", fold)
        state["group"] = "E0"
        ckpt_path = E0_CKPTS / f"E0_f{fold:02d}.best.pt"
        ckpt = torch_load(ckpt_path)
        if int(ckpt["fold"]) != fold or ckpt.get("group") != "E0" or ckpt.get("split_sha") != split_sha(fold):
            raise RuntimeError(f"historical E0 checkpoint identity mismatch fold {fold}")
        state["base"].load_state_dict(ckpt["base"], strict=True)
        state["cam"].load_state_dict(ckpt["cam"], strict=True)
        fd = parent.load_fold_data(fold)
        with torch.no_grad():
            output = forward_base_only(state, fd)
            value = metrics(output["logp"], state)
        saved = ckpt["metrics"]
        csv_row = next(row for row in historical_rows if int(row["fold"]) == fold)
        diffs = {key: abs(value[key] - float(saved[key])) for key in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")}
        if max(diffs.values()) > 1e-8 or any(abs(value[key] - float(csv_row[key])) > 1e-8 for key in diffs):
            raise RuntimeError(f"historical E0 independent readback mismatch fold {fold}: {diffs}")
        e0_audit.append({"fold": fold, "sha256": sha_file(ckpt_path), "best_epoch": int(ckpt["best_epoch"]), "metrics_match": True})
        del state, ckpt
        torch.cuda.empty_cache()

    init_rows = []
    branch_checks = {}
    for fold in range(1, 11):
        fd = parent.load_fold_data(fold)
        states = {}
        for group in GROUPS:
            states[group] = build_state(group, fold)
        reference = states["E1"]
        for group, state in states.items():
            for key in ("initial_shared_sha", "initial_schema_sha", "initial_cam_sha", "initial_rng_sha", "split_sha", "pair_sha", "initial_relation_sha", "adv_pair_sha"):
                if state[key] != reference[key]:
                    raise RuntimeError(f"initialization mismatch fold {fold} {group} {key}")
            init_rows.append({"group": group, "fold": fold, "shared_sha": state["initial_shared_sha"], "cam_sha": state["initial_cam_sha"], "rng_sha": state["initial_rng_sha"], "relation_sha": state["initial_relation_sha"], "split_sha": state["split_sha"], "pair_sha": state["pair_sha"], "adv_pair_sha": state["adv_pair_sha"]})
        if fold == 1:
            with torch.no_grad():
                base_e1 = states["E1"]["base"](build_parts(states["E1"]["feat"]))[0]
                branch_outputs = {g: forward_state(states[g], fd, False, propagation_enabled=True) for g in GROUPS}
                warm_outputs = {g: forward_state(states[g], fd, False, propagation_enabled=False) for g in GROUPS}
            branch_checks = {}
            for group in GROUPS:
                row = branch_outputs[group]["relation"]
                negative_norm = float(row["negative_norm"])
                if group in ("E1", "E3") and negative_norm != 0.0:
                    raise RuntimeError(f"negative propagation active in {group}")
                if group in ("E2", "E4", "E5", "E7", "E8") and negative_norm <= 0.0:
                    raise RuntimeError(f"negative propagation absent in {group}")
                if float(warm_outputs[group]["delta"].abs().max().cpu()) != 0.0:
                    raise RuntimeError(f"warmup propagation nonzero in {group}")
                branch_checks[group] = {
                    "positive_norm": float(row["positive_norm"]), "negative_norm": negative_norm,
                    "warmup_delta_max": float(warm_outputs[group]["delta"].abs().max().cpu()),
                    "active_delta_norm": float(branch_outputs[group]["delta"].norm().cpu()),
                }
                if group == "E6":
                    pa, na = row["positive_adjacency"], row["negative_adjacency"]
                    overlap = int(((pa > 0) & (na > 0)).sum().item())
                    if overlap:
                        raise RuntimeError("E6 edge is simultaneously counted as positive and negative")
                    branch_checks[group]["positive_negative_overlap"] = overlap
            if not torch.equal(base_e1, branch_outputs["E1"]["fused_e1"]):
                raise RuntimeError("E1 base forward differs from E1 GFR base output")
            if not torch.equal(branch_outputs["E4"]["fused_e1"], branch_outputs["E1"]["fused_e1"]):
                raise RuntimeError("shared GFR initialization parity failed")

            # Adversarial conditions and gradients: four sources must participate even though only CORR is reliable.
            adversarial_checks = {}
            for group in ("E3", "E4", "E5", "E6", "E7", "E8"):
                state = states[group]
                out = forward_state(state, fd, False, propagation_enabled=False)
                cond = out["relation"]["adv_condition"]
                if group == "E5":
                    if cond is None or float(cond.abs().sum().cpu()) != 0.0:
                        raise RuntimeError("E5 condition input is not all-zero")
                else:
                    expected = torch.nn.functional.one_hot(fd["pair_y"].long(), 2).repeat(4, 1).float()
                    if cond is None or not torch.equal(cond, expected):
                        raise RuntimeError(f"{group} conditional adversary does not use true outer-train pair condition")
                if out["relation"]["adv_logits"].shape != (4 * fd["pair_y"].numel(), 4) or float(out["relation"]["adv_loss"].detach().cpu()) <= 0:
                    raise RuntimeError(f"{group} adversarial head is not active on four modalities")
                state["opt_drap"].zero_grad(set_to_none=True)
                out["relation"]["adv_loss"].backward()
                disc_norm = module_grad_norm(state["drap"].relation.discriminator)
                enc_norm = module_grad_norm(state["drap"].relation.encoders)
                if disc_norm <= 0 or enc_norm <= 0:
                    raise RuntimeError(f"{group} adversarial gradient path is absent")
                adversarial_checks[group] = {"loss": float(out["relation"]["adv_loss"].detach().cpu()), "disc_grad": disc_norm, "encoder_grl_grad": enc_norm, "modalities": int(out["relation"]["adv_modal"].unique().numel()), "condition": "zero" if group == "E5" else "same_diagnosis_onehot"}
            if len(states["E4"]["reliable_modalities"]) != 1 or states["E4"]["reliable_modalities"] != [3]:
                raise RuntimeError("CORR-only relation reliability changed")
            if int(branch_outputs["E4"]["relation"]["adv_modal"].unique().numel()) != 4:
                raise RuntimeError("CORR anchor is not participating in four-source adversary")
            # Held-out-label invariance: only the class loss uses outer-train labels.
            target_before = states["E4"]["targ"].detach().clone()
            with torch.no_grad():
                pred_before = forward_state(states["E4"], fd, False, True)["logp"].clone()
                states["E4"]["targ"][fd["test"].long()] = 1 - states["E4"]["targ"][fd["test"].long()]
                pred_after = forward_state(states["E4"], fd, False, True)["logp"].clone()
                states["E4"]["targ"].copy_(target_before)
            if not torch.equal(pred_before, pred_after):
                raise RuntimeError("test-label permutation changed predictions")
            # E7 changes score-to-edge identity; E8 propagation ignores q altogether.
            q = branch_outputs["E7"]["relation"]["candidate_q"]
            qp = branch_outputs["E7"]["relation"]["q_prop"]
            corr_ids = states["E7"]["drap"].corr_edge_ids
            if torch.equal(q[corr_ids], qp[corr_ids]):
                raise RuntimeError("E7 relation identity permutation did not change scores")
            module = states["E8"]["drap"]
            cfg8 = group_config("E8")
            q0 = torch.zeros_like(branch_outputs["E8"]["relation"]["candidate_q"])
            q1 = torch.linspace(0, 1, q0.numel(), device=q0.device)
            d0 = module._propagate(branch_outputs["E8"]["raw"], q0, cfg8, True)[0]
            d1 = module._propagate(branch_outputs["E8"]["raw"], q1, cfg8, True)[0]
            if not torch.equal(d0, d1):
                raise RuntimeError("E8 ordinary propagation depends on disease q")
            # Relation gradients and small-q sensitivity must survive continuous weighting.
            e4 = states["E4"]
            train_out = forward_state(e4, fd, True, True)
            cls_grad = torch.autograd.grad(train_out["cls"], list(e4["drap"].relation.scorers.parameters()), retain_graph=True, allow_unused=True)
            if not any(g is not None and bool((g.abs().sum() > 0).item()) for g in cls_grad):
                raise RuntimeError("classification gradient does not reach disease relation scorer")
            qprobe = branch_outputs["E4"]["relation"]["candidate_q"].detach().clone().requires_grad_(True)
            dp = e4["drap"]._propagate(branch_outputs["E4"]["raw"].detach(), qprobe, group_config("E4"), True)[0]
            qgrad = torch.autograd.grad(dp.square().sum(), qprobe, allow_unused=True)[0]
            qgrad_norm = 0.0 if qgrad is None else float(qgrad.norm().cpu())
            if not qgrad_norm > 0:
                raise RuntimeError("continuous q-to-message gradient is absent")
            branch_checks["adversarial"] = adversarial_checks
            branch_checks["q_message_gradient_norm"] = qgrad_norm
            branch_checks["test_label_prediction_max_diff"] = float((pred_before - pred_after).abs().max().cpu())
            branch_checks["e7_identity_changed"] = True
            branch_checks["e8_q_invariant"] = True
            branch_checks["relation_classifier_gradient"] = True
        for state in states.values():
            del state
        torch.cuda.empty_cache()

    # Two independently constructed E4 models must reproduce two optimizer updates and RNG consumption.
    a = build_state("E4", 1); b = build_state("E4", 1)
    fd = parent.load_fold_data(1)
    start_rng = parent.pack_rng()
    equivalence = []
    for step in range(2):
        parent.restore_rng(start_rng)
        a["opt_base"].zero_grad(set_to_none=True); a["opt_cam"].zero_grad(set_to_none=True); a["opt_drap"].zero_grad(set_to_none=True)
        oa = forward_state(a, fd, True, propagation_enabled=(step == 1))
        oa["total"].backward()
        a["opt_base"].step(); a["opt_cam"].step(); a["opt_drap"].step()
        rng_a = parent.pack_rng()
        parent.restore_rng(start_rng)
        b["opt_base"].zero_grad(set_to_none=True); b["opt_cam"].zero_grad(set_to_none=True); b["opt_drap"].zero_grad(set_to_none=True)
        ob = forward_state(b, fd, True, propagation_enabled=(step == 1))
        ob["total"].backward()
        b["opt_base"].step(); b["opt_cam"].step(); b["opt_drap"].step()
        rng_b = parent.pack_rng()
        same_output = torch.equal(oa["logp"].detach(), ob["logp"].detach()) and torch.equal(oa["total"].detach(), ob["total"].detach())
        same_params = compare_tensors(state_cpu(a["base"]), state_cpu(b["base"])) and compare_tensors(state_cpu(a["cam"]), state_cpu(b["cam"])) and compare_tensors(state_cpu(a["drap"]), state_cpu(b["drap"]))
        same_rng = parent.rng_sha(rng_a) == parent.rng_sha(rng_b)
        if not (same_output and same_params and same_rng):
            raise RuntimeError(f"deterministic paired-model update failed at step {step}: output={same_output}, params={same_params}, rng={same_rng}")
        equivalence.append({"step": step + 1, "same_output_loss": same_output, "same_parameters": same_params, "same_epoch_rng": same_rng})
        start_rng = rng_a
    parent.restore_rng(start_rng)

    result = {
        "status": "PASS", "historical_e0_checkpoint_readback": e0_audit,
        "historical_e0_readback_count": len(e0_audit), "initialization_rows": init_rows,
        "initialization_group_count": len(init_rows), "branch_checks_fold1": branch_checks,
        "equivalent_two_update_test": equivalence,
        "preflight_sha256": {"config": sha_file(EXP / "config.json"), "design": sha_file(EXP / "design.md"), "run": sha_file(EXP / "run.py"), "drap": sha_file(EXP / "drap531.py")},
    }
    atomic_json(RT / "preflight.json", result)
    manifest = json.loads((EXP / "manifest.json").read_text())
    manifest["preflight"] = {"status": "PASS", "sha256": sha_file(RT / "preflight.json"), "historical_e0_readback": "10/10", "initialization_checks": "80/80", "determinism_two_updates": "PASS"}
    manifest["status"] = "PREFLIGHT_PASS_TRAINING_NOT_STARTED"
    atomic_json(EXP / "manifest.json", manifest)
    print("PREFLIGHT_PASS", flush=True)


def forward_base_only(state, fd):
    import torch
    state["base"].eval(); state["cam"].eval()
    parts = build_parts(state["feat"])
    fused, adjs, detail = state["base"](parts)
    logp, hidden, attn = state["cam"](fused)
    zero = fused.new_zeros(())
    return {"logp": logp, "fused": fused, "hidden": hidden, "attn": attn, "adjs": adjs, "total": zero}


def controller(groups, force_groups=()):
    ensure_dirs()
    if not (RT / "preflight.json").is_file() or json.loads((RT / "preflight.json").read_text()).get("status") != "PASS":
        raise RuntimeError("formal training blocked: preflight has not passed")
    groups = list(groups)
    if set(groups) != set(GROUPS):
        raise ValueError("only the frozen E1-E8 groups may be scheduled")
    force_groups = set(force_groups)
    if not force_groups.issubset(set(groups)):
        raise ValueError("forced groups must be included in the frozen schedule")
    jobs = [(g, f) for g in groups for f in range(1, 11)
            if g in force_groups or not (task_path(g, f).exists() and checkpoint_path(g, f).exists() and json.loads(task_path(g, f).read_text()).get("status") == "COMPLETE")]
    workers = min(5, int(os.environ.get("E531_WORKERS", "5")))
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "0"; env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "2"
    processes = []
    started = time.time()
    while jobs or processes:
        alive = sum(proc.poll() is None for _, proc, _ in processes)
        while alive < workers and jobs:
            group, fold = jobs.pop(0)
            suffix = ".deterministic" if group in force_groups else ""
            log = (LOGS / f"{group}_f{fold:02d}{suffix}.log").open("ab")
            cmd = [sys.executable, "-B", str(EXP / "run.py"), "worker", "--group", group, "--fold", str(fold)]
            if group in force_groups:
                cmd.append("--force")
            proc = subprocess.Popen(cmd, cwd=EXP, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((group, proc, log)); alive += 1
        for item in list(processes):
            if item[1].poll() is not None:
                item[2].close(); processes.remove(item)
        time.sleep(1)
    statuses = [json.loads(task_path(g, f).read_text()).get("status", "MISSING") if task_path(g, f).exists() else "MISSING" for g in groups for f in range(1, 11)]
    manifest = json.loads((EXP / "manifest.json").read_text())
    manifest.setdefault("controller_runs", []).append({"groups": groups, "workers": workers, "completed": statuses.count("COMPLETE"), "failed": statuses.count("FAILED"), "wall_seconds": time.time() - started})
    manifest["training_status"] = "80/80_COMPLETE" if statuses.count("COMPLETE") == 80 else "PARTIAL_OR_FAILED"
    atomic_json(EXP / "manifest.json", manifest)
    print("CONTROLLER_DONE", statuses.count("COMPLETE"), statuses.count("FAILED"), flush=True)
    if statuses.count("COMPLETE") != 80:
        raise SystemExit(3)


def aggregate():
    import numpy as np
    new_rows = []
    for group in GROUPS:
        for fold in range(1, 11):
            task = task_path(group, fold)
            if not task.exists():
                raise FileNotFoundError(task)
            row = json.loads(task.read_text())
            if row.get("status") != "COMPLETE":
                raise RuntimeError(row)
            new_rows.append(row)
    old_rows = [row for row in csv_read(E0_METRICS) if row.get("group") == "E0"]
    if len(old_rows) != 10:
        raise RuntimeError("historical E0 reference row count is not ten")
    for row in old_rows:
        row = dict(row); row["source"] = "historical_readonly"; row["name"] = "E51_E1_EXACT_REPLAY_READ_ONLY"
        new_rows.append(row)
    csv_write(RESULTS / "fold_metrics.csv", new_rows)
    summary = []
    for group in ("E0",) + GROUPS:
        rows = [r for r in new_rows if r["group"] == group]
        record = {"group": group, "name": rows[0].get("name", group), "folds": len(rows), "source": rows[0].get("source", "new_training")}
        for key in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"):
            values = np.asarray([float(r[key]) for r in rows])
            record[key + "_mean_pct"] = float(values.mean() * 100)
            record[key + "_sd_pct"] = float(values.std(ddof=0) * 100)
        summary.append(record)
    csv_write(RESULTS / "summary.csv", summary)
    by = {(r["group"], int(r["fold"])): r for r in new_rows}
    comparisons = [(f"{left}_minus_{right}", left, right) for left, right in (("E1","E0"),("E2","E1"),("E3","E1"),("E4","E2"),("E4","E3"),("E4","E5"),("E4","E6"),("E4","E7"),("E4","E8"),("E4","E0"))]
    pairs = []
    for name, left, right in comparisons:
        for key in ("acc", "hard_auc", "prob_auc", "spe", "sen"):
            diffs = np.asarray([(float(by[(left, f)][key]) - float(by[(right, f)][key])) * 100 for f in range(1, 11)])
            pairs.append({"comparison": name, "metric": key, "delta_pp": float(diffs.mean()), "wins": int((diffs > 0).sum()), "ties": int((diffs == 0).sum()), "losses": int((diffs < 0).sum()), "fold_deltas_pp": json.dumps(diffs.tolist())})
    csv_write(RESULTS / "paired_comparisons.csv", pairs)
    manifest = json.loads((EXP / "manifest.json").read_text())
    manifest.update({"status": "AGGREGATED_AUDIT_PENDING", "fold_metrics_sha256": sha_file(RESULTS / "fold_metrics.csv"), "summary_sha256": sha_file(RESULTS / "summary.csv"), "paired_comparisons_sha256": sha_file(RESULTS / "paired_comparisons.csv")})
    atomic_json(EXP / "manifest.json", manifest)
    print("AGGREGATE_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("preflight")
    ctl = sub.add_parser("controller"); ctl.add_argument("--groups", nargs="+", required=True)
    ctl.add_argument("--force-groups", nargs="*", default=[])
    worker = sub.add_parser("worker"); worker.add_argument("--group", choices=GROUPS, required=True); worker.add_argument("--fold", type=int, required=True); worker.add_argument("--force", action="store_true")
    sub.add_parser("aggregate")
    args = parser.parse_args()
    if args.cmd == "prepare": prepare()
    elif args.cmd == "preflight": preflight()
    elif args.cmd == "controller": controller(args.groups, args.force_groups)
    elif args.cmd == "worker": train_one(args.group, args.fold, args.force)
    elif args.cmd == "aggregate": aggregate()


if __name__ == "__main__":
    main()
