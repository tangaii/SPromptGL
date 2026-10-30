#!/usr/bin/env python3
"""Experiment 53 runner.

The parent Experiment 52 implementation is imported read-only. E0 uses its
LegacyFixedGFR construction and forward semantics, while E1-E8 add only the
new relation learner and independent bounded propagation implemented here.
"""
from __future__ import annotations

import argparse
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

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
PARENT = Path("/model/spromptgl/experiment/experiment52_drap_parametric_gcn_wd011")
ROOT = Path("/data/adodas/SPromptGL-main")
sys.path.insert(0, str(PARENT))
import importlib.util
_parent_spec = importlib.util.spec_from_file_location("experiment52_parent_run", PARENT / "run.py")
parent_run = importlib.util.module_from_spec(_parent_spec)
_parent_spec.loader.exec_module(parent_run)
sys.path.insert(0, str(EXP))
import drap53

CFG = json.loads((EXP / "config.json").read_text())
GROUPS = tuple(row["id"] for row in CFG["groups"])
GROUP_BY_ID = {row["id"]: row for row in CFG["groups"]}
DIMS = list(CFG["data"]["dims"])
MODS = ["PHENO", "ANAT", "FUNC", "CORR"]
RT = EXP / ".runtime"
TASKS = RT / "tasks"
CKPTS = RT / "checkpoints"
LATEST = RT / "latest"
LOGS = RT / "logs"
REL = RT / "relation_cache"
FOLD_DATA = RT / "fold_data"
RESULTS = EXP / "results"
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


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float, allow_nan=False))
    os.replace(tmp, path)


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
    if not Path(path).exists():
        return []
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_dirs():
    for path in (RT, TASKS, CKPTS, LATEST, LOGS, REL, FOLD_DATA, RESULTS):
        path.mkdir(parents=True, exist_ok=True)


def task_path(group, fold):
    return TASKS / f"{group}_f{fold:02d}.json"


def checkpoint_path(group, fold):
    return CKPTS / f"{group}_f{fold:02d}.best.pt"


def latest_path(group, fold):
    return LATEST / f"{group}_f{fold:02d}.latest.pt"


def relation_path(fold):
    return REL / f"fold{fold:02d}.pt"


def relation_meta_path(fold):
    return REL / f"fold{fold:02d}.json"


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
        torch.cuda.set_device(0)
        CTX = {"np": np, "pd": pd, "torch": torch, "device": torch.device("cuda")}
    return CTX


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
            raise AssertionError("ABIDE column mapping changed")
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


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def module_sha(module):
    h = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        h.update(name.encode())
        h.update(str(tuple(parameter.shape)).encode())
        h.update(str(parameter.dtype).encode())
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def schema_sha(module):
    return sha_obj([(name, tuple(p.shape), str(p.dtype)) for name, p in sorted(module.named_parameters())])


def state_cpu(module):
    return {name: parameter.detach().cpu().clone() for name, parameter in module.state_dict().items()}


def optimizer_cpu_state(optimizer):
    import copy
    state = copy.deepcopy(optimizer.state_dict())
    for item in state.get("state", {}).values():
        for key, value in list(item.items()):
            if hasattr(value, "detach"):
                item[key] = value.detach().cpu().clone()
    return state


def optimizer_to_cuda(optimizer):
    for item in optimizer.state.values():
        for key, value in list(item.items()):
            if hasattr(value, "to"):
                item[key] = value.cuda()


def finite_module(module):
    import torch
    return all(bool(torch.isfinite(p).all().item()) for p in module.parameters())


def module_grad_norm(module):
    import torch
    values = [(p.grad.detach() ** 2).sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt().cpu()) if values else 0.0


def build_parts(feat):
    c = runtime()
    import layers
    return layers.InitialDataLayer(DIMS)(feat)[1]


def make_anchor_graph_support(fold):
    """Build fixed initial GraphLearn supports without using diagnosis labels."""
    import numpy as np
    x, _ = load_data()[0]
    anchor = parent_run.fold_anchor(fold)
    starts = np.cumsum([0] + DIMS[:-1])
    all_pairs = set()
    modality_pairs = []
    for m, (start, dim) in enumerate(zip(starts, DIMS)):
        graph = anchor["model"]["gfr"]
        weight = graph[f"GraphConstruct{m}.p.weight"].cpu().numpy()
        bias = graph[f"GraphConstruct{m}.p.bias"].cpu().numpy()
        z = x[:, start:start + dim] @ weight.T + bias
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        sim = z @ z.T
        ii, jj = np.where(np.triu(sim > CFG["base"]["graph_threshold"], 1))
        pairs = sorted(zip(ii.astype(np.int64).tolist(), jj.astype(np.int64).tolist()))
        modality_pairs.append(pairs)
        all_pairs.update(pairs)
    union = sorted(all_pairs)
    lookup = {pair: idx for idx, pair in enumerate(union)}
    modality_idx = [np.asarray([lookup[pair] for pair in pairs], np.int64) for pairs in modality_pairs]
    u_i = np.asarray([p[0] for p in union], np.int64)
    u_j = np.asarray([p[1] for p in union], np.int64)
    src = np.concatenate([u_i, u_j])
    dst = np.concatenate([u_j, u_i])
    n = x.shape[0]
    groups = [[] for _ in range(n)]
    for edge_id, node in enumerate(src.tolist()):
        groups[node].append(edge_id)
    width = max(len(g) for g in groups) if groups else 0
    slots = np.full((n, width), -1, np.int64)
    for node, values in enumerate(groups):
        slots[node, :len(values)] = values
    reverse = np.concatenate([np.arange(len(union), 2 * len(union)), np.arange(0, len(union))]).astype(np.int64)
    rng = np.random.default_rng(530900 + fold)
    shuffle_perm = np.arange(len(union), dtype=np.int64)
    rng.shuffle(shuffle_perm)
    # Fixed ordinary-neighbor control: deterministic candidates, split into
    # positive and negative budgets without looking at labels or relation q.
    by_node = [[] for _ in range(n)]
    for edge_id, (i, j) in enumerate(union):
        by_node[i].append(edge_id)
        by_node[j].append(edge_id)
    uniform_pos, uniform_neg = set(), set()
    for node, values in enumerate(by_node):
        vals = list(values)
        rng.shuffle(vals)
        for rank, edge_id in enumerate(vals[:2 * int(CFG["relation"]["max_neighbors_per_sign"])]):
            (uniform_pos if rank % 2 == 0 else uniform_neg).add(int(edge_id))
    return {
        "u_i": u_i,
        "u_j": u_j,
        "src": src,
        "dst": dst,
        "slots": slots,
        "reverse": reverse,
        "modality_idx": modality_idx,
        "shuffle_perm": shuffle_perm,
        "uniform_pos": np.asarray(sorted(uniform_pos), np.int64),
        "uniform_neg": np.asarray(sorted(uniform_neg), np.int64),
        "modality_edge_counts": [int(len(v)) for v in modality_pairs],
        "union_edge_count": int(len(union)),
    }


def sample_pairs(indices, y, candidate_pairs, max_count, seed):
    import numpy as np
    indices = np.asarray(indices, np.int64)
    train_set = set(indices.tolist())
    candidate = [p for p in zip(candidate_pairs[0].tolist(), candidate_pairs[1].tolist()) if p[0] in train_set and p[1] in train_set]
    by_candidate = {
        1: [p for p in candidate if y[p[0]] == y[p[1]]],
        0: [p for p in candidate if y[p[0]] != y[p[1]]],
    }
    ii, jj = np.triu_indices(len(indices), 1)
    pi, pj = indices[ii], indices[jj]
    by_all = {
        1: list(zip(pi[(y[pi] == y[pj])].tolist(), pj[(y[pi] == y[pj])].tolist())),
        0: list(zip(pi[(y[pi] != y[pj])].tolist(), pj[(y[pi] != y[pj])].tolist())),
    }
    rng = np.random.default_rng(int(seed))
    per_class = min(max_count // 2, len(by_all[0]), len(by_all[1]))
    selected = []
    for cls in (0, 1):
        pool = by_candidate[cls] if len(by_candidate[cls]) >= per_class else by_all[cls]
        if len(pool) > per_class:
            take = rng.choice(len(pool), per_class, replace=False)
            selected.extend([pool[int(k)] for k in take])
        else:
            selected.extend(pool)
    selected_set = set(selected)
    for cls in (0, 1):
        need = per_class - sum(int((y[a] == y[b]) == (cls == 1)) for a, b in selected)
        if need > 0:
            remain = [p for p in by_all[cls] if p not in selected_set]
            if len(remain) > need:
                take = rng.choice(len(remain), need, replace=False)
                selected.extend([remain[int(k)] for k in take])
            else:
                selected.extend(remain)
    rng.shuffle(selected)
    pi = np.asarray([p[0] for p in selected], np.int64)
    pj = np.asarray([p[1] for p in selected], np.int64)
    py = (y[pi] == y[pj]).astype(np.int64)
    return pi, pj, py


def pair_tensors(pair):
    import torch
    return tuple(torch.from_numpy(a).cuda() for a in pair)


def relation_optimizer(core):
    return __import__("torch").optim.AdamW(core.optimizer_param_groups(
        lr=CFG["relation"]["lr"],
        matrix_decay=CFG["relation"]["weight_decay_matrix"],
    ))


def train_relation_core(x, pair, seed, epochs):
    import torch
    parts = [torch.from_numpy(x[:, start:start + dim]).float().cuda() for start, dim in zip([0, 48, 54, 64], DIMS)]
    pi, pj, py = pair_tensors(pair)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(int(seed))
        core = drap53.RelationCore(hidden=CFG["relation"]["hidden"], grl_scale=CFG["loss"]["grl_scale"]).cuda()
        optimizer = relation_optimizer(core)
        core.train()
        for _ in range(int(epochs)):
            optimizer.zero_grad()
            _, _, _, pair_loss, _, _, _ = core.pair_forward(parts, pi, pj, py, [], "none")
            if not torch.isfinite(pair_loss):
                raise FloatingPointError("non-finite relation pretraining loss")
            pair_loss.backward()
            for p in core.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError("non-finite relation pretraining gradient")
            optimizer.step()
    return core, parts


def ece_score(prob, label, bins=10):
    import numpy as np
    prob = np.asarray(prob)
    label = np.asarray(label)
    total = len(label)
    out = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (prob >= lo) & (prob <= hi if hi == 1 else prob < hi)
        if mask.any():
            out += mask.mean() * abs(prob[mask].mean() - label[mask].mean())
    return float(out)


def relation_pretrain_fold(fold):
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    if relation_path(fold).exists() and relation_meta_path(fold).exists():
        return
    (x, y), folds = load_data()
    tr, _ = folds[fold - 1]
    support = make_anchor_graph_support(fold)
    candidate_pairs = (support["u_i"], support["u_j"])
    inner = StratifiedKFold(3, shuffle=True, random_state=530100 + fold)
    oof_logits = [[] for _ in range(4)]
    oof_labels = []
    inner_rows = []
    for split_id, (itr, iva) in enumerate(inner.split(tr, y[tr])):
        inner_train = tr[itr]
        inner_valid = tr[iva]
        train_pair = sample_pairs(inner_train, y, candidate_pairs, CFG["relation"]["inner_pair_max"], 530200 + fold * 10 + split_id)
        valid_pair = sample_pairs(inner_valid, y, candidate_pairs, CFG["relation"]["inner_pair_max"], 530300 + fold * 10 + split_id)
        core, parts = train_relation_core(x, train_pair, 530400 + fold * 100 + split_id, CFG["relation"]["inner_epochs"])
        pi, pj, py = pair_tensors(valid_pair)
        core.eval()
        with torch.no_grad():
            z = core.encode(parts)
            probs = []
            for m, scorer in enumerate(core.scorers):
                feat = scorer.pair_features(z[m], pi, pj)
                probs.append(torch.sigmoid(scorer(feat)).cpu().numpy())
        oof_labels.extend(py.cpu().numpy().tolist())
        for m in range(4):
            oof_logits[m].extend(np.log(np.clip(probs[m], 1e-6, 1 - 1e-6) / np.clip(1 - probs[m], 1e-6, 1)).tolist())
        inner_rows.append({"split": split_id, "train_patients": int(len(inner_train)), "heldout_patients": int(len(inner_valid)), "train_pair_count": int(len(train_pair)), "heldout_pair_count": int(len(valid_pair)), "patient_disjoint": len(set(inner_train.tolist()) & set(inner_valid.tolist())) == 0})
        del core
        torch.cuda.empty_cache()
    oof_labels = np.asarray(oof_labels, np.int64)
    calibration_a = []
    calibration_b = []
    relation_metrics = []
    for m in range(4):
        logits = np.asarray(oof_logits[m], np.float64)
        raw_prob = 1.0 / (1.0 + np.exp(-logits))
        auc = float(roc_auc_score(oof_labels, raw_prob)) if len(np.unique(oof_labels)) == 2 else 0.5
        ap = float(average_precision_score(oof_labels, raw_prob)) if len(np.unique(oof_labels)) == 2 else 0.5
        pos = float(raw_prob[oof_labels == 1].mean())
        neg = float(raw_prob[oof_labels == 0].mean())
        cal = LogisticRegression(C=1e6, solver="lbfgs", random_state=0)
        cal.fit(logits.reshape(-1, 1), oof_labels)
        a = float(cal.coef_[0, 0])
        b = float(cal.intercept_[0])
        calibrated = 1.0 / (1.0 + np.exp(-(a * logits + b)))
        reliable = bool(auc >= CFG["relation"]["reliable_auc_threshold"] and calibrated[oof_labels == 1].mean() > calibrated[oof_labels == 0].mean())
        relation_metrics.append({
            "modality": MODS[m], "oof_auc": auc, "oof_pr_auc": ap,
            "oof_brier": float(brier_score_loss(oof_labels, calibrated)),
            "oof_ece": ece_score(calibrated, oof_labels),
            "oof_score_mean": float(calibrated.mean()), "oof_score_std": float(calibrated.std()),
            "oof_pos_mean": float(calibrated[oof_labels == 1].mean()),
            "oof_neg_mean": float(calibrated[oof_labels == 0].mean()),
            "oof_separation": float(calibrated[oof_labels == 1].mean() - calibrated[oof_labels == 0].mean()),
            "calibration_a": a, "calibration_b": b, "reliable": reliable,
        })
        calibration_a.append(a)
        calibration_b.append(b)
    reliable_ids = [m for m, row in enumerate(relation_metrics) if row["reliable"]]
    reliabilities = np.asarray([max(0.0, row["oof_auc"] - 0.5) if row["reliable"] else 0.0 for row in relation_metrics], np.float32)
    if reliabilities.sum() > 0:
        reliabilities /= reliabilities.sum()
    final_pair = sample_pairs(tr, y, candidate_pairs, CFG["relation"]["pair_max"], 530800 + fold)
    final_core, _ = train_relation_core(x, final_pair, 520000 + fold, CFG["relation"]["inner_epochs"])
    package = {
        "relation_state": {k: v.detach().cpu() for k, v in final_core.state_dict().items() if not k.startswith("discriminator.")},
        "candidate": support,
        "calibration_a": np.asarray(calibration_a, np.float32),
        "calibration_b": np.asarray(calibration_b, np.float32),
        "omega": reliabilities,
        "reliable_modalities": reliable_ids,
        "relation_metrics": relation_metrics,
        "inner_rows": inner_rows,
        "final_pair": {"i": final_pair[0], "j": final_pair[1], "y": final_pair[2]},
        "fold": fold,
        "split_sha": split_sha(fold),
        "relation_pretrain_seed": 520000 + fold,
        "test_label_excluded": True,
        "relation_parameter_sha": module_sha(final_core),
    }
    torch_save(relation_path(fold), package)
    atomic_json(relation_meta_path(fold), {
        "fold": fold,
        "relation_metrics": relation_metrics,
        "inner_rows": inner_rows,
        "calibration_a": [float(v) for v in calibration_a],
        "calibration_b": [float(v) for v in calibration_b],
        "omega": [float(v) for v in reliabilities.tolist()],
        "reliable_modalities": reliable_ids,
        "relation_pretrain_seed": 520000 + fold,
        "split_sha": split_sha(fold),
        "test_label_excluded": True,
        "candidate_edge_count": int(support["union_edge_count"]),
        "modality_edge_counts": support["modality_edge_counts"],
    })
    del final_core
    torch.cuda.empty_cache()


def make_relation_module(package, fold, group):
    import torch
    c = package["candidate"]
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(520000 + fold)
        core = drap53.RelationCore(hidden=CFG["relation"]["hidden"], grl_scale=CFG["loss"]["grl_scale"]).cuda()
    core.load_state_dict(package["relation_state"], strict=False)
    module = drap53.DiseaseRelationPropagation(
        core,
        torch.from_numpy(c["u_i"]).cuda(),
        torch.from_numpy(c["u_j"]).cuda(),
        [torch.from_numpy(v).cuda() for v in c["modality_idx"]],
        torch.from_numpy(c["slots"]).cuda(),
        torch.from_numpy(c["reverse"]).cuda(),
        torch.from_numpy(package["calibration_a"]).cuda(),
        torch.from_numpy(package["calibration_b"]).cuda(),
        torch.from_numpy(package["omega"]).cuda(),
        torch.from_numpy(c["shuffle_perm"]).cuda(),
        torch.from_numpy(c["uniform_pos"]).cuda(),
        torch.from_numpy(c["uniform_neg"]).cuda(),
        gamma_negative=CFG["propagation"]["gamma_negative"],
        lambda_prop=CFG["propagation"]["lambda_prop"],
        max_neighbors=CFG["relation"]["max_neighbors_per_sign"],
    ).cuda()
    return module


def build_state(group, fold):
    import torch
    base_state = parent_run.build_state("E0", fold)
    base_state["group"] = group
    base_state["initial_shared_sha"] = parent_run.digest_shared(base_state["base"], base_state["cam"])
    base_state["initial_schema_sha"] = parent_run.schema_shared(base_state["base"], base_state["cam"])
    base_state["initial_rng_sha"] = parent_run.rng_sha(base_state["initial_rng"])
    base_state["initial_cam_sha"] = module_sha(base_state["cam"])
    base_state["pair_sha"] = sha_file(parent_run.FD / f"fold{fold:02d}.npz")
    base_state["split_sha"] = split_sha(fold)
    if group != "E0":
        package = torch_load(relation_path(fold))
        base_state["relation_package"] = package
        base_state["drap"] = make_relation_module(package, fold, group)
        base_state["opt_drap"] = torch.optim.AdamW(
            base_state["drap"].relation.optimizer_param_groups(
                lr=CFG["relation"]["lr"],
                matrix_decay=CFG["relation"]["weight_decay_matrix"],
            )
        )
        base_state["reliable_modalities"] = list(package["reliable_modalities"])
        base_state["omega"] = package["omega"]
    else:
        base_state["drap"] = None
        base_state["opt_drap"] = None
        base_state["reliable_modalities"] = []
        base_state["omega"] = [0.0] * 4
    base_state["initial_relation_sha"] = module_sha(base_state["drap"].relation) if base_state["drap"] is not None else ""
    return base_state


def group_config(group):
    return GROUP_BY_ID[group]


def forward_state(state, fd, train_mode=True, propagation_override=None):
    import torch
    c = runtime()
    state["base"].train(train_mode)
    state["cam"].train(train_mode)
    if state["drap"] is not None:
        state["drap"].train(train_mode)
    parts = build_parts(state["feat"])
    raw = torch.cat(parts, dim=1)
    if state["group"] == "E0":
        fused_e1, adjs, base_details = state["base"](parts)
        delta = raw.new_zeros(raw.shape)
        relation_details = {
            "pair_loss": raw.new_zeros(()), "adv_loss": raw.new_zeros(()),
            "positive_edges": 0, "negative_edges": 0, "positive_norm": 0.0, "negative_norm": 0.0,
            "delta_norm": 0.0, "pair_logits": [], "adv_logits": None, "candidate_q": None,
        }
    else:
        fused_e1, adjs, base_details = state["base"](parts)
        cfg = group_config(state["group"])
        delta, relation_details = state["drap"](
            parts, fd["pair_i"].long(), fd["pair_j"].long(), fd["pair_y"].long(),
            state["group"], state["reliable_modalities"], cfg.get("adv", "none"),
            propagation_override=propagation_override,
        )
    fused = fused_e1 + delta
    logp, hidden, attn = state["cam"](fused)
    utils = parent_run.runtime()["utils"]
    graph = sum(
        utils.GraphConstructLoss(parts[m], adjs[m], CFG["base"]["theta_smooth"], CFG["base"]["theta_degree"], CFG["base"]["theta_sparsity"])
        for m in range(4)
    ) / 4.0
    cls = utils.ClsLoss(logp, state["targ"], state["train"], state["weight"])
    pair_loss = relation_details["pair_loss"]
    adv_loss = relation_details["adv_loss"]
    total = cls + graph + CFG["loss"]["pair"] * pair_loss + CFG["loss"]["adversarial"] * adv_loss if state["group"] != "E0" else cls + graph
    return {
        "logp": logp, "hidden": hidden, "attn": attn, "fused": fused, "fused_e1": fused_e1,
        "delta": delta, "raw": raw, "parts": parts, "adjs": adjs, "base_details": base_details,
        "relation": relation_details, "cls": cls, "graph": graph, "pair_loss": pair_loss,
        "adv_loss": adv_loss, "total": total,
    }


def metrics(logp, state):
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    idx = state["test"]
    yy = state["targ"].detach().cpu().numpy()[idx]
    prob = logp.detach().exp().cpu().numpy()[idx]
    pred = prob.argmax(1)
    cm = confusion_matrix(yy, pred, labels=[0, 1])
    tp, fn, fp, tn = map(int, cm.ravel())
    return {
        "acc": float((pred == yy).mean()),
        "hard_auc": float(roc_auc_score(np.eye(2)[yy], np.eye(2)[pred])),
        "prob_auc": float(roc_auc_score(yy, prob[:, 1])),
        "spe": float(tn / max(tn + fp, 1)),
        "sen": float(tp / max(tp + fn, 1)),
        "ba": float((tn / max(tn + fp, 1) + tp / max(tp + fn, 1)) / 2),
        "asd_f1": float(f1_score(yy, pred, pos_label=0, zero_division=0)),
        "asd_tp": tp, "asd_fn": fn, "td_fp": fp, "td_tn": tn, "n_test": int(len(idx)),
    }


def prepare():
    ensure_dirs()
    load_data()
    for fold in range(1, 11):
        if not (parent_run.FD / f"fold{fold:02d}.npz").exists():
            parent_run.prepare_fold(fold)
    parent_manifest = json.loads((PARENT / "manifest.json").read_text())
    manifest = {
        "experiment": CFG["experiment"], "status": "PREPARED",
        "formal_groups": list(GROUPS), "formal_folds": 90,
        "parent_manifest_sha256": sha_file(PARENT / "manifest.json"),
        "parent_code_sha256": parent_manifest.get("code_sha", {}),
        "parent_data_sha256": parent_manifest.get("protected_source_sha256", {}),
        "design_sha256": sha_file(EXP / "design.md"),
        "config_sha256": sha_file(EXP / "config.json"),
        "fold_sha256": {str(f): split_sha(f) for f in range(1, 11)},
        "anchor_sha256": {str(f): sha_file(parent_run.ANCHORS / f"fold{f:02d}.pt") for f in range(1, 11)},
        "seed_rule": "510000 + fold*100 + 1",
        "relation_rng_rule": "isolated fold-local pretraining and pair sampling",
        "test_labels_for_drap": "forbidden",
        "runtime_root": str(RT), "results_root": str(RESULTS),
    }
    atomic_json(EXP / "manifest.json", manifest)
    print("PREPARE_PASS", flush=True)


def pretrain_relations():
    ensure_dirs()
    for fold in range(1, 11):
        relation_pretrain_fold(fold)
        print("RELATION_CACHE_DONE", fold, flush=True)
    manifest = json.loads((EXP / "manifest.json").read_text())
    manifest["relation_cache_status"] = "COMPLETE"
    manifest["relation_cache_sha256"] = {str(f): sha_file(relation_path(f)) for f in range(1, 11)}
    manifest["relation_meta_sha256"] = {str(f): sha_file(relation_meta_path(f)) for f in range(1, 11)}
    atomic_json(EXP / "manifest.json", manifest)


def preflight():
    import torch
    ensure_dirs()
    if not all(relation_path(f).exists() for f in range(1, 11)):
        raise RuntimeError("relation cache incomplete")
    states = {group: build_state(group, 1) for group in GROUPS}
    fd = parent_run.load_fold_data(1)
    signatures = {
        group: {
            "shared": states[group]["initial_shared_sha"],
            "schema": states[group]["initial_schema_sha"],
            "cam": states[group]["initial_cam_sha"],
            "rng": states[group]["initial_rng_sha"],
            "split": states[group]["split_sha"],
            "pair": states[group]["pair_sha"],
            "relation": states[group]["initial_relation_sha"],
        }
        for group in GROUPS
    }
    ref = signatures["E0"]
    shared_equal = all(signatures[g][k] == ref[k] for g in GROUPS for k in ("shared", "schema", "cam", "rng", "split", "pair"))
    relation_equal = all(signatures[g]["relation"] == signatures["E1"]["relation"] for g in GROUPS if g != "E0")
    with torch.no_grad():
        base = forward_state(states["E0"], fd, False)
        outputs = {g: forward_state(states[g], fd, False, propagation_override="off") for g in GROUPS if g != "E0"}
    parity = {g: {"fused_e1": float((outputs[g]["fused_e1"] - base["fused"]).abs().max().cpu()), "logp": float((outputs[g]["logp"] - base["logp"]).abs().max().cpu())} for g in outputs}
    edge_symmetry = []
    for g in ("E1", "E4", "E7", "E8"):
        state = states[g]
        q = outputs[g]["relation"]["candidate_q"]
        edge_symmetry.append({"group": g, "candidate_q_finite": q is None or bool(torch.isfinite(q).all().item())})
    old = states["E4"]["targ"].detach().clone()
    with torch.no_grad():
        before = forward_state(states["E4"], fd, False)
        states["E4"]["targ"][fd["test"].long()] = 1 - states["E4"]["targ"][fd["test"].long()]
        after = forward_state(states["E4"], fd, False)
        states["E4"]["targ"].copy_(old)
    label_isolation = {"fused": float((before["fused"] - after["fused"]).abs().max().cpu()), "logp": float((before["logp"] - after["logp"]).abs().max().cpu())}
    probe = torch.randn(17, device="cuda", requires_grad=True)
    grl = torch.autograd.grad(drap53.grad_reverse(probe, 0.03).sum(), probe)[0]
    grl_ok = bool(torch.allclose(grl, torch.full_like(grl, -0.03), atol=1e-7, rtol=1e-6))
    with torch.no_grad():
        full = forward_state(states["E4"], fd, False)
        off = forward_state(states["E4"], fd, False, propagation_override="off")
    propagation_nonzero = float((full["delta"] - off["delta"]).norm().cpu()) > 0.0
    label_isolation_pass = label_isolation["fused"] < 1e-5 and label_isolation["logp"] < 1e-5
    result = {
        "status": "PASS" if shared_equal and relation_equal and all(max(v.values()) == 0.0 for v in parity.values()) and label_isolation_pass and grl_ok and propagation_nonzero else "FAIL",
        "shared_initialization_equal": shared_equal, "relation_initialization_equal": relation_equal,
        "signatures": signatures, "parity": parity, "edge_checks": edge_symmetry,
        "test_label_prediction_isolation": label_isolation, "test_label_prediction_isolation_pass": label_isolation_pass, "grl_ok": grl_ok,
        "propagation_nonzero_fold1": propagation_nonzero,
    }
    atomic_json(RT / "preflight.json", result)
    print("PREFLIGHT", result["status"], flush=True)
    if result["status"] != "PASS":
        raise SystemExit(2)


def gradient_audit(state, fd, train_output):
    import torch
    if state["drap"] is None:
        return {"classification_to_relation": 0.0, "relation": 0.0, "finite": True}
    params = list(state["drap"].relation.encoders.parameters()) + list(state["drap"].relation.scorers.parameters())
    grads = torch.autograd.grad(train_output["cls"], params, retain_graph=True, allow_unused=True)
    cls_grad = float(torch.stack([(g.detach() ** 2).sum() for g in grads if g is not None]).sum().sqrt().cpu()) if any(g is not None for g in grads) else 0.0
    train_output["total"].backward()
    return {
        "classification_to_relation": cls_grad,
        "relation": module_grad_norm(state["drap"].relation),
        "encoder": module_grad_norm(state["drap"].relation.encoders),
        "scorer": module_grad_norm(state["drap"].relation.scorers),
        "discriminator": module_grad_norm(state["drap"].relation.discriminator),
        "finite": all(torch.isfinite(p.grad).all().item() for p in state["drap"].parameters() if p.grad is not None),
    }


def train_one(group, fold):
    import torch
    ensure_dirs()
    task = task_path(group, fold)
    if task.exists() and json.loads(task.read_text()).get("status") == "COMPLETE":
        return
    state = build_state(group, fold)
    fd = parent_run.load_fold_data(fold)
    start = 0
    wait = 0
    best_acc = -1.0
    best_epoch = -1
    best_base = best_cam = best_drap = best_ob = best_oc = best_od = None
    best_diag = {}
    latest = latest_path(group, fold)
    if latest.exists():
        try:
            saved = torch_load(latest)
            if saved.get("group") != group or int(saved.get("fold", -1)) != fold:
                raise ValueError("latest identity mismatch")
            state["base"].load_state_dict(saved["base"], strict=True)
            state["cam"].load_state_dict(saved["cam"], strict=True)
            if state["drap"] is not None:
                state["drap"].load_state_dict(saved["drap"], strict=True)
                state["opt_drap"].load_state_dict(saved["opt_drap"]); optimizer_to_cuda(state["opt_drap"])
            state["opt_base"].load_state_dict(saved["opt_base"]); parent_run.optimizer_to_cuda(state["opt_base"])
            state["opt_cam"].load_state_dict(saved["opt_cam"]); parent_run.optimizer_to_cuda(state["opt_cam"])
            parent_run.restore_rng(saved["rng"])
            start = int(saved["epoch"]) + 1; wait = int(saved["wait"]); best_acc = float(saved["best_acc"]); best_epoch = int(saved["best_epoch"])
            best_base = saved.get("best_base"); best_cam = saved.get("best_cam"); best_drap = saved.get("best_drap")
            best_ob = saved.get("best_ob"); best_oc = saved.get("best_oc"); best_od = saved.get("best_od"); best_diag = saved.get("best_diag", {})
        except Exception:
            latest.unlink(missing_ok=True)
    t0 = time.time()
    epoch = start - 1
    try:
        for epoch in range(start, int(CFG["base"]["epochs"])):
            state["epoch"] = epoch
            use_rw = group != "E0" and epoch >= 10
            state["opt_base"].zero_grad(); state["opt_cam"].zero_grad()
            if state["opt_drap"] is not None: state["opt_drap"].zero_grad()
            out = forward_state(state, fd, True)
            if not torch.isfinite(out["total"]):
                raise FloatingPointError("non-finite loss")
            grads = gradient_audit(state, fd, out)
            if group == "E0":
                out["total"].backward()
                grads = {"classification_to_relation": 0.0, "relation": 0.0, "finite": True}
            if not grads.get("finite", True):
                raise FloatingPointError("non-finite gradient")
            state["opt_base"].step(); state["opt_cam"].step()
            if state["opt_drap"] is not None: state["opt_drap"].step()
            with torch.no_grad():
                ev = forward_state(state, fd, False)
                met = metrics(ev["logp"], state)
            if met["acc"] > best_acc:
                best_acc = met["acc"]; best_epoch = epoch; wait = 0
                best_base = state_cpu(state["base"]); best_cam = state_cpu(state["cam"])
                best_drap = state_cpu(state["drap"]) if state["drap"] is not None else None
                best_ob = optimizer_cpu_state(state["opt_base"]); best_oc = optimizer_cpu_state(state["opt_cam"])
                best_od = optimizer_cpu_state(state["opt_drap"]) if state["opt_drap"] is not None else None
                best_diag = {"cls": float(out["cls"].detach()), "graph": float(out["graph"].detach()), "pair": float(out["pair_loss"].detach()), "adv": float(out["adv_loss"].detach()), **grads}
                torch_save(checkpoint_path(group, fold), {
                    "group": group, "fold": fold, "seed": 0, "best_epoch": best_epoch, "epochs_executed": epoch + 1,
                    "metrics": met, "base": best_base, "cam": best_cam, "drap": best_drap,
                    "opt_base": best_ob, "opt_cam": best_oc, "opt_drap": best_od, "diagnostics": best_diag,
                    "best_use_reweight": use_rw, "split_sha": state["split_sha"], "pair_sha": state["pair_sha"],
                    "initial_shared_sha": state["initial_shared_sha"], "initial_schema_sha": state["initial_schema_sha"],
                    "initial_cam_sha": state["initial_cam_sha"], "initial_rng_sha": state["initial_rng_sha"],
                    "initial_relation_sha": state["initial_relation_sha"], "reliable_modalities": state["reliable_modalities"],
                    "omega": state["omega"], "source_run_sha": sha_file(PARENT / "run.py"), "source_drap_sha": sha_file(EXP / "drap53.py"),
                })
            else:
                wait += 1
            if epoch % 10 == 0:
                torch_save(latest, {
                    "group": group, "fold": fold, "epoch": epoch, "wait": wait, "best_acc": best_acc, "best_epoch": best_epoch,
                    "base": state_cpu(state["base"]), "cam": state_cpu(state["cam"]), "drap": state_cpu(state["drap"]) if state["drap"] is not None else None,
                    "opt_base": optimizer_cpu_state(state["opt_base"]), "opt_cam": optimizer_cpu_state(state["opt_cam"]),
                    "opt_drap": optimizer_cpu_state(state["opt_drap"]) if state["opt_drap"] is not None else None,
                    "rng": parent_run.pack_rng(), "best_base": best_base, "best_cam": best_cam, "best_drap": best_drap,
                    "best_ob": best_ob, "best_oc": best_oc, "best_od": best_od, "best_diag": best_diag,
                })
            if wait > int(CFG["base"]["early"]):
                break
        if best_base is None:
            raise RuntimeError("best checkpoint missing")
        state["base"].load_state_dict(best_base, strict=True); state["cam"].load_state_dict(best_cam, strict=True)
        if state["drap"] is not None: state["drap"].load_state_dict(best_drap, strict=True)
        with torch.no_grad():
            check = metrics(forward_state(state, fd, False)["logp"], state)
        if abs(check["acc"] - best_acc) > 1e-8:
            raise AssertionError("best readback mismatch")
        row = {
            "group": group, "name": GROUP_BY_ID[group]["name"], "fold": fold, "seed": 0, "status": "COMPLETE",
            "source": "new_training", "best_epoch": best_epoch, "epochs_executed": epoch + 1,
            "warmup_active_at_best": int(group != "E0" and best_epoch < 10), "wall_seconds": time.time() - t0,
            "best_sha": sha_file(checkpoint_path(group, fold)), "split_sha": state["split_sha"], "pair_sha": state["pair_sha"],
            "initial_shared_sha": state["initial_shared_sha"], "initial_schema_sha": state["initial_schema_sha"],
            "initial_cam_sha": state["initial_cam_sha"], "initial_rng_sha": state["initial_rng_sha"],
            "initial_relation_sha": state["initial_relation_sha"], "reliable_modalities": json.dumps(state["reliable_modalities"]),
            "omega": json.dumps([float(v) for v in state["omega"]]), **check,
        }
        atomic_json(task, row); latest.unlink(missing_ok=True)
        print("DONE", group, fold, best_epoch, row["acc"], flush=True)
    except Exception as exc:
        atomic_json(task, {"group": group, "fold": fold, "status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()})
        raise


def controller(groups):
    ensure_dirs()
    groups = list(groups)
    if any(g not in GROUPS for g in groups):
        raise ValueError(groups)
    jobs = [(g, f) for g in groups for f in range(1, 11)]
    workers = min(5, int(os.environ.get("E53_WORKERS", "5")))
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "0"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "2"
    processes = []
    started = time.time()
    while jobs or processes:
        alive = sum(p.poll() is None for _, p, _ in processes)
        while alive < workers and jobs:
            g, f = jobs.pop(0)
            log = (LOGS / f"{g}_f{f:02d}.log").open("ab")
            p = subprocess.Popen([sys.executable, "-B", str(EXP / "run.py"), "worker", "--group", g, "--fold", str(f)], cwd=EXP, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((g, p, log)); alive += 1
        for item in list(processes):
            if item[1].poll() is not None:
                item[2].close(); processes.remove(item)
        time.sleep(0.5)
    statuses = []
    for g in groups:
        statuses += [json.loads(task_path(g, f).read_text()).get("status", "MISSING") if task_path(g, f).exists() else "MISSING" for f in range(1, 11)]
    manifest = json.loads((EXP / "manifest.json").read_text())
    manifest.setdefault("controller_runs", []).append({"groups": groups, "workers": workers, "completed": statuses.count("COMPLETE"), "failed": statuses.count("FAILED"), "wall_seconds": time.time() - started})
    manifest["training_status"] = "COMPLETE" if all(v == "COMPLETE" for v in statuses) else "PARTIAL_OR_FAILED"
    atomic_json(EXP / "manifest.json", manifest)
    print("CONTROLLER_DONE", groups, statuses.count("COMPLETE"), statuses.count("FAILED"), flush=True)
    if not all(v == "COMPLETE" for v in statuses):
        raise SystemExit(3)


def aggregate():
    import numpy as np
    rows = []
    for g in GROUPS:
        for f in range(1, 11):
            p = task_path(g, f)
            if not p.exists(): raise FileNotFoundError(p)
            row = json.loads(p.read_text())
            if row.get("status") != "COMPLETE": raise RuntimeError(row)
            rows.append(row)
    csv_write(RESULTS / "fold_metrics.csv", rows)
    summary = []
    for g in GROUPS:
        rr = [r for r in rows if r["group"] == g]
        q = {"group": g, "name": rr[0]["name"], "folds": 10, "source": "new_training"}
        for k in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"):
            v = np.asarray([float(r[k]) for r in rr]); q[k + "_mean_pct"] = float(v.mean() * 100); q[k + "_sd_pct"] = float(v.std(ddof=0) * 100)
        summary.append(q)
    csv_write(RESULTS / "summary.csv", summary)
    by = {(r["group"], int(r["fold"])): r for r in rows}
    pairs = []
    comparisons = [("E1_minus_E0","E1","E0"),("E2_minus_E0","E2","E0"),("E3_minus_E0","E3","E0"),("E4_minus_E0","E4","E0"),("E4_minus_E2","E4","E2"),("E4_minus_E3","E4","E3"),("E4_minus_E5","E4","E5"),("E4_minus_E6","E4","E6"),("E4_minus_E7","E4","E7"),("E4_minus_E8","E4","E8")]
    for name, left, right in comparisons:
        for k in ("acc","hard_auc","prob_auc","spe","sen"):
            d = np.asarray([(float(by[(left,f)][k]) - float(by[(right,f)][k])) * 100 for f in range(1,11)])
            pairs.append({"comparison": name, "metric": k, "delta_pp": float(d.mean()), "wins": int((d>0).sum()), "ties": int((d==0).sum()), "losses": int((d<0).sum()), "fold_deltas_pp": json.dumps(d.tolist())})
    csv_write(RESULTS / "paired_comparisons.csv", pairs)
    manifest = json.loads((EXP / "manifest.json").read_text())
    manifest.update({"status": "AGGREGATED", "fold_metrics_sha256": sha_file(RESULTS / "fold_metrics.csv"), "summary_sha256": sha_file(RESULTS / "summary.csv"), "paired_sha256": sha_file(RESULTS / "paired_comparisons.csv")})
    atomic_json(EXP / "manifest.json", manifest)
    print("AGGREGATE_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare"); sub.add_parser("pretrain-relations"); sub.add_parser("preflight")
    c = sub.add_parser("controller"); c.add_argument("--groups", nargs="+", required=True)
    w = sub.add_parser("worker"); w.add_argument("--group", choices=GROUPS, required=True); w.add_argument("--fold", type=int, required=True)
    sub.add_parser("aggregate")
    args = parser.parse_args()
    if args.cmd == "prepare": prepare()
    elif args.cmd == "pretrain-relations": pretrain_relations()
    elif args.cmd == "preflight": preflight()
    elif args.cmd == "controller": controller(args.groups)
    elif args.cmd == "worker": train_one(args.group, args.fold)
    elif args.cmd == "aggregate": aggregate()


if __name__ == "__main__":
    main()
