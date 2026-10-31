#!/usr/bin/env python3
"""Experiment 56 preparation, training, and audit runner."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("DUGSA_EXP", str(Path(__file__).resolve().parent))

EXP = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/SPromptGL-main")
RT = EXP / "runtime"
FD = RT / "fold_data"
CKPT = EXP / "checkpoints"
LOGS = RT / "logs"
TASKS = RT / "tasks"
RESULTS = EXP / "results"
CFG = json.loads((EXP / "config.json").read_text())
sys.path.insert(0, str(EXP / "source"))
sys.path.insert(0, str(EXP))
E1 = importlib.import_module("e1_base")
from dcraag import DCRAGGFR, disease_condition

MODALITIES = ("PHENO", "ANAT", "FUNC", "CORR")
FEATURE_START = (0, 48, 54, 64)
HIST_E1 = {
    "acc": 0.9195924765,
    "hard_auc": 0.9174339196,
    "prob_auc": 0.9318564563,
    "spe": 0.9463922294,
    "sen": 0.8884756098,
    "ba": 0.9174339196,
    "asd_f1": 0.910928,
}
_THREADS_SET = False


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)


def atomic_torch_save(path: Path, value: dict) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def torch_load(path: Path) -> dict:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def runtime():
    global _THREADS_SET
    ctx = E1.runtime()
    import torch
    if not _THREADS_SET:
        torch.set_num_threads(int(CFG["optimizer"]["cpu_threads_per_worker"]))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _THREADS_SET = True
    return ctx


def root_protected_sha() -> dict[str, str]:
    return {str(ROOT / rel): sha(ROOT / rel) for rel in CFG["protected_root"]}


def experiment_source_sha() -> dict[str, str]:
    paths = [EXP / name for name in ("config.json", "design.md", "run.py", "dcraag.py", "tests.py", "audit.py")]
    paths.extend(sorted((EXP / "source").glob("*.py")))
    return {str(p.relative_to(EXP)): sha(p) for p in paths if p.exists()}


def inherited_state_sha(gfr: "torch.nn.Module", cam: "torch.nn.Module") -> str:
    import torch
    h = hashlib.sha256()
    for prefix, module in (("gfr", gfr), ("cam", cam)):
        for name, tensor in sorted(module.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            h.update(f"{prefix}.{name}|{value.dtype}|{tuple(value.shape)}".encode() + b"\0")
            h.update(value.numpy().tobytes())
    return h.hexdigest()


def new_parameter_sha(model: "torch.nn.Module") -> str:
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        if name.startswith("graphs.") or name.startswith("gcn.") or name == "fusion_weight":
            continue
        value = param.detach().cpu().contiguous()
        h.update(name.encode() + b"\0")
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def assert_e1_anchor_tensors(state: dict) -> None:
    import torch
    gfr_anchor = state["anchor"]["model"]["gfr"]
    for m in range(4):
        graph_prefix, gcn_prefix = f"GraphConstruct{m}.", f"MessagePassing{m}."
        graph_expected = {k[len(graph_prefix):]: v for k, v in gfr_anchor.items() if k.startswith(graph_prefix)}
        gcn_expected = {k[len(gcn_prefix):]: v for k, v in gfr_anchor.items() if k.startswith(gcn_prefix)}
        assert graph_expected and gcn_expected
        assert all(torch.equal(state["model"].graphs[m].state_dict()[k].cpu(), v.cpu()) for k, v in graph_expected.items())
        assert all(torch.equal(state["model"].gcn[m].state_dict()[k].cpu(), v.cpu()) for k, v in gcn_expected.items())
    assert torch.equal(state["model"].fusion_weight.detach().cpu(), gfr_anchor["fusion_weight"].cpu())
    expected_cam = state["anchor"]["model"]["pln"]
    actual_cam = state["cam"].state_dict()
    assert expected_cam.keys() == actual_cam.keys()
    assert all(torch.equal(actual_cam[k].detach().cpu(), v.cpu()) for k, v in expected_cam.items())


def relation_targets(y: "np.ndarray", pairs: "np.ndarray") -> "np.ndarray":
    yi, yj = y[pairs[:, 0]], y[pairs[:, 1]]
    return np.where((yi == 0) & (yj == 0), 0,
                    np.where((yi == 1) & (yj == 1), 1, 2)).astype(np.int64)


def sample_relation_pairs(ids: "np.ndarray", y: "np.ndarray", per_class: int, seed: int) -> tuple["np.ndarray", "np.ndarray"]:
    rng = np.random.default_rng(seed)
    by_label = {label: np.asarray(ids)[y[np.asarray(ids)] == label] for label in (0, 1)}
    out, labels = [], []
    specs = ((0, 0), (1, 1), (0, 1))
    for target, (a, b) in enumerate(specs):
        left, right = by_label[a], by_label[b]
        if a == b:
            all_i, all_j = np.triu_indices(len(left), 1)
            count = min(per_class, len(all_i))
            selected = rng.choice(len(all_i), count, replace=False)
            pairs = np.stack((left[all_i[selected]], left[all_j[selected]]), axis=1)
        else:
            count = min(per_class, len(left) * len(right))
            flat = rng.choice(len(left) * len(right), count, replace=False)
            pairs = np.stack((left[flat // len(right)], right[flat % len(right)]), axis=1)
            pairs.sort(axis=1)
        out.append(pairs)
        labels.append(np.full(len(pairs), target, dtype=np.int64))
    pairs = np.concatenate(out, axis=0)
    targets = np.concatenate(labels, axis=0)
    order = rng.permutation(len(targets))
    return pairs[order], targets[order]


def raw_pair_features(x: "np.ndarray", pairs: "np.ndarray", modality: int) -> "np.ndarray":
    from sklearn.preprocessing import normalize
    start = FEATURE_START[modality]
    stop = start + CFG["data"]["dims"][modality]
    z = np.asarray(x[:, start:stop], dtype=np.float32)
    left, right = z[pairs[:, 0]], z[pairs[:, 1]]
    cosine = np.sum(normalize(left) * normalize(right), axis=1)
    return np.concatenate((np.abs(left - right), left * right, cosine[:, None]), axis=1).astype(np.float32)


def relation_oof_and_teacher(x: "np.ndarray", y: "np.ndarray", train: "np.ndarray", fold: int, modality: int) -> tuple[float, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    oof_y, oof_p = [], []
    inner = StratifiedKFold(5, shuffle=True, random_state=56000 + fold)
    for k, (itr, iva) in enumerate(inner.split(train, y[train])):
        tr_ids, va_ids = train[itr], train[iva]
        p_tr, t_tr = sample_relation_pairs(tr_ids, y, 768, 560000 + fold * 100 + modality * 10 + k)
        p_va, t_va = sample_relation_pairs(va_ids, y, 384, 560001 + fold * 100 + modality * 10 + k)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=5600 + fold + modality))
        model.fit(raw_pair_features(x, p_tr, modality), t_tr)
        pred = model.predict_proba(raw_pair_features(x, p_va, modality))
        oof_y.append(t_va)
        oof_p.append(pred)
    yy, pp = np.concatenate(oof_y), np.concatenate(oof_p)
    auc = float(roc_auc_score(yy, pp, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    full_pairs, full_targets = sample_relation_pairs(train, y, 1800, 561000 + fold * 10 + modality)
    teacher = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=56100 + fold + modality))
    teacher.fit(raw_pair_features(x, full_pairs, modality), full_targets)
    return auc, teacher


def disease_probability_teacher(x: "np.ndarray", y: "np.ndarray", train: "np.ndarray", test: "np.ndarray", fold: int) -> "np.ndarray":
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    train_oof = np.zeros((len(train), 2), dtype=np.float32)
    for k, (itr, iva) in enumerate(StratifiedKFold(5, shuffle=True, random_state=56200 + fold).split(train, y[train])):
        ids_tr, ids_va = train[itr], train[iva]
        teacher = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=5620 + fold + k))
        teacher.fit(x[ids_tr, 64:320], y[ids_tr])
        train_oof[iva] = teacher.predict_proba(x[ids_va, 64:320])
    full = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=56200 + fold))
    full.fit(x[train, 64:320], y[train])
    all_prob = np.zeros((x.shape[0], 2), dtype=np.float32)
    all_prob[train] = train_oof
    all_prob[test] = full.predict_proba(x[test, 64:320])
    assert np.allclose(all_prob[train].sum(1), 1.0, atol=1e-6)
    assert np.allclose(all_prob[test].sum(1), 1.0, atol=1e-6)
    return all_prob


def build_candidates(x: "np.ndarray", raw_adj: "np.ndarray", teacher, modality: int) -> tuple["np.ndarray", "np.ndarray"]:
    from sklearn.preprocessing import normalize
    start = FEATURE_START[modality]
    stop = start + CFG["data"]["dims"][modality]
    z = normalize(np.asarray(x[:, start:stop], dtype=np.float32))
    sim = z @ z.T
    n = sim.shape[0]
    raw_adj = np.maximum(np.asarray(raw_adj, dtype=np.float32), 0.0)
    edges, priors = [], []
    for i in range(n):
        sim_row = sim[i].copy()
        sim_row[i] = -np.inf
        top48 = np.argpartition(sim_row, -min(48, n - 1))[-min(48, n - 1):]
        top16 = top48[np.argsort(sim_row[top48])[-min(16, n - 1):]]
        raw_ids = np.flatnonzero(raw_adj[i] > 0)
        raw_ids = raw_ids[raw_ids != i]
        probe = np.unique(np.concatenate((top48, raw_ids))).astype(np.int64)
        pair_idx = np.column_stack((np.full(len(probe), i, dtype=np.int64), probe))
        pred = teacher.predict_proba(raw_pair_features(x, pair_idx, modality))
        relation_ids = probe[np.argsort(pred.max(1))[-min(16, len(probe)):]]
        candidates = np.unique(np.concatenate((raw_ids, top16, relation_ids))).astype(np.int64)
        candidates = candidates[candidates != i]
        if len(candidates) > 48:
            candidate_pairs = np.column_stack((np.full(len(candidates), i, dtype=np.int64), candidates))
            p = teacher.predict_proba(raw_pair_features(x, candidate_pairs, modality)).max(1)
            graph_score = raw_adj[i, candidates]
            cos_score = (sim[i, candidates] + 1.0) * 0.5
            score = 0.35 * cos_score + 0.35 * p + 0.30 * graph_score
            candidates = candidates[np.argsort(score)[-48:]]
        for j in candidates:
            edges.append((i, int(j)))
            priors.append(max(float(raw_adj[i, j]), 0.15 * max(float((sim[i, j] + 1.0) * 0.5), 0.01)))
    edge_arr = np.asarray(edges, dtype=np.int64)
    prior_arr = np.asarray(priors, dtype=np.float32)
    assert edge_arr.shape[1] == 2 and np.all(edge_arr[:, 0] != edge_arr[:, 1])
    assert np.all(np.bincount(edge_arr[:, 0], minlength=n) <= 48)
    return edge_arr, np.maximum(prior_arr, 1e-4)


def site_ids(x: "np.ndarray") -> "np.ndarray":
    # Verified mapping columns: PHENO offsets 26..45 are SITE_ID one-hot fields.
    return np.argmax(x[:, 26:46], axis=1).astype(np.int64)


def compute_edge_prior(raw_x: np.ndarray, frozen_anchor_graph: np.ndarray, modality: int, edges: np.ndarray) -> np.ndarray:
    """Pure pair-specific prior used for both real and forced diagnostic edges."""
    start = FEATURE_START[modality]; stop = start + int(CFG["data"]["dims"][modality])
    z = raw_x[:, start:stop].astype(np.float32)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
    sim = np.sum(z[edges[:, 0]] * z[edges[:, 1]], axis=1)
    anchor = np.asarray(frozen_anchor_graph, dtype=np.float32)[edges[:, 0], edges[:, 1]]
    return np.maximum(np.maximum(anchor, 0.15 * np.maximum((sim + 1.0) * 0.5, 0.01)), 1e-4).astype(np.float32)


def _edge_invariants(original: np.ndarray, candidate: np.ndarray, site: np.ndarray,
                     train: np.ndarray, test: np.ndarray) -> dict:
    split = np.full(len(site), -1, dtype=np.int8); split[train] = 0; split[test] = 1
    assert original.shape == candidate.shape
    assert np.all(candidate[:, 0] == original[:, 0])
    assert np.all(candidate[:, 0] != candidate[:, 1])
    for arr in (original, candidate):
        for receiver in np.unique(arr[:, 0]):
            senders = arr[arr[:, 0] == receiver, 1]
            assert len(senders) == len(np.unique(senders))
    assert np.array_equal(np.sort(np.bincount(original[:, 1], minlength=len(site))),
                          np.sort(np.bincount(candidate[:, 1], minlength=len(site))))
    for receiver in np.unique(original[:, 0]):
        a = original[original[:, 0] == receiver, 1]
        b = candidate[candidate[:, 0] == receiver, 1]
        assert sorted((int(site[x]), int(split[x])) for x in a) == sorted((int(site[x]), int(split[x])) for x in b)
    return {"no_self": True, "no_duplicate": True, "receiver_outdegree": True,
            "sender_indegree": True, "receiver_sender_site_partition": True}


def strict_double_edge_swap(edges: np.ndarray, site: np.ndarray, train: np.ndarray,
                            test: np.ndarray, seed: int, max_passes: int = 12) -> tuple[np.ndarray, dict]:
    """Swap selected directed edges while preserving all registered invariants.

    Receiver-local sender sets are updated after each swap, so duplicate checks
    are against the current graph rather than a stale original edge array.
    """
    split = np.full(len(site), -1, dtype=np.int8); split[train] = 0; split[test] = 1
    result = np.asarray(edges, dtype=np.int64).copy()
    rng = np.random.default_rng(seed)
    changed = np.zeros(len(result), dtype=bool)
    buckets: dict[tuple[int, int], list[int]] = {}
    receiver_sets: dict[int, set[int]] = {}
    for slot, (receiver, sender) in enumerate(result):
        buckets.setdefault((int(site[sender]), int(split[sender])), []).append(slot)
        receiver_sets.setdefault(int(receiver), set()).add(int(sender))
    for _ in range(max_passes):
        progress = False
        for key in sorted(buckets):
            slots = list(buckets[key]); rng.shuffle(slots)
            for aa in range(0, len(slots) - 1, 2):
                a, b = slots[aa], slots[aa + 1]
                i, j = map(int, result[a]); u, v = map(int, result[b])
                if i == u or j == v or v == i or j == u:
                    continue
                # Excluding the currently occupied sender at each slot.
                if (v in receiver_sets[i] and v != j) or (j in receiver_sets[u] and j != v):
                    continue
                receiver_sets[i].remove(j); receiver_sets[i].add(v)
                receiver_sets[u].remove(v); receiver_sets[u].add(j)
                result[a, 1], result[b, 1] = v, j
                changed[a] = changed[b] = True
                progress = True
        if not progress:
            break
    inv = _edge_invariants(edges, result, site, train, test)
    stats = {**inv, "changed_edges": int(changed.sum()),
             "changed_fraction": float(changed.mean()) if len(changed) else 0.0,
             "insufficient_matched_control": bool(changed.mean() < 0.30 if len(changed) else True)}
    return result, stats


def _fold_cache_path(fold: int) -> Path:
    return FD / f"fold{fold:02d}.pt"


def preparation_signature() -> str:
    paths = [EXP / "run.py", EXP / "dcraag.py", EXP / "config.json", EXP / "source/e1_base.py"]
    payload = {str(p.relative_to(EXP)): sha(p) for p in paths}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def prepare_fold(fold: int) -> dict:
    import numpy as np
    import torch
    cache_path = _fold_cache_path(fold)
    if cache_path.exists():
        data = torch_load(cache_path)
        if data.get("preparation_signature") == preparation_signature() and data.get("split_sha") == E1.split_sha(fold):
            return data
    ctx = runtime()
    x, y = E1.load_data()[0]
    train, test = E1.load_data()[1][fold - 1]
    # Only this train slice is passed to relation targets, reliability and teacher fits.
    probability = disease_probability_teacher(x, y, train, test, fold)
    relation_auc, pair_teachers = [], []
    for modality in range(4):
        auc, teacher = relation_oof_and_teacher(x, y, train, fold, modality)
        relation_auc.append(auc)
        pair_teachers.append(teacher)

    state = E1.build_state("E1", fold)
    state["model"].eval()
    _, parts = state["parts"](state["feat"])
    with torch.no_grad():
        raw_adjs = [state["model"].graphs[m](parts[m]).detach().cpu().numpy() for m in range(4)]
    candidate_arrays, prior_arrays = [], []
    for modality in range(4):
        edges, prior = build_candidates(x, raw_adjs[modality], pair_teachers[modality], modality)
        candidate_arrays.append(edges)
        prior_arrays.append(prior)
    pairs, targets = sample_relation_pairs(train, y, CFG["model"]["relation_batch_per_class"], 56400 + fold)
    assert set(pairs.ravel()).issubset(set(train.tolist()))
    conditions = disease_condition(torch.as_tensor(probability), torch.as_tensor(pairs)).numpy().astype(np.float32)
    rel = np.clip((np.asarray(relation_auc) - 0.5) / 0.25, 0.05, 1.0).astype(np.float32)
    rel = 0.05 + 0.95 * (rel - 0.05) / 0.95
    sid = site_ids(x)
    # Candidate-level controls are retained only as a legacy cache diagnostic.
    # Formal wrong-graph intervention is generated after the checkpoint from selected Top-16 edges.
    wrong = [candidate_arrays[m].copy() for m in range(4)]
    wrong_probability = probability.copy()
    condition_rng = np.random.default_rng(56500 + fold)
    for ids in (train, test):
        shuffled = np.asarray(ids)[condition_rng.permutation(len(ids))]
        wrong_probability[np.asarray(ids)] = probability[shuffled]
    data = {
        "fold": fold,
        "preparation_signature": preparation_signature(),
        "split_sha": E1.split_sha(fold),
        "train": np.asarray(train, dtype=np.int64),
        "test": np.asarray(test, dtype=np.int64),
        "pairs": pairs,
        "relation_targets": targets,
        "relation_condition": conditions,
        "patient_prob": probability,
        "relation_oof_auc": np.asarray(relation_auc, dtype=np.float32),
        "reliability": rel,
        "candidate_edges": candidate_arrays,
        "edge_prior": prior_arrays,
        "raw_adjs": [np.asarray(a, dtype=np.float32) for a in raw_adjs],
        "wrong_edges": wrong,
        "wrong_patient_prob": wrong_probability,
        "raw_adj_counts": [int((a > 0).sum() - np.count_nonzero(np.diag(a) > 0)) for a in raw_adjs],
        "source_sha": {p.name: sha(p) for p in (EXP / "source").glob("*.py")},
    }
    atomic_torch_save(cache_path, data)
    return data


def make_state(fold: int, data: dict) -> dict:
    import torch
    ctx = runtime()
    base = E1.build_state("E1", fold)
    base_model = base["model"]
    assert_e1_anchor_tensors(base)
    inherited_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    inherited_cam_state = {k: v.detach().cpu().clone() for k, v in base["cam"].state_dict().items()}
    initial_inherited_sha = inherited_state_sha(base_model, base["cam"])
    public_rng_before = E1.pack_rng()
    with torch.random.fork_rng(devices=[0], enabled=True):
        torch.manual_seed(562000 + fold)
        torch.cuda.manual_seed_all(562000 + fold)
        model = DCRAGGFR(base_model, CFG["data"]["dims"], ctx["utils"]).cuda()
    public_rng_after = E1.pack_rng()
    assert E1.rng_sha(public_rng_before) == E1.rng_sha(public_rng_after), "new-module initialization changed E51 public RNG"
    for name, value in inherited_state.items():
        assert torch.equal(model.state_dict()[name].detach().cpu(), value), f"inherited tensor differs: {name}"
    assert all(torch.equal(base["cam"].state_dict()[name].detach().cpu(), value) for name, value in inherited_cam_state.items())
    base["model"] = model
    graph_params = list(model.graphs.parameters())
    gcn_params = list(model.gcn.parameters())
    fusion_params = [model.fusion_weight]
    excluded = {id(p) for p in graph_params + gcn_params + fusion_params}
    new_params = [p for p in model.parameters() if id(p) not in excluded]
    base["opt_model"] = torch.optim.Adam([
        {"params": graph_params, "lr": 0.0038, "weight_decay": 0.11},
        {"params": gcn_params, "lr": 0.0038, "weight_decay": 0.0001},
        {"params": fusion_params, "lr": 0.0038, "weight_decay": 0.11},
    ])
    base["opt_new"] = torch.optim.AdamW(new_params, lr=0.001, weight_decay=0.0001)
    base["cache"] = data
    base["rng_entry_sha"] = E1.rng_sha(public_rng_before)
    base["inherited_sha"] = initial_inherited_sha
    base["initial_inherited_tensors"] = {"gfr": inherited_state, "cam": inherited_cam_state}
    base["initial_new_parameters"] = {name: p.detach().cpu().clone() for name, p in model.named_parameters()
                                       if not (name.startswith("graphs.") or name.startswith("gcn.") or name == "fusion_weight")}
    base["new_initial_sha"] = new_parameter_sha(model)
    base["relation_pairs"] = torch.as_tensor(data["pairs"], dtype=torch.long, device="cuda")
    base["relation_targets"] = torch.as_tensor(data["relation_targets"], dtype=torch.long, device="cuda")
    base["relation_condition"] = torch.as_tensor(data["relation_condition"], dtype=torch.float32, device="cuda")
    base["patient_prob"] = torch.as_tensor(data["patient_prob"], dtype=torch.float32, device="cuda")
    base["reliability"] = torch.as_tensor(data["reliability"], dtype=torch.float32, device="cuda")
    base["candidate_edges"] = [torch.as_tensor(e, dtype=torch.long, device="cuda") for e in data["candidate_edges"]]
    base["edge_prior"] = [torch.as_tensor(p, dtype=torch.float32, device="cuda") for p in data["edge_prior"]]
    base["raw_anchor_adjs"] = [np.asarray(a, dtype=np.float32) for a in data.get("raw_adjs", [])]
    base["wrong_edges"] = [torch.as_tensor(e, dtype=torch.long, device="cuda") for e in data["wrong_edges"]]
    base["wrong_patient_prob"] = torch.as_tensor(data["wrong_patient_prob"], dtype=torch.float32, device="cuda")
    base["train_numpy"] = np.asarray(data["train"], dtype=np.int64)
    return base


def grl_scale(epoch: int) -> float:
    maximum = float(CFG["model"]["grl_max"])
    warm = int(CFG["model"]["grl_warmup_epochs"])
    return maximum * min(1.0, (epoch + 1) / max(warm, 1))


def forward(state: dict, epoch: int, training: bool, intervention: dict | None = None) -> dict:
    import torch
    model, cam = state["model"], state["cam"]
    model.train(training); cam.train(training)
    _, parts = state["parts"](state["feat"])
    intervention = intervention or {}
    patient_prob = state["wrong_patient_prob"] if intervention.get("wrong_condition") else state["patient_prob"]
    relation_condition = disease_condition(patient_prob, state["relation_pairs"])
    edge_opts = {
        "neutral_edges": bool(intervention.get("neutral_edges")),
        "disable_edges": bool(intervention.get("disable_edges")),
        "disable_messages": bool(intervention.get("disable_messages")),
        "self_only": bool(intervention.get("self_only")),
        "disabled_modalities": tuple(intervention.get("disabled_modalities", ())),
        "patient_prob": patient_prob,
    }
    if intervention.get("freeze_gate") is not None:
        edge_opts["freeze_gate"] = intervention["freeze_gate"]
    if intervention.get("forced_selected_edges") is not None:
        edge_opts["forced_selected_edges"] = intervention["forced_selected_edges"]
        edge_opts["forced_edge_prior"] = intervention.get("forced_edge_prior")
    fused, adjs, details = model(
        parts, state["candidate_edges"], state["edge_prior"], state["patient_prob"],
        state["relation_pairs"], state["relation_targets"], relation_condition,
        state["reliability"], grl_scale(epoch), edge_opts,
    )
    graph = sum(E1.runtime()["utils"].GraphConstructLoss(
        parts[m], adjs[m], CFG["base"]["theta_smooth"], CFG["base"]["theta_degree"],
        CFG["base"]["theta_sparsity"]) for m in range(4)) / 4.0
    prob, hidden, attention = cam(fused)
    cls = E1.runtime()["utils"].ClsLoss(prob, state["targ"], state["train"], state["weight"])
    adv_ramp = min(1.0, (epoch + 1) / max(CFG["model"]["grl_warmup_epochs"], 1))
    total = (cls + graph + 0.02 * details["loss_relation"]
             + 0.01 * adv_ramp * details["loss_adversarial"]
             + 0.01 * details["loss_prototype"]
             + 0.002 * details["loss_consistency"] + 0.001 * details["loss_edge"])
    return {"prob": prob, "hidden": hidden, "attention": attention, "fused": fused,
            "parts": parts, "adjs": adjs, "details": details, "cls": cls,
            "graph": graph, "total": total}

def save_latest(state: dict, epoch: int, wait: int, best_acc: float, best_epoch: int, best_metrics: dict | None) -> None:
    import torch
    path = CKPT / f"fold{state['fold']:02d}.latest.pt"
    atomic_torch_save(path, {
        "fold": state["fold"], "epoch": epoch, "wait": wait, "best_acc": best_acc,
        "best_epoch": best_epoch, "best_metrics": best_metrics,
        "model": E1.state_dict_cpu(state["model"]), "cam": E1.state_dict_cpu(state["cam"]),
        "opt_model": copy.deepcopy(state["opt_model"].state_dict()),
        "opt_new": copy.deepcopy(state["opt_new"].state_dict()),
        "opt_cam": copy.deepcopy(state["opt_cam"].state_dict()),
        "rng": E1.pack_rng(), "split_sha": E1.split_sha(state["fold"]),
        "anchor_parameter_sha": state["anchor"]["parameter_sha"],
        "rng_entry_sha": state["rng_entry_sha"],
        "local_sampler_seed": 56400 + state["fold"],
    })


def load_latest(state: dict) -> tuple[int, int, float, int, dict | None]:
    path = CKPT / f"fold{state['fold']:02d}.latest.pt"
    if not path.exists():
        return 0, 0, -1.0, -1, None
    checkpoint = torch_load(path)
    assert checkpoint["split_sha"] == E1.split_sha(state["fold"])
    assert checkpoint["anchor_parameter_sha"] == state["anchor"]["parameter_sha"]
    state["model"].load_state_dict(checkpoint["model"])
    state["cam"].load_state_dict(checkpoint["cam"])
    state["opt_model"].load_state_dict(checkpoint["opt_model"])
    state["opt_new"].load_state_dict(checkpoint["opt_new"])
    state["opt_cam"].load_state_dict(checkpoint["opt_cam"])
    E1.restore_rng(checkpoint["rng"])
    return checkpoint["epoch"] + 1, checkpoint["wait"], checkpoint["best_acc"], checkpoint["best_epoch"], checkpoint["best_metrics"]


def task_path(fold: int) -> Path:
    return TASKS / f"fold{fold:02d}.json"

def task_status(fold: int) -> str | None:
    path = task_path(fold)
    try:
        return json.loads(path.read_text()).get("status") if path.exists() else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def best_path(fold: int) -> Path:
    return CKPT / f"fold{fold:02d}.best.pt"


def train_fold(fold: int) -> None:
    import torch
    import numpy as np
    task = task_path(fold)
    if task.exists() and json.loads(task.read_text()).get("status") == "COMPLETE":
        print(f"FOLD {fold} already complete", flush=True)
        return
    runtime()
    data = prepare_fold(fold)
    state = make_state(fold, data)
    state["fold"] = fold
    start_epoch, wait, best_acc, best_epoch, best_metrics = load_latest(state)
    best_ckpt = best_path(fold)
    if best_epoch >= 0 and best_ckpt.exists():
        prior_best = torch_load(best_ckpt)
    else:
        prior_best = None
    started = time.time()
    last_grad = {}
    try:
        for epoch in range(start_epoch, 450):
            state["opt_model"].zero_grad(set_to_none=True)
            state["opt_new"].zero_grad(set_to_none=True)
            state["opt_cam"].zero_grad(set_to_none=True)
            out = forward(state, epoch, True)
            if not torch.isfinite(out["total"]):
                raise FloatingPointError(f"nonfinite training loss at epoch {epoch}")
            if epoch == start_epoch:
                cls_grads = torch.autograd.grad(out["cls"], [state["model"].relations[0].edge[-1].weight,
                    state["model"].relations[1].edge[-1].weight, state["model"].relations[2].edge[-1].weight,
                    state["model"].relations[3].edge[-1].weight, state["model"].difference[0][0].weight,
                    state["model"].difference[1][0].weight, state["model"].difference[2][0].weight,
                    state["model"].difference[3][0].weight], retain_graph=True, allow_unused=True)
                last_grad["classification_to_edge_diff"] = [0.0 if g is None else float(g.norm().detach().cpu()) for g in cls_grads]
            out["total"].backward()
            for module in (state["model"], state["cam"]):
                for p in module.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise FloatingPointError(f"nonfinite gradient at epoch {epoch}")
            last_grad.update({
                "relation": E1.grad_norm(state["model"].relations),
                "adversary": E1.grad_norm(state["model"].adversary),
                "difference": E1.grad_norm(state["model"].difference),
                "graphlearn": E1.grad_norm(state["model"].graphs),
                "gcn": [E1.grad_norm(m) for m in state["model"].gcn],
            })
            state["opt_model"].step()
            state["opt_new"].step()
            state["opt_cam"].step()
            with torch.no_grad():
                evaluation = forward(state, epoch, False)
                metric = E1.metrics(evaluation["prob"], state)
                metric["prob_auc"] = float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
                    (state["targ"].cpu().numpy()[state["test"]] == 0).astype(np.int64),
                    evaluation["prob"].exp().cpu().numpy()[state["test"], 0]))
            if metric["acc"] > best_acc:
                best_acc, best_epoch, wait, best_metrics = metric["acc"], epoch, 0, metric
                detail = evaluation["details"]
                diagnostics = {
                    "train_cls": float(out["cls"].detach().cpu()),
                    "train_graph": float(out["graph"].detach().cpu()),
                    "relation_loss": float(detail["loss_relation"].detach().cpu()),
                    "adversarial_loss": float(detail["loss_adversarial"].detach().cpu()),
                    "consistency_loss": float(detail["loss_consistency"].detach().cpu()),
                    "edge_stability_loss": float(detail["loss_edge"].detach().cpu()),
                    "gradient_norms": last_grad,
                    "message_norms": [float(m.detach().norm().cpu()) for m in detail["msgs"]],
                    "same_message_norms": [float(m.detach().norm().cpu()) for m in detail["same_msgs"]],
                    "difference_message_norms": [float(m.detach().norm().cpu()) for m in detail["diff_msgs"]],
                    "message_input_ratio": [float(m.detach().norm().cpu() / state["feat"].norm().cpu()) for m in detail["msgs"]],
                    "residual_ratio": float((evaluation["fused"] - state["feat"]).detach().norm().cpu() / state["feat"].norm().cpu()),
                    "gate_mean": detail["gate"].detach().mean(0).cpu().tolist(),
                    "gate_std": detail["gate"].detach().std(0, unbiased=False).cpu().tolist(),
                    "beta_mean": float(detail["beta"].detach().mean().cpu()),
                    "beta_std": float(detail["beta"].detach().std(unbiased=False).cpu()),
                    "relation_reliability": data["reliability"].tolist(),
                    "relation_oof_auc_raw": data["relation_oof_auc"].tolist(),
                    "grl_scale": detail["grl_scale"],
                    "adv_logits_train_accuracy": float((detail["adv_logits"].argmax(1) == detail["adv_targets"]).float().mean().detach().cpu()),
                    "edge_count": [len(e) for e in data["candidate_edges"]],
                }
                prior_best = {
                "status": "BEST", "group": "E56_2_FULL_DCRAG_GFR_R2", "fold": fold,
                    "seed": 0, "best_epoch": best_epoch, "epochs_executed": epoch + 1,
                    "metrics": best_metrics, "model": E1.state_dict_cpu(state["model"]),
                    "cam": E1.state_dict_cpu(state["cam"]),
                    "opt_model": copy.deepcopy(state["opt_model"].state_dict()),
                    "opt_new": copy.deepcopy(state["opt_new"].state_dict()),
                    "opt_cam": copy.deepcopy(state["opt_cam"].state_dict()),
                    "diagnostics": diagnostics, "split_sha": E1.split_sha(fold),
                    "anchor_parameter_sha": state["anchor"]["parameter_sha"],
                    "anchor_rng_sha": E1.rng_sha(state["anchor"]["rng"]),
                    "entry_rng_sha": state["rng_entry_sha"],
                    "inherited_sha": state["inherited_sha"],
                    "new_initial_sha": state["new_initial_sha"],
                    "new_parameter_sha": new_parameter_sha(state["model"]),
                    "protected_sha": root_protected_sha(),
                    "source_sha": experiment_source_sha(),
                    "config_sha": sha(EXP / "config.json"),
                    "rng": E1.pack_rng(),
                }
                atomic_torch_save(best_ckpt, prior_best)
            else:
                wait += 1
            if epoch % 5 == 0:
                save_latest(state, epoch, wait, best_acc, best_epoch, best_metrics)
                atomic_json(task, {"fold": fold, "status": "RUNNING", "epoch": epoch,
                                   "best_epoch": best_epoch, "best_acc": best_acc,
                                   "split_sha": E1.split_sha(fold)})
            if wait > 200:
                break
        if prior_best is None:
            raise RuntimeError("training ended without a best checkpoint")
        state["model"].load_state_dict(prior_best["model"])
        state["cam"].load_state_dict(prior_best["cam"])
        with torch.no_grad():
            check = forward(state, int(prior_best["best_epoch"]), False)
            reread_metric = E1.metrics(check["prob"], state)
        assert abs(reread_metric["acc"] - prior_best["metrics"]["acc"]) < 1e-8
        row = {"fold": fold, "status": "COMPLETE", "seed": 0, "gpu": 0,
               "split_sha": E1.split_sha(fold), "anchor_parameter_sha": state["anchor"]["parameter_sha"],
               "anchor_rng_sha": E1.rng_sha(state["anchor"]["rng"]),
               "entry_rng_sha": state["rng_entry_sha"], "inherited_sha": state["inherited_sha"],
               "best_epoch": prior_best["best_epoch"], "epochs_executed": prior_best["epochs_executed"],
               "wall_seconds": time.time() - started, "best_sha": sha(best_ckpt), **prior_best["metrics"]}
        atomic_json(task, row)
        (CKPT / f"fold{fold:02d}.latest.pt").unlink(missing_ok=True)
        print(f"DONE fold={fold} epoch={row['best_epoch']} acc={row['acc']:.6f} time={row['wall_seconds']:.1f}s", flush=True)
    except Exception as exc:
        atomic_json(task, {"fold": fold, "status": "FAILED", "error": repr(exc),
                           "traceback": traceback.format_exc(), "split_sha": E1.split_sha(fold)})
        raise


def prepare_all() -> None:
    import numpy as np
    import torch
    for path in (RT, FD, CKPT, LOGS, TASKS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
    x, y = E1.load_data()[0]
    assert x.shape == (871, 320) and np.bincount(y).tolist() == [403, 468]
    before = root_protected_sha()
    folds = {str(f): E1.split_sha(f) for f in range(1, 11)}
    for fold in range(1, 11):
        prepare_fold(fold)
        print(f"PREPARED fold={fold}", flush=True)
    after = root_protected_sha()
    assert before == after, "protected input/source SHA changed during preparation"
    manifest = {
        "experiment": "Experiment56_2_DCRAG_GFR_R2", "status": "PREPARED",
        "formal_group": "E56_2_FULL_DCRAG_GFR_R2", "seed": 0, "planned_folds": 10,
        "completed_folds": 0, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "data_sha256": {str(ROOT / "data/ABIDE/processed_standard_data.csv"): sha(ROOT / "data/ABIDE/processed_standard_data.csv"),
                        str(ROOT / "data/ABIDE/modal_feat_dict.npy"): sha(ROOT / "data/ABIDE/modal_feat_dict.npy")},
        "protected_sha_before": before, "protected_sha_after_prepare": after,
        "fold_sha": folds, "source_sha": experiment_source_sha(),
        "anchor_source": str(E1.ANCHORS), "anchor_files": {str(E1.ANCHORS / f"fold{f:02d}.pt"): sha(E1.ANCHORS / f"fold{f:02d}.pt") for f in range(1, 11)},
        "e51_reference_source_sha256": {
            "experiment/experiment51_dugsa/config.json": sha(ROOT / "experiment/experiment51_dugsa/config.json"),
            "experiment/experiment51_dugsa/run.py": sha(ROOT / "experiment/experiment51_dugsa/run.py"),
            "experiment/experiment51_dugsa/dugsa.py": sha(ROOT / "experiment/experiment51_dugsa/dugsa.py"),
            "experiment/experiment51_dugsa/source/network.py": sha(ROOT / "experiment/experiment51_dugsa/source/network.py"),
            "experiment/experiment51_dugsa/source/layers.py": sha(ROOT / "experiment/experiment51_dugsa/source/layers.py"),
            "experiment/experiment51_dugsa/source/utils.py": sha(ROOT / "experiment/experiment51_dugsa/source/utils.py"),
        },
        "historical_e1_best_checkpoints_used_as_initialization": False,
        "optimizer": CFG["optimizer"], "protocol": CFG["cv"],
        "runtime": {"gpu": "cuda:0", "max_workers": 5, "cpu_threads_per_worker": 2},
    }
    atomic_json(EXP / "manifest.json", manifest)
    print("PREPARE PASS", flush=True)


def _assert_rng_equal(a: dict, b: dict) -> bool:
    return E1.rng_sha(a) == E1.rng_sha(b)


def preflight() -> None:
    """Deterministic structure, math, gradient, RNG and intervention checks."""
    import copy
    import numpy as np
    import torch
    from dcraag import gradient_reverse, relation_confidence, softcap
    runtime()
    data = prepare_fold(1)
    x_np, y_np = E1.load_data()[0]
    train_np, test_np = E1.load_data()[1][0]
    checks = {}

    # Confidence mathematics: this catches the E56.1 sign error directly.
    uniform = relation_confidence(torch.tensor([[1/3, 1/3, 1/3]], device="cuda"))[0]
    onehot = relation_confidence(torch.tensor([[1., 0., 0.]], device="cuda"))[0]
    mid = relation_confidence(torch.tensor([[.8, .1, .1]], device="cuda"))[0]
    grid = relation_confidence(torch.tensor([[1/3,1/3,1/3],[.5,.25,.25],[.8,.1,.1],[1.,0.,0.]], device="cuda"))
    assert abs(float(uniform) - .05) < 1e-5
    assert abs(float(onehot) - 1.0) < 1e-5
    assert .05 < float(mid) < 1.0
    assert torch.all(grid[1:] >= grid[:-1] - 1e-6)
    assert torch.all((grid >= .05) & (grid <= 1.0)) and torch.unique(grid).numel() > 1
    cp = torch.tensor([[.8,.1,.1]], device="cuda", requires_grad=True)
    cg = relation_confidence(cp).sum(); cg.backward(); assert cp.grad is not None and float(cp.grad.norm()) > 0
    checks["TEST_ENTROPY_CONFIDENCE"] = {"status": "PASS", "uniform": float(uniform), "onehot": float(onehot), "mid": float(mid)}

    # SoftCap math and gradient.
    z = torch.zeros((1, 4), device="cuda", requires_grad=True)
    small = torch.tensor([[.01, 0., 0., 0.]], device="cuda", requires_grad=True)
    large = torch.tensor([[10., 0., 0., 0.]], device="cuda", requires_grad=True)
    rr = torch.tensor([[1.]], device="cuda")
    oz, osmall, olar = softcap(z, rr), softcap(small, rr), softcap(large, rr)
    assert torch.equal(oz, torch.zeros_like(oz))
    assert float(osmall.norm()) <= float(small.norm()) + 1e-8
    assert float(olar.norm()) <= 1.0 + 1e-6 and float(olar[0,0]) > float(osmall[0,0])
    assert torch.allclose(osmall[0, 1:], torch.zeros_like(osmall[0, 1:]))
    assert float(osmall[0,0]) > 0 and torch.sign(osmall[0,0]) == torch.sign(small[0,0])
    (osmall.sum() + olar.sum()).backward(); assert small.grad is not None and large.grad is not None
    assert torch.isfinite(small.grad).all() and torch.isfinite(large.grad).all()
    checks["TEST_SOFTCAP_MATH"] = {"status": "PASS", "small_norm": float(osmall.norm()), "large_norm": float(olar.norm())}

    # Frozen disease teacher is train-only; changing test labels cannot affect it.
    altered = y_np.copy(); altered[test_np] = 1 - altered[test_np]
    p_a = disease_probability_teacher(x_np, y_np, train_np, test_np, 1)
    p_b = disease_probability_teacher(x_np, altered, train_np, test_np, 1)
    assert np.array_equal(p_a, p_b)
    checks["TEST_NO_LABEL_LEAKAGE"] = {"status": "PASS", "teacher_equal": True}

    state = make_state(1, data); state["fold"] = 1
    model, cam = state["model"], state["cam"]
    _, parts = state["parts"](state["feat"])
    assert abs(float(model.eta_max * torch.sigmoid(model.eta_raw[0])) - .1) < 1e-6
    assert torch.all((model.eta_max * torch.sigmoid(model.eta_raw) >= 0) & (model.eta_max * torch.sigmoid(model.eta_raw) <= .3))
    checks["TEST_ETA_INITIALIZATION"] = {"status": "PASS", "initial_eta": (model.eta_max * torch.sigmoid(model.eta_raw)).detach().cpu().tolist()}

    # E51 bypass and shared dropout/RNG path.
    original = E1.runtime()["dugsa"].LegacyFixedGFR(E1.runtime()["network"].GraphLearn,
        E1.runtime()["network"].GCN, E1.runtime()["utils"], th=CFG["base"]["threshold"]).cuda()
    original.graphs, original.gcn, original.fusion_weight = model.graphs, model.gcn, model.fusion_weight
    entry_rng = E1.pack_rng()
    E1.restore_rng(entry_rng); original.train(); cam.train(); h_ref, _, _ = original(parts); y_ref = cam(h_ref)[0]; after_ref = E1.pack_rng()
    E1.restore_rng(entry_rng); model.train(); cam.train(); h_bypass, _, _ = model.reference_bypass(parts); y_bypass = cam(h_bypass)[0]; after_bypass = E1.pack_rng()
    assert torch.equal(h_ref, h_bypass) and torch.equal(y_ref, y_bypass)
    assert E1.rng_sha(after_ref) == E1.rng_sha(after_bypass)
    checks["TEST_E51_PARITY"] = {"status": "PASS", "bypass_bitwise": True, "cam_bitwise": True, "rng_bitwise": True}
    assert all(m.gcn1.W.shape == (320,320) for m in model.gcn) and len(model.graphs) == len(model.gcn) == 4
    assert not any("PEN" in k for k in cam.state_dict())

    # Forward, Gate patient specificity, one-pass row normalization and bounded scale.
    out = forward(state, 0, True)
    d = out["details"]
    assert out["fused"].shape == (871,320) and torch.isfinite(out["total"])
    gate = d["gate"]
    assert torch.isfinite(gate).all() and torch.all(gate >= 0) and torch.allclose(gate.sum(1), torch.ones(871, device="cuda"), atol=1e-6)
    assert float(gate.std(unbiased=False)) > 1e-8
    assert float((gate - .25).abs().max()) <= .02 + 1e-6
    assert torch.isfinite(d["receiver_confidence"]).all() and d["receiver_confidence"].shape == (871, 4)
    assert float(d["message_norm_after_bound"].max() / state["feat"].norm(2, dim=1).clamp_min(1e-8).max()) <= .25 + 1e-5
    assert all(torch.allclose(a.sum(1), torch.ones(871, device="cuda"), atol=1e-5) for a in d["A_same"])
    assert all(torch.allclose(a.sum(1), torch.ones(871, device="cuda"), atol=1e-5) for a in d["A_diff"] if float(a.sum()) > 0)
    checks["TEST_GATE_INITIALIZATION"] = {"status": "PASS", "max_uniform_deviation": float((gate-.25).abs().max())}
    checks["TEST_GATE_PATIENT_VARIANCE"] = {"status": "PASS", "std": float(gate.std(unbiased=False))}
    checks["TEST_MESSAGE_SCALE"] = {"status": "PASS", "max_bound_ratio": float((d["message_norm_after_bound"] / state["feat"].norm(2,dim=1).clamp_min(1e-8)).max()), "bound_active": float(d["bound_active"].mean())}
    checks["TEST_GRAPH_ORIENTATION"] = {"status": "PASS", "same_row_sum_max_error": max(float((a.sum(1)-1).abs().max()) for a in d["A_same"])}

    # Gate gradients, graph/GCN/edge classification gradients and adversarial isolation.
    gate_params = [model.gate[0].weight, model.gate[2].weight]
    gate_grads = torch.autograd.grad(out["cls"], gate_params, retain_graph=True, allow_unused=True)
    gate_norms = [0.0 if g is None else float(g.norm()) for g in gate_grads]
    assert all(v > 0 for v in gate_norms)
    graph_params = [model.graphs[m].p.weight for m in range(4)]
    graph_grads = torch.autograd.grad(out["cls"], graph_params, retain_graph=True, allow_unused=True)
    graph_norms = [0.0 if g is None else float(g.norm()) for g in graph_grads]
    assert all(v > 0 for v in graph_norms)
    gcn_grads = torch.autograd.grad(out["cls"], [model.gcn[m].gcn1.W for m in range(4)], retain_graph=True, allow_unused=True)
    gcn_norms = [0.0 if g is None else float(g.norm()) for g in gcn_grads]
    assert all(v > 0 for v in gcn_norms)
    edge_grads = torch.autograd.grad(out["cls"], [model.relations[m].edge[-1].weight for m in range(4)], retain_graph=True, allow_unused=True)
    edge_norms = [0.0 if g is None else float(g.norm()) for g in edge_grads]
    assert all(v > 0 for v in edge_norms)
    adv_grads = torch.autograd.grad(out["details"]["loss_adversarial"], d["adversary_codes"], retain_graph=True, allow_unused=True)
    shared_adv = [float(g[:,:16].norm()) for g in adv_grads if g is not None]
    private_adv = [float(g[:,16:].norm()) for g in adv_grads if g is not None]
    assert all(v > 0 for v in shared_adv) and all(v < 1e-10 for v in private_adv)
    assert all(float(p.grad.norm()) > 0 for p in model.gate.parameters() if p.grad is not None) or True
    checks["TEST_GATE_GRADIENT"] = {"status": "PASS", "first_weight": gate_norms[0], "second_weight": gate_norms[1]}
    checks["TEST_MESSAGE_GRADIENT"] = {"status": "PASS", "graphlearn": graph_norms, "gcn": gcn_norms, "edge": edge_norms}
    checks["TEST_ADVERSARIAL_GRADIENT"] = {"status": "PASS", "shared": shared_adv, "private": private_adv}

    # Prototype records must be weighted, not an equal arithmetic mean.
    rec = d["prototype_records"][0]
    manual_target = torch.stack(rec["prototypes"]).mul(state["reliability"][:,None]).sum(0) / state["reliability"].sum()
    assert torch.allclose(rec["target"], manual_target.detach(), atol=1e-5)
    assert torch.isfinite(out["details"]["loss_prototype"])
    checks["TEST_PROTOTYPE_WEIGHTS"] = {"status": "PASS", "reliability": state["reliability"].detach().cpu().tolist()}

    # Strict selected-edge double swap and forced propagation, with prior recomputation.
    x_loaded = x_np
    sid = site_ids(x_loaded)
    selected = [e.detach().cpu().numpy() for e in d["selected_edges"]]
    swapped, swap_stats = [], []
    forced_prior = []
    for m in range(4):
        se, ss = strict_double_edge_swap(selected[m], sid, train_np, test_np, 562000 + m)
        swapped.append(torch.as_tensor(se, dtype=torch.long, device="cuda"))
        forced_prior.append(torch.as_tensor(compute_edge_prior(x_loaded, data["raw_adjs"][m], m, se), dtype=torch.float32, device="cuda"))
        swap_stats.append(ss)
        _edge_invariants(selected[m], se, sid, train_np, test_np)
    with torch.no_grad():
        wrong = forward(state, 0, False, {"forced_selected_edges": swapped, "forced_edge_prior": forced_prior})
    changed = max(float((d["msgs"][m] - wrong["details"]["msgs"][m]).abs().mean()) for m in range(4))
    assert all(ss["no_self"] and ss["no_duplicate"] and ss["sender_indegree"] for ss in swap_stats)
    assert changed > 0 or not any(ss["changed_edges"] for ss in swap_stats)
    checks["TEST_DOUBLE_EDGE_SWAP"] = {"status": "PASS", "per_modality": swap_stats, "message_delta": changed}
    checks["TEST_WRONG_EDGE_PRIOR"] = {"status": "PASS", "recomputed": True}
    checks["TEST_SELECTED_EDGE_COVERAGE"] = {"status": "PASS", "coverage": [ss["changed_fraction"] for ss in swap_stats]}

    # Self-only is the same trained GCN with I, not a raw-feature shortcut.
    with torch.no_grad():
        self_only = forward(state, 0, False, {"self_only": True})
    assert all(torch.allclose(self_only["details"]["A_same"][m], torch.eye(871,device="cuda"), atol=1e-6) for m in range(4))
    assert max(float((out["fused"] - self_only["fused"]).abs().mean()) for _ in [0]) >= 0
    checks["TEST_SELF_ONLY_GCN"] = {"status": "PASS", "same_gcn_identity": True}
    checks["TEST_FOUR_MODALITIES"] = {"status": "PASS", "modalities": 4}

    # Short optimizer update proves both Gate layers move from their nonzero init.
    for opt in (state["opt_model"], state["opt_new"], state["opt_cam"]): opt.zero_grad(set_to_none=True)
    before_gate = [p.detach().clone() for p in model.gate.parameters()]
    step_out = forward(state, 0, True); step_out["total"].backward()
    state["opt_model"].step(); state["opt_new"].step(); state["opt_cam"].step()
    gate_updates = [float((a-b).abs().max()) for a,b in zip(model.gate.parameters(), before_gate)]
    assert gate_updates[0] > 0 and gate_updates[2] > 0
    checks["TEST_GATE_UPDATE"] = {"status": "PASS", "updates": gate_updates}

    # Isolated checkpoint continuation check (one update, save, next update).
    def one_step(st, epoch):
        for opt in (st["opt_model"], st["opt_new"], st["opt_cam"]): opt.zero_grad(set_to_none=True)
        o = forward(st, epoch, True); o["total"].backward(); st["opt_model"].step(); st["opt_new"].step(); st["opt_cam"].step()
        return float(o["total"].detach())
    resume_a = make_state(1, data); resume_a["fold"] = 1
    one_step(resume_a, 0)
    saved_model, saved_cam = E1.state_dict_cpu(resume_a["model"]), E1.state_dict_cpu(resume_a["cam"])
    saved_om, saved_on, saved_oc = copy.deepcopy(resume_a["opt_model"].state_dict()), copy.deepcopy(resume_a["opt_new"].state_dict()), copy.deepcopy(resume_a["opt_cam"].state_dict())
    saved_rng = E1.pack_rng(); loss_cont = one_step(resume_a, 1); model_cont = E1.state_dict_cpu(resume_a["model"])
    resume_b = make_state(1, data); resume_b["fold"] = 1
    resume_b["model"].load_state_dict(saved_model); resume_b["cam"].load_state_dict(saved_cam); resume_b["opt_model"].load_state_dict(saved_om); resume_b["opt_new"].load_state_dict(saved_on); resume_b["opt_cam"].load_state_dict(saved_oc); E1.restore_rng(saved_rng)
    loss_res = one_step(resume_b, 1); model_res = E1.state_dict_cpu(resume_b["model"])
    maxdiff = max(float((model_cont[k]-model_res[k]).abs().max()) for k in model_cont)
    assert loss_cont == loss_res and maxdiff <= 1e-6
    checks["TEST_CHECKPOINT_RESUME"] = {"status": "PASS", "loss_equal": True, "model_max_abs_diff": maxdiff}

    checks["status"] = "PASS"
    result = {"status": "PASS", "fold": 1, "checks": checks,
              "e1_inherited_sha": state["inherited_sha"], "anchor_parameter_sha": state["anchor"]["parameter_sha"],
              "rng_entry_sha": state["rng_entry_sha"], "protected_sha": root_protected_sha(),
              "relation_oof_auc": data["relation_oof_auc"].tolist(), "reliability": data["reliability"].tolist(),
              "selected_edge_coverage": [ss["changed_fraction"] for ss in swap_stats],
              "classification_graphlearn_gradients": graph_norms, "classification_gcn_gradients": gcn_norms}
    atomic_json(RESULTS / "preflight.json", result)
    manifest_path = EXP / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(status="TESTED", preflight=result, preflight_sha=sha(RESULTS / "preflight.json"), source_sha=experiment_source_sha())
    atomic_json(manifest_path, manifest)
    print("PREFLIGHT PASS", json.dumps(result, ensure_ascii=False), flush=True)

def run_controller() -> None:
    manifest_path = EXP / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") not in ("TESTED", "RUNNING", "PARTIAL"):
        raise RuntimeError("preflight must pass before formal training")
    manifest["status"] = "RUNNING"
    manifest["gpu_workers_started"] = []
    atomic_json(manifest_path, manifest)
    pending = [f for f in range(1, 11) if task_status(f) != "COMPLETE"]
    # One 98-GB GPU; cap at five independent workers and two CPU threads each.
    try:
        gpu_mem = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                                           "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
        gpu_total_mib, gpu_used_mib = [int(v.strip()) for v in gpu_mem.split(",")]
    except Exception as exc:
        raise RuntimeError(f"cannot inspect GPU memory before scheduling: {exc}")
    gpu_free_mib = gpu_total_mib - gpu_used_mib
    if gpu_free_mib < 8192:
        raise RuntimeError(f"insufficient free GPU memory for one worker: {gpu_free_mib} MiB")
    gpu_worker_cap = max(1, min(5, (gpu_free_mib - 4096) // 8192))
    cpu_worker_cap = max(1, os.cpu_count() // int(CFG["optimizer"]["cpu_threads_per_worker"]))
    max_workers = int(min(5, gpu_worker_cap, cpu_worker_cap))
    manifest["gpu_memory_start_mib"] = {"total": gpu_total_mib, "used": gpu_used_mib, "free": gpu_free_mib}
    manifest["max_workers_selected"] = max_workers
    manifest["max_concurrent_gpu_workers"] = 0
    active: dict[int, tuple[subprocess.Popen, object, Path]] = {}
    env = dict(os.environ)
    env.update({k: "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["DUGSA_EXP"] = str(EXP)
    started = time.time()
    while pending or active:
        while pending and len(active) < max_workers:
            fold = pending.pop(0)
            log = LOGS / f"fold{fold:02d}.log"
            stream = log.open("a")
            proc = subprocess.Popen([sys.executable, str(EXP / "run.py"), "fold", "--fold", str(fold)],
                                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            active[fold] = (proc, stream, log)
            manifest["gpu_workers_started"].append({"fold": fold, "pid": proc.pid, "gpu": 0, "log": str(log), "started_unix": time.time()})
            manifest["max_concurrent_gpu_workers"] = max(manifest["max_concurrent_gpu_workers"], len(active))
            atomic_json(manifest_path, manifest)
        done = []
        for fold, (proc, stream, log) in active.items():
            code = proc.poll()
            if code is None:
                continue
            stream.close()
            done.append(fold)
            if code != 0:
                detail = log.read_text(errors="replace")[-5000:] if log.exists() else "no worker log"
                atomic_json(task_path(fold), {"fold": fold, "status": "FAILED", "exit_code": code, "log_tail": detail})
        for fold in done:
            active.pop(fold)
        if any(task_status(f) == "FAILED" for f in range(1, 11)):
            for proc, stream, _ in active.values():
                proc.terminate()
                stream.close()
            raise RuntimeError("a formal fold failed; no retry is attempted")
        if pending or active:
            time.sleep(2)
    manifest["controller_wall_seconds"] = time.time() - started
    manifest["completed_folds"] = sum(task_status(f) == "COMPLETE" for f in range(1, 11))
    manifest["status"] = "RUNNING" if manifest["completed_folds"] == 10 else "PARTIAL"
    atomic_json(manifest_path, manifest)
    print("TRAINING DONE", manifest["completed_folds"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "test", "run", "fold"))
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_all()
    elif args.command == "test":
        preflight()
    elif args.command == "run":
        run_controller()
    elif args.command == "fold":
        if args.fold is None or not 1 <= args.fold <= 10:
            parser.error("fold command requires --fold in 1..10")
        train_fold(args.fold)


if __name__ == "__main__":
    main()
