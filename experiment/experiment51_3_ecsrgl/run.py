#!/usr/bin/env python3
"""Experiment 51.3 EC-SRGL runner.

The controller uses one CUDA device and independent group/fold workers.  The
fold data, candidate pair sets, checkpoints, and task states are all local to
the experiment directory so the run can resume without changing completed
folds.
"""
from __future__ import annotations

import argparse
import copy
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
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(os.environ.get("ECSR_EXP", Path(__file__).resolve().parent))
ROOT = Path("/data/adodas/SPromptGL-main")
RT = EXP / ".runtime"
TASKS = RT / "tasks"
CKPTS = RT / "checkpoints"
LOGS = RT / "logs"
FD = RT / "fold_data"
RESULTS = EXP / "results"
SRC = EXP / "source"
ANCHORS = ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
HIST = ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/checkpoints"
E51_2 = Path("/tmp/SPromptGL-E51_2/experiment51_2_pdcsra_gfr")
E51_2_CKPTS = E51_2 / ".runtime/checkpoints"
CFG_PATH = EXP / "config.json"
CFG = json.loads(CFG_PATH.read_text())
DIMS = [48, 6, 10, 256]
MODS = ["PHENO", "ANAT", "FUNC", "CORR"]
CTX = None
DATA = None
FOLDS = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + f".tmp.{os.getpid()}")
    q.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(q, p)


def torch_save(path, obj):
    import torch
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + f".tmp.{os.getpid()}")
    with q.open("wb") as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    os.replace(q, p)


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def csv_write(path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    q = p.with_name(p.name + f".tmp.{os.getpid()}")
    with q.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(q, p)


def csv_read(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pack_rng():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(r):
    import numpy as np
    import torch
    random.setstate(r["python"]); np.random.set_state(r["numpy"])
    torch.set_rng_state(r["cpu"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in r["cuda"]])


def rng_sha(r):
    h = hashlib.sha256(pickle.dumps((r["python"], r["numpy"]), protocol=4))
    h.update(r["cpu"].cpu().numpy().tobytes())
    for v in r["cuda"]:
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def runtime():
    global CTX
    if CTX is None:
        import torch
        import numpy as np
        import pandas as pd
        torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.cuda.set_device(0)
        sys.path.insert(0, str(SRC)); sys.path.insert(0, str(EXP))
        import layers, network, utils, ecsrgl
        CTX = {"torch": torch, "np": np, "pd": pd, "layers": layers,
               "network": network, "utils": utils, "ecsrgl": ecsrgl,
               "device": torch.device("cuda")}
    return CTX


def load_data():
    global DATA, FOLDS
    if DATA is not None:
        return DATA, FOLDS
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    base = ROOT / "data/ABIDE"
    frame = pd.read_csv(base / "processed_standard_data.csv")
    mapping = np.load(base / "modal_feat_dict.npy", allow_pickle=True).item()
    expected = [str(v) for key in ("PHENO", "ANAT", "FUNC", "Correlation") for v in mapping[key]]
    assert list(frame.columns[:-1]) == expected
    x = frame.iloc[:, :-1].to_numpy(np.float32)
    y = frame.iloc[:, -1].to_numpy(np.int64) - 1
    assert x.shape == (871, 320) and np.isfinite(x).all()
    assert np.bincount(y).tolist() == [403, 468]
    assert [len(mapping[k]) for k in ("PHENO", "ANAT", "FUNC", "Correlation")] == DIMS
    FOLDS = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y))
    DATA = (x, y)
    return DATA, FOLDS


def split_sha(fold):
    import numpy as np
    tr, te = load_data()[1][fold - 1]
    return hashlib.sha256(np.asarray(tr, np.int64).tobytes() + np.asarray(te, np.int64).tobytes()).hexdigest()


def hp():
    b = CFG["base"]
    return SimpleNamespace(n_hidden=b["n_hidden"], n_head=b["n_head"], dropout=b["dropout"],
                           nlayer=b["nlayer"], nmodal=4, nclass=2, lr=b["lr"], reg=b["weight_decay"],
                           th=b["threshold"], GC_mode="weighted-cosine", MP_mode="GCN",
                           MF_mode="concat", alpha=.5, theta_smooth=b["theta_smooth"],
                           theta_degree=b["theta_degree"], theta_sparsity=b["theta_sparsity"])


def model_digest(model):
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters()):
        h.update(n.encode() + b"\0" + p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def anchor_for(fold):
    a = torch_load(ANCHORS / f"fold{fold:02d}.pt")
    assert a["split_sha"] == split_sha(fold)
    return a


def build_initial_base(fold):
    """Build exactly the Experiment 51 E1 pre-training model state."""
    import torch
    c = runtime(); x, y = load_data()[0]; anchor = anchor_for(fold)
    seed_all(510000 + fold * 100 + 1)
    base = c["ecsrgl"].LegacyFixedGFR(c["network"].GraphLearn, c["network"].GCN,
                                      c["utils"], th=CFG["base"]["threshold"]).cuda()
    cam = c["ecsrgl"].PLNNoPEN(DIMS, hp(), c["layers"]).cuda()
    cam.apply(c["utils"].my_weight_init)
    c["ecsrgl"].load_e1_anchor(base, anchor["model"]["gfr"])
    cam.load_state_dict(anchor["model"]["pln"])
    return base, cam, pack_rng()


def site_ids(x):
    import numpy as np
    # PHENO columns 26:46 are the validated one-hot site block in this dataset.
    return np.argmax(x[:, 26:46], axis=1).astype(np.int64)


def split_site_permutation(x, tr, te, fold):
    import numpy as np
    sid = site_ids(x); rng = np.random.default_rng(513000 + fold)
    perm = np.arange(len(x), dtype=np.int64)
    for ids in (np.asarray(tr), np.asarray(te)):
        buckets = {}
        for i in ids:
            buckets.setdefault(int(sid[i]), []).append(int(i))
        for vals in buckets.values():
            if len(vals) > 1:
                arr = np.asarray(vals, dtype=np.int64); sh = arr.copy(); rng.shuffle(sh)
                if np.all(sh == arr):
                    sh = np.roll(arr, 1)
                perm[arr] = sh
    return perm


def _take_stratified(candidates, k, sid, y, rng):
    import numpy as np
    if len(candidates) <= k:
        return np.asarray(candidates, dtype=np.int64)
    # Keep site-matched and site-crossed pairs present where possible.
    sm = np.asarray([int(sid[a]) == int(sid[b]) for a, b in candidates])
    parts = [np.where(sm)[0], np.where(~sm)[0]]
    chosen = []
    quotas = [min(len(parts[0]), k // 2), min(len(parts[1]), k - k // 2)]
    for ids, q in zip(parts, quotas):
        if q:
            chosen.extend(rng.choice(ids, q, replace=False).tolist())
    if len(chosen) < k:
        left = np.setdiff1d(np.arange(len(candidates)), np.asarray(chosen, dtype=np.int64), assume_unique=False)
        chosen.extend(rng.choice(left, k - len(chosen), replace=False).tolist())
    rng.shuffle(chosen)
    return np.asarray(candidates[np.asarray(chosen, dtype=np.int64)], dtype=np.int64)


def sample_supported_pairs(support, tr, y, x, max_pairs, rng):
    import numpy as np
    sid = site_ids(x); train_mask = np.zeros(len(y), dtype=bool); train_mask[np.asarray(tr)] = True
    ii, jj = np.where(np.triu(support, 1) & train_mask[:, None] & train_mask[None, :])
    pairs = np.stack([ii, jj], axis=1) if len(ii) else np.zeros((0, 2), dtype=np.int64)
    same = (y[pairs[:, 0]] == y[pairs[:, 1]]) if len(pairs) else np.zeros(0, dtype=bool)
    pos = pairs[same]; neg = pairs[~same]
    k = min(len(pos), len(neg), max_pairs // 2)
    if k == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float32), {"same": 0, "diff": 0}
    pos = _take_stratified(pos, k, sid, y, rng)
    neg = _take_stratified(neg, k, sid, y, rng)
    out = np.concatenate([pos, neg], axis=0); rng.shuffle(out)
    labels = (y[out[:, 0]] == y[out[:, 1]]).astype(np.float32)
    return out, labels, {"same": int((labels == 1).sum()), "diff": int((labels == 0).sum())}


def balanced_align_ids(tr, y, n, rng):
    import numpy as np
    tr = np.asarray(tr); a = tr[y[tr] == 0]; b = tr[y[tr] == 1]
    k = min(n // 2, len(a), len(b))
    return np.concatenate([np.sort(rng.choice(a, k, replace=False)), np.sort(rng.choice(b, k, replace=False))])


def prepare_fold(fold):
    """Freeze support-respecting pair pools and split-local negative controls."""
    import numpy as np
    import torch
    p = FD / f"fold{fold:02d}.npz"
    meta = FD / f"fold{fold:02d}.json"
    if p.exists() and meta.exists():
        return
    c = runtime(); x, y = load_data()[0]; tr, te = load_data()[1][fold - 1]
    base, _, _ = build_initial_base(fold)
    feat = torch.from_numpy(x).float().cuda()
    with torch.no_grad():
        _, parts = c["layers"].InitialDataLayer(DIMS)(feat)
        _, initial_adjs, _ = base(parts)
    supports = [(a.detach().cpu().numpy() > 0) for a in initial_adjs]
    maxp = int(CFG["new_module"]["pair_max"]); rng0 = np.random.default_rng(513100 + fold)
    pi = np.full((4, maxp), -1, dtype=np.int64); pj = np.full((4, maxp), -1, dtype=np.int64)
    py = np.zeros((4, maxp), dtype=np.float32); pv = np.zeros((4, maxp), dtype=bool); pair_meta = []
    for m in range(4):
        pairs, labels, stat = sample_supported_pairs(supports[m], tr, y, x, maxp,
                                                     np.random.default_rng(513100 + fold * 10 + m))
        n = len(pairs); pi[m, :n] = pairs[:, 0] if n else -1; pj[m, :n] = pairs[:, 1] if n else -1
        if n: py[m, :n] = labels; pv[m, :n] = True
        pair_meta.append({"n": int(n), **stat, "candidate_edges": int(np.triu(supports[m], 1).sum())})
    align_ids = balanced_align_ids(tr, y, int(CFG["new_module"]["align_n"]), np.random.default_rng(513200 + fold))
    ai_lists, aj_lists, order_lists = [], [], []
    for k, m in enumerate((0, 1, 2)):
        ii, jj = np.triu_indices(len(align_ids), 1); gi, gj = align_ids[ii], align_ids[jj]
        keep = supports[m][gi, gj] & supports[3][gi, gj]
        pairs = np.stack([gi[keep], gj[keep]], axis=1) if int(keep.sum()) else np.zeros((0, 2), dtype=np.int64)
        ai_lists.append(pairs[:, 0] if len(pairs) else np.zeros(0, dtype=np.int64))
        aj_lists.append(pairs[:, 1] if len(pairs) else np.zeros(0, dtype=np.int64))
        order = np.arange(len(pairs), dtype=np.int64); rr = np.random.default_rng(513300 + fold * 10 + m)
        if len(order) > 1:
            rr.shuffle(order)
            if np.all(order == np.arange(len(order))): order = np.roll(order, 1)
        order_lists.append(order)
    maxq = max([len(v) for v in ai_lists] + [0])
    ai = np.full((3, maxq), -1, dtype=np.int64); aj = np.full((3, maxq), -1, dtype=np.int64)
    ao = np.full((3, maxq), -1, dtype=np.int64); av = np.zeros((3, maxq), dtype=bool)
    for k in range(3):
        n = len(ai_lists[k]); ai[k, :n] = ai_lists[k]; aj[k, :n] = aj_lists[k]; ao[k, :n] = order_lists[k]; av[k, :n] = True
    perm = split_site_permutation(x, tr, te, fold)
    np.savez_compressed(p, pair_i=pi, pair_j=pj, pair_y=py, pair_valid=pv,
                        align_i=ai, align_j=aj, align_order=ao, align_valid=av,
                        relation_perm=perm, train=np.asarray(tr), test=np.asarray(te), align_ids=align_ids)
    atomic_json(meta, {"fold": fold, "split_sha": split_sha(fold), "pair_meta": pair_meta,
                       "align_counts": [int(v.sum()) for v in av],
                       "relation_permutation_fraction": float(np.mean(perm != np.arange(len(perm))))})


def load_fold(fold):
    import numpy as np
    import torch
    z = np.load(FD / f"fold{fold:02d}.npz")
    return {k: torch.from_numpy(z[k]).cuda() for k in z.files}


def build_state(group, fold):
    import numpy as np
    import torch
    c = runtime(); x, y = load_data()[0]; tr, te = load_data()[1][fold - 1]; anchor = anchor_for(fold)
    seed_all(510000 + fold * 100 + 1)
    base = c["ecsrgl"].LegacyFixedGFR(c["network"].GraphLearn, c["network"].GCN,
                                      c["utils"], th=CFG["base"]["threshold"]).cuda()
    cam = c["ecsrgl"].PLNNoPEN(DIMS, hp(), c["layers"]).cuda(); cam.apply(c["utils"].my_weight_init)
    if group == "E1":
        model = base; c["ecsrgl"].load_e1_anchor(model, anchor["model"]["gfr"])
    else:
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(520000 + fold)
            model = c["ecsrgl"].ECSRGModel(c["network"].GraphLearn, c["network"].GCN,
                                            c["layers"], c["utils"], th=CFG["base"]["threshold"],
                                            mode=group, hidden=CFG["new_module"]["hidden"],
                                            s_dim=CFG["new_module"]["s_dim"],
                                            c_dim=CFG["new_module"]["c_dim"], base=base).cuda()
        c["ecsrgl"].load_e1_anchor(model, anchor["model"]["gfr"])
    cam.load_state_dict(anchor["model"]["pln"])
    post_init_rng = pack_rng()
    feat = torch.from_numpy(x).float().cuda(); targ = torch.from_numpy(y).long().cuda(); train = torch.as_tensor(tr, dtype=torch.long, device="cuda")
    counts = np.asarray([int((y[tr] == 0).sum()), int((y[tr] == 1).sum())], dtype=np.float32)
    weights = torch.from_numpy(len(tr) / (2.0 * counts)).float().cuda()
    if group == "E1":
        opt_base = torch.optim.Adam(model.parameters(), lr=CFG["base"]["lr"], weight_decay=CFG["base"]["weight_decay"])
        opt_new = None
    else:
        base_params = list(model.base.parameters()); base_ids = {id(p) for p in base_params}
        new_params = [p for p in model.parameters() if id(p) not in base_ids]
        opt_base = torch.optim.Adam(base_params, lr=CFG["base"]["lr"], weight_decay=CFG["base"]["weight_decay"])
        opt_new = torch.optim.Adam(new_params, lr=CFG["new_module"]["lr"], weight_decay=CFG["new_module"]["weight_decay"])
    opt_cam = torch.optim.Adam(cam.parameters(), lr=CFG["base"]["lr"], weight_decay=CFG["base"]["weight_decay"])
    return {"group": group, "fold": fold, "model": model, "cam": cam,
            "opt_base": opt_base, "opt_new": opt_new, "opt_cam": opt_cam,
            "feat": feat, "targ": targ, "train": train, "test": np.asarray(te),
            "weight": weights, "parts": c["layers"].InitialDataLayer(DIMS),
            "initial_rng_sha": rng_sha(post_init_rng),
            "anchor_rng_sha": rng_sha(anchor["rng"]) if anchor.get("rng") is not None else "",
            "initial_parameter_sha": anchor["parameter_sha"]}


def metrics(logp, state):
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    idx = state["test"]; y = state["targ"].detach().cpu().numpy()[idx]
    probs = logp.detach().exp().cpu().numpy()[idx]; pred = probs.argmax(1)
    cm = confusion_matrix(y, pred, labels=[0, 1]); tp, fn, fp, tn = map(int, cm.ravel())
    hard = float(roc_auc_score(np.eye(2)[y], np.eye(2)[pred])); pa = float(roc_auc_score(y, probs[:, 1]))
    spe = tn / max(tn + fp, 1); sen = tp / max(tp + fn, 1)
    return {"acc": float(np.mean(pred == y)), "hard_auc": hard, "prob_auc": pa,
            "spe": spe, "sen": sen, "ba": (spe + sen) / 2,
            "asd_f1": float(f1_score(y, pred, pos_label=0, zero_division=0)),
            "asd_tp": tp, "asd_fn": fn, "td_fp": fp, "td_tn": tn, "n_test": len(idx)}


def grad_norm(module):
    import torch
    vals = [(p.grad.detach() ** 2).sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(vals).sum().sqrt().cpu()) if vals else 0.0


def forward_state(state, fd, train_mode=True, intervention=None):
    import torch
    c = runtime(); model, cam = state["model"], state["cam"]
    model.train(train_mode); cam.train(train_mode)
    _, parts = state["parts"](state["feat"])
    if state["group"] == "E1":
        fused, adjs, details = model(parts)
    else:
        base_fused, adjs, base_details = model.base(parts)
        old_cam = cam.training; cam.eval()
        with torch.no_grad():
            pcond = cam(base_fused.detach())[0].exp().detach()
        cam.train(old_cam)
        rel = fd["relation_perm"].long() if state["group"] == "E10" else None
        fused, adjs, details = model(parts, pcond, precomputed=(base_fused, adjs, base_details), relation_perm=rel, intervention=intervention)
    prob, hidden, attn = cam(fused)
    graph = sum(c["utils"].GraphConstructLoss(parts[m], adjs[m], CFG["base"]["theta_smooth"], CFG["base"]["theta_degree"], CFG["base"]["theta_sparsity"]) for m in range(4)) / 4
    cls = c["utils"].ClsLoss(prob, state["targ"], state["train"], state["weight"])
    out = {"prob": prob, "hidden": hidden, "fused": fused, "parts": parts,
           "adjs": adjs, "details": details, "cls": cls, "graph": graph,
           "pair": fused.new_zeros(()), "neighbor": fused.new_zeros(()),
           "align": fused.new_zeros(()), "pair_raw": 0., "neighbor_raw": 0.,
           "align_raw": 0., "align_weights": [], "pair_counts": [], "align_counts": []}
    if state["group"] == "E1":
        out["total"] = cls + graph
        return out
    pair_on = state["group"] in {"E6", "E7", "E8", "E9", "E10", "E11", "E12"}
    neighbor_on = state["group"] in {"E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12"}
    align_on = state["group"] in {"E7", "E8", "E9", "E10", "E11", "E12"}
    pair, counts = model.pair_loss(fd["pair_i"].long(), fd["pair_j"].long(), fd["pair_y"], fd["pair_valid"], pair_on, CFG["new_module"]["pair_temperature"])
    neighbor = model.neighbor_loss(state["train"], state["targ"], neighbor_on and state["group"] != "E11")
    align, aw, ac = model.align_loss(fd["align_i"].long(), fd["align_j"].long(), fd["align_valid"], fd["align_order"], align_on, wrong_correspondence=state["group"] == "E9")
    out.update({"pair": CFG["new_module"]["lambda_pair"] * pair,
                "neighbor": CFG["new_module"]["lambda_neighbor"] * neighbor if state["group"] != "E11" else neighbor.new_zeros(()),
                "align": CFG["new_module"]["lambda_align"] * align,
                "pair_raw": float(pair.detach()), "neighbor_raw": float(neighbor.detach()),
                "align_raw": float(align.detach()), "align_weights": aw,
                "pair_counts": counts, "align_counts": ac})
    out["total"] = cls + graph + out["pair"] + out["neighbor"] + out["align"]
    return out


def state_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def task_path(group, fold): return TASKS / f"{group}_f{fold:02d}.json"
def ckpt_path(group, fold): return CKPTS / f"{group}_f{fold:02d}.best.pt"
def latest_path(group, fold): return CKPTS / f"{group}_f{fold:02d}.latest.pt"


def diagnostics(state, fd, out):
    import torch
    d = out.get("details", {}); raw = state["feat"].detach(); fused = out["fused"].detach()
    base = d.get("base_fused", fused).detach() if isinstance(d, dict) else fused
    rg = torch.cat(d.get("residuals_graph", []), 1) if d.get("residuals_graph") else torch.zeros_like(raw)
    rc = torch.cat(d.get("residuals_private", []), 1) if d.get("residuals_private") else torch.zeros_like(raw)
    rows = {"group": state["group"], "fold": state["fold"],
            "base_message_ratio": float((base - raw).norm() / raw.norm().clamp_min(1e-8)),
            "graph_residual_ratio_input": float(rg.norm() / raw.norm().clamp_min(1e-8)),
            "private_residual_ratio_input": float(rc.norm() / raw.norm().clamp_min(1e-8)),
            "new_residual_ratio_input": float((rg + rc).norm() / raw.norm().clamp_min(1e-8)),
            "pair_loss": float(out.get("pair_raw", 0.)), "neighbor_loss": float(out.get("neighbor_raw", 0.)),
            "align_loss": float(out.get("align_raw", 0.)),
            "align_w_mean": float(sum(out.get("align_weights", [])) / max(len(out.get("align_weights", [])), 1)),
            "rho_graph_mean": float(d.get("rho_graph", torch.zeros(4)).mean()) if isinstance(d, dict) else 0.,
            "rho_private_mean": float(d.get("rho_private", torch.zeros(4)).mean()) if isinstance(d, dict) else 0.}
    if isinstance(d, dict):
        for m, b in enumerate(d.get("b", [])):
            nz = b[b > 0]
            rows[f"edge_count_{m}"] = int(nz.numel())
            rows[f"edge_mean_{m}"] = float(nz.mean().detach().cpu()) if nz.numel() else 0.
            rows[f"edge_sd_{m}"] = float(nz.std(unbiased=False).detach().cpu()) if nz.numel() else 0.
        rows["neighbor_norm"] = float(d.get("neighbor", raw.new_zeros((len(raw), 32))).norm().cpu())
        rows["neighbor_grad_norm"] = 0.
        if d.get("logits"):
            from sklearn.metrics import roc_auc_score
            for m, lg in enumerate(d["logits"]):
                i, j = fd["pair_i"][m], fd["pair_j"][m]; valid = fd["pair_valid"][m] & (i >= 0) & (j >= 0)
                try:
                    rows[f"pair_auc_{m}"] = float(roc_auc_score(fd["pair_y"][m][valid].detach().cpu().numpy(), (lg[i[valid], j[valid]] / CFG["new_module"]["pair_temperature"]).detach().cpu().numpy()))
                except Exception:
                    rows[f"pair_auc_{m}"] = .5
    return rows


def train_one(group, fold):
    import torch
    runtime(); prepare_fold(fold); tp = task_path(group, fold)
    if tp.exists() and json.loads(tp.read_text()).get("status") == "COMPLETE":
        return
    state = build_state(group, fold); fd = load_fold(fold)
    start = 0; wait = 0; best_acc = -1.; best_epoch = -1; best_model = best_cam = best_ob = best_on = best_oc = best_diag = None
    lp = latest_path(group, fold)
    if lp.exists():
        try:
            d = torch_load(lp); state["model"].load_state_dict(d["model"]); state["cam"].load_state_dict(d["cam"])
            state["opt_base"].load_state_dict(d["opt_base"])
            if state["opt_new"] is not None: state["opt_new"].load_state_dict(d["opt_new"])
            state["opt_cam"].load_state_dict(d["opt_cam"]); restore_rng(d["rng"])
            start = int(d["epoch"]) + 1; wait = int(d["wait"]); best_acc = float(d["best_acc"]); best_epoch = int(d["best_epoch"])
            best_model, best_cam, best_ob, best_on, best_oc, best_diag = d.get("best_model"), d.get("best_cam"), d.get("best_ob"), d.get("best_on"), d.get("best_oc"), d.get("best_diag")
        except Exception:
            start = 0
    t0 = time.time(); epoch = start - 1
    try:
        for epoch in range(start, CFG["base"]["epochs"]):
            state["opt_base"].zero_grad(); state["opt_cam"].zero_grad()
            if state["opt_new"] is not None: state["opt_new"].zero_grad()
            out = forward_state(state, fd, True)
            assert torch.isfinite(out["total"])
            out["total"].backward()
            for mod in (state["model"], state["cam"]):
                for p in mod.parameters():
                    if p.grad is not None: assert torch.isfinite(p.grad).all()
            state["opt_base"].step(); state["opt_cam"].step()
            if state["opt_new"] is not None: state["opt_new"].step()
            with torch.no_grad():
                ev = forward_state(state, fd, False); met = metrics(ev["prob"], state)
            if met["acc"] > best_acc:
                best_acc = met["acc"]; best_epoch = epoch; wait = 0
                best_model = state_cpu(state["model"]); best_cam = state_cpu(state["cam"])
                best_ob = copy.deepcopy(state["opt_base"].state_dict()); best_on = copy.deepcopy(state["opt_new"].state_dict()) if state["opt_new"] is not None else None; best_oc = copy.deepcopy(state["opt_cam"].state_dict())
                best_diag = diagnostics(state, fd, ev)
                torch_save(ckpt_path(group, fold), {"group": group, "fold": fold, "seed": 0,
                    "best_epoch": best_epoch, "epochs_executed": epoch + 1, "metrics": met,
                    "model": best_model, "cam": best_cam, "opt_base": best_ob, "opt_new": best_on,
                    "opt_cam": best_oc, "diagnostics": best_diag, "split_sha": split_sha(fold),
                    "anchor_parameter_sha": state["initial_parameter_sha"], "anchor_rng_sha": state["initial_rng_sha"],
                    "historical_anchor_rng_sha": state["anchor_rng_sha"], "source_sha": sha(EXP / "ecsrgl.py")})
            else:
                wait += 1
            if epoch % 10 == 0:
                torch_save(lp, {"epoch": epoch, "wait": wait, "best_acc": best_acc, "best_epoch": best_epoch,
                    "model": state_cpu(state["model"]), "cam": state_cpu(state["cam"]),
                    "opt_base": state["opt_base"].state_dict(), "opt_new": state["opt_new"].state_dict() if state["opt_new"] is not None else None,
                    "opt_cam": state["opt_cam"].state_dict(), "rng": pack_rng(), "best_model": best_model,
                    "best_cam": best_cam, "best_ob": best_ob, "best_on": best_on, "best_oc": best_oc, "best_diag": best_diag})
            if wait > CFG["base"]["early"]:
                break
        assert best_model is not None
        state["model"].load_state_dict(best_model); state["cam"].load_state_dict(best_cam)
        with torch.no_grad():
            check_out = forward_state(state, fd, False); check = metrics(check_out["prob"], state)
        assert abs(check["acc"] - best_acc) < 1e-8
        row = {"group": group, "group_id": int(group[1:]), "name": CFG["groups"][str(int(group[1:]))]["name"],
               "fold": fold, "seed": 0, "gpu": 0, "status": "COMPLETE", "source": "new_training",
               "split_sha": split_sha(fold), "anchor_parameter_sha": state["initial_parameter_sha"],
               "anchor_rng_sha": state["initial_rng_sha"], "historical_anchor_rng_sha": state["anchor_rng_sha"],
               "best_epoch": best_epoch, "epochs_executed": epoch + 1, "wall_seconds": time.time() - t0,
               "best_sha": sha(ckpt_path(group, fold)), **check}
        atomic_json(tp, row); lp.unlink(missing_ok=True)
        print("DONE", group, fold, best_epoch, row["acc"], flush=True)
    except Exception as e:
        atomic_json(tp, {"group": group, "fold": fold, "status": "FAILED", "error": repr(e), "traceback": traceback.format_exc()})
        raise


def historical_e0_rows():
    """Independent readback of the physical B0-B checkpoints."""
    import numpy as np
    import torch
    c = runtime(); x, y = load_data()[0]; rows = []
    old = {int(r["fold"]): r for r in csv_read(ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/results/fold_metrics.csv") if r.get("group") == "E00_B0B_REPLAY"}
    for fold, (tr, te) in enumerate(load_data()[1], 1):
        d = torch_load(HIST / f"g0_f{fold:02d}.best.pt")
        gfr = c["network"].GFR(DIMS, DIMS, th=CFG["base"]["threshold"], mode="weighted-cosine").cuda()
        cam = c["ecsrgl"].PLNNoPEN(DIMS, hp(), c["layers"]).cuda()
        gfr.load_state_dict(d["model"]["gfr"]); cam.load_state_dict(d["model"]["pln"]); gfr.eval(); cam.eval()
        feat = torch.from_numpy(x).float().cuda(); targ = torch.from_numpy(y).long().cuda(); ti = torch.as_tensor(tr, device="cuda")
        with torch.no_grad():
            _, parts = c["layers"].InitialDataLayer(DIMS)(feat); fused, _, _ = gfr(parts, 4, targ, ti); lp, _, _ = cam(fused)
        st = {"targ": targ, "test": np.asarray(te)}; met = metrics(lp, st); ref = old[fold]
        met.update({"group": "E0", "group_id": 0, "name": "B0-B_REFERENCE", "fold": fold, "seed": 0,
                    "status": "REFERENCE", "source": "Experiment43.1_verified_readback",
                    "best_epoch": int(ref["best_epoch"]), "epochs_executed": int(ref["epochs_executed"]),
                    "split_sha": split_sha(fold), "best_sha": ref["best_sha"]})
        rows.append(met)
    return rows


def historical_e1_rows():
    """Read the strictly reproduced Experiment 51 E1 fold checkpoints."""
    import numpy as np
    import torch
    c = runtime(); x, y = load_data()[0]; rows = []
    src = E51_2_CKPTS
    assert all((src / f"E1_f{f:02d}.best.pt").exists() for f in range(1, 11))
    for fold, (tr, te) in enumerate(load_data()[1], 1):
        d = torch_load(src / f"E1_f{fold:02d}.best.pt")
        model = c["ecsrgl"].LegacyFixedGFR(c["network"].GraphLearn, c["network"].GCN, c["utils"], th=CFG["base"]["threshold"]).cuda()
        cam = c["ecsrgl"].PLNNoPEN(DIMS, hp(), c["layers"]).cuda(); model.load_state_dict(d["model"]); cam.load_state_dict(d["cam"]); model.eval(); cam.eval()
        feat = torch.from_numpy(x).float().cuda(); targ = torch.from_numpy(y).long().cuda()
        with torch.no_grad():
            _, parts = c["layers"].InitialDataLayer(DIMS)(feat); fused, _, _ = model(parts); lp, _, _ = cam(fused)
        st = {"targ": targ, "test": np.asarray(te)}; met = metrics(lp, st)
        met.update({"group": "E1", "group_id": 1, "name": "E1_AGGREGATION_REPLAY", "fold": fold, "seed": 0,
                    "status": "REFERENCE", "source": str(src), "best_epoch": int(d["best_epoch"]),
                    "epochs_executed": int(d["epochs_executed"]), "split_sha": split_sha(fold),
                    "best_sha": sha(src / f"E1_f{fold:02d}.best.pt"), "anchor_parameter_sha": d.get("anchor_parameter_sha", ""),
                    "anchor_rng_sha": d.get("anchor_rng_sha", ""), "historical_anchor_rng_sha": d.get("historical_anchor_rng_sha", "")})
        rows.append(met)
    return rows


def prepare():
    import numpy as np
    for p in (RT, TASKS, CKPTS, LOGS, FD, RESULTS):
        p.mkdir(parents=True, exist_ok=True)
    load_data(); [prepare_fold(f) for f in range(1, 11)]
    e0 = historical_e0_rows(); e1 = historical_e1_rows(); csv_write(RESULTS / "fold_metrics.csv", e0 + e1)
    man = {"experiment": "Experiment51.3_EC-SRGL", "status": "PREPARED", "config_sha": sha(CFG_PATH),
           "code_sha": {p.name: sha(p) for p in (EXP / "run.py", EXP / "ecsrgl.py", EXP / "dcsra_base.py")},
           "source_sha": {p.name: sha(p) for p in SRC.glob("*.py")},
           "protected_sha256": {str(ROOT / n): sha(ROOT / n) for n in ["main.py", "model.py", "network.py", "layers.py", "utils.py", "data/ABIDE/processed_standard_data.csv", "data/ABIDE/modal_feat_dict.npy"]},
           "data_shape": list(load_data()[0][0].shape), "label_counts": np.bincount(load_data()[0][1]).tolist(),
           "folds": {str(f): split_sha(f) for f in range(1, 11)}, "groups": 13, "historical_e0_folds": 10,
           "e1_readback_folds": 10, "planned_new_folds": 110, "e1_seed_rule": "510000 + fold*100 + 1",
           "new_module_seed_rule": "isolated torch fork 520000 + fold", "runtime_excluded_from_delivery": True}
    atomic_json(EXP / "manifest.json", man); print("PREPARE_PASS")


def controller():
    groups = [f"E{i}" for i in range(2, 13)]; tasks = [(g, f) for g in groups for f in range(1, 11)]
    procs = []; start = time.time(); workers = min(5, int(os.environ.get("ECSR_WORKERS", CFG["gpu_plan"]["workers"])))
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "0"; env.update({k: "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    while tasks or procs:
        alive = sum(p.poll() is None for _, _, p, _ in procs)
        while alive < workers and tasks:
            g, f = tasks.pop(0); log = (LOGS / f"{g}_f{f:02d}.log").open("ab")
            p = subprocess.Popen([sys.executable, "-B", str(EXP / "run.py"), "worker", "--group", g, "--fold", str(f)], cwd=EXP, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            procs.append((g, f, p, log)); alive += 1
        for item in list(procs):
            if item[2].poll() is not None:
                item[3].close(); procs.remove(item)
        time.sleep(.5)
    status = [json.loads(task_path(g, f).read_text()).get("status") if task_path(g, f).exists() else "MISSING" for g in groups for f in range(1, 11)]
    m = json.loads((EXP / "manifest.json").read_text()); m.update(status="TRAINING_COMPLETE" if status.count("COMPLETE") == 110 else "TRAINING_PARTIAL", completed_new_folds=status.count("COMPLETE"), failed_tasks=status.count("FAILED"), controller_wall_seconds=time.time() - start)
    atomic_json(EXP / "manifest.json", m); print("CONTROLLER_DONE", status.count("COMPLETE"), status.count("FAILED"))


def aggregate():
    import numpy as np
    rows = historical_e0_rows() + historical_e1_rows()
    for g in [f"E{i}" for i in range(2, 13)]:
        for f in range(1, 11):
            r = json.loads(task_path(g, f).read_text()); assert r["status"] == "COMPLETE"; rows.append(r)
    csv_write(RESULTS / "fold_metrics.csv", rows); sums = []
    for g in [f"E{i}" for i in range(13)]:
        rr = [r for r in rows if r["group"] == g]; q = {"group": g, "name": CFG["groups"][str(int(g[1:]))]["name"], "folds": len(rr)}
        for k in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"):
            z = np.asarray([float(r[k]) for r in rr]); q[k + "_mean"] = float(z.mean()); q[k + "_sd"] = float(z.std(ddof=0))
        sums.append(q)
    csv_write(RESULTS / "summary.csv", sums)
    by = {(r["group"], int(r["fold"])): r for r in rows}; pairs = []
    for other in ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E8", "E9", "E10", "E11", "E12"]:
        for k in ("acc", "hard_auc", "prob_auc", "spe", "sen"):
            d = np.asarray([(float(by[("E7", f)][k]) - float(by[(other, f)][k])) * 100 for f in range(1, 11)])
            pairs.append({"comparison": f"E7_vs_{other}", "metric": k, "delta_pp": float(d.mean()), "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum())})
    csv_write(RESULTS / "paired_comparisons.csv", pairs)
    m = json.loads((EXP / "manifest.json").read_text()); m.update(status="AGGREGATED", folds_complete=130, summary_sha=sha(RESULTS / "summary.csv"), paired_sha=sha(RESULTS / "paired_comparisons.csv")); atomic_json(EXP / "manifest.json", m); print("AGGREGATE_PASS")


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prepare"); sp.add_parser("controller"); sp.add_parser("aggregate")
    w = sp.add_parser("worker"); w.add_argument("--group", required=True); w.add_argument("--fold", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "prepare": prepare()
    elif a.cmd == "controller": controller()
    elif a.cmd == "aggregate": aggregate()
    else: train_one(a.group, a.fold)


if __name__ == "__main__":
    main()
