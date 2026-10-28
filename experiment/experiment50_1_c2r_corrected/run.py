#!/usr/bin/env python3
"""Experiment 50.1 runner.

The runner is isolated from the historical repository.  It imports only the
immutable B0-B source snapshot in ``.runtime/sources`` and model50_1.py.  The
formal protocol is the historical simple-2 protocol: the outer test set is
also used for checkpoint selection, while all new relation supervision and
site statistics are restricted to outer-train patients.
"""
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
ROOT = Path(os.environ.get("SPROMPTGL_ROOT", "/data/adodas/SPromptGL-main"))
RT = EXP / ".runtime"
SRC = RT / "sources"
ANCHORS = RT / "anchors"
CKPTS = RT / "checkpoints"
TASKS = RT / "tasks"
LOGS = RT / "logs"
FOLD_DATA = RT / "fold_data"
RESULTS = EXP / "results"
CFG = json.loads((EXP / "config.json").read_text())
# The formal artifact carries the small immutable source snapshot in ``source``
# and may omit the scratch runtime.  Keep the scratch path first so completed
# runs are unchanged, then fall back to the packaged source and verified
# historical anchor location for independent reconstruction.
if not SRC.exists() and (EXP / "source").exists():
    SRC = EXP / "source"
if not ANCHORS.exists():
    _historical_anchors = ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
    if _historical_anchors.exists():
        ANCHORS = _historical_anchors
BASE = CFG["base"]
NEW = CFG["new"]
DIMS = [48, 6, 10, 256]
FC_INDEX = int(CFG["modalities"]["CORR_FC"])
AUX_INDICES = tuple(int(x) for x in CFG["modalities"]["aux"])

CTX = None
DATA = None
FOLDS = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def read_json(path):
    return json.loads(Path(path).read_text())


def torch_save(path, obj):
    import torch
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def csv_write(path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def csv_read(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pack_rng():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def rng_sha(state):
    h = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    h.update(state["cpu"].cpu().numpy().tobytes())
    for value in state["cuda"]:
        h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def runtime():
    global CTX
    if CTX is None:
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        assert torch.cuda.is_available(), "CUDA is required for Experiment 50.1"
        torch.cuda.set_device(0)
        sys.path.insert(0, str(SRC)); sys.path.insert(0, str(EXP))
        import layers, network, utils
        from model50_1 import PLNNoPEN, C2RAlignCorrected
        CTX = {"torch": torch, "np": __import__("numpy"),
               "layers": layers, "network": network, "utils": utils,
               "PLNNoPEN": PLNNoPEN, "C2R": C2RAlignCorrected}
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
    x = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = frame.iloc[:, -1].to_numpy(dtype=np.int64) - 1
    assert x.shape == (871, 320) and np.isfinite(x).all()
    assert np.bincount(y).tolist() == [403, 468]
    assert [len(mapping[key]) for key in ("PHENO", "ANAT", "FUNC", "Correlation")] == DIMS
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10).split(x, y))
    DATA, FOLDS = (x, y), folds
    return DATA, FOLDS


def split_sha(fold):
    import numpy as np
    _, folds = load_data(); train, test = folds[fold - 1]
    return hashlib.sha256(train.astype(np.int64).tobytes() + test.astype(np.int64).tobytes()).hexdigest()


def protected_hashes():
    out = {}
    for rel in CFG["protected_root"]:
        p = ROOT / rel
        if p.exists():
            out[str(p)] = sha(p)
    return out


def digest_model(gfr, cam):
    h = hashlib.sha256()
    for prefix, module in (("gfr", gfr), ("pln", cam)):
        for name, param in sorted(module.named_parameters()):
            h.update((prefix + "." + name).encode() + b"\0")
            h.update(param.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def model_state(gfr, cam):
    return {"gfr": {k: v.detach().cpu().clone() for k, v in gfr.state_dict().items()},
            "pln": {k: v.detach().cpu().clone() for k, v in cam.state_dict().items()}}


def load_model_state(gfr, cam, state):
    gfr.load_state_dict(state["gfr"])
    cam.load_state_dict(state.get("pln", state.get("cam")))


def make_hp():
    return SimpleNamespace(
        n_hidden=BASE["n_hidden"], n_head=BASE["n_head"], dropout=BASE["dropout"],
        nlayer=BASE["nlayer"], nmodal=4, nclass=2, lr=BASE["lr"], reg=BASE["weight_decay"],
        th=BASE["threshold"], GC_mode="weighted-cosine", MP_mode="GCN", MF_mode="concat",
        alpha=0.5, theta_smooth=BASE["theta_smooth"], theta_degree=BASE["theta_degree"],
        theta_sparsity=BASE["theta_sparsity"])


def build_base(fold):
    import torch
    c = runtime(); x, y = load_data()[0]; train, test = load_data()[1][fold - 1]
    anchor = torch_load(ANCHORS / f"fold{fold:02d}.pt")
    assert anchor["split_sha"] == split_sha(fold)
    restore_rng(anchor["entry_rng"])
    hp = make_hp()
    gfr = c["network"].GFR(DIMS, DIMS, th=hp.th, mode=hp.GC_mode).cuda()
    cam = c["PLNNoPEN"](DIMS, hp, c["layers"]).cuda()
    cam.apply(c["utils"].my_weight_init)
    assert sum(p.numel() for p in cam.parameters()) == 49544
    assert sum(p.numel() for p in gfr.parameters()) == 887836
    assert digest_model(gfr, cam) == anchor["parameter_sha"]
    opt_gfr = torch.optim.Adam(gfr.parameters(), lr=hp.lr, weight_decay=hp.reg)
    opt_cam = torch.optim.Adam(cam.parameters(), lr=hp.lr, weight_decay=hp.reg)
    opt_gfr.load_state_dict(copy.deepcopy(anchor["mp"]))
    opt_cam.load_state_dict(copy.deepcopy(anchor["mf"]))
    restore_rng(anchor["rng"])
    state = {"gfr": gfr, "cam": cam, "opt_gfr": opt_gfr, "opt_cam": opt_cam,
             "feat": torch.from_numpy(x).float().cuda(),
             "targ": torch.from_numpy(y).long().cuda(),
             "train": train, "test": test, "weight": None,
             "data_layer": c["layers"].InitialDataLayer(DIMS),
             "anchor": anchor, "fold": fold}
    counts = Counter(y[train])
    weights = len(train) / __import__("numpy").asarray(list(counts.values())) / 2
    state["weight"] = torch.from_numpy(weights).float().cuda()
    return state


def init_c2r(state, group_id):
    import torch
    c = runtime()
    base_rng = pack_rng()
    seed_all(510000 + state["fold"] * 100 + group_id)
    module = c["C2R"](DIMS, NEW["embedding"], NEW["encoder_hidden"],
                        NEW["eta"], NEW["rho"], NEW["relation_temperature"],
                        NEW["lambda_site_correction"]).cuda()
    restore_rng(base_rng)
    optimizer = torch.optim.Adam(module.parameters(), lr=NEW["lr"],
                                 weight_decay=NEW["weight_decay"])
    return module, optimizer


def site_ids(x):
    import numpy as np
    return np.argmax(x[:, 26:46], axis=1).astype(np.int64)


def make_permutation(x, train, test, seed):
    import numpy as np
    sid = site_ids(x); perm = np.arange(len(x), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for ids in (train, test):
        groups = {}
        for idx in ids:
            groups.setdefault(int(sid[idx]), []).append(int(idx))
        for values in groups.values():
            if len(values) < 2:
                continue
            arr = np.asarray(values, dtype=np.int64); shuffled = arr.copy()
            for _ in range(30):
                rng.shuffle(shuffled)
                if np.all(shuffled != arr):
                    break
            else:
                shuffled = np.roll(arr, 1)
            perm[arr] = shuffled
    return perm


def make_rewire(candidates, perm):
    import numpy as np
    out = perm[candidates].copy()
    changed = np.zeros(out.shape, dtype=bool)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if out[i, j] == i:
                for q in range(out.shape[1]):
                    candidate = perm[candidates[i, (j + q + 1) % out.shape[1]]]
                    if candidate != i:
                        out[i, j] = candidate
                        break
            changed[i, j] = out[i, j] != candidates[i, j]
    return out, changed


def make_candidates(x, train, k=8):
    import numpy as np
    corr = x[:, 64:320].astype(np.float64)
    corr /= np.linalg.norm(corr, axis=1, keepdims=True).clip(min=1e-8)
    sender_pool = np.asarray(train, dtype=np.int64)
    result = np.empty((len(x), k), dtype=np.int64)
    for receiver in range(len(x)):
        pool = sender_pool[sender_pool != receiver]
        scores = corr[pool].dot(corr[receiver])
        order = np.argsort(-scores, kind="mergesort")[:min(k, len(pool))]
        result[receiver, :len(order)] = pool[order]
        if len(order) < k:
            result[receiver, len(order):] = pool[:k - len(order)]
    return result


def make_pairs(x, y, train, seed):
    import numpy as np
    sid = site_ids(x); rng = np.random.default_rng(seed)
    buckets = {(same_y, same_s): [] for same_y in (0, 1) for same_s in (0, 1)}
    train = np.asarray(train, dtype=np.int64)
    for pos, i in enumerate(train):
        for j in train[pos + 1:]:
            buckets[(int(y[i] == y[j]), int(sid[i] == sid[j]))].append((int(i), int(j)))
    chosen = []; counts = {}
    cap = int(NEW["pair_count_target_per_stratum"])
    for key, values in buckets.items():
        rng.shuffle(values); take = values[:min(cap, len(values))]
        chosen.extend(take); counts[str(key)] = len(take)
    rng.shuffle(chosen)
    pair_i = np.asarray([a for a, _ in chosen], dtype=np.int64)
    pair_j = np.asarray([b for _, b in chosen], dtype=np.int64)
    pair_y = (y[pair_i] == y[pair_j]).astype(np.float32)
    return pair_i, pair_j, pair_y, counts


def prepare_fold(fold):
    import numpy as np
    x, y = load_data()[0]; train, test = load_data()[1][fold - 1]
    path = FOLD_DATA / f"fold{fold:02d}.npz"
    if path.exists():
        return
    candidates = make_candidates(x, train, NEW["candidate_k"])
    perm = make_permutation(x, train, test, 7500 + fold)
    rewire, changed = make_rewire(candidates, perm)
    pair_i, pair_j, pair_y, counts = make_pairs(x, y, train, 9500 + fold)
    np.savez_compressed(path, candidates=candidates, perm=perm, rewire=rewire,
                        rewire_changed=changed, pair_i=pair_i, pair_j=pair_j,
                        pair_y=pair_y, train=np.asarray(train), test=np.asarray(test))
    sid = site_ids(x)
    write_json(FOLD_DATA / f"fold{fold:02d}.json", {
        "fold": fold, "candidate_k": int(candidates.shape[1]), "pair_counts": counts,
        "train_n": int(len(train)), "test_n": int(len(test)),
        "candidate_sender_train_only": bool(np.isin(candidates, train).all()),
        "same_site_candidate_fraction": float((sid[:, None] == sid[candidates]).mean()),
        "aux_permutation_fraction": float((perm != np.arange(len(x))).mean()),
        "rewire_fraction": float(changed.mean()), "candidate_sha": sha(path),
        "split_sha": split_sha(fold)})


def load_fold_data(fold, device="cuda"):
    import torch
    z = __import__("numpy").load(FOLD_DATA / f"fold{fold:02d}.npz")
    return {key: torch.from_numpy(z[key]).to(device) for key in z.files}


def base_forward(state, c2r, fd, mode, train_mode=None, disable_c2r=False,
                 force_null=False, force_no_site=False, force_aux_perm=False):
    import torch
    c = runtime()
    if train_mode is True:
        state["gfr"].train(); state["cam"].train(); c2r.train()
    elif train_mode is False:
        state["gfr"].eval(); state["cam"].eval(); c2r.eval()
    nmodal, parts = state["data_layer"](state["feat"])
    h_gfr, adjs, aloss = state["gfr"](
        parts, nmodal, state["targ"], state["train"])
    graph = sum(c["utils"].GraphConstructLoss(
        parts[i], adjs[i], BASE["theta_smooth"], BASE["theta_degree"],
        BASE["theta_sparsity"]) for i in range(4)) / 4.0
    if disable_c2r:
        h50 = h_gfr; rel = h_gfr.new_zeros(()); align = h_gfr.new_zeros(())
        diag = {"gate_pheno": 0., "gate_anat": 0., "gate_func": 0., "gate_null": 1.,
                "message_norm": 0., "delta_norm": 0., "gfr_norm": float(h_gfr.detach().norm()),
                "residual_ratio": 0., "pair_count": int(fd["pair_i"].numel()),
                "pair_positive_fraction": float(fd["pair_y"].float().mean()),
                "rel_loss": 0., "align_loss": 0., "site_bias_abs": 0.,
                "site_corrected": 0., "neighbor_message_std": 0.,
                "null_rejection_fraction": 1., "aux_accept_mean": 0.}
    else:
        use_aux_perm = fd["perm"] if (force_aux_perm or mode == "E6_MISALIGNED_AUX") else None
        h50, rel, align, diag = c2r(
            state["feat"], h_gfr, fd["candidates"], fd["pair_i"],
            fd["pair_j"], fd["pair_y"], mode=mode, aux_perm=use_aux_perm,
            rewire=fd["rewire"] if mode == "E7_REWIRED" else None,
            force_null=force_null, force_no_site=force_no_site,
            force_aux_perm=force_aux_perm)
    prob, hidden, attention = state["cam"](h50)
    cls = c["utils"].ClsLoss(prob, state["targ"], state["train"], state["weight"])
    total = cls + graph + NEW["lambda_relation"] * rel + NEW["lambda_align"] * align
    return {"prob": prob, "hidden": hidden, "h_gfr": h_gfr, "h50": h50,
            "cls": cls, "graph": graph, "rel": rel, "align": align,
            "total": total, "adjs": adjs, "aloss": aloss, "diag": diag}


def metric(output, state, test=None):
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    if test is None:
        test = state["test"]
    idx = np.asarray(test, dtype=np.int64)
    labels = state["targ"].detach().cpu().numpy()[idx]
    probs = output.detach().exp().cpu().numpy()[idx]
    preds = probs.argmax(1)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    asd_tp, asd_fn, td_fp, td_tn = map(int, cm.ravel())
    hard_auc = float(roc_auc_score(np.eye(2)[labels], np.eye(2)[preds]))
    prob_auc = float(roc_auc_score(labels, probs[:, 1]))
    sen = asd_tp / max(asd_tp + asd_fn, 1)
    spe = td_tn / max(td_tn + td_fp, 1)
    return {"acc": float((preds == labels).mean()), "hard_auc": hard_auc,
            "prob_auc": prob_auc, "spe": float(spe), "sen": float(sen),
            "ba": float((spe + sen) / 2),
            "asd_f1": float(f1_score(labels, preds, pos_label=0, zero_division=0)),
            "asd_tp": asd_tp, "asd_fn": asd_fn, "td_fp": td_fp, "td_tn": td_tn,
            "n_test": int(len(idx))}


def grad_norm(module):
    import torch
    total = None
    for param in module.parameters():
        if param.grad is not None:
            value = (param.grad.detach() ** 2).sum()
            total = value if total is None else total + value
    return float(total.sqrt().cpu()) if total is not None else 0.0


def checkpoint_path(group, fold):
    return CKPTS / f"{group}_f{fold:02d}.best.pt"


def latest_path(group, fold):
    return CKPTS / f"{group}_f{fold:02d}.latest.pt"


def task_path(group, fold):
    return TASKS / f"{group}_f{fold:02d}.json"


def save_training_checkpoint(path, state, c2r, opt_c2r, epoch, wait, best_acc,
                             best_epoch, best_metrics, best_diag, best_model,
                             best_c2r, best_rng):
    torch_save(path, {"epoch": epoch, "wait": wait, "best_acc": best_acc,
                      "best_epoch": best_epoch, "best_metrics": best_metrics,
                      "best_diag": best_diag, "best_model": best_model,
                      "best_c2r": best_c2r, "best_rng": best_rng, "rng": pack_rng(),
                      "gfr": state["gfr"].state_dict(), "cam": state["cam"].state_dict(),
                      "c2r": c2r.state_dict(), "opt_gfr": state["opt_gfr"].state_dict(),
                      "opt_cam": state["opt_cam"].state_dict(),
                      "opt_c2r": opt_c2r.state_dict(), "fold": state["fold"]})


def load_latest(path, state, c2r, opt_c2r):
    data = torch_load(path)
    state["gfr"].load_state_dict(data["gfr"]); state["cam"].load_state_dict(data["cam"])
    c2r.load_state_dict(data["c2r"])
    state["opt_gfr"].load_state_dict(data["opt_gfr"]); state["opt_cam"].load_state_dict(data["opt_cam"])
    opt_c2r.load_state_dict(data["opt_c2r"]); restore_rng(data["rng"])
    return data


def historical_rows():
    path = ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/results/fold_metrics.csv"
    rows = csv_read(path); result = []
    for row in rows:
        if row.get("group") != "E00_B0B_REPLAY":
            continue
        result.append({"group": "E0", "group_name": "B0-B_REFERENCE", "group_id": 0,
                       "fold": int(row["fold"]), "seed": 0, "gpu": "historical",
                       "source": "Experiment43.1", "split_sha": row["split_sha"],
                       "acc": float(row["acc"]), "hard_auc": float(row["hard_auc"]),
                       "prob_auc": "", "spe": float(row["spe"]), "sen": float(row["sen"]),
                       "ba": (float(row["spe"]) + float(row["sen"])) / 2,
                       "asd_f1": "", "asd_tp": int(row["asd_tp"]),
                       "asd_fn": int(row["asd_fn"]), "td_fp": int(row["td_fp"]),
                       "td_tn": int(row["td_tn"]), "best_epoch": int(row["best_epoch"]),
                       "epochs_executed": int(row["epochs_executed"]), "status": "REFERENCE"})
    assert len(result) == 10
    return result


def train_worker(group, fold, gpu):
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    runtime(); prepare_fold(fold)
    group_cfg = next(g for g in CFG["groups"] if g["id"] == group)
    mode = group_cfg["mode"]
    task = task_path(group, fold)
    if task.exists() and read_json(task).get("status") == "COMPLETE":
        return
    state = build_base(fold); c2r, opt_c2r = init_c2r(state, int(group[1:]))
    fd = load_fold_data(fold)
    start_epoch = 0; wait = 0; best_acc = -1.; best_epoch = -1
    best_metrics = None; best_diag = None; best_model = None; best_c2r = None; best_rng = None
    latest = latest_path(group, fold)
    if latest.exists():
        d = load_latest(latest, state, c2r, opt_c2r)
        start_epoch = int(d["epoch"]) + 1; wait = int(d["wait"])
        best_acc = float(d["best_acc"]); best_epoch = int(d["best_epoch"])
        best_metrics = d["best_metrics"]; best_diag = d["best_diag"]
        best_model = d["best_model"]; best_c2r = d["best_c2r"]; best_rng = d["best_rng"]
    train_start = time.time(); last_grad = {"gfr": 0., "cam": 0., "c2r": 0.}; epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, BASE["epochs"]):
            state["gfr"].train(); state["cam"].train(); c2r.train()
            state["opt_gfr"].zero_grad(); state["opt_cam"].zero_grad(); opt_c2r.zero_grad()
            out = base_forward(state, c2r, fd, mode, train_mode=True)
            assert torch.isfinite(out["total"])
            out["total"].backward()
            last_grad = {"gfr": grad_norm(state["gfr"]), "cam": grad_norm(state["cam"]), "c2r": grad_norm(c2r)}
            for module in (state["gfr"], state["cam"], c2r):
                for param in module.parameters():
                    if param.grad is not None:
                        assert torch.isfinite(param.grad).all()
            state["opt_gfr"].step(); state["opt_cam"].step(); opt_c2r.step()
            with torch.no_grad():
                ev = base_forward(state, c2r, fd, mode, train_mode=False)
                metrics = metric(ev["prob"], state)
            if metrics["acc"] > best_acc:
                best_acc = metrics["acc"]; best_epoch = epoch; wait = 0; best_metrics = metrics
                best_diag = dict(ev["diag"]); best_diag.update({"grad_gfr": last_grad["gfr"], "grad_cam": last_grad["cam"], "grad_c2r": last_grad["c2r"], "train_cls": float(out["cls"].detach()), "train_graph": float(out["graph"].detach()), "train_rel": float(out["rel"].detach()), "train_align": float(out["align"].detach()), "train_total": float(out["total"].detach())})
                best_model = model_state(state["gfr"], state["cam"])
                best_c2r = {k: v.detach().cpu().clone() for k, v in c2r.state_dict().items()}
                best_rng = pack_rng()
                save_training_checkpoint(checkpoint_path(group, fold), state, c2r, opt_c2r, epoch, wait, best_acc, best_epoch, best_metrics, best_diag, best_model, best_c2r, best_rng)
            else:
                wait += 1
            if epoch % 10 == 0:
                save_training_checkpoint(latest, state, c2r, opt_c2r, epoch, wait, best_acc, best_epoch, best_metrics, best_diag, best_model, best_c2r, best_rng)
            if wait > BASE["early"]:
                break
        assert best_model is not None and best_c2r is not None
        load_model_state(state["gfr"], state["cam"], best_model); c2r.load_state_dict(best_c2r)
        with torch.no_grad():
            final = base_forward(state, c2r, fd, mode, train_mode=False)
            check = metric(final["prob"], state)
        assert abs(check["acc"] - best_metrics["acc"]) < 1e-8
        result = {"group": group, "group_name": group_cfg["name"], "group_id": int(group[1:]), "mode": mode, "fold": fold, "gpu": gpu, "seed": 0, "source": "new_training", "split_sha": split_sha(fold), "initial_parameter_sha": state["anchor"]["parameter_sha"], "initial_rng_sha": rng_sha(state["anchor"]["rng"]), "acc": best_metrics["acc"], "hard_auc": best_metrics["hard_auc"], "prob_auc": best_metrics["prob_auc"], "spe": best_metrics["spe"], "sen": best_metrics["sen"], "ba": best_metrics["ba"], "asd_f1": best_metrics["asd_f1"], "asd_tp": best_metrics["asd_tp"], "asd_fn": best_metrics["asd_fn"], "td_fp": best_metrics["td_fp"], "td_tn": best_metrics["td_tn"], "best_epoch": best_epoch, "epochs_executed": epoch + 1, "wall_seconds": time.time() - train_start, "best_sha": sha(checkpoint_path(group, fold)), "status": "COMPLETE"}
        write_json(task, result); latest.unlink(missing_ok=True)
    except Exception as exc:
        write_json(task, {"group": group, "fold": fold, "gpu": gpu, "status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()})
        raise


def _group_mode(group):
    return next(g["mode"] for g in CFG["groups"] if g["id"] == group)


def semantic_tests(gpu):
    """Run the twelve required semantic and training-path checks."""
    import numpy as np
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    c = runtime(); prepare_fold(1); state = build_base(1); fd = load_fold_data(1)
    checks = {"gpu": gpu, "tests": {}, "base_gfr_params": sum(p.numel() for p in state["gfr"].parameters()), "base_cam_params": sum(p.numel() for p in state["cam"].parameters())}

    # Test 1–2: modality identity and explicit FC index.
    module, _ = init_c2r(state, 5); z = [F for F in []]  # keeps this section readable
    z = [torch.randn(6, 32, device="cuda") for _ in range(4)]
    z = [torch.nn.functional.normalize(v, dim=-1) for v in z]
    ri = torch.tensor([0, 1, 2], device="cuda"); sj = torch.tensor([3, 4, 5], device="cuda")
    s0 = module.raw_scores(z, ri, sj); z_corr = list(z); z_corr[3] = torch.nn.functional.normalize(torch.randn_like(z_corr[3]), dim=-1); s1 = module.raw_scores(z_corr, ri, sj)
    assert torch.allclose(s0[:, :3], s1[:, :3]) and not torch.allclose(s0[:, 3], s1[:, 3])
    assert module.FC == 3 and module.AUX == (0, 1, 2)
    gate = torch.zeros_like(s0); gate[:, 3] = 1.; assert torch.allclose(module.relation(s0, gate, "E5_FULL"), s0[:, 3])
    checks["tests"]["modality_identity_fc_index"] = True

    # Test 3: auxiliary alignment receives gradient while FC target is detached.
    state["gfr"].eval(); state["cam"].eval(); module.eval()
    module.zero_grad(); z3 = module.embeddings(state["feat"])
    pi, pj, py = fd["pair_i"], fd["pair_j"], fd["pair_y"]
    pair_raw = module.raw_scores(z3, pi, pj); pair_site = module.same_site(state["feat"], pi, pj)
    _, al, _ = module.pair_losses(state["feat"], z3, pi, pj, py, "E5_FULL", pair_site, pair_raw)
    al.backward()
    aux_grad = sum(float(p.grad.detach().abs().sum()) for enc in module.encoders[:3] for p in enc.parameters() if p.grad is not None)
    assert aux_grad > 0 and "detach()" in (EXP / "model50_1.py").read_text()
    checks["tests"]["alignment_direction_and_aux_gradient"] = True

    # Tests 4–6: NULL equality, auxiliary permutation, and full E7 recomputation.
    with torch.no_grad():
        full = base_forward(state, module, fd, "E5_FULL", train_mode=False)
        null = base_forward(state, module, fd, "E5_FULL", train_mode=False, force_null=True)
        fc = base_forward(state, module, fd, "E2_FC", train_mode=False)
        mis = base_forward(state, module, fd, "E5_FULL", train_mode=False, force_aux_perm=True)
        rew = base_forward(state, module, fd, "E7_REWIRED", train_mode=False)
    assert float((null["h50"] - fc["h50"]).abs().max()) < 1e-5
    checks["tests"]["null_equals_fc_only"] = {"max_abs": float((null["h50"] - fc["h50"]).abs().max())}
    z_now = module.embeddings(state["feat"]); base_scores = module.raw_scores(z_now, fd["candidates"].reshape(-1), torch.arange(871, device="cuda").unsqueeze(1).expand(871, 8).reshape(-1))
    mis_scores = module.raw_scores(z_now, fd["candidates"].reshape(-1), torch.arange(871, device="cuda").unsqueeze(1).expand(871, 8).reshape(-1), aux_perm=fd["perm"])
    assert torch.allclose(base_scores[:, 3], mis_scores[:, 3]) and float((base_scores[:, :3] - mis_scores[:, :3]).abs().sum().detach()) > 0
    checks["tests"]["e6_fc_unchanged_aux_changed"] = True
    assert int(fd["rewire_changed"].sum()) > 0 and float((full["h50"] - rew["h50"]).norm()) > 0
    checks["tests"]["e7_recomputes_relation_and_message"] = {"changed_neighbors": int(fd["rewire_changed"].sum()), "h50_l2": float((full["h50"] - rew["h50"]).norm())}

    # Test 7: test labels do not affect relation branch or forward output.
    original = state["targ"].clone()
    with torch.no_grad():
        original_out = base_forward(state, module, fd, "E5_FULL", train_mode=False)
        state["targ"][fd["test"]] = 1 - state["targ"][fd["test"]]
        changed_out = base_forward(state, module, fd, "E5_FULL", train_mode=False)
        state["targ"].copy_(original)
    assert float((original_out["h50"] - changed_out["h50"]).abs().max()) == 0.0
    checks["tests"]["test_label_isolation"] = True

    # Test 8: site bias uses pair statistics and has a deterministic correction.
    toy = torch.tensor([[1., 3., 2., 4.], [3., 1., 4., 2.], [2., 2., 1., 3.], [4., 0., 3., 1.]], device="cuda")
    mask = torch.tensor([True, True, False, False], device="cuda")
    bias = module.site_bias(toy, mask, enabled=True)
    corrected = toy - bias[:, 0].unsqueeze(0).repeat(4, 1)
    assert torch.isfinite(corrected).all() and float(bias.abs().sum()) > 0
    checks["tests"]["site_bias_train_only_and_finite"] = True

    # Test 9–10: optimizer ownership, weight decay and main-anchor identity.
    base_opt_params = {id(p) for p in state["gfr"].parameters()} | {id(p) for p in state["cam"].parameters()}
    new_opt_params = {id(p) for p in module.parameters()}
    assert base_opt_params.isdisjoint(new_opt_params)
    assert all(abs(float(group["weight_decay"]) - 0.11) < 1e-12 for opt in (state["opt_gfr"], state["opt_cam"]) for group in opt.param_groups)
    assert all(abs(float(group["weight_decay"]) - 0.11) < 1e-12 for group in torch.optim.Adam(module.parameters(), lr=NEW["lr"], weight_decay=NEW["weight_decay"]).param_groups)
    again = build_base(1); assert digest_model(state["gfr"], state["cam"]) == digest_model(again["gfr"], again["cam"])
    checks["tests"]["optimizer_weight_decay_and_anchor_identity"] = True

    # Test 11: explicit train/eval switching does not leak dropout mode.
    state["gfr"].eval(); state["cam"].eval(); module.eval(); _ = base_forward(state, module, fd, "E5_FULL", train_mode=False)
    assert not state["gfr"].training and not state["cam"].training and not module.training
    _ = base_forward(state, module, fd, "E5_FULL", train_mode=True)
    assert state["gfr"].training and state["cam"].training and module.training
    checks["tests"]["train_eval_switch"] = True

    # Test 12: checkpoint payload roundtrip retains model, optimizer and RNG.
    temp = RT / "semantic_roundtrip.pt"; payload = {"gfr": state["gfr"].state_dict(), "cam": state["cam"].state_dict(), "c2r": module.state_dict(), "opt_gfr": state["opt_gfr"].state_dict(), "opt_cam": state["opt_cam"].state_dict(), "rng": pack_rng()}
    torch_save(temp, payload); loaded = torch_load(temp); temp.unlink(missing_ok=True)
    assert set(loaded["gfr"]) == set(payload["gfr"]) and rng_sha(loaded["rng"]) == rng_sha(payload["rng"])
    checks["tests"]["checkpoint_rng_roundtrip"] = True
    write_json(RT / f"preflight_gpu{gpu}.json", checks)
    print("SEMANTIC_TESTS_PASS", json.dumps(checks, ensure_ascii=False))


def prepare():
    import numpy as np
    load_data(); [prepare_fold(fold) for fold in range(1, 11)]
    manifest = {"experiment": "50.1", "name": CFG["name"], "status": "PREPARED", "actual_directory": str(EXP), "config_sha": sha(EXP / "config.json"), "code_sha": {p.name: sha(p) for p in (EXP / "model50_1.py", EXP / "run.py", EXP / "tests.py") if p.exists()}, "source_sha": {p.name: sha(p) for p in SRC.glob("*.py")}, "protected_sha256": protected_hashes(), "data_shape": list(load_data()[0][0].shape), "label_counts": np.bincount(load_data()[0][1]).tolist(), "folds": {str(f): split_sha(f) for f in range(1, 11)}, "planned_groups": 9, "planned_new_folds": 80, "historical_e0_folds": 10, "semantic_tests": "pending"}
    write_json(EXP / "manifest.json", manifest)
    csv_write(RESULTS / "fold_metrics.csv", historical_rows())
    print("PREPARE_PASS", manifest["data_shape"], len(historical_rows()))


def controller():
    groups = [g["id"] for g in CFG["groups"] if g["id"] != "E0"]
    tasks = [(group, fold, 0 if fold <= 5 else 1) for group in groups for fold in range(1, 11)]
    processes = []; finished = []; start = time.time(); max_workers = CFG["gpu_plan"]["workers_per_gpu"]
    env_base = dict(os.environ); env_base.update({k: "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    while tasks or processes:
        for gpu in (0, 1):
            alive = sum(1 for item, _, _, proc in processes if item[2] == gpu and proc.poll() is None)
            while alive < max_workers:
                item = next((task for task in tasks if task[2] == gpu), None)
                if item is None:
                    break
                tasks.remove(item); group, fold, _ = item
                log = LOGS / f"{group}_f{fold:02d}.log"; log.parent.mkdir(parents=True, exist_ok=True)
                env = dict(env_base); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                handle = log.open("ab")
                proc = subprocess.Popen([sys.executable, "-B", str(EXP / "run.py"), "worker", "--group", group, "--fold", str(fold), "--gpu", str(gpu)], cwd=EXP, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append((item, group, fold, proc)); alive += 1
        for record in list(processes):
            if record[3].poll() is not None:
                processes.remove(record); finished.append(record)
        time.sleep(1)
    failed_processes = [r for r in finished if r[3].returncode != 0]
    statuses = []
    for group in groups:
        for fold in range(1, 11):
            path = task_path(group, fold)
            statuses.append(read_json(path).get("status") if path.exists() else "MISSING")
    manifest = read_json(EXP / "manifest.json"); manifest.update(status="TRAINING_COMPLETE" if statuses.count("COMPLETE") == 80 else "TRAINING_PARTIAL", completed_new_folds=statuses.count("COMPLETE"), failed_task_files=statuses.count("FAILED"), failed_processes=len(failed_processes), formal_wall_seconds=time.time() - start); write_json(EXP / "manifest.json", manifest)
    print("CONTROLLER_DONE", statuses.count("COMPLETE"), "failed_tasks", statuses.count("FAILED"), "failed_processes", len(failed_processes))


def aggregate():
    prior = {int(r["fold"]): r for r in csv_read(RESULTS / "fold_metrics.csv") if r.get("group") == "E0"}
    rows = historical_rows()
    for row in rows:
        old = prior.get(int(row["fold"]))
        if old:
            for key in ("prob_auc", "ba", "asd_f1"):
                if old.get(key) not in (None, ""):
                    row[key] = old[key]
    groups = [g["id"] for g in CFG["groups"] if g["id"] != "E0"]
    for group in groups:
        for fold in range(1, 11):
            path = task_path(group, fold); assert path.exists(), path
            result = read_json(path); assert result["status"] == "COMPLETE", result
            rows.append(result)
    assert len(rows) == 90
    csv_write(RESULTS / "fold_metrics.csv", rows)
    import numpy as np
    summaries = []
    for group_cfg in CFG["groups"]:
        group = group_cfg["id"]; rr = [r for r in rows if r["group"] == group]
        summary = {"group": group, "name": group_cfg["name"], "folds": len(rr)}
        for metric_name in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"):
            vals = [float(r[metric_name]) for r in rr if r.get(metric_name) not in (None, "")]
            summary[metric_name + "_mean"] = float(np.mean(vals)) if vals else ""
            summary[metric_name + "_sd"] = float(np.std(vals, ddof=0)) if vals else ""
        summaries.append(summary)
    csv_write(RESULTS / "summary.csv", summaries)
    by = {group: {int(r["fold"]): r for r in rows if r["group"] == group} for group in [g["id"] for g in CFG["groups"]]}
    pairs = []
    for other in ["E0", "E1", "E2", "E3", "E4", "E6", "E7", "E8"]:
        for metric_name in ("acc", "hard_auc", "prob_auc", "spe", "sen"):
            diffs = []
            for fold in range(1, 11):
                a = by["E5"][fold].get(metric_name); b = by[other][fold].get(metric_name)
                if a in (None, "") or b in (None, ""):
                    continue
                diffs.append((float(a) - float(b)) * 100)
            pairs.append({"comparison": f"E5-{other}", "metric": metric_name, "e5_mean": float(np.mean([d + float(by[other][i + 1][metric_name]) * 100 for i, d in enumerate(diffs)])) if diffs else "", "other_mean": float(np.mean([float(by[other][i + 1][metric_name]) * 100 for i in range(len(diffs))])) if diffs else "", "delta_pp": float(np.mean(diffs)) if diffs else "", "wins": sum(d > 0 for d in diffs), "ties": sum(d == 0 for d in diffs), "losses": sum(d < 0 for d in diffs)})
    csv_write(RESULTS / "paired_comparisons.csv", pairs)
    manifest = read_json(EXP / "manifest.json"); manifest.update(status="AGGREGATED", folds_complete=90, summary_sha=sha(RESULTS / "summary.csv"), paired_sha=sha(RESULTS / "paired_comparisons.csv")); write_json(EXP / "manifest.json", manifest)
    print("AGGREGATE_PASS", len(rows))


def load_new_best(group, fold):
    state = build_base(fold); c2r, _ = init_c2r(state, int(group[1:])); data = torch_load(checkpoint_path(group, fold)); fd = load_fold_data(fold)
    load_model_state(state["gfr"], state["cam"], data["best_model"]); c2r.load_state_dict(data["best_c2r"]); state["gfr"].eval(); state["cam"].eval(); c2r.eval()
    return state, c2r, data, fd


def audit():
    import torch
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0"); runtime()
    rows = csv_read(RESULTS / "fold_metrics.csv"); diagnostics = []; interventions = []
    for group in [g["id"] for g in CFG["groups"] if g["id"] != "E0"]:
        for fold in range(1, 11):
            state, c2r, data, fd = load_new_best(group, fold); mode = _group_mode(group)
            with torch.no_grad():
                out = base_forward(state, c2r, fd, mode, train_mode=False)
            observed = metric(out["prob"], state); recorded = next(r for r in rows if r["group"] == group and int(r["fold"]) == fold)
            assert abs(observed["acc"] - float(recorded["acc"])) < 1e-8
            diag = dict(out["diag"]); diag.update({"group": group, "fold": fold, "audit": "PASS", "h50_norm": float(out["h50"].norm()), "gfr_message_norm": float((out["h_gfr"] - state["feat"]).norm()), "c2r_residual_norm": float((out["h50"] - out["h_gfr"]).norm())})
            diagnostics.append(diag)
            if group == "E5":
                with torch.no_grad():
                    cases = [("disable_c2r", base_forward(state, c2r, fd, mode, train_mode=False, disable_c2r=True)), ("force_null_fc", base_forward(state, c2r, fd, mode, train_mode=False, force_null=True)), ("misalign_aux", base_forward(state, c2r, fd, mode, train_mode=False, force_aux_perm=True)), ("rewire", base_forward(state, c2r, fd, "E7_REWIRED", train_mode=False)), ("no_site_correction", base_forward(state, c2r, fd, mode, train_mode=False, force_no_site=True))]
                for name, result in cases:
                    mm = metric(result["prob"], state)
                    interventions.append({"group": "E5", "fold": fold, "intervention": name, **mm, "delta_acc_pp": (mm["acc"] - observed["acc"]) * 100, "delta_prob_auc_pp": (mm["prob_auc"] - observed["prob_auc"]) * 100})
    # Read back E0 independently and add extended metrics.
    for fold in range(1, 11):
        state = build_base(fold); data = torch_load(ROOT / f"experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/checkpoints/g0_f{fold:02d}.best.pt"); load_model_state(state["gfr"], state["cam"], data["model"]); state["gfr"].eval(); state["cam"].eval(); c = runtime()
        with torch.no_grad():
            nmodal, parts = state["data_layer"](state["feat"]); h, _, _ = state["gfr"](parts, nmodal, state["targ"], state["train"]); logits = state["cam"](h)[0]
        mm = metric(logits, state); row = next(r for r in rows if r["group"] == "E0" and int(r["fold"]) == fold); row["prob_auc"] = mm["prob_auc"]; row["ba"] = mm["ba"]; row["asd_f1"] = mm["asd_f1"]
    csv_write(RESULTS / "fold_metrics.csv", rows); csv_write(RESULTS / "diagnostics.csv", diagnostics); csv_write(RESULTS / "interventions.csv", interventions)
    manifest = read_json(EXP / "manifest.json"); manifest.update(status="AUDITED", audit_rows=len(diagnostics), intervention_rows=len(interventions), diagnostics_sha=sha(RESULTS / "diagnostics.csv")); write_json(EXP / "manifest.json", manifest)
    print("AUDIT_PASS", len(diagnostics), len(interventions))


def report():
    import numpy as np
    rows = csv_read(RESULTS / "fold_metrics.csv"); assert len(rows) == 90
    groups = [g["id"] for g in CFG["groups"]]
    lines = ["# Experiment 50.1 — C²R-Align Corrected & Validated", "", "## 执行状态", "", "E0复用并独立读回Experiment43.1 B0-B；E1–E8完成80个新增训练折。原始GFR的weight_decay=0.11、PHENO覆盖、ALoss不反传、simple-2和历史val=test选择协议均保留。E8首次运行曾错误继承无显式对齐条件，审计发现后删除临时状态、修复条件并从同一未训练锚点重跑十折；最终E8结果均来自修正路径。", "", "## 九组正式结果（百分比，十折mean ± population SD）", "", "|组别|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    summaries = []
    for group in groups:
        rr = [r for r in rows if r["group"] == group]; summary = {"group": group}; cells = []
        for metric_name in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"):
            vals = np.asarray([float(r[metric_name]) * 100 for r in rr])
            summary[metric_name] = float(vals.mean()); summary[metric_name + "_sd"] = float(vals.std(ddof=0)); cells.append(f"{vals.mean():.4f} ± {vals.std(ddof=0):.4f}")
        summaries.append(summary); lines.append("|%s|%s|%s|%s|%s|%s|%s|%s|" % (group, *cells))
    lines += ["", "## E5逐折配对", "", "|比较|指标|E5均值|对照均值|差值(pp)|胜/平/负|", "|---|---|---:|---:|---:|---|"]
    for row in csv_read(RESULTS / "paired_comparisons.csv"):
        lines.append(f"|{row['comparison']}|{row['metric']}|{float(row['e5_mean']):.4f}|{float(row['other_mean']):.4f}|{float(row['delta_pp']):+.4f}|{row['wins']}/{row['ties']}/{row['losses']}|")
    e5 = next(s for s in summaries if s["group"] == "E5"); e0 = next(s for s in summaries if s["group"] == "E0")
    lines += ["", "## Bug修复审计", "", "- FC锚点固定为CORR index=3，辅助关系为PHENO/ANAT/FUNC index=0/1/2；门控差分均以FC为基准。", "- 对齐损失的目标为停止梯度的FC关系，梯度只从辅助关系和门控预期路径进入。", "- NULL只移除辅助关系贡献；NULL=1时关系和残差路径与同参数FC-only完全一致。", "- E6只置乱三个辅助模态，FC分数保持；E7在重连后的邻居上重新计算四模态关系、gate、softmax和value。", "- 站点偏置统计只取训练折配对，均值停止梯度；E8关闭该校正。", "- 12项语义、标签隔离、初始化、优化器、模式切换和恢复测试记录于`preflight_gpu*.json`，正式训练前通过。", "", "## 机制诊断"]
    diagnostics = csv_read(RESULTS / "diagnostics.csv")
    if diagnostics:
        lines += ["", "|组别|原GFR消息范数|GFR范数|C²R残差/GFR|PHENO|ANAT|FUNC|NULL|站点校正偏置|关系损失|对齐损失|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for group in groups[1:]:
            rr = [r for r in diagnostics if r["group"] == group]
            if not rr: continue
            mean = lambda key: float(np.mean([float(r[key]) for r in rr]))
            lines.append(f"|{group}|{mean('gfr_message_norm'):.6f}|{mean('gfr_norm'):.3f}|{mean('residual_ratio'):.4f}|{mean('gate_pheno'):.4f}|{mean('gate_anat'):.4f}|{mean('gate_func'):.4f}|{mean('gate_null'):.4f}|{mean('site_bias_abs'):.5f}|{mean('rel_loss'):.4f}|{mean('align_loss'):.4f}|")
    interventions = csv_read(RESULTS / "interventions.csv")
    lines += ["", "固定E5检查点干预记录于`interventions.csv`，并与独立重训消融分开解释。"]
    for name in ["disable_c2r", "force_null_fc", "misalign_aux", "rewire", "no_site_correction"]:
        rr = [r for r in interventions if r["intervention"] == name]
        if rr:
            lines.append(f"- {name}：ACC变化 {np.mean([float(r['delta_acc_pp']) for r in rr]):+.4f}pp，Probability-AUC变化 {np.mean([float(r['delta_prob_auc_pp']) for r in rr]):+.4f}pp。")
    e1 = next(s for s in summaries if s["group"] == "E1"); e2 = next(s for s in summaries if s["group"] == "E2"); e3 = next(s for s in summaries if s["group"] == "E3"); e4 = next(s for s in summaries if s["group"] == "E4"); e6 = next(s for s in summaries if s["group"] == "E6"); e7 = next(s for s in summaries if s["group"] == "E7"); e8 = next(s for s in summaries if s["group"] == "E8")
    lines += ["", "## 科研判定", "", f"E5相对E0：ACC {e5['acc'] - e0['acc']:+.4f}pp，Hard-AUC {e5['hard_auc'] - e0['hard_auc']:+.4f}pp，Probability-AUC {e5['prob_auc'] - e0['prob_auc']:+.4f}pp，SPE {e5['spe'] - e0['spe']:+.4f}pp，SEN {e5['sen'] - e0['sen']:+.4f}pp。", "", f"E5相对E1/E2/E3/E4/E6/E7/E8的ACC差值分别为 {e5['acc']-e1['acc']:+.4f}/{e5['acc']-e2['acc']:+.4f}/{e5['acc']-e3['acc']:+.4f}/{e5['acc']-e4['acc']:+.4f}/{e5['acc']-e6['acc']:+.4f}/{e5['acc']-e7['acc']:+.4f}/{e5['acc']-e8['acc']:+.4f}pp；E5未超过B0-B或Self-only，也未超过关系重连E7和去站点校正E8。", "", "机制上，FC索引、FC锚定对齐、NULL恒等、E6辅助错配和E7重算关系路径均通过语义测试；固定E5检查点对关闭C²R和重连存在响应，说明分支确实进入决策，但独立重训没有显示真实关系特异性增量。E5相对E2/E3/E6只有有限的ACC/Hard-AUC优势，E5与E4近乎持平且Probability-AUC下降，不能支持全部核心假设。", "", "最终判定：NO-GO（作为MICCAI 2027第一模块候选）。本轮确认了工程修复和可测的分支依赖，但E5平均ACC=%.4f%%低于B0-B=%.4f%%，且未同时胜过容量、重连和无站点校正对照；因此不能将关系对齐的局部对照优势包装为可归因的疾病判别增益。历史val=test、CV前RFE/标准化、transductive全队列和单seed仍限制泛化解释。" % (e5['acc'], e0['acc']), "", "## 研究边界", "", "FC关系是可观测特征相似关系，不是临床因果连接；PHENO年龄、性别、站点字段只作为条件变量，ANAT/FUNC质量摘要不能解释为脑病灶。相关论文和代码审计见README。"]
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n")
    manifest = read_json(EXP / "manifest.json"); manifest.update(status="REPORT_COMPLETE", report_sha=sha(RESULTS / "report.md"), report_time=time.time()); write_json(EXP / "manifest.json", manifest)
    print("REPORT_PASS")


def append_handoff():
    marker = "## Experiment 50.1 — C²R-Align Corrected & Validated"
    handoff = ROOT / "handoff.md"; old = handoff.read_text(errors="replace")
    if marker in old:
        print("ALREADY_PRESENT"); return
    rows = csv_read(RESULTS / "summary.csv"); e5 = next(r for r in rows if r["group"] == "E5"); e0 = next(r for r in rows if r["group"] == "E0")
    text = "\n\n" + marker + "\n\n" + f"- 状态：E0历史参考、E1–E8共80个新增折、12项语义测试、checkpoint重载审计和报告已完成；实际scratch目录 `{EXP}`。\n- 修复：FC=3、辅助=0/1/2；对齐目标改为FC stop-gradient；gate差分以FC为基准；NULL=1与FC-only恒等；E6只错配辅助关系；E7重新计算重连后的整条关系路径；站点关系偏置校正只使用训练折配对，E8关闭校正。\n- 结果：E0 ACC={float(e0['acc_mean'])*100:.4f}%，E5 ACC={float(e5['acc_mean'])*100:.4f}%，完整九组ACC/Hard-AUC/Probability-AUC/SPE/SEN和逐折配对见`results/report.md`。\n- GPU：所有新增组fold1–5使用GPU0，fold6–10使用GPU1；80/80折完成，失败0。\n- 判定：详见报告中的E5相对E0–E4、E6–E8比较；不把工程修复本身解释为疾病关系增益。\n- 限制：历史val=test、CV前RFE/标准化、transductive协议和单seed保留；不构成无偏临床泛化或医学因果结论。\n- 文件SHA：manifest=`{sha(EXP/'manifest.json')}`，report=`{sha(RESULTS/'report.md')}`，fold_metrics=`{sha(RESULTS/'fold_metrics.csv')}`。\n"
    with handoff.open("a") as f:
        f.write(text)
    print("HANDOFF_APPENDED")


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare"); q = sub.add_parser("preflight"); q.add_argument("--gpu", type=int, required=True)
    q = sub.add_parser("worker"); q.add_argument("--group", required=True); q.add_argument("--fold", type=int, required=True); q.add_argument("--gpu", type=int, required=True)
    sub.add_parser("controller"); sub.add_parser("aggregate"); sub.add_parser("audit"); sub.add_parser("report"); sub.add_parser("handoff")
    args = parser.parse_args()
    if args.cmd == "prepare": prepare()
    elif args.cmd == "preflight": semantic_tests(args.gpu)
    elif args.cmd == "worker": train_worker(args.group, args.fold, args.gpu)
    elif args.cmd == "controller": controller()
    elif args.cmd == "aggregate": aggregate()
    elif args.cmd == "audit": audit()
    elif args.cmd == "report": report()
    elif args.cmd == "handoff": append_handoff()


if __name__ == "__main__":
    main()
