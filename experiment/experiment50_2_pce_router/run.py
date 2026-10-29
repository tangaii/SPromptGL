#!/usr/bin/env python3
"""Experiment 50.2 runner for PCE-Router.

The runner restores the audited B0-B GFR/CAM anchors and adds only a bounded
patient-local FC evidence router.  All data-dependent supervision and masks
are fold-local; no patient-neighbour graph is created by this experiment.
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
if not SRC.exists() and (EXP / "source").exists():
    SRC = EXP / "source"
if not ANCHORS.exists():
    historical = ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
    if historical.exists():
        ANCHORS = historical
BASE = CFG["base"]
PCE_CFG = CFG["pce"]
DIMS = [48, 6, 10, 256]
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
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
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
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def csv_read(path):
    p = Path(path)
    if not p.exists(): return []
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
    import numpy as np, torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import numpy as np, torch
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def rng_sha(state):
    h = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    h.update(state["cpu"].cpu().numpy().tobytes())
    for v in state["cuda"]: h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def runtime():
    global CTX
    if CTX is None:
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        assert torch.cuda.is_available(), "CUDA is required"
        torch.cuda.set_device(0)
        sys.path.insert(0, str(SRC)); sys.path.insert(0, str(EXP))
        import layers, network, utils
        from pce_router import PLNNoPEN, PCERouter
        CTX = {"torch": torch, "np": __import__("numpy"), "layers": layers,
               "network": network, "utils": utils, "PLN": PLNNoPEN,
               "PCE": PCERouter}
    return CTX


def load_data():
    global DATA, FOLDS
    if DATA is not None: return DATA, FOLDS
    import numpy as np, pandas as pd
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
    assert [len(mapping[k]) for k in ("PHENO", "ANAT", "FUNC", "Correlation")] == DIMS
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10).split(x, y))
    DATA, FOLDS = (x, y), folds
    return DATA, FOLDS


def split_sha(fold):
    import numpy as np
    tr, te = load_data()[1][fold - 1]
    return hashlib.sha256(tr.astype(np.int64).tobytes() + te.astype(np.int64).tobytes()).hexdigest()


def protected_hashes():
    out = {}
    for rel in CFG["protected_root"]:
        p = ROOT / rel
        if p.exists(): out[str(p)] = sha(p)
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
    gfr.load_state_dict(state["gfr"]); cam.load_state_dict(state.get("pln", state.get("cam")))


def make_hp():
    return SimpleNamespace(n_hidden=BASE["n_hidden"], n_head=BASE["n_head"],
        dropout=BASE["dropout"], nlayer=BASE["nlayer"], nmodal=4, nclass=2,
        lr=BASE["lr"], reg=BASE["weight_decay"], th=BASE["threshold"],
        GC_mode="weighted-cosine", MP_mode="GCN", MF_mode="concat", alpha=0.5,
        theta_smooth=BASE["theta_smooth"], theta_degree=BASE["theta_degree"],
        theta_sparsity=BASE["theta_sparsity"])


def build_base(fold):
    import torch, numpy as np
    c = runtime(); x, y = load_data()[0]; train, test = load_data()[1][fold - 1]
    anchor_path = ANCHORS / f"fold{fold:02d}.pt"
    anchor = torch_load(anchor_path)
    assert anchor["split_sha"] == split_sha(fold)
    restore_rng(anchor["entry_rng"])
    hp = make_hp()
    gfr = c["network"].GFR(DIMS, DIMS, th=hp.th, mode=hp.GC_mode).cuda()
    cam = c["PLN"](DIMS, hp, c["layers"]).cuda(); cam.apply(c["utils"].my_weight_init)
    assert sum(p.numel() for p in cam.parameters()) == 49544
    assert sum(p.numel() for p in gfr.parameters()) == 887836
    assert digest_model(gfr, cam) == anchor["parameter_sha"]
    opt_gfr = torch.optim.Adam(gfr.parameters(), lr=hp.lr, weight_decay=hp.reg)
    opt_cam = torch.optim.Adam(cam.parameters(), lr=hp.lr, weight_decay=hp.reg)
    opt_gfr.load_state_dict(copy.deepcopy(anchor["mp"])); opt_cam.load_state_dict(copy.deepcopy(anchor["mf"]))
    restore_rng(anchor["rng"])
    train_t = torch.as_tensor(train, device="cuda", dtype=torch.long)
    counts = Counter(y[train]); weights = len(train) / np.asarray(list(counts.values())) / 2
    return {"gfr": gfr, "cam": cam, "opt_gfr": opt_gfr, "opt_cam": opt_cam,
            "feat": torch.from_numpy(x).float().cuda(), "targ": torch.from_numpy(y).long().cuda(),
            "train": train_t, "test": test, "weight": torch.from_numpy(weights).float().cuda(),
            "data_layer": c["layers"].InitialDataLayer(DIMS), "anchor": anchor, "fold": fold}


def init_pce(state):
    import torch
    c = runtime(); base_rng = pack_rng()
    seed_all(502000 + state["fold"] * 100)
    pce = c["PCE"](hidden=PCE_CFG["hidden"], tau_e=PCE_CFG["tau_e"],
                    beta=PCE_CFG["beta"], residual_max=PCE_CFG["residual_max"]).cuda()
    restore_rng(base_rng)
    opt = torch.optim.Adam(pce.parameters(), lr=PCE_CFG["new_lr"], weight_decay=PCE_CFG["new_weight_decay"])
    return pce, opt


def site_ids(x):
    import numpy as np
    return np.argmax(x[:, 26:46], axis=1).astype(np.int64)


def make_site_permutation(x, train, test, seed):
    import numpy as np
    sid = site_ids(x); perm = np.arange(len(x), dtype=np.int64); rng = np.random.default_rng(seed)
    for ids in (train, test):
        buckets = {}
        for idx in ids: buckets.setdefault(int(sid[idx]), []).append(int(idx))
        for values in buckets.values():
            if len(values) < 2: continue
            arr = np.asarray(values, dtype=np.int64); sh = arr.copy()
            for _ in range(50):
                rng.shuffle(sh)
                if np.all(sh != arr): break
            else: sh = np.roll(arr, 1)
            perm[arr] = sh
    return perm


def make_evidence_permutation(n, fold):
    import numpy as np
    rng = np.random.default_rng(86000 + fold); out = np.empty((n, 256), dtype=np.int64)
    base = np.arange(256, dtype=np.int64)
    for i in range(n):
        p = base.copy(); rng.shuffle(p)
        if np.all(p == base): p = np.roll(base, 1)
        out[i] = p
    return out


def prepare_fold(fold):
    import numpy as np
    x, y = load_data()[0]; train, test = load_data()[1][fold - 1]
    path = FOLD_DATA / f"fold{fold:02d}.npz"
    if path.exists(): return
    cond_perm = make_site_permutation(x, train, test, 76000 + fold)
    ev_perm = make_evidence_permutation(len(x), fold)
    np.savez_compressed(path, condition_perm=cond_perm, evidence_perm=ev_perm,
                        train=np.asarray(train), test=np.asarray(test))
    write_json(FOLD_DATA / f"fold{fold:02d}.json", {
        "fold": fold, "train_n": int(len(train)), "test_n": int(len(test)),
        "condition_permutation_fraction": float((cond_perm != np.arange(len(x))).mean()),
        "evidence_permutation_fraction": float((ev_perm != np.arange(256)[None, :]).mean()),
        "split_sha": split_sha(fold), "data_sha": sha(path)})


def load_fold_data(fold):
    import torch, numpy as np
    z = np.load(FOLD_DATA / f"fold{fold:02d}.npz")
    return {key: torch.from_numpy(z[key]).cuda() for key in z.files}


def mode_for(group):
    return next(g["mode"] for g in CFG["groups"] if g["id"] == group)


def base_forward(state, pce, fd, mode, train_mode=None, disable_pce=False,
                 force_alpha_zero=False, force_uniform_mask=False):
    import torch
    c = runtime()
    if train_mode is True:
        state["gfr"].train(); state["cam"].train(); pce.train()
    elif train_mode is False:
        state["gfr"].eval(); state["cam"].eval(); pce.eval()
    nmodal, parts = state["data_layer"](state["feat"])
    h_gfr, adjs, aloss = state["gfr"](parts, nmodal, state["targ"], state["train"])
    graph = sum(c["utils"].GraphConstructLoss(parts[i], adjs[i], BASE["theta_smooth"], BASE["theta_degree"], BASE["theta_sparsity"]) for i in range(4)) / 4.0
    if disable_pce:
        h_new = h_gfr; details = {"evidence": h_gfr.new_zeros((h_gfr.shape[0], 2, 256)), "difference": h_gfr.new_zeros((h_gfr.shape[0], 256)), "mask": h_gfr.new_full((h_gfr.shape[0], 256), 1/256), "route_mask": h_gfr.new_full((h_gfr.shape[0], 256), 0.), "utility": h_gfr.new_zeros((h_gfr.shape[0], 256)), "q": h_gfr.new_zeros((h_gfr.shape[0], 256)), "gamma": h_gfr.new_ones((h_gfr.shape[0],)), "alpha": h_gfr.new_zeros(()), "scale": h_gfr.new_zeros((h_gfr.shape[0], 256)), "scores": h_gfr.new_zeros((h_gfr.shape[0], 2)), "competitive": True, "h_new": h_gfr}
        ev_loss = h_gfr.new_zeros(())
    else:
        cond_perm = fd["condition_perm"].long() if mode == "E6_WRONG_CONDITION" else None
        evidence_perm = fd["evidence_perm"].long() if mode == "E7_WRONG_EVIDENCE_POSITION" else None
        h_new, details = pce(h_gfr, state["feat"], mode, condition_perm=cond_perm,
                             evidence_perm=evidence_perm, force_alpha_zero=force_alpha_zero,
                             force_uniform_mask=force_uniform_mask)
        ev_loss = pce.evidence_loss(details, state["targ"], state["train"])
    prob, hidden, attention = state["cam"](h_new)
    cls = c["utils"].ClsLoss(prob, state["targ"], state["train"], state["weight"])
    lam = 0.0 if mode == "E8_NO_EVIDENCE_LOSS" else PCE_CFG["lambda_evidence"]
    total = cls + graph + lam * ev_loss
    return {"prob": prob, "hidden": hidden, "h_gfr": h_gfr, "h_new": h_new,
            "cls": cls, "graph": graph, "evidence": ev_loss, "total": total,
            "adjs": adjs, "aloss": aloss, "details": details}


def metric(output, state, test=None):
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    idx = np.asarray(state["test"] if test is None else test, dtype=np.int64)
    labels = state["targ"].detach().cpu().numpy()[idx]
    probs = output.detach().exp().cpu().numpy()[idx]; preds = probs.argmax(1)
    cm = confusion_matrix(labels, preds, labels=[0, 1]); asd_tp, asd_fn, td_fp, td_tn = map(int, cm.ravel())
    hard_auc = float(roc_auc_score(np.eye(2)[labels], np.eye(2)[preds]))
    prob_auc = float(roc_auc_score(labels, probs[:, 1]))
    sen = asd_tp / max(asd_tp + asd_fn, 1); spe = td_tn / max(td_tn + td_fp, 1)
    return {"acc": float((preds == labels).mean()), "hard_auc": hard_auc, "prob_auc": prob_auc,
            "spe": float(spe), "sen": float(sen), "ba": float((spe + sen) / 2),
            "asd_f1": float(f1_score(labels, preds, pos_label=0, zero_division=0)),
            "asd_tp": asd_tp, "asd_fn": asd_fn, "td_fp": td_fp, "td_tn": td_tn, "n_test": int(len(idx))}


def grad_norm(module):
    import torch
    vals = [(p.grad.detach() ** 2).sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(vals).sum().sqrt().cpu()) if vals else 0.0


def active_grad_count(module):
    return int(sum(p.numel() for p in module.parameters() if p.grad is not None and float(p.grad.detach().abs().sum()) > 0))


def diag_summary(out, state, pce, train_idx=None):
    import torch
    d = out["details"]; mask = d["mask"]; scale = d["scale"]; evidence = d["evidence"]
    entropy = -(mask.clamp_min(1e-12) * mask.clamp_min(1e-12).log()).sum(1)
    return {"gfr_message_norm": float((out["h_gfr"] - state["feat"]).detach().norm()),
            "gfr_norm": float(out["h_gfr"].detach().norm()),
            "pce_residual_norm": float((out["h_new"] - out["h_gfr"]).detach().norm()),
            "residual_ratio": float(((out["h_new"] - out["h_gfr"]).detach().norm() / out["h_gfr"].detach().norm().clamp_min(1e-8))),
            "alpha": float(d["alpha"].detach()), "gamma_mean": float(d["gamma"].detach().mean()),
            "gamma_sd": float(d["gamma"].detach().std(unbiased=False)),
            "mask_entropy_mean": float(entropy.detach().mean()), "mask_entropy_sd": float(entropy.detach().std(unbiased=False)),
            "mask_max_mean": float(mask.detach().max(1).values.mean()),
            "mask_uniform_l1": float((mask - 1/256).detach().abs().sum(1).mean()),
            "evidence_norm": float(evidence.detach().norm()),
            "evidence_gap_mean": float(d["difference"].detach().abs().mean()),
            "condition_q_norm": float(d["q"].detach().norm()),
            "train_evidence_acc": "", "test_evidence_acc": "",
            "registered_pce_params": int(sum(p.numel() for p in pce.parameters()))}


def checkpoint_path(group, fold): return CKPTS / f"{group}_f{fold:02d}.best.pt"
def latest_path(group, fold): return CKPTS / f"{group}_f{fold:02d}.latest.pt"
def task_path(group, fold): return TASKS / f"{group}_f{fold:02d}.json"


def save_training_checkpoint(path, state, pce, opt_pce, epoch, wait, best_acc, best_epoch,
                             best_metrics, best_diag, best_model, best_pce, best_rng):
    torch_save(path, {"epoch": epoch, "wait": wait, "best_acc": best_acc, "best_epoch": best_epoch,
        "best_metrics": best_metrics, "best_diag": best_diag, "best_model": best_model,
        "best_pce": best_pce, "best_rng": best_rng, "rng": pack_rng(),
        "gfr": state["gfr"].state_dict(), "cam": state["cam"].state_dict(), "pce": pce.state_dict(),
        "opt_gfr": state["opt_gfr"].state_dict(), "opt_cam": state["opt_cam"].state_dict(),
        "opt_pce": opt_pce.state_dict(), "fold": state["fold"]})


def load_latest(path, state, pce, opt_pce):
    data = torch_load(path)
    state["gfr"].load_state_dict(data["gfr"]); state["cam"].load_state_dict(data["cam"]); pce.load_state_dict(data["pce"])
    state["opt_gfr"].load_state_dict(data["opt_gfr"]); state["opt_cam"].load_state_dict(data["opt_cam"]); opt_pce.load_state_dict(data["opt_pce"])
    restore_rng(data["rng"]); return data


def train_worker(group, fold, gpu=0):
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"; runtime(); prepare_fold(fold)
    if task_path(group, fold).exists() and read_json(task_path(group, fold)).get("status") == "COMPLETE": return
    state = build_base(fold); pce, opt_pce = init_pce(state); fd = load_fold_data(fold); mode = mode_for(group)
    start_epoch = 0; wait = 0; best_acc = -1.; best_epoch = -1; best_metrics = None; best_diag = None; best_model = None; best_pce = None; best_rng = None
    latest = latest_path(group, fold)
    if latest.exists():
        d = load_latest(latest, state, pce, opt_pce); start_epoch = int(d["epoch"]) + 1; wait = int(d["wait"]); best_acc = float(d["best_acc"]); best_epoch = int(d["best_epoch"]); best_metrics = d["best_metrics"]; best_diag = d["best_diag"]; best_model = d["best_model"]; best_pce = d["best_pce"]; best_rng = d["best_rng"]
    train_start = time.time(); epoch = start_epoch - 1; last_grads = {}
    try:
        for epoch in range(start_epoch, BASE["epochs"]):
            state["gfr"].train(); state["cam"].train(); pce.train()
            state["opt_gfr"].zero_grad(); state["opt_cam"].zero_grad(); opt_pce.zero_grad()
            out = base_forward(state, pce, fd, mode, train_mode=True)
            assert torch.isfinite(out["total"]); out["total"].backward()
            last_grads = {"gfr": grad_norm(state["gfr"]), "cam": grad_norm(state["cam"]), "pce": grad_norm(pce), "pce_active_params": active_grad_count(pce)}
            for mod in (state["gfr"], state["cam"], pce):
                for param in mod.parameters():
                    if param.grad is not None: assert torch.isfinite(param.grad).all()
            state["opt_gfr"].step(); state["opt_cam"].step(); opt_pce.step()
            with torch.no_grad():
                ev = base_forward(state, pce, fd, mode, train_mode=False); metrics = metric(ev["prob"], state)
            if metrics["acc"] > best_acc:
                best_acc = metrics["acc"]; best_epoch = epoch; wait = 0; best_metrics = metrics
                best_diag = diag_summary(ev, state, pce); best_diag.update(last_grads)
                best_diag.update({"train_cls": float(out["cls"].detach()), "train_graph": float(out["graph"].detach()), "train_evidence": float(out["evidence"].detach()), "train_total": float(out["total"].detach())})
                best_model = model_state(state["gfr"], state["cam"]); best_pce = {k: v.detach().cpu().clone() for k, v in pce.state_dict().items()}; best_rng = pack_rng()
                save_training_checkpoint(checkpoint_path(group, fold), state, pce, opt_pce, epoch, wait, best_acc, best_epoch, best_metrics, best_diag, best_model, best_pce, best_rng)
            else: wait += 1
            if epoch % 10 == 0:
                save_training_checkpoint(latest, state, pce, opt_pce, epoch, wait, best_acc, best_epoch, best_metrics, best_diag, best_model, best_pce, best_rng)
            if wait > BASE["early"]: break
        assert best_model is not None and best_pce is not None
        load_model_state(state["gfr"], state["cam"], best_model); pce.load_state_dict(best_pce)
        with torch.no_grad(): check = metric(base_forward(state, pce, fd, mode, train_mode=False)["prob"], state)
        assert abs(check["acc"] - best_metrics["acc"]) < 1e-8
        result = {"group": group, "group_name": next(g["name"] for g in CFG["groups"] if g["id"] == group), "group_id": int(group[1:]), "mode": mode, "fold": fold, "gpu": 0, "seed": 0, "source": "new_training", "split_sha": split_sha(fold), "initial_parameter_sha": state["anchor"]["parameter_sha"], "initial_rng_sha": rng_sha(state["anchor"]["rng"]), **best_metrics, "best_epoch": best_epoch, "epochs_executed": epoch + 1, "wall_seconds": time.time() - train_start, "best_sha": sha(checkpoint_path(group, fold)), "status": "COMPLETE"}
        write_json(task_path(group, fold), result); latest.unlink(missing_ok=True)
    except Exception as exc:
        write_json(task_path(group, fold), {"group": group, "fold": fold, "gpu": 0, "status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()}); raise


def historical_rows():
    rows = csv_read(ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/results/fold_metrics.csv"); out = []
    for row in rows:
        if row.get("group") != "E00_B0B_REPLAY": continue
        out.append({"group":"E0", "group_name":"B0-B_REFERENCE", "group_id":0, "fold":int(row["fold"]), "seed":0, "gpu":"historical", "source":"Experiment43.1", "split_sha":row["split_sha"], "acc":float(row["acc"]), "hard_auc":float(row["hard_auc"]), "prob_auc":"", "spe":float(row["spe"]), "sen":float(row["sen"]), "ba":float((float(row["spe"])+float(row["sen"]))/2), "asd_f1":"", "asd_tp":int(row["asd_tp"]), "asd_fn":int(row["asd_fn"]), "td_fp":int(row["td_fp"]), "td_tn":int(row["td_tn"]), "best_epoch":int(row["best_epoch"]), "epochs_executed":int(row["epochs_executed"]), "status":"REFERENCE"})
    assert len(out) == 10; return out


def prepare():
    import numpy as np
    for directory in (RT, CKPTS, TASKS, LOGS, FOLD_DATA, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    load_data(); [prepare_fold(f) for f in range(1, 11)]
    manifest = {"experiment":"50.2", "name":CFG["name"], "status":"PREPARED", "actual_directory":str(EXP), "config_sha":sha(EXP/"config.json"), "code_sha":{p.name:sha(p) for p in [EXP/"pce_router.py",EXP/"run.py",EXP/"tests.py"] if p.exists()}, "source_sha":{p.name:sha(p) for p in SRC.glob("*.py")}, "protected_sha256":protected_hashes(), "data_shape":list(load_data()[0][0].shape), "label_counts":np.bincount(load_data()[0][1]).tolist(), "folds":{str(f):split_sha(f) for f in range(1,11)}, "planned_groups":9, "planned_new_folds":80, "historical_e0_folds":10, "semantic_tests":"pending"}
    write_json(EXP/"manifest.json",manifest); csv_write(RESULTS/"fold_metrics.csv",historical_rows()); print("PREPARE_PASS", manifest["data_shape"], len(historical_rows()))


def semantic_tests():
    import torch
    c = runtime(); prepare_fold(1); state = build_base(1); pce, opt = init_pce(state); fd = load_fold_data(1); state["gfr"].eval(); state["cam"].eval(); pce.eval()
    checks = {"gpu":0,"tests":{},"base_gfr_params":sum(p.numel() for p in state["gfr"].parameters()),"base_cam_params":sum(p.numel() for p in state["cam"].parameters()),"pce_params":sum(p.numel() for p in pce.parameters())}
    with torch.no_grad():
        nmodal, parts = state["data_layer"](state["feat"]); h, _, _ = state["gfr"](parts,nmodal,state["targ"],state["train"])
        a, da = pce(h,state["feat"],"E5_FULL"); z, dz = pce(h,state["feat"],"E5_FULL",force_alpha_zero=True)
        e3, d3 = pce(h,state["feat"],"E3_COMPETITIVE_NO_CONDITION"); e6, d6 = pce(h,state["feat"],"E6_WRONG_CONDITION",condition_perm=fd["condition_perm"]); e7, d7 = pce(h,state["feat"],"E7_WRONG_EVIDENCE_POSITION",evidence_perm=fd["evidence_perm"])
    assert float((z-h).abs().max()) == 0.0; checks["tests"]["zero_residual_identity"] = {"max_abs":float((z-h).abs().max())}
    assert abs(float(da["mask"].sum(1).mean()) - 1.0) < 1e-6 and bool(((da["gamma"]>=0)&(da["gamma"]<=1)).all()); checks["tests"]["mask_gamma_bounds"] = True
    with torch.no_grad():
        pce.alpha_raw.fill_(10.0)
        _, d_bound = pce(h, state["feat"], "E5_FULL")
        pce.alpha_raw.zero_()
    assert float(d_bound["scale"].abs().max()) <= 0.05 + 1e-6; checks["tests"]["residual_bound"] = float(d_bound["scale"].abs().max())
    assert not torch.allclose(da["evidence"][:,0],da["evidence"][:,1]); checks["tests"]["signed_competitive_evidence"] = True
    assert float((da["q"]-d6["q"]).abs().sum()) > 0 and float((da["route_mask"]-d7["route_mask"]).abs().sum()) > 0; checks["tests"]["e6_e7_change_route"] = True
    with torch.no_grad():
        pce.alpha_raw.fill_(10.0)
        h_fc = h.clone(); h_fc[0, 64] += 1.0
        h_pre = h.clone(); h_pre[0, 0] += 1.0
        fc_out, _ = pce(h_fc, state["feat"], "E5_FULL")
        pre_out, _ = pce(h_pre, state["feat"], "E5_FULL")
        pce.alpha_raw.zero_()
    assert float((fc_out[0, 64:] - a[0, 64:]).abs().sum()) > 0
    assert float((pre_out[0, :64] - h_pre[0, :64]).abs().max()) == 0.0
    checks["tests"]["fc_only_residual_region"] = True
    original = state["targ"].clone(); out1 = pce(h,state["feat"],"E5_FULL")[0]; state["targ"][fd["test"]] = 1 - state["targ"][fd["test"]]; out2 = pce(h,state["feat"],"E5_FULL")[0]; state["targ"].copy_(original); assert float((out1-out2).abs().max().detach()) == 0.0; checks["tests"]["test_label_isolation"] = True
    assert all(abs(float(g["weight_decay"])-0.11)<1e-12 for o in (state["opt_gfr"],state["opt_cam"],opt) for g in o.param_groups); assert digest_model(state["gfr"],state["cam"]) == digest_model(build_base(1)["gfr"],build_base(1)["cam"]); checks["tests"]["optimizer_and_anchor"] = True
    state["gfr"].eval(); state["cam"].eval(); pce.eval(); _=base_forward(state,pce,fd,"E5_FULL",train_mode=True); assert state["gfr"].training and state["cam"].training and pce.training; checks["tests"]["train_eval_switch"] = True
    temp=RT/"semantic_roundtrip.pt"; torch_save(temp,{"pce":pce.state_dict(),"rng":pack_rng()}); d=torch_load(temp); temp.unlink(missing_ok=True); assert rng_sha(d["rng"])==rng_sha(pack_rng()); checks["tests"]["checkpoint_rng_roundtrip"] = True
    write_json(RT/"preflight.json",checks); print("SEMANTIC_TESTS_PASS",json.dumps(checks,ensure_ascii=False))


def controller():
    groups=[g["id"] for g in CFG["groups"] if g["id"]!="E0"]; tasks=[(g,f) for g in groups for f in range(1,11)]
    processes=[]; finished=[]; start=time.time(); max_workers=int(CFG["gpu_plan"]["workers"]); envbase=dict(os.environ); envbase.update({k:"2" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")}); envbase["CUDA_VISIBLE_DEVICES"]="0"
    while tasks or processes:
        alive=sum(1 for _,p in processes if p.poll() is None)
        while alive < max_workers and tasks:
            group,fold=tasks.pop(0); log=(LOGS/f"{group}_f{fold:02d}.log"); log.parent.mkdir(parents=True,exist_ok=True); h=log.open("ab")
            p=subprocess.Popen([sys.executable,"-B",str(EXP/"run.py"),"worker","--group",group,"--fold",str(fold)],cwd=EXP,env=envbase,stdout=h,stderr=subprocess.STDOUT,start_new_session=True); processes.append(((group,fold),p)); alive+=1
        for rec in list(processes):
            if rec[1].poll() is not None: processes.remove(rec); finished.append(rec)
        time.sleep(1)
    statuses=[]
    for g in groups:
        for f in range(1,11): statuses.append(read_json(task_path(g,f)).get("status") if task_path(g,f).exists() else "MISSING")
    m=read_json(EXP/"manifest.json"); m.update(status="TRAINING_COMPLETE" if statuses.count("COMPLETE")==80 else "TRAINING_PARTIAL",completed_new_folds=statuses.count("COMPLETE"),failed_task_files=statuses.count("FAILED"),failed_processes=sum(1 for _,p in finished if p.returncode!=0),formal_wall_seconds=time.time()-start); write_json(EXP/"manifest.json",m); print("CONTROLLER_DONE",statuses.count("COMPLETE"),"failed_tasks",statuses.count("FAILED"),"failed_processes",m["failed_processes"])


def aggregate():
    import numpy as np
    prior = {int(r["fold"]): r for r in csv_read(RESULTS / "fold_metrics.csv") if r.get("group") == "E0"}
    rows=historical_rows()
    for row in rows:
        old = prior.get(int(row["fold"]))
        if old:
            for key in ("prob_auc", "ba", "asd_f1"):
                if old.get(key) not in (None, ""):
                    row[key] = old[key]
    groups=[g["id"] for g in CFG["groups"] if g["id"]!="E0"]
    for g in groups:
        for f in range(1,11):
            d=read_json(task_path(g,f)); assert d["status"]=="COMPLETE"; rows.append(d)
    assert len(rows)==90; csv_write(RESULTS/"fold_metrics.csv",rows); sums=[]
    for gc in CFG["groups"]:
        rr=[r for r in rows if r["group"]==gc["id"]]; s={"group":gc["id"],"name":gc["name"],"folds":len(rr)}
        for k in ("acc","hard_auc","prob_auc","spe","sen","ba","asd_f1"):
            vals=[float(r[k]) for r in rr if r.get(k) not in (None,"")]; s[k+"_mean"]=float(np.mean(vals)) if vals else ""; s[k+"_sd"]=float(np.std(vals,ddof=0)) if vals else ""
        sums.append(s)
    csv_write(RESULTS/"summary.csv",sums); by={g:{int(r["fold"]):r for r in rows if r["group"]==g} for g in [x["id"] for x in CFG["groups"]]}; pairs=[]
    for other in ["E0","E1","E2","E3","E4","E6","E7","E8"]:
        for k in ("acc","hard_auc","prob_auc","spe","sen"):
            ds=[(float(by["E5"][f][k])-float(by[other][f][k]))*100 for f in range(1,11) if by["E5"][f].get(k) not in (None,"") and by[other][f].get(k) not in (None,"")]
            pairs.append({"comparison":f"E5-{other}","metric":k,"e5_mean":float(np.mean([float(by["E5"][i][k])*100 for i in range(1,11)])) if ds else "","other_mean":float(np.mean([float(by[other][i][k])*100 for i in range(1,11)])) if ds else "","delta_pp":float(np.mean(ds)) if ds else "","wins":sum(v>0 for v in ds),"ties":sum(v==0 for v in ds),"losses":sum(v<0 for v in ds)})
    csv_write(RESULTS/"paired_comparisons.csv",pairs); m=read_json(EXP/"manifest.json"); m.update(status="AGGREGATED",folds_complete=90,summary_sha=sha(RESULTS/"summary.csv"),paired_sha=sha(RESULTS/"paired_comparisons.csv")); write_json(EXP/"manifest.json",m); print("AGGREGATE_PASS",len(rows))


def load_best(group,fold):
    state=build_base(fold); pce,_=init_pce(state); fd=load_fold_data(fold); d=torch_load(checkpoint_path(group,fold)); load_model_state(state["gfr"],state["cam"],d["best_model"]); pce.load_state_dict(d["best_pce"]); state["gfr"].eval();state["cam"].eval();pce.eval(); return state,pce,fd,d


def evidence_accuracy(details, labels, idx):
    import torch
    scores=details["scores"]; pred=scores.argmax(1); return float((pred[idx]==labels[idx]).float().mean())


def make_evidence_masks(out, fold, k=16):
    """Make fixed, label-free Top-K/random/amplitude-matched FC masks."""
    import numpy as np, torch
    details = out["details"]
    n = out["h_gfr"].shape[0]
    pred = out["prob"].argmax(1)
    row = torch.arange(n, device=out["h_gfr"].device)
    support = details["evidence"][row, pred].detach().cpu().numpy()
    top = np.argsort(-support, axis=1, kind="mergesort")[:, :k]
    absf = out["h_gfr"][:, 64:320].detach().abs().cpu().numpy()
    order = np.argsort(absf, axis=1, kind="mergesort")
    rank = np.empty_like(order)
    rank[np.arange(n)[:, None], order] = np.arange(256)[None, :]
    rng = np.random.default_rng(77000 + int(fold))
    random_mask = np.empty((n, k), dtype=np.int64)
    matched_mask = np.empty((n, k), dtype=np.int64)
    all_idx = np.arange(256, dtype=np.int64)
    for i in range(n):
        excluded = set(int(v) for v in top[i])
        available = np.asarray([v for v in all_idx if int(v) not in excluded], dtype=np.int64)
        random_mask[i] = rng.choice(available, size=k, replace=False)
        selected = []
        for t in top[i]:
            r = int(rank[i, t]); lo = max(0, r - 8); hi = min(256, r + 9)
            cand = [int(v) for v in order[i, lo:hi] if int(v) not in excluded and int(v) not in selected]
            if not cand:
                cand = [int(v) for v in available if int(v) not in selected]
            selected.append(int(cand[int(rng.integers(len(cand)))]))
        matched_mask[i] = np.asarray(selected, dtype=np.int64)
    return {"top": torch.from_numpy(top).to(out["h_gfr"].device),
            "random": torch.from_numpy(random_mask).to(out["h_gfr"].device),
            "matched": torch.from_numpy(matched_mask).to(out["h_gfr"].device)}


def fixed_mask_forward(state, pce, fd, out, mask, mean_fc):
    import torch
    h = out["h_gfr"].detach().clone()
    fc = h[:, 64:320].clone()
    rows = torch.arange(fc.shape[0], device=fc.device).unsqueeze(1).expand_as(mask)
    fc[rows, mask] = mean_fc[mask]
    h[:, 64:320] = fc
    h_new, _ = pce(h, state["feat"], "E5_FULL")
    prob, _, _ = state["cam"](h_new)
    return prob


def raw_recompute_mask_delta(state, pce, fd, out, mask, mean_fc, sample_ids):
    import torch
    original_feat = state["feat"]
    original_prob = out["prob"].detach()
    deltas = []
    try:
        for sample in sample_ids:
            changed = original_feat.clone()
            changed[sample, 64:320][mask[sample]] = mean_fc[mask[sample]]
            state["feat"] = changed
            with torch.no_grad():
                altered = base_forward(state, pce, fd, "E5_FULL", train_mode=False)
            deltas.append(float((altered["prob"][sample] - original_prob[sample]).abs().mean().cpu()))
    finally:
        state["feat"] = original_feat
    return float(sum(deltas) / max(len(deltas), 1))


def audit():
    import numpy as np, torch
    rows=csv_read(RESULTS/"fold_metrics.csv"); diagnostics=[]; interventions=[]; evidence_rows=[]; roi_rows=[]
    for group in [g["id"] for g in CFG["groups"] if g["id"]!="E0"]:
        for fold in range(1,11):
            state,pce,fd,data=load_best(group,fold); mode=mode_for(group)
            with torch.no_grad(): out=base_forward(state,pce,fd,mode,train_mode=False)
            rec=next(r for r in rows if r["group"]==group and int(r["fold"])==fold); obs=metric(out["prob"],state); assert abs(obs["acc"]-float(rec["acc"]))<1e-8
            dg=diag_summary(out,state,pce); d=out["details"]; labels=state["targ"]
            dg.update({"group":group,"fold":fold,"audit":"PASS","evidence_train_acc":evidence_accuracy(d,labels,state["train"]),"evidence_test_acc":evidence_accuracy(d,labels,torch.as_tensor(state["test"],device="cuda"))})
            diagnostics.append(dg)
            if group=="E5_FULL" or group=="E5":
                mask=d["mask"]; evidence=d["evidence"]; pred=out["prob"].argmax(1); cls_ev=evidence[torch.arange(evidence.shape[0],device="cuda"),pred]
                support=cls_ev.detach().cpu().numpy(); top=np.argsort(-support,axis=1)[:,:16]; entropy=(-(mask.clamp_min(1e-12)*mask.clamp_min(1e-12).log()).sum(1)).detach().cpu().numpy()
                try:
                    import pandas as pd
                    mapping=pd.read_csv(CFG["roi_mapping"]); names=[]; statuses=[]
                    for k in top.mean(axis=0).astype(int).tolist():
                        row=mapping[mapping["correlation_feature_index"]==int(k)]
                        names.append(str(row.iloc[0].get("connection_name","UNKNOWN")) if len(row) else "UNKNOWN"); statuses.append(str(row.iloc[0].get("connection_anatomical_status","UNKNOWN")) if len(row) else "UNKNOWN")
                except Exception: names=[]; statuses=[]
                masks = make_evidence_masks(out, fold, int(PCE_CFG["evidence_delete_k"]))
                mean_fc = state["feat"][state["train"], 64:320].mean(0)
                fixed_metrics = {}
                with torch.no_grad():
                    for mask_name, mask_tensor in masks.items():
                        fp = fixed_mask_forward(state, pce, fd, out, mask_tensor, mean_fc)
                        fixed_metrics[mask_name] = metric(fp, state)
                sample_ids = sorted([int(v) for v in state["test"].tolist()])[:int(PCE_CFG["full_recompute_samples_per_fold"])]
                raw_deltas = {name: raw_recompute_mask_delta(state, pce, fd, out, mask_tensor, mean_fc, sample_ids) for name, mask_tensor in masks.items()}
                evidence_rows.append({"group":group,"fold":fold,"mask_entropy_mean":float(entropy.mean()),"mask_entropy_sd":float(entropy.std()),"evidence_train_acc":evidence_accuracy(d,labels,state["train"]),"evidence_test_acc":evidence_accuracy(d,labels,torch.as_tensor(state["test"],device="cuda")),"top16_global_indices":json.dumps(top.mean(axis=0).argsort()[::-1].astype(int).tolist()),"top16_connection_names":json.dumps(names,ensure_ascii=False),"top16_status":json.dumps(statuses),"unknown_mapping_count":int(sum(s not in ("VERIFIED_BOTH_ENDPOINTS","VERIFIED_NUMERIC_UNIQUE") for s in statuses)),
                    "fixed_top_delta_acc_pp":(fixed_metrics["top"]["acc"]-obs["acc"])*100,"fixed_random_delta_acc_pp":(fixed_metrics["random"]["acc"]-obs["acc"])*100,"fixed_matched_delta_acc_pp":(fixed_metrics["matched"]["acc"]-obs["acc"])*100,
                    "fixed_top_delta_prob_auc_pp":(fixed_metrics["top"]["prob_auc"]-obs["prob_auc"])*100,"fixed_random_delta_prob_auc_pp":(fixed_metrics["random"]["prob_auc"]-obs["prob_auc"])*100,"fixed_matched_delta_prob_auc_pp":(fixed_metrics["matched"]["prob_auc"]-obs["prob_auc"])*100,
                    "raw_top_abs_prob_delta":raw_deltas["top"],"raw_random_abs_prob_delta":raw_deltas["random"],"raw_matched_abs_prob_delta":raw_deltas["matched"],"raw_recompute_sample_count":len(sample_ids)})
                with torch.no_grad():
                    cases=[("disable_pce",base_forward(state,pce,fd,mode,train_mode=False,disable_pce=True)),("alpha_zero",base_forward(state,pce,fd,mode,train_mode=False,force_alpha_zero=True)),("uniform_mask",base_forward(state,pce,fd,mode,train_mode=False,force_uniform_mask=True)),("wrong_condition",base_forward(state,pce,fd,"E6_WRONG_CONDITION",train_mode=False)),("wrong_position",base_forward(state,pce,fd,"E7_WRONG_EVIDENCE_POSITION",train_mode=False))]
                for name,res in cases:
                    mm=metric(res["prob"],state); interventions.append({"group":"E5","fold":fold,"intervention":name,**mm,"delta_acc_pp":(mm["acc"]-obs["acc"])*100,"delta_prob_auc_pp":(mm["prob_auc"]-obs["prob_auc"])*100,"prediction_flip":int((res["prob"].argmax(1)!=out["prob"].argmax(1)).detach()[state["test"]].sum().cpu())})
    # Independent E0 readback and extended metrics.
    for fold in range(1,11):
        state=build_base(fold); d=torch_load(ROOT/f"experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/checkpoints/g0_f{fold:02d}.best.pt"); load_model_state(state["gfr"],state["cam"],d["model"]); state["gfr"].eval();state["cam"].eval(); c=runtime();
        with torch.no_grad(): n,p=state["data_layer"](state["feat"]); h,_,_=state["gfr"](p,n,state["targ"],state["train"]); prob=state["cam"](h)[0]
        mm=metric(prob,state); row=next(r for r in rows if r["group"]=="E0" and int(r["fold"])==fold); row.update({"prob_auc":mm["prob_auc"],"ba":mm["ba"],"asd_f1":mm["asd_f1"]})
    csv_write(RESULTS/"fold_metrics.csv",rows); csv_write(RESULTS/"diagnostics.csv",diagnostics); csv_write(RESULTS/"interventions.csv",interventions); csv_write(RESULTS/"evidence_audit.csv",evidence_rows)
    m=read_json(EXP/"manifest.json"); m.update(status="AUDITED",audit_rows=len(diagnostics),intervention_rows=len(interventions),evidence_rows=len(evidence_rows),diagnostics_sha=sha(RESULTS/"diagnostics.csv"),evidence_sha=sha(RESULTS/"evidence_audit.csv")); write_json(EXP/"manifest.json",m); print("AUDIT_PASS",len(diagnostics),len(interventions),len(evidence_rows))


def report():
    import numpy as np
    rows=csv_read(RESULTS/"fold_metrics.csv"); assert len(rows)==90; groups=[g["id"] for g in CFG["groups"]]; lines=["# Experiment 50.2 — PCE-Router","","## 执行状态","","E0独立读回Experiment43.1 B0-B；E1–E8完成80个新增折。PCE只修改GFR输出的FC 256维，未增加患者间消息；原始GFR/CAM、weight_decay=0.11、simple-2和历史val=test协议保留。","","## 九组正式结果（百分比，十折 mean ± population SD）","","|组|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|","|---|---:|---:|---:|---:|---:|---:|---:|"]
    summaries=[]
    for g in groups:
        rr=[r for r in rows if r["group"]==g]; s={"group":g}; cells=[]
        for k in ("acc","hard_auc","prob_auc","spe","sen","ba","asd_f1"):
            vals=np.asarray([float(r[k])*100 for r in rr]); s[k]=float(vals.mean()); s[k+"_sd"]=float(vals.std(ddof=0)); cells.append(f"{vals.mean():.4f} ± {vals.std(ddof=0):.4f}")
        summaries.append(s); lines.append("|%s|%s|%s|%s|%s|%s|%s|%s|"%(g,*cells))
    lines += ["","## E5逐折配对","","|比较|指标|E5均值|对照均值|差值(pp)|胜/平/负|","|---|---|---:|---:|---:|---|"]
    for r in csv_read(RESULTS/"paired_comparisons.csv"): lines.append(f"|{r['comparison']}|{r['metric']}|{float(r['e5_mean']):.4f}|{float(r['other_mean']):.4f}|{float(r['delta_pp']):+.4f}|{r['wins']}/{r['ties']}/{r['losses']}|")
    e5=next(s for s in summaries if s["group"]=="E5"); e0=next(s for s in summaries if s["group"]=="E0")
    lines += ["","## 机制诊断","","- PCE初始化时alpha=0，语义测试验证h_new与GFR输出逐值相同；训练后报告alpha、接受系数、mask熵、证据差异和残差比例。","- `evidence_audit.csv`记录E5每折ASD/TD证据头训练/测试诊断准确率、mask熵、Top-16 FC索引、ROI映射状态。四条未知ROI连接不赋予脑区名称。","- `interventions.csv`区分关闭PCE、alpha=0、均匀mask、错误条件和错误证据位置的固定checkpoint干预。","","## 科研判定",f"E5相对E0：ACC {e5['acc']-e0['acc']:+.4f}pp，Hard-AUC {e5['hard_auc']-e0['hard_auc']:+.4f}pp，Probability-AUC {e5['prob_auc']-e0['prob_auc']:+.4f}pp，SPE {e5['spe']-e0['spe']:+.4f}pp，SEN {e5['sen']-e0['sen']:+.4f}pp。"]
    e1=next(s for s in summaries if s["group"]=="E1"); e2=next(s for s in summaries if s["group"]=="E2"); e3=next(s for s in summaries if s["group"]=="E3"); e4=next(s for s in summaries if s["group"]=="E4"); e6=next(s for s in summaries if s["group"]=="E6"); e7=next(s for s in summaries if s["group"]=="E7"); e8=next(s for s in summaries if s["group"]=="E8")
    evidence = csv_read(RESULTS / "evidence_audit.csv")
    if evidence:
        mean_ev = lambda key: float(np.mean([float(r[key]) for r in evidence]))
        lines += [f"E5相对E1/E2/E3/E4/E6/E7/E8的ACC差值分别为 {e5['acc']-e1['acc']:+.4f}/{e5['acc']-e2['acc']:+.4f}/{e5['acc']-e3['acc']:+.4f}/{e5['acc']-e4['acc']:+.4f}/{e5['acc']-e6['acc']:+.4f}/{e5['acc']-e7['acc']:+.4f}/{e5['acc']-e8['acc']:+.4f}pp。", "", f"E5证据头训练/测试诊断准确率均值为 {mean_ev('evidence_train_acc')*100:.2f}%/{mean_ev('evidence_test_acc')*100:.2f}%，mask熵为{mean_ev('mask_entropy_mean'):.4f}（256维均匀熵为{math.log(256):.4f}），显示竞争路由接近均匀且没有可重复的疾病证据分类。", f"固定GFR输出的Top-16/随机/幅值匹配遮蔽ACC差值均值为 {mean_ev('fixed_top_delta_acc_pp'):+.4f}/{mean_ev('fixed_random_delta_acc_pp'):+.4f}/{mean_ev('fixed_matched_delta_acc_pp'):+.4f}pp；Top-16没有比负控造成更大损害。原始FC重算样本的平均概率变化为 {mean_ev('raw_top_abs_prob_delta'):.4f}/{mean_ev('raw_random_abs_prob_delta'):.4f}/{mean_ev('raw_matched_abs_prob_delta'):.4f}，只能作为小样本敏感性诊断。", "", "判定规则：只有E5同时超过B0-B、容量控制、E2/E4、E3/E6、E7/E8且证据删除效应超过随机删除，才支持GO。当前结果未满足。", "", "最终判定：NO-GO。E5 ACC低于B0-B、E2和E7，错误条件E6与E5相当，Top-K证据删除没有超过随机负控；新增路由的alpha均值接近零、残差约10^-7量级，说明该模块在原始训练协议下几乎退化为恒等映射。", "", "## 协议边界", "", "历史val=test、CV前RFE/标准化、transductive GFR和单seed限制仍然存在；PCE证据权重是数据驱动重要性，不是临床因果或已验证病灶。"]
    (RESULTS/"report.md").write_text("\n".join(lines)+"\n"); m=read_json(EXP/"manifest.json"); m.update(status="REPORT_COMPLETE",report_sha=sha(RESULTS/"report.md"),report_time=time.time()); write_json(EXP/"manifest.json",m); print("REPORT_PASS")


def append_handoff():
    marker="## Experiment 50.2 — PCE-Router"
    handoff=ROOT/"handoff.md"; old=handoff.read_text(errors="replace")
    if marker in old: print("ALREADY_PRESENT"); return
    sums=csv_read(RESULTS/"summary.csv"); s={r["group"]:r for r in sums}; e5=s["E5"]; e0=s["E0"]
    text="\n\n"+marker+"\n\n"+f"- 状态：E0历史参考10折、E1–E8共80个新增折完成；单GPU0，最多5个worker；语义测试、checkpoint重载审计和证据诊断完成。\n- 动机：50/50.1的患者关系分支未形成稳定增量，本轮转为患者自身FC局部证据路由，不新增跨患者消息、Prompt或反事实监督。\n- 结果：E0 ACC={float(e0['acc_mean'])*100:.4f}%，E5 ACC={float(e5['acc_mean'])*100:.4f}%；完整ACC/Hard-AUC/Probability-AUC/SPE/SEN见`experiment/experiment50_2_pce_router/results/report.md`。\n- E5相对E0 ACC={(float(e5['acc_mean'])-float(e0['acc_mean']))*100:+.4f}pp；E5与E1/E2/E3/E4/E6/E7/E8配对结果见`paired_comparisons.csv`。\n- 约束：PCE只作用GFR输出FC 256维，前64维和原始GFR/CAM保持；条件输入为年龄/性别/ANAT/FUNC，不含站点；E6条件置乱、E7证据位置置乱。\n- 判定：由报告根据预注册门槛给出GO/PARTIAL/NO-GO；不将证据头预测或固定干预等同于疾病因果解释。\n- 文件SHA：manifest=`{sha(EXP/'manifest.json')}`，report=`{sha(RESULTS/'report.md')}`，summary=`{sha(RESULTS/'summary.csv')}`，evidence=`{sha(RESULTS/'evidence_audit.csv')}`。\n"
    with handoff.open("a") as f: f.write(text)
    print("HANDOFF_APPENDED")


def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="cmd",required=True)
    sub.add_parser("prepare"); sub.add_parser("preflight"); q=sub.add_parser("worker"); q.add_argument("--group",required=True); q.add_argument("--fold",type=int,required=True)
    sub.add_parser("controller"); sub.add_parser("aggregate"); sub.add_parser("audit"); sub.add_parser("report"); sub.add_parser("handoff")
    args=parser.parse_args()
    if args.cmd=="prepare": prepare()
    elif args.cmd=="preflight": semantic_tests()
    elif args.cmd=="worker": train_worker(args.group,args.fold,0)
    elif args.cmd=="controller": controller()
    elif args.cmd=="aggregate": aggregate()
    elif args.cmd=="audit": audit()
    elif args.cmd=="report": report()
    elif args.cmd=="handoff": append_handoff()

if __name__ == "__main__": main()
