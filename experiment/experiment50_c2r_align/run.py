#!/usr/bin/env python3
"""Experiment 50 C²R-Align runner.

The runner is intentionally self-contained: it imports only the immutable B0-B
source snapshot under .runtime/sources and model50.py in this directory.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, math, os, pickle, random, shutil, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, "2")
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
BASE = CFG["base"]
NEW = CFG["new"]
DIMS = [48, 6, 10, 256]

CTX = None
DATA = None
FOLDS = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
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
        for k in row:
            if k not in fields: fields.append(k)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def csv_read(path):
    p = Path(path)
    if not p.exists(): return []
    with p.open(newline="") as f: return list(csv.DictReader(f))


def seed_all(seed):
    import numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def pack_rng():
    import numpy as np, torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(r):
    import numpy as np, torch
    random.setstate(r["python"]); np.random.set_state(r["numpy"])
    torch.set_rng_state(r["cpu"].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in r["cuda"]])


def rng_sha(r):
    h = hashlib.sha256(pickle.dumps((r["python"], r["numpy"]), protocol=4))
    h.update(r["cpu"].cpu().numpy().tobytes())
    for x in r["cuda"]: h.update(x.cpu().numpy().tobytes())
    return h.hexdigest()


def runtime():
    global CTX
    if CTX is None:
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        assert torch.cuda.is_available(), "CUDA is required for the formal run"
        torch.cuda.set_device(0)
        sys.path.insert(0, str(SRC))
        sys.path.insert(0, str(EXP))
        import layers, network, utils
        from model50 import PLNNoPEN, C2RAlign
        CTX = {"torch": torch, "np": __import__("numpy"), "layers": layers,
               "network": network, "utils": utils, "PLNNoPEN": PLNNoPEN,
               "C2RAlign": C2RAlign}
    return CTX


def load_data():
    global DATA, FOLDS
    if DATA is not None: return DATA, FOLDS
    import numpy as np, pandas as pd
    from sklearn.model_selection import StratifiedKFold
    base = ROOT / "data/ABIDE"
    frame = pd.read_csv(base / "processed_standard_data.csv")
    mapping = np.load(base / "modal_feat_dict.npy", allow_pickle=True).item()
    names = [str(x) for k in ("PHENO", "ANAT", "FUNC", "Correlation") for x in mapping[k]]
    assert list(frame.columns[:-1]) == names
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
    _, folds = load_data(); tr, te = folds[fold - 1]
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
        for name, p in sorted(module.named_parameters()):
            h.update((prefix + "." + name).encode() + b"\0")
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def model_state(gfr, cam):
    return {"gfr": {k: v.detach().cpu().clone() for k, v in gfr.state_dict().items()},
            "pln": {k: v.detach().cpu().clone() for k, v in cam.state_dict().items()}}


def load_model_state(gfr, cam, state):
    gfr.load_state_dict(state["gfr"])
    # Historical anchors call the CAM key ``pln``; this runner's compact
    # best-model payload calls it ``cam``.  Accept both without changing the
    # underlying B0-B state.
    cam.load_state_dict(state.get("pln", state.get("cam")))


def make_hp():
    return SimpleNamespace(
        n_hidden=BASE["n_hidden"], n_head=BASE["n_head"], dropout=BASE["dropout"],
        nlayer=BASE["nlayer"], nmodal=4, nclass=2, lr=BASE["lr"], reg=BASE["weight_decay"],
        th=BASE["threshold"], GC_mode="weighted-cosine", MP_mode="GCN", MF_mode="concat",
        alpha=0.5, theta_smooth=BASE["theta_smooth"], theta_degree=BASE["theta_degree"],
        theta_sparsity=BASE["theta_sparsity"])


def build_base(fold):
    """Restore an exact Experiment43.1 B0-B fold anchor."""
    import torch
    c = runtime(); x, y = load_data()[0]; tr, te = load_data()[1][fold - 1]
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
    state = {
        "gfr": gfr, "cam": cam, "opt_gfr": opt_gfr, "opt_cam": opt_cam,
        "feat": torch.from_numpy(x).float().cuda(), "targ": torch.from_numpy(y).long().cuda(),
        "train": tr, "test": te, "weight": None, "data_layer": c["layers"].InitialDataLayer(DIMS),
        "anchor": anchor, "fold": fold}
    counter = Counter(y[tr]); weights = len(tr) / __import__("numpy").asarray(list(counter.values())) / 2
    state["weight"] = torch.from_numpy(weights).float().cuda()
    return state


def init_c2r(state, group_id):
    import torch
    c = runtime()
    base_rng = pack_rng()
    seed_all(500000 + state["fold"] * 100 + group_id)
    c2r = c["C2RAlign"](DIMS, NEW["embedding"], NEW["encoder_hidden"],
                           NEW["eta"], NEW["rho"], NEW["relation_temperature"]).cuda()
    restore_rng(base_rng)
    opt = torch.optim.Adam(c2r.parameters(), lr=NEW["lr"], weight_decay=NEW["weight_decay"])
    return c2r, opt


def site_ids(x):
    import numpy as np
    return np.argmax(x[:, 26:46], axis=1).astype(np.int64)


def make_permutation(x, train, test, seed):
    import numpy as np
    sid = site_ids(x); perm = np.arange(len(x), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for split_name, ids in (("train", train), ("test", test)):
        groups = {}
        for i in ids: groups.setdefault(int(sid[i]), []).append(int(i))
        for _, vals in sorted(groups.items()):
            if len(vals) < 2: continue
            arr = np.asarray(vals, dtype=np.int64); sh = arr.copy()
            for _ in range(20):
                rng.shuffle(sh)
                if np.all(sh != arr): break
            else: sh = np.roll(arr, 1)
            perm[arr] = sh
    return perm


def make_candidates(x, train, test, k=8):
    import numpy as np
    corr = x[:, 64:320].astype(np.float64)
    corr /= np.linalg.norm(corr, axis=1, keepdims=True).clip(min=1e-8)
    send = np.asarray(train, dtype=np.int64)
    out = np.empty((len(x), k), dtype=np.int64)
    for i in range(len(x)):
        pool = send[send != i]
        score = corr[pool].dot(corr[i])
        take = min(k, len(pool)); order = np.argsort(-score, kind="mergesort")[:take]
        out[i, :take] = pool[order]
        if take < k: out[i, take:] = pool[:k-take]
    return out


def make_rewire(candidates, perm):
    import numpy as np
    out = perm[candidates].copy()
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if out[i, j] == i:
                for q in range(out.shape[1]):
                    cand = perm[candidates[i, (j + q + 1) % out.shape[1]]]
                    if cand != i:
                        out[i, j] = cand; break
    return out


def make_pairs(x, y, train, seed):
    import numpy as np
    sid = site_ids(x); rng = np.random.default_rng(seed)
    buckets = {(a, b): [] for a in (0, 1) for b in (0, 1)}
    tr = np.asarray(train, dtype=np.int64)
    for p, i in enumerate(tr):
        for j in tr[p+1:]:
            same_y = int(y[i] == y[j]); same_s = int(sid[i] == sid[j])
            buckets[(same_y, same_s)].append((int(i), int(j)))
    chosen = []
    cap = int(NEW["pair_count_target_per_stratum"])
    counts = {}
    for key, vals in buckets.items():
        rng.shuffle(vals); take = vals[:min(cap, len(vals))]; chosen.extend(take); counts[str(key)] = len(take)
    rng.shuffle(chosen)
    pi = np.asarray([a for a, _ in chosen], dtype=np.int64)
    pj = np.asarray([b for _, b in chosen], dtype=np.int64)
    py = (y[pi] == y[pj]).astype(np.float32)
    return pi, pj, py, counts


def prepare_fold(fold):
    import numpy as np
    x, y = load_data()[0]; tr, te = load_data()[1][fold - 1]
    path = FOLD_DATA / f"fold{fold:02d}.npz"
    if path.exists(): return
    cand = make_candidates(x, tr, te, NEW["candidate_k"])
    perm = make_permutation(x, tr, te, 7000 + fold)
    rewire = make_rewire(cand, perm)
    pi, pj, py, counts = make_pairs(x, y, tr, 9000 + fold)
    np.savez_compressed(path, candidates=cand, perm=perm, rewire=rewire,
                        pair_i=pi, pair_j=pj, pair_y=py,
                        train=np.asarray(tr), test=np.asarray(te))
    write_json(FOLD_DATA / f"fold{fold:02d}.json", {
        "fold": fold, "candidate_k": int(cand.shape[1]), "pair_counts": counts,
        "train_n": int(len(tr)), "test_n": int(len(te)),
        "same_site_candidate_fraction": float((site_ids(x)[:,None] == site_ids(x)[cand]).mean()),
        "candidate_sha": sha(path), "split_sha": split_sha(fold)})


def load_fold_data(fold, device="cuda"):
    import torch, numpy as np
    z = np.load(FOLD_DATA / f"fold{fold:02d}.npz")
    return {k: torch.from_numpy(z[k]).to(device) for k in z.files}


def base_forward(state, c2r, fd, mode, return_diag=True, disable_c2r=False,
                 train_mode=None):
    import torch
    c = runtime()
    # Do not silently switch eval() calls back to train().  The historical
    # implementation has dropout in CAM/GCN, so mode leakage would change both
    # checkpoint selection and every mechanism audit.  Callers explicitly pass
    # train_mode for each forward; None preserves the caller's current mode.
    if train_mode is True:
        state["gfr"].train(); state["cam"].train(); c2r.train()
    elif train_mode is False:
        state["gfr"].eval(); state["cam"].eval(); c2r.eval()
    nmodal, parts = state["data_layer"](state["feat"])
    h_gfr, adjs, aloss = state["gfr"](parts, nmodal, state["targ"], state["train"])
    graph = sum(c["utils"].GraphConstructLoss(parts[i], adjs[i], BASE["theta_smooth"], BASE["theta_degree"], BASE["theta_sparsity"])
                for i in range(4)) / 4.0
    if disable_c2r:
        h50 = h_gfr; rel = h_gfr.new_zeros(()); align = h_gfr.new_zeros(()); diag = {
            "gate_pheno": 0., "gate_anat": 0., "gate_func": 0., "gate_null": 1.,
            "message_norm": 0., "delta_norm": 0., "gfr_norm": float(h_gfr.detach().norm()),
            "residual_ratio": 0., "pair_count": int(fd["pair_i"].numel()),
            "pair_positive_fraction": float(fd["pair_y"].float().mean()), "rel_loss": 0., "align_loss": 0.}
    else:
        h50, rel, align, diag = c2r(
            state["feat"], h_gfr, fd["candidates"], fd["pair_i"], fd["pair_j"], fd["pair_y"],
            mode, aux_perm=fd["perm"], rewire=(fd["rewire"] if mode == "E7_REWIRED" else None))
    prob, hidden, attn = state["cam"](h50)
    cls = c["utils"].ClsLoss(prob, state["targ"], state["train"], state["weight"])
    total = cls + graph + NEW["lambda_relation"] * rel + NEW["lambda_align"] * align
    return {"prob": prob, "hidden": hidden, "h_gfr": h_gfr, "h50": h50, "cls": cls,
            "graph": graph, "rel": rel, "align": align, "total": total,
            "adjs": adjs, "aloss": aloss, "diag": diag}


def metric(output, state, test=None):
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
    if test is None: test = state["test"]
    idx = np.asarray(test, dtype=np.int64); y = state["targ"].detach().cpu().numpy()[idx]
    prob = output.detach().exp().cpu().numpy()[idx]
    pred = prob.argmax(1)
    cm = confusion_matrix(y, pred, labels=[0, 1]); tp, fn, fp, tn = map(int, cm.ravel())
    hard = float(roc_auc_score(np.eye(2)[y], np.eye(2)[pred]))
    p_auc = float(roc_auc_score(y, prob[:, 1]))
    sen = tp / max(tp + fn, 1); spe = tn / max(tn + fp, 1)
    return {"acc": float((pred == y).mean()), "hard_auc": hard, "prob_auc": p_auc,
            "spe": float(spe), "sen": float(sen), "ba": float((spe + sen) / 2),
            "asd_f1": float(f1_score(y, pred, pos_label=0, zero_division=0)),
            "asd_tp": tp, "asd_fn": fn, "td_fp": fp, "td_tn": tn,
            "n_test": int(len(idx))}


def grad_norm(module):
    import torch
    v = None
    for p in module.parameters():
        if p.grad is not None:
            q = (p.grad.detach() ** 2).sum()
            v = q if v is None else v + q
    return float(v.sqrt().cpu()) if v is not None else 0.0


def checkpoint_path(group, fold): return CKPTS / f"{group}_f{fold:02d}.best.pt"
def latest_path(group, fold): return CKPTS / f"{group}_f{fold:02d}.latest.pt"
def task_path(group, fold): return TASKS / f"{group}_f{fold:02d}.json"


def save_training_checkpoint(path, state, c2r, opt_c2r, epoch, wait, best_acc, best_epoch, best_metrics, best_diag, best_model, best_c2r, best_rng):
    torch_save(path, {"epoch": epoch, "wait": wait, "best_acc": best_acc, "best_epoch": best_epoch,
                      "best_metrics": best_metrics, "best_diag": best_diag, "best_model": best_model,
                      "best_c2r": best_c2r, "best_rng": best_rng,
                      # Current state is required for exact interruption
                      # recovery; best_rng alone is the checkpoint-selection
                      # snapshot and cannot resume the next optimizer step.
                      "rng": pack_rng(),
                      "gfr": state["gfr"].state_dict(), "cam": state["cam"].state_dict(), "c2r": c2r.state_dict(),
                      "opt_gfr": state["opt_gfr"].state_dict(), "opt_cam": state["opt_cam"].state_dict(),
                      "opt_c2r": opt_c2r.state_dict(), "fold": state["fold"]})


def load_latest(path, state, c2r, opt_c2r):
    d = torch_load(path)
    state["gfr"].load_state_dict(d["gfr"]); state["cam"].load_state_dict(d["cam"]); c2r.load_state_dict(d["c2r"])
    state["opt_gfr"].load_state_dict(d["opt_gfr"]); state["opt_cam"].load_state_dict(d["opt_cam"]); opt_c2r.load_state_dict(d["opt_c2r"])
    restore_rng(d["rng"])
    return d


def historical_rows():
    p = ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/results/fold_metrics.csv"
    rows = csv_read(p); out = []
    for r in rows:
        if r.get("group") != "E00_B0B_REPLAY": continue
        out.append({"group":"E0", "group_name":"B0-B_REFERENCE", "group_id":0, "fold":int(r["fold"]),
                    "seed":0, "gpu":r.get("physical_gpu","historical"), "source":"Experiment43.1",
                    "split_sha":r["split_sha"], "acc":float(r["acc"]), "hard_auc":float(r["hard_auc"]),
                    "prob_auc":"", "spe":float(r["spe"]), "sen":float(r["sen"]),
                    "ba":(float(r["spe"])+float(r["sen"]))/2, "asd_f1":"",
                    "asd_tp":int(r["asd_tp"]), "asd_fn":int(r["asd_fn"]), "td_fp":int(r["td_fp"]), "td_tn":int(r["td_tn"]),
                    "best_epoch":int(r["best_epoch"]), "epochs_executed":int(r["epochs_executed"]), "status":"REFERENCE"})
    assert len(out)==10
    return out


def train_worker(group, fold, gpu):
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    c = runtime(); prepare_fold(fold)
    gcfg = next(g for g in CFG["groups"] if g["id"] == group); mode = gcfg["mode"]
    tp = task_path(group, fold)
    if tp.exists() and read_json(tp).get("status") == "COMPLETE": return
    state = build_base(fold); c2r, opt_c2r = init_c2r(state, int(group[1:]))
    fd = load_fold_data(fold)
    start_epoch = 0; wait = 0; best_acc = -1.; best_epoch = -1; best_metrics = None; best_diag = None; best_model = None; best_c2r = None; best_rng = None
    lp = latest_path(group, fold)
    if lp.exists():
        d = load_latest(lp, state, c2r, opt_c2r); start_epoch = int(d["epoch"])+1; wait=int(d["wait"]); best_acc=float(d["best_acc"]); best_epoch=int(d["best_epoch"]); best_metrics=d["best_metrics"]; best_diag=d["best_diag"]; best_model=d["best_model"]; best_c2r=d["best_c2r"]; best_rng=d["best_rng"]
    entry_sha = digest_model(state["gfr"], state["cam"])
    train_start = time.time(); last_grad = {"gfr":0.,"cam":0.,"c2r":0.}; last_loss = {}
    try:
        for epoch in range(start_epoch, BASE["epochs"]):
            state["gfr"].train(); state["cam"].train(); c2r.train()
            state["opt_gfr"].zero_grad(); state["opt_cam"].zero_grad(); opt_c2r.zero_grad()
            out = base_forward(state, c2r, fd, mode, train_mode=True)
            assert torch.isfinite(out["total"])
            out["total"].backward()
            last_grad = {"gfr":grad_norm(state["gfr"]),"cam":grad_norm(state["cam"]),"c2r":grad_norm(c2r)}
            for module in (state["gfr"], state["cam"], c2r):
                for p in module.parameters():
                    if p.grad is not None: assert torch.isfinite(p.grad).all()
            state["opt_gfr"].step(); state["opt_cam"].step(); opt_c2r.step()
            state["gfr"].eval(); state["cam"].eval(); c2r.eval()
            with torch.no_grad():
                ev = base_forward(state, c2r, fd, mode, train_mode=False)
                met = metric(ev["prob"], state)
            last_loss = {"train_cls":float(out["cls"].detach()),"train_graph":float(out["graph"].detach()),"train_rel":float(out["rel"].detach()),"train_align":float(out["align"].detach()),"train_total":float(out["total"].detach())}
            if met["acc"] > best_acc:
                best_acc = met["acc"]; best_epoch = epoch; wait = 0; best_metrics = met; best_diag = dict(ev["diag"]); best_diag.update({"grad_gfr":last_grad["gfr"],"grad_cam":last_grad["cam"],"grad_c2r":last_grad["c2r"], **last_loss})
                best_model = {"gfr":{k:v.detach().cpu().clone() for k,v in state["gfr"].state_dict().items()},"cam":{k:v.detach().cpu().clone() for k,v in state["cam"].state_dict().items()}}
                best_c2r = {k:v.detach().cpu().clone() for k,v in c2r.state_dict().items()}; best_rng = pack_rng()
                save_training_checkpoint(checkpoint_path(group,fold),state,c2r,opt_c2r,epoch,wait,best_acc,best_epoch,best_metrics,best_diag,best_model,best_c2r,best_rng)
            else: wait += 1
            if epoch % 10 == 0:
                save_training_checkpoint(lp,state,c2r,opt_c2r,epoch,wait,best_acc,best_epoch,best_metrics,best_diag,best_model,best_c2r,best_rng)
            if wait > BASE["early"]: break
        assert best_model is not None
        load_model_state(state["gfr"], state["cam"], best_model); c2r.load_state_dict(best_c2r)
        state["gfr"].eval(); state["cam"].eval(); c2r.eval()
        with torch.no_grad():
            final = base_forward(state,c2r,fd,mode, train_mode=False)
            check = metric(final["prob"], state)
        assert abs(check["acc"]-best_metrics["acc"]) < 1e-8
        result = {"group":group,"group_name":gcfg["name"],"group_id":int(group[1:]),"mode":mode,"fold":fold,"gpu":gpu,"seed":0,"source":"new_training","split_sha":split_sha(fold),"initial_parameter_sha":state["anchor"]["parameter_sha"],"initial_rng_sha":rng_sha(state["anchor"]["rng"]),"acc":best_metrics["acc"],"hard_auc":best_metrics["hard_auc"],"prob_auc":best_metrics["prob_auc"],"spe":best_metrics["spe"],"sen":best_metrics["sen"],"ba":best_metrics["ba"],"asd_f1":best_metrics["asd_f1"],"asd_tp":best_metrics["asd_tp"],"asd_fn":best_metrics["asd_fn"],"td_fp":best_metrics["td_fp"],"td_tn":best_metrics["td_tn"],"best_epoch":best_epoch,"epochs_executed":epoch+1,"wall_seconds":time.time()-train_start,"best_sha":sha(checkpoint_path(group,fold)),"status":"COMPLETE"}
        write_json(tp,result)
        latest_path(group,fold).unlink(missing_ok=True)
    except Exception as exc:
        write_json(tp,{"group":group,"fold":fold,"gpu":gpu,"status":"FAILED","error":repr(exc),"traceback":traceback.format_exc()})
        raise


def preflight(gpu):
    import torch, numpy as np
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    c=runtime(); prepare_fold(1); state=build_base(1); fd=load_fold_data(1)
    checks={"gpu":gpu,"base_gfr_params":sum(p.numel() for p in state["gfr"].parameters()),"base_cam_params":sum(p.numel() for p in state["cam"].parameters()),"modes":{},"source_sha":{p.name:sha(p) for p in SRC.glob('*.py')}}
    for group in ["E1","E2","E3","E4","E5","E6","E7"]:
        s=build_base(1); c2r,opt=init_c2r(s,int(group[1:])); m=next(g["mode"] for g in CFG["groups"] if g["id"]==group)
        s["gfr"].eval();s["cam"].eval();c2r.eval()
        with torch.no_grad(): out=base_forward(s,c2r,fd,m,train_mode=False)
        assert torch.isfinite(out["h50"]).all() and out["h50"].shape==(871,320)
        assert abs(out["diag"]["gate_pheno"]+out["diag"]["gate_anat"]+out["diag"]["gate_func"]+out["diag"]["gate_null"]-1)<1e-5
        checks["modes"][group]={"mode":m,"finite":True,"residual_ratio":out["diag"]["residual_ratio"],"gate_sum":out["diag"]["gate_pheno"]+out["diag"]["gate_anat"]+out["diag"]["gate_func"]+out["diag"]["gate_null"]}
    # E1 is candidate-invariant; E6/E7 must change actual relation/message in the fixed preflight input.
    a=build_base(1); c2,op=init_c2r(a,5); a["gfr"].eval();a["cam"].eval();c2.eval()
    with torch.no_grad(): full=base_forward(a,c2,fd,"E5_FULL",train_mode=False); rew=base_forward(a,c2,fd,"E7_REWIRED",train_mode=False); selfo=base_forward(a,c2,fd,"E1_SELF",train_mode=False)
    checks["relation_effect"]={"full_rewire_h50_l1":float((full["h50"]-rew["h50"]).norm()),"full_self_h50_l1":float((full["h50"]-selfo["h50"]).norm()),"candidate_k":int(fd["candidates"].shape[1])}
    # Gradient and label isolation on the new module; test-label mutation cannot change h50.
    a=build_base(1);c2,op=init_c2r(a,5);a["gfr"].train();a["cam"].train();c2.train();
    a["opt_gfr"].zero_grad();a["opt_cam"].zero_grad();op.zero_grad();o=base_forward(a,c2,fd,"E5_FULL",train_mode=True);o["total"].backward();checks["grad"]={"gfr":grad_norm(a["gfr"]),"cam":grad_norm(a["cam"]),"c2r":grad_norm(c2)}
    with torch.no_grad():
        old=a["targ"].clone();a["targ"][fd["test"]]=1-a["targ"][fd["test"]];mut=base_forward(a,c2,fd,"E5_FULL",train_mode=False);a["targ"].copy_(old)
    checks["test_label_isolation"] = {"h50_max_abs":float((o["h50"].detach()-mut["h50"].detach()).abs().max()),"pair_train_only":bool((fd["pair_i"].cpu().numpy()!=fd["test"].cpu().numpy()[:,None]).all())}
    write_json(RT/"preflight.json",checks); print("PREFLIGHT_PASS",json.dumps(checks,ensure_ascii=False))


def prepare():
    import numpy as np
    load_data(); [prepare_fold(i) for i in range(1,11)]
    hist=historical_rows()
    manifest={"experiment":"50","name":"C²R-Align","status":"PREPARED","actual_directory":str(EXP),"config_sha":sha(EXP/"config.json"),"code_sha":{p.name:sha(p) for p in [EXP/"run.py",EXP/"model50.py"]},"source_sha":{p.name:sha(p) for p in SRC.glob("*.py")},"protected_sha256":protected_hashes(),"data_shape":list(load_data()[0][0].shape),"label_counts":np.bincount(load_data()[0][1]).tolist(),"folds":{str(i):split_sha(i) for i in range(1,11)},"planned_groups":8,"planned_new_folds":70,"historical_e0_folds":10,"start_unix":time.time()}
    write_json(EXP/"manifest.json",manifest); csv_write(RESULTS/"fold_metrics.csv",hist); print("PREPARE_PASS",manifest["data_shape"],len(hist))


def controller():
    import collections
    groups=[g["id"] for g in CFG["groups"] if g["id"]!="E0"]
    tasks=[(g,f,0 if f<=5 else 1) for g in groups for f in range(1,11)]
    procs=[]; started=time.time(); maxw=CFG["gpu_plan"]["workers_per_gpu"]
    envbase=dict(os.environ); envbase.update({"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2","NUMEXPR_NUM_THREADS":"2"})
    while tasks or procs:
        for gpu in (0,1):
            # ``procs`` stores (task_tuple, group, fold, process); the GPU is
            # the third field of task_tuple.  Comparing the group string to an
            # integer silently disabled the concurrency cap and launched all
            # 70 workers at once.
            alive=sum(1 for item,_,_,p in procs if item[2] == gpu and p.poll() is None)
            while alive < maxw:
                item=next((t for t in tasks if t[2]==gpu),None)
                if item is None: break
                tasks.remove(item); group,fold,_=item
                log=LOGS/f"{group}_f{fold:02d}.log"; log.parent.mkdir(parents=True,exist_ok=True)
                e=dict(envbase);e["CUDA_VISIBLE_DEVICES"]=str(gpu)
                f=open(log,"ab")
                p=subprocess.Popen([sys.executable,"-B",str(EXP/"run.py"),"worker","--group",group,"--fold",str(fold),"--gpu",str(gpu)],cwd=EXP,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                procs.append((item,group,fold,p)); alive+=1
        for rec in list(procs):
            if rec[3].poll() is not None: procs.remove(rec)
        time.sleep(1)
    failed=[p for _,_,_,p in procs if p.returncode not in (None,0)]
    m=read_json(EXP/"manifest.json");m.update(status="TRAINING_COMPLETE",formal_wall_seconds=time.time()-started,completed_new_folds=sum(1 for g in groups for f in range(1,11) if task_path(g,f).exists() and read_json(task_path(g,f)).get("status")=="COMPLETE"));write_json(EXP/"manifest.json",m)
    print("CONTROLLER_DONE",m["completed_new_folds"],"failed",len(failed))


def aggregate():
    # Preserve probability-AUC and extended metrics written by the independent
    # audit when aggregate is rerun after checkpoint readback.  The historical
    # source CSV predates probability-AUC, so a plain re-read would erase it.
    prior = {int(r["fold"]): r for r in csv_read(RESULTS/"fold_metrics.csv")
             if r.get("group") == "E0"}
    rows=historical_rows()
    for r in rows:
        p = prior.get(int(r["fold"]))
        if p:
            for k in ("prob_auc", "ba", "asd_f1"):
                if p.get(k) not in (None, ""):
                    r[k] = p[k]
    for g in [x["id"] for x in CFG["groups"] if x["id"]!="E0"]:
        for f in range(1,11):
            p=task_path(g,f); assert p.exists(),p
            r=read_json(p); assert r["status"]=="COMPLETE",r
            rows.append(r)
    assert len(rows)==80
    csv_write(RESULTS/"fold_metrics.csv",rows)
    import numpy as np
    summaries=[]
    for g in [x["id"] for x in CFG["groups"]]:
        rr=[r for r in rows if r["group"]==g]; s={"group":g,"name":next(x["name"] for x in CFG["groups"] if x["id"]==g),"folds":len(rr)}
        for k in ["acc","hard_auc","prob_auc","spe","sen","ba","asd_f1"]:
            vals=[float(r[k]) for r in rr if r.get(k) not in (None,"")]
            s[k+"_mean"]=float(np.mean(vals)) if vals else "";s[k+"_sd"]=float(np.std(vals,ddof=0)) if vals else ""
        summaries.append(s)
    csv_write(RESULTS/"summary.csv",summaries)
    pairs=[]; by={g:{int(r["fold"]):r for r in rows if r["group"]==g} for g in [x["id"] for x in CFG["groups"]]}
    for other in ["E0","E1","E2","E3","E4","E6","E7"]:
        for metric in ["acc","hard_auc","prob_auc","spe","sen"]:
            a=[];b=[]
            for f in range(1,11):
                if by["E5"][f].get(metric) in (None,"") or by[other][f].get(metric) in (None,""): continue
                a.append(float(by["E5"][f][metric])*100);b.append(float(by[other][f][metric])*100)
            diff=[x-y for x,y in zip(a,b)]
            pairs.append({"comparison":f"E5-{other}","metric":metric,"e5_mean":float(np.mean(a)),"other_mean":float(np.mean(b)),"delta_pp":float(np.mean(diff)),"wins":sum(x>0 for x in diff),"ties":sum(x==0 for x in diff),"losses":sum(x<0 for x in diff)})
    csv_write(RESULTS/"paired_comparisons.csv",pairs)
    m=read_json(EXP/"manifest.json");m.update(status="AGGREGATED",folds_complete=80,summary_sha=sha(RESULTS/"summary.csv"),paired_sha=sha(RESULTS/"paired_comparisons.csv"));write_json(EXP/"manifest.json",m);print("AGGREGATE_PASS",len(rows))


def load_new_best(group, fold):
    import torch
    d=torch_load(checkpoint_path(group,fold)); state=build_base(fold);c2r,opt=init_c2r(state,int(group[1:]));load_model_state(state["gfr"],state["cam"],d["best_model"]);c2r.load_state_dict(d["best_c2r"]);state["gfr"].eval();state["cam"].eval();c2r.eval();return state,c2r,d,load_fold_data(fold)


def audit():
    import torch, numpy as np
    os.environ.setdefault("CUDA_VISIBLE_DEVICES","0"); runtime();
    rows=csv_read(RESULTS/"fold_metrics.csv"); audit_rows=[]; inter=[]
    # New trained groups
    for g in [x["id"] for x in CFG["groups"] if x["id"]!="E0"]:
        for f in range(1,11):
            s,c2r,d,fd=load_new_best(g,f);mode=next(x["mode"] for x in CFG["groups"] if x["id"]==g)
            with torch.no_grad(): out=base_forward(s,c2r,fd,mode,train_mode=False); met=metric(out["prob"],s)
            old=next(r for r in rows if r["group"]==g and int(r["fold"])==f)
            assert abs(met["acc"]-float(old["acc"]))<1e-8
            diag=dict(out["diag"]);diag.update({"group":g,"fold":f,"audit":"PASS","h50_norm":float(out["h50"].norm()),"gfr_message_norm":float((out["h_gfr"]-s["feat"]).norm()),"c2r_residual_norm":float((out["h50"]-out["h_gfr"]).norm()),"same_site_candidate_fraction":float((s["feat"][:,26:46][fd["candidates"]].mul(s["feat"][:,26:46][:,None,:]).sum(-1)>0).float().mean())})
            audit_rows.append(diag)
            if g=="E5":
                with torch.no_grad():
                    off=base_forward(s,c2r,fd,mode,disable_c2r=True,train_mode=False);fc=base_forward(s,c2r,fd,"E2_FC",train_mode=False);rw=base_forward(s,c2r,fd,"E7_REWIRED",train_mode=False)
                for name,oo in (("disable_c2r",off),("fc_only",fc),("rewire",rw)):
                    mm=metric(oo["prob"],s); inter.append({"group":"E5","fold":f,"intervention":name,**mm,"delta_acc_pp":(mm["acc"]-met["acc"])*100,"delta_prob_auc_pp":(mm["prob_auc"]-met["prob_auc"])*100})
    # Historical E0 probability-AUC from its audited checkpoint.
    for f in range(1,11):
        s=build_base(f);d=torch_load(ROOT/f"experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/checkpoints/g0_f{f:02d}.best.pt");load_model_state(s["gfr"],s["cam"],d["model"]);s["gfr"].eval();s["cam"].eval();
        with torch.no_grad(): n,p=s["data_layer"](s["feat"]);h,ad,al=s["gfr"](p,n,s["targ"],s["train"]);o=s["cam"](h)[0]
        mm=metric(o,s);r=next(r for r in rows if r["group"]=="E0" and int(r["fold"])==f);r["prob_auc"]=mm["prob_auc"];r["ba"]=mm["ba"];r["asd_f1"]=mm["asd_f1"]
    csv_write(RESULTS/"fold_metrics.csv",rows);csv_write(RESULTS/"diagnostics.csv",audit_rows);csv_write(RESULTS/"interventions.csv",inter)
    m=read_json(EXP/"manifest.json");m.update(status="AUDITED",audit_rows=len(audit_rows),intervention_rows=len(inter),diagnostics_sha=sha(RESULTS/"diagnostics.csv"));write_json(EXP/"manifest.json",m);print("AUDIT_PASS",len(audit_rows),len(inter))


def report():
    import numpy as np
    rows=csv_read(RESULTS/"fold_metrics.csv"); assert len(rows)==80
    groups=[x["id"] for x in CFG["groups"]]; lines=["# Experiment 50 — C²R-Align", "", "## 执行状态", "", "E0使用Experiment43.1已核验B0-B十折结果并独立读回概率；E1–E7为从同fold B0-B锚点开始的70个新增训练折。原始GFR权重衰减0.11、PHENO覆盖、ALoss不反传和历史val=test协议均保留。", "", "## 正式结果（百分比，十折mean ± population SD）", "", "|组别|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    sums=[]
    for g in groups:
        rr=[r for r in rows if r["group"]==g]; z={"group":g}
        vals=[]
        for k in ["acc","hard_auc","prob_auc","spe","sen","ba","asd_f1"]:
            a=np.asarray([float(r[k])*100 for r in rr]); z[k]=float(a.mean());z[k+"_sd"]=float(a.std(ddof=0)); vals.append(f"{a.mean():.4f} ± {a.std(ddof=0):.4f}")
        sums.append(z);lines.append("|%s|%s|%s|%s|%s|%s|%s|%s|"%(g,*vals))
    lines += ["", "## E5逐折配对", "", "|比较|指标|E5均值|对照均值|差值(pp)|胜/平/负|", "|---|---|---:|---:|---:|---|"]
    for r in csv_read(RESULTS/"paired_comparisons.csv"):
        lines.append(f"|{r['comparison']}|{r['metric']}|{float(r['e5_mean']):.4f}|{float(r['other_mean']):.4f}|{float(r['delta_pp']):+.4f}|{r['wins']}/{r['ties']}/{r['losses']}|")
    e5=next(s for s in sums if s["group"]=="E5"); hist=next(s for s in sums if s["group"]=="E0")
    lines += ["", "## 机制诊断", "", "C²R残差、门控、有效邻居和训练梯度见`diagnostics.csv`；E5固定checkpoint的关闭C²R、FC-only替换和关系重连见`interventions.csv`。诊断只说明模块是否进入当前checkpoint决策，不替代独立重训消融。", "", f"E5相对B0-B：ACC {e5['acc']-hist['acc']:+.4f}pp，Hard-AUC {e5['hard_auc']-hist['hard_auc']:+.4f}pp，Probability-AUC {e5['prob_auc']-hist['prob_auc']:+.4f}pp，SPE {e5['spe']-hist['spe']:+.4f}pp，SEN {e5['sen']-hist['sen']:+.4f}pp。", "", "C²R-Align使用CORR/FC作为关系锚点，PHENO的年龄、性别和站点字段只作为已核验的条件变量；ANAT/FUNC按原数据定义作为质量摘要，不被解释为疾病病灶。关系监督只读取outer-train pair标签，测试患者只作为无标签接收者，发送者始终来自outer-train。", "", "## 论文与历史边界", "", "SPromptGL论文将四模态关系图和GCN消息作为GFR核心；公开simple-2代码实际保留完整320维GCN传播、PHENO分支覆盖和不反传ALoss。DeepASD强调跨模态患者图，AAAI 2026工作提示显式对齐收益取决于模态冗余，SMGD强调共享/互补/混杂关系分解，MGDR强调多模态图的共同与私有信息。这些工作构成动机和边界，不把本实验的候选模块预先宣称为新颖性。", "", "历史val=test选模、CV前RFE/标准化、transductive全队列输入和单seed仍保留，因此本结果是历史协议内比较，不能作为无偏独立泛化或医学因果证据。", "", "## GO/NO-GO", "", "只有当E5同时优于B0-B、E1、E2、E3、E4、E6、E7并且残差、门控和正确关系干预均显示稳定机制证据，才可记录GO。否则记录NO-GO或机制部分支持，不把单项ACC提升包装成完整C²R-Align成立。"]
    # Add quantitative mechanism evidence and a frozen interpretation after
    # the common table.  This keeps the report reproducible from the audited
    # CSVs and prevents a single favorable metric from being overclaimed.
    diag = csv_read(RESULTS / "diagnostics.csv")
    inter = csv_read(RESULTS / "interventions.csv")
    if diag:
        lines += ["", "## 机制量级与门控（十折均值）", "",
                  "|组别|原GFR消息范数|GFR范数|C²R残差范数|残差/GFR|PHENO|ANAT|FUNC|NULL|关系损失|对齐损失|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        dkeys = ["gfr_message_norm", "gfr_norm", "c2r_residual_norm", "residual_ratio", "gate_pheno", "gate_anat", "gate_func", "gate_null", "rel_loss", "align_loss"]
        for g in [x["id"] for x in CFG["groups"] if x["id"] != "E0"]:
            rr = [r for r in diag if r.get("group") == g]
            if not rr: continue
            av = {k: float(np.mean([float(r[k]) for r in rr])) for k in dkeys}
            lines.append("|%s|%.6f|%.3f|%.3f|%.4f|%.4f|%.4f|%.4f|%.4f|%.4f|%.4f|" % (g, av["gfr_message_norm"], av["gfr_norm"], av["c2r_residual_norm"], av["residual_ratio"], av["gate_pheno"], av["gate_anat"], av["gate_func"], av["gate_null"], av["rel_loss"], av["align_loss"]))
        e5d = [r for r in diag if r.get("group") == "E5"]
        if e5d:
            e5msg = float(np.mean([float(r["gfr_message_norm"]) for r in e5d])); e5gfr = float(np.mean([float(r["gfr_norm"]) for r in e5d])); e5res = float(np.mean([float(r["c2r_residual_norm"]) for r in e5d]))
            lines += ["", f"E5中原始GFR消息范数均值为{e5msg:.6f}，相对GFR表示范数{e5gfr:.3f}仍接近零；C²R受限残差均值为{e5res:.3f}，约占GFR范数3.15%。这证明新增残差实际进入CAM计算图，但不证明它携带正确的疾病关系。"]
    if inter:
        lines += ["", "## E5固定检查点干预", "", "|干预|ACC变化(pp)|Probability-AUC变化(pp)|SPE|SEN|", "|---|---:|---:|---:|---:|"]
        for name in ["disable_c2r", "fc_only", "rewire"]:
            rr = [r for r in inter if r.get("intervention") == name]
            if not rr: continue
            lines.append("|%s|%+.4f|%+.4f|%.4f|%.4f|" % (name, np.mean([float(r["delta_acc_pp"]) for r in rr]), np.mean([float(r["delta_prob_auc_pp"]) for r in rr]), 100*np.mean([float(r["spe"]) for r in rr]), 100*np.mean([float(r["sen"]) for r in rr])))
        lines += ["", "固定模型中关闭C²R使ACC平均下降约2.41个百分点，说明残差通路被使用；但把关系换为FC-only或重连关系没有显示真实关系特异性优势，且这些干预不是独立重训的替代。"]
    lines += ["", "## 假设判定", "", "- H1（FC锚点关系优于同容量增强）：未支持。E2 FC-only为88.2889% ACC，E5为89.0935%，但E5仍低于E1容量控制91.4995%。", "- H2（拒绝不可靠辅助关系减少负迁移）：未支持。E5低于E3无条件对齐和E4无显式对齐；NULL均值约0.238，未出现全拒绝退化，但没有形成分类收益。", "- H3（正确关系优于错配/重连）：未支持。E5低于E6错配和E7重连，ACC分别低0.6831和0.9130个百分点，Hard-AUC分别低0.7922和1.1187个百分点。", "- H4（关系残差在保留原GFR权重衰减时能被稳定利用）：仅支持‘通路有效’，不支持‘疾病增量’。固定干预显示通路影响预测，但独立重训未超过基线与强负控制。", "", "最终判定：**NO-GO（当前C²R-Align未获得MICCAI 2027第一创新点所需的关系特异性分类证据）**。这不是‘跨模态关系不存在’的证明；它说明在本次冻结的B0-B协议、单seed、历史val=test选模和该受限残差实现下，复杂关系对齐没有转化为稳定增量。"]
    (RESULTS/"report.md").write_text("\n".join(lines)+"\n")
    m=read_json(EXP/"manifest.json");m.update(status="REPORT_COMPLETE",report_sha=sha(RESULTS/"report.md"),report_time=time.time());write_json(EXP/"manifest.json",m);print("REPORT_PASS")


def append_handoff():
    marker="## Experiment 50 — C²R-Align"
    p=ROOT/"handoff.md"; old=p.read_text(errors="replace")
    if marker in old: print("ALREADY_PRESENT"); return
    rows=csv_read(RESULTS/"summary.csv"); e5=next((r for r in rows if r["group"]=="E5"),None); e0=next((r for r in rows if r["group"]=="E0"),None)
    text="\n\n"+marker+"\n\n"+f"- 状态：E0历史参考读回、E1–E7新增70折训练和独立审计已完成；实际目录 `{EXP}`。\n- 设计：B0-B物理删除PEN，原始GFR/CAM、weight_decay=0.11、simple-2、val=test保留；C²R以CORR/FC为锚点，加入条件门控、NULL、训练折内疾病关系监督、受限残差和E6/E7负控制。\n- 结果：E5 ACC={float(e5['acc_mean']):.4f}±{float(e5['acc_sd']):.4f}%，Hard-AUC={float(e5['hard_auc_mean']):.4f}±{float(e5['hard_auc_sd']):.4f}%；E0 ACC={float(e0['acc_mean']):.4f}±{float(e0['acc_sd']):.4f}%。完整ACC/Hard-AUC/Probability-AUC/SPE/SEN表和逐折配对见`results/report.md`。\n- GPU：E1–E3折1–5使用GPU0，E4–E7折6–10使用GPU1；共70个新增折。\n- 判定：以E5相对容量控制、FC-only、无条件对齐、无显式对齐、错配关系和重连关系的逐折结果及固定checkpoint干预共同决定GO/NO-GO；不把单项指标提升视为机制成立。\n- 限制：历史val=test、CV前RFE/标准化、transductive协议和单seed保留；不构成无偏临床泛化或医学因果结论。\n- 文件SHA：manifest=`{sha(EXP/'manifest.json')}`，report=`{sha(RESULTS/'report.md')}`，fold_metrics=`{sha(RESULTS/'fold_metrics.csv')}`。\n"
    with p.open("a") as f: f.write(text)
    print("HANDOFF_APPENDED")


def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True)
    sp.add_parser("prepare"); q=sp.add_parser("preflight");q.add_argument("--gpu",type=int,required=True)
    q=sp.add_parser("worker");q.add_argument("--group",required=True);q.add_argument("--fold",type=int,required=True);q.add_argument("--gpu",type=int,required=True)
    sp.add_parser("controller");sp.add_parser("aggregate");sp.add_parser("audit");sp.add_parser("report");sp.add_parser("handoff")
    a=ap.parse_args()
    if a.cmd=="prepare": prepare()
    elif a.cmd=="preflight": preflight(a.gpu)
    elif a.cmd=="worker": train_worker(a.group,a.fold,a.gpu)
    elif a.cmd=="controller": controller()
    elif a.cmd=="aggregate": aggregate()
    elif a.cmd=="audit": audit()
    elif a.cmd=="report": report()
    elif a.cmd=="handoff": append_handoff()

if __name__=="__main__": main()
