#!/usr/bin/env python3
"""Experiment 52 runner.

Only E0 (exact E51-E1 replay) and E1 (DRAP) are scheduled.  The root
SPromptGL implementation and all historical checkpoints are read-only; this
file imports private copies in ``source/`` and writes every new artifact under
the Experiment 52 directory.
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

EXP = Path(os.environ.get("E52_EXP", Path(__file__).resolve().parent))
ROOT = Path("/data/adodas/SPromptGL-main")
SRC = EXP / "source"
RT = EXP / ".runtime"
TASKS = RT / "tasks"
CKPTS = RT / "checkpoints"
LATEST = RT / "latest"
LOGS = RT / "logs"
FD = RT / "fold_data"
RESULTS = EXP / "results"
CFG_PATH = EXP / "config.json"
CFG = json.loads(CFG_PATH.read_text())
DIMS = [48, 6, 10, 256]
MODS = ["PHENO", "ANAT", "FUNC", "CORR"]
DRAP_GROUPS = ("E1", "E2", "E3")
SUPPLEMENTAL_GFR_WEIGHT_DECAY = {"E2": 0.20, "E3": 0.50}
ANCHORS = ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
HIST_E51_2 = Path("/tmp/SPromptGL-E51_2/experiment51_2_pdcsra_gfr")
HIST_E51_2_CK = HIST_E51_2 / ".runtime/checkpoints"
CTX = None
DATA = None
FOLDS = None


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_obj(obj) -> str:
    return hashlib.sha256(pickle.dumps(obj, protocol=4)).hexdigest()


def atomic_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float, allow_nan=False))
    os.replace(tmp, path)


def torch_save(path: Path, obj):
    import torch
    path = Path(path)
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


def csv_write(path: Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def csv_read(path: Path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def seed_all(seed: int):
    import numpy as np
    import torch
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pack_rng():
    import numpy as np
    import torch
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def rng_sha(state) -> str:
    import numpy as np
    h = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    h.update(state["cpu"].cpu().numpy().tobytes())
    for x in state["cuda"]:
        h.update(x.cpu().numpy().tobytes())
    return h.hexdigest()


def runtime():
    global CTX
    if CTX is None:
        import numpy as np
        import pandas as pd
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        sys.path.insert(0, str(SRC))
        sys.path.insert(0, str(EXP))
        import layers
        import network
        import utils
        import drap
        CTX = {
            "torch": torch,
            "np": np,
            "pd": pd,
            "layers": layers,
            "network": network,
            "utils": utils,
            "drap": drap,
            "device": torch.device("cuda"),
        }
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
    assert x.shape == (871, 320)
    assert np.isfinite(x).all()
    assert np.bincount(y).tolist() == [403, 468]
    assert [len(mapping[k]) for k in ("PHENO", "ANAT", "FUNC", "Correlation")] == DIMS
    folds = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y))
    DATA, FOLDS = (x, y), folds
    return DATA, FOLDS


def split_sha(fold: int) -> str:
    import numpy as np
    tr, te = load_data()[1][fold - 1]
    return hashlib.sha256(np.asarray(tr, np.int64).tobytes() + np.asarray(te, np.int64).tobytes()).hexdigest()


def fold_anchor(fold: int):
    anchor = torch_load(ANCHORS / f"fold{fold:02d}.pt")
    expected = split_sha(fold)
    if anchor.get("split_sha") not in (None, expected):
        raise RuntimeError(f"anchor split mismatch fold {fold}: {anchor.get('split_sha')} != {expected}")
    return anchor


def hp():
    b = CFG["base"]
    return SimpleNamespace(
        n_hidden=b["n_hidden"], n_head=b["n_head"], dropout=b["dropout"],
        nlayer=b["nlayer"], nmodal=4, nclass=2,
        lr=b["lr"], reg=b["weight_decay"], th=b["threshold"],
        GC_mode="weighted-cosine", MP_mode="GCN", MF_mode="concat",
        alpha=0.5, theta_smooth=b["theta_smooth"],
        theta_degree=b["theta_degree"], theta_sparsity=b["theta_sparsity"],
    )


def is_drap_group(group: str) -> bool:
    return group in DRAP_GROUPS


def gfr_weight_decay(group: str) -> float:
    return float(SUPPLEMENTAL_GFR_WEIGHT_DECAY.get(group, CFG["base"]["weight_decay"]))


def state_cpu(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def optimizer_cpu_state(opt):
    d = copy.deepcopy(opt.state_dict())
    for state in d.get("state", {}).values():
        for k, v in list(state.items()):
            if hasattr(v, "detach"):
                state[k] = v.detach().cpu().clone()
    return d


def optimizer_to_cuda(opt):
    for state in opt.state.values():
        for k, v in list(state.items()):
            if hasattr(v, "to"):
                state[k] = v.cuda()


def digest_shared(base, cam):
    h = hashlib.sha256()
    for prefix, module in (("gfr", base), ("cam", cam)):
        for name, p in sorted(module.named_parameters()):
            h.update((prefix + ":" + name).encode())
            h.update(str(tuple(p.shape)).encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def schema_shared(base, cam):
    rows = []
    for prefix, module in (("gfr", base), ("cam", cam)):
        rows.extend((prefix + ":" + n, tuple(p.shape), str(p.dtype)) for n, p in sorted(module.named_parameters()))
    return sha_obj(rows)


def load_base_anchor(base, anchor_gfr):
    base.fusion_weight.data.copy_(anchor_gfr["fusion_weight"])
    for m in range(4):
        gs = {k[len(f"GraphConstruct{m}."):]: v for k, v in anchor_gfr.items() if k.startswith(f"GraphConstruct{m}.")}
        ms = {k[len(f"MessagePassing{m}."):]: v for k, v in anchor_gfr.items() if k.startswith(f"MessagePassing{m}.")}
        base.graphs[m].load_state_dict(gs, strict=True)
        base.gcn[m].load_state_dict(ms, strict=True)


def initial_candidate_pairs(x, y, tr, anchor):
    """Union initial GraphLearn supports, then balance with local RNG only."""
    import numpy as np
    import torch
    tr = np.asarray(tr, dtype=np.int64)
    starts = np.cumsum([0] + DIMS[:-1])
    candidate = set()
    for m, (start, dim) in enumerate(zip(starts, DIMS)):
        g = anchor["model"]["gfr"]
        w = g[f"GraphConstruct{m}.p.weight"].cpu().numpy()
        b = g[f"GraphConstruct{m}.p.bias"].cpu().numpy()
        z = x[:, start:start + dim] @ w.T + b
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        s = z @ z.T
        ii, jj = np.where(np.triu(s > CFG["base"]["threshold"], 1))
        trset = set(tr.tolist())
        for i, j in zip(ii.tolist(), jj.tolist()):
            if i in trset and j in trset:
                candidate.add((int(i), int(j)))
    ii, jj = np.triu_indices(len(tr), 1)
    pi, pj = tr[ii], tr[jj]
    same = (y[pi] == y[pj])
    all_by = {1: list(zip(pi[same].tolist(), pj[same].tolist())), 0: list(zip(pi[~same].tolist(), pj[~same].tolist()))}
    cand_by = {1: [p for p in candidate if y[p[0]] == y[p[1]]], 0: [p for p in candidate if y[p[0]] != y[p[1]]]}
    rng = np.random.default_rng(520300 + int(tr[0]) % 991)
    k = min(CFG["drap"]["pair_max"] // 2, len(all_by[0]), len(all_by[1]))
    selected = []
    selected_set = set()
    for cls in (0, 1):
        c = list(cand_by[cls])
        if len(c) > k:
            pick = rng.choice(len(c), size=k, replace=False)
            chosen = [c[int(i)] for i in pick]
        else:
            chosen = c[:]
        selected.extend(chosen)
        selected_set.update(chosen)
    for cls in (0, 1):
        remaining = [p for p in all_by[cls] if p not in selected_set]
        need = k - sum(1 for p in selected if (y[p[0]] == y[p[1]]) == (cls == 1))
        if need > 0:
            take = rng.choice(len(remaining), size=need, replace=False) if len(remaining) > need else np.arange(len(remaining))
            chosen = [remaining[int(i)] for i in np.asarray(take).reshape(-1)]
            selected.extend(chosen)
            selected_set.update(chosen)
    rng.shuffle(selected)
    pi = np.asarray([p[0] for p in selected], dtype=np.int64)
    pj = np.asarray([p[1] for p in selected], dtype=np.int64)
    py = (y[pi] == y[pj]).astype(np.int64)
    cand_selected = sum(p in candidate for p in selected)
    return pi, pj, py, {
        "pair_count": int(len(selected)),
        "same_count": int(py.sum()),
        "different_count": int((1 - py).sum()),
        "candidate_edge_count_outer_train": int(len(candidate)),
        "candidate_edge_coverage": float(cand_selected / max(len(candidate), 1)),
        "rng_seed": int(520300 + int(tr[0]) % 991),
    }


def prepare_fold(fold: int):
    import numpy as np
    FD.mkdir(parents=True, exist_ok=True)
    npz = FD / f"fold{fold:02d}.npz"
    meta_path = FD / f"fold{fold:02d}.json"
    if npz.exists() and meta_path.exists():
        return
    x, y = load_data()[0]
    tr, te = load_data()[1][fold - 1]
    anchor = fold_anchor(fold)
    pi, pj, py, pm = initial_candidate_pairs(x, y, tr, anchor)
    np.savez_compressed(npz, train=np.asarray(tr), test=np.asarray(te), pair_i=pi, pair_j=pj, pair_y=py)
    atomic_json(meta_path, {
        "fold": fold, "split_sha": split_sha(fold), "pair_sampling": pm,
        "pair_sha": sha_file(npz), "test_label_excluded_from_pair_sampling": True,
    })


def load_fold_data(fold: int):
    import numpy as np
    import torch
    z = np.load(FD / f"fold{fold:02d}.npz")
    return {k: torch.from_numpy(z[k]).cuda() for k in z.files}


def build_state(group: str, fold: int):
    import torch
    import numpy as np
    c = runtime()
    x, y = load_data()[0]
    tr, te = load_data()[1][fold - 1]
    anchor = fold_anchor(fold)
    seed = 510000 + fold * 100 + 1
    seed_all(seed)
    # This is the exact E51 base/CAM construction order.
    base = c["drap"].LegacyFixedGFR(c["network"].GraphLearn, c["network"].GCN, c["utils"], th=CFG["base"]["threshold"]).cuda()
    cam = c["drap"].PLNNoPEN(DIMS, hp(), c["layers"]).cuda()
    cam.apply(c["utils"].my_weight_init)
    if is_drap_group(group):
        # New parameters are initialized in an isolated Torch RNG scope.  No
        # Python/NumPy/CUDA training RNG from the E1 base is consumed here.
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(520000 + fold)
            model = c["drap"].DRAPGFR(
                base, dims=DIMS,
                hidden=CFG["drap"]["relation_hidden"],
                grl_scale=CFG["drap"]["grl_scale"],
            ).cuda()
    else:
        model = base
    load_base_anchor(base, anchor["model"]["gfr"])
    cam.load_state_dict(anchor["model"]["pln"], strict=True)
    post_construct_rng = pack_rng()
    feat = torch.from_numpy(x).float().cuda()
    targ = torch.from_numpy(y).long().cuda()
    train = torch.as_tensor(tr, dtype=torch.long, device="cuda")
    counts = Counter(y[tr])
    weight = torch.from_numpy((len(tr) / np.asarray(list(counts.values())) / 2).astype(np.float32)).cuda()
    opt_base = torch.optim.Adam(base.parameters(), lr=CFG["base"]["lr"], weight_decay=gfr_weight_decay(group))
    opt_cam = torch.optim.Adam(cam.parameters(), lr=CFG["base"]["lr"], weight_decay=CFG["base"]["weight_decay"])
    if is_drap_group(group):
        base_ids = {id(p) for p in base.parameters()}
        new_params = [p for p in model.parameters() if id(p) not in base_ids]
        opt_drap = torch.optim.Adam(new_params, lr=CFG["drap"]["optimizer_lr"], weight_decay=CFG["drap"]["optimizer_weight_decay"])
    else:
        opt_drap = None
    return {
        "group": group, "fold": fold, "model": model, "base": base, "cam": cam,
        "opt_base": opt_base, "opt_cam": opt_cam, "opt_drap": opt_drap,
        "feat": feat, "targ": targ, "train": train, "test": np.asarray(te),
        "weight": weight, "anchor": anchor, "initial_rng": post_construct_rng,
        "initial_rng_sha": rng_sha(post_construct_rng),
        "anchor_rng_sha": rng_sha(anchor["rng"]),
        "entry_rng_sha": rng_sha(anchor.get("entry_rng", anchor["rng"])),
        "initial_shared_sha": digest_shared(base, cam),
        "initial_schema_sha": schema_shared(base, cam),
        "epoch": 0,
    }


def metrics(logp, state):
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    idx = state["test"]
    yy = state["targ"].detach().cpu().numpy()[idx]
    probs = logp.detach().exp().cpu().numpy()[idx]
    pred = probs.argmax(1)
    cm = confusion_matrix(yy, pred, labels=[0, 1])
    tp, fn, fp, tn = map(int, cm.ravel())
    return {
        "acc": float(np.mean(pred == yy)),
        "hard_auc": float(roc_auc_score(np.eye(2)[yy], np.eye(2)[pred])),
        "prob_auc": float(roc_auc_score(yy, probs[:, 1])),
        "spe": float(tn / max(tn + fp, 1)),
        "sen": float(tp / max(tp + fn, 1)),
        "ba": float((tn / max(tn + fp, 1) + tp / max(tp + fn, 1)) / 2),
        "asd_f1": float(f1_score(yy, pred, pos_label=0, zero_division=0)),
        "asd_tp": tp, "asd_fn": fn, "td_fp": fp, "td_tn": tn,
        "n_test": int(len(idx)),
    }


def module_grad_norm(module):
    import torch
    vals = [(p.grad.detach() ** 2).sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(vals).sum().sqrt().cpu()) if vals else 0.0


def forward_state(state, fd, train_mode=True, use_reweight=True, force_self=False):
    import torch
    c = runtime()
    state["model"].train(train_mode)
    state["cam"].train(train_mode)
    _, parts = c["layers"].InitialDataLayer(DIMS)(state["feat"])
    if not is_drap_group(state["group"]):
        fused, adjs, details = state["base"](parts)
        pair_loss = fused.new_zeros(())
        adv_loss = fused.new_zeros(())
    else:
        fused, adjs, details = state["model"](
            parts,
            pair_i=fd["pair_i"].long(), pair_j=fd["pair_j"].long(),
            pair_y=fd["pair_y"].long(),
            use_reweight=use_reweight, force_self=force_self,
        )
        pair_loss = details["pair_loss"]
        adv_loss = details["adv_loss"]
    logp, hidden, attn = state["cam"](fused)
    graph = sum(
        c["utils"].GraphConstructLoss(parts[m], adjs[m], CFG["base"]["theta_smooth"], CFG["base"]["theta_degree"], CFG["base"]["theta_sparsity"])
        for m in range(4)
    ) / 4.0
    cls = c["utils"].ClsLoss(logp, state["targ"], state["train"], state["weight"])
    if not is_drap_group(state["group"]):
        total = cls + graph
    else:
        total = cls + graph + CFG["drap"]["lambda_pair"] * pair_loss + CFG["drap"]["lambda_adv"] * adv_loss
    return {
        "logp": logp, "hidden": hidden, "attn": attn, "fused": fused,
        "parts": parts, "adjs": adjs, "details": details,
        "cls": cls, "graph": graph, "pair_loss": pair_loss,
        "adv_loss": adv_loss, "total": total,
    }


def task_path(group, fold):
    return TASKS / f"{group}_f{fold:02d}.json"


def checkpoint_path(group, fold):
    return CKPTS / f"{group}_f{fold:02d}.best.pt"


def latest_path(group, fold):
    return LATEST / f"{group}_f{fold:02d}.latest.pt"


def train_one(group: str, fold: int):
    import torch
    runtime()
    prepare_fold(fold)
    tp = task_path(group, fold)
    if tp.exists() and json.loads(tp.read_text()).get("status") == "COMPLETE":
        return
    state = build_state(group, fold)
    fd = load_fold_data(fold)
    start = 0
    wait = 0
    best_acc = -1.0
    best_epoch = -1
    best_model = best_cam = best_ob = best_oc = best_od = None
    best_diag = {}
    best_use_reweight = True
    first_active_cls_grad = 0.0
    first_total_drap_grad = 0.0
    lp = latest_path(group, fold)
    if lp.exists():
        try:
            d = torch_load(lp)
            if d.get("group") != group or int(d.get("fold", -1)) != fold:
                raise ValueError("latest identity mismatch")
            state["model"].load_state_dict(d["model"])
            state["cam"].load_state_dict(d["cam"])
            state["opt_base"].load_state_dict(d["opt_base"]); optimizer_to_cuda(state["opt_base"])
            state["opt_cam"].load_state_dict(d["opt_cam"]); optimizer_to_cuda(state["opt_cam"])
            if state["opt_drap"] is not None and d.get("opt_drap") is not None:
                state["opt_drap"].load_state_dict(d["opt_drap"]); optimizer_to_cuda(state["opt_drap"])
            restore_rng(d["rng"])
            start = int(d["epoch"]) + 1
            wait = int(d["wait"]); best_acc = float(d["best_acc"]); best_epoch = int(d["best_epoch"])
            best_model = d.get("best_model"); best_cam = d.get("best_cam"); best_ob = d.get("best_ob"); best_oc = d.get("best_oc"); best_od = d.get("best_od"); best_diag = d.get("best_diag", {})
            best_use_reweight = bool(d.get("best_use_reweight", True)); first_active_cls_grad = float(d.get("first_active_cls_grad", 0.0)); first_total_drap_grad = float(d.get("first_total_drap_grad", 0.0))
        except Exception:
            start = 0
    t0 = time.time()
    epoch = start - 1
    try:
        for epoch in range(start, CFG["base"]["epochs"]):
            state["epoch"] = epoch
            use_rw = (not is_drap_group(group)) or (epoch >= CFG["drap"]["warmup_epochs"])
            state["opt_base"].zero_grad(); state["opt_cam"].zero_grad()
            if state["opt_drap"] is not None:
                state["opt_drap"].zero_grad()
            out = forward_state(state, fd, train_mode=True, use_reweight=use_rw)
            assert torch.isfinite(out["total"])
            if is_drap_group(group) and use_rw and first_active_cls_grad == 0.0:
                rel_params = list(state["model"].relation_encoders.parameters()) + list(state["model"].relation_scorers.parameters())
                cls_grads = torch.autograd.grad(out["cls"], rel_params, retain_graph=True, allow_unused=True)
                first_active_cls_grad = float(torch.stack([(g.detach() ** 2).sum() for g in cls_grads if g is not None]).sum().sqrt().cpu()) if any(g is not None for g in cls_grads) else 0.0
            out["total"].backward()
            for mod in (state["model"], state["cam"]):
                for p in mod.parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all()
            if is_drap_group(group) and first_total_drap_grad == 0.0:
                first_total_drap_grad = module_grad_norm(state["model"])
            gn_base = module_grad_norm(state["base"])
            gn_cam = module_grad_norm(state["cam"])
            gn_drap = module_grad_norm(state["model"]) if is_drap_group(group) else 0.0
            state["opt_base"].step(); state["opt_cam"].step()
            if state["opt_drap"] is not None:
                state["opt_drap"].step()
            with torch.no_grad():
                ev = forward_state(state, fd, train_mode=False, use_reweight=use_rw)
                met = metrics(ev["logp"], state)
            if met["acc"] > best_acc:
                best_acc = met["acc"]; best_epoch = epoch; wait = 0; best_use_reweight = use_rw
                best_model = state_cpu(state["model"]); best_cam = state_cpu(state["cam"])
                best_ob = optimizer_cpu_state(state["opt_base"]); best_oc = optimizer_cpu_state(state["opt_cam"])
                best_od = optimizer_cpu_state(state["opt_drap"]) if state["opt_drap"] is not None else None
                best_diag = {
                    "train_cls": float(out["cls"].detach()), "train_graph": float(out["graph"].detach()),
                    "pair_loss": float(out["pair_loss"].detach()), "adv_loss": float(out["adv_loss"].detach()),
                    "base_grad_norm": gn_base, "cam_grad_norm": gn_cam, "drap_grad_norm": gn_drap,
                    "classification_to_relation_grad_norm": first_active_cls_grad,
                    "initial_total_drap_grad_norm": first_total_drap_grad,
                }
                torch_save(checkpoint_path(group, fold), {
                    "group": group, "fold": fold, "seed": 0,
                    "best_epoch": best_epoch, "epochs_executed": epoch + 1,
                    "metrics": met, "model": best_model, "cam": best_cam,
                    "opt_base": best_ob, "opt_cam": best_oc, "opt_drap": best_od,
                    "diagnostics": best_diag, "best_use_reweight": best_use_reweight,
                    "split_sha": split_sha(fold), "pair_sha": sha_file(FD / f"fold{fold:02d}.npz"),
                    "initial_shared_parameter_sha": state["initial_shared_sha"],
                    "initial_shared_schema_sha": state["initial_schema_sha"],
                    "initial_rng_sha": state["initial_rng_sha"],
                    "anchor_rng_sha": state["anchor_rng_sha"],
                    "entry_rng_sha": state["entry_rng_sha"],
                    "source_sha": sha_file(EXP / "drap.py"),
                })
            else:
                wait += 1
            if epoch % 10 == 0:
                torch_save(latest_path(group, fold), {
                    "group": group, "fold": fold, "epoch": epoch, "wait": wait,
                    "best_acc": best_acc, "best_epoch": best_epoch,
                    "best_use_reweight": best_use_reweight,
                    "model": state_cpu(state["model"]), "cam": state_cpu(state["cam"]),
                    "opt_base": optimizer_cpu_state(state["opt_base"]), "opt_cam": optimizer_cpu_state(state["opt_cam"]),
                    "opt_drap": optimizer_cpu_state(state["opt_drap"]) if state["opt_drap"] is not None else None,
                    "rng": pack_rng(), "best_model": best_model, "best_cam": best_cam,
                    "best_ob": best_ob, "best_oc": best_oc, "best_od": best_od,
                    "best_diag": best_diag, "first_active_cls_grad": first_active_cls_grad,
                    "first_total_drap_grad": first_total_drap_grad,
                })
            if wait > CFG["base"]["early"]:
                break
        assert best_model is not None
        state["model"].load_state_dict(best_model); state["cam"].load_state_dict(best_cam)
        with torch.no_grad():
            check = metrics(forward_state(state, fd, train_mode=False, use_reweight=best_use_reweight)["logp"], state)
        assert abs(check["acc"] - best_acc) < 1e-8
        row = {
            "group": group, "group_id": int(group[1:]),
            "name": next(g["name"] for g in CFG["groups"] if g["id"] == group),
            "fold": fold, "seed": 0, "gpu": 0, "status": "COMPLETE", "source": "new_training",
            "split_sha": split_sha(fold), "pair_sha": sha_file(FD / f"fold{fold:02d}.npz"),
            "best_epoch": best_epoch, "epochs_executed": epoch + 1,
            "warmup_active_at_best": int(not best_use_reweight),
            "wall_seconds": time.time() - t0,
            "best_sha": sha_file(checkpoint_path(group, fold)),
            "initial_shared_parameter_sha": state["initial_shared_sha"],
            "initial_shared_schema_sha": state["initial_schema_sha"],
            "initial_rng_sha": state["initial_rng_sha"],
            "anchor_rng_sha": state["anchor_rng_sha"], "entry_rng_sha": state["entry_rng_sha"],
            "classification_to_relation_grad_norm": first_active_cls_grad,
            "initial_total_drap_grad_norm": first_total_drap_grad,
            **check,
        }
        atomic_json(tp, row)
        latest_path(group, fold).unlink(missing_ok=True)
        print("DONE", group, fold, best_epoch, row["acc"], flush=True)
    except Exception as exc:
        atomic_json(tp, {"group": group, "fold": fold, "status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()})
        raise


def prepare():
    import numpy as np
    for p in (RT, TASKS, CKPTS, LATEST, LOGS, FD, RESULTS):
        p.mkdir(parents=True, exist_ok=True)
    x, y = load_data()[0]
    for fold in range(1, 11):
        prepare_fold(fold)
    manifest = {
        "experiment": CFG["experiment"], "status": "PREPARED",
        "config_sha": sha_file(CFG_PATH),
        "code_sha": {p.name: sha_file(p) for p in [EXP / "run.py", EXP / "drap.py"]},
        "private_source_sha": {p.name: sha_file(p) for p in SRC.glob("*.py")},
        "protected_source_sha256": {str(ROOT / n): sha_file(ROOT / n) for n in [
            "main.py", "model.py", "network.py", "layers.py", "utils.py",
            "data/ABIDE/processed_standard_data.csv", "data/ABIDE/modal_feat_dict.npy"]},
        "historical_source_sha256": {
            "experiment51_dugsa": {str(p): sha_file(p) for p in (ROOT / "experiment/experiment51_dugsa").glob("*.py")},
            "experiment51_2_run": sha_file(HIST_E51_2 / "run.py"),
            "experiment51_2_dcsra_base": sha_file(HIST_E51_2 / "dcsra_base.py"),
            "experiment51_2_pdcsra": sha_file(HIST_E51_2 / "pdcsra.py"),
        },
        "data_shape": list(x.shape), "label_counts": np.bincount(y).tolist(),
        "folds": {str(f): split_sha(f) for f in range(1, 11)},
        "anchor_sha256": {str(f): sha_file(ANCHORS / f"fold{f:02d}.pt") for f in range(1, 11)},
        "historical_e51_2_checkpoint_sha256": {str(f): sha_file(HIST_E51_2_CK / f"E1_f{f:02d}.best.pt") for f in range(1, 11)},
        "formal_groups": ["E0", "E1"], "formal_folds": 20,
        "planned_new_folds": 20, "e1_seed_rule": "510000 + fold*100 + 1",
        "new_module_rng_rule": "isolated torch.random.fork_rng seed 520000+fold",
        "test_labels_for_drap": "forbidden",
    }
    atomic_json(EXP / "manifest.json", manifest)
    print("PREPARE_PASS")


def controller(groups):
    groups = [g.strip() for g in groups if g.strip()]
    assert groups in (["E0"], ["E1"], ["E0", "E1"]), groups
    jobs = [(g, f) for g in groups for f in range(1, 11)]
    procs = []
    workers = min(5, int(os.environ.get("E52_WORKERS", CFG["runtime"]["workers_max"])))
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "0"
    env.update({k: "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    start = time.time()
    while jobs or procs:
        alive = sum(p.poll() is None for _, p, _ in procs)
        while alive < workers and jobs:
            g, f = jobs.pop(0)
            log = (LOGS / f"{g}_f{f:02d}.log").open("ab")
            p = subprocess.Popen([sys.executable, "-B", str(EXP / "run.py"), "worker", "--group", g, "--fold", str(f)], cwd=EXP, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            procs.append((g, p, log)); alive += 1
        for item in list(procs):
            if item[1].poll() is not None:
                item[2].close(); procs.remove(item)
        time.sleep(0.5)
    statuses = []
    for g in groups:
        statuses.extend(json.loads(task_path(g, f).read_text()).get("status", "MISSING") if task_path(g, f).exists() else "MISSING" for f in range(1, 11))
    manifest = json.loads((EXP / "manifest.json").read_text())
    manifest.setdefault("controller_runs", []).append({"groups": groups, "completed": statuses.count("COMPLETE"), "failed": statuses.count("FAILED"), "wall_seconds": time.time() - start})
    manifest["status"] = "E0_COMPLETE" if groups == ["E0"] and statuses.count("COMPLETE") == 10 else ("E1_COMPLETE" if groups == ["E1"] and statuses.count("COMPLETE") == 10 else manifest.get("status", "TRAINING_PARTIAL"))
    atomic_json(EXP / "manifest.json", manifest)
    print("CONTROLLER_DONE", groups, statuses.count("COMPLETE"), statuses.count("FAILED"), flush=True)


def aggregate():
    import numpy as np
    rows = []
    for g in ("E0", "E1"):
        for f in range(1, 11):
            p = task_path(g, f)
            assert p.exists(), p
            r = json.loads(p.read_text()); assert r["status"] == "COMPLETE", r
            rows.append(r)
    csv_write(RESULTS / "fold_metrics.csv", rows)
    summaries = []
    for g in ("E0", "E1"):
        rr = [r for r in rows if r["group"] == g]
        q = {"group": g, "name": rr[0]["name"], "folds": len(rr)}
        for k in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"):
            z = np.asarray([float(r[k]) for r in rr])
            q[k + "_mean_pct"] = float(z.mean() * 100)
            q[k + "_sd_pct"] = float(z.std(ddof=0) * 100)
        summaries.append(q)
    csv_write(RESULTS / "summary.csv", summaries)
    by = {(r["group"], int(r["fold"])): r for r in rows}
    pairs = []
    for k in ("acc", "hard_auc", "prob_auc", "spe", "sen"):
        d = np.asarray([(float(by[("E1", f)][k]) - float(by[("E0", f)][k])) * 100 for f in range(1, 11)])
        pairs.append({"comparison": "E1_minus_E0", "metric": k, "delta_pp": float(d.mean()), "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()), "fold_deltas_pp": json.dumps(d.tolist())})
    csv_write(RESULTS / "paired_comparisons.csv", pairs)
    manifest = json.loads((EXP / "manifest.json").read_text())
    manifest.update({"status": "AGGREGATED", "formal_folds_complete": 20, "fold_metrics_sha": sha_file(RESULTS / "fold_metrics.csv"), "summary_sha": sha_file(RESULTS / "summary.csv"), "paired_sha": sha_file(RESULTS / "paired_comparisons.csv")})
    atomic_json(EXP / "manifest.json", manifest)
    print("AGGREGATE_PASS")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prepare")
    c = sp.add_parser("controller"); c.add_argument("--groups", nargs="+", default=["E0"])
    sp.add_parser("aggregate")
    w = sp.add_parser("worker"); w.add_argument("--group", required=True); w.add_argument("--fold", required=True, type=int)
    args = ap.parse_args()
    if args.cmd == "prepare": prepare()
    elif args.cmd == "controller": controller(args.groups)
    elif args.cmd == "aggregate": aggregate()
    else: train_one(args.group, args.fold)


if __name__ == "__main__":
    main()
