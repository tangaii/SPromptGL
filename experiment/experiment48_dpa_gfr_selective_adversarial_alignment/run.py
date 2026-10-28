#!/usr/bin/env python3
"""Experiment 48 isolated runner, resumable at task/checkpoint granularity."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/SPromptGL-main")
RT = HERE / ".runtime"
RESULTS = HERE / "results"
CFG = json.loads((HERE / "config.json").read_text())
GROUPS = {int(g["id"]): g for g in CFG["groups"]}
GROUP_NAMES = {k: v["name"] for k, v in GROUPS.items()}
DATA_CSV = ROOT / "data/ABIDE/processed_standard_data.csv"
DATA_MAP = ROOT / "data/ABIDE/modal_feat_dict.npy"
SOURCE_DIR = RT / "sources"
CTX = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def save_torch(path, value):
    import torch
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        torch.save(value, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def load_torch(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def load_context(device_index=0):
    global CTX
    if CTX is not None:
        return CTX
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 48 requires a CUDA device")
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sys.path.insert(0, str(SOURCE_DIR))
    import layers, network, utils
    import method
    CTX = dict(np=np, pd=pd, torch=torch, layers=layers, network=network,
               utils=utils, method=method, device=torch.device("cuda"))
    return CTX


def seed_everything(seed=0):
    c = load_context()
    torch, np = c["torch"], c["np"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    c = load_context(); torch, np = c["torch"], c["np"]
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    c = load_context(); torch, np = c["torch"], c["np"]
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def rng_sha(state):
    h = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    h.update(state["cpu"].cpu().numpy().tobytes())
    for x in state["cuda"]: h.update(x.cpu().numpy().tobytes())
    return h.hexdigest()


def load_data():
    c = load_context(); np, pd = c["np"], c["pd"]
    from sklearn.model_selection import StratifiedKFold
    frame = pd.read_csv(DATA_CSV)
    mapping = np.load(DATA_MAP, allow_pickle=True).item()
    assert list(mapping) == ["PHENO", "ANAT", "FUNC", "Correlation"]
    assert [len(mapping[k]) for k in mapping] == [48, 6, 10, 256]
    assert list(frame.columns[:-1]) == [str(i) for k in mapping for i in mapping[k]]
    x = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = frame.iloc[:, -1].to_numpy(dtype=np.int64) - 1
    assert x.shape == (871, 320) and np.isfinite(x).all()
    assert np.bincount(y).tolist() == [403, 468]
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10).split(x, y))
    return x, y, folds


def fold_sha(train, test):
    import numpy as np
    return hashlib.sha256(np.asarray(train, dtype=np.int64).tobytes() + np.asarray(test, dtype=np.int64).tobytes()).hexdigest()


def base_digest(a):
    h = hashlib.sha256()
    for prefix, module in (("gfr", a.MessagePassing0), ("pln", a.ModalFusion)):
        for name, p in sorted(module.named_parameters()):
            h.update((prefix + "." + name).encode() + b"\0")
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def equal_tensors(a, b):
    c = load_context(); torch, np = c["torch"], c["np"]
    if isinstance(a, torch.Tensor): return bool(torch.equal(a, b))
    if isinstance(a, np.ndarray): return bool(np.array_equal(a, b))
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal_tensors(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(equal_tensors(x, y) for x, y in zip(a, b))
    return a == b


def model_bundle(a):
    return {"base": {"gfr": {k: v.detach().cpu().clone() for k, v in a.MessagePassing0.state_dict().items()},
                      "pln": {k: v.detach().cpu().clone() for k, v in a.ModalFusion.state_dict().items()}},
            "evidence": {k: v.detach().cpu().clone() for k, v in a.evidence.state_dict().items()}}


def load_bundle(a, bundle):
    a.MessagePassing0.load_state_dict(bundle["base"]["gfr"])
    a.ModalFusion.load_state_dict(bundle["base"]["pln"])
    a.evidence.load_state_dict(bundle["evidence"])


def instantiate_base(x, y, train, test):
    c = load_context(); m = c["method"]
    return m.build_base(c["torch"], c["layers"], c["network"], c["utils"], c["device"], x, y, train, test)


def new_agent(x, y, train, test, fold, group_id, anchor):
    c = load_context(); m = c["method"]
    # Recreate the immutable B0-B entry stream before the new module fork.
    restore_rng(anchor["entry_rng"])
    a = m.make_new_agent(c["torch"], c["layers"], c["network"], c["utils"], c["device"],
                         x, y, train, test, fold, group_id, anchor,
                         init_seed=480000 + int(fold))
    restore_rng(anchor["rng"])
    return a


def metric(output, y, test):
    return load_context()["method"].metric_row(output, y, test)


def verified_e0(fold, x, y, train, test, gpu):
    c = load_context(); m = c["method"]
    saved = load_torch(RT / "checkpoints" / f"E0_f{fold:02d}.best.pt")
    a = instantiate_base(x, y, train, test)
    a.MessagePassing0.load_state_dict(saved["model"]["gfr"])
    a.ModalFusion.load_state_dict(saved["model"]["pln"])
    with c["torch"].no_grad():
        out = m.evaluate(a, 0, saved.get("best_epoch", 0), c["utils"])
    row = metric(out, y, test)
    # Independent readback must reproduce the historical saved vector/metrics.
    pred = out["logp"].detach().cpu().numpy()[test].argmax(1)
    assert equal_tensors(pred, saved["prediction"]), f"E0 prediction mismatch fold {fold}"
    for k in ("acc", "hard_auc", "sen", "spe", "asd_tp", "asd_fn", "td_fp", "td_tn"):
        assert abs(float(row[k]) - float(saved["metrics"][k])) < 1e-12, (fold, k, row[k], saved["metrics"][k])
    return {
        "group_id": 0, "group": GROUP_NAMES[0], "fold": fold, "seed": 0,
        "physical_gpu": gpu, "status": "COMPLETE", "reused_verified": True,
        "history_checkpoint": str(RT / "checkpoints" / f"E0_f{fold:02d}.best.pt"),
        "best_epoch": int(saved["best_epoch"]), "epochs_executed": int(saved["epochs_executed"]),
        "wall_seconds": 0.0, "prediction_sha256": hashlib.sha256(pred.tobytes()).hexdigest(),
        "initial_parameter_sha256": saved["initial_parameter_sha256"],
        "initial_rng_sha256": saved["initial_rng_sha256"],
        **row,
        "diagnostics": [{"group": GROUP_NAMES[0], "group_id": 0, "fold": fold,
                         "base_dependency": 0.0, "graph_dependency": 0.0,
                         "graph_only_acc": None, "graph_only_hard_auc": None,
                         "message_norm": 0.0, "gamma": 0.0, "domain_acc_last": None,
                         "conflict_rate": 0.0, "projected_rate": 0.0}],
        "trace": []
    }


def scalar(v):
    try: return float(v)
    except Exception: return None


def diagnostics(a, group_id, fold, epoch, y, test, trace, out):
    c = load_context(); t = c["torch"]
    rows = []
    base_lp = out["base_logp"]
    final_lp = out["logp"]
    test_t = t.as_tensor(test, device=final_lp.device, dtype=t.long)
    base_pred = base_lp[test_t].argmax(1)
    final_pred = final_lp[test_t].argmax(1)
    dep = float((final_lp[test_t] - base_lp[test_t]).abs().mean().detach())
    flip = float((base_pred != final_pred).float().mean().detach())
    graph_acc = graph_auc = None
    if out.get("graph_logits") is not None:
        from sklearn.metrics import roc_auc_score
        gp = out["graph_logits"][test_t].detach().cpu().numpy()
        gy = y[test]
        graph_pred = gp.argmax(1)
        graph_acc = float((graph_pred == gy).mean())
        graph_auc = float(roc_auc_score(gy, gp[:, 1]))
    last = trace[-1] if trace else {}
    conflict_steps = [r for r in trace if r.get("lambda_adv", 0.0) > 0]
    conflicts = sum(int(r.get("conflict", 0)) for r in conflict_steps)
    projected = sum(int(r.get("conflict", 0)) for r in conflict_steps)
    common = dict(group=GROUP_NAMES[group_id], group_id=group_id, fold=fold,
                  graph_dependency_logp_mean=dep, graph_prediction_flip=flip,
                  graph_only_acc=graph_acc, graph_only_probability_auc=graph_auc,
                  gamma=float(out.get("gamma", 0.0)), domain_acc_last=last.get("domain_acc", 0.0),
                  domain_loss_last=last.get("adv", 0.0), conflict_steps=conflicts,
                  adv_steps=len(conflict_steps), conflict_rate=(conflicts / len(conflict_steps) if conflict_steps else 0.0),
                  projected_rate=(projected / len(conflict_steps) if conflict_steps else 0.0),
                  pcgrad_cosine_last=last.get("conflict_cosine", 0.0),
                  gate_pheno=last.get("gate_pheno"), gate_anat=last.get("gate_anat"),
                  gate_func=last.get("gate_func"), gate_corr=last.get("gate_corr"))
    if group_id == 0:
        rows.append(common)
    else:
        for i, info in enumerate(out["meta"]):
            r = dict(common, relation=c.MODALITIES[i] if hasattr(c, "MODALITIES") else ["PHENO", "ANAT", "FUNC", "CORR"][i],
                     edge_count=info.get("edge_count", 0), self_edge_count=info.get("self_edge_count", 0),
                     weight_mean=info.get("weight_mean", 0.0), weight_sd=info.get("weight_sd", 0.0),
                     same_sender_set=info.get("same_sender_set", False), message_norm=float(out["msgs"][i].norm().detach()))
            rows.append(r)
    return rows


def task_complete(group_id, fold):
    p = RT / "tasks" / f"g{group_id}_f{fold:02d}.json"
    if not p.exists(): return False
    try:
        d = json.loads(p.read_text())
        return d.get("status") == "COMPLETE" and d.get("signature") == code_signature()
    except Exception:
        return False


def code_signature():
    files = [HERE / "run.py", HERE / "method.py", HERE / "config.json"] + list(SOURCE_DIR.glob("*.py"))
    return hashlib.sha256(json.dumps({str(p): sha(p) for p in files}, sort_keys=True).encode()).hexdigest()


def execute_new(group_id, fold, gpu, x, y, train, test, folds):
    c = load_context(); m = c["method"]
    key = f"g{group_id}_f{fold:02d}"
    task_path = RT / "tasks" / f"{key}.json"
    best_path = RT / "checkpoints" / f"{key}.best.pt"
    latest_path = RT / "checkpoints" / f"{key}.latest.pt"
    if task_complete(group_id, fold):
        return json.loads(task_path.read_text())
    anchor = load_torch(RT / "anchors" / f"fold{fold:02d}.pt")
    assert anchor["split_sha"] == fold_sha(train, test)
    a = new_agent(x, y, train, test, fold, group_id, anchor)
    base = dict(group_id=group_id, group=GROUP_NAMES[group_id], fold=fold, seed=0,
                physical_gpu=gpu, signature=code_signature(), split_sha=fold_sha(train, test),
                anchor_sha=sha(RT / "anchors" / f"fold{fold:02d}.pt"),
                initial_parameter_sha256=anchor["parameter_sha"],
                initial_rng_sha256=rng_sha(anchor["entry_rng"]),
                train_sender_count=len(train), top_k=16)
    begin = 0; bestacc = -1.0; bestepoch = -1; wait = 0; best = None; trace = []; elapsed = 0.0
    if latest_path.exists():
        s = load_torch(latest_path)
        assert s["signature"] == code_signature() and s["split_sha"] == base["split_sha"]
        load_bundle(a, s["current"])
        a.optimizer_MP0.load_state_dict(s["mp"]); a.optimizer_MF.load_state_dict(s["mf"])
        a.optimizer_new.load_state_dict(s["new"]); a.optimizer_domain.load_state_dict(s["domain"])
        restore_rng(s["rng"]); begin = int(s["epoch"]) + 1; bestacc = float(s["bestacc"])
        bestepoch = int(s["bestepoch"]); wait = int(s["wait"]); best = s["best"]
        trace = s["trace"]; elapsed = float(s["elapsed"])
        print("RESUME", key, begin, flush=True)
    write_json(task_path, dict(**base, status="RUNNING", epoch=begin - 1, best_epoch=bestepoch))
    started = time.monotonic()
    epoch = begin - 1
    for epoch in range(begin, int(CFG["base_protocol"]["epochs"])):
        values = m.train_step(a, group_id, epoch, c["utils"])
        with c["torch"].no_grad(): out = m.evaluate(a, group_id, epoch, c["utils"])
        acc = float(out["logp"][a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean().item())
        if acc > bestacc:
            bestacc = acc; bestepoch = epoch; wait = 0; best = model_bundle(a)
        else:
            wait += 1
        if epoch % 10 == 0 or epoch == int(CFG["base_protocol"]["epochs"]) - 1:
            trace.append(dict(epoch=epoch, **values))
        if ((epoch + 1) % 10 == 0) or wait > int(CFG["base_protocol"]["early"]) or epoch == int(CFG["base_protocol"]["epochs"]) - 1:
            save_torch(latest_path, dict(**base, current=model_bundle(a), mp=a.optimizer_MP0.state_dict(),
                                        mf=a.optimizer_MF.state_dict(), new=a.optimizer_new.state_dict(),
                                        domain=a.optimizer_domain.state_dict(), rng=rng_state(), epoch=epoch,
                                        bestacc=bestacc, bestepoch=bestepoch, wait=wait, best=best,
                                        trace=trace, elapsed=elapsed + time.monotonic() - started))
            write_json(task_path, dict(**base, status="RUNNING", epoch=epoch, best_epoch=bestepoch))
        if wait > int(CFG["base_protocol"]["early"]): break
    if best is None: raise RuntimeError(f"no best state for {key}")
    load_bundle(a, best)
    with c["torch"].no_grad(): out = m.evaluate(a, group_id, bestepoch, c["utils"])
    row = metric(out, y, test)
    pred = out["logp"].detach().cpu().numpy()[test].argmax(1)
    save_torch(best_path, dict(**base, model=best, prediction=pred, test_indices=test,
                              best_epoch=bestepoch, epochs_executed=epoch + 1, metrics=row,
                              trace=trace, diagnostics=diagnostics(a, group_id, fold, bestepoch, y, test, trace, out),
                              rng_boundary=rng_state(), elapsed=elapsed + time.monotonic() - started))
    task = dict(**base, status="COMPLETE", best_epoch=bestepoch, epochs_executed=epoch + 1,
                wall_seconds=elapsed + time.monotonic() - started,
                prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(), **row,
                diagnostics=diagnostics(a, group_id, fold, bestepoch, y, test, trace, out), trace=trace,
                best_sha=sha(best_path))
    write_json(task_path, task)
    latest_path.unlink(missing_ok=True)
    print("FOLD_DONE", key, "gpu", gpu, "ACC", row["acc"], "epoch", bestepoch,
          "seconds", round(task["wall_seconds"], 2), flush=True)
    return task


def worker(gpu, fold_list):
    c = load_context(gpu); seed_everything(0)
    x, y, folds = load_data()
    # E0 is an allowed verified historical reuse, but it is independently
    # instantiated and read back in this process.
    for fold in fold_list:
        train, test = folds[fold - 1]
        if not task_complete(0, fold):
            r = verified_e0(fold, x, y, train, test, gpu)
            r["signature"] = code_signature(); r["split_sha"] = fold_sha(train, test)
            write_json(RT / "tasks" / f"g0_f{fold:02d}.json", r)
        for group_id in range(1, 8):
            execute_new(group_id, fold, gpu, x, y, train, test, folds)
    print("WORKER_DONE", gpu, flush=True)


def refresh_csv():
    rows = []; diags = []; traces = []
    for p in sorted((RT / "tasks").glob("g*_f*.json")):
        d = json.loads(p.read_text())
        if d.get("status") != "COMPLETE": continue
        row = {k: v for k, v in d.items() if k not in ("diagnostics", "trace")}
        rows.append(row); diags.extend(d.get("diagnostics", []))
        for t in d.get("trace", []): traces.append(dict(group=d["group"], group_id=d["group_id"], fold=d["fold"], **t))
    rows.sort(key=lambda r: (int(r["group_id"]), int(r["fold"])))
    for path, data in [(RESULTS / "fold_metrics.csv", rows), (RESULTS / "diagnostics.csv", diags), (RESULTS / "loss_trace.csv", traces)]:
        if not data:
            continue
        fields = list(dict.fromkeys(k for r in data for k in r))
        tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    return rows, diags, traces


def run_preflight():
    c = load_context(); t = c["torch"]; x, y, folds = load_data()
    checks = {"status": "PASS", "time": time.time(), "groups": {}}
    # Check the B0-B initial identity for every fold, and exact historical E0
    # readback for one fold before any formal task starts.
    seed_everything(0)
    for fold, (train, test) in enumerate(folds, 1):
        anchor = load_torch(RT / "anchors" / f"fold{fold:02d}.pt")
        restore_rng(anchor["entry_rng"])
        a = instantiate_base(x, y, train, test)
        assert base_digest(a) == anchor["parameter_sha"], fold
        assert len(list(a.ModalFusion.parameters())) == 35
        checks["groups"][str(fold)] = {"base_initial_sha": base_digest(a), "anchor_sha": sha(RT / "anchors" / f"fold{fold:02d}.pt")}
    # Explicitly verify all group routing and GRL/PCGrad semantics on one fold.
    train, test = folds[0]; anchor = load_torch(RT / "anchors/fold01.pt")
    for gid in range(1, 8):
        a = new_agent(x, y, train, test, 1, gid, anchor)
        before = rng_sha(anchor["rng"]); assert rng_sha(rng_state()) == before
        val = c["method"].train_step(a, gid, 0, c["utils"])
        assert all(t.isfinite(p).all() for p in list(a.MessagePassing0.parameters()) + list(a.ModalFusion.parameters()) + list(a.evidence.parameters()))
        checks["groups"][f"E{gid}"] = {"first_step": val, "rng_restored": True}
    # Domain-label and GRL sign microcheck independent of ABIDE labels.
    x0 = t.tensor([[1., 0.], [2., 0.]], device="cuda", requires_grad=True)
    y0 = c["method"].grad_reverse(x0, 0.3).sum(); g = t.autograd.grad(y0, x0)[0]
    assert t.allclose(g, t.full_like(g, -0.3)); checks["grl_sign"] = "PASS"
    write_json(RT / "preflight.json", checks)
    print("PREFLIGHT_PASS", json.dumps({"folds": 10, "groups": 7, "grl": "PASS"}), flush=True)


def initialize_manifest():
    protected = {}
    for p in [ROOT / n for n in ["main.py", "model.py", "network.py", "layers.py", "utils.py", "data/ABIDE/processed_standard_data.csv", "data/ABIDE/modal_feat_dict.npy"]]:
        protected[str(p)] = sha(p)
    for p in sorted((ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/sources").glob("*.py")):
        protected[str(p)] = sha(p)
    manifest = {"status": "INITIALIZED", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "experiment_dir": str(HERE), "config_sha256": sha(HERE / "config.json"),
                "protected_sha256": protected, "data_sha256": {str(DATA_CSV): sha(DATA_CSV), str(DATA_MAP): sha(DATA_MAP)},
                "historical_anchor_dir": str(ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"),
                "source_sha256": {str(p): sha(p) for p in sorted(SOURCE_DIR.glob("*.py"))},
                "code_signature": code_signature(), "gpu": str(load_context()["torch"].cuda.get_device_properties(0))}
    write_json(HERE / "manifest.json", manifest)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("preflight")
    w = sub.add_parser("worker"); w.add_argument("--gpu", type=int, required=True); w.add_argument("--folds", required=True)
    sub.add_parser("refresh")
    args = ap.parse_args()
    if args.cmd == "init": initialize_manifest()
    elif args.cmd == "preflight": run_preflight()
    elif args.cmd == "worker": worker(args.gpu, [int(v) for v in args.folds.split(",")])
    elif args.cmd == "refresh":
        rows, diags, traces = refresh_csv(); print("REFRESH", len(rows), "rows", len(diags), "diagnostics")


if __name__ == "__main__":
    main()
