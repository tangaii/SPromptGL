#!/usr/bin/env python3
"""Independent, frozen 14-group AG-CDEA experiment and continuous-fold recovery."""
import argparse
import contextlib
import csv
import difflib
import fcntl
import hashlib
import io
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"
if "preflight" in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from cdea import (CONFIGURATIONS, DESIGNS, BANKS, DOMAIN_PERMUTATION, EvidenceAlignment, TEMPERATURE, LAMBDA_EVIDENCE, RESIDUAL_LIMIT)

EXP = Path(__file__).resolve().parent
ROOT = EXP.parent.parent
STATE = Path("/tmp") / ("spromptgl19-state-" + hashlib.sha256(str(EXP).encode()).hexdigest()[:12])
SOURCES = STATE / "sources"
MANIFEST, RESULTS, LOGS = EXP / "manifest.json", EXP / "results", EXP / "logs"
CORE = ["main.py", "model.py", "network.py", "layers.py", "utils.py", "ABIDE-simple-2-concat-weighted-cosine.sh"]
CONFIGS = list(DESIGNS)
TASKS = [(c, s, (i * 3 + s) % 2) for i, c in enumerate(CONFIGS) for s in range(3)]
METRICS = ["acc", "ba", "hard_auc", "probability_auc", "sensitivity_asd", "specificity_td"]
CONDITIONS = ["REAL", "NEUTRAL", "UNRELATED", "CLASS_ONLY", "SWAPPED", "DOMAIN_PERMUTED", "ANATOMY_PERMUTED", "RANDOM_ROTATED", "NULL_OFF", "RESIDUAL_OFF", "FC_SHUFFLED"]
PAIRS = [("C1","C0"),("C2","C1"),("C4","C2"),("C4","C3")] + [("C4",f"C{i}") for i in range(5,14)] + [("C4","C0")]
PROTOCOL = dict(datadir=str(ROOT / "data") + "/", datname="ABIDE", cpu=False, gpu="-1",
                nepoch=450, early=200, lr=.0038, reg=.11, dropout=.35, nlayer=1,
                n_hidden=18, n_head=2, n_iter=10, nmodal=4, th=.9,
                GC_mode="weighted-cosine", MP_mode="GCN", MF_mode="concat", alpha=.5,
                theta_smooth=1., theta_degree=.5, theta_sparsity=0., nclass=2, mode="simple-2")

_CTX = None

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with open(temporary, "wb") as handle:
        handle.write(data.encode() if isinstance(data, str) else data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def save_json(path, value):
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def read_json(path):
    return json.loads(Path(path).read_text())

def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, buf.getvalue())

def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))

def save_checkpoint(path, obj):
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    atomic(path, b"SP19" + hashlib.sha256(data).hexdigest().encode() + data)

def checkpoint(path):
    import torch
    data = Path(path).read_bytes()
    assert data[:4] == b"SP19" and hashlib.sha256(data[68:]).hexdigest().encode() == data[4:68]
    return torch.load(io.BytesIO(data[68:]), map_location="cpu", weights_only=False)

def seed_rng(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())

def restore_rng(saved):
    import numpy as np
    import torch
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["cpu"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda"]])

def rng_hash(saved):
    h = hashlib.sha256(pickle.dumps((saved["python"], saved["numpy"]), protocol=4))
    for v in [saved["cpu"]] + saved["cuda"]:
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()

def data():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    table = pd.read_csv(ROOT / "data/ABIDE/processed_standard_data.csv")
    mapping = np.load(ROOT / "data/ABIDE/modal_feat_dict.npy", allow_pickle=True).item()
    assert list(mapping) == ["PHENO", "ANAT", "FUNC", "Correlation"]
    assert [len(v) for v in mapping.values()] == [48, 6, 10, 256]
    assert table.columns[:-1].tolist() == [str(v) for fields in mapping.values() for v in fields]
    values = table.values
    assert values.shape == (871, 321) and np.isfinite(values).all()
    x, y = values[:, :-1], values[:, -1].astype(int) - 1
    assert int((y == 0).sum()) == 403 and int((y == 1).sum()) == 468
    pheno = [i for i, n in enumerate(mapping["PHENO"]) if str(n).startswith(("AGE_AT_SCAN_", "SEX_"))]
    sites = [i for i, n in enumerate(mapping["PHENO"]) if str(n).startswith("SITE_ID_")]
    assert len(pheno) == 28 and len(sites) == 20 and not set(pheno).intersection(sites)
    folds = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y))
    audit = dict(shape=list(x.shape), labels={"ASD_0": 403, "TD_1": 468},
                 modality_columns={k: list(map(str, v)) for k, v in mapping.items()},
                 pheno_age_sex_indices=pheno, site_indices=sites)
    return x, y, folds, audit

def patched_sources():
    network = (ROOT / "network.py").read_text()
    model = (ROOT / "model.py").read_text()
    old = "    def forward(self, x):\n        bs = x.size(0)"
    assert network.count(old) == 1
    network = network.replace(old, "    def forward(self, x, raw_modalities=None):\n        bs = x.size(0)")
    old = "        text_features = self.prompt_text_features\n"
    assert network.count(old) == 1
    network = network.replace(old, "        if hasattr(self, 'cdea'):\n            x = self.cdea(x, raw_modalities)\n" + old)
    model = model.replace("from network import *\n", "from network import *\nimport cdea_runtime as _cdea_runtime\n", 1)
    old = "        self.ModalFusion.apply(my_weight_init)"
    assert model.count(old) == 1
    model = model.replace(old, old + "\n        _cdea_runtime.install_branch(self)")
    model = model.replace("prob,_,_ = self.ModalFusion(feat)", "prob,_,_ = self.ModalFusion(feat, self._cdea_raw)")
    model = model.replace("prob, fusion_feat, attn = self.ModalFusion(feat)", "prob, fusion_feat, attn = self.ModalFusion(feat, self._cdea_raw)")
    model = model.replace("prob, _, attn = self.ModalFusion(feat)", "prob, _, attn = self.ModalFusion(feat, self._cdea_raw)")
    old = "            loss = cls_loss + all_graph_loss"
    assert model.count(old) == 1
    model = model.replace(old, old + "\n            self._cdea_original_losses = (cls_loss.detach(), all_graph_loss.detach())\n            if hasattr(self.ModalFusion, 'cdea'):\n                loss = loss + self.ModalFusion.cdea.losses(self.targ)")
    return {"network.py": network, "model.py": model}

def make_sources():
    expected = patched_sources()
    SOURCES.mkdir(parents=True, exist_ok=True)
    for name in CORE:
        content = expected.get(name, (ROOT / name).read_text())
        path = SOURCES / name
        if path.exists():
            assert path.read_text() == content, ("temporary source differs", name)
        else:
            atomic(path, content)
    link = SOURCES / "CLIP"
    if not link.exists():
        link.symlink_to(ROOT / "CLIP", target_is_directory=True)

def historical_folders():
    return sorted(p for p in (ROOT / "experiment").iterdir() if p.is_dir() and
                  p.name.startswith("experiment") and p.name.split("_", 1)[0][10:].isdigit() and
                  1 <= int(p.name.split("_", 1)[0][10:]) <= 18)

def agent_for(config, seed, fold, original=False):
    c = context()
    c.update(current_config=config, current_fold=fold)
    cls = c["model"].EvalHelper
    if original:
        module = types.ModuleType("cdea_original_network")
        sys.modules[module.__name__] = module
        exec(compile((ROOT / "network.py").read_text(), "original_network.py", "exec"), module.__dict__)
        original_model = types.ModuleType("cdea_original_model")
        source = (ROOT / "model.py").read_text().replace("from network import *", "from cdea_original_network import *")
        exec(compile(source, "original_model.py", "exec"), original_model.__dict__)
        cls = original_model.EvalHelper
    train, test = c["folds"][fold - 1]
    with contextlib.redirect_stdout(io.StringIO()):
        a = cls([48, 6, 10, 256], c["x"], c["y"], SimpleNamespace(**PROTOCOL, seed=seed), train, test)
    a.MF_sav.close()
    a.GCMP_sav.close()
    assert (a.val_idx == a.tst_idx).all()
    return a

def branch(a):
    return getattr(a.ModalFusion, "cdea", None)

def model_state(a):
    return {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            for name, module in [("gfr", a.MessagePassing0), ("pln", a.ModalFusion)]}

def load_state(a, saved):
    a.MessagePassing0.load_state_dict(saved["gfr"])
    a.ModalFusion.load_state_dict(saved["pln"])

def weight_hash(a, auxiliary=False, legacy13=False):
    h = hashlib.sha256()
    for name, module in [("gfr", a.MessagePassing0), ("pln", a.ModalFusion)]:
        for key, value in sorted(module.named_parameters()):
            if key.startswith("cdea.") != auxiliary:
                continue
            tag = key if legacy13 else name + "." + key
            h.update(tag.encode() + value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def optimizer_audit(a, config):
    groups = a.optimizer_MP0.param_groups + a.optimizer_MF.param_groups
    all_params = [p for module in [a.MessagePassing0, a.ModalFusion] for p in module.parameters()]
    supplied = [p for group in groups for p in group["params"]]
    assert len(supplied) == len(set(map(id, supplied)))
    assert set(map(id, supplied)) == set(map(id, all_params))
    assert all(g["lr"] == .0038 for g in groups)
    assert a.optimizer_MP0.param_groups[0]["weight_decay"] == .11
    assert a.optimizer_MF.param_groups[0]["weight_decay"] == .11
    total = sum(p.numel() for p in all_params)
    assert total == 1462957 + read_json(MANIFEST)["additional_parameters"][config]
    if config != "C0":
        assert len(a.optimizer_MF.param_groups) == 2
        assert a.optimizer_MF.param_groups[1]["weight_decay"] == 0.
    return total

def forward(a):
    import torch
    a.MessagePassing0.eval()
    a.ModalFusion.eval()
    def capture(module, inputs, output):
        a._raw_logits = output.detach().clone()
    hook = a.ModalFusion.Outputlayer.classifier.register_forward_hook(capture)
    try:
        with torch.no_grad():
            n, parts = a.dataLayer(a.feat)
            hidden, _, _ = a.MessagePassing0(parts, n, a.targ, a.trn_idx)
            out, _, _ = a.ModalFusion(hidden, a._cdea_raw) if branch(a) else a.ModalFusion(hidden)
            assert torch.isfinite(out).all()
        return out.clone()
    finally:
        hook.remove()

def metrics(out, indices):
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_auc_score
    target = context()["y"][indices]
    probability = out.exp().detach().cpu().numpy()[indices]
    pred = probability.argmax(1)
    cm = confusion_matrix(target, pred, labels=[0, 1])
    sen, spe = float(cm[0, 0] / cm[0].sum()), float(cm[1, 1] / cm[1].sum())
    return dict(acc=float((pred == target).mean()), ba=(sen + spe) / 2,
                hard_auc=float(roc_auc_score(np.eye(2)[target], np.eye(2)[pred])),
                probability_auc=float(roc_auc_score(target == 0, probability[:, 0])),
                sensitivity_asd=sen, specificity_td=spe)

def parameter_norms(a):
    import torch
    b = branch(a)
    if b is None:
        return {}
    return {k: float(torch.stack([p.detach().square().sum() for p in pp]).sum().sqrt())
            for k, pp in b.component_parameters().items()}

def gradients(a):
    import torch
    b = branch(a)
    parts = {} if b is None else b.component_parameters()
    parts.update(original_gfr=list(a.MessagePassing0.parameters()),
                 original_pln=[p for k, p in a.ModalFusion.named_parameters() if not k.startswith("cdea.")])
    result = {}
    for k, pp in parts.items():
        gg = [p.grad.detach() for p in pp if p.grad is not None]
        result[k] = float(torch.stack([g.square().sum() for g in gg]).sum().sqrt()) if gg else 0.
    assert all(math.isfinite(v) for v in result.values())
    return result

def assert_historical_fold(row):
    if row["config"] != "C0":
        return
    for number, config in [(13, "F0"), (14, "E00"), (15, "A0"), (16, "B0"), (17, "D0"), (18, "V0")]:
        folder = next((ROOT / "experiment").glob(f"experiment{number}_*"))
        ref = next(r for r in read_csv(folder / "results/fold_metrics.csv")
                   if r["config"] == config and int(r["seed"]) == int(row["seed"]) and int(r["fold"]) == int(row["fold"]))
        for key in METRICS:
            reference_key = "balanced_accuracy" if number == 14 and key == "ba" else key
            assert abs(float(row[key]) - float(ref[reference_key])) < 1e-12, ("P0 parity metrics", number, row["seed"], row["fold"], key)
        for key in ("best_epoch", "epochs_executed"):
            assert int(row[key]) == int(ref[key]), ("P0 parity epoch", number, key)
        initial_key = "initial_trunk_legacy13_sha256" if number == 13 else "initial_trunk_sha256"
        assert row[initial_key] == ref["initial_trunk_sha256"], ("P0 initialization", number)
        rng_key = "fold_entry_rng_sha256" if number == 14 else "entry_rng_sha256"
        if rng_key in ref:
            assert row["entry_rng_sha256"] == ref[rng_key], ("P0 RNG parity", number)

def assert_equal_tree(a, b):
    import numpy as np
    import torch
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for v, w in zip(a, b):
            assert_equal_tree(v, w)
    else:
        assert a == b

def acquire_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle

def worker(config, seed):
    import torch
    m = integrity()
    assert m["preflight"]["status"] == "PASS"
    task = f"{config}_seed{seed}"
    gpu = int(os.environ["CDEA19_PHYSICAL_GPU"])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == str(gpu) and m["tasks"][task]["physical_gpu"] == gpu
    handle = acquire_lock(STATE / (task + ".lock"))
    result_path, checkpoint_path = STATE / (task + ".json"), STATE / (task + ".pt")
    sig = task_signature(config, seed)
    if result_path.exists():
        old = read_json(result_path)
        if old["status"] == "COMPLETE":
            assert old["signature"] == sig and len(old["metrics"]) == 10
            print("SKIP_COMPLETE " + task, flush=True)
            return
    c = context()
    saved = checkpoint(checkpoint_path) if checkpoint_path.exists() else None
    if saved:
        assert saved["signature"] == sig
        result = saved["result"]
        # Recreate the lazy CLIP cache BEFORE restoring the continuous RNG stream.
        c["network"].get_prompt_text_features()
        restore_rng(saved["rng"])
        start_fold = int(saved["fold"]) + (1 if saved["phase"] == "BETWEEN_FOLDS" else 0)
        if saved["phase"] == "BETWEEN_FOLDS":
            save_json(result_path, result)
        print(f"RESUME {task} phase={saved['phase']} fold={start_fold}", flush=True)
    else:
        seed_rng(seed)
        start_fold = 1
        result = dict(signature=sig, task=task, config=config, seed=seed, physical_gpu=gpu, status="RUNNING",
                      metrics=[], diagnostics=[], counterfactual=[], started_utc=utc(), seconds=0.)
    print(f"WORKER {task} GPU{gpu} logical=cuda:0 device={torch.cuda.get_device_name(0)} signature={sig}", flush=True)
    for fold in range(start_fold, 11):
        entry = rng()
        a = agent_for(config, seed, fold)
        optimizer_audit(a, config)
        initial, legacy13, branch_initial = weight_hash(a), weight_hash(a, legacy13=True), weight_hash(a, True)
        initial_norms = parameter_norms(a)
        tracker = {k: 0. for k in gradients(a)}
        tracker.update({"nonzero_parameters": 0, "nonzero_key_parameters": 0})
        best_acc, best_epoch, best_model, wait, start, prior_seconds = 0., -1, None, 0, 0, 0.
        if saved and saved["phase"] == "EPOCH" and int(saved["fold"]) == fold:
            load_state(a, saved["model"])
            a.optimizer_MP0.load_state_dict(saved["mp"])
            a.optimizer_MF.load_state_dict(saved["mf"])
            restore_rng(saved["rng"])
            best_acc, best_epoch, best_model, wait, start = (saved[k] for k in ["best_acc", "best_epoch", "best_model", "wait", "next_epoch"])
            tracker, entry, initial, legacy13, branch_initial = (saved[k] for k in ["tracker", "entry_rng", "initial", "legacy13", "branch_initial"])
            initial_norms, prior_seconds = saved["initial_norms"], saved["fold_seconds"]
        saved = None
        started, last = time.monotonic(), start - 1
        print(f"FOLD_START {fold} initial={initial} entry_rng={rng_hash(entry)}", flush=True)
        for epoch in (range(start, 450) if wait <= 200 else []):
            print(f"EPOCH {fold}:{epoch} ", end="")
            a.run_epoch("simple-2")
            if epoch in (0, 1) or (epoch + 1) % 25 == 0:
                for key, value in gradients(a).items():
                    tracker[key] = max(tracker[key], value)
                tracker["nonzero_parameters"] = max(tracker["nonzero_parameters"], nonzero_parameters(a))
                tracker["nonzero_key_parameters"] = max(tracker["nonzero_key_parameters"], nonzero_parameters(a, key_only=True))
            out = forward(a)
            validation = float(out[a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean())
            print(f"val-acc={validation:.12f}", flush=True)
            if validation > best_acc:
                best_acc, best_epoch, best_model, wait = validation, epoch, model_state(a), 0
            else:
                wait += 1
            last = epoch
            stop = wait > 200 or epoch == 449
            if (epoch + 1) % 25 == 0 or stop:
                if branch(a):
                    ll = branch(a).last
                    assert float((ll["delta"].norm(dim=-1) / (ll["base"].norm(dim=-1) + 1e-8)).max()) <= .100001
                save_checkpoint(checkpoint_path, dict(signature=sig, phase="EPOCH", fold=fold, next_epoch=epoch + 1,
                    model=model_state(a), mp=a.optimizer_MP0.state_dict(), mf=a.optimizer_MF.state_dict(), rng=rng(),
                    best_acc=best_acc, best_epoch=best_epoch, best_model=best_model, wait=wait, tracker=tracker,
                    entry_rng=entry, initial=initial, legacy13=legacy13, branch_initial=branch_initial,
                    initial_norms=initial_norms, fold_seconds=prior_seconds + time.monotonic() - started, result=result))
            if stop:
                break
        assert best_model is not None
        continuation = rng()
        load_state(a, best_model)
        out = forward(a)
        values = metrics(out, a.tst_idx)
        assert abs(values["acc"] - best_acc) < 1e-12 and abs(values["ba"] - values["hard_auc"]) < 1e-12
        row = dict(config=config, seed=seed, fold=fold, physical_gpu=gpu, task_signature=sig, **values, best_epoch=best_epoch,
                   epochs_executed=last + 1, selected_val_acc=best_acc, train_n=len(a.trn_idx), test_n=len(a.tst_idx),
                   val_equals_test=True, initial_trunk_sha256=initial, initial_trunk_legacy13_sha256=legacy13,
                   initial_branch_sha256=branch_initial if config != "C0" else "", entry_rng_sha256=rng_hash(entry),
                   additional_parameters=m["additional_parameters"][config])
        assert_historical_fold(row)
        diagnostic = diagnostics(a, config, seed, fold, out, tracker, initial_norms)
        diagnostic.update(task_signature=sig, best_epoch=best_epoch, epochs_executed=last + 1)
        cf = counterfactual(a, seed, fold, out) if config == "C4" else []
        for record in cf:
            record["task_signature"] = sig
        row["seconds"] = prior_seconds + time.monotonic() - started
        result["metrics"].append(row)
        result["diagnostics"].append(diagnostic)
        result["counterfactual"].extend(cf)
        restore_rng(continuation)
        result.update(seconds=sum(r["seconds"] for r in result["metrics"]), status="COMPLETE" if fold == 10 else "RUNNING", updated_utc=utc())
        save_checkpoint(checkpoint_path, dict(signature=sig, phase="BETWEEN_FOLDS", fold=fold, rng=continuation, result=result))
        save_json(result_path, result)
        print("FOLD_COMPLETE " + json.dumps(row, ensure_ascii=False), flush=True)
        del a, best_model
        if fold == 10:
            checkpoint_path.unlink(missing_ok=True)
    print("TASK_COMPLETE " + task, flush=True)
    handle.close()

def live_workers():
    import psutil
    found = {}
    for process in psutil.process_iter(["pid", "cmdline"]):
        args = process.info["cmdline"] or []
        if str(EXP / "run.py") in args and "--worker" in args:
            i = args.index("--worker")
            found[f"{args[i + 1]}_seed{args[i + 2]}"] = process.info["pid"]
    return found

def task_results():
    results = []
    for config, seed, gpu in TASKS:
        path = STATE / f"{config}_seed{seed}.json"
        if not path.exists():
            continue
        result = read_json(path)
        assert result["signature"] == task_signature(config, seed) and result["physical_gpu"] == gpu
        assert result["task"] == f"{config}_seed{seed}" and result["config"] == config and result["seed"] == seed
        folds = [r["fold"] for r in result["metrics"]]
        assert folds == list(range(1, len(folds) + 1)) and len(result["diagnostics"]) == len(folds)
        assert len(result["counterfactual"]) == (11 * len(folds) if config == "C4" else 0)
        if result["status"] == "COMPLETE":
            assert len(folds) == 10
        results.append(result)
    return results

def refresh():
    m = read_json(MANIFEST)
    if m["run_status"] == "COMPLETE" and not STATE.exists():
        return m
    completed = task_results()
    metrics_rows = [r for t in completed for r in t["metrics"]]
    ds = [r for t in completed for r in t["diagnostics"]]
    cf = [r for t in completed for r in t["counterfactual"]]
    assert len(set((r["config"], r["seed"], r["fold"]) for r in metrics_rows)) == len(metrics_rows)
    for task in completed:
        m["tasks"][task["task"]].update(status=task["status"], signature=task["signature"], folds=len(task["metrics"]), seconds=task["seconds"], updated_utc=task["updated_utc"])
    m.update(completed_folds=len(metrics_rows), completed_tasks=sum(t["status"] == "COMPLETE" for t in completed),
             diagnostic_rows=len(ds), counterfactual_rows=len(cf), updated_utc=utc())
    for name, rows in [("fold_metrics", metrics_rows), ("diagnostics", ds), ("counterfactual", cf)]:
        save_csv(RESULTS / (name + ".csv"), rows)
    save_json(MANIFEST, m)
    return m

def worker_env(gpu):
    return dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu), CDEA19_PHYSICAL_GPU=str(gpu),
                OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2",
                PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")

def resources():
    import psutil
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"], text=True)
    cards = []
    for line in raw.strip().splitlines():
        index, name, total, used, utilization = [v.strip() for v in line.split(",")]
        cards.append(dict(index=int(index), name=name, total_mib=int(total.split()[0]), used_mib=int(used.split()[0]), utilization=int(utilization.split()[0])))
    pids = list(live_workers().values())
    memory, cpu_seconds = 0, 0.
    for pid in pids:
        try:
            p = psutil.Process(pid)
            memory += p.memory_info().rss
            cpu_seconds += sum(p.cpu_times()[:2])
        except psutil.NoSuchProcess:
            pass
    return dict(utc=utc(), gpu=cards, worker_rss_sum_gib=memory / 2**30, worker_cpu_seconds=cpu_seconds,
                host_available_gib=psutil.virtual_memory().available / 2**30, system_cpu_percent=psutil.cpu_percent(),
                active_workers=len(pids), gpu_memory_scope="whole device including unrelated jobs")

def schedule(workers_per_gpu=4):
    assert 1 <= workers_per_gpu <= 4
    if MANIFEST.exists() and read_json(MANIFEST)["run_status"] == "COMPLETE":
        validate_complete()
        print("ALREADY_COMPLETE 42 tasks / 420 folds", flush=True)
        return
    prepare()
    handle = acquire_lock(STATE / "scheduler.lock")
    m = read_json(MANIFEST)
    if not m["preflight"] or m["preflight"]["status"] != "PASS":
        with open(LOGS / "preflight.log", "a") as output:
            subprocess.run([sys.executable, str(EXP / "run.py"), "preflight"], cwd=EXP, env=worker_env(0), stdout=output, stderr=subprocess.STDOUT, check=True)
    m = read_json(MANIFEST)
    assert m["preflight"]["status"] == "PASS"
    m.update(run_status="RUNNING", workers_per_gpu=workers_per_gpu)
    m.setdefault("started_unix", time.time())
    m.setdefault("started_utc", utc())
    m["launch_resources"] = resources()
    save_json(MANIFEST, m)
    failures = dict(m.get("failures_and_restarts", {}))
    active = {task:dict(pid=pid, process=None, gpu=m["tasks"][task]["physical_gpu"]) for task, pid in live_workers().items()}
    limits = {0:workers_per_gpu, 1:workers_per_gpu}
    last_refresh, last_resource = 0., 0.
    print("SCHEDULER_START " + utc(), flush=True)
    while True:
        for task, info in list(active.items()):
            proc = info["process"]
            exit_code = proc.poll() if proc else (None if Path("/proc", str(info["pid"])).exists() else 0)
            if exit_code is None:
                continue
            if "log" in info:
                info["log"].close()
            del active[task]
            if exit_code != 0:
                failures[task] = failures.get(task, 0) + 1
                m = read_json(MANIFEST)
                m["failures_and_restarts"] = failures
                save_json(MANIFEST, m)
                print(f"RESTART {task} exit={exit_code} from atomic checkpoint", flush=True)
                if failures[task] >= 3:
                    raise RuntimeError("Repeated task failure: " + task)
        done = {t["task"] for t in task_results() if t["status"] == "COMPLETE"}
        for gpu in (0, 1):
            for config, seed, assigned in TASKS:
                task = f"{config}_seed{seed}"
                if assigned != gpu or task in done or task in active:
                    continue
                if sum(i["gpu"] == gpu for i in active.values()) >= limits[gpu]:
                    break
                output = open(LOGS / (task + ".log"), "a")
                output.write("\nATTEMPT_START " + utc() + "\n")
                output.flush()
                proc = subprocess.Popen([sys.executable, str(EXP / "run.py"), "--worker", config, str(seed)],
                                        cwd=EXP, env=worker_env(gpu), stdout=output, stderr=subprocess.STDOUT)
                active[task] = dict(pid=proc.pid, process=proc, gpu=gpu, log=output)
                print(f"LAUNCH {task} GPU{gpu} pid={proc.pid}", flush=True)
        if time.monotonic() - last_refresh > 20 or not active:
            m = refresh()
            if time.monotonic() - last_resource >= 60:
                sample = resources()
                peaks = m.setdefault("runtime_resource_peaks", dict(worker_rss_sum_gib=0., gpu0_mib=0, gpu1_mib=0))
                peaks["worker_rss_sum_gib"] = max(peaks["worker_rss_sum_gib"], sample["worker_rss_sum_gib"])
                for card in sample["gpu"]:
                    gpu = card["index"]
                    peaks[f"gpu{gpu}_mib"] = max(peaks[f"gpu{gpu}_mib"], card["used_mib"])
                    limit = 1 if card["used_mib"] > 85000 or sample["host_available_gib"] < 20 else workers_per_gpu
                    if limits[gpu] != limit:
                        m.setdefault("resource_throttling", []).append(dict(utc=utc(), gpu=gpu, new_limit=limit))
                        limits[gpu] = limit
                save_json(MANIFEST, m)
                print("RESOURCE " + json.dumps(sample), flush=True)
                last_resource = time.monotonic()
            print(f"PROGRESS tasks={len(done)}/42 folds={m['completed_folds']}/420 active={len(active)}", flush=True)
            last_refresh = time.monotonic()
        if len(done) == 42 and not active:
            break
        time.sleep(2)
    m = refresh()
    m.update(run_status="TRAINING_COMPLETE", training_completed_utc=utc(), training_wall_seconds=time.time() - m["started_unix"], failures_and_restarts=failures)
    save_json(MANIFEST, m)
    validate_complete()
    report()
    append_handoff()
    assert not live_workers()
    integrity(full=True)
    handle.close()
    shutil.rmtree(STATE)
    m = read_json(MANIFEST)
    m.update(run_status="COMPLETE", completed_utc=utc(), temporary_state_cleaned=True, no_training_processes=True,
             handoff_sha256=sha(ROOT / "handoff.md"), report_sha256=sha(RESULTS / "report.md"),
             results_sha256={n: sha(RESULTS / (n + ".csv")) for n in ["fold_metrics", "diagnostics", "counterfactual"]})
    save_json(MANIFEST, m)
    print("COMPLETE 42/42 tasks 420/420 folds 420 diagnostics 330 counterfactual; temporary state cleaned", flush=True)

def summary(rows):
    import numpy as np
    result = {}
    configs = sorted(set(r["config"] for r in rows), key=lambda v: (v[0], int(v[1:])))
    for config in configs:
        result[config] = {}
        for key in METRICS:
            seeds = [float(np.mean([float(r[key]) for r in rows if r["config"] == config and int(r["seed"]) == seed])) for seed in range(3)]
            result[config][key] = dict(mean=float(np.mean(seeds)), sd=float(np.std(seeds)), seeds=seeds)
    return result

def paired(rows, left, right, key):
    import numpy as np
    lookup = {(r["config"], int(r["seed"]), int(r["fold"])):float(r[key]) for r in rows}
    differences = np.array([[lookup[left, seed, fold] - lookup[right, seed, fold] for fold in range(1, 11)] for seed in range(3)])
    return dict(mean=float(differences.mean()), seeds=differences.mean(1).tolist(),
                wins=int((differences > 1e-12).sum()), ties=int((np.abs(differences) <= 1e-12).sum()), losses=int((differences < -1e-12).sum()))

def mean_field(rows, key):
    values = [float(r[key]) for r in rows if r.get(key, "") != ""]
    return sum(values) / len(values) if values else 0.

def history_e9():
    folder = next((ROOT / "experiment").glob("experiment9_*"))
    return [dict(r, ba=r["balanced_accuracy"], specificity_td=r["specificity_asd"])
            for r in read_csv(folder / "results/fold_metrics.csv") if r["config"] == "E9"]

# Audit conclusions were frozen before implementing the branch; no performance pilot.
AUDIT = {
 'source': 'GFR在四模态各建受试者图；i=0赋值而i=1覆盖combined_feat，i=2/3累加。原标签先将ASD转-1、held-out置0来生成A，但A仅用于未反传的ALoss。simple-2反传原加权NLL+四图正则均值。保留这一实际行为，不修补论文/源码差异。',
 'pln': '原InputLayer/CAM→Encoder→FeedForward输出871×4×36；新增分支安装于原my_weight_init及两个Adam之后，注入最后FeedForward之后、SPGS之前。P0八句冻结RN50文本为8×1024，拼接2×4096→Linear4096/128/9 ReLU→cosine→softmax/max，产生871×4×4标量权重并乘潜在Token。global平均池化、local通道softmax、RES和log-softmax原分类器不改。4×4×9为潜在Token，不能解释为ROI。',
 'paper': '阅读本地SPromptGL十一页PDF的Fig.1文字、Section2.1/2.2、Eq1–7及ABIDE消融。CAM86.48→GFR91.73→RES92.08→SPGS92.30，主要增量是GFR。论文图加权和与实际覆盖、类别图项与源码ALoss不反传存在差异；本轮沿用实际历史脚本。',
 'history_9_14': '已阅读9/10/14补丁及报告、11映射表/审计、12/13真实模型。9 E9 ACC91.8826/BA91.6915/ProbAUC93.8227，疾病/P3概率高度相关。10真ASD相对ADHD/中性只微小且seed混合；不证明疾病必要性。12真实ROI文本在不同内层选模协议不如错配/身份，投影趋同。13患者FC状态文字未稳定超过数值/错配。14层级文本仍投影趋同。',
 'history_15_18': '已阅读15/16模型、Prompt和反事实；17 DS3A实现及18分支/独立runner。15 raw-FC decoder旁路与饱和门控使文本变化不传到预测，28/30折关键文本反事实逐值相同；16修复通路后B5残差32.33%、ACC/BA/ProbAUC均低于P0；17类原型cos约0.9997/0.9986，真实文本反事实零翻转；18保护FC Shared判别性但年龄性别OOF约49.78%BA、FC84.23%，选择代理没有稳定净收益。停止同一年龄性别对齐路线；不将梯度、敏感性或Probe当医学创新。',
 'data': 'PHENO含26年龄分箱/20站点/2性别，ANAT6与FUNC10为QC，func_gsr是ghost-to-signal ratio。FC来自Nilearn0.7.1默认LedoitWolf相关矩阵再Fisher-z，非直接Pearson、非生理兴奋/抑制。全部原320维保留；可信252连接仅用于新辅助分支，4条未知ROI端点连接仍在原主干。',
 'identifiability': '预训练前数学风险：对满行秩固定Q(4×1024)，自由Wkey(1024×32)令QW可表达任意4×32有效查询；同几何正交旋转、整体符号及域行置换均可被Wkey重新参数化吸收。因此C4/C8/C9/C11容量相同，优化轨迹差异不能单独证明医学先验。C7重复查询为有意rank1对照，C1零查询为有意rank0；不可为满足区分性检查而改写它们。',
 'novelty': 'ROI语义、图文对齐、类别查询、局部cross-attention及对比监督都有先例。候选差异只能是可信连接范围、疾病差异查询、NULL拒绝和破坏性对照联合验证；NULL不是校准的不确定性，静态Prompt没有患者级临床真值。'
}
LITERATURE = {
 'SPromptGL':dict(url='https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf',read='本地全文上述章节/公式与Fig1图注',conclusion='保留原GFR/CAM/SPGS/RES和实际历史代码。'),
 'MGDR':dict(url='https://papers.miccai.org/miccai-2024/paper/2494_paper.pdf',read='官方可解析全文方法3.1–3.2及实验表',conclusion='多图、SVD共享/私有表示及私有对比约束已有；本轮不用这些模块。'),
 'BrainPrompt':dict(url='https://arxiv.org/abs/2504.16096',read='本地11页PDF方法与图1文字',conclusion='ROI文本加到消息传递、人口属性与疾病级Prompt已有；不能把ROI语义注入本身称新。'),
 'G-LaD':dict(url='https://proceedings.mlr.press/v308/gupta26a.html',read='官方摘要与论文书目信息，未声称读全文',conclusion='GIN去噪、Sinkhorn图文分布对齐、LLM推理；NeuroAI Workshop PMLR308，不等同AAAI主会。'),
 'SABER':dict(url='https://arxiv.org/abs/2607.01901',read='用户本地6页PDF方法Eq8–13等；作者出版列表核验ICME2026',conclusion='ROI/多尺度超图、Graph-CLS局部cross-attention、门控残差及疾病对比监督已有；作者出版列表https://hanxiangmin.github.io/publications/列为ICME2026。'),
 'Class-Aware Semantic Decoupling':dict(url='https://papers.miccai.org/miccai-2026/1073-Paper1452.html',read='官方摘要，未读可靠可解析全文',conclusion='类别感知共享/特异语义分解及融合已有，不能将类别查询概念作为首次。'),
 'LGDEA':dict(url='https://arxiv.org/html/2602.07540v1',read='原始作者预印本方法III-A/B及图注；作者出版列表核验ICML2026',conclusion='从真实放射报告构建证据空间、可学习诊断原型与病灶查询；与本轮四域静态弱先验不同。作者出版列表https://zjuchenlong.github.io/Publications/列为ICML2026，阅读方法依据预印本版本。')
}
MEDICAL = [
 dict(domain='D1',url='https://pubmed.ncbi.nlm.nih.gov/22791801/',source='Gotts et al. 2012 Fractionation of social brain circuits',boundary='群体社会网络连通性关联；不推出全部颞顶连接同向变化或个体病灶。'),
 dict(domain='D2',url='https://pmc.ncbi.nlm.nih.gov/articles/PMC4500121/',source='Just et al. 2007 frontal-parietal executive underconnectivity',boundary='任务fMRI研究，仅支持宽泛候选先验；不可直接变成静息FC固定下降标签。'),
 dict(domain='D3',url='https://pmc.ncbi.nlm.nih.gov/articles/PMC3673456/',source='Nair et al. 2013 thalamocortical connectivity',boundary='丘脑皮层研究支持候选范围，异质方向不设hyper/hypo真值。'),
 dict(domain='D4',url='https://pubmed.ncbi.nlm.nih.gov/23803651/',source='Uddin et al. 2013 salience network connectivity',boundary='岛叶/前扣带线索不验证整个扣带/岛叶域为单一salience网络；当前HO锚点刻意粗粒度。')
]
ANCHORS = [
 ['Superior Temporal Gyrus','Middle Temporal Gyrus','Angular Gyrus','Supramarginal Gyrus','Temporal Pole'],
 ['Superior Frontal Gyrus','Middle Frontal Gyrus','Inferior Frontal Gyrus','Frontal Pole','Superior Parietal Lobule'],
 ['Thalamus','Precentral Gyrus','Postcentral Gyrus',"Heschl's Gyrus",'Central Opercular Cortex'],
 ['Cingulate Gyrus','Insular Cortex','Frontal Operculum Cortex']]
REAL_TEXT = [
 ['Autism spectrum disorder has been investigated in relation to heterogeneous functional connectivity differences involving temporal and temporoparietal systems associated with social information processing.', 'In typical development, temporal and temporoparietal functional systems contribute to social information processing, with substantial individual variation in connectivity.'],
 ['Research on autism has examined variable frontoparietal connectivity patterns potentially related to executive control and attention regulation.', 'Typical development includes variable frontoparietal connectivity supporting executive control and attention regulation.'],
 ['Autism research has investigated heterogeneous thalamocortical and sensorimotor connectivity patterns in relation to sensory processing differences.', 'In typical development, thalamocortical and sensorimotor connectivity contributes to sensory processing with individual variability.'],
 ['Autism studies have explored differences in cingulate and insular connectivity potentially associated with salience and internal signal processing.', 'Cingulate and insular systems in typical development support processing of salient and internal information with individual variation.']]
NEUTRAL_TEXT = [
 ['A technical description organizes temporal and temporoparietal connectivity into a matrix of measurements.', 'A technical reference lists temporal and temporoparietal connectivity measurements in a standardized table.'],
 ['A computational procedure aggregates frontoparietal connectivity measurements into a feature vector.', 'A computational reference stores frontoparietal connectivity measurements in a numerical array.'],
 ['A measurement procedure summarizes thalamocortical and sensorimotor connectivity across recorded signals.', 'A measurement reference arranges thalamocortical and sensorimotor connectivity in an indexed data table.'],
 ['A technical workflow groups cingulate and insular connectivity measurements by their indices.', 'A technical reference describes cingulate and insular connectivity measurements using standardized numerical fields.']]
UNRELATED_TEXT = [
 ['Cardiac disease research examines heterogeneous electrical conduction patterns associated with heart rhythm disturbances.', 'Typical cardiac physiology includes variable electrical conduction patterns that coordinate heart rhythm.'],
 ['Kidney disease research examines heterogeneous filtration patterns associated with altered renal function.', 'Typical renal physiology includes variable filtration patterns that maintain fluid balance.'],
 ['Pulmonary disease research examines heterogeneous ventilation patterns associated with impaired gas exchange.', 'Typical pulmonary physiology includes variable ventilation patterns that support gas exchange.'],
 ['Metabolic disease research examines heterogeneous glucose regulation patterns associated with endocrine dysfunction.', 'Typical metabolic physiology includes variable glucose regulation patterns that support energy balance.']]


def build_mapping():
    import numpy as np
    folder = ROOT / 'experiment/experiment11_roi_mapping_recovery/results'
    roi = {int(r['roi_index_zero_based']):r for r in read_csv(folder/'roi_111_labels.csv')}
    fc = read_csv(folder/'roi_256_named_mapping.csv')
    assert len(roi)==111 and len(fc)==256
    upper = np.triu_indices(111,1)
    rows, rejected, endpoints = [], [], set()
    for expected,r in enumerate(fc):
        e,i,j = (int(r[k]) for k in ('correlation_feature_index','roi_i_zero_based','roi_j_zero_based'))
        assert e==expected and (i,j)==tuple(v[int(r['original_connection_index'])] for v in upper)
        assert r['verification_status']=='VERIFIED_NUMERIC_UNIQUE' and r['unique_match']=='True'
        assert int(r['n_subjects_verified'])==871 and float(r['max_absolute_error'])<1e-10
        assert int(r['roi_i_atlas_value'])==int(roi[i]['atlas_integer_value'])
        assert int(r['roi_j_atlas_value'])==int(roi[j]['atlas_integer_value'])
        valid=all(roi[v]['name_verified']=='True' and roi[v]['column_order_verified']=='True' for v in (i,j))
        assert valid==(r['usable_for_roi_semantic_prompt']=='True')
        if not valid:
            rejected.append(e); continue
        names=[roi[v]['roi_name'] for v in (i,j)]
        # Match anchor at the beginning after stripping hemisphere: excludes Paracingulate.
        clean=[n.removeprefix('Left ').removeprefix('Right ').casefold() for n in names]
        flags=[int(any(n.startswith(a.casefold()) for n in clean for a in anchors)) for anchors in ANCHORS]
        row=dict(correlation_feature_index=e, original_connection_index=int(r['original_connection_index']),
                 roi_i_zero_based=i,roi_j_zero_based=j,roi_i_name=names[0],roi_j_name=names[1],verified=True)
        row.update({f'D{k+1}':v for k,v in enumerate(flags)})
        rows.append(row); endpoints.update((i,j))
    assert rejected==[195,225,232,236] and len(rows)==252 and len(endpoints)==110
    mask=np.array([[r[f'D{k+1}'] for r in rows] for k in range(4)],dtype=bool)
    assert mask.any(1).all()
    g=np.random.default_rng(191010)
    perm=g.permutation(252)
    while (perm==np.arange(252)).any(): perm=g.permutation(252)
    mp=mask[:,perm]
    assert np.array_equal(mask.sum(1),mp.sum(1))
    assert np.array_equal(mask.astype(int)@mask.T.astype(int),mp.astype(int)@mp.T.astype(int))
    degree=np.zeros(111,dtype=int)
    for r in rows: degree[r['roi_i_zero_based']]+=1;degree[r['roi_j_zero_based']]+=1
    audit=dict(valid_connections=252,excluded_unknown_features=rejected,unknown_roi=82,
               trusted_roi_count=len(endpoints),candidate_counts=mask.sum(1).tolist(),
               domain_overlap=(mask.astype(int)@mask.T.astype(int)).tolist(),covered_edge_count=int(mask.any(0).sum()),
               edges_in_no_domain=int((~mask.any(0)).sum()),roi_degrees=degree.tolist(),anchors=ANCHORS,
               anatomy_derangement_seed=191010,anatomy_derangement_sha256=canonical(perm.tolist()),
               mapping_source='exp11 numeric unique 871/871; names joined from roi_111_labels.csv, NOT UNKNOWN legacy columns')
    return rows,dict(mask=mask,mask_permuted=mp,edge_permutation=perm,
                     valid_indices=np.array([r['correlation_feature_index'] for r in rows],dtype=np.int64)),audit


def encode_semantics(prompt):
    import numpy as np
    import torch
    from torch.nn import functional as F
    sys.path.insert(0,str(ROOT/'CLIP'))
    import clip
    texts=[t for b in ('REAL','NEUTRAL','UNRELATED','CLASS_ONLY') for pair in prompt['banks'][b] for t in pair]
    tokens=clip.tokenize(texts,truncate=False)
    encoder,_=clip.load('RN50',device='cuda:0',jit=False)
    encoder.eval()
    for p in encoder.parameters():p.requires_grad_(False)
    with torch.no_grad(): emb=encoder.encode_text(tokens.cuda()).float().cpu()
    del encoder
    out, audit={},{}
    for i,b in enumerate(('REAL','NEUTRAL','UNRELATED','CLASS_ONLY')):
        raw=emb[i*8:(i+1)*8].reshape(4,2,1024)
        prototypes=F.normalize(raw,dim=-1)
        diff=prototypes[:,0]-prototypes[:,1]
        q=F.normalize(diff,dim=-1)
        assert (diff.norm(dim=-1)>1e-6).all()
        out.update({f'embeddings_{b}':raw.numpy(),f'prototypes_{b}':prototypes.numpy(),f'query_{b}':q.numpy(),f'token_lengths_{b}':tokens[i*8:(i+1)*8].count_nonzero(1).numpy()})
        audit[b]=dict(raw_norms=raw.norm(dim=-1).tolist(),difference_norms=diff.norm(dim=-1).tolist(),
                      query_gram=(q@q.T).tolist(),query_rank=int(torch.linalg.matrix_rank(q)),
                      token_lengths=out[f'token_lengths_{b}'].tolist())
        assert torch.isfinite(q).all() and tokens[i*8:(i+1)*8].count_nonzero(1).max()<=77
        if b!='CLASS_ONLY': assert (q@q.T)[~torch.eye(4,dtype=torch.bool)].max()<.99999
        else: assert torch.equal(q,q[:1].expand_as(q))
    # An isometry of the whole rank-four span into an orthogonal random subspace.
    q=out['query_REAL'].astype(np.float64)
    basis=np.linalg.qr(q.T,mode='reduced')[0]
    g=np.random.default_rng(191100)
    new=g.normal(size=(1024,4));new-=basis@(basis.T@new)
    new=np.linalg.qr(new,mode='reduced')[0]
    rotated=((q@basis)@new.T).astype(np.float32)
    gram_error=float(np.abs(rotated@rotated.T-q@q.T).max())
    direction_cos=(rotated*q).sum(1)
    assert gram_error<1e-6 and np.abs(direction_cos).max()<1e-5
    out['query_RANDOM_ROTATED']=rotated
    audit['RANDOM_ROTATED']=dict(gram_max_error=gram_error,cosine_to_real=direction_cos.tolist(),rank=int(np.linalg.matrix_rank(rotated)),seed=191100)
    return out,audit


def prepare():
    import numpy as np
    if MANIFEST.exists():
        m=integrity(full=True)
        if m['run_status']!='COMPLETE':make_sources()
        print('PREPARE_VERIFIED',flush=True);return m
    LOGS.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True);STATE.mkdir(parents=True,exist_ok=True)
    x,y,folds,audit=data()
    protected={str(p.relative_to(ROOT)):sha(p) for folder in historical_folders() for p in sorted(folder.rglob('*')) if p.is_file()}
    source_names=CORE+['README.md','data/ABIDE/modal_feat_dict.npy','data/ABIDE/processed_standard_data.csv','data/ABIDE/data_processing.ipynb','CLIP/clip/clip.py','CLIP/clip/model.py','CLIP/clip/simple_tokenizer.py']
    sources={str(ROOT/n):sha(ROOT/n) for n in source_names}
    prompt=dict(class_order=['ASD','TD'],domain_order=['D1','D2','D3','D4'],banks=dict(REAL=REAL_TEXT,NEUTRAL=NEUTRAL_TEXT,UNRELATED=UNRELATED_TEXT,CLASS_ONLY=[['An individual diagnosed with autism spectrum disorder.','An individual with typical development.'] for _ in range(4)]),
                real_candidate_changes=0,real_text_status='Provided cautious candidate wording frozen unchanged; no outcome-based edits.',medical_sources=MEDICAL,
                original_p0=['Nromal brain volume','Uniform cortical thickness','Symmetrical brain structure','Normal ventricular morphology','Altered brain volume','Abnormal functional connectivity','White matter integrity issues','Different activation patterns'])
    save_json(EXP/'prompts.json',prompt)
    mapping,cache,mapping_audit=build_mapping()
    semantic,semantic_audit=encode_semantics(prompt);cache.update(semantic)
    buf=io.BytesIO();np.savez_compressed(buf,**cache);atomic(EXP/'semantic_cache.npz',buf.getvalue())
    save_csv(EXP/'domain_mapping.csv',mapping)
    patch=''.join(''.join(difflib.unified_diff((ROOT/n).read_text().splitlines(True),t.splitlines(True),fromfile='a/'+n,tofile='b/'+n)) for n,t in patched_sources().items())
    atomic(EXP/'integration_patch.diff',patch)
    params={c:0 if c=='C0' else sum(p.numel() for p in EvidenceAlignment(c,cache,0,folds[0][0],np.arange(871)).parameters()) for c in CONFIGS}
    papers={str(p.relative_to(ROOT)):sha(p) for pattern in ['*SPromptGL*.pdf','*BrainPrompt*.pdf','*SABER*.pdf'] for p in ROOT.glob(pattern)}
    m=dict(experiment='AG-CDEA',schema_version=1,run_status='PREPARED',prepared_utc=utc(),root=str(ROOT),experiment_directory=str(EXP),temporary_state=str(STATE),
           protocol=PROTOCOL,configurations=CONFIGURATIONS,audit=AUDIT,literature_audit=LITERATURE,medical_sources=MEDICAL,data_audit=audit,mapping_audit=mapping_audit,semantic_audit=semantic_audit,
           additional_parameters=params,source_data_sha256=sources,protected_history_sha256=protected,paper_sha256=papers,
           frozen_file_sha256={n:sha(EXP/n) for n in ['prompts.json','domain_mapping.csv','semantic_cache.npz','integration_patch.diff']},code_sha256={n:sha(EXP/n) for n in ['run.py','cdea.py']},
           fixed_design=dict(edge_dim=32,edge_id_embedding=True,fusion=[136,64,36],temperature=TEMPERATURE,lambda_evidence=LAMBDA_EVIDENCE,new_weight_decay=0.,residual_limit=RESIDUAL_LIMIT,
             auxiliary_init_seed='190000+task seed; fork_rng after my_weight_init and both Adams',domain_derangement=list(DOMAIN_PERMUTATION),cf_shuffle_seed='19300+fold; test-only derangement',learned_query_seed='191200+seed'),
           scientific_criteria=dict(primary_metrics=['acc','ba','probability_auc'],stable='each of three seed differences strictly positive; no post-hoc tolerance',pairs=PAIRS,
             semantic_controls=['C5','C6','C7','C8','C9','C11','C12'],hypothesis_frozen_before_training=True,query_reparameterization_limitation=AUDIT['identifiability']),
           split_sha256=canonical([dict(train=tr.tolist(),test=te.tolist()) for tr,te in folds]),instruction_prompt_sha256=sha(ROOT/'prompt'),
           clip_weight_sha256=sha(Path.home()/'.cache/clip/RN50.pt'),git_head_at_prepare=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
           handoff_prefix_sha256=sha(ROOT/'handoff.md'),handoff_prefix_bytes=(ROOT/'handoff.md').stat().st_size,
           preflight=None,completed_folds=0,completed_tasks=0,workers_per_gpu=4,tasks={f'{c}_seed{s}':dict(config=c,seed=s,physical_gpu=g,status='PENDING',folds=0) for c,s,g in TASKS})
    save_json(MANIFEST,m);make_sources();write_readme(m)
    print('PREPARED',json.dumps(dict(params=params,candidates=mapping_audit['candidate_counts'],gram_error=semantic_audit['RANDOM_ROTATED']['gram_max_error'])),flush=True)
    return m


def integrity(full=False):
    m=read_json(MANIFEST)
    for p,h in m['source_data_sha256'].items():assert sha(p)==h,('source/data changed',p)
    for field in ('code_sha256','frozen_file_sha256'):
        for n,h in m[field].items():assert sha(EXP/n)==h,('frozen experiment changed',n)
    if full:
        current={str(p.relative_to(ROOT)) for folder in historical_folders() for p in folder.rglob('*') if p.is_file()}
        assert current==set(m['protected_history_sha256'])
        for p,h in m['protected_history_sha256'].items():assert sha(ROOT/p)==h,('history changed',p)
        for p,h in m['paper_sha256'].items():assert sha(ROOT/p)==h
        assert sha(Path.home()/'.cache/clip/RN50.pt')==m['clip_weight_sha256']
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==m['git_head_at_prepare']
    return m


def signature():
    m=read_json(MANIFEST)
    return canonical({k:m[k] for k in ('source_data_sha256','code_sha256','frozen_file_sha256','protocol','configurations','fixed_design','split_sha256','clip_weight_sha256')})


def task_signature(config, seed):
    return canonical(dict(experiment_signature=signature(), config=config, seed=int(seed)))


def context():
    global _CTX
    if _CTX is None:
        import numpy as np
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        make_sources();sys.path.insert(0,str(SOURCES));sys.modules['cdea_runtime']=sys.modules[__name__]
        import network,model
        x,y,folds,audit=data()
        with np.load(EXP/'semantic_cache.npz',allow_pickle=False) as packed:cache={k:packed[k].copy() for k in packed.files}
        _CTX=dict(torch=torch,network=network,model=model,x=x,y=y,folds=folds,audit=audit,cache=cache)
    return _CTX


def install_branch(a):
    import numpy as np
    c=context();a._cdea_raw=a.dataLayer(a.feat)[1]
    if c['current_config']=='C0':return
    fold=c['current_fold'];tr,te=c['folds'][fold-1]
    g=np.random.default_rng(19300+fold);p=g.permutation(te)
    while (p==te).any():p=g.permutation(te)
    permutation=np.arange(871);permutation[te]=p
    assert np.array_equal(permutation[tr],tr) and set(p)==set(te)
    with c['torch'].random.fork_rng(devices=[0]):
        c['torch'].manual_seed(190000+int(a.hyperpm.seed))
        a.ModalFusion.cdea=EvidenceAlignment(c['current_config'],c['cache'],a.hyperpm.seed,tr,permutation).cuda()
    a.optimizer_MF.add_param_group(dict(params=a.ModalFusion.cdea.parameters(),lr=.0038,weight_decay=0.))


def nonzero_parameters(a,key_only=False):
    b=branch(a)
    if b is None:return 0
    pp=b.key.parameters() if key_only else b.parameters()
    return sum(int(p.grad.count_nonzero()) for p in pp if p.grad is not None)


def snapshot(a):
    b=branch(a)
    return {k:v.detach().clone() for k,v in b.last.items() if hasattr(v,'detach')} if b else {}

def objectives(a):
    c=context();a.MessagePassing0.eval();a.ModalFusion.eval()
    n,parts=a.dataLayer(a.feat)
    hidden,adj,_=a.MessagePassing0(parts,n,a.targ,a.trn_idx)
    out,_,_=a.ModalFusion(hidden,a._cdea_raw) if branch(a) else a.ModalFusion(hidden)
    cls=c['model'].ClsLoss(out,a.targ,a.trn_idx,a.weight)
    graph=sum(c['model'].GraphConstructLoss(parts[i],adj[i],1.,.5,0.) for i in range(n))/4
    b=branch(a);evidence=cls.new_zeros(())
    if b:evidence=b.losses(a.targ)
    return dict(classification=cls,original_graph=graph,evidence=evidence)


def correlation(x,y):
    import numpy as np
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    # A constant has no defined correlation; record 0 with a separate SD below.
    return float(np.corrcoef(x,y)[0,1]) if x.std()>1e-10 and y.std()>1e-10 else 0.


def geometry(q):
    import torch
    from torch.nn import functional as F
    q=q.detach().double()
    gram=F.normalize(q,dim=-1)@F.normalize(q,dim=-1).T
    singular=torch.linalg.svdvals(q-q.mean(0,keepdim=True));p=singular/singular.sum().clamp_min(1e-20)
    rank=float((-(p*p.clamp_min(1e-20).log()).sum()).exp()) if singular.sum()>1e-12 else 0.
    off=gram[~torch.eye(4,dtype=torch.bool,device=q.device)]
    return dict(cosine_mean=float(off.mean()),cosine_max=float(off.max()),centered_effective_rank=rank,
                gram=json.dumps(gram.cpu().tolist(),separators=(',',':')))


def diagnostics(a,config,seed,fold,real,tracker,initial_norms):
    import numpy as np
    import torch
    from sklearn.metrics import balanced_accuracy_score,roc_auc_score
    row=dict(config=config,seed=seed,fold=fold,evidence_enabled=bool(DESIGNS[config].evidence and config!='C0'),
             residual_relative_mean=0.,residual_relative_max=0.,noncorr_delta_max=0.)
    losses=objectives(a)
    row.update({'loss_'+k:float(v.detach()) for k,v in losses.items()})
    a.optimizer_MP0.zero_grad();a.optimizer_MF.zero_grad();sum(losses.values()).backward()
    row.update({'best_checkpoint_gradient_l2_'+k:v for k,v in gradients(a).items()})
    row.update({'sampled_gradient_max_'+k:v for k,v in tracker.items()})
    if branch(a) is None:return row
    b=branch(a);forward(a);ll=snapshot(a)
    for k,v in b.effective_parameters().items():
        if isinstance(v,int):row['parameters_'+k]=v
    row['parameters_sampled_nonzero_gradient']=int(tracker['nonzero_parameters'])
    row['parameters_sampled_nonzero_key_gradient']=int(tracker['nonzero_key_parameters'])
    for k,v in parameter_norms(a).items():
        row['parameter_norm_'+k]=v;row['parameter_norm_ratio_'+k]=v/(initial_norms[k]+1e-12)
    row['alpha']=float(b.alpha.detach());row['alpha_sign']=int(np.sign(row['alpha']))
    row['signed_residual_scale']=.1*math.tanh(row['alpha'])
    delta,base=ll['delta'],ll['base']
    relative=delta[:,3].norm(dim=-1)/(base[:,3].norm(dim=-1)+1e-8)
    b.losses(a.targ)  # forward() replaced last; record raw BCE also in disabled-loss controls.
    row.update(residual_relative_mean=float(relative.mean()),residual_relative_max=float(relative.max()),
               residual_nonzero_fraction=float((delta[:,3].abs().sum(-1)>1e-12).float().mean()),
               noncorr_delta_max=float(delta[:,:3].abs().max()),predictions_finite=bool(torch.isfinite(real).all()),
               evidence_raw_bce=float(b.last['evidence_loss'].detach()))
    for prefix,q in [('query',ll['q']),('effective_query',ll['q_eff'])]:
        row.update({prefix+'_'+k:v for k,v in geometry(q).items()})
    tr,te=a.trn_idx,a.tst_idx
    u=ll['u_global'].cpu().numpy();target=context()['y']==0
    row.update(evidence_train_ba=float(balanced_accuracy_score(target[tr],u[tr]>=0)),
               evidence_train_probability_auc=float(roc_auc_score(target[tr],u[tr])),
               evidence_score_train_sd=float(u[tr].std()),evidence_score_test_sd=float(u[te].std()))
    att=ll['attention'];pn=ll['p_null'];score=ll['score'];z=ll['z']
    # Attention diagnostics include NULL and conditional edge-only distributions.
    for k in range(4):
        prefix=f'D{k+1}_';p=pn[:,k];v=att[:,k];conditional=v/v.sum(-1,keepdim=True).clamp_min(1e-12)
        ent=-(v*v.clamp_min(1e-30).log()).sum(-1)-p*p.clamp_min(1e-30).log()
        edge_ent=-(conditional*conditional.clamp_min(1e-30).log()).sum(-1)
        vals=dict(candidates=int(ll['mask'][k].sum()),null_mean=float(p.mean()),null_sd=float(p.std(unbiased=False)),
                  null_p05=float(p.quantile(.05)),null_p50=float(p.quantile(.5)),null_p95=float(p.quantile(.95)),
                  null_reject95_fraction=float((p>.95).float().mean()),attention_entropy=float(ent.mean()),
                  conditional_edge_entropy=float(edge_ent.mean()),effective_edges=float(edge_ent.exp().mean()),
                  patient_attention_sd=float(v.std(0,unbiased=False)[ll['mask'][k]].mean()),
                  score_mean=float(ll['u'][:,k].mean()),score_sd=float(ll['u'][:,k].std(unbiased=False)),
                  positive_fc_attention_mass=float((v*(z>0)).sum(-1).mean()),negative_fc_attention_mass=float((v*(z<0)).sum(-1).mean()),
                  positive_fc_score_contribution=float((v*score[:,k]*(z>0)).sum(-1).mean()),
                  negative_fc_score_contribution=float((v*score[:,k]*(z<0)).sum(-1).mean()))
        for split,idx in [('train',tr),('test',te)]:
            for label,flag in [('asd',True),('td',False)]:
                chosen=idx[target[idx]==flag];vals[split+'_'+label+'_score_mean']=float(ll['u'][chosen,k].mean())
        row.update({prefix+n:value for n,value in vals.items()})
        mean=v[te].mean(0);top=mean.topk(5).indices.cpu().numpy()
        row[prefix+'top_edges']=json.dumps([dict(feature=int(b.valid_indices[e]),mean_attention=float(mean[e])) for e in top],separators=(',',':'))
    row['within_patient_domain_attention_mean_abs_difference']=float(torch.stack([(att[:,i]-att[:,j]).abs().mean() for i in range(4) for j in range(i)]).mean())
    actual_prob=real.exp()[:,0].cpu().numpy()
    try:
        b.condition='RESIDUAL_OFF';off=forward(a);off_prob=off.exp()[:,0].cpu().numpy()
    finally:b.condition='REAL'
    assert torch.equal(forward(a),real)
    change=actual_prob-off_prob
    row.update(mean_abs_asd_probability_change_vs_off=float(np.abs(change[te]).mean()),
               prediction_flip_rate_vs_off=float((real[te].argmax(1)!=off[te].argmax(1)).double().mean()),
               evidence_main_probability_correlation=correlation(u[te],actual_prob[te]),
               evidence_off_probability_correlation=correlation(u[te],off_prob[te]),
               evidence_main_change_correlation=correlation(u[te],change[te]))
    x=context()['x'];site_cols=context()['audit']['site_indices'];site_names=context()['audit']['modality_columns']['PHENO']
    site=x[:,site_cols].argmax(1)
    # Only site aggregates, no patient-level artifact or fitted site estimator.
    row['site_descriptive']=json.dumps([dict(site=site_names[site_cols[s]],n=int((site==s).sum()),
          residual_mean=float(relative[site==s].mean()),null_mean=float(pn[site==s].mean())) for s in range(20) if (site==s).any()],separators=(',',':'))
    assert all(math.isfinite(v) for v in row.values() if isinstance(v,float))
    return row


def counterfactual(a,seed,fold,real):
    import torch
    b=branch(a);assert a.ModalFusion.training is False and b.condition=='REAL'
    real_out=forward(a);assert torch.equal(real,real_out)
    ll=snapshot(a);idx=a.tst_idx;baseline=metrics(real,idx);logits=a._raw_logits.clone()
    parameter_hash=(weight_hash(a),weight_hash(a,True));random_hash=rng_hash(rng());records=[]
    try:
        for condition in CONDITIONS:
            b.condition=condition;out=forward(a);now=snapshot(a)
            assert torch.equal(now['base'],ll['base'])
            v=metrics(out,idx)
            r=dict(config='C4',seed=seed,fold=fold,condition=condition,**v,
                 prediction_flip_rate=float((out[idx].argmax(1)!=real[idx].argmax(1)).double().mean()),
                 mean_abs_asd_probability_change=float((out.exp()[idx,0]-real.exp()[idx,0]).abs().mean()),
                 mean_abs_logit_change=float((a._raw_logits[idx]-logits[idx]).abs().mean()),
                 attention_mean_abs_change=float((now['attention'][idx]-ll['attention'][idx]).abs().mean()),
                 null_mean_abs_change=float((now['p_null'][idx]-ll['p_null'][idx]).abs().mean()),
                 residual_relative_mean_change=float(((now['delta'][idx,3]-ll['delta'][idx,3]).norm(dim=-1)/(ll['base'][idx,3].norm(dim=-1)+1e-8)).mean()),
                 same_checkpoint=True,base_tokens_exact=True,parameters_unchanged=True,retrained=False,test_labels_used_for_shuffle=False)
            r.update({key+'_difference_vs_real':v[key]-baseline[key] for key in METRICS})
            records.append(r)
    finally:b.condition='REAL'
    assert torch.equal(forward(a),real)
    assert parameter_hash==(weight_hash(a),weight_hash(a,True)) and rng_hash(rng())==random_hash
    return records


def preflight():
    import numpy as np
    import torch
    from torch.nn import functional as F
    m=integrity(full=True)
    if m.get('preflight') and m['preflight']['status']=='PASS':print('PREFLIGHT_ALREADY_PASS',flush=True);return
    c=context();checks=dict(status='RUNNING',started_utc=utc(),initializations={})
    for seed in range(3):
        seed_rng(seed);entry=rng();native=agent_for('C0',seed,1,original=True)
        native_init,native_rng,native_out=weight_hash(native),rng(),forward(native)
        common=None
        for config in CONFIGS:
            restore_rng(entry);c['network']._prompt_text_features=None
            a=agent_for(config,seed,1)
            assert weight_hash(a)==native_init and rng_hash(rng())==rng_hash(native_rng),(config,'legacy initialization/RNG')
            assert torch.equal(forward(a),native_out),(config,'zero-init forward')
            assert torch.equal(a.ModalFusion.prompt_text_features,native.ModalFusion.prompt_text_features)
            assert not a.ModalFusion.prompt_text_features.requires_grad
            count=optimizer_audit(a,config);b=branch(a)
            if b:
                assert b.alpha.count_nonzero()==0 and b.last['delta'].count_nonzero()==0
                h=hashlib.sha256()
                for name,p in b.named_parameters():
                    if name!='learned_queries':h.update(name.encode()+p.detach().cpu().numpy().tobytes())
                common=common or h.hexdigest();assert common==h.hexdigest()
                ll=b.last
                assert ll['fusion_input'].shape==(871,136) and ll['q_eff'].shape==(4,32)
                assert torch.allclose(ll['attention'].sum(-1)+ll['p_null'],torch.ones_like(ll['p_null']),atol=3e-7)
                assert torch.isfinite(ll['p_null']).all() and ll['p_null'].min()>=0 and ll['p_null'].max()<=1
                if config=='C13':assert ll['p_null'].count_nonzero()==0
                if config=='C1':
                    assert ll['q'].count_nonzero()==0 and torch.allclose(ll['p_null'],torch.full_like(ll['p_null'],.5),atol=2e-7)
                if config=='C12':assert torch.allclose(b.learned_queries.norm(dim=-1),torch.ones(4,device='cuda'),atol=2e-7)
                assert not b.queries.requires_grad
                expected_q=torch.as_tensor(c['cache']['query_REAL'],device='cuda')
                if config=='C8':assert torch.allclose(ll['q'],-expected_q,atol=2e-7)
                if config=='C9':assert torch.allclose(ll['q'],expected_q[list(DOMAIN_PERMUTATION)],atol=2e-7)
                if config=='C10':assert torch.equal(ll['mask'],b.mask_permuted)
                if config=='C3':assert ll['mask'].all()
                if config=='C7':assert torch.equal(ll['q'],ll['q'][:1].expand_as(ll['q']))
            checks['initializations'][f'{config}_seed{seed}']=dict(total_parameters=count,initial_trunk_sha256=native_init,common_branch_sha256=common or '',output_exact=True,cpu_cuda_rng_exact=True)
            del a
        restore_rng(native_rng);oracles=[]
        for _ in range(2):
            native.run_epoch('simple-2')
            oracles.append((model_state(native),rng_hash(rng()),forward(native),pickle.loads(pickle.dumps([native.optimizer_MP0.state_dict(),native.optimizer_MF.state_dict()]))))
        restore_rng(entry);c['network']._prompt_text_features=None;a=agent_for('C0',seed,1)
        for i in range(2):
            a.run_epoch('simple-2');oracle=oracles[i]
            assert_equal_tree(model_state(a),oracle[0]);assert rng_hash(rng())==oracle[1] and torch.equal(forward(a),oracle[2])
            assert_equal_tree(a.optimizer_MP0.state_dict(),oracle[3][0]);assert_equal_tree(a.optimizer_MF.state_dict(),oracle[3][1])
        del a,native,oracles
        print('PREFLIGHT_INIT_AND_P0_PARITY seed='+str(seed),flush=True)
    checks['p0_forward_two_adam_rng_exact']=True
    checks['two_step_gradients']={}
    for config in CONFIGS[1:]:
        seed_rng(0);a=agent_for(config,0,1);samples=[]
        for _ in range(2):
            a.run_epoch('simple-2');samples.append(gradients(a))
        assert samples[0]['alpha']>0,(config,'initial alpha disconnected')
        for key in branch(a).effective_parameters()['active_components']:
            assert max(v[key] for v in samples)>0,(config,key,'disconnected')
        if config=='C1':assert all(v['key']==0 for v in samples)
        checks['two_step_gradients'][config]=samples
        current=forward(a)
        diagnostics(a,config,0,1,current,dict(gradients(a),nonzero_parameters=nonzero_parameters(a),nonzero_key_parameters=nonzero_parameters(a,True)),parameter_norms(a))
        print('PREFLIGHT_DIAGNOSTIC_WRITER '+config,flush=True)
    del a
    # Value and derivative equivalence of the low-memory association (double precision).
    g=torch.Generator().manual_seed(19022)
    h=torch.randn(7,11,32,generator=g,dtype=torch.float64,requires_grad=True)
    w=torch.randn(1024,32,generator=g,dtype=torch.float64,requires_grad=True)
    q=torch.randn(4,1024,generator=g,dtype=torch.float64,requires_grad=True)
    explicit=torch.einsum('nef,kf->nke',h@w.T,q)/math.sqrt(32)
    efficient=torch.einsum('nef,kf->nke',h,q@w)/math.sqrt(32)
    ga=torch.autograd.grad(explicit.square().sum(),(h,w,q));gb=torch.autograd.grad(efficient.square().sum(),(h,w,q))
    assert torch.allclose(explicit,efficient,atol=1e-12,rtol=1e-12)
    assert all(torch.allclose(x,y,atol=1e-9,rtol=1e-11) for x,y in zip(ga,gb))
    checks['associative_score_and_gradients_exact_tolerance']=True
    checks['all_config_diagnostic_writers_finite']=True
    for config in ('C0','C4','C12'):
        seed_rng(1);a=agent_for(config,1,1)
        for _ in range(2):a.run_epoch('simple-2')
        path=STATE/'preflight-recovery.pt'
        save_checkpoint(path,dict(model=model_state(a),mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),rng=rng()))
        a.run_epoch('simple-2');expected,expected_rng,out=model_state(a),rng_hash(rng()),forward(a)
        saved=checkpoint(path);load_state(a,saved['model']);a.optimizer_MP0.load_state_dict(saved['mp']);a.optimizer_MF.load_state_dict(saved['mf']);restore_rng(saved['rng'])
        a.run_epoch('simple-2');assert_equal_tree(model_state(a),expected)
        assert rng_hash(rng())==expected_rng and torch.equal(forward(a),out);path.unlink()
    checks['full_checkpoint_third_step_exact']=True
    rows,cache,ma=build_mapping()
    for k in ('mask','mask_permuted','valid_indices','edge_permutation'):assert np.array_equal(cache[k],c['cache'][k])
    assert not set(cache['valid_indices']).intersection([195,225,232,236]) and np.array_equal(cache['mask'].sum(1),cache['mask_permuted'].sum(1))
    realq=c['cache']['query_REAL'];randomq=c['cache']['query_RANDOM_ROTATED']
    assert np.max(np.abs(realq@realq.T-randomq@randomq.T))<1e-6 and np.max(np.abs((realq*randomq).sum(1)))<1e-5
    # Learned effective query row changes preserve no fixed REAL initialization.
    checks['mapping_masks_and_query_geometry']=True
    checks['fixed_banks_no_grad_and_destructive_controls_exact']=True
    invalid=dict(c['cache']);invalid['mask']=invalid['mask'].copy();invalid['mask'][0]=False
    try:EvidenceAlignment('C4',invalid,0,c['folds'][0][0],np.arange(871))
    except ValueError:pass
    else:raise AssertionError('empty domain accepted')
    checks['empty_domains_rejected']=True
    seed_rng(2);a=agent_for('C4',2,1);b=branch(a)
    with torch.no_grad():b.alpha.fill_(100.)
    real=forward(a);ll=snapshot(a)
    relative=ll['delta'][:,3].norm(dim=-1)/(ll['base'][:,3].norm(dim=-1)+1e-8)
    assert relative.max()<=.100001 and ll['delta'][:,:3].count_nonzero()==0
    labels=a.targ.clone();labels[a.tst_idx]=1-labels[a.tst_idx]
    assert torch.equal(b.losses(a.targ),b.losses(labels))
    original=a.targ;a.targ=labels;assert torch.equal(forward(a),real);a.targ=original
    cf=counterfactual(a,2,1,real)
    assert all(r['mean_abs_asd_probability_change']>0 for r in cf if r['condition']!='REAL')
    # Trace one raw verified FC column; no other input or original trunk is modified.
    forward(a);ll=snapshot(a);raw=[v.clone() for v in a._cdea_raw]
    e=int(b.mask_real.any(0).nonzero()[7]);feature=int(b.valid_indices[e]);raw[3][0,feature]+=.1
    b(ll['base'],raw);changed=snapshot(a)
    assert float(changed['z'][0,e]-ll['z'][0,e])>.09999
    assert torch.count_nonzero(changed['z']-ll['z'])==1 and not torch.equal(changed['attention'],ll['attention'])
    raw=[v.clone() for v in a._cdea_raw];raw[3][:,[195,225,232,236]]+=100
    assert torch.equal(b(ll['base'],raw),ll['enhanced'])
    seed_rng(0);numerical=agent_for('C1',0,1);nb=branch(numerical)
    with torch.no_grad():nb.alpha.fill_(1.)
    forward(numerical);nl=snapshot(numerical);raw=[v.clone() for v in numerical._cdea_raw];raw[3][0,feature]+=.1
    assert not torch.equal(nb(nl['base'],raw),nl['enhanced'])
    checks.update(zero_residual_all_groups=True,train_only_evidence_loss=True,test_labels_prediction_invariant=True,
        vectorized_verified_fc_trace=True,unknown_edges_excluded=True,numeric_control_responds=True,residual_bound=True,
        counterfactual_same_checkpoint_base_exact=True,cf_probe={r['condition']:r['mean_abs_asd_probability_change'] for r in cf},
        fixed_clip_no_grad=True,class_only_repeated_query_expected=True,zero_query_key_inactive_expected=True)
    forward(a);diagnostic=diagnostics(a,'C4',2,1,real,dict(gradients(a),nonzero_parameters=nonzero_parameters(a),nonzero_key_parameters=nonzero_parameters(a,True)),parameter_norms(a))
    checkdir=STATE/'patch-check';checkdir.mkdir(exist_ok=True)
    for name in ('model.py','network.py'):shutil.copy2(ROOT/name,checkdir/name)
    subprocess.run(['git','apply','--check',str(EXP/'integration_patch.diff')],cwd=checkdir,check=True)
    subprocess.run(['git','apply',str(EXP/'integration_patch.diff')],cwd=checkdir,check=True)
    assert all((checkdir/n).read_text()==t for n,t in patched_sources().items());shutil.rmtree(checkdir)
    integrity(full=True);checks.update(status='PASS',completed_utc=utc(),patch_reconstruction_exact=True,root_history_unchanged=True,diagnostic_writer_finite=True)
    m=read_json(MANIFEST);m['preflight']=checks;save_json(MANIFEST,m);print('PREFLIGHT_PASS',flush=True)

def validate_complete():
    m=integrity(full=True);rows=read_csv(RESULTS/'fold_metrics.csv');ds=read_csv(RESULTS/'diagnostics.csv');cf=read_csv(RESULTS/'counterfactual.csv')
    keys={(c,str(s),str(f)) for c in CONFIGS for s in range(3) for f in range(1,11)}
    assert len(rows)==len(ds)==420
    for records in (rows,ds):assert {(r['config'],r['seed'],r['fold']) for r in records}==keys
    assert len(cf)==330 and {(r['seed'],r['fold'],r['condition']) for r in cf}=={(str(s),str(f),c) for s in range(3) for f in range(1,11) for c in CONDITIONS}
    assert all(t['status']=='COMPLETE' and t['folds']==10 for t in m['tasks'].values())
    assert [sum(int(r['physical_gpu'])==g for r in rows) for g in (0,1)]==[210,210]
    for r in rows:
        assert_historical_fold(r)
        assert r['task_signature']==task_signature(r['config'],int(r['seed']))
        assert all(math.isfinite(float(r[k])) for k in METRICS)
        assert abs(float(r['ba'])-float(r['hard_auc']))<1e-12 and abs(float(r['acc'])-float(r['selected_val_acc']))<1e-12
    for r in ds:
        assert float(r['residual_relative_max'])<=.100001 and float(r['noncorr_delta_max'])==0
        for key,v in r.items():
            if v and (key.startswith(('loss_','best_checkpoint_gradient_','sampled_gradient_','parameter_norm_'))):assert math.isfinite(float(v)),(key,v)
        if r['config']=='C13':assert all(float(r[f'D{k}_null_mean'])==0 for k in range(1,5))
        if r['config']=='C1':assert float(r['sampled_gradient_max_key'])==0
    assert all(r['same_checkpoint']==r['base_tokens_exact']==r['parameters_unchanged']=='True' and r['retrained']==r['test_labels_used_for_shuffle']=='False' for r in cf)
    assert all(float(r['mean_abs_asd_probability_change'])==float(r['prediction_flip_rate'])==0 for r in cf if r['condition']=='REAL')
    matches={c:sum(r['initial_trunk_sha256']==next(t['initial_trunk_sha256'] for t in rows if t['config']=='C0' and t['seed']==r['seed'] and t['fold']==r['fold']) for r in rows if r['config']==c) for c in CONFIGS}
    m['final_validation']=dict(status='PASS',groups=14,tasks=42,folds=420,diagnostics=420,counterfactual=330,gpu_tasks=[21,21],gpu_folds=[210,210],
        unique_finite_results=True,historical_p0_parity={str(n):30 for n in range(13,19)},root_history_hashes_unchanged=True,
        checkpoint_rng_recovery=True,initial_trunk_matches_c0=matches,verified_edges=252,unknown_edges_excluded=4)
    save_json(MANIFEST,m);return m


def write_readme(m):
    lines=['# 实验十九：AG-CDEA 解剖锚定疾病对比证据对齐','',
        '冻结14组×3seeds×连续10fold=420折；C4的11项同checkpoint反事实共330行。原P0完整重训30折，不进行超参数搜索。','',
        '## 源码、数据及历史审计','']
    for k,v in AUDIT.items():lines.extend([f'**{k}**：{v}',''])
    lines.extend(['## 真实解剖范围','',f"实验十一871/871数值唯一匹配，依据原111列索引联合ROI标签恢复名称。未知索引195/225/232/236不进入新分支；不删除原256输入。四域候选数为 {m['mapping_audit']['candidate_counts']}，域间允许重叠；至少一个端点满足给定HO名称前缀。Cingulate前缀不包含Paracingulate。所有具体名称与标志见domain_mapping.csv。四域是粗先验，不是验证过的单一功能网络或ASD病灶。",''])
    for item in MEDICAL:lines.append(f"- {item['domain']} [{item['source']}]({item['url']})：{item['boundary']}")
    lines.extend(['','## 固定模型与控制','',
       '每域ASD/TD冻结RN50文本先各L2归一化，差再L2归一化为1024维q。REAL候选8句逐字保留，NEUTRAL/UNRELATED/CLASS_ONLY共24句一次冻结；P0原8句仍用原函数与原缓存。semantic_cache仅含小型文本向量、mask和置换，不含患者数据。原CLIP构造首次消费的CPU随机流照旧保留；新分支不每epoch编码。',
       '可信z→[z,abs(z),tanh(z)]→Linear3/32→GELU→LayerNorm；加252×32边ID嵌入后再次LayerNorm。边ID仅数值身份，不代表医学知识。Key32/1024与q结合为q_eff=q@Wkey，score=h·q_eff/sqrt32。候选logit=score/0.2，NULL logit=log(候选数)，无效候选为−∞，NULL值为0。用完整softmax的边权重读取Value32/32及域分数u=sum(a*score)。u_global为四域均值。',
       'concat四域Value128、u4、NULL4共136维→Linear64→GELU→Linear36。delta=0.1*tanh(alpha)*detach(norm(base_CORR))*raw/sqrt(sum(raw²)+1e−8)，alpha=0。只增强位置3/CORR；其他三位置严格0。原SPGS、RES、输出头不变。',
       'Ltotal=原加权NLL+原四图正则均值+启用时0.1*BCEWithLogits(u_global[train],(y[train]==0))。只有outer-train标签参与新损失；Attention不读标签。C1/C2关闭新损失。该BCE是弱监督判别目标，不是患者文本—影像真值InfoNCE。','',
       '|组|名称|查询|解剖范围|证据监督|NULL|新增参数|','|---|---|---|---|---|---|---:|'])
    for c,d in DESIGNS.items():lines.append(f'|{c}|{d.name}|{d.query}|{d.anatomy}|{d.evidence if c!="C0" else False}|{d.null if c!="C0" else "原模型"}|{m["additional_parameters"][c]}|')
    lines.extend(['',
       'C1–C13除C12外新增53221参数，C12为57317；C1 Key32768注册但零查询导致结构失活，有效结构上界20453。其他组Key虽全部标量参与矩阵乘，固定查询rank r使其函数自由度仅r×32：真实/同几何随机rank4、类别文字rank1，不能将注册参数误作独立功能容量。C12的额外查询可学习，但自由Key本来已可表达4×32任意方向，其参数更多不等于增加同等函数自由度。',
       'C8交换ASD/TD仅将q取负；C9固定[2,3,0,1]；C10对252边列统一确定性无固定点置换，保留每域候选数与重叠，真实数值不动；C11把真实rank4张成空间等距映射到正交随机子空间，Gram误差<1e−6、方向cos接近0。C7四个相同类别查询为有意消除功能域文字对照。','',
       '## 初始化、协议、恢复与运行','',
       '原构造、双Adam和my_weight_init之后，fork_rng(seed=190000+task seed)创建新模块，恢复原CPU/CUDA随机流；原参数lr0.0038/WD0.11、新参数独立组lr0.0038/WD0。preflight验证原前向/双Adam两步逐值一致、新组初始前向一致、两步梯度、第三步RNG恢复和只干预辅助通路。',
       '固定871人、320维、ASD0/TD1，原450epoch、wait>200、val=test严格ACC提高选首次checkpoint。一个config×seed仅播种一次，连续原10折，绝不逐折重播种。各组早停不同使后续随机流分叉，逐折输入/初始化RNG哈希如实保存。原全局标准化/RFE、transductive及test选模不能作为独立无泄漏泛化证明。',
       '两卡各21任务/210折，每卡最多4worker，每worker2CPU线程。按(config序号*3+seed)%2固定物理卡；每worker仅暴露一张逻辑cuda0。资源压力只调整后续并发。每25epoch/折边界原子保存带SHA校验的模型、双Adam、RNG、best/wait、唯一签名及已提交结果；恢复先重建原CLIP缓存再恢复随机流。最终删临时源码和全部权重。独立runner没有import旧实验runner。','',
       '```bash','python experiment/experiment19_contrastive_disease_evidence_alignment/run.py prepare',
       'python experiment/experiment19_contrastive_disease_evidence_alignment/run.py preflight',
       'python experiment/experiment19_contrastive_disease_evidence_alignment/run.py all',
       'python experiment/experiment19_contrastive_disease_evidence_alignment/run.py resume',
       'python experiment/experiment19_contrastive_disease_evidence_alignment/run.py refresh',
       'python experiment/experiment19_contrastive_disease_evidence_alignment/run.py report','```','',
       'COMPLETE后all/resume只核对已完成结果；不自动重训或混入新attempt。根六源码与integration_patch.diff可独立重建临时源码；cdea.py+run.py+冻结JSON/NPZ/mapping以及既有数据哈希足以复核，不需要永久临时源副本。','',
       '## 诊断、反事实及判断','',
       '每fold一行420诊断，包含损失/梯度/参数、查询几何、四域NULL分布和entropy、患者注意力差别、正负FC权重、train/test类别证据描述、CORR残差及站点聚合。没有患者级预测文件。梯度为训练头两步/每25步采样最大值及最佳checkpoint eval损失反传值，后者不更新权重。结构计数为参与计算的上界，另报告Key函数自由度。',
       'C4同最佳模型测试REAL/NEUTRAL/UNRELATED/CLASS_ONLY/SWAPPED/DOMAIN_PERMUTED/ANATOMY_PERMUTED/RANDOM_ROTATED/NULL_OFF/RESIDUAL_OFF/FC_SHUFFLED，共330行。FC只在test内固定无自身置乱，不读标签；主干输入和注入前CAM逐值不变，参数不改、无训练，结束恢复REAL逐值相同。OFF是同已训练模型消除残差，不是另训练P0。',
       'ACC/BA/HardAUC/ProbAUC/SEN-ASD/SPE-TD；ProbAUC用exp(log-softmax)第0类，HardAUC=BA。先每seed十折均值，再三seed均值±population SD。全部14组与预定14配对完整报告；稳定标准为三个seed在ACC/BA/ProbAUC各自都正向，不做30独立样本显著性推断。真正语义与解剖必要性须同时经重训破坏对照和同模型反事实；敏感不等于正确。工程PASS而语义/分类FAIL允许且完整执行，不追加调参版本。','',
       '## 相关工作与可核验阅读范围',''])
    for name,item in LITERATURE.items():lines.append(f"- [{name}]({item['url']})：{item['conclusion']} 阅读范围：{item['read']}。")
    atomic(EXP/'README.md','\n'.join(lines)+'\n')


def stable_support(rows,left,right):
    return all(v>1e-12 for k in ('acc','ba','probability_auc') for v in paired(rows,left,right,k)['seeds'])


def report():
    import numpy as np
    from collections import Counter
    m=validate_complete();rows=read_csv(RESULTS/'fold_metrics.csv');ds=read_csv(RESULTS/'diagnostics.csv');cf=read_csv(RESULTS/'counterfactual.csv');agg=summary(rows)
    judgment={left+'-'+right:stable_support(rows,left,right) for left,right in PAIRS}
    candidate=all(judgment['C4-'+c] for c in ('C0','C3','C5','C6','C7','C8','C9','C10','C11','C12','C13'))
    m['scientific_judgment']=dict(stable_comparisons=judgment,first_module_candidate_supported=candidate,
         independent_unbiased_generalization_proven=False,medical_causal_evidence_proven=False,criterion=m['scientific_criteria']['stable'])
    save_json(MANIFEST,m)
    lines=['# 实验十九：AG-CDEA 解剖锚定疾病对比证据对齐正式报告','',
       f"工程验收PASS：14组/42任务/420折/420诊断/330同模型反事实；GPU0/1各21任务210折。科学稳定判据结果：{json.dumps(judgment,ensure_ascii=False)}。第一创新模块候选证据{'得到当前历史协议内的有限支持' if candidate else '不成立'}；不声称临床或独立泛化验证。",'',
       '## 1. 研究问题与历史失败经验','',AUDIT['history_9_14'],AUDIT['history_15_18'],'',
       '方案一年龄/性别Shared对齐停止：输入互补性及选择代理未体现稳定最终收益。方案二只检验可信FC解剖范围内，静态疾病差异查询是否选出更有用证据；不把新BCE或参数增加自动解释为语义知识。','',
       '## 2. 原论文与实际源码','',AUDIT['source'],AUDIT['pln'],AUDIT['paper'],AUDIT['data'],'',
       '## 3. ROI与连接恢复的可信性和限制','',
       '实验十一全871人/256列均为唯一数值匹配，最大误差量级10⁻¹⁵。历史表旧roi_i_name/j_name列为UNKNOWN，本轮从真实零基索引连接111标签表校验atlas值，不直接使用UNKNOWN字符串。110已核验ROI、额外未知82号两体素未命名。辅助252条可信连接，不重算RFE、不删除原主干未知4列。数据经过LedoitWolf相关及Fisher-z，不能直接解释为兴奋/抑制。',
       f"候选数{m['mapping_audit']['candidate_counts']}；重叠矩阵{m['mapping_audit']['domain_overlap']}；覆盖{m['mapping_audit']['covered_edge_count']}/252连接，域外{m['mapping_audit']['edges_in_no_domain']}条仅保留原主干。候选按至少一端点匹配锚点、可重叠，不构造不存在连接。",'',
       '## 4. 四域先验及医学来源','']
    for item in MEDICAL:lines.append(f"- {item['domain']} [{item['source']}]({item['url']})：{item['boundary']}")
    lines.extend(['','提供的REAL八句均采用heterogeneous/variable/potentially，不添加单向hyper/hypo假设、个体严重度或脑区病灶。医学候选修订0次；其余文字一次冻结，见prompts.json。','',
       '## 5. 冻结数学及注入位置','',
       'q=normalize(normalize(CLIP(T_ASD))−normalize(CLIP(T_TD)))。z的[值,绝对值,tanh值]→共享3/32编码、GELU、LN，加入252×32边ID后LN。score=h·(q@Wkey)/sqrt32，key32/1024、value32/32。真实mask无效边−∞；边logit score/0.2、NULL logit log(n)，softmax读取v与u=sum(a*score)。拼接v128/u4/NULL4→136/64/36，alpha零初始化的0.1*tanh(alpha)归一化方向残差只注入CAM后CORR位置3、SPGS前。所有原模块及优化数学保留。',
       '启用时总损失仅额外0.1 BCEWithLogits(mean_domain(u)[train], ytrain==0)，不是临床配对对比学习。NULL固定零Value不是医学不确定性概率。','',
       '## 6. 十四组控制与参数公平性','',
       '|组|名称|查询/解剖/NULL/证据|新增注册参数|结构参与上界|','|---|---|---|---:|---:|'])
    for c,d in DESIGNS.items():
        rr=[r for r in ds if r['config']==c]
        lines.append(f"|{c}|{d.name}|{d.query}/{d.anatomy}/{d.null}/{d.evidence if c!='C0' else False}|{m['additional_parameters'][c]}|{int(mean_field(rr,'parameters_structurally_active'))}|" )
    lines.extend(['','C1零查询使Key32768失活；C12新增4096个随机单位可学习查询，与其他组共有参数初始化仍完全相同。其他组Key32768标量进入计算，但函数自由度只有rank(Q)×32，rank4为128、C7重复类别rank1为32。实际采样非零梯度参数数见diagnostics，不把稠密梯度或注册计数当独立函数容量。','',AUDIT['identifiability'],'',
       '## 7. 原P0 parity、初始化及完整恢复','',
       '原P0前向、两步双Adam/CPU-CUDA随机流逐值一致；所有新组alpha=0初始预测相同；新共同参数独立fork_rng创建。实际C0三seed30折与实验13–18历史P0六指标、best epoch、执行epoch、初始化及可比RNG哈希全部一致。SP19原子校验checkpoint含两Adam和四类RNG；C0/C4/C12第三步逐值复演PASS。语义Cache不进入原CLIP首次加载轨迹。',
       f"相同seed/fold与C0的原参数初始化一致折数：{m['final_validation']['initial_trunk_matches_c0']}。由于一seed连续十fold且不同组早停不同，后续随机流自然分叉；配对同一数据折不代表所有后续折相同初始化。",'',
       '## 8. 全部420折六指标','',
       '单位%；每seed先平均10折，再三seed均值±population SD。Hard-AUC等于二分类BA，Prob-AUC来自exp(log-softmax)的ASD第0类。','',
       '|配置|ACC|BA|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|','|---|---:|---:|---:|---:|---:|---:|'])
    for c in CONFIGS:lines.append('|'+c+' '+DESIGNS[c].name+'|'+'|'.join(f"{agg[c][k]['mean']*100:.4f}±{agg[c][k]['sd']*100:.4f}" for k in METRICS)+'|')
    lines.extend(['','## 9. 三seed全部指标','', '|配置|seed|ACC|BA|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|','|---|---:|---:|---:|---:|---:|---:|---:|'])
    for c in CONFIGS:
        for s in range(3):lines.append(f'|{c}|{s}|'+'|'.join(f"{agg[c][k]['seeds'][s]*100:.4f}" for k in METRICS)+'|')
    lines.extend(['','## 10. 所有预定配对比较','',
       '每格为均值差pp [seed0,seed1,seed2]；胜/平/负按同seed/fold，三seed共享同一套数据划分，绝不把30折当30外部独立样本。无独立样本显著性检验。',''])
    for i,(left,right) in enumerate(PAIRS):
        lines.extend([f'### {chr(65+i)}. {left}−{right}：跨seed三主要指标全部正向={judgment[left+"-"+right]}','', '|指标|均值差pp [三个seed差pp]|逐折胜/平/负|','|---|---|---|'])
        for k in METRICS:
            p=paired(rows,left,right,k);lines.append(f"|{k}|{p['mean']*100:+.4f} [{', '.join(f'{v*100:+.4f}' for v in p['seeds'])}]|{p['wins']}/{p['ties']}/{p['losses']}|")
        lines.append('')
    lines.extend(['## 11. 正确医学文字与类别方向','',
       f"C4优于C5/6/7/8的预定跨seed三指标标准分别为{[judgment['C4-'+c] for c in ('C5','C6','C7','C8')]}。即使单一均值较高，也不升级为稳定语义证据。方向翻转可由自由Key吸收，功能域文字rank/几何与类别词rank1不同，更须结合C11/C12。",'',
       '## 12. 功能域对应与真实解剖对应','',
       f"C4−C9功能域对应稳定支持={judgment['C4-C9']}；C4−C10解剖对应稳定支持={judgment['C4-C10']}；C4−C3范围约束稳定支持={judgment['C4-C3']}。三者是不同问题，不能用全连接较差代替正确解剖优于等密度错配的证据。",'',
       '## 13. 同几何随机查询和可学习查询','',
       f"C11 Gram最大绝对误差{m['semantic_audit']['RANDOM_ROTATED']['gram_max_error']:.3g}，与真实对应方向cos={m['semantic_audit']['RANDOM_ROTATED']['cosine_to_real']}；C12独立随机初始化且不将REAL作为实际查询。C4−C11稳定支持={judgment['C4-C11']}，C4−C12={judgment['C4-C12']}。原空间旋转可重参数化，故随机方向表现近似尤其不能被解释成医学知识成立。",'',
       '## 14. NULL拒绝分布与强制匹配','',
       '|配置|域|NULL均值±跨fold均值SD|平均p05/p50/p95|拒绝>0.95比例|','|---|---|---:|---:|---:|'])
    for c in ('C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13'):
        rr=[r for r in ds if r['config']==c]
        for k in range(1,5):
            vals=[float(r[f'D{k}_null_mean']) for r in rr]
            lines.append(f"|{c}|D{k}|{np.mean(vals):.5f}±{np.std(vals):.5f}|{'/'.join(f'{mean_field(rr,f"D{k}_null_p{p}"):.5f}' for p in ('05','50','95'))}|{mean_field(rr,f'D{k}_null_reject95_fraction'):.5f}|")
    lines.extend(['',f"C4−C13稳定增益={judgment['C4-C13']}。NULL均值、分位数和患者SD共同检查；若仅降低所有连接质量或几乎拒绝所有人，不能称个体可靠度。NULL从未用标签直接决策，也未做概率校准。",'',
       '## 15. 证据监督、容量控制与数值梯度','',
       f"C1−C0={judgment['C1-C0']}；C2−C1={judgment['C2-C1']}；C4−C2={judgment['C4-C2']}（各为预定严格稳定判据）。证据监督可能只提供额外判别训练，需错误语义组共同区分。",'',
       '|配置|原分类/图损失|额外加权证据损失|证据train BA/AUC|Key/Value采样最大梯度均值|采样非零梯度参数均值|','|---|---:|---:|---:|---:|---:|'])
    for c in CONFIGS:
        rr=[r for r in ds if r['config']==c]
        lines.append(f"|{c}|{mean_field(rr,'loss_classification'):.5f}/{mean_field(rr,'loss_original_graph'):.5f}|{mean_field(rr,'loss_evidence'):.5f}|{mean_field(rr,'evidence_train_ba')*100:.3f}/{mean_field(rr,'evidence_train_probability_auc')*100:.3f}|{mean_field(rr,'sampled_gradient_max_key'):.5g}/{mean_field(rr,'sampled_gradient_max_value'):.5g}|{mean_field(rr,'parameters_sampled_nonzero_gradient'):.1f}|")
    lines.extend(['','证据BA/AUC为最佳checkpoint训练集描述，无独立证据标签或临床真值，不参与选模；C1恒0分数，阈值0决定预测常量，AUC0.5。参数norm/原GFR与PLN梯度全见CSV。','',
       '## 16. 语义几何是否趋同','',
       '|配置|原q域间cos均值|q_eff域间cos均值/最大值|q_eff中心化有效秩|','|---|---:|---:|---:|'])
    for c in CONFIGS[1:]:
        rr=[r for r in ds if r['config']==c];lines.append(f"|{c}|{mean_field(rr,'query_cosine_mean'):.6f}|{mean_field(rr,'effective_query_cosine_mean'):.6f}/{mean_field(rr,'effective_query_cosine_max'):.6f}|{mean_field(rr,'effective_query_centered_effective_rank'):.4f}|")
    lines.extend(['','C7重复查询、C1零查询是设计控制，低秩不是实现错误。其余组即使投影区分也不自动有增量；自由Key的可重参数化风险在训练前已记录。','',
       '## 17. 真实FC数值、注意力及正负贡献','',
       '|C4域|有效候选边数|边条件entropy|患者注意力SD|域u均值±SD|正/负FC权重|正/负FC分数贡献|','|---|---:|---:|---:|---:|---:|---:|'])
    c4=[r for r in ds if r['config']=='C4']
    for k in range(1,5):
        p=f'D{k}_';lines.append(f"|D{k}|{mean_field(c4,p+'effective_edges'):.3f}|{mean_field(c4,p+'conditional_edge_entropy'):.4f}|{mean_field(c4,p+'patient_attention_sd'):.5g}|{mean_field(c4,p+'score_mean'):.4f}±{mean_field(c4,p+'score_sd'):.4f}|{mean_field(c4,p+'positive_fc_attention_mass'):.4f}/{mean_field(c4,p+'negative_fc_attention_mass'):.4f}|{mean_field(c4,p+'positive_fc_score_contribution'):.4f}/{mean_field(c4,p+'negative_fc_score_contribution'):.4f}|")
    lines.extend(['',f"同患者域间注意力平均绝对差={mean_field(c4,'within_patient_domain_attention_mean_abs_difference'):.6g}。preflight逐列追踪仅改一条可信z，未知4列变化对新分支逐值无影响；C1也响应数值。实际FC依赖结合下面FC_SHUFFLED，不能把正负z解释为病理方向。",'',
       '## 18. 候选重要连接的跨折稳定性','',
       '每fold按测试受试者平均注意力列出每域Top5；以下为30折出现频数Top5。测试统计仅描述、不用于重新选模型或修改候选。域间重叠与边ID可能驱动选择；不称病灶或因果回路。','',
       '|域|FC列|真实HO端点|进入Top5次数/30|','|---|---:|---|---:|'])
    mapped={int(r['correlation_feature_index']):r for r in read_csv(EXP/'domain_mapping.csv')}
    for k in range(1,5):
        counts=Counter(e['feature'] for r in c4 for e in json.loads(r[f'D{k}_top_edges']))
        for e,n in counts.most_common(5):
            r=mapped[e];lines.append(f"|D{k}|{e}|{r['roi_i_name']} ↔ {r['roi_j_name']}|{n}|")
    lines.extend(['','## 19. 同最佳checkpoint反事实','',
       '|条件|ΔACC pp|ΔBA pp|ΔProb-AUC pp|翻转%|ASD概率平均绝对变化|logit绝对变化|attention绝对变化|NULL绝对变化|残差相对变化|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for condition in CONDITIONS:
        rr=[r for r in cf if r['condition']==condition]
        values=[mean_field(rr,k) for k in ('acc_difference_vs_real','ba_difference_vs_real','probability_auc_difference_vs_real','prediction_flip_rate','mean_abs_asd_probability_change','mean_abs_logit_change','attention_mean_abs_change','null_mean_abs_change','residual_relative_mean_change')]
        lines.append('|'+condition+'|'+'|'.join(f'{v*100:+.5f}' if i<3 else f'{v*100:.5f}' if i==3 else f'{v:.7g}' for i,v in enumerate(values))+'|')
    lines.extend(['','全部330行同参数、无重训、无测试标签置乱、CAM前Token逐值相同；结束REAL输出逐值恢复。反事实对已有文本方向敏感与重训时正确文字有优势是两个命题。FC_SHUFFLED/OFF有分布变化，只说明依赖、不证明医学必要性。','',
       '## 20. 有界残差、原分类器关系及旁路','',
       '|配置|CORR平均相对L2%|最大相对L2%|非零患者比例|signed alpha|与OFF概率变化|证据/主概率相关|证据/OFF概率相关|','|---|---:|---:|---:|---:|---:|---:|---:|'])
    for c in CONFIGS[1:]:
        rr=[r for r in ds if r['config']==c]
        lines.append(f"|{c}|{mean_field(rr,'residual_relative_mean')*100:.4f}|{max(float(r['residual_relative_max']) for r in rr)*100:.4f}|{mean_field(rr,'residual_nonzero_fraction'):.4f}|{mean_field(rr,'alpha'):+.5f}|{mean_field(rr,'mean_abs_asd_probability_change_vs_off'):.7g}|{mean_field(rr,'evidence_main_probability_correlation'):.5f}|{mean_field(rr,'evidence_off_probability_correlation'):.5f}|")
    near=sum(float(r['residual_relative_mean'])<1e-6 for r in c4)
    lines.extend(['',f"C4近零残差(折平均相对L2<1e−6)={near}/30。所有患者上界≤10%，非CORR三位置严格0。原主干仍含原FC是预先保留的保护路径；新增融合没有raw-FC直送decoder通道，但边ID/Value是数值容量，不保证医学语义。",'',
       '站点分布仅按既有20个site one-hot作聚合，不拟合或使用test标签；每折JSON列记录样本数/残差/NULL。此归一化方向使同checkpoint残差相对范数几乎由共享alpha固定，不能把站点间相同强度包装成患者可靠度；患者差别主要在方向/NULL/注意力。','',
       '## 21. 相对原P0与历史E9',''])
    e9=history_e9();e9agg=summary(e9)['E9']
    for key in METRICS:lines.append(f"- {key}：C4−C0={paired(rows,'C4','C0',key)['mean']*100:+.4f}pp；历史E9均值{e9agg[key]['mean']*100:.4f}%，C4−E9={(agg['C4'][key]['mean']-e9agg[key]['mean'])*100:+.4f}pp。")
    lines.extend(['','E9结构不同，仅预先指定的描述性参考，不声称相同初始化精确配对。','',
       '## 22. 是否克服实验12/15/16负结果','',
       '工程上：不再用4096→9旧投影输出构造辅助语义；直接固定CLIP差异，经Key投影选择252真实连接；不存在15的decoder raw-FC直送；Key/Value与alpha梯度、同模型敏感性均实测；残差≤10%控制16的32%过强问题。科学上：这些修复只消除断路/过强的替代解释，是否克服语义无净收益必须由全部破坏性对照和C4−C0共同回答，不能用工程成功代替。','',
       '## 23. 调度、耗时、恢复及文件审计','',
       f"正式训练墙钟{m.get('training_wall_seconds',0)/3600:.3f}小时；开始{m.get('started_utc')}，训练结束{m.get('training_completed_utc')}。每卡21任务/210折，每卡最多{m['workers_per_gpu']}worker、每worker2线程；任务实际物理卡、epoch和秒数见fold_metrics，完整日志保留。失败/恢复计数{m.get('failures_and_restarts',{})}；资源峰值{m.get('runtime_resource_peaks',{})}；调度节流{m.get('resource_throttling',[])}。GPU整卡采样含其他项目，不当成本任务独占显存。",'',
       '启动/训练均监测GPU显存、利用率、CPU、RSS、host可用内存。单config×seed进程连续10fold、唯一任务签名、每25epoch原子校验恢复，结果只取已提交完整折；无未完成attempt混入。最终临时源码/权重清理和无残留worker由manifest最终COMPLETE字段确认，报告生成发生在清理前。','',
       '## 24. 文献与创新边界','',AUDIT['novelty'],''])
    for name,item in LITERATURE.items():lines.append(f"- [{name}]({item['url']})：{item['conclusion']} 阅读范围：{item['read']}。")
    lines.extend(['','## 25. 是否支持MICCAI2027第一创新模块','',
       f"预定保守联合判据={'满足' if candidate else '未满足'}。{'可继续作为历史协议内的候选，但须独立严格验证。' if candidate else '不建议把这一固定实现包装为已成立的第一创新模块；保留完整负结果，不自动调参或新增配置。'} 无论结果如何，test选模、全局RFE与transductive限制使当前结果不能证明独立无泄漏泛化。",'',
       '## 26. 最可能失效的科学假设','',
       '若真实文字不优于同几何随机/可学习查询，最直接解释是四个静态差异方向在自由Key下只是可重参数化查询代码，并无患者级临床证据约束；若解剖错配不差，粗HO候选/稀疏既有RFE未携带额外定位优势；若NULL恒近1/证据主概率高度相关，辅助BCE可能学习冗余判别而非有用增量。上述解释按对照逐一判读，不从负结果推出所有医学知识都无用。','',
       '## 27. 下一步唯一问题与复核入口','',
       '唯一优先问题：能否获得可独立核验、带患者级变化且与真实FC定位对应的临床证据，并在限制自由投影吸收语义后，仍超过同几何匿名查询？在回答前停止该静态Prompt+自由Key具体实现路线的措辞/超参数搜索。此处仅研究建议，本轮不添加训练。',
       f"代码HEAD `{m['git_head_at_prepare']}`；目录 `{EXP}`；数据与根源码哈希、所有历史保护哈希、文本/Cache/mask/源码签名和preflight明细均在 `{MANIFEST}`。复核源码 `[run.py](../run.py)`、`[cdea.py](../cdea.py)`、`[integration_patch.diff](../integration_patch.diff)`；三CSV为真实唯一420/420/330行。根源码/实验1–18不改，不执行commit/push/pull/reset/checkout。"])
    atomic(RESULTS/'report.md','\n'.join(lines)+'\n')
    print('REPORT_WRITTEN',flush=True)


def append_handoff():
    m=read_json(MANIFEST);assert m['final_validation']['status']=='PASS'
    path=ROOT/'handoff.md';marker='<!-- experiment19_contrastive_disease_evidence_alignment -->'
    with open(ROOT/'handoff.md.lock','a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX);content=path.read_bytes()
        assert hashlib.sha256(content[:m['handoff_prefix_bytes']]).hexdigest()==m['handoff_prefix_sha256']
        if marker.encode() not in content:
            block='\n\n'+marker+'\n\n## 实验十九：AG-CDEA解剖锚定疾病对比证据对齐\n\n'
            block+=f"正式42任务420折/420诊断/330反事实已验收；两卡各210折。源码HEAD {m['git_head_at_prepare']}，报告SHA256 {sha(RESULTS/'report.md')}；run.py {sha(EXP/'run.py')}；cdea.py {sha(EXP/'cdea.py')}；文本/数据/映射全签名见 {MANIFEST}。旧内容逐字节保留。\n\n"
            block+=(RESULTS/'report.md').read_text()
            with open(path,'ab') as output:output.write(block.encode());output.flush();os.fsync(output.fileno())
    print('HANDOFF_APPENDED',flush=True)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", nargs="?", choices=["prepare", "preflight", "all", "resume", "report", "refresh"])
    parser.add_argument("--worker", nargs=2, metavar=("CONFIG", "SEED"))
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]))
    elif args.action == "prepare":
        prepare()
    elif args.action == "preflight":
        prepare()
        preflight()
    elif args.action in ("all", "resume"):
        schedule(args.workers_per_gpu)
    elif args.action == "report":
        report()
    elif args.action == "refresh":
        print(json.dumps(refresh(), ensure_ascii=False)[:1600])
    else:
        parser.error("action or --worker is required")

if __name__ == "__main__":
    main()
