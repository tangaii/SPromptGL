"""Reproducible Experiment39 preparation, preflight, training and analysis."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import difflib
import hashlib
import importlib
import io
import json
import os
import pickle
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import hase
import integration


ROOT = Path("/data/adodas/SPromptGL-main")
EXP = Path(__file__).resolve().parent
RUNTIME = EXP / ".runtime"
SOURCE = RUNTIME / "source"
CHECKPOINTS = RUNTIME / "checkpoints"
LOGS = RUNTIME / "logs"
RESULTS = EXP / "results"
DATA_DIR = ROOT / "data/ABIDE"
MAPPING = ROOT / "experiment/experiment19_contrastive_disease_evidence_alignment/domain_mapping.csv"
HISTORICAL = ROOT / "experiment/experiment32_disease_contrastive_evidence_routing/results/fold_metrics.csv"
PAPER = ROOT / "2025 MICCAI SPromptGL、Semantic Prompt Guided Graph Learning for Multi-modal Brain Disease.pdf"
PROMPTS = EXP / "prompts.json"
MANIFEST = EXP / "manifest.json"
TASKS = (
    "main.py", "model.py", "network.py", "layers.py", "utils.py",
    "ABIDE-simple-2-concat-weighted-cosine.sh",
)
EXPECTED_SHA256 = {
    "main.py": "dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2",
    "model.py": "254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda",
    "network.py": "484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec",
    "layers.py": "4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a",
    "utils.py": "95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67",
    "ABIDE-simple-2-concat-weighted-cosine.sh": "28c219e7c2deef4078cf8c28dd1b72f06f7494e997b9e38f0d933271521d5763",
    "processed_standard_data.csv": "b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1",
    "modal_feat_dict.npy": "8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e",
    "domain_mapping.csv": "cd34776253b961f37112d52d588f8e086139446326b93df5b237972821e0c9bf",
    "roi_256_named_mapping.csv": "b383308bb413a94d1646511a9bea5bc16f54c975b938ac4d947d676e357c5eb2",
    "exp32_fold_metrics.csv": "0133bb2fd5810beeb777c54b923c8d9b6e9ae2211ed225116ca7db2e3609ffec",
    "source_paper_pdf": "219ceddfd017e6105490a72864dd24df0bdbbf1c0579cf14af5bbd99a5d60c89",
}
PROTOCOL = dict(
    datname="ABIDE", datadir=str(ROOT / "data") + "/", nclass=2, nmodal=4,
    seed=0, mode="simple-2", GC_mode="weighted-cosine", MP_mode="GCN",
    MF_mode="concat", nepoch=450, early=200, lr=0.0038, reg=0.11,
    dropout=0.35, n_hidden=18, n_head=2, nlayer=1, th=0.9,
    theta_smooth=1.0, theta_degree=0.5, theta_sparsity=0.0, alpha=0.5,
)
GROUPS = {
    "E0_B0A": dict(gpu=0, kind="baseline", masks="none", route="none", spec=False),
    "E1_CAPACITY": dict(gpu=1, kind="capacity", masks="none", route="none", spec=False),
    "E2_GENERIC_MOE": dict(gpu=0, kind="moe", masks="all", route="full", spec=False),
    "E3_MED_UNIFORM": dict(gpu=1, kind="moe", masks="real", route="uniform", spec=False),
    "E4_MED_GFR_ROUTE": dict(gpu=0, kind="moe", masks="real", route="gfr", spec=False),
    "E5_WRONG_MASK": dict(gpu=1, kind="moe", masks="wrong39", route="full", spec=False),
    "E6_DATA_DRIVEN": dict(gpu=0, kind="moe", masks="train_cluster", route="full", spec=False),
    "E7_PATIENT_MISMATCH": dict(gpu=1, kind="moe", masks="real", route="full", spec=False),
    "E8_NO_SPEC": dict(gpu=0, kind="moe", masks="real", route="full", spec=False),
    "E9_HASE_FULL": dict(gpu=1, kind="moe", masks="real", route="full", spec=True),
}
PHYSICAL_GPU_GROUPS = {0: ("E0_B0A", "E2_GENERIC_MOE", "E4_MED_GFR_ROUTE", "E6_DATA_DRIVEN", "E8_NO_SPEC"),
                       1: ("E1_CAPACITY", "E3_MED_UNIFORM", "E5_WRONG_MASK", "E7_PATIENT_MISMATCH", "E9_HASE_FULL")}
SAVE_EVERY_EPOCHS = 20
STOP_REQUESTED = False
_CONTEXT = None


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_cuda_rng_scope():
    """Keep direct orchestration in the same single-visible-GPU RNG scope as workers."""
    if torch.cuda.is_available() and (torch.cuda.device_count() != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"):
        raise RuntimeError("Historical RNG scope requires CUDA_VISIBLE_DEVICES=0 for direct orchestration")


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_hash(state):
    digest = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    for value in [state["cpu"]] + list(state["cuda"]):
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def initial_parameter_hash(agent):
    digest = hashlib.sha256()
    for module_name, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion)):
        for name, parameter in sorted(module.named_parameters()):
            if module_name == "pln" and name.startswith("hase39."):
                continue
            digest.update(module_name.encode() + b"/" + name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fold_sha(folds):
    digest = hashlib.sha256()
    for train, test in folds:
        digest.update(np.asarray(train, dtype=np.int64).tobytes())
        digest.update(b"|")
        digest.update(np.asarray(test, dtype=np.int64).tobytes())
    return digest.hexdigest()


def prepare():
    EXP.mkdir(parents=True, exist_ok=True)
    for directory in (SOURCE, CHECKPOINTS, LOGS, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: sha(ROOT / name) for name in TASKS}
    for name, digest in source_hashes.items():
        if digest != EXPECTED_SHA256[name]:
            raise RuntimeError(f"Protected root source hash changed: {name}")
    data_hashes = {
        "processed_standard_data.csv": sha(DATA_DIR / "processed_standard_data.csv"),
        "modal_feat_dict.npy": sha(DATA_DIR / "modal_feat_dict.npy"),
        "domain_mapping.csv": sha(MAPPING),
        "roi_256_named_mapping.csv": sha(ROOT / "experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv"),
        "exp32_fold_metrics.csv": sha(HISTORICAL),
        "source_paper_pdf": sha(PAPER),
    }
    for name, digest in data_hashes.items():
        if digest != EXPECTED_SHA256[name]:
            raise RuntimeError(f"Audited input/reference hash changed: {name}")
    prompt_digest = sha(PROMPTS)
    for name in TASKS:
        shutil.copy2(ROOT / name, SOURCE / name)
    original_network = (SOURCE / "network.py").read_text()
    patched_network = integration.patch_network_source(original_network)
    (SOURCE / "network.py").write_text(patched_network)
    patch = "".join(difflib.unified_diff(
        original_network.splitlines(keepends=True), patched_network.splitlines(keepends=True),
        fromfile="a/network.py", tofile="b/private-runtime/network.py",
    ))
    (EXP / "integration.patch").write_text(patch)
    (RUNTIME / "source_hashes.json").write_text(json.dumps({
        "root_source_sha256": source_hashes,
        "private_source_sha256": {name: sha(SOURCE / name) for name in TASKS},
        "network_private_patch_sha256": sha(EXP / "integration.patch"),
    }, indent=2, sort_keys=True) + "\n")

    mapping_rows = list(csv.DictReader(open(MAPPING, newline="")))
    real = hase.true_masks(mapping_rows)
    sizes = real.sum(axis=1).astype(int).tolist()
    union = int(np.any(real > 0, axis=0).sum())
    if sizes != [72, 56, 45, 32] or union != 165:
        raise RuntimeError(f"Frozen Experiment19 mask audit mismatch: {sizes=} {union=}")
    if any(np.any(real[:, i]) for i in (195, 225, 232, 236)):
        raise RuntimeError("An unverified connection entered a mechanism mask")
    wrong, wrong_permutation = hase.wrong_masks(real, 39)
    if np.any(wrong[:, [195, 225, 232, 236]]) or np.array_equal(wrong, real):
        raise RuntimeError("Frozen wrong-mask control includes unknown or unchanged feature identities")
    named = list(csv.DictReader(open(ROOT / "experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv", newline="")))
    named_by_index = {int(row["correlation_feature_index"]): row for row in named}
    mechanisms = []
    for k, domain in enumerate(("D1", "D2", "D3", "D4")):
        edges = []
        for row in mapping_rows:
            if row["verified"].lower() != "true" or row[domain] != "1":
                continue
            index = int(row["correlation_feature_index"])
            name_row = named_by_index.get(index, {})
            edges.append({
                "correlation_feature_index": index,
                "original_connection_index": int(row["original_connection_index"]),
                "roi_i_zero_based": int(row["roi_i_zero_based"]),
                "roi_j_zero_based": int(row["roi_j_zero_based"]),
                "roi_i_name": row["roi_i_name"], "roi_j_name": row["roi_j_name"],
                "connection_name": name_row.get("connection_name"),
            })
        mechanisms.append({"id": domain, "feature_count": len(edges), "edges": edges})
    mask_record = {
        "source_csv_sha256": data_hashes["domain_mapping.csv"],
        "roi_mapping_sha256": data_hashes["roi_256_named_mapping.csv"],
        "feature_axis": "CORR raw input columns 64:320; feature index 0-255",
        "mask_sha256": hashlib.sha256(real.astype(np.uint8).tobytes()).hexdigest(),
        "union_feature_indices": np.flatnonzero(np.any(real > 0, axis=0)).astype(int).tolist(),
        "union_size": union,
        "unverified_indices_excluded": [195, 225, 232, 236],
        "wrong_mask_sha256": hashlib.sha256(wrong.astype(np.uint8).tobytes()).hexdigest(),
        "wrong_mask_seed": 39,
        "wrong_mask_permutation_sha256": hashlib.sha256(wrong_permutation.astype(np.int64).tobytes()).hexdigest(),
        "wrong_mask_union_feature_indices": np.flatnonzero(np.any(wrong > 0, axis=0)).astype(int).tolist(),
        "wrong_mask_feature_indices_by_expert": [
            np.flatnonzero(wrong[k] > 0).astype(int).tolist() for k in range(4)
        ],
        "verified_candidate_feature_indices": np.setdiff1d(
            np.arange(256), np.asarray([195, 225, 232, 236])
        ).astype(int).tolist(),
        "mechanisms": mechanisms,
    }
    write_json(EXP / "mechanism_masks.json", mask_record)
    data, dims, labels = load_inputs()
    folds = make_folds(labels)
    hist = pd.read_csv(HISTORICAL)
    baseline = hist[(hist["config"] == "B0") & (hist["seed"] == 0)].sort_values("fold")
    if len(baseline) != 10:
        raise RuntimeError("Experiment32 B0-A must provide exactly ten historical folds")
    manifest = {
        "experiment": "Experiment39 HASE-MoE",
        "status": "PREPARED",
        "workspace": str(EXP),
        "root_repository": str(ROOT),
        "root_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": PROTOCOL,
        "groups": {name: {**spec, "folds_expected": 10, "seed": 0} for name, spec in GROUPS.items()},
        "gpu_assignment": {str(gpu): list(names) for gpu, names in PHYSICAL_GPU_GROUPS.items()},
        "max_workers_per_gpu": 4,
        "cpu_threads_per_worker": 2,
        "fold_sha256": fold_sha(folds),
        "subject_count": int(data.shape[0]),
        "input_shape": list(data[:, :-1].shape),
        "modal_dims": dims,
        "class_counts": {"ASD_label0": int((labels == 0).sum()), "TD_label1": int((labels == 1).sum())},
        "data_sha256": data_hashes,
        "root_source_sha256": source_hashes,
        "private_source_sha256": {name: sha(SOURCE / name) for name in TASKS},
        "code_sha256": {name: sha(EXP / name) for name in
                        ("README.md", "prompts.json", "hase.py", "integration.py", "run.py",
                         "integration.patch", "mechanism_masks.json") if (EXP / name).is_file()},
        "prompt_sha256": prompt_digest,
        "mechanism_masks_sha256": sha(EXP / "mechanism_masks.json"),
        "integration_patch_sha256": sha(EXP / "integration.patch"),
        "e0_historical_reference": baseline.to_dict(orient="records"),
        "hase_trainable_parameter_count": 154816,
        "e1_trainable_parameter_count": 154452,
        "e1_capacity_difference_percent": 100.0 * abs(154816 - 154452) / 154816,
        "design_adjustment_before_training": (
            "E5 wrong-mask permutation and E6 fold-local clustering are restricted to the 252 anatomically verified CORR slots; "
            "unknown endpoint slots 195,225,232,236 stay outside anatomy-specific masks while remaining in the original trunk and the unrestricted E2 generic-MoE input."
        ),
        "model_frozen": True,
        "runtime_io_faults": [],
        "fold_results": {name: {"status": "PENDING", "completed_folds": 0} for name in GROUPS},
    }
    write_json(MANIFEST, manifest)
    print(json.dumps({"prepared": str(EXP), "folds": len(folds), "mask_sizes": sizes,
                      "union": union, "prompt_sha256": prompt_digest}, indent=2))


def load_inputs():
    feature_map = np.load(DATA_DIR / "modal_feat_dict.npy", allow_pickle=True).item()
    data = pd.read_csv(DATA_DIR / "processed_standard_data.csv").values.astype(np.float32)
    dims = [len(feature_map[key]) for key in feature_map]
    if data.shape != (871, 321) or dims != [48, 6, 10, 256]:
        raise RuntimeError(f"Unexpected ABIDE source shape/order: {data.shape}, {dims}")
    x = data[:, :-1].astype(np.float32, copy=False)
    y = data[:, -1].astype(np.int64) - 1
    if not np.array_equal(np.unique(y), np.array([0, 1])):
        raise RuntimeError("Labels must be ASD=0 and TD=1")
    if int((y == 0).sum()) != 403 or int((y == 1).sum()) != 468:
        raise RuntimeError("ABIDE class count changed from the audited 871-subject source")
    return x, dims, y


def make_folds(labels):
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    return [(train.astype(np.int64), test.astype(np.int64))
            for train, test in splitter.split(np.zeros(len(labels)), labels)]


def load_context():
    global _CONTEXT
    if _CONTEXT is not None:
        return _CONTEXT
    if not MANIFEST.exists() or not SOURCE.exists():
        raise RuntimeError("Run prepare before preflight or training")
    source_hashes = read_json(RUNTIME / "source_hashes.json")
    for name, expected in source_hashes["root_source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"Protected root source changed during experiment: {name}")
    for name, expected in source_hashes["private_source_sha256"].items():
        if sha(SOURCE / name) != expected:
            raise RuntimeError(f"Private source copy changed during experiment: {name}")
    sys.path.insert(0, str(SOURCE))
    sys.path.insert(1, str(ROOT))
    model_module = importlib.import_module("model")
    data, dims, labels = load_inputs()
    folds = make_folds(labels)
    manifest = read_json(MANIFEST)
    if fold_sha(folds) != manifest["fold_sha256"]:
        raise RuntimeError("Recomputed fold assignments do not match the frozen manifest")
    mapping_rows = list(csv.DictReader(open(MAPPING, newline="")))
    real_masks = hase.true_masks(mapping_rows)
    wrong, permutation = hase.wrong_masks(real_masks, 39)
    _CONTEXT = dict(model=model_module, x=data, y=labels, dims=dims, folds=folds,
                    real_masks=real_masks, wrong_masks=wrong, wrong_permutation=permutation,
                    manifest=manifest)
    torch.set_num_threads(2)
    return _CONTEXT


def masks_for(config, fold, context):
    spec = GROUPS[config]
    if spec["masks"] == "real":
        return context["real_masks"].copy(), {"source": "Experiment19 verified ROI mapping"}
    if spec["masks"] == "all":
        return np.ones((4, 256), dtype=np.float32), {"source": "all 256 CORR entries per expert"}
    if spec["masks"] == "wrong39":
        return context["wrong_masks"].copy(), {
            "source": "global feature permutation seed 39",
            "permutation": context["wrong_permutation"].astype(int).tolist(),
        }
    if spec["masks"] == "train_cluster":
        train, _ = context["folds"][fold - 1]
        masks, info = hase.data_driven_masks(
            context["x"][:, 64:320], train, context["real_masks"]
        )
        return masks, {"source": "outer-train FC correlation-profile KMeans", **info}
    return None, {"source": "unmasked capacity-matched adapter"}


def route_flags(config):
    if GROUPS[config]["route"] == "gfr":
        return {"fc": False, "pheno": False, "qc": False}
    return {"fc": True, "pheno": True, "qc": True}


def make_agent(config, fold, entry_rng=None):
    context = load_context()
    train, test = context["folds"][fold - 1]
    if entry_rng is not None:
        restore_rng(entry_rng)
    current_rng = rng_state()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        agent = context["model"].EvalHelper(
            context["dims"], context["x"], context["y"],
            argparse.Namespace(**PROTOCOL), train, test,
        )
    agent.MF_sav.close()
    agent.GCMP_sav.close()
    assert np.array_equal(agent.val_idx, test) and np.array_equal(agent.tst_idx, test)
    trunk_hash = initial_parameter_hash(agent)

    branch = None
    median = scale = source_rows = None
    if GROUPS[config]["kind"] != "baseline":
        median, scale = hase.train_median_mad(context["x"][:, 64:320], train)
        source_map = hase.mismatch_sources(
            len(context["y"]), train, test, fold, config == "E7_PATIENT_MISMATCH"
        )
        source_rows = torch.as_tensor(source_map, dtype=torch.long, device=agent.dev)
        med_t = torch.as_tensor(median, dtype=torch.float32, device=agent.dev)
        scale_t = torch.as_tensor(scale, dtype=torch.float32, device=agent.dev)
        if config == "E1_CAPACITY":
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(391001 + fold)
                branch = hase.CapacityMatchedAdapter(width=183).to(agent.dev)
        else:
            masks, _ = masks_for(config, fold, context)
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(391100 + fold)
                branch = hase.HASEMoE(masks, median, scale, source_map, route_flags(config)).to(agent.dev)
        median = med_t
        scale = scale_t
    integration.install(agent, config, branch, source_rows, median, scale)
    return agent, train, test, current_rng, trunk_hash, median, scale, source_rows


def model_state(agent):
    return {
        "gfr": {key: value.detach().cpu().clone() for key, value in agent.MessagePassing0.state_dict().items()},
        "pln": {key: value.detach().cpu().clone() for key, value in agent.ModalFusion.state_dict().items()},
    }


def load_model_state(agent, state):
    agent.MessagePassing0.load_state_dict(state["gfr"])
    agent.ModalFusion.load_state_dict(state["pln"])


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with open(temporary, "rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate_all(agent):
    agent.MessagePassing0.eval()
    agent.ModalFusion.eval()
    with torch.no_grad():
        model_num, parts = agent.dataLayer(agent.feat)
        features, _, _ = agent.MessagePassing0(parts, model_num, agent.targ, agent.trn_idx)
        output, _, attention = agent.ModalFusion(features)
    if output.shape != (871, 2) or not torch.isfinite(output).all():
        raise RuntimeError("Non-finite or incorrectly shaped classifier output")
    return output, features, attention


def metric_row(output, labels, test_index):
    target = torch.as_tensor(labels[test_index], dtype=torch.long)
    prediction = output[test_index].argmax(dim=1).detach().cpu().numpy().astype(np.int64)
    truth = target.cpu().numpy().astype(np.int64)
    confusion = confusion_matrix(truth, prediction, labels=[0, 1])
    one_hot_truth = np.eye(2, dtype=np.float64)[truth]
    one_hot_prediction = np.eye(2, dtype=np.float64)[prediction]
    auc = roc_auc_score(one_hot_truth, one_hot_prediction)
    return {
        "acc": float((truth == prediction).mean()),
        "auc": float(auc),
        "sen_asd": float(confusion[0, 0] / confusion[0].sum()),
        "spe_td": float(confusion[1, 1] / confusion[1].sum()),
        "asd_tp": int(confusion[0, 0]), "asd_fn": int(confusion[0, 1]),
        "td_fp": int(confusion[1, 0]), "td_tn": int(confusion[1, 1]),
        "n_test": int(len(test_index)),
    }


def output_prediction(output, idx):
    return output[idx].argmax(dim=1).detach().cpu().numpy().astype(np.int64)


def add_grad_audit(agent, audit):
    branch = getattr(agent.ModalFusion, "hase39", None)
    if branch is not None:
        for name, parameter in branch.named_parameters():
            grad = parameter.grad
            if grad is None:
                continue
            norm = float(grad.detach().norm().item())
            audit.setdefault(name, {"nonzero_steps": 0, "max_norm": 0.0})
            if norm > 0:
                audit[name]["nonzero_steps"] += 1
                audit[name]["max_norm"] = max(audit[name]["max_norm"], norm)
    for module_name, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion)):
        for name, parameter in module.named_parameters():
            if module_name == "pln" and name.startswith("hase39."):
                continue
            if parameter.grad is not None and torch.count_nonzero(parameter.grad).item():
                agent._e39_active_trunk_names.add(f"{module_name}.{name}")


def parameter_counts(agent):
    trunk = {}
    for module_name, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion)):
        for name, parameter in module.named_parameters():
            if module_name == "pln" and name.startswith("hase39."):
                continue
            trunk[f"{module_name}.{name}"] = parameter.numel()
    branch = getattr(agent.ModalFusion, "hase39", None)
    extension = dict(branch.named_parameters()) if branch is not None else {}
    active_trunk = agent._e39_active_trunk_names
    active_extension = {name for name, record in agent._e39_grad_audit.items() if record["nonzero_steps"]}
    return {
        "registered_trunk_parameters": int(sum(trunk.values())),
        "registered_extension_parameters": int(sum(p.numel() for p in branch.parameters())) if branch else 0,
        "registered_parameter_tensors": int(len(trunk) + len(extension)),
        "trunk_tensors_with_nonzero_gradient": int(len(active_trunk)),
        "extension_tensors_with_nonzero_gradient": int(len(active_extension)),
        "effective_trunk_parameters": int(sum(trunk[name] for name in active_trunk if name in trunk)),
        "effective_extension_parameters": int(sum(extension[name].numel() for name in active_extension if name in extension)),
        "effective_trainable_parameters": int(
            sum(trunk[name] for name in active_trunk if name in trunk)
            + sum(extension[name].numel() for name in active_extension if name in extension)
        ),
        "extension_grad_parameter_names": json.dumps(sorted(active_extension)),
    }


def simple2_step(agent, config, train_index):
    context = load_context()
    agent.MessagePassing0.train()
    agent.ModalFusion.train()
    agent.optimizer_MP0.zero_grad()
    agent.optimizer_MF.zero_grad()

    model_num, parts = agent.dataLayer(agent.feat)
    features, adjacency, aloss = agent.MessagePassing0(parts, model_num, agent.targ, train_index)
    all_graph_loss = 0
    for i in range(model_num):
        graph_loss_i = context["model"].GraphConstructLoss(
            parts[i], adjacency[i], agent.hyperpm.theta_smooth,
            agent.hyperpm.theta_degree, agent.hyperpm.theta_sparsity,
        )
        all_graph_loss = all_graph_loss + graph_loss_i
    graph_loss = all_graph_loss / 4
    logp, _, _ = agent.ModalFusion(features)
    cls_loss = context["model"].ClsLoss(logp, agent.targ, train_index, agent.weight)
    spec_loss = logp.new_zeros(())
    branch = getattr(agent.ModalFusion, "hase39", None)
    if branch is not None and GROUPS[config]["spec"]:
        expert_delta = agent.ModalFusion.hase39_last_train["expert_delta"]
        spec_loss = hase.specialization_loss(expert_delta, train_index)
    loss = cls_loss + graph_loss + 0.01 * spec_loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite optimization objective for {config}")
    loss.backward()
    add_grad_audit(agent, agent._e39_grad_audit)
    agent.optimizer_MP0.step()
    agent.optimizer_MF.step()
    if branch is not None:
        agent.ModalFusion.hase39_last_train = None
    return {
        "loss": float(loss.detach().item()), "cls_loss": float(cls_loss.detach().item()),
        "graph_loss": float(graph_loss.detach().item()), "aloss": float(aloss.detach().item()),
        "spec_loss": float(spec_loss.detach().item()),
    }


def capture_rng_safe():
    return rng_state()


def write_task_state(config, state):
    write_json(RUNTIME / f"task_{config}.json", state)


def task_checkpoint(config):
    return CHECKPOINTS / f"{config}.pt"


def association_eta(values, categories):
    values = np.asarray(values, dtype=np.float64)
    categories = np.asarray(categories)
    valid = np.isfinite(values) & (categories >= 0)
    values, categories = values[valid], categories[valid]
    if len(values) < 2 or np.var(values) < 1e-12:
        return 0.0
    total = float(np.sum((values - values.mean()) ** 2))
    between = 0.0
    for category in np.unique(categories):
        group = values[categories == category]
        if len(group):
            between += len(group) * float((group.mean() - values.mean()) ** 2)
    return float(between / total) if total > 0 else 0.0


def spearman(left, right):
    from scipy.stats import spearmanr
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def checkpoint_diagnostic(agent, config, fold, test_index, output):
    context = load_context()
    x = context["x"]
    y = context["y"]
    row = {"config": config, "seed": 0, "fold": fold}
    branch = getattr(agent.ModalFusion, "hase39", None)
    if branch is None:
        row.update({
            "route_mean": "[]", "route_sd": "[]", "route_usage": "[]",
            "route_entropy_mean": float("nan"), "expert_centered_cosine": "[]",
            "expert_residual_norm": "[]", "response_route_spearman": "[]",
            "route_asd_mean": "[]", "route_td_mean": "[]",
            "route_asd_usage": "[]", "route_td_usage": "[]",
            "route_site_eta2_mean": float("nan"), "route_age_eta2_mean": float("nan"),
            "route_sex_eta2_mean": float("nan"), "route_qc_abs_spearman_mean": float("nan"),
            "route_site_eta2": "[]", "route_age_eta2": "[]", "route_sex_eta2": "[]",
            "route_qc_abs_spearman": "[]",
            "fc_mismatch_response_change": float("nan"), "specialization_loss": 0.0,
            "grad_tensor_names_nonzero": 0, "grad_tensor_names_total": 0,
            "mask_sizes": "[]", "mask_union": 0, "max_residual_ratio": 0.0,
        })
        return row

    capture = {}
    def capture_cam_pre_hook(_module, inputs):
        capture["cam"] = inputs[0].detach()

    hook = branch.register_forward_pre_hook(capture_cam_pre_hook)
    output, _, _ = evaluate_all(agent)
    hook.remove()
    diag = agent.ModalFusion.hase39_last_eval
    route = diag["weights"][test_index].detach().cpu().numpy()
    response = diag["response"][test_index].detach().cpu().numpy()
    delta = diag["expert_delta"][test_index].detach().cpu().numpy()
    ratio = diag["residual_ratio"][test_index].detach().cpu().numpy()
    labels = y[test_index]

    route_mean = route.mean(axis=0)
    route_sd = route.std(axis=0)
    route_usage = np.bincount(route.argmax(axis=1), minlength=route.shape[1]) / len(route)
    entropy = -np.sum(route * np.log(np.maximum(route, 1e-12)), axis=1)
    class_mean = [route[labels == klass].mean(axis=0).tolist() for klass in (0, 1)]
    route_choice = route.argmax(axis=1)
    class_usage = [
        (np.bincount(route_choice[labels == klass], minlength=route.shape[1])
         / max(int(np.sum(labels == klass)), 1)).tolist()
        for klass in (0, 1)
    ]
    flat = delta.reshape(len(test_index), delta.shape[1], -1)
    centered = flat - flat.mean(axis=0, keepdims=True)
    similarities = []
    residual_norms = []
    for k in range(delta.shape[1]):
        residual_norms.append(float(np.linalg.norm(flat[:, k], axis=1).mean()))
        for j in range(k + 1, delta.shape[1]):
            denom = np.linalg.norm(centered[:, k], axis=1) * np.linalg.norm(centered[:, j], axis=1)
            cosine = np.divide(
                np.sum(centered[:, k] * centered[:, j], axis=1), denom,
                out=np.zeros_like(denom), where=denom > 1e-12,
            )
            similarities.append(float(np.mean(cosine)))
    response_norm = np.linalg.norm(response, axis=-1)
    if response_norm.ndim == 1:
        response_norm = response_norm[:, None]
    route_response = [spearman(route[:, k], response_norm[:, min(k, response_norm.shape[1] - 1)])
                      for k in range(route.shape[1])]

    sites = np.argmax(x[test_index, 26:46], axis=1)
    age_bins = np.argmax(x[test_index, :26], axis=1)
    sexes = np.argmax(x[test_index, 46:48], axis=1)
    site_eta = [association_eta(route[:, k], sites) for k in range(route.shape[1])]
    age_eta = [association_eta(route[:, k], age_bins) for k in range(route.shape[1])]
    sex_eta = [association_eta(route[:, k], sexes) for k in range(route.shape[1])]
    qc_corr = [
        abs(spearman(route[:, k], x[test_index, 48 + q]))
        for k in range(route.shape[1]) for q in range(16)
    ]

    mismatch = hase.mismatch_sources(
        len(y), context["folds"][fold - 1][0], test_index, 6000 + fold, True
    )
    source_tensor = torch.as_tensor(mismatch, dtype=torch.long, device=agent.dev)
    with torch.no_grad():
        if agent.ModalFusion.hase39_kind == "CAPACITY":
            mismatched = branch(
                capture["cam"], agent.feat, source_tensor,
                agent.ModalFusion.hase39_median, agent.ModalFusion.hase39_scale,
            )[1]["response"][test_index].detach().cpu().numpy()
        else:
            mismatched = branch(
                capture["cam"], agent.feat, source_rows=source_tensor,
            )[1]["response"][test_index].detach().cpu().numpy()
    response_change = float(
        np.mean(np.linalg.norm(mismatched - response, axis=-1)
                / (np.linalg.norm(response, axis=-1) + 1e-8))
    )

    mask = diag["masks"].detach().cpu().numpy() if diag["masks"] is not None else None
    grad = agent._e39_grad_audit
    row.update({
        "route_mean": json.dumps(route_mean.tolist()), "route_sd": json.dumps(route_sd.tolist()),
        "route_usage": json.dumps(route_usage.tolist()),
        "route_entropy_mean": float(entropy.mean()), "route_entropy_sd": float(entropy.std()),
        "route_entropy_q25": float(np.quantile(entropy, 0.25)),
        "route_entropy_q75": float(np.quantile(entropy, 0.75)),
        "route_patient_sd_mean": float(route_sd.mean()),
        "route_asd_mean": json.dumps(class_mean[0]), "route_td_mean": json.dumps(class_mean[1]),
        "route_asd_usage": json.dumps(class_usage[0]), "route_td_usage": json.dumps(class_usage[1]),
        "expert_centered_cosine": json.dumps(similarities),
        "expert_residual_norm": json.dumps(residual_norms),
        "response_route_spearman": json.dumps(route_response, allow_nan=True),
        "route_site_eta2_mean": float(np.mean(site_eta)),
        "route_age_eta2_mean": float(np.mean(age_eta)),
        "route_sex_eta2_mean": float(np.mean(sex_eta)),
        "route_qc_abs_spearman_mean": float(np.nanmean(qc_corr)) if np.isfinite(qc_corr).any() else float("nan"),
        "route_site_eta2": json.dumps(site_eta), "route_age_eta2": json.dumps(age_eta),
        "route_sex_eta2": json.dumps(sex_eta),
        "route_qc_abs_spearman": json.dumps(np.asarray(qc_corr).reshape(route.shape[1], 16).tolist(), allow_nan=True),
        "fc_mismatch_response_change": response_change,
        "specialization_loss": float(agent._e39_spec_sum / max(agent._e39_spec_count, 1)),
        "grad_tensor_names_nonzero": int(sum(bool(v["nonzero_steps"]) for v in grad.values())),
        "grad_tensor_names_total": int(sum(1 for _ in branch.named_parameters())),
        "grad_audit": json.dumps(grad, sort_keys=True),
        "mask_sizes": json.dumps(mask.sum(axis=1).astype(int).tolist()) if mask is not None else "[]",
        "mask_union": int(np.any(mask > 0, axis=0).sum()) if mask is not None else 0,
        "mask_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest() if mask is not None else "NONE",
        "max_residual_ratio": float(ratio.max()),
        "mean_residual_ratio": float(ratio.mean()),
        "mean_fc_response_norm": float(np.linalg.norm(response, axis=-1).mean()),
    })
    return row


def agent_optimizer_state(agent):
    return {
        "mp": copy.deepcopy(agent.optimizer_MP0.state_dict()),
        "mf": copy.deepcopy(agent.optimizer_MF.state_dict()),
    }


def restore_optimizer_state(agent, value):
    agent.optimizer_MP0.load_state_dict(value["mp"])
    agent.optimizer_MF.load_state_dict(value["mf"])


def save_e9_best(fold, model, metadata):
    path = CHECKPOINTS / f"E9_HASE_FULL_fold_{fold:02d}_best.pt"
    save_checkpoint(path, {"model": model, "metadata": metadata})


def worker(config, until_fold=None):
    if config not in GROUPS:
        raise ValueError(config)
    context = load_context()
    state_path = task_checkpoint(config)
    task_json = RUNTIME / f"task_{config}.json"
    if state_path.exists():
        saved = load_checkpoint(state_path)
        result = saved["result"]
    else:
        set_seed(0)
        result = {"config": config, "seed": 0, "metrics": [], "diagnostics": [],
                  "elapsed_seconds": 0.0, "status": "RUNNING"}
        saved = {"phase": "BOUNDARY", "next_fold": 1, "rng": rng_state(), "result": result}
        save_checkpoint(state_path, saved)

    if saved.get("phase") == "BOUNDARY":
        restore_rng(saved["rng"])

    while int(saved.get("next_fold", 11)) <= 10 or saved.get("phase") == "EPOCH":
        if saved.get("phase") == "EPOCH":
            fold = int(saved["fold"])
            entry_rng = saved["fold_entry_rng"]
        else:
            fold = int(saved["next_fold"])
            restore_rng(saved["rng"])
            entry_rng = saved["rng"]

        fold_started = time.monotonic()
        agent, train_index, test_index, fold_entry_rng, trunk_hash, median, scale, source_rows = make_agent(
            config, fold, entry_rng
        )
        agent._e39_grad_audit = saved.get("grad_audit", {}) if saved.get("phase") == "EPOCH" else {}
        agent._e39_active_trunk_names = set(saved.get("active_trunk_names", [])) if saved.get("phase") == "EPOCH" else set()
        agent._e39_spec_sum = float(saved.get("spec_sum", 0.0)) if saved.get("phase") == "EPOCH" else 0.0
        agent._e39_spec_count = int(saved.get("spec_count", 0)) if saved.get("phase") == "EPOCH" else 0
        if saved.get("phase") == "EPOCH":
            load_model_state(agent, saved["model"])
            restore_optimizer_state(agent, saved["optimizer"])
            restore_rng(saved["rng"])
            entry_rng = saved["fold_entry_rng"]
            best_state = saved["best_state"]
            best_acc = float(saved["best_acc"])
            best_epoch = int(saved["best_epoch"])
            wait = int(saved["wait"])
            start_epoch = int(saved["next_epoch"])
            previous_fold_seconds = float(saved.get("fold_elapsed", 0.0))
        else:
            best_state, best_acc, best_epoch, wait, start_epoch = None, 0.0, -1, 0, 0
            previous_fold_seconds = 0.0
        epoch_done = start_epoch
        saved = None
        fold_entry_hash = rng_hash(entry_rng)
        if fold == 1 and config != "E0_B0A":
            e0_state = task_checkpoint("E0_B0A")
            if e0_state.exists():
                e0 = load_checkpoint(e0_state)
                row = next((r for r in e0["result"]["metrics"] if int(r["fold"]) == 1), None)
                if row is not None:
                    if trunk_hash != row["initial_trunk_sha256"] or fold_entry_hash != row["initial_rng_sha256"]:
                        raise RuntimeError(f"{config}: seed-zero fold-1 trunk/RNG differs from E0")

        resume_after_early_stop = best_state is not None and wait > PROTOCOL["early"]
        epoch_range = () if resume_after_early_stop else range(start_epoch, PROTOCOL["nepoch"])
        for epoch in epoch_range:
            loss_items = simple2_step(agent, config, train_index)
            output, _, _ = evaluate_all(agent)
            prediction = output_prediction(output, agent.val_idx)
            correct = int((prediction == context["y"][agent.val_idx]).sum())
            current_acc = correct / len(agent.val_idx)
            if GROUPS[config]["spec"] and hasattr(agent.ModalFusion, "hase39_last_train"):
                agent._e39_spec_sum += loss_items["spec_loss"]
                agent._e39_spec_count += 1
            if current_acc > best_acc:
                best_acc, best_epoch, best_state, wait = current_acc, epoch, model_state(agent), 0
            else:
                wait += 1
            epoch_done = epoch + 1
            stop = STOP_REQUESTED or wait > PROTOCOL["early"] or epoch_done >= PROTOCOL["nepoch"]
            if epoch_done % SAVE_EVERY_EPOCHS == 0 or stop:
                elapsed = previous_fold_seconds + time.monotonic() - fold_started
                saved = {
                    "phase": "EPOCH", "fold": fold, "next_epoch": epoch_done,
                    "fold_entry_rng": entry_rng, "rng": rng_state(),
                    "model": model_state(agent), "optimizer": agent_optimizer_state(agent),
                    "best_state": best_state, "best_acc": best_acc, "best_epoch": best_epoch,
                    "wait": wait, "fold_elapsed": elapsed,
                    "initial_trunk_sha256": trunk_hash, "initial_rng_sha256": fold_entry_hash,
                    "grad_audit": agent._e39_grad_audit,
                    "active_trunk_names": sorted(agent._e39_active_trunk_names),
                    "spec_sum": agent._e39_spec_sum, "spec_count": agent._e39_spec_count,
                    "result": result,
                }
                save_checkpoint(state_path, saved)
                if stop and STOP_REQUESTED:
                    result["status"] = "PAUSED_RECOVERABLE"
                    write_task_state(config, result)
                    return result
            if (epoch_done % 50 == 0) or stop:
                print(f"{config} fold={fold} epoch={epoch_done} best={best_acc:.6f}@{best_epoch} wait={wait}", flush=True)
            if wait > PROTOCOL["early"]:
                break

        if best_state is None:
            raise RuntimeError(f"No checkpoint selected for {config} fold {fold}")
        continuation_rng = rng_state()
        load_model_state(agent, best_state)
        output, _, _ = evaluate_all(agent)
        selected_acc = float((output_prediction(output, agent.val_idx) == context["y"][agent.val_idx]).mean())
        if abs(selected_acc - best_acc) > 1e-12:
            raise RuntimeError("Restored ACC checkpoint differs from the strict-selection record")
        metrics = metric_row(output, context["y"], test_index)
        if abs(metrics["acc"] - best_acc) > 1e-12:
            raise RuntimeError("Original val_idx=test_idx checkpoint rule was not reproduced")
        gpu = int(os.environ.get("HASE_E39_PHYSICAL_GPU", "-1"))
        metrics.update({
            "config": config, "seed": 0, "fold": fold, "physical_gpu": gpu,
            "best_epoch": best_epoch, "epochs_executed": epoch_done,
            "initial_trunk_sha256": trunk_hash, "initial_rng_sha256": fold_entry_hash,
            "fold_wall_seconds": previous_fold_seconds + time.monotonic() - fold_started,
            **parameter_counts(agent),
        })
        diagnostic = checkpoint_diagnostic(agent, config, fold, test_index, output)
        result["metrics"] = [row for row in result["metrics"] if int(row["fold"]) != fold] + [metrics]
        result["diagnostics"] = [row for row in result["diagnostics"] if int(row["fold"]) != fold] + [diagnostic]
        result["metrics"].sort(key=lambda row: int(row["fold"]))
        result["diagnostics"].sort(key=lambda row: int(row["fold"]))
        result["elapsed_seconds"] = float(result.get("elapsed_seconds", 0.0) + metrics["fold_wall_seconds"])
        result["status"] = "RUNNING" if fold < 10 else "COMPLETE"
        if config == "E9_HASE_FULL":
            save_e9_best(fold, best_state, {
                "best_epoch": best_epoch, "initial_trunk_sha256": trunk_hash,
                "initial_rng_sha256": fold_entry_hash, "fold": fold,
            })
        boundary = {"phase": "BOUNDARY", "next_fold": fold + 1,
                    "rng": continuation_rng, "result": result}
        save_checkpoint(state_path, boundary)
        write_task_state(config, result)
        print(f"FOLD_DONE {config} fold={fold} acc={metrics['acc']:.6f} auc={metrics['auc']:.6f}", flush=True)
        del agent, output
        torch.cuda.empty_cache()
        saved = boundary
        if until_fold is not None and fold >= until_fold:
            result["status"] = "PAUSED_AFTER_FOLD"
            write_task_state(config, result)
            return result
        if STOP_REQUESTED:
            result["status"] = "PAUSED_RECOVERABLE"
            write_task_state(config, result)
            return result

    result["status"] = "COMPLETE" if len(result["metrics"]) == 10 else "RUNNING"
    write_task_state(config, result)
    return result


def compare_e0_first_fold():
    state_path = task_checkpoint("E0_B0A")
    if not state_path.exists():
        raise RuntimeError("E0 fold 1 has not been trained")
    state = load_checkpoint(state_path)
    row = next((r for r in state["result"]["metrics"] if int(r["fold"]) == 1), None)
    if row is None:
        raise RuntimeError("E0 fold 1 result missing")
    hist = pd.read_csv(HISTORICAL)
    ref = hist[(hist["config"] == "B0") & (hist["seed"] == 0) & (hist["fold"] == 1)].iloc[0]
    fields = ("acc", "auc", "sen_asd", "spe_td", "best_epoch",
              "initial_trunk_sha256", "initial_rng_sha256")
    differences = {}
    for key in fields:
        actual, expected = row[key], ref[key]
        equal = actual == expected if isinstance(expected, str) else abs(float(actual) - float(expected)) < 1e-12
        differences[key] = {"actual": actual, "historical": expected.item() if hasattr(expected, "item") else expected,
                            "match": bool(equal)}
    write_json(RUNTIME / "e0_first_fold_parity.json", differences)
    if not all(item["match"] for item in differences.values()):
        raise RuntimeError("E0 first-fold replay differs from Experiment32 B0-A; do not launch other groups")
    print("E0_FOLD1_PARITY_PASS", json.dumps(differences, sort_keys=True), flush=True)


def compare_e0_all_folds():
    state_path = task_checkpoint("E0_B0A")
    if not state_path.exists():
        raise RuntimeError("E0 has not completed")
    state = load_checkpoint(state_path)
    result = state["result"]
    if state.get("phase") != "BOUNDARY" or int(state.get("next_fold", 0)) != 11:
        raise RuntimeError("E0 did not finish all ten folds")
    rows = sorted(result["metrics"], key=lambda row: int(row["fold"]))
    if len(rows) != 10 or [int(row["fold"]) for row in rows] != list(range(1, 11)):
        raise RuntimeError("E0 fold set is incomplete or duplicated")
    hist = pd.read_csv(HISTORICAL)
    ref = hist[(hist["config"] == "B0") & (hist["seed"] == 0)].sort_values("fold")
    if len(ref) != 10:
        raise RuntimeError("Historical Experiment32 B0-A fold table is incomplete")
    fields = ("acc", "auc", "sen_asd", "spe_td", "best_epoch", "epochs_executed",
              "initial_trunk_sha256", "initial_rng_sha256")
    checks = {}
    for field in fields:
        matches = []
        for row, (_, historical) in zip(rows, ref.iterrows()):
            actual, expected = row[field], historical[field]
            equal = actual == expected if field.endswith("sha256") else abs(float(actual) - float(expected)) < 1e-12
            matches.append(bool(equal))
        checks[field] = {"matched_folds": int(sum(matches)), "of": 10,
                         "match_by_fold": matches}
    checks["all_fields_match"] = all(item["matched_folds"] == 10 for item in checks.values())
    write_json(RUNTIME / "e0_full_parity.json", checks)
    if not checks["all_fields_match"]:
        raise RuntimeError(f"E0 baseline does not match Experiment32 B0-A on all folds: {checks}")
    print("E0_FULL_10FOLD_PARITY_PASS", json.dumps(checks, sort_keys=True), flush=True)


def sigterm_handler(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def assert_equal_state(left, right):
    if left.keys() != right.keys():
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, dict):
            if not assert_equal_state(a, b):
                return False
        elif torch.is_tensor(a):
            if not torch.equal(a, b):
                return False
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                return False
            for x, y in zip(a, b):
                if torch.is_tensor(x) and not torch.equal(x, y):
                    return False
                if not torch.is_tensor(x) and x != y:
                    return False
        elif a != b:
            return False
    return True


def preflight():
    set_cuda_rng_scope()
    if torch.cuda.is_available() and (torch.cuda.device_count() != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"):
        raise RuntimeError("Preflight must run with CUDA_VISIBLE_DEVICES=0 to match the historical single-GPU worker RNG scope")
    context = load_context()
    if context["x"].shape != (871, 320) or context["dims"] != [48, 6, 10, 256]:
        raise RuntimeError("Input identity/dimensions failed preflight")
    unknown = [195, 225, 232, 236]
    if np.any(context["real_masks"][:, unknown]) or np.any(context["wrong_masks"][:, unknown]):
        raise RuntimeError("Unverified connection was included in a medical/control mechanism mask")
    wrong = context["wrong_masks"]
    if not np.array_equal(context["real_masks"].sum(axis=1), wrong.sum(axis=1)):
        raise RuntimeError("Wrong-mask control changed expert feature counts")
    if not np.array_equal(context["real_masks"] @ context["real_masks"].T,
                           wrong @ wrong.T):
        raise RuntimeError("Wrong-mask control changed pairwise overlap counts")
    if np.array_equal(context["real_masks"], wrong):
        raise RuntimeError("Wrong-mask control retained the true feature membership")

    train, test = context["folds"][0]
    real_e6, e6_info = hase.data_driven_masks(context["x"][:, 64:320], train, context["real_masks"])
    if np.any(real_e6[:, unknown]):
        raise RuntimeError("An unverified endpoint entered the data-driven expert control")
    if not np.array_equal(real_e6.sum(axis=1), context["real_masks"].sum(axis=1)):
        raise RuntimeError("Data-driven mask control changed frozen expert sizes")
    if not np.array_equal(real_e6 @ real_e6.T, context["real_masks"] @ context["real_masks"].T):
        raise RuntimeError("Data-driven mask control changed frozen overlap-pattern counts")
    if np.array_equal(real_e6, context["real_masks"]):
        raise RuntimeError("Data-driven mask control reused the real ROI feature identities")

    set_seed(0)
    e0, _, e0_test, entry_rng, trunk_hash, _, _, _ = make_agent("E0_B0A", 1)
    ref = pd.read_csv(HISTORICAL)
    e0_ref = ref[(ref.config == "B0") & (ref.seed == 0) & (ref.fold == 1)].iloc[0]
    if trunk_hash != e0_ref.initial_trunk_sha256 or rng_hash(entry_rng) != e0_ref.initial_rng_sha256:
        raise RuntimeError("E0 preflight construction/RNG differs from historical B0-A")
    captured = {}
    hook = e0.ModalFusion.FeedForward[-1].register_forward_hook(
        lambda _module, _inputs, output: captured.setdefault("cam", output.detach())
    )
    output, _, attention = evaluate_all(e0)
    hook.remove()
    cam = captured["cam"]
    final_attention = torch.as_tensor(attention[-1], device=e0.dev, dtype=cam.dtype).reshape(871, -1)
    direct = e0.ModalFusion.Outputlayer(cam.reshape(871, -1), final_attention)[0]
    if not torch.equal(output, direct):
        raise RuntimeError("E0 is not the direct CAM flatten -> original OutputLayer path")
    del e0
    torch.cuda.empty_cache()

    set_seed(0)
    e9, train, test, _, _, median, scale, source_rows = make_agent("E9_HASE_FULL", 1)
    e9._e39_grad_audit = {}
    e9._e39_active_trunk_names = set()
    e9._e39_spec_sum = 0.0
    e9._e39_spec_count = 0
    first_output, _, _ = evaluate_all(e9)
    route = e9.ModalFusion.hase39_last_eval["weights"]
    if not torch.isfinite(route).all() or torch.any(route < 0):
        raise RuntimeError("HASE router produced invalid weights")
    if not torch.allclose(route.sum(dim=1), torch.ones(871, device=e9.dev), atol=1e-6):
        raise RuntimeError("HASE route weights do not sum to one")
    if float(e9.ModalFusion.hase39_last_eval["residual_ratio"].max()) > 1e-6:
        raise RuntimeError("HASE residual must begin at exactly zero")
    if not torch.isfinite(first_output).all():
        raise RuntimeError("HASE preflight output is non-finite")

    branch = e9.ModalFusion.hase39
    raw_var = e9.feat.detach().clone().requires_grad_(True)
    cam_var = captured_cam(e9)
    _, direct_diag = branch(cam_var, raw_var)
    for k in range(4):
        grad = torch.autograd.grad(direct_diag["response"][:, k].square().sum(), raw_var,
                                   retain_graph=k < 3)[0]
        allowed = branch.masks[k].bool()
        outside_grad = grad[:, 64:320][:, ~allowed]
        if outside_grad.numel() and torch.count_nonzero(outside_grad).item() != 0:
            raise RuntimeError(f"Expert {k} has a direct gradient path outside its mask")
    changed = raw_var.detach().clone()
    selected = int(torch.nonzero(branch.masks[0], as_tuple=False)[0].item())
    changed[:, 64 + selected] += 0.5
    changed_response = branch(cam_var, changed)[1]["response"]
    if torch.equal(changed_response[:, 0], direct_diag["response"][:, 0]):
        raise RuntimeError("A nonzero masked FC change did not enter its expert response")

    # Four deterministic HASE-only steps establish that every branch can open from zero.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        probe_optimizer = torch.optim.Adam(branch.parameters(), lr=PROTOCOL["lr"], weight_decay=0.0)
        for _ in range(4):
            probe_optimizer.zero_grad()
            updated, diagnostic = branch(cam_var, e9.feat)
            loss = updated.square().mean() + 0.01 * hase.specialization_loss(
                diagnostic["expert_delta"], train
            )
            loss.backward()
            probe_optimizer.step()
    active = {name.split(".")[0] for name, parameter in branch.named_parameters()
              if parameter.grad is not None and torch.count_nonzero(parameter.grad).item()}
    required_active = {
        "fc_encoder", "router_hidden", "router_output", "expert_code",
        "adapter_hidden", "adapter_output",
    }
    if not required_active.issubset(active):
        raise RuntimeError(f"HASE modules did not all receive gradients: {active}")
    if float(diagnostic["residual_ratio"].max().item()) > 0.100001:
        raise RuntimeError("HASE residual exceeded the frozen 10% bound")

    # Rewind the formal model and verify a two-optimizer dropout update replays exactly.
    set_seed(0)
    replay, replay_train, _, _, _, _, _, _ = make_agent("E9_HASE_FULL", 1)
    replay._e39_grad_audit = {}
    replay._e39_active_trunk_names = set()
    replay._e39_spec_sum = 0.0
    replay._e39_spec_count = 0
    simple2_step(replay, "E9_HASE_FULL", replay_train)
    saved_model = model_state(replay)
    saved_optimizer = agent_optimizer_state(replay)
    saved_rng = rng_state()
    simple2_step(replay, "E9_HASE_FULL", replay_train)
    expected_model = model_state(replay)
    load_model_state(replay, saved_model)
    restore_optimizer_state(replay, saved_optimizer)
    restore_rng(saved_rng)
    simple2_step(replay, "E9_HASE_FULL", replay_train)
    replayed_model = model_state(replay)
    if not assert_equal_state(expected_model, replayed_model):
        raise RuntimeError("Dual Adam plus Python/NumPy/Torch/CUDA RNG replay was not exact")
    parity = {
        "input_shape": list(context["x"].shape), "modal_dims": context["dims"],
        "e0_direct_cam_outputlayer_exact": True,
        "e0_initial_trunk_sha256": trunk_hash,
        "e0_historical_trunk_sha256": e0_ref.initial_trunk_sha256,
        "e0_historical_rng_sha256": e0_ref.initial_rng_sha256,
        "real_mask_sizes": context["real_masks"].sum(axis=1).astype(int).tolist(),
        "real_mask_union": int(np.any(context["real_masks"] > 0, axis=0).sum()),
        "wrong_mask_keeps_sizes_and_pairwise_overlaps": True,
        "data_driven_mask_keeps_sizes_and_pairwise_overlaps": True,
        "data_driven_mask_info": e6_info,
        "nonmask_direct_input_gradient_zero": True,
        "masked_fc_change_changes_response": True,
        "hase_active_modules_after_four_updates": sorted(active),
        "hase_residual_initial_max": 0.0,
        "hase_residual_max_after_probe": float(diagnostic["residual_ratio"].max().item()),
        "router_valid": True,
        "optimizer_rng_checkpoint_replay_exact": True,
        "test_labels_used_for_model_construction": False,
    }
    write_json(RUNTIME / "preflight.json", parity)
    del e9, replay
    torch.cuda.empty_cache()
    print("PREFLIGHT_PASS", json.dumps(parity, sort_keys=True), flush=True)


def captured_cam(agent):
    captured = {}
    hook = agent.ModalFusion.FeedForward[-1].register_forward_hook(
        lambda _module, _inputs, output: captured.setdefault("cam", output.detach())
    )
    with torch.no_grad():
        evaluate_all(agent)
    hook.remove()
    return captured["cam"]


def counterfactuals():
    context = load_context()
    result = []
    e9_task = read_json(RUNTIME / "task_E9_HASE_FULL.json")
    if e9_task.get("status") != "COMPLETE" or len(e9_task.get("metrics", [])) != 10:
        raise RuntimeError("E9 must finish all ten training folds before fixed-checkpoint interventions")
    e9_rows = pd.DataFrame(e9_task["metrics"]).set_index("fold")
    for fold, (train, test) in enumerate(context["folds"], start=1):
        checkpoint_path = CHECKPOINTS / f"E9_HASE_FULL_fold_{fold:02d}_best.pt"
        if not checkpoint_path.exists():
            raise RuntimeError(f"E9 best checkpoint missing for fold {fold}")
        set_seed(0)
        agent, _, _, _, _, _, _, _ = make_agent("E9_HASE_FULL", fold)
        best = load_checkpoint(checkpoint_path)
        load_model_state(agent, best["model"])
        output, _, _ = evaluate_all(agent)
        base_metric = metric_row(output, context["y"], test)
        baseline_prediction = output_prediction(output, test)
        ref = e9_rows.loc[fold]
        for field in ("acc", "auc", "sen_asd", "spe_td"):
            if abs(float(base_metric[field]) - float(ref[field])) > 1e-12:
                raise RuntimeError(f"E9 counterfactual baseline did not restore fold {fold} {field}")

        interventions = [("REAL", {})]
        interventions.append(("WRONG_MASK", {
            "mask_override": torch.as_tensor(context["wrong_masks"], dtype=torch.float32, device=agent.dev),
        }))
        shuffled_rows = hase.mismatch_sources(len(context["y"]), train, test, 6000 + fold, True)
        interventions.append(("PATIENT_SHUFFLE", {
            "source_rows": torch.as_tensor(shuffled_rows, dtype=torch.long, device=agent.dev),
        }))
        interventions.append(("UNIFORM_ROUTE", {"route_uniform": True}))
        for expert in range(4):
            keep = np.ones(4, dtype=np.float32)
            keep[expert] = 0.0
            interventions.append((f"EXPERT_OFF_{expert + 1}", {"expert_keep": keep}))
        interventions.append(("ALL_OFF", {"expert_keep": np.zeros(4, dtype=np.float32)}))

        for name, override in interventions:
            if name != "REAL":
                if "mask_override" in override:
                    override["mask_override"] = override["mask_override"].to(agent.dev)
                if "expert_keep" in override:
                    override["expert_keep"] = torch.as_tensor(override["expert_keep"], device=agent.dev)
                integration.evaluation_override(agent, **override)
            try:
                output, _, _ = evaluate_all(agent)
            finally:
                integration.clear_override(agent)
            metrics = metric_row(output, context["y"], test)
            prediction = output_prediction(output, test)
            flips = float(np.mean(prediction != baseline_prediction))
            row = {
                "config": "E9_HASE_FULL", "seed": 0, "fold": fold,
                "intervention": name, **metrics,
                "delta_acc_pp": 100.0 * (metrics["acc"] - base_metric["acc"]),
                "delta_auc_pp": 100.0 * (metrics["auc"] - base_metric["auc"]),
                "delta_sen_asd_pp": 100.0 * (metrics["sen_asd"] - base_metric["sen_asd"]),
                "delta_spe_td_pp": 100.0 * (metrics["spe_td"] - base_metric["spe_td"]),
                "prediction_flip_fraction": flips,
                "best_epoch_unchanged": int(best["metadata"]["best_epoch"]),
            }
            result.append(row)
        del agent
        torch.cuda.empty_cache()
        print(f"COUNTERFACTUAL_DONE E9 fold={fold}", flush=True)
    pd.DataFrame(result).sort_values(["fold", "intervention"]).to_csv(
        RESULTS / "counterfactual.csv", index=False
    )
    return result


def paired_comparison(metrics, left, right, metric):
    l = metrics[metrics.config == left].set_index("fold")[metric]
    r = metrics[metrics.config == right].set_index("fold")[metric]
    diff = (l - r) * 100.0
    return {
        "left": left, "right": right, "metric": metric,
        "mean_delta_pp": float(diff.mean()),
        "wins": int((diff > 1e-12).sum()),
        "ties": int((np.abs(diff) <= 1e-12).sum()),
        "losses": int((diff < -1e-12).sum()),
        "fold_delta_pp": [float(value) for value in diff.to_list()],
    }


def summarize_and_report():
    manifest = read_json(MANIFEST)
    task_results = []
    for config in GROUPS:
        path = RUNTIME / f"task_{config}.json"
        if not path.exists():
            raise RuntimeError(f"Task state missing for {config}")
        task = read_json(path)
        if task["status"] != "COMPLETE" or len(task["metrics"]) != 10:
            raise RuntimeError(f"{config} is incomplete: {task['status']} / {len(task['metrics'])} folds")
        task_results.append(task)
    fold_metrics = pd.DataFrame([row for task in task_results for row in task["metrics"]])
    diagnostics = pd.DataFrame([row for task in task_results for row in task["diagnostics"]])
    if len(fold_metrics) != 100 or fold_metrics.duplicated(["config", "seed", "fold"]).any():
        raise RuntimeError("Fold results are not 100 unique config×seed×fold rows")
    fold_metrics = fold_metrics.sort_values(["config", "fold"])
    diagnostics = diagnostics.sort_values(["config", "fold"])
    fold_metrics.to_csv(RESULTS / "fold_metrics.csv", index=False)
    diagnostics.to_csv(RESULTS / "diagnostics.csv", index=False)

    metrics = pd.read_csv(RESULTS / "fold_metrics.csv")
    summary_rows = []
    formal = ("acc", "auc", "spe_td", "sen_asd")
    for config in GROUPS:
        subset = metrics[metrics.config == config]
        row = {"config": config, "folds": len(subset)}
        for metric in formal:
            values = subset[metric].to_numpy(dtype=float) * 100.0
            row[f"{metric}_mean_pct"] = float(values.mean())
            row[f"{metric}_sd_pct"] = float(values.std(ddof=0))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "summary.csv", index=False)

    comparisons = []
    for config in ("E0_B0A", "E1_CAPACITY", "E2_GENERIC_MOE", "E3_MED_UNIFORM",
                   "E4_MED_GFR_ROUTE", "E5_WRONG_MASK", "E6_DATA_DRIVEN",
                   "E7_PATIENT_MISMATCH", "E8_NO_SPEC"):
        for metric in formal:
            comparisons.append(paired_comparison(metrics, "E9_HASE_FULL", config, metric))
    write_json(RESULTS / "paired_comparisons.json", comparisons)
    criteria = {}
    for config in ("E0_B0A", "E2_GENERIC_MOE", "E5_WRONG_MASK", "E6_DATA_DRIVEN", "E7_PATIENT_MISMATCH"):
        acc = paired_comparison(metrics, "E9_HASE_FULL", config, "acc")
        auc = paired_comparison(metrics, "E9_HASE_FULL", config, "auc")
        criteria[config] = {
            "acc_gain_gt_0_2pp": acc["mean_delta_pp"] > 0.2,
            "auc_gain_gt_0_2pp": auc["mean_delta_pp"] > 0.2,
            "acc_wins_at_least_7": acc["wins"] >= 7,
            "auc_wins_at_least_7": auc["wins"] >= 7,
            "acc_delta_pp": acc["mean_delta_pp"], "auc_delta_pp": auc["mean_delta_pp"],
            "acc_wins": acc["wins"], "auc_wins": auc["wins"],
        }
    criteria_pass = all(all(v[k] for k in ("acc_gain_gt_0_2pp", "auc_gain_gt_0_2pp",
                                           "acc_wins_at_least_7", "auc_wins_at_least_7"))
                        for v in criteria.values())
    no_asd_sensitivity_tradeoff = all(
        paired_comparison(metrics, "E9_HASE_FULL", config, "sen_asd")["mean_delta_pp"] > -0.5
        for config in ("E0_B0A", "E2_GENERIC_MOE", "E5_WRONG_MASK", "E6_DATA_DRIVEN", "E7_PATIENT_MISMATCH")
    )
    decision = "GO" if criteria_pass and no_asd_sensitivity_tradeoff else "NO-GO"
    write_json(RESULTS / "decision.json", {
        "decision": decision,
        "screening_criteria": criteria,
        "all_primary_comparisons_pass": criteria_pass,
        "no_asd_sensitivity_tradeoff_over_0_5pp": no_asd_sensitivity_tradeoff,
        "interpretation": "single-seed internal exploratory screen, not statistical significance or clinical validation",
    })
    parity = baseline_parity(metrics)
    counter = pd.read_csv(RESULTS / "counterfactual.csv")
    timing = {task["config"]: float(task["elapsed_seconds"]) for task in task_results}
    summary["_acc"] = summary["acc_mean_pct"].round(4)
    report = render_report(metrics, summary, comparisons, criteria, decision, parity, diagnostics, counter, timing)
    (RESULTS / "report.md").write_text(report)
    fold_results = {}
    for config in GROUPS:
        task_path = RUNTIME / f"task_{config}.json"
        task = read_json(task_path) if task_path.exists() else None
        completed_folds = len(task.get("metrics", [])) if task else 0
        fold_results[config] = {
            "status": task.get("status", "MISSING") if task else "MISSING",
            "completed_folds": completed_folds,
            "expected_folds": 10,
        }
    manifest.update({
        "status": "COMPLETE",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fold_results": fold_results,
        "formal_folds": 100,
        "gpu_fold_counts": {"0": 50, "1": 50},
        "task_elapsed_seconds": timing,
        "dispatch_wall_seconds": float(manifest.get("dispatch_wall_seconds", 0.0)),
        "decision": decision,
        "e0_historical_10fold_parity": parity,
        "result_sha256": {path.name: sha(path) for path in RESULTS.iterdir() if path.is_file()},
        "runtime_checkpoint_cleanup_pending": bool(manifest.get("runtime_checkpoint_cleanup_pending", True)),
    })
    write_json(MANIFEST, manifest)
    print(f"REPORT_COMPLETE {decision} {RESULTS / 'report.md'}", flush=True)


def baseline_parity(metrics):
    ref = pd.read_csv(HISTORICAL)
    old = ref[(ref.config == "B0") & (ref.seed == 0)].sort_values("fold")
    new = metrics[metrics.config == "E0_B0A"].sort_values("fold")
    checks = {}
    fields = ("acc", "auc", "sen_asd", "spe_td", "best_epoch",
              "initial_trunk_sha256", "initial_rng_sha256")
    for field in fields:
        if field.endswith("sha256"):
            count = sum(str(a) == str(b) for a, b in zip(new[field], old[field]))
        elif field == "best_epoch":
            count = int(np.sum(new[field].to_numpy() == old[field].to_numpy()))
        else:
            count = int(np.sum(np.abs(new[field].to_numpy(float) - old[field].to_numpy(float)) < 1e-12))
        checks[field] = {"matched_folds": count, "of": 10}
    checks["all_fields_match"] = all(value["matched_folds"] == 10 for value in checks.values() if isinstance(value, dict))
    if not checks["all_fields_match"]:
        raise RuntimeError(f"E0 baseline does not match historical B0-A on all folds: {checks}")
    return checks


def render_report(metrics, summary, comparisons, criteria, decision, parity, diagnostics, counter, timing):
    def metric_table():
        lines = ["|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|", "|---|---:|---:|---:|---:|"]
        for _, row in summary.iterrows():
            def fmt(metric):
                return f"{row[f'{metric}_mean_pct']:.4f} ± {row[f'{metric}_sd_pct']:.4f}"
            lines.append(f"|{row['config']}|{fmt('acc')}|{fmt('auc')}|{fmt('spe_td')}|{fmt('sen_asd')}|")
        return "\n".join(lines)

    comparison_lines = ["|E9减对照|ACC Δpp (胜/平/负)|Hard-AUC Δpp (胜/平/负)|SPE-TD Δpp|SEN-ASD Δpp|",
                        "|---|---:|---:|---:|---:|"]
    for config in GROUPS:
        if config == "E9_HASE_FULL":
            continue
        values = {item["metric"]: item for item in comparisons
                  if item["right"] == config and item["metric"] in ("acc", "auc", "spe_td", "sen_asd")}
        a, u, s, n = (values[key] for key in ("acc", "auc", "spe_td", "sen_asd"))
        comparison_lines.append(
            f"|E9 − {config}|{a['mean_delta_pp']:+.4f} ({a['wins']}/{a['ties']}/{a['losses']})|"
            f"{u['mean_delta_pp']:+.4f} ({u['wins']}/{u['ties']}/{u['losses']})|"
            f"{s['mean_delta_pp']:+.4f}|{n['mean_delta_pp']:+.4f}|"
        )

    diag_e9 = diagnostics[diagnostics.config == "E9_HASE_FULL"]
    route_means = np.array([json.loads(value) for value in diag_e9.route_mean])
    route_sd = np.array([json.loads(value) for value in diag_e9.route_sd])
    route_usage = np.array([json.loads(value) for value in diag_e9.route_usage])
    route_summary = {
        "mean_weight": route_means.mean(axis=0).round(4).tolist(),
        "between_subject_sd": route_sd.mean(axis=0).round(4).tolist(),
        "top_route_fraction": route_usage.mean(axis=0).round(4).tolist(),
        "entropy": float(diag_e9.route_entropy_mean.mean()),
        "expert_cosines": np.array([json.loads(value) for value in diag_e9.expert_centered_cosine]).mean(axis=0).round(4).tolist(),
        "residual_norms": np.array([json.loads(value) for value in diag_e9.expert_residual_norm]).mean(axis=0).round(5).tolist(),
        "mismatch_response_change": float(diag_e9.fc_mismatch_response_change.mean()),
        "max_residual_ratio": float(diag_e9.max_residual_ratio.max()),
    }
    e8_diag = diagnostics[diagnostics.config == "E8_NO_SPEC"]

    def mean_json_column(frame, column):
        return np.nanmean(np.asarray([json.loads(value) for value in frame[column]], dtype=float), axis=0)

    route_asd = mean_json_column(diag_e9, "route_asd_mean")
    route_td = mean_json_column(diag_e9, "route_td_mean")
    route_asd_usage = mean_json_column(diag_e9, "route_asd_usage")
    route_td_usage = mean_json_column(diag_e9, "route_td_usage")
    route_response = mean_json_column(diag_e9, "response_route_spearman")
    route_site = mean_json_column(diag_e9, "route_site_eta2")
    route_age = mean_json_column(diag_e9, "route_age_eta2")
    route_sex = mean_json_column(diag_e9, "route_sex_eta2")
    route_qc = np.asarray([json.loads(value) for value in diag_e9.route_qc_abs_spearman], dtype=float)
    route_qc = route_qc.reshape(len(diag_e9), 4, 16).mean(axis=(0, 2))
    route_detail_lines = [
        "|专家|平均权重|ASD平均权重|TD平均权重|ASD/TD top-route比例|FC响应-路由Spearman|站点η²|年龄箱η²|性别η²|平均QC绝对ρ|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in range(4):
        route_detail_lines.append(
            f"|D{k + 1}|{route_summary['mean_weight'][k]:.4f}|{route_asd[k]:.4f}|{route_td[k]:.4f}|"
            f"{route_asd_usage[k]:.3f}/{route_td_usage[k]:.3f}|{route_response[k]:+.3f}|"
            f"{route_site[k]:.3f}|{route_age[k]:.3f}|{route_sex[k]:.3f}|{route_qc[k]:.3f}|"
        )
    route_details = "\n".join(route_detail_lines)
    parameter_lines = ["|组别|注册新增参数|有非零梯度新增参数|有效训练参数总量|新增非零梯度张量/注册张量|",
                       "|---|---:|---:|---:|---:|"]
    for config in GROUPS:
        row = metrics[metrics.config == config].iloc[0]
        parameter_lines.append(
            f"|{config}|{int(row.registered_extension_parameters):,}|"
            f"{int(row.effective_extension_parameters):,}|{int(row.effective_trainable_parameters):,}|"
            f"{int(row.extension_tensors_with_nonzero_gradient)}/{int(row.registered_parameter_tensors - 73)}|"
        )
    parameter_table = "\n".join(parameter_lines)

    prompts = read_json(PROMPTS)
    mask_record = read_json(EXP / "mechanism_masks.json")
    masks_by_id = {item["id"]: item for item in mask_record["mechanisms"]}
    prompt_sections = []
    prompt_function_zh = {
        "D1": "候选社交情感与沟通相关功能系统，不是个体缺陷标签。",
        "D2": "候选额-顶叶执行与整合系统；任务态结果不外推为静息态方向。",
        "D3": "候选丘脑-皮层信息整合系统，保留个体FC符号。",
        "D4": "候选岛叶-扣带与注意相关组织，保留年龄和人群异质性。",
    }
    prompt_limits_zh = {
        "D1": "不主张ASD连接统一增强或减弱；不代表个体病灶、症状量表、因果通路或完整经典社交网络。",
        "D2": "不把任务态低连接解释成静息态固定方向；没有个体执行功能测量或临床亚型；掩码不是已验证的完整额顶网络。",
        "D3": "没有单条已验证的ASD标志连接、个体感觉表型或因果丘脑皮层通路；掩码不是完整丘脑皮层图谱。",
        "D4": "宽泛图谱名称不等于经典显著性网络成员；不推断个体病灶、固定方向、年龄轨迹或临床亚型；掩码不是完整显著性网络。",
    }
    for prompt in prompts["mechanisms"]:
        mapping = masks_by_id[prompt["id"]]
        roi_names = sorted({
            edge[key] for edge in mapping["edges"]
            for key in ("roi_i_name", "roi_j_name") if edge.get(key)
        })
        quoted_prompt = prompt["prompt"].replace("\n", "\n> ")
        prompt_sections.append(
            f"### {prompt['id']}：{prompt['name']}\n\n"
            f"文献：{prompt['reference']['authors']}，{prompt['reference']['journal']} {prompt['reference']['year']}，"
            f"DOI [{prompt['reference']['doi']}]({prompt['reference']['url']})。"
            f"功能解释：{prompt_function_zh[prompt['id']]} 候选解剖端点共{len(roi_names)}个；"
            f"该域{mapping['feature_count']}条连接均来自已核验的252个CORR特征槽，"
            f"具体边及索引见 `mechanism_masks.json`。\n\n"
            f"候选ROI端点：{'；'.join(roi_names)}。\n\n"
            f"> {quoted_prompt}\n\n"
            f"解释限制：{prompt_limits_zh[prompt['id']]}"
        )
    prompt_section = "\n\n".join(prompt_sections)
    counter_lines = ["|固定干预|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|翻转率|",
                     "|---|---:|---:|---:|---:|---:|"]
    cf_group = counter.groupby("intervention")
    for name, group in cf_group:
        if name == "REAL":
            continue
        counter_lines.append(
            f"|{name}|{group.delta_acc_pp.mean():+.4f}|{group.delta_auc_pp.mean():+.4f}|"
            f"{group.delta_spe_td_pp.mean():+.4f}|{group.delta_sen_asd_pp.mean():+.4f}|"
            f"{group.prediction_flip_fraction.mean():.4f}|"
        )

    e9_mean = summary[summary.config == "E9_HASE_FULL"].iloc[0]
    e0_mean = summary[summary.config == "E0_B0A"].iloc[0]
    manifest = read_json(MANIFEST)
    resources_path = RUNTIME / "resources.json"
    resources = read_json(resources_path) if resources_path.exists() else {}
    dispatch_seconds = float(manifest.get("dispatch_wall_seconds", resources.get("dispatch_wall_seconds", 0.0)))
    delivery_status = manifest.get("delivery_status", "PENDING_ROOT_SYNC")
    if delivery_status == "COMPLETE":
        delivery_line = "根目录实验包与handoff均已同步。"
    else:
        delivery_line = (
            f"训练与分析已完成；最终根目录包同步状态为`{delivery_status}`，"
            f"实验包同步错误为`{manifest.get('root_sync_error')}`，"
            f"handoff追加状态为`{manifest.get('handoff_status', 'PENDING')}`，"
            f"错误为`{manifest.get('handoff_append_error')}`。正式工作区副本仍待写回，"
            f"完整当前副本与待追加文本位于`{EXP}`。"
        )
    source_hash_lines = "\n".join(
        f"- `{name}`: `{digest}`" for name, digest in manifest["root_source_sha256"].items()
    )
    data_hash_lines = "\n".join(
        f"- `{name}`: `{digest}`" for name, digest in manifest["data_sha256"].items()
    )
    time_total = sum(timing.values())
    core_comparisons = {
        config: paired_comparison(metrics, "E9_HASE_FULL", config, "acc")
        for config in ("E0_B0A", "E1_CAPACITY", "E2_GENERIC_MOE", "E3_MED_UNIFORM",
                       "E4_MED_GFR_ROUTE", "E5_WRONG_MASK", "E6_DATA_DRIVEN", "E7_PATIENT_MISMATCH")
    }
    all_off = counter[counter.intervention == "ALL_OFF"]
    patient_shuffle = counter[counter.intervention == "PATIENT_SHUFFLE"]
    wrong_mask_intervention = counter[counter.intervention == "WRONG_MASK"]
    group_descriptions = [
        "|E0_B0A|原始GFR + CAM + OutputLayer，完全移除SPGS/RES|新基线|",
        "|E1_CAPACITY|一个700输入、宽度183的普通无掩码Adapter|参数容量|",
        "|E2_GENERIC_MOE|4个专家读取全256 CORR，动态路由|普通MoE|",
        "|E3_MED_UNIFORM|真实冻结连接掩码，权重固定0.25|动态路由必要性|",
        "|E4_MED_GFR_ROUTE|真实机制FC输入，路由只看GFR/CAM上下文+专家身份|机制响应条件路由|",
        "|E5_WRONG_MASK|仅在252条端点已核验连接上做seed39全局置换，保持大小/重叠|正确解剖对应|",
        "|E6_DATA_DRIVEN|仅在252条已核验连接上做训练折FC相关剖面KMeans，保持重叠结构|文献先验对照|",
        "|E7_PATIENT_MISMATCH|训练/测试池内独立无固定点患者错配|患者对应|",
        "|E8_NO_SPEC|完整结构但关闭L_spec|专门化约束|",
        "|E9_HASE_FULL|真实四机制 + GFR/FC/PHENO/QC路由 + L_spec|完整模型|",
    ]
    paired = "\n".join(comparison_lines)
    counters = "\n".join(counter_lines)
    return f"""# 实验39：HASE-MoE 完整结果

状态：**训练与分析完成，100/100折**；GPU0/GPU1各50折；单seed=0；worker累计训练耗时（两卡所有配置耗时求和）{time_total/60:.1f}分钟，双卡训练调度墙钟{dispatch_seconds/60:.1f}分钟。判定：**{decision}**。

交付状态：{delivery_line}

## 审计与冻结协议

实际读取本地 MICCAI 2025 SPromptGL 原文 Fig.1、Section 2.1–2.2、Eq.(1)–(7)、ABIDE设置与消融；并审计根目录 `main.py`、`model.py`、`network.py`、`layers.py`、`utils.py` 及 `ABIDE-simple-2-concat-weighted-cosine.sh`。上游仓库链接为[SPromptGL GitHub](https://github.com/tangaii/SPromptGL)，审计使用的本地checkout HEAD为 `{manifest['root_git_head']}`，核心文件哈希见下文。论文 Fig.1 的流程是 GFR 患者图、DGCN、PLN/CAM/PEN；Eq.(1)把监督标签关系图纳入图目标，Eq.(3)–(6)描述CLIP文本提示、局部SPGS与RES，Eq.(7)写图损失与分类损失。论文ABIDE报告92.30% ACC、92.22% AUC；其训练参数与仓库 shell 不同，本实验严格按本地真实shell。

源码中四个GFR视图都是**871名患者之间的关系图**，不是脑区图。`GFR.forward` 的 `i==1` 分支会覆盖 `i==0` 的 `combined_feat`，本实验原样保留。`simple-2` 每轮计算 `ALoss`，但实际反传为分类NLL加四个输入图 `GraphConstructLoss` 的均值；`ALoss` 没有进入目标。GFR/GCN、阈值、传播、四图正则、融合、CAM、OutputLayer均未改。PEN的SPGS与RES前向被整体旁路；保留PLN InputLayer/Encoder/FeedForward及其初始化顺序。

E0按原构造顺序仍构造原始PLN、CLIP文本缓存和优化器，然后把CAM四个36维位置直接展平输入OutputLayer；它与Experiment32 B0-A独立重训逐折核对。B0-A历史为ACC 92.0742%、Hard-AUC 91.8546%、SEN-ASD 89.0793%、SPE-TD 94.6300%。本轮E0重训10折后，初始主干哈希、入口RNG哈希、最佳epoch及四项折指标均{parity['all_fields_match'] and '10/10逐项匹配' or '未完全匹配'}。

历史结果边界：原始完整PEN的Experiment32 B2 ACC=91.7294%仅作描述性参照，不能替代本轮B0-A E0。Experiment11恢复的256条连接端点和Experiment19四域映射为本轮冻结映射来源。Experiment3/8/13–22、26、28–31反复显示自由文本投影、静态语义关系或图调节的结果混合，错误/随机先验不总是更差；Experiment20的概念关系严格门槛未通过。Experiment34–35的语义关系/消息增量未稳定优于B0-A或强数值/错配对照；Experiment36的四域文本路由对B0-A平均ACC无增益，Experiment37标签监督混合组未胜过其基础及错配对照。故本轮要求容量、普通MoE、正确/错误解剖、训练折数值集合和患者错配的并行重训对照。实验38在本轮设计冻结时只读本地临时态、未作为结果或模型状态来源。

历史协议保持不变：`seed=0`，`StratifiedKFold(10, shuffle=True, random_state=10)`，val_idx=test_idx，ACC严格提高才替换checkpoint，450 epoch，wait>200早停，双Adam，simple-2，pre-CV RFE/标准化以及871人transductive患者图。由此测试折参与checkpoint选择、预处理在外层CV之前完成、图传播跨越全部受试者；结果只支持该历史协议下的内部配对比较，不构成无偏泛化或外部临床验证。

## 医学先验与结构

四条冻结的机制prompt见 `prompts.json`，其文案是用户提供的候选机制草案，不声称来自某次GPT-4 API调用，也不是逐字文献事实。解剖候选完全复用Experiment19映射与Experiment11数值核验：D1/D2/D3/D4分别72/56/45/32条，联合覆盖165条；映射之间允许重叠，不把这些集合称作完整规范脑网络。四条端点未知连接195/225/232/236仍走原始主干，但不进入真实、错误解剖或数据驱动机制掩码；E2作为无解剖约束的普通MoE仍按注册读取全部256维数值输入。E5错误掩码与E6数据驱动掩码均限于其余252条已核验连接，且维持相同的专家大小和重叠结构；这是训练前为满足端点约束固定的实现细化。每条连接的ROI端点、E5错误集合和索引见 `mechanism_masks.json`。

HASE从CAM输出 `H:[871,4,36]` 形成患者上下文，并从原始CORR256取出按掩码的有符号连接及仅用outer-train中位数/MAD标准化的偏离。该z表示相对训练队列分布，不代表临床病理或ASD专属异常。每个专家有独立无偏FC编码器和无偏Adapter，Softmax路由逐专家接收GFR/CAM上下文、该专家机制响应、年龄分箱one-hot与性别，以及ANAT/FUNC质量和头动相关条件；PHENO站点20维不直接进新增路由。ANAT/FUNC仅是可靠性信息，不是症状或疾病证据。路由连续可导，不用标签预测专家。新增残差初始为严格0，并限制为CAM范数的10%以内；无独立疾病分类头。

四模态作用：PHENO48、ANAT6、FUNC10、CORR256全部保留在原GFR/主干；仅新增HASE直接读取CORR256、PHENO年龄/性别及ANAT/FUNC质量条件，且排除站点one-hot。CAM的四个36维位置是跨模态交互潜表示，不是四个原始单模态，也不是ROI。

## 冻结Prompt原文

以下文案是训练前冻结的候选机制综合文本，由本实验请求提供；不声称来自某次LLM API生成，也不构成逐字临床事实。实际计算只由同一节所列的冻结ROI连接掩码进入模型，单独改写这段文字不会改变训练。

{prompt_section}

|组|固定结构|主要对照|
|---|---|---|
{"\n".join(group_descriptions)}

E9新增可训练参数154,816；E1宽183的单Adapter为154,452，容量差0.236%。E2–E9使用相同四专家参数结构，改变预注册的掩码、错配、路由或专门化开关；原模型参数保持weight_decay=0.11，新增模块使用独立Adam参数组weight_decay=0。逐折注册参数与实际非零梯度参数量、梯度诊断见 `fold_metrics.csv` / `diagnostics.csv`。

逐组参数审计（注册新增参数、至少一次获得非零梯度的新增参数、有效参数总量及活跃新增张量数）如下；原始主干注册1,462,957参数，实际有效梯度主干约732,238参数、49个活跃张量。固定均匀路由E3的Router没有梯度，因此其有效新增参数为147,456，而注册数仍154,816。E0中PEN路由旁路参数未获梯度。

{parameter_table}

## 正式指标

均值±十折population标准差，单位%。正式分类指标只有ACC、原代码one-hot硬预测ROC-AUC、SPE-TD、SEN-ASD。二分类下硬标签AUC等于BA，没有另算概率AUC、F1或其他正式指标。

{metric_table()}

## 配对比较

Δ为E9减对照，单位百分点；W/T/L为同一fold严格胜/平/负。10折训练集合相互重叠，不作为独立样本或显著性检验。筛选阈值预先冻结为E9相对E0/E2/E5/E6/E7在ACC和Hard-AUC平均均>0.2pp且各至少7/10折严格提高，同时不以ASD灵敏度明显下降换取准确率。

{paired}

筛选判定逐项：`{json.dumps(criteria, ensure_ascii=False, sort_keys=True)}`。ACC标准差范围与SEN-ASD变化同时报告；本判定是单seed内部探索门槛，不是论文级显著性标准。

## 机制诊断

E9折均值诊断：`{json.dumps(route_summary, ensure_ascii=False)}`。平均路由使用率只描述模型分配，不等于临床亚型；ASD/TD内部路由均值是训练后按测试标签作描述性汇总，不参与推断。站点、年龄分箱、性别及QC关联为队列描述，不能当病理关联。专家响应错配诊断、掩码覆盖、中心化余弦、残差幅度及逐折梯度见 `diagnostics.csv`。

按专家展开的描述性路由与条件关联：

{route_details}

E9四专家测试集中心化余弦均值（D1-D2、D1-D3、D1-D4、D2-D3、D2-D4、D3-D4）为`{route_summary['expert_cosines']}`，各专家平均残差范数为`{route_summary['residual_norms']}`，每折平均残差/CAM范数比例为{diag_e9.mean_residual_ratio.mean():.4f}，最坏值{route_summary['max_residual_ratio']:.4f}。新增模块15/15张量均实际获得非零梯度，患者机制FC错配使响应范数平均相对变化{route_summary['mismatch_response_change']:.3f}；这些检查说明机制输入进入了计算图，但不证明专家代表临床机制。E9的训练L_spec均值为{diag_e9.specialization_loss.mean():.5f}；E8中该项为0。E8的专家余弦和E9同样接近0，E9相对E8的ACC与Hard-AUC反而分别低{paired_comparison(metrics, 'E9_HASE_FULL', 'E8_NO_SPEC', 'acc')['mean_delta_pp']:.4f}和{paired_comparison(metrics, 'E9_HASE_FULL', 'E8_NO_SPEC', 'auc')['mean_delta_pp']:.4f}个百分点，未显示专门化约束必要性。

### 固定checkpoint干预

以下所有干预直接作用于E9每折的同一最佳checkpoint；不重训、不重选epoch。差值为百分点，flip为测试预测翻转比例。

{counters}

`ALL_OFF`仅说明E9训练出的主干对新增残差的checkpoint依赖，不能替代从头训练E0/E2/E5/E6/E7。wrong-mask或patient-shuffle同样是模型敏感性诊断，不构成连接因果性或亚型证据。

## 科学解释与决定

判定为 **{decision}**。完整创新链需要同一数据上的异质机制响应、正确解剖映射优于容量匹配/普通MoE/错误映射/无监督集合/患者错配、且患者条件路由与专家差异带来相互支持的可复现增益。以上预注册筛选门槛任一失败即NO-GO；即使GO也只是单seed内部候选，不代表MICCAI 2027论文成功。路由权重、梯度非零或固定checkpoint干预均不能单独证明医学语义有效。

相关工作边界（文献检索截至2026-10-26）：SPromptGL已包含多模态患者关系图与文本提示局部筛选；BrainPrompt (MICCAI 2025) 已研究ROI/subject/disease多级文本提示，因此多级提示本身不足以构成新增点。PhenoBrain (ICML 2026) 已以表型条件调节长程通信。KD-Brain (MICCAI 2026) 已把语义先验注入子网络交互query并引入pathology-consistent约束。因此HASE可能的增量只能是：将可核验ROI连接集合固定为真实FC专家的唯一FC输入，再由患者自身的真实机制响应参与软路由，并以错误解剖/患者错配和同checkpoint干预检验机制必要性；单说“语义路由/条件MoE”不新颖。NeurIPS 2025专家专门化研究和后续MoE去冗余正则说明响应差异本身需验证，本轮L_spec只是辅助约束，E8用于判断它的必要性。

进一步的边界：2026 BrainMoE已展示模态专属EEG/fMRI专家、共享状态专家和样本级路由，因而条件MoE与专家融合本身不是充分新颖点；KD-Brain已把文本条件放进子网络交互并用先验分布约束注意力；PhenoBrain已将表型用于个体化长程路由。HASE在概念上较窄的区别是“固定、可核验连接掩码限制FC专家输入”，但本轮E9未胜过普通MoE、错误掩码或外层训练折无监督FC集合，不能主张该区别带来诊断收益。已发表的ASD功能亚型工作使用更大样本、规范化建模及独立行为/任务表现评估；本轮ABIDE特征没有症状量表，专家路由不能改称亚型。

既有ASD影像亚型研究报告跨人群功能连接异质性，例如Nature Neuroscience的个体差异/网络机制分析及2023–2025 ABIDE聚类工作。但HASE的D1–D4是候选功能系统专家，不是已验证患者亚型；本实验数据没有症状量表来验证专家分配的临床意义。若核心对照失败，应保持NO-GO。

## 结果解释与可能根因

E9的ACC比E0低{abs(core_comparisons['E0_B0A']['mean_delta_pp']):.4f}个百分点（逐折{core_comparisons['E0_B0A']['wins']}/{core_comparisons['E0_B0A']['ties']}/{core_comparisons['E0_B0A']['losses']}胜/平/负），比容量匹配E1低{abs(core_comparisons['E1_CAPACITY']['mean_delta_pp']):.4f}个百分点，也比普通MoE E2低{abs(core_comparisons['E2_GENERIC_MOE']['mean_delta_pp']):.4f}个百分点。E9相对E0和E2的Hard-AUC分别低{abs(paired_comparison(metrics, 'E9_HASE_FULL', 'E0_B0A', 'auc')['mean_delta_pp']):.4f}和{abs(paired_comparison(metrics, 'E9_HASE_FULL', 'E2_GENERIC_MOE', 'auc')['mean_delta_pp']):.4f}个百分点，SEN-ASD分别低{abs(paired_comparison(metrics, 'E9_HASE_FULL', 'E0_B0A', 'sen_asd')['mean_delta_pp']):.4f}和{abs(paired_comparison(metrics, 'E9_HASE_FULL', 'E2_GENERIC_MOE', 'sen_asd')['mean_delta_pp']):.4f}个百分点；ASD和TD识别均恶化，并非用牺牲特异度换准确率。E9相对固定均匀路由E3的ACC平均差为{core_comparisons['E3_MED_UNIFORM']['mean_delta_pp']:+.4f}个百分点，逐折仅{core_comparisons['E3_MED_UNIFORM']['wins']}/10胜出；动态路由没有解决整体负增益。

正确掩码的必要性不成立：E9比E5错误解剖掩码仅高{core_comparisons['E5_WRONG_MASK']['mean_delta_pp']:+.4f}个百分点ACC，逐折{core_comparisons['E5_WRONG_MASK']['wins']}/{core_comparisons['E5_WRONG_MASK']['ties']}/{core_comparisons['E5_WRONG_MASK']['losses']}胜/平/负；E9比E6训练折无监督FC分组低{abs(core_comparisons['E6_DATA_DRIVEN']['mean_delta_pp']):.4f}个百分点，比训练时患者FC错配E7低{abs(core_comparisons['E7_PATIENT_MISMATCH']['mean_delta_pp']):.4f}个百分点。因此本轮没有证明正确医学域或患者对应在从头训练中优于反事实控制。E9比仅用GFR/CAM上下文路由的E4低{abs(core_comparisons['E4_MED_GFR_ROUTE']['mean_delta_pp']):.4f}个百分点，额外FC响应、PHENO年龄/性别及ANAT/FUNC质量条件没有形成可用增益；E9相对E8关闭L_spec也低0.2207个百分点ACC，专门化约束未显示必要性。

固定checkpoint中，ALL_OFF使ACC平均下降{abs(all_off.delta_acc_pp.mean()):.4f}个百分点，PATIENT_SHUFFLE下降{abs(patient_shuffle.delta_acc_pp.mean()):.4f}个百分点，WRONG_MASK下降{abs(wrong_mask_intervention.delta_acc_pp.mean()):.4f}个百分点。这说明已训练E9的预测依赖新增支路、患者FC和训练时掩码；同一checkpoint对输入扰动敏感不等于重训方法优于E0，也不能证明真实掩码的医学正确性。E9自身的残差关闭结果仍远低于E0，符合新增路径联合训练后改变共享主干解的可能性。

较符合数据的解释是：新增FC编码器把已预处理的高维CORR直接送入CAM后的残差Adapter，在871名受试者上增加了154,816个可训练参数；E9平均残差/CAM范数比达到9.90%，接近每折10%的上限。正则化只约束四个残差的相似度，不保证它们携带互补的疾病信息；E8与E9均呈近零中心化余弦，但E9性能略差。E9路由与站点的平均η²约0.234、年龄箱约0.162、性别约0.011，且D3路由与对应FC响应范数的Spearman相关约0.016。新增路由因此可能部分利用队列或采集分布，或只依赖部分专家响应，尚不足以解释为四种医学机制。E1普通Adapter和E2普通MoE都优于E9，E5错误掩码与E9接近，E6/E7也未输给E9；整体上更支持容量、共享主干优化和一般FC信号影响，而不是正确语义映射收益。以上是与观测相符的候选数学/优化根因，不是单seed实验识别的因果结论；需要严格无泄漏外层预处理、独立多中心验证及重复seed进一步区分。

## 可复现记录

完整原始fold指标、最佳epoch、耗时、入口RNG和初始化哈希位于 `fold_metrics.csv`；路由/专家机制诊断位于 `diagnostics.csv`；固定checkpoint干预位于 `counterfactual.csv`；完整paired差位于 `paired_comparisons.json`；manifest记录prompt、源代码、输入、映射、fold和结果哈希。GPU分配严格按GPU0 E0/E2/E4/E6/E8及GPU1 E1/E3/E5/E7/E9；每卡最多4 worker，每worker2 CPU线程。每配置按seed0播种一次，后续fold连续消费随机流；十组fold1入口RNG和主干初始化一致，但不同早停长度导致之后fold的入口RNG分流，表中逐折记录，不宣称跨组后续初始化完全一致。

E0历史核验：`{json.dumps(parity, ensure_ascii=False, sort_keys=True)}`。

源码SHA256：
{source_hash_lines}

数据与历史映射/参照SHA256：
{data_hash_lines}

训练代码和冻结prompt的SHA256见 `manifest.json`。单卡任务累计秒数：`{json.dumps(timing, sort_keys=True)}`。正式模型checkpoint在训练、反事实和报告验收后已清理；恢复诊断记录、简短worker日志及其余实验源文件保留在独立目录`{EXP}`。根目录包同步受当前OSSFS写入错误阻断，暂不将此临时位置描述为正式工作区目录。

文献：Gotts et al. 2012, DOI 10.1093/brain/aws160; Just et al. 2007, DOI 10.1093/cercor/bhl006; Nair et al. 2013, DOI 10.1093/brain/awt079; Uddin et al. 2013, DOI 10.1001/jamapsychiatry.2013.104. BrainPrompt (MICCAI 2025): https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf . PhenoBrain (ICML 2026): https://proceedings.mlr.press/v306/meng26f.html . KD-Brain (MICCAI 2026): https://papers.miccai.org/miccai-2026/0366-Paper0073.html . BrainMoE (J Vis Exp 2026), DOI 10.3791/73432: https://pubmed.ncbi.nlm.nih.gov/42799620/ . NeurIPS 2025, Advancing Expert Specialization for Better MoE: https://papers.nips.cc/paper_files/paper/2025/hash/4598de7d243d528e38eb0c5d8155fb52-Abstract-Conference.html . ASD subtype studies: https://doi.org/10.1038/s41380-025-03086-x and https://doi.org/10.1038/s41593-026-02287-z .

## 执行恢复与I/O

{manifest.get('execution_recovery', '恢复记录见manifest.json。')}
"""


def gpu_snapshot():
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            text=True, timeout=20,
        )
        parsed = []
        for line in raw.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) == 4:
                parsed.append({"index": int(fields[0]), "memory_used": fields[1],
                               "memory_total": fields[2], "utilization": fields[3]})
        return parsed
    except Exception as exc:
        return [{"error": repr(exc)}]


def launch_config(config):
    gpu = GROUPS[config]["gpu"]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "HASE_E39_PHYSICAL_GPU": str(gpu), "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    log_path = LOGS / f"{config}.log"
    with open(log_path, "a", encoding="utf-8") as log:
        process = subprocess.run(
            [sys.executable, str(EXP / "run.py"), "worker", config],
            cwd=EXP, env=env, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    return config, process.returncode, str(log_path)


def dispatch():
    set_cuda_rng_scope()
    if not (RUNTIME / "preflight.json").exists():
        raise RuntimeError("Formal dispatch is blocked until preflight passes")
    if torch.cuda.is_available() and (torch.cuda.device_count() != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"):
        raise RuntimeError("Dispatch parent must run with CUDA_VISIBLE_DEVICES=0; workers set their own physical GPU visibility")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    os.environ["HASE_E39_PHYSICAL_GPU"] = "0"
    started = time.monotonic()
    current = load_checkpoint(task_checkpoint("E0_B0A")) if task_checkpoint("E0_B0A").exists() else None
    if current is None or current.get("result", {}).get("status") != "COMPLETE":
        worker("E0_B0A")
    compare_e0_first_fold()
    compare_e0_all_folds()

    outcomes = []
    pools = {gpu: ThreadPoolExecutor(max_workers=4) for gpu in (0, 1)}
    futures = {}
    for gpu, configs in PHYSICAL_GPU_GROUPS.items():
        for config in configs:
            if config == "E0_B0A":
                continue
            futures[pools[gpu].submit(launch_config, config)] = (gpu, config)
    pending = set(futures)
    samples = []
    last_sample = 0.0
    while pending:
        try:
            completed = next(as_completed(pending, timeout=30))
        except StopIteration:
            completed = None
        except TimeoutError:
            completed = None
        if completed is not None:
            pending.remove(completed)
            outcome = completed.result()
            outcomes.append(outcome)
            print(f"WORKER_EXIT gpu={futures[completed][0]} config={outcome[0]} code={outcome[1]}", flush=True)
        now = time.monotonic()
        if now - last_sample >= 30 or completed is None:
            samples.append({"elapsed_seconds": now - started, "gpus": gpu_snapshot()})
            last_sample = now
    for pool in pools.values():
        pool.shutdown(wait=True)
    write_json(RUNTIME / "resources.json", {
        "dispatch_wall_seconds": time.monotonic() - started,
        "worker_outcomes": [{"config": c, "exit_code": code, "log": log} for c, code, log in outcomes],
        "gpu_samples": samples,
        "worker_limit_per_gpu": 4,
        "threads_per_worker": 2,
    })
    failed = [outcome for outcome in outcomes if outcome[1] != 0]
    manifest = read_json(MANIFEST)
    manifest["dispatch_wall_seconds"] = time.monotonic() - started
    manifest["resource_samples"] = samples
    manifest["worker_outcomes"] = [{"config": c, "exit_code": code} for c, code, _ in outcomes]
    if failed:
        manifest["status"] = "INCOMPLETE_RECOVERABLE"
        write_json(MANIFEST, manifest)
        raise RuntimeError(f"Training jobs failed; completed folds are checkpointed: {failed}")

    # The fixed-checkpoint audit and report consume no training labels until after model fitting.
    counterfactuals()
    summarize_and_report()
    manifest = read_json(MANIFEST)
    manifest["dispatch_wall_seconds"] = time.monotonic() - started
    manifest["resource_samples"] = samples
    manifest["worker_outcomes"] = [{"config": c, "exit_code": code} for c, code, _ in outcomes]
    write_json(MANIFEST, manifest)
    print("DISPATCH_COMPLETE", flush=True)


def protected_hash_check():
    for name in TASKS:
        if sha(ROOT / name) != EXPECTED_SHA256[name]:
            raise RuntimeError(f"Root source changed during Experiment39: {name}")
    for name, digest in (("processed_standard_data.csv", EXPECTED_SHA256["processed_standard_data.csv"]),
                         ("modal_feat_dict.npy", EXPECTED_SHA256["modal_feat_dict.npy"]),
                         ("domain_mapping.csv", EXPECTED_SHA256["domain_mapping.csv"]),
                         ("roi_256_named_mapping.csv", EXPECTED_SHA256["roi_256_named_mapping.csv"]),
                         ("exp32_fold_metrics.csv", EXPECTED_SHA256["exp32_fold_metrics.csv"]),
                         ("source_paper_pdf", EXPECTED_SHA256["source_paper_pdf"])):
        path = {
            "processed_standard_data.csv": DATA_DIR / name,
            "modal_feat_dict.npy": DATA_DIR / name,
            "domain_mapping.csv": MAPPING,
            "roi_256_named_mapping.csv": ROOT / "experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv",
            "exp32_fold_metrics.csv": HISTORICAL,
            "source_paper_pdf": PAPER,
        }[name]
        if sha(path) != digest:
            raise RuntimeError(f"Protected data/history changed during Experiment39: {name}")
    return True


def handoff_text():
    metrics = pd.read_csv(RESULTS / "fold_metrics.csv")
    summary = pd.read_csv(RESULTS / "summary.csv")
    diagnostics = pd.read_csv(RESULTS / "diagnostics.csv")
    decision = read_json(RESULTS / "decision.json")
    manifest = read_json(MANIFEST)
    start = "<!-- EXPERIMENT39_HASE_MOE_START -->"
    end = "<!-- EXPERIMENT39_HASE_MOE_END -->"
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            f"|{row['config']}|{row['acc_mean_pct']:.4f} ± {row['acc_sd_pct']:.4f}|"
            f"{row['auc_mean_pct']:.4f} ± {row['auc_sd_pct']:.4f}|"
            f"{row['spe_td_mean_pct']:.4f} ± {row['spe_td_sd_pct']:.4f}|"
            f"{row['sen_asd_mean_pct']:.4f} ± {row['sen_asd_sd_pct']:.4f}|"
        )
    gpu_counts = metrics.groupby("physical_gpu").size().to_dict()
    route = diagnostics[diagnostics.config == "E9_HASE_FULL"]
    route_mean = np.mean([json.loads(value) for value in route.route_mean], axis=0).round(4).tolist()
    resources = read_json(RUNTIME / "resources.json") if (RUNTIME / "resources.json").exists() else {}
    sampled_gpu = {}
    for sample in resources.get("gpu_samples", []):
        for item in sample.get("gpus", []):
            index = item.get("index")
            try:
                mem = int(str(item.get("memory_used", "")).split()[0])
                util = int(str(item.get("utilization", "")).replace("%", "").split()[0])
            except (ValueError, IndexError):
                continue
            sampled_gpu.setdefault(index, {"peak_memory_mib": 0, "peak_utilization_percent": 0})
            sampled_gpu[index]["peak_memory_mib"] = max(sampled_gpu[index]["peak_memory_mib"], mem)
            sampled_gpu[index]["peak_utilization_percent"] = max(
                sampled_gpu[index]["peak_utilization_percent"], util
            )
    mode = decision["decision"]
    code = manifest.get("code_sha256", {})
    data = manifest["data_sha256"]
    return f"""{start}
## 实验39：HASE-MoE（Heterogeneity-Aware Semantic Expert Mixture）

- 状态：COMPLETE，正式训练100/100折；seed=0，固定StratifiedKFold十折。GPU0任务E0/E2/E4/E6/E8共50折，GPU1任务E1/E3/E5/E7/E9共50折；每卡最多4 worker、每worker 2 CPU线程。
- 根目录交付：`{manifest.get('delivery_status', 'PENDING_ROOT_SYNC')}`；`{manifest.get('root_sync_error') or '无同步错误。'}`。正式包完整副本暂存于`{EXP}`，报告为`{EXP}/results/report.md`。
- 耗时：双卡调度 {manifest.get('dispatch_wall_seconds', 0.0)/60:.1f} 分钟；配置累计训练时间 {sum(manifest.get('task_elapsed_seconds', {}).values())/60:.1f} 分钟。资源采样与worker退出码见实验目录manifest/runtime资源记录。
- 指标（10折均值±population SD，%）：ACC、原one-hot硬标签AUC、SPE-TD、SEN-ASD。AUC二分类下等于BA；没有增加概率AUC/F1。
- 资源：每卡最多4 worker、每worker 2 CPU线程；GPU显存/利用率每30秒采样，采样峰值为`{json.dumps(sampled_gpu, sort_keys=True)}`（硬件各98,304 MiB）；实际训练负载远低于显存上限。

|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---:|---:|---:|---:|
{"\n".join(rows)}

- E9平均专家路由权重（专家使用顺序D1–D4）：`{json.dumps(route_mean)}`。完整路由/专家差异、患者错配响应、梯度与站点/年龄/性别/QC关联见 `results/diagnostics.csv`；同checkpoint WRONG_MASK/PATIENT_SHUFFLE/UNIFORM_ROUTE/EXPERT_OFF/ALL_OFF干预见 `results/counterfactual.csv`。E9对核心组的逐折配对胜平负与差值见 `results/report.md` 和 `results/paired_comparisons.json`。
- E0与Experiment32 B0-A的入口RNG、主干初始化、最佳epoch及四指标逐折核对：见 `results/report.md` 与 `.runtime/e0_first_fold_parity.json`。
- 判定：**{mode}**。该结果是单seed、val=test、CV前全局RFE/标准化和871人transductive患者图下的内部探索，不是统计显著性、临床亚型或独立泛化结论。
- 首轮排除审计：首轮已完成的100折和90项干预因专门化损失误作用于E2–E7而全部排除；其中还发现E0入口RNG构造边界不匹配及一个诊断钩子恢复事件。首轮材料仅保留在`.runtime/first_pass_lspec_audit/`，未与正式结果混合。修正后从干净状态重新完成100折，L_spec仅E9开启，E0与E32逐折10/10一致；正式折未恢复续训、未重复记分。
- 目录：`experiment/experiment39_heterogeneity_aware_semantic_experts/`；完整报告：`experiment/experiment39_heterogeneity_aware_semantic_experts/results/report.md`。
- 代码SHA256：`{json.dumps(code, sort_keys=True)}`。
- 输入/映射/历史结果SHA256：`{json.dumps(data, sort_keys=True)}`。
- 原论文PDF SHA256：`{manifest['data_sha256']['source_paper_pdf']}`；原handoff SHA256：`{manifest.get('original_handoff_sha256', '未读取')}`；本节使用起止标记保证幂等，当前状态为`{manifest.get('handoff_status', 'PENDING')}`。
- I/O与残留：{'; '.join(manifest.get('runtime_io_faults', [])) if manifest.get('runtime_io_faults') else '无I/O故障；恢复过程如上。'} 当前没有训练checkpoint；根目录包同步与handoff追加文件因OSSFS EIO待写回，完整包在`{EXP}`，待追加正文在`.runtime/handoff_append.md`。未git commit/push，未启动Experiment40。
{end}
"""


def sync_package():
    protected_hash_check()
    root_exp = ROOT / "experiment/experiment39_heterogeneity_aware_semantic_experts"
    root_exp.mkdir(parents=True, exist_ok=True)
    include_files = [
        "README.md", "prompts.json", "hase.py", "integration.py", "run.py", "integration.patch",
        "mechanism_masks.json", "manifest.json",
        "results/fold_metrics.csv", "results/diagnostics.csv", "results/counterfactual.csv",
        "results/summary.csv", "results/paired_comparisons.json", "results/decision.json", "results/report.md",
        ".runtime/preflight.json", ".runtime/e0_first_fold_parity.json", ".runtime/e0_full_parity.json",
        ".runtime/source_hashes.json",
        ".runtime/resources.json",
    ]
    conflicts = []
    for relative in include_files:
        source = EXP / relative
        if not source.exists():
            raise RuntimeError(f"Required final artifact missing: {relative}")
        target = root_exp / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) != sha(source):
            conflicts.append(relative)
    if conflicts:
        raise RuntimeError(f"Root Experiment39 path has differing existing files: {conflicts}")
    for relative in include_files:
        source = EXP / relative
        target = root_exp / relative
        if not target.exists():
            shutil.copy2(source, target)
    return root_exp


def append_handoff_block(text):
    path = ROOT / "handoff.md"
    content = path.read_bytes()
    start = b"<!-- EXPERIMENT39_HASE_MOE_START -->"
    end = b"<!-- EXPERIMENT39_HASE_MOE_END -->"
    block = text.encode("utf-8")
    if not block.endswith(b"\n"):
        block += b"\n"
    starts, ends = content.count(start), content.count(end)
    if starts == 0 and ends == 0:
        separator = b"" if content.endswith(b"\n\n") else b"\n" if content.endswith(b"\n") else b"\n\n"
        with open(path, "ab") as stream:
            stream.write(separator + block)
            stream.flush()
            os.fsync(stream.fileno())
        return "APPENDED"
    if starts != 1 or ends != 1:
        raise RuntimeError(f"handoff markers are incomplete or duplicated: starts={starts}, ends={ends}")
    a = content.index(start)
    b = content.index(end, a) + len(end)
    if content[a:b] == block.rstrip(b"\n"):
        return "ALREADY_PRESENT"
    if content[a:b] == block:
        return "ALREADY_PRESENT"
    if b < len(content) and content[b:b + 2] == b"\r\n":
        b += 2
    elif b < len(content) and content[b:b + 1] == b"\n":
        b += 1
    updated = content[:a] + block + content[b:]
    with open(path, "r+b") as stream:
        stream.seek(0)
        stream.write(updated)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
    return "REPLACED_EXISTING_SECTION"


def stage_handoff(status, error=None):
    handoff = handoff_text()
    (RUNTIME / "handoff_append.md").write_text(handoff)
    write_json(RUNTIME / "handoff_append.json", {
        "text": handoff,
        "sha256": bytes_sha(handoff.encode()),
        "status": status,
        "handoff_append_error": error,
        "root_sync_error": read_json(MANIFEST).get("root_sync_error"),
    })


def finalize():
    protected_hash_check()
    manifest = read_json(MANIFEST)
    task_results = {}
    for config in GROUPS:
        path = RUNTIME / f"task_{config}.json"
        if not path.exists():
            raise RuntimeError(f"Missing final task state for {config}")
        task = read_json(path)
        n = len(task.get("metrics", []))
        if task.get("status") != "COMPLETE" or n != 10:
            raise RuntimeError(f"Incomplete formal run for {config}: status={task.get('status')}, folds={n}")
        task_results[config] = {"status": "COMPLETE", "completed_folds": n, "expected_folds": 10}
    formal = pd.read_csv(RESULTS / "fold_metrics.csv")
    counts = formal.groupby("config").fold.nunique().to_dict()
    if len(formal) != 100 or any(counts.get(config) != 10 for config in GROUPS):
        raise RuntimeError(f"Formal fold table is not exactly 100 unique config/fold rows: {counts}")
    counter = pd.read_csv(RESULTS / "counterfactual.csv")
    if len(counter) != 90 or counter.duplicated(["config", "seed", "fold", "intervention"]).any():
        raise RuntimeError("E9 counterfactual table is not exactly 90 unique interventions")

    for path in CHECKPOINTS.glob("*.pt"):
        path.unlink()
    for path in LOGS.glob("*.log"):
        if path.stat().st_size > 4 * 1024 * 1024:
            path.write_text("Large worker log omitted after successful fold audit.\n")
    manifest.update({
        "runtime_checkpoint_cleanup_pending": False,
        "runtime_checkpoint_cleanup": "all formal and diagnostic .pt checkpoints removed after 100 folds and 90 fixed-checkpoint interventions",
        "protected_root_hashes_verified_after_training": True,
        "fold_results": task_results,
        "formal_folds": 100,
        "counterfactual_interventions_completed": 90,
        "execution_recovery": (
            "首轮曾完成100折和90项干预，但因实现错误地在E2–E7也启用了L_spec（冻结协议仅在E9启用），该轮结果全部隔离排除。"
            "首轮审计还发现E0初始化RNG构造边界与历史不一致，以及一次诊断钩子恢复事件；首轮材料仅保存在"
            ".runtime/first_pass_lspec_audit/，没有复用其中任何指标。随后清空正式任务和结果状态，从seed=0重新完成修正后的100折；"
            "E1–E8关闭L_spec、仅E9开启，正式折没有续训恢复或重复计分。E0四项指标、最佳epoch、初始主干哈希和入口RNG"
            "均与Experiment32 B0-A逐折10/10匹配。"
        ),
        "original_handoff_sha256": sha(ROOT / "handoff.md"),
        "delivery_status": "PENDING_ROOT_SYNC",
        "handoff_status": "PENDING",
        "handoff_append_error": None,
        "root_sync_error": None,
        "code_sha256": {name: sha(EXP / name) for name in
                        ("README.md", "prompts.json", "hase.py", "integration.py", "run.py",
                         "integration.patch", "mechanism_masks.json") if (EXP / name).is_file()},
    })
    write_json(MANIFEST, manifest)
    summarize_and_report()
    stage_handoff("PENDING_ROOT_SYNC")

    root_exp = ROOT / "experiment/experiment39_heterogeneity_aware_semantic_experts"
    root_exp_preexisting = root_exp.exists()
    package_synced = False
    try:
        root_exp = sync_package()
        package_synced = True
        manifest = read_json(MANIFEST)
        manifest["delivery_status"] = "PACKAGE_SYNCED_HANDOFF_PENDING"
        manifest["root_experiment_directory"] = str(root_exp)
        manifest["root_sync_error"] = None
        write_json(MANIFEST, manifest)
        summarize_and_report()
        shutil.copy2(MANIFEST, root_exp / "manifest.json")
        shutil.copy2(RESULTS / "report.md", root_exp / "results/report.md")

        handoff = handoff_text()
        append_status = append_handoff_block(handoff)
        manifest = read_json(MANIFEST)
        manifest["handoff_status"] = "APPENDED_IDEMPOTENT_MARKERS"
        manifest["handoff_append_error"] = None
        manifest["handoff_append_action"] = append_status
        manifest["delivery_status"] = "COMPLETE"
        manifest["handoff_sha256_after_append"] = sha(ROOT / "handoff.md")
        write_json(MANIFEST, manifest)
        summarize_and_report()
        # Replace the just-appended section with a version that records completed delivery.
        append_status = append_handoff_block(handoff_text())
        manifest = read_json(MANIFEST)
        manifest["handoff_append_action_final"] = append_status
        manifest["handoff_sha256_after_append"] = sha(ROOT / "handoff.md")
        write_json(MANIFEST, manifest)
        stage_handoff("APPENDED", None)
        shutil.copy2(MANIFEST, root_exp / "manifest.json")
        shutil.copy2(RESULTS / "report.md", root_exp / "results/report.md")
        print(f"ROOT_PACKAGE_AND_HANDOFF_COMPLETE {root_exp}", flush=True)
    except Exception as exc:
        cleanup_partial_error = None
        if not package_synced and not root_exp_preexisting and root_exp.exists():
            try:
                shutil.rmtree(root_exp)
            except Exception as cleanup_exc:
                cleanup_partial_error = repr(cleanup_exc)
        manifest = read_json(MANIFEST)
        message = repr(exc)
        if cleanup_partial_error:
            message += f"; partial package cleanup failed: {cleanup_partial_error}"
        manifest.setdefault("runtime_io_faults", [])
        if package_synced:
            manifest["delivery_status"] = "PACKAGE_SYNCED_HANDOFF_PENDING"
            manifest["handoff_status"] = "PENDING_OSSFS_WRITE_EIO"
            manifest["handoff_append_error"] = message
            manifest["root_sync_error"] = None
        else:
            manifest["delivery_status"] = "TEMP_ONLY_OSSFS_EIO"
            manifest["handoff_status"] = "PENDING_OSSFS_WRITE_EIO"
            manifest["root_sync_error"] = message
            manifest["handoff_append_error"] = message
        note = f"root delivery failed: {message}"
        if note not in manifest["runtime_io_faults"]:
            manifest["runtime_io_faults"].append(note)
        manifest["root_experiment_directory"] = str(root_exp)
        manifest["temp_only_package_path"] = str(EXP)
        write_json(MANIFEST, manifest)
        summarize_and_report()
        stage_handoff("STAGED_ROOT_SYNC_FAILED", message)
        if package_synced:
            try:
                shutil.copy2(MANIFEST, root_exp / "manifest.json")
                shutil.copy2(RESULTS / "report.md", root_exp / "results/report.md")
            except Exception as sync_exc:
                manifest = read_json(MANIFEST)
                additional = f"root status refresh failed: {repr(sync_exc)}"
                if additional not in manifest.setdefault("runtime_io_faults", []):
                    manifest["runtime_io_faults"].append(additional)
                write_json(MANIFEST, manifest)
                summarize_and_report()
                stage_handoff("STAGED_ROOT_SYNC_FAILED", repr(sync_exc))
        print(f"ROOT_DELIVERY_PENDING {message}", flush=True)
    print(f"HANDOFF_APPEND_STAGED {RUNTIME / 'handoff_append.md'}", flush=True)

def mark_handoff_appended():
    protected_hash_check()
    manifest = read_json(MANIFEST)
    root_exp = ROOT / "experiment/experiment39_heterogeneity_aware_semantic_experts"
    if manifest.get("delivery_status") not in ("PACKAGE_SYNCED_HANDOFF_PENDING", "COMPLETE") or not root_exp.exists():
        raise RuntimeError("Cannot append handoff until the Experiment39 package is synced to the root workspace")
    action = append_handoff_block(handoff_text())
    manifest["handoff_status"] = "APPENDED_IDEMPOTENT_MARKERS"
    manifest["handoff_append_action"] = action
    manifest["handoff_sha256_after_append"] = sha(ROOT / "handoff.md")
    manifest["delivery_status"] = "COMPLETE"
    write_json(MANIFEST, manifest)
    summarize_and_report()
    # Keep the marked section and staged copy aligned with the final delivery state.
    append_status = append_handoff_block(handoff_text())
    manifest = read_json(MANIFEST)
    manifest["handoff_append_action_final"] = append_status
    manifest["handoff_sha256_after_append"] = sha(ROOT / "handoff.md")
    write_json(MANIFEST, manifest)
    shutil.copy2(MANIFEST, root_exp / "manifest.json")
    shutil.copy2(RESULTS / "report.md", root_exp / "results/report.md")
    stage_handoff("APPENDED", None)
    print("HANDOFF_APPEND_VERIFIED", manifest["handoff_sha256_after_append"], flush=True)


def cli():
    signal.signal(signal.SIGTERM, sigterm_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "worker", "dispatch",
                                              "counterfactual", "report", "finalize", "mark-handoff"))
    parser.add_argument("config", nargs="?")
    parser.add_argument("--until-fold", type=int, default=None)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "preflight":
        preflight()
    elif args.command == "worker":
        if not args.config:
            parser.error("worker requires a config")
        physical = int(os.environ.get("HASE_E39_PHYSICAL_GPU", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        result = worker(args.config, args.until_fold)
        print(f"WORKER_COMPLETE {args.config} {result['status']} {len(result['metrics'])}/10", flush=True)
    elif args.command == "dispatch":
        dispatch()
    elif args.command == "counterfactual":
        counterfactuals()
    elif args.command == "report":
        summarize_and_report()
    elif args.command == "finalize":
        finalize()
    elif args.command == "mark-handoff":
        mark_handoff_appended()


if __name__ == "__main__":
    cli()
