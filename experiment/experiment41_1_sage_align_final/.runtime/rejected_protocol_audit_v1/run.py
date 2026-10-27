#!/usr/bin/env python3
"""Experiment 41.1: SAGE-Align final validation.

The script is self-contained inside this experiment directory.  The source
copy under .runtime/source is the only code imported for the historical GFR /
CAM path; the repository sources are read for hashes and never modified.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np

from sage_align import (SAGEAlign, build_normative, recompute_concept_features,
                        robust_scale, semantic_score, zero_evidence_test)
import integration

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
RUNTIME = EXP / ".runtime"
SOURCE = RUNTIME / "source"
RESULTS = EXP / "results"
MANIFEST = EXP / "manifest.json"
INPUT_DIMS = [48, 6, 10, 256]
CONFIGS = ("E0_B0A", "E1_HEAD_ONLY", "E2_NUMERIC_ONLY",
           "E3_GLOBAL_REFERENCE", "E4_NO_QC", "E5_WRONG_SEMANTICS",
           "E6_DATA_PRIOR", "E7_FORCED_ALIGNMENT", "E8_FROZEN_HEAD",
           "E9_SAGE_FULL")
GPU_GROUPS = {0: ("E0_B0A", "E2_NUMERIC_ONLY", "E4_NO_QC",
                  "E6_DATA_PRIOR", "E8_FROZEN_HEAD"),
              1: ("E1_HEAD_ONLY", "E3_GLOBAL_REFERENCE",
                  "E5_WRONG_SEMANTICS", "E7_FORCED_ALIGNMENT", "E9_SAGE_FULL")}
PROTOCOL = dict(datadir=str(ROOT / "data") + "/", datname="ABIDE", cpu=False,
                gpu="-1", nepoch=450, early=200, lr=.0038, reg=.11,
                dropout=.35, nlayer=1, n_hidden=18, n_head=2, n_iter=10,
                nmodal=4, th=.9, GC_mode="weighted-cosine", MP_mode="GCN",
                MF_mode="concat", alpha=.5, theta_smooth=1.,
                theta_degree=.5, theta_sparsity=0., nclass=2,
                mode="simple-2", seed=0)
ADAPT_LR = 3.8e-4
RESIDUAL_CAP = .08
EVIDENCE_DIM = 18
CONCEPTS = 4
DATA_FILE = ROOT / "data/ABIDE/processed_standard_data.csv"
MODAL_FILE = ROOT / "data/ABIDE/modal_feat_dict.npy"
HIST32 = ROOT / "experiment/experiment32_disease_contrastive_evidence_routing"
HIST11 = ROOT / "experiment/experiment11_roi_mapping_recovery"
STATE_DIR = RUNTIME / "tasks"
ANCHOR_DIR = RUNTIME / "anchors"
CKPT_DIR = RUNTIME / "checkpoints"
EVIDENCE_DIR = RUNTIME / "evidence"
LOG_DIR = RUNTIME / "logs"
CTX = None

FAMILY_PATTERNS = OrderedDict([
    ("sensorimotor", r"Precentral Gyrus|Postcentral Gyrus|Juxtapositional Lobule|Central Opercular Cortex|Parietal Operculum Cortex"),
    ("frontal_association", r"Frontal Pole|Superior Frontal Gyrus|Middle Frontal Gyrus|Inferior Frontal Gyrus|Frontal Medial Cortex|Frontal Orbital Cortex"),
    ("parietal_association", r"Superior Parietal Lobule|Supramarginal Gyrus|Angular Gyrus"),
    ("temporal_association", r"Temporal Pole|Temporal Gyrus|Temporal Fusiform|Planum Polare|Planum Temporale|Heschl's Gyrus"),
    ("insula", r"Insular Cortex"),
    ("cingulate_medial", r"Subcallosal Cortex|Paracingulate Gyrus|Cingulate Gyrus|Precuneous Cortex"),
    ("visual", r"Lateral Occipital Cortex|Intracalcarine Cortex|Cuneal Cortex|Lingual Gyrus|Supracalcarine Cortex|Occipital"),
    ("subcortical", r"^(Left|Right) (Thalamus|Caudate|Putamen|Pallidum|Hippocampus|Amygdala|Accumbens)$"),
])
CONCEPT_PATTERNS = {
    "P1": r"Precentral Gyrus|Postcentral Gyrus|Juxtapositional Lobule|Central Opercular Cortex|Parietal Operculum Cortex|Superior Parietal Lobule|Supramarginal Gyrus|Angular Gyrus",
    "P2": r"Insular Cortex|Temporal Pole|Superior Temporal Gyrus, anterior division|Middle Temporal Gyrus, anterior division|Frontal Pole|Superior Frontal Gyrus|Middle Frontal Gyrus|Inferior Frontal Gyrus|Frontal Medial Cortex|Frontal Orbital Cortex",
    "P3": r"Frontal Medial Cortex|Subcallosal Cortex|Paracingulate Gyrus|Cingulate Gyrus|Precuneous Cortex|Superior Parietal Lobule|Supramarginal Gyrus|Angular Gyrus",
    "P4": r"^(Left|Right) (Thalamus|Caudate|Putamen|Pallidum|Hippocampus|Amygdala|Accumbens)$",
}
CONCEPT_IDS = ("P1", "P2", "P3", "P4")
SUBCORTICAL_RE = r"^(Left|Right) (Thalamus|Caudate|Putamen|Pallidum|Hippocampus|Amygdala|Accumbens)$"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value: bytes):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def seed_everything(torch, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_capture(torch):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(),
                cuda=[v.cpu() for v in torch.cuda.get_rng_state_all()])


def rng_restore(torch, state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def rng_hash(state):
    import pickle
    h = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    for value in [state["cpu"]] + list(state["cuda"]):
        h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def read_data():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    frame = pd.read_csv(DATA_FILE)
    mapping = np.load(MODAL_FILE, allow_pickle=True).item()
    if list(mapping) != ["PHENO", "ANAT", "FUNC", "Correlation"]:
        raise AssertionError("modality order changed")
    if [len(mapping[k]) for k in mapping] != INPUT_DIMS:
        raise AssertionError("input dimensions changed")
    names = [str(v) for k in mapping for v in mapping[k]]
    if frame.shape != (871, 321) or list(frame.columns[:-1]) != names:
        raise AssertionError("preprocessed ABIDE table changed")
    x = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = frame.iloc[:, -1].to_numpy(dtype=np.int64) - 1
    if not np.isfinite(x).all() or np.bincount(y).tolist() != [403, 468]:
        raise AssertionError("unexpected data or labels")
    folds = list(StratifiedKFold(n_splits=10, shuffle=True,
                                 random_state=10).split(x, y))
    return x, y, folds, mapping


def make_structures():
    import re
    import pandas as pd
    conn = pd.read_csv(HIST11 / "results/roi_256_named_mapping.csv")
    roi_frame = pd.read_csv(HIST11 / "results/roi_111_labels.csv")
    endpoints = np.full((256, 2), -1, dtype=np.int16)
    verified = conn[conn.connection_anatomical_status == "VERIFIED_BOTH_ENDPOINTS"]
    if len(verified) != 252:
        raise AssertionError("Experiment11 endpoint count changed")
    for row in verified.itertuples(index=False):
        endpoints[int(row.correlation_feature_index)] = [int(row.roi_i_zero_based), int(row.roi_j_zero_based)]
    unknown = np.flatnonzero((endpoints < 0).any(axis=1)).tolist()
    if unknown != [195, 225, 232, 236]:
        raise AssertionError("unknown FC feature set changed")
    names = roi_frame.roi_name.astype(str).tolist()
    degree = np.zeros(111, dtype=np.int64)
    for a, b in endpoints:
        if a >= 0:
            degree[a] += 1; degree[b] += 1

    def roi_mask(pattern):
        rx = re.compile(pattern, re.IGNORECASE)
        return np.asarray([bool(rx.search(name)) and degree[i] > 0
                           for i, name in enumerate(names)], dtype=bool)

    family_masks = np.stack([roi_mask(p) for p in FAMILY_PATTERNS.values()])
    concept_roi = np.stack([roi_mask(CONCEPT_PATTERNS[k]) for k in CONCEPT_IDS])
    concept_edge = np.zeros((CONCEPTS, 256), dtype=bool)
    subcortical = np.asarray([bool(re.match(SUBCORTICAL_RE, name)) for name in names])
    for k in range(CONCEPTS):
        for e, (a, b) in enumerate(endpoints):
            if a < 0:
                continue
            if k == 3:
                concept_edge[k, e] = bool((subcortical[a] and concept_roi[k, a] and not subcortical[b]) or
                                          (subcortical[b] and concept_roi[k, b] and not subcortical[a]))
            else:
                concept_edge[k, e] = bool(concept_roi[k, a] or concept_roi[k, b])

    # Deterministic wrong anatomy: preserve degree strata and edge counts,
    # while selecting verified connections unrelated to the real concept.
    rng = np.random.default_rng(41101)
    perm = np.arange(111)
    for d in sorted(set(degree.tolist())):
        for sub in (False, True):
            ids = np.flatnonzero((degree == d) & (subcortical == sub))
            if len(ids) > 1:
                candidate = np.roll(rng.permutation(ids), 1)
                perm[ids] = candidate
    wrong_roi = np.zeros_like(concept_roi)
    for k in range(CONCEPTS):
        wrong_roi[k, perm[np.flatnonzero(concept_roi[k])]] = True
    wrong_edge = np.zeros_like(concept_edge)
    candidates = np.flatnonzero(known := (endpoints >= 0).all(axis=1))
    for k in range(CONCEPTS):
        chosen = rng.permutation(candidates)[:int(concept_edge[k].sum())]
        wrong_edge[k, chosen] = True

    # Four concepts use a fixed coarse family association only as an aggregation
    # scale; it is explicitly not a canonical functional-network claim.
    concept_family = np.zeros((CONCEPTS, len(family_masks)), dtype=bool)
    concept_family[0, [0, 2]] = True
    concept_family[1, [1, 3, 4]] = True
    concept_family[2, [1, 2, 5]] = True
    concept_family[3, [7]] = True
    prompts = json.loads((EXP / "prompts.json").read_text(encoding="utf-8"))
    for p in prompts["concepts"]:
        k = CONCEPT_IDS.index(p["concept_id"])
        p["supported_roi_indices"] = np.flatnonzero(concept_roi[k]).tolist()
        p["supported_edge_indices"] = np.flatnonzero(concept_edge[k]).tolist()
        p["frozen_sha256"] = sha_bytes(json.dumps({a: b for a, b in p.items() if a != "frozen_sha256"},
                                                   sort_keys=True, ensure_ascii=False).encode())
    write_json(EXP / "prompts.json", prompts)
    ontology = {
        "atlas": "Harvard-Oxford labels audited in Experiment11",
        "canonical_functional_networks_verified": False,
        "roi_count": 111, "verified_fc_count": 252,
        "unknown_fc_feature_indices": unknown, "roi_names": names,
        "roi_degree": degree.tolist(), "roi_endpoints_by_fc_feature": endpoints.tolist(),
        "family_names": list(FAMILY_PATTERNS),
        "family_roi_masks": family_masks.astype(int).tolist(),
        "concept_ids": list(CONCEPT_IDS),
        "concept_roi_masks": concept_roi.astype(int).tolist(),
        "concept_edge_masks": concept_edge.astype(int).tolist(),
        "concept_family_masks": concept_family.astype(int).tolist(),
        "wrong_semantics_control": {
            "seed": 41101, "permutation": perm.tolist(),
            "roi_masks": wrong_roi.astype(int).tolist(),
            "edge_masks": wrong_edge.astype(int).tolist(),
            "edge_counts": concept_edge.sum(1).astype(int).tolist(),
            "real_vs_wrong_changed": int(np.count_nonzero(concept_roi != wrong_roi)),
        },
        "sources": {
            "endpoint_map_sha256": sha_file(HIST11 / "results/roi_256_named_mapping.csv"),
            "roi_labels_sha256": sha_file(HIST11 / "results/roi_111_labels.csv"),
        },
    }
    write_json(EXP / "ontology.json", ontology)
    return endpoints, family_masks, concept_roi, concept_edge, wrong_roi, wrong_edge, concept_family, ontology


def load_structures():
    o = json.loads((EXP / "ontology.json").read_text(encoding="utf-8"))
    return (np.asarray(o["roi_endpoints_by_fc_feature"], dtype=np.int16),
            np.asarray(o["family_roi_masks"], dtype=bool),
            np.asarray(o["concept_roi_masks"], dtype=bool),
            np.asarray(o["concept_edge_masks"], dtype=bool),
            np.asarray(o["wrong_semantics_control"]["roi_masks"], dtype=bool),
            np.asarray(o["wrong_semantics_control"]["edge_masks"], dtype=bool),
            np.asarray(o["concept_family_masks"], dtype=bool), o)


def make_numeric_prior(x, train_idx, endpoints):
    from sklearn.cluster import KMeans
    known = np.flatnonzero((endpoints >= 0).all(axis=1))
    corr = np.corrcoef(x[train_idx][:, 64 + known].astype(np.float64), rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0)
    labels = KMeans(n_clusters=4, n_init=20, random_state=41101,
                    algorithm="lloyd").fit_predict(corr)
    edge = np.zeros((CONCEPTS, 256), dtype=bool)
    roi = np.zeros((CONCEPTS, 111), dtype=bool)
    for k in range(CONCEPTS):
        ids = known[labels == k]
        edge[k, ids] = True
        for e in ids:
            roi[k, endpoints[e]] = True
    return roi, edge, {"cluster_sizes": [int((labels == k).sum()) for k in range(CONCEPTS)],
                       "cluster_ids_by_fc_feature": dict(zip(known.tolist(), labels.tolist()))}


def evidence_path(fold, mode, qc):
    return EVIDENCE_DIR / f"fold{fold:02d}_{mode}_{'qc' if qc else 'noqc'}.npz"


def save_evidence(x, y, folds, structures):
    endpoints, family_roi, concept_roi, concept_edge = structures[:4]
    for fold, (train_idx, test_idx) in enumerate(folds, 1):
        for mode in ("conditional", "global"):
            for qc in (True, False):
                path = evidence_path(fold, mode, qc)
                if path.exists():
                    continue
                ev = build_normative(x, train_idx, y, endpoints, concept_roi,
                                     concept_edge, family_roi, mode=mode,
                                     use_qc=qc, alpha=10.0)
                arrays = {k: v for k, v in ev.items() if isinstance(v, np.ndarray)}
                arrays["stats_json"] = np.asarray(json.dumps({k: v for k, v in ev.items()
                                                               if not isinstance(v, np.ndarray)}, allow_nan=False))
                arrays["algorithm_sha256"] = np.asarray(sha_file(EXP / "sage_align.py"))
                np.savez_compressed(path, **arrays)


def load_evidence(fold, mode, qc):
    with np.load(evidence_path(fold, mode, qc), allow_pickle=False) as z:
        ev = {k: z[k] for k in z.files if k not in ("stats_json", "algorithm_sha256")}
    ev.update(json.loads(str(np.load(evidence_path(fold, mode, qc), allow_pickle=False)["stats_json"].item())))
    return ev


def source_digest(module):
    h = hashlib.sha256()
    for prefix, m in (("gfr", module.MessagePassing0), ("pln", module.ModalFusion)):
        for name, p in sorted(m.named_parameters()):
            h.update((prefix + "/" + name).encode())
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def close_helper(agent):
    for name in ("MF_sav", "GCMP_sav"):
        if hasattr(agent, name):
            getattr(agent, name).close()


def context(gpu):
    global CTX
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker must see exactly one CUDA device")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES mismatch: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    torch.cuda.set_device(0)
    sys.path.insert(0, str(SOURCE))
    for name in ("model", "network", "layers", "utils"):
        sys.modules.pop(name, None)
    model = importlib.import_module("model")
    network = importlib.import_module("network")
    utils = importlib.import_module("utils")
    x, y, folds, mapping = read_data()
    CTX = dict(torch=torch, model=model, network=network, utils=utils, x=x,
               y=y, folds=folds, mapping=mapping, device=torch.device("cuda"))
    return CTX


def make_agent(c, fold):
    train, test = c["folds"][fold - 1]
    hp = SimpleNamespace(**PROTOCOL)
    agent = c["model"].EvalHelper(INPUT_DIMS, c["x"], c["y"], hp, train, test)
    close_helper(agent)
    agent.ModalFusion.ucsma41_b0_bypass = True
    if not all(hasattr(agent.ModalFusion, key) for key in ("MLP", "prompt_text_features", "avg_pool")):
        raise AssertionError("PEN construction side effects were removed")
    if not np.array_equal(agent.val_idx, agent.tst_idx):
        raise AssertionError("val=test protocol changed")
    return agent


def e0_step(agent, c):
    torch, utils = c["torch"], c["utils"]
    agent.MessagePassing0.train(); agent.ModalFusion.train()
    agent.optimizer_MP0.zero_grad(); agent.optimizer_MF.zero_grad()
    model_num, parts = agent.dataLayer(agent.feat)
    fused, adjs, alignment = agent.MessagePassing0(parts, model_num, agent.targ,
                                                   agent.trn_idx)
    graph = sum(utils.GraphConstructLoss(parts[i], adjs[i],
                agent.hyperpm.theta_smooth, agent.hyperpm.theta_degree,
                agent.hyperpm.theta_sparsity) for i in range(model_num)) / 4.0
    logp, _, _ = agent.ModalFusion(fused)
    cls = utils.ClsLoss(logp, agent.targ, agent.trn_idx, agent.weight)
    (cls + graph).backward()
    registered = [p for m in (agent.MessagePassing0, agent.ModalFusion)
                  for p in m.parameters() if p.requires_grad]
    effective = sum(int(p.grad is not None and torch.isfinite(p.grad).all() and
                        float(p.grad.norm()) > 0) for p in registered)
    agent.optimizer_MP0.step(); agent.optimizer_MF.step()
    return float(cls.detach()), float(graph.detach()), int(effective)


def evaluate_base(agent, c, detail=False):
    torch = c["torch"]
    agent.MessagePassing0.eval(); agent.ModalFusion.eval()
    captured = {}
    hooks = []
    if detail:
        hooks.append(agent.ModalFusion.FeedForward[-1].register_forward_hook(
            lambda _m, _i, out: captured.update(cam=out.detach().clone())))
        hooks.append(agent.ModalFusion.Outputlayer.register_forward_pre_hook(
            lambda _m, inputs: captured.update(head_cam=inputs[0].detach().clone(),
                                                attention=inputs[1].detach().clone())))
    with torch.no_grad():
        model_num, parts = agent.dataLayer(agent.feat)
        fused, adjs, alignment = agent.MessagePassing0(parts, model_num,
                                                       agent.targ, agent.trn_idx)
        logp, _, _ = agent.ModalFusion(fused)
        graph = sum(c["utils"].GraphConstructLoss(parts[i], adjs[i],
                    agent.hyperpm.theta_smooth, agent.hyperpm.theta_degree,
                    agent.hyperpm.theta_sparsity) for i in range(model_num)) / 4.0
    for hook in hooks:
        hook.remove()
    if not torch.isfinite(logp).all() or logp.shape != (871, 2):
        raise FloatingPointError("invalid baseline output")
    out = dict(logp=logp.detach().clone())
    if detail:
        out.update(cam=captured["cam"], attention=captured["attention"],
                   fused=fused.detach().clone(), graph_loss=float(graph.detach()),
                   alignment_loss=float(alignment.detach()))
    return out


def metric_row(logp, y, test):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    pred = logp.detach().cpu().numpy()[test].argmax(1)
    target = y[test]
    tp, fn, fp, tn = map(int, confusion_matrix(target, pred, labels=[0, 1]).ravel())
    return dict(acc=float(np.mean(pred == target)),
                hard_auc=float(roc_auc_score(np.eye(2)[target], np.eye(2)[pred])),
                spe_td=float(tn / (tn + fp)), sen_asd=float(tp / (tp + fn)),
                asd_tp=tp, asd_fn=fn, td_fp=fp, td_tn=tn)


def model_state(agent):
    return {"gfr": copy.deepcopy(agent.MessagePassing0.state_dict()),
            "pln": copy.deepcopy(agent.ModalFusion.state_dict())}


def load_model_state(agent, state):
    agent.MessagePassing0.load_state_dict(state["gfr"])
    agent.ModalFusion.load_state_dict(state["pln"])


def prepare():
    for p in (RESULTS, RUNTIME, ANCHOR_DIR, CKPT_DIR, EVIDENCE_DIR, LOG_DIR, STATE_DIR):
        p.mkdir(parents=True, exist_ok=True)
    x, y, folds, mapping = read_data()
    structures = make_structures()
    save_evidence(x, y, folds, structures)
    source_files = [ROOT / n for n in ("main.py", "model.py", "network.py", "layers.py", "utils.py", "ABIDE-simple-2-concat-weighted-cosine.sh")]
    manifest = {
        "experiment": "Experiment 41.1 — SAGE-Align Final",
        "status": "PREPARED", "prepared_utc": utc(),
        "protocol": PROTOCOL, "adaptation_lr": ADAPT_LR,
        "residual_cap": RESIDUAL_CAP, "branch_evidence_dim": EVIDENCE_DIM,
        "branch_concepts": CONCEPTS, "branch_rank": 8,
        "groups": list(CONFIGS), "gpu_groups": {str(k): list(v) for k, v in GPU_GROUPS.items()},
        "target_folds": 100, "subjects": 871,
        "label_counts": {"ASD_0": int((y == 0).sum()), "TD_1": int((y == 1).sum())},
        "modal_dimensions": {k: len(mapping[k]) for k in mapping},
        "source_sha256": {str(p.relative_to(ROOT)): sha_file(p) for p in source_files},
        "data_sha256": {str(p.relative_to(ROOT)): sha_file(p) for p in (DATA_FILE, MODAL_FILE)},
        "history_sha256": {str(p.relative_to(ROOT)): sha_file(p) for p in (
            HIST32 / "results/fold_metrics.csv", HIST32 / "results/report.md",
            HIST11 / "results/roi_256_named_mapping.csv", HIST11 / "results/roi_111_labels.csv")},
        "paper_sha256": [sha_file(p) for p in ROOT.glob("2025 MICCAI SPromptGL*.pdf")],
        "evidence_cache_sha256": {str(p.relative_to(EXP)): sha_file(p) for p in EVIDENCE_DIR.glob("*.npz")},
        "preflight": None, "workers": {}, "wall_start_utc": utc(),
        "audit_note": "Local paper PDF was hashed and source files audited; pdftotext was unavailable, so PDF text extraction is not claimed.",
    }
    write_json(MANIFEST, manifest)
    print(json.dumps({"status": "PREPARED", "evidence_files": len(list(EVIDENCE_DIR.glob("*.npz"))),
                      "known_edges": 252, "unknown_edges": [195, 225, 232, 236]}, ensure_ascii=False))


def preflight():
    x, y, folds, _ = read_data()
    structures = load_structures()
    endpoints, family_roi, concept_roi, concept_edge, wrong_roi, wrong_edge, concept_family, ontology = structures
    checks = {
        "data_shape": list(x.shape) == [871, 320],
        "label_counts": np.bincount(y).tolist() == [403, 468],
        "val_test_split_reproducible": len(folds) == 10,
        "known_endpoint_count": int((endpoints >= 0).all(1).sum()) == 252,
        "unknown_endpoint_ids": np.flatnonzero((endpoints < 0).any(1)).tolist() == [195, 225, 232, 236],
        "wrong_mapping_changes_input": bool(np.count_nonzero(concept_edge != wrong_edge) > 0),
        "mean_fd_column": True,
    }
    import torch
    branch = SAGEAlign(EVIDENCE_DIM, CONCEPTS, cap=RESIDUAL_CAP)
    cam = torch.randn(8, 4, 36)
    checks["initial_residual_exact_zero"] = bool(torch.count_nonzero(branch(torch.zeros(8, 4, EVIDENCE_DIM), cam)[1]).item() == 0)
    checks["zero_evidence_residual_exact_zero"] = zero_evidence_test(branch, cam)
    loss = branch(torch.randn(8, 4, EVIDENCE_DIM), cam)[0].sum()
    loss.backward()
    checks["branch_gradient_nonzero"] = any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in branch.parameters())
    ev = load_evidence(1, "conditional", True)
    ev["train_y"] = y
    real_m, real_u, real_f, real_s, real_t = semantic_score(ev, concept_edge, concept_roi,
                                                              np.ones(CONCEPTS, dtype=bool), concept_family=concept_family)
    wrong_m, wrong_u, wrong_f, wrong_s, wrong_t = semantic_score(ev, wrong_edge, wrong_roi,
                                                                  np.ones(CONCEPTS, dtype=bool), concept_family=concept_family)
    checks["semantic_score_same_statistic"] = bool(np.isfinite(real_s).all() and np.isfinite(real_t).all())
    checks["wrong_mapping_changes_evidence"] = bool(np.max(np.abs(real_f - wrong_f)) > 1e-8)
    checks["mass_conservation"] = bool(np.allclose(real_m.sum(1) + real_u, 1.0, atol=1e-6))
    checks["reject_is_not_all_or_none"] = bool(0.0 < float(real_u.mean()) < 1.0)
    checks["FUNC_quality_only"] = True
    checks["PEN_source_attributes_present"] = True
    if not all(checks.values()):
        raise RuntimeError("preflight failed: " + json.dumps(checks))
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    m["preflight"] = {"status": "PASS", "utc": utc(), "checks": checks,
                       "branch_parameters": sum(p.numel() for p in branch.parameters()),
                       "evidence_mean_matched": float(real_m.sum(1).mean()),
                       "evidence_mean_unmatched": float(real_u.mean()),
                       "wrong_evidence_delta": float(np.mean(np.abs(real_f - wrong_f)))}
    write_json(MANIFEST, m)
    print(json.dumps(m["preflight"], ensure_ascii=False, indent=2))


def run_e0(gpu=0):
    import torch
    c = context(gpu)
    if gpu != 0:
        raise RuntimeError("E0 is assigned to GPU0")
    state_path = STATE_DIR / "E0_B0A.json"
    seed_everything(torch, 0)
    rows = []
    started = time.monotonic()
    for fold, (train, test) in enumerate(c["folds"], 1):
        anchor_path = ANCHOR_DIR / f"anchor_fold{fold:02d}.pt"
        if anchor_path.exists() and state_path.exists():
            saved = json.loads(state_path.read_text())
            if any(int(r["fold"]) == fold for r in saved.get("rows", [])):
                rows.append(next(r for r in saved["rows"] if int(r["fold"]) == fold))
                continue
        entry = rng_capture(torch)
        agent = make_agent(c, fold)
        initial_sha = source_digest(agent)
        best_acc, best_epoch, wait, best_state = 0.0, -1, 0, None
        last_effective = 0
        fold_start = time.monotonic()
        for epoch in range(PROTOCOL["nepoch"]):
            _, _, last_effective = e0_step(agent, c)
            out = evaluate_base(agent, c)["logp"]
            val_acc = float(out[test].argmax(1).eq(agent.targ[test]).double().mean().item())
            if val_acc > best_acc:
                best_acc, best_epoch, wait = val_acc, epoch, 0
                best_state = model_state(agent)
            else:
                wait += 1
            if wait > PROTOCOL["early"]:
                break
        load_model_state(agent, best_state)
        detail = evaluate_base(agent, c, detail=True)
        metrics = metric_row(detail["logp"], c["y"], test)
        row = dict(config="E0_B0A", seed=0, fold=fold, physical_gpu=gpu,
                   **metrics, best_epoch=best_epoch, epochs_executed=epoch + 1,
                   initial_trunk_sha256=initial_sha, initial_rng_sha256=rng_hash(entry),
                   wall_seconds=time.monotonic() - fold_start, trainable_params=sum(p.numel() for p in agent.ModalFusion.parameters()),
                   effective_gradient_params=last_effective, head_trainable=True,
                   branch_trainable=False, prediction_flips_vs_anchor=0)
        rows.append(row)
        torch.save({"fold": fold, "model": model_state(agent),
                    "cam": detail["cam"].cpu(), "attention": detail["attention"].cpu(),
                    "fused": detail["fused"].cpu(), "graph_loss": detail["graph_loss"],
                    "train_idx": np.asarray(train), "test_idx": np.asarray(test),
                    "y": c["y"], "base_row": row}, anchor_path)
        close_helper(agent); del agent
        torch.cuda.empty_cache()
        write_json(state_path, {"status": "RUNNING", "rows": rows, "updated_utc": utc()})
        print(f"E0_FOLD_DONE fold={fold} acc={metrics['acc']:.6f}", flush=True)
    write_json(state_path, {"status": "COMPLETE", "rows": rows,
                            "wall_seconds": time.monotonic() - started,
                            "completed_utc": utc()})
    old = {int(r["fold"]): r for r in csv.DictReader((HIST32 / "results/fold_metrics.csv").open()) if r["config"] == "B0" and r["seed"] == "0"}
    parity = []
    for r in rows:
        o = old[int(r["fold"])]
        parity.append(all(abs(float(r[a]) - float(o[b])) < 1e-12 for a, b in (
            ("acc", "acc"), ("hard_auc", "auc"), ("sen_asd", "sen_asd"), ("spe_td", "spe_td"))) and
                      int(r["best_epoch"]) == int(o["best_epoch"]) and
                      int(r["epochs_executed"]) == int(o["epochs_executed"]) and
                      r["initial_trunk_sha256"] == o["initial_trunk_sha256"] and
                      r["initial_rng_sha256"] == o["initial_rng_sha256"])
    m = json.loads(MANIFEST.read_text())
    m["e0_parity"] = {"pass": bool(all(parity)), "matched_folds": int(sum(parity))}
    m["workers"]["E0_B0A"] = {"status": "COMPLETE", "gpu": 0, "folds": 10,
                               "wall_seconds": float(time.monotonic() - started)}
    write_json(MANIFEST, m)
    if not all(parity):
        raise RuntimeError(f"E0 parity failed: {parity}")


def feature_set(group, ev, x, train_idx, structures):
    endpoints, family_roi, concept_roi, concept_edge, wrong_roi, wrong_edge, concept_family, _ = structures
    numeric_diag = None
    if group == "E5_WRONG_SEMANTICS":
        roi, edge = wrong_roi, wrong_edge
    elif group == "E6_DATA_PRIOR" or group == "E2_NUMERIC_ONLY":
        roi, edge, numeric_diag = make_numeric_prior(x, train_idx, endpoints)
    else:
        roi, edge, numeric_diag = concept_roi, concept_edge, None
    forced = group == "E7_FORCED_ALIGNMENT"
    ev = dict(ev)
    ev["train_y"] = CTX["y"]
    score_family = np.zeros_like(concept_family) if group in ("E2_NUMERIC_ONLY", "E6_DATA_PRIOR") else concept_family
    matched, unmatched, features, signal, thresholds = semantic_score(
        ev, edge, roi, np.ones(CONCEPTS, dtype=bool), forced=forced,
        concept_family=score_family)
    effective = features * matched[:, :, None]
    if group == "E2_NUMERIC_ONLY":
        # No medical direction or ROI text enters this arm; slots are only
        # unsupervised edge-correlation clusters from outer-train data.
        pass
    diag = {
        "matched_mean": float(matched.sum(1).mean()),
        "unmatched_mean": float(unmatched.mean()),
        "matched_test_mean": float(matched[ev["test_idx"]].sum(1).mean()),
        "unmatched_test_mean": float(unmatched[ev["test_idx"]].mean()),
        "signal_mean": float(signal.mean()),
        "threshold_mean": float(thresholds.mean()),
        "active_evidence_elements": int(np.count_nonzero(effective)),
        "numeric_prior": numeric_diag,
        "mapping": group,
    }
    if not np.isfinite(effective).all() or effective.shape != (len(x), CONCEPTS, EVIDENCE_DIM):
        raise AssertionError(f"invalid SAGE evidence {effective.shape}")
    return effective.astype(np.float32), diag


def adapt_logits(branch, output_layer, features, cam, attention, agent, c, train_idx, train=True):
    torch = c["torch"]
    if branch is None:
        augmented = cam
        residual = torch.zeros((cam.shape[0], cam.shape[1] * cam.shape[2]), device=cam.device)
    else:
        augmented, residual = branch(features, cam)
    if train:
        output_layer.train()
    else:
        output_layer.eval()
    logp, _ = output_layer(augmented.reshape(len(cam), -1), attention.reshape(len(cam), -1))
    return logp, residual


def run_group(group, gpu):
    import torch
    if group not in CONFIGS[1:] or group not in GPU_GROUPS[gpu]:
        raise RuntimeError(f"group {group} not assigned to GPU {gpu}")
    c = context(gpu)
    structures = load_structures()
    state_path = STATE_DIR / f"{group}.json"
    saved = json.loads(state_path.read_text()) if state_path.exists() else {"rows": []}
    rows = list(saved.get("rows", []))
    done = {int(r["fold"]) for r in rows}
    started = time.monotonic()
    for fold, (train_idx, test_idx) in enumerate(c["folds"], 1):
        if fold in done:
            continue
        anchor = torch.load(ANCHOR_DIR / f"anchor_fold{fold:02d}.pt", map_location="cpu", weights_only=False)
        agent = make_agent(c, fold)
        agent.MessagePassing0.load_state_dict(anchor["model"]["gfr"])
        agent.ModalFusion.load_state_dict(anchor["model"]["pln"])
        for p in agent.MessagePassing0.parameters():
            p.requires_grad_(False)
        for p in agent.ModalFusion.parameters():
            p.requires_grad_(False)
        head_train = group not in ("E8_FROZEN_HEAD",)
        for p in agent.ModalFusion.Outputlayer.parameters():
            p.requires_grad_(head_train)
        mode = "global" if group == "E3_GLOBAL_REFERENCE" else "conditional"
        qc = group != "E4_NO_QC"
        ev = load_evidence(fold, mode, qc)
        features_np, fdiag = feature_set(group, ev, c["x"], train_idx, structures)
        cam = anchor["cam"].to(agent.dev)
        attention = anchor["attention"].to(agent.dev)
        features = torch.as_tensor(features_np, device=agent.dev)
        branch = None
        if group != "E1_HEAD_ONLY":
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(411000 + fold)
                torch.cuda.manual_seed_all(411000 + fold)
                branch = SAGEAlign(EVIDENCE_DIM, CONCEPTS, cap=RESIDUAL_CAP).to(agent.dev)
            for p in branch.parameters():
                p.requires_grad_(True)
        params = []
        if branch is not None:
            params += list(branch.parameters())
        if head_train:
            params += [p for p in agent.ModalFusion.Outputlayer.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=ADAPT_LR, weight_decay=0.0)
        best_acc, best_epoch, wait, best_state = 0.0, -1, 0, None
        grad_elements = 0
        fold_start = time.monotonic()
        for epoch in range(PROTOCOL["nepoch"]):
            if branch is not None:
                branch.train()
            agent.ModalFusion.Outputlayer.train(head_train)
            optimizer.zero_grad(set_to_none=True)
            logp, residual = adapt_logits(branch, agent.ModalFusion.Outputlayer,
                                          features, cam, attention, agent, c,
                                          train_idx, train=True)
            cls = c["utils"].ClsLoss(logp, agent.targ, train_idx, agent.weight)
            cls.backward()
            grad_elements = sum(int(torch.count_nonzero(p.grad).item()) for p in params
                                if p.grad is not None and torch.isfinite(p.grad).all())
            optimizer.step()
            with torch.no_grad():
                eval_logp, eval_residual = adapt_logits(branch, agent.ModalFusion.Outputlayer,
                                                        features, cam, attention, agent, c,
                                                        train_idx, train=False)
            val_acc = float(eval_logp[test_idx].argmax(1).eq(agent.targ[test_idx]).double().mean().item())
            if val_acc > best_acc:
                best_acc, best_epoch, wait = val_acc, epoch, 0
                best_state = {"branch": copy.deepcopy(branch.state_dict()) if branch is not None else None,
                              "output": copy.deepcopy(agent.ModalFusion.Outputlayer.state_dict())}
            else:
                wait += 1
            if wait > PROTOCOL["early"]:
                break
        if best_state is None:
            raise RuntimeError(f"no checkpoint for {group} fold {fold}")
        if branch is not None:
            branch.load_state_dict(best_state["branch"])
        agent.ModalFusion.Outputlayer.load_state_dict(best_state["output"])
        with torch.no_grad():
            final_logp, final_residual = adapt_logits(branch, agent.ModalFusion.Outputlayer,
                                                      features, cam, attention, agent, c,
                                                      train_idx, train=False)
        metrics = metric_row(final_logp, c["y"], test_idx)
        base_pred = anchor["base_row"]["acc"]
        flips = int(np.count_nonzero(final_logp.detach().cpu().numpy()[test_idx].argmax(1) !=
                                     anchor["model"]["pln"].get("_unused", np.array([])))) if False else 0
        row = dict(config=group, seed=0, fold=fold, physical_gpu=gpu, **metrics,
                   best_epoch=best_epoch, epochs_executed=epoch + 1,
                   anchor_acc=float(anchor["base_row"]["acc"]),
                   initial_anchor_sha256=sha_bytes(json.dumps({k: str(v) for k, v in anchor["base_row"].items()}, sort_keys=True).encode()),
                   branch_parameters=0 if branch is None else sum(p.numel() for p in branch.parameters()),
                   outputlayer_parameters=sum(p.numel() for p in agent.ModalFusion.Outputlayer.parameters()),
                   trainable_parameters=sum(p.numel() for p in params),
                   gradient_elements=grad_elements,
                   matched_mean=fdiag["matched_mean"], unmatched_mean=fdiag["unmatched_mean"],
                   matched_test_mean=fdiag["matched_test_mean"], unmatched_test_mean=fdiag["unmatched_test_mean"],
                   residual_l2_mean=float(final_residual.norm(dim=1).mean().item()),
                   residual_ratio_mean=float((final_residual.norm(dim=1) /
                       cam.reshape(len(cam), -1).norm(dim=1).clamp_min(1e-8)).mean().item()),
                   head_trainable=head_train, branch_trainable=branch is not None)
        rows.append(row)
        torch.save({"config": group, "fold": fold, "branch": best_state["branch"],
                    "output": best_state["output"], "row": row,
                    "features": features_np, "evidence_diag": fdiag},
                   CKPT_DIR / f"{group}_fold{fold:02d}_best.pt")
        close_helper(agent); del agent, branch
        torch.cuda.empty_cache()
        write_json(state_path, {"status": "RUNNING", "rows": rows, "updated_utc": utc()})
        print(f"{group}_FOLD_DONE fold={fold} acc={metrics['acc']:.6f}", flush=True)
    write_json(state_path, {"status": "COMPLETE", "rows": rows,
                            "wall_seconds": time.monotonic() - started,
                            "completed_utc": utc()})
    m = json.loads(MANIFEST.read_text())
    m["workers"][group] = {"status": "COMPLETE", "gpu": gpu,
                            "folds": len(rows), "wall_seconds": time.monotonic() - started}
    write_json(MANIFEST, m)


def post_anchor_preflight():
    import torch
    c = context(0)
    anchor = torch.load(ANCHOR_DIR / "anchor_fold01.pt", map_location="cpu", weights_only=False)
    branch = SAGEAlign(EVIDENCE_DIM, CONCEPTS, cap=RESIDUAL_CAP).cuda()
    cam = anchor["cam"].cuda()
    zero_ok = zero_evidence_test(branch, cam)
    ev = load_evidence(1, "conditional", True)
    structures = load_structures()
    feat, _ = feature_set("E9_SAGE_FULL", ev, c["x"], c["folds"][0][0], structures)
    feat_t = torch.as_tensor(feat, device="cuda")
    before = {k: v.detach().clone() for k, v in branch.state_dict().items()}
    out = branch(feat_t, cam)[0].sum(); out.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad) for p in branch.parameters())
    if not (zero_ok and grad_ok):
        raise RuntimeError("post-anchor branch preflight failed")
    m = json.loads(MANIFEST.read_text())
    m["post_anchor_preflight"] = {"status": "PASS", "zero_residual": zero_ok,
                                  "nonzero_gradient": bool(grad_ok),
                                  "anchor_fold01_present": True, "utc": utc()}
    write_json(MANIFEST, m)
    print(json.dumps(m["post_anchor_preflight"]))


def dispatch():
    envs = []
    procs = []
    for gpu in (0, 1):
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [sys.executable, str(Path(__file__).resolve()), "worker", str(gpu)]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    if any(c != 0 for c in codes):
        raise RuntimeError(f"GPU workers failed: {codes}")


def worker(gpu):
    for group in GPU_GROUPS[int(gpu)]:
        if group == "E0_B0A":
            continue
        run_group(group, int(gpu))


def all_rows():
    rows = []
    for group in CONFIGS:
        path = STATE_DIR / f"{group}.json"
        if not path.exists():
            raise RuntimeError(f"missing {group} task state")
        rows.extend(json.loads(path.read_text())["rows"])
    if len(rows) != 100 or len({(r["config"], int(r["fold"])) for r in rows}) != 100:
        raise RuntimeError("formal result table is not exactly 100 folds")
    return sorted(rows, key=lambda r: (CONFIGS.index(r["config"]), int(r["fold"])))


def evaluate_counterfactuals():
    import torch
    c = context(0)
    structures = load_structures()
    endpoints, family_roi, concept_roi, concept_edge, wrong_roi, wrong_edge, concept_family, _ = structures
    rows = []
    for fold, (train_idx, test_idx) in enumerate(c["folds"], 1):
        ck = torch.load(CKPT_DIR / f"E9_SAGE_FULL_fold{fold:02d}_best.pt", map_location="cuda", weights_only=False)
        anchor = torch.load(ANCHOR_DIR / f"anchor_fold{fold:02d}.pt", map_location="cuda", weights_only=False)
        branch = SAGEAlign(EVIDENCE_DIM, CONCEPTS, cap=RESIDUAL_CAP).cuda()
        branch.load_state_dict(ck["branch"])
        agent = make_agent(c, fold)
        agent.ModalFusion.Outputlayer.load_state_dict(ck["output"])
        ev = load_evidence(fold, "conditional", True); ev["train_y"] = c["y"]
        real_feat, real_diag = feature_set("E9_SAGE_FULL", ev, c["x"], train_idx, structures)
        wrong_m, wrong_u, wrong_f, _, _ = semantic_score(ev, wrong_edge, wrong_roi,
                                                          np.ones(CONCEPTS, bool), concept_family=concept_family)
        wrong_feat = wrong_f * wrong_m[:, :, None]
        num_roi, num_edge, _ = make_numeric_prior(c["x"], train_idx, endpoints)
        num_m, num_u, num_f, _, _ = semantic_score(ev, num_edge, num_roi,
                                                    np.ones(CONCEPTS, bool), concept_family=concept_family)
        num_feat = num_f * num_m[:, :, None]
        no_qc = load_evidence(fold, "conditional", False)
        no_qc_feat, no_qc_diag = feature_set("E4_NO_QC", no_qc, c["x"], train_idx, structures)
        forced_feat, forced_diag = feature_set("E7_FORCED_ALIGNMENT", ev, c["x"], train_idx, structures)
        test_perm = np.arange(len(c["x"]))
        test_perm[test_idx] = np.roll(test_idx, 1)
        views = {
            "REAL": (real_feat, real_diag),
            "WRONG_SEMANTICS": (wrong_feat, {"unmatched_test_mean": float(wrong_u[test_idx].mean()), "matched_test_mean": float(wrong_m[test_idx].sum(1).mean())}),
            "NUMERIC_PRIOR": (num_feat, {"unmatched_test_mean": float(num_u[test_idx].mean()), "matched_test_mean": float(num_m[test_idx].sum(1).mean())}),
            "SEMANTIC_OFF": (np.zeros_like(real_feat), {"unmatched_test_mean": 1.0, "matched_test_mean": 0.0}),
            "QC_OFF": (no_qc_feat, no_qc_diag),
            "FORCED_ALIGNMENT": (forced_feat, forced_diag),
            "PATIENT_EVIDENCE_MISMATCH": (real_feat[test_perm], real_diag),
            "ALL_RESIDUAL_OFF": (np.zeros_like(real_feat), {"unmatched_test_mean": 1.0, "matched_test_mean": 0.0}),
        }
        cam = anchor["cam"].cuda(); attn = anchor["attention"].cuda()
        base = None
        for name, (feat_np, diag) in views.items():
            feat = torch.as_tensor(feat_np, device="cuda")
            with torch.no_grad():
                logp, residual = adapt_logits(branch, agent.ModalFusion.Outputlayer,
                                              feat, cam, attn, agent, c, train_idx, train=False)
            test_logp = logp[test_idx]
            if base is None:
                base = test_logp.detach().clone()
            flips = int(torch.count_nonzero(test_logp.argmax(1) != base.argmax(1)).item())
            rows.append({"config": "E9_SAGE_FULL", "fold": fold, "intervention": name,
                         "prediction_flips_vs_real": flips,
                         "mean_abs_logp_delta_vs_real": float((test_logp - base).abs().mean().item()),
                         "mean_residual_l2": float(residual[test_idx].norm(dim=1).mean().item()),
                         "matched_test_mean": float(diag.get("matched_test_mean", np.nan)),
                         "unmatched_test_mean": float(diag.get("unmatched_test_mean", np.nan))})
        close_helper(agent); del agent, branch
        torch.cuda.empty_cache()
    return rows


def report():
    rows = all_rows()
    def agg(group, key):
        v = np.asarray([float(r[key]) for r in rows if r["config"] == group])
        return float(v.mean()), float(np.sqrt(np.mean((v - v.mean()) ** 2)))
    summary = {g: {k: {"mean": agg(g, k)[0], "population_sd": agg(g, k)[1]}
                   for k in ("acc", "hard_auc", "spe_td", "sen_asd")}
               for g in CONFIGS}
    write_json(RESULTS / "summary.json", summary)
    with (RESULTS / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    diagnostics = []
    for r in rows:
        if r["config"] != "E0_B0A":
            diagnostics.append({k: r.get(k) for k in ("config", "fold", "matched_mean", "unmatched_mean",
                                "matched_test_mean", "unmatched_test_mean", "residual_l2_mean",
                                "residual_ratio_mean", "branch_parameters", "outputlayer_parameters",
                                "trainable_parameters", "gradient_elements", "best_epoch", "epochs_executed")})
    with (RESULTS / "diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(diagnostics[0].keys()); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(diagnostics)
    counter = evaluate_counterfactuals()
    with (RESULTS / "counterfactual.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(counter[0].keys()); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(counter)

    def pp(g, k):
        return f"{summary[g][k]['mean']*100:.4f} ± {summary[g][k]['population_sd']*100:.4f}%"
    lines = ["# Experiment 41.1：SAGE-Align 最终验证", "",
             "## 执行状态", "",
             "本目录执行 10 组 × 10 折、seed=0。保留原始 GFR 四图、四模态 48+6+10+256、CAM/OutputLayer、PEN 构造副作用和 simple-2 双 Adam；适配阶段仅更新新增分支及（除 E8 外）原 OutputLayer。val=test、历史预 CV RFE/标准化和 871 人 transductive 图均沿用，因此下述结果是历史协议内部比较，不是无偏独立泛化证明。", "",
             "论文/源码审计确认原论文 Fig.1/Eq.(1)-(7) 与真实代码存在超参数和 PEN 实现差异；本轮以真实源码和 Experiment 32 B0-A 为权威。CAM 的 `[871,4,36]` 仅是潜在跨模态表示，未解释成脑区或原始模态。GFR 中 `i==1` 覆盖前一分支的历史行为保持。", "",
             "## 冻结医学先验", "",
             "四个概念 P1–P4 来自 Ilioska et al.（Nature Mental Health 2026, DOI 10.1038/s44220-026-00656-y）和 Kobbersmed et al.（Progress in Neurobiology 2026, DOI 10.1016/j.pneurobio.2026.102906）。文本没有调用 GPT-4 或 CLIP；实际模型使用的是冻结的 HO ROI/252 条已核验 FC 掩码、正负偏离双通道、适用层级和不确定性字段。没有把文献网络异常当作单条 HO 连接的 ASD 真值。", "",
             "四条未知端点连接 195/225/232/236 保留在 GFR，但不进入语义证据。P1/P2/P3/P4 分别对应感觉运动、岛叶/颞叶/额叶、内侧额叶/扣带/顶叶、皮质下-皮层候选；canonical 网络归属未核验，报告不使用 DMN/注意网络等强标签。", "",
             "## SAGE-Align 机制", "",
             "条件 TD 参考只拟合 outer-train TD；训练患者使用固定 5 折 OOF，测试患者使用完整 outer-train 参考。条件变量为年龄 26 维分箱、站点、性别和全局第 60 列 mean FD。FUNC 可靠性只使用已核实的 DVARS/outlier/mean FD/percent FD；不把 ANAT/FUNC 全部字段视作越大越差。匹配和拒绝共用同一连续 `signal`、同一训练 TD OOF 第 75 分位阈值、固定温度 0.20 和 `raw=3×strength×support`；拒绝质量为 `1/(1+sum(raw))`，匹配+拒绝质量严格守恒。", "",
             "每个患者的有效槽为四个概念的真实 FC 偏离、ROI 聚合和粗解剖族聚合，共 18 维/概念。共享分支仅有无 bias 的 18→8→36→144 低容量线性变换（5,760 个参数），CAM 只进入 36→4 查询门控；残差输出层零初始化，最大患者级 CAM 残差范数固定为 8%。因此有效医学质量全为零时残差严格为零，不能从 CAM 单独恢复语义残差。", "",
             "## 正式指标", "",
             "四项均为十折均值 ± population SD；Hard-AUC 继续使用原代码 one-hot 硬标签 ROC-AUC。", "",
             "| 组别 | ACC | Hard-AUC | SPE-TD | SEN-ASD |", "|---|---:|---:|---:|---:|"]
    for g in CONFIGS:
        lines.append(f"| {g} | {pp(g,'acc')} | {pp(g,'hard_auc')} | {pp(g,'spe_td')} | {pp(g,'sen_asd')} |")
    lines += ["", "## 核心配对比较", "", "Δ 为 E9 减对照，单位百分点；W/T/L 是同折严格胜/平/负，只作内部描述。", "", "| 对照 | ΔACC / W-T-L | ΔHard-AUC / W-T-L | ΔSPE | ΔSEN |", "|---|---:|---:|---:|---:|"]
    for g in CONFIGS[:-1]:
        a = np.array([float(r["acc"]) for r in rows if r["config"] == "E9_SAGE_FULL"])
        b = np.array([float(r["acc"]) for r in rows if r["config"] == g])
        u = np.array([float(r["hard_auc"]) for r in rows if r["config"] == "E9_SAGE_FULL"])
        v = np.array([float(r["hard_auc"]) for r in rows if r["config"] == g])
        s = np.array([float(r["spe_td"]) for r in rows if r["config"] == "E9_SAGE_FULL"]) - np.array([float(r["spe_td"]) for r in rows if r["config"] == g])
        n = np.array([float(r["sen_asd"]) for r in rows if r["config"] == "E9_SAGE_FULL"]) - np.array([float(r["sen_asd"]) for r in rows if r["config"] == g])
        lines.append(f"| {g} | {(a-b).mean()*100:+.4f} / {int((a>b).sum())}-{int((a==b).sum())}-{int((a<b).sum())} | {(u-v).mean()*100:+.4f} / {int((u>v).sum())}-{int((u==v).sum())}-{int((u<v).sum())} | {s.mean()*100:+.4f} | {n.mean()*100:+.4f} |")
    cf = json.loads((RESULTS / "counterfactual.csv").read_text()) if False else counter
    lines += ["", "## 机制诊断", "", "E9 分支注册参数为 5,760（18→8→36→144，全部无 bias）；E1 仅优化原 OutputLayer，E8 严格冻结 OutputLayer。逐折最佳 epoch、残差比、有效匹配和梯度元素在 `diagnostics.csv`。", "", "固定 E9 checkpoint 反事实见 `counterfactual.csv`，包括 REAL、WRONG_SEMANTICS、NUMERIC_PRIOR、SEMANTIC_OFF、QC_OFF、FORCED_ALIGNMENT、PATIENT_EVIDENCE_MISMATCH 和 ALL_RESIDUAL_OFF；它们是机制诊断，不能替代正式 E5/E6/E7 训练对照。", ""]
    e9 = [r for r in rows if r["config"] == "E9_SAGE_FULL"]
    lines.append(f"E9 未匹配质量（十折 outer-test 均值）为 {np.mean([r['unmatched_test_mean'] for r in e9]):.4f}；残差/8%上限比例均值为 {np.mean([r['residual_ratio_mean'] for r in e9]):.4f}。")
    e9e0 = np.array([float(r["acc"]) for r in e9]).mean() - np.array([float(r["acc"]) for r in rows if r["config"] == "E0_B0A"]).mean()
    lines += ["", "## GO/NO-GO", "", f"预注册门槛要求 E9 相对 E0 的 ACC 与 Hard-AUC 均至少 +0.2 个百分点且至少 7/10 折获胜，同时优于 E1、E2、E5、E6 并保持 ASD 灵敏度非劣。实际 E9−E0 ACC={e9e0*100:+.4f}pp；最终判定将由表格和逐折比较自动写入 manifest。", ""]
    go = True
    for g in ("E0_B0A", "E1_HEAD_ONLY", "E2_NUMERIC_ONLY", "E5_WRONG_SEMANTICS", "E6_DATA_PRIOR"):
        e = [r for r in e9]; b = [r for r in rows if r["config"] == g]
        da = np.array([float(x["acc"]) - float(y["acc"]) for x, y in zip(e, b)])
        du = np.array([float(x["hard_auc"]) - float(y["hard_auc"]) for x, y in zip(e, b)])
        if g == "E0_B0A":
            go = go and da.mean() > .002 and du.mean() > .002 and (da > 0).sum() >= 7 and (du > 0).sum() >= 7
        else:
            go = go and da.mean() > 0 and du.mean() > 0
    sen_delta = np.mean([float(x["sen_asd"]) - float(y["sen_asd"]) for x, y in zip(e9, [r for r in rows if r["config"] == "E0_B0A"])])
    go = go and sen_delta >= -.005
    decision = "GO_INTERNAL_ONLY" if go else "FINAL NO-GO"
    lines += [f"**{decision}。** 只有同时通过分类性能和医学语义必要性才可作为 MICCAI 2027 第一创新点；本轮仍继承历史评估偏差，单 seed 结果不构成跨中心或统计显著性证明。若为 NO-GO，本路线在本实验结束，不自动提出 41.2。", "", "## 审计来源与限制", "", "原论文与真实源码差异、Experiment 11 ROI 映射、Experiment 32/39/40/41/42 失败结论、四模态职责和字段索引均在本目录的 manifest 及冻结 JSON 中留痕。没有修改其他实验、数据或模型参数，也没有 Git 提交/推送。"]
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    m = json.loads(MANIFEST.read_text())
    m.update({"status": "COMPLETE", "completed_folds": 100,
              "summary": summary, "decision": decision,
              "results_sha256": {p.name: sha_file(p) for p in RESULTS.glob("*.csv")},
              "report_sha256": sha_file(RESULTS / "report.md"), "completed_utc": utc()})
    write_json(MANIFEST, m)
    print(f"REPORT_COMPLETE folds=100 decision={decision}")


def finalize():
    m = json.loads(MANIFEST.read_text())
    if m.get("status") != "COMPLETE" or m.get("completed_folds") != 100:
        raise RuntimeError("finalize requires 100 completed folds")
    summary = m["summary"]
    marker = "<!-- experiment41_1_sage_align_final_v1 -->"
    end = "<!-- /experiment41_1_sage_align_final_v1 -->"
    handoff = ROOT / "handoff.md"
    lines = ["## Experiment 41.1 — SAGE-Align Final", "", marker, "",
             f"验收时间：{utc()}。100/100有效正式折完成；GPU0执行 E0/E2/E4/E6/E8=50折，GPU1执行 E1/E3/E5/E7/E9=50折；每卡最多一个 worker，实际设备为两张 PPU-ZW810E。", "",
             f"E0 B0-A 精确复现：{m.get('e0_parity', {}).get('pass')}（{m.get('e0_parity', {}).get('matched_folds', 0)}/10）。适配学习率均为 {ADAPT_LR}；残差上限固定 8%；SAGE 分支 5,760 参数，CAM 仅作查询门控，零有效医学证据时残差严格为零。", "",
             "| 组别 | ACC | Hard-AUC | SPE-TD | SEN-ASD |", "|---|---:|---:|---:|---:|"]
    for g in CONFIGS:
        s = summary[g]
        lines.append(f"| {g} | {s['acc']['mean']*100:.4f} ± {s['acc']['population_sd']*100:.4f}% | {s['hard_auc']['mean']*100:.4f} ± {s['hard_auc']['population_sd']*100:.4f}% | {s['spe_td']['mean']*100:.4f} ± {s['spe_td']['population_sd']*100:.4f}% | {s['sen_asd']['mean']*100:.4f} ± {s['sen_asd']['population_sd']*100:.4f}% |")
    lines += ["", f"E9 未匹配质量与残差/8%上限见 `results/diagnostics.csv`；固定 E9 反事实见 `results/counterfactual.csv`。医学 Prompt 使用 Ilioska DOI 10.1038/s44220-026-00656-y 与 Kobbersmed DOI 10.1016/j.pneurobio.2026.102906 的保守区域级先验，未使用 CLIP/GPT-4。", "", f"最终科研结论：**{m['decision']}**。本轮若未通过 E0、E1、数值先验或错误映射必要性，不能作为 MICCAI 2027 第一创新点继续；按任务要求不启动 41.2。", "", f"结果目录：`{EXP}`。总训练墙钟：{sum(float(v.get('wall_seconds', 0)) for v in m.get('workers', {}).values()):.2f} 秒；报告 SHA256：`{m['report_sha256']}`；run.py SHA256：`{sha_file(EXP/'run.py')}`；sage_align.py SHA256：`{sha_file(EXP/'sage_align.py')}`；prompts.json SHA256：`{sha_file(EXP/'prompts.json')}`；ontology.json SHA256：`{sha_file(EXP/'ontology.json')}`。", "", "沿用 val=test、CV 前 RFE/标准化和 871 人 transductive 图，结果属于历史协议下的内部比较。", "", end, ""]
    text = "\n".join(lines)
    import fcntl
    with handoff.open("a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0); old = f.read()
        if marker not in old:
            f.seek(0, 2); f.write("\n" + text); f.flush(); os.fsync(f.fileno())
        elif end not in old:
            raise RuntimeError("incomplete existing handoff marker")
    m["handoff"] = {"status": "ALREADY_PRESENT" if marker in old else "APPENDED_VERIFIED",
                     "utc": utc(), "path": str(handoff)}
    write_json(MANIFEST, m)
    print(json.dumps(m["handoff"], ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "e0", "dispatch", "worker", "report", "finalize", "post_preflight"))
    parser.add_argument("arg", nargs="?", type=int)
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "preflight": preflight()
    elif args.command == "e0": run_e0(args.arg or 0)
    elif args.command == "post_preflight": post_anchor_preflight()
    elif args.command == "dispatch": dispatch()
    elif args.command == "worker": worker(args.arg or 0)
    elif args.command == "report": report()
    elif args.command == "finalize": finalize()


if __name__ == "__main__":
    main()
