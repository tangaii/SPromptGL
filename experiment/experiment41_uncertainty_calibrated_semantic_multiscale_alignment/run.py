#!/usr/bin/env python3
"""Experiment 41 UCSMA-v2: isolated prepare, preflight, train and report."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import difflib
import fcntl
import hashlib
import importlib
import io
import json
import os
import pickle
import random
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np

from integration import emit
from ucsma import (UCSMAResidual, fold_normative_features, pool_mask,
                   ridge_fit, semantic_mass, summarize_concepts)

ROOT = Path("/data/adodas/SPromptGL-main")
EXP = Path(__file__).resolve().parent
RUNTIME = EXP / ".runtime"
SOURCE = RUNTIME / "source"
RESULTS = EXP / "results"
MANIFEST_PATH = EXP / "manifest.json"
INPUT_DIMS = [48, 6, 10, 256]
CONFIGS = ("E0_B0A", "E1_RAW_CAPACITY", "E2_GLOBAL_NORM", "E3_COND_NORM",
           "E4_ROI_NORM", "E5_MULTISCALE_NUM", "E6_REAL_NO_REJECT",
           "E7_WRONG_SEMANTICS", "E8_DATA_PRIOR", "E9_UCSMA_FULL")
GPU_CONFIGS = {0: ("E0_B0A", "E2_GLOBAL_NORM", "E4_ROI_NORM",
                   "E6_REAL_NO_REJECT", "E8_DATA_PRIOR"),
               1: ("E1_RAW_CAPACITY", "E3_COND_NORM", "E5_MULTISCALE_NUM",
                   "E7_WRONG_SEMANTICS", "E9_UCSMA_FULL")}
PROTOCOL = dict(datadir=str(ROOT / "data") + "/", datname="ABIDE", cpu=False,
                gpu="-1", nepoch=450, early=200, lr=.0038, reg=.11,
                dropout=.35, nlayer=1, n_hidden=18, n_head=2, n_iter=10,
                nmodal=4, th=.9, GC_mode="weighted-cosine", MP_mode="GCN",
                MF_mode="concat", alpha=.5, theta_smooth=1.,
                theta_degree=.5, theta_sparsity=0., nclass=2,
                mode="simple-2", seed=0)
CHECKPOINT_EVERY = 25
HIST32 = ROOT / "experiment/experiment32_disease_contrastive_evidence_routing"
HIST11 = ROOT / "experiment/experiment11_roi_mapping_recovery"
DATA_FILES = ("data/ABIDE/processed_standard_data.csv",
              "data/ABIDE/modal_feat_dict.npy")
CORE_FILES = ("main.py", "model.py", "network.py", "layers.py", "utils.py",
              "ABIDE-simple-2-concat-weighted-cosine.sh")
HISTORY_FILES = (
    "experiment/experiment32_disease_contrastive_evidence_routing/manifest.json",
    "experiment/experiment32_disease_contrastive_evidence_routing/results/fold_metrics.csv",
    "experiment/experiment32_disease_contrastive_evidence_routing/results/report.md",
    "experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv",
    "experiment/experiment11_roi_mapping_recovery/results/roi_111_labels.csv",
    "experiment/experiment11_roi_mapping_recovery/results/roi_256_mapping.csv",
)
PAPER_PATHS = tuple(ROOT.glob("2025 MICCAI SPromptGL*.pdf"))
EXTERNAL_HISTORY_REPORTS = (
    Path("/tmp/SPromptGL-E38-B0A-stage/experiment/experiment38_lcher_b0a_anchor_rerun/results/report.md"),
    Path("/tmp/SPromptGL-E39-HASE-stage/experiment/experiment39_heterogeneity_aware_semantic_experts/results/report.md"),
    Path("/tmp/SPromptGL-E40-HCER-v2-stage/experiment/experiment40_hcer_v2_b0a_evidence/results/report.md"),
    Path("/tmp/SPromptGL-E34-SeRCA-stage-20261022/experiment/experiment34_serca_semantic_relational_alignment/results/report.md"),
    Path("/tmp/SPromptGL-E35-SeRCA-v2-stage/experiment/experiment35_serca_v2_relational_pattern_alignment/results/report.md"),
    Path("/tmp/SPromptGL-E36-PCPR-stage/experiment/experiment36_phenotype_conditioned_patient_routing/results/report.md"),
    Path("/tmp/SPromptGL-E37-LSPCR-stage/experiment/experiment37_label_supervised_phenotype_routing/results/report.md"),
)
FAMILY_PATTERNS = {
    "sensorimotor": r"Precentral Gyrus|Postcentral Gyrus|Juxtapositional Lobule|Central Opercular Cortex|Parietal Opercular Cortex",
    "frontal_association": r"Frontal Pole|Superior Frontal Gyrus|Middle Frontal Gyrus|Inferior Frontal Gyrus|Frontal Medial Cortex|Frontal Orbital Cortex",
    "parietal_association": r"Superior Parietal Lobule|Supramarginal Gyrus|Angular Gyrus",
    "temporal_association": r"Temporal Pole|Temporal Gyrus|Temporal Fusiform|Planum Polare|Planum Temporale|Heschl's Gyrus",
    "insula": r"Insular Cortex",
    "cingulate_medial": r"Subcallosal Cortex|Paracingulate Gyrus|Cingulate Gyrus|Precuneous Cortex",
    "visual": r"Lateral Occipital Cortex|Intracalcarine Cortex|Cuneal Cortex|Lingual Gyrus|Supracalcarine Cortex|Occipital",
    "subcortical": r"^(Left|Right) (Thalamus|Caudate|Putamen|Pallidum|Hippocampus|Amygdala|Accumbens)$",
}
CONCEPT_PATTERNS = {
    "C01": r"Precentral Gyrus|Postcentral Gyrus|Juxtapositional Lobule|Central Opercular Cortex|Parietal Opercular Cortex|Superior Parietal Lobule|Supramarginal Gyrus|Angular Gyrus",
    "C02": r"Insular Cortex|Temporal Pole|Superior Temporal Gyrus, anterior division|Middle Temporal Gyrus, anterior division|Frontal Pole|Superior Frontal Gyrus|Middle Frontal Gyrus|Inferior Frontal Gyrus|Frontal Medial Cortex|Frontal Orbital Cortex",
    "C03": r"Frontal Medial Cortex|Subcallosal Cortex|Paracingulate Gyrus|Cingulate Gyrus|Angular Gyrus|Supramarginal Gyrus",
    "C06": r"^(Left|Right) (Thalamus|Caudate|Putamen|Pallidum|Hippocampus|Amygdala|Accumbens)$",
}
CONCEPT_IDS = ("C01", "C02", "C03", "C04", "C05", "C06", "C07")
ELIGIBLE = np.array([True, True, True, False, False, True, False], dtype=bool)
CONCEPT_TAIL_QUANTILE = 0.90
SEMANTIC_MATCH_SCALE = 20.0
FIELD_SIZES = (144, 256, 256 * 6, 111 * 6, 8 * 6, 7 * 6, 8)
FEATURE_DIM = sum(FIELD_SIZES)
_CTX = None


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path | str) -> str:
    return sha_bytes(Path(path).read_bytes())


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def json_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(native(value), stream, indent=2, ensure_ascii=False,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def update_manifest(mutator):
    lock_path = RUNTIME / "manifest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        mutator(manifest)
        json_write(MANIFEST_PATH, manifest)


def csv_write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{k: native(v) for k, v in row.items()} for row in rows])
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def torch_write(path: Path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def torch_load(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def tree_equal(left, right):
    import torch
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(
            left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and
                left.keys() == right.keys() and
                all(tree_equal(left[k], right[k]) for k in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and
                len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right)))
    return left == right


def file_hashes(paths):
    return {str(p): sha_file(p) for p in paths if Path(p).is_file()}


def source_hashes():
    return {name: sha_file(ROOT / name) for name in CORE_FILES}


def code_hashes():
    return {name: sha_file(EXP / name) for name in
            ("README.md", "prompts.json", "integration.py", "integration.patch",
             "ucsma.py", "run.py", "ontology.json") if (EXP / name).is_file()}


def seed_everything(torch, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_capture(torch):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def rng_restore(torch, state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_hash(state):
    digest = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    digest.update(state["cpu"].cpu().numpy().tobytes())
    for value in state["cuda"]:
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def parameter_hash(agent):
    digest = hashlib.sha256()
    for module_name, module in (("gfr", agent.MessagePassing0),
                                ("pln", agent.ModalFusion)):
        for name, parameter in sorted(module.named_parameters()):
            digest.update(module_name.encode() + b"/" + name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def close_helper(agent):
    for name in ("MF_sav", "GCMP_sav"):
        stream = getattr(agent, name, None)
        if stream is not None:
            stream.close()


def read_data():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    base = ROOT / "data/ABIDE"
    frame = pd.read_csv(base / "processed_standard_data.csv")
    mapping = np.load(base / "modal_feat_dict.npy", allow_pickle=True).item()
    if list(mapping) != ["PHENO", "ANAT", "FUNC", "Correlation"]:
        raise AssertionError(f"Unexpected modality order: {list(mapping)}")
    if [len(mapping[k]) for k in mapping] != INPUT_DIMS:
        raise AssertionError("Input dimensions must remain [48,6,10,256]")
    names = [str(name) for key in mapping for name in mapping[key]]
    if list(frame.columns[:-1]) != names or frame.shape != (871, 321):
        raise AssertionError("Preprocessed ABIDE columns/shape changed")
    pnames = [str(v) for v in mapping["PHENO"]]
    fnames = [str(v).lower() for v in mapping["FUNC"]]
    if not (all(v.startswith("AGE_AT_SCAN_") for v in pnames[:26]) and
            all(v.startswith("SITE_ID_") for v in pnames[26:46]) and
            all(v.startswith("SEX_") for v in pnames[46:48])):
        raise AssertionError("PHENO age/site/sex identities differ from frozen layout")
    if len(fnames) != 10 or "mean_fd" not in fnames[6]:
        raise AssertionError(f"FUNC index 6 is not verified mean FD: {fnames}")
    if "mean_fd" not in names[60].lower():
        raise AssertionError("Conditional reference must read global mean FD column 60")
    x = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = frame.iloc[:, -1].to_numpy(dtype=np.int64) - 1
    if not np.isfinite(x).all() or np.bincount(y).tolist() != [403, 468]:
        raise AssertionError("Expected finite 871x320 data; ASD=0:403, TD=1:468")
    folds = list(StratifiedKFold(n_splits=10, shuffle=True,
                                 random_state=10).split(x, y))
    return x, y, folds, mapping


def make_structures():
    import pandas as pd
    conn = pd.read_csv(HIST11 / "results/roi_256_named_mapping.csv")
    roi_frame = pd.read_csv(HIST11 / "results/roi_111_labels.csv")
    endpoints = np.full((256, 2), -1, dtype=np.int16)
    known = conn[conn.connection_anatomical_status == "VERIFIED_BOTH_ENDPOINTS"]
    if len(known) != 252 or known.correlation_feature_index.nunique() != 252:
        raise AssertionError("Experiment11 endpoint map no longer has 252 unique verified edges")
    for row in known.itertuples(index=False):
        endpoints[int(row.correlation_feature_index)] = [int(row.roi_i_zero_based),
                                                         int(row.roi_j_zero_based)]
    unknown = np.flatnonzero((endpoints < 0).any(axis=1)).tolist()
    if unknown != [195, 225, 232, 236]:
        raise AssertionError(f"Unknown endpoint inputs changed: {unknown}")
    names = roi_frame.roi_name.astype(str).tolist()
    degree = np.zeros(111, dtype=np.int64)
    for a, b in endpoints:
        if a >= 0:
            degree[a] += 1
            degree[b] += 1
    if (int((degree >= 3).sum()), int(((degree > 0) & (degree < 3)).sum()),
            int((degree == 0).sum()), int(degree.max())) != (91, 19, 1, 12):
        raise AssertionError("Verified ROI coverage distribution changed")

    def roi_mask(pattern):
        rx = re.compile(pattern, re.IGNORECASE)
        return np.asarray([bool(rx.search(name)) and degree[i] > 0
                           for i, name in enumerate(names)], dtype=bool)

    family_masks = np.stack([roi_mask(pattern) for pattern in FAMILY_PATTERNS.values()])
    real_roi_masks = np.zeros((7, 111), dtype=bool)
    for k, concept_id in enumerate(CONCEPT_IDS):
        if concept_id in CONCEPT_PATTERNS:
            real_roi_masks[k] = roi_mask(CONCEPT_PATTERNS[concept_id])
    concept_edges = _edge_masks(endpoints, real_roi_masks, subcortical_index=5,
                                roi_names=names)
    family_edges = _edge_masks(endpoints, family_masks, subcortical_index=None,
                               roi_names=names)

    # Degree-stratified derangement preserves each concept's ROI count, overlap,
    # and low-coverage distribution while scrambling atlas identity.
    rng = np.random.default_rng(4107)
    permutation = np.arange(111)
    subcortical = np.asarray([bool(re.match(
        r"^(Left|Right) (Thalamus|Caudate|Putamen|Pallidum|Hippocampus|Amygdala|Accumbens)$",
        name)) for name in names])
    for d, is_subcortical in sorted(set(zip(degree.tolist(), subcortical.tolist()))):
        ids = np.flatnonzero((degree == d) & (subcortical == is_subcortical))
        if len(ids) <= 1:
            continue
        for _ in range(100):
            candidate = rng.permutation(ids)
            candidate = np.roll(candidate, 1)
            if np.all(candidate != ids):
                permutation[ids] = candidate
                break
        else:
            permutation[ids] = np.roll(ids, 1)
    wrong_roi_masks = np.zeros_like(real_roi_masks)
    for k in range(len(CONCEPT_IDS)):
        wrong_roi_masks[k, permutation[np.flatnonzero(real_roi_masks[k])]] = True
    wrong_edges_raw = _edge_masks(endpoints, wrong_roi_masks, subcortical_index=5,
                                  roi_names=names)
    # Preserve each real concept's edge count while changing its atlas
    # correspondence. The chosen candidates are deterministic and drawn only
    # from verified edges; no outer-test value enters this control.
    wrong_edges = np.zeros_like(wrong_edges_raw)
    for k in range(len(CONCEPT_IDS)):
        target_count = int(concept_edges[k].sum())
        candidates = np.flatnonzero((endpoints >= 0).all(axis=1))
        chosen = rng.permutation(candidates)[:target_count]
        wrong_edges[k, chosen] = True

    concepts = json.loads((EXP / "prompts.json").read_text(encoding="utf-8"))["concepts"]
    for concept in concepts:
        k = CONCEPT_IDS.index(concept["concept_id"])
        concept["supported_roi_indices"] = np.flatnonzero(real_roi_masks[k]).astype(int).tolist()
        concept["supported_edge_indices"] = np.flatnonzero(concept_edges[k]).astype(int).tolist()
        concept["frozen_sha256"] = sha_bytes(json.dumps(
            {key: value for key, value in concept.items() if key != "frozen_sha256"},
            sort_keys=True, ensure_ascii=False).encode())
    prompts = json.loads((EXP / "prompts.json").read_text(encoding="utf-8"))
    prompts["concepts"] = concepts
    json_write(EXP / "prompts.json", prompts)
    ontology = {
        "atlas": "Harvard-Oxford labels as audited in Experiment11",
        "canonical_functional_networks_verified": False,
        "roi_count": 111,
        "verified_fc_count": 252,
        "unknown_fc_feature_indices": unknown,
        "roi_names": names,
        "roi_degree": degree.tolist(),
        "roi_endpoints_by_fc_feature": endpoints.tolist(),
        "anatomical_families": [{
            "family_id": key,
            "interpretation": "coarse anatomical-name proxy; not a canonical functional network",
            "roi_indices": np.flatnonzero(family_masks[i]).astype(int).tolist(),
            "edge_indices": np.flatnonzero(family_edges[i]).astype(int).tolist(),
        } for i, key in enumerate(FAMILY_PATTERNS)],
        "concepts": [{
            "concept_id": concept["concept_id"],
            "roi_indices": np.flatnonzero(real_roi_masks[k]).astype(int).tolist(),
            "edge_indices": np.flatnonzero(concept_edges[k]).astype(int).tolist(),
            "eligible": bool(concept.get("model_eligible", False) and concept_edges[k].any()),
            "edge_rule": "at least one verified endpoint in candidate ROI set" if concept["concept_id"] != "C06" else "exactly one candidate subcortical endpoint and one cortical endpoint",
        } for k, concept in enumerate(concepts)],
        "wrong_semantics_control": {
            "seed": 4107,
            "permutation": permutation.tolist(),
            "roi_masks": wrong_roi_masks.astype(int).tolist(),
            "edge_masks": wrong_edges.astype(int).tolist(),
            "preserved": "ROI count, pairwise ROI-mask overlap, and ROI degree strata",
            "real_vs_wrong_changed_roi_assignments": int(np.count_nonzero(
                real_roi_masks[:4] != wrong_roi_masks[:4])),
        },
        "region_family_masks": family_masks.astype(int).tolist(),
        "region_family_edge_masks": family_edges.astype(int).tolist(),
        "real_concept_roi_masks": real_roi_masks.astype(int).tolist(),
        "real_concept_edge_masks": concept_edges.astype(int).tolist(),
        "roi_map_source_sha256": sha_file(HIST11 / "results/roi_256_named_mapping.csv"),
        "roi_labels_source_sha256": sha_file(HIST11 / "results/roi_111_labels.csv"),
    }
    json_write(EXP / "ontology.json", ontology)
    return endpoints, family_masks, family_edges, real_roi_masks, concept_edges, ontology


def _edge_masks(endpoints: np.ndarray, roi_masks: np.ndarray,
                subcortical_index: int | None,
                roi_names: list[str] | None = None) -> np.ndarray:
    result = np.zeros((roi_masks.shape[0], 256), dtype=bool)
    # Region evidence includes all verified incident connections; C06 uses
    # candidate subcortical-to-cortical edges only.
    for k, mask in enumerate(roi_masks):
        if k == subcortical_index:
            continue
        for e, (a, b) in enumerate(endpoints):
            if a >= 0 and (mask[a] or mask[b]):
                result[k, e] = True
    # C06: use atlas category verified by the Experiment11 ROI names.
    if subcortical_index is not None and roi_masks.shape[0] > subcortical_index:
        if roi_names is None:
            raise ValueError("ROI names are required for C06 endpoint typing")
        subcortical = np.asarray([bool(re.match(
            r"^(Left|Right) (Thalamus|Caudate|Putamen|Pallidum|Hippocampus|Amygdala|Accumbens)$",
            name)) for name in roi_names])
        for e, (a, b) in enumerate(endpoints):
            if a >= 0 and ((subcortical[a] and not subcortical[b]) or
                           (subcortical[b] and not subcortical[a])):
                # C06 is eligible only when its ROI candidate is the subcortical endpoint.
                if roi_masks[subcortical_index, a] or roi_masks[subcortical_index, b]:
                    result[subcortical_index, e] = True
    return result


def prepare():
    if MANIFEST_PATH.is_file():
        previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if previous.get("status") not in ("PREPARED", "PREFLIGHT_PASS"):
            raise RuntimeError("Frozen experiment cannot be re-prepared after training starts")
    for path in (RUNTIME, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
    patch = emit(ROOT, SOURCE, EXP / "integration.patch")
    x, y, folds, mapping = read_data()
    structures = make_structures()
    remote_head = "unknown"
    try:
        remote_head = subprocess.check_output(
            ["git", "ls-remote", "https://github.com/tangaii/SPromptGL.git", "refs/heads/main"],
            text=True, timeout=20).split()[0]
    except Exception as exc:
        remote_head = f"unavailable: {type(exc).__name__}: {exc}"
    historical = json.loads((HIST32 / "manifest.json").read_text(encoding="utf-8"))
    local_reports = []
    for number in list(range(17, 20)) + list(range(28, 34)):
        local_reports.extend((ROOT / "experiment").glob(
            f"experiment{number}_*/results/report.md"))
    historical_report_hashes = file_hashes(local_reports + list(EXTERNAL_HISTORY_REPORTS))
    environment = {"python": sys.version,
                   "platform": __import__("platform").platform()}
    try:
        import torch
        environment.update(torch=torch.__version__, cuda=torch.version.cuda,
                           cuda_available=torch.cuda.is_available(),
                           visible_device_count=torch.cuda.device_count())
    except Exception as exc:
        environment["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from importlib.metadata import version
        for package in ("numpy", "scikit-learn", "pandas", "scipy"):
            environment[package] = version(package)
    except Exception:
        pass
    source = {"local_git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "github_main_head": remote_head,
        "core_sha256": source_hashes(),
        "data_sha256": file_hashes([ROOT / p for p in DATA_FILES]),
        "history_sha256": file_hashes([ROOT / p for p in HISTORY_FILES]),
        "historical_reports_sha256": historical_report_hashes,
        "experiment32_source_sha256": historical.get("source_sha256", {}),
        "experiment32_historical_fold_csv_sha256": sha_file(
            HIST32 / "results/fold_metrics.csv"),
        "prompt_source_sha256": sha_file(EXP / "prompts.json"),
        "experiment_code_sha256": code_hashes(),
        "ontology_sha256": sha_file(EXP / "ontology.json"),
        "local_paper_sha256": file_hashes(PAPER_PATHS),
        "patch_sha256": sha_bytes(patch.encode()),
    }
    config = {
        "experiment": "Experiment 41 — UCSMA-v2",
        "protocol": PROTOCOL,
        "seed": 0,
        "outer_folds": 10,
        "folding": "StratifiedKFold(n_splits=10, shuffle=True, random_state=10)",
        "feature_shape": list(x.shape),
        "label_counts": {"ASD_0": int((y == 0).sum()), "TD_1": int((y == 1).sum())},
        "modal_dimensions": {key: len(mapping[key]) for key in mapping},
        "groups": CONFIGS,
        "gpu_assignment": GPU_CONFIGS,
        "branch_feature_fields": FIELD_SIZES,
        "branch_feature_dim": FEATURE_DIM,
        "branch_hidden_dim": 64,
        "branch_residual_l2_bound": 0.05,
        "branch_output_initialization": "zero weight, no bias",
        "branch_weight_decay": 0.0,
        "normative": {"fit_rows": "outer-train TD only", "inner_crossfit": "5-fold KFold over outer-train participants; each fit uses inner-train TD only", "ridge_alpha": 10.0, "residual_scale": "1.4826*MAD with 0.05 lower bound", "edge_threshold": "outer-train TD OOF |z| 95th quantile, minimum 0.5", "concept_threshold_quantile": CONCEPT_TAIL_QUANTILE, "semantic_match_scale": SEMANTIC_MATCH_SCALE, "z_clip": 4.0, "condition_support": "age-bin/site/sex support and ridge leverage", "qc": "ANAT6/FUNC10 robust distance reliability proxy; no disease interpretation"},
        "selection": "val_idx=test_idx; strict ACC improvement; 450 epochs; wait>200",
        "go_screen": {"comparators": ["E0_B0A", "E5_MULTISCALE_NUM", "E7_WRONG_SEMANTICS", "E8_DATA_PRIOR"], "mean_gain_pp": 0.2, "fold_wins_min": 7, "sen_asd_noninferiority_pp": -0.5},
    }
    manifest = {"status": "PREPARED", "prepared_utc": utc(),
                "source_and_input": source, "configuration": config,
                "environment": environment,
                "preflight": None, "workers": {}, "completed_folds": 0,
                "fold_status": {}, "e0_history_verification": None,
                "issues": [], "patch_sha256": source["patch_sha256"]}
    json_write(MANIFEST_PATH, manifest)
    print(json.dumps({"status": "PREPARED", "local_head": source["local_git_head"],
                      "github_head": remote_head, "known_edges": 252,
                      "unknown_edges": [195, 225, 232, 236],
                      "feature_dim": FEATURE_DIM}, ensure_ascii=False))


def context(gpu: int):
    global _CTX
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Worker must see exactly one assigned CUDA device")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match assigned physical GPU")
    torch.cuda.set_device(0)
    sys.path.insert(0, str(SOURCE))
    for name in ("model", "network", "layers", "utils"):
        sys.modules.pop(name, None)
    model = importlib.import_module("model")
    network = importlib.import_module("network")
    utils = importlib.import_module("utils")
    x, y, folds, modalities = read_data()
    _CTX = dict(torch=torch, model=model, network=network, utils=utils,
                x=x, y=y, folds=folds, modalities=modalities,
                device=torch.device("cuda"))
    return _CTX


def make_agent(c, fold):
    train_idx, test_idx = c["folds"][fold - 1]
    hp = SimpleNamespace(**PROTOCOL)
    with contextlib.redirect_stdout(io.StringIO()):
        agent = c["model"].EvalHelper(INPUT_DIMS, c["x"], c["y"], hp,
                                      train_idx, test_idx)
    close_helper(agent)
    agent.ModalFusion.ucsma41_b0_bypass = True
    if not (hasattr(agent.ModalFusion, "MLP") and
            hasattr(agent.ModalFusion, "prompt_text_features") and
            hasattr(agent.ModalFusion, "avg_pool")):
        raise AssertionError("PEN construction/CLIP cache side effects were removed")
    if not np.array_equal(agent.val_idx, agent.tst_idx):
        raise AssertionError("Historical val=test protocol changed")
    return agent


def train_step(agent, c):
    torch, utils = c["torch"], c["utils"]
    agent.MessagePassing0.train()
    agent.ModalFusion.train()
    agent.optimizer_MP0.zero_grad()
    agent.optimizer_MF.zero_grad()
    model_num, parts = agent.dataLayer(agent.feat)
    fused, adjs, a_loss = agent.MessagePassing0(parts, model_num,
                                                 agent.targ, agent.trn_idx)
    graph_loss = 0.0
    for i in range(model_num):
        graph_loss = graph_loss + utils.GraphConstructLoss(
            parts[i], adjs[i], agent.hyperpm.theta_smooth,
            agent.hyperpm.theta_degree, agent.hyperpm.theta_sparsity)
    graph_loss = graph_loss / 4.0
    logp, _, _ = agent.ModalFusion(fused)
    cls_loss = utils.ClsLoss(logp, agent.targ, agent.trn_idx, agent.weight)
    loss = cls_loss + graph_loss
    loss.backward()
    registered = [p for module in (agent.MessagePassing0, agent.ModalFusion)
                  for p in module.parameters() if p.requires_grad]
    effective = sum(p.numel() for p in registered
                    if p.grad is not None and torch.isfinite(p.grad).all() and
                    float(p.grad.norm()) > 0.0)
    agent.optimizer_MP0.step()
    agent.optimizer_MF.step()
    return {"loss": float(loss.detach()), "cls_loss": float(cls_loss.detach()),
            "graph_loss": float(graph_loss.detach()),
            "registered_params": sum(p.numel() for p in registered),
            "effective_gradient_params": effective,
            "output": logp.detach().clone()}


def evaluate_base(agent, c, detail=False):
    torch = c["torch"]
    agent.MessagePassing0.eval()
    agent.ModalFusion.eval()
    captured, hooks = {}, []
    if detail:
        def cam_hook(_module, _inputs, output):
            captured["cam"] = output.detach().clone()
        def head_hook(_module, inputs):
            captured["head_cam"] = inputs[0].detach().clone()
            captured["attention"] = inputs[1].detach().clone()
        hooks.append(agent.ModalFusion.FeedForward[-1].register_forward_hook(cam_hook))
        hooks.append(agent.ModalFusion.Outputlayer.register_forward_pre_hook(head_hook))
    with torch.no_grad():
        model_num, parts = agent.dataLayer(agent.feat)
        fused, adjs, a_loss = agent.MessagePassing0(parts, model_num,
                                                    agent.targ, agent.trn_idx)
        logp, hidden, attn = agent.ModalFusion(fused)
        graph_loss = 0.0
        if detail:
            for i in range(model_num):
                graph_loss = graph_loss + c["utils"].GraphConstructLoss(
                    parts[i], adjs[i], agent.hyperpm.theta_smooth,
                    agent.hyperpm.theta_degree, agent.hyperpm.theta_sparsity)
            graph_loss = graph_loss / 4.0
    for hook in hooks:
        hook.remove()
    if logp.shape != (871, 2) or not torch.isfinite(logp).all():
        raise FloatingPointError("Non-finite or mis-shaped baseline output")
    result = {"logp": logp.detach().clone()}
    if detail:
        result.update(cam=captured["cam"], head_cam=captured["head_cam"],
                      attention=captured["attention"], gfr=fused.detach().clone(),
                      adjs=[v.detach().clone() for v in adjs],
                      graph_loss=float(graph_loss.detach()),
                      alignment_loss=float(a_loss.detach()))
    return result


def metrics_row(output, y, test):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    pred = np.asarray(output)[test].argmax(axis=1)
    target = y[test]
    tp, fn, fp, tn = map(int, confusion_matrix(target, pred, labels=[0, 1]).ravel())
    auc = float(roc_auc_score(np.eye(2)[target], np.eye(2)[pred]))
    return dict(acc=float(np.mean(pred == target)), hard_auc=auc,
                sen_asd=float(tp / (tp + fn)), spe_td=float(tn / (tn + fp)),
                asd_tp=tp, asd_fn=fn, td_fp=fp, td_tn=tn)


def model_state(agent):
    return dict(gfr=copy.deepcopy(agent.MessagePassing0.state_dict()),
                pln=copy.deepcopy(agent.ModalFusion.state_dict()))


def load_model_state(agent, state):
    agent.MessagePassing0.load_state_dict(state["gfr"])
    agent.ModalFusion.load_state_dict(state["pln"])


def make_digest(agent):
    return parameter_hash(agent)


def historical_rows():
    with (HIST32 / "results/fold_metrics.csv").open(newline="", encoding="utf-8") as stream:
        return [r for r in csv.DictReader(stream)
                if r.get("config") == "B0" and r.get("seed") == "0"]


def verify_e0(rows):
    old = sorted(historical_rows(), key=lambda r: int(r["fold"]))
    current = sorted([r for r in rows if r["config"] == "E0_B0A"],
                     key=lambda r: int(r["fold"]))
    if len(old) != 10 or len(current) != 10:
        return {"pass": False, "reason": "requires 10 completed folds"}
    fields = {"acc": "acc", "auc": "hard_auc", "sen_asd": "sen_asd",
              "spe_td": "spe_td", "best_epoch": "best_epoch",
              "epochs_executed": "epochs_executed",
              "initial_trunk_sha256": "initial_trunk_sha256",
              "initial_rng_sha256": "initial_rng_sha256"}
    per_fold = []
    for prev, cur in zip(old, current):
        checks = {}
        for old_key, new_key in fields.items():
            if old_key.endswith("sha256") or old_key == "best_epoch" or old_key == "epochs_executed":
                checks[old_key] = str(prev[old_key]) == str(cur[new_key])
            else:
                checks[old_key] = abs(float(prev[old_key]) - float(cur[new_key])) <= 1e-12
        per_fold.append({"fold": int(cur["fold"]), "checks": checks,
                         "all_match": all(checks.values())})
    return {"pass": all(v["all_match"] for v in per_fold),
            "matched_folds": sum(v["all_match"] for v in per_fold),
            "fields": fields, "folds": per_fold}


def save_anchor(agent, c, fold, base_detail, row, rng_state):
    path = RUNTIME / "anchors" / f"anchor_fold{fold:02d}.pt"
    train_idx, test_idx = c["folds"][fold - 1]
    payload = dict(fold=fold, model=model_state(agent),
                   cam=base_detail["cam"].detach().cpu(),
                   attention=base_detail["attention"].detach().cpu(),
                   gfr=base_detail["gfr"].detach().cpu(),
                   graph_loss=base_detail["graph_loss"],
                   train_idx=train_idx.astype(np.int64),
                   test_idx=test_idx.astype(np.int64), y=c["y"].astype(np.int64),
                   base_row=row, continuation_rng=rng_state,
                   source_signature=sha_bytes(json.dumps(source_hashes(), sort_keys=True).encode()))
    torch_write(path, payload)
    return path


def run_e0_worker(gpu=0):
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("E0 worker GPU assignment mismatch")
    c = context(gpu)
    seed_everything(torch, 0)
    if not (Path.home() / ".cache/clip/RN50.pt").is_file():
        raise FileNotFoundError("Cached RN50 weights required; training will not download models")
    state_path = RUNTIME / "tasks/E0_B0A.pt"
    lock_path = RUNTIME / "tasks/E0_B0A.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = torch_load(state_path) if state_path.exists() else None
        if saved and saved.get("phase") == "COMPLETE":
            print("E0 already complete")
            return
        if saved:
            c["network"].get_prompt_text_features()
            rng_restore(torch, saved["rng"])
        started = saved.get("started_utc", utc()) if saved else utc()
        rows = list(saved.get("rows", [])) if saved else []
        fold = int(saved.get("fold", len(rows) + 1)) if saved else 1
        phase = saved.get("phase") if saved else None
        previous_seconds = float(saved.get("seconds", 0.0)) if saved else 0.0
        log_path = RUNTIME / "logs/E0_B0A.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", buffering=1) as log, \
                contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            while fold <= 10:
                train_idx, test_idx = c["folds"][fold - 1]
                entry_rng = rng_capture(torch)
                agent = make_agent(c, fold)
                if phase == "EPOCH" and saved and int(saved["fold"]) == fold:
                    load_model_state(agent, saved["model"])
                    agent.optimizer_MP0.load_state_dict(saved["optimizer_mp"])
                    agent.optimizer_MF.load_state_dict(saved["optimizer_mf"])
                    rng_restore(torch, saved["rng"])
                    best_state = saved["best_state"]
                    best_acc = float(saved["best_acc"])
                    best_epoch = int(saved["best_epoch"])
                    wait = int(saved["wait"])
                    start_epoch = int(saved["epoch"]) + 1
                    fold_seconds_before = float(saved.get("fold_seconds", 0.0))
                    trunk_sha = saved["initial_trunk_sha256"]
                    initial_rng_sha = saved["initial_rng_sha256"]
                    last_step = saved.get("last_step_stats")
                    resume_stopped = wait > PROTOCOL["early"]
                else:
                    best_state, best_acc, best_epoch, wait = None, 0.0, -1, 0
                    start_epoch, fold_seconds_before = 0, 0.0
                    trunk_sha, initial_rng_sha = make_digest(agent), rng_hash(entry_rng)
                    last_step = None
                    resume_stopped = False
                phase = None
                fold_started = time.monotonic()
                last_epoch = start_epoch - 1
                epoch_range = () if resume_stopped else range(start_epoch, PROTOCOL["nepoch"])
                for epoch in epoch_range:
                    last_step = train_step(agent, c)
                    output = evaluate_base(agent, c)["logp"]
                    acc = float(output[agent.val_idx].argmax(1).eq(
                        agent.targ[agent.val_idx]).double().mean().item())
                    if acc > best_acc:
                        best_acc, best_epoch, wait = acc, epoch, 0
                        best_state = model_state(agent)
                    else:
                        wait += 1
                    last_epoch = epoch
                    stop = wait > PROTOCOL["early"]
                    if (epoch + 1) % CHECKPOINT_EVERY == 0 or stop:
                        torch_write(state_path, dict(
                            phase="EPOCH", fold=fold, epoch=epoch,
                            model=model_state(agent),
                            optimizer_mp=agent.optimizer_MP0.state_dict(),
                            optimizer_mf=agent.optimizer_MF.state_dict(),
                            rng=rng_capture(torch), best_state=best_state,
                            best_acc=best_acc, best_epoch=best_epoch, wait=wait,
                            last_step_stats=last_step,
                            fold_seconds=fold_seconds_before + time.monotonic() - fold_started,
                            initial_trunk_sha256=trunk_sha,
                            initial_rng_sha256=initial_rng_sha,
                            rows=rows, seconds=previous_seconds + time.monotonic() - fold_started,
                            started_utc=started))
                    if stop:
                        break
                if best_state is None:
                    raise AssertionError("E0 checkpoint selection produced no checkpoint")
                if last_step is None:
                    last_step = dict(registered_params=sum(p.numel() for module in
                        (agent.MessagePassing0, agent.ModalFusion)
                        for p in module.parameters() if p.requires_grad),
                        effective_gradient_params=0)
                continuation_rng = rng_capture(torch)
                load_model_state(agent, best_state)
                detail = evaluate_base(agent, c, detail=True)
                metrics = metrics_row(detail["logp"], c["y"], test_idx)
                if abs(metrics["acc"] - best_acc) > 1e-12:
                    raise AssertionError("E0 selected checkpoint ACC mismatch")
                row = dict(config="E0_B0A", seed=0, fold=fold, physical_gpu=gpu,
                           **metrics, best_epoch=best_epoch,
                           epochs_executed=last_epoch + 1,
                           initial_trunk_sha256=trunk_sha,
                           initial_rng_sha256=initial_rng_sha,
                           wall_seconds=fold_seconds_before + time.monotonic() - fold_started,
                           train_count=len(train_idx), test_count=len(test_idx),
                           trainable_params=int(last_step["registered_params"]),
                           effective_gradient_params=int(last_step["effective_gradient_params"]),
                           trained_from_scratch=True)
                rows.append(row)
                save_anchor(agent, c, fold, detail, row, continuation_rng)
                close_helper(agent)
                del agent, detail, output
                torch.cuda.empty_cache()
                previous_seconds += time.monotonic() - fold_started
                fold += 1
                torch_write(state_path, dict(phase="BOUNDARY", fold=fold, rng=continuation_rng,
                                             rows=rows, seconds=previous_seconds,
                                             started_utc=started))
                print(f"E0_FOLD_DONE fold={fold-1} acc={metrics['acc']:.6f}", flush=True)
            torch_write(state_path, dict(phase="COMPLETE", rows=rows,
                                         seconds=previous_seconds, started_utc=started,
                                         completed_utc=utc()))
    verification = verify_e0(rows)
    def finish_e0(manifest):
        manifest["e0_history_verification"] = verification
        manifest["workers"]["E0_B0A"] = dict(status="COMPLETE", gpu=0,
                                             folds_completed=10, seconds=previous_seconds,
                                             completed_utc=utc())
        manifest["completed_folds"] = sum(int(v.get("folds_completed", 0))
                                          for v in manifest["workers"].values())
    update_manifest(finish_e0)
    if not verification["pass"]:
        raise RuntimeError("Independent E0 replay differs from Experiment32 B0-A; inspect parity before continuing")


def load_structures():
    o = json.loads((EXP / "ontology.json").read_text(encoding="utf-8"))
    endpoints = np.asarray(o["roi_endpoints_by_fc_feature"], dtype=np.int16)
    fam_roi = np.asarray(o["region_family_masks"], dtype=bool)
    fam_edge = np.asarray(o["region_family_edge_masks"], dtype=bool)
    real_roi = np.asarray(o["real_concept_roi_masks"], dtype=bool)
    real_edge = np.asarray(o["real_concept_edge_masks"], dtype=bool)
    wrong_roi = np.asarray(o["wrong_semantics_control"]["roi_masks"], dtype=bool)
    wrong_edge = np.asarray(o["wrong_semantics_control"]["edge_masks"], dtype=bool)
    return endpoints, fam_roi, fam_edge, real_roi, real_edge, wrong_roi, wrong_edge, o


def make_numeric_prior(x, train_idx, endpoints):
    from sklearn.cluster import KMeans
    known_ids = np.flatnonzero((endpoints >= 0).all(axis=1))
    data = x[train_idx][:, 64 + known_ids].astype(np.float64)
    corr = np.corrcoef(data, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    if corr.shape != (len(known_ids), len(known_ids)):
        raise AssertionError("Unsupervised edge covariance shape mismatch")
    labels = KMeans(n_clusters=4, n_init=20, random_state=41041,
                    algorithm="lloyd").fit_predict(corr)
    masks = np.zeros((7, 256), dtype=bool)
    roi_masks = np.zeros((7, 111), dtype=bool)
    for k in range(4):
        masks[k, known_ids[labels == k]] = True
        for edge in known_ids[labels == k]:
            roi_masks[k, endpoints[edge]] = True
    return roi_masks, masks, {"cluster_sizes": [int((labels == k).sum()) for k in range(4)],
                              "cluster_ids_by_fc_feature": dict(zip(known_ids.tolist(), labels.tolist()))}


def evidence_cache_path(fold, mode):
    return RUNTIME / "evidence" / f"fold{fold:02d}_{mode}.npz"


def write_evidence_cache(x, y, folds, structures):
    endpoints, fam_roi, fam_edge, real_roi, real_edge = structures[:5]
    cache_meta = {}
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        for mode in ("global", "conditional"):
            path = evidence_cache_path(fold, mode)
            if path.exists():
                cached = np.load(path, allow_pickle=False)
                if (str(cached["input_sha256"].item()) == sha_file(ROOT / DATA_FILES[0])
                        and "algorithm_sha256" in cached.files
                        and str(cached["algorithm_sha256"].item()) == sha_file(EXP / "ucsma.py")):
                    cache_meta[path.name] = sha_file(path)
                    cached.close()
                    continue
                cached.close()
            result = fold_normative_features(
                x, train_idx, y, endpoints, real_roi, real_edge,
                fam_roi, fam_edge, mode=mode, alpha=10.0)
            arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
            arrays["stats_json"] = np.asarray(json.dumps({
                k: v for k, v in result.items() if not isinstance(v, np.ndarray)},
                allow_nan=False))
            arrays["input_sha256"] = np.asarray(sha_file(ROOT / DATA_FILES[0]))
            arrays["algorithm_sha256"] = np.asarray(sha_file(EXP / "ucsma.py"))
            arrays["train_idx"] = train_idx.astype(np.int64)
            arrays["test_idx"] = test_idx.astype(np.int64)
            arrays["mode"] = np.asarray(mode)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + ".tmp.npz")
            np.savez_compressed(temp, **arrays)
            os.replace(temp, path)
            cache_meta[path.name] = sha_file(path)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["evidence_cache_sha256"] = cache_meta
    manifest["evidence_cache_complete"] = len(cache_meta) == 20
    json_write(MANIFEST_PATH, manifest)


def load_evidence(fold, mode):
    with np.load(evidence_cache_path(fold, mode), allow_pickle=False) as values:
        result = {k: values[k] for k in values.files if k not in
                  {"stats_json", "input_sha256", "algorithm_sha256", "train_idx", "test_idx", "mode"}}
        result["stats"] = json.loads(str(values["stats_json"].item()))
        result["train_idx"] = values["train_idx"]
        result["test_idx"] = values["test_idx"]
        return result


def feature_vector(group, cam, raw_corr, evidence, masks, data_prior=None,
                   reject_override: bool | None = None):
    n = len(cam)
    fields = [cam.reshape(n, -1).astype(np.float32),
              np.zeros((n, 256), dtype=np.float32),
              np.zeros((n, 256, 6), dtype=np.float32),
              np.zeros((n, 111, 6), dtype=np.float32),
              np.zeros((n, 8, 6), dtype=np.float32),
              np.zeros((n, 7, 6), dtype=np.float32),
              np.zeros((n, 8), dtype=np.float32)]
    if group == "E1_RAW_CAPACITY":
        raw = np.asarray(raw_corr, dtype=np.float32)
        rng = np.random.default_rng(4101)

        def fixed_projection(width):
            matrix = rng.normal(0.0, 1.0 / np.sqrt(256.0), size=(256, width))
            return raw @ matrix.astype(np.float32)

        fields[1] = raw
        fields[2] = np.repeat(raw[:, :, None], 6, axis=2)
        fields[3] = fixed_projection(111).reshape(n, 111, 1).repeat(6, axis=2)
        fields[4] = fixed_projection(8).reshape(n, 8, 1).repeat(6, axis=2)
        fields[5] = fixed_projection(7).reshape(n, 7, 1).repeat(6, axis=2)
        fields[6] = np.column_stack([
            raw.mean(1), raw.std(1), np.quantile(raw, .1, axis=1),
            np.quantile(raw, .25, axis=1), np.median(raw, axis=1),
            np.quantile(raw, .75, axis=1), np.quantile(raw, .9, axis=1),
            np.mean(np.abs(raw), axis=1)])
    if group in ("E2_GLOBAL_NORM", "E3_COND_NORM", "E4_ROI_NORM",
                 "E5_MULTISCALE_NUM", "E6_REAL_NO_REJECT", "E7_WRONG_SEMANTICS",
                 "E8_DATA_PRIOR", "E9_UCSMA_FULL"):
        fields[2] = evidence["edge_features"].astype(np.float32)
    if group in ("E4_ROI_NORM", "E5_MULTISCALE_NUM", "E6_REAL_NO_REJECT",
                 "E7_WRONG_SEMANTICS", "E8_DATA_PRIOR", "E9_UCSMA_FULL"):
        fields[3] = evidence["roi_features"].astype(np.float32)
    if group in ("E5_MULTISCALE_NUM", "E6_REAL_NO_REJECT", "E7_WRONG_SEMANTICS",
                 "E8_DATA_PRIOR", "E9_UCSMA_FULL"):
        fields[4] = evidence["family_features"].astype(np.float32)
    diag = dict(concept_delta=0.0, matched_mean=0.0, reject_mean=0.0,
                eligible_concepts=int(ELIGIBLE.sum()), wrong_edges=None,
                numeric_prior=None)
    if group in ("E6_REAL_NO_REJECT", "E7_WRONG_SEMANTICS",
                 "E8_DATA_PRIOR", "E9_UCSMA_FULL"):
        _, _, _, real_roi, real_edge, wrong_roi, wrong_edge, ontology = masks
        if group == "E7_WRONG_SEMANTICS":
            use_roi, use_edge = wrong_roi, wrong_edge
        elif group == "E8_DATA_PRIOR":
            use_roi, use_edge, prior_diag = data_prior
            diag["numeric_prior"] = prior_diag
        else:
            use_roi, use_edge = real_roi, real_edge
        known = evidence["known_edge_mask"].astype(bool)
        edge_support = evidence["alignment_edge_support"].astype(np.float64)
        concept_features = summarize_concepts(
            evidence["edge_features"], edge_support, evidence["roi_features"],
            use_roi, use_edge, known).astype(np.float32)
        eligible = np.asarray([bool(use_edge[k].any()) for k in range(7)], dtype=bool)
        thresholds = np.zeros(7, dtype=np.float64)
        train_idx = evidence["train_idx"]
        # Calibration uses only the outer-train TD reference subjects.
        # Labels are read from the train partition in the caller and never from test.
        y = evidence["y"]
        td_train = train_idx[y[train_idx] == 1]
        for k in range(7):
            magnitude = 1.0 - np.exp(-0.25 * np.clip(
                concept_features[td_train, k, 0] + concept_features[td_train, k, 1],
                0.0, 8.0))
            signal = .55 * magnitude + .30 * np.maximum(
                concept_features[td_train, k, 2], concept_features[td_train, k, 3])
            if len(signal):
                thresholds[k] = max(float(np.quantile(signal, CONCEPT_TAIL_QUANTILE)), .02)
        reject = (group in ("E7_WRONG_SEMANTICS", "E8_DATA_PRIOR",
                            "E9_UCSMA_FULL") if reject_override is None
                  else reject_override)
        direction_weight = np.asarray([p["directional_prior_weight_negative"]
            for p in json.loads((EXP / "prompts.json").read_text(encoding="utf-8"))["concepts"]])
        if group == "E8_DATA_PRIOR":
            direction_weight = np.full(7, 0.5, dtype=np.float64)
        matched, unmatched = semantic_mass(concept_features, eligible, reject,
                                           thresholds, direction_weight,
                                           match_scale=SEMANTIC_MATCH_SCALE)
        # Unmatched evidence remains in numeric fields, while disease-semantic
        # evidence must be gated by its actual partial matching mass.
        fields[5] = concept_features * matched[:, :, None]
        fields[6] = np.column_stack([matched, unmatched]).astype(np.float32)
        diag.update(concept_delta=float(np.mean(np.abs(concept_features -
                                                       evidence["concept_features"]))),
                    matched_mean=float(matched.sum(1).mean()),
                    reject_mean=float(unmatched.mean()),
                    eligible_concepts=int(eligible.sum()),
                    wrong_edges=[int(np.count_nonzero(v)) for v in use_edge[:6]])
    vector = np.concatenate([v.reshape(n, -1) for v in fields], axis=1)
    if vector.shape != (n, FEATURE_DIM) or not np.isfinite(vector).all():
        raise AssertionError(f"Invalid branch feature tensor: {vector.shape}")
    return vector.astype(np.float32), diag


def rng_subseed(fold):
    # The registered model seed remains 0; this fork makes adapter initialization
    # exactly fold-matched across arms without consuming the continuous worker RNG.
    return 410000 + int(fold)


def branch_step(branch, output_layer, optimizer, features, cam, attention,
                agent, c, train_idx, graph_loss):
    torch, utils = c["torch"], c["utils"]
    branch.train()
    output_layer.eval()
    optimizer.zero_grad(set_to_none=True)
    augmented, residual = branch(features, cam)
    logp, hidden = output_layer(augmented.reshape(871, -1), attention)
    cls_loss = utils.ClsLoss(logp, agent.targ, train_idx, agent.weight)
    loss = cls_loss + float(graph_loss)
    loss.backward()
    grad_nonzero = sum(int(p.grad is not None and torch.isfinite(p.grad).all() and
                           float(p.grad.norm()) > 0.0) for p in branch.parameters())
    effective_params = sum(int(torch.count_nonzero(p.grad).item())
                           for p in branch.parameters()
                           if p.grad is not None and torch.isfinite(p.grad).all())
    optimizer.step()
    return dict(loss=float(loss.detach()), cls_loss=float(cls_loss.detach()),
                residual=residual.detach().clone(), output=logp.detach().clone(),
                grad_nonzero=grad_nonzero,
                effective_gradient_params=effective_params)


def evaluate_branch(branch, output_layer, features, cam, attention):
    branch.eval()
    output_layer.eval()
    with __import__("torch").no_grad():
        augmented, residual = branch(features, cam)
        logp, _ = output_layer(augmented.reshape(871, -1), attention)
    if not __import__("torch").isfinite(logp).all():
        raise FloatingPointError("Non-finite branch prediction")
    return logp.detach().clone(), residual.detach().clone()


def branch_state(branch):
    return copy.deepcopy(branch.state_dict())


def derangement(idx: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    if len(idx) <= 1:
        return idx.copy()
    for _ in range(100):
        order = rng.permutation(idx)
        shifted = np.roll(order, 1)
        mapping = dict(zip(order.tolist(), shifted.tolist()))
        candidate = np.asarray([mapping[int(v)] for v in idx], dtype=np.int64)
        if np.all(candidate != idx):
            return candidate
    return np.roll(idx, 1)


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    if len(values) == 0 or np.var(values) <= 1e-15:
        return 0.0
    grand = values.mean()
    between = 0.0
    for key in np.unique(groups):
        sample = values[groups == key]
        if len(sample):
            between += len(sample) * float((sample.mean() - grand) ** 2)
    return float(np.clip(between / (len(values) * values.var()), 0.0, 1.0))


def spearman_safe(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.stats import spearmanr
    left, right = np.asarray(left), np.asarray(right)
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def task_path(group):
    return RUNTIME / "tasks" / f"{group}.pt"


def run_branch_worker(group: str, gpu: int):
    import torch
    if group == "E0_B0A" or group not in CONFIGS:
        raise ValueError(group)
    c = context(gpu)
    manifest_check = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest_check["source_and_input"].get("experiment_code_sha256") != code_hashes():
        raise RuntimeError("Experiment code changed after design freeze")
    if manifest_check["source_and_input"].get("core_sha256") != source_hashes():
        raise RuntimeError("Root core source changed after design freeze")
    state_path = task_path(group)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = RUNTIME / "tasks" / f"{group}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = torch_load(state_path) if state_path.exists() else None
        if saved and saved.get("phase") == "COMPLETE":
            print(f"{group} already complete")
            return
        seed_everything(torch, 0)
        if saved:
            c["network"].get_prompt_text_features()
            rng_restore(torch, saved["rng"])
        start_utc = saved.get("started_utc", utc()) if saved else utc()
        rows = list(saved.get("rows", [])) if saved else []
        diagnostics = list(saved.get("diagnostics", [])) if saved else []
        fold = int(saved.get("fold", len(rows) + 1)) if saved else 1
        phase = saved.get("phase") if saved else None
        previous = float(saved.get("seconds", 0.0)) if saved else 0.0
        log_path = RUNTIME / "logs" / f"{group}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", buffering=1) as log, \
                contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            while fold <= 10:
                train_idx, test_idx = c["folds"][fold - 1]
                anchor_path = RUNTIME / "anchors" / f"anchor_fold{fold:02d}.pt"
                anchor = torch_load(anchor_path)
                entry_rng = rng_capture(torch)
                agent = make_agent(c, fold)
                load_model_state(agent, anchor["model"])
                for parameter in list(agent.MessagePassing0.parameters()) + list(agent.ModalFusion.parameters()):
                    parameter.requires_grad_(False)
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(rng_subseed(fold))
                    torch.cuda.manual_seed_all(rng_subseed(fold))
                    branch = UCSMAResidual(FEATURE_DIM, 64, 144).to(agent.dev)
                optimizer = torch.optim.Adam(branch.parameters(), lr=PROTOCOL["lr"],
                                             weight_decay=0.0)
                evidence_mode = "global" if group == "E2_GLOBAL_NORM" else "conditional"
                evidence = load_evidence(fold, evidence_mode)
                evidence["y"] = c["y"]
                endpoints, family_roi, family_edge, real_roi, real_edge, wrong_roi, wrong_edge, ontology = load_structures()
                structures = (endpoints, family_roi, family_edge, real_roi, real_edge,
                              wrong_roi, wrong_edge, ontology)
                numeric_prior = None
                if group == "E8_DATA_PRIOR":
                    nr, ne, ndiag = make_numeric_prior(c["x"], train_idx, endpoints)
                    numeric_prior = (nr, ne, ndiag)
                vector_np, group_diag = feature_vector(
                    group, anchor["cam"].numpy(), c["x"][:, 64:], evidence,
                    structures, data_prior=numeric_prior)
                features = torch.as_tensor(vector_np, device=agent.dev)
                cam = anchor["cam"].to(agent.dev)
                attention = anchor["attention"].to(agent.dev)
                base = anchor["base_row"]
                best_acc, best_epoch, wait, best_state = 0.0, -1, 0, None
                fold_seconds_before = 0.0
                start_epoch = 0
                init_sha = sha_bytes(b"".join(p.detach().cpu().numpy().tobytes()
                                               for p in branch.parameters()))
                init_rng_sha = rng_hash(entry_rng)
                last_effective_params = 0
                resume_stopped = False
                if phase == "EPOCH" and saved and int(saved["fold"]) == fold:
                    branch.load_state_dict(saved["branch"])
                    optimizer.load_state_dict(saved["optimizer"])
                    best_acc, best_epoch, wait = (float(saved["best_acc"]),
                                                  int(saved["best_epoch"]),
                                                  int(saved["wait"]))
                    best_state = saved["best_state"]
                    start_epoch = int(saved["epoch"]) + 1
                    fold_seconds_before = float(saved.get("fold_seconds", 0.0))
                    init_sha = saved["initial_branch_sha256"]
                    init_rng_sha = saved["initial_rng_sha256"]
                    last_effective_params = int(saved.get("last_effective_params", 0))
                    resume_stopped = wait > PROTOCOL["early"]
                    rng_restore(torch, saved["rng"])
                phase = None
                fold_started = time.monotonic()
                last_epoch = start_epoch - 1
                grad_nonzero = 0
                epoch_range = () if resume_stopped else range(start_epoch, PROTOCOL["nepoch"])
                for epoch in epoch_range:
                    stepped = branch_step(branch, agent.ModalFusion.Outputlayer,
                                          optimizer, features, cam, attention,
                                          agent, c, train_idx,
                                          anchor["graph_loss"])
                    logp, residual = evaluate_branch(branch,
                        agent.ModalFusion.Outputlayer, features, cam, attention)
                    pred = logp[test_idx].argmax(1)
                    val_acc = float(pred.eq(agent.targ[test_idx]).double().mean().item())
                    if val_acc > best_acc:
                        best_acc, best_epoch, wait = val_acc, epoch, 0
                        best_state = branch_state(branch)
                    else:
                        wait += 1
                    grad_nonzero = max(grad_nonzero, stepped["grad_nonzero"])
                    last_effective_params = stepped["effective_gradient_params"]
                    last_epoch = epoch
                    stop = wait > PROTOCOL["early"]
                    if (epoch + 1) % CHECKPOINT_EVERY == 0 or stop:
                        torch_write(state_path, dict(
                            phase="EPOCH", config=group, fold=fold, epoch=epoch,
                            branch=branch_state(branch), optimizer=optimizer.state_dict(),
                            rng=rng_capture(torch), best_state=best_state,
                            best_acc=best_acc, best_epoch=best_epoch, wait=wait,
                            last_effective_params=last_effective_params,
                            fold_seconds=fold_seconds_before + time.monotonic() - fold_started,
                            initial_branch_sha256=init_sha,
                            initial_rng_sha256=init_rng_sha,
                            rows=rows, diagnostics=diagnostics,
                            seconds=previous + time.monotonic() - fold_started,
                            started_utc=start_utc))
                    if stop:
                        break
                if best_state is None:
                    raise AssertionError("Auxiliary checkpoint selection produced no state")
                continuation_rng = rng_capture(torch)
                branch.load_state_dict(best_state)
                logp, residual = evaluate_branch(branch,
                    agent.ModalFusion.Outputlayer, features, cam, attention)
                metric = metrics_row(logp, c["y"], test_idx)
                if abs(metric["acc"] - best_acc) > 1e-12:
                    raise AssertionError("Auxiliary best checkpoint ACC mismatch")
                ratio = float((residual[test_idx].norm(dim=1) /
                               cam[test_idx].reshape(len(test_idx), -1).norm(dim=1).clamp_min(1e-8)).max().item())
                if ratio > 0.050001:
                    raise AssertionError("Smooth residual bound exceeded 5%")
                trainable = sum(p.numel() for p in branch.parameters() if p.requires_grad)
                effective = last_effective_params
                row = dict(config=group, seed=0, fold=fold, physical_gpu=gpu,
                           **metric, best_epoch=best_epoch,
                           epochs_executed=last_epoch + 1,
                           initial_branch_sha256=init_sha,
                           initial_rng_sha256=init_rng_sha,
                           wall_seconds=fold_seconds_before + time.monotonic() - fold_started,
                           train_count=len(train_idx), test_count=len(test_idx),
                           trainable_params=trainable, effective_gradient_params=effective,
                           residual_ratio_mean=float((residual[test_idx].norm(dim=1) /
                               cam[test_idx].reshape(len(test_idx), -1).norm(dim=1).clamp_min(1e-8)).mean().item()),
                           residual_ratio_max=ratio, mean_reference_support=float(evidence["reference_support"][test_idx].mean()),
                           mean_qc_support=float(evidence["qc_support"][test_idx].mean()),
                           e0_anchor_acc=float(base["acc"]))
                alignment = vector_np[:, -8:].astype(np.float64)
                test_alignment = alignment[test_idx]
                test_labels = c["y"][test_idx]
                phenotype = c["x"][:, :48]
                age_code = np.argmax(phenotype[:, :26], axis=1)
                site_code = np.argmax(phenotype[:, 26:46], axis=1)
                sex_code = np.argmax(phenotype[:, 46:48], axis=1)
                raw_family_signal = evidence["concept_features"]
                alignment_diag = {
                    "test_alignment_patient_sd": float(test_alignment.std(0).mean()),
                    "test_alignment_entropy_mean": float(-np.sum(
                        np.clip(test_alignment, 1e-12, 1.0) *
                        np.log(np.clip(test_alignment, 1e-12, 1.0)), axis=1).mean()),
                    "test_unmatched_mean": float(test_alignment[:, 7].mean()),
                    "test_unmatched_asd_mean": float(test_alignment[test_labels == 0, 7].mean()),
                    "test_unmatched_td_mean": float(test_alignment[test_labels == 1, 7].mean()),
                    "alignment_site_eta2": float(np.mean([
                        eta_squared(test_alignment[:, k], site_code[test_idx])
                        for k in range(7)])),
                    "alignment_agebin_eta2": float(np.mean([
                        eta_squared(test_alignment[:, k], age_code[test_idx])
                        for k in range(7)])),
                    "alignment_sex_eta2": float(np.mean([
                        eta_squared(test_alignment[:, k], sex_code[test_idx])
                        for k in range(7)])),
                    "alignment_qc_spearman": float(np.mean([
                        spearman_safe(test_alignment[:, k],
                                      evidence["qc_support"][test_idx])
                        for k in range(7)])),
                    "alignment_mean_asd": float(test_alignment[test_labels == 0, :7].mean()),
                    "alignment_mean_td": float(test_alignment[test_labels == 1, :7].mean()),
                }
                for k, concept_id in enumerate(CONCEPT_IDS):
                    alignment_diag[f"test_{concept_id}_mean"] = float(test_alignment[:, k].mean())
                    alignment_diag[f"test_{concept_id}_asd_mean"] = float(
                        test_alignment[test_labels == 0, k].mean())
                    alignment_diag[f"test_{concept_id}_td_mean"] = float(
                        test_alignment[test_labels == 1, k].mean())
                    if k < raw_family_signal.shape[1]:
                        signal = .55 * (1.0 - np.exp(-0.25 * np.clip(
                            raw_family_signal[:, k, 0] + raw_family_signal[:, k, 1],
                            0.0, 8.0))) + .30 * np.maximum(
                            raw_family_signal[:, k, 2], raw_family_signal[:, k, 3])
                        alignment_diag[f"test_{concept_id}_mass_signal_spearman"] = \
                            spearman_safe(test_alignment[:, k], signal[test_idx])
                diag = dict(config=group, fold=fold, **group_diag,
                            **alignment_diag,
                            normative_mode=evidence_mode,
                            td_oof_abs_z_q95=float(evidence["stats"]["td_oof_abs_z_q95"]),
                            td_oof_abs_z_median=float(evidence["stats"]["td_oof_abs_z_median"]),
                            z_clip_fraction=float(evidence["stats"]["clip_fraction"]),
                            max_ridge_leverage=float(evidence["stats"]["max_leverage"]),
                            train_reference_support=float(evidence["stats"]["conditional_support_mean"]),
                            test_reference_support=float(evidence["reference_support"][test_idx].mean()),
                            train_qc_support=float(evidence["stats"]["qc_support_mean"]),
                            test_qc_support=float(evidence["qc_support"][test_idx].mean()),
                            roi_degree_min=int(evidence["roi_degree"][evidence["roi_degree"] > 0].min()),
                            roi_degree_max=int(evidence["roi_degree"].max()),
                            gradient_params_nonzero=effective,
                            registered_trainable_params=trainable,
                            optimizer_groups=len(optimizer.param_groups),
                            selected_epoch=best_epoch)
                rows.append(row)
                diagnostics.append(diag)
                torch_write(RUNTIME / "checkpoints" / f"{group}_fold{fold:02d}_best.pt",
                            dict(branch=best_state, fold=fold, config=group,
                                 best_epoch=best_epoch, metrics=metric,
                                 vector_sha256=sha_bytes(vector_np.tobytes()),
                                 branch_sha256=sha_bytes(b"".join(v.detach().cpu().numpy().tobytes()
                                     for _, v in sorted(best_state.items())))))
                close_helper(agent)
                del branch, optimizer, agent, anchor, evidence, features, cam, attention
                torch.cuda.empty_cache()
                previous += time.monotonic() - fold_started
                fold += 1
                torch_write(state_path, dict(phase="BOUNDARY", config=group,
                                             fold=fold, rng=continuation_rng,
                                             rows=rows, diagnostics=diagnostics,
                                             seconds=previous, started_utc=start_utc))
                def record_fold(manifest):
                    manifest["workers"][group] = dict(
                        status="RUNNING", gpu=gpu, folds_completed=len(rows),
                        completed_fold_ids=[int(r["fold"]) for r in rows],
                        seconds=previous, updated_utc=utc())
                    manifest["fold_status"][group] = ["COMPLETE"] * len(rows) + \
                        ["PENDING"] * (10 - len(rows))
                    manifest["completed_folds"] = sum(
                        int(v.get("folds_completed", 0))
                        for v in manifest["workers"].values())
                update_manifest(record_fold)
                print(f"{group}_FOLD_DONE fold={fold-1} acc={metric['acc']:.6f}", flush=True)
            torch_write(state_path, dict(phase="COMPLETE", config=group,
                                         rows=rows, diagnostics=diagnostics,
                                         seconds=previous, started_utc=start_utc,
                                         completed_utc=utc()))
    def finish_group(manifest):
        manifest["workers"][group] = dict(status="COMPLETE", gpu=gpu,
                                          folds_completed=10, seconds=previous,
                                          completed_utc=utc())
        manifest["fold_status"][group] = ["COMPLETE"] * 10
        manifest["completed_folds"] = sum(
            int(v.get("folds_completed", 0))
            for v in manifest["workers"].values())
    update_manifest(finish_group)


def preflight(gpu=0):
    import torch
    c = context(gpu)
    seed_everything(torch, 0)
    x, y, folds, _ = c["x"], c["y"], c["folds"], c["modalities"]
    endpoints, fam_roi, fam_edge, real_roi, real_edge, wrong_roi, wrong_edge, ontology = load_structures()
    train, test = folds[0]
    initial_rng = rng_capture(torch)
    agent = make_agent(c, 1)
    baseline_init_hash = parameter_hash(agent)
    historical = historical_rows()[0]
    if baseline_init_hash != historical["initial_trunk_sha256"]:
        raise AssertionError("E0 initialization differs from Experiment32 fold 1")
    with torch.no_grad():
        out_a = evaluate_base(agent, c, detail=True)
        out_b = evaluate_base(agent, c, detail=True)
    if not torch.equal(out_a["logp"], out_b["logp"]) or not torch.equal(out_a["cam"], out_b["cam"]):
        raise AssertionError("B0-A bypass forward is not deterministic in eval mode")
    # Round-trip both historical Adam optimizers, complete parameters and RNG,
    # then verify the very next original simple-2 update is replayable.
    dual_path = RUNTIME / "preflight_dual_adam.pt"
    dual_initial = dict(model=model_state(agent),
                        optimizer_mp=copy.deepcopy(agent.optimizer_MP0.state_dict()),
                        optimizer_mf=copy.deepcopy(agent.optimizer_MF.state_dict()),
                        rng=rng_capture(torch))
    torch_write(dual_path, dual_initial)
    first_step = train_step(agent, c)
    first_state = model_state(agent)
    first_rng = rng_capture(torch)
    restored_dual = torch_load(dual_path)
    load_model_state(agent, restored_dual["model"])
    agent.optimizer_MP0.load_state_dict(restored_dual["optimizer_mp"])
    agent.optimizer_MF.load_state_dict(restored_dual["optimizer_mf"])
    rng_restore(torch, restored_dual["rng"])
    second_step = train_step(agent, c)
    second_state = model_state(agent)
    dual_replay = (first_step["loss"] == second_step["loss"] and
                   all(torch.equal(first_state[mod][key], second_state[mod][key])
                       for mod in first_state for key in first_state[mod]) and
                   rng_hash(first_rng) == rng_hash(rng_capture(torch)))
    load_model_state(agent, dual_initial["model"])
    agent.optimizer_MP0.load_state_dict(dual_initial["optimizer_mp"])
    agent.optimizer_MF.load_state_dict(dual_initial["optimizer_mf"])
    rng_restore(torch, dual_initial["rng"])
    dual_path.unlink(missing_ok=True)
    if not dual_replay:
        raise AssertionError("Dual-Adam/RNG checkpoint did not replay the next update")
    rng_restore(torch, initial_rng)
    arrays = {}
    norm_cond = fold_normative_features(x, train, y, endpoints, real_roi,
        real_edge, fam_roi, fam_edge, mode="conditional")
    norm_global = fold_normative_features(x, train, y, endpoints, real_roi,
        real_edge, fam_roi, fam_edge, mode="global")
    mutated_y = y.copy()
    mutated_y[test] = 1 - mutated_y[test]
    check = fold_normative_features(x, train, mutated_y, endpoints, real_roi,
        real_edge, fam_roi, fam_edge, mode="conditional")
    if not np.array_equal(norm_cond["z"], check["z"]):
        raise AssertionError("Outer-test labels influenced normative features")
    if norm_cond["z"].shape != (871, 256) or norm_global["z"].shape != (871, 256):
        raise AssertionError("Normative feature shape mismatch")
    fd_probe = x.copy()
    fd_probe[:, 60] += .1 * np.sin(np.arange(len(x)))
    fd_changed = fold_normative_features(fd_probe, train, y, endpoints, real_roi,
        real_edge, fam_roi, fam_edge, mode="conditional")
    efc_probe = x.copy()
    efc_probe[:, 54] += .1 * np.sin(np.arange(len(x)))
    efc_changed = fold_normative_features(efc_probe, train, y, endpoints, real_roi,
        real_edge, fam_roi, fam_edge, mode="conditional")
    if (np.array_equal(norm_cond["z"], fd_changed["z"]) or
            not np.array_equal(norm_cond["z"], efc_changed["z"])):
        raise AssertionError("Reference covariate identity is not verified mean FD column 60")
    if not np.isfinite(norm_cond["z"]).all() or np.max(np.abs(norm_cond["z"])) > 4.0:
        raise AssertionError("Normative z-score finite/range check failed")
    if np.any(real_roi[:, 82]) or np.any(real_edge[:, [195, 225, 232, 236]]):
        raise AssertionError("Unknown ROI/FC features entered semantic masks")
    if int(ontology["wrong_semantics_control"]["real_vs_wrong_changed_roi_assignments"]) == 0:
        raise AssertionError("Wrong semantics mask did not change ROI identity")
    sem_real = summarize_concepts(norm_cond["edge_features"], norm_cond["alignment_edge_support"],
        norm_cond["roi_features"], real_roi, real_edge,
        norm_cond["known_edge_mask"])
    sem_wrong = summarize_concepts(norm_cond["edge_features"], norm_cond["alignment_edge_support"],
        norm_cond["roi_features"], wrong_roi, wrong_edge,
        norm_cond["known_edge_mask"])
    if np.allclose(sem_real, sem_wrong):
        raise AssertionError("Correct and wrong ontology produce identical evidence")
    if (np.any(norm_cond["alignment_edge_support"] > norm_cond["edge_support"] + 1e-7)
            or np.any(norm_cond["alignment_edge_support"][:, [195, 225, 232, 236]])):
        raise AssertionError("ROI coverage or unknown endpoint support constraint failed")
    td_sem = sem_real[train][y[train] == 1]
    td_mag = 1.0 - np.exp(-0.25 * np.clip(td_sem[:, :, 0] + td_sem[:, :, 1], 0.0, 8.0))
    thresholds = np.maximum(np.quantile(.55 * td_mag + .30 *
        np.maximum(td_sem[:, :, 2], td_sem[:, :, 3]), CONCEPT_TAIL_QUANTILE, axis=0), .02)
    direction_weight = np.asarray([p["directional_prior_weight_negative"]
        for p in json.loads((EXP / "prompts.json").read_text(encoding="utf-8"))["concepts"]])
    match, unmatched = semantic_mass(sem_real, ELIGIBLE, True, thresholds,
                                     direction_weight, SEMANTIC_MATCH_SCALE)
    if (not np.isfinite(match).all() or np.any(match < 0) or np.any(unmatched <= 0) or
            not np.allclose(match.sum(1) + unmatched, 1.0, atol=1e-10)):
        raise AssertionError("Rejectable semantic mass invalid")
    low_signal = np.zeros((2, 7, 6), dtype=np.float64)
    low_match, low_unmatched = semantic_mass(low_signal, ELIGIBLE, True,
                                              np.full(7, .02))
    if not np.array_equal(low_match, np.zeros_like(low_match)) or not np.all(low_unmatched == 1):
        raise AssertionError("Low-support evidence does not reject to unmatched channel")

    evidence_pre = dict(norm_cond, train_idx=train, test_idx=test, y=y,
        stats={"td_oof_abs_z_q95": norm_cond["td_oof_abs_z_q95"],
               "td_oof_abs_z_median": norm_cond["td_oof_abs_z_median"],
               "clip_fraction": norm_cond["clip_fraction"],
               "max_leverage": norm_cond["max_leverage"],
               "conditional_support_mean": norm_cond["conditional_support_mean"],
               "qc_support_mean": norm_cond["qc_support_mean"],
               "roi_degree": norm_cond["roi_degree"]})
    structures = load_structures()
    numeric_roi, numeric_edge, numeric_diag = make_numeric_prior(x, train, endpoints)
    vectors, group_diagnostics = {}, {}
    for group in CONFIGS[1:]:
        prior = ((numeric_roi, numeric_edge, numeric_diag)
                 if group == "E8_DATA_PRIOR" else None)
        vectors[group], group_diagnostics[group] = feature_vector(
            group, out_a["cam"].cpu().numpy(), x[:, 64:], evidence_pre,
            structures, data_prior=prior)
        if vectors[group].shape != (871, FEATURE_DIM) or not np.isfinite(vectors[group]).all():
            raise AssertionError(f"{group} preflight feature vector invalid")
    if not np.array_equal(vectors["E1_RAW_CAPACITY"][:, 144:400], x[:, 64:]):
        raise AssertionError("E1 raw capacity control does not retain all 256 inputs")
    for group in ("E6_REAL_NO_REJECT", "E7_WRONG_SEMANTICS",
                  "E8_DATA_PRIOR", "E9_UCSMA_FULL"):
        mass = vectors[group][:, -8:]
        if np.any(mass < 0) or not np.allclose(mass.sum(1), 1.0, atol=1e-6):
            raise AssertionError(f"{group} semantic/reject mass is not a probability partition")
    if not np.all(vectors["E6_REAL_NO_REJECT"][:, -1] == 0.0):
        raise AssertionError("E6 must disable reject mass")
    if np.allclose(vectors["E6_REAL_NO_REJECT"], vectors["E9_UCSMA_FULL"]):
        raise AssertionError("E6 and E9 reject ablation has no computed difference")
    unsupported = dict(evidence_pre)
    unsupported["alignment_edge_support"] = np.zeros_like(evidence_pre["alignment_edge_support"])
    unsupported["roi_features"] = evidence_pre["roi_features"].copy()
    unsupported["roi_features"][:, :, 4] = 0
    rejected_vector, _ = feature_vector("E9_UCSMA_FULL", out_a["cam"].cpu().numpy(),
        x[:, 64:], unsupported, structures)
    if (np.any(rejected_vector[:, -50:-8]) or
            not np.all(rejected_vector[:, -1] == 1.0)):
        raise AssertionError("Unsupported semantic features bypassed actual rejection gating")
    if (group_diagnostics["E9_UCSMA_FULL"]["reject_mean"] <= 0 or
            group_diagnostics["E9_UCSMA_FULL"]["reject_mean"] >= .98):
        raise AssertionError("E9 has no actual unmatched mass in preflight")

    branch = UCSMAResidual(FEATURE_DIM, 64, 144).to(agent.dev)
    input_np = vectors["E9_UCSMA_FULL"]
    feat = torch.as_tensor(input_np, device=agent.dev)
    cam = out_a["cam"].detach()
    z_cam, delta = branch(feat, cam)
    if not torch.equal(z_cam, cam) or float(delta.detach().norm()) != 0.0:
        raise AssertionError("Zero-initialized residual does not restore exact B0 CAM")
    agent.ModalFusion.eval()
    out_zero = agent.ModalFusion.Outputlayer(z_cam.reshape(871, -1),
                                              out_a["attention"])[0]
    if not torch.equal(out_zero, out_a["logp"]):
        raise AssertionError("Zero residual does not restore exact B0 prediction")
    for p in list(agent.MessagePassing0.parameters()) + list(agent.ModalFusion.parameters()):
        p.requires_grad_(False)
    for p in agent.ModalFusion.Outputlayer.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(branch.parameters(), lr=PROTOCOL["lr"], weight_decay=0.0)
    target = agent.targ
    for _ in range(2):
        branch.train()
        agent.ModalFusion.Outputlayer.eval()
        opt.zero_grad(set_to_none=True)
        aug, _ = branch(feat, cam)
        logp, _ = agent.ModalFusion.Outputlayer(aug.reshape(871, -1), out_a["attention"])
        loss = c["utils"].ClsLoss(logp, target, agent.trn_idx, agent.weight)
        loss.backward()
        opt.step()
    if any(p.grad is None or not torch.isfinite(p.grad).all() or p.grad.norm() == 0
           for p in branch.parameters()):
        raise AssertionError("Adapter did not get finite nonzero gradients after two steps")
    pred_after, delta_after = evaluate_branch(branch, agent.ModalFusion.Outputlayer,
                                               feat, cam, out_a["attention"])
    ratio = delta_after.norm(dim=1) / cam.reshape(871, -1).norm(dim=1).clamp_min(1e-8)
    if not torch.isfinite(pred_after).all() or float(ratio.max()) > .050001:
        raise AssertionError("Branch output or smooth residual bound check failed")

    # Exact save/restore of the branch optimizer and all RNG states.
    probe = RUNTIME / "preflight_restore.pt"
    saved = dict(branch=branch.state_dict(), optimizer=opt.state_dict(), rng=rng_capture(torch))
    torch_write(probe, saved)
    step_before = rng_capture(torch)
    saved_rng_hash = rng_hash(saved["rng"])
    replay_a_rng_hash = rng_hash(step_before)
    branch_before_a = {k: v.detach().clone() for k, v in branch.state_dict().items()}
    optimizer_before_a = copy.deepcopy(opt.state_dict())
    replay_a = branch_step(branch, agent.ModalFusion.Outputlayer, opt, feat, cam,
                           out_a["attention"], agent, c, train,
                           out_a["graph_loss"])
    reference = {k: v.detach().clone() for k, v in branch.state_dict().items()}
    disk_saved = torch_load(probe)
    branch.load_state_dict(disk_saved["branch"])
    opt.load_state_dict(disk_saved["optimizer"])
    rng_restore(torch, disk_saved["rng"])
    restored_rng_hash = rng_hash(rng_capture(torch))
    branch_restore_exact = all(torch.equal(branch_before_a[k].detach().cpu(),
        v.detach().cpu()) for k, v in branch.state_dict().items())
    optimizer_restore_exact = tree_equal(optimizer_before_a, opt.state_dict())
    replay_b = branch_step(branch, agent.ModalFusion.Outputlayer, opt, feat, cam,
                           out_a["attention"], agent, c, train,
                           out_a["graph_loss"])
    mismatch = [k for k, v in branch.state_dict().items()
                if not torch.equal(reference[k], v)]
    restored = (not mismatch and branch_restore_exact and optimizer_restore_exact and
                torch.equal(replay_a["output"], replay_b["output"]) and
                replay_a["loss"] == replay_b["loss"] and
                replay_a["cls_loss"] == replay_b["cls_loss"])
    replay_detail = dict(mismatch=mismatch,
                         loss_a=replay_a["loss"], loss_b=replay_b["loss"],
                         cls_a=replay_a["cls_loss"], cls_b=replay_b["cls_loss"],
                         output_equal=torch.equal(replay_a["output"], replay_b["output"]),
                         branch_restore_exact=branch_restore_exact,
                         optimizer_restore_exact=optimizer_restore_exact,
                         saved_rng_hash=saved_rng_hash,
                         replay_a_rng_hash=replay_a_rng_hash,
                         restored_rng_hash=restored_rng_hash)
    probe.unlink(missing_ok=True)
    if not restored:
        raise AssertionError(f"Optimizer/RNG restore did not replay the next update: {replay_detail}")
    close_helper(agent)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["preflight"] = dict(status="PASS", utc=utc(), gpu=gpu,
        first_fold_initial_trunk_sha256=baseline_init_hash,
        e32_fold1_initial_trunk_sha256=historical["initial_trunk_sha256"],
        data_shape=list(x.shape), labels={"ASD_0": 403, "TD_1": 468},
        endpoint_verified=252, endpoint_unknown=[195, 225, 232, 236],
        roi_counts=dict(covered=int((norm_cond["roi_degree"] > 0).sum()),
                        degree_ge3=int((norm_cond["roi_degree"] >= 3).sum()),
                        degree_1_2=int(((norm_cond["roi_degree"] > 0) &
                                        (norm_cond["roi_degree"] < 3)).sum()),
                        zero=int((norm_cond["roi_degree"] == 0).sum()),
                        max=int(norm_cond["roi_degree"].max())),
        test_label_mutation_invariant=True,
        all_branch_input_dimensions_match=True,
        semantic_and_reject_mass_valid=True,
        e6_only_reject_disabled=True,
        e9_nonzero_unmatched_mass=float(group_diagnostics["E9_UCSMA_FULL"]["reject_mean"]),
        e1_retains_all_256_processed_fc=True,
        dual_adam_rng_next_step_replay_exact=dual_replay,
        global_vs_conditional_z_mean_abs=float(np.mean(np.abs(norm_global["z"] - norm_cond["z"]))),
        semantic_wrong_map_changes_evidence=True,
        low_support_rejects=True,
        mean_fd_global_column=60,
        mean_fd_perturbation_changes_normative_z=True,
        efc_perturbation_does_not_change_normative_z=True,
        roi_coverage_propagates_to_concept_support=True,
        rejected_semantic_features_exact_zero=True,
        initial_residual_exact_zero=True,
        branch_max_residual_ratio=float(ratio.max()),
        two_step_nonzero_gradients=True,
        checkpoint_next_step_replay_exact=True,
        actual_feature_dim=FEATURE_DIM,
        trainable_adapter_parameters=sum(p.numel() for p in branch.parameters()))
    manifest["status"] = "PREFLIGHT_PASS"
    json_write(MANIFEST_PATH, manifest)
    print(json.dumps(manifest["preflight"], indent=2, ensure_ascii=False))


def worker_wrapper(group, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["E41_PHYSICAL_GPU"] = str(gpu)
    if group == "E0_B0A":
        run_e0_worker(gpu)
    else:
        run_branch_worker(group, gpu)


def _spawn(group, gpu):
    log = RUNTIME / "logs" / f"{group}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), E41_PHYSICAL_GPU=str(gpu),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    out = log.open("a", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-B", str(EXP / "run.py"),
                             "_worker", group, str(gpu)],
                            env=env, stdout=out, stderr=subprocess.STDOUT)
    return proc, out


def wait_batch(batch):
    failures = []
    for group, gpu, proc, stream in batch:
        code = proc.wait()
        stream.close()
        if code != 0:
            failures.append((group, gpu, code))
    if failures:
        raise RuntimeError(f"Worker failures (completed checkpoints remain resumable): {failures}")


def train():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not manifest.get("preflight") or manifest["preflight"].get("status") != "PASS":
        raise RuntimeError("Formal training requires a passed preflight")
    started = time.monotonic()
    p, f = _spawn("E0_B0A", 0)
    wait_batch([("E0_B0A", 0, p, f)])
    e0_state = torch_load(task_path("E0_B0A"))
    parity = verify_e0(e0_state["rows"])
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["e0_history_verification"] = parity
    manifest["workers"]["E0_B0A"] = dict(
        status="COMPLETE", gpu=0, folds_completed=10,
        seconds=float(e0_state.get("seconds", 0.0)), completed_utc=utc())
    json_write(MANIFEST_PATH, manifest)
    if not parity["pass"]:
        manifest["status"] = "E0_PARITY_FAILED"
        json_write(MANIFEST_PATH, manifest)
        raise RuntimeError("E0 parity check failed; no E1-E9 runs launched")
    x, y, folds, _ = read_data()
    structures = load_structures()
    write_evidence_cache(x, y, folds, structures)
    gpu0_groups = ["E2_GLOBAL_NORM", "E4_ROI_NORM", "E6_REAL_NO_REJECT", "E8_DATA_PRIOR"]
    gpu1_groups = ["E1_RAW_CAPACITY", "E3_COND_NORM", "E5_MULTISCALE_NUM", "E7_WRONG_SEMANTICS"]
    batch = []
    for group in gpu0_groups:
        p, f = _spawn(group, 0)
        batch.append((group, 0, p, f))
    for group in gpu1_groups:
        p, f = _spawn(group, 1)
        batch.append((group, 1, p, f))
    wait_batch(batch)
    p, f = _spawn("E9_UCSMA_FULL", 1)
    wait_batch([("E9_UCSMA_FULL", 1, p, f)])
    elapsed = time.monotonic() - started
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["e0_wall_seconds"] = float(e0_state.get("seconds", 0.0))
    manifest["auxiliary_wall_seconds"] = elapsed
    manifest["training_total_wall_seconds"] = float(e0_state.get("seconds", 0.0)) + elapsed
    manifest["status"] = "TRAINED"
    manifest["completed_folds"] = 100
    json_write(MANIFEST_PATH, manifest)
    print(f"TRAINING_COMPLETE folds=100 wall_seconds={elapsed:.2f}")


def task_rows(group):
    state = torch_load(task_path(group))
    if state.get("phase") != "COMPLETE":
        raise RuntimeError(f"{group} incomplete: {state.get('phase')}")
    return state["rows"], state.get("diagnostics", [])


def load_anchor_torch(fold):
    import torch
    return torch_load(RUNTIME / "anchors" / f"anchor_fold{fold:02d}.pt")


def counterfactuals():
    import torch
    c0 = context(1)
    seed_everything(torch, 0)
    structures = load_structures()
    endpoints, fam_roi, fam_edge, real_roi, real_edge, wrong_roi, wrong_edge, ontology = structures
    out = []
    e9_rows = {int(r["fold"]): r for r in task_rows("E9_UCSMA_FULL")[0]}
    for fold, (train, test) in enumerate(c0["folds"], 1):
        anchor = load_anchor_torch(fold)
        agent = make_agent(c0, fold)
        load_model_state(agent, anchor["model"])
        for p in list(agent.MessagePassing0.parameters()) + list(agent.ModalFusion.parameters()):
            p.requires_grad_(False)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(rng_subseed(fold))
            torch.cuda.manual_seed_all(rng_subseed(fold))
            branch = UCSMAResidual(FEATURE_DIM, 64, 144).to(agent.dev)
        best = torch_load(RUNTIME / "checkpoints" / f"E9_UCSMA_FULL_fold{fold:02d}_best.pt")
        branch.load_state_dict(best["branch"])
        evidence = load_evidence(fold, "conditional")
        evidence["y"] = c0["y"]
        x = c0["x"]
        real_vector, _ = feature_vector("E9_UCSMA_FULL", anchor["cam"].numpy(),
                                         x[:, 64:], evidence, structures)
        forced_vector, _ = feature_vector("E9_UCSMA_FULL", anchor["cam"].numpy(),
                                           x[:, 64:], evidence, structures,
                                           reject_override=False)
        wrong_vector, _ = feature_vector("E7_WRONG_SEMANTICS", anchor["cam"].numpy(),
                                          x[:, 64:], evidence, structures,
                                          reject_override=True)
        # Keep the same full UCSMA residual function and route mass while replacing
        # only the frozen anatomical correspondence with the registered wrong mask.
        wrong_vector = wrong_vector.copy()
        features_real = torch.as_tensor(real_vector, device=agent.dev)
        features_wrong = torch.as_tensor(wrong_vector, device=agent.dev)
        cam, attention = anchor["cam"].to(agent.dev), anchor["attention"].to(agent.dev)
        methods = {"REAL": features_real}
        methods["NO_REJECT"] = torch.as_tensor(forced_vector, device=agent.dev)
        # Patient mismatch shuffles all FC-derived fields within train and test
        # separately, leaving the true CAM, labels and split membership unchanged.
        patient = real_vector.copy()
        rng = np.random.default_rng(410000 + fold)
        for partition in (np.asarray(train), np.asarray(test)):
            perm = derangement(partition, 410000 + fold + len(partition))
            patient[partition, FIELD_SIZES[0]:] = real_vector[perm, FIELD_SIZES[0]:]
        methods["PATIENT_SHUFFLE"] = torch.as_tensor(patient, device=agent.dev)
        methods["WRONG_MASK"] = features_wrong
        uniform = real_vector.copy()
        start_align = sum(FIELD_SIZES[:-1])
        concept_start = sum(FIELD_SIZES[:-2])
        ungated_concepts = summarize_concepts(evidence["edge_features"],
            evidence["alignment_edge_support"], evidence["roi_features"],
            real_roi, real_edge, evidence["known_edge_mask"])
        uniform[:, concept_start:start_align] = (ungated_concepts *
            ELIGIBLE[None, :, None] * .25).reshape(len(x), -1)
        uniform[:, start_align:start_align + 7] = 0.0
        uniform[:, start_align + np.asarray([0, 1, 2, 5])] = .25
        uniform[:, start_align + 7] = 0.0
        methods["UNIFORM_ALIGNMENT"] = torch.as_tensor(uniform, device=agent.dev)
        no_sem = real_vector.copy()
        concept_start = sum(FIELD_SIZES[:-2])
        no_sem[:, concept_start:concept_start + FIELD_SIZES[-2]] = 0.0
        no_sem[:, start_align:] = 0.0
        methods["SEMANTIC_OFF"] = torch.as_tensor(no_sem, device=agent.dev)
        real_logp, real_residual = evaluate_branch(branch, agent.ModalFusion.Outputlayer,
                                                   features_real, cam, attention)
        with torch.no_grad():
            base_logp, _ = agent.ModalFusion.Outputlayer(cam.reshape(871, -1), attention)
        real_pred = real_logp[test].argmax(1)
        base_pred = base_logp[test].argmax(1)
        for name, feats in methods.items():
            logp, residual = evaluate_branch(branch, agent.ModalFusion.Outputlayer,
                                              feats, cam, attention)
            metric = metrics_row(logp, c0["y"], test)
            base_prediction = anchor["base_row"]
            out.append(dict(fold=fold, intervention=name, **metric,
                            residual_norm_test=float(residual[test].norm(dim=1).mean().item()),
                            residual_ratio_test=float((residual[test].norm(dim=1) /
                                cam[test].reshape(len(test), -1).norm(dim=1).clamp_min(1e-8)).mean().item()),
                            prediction_flips_vs_real=int((logp[test].argmax(1) != real_pred).sum()),
                            prediction_flips_vs_b0a=int((logp[test].argmax(1) != base_pred).sum()),
                            mean_abs_logp_delta_vs_real=float((logp[test]-real_logp[test]).abs().mean()),
                            mean_residual_delta_l2_vs_real=float((residual[test]-real_residual[test]).norm(dim=1).mean()),
                            e9_selected_epoch=int(e9_rows[fold]["best_epoch"])))
        with torch.no_grad():
            zero = torch.zeros_like(cam)
            off, _ = agent.ModalFusion.Outputlayer((cam + zero).reshape(871, -1), attention)
        out.append(dict(fold=fold, intervention="ALL_OFF", **metrics_row(off, c0["y"], test),
                        residual_norm_test=0.0, residual_ratio_test=0.0,
                        prediction_flips_vs_real=int((off[test].argmax(1) != real_pred).sum()),
                        prediction_flips_vs_b0a=int((off[test].argmax(1) != base_pred).sum()),
                        mean_abs_logp_delta_vs_real=float((off[test]-real_logp[test]).abs().mean()),
                        mean_residual_delta_l2_vs_real=float(real_residual[test].norm(dim=1).mean()),
                        e9_selected_epoch=int(e9_rows[fold]["best_epoch"])))
        close_helper(agent)
        del agent, branch, anchor, evidence
        torch.cuda.empty_cache()
    return out


def posthoc_audits():
    """Read-only checkpoint diagnostics: no optimizer step or model selection."""
    import ast
    import types
    import torch
    c = context(1)
    structures = load_structures()
    endpoints, _, _, real_roi, real_edge, _, wrong_edge, ontology = structures
    # Evaluate the actual historical Experiment32 B0 forward implementation,
    # compiled from its audited patch function without importing its experiment.
    historical_integration = HIST32 / "integration.py"
    tree = ast.parse(historical_integration.read_text())
    patch_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "patch_network")
    ns = {"ast": ast}
    exec(compile(ast.Module(body=[patch_fn], type_ignores=[]), str(historical_integration), "exec"), ns)
    ref_source = ns["patch_network"]((ROOT / "network.py").read_text())
    ref_tree = ast.parse(ref_source)
    ref_class = next(n for n in ref_tree.body if isinstance(n, ast.ClassDef) and n.name == "PLN")
    ref_forward = next(n for n in ref_class.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    fns = dict(c["network"].__dict__)
    exec(compile(ast.Module(body=[ref_forward], type_ignores=[]), "E32_B0_forward", "exec"), fns)
    counts, parity_rows, norm_rows = [], [], []
    for fold, (train, test) in enumerate(c["folds"], 1):
        anchor = load_anchor_torch(fold)
        agent = make_agent(c, fold)
        load_model_state(agent, anchor["model"])
        own = evaluate_base(agent, c, detail=True)
        saved_forward = agent.ModalFusion.forward
        agent.ModalFusion.dcer32_config = "B0"
        agent.ModalFusion.forward = types.MethodType(fns["forward"], agent.ModalFusion)
        ref = evaluate_base(agent, c, detail=True)
        agent.ModalFusion.forward = saved_forward
        same = {k: bool(torch.equal(own[k], ref[k]))
                for k in ("gfr", "cam", "head_cam", "attention", "logp")}
        same["all_four_adjacencies"] = all(torch.equal(a, b)
            for a, b in zip(own["adjs"], ref["adjs"]))
        if not all(same.values()):
            raise AssertionError(f"Layerwise E32 parity failed at fold {fold}: {same}")
        parity_rows.append(dict(fold=fold, **same))
        for parameter in list(agent.MessagePassing0.parameters()) + list(agent.ModalFusion.parameters()):
            parameter.requires_grad_(False)
        cam, attention = anchor["cam"].to(agent.dev), anchor["attention"].to(agent.dev)
        numeric_prior = make_numeric_prior(c["x"], train, endpoints)
        for mode in ("global", "conditional"):
            e = load_evidence(fold, mode)
            norm_rows.append(dict(fold=fold, mode=mode, **e["stats"],
                scale_floor_columns=int((e["sigma"] <= .05000001).sum()),
                train_td_count=int((c["y"][train] == 1).sum())))
        for group in CONFIGS[1:]:
            mode = "global" if group == "E2_GLOBAL_NORM" else "conditional"
            evidence = load_evidence(fold, mode)
            evidence["y"] = c["y"]
            vector, _ = feature_vector(group, anchor["cam"].numpy(), c["x"][:, 64:],
                evidence, structures, data_prior=numeric_prior if group == "E8_DATA_PRIOR" else None)
            checkpoint = torch_load(RUNTIME / "checkpoints" / f"{group}_fold{fold:02d}_best.pt")
            if sha_bytes(vector.tobytes()) != checkpoint["vector_sha256"]:
                raise AssertionError(f"Checkpoint feature identity mismatch: {group}/{fold}")
            branch = UCSMAResidual(FEATURE_DIM, 64, 144).to(agent.dev)
            branch.load_state_dict(checkpoint["branch"])
            features = torch.as_tensor(vector, device=agent.dev)
            aug, residual = branch(features, cam)
            logp, _ = agent.ModalFusion.Outputlayer(aug.reshape(871, -1), attention)
            loss = c["utils"].ClsLoss(logp, agent.targ, train, agent.weight)
            loss.backward()
            active_columns = int(np.any(vector[train] != 0.0, axis=0).sum())
            nonzero_gradient = sum(int(torch.count_nonzero(p.grad))
                for p in branch.parameters() if p.grad is not None)
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in branch.parameters()):
                raise AssertionError("Non-finite or absent posthoc branch gradient")
            with torch.no_grad():
                raw = branch.output(torch.nn.functional.gelu(branch.input(features)))
                cap = .05 * cam.reshape(871, -1).norm(dim=1).clamp_min(1e-8)
                dose = raw.norm(dim=1) / cap
                base_logp, _ = agent.ModalFusion.Outputlayer(cam.reshape(871, -1), attention)
            counts.append(dict(config=group, fold=fold,
                registered_trainable_params=sum(p.numel() for p in branch.parameters()),
                active_input_columns=active_columns,
                input_connected_parameters=active_columns*64 + 64*144,
                best_checkpoint_nonzero_gradient_parameters=nonzero_gradient,
                best_checkpoint_prediction_flips_vs_b0a=int((logp[test].argmax(1) != base_logp[test].argmax(1)).sum()),
                raw_residual_over_cap_median=float(dose[test].median()),
                raw_residual_over_cap_max=float(dose[test].max()),
                smooth_cap_fraction_above_99pct=float((torch.tanh(dose[test]) > .99).double().mean()),
                best_checkpoint_feature_identity_verified=True))
            del branch, features
        close_helper(agent)
        del agent, anchor
        torch.cuda.empty_cache()
    # Audits describe the actual control, not a fictional exactly matched one.
    real = real_edge.astype(int)
    wrong = wrong_edge.astype(int)
    controls = dict(real_edge_counts=real.sum(1).tolist(), wrong_edge_counts=wrong.sum(1).tolist(),
        real_pairwise_edge_overlap=(real @ real.T).tolist(),
        wrong_pairwise_edge_overlap=(wrong @ wrong.T).tolist(),
        real_union_edge_count=int(real.any(0).sum()), wrong_union_edge_count=int(wrong.any(0).sum()),
        same_concept_real_wrong_overlap=(real*wrong).sum(1).tolist(),
        wrong_control_actual_algorithm="seed4107 fixed random verified-edge sets matching each concept edge count; degree-stratified ROI metadata is not the computational edge mask")
    return dict(branch_counts=counts, normative=norm_rows,
                e0_layer_parity=parity_rows, ontology_controls=controls,
                historical_forward_source_sha256=sha_file(historical_integration),
                read_only_no_optimizer_step=True)


def summarize(rows):
    summary = {}
    for group in CONFIGS:
        selected = sorted([r for r in rows if r["config"] == group],
                          key=lambda r: int(r["fold"]))
        if len(selected) != 10:
            raise RuntimeError(f"{group} has {len(selected)}/10 rows")
        summary[group] = {}
        for metric in ("acc", "hard_auc", "spe_td", "sen_asd"):
            values = np.asarray([float(r[metric]) for r in selected])
            summary[group][metric] = dict(mean=float(values.mean()),
                                          population_sd=float(values.std(ddof=0)),
                                          values=values.tolist())
    return summary


def paired(rows, left="E9_UCSMA_FULL"):
    by = {(r["config"], int(r["fold"])): r for r in rows}
    output = {}
    for right in CONFIGS:
        if right == left:
            continue
        diffs = {m: [float(by[(left, f)][m]) - float(by[(right, f)][m])
                     for f in range(1, 11)]
                 for m in ("acc", "hard_auc", "spe_td", "sen_asd")}
        stats = {}
        for metric, values in diffs.items():
            pp = np.asarray(values) * 100.0
            stats[metric] = dict(mean_delta_pp=float(pp.mean()),
                                 population_sd_pp=float(pp.std(ddof=0)),
                                 wins=int((pp > 1e-12).sum()),
                                 ties=int((np.abs(pp) <= 1e-12).sum()),
                                 losses=int((pp < -1e-12).sum()),
                                 fold_delta_pp=pp.tolist())
        output[right] = stats
    return output


def fmt_percent(metric):
    return f"{100*metric['mean']:.4f} ± {100*metric['population_sd']:.4f}"


def make_report(rows, diagnostics, counter, summary, paired_results, manifest):
    parity = manifest.get("e0_history_verification") or {}
    ontology = json.loads((EXP / "ontology.json").read_text(encoding="utf-8"))
    prompt = json.loads((EXP / "prompts.json").read_text(encoding="utf-8"))
    lines = [
        "# 实验41：UCSMA-v2 完整结果", "",
        f"状态：**{manifest.get('status')}，{manifest.get('completed_folds', 0)}/100折**；E0历史复现：**{'PASS' if parity.get('pass') else 'FAIL'}（{parity.get('matched_folds', 0)}/10）**。",
        f"有效正式折GPU分配：GPU0为E0/E2/E4/E6/E8，共50折；GPU1为E1/E3/E5/E7/E9，共50折。E0独立训练 {manifest.get('e0_wall_seconds', 0):.2f} 秒，初次增量调度 {manifest.get('initial_auxiliary_wall_seconds', 0):.2f} 秒，更正后调度 {manifest.get('corrected_auxiliary_wall_seconds', 0):.2f} 秒，含作废运行的训练调度累计墙钟 {manifest.get('all_computation_training_wall_seconds', 0):.2f} 秒。静态审计、预检、报告和人工中断不计入该训练耗时。", "",
        "### 实现更正与排除记录", "",
        "最终字段/计算图复核发现初次条件参考误取FUNC EFC（ANAT6+FUNC10局部索引6），不是mean FD（局部索引12、全局索引60）；同时概念特征此前没有乘以实际匹配质量，且直接边池化的概念支持没有继承ROI连接度收缩。三项均纠正为原定字段身份、覆盖约束与拒绝约束，未改变Prompt、任何映射、宽度、学习率、阈值、损失、残差上限或GO规则。E3–E9旧70折全部作废，不参与正式统计；更正预检后重算70折。E0/E1/E2保留，E2十折输入逐位一致已核验。总执行170次fold拟合，其中100折为有效正式结果、70折为错误实现的排除记录。GPU0额外作废30折、GPU1额外作废40折。该错误发生在首次正式运行后，是明确的执行偏差，不能写成一次无差错的预注册运行；完整原因、旧结果和代码哈希在`correction_audit.json`。", "",
        "## 源码与论文审计", "",
        f"本地HEAD `{manifest['source_and_input']['local_git_head']}`；审计时GitHub main HEAD `{manifest['source_and_input']['github_main_head']}`。两者不同，以Experiment32冻结的本地源码作为B0-A数学路径；六个核心本地文件SHA256见manifest。", "",
        "本地原论文 Fig.1、Section 2.1–2.2、Eq.(1)–(7)、ABIDE设置和消融均已阅读；另与公开PDF交叉核对。论文描述的SPPromptGL由GFR、PLN/CAM、PEN构成，但本地实现和shell是实验权威。GFR四支均按患者特征构造患者图；每图的节点特征是完整320维输入；`i==1`覆盖`i==0`写入的`combined_feat`，本实验私有源保持原样。四支GraphConstructLoss均值实际进入`simple-2`反向传播，分类损失使用训练集类别权重；`ALoss`会被计算和打印，但不进入loss。两个原始Adam、lr=0.0038、weight_decay=0.11、dropout=0.35、450 epoch、early=200、`val_idx=test_idx`均按历史入口保留。PEN的SPGS/RES前向被旁路；PLN、CLIP缓存和优化器构造仍保留以维持E0随机初始化副作用。", "",
        "公式逐项核查：Eq.(1)纸面图目标包含平滑/图约束和标签监督邻接距离，后者对应本地ALoss但实际未优化；Eq.(2)从完整模态拼接进行患者图传播；Eq.(3)为离线疾病概念生成；Eq.(4)为CAM交互；Eq.(5)投影文本与局部潜变量相似后取max；Eq.(6)RES将局部信息回写全局；Eq.(7)联合图/标签损失。本轮只用Eq.(1)/(2)/(4)对应的历史基座路径，旁路SPGS与RES；局部潜变量不是病灶ROI。", "",
        "原论文ABIDE设置写为dropout=0.75、lr=0.0023、hidden=10、encoding layer=4、head=4，与本地历史shell不同，因此不能把改成论文超参数当作精确复现。原论文Fig.2中ABIDE的GFR、GFR+RES、完整PEN ACC约为91.73%、92.08%、92.30%，Table2完整模型报告AUC为92.22%；这些仅作描述。E32历史B0-A 92.0742%与完整PEN B2 91.7294%才是本地协议参照，正式E0由独立重训产生。Fig.1上部AD示例文本也不能当成本轮ASD知识来源。", "",
        "CAM输出`[871,4,36]`是跨模态交互后的潜在表征，不是四种原始模态，也不是ROI。PHENO48为年龄箱26维、站点20维、性别2维；ANAT6/FUNC10是已有质量与运动特征，FUNC第7列经字段名核实为mean FD；CORR256为预先RFE并标准化后的rs-fMRI连接特征。其正负值按已有处理输入解释，不解释成神经兴奋或抑制。", "",
        "实际shell参数为`--GC_mode weighted-cosine --MF_mode concat --MP_mode GCN --datname ABIDE --dropout 0.35 --lr 0.0038 --mode simple-2 --n_head 2 --n_hidden 18 --nlayer 1 --reg 0.11`。", "",
        "历史审计覆盖Experiment17–19、28–32、34–40：17–18表明强迫CORR/PHENO共享空间可能损伤判别；19–30的CLIP相似度、自由文本投影和静态Query没有稳定独立收益；Experiment38–40已测试规范偏离、疾病文本选边与拒绝/反事实控制，其中E40真实假说ACC 82.8788%、数值Top-|z|对照91.7294%、B0-A 92.0742%。Experiment39正确医学掩码MoE也未胜过容量/随机/数值对照，且强残差风险突出。Experiment34–37未证明语义、条件路由或标签监督的医学语义独立贡献。Experiment41不重复39条选边、自由CLIP文本Query、强制MoE或共享疾病空间。", "",
        "## 数据与医学映射", "",
        "数据为本地既有871人、320维、ASD=0 403人、TD=1 468人；没有换样本或重新预处理。Experiment11的256列恢复映射中252条可核验端点连接，4条未知特征索引195、225、232、236保留主干但不进入ROI/语义掩码。252条连接覆盖110/111 ROI；91个ROI度数≥3，19个ROI度数1–2，1个ROI度数0，最高度数12。", "",
        "未找到可信canonical网络映射，因此仅使用按HO精确名称形成的8种粗粒度解剖区域族代理：感觉运动、额叶关联、顶叶关联、颞叶关联、岛叶、扣带/内侧、视觉、皮质下。它们不等同于Schaefer七网络、DMN、显著性网络或完整脑网络。ROI聚合的正/负偏离分别保留；低度ROI以`degree/(degree+3)`收缩，未观测边不补齐。", "",
        "## 冻结医学Prompt", "",
        "未调用GPT-4/API。以下英文候选文本由用户提供，经过来源与ROI身份审查后冻结；自然语言不送入自由文本编码器，实际模型输入是prompts.json/ontology.json中可核验的结构化掩码。", "",
    ]
    for concept in prompt["concepts"]:
        lines.extend([f"### {concept['concept_id']}：{concept['concept_name']}", "",
                      f"> {concept['prompt_text']}", "",
                      f"来源：{concept['source_title']}；DOI：{concept['doi'] or '方法拒绝通道'}。证据尺度：{concept['evidence_scale']}；方向：{concept['supported_direction']}；可计算：{concept['model_eligible']}；HO ROI数 {len(concept['supported_roi_indices'])}，当前252条连接中适用 {len(concept['supported_edge_indices'])} 条。",
                      ""])
    lines += ["### 可核验的结构对应", "",
              "下表均为零基索引；具体左右半球、分区全称可由ontology.json的roi_names逐项核对。C01是感觉运动与顶叶解剖候选代理，并未核验注意网络归属；C02的HO岛叶没有可靠前/后细分，因此仅作宽泛岛叶候选。C03的内侧/扣带/顶叶标签也不构成已验证DMN。", "",
              "| 概念 | HO ROI索引 | CORR256特征索引 |", "|---|---|---|"]
    for concept in prompt["concepts"]:
        if concept["model_eligible"]:
            lines.append(f"| {concept['concept_id']} | {', '.join(map(str, concept['supported_roi_indices']))} | {', '.join(map(str, concept['supported_edge_indices']))} |")
    lines += [
        "C01–C03和C06为有限、候选的区域或连接集合。C04（DMN/认知控制）与C05（躯体运动—联络系统）因缺少本数据ROI到canonical system的独立对应，只保留文献元数据，不参与建模。C07为未匹配通道。C01负向、C03正向群体描述仅在固定分数15%项中作为弱方向权重（其余方向未知概念对称处理）；个体正、负偏离通道都保留，不把单连接符号转成临床真值。Ilioska et al.报告的是大样本、多尺度规范偏离收敛，不能把其75,855条全连接或高密度分区结果直接外推到本RFE后稀疏子集；Kobbersmed et al.强调系统尺度规范建模，但本数据不具备其全脑覆盖与样本量。", "",
        "主要文献：[Ilioska et al., Nature Mental Health 2026](https://doi.org/10.1038/s44220-026-00656-y)，[Kobbersmed et al., Progress in Neurobiology 2026](https://doi.org/10.1016/j.pneurobio.2026.102906)。[PhenoBrain](https://proceedings.mlr.press/v306/meng26f.html)已有表型条件化脑网络通信，[KD-Brain](https://papers.miccai.org/miccai-2026/0366-Paper0073.html)已有语义条件子网络交互，[BrainPrompt](https://papers.miccai.org/miccai-2025/0116-Paper2260.html)与[BrainPrompt+](https://doi.org/10.1109/TMI.2026.3692958)已有多层脑提示。因此，本轮不把条件化、层次Prompt或语义注意力本身当作新颖性。[To Align or Not to Align, AAAI 2026](https://doi.org/10.1609/aaai.v40i25.39248)支持按证据避免一概强制对齐。", "",
        "## 规范偏离与模块", "",
        "每个外层折的条件参考仅拟合outer-train TD；训练TD与ASD均按同一固定5折内层划分排除自身所在子折，以避免自我参考；测试患者由全部outer-train TD模型预测。条件为年龄箱one-hot、性别、站点one-hot和全局第60列mean FD；年龄箱未被转换成精确年龄。ANAT6/FUNC10共16维质量字段形成稳健距离可靠性代理。Ridge alpha=10；残差尺度为训练TD OOF的1.4826×MAD，尺度下限0.05；连接异常阈值为训练TD OOF经验|z|第95百分位（最低0.5），z截断±4。小站点/年龄条件通过计数支持和ridge leverage降低可信度；计数不足20（年龄/站点）或50（性别）时连续收缩，最低支持0.05，不按测试结果选择站点回退。没有使用测试标签拟合参考、尺度、阈值或Prompt。", "",
        "连接级输入保持符号、幅度、尾部覆盖、参考可靠性和不确定性；ROI尺度按真实端点取加权平均而非异常总数，再以d/(d+3)降低低度ROI可靠性；区域族尺度只做粗粒度代理。直接概念边池化的支持再乘两端ROI度数收缩的几何均值，未知端点支持为零。概念分数由当前患者正负偏离、尾部覆盖、冻结解剖掩码、规范参考支持和QC支持决定。E9匹配质量为`m_k=raw_k/(1+Σraw)`，未匹配质量为`1/(1+Σraw)`；输入Adapter的概念特征明确为`m_k × concept_features_k`，拒绝后语义贡献为零，原数值证据保留。E6仅将匹配质量强制归一化，使用同一特征门控、剂量和网络。没有自由Text→Query投影。", "",
        f"新分支由固定输入维度{FEATURE_DIM}、隐藏宽度64及输出144的无偏置Adapter构成，共{FEATURE_DIM*64 + 64*144:,}个参数；输出层权重严格零初始化，CAM残差使用`smooth tanh`范数限制为不超过CAM范数5%。原GFR/CAM/OutputLayer参数冻结，且没有新增分类头。E1–E9注册容量相同；实际梯度参数按fold记录。", "",
        "E0完整保留双Adam与原simple-2。E1–E9的原参数与OutputLayer处于冻结eval状态，只用一个增量Adam更新Adapter（lr=0.0038、weight_decay=0）；目标为原训练索引加权分类损失加缓存图正则常数，不另堆叠对比、蒸馏、正交或语义分类损失。图正则不再对冻结参数产生梯度。分支初始化在RNG fork内使用410000+fold的固定派生流，所有组同折初值相同；E0仍为历史seed0一次播种后连续十折，两种随机协议明确不同。", "",
        "## 实验组与正式指标", "",
        "E0独立复现B0-A；E1原始CORR容量控制（固定随机投影/重复填充相同输入宽度）；E2全局TD规范参考；E3表型/站点/meanFD条件参考；E4连接+ROI覆盖汇聚；E5再加数值解剖区域族多尺度；E6真实概念但强制匹配；E7固定seed4107随机已核验连接集合，匹配逐概念边数；E8仅在outer-train拟合|FC相关|轮廓的4簇k-means，seed41041、n_init=20；E9真实候选映射及显式拒绝。E7的ROI度数分层置换仅为元数据，实际边集合不保证相同ROI覆盖或边重叠；E8簇大小和交叠也不匹配医学先验。不能声称二者是完全等覆盖对照。E1–E9具有同一注册Adapter，但关闭证据块会降低输入连接的有效容量。", "",
        "正式指标只有ACC、原one-hot硬标签ROC-AUC、SPE-TD、SEN-ASD，单位均为百分比；二分类hard-label AUC等于balanced accuracy，不是概率排序AUC。均值±population SD为十折描述统计。", "",
        "| 组别 | ACC | Hard-AUC | SPE-TD | SEN-ASD |", "|---|---:|---:|---:|---:|",
    ]
    for group in CONFIGS:
        s = summary[group]
        lines.append(f"| {group} | {fmt_percent(s['acc'])} | {fmt_percent(s['hard_auc'])} | {fmt_percent(s['spe_td'])} | {fmt_percent(s['sen_asd'])} |")
    lines += ["", "## 配对比较", "",
              "Δ为E9减对照，单位百分点；W/T/L为相同fold严格胜/平/负。十折训练集重叠，折数不是独立队列，配对数仅作描述，不作显著性检验。", "",
              "| E9 − 对照 | ACC Δ / W-T-L | Hard-AUC Δ / W-T-L | SPE Δ | SEN Δ |", "|---|---:|---:|---:|---:|"]
    for group, val in paired_results.items():
        acc, auc = val["acc"], val["hard_auc"]
        lines.append(f"| {group} | {acc['mean_delta_pp']:+.4f} / {acc['wins']}-{acc['ties']}-{acc['losses']} | {auc['mean_delta_pp']:+.4f} / {auc['wins']}-{auc['ties']}-{auc['losses']} | {val['spe_td']['mean_delta_pp']:+.4f} | {val['sen_asd']['mean_delta_pp']:+.4f} |")
    lines += ["", "各核心比较的逐折差（折1→10，单位pp；四项顺序ACC / Hard-AUC / SPE-TD / SEN-ASD）：", ""]
    for group in ("E0_B0A", "E1_RAW_CAPACITY", "E4_ROI_NORM", "E5_MULTISCALE_NUM",
                  "E6_REAL_NO_REJECT", "E7_WRONG_SEMANTICS", "E8_DATA_PRIOR"):
        fields = ["[" + ", ".join(f"{v:+.4f}" for v in paired_results[group][key]["fold_delta_pp"]) + "]"
                  for key in ("acc", "hard_auc", "spe_td", "sen_asd")]
        lines.append(f"- E9−{group}：" + " / ".join(fields))
    lines += ["", "E3−E2（条件参考）、E4−E3（ROI汇聚）、E5−E4（区域族汇聚）的四项逐折差均为零；因此训练TD参考拟合改善并未转化为可观测分类增益。", "",
              "### 参数与实际梯度", "",
              "所有新增组均注册182,016参数，但不是所有输入列都激活。下表对每个最佳checkpoint额外进行一次训练索引反向诊断，不更新优化器、不重选checkpoint；有效输入连接参数=训练样本上非零输入列×64+64×144，梯度参数按非零梯度元素计数。初次E1/E2训练日志中的182,016是整个非零梯度张量的参数总数，不能用于证明有效容量相同；本表统一采用元素级诊断。", "",
              "| 组别 | 注册新增参数 | 非零输入列范围 | 输入连接参数范围 | 最佳checkpoint非零梯度元素范围 |", "|---|---:|---:|---:|---:|"]
    for group in CONFIGS[1:]:
        subset = [d for d in diagnostics if d["config"] == group]
        def bounds(key):
            vals = [int(d[key]) for d in subset]
            return f"{min(vals)}–{max(vals)}"
        lines.append(f"| {group} | 182016 | {bounds('active_input_columns')} | {bounds('input_connected_parameters')} | {bounds('best_checkpoint_nonzero_gradient_parameters')} |")
    lines += ["", "E1使用原始FC的固定投影/重复填充，虽有更多活跃权重，但其特征存在代数冗余；注册或非零梯度计数不等同于独立信息维数。E0没有新增分支参数。E6/E7/E8/E9同架构，C04/C05/C07医学概念块保持关闭；E8四个数值簇使用四个数值槽位，不将它们冒称医学概念。不能把本实验描述为所有组严格等有效容量。", ""]
    e0_count_rows = [r for r in rows if r["config"] == "E0_B0A"]
    lines += [f"E0原始GFR+PLN构造注册{int(e0_count_rows[0]['trainable_params']):,}参数，历史训练中有梯度的参数张量包含{int(e0_count_rows[0]['effective_gradient_params']):,}参数（张量级统计，包含保留构造但不参与PEN前向的结构）；完整逐折记录在fold_metrics.csv，不能与上述元素级计数直接等同。E1–E9原主干参数均冻结，实际优化参数仅为本表中的分支。", ""]
    lines += ["", "## 机制诊断", "",
              "`diagnostics.csv`保存每折参考拟合、OOF尾部、z截断、规范支持、质量代理、ROI度数、概念质量/拒绝质量、E8簇规模、实际梯度参数和残差比例。诊断不能替代四项正式指标。E9固定checkpoint干预在`counterfactual.csv`比较真实映射、错误映射、分区内患者机制错配、均匀语义权重、关闭语义输入及ALL_OFF；它只说明已训练checkpoint依赖，不能替代E0/E5/E7/E8从头训练比较。", ""]
    for group in ("E3_COND_NORM", "E4_ROI_NORM", "E5_MULTISCALE_NUM",
                  "E6_REAL_NO_REJECT", "E7_WRONG_SEMANTICS", "E8_DATA_PRIOR", "E9_UCSMA_FULL"):
        subset = [d for d in diagnostics if d["config"] == group]
        if not subset:
            continue
        lines.append(f"- `{group}`：outer-test平均参考支持={np.mean([d['test_reference_support'] for d in subset]):.4f}，|z|截断比例={np.mean([d['z_clip_fraction'] for d in subset]):.4f}，非零有效梯度参数={int(np.mean([d['gradient_params_nonzero'] for d in subset]))}。")
    norm = manifest["posthoc_audit"]["normative"]
    lines += ["", "训练TD交叉拟合参考诊断（十折均值；只用于报告机制，未据此选择超参数）：", "",
              "| 参考 | OOF MSE | OOF MAE | OOF |z| 95%分位 | 全体z截断比例 | 训练参考支持 |", "|---|---:|---:|---:|---:|---:|"]
    for mode in ("global", "conditional"):
        subset = [d for d in norm if d["mode"] == mode]
        vals = [np.mean([d[k] for d in subset]) for k in
                ("td_oof_mse", "td_oof_mae", "td_oof_abs_z_q95", "clip_fraction", "conditional_support_mean")]
        lines.append(f"| {mode} | " + " | ".join(f"{v:.6f}" for v in vals) + " |")
    controls = manifest["posthoc_audit"]["ontology_controls"]
    lines += ["", f"真实与错误语义的逐概念连接数均为{controls['real_edge_counts']}；真实先验覆盖并集{controls['real_union_edge_count']}条，错误先验覆盖并集{controls['wrong_union_edge_count']}条；同概念真实/错误集合交集数为{controls['same_concept_real_wrong_overlap']}。完整交叠矩阵在manifest。E7只匹配逐概念边数，覆盖并集及重叠差异是实质性控制局限。", "",
              "参考支持结合OOF残差尺度、年龄/站点/性别计数、ridge leverage、QC稳健距离和ROI连接度；这些是经验不确定性/可靠性代理，未证明临床概率校准，也未将Softmax称为校准不确定性。条件模型OOF误差下降只说明部分采集/表型条件有预测作用，不能把条件项当成ASD病理证据。", ""]
    lines += ["", "### 固定E9 checkpoint干预", "",
              "以下干预不重训、不重新选epoch；均在同一折的E9最佳checkpoint上执行。REAL是训练时冻结映射，NO_REJECT关闭拒绝质量，PATIENT_SHUFFLE只在训练/测试分区内部错配FC证据，WRONG_MASK替换预注册错误HO掩码，UNIFORM_ALIGNMENT固定可用概念权重，SEMANTIC_OFF清除概念段，ALL_OFF清除整个残差。", "",
              "| 干预 | ACC | Hard-AUC | SPE-TD | SEN-ASD |", "|---|---:|---:|---:|---:|"]
    for intervention in ("REAL", "NO_REJECT", "PATIENT_SHUFFLE", "WRONG_MASK",
                         "UNIFORM_ALIGNMENT", "SEMANTIC_OFF", "ALL_OFF"):
        selected = [row for row in counter if row["intervention"] == intervention]
        lines.append("| %s | %.4f | %.4f | %.4f | %.4f |" % (
            intervention,
            100 * np.mean([row["acc"] for row in selected]),
            100 * np.mean([row["hard_auc"] for row in selected]),
            100 * np.mean([row["spe_td"] for row in selected]),
            100 * np.mean([row["sen_asd"] for row in selected])))
    lines += ["", "| 固定checkpoint干预 | 相对REAL预测翻转总数（871次外层测试） | 平均绝对log-prob变化 | 平均残差L2变化 |", "|---|---:|---:|---:|"]
    for intervention in ("REAL", "NO_REJECT", "PATIENT_SHUFFLE", "WRONG_MASK",
                         "UNIFORM_ALIGNMENT", "SEMANTIC_OFF", "ALL_OFF"):
        selected = [row for row in counter if row["intervention"] == intervention]
        lines.append(f"| {intervention} | {sum(r['prediction_flips_vs_real'] for r in selected)} | {np.mean([r['mean_abs_logp_delta_vs_real'] for r in selected]):.8f} | {np.mean([r['mean_residual_delta_l2_vs_real'] for r in selected]):.8f} |")
    e9_diag = [d for d in diagnostics if d["config"] == "E9_UCSMA_FULL"]
    e9_rows = [r for r in rows if r["config"] == "E9_UCSMA_FULL"]
    lines += ["", f"E9的未匹配质量均值为{np.mean([d['test_unmatched_mean'] for d in e9_diag]):.4f}±{np.std([d['test_unmatched_mean'] for d in e9_diag]):.4f}，路由质量熵为{np.mean([d['test_alignment_entropy_mean'] for d in e9_diag]):.4f}±{np.std([d['test_alignment_entropy_mean'] for d in e9_diag]):.4f}；这表示拒绝通道真实被使用，但也说明当前稀疏映射下大多数测试患者证据不足，不能包装成广泛疾病语义覆盖。E9与站点、年龄箱和质量代理的平均η²/秩相关分别为{np.mean([d['alignment_site_eta2'] for d in e9_diag]):.4f}、{np.mean([d['alignment_agebin_eta2'] for d in e9_diag]):.4f}和{np.mean([d['alignment_qc_spearman'] for d in e9_diag]):.4f}；这些关联是诊断性统计，不代表医学亚型。", "",
        f"E9最佳epoch为0的fold数为{sum(int(r['best_epoch']) == 0 for r in e9_rows)}/10；其余fold也没有形成超过B0-A的分类改善。非零梯度和固定checkpoint敏感性只能说明分支可训练/可扰动，不能证明真实医学先验必要。", "",
        "E5多尺度数值组、E7错误对应和E8训练折数值先验均与E9得到完全相同的四项十折均值；E9相对它们的ACC/Hard-AUC逐折全为平局。结合E9相对E0也完全平局，本轮没有证明医学语义超过数值先验、错误对应或单纯基座。", ""]
    lines += ["### 数学与优化原因", "",
        f"E9最佳checkpoint的测试患者中，平滑残差剂量达到上限99%以上的比例为{np.mean([d['smooth_cap_fraction_above_99pct'] for d in e9_diag]):.4f}；原始残差/5%预算之比的逐折中位数均值为{np.mean([d['raw_residual_over_cap_median'] for d in e9_diag]):.2f}。tanh并未消除剂量饱和，说明固定历史学习率与输入尺度对极小残差预算偏大；没有在看到测试表现后调学习率或重缩放。", "",
        "零初始化输出层使第一步只有输出层获得分类梯度；后续输入层获得有效梯度，精确恢复预检已通过。但5%受限分支、冻结的原分类器和基座本已较高的判别性能，使概率/残差变化不足以跨越多数硬标签边界。epoch=0表示第一步优化后的checkpoint，绝非偷偷恢复未训练的B0-A；严格ACC相同不更新，故9折停留在首步，保留了非零残差。", "",
        "语义支持在稀疏RFE连接上不足：多数证据质量流入NULL。校准阈值采用训练TD的幅度+最大方向覆盖，而匹配分数使用平均方向覆盖加弱方向项；这两个固定统计并不相同，可能增加保守拒绝。ROI最多12条连接、19个ROI仅1–2条，也限制了区域级收敛证据。不能把高拒绝率自动解释为临床可靠，也不能把非零匹配解释为真实疾病机制。", "",
        "共享Adapter同时读取CAM与数值证据，仍存在依赖非语义输入的路径。固定医学门控约束了语义特征，但没有保证分类器必须依赖它；这正需要E5/E7/E8和SEMANTIC_OFF对照。本轮这些对照没有可测的医学知识必要性增益。站点/年龄η²受类别数量与小测试折影响，QC相关部分由可靠性公式构造，均不是亚型或病理证据。", ""]
    lines += ["", "## 决定与限制", ""]
    conditions = []
    for group in ("E0_B0A", "E5_MULTISCALE_NUM", "E7_WRONG_SEMANTICS", "E8_DATA_PRIOR"):
        p = paired_results[group]
        conditions.append({"comparison": group,
                           "acc_mean_gain_gt_0_2pp": p["acc"]["mean_delta_pp"] > .2,
                           "auc_mean_gain_gt_0_2pp": p["hard_auc"]["mean_delta_pp"] > .2,
                           "acc_wins_at_least_7": p["acc"]["wins"] >= 7,
                           "auc_wins_at_least_7": p["hard_auc"]["wins"] >= 7})
    sen_ok = paired_results["E0_B0A"]["sen_asd"]["mean_delta_pp"] >= -.5
    go = all(all(v[k] for k in ("acc_mean_gain_gt_0_2pp", "auc_mean_gain_gt_0_2pp",
                                "acc_wins_at_least_7", "auc_wins_at_least_7"))
             for v in conditions) and sen_ok
    lines += [f"预注册内部筛选结果：**{'GO（仅内部探索候选）' if go else 'NO-GO'}**。判定还需结合真实医学先验相对错误映射和数值先验、拒绝质量及响应诊断；单seed GO不等于论文级成功。逐项条件：`{json.dumps(conditions, ensure_ascii=False)}`。E9对E0 SEN-ASD变化为{paired_results['E0_B0A']['sen_asd']['mean_delta_pp']:+.4f}pp；灵敏度非劣阈值为−0.5pp。", "",
        "既有方法中，FUNCOIN/规范模型的完整系统分析基于更密集连接和更大样本；PhenoBrain已研究表型条件化网络路由；KD-Brain已有语义条件子网络交互；BrainPrompt/BrainPrompt+已有多层提示；AAAI 2026对齐研究支持选择性/部分对齐。UCSMA-v2最多新增“稀疏ROI真实覆盖约束下的训练折条件TD偏离、解剖ROI聚合与可拒绝结构语义质量”，不能主张条件化、Prompt或医学注意力本身首创。", "",
        "| 相关研究 | 已有内容 | 本轮具体区别与边界 |", "|---|---|---|",
        "| Ilioska等，Nature Mental Health 2026 | 高密度连接的个体异质性与区域/网络尺度群体收敛 | 本轮是252条稀疏RFE边的分类增量验证，不复现完整网络或个体病灶结论 |",
        "| FUNCOIN，Progress in Neurobiology 2026 | 全脑系统层规范模型；大规模健康参考；神经/精神疾病偏离 | 本轮仅约421名outer-train TD的岭参考与解剖代理，不声称复现其临床分组 |",
        "| PhenoBrain，ICML 2026 | 表型条件化长程通信与注意调节 | 本轮表型只进入TD参考和支持估计；患者图保持原GFR |",
        "| KD-Brain，MICCAI 2026 | 语义条件Query及病理先验分布约束 | 本轮不用自由语义Query或LLM概率真值，采用冻结解剖候选及经验拒绝 |",
        "| BrainPrompt / BrainPrompt+ | ROI/患者/疾病提示；后者还含频谱、空间提示 | 本轮无BOLD频谱或全连接组，不把多层Prompt本身作为首创 |",
        "| AAAI 2026 To Align or Not to Align | 显式对齐强度与模态冗余关系的研究 | 仅提供避免强制共享空间的方法动机，不是本轮医学拒绝规则的临床验证 |", "",
        "本轮完整创新链在最后两环失败：条件参考和拒绝确实可计算，真实医学结构也能改变内部响应，但未产生超过数值/错误先验的判别互补性与独立收益。因此当前实现不值得作为MICCAI 2027第一创新点继续主推；本轮不联合微调、不追加seed、不调参寻找最好结果，也不启动实验42。", "",
        "继承协议仍有`val_idx=test_idx`的测试集checkpoint选择、RFE/标准化先于外层CV，以及871人transductive患者图；10折训练样本重叠，且仅一个seed。故结果是历史设置下的内部比较，不是无偏泛化、独立中心或跨seed稳定性证据。", "",
        f"数据/源码SHA256及E0逐折复现记录见`manifest.json`；100折正式结果见`fold_metrics.csv`，机制诊断见`diagnostics.csv`，固定checkpoint干预见`counterfactual.csv`。本实验产物目录为`{EXP}`。", ""]
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return go


def report():
    rows, diagnostics = [], []
    for group in CONFIGS:
        group_rows, group_diag = task_rows(group)
        rows.extend(group_rows)
        diagnostics.extend(group_diag)
    rows.sort(key=lambda r: (CONFIGS.index(r["config"]), int(r["fold"])))
    diagnostics.sort(key=lambda r: (r["config"], int(r["fold"])))
    if {(r["config"], int(r["fold"])) for r in rows} != {
            (g, f) for g in CONFIGS for f in range(1, 11)} or len(rows) != 100:
        raise AssertionError("Formal table must contain exactly 100 unique valid folds")
    audits = posthoc_audits()
    by_count = {(v["config"], int(v["fold"])): v for v in audits["branch_counts"]}
    by_norm = {(v["mode"], int(v["fold"])): v for v in audits["normative"]}
    for row in rows:
        if row["config"] == "E0_B0A":
            continue
        count = by_count[(row["config"], int(row["fold"]))]
        row.update({k: v for k, v in count.items() if k not in ("config", "fold")})
        row["training_gradient_count_scope"] = ("nonzero_tensor_parameter_count_legacy"
            if row["config"] in ("E1_RAW_CAPACITY", "E2_GLOBAL_NORM")
            else "nonzero_gradient_elements_last_training_step")
    for diag in diagnostics:
        diag.update(by_count[(diag["config"], int(diag["fold"]))])
        norm = by_norm[(diag["normative_mode"], int(diag["fold"]))]
        diag.update({k: norm[k] for k in ("td_oof_mse", "td_oof_mae", "scale_floor_columns")})
        diag["gradient_params_nonzero"] = diag["best_checkpoint_nonzero_gradient_parameters"]
    csv_write(RESULTS / "fold_metrics.csv", rows)
    csv_write(RESULTS / "diagnostics.csv", diagnostics)
    counter = counterfactuals()
    csv_write(RESULTS / "counterfactual.csv", counter)
    summary = summarize(rows)
    paired_results = paired(rows)
    json_write(RESULTS / "paired_comparisons.json", paired_results)
    json_write(RESULTS / "summary.json", summary)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["posthoc_audit"] = audits
    e0_state = torch_load(task_path("E0_B0A"))
    e0_parity = verify_e0(e0_state["rows"])
    manifest["e0_history_verification"] = e0_parity
    manifest["workers"]["E0_B0A"] = dict(
        status="COMPLETE", gpu=0, folds_completed=10,
        seconds=float(e0_state.get("seconds", 0.0)), completed_utc=utc())
    manifest["e0_wall_seconds"] = float(e0_state.get("seconds", 0.0))
    if "auxiliary_wall_seconds" not in manifest:
        manifest["auxiliary_wall_seconds"] = float(manifest.get("training_wall_seconds", 0.0))
    manifest["training_total_wall_seconds"] = (
        manifest["e0_wall_seconds"] + manifest["auxiliary_wall_seconds"])
    manifest["completed_folds"] = len(rows)
    json_write(MANIFEST_PATH, manifest)
    manifest["source_and_input"]["final_core_sha256"] = source_hashes()
    manifest["source_and_input"]["final_history_sha256"] = file_hashes([ROOT / p for p in HISTORY_FILES])
    manifest["source_and_input"]["final_data_sha256"] = file_hashes([ROOT / p for p in DATA_FILES])
    manifest["source_and_input"]["final_historical_reports_sha256"] = file_hashes(
        [Path(p) for p in manifest["source_and_input"]["historical_reports_sha256"]])
    for original, final in (("core_sha256", "final_core_sha256"),
                            ("data_sha256", "final_data_sha256"),
                            ("history_sha256", "final_history_sha256"),
                            ("historical_reports_sha256", "final_historical_reports_sha256")):
        if manifest["source_and_input"][original] != manifest["source_and_input"][final]:
            raise AssertionError(f"Protected source/input hash changed: {original}")
    manifest["source_and_input"]["experiment_code_sha256"] = code_hashes()
    manifest["result_sha256"] = file_hashes([RESULTS / p for p in
        ("fold_metrics.csv", "diagnostics.csv", "counterfactual.csv",
         "paired_comparisons.json", "summary.json")])
    manifest["status"] = "COMPLETE"
    manifest["report_only_additions_after_training"] = (
        "read-only layer/gradient/feature/counterfactual audits, tables and correction disclosure; no trainable-model, evidence, optimizer, prompt or ontology change")
    go = make_report(rows, diagnostics, counter, summary, paired_results, manifest)
    manifest["scientific_decision"] = "GO_INTERNAL_ONLY" if go else "NO-GO"
    manifest["completed_folds"] = len(rows)
    manifest["counterfactual_rows"] = len(counter)
    manifest["report_sha256"] = sha_file(RESULTS / "report.md")
    manifest["completed_utc"] = utc()
    json_write(MANIFEST_PATH, manifest)
    print(f"REPORT_COMPLETE folds={len(rows)} decision={manifest['scientific_decision']}")


def finalize():
    """Verify, sync this experiment, then append one immutable handoff section."""
    import shutil
    m = json.loads(MANIFEST_PATH.read_text())
    if m.get("status") != "COMPLETE" or m.get("completed_folds") != 100:
        raise RuntimeError("Finalize requires 100 valid completed folds and a report")
    if not m.get("e0_history_verification", {}).get("pass"):
        raise AssertionError("E0 parity must pass before delivery")
    for name, current in (("core_sha256", source_hashes()),
                          ("data_sha256", file_hashes([ROOT / p for p in DATA_FILES]))):
        if current != m["source_and_input"][name]:
            raise AssertionError(f"Protected {name} differs at delivery")
    for name in ("history_sha256", "historical_reports_sha256"):
        old = m["source_and_input"][name]
        if old != file_hashes([Path(p) for p in old]):
            raise AssertionError(f"Read-only historical input changed: {name}")
    if sha_file(RESULTS / "report.md") != m["report_sha256"]:
        raise AssertionError("Final report digest changed")
    target = ROOT / "experiment" / EXP.name
    summary = json.loads((RESULTS / "summary.json").read_text())
    m["delivery"] = dict(stage=str(EXP), target=str(target), runtime_bytes=sum(
        p.stat().st_size for p in RUNTIME.rglob("*") if p.is_file() and not p.is_symlink()),
        sync_status="PENDING", handoff_status="PENDING", utc=utc())
    m["source_and_input"]["clip_cache_sha256"] = file_hashes([Path.home()/".cache/clip/RN50.pt"])
    m["source_and_input"]["clip_python_source_sha256"] = file_hashes(sorted((ROOT/"CLIP").rglob("*.py")))
    m["source_and_input"]["experiment_code_sha256"] = code_hashes()
    m["artifact_sha256"] = file_hashes([p for p in EXP.rglob("*")
        if p.is_file() and not p.is_symlink() and ".runtime" not in p.parts
        and "__pycache__" not in p.parts and p.name not in ("manifest.json", "handoff_pending.md")])
    json_write(MANIFEST_PATH, m)

    def sync():
        target.mkdir(parents=True, exist_ok=True)
        copied = checked = 0
        for src in sorted(EXP.rglob("*")):
            rel = src.relative_to(EXP)
            if "__pycache__" in rel.parts or src.suffix in (".lock", ".tmp"):
                continue
            dst = target / rel
            if src.is_symlink():
                if dst.is_symlink() and os.readlink(dst) == os.readlink(src):
                    continue
                if dst.exists() or dst.is_symlink():
                    raise RuntimeError(f"Unexpected delivery symlink conflict: {dst}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(os.readlink(src), target_is_directory=src.is_dir())
            elif src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            elif src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                before = sha_file(src)
                if not dst.is_file() or sha_file(dst) != before:
                    shutil.copy2(src, dst)
                    copied += 1
                if sha_file(dst) != before:
                    raise OSError(f"Post-copy digest mismatch: {dst}")
                checked += 1
        return dict(files_copied=copied, files_verified=checked)

    errors = []
    try:
        sync_stats = sync()
        m["delivery"].update(sync_status="VERIFIED", **sync_stats)
    except Exception as exc:
        errors.append(f"sync: {type(exc).__name__}: {exc}")
        m["delivery"].update(sync_status="FAILED", sync_error=errors[-1])
    now = datetime.now(timezone.utc)
    start = datetime.fromisoformat(torch_load(task_path("E0_B0A"))["started_utc"])
    m["delivery"]["elapsed_since_e0_start_seconds_including_interruptions"] = (now-start).total_seconds()
    diagnostics = list(csv.DictReader((RESULTS / "diagnostics.csv").open()))
    e9 = [d for d in diagnostics if d["config"] == "E9_UCSMA_FULL"]
    header = "## Experiment 41 — UCSMA-v2"
    begin = "<!-- experiment41_ucsma_v2_completion_v1 -->"
    end = "<!-- /experiment41_ucsma_v2_completion_v1 -->"
    body = [header, "", begin, "", f"验收时间：{utc()}。计算状态：**100/100有效正式折完成；NO-GO**。", "",
        "E0独立重训，十折四指标、最佳epoch、执行epoch、初始参数SHA256及fold入口RNG全部精确匹配Experiment32 B0-A；另逐层核对原E32 B0 forward，GFR、四图邻接、CAM、attention与输出均10/10逐位一致。PEN保留构造但完全旁路前向。", "",
        "有效正式任务：GPU0 E0/E2/E4/E6/E8=50折；GPU1 E1/E3/E5/E7/E9=50折；每卡最多4 worker，CPU/BLAS线程1。设备实际为两张CUDA兼容PPU-ZW810E，各98,304MiB，未修改其他任务或杀进程。", "",
        f"训练调度累计墙钟{m['all_computation_training_wall_seconds']:.2f}秒（E0 {m['e0_wall_seconds']:.2f}秒、初次增量{m['initial_auxiliary_wall_seconds']:.2f}秒、更正调度{m['corrected_auxiliary_wall_seconds']:.2f}秒）；不含审计/报告。自E0开始至验收{(now-start).total_seconds():.2f}秒，包含人工中断与等待，不等于连续计算时间。有效worker耗时合计{sum(w.get('seconds',0) for w in m['workers'].values()):.2f}秒；runtime约{m['delivery']['runtime_bytes']/2**20:.2f}MiB。", "",
        "最终复核修复了条件参考mean FD索引（EFC→全局60列）、概念拒绝门控及ROI覆盖支持传递。初次E3–E9共70折作废并排除；更正预检后重算，E0/E1/E2保留，E2输入逐位等价。总执行170次fold拟合，其中仅100次进入正式结果；GPU0额外作废30折、GPU1额外作废40折。Prompt、先验、超参、GO规则均未按测试成绩更改。排除记录及旧哈希见results/correction_audit.json。", "",
        "四指标：十折均值±population SD（%），AUC为硬标签one-hot ROC-AUC，二分类等于BA。", "",
        "| 组别 | ACC | Hard-AUC | SPE-TD | SEN-ASD |", "|---|---:|---:|---:|---:|"]
    for g in CONFIGS:
        body.append("| " + g + " | " + " | ".join(fmt_percent(summary[g][k])
            for k in ("acc", "hard_auc", "spe_td", "sen_asd")) + " |")
    body += ["", f"核心机制：条件参考TD OOF MSE由0.049654降至0.045132，但E9−E0/E5/E7/E8的ACC及Hard-AUC均为0.0000pp，逐折0胜/10平/0负；ASD灵敏度无增益。E9平均未匹配质量{np.mean([float(d['test_unmatched_mean']) for d in e9]):.4f}；正确/错误医学映射、关闭语义、全部残差关闭均未改变硬标签，患者错配仅改变2个外层测试预测。真实医学语义没有显示超过多尺度数值、错误或数值先验的独立增益。", "",
        "新增组注册参数182016；有效输入连接参数E1=182016，E2/E3=116736，E4=158976，E5=162048，E6=163840，E7/E8/E9=163904；元素级梯度范围和剂量饱和见报告。E7不精确匹配重叠/覆盖，E8簇规模不同，不能宣称严格等容量或等覆盖。", "",
        "结论：**NO-GO，不作为当前MICCAI 2027第一创新点主推**。不把内部单seed结果当作跨seed、独立中心或临床泛化；继承val=test选模、CV前RFE/标准化与871人transductive图限制。下一步仅建议重新评估问题与数据覆盖，不在本轮调参、联合微调、追加seed或启动实验42。", "",
        f"计算目录：`{EXP}`；指定交付目录：`{target}`（同步状态{m['delivery']['sync_status']}）；中文报告：`{target / 'results/report.md' if m['delivery']['sync_status']=='VERIFIED' else RESULTS / 'report.md'}`。", "",
        f"报告SHA256：`{m['report_sha256']}`；run.py：`{sha_file(EXP/'run.py')}`；ucsma.py：`{sha_file(EXP/'ucsma.py')}`；Prompt：`{sha_file(EXP/'prompts.json')}`；ontology：`{sha_file(EXP/'ontology.json')}`。", "",
        "原始核心源码、数据、原论文、历史结果及CLIP缓存SHA256详见manifest；交付前复核受保护核心源码、两份数据及所有已读历史报告均未变化。processed_standard_data.csv SHA256=`b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1`；modal_feat_dict.npy SHA256=`8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e`。", "",
        f"I/O与恢复：计算阶段无OSSFS I/O故障；双Adam/RNG及增量Adam/RNG下一步重放预检均通过。最终同步：{m['delivery']['sync_status']}；无自动Git提交/推送。" + (" 错误：" + "; ".join(errors) if errors else ""), "", end, ""]
    text_body = "\n\n" + "\n".join(body)
    pending = EXP / "handoff_pending.md"
    pending.write_text(text_body, encoding="utf-8")
    try:
        handoff = ROOT / "handoff.md"
        if not handoff.is_file():
            raise FileNotFoundError("Existing handoff.md required; refusing to replace history")
        with handoff.open("r+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            before = stream.read()
            if begin.encode() in before:
                if end.encode() not in before:
                    raise RuntimeError("Incomplete earlier handoff append requires inspection")
                existing = before.split(begin.encode(), 1)[1].split(end.encode(), 1)[0]
                if m["report_sha256"].encode() not in existing:
                    raise RuntimeError("Existing completion section refers to a different report; refusing duplicate")
                m["delivery"]["handoff_status"] = "ALREADY_PRESENT"
                m["delivery"]["handoff_after_sha256"] = sha_bytes(before)
            else:
                stream.seek(0, 2)
                stream.write(text_body.encode())
                stream.flush()
                os.fsync(stream.fileno())
                stream.seek(0)
                after = stream.read()
                if not after.startswith(before) or after.count(begin.encode()) != 1 or after.count(end.encode()) != 1:
                    raise OSError("Handoff append verification failed")
                m["delivery"].update(handoff_status="APPENDED_VERIFIED",
                    handoff_before_sha256=sha_bytes(before), handoff_after_sha256=sha_bytes(after),
                    handoff_original_prefix_preserved=True)
        pending.unlink()
    except Exception as exc:
        errors.append(f"handoff: {type(exc).__name__}: {exc}")
        m["delivery"].update(handoff_status="FAILED_PENDING_SAVED", handoff_error=errors[-1])
    m["delivery"]["errors"] = errors
    m["delivery"]["finalized_utc"] = utc()
    json_write(MANIFEST_PATH, m)
    if m["delivery"]["sync_status"] == "VERIFIED":
        try:
            shutil.copy2(MANIFEST_PATH, target/"manifest.json")
            if sha_file(target/"manifest.json") != sha_file(MANIFEST_PATH):
                raise OSError("Final manifest sync digest mismatch")
        except Exception as exc:
            m["delivery"].update(sync_status="FINAL_MANIFEST_FAILED", sync_error=str(exc))
            json_write(MANIFEST_PATH, m)
    print(json.dumps(m["delivery"], indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "train", "dispatch", "resume", "report", "finalize",
                                             "_worker"))
    parser.add_argument("group", nargs="?")
    parser.add_argument("gpu", nargs="?", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "preflight":
        preflight(args.gpu)
    elif args.command in ("train", "dispatch", "resume"):
        train()
    elif args.command == "report":
        report()
    elif args.command == "finalize":
        finalize()
    elif args.command == "_worker":
        if args.group is None:
            raise SystemExit("worker requires group and GPU")
        worker_wrapper(args.group, args.gpu)


if __name__ == "__main__":
    main()
