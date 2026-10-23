#!/usr/bin/env python3
"""Prepare, preflight, run and audit the 120-fold SeRCA-v2 experiment."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import datetime as dt
import difflib
import fcntl
import hashlib
import importlib
import io
import json
import os
import pickle
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path("/data/adodas/SPromptGL-main")
EXP = Path(__file__).resolve().parent
RUNTIME = EXP / ".runtime"
SOURCE = RUNTIME / "source"
PRIOR_DIR = RUNTIME / "prior"
RNG_DIR = RUNTIME / "rng"
STATE_DIR = RUNTIME / "state"
LOG_DIR = RUNTIME / "logs"
TMP_DIR = RUNTIME / "tmp"
RESULTS = EXP / "results"
MANIFEST = EXP / "manifest.json"
PRIOR_FILE = EXP / "medical_prior.json"
CORE_FILES = ("main.py", "model.py", "network.py", "layers.py", "utils.py",
              "ABIDE-simple-2-concat-weighted-cosine.sh")
SOURCE_NAMES = CORE_FILES[:5]
DATA_FILES = ("data/ABIDE/processed_standard_data.csv", "data/ABIDE/modal_feat_dict.npy")
EXP11 = ROOT / "experiment/experiment11_roi_mapping_recovery/results"
EXP19 = ROOT / "experiment/experiment19_contrastive_disease_evidence_alignment"
EXP32 = ROOT / "experiment/experiment32_disease_contrastive_evidence_routing"
EXP33 = ROOT / "experiment/experiment33_qcar_conditional_alignment"
EXP34 = Path("/tmp/SPromptGL-E34-SeRCA-stage-20261022/experiment/experiment34_serca_semantic_relational_alignment")
AUDIT = ROOT / "experiment/b0_pen_physical_removal_equivalence_audit"
CONFIGS = (
    "E0_B0A", "E1_SERCA34_CAL", "E2_NUMERIC_GLOBAL", "E3_FIRST_ORDER",
    "E4_ALL_PAIR", "E5_RANDOM_RULE", "E6_PATIENT_SHUFFLED",
    "E7_ORIGINAL_EDGES", "E8_CORR_ONLY", "E9_AUX_ONLY",
    "E10_DIRECT_MATCHED", "E11_SERCA_V2_FULL",
)
GPU_CONFIGS = {
    0: ("E0_B0A", "E2_NUMERIC_GLOBAL", "E4_ALL_PAIR", "E6_PATIENT_SHUFFLED",
        "E8_CORR_ONLY", "E10_DIRECT_MATCHED"),
    1: ("E1_SERCA34_CAL", "E3_FIRST_ORDER", "E5_RANDOM_RULE", "E7_ORIGINAL_EDGES",
        "E9_AUX_ONLY", "E11_SERCA_V2_FULL"),
}
GPU_OF = {name: gpu for gpu, configs in GPU_CONFIGS.items() for name in configs}
INPUT_DIMS = (48, 6, 10, 256)
RELATION_NAMES = ("PHENO", "ANAT", "FUNC", "CORR")
PROTOCOL = dict(
    datadir=str(ROOT / "data") + "/", datname="ABIDE", cpu=False, gpu="-1",
    nepoch=450, early=200, lr=.0038, reg=.11, dropout=.35, nlayer=1,
    n_hidden=18, n_head=2, n_iter=10, nmodal=4, th=.9,
    GC_mode="weighted-cosine", MP_mode="GCN", MF_mode="concat",
    alpha=.5, theta_smooth=1., theta_degree=.5, theta_sparsity=0.,
    nclass=2, mode="simple-2", seed=0,
)
METRICS = ("acc", "hard_auc", "sen_asd", "spe_td")
CONCEPT_NAMES = (
    "Social/temporal-temporoparietal connectivity",
    "Frontoparietal/executive connectivity",
    "Thalamocortical/sensorimotor connectivity",
    "Cingulate-insular/salience connectivity",
)
RANDOM_RULE_SEED = 350500
PATIENT_PERMUTATION_SEED = 350600
CHECKPOINT_EVERY = 10
WORKERS_PER_GPU = 4
SEN_NONINFERIORITY_PP = .50
_CONTEXT = None

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLCONFIGDIR"] = str(RUNTIME / "mpl")
os.environ["XDG_CACHE_HOME"] = str(RUNTIME / "xdg-cache")
os.environ["TMPDIR"] = str(TMP_DIR)

import numpy as np


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return sha_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                allow_nan=False).encode("utf-8"))


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def csv_write(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def pickle_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=4)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def torch_write(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def torch_read(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def record(message):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "scheduler.log").open("a", encoding="utf-8") as stream:
        stream.write(utc() + " " + str(message) + "\n")
        stream.flush()
    print(message, flush=True)


def input_hashes():
    return {name: sha_file(ROOT / name) for name in DATA_FILES}


def source_hashes():
    return {name: sha_file(ROOT / name) for name in CORE_FILES}


def history_hashes():
    paths = {
        "experiment11_roi_256_mapping": EXP11 / "roi_256_mapping.csv",
        "experiment11_roi_256_named_mapping": EXP11 / "roi_256_named_mapping.csv",
        "experiment11_report": EXP11 / "report.md",
        "experiment19_domain_mapping": EXP19 / "domain_mapping.csv",
        "experiment19_prompts": EXP19 / "prompts.json",
        "experiment19_report": EXP19 / "results/report.md",
        "experiment32_manifest": EXP32 / "manifest.json",
        "experiment32_fold_metrics": EXP32 / "results/fold_metrics.csv",
        "experiment32_report": EXP32 / "results/report.md",
        "experiment33_manifest": EXP33 / "manifest.json",
        "experiment33_fold_metrics": EXP33 / "results/fold_metrics.csv",
        "experiment33_report": EXP33 / "results/report.md",
        "experiment34_manifest": EXP34 / "manifest.json",
        "experiment34_fold_metrics": EXP34 / "results/fold_metrics.csv",
        "experiment34_report": EXP34 / "results/report.md",
        "b0_b_equivalence_report": AUDIT / "report.md",
    }
    return {key: sha_file(path) for key, path in paths.items()}


def read_data():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    base = ROOT / "data/ABIDE"
    frame = pd.read_csv(base / "processed_standard_data.csv")
    mapping = np.load(base / "modal_feat_dict.npy", allow_pickle=True).item()
    assert list(mapping) == ["PHENO", "ANAT", "FUNC", "Correlation"]
    assert tuple(len(mapping[key]) for key in mapping) == INPUT_DIMS
    columns = [str(name) for group in mapping for name in mapping[group]]
    assert list(frame.columns[:-1]) == columns and frame.shape == (871, 321)
    pheno_names = [str(name) for name in mapping["PHENO"]]
    assert all(name.startswith("AGE_AT_SCAN_") for name in pheno_names[:26])
    assert all(name.startswith("SITE_ID_") for name in pheno_names[26:46])
    assert all(name.startswith("SEX_") for name in pheno_names[46:48])
    x = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = frame.iloc[:, -1].to_numpy(dtype=np.int64) - 1
    assert x.shape == (871, 320) and np.isfinite(x).all()
    assert np.bincount(y).tolist() == [403, 468]
    folds = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y))
    assert len(folds) == 10
    for train, test in folds:
        assert len(set(train) & set(test)) == 0
        assert len(set(train) | set(test)) == 871
    return x, y, folds, mapping


def patch_network(source):
    assert source.count("    def forward(self, x):\n        bs = x.size(0)\n") == 1
    assert source.count("        x, _attn = self.InputLayer(x)\n") == 1
    assert source.count("        text_features = self.prompt_text_features\n") == 1
    assert source.count("        all_adj = []\n") == 1
    assert source.count("            all_adj.append(adj)\n") == 1
    for relation in range(4):
        assert source.count(f"                normalized_adj{relation} = normalize_adj(") == 1
    modified = source.replace(
        "    def forward(self, x):\n        bs = x.size(0)\n",
        "    def forward(self, x, serca_payload=None):\n        bs = x.size(0)\n", 1,
    )
    modified = modified.replace(
        "        x, _attn = self.InputLayer(x)\n",
        "        x, _attn = self.InputLayer(x)\n"
        "        if serca_payload is not None:\n"
        "            x = self.serca35(x, serca_payload)\n", 1,
    )
    bypass_marker = "        text_features = self.prompt_text_features\n"
    bypass = (
        "        if getattr(self, 'serca35_bypass_pen', False):\n"
        "            x = x.view(bs, -1)\n"
        "            attn_embedding = attn.view(bs, -1)\n"
        "            output, hidden = self.Outputlayer(x, attn_embedding)\n"
        "            return output, hidden, attn_map\n"
    )
    modified = modified.replace(bypass_marker, bypass + bypass_marker, 1)
    modified = modified.replace("        all_adj = []\n", "        all_adj = []\n        all_normalized_adjs = []\n", 1)
    for relation in range(4):
        marker = f"                normalized_adj{relation} = normalize_adj("
        start = modified.index(marker)
        end = modified.index("\n", start)
        modified = modified[:end + 1] + f"                normalized_adj_current = normalized_adj{relation}\n" + modified[end + 1:]
    modified = modified.replace(
        "            all_adj.append(adj)\n",
        "            all_adj.append(adj)\n            all_normalized_adjs.append(normalized_adj_current)\n", 1,
    )
    modified = modified.replace(
        "        return combined_feat , all_adj, ALoss/model_num\n",
        "        if self.training:\n"
        "            self.last_raw_adjs = all_adj\n"
        "            self.last_normalized_adjs = all_normalized_adjs\n"
        "        else:\n"
        "            self.last_raw_adjs = [value.detach() for value in all_adj]\n"
        "            self.last_normalized_adjs = [value.detach() for value in all_normalized_adjs]\n"
        "        return combined_feat , all_adj, ALoss/model_num\n", 1,
    )
    assert modified.count("serca_payload=None") == 1
    compile(modified, "network.py", "exec")
    return modified


def patch_model(source):
    assert source.count("from network import *\n") == 1
    assert source.count("        self.ModalFusion.apply(my_weight_init)\n") == 1
    assert source.count("            prob,_,_ = self.ModalFusion(feat)\n") == 1
    assert source.count("            prob, fusion_feat, attn = self.ModalFusion(feat)\n") == 1
    modified = source.replace(
        "from network import *\n", "from network import *\nimport integration\n", 1,
    )
    modified = modified.replace(
        "        self.ModalFusion.apply(my_weight_init)\n",
        "        self.ModalFusion.apply(my_weight_init)\n        integration.install(self)\n", 1,
    )
    modified = modified.replace(
        "            prob,_,_ = self.ModalFusion(feat)\n",
        "            prob,_,_ = integration.forward_modal(self, feat)\n",
    )
    modified = modified.replace(
        "            prob, fusion_feat, attn = self.ModalFusion(feat)\n",
        "            prob, fusion_feat, attn = integration.forward_modal(self, feat)\n", 1,
    )
    assert "integration.forward_modal(self, feat)" in modified
    compile(modified, "model.py", "exec")
    return modified


def emit_sources():
    SOURCE.mkdir(parents=True, exist_ok=True)
    can_refresh = not MANIFEST.exists()
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        can_refresh = existing.get("status") == "PREPARED" and not list(STATE_DIR.glob("*.pt")) \
            and not list(RNG_DIR.glob("fold_*.pkl"))
    patch_lines = []
    for name in SOURCE_NAMES:
        original = (ROOT / name).read_text(encoding="utf-8")
        if name == "network.py":
            modified = patch_network(original)
        elif name == "model.py":
            modified = patch_model(original)
        else:
            modified = original
        target = SOURCE / name
        if target.exists():
            if target.read_text(encoding="utf-8") != modified:
                assert can_refresh, f"isolated source drift after training started: {name}"
                target.write_text(modified, encoding="utf-8")
        else:
            target.write_text(modified, encoding="utf-8")
        patch_lines.extend(difflib.unified_diff(
            original.splitlines(True), modified.splitlines(True),
            fromfile="a/" + name, tofile="b/" + name,
        ))
    clip = SOURCE / "CLIP"
    if not clip.exists():
        clip.symlink_to(ROOT / "CLIP", target_is_directory=True)
    patch = "".join(patch_lines)
    patch_path = EXP / "integration.patch"
    if patch_path.exists():
        if patch_path.read_text(encoding="utf-8") != patch:
            assert can_refresh, "integration patch changed after training started"
            patch_path.write_text(patch, encoding="utf-8")
    else:
        patch_path.write_text(patch, encoding="utf-8")
    return patch


def build_concepts():
    from serca_v2 import DOMAIN_PAIRS
    prior = json.loads(PRIOR_FILE.read_text(encoding="utf-8"))
    with (EXP19 / "domain_mapping.csv").open(newline="", encoding="utf-8") as stream:
        domain_rows = list(csv.DictReader(stream))
    with (EXP11 / "roi_256_mapping.csv").open(newline="", encoding="utf-8") as stream:
        numeric = {int(row["correlation_feature_index"]): row for row in csv.DictReader(stream)}
    with (EXP11 / "roi_256_named_mapping.csv").open(newline="", encoding="utf-8") as stream:
        named = {int(row["correlation_feature_index"]): row for row in csv.DictReader(stream)}
    assert set(numeric) == set(named) == set(range(256))
    features = {f"D{i}": [] for i in range(1, 5)}
    unknown = sorted(set(range(256)) - {int(row["correlation_feature_index"]) for row in domain_rows})
    assert unknown == [195, 225, 232, 236]
    assert len(domain_rows) == 252 and all(row["verified"].lower() == "true" for row in domain_rows)
    connections = {}
    for row in domain_rows:
        index = int(row["correlation_feature_index"])
        assert numeric[index]["verification_status"] == "VERIFIED_NUMERIC_UNIQUE"
        assert named[index]["connection_anatomical_status"] == "VERIFIED_BOTH_ENDPOINTS"
        assert named[index]["usable_for_roi_semantic_prompt"].lower() == "true"
        assert named[index]["roi_i_name_verified"].lower() == "true"
        assert named[index]["roi_j_name_verified"].lower() == "true"
        assert int(row["original_connection_index"]) == int(numeric[index]["original_connection_index"])
        memberships = []
        for domain in features:
            if row[domain] == "1":
                features[domain].append(index)
                memberships.append(domain)
        connections[str(index)] = dict(
            original_connection_index=int(row["original_connection_index"]),
            roi_i_name=named[index]["roi_i_name_verified"],
            roi_j_name=named[index]["roi_j_name_verified"],
            connection_name=named[index]["connection_name"],
            concepts=memberships,
        )
    assert [len(features[f"D{i}"]) for i in range(1, 5)] == [72, 56, 45, 32]
    overlap = {f"D{a+1}-D{b+1}": len(set(features[f"D{a+1}"]) & set(features[f"D{b+1}"]))
               for a, b in DOMAIN_PAIRS}
    prior["feature_mapping"].update(
        source_sha256={
            "prompts.json": sha_file(EXP19 / "prompts.json"),
            "domain_mapping.csv": sha_file(EXP19 / "domain_mapping.csv"),
            "roi_256_mapping.csv": sha_file(EXP11 / "roi_256_mapping.csv"),
            "roi_256_named_mapping.csv": sha_file(EXP11 / "roi_256_named_mapping.csv"),
        },
        unknown_feature_indices_excluded_from_semantic_prior=unknown,
        concepts={f"D{i}": dict(prior["feature_mapping"]["concepts"][f"D{i}"],
                                feature_indices=features[f"D{i}"])
                  for i in range(1, 5)},
        pairwise_support_overlap_counts=overlap,
        verified_connection_records=connections,
    )
    assert [tuple(item["concepts"]) for item in prior["relation_rules"]] == [("D2", "D4"), ("D3", "D4")]
    assert [len(set(features[a]) & set(features[b])) for a, b in (("D2", "D4"), ("D3", "D4"))] == [5, 2]
    json_write(PRIOR_FILE, prior)
    return prior, {key: np.asarray(value, dtype=np.int64) for key, value in features.items()}


def frozen_random_rules():
    import itertools
    from serca_v2 import TRUE_RULES
    possible = list(itertools.combinations(range(4), 2))
    degree_target = sorted([sum(i in edge for edge in TRUE_RULES) for i in range(4)])
    candidates = []
    for graph in itertools.combinations(possible, len(TRUE_RULES)):
        chosen = tuple(sorted(graph))
        if set(chosen) & set(TRUE_RULES):
            continue
        degrees = sorted(sum(i in edge for edge in chosen) for i in range(4))
        if degrees == degree_target:
            candidates.append(chosen)
    rng = np.random.Generator(np.random.PCG64(RANDOM_RULE_SEED))
    return candidates[int(rng.integers(len(candidates)))], candidates


def normative_deviations(x, y, train, fold):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    train = np.asarray(train, dtype=np.int64)
    td = train[np.asarray(y)[train] == 1]
    assert len(td) >= 5 and np.all(np.isin(np.asarray(y)[train], (0, 1)))
    covariates = np.asarray(x[:, :48], dtype=np.float64)
    fc = np.asarray(x[:, 64:], dtype=np.float64)
    oof_prediction = np.empty((len(td), fc.shape[1]), dtype=np.float64)
    splitter = KFold(n_splits=5, shuffle=True, random_state=350000 + fold)
    for fit_pos, hold_pos in splitter.split(td):
        fit_ids, hold_ids = td[fit_pos], td[hold_pos]
        scaler = StandardScaler().fit(covariates[fit_ids])
        model = Ridge(alpha=10.0).fit(scaler.transform(covariates[fit_ids]), fc[fit_ids])
        oof_prediction[hold_pos] = model.predict(scaler.transform(covariates[hold_ids]))
    full_scaler = StandardScaler().fit(covariates[td])
    full_model = Ridge(alpha=10.0).fit(full_scaler.transform(covariates[td]), fc[td])
    full_prediction = full_model.predict(full_scaler.transform(covariates))
    oof_residual = fc[td] - oof_prediction
    sigma = oof_residual.std(axis=0, ddof=1)
    positive = sigma[sigma > 1e-8]
    floor = max(1e-4, 0.10 * float(np.median(positive)) if len(positive) else 1e-4)
    sigma = np.maximum(sigma, floor)
    residual = fc - full_prediction
    train_td_pos = {int(subject): pos for pos, subject in enumerate(td)}
    for subject, pos in train_td_pos.items():
        residual[subject] = oof_residual[pos]
    z = np.clip(residual / sigma[None, :], -3.0, 3.0)
    return z.astype(np.float32), dict(
        td_train_count=int(len(td)), ridge_alpha=10.0, oof_folds=5,
        td_oof_residual_scale_min=float(sigma.min()),
        td_oof_residual_scale_median=float(np.median(sigma)),
        td_oof_residual_scale_max=float(sigma.max()), residual_scale_floor=floor,
        td_oof_z_mean=float(np.mean(oof_residual / sigma[None, :])),
        td_oof_z_std=float(np.std(oof_residual / sigma[None, :])),
        fullfit_test_td_z_mean=float(np.mean(z[np.asarray(y) == 1])) if False else None,
        confounders="PHENO age/site/sex one-hot columns only",
        heldout_labels_used=False,
    )


def first_order_scores(z, supports):
    scores = np.zeros((z.shape[0], 8), dtype=np.float64)
    for domain_index in range(4):
        values = z[:, supports[f"D{domain_index+1}"]]
        scores[:, 2 * domain_index] = np.maximum(values - 1.0, 0.0).mean(axis=1)
        scores[:, 2 * domain_index + 1] = np.maximum(-values - 1.0, 0.0).mean(axis=1)
    return scores


def pair_interactions(first_order, train):
    from serca_v2 import DOMAIN_PAIRS
    train = np.asarray(train, dtype=np.int64)
    design = np.column_stack([np.ones(len(first_order)), first_order])
    values = np.zeros((len(first_order), 24), dtype=np.float64)
    reports = []
    for pair_index, (left, right) in enumerate(DOMAIN_PAIRS):
        for mode, (left_sign, right_sign) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            slot = 4 * pair_index + mode
            raw = first_order[:, 2 * left + left_sign] * first_order[:, 2 * right + right_sign]
            coef, *_ = np.linalg.lstsq(design[train], raw[train], rcond=None)
            residual = raw - design @ coef
            mean = float(residual[train].mean())
            scale = float(residual[train].std())
            raw_scale = float(raw[train].std())
            r2 = (1.0 - scale ** 2 / max(raw_scale ** 2, 1e-12)) if raw_scale > 1e-8 else 1.0
            if scale > 1e-8:
                values[:, slot] = (residual - mean) / scale
            reports.append(dict(
                pair=[left + 1, right + 1], sign_mode=mode,
                raw_train_std=raw_scale, residual_train_std=scale,
                first_order_linear_r2=float(r2), nondegenerate=bool(scale > 1e-8),
            ))
    return values.astype(np.float32), reports


def fit_kernel(signature, mask, train):
    signature = np.asarray(signature, dtype=np.float64)
    active = np.flatnonzero(mask)
    assert signature.ndim == 2 and signature.shape[0] == 871 and len(mask) == signature.shape[1]
    if not len(active):
        raise ValueError("semantic signature has no active dimensions")
    x = signature[:, active]
    mean = x[train].mean(axis=0)
    scale = x[train].std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - mean) / scale
    norm = np.einsum("ij,ij->i", z, z)
    dist2 = np.maximum(norm[:, None] + norm[None, :] - 2.0 * (z @ z.T), 0.0)
    dist2 = (dist2 + dist2.T) * 0.5
    tri = np.triu_indices(len(train), 1)
    train_dist = dist2[np.ix_(train, train)][tri]
    positive = train_dist[train_dist > 1e-12]
    tau = float(np.median(positive)) if len(positive) else 1.0
    kernel = np.exp(-dist2 / tau).astype(np.float32)
    np.fill_diagonal(kernel, 1.0)
    train_kernel = kernel[np.ix_(train, train)][tri]
    singular = np.linalg.svd(z[train], compute_uv=False)
    power = singular ** 2
    participation = float(power.sum() ** 2 / max(np.square(power).sum(), 1e-12))
    return kernel, dict(
        active_dimensions=int(len(active)), signature_train_rank=int(np.linalg.matrix_rank(z[train])),
        signature_effective_rank=participation, kernel_std=float(kernel.std()),
        train_similarity_mean=float(train_kernel.mean()) if len(train_kernel) else 1.0,
        train_similarity_std=float(train_kernel.std()) if len(train_kernel) else 0.0,
        train_distance_p05=float(np.quantile(positive, .05)) if len(positive) else 0.0,
        train_distance_median=tau,
        train_distance_p95=float(np.quantile(positive, .95)) if len(positive) else 0.0,
        tau=tau, kernel_center=float(train_kernel.mean()) if len(train_kernel) else 1.0,
    )


def make_fold_cache(x, y, folds, prior, supports):
    from serca_v2 import DOMAIN_PAIRS, RANDOM_RULES, SEMANTIC_TOPK, TRUE_RULES, make_semantic_neighbors
    PRIOR_DIR.mkdir(parents=True, exist_ok=True)
    fc = np.asarray(x[:, 64:], dtype=np.float64)
    sites = np.asarray(x[:, 26:46]).argmax(axis=1)
    random_pairs, random_candidates = frozen_random_rules()
    assert random_pairs == RANDOM_RULES
    assert set(random_pairs).isdisjoint(TRUE_RULES)
    cache_hashes, fold_meta = {}, {}
    for fold, (train, test) in enumerate(folds, start=1):
        z, td_meta = normative_deviations(x, y, train, fold)
        first = first_order_scores(z, supports)
        pair_sig, pair_reports = pair_interactions(first, train)
        if fold == 1:
            y_changed = np.asarray(y).copy()
            y_changed[test] = 1 - y_changed[test]
            z_changed, _ = normative_deviations(x, y_changed, train, fold)
            assert np.array_equal(z, z_changed), "held-out labels changed TD prior"
        signatures, masks, group_stats = {}, {}, {}
        for config in CONFIGS[1:]:
            sig = np.zeros((871, 24), dtype=np.float32)
            mask = np.zeros(24, dtype=bool)
            if config == "E1_SERCA34_CAL":
                per_domain = []
                route_values = []
                for domain_index in range(4):
                    support = supports[f"D{domain_index+1}"]
                    raw = fc[:, support]
                    mean = raw[train].mean(axis=0)
                    scale = raw[train].std(axis=0)
                    scale = np.where(scale > 1e-8, scale, 1.0)
                    standardized = (raw - mean) / scale
                    norm = np.einsum("ij,ij->i", standardized, standardized)
                    d2 = np.maximum(norm[:, None] + norm[None, :] - 2 * standardized @ standardized.T, 0)
                    train_d = d2[np.ix_(train, train)][np.triu_indices(len(train), 1)]
                    train_d = train_d[train_d > 1e-12]
                    tau = float(np.median(train_d)) if len(train_d) else 1.0
                    per_domain.append(np.exp(-d2 / max(tau, 1e-12)).astype(np.float32))
                    route_values.extend([standardized.mean(axis=1), standardized.std(axis=1)])
                sig[:, :8] = np.column_stack(route_values).astype(np.float32)
                mask[:8] = True
                kernel = np.mean(per_domain, axis=0).astype(np.float32)
                np.fill_diagonal(kernel, 1.0)
                tri = np.triu_indices(len(train), 1)
                train_values = kernel[np.ix_(train, train)][tri]
                stats = dict(
                    active_dimensions=8, signature_train_rank=int(np.linalg.matrix_rank(sig[train, :8])),
                    signature_effective_rank=float(np.linalg.matrix_rank(sig[train, :8])),
                    kernel_std=float(kernel.std()), train_similarity_mean=float(train_values.mean()),
                    train_similarity_std=float(train_values.std()), tau="mean of four domain median distances",
                    kernel_center=float(train_values.mean()), per_domain_kernels=4,
                )
            elif config == "E2_NUMERIC_GLOBAL":
                rng = np.random.Generator(np.random.PCG64(350200))
                projection = rng.choice(np.asarray([-1.0, 1.0]), size=(256, 24)) / np.sqrt(256.0)
                projected = z.astype(np.float64) @ projection
                mean, scale = projected[train].mean(axis=0), projected[train].std(axis=0)
                scale = np.where(scale > 1e-8, scale, 1.0)
                sig = ((projected - mean) / scale).astype(np.float32)
                mask[:] = True
                kernel, stats = fit_kernel(z.astype(np.float64), np.ones(256, dtype=bool), train)
                stats["router_signature_projection"] = "fixed Rademacher 256x24, local PCG64 seed 350200"
            elif config == "E3_FIRST_ORDER":
                sig[:, :8] = first
                mask[:8] = True
                kernel, stats = fit_kernel(sig, mask, train)
            else:
                selected = DOMAIN_PAIRS if config == "E4_ALL_PAIR" else TRUE_RULES
                if config == "E5_RANDOM_RULE":
                    selected = random_pairs
                sig = pair_sig.copy()
                for pair_index, pair in enumerate(DOMAIN_PAIRS):
                    if pair in selected:
                        mask[4 * pair_index:4 * pair_index + 4] = True
                if config == "E6_PATIENT_SHUFFLED":
                    rng = np.random.Generator(np.random.PCG64(PATIENT_PERMUTATION_SEED + fold))
                    permutation = np.arange(871, dtype=np.int64)
                    train_mask = np.zeros(871, dtype=bool)
                    train_mask[train] = True
                    singleton_count = 0
                    for split_mask in (train_mask, ~train_mask):
                        for site in np.unique(sites):
                            ids = np.flatnonzero((sites == site) & split_mask)
                            if len(ids) <= 1:
                                singleton_count += len(ids)
                            else:
                                permutation[ids] = rng.permutation(ids)
                    assert np.array_equal(np.sort(permutation), np.arange(871))
                    assert np.all(train_mask[permutation[train_mask]])
                    assert np.all(~train_mask[permutation[~train_mask]])
                    sig = sig[permutation]
                    stats_shuffle = dict(
                        permutation_sha256=sha_bytes(permutation.tobytes()),
                        changed_count=int(np.count_nonzero(permutation != np.arange(871))),
                        changed_fraction=float(np.mean(permutation != np.arange(871))),
                        singleton_within_site_and_split=singleton_count,
                        bijection=True, no_cross_train_test_mapping=True,
                    )
                else:
                    stats_shuffle = {}
                kernel, stats = fit_kernel(sig, mask, train)
                stats.update(pair_features=pair_reports, selected_rule_pairs=[list(p) for p in selected],
                             patient_shuffle=stats_shuffle)
            if config == "E11_SERCA_V2_FULL":
                # The full model shares the registered pair signature with its ablations.
                pass
            if config in ("E7_ORIGINAL_EDGES", "E8_CORR_ONLY", "E9_AUX_ONLY",
                          "E10_DIRECT_MATCHED", "E11_SERCA_V2_FULL"):
                sig = pair_sig.copy()
                mask[:] = False
                for pair_index, pair in enumerate(DOMAIN_PAIRS):
                    if pair in TRUE_RULES:
                        mask[4 * pair_index:4 * pair_index + 4] = True
                kernel, stats = fit_kernel(sig, mask, train)
                stats.update(pair_features=pair_reports,
                             selected_rule_pairs=[list(p) for p in TRUE_RULES])
            signatures[config] = sig
            masks[config] = mask
            if config == "E7_ORIGINAL_EDGES":
                neighbor = np.zeros_like(kernel)
                neighbor_meta = dict(directed_edge_count=0, train_source_only=True,
                                     test_rows_with_edges=0, train_rows_with_edges=0, topk=0)
            else:
                neighbor, neighbor_meta = make_semantic_neighbors(kernel, train, SEMANTIC_TOPK)
            same_site = sites[train, None] == sites[train][None, :]
            train_kernel = kernel[np.ix_(train, train)]
            upper = np.triu(np.ones(same_site.shape, dtype=bool), 1)
            same_values = train_kernel[upper & same_site]
            diff_values = train_kernel[upper & ~same_site]
            stats.update(
                semantic_neighbor_graph=neighbor_meta,
                same_site_train_similarity_mean=float(same_values.mean()) if len(same_values) else None,
                different_site_train_similarity_mean=float(diff_values.mean()) if len(diff_values) else None,
                site_label_source="PHENO SITE_ID one-hot only",
            )
            group_stats[config] = stats
            signatures[config + "_K"] = kernel
            signatures[config + "_N"] = neighbor
        meta = dict(
            fold=fold, train_count=int(len(train)), test_count=int(len(test)),
            train_indices_sha256=sha_bytes(np.asarray(train, dtype=np.int64).tobytes()),
            test_indices_sha256=sha_bytes(np.asarray(test, dtype=np.int64).tobytes()),
            normative_reference=td_meta, first_order_summary={
                "mean": first[train].mean(axis=0).tolist(),
                "std": first[train].std(axis=0).tolist(),
            },
            joint_interaction_diagnostics=pair_reports,
            true_pair_non_degenerate={
                str(pair): any(item["nondegenerate"] for item in pair_reports
                               if tuple(item["pair"]) == (pair[0] + 1, pair[1] + 1))
                for pair in TRUE_RULES
            },
            random_rule_pairs=[list(pair) for pair in random_pairs],
            random_rule_candidates=[list(graph) for graph in random_candidates],
            groups=group_stats, heldout_label_invariance_checked=(fold == 1),
        )
        path = PRIOR_DIR / f"fold_{fold:02d}.npz"
        cache_values = {}
        for config in CONFIGS[1:]:
            cache_values[config + "_SIG"] = signatures[config]
            cache_values[config + "_MASK"] = masks[config]
            cache_values[config + "_K"] = signatures[config + "_K"]
            cache_values[config + "_N"] = signatures[config + "_N"]
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                assert saved["metadata"].item() == json.dumps(meta, sort_keys=True, ensure_ascii=False)
                for key, value in cache_values.items():
                    assert np.array_equal(saved[key], value), key
        else:
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **cache_values,
                                    metadata=np.asarray(json.dumps(meta, sort_keys=True, ensure_ascii=False)))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        cache_hashes[str(fold)] = sha_file(path)
        fold_meta[str(fold)] = meta
    return cache_hashes, fold_meta


def group_spec(config):
    relations = [0, 1, 2, 3]
    if config == "E8_CORR_ONLY":
        relations = [3]
    elif config == "E9_AUX_ONLY":
        relations = [0, 1, 2]
    return dict(
        relation_indices=relations,
        direct_message=config == "E10_DIRECT_MATCHED",
        add_semantic_neighbors=config != "E7_ORIGINAL_EDGES",
        allocated_trainable_parameter_elements=0 if config == "E0_B0A" else 12101,
        registered_prior=config,
    )


def load_prior(fold, config):
    spec = group_spec(config)
    if config == "E0_B0A":
        return None
    with np.load(PRIOR_DIR / f"fold_{fold:02d}.npz", allow_pickle=False) as cache:
        meta = json.loads(cache["metadata"].item())
        signature = cache[config + "_SIG"].copy()
        signature_mask = cache[config + "_MASK"].copy()
        kernel = cache[config + "_K"].copy()
        neighbors = cache[config + "_N"].copy()
    stats = meta["groups"][config]
    return dict(
        signature=signature.astype(np.float32), signature_mask=signature_mask.astype(bool),
        semantic_kernel=kernel.astype(np.float32),
        kernel_center=float(stats["kernel_center"]),
        semantic_neighbors=neighbors.astype(np.float32),
        relation_mask=np.asarray([i in spec["relation_indices"] for i in range(4)], dtype=bool),
        stats=stats, cache_meta=meta, spec=spec,
    )


def entry_rng_path(fold):
    return RNG_DIR / f"fold_{fold:02d}.pkl"


def task_path(config):
    return STATE_DIR / f"{config}.pt"


def task_status_path(config):
    return STATE_DIR / f"{config}.json"


def task_signature(config):
    payload = dict(config=config, sources=source_hashes(), code={
        name: sha_file(EXP / name) for name in
        ("README.md", "serca_v2.py", "medical_prior.json", "integration.py",
         "integration.patch", "run.py")},
        inputs=input_hashes(), history=history_hashes(), protocol=PROTOCOL,
        cache_hashes=json.loads(MANIFEST.read_text(encoding="utf-8"))["prior_cache_sha256"])
    return canonical_hash(payload)


def history_reference_rows():
    with (EXP32 / "results/fold_metrics.csv").open(newline="", encoding="utf-8") as stream:
        e32 = {int(row["fold"]): row for row in csv.DictReader(stream) if row["config"] == "B0"}
    with (EXP33 / "results/fold_metrics.csv").open(newline="", encoding="utf-8") as stream:
        e33 = {int(row["fold"]): row for row in csv.DictReader(stream) if row["config"] == "G0_B0A"}
    with (EXP34 / "results/fold_metrics.csv").open(newline="", encoding="utf-8") as stream:
        e34 = {int(row["fold"]): row for row in csv.DictReader(stream) if row["config"] == "S0_B0A"}
    assert set(e32) == set(e33) == set(e34) == set(range(1, 11))
    for fold in range(1, 11):
        for row, auc_name in ((e33[fold], "hard_auc"), (e34[fold], "hard_auc")):
            for metric in ("acc", "sen_asd", "spe_td"):
                assert abs(float(e32[fold][metric]) - float(row[metric])) <= 1e-12
            assert abs(float(e32[fold]["auc"]) - float(row[auc_name])) <= 1e-12
            assert e32[fold]["initial_trunk_sha256"] == row["initial_trunk_sha256"]
            assert e32[fold]["initial_rng_sha256"] == row["initial_rng_sha256"]
            assert int(e32[fold]["best_epoch"]) == int(row["best_epoch"])
    return e32, e33, e34


def prepare():
    for directory in (RUNTIME, SOURCE, PRIOR_DIR, RNG_DIR, STATE_DIR, LOG_DIR, TMP_DIR, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    assert (Path.home() / ".cache/clip/RN50.pt").is_file(), "cached RN50 checkpoint missing; no download"
    assert EXP34.is_dir() and (EXP34 / "results/fold_metrics.csv").is_file()
    prior, supports = build_concepts()
    patch = emit_sources()
    x, y, folds, mapping = read_data()
    cache_hashes, fold_meta = make_fold_cache(x, y, folds, prior, supports)
    e32, e33, e34 = history_reference_rows()
    handoff = ROOT / "handoff.md"
    random_pairs, random_candidates = frozen_random_rules()
    code_names = ("README.md", "serca_v2.py", "medical_prior.json", "integration.py",
                  "integration.patch", "run.py")
    manifest = dict(
        experiment="Experiment 35 — SeRCA-v2 Disease-Semantic Relational Pattern Alignment",
        status="PREPARED", created_utc=utc(), seed=0, configs=list(CONFIGS),
        target_folds=120, completed_folds=0, source_sha256=source_hashes(),
        input_sha256=input_hashes(), historical_sha256=history_hashes(),
        code_sha256={name: sha_file(EXP / name) for name in code_names},
        runtime_source_sha256={name: sha_file(SOURCE / name) for name in SOURCE_NAMES},
        integration_patch_sha256=sha_bytes(patch.encode()),
        prior_cache_sha256=cache_hashes,
        prior_fold_statistics=fold_meta,
        handoff_prefix=dict(bytes=handoff.stat().st_size, sha256=sha_file(handoff)),
        paper_audit=dict(
            paper_url="https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf",
            author_repository="https://github.com/wanxixi11/SPromptGL",
            current_repository="https://github.com/tangaii/SPromptGL",
            implementation_authority="Current local checkout hashes in source_sha256; no root file is edited.",
            verified_behavior="Four GraphLearn outputs are participant graphs, thresholded at cosine >0.9 and returned raw; GFR propagates full 320D concatenated features using normalize_adj(A+I), and i==1 overwrites i==0 combined_feat. PLN original construction order and cached CLIP initialization are preserved; B0-A bypasses PEN forward. simple-2 backpropagates ClsLoss plus mean four GraphConstructLoss; ALoss is computed but not optimized.",
            cam_boundary="InputLayer outputs [871,4,36] mixed latent positions; the four positions are not independent raw modalities or ROIs.",
            historical_result_boundary="Paper full-model ABIDE ACC 92.30% differs from local B0-A re-training and Experiment32's 92.0742%.",
            related_work_reviewed=["SPromptGL MICCAI 2025", "MMGPL MedIA 2024", "MGDR MICCAI 2024", "KD-Brain MICCAI 2026", "InfoBridge ICCV 2025", "To Align or Not to Align AAAI 2026", "Autism FC heterogeneity Nature Mental Health 2026", "Age-atypical ASD networks 2024"]),
        protocol=PROTOCOL,
        data=dict(shape=list(x.shape), labels={"ASD_0": int(np.sum(y == 0)), "TD_1": int(np.sum(y == 1))},
                  modality_order=list(mapping), modality_dimensions=INPUT_DIMS,
                  pheno_ranges={"age": [0, 26], "site": [26, 46], "sex": [46, 48]},
                  train_sizes=[int(len(tr)) for tr, _ in folds],
                  test_sizes=[int(len(te)) for _, te in folds]),
        concepts=prior["feature_mapping"],
        medical_relation_rules=prior["relation_rules"],
        random_rule_control=dict(seed=RANDOM_RULE_SEED,
                                 selected_zero_based_pairs=[list(pair) for pair in random_pairs],
                                 candidate_graphs=[list(graph) for graph in random_candidates],
                                 degree_multiset_matches=True, disjoint_from_true=True),
        method=dict(
            location="after PLN InputLayer and before Encoder",
            td_reference="Ridge alpha=10 on age/site/sex PHENO one-hot, fit on outer-train TD only; train-TD residuals are fixed 5-fold OOF; train-ASD/test residuals use full outer-train TD fit; z uses OOF TD residual scale, clipped [-3,3].",
            interaction="signed u+/u- concept deviations above absolute z=1; 4 signed pair products; regress each product on 8 first-order scores using outer-train only, then center/scale on outer-train.",
            semantic_prior="two frozen undirected concept-level rules D2-D4 and D3-D4; no direction assumed; no physical between-network FC edge is inferred.",
            graph="same-fold K_sem from active signature; directed top-8 recipient-to-outer-train-source edges; row-normalized added branch. Existing raw GFR support is modulated by clipped 1+0.25*(K-center), and 0.25*K_sparse is added; original GFR symmetric normalize_adj(A+I) remains unchanged.",
            message="V_m=LayerNorm(Linear(X_m,36)); M_base_m=T_base_m@V_m; M_sem_m=T_sem_m@V_m; route weighted per-patient messages with a learned null plus four source logits; FULL uses M_sem-M_base.",
            residual="scale=RMS_train(H)/max(RMS_train(R),0.1*RMS_train(H)); detached scale; Delta=0.05*tanh(alpha)*R_hat, alpha[4]=0; each subject/token delta capped at 5% of H norm.",
            branch_parameters="four modality value projections, LayerNorms, one Linear router, alpha[4]; added PLN Adam group lr=.0038, weight_decay=0; same module shape in E1-E11.",
            original_graph_loss="unchanged GraphConstructLoss on the four original raw A_m matrices",
            labels_in_prior=False, clip_text_encoder_calls_added=0,
            allocated_trainable_parameter_elements_by_group={key: group_spec(key)["allocated_trainable_parameter_elements"] for key in CONFIGS},
        ),
        groups={key: group_spec(key) for key in CONFIGS},
        gpu_assignment={str(gpu): list(names) for gpu, names in GPU_CONFIGS.items()},
        worker_protocol="E0 runs first as one continuous 10-fold seed0 worker and snapshots full Python/NumPy/torch CPU/CUDA RNG before each construction. E1-E11 run independent continuous 10-fold workers, restoring the corresponding E0 fold-entry RNG before each construction; this is paired fold-entry control, not natural continuous cross-fold RNG.",
        worker_limit_per_gpu=WORKERS_PER_GPU,
        checkpoint=dict(epoch_interval=CHECKPOINT_EVERY, atomic=True,
                        includes_gfr_pln_serca_v2_both_adams_best_state_wait_epoch_rng_fold_entry_rng=True,
                        completed_fold_rows_committed_once=True),
        frozen_internal_screen=dict(
            e11_acc_and_hard_auc_above_e0=True, e11_acc_gain_min_pp=.23,
            e11_vs_e1_e2_e3_e4_e5_e6_e7_e8_e9_e10_registered_comparisons_required=True,
            sen_noninferiority_drop_max_pp=SEN_NONINFERIORITY_PP,
            all_conditions_are_single_seed_internal_screen_not_significance=True),
        baseline_reference=dict(experiment32_b0_fold_metrics_sha256=sha_file(EXP32 / "results/fold_metrics.csv"),
                                experiment33_b0_fold_metrics_sha256=sha_file(EXP33 / "results/fold_metrics.csv"),
                                experiment34_b0_fold_metrics_sha256=sha_file(EXP34 / "results/fold_metrics.csv"),
                                no_historical_checkpoint_reused=True,
                                expected_initial_parameters=[e32[i]["initial_trunk_sha256"] for i in range(1, 11)],
                                expected_initial_rng=[e32[i]["initial_rng_sha256"] for i in range(1, 11)],
                                expected_best_epochs=[int(e32[i]["best_epoch"]) for i in range(1, 11)]),
        preflight=dict(passed=False),
        workers={key: dict(gpu=GPU_OF[key], status="PENDING", completed_folds=0) for key in CONFIGS},
    )
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert existing["source_sha256"] == manifest["source_sha256"]
        assert existing["input_sha256"] == manifest["input_sha256"]
        assert existing["historical_sha256"] == manifest["historical_sha256"]
        if existing["code_sha256"] != manifest["code_sha256"]:
            assert existing.get("status") == "PREPARED"
            assert not list(STATE_DIR.glob("*.pt")) and not list(RNG_DIR.glob("fold_*.pkl"))
            existing["code_sha256"] = manifest["code_sha256"]
            existing["runtime_source_sha256"] = manifest["runtime_source_sha256"]
            existing["integration_patch_sha256"] = manifest["integration_patch_sha256"]
            existing["preflight"] = dict(passed=False)
        for key in ("groups", "method", "gpu_assignment", "worker_protocol",
                    "worker_limit_per_gpu", "checkpoint", "frozen_internal_screen"):
            existing[key] = manifest[key]
        existing["prior_cache_sha256"] = cache_hashes
        existing["prior_fold_statistics"] = fold_meta
        existing["prepared_utc"] = utc()
        json_write(MANIFEST, existing)
        return existing
    json_write(MANIFEST, manifest)
    return manifest


def setup_runtime_imports():
    os.environ["TMPDIR"] = str(TMP_DIR)
    os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME / "mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME / "xdg-cache"))
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SOURCE))
    sys.path.insert(0, str(EXP))


def get_context():
    global _CONTEXT
    if _CONTEXT is not None:
        return _CONTEXT
    setup_runtime_imports()
    x, y, folds, mapping = read_data()
    model = importlib.import_module("model")
    network = importlib.import_module("network")
    utils = importlib.import_module("utils")
    import integration
    import torch
    _CONTEXT = dict(x=x, y=y, folds=folds, mapping=mapping,
                    model=model, network=network, utils=utils,
                    integration=integration, torch=torch)
    return _CONTEXT


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
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_hash(state):
    digest = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    digest.update(state["cpu"].cpu().numpy().tobytes())
    for value in state["cuda"]:
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def trunk_hash(agent):
    digest = hashlib.sha256()
    for module_name, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion)):
        for name, parameter in sorted(module.named_parameters()):
            if module_name == "pln" and name.startswith("serca35."):
                continue
            digest.update(module_name.encode() + b"/" + name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def close_agent(agent):
    for name in ("MF_sav", "GCMP_sav"):
        stream = getattr(agent, name, None)
        if stream is not None:
            stream.close()


def capture_state(agent):
    return dict(gfr=copy.deepcopy(agent.MessagePassing0.state_dict()),
                pln=copy.deepcopy(agent.ModalFusion.state_dict()))


def load_state(agent, state):
    agent.MessagePassing0.load_state_dict(state["gfr"])
    agent.ModalFusion.load_state_dict(state["pln"])


def optimizer_ids(agent):
    return [id(parameter) for optimizer in (agent.optimizer_MP0, agent.optimizer_MF)
            for group in optimizer.param_groups for parameter in group["params"]]


def parameter_ids(agent):
    return {id(parameter) for module in (agent.MessagePassing0, agent.ModalFusion)
            for parameter in module.parameters() if parameter.requires_grad}


def make_agent(config, fold, context, prior):
    torch = context["torch"]
    train, test = context["folds"][fold - 1]
    context["integration"].configure(config=config, prior=prior)
    hp = SimpleNamespace(**PROTOCOL)
    with contextlib.redirect_stdout(io.StringIO()):
        agent = context["model"].EvalHelper(INPUT_DIMS, context["x"], context["y"], hp, train, test)
    assert np.array_equal(agent.val_idx, agent.tst_idx)
    assert agent.ModalFusion.serca35_bypass_pen is True
    assert (hasattr(agent.ModalFusion, "serca35") == (config != "E0_B0A"))
    ids = optimizer_ids(agent)
    assert len(ids) == len(set(ids)) and set(ids) == parameter_ids(agent)
    assert all(abs(group["lr"] - PROTOCOL["lr"]) < 1e-12
               for optimizer in (agent.optimizer_MP0, agent.optimizer_MF)
               for group in optimizer.param_groups)
    if config != "E0_B0A":
        assert agent.optimizer_MF.param_groups[-1]["weight_decay"] == 0.0
        assert agent.optimizer_MF.param_groups[-1]["params"] == [
            parameter for parameter in agent.ModalFusion.serca35.parameters() if parameter.requires_grad]
    return agent


def prediction_metrics(agent, context, fold, collect=False):
    module = getattr(agent.ModalFusion, "serca35", None)
    if module is not None:
        module.collect_diagnostics = bool(collect)
        module.last_diagnostics = None
    train, test = context["folds"][fold - 1]
    with contextlib.redirect_stdout(io.StringIO()):
        acc, auc, target, prediction = agent._print_acc(np.asarray(test), PROTOCOL["mode"], tst=False, end="")
    if module is not None:
        module.collect_diagnostics = False
    target = target.detach().cpu().numpy().astype(np.int64)
    prediction = prediction.detach().cpu().numpy().astype(np.int64)
    tp = int(np.count_nonzero((target == 0) & (prediction == 0)))
    fn = int(np.count_nonzero((target == 0) & (prediction == 1)))
    fp = int(np.count_nonzero((target == 1) & (prediction == 0)))
    tn = int(np.count_nonzero((target == 1) & (prediction == 1)))
    metrics = dict(acc=float(acc), hard_auc=float(auc), sen_asd=tp / (tp + fn),
                   spe_td=tn / (tn + fp), asd_tp=tp, asd_fn=fn, td_fp=fp, td_tn=tn)
    assert abs(metrics["acc"] - (tp + tn) / len(target)) < 1e-12
    return metrics, prediction


def evaluate_fold(agent, context, fold, collect=False):
    return prediction_metrics(agent, context, fold, collect)[0]


def full_counterfactuals(agent, context, fold, base_predictions):
    module = agent.ModalFusion.serca35
    result = {}
    old_direct = module.direct_message
    old_override = module.source_override
    module.direct_message = True
    direct_metrics, direct_predictions = prediction_metrics(agent, context, fold, collect=True)
    direct_diag = module.last_diagnostics
    result["direct_same_checkpoint"] = dict(
        changed_predictions=int(np.count_nonzero(direct_predictions != base_predictions)),
        changed_fraction=float(np.mean(direct_predictions != base_predictions)),
        metrics={key: direct_metrics[key] for key in METRICS},
        raw_message_rms=direct_diag["raw_message_rms"],
        calibrated_message_rms=direct_diag["calibrated_message_rms"],
        calibration_scale=direct_diag["calibration_scale"],
        max_residual_ratio=direct_diag["max_token_residual_ratio_observed"],
    )
    module.direct_message = old_direct
    source_changes = {}
    for source in range(4):
        override = [index == source for index in range(4)]
        module.source_override = override
        metrics, predictions = prediction_metrics(agent, context, fold, collect=True)
        diag = module.last_diagnostics
        source_changes[RELATION_NAMES[source]] = dict(
            changed_predictions=int(np.count_nonzero(predictions != base_predictions)),
            changed_fraction=float(np.mean(predictions != base_predictions)),
            metrics={key: metrics[key] for key in METRICS},
            route_mean=diag["route_mean"],
            source_contribution_rms=diag["source_contribution_rms"],
        )
    module.source_override = old_override
    module.direct_message = old_direct
    result["source_ablations_same_checkpoint"] = source_changes
    return result


def fold_diagnostics(agent, context, fold, config, prior, gpu):
    torch = context["torch"]
    module = getattr(agent.ModalFusion, "serca35", None)
    train, _ = context["folds"][fold - 1]
    row = dict(config=config, seed=0, fold=fold, physical_gpu=gpu,
               train_count=len(train), clip_model_loaded_in_pln_construction=True,
               original_pen_forward_bypassed=True)
    raw_adjs = agent.MessagePassing0.last_raw_adjs
    raw_edges, densities = [], []
    for raw in raw_adjs:
        off = raw.detach().clone()
        off.fill_diagonal_(0)
        edge_count = int(torch.count_nonzero(off).item())
        raw_edges.append(edge_count)
        densities.append(edge_count / (871 * 870))
    if module is None:
        row.update(alpha_json="[]", route_mean_json="[]", source_weights_json="[]",
                   null_weight_mean=0.0, residual_ratio_mean_json="[]",
                   residual_ratio_max_json="[]", source_message_rms_json="[]",
                   source_contribution_rms_json="[]", semantic_edge_count=0,
                   semantic_neighbor_overlap_json="[]", existing_edge_reweight_json="[]",
                   relation_graph_offdiag_edges_json=json.dumps(raw_edges),
                   relation_graph_density_json=json.dumps(densities),
                   difference_message=False, effective_trainable_parameters=0,
                   allocated_trainable_parameter_elements=0)
        return row
    diag = module.last_diagnostics
    assert diag is not None
    pmeta = prior["cache_meta"]["groups"][config]
    row.update(
        alpha_json=json.dumps(diag["alpha"]),
        route_mean_json=json.dumps(diag["route_mean"]),
        source_weights_json=json.dumps(diag["source_weights_mean"]),
        null_weight_mean=diag["null_weight_mean"],
        residual_ratio_mean_json=json.dumps(diag["mean_token_residual_ratio"]),
        residual_ratio_max_json=json.dumps(diag["max_token_residual_ratio"]),
        residual_ratio_max_observed=diag["max_token_residual_ratio_observed"],
        source_message_rms_json=json.dumps(diag["source_message_rms"]),
        source_contribution_rms_json=json.dumps(diag["source_contribution_rms"]),
        raw_message_rms=diag["raw_message_rms"], target_rms=diag["target_rms"],
        calibration_scale=diag["calibration_scale"],
        calibrated_message_rms=diag["calibrated_message_rms"],
        semantic_edge_count=diag["semantic_edge_count"],
        semantic_neighbor_overlap_json=json.dumps(diag["semantic_neighbor_overlap_fraction"]),
        existing_edge_reweight_json=json.dumps(diag["existing_edge_relative_reweight"]),
        semantic_row_normalization_error_json=json.dumps(diag["semantic_row_normalization_error"]),
        signature_kernel_std=pmeta["kernel_std"],
        signature_effective_rank=pmeta["signature_effective_rank"],
        signature_train_rank=pmeta["signature_train_rank"],
        signature_active_dimensions=pmeta["active_dimensions"],
        same_site_similarity=pmeta.get("same_site_train_similarity_mean"),
        different_site_similarity=pmeta.get("different_site_train_similarity_mean"),
        relation_graph_offdiag_edges_json=json.dumps(raw_edges),
        relation_graph_density_json=json.dumps(densities),
        patient_permutation_changed_fraction=pmeta.get("patient_shuffle", {}).get("changed_fraction", 0.0),
        patient_permutation_singletons=pmeta.get("patient_shuffle", {}).get("singleton_within_site_and_split", 0),
        patient_permutation_sha256=pmeta.get("patient_shuffle", {}).get("permutation_sha256", ""),
        patient_permutation_bijection=pmeta.get("patient_shuffle", {}).get("bijection", True),
        no_cross_train_test_mapping=pmeta.get("patient_shuffle", {}).get("no_cross_train_test_mapping", True),
        difference_message=not module.direct_message,
        effective_trainable_parameters=group_spec(config)["allocated_trainable_parameter_elements"],
        allocated_trainable_parameter_elements=group_spec(config)["allocated_trainable_parameter_elements"],
    )
    return row


def save_status(config, gpu, status, rows=0, **values):
    json_write(task_status_path(config), dict(config=config, gpu=gpu, status=status,
                                               completed_folds=rows, updated_utc=utc(), **values))


def ensure_clip_loaded_without_rng(context):
    torch, network = context["torch"], context["network"]
    if network._prompt_text_features is not None:
        return
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        network.get_prompt_text_features()


def run_worker(config, gpu):
    import torch
    assert config in CONFIGS and GPU_OF[config] == gpu
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu)
    assert int(os.environ.get("SERCA_PHYSICAL_GPU", -1)) == gpu
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    context = get_context()
    context["device"] = torch.device("cuda")
    for directory in (TMP_DIR, STATE_DIR, LOG_DIR, RNG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    lock_path = STATE_DIR / f"{config}.lock"
    lock = lock_path.open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source_sha256"] == source_hashes()
    assert manifest["input_sha256"] == input_hashes()
    saved_path = task_path(config)
    saved = torch_read(saved_path) if saved_path.exists() else None
    signature = task_signature(config)
    if saved:
        assert saved["signature"] == signature and saved["config"] == config
        if saved["phase"] == "COMPLETE":
            print(f"SERCA_SKIP_COMPLETE {config}", flush=True)
            return
    seed_everything(torch, 0)
    network = context["network"]
    calls = {"clip_load": 0, "prompt_features": 0}
    original_load, original_get = network.clip.load, network.get_prompt_text_features

    def tracked_load(*args, **kwargs):
        calls["clip_load"] += 1
        return original_load(*args, **kwargs)

    def tracked_get(*args, **kwargs):
        calls["prompt_features"] += 1
        return original_get(*args, **kwargs)

    network.clip.load = tracked_load
    network.get_prompt_text_features = tracked_get
    if saved:
        calls.update(saved.get("clip_calls", {}))
    phase = saved.get("phase") if saved else None
    rows = list(saved.get("rows", [])) if saved else []
    diagnostics = list(saved.get("diagnostics", [])) if saved else []
    first_fold = int(saved.get("next_fold", len(rows) + 1)) if phase == "BOUNDARY" else len(rows) + 1
    epoch_resume = saved if phase == "EPOCH" else None
    if saved and (first_fold > 1 or (epoch_resume and int(epoch_resume["fold"]) > 1)):
        ensure_clip_loaded_without_rng(context)
    if phase == "BOUNDARY":
        rng_restore(torch, saved["rng"])
    elif phase == "EPOCH" and int(epoch_resume["fold"]) > 1:
        rng_restore(torch, epoch_resume["fold_entry_rng"])
    seconds = float(saved.get("seconds", 0.0)) if saved else 0.0
    started_utc = saved.get("started_utc", utc()) if saved else utc()
    previous_started = time.monotonic()
    stop = {"requested": False}

    def request_stop(_signum, _frame):
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    save_status(config, gpu, "RUNNING", len(rows), started_utc=started_utc)
    log_path = LOG_DIR / f"{config}.log"
    epoch_resume = saved if phase == "EPOCH" else None
    with log_path.open("a", encoding="utf-8", buffering=1) as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print(f"WORKER_START config={config} gpu={gpu} resumes={saved is not None}", flush=True)
        for fold in range(first_fold, 11):
            if stop["requested"]:
                break
            train, _ = context["folds"][fold - 1]
            prior = load_prior(fold, config)
            if config == "E0_B0A":
                if epoch_resume is not None and int(epoch_resume["fold"]) == fold:
                    fold_entry = epoch_resume["fold_entry_rng"]
                    rng_restore(torch, fold_entry)
                else:
                    fold_entry = rng_capture(torch)
                    rng_path = entry_rng_path(fold)
                    if rng_path.exists():
                        old_entry = pickle.loads(rng_path.read_bytes())
                        assert rng_hash(old_entry) == rng_hash(fold_entry)
                    else:
                        pickle_write(rng_path, fold_entry)
            else:
                rng_path = entry_rng_path(fold)
                assert rng_path.is_file(), f"E0 fold-entry RNG missing: {fold}"
                fold_entry = pickle.loads(rng_path.read_bytes())
                rng_restore(torch, fold_entry)
            fold_entry_hash = rng_hash(fold_entry)
            agent = make_agent(config, fold, context, prior)
            initial_hash = trunk_hash(agent)
            if epoch_resume is not None and int(epoch_resume["fold"]) == fold:
                load_state(agent, epoch_resume["model"])
                agent.optimizer_MP0.load_state_dict(epoch_resume["optimizer_mp"])
                agent.optimizer_MF.load_state_dict(epoch_resume["optimizer_mf"])
                rng_restore(torch, epoch_resume["rng"])
                best_state = epoch_resume["best_state"]
                best_acc, best_epoch = float(epoch_resume["best_acc"]), int(epoch_resume["best_epoch"])
                wait = int(epoch_resume["wait"])
                start_epoch = int(epoch_resume["epoch"]) + 1
                fold_seconds_before = float(epoch_resume.get("fold_seconds", 0.0))
                initial_hash = epoch_resume["initial_trunk_sha256"]
                fold_entry_hash = epoch_resume["initial_rng_sha256"]
                already_stopped = wait > PROTOCOL["early"]
            else:
                best_state, best_acc, best_epoch, wait = None, 0.0, -1, 0
                start_epoch, fold_seconds_before, already_stopped = 0, 0.0, False
            epoch_resume = None
            fold_started = time.monotonic()
            last_epoch = start_epoch - 1
            for epoch in (() if already_stopped else range(start_epoch, PROTOCOL["nepoch"])):
                agent.run_epoch("simple-2")
                val_metrics = evaluate_fold(agent, context, fold)
                val_acc = val_metrics["acc"]
                if val_acc > best_acc:
                    best_acc, best_epoch, wait = val_acc, epoch, 0
                    best_state = capture_state(agent)
                else:
                    wait += 1
                last_epoch = epoch
                should_stop = wait > PROTOCOL["early"] or stop["requested"]
                if (epoch + 1) % CHECKPOINT_EVERY == 0 or should_stop:
                    torch_write(saved_path, dict(
                        signature=signature, config=config, phase="EPOCH", fold=fold,
                        epoch=epoch, model=capture_state(agent),
                        optimizer_mp=agent.optimizer_MP0.state_dict(),
                        optimizer_mf=agent.optimizer_MF.state_dict(), rng=rng_capture(torch),
                        fold_entry_rng=fold_entry, best_state=best_state, best_acc=best_acc,
                        best_epoch=best_epoch, wait=wait,
                        fold_seconds=fold_seconds_before + time.monotonic() - fold_started,
                        initial_trunk_sha256=initial_hash, initial_rng_sha256=fold_entry_hash,
                        clip_calls=calls,
                        rows=rows, diagnostics=diagnostics, seconds=seconds + time.monotonic() - previous_started,
                        started_utc=started_utc,
                    ))
                    save_status(config, gpu, "RUNNING", len(rows), current_fold=fold,
                                current_epoch=epoch, started_utc=started_utc)
                if should_stop:
                    break
            if stop["requested"]:
                close_agent(agent)
                break
            assert best_state is not None, f"no best model recorded: {config} fold {fold}"
            continuation_rng = rng_capture(torch)
            load_state(agent, best_state)
            metrics, best_predictions = prediction_metrics(agent, context, fold, collect=True)
            assert abs(metrics["acc"] - best_acc) <= 1e-12, \
                "restored best model and saved val=test accuracy diverged"
            diagnostic = fold_diagnostics(agent, context, fold, config, prior, gpu)
            if config == "E11_SERCA_V2_FULL":
                diagnostic["same_checkpoint_counterfactual_json"] = json.dumps(
                    full_counterfactuals(agent, context, fold, best_predictions),
                    ensure_ascii=False, allow_nan=False,
                )
            row = dict(config=config, seed=0, fold=fold, physical_gpu=gpu,
                       **metrics, best_epoch=best_epoch, epochs_executed=last_epoch + 1,
                       initial_trunk_sha256=initial_hash, initial_rng_sha256=fold_entry_hash,
                       clip_load_calls_so_far=calls["clip_load"],
                       clip_feature_calls_so_far=calls["prompt_features"],
                       wall_seconds=fold_seconds_before + time.monotonic() - fold_started)
            rows = [item for item in rows if int(item["fold"]) != fold] + [row]
            diagnostics = [item for item in diagnostics if int(item["fold"]) != fold] + [diagnostic]
            seconds += time.monotonic() - previous_started
            previous_started = time.monotonic()
            rng_restore(torch, continuation_rng)
            torch_write(saved_path, dict(
                signature=signature, config=config, phase="BOUNDARY", next_fold=fold + 1,
                rng=continuation_rng, rows=rows, diagnostics=diagnostics, seconds=seconds,
                started_utc=started_utc, clip_calls=calls,
            ))
            save_status(config, gpu, "RUNNING", len(rows), completed_fold=fold,
                        started_utc=started_utc, seconds=seconds)
            print(f"FOLD_DONE {config} fold={fold} acc={metrics['acc']:.6f} "
                  f"hard_auc={metrics['hard_auc']:.6f} best_epoch={best_epoch}", flush=True)
            close_agent(agent)
            del agent, best_state, prior
            torch.cuda.empty_cache()
        if not stop["requested"]:
            assert len(rows) == 10 and len(diagnostics) == 10
            seconds += time.monotonic() - previous_started
            torch_write(saved_path, dict(signature=signature, config=config, phase="COMPLETE",
                                         rows=rows, diagnostics=diagnostics, seconds=seconds,
                                         started_utc=started_utc, completed_utc=utc(), clip_calls=calls))
            save_status(config, gpu, "COMPLETE", 10, seconds=seconds, clip_calls=calls,
                        started_utc=started_utc, completed_utc=utc())
            print(f"WORKER_COMPLETE {config} seconds={seconds:.2f} clip={calls}", flush=True)


def snapshot(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(snapshot(item) for item in value)
    if isinstance(value, list):
        return [snapshot(item) for item in value]
    return copy.deepcopy(value)


def recursive_equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            recursive_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            recursive_equal(a, b) for a, b in zip(left, right))
    return left == right


def trunk_state(agent):
    return dict(
        gfr={key: snapshot(value) for key, value in agent.MessagePassing0.state_dict().items()},
        pln={key: snapshot(value) for key, value in agent.ModalFusion.state_dict().items()
             if not key.startswith("serca35.")},
    )


def trunk_gradients(agent):
    result = {}
    for prefix, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion)):
        for name, parameter in module.named_parameters():
            if prefix == "pln" and name.startswith("serca35."):
                continue
            result[prefix + "." + name] = None if parameter.grad is None else snapshot(parameter.grad)
    return result


def optimizer_base_state(optimizer):
    state = optimizer.state_dict()
    group = state["param_groups"][0]
    identifiers = set(group["params"])
    return dict(
        group={key: snapshot(value) for key, value in group.items()},
        state={key: snapshot(value) for key, value in state["state"].items() if key in identifiers},
    )


def capture_one_update(agent, context):
    torch = context["torch"]
    model = context["model"]
    captured, handles = {}, []

    def output_hook(key):
        def hook(_module, _inputs, output):
            captured[key] = snapshot(output)
        return hook

    def input_hook(key):
        def hook(_module, inputs):
            captured[key] = snapshot(inputs)
        return hook

    handles.append(agent.dataLayer.register_forward_hook(output_hook("data_layer")))
    handles.append(agent.MessagePassing0.register_forward_pre_hook(input_hook("gfr_input")))
    handles.append(agent.MessagePassing0.register_forward_hook(output_hook("gfr_output")))
    handles.append(agent.ModalFusion.InputLayer.register_forward_hook(output_hook("input_layer")))
    for index, layer in enumerate(agent.ModalFusion.Encoder):
        handles.append(layer.register_forward_hook(output_hook(f"encoder_{index}")))
    for index, layer in enumerate(agent.ModalFusion.FeedForward):
        handles.append(layer.register_forward_hook(output_hook(f"feedforward_{index}")))
    handles.append(agent.ModalFusion.Outputlayer.register_forward_pre_hook(input_hook("output_layer_input")))
    handles.append(agent.ModalFusion.Outputlayer.register_forward_hook(output_hook("output_layer_output")))
    if hasattr(agent.ModalFusion, "serca35"):
        handles.append(agent.ModalFusion.serca35.register_forward_hook(output_hook("serca_output")))

    old_cls, old_graph = model.ClsLoss, model.GraphConstructLoss
    graph_losses = []

    def cls_loss(*args, **kwargs):
        value = old_cls(*args, **kwargs)
        captured["cls_loss"] = snapshot(value)
        return value

    def graph_loss(*args, **kwargs):
        value = old_graph(*args, **kwargs)
        graph_losses.append(snapshot(value))
        return value

    model.ClsLoss, model.GraphConstructLoss = cls_loss, graph_loss
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            agent.run_epoch("simple-2")
    finally:
        model.ClsLoss, model.GraphConstructLoss = old_cls, old_graph
        for handle in handles:
            handle.remove()
    captured["graph_losses"] = graph_losses
    captured["raw_adjs"] = snapshot(agent.MessagePassing0.last_raw_adjs)
    captured["normalized_adjs"] = snapshot(agent.MessagePassing0.last_normalized_adjs)
    captured["gradients"] = trunk_gradients(agent)
    return captured


def _legacy_preflight_checks_e34(gpu):
    import torch
    assert int(os.environ.get("SERCA_PHYSICAL_GPU", -1)) == gpu
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu)
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    context = get_context()
    context["device"] = torch.device("cuda")
    assert len(context["folds"]) == 10
    assert context["x"].shape == (871, 320)
    checks = dict(gpu=gpu, cuda_name=torch.cuda.get_device_name(0), folds=10,
                  sample_count=871, input_shape=[871, 320], formal_groups=len(CONFIGS),
                  formal_folds=len(CONFIGS) * 10, configured_workers_per_gpu=WORKERS_PER_GPU)

    history, _ = history_reference_rows()
    prior = load_prior(1, "S9_SERCA_FULL")
    network = context["network"]
    seed_everything(torch, 0)
    entry = rng_capture(torch)
    assert rng_hash(entry) == history[1]["initial_rng_sha256"]
    base = make_agent("S0_B0A", 1, context, None)
    assert trunk_hash(base) == history[1]["initial_trunk_sha256"]
    assert not hasattr(base.ModalFusion, "serca34")
    assert base.ModalFusion.serca34_bypass_pen is True
    baseline_post_construct_rng = rng_capture(torch)
    baseline_initial_state = trunk_state(base)
    baseline_entry_hash = rng_hash(entry)
    checks["s0_fold1_rng_and_initial_trunk_match_history"] = True
    checks["clip_constructed_and_pen_forward_bypassed"] = dict(
        cached_feature_shape=list(base.ModalFusion.prompt_text_features.shape),
        original_mlp_present=hasattr(base.ModalFusion, "MLP"),
        original_avg_pool_present=hasattr(base.ModalFusion, "avg_pool"),
        no_serca_module=True,
    )

    train_rng = rng_capture(torch)
    base_capture = capture_one_update(base, context)
    base_post_update = trunk_state(base)
    base_grad = base_capture["gradients"]
    base_mp_state = optimizer_base_state(base.optimizer_MP0)
    base_mf_state = optimizer_base_state(base.optimizer_MF)

    network._prompt_text_features = None
    rng_restore(torch, entry)
    context["integration"].configure(
        config="S9_SERCA_FULL", similarities=prior["similarities"], centers=prior["centers"],
        relation_mask=prior["relation_mask"], concept_mask=prior["concept_mask"],
    )
    full = make_agent("S9_SERCA_FULL", 1, context, prior)
    assert trunk_hash(full) == history[1]["initial_trunk_sha256"]
    assert rng_hash(rng_capture(torch)) == rng_hash(baseline_post_construct_rng)
    assert rng_hash(entry) == baseline_entry_hash
    assert recursive_equal(trunk_state(full), baseline_initial_state)
    assert full.ModalFusion.serca34.alpha.count_nonzero().item() == 0
    assert full.ModalFusion.serca34.theta.count_nonzero().item() == 0
    assert len(full.optimizer_MF.param_groups) == 2
    assert full.optimizer_MF.param_groups[-1]["weight_decay"] == 0.0
    checks["fold1_construction_rng_and_original_parameters_equal"] = True
    checks["clip_warm_and_b0a_cache_path_preserved"] = dict(
        prompt_text_feature_shape=list(full.ModalFusion.prompt_text_features.shape),
        serca_added_clip_encoder_calls=0,
        module_init_rng_isolated=True,
        alpha_theta_zero=True,
    )

    rng_restore(torch, train_rng)
    full_capture = capture_one_update(full, context)
    exact_keys = ("data_layer", "gfr_input", "gfr_output", "input_layer", "encoder_0",
                  "feedforward_0", "output_layer_input", "output_layer_output", "raw_adjs",
                  "normalized_adjs", "graph_losses", "cls_loss")
    exact = {key: recursive_equal(base_capture[key], full_capture[key]) for key in exact_keys}
    assert all(exact.values()), {key: value for key, value in exact.items() if not value}
    assert recursive_equal(base_grad, full_capture["gradients"])
    assert recursive_equal(base_post_update, trunk_state(full))
    assert recursive_equal(base_mp_state, optimizer_base_state(full.optimizer_MP0))
    assert recursive_equal(base_mf_state, optimizer_base_state(full.optimizer_MF))
    alpha_grad = full.ModalFusion.serca34.alpha.grad
    theta_zero_grad = full.ModalFusion.serca34.theta.grad
    assert alpha_grad is not None and torch.isfinite(alpha_grad).all() and alpha_grad.norm().item() > 0
    assert theta_zero_grad is not None and torch.count_nonzero(theta_zero_grad).item() == 0
    checks["alpha_zero_forward_loss_gradient_optimizer_equivalence"] = dict(
        all_key_tensors_bitwise_equal=exact, old_parameter_gradients_bitwise_equal=True,
        both_original_adam_updates_bitwise_equal=True,
        alpha_initial_gradient_norm=float(alpha_grad.norm().item()),
        expected_zero_theta_gradient_at_alpha_zero=float(theta_zero_grad.norm().item()),
    )

    with contextlib.redirect_stdout(io.StringIO()):
        full.run_epoch("simple-2")
    theta_grad = full.ModalFusion.serca34.theta.grad
    assert theta_grad is not None and torch.isfinite(theta_grad).all() and theta_grad.norm().item() > 0
    checks["theta_gradient_after_alpha_update"] = float(theta_grad.norm().item())
    assert full.ModalFusion.serca34.alpha.detach().abs().max().item() > 0

    full.MessagePassing0.eval()
    full.ModalFusion.eval()
    with torch.no_grad():
        model_num, parts = full.dataLayer(full.feat)
        gfr, raw, _ = full.MessagePassing0(parts, model_num, full.targ, full.trn_idx)
        hidden, _ = full.ModalFusion.InputLayer(gfr)
        output = full.ModalFusion.serca34(hidden, raw,
                                         full.MessagePassing0.last_normalized_adjs,
                                         context["utils"].normalize_adj)
    module = full.ModalFusion.serca34
    module.collect_diagnostics = True
    with torch.no_grad():
        _ = module(hidden, raw, full.MessagePassing0.last_normalized_adjs,
                    context["utils"].normalize_adj)
    module.collect_diagnostics = False
    assert not torch.equal(hidden, output)
    diag = module.last_diagnostics
    assert diag is not None and diag["max_token_residual_ratio_observed"] <= .100001
    assert diag["max_token_residual_ratio_observed"] > 0

    from serca import build_conditioned_graphs
    similarity = torch.ones((4, 871, 871), device="cuda", dtype=torch.float32)
    centers = torch.ones(4, device="cuda", dtype=torch.float32)
    conditioned_const, differences_const = build_conditioned_graphs(
        raw, full.MessagePassing0.last_normalized_adjs, similarity, centers,
        context["utils"].normalize_adj,
    )
    stacked_base = torch.stack(full.MessagePassing0.last_normalized_adjs).unsqueeze(1).expand_as(conditioned_const)
    assert torch.equal(conditioned_const, stacked_base)
    assert torch.count_nonzero(differences_const).item() == 0
    conditioned, difference = build_conditioned_graphs(
        raw, full.MessagePassing0.last_normalized_adjs, module.similarities, module.centers,
        context["utils"].normalize_adj,
    )
    graph_rows = []
    for relation in range(4):
        adjacency = raw[relation]
        off = adjacency - torch.diag(torch.diagonal(adjacency))
        edges = int(torch.count_nonzero(off).item())
        assert edges > 0
        assert torch.all(adjacency >= 0) and torch.isfinite(adjacency).all()
        assert torch.equal(adjacency, adjacency.T)
        expected_base = context["utils"].normalize_adj(adjacency + torch.eye(871, device="cuda"))
        assert torch.equal(expected_base, full.MessagePassing0.last_normalized_adjs[relation])
        for concept in range(4):
            assert torch.allclose(conditioned[relation, concept], conditioned[relation, concept].T,
                                  rtol=0, atol=2e-7)
            assert torch.count_nonzero((conditioned[relation, concept] > 0) &
                                       (adjacency == 0) &
                                       (~torch.eye(871, device="cuda", dtype=torch.bool))).item() == 0
        graph_rows.append(dict(relation=relation, off_diagonal_edge_count=edges,
                               density=edges / (871 * 870)))
    assert torch.count_nonzero(difference).item() > 0
    checks["same_forward_graph_and_normalization_audit"] = dict(
        graph_shapes=[list(value.shape) for value in raw],
        relations=graph_rows, constant_similarity_exact_fallback=True,
        real_conditioned_difference_nonzero=True,
        phantom_off_diagonal_edges=0,
        max_observed_residual_ratio=diag["max_token_residual_ratio_observed"],
    )

    full.optimizer_MP0.zero_grad(set_to_none=True)
    for parameter in module.parameters():
        parameter.grad = None
    model_num, parts = full.dataLayer(full.feat)
    _, raw_live, _ = full.MessagePassing0(parts, model_num, full.targ, full.trn_idx)
    base_live = full.MessagePassing0.last_normalized_adjs
    with torch.no_grad():
        gfr_detached, _, _ = full.MessagePassing0(parts, model_num, full.targ, full.trn_idx)
        hidden_live, _ = full.ModalFusion.InputLayer(gfr_detached)
    residual = module(hidden_live.detach(), raw_live, base_live, context["utils"].normalize_adj) - hidden_live.detach()
    residual.square().sum().backward()
    graph_grad_norm = sum(
        float(parameter.grad.norm().item()) for name, parameter in full.MessagePassing0.named_parameters()
        if name.startswith("GraphConstruct") and parameter.grad is not None
    )
    assert np.isfinite(graph_grad_norm) and graph_grad_norm > 0
    checks["conditioned_branch_retains_raw_graph_parameter_gradient"] = graph_grad_norm

    rng_before_checkpoint = rng_capture(torch)
    saved = dict(model=capture_state(full), optimizer_mp=copy.deepcopy(full.optimizer_MP0.state_dict()),
                 optimizer_mf=copy.deepcopy(full.optimizer_MF.state_dict()), rng=rng_before_checkpoint)
    probe = STATE_DIR / f"preflight_resume_gpu{gpu}.pt"
    torch_write(probe, saved)
    disk_saved = torch_read(probe)
    rng_restore(torch, disk_saved["rng"])
    with contextlib.redirect_stdout(io.StringIO()):
        full.run_epoch("simple-2")
    expected_model = capture_state(full)
    expected_mp = copy.deepcopy(full.optimizer_MP0.state_dict())
    expected_mf = copy.deepcopy(full.optimizer_MF.state_dict())
    expected_rng = rng_capture(torch)
    load_state(full, disk_saved["model"])
    full.optimizer_MP0.load_state_dict(disk_saved["optimizer_mp"])
    full.optimizer_MF.load_state_dict(disk_saved["optimizer_mf"])
    rng_restore(torch, disk_saved["rng"])
    with contextlib.redirect_stdout(io.StringIO()):
        full.run_epoch("simple-2")
    assert recursive_equal(expected_model, capture_state(full))
    assert recursive_equal(expected_mp, full.optimizer_MP0.state_dict())
    assert recursive_equal(expected_mf, full.optimizer_MF.state_dict())
    assert rng_hash(expected_rng) == rng_hash(rng_capture(torch))
    probe.unlink()
    checks["atomic_checkpoint_both_adams_rng_next_update_exact"] = True

    test_ids = context["folds"][0][1]
    fc = context["x"][:, 64:].astype(np.float64)
    support = np.asarray(json.loads(CONCEPTS_FILE.read_text(encoding="utf-8"))["concepts"][0]["fc_feature_indices"])
    from serca import fit_similarity
    original_similarity, original_stats = fit_similarity(fc, support, context["folds"][0][0])
    changed_fc = fc.copy()
    changed_fc[test_ids, support[0]] += 2.0
    changed_similarity, changed_stats = fit_similarity(changed_fc, support, context["folds"][0][0])
    assert original_stats["tau"] == changed_stats["tau"]
    assert original_stats["center"] == changed_stats["center"]
    assert np.array_equal(original_similarity[np.ix_(context["folds"][0][0],
                                                   context["folds"][0][0])],
                          changed_similarity[np.ix_(context["folds"][0][0],
                                                   context["folds"][0][0])])
    assert not np.array_equal(original_similarity[test_ids], changed_similarity[test_ids])
    checks["label_free_train_only_similarity_scales_transductive_fc_allowed"] = True
    checks["random_concept_shape_and_overlap"] = dict(
        supports=load_prior(1, "S2_RANDOM_CONCEPTS")["stats"],
        trusted_connections=252, unknown_indices=[195, 225, 232, 236],
    )

    json_write(RUNTIME / f"preflight_gpu{gpu}.json",
               dict(passed=True, completed_utc=utc(), checks=checks))
    close_agent(base)
    close_agent(full)
    print(f"PREFLIGHT_GPU{gpu}_PASS " + json.dumps(checks, ensure_ascii=False, allow_nan=False), flush=True)


def preflight_checks(gpu):
    import torch
    assert int(os.environ.get("SERCA_PHYSICAL_GPU", -1)) == gpu
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu)
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    context = get_context()
    context["device"] = torch.device("cuda")
    e32, _, _ = history_reference_rows()
    prior = load_prior(1, "E11_SERCA_V2_FULL")
    network = context["network"]
    seed_everything(torch, 0)
    entry = rng_capture(torch)
    assert rng_hash(entry) == e32[1]["initial_rng_sha256"]
    base = make_agent("E0_B0A", 1, context, None)
    assert trunk_hash(base) == e32[1]["initial_trunk_sha256"]
    assert not hasattr(base.ModalFusion, "serca35")
    baseline_after_construct_rng = rng_capture(torch)
    baseline_initial_state = trunk_state(base)
    base_update_rng = rng_capture(torch)
    base_capture = capture_one_update(base, context)
    baseline_updated_state = trunk_state(base)
    baseline_gradients = base_capture["gradients"]
    baseline_mp = optimizer_base_state(base.optimizer_MP0)
    baseline_mf = optimizer_base_state(base.optimizer_MF)

    network._prompt_text_features = None
    rng_restore(torch, entry)
    full = make_agent("E11_SERCA_V2_FULL", 1, context, prior)
    module = full.ModalFusion.serca35
    assert trunk_hash(full) == e32[1]["initial_trunk_sha256"]
    assert rng_hash(rng_capture(torch)) == rng_hash(baseline_after_construct_rng)
    assert recursive_equal(trunk_state(full), baseline_initial_state)
    assert torch.count_nonzero(module.alpha).item() == 0
    assert len(full.optimizer_MF.param_groups) == 2
    assert full.optimizer_MF.param_groups[-1]["weight_decay"] == 0.0
    rng_restore(torch, base_update_rng)
    full_capture = capture_one_update(full, context)
    exact_keys = ("data_layer", "gfr_input", "gfr_output", "input_layer", "encoder_0",
                  "feedforward_0", "output_layer_input", "output_layer_output", "raw_adjs",
                  "normalized_adjs", "graph_losses", "cls_loss")
    equal = {key: recursive_equal(base_capture[key], full_capture[key]) for key in exact_keys}
    assert all(equal.values()), {key: key_equal for key, key_equal in equal.items() if not key_equal}
    assert recursive_equal(baseline_gradients, full_capture["gradients"])
    assert recursive_equal(baseline_updated_state, trunk_state(full))
    assert recursive_equal(baseline_mp, optimizer_base_state(full.optimizer_MP0))
    assert recursive_equal(baseline_mf, optimizer_base_state(full.optimizer_MF))
    assert torch.equal(full_capture["input_layer"][0], full_capture["serca_output"])
    alpha_grad = module.alpha.grad
    assert alpha_grad is not None and torch.isfinite(alpha_grad).all() and alpha_grad.norm().item() > 0

    with contextlib.redirect_stdout(io.StringIO()):
        full.run_epoch("simple-2")
    nonzero_grad = {
        name: float(parameter.grad.norm().item())
        for name, parameter in module.named_parameters()
        if parameter.grad is not None and parameter.grad.norm().item() > 0
    }
    assert any(name.startswith("router.") for name in nonzero_grad)
    assert any(name.startswith("value_projections.") for name in nonzero_grad)
    assert module.alpha.detach().abs().max().item() > 0

    module.collect_diagnostics = True
    with torch.no_grad():
        eval_metrics = evaluate_fold(full, context, 1, collect=True)
    diag = module.last_diagnostics
    module.collect_diagnostics = False
    assert diag and diag["semantic_edge_count"] > 0
    assert diag["semantic_row_normalization_error"] and max(diag["semantic_row_normalization_error"]) < 1e-6
    assert diag["max_token_residual_ratio_observed"] <= 0.050001
    assert diag["raw_message_rms"] > 0
    assert np.isfinite(eval_metrics["acc"])

    original_alpha = module.alpha.detach().clone()
    original_state = capture_state(full)
    module.source_override = [True, True, True, True]
    module.alpha.data.fill_(10.0)
    model_num, parts = full.dataLayer(full.feat)
    gfr, raw, _ = full.MessagePassing0(parts, model_num, full.targ, full.trn_idx)
    hidden, _ = full.ModalFusion.InputLayer(gfr)
    with torch.no_grad():
        null_output = module(hidden, dict(raw_inputs=full.feat, raw_adjs=raw,
                                          train_indices=full.trn_idx))
    assert torch.equal(hidden, null_output), "null route did not disable all semantic messages"
    module.source_override = None
    module.alpha.data.fill_(100.0)
    module.collect_diagnostics = True
    with torch.no_grad():
        capped = module(hidden, dict(raw_inputs=full.feat, raw_adjs=raw,
                                     train_indices=full.trn_idx))
    assert module.last_diagnostics["max_token_residual_ratio_observed"] <= 0.050001
    assert torch.isfinite(capped).all()
    load_state(full, original_state)
    module.direct_message = not module.direct_message
    module.direct_message = not module.direct_message

    cache_checks = {}
    for config in ("E2_NUMERIC_GLOBAL", "E4_ALL_PAIR", "E5_RANDOM_RULE",
                   "E6_PATIENT_SHUFFLED", "E7_ORIGINAL_EDGES", "E11_SERCA_V2_FULL"):
        item = load_prior(1, config)
        cache_checks[config] = dict(
            kernel_std=item["stats"]["kernel_std"],
            signature_rank=item["stats"]["signature_train_rank"],
            semantic_edges=int(np.count_nonzero(item["semantic_neighbors"])),
        )
        assert item["stats"]["kernel_std"] > 0
    assert not np.array_equal(load_prior(1, "E11_SERCA_V2_FULL")["semantic_kernel"],
                              load_prior(1, "E5_RANDOM_RULE")["semantic_kernel"])
    assert np.count_nonzero(load_prior(1, "E7_ORIGINAL_EDGES")["semantic_neighbors"]) == 0
    assert load_prior(1, "E6_PATIENT_SHUFFLED")["cache_meta"]["groups"]["E6_PATIENT_SHUFFLED"][
        "patient_shuffle"]["no_cross_train_test_mapping"]

    save_point = dict(model=capture_state(full), optimizer_mp=copy.deepcopy(full.optimizer_MP0.state_dict()),
                      optimizer_mf=copy.deepcopy(full.optimizer_MF.state_dict()), rng=rng_capture(torch))
    probe = STATE_DIR / f"preflight_resume_gpu{gpu}.pt"
    torch_write(probe, save_point)
    disk_point = torch_read(probe)
    with contextlib.redirect_stdout(io.StringIO()):
        full.run_epoch("simple-2")
    expected_model = capture_state(full)
    expected_mp = copy.deepcopy(full.optimizer_MP0.state_dict())
    expected_mf = copy.deepcopy(full.optimizer_MF.state_dict())
    expected_rng = rng_capture(torch)
    load_state(full, disk_point["model"])
    full.optimizer_MP0.load_state_dict(disk_point["optimizer_mp"])
    full.optimizer_MF.load_state_dict(disk_point["optimizer_mf"])
    rng_restore(torch, disk_point["rng"])
    with contextlib.redirect_stdout(io.StringIO()):
        full.run_epoch("simple-2")
    assert recursive_equal(expected_model, capture_state(full))
    assert recursive_equal(expected_mp, full.optimizer_MP0.state_dict())
    assert recursive_equal(expected_mf, full.optimizer_MF.state_dict())
    assert rng_hash(expected_rng) == rng_hash(rng_capture(torch))
    probe.unlink()
    checks = dict(
        gpu=gpu, device=torch.cuda.get_device_name(0), fold1_rng_and_trunk_match=True,
        alpha_zero_forward_loss_gradients_and_two_original_adam_updates_equal=equal,
        alpha_initial_gradient_norm=float(alpha_grad.norm().item()),
        active_parameter_gradients_after_updates=nonzero_grad,
        alpha_zero_hidden_exact=True, null_route_exact_exit=True,
        max_residual_ratio=diag["max_token_residual_ratio_observed"],
        raw_message_rms=diag["raw_message_rms"],
        cache_non_degeneracy=cache_checks,
        heldout_label_invariance=load_prior(1, "E11_SERCA_V2_FULL")["cache_meta"][
            "heldout_label_invariance_checked"],
        patient_shuffle_fold_boundary_passed=True,
        checkpoint_resume_next_update_exact=True,
        historical_clip_cache_shape=list(base.ModalFusion.prompt_text_features.shape),
        original_pen_constructed_and_forward_bypassed=True,
        test_fold_preview_acc=eval_metrics["acc"],
    )
    json_write(RUNTIME / f"preflight_gpu{gpu}.json", dict(passed=True, completed_utc=utc(), checks=checks))
    close_agent(base)
    close_agent(full)
    print(f"PREFLIGHT_GPU{gpu}_PASS " + json.dumps(checks, ensure_ascii=False, allow_nan=False), flush=True)


def run_preflight():
    assert MANIFEST.is_file(), "run prepare first"
    procs = []
    for gpu in (0, 1):
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           SERCA_PHYSICAL_GPU=str(gpu), OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        log_path = LOG_DIR / f"preflight_gpu{gpu}.log"
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-B", str(EXP / "run.py"), "_preflight", str(gpu)],
            cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
        )
        procs.append((gpu, process, log))
    failures = []
    for gpu, process, log in procs:
        code = process.wait()
        log.close()
        if code:
            failures.append((gpu, code, (LOG_DIR / f"preflight_gpu{gpu}.log").read_text(
                encoding="utf-8", errors="replace")[-5000:]))
    if failures:
        raise RuntimeError(f"SeRCA preflight failed: {failures}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["preflight"] = dict(
        passed=True, completed_utc=utc(),
        devices=[json.loads((RUNTIME / f"preflight_gpu{gpu}.json").read_text(encoding="utf-8"))
                 for gpu in (0, 1)],
    )
    json_write(MANIFEST, manifest)
    print("SERCA_PREFLIGHT_BOTH_GPUS_PASS", flush=True)


def compare_s0_to_history(rows):
    e32, e33, e34 = history_reference_rows()
    s0 = {int(row["fold"]): row for row in rows if row["config"] == "E0_B0A"}
    assert set(s0) == set(e32) == set(e33) == set(e34) == set(range(1, 11))
    counts = dict(initial_parameter_hash_exact=0, initial_rng_hash_exact=0,
                  best_epoch_exact=0, four_metrics_exact=0)
    for fold in range(1, 11):
        row, old32, old33, old34 = s0[fold], e32[fold], e33[fold], e34[fold]
        if row["initial_trunk_sha256"] == old32["initial_trunk_sha256"] == old33["initial_trunk_sha256"] == old34["initial_trunk_sha256"]:
            counts["initial_parameter_hash_exact"] += 1
        if row["initial_rng_sha256"] == old32["initial_rng_sha256"] == old33["initial_rng_sha256"] == old34["initial_rng_sha256"]:
            counts["initial_rng_hash_exact"] += 1
        if int(row["best_epoch"]) == int(old32["best_epoch"]) == int(old33["best_epoch"]) == int(old34["best_epoch"]):
            counts["best_epoch_exact"] += 1
        exact = all(
            abs(float(row[current]) - float(old[reference])) <= 1e-12
            for old, auc_name in ((old32, "auc"), (old33, "hard_auc"), (old34, "hard_auc"))
            for current, reference in (("acc", "acc"), ("hard_auc", auc_name),
                                       ("sen_asd", "sen_asd"), ("spe_td", "spe_td"))
        )
        if exact:
            counts["four_metrics_exact"] += 1
    return counts


def verify_s0():
    state = torch_read(task_path("E0_B0A"))
    assert state["phase"] == "COMPLETE" and len(state["rows"]) == 10
    comparison = compare_s0_to_history(state["rows"])
    if any(value != 10 for value in comparison.values()):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest.update(status="E0_PARITY_FAILED", completed_folds=10,
                        s0_historical_reproduction=comparison, updated_utc=utc())
        json_write(MANIFEST, manifest)
        raise RuntimeError(f"E0 does not exactly reproduce historical Experiment 32/33/34: {comparison}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest.update(status="E0_VERIFIED", completed_folds=10,
                    s0_historical_reproduction=comparison, s0_verified_utc=utc())
    json_write(MANIFEST, manifest)
    return comparison


def worker_environment(gpu):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       SERCA_PHYSICAL_GPU=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    return environment


def launch_configs(configs):
    running, pending, failures = {}, {gpu: [] for gpu in (0, 1)}, []
    for config in configs:
        pending[GPU_OF[config]].append(config)
    launch_time = time.monotonic()

    def start(config, gpu):
        log_path = LOG_DIR / f"{config}.controller.log"
        stream = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-B", str(EXP / "run.py"), "_worker", config, str(gpu)],
            cwd=ROOT, env=worker_environment(gpu), stdout=stream, stderr=subprocess.STDOUT,
        )
        running[process] = (config, gpu, stream)
        record(f"worker started {config} GPU{gpu}; active={sum(v[1] == gpu for v in running.values())}")

    for gpu in (0, 1):
        for _ in range(min(WORKERS_PER_GPU, len(pending[gpu]))):
            start(pending[gpu].pop(0), gpu)
    while running:
        completed = []
        for process, (config, gpu, stream) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            completed.append(process)
            if code:
                failures.append((config, gpu, code))
                save_status(config, gpu, "FAILED", error_code=code)
                record(f"worker failed {config} GPU{gpu}, exit={code}")
            else:
                record(f"worker exited {config} GPU{gpu}, elapsed={time.monotonic()-launch_time:.1f}s")
        for process in completed:
            _, gpu, _ = running.pop(process)
            if process.returncode == 0 and pending[gpu] and not any(f[1] == gpu for f in failures):
                start(pending[gpu].pop(0), gpu)
        if not completed:
            time.sleep(5)
    return failures


def launch_workers():
    assert MANIFEST.is_file(), "run prepare first"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest.get("preflight", {}).get("passed") is True, "both-GPU preflight must pass first"
    assert manifest["source_sha256"] == source_hashes() and manifest["input_sha256"] == input_hashes()
    lock = (RUNTIME / "controller.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = utc()
    controller_start = time.monotonic()
    manifest.update(status="RUNNING_E0", formal_started_utc=started)
    json_write(MANIFEST, manifest)
    s0_failure = launch_configs(["E0_B0A"])
    if s0_failure:
        raise RuntimeError(f"E0 worker failed; checkpoint remains resumable: {s0_failure}")
    comparison = verify_s0()
    record(f"E0 passed exact historical parity: {comparison}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest.update(status="RUNNING_E1_E11", other_groups_started_utc=utc())
    json_write(MANIFEST, manifest)
    rest = [config for config in CONFIGS if config != "E0_B0A"]
    failures = launch_configs(rest)
    if failures:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest.update(status="INCOMPLETE_RESUMABLE", worker_failures=failures,
                        elapsed_seconds=time.monotonic()-controller_start)
        json_write(MANIFEST, manifest)
        raise RuntimeError(f"workers failed; completed folds remain resumable: {failures}")
    result = aggregate_and_report(controller_start)
    return result


def summarize(rows):
    result = {}
    for config in CONFIGS:
        group = [row for row in rows if row["config"] == config]
        assert len(group) == 10 and {int(row["fold"]) for row in group} == set(range(1, 11))
        result[config] = {}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            result[config][metric] = dict(mean=float(values.mean()), population_sd=float(values.std(ddof=0)))
    return result


def paired_comparisons(rows):
    lookup = {(row["config"], int(row["fold"])): row for row in rows}
    comparisons = {}
    for control in CONFIGS[:-1]:
        metrics = {}
        for metric in METRICS:
            differences = [float(lookup[("E11_SERCA_V2_FULL", fold)][metric]) -
                           float(lookup[(control, fold)][metric]) for fold in range(1, 11)]
            metrics[metric] = dict(
                mean_delta=float(np.mean(differences)),
                mean_delta_percentage_points=float(np.mean(differences) * 100.0),
                fold_deltas=differences,
                wins=sum(value > 1e-12 for value in differences),
                ties=sum(abs(value) <= 1e-12 for value in differences),
                losses=sum(value < -1e-12 for value in differences),
            )
        comparisons[control] = metrics
    return comparisons


def decision_from(summary, comparisons, diagnostics):
    e0, full = summary["E0_B0A"], summary["E11_SERCA_V2_FULL"]
    def beats(control):
        return (comparisons[control]["acc"]["mean_delta"] > 0 and
                comparisons[control]["hard_auc"]["mean_delta"] > 0)
    conditions = {
        "e11_acc_and_hard_auc_above_e0": full["acc"]["mean"] > e0["acc"]["mean"] and
                                         full["hard_auc"]["mean"] > e0["hard_auc"]["mean"],
        "e11_acc_gain_at_least_0.23pp": 100 * (full["acc"]["mean"] - e0["acc"]["mean"]) >= .23,
        "e11_beats_e1_calibrated_e34": beats("E1_SERCA34_CAL"),
        "e11_beats_e2_numeric_global": beats("E2_NUMERIC_GLOBAL"),
        "e11_beats_e3_first_order": beats("E3_FIRST_ORDER"),
        "e11_beats_e4_all_pair_unstructured": beats("E4_ALL_PAIR"),
        "e11_beats_e5_random_rule": beats("E5_RANDOM_RULE"),
        "e11_beats_e6_patient_shuffled": beats("E6_PATIENT_SHUFFLED"),
        "e11_beats_e7_original_edges": beats("E7_ORIGINAL_EDGES"),
        "e11_beats_e8_corr_only": beats("E8_CORR_ONLY"),
        "e11_beats_e9_aux_only": beats("E9_AUX_ONLY"),
        "e11_beats_e10_direct_matched": beats("E10_DIRECT_MATCHED"),
        "sen_asd_drop_within_0.50pp": 100 * comparisons["E0_B0A"]["sen_asd"]["mean_delta"] >= -SEN_NONINFERIORITY_PP,
        "nonzero_semantic_residual": any(
            any(abs(value) > 0 for value in json.loads(row["alpha_json"])) and
            float(row["residual_ratio_max_observed"]) > 0
            for row in diagnostics if row["config"] == "E11_SERCA_V2_FULL"),
        "residual_within_5pct_cap": all(
            float(row["residual_ratio_max_observed"]) <= .050001
            for row in diagnostics if row["config"] == "E11_SERCA_V2_FULL"),
        "same_checkpoint_counterfactuals_recorded": all(
            "same_checkpoint_counterfactual_json" in row
            for row in diagnostics if row["config"] == "E11_SERCA_V2_FULL"),
    }
    semantic = all(conditions[key] for key in (
        "e11_beats_e2_numeric_global", "e11_beats_e4_all_pair_unstructured",
        "e11_beats_e5_random_rule", "e11_beats_e6_patient_shuffled"))
    cross = conditions["e11_beats_e8_corr_only"] and conditions["e11_beats_e9_aux_only"]
    if not conditions["e11_acc_and_hard_auc_above_e0"]:
        status = "NO-GO"
    elif not semantic:
        status = "NUMERIC_GAIN_ONLY — SEMANTIC_CLAIM_NOT_SUPPORTED"
    elif not cross:
        status = "SEMANTIC_RELATION_GAIN_ONLY"
    elif all(conditions.values()):
        status = "PROVISIONAL_GO — MULTISEED_AND_INDEPENDENT_VALIDATION_REQUIRED"
    else:
        status = "NO-GO"
    return dict(status=status, conditions=conditions,
                failed_conditions=[key for key, value in conditions.items() if not value],
                semantic_controls_passed=semantic,
                cross_modal_complementarity_demonstrated=cross,
                single_seed_internal_screen_only=True, statistical_significance_claimed=False,
                interpretation=("The registered semantic-rule, patient-correspondence and cross-source screens passed; external and multiseed validation remain required."
                                if status.startswith("PROVISIONAL_GO") else
                                "The frozen method was evaluated as registered. The results do not satisfy all internal criteria for the claimed semantic and cross-source mechanism."))


def percent_summary(value):
    return f"{100*value['mean']:.4f} ± {100*value['population_sd']:.4f}%"


def _legacy_write_report_e34(rows, diagnostics, summary, comparisons, decision, manifest):
    concepts = json.loads(CONCEPTS_FILE.read_text(encoding="utf-8"))
    e32, e33 = history_reference_rows()
    s0_rows = {int(row["fold"]): row for row in rows if row["config"] == "S0_B0A"}
    parity = manifest["s0_historical_reproduction"]
    lines = [
        "# Experiment 34 — SeRCA: Semantic-guided Relational Complementary Alignment", "",
        f"Status: **{manifest['status']}**. Completed {manifest['completed_folds']}/100 folds; GPU0 {manifest['gpu_completed_folds']['0']}/50, GPU1 {manifest['gpu_completed_folds']['1']}/50. Total elapsed wall time: {manifest['elapsed_wall_seconds']:.1f}s ({manifest['elapsed_wall_seconds']/3600:.2f}h); summed worker time: {manifest['summed_worker_seconds']:.1f}s. Full worker timing and RNG evidence are in `manifest.json` and `.runtime/`.", "",
        "## Research Question and Audit", "",
        "This experiment tests whether frozen GPT-4 disease-functional-network concepts, constrained by the previously verified HO FC feature identities and medical sources, can condition the four already learned GFR subject-relation matrices. SeRCA operates on between-subject relations; it does not treat those matrices as ROI graphs, does not force modality feature equality, and does not use labels to create patient similarities.", "",
        "The MICCAI 2025 SPromptGL paper reports 92.30% ABIDE accuracy for its full method. The local code is the implementation authority here: its four GraphLearn outputs are patient-to-patient matrices thresholded at cosine >0.9, GFR propagates the full 320D concatenated feature through each matrix normalized with the repository `normalize_adj(A+I)`, and the `i==1` branch overwrites the `i==0` `combined_feat`. PLN is constructed in its original order, but the B0-A forward bypasses PEN and passes CAM output directly to the unchanged OutputLayer. The original `simple-2` objective is `ClsLoss + mean(four GraphConstructLoss)`, while computed ALoss is not in backward. CAM `[871,4,36]` contains mixed latent positions; four 9D slices are not four anatomical ROIs.", "",
        "Paper Eq. (1)-(7) and Figure 2 describe the broader SPromptGL components but do not resolve these local implementation details. The paper's 92.30% result is not the local re-trained B0-A result. Current Experiment 32/33 B0-A (92.0742% ACC) is the baseline reference. Experiments 11 and 19 supply the verified numeric FC identity map and the frozen disease-concept support sets. Experiments 12-18, 20-31 show repeated failures or weak evidence for free Prompt keys, forced feature alignment, semantic graph calibration, highly similar CLIP category vectors, or QC evidence; Experiment 32 found incorrect Prompt controls were not reliably worse, and Experiment 33 found wrong pairing and FC-only controls could beat its proposed QC condition.", "",
        "Related work already includes semantic-conditioned brain interactions (KD-Brain), concept weighting/graphs (MMGPL), multimodal patient graph and shared/private learning (MGDR), conditional dependency modeling (InfoBridge), and evidence that forced alignment can help or hurt by task (To Align or Not to Align). SeRCA's narrow test is whether a frozen, anatomy-supported FC concept changes existing GFR edges and contributes a difference-message through the existing CAM path; semantic guidance, concepts, graph alignment, and complementarity are not claimed as novel by themselves.", "",
        "## Frozen Concepts and Operator", "",
        "| Concept | FC support |", "|---|---:|"]
    for concept in concepts["concepts"]:
        lines.append(f"| {concept['id']} — {concept['name']} | {concept['support_count']} |")
    lines += ["", "D1-D4 overlaps (shared FC features) are " + ", ".join(
        f"{key}={value}" for key, value in concepts["concept_overlap"].items()) +
        ". Features 195, 225, 232 and 236 remain in the original B0-A CORR input but are excluded from semantic concepts because verified anatomical endpoint names are unavailable. Concept membership is candidate measurement support, not evidence that every ASD participant has a lesion or a common FC direction. Random concepts use one local PCG64 bijection on the 252 verified feature identities, preserving each support size and all pairwise overlaps, but not anatomical topology.", "",
        "For each concept, signed FC columns are standardized with outer-train means/scales. The concept similarity is `S^k_ij=exp(-||Z_i^k-Z_j^k||²/tau_k)`, where `tau_k` is the median positive outer-train pair distance and `center_k` is its outer-train similarity mean. These matrices are fixed by fold, label-free, and transductive over all 871 participants; held-out FC values may affect pair similarities, but not fitted scales.", "",
        "For existing raw GFR adjacency `A_m`, SeRCA forms `W_mk=diag(A_m)+offdiag(A_m)⊙exp(S^k-center_k)`, preserving the diagonal and existing off-diagonal support, then `B_mk=normalize_adj(W_mk+I)` using the root utility. The ordinary `B_m=normalize_adj(A_m+I)` is captured from the same GFR forward. With CAM input `H` immediately after the existing InputLayer, SeRCA pools `R[t,m,k]=(B_mk-B_m)@H[:,t,:]` using a low-capacity softmax over relation/concept coefficients. Its bounded residual is `Delta[t]=0.10*tanh(alpha_t)*sum(pi[t,m,k]R[t,m,k])`, capped per subject/token at 10% of `||H||`; `alpha` starts at zero. No graph is learned separately, no GFR graph loss changed, and no CLIP text encoding was added.", "",
        "## Frozen Groups and Protocol", "",
        "| Group | Registered contrast | GPU |", "|---|---|---:|"]
    group_labels = {
        "S0_B0A": "B0-A, no SeRCA intervention",
        "S1_NUMERIC_GLOBAL": "one 256D numeric FC kernel copied into four concept slots",
        "S2_RANDOM_CONCEPTS": "fixed size/overlap matched random FC feature identities",
        "S3_SINGLE_CONCEPT": "D1 only",
        "S4_CORR_REL_ONLY": "four concepts, CORR relation only",
        "S5_AUX_REL_ONLY": "four concepts, PHENO/ANAT/FUNC relation sources only",
        "S6_UNIFORM_RELATION": "fixed uniform relation/concept weights; theta frozen",
        "S7_DIRECT_MESSAGE": "conditioned neighborhood message without subtracting base message",
        "S8_PATIENT_PERMUTED": "within-site FC concept patient bijection; original input/GFR unchanged",
        "S9_SERCA_FULL": "registered full SeRCA primary model",
    }
    for config in CONFIGS:
        lines.append(f"| {config} | {group_labels[config]} | {GPU_OF[config]} |")
    lines += ["", "All ten groups ran one seed-0, ten-fold stratified CV over the existing standardized/RFE-selected 871×320 data, labels ASD=0 (403), TD=1 (468), and all-subject transductive GFR. Splits use `StratifiedKFold(10, shuffle=True, random_state=10)`. The protocol is `nepoch=450`, `early=200`, `lr=.0038`, `weight_decay=.11`, `dropout=.35`, `n_hidden=18`, `n_head=2`, `nlayer=1`, `weighted-cosine`, `GCN`, `concat`, `simple-2`. `val_idx=test_idx`; strict test/validation ACC improvement replaces the checkpoint and ties preserve the first. Hard-AUC uses the original hard one-hot predictions. The four formal metrics are ACC, Hard-AUC, SEN-ASD (ASD positive), SPE-TD.", "",
        "S0 ran first as a continuous seed-0 fold stream and recorded Python, NumPy, PyTorch CPU and CUDA entry states before each model construction. S1-S9 each ran an independent continuous ten-fold worker but restored the matching S0 fold-entry RNG before each construction. This is a paired construction control; it is not natural continuous cross-fold RNG for those nine groups. Both original Adam optimizers and losses were preserved; new parameters use one additional PLN optimizer group at lr=.0038 and weight decay 0. S6 freezes theta and has four trainable alpha values rather than 68 trainable alpha/theta elements. Checkpoints include model, best model, both Adams, wait/epoch and RNG states and are written atomically.", "",
        "## Four-Metric Results", "",
        "Mean ± population SD across ten folds, in percent:", "",
        "| Group | ACC | Hard-AUC | SEN-ASD | SPE-TD |", "|---|---:|---:|---:|---:|"]
    for config in CONFIGS:
        lines.append("| " + config + " | " + " | ".join(
            percent_summary(summary[config][metric]) for metric in METRICS) + " |")
    lines += ["", "The complete 100 fold rows, best epoch, hard-confusion counts, initialization/RNG hashes and all same-fold deltas are in `fold_metrics.csv`.", "",
        "## Paired Comparisons", "",
        "Every delta is S9 minus its same-fold control. Mean differences are percentage points; W/T/L counts are shown for ACC and Hard-AUC. Each metric row includes all ten fold deltas in fold order 1→10. The counts are descriptive, not inferential statistics.", "",
        "| Contrast | Metric | Mean Δ (pp) | W/T/L | Fold deltas (pp, folds 1→10) |", "|---|---|---:|---:|---|"]
    for control in CONFIGS[:-1]:
        for metric, label in zip(METRICS, ("ACC", "Hard-AUC", "SEN-ASD", "SPE-TD")):
            item = comparisons[control][metric]
            deltas = ", ".join(f"{100*value:+.4f}" for value in item["fold_deltas"])
            wtl = f"{item['wins']}/{item['ties']}/{item['losses']}" if metric in ("acc", "hard_auc") else "—"
            lines.append(f"| S9 − {control} | {label} | {item['mean_delta_percentage_points']:+.4f} | {wtl} | {deltas} |")
    lines += ["", "## B0-A Reproduction", "",
        f"The new S0 versus Experiment 32 and Experiment 33 matches initial trunk hashes on {parity['initial_parameter_hash_exact']}/10 folds, pre-construction RNG hashes on {parity['initial_rng_hash_exact']}/10, best epochs on {parity['best_epoch_exact']}/10, and all four metrics on {parity['four_metrics_exact']}/10. No prior checkpoint was read or loaded. Historical Experiment 32 B0-A has ACC=92.0742%, Hard-AUC=91.8546%, SEN-ASD=89.0793%, SPE-TD=94.6300%. The MICCAI paper full-model result is 92.30% and is not the same comparison.", "",
        "| Fold | E32 ACC | S0 ACC | Δ pp | E32 Hard-AUC | S0 Hard-AUC | Δ pp | E32 SEN | S0 SEN | E32 SPE | S0 SPE |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for fold in range(1, 11):
        row, old = s0_rows[fold], e32[fold]
        lines.append(f"| {fold} | {100*float(old['acc']):.4f} | {100*row['acc']:.4f} | {100*(row['acc']-float(old['acc'])):+.4f} | {100*float(old['auc']):.4f} | {100*row['hard_auc']:.4f} | {100*(row['hard_auc']-float(old['auc'])):+.4f} | {100*float(old['sen_asd']):.4f} | {100*row['sen_asd']:.4f} | {100*float(old['spe_td']):.4f} | {100*row['spe_td']:.4f} |")
    lines += ["", "## Mechanism Audit", "",
        "Fold-level graph density, edge reweighting magnitude, similarities/distances, learned alpha/theta weights, relation contributions, residual fraction and patient permutation evidence are in `diagnostics.csv`. S9 uses one semantic prior source (CORR FC) to alter relation weights; PHENO/ANAT/FUNC are population/QC covariates, not independent disease imaging measurements. We do not infer useful complementarity merely from nonzero coefficients: the paired S9-vs-S4 results test whether auxiliary relation sources improve classification, and S9-vs-S5 tests whether CORR is needed. S9-vs-S7 tests whether subtracting the ordinary message adds value. S9-vs-S1/S2/S8 tests whether correct medical concepts and patient correspondence matter beyond numeric propagation.", "",
        "Diagnostic summaries must be interpreted alongside the paired accuracy results. If S9 only matches S4, the four-relation complementarity claim is not established. If S9 fails against S1/S2/S8, the medical semantic necessity claim is not established even if S9 beats S0. Correct concept supports are candidate literature-grounded measurements and do not prove patient-specific pathology.", "",
        f"## Decision: {decision['status']}", "",
        decision["interpretation"], "",
        "The internal screen is single-seed and not a significance test. This inherited protocol uses `val=test` for checkpoint selection, preprocessing/RFE before cross-validation and transductive graphs over all 871 participants; its CV results are not unbiased independent-test generalization evidence. A positive screen remains provisional and requires multiseed and independent-site validation.", "",
        "## Reproducibility and Files", "",
        "`manifest.json` records protocol, group/GPU assignment, original/upstream source hashes, data and history hashes, concept/prior generation hashes, S0 parity, preflight evidence, worker timing, completion counts and decisions. `integration.patch` shows the isolated runtime-only source edits; `.runtime/source/` holds their exact source copies and a symlink to the original CLIP package. RNG entry snapshots are retained. Epoch checkpoints and derived similarity caches are removed after successful verification to keep the final artifact compact; the preparation code regenerates caches deterministically. No root model, data or historical experiment was modified and no Git action was performed.", "",
        "## Fold Metrics", "",
        "All 100 folds with ACC, Hard-AUC, SEN-ASD, SPE-TD, confusion counts, best epoch, both hashes and fold-paired deltas are in `fold_metrics.csv`.", ""]
    path = RESULTS / "report.md"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_report(rows, diagnostics, summary, comparisons, decision, manifest):
    prior = json.loads(PRIOR_FILE.read_text(encoding="utf-8"))
    labels = {
        "E0_B0A": "Fresh B0-A baseline",
        "E1_SERCA34_CAL": "Experiment34 four single-concept kernels with calibrated E35 messages",
        "E2_NUMERIC_GLOBAL": "All-256-FC numeric relation, fixed projection only for router inputs",
        "E3_FIRST_ORDER": "D1-D4 signed first-order deviations, no pair products",
        "E4_ALL_PAIR": "All six pair interactions without semantic rule selection",
        "E5_RANDOM_RULE": "Degree-multiset-matched wrong pair rules",
        "E6_PATIENT_SHUFFLED": "Correct rules, train/test/site-constrained patient shuffle",
        "E7_ORIGINAL_EDGES": "Correct rules; no new semantic Top-k neighbors",
        "E8_CORR_ONLY": "Correct rules; CORR relation source only",
        "E9_AUX_ONLY": "Correct rules; PHENO/ANAT/FUNC sources only",
        "E10_DIRECT_MATCHED": "Correct rules; direct messages with same train-RMS calibration and cap",
        "E11_SERCA_V2_FULL": "Registered SeRCA-v2 full model",
    }
    parity = manifest["s0_historical_reproduction"]
    e11_diag = [row for row in diagnostics if row["config"] == "E11_SERCA_V2_FULL"]
    def mean_json(field):
        values = [np.asarray(json.loads(row[field]), dtype=np.float64)
                  for row in e11_diag if row.get(field)]
        return np.mean(np.stack(values), axis=0).tolist() if values else []
    e11_alpha = mean_json("alpha_json")
    e11_route = mean_json("route_mean_json")
    e11_source_contrib = mean_json("source_contribution_rms_json")
    e11_cap_multiplier = mean_json("residual_ratio_mean_json")
    e11_fold_max_residual_ratio = float(np.mean([
        float(row["residual_ratio_max_observed"]) for row in e11_diag
    ])) if e11_diag else float("nan")
    e11_worst_residual_ratio = max(
        (float(row["residual_ratio_max_observed"]) for row in e11_diag), default=float("nan"))
    def total_confusion(config):
        group = [row for row in rows if row["config"] == config]
        return {key: sum(int(row[key]) for row in group)
                for key in ("asd_tp", "asd_fn", "td_fp", "td_tn")}
    e0_conf, e10_conf, e11_conf = (total_confusion(config) for config in
                                    ("E0_B0A", "E10_DIRECT_MATCHED", "E11_SERCA_V2_FULL"))
    e0_correct = e0_conf["asd_tp"] + e0_conf["td_tn"]
    e10_correct = e10_conf["asd_tp"] + e10_conf["td_tn"]
    e11_correct = e11_conf["asd_tp"] + e11_conf["td_tn"]
    e11_direct = [json.loads(row["same_checkpoint_counterfactual_json"])
                  for row in e11_diag if row.get("same_checkpoint_counterfactual_json")]
    direct_changes = float(np.mean([item["direct_same_checkpoint"]["changed_fraction"]
                                    for item in e11_direct])) if e11_direct else float("nan")
    direct_rms = float(np.mean([item["direct_same_checkpoint"]["calibrated_message_rms"]
                                for item in e11_direct])) if e11_direct else float("nan")
    source_cf = {}
    for source in RELATION_NAMES:
        entries = [item["source_ablations_same_checkpoint"][source] for item in e11_direct]
        source_cf[source] = dict(
            changed_fraction=float(np.mean([entry["changed_fraction"] for entry in entries])) if entries else None,
            acc=float(np.mean([entry["metrics"]["acc"] for entry in entries])) if entries else None,
            hard_auc=float(np.mean([entry["metrics"]["hard_auc"] for entry in entries])) if entries else None,
        )
    lines = [
        "# Experiment 35 — SeRCA-v2: Disease-Semantic Relational Pattern Alignment", "",
        f"Status: **{manifest['status']}**; {manifest['completed_folds']}/120 folds completed. GPU0 {manifest['gpu_completed_folds']['0']}/60 and GPU1 {manifest['gpu_completed_folds']['1']}/60. Wall time {manifest['elapsed_wall_seconds']:.1f}s ({manifest['elapsed_wall_seconds']/3600:.2f} h); summed worker time {manifest['summed_worker_seconds']:.1f}s.", "",
        "## Question and Audit", "",
        "The experiment asks whether frozen disease-domain concepts can define useful patient-level joint-deviation relations beyond ordinary FC similarity, and whether those relations improve messages from multiple GFR sources. Pair rules are a conservative human synthesis from the frozen Experiment 19 prompt wording and cited studies. The retained Experiment 19 package does not include a raw GPT-4 response transcript or exact model version, so no new pair rule is attributed verbatim to GPT-4.", "",
        "Source audit preserved the local code behavior: four GraphLearn outputs are participant-to-participant relations, thresholded at cosine >0.9 and returned raw; GFR uses `normalize_adj(A+I)` and full 320D concatenated features, with the `i==1` overwrite of `i==0`; original PLN construction and CLIP cache lifecycle stay intact; B0-A bypasses PEN forward. The `simple-2` backward objective remains the classification loss plus mean raw-graph losses; ALoss is not optimized. The CAM `[871,4,36]` positions are mixed latent features, not four original modalities or ROIs. PHENO contains age/site/sex encodings; ANAT6 and FUNC10 are scan-quality features.", "",
        "The MICCAI 2025 paper's full-model 92.30% is distinct from the local fresh B0-A 92.0742%. Related work already covers concept weighting/token graphs (MMGPL), multimodal graph disentanglement (MGDR), semantic-conditioned brain interactions/pathology constraints (KD-Brain), conditional dependency modeling (InfoBridge), and task-dependent alignment (To Align or Not to Align). This run tests a narrow co-deviation rule and message intervention; those ingredients alone are not claimed as novel.", "",
        "## Frozen Medical Prior", "",
        "The two retained undirected rules are D2–D4 and D3–D4. D2–D4 is motivated by ASD coactivation-pattern work finding less frequent midcingulo-insular with central-executive/lateral-frontoparietal configurations. D3–D4 is motivated by thalamocortical/sensory, insular and salience associations with sensory responsivity. These studies motivate hypotheses, not exact feature identities or universal ASD effects. The observed mapped set overlaps are D2–D4=5 features and D3–D4=2; none is interpreted as a direct cross-network FC edge. The feature counts are D1=72, D2=56, D3=45, D4=32; unknown feature identities 195, 225, 232, 236 remain in the B0-A input and are excluded from the semantic branch. Full source URLs, prompt text, limitations, and all feature indices are in `medical_prior.json`.", "",
        "## Method", "",
        "For each outer fold, a Ridge(alpha=10) FC reference is fitted only on outer-train TD using PHENO age/site/sex one-hot values. Train TD predictions are 5-fold OOF; train ASD and held-out patients use the full outer-train-TD fit. OOF-TD residual scales set signed z values, clipped to [-3,3]. Each domain has positive and negative deviations above |z|=1. Four pairwise sign products are residualized against the eight first-order scores using outer-train rows only and then train-standardized. FULL uses the eight channels for D2–D4 and D3–D4. E4 uses all six pairs; E5 uses fixed wrong pairs with the same pair count and degree multiset.", "",
        "The semantic kernel uses a train-only median squared-distance scale. Directed Top-8 recipients connect only to outer-train sources; the added branch row-normalizes independently of the original symmetric GFR normalization. The same GFR raw matrices are reused from the same forward. Each message source uses a lightweight modality-specific Linear+LayerNorm to 36 dimensions. A low-capacity router chooses null or PHENO/ANAT/FUNC/CORR. The full difference is `T_sem@V - T_base@V`. Detached outer-train RMS scaling is capped at 10x when messages are tiny; `0.05*tanh(alpha)` starts at zero and each token residual is capped at 5% of its input norm. E10 direct-message calibration uses the same rule; a same-checkpoint direct counterfactual is also saved for E11.", "",
        "## Groups and Protocol", "",
        "All groups use one seed, ten StratifiedKFold splits (`shuffle=True`, `random_state=10`), 871 subjects (ASD=403 label 0, TD=468 label 1), the existing globally preprocessed/RFE-selected 320 inputs, all-node transductive original GFR, `nepoch=450`, `early=200`, `lr=.0038`, `weight_decay=.11`, `dropout=.35`, `n_hidden=18`, `n_head=2`, and the historical `simple-2` loss. `val_idx=test_idx`; strict ACC improvement selects a checkpoint and ties keep the first. Hard-AUC uses original hard one-hot predictions. The four formal metrics are ACC, Hard-AUC, SEN-ASD, and SPE-TD.", "",
        "| Group | Frozen contrast | GPU |", "|---|---|---:|"]
    for config in CONFIGS:
        lines.append(f"| {config} | {labels[config]} | {GPU_OF[config]} |")
    lines += ["", "E0 preserves the natural continuous seed-0 fold stream and stores each pre-construction Python/NumPy/torch CPU/CUDA state. E1-E11 restore the corresponding E0 fold-entry state before constructing each fold. This pairs initialization, not later training trajectories. Every worker has atomic epoch/fold-boundary checkpoints containing both original Adams, model parameters, best state, early-stop state and RNG; completed folds are committed once.", "",
        "## Four Metrics", "", "Mean ± population SD across folds, percent:", "",
        "| Group | ACC | Hard-AUC | SEN-ASD | SPE-TD |", "|---|---:|---:|---:|---:|"]
    for config in CONFIGS:
        lines.append("| " + config + " | " + " | ".join(percent_summary(summary[config][metric])
                                                        for metric in METRICS) + " |")
    lines += ["", "`fold_metrics.csv` contains all 120 folds, confusion counts, best epochs, initialization/RNG hashes and E0 deltas.", "",
        "## Paired Comparisons", "", "Each result is E11 minus its same-fold control; deltas are percentage points. W/T/L are fold wins/ties/losses for ACC and Hard-AUC. These ten overlapping CV fits are descriptive, not independent samples.", "",
        "| Control | Metric | Mean Δ pp | W/T/L | Fold deltas 1–10 (pp) |", "|---|---|---:|---:|---|"]
    for control in CONFIGS[:-1]:
        for metric, label in zip(METRICS, ("ACC", "Hard-AUC", "SEN-ASD", "SPE-TD")):
            item = comparisons[control][metric]
            wtl = f"{item['wins']}/{item['ties']}/{item['losses']}" if metric in ("acc", "hard_auc") else "—"
            deltas = ", ".join(f"{100*value:+.4f}" for value in item["fold_deltas"])
            lines.append(f"| E11 − {control} | {label} | {item['mean_delta_percentage_points']:+.4f} | {wtl} | {deltas} |")
    lines += ["", "## B0-A and Mechanisms", "",
        f"E0 matches Experiments 32, 33 and staged Experiment 34 on initial trunk hash {parity['initial_parameter_hash_exact']}/10, pre-construction RNG {parity['initial_rng_hash_exact']}/10, best epoch {parity['best_epoch_exact']}/10, and all four metrics {parity['four_metrics_exact']}/10. It reads no historical checkpoint.",
        f"Across 871 held-out predictions, E0 has {e0_correct} correct (ASD FN={e0_conf['asd_fn']}, TD FP={e0_conf['td_fp']}); E11 has {e11_correct} correct (ASD FN={e11_conf['asd_fn']}, TD FP={e11_conf['td_fp']}), a net change of {e11_correct-e0_correct} correct, {e11_conf['asd_fn']-e0_conf['asd_fn']:+d} ASD FN and {e11_conf['td_fp']-e0_conf['td_fp']:+d} TD FP. E10 has {e10_correct} correct (ASD FN={e10_conf['asd_fn']}, TD FP={e10_conf['td_fp']}).", "",
        f"For E11, mean alpha by CAM position is `{e11_alpha}`; mean route weights (null, PHENO, ANAT, FUNC, CORR) are `{e11_route}`; mean source contribution RMS is `{e11_source_contrib}`. The per-token vector `{e11_cap_multiplier}` contains mean residual-cap multipliers, not residual/input ratios. The mean across folds of the per-fold maximum actual residual/input norm ratio is {100*e11_fold_max_residual_ratio:.4f}%, and the worst observed value is {100*e11_worst_residual_ratio:.7f}% (floating-point tolerance at the 5% cap). Same-checkpoint direct-message predictions changed on {100*direct_changes:.2f}% of test subjects on average after matching train RMS and using the same 5% cap; calibrated direct RMS was {direct_rms:.6g}. Per-source same-checkpoint ablations are summarized here: `{json.dumps(source_cf, ensure_ascii=False)}`. Full graph overlap, per-fold scales, station similarity, message contributions, and route details are in `diagnostics.csv`.", "",
        "A nonzero gate or residual is not proof of predictive complementarity. The decisive evidence is E11 versus E2/E4/E5/E6 for semantic necessity, E11 versus E8/E9 for source complementarity, E11 versus E10 for difference messaging, and the fixed-checkpoint source ablations. PHENO is demographics, ANAT/FUNC are QC; neither is an independent disease imaging modality.", "",
        f"## Decision: {decision['status']}", "",
        decision["interpretation"], "",
        "This is a single-seed internal screen, not a significance test or SOTA claim. The inherited protocol selects on `val=test`, performs RFE/preprocessing before outer CV, and uses transductive graphs; it cannot establish unbiased independent-site generalization.", "",
        "## Reproducibility and Delivery", "",
        "`manifest.json` records source/data/history hashes, frozen configuration, priors, caches, RNG pairing, per-worker timing, completion and decision. `integration.patch` and `.runtime/source/` record the isolated source edits. Fold prior caches and epoch checkpoints are removed only after successful aggregation; deterministic code and retained RNG-entry snapshots regenerate the inputs. The intended final `/data` path and root `handoff.md` append status are recorded in the manifest; the working artifact is staged under `/tmp` because the OSSFS experiment directory returned I/O error.", ""]
    target = RESULTS / "report.md"
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def aggregate_and_report(controller_start=None):
    rows, diagnostics = [], []
    for config in CONFIGS:
        state = torch_read(task_path(config))
        assert state["phase"] == "COMPLETE" and len(state["rows"]) == 10
        rows.extend(state["rows"])
        diagnostics.extend(state["diagnostics"])
    assert len(rows) == len(diagnostics) == 120
    assert len({(row["config"], int(row["fold"])) for row in rows}) == 120
    for row in rows:
        row["historical_exp32_b0a_acc"] = float(e32_value(int(row["fold"]), "acc"))
    s0_parity = verify_s0()
    for row in rows:
        fold = int(row["fold"])
        baseline = next(item for item in rows if item["config"] == "E0_B0A" and int(item["fold"]) == fold)
        for metric in METRICS:
            row["delta_vs_e0_" + metric] = float(row[metric]) - float(baseline[metric])
    rows.sort(key=lambda row: (CONFIGS.index(row["config"]), int(row["fold"])))
    diagnostics.sort(key=lambda row: (CONFIGS.index(row["config"]), int(row["fold"])))
    csv_write(RESULTS / "fold_metrics.csv", rows)
    csv_write(RESULTS / "diagnostics.csv", diagnostics)
    summary = summarize(rows)
    comparisons = paired_comparisons(rows)
    decision = decision_from(summary, comparisons, diagnostics)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    workers = {}
    for config in CONFIGS:
        task = torch_read(task_path(config))
        worker = json.loads(task_status_path(config).read_text(encoding="utf-8"))
        workers[config] = worker
        worker["worker_elapsed_seconds"] = float(task["seconds"])
        worker["clip_calls"] = task.get("clip_calls", {})
        worker["started_utc"] = task.get("started_utc")
        worker["completed_utc"] = task.get("completed_utc")
    gpu_fold_counts = {str(gpu): sum(1 for row in rows if int(row["physical_gpu"]) == gpu)
                       for gpu in (0, 1)}
    gpu_intervals = {}
    for gpu in (0, 1):
        on_gpu = [workers[config] for config in CONFIGS if GPU_OF[config] == gpu]
        started = [dt.datetime.fromisoformat(item["started_utc"]).timestamp()
                   for item in on_gpu if item.get("started_utc")]
        completed = [dt.datetime.fromisoformat(item["completed_utc"]).timestamp()
                     for item in on_gpu if item.get("completed_utc")]
        gpu_intervals[str(gpu)] = dict(
            folds=gpu_fold_counts[str(gpu)],
            summed_worker_seconds=sum(float(item.get("worker_elapsed_seconds", 0)) for item in on_gpu),
            wall_seconds=max(completed) - min(started) if started and completed else None,
        )
    total_wall = (time.monotonic() - controller_start) if controller_start is not None else None
    if total_wall is None:
        starts = [dt.datetime.fromisoformat(worker["started_utc"]).timestamp() for worker in workers.values()]
        ends = [dt.datetime.fromisoformat(worker["completed_utc"]).timestamp() for worker in workers.values()]
        total_wall = max(ends) - min(starts)
    manifest.update(
        status="COMPLETE", completed_utc=utc(), completed_folds=120,
        gpu_completed_folds=gpu_fold_counts, gpu_timing=gpu_intervals,
        workers=workers, elapsed_wall_seconds=float(total_wall),
        summed_worker_seconds=sum(item["worker_elapsed_seconds"] for item in workers.values()),
        s0_historical_reproduction=s0_parity,
        summary=summary, paired_comparisons=comparisons, decision=decision,
        no_historical_checkpoint_reused=True,
        active_gpu_worker_limit=WORKERS_PER_GPU,
    )
    write_report(rows, diagnostics, summary, comparisons, decision, manifest)
    json_write(MANIFEST, manifest)
    for path in STATE_DIR.glob("*.pt"):
        path.unlink()
    for path in PRIOR_DIR.glob("fold_*.npz"):
        path.unlink()
    manifest["cleanup"] = dict(epoch_and_boundary_checkpoints_removed=True,
                               deterministic_prior_npz_cache_removed=True,
                               retained_fold_rng_snapshots=10,
                               retained_worker_logs=True)
    json_write(MANIFEST, manifest)
    return manifest


def e32_value(fold, metric):
    with (EXP32 / "results/fold_metrics.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["config"] == "B0" and int(row["fold"]) == fold:
                return float(row[metric])
    raise KeyError(f"Experiment 32 B0 fold {fold} missing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run", "aggregate",
                                             "_worker", "_preflight"))
    parser.add_argument("config", nargs="?")
    parser.add_argument("gpu", nargs="?", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare()
        print(f"PREPARED {manifest['target_folds']} folds in {EXP}", flush=True)
    elif args.command == "preflight":
        run_preflight()
    elif args.command == "run":
        result = launch_workers()
        print(f"COMPLETE {result['completed_folds']}/120; decision={result['decision']['status']}", flush=True)
    elif args.command == "aggregate":
        result = aggregate_and_report()
        print(f"AGGREGATED {result['completed_folds']}/120", flush=True)
    elif args.command == "_preflight":
        gpu = args.gpu if args.gpu is not None else int(args.config)
        preflight_checks(gpu)
    elif args.command == "_worker":
        assert args.config is not None and args.gpu is not None
        run_worker(args.config, args.gpu)


if __name__ == "__main__":
    main()

