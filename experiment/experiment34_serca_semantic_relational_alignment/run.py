#!/usr/bin/env python3
"""Prepare, preflight, run and audit the 100-fold SeRCA experiment."""
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
CONCEPTS_FILE = EXP / "concepts.json"
CORE_FILES = ("main.py", "model.py", "network.py", "layers.py", "utils.py",
              "ABIDE-simple-2-concat-weighted-cosine.sh")
SOURCE_NAMES = CORE_FILES[:5]
DATA_FILES = ("data/ABIDE/processed_standard_data.csv", "data/ABIDE/modal_feat_dict.npy")
EXP11 = ROOT / "experiment/experiment11_roi_mapping_recovery/results"
EXP19 = ROOT / "experiment/experiment19_contrastive_disease_evidence_alignment"
EXP32 = ROOT / "experiment/experiment32_disease_contrastive_evidence_routing"
EXP33 = ROOT / "experiment/experiment33_qcar_conditional_alignment"
AUDIT = ROOT / "experiment/b0_pen_physical_removal_equivalence_audit"
CONFIGS = (
    "S0_B0A", "S1_NUMERIC_GLOBAL", "S2_RANDOM_CONCEPTS", "S3_SINGLE_CONCEPT",
    "S4_CORR_REL_ONLY", "S5_AUX_REL_ONLY", "S6_UNIFORM_RELATION",
    "S7_DIRECT_MESSAGE", "S8_PATIENT_PERMUTED", "S9_SERCA_FULL",
)
GPU_CONFIGS = {
    0: ("S0_B0A", "S2_RANDOM_CONCEPTS", "S4_CORR_REL_ONLY",
        "S6_UNIFORM_RELATION", "S8_PATIENT_PERMUTED"),
    1: ("S1_NUMERIC_GLOBAL", "S3_SINGLE_CONCEPT", "S5_AUX_REL_ONLY",
        "S7_DIRECT_MESSAGE", "S9_SERCA_FULL"),
}
GPU_OF = {name: gpu for gpu, configs in GPU_CONFIGS.items() for name in configs}
INPUT_DIMS = (48, 6, 10, 256)
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
RANDOM_CONCEPT_SEED = 3402026
PATIENT_PERMUTATION_SEED = 3408000
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
        "            x = self.serca34(x, serca_payload['raw_adjs'],\n"
        "                             serca_payload['base_adjs'], serca_payload['normalize_adj'])\n", 1,
    )
    bypass_marker = "        text_features = self.prompt_text_features\n"
    bypass = (
        "        if getattr(self, 'serca34_bypass_pen', False):\n"
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
    can_refresh = False
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
    with (EXP19 / "prompts.json").open(encoding="utf-8") as stream:
        prompts = json.load(stream)
    with (EXP19 / "domain_mapping.csv").open(newline="", encoding="utf-8") as stream:
        domain_rows = list(csv.DictReader(stream))
    with (EXP11 / "roi_256_mapping.csv").open(newline="", encoding="utf-8") as stream:
        numeric_rows = {int(row["correlation_feature_index"]): row for row in csv.DictReader(stream)}
    with (EXP11 / "roi_256_named_mapping.csv").open(newline="", encoding="utf-8") as stream:
        named_rows = {int(row["correlation_feature_index"]): row for row in csv.DictReader(stream)}
    assert len(numeric_rows) == len(named_rows) == 256
    assert set(numeric_rows) == set(named_rows) == set(range(256))
    assert len({int(row["original_connection_index"]) for row in numeric_rows.values()}) == 256
    assert len({int(row["original_connection_index"]) for row in named_rows.values()}) == 256
    unknown = sorted(set(range(256)) - set(int(row["correlation_feature_index"]) for row in domain_rows))
    assert unknown == [195, 225, 232, 236]
    assert all(row["verification_status"] == "VERIFIED_NUMERIC_UNIQUE" for row in numeric_rows.values())
    assert all(row["usable_for_roi_semantic_prompt"].lower() == "true"
               for index, row in named_rows.items() if index not in unknown)
    assert all(named_rows[index]["usable_for_roi_semantic_prompt"].lower() != "true" for index in unknown)
    assert len(domain_rows) == len({int(row["correlation_feature_index"]) for row in domain_rows}) == 252
    features = {name: set() for name in ("D1", "D2", "D3", "D4")}
    connection_records = {}
    for row in domain_rows:
        feature = int(row["correlation_feature_index"])
        assert row["verified"].lower() == "true" and feature in named_rows
        named = named_rows[feature]
        assert named["roi_i_name_verified"].lower() == "true"
        assert named["roi_j_name_verified"].lower() == "true"
        assert named["connection_anatomical_status"] == "VERIFIED_BOTH_ENDPOINTS"
        assert named["usable_for_roi_semantic_prompt"].lower() == "true"
        assert int(row["original_connection_index"]) == int(numeric_rows[feature]["original_connection_index"])
        memberships = []
        for concept in features:
            assert row[concept] in ("0", "1")
            if row[concept] == "1":
                features[concept].add(feature)
                memberships.append(concept)
        connection_records[str(feature)] = dict(
            original_connection_index=int(row["original_connection_index"]),
            roi_i_zero_based=int(row["roi_i_zero_based"]),
            roi_j_zero_based=int(row["roi_j_zero_based"]),
            roi_i_name=named["roi_i_name_verified"],
            roi_j_name=named["roi_j_name_verified"],
            connection_name=named["connection_name"],
            concepts=memberships,
        )
    assert [len(features[f"D{i}"]) for i in range(1, 5)] == [72, 56, 45, 32]
    trusted = np.asarray(sorted(set(range(256)) - set(unknown)), dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(RANDOM_CONCEPT_SEED))
    shuffled = rng.permutation(trusted)
    global_map = {int(old): int(new) for old, new in zip(trusted, shuffled)}
    random_features = {key: sorted(global_map[index] for index in values)
                       for key, values in features.items()}
    for key in features:
        assert len(random_features[key]) == len(features[key])
        assert 195 not in random_features[key] and 225 not in random_features[key]
        assert 232 not in random_features[key] and 236 not in random_features[key]
    overlap = {f"{a}_{b}": len(features[a] & features[b]) for a, b in
               (("D1", "D2"), ("D1", "D3"), ("D1", "D4"),
                ("D2", "D3"), ("D2", "D4"), ("D3", "D4"))}
    random_overlap = {f"{a}_{b}": len(set(random_features[a]) & set(random_features[b])) for a, b in
                      (("D1", "D2"), ("D1", "D3"), ("D1", "D4"),
                       ("D2", "D3"), ("D2", "D4"), ("D3", "D4"))}
    assert overlap == random_overlap
    order = prompts["domain_order"]
    concepts = []
    for index, key in enumerate(order):
        prompt_pair = prompts["banks"]["REAL"][index]
        concepts.append(dict(
            id=key, name=CONCEPT_NAMES[index],
            prompt_asd=prompt_pair[0], prompt_td=prompt_pair[1],
            fc_feature_indices=sorted(features[key]),
            support_count=len(features[key]),
            overlaps={other: len(features[key] & features[other])
                      for other in features if other != key},
            medical_source=prompts["medical_sources"][index],
        ))
    result = dict(
        class_prompts_used_for_subject_routing=False,
        disease_prompt_source="Experiment 19 prompts.json; frozen unchanged",
        domain_mapping_source="Experiment 19 domain_mapping.csv",
        numeric_mapping_source="Experiment 11 roi_256_mapping.csv",
        named_mapping_source="Experiment 11 roi_256_named_mapping.csv",
        source_sha256={
            "prompts.json": sha_file(EXP19 / "prompts.json"),
            "domain_mapping.csv": sha_file(EXP19 / "domain_mapping.csv"),
            "roi_256_mapping.csv": sha_file(EXP11 / "roi_256_mapping.csv"),
            "roi_256_named_mapping.csv": sha_file(EXP11 / "roi_256_named_mapping.csv"),
        },
        fc_feature_order="existing 256-column order; no RFE or mapping rebuild",
        trusted_feature_count=len(trusted), unknown_excluded_feature_indices=unknown,
        concepts=concepts, verified_connections=connection_records,
        concept_overlap=overlap,
        random_concepts=dict(
            seed=RANDOM_CONCEPT_SEED, generator="local numpy.random.PCG64",
            trusted_index_permutation=shuffled.tolist(),
            fc_feature_indices=random_features, overlap=random_overlap,
            preserves_set_sizes_and_overlap=True,
            preserves_anatomical_topology=False,
            unknown_feature_indices_excluded=True,
        ),
        patient_permutation=dict(
            generator="local numpy.random.PCG64",
            fold_seed_formula=f"{PATIENT_PERMUTATION_SEED} + fold_1based",
            grouping="all 871 participants within existing 20-way one-hot site ID",
            singleton_site_members_remain_fixed=True,
            permutation_is_global_bijection_each_fold=True,
        ),
    )
    json_write(CONCEPTS_FILE, result)
    return result


def pairwise_similarity(fc, support, train):
    if str(EXP) not in sys.path:
        sys.path.insert(0, str(EXP))
    from serca import fit_similarity
    return fit_similarity(fc, np.asarray(support, dtype=np.int64), train)


def make_fold_cache(x, folds, concepts):
    PRIOR_DIR.mkdir(parents=True, exist_ok=True)
    real_supports = [concept["fc_feature_indices"] for concept in concepts["concepts"]]
    random_supports = [concepts["random_concepts"]["fc_feature_indices"][f"D{i}"]
                       for i in range(1, 5)]
    fc = x[:, 64:].astype(np.float64)
    sites = x[:, 26:46].argmax(axis=1)
    cache_hashes, fold_meta = {}, {}
    for fold, (train, _) in enumerate(folds, start=1):
        real, random_sims, patient_sims = [], [], []
        real_stats, random_stats, patient_stats = [], [], []
        for supports, sim_out, stats_out in (
            (real_supports, real, real_stats),
            (random_supports, random_sims, random_stats),
        ):
            for support in supports:
                similarity, stats = pairwise_similarity(fc, support, train)
                sim_out.append(similarity)
                stats_out.append(stats)
        rng = np.random.Generator(np.random.PCG64(PATIENT_PERMUTATION_SEED + fold))
        permutation = np.arange(len(sites), dtype=np.int64)
        singleton = 0
        for site in np.unique(sites):
            ids = np.flatnonzero(sites == site)
            if len(ids) <= 1:
                singleton += len(ids)
                continue
            permutation[ids] = rng.permutation(ids)
        assert np.array_equal(np.sort(permutation), np.arange(len(sites)))
        changed = int(np.count_nonzero(permutation != np.arange(len(sites))))
        permuted_fc = fc[permutation]
        for support in real_supports:
            similarity, stats = pairwise_similarity(permuted_fc, support, train)
            patient_sims.append(similarity)
            patient_stats.append(stats)
        numeric, numeric_stats = pairwise_similarity(fc, np.arange(256), train)
        meta = dict(
            fold=fold, train_count=int(len(train)),
            train_indices_sha256=sha_bytes(np.asarray(train, dtype=np.int64).tobytes()),
            real_stats=real_stats, random_stats=random_stats,
            patient_permuted_stats=patient_stats, numeric_stats=numeric_stats,
            patient_permutation_sha256=sha_bytes(permutation.tobytes()),
            patient_permutation_changed_count=changed,
            patient_permutation_changed_fraction=float(changed / len(sites)),
            patient_permutation_site_singletons=singleton,
            patient_permutation_bijection=True,
            concept_labels_read=False,
        )
        path = PRIOR_DIR / f"fold_{fold:02d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                assert saved["metadata"].item() == json.dumps(meta, sort_keys=True, ensure_ascii=False)
                for key, arr in (("real", real), ("random", random_sims),
                                 ("patient", patient_sims), ("numeric", numeric)):
                    assert np.array_equal(saved[key], np.stack(arr) if isinstance(arr, list) else arr)
        else:
            tmp = path.with_suffix(".npz.tmp")
            with tmp.open("wb") as stream:
                np.savez_compressed(
                    stream, real=np.stack(real), random=np.stack(random_sims),
                    patient=np.stack(patient_sims), numeric=numeric,
                    metadata=np.asarray(json.dumps(meta, sort_keys=True, ensure_ascii=False)),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        cache_hashes[str(fold)] = sha_file(path)
        fold_meta[str(fold)] = meta
    return cache_hashes, fold_meta


def group_spec(config):
    relations = [0, 1, 2, 3]
    concepts = [0, 1, 2, 3]
    if config == "S3_SINGLE_CONCEPT":
        concepts = [0]
    elif config == "S4_CORR_REL_ONLY":
        relations = [3]
    elif config == "S5_AUX_REL_ONLY":
        relations = [0, 1, 2]
    prior = "real"
    if config == "S1_NUMERIC_GLOBAL":
        prior = "numeric_replicated"
    elif config == "S2_RANDOM_CONCEPTS":
        prior = "random"
    elif config == "S8_PATIENT_PERMUTED":
        prior = "patient_permuted"
    active_theta = 4 * len(relations) * len(concepts) if config != "S6_UNIFORM_RELATION" else 0
    return dict(prior=prior, relation_indices=relations, concept_indices=concepts,
                difference_message=config not in ("S0_B0A", "S7_DIRECT_MESSAGE"),
                learned_relation_weights=config not in ("S0_B0A", "S6_UNIFORM_RELATION"),
                active_parameter_tensors=0 if config == "S0_B0A" else
                (1 if config == "S6_UNIFORM_RELATION" else 2),
                active_parameter_elements=0 if config == "S0_B0A" else
                (4 if config == "S6_UNIFORM_RELATION" else 4 + active_theta),
                allocated_trainable_parameter_elements=0 if config == "S0_B0A" else
                (4 if config == "S6_UNIFORM_RELATION" else 68))


def load_prior(fold, config):
    spec = group_spec(config)
    if config == "S0_B0A":
        return None
    with np.load(PRIOR_DIR / f"fold_{fold:02d}.npz", allow_pickle=False) as cache:
        meta = json.loads(cache["metadata"].item())
        if spec["prior"] == "real":
            similarities, stats = cache["real"].copy(), meta["real_stats"]
        elif spec["prior"] == "random":
            similarities, stats = cache["random"].copy(), meta["random_stats"]
        elif spec["prior"] == "patient_permuted":
            similarities, stats = cache["patient"].copy(), meta["patient_permuted_stats"]
        else:
            numeric, stats0 = cache["numeric"].copy(), meta["numeric_stats"]
            similarities = np.repeat(numeric[None, :, :], 4, axis=0)
            stats = [dict(**stats0, copied_into_concept_slot=k + 1) for k in range(4)]
    centers = np.asarray([item["center"] for item in stats], dtype=np.float32)
    return dict(
        similarities=similarities.astype(np.float32), centers=centers,
        relation_mask=np.asarray([i in spec["relation_indices"] for i in range(4)], dtype=bool),
        concept_mask=np.asarray([i in spec["concept_indices"] for i in range(4)], dtype=bool),
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
        ("README.md", "serca.py", "integration.py", "integration.patch", "run.py")},
        inputs=input_hashes(), history=history_hashes(), protocol=PROTOCOL,
        cache_hashes=json.loads(MANIFEST.read_text(encoding="utf-8"))["prior_cache_sha256"])
    return canonical_hash(payload)


def history_reference_rows():
    with (EXP32 / "results/fold_metrics.csv").open(newline="", encoding="utf-8") as stream:
        e32 = {int(row["fold"]): row for row in csv.DictReader(stream) if row["config"] == "B0"}
    with (EXP33 / "results/fold_metrics.csv").open(newline="", encoding="utf-8") as stream:
        e33 = {int(row["fold"]): row for row in csv.DictReader(stream) if row["config"] == "G0_B0A"}
    assert set(e32) == set(e33) == set(range(1, 11))
    for fold in range(1, 11):
        for metric in ("acc", "sen_asd", "spe_td"):
            assert abs(float(e32[fold][metric]) - float(e33[fold][metric])) <= 1e-12
        assert abs(float(e32[fold]["auc"]) - float(e33[fold]["hard_auc"])) <= 1e-12
        assert e32[fold]["initial_trunk_sha256"] == e33[fold]["initial_trunk_sha256"]
        assert e32[fold]["initial_rng_sha256"] == e33[fold]["initial_rng_sha256"]
        assert int(e32[fold]["best_epoch"]) == int(e33[fold]["best_epoch"])
    return e32, e33


def prepare():
    for directory in (RUNTIME, SOURCE, PRIOR_DIR, RNG_DIR, STATE_DIR, LOG_DIR, TMP_DIR, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    assert (Path.home() / ".cache/clip/RN50.pt").is_file(), "cached RN50 checkpoint missing; no download"
    concepts = build_concepts()
    patch = emit_sources()
    x, y, folds, mapping = read_data()
    cache_hashes, fold_meta = make_fold_cache(x, folds, concepts)
    e32, e33 = history_reference_rows()
    handoff = ROOT / "handoff.md"
    manifest = dict(
        experiment="Experiment 34 — SeRCA Semantic-guided Relational Complementary Alignment",
        status="PREPARED", created_utc=utc(), seed=0, configs=list(CONFIGS),
        target_folds=100, completed_folds=0, source_sha256=source_hashes(),
        input_sha256=input_hashes(), historical_sha256=history_hashes(),
        code_sha256={name: sha_file(EXP / name) for name in
                     ("README.md", "serca.py", "integration.py", "integration.patch", "run.py", "concepts.json")},
        runtime_source_sha256={name: sha_file(SOURCE / name) for name in SOURCE_NAMES},
        integration_patch_sha256=sha_bytes(patch.encode()),
        prior_cache_sha256=cache_hashes,
        prior_fold_statistics=fold_meta,
        handoff_prefix=dict(bytes=handoff.stat().st_size, sha256=sha_file(handoff)),
        current_git=dict(head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                         origin_main=subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()),
        upstream_author_core_sha256=dict(
            main="dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2",
            model="254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda",
            network="d0ac59ead2fecdf517845e2c389d361ef9924abff29707386f847e1fbaa22166",
            layers="4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a",
            utils="25afd06c4a46f278aeab2eef166b930bdfab53b901956c30e03bb3b499688001",
        ),
        paper_audit=dict(
            paper_url="https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf",
            author_repository="https://github.com/wanxixi11/SPromptGL",
            current_repository="https://github.com/tangaii/SPromptGL",
            actual_code="GFR is four between-subject graphs; GraphLearn returns raw thresholded cosine; GFR applies normalize_adj(A+I), propagates full 320D concatenated features and preserves i=1 overwrite; PLN is InputLayer->Encoder->FeedForward->PEN/OutputLayer; current simple-2 optimizes ClsLoss plus four raw-adjacency GraphConstructLoss mean and does not optimize ALoss.",
            paper_code_difference="Paper Eq.(1) includes a label graph loss absent from optimized local simple-2; paper Eq.(2) describes multi-relation fusion but local GFR i=1 assignment overwrites i=0; local shell hyperparameters differ from paper ABIDE table. Current checkout and Experiment32 protocol govern reproduction.",
            baseline_p0="Construct original PLN in original order including inactive PEN MLP, cached CLIP prompt buffer and AvgPool; bypass only PEN forward. CAM output is [N,4,36] mixed latent positions; four 9D blocks are not ROIs.",
            ablation_paper="Figure 2 reports baseline/GFR/RES/full PEN; largest claimed gain is GFR. Ablation does not isolate semantic text necessity.",
            author_vs_current="The five current checkout core source hashes above match origin/main. Upstream author network.py and utils.py have distinct SHA-256 from the current repository; all training uses the locally audited Experiment32 current checkout, not reconstructed author code."),
        related_work=dict(
            MMGPL="GPT-4 concepts weight patches and define a token graph inside neuroimaging; SeRCA does not claim concept weighting or semantic graphs as novel.",
            MGDR="Per-modality patient graphs plus shared/private disentanglement and contrastive learning; supports complementarity as a known goal.",
            KDBrain="Semantic-conditioned attention queries plus pathology-consistent KL over subnetworks; disease-guided interactions and clinical priors are prior art.",
            InfoBridge="Conditional MI with protective margin for sharing and preserving modality-specific detail; different objective and domain.",
            ToAlign="Directly tests that forced representation alignment can help or hurt depending on task; motivates aligning messages without forcing modality feature equality."),
        protocol=PROTOCOL,
        data=dict(shape=list(x.shape), labels={"ASD_0": int(np.sum(y == 0)), "TD_1": int(np.sum(y == 1))},
                  modality_order=list(mapping), modality_dimensions=INPUT_DIMS,
                  pheno_ranges={"age": [0, 26], "site": [26, 46], "sex": [46, 48]},
                  train_sizes=[int(len(tr)) for tr, _ in folds],
                  test_sizes=[int(len(te)) for _, te in folds]),
        concepts=dict(ids=["D1", "D2", "D3", "D4"],
                      support_counts=[c["support_count"] for c in concepts["concepts"]],
                      overlaps=concepts["concept_overlap"], unknown_excluded=concepts["unknown_excluded_feature_indices"],
                      random_support_size_and_overlap_preserved=True,
                      patient_permutation_labels_not_used=True),
        method=dict(
            location="after PLN InputLayer and before Encoder",
            raw_graph_source="same GFR forward raw GraphLearn matrices; no second GraphLearn call",
            graph_reweight="W_mk=diag(A_m)+offdiag(A_m)*exp(1.0*(S_k-center_k)); B_mk=repository.normalize_adj(W_mk+I)",
            base_graph="B_m=repository.normalize_adj(A_m+I), reused from the same GFR forward",
            message="R_t=sum_mk softmax(theta[t,:,:])_mk * (B_mk-B_m) @ H[:,t,:]",
            residual="Delta_t=0.10*tanh(alpha_t)*R_t; per-patient token L2 cap=0.10*||H_i,t||2; alpha starts at zero",
            branch_parameters="alpha[4], theta[4,4,4]; Adam PLN group, lr=0.0038, weight_decay=0; S6 freezes theta and uses exact uniform weights",
            original_graph_loss="unchanged GraphConstructLoss on the four original raw A_m matrices",
            labels_in_prior=False, clip_text_encoder_calls_added=0,
            trainable_parameter_elements_by_group={key: group_spec(key)["active_parameter_elements"] for key in CONFIGS},
        ),
        groups={key: group_spec(key) for key in CONFIGS},
        gpu_assignment={str(gpu): list(names) for gpu, names in GPU_CONFIGS.items()},
        worker_protocol="S0 runs first as one continuous 10-fold seed0 worker and snapshots full Python/NumPy/torch CPU/CUDA RNG before each construction. Each S1-S9 worker runs its own 10 folds, restoring the paired S0 fold-entry RNG; it does not claim natural cross-fold RNG continuity.",
        worker_limit_per_gpu=WORKERS_PER_GPU,
        checkpoint=dict(epoch_interval=CHECKPOINT_EVERY, atomic=True,
                        includes_gfr_pln_serca_both_adams_best_state_wait_epoch_rng_fold_entry_rng=True,
                        completed_fold_rows_committed_once=True),
        frozen_internal_screen=dict(
            s9_acc_and_hard_auc_above_s0=True, s9_acc_gain_min_pp=.23,
            s9_vs_s1_s2_s3_s4_s5_s6_s7_s8_acc_and_auc_mean_deltas_positive=True,
            s9_vs_key_semantic_controls_acc_and_auc_wins_exceed_losses=True,
            sen_noninferiority_drop_max_pp=SEN_NONINFERIORITY_PP,
            all_conditions_are_single_seed_internal_screen_not_significance=True),
        baseline_reference=dict(experiment32_b0_fold_metrics_sha256=sha_file(EXP32 / "results/fold_metrics.csv"),
                                experiment33_b0_fold_metrics_sha256=sha_file(EXP33 / "results/fold_metrics.csv"),
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
            if module_name == "pln" and name.startswith("serca34."):
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
    context["integration"].configure(
        config=config,
        similarities=None if prior is None else prior["similarities"],
        centers=None if prior is None else prior["centers"],
        relation_mask=None if prior is None else prior["relation_mask"],
        concept_mask=None if prior is None else prior["concept_mask"],
    )
    hp = SimpleNamespace(**PROTOCOL)
    with contextlib.redirect_stdout(io.StringIO()):
        agent = context["model"].EvalHelper(INPUT_DIMS, context["x"], context["y"], hp, train, test)
    assert np.array_equal(agent.val_idx, agent.tst_idx)
    assert agent.ModalFusion.serca34_bypass_pen is True
    assert (hasattr(agent.ModalFusion, "serca34") == (config != "S0_B0A"))
    ids = optimizer_ids(agent)
    assert len(ids) == len(set(ids)) and set(ids) == parameter_ids(agent)
    assert all(abs(group["lr"] - PROTOCOL["lr"]) < 1e-12
               for optimizer in (agent.optimizer_MP0, agent.optimizer_MF)
               for group in optimizer.param_groups)
    if config != "S0_B0A":
        assert agent.optimizer_MF.param_groups[-1]["weight_decay"] == 0.0
        assert agent.optimizer_MF.param_groups[-1]["params"] == [
            parameter for parameter in agent.ModalFusion.serca34.parameters() if parameter.requires_grad]
    return agent


def evaluate_fold(agent, context, fold, collect=False):
    module = getattr(agent.ModalFusion, "serca34", None)
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
    return metrics


def fold_diagnostics(agent, context, fold, config, prior, gpu):
    torch = context["torch"]
    module = getattr(agent.ModalFusion, "serca34", None)
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
        row.update(alpha_json="[]", relation_weights_json="[]", concept_weights_json="[]",
                   residual_ratio_mean_json="[]", residual_ratio_max_json="[]",
                   relation_message_contribution_norm_json="[]", relation_similarity_mean_json="[]",
                   relation_graph_offdiag_edges_json=json.dumps(raw_edges),
                   relation_graph_density_json=json.dumps(densities),
                   relation_graph_changed_edge_fraction_json="[]", relation_graph_relative_edge_change_json="[]",
                   relation_graph_changed_edge_weight_fraction_json="[]",
                   normalized_graph_relative_offdiag_change_json="[]",
                   normalized_graph_changed_offdiag_edge_fraction_json="[]",
                   patient_permutation_changed_fraction=0.0, patient_permutation_site_singletons=0,
                   patient_permutation_bijection=True, difference_message=False,
                   relation_weight_learning=False, effective_trainable_parameters=0,
                   allocated_trainable_parameter_elements=0)
        return row
    diag = module.last_diagnostics
    assert diag is not None
    spec = group_spec(config)
    changed_fraction, changed_weight_fraction, relative_change = [], [], []
    normalized_relative_change, normalized_changed_fraction = [], []
    for m, raw in enumerate(raw_adjs):
        n_edges = raw_edges[m]
        if m not in spec["relation_indices"]:
            changed_fraction.append(0.0)
            changed_weight_fraction.append(0.0)
            relative_change.append(0.0)
            normalized_relative_change.append(0.0)
            normalized_changed_fraction.append(0.0)
            continue
        mean_changed, mean_relative = [], []
        mean_weighted_fraction = []
        for k in spec["concept_indices"]:
            similarity = module.similarities[k]
            center = module.centers[k]
            factor = torch.exp(similarity - center)
            weight = torch.diagonal(raw)[:, None] * torch.eye(871, device=raw.device)
            raw_off = raw - torch.diag(torch.diagonal(raw))
            new = weight + raw_off * factor
            new_off = new - torch.diag(torch.diagonal(new))
            changed = (new_off - raw_off).abs()
            mean_changed.append(float((changed > 1e-10).sum().item() / max(n_edges, 1)))
            mean_weighted_fraction.append(float((changed[raw_off != 0] > 1e-10).sum().item() / max(n_edges, 1)))
            mean_relative.append(float(changed.sum().item() / raw_off.abs().sum().clamp_min(1e-12).item()))
        changed_fraction.append(float(np.mean(mean_changed)))
        changed_weight_fraction.append(float(np.mean(mean_weighted_fraction)))
        relative_change.append(float(np.mean(mean_relative)))
        normalized_relative_change.append(float(np.mean(
            diag["mean_conditioned_normalized_offdiag_relative_change"][m, spec["concept_indices"]])))
        normalized_changed_fraction.append(float(np.mean(
            diag["mean_conditioned_normalized_offdiag_changed_edge_fraction"][m, spec["concept_indices"]])))
    pmeta = prior["cache_meta"]
    row.update(
        alpha_json=json.dumps(diag["alpha"]),
        relation_weights_json=json.dumps(diag["weights"].mean(axis=0).sum(axis=1).tolist()),
        relation_weights_by_target_json=json.dumps(diag["weights"].sum(axis=2).tolist()),
        concept_weights_json=json.dumps(diag["weights"].mean(axis=0).sum(axis=0).tolist()),
        residual_ratio_mean_json=json.dumps(diag["mean_token_residual_ratio"]),
        residual_ratio_max_json=json.dumps(diag["max_token_residual_ratio"]),
        residual_ratio_max_observed=diag["max_token_residual_ratio_observed"],
        relation_message_contribution_norm_json=json.dumps(diag["mean_relation_message_contribution"].mean(axis=0).tolist()),
        relation_message_contribution_by_target_json=json.dumps(diag["mean_relation_message_contribution"].tolist()),
        relation_similarity_mean_json=json.dumps([item["similarity_mean"] for item in prior["stats"]]),
        relation_similarity_std_json=json.dumps([item["similarity_std"] for item in prior["stats"]]),
        relation_tau_json=json.dumps([item["tau"] for item in prior["stats"]]),
        relation_center_json=json.dumps([item["center"] for item in prior["stats"]]),
        relation_graph_offdiag_edges_json=json.dumps(raw_edges),
        relation_graph_density_json=json.dumps(densities),
        relation_graph_changed_edge_fraction_json=json.dumps(changed_fraction),
        relation_graph_changed_edge_weight_fraction_json=json.dumps(changed_weight_fraction),
        relation_graph_relative_edge_change_json=json.dumps(relative_change),
        normalized_graph_relative_offdiag_change_json=json.dumps(normalized_relative_change),
        normalized_graph_changed_offdiag_edge_fraction_json=json.dumps(normalized_changed_fraction),
        patient_permutation_changed_fraction=pmeta["patient_permutation_changed_fraction"] if config == "S8_PATIENT_PERMUTED" else 0.0,
        patient_permutation_site_singletons=pmeta["patient_permutation_site_singletons"] if config == "S8_PATIENT_PERMUTED" else 0,
        patient_permutation_sha256=pmeta["patient_permutation_sha256"] if config == "S8_PATIENT_PERMUTED" else "",
        patient_permutation_bijection=pmeta["patient_permutation_bijection"] if config == "S8_PATIENT_PERMUTED" else True,
        difference_message=not module.direct_message,
        relation_weight_learning=not module.uniform_relation,
        effective_trainable_parameters=group_spec(config)["active_parameter_elements"],
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
            if config == "S0_B0A":
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
                assert rng_path.is_file(), f"S0 fold-entry RNG missing: {fold}"
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
            metrics = evaluate_fold(agent, context, fold, collect=True)
            assert abs(metrics["acc"] - best_acc) <= 1e-12, \
                "restored best model and saved val=test accuracy diverged"
            diagnostic = fold_diagnostics(agent, context, fold, config, prior, gpu)
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
             if not key.startswith("serca34.")},
    )


def trunk_gradients(agent):
    result = {}
    for prefix, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion)):
        for name, parameter in module.named_parameters():
            if prefix == "pln" and name.startswith("serca34."):
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
    if hasattr(agent.ModalFusion, "serca34"):
        handles.append(agent.ModalFusion.serca34.register_forward_hook(output_hook("serca_output")))

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
    e32, e33 = history_reference_rows()
    s0 = {int(row["fold"]): row for row in rows if row["config"] == "S0_B0A"}
    assert set(s0) == set(e32) == set(e33) == set(range(1, 11))
    counts = dict(initial_parameter_hash_exact=0, initial_rng_hash_exact=0,
                  best_epoch_exact=0, four_metrics_exact=0)
    for fold in range(1, 11):
        row, old32, old33 = s0[fold], e32[fold], e33[fold]
        if row["initial_trunk_sha256"] == old32["initial_trunk_sha256"] == old33["initial_trunk_sha256"]:
            counts["initial_parameter_hash_exact"] += 1
        if row["initial_rng_sha256"] == old32["initial_rng_sha256"] == old33["initial_rng_sha256"]:
            counts["initial_rng_hash_exact"] += 1
        if int(row["best_epoch"]) == int(old32["best_epoch"]) == int(old33["best_epoch"]):
            counts["best_epoch_exact"] += 1
        exact = all(abs(float(row[current]) - float(old32[reference])) <= 1e-12
                    for current, reference in (("acc", "acc"), ("hard_auc", "auc"),
                                               ("sen_asd", "sen_asd"), ("spe_td", "spe_td")))
        if exact:
            counts["four_metrics_exact"] += 1
    return counts


def verify_s0():
    state = torch_read(task_path("S0_B0A"))
    assert state["phase"] == "COMPLETE" and len(state["rows"]) == 10
    comparison = compare_s0_to_history(state["rows"])
    if any(value != 10 for value in comparison.values()):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest.update(status="S0_PARITY_FAILED", completed_folds=10,
                        s0_historical_reproduction=comparison, updated_utc=utc())
        json_write(MANIFEST, manifest)
        raise RuntimeError(f"S0 does not exactly reproduce historical Experiment 32/33: {comparison}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest.update(status="S0_VERIFIED", completed_folds=10,
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
    manifest.update(status="RUNNING_S0", formal_started_utc=started)
    json_write(MANIFEST, manifest)
    s0_failure = launch_configs(["S0_B0A"])
    if s0_failure:
        raise RuntimeError(f"S0 worker failed; checkpoint remains resumable: {s0_failure}")
    comparison = verify_s0()
    record(f"S0 passed exact historical parity: {comparison}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest.update(status="RUNNING_S1_S9", other_groups_started_utc=utc())
    json_write(MANIFEST, manifest)
    rest = [config for config in CONFIGS if config != "S0_B0A"]
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
            differences = [float(lookup[("S9_SERCA_FULL", fold)][metric]) -
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
    s0, full = summary["S0_B0A"], summary["S9_SERCA_FULL"]
    conditions = {
        "s9_acc_and_hard_auc_above_s0":
            full["acc"]["mean"] > s0["acc"]["mean"] and full["hard_auc"]["mean"] > s0["hard_auc"]["mean"],
        "s9_acc_gain_at_least_0.23pp":
            100 * (full["acc"]["mean"] - s0["acc"]["mean"]) >= .23,
        "s9_acc_and_hard_auc_above_s1_numeric":
            comparisons["S1_NUMERIC_GLOBAL"]["acc"]["mean_delta"] > 0 and
            comparisons["S1_NUMERIC_GLOBAL"]["hard_auc"]["mean_delta"] > 0,
        "s9_acc_and_hard_auc_above_s2_random_concepts":
            comparisons["S2_RANDOM_CONCEPTS"]["acc"]["mean_delta"] > 0 and
            comparisons["S2_RANDOM_CONCEPTS"]["hard_auc"]["mean_delta"] > 0,
        "s9_acc_and_hard_auc_above_s8_patient_permuted":
            comparisons["S8_PATIENT_PERMUTED"]["acc"]["mean_delta"] > 0 and
            comparisons["S8_PATIENT_PERMUTED"]["hard_auc"]["mean_delta"] > 0,
        "s9_not_weaker_than_s4_corr_only":
            comparisons["S4_CORR_REL_ONLY"]["acc"]["mean_delta"] >= -1e-12 and
            comparisons["S4_CORR_REL_ONLY"]["hard_auc"]["mean_delta"] >= -1e-12,
        "s9_acc_and_hard_auc_above_s7_direct_message":
            comparisons["S7_DIRECT_MESSAGE"]["acc"]["mean_delta"] > 0 and
            comparisons["S7_DIRECT_MESSAGE"]["hard_auc"]["mean_delta"] > 0,
        "sen_asd_drop_within_0.50pp":
            100 * comparisons["S0_B0A"]["sen_asd"]["mean_delta"] >= -SEN_NONINFERIORITY_PP,
        "nonzero_semantic_residual_and_active_parameters":
            any(abs(value) > 0 for row in diagnostics if row["config"] == "S9_SERCA_FULL"
                for value in json.loads(row["alpha_json"])) and
            any(float(row["residual_ratio_max_observed"]) > 0
                for row in diagnostics if row["config"] == "S9_SERCA_FULL"),
        "cross_relation_gain_over_corr_only":
            comparisons["S4_CORR_REL_ONLY"]["acc"]["mean_delta"] > 0 and
            comparisons["S4_CORR_REL_ONLY"]["hard_auc"]["mean_delta"] > 0,
        "difference_message_beats_direct_conditioned_message":
            comparisons["S7_DIRECT_MESSAGE"]["acc"]["mean_delta"] > 0 and
            comparisons["S7_DIRECT_MESSAGE"]["hard_auc"]["mean_delta"] > 0,
    }
    core_semantic = all(conditions[key] for key in (
        "s9_acc_and_hard_auc_above_s1_numeric",
        "s9_acc_and_hard_auc_above_s2_random_concepts",
        "s9_acc_and_hard_auc_above_s8_patient_permuted",
    ))
    gates = all(conditions.values())
    if gates:
        status = "PROVISIONAL_GO — MULTISEED_AND_INDEPENDENT_VALIDATION_REQUIRED"
    elif full["acc"]["mean"] > s0["acc"]["mean"] and not core_semantic:
        status = "NUMERIC_GAIN_ONLY — SEMANTIC_CLAIM_NOT_SUPPORTED"
    else:
        status = "NO-GO"
    return dict(status=status, conditions=conditions,
                failed_conditions=[key for key, value in conditions.items() if not value],
                semantic_controls_passed=core_semantic,
                single_seed_internal_screen_only=True,
                statistical_significance_claimed=False,
                cross_modal_complementarity_demonstrated=conditions["cross_relation_gain_over_corr_only"],
                interpretation=("Correct disease concepts add evidence beyond the numeric, random and patient-mismatch controls, and non-CORR relation sources improve the full model."
                                if gates else
                                "The frozen method was evaluated as registered. The results do not satisfy every internal criterion required to claim that correct medical concepts guide cross-modal complementary relations."))


def percent_summary(value):
    return f"{100*value['mean']:.4f} ± {100*value['population_sd']:.4f}%"


def write_report(rows, diagnostics, summary, comparisons, decision, manifest):
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


def aggregate_and_report(controller_start=None):
    rows, diagnostics = [], []
    for config in CONFIGS:
        state = torch_read(task_path(config))
        assert state["phase"] == "COMPLETE" and len(state["rows"]) == 10
        rows.extend(state["rows"])
        diagnostics.extend(state["diagnostics"])
    assert len(rows) == len(diagnostics) == 100
    assert len({(row["config"], int(row["fold"])) for row in rows}) == 100
    for row in rows:
        row["historical_exp32_b0a_acc"] = float(e32_value(int(row["fold"]), "acc"))
    s0_parity = verify_s0()
    for row in rows:
        fold = int(row["fold"])
        baseline = next(item for item in rows if item["config"] == "S0_B0A" and int(item["fold"]) == fold)
        for metric in METRICS:
            row["delta_vs_s0_" + metric] = float(row[metric]) - float(baseline[metric])
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
        status="COMPLETE", completed_utc=utc(), completed_folds=100,
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
        print(f"COMPLETE {result['completed_folds']}/100; decision={result['decision']['status']}", flush=True)
    elif args.command == "aggregate":
        result = aggregate_and_report()
        print(f"AGGREGATED {result['completed_folds']}/100", flush=True)
    elif args.command == "_preflight":
        gpu = args.gpu if args.gpu is not None else int(args.config)
        preflight_checks(gpu)
    elif args.command == "_worker":
        assert args.config is not None and args.gpu is not None
        run_worker(args.config, args.gpu)


if __name__ == "__main__":
    main()

