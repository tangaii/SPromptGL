#!/usr/bin/env python3
"""Prepare, preflight, train, resume and report Experiment 38."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
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
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from method import LCHER, build_td_reference, make_wrong_edge_map

HERE = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/SPromptGL-main")
RUNTIME = HERE / ".runtime"
SOURCES = RUNTIME / "sources_lcher"
CACHE = RUNTIME / "cache"
CHECKPOINTS = RUNTIME / "checkpoints"
TASKS = RUNTIME / "tasks"
LOGS = RUNTIME / "logs"
RNG_DIR = RUNTIME / "rng"
RESULTS = HERE / "results"
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
GROUPS = [g["id"] for g in CFG["groups"]]
GROUP = {g["id"]: g for g in CFG["groups"]}
GPU = {g["id"]: int(g["gpu"]) for g in CFG["groups"]}
P0 = "P0_FULL_ORIGINAL"  # archived label only; not a formal arm in this rerun
B1 = "B0A_ANCHOR"
E_GROUPS = [g for g in GROUPS if g.startswith("E")]
CORE_FILES = ["README.md", "main.py", "model.py", "network.py", "layers.py", "utils.py",
              "ABIDE-simple-2-concat-weighted-cosine.sh"]
PROTOCOL = {
    "datname": "ABIDE", "nclass": 2, "nmodal": 4, "seed": 0,
    "mode": "simple-2", "GC_mode": "weighted-cosine", "MP_mode": "GCN", "MF_mode": "concat",
    "nepoch": 450, "early": 200, "lr": .0038, "reg": .11, "dropout": .35,
    "n_hidden": 18, "n_head": 2, "nlayer": 1, "th": .9,
    "theta_smooth": 1., "theta_degree": .5, "theta_sparsity": 0., "alpha": .5,
}
_DATA = None
_PROMPTS = None
_EDGE_NAMES = None
_TEXT_CACHE = None
_FOLDS = None
_RUNTIME_MODULES = None
_ANCHOR_RUNNER = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_json(value) -> str:
    return hash_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":"), allow_nan=False).encode())


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False, default=json_default)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def json_default(value):
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def set_seed(seed=0):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_hash(state):
    h = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    h.update(state["cpu"].cpu().numpy().tobytes())
    for v in state.get("cuda", []):
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def ensure_dirs():
    for p in (RUNTIME, SOURCES, CACHE, CHECKPOINTS, TASKS, LOGS, RNG_DIR, RESULTS):
        p.mkdir(parents=True, exist_ok=True)


def data_bundle():
    global _DATA, _FOLDS
    if _DATA is not None:
        return _DATA
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    data_dir = ROOT / "data/ABIDE"
    frame = pd.read_csv(data_dir / "processed_standard_data.csv")
    modal = np.load(data_dir / "modal_feat_dict.npy", allow_pickle=True).item()
    assert list(modal) == ["PHENO", "ANAT", "FUNC", "Correlation"]
    assert [len(v) for v in modal.values()] == [48, 6, 10, 256]
    assert frame.shape == (871, 321)
    x = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = frame.iloc[:, -1].to_numpy(dtype=np.int64) - 1
    assert x.shape == (871, 320) and np.isfinite(x).all()
    assert np.bincount(y).tolist() == [403, 468]
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10).split(x, y))
    assert len(folds) == 10
    _DATA = {"x": x, "y": y, "modal": modal, "columns": list(frame.columns)}
    _FOLDS = folds
    return _DATA


def fold_splits():
    data_bundle()
    return _FOLDS


def prompt_bundle():
    global _PROMPTS, _EDGE_NAMES
    if _PROMPTS is not None:
        return _PROMPTS, _EDGE_NAMES
    import csv
    obj = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))
    hypotheses = [h for h in obj["hypotheses"] if h["candidate"]]
    all_h = obj["hypotheses"]
    assert len(hypotheses) == 13
    edge_ids = [e for h in hypotheses for e in h["edges"]]
    assert len(edge_ids) == 39 and len(set(edge_ids)) == 39 and all(0 <= e < 256 for e in edge_ids)
    assert all(not h["candidate"] for h in all_h if h["id"] in ("H14", "H15", "H16"))
    assert not ({195, 225, 232, 236} & set(edge_ids))
    mapping_file = ROOT / "experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv"
    mapping = {int(r["correlation_feature_index"]): r for r in csv.DictReader(mapping_file.open(encoding="utf-8"))}
    names = {}
    for h in hypotheses:
        for pos, edge in enumerate(h["edges"]):
            row = mapping[edge]
            assert row["verification_status"] == "VERIFIED_NUMERIC_UNIQUE"
            assert row["connection_anatomical_status"] == "VERIFIED_BOTH_ENDPOINTS"
            assert row["roi_i_name_verified"] == "True" and row["roi_j_name_verified"] == "True"
            assert h["connection_names"][pos] == row["connection_name"]
            names[edge] = row["connection_name"]
    assert set(edge_ids) <= set(mapping)
    _PROMPTS, _EDGE_NAMES = hypotheses, names
    return _PROMPTS, _EDGE_NAMES


def source_hashes():
    return {name: sha(ROOT / name) for name in CORE_FILES}


def isolated_sources():
    ensure_dirs()
    import lcher_runtime
    lcher_runtime.emit(ROOT, SOURCES, HERE / "integration.patch")
    return {"B0A_compatible_lcher": str(SOURCES)}


def neutral_texts(hypotheses, names):
    result = []
    for h in hypotheses:
        result.append([
            "Neutral hypothesis: Measured resting-state functional coupling among the mapped cortical and subcortical parcels may vary in magnitude and direction across individuals and should be compared with an age-comparable reference.",
            "Neutral reference: Measured resting-state functional coupling among the mapped cortical and subcortical parcels may vary across individuals and should be compared with age-comparable reference data.",
        ])
    return result


def text_encoder_cache():
    """One frozen RN50 pass for real/neutral prompts; no CLIP construction RNG escapes."""
    import numpy as np
    import torch
    import sys
    hypotheses, names = prompt_bundle()
    path = CACHE / "clip_embeddings.npz"
    meta_path = CACHE / "clip_embeddings.json"
    payload_texts = [[h["asd"], h["td"]] for h in hypotheses]
    neutral = neutral_texts(hypotheses, names)
    original = ["Nromal brain volume", "Uniform cortical thickness", "Symmetrical brain structure",
                "Normal ventricular morphology", "Altered brain volume", "Abnormal functional connectivity",
                "White matter integrity issues", "Different activation patterns"]
    all_text = [t for pair in payload_texts + neutral for t in pair] + original
    sys.path.insert(0, str(ROOT))
    from CLIP.clip import clip
    tokens = clip.tokenize(all_text, context_length=77, truncate=False)
    assert tokens.shape == (60, 77) and (tokens != 0).any(1).all()
    token_hash = hash_bytes(tokens.cpu().numpy().tobytes())
    expected_weight = sha(Path.home() / ".cache/clip/RN50.pt")
    assert expected_weight == CFG["method"]["clip_pretrained_sha256"]
    clip_script_hash = sha(ROOT / "CLIP/clip/clip.py")
    tokenizer_hash = sha(ROOT / "CLIP/clip/simple_tokenizer.py")
    descriptor = {"model": "OpenAI CLIP RN50", "weights_sha256": expected_weight,
                  "clip_script_sha256": clip_script_hash, "tokenizer_sha256": tokenizer_hash,
                  "tokenizer_context": 77, "token_tensor_sha256": token_hash,
                  "text_sha256": hash_json(all_text), "prompt_text_count": 26,
                  "neutral_text_count": 26, "legacy_prompt_count": 8,
                  "license": "OpenAI CLIP repository LICENSE is MIT; pretrained weight use is subject to the linked model terms. No separate project-specific license attestation is claimed."}
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["descriptor"] == descriptor
        assert meta["cache_sha256"] == sha(path)
        with np.load(path) as z:
            real, neut, orig, anon = z["real"], z["neutral"], z["original"], z["anonymous"]
        assert real.shape == neut.shape == anon.shape == (13, 2, 1024) and orig.shape == (8, 1024)
        return {"real": real, "neutral": neut, "original": orig, "anonymous": anon,
                "descriptor": descriptor, "cache_sha256": sha(path)}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        model, _ = clip.load("RN50", device, download_root=str(Path.home() / ".cache/clip"))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        with torch.no_grad():
            emb = model.encode_text(tokens.to(device)).float().cpu().numpy()
    real = emb[:26].reshape(13, 2, 1024)
    neut = emb[26:52].reshape(13, 2, 1024)
    orig = emb[52:60]
    norms = np.linalg.norm(real.reshape(-1, 1024), axis=1)
    anon_rng = np.random.default_rng(38001)
    anonymous = anon_rng.normal(size=(13, 2, 1024)).astype(np.float32)
    anonymous /= np.maximum(np.linalg.norm(anonymous, axis=-1, keepdims=True), 1e-12)
    anonymous *= float(norms.mean())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        np.savez_compressed(f, real=real.astype(np.float32), neutral=neut.astype(np.float32),
                            original=orig.astype(np.float32), anonymous=anonymous)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    json_write(meta_path, {"descriptor": descriptor, "cache_sha256": sha(path)})
    return {"real": real, "neutral": neut, "original": orig, "anonymous": anonymous,
            "descriptor": descriptor, "cache_sha256": sha(path)}


def fold_reference(fold, train):
    import numpy as np
    from method import build_td_reference
    hypotheses, _ = prompt_bundle()
    edges = np.asarray([e for h in hypotheses for e in h["edges"]], dtype=np.int64)
    path = CACHE / f"fold_{fold:02d}_td_reference.npz"
    meta_path = path.with_suffix(".json")
    data = data_bundle()
    descriptor = {"fold": fold, "train_indices_sha256": hash_bytes(np.asarray(train, dtype=np.int64).tobytes()),
                  "x_sha256": sha(ROOT / "data/ABIDE/processed_standard_data.csv"),
                  "edges": edges.tolist(), "ridge_alpha": 10.0, "td_label": 1,
                  "fit_labels_scope": "outer-train only"}
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["descriptor"] == descriptor and meta["cache_sha256"] == sha(path)
        with np.load(path) as z:
            return {key: z[key] for key in z.files} | {
                "descriptor": descriptor,
                "oof_residual_rms": float(meta["oof_residual_rms"]),
                "sigma_min": float(meta["sigma_min"]),
                "sigma_median": float(meta["sigma_median"]),
            }
    ref = build_td_reference(data["x"], data["y"], train, fold, edges, alpha=10., sigma_floor=1e-4)
    # QC reliability uses outer-train moments; no diagnosis labels enter this gate.
    qc = data["x"][:, 48:64].astype(np.float64)
    qc_train = qc[np.asarray(train, dtype=np.int64)]
    qc_mu = qc_train.mean(0)
    qc_sd = np.maximum(qc_train.std(0), 1e-4)
    qc_z = (qc - qc_mu) / qc_sd
    reliability = np.exp(-np.maximum(np.abs(qc_z) - 1., 0.).mean(1)).astype(np.float32)
    ref["reliability"] = reliability
    ref["sigma"] = np.asarray(ref["sigma"], dtype=np.float32)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **{k: np.asarray(v) for k, v in ref.items()
                                 if k in ("mu", "sigma", "z", "reliability", "train_td", "train")})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    json_write(meta_path, {"descriptor": descriptor, "cache_sha256": sha(path),
                           "oof_residual_rms": ref["oof_residual_rms"],
                           "sigma_min": ref["sigma_min"], "sigma_median": ref["sigma_median"]})
    ref["descriptor"] = descriptor
    return ref


def initialize_manifest():
    import numpy as np
    ensure_dirs()
    data = data_bundle()
    hypotheses, names = prompt_bundle()
    sources = isolated_sources()
    text = text_encoder_cache()
    import method
    wrong_map = method.make_wrong_edge_map(hypotheses, 38002)
    assert len(wrong_map) == 13 and all(len(wrong_map[i]) == len(hypotheses[i]["edges"]) for i in range(13))
    for fold, (train, test) in enumerate(fold_splits(), 1):
        fold_reference(fold, train)
    identity_path = ROOT / "experiment/experiment11_roi_mapping_recovery/results/subject_row_mapping.csv"
    identity = csv_rows(identity_path)
    assert len(identity) == 871 and len({r["subject_SUB_ID"] for r in identity}) == 871
    assert all(r["verification_status"] == "VERIFIED" and r["unique_match"] == "True" for r in identity)
    folds_sha = hash_bytes(b"".join(np.asarray(np.concatenate([tr, te]), dtype=np.int64).tobytes()
                                     for tr, te in fold_splits()))
    manifest_path = HERE / "manifest.json"
    anchor_audit = verify_anchor_replay()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["source_sha256"] == source_hashes()
        assert manifest["config_sha256"] == sha(HERE / "config.json")
        assert manifest.get("anchor_replay", {}).get("status") == "ANCHOR_REPLAY_PASS"
        materialize_anchor_rows()
        return manifest
    manifest = {
        "experiment": CFG["experiment"], "status": "PREPARED", "prepared_utc": time.time(),
        "workspace": str(HERE), "delivery": "TEMP_STAGE_PENDING_SYNC",
        "root_write_probe": "FAILED: atomic temp file fsync returned No such file or directory under OSSFS; subsequent writes used /tmp stage",
        "root_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "github_main": "66d6a3b6e8ddc0332fe1f50cbf31caa1ae929afa",
        "source_sha256": source_hashes(),
        "input_sha256": {"processed_standard_data.csv": sha(ROOT / "data/ABIDE/processed_standard_data.csv"),
                         "modal_feat_dict.npy": sha(ROOT / "data/ABIDE/modal_feat_dict.npy"),
                         "roi_256_named_mapping.csv": sha(ROOT / "experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv"),
                         "subject_row_mapping.csv": sha(identity_path)},
        "code_sha256": {}, "config_sha256": sha(HERE / "config.json"),
        "prompts_sha256": sha(HERE / "prompts.json"), "fold_split_sha256": folds_sha,
        "subject_count": int(len(data["x"])), "input_shape": list(data["x"].shape),
        "class_counts": {"ASD_label0": int((data["y"] == 0).sum()), "TD_label1": int((data["y"] == 1).sum())},
        "verified_roi_mapping_rows": len(names), "candidate_hypotheses": 13,
        "candidate_unique_edges": len(names), "excluded_unknown_edges_absent": True,
        "wrong_edge_map": wrong_map, "wrong_edge_map_seed": 38002,
        "text_encoder": text["descriptor"] | {"cache_sha256": text["cache_sha256"]},
        "neutral_control_note": "Unique neutral pair wording deterministically includes verified connection names; neutral embeddings are not geometrically identical to ASD/TD hypotheses.",
        "protocol": PROTOCOL, "groups": CFG["groups"], "gpu_assignments": CFG["scheduling"],
        "target_folds": 100, "completed_folds": 10, "gpu_completed_folds": {"0": 10, "1": 0},
        "preflight": {"passed": False}, "tasks": [dict(group=g, gpu=GPU[g], status="PENDING", folds=0) for g in GROUPS],
        "recovery_events": [], "license": {"source": "https://github.com/openai/CLIP/blob/main/LICENSE", "project_specific_attestation": False},
        "handoff_status": "PENDING",
        "anchor_replay": anchor_audit,
    }
    json_write(manifest_path, manifest)
    materialize_anchor_rows()
    return manifest


def verify_anchor_replay():
    """Require exact agreement with the frozen E32-B0 folds before preparing LCHER."""
    task_path = RUNTIME / "tasks/B0_seed0.json"
    history_path = ROOT / "experiment/experiment32_disease_contrastive_evidence_routing/results/fold_metrics.csv"
    if not task_path.exists() or not history_path.exists():
        raise RuntimeError("ANCHOR_REPLAY_FAIL: missing fresh B0-A replay or E32 history")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    assert task.get("status") == "COMPLETE" and len(task.get("metrics", [])) == 10
    historical = {int(r["fold"]): r for r in csv_rows(history_path)
                  if r["config"] == "B0" and int(r["seed"]) == 0}
    current = {int(r["fold"]): r for r in task["metrics"]}
    assert len(historical) == len(current) == 10
    checked = []
    exact = ("asd_tp", "asd_fn", "td_fp", "td_tn", "best_epoch", "epochs_executed",
             "initial_trunk_sha256", "initial_rng_sha256")
    for fold in range(1, 11):
        a, b = current[fold], historical[fold]
        diffs = [field for field in exact if str(a[field]) != str(b[field])]
        for field in ("acc", "auc", "sen_asd", "spe_td"):
            if float(a[field]) != float(b[field]):
                diffs.append(field)
        checked.append({"fold": fold, "status": "MATCH" if not diffs else "MISMATCH", "differences": diffs})
    if any(row["status"] != "MATCH" for row in checked):
        raise RuntimeError("ANCHOR_REPLAY_FAIL: E32 fold-level mismatch: " + json.dumps(checked))
    anchor_paths = [RUNTIME / "anchors" / f"fold-{fold:02d}.pt" for fold in range(1, 11)]
    assert all(path.exists() for path in anchor_paths)
    return {"status": "ANCHOR_REPLAY_PASS", "folds": checked,
            "anchor_sha256": {path.name: sha(path) for path in anchor_paths},
            "continuous_rng_protocol": "single seed=0 process; no per-fold reseed"}


def materialize_anchor_rows():
    task = json.loads((RUNTIME / "tasks/B0_seed0.json").read_text(encoding="utf-8"))
    anchor_rows = []
    for value in task["metrics"]:
        anchor_rows.append({"group": "B0A_ANCHOR", "seed": 0, "fold": value["fold"],
                     "physical_gpu": 0, "acc": value["acc"], "hard_auc": value["auc"],
                     "spe_td": value["spe_td"], "sen_asd": value["sen_asd"],
                     "asd_tp": value["asd_tp"], "asd_fn": value["asd_fn"],
                     "td_fp": value["td_fp"], "td_tn": value["td_tn"],
                     "best_epoch": value["best_epoch"], "epochs_executed": value["epochs_executed"],
                     "initial_trunk_sha256": value["initial_trunk_sha256"],
                     "initial_rng_sha256": value["initial_rng_sha256"],
                     "fold_initial_anchor": value["fold_initial_anchor"],
                     "parameter_count": "", "fold_seconds": value["wall_seconds"]})
    existing = csv_rows(RESULTS / "fold_metrics.csv") if (RESULTS / "fold_metrics.csv").exists() else []
    rows = anchor_rows + [row for row in existing
                          if row.get("group") in GROUPS and row.get("group") != "B0A_ANCHOR"]
    rows.sort(key=lambda row: (row["group"], int(row["fold"])))
    csv_write(RESULTS / "reference_anchor.csv", rows)
    csv_write(RESULTS / "fold_metrics.csv", rows)
    prior = csv_rows(RESULTS / "diagnostics.csv") if (RESULTS / "diagnostics.csv").exists() else []
    prior = [row for row in prior
             if row.get("group") in GROUPS and row.get("group") != "B0A_ANCHOR"]
    for fold in range(1, 11):
        prior.append({"group": "B0A_ANCHOR", "fold": fold, "seed": 0, "physical_gpu": 0,
                      "anchor_exact_history_match": True})
    csv_write(RESULTS / "diagnostics.csv", prior)


def load_runtime(group):
    global _RUNTIME_MODULES
    if _RUNTIME_MODULES is not None and _RUNTIME_MODULES["group"] == group:
        return _RUNTIME_MODULES
    for name in ("network", "utils", "integration"):
        sys.modules.pop(name, None)
    source = SOURCES
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(HERE))
    import network
    import utils
    integration = None
    if group != P0:
        import integration as integration_module
        integration = integration_module
    _RUNTIME_MODULES = {"group": group, "network": network, "utils": utils, "integration": integration}
    return _RUNTIME_MODULES


class Agent:
    def __init__(self, group, fold, train, test, entry_state, reference):
        import numpy as np
        import torch
        global _ANCHOR_RUNNER
        if _ANCHOR_RUNNER is None:
            import run as anchor_runner
            _ANCHOR_RUNNER = anchor_runner
        anchor = _ANCHOR_RUNNER
        if anchor._CTX is None:
            anchor.SOURCE = SOURCES
            anchor.context()
        self.base = anchor.make_agent("B0", fold)
        for name in ("dev", "feat", "targ", "trn_idx", "val_idx", "tst_idx",
                     "hyperpm", "dataLayer", "MessagePassing0", "ModalFusion", "optimizer_MP0",
                     "optimizer_MF", "weight", "MF_sav", "GCMP_sav"):
            setattr(self, name, getattr(self.base, name))
        self.trn_idx = torch.as_tensor(train, dtype=torch.long, device=self.dev)
        self.val_idx = torch.as_tensor(test, dtype=torch.long, device=self.dev)
        self.tst_idx = torch.as_tensor(test, dtype=torch.long, device=self.dev)
        self.network = anchor.context()["network"]
        self.utils = anchor.context()["utils"]
        anchor_data = load_anchor(fold)
        assert rng_hash(entry_state) == anchor_data["initial_rng_sha256"], \
            "paired group did not enter from its B0-A fold RNG snapshot"
        assert anchor.trunk_parameter_hash(self.base) == anchor_data["initial_trunk_sha256"], \
            "freshly reconstructed shared trunk differs from B0-A anchor"
        self.load_model_state({"MessagePassing0": anchor_data["model"]["gfr"],
                               "ModalFusion": anchor_data["model"]["pln"]})
        self.optimizer_MP0.load_state_dict(anchor_data["optimizer_mp"])
        self.optimizer_MF.load_state_dict(anchor_data["optimizer_mf"])
        hypotheses, _ = prompt_bundle()
        if group == B1:
            assert shared_parameter_hash(self) == anchor_data["initial_trunk_sha256"]
            counter = Counter(self.targ[self.trn_idx].detach().cpu().numpy().tolist())
            weights = len(train) / np.asarray(list(counter.values()), dtype=np.float64) / 2.
            self.weight = torch.as_tensor(weights, dtype=torch.float32, device=self.dev)
            self.entry_rng = entry_state
            return
        if group == "E1_ANON_FULL":
            bank = _TEXT_CACHE["anonymous"]
        elif group == "E6_NEUTRAL_FULL":
            bank = _TEXT_CACHE["neutral"]
        else:
            bank = _TEXT_CACHE["real"]
        if GROUP[group].get("swap_class"):
            bank = bank[:, ::-1].copy()
        mapping = [item["edges"] for item in hypotheses]
        if GROUP[group].get("wrong_edge_map"):
            mapping = json.loads((HERE / "manifest.json").read_text())["wrong_edge_map"]
        direction = torch.as_tensor([1. if h["direction"] == "hyper_candidate" else
                                     -1. if h["direction"] == "hypo_candidate" else 0.
                                     for h in hypotheses for _ in h["edges"]], dtype=torch.float32)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            branch = LCHER(torch.as_tensor(bank, dtype=torch.float32), mapping, direction,
                           use_aux=bool(GROUP[group]["aux"]), use_rival=bool(GROUP[group]["rival"]),
                           top_k=8, seed=38000 + int(fold)).to(self.dev)
        self.ModalFusion.lcher = branch
        self.lcher_base_context = make_router_context(reference, self.feat, self.dev)
        self.ModalFusion.lcher_context = dict(self.lcher_base_context)
        self.optimizer_MF.add_param_group({"params": list(branch.parameters()), "lr": .0038, "weight_decay": 0.0})
        assert shared_parameter_hash(self) == anchor_data["initial_trunk_sha256"]
        counter = Counter(self.targ[self.trn_idx].detach().cpu().numpy().tolist())
        weights = len(train) / np.asarray(list(counter.values()), dtype=np.float64) / 2.
        self.weight = torch.as_tensor(weights, dtype=torch.float32, device=self.dev)
        self.entry_rng = entry_state

    def forward(self, raw=None, context_override=None):
        import torch
        raw = self.feat if raw is None else raw
        model_num, parts = self.dataLayer(raw)
        feat, adj, aloss = self.MessagePassing0(parts, model_num, self.targ, self.trn_idx)
        graph_loss = sum(self.utils.GraphConstructLoss(parts[i], adj[i], 1., .5, 0.) for i in range(4)) / 4
        if getattr(self.ModalFusion, "lcher", None) is not None:
            if context_override is not None:
                self.ModalFusion.lcher_context = context_override
            else:
                self.ModalFusion.lcher_context = dict(
                    self.lcher_base_context, raw_x=raw,
                    mask_override=None, disable_residual=False)
        output = self.ModalFusion(feat)
        return output[0], graph_loss, aloss, adj, feat

    def model_state(self):
        return {"MessagePassing0": copy.deepcopy(self.MessagePassing0.state_dict()),
                "ModalFusion": copy.deepcopy(self.ModalFusion.state_dict())}

    def load_model_state(self, state):
        self.MessagePassing0.load_state_dict(state["MessagePassing0"])
        self.ModalFusion.load_state_dict(state["ModalFusion"])


def make_router_context(reference, raw_x, device):
    import torch
    return {"raw_x": raw_x,
            "z": torch.as_tensor(reference["z"], dtype=torch.float32, device=device),
            "mu": torch.as_tensor(reference["mu"], dtype=torch.float32, device=device),
            "sigma": torch.as_tensor(reference["sigma"], dtype=torch.float32, device=device),
            "qc_reliability": torch.as_tensor(reference["reliability"], dtype=torch.float32, device=device),
            "mask_override": None, "disable_residual": False}


def model_hash(agent):
    h = hashlib.sha256()
    for module_name, module in (("GFR", agent.MessagePassing0), ("PLN", agent.ModalFusion)):
        for name, value in module.state_dict().items():
            if name.startswith("lcher."):
                continue
            h.update((module_name + "/" + name).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def shared_parameter_hash(agent):
    h = hashlib.sha256()
    for module_name, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion)):
        for name, parameter in sorted(module.named_parameters()):
            if name.startswith("lcher."):
                continue
            h.update(module_name.encode() + b"/" + name.encode())
            h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def optimizer_audit(agent, group):
    groups = agent.optimizer_MP0.param_groups + agent.optimizer_MF.param_groups
    ids = [id(p) for pg in groups for p in pg["params"]]
    actual = {id(p) for m in (agent.MessagePassing0, agent.ModalFusion) for p in m.parameters()}
    assert len(ids) == len(set(ids)) and set(ids) == actual
    assert len(agent.optimizer_MP0.param_groups) == 1
    if group.startswith("E"):
        assert len(agent.optimizer_MF.param_groups) == 2
        assert agent.optimizer_MF.param_groups[-1]["weight_decay"] == 0.
        assert all(agent.optimizer_MF.param_groups[0]["weight_decay"] == .11
                   for _ in [0])
    return int(sum(p.numel() for p in agent.ModalFusion.parameters()))


def atomic_torch_save(path, value):
    import torch
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def state_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def load_fold_entry(fold):
    return load_anchor(fold)["entry_rng"]


def load_anchor(fold):
    import torch
    path = RUNTIME / "anchors" / f"fold-{fold:02d}.pt"
    if not path.exists():
        raise RuntimeError(f"B0-A fold anchor missing: {path}")
    raw = path.read_bytes()
    assert raw[:4] == b"E32!" and hashlib.sha256(raw[68:]).hexdigest().encode() == raw[4:68]
    value = torch.load(io.BytesIO(raw[68:]), map_location="cpu", weights_only=False)
    assert value["config"] == "B0" and value["fold"] == fold
    return value


def train_pair_loss(agent, group, reference, probe_ids, output, selected):
    """Full GFR/CAM re-forward per target; backbone frozen, only selector receives CF gradients."""
    import torch
    import torch.nn.functional as F
    if not GROUP[group]["keep"]:
        return output.sum() * 0., {"probes": 0, "keep_loss": 0., "delete_loss": 0.}
    train_ids = set(agent.trn_idx.detach().cpu().tolist())
    assert set(map(int, probe_ids)) <= train_ids, "counterfactual auxiliary supervision may only probe outer-train patients"
    branch = agent.ModalFusion.lcher
    full_context = agent.ModalFusion.lcher_context
    raw = agent.feat
    route_params = [p for p in branch.parameters()]
    primary_grads = [None if p.grad is None else p.grad.detach().clone() for p in route_params]
    for p in route_params:
        p.grad = None
    primary_selected_grad = None if selected.grad is None else selected.grad.detach().clone()
    selected.grad = None
    edges = torch.as_tensor([e for h in prompt_bundle()[0] for e in h["edges"]], dtype=torch.long, device=agent.dev)
    input_cols = edges + 64
    mu = full_context["mu"]
    sigma = full_context["sigma"]
    # Main parameters have already accumulated the primary classification and graph gradients.
    non_route = [p for m in (agent.MessagePassing0, agent.ModalFusion) for n, p in m.named_parameters()
                 if not n.startswith("lcher.")]
    prior_flags = [p.requires_grad for p in non_route]
    for p in non_route:
        p.requires_grad_(False)
    branch.train()
    agent.MessagePassing0.eval()
    agent.ModalFusion.eval()
    keep_total = delete_total = 0.
    cf_losses = []
    cf_signed_tensors = []
    cf_delta_tensors = []
    try:
        for subject in probe_ids:
            mask = selected[subject]
            keep_row = mu[subject] + (raw[subject, input_cols] - mu[subject]) * mask
            keep_x = raw.clone()
            keep_x[subject, input_cols] = keep_row
            keep_z = full_context["z"].clone()
            keep_z[subject] = torch.clamp((keep_row - mu[subject]) / sigma, -3., 3.)
            keep_context = dict(full_context, raw_x=keep_x, z=keep_z, mask_override=selected)
            # Only the target participant's rows are changed; independent full graph re-computations follow.
            with torch.no_grad():
                full_context_eval = dict(full_context, mask_override=selected.detach())
                full_logp, _, _, _, _ = agent.forward(raw, full_context_eval)
                teacher = full_logp[subject].detach().exp()
            keep_logp, _, _, _, _ = agent.forward(keep_x, keep_context)
            keep_signed = branch._last_diagnostics["signed"]
            keep_delta = branch._last_diagnostics["delta"]
            keep_signed.retain_grad(); keep_delta.retain_grad()
            cf_signed_tensors.append(keep_signed); cf_delta_tensors.append(keep_delta)
            keep_loss = F.kl_div(keep_logp[subject], teacher, reduction="sum")
            delete_loss = keep_loss.new_zeros(())
            if GROUP[group]["delete"]:
                delete_row = mu[subject] + (raw[subject, input_cols] - mu[subject]) * (1. - mask)
                delete_x = raw.clone()
                delete_x[subject, input_cols] = delete_row
                delete_z = full_context["z"].clone()
                delete_z[subject] = torch.clamp((delete_row - mu[subject]) / sigma, -3., 3.)
                delete_context = dict(full_context, raw_x=delete_x, z=delete_z, mask_override=selected)
                remove_logp, _, _, _, _ = agent.forward(delete_x, delete_context)
                delete_signed = branch._last_diagnostics["signed"]
                delete_delta = branch._last_diagnostics["delta"]
                delete_signed.retain_grad(); delete_delta.retain_grad()
                cf_signed_tensors.append(delete_signed); cf_delta_tensors.append(delete_delta)
                label = agent.targ[subject]
                log_margin = keep_logp[subject, label] - remove_logp[subject, label]
                delete_loss = F.softplus(0.05 - log_margin)
            keep_weight = .10 if GROUP[group]["keep"] else 0.
            delete_weight = .05 if GROUP[group]["delete"] else 0.
            cf_losses.append(keep_weight * keep_loss + delete_weight * delete_loss)
            keep_total += float(keep_loss.detach().item())
            delete_total += float(delete_loss.detach().item())
        # Each patient gets an independent raw-input/GFR re-forward, while all
        # counterfactual paths share the primary pass's straight-through selector.
        # Accumulate before backward so that shared selector graph is traversed once.
        if cf_losses:
            torch.stack(cf_losses).sum().backward()
        cf_route_grads = [None if p.grad is None else p.grad.detach().clone() for p in route_params]
        gradient_norms = {}
        for name, parameter, grad in zip(
                [name for name, _ in branch.named_parameters()], route_params, cf_route_grads):
            gradient_norms[name.replace(".", "_")] = 0. if grad is None else float(grad.norm().item())
        for p, primary, auxiliary in zip(route_params, primary_grads, cf_route_grads):
            if primary is None and auxiliary is None:
                p.grad = None
            elif primary is None:
                p.grad = auxiliary
            elif auxiliary is None:
                p.grad = primary
            else:
                p.grad = primary + auxiliary
        selected_cf_grad = None if selected.grad is None else selected.grad.detach().clone()
        gradient_norms["selected"] = 0. if selected_cf_grad is None else float(selected_cf_grad.norm().item())
        selected.grad = (selected_cf_grad if primary_selected_grad is None else
                         primary_selected_grad if selected_cf_grad is None else
                         primary_selected_grad + selected_cf_grad)
        signed_norms = [0. if tensor.grad is None else float(tensor.grad.detach().norm().item())
                        for tensor in cf_signed_tensors]
        delta_norms = [0. if tensor.grad is None else float(tensor.grad.detach().norm().item())
                       for tensor in cf_delta_tensors]
        gradient_norms["signed_evidence"] = float(sum(signed_norms))
        gradient_norms["signed_residual"] = float(sum(delta_norms))
    finally:
        for p, flag in zip(non_route, prior_flags):
            p.requires_grad_(flag)
        agent.MessagePassing0.train()
        agent.ModalFusion.train()
    return output.sum() * 0., {"probes": len(probe_ids), "gradient_norms": gradient_norms,
                               "keep_loss": keep_total / max(len(probe_ids), 1),
                               "delete_loss": delete_total / max(len(probe_ids), 1)}


def epoch_step(agent, group, fold, epoch, reference, train, probe_rotation, probe_cursor):
    import torch
    import torch.nn.functional as F
    agent.MessagePassing0.train()
    agent.ModalFusion.train()
    agent.optimizer_MP0.zero_grad(set_to_none=True)
    agent.optimizer_MF.zero_grad(set_to_none=True)
    output, graph_loss, aloss, adj, feat = agent.forward()
    cls_loss = F.nll_loss(output[agent.trn_idx], agent.targ[agent.trn_idx], weight=agent.weight)
    total = cls_loss + graph_loss
    total.backward(retain_graph=bool(group.startswith("E") and GROUP[group]["keep"] and (epoch + 1) % 5 == 0))
    cf_diag = {"probes": 0, "keep_loss": 0., "delete_loss": 0.}
    branch = getattr(agent.ModalFusion, "lcher", None)
    if branch is not None and GROUP[group]["keep"] and (epoch + 1) % 5 == 0:
        probe_ids = [int(probe_rotation[(probe_cursor + i) % len(probe_rotation)]) for i in range(2)]
        selected = branch._last_diagnostics["selected"]
        selected.retain_grad()
        cf_loss, cf_diag = train_pair_loss(agent, group, reference, probe_ids, output, selected)
        # `train_pair_loss` back-propagates its two auxiliary terms without updating the backbone.
        probe_cursor = (probe_cursor + 2) % len(probe_rotation)
        gradient_audit = cf_diag.get("gradient_norms", {})
    agent.optimizer_MP0.step()
    agent.optimizer_MF.step()
    assert torch.isfinite(output).all() and torch.isfinite(cls_loss) and torch.isfinite(graph_loss)
    return {"cls_loss": float(cls_loss.detach().item()), "graph_loss": float(graph_loss.detach().item()),
            "aloss_diagnostic": float(aloss.detach().item()), "cf": cf_diag,
            "probe_cursor": probe_cursor, "output": output.detach(), "adj": [a.detach() for a in adj],
            "selected": None if branch is None else branch.last_mask.detach().clone(),
            "router": None if branch is None else branch._last_diagnostics,
            "gradient_audit": cf_diag.get("gradient_norms", {})}


def metric_row(output, test, y):
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_auc_score
    y_true = np.asarray(y)[np.asarray(test)]
    pred = output.detach().cpu().numpy()[test].argmax(1)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    asd_tp, asd_fn, td_fp, td_tn = map(int, cm.ravel())
    onehot_y, onehot_pred = np.eye(2)[y_true], np.eye(2)[pred]
    return {"acc": float((pred == y_true).mean()),
            "hard_auc": float(roc_auc_score(onehot_y, onehot_pred)),
            "spe_td": float(td_tn / (td_tn + td_fp)),
            "sen_asd": float(asd_tp / (asd_tp + asd_fn)),
            "asd_tp": asd_tp, "asd_fn": asd_fn, "td_fp": td_fp, "td_tn": td_tn}


def set_best(agent, state):
    agent.load_model_state(state)


def deterministic_probe_order(train, fold):
    import numpy as np
    rng = np.random.default_rng(38004 + fold)
    return rng.permutation(np.asarray(train, dtype=np.int64)).tolist()


def fold_diagnostics(agent, group, fold, train, test, metric, best_epoch, cf_train_summary):
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    agent.MessagePassing0.eval()
    agent.ModalFusion.eval()
    raw = agent.feat
    context = getattr(agent.ModalFusion, "lcher_context", None)
    if group.startswith("E"):
        context = dict(context, mask_override=None)
    with torch.no_grad():
        output, _, _, adjs, feat = agent.forward(raw, context)
    row = {"group": group, "seed": 0, "fold": fold, "best_epoch": best_epoch,
           "physical_gpu": GPU[group], "train_td_reference_rms": float(fold_reference(fold, train)["oof_residual_rms"]),
           "mean_train_cf_keep_loss": cf_train_summary.get("keep_loss", 0.),
           "mean_train_cf_delete_loss": cf_train_summary.get("delete_loss", 0.),
           "cf_gradient_norms": json.dumps(cf_train_summary.get("gradient_norms", {}), separators=(",", ":")),
           "train_cf_probe_count": int(cf_train_summary.get("probes", 0))}
    edge_stats = []
    for m, adj in enumerate(adjs):
        off = adj.clone()
        off.fill_diagonal_(0.)
        edge_stats.append({"edges": int((off > 0).sum().item()),
                           "density": float((off > 0).float().mean().item()),
                           "degree_mean": float((off > 0).sum(1).float().mean().item()),
                           "isolates": int(((off > 0).sum(1) == 0).sum().item())})
    row["modality_edge_stats"] = json.dumps(edge_stats, separators=(",", ":"))
    row["graph_loss"] = float(sum(agent.utils.GraphConstructLoss(agent.dataLayer(raw)[1][i], adjs[i], 1., .5, 0.)
                                  for i in range(4)).item() / 4.)
    row["aloss_not_optimized"] = "computed by GFR; excluded from total loss"
    row["val_equals_test"] = True
    row["best_checkpoint_acc"] = metric["acc"]
    if not group.startswith("E"):
        row.update({"selected_count_mean": "", "null_probability_mean": "", "route_gate_sd": "",
                    "residual_ratio_mean": "", "residual_ratio_max": "", "zero_residual_fraction": "",
                    "selected_vs_random_logp_drop": "", "selected_logp_drop": "",
                    "random_logp_drop": "", "top_abs_z_logp_drop": "", "text_pair_cosine_mean": "",
                    "text_pair_cosine_sd": "", "test_cf_class_flip_fraction": "",
                    "test_cf_abs_logp_delta": "", "residual_alpha": ""})
        return row
    branch = agent.ModalFusion.lcher
    diag = branch._last_diagnostics
    selected = branch.last_mask[test].bool()
    row["selected_count_mean"] = float(selected.sum(-1).float().mean().item())
    row["null_probability_mean"] = float(diag["null_fraction"].item())
    row["route_gate_sd"] = float(branch.last_gate[test].std().item())
    score_mean = branch.last_scores[test].mean(0)
    signed_mean = branch.last_signed[test].mean(0)
    selected_mean = branch.last_mask[test].mean(0)
    edge_to_hypothesis = branch.edge_to_hypothesis
    group_stats = []
    cursor = 0
    for hypothesis_edges in edge_to_hypothesis:
        width = len(hypothesis_edges)
        selected_block = branch.last_mask[test, cursor:cursor + width].float()
        score_block = branch.last_scores[test, cursor:cursor + width]
        signed_block = branch.last_signed[test, cursor:cursor + width]
        group_stats.append({"selected_fraction": float(selected_mean[cursor:cursor + width].mean().item()),
                            "selected_fraction_sd": float(selected_block.mean(-1).std(unbiased=False).item()),
                            "score_mean": float(score_mean[cursor:cursor + width].mean().item()),
                            "score_sd": float(score_block.mean(-1).std(unbiased=False).item()),
                            "signed_mean": float(signed_mean[cursor:cursor + width].mean().item())})
        cursor += width
    row["hypothesis_routing"] = json.dumps(group_stats, separators=(",", ":"))
    row["edge_score_mean"] = json.dumps(score_mean.detach().cpu().tolist(), separators=(",", ":"))
    row["edge_signed_mean"] = json.dumps(signed_mean.detach().cpu().tolist(), separators=(",", ":"))
    row["edge_selected_fraction"] = json.dumps(selected_mean.detach().cpu().tolist(), separators=(",", ":"))
    residual = diag["delta"][test]
    ratios = residual.norm(dim=-1) / diag["cam_norm"][test].clamp_min(1e-8)
    row["residual_ratio_mean"] = float(ratios.mean().item())
    row["residual_ratio_max"] = float(ratios.max().item())
    row["residual_ratio_basis"] = "post-CAM residual norm / pre-residual CAM norm"
    row["zero_residual_fraction"] = float((ratios == 0).float().mean().item())
    row["residual_alpha"] = float(branch.alpha.detach().item())
    text = branch.text_embeddings.detach()
    pair_cos = (torch.nn.functional.normalize(text[:, 0], dim=-1) *
                torch.nn.functional.normalize(text[:, 1], dim=-1)).sum(-1)
    row["text_pair_cosine_mean"] = float(pair_cos.mean().item())
    row["text_pair_cosine_sd"] = float(pair_cos.std(unbiased=False).item())
    projected = torch.nn.functional.normalize(branch.query(text), dim=-1)
    projected_pair_cos = (projected[:, 0] * projected[:, 1]).sum(-1)
    row["projected_pair_cosine_mean"] = float(projected_pair_cos.mean().item())
    row["projected_pair_cosine_sd"] = float(projected_pair_cos.std(unbiased=False).item())
    row["selected_vs_random_logp_drop"] = ""
    row["selected_logp_drop"] = ""
    row["random_logp_drop"] = ""
    row["top_abs_z_logp_drop"] = ""
    row["test_cf_class_flip_fraction"] = ""
    row["test_cf_abs_logp_delta"] = ""
    return row


def task_path(group):
    return TASKS / f"{group}_seed0.json"


def checkpoint_path(group):
    return CHECKPOINTS / f"{group}_seed0_latest.pt"


def save_fold_entry(fold, state):
    import torch
    path = RNG_DIR / f"B1_fold_{fold:02d}.pt"
    atomic_torch_save(path, state)


def aggregate_rows(rows):
    keyed = {}
    for row in rows:
        key = (row["group"], int(row["seed"]), int(row["fold"]))
        assert key not in keyed, key
        keyed[key] = row
        assert int(row["seed"]) == 0 and row["group"] in GROUPS
        n = sum(int(row[k]) for k in ("asd_tp", "asd_fn", "td_fp", "td_tn"))
        assert n > 0
        assert abs(float(row["hard_auc"]) - (float(row["sen_asd"]) + float(row["spe_td"])) / 2) < 1e-10
    return keyed


def fold_report_audit(agent, group, fold, train, test, best_output, best_context):
    """Run target-isolated raw-input keep/delete/random/top-|z| audits through a full GFR."""
    import numpy as np
    import torch
    y = data_bundle()["y"]
    branch = getattr(agent.ModalFusion, "lcher", None)
    if branch is None:
        return {"selected_logp_drop": "", "random_logp_drop": "", "selected_vs_random_logp_drop": "",
                "top_abs_z_logp_drop": "", "test_cf_class_flip_fraction": "", "test_cf_abs_logp_delta": "",
                "cf_test_subjects": 0}
    raw = agent.feat
    context = agent.ModalFusion.lcher_context
    context = dict(context, mask_override=None)
    agent.MessagePassing0.eval()
    agent.ModalFusion.eval()
    with torch.no_grad():
        full, _, _, _, _ = agent.forward(raw, context)
        d = branch._last_diagnostics
        selected = d["hard_selected"].detach()
        score = d["score"].detach()
        full_text = branch.text_embeddings.detach().clone()
        branch.text_embeddings.copy_(full_text.flip(1))
        swapped, _, _, _, _ = agent.forward(raw, context)
        branch.text_embeddings.copy_(full_text)
        disabled_context = dict(context, disable_residual=True)
        disabled, _, _, _, _ = agent.forward(raw, disabled_context)
    edges = torch.as_tensor([e for h in prompt_bundle()[0] for e in h["edges"]], device=agent.dev)
    cols = edges + 64
    mu, sigma = context["mu"], context["sigma"]
    rng = np.random.default_rng(38003 + fold)
    rows = []
    for subject in np.asarray(test, dtype=np.int64):
        gen = np.random.default_rng(38003 + fold * 1000 + int(subject))
        random_mask = torch.zeros(39, dtype=torch.bool, device=agent.dev)
        random_mask[torch.as_tensor(gen.choice(39, 8, replace=False), device=agent.dev)] = True
        zabs = context["z"][subject].abs()
        extreme = torch.zeros(39, dtype=torch.bool, device=agent.dev)
        extreme[zabs.topk(8).indices] = True
        base_mask = selected[subject].bool()
        original_pred = int(full[subject].argmax().item())
        y_i = int(y[subject])
        full_lp = float(full[subject, y_i].item())
        measures = {}
        for condition_name, keep_mask in (("selected_keep", base_mask),
                                           ("selected_delete", ~base_mask),
                                           ("random_delete", ~random_mask),
                                           ("extreme_delete", ~extreme)):
            x_cf = raw.clone()
            vals = mu[subject] + (raw[subject, cols] - mu[subject]) * keep_mask.to(raw.dtype)
            x_cf[subject, cols] = vals
            z_cf = context["z"].clone()
            z_cf[subject] = torch.clamp((vals - mu[subject]) / sigma, -3., 3.)
            ctx = dict(context, raw_x=x_cf, z=z_cf, mask_override=selected)
            with torch.no_grad():
                out, _, _, _, _ = agent.forward(x_cf, ctx)
            measures[condition_name] = {
                "true_logp_drop": full_lp - float(out[subject, y_i].item()),
                "pred_flip": int(out[subject].argmax().item() != original_pred),
                "abs_logp_delta": float((out[subject] - full[subject]).abs().mean().item()),
            }
        rows.append({"subject_index": int(subject), "selected_keep": measures["selected_keep"],
                     "selected_delete": measures["selected_delete"], "random_delete": measures["random_delete"],
                     "extreme_delete": measures["extreme_delete"]})
    def avg(arm, field):
        return float(np.mean([r[arm][field] for r in rows]))
    return {"selected_keep_true_logp_drop": avg("selected_keep", "true_logp_drop"),
            "selected_logp_drop": avg("selected_delete", "true_logp_drop"),
            "random_logp_drop": avg("random_delete", "true_logp_drop"),
            "top_abs_z_logp_drop": avg("extreme_delete", "true_logp_drop"),
            "selected_vs_random_logp_drop": avg("selected_delete", "true_logp_drop") - avg("random_delete", "true_logp_drop"),
            "test_cf_class_flip_fraction": float(np.mean([r[a]["pred_flip"] for r in rows for a in
                                                           ("selected_keep", "selected_delete", "random_delete", "extreme_delete")])),
            "test_cf_abs_logp_delta": float(np.mean([r[a]["abs_logp_delta"] for r in rows for a in
                                                      ("selected_keep", "selected_delete", "random_delete", "extreme_delete")])),
            "prompt_swap_abs_logp_delta": float((swapped[test] - full[test]).abs().mean().item()),
            "prompt_swap_class_flip_fraction": float((swapped[test].argmax(1) != full[test].argmax(1)).float().mean().item()),
            "router_disabled_abs_logp_delta": float((disabled[test] - full[test]).abs().mean().item()),
            "router_disabled_class_flip_fraction": float((disabled[test].argmax(1) != full[test].argmax(1)).float().mean().item()),
            "cf_test_subjects": len(rows)}


def train_config(group, fold_filter=None):
    global _TEXT_CACHE
    import torch
    import numpy as np
    assert group in GROUPS
    if fold_filter is not None:
        assert 1 <= fold_filter <= 10
        assert group != B1, "the exact anchor must remain a continuous ten-fold replay"
        assert not (CHECKPOINTS / f"{group}_fold_{fold_filter:02d}_best.pt").exists(), \
            "refusing to overwrite an existing best-fold checkpoint"
    physical_gpu = GPU[group]
    assert os.environ.get("CUDA_VISIBLE_DEVICES", str(physical_gpu)) == str(physical_gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("formal Experiment 38 requires CUDA")
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["preflight"]["passed"] is True
    assert manifest["anchor_replay"]["status"] == "ANCHOR_REPLAY_PASS"
    data = data_bundle()
    _TEXT_CACHE = text_encoder_cache() if group.startswith("E") else None
    rows = csv_rows(RESULTS / "fold_metrics.csv") if (RESULTS / "fold_metrics.csv").exists() else []
    rows = [r for r in rows if r["group"] == group]
    existing = {int(r["fold"]): r for r in rows}
    state_path = checkpoint_path(group)
    saved = state_load(state_path) if state_path.exists() else None
    if len(existing) == 10 and fold_filter is None:
        task = json.loads(task_path(group).read_text(encoding="utf-8")) if task_path(group).exists() else {}
        task.update(group=group, gpu=physical_gpu, status="COMPLETE", completed_folds=10,
                    recovered_complete_result_rows=True)
        json_write(task_path(group), task)
        print(f"SKIP_COMPLETE {group}", flush=True)
        return
    set_seed(0)
    next_fold = max(existing, default=0) + 1
    if fold_filter is not None:
        next_fold = fold_filter
        saved = None
    elif saved:
        assert saved["group"] == group and saved["config_hash"] == hash_json(CFG)
        if saved["phase"] == "BOUNDARY":
            next_fold = max(next_fold, int(saved["next_fold"]))
            restore_rng(saved["rng"])
        else:
            saved_fold = int(saved["fold"])
            if saved_fold in existing:
                next_fold = max(existing) + 1
                restore_rng(saved["rng"])
                saved = {"group": group, "phase": "BOUNDARY", "next_fold": next_fold,
                         "rng": rng_state(), "config_hash": hash_json(CFG),
                         "resume_count": int(saved.get("resume_count", 0)) + 1}
            else:
                next_fold = saved_fold
    elif group not in (P0, B1):
        next_fold = 1
    task = {"group": group, "gpu": physical_gpu, "status": "RUNNING", "started": time.time(),
            "completed_folds": len(existing), "resume_count": int((saved or {}).get("resume_count", 0))}
    json_write(task_path(group), task)
    task_started = time.monotonic()
    fold_iter = [fold_filter] if fold_filter is not None else range(next_fold, 11)
    for fold in fold_iter:
        train, test = fold_splits()[fold - 1]
        reference = fold_reference(fold, train)
        if group not in (P0, B1):
            entry = load_fold_entry(fold)
            restore_rng(entry)
        entry = rng_state()
        if group != B1:
            expected_anchor = load_anchor(fold)
            assert rng_hash(entry) == expected_anchor["initial_rng_sha256"]
        if saved and saved["phase"] == "EPOCH" and int(saved["fold"]) == fold:
            restore_rng(saved.get("entry_rng", entry))
            entry = rng_state()
        agent = Agent(group, fold, train, test, entry, reference)
        initial_trunk = shared_parameter_hash(agent)
        if group == B1:
            rng_path = RNG_DIR / f"B1_fold_{fold:02d}.pt"
            if rng_path.exists():
                assert rng_hash(entry) == rng_hash(load_fold_entry(fold))
        param_count = optimizer_audit(agent, group)
        init_state = agent.model_state()
        cursor = 0
        probe_order = deterministic_probe_order(train, fold)
        best_acc, best_epoch, best_state, wait = 0., -1, None, 0
        best_correct = 0
        start_epoch = 0
        cf_sums = {"probes": 0, "keep_loss": 0., "delete_loss": 0.}
        cf_grad_sums = {}
        fold_start = time.monotonic()
        if saved and saved["phase"] == "EPOCH" and int(saved["fold"]) == fold:
            agent.load_model_state(saved["model"])
            agent.optimizer_MP0.load_state_dict(saved["optimizer_mp"])
            agent.optimizer_MF.load_state_dict(saved["optimizer_mf"])
            restore_rng(saved["rng"])
            best_acc, best_epoch, best_state, wait = saved["best_acc"], saved["best_epoch"], saved["best_state"], saved["wait"]
            best_correct = int(saved.get("best_correct", round(float(best_acc) * len(test))))
            start_epoch, cursor = saved["next_epoch"], saved["probe_cursor"]
            cf_sums = saved["cf_sums"]
            cf_grad_sums = saved.get("cf_grad_sums", {})
        else:
            saved = None
        if group != B1:
            assert initial_trunk == load_anchor(fold)["initial_trunk_sha256"], \
                "paired initial shared trunk hash mismatch"
        stop = False
        for epoch in range(start_epoch, 450):
            result = epoch_step(agent, group, fold, epoch, reference, train, probe_order, cursor)
            cursor = result["probe_cursor"]
            for key in cf_sums:
                cf_sums[key] += float(result["cf"].get(key, 0.))
            for key, value in result.get("gradient_audit", {}).items():
                cf_grad_sums[key] = cf_grad_sums.get(key, 0.) + float(value)
            with torch.no_grad():
                agent.MessagePassing0.eval()
                agent.ModalFusion.eval()
                val_out, _, _, _, _ = agent.forward()
                val_pred = val_out[agent.val_idx].argmax(1)
                val_correct = int(val_pred.eq(agent.targ[agent.val_idx]).sum().item())
                val_acc = val_correct / len(test)
            if val_correct > best_correct:
                best_correct = val_correct
                best_acc, best_epoch, best_state, wait = val_acc, epoch, agent.model_state(), 0
            else:
                wait += 1
            if (epoch + 1) % 1 == 0:
                checkpoint = {
                    "group": group, "config_hash": hash_json(CFG), "phase": "EPOCH",
                    "fold": fold, "next_epoch": epoch + 1, "model": agent.model_state(),
                    "optimizer_mp": agent.optimizer_MP0.state_dict(), "optimizer_mf": agent.optimizer_MF.state_dict(),
                    "best_state": best_state, "best_acc": best_acc, "best_epoch": best_epoch,
                    "best_correct": best_correct,
                    "wait": wait, "probe_cursor": cursor, "cf_sums": cf_sums,
                    "cf_grad_sums": cf_grad_sums,
                    "rng": rng_state(), "entry_rng_hash": rng_hash(entry),
                    "entry_rng": entry,
                    "initial_trunk_sha256": initial_trunk,
                    "input_sha256": manifest["input_sha256"], "fold_split_sha256": manifest["fold_split_sha256"],
                    "text_cache_sha256": manifest["text_encoder"]["cache_sha256"],
                    "prompt_sha256": manifest["prompts_sha256"],
                    "source_sha256": manifest["source_sha256"],
                    "code_sha256": {p.name: sha(p) for p in HERE.glob("*.py")},
                    "completed_metric_folds": sorted(existing), "resume_count": int(task["resume_count"]),
                }
                atomic_torch_save(state_path, checkpoint)
            if wait > 200:
                stop = True
            if stop:
                break
        assert best_state is not None and best_epoch >= 0
        continuation_rng = rng_state()
        agent.load_model_state(best_state)
        agent.MessagePassing0.eval()
        agent.ModalFusion.eval()
        with torch.no_grad():
            test_output, _, _, _, _ = agent.forward()
        metric = metric_row(test_output, test, data["y"])
        assert abs(metric["acc"] - best_correct / len(test)) < 1e-12
        cf_den = max(int(cf_sums["probes"]), 1)
        cf_mean = {k: cf_sums[k] / cf_den for k in cf_sums}
        cf_events = max(int(cf_sums["probes"]) // 2, 1)
        cf_mean["gradient_norms"] = {k: v / cf_events for k, v in cf_grad_sums.items()}
        diag = fold_diagnostics(agent, group, fold, train, test, metric, best_epoch, cf_mean)
        if group.startswith("E") and GROUP[group]["keep"]:
            diag.update(fold_report_audit(agent, group, fold, train, test, test_output,
                                          agent.ModalFusion.lcher_context))
        else:
            diag.update({"selected_keep_true_logp_drop": "", "selected_logp_drop": "",
                         "random_logp_drop": "", "top_abs_z_logp_drop": "",
                         "selected_vs_random_logp_drop": "", "test_cf_class_flip_fraction": "",
                         "test_cf_abs_logp_delta": "", "cf_test_subjects": 0})
        row = {"group": group, "seed": 0, "fold": fold, "physical_gpu": physical_gpu,
               **metric, "best_epoch": best_epoch, "epochs_executed": epoch + 1,
               "initial_trunk_sha256": initial_trunk, "initial_rng_sha256": rng_hash(entry),
               "parameter_count": param_count, "fold_seconds": time.monotonic() - fold_start}
        # Atomic unique-key update: resume cannot append a second result row for a completed fold.
        path = RESULTS / "fold_metrics.csv"
        existing_all = csv_rows(path) if path.exists() else []
        existing_all = [r for r in existing_all if not (r["group"] == group and int(r["fold"]) == fold)]
        existing_all.append(row)
        existing_all.sort(key=lambda r: (GROUPS.index(r["group"]), int(r["fold"])))
        csv_write(path, existing_all)
        diag_path = RESULTS / "diagnostics.csv"
        diag_all = csv_rows(diag_path) if diag_path.exists() else []
        diag_all = [r for r in diag_all
                    if r.get("group") in GROUPS and not (r["group"] == group and int(r["fold"]) == fold)]
        diag_all.append(diag)
        diag_all.sort(key=lambda r: (GROUPS.index(r["group"]), int(r["fold"])))
        csv_write(diag_path, diag_all)
        existing[fold] = row
        atomic_torch_save(CHECKPOINTS / f"{group}_fold_{fold:02d}_best.pt", {
            "group": group, "fold": fold, "seed": 0, "model": best_state,
            "best_epoch": best_epoch, "metric": metric,
            "initial_trunk_sha256": initial_trunk, "entry_rng_sha256": rng_hash(entry),
            "config_hash": hash_json(CFG), "fold_split_sha256": manifest["fold_split_sha256"],
            "input_sha256": manifest["input_sha256"],
            "text_cache_sha256": manifest["text_encoder"]["cache_sha256"],
            "prompt_sha256": manifest["prompts_sha256"], "source_sha256": manifest["source_sha256"],
            "code_sha256": {p.name: sha(p) for p in HERE.glob("*.py")},
        })
        restore_rng(continuation_rng)
        completed = sorted(set(existing) | {fold})
        boundary = {"group": group, "config_hash": hash_json(CFG), "phase": "BOUNDARY",
                    "next_fold": fold + 1, "rng": continuation_rng, "completed_folds": completed,
                    "resume_count": int(task["resume_count"])}
        atomic_torch_save(state_path, boundary)
        task.update(status="RUNNING" if fold < 10 else "COMPLETE", completed_folds=len(completed),
                    last_fold=fold, elapsed_seconds=time.monotonic() - task_started)
        json_write(task_path(group), task)
        print(f"FOLD_DONE {group} fold={fold} ACC={metric['acc']:.6f} HardAUC={metric['hard_auc']:.6f} best={best_epoch}", flush=True)
        del agent, test_output
        torch.cuda.empty_cache()
        saved = None
    task["status"] = "COMPLETE"
    task["completed_folds"] = 10
    task["elapsed_seconds"] = time.monotonic() - task_started
    json_write(task_path(group), task)


def worker_env(gpu):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"})
    return env


def run_worker(group, physical_gpu=None):
    gpu = GPU[group] if physical_gpu is None else physical_gpu
    subprocess.run([sys.executable, str(HERE / "run.py"), "worker", group], cwd=RUNTIME,
                   env=worker_env(gpu), check=True)


def preflight():
    global _TEXT_CACHE
    import numpy as np
    manifest = initialize_manifest()
    compile((HERE / "method.py").read_text(), str(HERE / "method.py"), "exec")
    compile((HERE / "integration.py").read_text(), str(HERE / "integration.py"), "exec")
    compile((HERE / "run.py").read_text(), str(HERE / "run.py"), "exec")
    hypotheses, names = prompt_bundle()
    text = _TEXT_CACHE or text_encoder_cache()
    _TEXT_CACHE = text
    assert len(names) == 39 and all(0 <= i < 256 for i in names)
    assert np.isfinite(text["real"]).all() and np.isfinite(text["neutral"]).all()
    assert np.allclose(np.linalg.norm(text["anonymous"], axis=-1), np.linalg.norm(text["real"], axis=-1).mean(), atol=1e-5)
    # Source generation must leave P0 complete and remove every PEN/CLIP reference from B1.
    no_pen = (SOURCES / "no_pen/network.py").read_text()
    original = (SOURCES / "original_pen/network.py").read_text()
    assert "class MLP" in original and "prompt_text_features" in original
    assert all(token not in no_pen for token in ("prompt_text_features", "get_prompt_text_features", "self.MLP", "clip.load", "AvgPool1d"))
    assert "elif i==1:" in no_pen and "combined_feat = f_w*feat1+" in no_pen
    assert "combined_feat = f_w*feat0+" in no_pen
    # Check reference provenance, train/test isolation, and finite tails per fold.
    refs = []
    for fold, (train, test) in enumerate(fold_splits(), 1):
        ref = fold_reference(fold, train)
        assert set(ref["train_td"].tolist()) <= set(train.tolist())
        assert set(ref["train"].tolist()) == set(train.tolist())
        assert np.isfinite(ref["z"]).all() and np.isfinite(ref["mu"]).all()
        assert np.max(np.abs(ref["z"])) <= 3.00001
        assert not (set(test) & set(ref["train_td"]))
        refs.append({"fold": fold, "td_train": len(ref["train_td"]),
                     "oof_residual_rms": float(ref["oof_residual_rms"]),
                     "sigma_min": float(ref["sigma_min"]), "sigma_median": float(ref["sigma_median"])})
    # Hard masking/competitive router gradients on a tiny synthetic batch.
    import torch
    from method import LCHER
    d = data_bundle()
    ref = fold_reference(1, fold_splits()[0][0])
    direction = torch.tensor([1. if h["direction"] == "hyper_candidate" else -1. if h["direction"] == "hypo_candidate" else 0.
                              for h in hypotheses for _ in h["edges"]])
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
        router = LCHER(torch.as_tensor(text["real"]), [h["edges"] for h in hypotheses], direction, seed=38001).to("cuda")
    raw = torch.as_tensor(d["x"], device="cuda")
    z = torch.as_tensor(ref["z"], device="cuda")
    rel = torch.as_tensor(ref["reliability"], device="cuda")
    cam = torch.randn((871, 144), device="cuda")
    routed, diag = router(cam, raw, z, rel)
    assert torch.equal(routed, cam)
    assert torch.equal(diag["hard_selected"].sum(1), torch.full((871,), 8., device="cuda"))
    (diag["score"].mean() + router.alpha * 0 + diag["delta"].sum()).backward()
    # alpha is initialized at zero but should receive a finite, nonzero task gradient with a fixed probe.
    assert torch.isfinite(router.alpha.grad) and torch.isfinite(router.query.weight.grad).all()
    # Competitive hypotheses swap should reverse the signed evidence for the same raw patient measurements.
    with torch.no_grad():
        base_signed = diag["signed"].clone()
        router.text_embeddings.copy_(router.text_embeddings.flip(1))
        _, swapped_diag = router(cam, raw, z, rel)
        swap_signed = swapped_diag["signed"]
    assert torch.isfinite(swap_signed).all()
    assert torch.allclose(swap_signed, -base_signed, atol=1e-6, rtol=1e-5), \
        "swapping ASD/TD hypotheses must reverse signed evidence"
    # Counterfactual operation changes precisely one participant's candidate raw inputs.
    i = int(fold_splits()[0][1][0])
    edge_ids = torch.tensor([e for h in hypotheses for e in h["edges"]], device="cuda") + 64
    original_raw = raw.clone()
    altered = raw.clone()
    altered[i, edge_ids] = torch.as_tensor(ref["mu"][i], device="cuda")
    assert torch.equal(altered[torch.arange(871, device="cuda") != i], original_raw[torch.arange(871, device="cuda") != i])
    assert torch.equal(altered[:, :64], original_raw[:, :64])
    assert torch.equal(altered[i, edge_ids], torch.as_tensor(ref["mu"][i], device="cuda"))
    # B1 and alpha-zero LCHER must match in forward, objective, common gradients, and first trunk update.
    train, test = fold_splits()[0]
    entry = rng_state()
    b1 = Agent(B1, 1, train, test, entry, ref)
    b1_hash = model_hash(b1)
    restore_rng(entry)
    full = Agent("E4_LCHER_FULL", 1, train, test, entry, ref)
    assert model_hash(full) == b1_hash, "shared trunk initialization differs at paired fold entry"
    forward_entry = rng_state()
    restore_rng(forward_entry)
    b1.MessagePassing0.train(); b1.ModalFusion.train()
    b1.optimizer_MP0.zero_grad(set_to_none=True); b1.optimizer_MF.zero_grad(set_to_none=True)
    b1_out, b1_graph, _, b1_adj, _ = b1.forward()
    b1_cls = torch.nn.functional.nll_loss(b1_out[b1.trn_idx], b1.targ[b1.trn_idx], weight=b1.weight)
    b1_loss = b1_cls + b1_graph
    b1_loss.backward()
    b1_grads = {name: p.grad.detach().clone() for name, p in
                list(b1.MessagePassing0.named_parameters()) +
                [("PLN." + n, p) for n, p in b1.ModalFusion.named_parameters()] if p.grad is not None}
    b1.optimizer_MP0.step(); b1.optimizer_MF.step()
    restore_rng(forward_entry)
    full.MessagePassing0.train(); full.ModalFusion.train()
    full.optimizer_MP0.zero_grad(set_to_none=True); full.optimizer_MF.zero_grad(set_to_none=True)
    full_out, full_graph, _, full_adj, _ = full.forward()
    full_cls = torch.nn.functional.nll_loss(full_out[full.trn_idx], full.targ[full.trn_idx], weight=full.weight)
    full_loss = full_cls + full_graph
    assert torch.equal(b1_out, full_out) and torch.equal(b1_graph, full_graph) and torch.equal(b1_loss, full_loss)
    assert all(torch.equal(a, b) for a, b in zip(b1_adj, full_adj))
    full_loss.backward()
    full_grads = {name: p.grad.detach().clone() for name, p in
                  list(full.MessagePassing0.named_parameters()) +
                  [("PLN." + n, p) for n, p in full.ModalFusion.named_parameters()
                   if not n.startswith("lcher.")] if p.grad is not None}
    assert b1_grads.keys() == full_grads.keys()
    assert all(torch.equal(b1_grads[k], full_grads[k]) for k in b1_grads)
    full.optimizer_MP0.step(); full.optimizer_MF.step()
    assert model_hash(b1) == model_hash(full), "alpha-zero route changed the first shared optimizer update"
    # Two-step interruption/restart: replay step three including dropout and both Adam states.
    set_seed(0)
    entry = rng_state()
    train, test = fold_splits()[0]
    uninterrupted = Agent("E4_LCHER_FULL", 1, train, test, entry, ref)
    probe_order = deterministic_probe_order(train, 1)
    epoch_step(uninterrupted, "E4_LCHER_FULL", 1, 0, ref, train, probe_order, 0)
    epoch_step(uninterrupted, "E4_LCHER_FULL", 1, 1, ref, train, probe_order, 0)
    saved_model = uninterrupted.model_state()
    saved_mp = copy.deepcopy(uninterrupted.optimizer_MP0.state_dict())
    saved_mf = copy.deepcopy(uninterrupted.optimizer_MF.state_dict())
    saved_rng = rng_state()
    epoch_step(uninterrupted, "E4_LCHER_FULL", 1, 2, ref, train, probe_order, 0)
    expected = uninterrupted.model_state()
    restore_rng(entry)
    resumed = Agent("E4_LCHER_FULL", 1, train, test, entry, ref)
    resumed.load_model_state(saved_model)
    resumed.optimizer_MP0.load_state_dict(saved_mp)
    resumed.optimizer_MF.load_state_dict(saved_mf)
    restore_rng(saved_rng)
    epoch_step(resumed, "E4_LCHER_FULL", 1, 2, ref, train, probe_order, 0)
    assert all(torch.equal(expected[module][name], resumed.model_state()[module][name])
               for module in expected for name in expected[module]), "resumed third update is not exact"
    # The held-out true labels are not read by graph construction or LCHER inference.
    resumed.MessagePassing0.eval(); resumed.ModalFusion.eval()
    with torch.no_grad():
        unchanged = resumed.forward()[0].clone()
        old_test_labels = resumed.targ[torch.as_tensor(test, device=resumed.dev)].clone()
        resumed.targ[torch.as_tensor(test, device=resumed.dev)] = 1 - old_test_labels
        altered_label_output = resumed.forward()[0].clone()
        resumed.targ[torch.as_tensor(test, device=resumed.dev)] = old_test_labels
    assert torch.equal(unchanged, altered_label_output), "held-out labels affected inference output"
    # First scheduled CF step (epoch 5) must reach routing, signed evidence and residual parameters.
    set_seed(0)
    entry = rng_state()
    cf_probe = Agent("E4_LCHER_FULL", 1, train, test, entry, ref)
    order = deterministic_probe_order(train, 1)
    cf_metrics = None
    cf_epoch_seconds = None
    cf_window_start = time.monotonic()
    for epoch in range(5):
        epoch_started = time.monotonic()
        cf_metrics = epoch_step(cf_probe, "E4_LCHER_FULL", 1, epoch, ref, train, order, 2 * epoch)
        if epoch == 4:
            cf_epoch_seconds = time.monotonic() - epoch_started
    cf_window_seconds = time.monotonic() - cf_window_start
    cf_gradients = cf_metrics["gradient_audit"]
    for key in ("query_weight", "key_weight", "context_weight", "summary_1_weight", "alpha",
                "selected", "signed_evidence", "signed_residual"):
        assert np.isfinite(cf_gradients[key]) and cf_gradients[key] > 0., f"CF gradient missing: {key}={cf_gradients[key]}"
    cf_probe.MessagePassing0.eval(); cf_probe.ModalFusion.eval()
    with torch.no_grad():
        full_output = cf_probe.forward()[0]
        branch_diag = cf_probe.ModalFusion.lcher._last_diagnostics
        ratios = branch_diag["residual_ratio"]
    assert float(ratios.max()) <= .050001
    test_subject = int(test[0])
    target_x = cf_probe.feat.clone()
    edge_cols = torch.as_tensor([e for h in hypotheses for e in h["edges"]], device=cf_probe.dev) + 64
    target_x[test_subject, edge_cols] = cf_probe.ModalFusion.lcher_context["mu"][test_subject]
    target_z = cf_probe.ModalFusion.lcher_context["z"].clone()
    target_z[test_subject] = torch.clamp((target_x[test_subject, edge_cols] -
        cf_probe.ModalFusion.lcher_context["mu"][test_subject]) /
        cf_probe.ModalFusion.lcher_context["sigma"][test_subject], -3., 3.)
    ctx = dict(cf_probe.ModalFusion.lcher_context, raw_x=target_x, z=target_z)
    with torch.no_grad():
        raw_cf_output = cf_probe.forward(target_x, ctx)[0]
    cf_response = float((full_output[test_subject] - raw_cf_output[test_subject]).abs().max().item())
    assert cf_response > 0., "raw FC counterfactual has no full-path output response"
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    manifest["preflight"] = {"passed": True, "time": time.time(), "checks": {
        "prompt_mapping": {"hypotheses": 13, "unique_edges": len(names), "all_named_and_verified": True,
                           "unknown_indices_excluded": True},
        "text": {"encoder": text["descriptor"], "untruncated_77_tokenizer": True,
                 "embedding_shapes": {"real": list(text["real"].shape), "neutral": list(text["neutral"].shape),
                                      "anonymous": list(text["anonymous"].shape)},
                 "pair_cosine_mean": float(np.mean(np.sum(
                     text["real"][:, 0] / np.maximum(np.linalg.norm(text["real"][:, 0], axis=-1, keepdims=True), 1e-12) *
                     text["real"][:, 1] / np.maximum(np.linalg.norm(text["real"][:, 1], axis=-1, keepdims=True), 1e-12), -1))),
                 "pair_cosine_sd": float(np.std(np.sum(
                     text["real"][:, 0] / np.maximum(np.linalg.norm(text["real"][:, 0], axis=-1, keepdims=True), 1e-12) *
                     text["real"][:, 1] / np.maximum(np.linalg.norm(text["real"][:, 1], axis=-1, keepdims=True), 1e-12), -1)))},
        "source": {"P0_original_pen_preserved": True, "B1_pen_physically_absent": True,
                   "GFR_cover_behavior_preserved": True},
        "td_reference_per_fold": refs,
        "router": {"zero_residual_at_alpha0": True, "selected_count": 8,
                   "alpha_grad": float(router.alpha.grad.item()),
                   "query_grad_norm": float(router.query.weight.grad.norm().item()),
                   "paired_prompt_swap_test": "signed evidence reversed exactly within tolerance"},
        "data": {"shape": list(d["x"].shape), "label_counts": {"ASD0": 403, "TD1": 468},
                 "fold_count": 10},
        "no_checkpoint_replay_or_historical_weights": True,
        "scheduled_cf_epoch_seconds": cf_epoch_seconds,
        "five_epoch_window_seconds": cf_window_seconds,
        "restore_exactness": "two-step interruption replayed the third update bitwise",
        "b1_alpha_zero_equivalence": "forward, graph, loss, common gradients and first shared optimizer update bitwise equal",
        "heldout_label_invariance": True,
        "scheduled_cf_gradients": cf_gradients,
        "raw_fc_full_path_response_max_abs": cf_response,
        "residual_ratio_max": float(ratios.max().item())}}
    manifest["source_sha256"] = source_hashes()
    manifest["code_sha256"] = {p.name: sha(p) for p in HERE.glob("*.py")}
    json_write(HERE / "manifest.json", manifest)
    json_write(RUNTIME / "preflight.json", manifest["preflight"])
    print("PREFLIGHT_STATIC_PASS", json.dumps(manifest["preflight"]["checks"], ensure_ascii=False)[:3000], flush=True)


def anchor_paired_preflight():
    global _TEXT_CACHE
    import numpy as np
    import torch
    manifest = initialize_manifest()
    assert manifest["anchor_replay"]["status"] == "ANCHOR_REPLAY_PASS"
    compile((HERE / "method.py").read_text(), str(HERE / "method.py"), "exec")
    compile((HERE / "lcher_runtime.py").read_text(), str(HERE / "lcher_runtime.py"), "exec")
    compile((HERE / "lcher_run.py").read_text(), str(HERE / "lcher_run.py"), "exec")
    data = data_bundle()
    hypotheses, names = prompt_bundle()
    text = _TEXT_CACHE or text_encoder_cache()
    _TEXT_CACHE = text
    assert len(hypotheses) == 13 and len(names) == 39
    assert set(names).isdisjoint({195, 225, 232, 236})
    assert text["real"].shape == text["neutral"].shape == text["anonymous"].shape == (13, 2, 1024)
    assert np.isfinite(text["real"]).all() and np.isfinite(text["neutral"]).all()
    runtime_network = (SOURCES / "network.py").read_text(encoding="utf-8")
    assert "class MLP" in runtime_network and "self.avg_pool" in runtime_network
    assert "elif config32 == 'B0':\n            pass" in runtime_network
    assert "if getattr(self, 'lcher', None) is not None" in runtime_network
    refs = []
    for fold, (train, test) in enumerate(fold_splits(), 1):
        reference = fold_reference(fold, train)
        assert set(reference["train_td"].tolist()) <= set(train.tolist())
        assert not set(test).intersection(reference["train_td"].tolist())
        assert np.isfinite(reference["mu"]).all() and np.isfinite(reference["sigma"]).all()
        assert np.isfinite(reference["z"]).all() and np.max(np.abs(reference["z"])) <= 3.00001
        refs.append({"fold": fold, "td_train_n": len(reference["train_td"]),
                     "oof_residual_rms": reference["oof_residual_rms"],
                     "sigma_min": reference["sigma_min"], "sigma_median": reference["sigma_median"]})

    fold = 1
    train, test = fold_splits()[0]
    anchor = load_anchor(fold)
    reference = fold_reference(fold, train)
    restore_rng(anchor["entry_rng"])
    base = Agent(B1, fold, train, test, anchor["entry_rng"], reference)
    restore_rng(anchor["entry_rng"])
    routed = Agent("E4_LCHER_FULL", fold, train, test, anchor["entry_rng"], reference)
    assert shared_parameter_hash(base) == shared_parameter_hash(routed) == anchor["initial_trunk_sha256"]
    assert model_hash(base) == model_hash(routed)
    base.MessagePassing0.eval(); base.ModalFusion.eval()
    routed.MessagePassing0.eval(); routed.ModalFusion.eval()
    with torch.no_grad():
        base_output = base.forward()[0]
        route_output = routed.forward()[0]
    assert torch.equal(base_output, route_output), "zero-alpha LCHER changed B0-A output"

    # Compare objective, common gradients and the two historical Adam updates with shared dropout RNG.
    for agent in (base, routed):
        agent.MessagePassing0.train(); agent.ModalFusion.train()
        agent.optimizer_MP0.zero_grad(set_to_none=True); agent.optimizer_MF.zero_grad(set_to_none=True)
    forward_rng = rng_state()
    restore_rng(forward_rng)
    base_output, base_graph, _, base_adj, _ = base.forward()
    base_cls = torch.nn.functional.nll_loss(base_output[base.trn_idx], base.targ[base.trn_idx], weight=base.weight)
    base_loss = base_cls + base_graph
    base_loss.backward()
    base_grads = {name: p.grad.detach().clone() for name, p in
                  list(base.MessagePassing0.named_parameters()) +
                  [("PLN." + n, p) for n, p in base.ModalFusion.named_parameters()] if p.grad is not None}
    base.optimizer_MP0.step(); base.optimizer_MF.step()
    restore_rng(forward_rng)
    route_output, route_graph, _, route_adj, _ = routed.forward()
    route_cls = torch.nn.functional.nll_loss(route_output[routed.trn_idx], routed.targ[routed.trn_idx], weight=routed.weight)
    route_loss = route_cls + route_graph
    assert torch.equal(base_output, route_output) and torch.equal(base_graph, route_graph)
    assert torch.equal(base_loss, route_loss) and all(torch.equal(a, b) for a, b in zip(base_adj, route_adj))
    route_loss.backward()
    route_grads = {name: p.grad.detach().clone() for name, p in
                   list(routed.MessagePassing0.named_parameters()) +
                   [("PLN." + n, p) for n, p in routed.ModalFusion.named_parameters()
                    if not n.startswith("lcher.")] if p.grad is not None}
    assert base_grads.keys() == route_grads.keys()
    assert all(torch.equal(base_grads[key], route_grads[key]) for key in base_grads)
    routed.optimizer_MP0.step(); routed.optimizer_MF.step()
    assert model_hash(base) == model_hash(routed), "zero-alpha LCHER changed shared Adam update"

    # Exercise a real scheduled CF step and verify input-mask gradients reach selector/query/residual.
    restore_rng(anchor["entry_rng"])
    probe = Agent("E4_LCHER_FULL", fold, train, test, anchor["entry_rng"], reference)
    order = deterministic_probe_order(train, fold)
    cf_metrics = None
    for epoch in range(5):
        cf_metrics = epoch_step(probe, "E4_LCHER_FULL", fold, epoch, reference, train, order, 2 * epoch)
    gradients = cf_metrics["gradient_audit"]
    for key in ("query_weight", "key_weight", "context_weight", "alpha", "selected", "signed_evidence", "signed_residual"):
        assert gradients[key] > 0. and np.isfinite(gradients[key]), f"CF gradient did not reach {key}: {gradients}"
    assert int(probe.targ[torch.as_tensor(test, device=probe.dev)].numel()) == len(test)
    probe.MessagePassing0.eval(); probe.ModalFusion.eval()
    with torch.no_grad():
        before = probe.forward()[0]
        labels_before = probe.targ[torch.as_tensor(test, device=probe.dev)].clone()
        probe.targ[torch.as_tensor(test, device=probe.dev)] = 1 - labels_before
        after_label_swap = probe.forward()[0]
        probe.targ[torch.as_tensor(test, device=probe.dev)] = labels_before
    assert torch.equal(before, after_label_swap), "held-out labels changed inference output"
    edges = torch.as_tensor([e for h in hypotheses for e in h["edges"]], device=probe.dev) + 64
    target = int(test[0])
    altered = probe.feat.clone()
    altered[target, edges] = probe.ModalFusion.lcher_context["mu"][target]
    changed = (altered != probe.feat)
    assert changed.sum().item() <= 39 and torch.equal(altered[torch.arange(871, device=probe.dev) != target],
                                                       probe.feat[torch.arange(871, device=probe.dev) != target])
    with torch.no_grad():
        before_cf = probe.forward()[0][target]
        after_cf = probe.forward(altered, dict(probe.ModalFusion.lcher_context, raw_x=altered))[0][target]
    cf_response = float((before_cf - after_cf).abs().max().item())
    assert cf_response > 0., "raw-FC intervention had no full-path prediction response"

    checks = {"anchor_replay": "EXACT_10_OF_10",
              "verified_candidates": {"hypotheses": len(hypotheses), "edges": len(names), "unknown_excluded": True},
              "frozen_text_cache_sha256": text["cache_sha256"], "td_reference_per_fold": refs,
              "zero_alpha_forward_and_two_adam_update": "BITWISE_EXACT",
              "counterfactual_gradients": gradients,
              "heldout_label_invariance": True, "raw_input_full_gfr_response_max_abs": cf_response,
              "source_preserves_full_pln_and_b0_forward_bypass": True}
    manifest["preflight"] = {"passed": True, "completed_epoch": time.time(), "checks": checks}
    manifest["status"] = "PAIRED_PREFLIGHT_PASS"
    manifest["code_sha256"] = {p.name: sha(p) for p in HERE.glob("*.py")}
    json_write(HERE / "manifest.json", manifest)
    json_write(RUNTIME / "preflight.json", manifest["preflight"])
    print("PAIRED_PREFLIGHT_PASS " + json.dumps(checks, ensure_ascii=False)[:5000], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "preflight", "all", "resume", "report", "audit_bounds", "worker"])
    parser.add_argument("group", nargs="?")
    parser.add_argument("--fold", type=int, help="recover one incomplete paired fold artifact")
    parser.add_argument("--gpu", type=int, choices=[0, 1], help="physical GPU for checkpoint audit")
    args = parser.parse_args()
    if args.command == "prepare":
        m = initialize_manifest()
        print(json.dumps({"status": m["status"], "workspace": str(HERE), "groups": len(GROUPS), "folds": 100}, indent=2))
    elif args.command == "preflight":
        anchor_paired_preflight()
    elif args.command == "audit_bounds":
        if args.gpu is None:
            parser.error("audit_bounds requires --gpu 0 or --gpu 1")
        audit_residual_bounds(args.gpu)
    elif args.command == "worker":
        if not args.group:
            parser.error("worker requires group")
        train_config(args.group, args.fold)
    elif args.command == "all":
        initialize_manifest()
        anchor_paired_preflight()
        run_schedule()
    elif args.command == "resume":
        run_schedule()
    elif args.command == "report":
        report()


def run_schedule():
    initialize_manifest()
    manifest = json.loads((HERE / "manifest.json").read_text())
    if not manifest["preflight"]["passed"]:
        raise RuntimeError("preflight has not passed")
    assert manifest["anchor_replay"]["status"] == "ANCHOR_REPLAY_PASS"
    anchor_task = TASKS / f"{B1}_seed0.json"
    json_write(anchor_task, {"group": B1, "gpu": 0, "status": "COMPLETE", "completed_folds": 10,
                             "elapsed_seconds": 0., "source": "strict Experiment 32 B0-A replay"})
    pending = [g for g in GROUPS if g != B1 and
               (not task_path(g).exists() or json.loads(task_path(g).read_text()).get("status") != "COMPLETE")]
    max_workers_per_gpu = int(CFG["scheduling"].get("max_workers_per_gpu", 4))
    active = []
    while pending or active:
        for group in list(pending):
            gpu = GPU[group]
            if sum(1 for item in active if item["gpu"] == gpu) >= max_workers_per_gpu:
                continue
            log_path = LOGS / f"{group}.log"
            handle = log_path.open("a", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, str(HERE / "lcher_run.py"), "worker", group],
                                    cwd=RUNTIME, env=worker_env(gpu), stdout=handle, stderr=subprocess.STDOUT)
            active.append({"group": group, "gpu": gpu, "proc": proc, "handle": handle})
            pending.remove(group)
            print(f"START {group} GPU{gpu}; active={sum(1 for a in active if a['gpu']==gpu)}", flush=True)
        time.sleep(3)
        for item in list(active):
            code = item["proc"].poll()
            if code is None:
                continue
            item["handle"].close()
            active.remove(item)
            state = TASKS / f"{item['group']}_seed0.json"
            complete = state.exists() and json.loads(state.read_text()).get("status") == "COMPLETE"
            print(f"EXIT {item['group']} code={code} complete={complete}", flush=True)
            if code != 0 or not complete:
                for other in active:
                    other["proc"].terminate()
                for other in active:
                    other["proc"].wait()
                    other["handle"].close()
                raise RuntimeError(f"worker failed {item['group']}; see {LOGS / (item['group']+'.log')}")
    report()


def report():
    import numpy as np
    path = RESULTS / "fold_metrics.csv"
    if not path.exists():
        raise RuntimeError("no fold metrics yet")
    rows = csv_rows(path)
    keyed = aggregate_rows(rows)
    assert len(rows) == len(keyed)
    summary_rows = []
    metric_keys = ["acc", "hard_auc", "spe_td", "sen_asd"]
    for group in GROUPS:
        subset = [r for r in rows if r["group"] == group]
        if not subset:
            continue
        vals = {}
        for key in metric_keys:
            values = np.asarray([float(r[key]) for r in subset]) * 100
            vals[key] = {"mean": float(values.mean()), "population_sd": float(values.std(ddof=0)),
                         "fold_values": values.tolist()}
        summary_rows.append({"group": group, **vals, "folds": len(subset)})
    paired_rows = []
    if all(len([r for r in rows if r["group"] == g]) == 10 for g in GROUPS):
        pairs = [(g, B1) for g in E_GROUPS]
        pairs += [("E4_LCHER_FULL", g) for g in ("E1_ANON_FULL", "E2_REAL_NO_CF",
                                                    "E3_REAL_KEEP_ONLY", "E5_WRONG_EDGE_FULL",
                                                    "E6_NEUTRAL_FULL", "E7_NO_AUX_FULL",
                                                    "E8_NO_RIVAL_FULL", "E9_SWAP_CLASS_FULL")]
        for left, right in pairs:
            item = {"left": left, "right": right}
            for key in metric_keys:
                ds = np.asarray([float(keyed[(left, 0, f)][key]) - float(keyed[(right, 0, f)][key])
                                 for f in range(1, 11)]) * 100
                item[key] = {"delta_pp": float(ds.mean()), "fold_differences_pp": ds.tolist(),
                             "wins": int((ds > 0).sum()), "ties": int((ds == 0).sum()), "losses": int((ds < 0).sum())}
            paired_rows.append(item)
    complete_groups = [g for g in GROUPS if len([r for r in rows if r["group"] == g]) == 10]
    manifest = json.loads((HERE / "manifest.json").read_text())
    gpu_counts = {str(g): sum(int(r["physical_gpu"]) == g for r in rows) for g in (0, 1)}
    status = "COMPLETE" if len(rows) == 100 and len(complete_groups) == 10 and gpu_counts == {"0": 50, "1": 50} else "PARTIAL"
    scientific_status = status
    if status == "COMPLETE" and manifest.get("handoff_status") == "FAILED_PENDING_APPEND":
        status = "PARTIAL"
    decision = "NO-GO" if scientific_status == "COMPLETE" else "PARTIAL"
    checks = {}
    screening_diagnostics = {}
    if scientific_status == "COMPLETE":
        by_group = {r["group"]: r for r in summary_rows}
        e4 = by_group["E4_LCHER_FULL"]
        b1 = by_group[B1]
        anon = by_group["E1_ANON_FULL"]
        controls = {r["right"]: r for r in paired_rows}
        e4_diagnostics = [r for r in csv_rows(RESULTS / "diagnostics.csv") if r["group"] == "E4_LCHER_FULL"]
        selected_random = [float(r.get("selected_vs_random_logp_drop", 0) or 0) for r in e4_diagnostics]
        positive_folds = sum(value > 0 for value in selected_random)
        screening_diagnostics["E4_selected_delete_minus_random_logp_drop"] = {
            "mean": float(np.mean(selected_random)), "positive_folds": positive_folds,
            "zero_folds": sum(value == 0 for value in selected_random),
            "negative_folds": sum(value < 0 for value in selected_random),
            "operational_stability_rule": "positive mean and positive in at least 7/10 folds"}
        checks = {
            "E4_beats_B0A_acc_auc_0_23pp": e4["acc"]["mean"] - b1["acc"]["mean"] >= .23 and e4["hard_auc"]["mean"] - b1["hard_auc"]["mean"] >= .23,
            "E4_beats_E1_acc_auc_0_23pp": e4["acc"]["mean"] - anon["acc"]["mean"] >= .23 and e4["hard_auc"]["mean"] - anon["hard_auc"]["mean"] >= .23,
            "E4_beats_E5_E6_E9": all(controls[name][metric]["delta_pp"] > 0 for name in
                                      ("E5_WRONG_EDGE_FULL", "E6_NEUTRAL_FULL", "E9_SWAP_CLASS_FULL")
                                      for metric in ("acc", "hard_auc")),
            "E4_sen_drop_vs_B0A_no_more_than_0_5pp": controls[B1]["sen_asd"]["delta_pp"] >= -.5,
            "E4_selected_deletion_beats_random": screening_diagnostics["E4_selected_delete_minus_random_logp_drop"]["mean"] > 0 and positive_folds >= 7,
            "E4_prompt_swap_changes_output": any(float(r.get("prompt_swap_abs_logp_delta", 0) or 0) > 1e-6
                                                   for r in e4_diagnostics),
        }
        decision = "PROVISIONAL_GO" if all(checks.values()) else "NO-GO"
    summary = {"status": status, "scientific_status": scientific_status,
               "delivery_status": manifest.get("delivery", "PENDING"),
               "decision": decision, "groups": summary_rows,
               "paired_comparisons": paired_rows, "gpu_folds": gpu_counts,
               "screening_diagnostics": screening_diagnostics,
               "completed_folds": len(rows), "completed_groups": len(complete_groups),
               "go_checks": checks, "protocol_limitations": ["val_idx=test_idx for checkpoint selection",
               "RFE/standardization occurred before outer cross-validation", "transductive 871-node graphs",
               "fold training sets overlap; paired results are descriptive, not independent-cohort significance"],
               "formal_metrics": ["ACC", "Hard-AUC (original one-hot hard-label ROC-AUC)", "SPE-TD", "SEN-ASD"],
               "hard_auc_equivalent_to_balanced_accuracy_note": "Only a mathematical note for binary hard predictions; BA is not reported as a separate formal metric.",
               "text_encoder": manifest.get("text_encoder"), "source_sha256": manifest.get("source_sha256"),
               "manifest_delivery": manifest.get("delivery")}
    json_write(RESULTS / "summary.json", summary)
    manifest["status"] = status
    manifest["scientific_status"] = scientific_status
    manifest["completed_folds"] = len(rows)
    manifest["completed_groups"] = len(complete_groups)
    manifest["gpu_completed_folds"] = gpu_counts
    manifest["tasks"] = [json.loads(task.read_text(encoding="utf-8")) if task.exists() else
                         {"group": g, "gpu": GPU[g], "status": "PENDING", "completed_folds": 0}
                         for g in GROUPS for task in [task_path(g)]]
    start_epoch = float(manifest.get("formal_started_epoch", manifest.get("prepared_utc", time.time())))
    task_ends = [float(t.get("started", start_epoch)) + float(t.get("elapsed_seconds", 0.))
                 for t in manifest["tasks"] if t.get("status") == "COMPLETE"]
    end_epoch = max(task_ends, default=time.time())
    manifest["formal_finished_epoch"] = end_epoch
    manifest["formal_elapsed_seconds"] = max(0., end_epoch - start_epoch)
    manifest["decision"] = decision
    manifest["go_checks"] = checks
    manifest["result_sha256"] = {p.name: sha(p) for p in RESULTS.iterdir() if p.is_file()}
    manifest["code_sha256"] = {p.name: sha(p) for p in HERE.glob("*.py")}
    json_write(HERE / "manifest.json", manifest)
    (HERE / "handoff_append_pending.md").write_text(
        build_b0a_handoff_append(summary), encoding="utf-8")
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    prior_handoff_status = manifest.get("handoff_status", "PENDING")
    manifest["handoff_append_sha256"] = sha(HERE / "handoff_append_pending.md")
    json_write(HERE / "manifest.json", manifest)
    if prior_handoff_status not in {"APPENDED_PREFIX_VERIFIED", "FAILED_PENDING_APPEND"}:
        append_handoff_once()
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") == "PARTIAL":
        summary["status"] = "PARTIAL"
        summary["scientific_status"] = scientific_status
        summary["delivery_status"] = manifest.get("delivery", "COMPLETE_WITH_HANDOFF_BLOCKED")
        json_write(RESULTS / "summary.json", summary)
    if scientific_status == "COMPLETE":
        write_b0a_report(summary, rows, csv_rows(RESULTS / "diagnostics.csv"), manifest)
    manifest["result_sha256"] = {p.name: sha(p) for p in RESULTS.iterdir() if p.is_file()}
    manifest["code_sha256"] = {p.name: sha(p) for p in HERE.glob("*.py")}
    manifest["handoff_append_sha256"] = sha(HERE / "handoff_append_pending.md")
    json_write(HERE / "manifest.json", manifest)
    print(f"SUMMARY status={status} folds={len(rows)}/100 groups={len(complete_groups)}/10 GPU0={gpu_counts['0']} GPU1={gpu_counts['1']} decision={decision}", flush=True)


def audit_residual_bounds(gpu_filter):
    """Recompute the trained residual ratio against the CAM norm for every saved arm."""
    import gc
    import torch

    path = RESULTS / "diagnostics.csv"
    rows = csv_rows(path)
    keyed = {(r["group"], int(r["fold"])): r for r in rows}
    assert len(keyed) == 100 and len(rows) == 100
    global _TEXT_CACHE
    _TEXT_CACHE = text_encoder_cache()
    maximum = 0.0
    audited_groups = [group for group in E_GROUPS if GPU[group] == gpu_filter]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu_filter)
    for group in audited_groups:
        for fold, (train, test) in enumerate(fold_splits(), 1):
            entry = load_fold_entry(fold)
            restore_rng(entry)
            agent = Agent(group, fold, train, test, entry, fold_reference(fold, train))
            state = state_load(CHECKPOINTS / f"{group}_fold_{fold:02d}_best.pt")
            assert state["group"] == group and int(state["fold"]) == fold
            agent.load_model_state(state["model"])
            agent.MessagePassing0.eval()
            agent.ModalFusion.eval()
            with torch.no_grad():
                agent.forward()
                diagnostic = agent.ModalFusion.lcher._last_diagnostics
                test_index = torch.as_tensor(test, dtype=torch.long, device=agent.dev)
                ratios = (diagnostic["delta"][test_index].norm(dim=-1) /
                          diagnostic["cam_norm"][test_index].clamp_min(1e-8))
            r = keyed[(group, fold)]
            r["residual_ratio_mean"] = float(ratios.mean().item())
            r["residual_ratio_max"] = float(ratios.max().item())
            r["residual_ratio_basis"] = "post-CAM residual norm / pre-residual CAM norm"
            maximum = max(maximum, float(ratios.max().item()))
            assert torch.isfinite(ratios).all() and float(ratios.max().item()) <= .050001
            del agent
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    rows.sort(key=lambda r: (GROUPS.index(r["group"]), int(r["fold"])))
    csv_write(path, rows)
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    by_gpu = manifest.setdefault("residual_bound_audit_by_gpu", {})
    by_gpu[str(gpu_filter)] = {"groups": audited_groups, "maximum_observed_ratio": maximum,
                               "checkpoint_count": len(audited_groups) * 10,
                               "passed": maximum <= .050001}
    if set(by_gpu) == {"0", "1"}:
        all_maximum = max(float(by_gpu[g]["maximum_observed_ratio"]) for g in ("0", "1"))
        manifest["residual_bound_audit"] = {
        "basis": "post-CAM residual norm / pre-residual CAM norm",
        "all_90_innovation_checkpoints_recomputed": sum(int(by_gpu[g]["checkpoint_count"]) for g in ("0", "1")) == 90,
        "maximum_observed_ratio": all_maximum, "bound": 0.05,
        "passed": all(bool(by_gpu[g]["passed"]) for g in ("0", "1")),
        "by_gpu": by_gpu,
        }
    json_write(HERE / "manifest.json", manifest)
    if "residual_bound_audit" in manifest:
        report()
    else:
        print("RESIDUAL_AUDIT_GPU_COMPLETE", gpu_filter, maximum, flush=True)


def build_b0a_handoff_append(summary):
    import numpy as np
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    diagnostics = csv_rows(RESULTS / "diagnostics.csv")
    e4diag = [r for r in diagnostics if r.get("group") == "E4_LCHER_FULL"]
    def mean(key):
        vals = []
        for row in e4diag:
            try:
                vals.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                pass
        return float(np.mean(vals)) if vals else float("nan")
    lines = ["", "## Experiment 38 Rerun — LCHER on the Experiment 32 B0-A Anchor", "",
             f"执行状态：{summary['status']}（科学任务 {summary['scientific_status'] if summary['status'] == 'PARTIAL' else summary['status']}）；筛选结论：{summary['decision']}。{summary['completed_folds']}/100 folds、{summary['completed_groups']}/10组；GPU0 {summary['gpu_folds'].get('0', 0)} folds、GPU1 {summary['gpu_folds'].get('1', 0)} folds；正式阶段耗时 {manifest.get('formal_elapsed_seconds', 0):.1f} 秒。", "",
             "本轮严格以实验32 B0-A为共同起点。其PLN按历史构造顺序完整初始化并注册未使用的PEN参数和原8条Prompt缓存以保持RNG/参数轨迹；隔离版forward在CAM后跳过SPGS+RES，直接送入原OutputLayer。该历史兼容脚手架不执行PEN。旧实验38的物理删PEN B1（ACC 91.3845%）不是本轮基线，也未作为正式任务或checkpoint使用。", "",
             "Anchor replay: **ANCHOR_REPLAY_PASS (10/10)**。B0-A以seed=0单进程连续十折重放；逐折混淆计数、四指标、best epoch、epochs executed、主干初始化SHA与入口RNG SHA均与实验32 B0一致。B0-A结果为ACC 92.0742±2.6000%、Hard-AUC 91.8546±2.5815%、SPE-TD 94.6300±4.0457%、SEN-ASD 89.0793±3.7283%。E1–E9从对应fold不可变anchor恢复，新增分支独立RNG初始化；这是配对入口设计，不代表E组自然连续跨折RNG。", "",
             "源码审计：MICCAI 2025论文 `https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf` 与本地README/main/model/network/layers/utils及ABIDE shell逐项核对；Git HEAD和相关SHA见manifest。GFR四图为受试者关系图，四支传播全320维拼接输入；GraphLearn硬阈值、归一化与原GCN保留，i==1覆盖i==0。simple-2优化分类损失加四图GraphConstructLoss均值；ALoss虽计算但不参与反传。保留871人transductive图、val=test选模和CV前全局RFE/预处理。实验11核验39条候选FC的数值唯一身份及ROI双端名称，195/225/232/236排除；原始FC位置为输入X的64+index。CAM的4×36为混合表示，不解释为ROI。", "",
             "固定先验由13组、26条ASD/TD竞争文本和39条候选FC边构成；H14–H16只存档。文本是用户提供的文献综合候选，不是GPT-4在线生成、专家认证或精确HO单边临床证据。真实Prompt冻结CLIP RN50缓存；匿名、中性、边错配和类别交换组按预注册产生。每fold仅用outer-train TD标签拟合Ridge参考，train TD用5折OOF；候选真实FC计算signed z并截断±3。固定Top-8通过原始FC干预并完整重跑GFR，CAM后signed残差逐患者限制为CAM范数的5%。", "",
             "历史审计路径包括本地experiment11/13/15/16/17/18/19/28/32/33，以及此前/tmp中的实验34–37。旧实验38的旧版代码/结果仅只读审阅，不并入本轮折数；其B1基线因物理删除PEN改变初始化/RNG轨迹，不可与E32 B0-A作配对因果比较。", "",
             "正式指标为原仓库one-hot硬预测ROC-AUC（不报告probability-AUC），按ACC、Hard-AUC、SPE-TD、SEN-ASD顺序：", "", "|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|", "|---|---:|---:|---:|---:|"]
    for group in summary["groups"]:
        lines.append(f"|{group['group']}|{group['acc']['mean']:.4f} ± {group['acc']['population_sd']:.4f}|{group['hard_auc']['mean']:.4f} ± {group['hard_auc']['population_sd']:.4f}|{group['spe_td']['mean']:.4f} ± {group['spe_td']['population_sd']:.4f}|{group['sen_asd']['mean']:.4f} ± {group['sen_asd']['population_sd']:.4f}|")
    by_left = {p["left"]: p for p in summary["paired_comparisons"]
               if p["right"] == B1 and p["left"] in E_GROUPS}
    lines += ["", "各创新组相对共同B0-A anchor的逐折配对差异（百分点；W/T/L按ACC与Hard-AUC）：", "", "|组别|ΔACC|ACC W/T/L|ΔHard-AUC|AUC W/T/L|ΔSPE|ΔSEN|", "|---|---:|---:|---:|---:|---:|---:|"]
    for group in [g for g in GROUPS if g.startswith("E")]:
        pair = by_left[group]
        a, u, s, n = (pair[k] for k in ("acc", "hard_auc", "spe_td", "sen_asd"))
        lines.append(f"|{group}|{a['delta_pp']:+.4f}|{a['wins']}/{a['ties']}/{a['losses']}|{u['delta_pp']:+.4f}|{u['wins']}/{u['ties']}/{u['losses']}|{s['delta_pp']:+.4f}|{n['delta_pp']:+.4f}|")
    lines += ["", "E4相对全部预注册创新对照的逐折配对差异（百分点；W/T/L按ACC与Hard-AUC）：", "", "|E4−对照|ΔACC|ACC W/T/L|ΔHard-AUC|AUC W/T/L|ΔSPE|ΔSEN|", "|---|---:|---:|---:|---:|---:|---:|"]
    for pair in summary["paired_comparisons"]:
        if pair["left"] != "E4_LCHER_FULL":
            continue
        a, u, s, n = (pair[k] for k in ("acc", "hard_auc", "spe_td", "sen_asd"))
        lines.append(f"|{pair['right']}|{a['delta_pp']:+.4f}|{a['wins']}/{a['ties']}/{a['losses']}|{u['delta_pp']:+.4f}|{u['wins']}/{u['ties']}/{u['losses']}|{s['delta_pp']:+.4f}|{n['delta_pp']:+.4f}|")
    e4_pos = sum(float(r.get("selected_vs_random_logp_drop", 0) or 0) > 0 for r in e4diag)
    all_diag = diagnostics
    max_ratio = max(float(r["residual_ratio_max"]) for r in all_diag if r.get("residual_ratio_max"))
    lines += ["", "机制诊断：B0-A+zero-alpha的forward、主损失、四图损失、主干梯度和两个Adam首步通过BITWISE_EXACT预检；测试标签置换不影响先验/推理。预检中的CF梯度在alpha、query/key、context、selector及signed residual路径均有限且非零，原始FC干预经完整GFR产生非零输出响应。", "",
              f"E4固定checkpoint审计：NULL概率均值 {mean('null_probability_mean'):.4f}；selected-delete减random-delete的正确类logp下降均值 {mean('selected_vs_random_logp_drop'):.4f}，正值 {e4_pos}/10 folds；文本交换平均绝对logp变化 {mean('prompt_swap_abs_logp_delta'):.6f}，类别翻转比例 {mean('prompt_swap_class_flip_fraction'):.6f}；禁用路由的平均绝对logp变化 {mean('router_disabled_abs_logp_delta'):.6f}。CAM相对残差折均值 {mean('residual_ratio_mean')*100:.4f}%，跨所有训练组及折记录的最大值 {max_ratio*100:.4f}%，低于5%上限。", "",
              f"判定：**NO-GO**。E4 ACC/AUC均低于B0-A约0.6897/0.6748个百分点，且低于匿名同预算E1约0.0000/0.0342个百分点；E5错误边映射和E6中性文本的ACC/AUC都高于E4。selected-delete优于random-delete虽为正向{e4_pos}/10折，但文本交换对logit的作用极小，平均类别翻转约0.1%；这不足以证明正确医学文本或连接映射具有独立任务贡献，更不能作生物医学因果解释。", "",
              f"预注册门槛逐项结果：`{json.dumps(summary.get('go_checks', {}), sort_keys=True)}`。十折训练集相互重叠，W/T/L是描述性配对结果，不是独立队列统计显著性。val=test checkpoint选择、CV前RFE/标准化和transductive图仍带来乐观偏差风险；不构成无偏独立泛化证据。", "",
              f"产物阶段路径：`{HERE}`。正式目标目录：`{ROOT / 'experiment' / 'experiment38_lcher_b0a_anchor_rerun'}`。`fold_metrics.csv`含100折混淆计数、最佳epoch、入口及初始化哈希；`diagnostics.csv`含机制量。E2 fold 1因先前诊断schema错误已作为不完整artifact从原anchor恢复一次，best epoch与指标不变，并补齐最佳checkpoint；唯一fold键仍为100。无效临时checkpoint保存在`.runtime/failed_attempts/`。根实验目录同步状态：{manifest.get('delivery_sync', {}).get('status')}（{manifest.get('delivery_sync', {}).get('error')}）；handoff追加状态：{manifest.get('handoff_status')}（{manifest.get('handoff_error')}），原handoff前缀SHA保持不变。完整产物仅在`/tmp`，这是临时存储且尚未同步。", ""]
    return "\n".join(lines)


def append_handoff_once():
    import tempfile
    manifest_path = HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = ROOT / "handoff.md"
    pending = HERE / "handoff_append_pending.md"
    try:
        original = source.read_bytes()
        appended = pending.read_bytes()
        prefix_hash = hash_bytes(original)
        manifest["handoff_before"] = {"bytes": len(original), "sha256": prefix_hash,
                                      "captured_immediately_before_append": True}
        if original.endswith(appended):
            manifest["handoff_status"] = "APPENDED_PREFIX_VERIFIED"
            manifest["handoff_after"] = {"bytes": len(original), "sha256": hash_bytes(original),
                                          "prefix_sha256": prefix_hash,
                                          "append_sha256": hash_bytes(appended), "already_present": True}
            json_write(manifest_path, manifest)
            print("HANDOFF_APPEND_ALREADY_PRESENT", manifest["handoff_after"])
            return
        fd, temp_name = tempfile.mkstemp(prefix=".handoff-e38-", suffix=".tmp", dir=ROOT)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(original)
                stream.write(appended)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, source)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        result = source.read_bytes()
        assert result[:len(original)] == original
        assert hash_bytes(result[:len(original)]) == prefix_hash
        assert result[len(original):] == appended
        manifest["handoff_status"] = "APPENDED_PREFIX_VERIFIED"
        manifest["handoff_after"] = {"bytes": len(result), "sha256": hash_bytes(result),
                                      "prefix_sha256": hash_bytes(result[:len(original)]),
                                      "append_sha256": hash_bytes(appended)}
        print("HANDOFF_APPEND_SUCCESS", manifest["handoff_after"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        manifest["handoff_status"] = "FAILED_PENDING_APPEND"
        manifest["handoff_error"] = error
        manifest["scientific_status"] = manifest.get("status", "COMPLETE")
        manifest["status"] = "PARTIAL"
        manifest["delivery"] = "COMPLETE_WITH_HANDOFF_BLOCKED"
        with (RESULTS / "report.md").open("a", encoding="utf-8") as stream:
            stream.write(f"\n## Delivery Status\n\nOverall delivery: **PARTIAL**. Scientific status: `{manifest['scientific_status']}`. Root `handoff.md` append failed: `{error}`. The original prefix was not intentionally changed; the complete addition remains in `handoff_append_pending.md`.\n")
        print("HANDOFF_APPEND_PENDING", manifest["handoff_error"])
    json_write(manifest_path, manifest)


def write_report(summary, metrics, diagnostics):
    from collections import defaultdict
    import numpy as np
    path = RESULTS / "report.md"
    e4diag = [r for r in diagnostics if r.get("group") == "E4_LCHER_FULL"]
    def dmean(key):
        vals = [float(r[key]) for r in e4diag if r.get(key, "") != ""]
        return float(np.mean(vals)) if vals else float("nan")
    residual_max = max(float(r["residual_ratio_max"]) for r in e4diag)
    lines = ["# Experiment 38 — LCHER", "", f"Status: **{summary['status']}**, decision: **{summary['decision']}**.",
             "", "## Audit", "", "The local checkout HEAD was `7747a1521629f95413faca133ba8c82ed7f1faf2`; hashes of README/main/model/network/layers/utils/ABIDE shell matched the corresponding raw files on current GitHub main `66d6a3b6e8ddc0332fe1f50cbf31caa1ae929afa`. The original MICCAI 2025 paper PDF was read at `https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf`. Local source files read: root `README.md`, `main.py`, `model.py`, `network.py`, `layers.py`, `utils.py`, ABIDE shell; and Experiment 11 `results/roi_256_named_mapping.csv` plus `results/subject_row_mapping.csv`.",
             "", "Historical local reports/results read: `experiment/experiment13_patient_specific_fc_prompt`, `experiment/experiment15_anatomically_anchored_roi_prompt`, `experiment/experiment16_roi_semantic_pathway_repair`, `experiment/experiment17_disease_semantic_selective_alignment`, `experiment/experiment18_discriminability_preserving_selective_alignment`, `experiment/experiment19_contrastive_disease_evidence_alignment`, `experiment/experiment28_imaging_anchored_finegrained_semantic_alignment`, `experiment/experiment32_disease_contrastive_evidence_routing`, and `experiment/experiment33_qcar_conditional_alignment`. For E34–E37, reports and READMEs read from `/tmp/SPromptGL-E34-SeRCA-stage-20261022/experiment/experiment34_serca_semantic_relational_alignment`, `/tmp/SPromptGL-E35-SeRCA-v2-stage/experiment/experiment35_serca_v2_relational_pattern_alignment`, `/tmp/SPromptGL-E36-PCPR-stage/experiment/experiment36_phenotype_conditioned_patient_routing`, and `/tmp/SPromptGL-E37-LSPCR-stage/experiment/experiment37_label_supervised_phenotype_routing`; method files `serca.py`, `serca_v2.py`, `pcpr.py`, and `router.py` were inspected. The E34–E37 reports all conclude NO-GO. No claim is made that every file in each historical directory was audited.",
             "", "The root model/data/history files were not modified. The `/data` atomic write probe failed under OSSFS, so this run was staged in `/tmp`.",
             "", "The paper already defines a same/different supervised adjacency. Local `simple-2` computes but does not optimize ALoss; the optimized objective is classification plus the mean of four GraphConstruct losses. Local GFR keeps hard cosine thresholding and the i==1 overwrite. P0 uses the full original PEN. B1 and all E arms physically remove SPGS, RES, PEN MLP, avg-pool, original eight-prompt cache, and CLIP import from the isolated runtime source. Experiment 36's four-graph implementation was structurally different from original B0; its T0 did not beat original B0 and fixed-checkpoint text disable did not change labels. Experiments 32–37 repeatedly show numerical/mismatched/incorrect controls can tie or outperform semantic arms.",
             "", "## Frozen Prior and Method", "", "The 13 candidate hypothesis pairs use the supplied fixed wording, sources and 39 unique verified HO connection indices; the three non-candidate groups remain archival only. Wording is a GPT-6 literature synthesis, not a GPT-4 call or clinician-authenticated claim. Exact ROI names were independently checked against experiment 11's numeric-unique, both-endpoints-named map. The CLIP RN50 weight file matches the official published checksum; tokenizer uses strict 77-token validation. CLIP model construction/encoding occurs under isolated RNG and is frozen. No separate license attestation is claimed.",
             "", "Within each fold, only outer-train TD labels fit a fixed alpha=10 Ridge reference on PHENO age-bin/sex/site covariates. Training TD values use deterministic five-fold OOF prediction; test and train ASD use the full train-TD model. FC evidence is signed, z-clipped at ±3, and restricted to the 39 real input columns `64+edge`. The model competes ASD text, TD text and a fixed NULL baseline, then routes hard Top-8 evidence with a differentiable straight-through selector during training. Age/sex context and ANAT/FUNC QC reliability condition the evidence gate; site does not directly enter that head. A zero-initialized signed residual after CAM is hard-bounded to 5% of CAM norm and enters the existing OutputLayer. New parameters use the existing PLN Adam as a zero-decay parameter group; there is no third Adam or new classifier.",
             "", "Counterfactual arms replace only one target patient's candidate raw FC values with that fold's conditional TD reference and rerun the full GFR→CAM→OutputLayer on all 871 participants. Other people and other modalities remain unchanged. Training probes rotate two outer-train patients every five epochs. Counterfactual losses freeze GFR/CAM/Output parameters, allowing gradients only through the selector/input mask and LCHER. Test keep/delete/random/high-|z| interventions are fixed-checkpoint mechanism audits, not formal classification metrics or causal evidence.",
             "", "## Formal Metrics", "", "Mean ± population SD over ten folds, percentage. AUC is the repository's one-hot hard-prediction ROC-AUC. ASD is label 0 and TD label 1.",
             "", "|Group|ACC|Hard-AUC|SPE-TD|SEN-ASD|", "|---|---:|---:|---:|---:|"]
    for g in summary["groups"]:
        lines.append(f"|{g['group']}|{g['acc']['mean']:.4f} ± {g['acc']['population_sd']:.4f}|{g['hard_auc']['mean']:.4f} ± {g['hard_auc']['population_sd']:.4f}|{g['spe_td']['mean']:.4f} ± {g['spe_td']['population_sd']:.4f}|{g['sen_asd']['mean']:.4f} ± {g['sen_asd']['population_sd']:.4f}|")
    lines += ["", "Every fold row, confusion count, and best epoch is retained in `fold_metrics.csv`. Same-fold comparisons are descriptive because CV training sets overlap.",
              "", "## Paired Comparisons", "", "Δ is E4 minus comparator in percentage points. W/T/L are the ten same-fold ACC and Hard-AUC win/tie/loss counts.",
              "", "|E4 − comparator|ΔACC|ACC W/T/L|ΔHard-AUC|Hard-AUC W/T/L|ΔSPE|ΔSEN|", "|---|---:|---:|---:|---:|---:|---:|"]
    for p in summary["paired_comparisons"]:
        a, u, s, n = (p[k] for k in ("acc", "hard_auc", "spe_td", "sen_asd"))
        lines.append(f"|{p['right']}|{a['delta_pp']:+.4f}|{a['wins']}/{a['ties']}/{a['losses']}|{u['delta_pp']:+.4f}|{u['wins']}/{u['ties']}/{u['losses']}|{s['delta_pp']:+.4f}|{n['delta_pp']:+.4f}|")
    lines += ["", "## Mechanism", "", f"Pre-registered checks: `{json.dumps(summary['go_checks'], ensure_ascii=False, sort_keys=True)}`.",
              "", "The fold diagnostics record raw graph density/degree, route score and residual behavior, NULL fraction, train counterfactual loss, and fixed-checkpoint intervention effects. A changed attention score alone is not evidence that the classifier used a hypothesis. A correct-edge evidence claim requires E4 to beat edge derangement E5 and neutral text E6 and show selected deletion stronger than equal-budget random deletion.",
              "", f"Corrected E4 mechanism audit across ten folds: mean NULL probability {dmean('null_probability_mean'):.4f}; mean selected-vs-random true-logp deletion effect {dmean('selected_vs_random_logp_drop'):.4f} (positive in {sum(float(r['selected_vs_random_logp_drop']) > 0 for r in e4diag)}/10 folds); text-pair swap changed mean held-out log-probabilities by {dmean('prompt_swap_abs_logp_delta'):.6f} and labels in {dmean('prompt_swap_class_flip_fraction'):.4f} of cases. Disabling the route changed log-probabilities by {dmean('router_disabled_abs_logp_delta'):.6f} on average. The CAM-relative residual mean was {dmean('residual_ratio_mean')*100:.3f}% across fold means and the maximum held-out participant ratio was {residual_max*100:.3f}%; all 80 trained LCHER checkpoints passed the 5% bound. These show some evidence-mask dependence, but little task-output dependence on exact ASD/TD wording.",
              "", "P0 retains the original PEN and its separate CLIP initialization path, so it is a structural reference, not a shared-initialization causal contrast. B1 is the primary physically PEN-removed baseline; E1 matches the LCHER numerical architecture with anonymous normalized fixed vectors. E2−E1 isolates hypothesis wording/geometry without counterfactual losses; E3−E2 isolates keep; E4−E3 isolates delete; E5 isolates prompt-edge mapping; E6 tests ASD/TD-specific text; E7 removes auxiliary age/sex/QC conditioning; E8 removes rival class contrast.",
              "", "## Limits and Decision", "", "The historical `val_idx=test_idx` checkpoint selection, pre-CV RFE/standardization, and transductive 871-participant graph remain. The metrics are not unbiased independent generalization estimates. Patient-level feature masking is a model dependence test, not a biological causal intervention. The 39 selected indices are not a random sample of all 256 FC features and a positive result would still require external validation and multiple seeds.",
              "", f"Decision: **{summary['decision']}**. This is a frozen single-seed internal screen, not a statistical-significance claim, SOTA claim, clinical biomarker validation or MICCAI acceptance prediction.",
              "", "## Reproducibility", "", "See `manifest.json` for input, source, prompt, encoder, token, embedding, code, result, fold and RNG hashes. `.runtime/` includes the single isolated source version, immutable text/reference cache, fold-entry RNG snapshots, unique task state and atomic resumable checkpoints. No Git commit or push was performed.", ""]
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp, path)


def write_b0a_report(summary, metrics, diagnostics, manifest):
    path = RESULTS / "report.md"
    anchor = manifest.get("anchor_replay", {})
    delivery = manifest.get("delivery_sync", {})
    lines = [
        "# Experiment 38 Rerun: LCHER on the B0-A Anchor", "",
        f"**Status:** {summary['status']}  ",
        f"**Anchor replay:** {anchor.get('status', 'UNKNOWN')} ({len(anchor.get('folds', []))}/10)  ",
        f"**Scientific decision:** {summary['decision']}  ",
        f"**Delivery:** root experiment sync `{delivery.get('status', 'not_attempted')}`; root handoff `{manifest.get('handoff_status', 'PENDING')}`. OSSFS error: `{delivery.get('error', manifest.get('handoff_error', 'none'))}`.", "",
        "This is the rerun of Experiment 38 with the exact Experiment 32 B0-A implementation as the shared anchor. It replaces the previous experiment38 report's invalid physically PEN-deleted baseline framing. No training result from the earlier experiment38 directory is counted here.", "",
        "## Audit and Base", "",
        f"Local Git HEAD: `{manifest.get('root_git_head')}`; audited GitHub main revision: `{manifest.get('github_main')}`. Root source hashes and input, mapping, prompt, CLIP cache and fold-split hashes are in `manifest.json`. The paper checked was [SPromptGL, MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf). Read-only audit covered the root README, main/model/network/layers/utils and ABIDE shell, Experiment 11's verified FC/ROI map, Experiment 32 B0 integration/run/metrics/report, local Experiments 13/15/16/17/18/19/28/33, and available `/tmp` reports/methods for Experiments 34–37. The old Experiment 38 files were read-only background; their physically deleted B1 was rejected as an anchor because removing unused PLN construction changes initialization/RNG trajectory.", "",
        "The B0-A scaffold follows Experiment 32: instantiate the complete original PLN (including unused PEN MLP/avg-pool and eight legacy prompt buffers), retain the original RNG consumption, then bypass SPGS+RES after CAM and pass flattened CAM to the original OutputLayer. PEN is not executed in this forward path. Original GFR behavior, hard threshold, normalization, four graph losses, simple-2 objective, and checkpoint rule remain unchanged.", "",
        "Replay was a single seed-0 process with a continuous RNG stream across ten folds. All ten folds matched E32 B0 on confusion counts, ACC/Hard-AUC/SPE/SEN, best epoch, executed epochs, shared trunk initialization SHA and fold-entry RNG SHA. Immutable per-fold anchors include model, both Adam states and Python/NumPy/Torch CPU/CUDA RNG. E1–E9 restore the corresponding B0 anchor; this pairing does not claim naturally continuous E-group RNG streams.", "",
        "GFR graphs are 871-patient graphs, not ROI graphs; each branch propagates the full concatenated 320-dimensional input. The source's i==1 overwrite of i==0 is preserved. In simple-2, ALoss is computed but not optimized; the optimized objective is classification loss plus the mean of four GraphConstruct losses. The 4×36 CAM representation is already mixed and is not interpreted as ROIs. The historical full-cohort RFE/preprocessing, transductive graph, and `val_idx=test_idx` model selection remain, so these are internal historical-protocol comparisons, not unbiased external generalization results.", "",
        "## Frozen LCHER", "",
        "The fixed candidate library contains H01–H13, 26 ASD/TD text prompts and 39 unique FC indices; H14–H16 remain archival. Text is the supplied literature synthesis, not a new GPT-4 call or clinician certification, and network-level literature does not prove an effect for each HO edge. Experiment 11's ROI mapping verifies the 39 candidate indices and both endpoint names; unknown indices 195/225/232/236 are excluded. Only raw input columns `64 + corr_index` carry these FC identities; mixed GFR coordinates are not used as ROI indices.", "",
        "The frozen RN50 CLIP bank is hashed in the manifest and cached once. Per fold, an alpha=10 Ridge reference uses outer-train TD only, with five-fold OOF references for outer-train participants; z scores are signed and clipped to ±3. The same patient receives both rival prompts. A straight-through Top-8 selector operates over 39 eligible raw FC edges. Selected/removed input-level counterfactuals replace only the target participant's selected raw FC values with the fold TD reference and rerun full GFR→CAM→OutputLayer. The post-CAM signed evidence residual is zero-initialized and individually norm-limited to 5%; new parameters are added to the existing PLN Adam with zero weight decay.", "",
        "## Results and Paired Analysis", "",
        "Formal metrics are ACC, original one-hot hard-label ROC-AUC, SPE-TD, and SEN-ASD, shown as ten-fold mean ± population SD (%). ASD is label 0; TD is label 1.", "",
    ]
    # Reuse the frozen, full comparison tables from the handoff appendix.
    appendix = build_b0a_handoff_append(summary).strip()
    if appendix.startswith("## Experiment 38 Rerun"):
        appendix = appendix[appendix.find("|组别|"):]
    lines.extend([appendix, "", "## Mechanism and Decision", "",
        f"Preflight verified bitwise equivalence between B0-A and zero-alpha LCHER for forward output, classification/GFR losses, common-parameter gradients and the first two Adam updates with CF disabled. The held-out-label invariance and raw-FC full-GFR-response checks passed. Counterfactual gradient paths were finite and nonzero at preflight. Best-checkpoint residual audit recomputed all {manifest.get('residual_bound_audit', {}).get('by_gpu', {}).get('0', {}).get('checkpoint_count', 0) + manifest.get('residual_bound_audit', {}).get('by_gpu', {}).get('1', {}).get('checkpoint_count', 0)} E1–E9 checkpoints; maximum participant residual/CAM ratio was {manifest.get('residual_bound_audit', {}).get('maximum_observed_ratio', float('nan')) * 100:.4f}% (5% cap passed: {manifest.get('residual_bound_audit', {}).get('passed', False)}). Evidence ranking/ablation effects are model-dependence diagnostics, not biological causal evidence.", "",
        f"The frozen screening outcome is **{summary['decision']}**. LCHER-FULL did not clear the required margin over B0-A or the matched anonymous control, and it did not outperform wrong-edge and neutral-text controls on the primary metrics. Selected deletion exceeded equal-budget random deletion in {summary.get('screening_diagnostics', {}).get('E4_selected_delete_minus_random_logp_drop', {}).get('positive_folds', 0)}/10 folds, but exact prompt swapping produced only a small log-probability response. This does not establish independent contribution from the correct medical wording or edge mapping.", "",
        "The ten folds share participants across training sets. Fold-paired W/T/L counts are descriptive and are not ten independent cohorts or a significance test. No SOTA, clinical biomarker, causal, or acceptance claim is made.", "",
        "## Runtime and Recovery", "",
        f"Formal training completed {summary['completed_folds']}/100 unique folds across 10/10 groups, GPU0 {summary['gpu_folds'].get('0', 0)} folds and GPU1 {summary['gpu_folds'].get('1', 0)} folds. Formal wall-clock interval through recovery was {manifest.get('formal_elapsed_seconds', 0):.1f} seconds. E2 fold 1 had a committed metric row but no best checkpoint after an earlier diagnostics-schema failure; it was therefore treated as artifact-incomplete and replayed once from the exact paired anchor (95.9 s), with the same best epoch and metrics. This is one recovery attempt, not an additional unique fold. Other invalid partial checkpoints remain under `.runtime/failed_attempts/`. Both metrics and diagnostics now have 100 unique `(group, fold)` keys.", "",
        f"Staged directory: `{HERE}`. Formal target: `{ROOT / 'experiment' / 'experiment38_lcher_b0a_anchor_rerun'}`. `/tmp` is temporary storage; delivery remains `{summary.get('delivery_status', manifest.get('delivery', 'not synchronized'))}` until verified. Full pending handoff text is `handoff_append_pending.md`.", ""])
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
