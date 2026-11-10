#!/usr/bin/env python3
"""Frozen-backbone prompt utility audit for Experiment 75.4."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

import probe

EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
PROMPTS = json.loads((EXP / "prompt_banks.json").read_text(encoding="utf-8"))
BASE = Path(CFG["base"]["path"])
E60 = Path(CFG["references"]["e60"])
E751 = Path(CFG["references"]["e751"])
E752 = Path(CFG["references"]["e752"])
RT = EXP / ".runtime"
FEATURES = RT / "frozen_features"
RESULTS = EXP / "results"
GROUP = CFG["base"]["name"]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_object(value) -> str:
    return sha_bytes(pickle.dumps(value, protocol=4))


def tensor_hash(*modules) -> str:
    h = hashlib.sha256()
    for module in modules:
        for name, value in sorted(module.state_dict().items()):
            h.update(name.encode("utf-8"))
            arr = value.detach().cpu().contiguous().numpy()
            h.update(str(arr.dtype).encode("ascii"))
            h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
            h.update(arr.tobytes())
    return h.hexdigest()


def rng_digest() -> str:
    pieces = [random.getstate(), np.random.get_state(), torch.get_rng_state().cpu().numpy().tobytes()]
    if torch.cuda.is_available():
        pieces.extend(v.cpu().numpy().tobytes() for v in torch.cuda.get_rng_state_all())
    return sha_object(pieces)


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def base_runner():
    return load_module("e744_frozen_runner_754", BASE / "run.py")


def protected_source_paths():
    paths = [
        BASE / x for x in ("config.json", "run.py", "soft_positive_dcr.py", "audit.py", "manifest.json",
                           "results/report.md", "results/fold_metrics.csv", "results/summary.csv",
                           "results/mechanism.csv", "results/interventions.csv")
    ]
    paths += [BASE / ".runtime/checkpoints" / f"E74.4-G2-Positive-DCR_f{f:02d}.best.pt" for f in range(1, 11)]
    paths += [E751 / x for x in ("sa_dcer.py", "run.py", "audit.py", "config.json", "manifest.json",
                                 "results/report.md", "results/summary.csv", "results/paired_comparisons.csv",
                                 "results/mechanism.csv", "results/interventions.csv")]
    paths += [E752 / x for x in ("gc_sper.py", "run.py", "audit.py", "config.json", "manifest.json",
                                 "results/report.md", "results/summary.csv", "results/paired_comparisons.csv",
                                 "results/mechanism.csv", "results/interventions.csv")]
    root = Path("/data/adodas/SPromptGL-main")
    paths += [root / x for x in ("main.py", "model.py", "network.py", "layers.py", "utils.py",
                                 "data/ABIDE/processed_standard_data.csv", "data/ABIDE/modal_feat_dict.npy")]
    paths += [E60 / x for x in ("config.json", "text_features.py", ".runtime/text_embeddings.pt")]
    paths.append(Path(CFG["text_encoder"]["weights_path"]))
    return paths


def source_manifest():
    result = {}
    for path in protected_source_paths():
        if not path.is_file():
            raise FileNotFoundError(f"required locked source is absent: {path}")
        result[str(path.resolve())] = sha_file(path)
    return result


def verify_references(r):
    base_manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    if (base_manifest.get("status"), base_manifest.get("completed_tasks"), base_manifest.get("independent_audits")) != ("COMPLETE_AUDITED", 30, 30):
        raise RuntimeError("E74.4-G2 is not a complete 30/30 independently audited base")
    if base_manifest.get("source_sha256") != CFG["base"]["manifest_sha256"]:
        raise RuntimeError("E74.4 manifest SHA identity does not match the locked reference")
    current_sources = r.source_manifest()
    if current_sources != base_manifest.get("protected_sources"):
        raise RuntimeError("E74.4 protected source file hashes differ from its independently audited manifest")
    e744_source_current = r.sha_object(current_sources)
    integrity = base_manifest.get("source_integrity", {})
    audit_path = str((BASE / "audit.py").resolve())
    preflight_audit_sha = integrity.get("audit_source_sha256_at_preflight")
    if not preflight_audit_sha or not integrity.get("all_non_audit_source_hashes_match_preflight"):
        raise RuntimeError("E74.4 audit-only source repair history is incomplete")
    training_sources = dict(current_sources)
    training_sources[audit_path] = preflight_audit_sha
    e744_source_preflight = r.sha_object(training_sources)
    if e744_source_preflight != base_manifest.get("training_source_sha256") or e744_source_preflight != base_manifest.get("source_sha256"):
        raise RuntimeError(f"E74.4 locked preflight digest mismatch: {e744_source_preflight}")
    if integrity.get("audit_script_changed_after_training") and "no training code" not in integrity.get("audit_repair_scope", ""):
        raise RuntimeError("E74.4 audit changed outside its recorded audit-only repair scope")
    for path, tasks, audits in ((E751, 30, 30), (E752, 30, 30)):
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("status"), manifest.get("completed_tasks"), manifest.get("independent_audits")) != ("COMPLETE_AUDITED", tasks, audits):
            raise RuntimeError(f"read-only prior experiment is not fully audited: {path}")
    root = Path("/data/adodas/SPromptGL-main")
    net_text = (root / "network.py").read_text(encoding="utf-8")
    original = PROMPTS["banks"]["P1-SPromptGL-8"]
    if not all(item["text"] in net_text for item in original):
        raise RuntimeError("P1 prompt content/order source does not match network.py")
    if "part_x = x.view(871, 4, 4, 9)" not in net_text or "AvgPool1d(kernel_size=4, stride=4)" not in net_text:
        raise RuntimeError("original PEN latent feature partition/pooling source was not verified")
    if GROUP not in tuple(r.GROUPS):
        raise RuntimeError("the locked E74.4-G2 group is absent from the formal base groups")
    return {"e744_source_digest_preflight": e744_source_preflight,
            "e744_source_digest_current": e744_source_current,
            "e744_audit_current_sha256": sha_file(BASE / "audit.py"),
            "e744_audit_preflight_sha256": preflight_audit_sha,
            "e744_audit_only_repairs_recorded": integrity.get("audit_repairs", []),
            "e744_manifest_status": base_manifest["status"],
            "e744_tasks": base_manifest["completed_tasks"], "e744_audits": base_manifest["independent_audits"],
            "e751_status": "COMPLETE_AUDITED", "e752_status": "COMPLETE_AUDITED",
            "spromptgl_prompt_source": "network.py exact string check PASS",
            "spromptgl_token_partition_source": "view(871,4,4,9) and AvgPool1d(4,4) verified"}


def capture_fold_features(r, fold):
    state, _, identity = r.prepare(fold)
    state["fold"] = fold
    state["group"] = GROUP
    checkpoint_path = r.best_path(GROUP, fold)
    checkpoint = r.torch_load(checkpoint_path)
    if checkpoint.get("group") != GROUP or checkpoint.get("fold") != fold:
        raise RuntimeError(f"checkpoint group/fold mismatch: {checkpoint_path}")
    r.load_history().load_bundle(state, checkpoint["best_state"])
    state["cam"].eval(); state["model"].eval()
    captures = {"cam_calls": 0, "output_calls": 0}

    def capture_f(_module, args):
        captures["f"] = args[0].detach().cpu().clone()

    def capture_h(_module, args):
        captures["h"] = args[0].detach().cpu().clone()

    def count_cam(_module, _args, _output):
        captures["cam_calls"] += 1

    def count_output(_module, _args, _output):
        captures["output_calls"] += 1

    cam = state["cam"]
    if not hasattr(cam, "Outputlayer"):
        raise RuntimeError("E74.4 CAM Outputlayer hook boundary is unavailable")
    hooks = [cam.register_forward_pre_hook(capture_f),
             cam.register_forward_hook(count_cam),
             cam.Outputlayer.register_forward_pre_hook(capture_h),
             cam.Outputlayer.register_forward_hook(count_output)]
    before_rng = rng_digest()
    before_state = tensor_hash(state["model"], state["cam"])
    with torch.no_grad():
        output = r.forward(state, GROUP, training=False)
    after_rng = rng_digest()
    after_state = tensor_hash(state["model"], state["cam"])
    for hook in hooks:
        hook.remove()
    if captures["cam_calls"] != 1 or captures["output_calls"] != 1:
        raise RuntimeError("the frozen E74.4 path did not execute one CAM and one OutputLayer")
    if before_rng != after_rng:
        raise RuntimeError("E74.4 frozen eval changed the inherited main RNG")
    if before_state != after_state:
        raise RuntimeError("E74.4 frozen eval modified model parameters")
    if "f" not in captures or "h" not in captures:
        raise RuntimeError("feature hooks failed to capture F320/H144")

    x = torch.cat(state["parts"], dim=1).detach().cpu().numpy().astype(np.float32)
    f = captures["f"].numpy().astype(np.float32)
    h = captures["h"].numpy().astype(np.float32)
    logp = output["logp"].detach().cpu().numpy().astype(np.float64)
    probability_all = output["logp"].detach().exp().cpu().numpy().astype(np.float64)
    labels = np.asarray(state["labels_np"], dtype=np.int64)
    if tuple(x.shape) != (871, 320) or tuple(f.shape) != (871, 320) or tuple(h.shape) != (871, 144):
        raise RuntimeError(f"unexpected frozen feature shapes: {x.shape}, {f.shape}, {h.shape}")
    if not np.array_equal(labels, state["y"].detach().cpu().numpy().astype(np.int64)):
        raise RuntimeError("state y and label array differ")
    if not np.allclose(f, output["fused"].detach().cpu().numpy(), atol=0, rtol=0):
        raise RuntimeError("CAM hook input differs from returned DARG feature")
    test_idx = np.asarray(state["test_np"], dtype=np.int64)
    train_idx = np.asarray(state["train_np"], dtype=np.int64)
    y = torch.as_tensor(state["y"])
    base = r.load_history().load_e69().load_e61().load_base()
    metric = base.classification_metrics(output["logp"], state["test"], y)
    pred = np.asarray(metric["pred"], dtype=np.int64)
    prob = np.asarray(metric["prob"], dtype=np.float64)
    hist_pred = np.asarray(checkpoint["test_prediction"], dtype=np.int64)
    hist_prob = np.asarray(checkpoint["test_probability"], dtype=np.float64)
    if not np.array_equal(pred, hist_pred):
        raise RuntimeError(f"E74.4 frozen hard predictions differ on fold {fold}")
    current_prob = probability_all[test_idx]
    if not np.allclose(current_prob, hist_prob, atol=1e-8, rtol=0):
        raise RuntimeError(f"E74.4 frozen probabilities differ on fold {fold}")
    for name in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1"):
        if abs(float(metric[name]) - float(checkpoint["metrics"][name])) > 1e-10:
            raise RuntimeError(f"E74.4 metric mismatch for {name} fold {fold}")
    overlap = np.intersect1d(train_idx, test_idx)
    expected_overlap = np.asarray([376, 406], dtype=np.int64) if fold == 2 else np.asarray([], dtype=np.int64)
    if not np.array_equal(np.sort(overlap), expected_overlap):
        raise RuntimeError(f"unexpected historical train/test overlap in fold {fold}: {overlap}")
    probe_train = np.setdiff1d(train_idx, test_idx, assume_unique=False)
    if np.intersect1d(probe_train, test_idx).size:
        raise RuntimeError("probe training rows overlap the outer test fold")
    meta = {
        "fold": fold, "group": GROUP, "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha_file(checkpoint_path), "identity": identity,
        "train_index_sha256": sha_bytes(train_idx.astype(np.int64).tobytes()),
        "probe_train_index_sha256": sha_bytes(probe_train.astype(np.int64).tobytes()),
        "test_index_sha256": sha_bytes(test_idx.astype(np.int64).tobytes()),
        "overlap_rows_1based": (overlap + 1).astype(int).tolist(),
        "cam_call_count": captures["cam_calls"], "outputlayer_call_count": captures["output_calls"],
        "rng_before_forward_sha256": before_rng, "rng_after_forward_sha256": after_rng,
        "model_state_sha256_before": before_state, "model_state_sha256_after": after_state,
        "hard_prediction_sha256": sha_bytes(pred.astype(np.int64).tobytes()),
        "probability_sha256": sha_bytes(current_prob.astype(np.float64).tobytes()),
        "metrics": {k: float(metric[k]) for k in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")},
        "best_epoch_1based": int(checkpoint["best_epoch_1based"]),
        "train_count_for_new_probes": int(len(probe_train)),
        "test_count": int(len(test_idx)),
    }
    return {"x": x, "f": f, "h": h, "tokens": h.reshape(871, 16, 9),
            "logp": logp, "probability_all": probability_all,
            "labels": labels, "train": train_idx, "probe_train": probe_train,
            "test": test_idx, "prob": current_prob, "pred": pred, "meta": meta,
            "checkpoint_metrics": checkpoint["metrics"]}


def save_fold_feature(fold, data):
    FEATURES.mkdir(parents=True, exist_ok=True)
    path = FEATURES / f"fold{fold:02d}.npz"
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        np.savez_compressed(f, x=data["x"], f=data["f"], h=data["h"], tokens=data["tokens"],
                            logp=data["logp"], probability_all=data["probability_all"],
                            labels=data["labels"], train=data["train"],
                            probe_train=data["probe_train"], test=data["test"])
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    atomic_json(FEATURES / f"fold{fold:02d}.json", data["meta"])


def extract_all():
    r = base_runner()
    refs = verify_references(r)
    source_before = source_manifest()
    rows, metas = [], {}
    t0 = time.time()
    for fold in range(1, 11):
        data = capture_fold_features(r, fold)
        save_fold_feature(fold, data)
        metrics = data["meta"]["metrics"]
        rows.append({"group": GROUP, "fold": fold, "best_epoch_1based": data["meta"]["best_epoch_1based"],
                     "checkpoint_sha256": data["meta"]["checkpoint_sha256"], **metrics})
        metas[str(fold)] = data["meta"]
        print(f"[extract] fold={fold}/10 best_epoch={data['meta']['best_epoch_1based']} ACC={metrics['acc']*100:.4f}%", flush=True)
    source_after = source_manifest()
    if source_before != source_after:
        raise RuntimeError("protected source SHA changed during frozen extraction")
    write_csv(RESULTS / "fold_metrics.csv", rows)
    status = {"status": "FROZEN_FEATURES_EXTRACTED", "base_identity": refs,
              "protected_sources": source_before, "folds": metas,
              "wall_seconds": time.time() - t0, "base_training": False}
    atomic_json(RESULTS / "feature_extraction_audit.json", status)
    return status


def prompt_texts_and_classes():
    banks = PROMPTS["banks"]
    result = {}
    for name, items in banks.items():
        texts = [item["text"] for item in items]
        tags = [item["class"] for item in items]
        normalized = ["ASD" if tag == "ASD" or tag == "PSEUDO_B" else "TD" for tag in tags]
        if name == "C3-Unrelated-English-8":
            normalized = ["TD"] * 4 + ["ASD"] * 4
        result[name] = (texts, tags, normalized)
    return result


def encode_prompt_banks():
    cache_path = Path(CFG["text_encoder"]["cache_path"])
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    repo_path = str(Path("/data/adodas/SPromptGL-main").resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    encoder_module = load_module("e60_text_features_locked_for_e754", E60 / "text_features.py")
    expected_cache = {
        "knowledge_sha256": sha_file(E60 / "knowledge_base.json"),
        "rn50_sha256": sha_file(Path(CFG["text_encoder"]["weights_path"])),
        "config_sha256": sha_file(E60 / "config.json"),
        "text_features_sha256": sha_file(E60 / "text_features.py"),
    }
    if cache.get("source_hashes") != expected_cache:
        raise RuntimeError("E60 cached CLIP text embeddings do not match their locked sources")
    if expected_cache["rn50_sha256"] != CFG["text_encoder"]["weights_sha256"]:
        raise RuntimeError("CLIP RN50 weight SHA mismatch")
    banks = prompt_texts_and_classes()
    cached_map = {}
    for key, texts in cache["text_map"].items():
        vectors = cache["embeddings"][key].detach().cpu().numpy()
        for text, vector in zip(texts, vectors):
            cached_map[text] = vector.astype(np.float32)
    all_texts = sorted({text for texts, _, _ in banks.values() for text in texts})
    missing = [text for text in all_texts if text not in cached_map]
    if missing:
        if not Path(CFG["text_encoder"]["weights_path"]).is_file():
            raise RuntimeError("verified RN50 weights are unavailable; no replacement encoder will be used")
        pre_rng = rng_digest()
        devices = [0] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(754001)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(754001)
            new_vectors = encoder_module.encode_texts(missing, device="cuda:0" if torch.cuda.is_available() else "cpu")
        if rng_digest() != pre_rng:
            raise RuntimeError("offline CLIP encoding changed the process RNG")
        for text, vector in zip(missing, new_vectors.numpy()):
            cached_map[text] = vector.astype(np.float32)
    embeddings, metadata = {}, {}
    for bank, (texts, tags, normalized) in banks.items():
        arr = np.stack([cached_map[text] for text in texts]).astype(np.float32)
        if arr.shape != (len(texts), 1024) or not np.isfinite(arr).all():
            raise RuntimeError(f"invalid CLIP vectors for {bank}")
        key = f"bank_{len(embeddings):02d}"
        embeddings[key] = arr
        metadata[bank] = {"array_key": key, "texts": texts, "source_classes": tags,
                          "probe_classes": normalized, "vector_sha256": sha_bytes(arr.tobytes()),
                          "prompt_count": len(texts), "vector_norms": np.linalg.norm(arr, axis=1).astype(float).tolist()}
    # Three preregistered numerical lower bounds, generated without global RNG.
    for seed in CFG["analysis"]["random_embedding_seeds"]:
        key = f"random_unit_{seed}"
        arr = probe.random_unit_text(seed, count=8, dim=1024)
        embeddings[key] = arr
        metadata[key] = {"array_key": key, "texts": [], "source_classes": ["TD"]*4+["ASD"]*4,
                         "probe_classes": ["TD"]*4+["ASD"]*4,
                         "vector_sha256": sha_bytes(arr.tobytes()), "prompt_count": 8,
                         "vector_norms": np.linalg.norm(arr, axis=1).astype(float).tolist(),
                         "generator_seed": int(seed), "kind": "random_unit_vector"}
    RT.mkdir(parents=True, exist_ok=True)
    npz_path = RT / "prompt_embeddings.npz"
    tmp = npz_path.with_name(npz_path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **embeddings); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, npz_path)
    meta = {"encoder": "OpenAI CLIP RN50", "weights_sha256": expected_cache["rn50_sha256"],
            "cache_sha256": sha_file(cache_path), "cache_sources": expected_cache,
            "tokenizer_normalization": cache.get("tokenizer_normalization"),
            "newly_encoded_texts": missing, "bank_metadata": metadata,
            "embedding_npz_sha256": sha_file(npz_path), "official_brainprompt_prompt_url": "https://raw.githubusercontent.com/AngusMonroe/BrainPrompt/main/data/prompts/label_prompts/abide_label.txt",
            "official_brainprompt_text_sha256": sha_bytes((banks["P2-BrainPrompt-2"][0][0] + "\n" + banks["P2-BrainPrompt-2"][0][1] + "\n").encode("utf-8"))}
    atomic_json(RT / "prompt_embedding_manifest.json", meta)
    geometry_rows=[]
    for name, details in metadata.items():
        vectors=embeddings[details["array_key"]].astype(np.float64)
        norms=np.linalg.norm(vectors,axis=1)
        unit=vectors/np.maximum(norms[:,None],1e-12)
        gram=unit@unit.T
        tags=np.asarray(details["probe_classes"])
        asd=np.flatnonzero(tags=="ASD")
        td=np.flatnonzero(tags=="TD")
        intra_asd=[gram[i,j] for ii,i in enumerate(asd) for j in asd[ii+1:]]
        intra_td=[gram[i,j] for ii,i in enumerate(td) for j in td[ii+1:]]
        inter=[gram[i,j] for i in asd for j in td]
        row={"prompt_group":name,"prompt_count":len(vectors),"mean_vector_norm":float(norms.mean()),
             "min_vector_norm":float(norms.min()),"max_vector_norm":float(norms.max()),
             "mean_intra_asd_cosine":float(np.mean(intra_asd)) if intra_asd else None,
             "mean_intra_td_cosine":float(np.mean(intra_td)) if intra_td else None,
             "mean_inter_class_cosine":float(np.mean(inter)) if inter else None,
             "semantic_centroid_margin":float(1-np.mean(inter)) if inter else None,
             "class_prompt_count_asd":len(asd),"class_prompt_count_td":len(td),
             "vector_sha256":details["vector_sha256"]}
        geometry_rows.append(row)
    write_csv(RESULTS/"prompt_semantic_geometry.csv",geometry_rows)
    return embeddings, metadata, meta


def load_features(fold):
    with np.load(FEATURES / f"fold{fold:02d}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def metric_csv_row(fold, group, setting, method, y, p, train_n, test_n, selected_c=None, metadata=None):
    m = probe.binary_metrics(y, p)
    row = {"fold": fold, "prompt_group": group, "setting": setting, "method": method,
           "train_n": train_n, "test_n": test_n, "selected_C": selected_c,
           "acc_pct": 100*m["acc"], "hard_auc_pct": 100*m["hard_auc"],
           "prob_auc_pct": 100*m["prob_auc"], "sen_pct": 100*m["sen"],
           "spe_pct": 100*m["spe"], "asd_f1_pct": 100*m["asd_f1"],
           "log_loss": m["log_loss"], "brier": m["brier"]}
    if metadata:
        row.update(metadata)
    return row, m["prediction"]


def add_predictions(rows, fold, group, setting, method, indices, labels, probability, train_sha, c=None):
    pred = (np.asarray(probability) >= 0.5).astype(np.int64)
    for idx, y, p, yhat in zip(indices, labels, probability, pred):
        rows.append({"fold": fold, "prompt_group": group, "setting": setting, "method": method,
                     "subject_index_0based": int(idx), "y_asd": int(y), "prob_asd": float(p),
                     "pred_asd": int(yhat), "training_index_sha256": train_sha,
                     "selected_C": c})


def make_b1_variants(base_embeddings, bank_metadata, fold):
    variants = {}
    keys = [key for key in bank_metadata if key.startswith("P") or key == "C3-Unrelated-English-8"]
    for bank in keys:
        meta = bank_metadata[bank]
        variants[bank] = (base_embeddings[meta["array_key"]], np.asarray(meta["probe_classes"]), False)
    for seed in CFG["analysis"]["random_embedding_seeds"]:
        key = f"RND-Unit-{seed}"
        variants[key] = (base_embeddings[f"random_unit_{seed}"], np.asarray(["TD"]*4+["ASD"]*4), False)
    for source in ("P1-SPromptGL-8", "P4-Modality-Aware-8", "P5-Functional-Network-8"):
        meta = bank_metadata[source]
        variants[source.replace("-8", "") + "-WrongClass"] = (
            base_embeddings[meta["array_key"]],
            np.asarray(["TD" if c == "ASD" else "ASD" for c in meta["probe_classes"]]),
            False,
        )
    wt, wp = probe.fixed_projection(CFG["analysis"]["B1_projection_seed_base"] + fold)
    return variants, wt, wp


def token_diagnostics_for_bank(fold, bank, tokens, train_idx, test_idx, y_asd,
                               scores, sims, texts, classes, supervised_importance,
                               predictions, base_predictions, seed, bootstraps=200):
    rows = []
    train_scores = np.asarray(scores)[train_idx]
    test_scores = np.asarray(scores)[test_idx]
    abs_train_mean = np.mean(np.abs(train_scores), axis=0)
    top_group = np.argsort(abs_train_mean)[-4:]
    sup = np.asarray(supervised_importance, dtype=np.float64)
    top_supervised = np.argsort(sup)[-4:]
    overlap = len(set(top_group.tolist()) & set(top_supervised.tolist()))
    rng = np.random.default_rng(int(seed))
    boot_count = np.zeros(16, dtype=np.int64)
    for _ in range(int(bootstraps)):
        sample = rng.integers(0, len(train_idx), size=len(train_idx))
        rank = np.argsort(np.mean(np.abs(train_scores[sample]), axis=0))[-4:]
        boot_count[rank] += 1
    boot_freq = boot_count / max(1, int(bootstraps))
    pred = np.asarray(predictions, dtype=np.int64)
    base_pred = np.asarray(base_predictions, dtype=np.int64)
    y_test = np.asarray(y_asd, dtype=np.int64)
    correct = pred == y_test
    base_correct = base_pred == y_test
    sel = np.zeros((len(test_idx), 16), dtype=bool)
    if len(test_idx):
        sel[np.arange(len(test_idx))[:, None], np.argsort(np.abs(test_scores), axis=1)[:, -4:]] = True
    for tok in range(16):
        rows.append({"fold": fold, "prompt_group": bank, "diagnostic_type": "group_token",
                     "token": tok, "slot_origin": tok // 4, "token_within_slot": tok % 4,
                     "mean_abs_d_train": float(abs_train_mean[tok]),
                     "mean_similarity_asd_train": float(np.asarray(sims)[train_idx, tok][:, np.asarray(classes)=="ASD"].mean()),
                     "mean_similarity_td_train": float(np.asarray(sims)[train_idx, tok][:, np.asarray(classes)=="TD"].mean()),
                     "top4_mean_contrast": bool(tok in top_group), "bootstrap_top4_frequency": float(boot_freq[tok]),
                     "supervised_token_importance": float(sup[tok]), "supervised_top4": bool(tok in top_supervised),
                     "top4_overlap_with_supervised": overlap,
                     "test_top4_freq_correct": float(sel[correct, tok].mean()) if correct.any() else None,
                     "test_top4_freq_incorrect": float(sel[~correct, tok].mean()) if (~correct).any() else None,
                     "test_top4_freq_base_correct": float(sel[base_correct, tok].mean()) if base_correct.any() else None,
                     "test_top4_freq_base_incorrect": float(sel[~base_correct, tok].mean()) if (~base_correct).any() else None})
    prompt_top4 = []
    sims = np.asarray(sims)
    for pidx, text in enumerate(texts):
        prompt_score = sims[train_idx, :, pidx]
        p_top = np.argsort(prompt_score, axis=1)[:, -4:]
        p_freq = np.bincount(p_top.reshape(-1), minlength=16) / max(1, 4*len(train_idx))
        prompt_top4.append(set(np.argsort(prompt_score.mean(axis=0))[-4:].tolist()))
        for tok in range(16):
            rows.append({"fold": fold, "prompt_group": bank, "diagnostic_type": "prompt_token",
                         "prompt_index": pidx, "prompt_text": text, "token": tok,
                         "slot_origin": tok // 4, "token_within_slot": tok % 4,
                         "mean_prompt_similarity_train": float(prompt_score[:, tok].mean()),
                         "prompt_top4_frequency": float(p_freq[tok])})
    pair_jaccards = []
    for i in range(len(prompt_top4)):
        for j in range(i + 1, len(prompt_top4)):
            a, b = prompt_top4[i], prompt_top4[j]
            pair_jaccards.append(len(a & b) / max(1, len(a | b)))
    freq = np.mean(sel, axis=0) if len(sel) else np.zeros(16)
    entropy = float(-np.sum(np.where(freq > 0, freq*np.log(np.maximum(freq, 1e-12)), 0.0)))
    rows.append({"fold": fold, "prompt_group": bank, "diagnostic_type": "group_summary",
                 "prompt_count": len(texts), "group_top4_tokens": json.dumps(sorted(top_group.tolist())),
                 "supervised_top4_tokens": json.dumps(sorted(top_supervised.tolist())),
                 "top4_overlap_count": overlap,
                 "mean_within_group_prompt_top4_jaccard": float(np.mean(pair_jaccards)) if pair_jaccards else None,
                 "test_patient_selection_entropy_nats": entropy,
                 "test_patient_count": int(len(test_idx)),
                 "mean_test_top4_correct": float(sel[correct].mean()) if correct.any() else None,
                 "mean_test_top4_incorrect": float(sel[~correct].mean()) if (~correct).any() else None})
    return rows


def run_probes():
    if not (FEATURES / "fold10.npz").is_file():
        raise RuntimeError("frozen features are incomplete; run extraction first")
    embedding_file = np.load(RT / "prompt_embeddings.npz", allow_pickle=False)
    embeddings = {k: embedding_file[k] for k in embedding_file.files}
    bank_meta = json.loads((RT / "prompt_embedding_manifest.json").read_text(encoding="utf-8"))["bank_metadata"]
    prompt_sets = prompt_texts_and_classes()
    metric_rows, utility_rows, diagnostic_rows, pred_rows, c1_rows, gram_rows = [], [], [], [], [], []
    analysis_rng_before = rng_digest()
    started = time.time()
    anchor_tags = np.asarray(["TD"]*4 + ["ASD"]*4)
    for fold in range(1, 11):
        inner_seed = int(CFG["analysis"]["shared_inner_split_seed_base"]) + fold
        feat = load_features(fold)
        tokens = feat["tokens"].astype(np.float32)
        h = feat["h"].astype(np.float32)
        f = feat["f"].astype(np.float32)
        labels_orig = feat["labels"].astype(np.int64)
        y_all = probe.asd_target(labels_orig)
        tr = feat["probe_train"].astype(np.int64)
        te = feat["test"].astype(np.int64)
        ytr, yte = y_all[tr], y_all[te]
        logp = feat["logp"]
        pbase_all = feat["probability_all"][:, 0]
        base_odds = logp[:, 0] - logp[:, 1]
        base_metrics = probe.binary_metrics(yte, pbase_all[te])
        train_sha = sha_bytes(tr.tobytes())
        # S0: frozen base probabilities; M0: train-only calibration of base log-odds.
        base_row, _ = metric_csv_row(fold, "BASE", "frozen", "S0-Base", yte, pbase_all[te], len(tr), len(te))
        metric_rows.append(base_row)
        add_predictions(pred_rows, fold, "BASE", "frozen", "S0-Base", te, yte, pbase_all[te], train_sha)
        m0_prob, m0_c, m0_cv, _ = probe.nested_logistic(base_odds[tr, None], ytr, base_odds[te, None], inner_seed)
        m0_row, _ = metric_csv_row(fold, "NO_TEXT", "incremental", "M0-Calibrated-Base", yte, m0_prob, len(tr), len(te), m0_c)
        metric_rows.append(m0_row)
        add_predictions(pred_rows, fold, "NO_TEXT", "incremental", "M0-Calibrated-Base", te, yte, m0_prob, train_sha, m0_c)
        utility_rows.append({"fold": fold, "prompt_group": "ALL", "setting": "reference", "model": "M0",
                             "selected_C": m0_c, "inner_cv_log_loss": json.dumps(m0_cv), **{k:v for k,v in m0_row.items() if k.endswith("_pct") or k in ("log_loss","brier")}})

        # Unsupervised, label-independent local projections are the B1 M1 control.
        local_rng = np.random.default_rng(754700 + fold)
        local_axis = local_rng.standard_normal(9).astype(np.float32)
        local_axis /= max(float(np.linalg.norm(local_axis)), 1e-8)
        tn = tokens / np.maximum(np.linalg.norm(tokens, axis=-1, keepdims=True), 1e-8)
        m1_b1_features = np.einsum("ntd,d->nt", tn, local_axis).astype(np.float32)
        m1b1_prob, m1b1_c, m1b1_cv, _ = probe.nested_logistic(
            np.concatenate((base_odds[tr, None], m1_b1_features[tr]), axis=1), ytr,
            np.concatenate((base_odds[te, None], m1_b1_features[te]), axis=1), inner_seed)
        m1b1_row, _ = metric_csv_row(fold, "NO_TEXT", "incremental", "M1-B1-Local-NoText", yte, m1b1_prob, len(tr), len(te), m1b1_c)
        metric_rows.append(m1b1_row)
        add_predictions(pred_rows, fold, "NO_TEXT", "incremental", "M1-B1-Local-NoText", te, yte, m1b1_prob, train_sha, m1b1_c)

        # Supervised representation probes; C is selected using outer-train only.
        representation = [("S2-CAM-144", h), ("S3-DARG-320", f)]
        for name, matrix in representation:
            pp, cc, cv, _ = probe.nested_logistic(matrix[tr], ytr, matrix[te], inner_seed)
            row, _ = metric_csv_row(fold, "REPRESENTATION", "supervised", name, yte, pp, len(tr), len(te), cc)
            metric_rows.append(row)
            add_predictions(pred_rows, fold, "REPRESENTATION", "supervised", name, te, yte, pp, train_sha, cc)
        supervised_importance = np.zeros(16, dtype=np.float64)
        for tok in range(16):
            pp, cc, cv, model = probe.nested_logistic(tokens[tr, tok], ytr, tokens[te, tok], inner_seed)
            clf = model.named_steps["logisticregression"]
            supervised_importance[tok] = float(np.linalg.norm(clf.coef_))
            row, _ = metric_csv_row(fold, f"TOKEN-{tok:02d}", "supervised", f"S1-Token-{tok:02d}", yte, pp, len(tr), len(te), cc,
                                    {"inner_cv_log_loss": json.dumps(cv)})
            metric_rows.append(row)
            add_predictions(pred_rows, fold, f"TOKEN-{tok:02d}", "supervised", f"S1-Token-{tok:02d}", te, yte, pp, train_sha, cc)

        # B2 no-text control has the same trainable alignment and a fixed random anchor.
        wt2, wp2 = probe.fixed_projection(CFG["analysis"]["B2_fixed_text_projection_seed_base"] + fold)
        anchor_rng = np.random.default_rng(754800 + fold)
        anchor = anchor_rng.standard_normal(128).astype(np.float32)
        anchor /= max(float(np.linalg.norm(anchor)), 1e-8)
        m1b2_prob, m1b2_contrast, m1b2_c, m1b2_cv, _ = probe.nested_alignment(
            tokens[tr], ytr, tokens[te], np.zeros((8,128), dtype=np.float32), anchor_tags,
            inner_seed, train_base=base_odds[tr], eval_base=base_odds[te], random_anchor=anchor,
            steps=40)
        m1b2_row, _ = metric_csv_row(fold, "NO_TEXT", "incremental", "M1-B2-Trainable-NoText", yte, m1b2_prob, len(tr), len(te), m1b2_c)
        metric_rows.append(m1b2_row)
        add_predictions(pred_rows, fold, "NO_TEXT", "incremental", "M1-B2-Trainable-NoText", te, yte, m1b2_prob, train_sha, m1b2_c)

        b1_variants, wt1, wp1 = make_b1_variants(embeddings, bank_meta, fold)
        # Every bank receives the same fixed projection in a fold; B2 likewise shares its fixed text projection.
        b2_variants = {}
        for bank in [f"P{i}-" for i in range(1,7)]:
            match = next((k for k in bank_meta if k.startswith(bank)), None)
            if match is None:
                continue
            b2_variants[match] = (embeddings[bank_meta[match]["array_key"]], np.asarray(bank_meta[match]["probe_classes"]), False)
        b2_variants["C3-Unrelated-English-8"] = (embeddings[bank_meta["C3-Unrelated-English-8"]["array_key"]],
                                                  np.asarray(bank_meta["C3-Unrelated-English-8"]["probe_classes"]), False)
        for seed in CFG["analysis"]["random_embedding_seeds"]:
            b2_variants[f"RND-Unit-{seed}"] = (embeddings[f"random_unit_{seed}"], anchor_tags, False)
        for source in ("P1-SPromptGL-8", "P4-Modality-Aware-8", "P5-Functional-Network-8"):
            meta = bank_meta[source]
            wrong_classes = np.asarray(["TD" if c == "ASD" else "ASD" for c in meta["probe_classes"]])
            b2_variants[source.replace("-8", "") + "-WrongClass"] = (
                embeddings[meta["array_key"]], wrong_classes, False)
        anon_errors = {}
        for bank in ("P1-SPromptGL-8", "P5-Functional-Network-8"):
            meta = bank_meta[bank]
            e = embeddings[meta["array_key"]]
            q = e @ wp2
            q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
            anon, err = probe.anonymous_gram_matched(q, CFG["analysis"]["anonymous_orthogonal_seed_base"] + fold + (0 if bank.startswith("P1") else 1000))
            anon_key = bank.replace("-8", "") + "-Anonymous-Gram-Matched"
            b2_variants[anon_key] = (anon, np.asarray(meta["probe_classes"]), True)
            anon_errors[("B2", anon_key)] = err
        # Audit B1 and B2 anonymous Gram matrices independently in their actual use spaces.
        token_proj_b1 = tokens @ wt1
        token_proj_b1 /= np.maximum(np.linalg.norm(token_proj_b1, axis=-1, keepdims=True), 1e-8)
        for bank in ("P1-SPromptGL-8", "P5-Functional-Network-8"):
            meta = bank_meta[bank]
            q0 = embeddings[meta["array_key"]] @ wp1
            q0 /= np.maximum(np.linalg.norm(q0, axis=-1, keepdims=True), 1e-8)
            anon, err = probe.anonymous_gram_matched(q0, CFG["analysis"]["anonymous_orthogonal_seed_base"] + fold + (0 if bank.startswith("P1") else 1000))
            anon_key = bank.replace("-8", "") + "-Anonymous-Gram-Matched"
            b1_variants[anon_key] = (anon, np.asarray(meta["probe_classes"]), True)
            gram_rows.append({"fold":fold,"setting":"B1","pair":bank,"max_abs_gram_error":err})
        for (setting, bank), err in anon_errors.items():
            gram_rows.append({"fold":fold,"setting":setting,"pair":bank,"max_abs_gram_error":err})

        # M1-B1 is shared across prompt comparisons; retain its row beside each M0/M2 comparison.
        m0_metrics = probe.binary_metrics(yte, m0_prob)
        m1b1_metrics = probe.binary_metrics(yte, m1b1_prob)
        m1b2_metrics = probe.binary_metrics(yte, m1b2_prob)
        pred_base = np.argmax(np.exp(logp[te]), axis=1)
        pred_base_asd = 1 - pred_base
        prompt_keys = set(b1_variants) | set(b2_variants)
        for bank in sorted(prompt_keys):
            texts, source_tags, base_classes = prompt_sets.get(bank, ([], [], []))
            # C2 variants inherit the source Prompt text and labels for explanation.
            if bank.endswith("Anonymous-Gram-Matched"):
                source_name = "P1-SPromptGL-8" if bank.startswith("P1") else "P5-Functional-Network-8"
                texts = prompt_sets[source_name][0]
                source_tags = prompt_sets[source_name][1]
                if bank not in b2_variants:
                    base_classes = prompt_sets[source_name][2]
                else:
                    base_classes = prompt_sets[source_name][2]
            variants_here = []
            if bank in b1_variants:
                raw, classes, is_projected = b1_variants[bank]
                if is_projected:
                    d, sims, q = probe.b1_features_projected(token_proj_b1, raw, classes)
                else:
                    d, sims, q = probe.b1_features(tokens, raw, classes, wt1, wp1)
                b1prob, b1c, b1cv, b1model = probe.nested_logistic(d[tr], ytr, d[te], inner_seed)
                row, _ = metric_csv_row(fold, bank, "B1-fixed", "B1-Prompt", yte, b1prob, len(tr), len(te), b1c)
                metric_rows.append(row); add_predictions(pred_rows, fold, bank, "B1-fixed", "B1-Prompt", te, yte, b1prob, train_sha, b1c)
                m2xtr = np.concatenate((base_odds[tr,None], d[tr]), axis=1)
                m2xte = np.concatenate((base_odds[te,None], d[te]), axis=1)
                m2b1prob, m2b1c, m2b1cv, _ = probe.nested_logistic(m2xtr, ytr, m2xte, inner_seed)
                row, _ = metric_csv_row(fold, bank, "B1-fixed", "M2-B1-BasePlusPrompt", yte, m2b1prob, len(tr), len(te), m2b1c)
                metric_rows.append(row); add_predictions(pred_rows, fold, bank, "B1-fixed", "M2-B1-BasePlusPrompt", te, yte, m2b1prob, train_sha, m2b1c)
                m2b1_metrics = probe.binary_metrics(yte, m2b1prob)
                utility_rows.extend([
                    {"fold":fold,"prompt_group":bank,"setting":"B1","model":"M0","selected_C":m0_c,"inner_cv_log_loss":json.dumps(m0_cv),**{k:v for k,v in m0_row.items() if k.endswith("_pct") or k in ("log_loss","brier")}},
                    {"fold":fold,"prompt_group":bank,"setting":"B1","model":"M1","selected_C":m1b1_c,"inner_cv_log_loss":json.dumps(m1b1_cv),**{k:v for k,v in m1b1_row.items() if k.endswith("_pct") or k in ("log_loss","brier")}},
                    {"fold":fold,"prompt_group":bank,"setting":"B1","model":"M2","selected_C":m2b1c,"inner_cv_log_loss":json.dumps(m2b1cv),**{k:v for k,v in row.items() if k.endswith("_pct") or k in ("log_loss","brier")},"method":"M2-B1-BasePlusPrompt"},
                ])
                if bank.endswith("WrongClass"):
                    source_bank = {"P1-SPromptGL-WrongClass":"P1-SPromptGL-8",
                                   "P4-Modality-Aware-WrongClass":"P4-Modality-Aware-8",
                                   "P5-Functional-Network-WrongClass":"P5-Functional-Network-8"}[bank]
                    original_classes = np.asarray(prompt_sets[source_bank][2])
                    original_d, _, _ = probe.b1_features(tokens, raw, original_classes, wt1, wp1)
                    _, original_c, _, original_model = probe.nested_logistic(
                        original_d[tr], ytr, original_d[te], inner_seed)
                    # Keep a correct-class fit fixed while changing only the class assignment.
                    fixed_prob = original_model.predict_proba(-original_d[te])[:,1]
                    fixed_row, _ = metric_csv_row(fold, bank, "C1-fixed-head", "C1-WrongClass-FixedHead", yte, fixed_prob, len(tr), len(te), original_c)
                    metric_rows.append(fixed_row); add_predictions(pred_rows, fold, bank, "C1-fixed-head", "C1-WrongClass-FixedHead", te, yte, fixed_prob, train_sha, b1c)
                    swap_prob, swap_c, swap_cv, _ = probe.nested_logistic(d[tr], ytr, d[te], inner_seed)
                    swap_row, _ = metric_csv_row(fold, bank, "C1-refit-head", "C1-WrongClass-RefitHead", yte, swap_prob, len(tr), len(te), swap_c)
                    metric_rows.append(swap_row); add_predictions(pred_rows, fold, bank, "C1-refit-head", "C1-WrongClass-RefitHead", te, yte, swap_prob, train_sha, swap_c)
                if bank in ("P1-SPromptGL-8", "P2-BrainPrompt-2", "P3-Class-Definition-8", "P4-Modality-Aware-8", "P5-Functional-Network-8", "P6-Heterogeneity-Aware-8", "C3-Unrelated-English-8") or bank.endswith("Anonymous-Gram-Matched") or bank.startswith("RND-Unit-"):
                    diag_texts = texts or [f"vector_{i}" for i in range(len(classes))]
                    diag_rows = token_diagnostics_for_bank(
                        fold, bank, tokens, tr, te, yte, d, sims, diag_texts, classes,
                        supervised_importance, m2b1_metrics["prediction"], pred_base_asd,
                        754600+fold*101+len(bank), CFG["analysis"]["bootstrap_resamples"])
                    diagnostic_rows.extend(diag_rows)
                # Fixed-vs-refit wrong-class controls are summarized separately.
            if bank in b2_variants:
                raw2, classes2, projected2 = b2_variants[bank]
                if projected2:
                    q2 = raw2
                else:
                    q2 = raw2 @ wp2
                    q2 /= np.maximum(np.linalg.norm(q2, axis=-1, keepdims=True), 1e-8)
                # B2 prompt probe without base score.
                p_b2, d_b2, cb2, cvb2, _ = probe.nested_alignment(
                    tokens[tr], ytr, tokens[te], q2, classes2, inner_seed,
                    steps=40)
                row, _ = metric_csv_row(fold, bank, "B2-trained-alignment", "B2-Prompt", yte, p_b2, len(tr), len(te), cb2)
                metric_rows.append(row); add_predictions(pred_rows, fold, bank, "B2-trained-alignment", "B2-Prompt", te, yte, p_b2, train_sha, cb2)
                p_m2b2, d_m2b2, cm2, cvm2, _ = probe.nested_alignment(
                    tokens[tr], ytr, tokens[te], q2, classes2, inner_seed,
                    train_base=base_odds[tr], eval_base=base_odds[te], steps=40)
                row, _ = metric_csv_row(fold, bank, "B2-trained-alignment", "M2-B2-BasePlusPrompt", yte, p_m2b2, len(tr), len(te), cm2)
                metric_rows.append(row); add_predictions(pred_rows, fold, bank, "B2-trained-alignment", "M2-B2-BasePlusPrompt", te, yte, p_m2b2, train_sha, cm2)
                m2b2_metrics = probe.binary_metrics(yte, p_m2b2)
                utility_rows.extend([
                    {"fold":fold,"prompt_group":bank,"setting":"B2","model":"M0","selected_C":m0_c,"inner_cv_log_loss":json.dumps(m0_cv),**{k:v for k,v in m0_row.items() if k.endswith("_pct") or k in ("log_loss","brier")}},
                    {"fold":fold,"prompt_group":bank,"setting":"B2","model":"M1","selected_C":m1b2_c,"inner_cv_log_loss":json.dumps(m1b2_cv),**{k:v for k,v in m1b2_row.items() if k.endswith("_pct") or k in ("log_loss","brier")}},
                    {"fold":fold,"prompt_group":bank,"setting":"B2","model":"M2","selected_C":cm2,"inner_cv_log_loss":json.dumps(cvm2),**{k:v for k,v in row.items() if k.endswith("_pct") or k in ("log_loss","brier")},"method":"M2-B2-BasePlusPrompt"},
                ])
                if bank in ("P1-SPromptGL-8", "P5-Functional-Network-8"):
                    orig = embeddings[bank_meta[bank]["array_key"]]
                    q0 = orig @ wp2; q0 /= np.maximum(np.linalg.norm(q0, axis=-1, keepdims=True), 1e-8)
                    gram_rows.append({"fold":fold,"setting":"B2","pair":bank,
                                      "max_abs_gram_error":float(np.max(np.abs(q0@q0.T-q2@q2.T)))})
                if bank.endswith("WrongClass"):
                    source_bank = {"P1-SPromptGL-WrongClass":"P1-SPromptGL-8",
                                   "P4-Modality-Aware-WrongClass":"P4-Modality-Aware-8",
                                   "P5-Functional-Network-WrongClass":"P5-Functional-Network-8"}[bank]
                    original_classes = np.asarray(bank_meta[source_bank]["probe_classes"])
                    _, _, fixed_c, _, fixed_model = probe.nested_alignment(
                        tokens[tr], ytr, tokens[te], q2, original_classes, inner_seed,
                        steps=40)
                    fixed_prob, _ = probe.alignment_predict(fixed_model, tokens[te], q2, classes2)
                    fixed_row, _ = metric_csv_row(fold, bank, "C1-B2-fixed-head", "C1-B2-WrongClass-FixedHead",
                                                  yte, fixed_prob, len(tr), len(te), fixed_c)
                    metric_rows.append(fixed_row)
                    add_predictions(pred_rows, fold, bank, "C1-B2-fixed-head", "C1-B2-WrongClass-FixedHead",
                                    te, yte, fixed_prob, train_sha, fixed_c)
            print(f"[probe] fold={fold}/10 banks={len(prompt_keys)} B1/B2 complete", flush=True)

        # Store a check row for the repeated no-text M1 reference comparisons.
        c1_rows.append({"fold":fold,"base_acc_pct":100*base_metrics["acc"],"m0_log_loss":m0_metrics["log_loss"],
                        "m1_b1_log_loss":m1b1_metrics["log_loss"],"m1_b2_log_loss":m1b2_metrics["log_loss"],
                        "probe_train_count":len(tr),"fold2_overlap_removed":2 if fold==2 else 0})

    if rng_digest() != analysis_rng_before:
        raise RuntimeError("frozen-probe analysis unexpectedly changed process-global RNG")
    write_csv(RESULTS / "prompt_comparison.csv", metric_rows)
    write_csv(RESULTS / "incremental_utility.csv", utility_rows)
    write_csv(RESULTS / "token_diagnostics.csv", diagnostic_rows)
    write_csv(RESULTS / "probe_predictions.csv", pred_rows)
    write_csv(RESULTS / "anonymous_gram_audit.csv", gram_rows)
    write_csv(RESULTS / "no_text_reference.csv", c1_rows)
    atomic_json(RESULTS / "probe_execution.json", {
        "status":"PROBES_COMPLETE", "prompt_comparison_rows":len(metric_rows),
        "incremental_rows":len(utility_rows),"token_diagnostic_rows":len(diagnostic_rows),
        "prediction_rows":len(pred_rows),"global_rng_before":analysis_rng_before,
        "global_rng_after":rng_digest(),"elapsed_seconds":time.time()-started,
        "B2_steps":40,"B2_C_values":list(probe.C_VALUES),
        "shared_inner_split_seed_base":int(CFG["analysis"]["shared_inner_split_seed_base"]),
        "shared_inner_split_seeds":[int(CFG["analysis"]["shared_inner_split_seed_base"])+f for f in range(1,11)],
        "B1_fixed_projection_seeds":[CFG["analysis"]["B1_projection_seed_base"]+f for f in range(1,11)],
        "B2_projection_seeds":[CFG["analysis"]["B2_fixed_text_projection_seed_base"]+f for f in range(1,11)],
        "Fold2_new_probe_train_excluded_rows_1based":[377,407],
        "anonymous_max_gram_error":max((float(x["max_abs_gram_error"]) for x in gram_rows),default=0.0),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("extract", "encode", "probe"), default=None)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.phase in (None, "extract"):
        extract_all()
    if args.phase in (None, "encode"):
        encode_prompt_banks()
    if args.phase in (None, "probe"):
        run_probes()


if __name__ == "__main__":
    main()
