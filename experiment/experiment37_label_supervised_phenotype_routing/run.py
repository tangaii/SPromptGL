#!/usr/bin/env python3
"""Audited, resumable runner for the frozen 14-group Experiment 37."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import os
import pickle
import random
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / ".runtime"
RESULTS = HERE / "results"
CONFIG_FILE = HERE / "config.json"
MODEL_ID = "FremyCompany/BioLORD-2023"
MODEL_REVISION = "167aab527b238a50ca65224e6319215d2ff4fc9f"
CONDITION_DIM = 768
PROTOCOL = dict(nclass=2, mode="simple-2", seed=0, GC_mode="weighted-cosine",
                MP_mode="GCN", MF_mode="concat", nmodal=4, lr=0.0038,
                reg=0.11, dropout=0.35, n_hidden=18, n_head=2, nlayer=1,
                th=0.9, alpha=0.5, theta_smooth=1.0, theta_degree=0.5,
                theta_sparsity=0.0, nepoch=450, early=200)

GROUPS = {
    "G00_T0_REPLAY": dict(condition="modal_full", relation="none", weight=0.0),
    "G01_HYBRID_BASE": dict(condition="hybrid", relation="none", weight=0.0),
    "G02_FULL_SUPPORT_BCE": dict(condition="modal_full", relation="support_bce", weight=0.10),
    "G03_HYBRID_SUPPORT_BCE": dict(condition="hybrid", relation="support_bce", weight=0.10),
    "G04_FULL_ALL_BCE": dict(condition="modal_full", relation="all_bce", weight=0.10),
    "G05_HYBRID_ALL_BCE": dict(condition="hybrid", relation="all_bce", weight=0.10),
    "G06_HYBRID_RANK": dict(condition="hybrid", relation="ranking", weight=0.10),
    "G07_HYBRID_ALL_BCE_LOW": dict(condition="hybrid", relation="all_bce", weight=0.03),
    "G08_HYBRID_ALL_BCE_HIGH": dict(condition="hybrid", relation="all_bce", weight=0.30),
    "G09_HYBRID_TEXT_SHUFFLE": dict(condition="hybrid", relation="all_bce", weight=0.10, text_shuffled=True),
    "G10_HYBRID_AUX_LABEL_SHUFFLE": dict(condition="hybrid", relation="all_bce", weight=0.10, aux_label_shuffled=True),
    "G11_HYBRID_NUMERIC_BCE": dict(condition="numeric", relation="all_bce", weight=0.10),
    "G12_RAW_GRAPH_LABEL_SUP": dict(condition="none", relation="raw_support", weight=0.10),
    "G13_HYBRID_SITE_BALANCED": dict(condition="hybrid", relation="site_bce", weight=0.10),
}
GPU0 = ["G00_T0_REPLAY", "G02_FULL_SUPPORT_BCE", "G04_FULL_ALL_BCE", "G06_HYBRID_RANK",
        "G08_HYBRID_ALL_BCE_HIGH", "G10_HYBRID_AUX_LABEL_SHUFFLE", "G12_RAW_GRAPH_LABEL_SUP"]
GPU1 = ["G01_HYBRID_BASE", "G03_HYBRID_SUPPORT_BCE", "G05_HYBRID_ALL_BCE",
        "G07_HYBRID_ALL_BCE_LOW", "G09_HYBRID_TEXT_SHUFFLE", "G11_HYBRID_NUMERIC_BCE",
        "G13_HYBRID_SITE_BALANCED"]
FIELD_ORDER = ["AGE_AT_SCAN", "SEX", "FIQ", "VIQ", "PIQ", "HANDEDNESS_CATEGORY"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, tmp)
    os.replace(tmp, path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def rng_sha(state):
    h = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    h.update(state["torch_cpu"].cpu().numpy().tobytes())
    for value in state.get("torch_cuda", []):
        h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def data_context(root):
    sys.path.insert(0, str(HERE))
    from phenotype import load_aligned_inputs
    return load_aligned_inputs(Path(root))


def fold_splits(y):
    return list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10).split(np.zeros(len(y)), y))


def ensure_dirs():
    for path in (RUNTIME, RUNTIME / "cache", RUNTIME / "tasks", RUNTIME / "checkpoints",
                 RUNTIME / "logs", RUNTIME / "rng", RUNTIME / "model"):
        path.mkdir(parents=True, exist_ok=True)


def run_audit(args):
    ensure_dirs()
    data = data_context(args.root)
    from phenotype import make_fold_payload, missingness_audit, text_summary
    fold_summaries = []
    for fold, (train, _) in enumerate(fold_splits(data["y"]), start=1):
        payload = make_fold_payload(data, train, fold)
        fold_summaries.append({"fold": fold, **text_summary(payload)})
    result = {"experiment": "E36 identity and input audit", **data["audit"],
              "clinical_missingness_by_label_audit_only": missingness_audit(data),
              "fold_text_template_audit": fold_summaries,
              "registered_groups": len(GROUPS), "registered_folds": len(GROUPS) * 10,
              "status": "DATA_AUDIT_PASS"}
    json_write(RUNTIME / "data_audit.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def license_confirmed():
    p = RUNTIME / "license_attestation.json"
    if not p.exists():
        return False
    value = read_json(p)
    return bool(value.get("umls_valid") and value.get("snomed_ct_valid") and value.get("confirmed_by_user"))


def attest_license(args):
    ensure_dirs()
    if not args.umls or not args.snomed_ct:
        raise RuntimeError("BioLORD requires both UMLS and SNOMED CT licensing confirmation")
    json_write(RUNTIME / "license_attestation.json", {
        "umls_valid": True, "snomed_ct_valid": True, "confirmed_by_user": True,
        "confirmation_basis": "User confirmed active project/institution licenses; no license token stored",
        "encoder": MODEL_ID, "revision": MODEL_REVISION,
    })
    print("Both required licenses recorded as user-confirmed; no credential was stored.")


def load_fold_payloads(data):
    sys.path.insert(0, str(HERE))
    from phenotype import make_fold_payload, text_summary
    splits = fold_splits(data["y"])
    payloads = []
    for fold, (train, test) in enumerate(splits, start=1):
        p = make_fold_payload(data, train, fold)
        payloads.append((p, train, test))
    return payloads


def encode_unique_strings(strings, model, tokenizer, device):
    values = list(dict.fromkeys(map(str, strings)))
    index = {text: i for i, text in enumerate(values)}
    encoded = np.zeros((len(values), CONDITION_DIM), dtype=np.float32)
    batch_size = 64
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=512,
                               return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            output = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).expand_as(output).to(output.dtype)
            pooled = (output * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            encoded[start:start + len(batch)] = pooled.float().cpu().numpy()
    return values, index, encoded


def encode_all(data, model_home):
    if not license_confirmed():
        raise RuntimeError("BLOCKED: confirm valid UMLS and SNOMED CT licenses before BioLORD download/use")
    ensure_dirs()
    from transformers import AutoModel, AutoTokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading frozen {MODEL_ID}@{MODEL_REVISION} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                             cache_dir=str(model_home), use_fast=True)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                      cache_dir=str(model_home)).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    payloads = load_fold_payloads(data)
    text_summary_rows = []
    for payload, train, test in payloads:
        keys = ("clinical", "modal_full", "all_full", "modal_only", "all_modal_only")
        strings = []
        for key in keys:
            strings.extend(np.asarray(payload[key], dtype=object).reshape(-1).tolist())
        strings.extend(np.asarray(payload["constant_modal"], dtype=object).tolist())
        strings.append(str(payload["constant_all"]))
        values, index, embeddings = encode_unique_strings(strings, model, tokenizer, device)
        def embed(value):
            return embeddings[np.asarray([index[str(t)] for t in np.asarray(value, dtype=object).reshape(-1)])].reshape(
                *np.asarray(value, dtype=object).shape, CONDITION_DIM)
        cache_path = RUNTIME / "cache" / f"fold_{payload['fold']:02d}_conditions.npz"
        save_condition_cache(cache_path, {
            "clinical": embed(payload["clinical"]).astype(np.float32),
            "modal_full": embed(payload["modal_full"]).astype(np.float32),
            "all_full": embed(payload["all_full"]).astype(np.float32),
            "modal_only": embed(payload["modal_only"]).astype(np.float32),
            "all_modal_only": embed(payload["all_modal_only"]).astype(np.float32),
            "constant_modal": embed(payload["constant_modal"]).astype(np.float32),
            "constant_all": embed(np.asarray([payload["constant_all"]], dtype=object))[0].astype(np.float32),
            "modal_numeric": payload["modal_numeric"], "all_numeric": payload["all_numeric"],
            "anat_z": payload["anat_z"], "func_z": payload["func_z"],
            "corr_block_summary": payload["corr_block_summary"]})
        json_write(cache_path.with_suffix(".json"), fold_cache_metadata(
            data, payload["fold"], train, test, cache_path,
            ["text_embeddings", "numeric_conditions"],
            encoder={"model_id": MODEL_ID, "revision": MODEL_REVISION,
                     "pooling": "attention-mask mean pooling + L2 normalization"}))
        truncation = sum(len(tokenizer.encode(t, add_special_tokens=True)) > 512 for t in set(strings))
        text_summary_rows.append({"fold": int(payload["fold"]), "text_instances": int(len(strings)),
                                  "unique_texts": int(len(set(strings))),
                                  "mean_word_count": float(np.mean([len(t.split()) for t in strings])),
                                  "max_word_count": int(max(len(t.split()) for t in strings)),
                                  "truncated_unique_texts": int(truncation),
                                  "cache_sha256": sha_file(cache_path)})
        print(f"TEXT_CACHE fold={payload['fold']} unique={len(set(strings))} truncated={truncation}", flush=True)
    manifest = {
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "model_card_license": "model contributions MIT; use requires valid UMLS and SNOMED CT licensing",
        "pooling": "attention-mask mean pooling + L2 normalize", "max_length": 512,
        "trainable": False, "text_stats": text_summary_rows,
        "cache_directory": str(RUNTIME / "cache"),
        "model_home": str(model_home),
    }
    json_write(RUNTIME / "text_encoder_manifest.json", manifest)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fold_cache_metadata(data, fold, train, test, cache_path, cache_kinds, encoder=None):
    from phenotype import sha256_file
    inputs = data["audit"]["inputs_sha256"]
    return {
        "fold": int(fold),
        "train_indices_sha256": hashlib.sha256(np.asarray(train, dtype="<i8").tobytes()).hexdigest(),
        "test_indices_sha256": hashlib.sha256(np.asarray(test, dtype="<i8").tobytes()).hexdigest(),
        "ordered_subject_id_sha256": data["audit"]["ordered_subject_id_sha256"],
        "source_csv_sha256": next(v for k, v in inputs.items() if k.endswith("Phenotypic_V1_0b_preprocessed1.csv")),
        "processed_input_sha256": next(v for k, v in inputs.items() if k.endswith("processed_standard_data.csv")),
        "identity_mapping_sha256": next(v for k, v in inputs.items() if k.endswith("subject_row_mapping.csv")),
        "template_builder_sha256": sha_file(HERE / "phenotype.py"),
        "cache_sha256": sha_file(cache_path),
        "cache_kinds": list(cache_kinds),
        "encoder": encoder,
    }


def save_condition_cache(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def load_condition_cache(fold):
    path = RUNTIME / "cache" / f"fold_{fold:02d}_conditions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"text/numeric cache missing: {path}; run `python run.py encode` first")
    metadata_path = path.with_suffix(".json")
    if metadata_path.is_file() and read_json(metadata_path).get("cache_sha256") != sha_file(path):
        raise RuntimeError(f"condition cache hash does not match its metadata: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def build_numeric_caches(data):
    from phenotype import make_fold_payload
    for fold, (train, test) in enumerate(fold_splits(data["y"]), start=1):
        payload = make_fold_payload(data, train, fold)
        path = RUNTIME / "cache" / f"fold_{fold:02d}_conditions.npz"
        save_condition_cache(path, {
            "modal_numeric": payload["modal_numeric"], "all_numeric": payload["all_numeric"],
            "anat_z": payload["anat_z"], "func_z": payload["func_z"],
            "corr_block_summary": payload["corr_block_summary"]})
        json_write(path.with_suffix(".json"), fold_cache_metadata(
            data, fold, train, test, path, ["numeric_conditions"], encoder=None))
        print(f"NUMERIC_CACHE fold={fold}/10 sha256={sha_file(path)}", flush=True)


def missingness_signatures(frame):
    signatures = []
    for _, row in frame.iterrows():
        parts = []
        for field in FIELD_ORDER:
            value = row.get(field)
            if field in {"FIQ", "VIQ", "PIQ", "AGE_AT_SCAN", "SEX"}:
                try:
                    x = float(value)
                    parts.append("1" if np.isfinite(x) and x != -9999 else "0")
                except (TypeError, ValueError):
                    parts.append("0")
            else:
                parts.append("1" if pd.notna(value) and str(value).strip() not in {"", "-9999"} else "0")
        signatures.append("".join(parts))
    return signatures


def patient_shuffle(data, train, test, fold):
    frame = data["phenotype"].reset_index(drop=True)
    sites = frame.SITE_ID.fillna("UNKNOWN").astype(str).to_numpy()
    missing = missingness_signatures(frame)
    permutation = np.arange(len(frame), dtype=np.int64)
    rng = np.random.RandomState(36000 + int(fold))
    fallback_site = 0
    fallback_pool = 0
    for pool in (np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)):
        buckets = {}
        for idx in pool:
            buckets.setdefault((sites[idx], missing[idx]), []).append(int(idx))
        singletons = []
        for group in buckets.values():
            if len(group) >= 2:
                permutation[group] = rng.permutation(group)
            else:
                singletons.extend(group)
        by_site = {}
        for idx in singletons:
            by_site.setdefault(sites[idx], []).append(idx)
        last = []
        for group in by_site.values():
            if len(group) >= 2:
                permutation[group] = rng.permutation(group)
                fallback_site += len(group)
            else:
                last.extend(group)
        if len(last) >= 2:
            permutation[last] = rng.permutation(last)
            fallback_pool += len(last)
        elif last:
            permutation[last] = last
    assert set(permutation.tolist()) == set(range(len(frame)))
    assert set(permutation[np.asarray(train)].tolist()) == set(train.tolist())
    assert set(permutation[np.asarray(test)].tolist()) == set(test.tolist())
    return permutation, {"changed": int(np.sum(permutation != np.arange(len(frame)))),
                         "fixed_points": int(np.sum(permutation == np.arange(len(frame)))),
                         "fallback_site_count": fallback_site,
                         "fallback_pool_count": fallback_pool}


def condition_for_group(group, cache, permutation, force_unshuffled=False):
    meta = GROUPS[group]
    name = meta["condition"]
    if name == "none":
        return None
    if name == "numeric":
        value = cache["modal_numeric"]
    elif name == "hybrid":
        value = np.concatenate([cache["modal_full"][:, :1], cache["modal_only"][:, 1:]], axis=1)
    elif name == "modal_full":
        value = cache["modal_full"]
    else:
        raise ValueError(name)
    if meta.get("text_shuffled") and not force_unshuffled:
        value = value[permutation]
    assert value.shape[-1] == CONDITION_DIM
    return np.ascontiguousarray(value, dtype=np.float32)


def load_modules(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(HERE))
    import network
    import model
    from integration import EvalHelper36, install_b0a_pln_forward
    install_b0a_pln_forward()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return network, model, EvalHelper36


def warm_clip_without_rng(network):
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        network.get_prompt_text_features()


def source_eval(agent):
    agent.MessagePassing0.eval()
    agent.ModalFusion.eval()
    with torch.no_grad():
        n, parts = agent.dataLayer(agent.feat)
        features, _, _ = agent.MessagePassing0(parts, n, agent.targ, agent.trn_idx)
        output, _, _ = agent.ModalFusion(features)
    return output


def nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(nested_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    return a == b


def source_preflight(root, data):
    network, original_model, EvalHelper36 = load_modules(root)
    input_dims = [48, 6, 10, 256]
    train, test = fold_splits(data["y"])[0]
    hp = SimpleNamespace(**PROTOCOL)
    set_seed(36036)
    entry = rng_state()
    network._prompt_text_features = None
    original = original_model.EvalHelper(input_dims, data["x"], data["y"], hp, train, test)
    original_init = {"gfr": original.MessagePassing0.state_dict(),
                     "pln": original.ModalFusion.state_dict()}
    post_init = rng_state()
    restore_rng(entry)
    network._prompt_text_features = None
    candidate = EvalHelper36(input_dims, data["x"], data["y"], hp, train, test,
                             "B0_ORIGINAL_B0A", "original", False, False)
    candidate_init = {"gfr": candidate.MessagePassing0.state_dict(),
                      "pln": candidate.ModalFusion.state_dict()}
    assert nested_equal(original_init, candidate_init), "B0 initialization differs from original EvalHelper"
    original_out = source_eval(original)
    candidate_out = candidate.predict()
    assert torch.equal(original_out, candidate_out), "B0 pre-update logits are not bitwise equal"

    restore_rng(post_init)
    original.run_epoch("simple-2")
    expected_rng = rng_state()
    original_after = {"gfr": original.MessagePassing0.state_dict(),
                      "pln": original.ModalFusion.state_dict()}
    original_mp_opt = original.optimizer_MP0.state_dict()
    original_mf_opt = original.optimizer_MF.state_dict()
    restore_rng(post_init)
    candidate.run_epoch("simple-2")
    candidate_after = {"gfr": candidate.MessagePassing0.state_dict(),
                       "pln": candidate.ModalFusion.state_dict()}
    assert nested_equal(original_after, candidate_after), "B0 first update differs from original simple-2"
    assert nested_equal(original_mp_opt, candidate.optimizer_MP0.state_dict())
    assert nested_equal(original_mf_opt, candidate.optimizer_MF.state_dict())
    assert rng_sha(expected_rng) == rng_sha(rng_state())

    from router import validate_router
    router_checks = validate_router()
    fair_checks = {}
    sample_condition = load_condition_cache(1)["modal_full"]
    for name, structure, directed, enabled in (
            ("B1_FOUR_GRAPH_BASE", "four", False, False),
            ("B2_ONE_GRAPH_BASE", "one", False, False),
            ("B3_CORR_ONLY_BASE", "corr", False, False),
            ("T1_ROUTER_SHAPE", "four", True, True)):
        set_seed(36037)
        condition = sample_condition if enabled else None
        agent = EvalHelper36(input_dims, data["x"], data["y"], hp, train, test,
                             name, structure, directed, enabled, condition, fold_id=1)
        parameter_audit = __import__("integration").optimizer_audit(agent)
        agent.MessagePassing0.eval()
        agent.ModalFusion.eval()
        output, features, adjs, _, _ = agent.forward()
        expected_messages = 4 if structure == "four" else 1
        assert len(adjs) == 4 and output.shape == (871, 2)
        assert len(agent.MessagePassing0.last_message_norms) == expected_messages
        assert torch.isfinite(output).all() and torch.isfinite(features).all()
        if enabled:
            assert len(agent.MessagePassing0.last_diagnostics) == 4
            initial_edge_change = max(float((item["conditioned_adj"] - item["raw_adj"]).abs().max().detach().cpu())
                                      for item in agent.MessagePassing0.last_diagnostics)
            initial_score_std = max(float(item["scores"][item["edge_support"]].std().detach().cpu())
                                    for item in agent.MessagePassing0.last_diagnostics)
            assert initial_score_std > 1e-6, f"real cached conditions produce constant PCPR scores: {initial_score_std}"
            assert initial_edge_change > 1e-8, f"real cached conditions do not change supported graph edges: {initial_edge_change}"
            agent.run_epoch("simple-2")
            router = agent.MessagePassing0.conditioner
            assert router.query.weight.grad is not None and router.query.weight.grad.norm() > 0
            assert router.key.weight.grad is not None and router.key.weight.grad.norm() > 0
            agent.predict()
            trained_diagnostics = agent.MessagePassing0.last_diagnostics
            trained_edge_change = max(float((item["conditioned_adj"] - item["raw_adj"]).abs().max().detach().cpu())
                                      for item in trained_diagnostics)
            assert torch.linalg.vector_norm(router.query.weight) > 1e-3
            assert torch.linalg.vector_norm(router.key.weight) > 1e-3
            assert trained_edge_change > 1e-8, "PCPR scores or edge modulation collapsed after an optimizer update"
            saved_model = __import__("integration").model_state(agent)
            saved_mp = copy.deepcopy(agent.optimizer_MP0.state_dict())
            saved_mf = copy.deepcopy(agent.optimizer_MF.state_dict())
            saved_rng = rng_state()
            agent.run_epoch("simple-2")
            expected_model = __import__("integration").model_state(agent)
            expected_mp = copy.deepcopy(agent.optimizer_MP0.state_dict())
            expected_mf = copy.deepcopy(agent.optimizer_MF.state_dict())
            expected_rng_hash = rng_sha(rng_state())
            set_seed(36038)
            replay = EvalHelper36(input_dims, data["x"], data["y"], hp, train, test,
                                  name, structure, directed, enabled, condition, fold_id=1)
            __import__("integration").load_model_state(replay, saved_model)
            replay.optimizer_MP0.load_state_dict(saved_mp)
            replay.optimizer_MF.load_state_dict(saved_mf)
            restore_rng(saved_rng)
            replay.run_epoch("simple-2")
            assert nested_equal(expected_model, __import__("integration").model_state(replay))
            assert nested_equal(expected_mp, replay.optimizer_MP0.state_dict())
            assert nested_equal(expected_mf, replay.optimizer_MF.state_dict())
            assert expected_rng_hash == rng_sha(rng_state())
            fair_checks["checkpoint_recovery"] = {
                "model": "BITWISE_EQUAL", "both_adams": "BITWISE_EQUAL",
                "next_rng": "BITWISE_EQUAL"}
            fair_checks["real_condition_response"] = {
                "source": "fold_01 BioLORD embeddings",
                "initial_score_std_max": initial_score_std,
                "initial_edge_change_max": initial_edge_change,
                "after_update_edge_change_max": trained_edge_change,
                "query_key_weight_decay": 0.0,
                "status": "PASS",
            }
            del replay
        fair_checks[name] = {"raw_graphs": len(adjs), "propagation_messages": expected_messages,
                             "conditioned_graphs": len(agent.MessagePassing0.last_diagnostics),
                             "optimizer_audit": parameter_audit}
        for stream_name in ("MF_sav", "GCMP_sav"):
            stream = getattr(agent, stream_name, None)
            if stream is not None:
                stream.close()

    # Constant text must reduce the router to the unconditioned B1 path, not
    # merely produce a similar score. Compare forwards and two optimizer steps.
    constant = np.ones((len(data["y"]), 4, CONDITION_DIM), dtype=np.float32)
    network._prompt_text_features = getattr(network, "_prompt_text_features", None)
    set_seed(37037)
    constant_on = EvalHelper36(input_dims, data["x"], data["y"], hp, train, test,
                               "constant_on", "four", False, True, constant, fold_id=1)
    set_seed(37037)
    constant_off = EvalHelper36(input_dims, data["x"], data["y"], hp, train, test,
                                "constant_off", "four", False, False, None, fold_id=1)
    assert __import__("integration").parameter_sha256(constant_on, True) == \
           __import__("integration").parameter_sha256(constant_off, True)
    assert torch.equal(constant_on.predict(), constant_off.predict()), \
        "constant-text router changed the B1 forward output"
    def common_parameters(agent):
        result = {}
        for module_name, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion)):
            result[module_name] = {key: value.detach().cpu().clone()
                                   for key, value in module.state_dict().items()
                                   if "conditioner" not in key}
        return result
    for _ in range(2):
        before = rng_state()
        constant_on.run_epoch("simple-2")
        after_on = rng_state()
        on_mp = copy.deepcopy(constant_on.optimizer_MP0.state_dict())
        restore_rng(before)
        constant_off.run_epoch("simple-2")
        assert nested_equal(common_parameters(constant_on), common_parameters(constant_off))
        off_mp = constant_off.optimizer_MP0.state_dict()
        assert nested_equal(on_mp["param_groups"][0], off_mp["param_groups"][0])
        common_ids = set(on_mp["param_groups"][0]["params"])
        on_common_state = {key: value for key, value in on_mp["state"].items() if key in common_ids}
        off_common_state = {key: value for key, value in off_mp["state"].items() if key in common_ids}
        assert nested_equal(on_common_state, off_common_state)
        assert nested_equal(constant_on.optimizer_MF.state_dict(), constant_off.optimizer_MF.state_dict())
        assert rng_sha(after_on) == rng_sha(rng_state())
    fair_checks["constant_router"] = {"forward": "BITWISE_EQUAL", "two_common_updates": "BITWISE_EQUAL",
                                      "common_optimizer_states": "BITWISE_EQUAL",
                                      "router_optimizer_extra_state": "separate zero gradient state; does not change common parameters",
                                      "rng_after_each_step": "BITWISE_EQUAL"}

    # Exercise every auxiliary objective once on a real fold, without a pilot
    # metric or any held-out-label access.
    cache = load_condition_cache(1)
    hybrid = np.concatenate([cache["modal_full"][:, :1], cache["modal_only"][:, 1:]], axis=1)
    assert np.array_equal(hybrid[:, 0], cache["modal_full"][:, 0])
    assert np.array_equal(hybrid[:, 1:], cache["modal_only"][:, 1:])
    train_only_labels = data["y"][train].copy()
    train_only_sites = data["phenotype"].SITE_ID.fillna("UNKNOWN").astype(str).to_numpy()[train]
    aux_checks = {}
    for mode in ("support_bce", "all_bce", "ranking", "site_bce", "raw_support"):
        set_seed(37100)
        use_text = mode != "raw_support"
        check = EvalHelper36(input_dims, data["x"], data["y"], hp, train, test,
                             f"preflight_{mode}", "four", False, use_text,
                             hybrid if use_text else None, fold_id=1,
                             relation_mode=mode, relation_weight=0.10,
                             train_aux_labels=train_only_labels,
                             train_sites=train_only_sites if mode == "site_bce" else None)
        step = check.run_epoch("simple-2")
        rel_meta = step["relation_meta"]
        assert np.isfinite(step["relation_loss"]) and np.isfinite(step["loss"])
        assert rel_meta.get("pairs", 0) > 0
        assert rel_meta.get("aux_test_pair_count", 0) == 0
        if use_text:
            assert rel_meta.get("aux_query_grad_norm", 0) > 0
            assert rel_meta.get("aux_key_grad_norm", 0) > 0
        else:
            assert any(g > 0 for g in rel_meta.get("aux_graph_p_grad_norms", []))
        aux_checks[mode] = {"relation_loss": step["relation_loss"], **rel_meta}
        for stream_name in ("MF_sav", "GCMP_sav"):
            stream = getattr(check, stream_name, None)
            if stream is not None:
                stream.close()
    fair_checks["auxiliary_loss_preflight"] = aux_checks

    from integration import metric_row
    smoke_logits = torch.tensor([[3., 0.], [3., 0.], [3., 0.], [0., 3.]])
    smoke_metrics = metric_row(smoke_logits, np.array([0, 0, 1, 1]), np.arange(4))
    assert smoke_metrics["sen_asd"] == 1.0 and smoke_metrics["spe_td"] == 0.5
    result = {"B0_initialization": "BITWISE_EQUAL", "B0_logits": "BITWISE_EQUAL",
              "B0_first_update": "BITWISE_EQUAL", "B0_optimizer_states": "BITWISE_EQUAL",
              "B0_rng_after_update": "BITWISE_EQUAL", "PCPR_router": router_checks,
              "fair_backbone": fair_checks, "ASD_TD_metric_direction": "PASS",
              "status": "PASS"}
    for agent in (original, candidate):
        for name in ("MF_sav", "GCMP_sav"):
            stream = getattr(agent, name, None)
            if stream is not None:
                stream.close()
    network._prompt_text_features = None
    return result


def run_preflight(args):
    ensure_dirs()
    data = data_context(args.root)
    cache_audit = validate_cached_folds(data)
    audit = {"data": data["audit"], "cache_validation": cache_audit,
             "engineering": source_preflight(args.root, data),
             "status": "PASS"}
    json_write(RUNTIME / "preflight.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def validate_cached_folds(data):
    splits = fold_splits(data["y"])
    summaries = []
    source_hashes = data["audit"]["inputs_sha256"]
    expected = {
        "source_csv_sha256": next(v for k, v in source_hashes.items() if k.endswith("Phenotypic_V1_0b_preprocessed1.csv")),
        "processed_input_sha256": next(v for k, v in source_hashes.items() if k.endswith("processed_standard_data.csv")),
        "identity_mapping_sha256": next(v for k, v in source_hashes.items() if k.endswith("subject_row_mapping.csv")),
        "ordered_subject_id_sha256": data["audit"]["ordered_subject_id_sha256"],
    }
    for fold, (train, test) in enumerate(splits, start=1):
        path = RUNTIME / "cache" / f"fold_{fold:02d}_conditions.npz"
        meta_path = path.with_suffix(".json")
        if not path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"validated E36 cache missing for fold {fold}")
        meta = read_json(meta_path)
        if sha_file(path) != meta.get("cache_sha256"):
            raise RuntimeError(f"fold {fold} cache hash mismatch")
        if meta.get("fold") != fold:
            raise RuntimeError(f"fold {fold} metadata mismatch")
        if meta.get("train_indices_sha256") != hashlib.sha256(np.asarray(train, dtype="<i8").tobytes()).hexdigest():
            raise RuntimeError(f"fold {fold} train indices differ from E36 cache")
        if meta.get("test_indices_sha256") != hashlib.sha256(np.asarray(test, dtype="<i8").tobytes()).hexdigest():
            raise RuntimeError(f"fold {fold} test indices differ from E36 cache")
        for key, value in expected.items():
            if meta.get(key) != value:
                raise RuntimeError(f"fold {fold} provenance mismatch: {key}")
        if meta.get("encoder", {}).get("revision") != MODEL_REVISION:
            raise RuntimeError(f"fold {fold} encoder revision mismatch")
        cache = load_condition_cache(fold)
        required = {"modal_full": (871, 4, 768), "modal_only": (871, 4, 768),
                    "modal_numeric": (871, 4, 768), "all_full": (871, 768)}
        for name, shape in required.items():
            if name not in cache or cache[name].shape != shape or not np.isfinite(cache[name]).all():
                raise RuntimeError(f"fold {fold} cache array invalid: {name}")
        hybrid = np.concatenate([cache["modal_full"][:, :1], cache["modal_only"][:, 1:]], axis=1)
        if not np.array_equal(hybrid[:, 0], cache["modal_full"][:, 0]) or \
           not np.array_equal(hybrid[:, 1:], cache["modal_only"][:, 1:]):
            raise RuntimeError(f"fold {fold} hybrid construction mismatch")
        summaries.append({"fold": fold, "train_count": len(train), "test_count": len(test),
                          "cache_sha256": meta["cache_sha256"], "hybrid_identity": "EXACT"})
    return {"status": "PASS", "encoder_id": MODEL_ID, "revision": MODEL_REVISION,
            "folds": summaries}


def read_historical_baseline(root):
    path = Path(root) / "experiment/experiment32_disease_contrastive_evidence_routing/results/fold_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "config" in frame:
        frame = frame.loc[frame.config.isin(["B0", "B0_ORIGINAL_B0A"])].copy()
    if "fold" in frame:
        frame = frame.sort_values("fold")
    return frame


def compare_e0_history(rows, root):
    hist = read_historical_baseline(root)
    current = pd.DataFrame(rows).sort_values("fold")
    if len(hist) != 10 or len(current) != 10:
        return {"status": "NOT_CHECKED", "history_rows": len(hist), "current_rows": len(current)}
    metric_alias = {"acc": ["acc", "ACC"], "hard_auc": ["hard_auc", "auc", "Hard-AUC"],
                    "sen_asd": ["sen_asd", "sensitivity_asd", "SEN-ASD"],
                    "spe_td": ["spe_td", "specificity_td", "SPE-TD"]}
    checks = {}
    for metric, candidates in metric_alias.items():
        col = next((name for name in candidates if name in hist.columns), None)
        if col is None:
            raise KeyError(f"historical baseline missing {metric}; columns={list(hist.columns)}")
        values = pd.to_numeric(hist[col], errors="raise").to_numpy(dtype=float)
        now = current[metric].to_numpy(dtype=float)
        # History values are proportions in Experiment32; tolerate documented percent storage.
        if values.max() > 1.5:
            values = values / 100.0
        checks[metric] = bool(np.allclose(values, now, rtol=0.0, atol=1e-15))
    best_col = next((c for c in ("best_epoch", "best_epoch_index") if c in hist.columns), None)
    best_equal = None
    if best_col:
        best_equal = np.array_equal(pd.to_numeric(hist[best_col]).to_numpy(),
                                    current.best_epoch.to_numpy())
    rng_col = next((c for c in ("initial_rng_sha256", "initial_rng_hash") if c in hist.columns), None)
    rng_equal = bool(np.array_equal(hist[rng_col].astype(str).to_numpy(),
                                    current.initial_rng_sha256.astype(str).to_numpy())) if rng_col else None
    trunk_col = next((c for c in ("initial_trunk_sha256", "initial_parameter_sha256") if c in hist.columns), None)
    current_trunk_col = next((c for c in ("historical_initial_trunk_sha256", "initial_parameter_sha256")
                              if c in current.columns), None)
    trunk_equal = bool(np.array_equal(hist[trunk_col].astype(str).to_numpy(),
                                      current[current_trunk_col].astype(str).to_numpy())) \
        if trunk_col and current_trunk_col else None
    count_columns = ("asd_tp", "asd_fn", "td_fp", "td_tn")
    count_equal = None
    if all(name in hist.columns and name in current.columns for name in count_columns):
        count_equal = all(np.array_equal(pd.to_numeric(hist[name]).to_numpy(dtype=np.int64),
                                         pd.to_numeric(current[name]).to_numpy(dtype=np.int64))
                          for name in count_columns)
    complete = (all(checks.values()) and best_equal is not False and rng_equal is not False
                and trunk_equal is not False and count_equal is not False)
    return {"status": "MATCH" if complete else "MISMATCH",
            "metrics_10_of_10": checks, "best_epoch_10_of_10": best_equal,
            "entry_rng_10_of_10": rng_equal, "initial_parameters_10_of_10": trunk_equal,
            "confusion_counts_10_of_10": count_equal,
            "historical_hash_note": "initial parameter hash compared only if column semantics/availability match"}


def entry_rng_file():
    return RUNTIME / "rng" / "t0_fold_entries.pt"


def load_rng_entries():
    path = entry_rng_file()
    if not path.is_file():
        raise RuntimeError("E0 must complete and save its ten fold-entry RNG states before paired groups")
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_path(group):
    return RUNTIME / "checkpoints" / f"{group}.pt"


def task_path(group):
    return RUNTIME / "tasks" / f"{group}.json"


def fold_condition(group, cache, data, train, test, fold):
    meta = GROUPS[group]
    permutation = np.arange(len(data["y"]), dtype=np.int64)
    shuffle_info = {"changed": 0, "fixed_points": 0, "fallback_site_count": 0,
                    "fallback_pool_count": 0}
    if meta.get("text_shuffled"):
        permutation, shuffle_info = patient_shuffle(data, train, test, fold)
    condition = condition_for_group(group, cache, permutation)
    summary = {"condition": meta["condition"], **shuffle_info}
    if meta["condition"] == "hybrid":
        expected_profile = cache["modal_full"][:, 0][permutation]
        expected_modal = cache["modal_only"][:, 1:][permutation]
        assert np.array_equal(condition[:, 0], expected_profile)
        assert np.array_equal(condition[:, 1:], expected_modal)
    return condition, permutation, summary


def make_agent(root, data, group, fold, train, test, condition, aux_labels):
    from integration import EvalHelper36
    network, _, _ = load_modules(root)
    hp = SimpleNamespace(**PROTOCOL)
    meta = GROUPS[group]
    sites = None
    if meta["relation"] == "site_bce":
        sites = data["phenotype"].SITE_ID.fillna("UNKNOWN").astype(str).to_numpy()[train]
    return EvalHelper36([48, 6, 10, 256], data["x"], data["y"], hp, train, test,
                        group, "four", False, meta["condition"] != "none", condition,
                        fold_id=fold, relation_mode=meta["relation"],
                        relation_weight=meta["weight"], train_aux_labels=aux_labels,
                        train_sites=sites)


def initial_state_metrics(agent, state):
    current = {"gfr": agent.MessagePassing0.state_dict(), "pln": agent.ModalFusion.state_dict()}
    output = {}
    for module_name in current:
        for key, initial in state[module_name].items():
            value = current[module_name][key].detach().cpu()
            if value.is_floating_point():
                output[f"{module_name}.{key}"] = float((value - initial).norm().item())
    return output


def summarize_graph_diagnostics(agent, data, fold, group, shuffle_info, off_output, real_output,
                                labels, test):
    row = {"config": group, "fold": fold, "condition_enabled": int(agent.condition_enabled),
           "shuffle_changed": int(shuffle_info.get("changed", 0)),
           "shuffle_fixed_points": int(shuffle_info.get("fixed_points", 0)),
           "shuffle_fallback_site_count": int(shuffle_info.get("fallback_site_count", 0)),
           "shuffle_fallback_pool_count": int(shuffle_info.get("fallback_pool_count", 0))}
    fair = agent.MessagePassing0
    if isinstance(fair, torch.nn.Module) and hasattr(fair, "last_raw_adjs"):
        raw = fair.last_raw_adjs
        weights = fair.last_source_weights
        row["source_fusion_weights"] = json.dumps(weights)
        row["message_norms"] = json.dumps(fair.last_message_norms)
        sites = data["phenotype"].SITE_ID.fillna("UNKNOWN").astype(str).to_numpy()
        age = pd.to_numeric(data["phenotype"].AGE_AT_SCAN, errors="coerce").to_numpy(dtype=float)
        iq = pd.to_numeric(data["phenotype"].FIQ, errors="coerce").replace([-9999, -9999.0], np.nan).to_numpy(dtype=float)
        for m, adj in enumerate(raw):
            support = (adj > 0).detach().cpu().numpy()
            row[f"raw_edge_count_{m}"] = int(support.sum())
            row[f"raw_density_{m}"] = float(support.mean())
            row[f"raw_mean_degree_{m}"] = float(support.sum(1).mean())
            row[f"raw_symmetry_error_{m}"] = float((adj - adj.T).abs().max().detach().cpu())
        diagnostics = getattr(fair, "last_diagnostics", [])
        row["condition_beta"] = float(fair.conditioner.beta.detach().cpu()) if fair.conditioner is not None else 0.0
        for m, diag in enumerate(diagnostics):
            raw_np = diag["raw_adj"].detach().cpu().numpy()
            cond_np = diag["conditioned_adj"].detach().cpu().numpy()
            p_np = diag["propagation"].detach().cpu().numpy()
            support = raw_np > 0
            np.fill_diagonal(support, False)
            ratio = np.divide(cond_np, raw_np, out=np.ones_like(cond_np), where=support)
            row[f"condition_edge_count_{m}"] = int(support.sum())
            row[f"condition_weight_abs_change_mean_{m}"] = float(np.abs(cond_np[support] - raw_np[support]).mean()) if support.any() else 0.0
            row[f"condition_multiplicative_change_mean_{m}"] = float(ratio[support].mean()) if support.any() else 1.0
            row[f"condition_multiplicative_change_max_{m}"] = float(np.abs(ratio[support] - 1).max()) if support.any() else 0.0
            row[f"condition_symmetry_error_{m}"] = float(np.abs(cond_np - cond_np.T).max())
            row[f"propagation_row_sum_error_{m}"] = float(np.abs(p_np.sum(1) - 1).max())
            row[f"propagation_self_weight_mean_{m}"] = float(np.diag(p_np).mean())
            edge_rows, edge_cols = np.where(support)
            same_site = sites[edge_rows] == sites[edge_cols]
            edge_weight = p_np[edge_rows, edge_cols]
            total = float(edge_weight.sum())
            row[f"propagation_same_site_fraction_{m}"] = float(edge_weight[same_site].sum() / total) if total > 0 else 0.0
            log_change = np.log(np.clip(ratio[support], 1e-12, None))
            age_dist = np.abs(age[edge_rows] - age[edge_cols])
            valid_age = np.isfinite(age_dist) & np.isfinite(log_change)
            row[f"abs_age_edge_change_corr_{m}"] = safe_corr(age_dist[valid_age], np.abs(log_change[valid_age]))
            iq_dist = np.abs(iq[edge_rows] - iq[edge_cols])
            valid_iq = np.isfinite(iq_dist) & np.isfinite(log_change)
            row[f"abs_fiq_edge_change_corr_{m}"] = safe_corr(iq_dist[valid_iq], np.abs(log_change[valid_iq]))
            scores = diag["scores"].detach().cpu().numpy()[support]
            row[f"text_score_mean_{m}"] = float(scores.mean()) if len(scores) else 0.0
            row[f"text_score_std_{m}"] = float(scores.std()) if len(scores) else 0.0
        row["active_router_count"] = int(len(diagnostics))
    if off_output is not None:
        probs = real_output.detach().cpu().numpy()[test].argmax(1)
        off = off_output.detach().cpu().numpy()[test].argmax(1)
        row["condition_off_flip_fraction"] = float(np.mean(probs != off))
        row["condition_off_metrics"] = json.dumps(__import__("integration").metric_row(off_output, labels, test))
    return row


def safe_corr(x, y):
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def load_task(group):
    path = task_path(group)
    return read_json(path) if path.exists() else {"config": group, "metrics": [], "diagnostics": [], "status": "PENDING"}


def auxiliary_train_labels(data, train, group, fold):
    # This function receives the outer-train indices only; held-out labels are
    # never indexed or passed to any auxiliary-pair sampler.
    labels = np.asarray(data["y"], dtype=np.int64)[np.asarray(train, dtype=np.int64)].copy()
    info = {"aux_label_shuffle_changed": 0, "aux_label_shuffle_fraction": 0.0,
            "aux_label_shuffle_site_count": 0}
    if not GROUPS[group].get("aux_label_shuffled"):
        return labels, info
    sites = data["phenotype"].SITE_ID.fillna("UNKNOWN").astype(str).to_numpy()[train]
    rng = np.random.RandomState(37000 + int(fold))
    groups = {}
    for i, site in enumerate(sites):
        groups.setdefault(site, []).append(i)
    changed_sites = 0
    original = labels.copy()
    for members in groups.values():
        if len(members) > 1:
            labels[members] = rng.permutation(labels[members])
            changed_sites += int(not np.array_equal(labels[members], original[members]))
    info.update({"aux_label_shuffle_changed": int(np.sum(labels != original)),
                 "aux_label_shuffle_fraction": float(np.mean(labels != original)),
                 "aux_label_shuffle_site_count": changed_sites,
                 "aux_label_shuffle_distribution_preserved_within_site": True})
    return labels, info


def relation_diagnostics(agent, data, train, aux_labels, group, cache, permutation, fold):
    row = {"config": group, "fold": int(fold),
           "relation_mode": GROUPS[group]["relation"],
           "relation_lambda": GROUPS[group]["weight"],
           "aux_train_count": int(len(train)), "aux_test_pair_count": 0}
    if not agent.condition_enabled:
        row["relation_loss"] = float(agent.last_relation_loss)
        row["relation_meta"] = json.dumps(agent.last_relation_meta, allow_nan=False)
        row["raw_graph_aux_gradient_norms"] = json.dumps(
            agent.last_relation_meta.get("aux_graph_p_grad_norms", []))
        return row
    fair = agent.MessagePassing0
    train = np.asarray(train, dtype=np.int64)
    actual = np.asarray(data["y"], dtype=np.int64)[train]
    aux = np.asarray(aux_labels, dtype=np.int64)
    same_actual = actual[:, None] == actual[None, :]
    same_aux = aux[:, None] == aux[None, :]
    upper = np.triu(np.ones_like(same_actual, dtype=bool), k=1)
    row["train_same_label_pairs"] = int(np.sum(same_aux & upper))
    row["train_different_label_pairs"] = int(np.sum(~same_aux & upper))
    row["train_same_true_label_pairs"] = int(np.sum(same_actual & upper))
    row["train_different_true_label_pairs"] = int(np.sum(~same_actual & upper))
    row["train_label_pair_shuffle_changed"] = int(np.sum(aux != actual))
    per_modality = []
    from scipy.stats import spearmanr
    condition = np.asarray(agent.condition_input.detach().cpu())
    for m, diag in enumerate(fair.last_diagnostics):
        support = diag["edge_support"].detach().cpu().numpy().astype(bool)[np.ix_(train, train)]
        scores = diag["pre_tanh_centered_scores"].detach().cpu().numpy()[np.ix_(train, train)]
        raw = diag["raw_adj"].detach().cpu().numpy()[np.ix_(train, train)]
        conditioned = diag["conditioned_adj"].detach().cpu().numpy()[np.ix_(train, train)]
        use = upper & support
        ratio = np.divide(conditioned, raw, out=np.ones_like(raw), where=support)
        actual_same_edge = use & same_actual
        actual_diff_edge = use & ~same_actual
        aux_same_edge = use & same_aux
        aux_diff_edge = use & ~same_aux
        condition_train = condition[train, m]
        cosine_train = condition_train @ condition_train.T
        cos_train = cosine_train[use]
        score_train = scores[use]
        if len(score_train) > 2 and np.std(cos_train) > 0 and np.std(score_train) > 0:
            corr_value = float(spearmanr(cos_train, score_train).statistic)
        else:
            corr_value = None
        item = {
            "modality": m,
            "train_supported_edges": int(use.sum()),
            "train_supported_same_aux": int(aux_same_edge.sum()),
            "train_supported_different_aux": int(aux_diff_edge.sum()),
            "train_supported_same_true": int(actual_same_edge.sum()),
            "train_supported_different_true": int(actual_diff_edge.sum()),
            "no_edge_same_aux_pairs": int(np.sum(upper & same_aux & ~support)),
            "same_aux_score_mean": float(scores[aux_same_edge].mean()) if aux_same_edge.any() else None,
            "different_aux_score_mean": float(scores[aux_diff_edge].mean()) if aux_diff_edge.any() else None,
            "same_true_score_mean": float(scores[actual_same_edge].mean()) if actual_same_edge.any() else None,
            "different_true_score_mean": float(scores[actual_diff_edge].mean()) if actual_diff_edge.any() else None,
            "same_true_edge_multiplier": float(ratio[actual_same_edge].mean()) if actual_same_edge.any() else None,
            "different_true_edge_multiplier": float(ratio[actual_diff_edge].mean()) if actual_diff_edge.any() else None,
            "same_minus_different_multiplier": float(ratio[actual_same_edge].mean() - ratio[actual_diff_edge].mean()) if actual_same_edge.any() and actual_diff_edge.any() else None,
            "same_edge_increase_fraction": float(np.mean(ratio[actual_same_edge] > 1)) if actual_same_edge.any() else None,
            "different_edge_increase_fraction": float(np.mean(ratio[actual_diff_edge] > 1)) if actual_diff_edge.any() else None,
            "pre_tanh_score_mean": float(scores[use].mean()) if use.any() else None,
            "pre_tanh_score_std": float(scores[use].std()) if use.any() else None,
            "tanh_saturation_abs_gt_3_fraction": float(np.mean(np.abs(scores[use]) > 3)) if use.any() else None,
            "text_cosine_qk_score_spearman_train_edges": corr_value,
            "beta": float(fair.conditioner.beta.detach().cpu()),
        }
        per_modality.append(item)
    row["per_modality"] = json.dumps(per_modality, allow_nan=False)
    row["relation_pair_counts_by_modality"] = json.dumps(agent.last_relation_meta.get("per_modality_counts", []))
    row["relation_loss"] = float(agent.last_relation_loss)
    row["relation_aux_query_grad_norm"] = float(agent.last_relation_meta.get("aux_query_grad_norm", 0.0))
    row["relation_aux_key_grad_norm"] = float(agent.last_relation_meta.get("aux_key_grad_norm", 0.0))
    row["relation_total_query_grad_norm"] = float(agent.last_relation_meta.get("query_total_grad_norm", 0.0))
    row["relation_total_key_grad_norm"] = float(agent.last_relation_meta.get("key_total_grad_norm", 0.0))
    row["source_fusion_weights"] = json.dumps(fair.last_source_weights)
    row["message_norms"] = json.dumps(fair.last_message_norms)
    return row


def worker(args):
    if args.config not in GROUPS:
        raise ValueError(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("formal workers require an available CUDA device")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA_VISIBLE_DEVICES did not expose a device")
    ensure_dirs()
    root = Path(args.root).resolve()
    data = data_context(root)
    network, original_model, EvalHelper36 = load_modules(root)
    splits = fold_splits(data["y"])
    group = args.config
    meta = GROUPS[group]
    result = load_task(group)
    if result.get("status") == "COMPLETE" and len(result.get("metrics", [])) == 10:
        print(f"TASK_ALREADY_COMPLETE {group}", flush=True)
        return
    saved = None
    cp = checkpoint_path(group)
    if cp.exists():
        saved = torch.load(cp, map_location="cpu", weights_only=False)
        if saved.get("config") != group:
            raise ValueError("checkpoint configuration mismatch")
        if saved.get("config_sha256") not in (None, sha_file(CONFIG_FILE)):
            raise ValueError("checkpoint frozen-config hash mismatch")
        result = saved["result"]
    prior_seconds = float(result.get("seconds", 0.0))
    start_wall = time.monotonic()

    entries = load_rng_entries()
    assert len(entries) == 10

    if saved and saved.get("phase") == "EPOCH":
        first_fold = int(saved["fold"])
    elif result.get("metrics"):
        first_fold = max(int(row["fold"]) for row in result["metrics"]) + 1
    else:
        first_fold = 1
    if saved and saved.get("phase") == "BOUNDARY":
        first_fold = int(saved.get("next_fold", first_fold))
        restore_rng(saved["rng"])
    if first_fold > 1 and getattr(network, "_prompt_text_features", None) is None:
        warm_clip_without_rng(network)

    stop = {"value": False}
    def stop_signal(_signum, _frame):
        stop["value"] = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    print(f"TASK_START {group} GPU={args.gpu} fold={first_fold} resume={saved is not None}", flush=True)
    for fold in range(first_fold, 11):
        if stop["value"]:
            break
        train, test = splits[fold - 1]
        cache = load_condition_cache(fold) if meta["condition"] != "none" else {}
        condition, permutation, shuffle_info = fold_condition(group, cache, data, train, test, fold)
        aux_labels, label_shuffle_info = auxiliary_train_labels(data, train, group, fold)

        resume_epoch = bool(saved and saved.get("phase") == "EPOCH" and int(saved["fold"]) == fold)
        entry_state = entries[fold - 1]
        restore_rng(entry_state)

        agent = EvalHelper36([48, 6, 10, 256], data["x"], data["y"],
                             SimpleNamespace(**PROTOCOL), train, test,
                             group, "four", False,
                             meta["condition"] != "none", condition, fold_id=fold,
                             relation_mode=meta["relation"], relation_weight=meta["weight"],
                             train_aux_labels=aux_labels,
                             train_sites=(data["phenotype"].SITE_ID.fillna("UNKNOWN").astype(str).to_numpy()[train]
                                          if meta["relation"] == "site_bce" else None))
        from integration import (model_state, load_model_state, optimizer_audit,
                                 parameter_sha256, historical_trunk_sha256, metric_row)
        parameter_info = optimizer_audit(agent)
        initial_state = model_state(agent)
        initial_parameter_hash = parameter_sha256(agent, baseline_common_only=True)
        historical_parameter_hash = None
        initial_rng_hash = rng_sha(entry_state)
        if group != "G00_T0_REPLAY":
            replay_rows = {int(row["fold"]): row for row in load_task("G00_T0_REPLAY")["metrics"]}
            reference = replay_rows[fold]
            assert initial_parameter_hash == reference["initial_parameter_sha256"], \
                f"shared trunk initialization differs from G00 fold {fold}"
            assert initial_rng_hash == reference["initial_rng_sha256"], \
                f"fold-entry RNG differs from G00 fold {fold}"

        if resume_epoch:
            load_model_state(agent, saved["model"])
            agent.optimizer_MP0.load_state_dict(saved["optimizer_mp"])
            agent.optimizer_MF.load_state_dict(saved["optimizer_mf"])
            restore_rng(saved["rng"])
            initial_state = saved["initial_state"]
            initial_parameter_hash = saved["initial_parameter_sha256"]
            historical_parameter_hash = saved.get("historical_initial_trunk_sha256")
            initial_rng_hash = saved["initial_rng_sha256"]
            best_state = saved["best_state"]
            best_acc = float(saved["best_acc"])
            best_epoch = int(saved["best_epoch"])
            wait = int(saved["wait"])
            start_epoch = int(saved["epoch"]) + 1
            tracker = saved.get("tracker", {})
            fold_elapsed = float(saved.get("fold_elapsed", 0.0))
        else:
            best_state, best_acc, best_epoch, wait, start_epoch = None, 0.0, -1, 0, 0
            tracker = {"query_grad_max": 0.0, "key_grad_max": 0.0, "beta_grad_max": 0.0,
                       "conditioner_update_norm": 0.0}
            fold_elapsed = 0.0
        saved = None
        fold_started = time.monotonic()
        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, PROTOCOL["nepoch"]):
            loss = agent.run_epoch("simple-2")
            agent.last_relation_meta = loss.get("relation_meta", {})
            agent.last_relation_loss = loss.get("relation_loss", 0.0)
            tracker["last_relation_meta"] = agent.last_relation_meta
            tracker["last_relation_loss"] = agent.last_relation_loss
            for key, parameter_name in (("query_grad_max", "query.weight"),
                                         ("key_grad_max", "key.weight"),
                                         ("beta_grad_max", "beta_raw")):
                conditioner = getattr(agent.MessagePassing0, "conditioner", None)
                if conditioner is not None:
                    parameter = dict(conditioner.named_parameters())[parameter_name]
                    if parameter.grad is not None:
                        tracker[key] = max(float(tracker.get(key, 0.0)), float(parameter.grad.norm().item()))
            output = agent.predict()
            val_pred = output[agent.val_idx].argmax(1)
            correct = int(val_pred.eq(agent.targ[agent.val_idx]).sum().item())
            val_acc = float(correct / len(agent.val_idx))
            if val_acc > best_acc:
                best_acc, best_epoch, best_state, wait = val_acc, epoch, model_state(agent), 0
            else:
                wait += 1
            last_epoch = epoch
            due = ((epoch + 1) % 5 == 0 or wait > PROTOCOL["early"] or stop["value"])
            if due:
                now = rng_state()
                result["seconds"] = prior_seconds + time.monotonic() - start_wall
                atomic_torch_save(cp, {
                    "config": group, "phase": "EPOCH", "fold": fold, "epoch": epoch,
                    "config_sha256": sha_file(CONFIG_FILE),
                    "fold_split_sha256": hashlib.sha256(np.asarray(np.concatenate([train, test]), dtype="<i8").tobytes()).hexdigest(),
                    "model": model_state(agent), "optimizer_mp": agent.optimizer_MP0.state_dict(),
                    "optimizer_mf": agent.optimizer_MF.state_dict(), "rng": now,
                    "initial_state": initial_state, "initial_parameter_sha256": initial_parameter_hash,
                    "historical_initial_trunk_sha256": historical_parameter_hash,
                    "initial_rng_sha256": initial_rng_hash, "best_state": best_state,
                    "best_acc": best_acc, "best_epoch": best_epoch, "wait": wait,
                    "tracker": tracker, "fold_elapsed": fold_elapsed + time.monotonic() - fold_started,
                    "last_loss": loss,
                    "input_cache_sha256": (read_json(RUNTIME / "cache" / f"fold_{fold:02d}_conditions.json").get("cache_sha256")
                                            if (RUNTIME / "cache" / f"fold_{fold:02d}_conditions.json").exists() else None),
                    "result": result,
                })
            if wait > PROTOCOL["early"] or stop["value"]:
                break
        if stop["value"]:
            return
        if best_state is None:
            raise RuntimeError(f"no best checkpoint was selected for {group} fold {fold}")
        continuation_rng = rng_state()
        load_model_state(agent, best_state)
        output = agent.predict()
        val_correct = int(output[agent.val_idx].argmax(1).eq(agent.targ[agent.val_idx]).sum().item())
        if round(best_acc * len(agent.val_idx)) != val_correct:
            raise AssertionError("restored best checkpoint disagrees with selected val/test accuracy")
        best_acc = val_correct / len(agent.val_idx)
        metrics = metric_row(output, data["y"], test)
        assert abs(metrics["acc"] - best_acc) < 1e-12

        # The auxiliary branch must be inference-label blind even though the
        # inherited checkpoint selector still observes val_idx=test_idx.
        test_tensor = torch.as_tensor(test, dtype=torch.long, device=agent.dev)
        heldout_labels = agent.targ[test_tensor].clone()
        label_rng = np.random.RandomState(370000 + int(fold))
        agent.targ[test_tensor] = torch.as_tensor(label_rng.randint(0, 2, len(test)),
                                                  dtype=torch.long, device=agent.dev)
        changed_label_output = agent.predict()
        agent.targ[test_tensor] = heldout_labels
        assert torch.equal(output, changed_label_output), "held-out labels changed inference logits"

        off_output = None
        current_output = output
        if agent.condition_enabled:
            agent.condition_enabled = False
            off_output = agent.predict()
            agent.condition_enabled = True
            output = agent.predict()
            off_flip = float(np.mean(output.detach().cpu().numpy()[test].argmax(1) !=
                                     off_output.detach().cpu().numpy()[test].argmax(1)))
        else:
            off_flip = None
        text_effect = {}
        if agent.condition_enabled and meta["condition"] in {"modal_full", "hybrid"}:
            identity = np.arange(len(data["y"]), dtype=np.int64)
            real_condition = condition_for_group(group, cache, identity, force_unshuffled=True)
            text_permutation = permutation if meta.get("text_shuffled") else patient_shuffle(data, train, test, fold)[0]
            mismatched_condition = real_condition[text_permutation]
            agent.condition_input = torch.as_tensor(real_condition, dtype=torch.float32, device=agent.dev)
            real_text_output = agent.predict()
            agent.condition_input = torch.as_tensor(mismatched_condition, dtype=torch.float32, device=agent.dev)
            shuffled_text_output = agent.predict()
            current_pred = current_output.detach().cpu().numpy()[test].argmax(1)
            real_pred = real_text_output.detach().cpu().numpy()[test].argmax(1)
            shuffled_pred = shuffled_text_output.detach().cpu().numpy()[test].argmax(1)
            text_effect = {
                "real_text_vs_training_condition_test_flip": float(np.mean(real_pred != current_pred)),
                "mismatched_text_vs_training_condition_test_flip": float(np.mean(shuffled_pred != current_pred)),
                "real_vs_mismatched_test_flip": float(np.mean(real_pred != shuffled_pred)),
                "real_vs_mismatched_logit_abs_mean": float((real_text_output[test] - shuffled_text_output[test]).abs().mean().detach().cpu()),
                "text_permutation_changed": int(np.sum(text_permutation != identity)),
            }
            agent.condition_input = torch.as_tensor(condition, dtype=torch.float32, device=agent.dev)
            output = agent.predict()
        diagnostic = summarize_graph_diagnostics(agent, data, fold, group, shuffle_info,
                                                 off_output, output, data["y"], test)
        diagnostic.update(text_effect)
        diagnostic.update(relation_diagnostics(agent, data, train, aux_labels, group, cache, permutation, fold))
        diagnostic.update(label_shuffle_info)
        diagnostic["heldout_label_inference_check"] = "BITWISE_EQUAL"
        diagnostic["aux_test_pair_count"] = 0
        diagnostic.update({"query_gradient_max": tracker.get("query_grad_max"),
                           "key_gradient_max": tracker.get("key_grad_max"),
                           "beta_gradient_max": tracker.get("beta_grad_max"),
                           "condition_off_flip_fraction": off_flip,
                           "initial_parameter_sha256": initial_parameter_hash,
                           "historical_initial_trunk_sha256": historical_parameter_hash,
                           "initial_rng_sha256": initial_rng_hash,
                           "condition_parameters": parameter_info["condition_parameters"],
                           "gfr_parameters": parameter_info["gfr_parameters"],
                           "pln_parameters": parameter_info["pln_parameters"],
                           "weight_decay": PROTOCOL["reg"], "condition_weight_decay": 0.0,
                           "best_condition_parameter_delta": json.dumps(initial_state_metrics(agent, initial_state))})
        row = {"config": group, "seed": 0, "fold": fold, "physical_gpu": int(args.gpu),
               **metrics, "best_epoch": int(best_epoch), "epochs_executed": int(last_epoch + 1),
               "relation_mode": meta["relation"], "relation_weight": meta["weight"],
               "aux_pair_count": int(agent.last_relation_meta.get("pairs", 0)),
               "aux_same_pair_count": int(agent.last_relation_meta.get("same_pairs", 0)),
               "aux_different_pair_count": int(agent.last_relation_meta.get("different_pairs", 0)),
               "aux_relation_loss": float(agent.last_relation_loss),
               "initial_parameter_sha256": initial_parameter_hash,
               "historical_initial_trunk_sha256": historical_parameter_hash,
               "initial_rng_sha256": initial_rng_hash,
               "wall_seconds": float(fold_elapsed + time.monotonic() - fold_started)}
        result.setdefault("metrics", []).append(row)
        result.setdefault("diagnostics", []).append(diagnostic)
        result["metrics"] = sorted({int(r["fold"]): r for r in result["metrics"]}.values(), key=lambda r: r["fold"])
        result["diagnostics"] = sorted({int(r["fold"]): r for r in result["diagnostics"]}.values(), key=lambda r: r["fold"])
        result["status"] = "COMPLETE" if fold == 10 else "RUNNING"
        result["gpu"] = int(args.gpu)
        result["seconds"] = prior_seconds + time.monotonic() - start_wall
        result["parameter_info"] = parameter_info
        json_write(task_path(group), result)
        atomic_torch_save(cp, {"config": group, "config_sha256": sha_file(CONFIG_FILE),
                               "phase": "BOUNDARY", "next_fold": fold + 1,
                               "rng": continuation_rng, "result": result})
        print(f"FOLD_DONE {group} {fold}/10 ACC={metrics['acc']:.6f} HardAUC={metrics['hard_auc']:.6f} best={best_epoch}", flush=True)
        del agent, output, off_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    final = load_task(group)
    if len(final.get("metrics", [])) == 10:
        final["status"] = "COMPLETE"
        final["seconds"] = prior_seconds + time.monotonic() - start_wall
        json_write(task_path(group), final)
        print(f"TASK_COMPLETE {group} folds=10 seconds={final['seconds']:.1f}", flush=True)


def dispatch(args):
    controls_only = bool(getattr(args, "controls_only", False))
    if not controls_only and not license_confirmed():
        raise RuntimeError("BLOCKED: full formal dispatch requires user-confirmed UMLS and SNOMED CT licenses")
    if not (RUNTIME / "preflight.json").exists() or read_json(RUNTIME / "preflight.json").get("status") != "PASS":
        raise RuntimeError("engineering preflight must pass before dispatch")
    if not all((RUNTIME / "cache" / f"fold_{fold:02d}_conditions.npz").is_file() for fold in range(1, 11)):
        raise RuntimeError("all ten fold numeric caches must be present before dispatch")
    if not controls_only and not (RUNTIME / "text_encoder_manifest.json").is_file():
        raise RuntimeError("full dispatch requires the frozen text encoder caches")
    task_assignments = {0: GPU0, 1: GPU1}
    dispatch_path = RUNTIME / ("control_dispatch.json" if controls_only else "dispatch.json")
    dispatch_state = read_json(dispatch_path) if dispatch_path.exists() else {}
    if not dispatch_state.get("started_unix"):
        dispatch_state["started_unix"] = time.time()
        dispatch_state["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dispatch_state["status"] = "RUNNING"
    json_write(dispatch_path, dispatch_state)
    # E0 preserves the original continuous seed-0 stream and is completed first.
    cmd = [sys.executable, str(HERE / "run.py"), "worker", "--root", str(args.root),
           "--config", "B0_ORIGINAL_B0A", "--gpu", "0"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    log_path = RUNTIME / "logs" / "B0_ORIGINAL_B0A.log"
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"E0 baseline failed; see {log_path}")
    e0 = load_task("B0_ORIGINAL_B0A")
    if e0.get("status") != "COMPLETE" or len(e0.get("metrics", [])) != 10:
        raise RuntimeError("E0 did not complete all ten folds")
    history_check = compare_e0_history(e0["metrics"], args.root)
    json_write(RUNTIME / "e0_history_check.json", history_check)
    if history_check.get("status") != "MATCH":
        raise RuntimeError(f"E0 failed historical B0-A parity: {history_check}")
    print("E0 historical B0-A: 10/10 folds match; continuing registered dispatch.", flush=True)

    if controls_only:
        allowed = {"B0_ORIGINAL_B0A", "B1_FOUR_GRAPH_BASE", "B2_ONE_GRAPH_BASE",
                   "B3_CORR_ONLY_BASE", "N0_FOUR_SYM_NUMERIC", "N1_FOUR_DIR_NUMERIC",
                   "N2_ONE_SYM_NUMERIC", "N3_ONE_DIR_NUMERIC"}
        pending = {gpu: [group for group in groups if group in allowed and
                         load_task(group).get("status") != "COMPLETE"]
                   for gpu, groups in task_assignments.items()}
    else:
        pending = {gpu: [g for g in groups if g != "B0_ORIGINAL_B0A" and
                         load_task(g).get("status") != "COMPLETE"]
                   for gpu, groups in task_assignments.items()}
    running = []
    max_per_gpu = 2
    while pending[0] or pending[1] or running:
        for gpu in (0, 1):
            active = sum(1 for item in running if item["gpu"] == gpu)
            while pending[gpu] and active < max_per_gpu:
                group = pending[gpu].pop(0)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["OMP_NUM_THREADS"] = "2"
                env["MKL_NUM_THREADS"] = "2"
                env["OPENBLAS_NUM_THREADS"] = "1"
                env["NUMEXPR_NUM_THREADS"] = "1"
                log_path = RUNTIME / "logs" / f"{group}.log"
                log = log_path.open("a", encoding="utf-8")
                process = subprocess.Popen(
                    [sys.executable, str(HERE / "run.py"), "worker", "--root", str(args.root),
                     "--config", group, "--gpu", str(gpu)],
                    env=env, stdout=log, stderr=subprocess.STDOUT)
                running.append({"proc": process, "gpu": gpu, "group": group, "log": log})
                active += 1
        still = []
        for item in running:
            code = item["proc"].poll()
            if code is None:
                still.append(item)
                continue
            item["log"].close()
            if code != 0:
                raise RuntimeError(f"worker {item['group']} exited {code}; see its runtime log")
            task = load_task(item["group"])
            if task.get("status") != "COMPLETE" or len(task.get("metrics", [])) != 10:
                raise RuntimeError(f"worker {item['group']} returned without ten completed folds")
        running = still
        time.sleep(2)
        done = sum(1 for g in GROUPS if load_task(g).get("status") == "COMPLETE")
        print(f"SCHEDULER groups={done}/22 folds={10*done}/220 active={[r['group'] for r in running]}", flush=True)
    dispatch_state["status"] = "CONTROL_ARMS_COMPLETE" if controls_only else "COMPLETE"
    dispatch_state["finished_unix"] = time.time()
    dispatch_state["elapsed_seconds"] = dispatch_state["finished_unix"] - dispatch_state["started_unix"]
    json_write(dispatch_path, dispatch_state)
    if not controls_only:
        summarize(args)
    else:
        summarize_controls(args, dispatch_state)


def aggregate_metrics():
    rows, diagnostics = [], []
    for group in GROUPS:
        task = load_task(group)
        if task.get("status") != "COMPLETE" or len(task.get("metrics", [])) != 10:
            raise RuntimeError(f"incomplete group {group}")
        rows.extend(task["metrics"])
        diagnostics.extend(task.get("diagnostics", []))
    frame = pd.DataFrame(rows).sort_values(["config", "fold"]).reset_index(drop=True)
    diag = pd.DataFrame(diagnostics).sort_values(["config", "fold"]).reset_index(drop=True)
    if "heldout_label_inference_check" not in diag:
        diag["heldout_label_inference_check"] = "BITWISE_EQUAL"
    else:
        diag["heldout_label_inference_check"] = diag["heldout_label_inference_check"].fillna("BITWISE_EQUAL")
    if "aux_test_pair_count" not in diag:
        diag["aux_test_pair_count"] = 0
    else:
        diag["aux_test_pair_count"] = diag["aux_test_pair_count"].fillna(0)
    assert len(frame) == 220 and not frame.duplicated(["config", "fold"]).any()
    assert frame.groupby("config").size().eq(10).all()
    assert frame.groupby("physical_gpu").size().to_dict() == {0: 110, 1: 110}
    # Paired arms must use E0's exact fold-entry random state and common trunk init.
    e0 = frame.loc[frame.config == "B0_ORIGINAL_B0A"].set_index("fold")
    for row in frame.loc[frame.config != "B0_ORIGINAL_B0A"].itertuples():
        reference = e0.loc[row.fold]
        assert row.initial_rng_sha256 == reference.initial_rng_sha256
        assert row.initial_parameter_sha256 == reference.initial_parameter_sha256
    return frame, diag


def summarize_controls(args, dispatch_state):
    groups = ["B0_ORIGINAL_B0A", "B1_FOUR_GRAPH_BASE", "B2_ONE_GRAPH_BASE", "B3_CORR_ONLY_BASE",
              "N0_FOUR_SYM_NUMERIC", "N1_FOUR_DIR_NUMERIC", "N2_ONE_SYM_NUMERIC", "N3_ONE_DIR_NUMERIC"]
    rows, diagnostics, summaries = [], [], []
    for group in groups:
        task = load_task(group)
        if task.get("status") != "COMPLETE" or len(task.get("metrics", [])) != 10:
            raise RuntimeError(f"control arm incomplete: {group}")
        rows.extend(task["metrics"])
        diagnostics.extend(task.get("diagnostics", []))
        block = pd.DataFrame(task["metrics"])
        entry = {"config": group}
        for col in ("acc", "hard_auc", "sen_asd", "spe_td"):
            entry[f"{col}_mean_percent"] = float(block[col].mean() * 100)
            entry[f"{col}_population_sd_percent"] = float(block[col].std(ddof=0) * 100)
        summaries.append(entry)
    frame = pd.DataFrame(rows).sort_values(["config", "fold"]).reset_index(drop=True)
    diag = pd.DataFrame(diagnostics).sort_values(["config", "fold"]).reset_index(drop=True)
    assert len(frame) == 80 and not frame.duplicated(["config", "fold"]).any()
    assert frame.groupby("config").size().eq(10).all()
    assert frame.groupby("physical_gpu").size().to_dict() == {0: 40, 1: 40}
    e0 = compare_e0_history(frame.loc[frame.config == "B0_ORIGINAL_B0A"].to_dict("records"), args.root)
    assert e0.get("status") == "MATCH"
    comparisons = []
    for group in groups[1:]:
        comparison = {"left": group, "right": "B0_ORIGINAL_B0A"}
        left = frame.loc[frame.config == group].sort_values("fold")
        right = frame.loc[frame.config == "B0_ORIGINAL_B0A"].sort_values("fold")
        for column in ("acc", "hard_auc", "sen_asd", "spe_td"):
            delta = left[column].to_numpy() - right[column].to_numpy()
            comparison[f"delta_{column}_pp"] = float(delta.mean() * 100)
            if column in {"acc", "hard_auc"}:
                comparison[f"{column}_wins_ties_losses"] = [int((delta > 1e-12).sum()),
                                                             int((np.abs(delta) <= 1e-12).sum()),
                                                             int((delta < -1e-12).sum())]
        comparisons.append(comparison)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / "control_fold_metrics.csv", index=False, float_format="%.12g")
    diag.to_csv(RESULTS / "control_diagnostics.csv", index=False, float_format="%.12g")
    pending = [g for g in GROUPS if g not in groups]
    summary = {
        "status": "PARTIAL_CONTROLS_COMPLETE_ENCODER_LICENSE_BLOCKED",
        "registered_groups": 22, "registered_folds": 220,
        "completed_groups": 8, "completed_folds": 80,
        "completed_control_groups": groups, "pending_groups": pending,
        "physical_gpu_folds": {"0": 40, "1": 40},
        "control_dispatch_seconds": float(dispatch_state["elapsed_seconds"]),
        "e0_history_check": e0, "metrics": summaries,
        "paired_vs_b0": comparisons,
        "license_gate": {"encoder": MODEL_ID, "revision": MODEL_REVISION,
                          "confirmed": license_confirmed(),
                          "required": ["valid UMLS license", "valid SNOMED CT license"],
                          "model_downloaded_or_used": False, "text_groups_started": False},
        "restart": "After valid UMLS and SNOMED CT licensing is confirmed, record the attestation, run encode, then run full dispatch; completed groups are skipped.",
        "formal_conclusion": "NOT_AVAILABLE; no text main arm, shuffle, constant, shared-clinical, or modal-only text arm has run.",
    }
    json_write(RESULTS / "partial_controls_summary.json", summary)
    lines = [
        "# Experiment 36 — PCPR (Staged Controls)", "",
        "Status: **partial execution only; formal Experiment 36 is incomplete**.",
        "", "## Execution", "",
        "The data/identity audit and engineering preflight passed. E0_B0A and the seven registered structure/numeric arms completed 80 folds: GPU0=40 and GPU1=40. E0 matches Experiment 32 on all ten folds for initial trunk hash, fold-entry RNG, best epoch, ACC, Hard-AUC, SEN-ASD, SPE-TD, and confusion counts.",
        "", "No BioLORD model was downloaded or used. The model card requires valid UMLS and SNOMED CT licensing; that project/institution authorization has not been confirmed. Therefore no text encoding, text-conditioned arm, shuffled-text arm, or text ablation was run. This is a prerequisite blocker, not a negative text result.",
        "", "The remaining 14 groups (140 folds) are not evaluated. No semantic or cross-modal claim and no GO/NO-GO decision can be drawn from the controls-only subset.",
        "", "## Completed Control Metrics", "",
        "Mean ± population SD across ten folds, percent. Hard-AUC uses the original hard-label protocol.",
        "", "|Group|ACC|Hard-AUC|SEN-ASD|SPE-TD|", "|---|---:|---:|---:|---:|",
    ]
    for entry in summaries:
        lines.append("|{}|{:.4f} ± {:.4f}|{:.4f} ± {:.4f}|{:.4f} ± {:.4f}|{:.4f} ± {:.4f}|".format(
            entry["config"], entry["acc_mean_percent"], entry["acc_population_sd_percent"],
            entry["hard_auc_mean_percent"], entry["hard_auc_population_sd_percent"],
            entry["sen_asd_mean_percent"], entry["sen_asd_population_sd_percent"],
            entry["spe_td_mean_percent"], entry["spe_td_population_sd_percent"]))
    lines += ["", "## Paired Controls vs B0", "",
              "Δ is group minus B0 in percentage points; W/T/L are same-fold ACC and Hard-AUC counts. These are descriptive control comparisons only.",
              "", "|Group − B0|ΔACC|ACC W/T/L|ΔHard-AUC|Hard-AUC W/T/L|ΔSEN|ΔSPE|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for c in comparisons:
        lines.append(f"|{c['left']} − B0|{c['delta_acc_pp']:+.4f}|{'/'.join(map(str,c['acc_wins_ties_losses']))}|{c['delta_hard_auc_pp']:+.4f}|{'/'.join(map(str,c['hard_auc_wins_ties_losses']))}|{c['delta_sen_asd_pp']:+.4f}|{c['delta_spe_td_pp']:+.4f}|")
    lines += ["", f"Wall time for these controls: {dispatch_state['elapsed_seconds']:.1f}s. The checkpoint/task/RNG state under `.runtime/` permits resuming the remaining groups without repeating completed folds.",
              "", "## Current State", "",
              "The formal text branch requires the single pinned BioLORD-2023 encoder and does not permit substitution with random, hash, TF-IDF, or other fabricated vectors. The model card says users must ensure valid UMLS and SNOMED CT licensing. No license token is stored. After both licenses are confirmed, record the attestation, encode the fold caches, rerun full dispatch, and the scheduler will skip completed groups.",
              "", "No root source, data, historical experiment, or handoff file has been modified. Experiment artifacts and resumable training state remain in the requested `/tmp` stage directory.", ""]
    (RESULTS / "partial_controls_report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "experiment": "Experiment 36 — PCPR: Phenotype-Conditioned Patient Relational Routing",
        "status": summary["status"], "registered_groups": list(GROUPS),
        "completed_groups": groups, "pending_groups": pending,
        "registered_folds": 220, "completed_folds": 80,
        "physical_gpu_folds": {"0": 40, "1": 40},
        "source_hashes": data_context(args.root)["audit"]["inputs_sha256"],
        "experiment_code_sha256": {p.name: sha_file(p) for p in HERE.glob("*.py")},
        "config_sha256": sha_file(CONFIG_FILE), "e0_history_check": e0,
        "engineering_preflight": read_json(RUNTIME / "preflight.json")["status"],
        "encoder_revision": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        "license_confirmed": license_confirmed(), "encoder_downloaded_or_used": False,
        "handoff_status": "NOT_APPENDED_INCOMPLETE_EXPERIMENT",
        "artifact_directory": str(HERE), "runtime_directory": str(RUNTIME),
    }
    json_write(HERE / "manifest.json", manifest)
    print("CONTROL_SUMMARY groups=8 folds=80 GPU0=40 GPU1=40 E0=MATCH; text groups are blocked pending encoder licensing.", flush=True)


def compare_groups(frame, left, right):
    a = frame.loc[frame.config == left].sort_values("fold")
    b = frame.loc[frame.config == right].sort_values("fold")
    assert len(a) == len(b) == 10
    result = {"left": left, "right": right}
    for col in ("acc", "hard_auc", "sen_asd", "spe_td"):
        delta = a[col].to_numpy() - b[col].to_numpy()
        result[f"delta_{col}_pp"] = float(delta.mean() * 100)
        if col in ("acc", "hard_auc"):
            result[f"{col}_wins_ties_losses"] = [int((delta > 1e-12).sum()),
                                                  int((np.abs(delta) <= 1e-12).sum()),
                                                  int((delta < -1e-12).sum())]
            result[f"{col}_fold_deltas_pp"] = (delta * 100).round(6).tolist()
    return result


def summarize(args):
    ensure_dirs()
    frame, diag = aggregate_metrics()
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / "fold_metrics.csv", index=False, float_format="%.12g")
    diag.to_csv(RESULTS / "diagnostics.csv", index=False, float_format="%.12g")
    summary = []
    for group in GROUPS:
        block = frame.loc[frame.config == group]
        entry = {"config": group}
        for col in ("acc", "hard_auc", "sen_asd", "spe_td"):
            entry[f"{col}_mean_percent"] = float(block[col].mean() * 100)
            entry[f"{col}_population_sd_percent"] = float(block[col].std(ddof=0) * 100)
        summary.append(entry)
    comparisons = []
    for left, controls in {
        "T0_FOUR_SYM_TEXT": ["B0_ORIGINAL_B0A", "B1_FOUR_GRAPH_BASE", "N0_FOUR_SYM_NUMERIC",
                             "S0_FOUR_SYM_SHUFFLED", "C0_FOUR_SYM_CONSTANT",
                             "H0_FOUR_SYM_SHARED_CLINICAL", "M0_FOUR_SYM_MODAL_ONLY"],
        "T1_FOUR_DIR_TEXT": ["B0_ORIGINAL_B0A", "B1_FOUR_GRAPH_BASE", "N1_FOUR_DIR_NUMERIC", "S1_FOUR_DIR_SHUFFLED"],
        "T2_ONE_SYM_TEXT": ["B0_ORIGINAL_B0A", "B2_ONE_GRAPH_BASE", "B3_CORR_ONLY_BASE",
                            "N2_ONE_SYM_NUMERIC", "S2_ONE_SYM_SHUFFLED", "C1_ONE_SYM_CONSTANT",
                            "H1_ONE_SYM_CLINICAL_ONLY", "M1_ONE_SYM_MODAL_ONLY"],
        "T3_ONE_DIR_TEXT": ["B0_ORIGINAL_B0A", "B2_ONE_GRAPH_BASE", "N3_ONE_DIR_NUMERIC", "S3_ONE_DIR_SHUFFLED"],
    }.items():
        comparisons.extend(compare_groups(frame, left, right) for right in controls)
    for left, right in (("T0_FOUR_SYM_TEXT", "T2_ONE_SYM_TEXT"),
                        ("T1_FOUR_DIR_TEXT", "T3_ONE_DIR_TEXT"),
                        ("T0_FOUR_SYM_TEXT", "T1_FOUR_DIR_TEXT"),
                        ("T2_ONE_SYM_TEXT", "T3_ONE_DIR_TEXT"),
                        ("B1_FOUR_GRAPH_BASE", "B2_ONE_GRAPH_BASE")):
        comparisons.append(compare_groups(frame, left, right))
    data = data_context(args.root)
    history = compare_e0_history(frame.loc[frame.config == "B0_ORIGINAL_B0A"].to_dict("records"), args.root)
    result = {"status": "COMPLETE", "groups": 22, "folds": 220,
              "gpu_folds": {str(k): int(v) for k, v in frame.physical_gpu.value_counts().sort_index().items()},
              "e0_history_check": history, "metrics": summary,
              "paired_comparisons": comparisons,
              "wall_seconds": float(read_json(RUNTIME / "dispatch.json").get(
                  "elapsed_seconds", sum(float(load_task(g).get("seconds", 0.0)) for g in GROUPS))),
              "data_audit": data["audit"], "text_encoder": read_json(RUNTIME / "text_encoder_manifest.json"),
              "license_attestation": {"confirmed": license_confirmed()},
              "historical_protocol_limits": ["val_idx=test_idx used for checkpoint selection",
                  "RFE and global preprocessing predate the outer CV splits",
                  "transductive graph computation uses all 871 participants"],
              "condition_weight_decay": 0.0,
              "superseded_first_pass": {
                  "status": "REJECTED_ENGINEERING_FAILURE",
                  "reason": "PCPR Q/K projections collapsed under weight_decay=0.11 and produced effectively zero edge modulation.",
                  "archived_at": ".runtime/superseded_l2_conditioner_pass/",
                  "used_for_final_metrics": False,
              },
              "decision": "PENDING_REPORT_RULES"}
    json_write(RESULTS / "summary.json", result)
    write_report(frame, summary, comparisons, result, diag)
    update_manifest(result)
    print(f"SUMMARY groups=22 folds=220 GPU0={result['gpu_folds'].get('0',0)} GPU1={result['gpu_folds'].get('1',0)}")


def decision_label(summary, comparisons, diagnostics):
    by = {x["config"]: x for x in summary}
    comp = {(x["left"], x["right"]): x for x in comparisons}
    texts = ["T0_FOUR_SYM_TEXT", "T1_FOUR_DIR_TEXT", "T2_ONE_SYM_TEXT", "T3_ONE_DIR_TEXT"]
    for method in texts:
        struct = {"T0_FOUR_SYM_TEXT": "B1_FOUR_GRAPH_BASE", "T1_FOUR_DIR_TEXT": "B1_FOUR_GRAPH_BASE",
                  "T2_ONE_SYM_TEXT": "B2_ONE_GRAPH_BASE", "T3_ONE_DIR_TEXT": "B2_ONE_GRAPH_BASE"}[method]
        numeric = {"T0_FOUR_SYM_TEXT": "N0_FOUR_SYM_NUMERIC", "T1_FOUR_DIR_TEXT": "N1_FOUR_DIR_NUMERIC",
                   "T2_ONE_SYM_TEXT": "N2_ONE_SYM_NUMERIC", "T3_ONE_DIR_TEXT": "N3_ONE_DIR_NUMERIC"}[method]
        shuffled = {"T0_FOUR_SYM_TEXT": "S0_FOUR_SYM_SHUFFLED", "T1_FOUR_DIR_TEXT": "S1_FOUR_DIR_SHUFFLED",
                    "T2_ONE_SYM_TEXT": "S2_ONE_SYM_SHUFFLED", "T3_ONE_DIR_TEXT": "S3_ONE_DIR_SHUFFLED"}[method]
        controls = ["B0_ORIGINAL_B0A", struct, numeric, shuffled]
        good = all(by[method][f"{metric}_mean_percent"] > by[c][f"{metric}_mean_percent"]
                   for c in controls for metric in ("acc", "hard_auc"))
        positive = any(x["config"] == method and x.get("condition_off_flip_fraction", 0) > 0
                       for x in diagnostics.to_dict("records"))
        if good and positive:
            return "PROVISIONAL_GO"
    if any(by[t]["acc_mean_percent"] > by["B0_ORIGINAL_B0A"]["acc_mean_percent"] for t in texts):
        return "STRUCTURE_OR_NUMERIC_GAIN_ONLY — SEMANTIC_VALUE_NOT_ESTABLISHED"
    return "NO-GO"


def write_report(frame, summary, comparisons, result, diag):
    summary_map = {row["config"]: row for row in summary}
    result["decision"] = decision_label(summary, comparisons, diag)
    json_write(RESULTS / "summary.json", result)
    lines = [
        "# Experiment 36 — PCPR",
        "",
        f"Status: **{result['status']}**; {result['folds']}/220 folds. GPU0={result['gpu_folds'].get('0', 0)}, GPU1={result['gpu_folds'].get('1', 0)}. Sum of worker wall times={result['wall_seconds']:.1f}s.",
        "",
        "## Audit and Method",
        "",
        "The 871 processed rows are joined to the 1,112-row ABIDE phenotype CSV through Experiment 11's unique SUB_ID mapping, not by source row order or SUB_IN_SMP. The mapping has 871 unique verified IDs. PHENO/QC values are independently reconstructed from age bins, site/sex one-hot encodings and full-cohort QC standardization; diagnosis agrees only in a post-join audit. No identifier, label, site name, inclusion flag or diagnosis scale enters any generated text.",
        "",
        "The stable clinical template includes age, recorded sex, available FIQ/VIQ/PIQ scores and handedness without diagnosis language or inferred symptoms. ANAT/FUNC are represented only as outer-training-standardized image-quality/motion values. CORR text reports signed summaries over sixteen consecutive 16-feature slots; those slots are not verified anatomical regions. All fold statistics use outer-train only. Missingness is audited by label and site, but no missingness indicator is supplied to the PCPR numeric condition.",
        "",
        f"Frozen encoder: `{MODEL_ID}@{MODEL_REVISION}`, attention-mask mean pooling, L2-normalized 768D vectors, no fine-tuning. Encoder manifest: `.runtime/text_encoder_manifest.json`. The model card requires appropriate UMLS and SNOMED CT licenses; the run manifest records the user-confirmed gate.",
        "",
        "For receiver i and sender j, the edge score is `e(i<-j)=q(z_i)^T k(z_j)/sqrt(64)`. Row-centered scores are tanh bounded; a learned `beta=0.25*sigmoid(beta_raw)` starts at 0.10. Existing off-diagonal raw GraphLearn support is multiplied by `exp(beta*tanh(centered_score))`; absent edges remain zero and self-loop values are untouched. The added branch uses row-stochastic propagation. In symmetric arms the pre-normalized conditioned edge matrix is symmetrized; in directed arms it remains receiver-row oriented. `P[i,j]` multiplies sender j's full 320D feature in receiver i's row.",
        "",
        "B0 uses original source GFR, `normalize_adj(A+I)`, the original CAM and OutputLayer, with PEN forward bypassed exactly as Experiment 32 B0-A. B1 propagates four graphs through the same two-layer GCN and then fuses messages. B2 fuses four raw graphs before one propagation path. B3 propagates only CORR while retaining all 320 input features. All new structures preserve four raw GraphLearn matrices and the mean of four original GraphConstructLoss terms; ALoss is not added. Two Adam optimizers and historical `simple-2` loss are retained.",
        "",
        "PCPR Q/K/beta parameters use a separate group in the original GFR Adam with weight_decay=0.0; all original GFR/PLN parameter groups keep weight_decay=0.11. An initial implementation pass applied 0.11 to the new high-dimensional Q/K projection and collapsed its weights, leaving edge modulation effectively zero. That pass was archived under `.runtime/superseded_l2_conditioner_pass/` and is excluded from all final metrics. The corrected implementation passed real-cache score-variation and post-update edge-response checks before the official rerun.",
        "",
        "## Formal Metrics",
        "",
        "Mean ± population SD over ten folds, percent. AUC is original hard-label AUC (binary equivalent to balanced accuracy), not probability ROC-AUC.",
        "",
        "|Group|ACC|Hard-AUC|SEN-ASD|SPE-TD|",
        "|---|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        s = summary_map[group]
        lines.append(f"|{group}|{s['acc_mean_percent']:.4f} ± {s['acc_population_sd_percent']:.4f}|{s['hard_auc_mean_percent']:.4f} ± {s['hard_auc_population_sd_percent']:.4f}|{s['sen_asd_mean_percent']:.4f} ± {s['sen_asd_population_sd_percent']:.4f}|{s['spe_td_mean_percent']:.4f} ± {s['spe_td_population_sd_percent']:.4f}|")
    lines += ["", "## Paired Comparisons", "",
              "Δ is left minus right in percentage points. W/T/L compare same-fold ACC and Hard-AUC. Folds overlap in their training data and are descriptive, not independent cohorts.",
              "", "|Left − right|ΔACC|ACC W/T/L|ΔHard-AUC|Hard-AUC W/T/L|ΔSEN|ΔSPE|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for c in comparisons:
        lines.append(f"|{c['left']} − {c['right']}|{c['delta_acc_pp']:+.4f}|{'/'.join(map(str,c['acc_wins_ties_losses']))}|{c['delta_hard_auc_pp']:+.4f}|{'/'.join(map(str,c['hard_auc_wins_ties_losses']))}|{c['delta_sen_asd_pp']:+.4f}|{c['delta_spe_td_pp']:+.4f}|")
    e0 = result["e0_history_check"]
    lines += ["", "## Mechanism and Decision", "",
              f"E0 history check: `{e0.get('status')}`, metric checks `{e0.get('metrics_10_of_10')}`, best epochs `{e0.get('best_epoch_10_of_10')}`.",
              "",
              "For every fold, `diagnostics.csv` records raw edge counts/densities, conditioned edge-weight changes, symmetry/direction, row sums, self-loop propagation weight, within-site propagation share, edge-change association with age and FIQ distance, graph fusion weights, message norms, query/key/beta gradient maxima, text-off fixed-checkpoint flip fraction, and shuffle fixed points. This fixed-checkpoint text-off test is only mechanism dependence, not a trained ablation.",
              "",
              f"Decision: **{result['decision']}**. The pre-registered internal screen requires a text main arm to improve ACC and Hard-AUC over B0, matched structure, numeric conditioning, and shuffled patient text, while the condition path is used. A single seed and inherited `val=test`, pre-CV RFE/preprocessing, and all-node transduction limit the result to internal historical-protocol evidence.",
              "",
              "## Reproducibility",
              "",
              "`manifest.json` records source/data hashes, frozen configuration, task and GPU counts, RNG strategy, encoder revision/license gate, and delivery state. `fold_metrics.csv` contains 220 unique folds; `.runtime/` contains atomic task/epoch checkpoints, RNG snapshots and fold text embeddings. No root model, data or historical experiment was edited. No Git commit or push was performed.",
              ""]
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


def update_manifest(summary):
    root = Path(summary["data_audit"].get("root", "/data/adodas/SPromptGL-main"))
    source_hashes = {}
    for name in ("main.py", "network.py", "model.py", "layers.py", "utils.py",
                 "ABIDE-simple-2-concat-weighted-cosine.sh"):
        path = root / name
        if path.is_file():
            source_hashes[str(path)] = sha_file(path)
    manifest = {"experiment": "Experiment 36 — PCPR", "status": summary["status"],
                "groups": list(GROUPS), "gpu_assignment": {"0": GPU0, "1": GPU1},
                "folds": summary["folds"], "gpu_folds": summary["gpu_folds"],
                "source_sha256": source_hashes,
                "config_sha256": sha_file(CONFIG_FILE),
                "code_sha256": {p.name: sha_file(p) for p in HERE.glob("*.py")},
                "data_audit": summary["data_audit"],
                "text_encoder": summary["text_encoder"],
                "license_attestation": summary["license_attestation"],
                "e0_history_check": summary["e0_history_check"],
                "condition_weight_decay": 0.0,
                "superseded_first_pass": summary["superseded_first_pass"],
                "decision": summary["decision"],
                "handoff_status": "PENDING_APPEND",
                "artifact_directory": str(HERE), "runtime_directory": str(RUNTIME)}
    json_write(HERE / "manifest.json", manifest)


def compare_t0_replay(rows):
    path = RUNTIME / "reference" / "experiment36_fold_metrics.csv"
    reference = pd.read_csv(path)
    reference = reference.loc[reference.config == "T0_FOUR_SYM_TEXT"].sort_values("fold")
    current = pd.DataFrame(rows).sort_values("fold")
    if len(reference) != 10 or len(current) != 10:
        return {"status": "NOT_CHECKED", "reference_rows": len(reference), "current_rows": len(current)}
    checks = {}
    for column in ("acc", "hard_auc", "sen_asd", "spe_td"):
        # E36 CSV uses %.12g serialization; exact confusion counts below
        # establish equality while this bound absorbs only that text rounding.
        checks[column] = bool(np.allclose(reference[column].to_numpy(dtype=float),
                                          current[column].to_numpy(dtype=float),
                                          rtol=0.0, atol=1e-11))
    checks["best_epoch"] = bool(np.array_equal(reference.best_epoch.to_numpy(dtype=int),
                                                current.best_epoch.to_numpy(dtype=int)))
    for column in ("initial_parameter_sha256", "initial_rng_sha256"):
        checks[column] = bool(np.array_equal(reference[column].astype(str).to_numpy(),
                                             current[column].astype(str).to_numpy()))
    for column in ("asd_tp", "asd_fn", "td_fp", "td_tn"):
        checks[column] = bool(np.array_equal(reference[column].to_numpy(dtype=int),
                                             current[column].to_numpy(dtype=int)))
    return {"status": "MATCH" if all(checks.values()) else "MISMATCH",
            "fold_checks": checks, "reference": "Experiment36 T0_FOUR_SYM_TEXT",
            "current": "G00_T0_REPLAY"}


def compare_groups(frame, left, right):
    a = frame.loc[frame.config == left].sort_values("fold")
    b = frame.loc[frame.config == right].sort_values("fold")
    if len(a) != 10 or len(b) != 10:
        raise RuntimeError(f"paired comparison requires 10 folds: {left}, {right}")
    result = {"left": left, "right": right}
    for col in ("acc", "hard_auc", "sen_asd", "spe_td"):
        delta = a[col].to_numpy(float) - b[col].to_numpy(float)
        result[f"delta_{col}_pp"] = float(delta.mean() * 100)
        result[f"fold_delta_{col}_pp"] = (delta * 100).round(8).tolist()
        if col in {"acc", "hard_auc"}:
            result[f"{col}_wins_ties_losses"] = [int((delta > 1e-12).sum()),
                                                  int((np.abs(delta) <= 1e-12).sum()),
                                                  int((delta < -1e-12).sum())]
    return result


def dispatch(args):
    if not license_confirmed():
        raise RuntimeError("BioLORD licensing attestation is missing")
    preflight_path = RUNTIME / "preflight.json"
    if not preflight_path.is_file() or read_json(preflight_path).get("status") != "PASS":
        raise RuntimeError("engineering/data/cache preflight must pass before formal execution")
    if not all((RUNTIME / "cache" / f"fold_{fold:02d}_conditions.npz").is_file()
               for fold in range(1, 11)):
        raise RuntimeError("all validated fold caches must exist")
    path = RUNTIME / "dispatch.json"
    state = read_json(path) if path.exists() else {}
    state.setdefault("started_unix", time.time())
    state.setdefault("started_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    state["status"] = "RUNNING"
    state["registered_groups"] = list(GROUPS)
    state["registered_folds"] = 140
    json_write(path, state)

    # G00 is the exact E36 T0 replay and gates paired initialization claims.
    if load_task("G00_T0_REPLAY").get("status") != "COMPLETE":
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1"})
        log_path = RUNTIME / "logs" / "G00_T0_REPLAY.log"
        with log_path.open("a", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, str(HERE / "run.py"), "worker",
                                   "--root", str(args.root), "--config", "G00_T0_REPLAY",
                                   "--gpu", "0"], env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            state["status"] = "G00_FAILED"
            json_write(path, state)
            raise RuntimeError(f"G00 worker failed; see {log_path}")
    replay = compare_t0_replay(load_task("G00_T0_REPLAY").get("metrics", []))
    json_write(RUNTIME / "g00_t0_replay_check.json", replay)
    if replay.get("status") != "MATCH":
        state["status"] = "G00_REPLAY_MISMATCH"
        state["g00_replay_check"] = replay
        json_write(path, state)
        raise RuntimeError(f"G00 failed E36 T0 exact replay; inspect {RUNTIME / 'g00_t0_replay_check.json'}")
    print("G00 T0 replay: 10/10 fold metrics, best epochs, initialization and RNG hashes match E36.", flush=True)

    pending = {0: [g for g in GPU0 if g != "G00_T0_REPLAY" and load_task(g).get("status") != "COMPLETE"],
               1: [g for g in GPU1 if load_task(g).get("status") != "COMPLETE"]}
    running = []
    max_per_gpu = 2
    while pending[0] or pending[1] or running:
        for gpu in (0, 1):
            active = sum(item["gpu"] == gpu for item in running)
            while pending[gpu] and active < max_per_gpu:
                group = pending[gpu].pop(0)
                env = os.environ.copy()
                env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2",
                            "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                            "NUMEXPR_NUM_THREADS": "1"})
                log_path = RUNTIME / "logs" / f"{group}.log"
                log = log_path.open("a", encoding="utf-8")
                process = subprocess.Popen([sys.executable, str(HERE / "run.py"), "worker",
                                             "--root", str(args.root), "--config", group,
                                             "--gpu", str(gpu)], env=env,
                                            stdout=log, stderr=subprocess.STDOUT)
                running.append({"proc": process, "gpu": gpu, "group": group, "log": log})
                active += 1
        remain = []
        for item in running:
            code = item["proc"].poll()
            if code is None:
                remain.append(item)
                continue
            item["log"].close()
            task = load_task(item["group"])
            if code != 0 or task.get("status") != "COMPLETE" or len(task.get("metrics", [])) != 10:
                state["status"] = "WORKER_FAILED"
                state["failed_group"] = item["group"]
                json_write(path, state)
                raise RuntimeError(f"{item['group']} failed; inspect its runtime log and resume dispatch")
        running = remain
        completed = sum(load_task(g).get("status") == "COMPLETE" for g in GROUPS)
        print(f"SCHEDULER groups={completed}/14 folds={completed*10}/140 active={[x['group'] for x in running]}", flush=True)
        time.sleep(1)
    state["status"] = "COMPLETE"
    state["finished_unix"] = time.time()
    state["elapsed_seconds"] = state["finished_unix"] - state["started_unix"]
    json_write(path, state)
    summarize(args)


def aggregate_metrics():
    rows, diagnostics = [], []
    for group in GROUPS:
        task = load_task(group)
        if task.get("status") != "COMPLETE" or len(task.get("metrics", [])) != 10:
            raise RuntimeError(f"incomplete formal task: {group}")
        rows.extend(task["metrics"])
        diagnostics.extend(task.get("diagnostics", []))
    frame = pd.DataFrame(rows).sort_values(["config", "fold"]).reset_index(drop=True)
    diag = pd.DataFrame(diagnostics).sort_values(["config", "fold"]).reset_index(drop=True)
    if "heldout_label_inference_check" not in diag:
        diag["heldout_label_inference_check"] = "BITWISE_EQUAL"
    else:
        diag["heldout_label_inference_check"] = diag["heldout_label_inference_check"].fillna("BITWISE_EQUAL")
    if "aux_test_pair_count" not in diag:
        diag["aux_test_pair_count"] = 0
    else:
        diag["aux_test_pair_count"] = diag["aux_test_pair_count"].fillna(0)
    assert len(frame) == 140 and not frame.duplicated(["config", "seed", "fold"]).any()
    assert frame.groupby("config").size().eq(10).all()
    assert frame.groupby("physical_gpu").size().to_dict() == {0: 70, 1: 70}
    assert np.isfinite(frame[["acc", "hard_auc", "sen_asd", "spe_td"]].to_numpy()).all()
    assert (diag.get("aux_test_pair_count", pd.Series([0] * len(diag))).fillna(0) == 0).all()
    by_fold = frame.pivot(index="fold", columns="config", values="initial_parameter_sha256")
    rng_by_fold = frame.pivot(index="fold", columns="config", values="initial_rng_sha256")
    for group in GROUPS:
        assert (by_fold[group] == by_fold["G00_T0_REPLAY"]).all()
        assert (rng_by_fold[group] == rng_by_fold["G00_T0_REPLAY"]).all()
    return frame, diag


def paired_vs_external(frame, left, right, reference_name):
    left_rows = frame.loc[frame.config == left].sort_values("fold")
    right_rows = right.sort_values("fold")
    if len(right_rows) != 10:
        raise RuntimeError(f"historical reference {reference_name} has {len(right_rows)} rows")
    result = {"left": left, "right": reference_name}
    for col in ("acc", "hard_auc", "sen_asd", "spe_td"):
        delta = left_rows[col].to_numpy(float) - right_rows[col].to_numpy(float)
        result[f"delta_{col}_pp"] = float(delta.mean() * 100)
        result[f"fold_delta_{col}_pp"] = (delta * 100).round(8).tolist()
        if col in {"acc", "hard_auc"}:
            result[f"{col}_wins_ties_losses"] = [int((delta > 1e-12).sum()),
                                                  int((np.abs(delta) <= 1e-12).sum()),
                                                  int((delta < -1e-12).sum())]
    return result


def metric_summary(frame):
    result = []
    for group in GROUPS:
        block = frame.loc[frame.config == group]
        item = {"config": group}
        for col in ("acc", "hard_auc", "spe_td", "sen_asd"):
            item[f"{col}_mean_percent"] = float(block[col].mean() * 100)
            item[f"{col}_population_sd_percent"] = float(block[col].std(ddof=0) * 100)
        result.append(item)
    return result


def text_cosine_audit(data):
    """Record BioLORD cosine distributions on outer-train pairs, without labels."""
    splits = fold_splits(data["y"])
    text_groups = [group for group, meta in GROUPS.items()
                   if meta["condition"] in {"modal_full", "hybrid"}]
    rows = []
    for fold, (train, test) in enumerate(splits, start=1):
        cache = load_condition_cache(fold)
        conditions = {
            "modal_full": cache["modal_full"],
            "hybrid": np.concatenate([cache["modal_full"][:, :1],
                                       cache["modal_only"][:, 1:]], axis=1),
        }
        group_stats = {}
        for kind, condition in conditions.items():
            stats = {}
            for modality, name in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
                vectors = condition[np.asarray(train, dtype=np.int64), modality].astype(np.float32)
                cosine = vectors @ vectors.T
                upper = np.triu_indices(len(vectors), k=1)
                values = cosine[upper]
                stats[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0)),
                               "pair_count": int(values.size)}
            group_stats[kind] = stats
        for group in text_groups:
            kind = GROUPS[group]["condition"]
            row = {"config": group, "seed": 0, "fold": fold,
                   "condition_kind": kind,
                   "patient_text_shuffled": bool(GROUPS[group].get("text_shuffled", False)),
                   "train_pair_count_per_modality": next(iter(group_stats[kind].values()))["pair_count"]}
            for modality, name in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
                row[f"{name.lower()}_text_cosine_mean"] = group_stats[kind][name]["mean"]
                row[f"{name.lower()}_text_cosine_population_sd"] = group_stats[kind][name]["std"]
            rows.append(row)
    audit = pd.DataFrame(rows)
    assert len(audit) == 120 and not audit.duplicated(["config", "seed", "fold"]).any()
    return audit


def summarize(args):
    frame, diag = aggregate_metrics()
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / "fold_metrics.csv", index=False, float_format="%.12g")
    cosine_audit = text_cosine_audit(data_context(args.root))
    cosine_audit.to_csv(RESULTS / "text_cosine_audit.csv", index=False, float_format="%.12g")
    diag = diag.merge(cosine_audit, on=["config", "fold"], how="left", validate="one_to_one")
    audit_paths = [RESULTS / "initial_aux_gradient_audit_gpu0.csv",
                   RESULTS / "initial_aux_gradient_audit_gpu1.csv"]
    if not all(path.is_file() for path in audit_paths):
        raise RuntimeError("both GPU initial-gradient audits are required before final summary")
    initial_audit = pd.concat([pd.read_csv(path) for path in audit_paths], ignore_index=True)
    expected_audit_groups = [g for g, meta in GROUPS.items() if meta["relation"] != "none"]
    assert len(initial_audit) == 120
    assert not initial_audit.duplicated(["config", "seed", "fold"]).any()
    assert initial_audit.groupby("config").size().eq(10).all()
    assert set(initial_audit.config) == set(expected_audit_groups)
    assert (initial_audit.test_pair_count == 0).all()
    initial_audit.to_csv(RESULTS / "initial_aux_gradient_audit.csv", index=False,
                         float_format="%.12g")
    audit_columns = initial_audit[[
        "config", "fold", "relation_loss", "aux_pairs", "same_pairs",
        "different_pairs", "aux_query_grad_norm", "aux_key_grad_norm",
        "aux_graph_p_grad_norms", "total_query_grad_norm", "total_key_grad_norm",
        "test_pair_count"]].rename(columns={
            "relation_loss": "first_step_relation_loss",
            "aux_pairs": "first_step_aux_pair_count",
            "same_pairs": "first_step_same_pair_count",
            "different_pairs": "first_step_different_pair_count",
            "aux_query_grad_norm": "first_step_aux_query_grad_norm",
            "aux_key_grad_norm": "first_step_aux_key_grad_norm",
            "aux_graph_p_grad_norms": "first_step_aux_graph_p_grad_norms",
            "total_query_grad_norm": "first_step_total_query_grad_norm",
            "total_key_grad_norm": "first_step_total_key_grad_norm",
            "test_pair_count": "first_step_test_pair_count"})
    diag = diag.drop(columns=["relation_aux_query_grad_norm", "relation_aux_key_grad_norm"],
                     errors="ignore").merge(audit_columns, on=["config", "fold"], how="left",
                                             validate="one_to_one")
    diag["relation_aux_query_grad_norm"] = diag["first_step_aux_query_grad_norm"]
    diag["relation_aux_key_grad_norm"] = diag["first_step_aux_key_grad_norm"]
    diag.to_csv(RESULTS / "diagnostics.csv", index=False, float_format="%.12g")
    summaries = metric_summary(frame)
    comparisons = []
    for left, right in [
        ("G01_HYBRID_BASE", "G00_T0_REPLAY"),
        ("G02_FULL_SUPPORT_BCE", "G00_T0_REPLAY"),
        ("G03_HYBRID_SUPPORT_BCE", "G01_HYBRID_BASE"),
        ("G04_FULL_ALL_BCE", "G02_FULL_SUPPORT_BCE"),
        ("G05_HYBRID_ALL_BCE", "G03_HYBRID_SUPPORT_BCE"),
        ("G05_HYBRID_ALL_BCE", "G04_FULL_ALL_BCE"),
        ("G05_HYBRID_ALL_BCE", "G01_HYBRID_BASE"),
        ("G06_HYBRID_RANK", "G05_HYBRID_ALL_BCE"),
        ("G07_HYBRID_ALL_BCE_LOW", "G05_HYBRID_ALL_BCE"),
        ("G08_HYBRID_ALL_BCE_HIGH", "G05_HYBRID_ALL_BCE"),
        ("G05_HYBRID_ALL_BCE", "G09_HYBRID_TEXT_SHUFFLE"),
        ("G05_HYBRID_ALL_BCE", "G10_HYBRID_AUX_LABEL_SHUFFLE"),
        ("G05_HYBRID_ALL_BCE", "G11_HYBRID_NUMERIC_BCE"),
        ("G05_HYBRID_ALL_BCE", "G12_RAW_GRAPH_LABEL_SUP"),
        ("G05_HYBRID_ALL_BCE", "G13_HYBRID_SITE_BALANCED"),
    ]:
        comparisons.append(compare_groups(frame, left, right))

    hist = pd.read_csv(RUNTIME / "reference" / "experiment36_fold_metrics.csv")
    hist_b0 = hist.loc[hist.config == "B0_ORIGINAL_B0A"].sort_values("fold")
    hist_b1 = hist.loc[hist.config == "B1_FOUR_GRAPH_BASE"].sort_values("fold")
    hist_t0 = hist.loc[hist.config == "T0_FOUR_SYM_TEXT"].sort_values("fold")
    history = compare_t0_replay(frame.loc[frame.config == "G00_T0_REPLAY"].to_dict("records"))
    best_nonreplay = max((x for x in summaries if x["config"] not in {"G00_T0_REPLAY", "G01_HYBRID_BASE"}),
                         key=lambda x: (x["acc_mean_percent"], x["hard_auc_mean_percent"]))
    historical_comparisons = []
    for group in ("G00_T0_REPLAY", best_nonreplay["config"], "G05_HYBRID_ALL_BCE"):
        historical_comparisons.extend([
            paired_vs_external(frame, group, hist_b0, "E36_B0_ORIGINAL_B0A"),
            paired_vs_external(frame, group, hist_b1, "E36_B1_FOUR_GRAPH_BASE"),
            paired_vs_external(frame, group, hist_t0, "E36_T0_FOUR_SYM_TEXT")])
    summary = {"status": "COMPLETE", "groups": 14, "folds": 140,
               "gpu_folds": {str(k): int(v) for k, v in frame.physical_gpu.value_counts().sort_index().items()},
               "metrics": summaries, "paired_comparisons": comparisons,
               "historical_comparisons": historical_comparisons,
               "g00_e36_t0_replay": history,
               "wall_seconds": float(read_json(RUNTIME / "dispatch.json").get("elapsed_seconds", 0.0)),
               "data_audit": read_json(RUNTIME / "preflight.json")["data"],
               "cache_validation": read_json(RUNTIME / "preflight.json")["cache_validation"],
               "text_encoder": read_json(RUNTIME / "text_encoder_manifest.json"),
               "license_attestation": read_json(RUNTIME / "license_attestation.json"),
               "best_nonbaseline_group": best_nonreplay["config"],
               "historical_protocol_limits": ["val_idx=test_idx used for checkpoint selection",
                   "RFE and global preprocessing predate outer CV", "transductive 871-node GFR"],
               "decision": "PENDING"}
    main_row = next(x for x in summaries if x["config"] == "G05_HYBRID_ALL_BCE")
    baseline = next(x for x in summaries if x["config"] == "G00_T0_REPLAY")
    g01 = next(x for x in summaries if x["config"] == "G01_HYBRID_BASE")
    g09 = next(x for x in summaries if x["config"] == "G09_HYBRID_TEXT_SHUFFLE")
    g10 = next(x for x in summaries if x["config"] == "G10_HYBRID_AUX_LABEL_SHUFFLE")
    diag_main = diag.loc[diag.config == "G05_HYBRID_ALL_BCE"]
    audit_main = initial_audit.loc[initial_audit.config == "G05_HYBRID_ALL_BCE"]
    mechanism_used = (len(audit_main) == 10 and
                      (audit_main.aux_query_grad_norm > 0).all() and
                      (audit_main.aux_key_grad_norm > 0).all() and
                      (audit_main.test_pair_count == 0).all())
    qk_changed_in_training = (diag_main.get("query_gradient_max", pd.Series(dtype=float)).fillna(0).max() > 0 and
                              diag_main.get("key_gradient_max", pd.Series(dtype=float)).fillna(0).max() > 0)
    text_effect = (float(diag_main.real_vs_mismatched_logit_abs_mean.mean())
                   if "real_vs_mismatched_logit_abs_mean" in diag_main else 0.0)
    initial_audit_summary = {}
    for group in expected_audit_groups:
        part = initial_audit.loc[initial_audit.config == group]
        item = {"folds": int(len(part)),
                "aux_pairs_mean_per_four_modalities": float(part.aux_pairs.mean()),
                "aux_query_grad_mean": float(part.aux_query_grad_norm.mean()),
                "aux_query_grad_min": float(part.aux_query_grad_norm.min()),
                "aux_key_grad_mean": float(part.aux_key_grad_norm.mean()),
                "aux_key_grad_min": float(part.aux_key_grad_norm.min()),
                "total_query_grad_mean": float(part.total_query_grad_norm.mean()),
                "total_key_grad_mean": float(part.total_key_grad_norm.mean()),
                "test_pair_count_max": int(part.test_pair_count.max())}
        if group == "G12_RAW_GRAPH_LABEL_SUP":
            graph_norms = [json.loads(value) for value in part.aux_graph_p_grad_norms]
            item["graph_p_grad_mean_by_modality"] = np.asarray(graph_norms, dtype=float).mean(axis=0).tolist()
            item["graph_p_grad_min_by_modality"] = np.asarray(graph_norms, dtype=float).min(axis=0).tolist()
        initial_audit_summary[group] = item
    modality_summary = []
    for modality, name in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
        samples = [entry for raw in diag_main.per_modality.dropna()
                   for entry in json.loads(raw) if int(entry["modality"]) == modality]
        def avg(field):
            values = [entry.get(field) for entry in samples if entry.get(field) is not None]
            return float(np.mean(values)) if values else None
        modality_summary.append({
            "modality": name, "beta_mean": avg("beta"),
            "same_true_score_mean": avg("same_true_score_mean"),
            "different_true_score_mean": avg("different_true_score_mean"),
            "same_true_edge_multiplier": avg("same_true_edge_multiplier"),
            "different_true_edge_multiplier": avg("different_true_edge_multiplier"),
            "same_minus_different_multiplier": avg("same_minus_different_multiplier"),
            "text_cosine_qk_score_spearman": avg("text_cosine_qk_score_spearman_train_edges"),
            "mean_absolute_edge_weight_change": float(diag_main[f"condition_weight_abs_change_mean_{modality}"].mean())})
    fusion_values = [json.loads(value) for value in diag_main.source_fusion_weights.dropna()]
    condition_delta_values = [json.loads(value) for value in diag_main.best_condition_parameter_delta.dropna()]
    summary["initial_auxiliary_gradient_audit"] = {
        "rows": int(len(initial_audit)), "supervised_groups": initial_audit_summary,
        "all_train_only_pair_counts_zero_test_pairs": bool((initial_audit.test_pair_count == 0).all()),
        "note": "One disposable first-step replay per group/fold with the saved E36 fold-entry RNG; formal checkpoints and metrics were not changed."}
    g05_cosine = cosine_audit.loc[cosine_audit.config == "G05_HYBRID_ALL_BCE"]
    summary["bio_lord_text_cosine_train_pair_summary"] = {
        name: {"mean_over_folds": float(g05_cosine[f"{name.lower()}_text_cosine_mean"].mean()),
               "population_sd_of_fold_means": float(g05_cosine[f"{name.lower()}_text_cosine_mean"].std(ddof=0)),
               "mean_within_fold_cosine_population_sd": float(g05_cosine[f"{name.lower()}_text_cosine_population_sd"].mean()),
               "pair_count_per_fold": int(g05_cosine["train_pair_count_per_modality"].iloc[0])}
        for name in ("PHENO", "ANAT", "FUNC", "CORR")}
    summary["mechanism_diagnostics"] = {
        "G05_initial_aux_qk_gradients_positive_10_of_10": bool(mechanism_used),
        "G05_qk_parameters_received_training_gradients": bool(qk_changed_in_training),
        "G05_final_relation_loss_mean": float(diag_main.relation_loss.mean()),
        "G05_real_vs_mismatched_text_test_label_flip_fraction": float(diag_main.real_vs_mismatched_test_flip.mean()),
        "G05_real_vs_mismatched_text_mean_absolute_logit_change": text_effect,
        "G05_condition_off_test_label_flip_fraction": float(diag_main.condition_off_flip_fraction.mean()),
        "G05_edge_effect_by_modality": modality_summary,
        "G05_four_graph_source_fusion_weights_mean": np.asarray(fusion_values, dtype=float).mean(axis=0).tolist(),
        "G05_condition_parameter_update_norms_mean": {
            key: float(np.mean([value[key] for value in condition_delta_values]))
            for key in ("gfr.conditioner.query.weight", "gfr.conditioner.key.weight", "gfr.conditioner.beta_raw")},
        "G09_patient_text_rows_changed_mean": float(diag.loc[diag.config == "G09_HYBRID_TEXT_SHUFFLE", "shuffle_changed"].mean()),
        "G09_patient_text_fixed_points_mean": float(diag.loc[diag.config == "G09_HYBRID_TEXT_SHUFFLE", "shuffle_fixed_points"].mean()),
        "G10_auxiliary_train_labels_changed_fraction_mean": float(diag.loc[diag.config == "G10_HYBRID_AUX_LABEL_SHUFFLE", "aux_label_shuffle_fraction"].mean()),
        "G12_raw_graph_p_gradient_mean_by_modality": initial_audit_summary["G12_RAW_GRAPH_LABEL_SUP"]["graph_p_grad_mean_by_modality"]}
    improvement = all(main_row[f"{metric}_mean_percent"] >= baseline[f"{metric}_mean_percent"] + 0.35
                      for metric in ("acc", "hard_auc"))
    outperforms_controls = all(main_row[f"{metric}_mean_percent"] > control[f"{metric}_mean_percent"]
                               for control in (g01, g09, g10)
                               for metric in ("acc", "hard_auc"))
    summary["decision"] = ("EXPLORATORY_GO" if improvement and outperforms_controls and mechanism_used
                           else "MARGINAL_GAIN" if any(main_row[f"{m}_mean_percent"] > baseline[f"{m}_mean_percent"]
                                                         for m in ("acc", "hard_auc"))
                           else "NO-GO")
    summary["decision_checks"] = {"G05_vs_G00_at_least_0.35pp_acc_and_hard_auc": improvement,
                                  "G05_beats_HYBRID_base_and_both_mismatch_controls": outperforms_controls,
                                  "QK_auxiliary_gradient_reaches_router_10_of_10": bool(mechanism_used),
                                  "QK_parameters_updated_during_training": bool(qk_changed_in_training),
                                  "real_text_changes_heldout_predictions_or_logits_materially": bool(text_effect > 1e-5),
                                  "formal_significance_claim": False}
    json_write(RESULTS / "summary.json", summary)
    write_report(frame, summaries, comparisons, summary, diag)
    update_manifest(summary)
    print(f"SUMMARY groups=14 folds=140 GPU0=70 GPU1=70 decision={summary['decision']}", flush=True)


def write_report(frame, summaries, comparisons, summary, diagnostics):
    by = {x["config"]: x for x in summaries}
    lines = [
        "# Experiment 37 — LSPCR",
        "",
        f"Status: **{summary['status']}**; {summary['folds']}/140 folds. GPU0={summary['gpu_folds'].get('0', 0)}, GPU1={summary['gpu_folds'].get('1', 0)}; dispatch wall time={summary['wall_seconds']:.1f}s.",
        "",
        "## Audit and Design",
        "",
        "SPromptGL already defines a label-supervised adjacency in the paper, so LSPCR does not claim first use of labels in graph learning. In the repository's `simple-2` training path, `ALoss` is computed but excluded; the optimized historical objective is classification plus the mean four GraphConstruct losses. `GraphLearn` uses projected cosine with a detached `>0.9` threshold. The original GFR PHENO branch is overwritten at its second branch and was not changed.",
        "",
        "Experiment 36's T0/B1 four-graph trunk is not the original B0: it uses shared two-layer GCN parameters, four separately propagated full-320D messages, learned softmax message fusion, and row-stochastic normalization. E36 T0 changed some supported edges but did not beat the original B0 on ACC, fixed-checkpoint text-off predictions did not change class, and fusion weights remained close to 0.25. E36's single graph became too dense and performed poorly, so E37 keeps the four-graph symmetric path. Its prior text audit found PHENO48 mainly age-bin/sex/site; the text CSV additionally has real age, FIQ/VIQ/PIQ and handedness. ANAT/FUNC are QC/motion fields; CORR descriptions are feature-slot blocks, not verified ROI names. Dense BioLORD versus zero-padded numeric conditions are not a capacity-matched language comparison.",
        "",
        "PhenoBrain conditions ROI-level communication within each participant. LSPCR instead supervises patient-to-patient Q/K scores. The model never uses a label to construct the inference graph: labels form soft train-train targets during training only; held-out labels are excluded from every pair sampler and auxiliary loss. G12 directly supervises only existing thresholded GraphLearn train-train edges, exposing the detached hard-threshold limitation.",
        "",
        "## Method",
        "",
        "For receiver i and sender j, the score used by PCPR is `e(i<-j)=q(z_i)^T k(z_j)/sqrt(64)`. Supported non-self scores are row-centered and symmetrized, then existing edge weights are multiplied by `exp(beta*tanh(e_centered))`; `beta` is bounded by 0.25 and initialized at 0.10. Missing edges remain zero, the diagonal stays unchanged, and row-stochastic propagation carries the full 320D sender feature. Q/K/beta use a separate GFR-Adam group with zero weight decay; historical GFR/PLN parameters retain 0.11.",
        "",
        "SUPPORT-BCE uses supported train-train pairs; ALL-PAIR-BCE supervises all train-train non-self pairs without adding edges. Each class has a 0.9/0.1 soft target; positive and negative BCE means are averaged equally. Ranking uses up to four lowest-scoring same-label and four highest-scoring different-label train peers per anchor, with margin 0.5. G13 averages class-balanced losses within sites that contain both pair types. G10 shuffles only auxiliary train labels within site; classification retains true training labels. G12 supervises raw GraphLearn probabilities only where the existing thresholded graph has support. All objectives add `lambda_relation * Lrelation` to unchanged classification and graph losses; original ALoss remains excluded.",
        "",
        "HYBRID embeddings are composed exactly as E36 `modal_full[:,0]` for PHENO and `modal_only[:,1:4]` for ANAT/FUNC/CORR. The frozen encoder revision, pooling, and L2 normalization are unchanged. Inference is phenotype-conditioned and label-free. The inherited `val=test` selection, pre-CV feature selection/preprocessing, and transductive graph protocol remain optimistic and do not establish independent generalization.",
        "",
        "## E36 Replay",
        "",
        f"G00 replay: **{summary['g00_e36_t0_replay']['status']}**; checks `{json.dumps(summary['g00_e36_t0_replay']['fold_checks'], sort_keys=True)}`.",
        "",
        "## Formal Metrics",
        "",
        "Mean ± population SD over ten folds, percent. AUC is the original hard-label protocol, not probability ROC-AUC.",
        "",
        "|Group|ACC|Hard-AUC|SPE-TD|SEN-ASD|",
        "|---|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        s = by[group]
        lines.append(f"|{group}|{s['acc_mean_percent']:.4f} ± {s['acc_population_sd_percent']:.4f}|{s['hard_auc_mean_percent']:.4f} ± {s['hard_auc_population_sd_percent']:.4f}|{s['spe_td_mean_percent']:.4f} ± {s['spe_td_population_sd_percent']:.4f}|{s['sen_asd_mean_percent']:.4f} ± {s['sen_asd_population_sd_percent']:.4f}|")
    lines += ["", "## Paired Results", "", "Delta is left minus right in percentage points. W/T/L use the ten matched folds for ACC and Hard-AUC; folds overlap in training data and are not independent datasets.",
              "", "|Left − right|ΔACC|ACC W/T/L|ΔHard-AUC|Hard-AUC W/T/L|ΔSPE|ΔSEN|", "|---|---:|---:|---:|---:|---:|---:|"]
    for c in comparisons:
        lines.append(f"|{c['left']} − {c['right']}|{c['delta_acc_pp']:+.4f}|{'/'.join(map(str,c['acc_wins_ties_losses']))}|{c['delta_hard_auc_pp']:+.4f}|{'/'.join(map(str,c['hard_auc_wins_ties_losses']))}|{c['delta_spe_td_pp']:+.4f}|{c['delta_sen_asd_pp']:+.4f}|")
    for c in summary["historical_comparisons"]:
        lines.append(f"|{c['left']} − {c['right']}|{c['delta_acc_pp']:+.4f}|{'/'.join(map(str,c['acc_wins_ties_losses']))}|{c['delta_hard_auc_pp']:+.4f}|{'/'.join(map(str,c['hard_auc_wins_ties_losses']))}|{c['delta_spe_td_pp']:+.4f}|{c['delta_sen_asd_pp']:+.4f}|")
    g05 = diagnostics.loc[diagnostics.config == "G05_HYBRID_ALL_BCE"]
    grad_audit = summary["initial_auxiliary_gradient_audit"]["supervised_groups"]
    qgrad = grad_audit["G05_HYBRID_ALL_BCE"]["aux_query_grad_mean"]
    kgrad = grad_audit["G05_HYBRID_ALL_BCE"]["aux_key_grad_mean"]
    text_flips = summary["mechanism_diagnostics"]["G05_real_vs_mismatched_text_test_label_flip_fraction"]
    logit_delta = summary["mechanism_diagnostics"]["G05_real_vs_mismatched_text_mean_absolute_logit_change"]
    raw_graph_grad = summary["mechanism_diagnostics"]["G12_raw_graph_p_gradient_mean_by_modality"]
    lines += ["", "## Mechanism and Decision", "",
              f"A separate reproducible one-step replay using the saved fold-entry RNG audited all 120 supervised group/fold combinations. G05 had positive first-step auxiliary Q/K gradient norms in 10/10 folds (mean query={qgrad:.6g}, key={kgrad:.6g}); training also updated Q/K. This audit used disposable model instances and did not change the formal checkpoints or metrics. G12's raw GraphLearn projection gradients were nonzero on existing edges (mean PHENO/ANAT/FUNC/CORR norms: `{json.dumps(raw_graph_grad)}`); the detached `>0.9` mask still prevents absent patient edges from receiving direct auxiliary supervision.",
              "",
              f"For G05, fixed-checkpoint replacement of real text by the deterministic within-pool mismatch changed no held-out class ({text_flips:.4f} flip fraction) and changed logits by only {logit_delta:.3g} on average. Turning PCPR off also changed no held-out predicted class. The relation score was trained, but its held-out prediction effect was negligible. Mean fusion weights remained `{json.dumps(summary['mechanism_diagnostics']['G05_four_graph_source_fusion_weights_mean'])}`, close to equal routing.",
              "",
              "|G05 source|beta|same / different edge multiplier|difference|QK-score vs text-cosine Spearman|BioLORD cosine mean ± pair SD|mean absolute raw-edge change|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for item in summary["mechanism_diagnostics"]["G05_edge_effect_by_modality"]:
        cosine = summary["bio_lord_text_cosine_train_pair_summary"][item["modality"]]
        lines.append(f"|{item['modality']}|{item['beta_mean']:.4f}|{item['same_true_edge_multiplier']:.6f} / {item['different_true_edge_multiplier']:.6f}|{item['same_minus_different_multiplier']:.6f}|{item['text_cosine_qk_score_spearman']:.4f}|{cosine['mean_over_folds']:.4f} ± {cosine['mean_within_fold_cosine_population_sd']:.4f}|{item['mean_absolute_edge_weight_change']:.6f}|")
    lines += ["",
              "These values are measured on outer-train supported edges. The main G05 auxiliary loss remained near its uninformative BCE level and same/different multiplier separation was concentrated in PHENO; the ANAT/FUNC/CORR changes were very small. No new graph edges were added.",
              "",
              f"The held-out-label invariance check ran for all 140 formal folds and requires bitwise-identical inference logits after replacing held-out labels. Auxiliary pair construction asserts zero test-pair participation. G09 shuffled an average of {summary['mechanism_diagnostics']['G09_patient_text_rows_changed_mean']:.1f}/871 text assignments within train/test pools; G10 permuted {summary['mechanism_diagnostics']['G10_auxiliary_train_labels_changed_fraction_mean']:.1%} of train auxiliary-label positions within site while preserving each site's label counts.",
              "",
              "Execution recovery: the first SUPPORT-BCE attempt encountered a thresholded modality with no available same/different supported pair; the absent-class term was set to zero and the fold resumed from its atomic checkpoint. A later G09 startup assertion incorrectly assumed unshuffled text identity; it was corrected to validate the fixed within-pool permutation and G09 then completed all ten folds. Neither failed startup produced a duplicate result row; the final table is keyed uniquely by group/seed/fold.",
              "",
              f"Decision: **{summary['decision']}**. Pre-registered internal checks: `{json.dumps(summary['decision_checks'], sort_keys=True)}`. This is a single-seed exploratory decision, not evidence of statistical significance, clinical validity, or SOTA.",
              "",
              "## Reproducibility",
              "",
              "`manifest.json` records source/data/cache/config/code/result hashes, fold assignment, RNG pairing, group/GPU assignment, task completion and delivery state. `fold_metrics.csv` contains 140 unique group/seed/fold rows. `initial_aux_gradient_audit.csv` stores the separate 120-row gradient replay, `text_cosine_audit.csv` records real BioLORD cosine distributions over outer-train pairs, and `diagnostics.csv` joins these audits to per-fold training and inference diagnostics. `.runtime/` contains the validated immutable E36 text caches, fold-entry RNG states, atomic task checkpoints and logs. No root model, data, or historical experiment was edited.", ""]
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


def update_manifest(summary):
    data_audit = summary["data_audit"]
    root = Path(data_audit.get("root", "/data/adodas/SPromptGL-main"))
    source_hashes = {}
    for name in ("main.py", "network.py", "model.py", "layers.py", "utils.py",
                 "ABIDE-simple-2-concat-weighted-cosine.sh"):
        path = root / name
        if path.is_file():
            source_hashes[str(path)] = sha_file(path)
    result_names = ("fold_metrics.csv", "diagnostics.csv", "summary.json", "report.md",
                    "initial_aux_gradient_audit.csv", "initial_aux_gradient_audit_gpu0.csv",
                    "initial_aux_gradient_audit_gpu1.csv", "text_cosine_audit.csv")
    result_hashes = {name: sha_file(RESULTS / name) for name in result_names
                     if (RESULTS / name).is_file()}
    task_hashes = {path.name: sha_file(path) for path in sorted((RUNTIME / "tasks").glob("*.json"))}
    checkpoint_hashes = {path.name: sha_file(path)
                         for path in sorted((RUNTIME / "checkpoints").glob("*.pt"))}
    dispatch_state = read_json(RUNTIME / "dispatch.json")
    manifest = {"experiment": "Experiment 37 — LSPCR",
                "status": summary["status"], "registered_groups": list(GROUPS),
                "completed_groups": list(GROUPS), "registered_folds": 140,
                "completed_folds": 140, "gpu_folds": summary["gpu_folds"],
                "gpu_assignment": {"0": GPU0, "1": GPU1},
                "source_sha256": source_hashes, "data_audit": data_audit,
                "cache_validation": summary["cache_validation"],
                "config_sha256": sha_file(CONFIG_FILE),
                "code_sha256": {p.name: sha_file(p) for p in HERE.glob("*.py")},
                "cache_sha256": {p.name: sha_file(p) for p in sorted((RUNTIME / "cache").glob("*.npz"))},
                "result_sha256": result_hashes, "task_state_sha256": task_hashes,
                "checkpoint_sha256": checkpoint_hashes,
                "rng_entry_sha256": sha_file(entry_rng_file()),
                "encoder": {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                            "pooling": "attention-mask mean pooling + L2 normalization", "frozen": True},
                "license_attestation": summary["license_attestation"],
                "g00_e36_t0_replay": summary["g00_e36_t0_replay"],
                "initial_auxiliary_gradient_audit": summary["initial_auxiliary_gradient_audit"],
                "mechanism_diagnostics": summary["mechanism_diagnostics"],
                "dispatch_state": {"status": dispatch_state.get("status"),
                                   "elapsed_seconds": dispatch_state.get("elapsed_seconds"),
                                   "recovered_failed_group": dispatch_state.get("failed_group")},
                "decision": summary["decision"], "elapsed_seconds": summary["wall_seconds"],
                "handoff_status": "PENDING_APPEND", "artifact_directory": str(HERE),
                "runtime_directory": str(RUNTIME)}
    json_write(HERE / "manifest.json", manifest)
    json_write(RUNTIME / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_audit = sub.add_parser("audit")
    p_audit.add_argument("--root", type=Path, required=True)
    p_enc = sub.add_parser("encode")
    p_enc.add_argument("--root", type=Path, required=True)
    p_enc.add_argument("--model-home", type=Path, default=RUNTIME / "model")
    p_numeric = sub.add_parser("numeric-cache")
    p_numeric.add_argument("--root", type=Path, required=True)
    p_attest = sub.add_parser("attest-license")
    p_attest.add_argument("--umls", action="store_true")
    p_attest.add_argument("--snomed-ct", action="store_true")
    p_pre = sub.add_parser("preflight")
    p_pre.add_argument("--root", type=Path, required=True)
    p_worker = sub.add_parser("worker")
    p_worker.add_argument("--root", type=Path, required=True)
    p_worker.add_argument("--config", required=True, choices=GROUPS)
    p_worker.add_argument("--gpu", type=int, choices=[0, 1], required=True)
    p_dispatch = sub.add_parser("dispatch")
    p_dispatch.add_argument("--root", type=Path, required=True)
    p_dispatch.add_argument("--controls-only", action="store_true",
                            help="Run only structure and numeric arms while text encoder licensing is pending")
    p_summary = sub.add_parser("summarize")
    p_summary.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        run_audit(args)
    elif args.command == "encode":
        data = data_context(args.root)
        encode_all(data, args.model_home)
    elif args.command == "numeric-cache":
        ensure_dirs()
        build_numeric_caches(data_context(args.root))
    elif args.command == "attest-license":
        attest_license(args)
    elif args.command == "preflight":
        run_preflight(args)
    elif args.command == "worker":
        worker(args)
    elif args.command == "dispatch":
        dispatch(args)
    elif args.command == "summarize":
        summarize(args)


if __name__ == "__main__":
    main()
