#!/usr/bin/env python3
"""Audited, resumable runner for the frozen 22-group Experiment 36."""
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
    "B0_ORIGINAL_B0A": dict(structure="original", condition="none", directed=False, modality="none"),
    "B1_FOUR_GRAPH_BASE": dict(structure="four", condition="none", directed=False, modality="none"),
    "B2_ONE_GRAPH_BASE": dict(structure="one", condition="none", directed=False, modality="none"),
    "B3_CORR_ONLY_BASE": dict(structure="corr", condition="none", directed=False, modality="none"),
    "N0_FOUR_SYM_NUMERIC": dict(structure="four", condition="numeric", directed=False, modality="four"),
    "N1_FOUR_DIR_NUMERIC": dict(structure="four", condition="numeric", directed=True, modality="four"),
    "N2_ONE_SYM_NUMERIC": dict(structure="one", condition="numeric", directed=False, modality="all"),
    "N3_ONE_DIR_NUMERIC": dict(structure="one", condition="numeric", directed=True, modality="all"),
    "T0_FOUR_SYM_TEXT": dict(structure="four", condition="modal_full", directed=False, modality="four"),
    "T1_FOUR_DIR_TEXT": dict(structure="four", condition="modal_full", directed=True, modality="four"),
    "T2_ONE_SYM_TEXT": dict(structure="one", condition="all_full", directed=False, modality="all"),
    "T3_ONE_DIR_TEXT": dict(structure="one", condition="all_full", directed=True, modality="all"),
    "S0_FOUR_SYM_SHUFFLED": dict(structure="four", condition="modal_full", directed=False, modality="four", shuffled=True),
    "S1_FOUR_DIR_SHUFFLED": dict(structure="four", condition="modal_full", directed=True, modality="four", shuffled=True),
    "S2_ONE_SYM_SHUFFLED": dict(structure="one", condition="all_full", directed=False, modality="all", shuffled=True),
    "S3_ONE_DIR_SHUFFLED": dict(structure="one", condition="all_full", directed=True, modality="all", shuffled=True),
    "C0_FOUR_SYM_CONSTANT": dict(structure="four", condition="constant_modal", directed=False, modality="four"),
    "C1_ONE_SYM_CONSTANT": dict(structure="one", condition="constant_all", directed=False, modality="all"),
    "H0_FOUR_SYM_SHARED_CLINICAL": dict(structure="four", condition="clinical", directed=False, modality="four"),
    "H1_ONE_SYM_CLINICAL_ONLY": dict(structure="one", condition="clinical", directed=False, modality="all"),
    "M0_FOUR_SYM_MODAL_ONLY": dict(structure="four", condition="modal_only", directed=False, modality="four"),
    "M1_ONE_SYM_MODAL_ONLY": dict(structure="one", condition="all_modal_only", directed=False, modality="all"),
}
GPU0 = ["B0_ORIGINAL_B0A", "B2_ONE_GRAPH_BASE", "N0_FOUR_SYM_NUMERIC", "N2_ONE_SYM_NUMERIC",
        "T0_FOUR_SYM_TEXT", "T2_ONE_SYM_TEXT", "S0_FOUR_SYM_SHUFFLED", "S2_ONE_SYM_SHUFFLED",
        "C0_FOUR_SYM_CONSTANT", "H0_FOUR_SYM_SHARED_CLINICAL", "M0_FOUR_SYM_MODAL_ONLY"]
GPU1 = ["B1_FOUR_GRAPH_BASE", "B3_CORR_ONLY_BASE", "N1_FOUR_DIR_NUMERIC", "N3_ONE_DIR_NUMERIC",
        "T1_FOUR_DIR_TEXT", "T3_ONE_DIR_TEXT", "S1_FOUR_DIR_SHUFFLED", "S3_ONE_DIR_SHUFFLED",
        "C1_ONE_SYM_CONSTANT", "H1_ONE_SYM_CLINICAL_ONLY", "M1_ONE_SYM_MODAL_ONLY"]
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
    from build_phenotype_text import load_aligned_inputs
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
    from build_phenotype_text import make_fold_payload, missingness_audit, text_summary
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
    from build_phenotype_text import make_fold_payload, text_summary
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
    from build_phenotype_text import sha256_file
    inputs = data["audit"]["inputs_sha256"]
    return {
        "fold": int(fold),
        "train_indices_sha256": hashlib.sha256(np.asarray(train, dtype="<i8").tobytes()).hexdigest(),
        "test_indices_sha256": hashlib.sha256(np.asarray(test, dtype="<i8").tobytes()).hexdigest(),
        "ordered_subject_id_sha256": data["audit"]["ordered_subject_id_sha256"],
        "source_csv_sha256": next(v for k, v in inputs.items() if k.endswith("Phenotypic_V1_0b_preprocessed1.csv")),
        "processed_input_sha256": next(v for k, v in inputs.items() if k.endswith("processed_standard_data.csv")),
        "identity_mapping_sha256": next(v for k, v in inputs.items() if k.endswith("subject_row_mapping.csv")),
        "template_builder_sha256": sha_file(HERE / "build_phenotype_text.py"),
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
    from build_phenotype_text import make_fold_payload
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


def condition_for_group(group, cache, permutation):
    meta = GROUPS[group]
    name = meta["condition"]
    if name == "none":
        return None
    if name == "numeric":
        value = cache["modal_numeric"] if meta["modality"] == "four" else cache["all_numeric"]
    elif name == "modal_full":
        value = cache["modal_full"]
    elif name == "all_full":
        value = cache["all_full"]
    elif name == "clinical":
        clinical = cache["clinical"]
        value = np.repeat(clinical[:, None, :], 4, axis=1) if meta["modality"] == "four" else clinical
    elif name == "constant_modal":
        value = np.repeat(cache["constant_modal"][None, :, :], len(cache["all_numeric"]), axis=0)
    elif name == "constant_all":
        value = np.repeat(cache["constant_all"][None, :], len(cache["all_numeric"]), axis=0)
    elif name == "modal_only":
        value = cache["modal_only"]
    elif name == "all_modal_only":
        value = cache["all_modal_only"]
    else:
        raise ValueError(name)
    if meta.get("shuffled"):
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

    from pcpr import validate_router
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
    audit = {"data": data["audit"], "engineering": source_preflight(args.root, data),
             "status": "PASS"}
    json_write(RUNTIME / "preflight.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


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
    return RUNTIME / "rng" / "e0_fold_entries.pt"


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
    if meta.get("shuffled"):
        permutation, shuffle_info = patient_shuffle(data, train, test, fold)
    from build_phenotype_text import text_summary
    condition = condition_for_group(group, cache, permutation)
    summary = {"condition": meta["condition"], **shuffle_info}
    if meta["condition"] in {"modal_full", "all_full", "clinical", "modal_only", "all_modal_only",
                              "constant_modal", "constant_all"}:
        summary["text_unique_rate"] = None
    return condition, permutation, summary


def make_agent(root, data, group, fold, train, test, condition):
    from integration import EvalHelper36
    network, _, _ = load_modules(root)
    hp = SimpleNamespace(**PROTOCOL)
    meta = GROUPS[group]
    return EvalHelper36([48, 6, 10, 256], data["x"], data["y"], hp, train, test,
                        group, meta["structure"], meta["directed"],
                        meta["condition"] != "none", condition, fold_id=fold)


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
        result = saved["result"]
    prior_seconds = float(result.get("seconds", 0.0))
    start_wall = time.monotonic()

    if group == "B0_ORIGINAL_B0A":
        entry_path = entry_rng_file()
        entries = torch.load(entry_path, map_location="cpu", weights_only=False) if entry_path.exists() else []
    else:
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
    elif group == "B0_ORIGINAL_B0A" and saved is None:
        # Match main.py: seed once immediately before the continuous fold stream.
        if not result.get("metrics"):
            entries = []
        set_seed(0)
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

        resume_epoch = bool(saved and saved.get("phase") == "EPOCH" and int(saved["fold"]) == fold)
        if group == "B0_ORIGINAL_B0A":
            if resume_epoch:
                if len(entries) < fold:
                    raise RuntimeError("missing E0 fold-entry RNG while resuming an interrupted fold")
                entry_state = entries[fold - 1]
            else:
                entry_state = rng_state()
                entries.append(entry_state)
                atomic_torch_save(entry_rng_file(), entries)
        else:
            entry_state = entries[fold - 1]
            restore_rng(entry_state)

        agent = EvalHelper36([48, 6, 10, 256], data["x"], data["y"],
                             SimpleNamespace(**PROTOCOL), train, test,
                             group, meta["structure"], meta["directed"],
                             meta["condition"] != "none", condition, fold_id=fold)
        from integration import (model_state, load_model_state, optimizer_audit,
                                 parameter_sha256, historical_trunk_sha256, metric_row)
        parameter_info = optimizer_audit(agent)
        initial_state = model_state(agent)
        initial_parameter_hash = parameter_sha256(agent, baseline_common_only=True)
        historical_parameter_hash = historical_trunk_sha256(agent) if group == "B0_ORIGINAL_B0A" else None
        initial_rng_hash = rng_sha(entry_state)
        if group != "B0_ORIGINAL_B0A":
            e0_rows = {int(row["fold"]): row for row in load_task("B0_ORIGINAL_B0A")["metrics"]}
            reference = e0_rows[fold]
            assert initial_parameter_hash == reference["initial_parameter_sha256"], \
                f"shared trunk initialization differs from E0 fold {fold}"
            assert initial_rng_hash == reference["initial_rng_sha256"], \
                f"fold-entry RNG differs from E0 fold {fold}"

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
                    "model": model_state(agent), "optimizer_mp": agent.optimizer_MP0.state_dict(),
                    "optimizer_mf": agent.optimizer_MF.state_dict(), "rng": now,
                    "initial_state": initial_state, "initial_parameter_sha256": initial_parameter_hash,
                    "historical_initial_trunk_sha256": historical_parameter_hash,
                    "initial_rng_sha256": initial_rng_hash, "best_state": best_state,
                    "best_acc": best_acc, "best_epoch": best_epoch, "wait": wait,
                    "tracker": tracker, "fold_elapsed": fold_elapsed + time.monotonic() - fold_started,
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

        off_output = None
        if agent.condition_enabled:
            agent.condition_enabled = False
            off_output = agent.predict()
            agent.condition_enabled = True
            output = agent.predict()
            off_flip = float(np.mean(output.detach().cpu().numpy()[test].argmax(1) !=
                                     off_output.detach().cpu().numpy()[test].argmax(1)))
        else:
            off_flip = None
        diagnostic = summarize_graph_diagnostics(agent, data, fold, group, shuffle_info,
                                                 off_output, output, data["y"], test)
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
        atomic_torch_save(cp, {"config": group, "phase": "BOUNDARY", "next_fold": fold + 1,
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
