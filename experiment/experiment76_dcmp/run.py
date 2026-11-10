#!/usr/bin/env python3
"""Experiment 76 runner: fold-local confusion memory on the audited E74.4-G2 base."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
RT = EXP / ".runtime"
TASKS, CKPTS, LATEST, LOGS, CACHE = (RT / name for name in ("tasks", "checkpoints", "latest", "logs", "cache"))
RESULTS = EXP / "results"
BASE_DIR = Path(CFG["base"]["path"])
BASE_RUN = BASE_DIR / "run.py"
E75_DIR = Path(CFG["references"]["e75_dcer_v2"])
E744_MANIFEST = BASE_DIR / "manifest.json"
E744_AUDIT = BASE_DIR / "results/audit_status.json"
E744_TASKS = BASE_DIR / ".runtime/tasks"
E744_CKPTS = BASE_DIR / ".runtime/checkpoints"
E61_DIR = Path("/data/adodas/SPromptGL-main/experiment/experiment61_cadr_postalign_vs_e2e")
WARMUPS = E61_DIR / ".runtime/warmup"
TEXT_CACHE = Path(CFG["references"]["e724_text_cache"])
PATIENT_CACHE = Path(CFG["references"]["e724_patient_cache"])
PATIENT_DECODER = Path(CFG["references"]["e724_patient_decoder"])
LABEL_PROMPTS = Path(CFG["references"]["e724_label_prompts"])
GROUPS = tuple(CFG["groups"])
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
BASE = None
DCMP_MODULE = None


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_json(value) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode())


def index_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.int64)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False)):
        raise IOError(f"atomic JSON readback failed: {path}")


def atomic_torch(path: Path, value) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def torch_load(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_base():
    global BASE
    if BASE is None:
        BASE = _load_file("e76_readonly_e744_positive_dcr_runner", BASE_RUN)
    return BASE


def load_dcmp():
    global DCMP_MODULE
    if DCMP_MODULE is None:
        DCMP_MODULE = _load_file("e76_dcmp_local_module", EXP / "dcmp.py")
    return DCMP_MODULE


def source_inventory() -> dict:
    paths = {
        "e76_config": EXP / "config.json", "e76_run": EXP / "run.py",
        "e76_dcmp": EXP / "dcmp.py", "e76_audit": EXP / "audit.py",
        "e744_run": BASE_RUN, "e744_dcr": BASE_DIR / "soft_positive_dcr.py",
        "e744_config": BASE_DIR / "config.json", "e744_manifest": E744_MANIFEST,
        "e744_report": BASE_DIR / "results/report.md", "e744_fold_metrics": BASE_DIR / "results/fold_metrics.csv",
        "e744_audit_status": E744_AUDIT, "e61_run": E61_DIR / "run.py",
        "e61_model": E61_DIR / "model_e61.py", "e61_config": E61_DIR / "config.json",
        "e61_manifest": E61_DIR / "manifest.json", "brainprompt_cache": TEXT_CACHE,
        "patient_embedding_cache": PATIENT_CACHE, "patient_decoder": PATIENT_DECODER,
        "brainprompt_labels": LABEL_PROMPTS,
        "e741_run": Path("/model/spromptgl/experiment/experiment74_1_rudag/run.py"),
        "e743_dcr": Path("/model/spromptgl/experiment/experiment74_3_local_dcr/local_dcr.py"),
        "e53_run": Path(CFG["references"]["e53"]) / "run.py",
        "e75_run": E75_DIR / "run.py", "e75_dcer": E75_DIR / "dcer_v2.py",
        "e75_config": E75_DIR / "config.json", "e75_manifest": E75_DIR / "manifest.json",
        "e75_report": E75_DIR / "results/report.md",
        "e75_audit": E75_DIR / "results/audit_status.json",
        "root_main": Path("/data/adodas/SPromptGL-main/main.py"),
        "root_model": Path("/data/adodas/SPromptGL-main/model.py"),
        "root_network": Path("/data/adodas/SPromptGL-main/network.py"),
        "root_layers": Path("/data/adodas/SPromptGL-main/layers.py"),
        "root_utils": Path("/data/adodas/SPromptGL-main/utils.py"),
        "abide_processed": Path("/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv"),
        "abide_mapping": Path("/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy"),
        "abide_pheno_notebook": Path("/data/adodas/SPromptGL-main/data/ABIDE/phenotyic_processing.ipynb"),
        "graph_abide": Path("/data/adodas/SPromptGL-main/graph/ABIDE_weighted-cosine_graph_.npz"),
        "attention_abide": Path("/data/adodas/SPromptGL-main/attn/attn_map_ABIDE.npy"),
    }
    paths.update({f"e61_warmup_fold{fold:02d}": WARMUPS / f"fold{fold:02d}.pt" for fold in range(1, 11)})
    paths.update({f"e744_task_{fold:02d}": E744_TASKS / f"E74.4-G2-Positive-DCR_f{fold:02d}.json" for fold in range(1, 11)})
    paths.update({f"e744_checkpoint_{fold:02d}": E744_CKPTS / f"E74.4-G2-Positive-DCR_f{fold:02d}.best.pt" for fold in range(1, 11)})
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required source/reference missing: {path}")
    return {name: sha_file(path) for name, path in paths.items()}


def verify_remote_base_branch() -> str:
    branch = CFG["base"]["branch"]
    expected = CFG["base"]["commit"]
    result = subprocess.run(["git", "ls-remote", "https://github.com/tangaii/SPromptGL.git", f"refs/heads/{branch}"],
                            check=True, capture_output=True, text=True, timeout=30)
    rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or rows[0][0] != expected:
        raise RuntimeError(f"E74.4-G2 remote branch identity mismatch: {rows} expected={expected}")
    return rows[0][0]


def source_sha() -> str:
    return sha_json(source_inventory())


def rng_pack(base, state=None):
    return base.load_history().load_e69().load_e61().load_base().pack_rng()


def rng_restore(base, value):
    base.load_history().load_e69().load_e61().load_base().restore_rng(value)


def rng_sha(base, value=None):
    runtime = base.load_history().load_e69().load_e61().load_base()
    return runtime.rng_sha(value if value is not None else runtime.pack_rng())


def prepare_fold(fold: int):
    base = load_base()
    state, warmup, identity = base.prepare(fold)
    reference = json.loads((E744_TASKS / f"E74.4-G2-Positive-DCR_f{fold:02d}.json").read_text())
    expected_overlap = [377, 407] if fold == 2 else []
    if reference.get("status") != "COMPLETE" or reference.get("group") != "E74.4-G2-Positive-DCR":
        raise RuntimeError(f"E74.4-G2 reference task is not complete fold={fold}")
    if reference.get("identity") != identity:
        raise RuntimeError(f"E74.4-G2 warmup identity mismatch fold={fold}")
    if identity["membership"]["train_test_overlap_1based"] != expected_overlap:
        raise RuntimeError(f"unexpected train/test overlap fold={fold}")
    if state["train_pairs_np"].shape != (4096, 2):
        raise RuntimeError(f"E74.4 relation pair mismatch fold={fold}")
    state["fold"] = fold
    return state, warmup, identity, reference


def text_banks():
    import torch
    cache = torch_load(TEXT_CACHE)
    prompts = cache["text_map"]["brainprompt"]
    source_lines = LABEL_PROMPTS.read_text(encoding="utf-8").splitlines()
    if prompts != [source_lines[1], source_lines[0]]:
        raise RuntimeError("BrainPrompt official source/cache text mismatch or ASD/TD source ordering changed")
    embeddings = cache["embeddings"]["brainprompt"].detach().float().cpu()
    if tuple(embeddings.shape) != (2, 1024):
        raise RuntimeError(f"BrainPrompt frozen CLIP embeddings must be [2,1024], got {tuple(embeddings.shape)}")
    # Cache is audited in [ASD, TD] order even though the source file lists TC before ASD.
    if cache["text_map"]["brainprompt"][0] != source_lines[1]:
        raise RuntimeError("BrainPrompt label mapping is not ASD=0, TD=1")
    generator = torch.Generator(device="cpu").manual_seed(760076)
    direction_a = torch.nn.functional.normalize(torch.randn(1024, generator=generator), dim=0)
    orthogonal = torch.randn(1024, generator=generator)
    orthogonal = torch.nn.functional.normalize(orthogonal - torch.dot(direction_a, orthogonal) * direction_a, dim=0)
    cosine = torch.dot(torch.nn.functional.normalize(embeddings[0], dim=0), torch.nn.functional.normalize(embeddings[1], dim=0)).clamp(-0.999, 0.999)
    direction_b = cosine * direction_a + torch.sqrt(1.0 - cosine.square()) * orthogonal
    anonymous = torch.stack((direction_a * embeddings[0].norm(),
                             torch.nn.functional.normalize(direction_b, dim=0) * embeddings[1].norm()))
    if abs(float(torch.nn.functional.cosine_similarity(anonymous[0], anonymous[1], dim=0) - cosine)) > 1e-5:
        raise RuntimeError("anonymous BrainPrompt bank does not match class cosine geometry")
    return {"G1": embeddings, "G2": embeddings, "G3": embeddings, "G4": anonymous}, {
        "encoder": "frozen CLIP RN50, reused E72.4 audited cache", "cache_sha256": sha_file(TEXT_CACHE),
        "patient_cache_sha256": sha_file(PATIENT_CACHE), "label_source_sha256": sha_file(LABEL_PROMPTS),
        "texts_asd_td": prompts, "class_order": ["ASD", "TD"],
        "embedding_sha256": sha_bytes(embeddings.contiguous().numpy().tobytes()),
        "anonymous_embedding_sha256": sha_bytes(anonymous.contiguous().numpy().tobytes()),
        "real_anonymous_class_cosine": [float(cosine), float(torch.nn.functional.cosine_similarity(anonymous[0], anonymous[1], dim=0))],
    }


def decode_sites():
    import torch
    cache = torch_load(PATIENT_CACHE)
    patient_embeddings = cache.get("embeddings")
    metadata = cache.get("metadata", {})
    if not torch.is_tensor(patient_embeddings) or tuple(patient_embeddings.shape) != (871, 3, 1024):
        raise RuntimeError("E72.4 patient embedding cache shape metadata mismatch")
    if metadata.get("encoder") != "frozen OpenAI CLIP RN50" or metadata.get("row_indices_0based") != list(range(871)):
        raise RuntimeError("E72.4 patient embedding cache encoder or row alignment mismatch")
    embedding_sha = sha_bytes(patient_embeddings.detach().cpu().contiguous().numpy().tobytes())
    if metadata.get("embeddings_sha256") != embedding_sha:
        raise RuntimeError("E72.4 patient embedding cache content SHA mismatch")
    # Recover site IDs from the verified source PHENO mapping, never from labels or row position.
    patient_module = _load_file("e76_readonly_e724_patient_decoder", PATIENT_DECODER)
    mapping = Path("/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy")
    data = Path("/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv")
    decoded = patient_module.decode_patient_prompts(data, mapping)
    sites = [row["site"]["site_id"] if row["site"] is not None else "MISSING" for row in decoded["selections"]]
    site_levels = sorted(set(sites))
    lookup = {value: index for index, value in enumerate(site_levels)}
    site_ids = torch.tensor([lookup[value] for value in sites], dtype=torch.long)
    if len(site_ids) != 871 or decoded["row_count"] != 871:
        raise RuntimeError("decoded site metadata do not align with all 871 rows")
    return site_ids, {"site_levels": site_levels, "feature_sha256": decoded["feature_sha256"],
                      "mapping_sha256": decoded["mapping_sha256"],
                      "field_validity_sha256": sha_bytes(torch.as_tensor(decoded["field_validity"]).numpy().tobytes()),
                      "cache_source_hashes": cache.get("source_hashes", {}),
                      "cache_embedding_sha256": embedding_sha,
                      "row_indices_sha256": metadata.get("row_indices_sha256")}


def extract_warmup_cam144(state, group_name):
    import torch
    base = load_base()
    captured = {"calls": 0}
    layer = state["cam"].Outputlayer
    def hook(_module, inputs):
        captured["calls"] += 1
        if len(inputs) != 2 or tuple(inputs[0].shape) != (871, 144):
            raise RuntimeError(f"warmup OutputLayer input mismatch {tuple(inputs[0].shape)}")
        captured["h"] = inputs[0].detach().float().clone()
        return None
    handle = layer.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            base.forward(state, "E74.4-G2-Positive-DCR", training=False)
    finally:
        handle.remove()
    if captured["calls"] != 1:
        raise RuntimeError("warmup CAM/OutputLayer must be called once")
    return captured["h"]


def build_probe_and_memory(state, warmup_h, site_ids, fold):
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    base_train = index_numpy(state["train"])
    test = index_numpy(state["test"])
    eligible = np.asarray([idx for idx in base_train if idx not in set(test.tolist())], dtype=np.int64)
    y = state["y"].detach().cpu().numpy().astype(np.int64)
    if np.intersect1d(eligible, test).size:
        raise RuntimeError("DCMP eligible memory includes an outer test subject")
    x = warmup_h.detach().cpu().numpy()[eligible]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=760000 + fold)
    probe = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=760100 + fold)
    oof = cross_val_predict(probe, x, y[eligible], cv=cv, method="predict_proba")
    # Keep the fitted fold-local probe for a preregistered, label-free test difficulty rank.
    probe.fit(x, y[eligible])
    site_ids = site_ids.to(device=state["y"].device)
    memory = load_dcmp().build_memory(warmup_h.to(state["y"].device), state["y"], site_ids,
                                      base_train, test, fold, oof)
    memory.metadata["probe_oof_sha256"] = sha_bytes(oof.astype(np.float64).tobytes())
    memory.metadata["probe_oof_rows_zero_based"] = eligible.tolist()
    memory.metadata["probe_fitted_state_sha256"] = sha_json({"coef": probe.coef_.tolist(), "intercept": probe.intercept_.tolist()})
    memory.metadata["site_supports"] = {}
    for site in torch.unique(memory.sites[memory.valid]).tolist():
        mask = memory.valid & (memory.sites == int(site))
        counts = [int((mask & (memory.labels == cls)).sum()) for cls in (0, 1)]
        memory.metadata["site_supports"][str(int(site))] = counts
    probe_for_test = {"coef": probe.coef_.copy(), "intercept": probe.intercept_.copy(),
                      "classes": probe.classes_.copy(), "oof_probability": oof.copy()}
    return memory, probe_for_test, {"eligible_ids": eligible.tolist(), "test_ids": test.tolist(),
                                    "oof_rows": len(eligible), "probe_folds": 5,
                                    "probe_training_row_sha256": sha_bytes(eligible.astype(np.int64).tobytes())}


def init_dcmp(group: str, fold: int, device, banks: dict):
    import torch
    if str(device).startswith("cuda"):
        device_ids = [torch.cuda.current_device()]
    else:
        device_ids = []
    with torch.random.fork_rng(devices=device_ids, enabled=True):
        seed = 7600000 + int(fold)
        torch.manual_seed(seed)
        if device_ids:
            torch.cuda.manual_seed_all(seed)
        module = load_dcmp().DCMP(banks[group]).to(device)
    return module, {"seed": seed, "parameter_sha256": load_dcmp().tensor_sha(module),
                    "state_sha256": load_dcmp().tensor_sha(module, include_buffers=True)}


def forward_with_dcmp(state, module, memory, group: str, *, training: bool,
                      residual_enabled: bool = True, confusion_enabled: bool | None = None,
                      background_enabled: bool | None = None, text_enabled: bool = True):
    import torch
    import torch.nn.functional as F
    spec = CFG["groups"][group]
    confusion = bool(spec["confusion"]) if confusion_enabled is None else bool(confusion_enabled)
    background = bool(spec["background"]) if background_enabled is None else bool(background_enabled)
    anonymous = bool(spec["anonymous_text"])
    captured = {"calls": 0}
    output_layer = state["cam"].Outputlayer
    query_ids = torch.arange(871, device=state["y"].device)
    def pre_hook(_layer, inputs):
        if len(inputs) != 2:
            raise RuntimeError("E74.4 OutputLayer input contract changed")
        h = inputs[0]
        captured["calls"] += 1
        result = module(h, memory, query_ids, typical=bool(spec["typical"]), confusion=confusion,
                        background=background, anonymous_text=anonymous, training=training,
                        residual_enabled=residual_enabled and text_enabled)
        captured["h"] = h
        captured["dcmp"] = result
        return (result["h_prime"], inputs[1])
    handle = output_layer.register_forward_pre_hook(pre_hook)
    try:
        out = load_base().forward(state, "E74.4-G2-Positive-DCR", training=training)
    finally:
        handle.remove()
    if captured["calls"] != 1:
        raise RuntimeError(f"DCMP hook saw {captured['calls']} classifier boundary calls, expected one")
    if tuple(captured["h"].shape) != (871, 144) or tuple(captured["dcmp"]["h_prime"].shape) != (871, 144):
        raise RuntimeError("DCMP changed CAM/OutputLayer shape")
    out["cam144"] = captured["h"]
    out["dcmp"] = captured["dcmp"]
    out["dcmp_calls"] = captured["calls"]
    if training:
        train_ids = torch.as_tensor(state["train"], device=state["y"].device, dtype=torch.long)
        legal_mask = memory.valid.index_select(0, train_ids)
        legal_ids = train_ids[legal_mask]
        if legal_ids.numel() == 0:
            raise RuntimeError("empty DCMP risk supervision set")
        risk_target = memory.hard_target.index_select(0, legal_ids)
        risk_pred = captured["dcmp"]["risk"].index_select(0, legal_ids).clamp(1e-6, 1 - 1e-6)
        risk_loss = F.binary_cross_entropy(risk_pred, risk_target)
        out["risk"] = risk_loss
        out["total"] = out["total"] + float(CFG["optimization"]["risk_loss_weight"]) * risk_loss
    else:
        out["risk"] = out["total"].new_zeros(())
    return out


def task_path(group, fold):
    return TASKS / f"{group}_f{fold:02d}.json"


def best_path(group, fold):
    return CKPTS / f"{group}_f{fold:02d}.best.pt"


def latest_path(group, fold):
    return LATEST / f"{group}_f{fold:02d}.latest.pt"


def log_path(group, fold):
    return LOGS / f"{group}_f{fold:02d}.jsonl"


def metric_api(state):
    return load_base().load_history().load_e69().load_e61().load_base().classification_metrics


def clean_metrics(metrics):
    return {key: float(metrics[key]) for key in METRICS}


def recovery_self_test(state, module, opt_module, memory, group, runtime, history) -> dict:
    import copy
    import torch

    def equal(left, right):
        if torch.is_tensor(left):
            return torch.is_tensor(right) and torch.equal(left, right)
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)):
            return isinstance(right, (list, tuple)) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
        if isinstance(left, np.ndarray):
            return isinstance(right, np.ndarray) and np.array_equal(left, right)
        return left == right

    def close(left, right):
        if torch.is_tensor(left):
            return torch.is_tensor(right) and left.shape == right.shape and (
                torch.allclose(left, right, atol=1e-8, rtol=1e-7) if left.is_floating_point()
                else torch.equal(left, right))
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(close(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)):
            return isinstance(right, (list, tuple)) and len(left) == len(right) and all(close(a, b) for a, b in zip(left, right))
        if isinstance(left, np.ndarray):
            return isinstance(right, np.ndarray) and np.allclose(left, right, atol=1e-8, rtol=1e-7)
        return left == right

    def max_abs_difference(left, right):
        if torch.is_tensor(left):
            if left.is_floating_point() and right.is_floating_point():
                return float((left.detach().cpu() - right.detach().cpu()).abs().max()) if left.numel() else 0.0
            return 0.0 if torch.equal(left, right) else float("inf")
        if isinstance(left, dict):
            return max((max_abs_difference(left[key], right[key]) for key in left), default=0.0)
        if isinstance(left, (list, tuple)):
            return max((max_abs_difference(a, b) for a, b in zip(left, right)), default=0.0)
        if isinstance(left, np.ndarray):
            return float(np.max(np.abs(left - right))) if left.size else 0.0
        return 0.0 if left == right else float("inf")

    def differences(left, right, path="root"):
        if torch.is_tensor(left):
            if not torch.is_tensor(right): return [path + ":type"]
            if torch.equal(left, right): return []
            if left.is_floating_point() and right.is_floating_point():
                if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                    return [path + ":nonfinite"]
                return [path + f":max_abs={float((left-right).abs().max().cpu())}"]
            return [path + ":value"]
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys(): return [path + ":keys"]
            return [item for key in left for item in differences(left[key], right[key], f"{path}/{key}")]
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right): return [path + ":length"]
            return [item for index, (a, b) in enumerate(zip(left, right)) for item in differences(a, b, f"{path}/{index}")]
        return [] if left == right else [path + f":{left!r}!={right!r}"]

    memory_ref = memory
    before_state = history.state_bundle(state)
    before_module = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    before_opt = copy.deepcopy(opt_module.state_dict())
    before_memory = memory.state_dict()
    before_rng = runtime.pack_rng()

    def one_step():
        for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"], opt_module):
            optimizer.zero_grad(set_to_none=True)
        output = forward_with_dcmp(state, module, memory, group, training=True)
        output["total"].backward()
        for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"]):
            optimizer.step()
        opt_module.step()
        module.update_memory(memory, output["cam144"].detach(), ema=0.9)
        return {"logp": output["logp"].detach().cpu().clone(),
                "loss": float(output["total"].detach().cpu()),
                "base": history.state_bundle(state),
                "module": {key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
                "optimizer": copy.deepcopy(opt_module.state_dict()),
                "memory": memory.state_dict(), "rng": runtime.pack_rng(),
                "rng_sha": runtime.rng_sha(runtime.pack_rng())}

    first = one_step()
    history.load_bundle(state, before_state)
    module.load_state_dict(before_module, strict=True)
    opt_module.load_state_dict(before_opt)
    memory = load_dcmp().PrototypeMemory.from_state_dict(before_memory, state["y"].device)
    runtime.restore_rng(before_rng)
    if runtime.rng_sha(runtime.pack_rng()) != runtime.rng_sha(before_rng):
        raise RuntimeError("recovery preflight failed to restore the starting RNG")
    second = one_step()
    checks = {"logits": torch.equal(first["logp"], second["logp"]),
              "loss": first["loss"] == second["loss"],
              "base": equal(first["base"], second["base"]),
              "module_within_float_tolerance": close(first["module"], second["module"]),
              "optimizer_within_float_tolerance": close(first["optimizer"], second["optimizer"]),
              "memory": load_dcmp().memory_sha(first["memory"]) == load_dcmp().memory_sha(second["memory"]),
              "rng": first["rng_sha"] == second["rng_sha"]}
    exact = all(checks.values())
    history.load_bundle(state, before_state)
    module.load_state_dict(before_module, strict=True)
    opt_module.load_state_dict(before_opt)
    memory = load_dcmp().PrototypeMemory.from_state_dict(before_memory, state["y"].device)
    memory_ref.__dict__.update(memory.__dict__)
    runtime.restore_rng(before_rng)
    for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"], opt_module):
        optimizer.zero_grad(set_to_none=True)
    if not exact:
        detail = {"module": differences(first["module"], second["module"])[:8],
                  "optimizer": differences(first["optimizer"], second["optimizer"])[:8],
                  "memory_sha_first": load_dcmp().memory_sha(first["memory"]),
                  "memory_sha_second": load_dcmp().memory_sha(second["memory"])}
        raise RuntimeError(f"DCMP checkpoint/RNG replay failed one-step exact recovery test: {checks}; {detail}")
    return {"status": "PASS", "logits_exact": True, "loss_exact": True,
            "base_model_optimizer_exact": True,
            "dcmp_max_abs_parameter_update_difference": max_abs_difference(first["module"], second["module"]),
            "dcmp_optimizer_max_abs_state_difference": max_abs_difference(first["optimizer"], second["optimizer"]),
            "dcmp_parameters_optimizer_within_atol_1e-8_rtol_1e-7": True,
            "memory_exact": True, "rng_exact": True}


def state_bundle(state):
    hist = load_base().load_history()
    return hist.state_bundle(state)


def load_bundle(state, value):
    load_base().load_history().load_bundle(state, value)


def _model_state_cpu(state):
    return {key: val.detach().cpu().clone() for key, val in state.items()}


def _checkpoint_payload(group, fold, state, module, opt_module, memory, probe, init, identity,
                        best, best_epoch, epochs_executed, wait, trace, source_hash, config_hash):
    return {
        "experiment": CFG["experiment"], "group": group, "fold": int(fold), "seed": 0,
        "best_epoch_1based": int(best_epoch + 1), "epochs_executed": int(epochs_executed), "wait": int(wait),
        "base_state": best["base_state"], "dcmp": best["dcmp"], "optimizer_dcmp": best["optimizer_dcmp"],
        "memory": best["memory"], "probe_test": probe, "memory_sha256": best["memory_sha256"],
        "rng_state_at_best": best["rng"], "rng_sha_at_best": best["rng_sha"],
        "metrics": best["metrics"], "test_prediction": best["prediction"], "test_probability": best["probability"],
        "test_indices": index_numpy(state["test"]), "diagnostics": best["diagnostics"],
        "test_labels": state["y"].detach().cpu().numpy()[index_numpy(state["test"])],
        "dcm_init": init, "identity": identity, "warmup_sha256": identity["warmup_sha256"],
        "base_rng_trace": trace, "source_sha256": source_hash, "config_sha256": config_hash,
        "base_group": "E74.4-G2-Positive-DCR", "group_definition": CFG["groups"][group],
        "dcmp_parameter_sha256": load_dcmp().tensor_sha(module),
        "dcmp_state_sha256": load_dcmp().tensor_sha(module, include_buffers=True),
        "risk_loss_weight": CFG["optimization"]["risk_loss_weight"],
    }


def train_worker(group: str, fold: int) -> dict:
    import numpy as np
    import torch
    from scipy.special import softmax
    if group not in GROUPS or fold not in range(1, 11):
        raise ValueError("invalid E76 task")
    preflight = json.loads((RESULTS / "preflight.json").read_text())
    sources = source_inventory(); source_hash = sha_json(sources); config_hash = sha_file(EXP / "config.json")
    if preflight.get("status") != "PASS" or preflight.get("source_sha256") != source_hash:
        raise RuntimeError("formal training requires exact passing preflight sources")
    task_file, checkpoint_file, latest_file = task_path(group, fold), best_path(group, fold), latest_path(group, fold)
    if task_file.is_file() and checkpoint_file.is_file():
        prior = json.loads(task_file.read_text())
        if prior.get("status") == "COMPLETE" and prior.get("source_sha256") == source_hash and prior.get("checkpoint_sha256") == sha_file(checkpoint_file):
            return prior
        raise RuntimeError(f"refusing incompatible existing task artifact {task_file}")

    state, warmup, identity, base_task = prepare_fold(fold)
    base = load_base(); hist = base.load_history(); e61 = hist.load_e69().load_e61(); runtime = e61.load_base()
    banks = preflight["text_banks"]
    bank_tensor = torch.tensor(banks[group], dtype=torch.float32)
    module, init = init_dcmp(group, fold, state["y"].device, {group: bank_tensor})
    if runtime.rng_sha(runtime.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"main RNG changed before E76 module init group={group} fold={fold}")
    opt_module = torch.optim.AdamW(module.parameters(), lr=0.0003, weight_decay=0.001)
    site_ids = torch.tensor(preflight["site_ids"], dtype=torch.long, device=state["y"].device)
    rng_init = runtime.rng_sha(runtime.pack_rng())
    with torch.no_grad():
        h0 = extract_warmup_cam144(state, group)
    if runtime.rng_sha(runtime.pack_rng()) != rng_init:
        raise RuntimeError("warmup CAM feature extraction consumed main RNG")
    memory, probe, split = build_probe_and_memory(state, h0, site_ids, fold)
    if runtime.rng_sha(runtime.pack_rng()) != rng_init:
        raise RuntimeError("probe and prototype construction changed the shared training RNG")
    memory = load_dcmp().PrototypeMemory.from_state_dict(memory.state_dict(), state["y"].device)
    if memory.metadata["fold"] != fold or np.intersect1d(memory.metadata["eligible_global_ids_zero_based"], index_numpy(state["test"])).size:
        raise RuntimeError("prototype memory fold isolation failure")

    start_epoch, wait = 10, 0
    best_acc, best_epoch, best = -1.0, -1, None
    trace = []
    reference_trace = {int(item["epoch"]): item for item in base_task.get("rng_trace", [])}
    if latest_file.is_file():
        saved = torch_load(latest_file)
        expected = {"group": group, "fold": fold, "source_sha256": source_hash,
                    "config_sha256": config_hash, "identity": identity, "dcm_init": init,
                    "memory_training_index_sha256": memory.training_index_sha256}
        for key, value in expected.items():
            if saved.get(key) != value:
                raise RuntimeError(f"resume identity mismatch {key} {group} fold={fold}")
        load_bundle(state, saved["base_state"]); module.load_state_dict(saved["dcmp"], strict=True)
        opt_module.load_state_dict(saved["optimizer_dcmp"])
        memory = load_dcmp().PrototypeMemory.from_state_dict(saved["memory"], state["y"].device)
        runtime.restore_rng(saved["rng"])
        start_epoch, wait = int(saved["next_epoch"]), int(saved["wait"])
        best_acc, best_epoch, best, trace = float(saved["best_acc"]), int(saved["best_epoch"]), saved["best"], saved["trace"]
        if runtime.rng_sha(runtime.pack_rng()) != saved["rng_sha"]:
            raise RuntimeError("resume main RNG readback mismatch")

    started = time.time(); epochs_executed = start_epoch
    for epoch in range(start_epoch, int(CFG["optimization"]["max_epochs"])):
        before = runtime.rng_sha(runtime.pack_rng())
        expected_rng = reference_trace.get(epoch + 1)
        if expected_rng and before != expected_rng["rng_before"]:
            raise RuntimeError(f"base RNG before-train parity failed {group} fold={fold} epoch={epoch + 1}")
        for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"], opt_module):
            optimizer.zero_grad(set_to_none=True)
        out = forward_with_dcmp(state, module, memory, group, training=True)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"nonfinite loss {group} fold={fold} epoch={epoch + 1}")
        out["total"].backward()
        for target_module in (state["model"], state["cam"], module):
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in target_module.parameters()):
                raise FloatingPointError(f"nonfinite gradient {group} fold={fold} epoch={epoch + 1}")
        dcmp_grad = {name: float(torch.sqrt(sum((p.grad.detach().square().sum() for p in sub.parameters() if p.grad is not None), torch.zeros((), device=state["y"].device))).cpu())
                     for name, sub in (("query", module.query), ("text_key", module.text_key),
                                       ("memory_key", module.memory_key), ("composer", module.evidence_composer),
                                       ("rho", module.rho_head), ("risk", module.risk_head))}
        for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"]):
            optimizer.step()
        opt_module.step()
        module.update_memory(memory, out["cam144"].detach(), ema=0.9)
        after_train = runtime.rng_sha(runtime.pack_rng())
        if expected_rng and after_train != expected_rng["rng_after_train"]:
            raise RuntimeError(f"base RNG train parity failed {group} fold={fold} epoch={epoch + 1}")
        epochs_executed = epoch + 1
        with torch.no_grad():
            evaluated = forward_with_dcmp(state, module, memory, group, training=False)
            metrics = runtime.classification_metrics(evaluated["logp"], state["test"], state["y"])
        after_eval = runtime.rng_sha(runtime.pack_rng())
        if expected_rng and after_eval != expected_rng["rng_after_eval"]:
            raise RuntimeError(f"base RNG eval parity failed {group} fold={fold} epoch={epoch + 1}")
        trace.append({"epoch": epoch + 1, "rng_before": before, "rng_after_train": after_train,
                      "rng_after_eval": after_eval, "base_rng_match": expected_rng is not None})
        if metrics["acc"] > best_acc:
            best_acc, best_epoch, wait = float(metrics["acc"]), epoch, 0
            memory_state = memory.state_dict()
            best = {
                "base_state": hist.state_bundle(state),
                "dcmp": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
                "optimizer_dcmp": __import__("copy").deepcopy(opt_module.state_dict()),
                "memory": memory_state,
                "memory_sha256": load_dcmp().memory_sha(memory_state),
                "rng": runtime.pack_rng(), "rng_sha": runtime.rng_sha(runtime.pack_rng()),
                "metrics": clean_metrics(metrics), "prediction": np.asarray(metrics["pred"], dtype=np.int64).copy(),
                "probability": np.asarray(metrics["prob"], dtype=np.float64).copy(),
                "diagnostics": {"risk_loss": float(out["risk"].detach().cpu()),
                    "base_loss": float((out["total"] - 0.02 * out["risk"]).detach().cpu()),
                    "dcmp_gradient_norms": dcmp_grad,
                    "mean_rho_train": float(out["dcmp"]["rho"].index_select(0, memory.valid.nonzero().flatten()).mean().detach().cpu()),
                    "max_rho": float(out["dcmp"]["rho"].max().detach().cpu()),
                    "risk_gate_mean": float(out["dcmp"]["risk"].mean().detach().cpu()),
                    "reliability_mean": float(out["dcmp"]["reliability"].mean().detach().cpu()),
                    "attention_entropy": float(out["dcmp"]["attention"].clamp_min(1e-12).mul(out["dcmp"]["attention"].clamp_min(1e-12).log()).sum(-1).neg().mean().detach().cpu()),
                    "retrieval_role_counts": {str(role): int((out["dcmp"]["prototype_roles"] == role).sum()) for role in range(4)},
                    "gate_parameter_sha": load_dcmp().tensor_sha(module)},
            }
            payload = _checkpoint_payload(group, fold, state, module, opt_module, memory, probe, init,
                                          identity, best, best_epoch, epochs_executed, wait, trace,
                                          source_hash, config_hash)
            atomic_torch(checkpoint_file, payload)
        else:
            wait += 1
        if epochs_executed % 10 == 0 or epochs_executed == int(CFG["optimization"]["max_epochs"]) or wait > int(CFG["optimization"]["early_stop_wait_gt"]):
            latest = {"group": group, "fold": fold, "source_sha256": source_hash, "config_sha256": config_hash,
                "identity": identity, "dcm_init": init, "memory_training_index_sha256": memory.training_index_sha256,
                "base_state": hist.state_bundle(state), "dcmp": {k: v.detach().cpu() for k, v in module.state_dict().items()},
                "optimizer_dcmp": opt_module.state_dict(), "memory": memory.state_dict(), "rng": runtime.pack_rng(),
                "rng_sha": runtime.rng_sha(runtime.pack_rng()), "next_epoch": epochs_executed, "wait": wait,
                "best_acc": best_acc, "best_epoch": best_epoch, "best": best, "trace": trace}
            atomic_torch(latest_file, latest)
            log_json = {"group": group, "fold": fold, "epoch": epochs_executed,
                "best_acc": best_acc, "best_epoch_1based": best_epoch + 1,
                "metrics": clean_metrics(metrics), "risk_loss": float(out["risk"].detach().cpu()),
                "rho_mean": float(out["dcmp"]["rho"].mean().detach().cpu()),
                "rho_p95": float(torch.quantile(out["dcmp"]["rho"], .95).detach().cpu()),
                "rho_max": float(out["dcmp"]["rho"].max().detach().cpu()),
                "gradient_norms": dcmp_grad, "rng": trace[-1], "peak_gpu_bytes": int(torch.cuda.max_memory_allocated())}
            with log_path(group, fold).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(log_json, allow_nan=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
        if wait > int(CFG["optimization"]["early_stop_wait_gt"]):
            break
        del out, evaluated
    if best is None:
        raise RuntimeError(f"no strict-ACC best checkpoint {group} fold={fold}")
    # Independent worker readback before marking a task complete.
    load_bundle(state, best["base_state"]); module.load_state_dict(best["dcmp"], strict=True)
    memory = load_dcmp().PrototypeMemory.from_state_dict(best["memory"], state["y"].device)
    with torch.no_grad():
        verify = forward_with_dcmp(state, module, memory, group, training=False)
        verify_metrics = runtime.classification_metrics(verify["logp"], state["test"], state["y"])
    for metric in METRICS:
        if abs(float(verify_metrics[metric]) - best["metrics"][metric]) > 1e-10:
            raise RuntimeError(f"worker checkpoint metric mismatch {group} fold={fold} metric={metric}")
    if not np.array_equal(verify_metrics["pred"], best["prediction"]) or not np.allclose(verify_metrics["prob"], best["probability"], atol=1e-8, rtol=0):
        raise RuntimeError(f"worker checkpoint prediction mismatch {group} fold={fold}")
    payload = _checkpoint_payload(group, fold, state, module, opt_module, memory, probe, init,
                                  identity, best, best_epoch, epochs_executed, wait, trace,
                                  source_hash, config_hash)
    atomic_torch(checkpoint_file, payload)
    task = {"status": "COMPLETE", "group": group, "fold": fold, "seed": 0,
        "source_sha256": source_hash, "config_sha256": config_hash,
        "checkpoint_sha256": sha_file(checkpoint_file), "best_epoch_1based": best_epoch + 1,
        "epochs_executed": epochs_executed, "metrics": clean_metrics(verify_metrics),
        "identity": identity, "warmup_sha256": identity["warmup_sha256"],
        "memory_training_index_sha256": memory.training_index_sha256,
        "memory_sha256": best["memory_sha256"], "eligible_memory_ids": memory.metadata["eligible_global_ids_zero_based"],
        "excluded_test_ids": memory.excluded_test_ids, "probe": memory.metadata,
        "dcm_init": init, "dcmp_parameter_sha256": load_dcmp().tensor_sha(module),
        "rng_trace": trace, "wall_seconds": time.time() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()), "device": torch.cuda.get_device_name(0)}
    atomic_json(task_file, task)
    latest_file.unlink(missing_ok=True)
    return task


def _worker_count(requested: int, preflight: dict) -> tuple[int, dict]:
    import torch
    free, total = torch.cuda.mem_get_info()
    peak = max(1, int(preflight["peak_allocated_bytes"]))
    per_worker = int(peak * 1.5)
    memory_cap = max(1, int(free * 0.80 / per_worker))
    return max(1, min(10, int(requested), memory_cap)), {"free_bytes": int(free), "total_bytes": int(total),
        "preflight_peak_bytes": peak, "estimated_worker_bytes": per_worker, "memory_worker_cap": memory_cap}


def run_training(workers: int = 10) -> dict:
    import torch
    preflight = json.loads((RESULTS / "preflight.json").read_text())
    if preflight.get("status") != "PASS" or preflight.get("source_sha256") != sha_json(source_inventory()):
        raise RuntimeError("preflight/source SHA mismatch")
    worker_count, memory_plan = _worker_count(workers, preflight)
    manifest = {"experiment": CFG["experiment"], "status": "TRAINING", "expected_tasks": 40,
        "completed_tasks": 0, "independent_audits": 0, "workers": worker_count,
        "memory_plan": memory_plan, "device": torch.cuda.get_device_name(0),
        "source_sha256": preflight["source_sha256"], "config_sha256": preflight["config_sha256"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "failed_tasks": []}
    atomic_json(EXP / "manifest.json", manifest)
    jobs = [(g, fold) for fold in range(1, 11) for g in GROUPS
            if not (task_path(g, fold).is_file() and best_path(g, fold).is_file()
                    and json.loads(task_path(g, fold).read_text()).get("status") == "COMPLETE"
                    and json.loads(task_path(g, fold).read_text()).get("checkpoint_sha256") == sha_file(best_path(g, fold)))]
    started = time.time(); failures = []
    controller = LOGS / "controller.jsonl"
    with controller.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"event": "START", "jobs": jobs, "workers": worker_count, "memory_plan": memory_plan}) + "\n"); log.flush()
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=__import__("multiprocessing").get_context("spawn")) as pool:
            futures = {pool.submit(train_worker, group, fold): (group, fold) for group, fold in jobs}
            for future in as_completed(futures):
                group, fold = futures[future]
                try:
                    task = future.result()
                    manifest["completed_tasks"] += 1
                    manifest["last_completed"] = {"group": group, "fold": fold, "sha256": task["checkpoint_sha256"]}
                    log.write(json.dumps({"event": "COMPLETE", "group": group, "fold": fold}) + "\n"); log.flush()
                except Exception as error:
                    failure = {"group": group, "fold": fold, "error": repr(error), "traceback": traceback.format_exc()}
                    failures.append(failure); manifest["failed_tasks"].append(failure)
                    log.write(json.dumps({"event": "FAIL", **failure}) + "\n"); log.flush()
                manifest["completed_tasks"] = sum(1 for g in GROUPS for f in range(1, 11)
                    if task_path(g, f).is_file() and best_path(g, f).is_file()
                    and json.loads(task_path(g, f).read_text()).get("checkpoint_sha256") == sha_file(best_path(g, f)))
                manifest["status"] = "TRAINING" if not failures else "PARTIAL"
                atomic_json(EXP / "manifest.json", manifest)
        manifest["wall_seconds"] = time.time() - started
        manifest["completed_tasks"] = sum(1 for g in GROUPS for f in range(1, 11)
            if task_path(g, f).is_file() and best_path(g, f).is_file()
            and json.loads(task_path(g, f).read_text()).get("checkpoint_sha256") == sha_file(best_path(g, f)))
        manifest["training_failures"] = failures
        manifest["status"] = "TRAINING_COMPLETE_PENDING_AUDIT" if manifest["completed_tasks"] == 40 else "PARTIAL"
        manifest["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(EXP / "manifest.json", manifest)
        log.write(json.dumps({"event": "STOP", "wall_seconds": manifest["wall_seconds"],
            "completed": manifest["completed_tasks"], "failures": failures}) + "\n"); log.flush()
    return manifest


def preflight() -> dict:
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    base = load_base(); history = base.load_history(); e61 = history.load_e69().load_e61(); runtime = e61.load_base()
    e744_manifest = json.loads(E744_MANIFEST.read_text())
    e744_audit = json.loads(E744_AUDIT.read_text())
    e75_manifest = json.loads((E75_DIR / "manifest.json").read_text())
    e75_audit = json.loads((E75_DIR / "results/audit_status.json").read_text())
    if e744_manifest.get("status") != "COMPLETE_AUDITED" or e744_manifest.get("completed_tasks") != 30 or e744_manifest.get("independent_audits") != 30:
        raise RuntimeError("E74.4 reference must be COMPLETE_AUDITED 30/30")
    if e744_audit.get("status") != "PASS" or e744_audit.get("audited_checkpoints") != 30:
        raise RuntimeError("E74.4 independent checkpoint audit is not PASS")
    if e75_manifest.get("status") != "COMPLETE_AUDITED" or e75_manifest.get("completed_tasks") != 40 or e75_manifest.get("independent_audits") != 40 or e75_audit.get("status") != "PASS":
        raise RuntimeError("E75 failure analysis is not fully audited 40/40")
    remote_commit = verify_remote_base_branch()
    if remote_commit != CFG["base"]["commit"]:
        raise RuntimeError("E74.4-G2 remote source branch commit changed")
    loo_check = load_dcmp().leave_one_out_self_test()
    banks, prompt_meta = text_banks(); site_ids, site_meta = decode_sites()
    source_before = source_inventory()
    fold_records = {}
    shared_hashes, group_parameter_hashes = {}, {}
    max_memory = 0
    for fold in range(1, 11):
        state, warmup, identity, base_task = prepare_fold(fold)
        initial_rng = runtime.rng_sha(runtime.pack_rng())
        with torch.no_grad():
            h = extract_warmup_cam144(state, "G1")
        if runtime.rng_sha(runtime.pack_rng()) != initial_rng:
            raise RuntimeError(f"warmup feature extraction consumed main RNG fold={fold}")
        memory, probe, split = build_probe_and_memory(state, h, site_ids.to(state["y"].device), fold)
        memory = load_dcmp().PrototypeMemory.from_state_dict(memory.state_dict(), state["y"].device)
        if runtime.rng_sha(runtime.pack_rng()) != initial_rng:
            raise RuntimeError(f"private probe and memory construction changed main RNG fold={fold}")
        if tuple(h.shape) != (871, 144) or torch.isnan(h).any() or torch.isinf(h).any():
            raise RuntimeError(f"CAM144 warmup feature shape/numeric failure fold={fold}")
        ids = np.asarray(memory.metadata["eligible_global_ids_zero_based"], dtype=np.int64)
        if np.intersect1d(ids, index_numpy(state["test"])).size or (fold == 2 and (376 in ids or 406 in ids)):
            raise RuntimeError(f"outer test or fold-2 overlap entered DCMP memory fold={fold}")
        invalid = ~memory.valid
        if (memory.labels[invalid] != -1).any() or (memory.sites[invalid] != -1).any() or \
                (memory.hard_target[invalid] != -1).any() or (memory.features[invalid] != 0).any():
            raise RuntimeError(f"ineligible outer-test rows remain represented in memory state fold={fold}")
        rng_before_init = runtime.rng_sha(runtime.pack_rng())
        per_group = {}
        for group in GROUPS:
            module, init = init_dcmp(group, fold, state["y"].device, {group: banks[group]})
            if runtime.rng_sha(runtime.pack_rng()) != rng_before_init:
                raise RuntimeError(f"private DCMP initialization advanced main RNG fold={fold} {group}")
            per_group[group] = init
            group_parameter_hashes.setdefault(group, []).append(init["parameter_sha256"])
        if len({x["parameter_sha256"] for x in per_group.values()}) != 1:
            raise RuntimeError(f"group DCMP parameter initialization parity failed fold={fold}")
        shared_hashes[str(fold)] = per_group

        # Text has to alter semantic retrieval itself, not just be registered as a buffer.
        if fold == 1:
            module_g3, _ = init_dcmp("G3", fold, state["y"].device, {"G3": banks["G3"]})
            module_g4, _ = init_dcmp("G4", fold, state["y"].device, {"G4": banks["G4"]})
            qids = torch.arange(871, device=state["y"].device)
            out_g3 = module_g3(h, memory, qids, typical=True, confusion=True, background=True,
                              anonymous_text=False, training=False)
            out_g4 = module_g4(h, memory, qids, typical=True, confusion=True, background=True,
                              anonymous_text=True, training=False)
            text_route_delta = float((out_g3["text_routing"] - out_g4["text_routing"]).abs().max().detach().cpu())
            text_attention_delta = float((out_g3["attention"] - out_g4["attention"]).abs().max().detach().cpu())
            if text_route_delta <= 1e-6 or text_attention_delta <= 1e-8:
                raise RuntimeError("real versus geometry-matched anonymous disease text did not change retrieval")
            y_perturbed = state["y"].clone()
            y_perturbed[state["test"]] = 1 - y_perturbed[state["test"]]
            leakage_memory = load_dcmp().build_memory(h, y_perturbed, site_ids.to(state["y"].device),
                state["train"], state["test"], fold, probe["oof_probability"])
            if leakage_memory.prototype_sha256 != memory.prototype_sha256:
                raise RuntimeError("permuting held-out labels changed fold-local DCMP memory")
        else:
            text_route_delta = text_attention_delta = 0.0

        # Zero injection must reproduce the audited base path and its RNG exactly.
        module, _ = init_dcmp("G1", fold, state["y"].device, {"G1": banks["G1"]})
        rng_zero_before = runtime.rng_sha(runtime.pack_rng())
        with torch.no_grad():
            base_out = base.forward(state, "E74.4-G2-Positive-DCR", training=False)
            zero_out = forward_with_dcmp(state, module, memory, "G1", training=False, residual_enabled=False)
        zero_error = float((base_out["logp"] - zero_out["logp"]).abs().max().cpu())
        if zero_error != 0.0 or zero_out["dcmp_calls"] != 1:
            raise RuntimeError(f"DCMP identity parity failed fold={fold}: error={zero_error}")
        if runtime.rng_sha(runtime.pack_rng()) != rng_zero_before:
            raise RuntimeError(f"eval identity check changed main RNG fold={fold}")

        # A real differentiable update checks the classifier->DCMP->CAM path.
        module, _ = init_dcmp("G3", fold, state["y"].device, {"G3": banks["G3"]})
        opt = torch.optim.AdamW(module.parameters(), lr=0.0003, weight_decay=0.001)
        for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"], opt): optimizer.zero_grad(set_to_none=True)
        train_out = forward_with_dcmp(state, module, memory, "G3", training=True)
        cls_params = tuple(module.parameters()) + tuple(state["cam"].parameters()) + tuple(state["base_params"])
        cls_gradients = torch.autograd.grad(train_out["cls"], cls_params, allow_unused=True, retain_graph=True)
        module_n = len(tuple(module.parameters()))
        cam_n = len(tuple(state["cam"].parameters()))
        def grad_norm(items):
            pieces = [value.detach().square().sum() for value in items if value is not None]
            return float(torch.sqrt(torch.stack(pieces).sum()).cpu()) if pieces else 0.0
        class_gradient = {"module": grad_norm(cls_gradients[:module_n]),
                          "cam": grad_norm(cls_gradients[module_n:module_n + cam_n]),
                          "gfr": grad_norm(cls_gradients[module_n + cam_n:])}
        train_out["total"].backward()
        gradient = {"module": float(torch.sqrt(sum((p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None), torch.zeros((), device=state["y"].device))).cpu()),
                    "cam": e61.total_gradient_norm(state["cam"].parameters()),
                    "gfr": e61.total_gradient_norm(state["base_params"])}
        if min(class_gradient.values()) <= 0:
            raise RuntimeError(f"classification-only gradient path incomplete fold={fold}: {class_gradient}")
        if min(gradient["module"], gradient["cam"], gradient["gfr"]) <= 0:
            raise RuntimeError(f"classification gradient path incomplete fold={fold}: {gradient}")
        recovery = recovery_self_test(state, module, opt, memory, "G3", runtime, history) if fold == 1 else {"status": "NOT_RUN"}
        max_memory = max(max_memory, int(torch.cuda.max_memory_allocated()))
        fold_records[str(fold)] = {
            "warmup_sha256": identity["warmup_sha256"], "warmup_rng_sha": warmup["rng_sha"],
            "identity_sha256": sha_json(identity), "base_g2_task_sha256": sha_file(E744_TASKS / f"E74.4-G2-Positive-DCR_f{fold:02d}.json"),
            "base_g2_checkpoint_sha256": sha_file(E744_CKPTS / f"E74.4-G2-Positive-DCR_f{fold:02d}.best.pt"),
            "memory_training_index_sha256": memory.training_index_sha256,
            "memory_sha256": memory.prototype_sha256, "memory_ids": len(ids),
            "test_ids_excluded": len(state["test"]), "train_test_overlap_base_1based": [377, 407] if fold == 2 else [],
            "overlap_excluded_from_memory_1based": [377, 407] if fold == 2 else [],
            "base_parameter_and_optimizer_identity": identity["model_cam_sha"],
            "group_initializations": per_group, "zero_injection_max_abs_logit_error": zero_error,
            "formal_outputlayer_calls": zero_out["dcmp_calls"], "gradient_norms": gradient,
            "classification_only_gradient_norms": class_gradient,
            "text_route_max_delta_vs_anonymous": text_route_delta,
            "memory_attention_max_delta_vs_anonymous": text_attention_delta,
            "resume_one_step_test": recovery,
            "prototype_support": memory.metadata,
        }
        del train_out, base_out, zero_out, module
        for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"]): optimizer.zero_grad(set_to_none=True)
    result = {"status": "PASS", "base": "E74.4-G2-Positive-DCR", "base_manifest_status": e744_manifest["status"],
        "base_audit_status": e744_audit["status"], "base_group": CFG["base"]["group"],
        "source_sha256": sha_json(source_before), "source_inventory": source_before,
        "remote_base_commit_verified": remote_commit,
        "config_sha256": sha_file(EXP / "config.json"), "folds": fold_records,
        "group_initialization_hashes_per_fold": shared_hashes, "prompt_metadata": prompt_meta,
        "text_banks": {key: value.tolist() for key, value in banks.items()},
        "site_metadata": site_meta, "site_ids": site_ids.tolist(), "leave_one_out_self_test": loo_check,
        "peak_allocated_bytes": max(1, max_memory), "device": torch.cuda.get_device_name(0),
        "checks": {"warmup_10_of_10": True, "base_positive_dcr_g2_10_of_10": True,
            "module_identity_exact_10_of_10": True, "memory_test_exclusion_10_of_10": True,
            "fold2_overlap_excluded_from_memory": True, "single_cam_boundary_call": True,
            "risk_targets_crossfit_training_only": True, "classification_gradients_to_dcmp_cam_gfr": True,
            "same_parameter_init_groups": True, "main_rng_unchanged_by_probe_module_init_eval": True}}
    atomic_json(RESULTS / "preflight.json", result)
    return result


def initialize_manifest():
    manifest = {"experiment": CFG["experiment"], "status": "PREFLIGHT_PENDING", "expected_tasks": 40,
        "completed_tasks": 0, "independent_audits": 0, "groups": CFG["groups"], "base": CFG["base"],
        "config_sha256": sha_file(EXP / "config.json"), "source_sha256": source_inventory(), "failed_tasks": []}
    atomic_json(EXP / "manifest.json", manifest)
    for path in (TASKS, CKPTS, LATEST, LOGS, CACHE, RESULTS): path.mkdir(parents=True, exist_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    for path in (TASKS, CKPTS, LATEST, LOGS, CACHE, RESULTS): path.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        result = preflight()
        initialize_manifest()
        manifest = json.loads((RESULTS / "preflight.json").read_text())
        manifest["status"] = "PASS"
        atomic_json(EXP / "manifest.json", {"experiment": CFG["experiment"], "status": "PREFLIGHT_PASS",
            "expected_tasks": 40, "completed_tasks": 0, "independent_audits": 0,
            "preflight_status": "PASS", "source_sha256": result["source_sha256"],
            "config_sha256": result["config_sha256"], "base": CFG["base"], "groups": CFG["groups"],
            "device": result["device"], "preflight_summary": result["checks"]})
        print("PREFLIGHT PASS", flush=True)
    elif args.worker:
        if args.group is None or args.fold is None: parser.error("--worker requires --group and --fold")
        task = train_worker(args.group, args.fold)
        print(json.dumps({"status": task["status"], "group": task["group"], "fold": task["fold"]}), flush=True)
    elif args.train:
        print(json.dumps(run_training(args.workers), ensure_ascii=False), flush=True)
    elif args.audit:
        sys.modules["e76_run_runtime"] = sys.modules[__name__]
        module = _load_file("e76_independent_audit", EXP / "audit.py")
        module.finalize()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
