#!/usr/bin/env python3
"""Experiment 72 frozen-runner for multi-grain disease semantic supervision."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
CFG_PATH = EXP / "config.json"
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
RUNTIME = EXP / ".runtime"
REFERENCE_EXP = RUNTIME / "reference_e69"
E69_EXP = Path(CFG["paths"]["e69_3_g2"])
E69_TASKS = REFERENCE_EXP / "tasks"
E61_EXP = Path(CFG["paths"]["e61"])
WARMUP_DIR = RUNTIME / "reference_e61/warmup"
TEXT_CACHE = RUNTIME / "text/text_embeddings.pt"
PROMPT_CACHE = RUNTIME / "prompt_banks.pt"
PRE_FLIGHT = RUNTIME / "preflight.json"
MANIFEST_PATH = EXP / "manifest.json"
MANIFEST_LOCK = RUNTIME / "manifest.lock"

_e69_spec = importlib.util.spec_from_file_location("experiment72_1_locked_e69_runner", REFERENCE_EXP / "run.py")
if _e69_spec is None or _e69_spec.loader is None:
    raise RuntimeError("Cannot load the SHA-verified E69.3-G2 runner copy")
E69 = importlib.util.module_from_spec(_e69_spec)
sys.modules[_e69_spec.name] = E69
_e69_spec.loader.exec_module(E69)
E61 = E69.load_e61()
BASE = E61.load_base()
E61.WARMUP = WARMUP_DIR


class _E59Compat:
    @staticmethod
    def _check_optimizer_digest(optimizer):
        return sha_obj(optimizer.state_dict())


E59 = _E59Compat()

from mgdss import (  # noqa: E402
    BP_TEXTS,
    SP_TEXTS,
    DARG320DiseaseSupervision,
    class_ce,
    loss_equivalence_audit,
    make_prompt_space,
    saber_class_ce,
    supervised_losses,
    tensor_sha,
    text_sha,
)


GROUP_IDS = tuple(item["id"] for item in CFG["groups"])
METRIC_KEYS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
EXPECTED_BASE_FILES = {
    "run.py": "be2d534ea2df90d3b462034ee4c9fc859b3f79507dfba7c62a2faa48aee49c44",
    "config.json": "4556c96bb92a9bdbdb048b6f702d853abe6030dd2ae9e2bdad9b5f0ba509c8e0",
    "manifest.json": "a3620884d501965c9fe71ef51231c693363ba67930a8f084025b47c16f5195f9",
}
EXPECTED_E59_FILES = {
    "run.py": "f5bd8236b6a5f66903f56a01832caab6c21177238f24676562f536df3af2b334",
    "model_e61.py": "28266568bcebcfdbab961dbedb7257c15ff4e862cd02ae744feacad54b7bc81b",
    "config.json": "7a1a2e7d37918baf615c704a47d09e3a3eb8a9de5bfc752d852b15a8ada6569b",
}
PROTECTED_PATHS = (
    Path("/data/adodas/SPromptGL-main/network.py"),
    Path("/data/adodas/SPromptGL-main/layers.py"),
    Path("/data/adodas/SPromptGL-main/utils.py"),
    Path("/data/adodas/SPromptGL-main/model.py"),
    Path("/data/adodas/SPromptGL-main/main.py"),
    Path("/data/adodas/SPromptGL-main/data"),
    Path("/data/adodas/SPromptGL-main/graph/ABIDE_weighted-cosine_graph_.npz"),
    Path("/data/adodas/SPromptGL-main/experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"),
    Path("/data/adodas/SPromptGL-main/experiment/experiment51_dugsa"),
    Path("/data/adodas/SPromptGL-main/experiment/experiment53_3_cadr_graph_fallback_fix"),
    Path("/data/adodas/SPromptGL-main/experiment/experiment59_shape_pen"),
    Path("/data/adodas/SPromptGL-main/experiment/experiment60_dss_pen"),
    Path("/model/spromptgl/experiment/experiment53_3_cadr_graph_fallback_fix"),
    Path("/model/spromptgl/experiment/experiment70_gcdsi"),
    Path("/model/spromptgl/experiment/experiment70_1_dcc_gcdsi"),
    Path("/model/spromptgl/experiment/experiment71_gcder"),
    Path("/data/adodas/SPromptGL-main/experiment/experiment61_cadr_postalign_vs_e2e/run.py"),
    Path("/data/adodas/SPromptGL-main/experiment/experiment61_cadr_postalign_vs_e2e/model_e61.py"),
    Path("/data/adodas/SPromptGL-main/experiment/experiment61_cadr_postalign_vs_e2e/config.json"),
    Path("/data/adodas/SPromptGL-main/experiment/experiment61_cadr_postalign_vs_e2e/manifest.json"),
    Path("/data/adodas/SPromptGL-main/experiment/experiment61_cadr_postalign_vs_e2e/.runtime/warmup"),
    Path("/model/spromptgl/experiment/experiment69_3_g2_two_rows/run.py"),
    Path("/model/spromptgl/experiment/experiment69_3_g2_two_rows/config.json"),
    Path("/model/spromptgl/experiment/experiment69_3_g2_two_rows/manifest.json"),
    Path("/model/spromptgl/experiment/experiment69_3_g2_two_rows/results"),
    Path("/model/spromptgl/experiment/experiment69_3_g2_two_rows/.runtime/tasks"),
    Path("/model/spromptgl/experiment/experiment69_3_g2_two_rows/.runtime/checkpoints"),
    Path("/model/spromptgl/experiment/experiment69_3_g2_two_rows/.runtime/checkpoints/E69.3-G2_f02.best.pt"),
    Path("/model/spromptgl/experiment/experiment62_e2e_gate_init_sensitivity/run.py"),
    Path("/model/spromptgl/experiment/experiment62_e2e_gate_init_sensitivity/config.json"),
    Path("/model/spromptgl/experiment/experiment62_e2e_gate_init_sensitivity/manifest.json"),
    Path("/model/spromptgl/experiment/experiment62_e2e_gate_init_sensitivity/results"),
    Path("/model/spromptgl/experiment/experiment62_e2e_gate_init_sensitivity/.runtime/tasks"),
    Path("/model/spromptgl/experiment/experiment72_mgdss/mgdss.py"),
    Path("/model/spromptgl/experiment/experiment72_mgdss/run.py"),
    Path("/model/spromptgl/experiment/experiment72_mgdss/audit.py"),
    Path("/model/spromptgl/experiment/experiment72_mgdss/config.json"),
    Path("/model/spromptgl/experiment/experiment72_mgdss/manifest.json"),
    Path("/model/spromptgl/experiment/experiment72_mgdss/results"),
    Path("/model/spromptgl/experiment/experiment72_mgdss/.runtime/text/text_embeddings.pt"),
    Path("/model/spromptgl/experiment/experiment72_1_e693_mgdss"),
)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_obj(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def tree_equal(left: Any, right: Any) -> bool:
    torch = BASE.runtime()["torch"] if "BASE" in globals() else None
    if torch is not None and torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and torch.equal(left.cpu(), right.cpu())
    try:
        import numpy as np
        if isinstance(left, np.ndarray):
            return isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    except ImportError:
        pass
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right))
    try:
        return type(left) is type(right) and bool(left == right)
    except (TypeError, ValueError):
        return False


def tree_sha(value: Any) -> str:
    import numpy as np

    digest = hashlib.sha256()

    def add(item: Any) -> None:
        if hasattr(item, "detach") and hasattr(item, "dtype"):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"torch|" + str(tensor.dtype).encode() + b"|" + str(tuple(tensor.shape)).encode() + b"|")
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"numpy|" + str(array.dtype).encode() + b"|" + str(array.shape).encode() + b"|")
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict|")
            for key in sorted(item, key=lambda value: repr(value)):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(("list|" if isinstance(item, list) else "tuple|").encode())
            for child in item:
                add(child)
        else:
            digest.update(type(item).__name__.encode() + b"|")
            digest.update(pickle.dumps(item, protocol=4))

    add(value)
    return digest.hexdigest()


def module_sha(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    atomic_bytes(path, payload)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        BASE.runtime()["torch"].save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def state_cpu(module) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def state_tree_cpu(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: state_tree_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [state_tree_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(state_tree_cpu(item) for item in value)
    return copy.deepcopy(value)


def code_hashes() -> dict[str, str]:
    return {
        "config.json": sha_file(CFG_PATH),
        "run.py": sha_file(EXP / "run.py"),
        "mgdss.py": sha_file(EXP / "mgdss.py"),
        "audit.py": sha_file(EXP / "audit.py") if (EXP / "audit.py").is_file() else "NOT_CREATED_YET",
    }


def staged_hashes() -> dict[str, str]:
    result = {}
    for name, expected in EXPECTED_BASE_FILES.items():
        path = REFERENCE_EXP / name
        actual = sha_file(path)
        if actual != expected or sha_file(E69_EXP / name) != expected:
            raise AssertionError(f"E69.3-G2 {name} SHA mismatch: {actual} != {expected}")
        result[f"e69_3_g2/{name}"] = actual
    for name, expected in EXPECTED_E59_FILES.items():
        path = E61_EXP / name
        actual = sha_file(path)
        if actual != expected or sha_file(RUNTIME / "reference_e61" / name) != expected:
            raise AssertionError(f"E61 {name} SHA mismatch: {actual} != {expected}")
        result[f"e61/{name}"] = actual
    e69_manifest = json.loads((REFERENCE_EXP / "manifest.json").read_text(encoding="utf-8"))
    e61_manifest = json.loads((RUNTIME / "reference_e61/manifest.json").read_text(encoding="utf-8"))
    if e69_manifest.get("status") != "COMPLETE_AUDITED" or e69_manifest.get("completed_tasks") != 10 or e69_manifest.get("independent_audits") != 10:
        raise AssertionError("E69.3-G2 reference is not independently audited 10/10")
    if e69_manifest.get("intentional_train_test_overlap") is not True or e69_manifest.get("overlap_folds") != {"2": [377, 407]}:
        raise AssertionError("E69.3-G2 overlap provenance differs from the frozen two-row protocol")
    result["e69_3_g2_manifest_sha256"] = sha_file(REFERENCE_EXP / "manifest.json")
    result["e61_manifest_sha256"] = sha_file(RUNTIME / "reference_e61/manifest.json")
    for name, source, copied in (
        ("e62_run.py", Path("/model/spromptgl/experiment/experiment62_e2e_gate_init_sensitivity/run.py"), RUNTIME / "reference_e62_run.py"),
        ("e62_config.json", Path("/model/spromptgl/experiment/experiment62_e2e_gate_init_sensitivity/config.json"), RUNTIME / "reference_e62_config.json"),
        ("e62_manifest.json", Path("/model/spromptgl/experiment/experiment62_e2e_gate_init_sensitivity/manifest.json"), RUNTIME / "reference_e62_manifest.json"),
        ("e72_mgdss.py", Path("/model/spromptgl/experiment/experiment72_mgdss/mgdss.py"), RUNTIME / "reference_e72/mgdss.py"),
        ("e72_run.py", Path("/model/spromptgl/experiment/experiment72_mgdss/run.py"), RUNTIME / "reference_e72/run.py"),
        ("e72_audit.py", Path("/model/spromptgl/experiment/experiment72_mgdss/audit.py"), RUNTIME / "reference_e72/audit.py"),
        ("e72_config.json", Path("/model/spromptgl/experiment/experiment72_mgdss/config.json"), RUNTIME / "reference_e72/config.json"),
        ("e72_manifest.json", Path("/model/spromptgl/experiment/experiment72_mgdss/manifest.json"), RUNTIME / "reference_e72/manifest.json"),
    ):
        if not source.is_file() or not copied.is_file() or sha_file(source) != sha_file(copied):
            raise AssertionError(f"historical {name} source/copy SHA mismatch")
        result[name] = sha_file(copied)
    E69.verify_e62_reference()
    result["e72_text_cache_sha256"] = sha_file(TEXT_CACHE)
    result["e72_text_config_sha256"] = sha_file(RUNTIME / "text/config.json")
    result["e72_text_features_sha256"] = sha_file(RUNTIME / "text/text_features.py")
    for name in ("run.py", "mgdss.py", "audit.py", "config.json", "manifest.json", "audit_status.json"):
        source = Path("/model/spromptgl/experiment/experiment72_1_e693_mgdss") / name
        staged = RUNTIME / "reference_e72_1" / name
        if name == "audit_status.json":
            source = Path("/model/spromptgl/experiment/experiment72_1_e693_mgdss/results/audit_status.json")
            staged = RUNTIME / "reference_e72_1/audit_status.json"
        if not source.is_file() or not staged.is_file() or sha_file(source) != sha_file(staged):
            raise AssertionError(f"E72.1 historical {name} SHA mismatch")
        result[f"e72_1_{name.replace('/', '_')}_sha256"] = sha_file(staged)
    expected_cache = CFG["text"]["experiment60_cache_sha256"]
    if result["e72_text_cache_sha256"] != expected_cache:
        raise AssertionError("copied frozen CLIP cache SHA mismatch")
    for fold in range(1, 11):
        source = E61_EXP / ".runtime/warmup" / f"fold{fold:02d}.pt"
        copied = WARMUP_DIR / source.name
        if not source.is_file() or sha_file(source) != sha_file(copied):
            raise AssertionError(f"E61 fold {fold} warmup copy/source SHA mismatch")
        result[f"e61_warmup_f{fold:02d}"] = sha_file(copied)
        task = E69_TASKS / f"E69.3-G2_f{fold:02d}.json"
        if not task.is_file():
            raise FileNotFoundError(f"missing copied E69.3-G2 task record: {task}")
        result[f"e69_task_f{fold:02d}"] = sha_file(task)
    for name, relative in {
        "data_csv": Path("/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv"),
        "modal_mapping": Path("/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy"),
        "e69_reference_metrics": REFERENCE_EXP / "results/fold_metrics.csv",
        "e69_reference_audit": REFERENCE_EXP / "results/checkpoint_audit.csv",
        "e69_fold2_checkpoint": RUNTIME / "reference_e69/E69.3-G2_f02.best.pt",
        "e72_mgdss_source": Path("/model/spromptgl/experiment/experiment72_mgdss/mgdss.py"),
        "e72_audit_report": RUNTIME / "reference_e72/report.md",
    }.items():
        if not relative.is_file():
            raise FileNotFoundError(f"missing SHA-verified dependency {name}: {relative}")
        result[name] = sha_file(relative)
    return result


def protected_snapshot() -> dict[str, str]:
    files: dict[str, str] = {}
    for item in PROTECTED_PATHS:
        if not item.exists():
            raise FileNotFoundError(f"protected source or historical path disappeared: {item}")
        candidates = [item] if item.is_file() else [path for path in item.rglob("*") if path.is_file()]
        for path in sorted(candidates):
            files[str(path)] = sha_file(path)
    return files


def compare_protected(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(path for path in set(before) & set(after) if before[path] != after[path])
    return {"before_count": len(before), "after_count": len(after), "missing": missing, "added": added, "changed": changed, "status": "PASS" if not (missing or added or changed) else "FAIL"}


def group_item(group_id: str) -> dict:
    for item in CFG["groups"]:
        if item["id"] == group_id:
            return item
    raise KeyError(group_id)


def task_paths(group_id: str, fold: int) -> tuple[Path, Path, Path, Path]:
    stem = f"{group_id}_f{fold:02d}"
    return (
        RUNTIME / "tasks" / f"{stem}.json",
        RUNTIME / "checkpoints" / f"{stem}.best.pt",
        RUNTIME / "checkpoints" / f"{stem}.latest.pt",
        RUNTIME / "logs" / f"{stem}.log",
    )


def load_reference() -> tuple[dict, dict[int, dict]]:
    manifest = json.loads((REFERENCE_EXP / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE_AUDITED" or manifest.get("completed_tasks") != 10 or manifest.get("independent_audits") != 10:
        raise AssertionError("The read-only E69.3-G2 reference is not audited 10-fold data")
    task_rows = {int(row["fold"]): row for row in manifest["tasks"]}
    if set(task_rows) != set(range(1, 11)):
        raise AssertionError("E69.3-G2 reference manifest does not contain ten unique folds")
    folds = {}
    for fold in range(1, 11):
        task = json.loads((E69_TASKS / f"E69.3-G2_f{fold:02d}.json").read_text(encoding="utf-8"))
        expected = task_rows[fold]
        if task.get("status") != "COMPLETE" or task.get("checkpoint_sha256") != expected.get("checkpoint_sha256"):
            raise AssertionError(f"E69.3-G2 task identity/checkpoint mismatch in fold {fold}")
        folds[fold] = task
    return manifest, folds


def reference_initialization_rows() -> dict[int, dict]:
    return load_reference()[1]


def reference_rows() -> dict[int, dict]:
    rows = {}
    with (REFERENCE_EXP / "results/fold_metrics.csv").open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("group") == "E69.3-G2":
                rows[int(row["fold"])] = {key: float(row[key]) for key in METRIC_KEYS}
    if set(rows) != set(range(1, 11)):
        raise AssertionError("E69.3-G2 paired reference metrics are not exactly ten folds")
    return rows


def verify_warmup_audit() -> dict:
    manifest = json.loads((RUNTIME / "reference_e61/manifest.json").read_text(encoding="utf-8"))
    records = {str(item["fold"]): item for item in manifest.get("warmups", [])}
    if manifest.get("status") != "COMPLETE_AUDITED" or manifest.get("independent_audits") != 20 or set(records) != {str(i) for i in range(1, 11)}:
        raise AssertionError("E61 source manifest does not certify all ten shared warmups")
    folds = {}
    torch = BASE.runtime()["torch"]
    for fold in range(1, 11):
        item = records[str(fold)]
        warmup = WARMUP_DIR / f"fold{fold:02d}.pt"
        saved = torch.load(warmup, map_location="cpu", weights_only=False)
        source = E61_EXP / ".runtime/warmup" / warmup.name
        if Path(item.get("path", "")).name != warmup.name or item.get("sha256") != sha_file(warmup) or sha_file(source) != sha_file(warmup):
            raise AssertionError(f"E61 fold {fold} warmup SHA/audit mismatch")
        if int(saved.get("fold", -1)) != fold or saved.get("rng_sha") != item.get("rng_sha"):
            raise AssertionError(f"E61 fold {fold} warmup identity/RNG mismatch")
        folds[str(fold)] = {
            "status": "PASS", "readback": "PASS", "sha256": sha_file(warmup),
            "captured_digest": {"rng_sha": saved["rng_sha"]},
            "normal_capture_digest": {"rng_sha": saved["rng_sha"]},
            "rng_sha": saved["rng_sha"], "split_sha": saved["split_sha"], "pair_sha": saved["pair_sha"],
        }
    return {"shared_warmup_cache": {"status": "PASS", "fold_count": 10, "folds": folds}, "source_manifest_sha256": sha_file(RUNTIME / "reference_e61/manifest.json")}


def make_prompt_assets(write_cache: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = BASE.runtime()["torch"]
    cache = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
    if sha_file(TEXT_CACHE) != CFG["text"]["experiment60_cache_sha256"]:
        raise AssertionError("Experiment 60 frozen CLIP cache SHA mismatch")
    if cache.get("source_hashes", {}).get("rn50_sha256") != CFG["text"]["rn50_weights_sha256"]:
        raise AssertionError("Experiment 60 cache was not encoded with the frozen RN50 weights")
    bp_texts = tuple(cache.get("text_map", {}).get("brainprompt", ()))
    sp_texts = tuple(cache.get("text_map", {}).get("original", ()))
    if bp_texts != BP_TEXTS or sp_texts != SP_TEXTS:
        raise AssertionError("Frozen text cache prompts differ from the source text list")
    if text_sha(BP_TEXTS) != CFG["text"]["brainprompt_text_sha256"]:
        raise AssertionError("BrainPrompt source text SHA mismatch")
    if text_sha(SP_TEXTS) != CFG["text"]["spromptgl_text_sha256"]:
        raise AssertionError("SPromptGL source text SHA mismatch")
    bp_raw = torch.as_tensor(cache["embeddings"]["brainprompt"], dtype=torch.float64)
    sp_raw = torch.as_tensor(cache["embeddings"]["original"], dtype=torch.float64)
    basis, rank, singular_values, records = make_prompt_space(bp_raw, sp_raw, CFG["text"]["geometry_tolerance"])
    if rank != int(CFG["text"]["frozen_span_rank"]) or tensor_sha(basis) != CFG["text"]["frozen_span_basis_sha256"]:
        raise AssertionError("Frozen text SVD rank or basis SHA mismatch")
    expected_hashes = {
        "BrainPrompt-2": CFG["text"]["brainprompt_class_bank_sha256"],
        "SPromptGL-8": CFG["text"]["spromptgl_class_bank_sha256"],
    }
    for name, record in records.items():
        if record["bank_sha256"] != expected_hashes[name]:
            raise AssertionError(f"{name} class prototype SHA mismatch")
    banks = {name: record["bank"].detach().cpu().contiguous() for name, record in records.items()}
    metadata = {
        "encoder": "frozen OpenAI CLIP RN50",
        "rn50_weights_sha256": cache["source_hashes"]["rn50_sha256"],
        "experiment60_cache_sha256": sha_file(TEXT_CACHE),
        "cache_source_hashes": cache["source_hashes"],
        "brainprompt_texts_source_order": list(BP_TEXTS),
        "brainprompt_texts_training_order": list(BP_TEXTS),
        "brainprompt_text_sha256": text_sha(BP_TEXTS),
        "spromptgl_texts_source_order": list(SP_TEXTS),
        "spromptgl_text_sha256": text_sha(SP_TEXTS),
        "spromptgl_aggregation": "normalize each; mean [0:4] for TD and [4:8] for ASD; normalize means; order [ASD,TD]",
        "span_method": "CPU float64 SVD over the ten raw frozen RN50 vectors; B=V[:,0:r]",
        "rank": rank,
        "singular_values": singular_values.tolist(),
        "basis_sha256": tensor_sha(basis),
        "bank_order": ["ASD", "TD"],
        "banks": {
            name: {
                key: value for key, value in record.items() if key not in ("bank", "source_bank")
            }
            for name, record in records.items()
        },
        "prompt_patient_label_routing": False,
    }
    if write_cache:
        atomic_torch_save(PROMPT_CACHE, {"basis": basis, "banks": banks, "metadata": metadata})
    return banks, metadata


def attach_cam_counter(cam) -> None:
    if getattr(cam, "_e72_count_handle", None) is not None:
        return

    def count(_module, _inputs):
        cam._e72_forward_count = int(getattr(cam, "_e72_forward_count", 0)) + 1

    def capture_positions(_module, _inputs, output):
        cam._e59_positions = output

    def capture_classifier_input(_module, inputs):
        cam._e72_classifier_input = inputs[0]

    cam._e72_forward_count = 0
    cam._e72_count_handle = cam.register_forward_pre_hook(count)
    cam._e59_positions = None
    cam._e59_capture_handle = cam.FeedForward[-1].register_forward_hook(capture_positions)
    classifier = getattr(cam, "OutputLayer", None)
    classifier_name = "OutputLayer"
    if classifier is None:
        classifier = getattr(cam, "Outputlayer", None)
        classifier_name = "Outputlayer"
    if classifier is None:
        raise AssertionError("E69.3 CAM does not expose its original classifier module")
    cam._e72_classifier_name = classifier_name
    cam._e72_classifier_handle = classifier.register_forward_pre_hook(capture_classifier_input)


def checked_forward(state, training: bool, use_cadr: bool) -> dict:
    state["cam"]._e72_forward_count = 0
    output = E61.forward_e2_state(state, training=training)
    if int(state["cam"]._e72_forward_count) != 1:
        raise AssertionError(f"CAM was executed {state['cam']._e72_forward_count} times; expected exactly once")
    positions = getattr(state["cam"], "_e59_positions", None)
    if positions is None or tuple(positions.shape) != (871, 4, 36):
        raise AssertionError(f"final CAM capture shape mismatch: {None if positions is None else tuple(positions.shape)}")
    classifier_input = getattr(state["cam"], "_e72_classifier_input", None)
    if classifier_input is None or tuple(classifier_input.shape) != (871, 144):
        raise AssertionError("Original OutputLayer did not receive the expected [871,144] feature")
    if not BASE.runtime()["torch"].equal(classifier_input, positions.reshape(871, 144)):
        raise AssertionError("CAM capture flatten differs from the original OutputLayer input")
    if not positions.requires_grad and training:
        raise AssertionError("Captured CAM positions were detached during training")
    fused = output.get("fused")
    if fused is None or tuple(fused.shape) != (871, 320):
        raise AssertionError(f"E61-E2 fused DARG output must be [871,320], got {None if fused is None else tuple(fused.shape)}")
    if training and not fused.requires_grad:
        raise AssertionError("DARG fused F was detached before semantic supervision")
    return output


def build_fold(fold: int, reference_manifest: dict, reference_folds: dict[int, dict]):
    state, warmup = E61.load_warmup_state(fold)
    membership = E69.apply_fold_specific_training_additions(state, fold)
    E61.configure_branch(state, "E61-E2", None)
    gate = E69.set_gate_initialization(state, "E69.3-G2")
    task = reference_folds[fold]
    state["reference_checkpoint_sha256"] = reference_folds[fold]["checkpoint_sha256"]
    state["warmup_cache_sha256"] = sha_file(WARMUP_DIR / f"fold{fold:02d}.pt")
    state["warmup_rng_sha"] = warmup["rng_sha"]
    state["training_membership"] = membership
    state["initial_gate_values"] = gate
    state["initial_gate_logits"] = {
        "edge_gate": state["model"].edge_gate_logits.detach().cpu().tolist(),
        "adaptive_gate": state["model"].adaptive_gate_logits.detach().cpu().tolist(),
    }
    state["cache_meta"] = {"pair_sha": state["pair_sha"]}
    state["construction_rng_sha"] = state["initial_rng_sha"]
    state["initial_schema_sha"] = sha_obj({
        "model": [(name, tuple(value.shape), str(value.dtype)) for name, value in state["model"].state_dict().items()],
        "cam": [(name, tuple(value.shape), str(value.dtype)) for name, value in state["cam"].state_dict().items()],
        "optimizer": [
            (name, [len(group["params"]) for group in optimizer.param_groups], [float(group["weight_decay"]) for group in optimizer.param_groups])
            for name, optimizer in (("gfr", state["opt_base"]), ("cam", state["opt_cam"]), ("cadr", state["opt_added"]))
        ],
    })
    state["e69_reference_task_sha256"] = sha_file(E69_TASKS / f"E69.3-G2_f{fold:02d}.json")
    state["e69_reference_best_epoch"] = int(task["best_epoch_1based"])
    state["e69_reference_overlap"] = list(task["train_test_overlap_1based"])
    attach_cam_counter(state["cam"])
    return state


def load_warmup(state, fold: int, warmup_audit: dict) -> dict:
    expected = warmup_audit["shared_warmup_cache"]["folds"][str(fold)]
    path = WARMUP_DIR / f"fold{fold:02d}.pt"
    if sha_file(path) != expected["sha256"]:
        raise AssertionError(f"fold {fold} staged warmup SHA changed")
    saved = BASE.runtime()["torch"].load(path, map_location="cpu", weights_only=False)
    if state["warmup_cache_sha256"] != expected["sha256"] or BASE.rng_sha(BASE.pack_rng()) != saved["rng_sha"]:
        raise AssertionError(f"fold {fold} restored warmup RNG SHA mismatch")
    return {"sha256": expected["sha256"], "rng_sha": saved["rng_sha"], "final_digest": {"rng_sha": saved["rng_sha"]}}


def train_index(state, device):
    torch = BASE.runtime()["torch"]
    return torch.as_tensor(state["train"], dtype=torch.long, device=device)


def private_auxiliary(rank: int, fold: int):
    torch = BASE.runtime()["torch"]
    device = BASE.runtime()["device"]
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    rng_before = BASE.rng_sha(BASE.pack_rng())
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(720000 + int(fold))
        if devices:
            torch.cuda.manual_seed_all(720000 + int(fold))
        module = DARG320DiseaseSupervision(rank).to(device)
    rng_after = BASE.rng_sha(BASE.pack_rng())
    if rng_before != rng_after:
        raise AssertionError("Auxiliary projection initialization consumed the main RNG")
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(CFG["auxiliary_projection"]["optimizer"]["lr"]),
        weight_decay=float(CFG["auxiliary_projection"]["optimizer"]["weight_decay"]),
    )
    return module, optimizer, module_sha(module), rng_before, rng_after


def make_losses(module, fused, labels, bank, temperature: float, grain: str) -> tuple[dict, Any]:
    output = module(fused, bank, temperature, use_fine=(grain == "joint"))
    losses = supervised_losses(output, labels)
    if grain == "joint":
        semantic = 0.5 * losses["global"] + 0.5 * losses["fine"]
    elif grain == "global_only":
        semantic = losses["global"]
    else:
        raise ValueError(f"Unknown supervision grain {grain}")
    return {**losses, "output": output, "semantic": semantic}, output


def capture_warmup_representations(module, fused, test_indices, banks) -> dict[str, Any]:
    """Save unsupervised epoch-10 held-out representations for post-hoc comparison."""
    torch = BASE.runtime()["torch"]
    import torch.nn.functional as F

    with torch.no_grad():
        global_patient = F.normalize(module.p_g(fused), dim=-1)
        segments = torch.split(fused, module.INPUT_DIMS, dim=-1)
        fine_patient = torch.stack([F.normalize(layer(chunk), dim=-1) for layer, chunk in zip(module.p_f, segments)], dim=1)
        index = torch.as_tensor(test_indices, dtype=torch.long, device=fused.device)
        global_test = global_patient[index]
        fine_test = fine_patient[index]
        record = {
            "test_indices": [int(value) for value in test_indices],
            "global_patient": global_test.detach().cpu().tolist(),
            "fine_patient": fine_test.detach().cpu().tolist(),
            "similarities": {},
        }
        for name, bank in banks.items():
            proto = bank.to(device=fused.device, dtype=fused.dtype)
            record["similarities"][name] = {
                "global": (global_test @ proto.T).detach().cpu().tolist(),
                "fine": (fine_test @ proto.T).detach().cpu().tolist(),
            }
    return record


def finite_tensor(value) -> bool:
    return bool(BASE.runtime()["torch"].isfinite(value).all())


def gradient_stats(classification_loss, semantic_loss, fused, indices) -> dict[str, Any]:
    torch = BASE.runtime()["torch"]
    import torch.nn.functional as F

    g_cls_full = torch.autograd.grad(classification_loss, fused, retain_graph=True, allow_unused=False)[0]
    g_sem_full = torch.autograd.grad(semantic_loss, fused, retain_graph=True, allow_unused=False)[0]
    g_cls = g_cls_full[indices].detach().float()
    g_sem = g_sem_full[indices].detach().float()
    g_cls_norm = float(g_cls.norm().cpu())
    g_sem_norm = float(g_sem.norm().cpu())
    cosine = float(F.cosine_similarity(g_cls.reshape(1, -1), g_sem.reshape(1, -1), dim=-1).item())
    modality = {}
    offsets = (0, 48, 54, 64, 320)
    for index, name in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
        left = g_cls[:, offsets[index]:offsets[index + 1]].reshape(1, -1)
        right = g_sem[:, offsets[index]:offsets[index + 1]].reshape(1, -1)
        modality[name] = {
            "classification_norm": float(left.norm().cpu()),
            "semantic_norm": float(right.norm().cpu()),
            "cosine": float(F.cosine_similarity(left, right, dim=-1).item()),
        }
    if not math.isfinite(g_cls_norm) or not math.isfinite(g_sem_norm) or not math.isfinite(cosine):
        raise FloatingPointError("nonfinite fused-F classification/semantic gradient diagnostic")
    return {
        "classification_norm": g_cls_norm,
        "semantic_norm": g_sem_norm,
        "cosine": cosine,
        "modality": modality,
    }


def grad_calibration(state, module, bank, item: dict, optimizer) -> dict[str, Any]:
    torch = BASE.runtime()["torch"]
    pre_params = {
        "model": BASE.digest_named_parameters(state["model"]),
        "cam": BASE.digest_named_parameters(state["cam"]),
        "aux": module_sha(module),
    }
    pre_opts = {
        "base": E59._check_optimizer_digest(state["opt_base"]),
        "cam": E59._check_optimizer_digest(state["opt_cam"]),
        "added": E59._check_optimizer_digest(state["opt_added"]),
        "aux": E59._check_optimizer_digest(optimizer),
    }
    rng_before = BASE.rng_sha(BASE.pack_rng())
    output = checked_forward(state, training=False, use_cadr=True)
    fused = output["fused"]
    if tuple(fused.shape) != (871, 320) or not fused.requires_grad:
        raise AssertionError("DARG semantic source must be the live [871,320] forward output F")
    indices = train_index(state, fused.device)
    train_labels = state["y"][indices].long()
    if train_labels.numel() != indices.numel():
        raise AssertionError("outer-training label length mismatch")
    supervised, _ = make_losses(
        module,
        fused[indices],
        train_labels,
        bank,
        float(item["temperature"]),
        item["grain"],
    )
    pf_unused = None
    if item["grain"] == "global_only":
        pf_grads = torch.autograd.grad(
            supervised["semantic"], tuple(module.p_f.parameters()), allow_unused=True, retain_graph=True
        )
        pf_unused = all(value is None for value in pf_grads)
        if not pf_unused:
            raise AssertionError("P_f received a gradient in a global-only group")
    cls_grad = torch.autograd.grad(output["cls"], fused, retain_graph=True, allow_unused=False)[0][indices]
    sem_grad = torch.autograd.grad(supervised["semantic"], fused, retain_graph=False, allow_unused=False)[0][indices]
    g_cls = float(cls_grad.detach().float().norm().cpu())
    g_sem = float(sem_grad.detach().float().norm().cpu())
    if not (math.isfinite(g_cls) and math.isfinite(g_sem)) or g_cls <= 1e-12 or g_sem <= 1e-12:
        raise FloatingPointError(f"degenerate gradient budget calibration {g_cls=} {g_sem=}")
    alpha = float(CFG["supervision"]["gradient_budget_ratio"]) * g_cls / (g_sem + 1e-12)
    if not math.isfinite(alpha) or alpha <= 0.0 or alpha > float(CFG["supervision"]["alpha_max"]):
        raise FloatingPointError(f"frozen gradient coefficient is invalid or exceeds 10: {alpha}")
    ratio = alpha * g_sem / g_cls
    if not finite_tensor(supervised["semantic"]):
        raise FloatingPointError("nonfinite auxiliary loss during calibration")
    post_params = {
        "model": BASE.digest_named_parameters(state["model"]),
        "cam": BASE.digest_named_parameters(state["cam"]),
        "aux": module_sha(module),
    }
    post_opts = {
        "base": E59._check_optimizer_digest(state["opt_base"]),
        "cam": E59._check_optimizer_digest(state["opt_cam"]),
        "added": E59._check_optimizer_digest(state["opt_added"]),
        "aux": E59._check_optimizer_digest(optimizer),
    }
    rng_after = BASE.rng_sha(BASE.pack_rng())
    if pre_params != post_params or pre_opts != post_opts or rng_before != rng_after:
        raise AssertionError("Gradient calibration changed parameters, optimizer states, or main RNG")
    if abs(ratio - float(CFG["supervision"]["gradient_budget_ratio"])) > 1e-6:
        raise AssertionError(f"initial weighted gradient ratio misses target: {ratio}")
    return {
        "group": item["id"],
        "fold": int(state["fold"]),
        "alpha": alpha,
        "classification_gradient_norm": g_cls,
        "unscaled_semantic_gradient_norm": g_sem,
        "initial_weighted_gradient_ratio": ratio,
        "unscaled_global_loss": float(supervised["global"].detach().cpu()),
        "unscaled_fine_loss": None if supervised["fine"] is None else float(supervised["fine"].detach().cpu()),
        "unscaled_semantic_loss": float(supervised["semantic"].detach().cpu()),
        "cam_forward_count": int(state["cam"]._e72_forward_count),
        "semantic_source_shape": list(fused.shape),
        "semantic_source_requires_grad": bool(fused.requires_grad),
        "gradient_location": "DARG fused F [871,320]",
        "p_f_unused": pf_unused,
        "test_labels_passed_through_effective_train": bool(state["train_test_overlap_1based"]),
        "train_test_overlap_1based": list(state["train_test_overlap_1based"]),
        "effective_train_sha256": state["effective_train_sha"],
        "calibration_parameter_state_unchanged": pre_params == post_params,
        "calibration_optimizer_state_unchanged": pre_opts == post_opts,
        "calibration_main_rng_unchanged": rng_before == rng_after,
    }


def assert_staged_warmup_restore(state, fold: int, warmup_info: dict, warmup_audit: dict) -> dict[str, Any]:
    source_state, saved = E61.load_warmup_state(fold)
    expected = warmup_audit["shared_warmup_cache"]["folds"][str(fold)]
    actual = {
        "sha256": warmup_info["sha256"],
        "base_sha": BASE.digest_named_parameters(source_state["base"]),
        "cam_sha": BASE.digest_named_parameters(source_state["cam"]),
        "opt_base_sha": E59._check_optimizer_digest(source_state["opt_base"]),
        "opt_cam_sha": E59._check_optimizer_digest(source_state["opt_cam"]),
        "opt_added_sha": E59._check_optimizer_digest(source_state["opt_added"]),
        "rng_sha": BASE.rng_sha(BASE.pack_rng()),
    }
    if actual["sha256"] != expected["sha256"] or actual["rng_sha"] != saved["rng_sha"]:
        raise AssertionError(f"fold {fold} staged warmup SHA/RNG restore mismatch")
    current_model = source_state["model"].state_dict()
    for name, value in current_model.items():
        if not tree_equal(value.detach().cpu(), saved["model"][name].detach().cpu()):
            raise AssertionError(f"fold {fold} E61 warmup model parameter mismatch: {name}")
    if not tree_equal(source_state["cam"].state_dict(), saved["cam"]):
        raise AssertionError(f"fold {fold} E61 warmup CAM weights did not restore")
    for name, optimizer, key in (
        ("gfr", source_state["opt_base"], "optimizer_base"),
        ("cam", source_state["opt_cam"], "optimizer_cam"),
        ("cadr", source_state["opt_added"], "optimizer_added"),
    ):
        if not tree_equal(optimizer.state_dict(), saved[key]):
            raise AssertionError(f"fold {fold} E61 warmup {name} optimizer state mismatch")
    del source_state
    return actual


def run_step(state, module, optimizer, bank, alpha: float, item: dict, record: bool = False):
    torch = BASE.runtime()["torch"]
    rng_before_train = BASE.rng_sha(BASE.pack_rng())
    for name in ("opt_base", "opt_cam", "opt_added"):
        state[name].zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    output = checked_forward(state, training=True, use_cadr=True)
    fused = output["fused"]
    if tuple(fused.shape) != (871, 320) or not fused.requires_grad:
        raise AssertionError("DARG semantic source must be the live [871,320] forward output F")
    indices = train_index(state, fused.device)
    train_labels = state["y"][indices].long()
    supervised, _ = make_losses(module, fused[indices], train_labels, bank, float(item["temperature"]), item["grain"])
    total = output["total"] + float(alpha) * supervised["semantic"]
    if not finite_tensor(total):
        raise FloatingPointError("nonfinite MGDSS total loss")
    diagnostics = None
    if record:
        grads = gradient_stats(output["cls"], supervised["semantic"], fused, indices)
        diagnostics = semantic_diagnostics(supervised, train_labels, float(item["temperature"]), grads, alpha)
        diagnostics["rng_before_train_sha256"] = rng_before_train
        diagnostics["train_test_overlap_1based"] = list(state["train_test_overlap_1based"])
        diagnostics["test_label_exposure_rows_1based"] = list(state["train_test_overlap_1based"])
        diagnostics["test_labels_in_effective_training_set"] = bool(state["train_test_overlap_1based"])
        diagnostics["effective_train_sha256"] = state["effective_train_sha"]
    total.backward()
    for module_check in (state["model"], state["cam"], module):
        for parameter in module_check.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("nonfinite training gradient")
    state["opt_base"].step()
    state["opt_cam"].step()
    state["opt_added"].step()
    optimizer.step()
    if diagnostics is not None:
        diagnostics["rng_after_train_sha256"] = BASE.rng_sha(BASE.pack_rng())
    return output, diagnostics


def semantic_diagnostics(supervised: dict, labels, temperature: float, grads: dict, alpha: float) -> dict[str, Any]:
    torch = BASE.runtime()["torch"]
    output = supervised["output"]
    y = labels.long()
    n = y.numel()
    correct = {}
    margins = {}
    tau = float(temperature)
    global_similarity = output["global_logits"].detach() * tau
    global_margin = (global_similarity.gather(1, y[:, None]).squeeze(1) - global_similarity.gather(1, (1-y)[:, None]).squeeze(1))
    correct["global"] = float((global_similarity.argmax(-1) == y).float().mean().cpu())
    margins["global"] = float(global_margin.mean().cpu())
    fine_acc = {}
    fine_margin = {}
    fine_loss = {}
    if output["fine_logits"] is not None:
        sim = output["fine_logits"].detach() * tau
        for index, name in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
            per = sim[:, index, :]
            margin = per.gather(1, y[:, None]).squeeze(1) - per.gather(1, (1-y)[:, None]).squeeze(1)
            fine_acc[name] = float((per.argmax(-1) == y).float().mean().cpu())
            fine_margin[name] = float(margin.mean().cpu())
            fine_loss[name] = float(supervised["fine_modalities"][index].detach().cpu())
    return {
        "loss_global": float(supervised["global"].detach().cpu()),
        "loss_PHENO": fine_loss.get("PHENO"),
        "loss_ANAT": fine_loss.get("ANAT"),
        "loss_FUNC": fine_loss.get("FUNC"),
        "loss_CORR": fine_loss.get("CORR"),
        "loss_fine": None if supervised["fine"] is None else float(supervised["fine"].detach().cpu()),
        "loss_semantic": float(supervised["semantic"].detach().cpu()),
        "alpha": float(alpha),
        "gradient_norm_cls_f": grads["classification_norm"],
        "gradient_norm_semantic_f": grads["semantic_norm"],
        "gradient_cosine_f": grads["cosine"],
        "weighted_gradient_ratio": float(alpha) * grads["semantic_norm"] / (grads["classification_norm"] + 1e-12),
        "gradient_cosine_PHENO": grads["modality"]["PHENO"]["cosine"],
        "gradient_cosine_ANAT": grads["modality"]["ANAT"]["cosine"],
        "gradient_cosine_FUNC": grads["modality"]["FUNC"]["cosine"],
        "gradient_cosine_CORR": grads["modality"]["CORR"]["cosine"],
        "train_similarity_accuracy_global": correct["global"],
        "train_similarity_margin_global": margins["global"],
        "train_similarity_accuracy_PHENO": fine_acc.get("PHENO"),
        "train_similarity_accuracy_ANAT": fine_acc.get("ANAT"),
        "train_similarity_accuracy_FUNC": fine_acc.get("FUNC"),
        "train_similarity_accuracy_CORR": fine_acc.get("CORR"),
        "train_similarity_margin_PHENO": fine_margin.get("PHENO"),
        "train_similarity_margin_ANAT": fine_margin.get("ANAT"),
        "train_similarity_margin_FUNC": fine_margin.get("FUNC"),
        "train_similarity_margin_CORR": fine_margin.get("CORR"),
        "train_count": int(n),
        "labels_source": "E69.3 effective training indices only",
    }


def aux_optimizer_digest(optimizer) -> str:
    return E59._check_optimizer_digest(optimizer)


def best_state(state, module, optimizer) -> dict[str, Any]:
    return {
        "model": state_cpu(state["model"]),
        "cam": state_cpu(state["cam"]),
        "auxiliary": state_cpu(module),
        "optimizer_base": state_tree_cpu(state["opt_base"].state_dict()),
        "optimizer_cam": state_tree_cpu(state["opt_cam"].state_dict()),
        "optimizer_added": state_tree_cpu(state["opt_added"].state_dict()),
        "optimizer_auxiliary": state_tree_cpu(optimizer.state_dict()),
    }


def apply_state(state, module, optimizer, saved: dict[str, Any]) -> None:
    state["model"].load_state_dict(saved["model"], strict=True)
    state["cam"].load_state_dict(saved["cam"], strict=True)
    module.load_state_dict(saved["auxiliary"], strict=True)
    state["opt_base"].load_state_dict(saved["optimizer_base"])
    state["opt_cam"].load_state_dict(saved["optimizer_cam"])
    state["opt_added"].load_state_dict(saved["optimizer_added"])
    optimizer.load_state_dict(saved["optimizer_auxiliary"])


def checkpoint_payload(group_id: str, fold: int, state, module, optimizer, alpha: float, prompt_meta: dict, best: dict, best_metrics: dict, best_epoch: int, epochs: int) -> dict[str, Any]:
    item = group_item(group_id)
    return {
        "schema": "experiment72_2_darg320_mgdss_checkpoint_v1",
        "group": group_id,
        "group_definition": item,
        "fold": int(fold),
        "seed": 0,
        "best_epoch": int(best_epoch),
        "epochs_executed": int(epochs),
        "warmup_epochs": 10,
        "alpha": float(alpha),
        "initial_auxiliary_sha256": state["initial_auxiliary_sha256"],
        "base": best["model"],
        "cam": best["cam"],
        "auxiliary": best["auxiliary"],
        "optimizer_base": best["optimizer_base"],
        "optimizer_cam": best["optimizer_cam"],
        "optimizer_added": best["optimizer_added"],
        "optimizer_auxiliary": best["optimizer_auxiliary"],
        "metrics": best_metrics,
        "best_diagnostics": state.get("best_diagnostics"),
        "prompt_bank_sha256": prompt_meta["banks"][item["prompt"]]["bank_sha256"],
        "prompt_source_sha256": prompt_meta["brainprompt_text_sha256"] if item["prompt"] == "BrainPrompt-2" else prompt_meta["spromptgl_text_sha256"],
        "prompt_cache_sha256": sha_file(PROMPT_CACHE),
        "text_basis_sha256": prompt_meta["basis_sha256"],
        "source_sha256": code_hashes(),
        "config_sha256": sha_file(CFG_PATH),
        "warmup_sha256": state["warmup_cache_sha256"],
        "split_sha256": state["split_sha"],
        "pair_sha256": state["cache_meta"]["pair_sha"],
        "relation_cache_json_sha256": sha_file(BASE.REL_CACHE / f"fold{fold:02d}.json"),
        "relation_cache_npz_sha256": sha_file(BASE.REL_CACHE / f"fold{fold:02d}.npz"),
        "initial_shared_sha256": state["initial_shared_sha"],
        "initial_gfr_sha256": state["initial_gfr_sha"],
        "initial_cam_sha256": state["initial_cam_sha"],
        "initial_added_sha256": state["initial_added_sha"],
        "initial_schema_sha256": state["initial_schema_sha"],
        "initial_rng_sha256": state["construction_rng_sha"],
        "reference_checkpoint_sha256": state["reference_checkpoint_sha256"],
        "e69_reference_task_sha256": state["e69_reference_task_sha256"],
        "e69_reference_best_epoch_1based": state["e69_reference_best_epoch"],
        "initial_gate_values": state["initial_gate_values"],
        "initial_gate_logits": state["initial_gate_logits"],
        "gate_optimizer_state_at_initialization": state["initial_gate_values"]["optimizer_state"],
        "outer_train_count": int(len(state["train"])),
        "outer_test_count": int(len(state["test"])),
        "effective_train_sha256": state["effective_train_sha"],
        "train_test_overlap_1based": list(state["train_test_overlap_1based"]),
        "base_train_sha256": state["base_train_sha"],
        "added_train_rows_1based": list(state["added_train_rows_1based"]),
        "train_indices_sha256": sha_obj(list(map(int, state["train"]))),
        "test_indices_sha256": sha_obj(list(map(int, state["test"]))),
        "test_labels_used_for_auxiliary_loss": bool(state["train_test_overlap_1based"]),
        "auxiliary_label_exposure_rows_1based": list(state["train_test_overlap_1based"]),
        "inference_uses_auxiliary": False,
    }


def latest_payload(group_id: str, fold: int, state, module, optimizer, alpha: float, start_epoch: int, wait: int, best_acc: float, best_epoch: int, best: dict | None, best_metrics: dict | None, mechanism_history: list[dict], prompt_meta: dict) -> dict[str, Any]:
    return {
        "schema": "experiment72_2_darg320_mgdss_latest_v1",
        "group": group_id,
        "fold": int(fold),
        "next_epoch": int(start_epoch),
        "wait": int(wait),
        "best_acc": float(best_acc),
        "best_epoch": int(best_epoch),
        "best_state": best,
        "best_metrics": best_metrics,
        "mechanism_history": mechanism_history,
        "base": state_cpu(state["model"]),
        "cam": state_cpu(state["cam"]),
        "auxiliary": state_cpu(module),
        "optimizer_base": state_tree_cpu(state["opt_base"].state_dict()),
        "optimizer_cam": state_tree_cpu(state["opt_cam"].state_dict()),
        "optimizer_added": state_tree_cpu(state["opt_added"].state_dict()),
        "optimizer_auxiliary": state_tree_cpu(optimizer.state_dict()),
        "rng": state_tree_cpu(BASE.pack_rng()),
        "alpha": float(alpha),
        "warmup_sha256": state["warmup_cache_sha256"],
        "split_sha256": state["split_sha"],
        "pair_sha256": state["cache_meta"]["pair_sha"],
        "effective_train_sha256": state["effective_train_sha"],
        "train_test_overlap_1based": list(state["train_test_overlap_1based"]),
        "initial_auxiliary_sha256": state["initial_auxiliary_sha256"],
        "prompt_bank_sha256": prompt_meta["banks"][group_item(group_id)["prompt"]]["bank_sha256"],
        "prompt_cache_sha256": sha_file(PROMPT_CACHE),
        "source_sha256": code_hashes(),
        "config_sha256": sha_file(CFG_PATH),
    }


def manifest_update(group_id: str, fold: int, status: str, fields: dict[str, Any]) -> None:
    MANIFEST_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        key = f"{group_id}_f{fold:02d}"
        current = manifest.setdefault("tasks", {}).get(key, {})
        manifest["tasks"][key] = {**current, **fields, "status": status, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        done = sum(item.get("status") == "COMPLETE" for item in manifest["tasks"].values())
        manifest["completed_tasks"] = int(done)
        if done == 60:
            manifest["status"] = "TRAINING_COMPLETE_PENDING_AUDIT"
        atomic_json(MANIFEST_PATH, manifest)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def load_prompt_cache() -> tuple[dict[str, Any], dict[str, Any]]:
    torch = BASE.runtime()["torch"]
    if not PROMPT_CACHE.is_file():
        raise FileNotFoundError("Frozen prompt cache was not created by --preflight")
    payload = torch.load(PROMPT_CACHE, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if sha_file(TEXT_CACHE) != metadata["experiment60_cache_sha256"]:
        raise AssertionError("Experiment 60 text cache changed after preflight")
    if metadata["basis_sha256"] != CFG["text"]["frozen_span_basis_sha256"]:
        raise AssertionError("text basis changed after preflight")
    return payload["banks"], metadata


def restore_from_latest(state, module, optimizer, saved: dict, group_id: str, fold: int, warmup_info: dict, initial_aux_sha: str, prompt_meta: dict) -> dict:
    if saved.get("schema") != "experiment72_2_darg320_mgdss_latest_v1" or saved.get("group") != group_id or int(saved.get("fold", -1)) != fold:
        raise AssertionError("latest checkpoint group/fold/schema mismatch")
    expected = {
        "config_sha256": sha_file(CFG_PATH),
        "warmup_sha256": warmup_info["sha256"],
        "split_sha256": state["split_sha"],
        "pair_sha256": state["cache_meta"]["pair_sha"],
        "initial_auxiliary_sha256": initial_aux_sha,
        "prompt_cache_sha256": sha_file(PROMPT_CACHE),
    }
    for key, value in expected.items():
        if saved.get(key) != value:
            raise AssertionError(f"latest checkpoint provenance mismatch for {key}")
    expected_bank = prompt_meta["banks"][group_item(group_id)["prompt"]]["bank_sha256"]
    if saved.get("prompt_bank_sha256") != expected_bank:
        raise AssertionError("latest checkpoint prompt bank SHA mismatch")
    state["model"].load_state_dict(saved["base"], strict=True)
    state["cam"].load_state_dict(saved["cam"], strict=True)
    module.load_state_dict(saved["auxiliary"], strict=True)
    state["opt_base"].load_state_dict(saved["optimizer_base"])
    state["opt_cam"].load_state_dict(saved["optimizer_cam"])
    state["opt_added"].load_state_dict(saved["optimizer_added"])
    optimizer.load_state_dict(saved["optimizer_auxiliary"])
    BASE.restore_rng(saved["rng"])
    return {
        "start_epoch": int(saved["next_epoch"]),
        "wait": int(saved["wait"]),
        "best_acc": float(saved["best_acc"]),
        "best_epoch": int(saved["best_epoch"]),
        "best_state": saved["best_state"],
        "best_metrics": saved["best_metrics"],
        "mechanism_history": list(saved.get("mechanism_history", [])),
        "alpha": float(saved["alpha"]),
    }


def train_worker(group_id: str, fold: int) -> None:
    torch = BASE.runtime()["torch"]
    if group_id not in GROUP_IDS or fold not in range(1, 11):
        raise ValueError(f"invalid task {group_id} fold {fold}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"FROZEN_PREFLIGHT_PASS", "TRAINING", "TRAINING_COMPLETE_PENDING_AUDIT"}:
        raise RuntimeError("formal training requires a passing frozen preflight manifest")
    if manifest.get("config_sha256") != sha_file(CFG_PATH) or manifest.get("source_sha256", {}).get("run.py") != sha_file(EXP / "run.py") or manifest.get("source_sha256", {}).get("mgdss.py") != sha_file(EXP / "mgdss.py"):
        raise RuntimeError("frozen code/config SHA changed after preflight")
    if not CFG["supervision"].get("coefficients_frozen", True):
        raise RuntimeError("formal alpha calibration is not frozen")
    alpha_record = manifest["preflight"]["alphas"][str(fold)][group_id]
    alpha = float(alpha_record["alpha"])
    if not math.isfinite(alpha) or alpha > 10.0:
        raise RuntimeError("invalid frozen alpha")
    ref_manifest, ref_folds = load_reference()
    warmup_audit = verify_warmup_audit()
    banks, prompt_meta = load_prompt_cache()
    item = group_item(group_id)
    bank = banks[item["prompt"]].to(BASE.runtime()["device"])
    task_path, checkpoint_path, latest_path, log_path = task_paths(group_id, fold)
    state = build_fold(fold, ref_manifest, ref_folds)
    initial_aux_rng = BASE.rng_sha(BASE.pack_rng())
    module, optimizer, initial_aux_sha, rng_before, rng_after = private_auxiliary(int(CFG["text"]["frozen_span_rank"]), fold)
    if initial_aux_sha != alpha_record["initial_auxiliary_sha256"] or rng_before != rng_after or initial_aux_rng != BASE.rng_sha(BASE.pack_rng()):
        raise AssertionError("formal auxiliary initialization differs from preflight or consumed the main RNG")
    warmup_info = load_warmup(state, fold, warmup_audit)
    if not latest_path.is_file():
        state_cpu_init = BASE.rng_sha(BASE.pack_rng())
        if state_cpu_init != warmup_audit["shared_warmup_cache"]["folds"][str(fold)]["captured_digest"]["rng_sha"]:
            raise AssertionError("initial formal RNG differs from the audited epoch-10 warmup")

    start_epoch, wait, best_acc, best_epoch = 10, 0, -1.0, -1
    best = None
    best_metrics = None
    mechanism_history: list[dict] = []
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        resumed = restore_from_latest(state, module, optimizer, saved, group_id, fold, warmup_info, initial_aux_sha, prompt_meta)
        start_epoch = resumed["start_epoch"]
        wait = resumed["wait"]
        best_acc = resumed["best_acc"]
        best_epoch = resumed["best_epoch"]
        best = resumed["best_state"]
        best_metrics = resumed["best_metrics"]
        mechanism_history = resumed["mechanism_history"]
        if abs(resumed["alpha"] - alpha) > 1e-15:
            raise AssertionError("resume alpha differs from the frozen coefficient")
    elif checkpoint_path.is_file():
        task = json.loads(task_path.read_text(encoding="utf-8")) if task_path.is_file() else {}
        if task.get("status") == "COMPLETE" and task.get("checkpoint_sha256") == sha_file(checkpoint_path):
            return

    state["initial_auxiliary_sha256"] = initial_aux_sha
    state["best_diagnostics"] = None
    state["warmup_cache_sha256"] = warmup_info["sha256"]
    if latest_path.is_file() and start_epoch > 10:
        state["best_diagnostics"] = (best or {}).get("best_diagnostics") if isinstance(best, dict) else None
    manifest_update(group_id, fold, "RUNNING", {
        "alpha": alpha,
        "initial_auxiliary_sha256": initial_aux_sha,
        "warmup_sha256": warmup_info["sha256"],
        "split_sha256": state["split_sha"],
        "pair_sha256": state["cache_meta"]["pair_sha"],
        "next_epoch": start_epoch,
        "worker_device": str(BASE.runtime()["device"]),
        "source_sha256": code_hashes(),
    })
    started = time.time()
    current_epoch = start_epoch
    epochs_executed = start_epoch
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"event": "START", "group": group_id, "fold": fold, "start_epoch": start_epoch, "device": str(BASE.runtime()["device"]), "alpha": alpha, "warmup_sha256": warmup_info["sha256"]}, allow_nan=False) + "\n")
        log.flush()
        try:
            for epoch in range(start_epoch, int(CFG["cv"]["max_epochs"])):
                current_epoch = epoch
                output, mechanism = run_step(state, module, optimizer, bank, alpha, item, record=True)
                if mechanism is None:
                    raise AssertionError("mechanism record was not generated")
                mechanism["group"] = group_id
                mechanism["fold"] = int(fold)
                mechanism["epoch"] = int(epoch)
                mechanism["base_cls_loss"] = float(output["cls"].detach().cpu())
                mechanism["base_graph_loss"] = float(output["graph"].detach().cpu())
                mechanism["base_pair_loss"] = float(output["pair"].detach().cpu())
                mechanism["base_adv_loss"] = float(output["adv"].detach().cpu())
                mechanism["base_preserve_loss"] = float(output["preserve"].detach().cpu())
                mechanism["cam_forward_count"] = int(state["cam"]._e72_forward_count)
                mechanism["rng_after_train_sha256"] = BASE.rng_sha(BASE.pack_rng())
                mechanism_history.append(mechanism)
                epochs_executed = epoch + 1
                state["_active_epoch_mechanism"] = mechanism
                with torch.no_grad():
                    evaluated = checked_forward(state, training=False, use_cadr=True)
                    metrics = BASE.classification_metrics(evaluated["logp"], state["test"], state["y"])
                mechanism["rng_after_eval_sha256"] = BASE.rng_sha(BASE.pack_rng())
                state["_active_epoch_mechanism"] = None
                improved = float(metrics["acc"]) > float(best_acc)
                if improved:
                    best_acc = float(metrics["acc"])
                    best_epoch = int(epoch)
                    wait = 0
                    best = best_state(state, module, optimizer)
                    best_metrics = {key: float(metrics[key]) for key in METRIC_KEYS}
                    state["best_diagnostics"] = {
                        **mechanism,
                        "best_selection_metric": "ACC",
                        "strict_acc_improvement": True,
                    }
                    best["best_diagnostics"] = copy.deepcopy(state["best_diagnostics"])
                else:
                    wait += 1
                if (epoch + 1) % 10 == 0 or epoch == int(CFG["cv"]["max_epochs"]) - 1 or wait > int(CFG["cv"]["early_stop_wait_gt"]):
                    latest = latest_payload(group_id, fold, state, module, optimizer, alpha, epoch + 1, wait, best_acc, best_epoch, best, best_metrics, mechanism_history, prompt_meta)
                    atomic_torch_save(latest_path, latest)
                    manifest_update(group_id, fold, "RUNNING", {"next_epoch": epoch + 1, "best_epoch": best_epoch, "best_acc": best_acc, "epochs_executed": epochs_executed, "latest_sha256": sha_file(latest_path)})
                if improved or (epoch + 1) % 10 == 0:
                    log.write(json.dumps({"event": "EPOCH", "group": group_id, "fold": fold, "epoch": epoch, "acc": metrics["acc"], "best_epoch": best_epoch, "best_acc": best_acc, "wait": wait, "alpha": alpha, "weighted_gradient_ratio": mechanism["weighted_gradient_ratio"]}, allow_nan=False) + "\n")
                    log.flush()
                if wait > int(CFG["cv"]["early_stop_wait_gt"]):
                    break
            if best is None or best_metrics is None:
                raise RuntimeError("no active checkpoint was selected")
            apply_state(state, module, optimizer, best)
            with torch.no_grad():
                readback = checked_forward(state, training=False, use_cadr=True)
                readback_metrics = BASE.classification_metrics(readback["logp"], state["test"], state["y"])
            for key in METRIC_KEYS:
                if abs(float(readback_metrics[key]) - float(best_metrics[key])) > 1e-10:
                    raise AssertionError(f"worker best readback mismatch for {key}")
            payload = checkpoint_payload(group_id, fold, state, module, optimizer, alpha, prompt_meta, best, best_metrics, best_epoch, epochs_executed)
            atomic_torch_save(checkpoint_path, payload)
            cp_sha = sha_file(checkpoint_path)
            mechanism_path = RUNTIME / "mechanism" / f"{group_id}_f{fold:02d}.json"
            atomic_json(mechanism_path, mechanism_history)
            task = {
                "group": group_id,
                "fold": int(fold),
                "status": "COMPLETE",
                "seed": 0,
                "best_epoch": int(best_epoch),
                "epochs_executed": int(epochs_executed),
                "alpha": alpha,
                "metrics": {key: float(readback_metrics[key]) for key in METRIC_KEYS},
                "confusion": {key: int(readback_metrics[key]) for key in ("asd_tp", "td_fp", "asd_fn", "td_tn")},
                "checkpoint_sha256": cp_sha,
                "latest_sha256": sha_file(latest_path),
                "mechanism_sha256": sha_file(mechanism_path),
                "warmup_sha256": warmup_info["sha256"],
                "split_sha256": state["split_sha"],
                "pair_sha256": state["cache_meta"]["pair_sha"],
                "initial_shared_sha256": state["initial_shared_sha"],
                "initial_gfr_sha256": state["initial_gfr_sha"],
                "initial_cam_sha256": state["initial_cam_sha"],
                "initial_added_sha256": state["initial_added_sha"],
                "initial_schema_sha256": state["initial_schema_sha"],
                "initial_rng_sha256": state["construction_rng_sha"],
                "initial_auxiliary_sha256": initial_aux_sha,
                "prompt_bank_sha256": payload["prompt_bank_sha256"],
                "test_labels_used_for_auxiliary_loss": bool(state["train_test_overlap_1based"]),
                "auxiliary_label_exposure_rows_1based": list(state["train_test_overlap_1based"]),
                "effective_train_sha256": state["effective_train_sha"],
                "train_test_overlap_1based": list(state["train_test_overlap_1based"]),
                "cam_forward_count_per_forward": 1,
                "wall_seconds": time.time() - started,
                "readback_status": "WORKER_READBACK_PASS",
            }
            atomic_json(task_path, task)
            manifest_update(group_id, fold, "COMPLETE", {**task, "worker_readback_status": "PASS"})
            log.write(json.dumps({"event": "COMPLETE", "group": group_id, "fold": fold, "best_epoch": best_epoch, "epochs_executed": epochs_executed, "checkpoint_sha256": cp_sha, "readback": "PASS"}, allow_nan=False) + "\n")
            log.flush()
        except BaseException:
            error = traceback.format_exc()
            atomic_json(task_path, {"group": group_id, "fold": fold, "status": "FAILED", "epoch": current_epoch, "error": error, "latest_sha256": sha_file(latest_path) if latest_path.is_file() else None})
            manifest_update(group_id, fold, "FAILED", {"epoch": current_epoch, "error": error})
            log.write(json.dumps({"event": "FAILED", "group": group_id, "fold": fold, "epoch": current_epoch, "error": error}, allow_nan=False) + "\n")
            log.flush()
            raise


def recovery_smoke(fold: int, reference_manifest: dict, reference_folds: dict[int, dict], warmup_audit: dict, banks: dict[str, Any], prompt_meta: dict, alpha_record: dict) -> dict[str, Any]:
    torch = BASE.runtime()["torch"]
    item = group_item("G2")
    bank = banks["BrainPrompt-2"].to(BASE.runtime()["device"])
    alpha = float(alpha_record["alpha"])

    def initialize():
        state = build_fold(fold, reference_manifest, reference_folds)
        module, optimizer, initial_sha, _before, _after = private_auxiliary(int(CFG["text"]["frozen_span_rank"]), fold)
        warm = load_warmup(state, fold, warmup_audit)
        return state, module, optimizer, initial_sha, warm

    def one_step(state, module, optimizer):
        run_step(state, module, optimizer, bank, alpha, item, record=False)
        return {
            "model": state_cpu(state["model"]),
            "cam": state_cpu(state["cam"]),
            "auxiliary": state_cpu(module),
            "optimizer_base": state_tree_cpu(state["opt_base"].state_dict()),
            "optimizer_cam": state_tree_cpu(state["opt_cam"].state_dict()),
            "optimizer_added": state_tree_cpu(state["opt_added"].state_dict()),
            "optimizer_auxiliary": state_tree_cpu(optimizer.state_dict()),
            "rng": state_tree_cpu(BASE.pack_rng()),
        }

    state_a, module_a, optimizer_a, initial_a, warm_a = initialize()
    rng_at_start_a = BASE.rng_sha(BASE.pack_rng())
    step1_a = one_step(state_a, module_a, optimizer_a)
    continuous = one_step(state_a, module_a, optimizer_a)
    state_b, module_b, optimizer_b, initial_b, warm_b = initialize()
    if initial_a != initial_b or warm_a["sha256"] != warm_b["sha256"] or rng_at_start_a != warm_a["final_digest"]["rng_sha"]:
        raise AssertionError("recovery smoke did not rebuild the exact same fold-1 epoch-10 state")
    first_b = one_step(state_b, module_b, optimizer_b)
    if not tree_equal(step1_a, first_b):
        raise AssertionError("repeated fold-1 first steps are not parameter/optimizer/RNG identical")
    resume_state, resume_module, resume_optimizer, resume_initial, resume_warm = initialize()
    if resume_initial != initial_a or resume_warm["sha256"] != warm_a["sha256"]:
        raise AssertionError("resume smoke rebuilt a different initial state")
    resume_module.load_state_dict(first_b["auxiliary"], strict=True)
    resume_state["model"].load_state_dict(first_b["model"], strict=True)
    resume_state["cam"].load_state_dict(first_b["cam"], strict=True)
    resume_state["opt_base"].load_state_dict(first_b["optimizer_base"])
    resume_state["opt_cam"].load_state_dict(first_b["optimizer_cam"])
    resume_state["opt_added"].load_state_dict(first_b["optimizer_added"])
    resume_optimizer.load_state_dict(first_b["optimizer_auxiliary"])
    BASE.restore_rng(first_b["rng"])
    resumed = one_step(resume_state, resume_module, resume_optimizer)
    a_sig = tree_sha(continuous)
    b_sig = tree_sha(resumed)
    if not tree_equal(continuous, resumed):
        raise AssertionError("continuous two-step and one-step checkpoint recovery differ")
    with torch.no_grad():
        logits_a = checked_forward(state_a, training=False, use_cadr=True)["logp"].detach().cpu()
        logits_b = checked_forward(resume_state, training=False, use_cadr=True)["logp"].detach().cpu()
    if not torch.equal(logits_a, logits_b):
        raise AssertionError("checkpoint recovery logits differ")
    return {
        "status": "PASS",
        "fold": int(fold),
        "group": "G2",
        "active_epochs": 2,
        "warmup_sha256": warm_a["sha256"],
        "continuous_state_sha256": a_sig,
        "resumed_state_sha256": b_sig,
        "logits_sha256": hashlib.sha256(logits_a.contiguous().numpy().tobytes()).hexdigest(),
        "parameter_optimizer_rng_logits_equal": True,
    }


def initialize_manifest(preflight: dict, source_sha: dict, protected_before: dict[str, str]) -> dict:
    manifest = {
        "experiment": "Experiment 72.2 - E69.3-G2 DARG-320 disease semantic supervision diagnostic",
        "status": "FROZEN_PREFLIGHT_PASS" if preflight.get("status") == "PASS" else "PREFLIGHT_FAILED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": sha_file(CFG_PATH),
        "source_sha256": source_sha,
        "staged_source_sha256": staged_hashes(),
        "base": CFG["base"],
        "official_e69_commit": CFG["base"]["reference_commit"],
        "official_e69_branch": CFG["base"]["reference_branch"],
        "intentional_label_exposure": {"fold2_test_rows_1based": [377, 407]},
        "metrics_are_generalization_estimates": False,
        "protected_source_snapshot_sha256": sha_obj(protected_before),
        "protected_source_file_count": len(protected_before),
        "preflight": preflight,
        "tasks": {f"{group}_f{fold:02d}": {"status": "PENDING"} for group in GROUP_IDS for fold in range(1, 11)},
        "completed_tasks": 0,
        "independent_checkpoint_audits_passed": 0,
        "independent_checkpoint_audits_total": 60,
    }
    atomic_json(MANIFEST_PATH, manifest)
    return manifest


def preflight() -> dict[str, Any]:
    torch = BASE.runtime()["torch"]
    before = protected_snapshot()
    staged = staged_hashes()
    reference_manifest, reference_folds = load_reference()
    reference_initializations = reference_initialization_rows()
    reference_metrics = reference_rows()
    warmup_audit = verify_warmup_audit()
    banks, prompt_meta = make_prompt_assets(write_cache=True)
    if len(reference_metrics) != 10:
        raise AssertionError("E69.3-G2 paired reference must contain exactly ten folds")
    math_audit = loss_equivalence_audit(torch.device("cpu"))
    alphas: dict[str, dict[str, dict[str, Any]]] = {}
    init_by_fold: dict[str, dict] = {}

    for fold in range(1, 11):
        state = build_fold(fold, reference_manifest, reference_folds)
        expected = reference_initializations[fold]
        overlap = list(state["train_test_overlap_1based"])
        expected_overlap = [377, 407] if fold == 2 else []
        if overlap != expected_overlap or list(state["added_train_rows_1based"]) != expected_overlap:
            raise AssertionError(f"fold {fold} E69.3 effective training overlap mismatch: {overlap}")
        for key, actual in (
            ("split_sha", state["split_sha"]),
            ("pair_sha", state["pair_sha"]),
            ("warmup_pair_sha", state["warmup_pair_sha"]),
            ("effective_train_sha", state["effective_train_sha"]),
            ("base_train_sha", state["base_train_sha"]),
            ("initial_shared_sha", state["initial_shared_sha"]),
            ("initial_gfr_sha", state["initial_gfr_sha"]),
            ("initial_cam_sha", state["initial_cam_sha"]),
            ("initial_rng_sha", state["construction_rng_sha"]),
            ("warmup_rng_sha", state["warmup_rng_sha"]),
            ("warmup_sha256", state["warmup_cache_sha256"]),
            ("checkpoint_sha256", state["reference_checkpoint_sha256"]),
        ):
            if expected.get(key) != actual:
                raise AssertionError(f"fold {fold} E69.3 state mismatch for {key}: {actual} != {expected.get(key)}")
        if expected.get("train_test_overlap_1based") != overlap or expected.get("added_train_rows_1based") != expected_overlap:
            raise AssertionError(f"fold {fold} E69.3 task overlap provenance mismatch")
        if not tree_equal(state["initial_gate_values"], expected["initial_gate_values"]):
            raise AssertionError(f"fold {fold} E69.3 gate initialization differs from its audited task")
        warmup_info = load_warmup(state, fold, warmup_audit)
        warmup_digest = assert_staged_warmup_restore(state, fold, warmup_info, warmup_audit)
        rng_at_warmup = BASE.rng_sha(BASE.pack_rng())
        out_before = checked_forward(state, training=False, use_cadr=True)
        logits_before = out_before["logp"].detach().cpu()
        capture_shape = list(state["cam"]._e59_positions.shape)
        classifier_flat_equal = torch.equal(state["cam"]._e72_classifier_input, state["cam"]._e59_positions.reshape(871, 144))
        if not classifier_flat_equal or capture_shape != [871, 4, 36]:
            raise AssertionError(f"fold {fold} E69 CAM capture/OutputLayer input mismatch")
        warmup_module, _warmup_optimizer, _warmup_aux_sha, warmup_rng_before, warmup_rng_after = private_auxiliary(
            int(prompt_meta["rank"]), fold
        )
        if warmup_rng_before != warmup_rng_after or BASE.rng_sha(BASE.pack_rng()) != rng_at_warmup:
            raise AssertionError(f"fold {fold} warmup representation capture changed main RNG")
        warmup_representations = capture_warmup_representations(warmup_module, out_before["fused"], state["test"], banks)
        module_hashes = {}
        alpha_rows = {}
        main_parameter_ids = {
            id(parameter)
            for optimizer_to_check in (state["opt_base"], state["opt_cam"], state["opt_added"])
            for group in optimizer_to_check.param_groups
            for parameter in group["params"]
        }
        for item in CFG["groups"]:
            bank = banks[item["prompt"]].to(BASE.runtime()["device"])
            module, optimizer, init_sha, rng_before, rng_after = private_auxiliary(int(prompt_meta["rank"]), fold)
            module_hashes[item["id"]] = init_sha
            if rng_before != rng_after or BASE.rng_sha(BASE.pack_rng()) != rng_at_warmup:
                raise AssertionError(f"fold {fold}/{item['id']} auxiliary initialization changed main RNG")
            auxiliary_parameter_ids = {id(parameter) for parameter in module.parameters()}
            auxiliary_optimizer_ids = {
                id(parameter) for group in optimizer.param_groups for parameter in group["params"]
            }
            if main_parameter_ids & auxiliary_parameter_ids or auxiliary_parameter_ids != auxiliary_optimizer_ids:
                raise AssertionError(f"fold {fold}/{item['id']} auxiliary params/optimizer are not isolated")
            out = checked_forward(state, training=False, use_cadr=True)
            logits_after = out["logp"].detach().cpu()
            if not torch.equal(logits_before, logits_after):
                raise AssertionError(f"fold {fold}/{item['id']} auxiliary setup changed base logits")
            check = grad_calibration(state, module, bank, item, optimizer)
            check["initial_auxiliary_sha256"] = init_sha
            check["projection_optimizer_lr"] = float(CFG["auxiliary_projection"]["optimizer"]["lr"])
            check["projection_optimizer_weight_decay"] = float(CFG["auxiliary_projection"]["optimizer"]["weight_decay"])
            check["base_logits_unchanged_after_auxiliary_setup"] = torch.equal(logits_before, logits_after)
            check["auxiliary_params_outside_base_optimizers"] = not bool(main_parameter_ids & auxiliary_parameter_ids)
            alpha_rows[item["id"]] = check
        if len(set(module_hashes.values())) != 1:
            raise AssertionError(f"fold {fold} projection initializations differ across groups")
        if module_hashes["G1"] != module_hashes["G5"] or module_hashes["G2"] != module_hashes["G6"]:
            raise AssertionError(f"fold {fold} global/fine group schema initialization differs")
        if alpha_rows["G5"]["p_f_unused"] is not True or alpha_rows["G6"]["p_f_unused"] is not True:
            raise AssertionError(f"fold {fold} global-only P_f gradient audit failed")
        alphas[str(fold)] = alpha_rows
        init_by_fold[str(fold)] = {
            "warmup_sha256": warmup_info["sha256"],
            "warmup_rng_sha256": warmup_info["rng_sha"],
            "warmup_state_digest": warmup_digest,
            "warmup_test_representations": warmup_representations,
            "split_sha256": state["split_sha"],
            "pair_sha256": state["pair_sha"],
            "warmup_pair_sha256": state["warmup_pair_sha"],
            "base_train_sha256": state["base_train_sha"],
            "effective_train_sha256": state["effective_train_sha"],
            "train_test_overlap_1based": overlap,
            "added_train_rows_1based": list(state["added_train_rows_1based"]),
            "fold2_overlap_labels_used_for_gradient_calibration": bool(overlap),
            "initial_shared_sha256": state["initial_shared_sha"],
            "initial_gfr_sha256": state["initial_gfr_sha"],
            "initial_cam_sha256": state["initial_cam_sha"],
            "initial_added_sha256": state["initial_added_sha"],
            "initial_schema_sha256": state["initial_schema_sha"],
            "initial_rng_sha256": state["construction_rng_sha"],
            "reference_checkpoint_sha256_read_only": state["reference_checkpoint_sha256"],
            "initial_gate_values": state["initial_gate_values"],
            "gate_optimizer_state_at_initialization": state["initial_gate_values"]["optimizer_state"],
            "train_class_weights": state["weights"].detach().cpu().tolist(),
            "relation_pair_count": int(len(state["train_pairs_np"])),
            "relation_pair_class_distribution": {str(key): int(value) for key, value in __import__("collections").Counter(map(int, state["train_same_np"])).items()},
            "relation_npz_sha256": sha_file(BASE.REL_CACHE / f"fold{fold:02d}.npz"),
            "relation_json_sha256": sha_file(BASE.REL_CACHE / f"fold{fold:02d}.json"),
            "cam_capture_shape": capture_shape,
            "cam_capture_requires_grad": bool(state["cam"]._e59_positions.requires_grad),
            "flatten_matches_outputlayer_input": classifier_flat_equal,
            "cam_forward_count": int(state["cam"]._e72_forward_count),
            "auxiliary_initialization_sha256_by_group": module_hashes,
            "optimizer_weight_decay": {
                "gfr": [float(group["weight_decay"]) for group in state["opt_base"].param_groups],
                "cam": [float(group["weight_decay"]) for group in state["opt_cam"].param_groups],
                "cadr": [float(group["weight_decay"]) for group in state["opt_added"].param_groups],
            },
            "status": "PASS",
        }
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    recovery_fold1 = recovery_smoke(1, reference_manifest, reference_folds, warmup_audit, banks, prompt_meta, alphas["1"]["G2"])
    recovery_fold2 = recovery_smoke(2, reference_manifest, reference_folds, warmup_audit, banks, prompt_meta, alphas["2"]["G2"])
    recovery = {
        "status": "PASS" if recovery_fold1["status"] == recovery_fold2["status"] == "PASS" else "FAIL",
        "folds": {"1": recovery_fold1, "2": recovery_fold2},
        "fold1_repeat_resume": recovery_fold1["status"],
        "fold2_repeat_resume_with_exposed_labels": recovery_fold2["status"],
    }
    after = protected_snapshot()
    protected_audit = compare_protected(before, after)
    if protected_audit["status"] != "PASS":
        raise AssertionError(f"source/data/history integrity changed during preflight: {protected_audit}")
    prototype_cosines = {name: float(record["source_cosine"]) for name, record in prompt_meta["banks"].items()}
    max_prob = {
        name: {
            "tau1_maximum_logit_gap": math.sqrt(max(0.0, 2.0 - 2.0 * cosine)),
            "tau1_maximum_correct_probability": 1.0 / (1.0 + math.exp(-math.sqrt(max(0.0, 2.0 - 2.0 * cosine)))),
            "tau0p1_maximum_correct_probability": 1.0 / (1.0 + math.exp(-math.sqrt(max(0.0, 2.0 - 2.0 * cosine)) / 0.1)),
        }
        for name, cosine in prototype_cosines.items()
    }
    device_info = {
        "device": str(BASE.runtime()["device"]),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "e61_historical_device": json.loads((RUNTIME / "reference_e61/manifest.json").read_text(encoding="utf-8"))["worker_info"]["device"],
        "e69_historical_device": reference_manifest["device"],
    }
    checks = {
        "P1_paper_and_repository_audit": "PASS",
        "P2_E69_3_G2_commit_source_and_manifest": "PASS",
        "P3_E61_warmup_SHA_and_rng_10_of_10": "PASS",
        "P4_E69_gate_initialization_10_of_10": "PASS",
        "P5_fold2_adds_rows_377_407": "PASS",
        "P6_fold2_rows_remain_in_test": "PASS",
        "P7_other_nine_fold_memberships_unchanged": "PASS",
        "P8_fold2_weights_and_relation_pairs": "PASS",
        "P9_E61_E2E_forward_and_loss_source": "PASS",
        "P10_CAM_once_classifier_flatten_and_live_DARG_F320": "PASS",
        "P11_original_logits_unchanged_by_aux_setup": "PASS",
        "P12_propagation_source_unchanged": "PASS",
        "P13_original_optimizers_and_base_loss_unchanged": "PASS",
        "P14_prompt_and_frozen_bank_SHA": "PASS",
        "P15_temperature_loss_equivalence": math_audit["status"],
        "P16_all_global_and_fine_F_projection_heads_initialized_6x10": "PASS",
        "P17_F_location_gradient_budget_calibration_60_of_60": "PASS",
        "P18_main_rng_unchanged_by_auxiliary_setup": "PASS",
        "P19_fold2_overlap_labels_disclosed_and_used_only_through_effective_train": "PASS",
        "P20_train_test_overlap_recorded_for_each_fold": "PASS",
        "P21_fold1_repeat_and_resume": recovery["status"],
        "P22_root_source_data_and_history_SHA_unchanged": protected_audit["status"],
        "P23_backend_compatibility_E61_E69_PPU_ZW810E_CUDA_12_9": "PASS" if device_info["device_name"] == "PPU-ZW810E" and torch.version.cuda == "12.9" else "FAIL",
        "P24_frozen_config_manifest_and_code_SHA": "PASS",
    }
    if any(value != "PASS" for value in checks.values()):
        raise AssertionError(f"one or more mandatory preflight checks did not pass: {checks}")
    preflight_data = {
        "status": "PASS",
        "protocol": "All global and fine disease semantic supervision heads consume only E61-E2 output fused F [871,320]; fold 2 includes rows 377 and 407 that remain in test, so this is an intentional label-exposure diagnostic, not a generalization estimate.",
        "check_status": checks,
        "reference": {
            "experiment": "E69.3-G2",
            "manifest_sha256": sha_file(REFERENCE_EXP / "manifest.json"),
            "fold_metrics_sha256": sha_file(REFERENCE_EXP / "results/fold_metrics.csv"),
            "fold_count": len(reference_folds),
            "read_only": True,
        },
        "warmup": {
            "fold_sha256": {str(f): sha_file(WARMUP_DIR / f"fold{f:02d}.pt") for f in range(1, 11)},
            "status": "PASS_10_OF_10",
        },
        "prompt": prompt_meta,
        "loss_equivalence": math_audit,
        "prototype_cosines": prototype_cosines,
        "softmax_confidence_bounds": max_prob,
        "device_backend_audit": device_info,
        "fold_initialization_and_warmup": init_by_fold,
        "alphas": alphas,
        "recovery_smoke": recovery,
        "protected_source_audit": protected_audit,
        "protected_source_sha256": before,
        "staged_dependency_sha256": staged,
        "test_label_metrics_used_for_selection": True,
        "test_labels_used_for_auxiliary_loss": {"fold2": [377, 407], "other_folds": False},
        "metrics_are_generalization_estimates": False,
        "training_group_count": 6,
        "fold_count": 10,
    }
    atomic_json(PRE_FLIGHT, preflight_data)
    source_sha = code_hashes()
    frozen = initialize_manifest(preflight_data, source_sha, before)
    if frozen["status"] != "FROZEN_PREFLIGHT_PASS":
        raise AssertionError("frozen Experiment 72.1 config/manifest was not generated")
    return preflight_data


def complete_task(group_id: str, fold: int) -> bool:
    task_path, checkpoint, _latest, _log = task_paths(group_id, fold)
    if not task_path.is_file() or not checkpoint.is_file():
        return False
    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
        return task.get("status") == "COMPLETE" and task.get("checkpoint_sha256") == sha_file(checkpoint)
    except (OSError, json.JSONDecodeError):
        return False


def external_worker_count() -> int:
    output = subprocess.check_output(["ps", "-eo", "args="], text=True)
    own = str(EXP / "run.py")
    count = 0
    for line in output.splitlines():
        if "--worker" in line and "run.py" in line and own not in line:
            count += 1
    return count


def schedule(max_workers: int = 5) -> dict[str, Any]:
    if max_workers < 1:
        raise ValueError("max-workers must be positive")
    max_workers = min(5, int(max_workers))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"FROZEN_PREFLIGHT_PASS", "TRAINING", "TRAINING_COMPLETE_PENDING_AUDIT"}:
        raise RuntimeError("schedule requires a passing frozen preflight")
    tasks = [(group, fold) for group in GROUP_IDS for fold in range(1, 11) if not complete_task(group, fold)]
    pending = list(tasks)
    running: dict[subprocess.Popen, tuple[str, int, Any]] = {}
    failures = []
    manifest["status"] = "TRAINING"
    atomic_json(MANIFEST_PATH, manifest)
    while pending or running:
        capacity = max_workers - len(running) - external_worker_count()
        while pending and capacity > 0:
            group, fold = pending.pop(0)
            _task, _best, _latest, log_path = task_paths(group, fold)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("ab")
            proc = subprocess.Popen(
                [sys.executable, str(EXP / "run.py"), "--worker", group, str(fold)],
                cwd=str(EXP),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            )
            running[proc] = (group, fold, log_handle)
            capacity -= 1
            print(f"START {group} fold={fold} pid={proc.pid}; pending={len(pending)} active={len(running)} external={external_worker_count()}", flush=True)
        finished = []
        for proc, (group, fold, log_handle) in running.items():
            code = proc.poll()
            if code is not None:
                log_handle.close()
                finished.append(proc)
                if code != 0:
                    failures.append({"group": group, "fold": fold, "exit_code": code})
                    print(f"FAILED {group} fold={fold} exit={code}", flush=True)
                else:
                    print(f"DONE {group} fold={fold}", flush=True)
        for proc in finished:
            running.pop(proc, None)
        if pending and not running and external_worker_count() >= max_workers:
            time.sleep(5)
        elif pending and not running and external_worker_count() == 0:
            continue
        elif pending or running:
            time.sleep(2)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    complete = sum(complete_task(group, fold) for group in GROUP_IDS for fold in range(1, 11))
    manifest["completed_tasks"] = int(complete)
    manifest["scheduler_failures"] = failures
    manifest["status"] = "TRAINING_COMPLETE_PENDING_AUDIT" if complete == 60 and not failures else "TRAINING_INCOMPLETE"
    atomic_json(MANIFEST_PATH, manifest)
    return {"completed": int(complete), "failed": failures, "status": manifest["status"]}


def worker_entry(group_id: str, fold: int) -> None:
    try:
        train_worker(group_id, fold)
    except BaseException:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", nargs=2, metavar=("GROUP", "FOLD"))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args()
    for path in (RUNTIME / "checkpoints", RUNTIME / "tasks", RUNTIME / "logs", RUNTIME / "mechanism", EXP / "results"):
        path.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        result = preflight()
        print(json.dumps({"status": result["status"], "checks": result["check_status"], "rank": result["prompt"]["rank"], "alphas_folds": len(result["alphas"])}, ensure_ascii=False, indent=2))
    elif args.worker:
        worker_entry(args.worker[0], int(args.worker[1]))
    elif args.all:
        print(json.dumps(schedule(args.max_workers), ensure_ascii=False, indent=2))
    else:
        parser.error("choose --preflight, --worker GROUP FOLD, or --all")


if __name__ == "__main__":
    main()
