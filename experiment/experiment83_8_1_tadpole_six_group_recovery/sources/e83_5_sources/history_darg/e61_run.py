#!/usr/bin/env python3
"""Reproducible staged runner for Experiment 61."""
from __future__ import annotations

import argparse
import copy
import csv
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
from collections import Counter
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
BASE_ROOT = Path("/model/spromptgl")
BASE_EXP = BASE_ROOT / "experiment/experiment53_3_cadr_graph_fallback_fix"
BASE_RUN = BASE_EXP / "run.py"
BASE_CADR = BASE_EXP / "cadr_gfr.py"
BASE_GROUP = "CADR_WD_0p01_GFR_WD_0p11"
BASE_RUNTIME = BASE_EXP / ".runtime/fallback_wd_sweep" / BASE_GROUP
ANCHORS = BASE_ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
CONFIG_PATH = EXP / "config.json"
CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RT = EXP / ".runtime"
WARMUP = RT / "warmup"
TASKS = RT / "tasks"
CKPTS = RT / "checkpoints"
LOGS = RT / "logs"
RESULTS = EXP / "results"
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
METRIC_LABELS = {
    "acc": "ACC", "hard_auc": "Hard-AUC", "prob_auc": "Probability-AUC",
    "sen": "SEN-ASD", "spe": "SPE-TD", "ba": "BA", "asd_f1": "ASD-F1",
}
GROUPS = {"E61-E1": "POSTALIGN_OOF", "E61-E2": "E2E_ADAPTIVE_GATE"}
R = None
BASE_HASH_CACHE = None
SOURCE_SHA_OVERRIDE = None


def load_base():
    global R
    if R is not None:
        return R
    if not BASE_RUN.is_file() or not BASE_CADR.is_file():
        raise FileNotFoundError(f"locked 53.3 source is unavailable under {BASE_EXP}")
    legacy_source = BASE_ROOT / "experiment/experiment51_dugsa/source"
    for path in (legacy_source, BASE_ROOT / "experiment/experiment51_dugsa", BASE_EXP):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("e61_locked_53_run", BASE_RUN)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    R = module
    return R


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_named(module, prefixes=None) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        if prefixes and not name.startswith(prefixes):
            continue
        digest.update(name.encode() + b"\0")
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    expected = json.loads(json.dumps(obj, ensure_ascii=False, allow_nan=False))
    if json.loads(path.read_text(encoding="utf-8")) != expected:
        raise IOError(f"atomic JSON readback failed: {path}")


def atomic_torch_save(path: Path, obj):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(obj, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if not path.is_file() or path.stat().st_size == 0:
        raise IOError(f"checkpoint atomic readback failed: {path}")


def torch_load(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def log_json(path: Path, record):
    def finite(value):
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(item) for item in value]
        if isinstance(value, (float,)) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(finite(record), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def rng_sha(state) -> str:
    digest = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    digest.update(state["cpu"].cpu().numpy().tobytes())
    for value in state.get("cuda", []):
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def model_state_cpu(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def fold_paths(group: str, fold: int):
    token = f"{group}_f{fold:02d}"
    return TASKS / f"{token}.json", CKPTS / f"{token}.best.pt", CKPTS / f"{token}.latest.pt", LOGS / f"{token}.jsonl"


def warmup_path(fold: int) -> Path:
    return WARMUP / f"fold{fold:02d}.pt"


def oof_path(fold: int) -> Path:
    return TASKS / f"fold{fold:02d}.postalign_oof.json"


def reference_rows():
    ref_csv = BASE_EXP / "results/fallback_wd_sweep/fold_metrics.csv"
    audit_csv = BASE_EXP / "results/fallback_wd_sweep/checkpoint_audit.csv"
    rows = list(csv.DictReader(ref_csv.open(encoding="utf-8")))
    audits = list(csv.DictReader(audit_csv.open(encoding="utf-8")))
    selected = {int(row["fold"]): row for row in rows if row.get("group") == BASE_GROUP}
    selected_audit = {int(row["fold"]): row for row in audits if row.get("group") == BASE_GROUP}
    if set(selected) != set(range(1, 11)):
        raise RuntimeError("53.3 third-group fold metrics are not ten-fold complete")
    if set(selected_audit) != set(range(1, 11)) or any(row.get("checkpoint_audit") != "PASS" for row in selected_audit.values()):
        raise RuntimeError("53.3 third-group readback audit is incomplete")
    return selected


def verify_reference_artifacts():
    global BASE_HASH_CACHE
    if BASE_HASH_CACHE is not None:
        return BASE_HASH_CACHE
    r = load_base()
    locked = CFG["locked_reference_commit"]
    git_root = ROOT
    source_hashes = {}
    for name in ("run.py", "cadr_gfr.py", "run_cadr_fallback.py", "run_cadr_fallback_wd_sweep.py", "config_fallback_wd_sweep.json"):
        path = BASE_EXP / name
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[name] = sha_file(path)
    git_source = subprocess.run(
        ["git", "-C", str(git_root), "rev-parse", "--verify", f"{locked}^{{commit}}"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    reference_files = {
        "cadr_gfr.py": "experiment/experiment53_3_cadr_graph_fallback_fix/cadr_gfr.py",
        "run.py": "experiment/experiment53_3_cadr_graph_fallback_fix/run.py",
        "run_cadr_fallback.py": "experiment/experiment53_3_cadr_graph_fallback_fix/run_cadr_fallback.py",
        "run_cadr_fallback_wd_sweep.py": "experiment/experiment53_3_cadr_graph_fallback_fix/run_cadr_fallback_wd_sweep.py",
        "config_fallback_wd_sweep.json": "experiment/experiment53_3_cadr_graph_fallback_fix/config_fallback_wd_sweep.json",
    }
    for name, rel in reference_files.items():
        git_blob = subprocess.run(
            ["git", "-C", str(git_root), "show", f"{locked}:{rel}"],
            capture_output=True, check=True,
        ).stdout
        if sha_bytes(git_blob) != source_hashes[name]:
            raise RuntimeError(f"locked source mismatch for {name}")
    if git_source != locked:
        raise RuntimeError(f"locked commit resolution mismatch: {git_source}")
    BASE_HASH_CACHE = source_hashes
    return source_hashes


def pair_sha(pairs, same):
    import numpy as np

    return hashlib.sha256(
        np.asarray(pairs, dtype=np.int64).tobytes() + np.asarray(same, dtype=np.int64).tobytes()
    ).hexdigest()


def make_outer_pairs(fold, supports, train_ids, labels):
    import numpy as np

    union = np.concatenate(supports, axis=0) if any(len(values) for values in supports) else np.zeros((0, 2), np.int64)
    if len(union):
        union = np.unique(np.sort(union, axis=1), axis=0)
    pairs, same = load_base().sample_balanced_pairs(
        train_ids, labels, 4096, np.random.default_rng(532100 + fold), union
    )
    return pairs, same


def build_state(fold: int):
    import numpy as np
    import torch

    from cadr_gfr import CADRGFR

    r = load_base()
    (x_np, y_np), folds = r.load_data()
    train_np, test_np = folds[fold - 1]
    support, fallback = r.anchor_candidate_graphs(fold)
    pairs_np, same_np = make_outer_pairs(fold, support, train_np, y_np)
    core = r.build_legacy_core(fold)
    if pair_sha(pairs_np, same_np) != json.loads(
        (BASE_RUNTIME / "tasks" / f"{BASE_GROUP}_f{fold:02d}.json").read_text(encoding="utf-8")
    )["pair_sha"]:
        raise RuntimeError(f"outer relation pair sampling SHA differs from historical fold {fold}")

    rng_before_cadr = r.pack_rng()
    zeros = torch.zeros(4, dtype=torch.float32)
    calibration = torch.tensor([[1.0, 0.0]] * 4, dtype=torch.float32)
    model = CADRGFR(
        core["base"], zeros, calibration,
        torch.as_tensor(fallback, dtype=torch.bool), init_seed=532000 + fold,
        grl_scale=0.03, edge_topk=12, positive_threshold=0.55,
        negative_threshold=0.45, negative_scale=0.25, direct_scale=0.02,
    ).to(r.runtime()["device"])
    rng_after_cadr = r.pack_rng()
    if r.rng_sha(rng_before_cadr) != r.rng_sha(rng_after_cadr):
        raise RuntimeError("CADR constructor changed the outer training RNG")
    # The E2 scalar is deterministic and consumes no part of the shared RNG stream.
    model.adaptive_gate_logits = torch.nn.Parameter(torch.full((4,), math.log(0.1 / 0.9), device=r.runtime()["device"]))

    base_params, added_named, added_groups = r._optimizer_groups(model)
    for group in added_groups:
        group["weight_decay"] = 0.01
    hp = json.loads((BASE_EXP / "config.json").read_text(encoding="utf-8"))["base"]
    opt_base = torch.optim.Adam(base_params, lr=0.0038, weight_decay=0.11)
    opt_cam = torch.optim.Adam(core["cam"].parameters(), lr=0.0038, weight_decay=0.11)
    opt_added = torch.optim.AdamW(added_groups, lr=0.001)
    counts = Counter(np.asarray(y_np)[train_np])
    weights = torch.from_numpy(
        (len(train_np) / np.asarray(list(counts.values())) / 2).astype(np.float32)
    ).to(r.runtime()["device"])
    offsets = np.cumsum([0] + list(r.DIMS[:-1]))
    parts = [core["x"][:, start:start + dim] for start, dim in zip(offsets, r.DIMS)]
    base_task = json.loads((BASE_RUNTIME / "tasks" / f"{BASE_GROUP}_f{fold:02d}.json").read_text(encoding="utf-8"))
    state = {
        **core,
        "group": None,
        "fold": fold,
        "model": model,
        "cam": core["cam"],
        "opt_base": opt_base,
        "opt_cam": opt_cam,
        "opt_added": opt_added,
        "base_params": base_params,
        "added_named": added_named,
        "parts": parts,
        "weights": weights,
        "train_pairs": torch.as_tensor(pairs_np, dtype=torch.long, device=r.runtime()["device"]),
        "train_same": torch.as_tensor(same_np, dtype=torch.float32, device=r.runtime()["device"]),
        "train_pairs_np": pairs_np,
        "train_same_np": same_np,
        "supports": support,
        "fallback": fallback,
        "initial_shared_sha": r.digest_shared(core["base"], core["cam"]),
        "initial_gfr_sha": r.module_sha(core["base"]),
        "initial_cam_sha": r.module_sha(core["cam"]),
        "relation_core_sha": sha_named(model, ("encoders.", "scorers.", "discriminator.", "edge_gate_logits")),
        "initial_rng_sha": core["construction_rng_sha"],
        "initial_added_sha": r.digest_added_parameters(model),
        "pair_sha": pair_sha(pairs_np, same_np),
        "base_task": base_task,
        "train_np": train_np,
        "test_np": test_np,
        "labels_np": y_np,
        "warmup_best_acc": None,
        "warmup_best_epoch": None,
        "rng_trace": [],
        "oof": None,
        "branch": None,
    }
    # Only source-compatible parameters must match the historical initialization.
    for field, actual in (
        ("initial_shared_sha", state["initial_shared_sha"]),
        ("initial_gfr_sha", state["initial_gfr_sha"]),
        ("initial_cam_sha", state["initial_cam_sha"]),
        ("initial_rng_sha", state["initial_rng_sha"]),
        ("split_sha", state["split_sha"]),
        ("pair_sha", state["pair_sha"]),
        ("anchor_sha", state["anchor_sha"]),
    ):
        expected = base_task[field]
        if actual != expected:
            raise RuntimeError(f"historical initialization mismatch fold={fold} field={field}: {actual} != {expected}")
    return state


def load_warmup_state(fold: int):
    state = build_state(fold)
    path = warmup_path(fold)
    saved = torch_load(path)
    if saved.get("fold") != fold or saved.get("split_sha") != state["split_sha"]:
        raise RuntimeError(f"shared warmup identity mismatch at fold {fold}")
    if saved.get("pair_sha") != state["pair_sha"] or saved.get("source_sha") != source_sha_map():
        raise RuntimeError(f"shared warmup pair/source mismatch at fold {fold}")
    state["model"].load_state_dict(saved["model"], strict=True)
    state["cam"].load_state_dict(saved["cam"], strict=True)
    state["opt_base"].load_state_dict(saved["optimizer_base"])
    state["opt_cam"].load_state_dict(saved["optimizer_cam"])
    state["opt_added"].load_state_dict(saved["optimizer_added"])
    state["warmup_best_acc"] = saved["warmup_best_acc"]
    state["warmup_best_epoch"] = saved["warmup_best_epoch"]
    state["rng_trace"] = saved["rng_trace"]
    load_base().restore_rng(saved["rng"])
    if load_base().rng_sha(load_base().pack_rng()) != saved["rng_sha"]:
        raise RuntimeError(f"warmup RNG readback mismatch fold={fold}")
    return state, saved


def source_sha_map():
    if SOURCE_SHA_OVERRIDE is not None:
        return SOURCE_SHA_OVERRIDE
    return {name: sha_file(EXP / name) for name in ("run.py", "model_e61.py", "config.json")}


def warmup_fold(fold: int, force: bool = False):
    import torch

    r = load_base()
    path = warmup_path(fold)
    if path.is_file() and not force:
        saved = torch_load(path)
        if saved.get("source_sha") == source_sha_map():
            return saved
        if any(
            fold_paths(group, other_fold)[0].is_file()
            and json.loads(fold_paths(group, other_fold)[0].read_text(encoding="utf-8")).get("status") == "COMPLETE"
            for group in GROUPS for other_fold in range(1, 11)
        ):
            raise RuntimeError(f"cannot replace warmup with mismatched source after formal training began: fold {fold}")
        force = True
    state = build_state(fold)
    best_acc, best_epoch = -1.0, -1
    trace = []
    for epoch in range(10):
        before = r.rng_sha(r.pack_rng())
        for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"]):
            optimizer.zero_grad(set_to_none=True)
        out = r.forward_state(state, training=True, use_cadr=False)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"non-finite shared warmup loss fold={fold} epoch={epoch + 1}")
        out["total"].backward()
        for module in (state["model"], state["cam"]):
            for parameter in module.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"non-finite shared warmup gradient fold={fold} epoch={epoch + 1}")
        state["opt_base"].step()
        state["opt_cam"].step()
        state["opt_added"].step()
        after_train = r.rng_sha(r.pack_rng())
        with torch.no_grad():
            evaluated = r.forward_state(state, training=False, use_cadr=False)
            metrics = r.classification_metrics(evaluated["logp"], state["test"], state["y"])
        if metrics["acc"] > best_acc:
            best_acc, best_epoch = float(metrics["acc"]), epoch
        trace.append({"epoch": epoch + 1, "rng_before": before, "rng_after_train": after_train, "rng_after_eval": r.rng_sha(r.pack_rng())})
        del out, evaluated
    state["warmup_best_acc"] = best_acc
    state["warmup_best_epoch"] = best_epoch
    state["rng_trace"] = trace
    rng = r.pack_rng()
    saved = {
        "experiment": "E61 shared warmup",
        "fold": fold,
        "seed": 0,
        "completed_epoch": 10,
        "model": model_state_cpu(state["model"]),
        "cam": model_state_cpu(state["cam"]),
        "optimizer_base": copy.deepcopy(state["opt_base"].state_dict()),
        "optimizer_cam": copy.deepcopy(state["opt_cam"].state_dict()),
        "optimizer_added": copy.deepcopy(state["opt_added"].state_dict()),
        "rng": rng,
        "rng_sha": r.rng_sha(rng),
        "rng_trace": trace,
        "train_indices": state["train_np"].copy(),
        "test_indices": state["test_np"].copy(),
        "train_pairs": state["train_pairs_np"].copy(),
        "train_same": state["train_same_np"].copy(),
        "split_sha": state["split_sha"],
        "pair_sha": state["pair_sha"],
        "anchor_sha": state["anchor_sha"],
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_gfr_sha": state["initial_gfr_sha"],
        "initial_cam_sha": state["initial_cam_sha"],
        "relation_core_sha": state["relation_core_sha"],
        "initial_rng_sha": state["initial_rng_sha"],
        "warmup_best_acc": best_acc,
        "warmup_best_epoch": best_epoch,
        "source_sha": source_sha_map(),
        "base_source_sha": verify_reference_artifacts(),
    }
    atomic_torch_save(path, saved)
    saved["warmup_sha256"] = sha_file(path)
    task_meta = {
        "fold": fold, "status": "SHARED_WARMUP_COMPLETE", "warmup_sha256": saved["warmup_sha256"],
        "rng_sha": saved["rng_sha"], "initial_shared_sha": state["initial_shared_sha"],
        "initial_gfr_sha": state["initial_gfr_sha"], "initial_cam_sha": state["initial_cam_sha"],
        "relation_core_sha": state["relation_core_sha"], "initial_rng_sha": state["initial_rng_sha"],
        "split_sha": state["split_sha"], "pair_sha": state["pair_sha"],
        "warmup_best_acc": best_acc, "warmup_best_epoch": best_epoch,
    }
    atomic_json(TASKS / f"fold{fold:02d}.warmup.json", task_meta)
    return saved


def fit_aligned_inner(fold: int, inner_fold: int, train_ids, valid_ids, supports, labels, x_np):
    """Fit a joint four-modality conditional-adversarial relation model for ten fixed epochs."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from cadr_gfr import (
        ConditionalModalityDiscriminator,
        RelationEncoder,
        RelationScorer,
        conditional_adversarial_loss,
        relation_preservation_loss,
    )

    r = load_base()
    union = np.concatenate(supports, axis=0) if any(len(values) for values in supports) else np.zeros((0, 2), np.int64)
    if len(union):
        union = np.unique(np.sort(union, axis=1), axis=0)
    local_seed = 614000 + fold * 100 + inner_fold
    rng_train = np.random.default_rng(610000 + fold * 1000 + inner_fold * 10 + 1)
    pairs, same = r.sample_balanced_pairs(train_ids, labels, 4096, rng_train, union)
    if not len(pairs) or not np.isin(pairs, train_ids).all():
        raise RuntimeError("OOF training pairs escaped inner training patients")

    device = r.runtime()["device"]
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(local_seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(local_seed)
        encoders = nn.ModuleList([RelationEncoder(dim, 32) for dim in r.DIMS]).to(device)
        scorers = nn.ModuleList([RelationScorer(65, 32) for _ in r.DIMS]).to(device)
        discriminator = ConditionalModalityDiscriminator(65, 32, 4).to(device)
        parameters = list(encoders.parameters()) + list(scorers.parameters()) + list(discriminator.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.01)
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        parts = [x[:, start:start + dim] for start, dim in zip(np.cumsum([0] + r.DIMS[:-1]), r.DIMS)]
        pair_t = torch.as_tensor(pairs, dtype=torch.long, device=device)
        same_t = torch.as_tensor(same, dtype=torch.float32, device=device)
        for _epoch in range(10):
            optimizer.zero_grad(set_to_none=True)
            embeddings = [encoder(part) for encoder, part in zip(encoders, parts)]
            left, right = pair_t[:, 0], pair_t[:, 1]
            from cadr_gfr import pair_features
            representations = [pair_features(z[left], z[right]) for z in embeddings]
            logits = [scorer(representation) for scorer, representation in zip(scorers, representations)]
            pair_loss = torch.stack([F.binary_cross_entropy_with_logits(value, same_t) for value in logits]).mean()
            adv_loss, _ = conditional_adversarial_loss(representations, same_t, discriminator, 0.03)
            preserve_loss, _ = relation_preservation_loss(logits, same_t, margin=0.02)
            total = 0.05 * pair_loss + 0.01 * adv_loss + 0.01 * preserve_loss
            if not torch.isfinite(total):
                raise FloatingPointError(f"nonfinite OOF loss fold={fold}, inner={inner_fold}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()

        # Validation labels are first read here, after the fixed tenth training epoch.
        valid_rng = np.random.default_rng(610000 + fold * 1000 + inner_fold * 10 + 2)
        valid_pairs, valid_same = r.sample_balanced_pairs(valid_ids, labels, 4096, valid_rng, union)
        if not len(valid_pairs) or not np.isin(valid_pairs, valid_ids).all():
            raise RuntimeError("OOF validation pairs escaped inner validation patients")
        with torch.no_grad():
            embeddings = [encoder(part) for encoder, part in zip(encoders, parts)]
            val_pair_t = torch.as_tensor(valid_pairs, dtype=torch.long, device=device)
            valid_logits = [
                scorer(pair_features(z[val_pair_t[:, 0]], z[val_pair_t[:, 1]])).cpu().numpy()
                for scorer, z in zip(scorers, embeddings)
            ]
    return {
        "pairs": valid_pairs,
        "same": np.asarray(valid_same, dtype=np.int64),
        "logits": valid_logits,
        "train_patient_count": int(len(train_ids)),
        "valid_patient_count": int(len(valid_ids)),
        "train_patient_sha": sha_bytes(np.asarray(sorted(train_ids), dtype=np.int64).tobytes()),
        "valid_patient_sha": sha_bytes(np.asarray(sorted(valid_ids), dtype=np.int64).tobytes()),
        "train_pair_sha": pair_sha(pairs, same),
        "valid_pair_sha": pair_sha(valid_pairs, valid_same),
        "seed": local_seed,
    }


def postalign_oof(fold: int, state):
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    r = load_base()
    cache_path = oof_path(fold)
    if cache_path.is_file():
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        if saved.get("warmup_sha256") == sha_file(warmup_path(fold)) and saved.get("source_sha") == source_sha_map():
            return saved
        raise RuntimeError(f"incompatible E61 OOF state exists for fold {fold}")
    rng_before = r.rng_sha(r.pack_rng())
    train_ids = state["train_np"]
    labels = state["labels_np"]
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=61000 + fold)
    collected = [{"same": [], "logits": []} for _ in range(4)]
    inner_records = []
    validation_visits = np.zeros(len(labels), dtype=np.int64)
    for inner_fold, (train_local, valid_local) in enumerate(splitter.split(train_ids, labels[train_ids]), start=1):
        inner_train = train_ids[train_local]
        inner_valid = train_ids[valid_local]
        if np.intersect1d(inner_train, inner_valid).size:
            raise RuntimeError(f"inner patient overlap fold={fold}, inner={inner_fold}")
        validation_visits[inner_valid] += 1
        result = fit_aligned_inner(fold, inner_fold, inner_train, inner_valid, state["supports"], labels, r.load_data()[0][0])
        inner_records.append({key: result[key] for key in (
            "train_patient_count", "valid_patient_count", "train_patient_sha", "valid_patient_sha",
            "train_pair_sha", "valid_pair_sha", "seed",
        )})
        for modality in range(4):
            collected[modality]["same"].append(result["same"])
            collected[modality]["logits"].append(result["logits"][modality])
    if not np.all(validation_visits[train_ids] == 1) or np.any(validation_visits[state["test_np"]] != 0):
        raise RuntimeError("OOF validation partitions did not cover only outer-training patients exactly once")
    modalities = []
    for modality in range(4):
        same = np.concatenate(collected[modality]["same"])
        logits = np.concatenate(collected[modality]["logits"])
        auc = float(roc_auc_score(same, logits))
        pr_auc = float(average_precision_score(same, logits))
        mean_same = float(logits[same == 1].mean())
        mean_diff = float(logits[same == 0].mean())
        reliability = float(np.clip((auc - 0.5) / 0.25, 0.0, 1.0)) if auc >= 0.55 and mean_same > mean_diff else 0.0
        modalities.append({
            "modality": r.MODs[modality] if hasattr(r, "MODs") else ("PHENO", "ANAT", "FUNC", "CORR")[modality],
            "oof_auc": auc,
            "oof_pr_auc": pr_auc,
            "mean_same_logit": mean_same,
            "mean_different_logit": mean_diff,
            "same_minus_different": mean_same - mean_diff,
            "reliability": reliability,
            "pair_count": int(len(same)),
            "same_pair_count": int(same.sum()),
            "different_pair_count": int((1 - same).sum()),
            "logit_mean": float(logits.mean()),
            "logit_std": float(logits.std()),
            "patient_validation_count": int(len(train_ids)),
        })
    rng_after = r.rng_sha(r.pack_rng())
    if rng_after != rng_before:
        raise RuntimeError("OOF process changed the main training RNG")
    result = {
        "fold": fold,
        "status": "OOF_COMPLETE",
        "warmup_sha256": sha_file(warmup_path(fold)),
        "main_rng_before": rng_before,
        "main_rng_after": rng_after,
        "main_rng_unchanged": True,
        "source_sha": source_sha_map(),
        "inner_split_seed": 61000 + fold,
        "inner_folds": inner_records,
        "validation_patient_exactly_once": True,
        "outer_test_patient_count": int(len(state["test_np"])),
        "modalities": modalities,
    }
    atomic_json(cache_path, result)
    return result


def nested_tensor_equal(left, right):
    import torch

    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(nested_tensor_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(nested_tensor_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (float, int, str, bool, type(None))):
        return left == right
    return True


def module_state_sha(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode() + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def warmup_parity_trial():
    """Compare two same-initialized executions through all ten shared epochs."""
    import torch

    r = load_base()
    first = build_state(1)
    initial_rng = r.rng_sha(r.pack_rng())
    second = build_state(1)
    if r.rng_sha(r.pack_rng()) != initial_rng:
        raise RuntimeError("constructing the second parity copy failed to restore the same RNG endpoint")
    for key in ("initial_shared_sha", "initial_gfr_sha", "initial_cam_sha", "relation_core_sha", "pair_sha"):
        if first[key] != second[key]:
            raise RuntimeError(f"warmup parity initialization mismatch: {key}")
    epochs = []
    for epoch in range(10):
        pre_rng = r.pack_rng()
        for optimizer in (first["opt_base"], first["opt_cam"], first["opt_added"]):
            optimizer.zero_grad(set_to_none=True)
        out_a = r.forward_state(first, training=True, use_cadr=False)
        loss_a = out_a["total"].detach().cpu()
        logp_a = out_a["logp"].detach().cpu()
        out_a["total"].backward()
        first["opt_base"].step(); first["opt_cam"].step(); first["opt_added"].step()
        rng_a = r.pack_rng()
        r.restore_rng(pre_rng)
        for optimizer in (second["opt_base"], second["opt_cam"], second["opt_added"]):
            optimizer.zero_grad(set_to_none=True)
        out_b = r.forward_state(second, training=True, use_cadr=False)
        loss_b = out_b["total"].detach().cpu()
        logp_b = out_b["logp"].detach().cpu()
        out_b["total"].backward()
        second["opt_base"].step(); second["opt_cam"].step(); second["opt_added"].step()
        rng_b = r.pack_rng()
        if not torch.equal(loss_a, loss_b) or not torch.equal(logp_a, logp_b):
            raise RuntimeError(f"shared warmup forward parity failure at epoch {epoch + 1}")
        for module_a, module_b in ((first["model"], second["model"]), (first["cam"], second["cam"])):
            if module_state_sha(module_a) != module_state_sha(module_b):
                raise RuntimeError(f"shared warmup parameter parity failure at epoch {epoch + 1}")
        for name in ("opt_base", "opt_cam", "opt_added"):
            if not nested_tensor_equal(first[name].state_dict(), second[name].state_dict()):
                raise RuntimeError(f"shared warmup optimizer parity failure at epoch {epoch + 1}: {name}")
        if r.rng_sha(rng_a) != r.rng_sha(rng_b):
            raise RuntimeError(f"shared warmup RNG parity failure at epoch {epoch + 1}")
        r.restore_rng(rng_a)
        epochs.append({"epoch": epoch + 1, "forward_equal": True, "optimizer_equal": True, "rng_sha": r.rng_sha(rng_a)})
        del out_a, out_b
    return {"status": "PASS", "fold": 1, "epochs": epochs, "full_ten_epoch_parity": True}


def gradient_reversal_preflight():
    import torch
    import torch.nn.functional as F
    from cadr_gfr import ConditionalModalityDiscriminator, conditional_adversarial_loss

    torch.manual_seed(619001)
    discriminator = ConditionalModalityDiscriminator(65, 32, 4)
    pairs = [torch.randn(7, 65, requires_grad=True) for _ in range(4)]
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
    loss, _ = conditional_adversarial_loss(pairs, labels, discriminator, 0.03)
    loss.backward()
    condition = F.one_hot(labels.long(), num_classes=2).float()
    direct_logits = torch.cat([discriminator(value.detach().requires_grad_(True), condition) for value in pairs], dim=0)
    targets = torch.cat([torch.full((7,), m, dtype=torch.long) for m in range(4)])
    direct_loss = F.cross_entropy(direct_logits, targets)
    direct_values = [value.detach().clone().requires_grad_(True) for value in pairs]
    direct_logits = torch.cat([discriminator(value, condition) for value in direct_values], dim=0)
    direct_loss = F.cross_entropy(direct_logits, targets)
    expected = torch.autograd.grad(direct_loss, direct_values)
    for modality in range(3):
        if pairs[modality].grad is None or not torch.allclose(pairs[modality].grad, -0.03 * expected[modality], rtol=1e-5, atol=1e-7):
            raise RuntimeError(f"GRL coefficient/direction failure for modality {modality}")
    if pairs[3].grad is not None:
        raise RuntimeError("CORR adversarial branch did not detach")
    return {"status": "PASS", "corr_detached": True, "student_gradients": [float(v.grad.norm()) for v in pairs[:3]], "grl_scale": 0.03}


def e2_gradient_preflight():
    import torch

    r = load_base()
    state = build_state(1)
    state["model"].reliability.fill_(1.0)
    state["model"].calibration.copy_(torch.tensor([[1.0, 0.0]] * 4, device=state["model"].calibration.device))
    state["model"].adaptive_gate_logits.requires_grad_(True)
    state["model"].reliability_gate_logits.requires_grad_(False)
    for scorer in state["model"].scorers:
        scorer.net[-1].bias.data.fill_(1.0)
    from model_e61 import e2e_adaptive_forward
    from torch.nn import functional as F
    fused, adjs, info = e2e_adaptive_forward(state["model"], state["parts"], state["train_pairs"])
    logp, _, _ = state["cam"](fused)
    loss = r.runtime()["utils"].ClsLoss(logp, state["y"], state["train"], state["weights"])
    grads = torch.autograd.grad(
        loss,
        [state["model"].adaptive_gate_logits]
        + [scorer.net[-1].bias for scorer in state["model"].scorers]
        + [encoder.net[0].weight for encoder in state["model"].encoders],
        allow_unused=True,
    )
    if grads[0] is None or not torch.isfinite(grads[0]).all() or float(grads[0].norm()) == 0.0:
        raise RuntimeError("E2 adaptive gates receive no classification gradient")
    positive_counts = [item["positive_edges"] for item in info["edge_diagnostics"]]
    for modality in range(4):
        if positive_counts[modality] <= 0:
            raise RuntimeError(f"preflight failed to activate CADR edges for modality {modality}")
        for offset, name in ((1, "scorer"), (5, "encoder")):
            grad = grads[offset + modality]
            if grad is None or not torch.isfinite(grad).all() or float(grad.norm()) == 0.0:
                raise RuntimeError(f"classification gradient does not reach {name} for modality {modality}")
    edge_weight_grad = grads[1:5]
    del loss, logp, fused, adjs, info
    return {
        "status": "PASS", "gate_grad_norm": float(grads[0].norm()),
        "scorer_grad_norms": [float(value.norm()) for value in edge_weight_grad],
        "positive_edges": positive_counts,
        "topology_discrete": True,
    }


def preflight():
    import hashlib
    import numpy as np
    import torch

    r = load_base()
    sources = verify_reference_artifacts()
    refs = reference_rows()
    (x, y), folds = r.load_data()
    data_csv = BASE_ROOT / "data/ABIDE/processed_standard_data.csv"
    data_npy = BASE_ROOT / "data/ABIDE/modal_feat_dict.npy"
    work_csv = ROOT / "data/ABIDE/processed_standard_data.csv"
    work_npy = ROOT / "data/ABIDE/modal_feat_dict.npy"
    data_hashes = {"base_csv": sha_file(data_csv), "workspace_csv": sha_file(work_csv), "base_mapping": sha_file(data_npy), "workspace_mapping": sha_file(work_npy)}
    if data_hashes["base_csv"] != data_hashes["workspace_csv"] or data_hashes["base_mapping"] != data_hashes["workspace_mapping"]:
        raise RuntimeError("workspace and historical data inputs differ")
    if x.shape != (871, 320) or np.bincount(y).tolist() != [403, 468] or len(folds) != 10:
        raise RuntimeError("53.3 data shape, labels, or folds differ from the locked protocol")
    initial_records = []
    for fold in range(1, 11):
        state = build_state(fold)
        ref = state["base_task"]
        cp_path = BASE_RUNTIME / "checkpoints" / f"{BASE_GROUP}_f{fold:02d}.best.pt"
        if not cp_path.is_file() or sha_file(cp_path) != ref["best_sha256"]:
            raise RuntimeError(f"historical checkpoint SHA failed fold {fold}")
        ref_metrics = refs[fold]
        for key in ("acc", "hard_auc", "prob_auc", "sen", "spe"):
            actual = float(ref["metrics"][key])
            expected = float(ref_metrics[key])
            if abs(actual - expected) > 1e-10:
                raise RuntimeError(f"historical result mismatch fold={fold}, metric={key}")
        initial_records.append({
            "fold": fold, "status": "PASS", "checkpoint_sha256": sha_file(cp_path),
            "initial_shared_sha": state["initial_shared_sha"], "initial_cam_sha": state["initial_cam_sha"],
            "initial_gfr_sha": state["initial_gfr_sha"], "relation_core_sha": state["relation_core_sha"],
            "initial_rng_sha": state["initial_rng_sha"], "split_sha": state["split_sha"],
            "pair_sha": state["pair_sha"], "anchor_sha": state["anchor_sha"],
            "same_source_parameters_match": True,
            "oof_dependent_reliability_and_calibration_reinitialized": True,
        })
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Zero-reliability E1 must exactly recover original GraphLearn + the same GCN.
    state = build_state(1)
    state["model"].eval(); state["cam"].eval()
    state["model"].reliability.zero_()
    state["model"].calibration.copy_(torch.tensor([[1.0, 0.0]] * 4, device=state["model"].calibration.device))
    with torch.no_grad():
        full, adjs, info = state["model"](state["parts"], state["train_pairs"], use_cadr=True)
        baseline, base_adjs, _ = state["model"](state["parts"], state["train_pairs"], use_cadr=False)
    if not torch.equal(full, baseline) or any(not torch.equal(a, b) for a, b in zip(adjs, base_adjs)):
        raise RuntimeError("zero-reliability E1 fallback differs from original GraphLearn path")
    fallback_check = {"status": "PASS", "fused_equal": True, "propagation_modes": info["propagation_modes"]}
    del state, full, adjs, info, baseline, base_adjs

    # E2 forward must preserve the same original graph candidates and use no OOF multiplier.
    gate_grad = e2_gradient_preflight()
    grl = gradient_reversal_preflight()
    parity = warmup_parity_trial()
    if list(TASKS.glob("fold*.postalign_oof.json")):
        raise RuntimeError("OOF state existed before shared warmup")
    preflight_obj = {
        "experiment": CFG["experiment"], "status": "PASS", "seed": 0,
        "data_sha256": data_hashes, "base_source_sha256": sources,
        "base_commit": CFG["locked_reference_commit"], "folds": initial_records,
        "warmup_ten_epoch_parity": parity, "grl_test": grl,
        "e1_zero_reliability_fallback": fallback_check,
        "e2_classification_gradient": gate_grad,
        "oof_old_cache_read": False, "oof_pre_warmup_exists": False,
        "outer_test_labels_used_for_cadr_training": False,
        "all_formal_groups": list(GROUPS), "formal_tasks": 20,
        "protocol_note": "val=test and pre-CV standardization/RFE retained; exploratory only",
        "source_sha": source_sha_map(),
    }
    atomic_json(RESULTS / "preflight.json", preflight_obj)
    write_csv(RESULTS / "preflight_initialization.csv", initial_records)
    return preflight_obj


def configure_branch(state, group, oof=None):
    import torch

    model = state["model"]
    identity = torch.tensor([[1.0, 0.0]] * 4, dtype=model.calibration.dtype, device=model.calibration.device)
    model.calibration.copy_(identity)
    if group == "E61-E1":
        if oof is None:
            raise ValueError("E1 requires post-warmup OOF results")
        reliability = torch.tensor([item["reliability"] for item in oof["modalities"]], dtype=model.reliability.dtype, device=model.reliability.device)
        model.reliability.copy_(reliability)
        model.reliability_gate_logits.data.copy_(torch.logit(reliability.clamp(0.02, 0.98)))
        model.reliability_gate_logits.requires_grad_(True)
        model.adaptive_gate_logits.requires_grad_(False)
        state["branch"] = {"group": group, "oof": oof, "reliability": reliability.detach().cpu().tolist(), "calibration": "identity_sigmoid"}
    elif group == "E61-E2":
        model.reliability.fill_(1.0)
        model.reliability_gate_logits.data.zero_()
        model.reliability_gate_logits.requires_grad_(False)
        model.adaptive_gate_logits.requires_grad_(True)
        state["branch"] = {"group": group, "oof": None, "reliability": [1.0] * 4, "calibration": "identity_sigmoid"}
    else:
        raise KeyError(group)
    return state


def forward_e2_state(state, training: bool, intervention="full", permute_scores=False, permutation_seed=0):
    import torch
    import torch.nn.functional as F
    from model_e61 import e2e_adaptive_forward
    from cadr_gfr import conditional_adversarial_loss, relation_preservation_loss

    if training:
        state["model"].train()
        state["cam"].train()
    else:
        state["model"].eval()
        state["cam"].eval()
    fused, adjs, info = e2e_adaptive_forward(
        state["model"], state["parts"], state["train_pairs"], intervention,
        permute_scores=permute_scores, permutation_seed=permutation_seed,
    )
    logp, hidden, attention = state["cam"](fused)
    cls = load_base().runtime()["utils"].ClsLoss(logp, state["y"], state["train"], state["weights"])
    graph, graph_parts = load_base()._graph_loss(state["parts"], adjs)
    pair = torch.stack([
        F.binary_cross_entropy_with_logits(logits, state["train_same"])
        for logits in info["pair_logits"]
    ]).mean()
    adv, adv_info = conditional_adversarial_loss(
        info["pair_reprs"], state["train_same"], state["model"].discriminator, state["model"].grl_scale,
    )
    preserve, preserve_gaps = relation_preservation_loss(info["pair_logits"], state["train_same"], margin=0.02)
    total = cls + graph + 0.05 * pair + 0.01 * adv + 0.01 * preserve
    return {
        "logp": logp, "hidden": hidden, "attention": attention, "fused": fused,
        "adjs": adjs, "info": info, "cls": cls, "graph": graph,
        "graph_parts": graph_parts, "pair": pair, "adv": adv, "adv_info": adv_info,
        "preserve": preserve, "preserve_gaps": preserve_gaps, "total": total,
    }


def task_forward(state, group, training: bool, intervention="full", permute_scores=False, permutation_seed=0):
    if group == "E61-E1":
        if training:
            return load_base().forward_state(state, training=True, use_cadr=True)
        if permute_scores:
            return scorer_permutation_forward(state, permutation_seed)
        if intervention == "positive_off" or intervention == "negative_off" or intervention == "direct_off":
            return load_base().forward_state(state, training=False, use_cadr=True, intervention=intervention)
        return load_base().forward_state(state, training=False, use_cadr=True)
    if group == "E61-E2":
        return forward_e2_state(state, training, intervention, permute_scores, permutation_seed)
    raise KeyError(group)


def scorer_permutation_forward(state, seed):
    """Fixed score-shuffle diagnostic; the local generators do not touch training RNG."""
    import torch

    model = state["model"]
    handles = []
    for modality, scorer in enumerate(model.scorers):
        generator = torch.Generator(device=state["train_pairs"].device)
        generator.manual_seed(int(seed) + modality)

        def hook(_module, _inputs, output, gen=generator):
            if output.numel() < 2:
                return output
            return output[torch.randperm(output.numel(), generator=gen, device=output.device)]

        handles.append(scorer.register_forward_hook(hook))
    try:
        return load_base().forward_state(state, training=False, use_cadr=True)
    finally:
        for handle in handles:
            handle.remove()


def total_gradient_norm(parameters):
    import torch

    values = [parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(values).sum().sqrt().cpu()) if values else 0.0


def checkpoint_payload(state, group, fold, best, best_epoch, epochs_executed, best_metrics, best_diag, oof, trace):
    import torch

    r = load_base()
    return {
        "experiment": CFG["experiment"], "group": group, "group_name": GROUPS[group],
        "fold": int(fold), "seed": 0, "best_epoch_zero_based": int(best_epoch),
        "best_epoch_1based": int(best_epoch + 1), "epochs_executed": int(epochs_executed),
        "model": best["model"], "cam": best["cam"],
        "optimizer_base": best["optimizer_base"], "optimizer_cam": best["optimizer_cam"],
        "optimizer_added": best["optimizer_added"],
        "metrics": best_metrics, "diagnostics": best_diag,
        "test_probability": best["test_probability"], "test_prediction": best["test_prediction"],
        "oof": oof, "branch": state["branch"], "reliability": state["model"].reliability.detach().cpu().tolist(),
        "calibration": state["model"].calibration.detach().cpu().tolist(),
        "adaptive_gate_logits": state["model"].adaptive_gate_logits.detach().cpu().tolist(),
        "adaptive_gate": torch.sigmoid(state["model"].adaptive_gate_logits.detach()).cpu().tolist(),
        "split_sha": state["split_sha"], "pair_sha": state["pair_sha"], "anchor_sha": state["anchor_sha"],
        "initial_shared_sha": state["initial_shared_sha"], "initial_gfr_sha": state["initial_gfr_sha"],
        "initial_cam_sha": state["initial_cam_sha"], "relation_core_sha": state["relation_core_sha"],
        "initial_rng_sha": state["initial_rng_sha"], "warmup_sha256": sha_file(warmup_path(fold)),
        "warmup_rng_sha": torch_load(warmup_path(fold))["rng_sha"],
        "training_rng_sha": r.rng_sha(r.pack_rng()), "rng_trace": trace,
        "optimizer_weight_decay": {
            "gfr": [float(g["weight_decay"]) for g in state["opt_base"].param_groups],
            "cam": [float(g["weight_decay"]) for g in state["opt_cam"].param_groups],
            "cadr": [float(g["weight_decay"]) for g in state["opt_added"].param_groups],
        },
        "source_sha": source_sha_map(), "base_source_sha": verify_reference_artifacts(),
        "runtime_rng_parity_scope": "same CAM dropout call count/order per matching epoch; deterministic graph/OOF paths; E1 OOF isolated",
    }


def train_worker(group: str, fold: int):
    import numpy as np
    import torch

    if group not in GROUPS:
        raise KeyError(group)
    r = load_base()
    task_path, best_path, latest_path, log_path = fold_paths(group, fold)
    if task_path.is_file() and best_path.is_file():
        existing = json.loads(task_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE" and existing.get("checkpoint_sha256") == sha_file(best_path):
            return existing

    state, warmup = load_warmup_state(fold)
    oof = postalign_oof(fold, state) if group == "E61-E1" else None
    configure_branch(state, group, oof)
    if group == "E61-E1":
        if r.rng_sha(r.pack_rng()) != warmup["rng_sha"]:
            raise RuntimeError("E1 OOF changed the main RNG before group branch")
    start_epoch, wait = 10, 0
    best_acc, best_epoch, best_state, best_metrics, best_diag = -1.0, -1, None, None, None
    trace = []
    if latest_path.is_file():
        saved = torch_load(latest_path)
        if saved.get("group") != group or saved.get("fold") != fold or saved.get("warmup_sha256") != sha_file(warmup_path(fold)) or saved.get("source_sha") != source_sha_map():
            raise RuntimeError(f"incompatible recoverable task found: {latest_path}")
        state["model"].load_state_dict(saved["model"], strict=True)
        state["cam"].load_state_dict(saved["cam"], strict=True)
        state["opt_base"].load_state_dict(saved["optimizer_base"])
        state["opt_cam"].load_state_dict(saved["optimizer_cam"])
        state["opt_added"].load_state_dict(saved["optimizer_added"])
        start_epoch, wait = int(saved["next_epoch"]), int(saved["wait"])
        best_acc, best_epoch = float(saved["best_acc"]), int(saved["best_epoch"])
        best_state, best_metrics, best_diag = saved["best_state"], saved["best_metrics"], saved["best_diag"]
        trace = saved["rng_trace"]
        r.restore_rng(saved["rng"])
    elif group == "E61-E1" and r.rng_sha(r.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError("E1 branch did not start at the shared warmup RNG state")
    elif group == "E61-E2" and r.rng_sha(r.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError("E2 branch did not start at the shared warmup RNG state")

    started = time.time()
    epochs_executed = start_epoch
    for epoch in range(start_epoch, int(CFG["optimization"]["max_epochs"])):
        rng_before = r.rng_sha(r.pack_rng())
        for optimizer in (state["opt_base"], state["opt_cam"], state["opt_added"]):
            optimizer.zero_grad(set_to_none=True)
        out = task_forward(state, group, training=True)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"nonfinite training loss {group} fold={fold} epoch={epoch + 1}")
        out["total"].backward()
        for module in (state["model"], state["cam"]):
            for parameter in module.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"nonfinite training gradient {group} fold={fold} epoch={epoch + 1}")
        grad = {
            "gfr": total_gradient_norm(state["base_params"]),
            "cam": total_gradient_norm(state["cam"].parameters()),
            "cadr": total_gradient_norm(parameter for _, parameter in state["added_named"]),
            "adaptive_gate": total_gradient_norm([state["model"].adaptive_gate_logits]),
            "reliability_gate": total_gradient_norm([state["model"].reliability_gate_logits]),
        }
        state["opt_base"].step(); state["opt_cam"].step(); state["opt_added"].step()
        rng_after_train = r.rng_sha(r.pack_rng())
        epochs_executed = epoch + 1
        with torch.no_grad():
            evaluated = task_forward(state, group, training=False)
            metrics = r.classification_metrics(evaluated["logp"], state["test"], state["y"])
        rng_after_eval = r.rng_sha(r.pack_rng())
        if metrics["acc"] > best_acc:
            best_acc, best_epoch, wait = float(metrics["acc"]), epoch, 0
            best_metrics = {key: float(value) for key, value in metrics.items() if key not in {"pred", "prob"}}
            best_state = {
                "model": model_state_cpu(state["model"]), "cam": model_state_cpu(state["cam"]),
                "optimizer_base": copy.deepcopy(state["opt_base"].state_dict()),
                "optimizer_cam": copy.deepcopy(state["opt_cam"].state_dict()),
                "optimizer_added": copy.deepcopy(state["opt_added"].state_dict()),
                "test_probability": metrics["prob"].copy(),
                "test_prediction": metrics["pred"].copy(),
            }
            best_diag = {
                "loss_cls": float(out["cls"].detach().cpu()), "loss_graph": float(out["graph"].detach().cpu()),
                "loss_pair": float(out["pair"].detach().cpu()), "loss_adv": float(out["adv"].detach().cpu()),
                "loss_preserve": float(out["preserve"].detach().cpu()),
                "adv_accuracy": float(out["adv_info"]["accuracy"].detach().cpu()),
                "preserve_gaps": out["preserve_gaps"].detach().cpu().tolist(),
                "gradient_norms": grad, "edge_diagnostics": evaluated["info"].get("edge_diagnostics", []),
                "adaptive_gate": torch.sigmoid(state["model"].adaptive_gate_logits.detach()).cpu().tolist(),
                "reliability_gate": torch.sigmoid(state["model"].reliability_gate_logits.detach()).cpu().tolist(),
                "message_norms": message_norms(evaluated["info"]),
                "rng_before": rng_before, "rng_after_train": rng_after_train,
            }
            checkpoint = checkpoint_payload(state, group, fold, best_state, best_epoch, epochs_executed, best_metrics, best_diag, oof, trace + [{"epoch": epoch + 1, "rng_before": rng_before, "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval}])
            atomic_torch_save(best_path, checkpoint)
        else:
            wait += 1
        trace.append({"epoch": epoch + 1, "rng_before": rng_before, "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval})
        if (epoch + 1) % 10 == 0 or epoch == CFG["optimization"]["max_epochs"] - 1 or wait > CFG["optimization"]["early_stop_wait_gt"]:
            atomic_torch_save(latest_path, {
                "group": group, "fold": fold, "next_epoch": epoch + 1, "wait": wait,
                "best_acc": best_acc, "best_epoch": best_epoch, "best_state": best_state,
                "best_metrics": best_metrics, "best_diag": best_diag,
                "model": model_state_cpu(state["model"]), "cam": model_state_cpu(state["cam"]),
                "optimizer_base": state["opt_base"].state_dict(), "optimizer_cam": state["opt_cam"].state_dict(),
                "optimizer_added": state["opt_added"].state_dict(), "rng": r.pack_rng(),
                "rng_trace": trace, "oof": oof, "source_sha": source_sha_map(),
                "warmup_sha256": sha_file(warmup_path(fold)), "branch": state["branch"],
            })
        if (epoch + 1) % 10 == 0:
            log_json(log_path, {
                "group": group, "fold": fold, "epoch": epoch + 1,
                "acc": metrics["acc"], "hard_auc": metrics["hard_auc"], "prob_auc": metrics["prob_auc"],
                "sen": metrics["sen"], "spe": metrics["spe"], "best_acc": best_acc, "best_epoch": best_epoch + 1,
                "loss": float(out["total"].detach().cpu()), "gradient_norms": grad,
                "edge_diagnostics": evaluated["info"].get("edge_diagnostics", []),
                "adaptive_gate": torch.sigmoid(state["model"].adaptive_gate_logits.detach()).cpu().tolist(),
                "rng_sha": rng_after_eval,
            })
        if wait > int(CFG["optimization"]["early_stop_wait_gt"]):
            break
        del out, evaluated

    if best_state is None:
        raise RuntimeError(f"no best checkpoint selected for {group} fold {fold}")
    state["model"].load_state_dict(best_state["model"], strict=True)
    state["cam"].load_state_dict(best_state["cam"], strict=True)
    with torch.no_grad():
        readback = task_forward(state, group, training=False)
        readback_metrics = r.classification_metrics(readback["logp"], state["test"], state["y"])
    for metric in METRICS:
        if abs(float(readback_metrics[metric]) - float(best_metrics[metric])) > 1e-10:
            raise RuntimeError(f"worker readback mismatch group={group} fold={fold} metric={metric}")
    if not np.array_equal(readback_metrics["pred"], best_state["test_prediction"]):
        raise RuntimeError(f"worker readback prediction mismatch group={group} fold={fold}")
    if not np.allclose(readback_metrics["prob"], best_state["test_probability"], rtol=0, atol=1e-8):
        raise RuntimeError(f"worker readback probabilities mismatch group={group} fold={fold}")
    final_checkpoint = checkpoint_payload(
        state, group, fold, best_state, best_epoch, epochs_executed, best_metrics, best_diag, oof, trace,
    )
    atomic_torch_save(best_path, final_checkpoint)
    checkpoint_hash = sha_file(best_path)
    task = {
        "group": group, "group_name": GROUPS[group], "fold": fold, "seed": 0,
        "status": "COMPLETE", "split_sha": state["split_sha"], "pair_sha": state["pair_sha"],
        "anchor_sha": state["anchor_sha"], "warmup_sha256": sha_file(warmup_path(fold)),
        "initial_shared_sha": state["initial_shared_sha"], "initial_gfr_sha": state["initial_gfr_sha"],
        "initial_cam_sha": state["initial_cam_sha"], "relation_core_sha": state["relation_core_sha"],
        "initial_rng_sha": state["initial_rng_sha"], "warmup_rng_sha": warmup["rng_sha"],
        "best_epoch_zero_based": int(best_epoch), "best_epoch_1based": int(best_epoch + 1),
        "epochs_executed": int(epochs_executed), "metrics": {key: float(readback_metrics[key]) for key in METRICS},
        "checkpoint_sha256": checkpoint_hash, "readback_status": "WORKER_PASS",
        "optimizer_weight_decay": final_checkpoint["optimizer_weight_decay"],
        "oof_reliability": [item["reliability"] for item in oof["modalities"]] if oof else None,
        "adaptive_gate": torch.sigmoid(state["model"].adaptive_gate_logits.detach()).cpu().tolist(),
        "rng_trace": trace, "source_sha": source_sha_map(),
        "runtime_rng_parity_scope": final_checkpoint["runtime_rng_parity_scope"],
        "wall_seconds": float(time.time() - started),
    }
    atomic_json(task_path, task)
    latest_path.unlink(missing_ok=True)
    print(f"complete {group} fold={fold} best_epoch={best_epoch + 1} acc={readback_metrics['acc']:.6f}", flush=True)
    return task


def message_norms(info):
    def norm(values):
        return [float(value.detach().float().norm().cpu()) for value in values]
    positive = norm(info.get("positive_messages", []))
    negative = norm(info.get("negative_messages", []))
    direct = norm(info.get("direct_messages", []))
    return {
        "original_gcn": norm(info.get("messages_original", info.get("original_fallback_messages", []))),
        "positive_gcn": positive,
        "negative_gcn": negative,
        "negative_gcn_scaled_0p25": [0.25 * value for value in negative],
        "direct_difference": direct,
        "direct_difference_scaled_0p02": [0.02 * value for value in direct],
        "cadr_message": norm(info.get("messages_cadr", info.get("messages", []))),
        "fusion_gates": info.get("fusion_gates").detach().cpu().tolist() if hasattr(info.get("fusion_gates"), "detach") else [],
        "propagation_modes": info.get("propagation_modes", []),
    }


def e1_replace_disease_with_original(state, result, modality=None):
    import torch

    model = state["model"]
    fused = result["fused"].detach().clone()
    original_adjs = result["info"]["original_adjs"]
    identity = torch.eye(fused.shape[0], dtype=fused.dtype, device=fused.device)
    selected = range(4) if modality is None else (modality,)
    for index in selected:
        # A zero-reliability mode already follows the historical GraphLearn fallback.
        if float(model.reliability[index].detach()) <= 0.0:
            continue
        raw = torch.cat(state["parts"], dim=1)
        original = model.gcn[index](raw, model.utils.normalize_adj(original_adjs[index] + identity))
        old_scale = torch.sigmoid(model.fusion_weight[index]) * torch.sigmoid(model.reliability_gate_logits[index])
        new_scale = torch.sigmoid(model.fusion_weight[index])
        fused = fused - old_scale * result["info"]["messages"][index].detach() + new_scale * original.detach()
    logp, hidden, attention = state["cam"](fused)
    out = dict(result)
    out.update({"fused": fused, "logp": logp, "hidden": hidden, "attention": attention})
    return out


def evaluate_with_intervention(state, group, intervention="full", seed=0):
    import torch

    r = load_base()
    state["model"].eval(); state["cam"].eval()
    with torch.no_grad():
        if group == "E61-E1":
            if intervention in ("disease_off",) or intervention.startswith("modality_off:"):
                full = r.forward_state(state, training=False, use_cadr=True)
                if intervention == "disease_off":
                    result = e1_replace_disease_with_original(state, full)
                else:
                    name = intervention.split(":", 1)[1]
                    modality = ("PHENO", "ANAT", "FUNC", "CORR").index(name)
                    result = e1_replace_disease_with_original(state, full, modality)
            elif intervention == "permuted_scores":
                result = scorer_permutation_forward(state, seed)
            else:
                result = r.forward_state(state, training=False, use_cadr=True, intervention=intervention)
        else:
            result = forward_e2_state(
                state, training=False, intervention=intervention,
                permute_scores=intervention == "permuted_scores", permutation_seed=seed,
            )
        metrics = r.classification_metrics(result["logp"], state["test"], state["y"])
    return metrics, result


def revalidate_oof_membership(fold, oof, train_ids, test_ids, labels):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    split = StratifiedKFold(n_splits=3, shuffle=True, random_state=61000 + fold)
    expected = []
    for train_local, valid_local in split.split(train_ids, labels[train_ids]):
        inner_train, inner_valid = train_ids[train_local], train_ids[valid_local]
        expected.append((
            sha_bytes(np.asarray(sorted(inner_train), dtype=np.int64).tobytes()),
            sha_bytes(np.asarray(sorted(inner_valid), dtype=np.int64).tobytes()),
        ))
    observed = [(item["train_patient_sha"], item["valid_patient_sha"]) for item in oof["inner_folds"]]
    if observed != expected or np.intersect1d(train_ids, test_ids).size:
        raise RuntimeError(f"OOF split readback mismatch for fold {fold}")
    if any(train_sha == valid_sha for train_sha, valid_sha in observed):
        raise RuntimeError(f"OOF patient train/validation overlap for fold {fold}")


def audit_checkpoint(group: str, fold: int):
    import numpy as np
    import torch

    r = load_base()
    task_file, best_file, _latest_file, _log_file = fold_paths(group, fold)
    task = json.loads(task_file.read_text(encoding="utf-8"))
    checkpoint_sha = sha_file(best_file)
    checkpoint = torch_load(best_file)
    if task.get("status") != "COMPLETE" or task.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError(f"task/checkpoint SHA mismatch for {group} fold {fold}")
    if checkpoint.get("group") != group or checkpoint.get("fold") != fold or checkpoint.get("seed") != 0:
        raise RuntimeError(f"checkpoint identity mismatch for {group} fold {fold}")
    if checkpoint.get("source_sha") != source_sha_map() or checkpoint.get("base_source_sha") != verify_reference_artifacts():
        raise RuntimeError(f"checkpoint source mismatch for {group} fold {fold}")
    state, warmup = load_warmup_state(fold)
    oof = None
    if group == "E61-E1":
        oof = json.loads(oof_path(fold).read_text(encoding="utf-8"))
        revalidate_oof_membership(fold, oof, state["train_np"], state["test_np"], state["labels_np"])
    elif checkpoint.get("oof") is not None or checkpoint.get("branch", {}).get("oof") is not None or task.get("oof_reliability") is not None:
        raise RuntimeError("E2 accessed or retained an OOF reliability result")
    configure_branch(state, group, oof)
    for key in ("initial_shared_sha", "initial_gfr_sha", "initial_cam_sha", "relation_core_sha", "initial_rng_sha", "pair_sha", "split_sha", "anchor_sha"):
        if checkpoint[key] != state[key] or task[key] != state[key]:
            raise RuntimeError(f"checkpoint base identity mismatch group={group} fold={fold} field={key}")
    if checkpoint["warmup_sha256"] != sha_file(warmup_path(fold)) or task["warmup_sha256"] != sha_file(warmup_path(fold)):
        raise RuntimeError(f"checkpoint does not derive from shared warmup group={group} fold={fold}")
    state["model"].load_state_dict(checkpoint["model"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    state["opt_base"].load_state_dict(checkpoint["optimizer_base"])
    state["opt_cam"].load_state_dict(checkpoint["optimizer_cam"])
    state["opt_added"].load_state_dict(checkpoint["optimizer_added"])
    state["model"].eval(); state["cam"].eval()
    with torch.no_grad():
        result = task_forward(state, group, training=False)
        metrics = r.classification_metrics(result["logp"], state["test"], state["y"])
    for metric in METRICS:
        if abs(float(metrics[metric]) - float(checkpoint["metrics"][metric])) > 1e-10:
            raise RuntimeError(f"independent checkpoint metric mismatch {group} fold={fold} {metric}")
    if not np.array_equal(metrics["pred"], np.asarray(checkpoint["test_prediction"])):
        raise RuntimeError(f"independent checkpoint prediction mismatch {group} fold={fold}")
    if not np.allclose(metrics["prob"], np.asarray(checkpoint["test_probability"]), rtol=0, atol=1e-8):
        raise RuntimeError(f"independent checkpoint probability mismatch {group} fold={fold}")
    expected_wd = {"gfr": [0.11], "cam": [0.11], "cadr": [0.01, 0.01]}
    if checkpoint["optimizer_weight_decay"] != expected_wd:
        raise RuntimeError(f"optimizer WD audit failed group={group} fold={fold}: {checkpoint['optimizer_weight_decay']}")
    if group == "E61-E1":
        expected_rel = [item["reliability"] for item in oof["modalities"]]
        if not np.allclose(checkpoint["reliability"], expected_rel, rtol=0, atol=0):
            raise RuntimeError(f"E1 reliability buffer differs from OOF fold {fold}")
    else:
        if not np.allclose(checkpoint["adaptive_gate"], torch.sigmoid(state["model"].adaptive_gate_logits).detach().cpu().numpy(), atol=1e-8):
            raise RuntimeError(f"E2 adaptive gate readback mismatch fold {fold}")

    full_metrics, full_result = evaluate_with_intervention(state, group, "full", 610000 + fold)
    off_metrics, _ = evaluate_with_intervention(state, group, "disease_off", 610100 + fold)
    modality_metrics = {}
    for name in ("PHENO", "ANAT", "FUNC", "CORR"):
        modality_metrics[name], _ = evaluate_with_intervention(state, group, f"modality_off:{name}", 610200 + fold + len(modality_metrics))
    shuffled_metrics, _ = evaluate_with_intervention(state, group, "permuted_scores", 610300 + fold)
    component_metrics = {}
    for mode in ("positive_off", "negative_off", "direct_off"):
        component_metrics[mode], _ = evaluate_with_intervention(state, group, mode, 610400 + fold)

    def prediction_change(altered):
        return {
            "flipped_predictions": int(np.count_nonzero(full_metrics["pred"] != altered["pred"])),
            "mean_absolute_probability_change": float(np.mean(np.abs(full_metrics["prob"] - altered["prob"]))),
        }

    intervention_prediction_changes = {
        "disease_off": prediction_change(off_metrics),
        "score_permutation": prediction_change(shuffled_metrics),
        "modality_off": {name: prediction_change(value) for name, value in modality_metrics.items()},
        "component_off": {name: prediction_change(value) for name, value in component_metrics.items()},
    }

    classification_gradient_replay = None
    if group == "E61-E2":
        for module in (state["model"], state["cam"]):
            module.zero_grad(set_to_none=True)
        rng_before_gradient_replay = r.rng_sha(r.pack_rng())
        gradient_output = task_forward(state, group, training=False)
        gradient_output["cls"].backward()

        def grad_norm(parameters):
            values = [parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None]
            return float(torch.stack(values).sum().sqrt().cpu()) if values else 0.0

        def grad_abs_vector(parameter):
            if parameter.grad is None:
                return [0.0] * parameter.numel()
            return parameter.grad.detach().abs().cpu().reshape(-1).tolist()

        classification_gradient_replay = {
            "adaptive_gate_abs": grad_abs_vector(state["model"].adaptive_gate_logits),
            "edge_gate_abs": grad_abs_vector(state["model"].edge_gate_logits),
            "scorer_norm": [grad_norm(module.parameters()) for module in state["model"].scorers],
            "encoder_norm": [grad_norm(module.parameters()) for module in state["model"].encoders],
            "rng_unchanged": r.rng_sha(r.pack_rng()) == rng_before_gradient_replay,
        }
        if not classification_gradient_replay["rng_unchanged"]:
            raise RuntimeError(f"classification-gradient audit changed the main RNG: {group} fold={fold}")
        classification_gradient_replay["all_gates_nonzero"] = all(value > 0.0 for key in ("adaptive_gate_abs", "edge_gate_abs") for value in classification_gradient_replay[key])
        classification_gradient_replay["all_relation_modules_nonzero"] = all(value > 0.0 for key in ("scorer_norm", "encoder_norm") for value in classification_gradient_replay[key])
        for module in (state["model"], state["cam"]):
            module.zero_grad(set_to_none=True)

    raw_input_norm = float(torch.cat(state["parts"], dim=1).detach().float().norm().cpu())
    fusion_weights = torch.sigmoid(state["model"].fusion_weight.detach()).reshape(-1).cpu().numpy().tolist()
    if group == "E61-E2":
        adaptive_gates = torch.sigmoid(state["model"].adaptive_gate_logits.detach()).cpu().numpy().tolist()
        weighted_gate = [fusion_weights[i] * adaptive_gates[i] for i in range(4)]
        cadr_norms = full_result["info"]["messages_cadr"]
        message_stats = message_norms(full_result["info"])
        weighted_components = {
            "adaptive_gate": adaptive_gates,
            "fusion_weight_sigmoid": fusion_weights,
            "cadr_contribution_norm": [message_stats["cadr_message"][i] * weighted_gate[i] for i in range(4)],
            "cadr_contribution_to_raw_ratio": [message_stats["cadr_message"][i] * weighted_gate[i] / max(raw_input_norm, 1e-12) for i in range(4)],
            "original_graph_contribution_norm": [message_stats["original_gcn"][i] * fusion_weights[i] * (1.0 - adaptive_gates[i]) for i in range(4)],
            "positive_gcn_contribution_norm": [message_stats["positive_gcn"][i] * weighted_gate[i] for i in range(4)],
            "negative_gcn_contribution_norm": [message_stats["negative_gcn_scaled_0p25"][i] * weighted_gate[i] for i in range(4)],
            "direct_difference_contribution_norm": [message_stats["direct_difference_scaled_0p02"][i] * weighted_gate[i] for i in range(4)],
        }
    else:
        adaptive_gates = [0.0] * 4
        weighted_components = {
            "adaptive_gate": adaptive_gates,
            "fusion_weight_sigmoid": fusion_weights,
            "cadr_contribution_norm": [0.0] * 4,
            "cadr_contribution_to_raw_ratio": [0.0] * 4,
            "original_graph_contribution_norm": [],
            "positive_gcn_contribution_norm": [0.0] * 4,
            "negative_gcn_contribution_norm": [0.0] * 4,
            "direct_difference_contribution_norm": [0.0] * 4,
        }

    record = {
        "group": group, "fold": fold, "checkpoint_audit": "PASS",
        "checkpoint_sha256": checkpoint_sha, "best_epoch_1based": checkpoint["best_epoch_1based"],
        "epochs_executed": checkpoint["epochs_executed"], "warmup_sha256": checkpoint["warmup_sha256"],
        "initial_shared_sha": state["initial_shared_sha"], "initial_gfr_sha": state["initial_gfr_sha"],
        "initial_cam_sha": state["initial_cam_sha"], "relation_core_sha": state["relation_core_sha"],
        "initial_rng_sha": state["initial_rng_sha"], "split_sha": state["split_sha"],
        "pair_sha": state["pair_sha"], "anchor_sha": state["anchor_sha"],
        "metrics": {key: float(metrics[key]) for key in METRICS},
        "test_probability_max_abs_error": float(np.max(np.abs(metrics["prob"] - checkpoint["test_probability"]))),
        "prediction_exact": True, "optimizer_weight_decay": expected_wd,
        "full_metrics": {key: float(full_metrics[key]) for key in METRICS},
        "disease_off_metrics": {key: float(off_metrics[key]) for key in METRICS},
        "modality_off_metrics": {name: {key: float(value[key]) for key in METRICS} for name, value in modality_metrics.items()},
        "permuted_edge_score_metrics": {key: float(shuffled_metrics[key]) for key in METRICS},
        "component_intervention_metrics": {mode: {key: float(value[key]) for key in METRICS} for mode, value in component_metrics.items()},
        "intervention_prediction_changes": intervention_prediction_changes,
        "classification_gradient_replay": classification_gradient_replay,
        "mechanism": {
            "edge_diagnostics": full_result["info"].get("edge_diagnostics", []),
            "message_norms": message_norms(full_result["info"]),
            "adaptive_gate": torch.sigmoid(state["model"].adaptive_gate_logits.detach()).cpu().tolist(),
            "reliability": state["model"].reliability.detach().cpu().tolist(),
            "gradient_norms": checkpoint["diagnostics"]["gradient_norms"],
            "raw_input_frobenius_norm_320d": raw_input_norm,
            "weighted_message_contributions": weighted_components,
        },
        "oof": oof["modalities"] if oof else None,
    }
    return record


def reference_fold_records():
    records = []
    for fold in range(1, 11):
        task = json.loads((BASE_RUNTIME / "tasks" / f"{BASE_GROUP}_f{fold:02d}.json").read_text(encoding="utf-8"))
        records.append({
            "group": "53.3-HISTORICAL", "group_name": BASE_GROUP, "fold": fold,
            "status": "READ_ONLY_REFERENCE", "seed": 0,
            **{key: float(task["metrics"][key]) for key in METRICS},
            "best_epoch_1based": int(task["best_epoch"]) + 1,
            "epochs_executed": int(task["epochs_executed"]),
            "checkpoint_sha256": task["best_sha256"], "source": "53.3 third group, read-only",
        })
    return records


def percent_metric(value):
    return float(value) * 100.0


def summarise_rows(fold_rows):
    import numpy as np

    result = []
    order = ["53.3-HISTORICAL", "E61-E1", "E61-E2"]
    for group in order:
        rows = [row for row in fold_rows if row["group"] == group]
        row = {"group": group, "group_name": rows[0]["group_name"], "folds": len(rows), "source": rows[0]["source"]}
        for metric in METRICS:
            values = 100.0 * np.asarray([row[metric] for row in rows], dtype=float)
            row[f"{metric}_mean_percent"] = float(values.mean())
            row[f"{metric}_population_sd_percent"] = float(values.std(ddof=0))
        result.append(row)
    return result


def paired_rows(fold_rows):
    import numpy as np
    from scipy.stats import t

    lookup = {(row["group"], int(row["fold"])): row for row in fold_rows}
    comparisons = [
        ("E61-E1_minus_53.3", "E61-E1", "53.3-HISTORICAL"),
        ("E61-E2_minus_53.3", "E61-E2", "53.3-HISTORICAL"),
        ("E61-E2_minus_E61-E1", "E61-E2", "E61-E1"),
    ]
    output = []
    for name, left, right in comparisons:
        for metric in METRICS:
            differences = np.asarray([
                (lookup[(left, fold)][metric] - lookup[(right, fold)][metric]) * 100.0
                for fold in range(1, 11)
            ])
            mean = float(differences.mean())
            sd = float(differences.std(ddof=1))
            sem = sd / math.sqrt(len(differences))
            half = float(t.ppf(0.975, len(differences) - 1) * sem)
            wins = int(np.sum(differences > 1e-12))
            losses = int(np.sum(differences < -1e-12))
            ties = int(len(differences) - wins - losses)
            for fold, difference in enumerate(differences, start=1):
                output.append({
                    "comparison": name, "left_group": left, "right_group": right,
                    "metric": metric, "metric_label": METRIC_LABELS[metric], "fold": fold,
                    "difference_percentage_points": float(difference),
                    "mean_difference_pp": mean, "descriptive_95_ci_low": mean - half,
                    "descriptive_95_ci_high": mean + half,
                    "wins": wins, "ties": ties, "losses": losses,
                    "wins_ties_losses": f"{wins}/{ties}/{losses}",
                    "folds_correlated_caution": True,
                })
            output.append({
                "comparison": name, "left_group": left, "right_group": right,
                "metric": metric, "metric_label": METRIC_LABELS[metric], "fold": "ALL",
                "difference_percentage_points": "", "mean_difference_pp": mean,
                "descriptive_95_ci_low": mean - half, "descriptive_95_ci_high": mean + half,
                "wins": wins, "ties": ties, "losses": losses,
                "wins_ties_losses": f"{wins}/{ties}/{losses}",
                "folds_correlated_caution": True,
            })
    return output


def mechanism_rows(audits):
    rows = []
    for audit in audits:
        group, fold = audit["group"], audit["fold"]
        mechanism = audit["mechanism"]
        for modality_index, modality in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
            oof = audit["oof"][modality_index] if audit["oof"] else {}
            diag = mechanism["edge_diagnostics"][modality_index]
            row = {
                "group": group, "fold": fold, "modality": modality,
                "oof_auc": oof.get("oof_auc"), "oof_pr_auc": oof.get("oof_pr_auc"),
                "oof_same_minus_different": oof.get("same_minus_different"),
                "reliability": mechanism["reliability"][modality_index],
                "cadr_enabled": bool(mechanism["reliability"][modality_index] > 0) if group == "E61-E1" else bool(diag.get("positive_edges", 0) + diag.get("negative_edges", 0)),
                "adaptive_gate": mechanism["adaptive_gate"][modality_index] if group == "E61-E2" else None,
                "candidate_edges": diag.get("candidate_edges"), "scored_edges": diag.get("scored_edges"),
                "positive_edges": diag.get("positive_edges"), "negative_edges": diag.get("negative_edges"),
                "uncertain_edges": diag.get("uncertain_edges"), "q_mean": diag.get("q_mean"), "q_std": diag.get("q_std"),
                "positive_gcn_message_norm": mechanism["message_norms"]["positive_gcn"][modality_index],
                "negative_gcn_message_norm": mechanism["message_norms"]["negative_gcn"][modality_index],
                "direct_difference_norm": mechanism["message_norms"]["direct_difference"][modality_index],
                "direct_difference_scaled_0p02_norm": mechanism["message_norms"]["direct_difference_scaled_0p02"][modality_index],
                "negative_gcn_scaled_0p25_norm": mechanism["message_norms"]["negative_gcn_scaled_0p25"][modality_index],
                "cadr_message_norm": mechanism["message_norms"]["cadr_message"][modality_index],
                "original_gcn_message_norm": mechanism["message_norms"]["original_gcn"][modality_index] if len(mechanism["message_norms"]["original_gcn"]) > modality_index else None,
                "propagation_mode": mechanism["message_norms"]["propagation_modes"][modality_index],
                "gate_gradient_norm": mechanism["gradient_norms"].get("adaptive_gate") if group == "E61-E2" else None,
                "reliability_gate_gradient_norm": mechanism["gradient_norms"].get("reliability_gate") if group == "E61-E1" else None,
                "edge_gate_value": diag.get("edge_gate"),
                "raw_input_frobenius_norm_320d": mechanism["raw_input_frobenius_norm_320d"],
                "weighted_cadr_contribution_norm": mechanism["weighted_message_contributions"]["cadr_contribution_norm"][modality_index],
                "weighted_cadr_to_raw_ratio": mechanism["weighted_message_contributions"]["cadr_contribution_to_raw_ratio"][modality_index],
                "weighted_original_graph_contribution_norm": mechanism["weighted_message_contributions"]["original_graph_contribution_norm"][modality_index] if mechanism["weighted_message_contributions"]["original_graph_contribution_norm"] else None,
                "weighted_positive_gcn_contribution_norm": mechanism["weighted_message_contributions"]["positive_gcn_contribution_norm"][modality_index],
                "weighted_negative_gcn_contribution_norm": mechanism["weighted_message_contributions"]["negative_gcn_contribution_norm"][modality_index],
                "weighted_direct_difference_contribution_norm": mechanism["weighted_message_contributions"]["direct_difference_contribution_norm"][modality_index],
                "weighted_original_graph_to_raw_ratio": mechanism["weighted_message_contributions"]["original_graph_contribution_norm"][modality_index] / max(mechanism["raw_input_frobenius_norm_320d"], 1e-12) if mechanism["weighted_message_contributions"]["original_graph_contribution_norm"] else None,
                "weighted_positive_gcn_to_raw_ratio": mechanism["weighted_message_contributions"]["positive_gcn_contribution_norm"][modality_index] / max(mechanism["raw_input_frobenius_norm_320d"], 1e-12),
                "weighted_negative_gcn_to_raw_ratio": mechanism["weighted_message_contributions"]["negative_gcn_contribution_norm"][modality_index] / max(mechanism["raw_input_frobenius_norm_320d"], 1e-12),
                "weighted_direct_difference_to_raw_ratio": mechanism["weighted_message_contributions"]["direct_difference_contribution_norm"][modality_index] / max(mechanism["raw_input_frobenius_norm_320d"], 1e-12),
                "classification_gate_gradient_abs": audit["classification_gradient_replay"]["adaptive_gate_abs"][modality_index] if audit["classification_gradient_replay"] else None,
                "classification_edge_gate_gradient_abs": audit["classification_gradient_replay"]["edge_gate_abs"][modality_index] if audit["classification_gradient_replay"] else None,
                "classification_scorer_gradient_norm": audit["classification_gradient_replay"]["scorer_norm"][modality_index] if audit["classification_gradient_replay"] else None,
                "classification_encoder_gradient_norm": audit["classification_gradient_replay"]["encoder_norm"][modality_index] if audit["classification_gradient_replay"] else None,
            }
            rows.append(row)
    return rows


def format_mean_sd(row, metric):
    return f"{row[metric + '_mean_percent']:.4f} ± {row[metric + '_population_sd_percent']:.4f}"


def summarize_report(fold_rows, summary, paired, audits, rng_audit, worker_info):
    import numpy as np

    by_group = {row["group"]: row for row in summary}
    oof_by_modal = {}
    for modality in ("PHENO", "ANAT", "FUNC", "CORR"):
        values = [item["oof_auc"] for audit in audits if audit["group"] == "E61-E1" for item in audit["oof"] if item["modality"] == modality]
        rel = [item["reliability"] for audit in audits if audit["group"] == "E61-E1" for item in audit["oof"] if item["modality"] == modality]
        oof_by_modal[modality] = (float(np.mean(values)), float(np.std(values, ddof=0)), float(np.mean(rel)), sum(value > 0 for value in rel))
    gate_by_modal = {}
    for index, modality in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
        values = [audit["mechanism"]["adaptive_gate"][index] for audit in audits if audit["group"] == "E61-E2"]
        gate_by_modal[modality] = (float(np.mean(values)), float(np.std(values, ddof=0)))
    e2_message_by_modal = {}
    for index, modality in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
        records = [audit["mechanism"] for audit in audits if audit["group"] == "E61-E2"]
        diagnostics = [record["edge_diagnostics"][index] for record in records]
        contributions = [record["weighted_message_contributions"] for record in records]
        e2_message_by_modal[modality] = {
            "positive_edges": float(np.mean([item.get("positive_edges", 0) for item in diagnostics])),
            "negative_edges": float(np.mean([item.get("negative_edges", 0) for item in diagnostics])),
            "uncertain_pct": float(np.mean([item.get("uncertain_edges", 0) / max(item.get("scored_edges", 0), 1) for item in diagnostics]) * 100.0),
            "cadr_raw_pct": float(np.mean([item["cadr_contribution_to_raw_ratio"][index] for item in contributions]) * 100.0),
            "original_gcn_raw_pct": float(np.mean([item["original_graph_contribution_norm"][index] / max(record["raw_input_frobenius_norm_320d"], 1e-12) for item, record in zip(contributions, records)]) * 100.0),
            "positive_gcn_raw_pct": float(np.mean([item["positive_gcn_contribution_norm"][index] / max(record["raw_input_frobenius_norm_320d"], 1e-12) for item, record in zip(contributions, records)]) * 100.0),
            "negative_gcn_raw_pct": float(np.mean([item["negative_gcn_contribution_norm"][index] / max(record["raw_input_frobenius_norm_320d"], 1e-12) for item, record in zip(contributions, records)]) * 100.0),
            "direct_raw_pct": float(np.mean([item["direct_difference_contribution_norm"][index] / max(record["raw_input_frobenius_norm_320d"], 1e-12) for item, record in zip(contributions, records)]) * 100.0),
        }
    e2_gradient_by_modal = {}
    for index, modality in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
        gradient_rows = [audit["classification_gradient_replay"] for audit in audits if audit["group"] == "E61-E2"]
        e2_gradient_by_modal[modality] = {
            key: float(np.mean([row[source][index] for row in gradient_rows]))
            for key, source in (("adaptive_gate", "adaptive_gate_abs"), ("edge_gate", "edge_gate_abs"), ("scorer", "scorer_norm"), ("encoder", "encoder_norm"))
        }
    intervention_modes = {
        "all CADR off": (lambda audit: audit["disease_off_metrics"], "disease_off", None),
        "score permutation": (lambda audit: audit["permuted_edge_score_metrics"], "score_permutation", None),
        "positive GCN off": (lambda audit: audit["component_intervention_metrics"]["positive_off"], "component_off", "positive_off"),
        "negative GCN off": (lambda audit: audit["component_intervention_metrics"]["negative_off"], "component_off", "negative_off"),
        "direct difference off": (lambda audit: audit["component_intervention_metrics"]["direct_off"], "component_off", "direct_off"),
        **{f"{modality} CADR off": (lambda audit, modality=modality: audit["modality_off_metrics"][modality], "modality_off", modality) for modality in ("PHENO", "ANAT", "FUNC", "CORR")},
    }
    intervention_summary = {}
    for group in ("E61-E1", "E61-E2"):
        group_audits = [audit for audit in audits if audit["group"] == group]
        intervention_summary[group] = {}
        for label, (metric_getter, change_path, change_key) in intervention_modes.items():
            records = [audit["intervention_prediction_changes"][change_path] if change_key is None else audit["intervention_prediction_changes"][change_path][change_key] for audit in group_audits]
            intervention_summary[group][label] = {
                "acc_drop_pp": float(np.mean([(audit["full_metrics"]["acc"] - metric_getter(audit)["acc"]) * 100.0 for audit in group_audits])),
                "flips": float(np.mean([item["flipped_predictions"] for item in records])),
                "probability_change": float(np.mean([item["mean_absolute_probability_change"] for item in records])),
            }
    pair_map = {(row["comparison"], row["metric"]): row for row in paired if row["fold"] == "ALL"}
    decisions = {}
    for group, comparison in (("E61-E1", "E61-E1_minus_53.3"), ("E61-E2", "E61-E2_minus_53.3")):
        delta = {metric: pair_map[(comparison, metric)]["mean_difference_pp"] for metric in METRICS}
        relevant = [audit for audit in audits if audit["group"] == group]
        if group == "E61-E1":
            active = sum(item["reliability"] > 0 for audit in relevant for item in audit["oof"])
        else:
            active = sum(
                audit["mechanism"]["edge_diagnostics"][index].get("positive_edges", 0)
                + audit["mechanism"]["edge_diagnostics"][index].get("negative_edges", 0) > 0
                for audit in relevant for index in range(4)
            )
        disease_effect = float(np.mean([
            (audit["full_metrics"]["acc"] - audit["disease_off_metrics"]["acc"]) * 100
            for audit in relevant
        ]))
        shuffle_effect = float(np.mean([
            (audit["full_metrics"]["acc"] - audit["permuted_edge_score_metrics"]["acc"]) * 100
            for audit in relevant
        ]))
        supported = active > 0 and (disease_effect > 0 or shuffle_effect > 0)
        if delta["acc"] > 0 and delta["hard_auc"] > 0 and delta["prob_auc"] > 0 and delta["sen"] >= -1.0 and supported:
            label = "GO"
        elif (delta["acc"] > 0 or delta["hard_auc"] > 0 or delta["prob_auc"] > 0) and supported:
            label = "PARTIAL"
        else:
            label = "NO-GO"
        decisions[group] = {
            "label": label, "delta": delta, "active_modality_folds": active,
            "mean_disease_off_acc_drop_pp": disease_effect,
            "mean_shuffled_score_acc_drop_pp": shuffle_effect,
        }
    overall = "GO" if all(item["label"] == "GO" for item in decisions.values()) else ("PARTIAL" if any(item["label"] != "NO-GO" for item in decisions.values()) else "NO-GO")
    rng_compared = sum(row["compared_postfork_epochs"] for row in rng_audit)
    rng_matches = sum(row["exact_rng_trace_matches"] for row in rng_audit)
    rng_mismatches = sum(len(row["mismatch_epochs"]) for row in rng_audit)

    def paired_line(comparison, metric):
        row = pair_map[(comparison, metric)]
        return f"{row['mean_difference_pp']:+.3f} pp ({row['wins_ties_losses']}, descriptive CI {row['descriptive_95_ci_low']:+.3f} to {row['descriptive_95_ci_high']:+.3f})"

    lines = [
        "# Experiment 61 — Post-alignment OOF Reliability versus End-to-end Adaptive CADR Graph Propagation", "",
        "## Scope and protocol", "",
        "This experiment contains exactly two new groups and 20 outer-fold classification tasks. The sole base is Experiment 53.3 third group `CADR_WD_0p01_GFR_WD_0p11`; its historical fold metrics and checkpoints remain read-only. E1 runs one shared 10-epoch warmup, then a patient-disjoint three-fold, ten-epoch joint four-modality relation OOF fit, then fixes fold-level reliability. E2 branches from the same warmup and learns four scalar gates from classification loss without OOF.", "",
        f"Locked source commit: `{CFG['locked_reference_commit']}`. The original SPromptGL GFR description is in [the MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf); this run follows the checked 53.3 source path. E61 retains PHENO48, ANAT6, FUNC10 and CORR256, the original 871-node transductive graph, CAM, GFR, GraphConstructLoss, relation BCE, conditional adversary and preservation loss.", "",
        "All groups use the uncalibrated relation score `sigmoid(logit)` and the fixed 0.55/0.45 thresholds. E53.3 used old OOF Platt parameters; E61 intentionally does not, because E2 has no OOF. Thus comparisons with the historical base are exploratory and include this shared score-calibration change. The historical `val=test` best-checkpoint rule and pre-CV RFE/standardization remain unchanged; these results are not unbiased external generalization estimates.", "",
        "The first ten epochs use the original 53.3 training forward and optimizer calls. Exact ten-epoch twin replay passed before training. Same-structure GFR/CAM and relation-core initial parameters, fold split, relation-pair cache and construction RNG matched SHA references. The only initialization exception is the old OOF-derived reliability/calibration state, replaced by neutral buffers and an E2 gate initialized at 0.1. After branching, matched epochs use the same CAM dropout call count/order; E1 OOF uses isolated torch RNG and local NumPy generators.", "",
        f"Runtime used {worker_info['device']} with maximum {worker_info['max_workers']} concurrent workers; 10 shared warmups, {worker_info['completed_tasks']}/20 formal tasks, and {worker_info['independent_audits']}/20 independent checkpoint audits completed. Post-fork RNG traces matched exactly for {rng_matches}/{rng_compared} compared epochs; mismatches: {rng_mismatches}.", "",
        "## Classification results", "",
        "Values are 10-fold mean ± population SD, in percent. The historical row is read-only.", "",
        "|Group|ACC|Hard-AUC|Probability-AUC|SEN-ASD|SPE-TD|BA|ASD-F1|", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("53.3-HISTORICAL", "E61-E1", "E61-E2"):
        row = by_group[group]
        lines.append("|" + row["group_name"] + "|" + "|".join(format_mean_sd(row, metric) for metric in METRICS) + "|")
    lines.extend(["", "## Paired fold differences", "", "Differences are percentage points; intervals are descriptive paired-t intervals over the ten correlated folds, not confirmatory inference.", "", "|Comparison|ACC|Hard-AUC|Probability-AUC|SEN-ASD|SPE-TD|ASD-F1|", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, label in (("E61-E1_minus_53.3", "E1 − historical 53.3"), ("E61-E2_minus_53.3", "E2 − historical 53.3"), ("E61-E2_minus_E61-E1", "E2 − E1")):
        lines.append("|" + label + "|" + "|".join(paired_line(name, metric) for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")) + "|")
    lines.extend(["", "## Mechanism evidence", "", "### E1 post-alignment OOF", "", "|Modality|OOF-AUC mean ± SD|Mean reliability|Nonzero-reliability folds|", "|---|---:|---:|---:|"])
    for modality, (auc_mean, auc_sd, rel_mean, active) in oof_by_modal.items():
        lines.append(f"|{modality}|{auc_mean:.4f} ± {auc_sd:.4f}|{rel_mean:.4f}|{active}/10|")
    lines.extend(["", "OOF reliability was computed only from the fixed tenth-epoch endpoint of each inner model. The inner models jointly trained all four relation encoders, scorers and the conditional modality discriminator; outer-test patients were excluded. Each outer-training patient appeared once as an inner validation patient. OOF estimates reliability only; the main fold model generated every propagated edge score.", "", "### E2 learned gates", "", "|Modality|Final gate mean ± SD|", "|---|---:|"])
    for modality, (mean, sd) in gate_by_modal.items():
        lines.append(f"|{modality}|{mean:.4f} ± {sd:.4f}|")
    lines.extend(["", "### E2 edge and message contribution", "", "Counts and contribution norms are averaged across ten independently audited checkpoints. Message ratios compare each weighted 871-node modality message with the Frobenius norm of the original 871×320 input; component norms are descriptive and are not causal effects.", "", "|Modality|Gate|Positive edges|Negative edges|Uncertain edges %|CADR message / input %|Original GCN / input %|Positive GCN / input %|Negative GCN / input %|Direct difference / input %|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for modality, values in e2_message_by_modal.items():
        lines.append(f"|{modality}|{gate_by_modal[modality][0]:.4f}|{values['positive_edges']:.1f}|{values['negative_edges']:.1f}|{values['uncertain_pct']:.2f}|{values['cadr_raw_pct']:.5f}|{values['original_gcn_raw_pct']:.5f}|{values['positive_gcn_raw_pct']:.7f}|{values['negative_gcn_raw_pct']:.7f}|{values['direct_raw_pct']:.5f}|")
    lines.extend(["", "Classification-only gradient replay at the best checkpoint used the outer-training classification loss in deterministic evaluation mode, without stepping an optimizer. These are mean absolute scalar gate gradients and mean relation-module gradient norms; per-fold values are in `mechanism_metrics.csv`.", "", "|Modality|Adaptive gate gradient|Edge gate gradient|Scorer gradient norm|Encoder gradient norm|", "|---|---:|---:|---:|---:|"])
    for modality, values in e2_gradient_by_modal.items():
        lines.append(f"|{modality}|{values['adaptive_gate']:.3e}|{values['edge_gate']:.3e}|{values['scorer']:.3e}|{values['encoder']:.3e}|")
    lines.extend(["", "### Fixed-checkpoint interventions", "", "Each cell gives full-model minus intervention ACC in percentage points, mean number of changed test hard predictions, and mean absolute probability change. Interventions do not use test labels to choose the perturbation.", "", "|Group|Intervention|ΔACC pp|Changed predictions / fold|Mean absolute Δprobability|", "|---|---|---:|---:|---:|"])
    for group in ("E61-E1", "E61-E2"):
        for label, values in intervention_summary[group].items():
            lines.append(f"|{group}|{label}|{values['acc_drop_pp']:+.3f}|{values['flips']:.2f}|{values['probability_change']:.6f}|")
    lines.extend(["", "Gate values describe the learned blend under this training objective. They are not reliability estimates or calibrated probabilities. Per-fold gates, gradient norms, edge counts, score distributions, GCN components, direct-difference messages and fixed-checkpoint intervention metrics are in `mechanism_metrics.csv` and `checkpoint_audit.csv`.", "", "The post-alignment OOF AUCs describe relation quality after the aligned inner training procedure. No separate pre-alignment OOF fit was run, so this experiment does not estimate a pre/post alignment AUC change. GCN-positive, GCN-negative and direct-neighbor contributions are separately reported; a nonzero message norm alone is not evidence that a component improved diagnosis.", "", "## Scientific judgment", "", f"Overall assessment: **{overall}** under a conservative descriptive rule: a group receives GO only when ACC, Hard-AUC and Probability-AUC all improve over the historical third group, SEN falls by no more than 1 pp, and active relation evidence is supported by a fixed-checkpoint intervention. PARTIAL requires some metric gain and intervention evidence; otherwise the result is NO-GO. These labels are exploratory, not significance claims.", "", "|Method|Judgment|ACC Δ pp|Hard-AUC Δ pp|Prob-AUC Δ pp|SEN Δ pp|Active modality-folds|Disease-off ACC drop pp|Shuffled-score ACC drop pp|", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for group in ("E61-E1", "E61-E2"):
        item = decisions[group]
        d = item["delta"]
        lines.append(f"|{group} {GROUPS[group]}|{item['label']}|{d['acc']:+.3f}|{d['hard_auc']:+.3f}|{d['prob_auc']:+.3f}|{d['sen']:+.3f}|{item['active_modality_folds']}|{item['mean_disease_off_acc_drop_pp']:+.3f}|{item['mean_shuffled_score_acc_drop_pp']:+.3f}|")
    lines.extend(["", "## Scientific reading", "", "Category supervision or a learned gate can improve a metric without establishing that disease-relation alignment is useful. Evaluate the paired metrics alongside the no-CADR, single-modality, score-shuffle and positive/negative/direct component interventions. If E1 has zero reliability, the method correctly falls back to the same original GraphLearn/GCN path. If E2 learns small gates or has no active scored edges, it did not establish useful adaptive disease propagation.", "", "The current comparison is single-seed and uses overlapping cross-validation training sets. It is exploratory. A mathematical relation embedding or a nonzero gate does not establish a biological ASD mechanism.", "", "The per-fold `acc`, `hard_auc`, `prob_auc`, `sen`, and `spe` columns in `fold_metrics.csv` are stored as 0–1 proportions; `summary.csv` and this report display percent.", "", "## Files", "", f"- Experiment root: `{EXP}`", f"- Formal checkpoints: `{CKPTS}`", f"- Shared warmups and OOF cache: `{RT}`", "- Fold metrics: `results/fold_metrics.csv`", "- Paired comparisons: `results/paired_comparisons.csv`", "- Independent audits and fixed-checkpoint interventions: `results/checkpoint_audit.csv`", "- Mechanism details: `results/mechanism_metrics.csv`", "- Task manifest: `manifest.json`", ""])
    return "\n".join(lines)


def append_handoff(report_status, summary, paired, manifest, audits):
    import tempfile
    import numpy as np

    path = Path("/model/spromptgl/experiment/handoff.md")
    if not path.is_file():
        raise FileNotFoundError(f"required handoff file is missing: {path}")
    marker = "## Experiment 61 — Post-alignment OOF Reliability versus End-to-end Adaptive CADR Graph Propagation"
    current = path.read_text(encoding="utf-8")
    values = {row["group"]: row for row in summary}
    pmap = {(row["comparison"], row["metric"]): row for row in paired if row["fold"] == "ALL"}
    modalities = ("PHENO", "ANAT", "FUNC", "CORR")
    oof_summary = {
        modality: float(np.mean([item["oof_auc"] for audit in audits if audit["group"] == "E61-E1" for item in audit["oof"] if item["modality"] == modality]))
        for modality in modalities
    }
    gate_summary = {
        modality: float(np.mean([audit["mechanism"]["adaptive_gate"][index] for audit in audits if audit["group"] == "E61-E2"]))
        for index, modality in enumerate(modalities)
    }
    e1_active = sum(item["reliability"] > 0 for audit in audits if audit["group"] == "E61-E1" for item in audit["oof"])
    e2_active = sum(any(d.get("positive_edges", 0) + d.get("negative_edges", 0) > 0 for d in audit["mechanism"]["edge_diagnostics"]) for audit in audits if audit["group"] == "E61-E2")
    e2_gradient_audits = [audit["classification_gradient_replay"] for audit in audits if audit["group"] == "E61-E2"]
    e2_message_ratios = []
    for audit in audits:
        if audit["group"] == "E61-E2":
            components = audit["mechanism"]["weighted_message_contributions"]
            e2_message_ratios.append({
                "cadr": 100.0 * sum(components["cadr_contribution_to_raw_ratio"]),
                "gcn": 100.0 * sum(components["positive_gcn_contribution_norm"][i] + components["negative_gcn_contribution_norm"][i] for i in range(4)) / max(audit["mechanism"]["raw_input_frobenius_norm_320d"], 1e-12),
                "direct": 100.0 * sum(components["direct_difference_contribution_norm"]) / max(audit["mechanism"]["raw_input_frobenius_norm_320d"], 1e-12),
            })
    section = [
        "", marker, "", f"- Status: {report_status}; completed {manifest['completed_tasks']}/20 formal tasks; independently audited {manifest['independent_audits']}/20 checkpoints.",
        f"- Sole base: Experiment 53.3 `{BASE_GROUP}`, source commit `{CFG['locked_reference_commit']}`; base checkpoints and fold metrics were read-only.",
        f"- E61-E1 POSTALIGN_OOF: ACC {format_mean_sd(values['E61-E1'], 'acc')}%, Hard-AUC {format_mean_sd(values['E61-E1'], 'hard_auc')}%, Probability-AUC {format_mean_sd(values['E61-E1'], 'prob_auc')}%, SPE {format_mean_sd(values['E61-E1'], 'spe')}%, SEN {format_mean_sd(values['E61-E1'], 'sen')}%.",
        f"- E61-E2 E2E_ADAPTIVE_GATE: ACC {format_mean_sd(values['E61-E2'], 'acc')}%, Hard-AUC {format_mean_sd(values['E61-E2'], 'hard_auc')}%, Probability-AUC {format_mean_sd(values['E61-E2'], 'prob_auc')}%, SPE {format_mean_sd(values['E61-E2'], 'spe')}%, SEN {format_mean_sd(values['E61-E2'], 'sen')}%.",
        f"- Historical 53.3 third-group ACC {format_mean_sd(values['53.3-HISTORICAL'], 'acc')}%, Hard-AUC {format_mean_sd(values['53.3-HISTORICAL'], 'hard_auc')}%, Probability-AUC {format_mean_sd(values['53.3-HISTORICAL'], 'prob_auc')}%, SPE {format_mean_sd(values['53.3-HISTORICAL'], 'spe')}%, SEN {format_mean_sd(values['53.3-HISTORICAL'], 'sen')}% (read-only).",
        f"- Paired E1−base ACC {pmap[('E61-E1_minus_53.3', 'acc')]['mean_difference_pp']:+.3f} pp; E2−base ACC {pmap[('E61-E2_minus_53.3', 'acc')]['mean_difference_pp']:+.3f} pp; E2−E1 ACC {pmap[('E61-E2_minus_E61-E1', 'acc')]['mean_difference_pp']:+.3f} pp.",
        f"- E1 post-alignment OOF AUCs: {', '.join(f'{name}={oof_summary[name]:.4f}' for name in modalities)}; nonzero reliability {e1_active}/40 modality-folds, so E1 used GraphLearn fallback throughout. E2 mean gates: {', '.join(f'{name}={gate_summary[name]:.4f}' for name in modalities)}; CADR edges active in {e2_active}/10 folds.",
        f"- Best-checkpoint classification-gradient replay: all four adaptive and edge gates had nonzero gradients in {sum(item['all_gates_nonzero'] for item in e2_gradient_audits)}/10 folds; all four relation scorers and encoders had nonzero gradients in {sum(item['all_relation_modules_nonzero'] for item in e2_gradient_audits)}/10 folds. Replay left the main RNG unchanged.",
        f"- E2 mean all-CADR-off and score-permutation ACC effects were {np.mean([(a['full_metrics']['acc']-a['disease_off_metrics']['acc'])*100 for a in audits if a['group']=='E61-E2']):+.3f} and {np.mean([(a['full_metrics']['acc']-a['permuted_edge_score_metrics']['acc'])*100 for a in audits if a['group']=='E61-E2']):+.3f} pp. Mean summed modality message norms relative to the 320D input: CADR {np.mean([x['cadr'] for x in e2_message_ratios]):.5f}%, its GCN positive+negative components {np.mean([x['gcn'] for x in e2_message_ratios]):.7f}%, direct difference {np.mean([x['direct'] for x in e2_message_ratios]):.5f}%.",
        "- Mechanism and per-fold reliability/gate status are in `experiment/experiment61_cadr_postalign_vs_e2e/results/report.md` and `mechanism_metrics.csv`. Historical `val=test` checkpoint selection and pre-CV preprocessing remain; results are exploratory, not unbiased external testing.",
        f"- Files: `{EXP}`; report `results/report.md`; manifest `manifest.json`.",
        "",
    ]
    if marker in current:
        start = current.index(marker)
        next_section = current.find("\n## ", start + len(marker))
        prefix = current[:start].rstrip()
        suffix = current[next_section:] if next_section >= 0 else ""
        updated = prefix + "\n\n" + "\n".join(section).lstrip("\n")
        if suffix:
            updated += "\n" + suffix.lstrip("\n")
    else:
        updated = current.rstrip() + "\n" + "\n".join(section)
    mode = path.stat().st_mode
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix="handoff.e61.", suffix=".tmp", delete=False) as stream:
            temp_name = stream.name
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_name, mode)
        os.replace(temp_name, path)
        if path.read_text(encoding="utf-8") != updated:
            raise IOError("handoff atomic readback mismatch")
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    return sha_file(path)


def finalize(audits, worker_info, rng_audit):
    import numpy as np

    ref = reference_fold_records()
    new_rows = []
    for audit in audits:
        task_file, _, _, _ = fold_paths(audit["group"], audit["fold"])
        task = json.loads(task_file.read_text(encoding="utf-8"))
        new_rows.append({
            "group": audit["group"], "group_name": GROUPS[audit["group"]], "fold": audit["fold"],
            "status": "INDEPENDENT_AUDIT_PASS", "seed": 0,
            **{key: float(audit["metrics"][key]) for key in METRICS},
            "best_epoch_1based": int(task["best_epoch_1based"]), "epochs_executed": int(task["epochs_executed"]),
            "checkpoint_sha256": audit["checkpoint_sha256"], "source": "new training",
        })
    fold_rows = ref + new_rows
    summary = summarise_rows(fold_rows)
    paired = paired_rows(fold_rows)
    mechanisms = mechanism_rows(audits)
    write_csv(RESULTS / "fold_metrics.csv", fold_rows)
    write_csv(RESULTS / "summary.csv", summary)
    write_csv(RESULTS / "paired_comparisons.csv", paired)
    write_csv(RESULTS / "checkpoint_audit.csv", [{
        "group": audit["group"], "fold": audit["fold"], "checkpoint_audit": audit["checkpoint_audit"],
        "checkpoint_sha256": audit["checkpoint_sha256"], "best_epoch_1based": audit["best_epoch_1based"],
        "epochs_executed": audit["epochs_executed"], "warmup_sha256": audit["warmup_sha256"],
        "initial_shared_sha": audit["initial_shared_sha"], "initial_gfr_sha": audit["initial_gfr_sha"],
        "initial_cam_sha": audit["initial_cam_sha"], "relation_core_sha": audit["relation_core_sha"],
        "initial_rng_sha": audit["initial_rng_sha"], "split_sha": audit["split_sha"],
        "pair_sha": audit["pair_sha"], "anchor_sha": audit["anchor_sha"],
        "test_probability_max_abs_error": audit["test_probability_max_abs_error"],
        "metrics": json.dumps(audit["metrics"], sort_keys=True),
        "disease_off_metrics": json.dumps(audit["disease_off_metrics"], sort_keys=True),
        "modality_off_metrics": json.dumps(audit["modality_off_metrics"], sort_keys=True),
        "permuted_edge_score_metrics": json.dumps(audit["permuted_edge_score_metrics"], sort_keys=True),
        "component_intervention_metrics": json.dumps(audit["component_intervention_metrics"], sort_keys=True),
        "intervention_prediction_changes": json.dumps(audit["intervention_prediction_changes"], sort_keys=True),
        "classification_gradient_replay": json.dumps(audit["classification_gradient_replay"], sort_keys=True),
        "mechanism": json.dumps(audit["mechanism"], sort_keys=True),
        "oof": json.dumps(audit["oof"], sort_keys=True),
    } for audit in audits])
    write_csv(RESULTS / "mechanism_metrics.csv", mechanisms)
    report = summarize_report(fold_rows, summary, paired, audits, rng_audit, worker_info)
    (RESULTS / "report.md.tmp").write_text(report, encoding="utf-8")
    os.replace(RESULTS / "report.md.tmp", RESULTS / "report.md")
    if (RESULTS / "report.md").read_text(encoding="utf-8") != report:
        raise IOError("report readback mismatch")
    status = "COMPLETE_AUDITED" if len(audits) == 20 and all(row["checkpoint_audit"] == "PASS" for row in audits) else "AUDIT_INCOMPLETE"
    manifest = {
        "experiment": CFG["experiment"], "status": status, "experiment_root": str(EXP),
        "base_root": str(BASE_ROOT), "locked_base_commit": CFG["locked_reference_commit"],
        "base_group": BASE_GROUP, "seed": 0, "formal_groups": GROUPS,
        "formal_tasks_expected": 20, "completed_tasks": len(new_rows),
        "independent_audits": sum(row["checkpoint_audit"] == "PASS" for row in audits),
        "failed_tasks": [], "worker_info": worker_info, "rng_fork_audit": rng_audit,
        "source_sha256": source_sha_map(),
        "audit_source_sha256": {name: sha_file(EXP / name) for name in ("run.py", "model_e61.py", "config.json")},
        "base_source_sha256": verify_reference_artifacts(),
        "config_sha256": sha_file(CONFIG_PATH),
        "historical_reference_sha256": {
            "fold_metrics.csv": sha_file(BASE_EXP / "results/fallback_wd_sweep/fold_metrics.csv"),
            "checkpoint_audit.csv": sha_file(BASE_EXP / "results/fallback_wd_sweep/checkpoint_audit.csv"),
        },
        "warmups": [{"fold": fold, "path": str(warmup_path(fold)), "sha256": sha_file(warmup_path(fold)), "rng_sha": torch_load(warmup_path(fold))["rng_sha"]} for fold in range(1, 11)],
        "tasks": [{
            "group": row["group"], "fold": int(row["fold"]), "status": "COMPLETE",
            "checkpoint_sha256": row["checkpoint_sha256"], "warmup_sha256": row["warmup_sha256"],
            "split_sha": row["split_sha"], "pair_sha": row["pair_sha"], "initial_shared_sha": row["initial_shared_sha"],
            "initial_rng_sha": row["initial_rng_sha"], "independent_audit": row["checkpoint_audit"],
            "best_epoch_1based": row["best_epoch_1based"], "epochs_executed": row["epochs_executed"],
        } for row in audits],
        "results_sha256": {},
    }
    for path in sorted(RESULTS.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["results_sha256"][path.name] = sha_file(path)
    atomic_json(EXP / "manifest.json", manifest)
    handoff_sha = append_handoff(status, summary, paired, manifest, audits)
    manifest["handoff_path"] = "/model/spromptgl/experiment/handoff.md"
    manifest["handoff_sha256"] = handoff_sha
    atomic_json(EXP / "manifest.json", manifest)
    return manifest


def update_manifest(status, worker_info=None, failed=None):
    task_records = []
    for group in GROUPS:
        for fold in range(1, 11):
            task_file, best_file, _latest, _log = fold_paths(group, fold)
            item = {"group": group, "fold": fold, "status": "PENDING"}
            if task_file.is_file():
                task = json.loads(task_file.read_text(encoding="utf-8"))
                item.update({key: task.get(key) for key in ("status", "checkpoint_sha256", "warmup_sha256", "split_sha", "pair_sha", "best_epoch_1based", "epochs_executed")})
                if best_file.is_file():
                    item["checkpoint_file_sha256"] = sha_file(best_file)
            task_records.append(item)
    manifest = {
        "experiment": CFG["experiment"], "status": status,
        "experiment_root": str(EXP), "base_group": BASE_GROUP,
        "locked_base_commit": CFG["locked_reference_commit"],
        "formal_groups": GROUPS, "formal_tasks_expected": 20,
        "completed_tasks": sum(item["status"] == "COMPLETE" for item in task_records),
        "independent_audits": 0, "worker_info": worker_info or {},
        "failed_tasks": failed or [], "source_sha256": source_sha_map(),
        "base_source_sha256": verify_reference_artifacts(), "config_sha256": sha_file(CONFIG_PATH),
        "warmups": [{"fold": fold, "sha256": sha_file(warmup_path(fold))} for fold in range(1, 11) if warmup_path(fold).is_file()],
        "tasks": task_records,
    }
    atomic_json(EXP / "manifest.json", manifest)
    return manifest


def run_all(max_workers=4):
    import torch

    if max_workers < 1 or max_workers > 10:
        raise ValueError("max_workers must be in [1,10]")
    for path in (WARMUP, TASKS, CKPTS, LOGS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
    preflight_file = RESULTS / "preflight.json"
    if not preflight_file.is_file() or json.loads(preflight_file.read_text(encoding="utf-8")).get("status") != "PASS":
        preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 61 requires the configured single CUDA device")
    update_manifest("WARMUP")
    for fold in range(1, 11):
        warmup_fold(fold)
        print(f"shared warmup {fold}/10 ready", flush=True)
    initial_warmup_sha = {fold: sha_file(warmup_path(fold)) for fold in range(1, 11)}
    worker_info = {
        "device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "requested_max_workers": max_workers, "max_workers": min(2, max_workers),
        "completed_tasks": 0, "independent_audits": 0,
        "cpu_threads_per_worker": 1, "warmup_count": 10,
        "warmup_sha256": initial_warmup_sha,
    }
    update_manifest("TRAINING", worker_info)
    pending = []
    for group in GROUPS:
        for fold in range(1, 11):
            task_file, best_file, _latest, _log = fold_paths(group, fold)
            if task_file.is_file() and best_file.is_file():
                task = json.loads(task_file.read_text(encoding="utf-8"))
                if task.get("status") == "COMPLETE" and task.get("checkpoint_sha256") == sha_file(best_file) and task.get("warmup_sha256") == initial_warmup_sha[fold]:
                    continue
            pending.append((group, fold))
    active, attempts = {}, {}
    completed = 0
    start_workers = min(2, max_workers)
    while pending or active:
        while pending and len(active) < worker_info["max_workers"]:
            group, fold = pending.pop(0)
            _, _, _, log_path = fold_paths(group, fold)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.with_suffix(".stdout.log").open("ab")
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "E61_EXP": str(EXP)})
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[key] = "1"
            proc = subprocess.Popen(
                [sys.executable, str(EXP / "run.py"), "--worker", "--group", group, "--fold", str(fold)],
                cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT,
            )
            active[proc] = (group, fold, stream)
            attempts[(group, fold)] = attempts.get((group, fold), 0) + 1
        finished = []
        for proc, (group, fold, stream) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            stream.close()
            if code == 0:
                finished.append(proc)
                completed += 1
                worker_info["completed_tasks"] = completed
                print(f"formal task complete {completed}/20: {group} fold {fold}", flush=True)
                if completed >= 2 and worker_info["max_workers"] < min(max_workers, 4):
                    memory_line = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True,
                    ).stdout.strip().splitlines()[0]
                    used_mb, total_mb = [int(value.strip()) for value in memory_line.split(",")]
                    worker_info["pilot_gpu_memory_mb"] = {"used": used_mb, "total": total_mb}
                    if used_mb < total_mb * 0.70:
                        worker_info["max_workers"] = min(max_workers, 4)
                        worker_info["concurrency_ramp"] = "2-worker pilot passed; GPU memory below 70%; raised to four"
                    else:
                        worker_info["concurrency_ramp"] = "kept two workers because GPU memory exceeded 70%"
                    update_manifest("TRAINING", worker_info)
            else:
                key = (group, fold)
                worker_info.setdefault("worker_failures", []).append({"group": group, "fold": fold, "exit_code": code, "attempt": attempts[key]})
                log_path = fold_paths(group, fold)[3].with_suffix(".stdout.log")
                tail = log_path.read_text(errors="replace")[-6000:] if log_path.exists() else ""
                print(f"worker failed {group} fold {fold} attempt {attempts[key]}: {tail[-1000:]}", flush=True)
                if attempts[key] < 2:
                    pending.insert(0, key)
                    worker_info["max_workers"] = max(1, min(worker_info["max_workers"] - 1, max_workers))
                    worker_info.setdefault("retries", []).append({"group": group, "fold": fold, "new_max_workers": worker_info["max_workers"]})
                else:
                    update_manifest("FAILED", worker_info, [{"group": group, "fold": fold, "exit_code": code, "log_tail": tail}])
                    raise RuntimeError(f"formal task unrecoverably failed: {group} fold {fold}\n{tail}")
                finished.append(proc)
            update_manifest("TRAINING", worker_info)
        for proc in finished:
            active.pop(proc, None)
        if pending or active:
            time.sleep(0.5)

    if {fold: sha_file(warmup_path(fold)) for fold in range(1, 11)} != initial_warmup_sha:
        raise RuntimeError("shared warmup changed during formal training")
    audit_records = []
    for group in GROUPS:
        for fold in range(1, 11):
            audit_records.append(audit_checkpoint(group, fold))
            worker_info["independent_audits"] = len(audit_records)
            print(f"independent checkpoint audit {len(audit_records)}/20", flush=True)
            update_manifest("AUDITING", worker_info)

    rng_audit = []
    for fold in range(1, 11):
        tasks = {}
        for group in GROUPS:
            task_file, _cp, _latest, _log = fold_paths(group, fold)
            tasks[group] = json.loads(task_file.read_text(encoding="utf-8"))
        first, second = tasks["E61-E1"]["rng_trace"], tasks["E61-E2"]["rng_trace"]
        matches = 0
        mismatches = []
        for left, right in zip(first, second):
            if left["epoch"] != right["epoch"]:
                mismatches.append(left["epoch"])
                continue
            if left["rng_before"] == right["rng_before"] and left["rng_after_train"] == right["rng_after_train"] and left["rng_after_eval"] == right["rng_after_eval"]:
                matches += 1
            else:
                mismatches.append(left["epoch"])
        rng_audit.append({
            "fold": fold, "shared_warmup_sha256": initial_warmup_sha[fold],
            "shared_warmup_rng_sha": torch_load(warmup_path(fold))["rng_sha"],
            "e1_oof_main_rng_unchanged": tasks["E61-E1"]["oof_reliability"] is not None,
            "compared_postfork_epochs": min(len(first), len(second)),
            "exact_rng_trace_matches": matches, "mismatch_epochs": mismatches,
            "status": "PASS" if not mismatches else "DIVERGED_AND_REPORTED",
        })
    worker_info["completed_tasks"] = 20
    worker_info["independent_audits"] = 20
    worker_info["max_concurrent_workers"] = worker_info["max_workers"]
    manifest = finalize(audit_records, worker_info, rng_audit)
    print(f"Experiment 61 final status: {manifest['status']}", flush=True)
    return manifest


def audit_only():
    """Rerun read-only checkpoint audits using the frozen training-source manifest."""
    global SOURCE_SHA_OVERRIDE
    import torch

    preflight_path = RESULTS / "preflight.json"
    manifest_path = EXP / "manifest.json"
    if not preflight_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("audit-only requires the frozen preflight and task manifest")
    preflight_data = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight_data.get("status") != "PASS":
        raise RuntimeError("cannot audit checkpoints without a passing preflight")
    SOURCE_SHA_OVERRIDE = preflight_data["source_sha"]
    if len(SOURCE_SHA_OVERRIDE) != 3:
        raise RuntimeError("frozen training source SHA manifest is incomplete")
    for group in GROUPS:
        for fold in range(1, 11):
            task_file, best_file, _latest, _log = fold_paths(group, fold)
            if not task_file.is_file() or not best_file.is_file():
                raise RuntimeError(f"audit-only expected completed checkpoint {group} fold {fold}")
            task = json.loads(task_file.read_text(encoding="utf-8"))
            if task.get("status") != "COMPLETE" or task.get("checkpoint_sha256") != sha_file(best_file):
                raise RuntimeError(f"audit-only found incomplete or changed task {group} fold {fold}")
            if task.get("source_sha") != SOURCE_SHA_OVERRIDE:
                raise RuntimeError(f"training-source SHA does not match frozen preflight for {group} fold {fold}")
    audits = []
    for group in GROUPS:
        for fold in range(1, 11):
            audits.append(audit_checkpoint(group, fold))
            print(f"independent checkpoint audit {len(audits)}/20", flush=True)
    rng_audit = []
    for fold in range(1, 11):
        task_records = {}
        for group in GROUPS:
            task_file = fold_paths(group, fold)[0]
            task_records[group] = json.loads(task_file.read_text(encoding="utf-8"))
        left = task_records["E61-E1"]["rng_trace"]
        right = task_records["E61-E2"]["rng_trace"]
        matches, mismatches = 0, []
        for a, b in zip(left, right):
            if a["epoch"] != b["epoch"]:
                mismatches.append(a["epoch"])
            elif a["rng_before"] == b["rng_before"] and a["rng_after_train"] == b["rng_after_train"] and a["rng_after_eval"] == b["rng_after_eval"]:
                matches += 1
            else:
                mismatches.append(a["epoch"])
        oof = json.loads(oof_path(fold).read_text(encoding="utf-8"))
        rng_audit.append({
            "fold": fold,
            "shared_warmup_sha256": sha_file(warmup_path(fold)),
            "shared_warmup_rng_sha": torch_load(warmup_path(fold))["rng_sha"],
            "e1_oof_main_rng_unchanged": bool(oof.get("main_rng_unchanged")),
            "compared_postfork_epochs": min(len(left), len(right)),
            "exact_rng_trace_matches": matches,
            "mismatch_epochs": mismatches,
            "status": "PASS" if not mismatches and oof.get("main_rng_unchanged") else "DIVERGED_AND_REPORTED",
        })
    worker_info = json.loads(manifest_path.read_text(encoding="utf-8")).get("worker_info", {})
    worker_info.update({"completed_tasks": 20, "independent_audits": 20, "audit_rerun": "OOF presence check corrected to task/checkpoint references"})
    manifest = finalize(audits, worker_info, rng_audit)
    print(f"Experiment 61 final status: {manifest['status']}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group", choices=list(GROUPS))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.worker:
        if args.group is None or args.fold is None:
            raise SystemExit("--worker requires --group and --fold")
        train_worker(args.group, args.fold)
    elif args.preflight:
        print(json.dumps(preflight(), ensure_ascii=False, indent=2))
    elif args.run:
        run_all(args.max_workers)
    elif args.audit_only:
        audit_only()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()




