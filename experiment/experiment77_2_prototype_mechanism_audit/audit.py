#!/usr/bin/env python3
"""Read-only, frozen-checkpoint mechanism audit for Experiment 77.2."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binomtest, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
OUT = HERE / "results"
E771 = Path(CFG["inputs"]["e771"])
E74 = Path(CFG["inputs"]["e74"])
sys.path.insert(0, str(E771))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


E77_AUDIT = load_module("e771_audit_readonly", E771 / "audit.py")
E77_RUN = E77_AUDIT.runner
E77_MODEL = load_module("e771_model_readonly", E771 / "model.py")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_tensors(state: dict) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        if torch.is_tensor(value):
            h.update(str(value.dtype).encode())
            h.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
        else:
            h.update(repr(value).encode())
    return h.hexdigest()


def pack_sha(runtime) -> str:
    return runtime.rng_sha(runtime.pack_rng())


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x / float(CFG["relu_diagnostics"]["gate_temperature"]))


def evidence_components(module, h: torch.Tensor, ids: torch.Tensor) -> dict:
    """Mirror compute_evidence without touching usage/query counters."""
    n = h.shape[0]
    features = h.reshape(n, 4, 36)
    semantics = module._semantic_relations(module.class_centers, (0, 1), 1)
    typical_feature = h.new_zeros((n, 2))
    typical_semantic = h.new_zeros((n, 2))
    scale = (0.15, 0.10, 0.10, 1.0)
    for modality in range(4):
        for cls in (0, 1):
            centers, valid = module._leave_one_out_centers(ids, modality=modality, disease_class=cls)
            plain, _, _ = module._retrieve(features[:, modality], centers, valid)
            semantic, _, _ = module._retrieve(features[:, modality], centers, valid,
                                               semantics[modality, cls].unsqueeze(0).expand(n, -1))
            typical_feature[:, cls] += plain * scale[modality]
            typical_semantic[:, cls] += semantic * scale[modality]

    hard_order = (1, 0)
    hard_sem = module._semantic_relations(module.hard_centers.permute(1, 0, 2, 3), hard_order, 1)
    contrast = h.new_zeros((n, 2, 4))
    valid_hard = torch.zeros((n, 2, 4), dtype=torch.bool, device=h.device)
    similarities = h.new_zeros((n, 2, 4))
    values = h.new_zeros((n, 2, 4))
    for modality in range(4):
        for direction in (0, 1):
            centers, valid = module._leave_one_out_centers(ids, hard_direction=direction, modality=modality)
            sim, _, _ = module._retrieve(features[:, modality], centers, valid,
                                          hard_sem[modality, direction].unsqueeze(0).expand(n, -1))
            target = 1 if direction == 0 else 0
            competing = 1 - target
            x = sim - typical_semantic[:, competing] / 1.35 - float(module.cfg["hard_contrast_margin"])
            active = valid.any(-1)
            gate = _sigmoid(x) * active.float()
            value = 0.5 * torch.tanh(2.0 * F.relu(x)) * gate
            contrast[:, direction, modality] = x
            valid_hard[:, direction, modality] = active
            similarities[:, direction, modality] = sim
            values[:, direction, modality] = value * scale[modality]
    raw_hard = h.new_zeros((n, 2))
    raw_hard[:, 1] = values[:, 0].sum(-1)
    raw_hard[:, 0] = values[:, 1].sum(-1)
    cap = float(CFG["relu_diagnostics"]["hard_evidence_cap"])
    clipped = raw_hard > cap
    hard = raw_hard.clamp(0, cap)
    return {
        "typical_feature": typical_feature,
        "typical_semantic": typical_semantic,
        "contrast": contrast,
        "valid": valid_hard,
        "hard_similarity": similarities,
        "hard_value_by_direction_modality": values,
        "hard_raw_by_class": raw_hard,
        "hard_by_class": hard,
        "hard_clipped_by_class": clipped,
    }


def protected_inputs() -> dict[str, str]:
    paths = [
        E771 / "config.json", E771 / "run.py", E771 / "model.py", E771 / "memory.py", E771 / "audit.py",
        E771 / "manifest.json", E771 / "results/report.md", E771 / "results/fold_metrics.csv",
        E771 / "results/interventions.csv", E771 / "results/memory_audit.json", E771 / "results/paired_comparisons.csv",
        E74 / "config.json", E74 / "run.py", E74 / "soft_positive_dcr.py", E74 / "audit.py",
        E74 / "manifest.json", E74 / "results/fold_metrics.csv",
        Path("/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv"),
        Path("/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy"),
    ]
    paths.extend(E771 / "checkpoints" / f"E77.1-G2_f{fold:02d}.best.pt" for fold in range(1, 11))
    paths.extend(E77_RUN.E74.best_path(E77_RUN.BASE_GROUP, fold) for fold in range(1, 11))
    result = {}
    for path in paths:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path)] = sha_file(path)
    return result


def _read_e77_fold(fold: int, preflight: dict) -> dict:
    group = CFG["inputs"]["group"]
    checkpoint, state, module, tracker, runtime, _old, checkpoint_sha, checkpoint_path = E77_AUDIT._load_checkpoint(
        group, fold, preflight
    )
    allowed = E77_RUN.allowed_train_ids(state)
    test = np.asarray(state["test_np"], dtype=np.int64)
    train = np.asarray(allowed, dtype=np.int64)
    overlap = sorted(set(np.asarray(state["train_np"], dtype=np.int64)) & set(test))
    expected = [376, 406] if fold == 2 else []
    if overlap != expected:
        raise RuntimeError(f"fold {fold} base overlap differs from locked E77.1 protocol: {overlap}")
    if np.intersect1d(train, test).size:
        raise RuntimeError(f"fold {fold} explicit test indices remain in allowed memory/probe train IDs")
    if np.isin(module.hard_subject_ids[module.hard_subject_ids >= 0].cpu().numpy(), test).any():
        raise RuntimeError(f"fold {fold} test ID found in frozen hard memory")

    module.eval(); state["model"].eval(); state["cam"].eval()
    query_ids = torch.arange(871, dtype=torch.long, device=state["x"].device)
    counters = {k: getattr(module, k).detach().clone() for k in
                ("class_usage", "class_queries", "hard_usage", "hard_queries")}
    memory_sha_before = digest_tensors(module.state_dict())
    model_sha_before = digest_tensors(state["model"].state_dict())
    cam_sha_before = digest_tensors(state["cam"].state_dict())
    rng_before = pack_sha(runtime)
    capture = {}
    best_local_epoch = int(checkpoint["best_epoch_1based"]) - 1 - int(E77_RUN.CFG["base"]["start_epoch_zero_based"])
    with torch.inference_mode():
        output, hook = E77_MODEL.forward_sc_perm(E77_RUN.E74, state, module, group,
                                                  training=False, local_epoch=best_local_epoch,
                                                  capture=capture)
    for key, value in counters.items():
        getattr(module, key).copy_(value)
    rng_after = pack_sha(runtime)
    if rng_before != rng_after:
        raise RuntimeError(f"fold {fold} frozen E77.1 eval advanced RNG")
    if digest_tensors(module.state_dict()) != memory_sha_before:
        raise RuntimeError(f"fold {fold} frozen E77.1 eval changed prototype/error memory")
    if digest_tensors(state["model"].state_dict()) != model_sha_before or digest_tensors(state["cam"].state_dict()) != cam_sha_before:
        raise RuntimeError(f"fold {fold} frozen E77.1 eval changed DARG/CAM weights")
    if hook["pre_calls"] != 1 or hook["post_calls"] != 1:
        raise RuntimeError(f"fold {fold} expected one CAM output hook call")
    h = capture["cam_features"]
    parts = evidence_components(module, h, query_ids)
    if not torch.isfinite(h).all() or not all(torch.isfinite(v).all() for v in parts.values() if torch.is_tensor(v)):
        raise FloatingPointError(f"fold {fold} non-finite frozen features/evidence")

    base_path = E77_RUN.E74.best_path(E77_RUN.BASE_GROUP, fold)
    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    if base_ckpt.get("group") != E77_RUN.BASE_GROUP or base_ckpt.get("fold") != fold:
        raise RuntimeError(f"fold {fold} E74.4 checkpoint identity mismatch")
    base_state, _warmup, base_identity = E77_RUN.E74.prepare(fold)
    if base_identity != checkpoint["base_identity"]:
        raise RuntimeError(f"fold {fold} E74.4/E77.1 fold identity mismatch")
    E77_RUN.E74.load_history().load_bundle(base_state, base_ckpt["best_state"])
    base_state["fold"] = fold; base_state["group"] = E77_RUN.BASE_GROUP
    base_state["model"].eval(); base_state["cam"].eval()
    base_runtime = E77_RUN.E74.load_history().load_e69().load_e61().load_base()
    base_rng_before = pack_sha(base_runtime)
    with torch.inference_mode():
        base_output = E77_RUN.E74.forward(base_state, E77_RUN.BASE_GROUP, training=False, collect=False)
    base_rng_after = pack_sha(base_runtime)
    if base_rng_before != base_rng_after:
        raise RuntimeError(f"fold {fold} frozen E74.4 eval advanced RNG")

    e77_logp = output["logp"].detach().float().cpu().numpy()
    base_logp = base_output["logp"].detach().float().cpu().numpy()
    raw_logp = capture["base_logp"].detach().float().cpu().numpy()
    h_np = h.detach().float().cpu().numpy()
    corr = state["parts"][3].detach().float().cpu().numpy()
    labels = np.asarray(state["labels_np"], dtype=np.int64)
    base_metrics = base_runtime.classification_metrics(base_output["logp"], test, state["y"])
    e77_metrics = base_runtime.classification_metrics(output["logp"], test, state["y"])
    # Retain no mutable model state; only these derived CPU arrays enter the analysis.
    mem = {k: getattr(module, k).detach().cpu().numpy().copy() for k in (
        "hard_subject_ids", "hard_first_error", "hard_last_error", "hard_frequency", "hard_votes",
        "hard_confidence", "hard_novelty", "hard_assignment", "hard_counts", "hard_centers",
        "class_centers", "class_counts", "class_assignment", "class_valid", "class_labels",
        "class_usage", "hard_usage")}
    tracker_np = {k: v.detach().cpu().numpy().copy() for k, v in checkpoint["tracker_state"].items()
                  if torch.is_tensor(v)}
    history = checkpoint["hard_diagnostics"]
    ys = labels
    typ_feature = parts["typical_feature"].detach().cpu().numpy()
    typ_sem = parts["typical_semantic"].detach().cpu().numpy()
    hard_by_class = parts["hard_by_class"].detach().cpu().numpy()
    soft_parts = softplus_counterfactual(module, parts)
    arrays = {
        "fold": fold,
        "corr256": corr, "cam144": h_np, "cam_corr36": h_np[:, 108:144],
        "e74_logp": base_logp, "e771_raw_logp": raw_logp, "e771_logp": e77_logp,
        "typical_feature_diff": typ_feature[:, 0] - typ_feature[:, 1],
        "typical_semantic_diff": typ_sem[:, 0] - typ_sem[:, 1],
        "hard_diff": hard_by_class[:, 0] - hard_by_class[:, 1],
        "hard_contrast": parts["contrast"].detach().cpu().numpy(),
        "hard_valid": parts["valid"].detach().cpu().numpy(),
        "hard_values": parts["hard_value_by_direction_modality"].detach().cpu().numpy(),
        "hard_raw_by_class": parts["hard_raw_by_class"].detach().cpu().numpy(),
        "hard_clipped": parts["hard_clipped_by_class"].detach().cpu().numpy(),
        "softplus_hard_by_class": soft_parts["hard_by_class"].detach().cpu().numpy(),
        "softplus_hard_values": soft_parts["hard_value_by_direction_modality"].detach().cpu().numpy(),
        "labels": ys, "train_ids": train, "test_ids": test,
        "memory": mem, "tracker": tracker_np, "history": history,
        "checkpoint_sha256": checkpoint_sha, "checkpoint_path": str(checkpoint_path),
        "base_checkpoint_sha256": sha_file(base_path), "base_checkpoint_path": str(base_path),
        "best_epoch_1based": int(checkpoint["best_epoch_1based"]),
        "memory_update_epoch": int(checkpoint["memory_update_epoch"]),
        "rng_unchanged": rng_before == rng_after and base_rng_before == base_rng_after,
        "hook_calls": {"pre": hook["pre_calls"], "post": hook["post_calls"]},
        "base_metrics": {k: float(base_metrics[k]) for k in ("acc", "hard_auc", "prob_auc", "sen", "spe")},
        "e771_metrics": {k: float(e77_metrics[k]) for k in ("acc", "hard_auc", "prob_auc", "sen", "spe")},
        "module_constants": {"lambda": float(capture["evidence"]["lambda"]),
                             "semantic_beta": float(capture["evidence"]["semantic_beta"])},
    }
    del output, base_output, base_state, state, module, tracker, checkpoint, base_ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return arrays


def softplus_counterfactual(module, parts: dict) -> dict:
    beta = float(CFG["relu_diagnostics"]["softplus_counterfactual_beta"])
    contrast = parts["contrast"]
    valid = parts["valid"].float()
    scale = (0.15, 0.10, 0.10, 1.0)
    soft = 0.5 * torch.tanh(2.0 * F.softplus(contrast, beta=beta)) * _sigmoid(contrast) * valid
    soft = soft * contrast.new_tensor(scale).view(1, 1, 4)
    n = contrast.shape[0]
    raw = contrast.new_zeros((n, 2))
    raw[:, 1] = soft[:, 0].sum(-1)
    raw[:, 0] = soft[:, 1].sum(-1)
    cap = float(CFG["relu_diagnostics"]["hard_evidence_cap"])
    return {"hard_by_class": raw.clamp(0, cap), "hard_value_by_direction_modality": soft}


def metrics_from_score(y: np.ndarray, score_asd: np.ndarray, threshold: float | None = None) -> dict:
    y_asd = (y == 0).astype(np.int64)
    score_asd = np.asarray(score_asd, dtype=np.float64)
    auc = float(roc_auc_score(y_asd, score_asd)) if np.unique(y_asd).size == 2 else float("nan")
    result = {"roc_auc": auc, "n_asd": int((y == 0).sum()), "n_td": int((y == 1).sum()),
              "mean_score_asd": float(score_asd[y == 0].mean()) if np.any(y == 0) else 0.0,
              "mean_score_td": float(score_asd[y == 1].mean()) if np.any(y == 1) else 0.0}
    if threshold is not None:
        pred = np.where(score_asd >= threshold, 0, 1)
        sen = float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
        spe = float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
        result.update({"acc": float((pred == y).mean()), "ba": (sen + spe) / 2, "sen": sen, "spe": spe})
    return result


def probabilities_and_predictions(logp: np.ndarray, y: np.ndarray) -> dict:
    p_asd = np.exp(logp[:, 0])
    pred = logp.argmax(1)
    return {
        "acc": float(accuracy_score(y, pred)),
        "ba": float(((((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)) +
                     (((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))) / 2),
        "prob_auc": float(roc_auc_score((y == 0).astype(int), p_asd)),
        "sen": float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "spe": float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)),
        "log_loss": float(log_loss((y == 0).astype(int), np.column_stack([p_asd, 1-p_asd]), labels=[0,1])),
        "brier": float(brier_score_loss((y == 0).astype(int), p_asd)),
        "pred": pred,
    }


def _fit_probe(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, yte: np.ndarray,
               setting: str, fold: int) -> dict:
    if not np.isfinite(xtr).all() or not np.isfinite(xte).all():
        raise FloatingPointError(f"non-finite probe input {setting} fold {fold}")
    scaler = StandardScaler()
    ztr = scaler.fit_transform(xtr)
    zte = scaler.transform(xte)
    clf = LogisticRegression(C=float(CFG["probe"]["C"]), solver=CFG["probe"]["solver"],
                             max_iter=int(CFG["probe"]["max_iter"]),
                             random_state=int(CFG["probe"]["random_state"]))
    clf.fit(ztr, ytr)
    p = clf.predict_proba(zte)
    idx_asd = int(np.where(clf.classes_ == 0)[0][0])
    probs = np.column_stack([p[:, idx_asd], 1 - p[:, idx_asd]])
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    score = probabilities_and_predictions(logp, yte)
    score.update({"group": "PIM-Audit-probe", "fold": fold, "setting": setting,
                  "n_train": int(len(ytr)), "n_test": int(len(yte)),
                  "converged": bool(np.max(clf.n_iter_) < int(CFG["probe"]["max_iter"]))})
    score.pop("pred")
    return score


def _current_hard_records(d: dict, fold: int) -> dict:
    memory = d["memory"]
    labels = d["labels"]
    train_set, test_set = set(d["train_ids"].tolist()), set(d["test_ids"].tolist())
    result = {}
    for direction, name, expected_label in ((0, "FP", 1), (1, "FN", 0)):
        ids = memory["hard_subject_ids"][direction]
        records = []
        for slot, subject in enumerate(ids.tolist()):
            if subject < 0:
                continue
            if int(labels[subject]) != expected_label:
                raise RuntimeError(f"fold {fold} {name} memory contains wrong true class for ID {subject}")
            if subject not in train_set or subject in test_set:
                raise RuntimeError(f"fold {fold} {name} memory contains non-allowed train ID {subject}")
            records.append({
                "id": int(subject), "slot": int(slot), "first_stable_epoch": int(memory["hard_first_error"][direction, slot]),
                "last_stable_epoch": int(memory["hard_last_error"][direction, slot]),
                "stable_update_frequency": int(memory["hard_frequency"][direction, slot]),
                "current_window_votes": int(memory["hard_votes"][direction, slot]),
                "prediction_confidence": float(memory["hard_confidence"][direction, slot]),
                "novelty": float(memory["hard_novelty"][direction, slot]),
                "prototype_assignment_by_modality": memory["hard_assignment"][direction, :, slot].astype(int).tolist(),
                "age_since_last_stable_error": int(d["memory_update_epoch"] - memory["hard_last_error"][direction, slot]),
            })
        result[name] = records
    return result


def _memory_diagnostics(d: dict, fold: int, predictions: np.ndarray) -> dict:
    memory = d["memory"]
    records = _current_hard_records(d, fold)
    labels = d["labels"]
    rows = {}
    for name, direction in (("FP", 0), ("FN", 1)):
        rr = records[name]
        ids = np.asarray([r["id"] for r in rr], dtype=np.int64)
        if ids.size:
            spans = np.asarray([r["last_stable_epoch"] - r["first_stable_epoch"] for r in rr])
            freq = np.asarray([r["stable_update_frequency"] for r in rr])
            wrong_pred = (predictions[ids] != labels[ids])
            current_correct = int((~wrong_pred).sum())
            current_wrong = int(wrong_pred.sum())
        else:
            spans = np.zeros(0); freq = np.zeros(0); current_correct = 0; current_wrong = 0
        assignment = memory["hard_assignment"][direction]
        center = memory["hard_centers"][direction]
        counts = memory["hard_counts"][direction]
        modal = []
        for m in range(4):
            active = np.flatnonzero(counts[m] > 0)
            sims = []
            if len(active) > 1:
                unit = center[m, active] / np.maximum(np.linalg.norm(center[m, active], axis=-1, keepdims=True), 1e-12)
                matrix = unit @ unit.T
                sims = matrix[np.triu_indices(len(active), 1)].tolist()
            cover = []
            if ids.size:
                assignments = assignment[m]
                for proto in active:
                    slots = np.flatnonzero(assignments == proto)
                    cover.append({"prototype": int(proto), "patient_count": int(len(slots))})
            modal.append({"active_centers": int(len(active)), "center_pair_cosine_mean": float(np.mean(sims)) if sims else 0.0,
                          "center_pair_cosine_max": float(np.max(sims)) if sims else 0.0,
                          "cluster_patient_counts": cover})
        rows[name] = {
            "pool_size": int(len(ids)), "current_best_checkpoint_wrong": current_wrong,
            "current_best_checkpoint_corrected_since_stable_error": current_correct,
            "current_wrong_fraction": float(current_wrong / len(ids)) if len(ids) else 0.0,
            "first_to_last_stable_epoch_span": {
                "mean": float(spans.mean()) if len(spans) else 0.0,
                "median": float(np.median(spans)) if len(spans) else 0.0,
                "max": int(spans.max()) if len(spans) else 0,
                "long_interval_ge_one_third_run": int((spans >= max(1, d["memory_update_epoch"] / 3)).sum()) if len(spans) else 0,
                "short_interval_le_two_epochs": int((spans <= 2).sum()) if len(spans) else 0,
            },
            "stable_update_frequency": {
                "mean": float(freq.mean()) if len(freq) else 0.0,
                "median": float(np.median(freq)) if len(freq) else 0.0,
                "max": int(freq.max()) if len(freq) else 0,
                "frequency_ge_3": int((freq >= 3).sum()) if len(freq) else 0,
                "frequency_eq_1": int((freq == 1).sum()) if len(freq) else 0,
            },
            "votes": {"mean": float(np.mean([r["current_window_votes"] for r in rr])) if rr else 0.0,
                      "confidence_mean": float(np.mean([r["prediction_confidence"] for r in rr])) if rr else 0.0,
                      "novelty_mean": float(np.mean([r["novelty"] for r in rr])) if rr else 0.0,
                      "age_mean": float(np.mean([r["age_since_last_stable_error"] for r in rr])) if rr else 0.0},
            "prototype_diversity_by_modality": modal,
            "patients": rr,
        }
    return rows


def _cross_fold_memory(folds: list[dict]) -> dict:
    rng = np.random.default_rng(int(CFG["cross_fold_randomization"]["seed"]))
    output = {}
    for name, expected_label in (("FP", 1), ("FN", 0)):
        sets = {}
        candidates = {}
        fold_roles = {}
        for d in folds:
            f = d["fold"]
            rec = {r["id"] for r in d["memory_diagnostics"][name]["patients"]}
            sets[f] = rec
            allowed = d["train_ids"]
            labels = d["labels"]
            candidates[f] = np.asarray([i for i in allowed if labels[i] == expected_label], dtype=np.int64)
            for sid in rec:
                fold_roles.setdefault(sid, []).append(f)
        pair_rows = []
        pair_null = {}
        for f1 in range(1, 11):
            for f2 in range(f1 + 1, 11):
                a, b = sets[f1], sets[f2]
                union = a | b
                obs_j = len(a & b) / len(union) if union else 0.0
                obs_overlap = len(a & b)
                pair_rows.append({"fold_a": f1, "fold_b": f2, "memory_a": len(a), "memory_b": len(b),
                                  "intersection": obs_overlap, "jaccard": obs_j})
        reps = int(CFG["cross_fold_randomization"]["replicates"])
        null_jaccard = {(x["fold_a"], x["fold_b"]): [] for x in pair_rows}
        for _ in range(reps):
            random_sets = {}
            for f in range(1, 11):
                n = len(sets[f]); pool = candidates[f]
                if n > len(pool):
                    raise RuntimeError(f"matched null cannot sample memory size for {name} fold {f}")
                random_sets[f] = set(rng.choice(pool, size=n, replace=False).tolist()) if n else set()
            for row in pair_rows:
                key = (row["fold_a"], row["fold_b"])
                a, b = random_sets[key[0]], random_sets[key[1]]
                u = len(a | b)
                null_jaccard[key].append(len(a & b) / u if u else 0.0)
        for row in pair_rows:
            key = (row["fold_a"], row["fold_b"])
            null = np.asarray(null_jaccard[key], dtype=np.float64)
            row["matched_random_jaccard_mean"] = float(null.mean())
            row["matched_random_jaccard_sd"] = float(null.std(ddof=0))
            row["standardized_recurrence"] = float((row["jaccard"] - null.mean()) / max(null.std(ddof=0), 1e-12))
            row["empirical_upper_tail_p"] = float((1 + np.sum(null >= row["jaccard"])) / (reps + 1))
        actual = np.asarray([r["jaccard"] for r in pair_rows])
        null_means = np.asarray([r["matched_random_jaccard_mean"] for r in pair_rows])
        direction_count = {}
        for sid, member_folds in fold_roles.items():
            direction_count[sid] = len(member_folds)
        output[name] = {
            "memory_patient_count_fold_sum_not_global_unique": int(sum(len(v) for v in sets.values())),
            "unique_global_memory_patient_ids": int(len(fold_roles)),
            "patient_repeated_in_memory_multiple_folds": int(sum(n > 1 for n in direction_count.values())),
            "memory_fold_count_distribution": {str(k): int(sum(n == k for n in direction_count.values())) for k in range(1, 11)},
            "observed_pairwise_jaccard_mean": float(actual.mean()),
            "matched_random_pairwise_jaccard_mean": float(null_means.mean()),
            "observed_over_random_ratio": float(actual.mean() / max(null_means.mean(), 1e-12)),
            "pairs_with_jaccard_above_random_mean": int(sum(r["jaccard"] > r["matched_random_jaccard_mean"] for r in pair_rows)),
            "pairs_empirical_p_below_0p05": int(sum(r["empirical_upper_tail_p"] < 0.05 for r in pair_rows)),
            "pairs": pair_rows,
        }
    return output


def _correctness_summaries(d: dict) -> dict:
    result = {}
    labels = d["labels"]
    ids_by_subset = {"all_test": d["test_ids"], "allowed_train": d["train_ids"]}
    for subset, ids in ids_by_subset.items():
        y = labels[ids]
        full = d["e771_logp"][ids]
        nohard = d["hard_off_logp"][ids]
        base = d["e74_logp"][ids]
        pfull, poff, pbase = full.argmax(1), nohard.argmax(1), base.argmax(1)
        ymargin_full = np.where(y == 0, full[:, 0] - full[:, 1], full[:, 1] - full[:, 0])
        ymargin_off = np.where(y == 0, nohard[:, 0] - nohard[:, 1], nohard[:, 1] - nohard[:, 0])
        correct_full, correct_off = pfull == y, poff == y
        result[subset] = {
            "n": int(len(ids)), "full_accuracy": float(correct_full.mean()),
            "hard_off_accuracy": float(correct_off.mean()), "base_e74_accuracy": float((pbase == y).mean()),
            "hard_evidence_wrong_to_correct": int((~correct_off & correct_full).sum()),
            "hard_evidence_correct_to_wrong": int((correct_off & ~correct_full).sum()),
            "margin_improved_same_prediction": int(((pfull == poff) & (ymargin_full > ymargin_off)).sum()),
            "margin_worsened_same_prediction": int(((pfull == poff) & (ymargin_full < ymargin_off)).sum()),
            "mean_true_margin_delta_full_minus_hard_off": float((ymargin_full - ymargin_off).mean()),
            "median_true_margin_delta_full_minus_hard_off": float(np.median(ymargin_full - ymargin_off)),
        }
    for direction, name, true_label in ((0, "FP", 1), (1, "FN", 0)):
        ids = np.asarray([r["id"] for r in d["memory_diagnostics"][name]["patients"]], dtype=np.int64)
        if ids.size:
            y = labels[ids]
            full = d["e771_logp"][ids]; nohard = d["hard_off_logp"][ids]
            mf = np.where(y == 0, full[:,0]-full[:,1], full[:,1]-full[:,0])
            mo = np.where(y == 0, nohard[:,0]-nohard[:,1], nohard[:,1]-nohard[:,0])
            correct_full, correct_off = full.argmax(1) == y, nohard.argmax(1) == y
            result[name + "_memory_patients"] = {
                "n": int(len(ids)), "corrected_wrong_to_right": int((~correct_off & correct_full).sum()),
                "correct_to_wrong": int((correct_off & ~correct_full).sum()),
                "mean_true_margin_delta": float((mf-mo).mean()),
                "margin_improved": int((mf > mo).sum()), "margin_worsened": int((mf < mo).sum()),
                "directional_prediction_rate": float((full.argmax(1) == true_label).mean()),
            }
        else:
            result[name + "_memory_patients"] = {"n": 0}
    return result


def _probe_and_score(folds: list[dict]) -> tuple[list[dict], list[dict], dict]:
    probe_rows, zero_fit, fold_uncertainty = [], [], []
    for d in folds:
        f = d["fold"]; train = d["train_ids"]; test = d["test_ids"]; labels = d["labels"]
        ytr, yte = labels[train], labels[test]
        p0 = np.exp(d["e74_logp"][:, 0])
        p0_train_score = d["e74_logp"][train, 0] - d["e74_logp"][train, 1]
        e74_score = d["e74_logp"][test, 0] - d["e74_logp"][test, 1]
        raw_margin = d["e771_raw_logp"][:, 0] - d["e771_raw_logp"][:, 1]
        final_margin = d["e771_logp"][:, 0] - d["e771_logp"][:, 1]
        zero_scores = {
            "E74.4_frozen_margin": e74_score,
            "E77.1_pre_correction_margin": raw_margin[test],
            "E77.1_full_final_margin": final_margin[test],
            "typical_feature_only_difference": d["typical_feature_diff"][test],
            "typical_semantic_conditioned_difference": d["typical_semantic_diff"][test],
            "hard_FP_FN_evidence_difference": d["hard_diff"][test],
            "typical_plus_hard_evidence_difference": d["typical_semantic_diff"][test] + d["hard_diff"][test],
        }
        row = {"fold": f}
        for name, score in zero_scores.items():
            stats = metrics_from_score(labels[test], score)
            row[name + "_roc_auc"] = stats["roc_auc"]
            row[name + "_mean_ASD"] = stats["mean_score_asd"]
            row[name + "_mean_TD"] = stats["mean_score_td"]
        row["e74_base_acc"] = probabilities_and_predictions(d["e74_logp"][test], labels[test])["acc"]
        row["e771_raw_acc"] = probabilities_and_predictions(d["e771_raw_logp"][test], labels[test])["acc"]
        row["e771_final_acc"] = probabilities_and_predictions(d["e771_logp"][test], labels[test])["acc"]
        zero_fit.append(row)

        evidence = np.column_stack([d["typical_feature_diff"], d["hard_diff"], d["typical_semantic_diff"]])
        settings = {
            "P0_CORR256": (d["corr256"][train], d["corr256"][test]),
            "P1_CORR256_plus_evidence": (np.column_stack([d["corr256"][train], evidence[train]]),
                                         np.column_stack([d["corr256"][test], evidence[test]])),
            "P2_CAM144": (d["cam144"][train], d["cam144"][test]),
            "P3_CAM144_plus_evidence": (np.column_stack([d["cam144"][train], evidence[train]]),
                                        np.column_stack([d["cam144"][test], evidence[test]])),
            "P4_margin_only": (raw_margin[train, None], raw_margin[test, None]),
            "P4_margin_plus_evidence": (np.column_stack([raw_margin[train], evidence[train]]),
                                        np.column_stack([raw_margin[test], evidence[test]])),
        }
        for setting, (xtr, xte) in settings.items():
            probe_rows.append(_fit_probe(xtr, ytr, xte, yte, setting, f))

        # Predeclared uncertainty bins use only training margins to define boundaries.
        cuts = np.quantile(np.abs(p0_train_score), np.linspace(0, 1, 6))
        cuts[0] -= 1e-12; cuts[-1] += 1e-12
        test_abs = np.abs(e74_score)
        bins = np.digitize(test_abs, cuts[1:-1], right=False)
        for b in range(5):
            mask = bins == b
            yy = labels[test][mask]
            score = d["typical_semantic_diff"][test][mask] + d["hard_diff"][test][mask]
            if mask.sum():
                auc = float(roc_auc_score((yy == 0).astype(int), score)) if np.unique(yy).size == 2 else float("nan")
                fold_uncertainty.append({"fold": f, "bin": b+1, "n": int(mask.sum()),
                    "base_abs_margin_lower_train_quantile": float(cuts[b]),
                    "base_abs_margin_upper_train_quantile": float(cuts[b+1]),
                    "mean_abs_base_margin": float(test_abs[mask].mean()),
                    "mean_prototype_evidence_diff": float(score.mean()),
                    "ASD_fraction": float((yy == 0).mean()), "prototype_auc_when_both_classes": auc})

    # Paired fold comparisons are descriptive; folds share patients and are not independent replicates.
    grouped = {}
    for setting in sorted({r["setting"] for r in probe_rows}):
        grouped[setting] = {int(r["fold"]): r for r in probe_rows if r["setting"] == setting}
    comparisons = []
    for a, b in (("P0_CORR256", "P1_CORR256_plus_evidence"), ("P2_CAM144", "P3_CAM144_plus_evidence"),
                 ("P4_margin_only", "P4_margin_plus_evidence")):
        for metric in ("ba", "prob_auc", "log_loss", "brier"):
            delta = np.asarray([grouped[b][f][metric] - grouped[a][f][metric] for f in range(1, 11)])
            better = int((delta > 1e-10).sum()) if metric not in ("log_loss", "brier") else int((delta < -1e-10).sum())
            worse = int((delta < -1e-10).sum()) if metric not in ("log_loss", "brier") else int((delta > 1e-10).sum())
            p_sign = float(binomtest(better, better + worse, 0.5, alternative="two-sided").pvalue) if better + worse else 1.0
            comparisons.append({"comparison": f"{b}_minus_{a}", "metric": metric,
                "mean_delta": float(delta.mean()), "sd_delta_population": float(delta.std(ddof=0)),
                "wins_ties_losses_for_augmented": [better, int((np.abs(delta) <= 1e-10).sum()), worse],
                "descriptive_two_sided_sign_test_p": p_sign})
    return probe_rows, zero_fit, {"uncertainty_bins": fold_uncertainty, "paired_probe_comparisons": comparisons}


def _relu_diagnostics(d: dict) -> dict:
    eps = float(CFG["relu_diagnostics"]["near_zero_epsilon"])
    grad_eps = float(CFG["relu_diagnostics"]["gradient_zero_epsilon"])
    x = d["hard_contrast"]
    valid = d["hard_valid"].astype(bool)
    values = d["hard_values"]
    raw_cls = d["hard_raw_by_class"]
    clipped = d["hard_clipped"].astype(bool)
    labels = d["labels"]
    output = {}
    for subset, ids in (("allowed_train", d["train_ids"]), ("test", d["test_ids"])):
        output[subset] = {}
        for direction, name, target, comp in ((0, "FP", 1, 0), (1, "FN", 0, 1)):
            idx = np.asarray(ids, dtype=np.int64)
            valid_s = valid[idx, direction]
            contrast_s = x[idx, direction]
            value_s = values[idx, direction]
            strata = {}
            for mode, mask in (("all_patients", np.ones(len(idx), dtype=bool)),
                               ("has_valid_hard_prototype", valid_s.any(-1)),
                               ("actually_retrieved_any_hard_prototype", valid_s.any(-1))):
                total = int(mask.sum())
                vm = valid_s[mask]
                xm = contrast_s[mask]
                valm = value_s[mask]
                valid_count = int(vm.sum())
                neg_count = int((vm & (xm <= 0)).sum())
                pos_count = int((vm & (xm > 0)).sum())
                zero_count = int((vm & (valm <= 0)).sum())
                positive_count = int((vm & (valm > 0)).sum())
                near_count = int((vm & (valm > 0) & (valm <= eps)).sum())
                strata[mode] = {"n_patients": total, "n_valid_patient_modality_queries": valid_count,
                    "contrast_le_zero_n": neg_count, "contrast_le_zero_fraction_valid": neg_count / max(valid_count, 1),
                    "contrast_gt_zero_n": pos_count, "contrast_gt_zero_fraction_valid": pos_count / max(valid_count, 1),
                    "value_zero_n": zero_count, "value_zero_fraction_valid": zero_count / max(valid_count, 1),
                    "value_positive_n": positive_count, "value_positive_fraction_valid": positive_count / max(valid_count, 1),
                    "near_zero_positive_n": near_count, "near_zero_positive_fraction_positive": near_count / max(positive_count, 1)}
            modality = []
            for m, label in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
                vm = valid_s[:, m]; xm = contrast_s[:, m]; vmul = value_s[:, m]
                modality.append({"modality": label, "valid_n": int(vm.sum()),
                    "contrast_le_zero_fraction_valid": float(((xm <= 0) & vm).sum()/max(int(vm.sum()),1)),
                    "zero_value_fraction_valid": float(((vmul <= 0) & vm).sum()/max(int(vm.sum()),1)),
                    "positive_value_fraction_valid": float(((vmul > 0) & vm).sum()/max(int(vm.sum()),1)),
                    "near_zero_positive_fraction_positive": float((((vmul > 0)&(vmul <= eps)&vm).sum())/max(int(((vmul > 0)&vm).sum()),1))})
            target_clipped = clipped[idx, target]
            rows = labels[idx] == (1 if name == "FP" else 0)
            # Every score is evaluated for both classes; this identifies the direction's target score cap.
            output[subset][name] = {"all_denominators": strata, "by_modality": modality,
                "target_class_hard_score_clipped_n": int(target_clipped.sum()),
                "target_class_hard_score_clipped_fraction": float(target_clipped.mean()),
                "true_direction_patient_n": int(rows.sum()),
                "true_direction_target_hard_evidence_zero_fraction": float((raw_cls[idx[rows], target] <= 0).mean()) if rows.any() else 0.0}

    # Exact local derivative of the registered ReLU/gate/tanh operator with respect to contrast.
    tx = torch.as_tensor(x, dtype=torch.float64)
    vm = torch.as_tensor(valid, dtype=torch.bool)
    gate = torch.sigmoid(tx / float(CFG["relu_diagnostics"]["gate_temperature"]))
    tanh = torch.tanh(2 * tx.clamp_min(0))
    derivative = (1 - tanh.square()) * gate + 0.5 * tanh * gate * (1 - gate) / float(CFG["relu_diagnostics"]["gate_temperature"])
    derivative = torch.where((tx > 0) & vm, derivative, torch.zeros_like(derivative))
    output["gradient"] = {}
    for subset, ids in (("allowed_train", d["train_ids"]), ("test", d["test_ids"])):
        ix = torch.as_tensor(ids, dtype=torch.long)
        der = derivative[ix]
        valid_subset = vm[ix]
        relu_zero = valid_subset & (tx[ix] <= 0)
        small = valid_subset & (tx[ix] > 0) & (der <= grad_eps)
        by_direction = {}
        for direction, name in ((0, "FP"), (1, "FN")):
            valid_d = valid_subset[:, direction]
            zero_d = valid_d & (tx[ix, direction] <= 0)
            small_d = valid_d & (tx[ix, direction] > 0) & (der[:, direction] <= grad_eps)
            by_direction[name] = {
                "valid_contrast_count": int(valid_d.sum().item()),
                "relu_zero_gradient_fraction_valid": float(zero_d.sum().item()/max(valid_d.sum().item(),1)),
                "positive_but_small_local_gradient_fraction_valid": float(small_d.sum().item()/max(valid_d.sum().item(),1)),
                "mean_local_derivative_on_valid": float(der[:, direction][valid_d].mean().item()) if valid_d.any() else 0.0,
            }
        output["gradient"][subset] = {
            "valid_contrast_count": int(valid_subset.sum().item()),
            "relu_zero_gradient_fraction_valid": float(relu_zero.sum().item()/max(valid_subset.sum().item(),1)),
            "positive_but_small_local_gradient_fraction_valid": float(small.sum().item()/max(valid_subset.sum().item(),1)),
            "mean_local_derivative_on_valid": float(der[valid_subset].mean().item()) if valid_subset.any() else 0.0,
            "by_direction": by_direction,
        }
    # Analyze the frozen checkpoint's stored current stable FP/FN records with the exact ranking objective.
    mem = d["memory"]; x_leaf = torch.as_tensor(x, dtype=torch.float64).clone().requires_grad_(True)
    scl = x_leaf.new_tensor((0.15,0.10,0.10,1.0)).view(1,1,4)
    vals = 0.5 * torch.tanh(2 * F.relu(x_leaf)) * torch.sigmoid(x_leaf/0.2) * torch.as_tensor(valid, dtype=torch.float64) * scl
    raw = x_leaf.new_zeros((871,2)); raw[:,1] = vals[:,0].sum(-1); raw[:,0] = vals[:,1].sum(-1)
    hard = raw.clamp(0.0, float(CFG["relu_diagnostics"]["hard_evidence_cap"]))
    fpids = np.asarray(mem["hard_subject_ids"][0]); fpids = fpids[fpids >= 0]
    fnids = np.asarray(mem["hard_subject_ids"][1]); fnids = fnids[fnids >= 0]
    terms=[]; individual=[]
    margin = float(E77_RUN.CFG["method"]["hard_ranking_margin"])
    if fpids.size:
        delta = hard[fpids,1]-hard[fpids,0]
        losses = F.relu(margin-delta)
        terms.append(losses.mean()); individual.extend(("FP",int(i),float(l.detach())) for i,l in zip(fpids,losses))
    if fnids.size:
        delta = hard[fnids,0]-hard[fnids,1]
        losses = F.relu(margin-delta)
        terms.append(losses.mean()); individual.extend(("FN",int(i),float(l.detach())) for i,l in zip(fnids,losses))
    if terms:
        rank_loss = torch.stack(terms).mean()
        grad = torch.autograd.grad(rank_loss, x_leaf)[0]
        grad_zero_subject = []
        for direction, sid, loss in individual:
            active_grad = grad[sid, 0 if direction == "FP" else 1]
            grad_zero_subject.append({"direction":direction,"id":sid,"loss":loss,
                                      "path_grad_l1":float(active_grad.abs().sum()),
                                      "zero_gradient":bool(active_grad.abs().max() <= grad_eps)})
        positive = [r for r in grad_zero_subject if r["loss"] > 0]
        output["gradient"]["ranking_loss"] = {
            "loss_value": float(rank_loss.detach()), "stable_memory_patient_count": len(individual),
            "patient_losses_positive": int(sum(r["loss"] > 0 for r in grad_zero_subject)),
            "positive_loss_patients_with_zero_contrast_gradient": int(sum(r["zero_gradient"] for r in positive)),
            "positive_loss_zero_gradient_fraction": float(sum(r["zero_gradient"] for r in positive)/max(len(positive),1)),
            "subject_details": grad_zero_subject,
        }
    else:
        output["gradient"]["ranking_loss"] = {"loss_value": 0.0, "stable_memory_patient_count": 0,
                                               "positive_loss_patients_with_zero_contrast_gradient": 0}
    return output


def _softplus_summary(d: dict) -> dict:
    out = {}
    for subset, ids in (("allowed_train", d["train_ids"]), ("test", d["test_ids"])):
        ids=np.asarray(ids,dtype=np.int64); y=d["labels"][ids]
        original=d["hard_raw_by_class"][ids].clip(0,0.5)
        soft=d["softplus_hard_by_class"][ids]
        valid=d["hard_valid"][ids]
        orig_nonzero=(original>1e-6).any(1); soft_nonzero=(soft>1e-6).any(1)
        full_margin=d["e771_logp"][ids,0]-d["e771_logp"][ids,1]
        base_margin=d["e771_raw_logp"][ids,0]-d["e771_raw_logp"][ids,1]
        typical=d["typical_semantic_diff"][ids]
        lam=d["module_constants"]["lambda"]
        orig_corr=lam*np.clip(typical+(original[:,0]-original[:,1]),-2,2)
        soft_corr=lam*np.clip(typical+(soft[:,0]-soft[:,1]),-2,2)
        margin_orig=base_margin+orig_corr
        margin_soft=base_margin+soft_corr
        true_margin_orig=np.where(y==0,margin_orig,-margin_orig)
        true_margin_soft=np.where(y==0,margin_soft,-margin_soft)
        pred_orig=np.where(margin_orig>=0,0,1); pred_soft=np.where(margin_soft>=0,0,1)
        correct=y==pred_orig; soft_correct=y==pred_soft
        out[subset]={
            "n":int(len(ids)),"valid_patient_fraction":float(valid.any((1,2)).mean()),
            "original_hard_evidence_active_patient_fraction":float(orig_nonzero.mean()),
            "softplus_hard_evidence_active_patient_fraction":float(soft_nonzero.mean()),
            "newly_covered_patient_count":int((~orig_nonzero&soft_nonzero).sum()),
            "original_mean_abs_evidence_margin":float(np.abs(original[:,0]-original[:,1]).mean()),
            "softplus_mean_abs_evidence_margin":float(np.abs(soft[:,0]-soft[:,1]).mean()),
            "mean_true_class_margin_original":float(true_margin_orig.mean()),
            "mean_true_class_margin_softplus":float(true_margin_soft.mean()),
            "mean_true_margin_delta_softplus_minus_original":float((true_margin_soft-true_margin_orig).mean()),
            "true_margin_improved_count":int((true_margin_soft>true_margin_orig).sum()),
            "true_margin_worsened_count":int((true_margin_soft<true_margin_orig).sum()),
            "original_correct_to_wrong":int((correct&~soft_correct).sum()),
            "original_wrong_to_correct":int((~correct&soft_correct).sum()),
            "accuracy_original":float(correct.mean()),"accuracy_softplus":float(soft_correct.mean()),
        }
    return out


def _write_report(folds: list[dict], probe_rows: list[dict], zero_fit: list[dict], extra: dict,
                  crossfold: dict, source_hashes: dict, sensitivity: dict) -> None:
    # report body is constructed from the saved structured details for a reproducible final handoff.
    h1 = extra["hard_memory_summary"]
    h2 = extra["incremental_summary"]
    h3 = extra["relu_summary"]
    means = extra["metric_means"]
    probe_means = {}
    for setting in sorted({r["setting"] for r in probe_rows}):
        rows = [r for r in probe_rows if r["setting"] == setting]
        probe_means[setting] = {metric: float(np.mean([r[metric] for r in rows]))
                                for metric in ("acc", "ba", "prob_auc", "sen", "spe", "log_loss", "brier")}
    probe_table = ["", "| Frozen probe | ACC | BA | ROC-AUC | SEN | SPE | Log loss | Brier |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for setting, label in (("P0_CORR256","P0 CORR256"),("P1_CORR256_plus_evidence","P1 CORR256 + evidence"),
                           ("P2_CAM144","P2 CAM144"),("P3_CAM144_plus_evidence","P3 CAM144 + evidence"),
                           ("P4_margin_only","P4 raw margin"),("P4_margin_plus_evidence","P4 margin + evidence")):
        m=probe_means[setting]
        probe_table.append(f"| {label} | {100*m['acc']:.3f} | {100*m['ba']:.3f} | {100*m['prob_auc']:.3f} | {100*m['sen']:.3f} | {100*m['spe']:.3f} | {m['log_loss']:.4f} | {m['brier']:.4f} |")
    zero_score_names=("E74.4_frozen_margin","E77.1_pre_correction_margin","E77.1_full_final_margin",
                      "typical_feature_only_difference","typical_semantic_conditioned_difference",
                      "hard_FP_FN_evidence_difference","typical_plus_hard_evidence_difference")
    zero_auc_table=["", "| Frozen zero-fit score | ROC-AUC mean ± SD (%) |", "|---|---:|"]
    for name in zero_score_names:
        vals=[float(r[name+"_roc_auc"]) for r in zero_fit]
        zero_auc_table.append(f"| {name} | {100*np.mean(vals):.3f} ± {100*np.std(vals,ddof=0):.3f} |")
    def q2_sensitivity(setting_a, setting_b, metric):
        a={r["fold"]:r for r in probe_rows if r["setting"]==setting_a and r["fold"]!=2}
        b={r["fold"]:r for r in probe_rows if r["setting"]==setting_b and r["fold"]!=2}
        return float(np.mean([b[f][metric]-a[f][metric] for f in sorted(a)]))
    fold2_probe_sensitivity=(q2_sensitivity("P0_CORR256","P1_CORR256_plus_evidence","ba"),
                             q2_sensitivity("P0_CORR256","P1_CORR256_plus_evidence","prob_auc"),
                             q2_sensitivity("P2_CAM144","P3_CAM144_plus_evidence","ba"),
                             q2_sensitivity("P2_CAM144","P3_CAM144_plus_evidence","prob_auc"))
    sens_lines=[]
    for name in ("e74","e771"):
        vals=sensitivity[name]
        sens_lines.append(f"{name.upper()} ACC/Hard-AUC/Prob-AUC/SEN/SPE = " + "/".join(f"{vals[k]['mean_percent']:.4f}±{vals[k]['population_sd_percent']:.4f}" for k in ("acc","hard_auc","prob_auc","sen","spe")) + "%")
    relu=h3["population"]
    grad=h3["gradient_population"]
    def hard_stat(subset, direction): return relu[subset][direction]
    def model_metric_line(name):
        m=means[name]
        keys=("acc","hard_auc","prob_auc","sen","spe")
        return "/".join(f"{m[k]:.4f}±{m['sd_'+k]:.4f}" for k in keys)
    lines = [
        "# Experiment 77.2: PIM-Audit", "", "**Execution: COMPLETE_READ_ONLY. Decision: " + extra["decision"] + ".**",
        "No E74.4/E77/E77.1 model, optimizer, checkpoint, prompt or memory was trained, updated or reinitialized.", "",
        "## Provenance and exact computation", "",
        f"Frozen E77.1-G2 source: `{E771}`; E74.4-G2 source: `{E74}`. Ten E77.1 and ten E74.4 best checkpoints were SHA-locked before analysis. See `manifest.json`.",
        "E77.1-G2 is `dugsa.PLNNoPEN`, with no active PEN. Raw CORR is `state.parts[3]` `[871,256]`; CAM's corresponding CORR latent block is captured from the CAM-to-OutputLayer `[871,144]` input slice `108:144` `[871,36]`. This is a shared patient/modality provenance, not an invertible or anatomically aligned representation.",
        "The exact hard evidence path is `sim = Top2Softmax(feature cosine + semantic relation)`, `contrast = sim - competing_typical/1.35 - 0.05`, `gate = sigmoid(contrast/0.2)*valid`, `value = 0.5*tanh(2*relu(contrast))*gate`; weighted modality evidence is summed by target class, clamped to `[0,0.5]`, and enters the bounded ASD-vs-TD log-probability correction. The read-only audit restored class/hard usage counters after forward and verified RNG, model/CAM state and memory state hashes were unchanged.",
        "Checkpoint hard state stores current patient IDs and first/last stable admission epoch, update frequency, current-window votes, confidence, novelty and prototype assignments. These times are first/last **stable-pool admissions**, not first/last single mistakes. `hard_diagnostics` preserves aggregate counts per epoch, but not epoch-by-epoch patient membership or the complete prediction sequence. Duration and recurrence below are therefore bounded summaries of stored state, not reconstructed trajectories.", "",
        "## Q1: Hard-memory stability", "",
        f"FP is true TD (label 1) predicted ASD (0); FN is true ASD (0) predicted TD (1). Across fold-specific G2 banks, the current pool contains {h1['fold_sum_fp']} FP and {h1['fold_sum_fn']} FN records (fold sums, not globally unique people), representing {h1['global_unique_fp']} unique global FP IDs and {h1['global_unique_fn']} unique global FN IDs. Mean per-fold pool sizes are {h1['mean_fp_per_fold']:.1f} FP and {h1['mean_fn_per_fold']:.1f} FN. Current best-checkpoint model predictions still err on {h1['still_wrong_fp']}/{h1['pool_fp']} FP and {h1['still_wrong_fn']}/{h1['pool_fn']} FN pool records; the rest are currently corrected.",
        f"Stored stable-admission spans have median {h1['median_span_fp']:.1f} epochs (FP) and {h1['median_span_fn']:.1f} epochs (FN); update-frequency medians are {h1['median_freq_fp']:.1f} and {h1['median_freq_fn']:.1f}. These are sparse admissions, not proof of continuous error. {crossfold['FP']['patient_repeated_in_memory_multiple_folds']}/{crossfold['FP']['unique_global_memory_patient_ids']} FP IDs and {crossfold['FN']['patient_repeated_in_memory_multiple_folds']}/{crossfold['FN']['unique_global_memory_patient_ids']} FN IDs recur in multiple folds. Fold-level same-direction Jaccard is {crossfold['FP']['observed_pairwise_jaccard_mean']:.4f} for FP and {crossfold['FN']['observed_pairwise_jaccard_mean']:.4f} for FN, versus matched-label/training-overlap random means {crossfold['FP']['matched_random_pairwise_jaccard_mean']:.4f} and {crossfold['FN']['matched_random_pairwise_jaccard_mean']:.4f}; observed/random ratios are {crossfold['FP']['observed_over_random_ratio']:.2f}x/{crossfold['FN']['observed_over_random_ratio']:.2f}x. {crossfold['FP']['pairs_empirical_p_below_0p05']}/45 FP and {crossfold['FN']['pairs_empirical_p_below_0p05']}/45 FN fold pairs exceed the matched random upper-tail threshold. Repeated subjects across folds are not independent external replications.",
        f"With hard evidence disabled at the same checkpoints, test-fold wrong-to-correct versus correct-to-wrong changes sum to {extra['hard_off_test_wrong_to_right']} versus {extra['hard_off_test_right_to_wrong']}; on stored FP/FN training-pool records the directionally corrected count is {extra['memory_wrong_to_right']} versus {extra['memory_right_to_wrong']}. Mean true-class test-margin change (full minus hard-off) is {extra['hard_off_test_margin_delta']:.5f}.", "",
        "## Q2: Conditional increment", "",
        "Zero-fit frozen scores are reported as ROC-AUC and class-conditional score distributions; prototype similarity is not presented as a calibrated probability. The CPU probes are the single predeclared L2 logistic fit per fold, C=1, with train-only scaling and test-overlap IDs removed. Prototype feature scores for training probes use the implementation's leave-one-subject-out centers.",
        f"P1−P0 (CORR256 + evidence vs CORR256) mean paired ΔBA/ΔAUC/log-loss is {100*h2['p1_p0_ba']:+.3f}/{100*h2['p1_p0_auc']:+.3f} pp/{h2['p1_p0_logloss']:+.5f}; fold wins for BA/AUC are {h2['p1_ba_wins']}/10 and {h2['p1_auc_wins']}/10. P3−P2 (CAM144 + evidence vs CAM144) is {100*h2['p3_p2_ba']:+.3f}/{100*h2['p3_p2_auc']:+.3f} pp/{h2['p3_p2_logloss']:+.5f}; fold wins are {h2['p3_ba_wins']}/10 and {h2['p3_auc_wins']}/10. P4 margin augmentation mean ΔAUC/BA is {100*h2['p4_auc']:+.3f}/{100*h2['p4_ba']:+.3f} pp.",
        *probe_table,
        *zero_auc_table,
        f"Frozen model test metrics, mean±population SD (%) in ACC/Hard-AUC/Prob-AUC/SEN/SPE order: E74.4-G2 {model_metric_line('E74.4-G2')}; E77.1-G2 {model_metric_line('E77.1-G2')}.",
        "The probe is a new supervised statistic fitted to frozen representations; it does not imply information theoretically independent of CORR/CAM or unbiased clinical generalization. The upstream E77.1 checkpoint itself remains test-ACC-selected.",
        f"After excluding Fold 2 (descriptive only), checkpoint means ± population SD (%) are: {'; '.join(sens_lines)}. Probe paired ΔBA/ΔAUC excluding Fold 2 (pp): P1−P0 {100*fold2_probe_sensitivity[0]:+.3f}/{100*fold2_probe_sensitivity[1]:+.3f}; P3−P2 {100*fold2_probe_sensitivity[2]:+.3f}/{100*fold2_probe_sensitivity[3]:+.3f}.", "",
        "## Q3: ReLU and gradient", "",
        f"For the test folds, FP has a valid prototype for {hard_stat('test','FP')['patients_with_valid_prototype']}/{hard_stat('test','FP')['all_patients']} patient queries and {hard_stat('test','FP')['valid_patient_modality_queries']} valid patient-modality queries; FN has {hard_stat('test','FN')['patients_with_valid_prototype']}/{hard_stat('test','FN')['all_patients']} and {hard_stat('test','FN')['valid_patient_modality_queries']}. Among valid patient-modality queries, contrast<=0 / exact-zero value are FP {h3['test_fp_relu_zero_fraction']:.1%}/{h3['test_fp_zero_evidence_fraction']:.1%} and FN {h3['test_fn_relu_zero_fraction']:.1%}/{h3['test_fn_zero_evidence_fraction']:.1%}; these are query-level fractions, not patient-level fractions. Near-zero-positive contributions occur in {hard_stat('test','FP')['near_zero_positive_fraction_positive']:.1%} of FP positive values and {hard_stat('test','FN')['near_zero_positive_fraction_positive']:.1%} of FN positive values. Target hard-score clipping is {hard_stat('test','FP')['target_class_hard_score_clipped_fraction_all_patients']:.1%}/{hard_stat('test','FN')['target_class_hard_score_clipped_fraction_all_patients']:.1%} of all test patient queries.",
        f"On allowed training queries the FP/FN valid-query zero-value fractions are {hard_stat('allowed_train','FP')['zero_value_fraction_valid']:.1%}/{hard_stat('allowed_train','FN')['zero_value_fraction_valid']:.1%}; on test queries they are {hard_stat('test','FP')['zero_value_fraction_valid']:.1%}/{hard_stat('test','FN')['zero_value_fraction_valid']:.1%}. Test modality zero-value fractions (PHENO/ANAT/FUNC/CORR) are FP {'/'.join(f'{x['zero_value_fraction']:.1%}' for x in hard_stat('test','FP')['by_modality'])} and FN {'/'.join(f'{x['zero_value_fraction']:.1%}' for x in hard_stat('test','FN')['by_modality'])}.",
        f"Direction-specific local derivative zero fractions on valid test queries are FP/FN {grad['test']['FP']['relu_zero_gradient_fraction_valid']:.1%}/{grad['test']['FN']['relu_zero_gradient_fraction_valid']:.1%}; small-positive-gradient fractions are {grad['test']['FP']['positive_but_small_local_gradient_fraction_valid']:.1%}/{grad['test']['FN']['positive_but_small_local_gradient_fraction_valid']:.1%}. On stable training memory records, frozen ranking loss is {h3['ranking_loss']:.5f}; among positive per-patient ranking losses, {h3['ranking_zero_grad_fraction']:.1%} have zero contrast-path gradient.",
        f"A fixed-weight Softplus(beta=5) counterfactual, with the same gate and 0.5 saturation cap, changes hard-evidence coverage from {h3['original_test_active_fraction']:.1%} to {h3['softplus_test_active_fraction']:.1%}, test true-class margin by {h3['softplus_test_margin_delta']:+.5f} on average, and changes predictions wrong-to-right/right-to-wrong {h3['softplus_test_wrong_to_right']}/{h3['softplus_test_right_to_wrong']}. This is an operator diagnosis only, not trained-model performance.", "",
        "## Hypothesis decisions and limits", "",
        "| Core hypothesis | Result | Continue? |", "|---|---|---|",
        f"| H1: historical errors are reproducible | {extra['h1_result']} | {'YES' if extra['h1_continue'] else 'NO'} |",
        f"| H2: prototypes add information beyond CAM | {extra['h2_result']} | {'YES' if extra['h2_continue'] else 'NO'} |",
        f"| H3: ReLU truncation is the main failure | {extra['h3_result']} | {'YES' if extra['h3_continue'] else 'NO'} |", "",
        f"Final decision: **{extra['decision']}**. {extra['decision_reason']}",
        "Limitations preserved: Fold-2 train/test overlap IDs 377 and 407; `val=test` checkpoint choice; pre-CV RFE; transductive full-graph features. The Fold-2-excluded nine-fold summaries above are descriptive only. No claim of independent generalization or cross-seed stability is made.",
        "All source/checkpoint/data hashes were identical before and after the audit. No model or checkpoint was saved or modified.", "",
        "Main artifacts: `summary.csv`, `probe.csv`, `details.json`, `manifest.json`.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict:
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("E77.2 requires the frozen-checkpoint GPU forward path; CUDA is unavailable")
    OUT.mkdir(parents=True, exist_ok=True)
    io_probe = OUT / ".io_probe"
    io_probe.write_text("ok", encoding="ascii")
    if io_probe.read_text(encoding="ascii") != "ok":
        raise RuntimeError("output I/O probe failed")
    io_probe.unlink()
    hashes_before = protected_inputs()
    preflight = json.loads((E771 / "results/preflight.json").read_text(encoding="utf-8"))
    fold_data = []
    for fold in range(1, 11):
        print(f"READ_ONLY_FOLD {fold}/10", flush=True)
        d = _read_e77_fold(fold, preflight)
        # Keep logits for the fixed hard-off intervention from the already audited inference path.
        checkpoint, state, module, tracker, runtime, _old, _sha, _path = E77_AUDIT._load_checkpoint(
            CFG["inputs"]["group"], fold, preflight)
        module.eval(); state["model"].eval(); state["cam"].eval()
        usage = {k: getattr(module,k).clone() for k in ("class_usage","class_queries","hard_usage","hard_queries")}
        rng0=pack_sha(runtime); capture={}
        loc=int(checkpoint["best_epoch_1based"])-1-int(E77_RUN.CFG["base"]["start_epoch_zero_based"])
        with torch.inference_mode():
            full,_=E77_MODEL.forward_sc_perm(E77_RUN.E74,state,module,CFG["inputs"]["group"],training=False,local_epoch=loc,capture={})
            off,_=E77_MODEL.forward_sc_perm(E77_RUN.E74,state,module,CFG["inputs"]["group"],training=False,local_epoch=loc,hard_off=True,capture=capture)
        for k,v in usage.items(): getattr(module,k).copy_(v)
        if pack_sha(runtime)!=rng0: raise RuntimeError(f"fold {fold} intervention eval advanced RNG")
        d["hard_off_logp"]=off["logp"].detach().float().cpu().numpy()
        d["full_logp_recheck"]=full["logp"].detach().float().cpu().numpy()
        if not np.allclose(d["full_logp_recheck"],d["e771_logp"],atol=2e-7,rtol=0):
            raise RuntimeError(f"fold {fold} full-model replay differs from extracted checkpoint output")
        d["memory_diagnostics"]=_memory_diagnostics(d,fold,d["e771_logp"].argmax(1))
        d["correctness"]=_correctness_summaries(d)
        d["relu_diagnostics"]=_relu_diagnostics(d)
        d["softplus_diagnostics"]=_softplus_summary(d)
        fold_data.append(d)
        del checkpoint,state,module,tracker,full,off
        torch.cuda.empty_cache()

    crossfold=_cross_fold_memory(fold_data)
    probe_rows,zero_fit,probe_extra=_probe_and_score(fold_data)
    # Fold-2 is retained in the primary protocol; summarize exclusion only descriptively.
    metrics_rows=[]
    for d in fold_data:
        y=d["labels"]; test=d["test_ids"]
        base=probabilities_and_predictions(d["e74_logp"][test],y[test])
        full=probabilities_and_predictions(d["e771_logp"][test],y[test])
        nohard=probabilities_and_predictions(d["hard_off_logp"][test],y[test])
        metrics_rows.append({"fold":d["fold"],"best_epoch_1based":d["best_epoch_1based"],
            "e74_acc":100*base["acc"],"e74_ba":100*base["ba"],"e74_prob_auc":100*base["prob_auc"],"e74_sen":100*base["sen"],"e74_spe":100*base["spe"],
            "e771_acc":100*full["acc"],"e771_ba":100*full["ba"],"e771_prob_auc":100*full["prob_auc"],"e771_sen":100*full["sen"],"e771_spe":100*full["spe"],
            "e771_hard_off_acc":100*nohard["acc"],"e771_hard_off_ba":100*nohard["ba"],"e771_hard_off_prob_auc":100*nohard["prob_auc"],
            "e771_hard_off_log_loss":nohard["log_loss"],"e771_hard_off_brier":nohard["brier"],
            "test_n":len(test),"allowed_train_n":len(d["train_ids"]),"fold2_overlap_excluded_from_probe_memory":d["fold"]==2,
            "rng_unchanged":d["rng_unchanged"],"e771_checkpoint_sha256":d["checkpoint_sha256"],"e74_checkpoint_sha256":d["base_checkpoint_sha256"]})
    metric_means={}
    for name,prefix in (("E74.4-G2","e74"),("E77.1-G2","e771")):
        metric_means[name]={}
        for metric in ("acc","ba","prob_auc","sen","spe"):
            values=[r[f"{prefix}_{metric}"] for r in metrics_rows]
            metric_means[name]["hard_auc" if metric=="ba" else metric]=float(np.mean(values))
            metric_means[name]["sd_"+("hard_auc" if metric=="ba" else metric)]=float(np.std(values,ddof=0))
    sensitivity={}
    for key in ("e74","e771"):
        sensitivity[key]={}
        for metric in ("acc","ba","prob_auc","sen","spe"):
            values=[r[f"{key}_{metric}"] for r in metrics_rows if r["fold"]!=2]
            sensitivity[key]["hard_auc" if metric=="ba" else metric]={"mean_percent":float(np.mean(values)),"population_sd_percent":float(np.std(values,ddof=0))}

    # Aggregate hard memory and intervention summaries.
    all_records={name:{} for name in ("FP","FN")}
    for d in fold_data:
        for name in ("FP","FN"):
            for rec in d["memory_diagnostics"][name]["patients"]:
                all_records[name][(d["fold"],rec["id"])]=rec
    hard_summary={
        "fold_sum_fp":sum(len(d["memory_diagnostics"]["FP"]["patients"]) for d in fold_data),
        "fold_sum_fn":sum(len(d["memory_diagnostics"]["FN"]["patients"]) for d in fold_data),
        "global_unique_fp":len({sid for _,sid in all_records["FP"]}),"global_unique_fn":len({sid for _,sid in all_records["FN"]}),
        "pool_fp":sum(len(d["memory_diagnostics"]["FP"]["patients"]) for d in fold_data),
        "pool_fn":sum(len(d["memory_diagnostics"]["FN"]["patients"]) for d in fold_data),
        "mean_fp_per_fold":float(np.mean([len(d["memory_diagnostics"]["FP"]["patients"]) for d in fold_data])),
        "mean_fn_per_fold":float(np.mean([len(d["memory_diagnostics"]["FN"]["patients"]) for d in fold_data])),
        "median_span_fp":float(np.median([r["last_stable_epoch"]-r["first_stable_epoch"] for r in all_records["FP"].values()])) if all_records["FP"] else 0.0,
        "median_span_fn":float(np.median([r["last_stable_epoch"]-r["first_stable_epoch"] for r in all_records["FN"].values()])) if all_records["FN"] else 0.0,
        "median_freq_fp":float(np.median([r["stable_update_frequency"] for r in all_records["FP"].values()])) if all_records["FP"] else 0.0,
        "median_freq_fn":float(np.median([r["stable_update_frequency"] for r in all_records["FN"].values()])) if all_records["FN"] else 0.0,
        "still_wrong_fp":sum(d["memory_diagnostics"]["FP"]["current_best_checkpoint_wrong"] for d in fold_data),
        "still_wrong_fn":sum(d["memory_diagnostics"]["FN"]["current_best_checkpoint_wrong"] for d in fold_data),
        "hard_memory_summary_per_fold":{str(d["fold"]):{
            name:{k:v for k,v in d["memory_diagnostics"][name].items() if k!="patients"} for name in ("FP","FN")} for d in fold_data},
    }
    memory_correctness={
        "wrong_to_right":sum(d["correctness"]["all_test"]["hard_evidence_wrong_to_correct"] for d in fold_data),
        "right_to_wrong":sum(d["correctness"]["all_test"]["hard_evidence_correct_to_wrong"] for d in fold_data),
        "memory_wrong_to_right":sum(d["correctness"]["FP_memory_patients"].get("corrected_wrong_to_right",0)+d["correctness"]["FN_memory_patients"].get("corrected_wrong_to_right",0) for d in fold_data),
        "memory_right_to_wrong":sum(d["correctness"]["FP_memory_patients"].get("correct_to_wrong",0)+d["correctness"]["FN_memory_patients"].get("correct_to_wrong",0) for d in fold_data),
        "test_margin_delta":float(np.mean([d["correctness"]["all_test"]["mean_true_margin_delta_full_minus_hard_off"] for d in fold_data])),
    }
    p1p0={r["metric"]:r for r in probe_extra["paired_probe_comparisons"] if r["comparison"]=="P1_CORR256_plus_evidence_minus_P0_CORR256"}
    p3p2={r["metric"]:r for r in probe_extra["paired_probe_comparisons"] if r["comparison"]=="P3_CAM144_plus_evidence_minus_P2_CAM144"}
    p4={r["metric"]:r for r in probe_extra["paired_probe_comparisons"] if r["comparison"]=="P4_margin_plus_evidence_minus_P4_margin_only"}
    def win_count(cmp,metric): return int(cmp[metric]["wins_ties_losses_for_augmented"][0])
    h2={"p1_p0_ba":p1p0["ba"]["mean_delta"],"p1_p0_auc":p1p0["prob_auc"]["mean_delta"],"p1_p0_logloss":p1p0["log_loss"]["mean_delta"],
        "p1_ba_wins":win_count(p1p0,"ba"),"p1_auc_wins":win_count(p1p0,"prob_auc"),
        "p3_p2_ba":p3p2["ba"]["mean_delta"],"p3_p2_auc":p3p2["prob_auc"]["mean_delta"],"p3_p2_logloss":p3p2["log_loss"]["mean_delta"],
        "p3_ba_wins":win_count(p3p2,"ba"),"p3_auc_wins":win_count(p3p2,"prob_auc"),
        "p4_auc":p4["prob_auc"]["mean_delta"],"p4_ba":p4["ba"]["mean_delta"]}
    relu_byfold=[d["relu_diagnostics"] for d in fold_data]
    def pooled_relu(subset, direction):
        rows=[r[subset][direction] for r in relu_byfold]
        all_patients=sum(r["all_denominators"]["all_patients"]["n_patients"] for r in rows)
        has_proto=sum(r["all_denominators"]["has_valid_hard_prototype"]["n_patients"] for r in rows)
        valid_queries=sum(r["all_denominators"]["has_valid_hard_prototype"]["n_valid_patient_modality_queries"] for r in rows)
        zero=sum(r["all_denominators"]["has_valid_hard_prototype"]["value_zero_n"] for r in rows)
        contrast_le_zero=sum(r["all_denominators"]["has_valid_hard_prototype"]["contrast_le_zero_n"] for r in rows)
        positive=sum(r["all_denominators"]["has_valid_hard_prototype"]["value_positive_n"] for r in rows)
        near=sum(r["all_denominators"]["has_valid_hard_prototype"]["near_zero_positive_n"] for r in rows)
        clipped_n=sum(r["target_class_hard_score_clipped_n"] for r in rows)
        direction_patient_n=sum(r["true_direction_patient_n"] for r in rows)
        true_target_zero=sum(r["true_direction_patient_n"]*r["true_direction_target_hard_evidence_zero_fraction"] for r in rows)
        modalities=[]
        for mi, modality_name in enumerate(("PHENO","ANAT","FUNC","CORR")):
            per=[r["by_modality"][mi] for r in rows]
            mn=sum(x["valid_n"] for x in per)
            modalities.append({"name":modality_name,"valid_queries":mn,
                "zero_value_fraction":sum(x["zero_value_fraction_valid"]*x["valid_n"] for x in per)/max(mn,1),
                "contrast_le_zero_fraction":sum(x["contrast_le_zero_fraction_valid"]*x["valid_n"] for x in per)/max(mn,1)})
        return {"all_patients":all_patients,"patients_with_valid_prototype":has_proto,
            "valid_patient_modality_queries":valid_queries,"contrast_le_zero_n":contrast_le_zero,
            "contrast_le_zero_fraction_valid":contrast_le_zero/max(valid_queries,1),
            "zero_value_n":zero,"zero_value_fraction_valid":zero/max(valid_queries,1),
            "positive_value_n":positive,"near_zero_positive_n":near,
            "near_zero_positive_fraction_positive":near/max(positive,1),
            "target_class_hard_score_clipped_n":clipped_n,
            "target_class_hard_score_clipped_fraction_all_patients":clipped_n/max(all_patients,1),
            "true_direction_patient_n":direction_patient_n,
            "true_direction_target_hard_evidence_zero_fraction":true_target_zero/max(direction_patient_n,1),
            "by_modality":modalities}
    def pooled_gradient(subset, direction):
        rows=[r["gradient"][subset]["by_direction"][direction] for r in relu_byfold]
        n=sum(r["valid_contrast_count"] for r in rows)
        return {"valid_contrast_count":n,
            "relu_zero_gradient_fraction_valid":sum(r["relu_zero_gradient_fraction_valid"]*r["valid_contrast_count"] for r in rows)/max(n,1),
            "positive_but_small_local_gradient_fraction_valid":sum(r["positive_but_small_local_gradient_fraction_valid"]*r["valid_contrast_count"] for r in rows)/max(n,1),
            "mean_local_derivative_on_valid":sum(r["mean_local_derivative_on_valid"]*r["valid_contrast_count"] for r in rows)/max(n,1)}
    softplus_byfold=[d["softplus_diagnostics"]["test"] for d in fold_data]
    relu_population={subset:{name:pooled_relu(subset,name) for name in ("FP","FN")} for subset in ("allowed_train","test")}
    gradient_population={subset:{name:pooled_gradient(subset,name) for name in ("FP","FN")} for subset in ("allowed_train","test")}
    h3={
        "population":relu_population,"gradient_population":gradient_population,
        "test_fp_relu_zero_fraction":relu_population["test"]["FP"]["contrast_le_zero_fraction_valid"],
        "test_fn_relu_zero_fraction":relu_population["test"]["FN"]["contrast_le_zero_fraction_valid"],
        "test_fp_zero_evidence_fraction":relu_population["test"]["FP"]["zero_value_fraction_valid"],
        "test_fn_zero_evidence_fraction":relu_population["test"]["FN"]["zero_value_fraction_valid"],
        "test_fp_zero_grad_fraction":gradient_population["test"]["FP"]["relu_zero_gradient_fraction_valid"],
        "test_fn_zero_grad_fraction":gradient_population["test"]["FN"]["relu_zero_gradient_fraction_valid"],
        "ranking_loss":float(np.mean([d["relu_diagnostics"]["gradient"]["ranking_loss"]["loss_value"] for d in fold_data])),
        "ranking_zero_grad_fraction":float(np.mean([d["relu_diagnostics"]["gradient"]["ranking_loss"].get("positive_loss_zero_gradient_fraction",0.0) for d in fold_data])),
        "original_test_active_fraction":float(np.mean([d["original_hard_evidence_active_patient_fraction"] for d in softplus_byfold])),
        "softplus_test_active_fraction":float(np.mean([d["softplus_hard_evidence_active_patient_fraction"] for d in softplus_byfold])),
        "softplus_test_margin_delta":float(np.mean([d["mean_true_margin_delta_softplus_minus_original"] for d in softplus_byfold])),
        "softplus_test_wrong_to_right":int(sum(d["original_wrong_to_correct"] for d in softplus_byfold)),
        "softplus_test_right_to_wrong":int(sum(d["original_correct_to_wrong"] for d in softplus_byfold)),
    }
    h1_continues=(crossfold["FP"]["observed_over_random_ratio"]>=1.5 and crossfold["FN"]["observed_over_random_ratio"]>=1.5 and
                  (hard_summary["still_wrong_fp"]+hard_summary["still_wrong_fn"])>0)
    h2_continues=(h2["p3_p2_ba"]>0 and h2["p3_p2_auc"]>0 and h2["p3_ba_wins"]>=8 and h2["p3_auc_wins"]>=8)
    h3_continues=(h3["test_fp_relu_zero_fraction"]>=0.5 or h3["test_fn_relu_zero_fraction"]>=0.5) and h3["ranking_zero_grad_fraction"]>=0.5
    if not h2_continues:
        increment="NO_CONDITIONAL_INCREMENT"
    elif h2["p1_p0_ba"]>0 and h2["p1_p0_auc"]>0 and (h2["p3_p2_ba"]<=0 or h2["p3_p2_auc"]<=0):
        increment="RAW_CORR_INCREMENT_ONLY"
    else:
        increment="CAM_COMPLEMENTARY_EVIDENCE_CANDIDATE"
    h1_result="PASS" if h1_continues else ("INCONCLUSIVE" if crossfold["FP"]["observed_over_random_ratio"]>1.0 or crossfold["FN"]["observed_over_random_ratio"]>1.0 else "FAIL")
    h2_result="PASS" if h2_continues else "FAIL"
    h3_result="PASS" if h3_continues else "FAIL"
    if h2_continues and h1_continues: decision="GO_REDESIGN"
    elif not h2_continues: decision="NO_GO_PROTOTYPE"
    else: decision="INCONCLUSIVE"
    extra={"hard_memory_summary":hard_summary,"incremental_summary":h2,"relu_summary":h3,"metric_means":metric_means,
           "hard_off_test_wrong_to_right":memory_correctness["wrong_to_right"],"hard_off_test_right_to_wrong":memory_correctness["right_to_wrong"],
           "memory_wrong_to_right":memory_correctness["memory_wrong_to_right"],"memory_right_to_wrong":memory_correctness["memory_right_to_wrong"],
           "hard_off_test_margin_delta":memory_correctness["test_margin_delta"],"h1_result":h1_result,"h2_result":h2_result,"h3_result":h3_result,
           "h1_continue":h1_continues,"h2_continue":h2_continues,"h3_continue":h3_continues,"decision":decision,
           "decision_reason":f"H1={h1_result} supports descriptive recurrence only, while H2={h2_result} found no stable conditional increment beyond CAM; stop adding prototype/Prompt modules under this representation. H3={h3_result} does not change that conclusion."}
    details={
        "experiment":"E77.2 PIM-Audit","status":"COMPLETE_READ_ONLY","decision":decision,
        "checkpoint_source":"E77.1-G2 + E74.4-G2 best checkpoints; no training or optimizer step",
        "folds":[],"cross_fold_memory":crossfold,"zero_fit_scores":zero_fit,"probe_comparisons":probe_extra,
        "hard_memory_aggregate":hard_summary,"correctness_and_margin_interventions":memory_correctness,
        "incremental_probe_summary":h2,"relu_summary":h3,"sensitivity_excluding_fold2":sensitivity,
        "historical_limits":{"fold2_overlap_1based":[377,407],"val_equals_test":True,"pre_cv_rfe":True,"transductive_full_graph":True,
                              "exclude_fold2_is_descriptive_only":True},
        "interpretation":{"increment_category":increment,"H1":h1_result,"H2":h2_result,"H3":h3_result},
    }
    for d in fold_data:
        compact={k:d[k] for k in ("fold","best_epoch_1based","memory_update_epoch","checkpoint_sha256","base_checkpoint_sha256","rng_unchanged","hook_calls",
            "module_constants","base_metrics","e771_metrics","memory_diagnostics","correctness","relu_diagnostics","softplus_diagnostics")}
        details["folds"].append(compact)
    save_csv(OUT/"summary.csv",metrics_rows)
    save_csv(OUT/"probe.csv",probe_rows)
    save_json(OUT/"details.json",details)
    _write_report(fold_data,probe_rows,zero_fit,extra,crossfold,hashes_before,sensitivity)
    hashes_after=protected_inputs()
    changed=[path for path in hashes_before if hashes_before[path]!=hashes_after.get(path)]
    if changed:
        raise RuntimeError("protected input hashes changed during read-only audit: "+", ".join(changed))
    source_hashes={
        "audit.py":sha_file(HERE/"audit.py"),"config.json":sha_file(HERE/"config.json"),
        "e771_sources":{str(p):sha_file(p) for p in (E771/"memory.py",E771/"model.py",E771/"run.py",E771/"audit.py",E771/"config.json")},
        "e74_sources":{str(p):sha_file(p) for p in (E74/"run.py",E74/"soft_positive_dcr.py",E74/"config.json")},
    }
    manifest={"experiment":"Experiment 77.2 - PIM-Audit","status":"COMPLETE_READ_ONLY","decision":decision,
        "directory":str(HERE),"seed":0,"analysis_rng_seed":CFG["analysis_rng_seed"],"training_or_optimizer_steps":0,
        "models_reinitialized":False,"memory_updated":False,"prompt_changed":False,
        "e771_checkpoints":[{"fold":d["fold"],"path":d["checkpoint_path"],"sha256":d["checkpoint_sha256"]} for d in fold_data],
        "e74_checkpoints":[{"fold":d["fold"],"path":d["base_checkpoint_path"],"sha256":d["base_checkpoint_sha256"]} for d in fold_data],
        "protected_input_sha256_before":hashes_before,"protected_input_sha256_after":hashes_after,"protected_inputs_unchanged":True,
        "analysis_sources":source_hashes,
        "artifacts":{"report.md":sha_file(OUT/"report.md"),"summary.csv":sha_file(OUT/"summary.csv"),"probe.csv":sha_file(OUT/"probe.csv"),"details.json":sha_file(OUT/"details.json")},
        "probe":"CPU-only, one fixed C=1.0 logistic fit per fold and setting; no tuning; no model files saved",
        "limitations":CFG["restrictions"]}
    save_json(HERE/"manifest.json",manifest)
    return manifest


if __name__ == "__main__":
    result=run()
    print(json.dumps({"status":result["status"],"decision":result["decision"],"folds":10,"training_steps":0},ensure_ascii=False))
