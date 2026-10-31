#!/usr/bin/env python3
"""Train and independently audit the single Experiment 53.3 fallback-fix group."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import run as R

EXP = R.EXP
ROOT = R.ROOT
CFG_PATH = EXP / "config.json"
EXP_CFG = json.loads(CFG_PATH.read_text())
GROUP = EXP_CFG["name"]
SWEEP_CFG = {"groups": [{"name": GROUP, "cadr_weight_decay": EXP_CFG["cadr_weight_decay"]}]}
SWEEP_ROOT = R.RT / "group_runs"
RESULTS = R.RESULTS
BASE_BUILD_STATE = R.build_state
BASE_BUILD_CHECKPOINT = R._build_checkpoint
BASE_TASK_PATH = R.task_path
BASE_BEST_PATH = R.best_path
BASE_LATEST_PATH = R.latest_path
BASE_RT = R.RT
BASE_CKPTS = R.CKPTS
BASE_LOGS = R.LOGS


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_grad_norm(values) -> float:
    import torch

    terms = [value.detach().float().square().sum() for value in values if value is not None]
    return float(torch.stack(terms).sum().sqrt().cpu()) if terms else 0.0


def group_config(group: str):
    for item in SWEEP_CFG["groups"]:
        if item["name"] == group:
            return item
    raise KeyError(group)


def group_paths(group: str):
    root = SWEEP_ROOT / group
    return root, root / "tasks", root / "checkpoints", root / "logs"


def task_path(group: str, fold: int) -> Path:
    return group_paths(group)[1] / f"{group}_f{fold:02d}.json"


def best_path(group: str, fold: int) -> Path:
    return group_paths(group)[2] / f"{group}_f{fold:02d}.best.pt"


def latest_path(group: str, fold: int) -> Path:
    return group_paths(group)[2] / f"{group}_f{fold:02d}.latest.pt"


def install_group(group: str):
    item = group_config(group)
    decay = float(item["cadr_weight_decay"])
    root, tasks, checkpoints, logs = group_paths(group)
    for path in (root, tasks, checkpoints, logs):
        path.mkdir(parents=True, exist_ok=True)

    # The original trainer resolves these paths from its module globals.
    R.RT = root
    R.CKPTS = checkpoints
    R.LOGS = logs
    R.task_path = lambda fold: task_path(group, fold)
    R.best_path = lambda fold: best_path(group, fold)
    R.latest_path = lambda fold: latest_path(group, fold)

    def build_state(fold, cache_meta=None):
        state = BASE_BUILD_STATE(fold, cache_meta)
        for param_group in state["opt_added"].param_groups:
            param_group["weight_decay"] = decay
        state["group"] = group
        state["cadr_weight_decay"] = decay
        assert all(float(pg["weight_decay"]) == decay for pg in state["opt_added"].param_groups)
        assert all(float(pg["weight_decay"]) == 0.11 for pg in state["opt_base"].param_groups)
        assert all(float(pg["weight_decay"]) == 0.11 for pg in state["opt_cam"].param_groups)
        return state

    def build_checkpoint(state, *args, **kwargs):
        checkpoint = BASE_BUILD_CHECKPOINT(state, *args, **kwargs)
        checkpoint["group"] = group
        checkpoint["cadr_weight_decay"] = decay
        checkpoint["optimizer_weight_decay"] = {
            "base": [float(pg["weight_decay"]) for pg in state["opt_base"].param_groups],
            "cam": [float(pg["weight_decay"]) for pg in state["opt_cam"].param_groups],
            "added": [float(pg["weight_decay"]) for pg in state["opt_added"].param_groups],
        }
        checkpoint["fallback_runner_sha256"] = sha_file(Path(__file__))
        checkpoint["fallback_config_sha256"] = sha_file(CFG_PATH)
        return checkpoint

    R.build_state = build_state
    R._build_checkpoint = build_checkpoint


def check_initialization(state, reference_by_fold):
    reference = reference_by_fold[state["fold"]]
    for key in (
        "initial_shared_sha", "initial_gfr_sha", "initial_schema_sha",
        "initial_cam_sha", "initial_added_sha",
    ):
        assert state[key] == reference[key], (state["fold"], key, state[key], reference[key])
    assert state["construction_rng_sha"] == reference["initial_rng_sha"]
    assert state["split_sha"] == reference["split_sha"]
    assert state["cache_meta"]["pair_sha"] == reference["pair_sha"]


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def load_reference():
    source = json.loads((EXP / ".runtime/reference_53_2_manifest.json").read_text())
    manifest = json.loads((EXP / ".runtime/reference_53_2_wd_sweep_manifest.json").read_text())
    reference_path = EXP / ".runtime/reference_53_2_wd0001_fold_metrics.csv"
    rows = read_rows(reference_path)
    by_fold = {int(row["fold"]): row for row in rows if row.get("group") == "CADR_WD_0p001"}
    assert len(by_fold) == 10
    assert manifest["status"] == "COMPLETE_AUDITED"
    assert manifest["independent_checkpoint_audits_passed"] == 40
    assert source["status"] == "COMPLETE_AUDITED" and source["checkpoint_audit_status"] == "PASS"
    assert EXP_CFG["cadr_weight_decay"] == 0.001
    runtime = ROOT / EXP_CFG["paths"]["historical_cadr_runtime"]
    source_folds = {int(item["fold"]): item for item in source["folds"] if item.get("group") == "E1_CADR_GFR_FULL"}
    assert len(source_folds) == 10
    for path, expected_sha in source["data_sha256"].items():
        assert Path(path).is_file() and sha_file(Path(path)) == expected_sha, ("data SHA", path)
    for path, expected_sha in source["anchor_sha256"].items():
        assert Path(path).is_file() and sha_file(Path(path)) == expected_sha, ("anchor SHA", path)
    historical_source = ROOT / "experiment/experiment53_2_cadr_gfr_integrated"
    for name, expected_sha in source["experiment_source_sha256"].items():
        if name in {"cadr_gfr.py", "run.py", "config.json", "audit.py"}:
            assert sha_file(historical_source / name) == expected_sha, ("historical source SHA", name)
    cache_dir = EXP / ".runtime/relation_cache"
    for fold in range(1, 11):
        cp_path = runtime / "checkpoints" / f"CADR_WD_0p001_f{fold:02d}.best.pt"
        assert cp_path.is_file()
        assert sha_file(cp_path) == by_fold[fold]["checkpoint_sha256"]
        assert float(by_fold[fold]["cadr_weight_decay"]) == 0.001
        assert json.loads(by_fold[fold]["cadr_optimizer_weight_decay"]) == [0.001, 0.001]
        source_fold = source_folds[fold]
        for key in (
            "initial_shared_sha", "initial_gfr_sha", "initial_schema_sha", "initial_cam_sha",
            "initial_added_sha", "initial_rng_sha", "split_sha", "pair_sha",
        ):
            if key in by_fold[fold]:
                assert source_fold[key] == by_fold[fold][key], (fold, key)
            by_fold[fold][key] = source_fold[key]
        cache_hashes = source["relation_cache_sha256"][f"fold{fold:02d}"]
        assert sha_file(cache_dir / f"fold{fold:02d}.json") == cache_hashes["json"]
        assert sha_file(cache_dir / f"fold{fold:02d}.npz") == cache_hashes["npz"]
    return {**manifest, "source_manifest": source}, by_fold


def preflight(reference_by_fold):
    import torch
    import importlib.util

    group = SWEEP_CFG["groups"][0]["name"]
    install_group(group)
    state = R.build_state(1, R.prepare_fold(1))
    check_initialization(state, reference_by_fold)
    added_wd = [float(pg["weight_decay"]) for pg in state["opt_added"].param_groups]
    assert added_wd == [0.001, 0.001], added_wd
    assert [float(pg["weight_decay"]) for pg in state["opt_base"].param_groups] == [0.11]
    assert [float(pg["weight_decay"]) for pg in state["opt_cam"].param_groups] == [0.11]
    with torch.no_grad():
        output = R.forward_state(state, training=False, use_cadr=True)
        raw = torch.cat(state["parts"], dim=1)
        identity = torch.eye(raw.shape[0], device=raw.device, dtype=raw.dtype)
        for modality, mode in enumerate(output["info"]["propagation_modes"]):
            reliability = float(state["cache_meta"]["reliability"][modality])
            if reliability == 0.0:
                expected = state["model"].gcn[modality](
                    raw,
                    state["model"].utils.normalize_adj(
                        output["info"]["original_adjs"][modality] + identity
                    ),
                )
                assert mode == "ORIGINAL_FALLBACK"
                assert torch.allclose(output["info"]["messages"][modality], expected, rtol=1e-6, atol=1e-7)
                assert torch.count_nonzero(output["info"]["positive_adjs"][modality]) == 0
                assert torch.count_nonzero(output["info"]["negative_adjs"][modality]) == 0
                assert torch.count_nonzero(output["info"]["positive_messages"][modality]) == 0
                assert torch.count_nonzero(output["info"]["negative_messages"][modality]) == 0
                assert torch.count_nonzero(output["info"]["direct_messages"][modality]) == 0
            else:
                assert mode == "CADR_DISEASE"

    # Compare the reliable CORR branch against the unchanged 53.2 implementation.
    old_path = ROOT / "experiment/experiment53_2_cadr_gfr_integrated/cadr_gfr.py"
    spec = importlib.util.spec_from_file_location("cadr_gfr_532_reference", old_path)
    old_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old_module
    spec.loader.exec_module(old_module)
    old_core = R.build_legacy_core(1)
    old_model = old_module.CADRGFR(
        old_core["base"], state["model"].reliability, state["model"].calibration,
        state["model"].fallback_modal, init_seed=532001,
    ).to(R.runtime()["device"])
    old_model.load_state_dict(state["model"].state_dict(), strict=True)
    old_model.eval()
    with torch.no_grad():
        old_fused, _, old_info = old_model(state["parts"], state["train_pairs"], use_cadr=True)
    assert torch.equal(output["info"]["messages"][3], old_info["messages"][3])
    assert torch.equal(output["info"]["positive_adjs"][3], old_info["positive_adjs"][3])
    assert torch.equal(output["info"]["negative_adjs"][3], old_info["negative_adjs"][3])
    assert torch.equal(output["info"]["direct_messages"][3], old_info["direct_messages"][3])

    # Disease-only intervention must not disable an unreliable modality's original graph.
    with torch.no_grad():
        pos_off = R.forward_state(state, training=False, use_cadr=True, intervention="positive_off")
    for modality, reliability in enumerate(state["cache_meta"]["reliability"]):
        if reliability == 0.0:
            assert torch.equal(output["info"]["messages"][modality], pos_off["info"]["messages"][modality])
    pheno_original = output["info"]["original_adjs"][0]
    pheno_original_message = output["info"]["messages"][0]
    pheno_self_message = state["model"].gcn[0](raw, identity)
    assert (pheno_original_message - pheno_self_message).norm().item() > 1e-8

    # Classification gradients must reach a fallback GCN and its GraphLearn.
    grad_out = R.forward_state(state, training=False, use_cadr=True)
    gcn_parameters = list(state["model"].gcn[0].parameters())
    graph_parameters = list(state["model"].graphs[0].parameters())
    grads = torch.autograd.grad(grad_out["cls"], gcn_parameters + graph_parameters, allow_unused=True)
    gcn_grad = grads[:len(gcn_parameters)]
    graph_grad = grads[len(gcn_parameters):]
    assert any(g is not None and torch.isfinite(g).all() and g.norm() > 0 for g in gcn_grad)
    assert any(g is not None and torch.isfinite(g).all() and g.norm() > 0 for g in graph_grad)

    outputs = [{
        "group": group,
        "fold": 1,
        "cadr_weight_decay": 0.001,
        "gfr_weight_decay": [0.11],
        "cam_weight_decay": [0.11],
        "cadr_optimizer_weight_decay": added_wd,
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_added_sha": state["initial_added_sha"],
        "initial_rng_sha": state["construction_rng_sha"],
        "fallback_modes": output["info"]["propagation_modes"],
        "corr_branch_equal_to_53_2": True,
        "fallback_gcn_and_graphlearn_cls_grad_nonzero": True,
        "original_neighbors_change_gcn_message": True,
        "positive_off_preserves_fallback": True,
    }]
    initialization_rows = []
    for fold in range(2, 11):
        fold_state = R.build_state(fold, R.prepare_fold(fold))
        check_initialization(fold_state, reference_by_fold)
        fold_wd = [float(pg["weight_decay"]) for pg in fold_state["opt_added"].param_groups]
        assert fold_wd == [0.001, 0.001]
        initialization_rows.append({
            "group": group,
            "fold": fold,
            "initial_shared_sha": fold_state["initial_shared_sha"],
            "initial_added_sha": fold_state["initial_added_sha"],
            "initial_rng_sha": fold_state["construction_rng_sha"],
            "split_sha": fold_state["split_sha"],
            "pair_sha": fold_state["cache_meta"]["pair_sha"],
            "reliability": fold_state["cache_meta"]["reliability"],
            "initialization_matches_53_2": True,
        })
        del fold_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    outputs.extend(initialization_rows)
    del state, output, old_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_rows(RESULTS / "preflight.csv", outputs)


def worker(group: str, fold: int):
    install_group(group)
    R.train_one(fold)
    task = json.loads(task_path(group, fold).read_text())
    task["group"] = group
    task["cadr_weight_decay"] = float(group_config(group)["cadr_weight_decay"])
    task["cadr_optimizer_weight_decay"] = [float(group_config(group)["cadr_weight_decay"])] * len(task["optimizer_weight_decay"]["added"])
    R.atomic_json(task_path(group, fold), task)


def audit_fold(group: str, fold: int, reference_by_fold):
    import numpy as np
    import torch

    install_group(group)
    state = R.build_state(fold, R.prepare_fold(fold))
    check_initialization(state, reference_by_fold)
    cp_path = best_path(group, fold)
    task_file = task_path(group, fold)
    checkpoint = R.torch_load(cp_path)
    task = json.loads(task_file.read_text())
    decay = float(group_config(group)["cadr_weight_decay"])
    expected_source = sha_file(Path(__file__))
    assert task["status"] == "COMPLETE" and task["best_sha256"] == sha_file(cp_path)
    assert checkpoint["group"] == group and checkpoint["cadr_weight_decay"] == decay
    assert checkpoint["fallback_runner_sha256"] == expected_source
    assert checkpoint["fallback_config_sha256"] == sha_file(CFG_PATH)
    assert checkpoint["fold"] == fold and checkpoint["seed"] == 0
    assert checkpoint["optimizer_weight_decay"]["base"] == [0.11]
    assert checkpoint["optimizer_weight_decay"]["cam"] == [0.11]
    assert all(value == decay for value in checkpoint["optimizer_weight_decay"]["added"])
    for key in (
        "initial_shared_sha", "initial_gfr_sha", "initial_schema_sha",
        "initial_cam_sha", "initial_added_sha", "initial_rng_sha", "pair_sha", "split_sha",
    ):
        cp_key = "construction_rng_sha" if key == "initial_rng_sha" else key
        ref_key = key
        expected = state[cp_key] if key == "initial_rng_sha" else state.get(key)
        if key == "pair_sha":
            expected = state["cache_meta"]["pair_sha"]
        if key == "split_sha":
            expected = state["split_sha"]
        if key == "initial_rng_sha":
            expected = state["construction_rng_sha"]
        assert checkpoint[key] == expected, (group, fold, key)
        if ref_key in reference_by_fold[fold]:
            assert expected == reference_by_fold[fold][ref_key], (group, fold, key, "baseline")
    assert checkpoint["best_epoch"] >= int(R.CFG["warmup"]["epochs"])
    state["model"].load_state_dict(checkpoint["model"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    state["opt_base"].load_state_dict(checkpoint["optimizer_base"])
    state["opt_cam"].load_state_dict(checkpoint["optimizer_cam"])
    state["opt_added"].load_state_dict(checkpoint["optimizer_added"])
    assert [float(pg["weight_decay"]) for pg in state["opt_added"].param_groups] == checkpoint["optimizer_weight_decay"]["added"]
    assert all(torch.isfinite(p).all() for p in state["model"].parameters())
    assert all(torch.isfinite(p).all() for p in state["cam"].parameters())
    state["model"].eval()
    state["cam"].eval()
    with torch.no_grad():
        out = R.forward_state(state, training=False, use_cadr=True)
        raw = torch.cat(state["parts"], dim=1)
        identity = torch.eye(raw.shape[0], dtype=raw.dtype, device=raw.device)
        for modality, reliability in enumerate(state["cache_meta"]["reliability"]):
            if float(reliability) == 0.0:
                assert out["info"]["propagation_modes"][modality] == "ORIGINAL_FALLBACK"
                assert torch.count_nonzero(out["info"]["positive_adjs"][modality]) == 0
                assert torch.count_nonzero(out["info"]["negative_adjs"][modality]) == 0
                assert torch.count_nonzero(out["info"]["positive_messages"][modality]) == 0
                assert torch.count_nonzero(out["info"]["negative_messages"][modality]) == 0
                assert torch.count_nonzero(out["info"]["direct_messages"][modality]) == 0
                expected = state["model"].gcn[modality](
                    raw,
                    state["model"].utils.normalize_adj(out["info"]["original_adjs"][modality] + identity),
                )
                assert torch.allclose(out["info"]["messages"][modality], expected, rtol=1e-6, atol=1e-7)
            else:
                assert out["info"]["propagation_modes"][modality] == "CADR_DISEASE"
        saved_labels = state["y"].clone()
        state["y"][torch.as_tensor(state["test"], dtype=torch.long, device=state["y"].device)] = 1 - state["y"][torch.as_tensor(state["test"], dtype=torch.long, device=state["y"].device)]
        label_permuted = R.forward_state(state, training=False, use_cadr=True)
        assert torch.equal(out["logp"], label_permuted["logp"])
        assert all(torch.equal(a, b) for a, b in zip(out["info"]["positive_adjs"], label_permuted["info"]["positive_adjs"]))
        state["y"].copy_(saved_labels)
    metrics = R.classification_metrics(out["logp"], state["test"], state["y"])
    for metric in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"):
        assert abs(float(metrics[metric]) - float(checkpoint["metrics"][metric])) <= 1e-10
    assert torch.isfinite(out["fused"]).all() and torch.isfinite(out["total"])
    raw = torch.cat(state["parts"], dim=1)
    info = out["info"]
    pos_ratio = float(torch.stack([value.norm() for value in info["positive_messages"]]).norm() / raw.norm().clamp_min(1e-12))
    neg_ratio = float(torch.stack([value.norm() for value in info["negative_messages"]]).norm() / raw.norm().clamp_min(1e-12))
    direct_ratio = float(torch.stack([value.norm() for value in info["direct_messages"]]).norm() / raw.norm().clamp_min(1e-12))
    diagnostic_rows = []
    grad_output = R.forward_state(state, training=False, use_cadr=True)
    for modality, name in enumerate(R.MODS):
        gcn = state["model"].gcn[modality]
        graph = state["model"].graphs[modality]
        gcn_grads = torch.autograd.grad(
            grad_output["cls"], list(gcn.parameters()), retain_graph=True, allow_unused=True
        )
        graph_grads = torch.autograd.grad(
            grad_output["cls"], list(graph.parameters()), retain_graph=True, allow_unused=True
        )
        gcn_grad_norm = _tensor_grad_norm(gcn_grads)
        graph_grad_norm = _tensor_grad_norm(graph_grads)
        original = info["original_adjs"][modality]
        message = info["messages"][modality]
        mode = info["propagation_modes"][modality]
        diagnostic_rows.append({
            "group": group,
            "fold": fold,
            "modality": name,
            "oof_reliability": float(state["cache_meta"]["reliability"][modality]),
            "oof_auc": float(state["cache_meta"]["relation_oof"][modality]["auc"]),
            "propagation_mode": mode,
            "original_graph_edges": int(torch.count_nonzero(torch.triu(original.detach() > 0, diagonal=1)).item()),
            "disease_positive_edges": int(torch.count_nonzero(torch.triu(info["positive_adjs"][modality].detach() > 0, diagonal=1)).item()),
            "disease_negative_edges": int(torch.count_nonzero(torch.triu(info["negative_adjs"][modality].detach() > 0, diagonal=1)).item()),
            "original_gcn_message_norm": float(message.detach().norm().cpu()),
            "original_gcn_message_input_ratio": float(message.detach().norm().div(raw.norm().clamp_min(1e-12)).cpu()),
            "cadr_positive_message_norm": float(info["positive_messages"][modality].detach().norm().cpu()),
            "cadr_negative_message_norm": float(info["negative_messages"][modality].detach().norm().cpu()),
            "direct_message_norm": float(info["direct_messages"][modality].detach().norm().cpu()),
            "classification_to_gcn_grad_norm": gcn_grad_norm,
            "classification_to_graphlearn_grad_norm": graph_grad_norm,
            "fusion_weight_sigmoid": float(torch.sigmoid(state["model"].fusion_weight[modality]).detach().cpu()),
            "cadr_reliability_gate": float(info["fusion_gates"][modality].detach().cpu()),
            "fallback_message_norm": float(info["original_fallback_messages"][modality].detach().norm().cpu()),
        })
    metric_row = {
        "group": group,
        "fold": fold,
        "seed": 0,
        "cadr_weight_decay": decay,
        "gfr_weight_decay": 0.11,
        "cam_weight_decay": 0.11,
        "cadr_optimizer_weight_decay": json.dumps(checkpoint["optimizer_weight_decay"]["added"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "epochs_executed": int(checkpoint["epochs_executed"]),
        **{metric: float(metrics[metric]) for metric in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")},
        "positive_gcn_message_input_ratio": pos_ratio,
        "negative_gcn_message_input_ratio": neg_ratio,
        "direct_message_input_ratio": direct_ratio,
        "final_minus_input_ratio": float((out["fused"] - raw).norm() / raw.norm().clamp_min(1e-12)),
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_added_sha": state["initial_added_sha"],
        "initial_rng_sha": state["construction_rng_sha"],
        "split_sha": state["split_sha"],
        "pair_sha": state["cache_meta"]["pair_sha"],
        "checkpoint_sha256": sha_file(cp_path),
        "checkpoint_audit": "PASS",
    }
    return metric_row, diagnostic_rows


def metric_summary(rows, reference_by_fold):
    import numpy as np

    metrics = ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")
    output = []
    for item in SWEEP_CFG["groups"]:
        group = item["name"]
        subset = sorted([row for row in rows if row["group"] == group], key=lambda row: int(row["fold"]))
        assert len(subset) == 10 and all(row["checkpoint_audit"] == "PASS" for row in subset)
        baseline = [reference_by_fold[int(row["fold"])] for row in subset]
        record = {
            "group": group,
            "cadr_weight_decay": float(item["cadr_weight_decay"]),
            "folds_audited": len(subset),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in subset]) * 100.0
            reference = np.asarray([float(row[metric]) for row in baseline]) * 100.0
            diff = values - reference
            record[f"{metric}_mean_percent"] = float(values.mean())
            record[f"{metric}_population_sd_percent"] = float(values.std(ddof=0))
            record[f"{metric}_delta_vs_53_2_wd_0p001_pp"] = float(diff.mean())
            record[f"{metric}_wins_ties_losses_vs_53_2_wd_0p001"] = f"{sum(diff > 1e-12)}/{sum(np.abs(diff) <= 1e-12)}/{sum(diff < -1e-12)}"
        for metric in ("best_epoch", "epochs_executed", "positive_gcn_message_input_ratio", "negative_gcn_message_input_ratio", "direct_message_input_ratio", "final_minus_input_ratio"):
            vals = np.asarray([float(row[metric]) for row in subset])
            record[f"{metric}_mean"] = float(vals.mean())
            record[f"{metric}_population_sd"] = float(vals.std(ddof=0))
        output.append(record)
    return output


def finalize_results(audited, diagnostics, reference_by_fold, reference_manifest):
    import numpy as np

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_rows(RESULTS / "fold_metrics.csv", audited)
    write_rows(RESULTS / "diagnostics.csv", diagnostics)
    audit_columns = (
        "group", "fold", "checkpoint_audit", "checkpoint_sha256", "best_epoch",
        "epochs_executed", "initial_shared_sha", "initial_added_sha", "initial_rng_sha",
        "split_sha", "pair_sha", "cadr_optimizer_weight_decay",
    )
    write_rows(RESULTS / "checkpoint_audit.csv", [{k: row[k] for k in audit_columns} for row in audited])
    summary = metric_summary(audited, reference_by_fold)
    write_rows(RESULTS / "summary.csv", summary)

    paired = []
    metric_names = ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")
    for row in sorted(audited, key=lambda item: int(item["fold"])):
        reference = reference_by_fold[int(row["fold"])]
        for metric in metric_names:
            value = float(row[metric])
            ref_value = float(reference[metric])
            paired.append({
                "comparison": "53.3_minus_53.2_CADR_WD_0p001",
                "fold": int(row["fold"]),
                "metric": metric,
                "experiment_53_3": value,
                "experiment_53_2": ref_value,
                "difference_pp": (value - ref_value) * 100.0,
            })
    for metric in metric_names:
        diffs = [r["difference_pp"] for r in paired if r["metric"] == metric]
        paired.append({
            "comparison": "53.3_minus_53.2_CADR_WD_0p001",
            "fold": "ALL",
            "metric": metric,
            "difference_pp": float(np.mean(diffs)),
            "wins_ties_losses": f"{sum(d > 1e-12 for d in diffs)}/{sum(abs(d) <= 1e-12 for d in diffs)}/{sum(d < -1e-12 for d in diffs)}",
        })
    write_rows(RESULTS / "paired_comparisons.csv", paired)

    labels = {
        "acc": "ACC", "hard_auc": "Hard-AUC", "prob_auc": "Probability-AUC",
        "spe": "SPE-TD", "sen": "SEN-ASD", "ba": "BA", "asd_f1": "ASD-F1",
    }
    table = ["| Metric | 53.2 WD=0.001 reference | 53.3 fallback fix | Delta (pp) | W/T/L |", "|---|---:|---:|---:|---:|"]
    for metric in metric_names:
        ref_vals = np.asarray([float(reference_by_fold[f][metric]) for f in range(1, 11)]) * 100.0
        values = np.asarray([float(row[metric]) for row in audited]) * 100.0
        diffs = values - ref_vals
        table.append(
            f"| {labels[metric]} | {ref_vals.mean():.4f} ± {ref_vals.std(ddof=0):.4f}% | "
            f"{values.mean():.4f} ± {values.std(ddof=0):.4f}% | {diffs.mean():+.4f} | "
            f"{sum(diffs > 1e-12)}/{sum(np.abs(diffs) <= 1e-12)}/{sum(diffs < -1e-12)} |"
        )
    mode_table = ["| Modality | Zero-reliability folds | Propagation mode | Original edges | Fallback message/input | GCN cls grad | GraphLearn cls grad |", "|---|---:|---|---:|---:|---:|---:|"]
    for modality in R.MODS:
        subset = [row for row in diagnostics if row["modality"] == modality]
        fallback_n = sum(row["propagation_mode"] == "ORIGINAL_FALLBACK" for row in subset)
        mode = "ORIGINAL_FALLBACK" if fallback_n else "CADR_DISEASE"
        mode_table.append(
            f"| {modality} | {fallback_n}/10 | {mode} | "
            f"{np.mean([row['original_graph_edges'] for row in subset]):.1f} | "
            f"{np.mean([row['original_gcn_message_input_ratio'] for row in subset if row['propagation_mode'] == 'ORIGINAL_FALLBACK']) if fallback_n else 0.0:.3e} | "
            f"{np.mean([row['classification_to_gcn_grad_norm'] for row in subset]):.3e} | "
            f"{np.mean([row['classification_to_graphlearn_grad_norm'] for row in subset]):.3e} |"
        )
    paired_summary = []
    for metric in ("acc", "hard_auc", "prob_auc", "spe", "sen"):
        diffs = [(float(row[metric]) - float(reference_by_fold[int(row["fold"])][metric])) * 100.0 for row in audited]
        paired_summary.append(f"| {labels[metric]} | {np.mean(diffs):+.4f} pp | {sum(d > 1e-12 for d in diffs)}/{sum(abs(d) <= 1e-12 for d in diffs)}/{sum(d < -1e-12 for d in diffs)} |")
    report = [
        "# Experiment 53.3 — CADR-GFR Original Graph Fallback Fix",
        "",
        "## Design and audit",
        "",
        "This run changes one CADR-GFR forward branch relative to Experiment 53.2 CADR_WD_0p001. A modality with fold-local OOF reliability exactly zero keeps its disease positive/negative adjacencies and direct disease message at zero, but now propagates through its own original GraphLearn adjacency with the already initialized two-layer GCN and the original sigmoid(fusion_weight) coefficient. Reliable modalities retain the original 53.2 CADR branch. `use_cadr=False` remains the historical E1 baseline path.",
        "",
        "The 53.2 relation cache, pair samples, OOF reliability, Platt calibration, untrained anchors, and WD=0.001 fold metrics/checkpoints were read-only sources. Cache JSON/NPZ SHA, split identity, reference checkpoint SHA, and initialization hashes were checked before training. All CADR AdamW parameter groups retain WD=0.001, including bias and LayerNorm groups; base GFR and CAM remain Adam WD=0.11.",
        "",
        "Preflight verified exact fallback-message equality with `GCN_m(X, normalize_adj(A_original + I))`, zero disease adjacencies/direct messages for fallback modalities, unchanged CORR CADR outputs against the 53.2 implementation, fallback persistence under `positive_off`, and nonzero classification gradients to the fallback GCN and GraphLearn. The zero-reliability decision is fold-local and derived only from inner patient-disjoint outer-train OOF data.",
        "",
        "## Classification results",
        "",
        *table,
        "",
        "## Modality fallback and gradient diagnostics",
        "",
        *mode_table,
        "",
        "## Paired differences",
        "",
        "| Metric | Mean paired difference | Wins / ties / losses |",
        "|---|---:|---:|",
        *paired_summary,
        "",
        "## Interpretation and limitations",
        "",
        "The table compares the new run fold-by-fold with the independently audited 53.2 WD=0.001 group. A restored original-graph message establishes that the base patient graph reaches the unreliable modality's GCN; it does not by itself establish a diagnostic gain. The `val=test` checkpoint-selection rule, CV-before-fold preprocessing, and transductive 871-node graphs can make estimates optimistic. This single-seed, shared-cohort comparison is exploratory, not an unbiased external generalization estimate.",
        "",
    ]
    (RESULTS / "report.md").write_text("\n".join(report), encoding="utf-8")

    source_hashes = {
        "config.json": sha_file(CFG_PATH),
        "run.py": sha_file(EXP / "run.py"),
        "cadr_gfr.py": sha_file(EXP / "cadr_gfr.py"),
        "run_cadr_fallback.py": sha_file(Path(__file__)),
    }
    runtime_manifest = {
        "experiment": "53.3",
        "status": "COMPLETE_AUDITED" if len(audited) == 10 and all(r["checkpoint_audit"] == "PASS" for r in audited) else "AUDIT_INCOMPLETE",
        "group": GROUP,
        "seed": 0,
        "required_folds": 10,
        "completed_folds": len(audited),
        "independent_checkpoint_audits_passed": sum(r["checkpoint_audit"] == "PASS" for r in audited),
        "cadr_weight_decay": 0.001,
        "base_gfr_weight_decay": 0.11,
        "cam_weight_decay": 0.11,
        "added_optimizer_groups_weight_decay": [0.001, 0.001],
        "reference_experiment_53_2_status": reference_manifest["status"],
        "reference_checkpoints_read_only": True,
        "relation_cache_source": "Experiment 53.2 .runtime/relation_cache, per-fold JSON and NPZ SHA verified",
        "data_sha256": reference_manifest["source_manifest"]["data_sha256"],
        "anchor_sha256": reference_manifest["source_manifest"]["anchor_sha256"],
        "experiment_53_2_source_sha256": reference_manifest["source_manifest"]["experiment_source_sha256"],
        "relation_cache_sha256": reference_manifest["source_manifest"]["relation_cache_sha256"],
        "source_sha256": source_hashes,
        "result_sha256": {path.name: sha_file(path) for path in RESULTS.iterdir() if path.is_file()},
        "folds": [{"fold": int(row["fold"]), "checkpoint_sha256": row["checkpoint_sha256"], "checkpoint_audit": row["checkpoint_audit"], "split_sha": row["split_sha"], "pair_sha": row["pair_sha"], "initial_shared_sha": row["initial_shared_sha"], "initial_added_sha": row["initial_added_sha"], "initial_rng_sha": row["initial_rng_sha"]} for row in audited],
        "handoff": "PENDING",
    }
    (EXP / "manifest.json").write_text(json.dumps(runtime_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_all(max_workers=2):
    manifest, reference = load_reference()
    preflight(reference)
    jobs = [(item["name"], fold) for item in SWEEP_CFG["groups"] for fold in range(1, 11)]
    pending = []
    for group, fold in jobs:
        tp, cp = task_path(group, fold), best_path(group, fold)
        if tp.exists() and cp.exists():
            task = json.loads(tp.read_text())
            if task.get("status") == "COMPLETE" and task.get("best_sha256") == sha_file(cp):
                continue
        pending.append((group, fold))
    active = {}
    max_workers = max(1, min(int(max_workers), 2))
    while pending or active:
        while pending and len(active) < max_workers:
            group, fold = pending.pop(0)
            _, _, _, logs = group_paths(group)
            logs.mkdir(parents=True, exist_ok=True)
            log_handle = (logs / f"{group}_f{fold:02d}.log").open("ab")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["CADR_EXP"] = str(EXP)
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__)), "--worker", "--group", group, "--fold", str(fold)],
                cwd=str(ROOT), env=env, stdout=log_handle, stderr=subprocess.STDOUT,
            )
            active[process] = (group, fold, log_handle)
        completed = []
        for process, (group, fold, log_handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log_handle.close()
            if code != 0:
                log_path = group_paths(group)[3] / f"{group}_f{fold:02d}.log"
                tail = log_path.read_text(errors="replace")[-10000:]
                raise RuntimeError(f"{group} fold {fold} failed ({code}):\n{tail}")
            completed.append(process)
            task = json.loads(task_path(group, fold).read_text())
            task["group"] = group
            task["cadr_weight_decay"] = float(group_config(group)["cadr_weight_decay"])
            task["cadr_optimizer_weight_decay"] = [float(group_config(group)["cadr_weight_decay"])] * len(task["optimizer_weight_decay"]["added"])
            R.atomic_json(task_path(group, fold), task)
            print(f"trained {group} fold {fold}/10", flush=True)
        for process in completed:
            active.pop(process, None)
        time.sleep(0.5)

    audited = []
    diagnostics = []
    for item in SWEEP_CFG["groups"]:
        group = item["name"]
        for fold in range(1, 11):
            row, fold_diagnostics = audit_fold(group, fold, reference)
            audited.append(row)
            diagnostics.extend(fold_diagnostics)
            print(f"independent checkpoint reads {len(audited)}/10", flush=True)
    finalize_results(audited, diagnostics, reference, manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    if args.worker:
        if not args.group or args.fold is None:
            raise SystemExit("--worker requires --group and --fold")
        worker(args.group, args.fold)
    elif args.preflight:
        _, reference = load_reference()
        preflight(reference)
        print("Experiment 53.3 fallback preflight passed; all ten initial states match Experiment 53.2", flush=True)
    elif args.run:
        run_all(args.max_workers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
