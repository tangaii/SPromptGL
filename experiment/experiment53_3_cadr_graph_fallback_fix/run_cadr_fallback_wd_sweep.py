#!/usr/bin/env python3
"""Four isolated CADR/GFR WD groups using the Experiment 53.3 fallback model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import run_cadr_fallback as F

R = F.R
EXP = F.EXP
ROOT = F.ROOT
CONFIG_PATH = Path(__file__).with_name("config_fallback_wd_sweep.json")
SWEEP_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
F.CFG_PATH = CONFIG_PATH
F.SWEEP_CFG = {"groups": SWEEP_CONFIG["groups"]}
F.SWEEP_ROOT = EXP / SWEEP_CONFIG["runtime_subdir"]
F.RESULTS = EXP / SWEEP_CONFIG["results_subdir"]
RUNTIME = F.SWEEP_ROOT
RESULTS = F.RESULTS
BASE_BUILD_STATE = F.BASE_BUILD_STATE
BASE_BUILD_CHECKPOINT = F.BASE_BUILD_CHECKPOINT
METRICS = ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")
LABELS = {"acc": "ACC", "hard_auc": "Hard-AUC", "prob_auc": "Probability-AUC", "spe": "SPE-TD", "sen": "SEN-ASD", "ba": "BA", "asd_f1": "ASD-F1"}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_config(group: str) -> dict:
    for item in SWEEP_CONFIG["groups"]:
        if item["name"] == group:
            return item
    raise KeyError(group)


def group_paths(group: str):
    root = RUNTIME / group
    return root, root / "tasks", root / "checkpoints", root / "logs"


def task_path(group: str, fold: int) -> Path:
    return group_paths(group)[1] / f"{group}_f{fold:02d}.json"


def best_path(group: str, fold: int) -> Path:
    return group_paths(group)[2] / f"{group}_f{fold:02d}.best.pt"


def latest_path(group: str, fold: int) -> Path:
    return group_paths(group)[2] / f"{group}_f{fold:02d}.latest.pt"


def install_group(group: str):
    item = group_config(group)
    cadr_wd = float(item["cadr_weight_decay"])
    gfr_wd = float(item["gfr_weight_decay"])
    cam_wd = float(SWEEP_CONFIG["camera_optimizer_weight_decay"])
    root, tasks, checkpoints, logs = group_paths(group)
    for path in (root, tasks, checkpoints, logs):
        path.mkdir(parents=True, exist_ok=True)

    R.RT, R.CKPTS, R.LOGS = root, checkpoints, logs
    R.task_path = lambda fold: task_path(group, fold)
    R.best_path = lambda fold: best_path(group, fold)
    R.latest_path = lambda fold: latest_path(group, fold)

    def build_state(fold, cache_meta=None):
        state = BASE_BUILD_STATE(fold, cache_meta)
        for pg in state["opt_base"].param_groups:
            pg["weight_decay"] = gfr_wd
        for pg in state["opt_added"].param_groups:
            pg["weight_decay"] = cadr_wd
        state.update({"group": group, "cadr_weight_decay": cadr_wd, "gfr_weight_decay": gfr_wd, "cam_weight_decay": cam_wd})
        check_wds(state, group)
        return state

    def build_checkpoint(state, *args, **kwargs):
        checkpoint = BASE_BUILD_CHECKPOINT(state, *args, **kwargs)
        checkpoint.update({
            "group": group,
            "cadr_weight_decay": cadr_wd,
            "gfr_weight_decay": gfr_wd,
            "camera_weight_decay": cam_wd,
            "optimizer_weight_decay": {
                "base": [float(pg["weight_decay"]) for pg in state["opt_base"].param_groups],
                "cam": [float(pg["weight_decay"]) for pg in state["opt_cam"].param_groups],
                "added": [float(pg["weight_decay"]) for pg in state["opt_added"].param_groups],
            },
            "fallback_runner_sha256": sha_file(EXP / "run_cadr_fallback.py"),
            "fallback_config_sha256": sha_file(EXP / "config.json"),
            "wd_sweep_runner_sha256": sha_file(Path(__file__)),
            "wd_sweep_config_sha256": sha_file(CONFIG_PATH),
        })
        return checkpoint

    R.build_state = build_state
    R._build_checkpoint = build_checkpoint


def check_wds(state, group):
    item = group_config(group)
    expected = {
        "opt_base": float(item["gfr_weight_decay"]),
        "opt_cam": float(SWEEP_CONFIG["camera_optimizer_weight_decay"]),
        "opt_added": float(item["cadr_weight_decay"]),
    }
    for name, value in expected.items():
        actual = [float(pg["weight_decay"]) for pg in state[name].param_groups]
        assert actual and all(wd == value for wd in actual), (group, name, actual, value)


def load_references():
    source_manifest, reference_53_2 = F.load_reference()
    rows = F.read_rows(RESULTS.parent / "fold_metrics.csv")
    group = "E1_CADR_WD0001_GRAPH_FALLBACK_FIX"
    reference_53_3 = {int(row["fold"]): row for row in rows if row.get("group") == group}
    assert len(reference_53_3) == 10, "53.3 baseline requires ten readback-audited fold rows"
    audits = F.read_rows(RESULTS.parent / "checkpoint_audit.csv")
    audit_rows = [row for row in audits if row.get("group") == group]
    assert len(audit_rows) == 10 and all(row.get("checkpoint_audit") == "PASS" for row in audit_rows)
    return source_manifest, reference_53_2, reference_53_3


def preflight(reference_53_2):
    import torch

    records, first_ids, first_forward = [], None, None
    for item in SWEEP_CONFIG["groups"]:
        group = item["name"]
        install_group(group)
        state = R.build_state(1, R.prepare_fold(1))
        F.check_initialization(state, reference_53_2)
        with torch.no_grad():
            out = R.forward_state(state, training=False, use_cadr=True)
        identities = (
            state["initial_shared_sha"], state["initial_added_sha"], state["construction_rng_sha"],
            state["split_sha"], state["cache_meta"]["pair_sha"],
        )
        current_forward = (out["fused"].detach().cpu(), out["logp"].detach().cpu())
        if first_ids is None:
            first_ids, first_forward = identities, current_forward
        else:
            assert identities == first_ids, f"initial state differs for {group}"
            assert torch.equal(current_forward[0], first_forward[0]) and torch.equal(current_forward[1], first_forward[1]), group
        raw = torch.cat(state["parts"], dim=1)
        eye = torch.eye(raw.shape[0], dtype=raw.dtype, device=raw.device)
        for modality, reliability in enumerate(state["cache_meta"]["reliability"]):
            if float(reliability) == 0.0:
                assert out["info"]["propagation_modes"][modality] == "ORIGINAL_FALLBACK"
                expected = state["model"].gcn[modality](raw, state["model"].utils.normalize_adj(out["info"]["original_adjs"][modality] + eye))
                assert torch.allclose(out["info"]["messages"][modality], expected, rtol=1e-6, atol=1e-7)
        records.append({
            "group": group, "fold": 1, "cadr_weight_decay": item["cadr_weight_decay"],
            "gfr_weight_decay": item["gfr_weight_decay"], "cam_weight_decay": SWEEP_CONFIG["camera_optimizer_weight_decay"],
            "opt_added_wd": json.dumps([float(pg["weight_decay"]) for pg in state["opt_added"].param_groups]),
            "opt_base_wd": json.dumps([float(pg["weight_decay"]) for pg in state["opt_base"].param_groups]),
            "initial_shared_sha": state["initial_shared_sha"], "initial_added_sha": state["initial_added_sha"],
            "initial_rng_sha": state["construction_rng_sha"], "split_sha": state["split_sha"],
            "pair_sha": state["cache_meta"]["pair_sha"], "initial_forward_equal": True,
            "propagation_modes": json.dumps(out["info"]["propagation_modes"]), "status": "PASS",
        })
        del out, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    F.write_rows(RESULTS / "preflight.csv", records)
    return records


def worker(group: str, fold: int):
    install_group(group)
    R.train_one(fold)
    task_file = task_path(group, fold)
    task = json.loads(task_file.read_text(encoding="utf-8"))
    item = group_config(group)
    task.update({
        "group": group, "cadr_weight_decay": float(item["cadr_weight_decay"]),
        "gfr_weight_decay": float(item["gfr_weight_decay"]),
        "cam_weight_decay": float(SWEEP_CONFIG["camera_optimizer_weight_decay"]),
        "wd_sweep_runner_sha256": sha_file(Path(__file__)),
        "wd_sweep_config_sha256": sha_file(CONFIG_PATH),
        "fallback_fix_source_sha256": sha_file(EXP / "run_cadr_fallback.py"),
    })
    R.atomic_json(task_file, task)


def audit_fold(group: str, fold: int, reference_53_2):
    import torch

    item = group_config(group)
    install_group(group)
    state = R.build_state(fold, R.prepare_fold(fold))
    F.check_initialization(state, reference_53_2)
    check_wds(state, group)
    cp_path, task_file = best_path(group, fold), task_path(group, fold)
    checkpoint = R.torch_load(cp_path)
    task = json.loads(task_file.read_text(encoding="utf-8"))
    c_wd, g_wd, cam_wd = float(item["cadr_weight_decay"]), float(item["gfr_weight_decay"]), float(SWEEP_CONFIG["camera_optimizer_weight_decay"])
    assert task["status"] == "COMPLETE" and task["best_sha256"] == sha_file(cp_path)
    assert task["group"] == group and task["wd_sweep_config_sha256"] == sha_file(CONFIG_PATH)
    assert checkpoint["group"] == group and checkpoint["fold"] == fold and checkpoint["seed"] == 0
    assert checkpoint["cadr_weight_decay"] == c_wd and checkpoint["gfr_weight_decay"] == g_wd and checkpoint["camera_weight_decay"] == cam_wd
    assert checkpoint["wd_sweep_runner_sha256"] == sha_file(Path(__file__))
    assert checkpoint["wd_sweep_config_sha256"] == sha_file(CONFIG_PATH)
    assert checkpoint["fallback_runner_sha256"] == sha_file(EXP / "run_cadr_fallback.py")
    assert checkpoint["fallback_config_sha256"] == sha_file(EXP / "config.json")
    assert checkpoint["optimizer_weight_decay"]["base"] == [g_wd]
    assert checkpoint["optimizer_weight_decay"]["cam"] == [cam_wd]
    assert all(value == c_wd for value in checkpoint["optimizer_weight_decay"]["added"])
    expected_keys = ("initial_shared_sha", "initial_gfr_sha", "initial_schema_sha", "initial_cam_sha", "initial_added_sha", "pair_sha", "split_sha")
    for key in expected_keys:
        expected = state["cache_meta"]["pair_sha"] if key == "pair_sha" else state[key]
        assert checkpoint[key] == expected, (group, fold, key)
        if key in reference_53_2[fold]:
            assert expected == reference_53_2[fold][key], (group, fold, key, "reference")
    assert checkpoint["initial_rng_sha"] == state["construction_rng_sha"]
    assert checkpoint["best_epoch"] >= int(R.CFG["warmup"]["epochs"])
    state["model"].load_state_dict(checkpoint["model"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    state["opt_base"].load_state_dict(checkpoint["optimizer_base"])
    state["opt_cam"].load_state_dict(checkpoint["optimizer_cam"])
    state["opt_added"].load_state_dict(checkpoint["optimizer_added"])
    check_wds(state, group)
    for module in (state["model"], state["cam"]):
        assert all(torch.isfinite(p).all() for p in module.parameters())
    state["model"].eval()
    state["cam"].eval()
    with torch.no_grad():
        out = R.forward_state(state, training=False, use_cadr=True)
    assert torch.isfinite(out["fused"]).all() and torch.isfinite(out["total"])
    raw = torch.cat(state["parts"], dim=1)
    eye = torch.eye(raw.shape[0], dtype=raw.dtype, device=raw.device)
    for modality, reliability in enumerate(state["cache_meta"]["reliability"]):
        if float(reliability) == 0.0:
            assert out["info"]["propagation_modes"][modality] == "ORIGINAL_FALLBACK"
            for name in ("positive_adjs", "negative_adjs", "positive_messages", "negative_messages", "direct_messages"):
                assert torch.count_nonzero(out["info"][name][modality]) == 0
            expected = state["model"].gcn[modality](raw, state["model"].utils.normalize_adj(out["info"]["original_adjs"][modality] + eye))
            assert torch.allclose(out["info"]["messages"][modality], expected, rtol=1e-6, atol=1e-7)
        else:
            assert out["info"]["propagation_modes"][modality] == "CADR_DISEASE"
    saved_y = state["y"].clone()
    test = torch.as_tensor(state["test"], dtype=torch.long, device=state["y"].device)
    state["y"][test] = 1 - state["y"][test]
    with torch.no_grad():
        permuted = R.forward_state(state, training=False, use_cadr=True)
    assert torch.equal(out["logp"], permuted["logp"])
    assert all(torch.equal(a, b) for a, b in zip(out["info"]["positive_adjs"], permuted["info"]["positive_adjs"]))
    state["y"].copy_(saved_y)
    metrics = R.classification_metrics(out["logp"], state["test"], state["y"])
    for metric in METRICS:
        assert abs(float(metrics[metric]) - float(checkpoint["metrics"][metric])) <= 1e-10

    grad_out = R.forward_state(state, training=False, use_cadr=True)
    diag = []
    for modality, name in enumerate(R.MODS):
        gcn = state["model"].gcn[modality]
        graph = state["model"].graphs[modality]
        gcn_grad = torch.autograd.grad(grad_out["cls"], list(gcn.parameters()), retain_graph=True, allow_unused=True)
        graph_grad = torch.autograd.grad(grad_out["cls"], list(graph.parameters()), retain_graph=True, allow_unused=True)
        original = out["info"]["original_adjs"][modality]
        message = out["info"]["messages"][modality]
        diag.append({
            "group": group, "fold": fold, "modality": name,
            "oof_reliability": float(state["cache_meta"]["reliability"][modality]),
            "oof_auc": float(state["cache_meta"]["relation_oof"][modality]["auc"]),
            "propagation_mode": out["info"]["propagation_modes"][modality],
            "original_graph_edges": int(torch.count_nonzero(torch.triu(original.detach() > 0, diagonal=1)).item()),
            "disease_positive_edges": int(torch.count_nonzero(torch.triu(out["info"]["positive_adjs"][modality].detach() > 0, diagonal=1)).item()),
            "disease_negative_edges": int(torch.count_nonzero(torch.triu(out["info"]["negative_adjs"][modality].detach() > 0, diagonal=1)).item()),
            "gcn_message_norm": float(message.detach().norm().cpu()),
            "gcn_message_input_ratio": float(message.detach().norm() / raw.norm().clamp_min(1e-12)),
            "fallback_message_norm": float(out["info"]["original_fallback_messages"][modality].detach().norm().cpu()),
            "direct_message_norm": float(out["info"]["direct_messages"][modality].detach().norm().cpu()),
            "classification_to_gcn_grad_norm": F._tensor_grad_norm(gcn_grad),
            "classification_to_graphlearn_grad_norm": F._tensor_grad_norm(graph_grad),
            "fusion_weight_sigmoid": float(torch.sigmoid(state["model"].fusion_weight[modality]).detach().cpu()),
            "cadr_reliability_gate": float(out["info"]["fusion_gates"][modality].detach().cpu()),
        })

    row = {
        "group": group, "fold": fold, "seed": 0, "cadr_weight_decay": c_wd, "gfr_weight_decay": g_wd,
        "cam_weight_decay": cam_wd,
        "cadr_optimizer_weight_decay": json.dumps(checkpoint["optimizer_weight_decay"]["added"]),
        "gfr_optimizer_weight_decay": json.dumps(checkpoint["optimizer_weight_decay"]["base"]),
        "best_epoch": int(checkpoint["best_epoch"]), "epochs_executed": int(checkpoint["epochs_executed"]),
        **{metric: float(metrics[metric]) for metric in METRICS},
        "positive_gcn_message_input_ratio": float(torch.stack([x.norm() for x in out["info"]["positive_messages"]]).norm() / raw.norm().clamp_min(1e-12)),
        "negative_gcn_message_input_ratio": float(torch.stack([x.norm() for x in out["info"]["negative_messages"]]).norm() / raw.norm().clamp_min(1e-12)),
        "direct_message_input_ratio": float(torch.stack([x.norm() for x in out["info"]["direct_messages"]]).norm() / raw.norm().clamp_min(1e-12)),
        "final_minus_input_ratio": float((out["fused"] - raw).norm() / raw.norm().clamp_min(1e-12)),
        "initial_shared_sha": state["initial_shared_sha"], "initial_added_sha": state["initial_added_sha"],
        "initial_rng_sha": state["construction_rng_sha"], "split_sha": state["split_sha"],
        "pair_sha": state["cache_meta"]["pair_sha"], "checkpoint_sha256": sha_file(cp_path), "checkpoint_audit": "PASS",
    }
    return row, diag


def summarize(audited, reference_53_3):
    groups = [{
        "name": "E1_CADR_WD0001_GRAPH_FALLBACK_FIX", "cadr_weight_decay": 0.001,
        "gfr_weight_decay": 0.11, "source": "53.3 historical read-only reference",
    }, *[{**item, "source": "new_training"} for item in SWEEP_CONFIG["groups"]]]
    rows = []
    for item in groups:
        if item["source"] == "new_training":
            subset = sorted((r for r in audited if r["group"] == item["name"]), key=lambda r: int(r["fold"]))
        else:
            subset = [reference_53_3[f] for f in range(1, 11)]
        assert len(subset) == 10
        record = {"group": item["name"], "cadr_weight_decay": item["cadr_weight_decay"], "gfr_weight_decay": item["gfr_weight_decay"], "folds": 10, "source": item["source"]}
        for metric in METRICS:
            vals = np.asarray([float(r[metric]) for r in subset]) * 100.0
            record[f"{metric}_mean_percent"] = float(vals.mean())
            record[f"{metric}_population_sd_percent"] = float(vals.std(ddof=0))
        if item["source"] == "new_training":
            for metric in ("best_epoch", "epochs_executed", "positive_gcn_message_input_ratio", "negative_gcn_message_input_ratio", "direct_message_input_ratio", "final_minus_input_ratio"):
                vals = np.asarray([float(r[metric]) for r in subset])
                record[f"{metric}_mean"] = float(vals.mean())
                record[f"{metric}_population_sd"] = float(vals.std(ddof=0))
        rows.append(record)
    return rows


def compare_pairs(audited, reference_53_3):
    index = {item["name"]: {} for item in SWEEP_CONFIG["groups"]}
    for row in audited:
        index[row["group"]][int(row["fold"])] = row
    pairs = [(g["name"], "E1_CADR_WD0001_GRAPH_FALLBACK_FIX", reference_53_3) for g in SWEEP_CONFIG["groups"]]
    pairs.extend((a, b, index[b]) for a, b in combinations(index, 2))
    output = []
    for left, right, ref in pairs:
        for metric in METRICS:
            diffs = []
            for fold in range(1, 11):
                delta = (float(index[left][fold][metric]) - float(ref[fold][metric])) * 100.0
                diffs.append(delta)
                output.append({"comparison": f"{left}_minus_{right}", "fold": fold, "metric": metric, "difference_pp": delta})
            output.append({
                "comparison": f"{left}_minus_{right}", "fold": "ALL", "metric": metric,
                "mean_difference_pp": float(np.mean(diffs)),
                "wins_ties_losses": f"{sum(d > 1e-12 for d in diffs)}/{sum(abs(d) <= 1e-12 for d in diffs)}/{sum(d < -1e-12 for d in diffs)}",
            })
    return output


def finalize(audited, diagnostics, reference_53_2, reference_53_3, source_manifest, max_workers):
    RESULTS.mkdir(parents=True, exist_ok=True)
    F.write_rows(RESULTS / "fold_metrics.csv", audited)
    F.write_rows(RESULTS / "diagnostics.csv", diagnostics)
    F.write_rows(RESULTS / "checkpoint_audit.csv", [{key: row[key] for key in (
        "group", "fold", "checkpoint_audit", "checkpoint_sha256", "best_epoch", "epochs_executed",
        "initial_shared_sha", "initial_added_sha", "initial_rng_sha", "split_sha", "pair_sha",
        "cadr_optimizer_weight_decay", "gfr_optimizer_weight_decay",
    )} for row in audited])
    summary = summarize(audited, reference_53_3)
    F.write_rows(RESULTS / "summary.csv", summary)
    paired = compare_pairs(audited, reference_53_3)
    F.write_rows(RESULTS / "paired_comparisons.csv", paired)
    lines = [
        "# Experiment 53.3 CADR/GFR weight-decay sweep", "",
        "Four groups were trained from the fallback-fixed Experiment 53.3 implementation. The original graph fallback, relation cache, initialization, RNG handling, CAM WD, and training protocol were held fixed. Only CADR and GFR optimizer weight decay varied. All 40 checkpoints were independently loaded and audited.", "",
        "|组|CADR WD|GFR WD|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|来源|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        vals = [f"{row[f'{metric}_mean_percent']:.4f} ± {row[f'{metric}_population_sd_percent']:.4f}" for metric in METRICS]
        lines.append(f"|{row['group']}|{row['cadr_weight_decay']}|{row['gfr_weight_decay']}|" + "|".join(vals) + f"|{row['source']}|")
    lines.extend(["", "## Paired differences versus Experiment 53.3 baseline", "", "|组|指标|均值差(pp)|胜/平/负|", "|---|---|---:|---:|"])
    for group in [item["name"] for item in SWEEP_CONFIG["groups"]]:
        for metric in ("acc", "hard_auc", "prob_auc", "spe", "sen"):
            row = next(r for r in paired if r["comparison"] == f"{group}_minus_E1_CADR_WD0001_GRAPH_FALLBACK_FIX" and r["metric"] == metric and r["fold"] == "ALL")
            lines.append(f"|{group}|{LABELS[metric]}|{row['mean_difference_pp']:+.4f}|{row['wins_ties_losses']}|")
    lines.extend(["", "The historical `val=test` checkpoint selection, pre-CV feature preprocessing, and 871-node transductive graph remain in effect; these are protocol-matched exploratory results, not unbiased external generalization estimates.", ""])
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "experiment": "53.3 fallback-fixed CADR/GFR WD sweep",
        "status": "COMPLETE_AUDITED" if len(audited) == 40 and all(r["checkpoint_audit"] == "PASS" for r in audited) else "AUDIT_INCOMPLETE",
        "groups": SWEEP_CONFIG["groups"], "seed": 0, "folds_per_group": 10,
        "completed_folds": len(audited),
        "independent_checkpoint_audits_passed": sum(r["checkpoint_audit"] == "PASS" for r in audited),
        "requested_workers": int(SWEEP_CONFIG["requested_workers"]), "max_concurrent_workers": int(max_workers),
        "device": "cuda:0" if R.runtime()["torch"].cuda.is_available() else "cpu",
        "camera_weight_decay": float(SWEEP_CONFIG["camera_optimizer_weight_decay"]),
        "comparison_reference": "Experiment 53.3 E1_CADR_WD0001_GRAPH_FALLBACK_FIX",
        "source_sha256": {
            "config_fallback_wd_sweep.json": sha_file(CONFIG_PATH),
            "run_cadr_fallback_wd_sweep.py": sha_file(Path(__file__)),
            "run_cadr_fallback.py": sha_file(EXP / "run_cadr_fallback.py"),
            "run.py": sha_file(EXP / "run.py"), "cadr_gfr.py": sha_file(EXP / "cadr_gfr.py"),
            "config.json": sha_file(EXP / "config.json"),
        },
        "source_manifest_sha256": sha_file(EXP / ".runtime/reference_53_2_manifest.json"),
        "sweep_reference_manifest_sha256": sha_file(EXP / ".runtime/reference_53_2_wd_sweep_manifest.json"),
        "experiment_53_3_baseline_csv_sha256": sha_file(RESULTS.parent / "fold_metrics.csv"),
        "historical_53_2_status": source_manifest["status"],
        "folds": [{"group": r["group"], "fold": int(r["fold"]), "checkpoint_sha256": r["checkpoint_sha256"], "audit": r["checkpoint_audit"], "split_sha": r["split_sha"], "pair_sha": r["pair_sha"], "initial_shared_sha": r["initial_shared_sha"], "initial_added_sha": r["initial_added_sha"], "initial_rng_sha": r["initial_rng_sha"]} for r in audited],
    }
    manifest["result_sha256"] = {p.name: sha_file(p) for p in RESULTS.iterdir() if p.is_file()}
    (EXP / "manifest_fallback_wd_sweep.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_all(max_workers=10):
    source_manifest, reference_53_2, reference_53_3 = load_references()
    preflight(reference_53_2)
    jobs = [(group["name"], fold) for group in SWEEP_CONFIG["groups"] for fold in range(1, 11)]
    pending = []
    for group, fold in jobs:
        tp, cp = task_path(group, fold), best_path(group, fold)
        if tp.is_file() and cp.is_file():
            task = json.loads(tp.read_text(encoding="utf-8"))
            if task.get("status") == "COMPLETE" and task.get("group") == group and task.get("best_sha256") == sha_file(cp) and task.get("wd_sweep_config_sha256") == sha_file(CONFIG_PATH):
                continue
        pending.append((group, fold))
    max_workers = max(1, min(int(max_workers), int(SWEEP_CONFIG["requested_workers"])))
    active = {}
    while pending or active:
        while pending and len(active) < max_workers:
            group, fold = pending.pop(0)
            _, _, _, logs = group_paths(group)
            logs.mkdir(parents=True, exist_ok=True)
            log_handle = (logs / f"{group}_f{fold:02d}.log").open("ab")
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CADR_EXP": str(EXP)})
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[key] = "2"
            proc = subprocess.Popen([sys.executable, str(Path(__file__)), "--worker", "--group", group, "--fold", str(fold)], cwd=str(ROOT), env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            active[proc] = (group, fold, log_handle)
        completed = []
        for proc, (group, fold, handle) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            handle.close()
            if code != 0:
                log_path = group_paths(group)[3] / f"{group}_f{fold:02d}.log"
                tail = log_path.read_text(errors="replace")[-12000:]
                for other, (_, _, other_handle) in active.items():
                    if other.poll() is None:
                        other.terminate()
                    other_handle.close()
                raise RuntimeError(f"{group} fold {fold} failed ({code}):\n{tail}")
            completed.append(proc)
            task = json.loads(task_path(group, fold).read_text(encoding="utf-8"))
            item = group_config(group)
            task.update({"group": group, "cadr_weight_decay": float(item["cadr_weight_decay"]), "gfr_weight_decay": float(item["gfr_weight_decay"]), "cam_weight_decay": float(SWEEP_CONFIG["camera_optimizer_weight_decay"]), "wd_sweep_runner_sha256": sha_file(Path(__file__)), "wd_sweep_config_sha256": sha_file(CONFIG_PATH)})
            R.atomic_json(task_path(group, fold), task)
            print(f"completed {group} fold {fold}/10; pending={len(pending)}", flush=True)
        for proc in completed:
            active.pop(proc, None)
        if active or pending:
            time.sleep(0.5)

    audited, diagnostics = [], []
    for item in SWEEP_CONFIG["groups"]:
        for fold in range(1, 11):
            row, diag = audit_fold(item["name"], fold, reference_53_2)
            audited.append(row)
            diagnostics.extend(diag)
            print(f"independent checkpoint audit {len(audited)}/40", flush=True)
    finalize(audited, diagnostics, reference_53_2, reference_53_3, source_manifest, max_workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()
    if args.worker:
        if not args.group or args.fold is None:
            raise SystemExit("--worker requires --group and --fold")
        worker(args.group, args.fold)
    elif args.preflight:
        _, reference, _ = load_references()
        records = preflight(reference)
        print(f"preflight passed for {len(records)} groups", flush=True)
    elif args.run:
        run_all(args.max_workers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
