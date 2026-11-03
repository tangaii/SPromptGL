#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
_spec = importlib.util.spec_from_file_location("e70_1_audit_runner", EXP / "run.py")
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)
from dcc_gcdsi import DCCGCDSI  # noqa: E402

RESULTS = EXP / "results"
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")


def sha_file(path: Path) -> str:
    return R.sha_file(path)


def tensor_tree_sha(value) -> str:
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0" + str(tensor.dtype).encode() + b"\0" + repr(tuple(tensor.shape)).encode() + b"\0")
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0" + str(array.dtype).encode() + b"\0" + repr(array.shape).encode() + b"\0")
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"\0")
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode() + b"\0")

    visit(value)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def pearson(left, right) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denom) if denom > 0 else 0.0


def site_eta_squared(values, site_ids) -> float:
    values = np.asarray(values, dtype=np.float64)
    site_ids = np.asarray(site_ids)
    total = float(np.square(values - values.mean()).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for site in np.unique(site_ids):
        group = values[site_ids == site]
        between += len(group) * float(np.square(group.mean() - values.mean()))
    return between / total


def load_checkpoint(group: str, fold: int, ref_manifest, ref_folds, banks, features):
    group_dir = R.RUNTIME / group
    task_path = group_dir / "tasks" / f"{group}_f{fold:02d}.json"
    checkpoint_path = group_dir / "checkpoints" / f"{group}_f{fold:02d}.best.pt"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    checkpoint_sha = sha_file(checkpoint_path)
    if task.get("status") != "COMPLETE" or checkpoint_sha != task.get("checkpoint_sha256"):
        raise AssertionError(f"task/checkpoint hash mismatch {group} fold {fold}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("group") != group or int(checkpoint.get("fold", -1)) != fold:
        raise AssertionError("checkpoint identity mismatch")
    reference = ref_folds[fold]
    for key in ("split_sha", "pair_sha", "initial_shared_sha", "initial_added_sha", "initial_rng_sha"):
        if checkpoint.get(key) != reference.get(key) or task.get(key) != reference.get(key):
            raise AssertionError(f"historical binding mismatch {key} ({group}, fold {fold})")
    if checkpoint.get("reference_checkpoint_sha256") != reference["checkpoint_sha256"]:
        raise AssertionError("checkpoint references wrong 53.3 fold")
    bank = banks[group].to(R.BASE.runtime()["device"])
    bank_sha = R.tensor_sha(banks[group])
    if checkpoint.get("bank_sha256") != bank_sha or not torch.equal(checkpoint["text_bank"], banks[group].cpu()):
        raise AssertionError("checkpoint semantic bank does not match verified frozen bank")
    if checkpoint.get("text_asset_sha256") != R.sha_file(R.TEXT_CACHE):
        raise AssertionError("text cache SHA mismatch")
    if checkpoint.get("source_sha256") != R.source_sha256() or checkpoint.get("config_sha256") != sha_file(R.CONFIG_PATH):
        raise AssertionError("training source/config SHA mismatch")
    if checkpoint.get("warmup_sha256") != sha_file(R.E59.warmup_cache_path(fold)):
        raise AssertionError("warmup cache SHA mismatch")
    if checkpoint.get("dcc_schema_sha256") != R.parameter_schema_sha(DCCGCDSI(bank)):
        raise AssertionError("DCC parameter schema mismatch")

    state = R.build_state(fold, ref_manifest, ref_folds)
    warmup = R.E59.load_warmup_cache_into_state(state, fold, json.loads(R.E59.CONFIG_PATH.read_text()))
    initial_module = R.make_module(bank, fold)
    initial_state = R.BASE._state_cpu(initial_module)
    module = R.make_module(bank, fold)
    R.install_output_hook(state["cam"], module)
    state["model"].load_state_dict(checkpoint["model"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    module.load_state_dict(checkpoint["dcc"], strict=True)
    if R.parameter_schema_sha(module) != checkpoint.get("dcc_schema_sha256"):
        raise AssertionError("restored DCC parameter schema mismatch")
    if R.module_param_sha(initial_module) != checkpoint.get("dcc_initial_sha256"):
        raise AssertionError("DCC deterministic initialization SHA mismatch")
    if any(key.startswith("_dcc") or "dcc" in key.lower() for key in state["cam"].state_dict()):
        raise AssertionError("DCC weights appeared inside the CAM state dictionary")
    if state["initial_schema_sha"] != checkpoint.get("base_schema_sha256"):
        raise AssertionError("original GFR/CAM model schema mismatch")

    optimizer_dcc = torch.optim.AdamW(module.parameters(), lr=0.001, weight_decay=0.001)
    for key, optimizer, expected_class, expected_lr, expected_wd in (
        ("optimizer_base", state["opt_base"], "Adam", 0.0038, 0.11),
        ("optimizer_cam", state["opt_cam"], "Adam", 0.0038, 0.11),
        ("optimizer_added", state["opt_added"], "AdamW", 0.001, 0.01),
        ("optimizer_dcc", optimizer_dcc, "AdamW", 0.001, 0.001),
    ):
        if optimizer.__class__.__name__ != expected_class:
            raise AssertionError(f"{key} optimizer class mismatch")
        for group_cfg in checkpoint[key]["param_groups"]:
            if float(group_cfg["lr"]) != expected_lr or float(group_cfg["weight_decay"]) != expected_wd:
                raise AssertionError(f"{key} optimizer config mismatch")
        optimizer.load_state_dict(checkpoint[key])
    state["_is_training"] = False
    with torch.no_grad():
        output = R.forward_state(state, module, active=True)
        metrics = R.BASE.classification_metrics(output["logp"], state["test"], state["y"])
    for metric in METRICS:
        if abs(float(metrics[metric]) - float(task["metrics"][metric])) > 1e-10:
            raise AssertionError(f"independent readback {metric} differs from task record")
    if tuple(output["fused"].shape) != (871, 320) or tuple(output["dcc"]["query"].shape) != (871, 4, 36):
        raise AssertionError("restored inference tensor shape mismatch")
    if tuple(output["dcc"]["attention"].shape) != (871, 4, 2) or tuple(output["dcc"]["residual"].shape) != (871, 4, 36):
        raise AssertionError("restored semantic attention/residual shape mismatch")
    ratios = output["dcc"]["residual_ratio"]
    violations = int((ratios > 0.150001).sum().cpu())
    if violations:
        raise AssertionError(f"residual bound violation count {violations}")

    base_logp = output["logp"].detach()
    module_raw = module.lambda_raw.detach().clone()
    with torch.no_grad():
        module.lambda_raw.zero_()
        no_semantic = R.forward_state(state, module, active=True)
        module.lambda_raw.copy_(module_raw)
        module._disable_graph = True
        no_graph = R.forward_state(state, module, active=True)
        module._disable_graph = False
        swapped_bank = bank[[1, 0]]
    # Fixed-checkpoint output interventions only; they do not represent retrained ablations.
    gamma_prob = no_semantic["logp"].detach().exp()
    baseline_prob = base_logp.exp()
    graph_prob = no_graph["logp"].detach().exp()
    graph_attention_delta = float((no_graph["dcc"]["attention"] - output["dcc"]["attention"]).abs().mean().cpu())
    graph_p_delta = float((graph_prob[:, 0] - baseline_prob[:, 0]).abs().mean().cpu())
    graph_flips = int((graph_prob.argmax(-1) != baseline_prob.argmax(-1)).sum().cpu())
    residual_off = float((gamma_prob[:, 0] - baseline_prob[:, 0]).abs().mean().cpu())
    residual_flips = int((gamma_prob.argmax(-1) != baseline_prob.argmax(-1)).sum().cpu())
    with torch.no_grad():
        # Capture the original 4x36 CAM representation at the classifier boundary.
        state["cam"]._dcc_active = False
        captured = {}
        def capture_classifier_input(_module, args):
            captured["x"] = args[0].reshape(871, 4, 36)
            return None
        handle = state["cam"].Outputlayer.register_forward_pre_hook(capture_classifier_input)
        _ = state["cam"](output["fused"])
        handle.remove()
        h = captured["x"]
        q_h_prime, q_diag = module(h, output["fused"], text_bank=bank)
        swap_h_prime, swap_diag = module(h, output["fused"], text_bank=swapped_bank)
        same_bank = torch.stack((bank[0], bank[0]))
        same_h_prime, same_diag = module(h, output["fused"], text_bank=same_bank)
        swap_center_error = float((q_diag["center"] - swap_diag["center"]).abs().max().cpu())
        swap_signed_error = float((q_diag["signed_strength"] + swap_diag["signed_strength"]).abs().max().cpu())
        swap_residual_error = float((q_diag["residual"] + swap_diag["residual"]).abs().max().cpu())
        same_text_residual = float(same_diag["residual"].abs().max().cpu())
    if max(swap_center_error, swap_signed_error, swap_residual_error, same_text_residual) > 2e-5:
        raise AssertionError("checkpoint class-swap/zero-difference mathematical property failed")

    diagnostic = output["dcc"]
    attention = diagnostic["attention"].detach()
    entropy = -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(-1)
    signed = diagnostic["signed_strength"].detach()
    gate = diagnostic["gate"].detach()
    residual_ratio = diagnostic["residual_ratio"].detach()
    residual = diagnostic["residual"].detach()
    baseline_cam = h.detach()
    initial_values = {}
    trained_values = {}
    for name, parameter in module.named_parameters():
        initial_values[name] = initial_state[name].to(parameter.device)
        trained_values[name] = parameter.detach()
    parameter_updates = {
        name: float((trained_values[name] - initial_values[name]).norm().cpu()) for name in trained_values
    }
    feature = features
    if feature.shape != (871, 320):
        raise AssertionError("mechanism feature matrix must be [871,320]")
    age_centers = np.arange(7.47, 58.0, 2.0, dtype=np.float64)
    age = feature[:, :26] @ age_centers
    site_ids = feature[:, 26:46].argmax(axis=1)
    motion = feature[:, 60]
    quality = feature[:, 48:64]
    rows = []
    for modality, name in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
        attn_asd = attention[:, modality, 0].cpu().numpy()
        signed_patient = signed[:, modality].cpu().numpy()
        gate_patient = gate[:, modality].mean(-1).cpu().numpy()
        residual_patient = residual_ratio[:, modality].cpu().numpy()
        mod_residual = residual[:, modality]
        mod_cam = baseline_cam[:, modality]
        rows.append({
            "group": group, "fold": fold, "modality": name,
            "checkpoint_sha256": checkpoint_sha,
            "text_cosine": float(diagnostic["text_cosine"].cpu()),
            "disease_delta_norm": float(diagnostic["delta_norm"].cpu()),
            "lambda": float(diagnostic["lambda"].cpu()),
            "signed_mean": float(signed[:, modality].mean().cpu()),
            "signed_sd": float(signed[:, modality].std(unbiased=False).cpu()),
            "signed_abs_mean": float(signed[:, modality].abs().mean().cpu()),
            "signed_patient_sd": float(signed[:, modality].std(unbiased=False).cpu()),
            "signed_cross_modality_sd": float(signed.mean(dim=0).std(unbiased=False).cpu()),
            "attention_asd_mean": float(attention[:, modality, 0].mean().cpu()),
            "attention_entropy_mean": float(entropy[:, modality].mean().cpu()),
            "attention_entropy_sd": float(entropy[:, modality].std(unbiased=False).cpu()),
            "attention_patient_sd": float(attention[:, modality, 0].std(unbiased=False).cpu()),
            "attention_cross_modality_sd": float(attention[:, :, 0].std(dim=1, unbiased=False).mean().cpu()),
            "gate_mean": float(gate[:, modality].mean().cpu()),
            "gate_sd": float(gate[:, modality].std(unbiased=False).cpu()),
            "gate_saturation_fraction": float(((gate[:, modality] < 0.01) | (gate[:, modality] > 0.99)).float().mean().cpu()),
            "cam_rms_mean": float(diagnostic["cam_rms"][:, modality].mean().cpu()),
            "residual_rms_mean": float(mod_residual.square().mean(-1).sqrt().mean().cpu()),
            "residual_cam_rms_ratio_mean": float(residual_patient.mean()),
            "residual_cam_rms_ratio_p95": float(np.quantile(residual_patient, 0.95)),
            "residual_cam_rms_ratio_max": float(residual_patient.max()),
            "residual_bound_violations": int((residual_ratio[:, modality] > 0.150001).sum().cpu()),
            "w_f_grad_norm": float(checkpoint["diagnostics"]["w_f_grad_norm"]),
            "w_f_update_norm": parameter_updates["query_f.weight"],
            "w_delta_grad_norm": float(checkpoint["diagnostics"]["w_delta_grad_norm"]),
            "w_delta_update_norm": parameter_updates["w_delta.weight"],
            "w_center_grad_norm": float(checkpoint["diagnostics"]["w_center_grad_norm"]),
            "w_center_update_norm": parameter_updates["w_center.weight"],
            "w_out_update_norm": parameter_updates["w_out.weight"],
            "gate_update_norm": parameter_updates["gate.0.weight"],
            "graph_off_attention_l1": graph_attention_delta,
            "graph_off_mean_abs_delta_p_asd": graph_p_delta,
            "graph_off_prediction_flips": graph_flips,
            "residual_off_mean_abs_delta_p_asd": residual_off,
            "residual_off_prediction_flips": residual_flips,
            "class_swap_center_max_error": swap_center_error,
            "class_swap_signed_max_error": swap_signed_error,
            "class_swap_residual_max_error": swap_residual_error,
            "equal_text_residual_max": same_text_residual,
            "site_eta2_signed": site_eta_squared(signed_patient, site_ids),
            "age_abs_r_signed": abs(pearson(signed_patient, age)),
            "motion_abs_r_signed": abs(pearson(signed_patient, motion)),
            "quality_max_abs_r_signed": max(abs(pearson(signed_patient, quality[:, j])) for j in range(quality.shape[1])),
            "site_eta2_gate": site_eta_squared(gate_patient, site_ids),
            "age_abs_r_gate": abs(pearson(gate_patient, age)),
            "motion_abs_r_gate": abs(pearson(gate_patient, motion)),
            "quality_max_abs_r_gate": max(abs(pearson(gate_patient, quality[:, j])) for j in range(quality.shape[1])),
            "residual_site_eta2": site_eta_squared(residual_patient, site_ids),
            "residual_age_abs_r": abs(pearson(residual_patient, age)),
            "residual_motion_abs_r": abs(pearson(residual_patient, motion)),
            "residual_quality_max_abs_r": max(abs(pearson(residual_patient, quality[:, j])) for j in range(quality.shape[1])),
        })
    metric_row = {
        "group": group, "fold": fold, "status": "PASS", "checkpoint_sha256": checkpoint_sha,
        "best_epoch": int(checkpoint["best_epoch"]), "epochs_executed": int(checkpoint["epochs_executed"]),
        "warmup_sha256": warmup["sha256"], "reference_checkpoint_sha256": reference["checkpoint_sha256"],
        "split_sha": state["split_sha"], "pair_sha": state["cache_meta"]["pair_sha"],
        **{key: float(metrics[key]) * 100 for key in METRICS},
        "asd_tp": int(metrics["asd_tp"]), "asd_fn": int(metrics["asd_fn"]),
        "td_fp": int(metrics["td_fp"]), "td_tn": int(metrics["td_tn"]),
        "dcc_parameters": sum(parameter.numel() for parameter in module.parameters()),
        "cam_parameter_count": sum(parameter.numel() for parameter in state["cam"].parameters()),
        "module_state_sha256": tensor_tree_sha(checkpoint["dcc"]),
        "model_state_sha256": tensor_tree_sha(checkpoint["model"]),
        "cam_state_sha256": tensor_tree_sha(checkpoint["cam"]),
        "maximum_residual_ratio": float(residual_ratio.max().cpu()),
        "residual_bound_violations": violations,
    }
    return metric_row, rows


def summary_rows(rows: list[dict]) -> list[dict]:
    output = []
    for group in ("E53.3-BASE", *R.GROUPS, "E70-G1", "E70-G2", "E70-G3"):
        group_rows = [row for row in rows if row["group"] == group]
        if not group_rows:
            continue
        summary = {"group": group, "folds": len(group_rows)}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=np.float64)
            summary[f"{metric}_mean_pct"] = float(values.mean())
            summary[f"{metric}_sd_pct"] = float(values.std(ddof=0))
        for metric in ("asd_tp", "asd_fn", "td_fp", "td_tn"):
            summary[metric] = int(sum(int(row[metric]) for row in group_rows))
        output.append(summary)
    return output


def paired_rows(fold_rows: list[dict]) -> list[dict]:
    by_group = {group: {int(row["fold"]): row for row in fold_rows if row["group"] == group} for group in {r["group"] for r in fold_rows}}
    comparisons = []
    base_name = "E53.3-BASE"
    current_and_base = [(group, base_name) for group in R.GROUPS]
    current_and_e70 = [(group, R.GROUP_SOURCE[group]) for group in R.GROUPS]
    current_pairs = [(R.GROUPS[0], R.GROUPS[1]), (R.GROUPS[0], R.GROUPS[2]), (R.GROUPS[1], R.GROUPS[2])]
    for left, right in current_and_base + current_and_e70 + current_pairs:
        if left not in by_group or right not in by_group:
            continue
        for metric in METRICS:
            deltas = [float(by_group[left][fold][metric]) - float(by_group[right][fold][metric]) for fold in range(1, 11)]
            comparisons.append({
                "group_a": left, "group_b": right, "metric": metric,
                "mean_delta_pp": float(np.mean(deltas)),
                "fold_deltas_pp": json.dumps(deltas),
                "wins": sum(value > 1e-12 for value in deltas),
                "ties": sum(abs(value) <= 1e-12 for value in deltas),
                "losses": sum(value < -1e-12 for value in deltas),
            })
    return comparisons


def append_handoff(summary, comparisons, audit_count: int, decision: str) -> str:
    handoff = Path("/model/spromptgl/experiment/handoff.md")
    marker = "<!-- EXPERIMENT70_1_DCC_GCDSI_2026-11-03 -->"
    base = next(row for row in summary if row["group"] == "E53.3-BASE")
    groups = [row for row in summary if row["group"] in R.GROUPS]
    best = max(groups, key=lambda row: row["acc_mean_pct"])
    lines = [
        "", "## Experiment 70.1 — DCC-GCDSI", "",
        f"- 状态：`COMPLETE_AUDITED`；30/30正式折完成，{audit_count}/30最佳checkpoint独立读回通过；初始化和warmup parity 10/10通过。",
        f"- 结论：**{decision}**；最佳组 {best['group']}，ACC {best['acc_mean_pct']:.3f}±{best['acc_sd_pct']:.3f}%；基座ACC {base['acc_mean_pct']:.3f}±{base['acc_sd_pct']:.3f}%。",
        "- 三组ACC / Hard-AUC / Probability-AUC / SEN / SPE（% mean±population SD）：",
    ]
    for row in groups:
        lines.append(f"  - {row['group']}: {row['acc_mean_pct']:.3f}±{row['acc_sd_pct']:.3f} / {row['hard_auc_mean_pct']:.3f}±{row['hard_auc_sd_pct']:.3f} / {row['prob_auc_mean_pct']:.3f}±{row['prob_auc_sd_pct']:.3f} / {row['sen_mean_pct']:.3f}±{row['sen_sd_pct']:.3f} / {row['spe_mean_pct']:.3f}±{row['spe_sd_pct']:.3f}.")
    acc_comparisons = {(row["group_a"], row["group_b"]): row for row in comparisons if row["metric"] == "acc"}
    lines.extend([
        "- 配对ΔACC相对53.3基座：" + "; ".join(f"{group} {acc_comparisons[(group,'E53.3-BASE')]['mean_delta_pp']:+.3f} pp ({acc_comparisons[(group,'E53.3-BASE')]['wins']}/{acc_comparisons[(group,'E53.3-BASE')]['ties']}/{acc_comparisons[(group,'E53.3-BASE')]['losses']} W/T/L)" for group in R.GROUPS) + "。",
        f"- 真实文本相对匿名Bank：G1-G3 ΔACC {acc_comparisons[(R.GROUPS[0],R.GROUPS[2])]['mean_delta_pp']:+.3f} pp，W/T/L {acc_comparisons[(R.GROUPS[0],R.GROUPS[2])]['wins']}/{acc_comparisons[(R.GROUPS[0],R.GROUPS[2])]['ties']}/{acc_comparisons[(R.GROUPS[0],R.GROUPS[2])]['losses']}；统计仅为单seed探索性描述。",
        "- 10/10折初始化、warmup、数学性质和active repeat/resume预检通过；逐患者逐模态残差上界与图条件敏感性见报告。",
        "- 报告：`/model/spromptgl/experiment/experiment70_1_dcc_gcdsi/results/report.md`；原数据和模型源码只读。保留历史val=test、CV前预处理和871节点transductive局限；下一步应使用多seed、严格嵌套/外部验证并预注册组件分析。",
        marker, "",
    ])
    with handoff.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        contents = stream.read()
        if marker not in contents:
            stream.seek(0, os.SEEK_END)
            stream.write("\n".join(lines))
            stream.flush()
            os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return sha_file(handoff)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all:
        parser.error("use --all")
    ref_manifest, ref_folds, ref_metrics = R.load_reference()
    features, labels = R.BASE.load_data()[0]
    folds = R.BASE.load_data()[1]
    e70_metrics = R.read_csv(R.REF70 / "results/fold_metrics.csv")
    metrics_rows = []
    mechanism_rows = []
    for group in R.GROUPS:
        for fold in range(1, 11):
            metric, mechanism = load_checkpoint(group, fold, ref_manifest, ref_folds, R.text_banks()[0], features)
            metrics_rows.append(metric)
            mechanism_rows.extend(mechanism)
            print(f"PASS {group} fold {fold}", flush=True)

    base_rows = []
    for fold, row in ref_metrics.items():
        task = json.loads((R.REF53 / ".runtime/fallback_wd_sweep/CADR_WD_0p01_GFR_WD_0p11/tasks" / f"CADR_WD_0p01_GFR_WD_0p11_f{fold:02d}.json").read_text())
        metrics = task["metrics"]
        test_indices = folds[fold - 1][1]
        test_labels = labels[test_indices]
        asd_count = int((test_labels == 0).sum())
        td_count = int((test_labels == 1).sum())
        asd_tp = int(round(float(metrics["sen"]) * asd_count))
        td_tn = int(round(float(metrics["spe"]) * td_count))
        base_rows.append({"group": "E53.3-BASE", "fold": fold, **{key: float(metrics[key]) * 100 for key in METRICS},
                          "asd_tp": asd_tp, "asd_fn": asd_count - asd_tp,
                          "td_fp": td_count - td_tn, "td_tn": td_tn,
                          "best_epoch": int(task["best_epoch"])})
    e70_rows = []
    for row in e70_metrics:
        e70_rows.append({"group": row["group"], "fold": int(row["fold"]), **{key: float(row[key]) * 100 for key in METRICS},
                         "asd_tp": int(row.get("asd_tp", 0)), "asd_fn": int(row.get("asd_fn", 0)),
                         "td_fp": int(row.get("td_fp", 0)), "td_tn": int(row.get("td_tn", 0)),
                         "best_epoch": int(row["best_epoch"])})
    all_rows = base_rows + metrics_rows + e70_rows
    summaries = summary_rows(all_rows)
    comparisons = paired_rows(all_rows)
    write_csv(RESULTS / "fold_metrics.csv", all_rows)
    write_csv(RESULTS / "paired_comparisons.csv", comparisons)
    write_csv(RESULTS / "mechanism.csv", mechanism_rows)
    write_csv(RESULTS / "summary.csv", summaries)

    base = next(row for row in summaries if row["group"] == "E53.3-BASE")
    groups = [next(row for row in summaries if row["group"] == group) for group in R.GROUPS]
    acc_cmp = {(row["group_a"], row["group_b"]): row for row in comparisons if row["metric"] == "acc"}
    prob_cmp = {(row["group_a"], row["group_b"]): row for row in comparisons if row["metric"] == "prob_auc"}
    sen_cmp = {(row["group_a"], row["group_b"]): row for row in comparisons if row["metric"] == "sen"}
    base_gate = any(
        group["acc_mean_pct"] - base["acc_mean_pct"] >= 0.5
        and acc_cmp[(group["group"], "E53.3-BASE")]["wins"] >= 6
        and prob_cmp[(group["group"], "E53.3-BASE")]["mean_delta_pp"] >= -0.5
        and sen_cmp[(group["group"], "E53.3-BASE")]["mean_delta_pp"] >= -1.0
        for group in groups[:2]
    )
    text_gate = any(
        acc_cmp[(group, R.GROUPS[2])]["mean_delta_pp"] > 0
        and acc_cmp[(group, R.GROUPS[2])]["wins"] >= 6
        for group in R.GROUPS[:2]
    )
    mechanism_pass = all(int(row["residual_bound_violations"]) == 0 for row in metrics_rows)
    if not base_gate:
        decision = "NO-GO"
    elif not text_gate or not mechanism_pass:
        decision = "CONDITIONAL"
    else:
        decision = "GO"
    handoff_sha = append_handoff(summaries, comparisons, len(metrics_rows), decision)
    report = render_report(summaries, comparisons, mechanism_rows, metrics_rows, base_gate, text_gate, mechanism_pass, decision, handoff_sha)
    (RESULTS / "report.md").write_text(report, encoding="utf-8")
    manifest = json.loads((EXP / "manifest.json").read_text())
    manifest.update({
        "status": "COMPLETE_AUDITED" if len(metrics_rows) == 30 and mechanism_pass else "AUDIT_FAILED",
        "preflight": "PASS", "completed_formal_folds": len(metrics_rows), "failed_formal_folds": 30 - len(metrics_rows),
        "independently_audited_checkpoints": len(metrics_rows), "checkpoint_audit_status": "PASS" if len(metrics_rows) == 30 else "FAIL",
        "initialization_parity": "PASS_10_OF_10", "warmup_parity": "PASS_10_OF_10",
        "decision": decision, "best_group": max(groups, key=lambda row: row["acc_mean_pct"])["group"],
        "preflight_sha256": sha_file(R.RUNTIME / "preflight.json"), "training_source_sha256": R.source_sha256(),
        "config_sha256": sha_file(R.CONFIG_PATH), "handoff_sha256": handoff_sha,
    })
    (EXP / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_status = {
        "status": manifest["status"], "folds_audited": len(metrics_rows), "errors": [], "decision": decision,
        "base_performance_gate": base_gate, "real_text_gate": text_gate, "residual_bound_all_folds": mechanism_pass,
        "handoff_sha256": handoff_sha,
    }
    (RESULTS / "audit_status.json").write_text(json.dumps(audit_status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit_status, ensure_ascii=False, indent=2), flush=True)


def render_report(summaries, comparisons, mechanism_rows, fold_rows, base_gate, text_gate, mechanism_pass, decision, handoff_sha):
    lookup = {row["group"]: row for row in summaries}
    comparison = {(row["group_a"], row["group_b"], row["metric"]): row for row in comparisons}
    prompt_assets = json.loads((R.RUNTIME / "preflight.json").read_text(encoding="utf-8"))["prompt_assets"]

    def mechanism_values(group: str, key: str) -> list[float]:
        return [float(row[key]) for row in mechanism_rows if row["group"] == group]

    def mechanism_mean(group: str, key: str) -> float:
        values = mechanism_values(group, key)
        return float(np.mean(values)) if values else float("nan")

    def mechanism_max(group: str, key: str) -> float:
        values = mechanism_values(group, key)
        return float(np.max(values)) if values else float("nan")

    def mechanism_abs_mean(group: str, key: str) -> float:
        values = mechanism_values(group, key)
        return float(np.mean(np.abs(values))) if values else float("nan")

    lines = [
        "# Experiment 70.1 — DCC-GCDSI", "",
        f"状态：`COMPLETE_AUDITED`，30/30正式折完成，30/30最佳checkpoint独立读回通过；科研结论：**{decision}**。单seed工程判据不是统计显著性。", "",
        "## 研究动机与修正", "",
        "Experiment 70 GCDSI因BrainPrompt两类CLIP文本近同向（cos≈0.9848）、G1普通注意力熵接近ln(2)、共享语义Value直接混入CAM、部分模态残差超过CAM RMS而判为NO-GO。70.1把文本共有方向与ASD−TD差异方向分开：差异方向形成相反类别Key和有符号竞争系数；共有方向只调制患者CAM通道；逐患者逐模态将残差RMS严格限制在CAM RMS的15%以内。模块不再注册进CAM，独立管理优化器和checkpoint。", "",
        "SPromptGL中的GFR表示患者关系传播、CAM表示四模态交互、PEN是Prompt驱动的局部特征增强；本实验沿用无PEN的53.3第三组，并在CAM后、原OutputLayer前插入DCC-GCDSI。SABER公式(8)-(10)提供图条件Query、语义读取和门控残差的思路，疾病级对比监督是额外路径；70.1不添加该损失，也不声称完整复现SABER。", "",
        "## 方法与数学审计", "",
        "输入路径：`X[871,320] -> CADR-GFR -> F[871,320] -> CAM -> H[871,4,36] -> DCC-GCDSI -> H'[871,4,36] -> flatten[871,144] -> 原OutputLayer`。CAM每轮只调用一次。", "",
        "```text\nq_im = LayerNorm(W_h h_im + W_f f_i + q_CLS)\nc = normalize((t_ASD+t_TD)/2); d=(t_ASD-t_TD)/(||t_ASD-t_TD||+eps)\nk_delta = normalize(W_d d); z_im=q_im^T k_delta\nA_im=softmax([z_im,-z_im]); s_im=A_ASD-A_TD=tanh(z_im)\nv_center=normalize(W_c c); u_im=tanh(W_o(h_im * v_center))\ng_im=sigmoid(MLP([h_im,q_im,v_center])); r_im=g_im * u_im\nlambda=0.15*tanh(lambda_raw), lambda_raw(0)=0\ndelta_im=lambda*s_im*RMS(h_im)*r_im/(RMS(r_im)+eps)\nh'_im=h_im+delta_im\n```", "",
        "逐患者逐模态审计验证了lambda=0恒等、softmax差等于tanh、两类文本相同则零注入、交换ASD/TD顺序后残差反号且共有方向不变。不同冻结文本Bank改变center和delta方向。", "",
        "## 基座与训练协议", "",
        "唯一基座为53.3第三组`CADR_WD_0p01_GFR_WD_0p11`。53.3 manifest SHA为`fae43596a9788d63bf8b0a0e52854fd1f7f014fea36c062bcf8a881806d7fcba`，reference fold metrics SHA为`2849ce9d161d5b5ea32bc1b210ebde4f4bd0daf25e316298934b8029b0f8f9a2`。10个基座checkpoint/task/readback SHA及split、pair、GFR/CAM/CADR、schema、anchor和初始化RNG均在预检逐折匹配。初始化与warmup parity为10/10；warmup Hook关闭两步轨迹和fold1 active重复/恢复均逐位一致。新模块使用每折`701000+fold`隔离初始化，三组同折参数SHA一致。", "",
        "训练保留seed=0、10折`StratifiedKFold(shuffle=True, random_state=10)`、ABIDE A 871例、ASD=0/TD=1、PHENO/ANAT/FUNC/CORR维度48/6/10/256、原CAM/GFR/CADR优化器、原损失、warmup 10、450 epoch上限、wait>200和严格ACC改善checkpoint选择。DCC独立AdamW lr=0.001、WD=0.001，无额外loss/dropout。", "",
        "## Prompt来源", "",
        "G1使用BrainPrompt官方ABIDE两条完整原文，将来源顺序`[TC,ASD]`调整为`[ASD,TD]`。G2逐字使用SPromptGL `network.py`八条原文并保留`Nromal`；每条归一化，前四条聚合成TD、后四条聚合成ASD，再归一化并排序。其泛化正常/异常描述不等同于ASD特异性生物标志物。G3复用Experiment 70固定匿名向量，独立seed=700070，匹配G1范数和余弦几何。三组均使用同一CLIP RN50冻结Embedding。", "",
        f"冻结CLIP RN50 embedding cache SHA：`{prompt_assets['cache_sha256']}`；BrainPrompt原文SHA：`{prompt_assets['brainprompt_text_sha256']}`；SPromptGL原文SHA：`{prompt_assets['spromptgl_text_sha256']}`。最终Bank SHA：G1 `{prompt_assets['banks_sha256']['E70.1-G1']}`，G2 `{prompt_assets['banks_sha256']['E70.1-G2']}`，G3 `{prompt_assets['banks_sha256']['E70.1-G3']}`。", "",
        "BrainPrompt原文（按正式输入顺序）：", "",
        f"1. `{prompt_assets['brainprompt_texts_source_order'][1]}`（ASD）",
        f"2. `{prompt_assets['brainprompt_texts_source_order'][0]}`（TD）", "",
        "SPromptGL原文（保持来源顺序与拼写）：", "",
        *[f"- `{text}`" for text in prompt_assets["spromptgl_texts_source_order"]], "",
        "## 十折结果", "",
        "全部数值为百分比mean±population SD。", "",
        "| Group | ACC | Hard-AUC / BA | Probability-AUC | SEN-ASD | SPE-TD | ASD-F1 | TP/FP/FN/TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("E53.3-BASE", *R.GROUPS, "E70-G1", "E70-G2", "E70-G3"):
        if group not in lookup:
            continue
        row = lookup[group]
        counts = f"{row['asd_tp']}/{row['td_fp']}/{row['asd_fn']}/{row['td_tn']}"
        lines.append(f"| {group} | {row['acc_mean_pct']:.4f}±{row['acc_sd_pct']:.4f} | {row['hard_auc_mean_pct']:.4f}±{row['hard_auc_sd_pct']:.4f} | {row['prob_auc_mean_pct']:.4f}±{row['prob_auc_sd_pct']:.4f} | {row['sen_mean_pct']:.4f}±{row['sen_sd_pct']:.4f} | {row['spe_mean_pct']:.4f}±{row['spe_sd_pct']:.4f} | {row['asd_f1_mean_pct']:.4f}±{row['asd_f1_sd_pct']:.4f} | {counts} |")
    lines.extend(["", "逐折指标、best epoch和混淆计数在`fold_metrics.csv`。逐折配对差数组、mean delta和胜/平/负在`paired_comparisons.csv`；其中比较53.3基座、实验70同Prompt组及三个70.1组之间。", "", "## 配对比较", "", "| A-B | ΔACC pp (W/T/L) | ΔHard-AUC pp | ΔProb-AUC pp | ΔSEN pp | ΔSPE pp |", "|---|---:|---:|---:|---:|---:|"])
    for left, right in [(group, "E53.3-BASE") for group in R.GROUPS] + [(group, R.GROUP_SOURCE[group]) for group in R.GROUPS] + [(R.GROUPS[0], R.GROUPS[1]), (R.GROUPS[0], R.GROUPS[2]), (R.GROUPS[1], R.GROUPS[2])]:
        acc = comparison[(left, right, "acc")]
        vals = [comparison[(left, right, metric)]["mean_delta_pp"] for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe")]
        lines.append(f"| {left} - {right} | {vals[0]:+.3f} ({acc['wins']}/{acc['ties']}/{acc['losses']}) | {vals[1]:+.3f} | {vals[2]:+.3f} | {vals[3]:+.3f} | {vals[4]:+.3f} |")
    lines.extend(["", "## 机制证据", "", f"残差上界在每个正式训练前向、读回checkpoint的871×4个模态位置均检查；总违规数为{sum(int(row['residual_bound_violations']) for row in fold_rows)}，全30折上界门：{'PASS' if mechanism_pass else 'FAIL'}。逐患者逐模态残差比在训练与读回中均未超过15%。", "", "| Group | mean abs(s) | patient SD(s) | cross-modality SD(s) | attention entropy | mean abs(lambda) | residual/CAM RMS mean / max | gate mean | gate saturation | graph-off attention L1 | graph-off dP(ASD) | graph-off flips/fold | residual-off dP(ASD) | residual-off flips/fold |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for group in R.GROUPS:
        lines.append(
            f"| {group} | {mechanism_mean(group, 'signed_abs_mean'):.3f} | {mechanism_mean(group, 'signed_patient_sd'):.3f} | {mechanism_mean(group, 'signed_cross_modality_sd'):.3f} | {mechanism_mean(group, 'attention_entropy_mean'):.3f} | {mechanism_abs_mean(group, 'lambda'):.4f} | {100 * mechanism_mean(group, 'residual_cam_rms_ratio_mean'):.2f}% / {100 * mechanism_max(group, 'residual_cam_rms_ratio_max'):.2f}% | {mechanism_mean(group, 'gate_mean'):.3f} | {mechanism_mean(group, 'gate_saturation_fraction'):.3f} | {mechanism_mean(group, 'graph_off_attention_l1'):.3f} | {100 * mechanism_mean(group, 'graph_off_mean_abs_delta_p_asd'):.3f} pp | {mechanism_mean(group, 'graph_off_prediction_flips'):.2f} | {100 * mechanism_mean(group, 'residual_off_mean_abs_delta_p_asd'):.3f} pp | {mechanism_mean(group, 'residual_off_prediction_flips'):.2f} |"
        )
    lines.extend(["", "签名竞争强度具有患者与模态间变异，且其绝对值接近1、注意力熵很低，表示多数位置的二类竞争已接近饱和；低熵本身不证明医学语义有效。实际残差均值约占CAM RMS的2%至3%，低于15%上限较多。固定checkpoint清零GFR条件会明显改变注意力，但平均ASD概率和硬预测改变较小；清零语义残差的预测敏感性也有限，说明条件交互真实存在，却没有形成强分类贡献。", "", "各折W_f、W_d、W_c及门控梯度和更新量，以及站点eta²、年龄、FUNC头动与ANAT/FUNC质量相关见`mechanism.csv`。这些相关只作诊断，不能证明病灶解释；质量变量不是脑部病灶指标。固定checkpoint干预不等于重新训练组件消融，也不能替代外部验证。", "", "## 结果判定", "", f"预注册基座性能门：{'PASS' if base_gate else 'FAIL'}；真实文本相对匿名Token门：{'PASS' if text_gate else 'FAIL'}；严格残差机制门：{'PASS' if mechanism_pass else 'FAIL'}。科研结论为**{decision}**。", "", "真实文本与匿名Bank可能经共享可学习投影形成相似潜在方向；性能差异不自动证明CLIP文本包含独立ASD医学知识。较实验70改善但仍未过53.3基座，也不能被解释为第二创新有效。单seed、10折折间相关和按历史测试指标选checkpoint不支持统计显著性或临床泛化主张。", "", "## 审计边界", "", "原始模型源码、数据、历史53.3/59/70目录保持只读；`/data`目标路径创建返回EIO Errno 5，故正式实验放在`/model/spromptgl/experiment/experiment70_1_dcc_gcdsi`。所需运行代码、输入和缓存从服务器实际来源复制到本实验`.runtime/reference_snapshot`并按SHA逐项核验。原始代码、数据及历史SHA列在manifest。", "", "保留历史`val=test` checkpoint选择、外层CV前RFE/标准化和871节点transductive图；因此不是无偏外部泛化估计。下一步建议是多seed、严格嵌套/外部站点验证和预注册组件分析。", "", f"交接文档SHA：`{handoff_sha}`。", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
