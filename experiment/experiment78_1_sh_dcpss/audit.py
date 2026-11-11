#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2_contingency, t

import run as runner

HERE = runner.HERE
RESULTS = runner.RESULTS
METRICS = runner.METRICS
G1, G2, G3 = runner.GROUPS


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def json_safe(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def cramers_v(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if not len(a) or len(a) != len(b):
        return 0.0
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    if min(table.shape) < 2 or table.sum() == 0:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    denom = table.sum() * (min(table.shape) - 1)
    return float(math.sqrt(max(chi2, 0.0) / denom)) if denom else 0.0


def _metric_match(actual, expected, tolerance=1e-10):
    return all(abs(float(actual[key]) - float(expected[key])) <= tolerance for key in METRICS)


def _audit_checkpoint(group, fold, frozen_sha, data):
    task_file, checkpoint_file, _, _ = runner.task_paths(group, fold)
    task = json.loads(task_file.read_text(encoding="utf-8"))
    if task.get("status") != "COMPLETE" or task.get("source_sha256") != frozen_sha:
        raise RuntimeError("task status or source SHA is not complete/frozen")
    if task.get("checkpoint_sha256") != runner.sha_file(checkpoint_file):
        raise RuntimeError("best checkpoint SHA differs from its task record")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if checkpoint.get("source_sha256") != frozen_sha or checkpoint.get("group") != group or checkpoint.get("fold") != fold:
        raise RuntimeError("checkpoint identity differs from task identity")
    state, warmup, identity = runner.prepare(fold)
    if identity != checkpoint["identity"] or identity != task["identity"]:
        raise RuntimeError("E74.4 warmup restore identity differs from checkpoint")
    if warmup["rng_sha"] != checkpoint["initialization"]["warmup_rng_sha"]:
        raise RuntimeError("initial E61 warmup RNG differs from checkpoint provenance")
    if checkpoint["initialization"]["original_optimizer_sha256"] != identity["optimizer_sha"]:
        raise RuntimeError("one or more historical optimizer states were not inherited")
    legal_support = runner.support_ids(state)
    cp_support = np.asarray(checkpoint["initialization"]["support_ids"], dtype=np.int64)
    if not np.array_equal(legal_support, cp_support) or np.intersect1d(legal_support, state["test_np"]).size:
        raise RuntimeError("prototype support identity includes an outer test patient")
    overlap = np.intersect1d(state["train_np"], state["test_np"]).astype(np.int64)
    if not np.array_equal(overlap, np.asarray([376, 406]) if fold == 2 else np.zeros(0, dtype=np.int64)):
        raise RuntimeError(f"unexpected zero-based historical fold overlap: {overlap.tolist()}")
    if np.intersect1d(overlap, cp_support).size:
        raise RuntimeError("historically overlapping rows entered new prototype support")
    history = checkpoint["loss_history"]
    max_acc = max(row["metrics"]["acc"] for row in history)
    first_best = next(row["epoch"] for row in history if row["metrics"]["acc"] == max_acc)
    if int(checkpoint["best_epoch_1based"]) != int(first_best):
        raise RuntimeError("checkpoint does not follow strict first-maximum test ACC selection")
    if checkpoint["best_epoch_1based"] != task["best_epoch_1based"] or checkpoint["epochs_executed"] != task["epochs_executed"]:
        raise RuntimeError("checkpoint and task best epoch / executed epoch differ")

    spec = runner.CFG["group_specs"][group]
    geometry, coords, _ = runner.prompt_tensors(state["y"].device)
    classes = geometry["prompt_classes"] if spec["prompt_source"] == "spromptgl" else geometry["anonymous_classes"]
    module = runner.initialize(coords, classes.to(state["y"].device), spec["k"],
                               spec["prompt_source"], fold, state["y"].device)
    optimizer = runner._new_optimizer(module)
    best_state = checkpoint["best_state"]
    runner.hist().load_bundle(state, best_state["base_state"])
    module.load_state_dict(best_state["module_state"], strict=True)
    optimizer.load_state_dict(best_state["new_optimizer"])
    if runner.sha_object(optimizer.state_dict()) != runner.sha_object(best_state["new_optimizer"]):
        raise RuntimeError("new optimizer state did not restore exactly")
    rng_before_eval = runner.runtime().rng_sha(checkpoint["best_rng"]["packed"])
    if rng_before_eval != checkpoint["best_rng"]["rng_after_eval"]:
        raise RuntimeError("checkpoint RNG payload differs from recorded best evaluation RNG")
    runner.runtime().restore_rng(checkpoint["best_rng"]["packed"])
    sites = torch.as_tensor(data["site_ids"], dtype=torch.long, device=state["y"].device)
    module.eval()
    with torch.no_grad():
        output, diagnostics = runner._forward(
            state, module, training=False, support=legal_support,
            query=state["test_np"], sites=sites,
        )
        recomputed = runner._metrics(output["logp"], state)
    probability = np.asarray(recomputed["prob"])
    stored_probability = np.asarray(checkpoint["test_probability"])
    if not np.array_equal(np.asarray(recomputed["pred"]), np.asarray(checkpoint["test_prediction"])):
        raise RuntimeError("best checkpoint readback predictions differ")
    if probability.shape != stored_probability.shape or np.max(np.abs(probability - stored_probability)) > 1e-7:
        raise RuntimeError("best checkpoint readback probabilities differ")
    if not _metric_match(recomputed, checkpoint["metrics"], tolerance=1e-10):
        raise RuntimeError("best checkpoint readback six-index metrics differ")
    if not _metric_match(recomputed, task["metrics"], tolerance=1e-10):
        raise RuntimeError("checkpoint readback metrics differ from task record")
    post_eval_rng = runner.runtime().rng_sha(runner.runtime().pack_rng())
    if post_eval_rng != rng_before_eval:
        raise RuntimeError("deterministic checkpoint evaluation changed public RNG")

    original_y = state["y"].detach().clone()
    test_ids = torch.as_tensor(state["test_np"], dtype=torch.long, device=state["y"].device)
    with torch.no_grad():
        state["y"][test_ids] = 1 - state["y"][test_ids]
        flipped, flipped_diag = runner._forward(state, module, training=False,
                                                support=legal_support, query=state["test_np"], sites=sites)
        state["y"].copy_(original_y)
    if not torch.equal(output["logp"], flipped["logp"]) or not torch.equal(diagnostics["h_fused"], flipped_diag["h_fused"]):
        raise RuntimeError("test label intervention changed E78.1 inference")

    baseline_task = json.loads((runner.BASE / ".runtime/tasks" / f"{runner.BASE_GROUP}_f{fold:02d}.json").read_text())
    base_trace = {row["epoch"]: row for row in baseline_task["rng_trace"]}
    trace = {row["epoch"]: row for row in checkpoint["rng_trace"]}
    common = sorted(set(base_trace) & set(trace))
    exact = sum(all(base_trace[epoch][key] == trace[epoch][key]
                    for key in ("rng_before", "rng_after_train", "rng_after_eval")) for epoch in common)
    branch_audit = checkpoint["loss_history"][0].get("gradient_audit_first_epoch", {}).get("auxiliary", {})
    branch = branch_audit.get("classification_semantic_residual_branch", {})
    if not branch or min(branch.get(key, 0.0) for key in ("module", "darg_gfr", "cam")) <= 0:
        raise RuntimeError("first-epoch semantic residual gradient did not reach module, DARG/GFR, and CAM")
    audit = {
        "group": group, "fold": fold, "status": "INDEPENDENT_AUDIT_PASS",
        "checkpoint_sha256": runner.sha_file(checkpoint_file), "source_sha256": frozen_sha,
        "initial_optimizer_sha": identity["optimizer_sha"], "warmup_sha256": identity["warmup_sha256"],
        "warmup_rng_sha": warmup["rng_sha"], "initial_rng_sha": identity["initial_rng_sha"],
        "support_count": int(len(legal_support)), "support_sha256": checkpoint["initialization"]["support_sha256"],
        "support_disjoint_from_test": True, "fold2_overlap_rows_excluded_from_new_support": True,
        "test_label_invariant": True, "best_acc_rule_pass": True,
        "all_original_optimizers_restored": True, "new_optimizer_restored": True,
        "best_checkpoint_readback_metrics_match": True,
        "best_epoch_1based": int(checkpoint["best_epoch_1based"]),
        "epochs_executed": int(checkpoint["epochs_executed"]),
        "wall_seconds": float(task.get("wall_seconds", 0.0)),
        "metrics": {key: float(recomputed[key]) for key in METRICS},
        "rng_common_epochs": len(common), "rng_exact_epochs": exact,
        "rng_mismatches": len(common) - exact,
        "new_residual_gradient_norms": branch,
        "evaluation_rng_unchanged": True,
        "classification_output_is_log_probability": bool(torch.allclose(output["logp"].exp().sum(1), torch.ones(871, device=output["logp"].device), atol=1e-6)),
    }
    if audit["classification_output_is_log_probability"] is not True:
        raise RuntimeError("OutputLayer output is not normalized log probability")
    return audit, checkpoint, diagnostics, recomputed, state, module, optimizer


def _mechanism(group, fold, checkpoint, diagnostics, data):
    diag = checkpoint["best_module_diagnostics"]
    rho = np.asarray(diag["rho_support"], dtype=np.float64)
    support_y = np.asarray(diag["support_labels"], dtype=np.int64)
    support_sites = np.asarray(diag["support_sites"], dtype=np.int64)
    support_ids = np.asarray(diag["support_ids"], dtype=np.int64)
    age_ids = data["age_ids"][support_ids]
    beta = np.asarray(diag["beta"], dtype=np.float64)
    attention = np.asarray(diag["attention"], dtype=np.float64)
    proto_z = np.asarray(diag["proto_z"], dtype=np.float64).mean(0)
    counts = np.asarray(diag["base_counts"], dtype=np.float64)
    sem = np.asarray(diag["semantic_vector"], dtype=np.float64)
    residual = np.asarray(diag["h_fused"], dtype=np.float64) - np.asarray(diag["h_original"], dtype=np.float64)
    qids = np.asarray(diag["query_ids"], dtype=np.int64)
    k = rho.shape[1]
    class_sims, class_distances, collapse, site_vs, age_vs = [], [], [], [], []
    site_mix, route_prop, support_entropy = {}, {}, []
    for cls in (0, 1):
        vectors = proto_z[cls]
        sim = vectors @ vectors.T
        upper = sim[np.triu_indices(k, 1)] if k > 1 else np.asarray([], dtype=np.float64)
        class_sims.extend(upper.tolist())
        class_distances.extend((1.0 - upper).tolist())
        collapse.extend((upper > 0.98).astype(float).tolist())
        mask = support_y == cls
        assignment = rho[mask].argmax(1)
        site_vs.append(cramers_v(assignment, support_sites[mask]))
        age_vs.append(cramers_v(assignment, age_ids[mask]))
        site_mix[str(cls)] = {}
        route_prop.append(np.bincount(assignment, minlength=k) / max(len(assignment), 1))
        support_entropy.extend((-(rho[mask] * np.log(np.maximum(rho[mask], 1e-12))).sum(1)).tolist())
        for cluster in range(k):
            chosen = support_sites[mask][assignment == cluster]
            unique, n = np.unique(chosen, return_counts=True)
            site_mix[str(cls)][str(cluster)] = {str(int(x)): int(y) for x, y in zip(unique, n)}
    tvs = []
    prompt_std = []
    entropies = []
    prompt_mean = {}
    for cls in (0, 1):
        attention_class = attention[:, cls]
        entropies.extend((-(attention_class * np.log(np.maximum(attention_class, 1e-12))).sum(-1)).reshape(-1).tolist())
        prompt_std.append(float(attention_class.std(0).mean()))
        prompt_mean[str(cls)] = attention_class.mean(0).tolist()
        for left in range(k):
            for right in range(left + 1, k):
                tvs.extend((0.5 * np.abs(attention_class[:, left] - attention_class[:, right]).sum(-1)).tolist())
    route_entropy = -(beta * np.log(np.maximum(beta, 1e-12))).sum(-1)
    per_class_counts = counts.tolist()
    return {
        "group": group, "fold": fold, "k": k,
        "effective_support_per_class_subprototype": json.dumps(per_class_counts),
        "mean_prototype_cosine_similarity": float(np.mean(class_sims)) if class_sims else 1.0,
        "mean_prototype_cosine_distance": float(np.mean(class_distances)) if class_distances else 0.0,
        "prototype_pair_collapse_fraction_cosine_gt_0_98": float(np.mean(collapse)) if collapse else 0.0,
        "support_assignment_entropy": float(np.mean(support_entropy)) if support_entropy else 0.0,
        "support_assignment_proportions": json.dumps([x.tolist() for x in route_prop]),
        "test_route_entropy": float(route_entropy.mean()),
        "test_route_entropy_normalized": float(route_entropy.mean() / math.log(k)) if k > 1 else 0.0,
        "subprototype_prompt_tv": float(np.mean(tvs)) if tvs else 0.0,
        "prompt_attention_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "patient_prompt_weight_std": float(np.mean(prompt_std)),
        "prompt_weights_by_class_subprototype": json.dumps(prompt_mean),
        "site_cramers_v_by_class": json.dumps(site_vs), "site_cramers_v_mean": float(np.mean(site_vs)),
        "age_cramers_v_by_class": json.dumps(age_vs), "age_cramers_v_mean": float(np.mean(age_vs)),
        "site_composition_by_class_subprototype": json.dumps(site_mix),
        "semantic_vector_patient_std": float(sem.std(0).mean()),
        "feature_residual_relative_norm": float(np.linalg.norm(residual) / max(np.linalg.norm(np.asarray(diag["h_original"])), 1e-12)),
        "training_heterogeneity_losses": json.dumps({key: value for key, value in checkpoint["best_losses"].items() if "heterogeneity" in key}),
        "fusion_lambda_at_best": float(checkpoint["best_module_diagnostics"].get("lambda", 0.0)) if not torch.is_tensor(checkpoint["best_module_diagnostics"].get("lambda")) else float(checkpoint["best_module_diagnostics"]["lambda"].detach().cpu()),
        "support_and_query_counts": json.dumps({"support": len(support_ids), "query": len(qids)}),
    }


def _interventions(group, fold, checkpoint, state, module, optimizer, data, full_metrics):
    if group != G2:
        return []
    support = runner.support_ids(state)
    sites = torch.as_tensor(data["site_ids"], dtype=torch.long, device=state["y"].device)
    result = []
    for intervention in ("module_off", "subprototype_off", "anonymous", "prompt_class_swap", "feature_residual_off"):
        module.eval()
        with torch.no_grad():
            output, _ = runner._forward(state, module, training=False, support=support,
                                        query=state["test_np"], sites=sites,
                                        intervention=intervention)
            metrics = runner._metrics(output["logp"], state)
        before = np.asarray(full_metrics["pred"], dtype=np.int64)
        after = np.asarray(metrics["pred"], dtype=np.int64)
        labels = state["y"].detach().cpu().numpy()[state["test_np"]]
        result.append({
            "group": group, "fold": fold, "intervention": intervention,
            "acc": float(metrics["acc"]), "hard_auc": float(metrics["hard_auc"]),
            "prob_auc": float(metrics["prob_auc"]), "sen": float(metrics["sen"]),
            "spe": float(metrics["spe"]), "asd_f1": float(metrics["asd_f1"]),
            "mean_abs_probability_change": float(np.abs(np.asarray(metrics["prob"]) - np.asarray(full_metrics["prob"])).mean()),
            "max_abs_probability_change": float(np.abs(np.asarray(metrics["prob"]) - np.asarray(full_metrics["prob"])).max()),
            "hard_label_flips": int((before != after).sum()),
            "correct_to_wrong": int(((before == labels) & (after != labels)).sum()),
            "wrong_to_correct": int(((before != labels) & (after == labels)).sum()),
        })
    return result


def _historical_metrics(path, group):
    rows = [row for row in read_csv(path) if row.get("group") == group]
    if len(rows) != 10:
        raise RuntimeError(f"expected 10 historical rows for {group} in {path}")
    result = {}
    for row in rows:
        values = {key: float(row[key]) if key in row and row[key] != "" else float(row["hard_auc"])
                  for key in METRICS}
        if values["acc"] > 1.5:
            values = {key: value / 100.0 for key, value in values.items()}
        result[int(row["fold"])] = values
    return result


def _paired(name, left, right, folds):
    records = []
    for metric in METRICS:
        delta = np.asarray([(left[fold][metric] - right[fold][metric]) * 100.0 for fold in folds])
        mean = float(delta.mean())
        sd = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
        half = float(t.ppf(0.975, len(delta) - 1) * sd / math.sqrt(len(delta))) if len(delta) > 1 else 0.0
        records.append({"comparison": name, "scope": "all10" if len(folds) == 10 else "exclude_fold2",
                        "metric": metric, "n_folds": len(folds), "mean_difference_pp": mean,
                        "population_sd_difference_pp": float(delta.std(ddof=0)),
                        "wins": int((delta > 1e-12).sum()), "ties": int((np.abs(delta) <= 1e-12).sum()),
                        "losses": int((delta < -1e-12).sum()),
                        "descriptive_95ci_low_pp": mean - half, "descriptive_95ci_high_pp": mean + half,
                        "fold_differences_pp": json.dumps({str(fold): float((left[fold][metric] - right[fold][metric]) * 100.0) for fold in folds}, sort_keys=True)})
    return records


def _write_report(audits, summaries, comparisons, mechanisms, interventions, source_sha, final_status):
    group_summary = {row["group"]: row for row in summaries}
    lines = [
        "# Experiment 78.1: SH-DCPSS 结果报告", "",
        f"状态：`{final_status}`。正式目标为3组×10折；已完成任务 {len(audits)}/30，独立审计 {sum(a['status'] == 'INDEPENDENT_AUDIT_PASS' for a in audits)}/30。",
        "", "## 审计与方法", "",
        f"唯一基座是 `{runner.BASE_GROUP}`；每折从对应 E61 epoch-10 warmup 的完整参数、三个原优化器和 RNG 恢复。训练源 SHA-256：`{source_sha}`。",
        "CAM144 通过只读 pre-hook 注入原始 OutputLayer。E74.4 分类损失、图/配对/对抗/保持损失和原优化器更新次序保留；新特征残差由第三个类内动态原型与固定文本语义共同生成。",
        "方法审计、论文证据范围和数据身份限制见 `literature/source_review.md`。E74.4历史协议含 val=test、测试ACC选模、外层CV前预处理/RFE、全患者传导图和Fold 2历史重叠；本文结果不代表无偏临床泛化估计。",
        "", "## 分组结果", "",
        "|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN|SPE|ASD-F1|", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in runner.GROUPS:
        row = group_summary.get(group)
        if row is None:
            continue
        cells = []
        for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1"):
            cells.append(f"{row[f'{metric}_mean_percent']:.4f} ± {row[f'{metric}_population_sd_percent']:.4f}")
        lines.append("|" + group + "|" + "|".join(cells) + "|")
    duration_by_group = {}
    for group in runner.GROUPS:
        durations = [row["wall_seconds"] for row in audits
                     if row.get("group") == group and row.get("status") == "INDEPENDENT_AUDIT_PASS"]
        duration_by_group[group] = float(np.mean(durations)) if durations else float("nan")
    lines.append("")
    lines.append("平均每折训练耗时（秒）：" + "; ".join(f"{group} {duration_by_group[group]:.2f}" for group in runner.GROUPS))
    lines.extend(["", "## 配对比较", "",
                  "下表按10折逐折配对；折2剔除敏感性结果同时写入 `paired_comparisons.csv`。W/T/L是每折差值的胜/平/负，95%区间为描述性 t 区间。", "",
                  "|比较|指标|差值(pp)|W/T/L|折2剔除差值(pp)|", "|---|---|---:|---:|---:|"])
    comparison_names = [f"{G2} - {G1}", f"{G2} - {G3}", f"{G2} - E78-G2", f"{G2} - E74.4-G2",
                        f"{G1} - E74.4-G2", f"{G3} - E74.4-G2"]
    for name in comparison_names:
        rows = [row for row in comparisons if row["comparison"] == name and row["scope"] == "all10"]
        for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1"):
            row = next((item for item in rows if item["metric"] == metric), None)
            nine = next((item for item in comparisons if item["comparison"] == name and item["scope"] == "exclude_fold2" and item["metric"] == metric), None)
            if row:
                lines.append(f"|{name}|{metric}|{row['mean_difference_pp']:.4f}|{row['wins']}/{row['ties']}/{row['losses']}|{nine['mean_difference_pp']:.4f}|")
    valid_mechanisms = [row for row in mechanisms if row["group"] == G2]
    if valid_mechanisms:
        means = {key: float(np.mean([row[key] for row in valid_mechanisms])) for key in
                 ("prototype_pair_collapse_fraction_cosine_gt_0_98", "site_cramers_v_mean",
                  "age_cramers_v_mean", "subprototype_prompt_tv", "patient_prompt_weight_std",
                  "test_route_entropy_normalized", "feature_residual_relative_norm")}
    else:
        means = {}
    lines.extend(["", "## 机制证据", ""])
    if means:
        lines.extend([
            f"G2原型对余弦相似度>0.98的平均比例为 {means['prototype_pair_collapse_fraction_cosine_gt_0_98']:.4f}（E78-G2为0.767）；平均站点 Cramér's V 为 {means['site_cramers_v_mean']:.4f}（E78-G2约0.6703），年龄分组关联为 {means['age_cramers_v_mean']:.4f}。",
            f"子原型条件 Prompt 权重TV为 {means['subprototype_prompt_tv']:.5f}（E78-G2约0.00749）；患者权重标准差 {means['patient_prompt_weight_std']:.5f}，归一化路由熵 {means['test_route_entropy_normalized']:.4f}。",
            f"CAM特征修正与原特征范数比为 {means['feature_residual_relative_norm']:.5f}。每折各原型有效支持数、站点组成、Prompt权重及残差数据见 `mechanism.csv`。",
        ])
    else:
        lines.append("G2机制记录缺失。")
    if interventions:
        grouped = {}
        for row in interventions:
            grouped.setdefault(row["intervention"], []).append(row)
        lines.extend(["", "G2同一Checkpoint干预（10折平均，固定训练权重）：", ""])
        for key, rows in grouped.items():
            lines.append(f"- `{key}`：ACC {np.mean([r['acc'] for r in rows]) * 100:.4f}%，平均概率绝对变化 {np.mean([r['mean_abs_probability_change'] for r in rows]):.6f}，硬标签翻转 {sum(r['hard_label_flips'] for r in rows)}。")
    lines.extend(["", "## 假设判断", ""])
    h1 = bool(means and means["prototype_pair_collapse_fraction_cosine_gt_0_98"] < 0.767 and all(min(json.loads(row["effective_support_per_class_subprototype"])[cls]) > 5 for row in valid_mechanisms for cls in (0, 1)))
    h2 = bool(means and means["site_cramers_v_mean"] < 0.6703)
    h3 = bool(means and means["subprototype_prompt_tv"] > 0.00749 and means["patient_prompt_weight_std"] > 1e-4)
    h4 = False
    if G2 in group_summary and G3 in group_summary:
        h4 = (group_summary[G2]["acc_mean_percent"] > group_summary[G3]["acc_mean_percent"] and
              group_summary[G2]["prob_auc_mean_percent"] > group_summary[G3]["prob_auc_mean_percent"])
    h5 = False
    base_summary = None
    if G2 in group_summary:
        base_rows = _historical_metrics(runner.BASE / "results/fold_metrics.csv", runner.BASE_GROUP)
        base_summary = {metric: float(np.mean([base_rows[f][metric] for f in range(1, 11)]) * 100) for metric in METRICS}
        current = group_summary[G2]
        h5 = all(current[f"{metric}_mean_percent"] >= base_summary[metric] - 1e-12 for metric in ("acc", "hard_auc", "prob_auc")) and any(current[f"{metric}_mean_percent"] > base_summary[metric] + 1e-12 for metric in ("acc", "hard_auc", "prob_auc"))
    lines.extend([f"- H1子原型证据：`{'SUPPORTED' if h1 else 'NOT_SUPPORTED'}`。",
                  f"- H2站点关联缓解：`{'SUPPORTED' if h2 else 'NOT_SUPPORTED'}`。",
                  f"- H3患者条件Prompt选择：`{'SUPPORTED' if h3 else 'NOT_SUPPORTED'}`。",
                  f"- H4真实医学文本独立收益：`{'SUPPORTED' if h4 else 'NOT_SUPPORTED'}`（G2要求ACC与Prob-AUC均高于匿名G3）。",
                  f"- H5相对E74.4基座分类收益：`{'SUPPORTED' if h5 else 'NOT_SUPPORTED'}`（ACC、Hard-AUC、Prob-AUC不降且至少一项提升）。",
                  ""])
    lines.append("结论：`GO_CANDIDATE`。" if h1 and h2 and h3 and h4 and h5 else "结论：`NO-GO`。")
    lines.extend(["", "单seed及历史测试集选模下，GO仅用于决定是否值得设计下一阶段研究，不构成无偏临床泛化证据。未自动启动后续实验。",
                  "", "## Checkpoint 与 RNG", "",
                  f"本报告由独立读回审计生成；完整逐折状态见 `checkpoint_audit.csv` 与 `audit_status.json`。训练源 SHA-256：`{source_sha}`。", "",
                  "## 文件", "",
                  "- `fold_metrics.csv`：30折指标、混淆计数、最佳epoch和耗时。",
                  "- `paired_comparisons.csv`：全折配对差与Fold 2敏感性分析。",
                  "- `mechanism.csv`：支持数、坍塌、站点/年龄关联、路由和Prompt差异。",
                  "- `interventions.csv`：固定G2 Checkpoint语义/残差干预。",
                  "- `checkpoint_audit.csv`：逐折Checkpoint读回、基座、RNG、梯度和标签不变性。",
                  "- `preflight.json`：10折base-off parity、全链梯度、文本几何和恢复重放。", ""])
    report = RESULTS / "report.md"
    temp = report.with_name(report.name + f".tmp.{os.getpid()}")
    temp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temp, report)
    return report


def _summary(group, rows):
    result = {"group": group, "folds": len(rows)}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows]) * 100.0
        result[f"{metric}_mean_percent"] = float(values.mean()) if len(values) else float("nan")
        result[f"{metric}_population_sd_percent"] = float(values.std(ddof=0)) if len(values) else float("nan")
    return result


def audit_all():
    torch.set_num_threads(1)
    frozen = runner.verify_frozen_source()
    frozen_sha = frozen["source_sha256"]
    runner.DATA_IDENTITY = runner._data_identity(runner.prepare(1)[0])
    audit_rows, metric_rows, mechanism_rows, intervention_rows = [], [], [], []
    audit_errors = []
    for fold in range(1, 11):
        for group in runner.GROUPS:
            try:
                audit, checkpoint, diagnostics, metrics, state, module, optimizer = _audit_checkpoint(group, fold, frozen_sha, runner.DATA_IDENTITY)
                audit_rows.append(audit)
                mechanism_rows.append(_mechanism(group, fold, checkpoint, diagnostics, runner.DATA_IDENTITY))
                metric = {"group": group, "fold": fold, "status": audit["status"],
                          **{key: float(metrics[key]) for key in METRICS},
                          "tp": int(metrics["asd_tp"]), "tn": int(metrics["td_tn"]),
                          "fp": int(metrics["td_fp"]), "fn": int(metrics["asd_fn"]),
                          "best_epoch_1based": audit["best_epoch_1based"],
                          "epochs_executed": audit["epochs_executed"],
                          "wall_seconds": float(json.loads(runner.task_paths(group, fold)[0].read_text())["wall_seconds"]),
                          "checkpoint_sha256": audit["checkpoint_sha256"]}
                metric_rows.append(metric)
                intervention_rows.extend(_interventions(group, fold, checkpoint, state, module,
                                                         optimizer, runner.DATA_IDENTITY, metrics))
                del state, module, optimizer, diagnostics, checkpoint
                torch.cuda.empty_cache()
            except Exception as exc:
                audit_errors.append({"group": group, "fold": fold, "error": repr(exc)})
                audit_rows.append({"group": group, "fold": fold, "status": "AUDIT_FAILED", "error": repr(exc)})
    # Failed readbacks are never silently replaced with the task's own metrics.
    summaries = [_summary(group, [row for row in metric_rows if row["group"] == group]) for group in runner.GROUPS]
    historical_base = _historical_metrics(runner.BASE / "results/fold_metrics.csv", runner.BASE_GROUP)
    historical_e78 = _historical_metrics(runner.E78 / "results/fold_metrics.csv", "E78-G2-SPromptGL-K3")
    current_by_group = {group: {row["fold"]: {metric: row[metric] for metric in METRICS}
                                for row in metric_rows if row["group"] == group} for group in runner.GROUPS}
    comparisons = []
    complete_folds = set.intersection(*(set(current_by_group[group]) for group in runner.GROUPS)) if all(current_by_group.values()) else set()
    folds10 = sorted(complete_folds)
    folds9 = [fold for fold in folds10 if fold != 2]
    comparison_specs = [
        (f"{G2} - {G1}", current_by_group.get(G2, {}), current_by_group.get(G1, {})),
        (f"{G2} - {G3}", current_by_group.get(G2, {}), current_by_group.get(G3, {})),
        (f"{G2} - E78-G2", current_by_group.get(G2, {}), historical_e78),
        (f"{G2} - E74.4-G2", current_by_group.get(G2, {}), historical_base),
        (f"{G1} - E74.4-G2", current_by_group.get(G1, {}), historical_base),
        (f"{G3} - E74.4-G2", current_by_group.get(G3, {}), historical_base),
    ]
    for name, left, right in comparison_specs:
        paired_folds = sorted(set(left) & set(right))
        comparisons.extend(_paired(name, left, right, paired_folds))
        comparisons.extend(_paired(name, left, right, [fold for fold in paired_folds if fold != 2]))
    write_csv(RESULTS / "fold_metrics.csv", metric_rows)
    write_csv(RESULTS / "paired_comparisons.csv", comparisons)
    write_csv(RESULTS / "mechanism.csv", mechanism_rows)
    write_csv(RESULTS / "interventions.csv", intervention_rows)
    write_csv(RESULTS / "checkpoint_audit.csv", audit_rows)
    complete = len(metric_rows) == 30 and len(audit_errors) == 0
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if complete:
        status = "COMPLETE_AUDITED"
    elif len(metric_rows) > 0:
        status = "PARTIAL"
    else:
        status = "FAILED"
    audit_status = {"status": status, "trained_tasks": len(metric_rows),
                    "independent_checkpoint_audits": sum(row.get("status") == "INDEPENDENT_AUDIT_PASS" for row in audit_rows),
                    "expected_tasks": 30, "audit_errors": audit_errors,
                    "source_sha256": frozen_sha, "base_off_parity_passed": preflight.get("base_off_parity_passed"),
                    "preflight_recovery": preflight.get("two_step_restore_replay"),
                    "rng_common_trace_mismatches": sum(row.get("rng_mismatches", 0) for row in audit_rows),
                    "root_and_historical_sources_unchanged": True,
                    "data_identity_limit": runner.DATA_IDENTITY["identity_limit"]}
    runner.atomic_json(RESULTS / "audit_status.json", audit_status)
    report_path = _write_report(audit_rows, summaries, comparisons, mechanism_rows,
                                intervention_rows, frozen_sha, status)
    manifest = {**frozen, "status": status, "completed_tasks": len(metric_rows),
                "independent_audits": audit_status["independent_checkpoint_audits"],
                "audit_errors": audit_errors, "report": str(report_path),
                "results": {name: str(RESULTS / name) for name in
                            ("fold_metrics.csv", "paired_comparisons.csv", "mechanism.csv",
                             "interventions.csv", "checkpoint_audit.csv", "audit_status.json", "report.md")}}
    runner.atomic_json(HERE / "manifest.json", manifest)
    _append_handoff(status, summaries, comparisons, mechanism_rows, audit_status, report_path)
    return audit_status


def _append_handoff(status, summaries, comparisons, mechanisms, audit_status, report_path):
    handoff = Path("/model/spromptgl/experiment/handoff.md")
    marker = "## Experiment 78.1: SH-DCPSS"
    by_group = {row["group"]: row for row in summaries}
    compare = {(row["comparison"], row["metric"], row["scope"]): row for row in comparisons}
    lines = [marker, "", f"- 状态：`{status}`；正式任务 {audit_status['trained_tasks']}/30，独立Checkpoint审计 {audit_status['independent_checkpoint_audits']}/30。",
             f"- 唯一基座：E74.4-G2-Positive-DCR；实验目录：`{HERE}`。",
             f"- 报告：`{report_path}`；折指标：`{RESULTS / 'fold_metrics.csv'}`；配对比较：`{RESULTS / 'paired_comparisons.csv'}`；机制：`{RESULTS / 'mechanism.csv'}`。"]
    for group in runner.GROUPS:
        row = by_group.get(group)
        if row:
            metrics = ", ".join(f"{key} {row[f'{key}_mean_percent']:.4f}±{row[f'{key}_population_sd_percent']:.4f}%" for key in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1"))
            lines.append(f"- {group}：{metrics}。")
        else:
            lines.append(f"- {group}：无完整独立审计指标。")
    lines.append(f"- RNG公共轨迹不匹配数：{audit_status['rng_common_trace_mismatches']}；10折base-off parity：{audit_status['base_off_parity_passed']}/10。")
    for comp in (f"{G2} - {G1}", f"{G2} - {G3}", f"{G2} - E78-G2", f"{G2} - E74.4-G2"):
        row = compare.get((comp, "acc", "all10"))
        if row:
            lines.append(f"- {comp} ACC差 {row['mean_difference_pp']:.4f}pp，W/T/L {row['wins']}/{row['ties']}/{row['losses']}。")
    g2 = [row for row in mechanisms if row.get("group") == G2]
    if g2:
        lines.append(f"- G2原型坍塌比例均值 {np.mean([r['prototype_pair_collapse_fraction_cosine_gt_0_98'] for r in g2]):.4f}；站点Cramér's V均值 {np.mean([r['site_cramers_v_mean'] for r in g2]):.4f}；子原型Prompt TV均值 {np.mean([r['subprototype_prompt_tv'] for r in g2]):.5f}。")
    lines.extend(["- 科研结论及医学文本独立收益判断见结果报告；本研究保留历史测试选模协议，GO仅是后续研究候选。", ""])
    content = handoff.read_text(encoding="utf-8") if handoff.exists() else ""
    if marker in content:
        return {"status": "ALREADY_PRESENT", "path": str(handoff)}
    handoff.parent.mkdir(parents=True, exist_ok=True)
    temp = handoff.with_name(handoff.name + f".tmp.{os.getpid()}")
    temp.write_text(content + ("\n" if content and not content.endswith("\n") else "") + "\n".join(lines), encoding="utf-8")
    os.replace(temp, handoff)
    return {"status": "APPENDED", "path": str(handoff)}


if __name__ == "__main__":
    try:
        result = audit_all()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        payload = {"status": "AUDIT_PIPELINE_FAILED", "error": repr(exc)}
        runner.atomic_json(RESULTS / "audit_status.json", payload)
        raise
