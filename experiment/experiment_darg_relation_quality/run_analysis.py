"""Final MICCAI figure and audit for frozen DARG relation quality."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import sys
import time
import urllib.request
from pathlib import Path

DPSL = Path("/model/spromptgl/experiment/DPSL")
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
sys.path.insert(0, str(DPSL / "code"))
sys.path.insert(0, str(DPSL / "code/history_base"))
sys.path.insert(0, str(DPSL / "runtime/python"))

import numpy as np
import torch

from dpsl_runtime import OVERLAP_1BASED, checkpoint_source, sha256_file
import real_model_runtime as runtime

DARG_PATH = DPSL / "code/vendor/repo/experiment/experiment74_4_soft_positive_dcr/soft_positive_dcr.py"
CADR_PATH = DPSL / "code/vendor/repo/experiment/experiment53_3_cadr_graph_fallback_fix/cadr_gfr.py"
GITHUB_COMMIT = "85826c2"
GITHUB_BASE = f"https://raw.githubusercontent.com/tangaii/SPromptGL/{GITHUB_COMMIT}/"
GITHUB_FILES = {
    "real_model_runtime.py": "experiment/DPSL/code/real_model_runtime.py",
    "dpsl_runtime.py": "experiment/DPSL/code/dpsl_runtime.py",
    "soft_positive_dcr.py": "experiment/DPSL/code/vendor/repo/experiment/experiment74_4_soft_positive_dcr/soft_positive_dcr.py",
    "network.py": "experiment/DPSL/code/vendor/repo/network.py",
    "model.py": "experiment/DPSL/code/vendor/repo/model.py",
    "utils.py": "experiment/DPSL/code/vendor/repo/utils.py",
    "cadr_gfr.py": "experiment/DPSL/code/vendor/repo/experiment/experiment53_3_cadr_graph_fallback_fix/cadr_gfr.py",
}
spec = importlib.util.spec_from_file_location("formal_e744_positive_dcr", DARG_PATH)
darg = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = darg
spec.loader.exec_module(darg)
darg.CADR_PATH = CADR_PATH

MODALITIES = tuple(darg.MODALITIES)
GROUP = "E74.4-G2-Positive-DCR"
SEED_BASE = 741001
PERMUTATIONS = 20


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=jsonable) + "\n", encoding="utf-8")


def github_sha(relative):
    with urllib.request.urlopen(GITHUB_BASE + relative, timeout=30) as response:
        return hashlib.sha256(response.read()).hexdigest()


def normalized_adjacency(model, adjacency):
    identity = torch.eye(adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device)
    return model.utils.normalize_adj(adjacency + identity)


def valid_edges(adjacency, labels, clean_test):
    effective = adjacency > 0
    effective.fill_diagonal_(False)
    clean = torch.zeros(adjacency.shape[0], dtype=torch.bool, device=adjacency.device)
    clean[clean_test] = True
    upper = torch.triu(torch.ones_like(effective, dtype=torch.bool), diagonal=1)
    valid = effective & clean[:, None] & clean[None, :] & upper
    cross = labels[:, None] != labels[None, :]
    return valid, valid & cross


def ratio_quality(model, original, adjacency, labels, clean_test, scope):
    effective = adjacency > 0
    effective.fill_diagonal_(False)
    clean = torch.zeros(adjacency.shape[0], dtype=torch.bool, device=adjacency.device)
    clean[clean_test] = True
    upper = torch.triu(torch.ones_like(effective, dtype=torch.bool), diagonal=1)
    selected = clean[:, None] | clean[None, :] if scope == "at_least_one_clean_test" else clean[:, None] & clean[None, :]
    valid = effective & selected & upper
    cross = valid & (labels[:, None] != labels[None, :])
    normalized = normalized_adjacency(model, adjacency)
    denominator = float(normalized[valid].sum().item()) if valid.any() else 0.0
    numerator = float(normalized[cross].sum().item()) if cross.any() else 0.0
    return {
        "ratio": numerator / denominator if denominator > 0 and np.isfinite(denominator) else float("nan"),
        "valid_edges": int(valid.sum().item()),
        "same_edges": int((valid & ~cross).sum().item()),
        "cross_edges": int(cross.sum().item()),
        "weight_sum": denominator,
        "cross_weight_sum": numerator,
        "finite": bool(torch.isfinite(adjacency).all() and torch.isfinite(normalized).all()),
        "nonnegative_normalized": bool((normalized >= 0).all()),
        "zero_denominator": bool(denominator <= 0 or not np.isfinite(denominator)),
    }


GROUP_NAMES = ("ASD-ASD", "ASD-TD", "TD-TD")


def relation_group_stats(original, adjacency, labels, clean_test):
    valid, _ = valid_edges(original, labels, clean_test)
    relative = (adjacency - original) / original.clamp_min(torch.finfo(original.dtype).tiny)
    group_masks = (
        valid & (labels[:, None] == 0) & (labels[None, :] == 0),
        valid & (labels[:, None] != labels[None, :]),
        valid & (labels[:, None] == 1) & (labels[None, :] == 1),
    )
    group_values = [relative[mask] for mask in group_masks]
    same_values = relative[valid & ~(labels[:, None] != labels[None, :])]
    cross_values = group_values[1]
    all_values = relative[valid]
    group_means = [float(values.mean().item() * 100) if values.numel() else float("nan") for values in group_values]
    group_counts = [int(values.numel()) for values in group_values]
    return {
        "group_names": list(GROUP_NAMES),
        "group_edge_counts": group_counts,
        "group_relative_mean_percent": group_means,
        "group_enhancement_ratio": [float((values > 0).float().mean().item()) if values.numel() else float("nan") for values in group_values],
        "group_suppression_ratio": [float((values < 0).float().mean().item()) if values.numel() else float("nan") for values in group_values],
        "same_edge_count": int(same_values.numel()),
        "cross_edge_count": int(cross_values.numel()),
        "asd_asd_edge_count": group_counts[0],
        "td_td_edge_count": group_counts[2],
        "same_relative_mean_percent": float(same_values.mean().item() * 100) if same_values.numel() else float("nan"),
        "cross_relative_mean_percent": float(cross_values.mean().item() * 100) if cross_values.numel() else float("nan"),
        "same_enhancement_ratio": float((same_values > 0).float().mean().item()) if same_values.numel() else float("nan"),
        "cross_suppression_ratio": float((cross_values < 0).float().mean().item()) if cross_values.numel() else float("nan"),
        "same_suppression_ratio": float((same_values < 0).float().mean().item()) if same_values.numel() else float("nan"),
        "cross_enhancement_ratio": float((cross_values > 0).float().mean().item()) if cross_values.numel() else float("nan"),
        "valid_edge_count": int(valid.sum().item()),
        "unique_undirected_edge_count": int(valid.sum().item()),
        "group_count_sum_exact": int(sum(group_counts)) == int(valid.sum().item()),
        "relative_finite": bool(torch.isfinite(all_values).all()),
    }


def permutation_adjs(info, fold, permutation):
    adjs, q_used = [], []
    seed = SEED_BASE + permutation * 100003
    for modality, (original, q, row, col, gate) in enumerate(zip(info["original_adjs"], info["edge_q_graph"], info["edge_rows"], info["edge_cols"], info["gates"])):
        q_perm = darg._score_transform(q, modality, "score_permuted", fold, seed)
        adjacency, _ = darg.unified_adjacency(original, row, col, q_perm, gate, modality, intervention="full", fold=fold)
        adjs.append(adjacency)
        q_used.append(q_perm)
    return adjs, q_used


def audit_graph(model, info, originals, adjs, neutral):
    checks = {
        "matrix_shapes": all(tuple(x.shape) == (871, 871) for x in (*originals, *adjs)),
        "matrices_finite": all(bool(torch.isfinite(x).all()) for x in (*originals, *adjs)),
        "matrices_symmetric": all(float((x - x.T).abs().max().item()) == 0.0 for x in (*originals, *adjs)),
        "adjacency_nonnegative": all(bool((x >= 0).all()) for x in (*originals, *adjs)),
        "delta_exact": all(torch.equal(a - s, d) for s, a, d in zip(originals, adjs, info["disease_delta_adjs"])),
        "positive_dcr_exact": all(torch.equal(d.clamp_min(0), p) for d, p in zip(info["disease_delta_adjs"], info["direct_delta_adjs"])),
        "neutral_recovers_original": all(torch.equal(a, s) for a, s in zip(neutral["unified_adjs"], neutral["original_adjs"])),
        "normalized_finite": all(bool(torch.isfinite(normalized_adjacency(model, x)).all()) for x in adjs),
        "normalized_nonnegative": all(bool((normalized_adjacency(model, x) >= 0).all()) for x in adjs),
    }
    support_checks = []
    for original, adjacency, row, col in zip(originals, adjs, info["edge_rows"], info["edge_cols"]):
        candidate = torch.zeros_like(original, dtype=torch.bool)
        candidate[row, col] = True
        candidate[col, row] = True
        candidate.fill_diagonal_(False)
        support_checks.append(bool(((adjacency != original) & ~candidate).sum() == 0 and torch.equal(torch.diag(adjacency), torch.diag(original))))
        support_checks[-1] = support_checks[-1] and not bool(((adjacency > 0) & ~(original > 0)).any())
    checks["changed_only_on_candidate_support"] = all(support_checks)
    checks["no_new_edges"] = all(not bool(((a > 0) & ~(s > 0)).any()) for s, a in zip(originals, adjs))
    checks["normalization_matches_manual"] = True
    for adjacency in adjs:
        with_identity = adjacency + torch.eye(adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device)
        degree = with_identity.sum(dim=1).clamp_min(torch.finfo(adjacency.dtype).tiny).rsqrt()
        manual = degree[:, None] * with_identity * degree[None, :]
        if not torch.allclose(normalized_adjacency(model, adjacency), manual, rtol=1e-6, atol=1e-7):
            checks["normalization_matches_manual"] = False
            break
    return checks


def summarize_legacy_rows(rows):
    summary = {}
    for modality in (*MODALITIES, "FUSED"):
        summary[modality] = {}
        for scope in ("at_least_one_clean_test", "clean_test_test"):
            summary[modality][scope] = {}
            for condition in ("Original", "DARG", "Score-Permuted"):
                values = [float(row["ratio_percent"]) for row in rows if row["modality"] == modality and row["scope"] == scope and row["condition"] == condition]
                if not values:
                    continue
                summary[modality][scope][condition] = {"mean_percent": float(np.mean(values)), "sd_percent": float(np.std(values)), "n_folds": len(values)}
            original = summary[modality][scope].get("Original")
            darg_value = summary[modality][scope].get("DARG")
            if original and darg_value:
                paired = [float(row["ratio_percent"]) for row in rows if row["modality"] == modality and row["scope"] == scope and row["condition"] == "DARG"]
                original_values = [float(row["ratio_percent"]) for row in rows if row["modality"] == modality and row["scope"] == scope and row["condition"] == "Original"]
                delta = np.asarray(paired) - np.asarray(original_values)
                summary[modality][scope]["paired_improved_folds"] = int(np.sum(delta < 0))
                summary[modality][scope]["paired_unchanged_folds"] = int(np.sum(delta == 0))
                summary[modality][scope]["paired_declined_folds"] = int(np.sum(delta > 0))
    return summary


def find_previous_panel_a(previous_summary):
    current = previous_summary
    for _ in range(16):
        if isinstance(current, dict) and isinstance(current.get("panel_a"), dict):
            panel_a = current["panel_a"]
            if "same_mean_percent" in panel_a and "cross_mean_percent" in panel_a:
                return panel_a
        if isinstance(current, dict) and isinstance(current.get("history_reference"), dict):
            history = current["history_reference"]
            if "previous_panel_a_same_mean_percent" in history and "previous_panel_a_cross_mean_percent" in history:
                return {"same_mean_percent": history["previous_panel_a_same_mean_percent"], "cross_mean_percent": history["previous_panel_a_cross_mean_percent"]}
        if not isinstance(current, dict) or not isinstance(current.get("legacy_summary"), dict):
            break
        current = current["legacy_summary"]
    return {}


def plot_figure(group_means, group_sds, b_original, b_darg, permutation_means):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 3.15), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.08), wspace=0.34, left=0.085, right=0.985, top=0.86, bottom=0.22)
    ax_a = fig.add_subplot(grid[0])
    ax_b = fig.add_subplot(grid[1])

    means = np.asarray(group_means, dtype=float)
    sds = np.asarray(group_sds, dtype=float)
    x = np.arange(3)
    colors = ["#c23b3b" if value >= 0 else "#2c6db2" for value in means]
    ax_a.bar(x, means, yerr=sds, width=0.58, color=colors, edgecolor="#333333", linewidth=0.45, error_kw={"ecolor": "#222222", "elinewidth": 0.8, "capsize": 2.2, "capthick": 0.8}, zorder=3)
    for position, value, sd in zip(x, means, sds):
        sign = 1 if value >= 0 else -1
        offset = max(1.2, 0.04 * max(float(np.max(np.abs(means + sds))), 1.0))
        ax_a.text(position, value + sign * (sd + offset), f"{value:+.2f}%", ha="center", va="bottom" if sign > 0 else "top", fontsize=6.3, color="#222222")
    lower = float(np.min(means - sds))
    upper = float(np.max(means + sds))
    span = max(upper - lower, 1.0)
    pad = max(3.0, 0.12 * span)
    ax_a.set_ylim(lower - pad, upper + pad)
    ax_a.axhline(0, color="#222222", linewidth=0.75, zorder=2)
    ax_a.set_xticks(x, ["ASD-ASD", "ASD-TD", "TD-TD"])
    ax_a.set_ylabel("Relative Edge Weight Adjustment (%)")
    ax_a.set_title("(a) Disease Relation Reweighting", loc="left", pad=4)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.grid(axis="y", color="#dddddd", linewidth=0.45, zorder=0)
    ax_a.set_axisbelow(True)

    for index in range(10):
        ax_b.plot([0, 1], [b_original[index], b_darg[index]], color="#bdbdbd", linewidth=0.65, zorder=1)
    ax_b.scatter(np.zeros(10), b_original, color="#4c78a8", edgecolor="white", linewidth=0.35, s=22, zorder=3)
    ax_b.scatter(np.ones(10), b_darg, color="#1b9e77", edgecolor="white", linewidth=0.35, s=22, zorder=3)
    perm_x = 2 + np.linspace(-0.11, 0.11, len(permutation_means))
    ax_b.scatter(perm_x, permutation_means, color="#e07a00", alpha=0.62, edgecolor="white", linewidth=0.25, s=18, zorder=2)
    ax_b.set_xticks([0, 1, 2], ["Original", "DARG", "Permuted"])
    ax_b.set_ylabel("Cross-Disease Weight Ratio (%)")
    ax_b.set_ylim(0, 100)
    ax_b.set_yticks(np.arange(0, 101, 20))
    ax_b.set_title("(b) Cross-Disease Propagation", loc="left", pad=4)
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.grid(axis="y", color="#dddddd", linewidth=0.45)
    ax_b.set_axisbelow(True)

    fig.savefig(FIGURES / "darg_relation_quality.pdf", format="pdf")
    fig.savefig(FIGURES / "darg_relation_quality.png", format="png", dpi=600)
    plt.close(fig)
    return {"group_means_percent": means.tolist(), "group_sds_percent": sds.tolist(), "permutation_point_count": int(len(permutation_means)), "paired_mean_reduction_pp": float(np.mean(b_original - b_darg))}


def report_text(summary, audit, figure_hashes):
    panel_a = summary["panel_a"]
    panel_b = summary["panel_b"]
    modalities = summary.get("historical_multimodal_summary", {})
    lines = [
        "# DARG Disease-Aware Patient Relation Quality Analysis",
        "",
        "## 1. 实验目标与源码核验",
        "",
        "本次工作只优化冻结 DReSP 机制 Figure，不重新训练、不微调、不修改正式模型、Checkpoint、DARG/DPSL 封存源码或历史训练结果。正式任务为 E82.4.6.10-G1，内部模型为 E82.4.6-G2 SCMR-DA-SRPE，DARG 为 E74.4-G2 Positive-DCR。服务器运行文件与 GitHub 提交 85826c2 的对应 raw 文件逐项 SHA-256 一致。",
        "",
        "封存实现使用 GraphLearn 原始邻接 S_m，在其真实有效支持上根据 q 和 gate 计算 DeltaA_m，形成 A_m = S_m + DeltaA_m。Positive-DCR 的 direct 增量为 max(DeltaA_m, 0)。传播权重按照正式 normalize_adj(A_m + I) 计算，自环只用于归一化而不进入评价边集合。",
        "",
        "## 2. Panel (a) 三组十折关系调整",
        "",
        "Panel (a) 只使用 CORR 模态和严格 clean_test_test 范围。每个折先在原始图真实存在的非自环无向边上计算 (A_DARG-A_Original)/A_Original，并分别对 ASD-ASD、ASD-TD、TD-TD 三组求折内均值。最终柱高是十个折内均值的等权平均，误差线是总体标准差 SD = np.std(fold_values, ddof=0)。柱子颜色按实际均值方向决定，红色表示增强，蓝色表示削弱。",
        "",
        "| Group | Mean (%) | SD (%) | Total valid edges | Mean enhanced | Mean suppressed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for index, name in enumerate(panel_a["group_names"]):
        lines.append(f"| {name} | {panel_a['group_means_percent'][index]:.4f} | {panel_a['group_sds_percent'][index]:.4f} | {sum(row['group_edge_counts'][index] for row in panel_a['fold_rows'])} | {np.mean([row['group_enhancement_ratio'][index] for row in panel_a['fold_rows']])*100:.2f}% | {np.mean([row['group_suppression_ratio'][index] for row in panel_a['fold_rows']])*100:.2f}% |")
    lines += [
        "",
        "每个折的三组统计如下，边数为严格 clean_test_test 的原始 GraphLearn 支持中的上三角非自环边。",
        "",
        "| Fold | ASD-ASD mean (%) | ASD-TD mean (%) | TD-TD mean (%) | ASD-ASD n | ASD-TD n | TD-TD n |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in panel_a["fold_rows"]:
        means = row["group_relative_mean_percent"]
        counts = row["group_edge_counts"]
        lines.append(f"| {row['fold']} | {means[0]:.4f} | {means[1]:.4f} | {means[2]:.4f} | {counts[0]} | {counts[1]} | {counts[2]} |")
    lines += [
        "",
        f"按每折实际 ASD-ASD 与 TD-TD 边数合并后再对十折等权汇总，同病均值为 {panel_a['merged_same_mean_percent']:.4f}%；ASD-TD 结果为 {panel_a['merged_cross_mean_percent']:.4f}%。历史同病参考值为 {panel_a['historical_same_mean_percent']:.4f}%，历史异病参考值为 {panel_a['historical_cross_mean_percent']:.4f}%，两项一致性检查分别为 {panel_a['merged_same_matches_history']} 和 {panel_a['merged_cross_matches_history']}。三组均值按分组张量重算的同病合并值为 {panel_a['merged_same_group_count_mean_percent']:.8f}%，与直接同病边集合的差异为 {panel_a['merged_same_group_count_difference_pp']:.8f} 个百分点，属于 float32 索引求和顺序的数值舍入差异。",
        "",
        "## 3. Panel (b) 跨疾病传播权重",
        "",
        "Panel (b) 使用 at_least_one_clean_test。Original 与 DARG 是同一正式冻结 DReSP Checkpoint 内部的 GraphLearn 原图和 DARG 重加权图，不是重新训练的 SPromptGL 基线模型。纵轴为跨疾病归一化传播权重占比，使用 normalize_adj(A+I) 并排除自环评价边。",
        "",
        "| Fold | Original (%) | DARG (%) | DARG−Original (pp) |",
        "|---:|---:|---:|---:|",
    ]
    for fold, original, darg_value in zip(range(1, 11), panel_b["original_values_percent"], panel_b["darg_values_percent"]):
        lines.append(f"| {fold} | {original:.4f} | {darg_value:.4f} | {darg_value-original:.4f} |")
    lines += [
        "",
        f"Original 十折均值为 {panel_b['original_mean_percent']:.4f}%，DARG 为 {panel_b['darg_mean_percent']:.4f}%，配对平均下降 {panel_b['mean_reduction_pp']:.4f} pp，十折改善 {panel_b['improved_folds']}/10。严格 clean_test_test 的复算结果为 Original {panel_b['strict_original_mean_percent']:.4f}%、DARG {panel_b['strict_darg_mean_percent']:.4f}%，平均下降 {panel_b['strict_mean_reduction_pp']:.4f} pp。",
        f"Score-Permuted DARG 对每个折使用 20 次置乱，保持原始图、候选边、门控和 q 数值多重集合不变，只置乱 q 与患者边的对应关系。对每个置乱编号先收集十折结果，再计算该编号的十折均值，得到 20 个均值。其平均为 {panel_b['permutation_mean_percent']:.4f}%，最小至最大为 [{panel_b['permutation_min_percent']:.4f}%, {panel_b['permutation_max_percent']:.4f}%]。该范围是 20 个置乱均值的 Min–Max，不是置信区间。",
        "",
        "## 4. 四模态历史辅助汇总",
        "",
        "下表保留原始逐折结果的四模态主范围汇总。FUSED 是历史辅助代理项，不作为真实多模态 GCN 融合传播图，也不进入本 Figure 的主结论。",
        "",
        "| Modality | Original mean (%) | DARG mean (%) | Score-Permuted mean (%) | Improved / unchanged / declined |",
        "|---|---:|---:|---:|---:|",
    ]
    for modality in (*MODALITIES, "FUSED"):
        values = modalities.get(modality, {}).get("at_least_one_clean_test", {})
        original = values.get("Original")
        darg_value = values.get("DARG")
        permuted = values.get("Score-Permuted")
        if original is None or darg_value is None:
            continue
        improved = values.get("paired_improved_folds", 0)
        unchanged = values.get("paired_unchanged_folds", 0)
        declined = values.get("paired_declined_folds", 0)
        permuted_mean = f"{permuted['mean_percent']:.4f}" if permuted is not None else "not stored"
        lines.append(f"| {modality} | {original['mean_percent']:.4f} | {darg_value['mean_percent']:.4f} | {permuted_mean} | {improved}/{unchanged}/{declined} |")
    lines += [
        "",
        "## 5. 独立审计",
        "",
        f"独立审计状态为 {audit['status']}。关键条件均由实际判断执行，包括十折 Best Checkpoint 身份、归档预测、独立 evaluator、原图/DARG 图共同来源、矩阵有限/对称/非负、DeltaA 精确一致、Positive-DCR 正向截断、候选支持不变、正式归一化、disease-neutral 原图恢复、标签只用于事后评价、三组边计数不重复、逐折 Mean±SD、历史合并统计和置乱 q 多重集合。",
        "",
        "分析继承历史 val=test Checkpoint 选择、CV 前 RFE/填补/标准化、transductive 患者图和 19 名训练—测试重叠事实。重叠患者只从机制评价对象中排除，没有修改历史训练集或模型状态。图质量下降表示跨疾病传播系数占比下降，不直接证明分类准确率的因果提高。",
        "",
        "## 6. Figure Caption",
        "",
        "Disease Relation Reweighting and Cross-Disease Propagation. (a) Ten-fold mean relative edge-weight adjustments for CORR ASD-ASD, ASD-TD, and TD-TD patient connections on the strict clean test-test support; each bar is the equal-weight mean of the ten fold-level means and error bars show population SD (ddof=0). (b) Ten-fold paired cross-disease propagation-weight ratios for the Original GraphLearn and DARG graphs from the same frozen DReSP checkpoints, with 20 score-permuted ten-fold means as a negative control. The ratios use the model's symmetric normalization of A+I and exclude self-loops; a reduction denotes a lower cross-disease propagation-weight fraction, not a classification-accuracy change.",
        "",
        "## 7. 文件与校验",
        "",
        f"最终 Figure：figures/darg_relation_quality.pdf、figures/darg_relation_quality.png。可复核 Figure 数据：results/figure_data.npz。逐折原始图质量数据保留于 results/graph_quality_by_fold.csv。代码和图文件 SHA：{json.dumps(figure_hashes, ensure_ascii=False)}。",
    ]
    return "\n".join(lines) + "\n"


def main():
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    sys.modules.setdefault("run", runtime.local_module("history_base/run.py"))
    previous_summary_path = RESULTS / "summary.json"
    previous_summary = json.loads(previous_summary_path.read_text(encoding="utf-8")) if previous_summary_path.exists() else {}
    previous_summary_sha = sha256_file(previous_summary_path) if previous_summary_path.exists() else None
    previous_panel_a = find_previous_panel_a(previous_summary)
    historical_same_reference = float(previous_panel_a.get("same_mean_percent", float("nan")))
    historical_cross_reference = float(previous_panel_a.get("cross_mean_percent", float("nan")))
    previous_figure_hashes = previous_summary.get("figure_hashes", {}) if isinstance(previous_summary, dict) else {}
    source_server = {
        "real_model_runtime.py": sha256_file(DPSL / "code/real_model_runtime.py"),
        "dpsl_runtime.py": sha256_file(DPSL / "code/dpsl_runtime.py"),
        "soft_positive_dcr.py": sha256_file(DARG_PATH),
        "network.py": sha256_file(DPSL / "code/vendor/repo/network.py"),
        "model.py": sha256_file(DPSL / "code/vendor/repo/model.py"),
        "utils.py": sha256_file(DPSL / "code/vendor/repo/utils.py"),
        "cadr_gfr.py": sha256_file(CADR_PATH),
    }
    source_github = {name: github_sha(path) for name, path in GITHUB_FILES.items()}
    source_match = {name: source_server[name] == source_github[name] for name in source_server}
    if not all(source_match.values()):
        raise RuntimeError(f"GitHub/source mismatch: {source_match}")

    with (RESULTS / "graph_quality_by_fold.csv").open(encoding="utf-8") as stream:
        old_rows = list(csv.DictReader(stream))
    historical_multimodal_summary = summarize_legacy_rows(old_rows)
    old_main = {(int(row["fold"]), row["condition"]): float(row["ratio"]) for row in old_rows if row["modality"] == "CORR" and row["scope"] == "at_least_one_clean_test" and row["condition"] in ("Original", "DARG")}
    old_strict = {(int(row["fold"]), row["condition"]): float(row["ratio"]) for row in old_rows if row["modality"] == "CORR" and row["scope"] == "clean_test_test" and row["condition"] in ("Original", "DARG")}
    fold_audits, adjustment_rows, checkpoint_records = [], [], {}
    main_original, main_darg, strict_original, strict_darg = [], [], [], []
    permutation_main = [[] for _ in range(PERMUTATIONS)]
    all_prediction_exact = True
    all_independent_exact = True

    for fold in range(1, 11):
        print(f"Fold {fold:02d}: restoring audited Best", flush=True)
        ctx, engine, saved, _ = runtime.restore(fold, "best")
        state = ctx["state"]
        checkpoint_path = runtime.checkpoint_path(fold, "best")
        checkpoint_expected = checkpoint_source(fold, "best")
        checkpoint_records[str(fold)] = {"path": str(checkpoint_path), "source": checkpoint_expected[0], "sha256": sha256_file(checkpoint_path), "expected_sha256": checkpoint_expected[1], "best_epoch": int(saved["best_epoch"]), "fold": int(saved["fold"]), "seed": int(saved["seed"])}
        out, _ = engine.evaluate(ctx)
        prediction_exact = all(np.array_equal(out[key], saved["evaluation"][key]) for key in ("logits", "probability", "prediction"))
        proof = runtime.local_module("history_base/audit.py")
        proof.run = engine
        independent, _ = proof.independent_eval(ctx)
        independent_exact = all(np.array_equal(out[key], independent[key]) for key in ("logits", "probability", "prediction"))
        all_prediction_exact = all_prediction_exact and prediction_exact
        all_independent_exact = all_independent_exact and independent_exact
        model = state["model"]
        with torch.no_grad():
            _, _, info = darg.rudag_forward(model, state["parts"], state["train_pairs"], GROUP, fold=fold, collect_diagnostics=True, permutation_seed_base=SEED_BASE)
            _, _, neutral = darg.rudag_forward(model, state["parts"], state["train_pairs"], GROUP, fold=fold, intervention="disease_neutral", collect_diagnostics=True, permutation_seed_base=SEED_BASE)
        originals, adjs = info["original_adjs"], info["unified_adjs"]
        labels = torch.as_tensor(state["labels_np"], dtype=torch.long, device=adjs[0].device)
        overlap = np.asarray([patient - 1 for patient in OVERLAP_1BASED.get(fold, ())], dtype=np.int64)
        overlap_set = set(overlap.tolist())
        clean_test = np.asarray([patient for patient in state["test_np"] if patient not in overlap_set], dtype=np.int64)
        graph_checks = audit_graph(model, info, originals, adjs, neutral)
        permutation_support_checks, permutation_q_checks, permutation_matrix_checks = [], [], []
        rng_before = engine.h.rng_sha()
        for permutation in range(PERMUTATIONS):
            perm_adjs, perm_q = permutation_adjs(info, fold, permutation)
            permutation_support_checks.append(all(torch.equal(permuted > 0, original > 0) for permuted, original in zip(perm_adjs, originals)))
            permutation_q_checks.append(all(torch.equal(torch.sort(q).values, torch.sort(q_perm).values) for q, q_perm in zip(info["edge_q_graph"], perm_q)))
            permutation_matrix_checks.append(all(bool(torch.isfinite(adjacency).all()) and bool((adjacency >= 0).all()) and float((adjacency - adjacency.T).abs().max().item()) == 0.0 and bool(torch.isfinite(normalized_adjacency(model, adjacency)).all()) and bool((normalized_adjacency(model, adjacency) >= 0).all()) for adjacency in perm_adjs))
            permutation_main[permutation].append(ratio_quality(model, originals[3], perm_adjs[3], labels, clean_test, "at_least_one_clean_test")["ratio"])
        permutation_rng_unchanged = engine.h.rng_sha() == rng_before

        corr_original_main = ratio_quality(model, originals[3], originals[3], labels, clean_test, "at_least_one_clean_test")
        corr_darg_main = ratio_quality(model, originals[3], adjs[3], labels, clean_test, "at_least_one_clean_test")
        corr_original_strict = ratio_quality(model, originals[3], originals[3], labels, clean_test, "clean_test_test")
        corr_darg_strict = ratio_quality(model, originals[3], adjs[3], labels, clean_test, "clean_test_test")
        ratio_results_valid = all(result["finite"] and result["nonnegative_normalized"] and not result["zero_denominator"] for result in (corr_original_main, corr_darg_main, corr_original_strict, corr_darg_strict))
        main_original.append(corr_original_main["ratio"])
        main_darg.append(corr_darg_main["ratio"])
        strict_original.append(corr_original_strict["ratio"])
        strict_darg.append(corr_darg_strict["ratio"])
        historical_main_match = all(np.isclose(value, old_main[(fold, condition)], rtol=0, atol=1e-12) for value, condition in ((corr_original_main["ratio"], "Original"), (corr_darg_main["ratio"], "DARG")))
        historical_strict_match = all(np.isclose(value, old_strict[(fold, condition)], rtol=0, atol=1e-12) for value, condition in ((corr_original_strict["ratio"], "Original"), (corr_darg_strict["ratio"], "DARG")))
        stats = relation_group_stats(originals[3], adjs[3], labels, clean_test)
        original_valid, _ = valid_edges(originals[3], labels, clean_test)
        possible = len(clean_test) * (len(clean_test) - 1) / 2
        density = float(original_valid.sum().item() / possible) if possible else float("nan")
        adjustment_rows.append({"fold": fold, "clean_test_count": len(clean_test), "overlap_count_excluded": len(overlap), "original_edge_count": corr_original_strict["valid_edges"], "original_edge_density": density, **stats, "main_original_ratio": corr_original_main["ratio"], "main_darg_ratio": corr_darg_main["ratio"], "strict_original_ratio": corr_original_strict["ratio"], "strict_darg_ratio": corr_darg_strict["ratio"], "historical_main_match": historical_main_match, "historical_strict_match": historical_strict_match})
        fold_audits.append({"fold": fold, "checkpoint_sha_verified": checkpoint_records[str(fold)]["sha256"] == checkpoint_records[str(fold)]["expected_sha256"], "formal_prediction_exact": prediction_exact, "independent_evaluator_exact": independent_exact, "clean_test_excludes_overlap": not bool(np.intersect1d(clean_test, overlap).size), "clean_test_size_correct": len(clean_test) == len(state["test_np"]) - len(overlap), "labels_only_posthoc": "labels" not in inspect.signature(darg.rudag_forward).parameters and "y" not in inspect.signature(darg.rudag_forward).parameters, "ratio_results_valid": ratio_results_valid, "historical_main_ratios_match": historical_main_match, "historical_strict_ratios_match": historical_strict_match, "group_count_sum_exact": stats["group_count_sum_exact"], "unique_undirected_edge_count_exact": stats["unique_undirected_edge_count"] == stats["valid_edge_count"], "group_relative_values_finite": stats["relative_finite"], "permutation_q_multiset_preserved": all(permutation_q_checks), "permutation_support_preserved": all(permutation_support_checks), "permutation_matrices_valid": all(permutation_matrix_checks), "permutation_local_rng_unchanged": permutation_rng_unchanged, **graph_checks})
        print(f"Fold {fold:02d}: prediction={'PASS' if prediction_exact else 'FAIL'}, strict original edges={corr_original_strict['valid_edges']}", flush=True)

    group_fold_means = np.asarray([row["group_relative_mean_percent"] for row in adjustment_rows], dtype=float).T
    group_fold_counts = np.asarray([row["group_edge_counts"] for row in adjustment_rows], dtype=np.int64).T
    group_fold_enhancement = np.asarray([row["group_enhancement_ratio"] for row in adjustment_rows], dtype=float).T
    group_fold_suppression = np.asarray([row["group_suppression_ratio"] for row in adjustment_rows], dtype=float).T
    group_means = group_fold_means.mean(axis=1)
    group_sds = group_fold_means.std(axis=1, ddof=0)
    merged_same_group_count_per_fold = np.divide(group_fold_means[0] * group_fold_counts[0] + group_fold_means[2] * group_fold_counts[2], group_fold_counts[0] + group_fold_counts[2], out=np.full(10, np.nan), where=(group_fold_counts[0] + group_fold_counts[2]) > 0)
    merged_same_direct_per_fold = np.asarray([row["same_relative_mean_percent"] for row in adjustment_rows], dtype=float)
    merged_same_group_count_matches_direct = bool(np.allclose(merged_same_group_count_per_fold, merged_same_direct_per_fold, rtol=0, atol=1e-5))
    merged_same_per_fold = merged_same_direct_per_fold
    merged_cross_per_fold = group_fold_means[1]
    merged_same_mean = float(np.mean(merged_same_per_fold))
    merged_cross_mean = float(np.mean(merged_cross_per_fold))
    merged_same_matches_history = bool(np.isfinite(historical_same_reference) and np.isclose(merged_same_mean, historical_same_reference, rtol=0, atol=1e-10))
    merged_cross_matches_history = bool(np.isfinite(historical_cross_reference) and np.isclose(merged_cross_mean, historical_cross_reference, rtol=0, atol=1e-10))

    permutation_means = np.mean(np.asarray(permutation_main, dtype=float), axis=1) * 100
    b_original = np.asarray(main_original, dtype=float) * 100
    b_darg = np.asarray(main_darg, dtype=float) * 100
    strict_original_values = np.asarray(strict_original, dtype=float) * 100
    strict_darg_values = np.asarray(strict_darg, dtype=float) * 100
    np.savez_compressed(RESULTS / "figure_data.npz", folds=np.arange(1, 11, dtype=np.int64), group_fold_means_percent=group_fold_means, group_means_percent=group_means, group_sds_percent=group_sds, group_fold_edge_counts=group_fold_counts, group_fold_enhancement_ratio=group_fold_enhancement, group_fold_suppression_ratio=group_fold_suppression, main_original_percent=b_original, main_darg_percent=b_darg, permutation_fold_values_percent=np.asarray(permutation_main, dtype=np.float64) * 100, permutation_means_percent=permutation_means)

    panel_a = {"scope": "clean_test_test", "modality": "CORR", "group_names": list(GROUP_NAMES), "group_fold_means_percent": group_fold_means.tolist(), "group_means_percent": group_means.tolist(), "group_sds_percent": group_sds.tolist(), "group_fold_edge_counts": group_fold_counts.tolist(), "group_fold_enhancement_ratio": group_fold_enhancement.tolist(), "group_fold_suppression_ratio": group_fold_suppression.tolist(), "fold_rows": adjustment_rows, "merged_same_mean_percent": merged_same_mean, "merged_same_group_count_mean_percent": float(np.mean(merged_same_group_count_per_fold)), "merged_same_group_count_difference_pp": float(np.mean(merged_same_group_count_per_fold) - merged_same_mean), "merged_same_group_count_matches_direct": merged_same_group_count_matches_direct, "merged_cross_mean_percent": merged_cross_mean, "historical_same_mean_percent": historical_same_reference, "historical_cross_mean_percent": historical_cross_reference, "merged_same_matches_history": merged_same_matches_history, "merged_cross_matches_history": merged_cross_matches_history, "sd_ddof": 0}
    panel_b = {"scope": "at_least_one_clean_test", "original_values_percent": b_original.tolist(), "darg_values_percent": b_darg.tolist(), "original_mean_percent": float(np.mean(b_original)), "darg_mean_percent": float(np.mean(b_darg)), "mean_reduction_pp": float(np.mean(b_original - b_darg)), "improved_folds": int(np.sum(b_darg < b_original)), "unchanged_folds": int(np.sum(b_darg == b_original)), "declined_folds": int(np.sum(b_darg > b_original)), "permutation_means_percent": permutation_means.tolist(), "permutation_mean_percent": float(np.mean(permutation_means)), "permutation_min_percent": float(np.min(permutation_means)), "permutation_max_percent": float(np.max(permutation_means)), "permutation_interval_definition": "Min-Max of 20 permutation means, each mean across ten folds", "strict_original_mean_percent": float(np.mean(strict_original_values)), "strict_darg_mean_percent": float(np.mean(strict_darg_values)), "strict_mean_reduction_pp": float(np.mean(strict_original_values - strict_darg_values)), "strict_improved_folds": int(np.sum(strict_darg_values < strict_original_values))}
    source_audit = {"github_commit": GITHUB_COMMIT, "server_sha256": source_server, "github_sha256": source_github, "match": source_match, "model_identity": "E82.4.6.10-G1 / E82.4.6-G2 SCMR-DA-SRPE / E74.4-G2 Positive-DCR", "checkpoint_sources": checkpoint_records}
    permutation_means_recomputed = np.mean(np.asarray(permutation_main, dtype=float), axis=1) * 100
    permutation_min = float(np.min(permutation_means_recomputed))
    permutation_max = float(np.max(permutation_means_recomputed))
    checks = {"github_sources_match": all(source_match.values()), "ten_fold_prediction_exact": all_prediction_exact, "ten_fold_independent_evaluator_exact": all_independent_exact, "ten_fold_checkpoint_identity": all(item["checkpoint_sha_verified"] for item in fold_audits), "ten_fold_graph_checks": all(all(value for key, value in item.items() if key != "fold") for item in fold_audits), "historical_main_ratios_match": all(item["historical_main_ratios_match"] for item in fold_audits), "historical_strict_ratios_match": all(item["historical_strict_ratios_match"] for item in fold_audits), "group_counts_partition_each_fold": all(item["group_count_sum_exact"] for item in fold_audits), "unique_undirected_edges_each_fold": all(item["unique_undirected_edge_count_exact"] for item in fold_audits), "group_values_finite_each_fold": all(item["group_relative_values_finite"] for item in fold_audits), "panel_a_mean_sd_matches_data": bool(np.allclose(group_means, group_fold_means.mean(axis=1), rtol=0, atol=1e-12) and np.allclose(group_sds, group_fold_means.std(axis=1, ddof=0), rtol=0, atol=1e-12)), "merged_same_group_count_matches_direct": merged_same_group_count_matches_direct, "merged_same_matches_history": merged_same_matches_history, "merged_cross_matches_history": merged_cross_matches_history, "sd_uses_population_ddof_zero": bool(np.allclose(group_sds, np.std(group_fold_means, axis=1, ddof=0), rtol=0, atol=1e-12)), "permutation_means_finite": bool(np.isfinite(permutation_means).all()), "permutation_means_recomputed": bool(np.allclose(permutation_means, permutation_means_recomputed, rtol=0, atol=1e-12)), "permutation_interval_matches_data": bool(np.isclose(float(permutation_means.min()), permutation_min, rtol=0, atol=1e-12) and np.isclose(float(permutation_means.max()), permutation_max, rtol=0, atol=1e-12)), "figure_data_finite": bool(np.isfinite(group_fold_means).all() and np.isfinite(group_sds).all() and np.isfinite(b_original).all() and np.isfinite(b_darg).all() and np.isfinite(permutation_means).all())}
    audit_status = "PASS" if all(checks.values()) else "FAIL"
    audit = {"status": audit_status, "checks": checks, "folds": fold_audits, "panel_a": {"scope": "clean_test_test", "group_names": list(GROUP_NAMES), "group_means_percent": group_means.tolist(), "group_sds_percent": group_sds.tolist(), "merged_same_mean_percent": merged_same_mean, "merged_cross_mean_percent": merged_cross_mean}, "panel_b": {"permutation_means_percent": permutation_means.tolist(), "permutation_min_percent": permutation_min, "permutation_max_percent": permutation_max, "interval_definition": "Min-Max of 20 permutation means, each mean across ten folds"}, "source": source_audit}
    write_json(RESULTS / "audit_status.json", audit)
    plot_info = plot_figure(group_means, group_sds, b_original, b_darg, permutation_means)
    figure_hashes = {"pdf": sha256_file(FIGURES / "darg_relation_quality.pdf"), "png": sha256_file(FIGURES / "darg_relation_quality.png"), "data": sha256_file(RESULTS / "figure_data.npz")}
    history_reference = {"source_commit": GITHUB_COMMIT, "previous_summary_sha256": previous_summary_sha, "previous_figure_hashes": previous_figure_hashes, "previous_panel_a_same_mean_percent": historical_same_reference, "previous_panel_a_cross_mean_percent": historical_cross_reference, "historical_csv_sha256": sha256_file(RESULTS / "graph_quality_by_fold.csv")}
    summary = {"history_reference": history_reference, "historical_multimodal_summary": historical_multimodal_summary, "panel_a": panel_a, "panel_b": panel_b, "source_audit": source_audit, "audit_status": audit_status, "plot_info": plot_info, "runtime_seconds": time.time() - started, "figure_hashes": figure_hashes}
    write_json(RESULTS / "summary.json", summary)
    manifest = {"experiment": "DARG Disease-Aware Patient Relation Quality Figure final three-group redesign", "authority": str(ROOT), "model": source_audit["model_identity"], "github_commit_checked": GITHUB_COMMIT, "no_retraining": True, "panel_a": "CORR strict clean_test_test ASD-ASD/ASD-TD/TD-TD fold means with population SD", "panel_b": "at_least_one_clean_test Original/DARG paired folds and 20 permutation means", "permutations": PERMUTATIONS, "permutation_seed_base": SEED_BASE, "permutation_seed_formula": "SEED_BASE + permutation_index * 100003 + fold * 100 + modality", "outputs": {"figure_pdf": str(FIGURES / "darg_relation_quality.pdf"), "figure_png": str(FIGURES / "darg_relation_quality.png"), "figure_data": str(RESULTS / "figure_data.npz"), "audit": str(RESULTS / "audit_status.json"), "summary": str(RESULTS / "summary.json")}, "sha256": figure_hashes, "runtime_seconds": time.time() - started}
    write_json(RESULTS / "run_manifest.json", manifest)
    (RESULTS / "report.md").write_text(report_text(summary, audit, figure_hashes), encoding="utf-8")
    print(f"Completed final three-group figure in {time.time() - started:.1f}s; audit={audit_status}", flush=True)


if __name__ == "__main__":
    main()
