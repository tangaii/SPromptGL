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
GITHUB_COMMIT = "9b0bc6b"
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


def relative_adjustment_stats(original, adjacency, labels, clean_test):
    valid, cross = valid_edges(original, labels, clean_test)
    relative = (adjacency - original) / original.clamp_min(torch.finfo(original.dtype).tiny)
    same_values = relative[valid & ~cross]
    cross_values = relative[cross]
    asd_asd = valid & ~cross & (labels[:, None] == 0) & (labels[None, :] == 0)
    td_td = valid & ~cross & (labels[:, None] == 1) & (labels[None, :] == 1)
    all_values = relative[valid]
    return {
        "same_edge_count": int(same_values.numel()),
        "cross_edge_count": int(cross_values.numel()),
        "asd_asd_edge_count": int(asd_asd.sum().item()),
        "td_td_edge_count": int(td_td.sum().item()),
        "same_relative_mean_percent": float(same_values.mean().item() * 100) if same_values.numel() else float("nan"),
        "cross_relative_mean_percent": float(cross_values.mean().item() * 100) if cross_values.numel() else float("nan"),
        "same_enhancement_ratio": float((same_values > 0).float().mean().item()) if same_values.numel() else float("nan"),
        "cross_suppression_ratio": float((cross_values < 0).float().mean().item()) if cross_values.numel() else float("nan"),
        "same_suppression_ratio": float((same_values < 0).float().mean().item()) if same_values.numel() else float("nan"),
        "cross_enhancement_ratio": float((cross_values > 0).float().mean().item()) if cross_values.numel() else float("nan"),
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
                summary[modality][scope][condition] = {
                    "mean_percent": float(np.mean(values)),
                    "sd_percent": float(np.std(values)),
                    "values_percent": values,
                }
    return summary


def find_legacy_modality_summary(previous_summary):
    current = previous_summary
    for _ in range(8):
        if isinstance(current, dict) and isinstance(current.get("summary"), dict) and any(modality in current["summary"] for modality in (*MODALITIES, "FUSED")):
            return current["summary"]
        if not isinstance(current, dict) or not isinstance(current.get("legacy_summary"), dict):
            break
        current = current["legacy_summary"]
    return {}


def plot_figure(adjustment_rows, b_original, b_darg, permutation_means, heatmap, heatmap_labels, representative_fold, display_info):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Patch

    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 5.55), facecolor="white")
    outer = fig.add_gridspec(1, 2, width_ratios=(1.32, 1.0), wspace=0.31, left=0.085, right=0.985, top=0.865, bottom=0.13)
    left = outer[0].subgridspec(2, 1, height_ratios=(1.02, 1.0), hspace=0.46)
    ax_a1 = fig.add_subplot(left[0])
    heatmap_grid = left[1].subgridspec(1, 2, width_ratios=(1.0, 0.045), wspace=0.035)
    ax_h = fig.add_subplot(heatmap_grid[0])
    cax = fig.add_subplot(heatmap_grid[1])
    ax_b = fig.add_subplot(outer[1])
    fig.text(0.085, 0.965, "(a)", fontsize=9, fontweight="bold", va="top")
    fig.text(0.126, 0.965, "Disease-Aware Relation Reweighting", fontsize=8.2, va="top")
    fig.text(0.565, 0.965, "(b)", fontsize=9, fontweight="bold", va="top")
    fig.text(0.606, 0.965, "Cross-Disease Propagation", fontsize=8.2, va="top")
    fig.text(0.085, 0.922, "Ten-fold relative adjustment on strict clean test-test edges", fontsize=6.2, color="#444444", va="top")
    fig.text(0.565, 0.922, f"Mean reduction: {float(np.mean(b_original - b_darg)):.2f} pp", fontsize=6.2, color="#333333", va="top")
    fig.text(0.565, 0.897, "Orange range: min-max of 20 permutation means", fontsize=5.8, color="#555555", va="top")

    same = np.asarray([row["same_relative_mean_percent"] for row in adjustment_rows], dtype=float)
    cross = np.asarray([row["cross_relative_mean_percent"] for row in adjustment_rows], dtype=float)
    jitter = np.linspace(-0.075, 0.075, len(same))
    for index in range(len(same)):
        ax_a1.plot([jitter[index], 1 + jitter[index]], [same[index], cross[index]], color="#bdbdbd", linewidth=0.65, zorder=1)
    box = ax_a1.boxplot([same, cross], positions=[0, 1], widths=0.42, patch_artist=True, showfliers=False, medianprops={"color": "#222222", "linewidth": 1.0}, whiskerprops={"color": "#555555", "linewidth": 0.7}, capprops={"color": "#555555", "linewidth": 0.7})
    for patch, color in zip(box["boxes"], ("#c23b3b", "#2c6db2")):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.9)
    ax_a1.scatter(jitter, same, color="#c23b3b", edgecolor="white", linewidth=0.35, s=22, zorder=3)
    ax_a1.scatter(1 + jitter, cross, color="#2c6db2", edgecolor="white", linewidth=0.35, s=22, zorder=3)
    ax_a1.axhline(0, color="#333333", linewidth=0.75, zorder=0)
    ax_a1.set_xticks([0, 1], ["Same-Disease", "Cross-Disease"])
    ax_a1.set_ylabel("Relative Edge Weight Adjustment (%)")
    ax_a1.spines[["top", "right"]].set_visible(False)
    ax_a1.grid(axis="y", color="#dddddd", linewidth=0.45)
    ax_a1.set_axisbelow(True)

    cmap = LinearSegmentedColormap.from_list("darg_adjustment", [(0.0, "#2166ac"), (0.5, "#ffffff"), (1.0, "#b2182b")])
    cmap.set_bad("#e4e4e4")
    norm = TwoSlopeNorm(vmin=-display_info["display_limit"], vcenter=0.0, vmax=display_info["display_limit"])
    image = ax_h.imshow(np.ma.masked_invalid(heatmap), cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    n_asd = int(np.count_nonzero(heatmap_labels == 0))
    boundary = n_asd - 0.5
    ax_h.axhline(boundary, color="#202020", linewidth=0.8)
    ax_h.axvline(boundary, color="#202020", linewidth=0.8)
    ax_h.set_xticks([])
    ax_h.set_yticks([])
    ax_h.set_title(f"Fold {representative_fold:02d}: CORR test-test", fontsize=6.7, pad=22)
    ax_h.text(n_asd / 2, 1.02, "ASD", transform=ax_h.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.0)
    ax_h.text(n_asd + (len(heatmap_labels) - n_asd) / 2, 1.02, "TD", transform=ax_h.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.0)
    ax_h.text(-0.035, n_asd / 2, "ASD", transform=ax_h.get_yaxis_transform(), ha="right", va="center", fontsize=6.2)
    ax_h.text(-0.035, n_asd + (len(heatmap_labels) - n_asd) / 2, "TD", transform=ax_h.get_yaxis_transform(), ha="right", va="center", fontsize=6.2)
    legend = [Patch(facecolor="#e4e4e4", edgecolor="#888888", linewidth=.35, label="Absent edge"), Patch(facecolor="white", edgecolor="#888888", linewidth=.35, label="Unchanged"), Patch(facecolor="#b2182b", edgecolor="none", label="Enhanced"), Patch(facecolor="#2166ac", edgecolor="none", label="Suppressed")]
    ax_h.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=4, frameon=False, fontsize=5.0, handlelength=1.0, columnspacing=0.55)
    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("Relative adjustment", fontsize=5.3)

    for index in range(10):
        ax_b.plot([0, 1], [b_original[index], b_darg[index]], color="#bdbdbd", linewidth=0.65, zorder=1)
    ax_b.scatter(np.zeros(10), b_original, color="#4c78a8", edgecolor="white", linewidth=.35, s=23, zorder=3, label="Original Graph")
    ax_b.scatter(np.ones(10), b_darg, color="#1b9e77", edgecolor="white", linewidth=.35, s=23, zorder=3, label="DARG Graph")
    perm_x = 2 + np.linspace(-0.10, 0.10, len(permutation_means))
    ax_b.scatter(perm_x, permutation_means, color="#e07a00", alpha=.58, edgecolor="white", linewidth=.3, s=20, zorder=2, label="Score-Permuted DARG")
    pmean = float(np.mean(permutation_means))
    pmin = float(np.min(permutation_means))
    pmax = float(np.max(permutation_means))
    ax_b.errorbar(2, pmean, yerr=[[pmean - pmin], [pmax - pmean]], fmt="D", color="#c55f00", markersize=4.2, capsize=2.4, linewidth=.9, zorder=4, label="Permutation mean")
    reduction = float(np.mean(b_original - b_darg))
    ax_b.set_xticks([0, 1, 2], ["Original", "DARG", "Permuted"])
    ax_b.set_ylabel("Cross-Disease Propagation Weight Ratio (%)")
    ax_b.set_ylim(0, 100)
    ax_b.set_yticks(np.arange(0, 101, 20))
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.grid(axis="y", color="#dddddd", linewidth=.45)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, 0.025), frameon=False, fontsize=5.2, ncol=2, handlelength=1.2, columnspacing=0.7)

    fig.savefig(FIGURES / "darg_relation_quality.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "darg_relation_quality.png", format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return {"permutation_mean": pmean, "permutation_min": pmin, "permutation_max": pmax, "paired_mean_reduction_pp": reduction}


def report_text(summary, audit, figure_hashes):
    pa = summary["panel_a"]
    pb = summary["panel_b"]
    modalities = summary.get("historical_multimodal_summary", {})
    lines = [
        "# DARG Disease-Aware Patient Relation Quality Analysis",
        "",
        "## 1. 实验目标与源码核验",
        "",
        "本次工作只优化冻结 DReSP 机制 Figure，不重新训练、不微调、不修改正式模型、Checkpoint、DARG/DPSL 封存源码或历史训练结果。正式任务为 E82.4.6.10-G1，内部模型为 E82.4.6-G2 SCMR-DA-SRPE，DARG 为 E74.4-G2 Positive-DCR。服务器关键运行文件与 GitHub 提交 9b0bc6b 的 raw 文件逐字 SHA-256 一致。",
        "",
        "封存实现使用 GraphLearn 原始邻接 S_m，在其真实有效支持上根据 q 和 gate 计算 DeltaA_m，形成 A_m = S_m + DeltaA_m。CORR 使用 2q-1，其余模态遵守源码中的 0.45/0.55 资格范围。Positive-DCR 对 direct 增量使用 max(DeltaA_m, 0)。传播权重按照正式 normalize_adj(A_m + I) 计算，自环只用于归一化而不进入评价边集合。",
        "",
        "## 2. Panel (a) 最终设计",
        "",
        "A1 为严格 clean_test_test 范围的十折统计。每个折先在原始图真实存在的非自环边上计算 (A_DARG-A_Original)/A_Original，分别对同病边和跨病边求折内均值，再汇总十折。红色为 Same-Disease，蓝色为 Cross-Disease，灰线连接同一折，箱线图只提供分布概览，不遮挡十个实际点。",
        "",
        "A2 为客观选择的代表折辅助热力图。选择规则是 CORR 原始 clean test-test 有效边密度最接近十折中位数，距离相同取较小折号。热力图中的浅灰色表示原始图无边，白色表示存在边且相对调整为零附近，红色表示增强，蓝色表示抑制。所有缺失边均被掩码，颜色显示范围以零对称，极端值裁剪只影响颜色映射。",
        "",
        "## 3. Panel (a) 十折结果",
        "",
        "| Fold | Clean test n | Same edges | Cross edges | Same mean (%) | Cross mean (%) | Same enhanced | Cross suppressed | Original density |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pa["fold_rows"]:
        lines.append(f"| {row['fold']} | {row['clean_test_count']} | {row['same_edge_count']} | {row['cross_edge_count']} | {row['same_relative_mean_percent']:.4f} | {row['cross_relative_mean_percent']:.4f} | {row['same_enhancement_ratio']*100:.2f}% | {row['cross_suppression_ratio']*100:.2f}% | {row['original_edge_density']*100:.4f}% |")
    lines += [
        "",
        f"十折同病相对调整均值为 {pa['same_mean_percent']:.4f}%，跨病为 {pa['cross_mean_percent']:.4f}%。同病增强比例按折均值为 {pa['same_enhancement_mean_percent']:.2f}%，跨病抑制比例按折均值为 {pa['cross_suppression_mean_percent']:.2f}%。",
        f"代表折为 Fold {pa['representative_fold']:02d}。其 clean test-test 原始 CORR 有效边为 {pa['representative_edge_count']}，患者数为 {pa['representative_clean_test_count']}，边密度为 {pa['representative_density']*100:.4f}%。十折密度中位数为 {pa['density_median']*100:.4f}%，该折到中位数的绝对距离为 {pa['representative_density_distance']*100:.4f} 个百分点。",
        f"Fold {pa['representative_fold']:02d} 热图有限相对调整值范围为 [{pa['heatmap_actual_min']:.4f}, {pa['heatmap_actual_max']:.4f}]，颜色显示对称范围为 [-{pa['heatmap_display_limit']:.4f}, {pa['heatmap_display_limit']:.4f}]，被颜色裁剪的有限值为 {pa['heatmap_clipped_values']}。",
        "",
        "## 4. Panel (b) 十折结果",
        "",
        "Panel (b) 使用主评价范围 at_least_one_clean_test。Original 与 DARG 是同一正式 DReSP Checkpoint 内部的 GraphLearn 原图和 DARG 重加权图，不是单独训练的基线模型。",
        "",
        "| Fold | Original (%) | DARG (%) | DARG−Original (pp) |",
        "|---:|---:|---:|---:|",
    ]
    for fold, original, darg_value in zip(range(1, 11), pb["original_values_percent"], pb["darg_values_percent"]):
        lines.append(f"| {fold} | {original:.4f} | {darg_value:.4f} | {darg_value-original:.4f} |")
    lines += [
        "",
        f"Original 十折均值为 {pb['original_mean_percent']:.4f}%，DARG 为 {pb['darg_mean_percent']:.4f}%，配对平均下降 {pb['mean_reduction_pp']:.4f} pp，十折改善 {pb['improved_folds']}/10。严格 clean_test_test 的复算结果为 Original {pb['strict_original_mean_percent']:.4f}%、DARG {pb['strict_darg_mean_percent']:.4f}%，平均下降 {pb['strict_mean_reduction_pp']:.4f} pp。",
        f"Score-Permuted DARG 对每个折使用 20 次置乱，保持原始图、候选边、门控和 q 数值多重集合不变，只置乱 q 与患者边的对应关系。对置乱编号 p，先收集十折结果，再计算该 p 的十折均值，得到 20 个十折平均值。图中橙色散点就是这 20 个均值，误差线明确表示这 20 个均值的最小值至最大值，不是置信区间。置乱均值为 {pb['permutation_mean_percent']:.4f}%，实际范围为 [{pb['permutation_min_percent']:.4f}%, {pb['permutation_max_percent']:.4f}%]。",
        "",
        "## 5. 四模态和融合辅助汇总",
        "",
        "下表保留上一轮正式图质量结果中的四模态十折主范围汇总。FUSED 是历史辅助代理项，不作为真实多模态 GCN 融合传播图，也不进入本 Figure 的主结论。",
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
        pair = np.asarray(darg_value["values_percent"]) - np.asarray(original["values_percent"])
        improved = int(np.sum(pair < 0))
        unchanged = int(np.sum(pair == 0))
        declined = int(np.sum(pair > 0))
        permuted_mean = f"{permuted['mean_percent']:.4f}" if permuted is not None else "not stored"
        lines.append(f"| {modality} | {original['mean_percent']:.4f} | {darg_value['mean_percent']:.4f} | {permuted_mean} | {improved}/{unchanged}/{declined} |")
    lines += [
        "",
        "## 6. 审计与历史限制",
        "",
        f"独立审计状态为 {audit['status']}。逐折检查正式 Best Checkpoint 身份、归档预测和独立 evaluator、原图/DARG 图共同来源、矩阵尺寸/有限性/对称性/非负性、DeltaA=A_DARG-A_Original、Positive-DCR 正向截断、无新边、正式归一化、disease-neutral 原图恢复、标签事后评价隔离、q 多重集合和置乱支持均由实际条件判断。原始主指标与历史结果逐折一致：{audit['checks']['historical_main_ratios_match']}。",
        "",
        "分析继承历史 val=test Checkpoint 选择、CV 前 RFE/填补/标准化、transductive 患者图和 19 名训练—测试重叠事实。重叠患者只从机制评价对象中排除，没有修改历史训练集或模型状态。该 Figure 只支持传播系数层面的机制结论，不直接证明分类准确率因果提高。",
        "",
        "## 7. Figure Caption",
        "",
        "Disease-Aware Patient Relation Analysis. (a) Ten-fold relative edge-weight adjustments on the strict clean test-test support, computed within each fold for same-disease and cross-disease patient pairs before averaging across folds. Points show all folds, with paired lines linking the same fold and boxplots summarizing the fold distribution. The auxiliary heatmap shows the signed adjustment on the original CORR GraphLearn support for the objectively selected fold whose clean test-test edge density is closest to the ten-fold median; absent original edges are masked. (b) Ten-fold paired cross-disease propagation-weight ratios from the same frozen DReSP checkpoints before and after DARG reweighting, together with 20 score-permuted negative-control means. Each permutation mean is first computed across the ten folds, and the orange range denotes the minimum to maximum of these 20 permutation means. Ratios use the model's symmetric normalization of A+I with self-loops excluded from the evaluation edges.",
        "",
        "## 8. 文件与校验",
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
    heatmap_candidates = {}
    all_prediction_exact = True
    all_independent_exact = True

    for fold in range(1, 11):
        print(f"Fold {fold:02d}: restoring audited Best", flush=True)
        ctx, engine, saved, _ = runtime.restore(fold, "best")
        state = ctx["state"]
        checkpoint_records[str(fold)] = {"path": str(runtime.checkpoint_path(fold, "best")), "source": checkpoint_source(fold, "best")[0], "sha256": sha256_file(runtime.checkpoint_path(fold, "best")), "expected_sha256": checkpoint_source(fold, "best")[1], "best_epoch": int(saved["best_epoch"]), "fold": int(saved["fold"]), "seed": int(saved["seed"])}
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
        stats = relative_adjustment_stats(originals[3], adjs[3], labels, clean_test)
        original_valid, _ = valid_edges(originals[3], labels, clean_test)
        possible = len(clean_test) * (len(clean_test) - 1) / 2
        density = float(original_valid.sum().item() / possible) if possible else float("nan")
        adjustment_rows.append({"fold": fold, "clean_test_count": len(clean_test), "overlap_count_excluded": len(overlap), "original_edge_count": corr_original_strict["valid_edges"], "original_edge_density": density, **stats, "main_original_ratio": corr_original_main["ratio"], "main_darg_ratio": corr_darg_main["ratio"], "strict_original_ratio": corr_original_strict["ratio"], "strict_darg_ratio": corr_darg_strict["ratio"], "historical_main_match": historical_main_match, "historical_strict_match": historical_strict_match})
        order = np.asarray(sorted(clean_test.tolist(), key=lambda patient: (int(labels[patient].item()), int(patient))), dtype=np.int64)
        base_sub = originals[3].detach().cpu().numpy()[np.ix_(order, order)]
        darg_sub = adjs[3].detach().cpu().numpy()[np.ix_(order, order)]
        heatmap_candidates[fold] = {"original": base_sub, "darg": darg_sub, "labels": state["labels_np"][order].astype(np.int64), "global_indices": order}
        fold_audits.append({"fold": fold, "checkpoint_sha_verified": checkpoint_records[str(fold)]["sha256"] == checkpoint_records[str(fold)]["expected_sha256"], "formal_prediction_exact": prediction_exact, "independent_evaluator_exact": independent_exact, "clean_test_excludes_overlap": not bool(np.intersect1d(clean_test, overlap).size), "clean_test_size_correct": len(clean_test) == len(state["test_np"]) - len(overlap), "labels_only_posthoc": "labels" not in inspect.signature(darg.rudag_forward).parameters and "y" not in inspect.signature(darg.rudag_forward).parameters, "ratio_results_valid": ratio_results_valid, "historical_main_ratios_match": historical_main_match, "historical_strict_ratios_match": historical_strict_match, "permutation_q_multiset_preserved": all(permutation_q_checks), "permutation_support_preserved": all(permutation_support_checks), "permutation_matrices_valid": all(permutation_matrix_checks), "permutation_local_rng_unchanged": permutation_rng_unchanged, **graph_checks})
        print(f"Fold {fold:02d}: prediction={'PASS' if prediction_exact else 'FAIL'}, strict original edges={corr_original_strict['valid_edges']}", flush=True)

    density_values = np.asarray([row["original_edge_density"] for row in adjustment_rows], dtype=float)
    density_median = float(np.median(density_values))
    distances = np.abs(density_values - density_median)
    representative_index = min(range(10), key=lambda index: (distances[index], index + 1))
    representative_fold = representative_index + 1
    candidate = heatmap_candidates[representative_fold]
    relative_matrix = np.full_like(candidate["original"], np.nan, dtype=np.float32)
    support = (candidate["original"] > 0) & ~np.eye(candidate["original"].shape[0], dtype=bool)
    relative_matrix[support] = ((candidate["darg"] - candidate["original"])[support] / candidate["original"][support]).astype(np.float32)
    finite = relative_matrix[np.isfinite(relative_matrix)]
    display_limit = max(float(np.quantile(np.abs(finite), .99)), 1e-12)
    clipped_values = int(np.count_nonzero(np.abs(finite) > display_limit))
    display_matrix = relative_matrix.copy()
    display_matrix[np.isfinite(display_matrix) & np.isclose(display_matrix, 0.0, atol=1e-12, rtol=0)] = 0.0
    relative_recomputed = np.full_like(candidate["original"], np.nan, dtype=np.float32)
    relative_recomputed[support] = ((candidate["darg"] - candidate["original"])[support] / candidate["original"][support]).astype(np.float32)
    n_asd = int(np.count_nonzero(candidate["labels"] == 0))
    heatmap_mask_matches_support = bool(np.array_equal(np.isfinite(relative_matrix), support))
    heatmap_values_match = bool(np.array_equal(relative_matrix[support], relative_recomputed[support]))
    heatmap_symmetric = bool(np.allclose(candidate["original"], candidate["original"].T, rtol=0, atol=0) and np.allclose(candidate["darg"], candidate["darg"].T, rtol=0, atol=0))
    heatmap_labels_grouped = bool(n_asd > 0 and n_asd < len(candidate["labels"]) and np.all(candidate["labels"][:n_asd] == 0) and np.all(candidate["labels"][n_asd:] == 1))
    heatmap_clipping_count_exact = clipped_values == int(np.count_nonzero(np.abs(finite) > display_limit))
    permutation_means = np.asarray([np.mean(values) for values in permutation_main], dtype=float) * 100
    np.savez_compressed(RESULTS / "figure_data.npz", folds=np.arange(1, 11, dtype=np.int64), same_relative_mean_percent=np.asarray([row["same_relative_mean_percent"] for row in adjustment_rows], dtype=np.float64), cross_relative_mean_percent=np.asarray([row["cross_relative_mean_percent"] for row in adjustment_rows], dtype=np.float64), same_edge_count=np.asarray([row["same_edge_count"] for row in adjustment_rows], dtype=np.int64), cross_edge_count=np.asarray([row["cross_edge_count"] for row in adjustment_rows], dtype=np.int64), same_enhancement_ratio=np.asarray([row["same_enhancement_ratio"] for row in adjustment_rows], dtype=np.float64), cross_suppression_ratio=np.asarray([row["cross_suppression_ratio"] for row in adjustment_rows], dtype=np.float64), original_edge_density=np.asarray(density_values, dtype=np.float64), main_original_percent=np.asarray(main_original, dtype=np.float64) * 100, main_darg_percent=np.asarray(main_darg, dtype=np.float64) * 100, permutation_fold_values_percent=np.asarray(permutation_main, dtype=np.float64) * 100, permutation_means_percent=permutation_means, representative_fold=np.asarray([representative_fold], dtype=np.int64), representative_global_indices=candidate["global_indices"], representative_labels=candidate["labels"], representative_relative_adjustment=display_matrix, representative_original=candidate["original"].astype(np.float32), representative_darg=candidate["darg"].astype(np.float32))

    adjustment_mean_same = float(np.mean([row["same_relative_mean_percent"] for row in adjustment_rows]))
    adjustment_mean_cross = float(np.mean([row["cross_relative_mean_percent"] for row in adjustment_rows]))
    same_enhancement_mean = float(np.mean([row["same_enhancement_ratio"] for row in adjustment_rows])) * 100
    cross_suppression_mean = float(np.mean([row["cross_suppression_ratio"] for row in adjustment_rows])) * 100
    panel_a = {"scope": "clean_test_test", "fold_rows": adjustment_rows, "same_mean_percent": adjustment_mean_same, "cross_mean_percent": adjustment_mean_cross, "same_enhancement_mean_percent": same_enhancement_mean, "cross_suppression_mean_percent": cross_suppression_mean, "density_median": density_median, "representative_fold": representative_fold, "representative_edge_count": adjustment_rows[representative_index]["original_edge_count"], "representative_clean_test_count": adjustment_rows[representative_index]["clean_test_count"], "representative_density": density_values[representative_index], "representative_density_distance": distances[representative_index], "heatmap_actual_min": float(finite.min()), "heatmap_actual_max": float(finite.max()), "heatmap_display_limit": display_limit, "heatmap_clipped_values": clipped_values}
    b_original = np.asarray(main_original, dtype=float) * 100
    b_darg = np.asarray(main_darg, dtype=float) * 100
    strict_original_values = np.asarray(strict_original, dtype=float) * 100
    strict_darg_values = np.asarray(strict_darg, dtype=float) * 100
    panel_b = {"scope": "at_least_one_clean_test", "original_values_percent": b_original.tolist(), "darg_values_percent": b_darg.tolist(), "original_mean_percent": float(np.mean(b_original)), "darg_mean_percent": float(np.mean(b_darg)), "mean_reduction_pp": float(np.mean(b_original - b_darg)), "improved_folds": int(np.sum(b_darg < b_original)), "permutation_means_percent": permutation_means.tolist(), "permutation_mean_percent": float(np.mean(permutation_means)), "permutation_min_percent": float(np.min(permutation_means)), "permutation_max_percent": float(np.max(permutation_means)), "strict_original_mean_percent": float(np.mean(strict_original_values)), "strict_darg_mean_percent": float(np.mean(strict_darg_values)), "strict_mean_reduction_pp": float(np.mean(strict_original_values - strict_darg_values)), "strict_improved_folds": int(np.sum(strict_darg_values < strict_original_values))}
    source_audit = {"github_commit": GITHUB_COMMIT, "server_sha256": source_server, "github_sha256": source_github, "match": source_match, "model_identity": "E82.4.6.10-G1 / E82.4.6-G2 SCMR-DA-SRPE / E74.4-G2 Positive-DCR", "checkpoint_sources": checkpoint_records}
    permutation_means_recomputed = np.mean(np.asarray(permutation_main, dtype=float), axis=1) * 100
    permutation_min = float(np.min(permutation_means_recomputed))
    permutation_max = float(np.max(permutation_means_recomputed))
    checks = {"github_sources_match": all(source_match.values()), "ten_fold_prediction_exact": all_prediction_exact, "ten_fold_independent_evaluator_exact": all_independent_exact, "ten_fold_checkpoint_identity": all(item["checkpoint_sha_verified"] for item in fold_audits), "ten_fold_graph_checks": all(all(value for key, value in item.items() if key != "fold") for item in fold_audits), "historical_main_ratios_match": all(item["historical_main_ratios_match"] for item in fold_audits), "historical_strict_ratios_match": all(item["historical_strict_ratios_match"] for item in fold_audits), "representative_rule_executed": representative_fold == min(range(1, 11), key=lambda fold: (abs(density_values[fold - 1] - density_median), fold)), "heatmap_mask_matches_original_support": heatmap_mask_matches_support, "heatmap_values_match_relative_adjustment": heatmap_values_match, "heatmap_matrices_symmetric": heatmap_symmetric, "heatmap_labels_asd_then_td": heatmap_labels_grouped, "heatmap_clipping_count_exact": heatmap_clipping_count_exact, "permutation_means_finite": bool(np.isfinite(permutation_means).all()), "permutation_means_recomputed": bool(np.allclose(permutation_means, permutation_means_recomputed, rtol=0, atol=1e-12)), "permutation_interval_matches_data": bool(np.isclose(float(permutation_means.min()), permutation_min, rtol=0, atol=1e-12) and np.isclose(float(permutation_means.max()), permutation_max, rtol=0, atol=1e-12)), "figure_data_finite": bool(np.isfinite(display_matrix[np.isfinite(display_matrix)]).all())}
    audit_status = "PASS" if all(checks.values()) else "FAIL"
    audit = {"status": audit_status, "checks": checks, "folds": fold_audits, "panel_a": {"representative_fold": representative_fold, "density_values": density_values.tolist(), "density_median": density_median, "selection_distances": distances.tolist()}, "panel_b": {"permutation_means_percent": permutation_means.tolist(), "permutation_min_percent": permutation_min, "permutation_max_percent": permutation_max, "interval_definition": "min-max of 20 permutation means, each mean across ten folds"}, "source": source_audit}
    write_json(RESULTS / "audit_status.json", audit)
    plot_info = plot_figure(adjustment_rows, b_original, b_darg, permutation_means, display_matrix, candidate["labels"], representative_fold, {"display_limit": display_limit, "clipped_values": clipped_values})
    figure_hashes = {"pdf": sha256_file(FIGURES / "darg_relation_quality.pdf"), "png": sha256_file(FIGURES / "darg_relation_quality.png"), "data": sha256_file(RESULTS / "figure_data.npz")}
    summary = {"legacy_summary": previous_summary, "historical_multimodal_summary": historical_multimodal_summary, "panel_a": panel_a, "panel_b": panel_b, "source_audit": source_audit, "audit_status": audit_status, "plot_info": plot_info, "runtime_seconds": time.time() - started, "figure_hashes": figure_hashes}
    write_json(RESULTS / "summary.json", summary)
    write_json(RESULTS / "run_manifest.json", {"experiment": "DARG Disease-Aware Patient Relation Quality Analysis figure redesign", "authority": str(ROOT), "model": source_audit["model_identity"], "github_commit_checked": GITHUB_COMMIT, "no_retraining": True, "permutations": PERMUTATIONS, "permutation_seed_base": SEED_BASE, "permutation_seed_formula": "SEED_BASE + permutation_index * 100003 + fold * 100 + modality", "representative_rule": "CORR original clean_test_test edge density closest to ten-fold median, lower fold on tie", "outputs": {"figure_pdf": str(FIGURES / "darg_relation_quality.pdf"), "figure_png": str(FIGURES / "darg_relation_quality.png"), "figure_data": str(RESULTS / "figure_data.npz"), "audit": str(RESULTS / "audit_status.json"), "summary": str(RESULTS / "summary.json")}, "sha256": figure_hashes, "runtime_seconds": time.time() - started})
    (RESULTS / "report.md").write_text(report_text(summary, audit, figure_hashes), encoding="utf-8")
    print(f"Completed figure redesign in {time.time() - started:.1f}s; audit={audit_status}; representative fold={representative_fold:02d}", flush=True)


if __name__ == "__main__":
    main()
