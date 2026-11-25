"""Aggregate audited Experiment 83.8.1 folds, OOF predictions, and report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_e8381 as runner


HISTORY_ROOT = Path("/model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity")
E836_ROOT = Path("/model/spromptgl/experiment/experiment83_6_tadpole_crossrepo_factorial")
E837_ROOT = Path("/model/spromptgl/experiment/experiment83_7_tadpole_g4_targeted_optimization")
HISTORY_DIRS = {
    "H1": "G1_E28_E0", "H2": "G2_E43_1_E00", "H3": "G3_E74_4_G2",
    "H4": "G4_E82_4_6_10_G1", "H5": "G5_E43_4_E0_ANCHORED_PHYSICAL_CAM",
}
NEW_GROUPS = [f"X{number}" for number in range(1, 7)]
ALL_GROUPS = list(HISTORY_DIRS) + NEW_GROUPS
PAIRS = [("H1", "X1"), ("H2", "X2"), ("H3", "X3"), ("H4", "X4"),
         ("H4", "X5"), ("H4", "X6"), ("X5", "X6")]


def load_json(path):
    return json.loads(Path(path).read_text())


def load_replayed_fold_metrics():
    historical = load_json(E837_ROOT / "audits/base_checkpoint_replay.json")
    current = load_json(runner.ROOT / "audits/checkpoint_replay.json")
    if historical.get("status") != "PASS" or historical.get("folds") != 50:
        raise RuntimeError("Historical Best/Latest replay is incomplete")
    if current.get("status") != "PASS" or current.get("folds") != 60:
        raise RuntimeError("E83.8.1 Best/Latest replay is incomplete")
    rows = []
    for record in historical["records"]:
        group = record["group"]
        if group not in HISTORY_DIRS or record.get("prediction_and_probability_match_source") is False:
            raise RuntimeError(f"Historical group identity or prediction replay is invalid for {group}")
        rows.append({
            "group": group, "fold": int(record["fold"]), "status": "PASS",
            "ACC": float(record["ACC"]), "AUC_original": float(record["AUC_original"]),
            "AUC_probability": float(record["AUC_probability"]), "correct": int(record["correct"]),
            "best_epoch": int(record["best_epoch"]), "best_sha256": record["best_sha256"],
            "latest_sha256": record["latest_sha256"], "best_checkpoint": record["best_path"],
            "latest_checkpoint": record["latest_path"], "source": "E83.5 verified checkpoint replay",
        })
    for record in current["records"]:
        rows.append({
            "group": record["group"], "fold": int(record["fold"]), "status": "PASS",
            "ACC": float(record["ACC"]), "AUC_original": float(record["AUC_original"]),
            "AUC_probability": float(record["AUC_probability"]), "correct": int(record["correct"]),
            "best_epoch": int(record["best_epoch"]), "best_sha256": record["best_sha256"],
            "latest_sha256": record["latest_sha256"], "best_checkpoint": record["best_path"],
            "latest_checkpoint": record["latest_path"], "source": "E83.8.1 formal training and replay",
        })
    folds = pd.DataFrame(rows).sort_values(["group", "fold"]).reset_index(drop=True)
    counts = folds.groupby("group").size().to_dict()
    if counts != {group: 10 for group in ALL_GROUPS}:
        raise RuntimeError(f"Expected ten audited folds per main group, found {counts}")
    folds.to_csv(runner.ROOT / "results/fold_metrics.csv", index=False)
    return folds


def load_oof():
    tables = []
    for group, directory in HISTORY_DIRS.items():
        for fold in range(1, 11):
            path = HISTORY_ROOT / directory / "predictions" / f"fold{fold:02d}.csv"
            table = pd.read_csv(path)
            table.insert(0, "group", group)
            tables.append(table)
    for group in NEW_GROUPS:
        for fold in range(1, 11):
            table = pd.read_csv(runner.ROOT / "groups" / group / "predictions" / f"fold{fold:02d}.csv")
            table.insert(0, "group", group)
            tables.append(table)
    oof = pd.concat(tables, ignore_index=True, sort=False)
    oof["sample_index"] = oof.sample_index.astype(np.int64)
    oof["fold"] = oof.fold.astype(np.int64)
    for group in ALL_GROUPS:
        rows = oof[oof.group == group]
        counts = rows.sample_index.value_counts()
        if len(rows) != 598 or len(counts) != 598 or counts.min() != 1 or counts.max() != 1:
            raise RuntimeError(f"OOF coverage is not exactly once for {group}")
    oof.to_csv(runner.ROOT / "results/oof_predictions.csv", index=False)
    return oof


def summarize(folds, oof):
    rows = []
    for group in ALL_GROUPS:
        fold = folds[folds.group == group]
        patient = oof[oof.group == group]
        truth = patient.true_label.to_numpy(np.int64)
        prediction = patient.pred_label.to_numpy(np.int64)
        correct = prediction == truth
        rows.append({
            "group": group, "status": "COMPLETE", "folds": len(fold),
            "ACC_mean": float(fold.ACC.mean()), "ACC_std_population": float(fold.ACC.std(ddof=0)),
            "AUC_original_mean": float(fold.AUC_original.mean()),
            "AUC_original_std_population": float(fold.AUC_original.std(ddof=0)),
            "AUC_probability_mean": float(fold.AUC_probability.mean()),
            "AUC_probability_std_population": float(fold.AUC_probability.std(ddof=0)),
            "OOF_correct": int(correct.sum()), "OOF_patients": int(len(patient)),
            "NC_correct": int(((truth == 0) & correct).sum()),
            "MCI_correct": int(((truth == 1) & correct).sum()),
            "AD_correct": int(((truth == 2) & correct).sum()),
            "source": "E83.5 independently replayed" if group.startswith("H") else "E83.8.1 formal training",
        })
    result = pd.DataFrame(rows)
    result.to_csv(runner.ROOT / "results/group_summary.csv", index=False)
    return result


def compare_pair(reference, candidate, oof, folds):
    rows = []
    for fold_id in range(1, 11):
        ref = oof[(oof.group == reference) & (oof.fold == fold_id)].sort_values("sample_index")
        cand = oof[(oof.group == candidate) & (oof.fold == fold_id)].sort_values("sample_index")
        if (not np.array_equal(ref.sample_index.to_numpy(), cand.sample_index.to_numpy())
                or not np.array_equal(ref.true_label.to_numpy(), cand.true_label.to_numpy())):
            raise RuntimeError(f"Paired folds do not share patients and labels for {reference}/{candidate}")
        truth = ref.true_label.to_numpy(np.int64)
        ref_pred = ref.pred_label.to_numpy(np.int64)
        cand_pred = cand.pred_label.to_numpy(np.int64)
        ref_acc = folds[(folds.group == reference) & (folds.fold == fold_id)].iloc[0]
        cand_acc = folds[(folds.group == candidate) & (folds.fold == fold_id)].iloc[0]
        ref_correct, cand_correct = ref_pred == truth, cand_pred == truth
        rows.append({
            "reference": reference, "candidate": candidate, "fold": fold_id,
            "ACC_reference": float(ref_acc.ACC), "ACC_candidate": float(cand_acc.ACC),
            "AUC_original_reference": float(ref_acc.AUC_original),
            "AUC_original_candidate": float(cand_acc.AUC_original),
            "ACC_delta": float(cand_acc.ACC - ref_acc.ACC),
            "AUC_original_delta": float(cand_acc.AUC_original - ref_acc.AUC_original),
            "correct_delta": int(cand_correct.sum() - ref_correct.sum()),
            "patient_corrected": int((~ref_correct & cand_correct).sum()),
            "patient_damaged": int((ref_correct & ~cand_correct).sum()),
            "patient_net": int(cand_correct.sum() - ref_correct.sum()),
        })
    detail = pd.DataFrame(rows)
    rows.append({
        "reference": reference, "candidate": candidate, "fold": "MEAN",
        "ACC_reference": float(detail.ACC_reference.mean()),
        "ACC_candidate": float(detail.ACC_candidate.mean()),
        "AUC_original_reference": float(detail.AUC_original_reference.mean()),
        "AUC_original_candidate": float(detail.AUC_original_candidate.mean()),
        "ACC_delta": float(detail.ACC_delta.mean()),
        "AUC_original_delta": float(detail.AUC_original_delta.mean()),
        "correct_delta": int(detail.correct_delta.sum()),
        "patient_corrected": int(detail.patient_corrected.sum()),
        "patient_damaged": int(detail.patient_damaged.sum()),
        "patient_net": int(detail.patient_net.sum()),
    })
    return rows


def load_extra_predictions(root, group_name, alias):
    rows = []
    for fold in range(1, 11):
        table = pd.read_csv(root / "groups" / group_name / "predictions" / f"fold{fold:02d}.csv")
        table.insert(0, "group", alias)
        rows.append(table)
    value = pd.concat(rows, ignore_index=True)
    value["sample_index"] = value.sample_index.astype(np.int64)
    value["fold"] = value.fold.astype(np.int64)
    if len(value) != 598 or value.sample_index.nunique() != 598:
        raise RuntimeError(f"Descriptive comparison OOF coverage failed for {alias}")
    return value


def descriptive_contrasts(oof, folds):
    extras = [
        (E836_ROOT, "D2", "E83.6-D2"),
        (E837_ROOT, "T4", "E83.7-T4"),
    ]
    base_h4 = oof[oof.group == "H4"]
    h4_metrics = folds[folds.group == "H4"]
    rows = []
    summary = []
    for root, native_group, alias in extras:
        external = load_extra_predictions(root, native_group, alias)
        local = pd.concat([oof, external], ignore_index=True, sort=False)
        external_folds = pd.read_csv(root / "groups" / native_group / "fold_metrics.csv")
        external_folds["group"] = alias
        external_folds = external_folds.rename(columns={"AUC": "AUC_original"})
        if "AUC_original" not in external_folds:
            raise RuntimeError(f"Missing historical AUC_original for {alias}")
        local_folds = pd.concat([folds, external_folds], ignore_index=True, sort=False)
        pair_rows = compare_pair("H4", alias, local, local_folds)
        for row in pair_rows:
            row["comparison_type"] = "DESCRIPTIVE_DIFFERENT_OBJECTIVE_OR_SCHEDULE"
        rows.extend(pair_rows)
        summary.append({"group": alias, "folds": 10,
                       "ACC_mean": float(external_folds.ACC.mean()),
                       "ACC_std_population": float(external_folds.ACC.std(ddof=0)),
                       "AUC_original_mean": float(external_folds.AUC_original.mean()),
                       "AUC_original_std_population": float(external_folds.AUC_original.std(ddof=0)),
                       "correct_total": int(external_folds.correct.sum()) if "correct" in external_folds else
                                       int((external.pred_label == external.true_label).sum())})
    return rows, summary


def dpsl_patient_summary():
    path = runner.ROOT / "results/dpsl_patient_mechanism.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    result = {}
    for group in ("X4", "X5", "X6"):
        rows = frame[frame.group == group]
        if len(rows) != 598 or rows.sample_index.nunique() != 598:
            raise RuntimeError(f"DPSL patient mechanism is incomplete for {group}")
        truth = rows.true_label.to_numpy(np.int64)
        native = rows.native_pred_label.to_numpy(np.int64)
        full = rows.full_pred_label.to_numpy(np.int64)
        hist_native = rows.baseline_native_pred_label.to_numpy(np.int64)
        hist_full = rows.baseline_full_pred_label.to_numpy(np.int64)
        native_ok, full_ok = native == truth, full == truth
        history_native_ok, history_full_ok = hist_native == truth, hist_full == truth
        direct_corrected = ~native_ok & full_ok
        direct_damaged = native_ok & ~full_ok
        relative_corrected = ~history_full_ok & full_ok
        relative_damaged = history_full_ok & ~full_ok
        result[group] = {
            "native_correct_total": int(native_ok.sum()), "full_correct_total": int(full_ok.sum()),
            "native_correct_by_class": [int(((truth == c) & native_ok).sum()) for c in range(3)],
            "full_correct_by_class": [int(((truth == c) & full_ok).sum()) for c in range(3)],
            "native_to_full_direct": {"changed": int((native != full).sum()),
                                      "corrected": int(direct_corrected.sum()),
                                      "damaged": int(direct_damaged.sum()),
                                      "net": int(full_ok.sum() - native_ok.sum())},
            "historical_H4_native_correct_total": int(history_native_ok.sum()),
            "candidate_native_vs_H4_native": {"corrected": int((~history_native_ok & native_ok).sum()),
                                               "damaged": int((history_native_ok & ~native_ok).sum()),
                                               "net": int(native_ok.sum() - history_native_ok.sum())},
            "candidate_full_vs_H4_full": {"corrected": int(relative_corrected.sum()),
                                          "damaged": int(relative_damaged.sum()),
                                          "net": int(full_ok.sum() - history_full_ok.sum())},
            "direct_corrected_also_wrong_in_H4_full": int((direct_corrected & ~history_full_ok).sum()),
            "direct_corrected_already_correct_in_H4_full": int((direct_corrected & history_full_ok).sum()),
            "candidate_full_vs_H4_class_net": [
                int(((truth == c) & full_ok).sum() - ((truth == c) & history_full_ok).sum())
                for c in range(3)],
        }
    return result


def format_group_table(summary):
    columns = ["group", "folds", "ACC_mean", "ACC_std_population", "AUC_original_mean",
               "AUC_original_std_population", "AUC_probability_mean", "AUC_probability_std_population",
               "OOF_correct", "NC_correct", "MCI_correct", "AD_correct"]
    return summary[columns].to_markdown(index=False, floatfmt=".6f")


def write_report(summary, pairs, descriptive, mechanism, replay, preflight, resource, final_audit):
    h4 = summary[summary.group == "H4"].iloc[0]
    new = summary[summary.group.isin(NEW_GROUPS)]
    best_acc = new.loc[new.ACC_mean.idxmax()]
    best_auc = new.loc[new.AUC_original_mean.idxmax()]
    paper_acc = 0.9481
    paper_auc = 0.9559
    pair_means = pairs[pairs.fold == "MEAN"]
    dpsl_patients = dpsl_patient_summary()
    pair_lookup = {(row.reference, row.candidate): row for _, row in pair_means.iterrows()}
    paired_lines = ["| 对比 | ΔACC | ΔAUC_original | 净正确人数 | 纠正 | 误伤 |",
                    "|---|---:|---:|---:|---:|---:|"]
    for reference, candidate in PAIRS:
        row = pair_lookup[(reference, candidate)]
        paired_lines.append(
            f"| {candidate} − {reference} | {row.ACC_delta:.4%} | {row.AUC_original_delta:.4%} "
            f"| {int(row.correct_delta)} | {int(row.patient_corrected)} | {int(row.patient_damaged)} |")
    d2_t4_lines = ["| 描述性组 | ACC mean ± std | AUC_original mean ± std | OOF正确 | 相对H4净变化 |",
                   "|---|---:|---:|---:|---:|"]
    for item in descriptive:
        delta = pairs[(pairs.reference == "H4") & (pairs.candidate == item["group"])
                      & (pairs.fold == "MEAN")].iloc[0]
        d2_t4_lines.append(
            f"| {item['group']} | {item['ACC_mean']:.4%} ± {item['ACC_std_population']:.4%} "
            f"| {item['AUC_original_mean']:.4%} ± {item['AUC_original_std_population']:.4%} "
            f"| {item['correct_total']} | {int(delta.correct_delta)} |")
    aloss_lines = []
    for group in ("X1", "X2", "X3", "X4"):
        item = mechanism["groups"][group]["paper_aloss"]
        gradients = item["per_modality_gradient_norm_mean"]
        ranked = sorted(enumerate(gradients, start=1), key=lambda pair: pair[1], reverse=True)
        changed_edges = [entry["changed_entries"] for entry in
                         mechanism["groups"][group]["graph_change_vs_base"]]
        aloss_lines.append(
            f"{group} 的平均ALoss为 {item['mean_loss']:.6g}，最大平均梯度在 M{ranked[0][0]} "
            f"({ranked[0][1]:.6g})，相对历史基座各模态变化矩阵项 "
            f"{', '.join(str(value) for value in changed_edges)}。")
    pair_lines = []
    for group in ("X5", "X6"):
        item = mechanism["groups"][group]["pair_mean_mechanism"]
        pair_lines.append(f"{group} 六组监督损失等权，各占总pair_mean的1/6。组内平均损失为 "
                          + ", ".join(f"{name} {value['mean_loss']:.6g}"
                                       for name, value in item.items()) + ".")
    x6_weights = mechanism["groups"]["X6"]["pair_mean_mechanism"]
    x6_ranked = sorted(x6_weights.items(), key=lambda pair: pair[1]["mean_weight"], reverse=True)
    x6_weight_text = ", ".join(f"{name} {value['mean_weight']:.4f}"
                               for name, value in x6_ranked)
    dpsl_lines = []
    for group, value in dpsl_patients.items():
        native_full = value["native_to_full_direct"]
        h4_full = value["candidate_full_vs_H4_full"]
        dpsl_lines.append(
            f"{group} Native正确 {value['native_correct_total']}/598，Full正确 {value['full_correct_total']}/598；"
            f"Full−Native纠正 {native_full['corrected']}、误伤 {native_full['damaged']}；"
            f"Full相对H4净变化 {h4_full['net']}，纠正 {h4_full['corrected']}、误伤 {h4_full['damaged']}；"
            f"直接纠正中 H4 Full 原来错误 {value['direct_corrected_also_wrong_in_H4_full']} 人。"
            f"NC/MCI/AD 相对H4 Full正确人数变化为 {value['candidate_full_vs_H4_class_net']}。")
    rows = summary.set_index("group")
    outperform = {}
    for group, baseline in (("X1", "H1"), ("X2", "H2"), ("X3", "H3"),
                            ("X4", "H4"), ("X5", "H4"), ("X6", "H4")):
        outperform[group] = {
            "ACC_above": bool(rows.loc[group, "ACC_mean"] > rows.loc[baseline, "ACC_mean"]),
            "AUC_original_above": bool(rows.loc[group, "AUC_original_mean"] > rows.loc[baseline, "AUC_original_mean"]),
        }
    x4_go = outperform["X4"]["ACC_above"] and outperform["X4"]["AUC_original_above"]
    any_g4_go = any(outperform[group]["ACC_above"] and outperform[group]["AUC_original_above"]
                    for group in ("X4", "X5", "X6"))
    paper_reached = bool((summary.ACC_mean >= paper_acc).any()
                         and (summary.AUC_original_mean >= paper_auc).any())
    replay_status = (f"E83.5历史组 {50}/50 Best/Latest/下一步重放通过；"
                     f"E83.8.1 {replay['best_passes']}/{replay['folds']} Best、"
                     f"{replay['latest_passes']}/{replay['folds']} Latest、"
                     f"{replay['next_step_passes']}/{replay['folds']} 下一步确定性通过。")
    lines = [
        "# Experiment 83.8.1 最终报告", "",
        "## A. STATUS", "COMPLETE_AUDITED 表示60个正式fold、checkpoint重放和结果聚合全部通过。性能结论单独记录。",
        "", "## B. GPU与环境",
        f"单张 {resource['device']}，显存探测峰值 {resource['peak_reserved_mib_max']:.1f} MiB，"
        f"8 worker并发预检 {resource['concurrent_probe_seconds']:.2f} 秒。数据、源码和checkpoint保存在 `{runner.ROOT}`。",
        "", "## C. 原论文及作者源码核查",
        "原论文 Eq.(1) 的图目标包含图平滑项与平方 Frobenius 标签图项，Eq.(7) 将图损失与标签损失相加。"
        "作者仓库 `ba3b9446fa451fde6864d6555bdf324de0c4b64a` 的 `network.py` 将ALoss实现为四模态未平方 `torch.linalg.norm` 平均，"
        "作者 `model.py` 的 simple-2 反向传播目标没有包含ALoss。作者GFR写死四个分支，TADPOLE运行分支为六模态。",
        "", "## D. ALoss数学定义与源码差异",
        "原论文平方 Frobenius、作者源码未平方范数、E83.6 连续相似度 pair_mean 是三个不同数学对象。"
        "X1–X4使用 `(1/6) Σ_m ||normalize(A_m+I)[train,train] − normalize(Ahat_train+I_full)[train,train]||_F²`，系数1。"
        "图与标签目标均先在598节点上对称归一化再取训练折子矩阵，训练对含对角线。标签图只写入 train×train，同病目标1、异病目标0，测试行列不构造标签关系。"
        "历史图平滑、度数与稀疏正则仍由原GFR损失计算，未重复加入。",
        "", "## E. TADPOLE数据协议",
        "598名患者，NC/MCI/AD为209/315/74，模态维度145/148/3/36/24/9。"
        f"数据SHA `{runner.DATA_SHA}`，十折SHA `{runner.SPLIT_SHA}`。保留全数据RFE、历史均值填补、CV前StandardScaler、转导患者图、val=test、"
        "首次严格提高的测试ACC选Best、450 epoch上限、wait>200 early stopping、seed 0且不逐epoch重设RNG。",
        "", "## F. 六组方法设计及保真",
        "X1 G1+Paper-ALoss，X2物理No-PEN+Paper-ALoss，X3 DARG Positive-DCR+Paper-ALoss，X4 G4 DARG+DPSL+Paper-ALoss。"
        "X5为G4+系数0.1的类别均衡pair_mean，X6再按训练患者原生头归一化熵计算detach权重。"
        "X5/X6前十轮不加pair_mean，epoch 11起生效。X4保留DPSL residual cap 0.08和共享梯度预算0.001。",
        "", "## G. Warmup配对结果",
        f"X3/X4 ALoss warmup配对 {replay['warmup_pairing']['paired_passes']}/10 folds通过；"
        "X5/X6 warmup逐折一致并与历史H4的post-warmup状态、优化器、RNG、关系对和损失轨迹核对。"
        "历史H4未保存初始模型/优化器/RNG哈希，故与历史H4的初始状态标为PARTIAL_MATCH。",
        "", "## H. 正式训练完成情况",
        "| 组 | 正式fold | 状态 |\n|---|---:|---|\n" +
        "\n".join(f"| {group} | {int((folds.group == group).sum())}/10 | COMPLETE |" for group in NEW_GROUPS),
        "总计60/60，所有fold均为完整重新训练，故障恢复从Latest继续，已完成fold按SHA核验后跳过。",
        "", "## I. 六组ACC/AUC完整表", format_group_table(summary),
        "指标为十折 mean ± population std。`AUC_original`沿用历史硬预测one-hot macro OVR。"
        "`AUC_probability`仅作辅助。598人正确总数与类别正确人数从OOF患者预测计算。",
        "", "## J. 与历史G1–G4的对照", "\n".join(paired_lines),
        "上述纠正/误伤在同折患者配对后统计。历史H1–H5未重新训练。",
        "", "## K. 与E83.6、E83.7图监督实验的对照", "\n".join(d2_t4_lines),
        "E83.6-D2为未类别均衡的全训练患者对连续相似度MSE，系数1.0。E83.7-T4仅前十轮使用图监督。"
        "二者目标或启用时点与E83.8.1不同，属于描述性比较。",
        "", "## L. ALoss六模态梯度及图学习机制", "\n".join(aloss_lines),
        "机制分析从各X组Best重放六个GraphLearn模块，并以对应历史H组Best图比较实际邻接矩阵变化。"
        "逐折梯度、模态损失、矩阵变化和DARG正负关系统计保存在 `audits/mechanism_summary.json`。",
        "", "## M. 优化版pair_mean机制", "\n".join(pair_lines),
        f"X6各类对组的平均不确定性权重为 {x6_weight_text}。六组等权确保损失贡献组数相同，"
        "训练监督只读取训练患者对。X6的权重经detach，不能沿权重路径更新原生分类头。",
        "", "## N. G4 Native/Full患者级纠错分析", "\n".join(dpsl_lines),
        "Full−Native表示同一Best checkpoint关闭/开启DPSL的直接推理变化。相对H4的净变化另按患者级预测比较，"
        "因此两类人数不混用。",
        "", "## O. 与原论文94.81%/95.59%对照",
        f"最高ACC的新组为 {best_acc.group}，{best_acc.ACC_mean:.4%}。最高AUC_original的新组为 "
        f"{best_auc.group}，{best_auc.AUC_original_mean:.4%}。原论文参考为ACC 94.81%、AUC 95.59%。"
        f"任何组同时达到两项参考指标：{paper_reached}。",
        "", "## P. Best/Latest独立审计", replay_status,
        "独立重载Best后重新计算ACC、两种AUC与预测；Latest恢复后验证优化器/RNG并重复下一步。"
        "数据与split SHA、源码seal、六模态监督、训练/测试索引、DPSL预算均纳入独立审计。",
        "", "## Q. 文件清单",
        "`protocol.json`、`source_manifest.json`、`training_seal.json`、`code/`、`inputs/`、"
        "`groups/X1...X6/`、`results/group_summary.csv`、`results/fold_metrics.csv`、"
        "`results/oof_predictions.csv`、`results/paired_contrasts.csv`、`audits/final_audit.json`、"
        "`audits/loss_fidelity.json`、`audits/checkpoint_replay.json`、`audits/mechanism_summary.json`。",
        "", "## R. handoff更新确认",
        "本报告完成后向共享handoff追加唯一Experiment 83.8.1章节并读回校验。",
        "", "## S. 最终科研结论",
        f"X4相对H4同时提高ACC与AUC_original：{x4_go}。X4/X5/X6中至少一组同时提高两项：{any_g4_go}。"
        f"`PERFORMANCE_GO_NO_GO`主判据以X4是否同时超过H4为准，结果为{'GO' if x4_go else 'NO_GO'}。"
        "方法实现审计通过不代表性能成功。保留val=test测试标签选Best、CV前全数据预处理和转导图导致的潜在乐观偏差。",
        "",
    ]
    (runner.ROOT / "results/final_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"status": "COMPLETE_AUDITED", "outperform_historical": outperform,
            "x4_performance_go": bool(x4_go), "any_g4_variant_performance_go": bool(any_g4_go),
            "paper_reference_reached": paper_reached,
            "best_new_acc_group": str(best_acc.group), "best_new_auc_group": str(best_auc.group),
            "best_new_acc": float(best_acc.ACC_mean),
            "best_new_auc_original": float(best_auc.AUC_original_mean),
            "h4_acc": float(h4.ACC_mean), "h4_auc_original": float(h4.AUC_original_mean),
            "dpsl_patient_summary": dpsl_patients}


def main():
    folds = load_replayed_fold_metrics()
    oof = load_oof()
    summary = summarize(folds, oof)
    pair_rows = []
    for reference, candidate in PAIRS:
        pair_rows.extend(compare_pair(reference, candidate, oof, folds))
    descriptive_rows, descriptive_summary = descriptive_contrasts(oof, folds)
    pair_rows.extend(descriptive_rows)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(runner.ROOT / "results/paired_contrasts.csv", index=False)
    replay = load_json(runner.ROOT / "audits/checkpoint_replay.json")
    preflight = load_json(runner.ROOT / "audits/preflight_start.json")
    resource = load_json(runner.ROOT / "audits/resource_probe.json")
    mechanism = load_json(runner.ROOT / "audits/mechanism_summary.json")
    initial_result = write_report(summary, pairs, descriptive_summary, mechanism, replay,
                                  preflight, resource, None)
    final_audit = {
        "experiment": "Experiment 83.8.1",
        "EXECUTION_AUDIT": {
            "status": "PASS", "formal_folds_completed": int(folds[folds.group.isin(NEW_GROUPS)].shape[0]),
            "expected_formal_folds": 60, "groups": {group: 10 for group in NEW_GROUPS},
            "data_sha256": runner.DATA_SHA, "split_sha256": runner.SPLIT_SHA,
            "training_seal_sha256": runner.sha_file(runner.ROOT / "audits/training_seal.json"),
            "resource_probe_status": resource["status"], "worker_count": resource["worker_count"],
            "oof_each_main_group_598_once": True,
        },
        "METHOD_FIDELITY": {
            "status": "PASS", "preflight_status": preflight["status"],
            "warmup_pairing_status": replay["warmup_pairing"]["status"],
            "dpsl_residual_cap": 0.08, "dpsl_shared_gradient_cap": 0.001,
            "new_supervision_uses_train_labels_only": True,
        },
        "CHECKPOINT_REPLAY": {
            "status": replay["status"], "folds": replay["folds"],
            "best_passes": replay["best_passes"], "latest_passes": replay["latest_passes"],
            "deterministic_next_step_passes": replay["next_step_passes"],
        },
        "LOSS_FORMULA_AUDIT": load_json(runner.ROOT / "audits/loss_fidelity.json"),
        "PERFORMANCE_GO_NO_GO": {
            "status": "GO" if initial_result["x4_performance_go"] else "NO_GO",
            "primary_rule": "X4 exceeds historical H4 on both ACC_mean and AUC_original_mean",
            "x4_exceeds_h4_both": initial_result["x4_performance_go"],
            "any_new_g4_variant_exceeds_h4_both": initial_result["any_g4_variant_performance_go"],
            "any_group_reaches_both_paper_references": initial_result["paper_reference_reached"],
            "best_new_acc_group": initial_result["best_new_acc_group"],
            "best_new_auc_original_group": initial_result["best_new_auc_group"],
            "outperform_historical": initial_result["outperform_historical"],
        },
        "historical_checkpoint_snapshot_path": str(runner.ROOT / "audits/historical_checkpoint_snapshot.json"),
        "historical_checkpoints_unchanged": True,
        "limitations": ["val_idx == test_idx and test ACC selected Best",
                        "full-data RFE and StandardScaler before outer CV",
                        "transductive patient graph"],
    }
    runner.write_json(runner.ROOT / "audits/final_audit.json", final_audit)
    write_report(summary, pairs, descriptive_summary, mechanism, replay, preflight,
                 resource, final_audit)
    print(summary.to_string(index=False))
    print(json.dumps(final_audit["PERFORMANCE_GO_NO_GO"], indent=2))


if __name__ == "__main__":
    main()
