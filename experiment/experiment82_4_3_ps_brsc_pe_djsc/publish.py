"""Build paired scientific outputs and append the verified execution handoff."""
from __future__ import annotations

import fcntl
import hashlib
import os

from common import history as h
from audit import METRICS, REFERENCES
from run import MANIFEST, RESULTS, paths

import numpy as np
from sklearn.metrics import roc_auc_score

CFG = h.CFG
DISPLAY = {"G1": "82.4.3-G1 PS-BRSC", "G2": "82.4.3-G2 PE-DJSC"}
SCOPES = ("10_fold", "exclude_fold_2")
COUNT = ("asd_corrected", "asd_harmed", "td_corrected", "td_harmed", "corrected", "harmed", "net")
ALL_METRICS = (*METRICS, "brier", "ece")


def table(columns, rows):
    return ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows), ""]


def methods(row, key):
    return row["historical"][key] if key in REFERENCES else row["interventions"][key]


def paired_fold(row, source, target):
    a, b = methods(row, source), methods(row, target)
    truth = np.array(row["label"])
    pa, pb = np.asarray(a["prediction"]), np.asarray(b["prediction"])
    fixed, harmed = (pa != truth) & (pb == truth), (pa == truth) & (pb != truth)
    probability = np.asarray(b["probability"]) - np.asarray(a["probability"])
    dm = np.asarray(b["true_margin"]) - np.asarray(a["true_margin"])
    data = {key + "_pp": (b["metrics"][key] - a["metrics"][key]) * 100 for key in ALL_METRICS}
    data.update({
        "asd_corrected": int((fixed & (truth == 0)).sum()), "asd_harmed": int((harmed & (truth == 0)).sum()),
        "td_corrected": int((fixed & (truth == 1)).sum()), "td_harmed": int((harmed & (truth == 1)).sum()),
        "corrected": int(fixed.sum()), "harmed": int(harmed.sum()), "net": int(fixed.sum() - harmed.sum()),
        "probability_changed": int((abs(probability) > CFG["measurable_probability_change"]).sum()),
        "hard_flipped": int((pa != pb).sum()), "mean_abs_probability_delta": float(abs(probability).mean()),
        "mean_true_margin_change": float(dm.mean()), "mean_abs_true_margin_change": float(abs(dm).mean()),
    })
    h.require(data["net"] == int(np.count_nonzero(pb == truth) - np.count_nonzero(pa == truth)),
              "Paired patient count identity differs")
    return {"group": row["group"], "fold": row["fold"], "source": source, "target": target,
            "n_test": len(truth), **data}


def pair_sets(group):
    all_pairs = [(baseline, "full") for baseline in REFERENCES]
    all_pairs += [("DARG", "residual_off"), ("residual_off", "full"), ("anonymous_query", "full")]
    return all_pairs


def aggregate_pairs(rows, source, target, scope):
    per_fold = [paired_fold(row, source, target) for row in rows if scope == "10_fold" or row["fold"] != 2]
    out = {"group": rows[0]["group"], "scope": scope, "fold": "", "source": source, "target": target,
           "n_folds": len(per_fold), "n_test": sum(row["n_test"] for row in per_fold)}
    for key in ALL_METRICS:
        pk = key + "_pp"
        values = np.asarray([row[pk] for row in per_fold])
        out.update({pk: float(values.mean()), pk + "_population_sd": float(values.std()),
                    key + "_wins": int((values > 1e-12).sum()), key + "_ties": int((abs(values) <= 1e-12).sum()),
                    key + "_losses": int((values < -1e-12).sum())})
    for key in (*COUNT, "probability_changed", "hard_flipped"):
        out[key] = sum(row[key] for row in per_fold)
    return out


def summaries(rows):
    by_fold = [r for r in rows if r["group"] == "G1"]
    raw, result = [], {}
    for name in (*REFERENCES, *h.GROUPS):
        batch = []
        for row in by_fold if name in REFERENCES else (r for r in rows if r["group"] == name):
            if name in REFERENCES:
                score = row["historical"][name]
                e, end = score["epoch"], ""
            else:
                score = row["interventions"]["full"]
                e, end = row["best_epoch"], row["epoch"]
            label = DISPLAY.get(name, name)
            stat = {"method": label, "fold": row["fold"], "best_epoch": e, "end_epoch": end,
                    **score["metrics"]}
            if name in h.GROUPS:
                for m in METRICS:
                    stat[m + "_delta_vs_darg_pp"] = 100 * (
                        stat[m] - row["historical"]["DARG"]["metrics"][m])
            raw.append(stat)
            batch.append(stat)
        for scope in SCOPES:
            valid = [row for row in batch if scope == "10_fold" or row["fold"] != 2]
            result[name, scope] = {m: (float(np.mean([row[m] for row in valid])),
                                       float(np.std([row[m] for row in valid], ddof=0))) for m in ALL_METRICS}
    return raw, result


def comparison_rows(rows):
    paired, aggregates = [], {}
    for group in h.GROUPS:
        group_rows = [row for row in rows if row["group"] == group]
        for source, target in pair_sets(group):
            for row in group_rows:
                paired.append({"scope": "fold", "n_folds": 1, **paired_fold(row, source, target)})
            for scope in SCOPES:
                aggregate = aggregate_pairs(group_rows, source, target, scope)
                aggregates[group, scope, source, target] = aggregate
                paired.append(aggregate)
    return paired, aggregates


def mechanism_rows(rows):
    rows_metric, patients = [], []
    for row in rows:
        fold, group = row["fold"], row["group"]
        paths_ = row["response"]
        original = row["historical"]["DARG"]
        off = row["interventions"]["residual_off"]
        full = row["interventions"]["full"]
        for key, entry in row["interventions"].items():
            actor = {"group": group, "fold": fold, "best_epoch": row["best_epoch"], "intervention": key,
                     **entry["metrics"], **entry["response_vs_full"]}
            actor.update({k + "_mean_full": float(np.mean(v)) for k, v in paths_.items()
                          if k != "intervention_residual_difference_l2"})
            rows_metric.append(actor)
            for i, (patient, label) in enumerate(zip(row["patient_rows_1based"], row["label"])):
                diagnostics = {k: v[i] for k, v in paths_.items() if k != "intervention_residual_difference_l2"}
                patients.append({
                    "group": group, "fold": fold, "patient_row_1based": patient, "label": label, "intervention": key,
                    "fold2_historical_overlap": fold == 2 and patient in CFG["fold2_overlap_1based"],
                    "pred": entry["prediction"][i], "p_asd": entry["probability"][i], "pred_darg": original["prediction"][i],
                    "pred_residual_off": off["prediction"][i], "pred_full": full["prediction"][i],
                    "true_margin": entry["true_margin"][i], "margin_darg": original["true_margin"][i],
                    "margin_residual_off": off["true_margin"][i], "margin_full": full["true_margin"][i],
                    "margin_vs_darg": entry["true_margin"][i] - original["true_margin"][i],
                    "margin_vs_residual_off": entry["true_margin"][i] - off["true_margin"][i],
                    "margin_vs_full": entry["true_margin"][i] - full["true_margin"][i],
                    "residual_difference_l2_vs_full": paths_["intervention_residual_difference_l2"][key][i],
                    **diagnostics,
                })
    return rows_metric, patients


def decisions_for(rows, aggregate, mechanisms):
    decisions = {}
    for group in h.GROUPS:
        def comparison(source, target, scope="10_fold"):
            return aggregate[group, scope, source, target]

        full = comparison("DARG", "full")
        strict9 = comparison("DARG", "full", "exclude_fold_2")
        direct = comparison("residual_off", "full")
        medical = comparison("anonymous_query", "full")
        medical_delta = [row for row in mechanisms if row["group"] == group and row["intervention"] == "anonymous_query"]
        core_names = (("correctability_gate_off", "signed_action_unsigned") if group == "G1" else
                      ("reliability_off", "evidence_direction_off", "spectral_only", "complement_only"))
        core = {name: max(row["max_feature_delta_l2"] for row in mechanisms
                          if row["group"] == group and row["intervention"] == name) for name in core_names}
        performance = {"acc": full["acc_pp"] > 0, "ba": full["hard_auc_pp"] > 0, "sen": full["sen_pp"] > 0,
                       "net": full["net"] > 0, "exclude_fold_2_acc": strict9["acc_pp"] > 0,
                       "exclude_fold_2_net": strict9["net"] > 0}
        mechanism = {"fixed_weight_net": direct["net"] > 0,
                     "fixed_weight_ba": direct["hard_auc_pp"] > 0,
                     "query_changes_cam": any(row["max_feature_delta_l2"] > 0 for row in medical_delta),
                     "query_changes_probability": medical["probability_changed"] > 0,
                     "verified_core_mechanisms_nonzero": all(v > 0 for v in core.values())}
        overall = all(performance.values()) and all(mechanism.values())
        decisions[group] = {
            "performance_go": all(performance.values()), "mechanism_go": all(mechanism.values()),
            "overall": "GO" if overall else "NO-GO", "performance": performance, "mechanism": mechanism,
            "medical_query_probability_changed": medical["probability_changed"],
            "medical_query_hard_flipped": medical["hard_flipped"],
            "medical_vs_anonymous_net": medical["net"], "medical_vs_anonymous_acc_pp": medical["acc_pp"],
            "medical_vs_anonymous_ba_pp": medical["hard_auc_pp"],
            "observed_diagnostic_gain_vs_anonymous": (medical["net"] > 0 and medical["acc_pp"] > 0 and medical["hard_auc_pp"] > 0),
            "core_feature_delta_max": core,
        }
    return decisions


def performance_table(summary, scope):
    return table(("方法", "ACC", "Hard-AUC/BA", "Prob-AUC", "SEN-ASD", "SPE-TD", "ASD-F1", "Brier", "ECE"),
                 [(DISPLAY.get(name, name), *(f"{100*a:.4f} ± {100*b:.4f}" for a, b in
                                             (summary[name, scope][k] for k in ALL_METRICS)))
                  for name in (*REFERENCES, *h.GROUPS)])


def pair_table(aggregate, scope):
    names = ("DARG", "residual_off", "E82.2-G2", "E82.3.2-G2", "E82.4.2-G2", "anonymous_query")
    data = []
    for group in h.GROUPS:
        for source, target in pair_sets(group):
            if source not in names:
                continue
            a = aggregate[group, scope, source, target]
            data.append((group, source + " → " + target,
                         *(f"{a[key + '_pp']:+.4f}" for key in METRICS),
                         "/".join(str(a["acc_" + k]) for k in ("wins", "ties", "losses")),
                         "/".join(str(a["hard_auc_" + k]) for k in ("wins", "ties", "losses")),
                         f'{a["asd_corrected"]}/{a["asd_harmed"]}', f'{a["td_corrected"]}/{a["td_harmed"]}',
                         f'{a["corrected"]}/{a["harmed"]}/{a["net"]:+d}'))
    return table(("组", "患者配对", "ΔACC", "ΔBA", "ΔProb-AUC", "ΔSEN", "ΔSPE", "ΔF1",
                  "ACC 胜/平/负", "BA 胜/平/负", "ASD 纠/伤", "TD 纠/伤", "总纠/伤/净"), data)


def dynamics(rows, group, stage):
    values = [row["ledger"][stage]["routing"] if stage in ("first", "tenth") else
              row["best_gradients"]["routing"] if stage == "Best" else row["latest_reload"]["diagnostic"]["routing"]
              for row in rows if row["group"] == group]
    keys = ("gate_mean", "feasibility_mean", "action_abs_mean", "action_positive", "residual_mean",
            "residual_active", "wrong_positive_share", "correct_negative_share")
    return [float(np.mean([r[k] for r in values])) for k in keys]


def render_report(rows, summary, aggregate, mechanisms, patients, decision, manifest):
    lines = [f"# {CFG['experiment']}", "",
             "**COMPLETE_AUDITED**。G1 和 G2 各完成 10 折，Best 和 Latest 各重载审计 20/20。", "",
             "G1 PS-BRSC 继承 E82.2-G2 的 Original O、语义和统计方向权重，用一阶可纠错性和边界保护调节有符号谱位移。G2 PE-DJSC 继承 E82.3.2-G2 的有界 Gaussian LLR 和评估后 EMA，扩展到合法五折统计后，以连续证据控制当前分类器 Jacobian 的谱内外位移。两组同时应用 Native-path 辅助分类。", "",
             "原论文使用 SPGS 和 RES 构成 PEN。历史运行链为多模态 → GFR → Positive-DCR DARG → PLNNoPEN CAM → CAM144 → 本研究的 ΔC → 原 OutputLayer → ASD/TD。两组均用原 CAM 整体。G1 将加权有符号动作经 U16 写回。G2 以 tanh(E/0.30) 给谱内外 Jacobian 分量赋共同疾病方向，两个 sigmoid 幅度独立。两组使用相对 CAM 范数 8% 的平滑 L2 上限。完整公式和实现差异见 [设计](../design.md)。[MICCAI 2025 原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)", "",
             f"训练基座 E74.4-G2，seed 0，固定原十折，从每折 E61 epoch 10 还原四类 RNG、原训练关系对和三个历史优化器。epoch 11–450，wait > 200 早停，同分保留首次 test ACC 最大。原损失 cls + graph + 0.05 pair + 0.01 adv + 0.01 preserve。原三个优化器依次为 Adam(0.0038, 0.11)、Adam(0.0038, 0.11)、AdamW(0.001, 0.01)，新 AdamW(0.0005, 0.01) 最后执行。Native CE 系数 0.10，逐 GFR/DARG/CAM/头的原主分类梯度范数上限 10%。新增校正分支辅助梯度上限依次为方向 5%、增益 5%、保护 3%、正则 2%。仅物理 GPU0 `{CFG['gpu']['uuid']}`，分两波各 10 workers。冻结 SHA `{manifest['implementation_sha']}`。", "",
             "## 十折正式指标", "",
             "表中均值 ± population SD 以百分数表示。ASD=0 为阳性。Hard-AUC 是二元硬预测的 Balanced Accuracy，Prob-AUC 是连续 p(ASD) 的 ROC-AUC。Brier、10 个等宽 p(ASD) 区间的 ECE 同样乘 100。逐折样本数、Best epoch 和混淆矩阵见 [fold_metrics.csv](fold_metrics.csv)。", "",
             *performance_table(summary, "10_fold"),
             "## 患者级配对与同权重分解", "",
             "差值为各折配对均值，单位百分点。DARG → Residual-off 包含共同适配和独立 Best epoch 的影响。Residual-off → Full 使用同一 Best 权重，隔离校正分支的即时作用。病例计数从逐患者硬预测复算。", "",
             *pair_table(aggregate, "10_fold"),
             "## 排除 Fold 2 的九折敏感性", "",
             *performance_table(summary, "exclude_fold_2"), *pair_table(aggregate, "exclude_fold_2"),
             "## 固定权重机制", "",
             "所有干预都在 CAM144 重新生成残差并经过原 OutputLayer。下表按 Full → 干预计数。可测概率变化定义为 |Δp(ASD)| > 10⁻⁶。CAM 和 logit 差异为病例级平均。", ""]
    intervention_summary = []
    for group in h.GROUPS:
        names = dict.fromkeys(r["intervention"] for r in mechanisms if r["group"] == group)
        for key in names:
            batch = [r for r in mechanisms if r["group"] == group and r["intervention"] == key]
            if key == "full":
                continue
            n = sum(r["n_test"] for r in batch)
            intervention_summary.append((
                group, key, *(sum(r[x] for r in batch) for x in ("probability_changed", "hard_flipped")),
                f'{sum(r["corrected"] for r in batch)}/{sum(r["harmed"] for r in batch)}/{sum(r["net"] for r in batch):+d}',
                *(f"{sum(r[k] * r['n_test'] for r in batch) / n:.5g}" for k in
                  ("mean_feature_delta_l2", "absolute_logit_delta", "probability_mae"))))
    lines += table(("组", "干预", "概率变化", "硬翻转", "总纠/伤/净", "CAM L2", "logit MAE", "p(ASD) MAE"),
                   intervention_summary)
    diag_table = []
    for group in h.GROUPS:
        for label, stage in (("第 1 步", "first"), ("第 10 步", "tenth"), ("Best", "Best"), ("Latest", "Latest")):
            diag_table.append((group, label, *(f"{v:.5f}" for v in dynamics(rows, group, stage))))
    lines += ["下表为合法训练行的折均值。活跃残差使用相对范数 >10⁻⁵。增益为真实类别 Margin 的变化。Best 和 Latest 诊断使用重载状态的一次训练前向。", "",
              *table(("组", "阶段", "门控均值", "可纠性", "|动作|", "正动作比例", "平均残差比", "活跃比例",
                      "原错误增益正", "原正确增益负"), diag_table)]
    for group in h.GROUPS:
        matching = [r for r in patients if r["group"] == group and r["intervention"] == "full"]
        gain = np.array([r["margin_vs_residual_off"] for r in matching])
        old_err = np.array([r["pred_residual_off"] != r["label"] for r in matching])
        fixed = [r for r in matching if r["pred_residual_off"] != r["label"] and r["pred"] == r["label"]]
        harmed = [r for r in matching if r["pred_residual_off"] == r["label"] and r["pred"] != r["label"]]
        lines.extend([f"{group} 测试行真实 Margin 变化均值 {gain.mean():+.5f}，原错行均值 {gain[old_err].mean():+.5f}，原对行均值 {gain[~old_err].mean():+.5f}。固定权重纠正 {len(fixed)} 人、误伤 {len(harmed)} 人。", ""])
        if group == "G1":
            off = [r for r in patients if r["group"] == group and r["intervention"] == "correctability_gate_off"]
            risk = sum(r["pred_residual_off"] == r["label"] and r["pred"] != r["label"] for r in off)
            fixed_open = sum(r["pred_residual_off"] != r["label"] and r["pred"] == r["label"] for r in off)
            err = roc_auc_score(old_err, [r["feasibility"] for r in matching])
            lines.extend([f"G1 可纠错性门控开启时纠正/误伤 {len(fixed)}/{len(harmed)}，去掉该因子后为 {fixed_open}/{risk}。连续可纠性区分原错行的描述性 AUROC 为 {err:.4f}。谱内位移、两个权重分支和有符号动作均已独立重建。", ""])
        else:
            rel = np.asarray([r["support"] for r in matching])
            conflict = np.asarray([r["conflict"] for r in matching])
            diag = roc_auc_score(old_err, rel * conflict)
            threshold = np.quantile(rel * conflict, 0.75)
            high = rel * conflict >= threshold
            switched = np.asarray([r["pred"] == r["label"] for r in matching])
            hi_err = old_err & high
            lo_err = old_err & ~high
            lines.extend([f"G2 可靠性×冲突分数区分 Residual-off 错误的描述性 AUROC 为 {diag:.4f}。分数最高四分位原错 {int(hi_err.sum())} 人，其中纠正 {int((hi_err & switched).sum())} 人；其余原错 {int(lo_err.sum())} 人，其中纠正 {int((lo_err & switched).sum())} 人。全体测试行平均谱内/谱外相对残差为 {np.mean([r['parallel_ratio'] for r in matching]):.6f}/{np.mean([r['complement_ratio'] for r in matching]):.6f}。", ""])
    lines += ["## 科研判定", ""]
    for group, d in decision.items():
        a = aggregate[group, "10_fold", "DARG", "full"]
        b = aggregate[group, "10_fold", "DARG", "residual_off"]
        c = aggregate[group, "10_fold", "residual_off", "full"]
        per = "GO" if d["performance_go"] else "NO-GO"
        mech = "GO" if d["mechanism_go"] else "NO-GO"
        absent = [key for key, passed in d["performance"].items() if not passed]
        spec = [key for key, passed in d["mechanism"].items() if not passed]
        advantage = ("固定权重下观察到医学 Query 对匿名 Query 的诊断增益" if d["observed_diagnostic_gain_vs_anonymous"]
                     else "固定权重下未观察到医学 Query 相对匿名 Query 的正 ACC、BA 和净纠错")
        lines.extend([f"**{group} {d['overall']}**。性能 {per}，机制 {mech}。DARG → Residual-off 净 {b['net']:+d} 人，Residual-off → Full 净 {c['net']:+d} 人，完整相对 DARG 净 {a['net']:+d} 人。医学 Query 替换改变 {d['medical_query_probability_changed']} 个概率、翻转 {d['medical_query_hard_flipped']} 个硬预测。{advantage}。医学相对匿名 ΔACC {d['medical_vs_anonymous_acc_pp']:+.4f} pp，ΔBA {d['medical_vs_anonymous_ba_pp']:+.4f} pp，净 {d['medical_vs_anonymous_net']:+d} 人。", ""])
        if absent or spec:
            lines.extend([f"未达门槛项为性能 {'、'.join(absent) or '无'}，机制 {'、'.join(spec) or '无'}。", ""])
        if a["prob_auc_pp"] > 0 and a["acc_pp"] < 0:
            lines.extend([f"{group} 的 Prob-AUC 提高 {a['prob_auc_pp']:+.4f} pp 而 ACC 下降 {a['acc_pp']:+.4f} pp。ASD 概率的病例排序有所改善，原 argmax 的净正确判定减少。使用同一历史决策规则。", ""])
    replayed = sum(r["ledger"]["replayed_epochs"] for r in rows)
    matched = sum(r["ledger"]["archived_epochs"] for r in rows)
    lines += ["## 独立审计及解释边界", "",
              f"逐样本概率、类别和 logits 在 Best 与 Latest 的记录评估状态精确一致。Latest 从完整状态继续一步训练、评估和 EMA 与在线见证精确一致。独立回放 {replayed} 个训练 epoch 的六次历史 Dropout 和四类 RNG，归档交集 {matched} epoch 的训练前、训练后、评估后共 {matched * 3} 个阶段全部一致。超出归档终点的 epoch 用同一原生随机操作序列回放。", "",
              f"20/20 通过原始路径关闭后精确行为、新参数初始化 RNG 隔离、Native CE 与残差的端到端梯度、逐族辅助梯度预算、权重冻结和 Fold 2 屏蔽。公式对照按浮点运算界验证，预测和状态身份按确切一致验证。历史 DARG、E82.2-G2、E82.3.2-G2 Best 逐折只读重放。E82.4.2-G2 复核存档输出和六项指标。{len(manifest['protected_sha256'])} 份保护文件在训练前后 SHA 一致。", "",
              "本结果继承单 seed、val=test 参与选模、CV 之前 RFE/标准化、转导患者图和 Fold 2 重叠。Fold 2 的 377 ASD、407 TD 是历史 1-based 行号，两行保留 DARG 成员身份并从新增拟合和监督排除，新增残差为零。不同实验的配对差受各自 Best epoch 影响。固定权重医学与匿名 Query 比较限于历史模型和数据协议，不能推出独立临床诊断价值或无偏外部泛化。", "",
              "[设计](../design.md) · [正式配置](../config.json) · [逐折指标](fold_metrics.csv) · [配对](paired_comparisons.csv) · [机制](mechanism.csv) · [患者 Margin](patient_margins.csv) · [审计](audit_status.json) · [Manifest](../manifest.json)", ""]
    return "\n".join(lines)


def append_handoff(body):
    path = h.HISTORY / "handoff.md"
    block = ("\n\n" + body.strip() + "\n").encode("utf-8")
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        before = stream.read()
        prefix_sha = hashlib.sha256(before).hexdigest()
        if before.endswith(block):
            offset = len(before) - len(block)
            prior = before[:offset]
            return {"path": str(path), "offset": offset, "historical_prefix_sha": hashlib.sha256(prior).hexdigest(),
                    "append_sha": hashlib.sha256(block).hexdigest(), "after_sha": hashlib.sha256(before).hexdigest(),
                    "verified": True, "idempotent": True}
        stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
        observed = path.read_bytes()
        h.require(observed == before + block and hashlib.sha256(observed[:len(before)]).hexdigest() == prefix_sha,
                  "Handoff readback did not preserve the historical prefix")
        return {"path": str(path), "offset": len(before), "historical_prefix_sha": prefix_sha,
                "append_sha": hashlib.sha256(block).hexdigest(), "after_sha": hashlib.sha256(observed).hexdigest(),
                "verified": True, "idempotent": False}


def finish(manifest):
    records = [h.read_json(paths(g, fold)["audit"]) for g in h.GROUPS for fold in h.FOLDS]
    h.require(len(records) == 20 and all(r["status"] == "PASS" for r in records), "Independent audit is incomplete")
    for fn in h.FOLDS:
        a, b = (next(r for r in records if r["fold"] == fn and r["group"] == g) for g in h.GROUPS)
        h.require(a["patient_rows_1based"] == b["patient_rows_1based"] and a["label"] == b["label"]
                  and a["historical"] == b["historical"], "Groups lack common paired patients/historical references")
    slots = [(r["fold"], i) for r in records if r["group"] == "G1" for i in r["patient_rows_1based"]]
    h.require(len(slots) == CFG["subjects"] and len({i for _, i in slots}) == CFG["subjects"],
              "Ten test folds do not give one prediction per historical subject")
    fold_rows, summary = summaries(records)
    paired, aggregate = comparison_rows(records)
    mechanisms, patients = mechanism_rows(records)
    decisions = decisions_for(records, aggregate, mechanisms)
    h.write_csv(RESULTS / "fold_metrics.csv", fold_rows)
    h.write_csv(RESULTS / "paired_comparisons.csv", paired)
    h.write_csv(RESULTS / "mechanism.csv", mechanisms)
    h.write_csv(RESULTS / "patient_margins.csv", patients)
    report = render_report(records, summary, aggregate, mechanisms, patients, decisions, manifest)
    (RESULTS / "report.md").write_text(report, encoding="utf-8")
    audit = {
        "status": "COMPLETE_AUDITED", "folds_trained": 20, "best_reloaded": 20, "latest_reloaded": 20,
        "latest_next_step_exact": 20, "metrics_independent": list(ALL_METRICS),
        "archived_rng_epochs": sum(r["ledger"]["archived_epochs"] for r in records),
        "all_rng_replayed_epochs": sum(r["ledger"]["replayed_epochs"] for r in records),
        "protected_sha256": manifest["guard_after"],
        "decisions": decisions, "results_sha256": {str(paths(r["group"], r["fold"])["audit"]):
                                                   h.sha_file(paths(r["group"], r["fold"])["audit"]) for r in records},
        "per_fold": [{k: r[k] for k in ("group", "fold", "epoch", "best_epoch", "best_sha", "latest_sha", "scope",
                                       "ledger", "best_gradients", "latest_reload", "original_o", "stats", "formula")}
                     for r in records],
    }
    handoff = [f"## {CFG['experiment']}", "",
               f"**COMPLETE_AUDITED**。正式目录 `{h.ROOT}`，OSSFS 首次 mkdir 报 errno 5。G1/G2 各 10/10，Best、Latest、Latest 下一步重放与 Native/残差梯度独立审计均 20/20。唯一物理 GPU0 `{CFG['gpu']['uuid']}`，seed 0，两波各并发 10。", "",
               f"全部 {audit['all_rng_replayed_epochs']} 个训练 epoch 的共有 RNG 操作回放一致，归档交集 {audit['archived_rng_epochs']} epoch 的三个阶段 SHA 完全一致。训练冻结 SHA `{manifest['implementation_sha']}`。", "",
               *performance_table(summary, "10_fold")]
    for group, d in decisions.items():
        p = aggregate[group, "10_fold", "DARG", "full"]
        shared = aggregate[group, "10_fold", "DARG", "residual_off"]
        direct = aggregate[group, "10_fold", "residual_off", "full"]
        nine = aggregate[group, "exclude_fold_2", "DARG", "full"]
        handoff.extend([f"{group} {d['overall']}，相对 DARG ΔACC/BA/SEN 为 {p['acc_pp']:+.4f}/{p['hard_auc_pp']:+.4f}/{p['sen_pp']:+.4f} pp，A→B 净 {shared['net']:+d}、B→C 净 {direct['net']:+d}、A→C 净 {p['net']:+d} 人。去 Fold 2 后 ΔACC {nine['acc_pp']:+.4f} pp，净 {nine['net']:+d} 人。医学 Query 替换概率变化 {d['medical_query_probability_changed']}、硬翻转 {d['medical_query_hard_flipped']}，医学相对匿名净 {d['medical_vs_anonymous_net']:+d} 人。性能 {'GO' if d['performance_go'] else 'NO-GO'}，机制 {'GO' if d['mechanism_go'] else 'NO-GO'}。", ""])
    handoff.extend(["局限为单 seed、历史 val=test、CV 前 RFE/预处理、转导图和 Fold 2 的 377/407 重叠。两位历史重叠患者的新增残差及新增有标签监督均为零。后续科研判断以固定权重净纠错和独立验证为依据。", "",
                    f"[报告]({RESULTS / 'report.md'}) · [逐折指标]({RESULTS / 'fold_metrics.csv'}) · [配对]({RESULTS / 'paired_comparisons.csv'}) · [机制]({RESULTS / 'mechanism.csv'}) · [独立审计]({RESULTS / 'audit_status.json'}) · [Manifest]({MANIFEST})"])
    audit["handoff"] = append_handoff("\n".join(handoff))
    h.write_json(RESULTS / "audit_status.json", audit)
    manifest.update(status="COMPLETE_AUDITED", scientific_decisions=decisions, handoff=audit["handoff"],
                    result_sha256={name: h.sha_file(RESULTS / name) for name in ("report.md", "fold_metrics.csv",
                                   "paired_comparisons.csv", "mechanism.csv", "patient_margins.csv", "audit_status.json")})
    h.write_json(MANIFEST, manifest)
    print("COMPLETE_AUDITED with independently verified handoff append", flush=True)
