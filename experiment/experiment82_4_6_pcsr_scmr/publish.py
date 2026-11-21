"""Build paired scientific outputs and append the verified execution handoff."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os

from common import history as h
from audit import METRICS, REFERENCES
from run import MANIFEST, RESULTS, paths

import numpy as np

CFG = h.CFG
DISPLAY = {g: f"82.4.6-{g} {name}" for g, name in CFG["groups"].items()}
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
    for key in ("mean_abs_probability_delta", "mean_true_margin_change", "mean_abs_true_margin_change"):
        out[key] = sum(row[key] * row["n_test"] for row in per_fold) / out["n_test"]
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
            actor["mean_true_margin_gain_vs_off"] = float(
                (np.asarray(entry["true_margin"]) - np.asarray(off["true_margin"])).mean())
            if key == "full":
                gate, ratio = (np.asarray(paths_[key]) for key in ("gate", "ratio"))
                for label, q in (("p10", .10), ("p50", .50), ("p90", .90)):
                    actor["gate_" + label] = float(np.quantile(gate, q))
                    actor["residual_ratio_" + label] = float(np.quantile(ratio, q))
                actor["gate_near_zero_fraction"] = float((gate < CFG["near_zero_gate_threshold"]).mean())
                actor["active_residual_fraction"] = float(
                    (np.asarray(paths_["ratio"]) > CFG["active_residual_relative"]).mean())
                actor["residual_exact_zero_fraction"] = float((ratio == 0).mean())
                actor["raw_above_cap_fraction"] = float(
                    (np.asarray(paths_["raw_ratio"]) > CFG["residual"]["fraction"]).mean())
                actor["material_tanh_shrink_fraction"] = float(
                    (np.asarray(paths_["bound_shrink"]) >= CFG["material_cap_attenuation"]).mean())
                actor.update({k: json.dumps(v, separators=(",", ":")) if isinstance(v, list) else v
                              for k, v in row["directions"].items()})
                actor["max_prompt_guidance_ratio"] = row["ledger"]["budget_maximum_ratio"]["prompt.guidance"]
                wrong = np.asarray(off["prediction"]) != np.asarray(row["label"])
                gain = np.asarray(entry["true_margin"]) - np.asarray(off["true_margin"])
                actor["originally_wrong_mean_margin_gain"] = float(gain[wrong].mean()) if wrong.any() else 0.
                actor["originally_correct_mean_margin_gain"] = float(gain[~wrong].mean()) if (~wrong).any() else 0.
                actor["originally_wrong_positive_gain_fraction"] = float((gain[wrong] > 0).mean()) if wrong.any() else 0.
                for family, tally in row["ledger"]["projection_routing"].items():
                    actor.update({f"{family}_{name}_per_step": tally[name] / tally["epochs"]
                                  for name in ("raw_norm", "tangent_norm", "applied_norm", "hist_norm")})
                    actor[f"{family}_max_applied_budget_fraction"] = row["ledger"]["budget_maximum_ratio"][family + ".full"]
            rows_metric.append(actor)
            for i, (patient, label) in enumerate(zip(row["patient_rows_1based"], row["label"])):
                diagnostics = {k: v[i] for k, v in paths_.items() if k != "intervention_residual_difference_l2"}
                patients.append({
                    "group": group, "fold": fold, "patient_row_1based": patient, "label": label, "intervention": key,
                    "fold2_historical_overlap": fold == 2 and patient in CFG["fold2_overlap_1based"],
                    "pred": entry["prediction"][i], "p_asd": entry["probability"][i], "pred_darg": original["prediction"][i],
                    "pred_residual_off": off["prediction"][i], "pred_full": full["prediction"][i],
                    "p_asd_darg": original["probability"][i], "p_asd_residual_off": off["probability"][i],
                    "p_asd_full": full["probability"][i], "p_asd_anonymous": row["interventions"]["anonymous_query"]["probability"][i],
                    "pred_anonymous": row["interventions"]["anonymous_query"]["prediction"][i],
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
    # At most 10 float64 rates in each paired mean. This tolerance removes
    # representational noise at zero and is far smaller than any score step.
    zero = 1e-12
    for group in h.GROUPS:
        def comparison(source, target, scope="10_fold"):
            return aggregate[group, scope, source, target]

        full = comparison("DARG", "full")
        nine = comparison("DARG", "full", "exclude_fold_2")
        direct = comparison("residual_off", "full")
        medical = comparison("anonymous_query", "full")
        gmech = [r for r in mechanisms if r["group"] == group]
        off = [r for r in gmech if r["intervention"] == "residual_off"]
        query = [r for r in gmech if r["intervention"] == "anonymous_query"]
        hard = {"acc": full["acc_pp"] >= -zero, "ba": full["hard_auc_pp"] >= -zero}
        parity = {**hard, "sen": full["sen_pp"] >= -zero, "spe": full["spe_pp"] >= -zero,
                  "prob_auc": full["prob_auc_pp"] >= -zero, "net": full["net"] >= 0}
        parity9 = {"acc": nine["acc_pp"] >= -zero, "net": nine["net"] >= 0}
        nontrivial = {"nonzero_cam": any(r["max_feature_delta_l2"] > 0 for r in off),
                      "nonzero_prompt": any(r["max_feature_delta_l2"] > 0 for r in query),
                      "nonzero_spectrum": any(r["parallel_ratio_mean_full"] > 0 for r in gmech),
                      "measurable_predictions": direct["probability_changed"] > 0,
                      "positive_net_or_ba": direct["net"] > 0 or direct["hard_auc_pp"] > zero,
                      "end_to_end": all(r["best_gradients"]["optimizer_receipt_exact"] and
                                       r["latest_reload"]["diagnostic"]["optimizer_receipt_exact"]
                                       for r in rows if r["group"] == group)}
        medical_evidence = {"nonzero_cam": any(r["max_feature_delta_l2"] > 0 for r in query),
                            "hard_flip": medical["hard_flipped"] > 0, "acc": medical["acc_pp"] > zero,
                            "ba": medical["hard_auc_pp"] > zero, "net": medical["net"] > 0}
        consistent = all(comp[key + "_wins"] >= comp[key + "_losses"] for comp in (full, nine)
                         for key in ("acc", "hard_auc"))
        candidate = {"strict_darg_parity": all(parity.values()), "nontrivial_residual": all(nontrivial.values()),
                     "acc": full["acc_pp"] > zero, "ba": full["hard_auc_pp"] > zero, "net": full["net"] > 0,
                     "exclude_fold_2_acc": nine["acc_pp"] > zero, "exclude_fold_2_ba": nine["hard_auc_pp"] > zero,
                     "exclude_fold_2_net": nine["net"] > 0, "medical_query_benefit": all(medical_evidence.values()),
                     "acc_ba_fold_consistency": consistent}
        performance_pass, mechanism_pass = all(parity.values()), all(nontrivial.values())
        decisions[group] = {
            "hard_classification_parity": all(hard.values()), "strict_darg_parity": performance_pass,
            "exclude_fold_2_parity": all(parity9.values()), "nontrivial_residual": mechanism_pass,
            "innovation_candidate": all(candidate.values()),
            "performance_mechanism": ("PERFORMANCE PARITY" if performance_pass else "PERFORMANCE NO-GO") +
                                     (" / MECHANISM GO" if mechanism_pass else " / MECHANISM NO-GO"),
            "parity_checks": parity, "exclude_fold_2_parity_checks": parity9, "residual_checks": nontrivial,
            "medical_checks": medical_evidence, "candidate_checks": candidate,
            "medical_query_probability_changed": medical["probability_changed"],
            "medical_query_hard_flipped": medical["hard_flipped"], "medical_vs_anonymous_net": medical["net"],
            "medical_vs_anonymous_acc_pp": medical["acc_pp"], "medical_vs_anonymous_ba_pp": medical["hard_auc_pp"],
            "observable_diagnostic_gain_vs_anonymous": all(medical_evidence.values()),
        }
    return decisions


def performance_table(summary, scope):
    return table(("方法", "ACC", "Hard-AUC/BA", "Prob-AUC", "SEN-ASD", "SPE-TD", "ASD-F1", "Brier", "ECE"),
                 [(DISPLAY.get(name, name), *(f"{100*a:.4f} ± {100*b:.4f}" for a, b in
                                             (summary[name, scope][k] for k in ALL_METRICS)))
                  for name in (*REFERENCES, *h.GROUPS)])


def pair_table(aggregate, scope):
    names = (*REFERENCES, "residual_off", "anonymous_query")
    names_display = {"residual_off": "Residual-off", "anonymous_query": "Anonymous Query", "full": "Full"}
    data = []
    for group in h.GROUPS:
        for source, target in pair_sets(group):
            if source not in names:
                continue
            a = aggregate[group, scope, source, target]
            data.append((group, names_display.get(source, source) + " → " + (
                "Medical Query" if source == "anonymous_query" else names_display.get(target, target)),
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
    keys = ("gate_mean", "gate_p10", "gate_p50", "gate_p90", "gate_near_zero", "residual_mean",
            "residual_active", "wrong_positive_share", "correct_negative_share")
    return [float(np.mean([r[k] for r in values])) for k in keys]


def group_mechanism(rows, mechanism, patients, group):
    tests = [r for r in patients if r["group"] == group and r["intervention"] == "full"]
    score = lambda name: np.asarray([r[name] for r in tests])
    gate, ratio = score("gate"), score("ratio")
    y, p_off = score("label"), score("pred_residual_off")
    gain, margin_off = score("margin_vs_residual_off"), score("margin_residual_off")
    wrong = p_off != y
    favorable = wrong & (gain > 0) & (margin_off < 0)
    progress = gain[favorable] / abs(margin_off[favorable])
    summary = {"gate_mean": gate.mean(), "gate_p10": np.quantile(gate, .10), "gate_p50": np.quantile(gate, .50),
               "gate_p90": np.quantile(gate, .90), "gate_near_zero": (gate < CFG["near_zero_gate_threshold"]).mean(),
               "ratio": ratio.mean(), "ratio_p10": np.quantile(ratio, .1), "ratio_p50": np.quantile(ratio, .5),
               "ratio_p90": np.quantile(ratio, .9), "active": (ratio > CFG["active_residual_relative"]).mean(),
               "cap_crossed": (score("raw_ratio") > CFG["residual"]["fraction"]).mean(),
               "exactly_zero": (ratio == 0).mean(), "practically_off": (ratio <= CFG["active_residual_relative"]).mean(),
               "material_shrink": (score("bound_shrink") >= CFG["material_cap_attenuation"]).mean(),
               "wrong_gain": gain[wrong].mean(), "correct_gain": gain[~wrong].mean(),
               "wrong_positive": (gain[wrong] > 0).mean(), "correct_negative": (gain[~wrong] < 0).mean(),
               "wrong_n": int(wrong.sum()), "positive_n": int(favorable.sum()),
               "hard_corrected_n": int((wrong & (score("pred_full") == y)).sum()),
               "progress_p50": np.median(progress) if len(progress) else 0.,
               "progress_max": progress.max() if len(progress) else 0.,
               "parallel": score("parallel_ratio").mean(), "complement": score("complement_ratio").mean(),
               "remaining_distance_off_wrong": np.maximum(-score("margin_full")[wrong], 0).mean(),
               "action_abs_mean": score("action_abs_mean").mean(),
               "query_control_abs_change": score("query_control_abs_change").mean()}
    darg_wrong = score("pred_darg") != y
    darg_gain = score("margin_vs_darg")
    darg_margins = score("margin_darg")
    predicted_correct = score("pred_full") == y
    darg_favorable = darg_wrong & (darg_gain > 0) & (darg_margins < 0)
    darg_progress = darg_gain[darg_favorable] / abs(darg_margins[darg_favorable])
    summary.update(darg_wrong_n=int(darg_wrong.sum()), darg_wrong_gain=darg_gain[darg_wrong].mean(),
                   darg_wrong_positive=(darg_gain[darg_wrong] > 0).mean(),
                   darg_wrong_crossed=int((darg_wrong & predicted_correct).sum()),
                   darg_remaining_distance=np.maximum(-score("margin_full")[darg_wrong], 0).mean(),
                   darg_progress_p50=np.median(darg_progress) if darg_progress.size else 0.,
                   darg_correct_negative=(darg_gain[~darg_wrong] < 0).mean(),
                   darg_correct_harmed=int((~darg_wrong & ~predicted_correct).sum()))
    if group == "G1":
        summary.update(query_direction_abs_change=score("query_direction_mean_abs_change").mean(),
                       rotation_max_radians=score("rotation_max_radians").max(),
                       amplitude_negative_fraction=(score("amplitude") < 0).mean())
    else:
        for key in ("llr_correct_sign", "target_shift_correct_sign"):
            summary[key + "_darg_wrong"] = score(key)[darg_wrong].mean()
            summary[key + "_off_wrong"] = score(key)[wrong].mean()
        summary.update(trust_mean=score("correction_reliability").mean(),
                       disagreement_mean=score("disagreement").mean(), allocation_mean=score("allocation").mean())
    prior_native_net, prior_direct_net, prior_gains, matched_diffs = 0, 0, [], []
    for record in (r for r in rows if r["group"] == group):
        truth = np.asarray(record["label"])
        prev = record["historical"]["E82.4.5-" + group]
        off_prev = np.asarray(prev["off_prediction"])
        direct_prev = np.asarray(prev["true_margin"]) - np.asarray(prev["off_true_margin"])
        darg = np.asarray(record["historical"]["DARG"]["prediction"])
        prior_native_net += int((off_prev == truth).sum() - (darg == truth).sum())
        prior_direct_net += int((np.asarray(prev["prediction"]) == truth).sum() - (off_prev == truth).sum())
        prior_gains.extend(direct_prev[off_prev != truth])
        local = [r for r in tests if r["fold"] == record["fold"]]
        h.require([r["patient_row_1based"] for r in local] == record["patient_rows_1based"], "History pairing lost")
        for i, patient in enumerate(local):
            if off_prev[i] != truth[i] and patient["pred_residual_off"] != truth[i]:
                matched_diffs.append(patient["margin_vs_residual_off"] - direct_prev[i])
    summary.update(prior_native_net=prior_native_net, prior_direct_net=prior_direct_net, prior_wrong_gain=np.mean(prior_gains),
                   matched_wrong_mean_gain_change=np.mean(matched_diffs) if matched_diffs else 0.,
                   matched_wrong=len(matched_diffs))
    current = [a for a in mechanism if a["group"] == group and a["intervention"] == "full"]
    summary["guidance_max_ratio"] = max(a["max_prompt_guidance_ratio"] for a in current)
    return {k: v.item() if isinstance(v, np.generic) else v for k, v in summary.items()}


def shared_gradient_table(rows):
    batch, movement = [], []
    for group in h.GROUPS:
        subset = [r for r in rows if r["group"] == group]
        for family in ("gfr", "darg", "cam", "head"):
            records = [r["ledger"]["projection_routing"][family] for r in subset]
            totals = {k: sum(r[k] for r in records) for k in records[0]}
            max_ratio = max(r["ledger"]["budget_maximum_ratio"][family + ".full"] for r in subset)
            max_dot = max(r["ledger"]["budget_maximum_ratio"][family + ".abs_dot_utilization"] for r in subset)
            n = totals["epochs"]
            batch.append((group, family, n, *(f"{totals[k]/n:.4g}" for k in
                          ("hist_norm", "raw_norm", "tangent_norm", "applied_norm")),
                          *(f"{totals[k]/n:.2%}" for k in ("applied", "capped")),
                          "/".join(str(int(totals[k])) for k in ("near_zero", "rounded_to_zero", "guard_reject", "nonfinite_difference")),
                          f"{max_ratio:.6%}", f"{max_dot:.3%}"))
            probes = [row[stage] for row in subset for stage in ("best_gradients", "latest_reload")]
            probes = [p.get("diagnostic", p)["gradients"][family]["optimizer_delta"] for p in probes]
            motion = [p["difference_norm"] / p["same_state_native_norm"] for p in probes if p["same_state_native_norm"] > 0]
            movement.append((group, family, f"{totals['update_norm']/n:.4g}",
                             f"{totals['update_dot_historical']/n:+.4g}", f"{totals['update_dot_extra']/n:+.4g}",
                             f"{np.mean(motion):.4%}", f"{max(motion):.4%}"))
    return [*table(("组", "历史族", "步数", "历史梯度", "原差分梯度", "切空间梯度", "写入附加梯度",
                    "施加率", "需缩放率", "近零/舍零/保护跳过/非有限差分", "最大附加比例", "舍入界最大占比"), batch),
            "范数为各步均值。下表的相对参数位移差取 Best 和 Latest 处从同一 Adam/AdamW 状态分出的真实更新。"
            "原生与增强更新的差异还累积进入优化器状态，单步测量不能推出整段参数轨迹相同。", "",
            *table(("组", "历史族", "实际位移范数", "位移·历史单位梯度", "位移·实际附加梯度",
                    "同状态位移差均值", "同状态位移差最大"), movement)]


def render_report(rows, summary, aggregate, mechanisms, decision, manifest, diag):
    lines = [f"# {CFG['experiment']}", "",
             "**COMPLETE_AUDITED**。两个正式组各完成十折，Best、Latest 与下一步精确重放各 20/20。", "",
             "本实验检验保留历史 DARG 主分类训练后，医学 Prompt 谱残差能否改正真实分类错误并保住基座性能。", "",
             "G1 PCSR-PSCE 继承 E82.2-G2 的 Original O 及医学、统计权重，保留原 Query 及公共方向部分抑制后的八条 Query，学习 U16 内患者旋转、有符号方向和可归零的幅度。G2 SCMR-DA-SRPE 继承 E82.3.2-G2 的合法五折 Gaussian LLR 统计与评估后 EMA，学习有界目标、可信度与两个 Jacobian 空间的贡献分配。", "",
             "原论文在 CAM 后用 SPGS 与 RES 构成 PEN。历史实际链为 GFR → Positive-DCR DARG → PLNNoPEN CAM → CAM144 → 校正特征 → 原 OutputLayer。两组固定原 U16、八条 RN50 Query、Null 及相对 CAM 范数 8% 平滑上限。[MICCAI 2025 原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)", "",
             "82.4.5 的完整确定性 Full CE 梯度即使在 Δ=0 时也可能改变历史更新。本轮共享附加量采用同一确定性头的 ∇(Full CE−Off CE)，投影至完整历史损失梯度的正交切空间，按原 dtype 写入后复核 0.1% 上限。新增模块直接接受 Full CE 和低强度有预算辅助梯度。", "",
             "G1 将 4(w ⊙ a) 通过两平面正交旋转后用 Original O 写回，患者幅度为有符号 tanh，初始化为零。G2 的共同目标变化为可信度乘有界 Margin 修正，按学习到的分配比例分到两个岭正则 Jacobian 载体。训练用真实原头计算谱内、谱外和联合候选的 Margin。两组统一做 8% tanh 平滑限幅。[数学定义与来源核验](../design.md)", "",
             f"seed 0。从逐折 E61 epoch 10 恢复，训练 epoch 11–450，wait > 200 早停。Best 为首次严格最大 test ACC。三历史优化器依次为 Adam(0.0038, 0.11)、Adam(0.0038, 0.11)、AdamW(0.001, 0.01)，新 AdamW(0.0005, 0.01) 最后执行。仅物理 GPU0 `{CFG['gpu']['uuid']}`，顺序两波最多十 workers。训练冻结 SHA `{manifest['implementation_sha']}`。", "",
             "## 十折正式指标和历史母方案", "",
             "均值 ± population SD。ASD=0 为阳性，Hard-AUC 是硬分类 BA，Prob-AUC 基于连续 p(ASD)。Brier 与等宽十区间 ECE 乘 100。逐折病例数、Best epoch、混淆矩阵和终止 epoch 同表保存。[逐折指标](fold_metrics.csv)", "",
             *performance_table(summary, "10_fold"),
             "## 逐折配对与固定权重纠错", "",
             "配对数值为各折差值均值，以百分点表示。DARG → Residual-off 同时包含训练轨迹和不同 Best epoch 的影响，Residual-off → Full 使用同一 Best 权重。病例纠错和误伤来自逐患者复算。[逐折配对](paired_comparisons.csv) · [患者标签、概率与 Margin](patient_margins.csv)", "",
             *pair_table(aggregate, "10_fold"),
             "## 医学 Query 与机制干预", "",
             "下表的病例变化方向为 Full → 干预。概率变化阈值为 |Δp(ASD)| > 10⁻⁶。干预均从当前 CAM144 重算残差，并经过真实原头。匿名 → 医学的指标变化列在配对表。", ""]
    intervention = []
    for group in h.GROUPS:
        for name in dict.fromkeys(r["intervention"] for r in mechanisms if r["group"] == group):
            if name == "full":
                continue
            block = [r for r in mechanisms if r["group"] == group and r["intervention"] == name]
            n = sum(r["n_test"] for r in block)
            intervention.append((group, name, *(sum(r[k] for r in block) for k in ("probability_changed", "hard_flipped")),
                                 "/".join(str(sum(r[k] for r in block)) for k in ("corrected", "harmed", "net")),
                                 *(f"{sum(r[k] * r['n_test'] for r in block)/n:.5g}" for k in
                                   ("mean_feature_delta_l2", "absolute_logit_delta", "probability_mae",
                                    "mean_true_margin_gain_vs_off"))))
    lines += table(("组", "干预", "概率变化", "硬翻转", "纠/伤/净", "CAM L2", "logit MAE", "概率 MAE", "相对 off 真类 Margin"), intervention)
    lines += ["训练诊断按各折取合法行。活跃残差的相对范数 >10⁻⁵，近零门控 <0.01。", "",
              *table(("组", "阶段", "门控均值", "P10", "P50", "P90", "近零比例", "残差比", "活跃比例",
                      "原错正增益", "原对负增益"),
                     [(group, label, *(f"{x:.5f}" for x in dynamics(rows, group, phase)))
                      for group in h.GROUPS for label, phase in (("第 1 步", "first"), ("第 10 步", "tenth"),
                                                                  ("Best", "Best"), ("Latest", "Latest"))])]
    for group, d in diag.items():
        a = aggregate[group, "10_fold", "DARG", "residual_off"]
        b = aggregate[group, "10_fold", "residual_off", "full"]
        lines += [
            f"{group} 门控绝对值均值 {d['gate_mean']:.4f}，P10/P50/P90 {d['gate_p10']:.4f}/{d['gate_p50']:.4f}/{d['gate_p90']:.4f}，近零率 {d['gate_near_zero']:.2%}。相对残差均值 {d['ratio']:.3%}，P10/P50/P90 {d['ratio_p10']:.3%}/{d['ratio_p50']:.3%}/{d['ratio_p90']:.3%}，精确关闭率 {d['exactly_zero']:.2%}，活跃率 {d['active']:.2%}。raw 超 8% {d['cap_crossed']:.2%}，tanh 缩幅至少 5% {d['material_shrink']:.2%}。", "",
            f"{group} 固定权重下 {d['wrong_n']} 个原错行的真类 Margin 平均变化 {d['wrong_gain']:+.5f}，正增益 {d['wrong_positive']:.2%}，实际跨界 {d['hard_corrected_n']} 人，剩余边界距离均值 {d['remaining_distance_off_wrong']:.5f}。正增益占跨界距离的中位 {d['progress_p50']:.1%}。原对行平均变化 {d['correct_gain']:+.5f}，负增益 {d['correct_negative']:.2%}。82.4.5 的原错均值为 {d['prior_wrong_gain']:+.5f}，两轮均原错的 {d['matched_wrong']} 人增益差均值 {d['matched_wrong_mean_gain_change']:+.5f}。", "",
            f"{group} 的 {d['darg_wrong_n']} 位 DARG 原错患者在 Full 中真类 Margin 平均增益 {d['darg_wrong_gain']:+.5f}，{d['darg_wrong_positive']:.2%} 正增益，实际跨界 {d['darg_wrong_crossed']} 人，剩余边界距离均值 {d['darg_remaining_distance']:.5f}。DARG 原对病例负向变化比例 {d['darg_correct_negative']:.2%}，误伤 {d['darg_correct_harmed']} 人。", "",
            f"{group} 的 DARG → Residual-off 净 {a['net']:+d} 人，82.4.5 为 {d['prior_native_net']:+d}。本轮直接残差纠正 {b['corrected']} 人、误伤 {b['harmed']} 人，净 {b['net']:+d}，82.4.5 净 {d['prior_direct_net']:+d}。谱内/外相对残差均值 {d['parallel']:.3%}/{d['complement']:.3%}，Prompt 导向辅助梯度占 Full 主梯度的峰值 {d['guidance_max_ratio']:.3%}。", ""]
    g1, g2 = diag["G1"], [r for r in rows if r["group"] == "G2"]
    calibrated = diag["G2"]
    average = lambda key: np.mean([r["directions"][key] for r in g2])
    lines += [
        f"G1 患者旋转的最大角 {g1['rotation_max_radians']:.5f} rad，负幅度比例 {g1['amplitude_negative_fraction']:.2%}。匿名替换的平均绝对动作变化 {g1['query_direction_abs_change']:.5f}，幅度变化 {g1['query_control_abs_change']:.5f}。", "",
        f"G2 在 DARG 原错病例中，Prompt 加权 Gaussian 证据与真类同号率 {calibrated['llr_correct_sign_darg_wrong']:.2%}，校准动作同号率 {calibrated['target_shift_correct_sign_darg_wrong']:.2%}。在当前 Residual-off 原错病例中分别为 {calibrated['llr_correct_sign_off_wrong']:.2%} 和 {calibrated['target_shift_correct_sign_off_wrong']:.2%}。平均可信度 {calibrated['trust_mean']:.4f}，五个参照银行的 LLR 离散度摘要均值 {calibrated['disagreement_mean']:.4f}，谱内分配均值 {calibrated['allocation_mean']:.4f}。", "",
        f"G2 谱内/外一阶预测变化反号率 {average('first_order_opposite_sign_share'):.2%}，原头实算的反号率 {average('actual_head_spaces_opposite_sign_share'):.2%}。仅谱内、仅谱外对当前原错行的真类 Margin 折均变化 {average('spectral_only_wrong_mean_gain'):+.5f}/{average('complement_only_wrong_mean_gain'):+.5f}。以相同可信度和分配、原 LLR 替代校准方向时为 {average('calibrated_evidence_off_wrong_mean_gain'):+.5f}。单空间干预删除另一位移后重新限幅，两个空间的最终幅度可不同。", "",
        "机制表包含 G1 的逐方向 w_sem、w_sp、动作、医学注意力、正常与异常对比和原头单方向实际 Margin，也包含 G2 的 Gaussian LLR、可靠度、谱内外动作及干预指标。[机制数据](mechanism.csv)", "",
        "## 共享梯度与实际参数位移", "",
        *shared_gradient_table(rows),
        "## 排除 Fold 2 的九折敏感性", "",
        *performance_table(summary, "exclude_fold_2"), *pair_table(aggregate, "exclude_fold_2"),
        "## 独立审计与适用范围", ""]
    replayed = sum(r["ledger"]["replayed_epochs"] for r in rows)
    comparable = sum(r["ledger"]["archived_epochs"] for r in rows)
    ema_count = sum(r["ledger"]["replayed_epochs"] for r in g2)
    exact_ema = sum(r["ledger"]["ema_bitwise_epochs"] for r in g2)
    lines += [
        f"逐患者概率、类别、logits 在 Best 及 Latest 的记录评估状态精确复现。Latest 在独立新对象上的下一步模型、四个优化器、评估、EMA 与在线见证精确一致。历史六次 Dropout 回放覆盖 {replayed} epoch，与归档重合的 {comparable} epoch 共 {3 * comparable} 个 RNG 阶段全部一致。独立 NumPy EMA 公式核对精确哈希的 epoch 为 {exact_ema}/{ema_count}，Best/Latest 的 EMA 与阶段按 SHA 验证，公式误差界见独立审计。", "",
        f"20/20 零残差首步保持前向、历史损失、梯度、实际 Adam/AdamW 更新和 RNG 完全一致。非零残差预检独立复算了差分梯度、原 dtype 实际写入预算、端到端路径、G1 正交旋转和 G2 校准后的双空间更新。历史 DARG、两母方案和 82.4.5 两组都只读复算 Best。{len(manifest['protected_sha256'])} 份保护文件前后 SHA 一致。", "",
        "单 seed、历史 val=test 选模、CV 前 RFE/预处理、转导患者图及 Fold 2 重叠限制外部泛化解释。Fold 2 训练成员中的 1-based 行 377 ASD 和 407 TD 保留历史图及关系对，新监督、统计和实际残差为零。医学与匹配匿名 Query 的配对描述限于本历史协议。", "",
        "## 科研判断", ""]
    for group, d in decision.items():
        a = aggregate[group, "10_fold", "DARG", "full"]
        native = aggregate[group, "10_fold", "DARG", "residual_off"]
        direct = aggregate[group, "10_fold", "residual_off", "full"]
        no = lambda gate: "、".join(k for k, value in d[gate].items() if not value) or "无"
        lines += [
            f"**{group} {d['performance_mechanism']}**。基础 ACC 和 BA 同追平 {'通过' if d['hard_classification_parity'] else '未通过'}，严格 DARG-PARITY {'通过' if d['strict_darg_parity'] else '未通过'}，排除 Fold 2 后 ACC 和净正确追平 {'通过' if d['exclude_fold_2_parity'] else '未通过'}，NONTRIVIAL-RESIDUAL {'通过' if d['nontrivial_residual'] else '未通过'}，INNOVATION-CANDIDATE {'通过' if d['innovation_candidate'] else '未通过'}。", "",
            f"未达条件为严格追平 {no('parity_checks')}，去除 Fold 2 后追平 {no('exclude_fold_2_parity_checks')}，残差 {no('residual_checks')}，创新候选 {no('candidate_checks')}。医学 Query 改变 {d['medical_query_probability_changed']} 人的概率、翻转 {d['medical_query_hard_flipped']} 人，匿名 → 医学净 {d['medical_vs_anonymous_net']:+d}，ΔACC/BA {d['medical_vs_anonymous_acc_pp']:+.4f}/{d['medical_vs_anonymous_ba_pp']:+.4f} pp。", "",
            f"完整相对 DARG 净 {a['net']:+d} 人。原路径变化 {native['net']:+d} 人，直接残差变化 {direct['net']:+d} 人。完整性能差可由这一患者级分解定位，当前协议无法进一步分离原路径的训练漂移和 Best 轮次差。", ""]
        if a["prob_auc_pp"] > 0 and a["acc_pp"] < 0:
            lines += [f"{group} 的 Prob-AUC 增加 {a['prob_auc_pp']:+.4f} pp，病例排序有所改善。沿原 argmax 的 ACC 同时下降 {a['acc_pp']:+.4f} pp。", ""]
    lines += ["后续研究应根据原错病例实际真类 Margin 的符号与剩余跨界距离界定可改进环节。候选的外部价值需要独立患者划分上的确认。", "",
              "[设计](../design.md) · [正式配置](../config.json) · [预检](preflight.json) · [独立审计](audit_status.json) · [Manifest](../manifest.json)", ""]
    return "\n".join(lines)


def append_handoff(body):
    path = h.HISTORY / "handoff.md"
    block = ("\n\n" + body.strip() + "\n").encode("utf-8")
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        before = stream.read()
        prefix_sha = hashlib.sha256(before).hexdigest()
        marker = f"## {CFG['experiment']}".encode()
        if block in before:
            offset = before.index(block)
            prior = before[:offset]
            return {"path": str(path), "offset": offset, "historical_prefix_sha": hashlib.sha256(prior).hexdigest(),
                    "append_sha": hashlib.sha256(block).hexdigest(), "after_sha": hashlib.sha256(before).hexdigest(),
                    "verified": True, "idempotent": True}
        h.require(marker not in before, "A different 82.4.6 handoff entry already exists")
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
    h.require(len(records) == 20 and all(r["status"] == "PASS" and r["zero_residual_exact"] for r in records), "Independent audit is incomplete")
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
    diagnostics = {g: group_mechanism(records, mechanisms, patients, g) for g in h.GROUPS}
    report = render_report(records, summary, aggregate, mechanisms, decisions, manifest, diagnostics)
    (RESULTS / "report.md").write_text(report, encoding="utf-8")
    audit = {
        "status": "COMPLETE_AUDITED", "folds_trained": 20, "best_reloaded": 20, "latest_reloaded": 20,
        "latest_next_step_exact": 20, "zero_residual_history_exact": sum(r["zero_residual_exact"] for r in records),
        "metrics_independent": list(ALL_METRICS),
        "archived_rng_epochs": sum(r["ledger"]["archived_epochs"] for r in records),
        "all_rng_replayed_epochs": sum(r["ledger"]["replayed_epochs"] for r in records),
        "protected_sha256": manifest["guard_after"],
        "decisions": decisions, "mechanisms": diagnostics, "results_sha256": {str(paths(r["group"], r["fold"])["audit"]):
                                                   h.sha_file(paths(r["group"], r["fold"])["audit"]) for r in records},
        "per_fold": [{k: r[k] for k in ("group", "fold", "epoch", "best_epoch", "best_sha", "latest_sha", "scope",
                                       "ledger", "best_gradients", "latest_reload", "difficulty_memory",
                                       "original_o", "stats", "formula", "directions", "zero_residual_exact")}
                     for r in records],
    }
    handoff = [f"## {CFG['experiment']}", "",
               f"**COMPLETE_AUDITED**。正式目录 `{h.ROOT}`，OSSFS 首次 mkdir 报 errno 5。G1/G2 各 10/10，Best、Latest、Latest 下一步重放与共享差分梯度独立审计均 20/20。唯一物理 GPU0 `{CFG['gpu']['uuid']}`，seed 0，两波各并发 10。", "",
               f"全部 {audit['all_rng_replayed_epochs']} 个训练 epoch 的共有 RNG 操作回放一致，归档交集 {audit['archived_rng_epochs']} epoch 的三个阶段 SHA 完全一致。训练冻结 SHA `{manifest['implementation_sha']}`。", "",
               f"服务器 Git HEAD `{manifest['sources']['server_commit']}`。G1 PCSR-PSCE 结合双语义、近恒等 U16 患者旋转和有符号幅度。G2 SCMR-DA-SRPE 以校准目标和可信度驱动两个正交空间。两组共享差分梯度 0.1% 上限，零残差时逐折复现原历史更新。", "",
               *performance_table(summary, "10_fold")]
    for group, d in decisions.items():
        p = aggregate[group, "10_fold", "DARG", "full"]
        shared = aggregate[group, "10_fold", "DARG", "residual_off"]
        direct = aggregate[group, "10_fold", "residual_off", "full"]
        nine = aggregate[group, "exclude_fold_2", "DARG", "full"]
        m = diagnostics[group]
        handoff.extend([f"{group} {d['performance_mechanism']}，相对 DARG ΔACC/BA/SEN 为 {p['acc_pp']:+.4f}/{p['hard_auc_pp']:+.4f}/{p['sen_pp']:+.4f} pp。DARG → 原路径净 {shared['net']:+d}、原路径 → Full 净 {direct['net']:+d}、DARG → Full 净 {p['net']:+d} 人。去 Fold 2 后 ΔACC {nine['acc_pp']:+.4f} pp，净 {nine['net']:+d} 人。残差相对范数均值 {m['ratio']:.3%}，原错真类 Margin 增益 {m['wrong_gain']:+.5f}。医学 Query 替换引起 {d['medical_query_probability_changed']} 人概率变化和 {d['medical_query_hard_flipped']} 人硬翻转，医学相对匿名净 {d['medical_vs_anonymous_net']:+d} 人。严格 DARG-PARITY {'通过' if d['strict_darg_parity'] else '未通过'}，INNOVATION-CANDIDATE {'通过' if d['innovation_candidate'] else '未通过'}。", ""])
    handoff.extend(["局限为单 seed、历史 val=test、CV 前 RFE/预处理、转导图和 Fold 2 的 377/407 重叠。两位历史重叠患者的新增残差及新增有标签监督均为零。后续科研判断以固定权重净纠错和独立验证为依据。", "",
                    f"[报告]({RESULTS / 'report.md'}) · [逐折指标]({RESULTS / 'fold_metrics.csv'}) · [配对]({RESULTS / 'paired_comparisons.csv'}) · [机制]({RESULTS / 'mechanism.csv'}) · [独立审计]({RESULTS / 'audit_status.json'}) · [Manifest]({MANIFEST})"])
    if manifest.get("context_recoveries"):
        handoff += ["", "未完成折的 GPU 上下文恢复边界已独立验证，详见 Manifest。"]
    audit["handoff"] = append_handoff("\n".join(handoff))
    h.write_json(RESULTS / "audit_status.json", audit)
    manifest.update(status="COMPLETE_AUDITED", scientific_decisions=decisions, handoff=audit["handoff"],
                    result_sha256={name: h.sha_file(RESULTS / name) for name in ("report.md", "fold_metrics.csv",
                                   "paired_comparisons.csv", "mechanism.csv", "patient_margins.csv", "audit_status.json")})
    h.write_json(MANIFEST, manifest)
    print("COMPLETE_AUDITED with independently verified handoff append", flush=True)
