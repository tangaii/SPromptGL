"""Recalculate and publish the six independent Fold 8 runs."""
from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path

from core import CONFIG, FOLD, GROUPS, MANIFEST, RESULTS, ROOT, RUNTIME, SOURCES, h, paths
from sources import (CAM_FIELDS, E6, E7, FIELDS, FOLDS, INTERVENTIONS, METRICS,
                     archived, read_csv, score, utc_now, verify_archived_metrics, verify_guard)

import numpy as np

controller = h.archive.load_module("_e82469_publish_controller", ROOT / "execute.py")
COMPLETE = controller.COMPLETE
FOLD2_PROOF = controller.FOLD2_PROOF
HISTORY_REPLAY = controller.HISTORY_REPLAY

FULL = "full"
E6_FULL = "e466_full"
E7_FULL = "e467_full"
OFF = "residual_off"
ANON = "anonymous"
METHOD = "E82.4.6.9"
LABEL = {0: "ASD", 1: "TD"}
COHORTS = {"A": "本组目标", "B": "其余五名候选 ASD", "C": "其余 81 人"}
HEADINGS = ("ACC", "Hard-AUC/BA", "Prob-AUC", "SEN-ASD", "SPE-TD", "ASD-F1")
HISTORICAL = ((E6, E6_FULL), (E7, E7_FULL))
HANDOFF = Path("/model/spromptgl/experiment/handoff.md")
MARKER = "## E82.4.6.9 — Six Independent Fold-8 ASD Patient Additions"


def flatten(folds):
    return [row for fold in FOLDS for row in folds[fold]]


def historical_values(row, source):
    return {f"prediction_{source}_{mode}": int(row[f"prediction_{mode}"]) for mode in INTERVENTIONS} | {
        f"{prefix}_{source}_{mode}": float(row[f"{prefix}_{mode}"])
        for mode in INTERVENTIONS for prefix in ("p_asd", "true_margin")
    }


def build_patients(group, old, history, certificate):
    target = GROUPS[group]["patient_id_1based"]
    candidates = {g["patient_id_1based"] for g in GROUPS.values()}
    new_rows = {row["patient_id_1based"]: row for row in certificate["patients"]}
    source8 = old["e466"][FOLD]
    h.require(list(new_rows) == [int(row["patient_id_1based"]) for row in source8],
              f"{group} Fold 8 patient order differs from the historical test set")
    result = {}
    artifact_sha = certificate["artifact_sha256"]["best"]
    for fold in FOLDS:
        historical_scope = history[str(fold)]
        original, direct = (old[source][fold] for source in ("e466", "e467"))
        result[fold] = []
        for source, reference in zip(original, direct):
            subject = int(source["patient_id_1based"])
            label = int(source["true_label"])
            local = fold == FOLD
            overlap = subject == target if local else subject in historical_scope["overlap_ids_1based"]
            pair_count = (certificate["membership"]["target_pair_endpoint_occurrences"] if overlap else 0) if local else (
                historical_scope["overlap_relation_occurrences"].get(str(subject), 0))
            rec = {
                "group": group, "method": f"{METHOD}-{group}", "fold": fold, "patient_id_1based": subject,
                "patient_index_0based": int(source["patient_index_0based"]), "true_label": label, "true_diagnosis": LABEL[label],
                "fold8_cohort": ("A" if overlap else "B" if subject in candidates else "C") if local else "",
                "group_target_1based": target, "darg_classification_supervised": overlap,
                "darg_relation_eligible": overlap, "darg_relation_sampled": pair_count > 0, "darg_relation_pair_occurrences": pair_count,
                "darg_relation_target_direct_gradient": certificate["best_gradients"]["relation_target_direct_bce_gradient"] if local and overlap else "",
                "scmr_label_supervised": False, "gaussian_member": False, "ema_member": False,
                "residual_masked": overlap, "residual_allowed_by_mask": not overlap,
                "best_sha256": artifact_sha if local else historical_scope["checkpoint"]["best_sha"],
                "checkpoint_source": f"{METHOD}-{group}" if local else " → ".join(historical_scope["checkpoint"]["lineage"]),
                "e466_best_sha256": source["current_best_sha"], "e467_best_sha256": reference["current_best_sha"],
            }
            rec.update(historical_values(source, "e466"))
            rec.update(historical_values(reference, "e467"))
            rec.update({f"e466_{key}": float(source[key]) for key in CAM_FIELDS})
            rec.update({f"e467_{key}": float(reference[key]) for key in CAM_FIELDS})
            if local:
                fresh = new_rows[subject]
                h.require(fresh["true_label"] == label, "Fold 8 ground truth changed")
                rec.update({key: value for key, value in fresh.items()
                            if key not in ("patient_id_1based", "true_label")})
            else:
                for key in (*FIELDS, *CAM_FIELDS):
                    rec[key] = (int if key.startswith("prediction") else float)(source[key])
                h.require(all(rec[key] == (int if key.startswith("prediction") else float)(source[key])
                              for key in (*FIELDS, *CAM_FIELDS))
                          and rec["best_sha256"] == rec["e466_best_sha256"], "Reused prediction or Best differs")
            for left, right, name in ((E6_FULL, FULL, "vs_e466"), (E7_FULL, FULL, "vs_e467"),
                                      (OFF, FULL, "residual"), (ANON, FULL, "medical")):
                a, b = rec[f"prediction_{left}"], rec[f"prediction_{right}"]
                rec.update({f"corrected_{name}": int(a != label and b == label),
                            f"harmed_{name}": int(a == label and b != label),
                            f"true_margin_change_{name}": rec[f"true_margin_{right}"] - rec[f"true_margin_{left}"]})
            result[fold].append(rec)
    local = result[FOLD]
    h.require([sum(row["fold8_cohort"] == k for row in local) for k in COHORTS] == [1, 5, 81],
              "Fold 8 patient cohort sizes differ")
    h.require(sum(len(rows) for rows in result.values()) == 871, "Ten-fold patient rows are incomplete")
    return result


def values(records, mode):
    return ([int(row["true_label"]) for row in records],
            [int(row[f"prediction_{mode}"]) for row in records],
            [float(row[f"p_asd_{mode}"]) for row in records])


def metric_rows(result, refs, history, audits):
    all_rows, totals, per_fold = [], {}, {}
    sample = next(iter(result.values()))
    tasks = [(method, "", mode, sample) for method, mode in HISTORICAL]
    tasks.extend((f"{METHOD}-{g}", g, mode, folds) for g, folds in result.items() for mode in INTERVENTIONS)
    for method, group, mode, folds in tasks:
        mode_name = mode if group else "full"
        record = {"method": method, "group": group, "intervention": mode_name}
        each = {}
        for fold, patients in folds.items():
            statistic = each[fold] = score(*values(patients, mode))
            local = fold == FOLD and bool(group)
            replay = history[str(fold)]["checkpoint"]
            source = {"checkpoint_source": " → ".join(replay["lineage"]),
                      "best_epoch": replay["best_epoch"], "latest_epoch": replay["latest_epoch"],
                      "best_sha256": replay["best_sha"]}
            if local:
                audit = audits[group]
                h.require(statistic == audit["best_metrics"][mode], f"{group} Fold 8 metrics differ from checkpoint replay")
                source = {"checkpoint_source": method, "best_epoch": audit["best_epoch"],
                          "latest_epoch": audit["latest_epoch"], "best_sha256": audit["artifact_sha256"]["best"]}
            elif method == E7 and fold == 2:
                source = refs
            elif mode in (FULL, E6_FULL, E7_FULL):
                h.require(statistic == history[str(fold)]["metrics"], f"Reused Fold {fold} metrics differ")
            all_rows.append({**record, "scope": "test_fold", "fold": fold, **source, **statistic})
        average = {key: float(np.mean([item[key] for item in each.values()])) for key in METRICS}
        deviation = {key: float(np.std([item[key] for item in each.values()], ddof=0)) for key in METRICS}
        pooled = score(*values(flatten(folds), mode))
        all_rows.extend([{**record, "scope": "macro_mean", **average},
                         {**record, "scope": "macro_population_sd", **deviation},
                         {**record, "scope": "pooled", **pooled}])
        if mode in (FULL, E6_FULL, E7_FULL):
            totals[method] = {"mean": average, "sd": deviation, "pooled": pooled}
            per_fold[method] = each
    return all_rows, totals, per_fold


def patient_ids(rows):
    return " ".join(str(row["patient_id_1based"]) for row in rows)


def pair_stats(left_rows, right_rows, left, right):
    y, a, p = values(left_rows, left)
    ly, b, q = values(right_rows, right)
    h.require(y == ly and [row["patient_id_1based"] for row in left_rows]
              == [row["patient_id_1based"] for row in right_rows], "Paired patient alignment differs")
    stats_a, stats_b = score(y, a, p), score(y, b, q)
    correction = [row for row, i, j, t in zip(right_rows, a, b, y) if i != t and j == t]
    harm = [row for row, i, j, t in zip(right_rows, a, b, y) if i == t and j != t]
    margin_delta = np.array([v[f"true_margin_{right}"] - u[f"true_margin_{left}"]
                             for u, v in zip(left_rows, right_rows)])
    wrong = np.asarray(a) != y
    wrong_delta = margin_delta[wrong]
    record = {
        "n_test": len(y), "n_asd": y.count(0), "n_td": y.count(1),
        "correct_before": stats_a["correct"], "correct_after": stats_b["correct"],
        "corrected": len(correction), "harmed": len(harm), "net": len(correction) - len(harm),
        "corrected_ids": patient_ids(correction), "harmed_ids": patient_ids(harm),
        "changed_hard_predictions": sum(x != z for x, z in zip(a, b)),
        "mean_p_asd_before": float(np.mean(p)), "mean_p_asd_after": float(np.mean(q)),
        "mean_p_asd_delta": float(np.mean(np.asarray(q) - np.asarray(p))),
        "mean_true_margin_before": float(np.mean([r[f"true_margin_{left}"] for r in left_rows])),
        "mean_true_margin_after": float(np.mean([r[f"true_margin_{right}"] for r in right_rows])),
        "mean_true_margin_delta": float(margin_delta.mean()), "baseline_wrong": int(wrong.sum()),
        "baseline_wrong_mean_true_margin_delta": float(wrong_delta.mean()) if wrong_delta.size else None,
        "baseline_wrong_margin_up": int((wrong_delta > 0).sum()),
        "baseline_wrong_margin_down": int((wrong_delta < 0).sum()),
        "baseline_wrong_margin_equal": int((wrong_delta == 0).sum()),
    }
    for tag, label in (("asd", 0), ("td", 1)):
        positive = [r for r in correction if r["true_label"] == label]
        negative = [r for r in harm if r["true_label"] == label]
        record.update({f"corrected_{tag}": len(positive), f"harmed_{tag}": len(negative),
                       f"corrected_{tag}_ids": patient_ids(positive), f"harmed_{tag}_ids": patient_ids(negative)})
    for key in METRICS:
        x, z = stats_a[key], stats_b[key]
        record.update({f"{key}_before_pct": 100 * x if x is not None else None,
                       f"{key}_after_pct": 100 * z if z is not None else None,
                       f"{key}_delta_pp": 100 * (z - x) if x is not None and z is not None else None})
    h.require(record["corrected"] + record["harmed"] == record["changed_hard_predictions"]
              and record["net"] == stats_b["correct"] - stats_a["correct"], "Paired counts are inconsistent")
    return record


def add_macro(record, before, after):
    record.update({f"macro_{key}_delta_pp": 100 * (after["mean"][key] - before["mean"][key])
                   for key in METRICS})
    record["pooled_prob_auc_delta_pp"] = 100 * (after["pooled"]["prob_auc"] - before["pooled"]["prob_auc"])


def comparisons(result, summary):
    table, selected = [], {}
    mode_tags = {"e467_to_e466_rollback": (E7_FULL, E6_FULL), "e466_to_new_full": (E6_FULL, FULL),
                 "e467_to_new_full": (E7_FULL, FULL), "cross_model_off_to_off": ("e466_residual_off", OFF),
                 "e466_full_to_new_off": (E6_FULL, OFF), "off_to_full": (OFF, FULL),
                 "off_to_anonymous": (OFF, ANON), "anonymous_to_medical": (ANON, FULL)}

    def add(group, comparison, scope, rows, right_rows=None, right_group=""):
        left, right = mode_tags.get(comparison, (FULL, FULL))
        rec = {"group": group, "right_group": right_group, "comparison": comparison, "scope": scope,
               "before_mode": left, "after_mode": right, **pair_stats(rows, rows if right_rows is None else right_rows, left, right)}
        table.append(rec)
        selected[(group, comparison, scope)] = rec
        return rec

    sample = next(iter(result.values()))
    add("", "e467_to_e466_rollback", "fold2", sample[2])
    rollback = add("", "e467_to_e466_rollback", "ten_fold", flatten(sample))
    add_macro(rollback, summary[E7], summary[E6])
    for group, folds in result.items():
        name = f"{METHOD}-{group}"
        for comparison, reference in (("e466_to_new_full", E6), ("e467_to_new_full", E7)):
            overall = add(group, comparison, "ten_fold", flatten(folds))
            add_macro(overall, summary[reference], summary[name])
        test = folds[FOLD]
        scopes = {"fold8": test, **{key: [row for row in test if row["fold8_cohort"] == key] for key in COHORTS}}
        scopes["unexposed_86"] = [row for row in test if row["fold8_cohort"] != "A"]
        scopes["unexposed_asd"] = [row for row in scopes["unexposed_86"] if row["true_label"] == 0]
        scopes["fold8_baseline_correct_asd_34"] = [
            row for row in test if row["true_label"] == row["prediction_e466_full"] == 0]
        scopes["fold8_td_47"] = [row for row in test if row["true_label"] == 1]
        h.require(len(scopes["fold8_baseline_correct_asd_34"]) == 34 and len(scopes["fold8_td_47"]) == 47,
                  "Fold 8 baseline cohorts are incomplete")
        for scope, rows in scopes.items():
            add(group, "e466_to_new_full", scope, rows)
            for mechanism in ("cross_model_off_to_off", "e466_full_to_new_off", "off_to_full",
                              "off_to_anonymous", "anonymous_to_medical"):
                add(group, mechanism, scope, rows)
        baseline8 = selected[(group, "e466_to_new_full", "fold8")]
        total6 = selected[(group, "e466_to_new_full", "ten_fold")]
        total7 = selected[(group, "e467_to_new_full", "ten_fold")]
        h.require(all(total6[k] == baseline8[k] and total7[k] == baseline8[k] + rollback[k]
                      for k in ("corrected", "harmed", "net")), "Fold 2 and Fold 8 attribution does not add up")
        h.require(all(abs(total7[f"macro_{k}_delta_pp"] - rollback[f"macro_{k}_delta_pp"]
                          - total6[f"macro_{k}_delta_pp"]) < 1e-10 for k in METRICS),
                  "Fold 2 and Fold 8 macro metric changes do not add up")
    for left, right in itertools.combinations(result, 2):
        add(left, "group_to_group", "fold8", result[left][FOLD], result[right][FOLD], right)
        row = add(left, "group_to_group", "ten_fold", flatten(result[left]), flatten(result[right]), right)
        add_macro(row, summary[f"{METHOD}-{left}"], summary[f"{METHOD}-{right}"])
    return table, selected


def count_text(row):
    return f"{row['corrected']}/{row['harmed']}/{row['net']:+d}"


def pct(value):
    return f"{100 * value:.4f}%"


def ids(row, key):
    return row[key].replace(" ", "、") or "—"


def diagnostic(row, mode):
    return f"{LABEL[row[f'prediction_{mode}']]} / {row[f'p_asd_{mode}']:.6f} / {row[f'true_margin_{mode}']:+.6f}"


def md_table(columns, rows):
    return "\n".join(["| " + " | ".join(columns) + " |",
                      "| " + " | ".join("---" for _ in columns) + " |",
                      *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)])


def build_report(manifest, audits, result, totals, per_fold, pairs, rollback):
    groups = list(result)
    label = lambda group: f"{group} · {GROUPS[group]['patient_id_1based']}"
    e6f8 = per_fold[E6][FOLD]
    sections = ["# E82.4.6.9 — Six Independent Fold-8 ASD Patient Additions",
                "## A. STATUS",
                f"**{manifest['status']}**。六组独立从历史 E61 Fold 8 epoch 10 的完整状态恢复，seed=0，正式训练从 epoch 11 开始。"
                f"GPU0 实测并发 {manifest['peak_formal_workers']}，采样峰值 {manifest['gpu_training_peak_mib']} MiB。",
                md_table(["组 · 目标患者", "组状态", "Best epoch", "Latest epoch", "终止条件"],
                         [(label(g), audits[g]["status"], audits[g]["best_epoch"], audits[g]["latest_epoch"],
                           audits[g]["journal"]["stop"]) for g in groups]),
                "全部使用历史 E82.4.6.4-G1 的 DARG-only 重叠监督。Checkpoint 内的基座 G2 标识 SCMR-DA-SRPE 结构，正式组号记录在 task_group 和文件名中。",
                "## B. 十折结果与历史比较",
                "ASD 为阳性。六项指标为十折宏平均 ± population SD，单位为 %。末列单列 871 人 pooled Prob-AUC。",
                md_table(["实验", *HEADINGS, "pooled Prob-AUC"],
                         [(method, *(f"{100 * totals[method]['mean'][key]:.4f} ± {100 * totals[method]['sd'][key]:.4f}" for key in METRICS),
                           pct(totals[method]["pooled"]["prob_auc"])) for method in (E6, E7, *(f"{METHOD}-{g}" for g in groups))]),
                "下表为相对 E82.4.6.6-G1 的变化。九折共 784 个测试结果逐患者相同，以下变化均来自 Fold 8。指标变化单位为百分点。",
                md_table(["组 · 目标患者", "纠正/误伤/净", *(f"Δ {key}" for key in HEADINGS), "Δ pooled Prob-AUC"],
                         [(label(g), count_text(row), *(f"{row[f'macro_{key}_delta_pp']:+.4f}" for key in METRICS),
                           f"{row['pooled_prob_auc_delta_pp']:+.4f}")
                          for g in groups for row in [pairs[g, "e466_to_new_full", "ten_fold"]]]),
                "E82.4.6.7-G1 → E82.4.6.6-G1 的 Fold 2 撤回事先独立回放。"
                "Best、Latest 和日志 SHA 核验通过。87 人的标签顺序、硬预测、ASD 概率和真类 Margin 在 Full、Residual-off、Anonymous 下逐值匹配。",
                f"Fold 2 权重链为 {' → '.join(rollback['checkpoint']['lineage'])}。DARG 人数 792，SCMR/Gaussian/EMA 各 784。"
                f"八名重叠患者为 {'、'.join(map(str, rollback['overlap_ids_1based']))}。"
                f"540 已退出分类与关系训练，联合残差 L2 恢复为 {rollback['patient_540_joint_residual_l2']:.8f}。"
                f"Fold 2 答对 {rollback['metrics']['correct']}/87，ACC {pct(rollback['metrics']['acc'])}，"
                f"BA {pct(rollback['metrics']['hard_auc'])}，Prob-AUC {pct(rollback['metrics']['prob_auc'])}。"
                f"撤回使患者 {ids(pairs['', 'e467_to_e466_rollback', 'fold2'], 'corrected_ids')} 纠正，"
                f"患者 {ids(pairs['', 'e467_to_e466_rollback', 'fold2'], 'harmed_ids')} 误伤，净 0。"
                f"折内 Prob-AUC 变化 {pairs['', 'e467_to_e466_rollback', 'fold2']['prob_auc_delta_pp']:+.4f} 个百分点。",
                "以 E82.4.6.7-G1 → E82.4.6.6-G1 → 本轮组为顺序。下表将 Fold 2 撤回与各组 Fold 8 变动分别列出，"
                "宏平均和 pooled 的分步变化分别在各自口径上重算。",
                md_table(["组 · 目标患者", "Fold 2 纠正/误伤/净", "Fold 8 纠正/误伤/净", "合计纠正/误伤/净",
                          "合计 Δ ACC", "合计 Δ BA", "合计 Δ 宏 Prob-AUC", "合计 Δ pooled Prob-AUC"],
                         [(label(g), count_text(pairs["", "e467_to_e466_rollback", "fold2"]),
                           count_text(pairs[g, "e466_to_new_full", "fold8"]), count_text(row),
                           *(f"{row[f'macro_{key}_delta_pp']:+.4f}" for key in ("acc", "hard_auc", "prob_auc")),
                           f"{row['pooled_prob_auc_delta_pp']:+.4f}")
                          for g in groups for row in [pairs[g, "e467_to_new_full", "ten_fold"]]]),
                "## C. Fold 8 结果",
                "原始 Fold 8 测试集 87 人，40 ASD、47 TD。历史 Best epoch 249 答对 77 人，其中 ASD 34 人、TD 43 人。",
                md_table(["实验", "答对/87", "ASD 答对/40", "TD 答对/47", *HEADINGS, "纠正/误伤/净"],
                         [(E6, e6f8["correct"], e6f8["tp"], e6f8["tn"], *(pct(e6f8[k]) for k in METRICS), "—"),
                          *[(label(g), f["correct"], f["tp"], f["tn"], *(pct(f[k]) for k in METRICS),
                             count_text(pairs[g, "e466_to_new_full", "fold8"]))
                            for g in groups for f in [per_fold[f"{METHOD}-{g}"][FOLD]]]]),
                "## D. 六名目标 ASD 患者",
                "每格为预测类别 / ASD 概率 / 真类 Margin。目标 ASD 的谱内、谱外和联合残差严格为零。",
                md_table(["组 · 目标患者", "原始 Full", "新 Residual-off", "新 Anonymous", "新 Full", "纠正"],
                         [(label(g), *(diagnostic(row, mode) for mode in (E6_FULL, OFF, ANON, FULL)),
                           "是" if row["corrected_vs_e466"] else "否")
                          for g in groups for row in result[g][FOLD] if row["fold8_cohort"] == "A"]),
                "## E. 其余五名未暴露候选 ASD",
                "每格为新 Full 硬预测 / 相对历史 Full 的真类 Margin 变化。目标患者在本组的变化见 D。",
                md_table(["组 · 目标患者", *(str(GROUPS[g]["patient_id_1based"]) for g in groups)],
                         [(label(g), *("本组目标" if g == other else
                                       f"{LABEL[row['prediction_full']]} / {row['true_margin_change_vs_e466']:+.4f}"
                                       for other in groups for row in result[g][FOLD]
                                       if row["patient_id_1based"] == GROUPS[other]["patient_id_1based"]))
                          for g in groups]),
                md_table(["组 · 目标患者", "其余五人纠正/误伤/净", "纠正患者", "误伤患者", "平均 Δ ASD 概率", "平均 Δ 真类 Margin"],
                         [(label(g), count_text(r), ids(r, "corrected_ids"), ids(r, "harmed_ids"),
                           f"{r['mean_p_asd_delta']:+.6f}", f"{r['mean_true_margin_delta']:+.6f}")
                          for g in groups for r in [pairs[g, "e466_to_new_full", "B"]]]),
                "## F. 未加入训练患者的纠错和误伤",
                "各组 86 名未加入训练者包含 39 名 ASD 与 47 名 TD。原始判对的 34 名 ASD 全部属于该范围。",
                md_table(["组 · 目标患者", "86 人纠正/误伤/净", "ASD 纠正", "ASD 误伤", "TD 纠正", "TD 误伤",
                          "81 人队列纠正/误伤/净", "原判错 10 人真类 Margin 平均变化", "Margin 上升/下降/持平"],
                         [(label(g), count_text(u), *(ids(u, key) for key in ("corrected_asd_ids", "harmed_asd_ids",
                                                                            "corrected_td_ids", "harmed_td_ids")),
                           count_text(pairs[g, "e466_to_new_full", "C"]),
                           f"{f['baseline_wrong_mean_true_margin_delta']:+.6f}",
                           f"{f['baseline_wrong_margin_up']}/{f['baseline_wrong_margin_down']}/{f['baseline_wrong_margin_equal']}")
                          for g in groups for u, f in [(pairs[g, "e466_to_new_full", "unexposed_86"],
                                                        pairs[g, "e466_to_new_full", "fold8"])] ]),
                "## G. 谱路径和医学 Prompt",
                "跨模型 Residual-off 的变化包含共同训练参数的变化。同权重 Residual-off → Full 测量联合残差直接作用，"
                "Anonymous → Full 测量以匿名 Query 为参照的医学 Query 作用。结果为固定权重的干预和描述性比较。",
                md_table(["组 · 目标患者", "跨模型 Off 纠正/误伤/净", "同权重 Off → Full 纠正/误伤/净",
                          "Δ Prob-AUC pp", "Δ 真类 Margin", "Anonymous → Full 纠正/误伤/净",
                          "Δ Prob-AUC pp", "Δ 真类 Margin", "Off 已纠正且 Full 保持的患者"],
                         [(label(g), count_text(cross), count_text(off), f"{off['prob_auc_delta_pp']:+.4f}",
                           f"{off['mean_true_margin_delta']:+.6f}", count_text(query), f"{query['prob_auc_delta_pp']:+.4f}",
                           f"{query['mean_true_margin_delta']:+.6f}", patient_ids(stable) or "—")
                          for g in groups
                          for cross, off, query in [(pairs[g, "cross_model_off_to_off", "fold8"], pairs[g, "off_to_full", "fold8"],
                                                     pairs[g, "anonymous_to_medical", "fold8"])]
                          for stable in [[r for r in result[g][FOLD]
                                          if r["prediction_e466_full"] != r["true_label"]
                                          and r["prediction_residual_off"] == r["prediction_full"] == r["true_label"]]]]),
                md_table(["组 · 目标患者", "86 人谱内 L2 均值", "86 人谱外 L2 均值", "86 人联合 L2 均值",
                          "86 人医学 Query 特征变化 L2 均值", "最大测试残差比例", "最大全程共享差分梯度比例"],
                         [(label(g), *(f"{np.mean([r[k] for r in result[g][FOLD] if r['fold8_cohort'] != 'A']):.6f}"
                                       for k in ("delta_parallel_l2", "delta_perp_l2", "delta_l2", "query_feature_change_l2")),
                           f"{100 * audits[g]['max_residual_ratio']:.6f}%",
                           f"{100 * max(audits[g]['journal']['budget_maximum_ratio'][family + '.full'] for family in ('gfr', 'darg', 'cam', 'head')):.6f}%")
                          for g in groups]),
                "六个目标的 Full、Off、Anonymous logits、概率及分类逐值相同。其余 86 人的谱内、谱外和联合残差均活跃，"
                "合法训练者的谱内、谱外与医学 Prompt 均具有端到端梯度。",
                "## H. 关系对覆盖与独立审计",
                "每组 DARG 训练 785 人，SCMR/Gaussian/EMA 各 784 人。历史 E61 采样器生成 4,096 对，"
                "同类与异类各 2,048 对，采样前后全局 RNG 一致。下表的关系梯度取 Best 的目标患者四模态端点 embedding 梯度的联合 L2。",
                md_table(["组 · 目标患者", "成为端点", "未覆盖", "目标端点次数", "其中同类/异类", "分类 CAM144 梯度 L2",
                          "直接关系 BCE 梯度 L2", "目标纠正"],
                         [(label(g), m["covered_rows"], m["uncovered_rows"], m["target_pair_endpoint_occurrences"],
                           f"{m['target_pair_same_occurrences']}/{m['target_pair_different_occurrences']}",
                           f"{v['classification_target_cam144_grad_l2']:.4e}",
                           f"{np.linalg.norm(v['relation_target_pair_bce_embedding_grad_l2_by_modality']):.4e}",
                           "是" if pairs[g, "e466_to_new_full", "A"]["corrected"] else "否")
                          for g in groups for m, v in [(audits[g]["membership"], audits[g]["best_gradients"])]]),
                *[f"- {g} 未覆盖患者 {', '.join(map(str, audits[g]['membership']['uncovered_ids_1based']))}。" for g in groups],
                "采样次数和纠错按组并列。每组仅有一个目标，六组观察用于描述。",
                md_table(["组 · 目标患者", "逐轮 RNG 回放", "与历史重叠轮次", "历史 RNG 三阶段比对", "EMA 逐位一致",
                          "Best/Latest 重载", "Latest 下一步训练/评估"],
                         [(label(g), j["replayed_epochs"], j["archived_epochs"], j["rng_phase_comparisons"],
                           j["ema_bitwise_epochs"], "PASS", "逐值一致")
                          for g in groups for j in [audits[g]["journal"]]]),
                "每轮六次历史 Dropout 顺序逐轮核对，评估和 EMA 更新保持 RNG。Gaussian 独立重建、"
                "8% CAM144 残差限幅、0.1% 共享差分梯度预算以及六组训练和评估掩码均通过审计。",
                "共享来源核验在任何正式 GPU worker 启动前完成。十折历史 Best 共 871 行独立回放。"
                "Fold 3 继续使用 E82.4.6.3 的 DARG-only 权重，Fold 4 使用 E82.4.6.6-G1 的审计权重。"
                "其余九折在六组之间逐患者一致。",
                "## I. 实验目录、Manifest 和 Handoff",
                f"权威目录 `{ROOT}`。OSSFS 首选路径返回 errno 5。"
                "统一 manifest.json 记录来源、训练前冻结的源码与配置、六组工件 SHA、逐折复用和 Handoff 追加证明。",
                "results/patient_level.csv 包含 5,226 行，每组 871 人。fold_metrics.csv 分列十折宏平均、population SD 与 871 人 pooled 指标。"
                "paired_comparisons.csv 保留两个历史基线、Fold 2 撤回、三种推理干预、患者分层和六组两两变化。"
                "membership_audit.json 和 audit_status.json 索引六组独立证书。",
                "科研判定 **NO-GO**（独立泛化及 MICCAI 创新模块正式结论）。"
                "本轮是预注册的有意训练—测试标签重叠机制诊断，沿用 val=test 选 Best、CV 前 RFE 和预处理以及转导图。"
                "同权重干预仅刻画该检查点的推理贡献。"]
    return "\n\n".join(sections) + "\n"


def verify_reuse(result, old, history):
    first = next(iter(result.values()))
    evidence = {}
    fields = ("patient_id_1based", "patient_index_0based", "true_label", *FIELDS, *CAM_FIELDS, "best_sha256")
    for fold in FOLDS:
        if fold == FOLD:
            continue
        expected = [tuple((int if k.startswith("prediction") or k in fields[:3] else float)(r[k])
                          if k != "best_sha256" else r["current_best_sha"] for k in fields)
                    for r in old["e466"][fold]]
        for name, current in result.items():
            observed = [tuple(row[k] for k in fields) for row in current[fold]]
            h.require(observed == expected, f"{name} reused Fold {fold} patient data differs from E466")
        evidence[str(fold)] = {"test_rows": len(first[fold]), "checkpoint": history[str(fold)]["checkpoint"],
                               "bitwise_values_e466": True, "identical_all_six_groups": True,
                               "projection_sha256": h.sha(expected)}
    h.require(sum(r["test_rows"] for r in evidence.values()) == 784, "Nine-fold reused cohort differs")
    return evidence


def common_rng(manifest):
    def track(group):
        with paths(group)["journal"].open() as stream:
            return [tuple(r["rng"][key] for key in ("rng_before", "rng_after_train", "rng_after_eval"))
                    for r in map(json.loads, stream)]

    tracks = {g: track(g) for g in GROUPS}
    longest = max(tracks.values(), key=len)
    for group, trace in tracks.items():
        h.require(trace == longest[:len(trace)], f"{group} common RNG trajectory differs")
        h.require(manifest["groups"][group]["training"]["epoch"] == 10 + len(trace), "Training epoch count changed")
    return {"status": "PASS", "earliest_terminal_epoch": 10 + min(map(len, tracks.values())),
            "longest_terminal_epoch": 10 + len(longest), "epochs_compared_by_group": {g: len(v) for g, v in tracks.items()},
            "rng_sha256_by_group": {g: h.sha(v) for g, v in tracks.items()}}


def prepare_outputs(manifest, audits):
    old = archived()
    verify_archived_metrics(old)
    history = h.read_json(HISTORY_REPLAY)
    rollback = h.read_json(FOLD2_PROOF)
    e7_result = h.read_json(SOURCES["e467"] / "manifest.json")["fold2_result"]
    refs = {"checkpoint_source": E7, "best_epoch": e7_result["best_epoch"],
            "latest_epoch": e7_result["latest_epoch"], "best_sha256": e7_result["best_sha"]}
    result = {group: build_patients(group, old, history, audits[group]) for group in GROUPS}
    reuse = verify_reuse(result, old, history)
    fold_rows, totals, per_fold = metric_rows(result, refs, history, audits)
    paired_rows, pairs = comparisons(result, totals)
    h.require(rollback["metrics"] == per_fold[E6][2], "Replayed Fold 2 report metrics differ")

    membership = {
        "status": "PASS", "experiment": CONFIG["experiment"],
        "label_encoding": CONFIG["label_encoding"], "fold8_original": history[str(FOLD)],
        "fold2_rollback": {key: val for key, val in rollback.items() if key != "predictions"},
        "nine_fold_reuse": reuse,
        "groups": {group: {"membership": audit["membership"], "gaussian": audit["gaussian"],
                           "preflight_gradients": h.read_json(paths(group)["preflight"])["gradients"],
                           "best_gradients": audit["best_gradients"], "latest_gradients": audit["latest_gradients"]}
                   for group, audit in audits.items()},
    }
    common = common_rng(manifest)
    success = all(audits[g]["status"] == "COMPLETE_AUDITED" for g in GROUPS)
    status = "COMPLETE_AUDITED" if success else "AUDITED_WITH_PROTOCOL_DEVIATION"
    audit_status = {
        "status": status, "experiment": CONFIG["experiment"], "scientific_decision": "NO_GO_GENERALIZATION_OR_MICCAI_MODULE",
        "independent_fold2_replay_before_formal_launch": manifest["historical_replay"]["fold2_completed_at_utc"] <
                                                         manifest["first_formal_launch_utc"],
        "fold2": membership["fold2_rollback"], "historical_protection": manifest["guard_after"],
        "common_rng": common, "reused_patient_rows_per_group": 784,
        "independent_patient_and_metrics_recalculation": True,
        "groups": {group: {**manifest["groups"][group]["audit"],
                           **{k: audit[k] for k in ("best_reloaded", "latest_reloaded", "latest_next_training_exact",
                                                    "latest_next_evaluation_exact", "relation_coverage", "best_epoch",
                                                    "latest_epoch", "artifact_sha256", "best_metrics", "max_residual_ratio")},
                           "replayed_epochs": audit["journal"]["replayed_epochs"],
                           "ema_bitwise_epochs": audit["journal"]["ema_bitwise_epochs"],
                           "rng_phase_comparisons": audit["journal"]["rng_phase_comparisons"],
                           "metrics": totals[f"{METHOD}-{group}"]}
                   for group, audit in audits.items()},
    }
    h.require(audit_status["independent_fold2_replay_before_formal_launch"], "Fold 2 audit timing differs")
    report = build_report({**manifest, "status": status}, audits, result, totals, per_fold, pairs, rollback)
    out = {"patient_level.csv": [row for group in GROUPS for row in flatten(result[group])],
           "fold_metrics.csv": fold_rows, "paired_comparisons.csv": paired_rows}
    for filename, rows in out.items():
        h.write_csv(RESULTS / filename, rows)
    h.write_json(RESULTS / "membership_audit.json", membership)
    h.write_json(RESULTS / "audit_status.json", audit_status)
    with h.archive.atomic_writer(RESULTS / "report.md", "w") as stream:
        stream.write(report)
    h.sync_dir(RESULTS)
    return status, report, totals, per_fold, pairs, reuse, common


def digest(data):
    return hashlib.sha256(data).hexdigest()


def append_handoff(manifest, report, artifacts):
    appendix = ("\n" + report.replace("# E82.4.6.9 — Six Independent Fold-8 ASD Patient Additions", MARKER, 1)
                + "\n### 审计文件 SHA-256\n\n"
                + "\n".join(f"- `{name}` `{sha}`" for name, sha in sorted(artifacts.items()))
                + "\n").encode()
    marker = MARKER.encode()
    with HANDOFF.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        before = stream.read()
        intent = manifest.get("handoff_intent")
        if intent is None:
            h.require(marker not in before, "Handoff already contains an unregistered E82.4.6.9 section")
            intent = {"path": str(HANDOFF), "prefix_bytes": len(before), "prefix_sha256": digest(before),
                      "appended_bytes": len(appendix), "appended_sha256": digest(appendix), "marker": MARKER}
            manifest["handoff_intent"] = intent
            h.write_json(MANIFEST, manifest)
            h.sync_dir(ROOT)
        offset = intent["prefix_bytes"]
        h.require(digest(before[:offset]) == intent["prefix_sha256"]
                  and digest(appendix) == intent["appended_sha256"], "Handoff prefix or intended section differs")
        tail = before[offset:]
        if len(tail) < len(appendix):
            h.require(appendix.startswith(tail), "Handoff has different bytes after the verified prefix")
            stream.write(appendix[len(tail):])
            stream.flush()
            os.fsync(stream.fileno())
        readback = HANDOFF.read_bytes()
        end = offset + len(appendix)
        h.require(digest(readback[:offset]) == intent["prefix_sha256"]
                  and digest(readback[offset:end]) == intent["appended_sha256"]
                  and readback[offset:end] == appendix, "Independent Handoff prefix or section readback differs")
    return {**intent, "historical_prefix_preserved": True, "verified_at_utc": utc_now(),
            "sha256_after_append": digest(readback)}


def verify_publication(manifest):
    for relative, expected in manifest["result_files"].items():
        h.require(h.sha_file(ROOT / relative) == expected, f"Published digest differs: {relative}")
    proof = manifest["handoff"]
    page = HANDOFF.read_bytes()
    o = proof["prefix_bytes"]
    h.require(digest(page[:o]) == proof["prefix_sha256"] and
              digest(page[o:o + proof["appended_bytes"]]) == proof["appended_sha256"],
              "Published Handoff readback differs")


def main():
    with (RUNTIME / "execute.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = h.read_json(MANIFEST)
        controller.require_preflight(manifest)
        if manifest.get("handoff"):
            verify_publication(manifest)
            print(f"{manifest['status']} historical prefix and results verified.", flush=True)
            return
        h.require(manifest["status"] in ("AUDITED", "PUBLISHING")
                  and all(group["status"] in COMPLETE for group in manifest["groups"].values()),
                  "Six independent certificates are required for final publication")
        audits = {}
        for group in GROUPS:
            entry = manifest["groups"][group]["audit"]
            p = paths(group)
            h.require(h.sha_file(p["audit"]) == entry["sha256"], f"{group} audit certificate changed")
            audit = audits[group] = h.read_json(p["audit"])
            h.require(audit["status"] == entry["status"] and audit["group"] == group
                      and all(h.sha_file(p[k]) == sha for k, sha in audit["artifact_sha256"].items()),
                      f"{group} audit artifact changed")
        guard = verify_guard(manifest["protected_sha256"])
        h.require(guard == manifest["guard_before"] == manifest["guard_prelaunch"] == manifest["guard_after"],
                  "Historical artifact verification stages disagree")
        status, report, totals, per_fold, pairs, reuse, rng = prepare_outputs(manifest, audits)
        result_names = ("report.md", "fold_metrics.csv", "paired_comparisons.csv", "patient_level.csv",
                        "membership_audit.json", "audit_status.json")
        artifacts = {f"results/{name}": h.sha_file(RESULTS / name) for name in result_names}
        manifest.update(
            status="PUBLISHING", result_files=artifacts, nine_fold_reuse=reuse, common_rng=rng,
            independent_metrics={"ten_fold": totals, "fold8": {k: v[FOLD] for k, v in per_fold.items()}},
            published_source_sha256={"publish.py": h.sha_file(ROOT / "publish.py")},
            scientific_decision="NO_GO_GENERALIZATION_OR_MICCAI_MODULE",
        )
        h.write_json(MANIFEST, manifest)
        manifest["handoff"] = append_handoff(manifest, report, artifacts)
        verify_publication(manifest)
        manifest.update(status=status, published_at_utc=utc_now())
        h.write_json(MANIFEST, manifest)
        h.sync_dir(ROOT)
        verify_publication(h.read_json(MANIFEST))
        print(f"{status}: results and Handoff appended and verified.", flush=True)


if __name__ == "__main__":
    main()
