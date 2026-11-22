"""Publish independent metrics, paired comparisons, scope evidence, and handoff."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

from replay import CURRENT, DIRECTORIES, ROOT, core, guards, h, load_manifests, np, verify_guard

RESULTS = ROOT / "results"
DARG, BASE, E463, E464, LOCAL = "E74.4-G2 DARG", "E82.4.6-G2", "E82.4.6.3", "E82.4.6.4-G1", "E82.4.6.5-G1"
METHODS = (DARG, BASE, E463, E464, LOCAL)
KEYS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
NAMES = ("ACC", "Hard-AUC/BA", "Prob-AUC", "SEN-ASD", "SPE-TD", "ASD-F1")
WITHDRAWN = frozenset((3, x) for x in (674, 697, 778, 781))
ADDED = frozenset((4, x) for x in (713, 727, 773))
EXPOSURE = frozenset((f, x + 1) for f, rows in core.EXPOSED.items() for x in rows)
OLD_EXPOSURE = (EXPOSURE - ADDED) | WITHDRAWN
UNCHANGED = (1, 2, 5, 6, 7, 8, 9, 10)
HANDOFF = ROOT.parent / "handoff.md"
HEADER = "## Experiment 82.4.6.5 — E82.4.6.4-G1 Fold-3-to-Fold-4 Patient Substitution"
FROZEN = "cc5ed0d34fa350104cf35e43a880cfbae56e72feac16dd6148d746e10c4df97b"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_archived(kind, filename, manifests):
    root = DIRECTORIES[kind]
    manifest = manifests[kind]
    expected = (manifest["result_sha256"][filename] if kind == "base"
                else manifest["result_files"]["results/" + filename])
    path = root / "results" / filename
    h.require(h.sha_file(path) == expected, f"Archived {kind}/{filename} digest differs")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def index(rows, id_column):
    rows = list(rows)
    keyed = {(int(row["fold"]), int(row[id_column])): row for row in rows}
    h.require(len(keyed) == 871 == len(rows) and len({i for _, i in keyed}) == 871, "Historical patient keys differ")
    return keyed


def entry(output, row, label):
    logits = output["logits"][row]
    return (int(output["prediction"][row]), float(output["probability"][row][0]),
            float((logits[0] - logits[1]) * (1 - 2 * label)))


def csv_entry(row, prediction, probability, margin):
    return (int(row[prediction]), float(row[probability]), float(row[margin]))


def patient_rows(replay, manifests):
    base = index((r for r in read_archived("base", "patient_margins.csv", manifests)
                  if r["group"] == "G2" and r["intervention"] == "full"), "patient_row_1based")
    e463 = index(read_archived("e463", "patient_level.csv", manifests), "patient_id_1based")
    e464 = index(read_archived("e464", "patient_level.csv", manifests), "patient_id_1based")
    h.require(base.keys() == e463.keys() == e464.keys(), "Archive patient identities differ")
    data, unchanged = [], {}
    for fold, d in sorted(replay["current"].items(), key=lambda pair: int(pair[0])):
        fold = int(fold)
        old = replay["e464_changed_folds"].get(str(fold), d)
        h.require(d["subject_id"] == old["subject_id"] and d["label"] == old["label"],
                  "Current and historical test order differs")
        count = 0
        for i, (identifier, label) in enumerate(zip(d["subject_id"], d["label"], strict=True)):
            key = (fold, identifier)
            label = int(label)
            b, t, q = base[key], e463[key], e464[key]
            h.require(label == int(b["label"]) == int(t["true_label"]) == int(q["true_label"]), "Label differs")
            darg = entry(d["darg"], i, label)
            h.require(darg[:2] == (int(b["pred_darg"]), float(b["p_asd_darg"])), "DARG probability differs from archive")
            original = csv_entry(b, "pred", "p_asd", "true_margin")
            e3 = csv_entry(t, "prediction_full", "p_asd_full", "true_margin_full")
            e4 = entry(old["full"], i, label)
            current = entry(d["full"], i, label)
            h.require(e4[:2] == (int(q["prediction_g1"]), float(q["p_asd_g1"])), "E82.4.6.4-G1 probability differs")
            if fold == 3:
                h.require(current == e3, "Reused E82.4.6.3 Fold 3 probability, prediction or margin differs")
            if fold in UNCHANGED:
                h.require(current == e4, "A purported unchanged fold differs from E82.4.6.4-G1")
                count += 1
            if fold not in (2, 3, 4):
                h.require(current == original, "E82.4.6-G2 reused fold differs")
            if fold == 4:
                h.require(e4 == original, "Original Fold 4 differs from E82.4.6-G2")
                h.require(label == 0 or key not in ADDED, "A Fold-4 training addition is not ASD")
            off = entry(d["off"], i, label)
            anonymous = entry(d["anonymous"], i, label)
            ratio = d["relative_residual"][i]
            residual = {k: d["residual_l2"][k][i] for k in ("delta", "delta_parallel", "delta_perp")}
            if key in EXPOSURE:
                h.require(identifier in d["masked_ids"] and ratio == 0 and
                          all(v == 0 for v in residual.values()) and off == current == anonymous,
                          f"Overlapping patient {identifier} residual is not exactly masked")
            else:
                h.require(identifier not in d["masked_ids"], f"Ordinary test patient {identifier} is masked")
            if key in WITHDRAWN:
                h.require(identifier in old["masked_ids"] and identifier not in d["masked_ids"],
                          f"Withdrawn patient {identifier} was not unmasked")
            category = ("fold3_withdrawn" if key in WITHDRAWN else "fold4_added" if key in ADDED else
                        "retained_overlap" if key in EXPOSURE else "outside_union_exposure")
            data.append({
                "fold": fold, "patient_id_1based": identifier, "patient_index_0based": identifier - 1,
                "true_label": label, "cohort": category, "current_native_training_overlap": key in EXPOSURE,
                "previous_native_training_overlap": key in OLD_EXPOSURE, "in_union_exposure": key in (EXPOSURE | OLD_EXPOSURE),
                "is_scmr_residual_masked": identifier in d["masked_ids"], "scmr_supervised_test": False,
                "darg": darg, "original_g2": original, "e82463": e3, "previous": e4,
                "full": current, "residual_off": off, "anonymous": anonymous,
                "previous_delta_l2": old["residual_l2"]["delta"][i],
                "previous_delta_parallel_l2": old["residual_l2"]["delta_parallel"][i],
                "previous_delta_perp_l2": old["residual_l2"]["delta_perp"][i],
                "delta_l2": residual["delta"], "delta_parallel_l2": residual["delta_parallel"],
                "delta_perp_l2": residual["delta_perp"], "cam_l2": d["cam_l2"][i], "cam_relative_delta": ratio,
                "query_feature_change_l2": d["query_feature_shift_l2"][i],
                "margin_gain_vs_previous": current[2] - e4[2],
                "margin_gain_off_to_full": current[2] - off[2], "margin_gain_anon_to_med": current[2] - anonymous[2],
                "corrected_vs_previous": e4[0] != label and current[0] == label,
                "harmed_vs_previous": e4[0] == label and current[0] != label,
                "residual_corrected": off[0] != label and current[0] == label,
                "residual_harmed": off[0] == label and current[0] != label,
                "medical_corrected": anonymous[0] != label and current[0] == label,
                "medical_harmed": anonymous[0] == label and current[0] != label,
                "current_best_sha": d["best_sha"], "previous_best_sha": old["best_sha"],
            })
        if fold in UNCHANGED:
            h.require(count == len(d["subject_id"]), "An unchanged fold did not pass")
            unchanged[str(fold)] = {"patients": count, "exact_logits_probability_and_prediction": True,
                                    "best_sha": d["best_sha"], "source_manifest_sha": d["source_manifest_sha"]}
    h.require(len(data) == 871 == len({(r["fold"], r["patient_id_1based"]) for r in data})
              and sum(r["in_union_exposure"] for r in data) == 20
              and sum(r["current_native_training_overlap"] for r in data) == 16,
              "Current or union exposure membership differs")
    return data, unchanged


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def measurements(labels, prediction, probability):
    asd, td = labels == 0, labels == 1
    tp = int(np.count_nonzero(prediction[asd] == 0))
    fn = int(asd.sum()) - tp
    fp = int(np.count_nonzero(prediction[td] == 0))
    tn = int(td.sum()) - fp
    sen, spe = ratio(tp, tp + fn), ratio(tn, tn + fp)
    auc = None
    if sen is not None and spe is not None:
        comparison = probability[asd, None] - probability[None, td]
        auc = float((np.count_nonzero(comparison > 0) + 0.5 * np.count_nonzero(comparison == 0)) / comparison.size)
    return {
        "n_test": len(labels), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "acc": ratio(tp + tn, len(labels)), "hard_auc": None if sen is None or spe is None else (sen + spe) / 2,
        "prob_auc": auc, "sen": sen, "spe": spe, "asd_f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def vectors(rows, key):
    labels = np.asarray([r["true_label"] for r in rows])
    v = np.asarray([r[key] for r in rows], dtype=np.float64)
    return labels, v[:, 0], v[:, 1], v[:, 2]


def values(rows, key):
    y, pred, p, _ = vectors(rows, key)
    return measurements(y, pred, p)


def compare_archived(actual, expected, description):
    for key in ("n_test", "tp", "fp", "tn", "fn"):
        h.require(actual[key] == int(expected[key]), f"{description} {key} differs")
    for key in KEYS:
        # Confusion ratios and pairwise ROC order are evaluated independently of
        # the source's ratio reduction and ROC integration summation order.
        h.require(abs(actual[key] - float(expected[key])) <= 8 * np.finfo(np.float64).eps,
                  f"{description} {key} exceeds floating-point rounding")


def fold_metrics(data, replay, manifests):
    archived = {}
    for kind in ("base", "e463", "e464"):
        historical = read_archived(kind, "fold_metrics.csv", manifests)
        archived[kind] = {(r["method"], int(r["fold"])): r for r in historical if r.get("scope", "all") == "all"}
    results = []
    sources = {
        DARG: ("darg", "base", "DARG"),
        BASE: ("original_g2", "base", "82.4.6-G2 SCMR-DA-SRPE"),
        E463: ("e82463", "e463", E463),
        E464: ("previous", "e464", E464),
        LOCAL: ("full", "local", LOCAL),
    }
    for fold in range(1, 11):
        selected = [r for r in data if r["fold"] == fold]
        current = replay["current"][str(fold)]
        base_model = replay["e464_changed_folds"].get(str(fold), current)
        for method, (column, kind, archive_key) in sources.items():
            actual = values(selected, column)
            checkpoint_sha = None
            end_epoch = None
            if kind == "local":
                epoch, end_epoch, checkpoint_sha = current["best_epoch"], current["latest_epoch"], current["best_sha"]
                source = current["source_kind"]
            else:
                stored = archived[kind][archive_key, fold]
                compare_archived(actual, stored, f"{archive_key} fold {fold}")
                source, epoch = kind, int(stored["best_epoch"])
                if method == E464:
                    epoch = base_model["best_epoch"]
                    checkpoint_sha = base_model["best_sha"]
                elif method == E463:
                    epoch = (current["best_epoch"] if fold == 3 else
                             int(stored["best_epoch"]) or base_model["best_epoch"])
                    checkpoint_sha = stored["best_sha"] if stored["best_sha"] != "archived" else None
                elif method == DARG:
                    checkpoint_sha = current["darg"]["sha"]
            row = {"method": method, "fold": fold, "scope": "test_fold", "best_epoch": epoch, "end_epoch": end_epoch,
                   "checkpoint_source": source, "best_sha": checkpoint_sha, **actual}
            row.update({key + "_percent": 100 * actual[key] for key in KEYS})
            results.append(row)
    return results


def fmt_value(value, multiplier=1.0):
    return "—" if value is None else f"{multiplier * value:.4f}"


def summaries(metric_rows):
    final = {}
    for method in METHODS:
        model = [row for row in metric_rows if row["method"] == method]
        final[method] = {key: {"mean_pct": float(np.mean([r[key] for r in model]) * 100),
                               "sd_pct": float(np.std([r[key] for r in model], ddof=0) * 100)} for key in KEYS}
    return final


def identity_string(rows, selection):
    return ",".join(f"{rows[i]['fold']}:{rows[i]['patient_id_1based']}" for i in np.flatnonzero(selection))


def delta(after, before):
    return None if before is None or after is None else 100 * (after - before)


def comparison(rows, before, after, scope, fold=None, macro=False):
    labels, pred0, p0, margin0 = vectors(rows, before)
    _, pred1, p1, margin1 = vectors(rows, after)
    wrong, right = pred0 != labels, pred0 == labels
    correct = wrong & (pred1 == labels)
    harm = right & (pred1 != labels)
    gain = margin1 - margin0
    dprob = p1 - p0
    result = {
        "source": before, "target": after, "scope": scope, "fold": fold, "n": len(rows),
        "corrected": int(correct.sum()), "harmed": int(harm.sum()), "net": int(correct.sum() - harm.sum()),
        "asd_corrected": int((correct & (labels == 0)).sum()), "asd_harmed": int((harm & (labels == 0)).sum()),
        "td_corrected": int((correct & (labels == 1)).sum()), "td_harmed": int((harm & (labels == 1)).sum()),
        "corrected_ids_fold_patient": identity_string(rows, correct),
        "harmed_ids_fold_patient": identity_string(rows, harm),
        "hard_flip": int(np.count_nonzero(pred0 != pred1)), "probability_changed": int(np.count_nonzero(p0 != p1)),
        "mean_abs_probability_change": float(np.mean(abs(dprob))),
        "mean_signed_probability_change": float(np.mean(dprob)), "mean_margin_change": float(gain.mean()),
        "margin_up": int((gain > 0).sum()), "margin_down": int((gain < 0).sum()),
        "baseline_wrong": int(wrong.sum()), "baseline_correct": int(right.sum()),
        "baseline_wrong_margin_up": int((wrong & (gain > 0)).sum()),
        "baseline_wrong_margin_down": int((wrong & (gain < 0)).sum()),
        "baseline_correct_margin_up": int((right & (gain > 0)).sum()),
        "baseline_correct_margin_down": int((right & (gain < 0)).sum()),
        "baseline_wrong_mean_margin_change": float(gain[wrong].mean()) if wrong.any() else None,
        "baseline_correct_mean_margin_change": float(gain[right].mean()) if right.any() else None,
    }
    if macro:
        subsets = [[r for r in rows if r["fold"] == f] for f in sorted({r["fold"] for r in rows})]
        metric_pairs = [(values(group, before), values(group, after)) for group in subsets]
        for key in KEYS:
            a, b = zip(*[(prior[key], later[key]) for prior, later in metric_pairs], strict=True)
            d = np.asarray(b) - np.asarray(a)
            result.update({key + "_before": float(np.mean(a)), key + "_after": float(np.mean(b)),
                           "delta_" + key + "_pp": float(100 * np.mean(d)),
                           "delta_" + key + "_sd_pp": float(100 * np.std(d, ddof=0)),
                           key + "_fold_wins": int((d > 0).sum()), key + "_fold_ties": int((d == 0).sum()),
                           key + "_fold_losses": int((d < 0).sum())})
    else:
        prior, later = measurements(labels, pred0, p0), measurements(labels, pred1, p1)
        for key in KEYS:
            result.update({key + "_before": prior[key], key + "_after": later[key],
                           "delta_" + key + "_pp": delta(later[key], prior[key])})
    return result


def paired(data):
    result = []
    specs = (("previous", "full"), ("darg", "full"), ("original_g2", "full"), ("e82463", "full"),
             ("residual_off", "full"), ("anonymous", "full"))
    filters = {
        "all_871_pooled": lambda x: True,
        "fold_3_4_pooled": lambda x: x["fold"] in (3, 4),
        "current_overlap_16": lambda x: x["current_native_training_overlap"],
        "retained_overlap_13": lambda x: x["cohort"] == "retained_overlap",
        "fold3_withdrawn_4": lambda x: x["cohort"] == "fold3_withdrawn",
        "fold4_added_3": lambda x: x["cohort"] == "fold4_added",
        "union_exposure_20": lambda x: x["in_union_exposure"],
        "outside_union_851": lambda x: not x["in_union_exposure"],
        "outside_union_changed_folds_162": lambda x: x["fold"] in (3, 4) and not x["in_union_exposure"],
    }
    for before, after in specs:
        result.append(comparison(data, before, after, "ten_fold_macro", macro=True))
        for fold in range(1, 11):
            result.append(comparison([r for r in data if r["fold"] == fold], before, after, "fold", fold))
        for name, select in filters.items():
            selected = [r for r in data if select(r)]
            result.append(comparison(selected, before, after, name))
    for subset in ("outside_union_851", "outside_union_changed_folds_162"):
        expected = int(subset.rsplit("_", 1)[-1])
        h.require(all(r["n"] == expected for r in result if r["scope"] == subset), "Union exposure count differs")
    return result


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    tmp.replace(path)


def atomic_text(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    tmp.replace(path)


def summary_row(method, summary):
    fields = [f"{summary[k]['mean_pct']:.4f} ± {summary[k]['sd_pct']:.4f}" for k in KEYS]
    return "| " + " | ".join([method, *fields]) + " |"


def selected(comparisons, before, scope, fold=None):
    return next(r for r in comparisons if r["source"] == before and r["target"] == "full"
                and r["scope"] == scope and r["fold"] == fold)


def pairing_table(pairs, before, scopes):
    lines = ["| 范围 | 人数 | 纠正 | 误伤 | 净纠错 | ΔACC (pp) | ΔBA (pp) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, scope, fold in scopes:
        r = selected(pairs, before, scope, fold)
        lines.append(f"| {label} | {r['n']} | {r['corrected']} | {r['harmed']} | {r['net']:+d} | "
                     f"{fmt_value(r['delta_acc_pp'])} | {fmt_value(r['delta_hard_auc_pp'])} |")
    return "\n".join(lines)


def report_text(audit, data, metrics, pairs, summary):
    source = audit["folds"]["4"]
    off = selected(pairs, "residual_off", "all_871_pooled")
    query = selected(pairs, "anonymous", "all_871_pooled")
    off_f4 = selected(pairs, "residual_off", "fold", 4)
    previous = selected(pairs, "previous", "ten_fold_macro")
    outside = selected(pairs, "previous", "outside_union_851")
    margins = selected(pairs, "residual_off", "fold_3_4_pooled")
    query_effect_count = sum(r["query_feature_change_l2"] > 0 for r in data)
    table = ["| 方法 | " + " | ".join(NAMES) + " |", "|---|" + "---:|" * len(KEYS)]
    table.extend(summary_row(method, summary[method]) for method in METHODS)
    per_fold = ["| Fold | 来源 | Best | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in metrics:
        if r["method"] == LOCAL and r["fold"] in (2, 3, 4):
            origin = {2: E464, 3: E463, 4: "本轮训练"}[r["fold"]]
            per_fold.append("| " + " | ".join([str(r["fold"]), origin, str(r["best_epoch"]),
                                                 *[fmt_value(r[k], 100) for k in KEYS]]) + " |")
    replaced = ["| Fold / 患者 | 标签 | 旧→新预测 | ASD 概率旧→新 | 真类 Margin 旧→新 | 联合残差 L2 旧→新 |",
                "|---|---|---|---|---|---|"]
    for row in data:
        if row["cohort"] not in ("fold3_withdrawn", "fold4_added"):
            continue
        old, now = row["previous"], row["full"]
        label = {0: "ASD", 1: "TD"}
        replaced.append(f"| {row['fold']} / {row['patient_id_1based']} | {label[row['true_label']]} | "
                        f"{label[old[0]]} → {label[now[0]]} | {old[1]:.6f} → {now[1]:.6f} | "
                        f"{old[2]:+.6f} → {now[2]:+.6f} | {row['previous_delta_l2']:.6f} → {row['delta_l2']:.6f} |")
    design = [
        "# Experiment 82.4.6.5", "",
        "**COMPLETE_AUDITED**", "",
        "本轮沿用 E82.4.6.4-G1 的 SCMR-DA-SRPE 与 DARG-only 监督，考察将 Fold 3 的四名重叠患者撤回、同时将 Fold 4 的三名 ASD 测试患者加入 DARG 训练的影响。训练故意暴露部分测试标签，结果用于机制诊断。", "",
        "## 成员与训练", "",
        "| Fold | DARG 训练 | SCMR 合法成员 | DARG 训练与测试交集，1-based | 复用或训练 |",
        "|---|---:|---:|---|---|",
        "| 2 | 792 | 784 | 377、407、681、779、784、260、288、375 | 复用 E82.4.6.4-G1 |",
        "| 3 | 789 | 784 | 46、66、172、266、388 | 精确复用 E82.4.6.3 |",
        "| 4 | 787 | 784 | 713、727、773 | 从各折 E61 epoch 10 基座恢复后训练 |", "",
        "Fold 3 的 674、697、778、781 均退出 DARG 分类和关系训练，保持在原测试集，并恢复正常 SCMR 推理。Fold 4 仅追加 713、727、773，原始标签均为 0。419 和 712 未被追加。其余七折继承 E82.4.6-G2。", "",
        "三折各自保留 4096 个关系训练对，同类、异类各 2048。当前 16 名重叠患者的标签仅用于 DARG 监督。SCMR 的监督和 Gaussian/EMA 成员仍为原 784 人，重叠行的谱内、谱外、联合残差及 Full 和 Residual-off 的输出逐元素一致。DARG 更新的共享表示仍可间接传递训练成员的影响。", "",
        "Fold 4 从第 11 轮训练至第 450 轮，第 267 轮首次取得最佳准确率 83/87，终止时 wait=183。保留同一原分类头、六次历史 Dropout、原有四个优化器的执行顺序、0.1% 共享差分梯度上限、8% CAM144 残差上限及评估后 EMA 更新。", "",
        "Fold 4 首次运行使用了与检查点保留字段重名的元数据键，导致恢复身份检验失败。改用独立元数据键后，从原 E61 状态重新执行该折并保留可恢复的提交。训练公式、模型和超参数沿用冻结配置。独立审计另修正掩码行索引，并在预检审计收据补记初始 EMA 摘要。两项审计修正及哈希见 manifest。", "",
        "## 十折结果", "",
        "分类指标以百分比报告，各折等权，标准差使用 population SD。AUC 使用连续 ASD 概率，Hard-AUC 等于硬分类的 Balanced Accuracy。指标由患者级预测及标签独立复算。", "",
        "\n".join(table), "",
        "相对 E82.4.6.4-G1 的十折均值变化（百分点）为 " +
        "、".join(f"{name} {previous['delta_' + key + '_pp']:+.4f}" for key, name in zip(KEYS, NAMES, strict=True)) + "。", "",
        "\n".join(per_fold), "",
        "## 患者级配对", "",
        "比较 E82.4.6.4-G1 → E82.4.6.5-G1。每个子集的 ACC 和 BA 使用该集合的患者直接计算，十折均值使用逐折宏平均。只有一类患者的子集缺少一侧灵敏度，BA 和 ROC-AUC 留空。", "",
        pairing_table(pairs, "previous", [
            ("十折", "ten_fold_macro", None), ("Fold 2", "fold", 2), ("Fold 3", "fold", 3), ("Fold 4", "fold", 4),
            ("本轮重叠 16 人", "current_overlap_16", None), ("两轮均重叠 13 人", "retained_overlap_13", None),
            ("Fold 3 撤回 4 人", "fold3_withdrawn_4", None), ("Fold 4 新加 3 人", "fold4_added_3", None),
            ("两轮联合暴露 20 人", "union_exposure_20", None), ("联合暴露之外 851 人", "outside_union_851", None),
            ("变化两折中联合暴露之外 162 人", "outside_union_changed_folds_162", None)]), "",
        f"纠正患者（折号:患者编号）为 {previous['corrected_ids_fold_patient'] or '无'}。误伤患者为 {previous['harmed_ids_fold_patient'] or '无'}。联合暴露集合之外纠正 {outside['corrected']} 人、误伤 {outside['harmed']} 人。", "",
        "\n".join(replaced), "",
        "## 同权重机制核验", "",
        "Residual-off → Full", "",
        pairing_table(pairs, "residual_off", [
            ("十折", "ten_fold_macro", None), ("Fold 2", "fold", 2), ("Fold 3", "fold", 3), ("Fold 4", "fold", 4),
            ("本轮重叠 16 人", "current_overlap_16", None), ("联合暴露之外 851 人", "outside_union_851", None)]), "",
        f"十折残差纠正 {off['corrected']} 人、误伤 {off['harmed']} 人。Fold 4 的直接纠正为 {off_f4['corrected_ids_fold_patient'] or '无'}，误伤为 {off_f4['harmed_ids_fold_patient'] or '无'}。"
        f"两变化折合计，Residual-off 的 {margins['baseline_wrong']} 名错判患者中有 {margins['baseline_wrong_margin_up']} 人真类 Margin 上升、{margins['baseline_wrong_margin_down']} 人下降。"
        f"原判对患者有 {margins['baseline_correct_margin_up']} 人上升、{margins['baseline_correct_margin_down']} 人下降。", "",
        "Anonymous Query → Medical Query", "",
        pairing_table(pairs, "anonymous", [
            ("十折", "ten_fold_macro", None), ("Fold 2", "fold", 2), ("Fold 3", "fold", 3), ("Fold 4", "fold", 4)]), "",
        f"医学 Query 改变 {query_effect_count} 人的 CAM144、{query['probability_changed']} 人的 ASD 概率和 {query['hard_flip']} 人的硬预测。"
        f"相对匹配匿名 Query 纠正 {query['corrected']} 人、误伤 {query['harmed']} 人，净 {query['net']:+d}。"
        f"校正相关的分类变化见配对表，概率和特征变化单独作为机制观测。", "",
        "## 独立审计", "",
        f"- 十折 Best、十折 Latest 和十折 Latest 下一步全部逐文件重载通过。Fold 4 复算了第 451 轮的训练和评估状态，用于恢复审计。",
        f"- Fold 3 的有序成员、标签、关系对、类别权重、谱基、Prompt、Gaussian 初始参考、EMA、困难记忆、模型、四个优化器和随机状态完全等价。第 11 轮损失、梯度踪迹、更新状态、患者输出和 RNG 与原归档提交逐项核对。",
        f"- Fold 4 重放 {source['rng_ledger']['replayed_epochs']} 轮随机状态，{source['rng_ledger']['archived_epochs']} 个可比历史 epoch 的三阶段合计 {source['rng_ledger']['rng_phase_comparisons']} 次 RNG 一致，独立复算的 EMA 状态逐轮精确。",
        f"- 本轮训练前后的保护集共 {audit['training_guard']['files']} 文件，历史清单扩展核验 {audit['archive_guard']['files']} 文件，SHA 通过。训练源码冻结 SHA 为 `{FROZEN}`。",
        f"- Fold 4 的历史共享梯度最大实际增量比为 {max(v for k, v in source['rng_ledger']['budget_maximum_ratio'].items() if k.endswith('.full')):.9f}，十折最大 CAM144 相对残差为 {max(r['cam_relative_delta'] for r in data):.8f}。Fold 4 另独立验证了两空间、Prompt 与端到端分类梯度。",
        f"- 其余八折相对 E82.4.6.4-G1 的患者概率、预测和指标逐项一致。Fold 4 的新增成员及其 DARG 分类、关系梯度和 SCMR 梯度边界见成员审计。",
        "", "## 解释限制", "",
        "十折仍保留单 seed=0、val=test 选择首次最高 ACC、CV 前 RFE 与历史预处理、871 节点转导图。16 名当前重叠患者的测试标签进入 DARG 训练。联合暴露集合之外的 851 人在两轮同折均未直接加入训练，结果仍受共享模型适配及历史选模影响。本轮用于标签暴露诊断，创新模块资格不由此实验判定。", "",
        "逐患者记录与复现身份见 [patient_level.csv](patient_level.csv)、[fold_metrics.csv](fold_metrics.csv)、[paired_comparisons.csv](paired_comparisons.csv)、"
        "[membership_audit.json](membership_audit.json)、[audit_status.json](audit_status.json) 和 [manifest.json](../manifest.json)。", "",
    ]
    return "\n".join(design)


def append_handoff(body):
    header = HEADER.encode()
    entry = ("\n\n" + HEADER + "\n\n" + body.rstrip() + "\n").encode("utf-8")
    with HANDOFF.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing = handle.read()
        h.require(existing, "Handoff history is empty")
        offset = existing.find(header)
        if offset >= 0:
            h.require(existing.count(header) == 1 and offset >= 2
                      and existing[offset - 2:offset - 2 + len(entry)] == entry,
                      "Existing 82.4.6.5 handoff entry differs")
            prefix = existing[:offset - 2]
            expected = existing
            added = False
        else:
            prefix = existing
            expected = prefix + entry
            prefix_sha = digest(prefix)
            count = handle.write(entry)
            h.require(count == len(entry), "Handoff append incomplete")
            handle.flush()
            os.fsync(handle.fileno())
            h.require(digest(prefix) == prefix_sha, "Handoff buffer changed")
            added = True
        # Check with a new read handle on disk, while the append lock is held.
        readback = HANDOFF.read_bytes()
        h.require(readback == expected and readback[:len(prefix)] == prefix and
                  readback[len(prefix):len(prefix) + len(entry)] == entry and readback.count(header) == 1,
                  "Handoff prefix or new section did not survive independent readback")
        return {"path": str(HANDOFF), "status": "PASS", "added_this_run": added, "prefix_bytes": len(prefix),
                "prefix_sha256": digest(prefix), "section_bytes": len(entry), "section_sha256": digest(entry),
                "readback_sha256": digest(readback), "historical_prefix_exact": True, "unique_section": True}


def main():
    manifests = load_manifests()
    manifest = manifests["local"]
    if manifest["status"] == "COMPLETE_AUDITED":
        h.require(all(h.sha_file(ROOT / path) == sha for path, sha in manifest["result_files"].items()),
                  "Completed result files changed")
        print("COMPLETE_AUDITED artifacts verified", flush=True)
        return
    h.require(manifest["status"] in ("FOLD4_AUDITED", "AUDIT_PASSED_PENDING_HANDOFF")
              and manifest["implementation_sha"] == FROZEN, "Unfrozen or incomplete training")
    replay = h.read_json(ROOT / "runtime/replayed_folds.json")
    h.require(replay["status"] == "PASS" and set(map(int, replay["current"])) == set(CURRENT),
              "Independent replay is incomplete")
    h.require(guards(manifests) == replay["archive_guard"] and verify_guard(manifest) == replay["train_guard"],
              "Historical file protections differ on publication")
    data, unchanged = patient_rows(replay, manifests)
    metrics = fold_metrics(data, replay, manifests)
    pairs = paired(data)
    summary = summaries(metrics)
    all_folds = {}
    for fold, item in replay["current"].items():
        keys = ("source_kind", "source_manifest", "source_manifest_sha", "source_implementation_sha", "best",
                "best_sha", "latest", "latest_sha", "journal", "journal_sha", "best_epoch", "latest_epoch",
                "best_state_sha", "latest_state_sha", "inference_sha", "preflight_sha", "preflight_audit_amendment",
                "rng_ledger", "geometry_audit", "scope_gradient", "best_gradient_budget")
        all_folds[fold] = {k: item[k] for k in keys}
        all_folds[fold]["latest_next_exact"] = item["latest_next"]["exact"]
        all_folds[fold]["latest_next_epoch"] = item["latest_epoch"] + 1
        all_folds[fold]["latest_next_witness_sha"] = h.sha(item["latest_next"]["next_witness"])
        all_folds[fold]["latest_next_gradient_diagnostic"] = item["latest_next"]["diagnostic"]
        all_folds[fold]["best_reloaded"] = all_folds[fold]["latest_reloaded"] = True
        all_folds[fold]["historical_baseline_darg_best"] = {k: item["darg"][k] for k in ("path", "sha")}
    auditor_sources = {str(p.relative_to(ROOT)): h.sha_file(p) for p in sorted((ROOT / "audit").glob("*.py"))}
    local = replay["current"]["4"]
    repaired = local["preflight_audit_amendment"]
    audit = {
        "status": "COMPLETE_AUDITED", "experiment": core.CONFIG["experiment"], "new_formal_folds": [4],
        "new_model_training_tasks": 1, "best_reloaded": 10, "latest_reloaded": 10, "latest_next_exact": 10,
        "folds": all_folds, "unchanged_vs_e82464_g1": unchanged, "fold3_reuse_exact": replay["fold3_equivalence"],
        "training_guard": replay["train_guard"], "archive_guard": replay["archive_guard"],
        "train_implementation_sha": manifest["implementation_sha"], "audit_source_sha256": auditor_sources,
        "historical_git_sources": replay["historical_git_sources"], "device": replay["device"],
        "actual_root": str(ROOT), "preferred_root_ossfs_errno": 5,
        "training_recovery": {
            "initial_fold4_attempt": "Checkpoint identity failed because native-membership metadata used the reserved membership key.",
            "repair": "Stored metadata under native_membership, then reran the same Fold-4 task from verified E61 epoch 10.",
            "model_loss_optimizer_configuration_unchanged": True,
        },
        "audit_entry_repair": {"masked_tensor_indexing_sha": auditor_sources["audit/audit_fix.py"],
                               "preflight_initial_ema": repaired},
        "test_label_exposure": {"current": 16, "two_round_union": 20, "outside_union": 851},
        "eligible_for_independent_generalization_or_innovation_go": False,
        "ten_fold_summary_pct": summary,
        "key_patient_comparisons": [r for r in pairs if r["scope"] in
            ("ten_fold_macro", "current_overlap_16", "fold3_withdrawn_4", "fold4_added_3", "outside_union_851")
            and r["source"] in ("previous", "residual_off", "anonymous")],
        "patient_recalculation": {"rows": len(data), "archived_metric_max_error_bound": float(8 * np.finfo(np.float64).eps),
                                  "full_test_identities_and_labels_exact": True},
    }
    membership = {
        "status": "PASS", "one_based_overlap": {str(f): [x + 1 for x in core.EXPOSED[f]] for f in (2, 3, 4)},
        "zero_based_overlap": {str(f): list(core.EXPOSED[f]) for f in (2, 3, 4)},
        "withdrawn_fold3_1based": sorted(i for _, i in WITHDRAWN), "added_fold4_1based": sorted(i for _, i in ADDED),
        "union_exposure_patients": [{"fold": f, "patient_id_1based": i} for f, i in sorted(EXPOSURE | OLD_EXPOSURE)],
        "excluded_outside_union": 851, "masked_prediction_exact_current_overlap": 16,
        "folds": {str(f): {
            "membership": replay["current"][str(f)]["membership_audit"],
            "gaussian": replay["current"][str(f)]["gaussian_audit"],
            "masked_test_labels": {str(r["patient_id_1based"]): r["true_label"] for r in data
                                   if r["fold"] == f and r["is_scmr_residual_masked"]},
        } for f in (2, 3, 4)},
    }
    exported = []
    for row in data:
        patient = {k: v for k, v in row.items() if not isinstance(v, tuple)}
        for model in ("darg", "original_g2", "e82463", "previous", "full", "residual_off", "anonymous"):
            p, prob, margin = row[model]
            patient.update({"prediction_" + model: p, "p_asd_" + model: prob, "true_margin_" + model: margin})
        exported.append(patient)
    write_csv(RESULTS / "fold_metrics.csv", metrics)
    write_csv(RESULTS / "paired_comparisons.csv", pairs)
    write_csv(RESULTS / "patient_level.csv", exported)
    h.write_json(RESULTS / "membership_audit.json", membership)

    report = report_text(audit, data, metrics, pairs, summary)
    versus = selected(pairs, "previous", "ten_fold_macro")
    off = selected(pairs, "residual_off", "ten_fold_macro")
    query = selected(pairs, "anonymous", "ten_fold_macro")
    outside = selected(pairs, "previous", "outside_union_851")
    body = "\n".join([
        "- 状态 **COMPLETE_AUDITED**。Fold 4 为唯一新训练折，E61 epoch 10 恢复，Best 第 267 轮（83/87），Latest 第 450 轮。Fold 2 逐患者继承 E82.4.6.4-G1，Fold 3 经成员、初始化、RNG 和第 11 轮更新等价核验后复用 E82.4.6.3，其余七折复用 E82.4.6-G2。",
        "- Fold 3 撤回 674、697、778、781，恢复正常 SCMR 推理。Fold 4 将 ASD 患者 713、727、773 加入 DARG，保留测试身份。Fold 2/3/4 DARG 成员 792/789/787，SCMR 各 784，当前重叠 16 人谱内、谱外和联合残差精确为零。",
        "- 十折指标（%）" + "，".join(f"{label} {summary[LOCAL][key]['mean_pct']:.4f} ± {summary[LOCAL][key]['sd_pct']:.4f}"
                                          for key, label in zip(KEYS, NAMES, strict=True)) + "。",
        f"- 相对 E82.4.6.4-G1，ACC {versus['delta_acc_pp']:+.4f} pp，纠正 {versus['corrected']}、误伤 {versus['harmed']}、净 {versus['net']:+d}。"
        f"联合暴露集合之外的 851 人纠正 {outside['corrected']}、误伤 {outside['harmed']}、净 {outside['net']:+d}。",
        f"- 固定权重开启残差纠正 {off['corrected']}、误伤 {off['harmed']}、净 {off['net']:+d}。医学 Query 相对匿名 Query 硬翻转 {query['hard_flip']} 人，净 {query['net']:+d}。",
        "- 10/10 Best、10/10 Latest 及 10/10 下一步恢复通过。Fold 4 的 440 轮 RNG、407 个历史可比 epoch、六次 Dropout、Gaussian/EMA、关系对及端到端预算通过。受保护历史文件 SHA 一致。",
        "- 本轮为有意训练—测试标签暴露机制诊断。单 seed、val=test、CV 前 RFE 和预处理及转导患者图共同限制解释。创新模块资格不由此实验判定。",
        f"- 权威目录 `{ROOT}`。冻结实施 SHA256 `{FROZEN}`。Fold 4 Best SHA256 `{local['best_sha']}`，Latest SHA256 `{local['latest_sha']}`。",
        f"- [报告]({RESULTS / 'report.md'}) SHA256 `{digest(report.encode('utf-8'))}`，"
        f"[患者级结果]({RESULTS / 'patient_level.csv'}) SHA256 `{h.sha_file(RESULTS / 'patient_level.csv')}`。",
    ])
    manifest["status"] = "AUDIT_PASSED_PENDING_HANDOFF"
    h.write_json(ROOT / "manifest.json", manifest)
    handoff = append_handoff(body)
    atomic_text(RESULTS / "report.md", report)
    audit["handoff"] = handoff
    h.write_json(RESULTS / "audit_status.json", audit)
    files = ("report.md", "fold_metrics.csv", "paired_comparisons.csv", "patient_level.csv",
             "membership_audit.json", "audit_status.json")
    result_sha = {"results/" + name: h.sha_file(RESULTS / name) for name in files}
    h.require(h.sha_file(RESULTS / "report.md") == digest(report.encode("utf-8")), "Report readback differs")
    manifest.pop("audit", None)
    manifest.update(status="COMPLETE_AUDITED", published_utc=datetime.now(timezone.utc).isoformat(),
                    authoritative_directory=str(ROOT), fallback={"used": True, "errno": 5},
                    actual_new_folds=[4], frozen_training_implementation_sha=FROZEN,
                    historical_git_sources=replay["historical_git_sources"], guard_after=replay["train_guard"],
                    archived_protected_sha=replay["archive_guard"], audit_entry_repair=audit["audit_entry_repair"],
                    audit_implementation_sha256=auditor_sources, result_files=result_sha, handoff=handoff,
                    checkpoint_sources={fold: {key: item[key] for key in
                        ("source_kind", "source_manifest", "source_manifest_sha", "source_implementation_sha",
                         "best", "best_sha", "latest", "latest_sha", "journal", "journal_sha", "best_epoch", "latest_epoch")}
                        for fold, item in all_folds.items()},
                    fold3_equivalence_sha=h.sha_file(ROOT / "runtime/fold03_reuse_equivalence.json"),
                    first_attempt_recovery=audit["training_recovery"], ten_fold_summary_pct=summary)
    h.write_json(ROOT / "manifest.json", manifest)
    h.require(h.read_json(ROOT / "manifest.json")["result_files"] == result_sha
              and h.read_json(RESULTS / "audit_status.json")["handoff"]["status"] == "PASS",
              "Final manifest/audit readback differs")
    print(summary_row(LOCAL, summary[LOCAL]), flush=True)
    print(f"COMPLETE_AUDITED | fold comparison {versus['net']:+d} | direct residual {off['net']:+d} | "
          f"medical query {query['net']:+d} | handoff prefix {handoff['prefix_sha256']}", flush=True)


if __name__ == "__main__":
    main()
