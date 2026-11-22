"""Recalculate E82.4.6.8 results, verify their readback, and append the handoff."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev

import replay
import core
import execute

ROOT = replay.ROOT
RESULTS = ROOT / "results"
RUNTIME = ROOT / "runtime"
E6, E7, H = "E82.4.6.6-G1", "E82.4.6.7-G1", "E82.4.6.8-G1"
STATUS = "AUDITED_WITH_PROTOCOL_DEVIATION"
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
NAMES = ("ACC", "BA", "Prob-AUC", "SEN-ASD", "SPE-TD", "ASD-F1")
SOURCES = {E6: replay.E466, E7: replay.E467}
REUSE_E7 = (1, 4, 5, 6, 7, 8, 9, 10)
REUSE = (1, 2, 4, 5, 6, 7, 8, 9, 10)
PREDICTION_FIELDS = ("prediction", "p_asd", "true_margin")
VARIANTS = ("full", "residual_off", "anonymous")
REASONS = (
    "The historical deterministic relation sampler accepted all 790 members but selected "
    "only 773 unique endpoints. Seventeen members, including patient 388, had no direct endpoint "
    "relation supervision. All 790 participated in DARG classification.",
    "The Fold 2 source manifests and digests were frozen before training. Independent "
    "checkpoint-to-prediction evaluation of the rollback was performed after the Fold 3 run.",
)


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return core.h.sha_file(path)


def read_json(path):
    return json.loads(path.read_bytes())


def write(path, content):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content.encode("utf-8"))
    os.replace(tmp, path)


def write_json(path, value):
    write(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    writer.writerows(rows)
    write(path, stream.getvalue())


def index(row):
    return int(row["fold"]), int(row["patient_id_1based"])


def matches(row, source, field):
    if field.endswith("_sha") or field in ("patient_id_1based", "patient_index_0based", "true_label"):
        return str(row[field]) == str(source[field])
    return float(row[field]) == float(source[field])


def score(rows, prefix):
    # Pair-order AUC with average tie credit, independent of the inference runtime.
    positive, negative = [], []
    tp = fp = tn = fn = 0
    for row in rows:
        label = int(row["true_label"])
        pred = int(row[f"prediction_{prefix}"])
        prob = float(row[f"p_asd_{prefix}"])
        check(label in (0, 1) and pred in (0, 1) and 0 <= prob <= 1, "Invalid prediction")
        if label == 0:
            positive.append(prob)
            tp += pred == 0
            fn += pred == 1
        else:
            negative.append(prob)
            fp += pred == 0
            tn += pred == 1
    sen = tp / len(positive) if positive else None
    spe = tn / len(negative) if negative else None
    f1_den = 2 * tp + fp + fn
    return {
        "n_test": len(rows), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "acc": (tp + tn) / len(rows),
        "hard_auc": (sen + spe) / 2 if sen is not None and spe is not None else None,
        "prob_auc": sum(2 * (a > b) + (a == b) for a in positive for b in negative)
        / (2 * len(positive) * len(negative)) if positive and negative else None,
        "sen": sen, "spe": spe, "asd_f1": 2 * tp / f1_den if f1_den else None,
    }


def verify_metrics(actual, reference):
    for key in METRICS:
        check(math.isclose(actual[key], float(reference[key]), rel_tol=0, abs_tol=3e-16),
              f"Recalculated {key} differs. {actual[key]} vs {reference[key]}")


def checkpoint_refs(sources):
    inherited = sources["checkpoint_sources"]
    refs = {str(fold): inherited[str(fold)] for fold in REUSE if fold != 4}
    for directory, key, stem in ((replay.E466, "fold4_result", "G1_f04"),
                                 (replay.E467, "fold2_result", "G1_f02")):
        source = read_json(directory / "manifest.json")[key]
        item = dict(source)
        item["source_manifest"] = str(directory / "manifest.json")
        item["source_manifest_sha"] = sha(directory / "manifest.json")
        for name, relative in (("best", f"checkpoints/{stem}.best.pt"),
                               ("latest", f"checkpoints/{stem}.latest.pt"),
                               ("journal", f"runtime/{stem}.jsonl")):
            path = directory / relative
            check(sha(path) == item[f"{name}_sha"], f"Checkpoint chain differs at {path}")
            item[name] = str(path)
        refs["4" if key == "fold4_result" else "reference_e8267_fold2"] = item
    refs["reference_fold3_e8263_darg_only"] = inherited["3"]
    return refs


def patient_rows(rep, scope, checkpoints):
    archived = {name: read_csv(path / "results/patient_level.csv") for name, path in SOURCES.items()}
    each = {name: {index(row): row for row in rows} for name, rows in archived.items()}
    rows = rep["patient_rows"]
    check(len(rows) == 871 and len({index(r) for r in rows}) == 871, "Patient rows incomplete")
    for row in rows:
        fold, pid = index(row)
        before = each[E7][fold, pid]
        baseline = each[E6][fold, pid]
        for field in ("patient_index_0based", "true_label"):
            check(matches(row, before, field) and matches(row, baseline, field), "Patient label or index differs")
        if fold != 3:
            source = baseline if fold == 2 else before
            for variant in VARIANTS:
                for field in PREDICTION_FIELDS:
                    key = f"{field}_{variant}"
                    check(matches(row, source, key), f"Reused value differs {fold} {pid} {key}")
            expected_sha = checkpoints[str(fold)]["best_sha"]
            check(row["current_best_sha"] == source["current_best_sha"] == expected_sha,
                  f"Reused Best identity differs {fold} {pid}")
        overlap = row["current_native_training_overlap"]
        row.pop("scmr_supervised_test", None)
        row["darg_classification_supervised"] = overlap
        row["darg_relation_eligible"] = overlap
        row["darg_relation_pair_rows_fold3"] = (
            scope["overlap_gradient"][str(pid)]["pair_rows"] if fold == 3 and overlap
            else 0 if fold == 3 else None
        )
        row["scmr_legal_supervision_member"] = False
        row["gaussian_legal_member"] = False
        row["ema_legal_member"] = False
        row["new_module_residual_inference_only"] = not overlap
        row["new_residual_masked"] = overlap
        for field in PREDICTION_FIELDS:
            value = before[f"{field}_residual_off"]
            row[f"{field}_previous_residual_off"] = int(value) if field == "prediction" else float(value)
    for fold in REUSE:
        name = E6 if fold == 2 else E7
        check([index(r) for r in rows if r["fold"] == fold] ==
              [index(r) for r in archived[name] if int(r["fold"]) == fold],
              f"Reused test order differs Fold {fold}")
    fold2 = [r for r in rows if r["fold"] == 2]
    fold3 = [r for r in rows if r["fold"] == 3]
    fold4 = [r for r in rows if r["fold"] == 4]
    check([r["patient_id_1based"] for r in fold4 if r["new_residual_masked"]] == [712, 713, 727, 773],
          "Fold 4 registration differs")
    check(len(fold3) == len(fold2) == 87, "Fold 2/3 test count differs")
    active = [r for r in fold3 if r["new_module_residual_inference_only"]]
    masked = [r for r in fold3 if r["new_residual_masked"]]
    check(len(active) == 81 and [r["patient_id_1based"] for r in masked] == [46, 66, 172, 266, 388, 413],
          "Fold 3 patient scope differs")
    for field in ("delta_l2", "delta_parallel_l2", "delta_perp_l2", "query_feature_change_l2"):
        check(all(r[field] > 0 for r in active) and all(r[field] == 0 for r in masked), f"Fold 3 {field} differs")
    p540 = next(r for r in fold2 if r["patient_id_1based"] == 540)
    check(not p540["darg_classification_supervised"] and p540["new_module_residual_inference_only"]
          and p540["delta_l2"] > 0, "Restored patient 540 scope differs")
    check(all(r["patient_index_0based"] == r["patient_id_1based"] - 1 for r in rows), "Index base differs")
    return rows


def compute_folds(rows, rep, checkpoints):
    values = {}
    output = []
    prefixes = {E6: "e8266", E7: "previous", H: "full"}
    for method, historical in rep["metrics"].items():
        fold_values = {}
        for fold in range(1, 11):
            expected = historical[str(fold)]
            tested = [row for row in rows if row["fold"] == fold]
            actual = score(tested, prefixes[method]) if method in prefixes else expected
            verify_metrics(actual, expected)
            value = {"method": method, "scope": "test_fold", "fold": fold, "n_test": len(tested), **actual}
            if method == H:
                ref = rep["fold3"] if fold == 3 else checkpoints[str(fold)]
                origin = "E61 epoch 10 → E82.4.6.8-G1" if fold == 3 else E6 if fold == 2 else E7
                value.update(checkpoint_source=origin, best_sha=ref["best_sha"],
                             best_epoch=ref.get("best_epoch"), latest_epoch=ref.get("latest_epoch"))
            fold_values[fold] = value
            output.append(value)
        values[method] = fold_values
    summaries = {}
    for method, folds in values.items():
        summaries[method] = summary(folds.values())
        record = {"method": method, "scope": "ten_fold_macro", "n_test": 871}
        for key, data in summaries[method].items():
            record[key] = data["mean_pct"] / 100
            record[f"{key}_population_sd"] = data["sd_pct"] / 100
        output.append(record)
        if method in prefixes:
            output.append({"method": method, "scope": "pooled_871", **score(rows, prefixes[method])})
    return output, summaries


def summary(fold_scores):
    return {key: {"mean_pct": 100 * mean(row[key] for row in fold_scores),
                  "sd_pct": 100 * pstdev(row[key] for row in fold_scores)} for key in METRICS}


def pair_stats(rows, source, target):
    old, new = score(rows, source), score(rows, target)
    corrected = [r for r in rows if r["true_label"] != r[f"prediction_{source}"] and
                 r["true_label"] == r[f"prediction_{target}"]]
    harmed = [r for r in rows if r["true_label"] == r[f"prediction_{source}"] and
              r["true_label"] != r[f"prediction_{target}"]]
    result = {"n": len(rows), "corrected": len(corrected), "harmed": len(harmed),
              "net": len(corrected) - len(harmed),
              "corrected_ids": ";".join(f"{r['fold']}:{r['patient_id_1based']}" for r in corrected),
              "harmed_ids": ";".join(f"{r['fold']}:{r['patient_id_1based']}" for r in harmed)}
    for number, name in enumerate(("asd", "td")):
        c = sum(r["true_label"] == number for r in corrected)
        h = sum(r["true_label"] == number for r in harmed)
        result.update({f"{name}_corrected": c, f"{name}_harmed": h, f"{name}_net": c - h})
    for field in ("p_asd", "true_margin"):
        differences = [r[f"{field}_{target}"] - r[f"{field}_{source}"] for r in rows]
        result[f"{field}_mean_delta"] = mean(differences)
        result[f"{field}_changed_rows"] = sum(x != 0 for x in differences)
    for key in METRICS:
        result[f"delta_{key}_pp"] = (new[key] - old[key]) * 100 if None not in (new[key], old[key]) else None
    return result


def compute_pairs(rows):
    comparison = (
        (f"{E7} → {H}", "previous", "full"),
        (f"{E6} → {H}", "e8266", "full"),
        ("Residual-off → Full", "residual_off", "full"),
        ("Anonymous Query → Medical Query", "anonymous", "full"),
        (f"{E7} Residual-off → {H} Residual-off (Fold 3)", "previous_residual_off", "residual_off"),
    )
    result = []
    for label, source, target in comparison:
        folds = range(1, 11) if source != "previous_residual_off" else [3]
        output = []
        for fold in folds:
            group = [r for r in rows if r["fold"] == fold]
            value = pair_stats(group, source, target)
            value["fold_result"] = "win" if value["net"] > 0 else "loss" if value["net"] < 0 else "tie"
            output.append({"scope": f"fold{fold}", "estimate": "fold", **value})
        if len(output) == 10:
            pooled = pair_stats(rows, source, target)
            counts = {label: sum(x["fold_result"] == label for x in output) for label in ("win", "tie", "loss")}
            macro = dict(pooled)
            for key in METRICS:
                name = f"delta_{key}_pp"
                macro[name] = mean(r[name] for r in output)
            macro["delta_prob_auc_macro_pp"] = macro["delta_prob_auc_pp"]
            pooled["delta_prob_auc_pooled_pp"] = pooled["delta_prob_auc_pp"]
            output = [
                {"scope": "ten_fold", "estimate": "macro", **macro, **counts},
                {"scope": "ten_fold", "estimate": "pooled_871", **pooled, **counts},
                *output,
            ]
        for scope in ("fold3_new_addition_413", "fold3_prior_overlap_5", "fold3_unexposed_81"):
            cohort = [r for r in rows if r["cohort"] == scope]
            output.append({"scope": scope, "estimate": "cohort", **pair_stats(cohort, source, target)})
        result.extend({"comparison": label, "source": source, "target": target, **one} for one in output)
    return result


def pair_line(row):
    return " | ".join(f"{row[f'delta_{key}_pp']:+.4f}" for key in METRICS)


def cohort_line(row, name):
    return (f"| {name} | {row['n']} | {row['corrected']}/{row['harmed']}/{row['net']:+d} "
            f"| {row['asd_corrected']}/{row['asd_harmed']} | {row['td_corrected']}/{row['td_harmed']} "
            f"| {row['p_asd_mean_delta']:+.6f} | {row['true_margin_mean_delta']:+.6f} |")


def build_report(rows, folds, summaries, pairs, rep, fold2, scope):
    def find(source, selected_scope, estimate):
        return next(r for r in pairs if r["source"] == source and r["scope"] == selected_scope
                    and r["estimate"] == estimate)

    patient = {r["patient_id_1based"]: r for r in rows}
    f2 = next(r for r in folds if r["method"] == H and r["scope"] == "test_fold" and r["fold"] == 2)
    f3 = next(r for r in folds if r["method"] == H and r["scope"] == "test_fold" and r["fold"] == 3)
    pooled = {r["method"]: r["prob_auc"] for r in folds if r["scope"] == "pooled_871"}
    lines = [
        f"# {H} · Fold 2 撤回 540，Fold 3 加入 413", "", "## A. STATUS", "",
        f"**{STATUS}**。唯一正式训练为 seed 0、物理 GPU0 的 Fold 3。"
        "保留历史 E61 确定性关系采样器，790 人有候选资格，773 人成为关系对端点。"
        "包括 388 在内的 17 人无直接端点关系监督，790 人均有分类梯度。"
        "Fold 2 权重回放于正式训练后完成。", "",
        "## B. 十折性能", "",
        "单位为 %，十折宏平均 ± population SD，ASD 为阳性。Hard-AUC 使用 BA。", "",
        "| 实验 | " + " | ".join(NAMES) + " |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, stats in summaries.items():
        cells = [f"{stats[k]['mean_pct']:.4f} ± {stats[k]['sd_pct']:.4f}" for k in METRICS]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += [
        "", "871 人 pooled Prob-AUC 分别为 "
        + "、".join(f"{name} {pooled[name]:.4%}" for name in (E6, E7, H)) + "。", "",
        "下表为百分点变化。胜/平/负按各折 ACC 计算。", "",
        "| 比较 | " + " | ".join(NAMES) + " | pooled Prob-AUC Δ | 纠正/误伤/净纠错 | 胜/平/负 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source, name in (("previous", E7), ("e8266", E6)):
        p = find(source, "ten_fold", "macro")
        q = find(source, "ten_fold", "pooled_871")
        lines.append(f"| {name} → {H} | {pair_line(p)} | {q['delta_prob_auc_pp']:+.4f} | "
                     f"{p['corrected']}/{p['harmed']}/{p['net']:+d} | {p['win']}/{p['tie']}/{p['loss']} |")
    lines += [
        "", "## C. Fold 2 撤回 540", "",
        "Best epoch 166 权重来自 E82.4.6.6-G1 → E82.4.6.5-G1 → E82.4.6.4-G1。"
        "独立回放核对 87 人的顺序、真值、概率、硬预测和 Margin。"
        "Full、Residual-off、Anonymous 三种输出逐值相同。Best、Latest、训练日志 SHA 均一致。"
        "恢复 792 人 DARG、784 人 SCMR/Gaussian/EMA、8 人屏蔽。"
        f"患者 540 已退出分类和关系训练，联合残差 L2 恢复为 {fold2['patient540_joint_residual_l2']:.8f}。", "",
        "Fold 2 指标依次为 " + "、".join(f"{n} {f2[k]:.4%}" for n, k in zip(NAMES, METRICS)) + "。"
        "对 E82.4.6.7-G1，撤回贡献为纠正 18、误伤 540，净 0，"
        f"折内 Prob-AUC 变化 {find('previous', 'fold2', 'fold')['delta_prob_auc_pp']:+.4f} pp。", "",
        "## D. Fold 3 新增 413", "",
        f"Best 答对 {f3['tp'] + f3['tn']}/87，比两个历史参考各少 1 人。"
        "纠正 TD 患者 46、778，误伤 ASD 患者 370、388、491。"
        f"折内六项指标变化依次为 {pair_line(find('previous', 'fold3', 'fold'))} pp。"
        "相对于 E82.4.6.6-G1，其余九折逐患者一致。", "",
        "| 分层 | 人数 | 纠正/误伤/净纠错 | ASD 纠正/误伤 | TD 纠正/误伤 | 平均 ASD 概率 Δ | 平均真类 Margin Δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for suffix, title in (("new_addition_413", "A · 413"), ("prior_overlap_5", "B · 既往暴露 5 人"),
                          ("unexposed_81", "C · 未暴露 81 人")):
        lines.append(cohort_line(find("previous", f"fold3_{suffix}", "cohort"), title))
    lines += ["", "## E. 关键患者", "",
              "每个单元格为预测类别 / ASD 概率 / 真类 Margin。", "",
              "| 患者 | 真值 | E82.4.6.7 Full | E82.4.6.8 Residual-off | E82.4.6.8 Anonymous | E82.4.6.8 Full |",
              "|---:|---|---|---|---|---|"]
    names = ("ASD", "TD")
    for pid in (540, 18, 413, 607, 46, 388, 370, 491, 778):
        r = patient[pid]
        cells = [f"{names[r[f'prediction_{p}']]} / {r[f'p_asd_{p}']:.8f} / {r[f'true_margin_{p}']:+.8f}"
                 for p in ("previous", "residual_off", "anonymous", "full")]
        lines.append(f"| {pid} | {names[r['true_label']]} | " + " | ".join(cells) + " |")
    lines += ["", "## F. 谱与 Prompt 机制", ""]
    for prefix, title in (("previous_residual_off", "跨模型 Residual-off → Residual-off"),
                          ("residual_off", "同权重 Residual-off → Full"),
                          ("anonymous", "同权重 Anonymous → Medical Query")):
        p = find(prefix, "fold3", "fold")
        lines.append(f"- {title}。Fold 3 纠正/误伤/净纠错为 {p['corrected']}/{p['harmed']}/{p['net']:+d}，"
                     f"Prob-AUC Δ {p['delta_prob_auc_pp']:+.4f} pp，"
                     f"平均真类 Margin Δ {p['true_margin_mean_delta']:+.6f}。")
    r = patient[413]
    lines += [
        "",
        f"患者 413 的 Residual-off 概率 {r['p_asd_previous_residual_off']:.8f} → {r['p_asd_residual_off']:.8f}，"
        "三种新权重推理均判 TD。跨模型 off 差值反映共同训练参数变化，历史同权重干预可直接测量本次残差和 Query 效应。"
        "六名屏蔽患者得到逐元素相同的原始及增强 CAM144，81 名未暴露患者的两种谱残差及 Query 特征变化均非零。"
        "十折同权重 Residual-off → Full 的两次纠正发生在复用的 Fold 5 和 Fold 9，"
        "Anonymous → Medical Query 的一次纠正发生在复用的 Fold 9。两种干预均无硬预测误伤。", "",
        "## G. 独立审计与复用核验", "",
    ]
    f = rep["fold3"]
    j = f["certificate"]["journal"]
    budget = max(j["budget_maximum_ratio"][f"{x}.full"] for x in ("gfr", "cam", "darg", "head"))
    lines += [
        f"Best epoch {f['best_epoch']}，Latest epoch {f['latest_epoch']}，历史早停正常触发。"
        "Best 和 Latest 重载、Latest 下一步训练与评估逐值复现。"
        f"377 个 epoch 的六次 Dropout 顺序和 RNG 均可重放，其中 {j['archived_epochs']} 个 epoch "
        f"有历史对应记录，三阶段 {j['rng_phase_comparisons']} 次精确一致。"
        f"EMA 逐位一致 {j['ema_bitwise_epochs']} 轮。Gaussian 重建和监督掩码通过。"
        f"最大测试 CAM144 残差比例 {f['max_cam_relative_delta']:.6%}，"
        f"最大全程共享差分梯度比例 {budget:.6%}。", "",
        f"4096 对关系样本同类/异类各 2048，生成过程的全局 RNG 保持精确一致。"
        f"413 共 {scope['overlap_gradient']['413']['pair_rows']} 条关系样本，关系梯度非零。"
        "谱内、谱外、医学 Query 及合法 SCMR 梯度均非零，六名重叠患者的 SCMR 分类残差梯度为零。"
        "初始模型、优化器、Gaussian/EMA、RNG、转导输入和历史训练配置均经比对。"
        "Fold 3 历史来源 E82.4.6.3 的 G2_f03 使用 DARG-only 范围。", "",
        "Fold 1、4、5、6、7、8、9、10 共 697 个测试行逐值匹配 E82.4.6.7-G1。"
        "Fold 4 继续使用 712、713、727、773 的配置。784 个复用测试行具备可核验权重来源。", "",
        "## H. 文件、Manifest 与 Handoff", "",
        f"唯一路径 `{ROOT}`。首选 OSSFS 路径返回 errno 5。"
        "源码与配置按训练前冻结的 SHA 校验。manifest.json 记录输出、源码、运行工件和权重 SHA。"
        "患者级表将 DARG 分类监督、关系候选资格、端点采样、SCMR/Gaussian/EMA 合法成员和残差推理分开。"
        "最终 handoff 追加段覆盖本实验早先的状态与发布 SHA，追加前字节逐个保留。", "",
        "本实验属于有意标签暴露的机制诊断，沿用 val=test 选 Best、CV 前预处理和转导图。"
        "结果不支持独立泛化性能推论。", "",
    ]
    return "\n".join(lines)


def append_handoff(text, files):
    path = ROOT.parent / "handoff.md"
    marker = "### E82.4.6.8-G1 final audit and corrected publication"
    appendix = ("\n" + marker + "\n\n" + text
                + "\nResult SHA256\n\n" + "\n".join(f"- `{name}` `{value}`" for name, value in files.items())
                + "\n").encode()
    # Lock the handoff file itself and append bytes. The existing file is never rewritten.
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        before = stream.read()
        token = marker.encode()
        if token in before:
            start = before.index(token) - 1
            check(before[start:] == appendix, "Final handoff section already exists with different content")
            before = before[:start]
        else:
            stream.write(appendix)
            stream.flush()
            os.fsync(stream.fileno())
        readback = path.read_bytes()
        check(readback == before + appendix, "Handoff prefix or appended content mismatch")
    return {"path": str(path), "historical_prefix_preserved": True, "prefix_bytes": len(before),
            "prefix_sha256": hashlib.sha256(before).hexdigest(),
            "appended_sha256": hashlib.sha256(appendix).hexdigest(),
            "readback_sha": hashlib.sha256(readback).hexdigest(), "section_marker": marker}


def main():
    manifest = read_json(ROOT / "manifest.json")
    check(manifest["implementation"] == core.implementation() and manifest["frozen_before_training"],
          "Frozen training code changed")
    checked_guard = execute.verify_guard(manifest)
    check(checked_guard == manifest["guard_before"], "Historical source guard differs")
    parents = replay.verify_historical_sources()
    checkpoints = checkpoint_refs(parents)
    rep = read_json(RUNTIME / "replayed_folds.json")
    fold2 = read_json(RUNTIME / "fold2_reuse_verification.json")
    scope = read_json(RUNTIME / "fold3_scope_verification.json")
    check(all(x["status"] == "PASS" for x in (rep, fold2, scope)), "Independent audit did not pass")
    check(scope["relation_sampled_unique_endpoint_rows"] == 773 and 388 in scope["relation_unsampled_ids_1based"],
          "Unexpected deterministic relation sample")
    rows = patient_rows(rep, scope, checkpoints)
    fold_metrics, summaries = compute_folds(rows, rep, checkpoints)
    pairs = compute_pairs(rows)
    fold2_metrics = [r for r in fold_metrics if r["scope"] == "test_fold" and r["fold"] == 2]
    verify_metrics(next(r for r in fold2_metrics if r["method"] == H),
                   next(r for r in fold2_metrics if r["method"] == E6))
    fold2["metrics_exact_e8266"] = True
    fold2["replay_after_formal_fold3_training"] = True
    member = {
        "status": "PASS", "only_formally_trained_fold": 3,
        "patient_id_base": 1, "test_patient_flags_define_same_fold_membership": True,
        "fold2_rollback": fold2,
        "fold3": {**scope, "patient413_label": 0, "patient413_python_index": 412,
                  "append_order": [46, 66, 172, 266, 388, 413],
                  "excluded_e8264_additions": [674, 697, 778, 781],
                  "gaussian": rep["fold3"]["certificate"]["gaussian"],
                  "test_rows": 87, "unexposed_test_rows": 81, "scmr_legal": 784, "gaussian_legal": 784, "ema_legal": 784},
        "overlap_by_fold": {fold: sorted(ids) for fold, ids in replay.CURRENT_EXPOSURE.items()},
    }
    write_csv(RESULTS / "patient_level.csv", rows)
    write_csv(RESULTS / "fold_metrics.csv", fold_metrics)
    write_csv(RESULTS / "paired_comparisons.csv", pairs)
    write_json(RESULTS / "membership_audit.json", member)
    body = build_report(rows, fold_metrics, summaries, pairs, rep, fold2, scope)
    write(RESULTS / "report.md", body)
    read_rows = read_csv(RESULTS / "patient_level.csv")
    check([index(r) for r in read_rows] == [index(r) for r in rows], "Patient CSV order changed")
    for original, published in zip(rows, read_rows):
        check(all(str(v) == published[k] for k, v in original.items() if v is not None)
              and all(published[k] == "" for k, v in original.items() if v is None), "Patient CSV value changed")
    for r in read_csv(RESULTS / "fold_metrics.csv"):
        if r["method"] == H and r["scope"] == "test_fold":
            verify_metrics(score([x for x in read_rows if x["fold"] == r["fold"]], "full"), r)
    check(len(read_csv(RESULTS / "paired_comparisons.csv")) == len(pairs), "Paired CSV incomplete")
    result_files = {f"results/{name}": sha(RESULTS / name) for name in
                    ("report.md", "fold_metrics.csv", "paired_comparisons.csv", "patient_level.csv", "membership_audit.json")}
    handoff = append_handoff(body[body.index("## A. STATUS"):], result_files)
    audit = {
        "status": STATUS, "formal_task_status": "TRAINED", "independent_audit_status": "PASS",
        "retrained_fold": 3, "reused_folds": list(REUSE), "protocol_deviations": list(REASONS),
        "fold3": rep["fold3"], "fold3_scope": scope, "fold2_rollback": fold2,
        "other_reuse": {"direct_reference": E7, "folds": list(REUSE_E7), "test_rows": 697,
                        "order_labels_probabilities_predictions_and_margins_exact": True},
        "checkpoint_sources": checkpoints, "historical_guard": checked_guard,
        "ten_fold_summary": summaries, "patient_csv_recalculation": "PASS",
        "fold3_pair_endpoint_coverage_requirement": "UNMET_IF_EVERY_PATIENT_REQUIRES_A_DIRECT_ENDPOINT",
        "handoff": handoff, "result_files": result_files,
    }
    write_json(RESULTS / "audit_status.json", audit)
    result_files = {**result_files, "results/audit_status.json": sha(RESULTS / "audit_status.json")}
    scripts = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "audit").glob("*.py"))}
    artifacts = {str(p.relative_to(ROOT)): sha(p) for p in sorted(RUNTIME.glob("*.json*"))}
    manifest.pop("audit_entry_repair_pending", None)
    manifest.update(
        status=STATUS, actual_new_folds=[3], formal_training_task="G1_f03", completed_task_group="G1",
        protocol_deviations=list(REASONS), result_files=result_files, runtime_files=artifacts,
        reused_folds=list(REUSE), checkpoint_sources=checkpoints, audit_implementation_sha256=scripts,
        guard_after=checked_guard, handoff=handoff,
        audit={"status": "PASS", "result_status": STATUS, "audit_status_sha": result_files["results/audit_status.json"]},
    )
    for name, digest in {**result_files, **scripts, **artifacts}.items():
        check(sha(ROOT / name) == digest, f"Final artifact differs {name}")
    write_json(ROOT / "manifest.json", manifest)
    check(read_json(ROOT / "manifest.json") == manifest, "Manifest readback differs")
    check(sha(Path(handoff["path"])) == handoff["readback_sha"], "Handoff readback SHA differs")
    print(json.dumps({"status": STATUS, "manifest_sha256": sha(ROOT / "manifest.json"),
                      "handoff_sha256": handoff["readback_sha"], "readback": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
