"""Publish paired patient and fold results after the independent audit."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import os
from pathlib import Path

import numpy as np

from independent import METRICS, SOURCE_METHOD, metric_values
from overlap import CONFIG, EXPOSED, FOLD, MANIFEST, RESULTS, ROOT, h

HANDOFF = Path("/model/spromptgl/experiment/handoff.md")
CURRENT = "82.4.6.1-G2 SCMR-DA-SRPE"
COLUMNS = ("ACC", "Hard-AUC / BA", "Prob-AUC", "SEN-ASD", "SPE-TD", "ASD-F1")
ABOUT = ("Fold 2 的两名重叠患者原已进入 DARG 历史训练集。本轮同时开放新增模块的监督、"
         "基于训练成员的 Gaussian 统计和 CAM144 残差。实验衡量标签暴露下的结果，无法支持独立泛化"
         "或 MICCAI 创新模块 GO。")


def write(path, data):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def write_csv(name, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    write(RESULTS / name, stream.getvalue().encode())


def evaluate(fold, condition, select):
    use = fold[condition]
    return metric_values(use["probability"][select], use["prediction"][select], fold["truth"][select])


def fold_row(fold, method, section, index, select):
    return {"method": method, "scope": section, "fold": index, "best_epoch": fold["best_epoch"],
            "source_checkpoint": fold["best_path"], "source_checkpoint_sha": fold["best_sha"],
            "reused_original_best": int(method == CURRENT and index != FOLD),
            **evaluate(fold, "full", select)}


def paired(before, after, fold, section, select, *, start="full", comparison="original_full_to_unmasked_full"):
    h.require(all(np.array_equal(before[key], after[key]) for key in ("truth", "subjects")),
              "Fold test labels or ordering differ")
    truth = before["truth"][select]
    prior = before[start]["prediction"][select]
    current = after["full"]["prediction"][select]
    fix, harm = (prior != truth) & (current == truth), (prior == truth) & (current != truth)
    b, a = evaluate(before, start, select), evaluate(after, "full", select)
    net = int(fix.sum() - harm.sum())
    h.require(net == a["tp"] + a["tn"] - b["tp"] - b["tn"], "Patient pairing disagrees with the confusions")
    return {"comparison": comparison, "scope": section, "fold": fold, "folds": 1, "n_test": int(len(truth)),
            "asd_corrected": int((fix & (truth == 0)).sum()), "asd_harmed": int((harm & (truth == 0)).sum()),
            "td_corrected": int((fix & (truth == 1)).sum()), "td_harmed": int((harm & (truth == 1)).sum()),
            "corrected": int(fix.sum()), "harmed": int(harm.sum()), "net": net,
            "wins": int(net > 0), "ties": int(net == 0), "losses": int(net < 0),
            **{f"{m}_before": b[m] for m in METRICS}, **{f"{m}_after": a[m] for m in METRICS},
            **{f"delta_{m}_pp": 100 * (a[m] - b[m]) for m in METRICS}}


def pooled(rows, scope):
    req = ("n_test", "asd_corrected", "asd_harmed", "td_corrected", "td_harmed", "corrected",
           "harmed", "net", "wins", "ties", "losses")
    ends = [f"{m}_{k}" for k in ("before", "after") for m in METRICS]
    result = {"comparison": rows[0]["comparison"], "scope": scope, "fold": "", "folds": len(rows),
              **{k: sum(r[k] for r in rows) for k in req},
              **{k: float(np.mean([r[k] for r in rows])) for k in ends}}
    result.update({f"delta_{m}_pp": 100 * (result[m + "_after"] - result[m + "_before"]) for m in METRICS})
    return result


def signed_margin(eval_result, i, y):
    logits = eval_result["logits"][i]
    return (1 - 2 * y) * (float(logits[0]) - float(logits[1]))


def patient_rows(number, old, new, gradients):
    h.require(np.array_equal(old["subjects"], new["subjects"]), "Patient identity mismatch")
    result = []
    for i, ident in enumerate(old["subjects"]):
        y = int(old["truth"][i])
        patient_id = int(ident)
        orig = int(old["full"]["prediction"][i])
        full = int(new["full"]["prediction"][i])
        off = int(new["off"]["prediction"][i])
        exposed = number == FOLD and patient_id - 1 in EXPOSED
        residual = new.get("residual", {})
        layer = {k: float(residual[k][i]) if k in residual else "" for k in
                 ("delta_l2", "delta_parallel_l2", "delta_perp_l2", "ratio")}
        old_margin = signed_margin(old["full"], i, y)
        off_margin = signed_margin(new["off"], i, y)
        full_margin = signed_margin(new["full"], i, y)
        grad = gradients[str(patient_id)]["best_supervision_gradient"] if exposed else {}
        result.append({"fold": number, "patient_id_1based": patient_id, "true_label": y,
                       "scope": ("reused_fold" if number != FOLD else "unmasked_pair" if exposed else "other_test"),
                       "prediction_original_g2": orig, "prediction_current_off": off, "prediction_current_full": full,
                       "p_asd_original_g2": float(old["full"]["probability"][i, 0]),
                       "p_asd_current_off": float(new["off"]["probability"][i, 0]),
                       "p_asd_current_full": float(new["full"]["probability"][i, 0]),
                       "original_to_current_flip": int(orig != full), "residual_flip": int(off != full),
                       "corrected_original_to_current": int(orig != y and full == y),
                       "harmed_original_to_current": int(orig == y and full != y),
                       "corrected_residual": int(off != y and full == y), "harmed_residual": int(off == y and full != y),
                       "true_margin_original_g2": old_margin, "true_margin_current_off": off_margin,
                       "true_margin_current_full": full_margin,
                       "true_margin_change_original_to_current": full_margin - old_margin,
                       "true_margin_change_off_to_full": full_margin - off_margin,
                       **layer, **{f"best_ce_gradient_{k}": grad.get(k, "") for k in
                                   ("full_ce_delta_l2", "prompt_l2", "control_l2", "spectral_candidate_l2",
                                    "complement_candidate_l2")},
                       "original_best_sha": old["best_sha"], "current_best_sha": new["best_sha"]})
    return result


def snapshot(historic, new, audit):
    fold_scores, pairs, patients, coverage = [], [], [], {}
    for fold, original in historic.items():
        current = new if fold == FOLD else original
        selected = np.ones(len(original["truth"]), dtype=bool)
        for method, value in ((SOURCE_METHOD, original), (CURRENT, current)):
            fold_scores.append(fold_row(value, method, "fold", fold, selected))
        pairs.append(paired(original, current, fold, "fold", selected))
        patients.extend(patient_rows(fold, original, current, audit["affected_patients"]))
        if fold != FOLD:
            coverage[str(fold)] = {key: original[key] for key in (
                "best_epoch", "best_sha", "best_path", "certificate_sha", "certificate_path", "inference_sha",
                "reference_sha", "partition_sha")}
            h.require(all(pairs[-1][f"delta_{m}_pp"] == 0 for m in METRICS) and pairs[-1]["net"] == 0,
                      "An unchanged fold differs")
            coverage[str(fold)]["prediction_probability_and_metrics_exact"] = True
    h.require(len(patients) == len({row["patient_id_1based"] for row in patients}) == 871, "Test CV coverage differs")
    pairs += [pooled(pairs, "ten_fold_mean"), pooled([r for r in pairs if r["fold"] != FOLD], "same_nine_fold_mean")]
    mask = np.isin(new["subjects"], [i + 1 for i in EXPOSED])
    sections = {"fold2_all": np.ones(len(mask), dtype=bool), "fold2_unmasked_two": mask, "fold2_other_85": ~mask}
    for section, scope in sections.items():
        if section != "fold2_all":
            for method, model in ((SOURCE_METHOD, historic[FOLD]), (CURRENT, new)):
                fold_scores.append(fold_row(model, method, section, FOLD, scope))
            pairs.append(paired(historic[FOLD], new, FOLD, section, scope))
        pairs.append(paired(new, new, FOLD, section, scope, start="off", comparison="same_best_off_to_full"))
    samples = {str(row["patient_id_1based"]): row for row in patients if row["scope"] == "unmasked_pair"}
    h.require(len(samples) == 2 and len(coverage) == 9 and int(mask.sum()) == 2 and int((~mask).sum()) == 85,
              "Invalid output scope")
    return fold_scores, pairs, patients, coverage, samples


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(map(str, cells)) + " |" for cells in rows)])


def ten_fold(fold_scores):
    summary = {}
    for name in (SOURCE_METHOD, CURRENT):
        rows = [row for row in fold_scores if row["method"] == name and row["scope"] == "fold"]
        h.require(len(rows) == 10, "Ten-fold macro metrics require every fold")
        summary[name] = {m: {"mean": float(np.mean([r[m] for r in rows])),
                             "population_sd": float(np.std([r[m] for r in rows], ddof=0))}
                         for m in METRICS}
    return summary


def summary_text(summary):
    return table(("模型", *COLUMNS), ((name, *(f"{marks[m]['mean']*100:.4f} ± {marks[m]['population_sd']*100:.4f}"
                                              for m in METRICS)) for name, marks in summary.items()))


def paired_text(rows):
    def name(r):
        return {"ten_fold_mean": "十折", "same_nine_fold_mean": "其余九折", "fold": "Fold 2 全部 87 人",
                "fold2_unmasked_two": "377、407", "fold2_other_85": "Fold 2 其余 85 人",
                "fold2_all": "Fold 2 全部 87 人"}[r["scope"]]
    return table(("比较", "范围", "ASD 纠正/误伤", "TD 纠正/误伤", "净纠错", "ΔACC pp", "ΔBA pp"), (
        ("原 G2 → 本轮" if row["comparison"] == "original_full_to_unmasked_full" else "本轮 Off → Full",
         name(row), f"{row['asd_corrected']}/{row['asd_harmed']}", f"{row['td_corrected']}/{row['td_harmed']}",
         f"{row['net']:+d}", f"{row['delta_acc_pp']:+.4f}", f"{row['delta_hard_auc_pp']:+.4f}") for row in rows))


def sample_text(samples):
    return table(("患者", "真值", "原 G2 → 本轮 Off → Full 预测", "原 G2 / Off / Full p(ASD)",
                  "联合 / 谱内 / 谱外 L2", "Off → Full 真类 Margin"), (
        (row["patient_id_1based"], "ASD" if row["true_label"] == 0 else "TD",
         " → ".join("ASD" if row[k] == 0 else "TD" for k in (
             "prediction_original_g2", "prediction_current_off", "prediction_current_full")),
         " / ".join(f"{row[k]:.7f}" for k in ("p_asd_original_g2", "p_asd_current_off", "p_asd_current_full")),
         " / ".join(f"{row[k]:.6g}" for k in ("delta_l2", "delta_parallel_l2", "delta_perp_l2")),
         f"{row['true_margin_change_off_to_full']:+.7f}") for row in samples.values()))


def append_handoff(body):
    block = ("\n\n" + body.rstrip() + "\n").encode()
    marker = ("## " + CONFIG["experiment"] + "\n").encode()
    digest = lambda data: hashlib.sha256(data).hexdigest()
    with HANDOFF.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        before = stream.read()
        if marker in before:
            i = before.find(block)
            h.require(i >= 0 and before.count(marker) == 1, "A conflicting 82.4.6.1 handoff already exists")
            prefix, expected, appended = before[:i], before, False
        else:
            prefix, expected, appended = before, before + block, True
            stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        after = HANDOFF.read_bytes()
        h.require(after == expected and digest(after[:len(prefix)]) == digest(prefix) and
                  after[len(prefix):len(prefix) + len(block)] == block, "Handoff append readback failed")
    return {"path": str(HANDOFF), "prefix_sha256": digest(prefix), "prefix_bytes": len(prefix),
            "append_sha256": digest(block), "append_bytes": len(block), "file_sha256_at_readback": digest(after),
            "historical_prefix_exact": True, "section_exact": True, "new_append": appended}


def publish(manifest, historic, new, audit):
    scores, pairs, patients, coverage, samples = snapshot(historic, new, audit)
    summary = ten_fold(scores)
    r = audit["rng"]
    pair_subset = [row for row in pairs if row["scope"] != "fold" or row["fold"] == FOLD]
    f2_scores = [row for row in scores if row["fold"] == FOLD]
    f2_text = table(("模型", "范围", "Best epoch", *COLUMNS), (
        ("原 G2" if row["method"] == SOURCE_METHOD else "本轮", {"fold": "全部 87 人",
         "fold2_unmasked_two": "377、407", "fold2_other_85": "其余 85 人"}[row["scope"]], row["best_epoch"],
         *(f"{100*row[m]:.4f}" for m in METRICS)) for row in f2_scores))
    maxima = audit["ledger"]["budget_maximum_ratio"]
    protection = ", ".join(f"{name}={100 * maxima[family + '.full']:.7g}%" for family, name in (
        ("gfr", "GFR"), ("darg", "DARG"), ("cam", "CAM"), ("head", "OutputLayer")))
    audited = (f"Fold 2 Best、Latest、下一步训练及评估均为 1/1 精确恢复。独立回放 {r['replayed_epochs']} 个 epoch，"
               f"与 DARG 的 {r['darg_common_epochs']} 个共有 epoch 和原 G2 的 {r['common_epochs']} 个共有 epoch "
               f"的前、训练后、评估后三阶段 RNG 全部一致。六次历史 Dropout 的调用顺序与张量形状一致。"
               f"float64 EMA 独立精确复算 {audit['ema_exact_epochs']}/{r['replayed_epochs']} 轮。"
               f"保护文件 {manifest['guard_after']['files']} 份 SHA 前后一致。")
    bankers = audit["banks"]["oof_exposed"]
    protocol = (f"物理 GPU0 `{manifest['device']['uuid']}`，单 worker，seed 0，从对应 E61 epoch 10 启动。"
                f"本轮首次最高 ACC 位于 epoch {audit['best_epoch']}，运行到 epoch {audit['end_epoch']}。"
                "保留历史 4096 对关系及顺序、原三历史优化器状态、类别权重、871 节点转导图和 val=test 选模。"
                "其余九折均只读复用原 Best，概率、硬分类和六项指标已逐折精确复算。")
    train_info = ("DARG 原成员仍为 786 人，新增监督与统计由 784 人扩至 786 人，新增标签与困难度按历史训练顺序对齐。"
                  f"Original O U16 和八条 RN50 医学 Prompt 沿用历史缓存，五折统计使用局部 RandomState "
                  f"{audit['banks']['stratified_seed']}。377 与 407 的训练统计银行分别为 "
                  f"{bankers['377']['inner_bank']}、{bankers['407']['inner_bank']}，各银行留出相应患者；推理银行包含两人。"
                  "其余 85 人从未进入新增监督和统计，翻转其标签的预检对预测、损失、统计均无影响。")
    gradient_text = ("两个病例的真实类别平衡 Full CE 梯度均可沿残差到达 Prompt、控制器及谱内外候选。"
                     "校准器零位移的首轮差分梯度与施加共享增量均严格为零，历史首步更新精确一致。"
                     "非零残差的实际写入共享差分梯度通过 0.1% 范数预算；Best 和 Latest 分别用其评估前和续训 EMA。")
    if protection:
        gradient_text += " 逐族最大实际比例为 " + protection + "。"
    doc = [f"# {CONFIG['experiment']}", "", "**COMPLETE_AUDITED**", "", ABOUT, "",
           "## 数据、方法与训练", "", train_info, "", protocol, "",
           f"服务器源码 commit `{CONFIG['server_commit']}`，GitHub 原 G2 分支 commit `{CONFIG['parent_remote_commit']}`。"
           f"十份关键文件字节核验通过。本轮冻结 SHA `{manifest['implementation_sha']}`。", "",
           "## 十折表现", "", "均值 ± population SD，百分比。Hard-AUC 是平衡准确率，Prob-AUC 使用连续 p(ASD)，ASD=0。",
           "", summary_text(summary), "", "## Fold 2 表现", "", f2_text, "",
           "## 人员配对及固定权重残差", "", paired_text(pair_subset), "",
           "十折与九折的 Δ 指标是逐折均值之差，人数来自逐患者配对。十折逐折及六项差值见 paired_comparisons.csv。", "",
           sample_text(samples), "", "## 独立审计", "", audited, "", gradient_text, "",
           "两名患者的逐阶段监督梯度、Gaussian 统计及原始 dtype 预算核验见 audit_status.json。", "",
           "## 解释范围", "", "原 G2 → 本轮含新增监督、统计变化与可能的 Best epoch 差异。"
           "本轮 Off → Full 衡量同一 Best 权重下开启残差的即时作用。"
           "历史单 seed、CV 前 RFE 与预处理、val=test 选模及转导图限制泛化解释；两名有意暴露患者的结果仅作诊断。", ""]
    handoff_doc = "\n".join([f"## {CONFIG['experiment']}", "", "**COMPLETE_AUDITED**", "", f"权威目录 `{ROOT}`。",
                             "", ABOUT, "", train_info, "", protocol, "", summary_text(summary), "",
                             paired_text([row for row in pair_subset if row["scope"] != "same_nine_fold_mean"]), "",
                             sample_text(samples), "", audited, "", gradient_text, "",
                             "其余九折及本轮两个 checkpoint 的 SHA 已登记于 manifest.json。"])
    RESULTS.mkdir(exist_ok=True)
    write_csv("fold_metrics.csv", scores)
    write_csv("paired_comparisons.csv", pairs)
    write_csv("patient_level.csv", patients)
    handoff = append_handoff(handoff_doc)
    write(RESULTS / "report.md", "\n".join(doc).encode())
    for patient, prediction in samples.items():
        audit["affected_patients"][patient]["predictions"] = prediction
    audit.update(status="COMPLETE_AUDITED", original_metrics_independently_recomputed=10,
                 reused_folds=coverage, ten_fold=summary, handoff=handoff, protected_sha256=manifest["guard_after"])
    h.write_json(RESULTS / "audit_status.json", audit)
    manifest.update(status="COMPLETE_AUDITED", unchanged_fold_sources=coverage, metrics=summary, handoff=handoff,
                    result_files={str(path.relative_to(ROOT)): h.sha_file(path)
                                  for path in sorted(RESULTS.iterdir()) if path.is_file()})
    manifest.pop("last_error", None)
    h.write_json(MANIFEST, manifest)
    print("COMPLETE_AUDITED | Fold 2 Best/Latest/next 1/1 | other nine exact | handoff verified", flush=True)
