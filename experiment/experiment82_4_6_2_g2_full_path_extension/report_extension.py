"""Report the diagnostic scope and commit validated patient-level results."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import os
from pathlib import Path

import numpy as np

from audit_extension import METRICS, score
from extension import ADDED, CONFIG, EXPOSED, FOLDS, MANIFEST, OLD_EXPOSED, RESULTS, ROOT, h

DARG = "E74.4-G2 DARG"
BASE = "E82.4.6-G2"
PRIOR = "E82.4.6.1"
CURRENT = "E82.4.6.2"
PRIMARY = {2: PRIOR, 3: BASE}
HEADER = ("ACC", "Hard-AUC / BA", "Prob-AUC", "SEN-ASD", "SPE-TD", "ASD-F1")
HANDOFF = Path("/model/spromptgl/experiment/handoff.md")
LIMIT = ("本轮共有 10 名训练—测试重叠患者，其中 8 人为本轮新增。患者真实标签进入 DARG 的分类和关系训练，"
         "同时进入新增谱/Prompt 的监督、校准和 Gaussian 统计。Fold 2、Fold 3 的测试与十折汇总均受标签暴露影响。"
         "本实验衡量训练成员扩充和标签暴露的敏感性，不能据此推断独立泛化性能或判定 MICCAI 创新模块 GO。")


def commit(path, data):
    scratch = path.with_name(path.name + ".tmp")
    with scratch.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(scratch, path)


def save_rows(file, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    commit(RESULTS / file, stream.getvalue().encode())


def table(cols, rows):
    return "\n".join(["| " + " | ".join(map(str, row)) + " |" for row in (cols, ["---"] * len(cols), *rows)])


def margin(item, condition):
    logit = item[condition]["logits"]
    return (1 - 2 * item["truth"]) * (logit[:, 0].astype(np.float64) - logit[:, 1])


def selectors(fold, ids):
    all_rows = np.ones(len(ids), dtype=bool)
    if fold not in FOLDS:
        return {"all": all_rows}
    out = {"all": all_rows, "all_exposed": np.isin(ids, [i + 1 for i in EXPOSED[fold]]),
           "newly_added": np.isin(ids, [i + 1 for i in ADDED[fold]])}
    out["held_out_82"] = ~out["all_exposed"]
    if OLD_EXPOSED[fold]:
        out["historical_pair"] = np.isin(ids, [i + 1 for i in OLD_EXPOSED[fold]])
    return out


def evaluate(item, condition, selection):
    pick = item[condition]
    return score(pick["probability"][selection], pick["prediction"][selection], item["truth"][selection])


def prediction_pair(before, after, mask, *, start, finish, name, fold, scope):
    h.require(np.array_equal(before["subjects"], after["subjects"]) and np.array_equal(before["truth"], after["truth"]),
              "The two predictions use a different patient order or label")
    y = before["truth"][mask]
    old = before[start]["prediction"][mask]
    new = after[finish]["prediction"][mask]
    wrong, right = old != y, old == y
    fix, harm = wrong & (new == y), right & (new != y)
    gain = (margin(after, finish) - margin(before, start))[mask]
    prev, post = evaluate(before, start, mask), evaluate(after, finish, mask)
    return {
        "comparison": name, "fold": fold, "scope": scope, "folds": 1, "n": len(y),
        "corrected": int(fix.sum()), "harmed": int(harm.sum()), "net": int(fix.sum()) - int(harm.sum()),
        "asd_corrected": int((fix & (y == 0)).sum()), "asd_harmed": int((harm & (y == 0)).sum()),
        "td_corrected": int((fix & (y == 1)).sum()), "td_harmed": int((harm & (y == 1)).sum()),
        "baseline_wrong": int(wrong.sum()), "baseline_right": int(right.sum()),
        "wrong_positive_margin_gain": int((wrong & (gain > 0)).sum()),
        "right_positive_margin_gain": int((right & (gain > 0)).sum()),
        "wrong_margin_mean_change": float(gain[wrong].mean()) if wrong.any() else "",
        "right_margin_mean_change": float(gain[right].mean()) if right.any() else "",
        **{k + "_before": prev[k] for k in METRICS}, **{k + "_after": post[k] for k in METRICS},
        **{"delta_" + k + "_pp": 100 * (post[k] - prev[k]) for k in METRICS},
    }


def fold_mean(rows, scope):
    counts = ("n", "corrected", "harmed", "net", "asd_corrected", "asd_harmed", "td_corrected", "td_harmed",
              "baseline_wrong", "baseline_right", "wrong_positive_margin_gain", "right_positive_margin_gain")
    keys = [m + "_" + phase for phase in ("before", "after") for m in METRICS]
    result = {"comparison": rows[0]["comparison"], "fold": "", "scope": scope, "folds": len(rows),
              **{key: sum(r[key] for r in rows) for key in counts}}
    for group in ("wrong", "right"):
        weighted = [r[group + "_margin_mean_change"] * r["baseline_" + group] for r in rows if r["baseline_" + group]]
        result[group + "_margin_mean_change"] = sum(weighted) / result["baseline_" + group] if weighted else ""
    result.update({k: float(np.mean([r[k] for r in rows])) for k in keys})
    result.update({"delta_" + k + "_pp": 100 * (result[k + "_after"] - result[k + "_before"]) for k in METRICS})
    return result


def folds_and_pairs(data):
    scores, pairs = [], []
    comparisons = {DARG + "_to_full": (DARG, "full", CURRENT, "full"),
                   BASE + "_to_full": (BASE, "full", CURRENT, "full"),
                   PRIOR + "_to_full": (PRIOR, "full", CURRENT, "full"),
                   "DARG_to_current_off": (DARG, "full", CURRENT, "off"),
                   "current_off_to_full": (CURRENT, "off", CURRENT, "full")}
    for fold in h.FOLDS:
        ids = data[BASE][fold]["subjects"]
        scoped = selectors(fold, ids)
        for scope, selected in scoped.items():
            for method, saved in data.items():
                item = saved[fold]
                scores.append({"method": method, "fold": fold, "scope": scope, "best_epoch": item["best_epoch"],
                               "best_path": item["best_path"], "best_sha": item["best_sha"],
                               **evaluate(item, "full", selected)})
            for name, (previous, start, current, finish) in comparisons.items():
                pairs.append(prediction_pair(data[previous][fold], data[current][fold], selected, start=start, finish=finish,
                                             name=name, fold=fold, scope=scope))
            prev = PRIMARY.get(fold, BASE)
            pairs.append(prediction_pair(data[prev][fold], data[CURRENT][fold], selected, start="full", finish="full",
                                         name="primary_reference_to_full", fold=fold, scope=scope))
            if fold == 2:
                item = data[PRIOR][fold]
                pairs.append(prediction_pair(item, item, selected, start="off", finish="full",
                                             name="82.4.6.1_off_to_full", fold=fold, scope=scope))
    for name in (*comparisons, "primary_reference_to_full"):
        rows = [r for r in pairs if r["comparison"] == name and r["scope"] == "all"]
        h.require(len(rows) == 10, "Fold-level pair coverage is incomplete")
        pairs += [fold_mean(rows, "ten_fold_mean"),
                  fold_mean([r for r in rows if r["fold"] not in FOLDS], "unchanged_eight_mean")]
    return scores, pairs


def patient_table(data, certificate):
    rows, summaries = [], {}
    columns = {"darg": (DARG, "full"), "original_g2": (BASE, "full"), "e82_4_6_1": (PRIOR, "full"),
               "current_off": (CURRENT, "off"), "current_full": (CURRENT, "full")}
    for fold in h.FOLDS:
        current = data[CURRENT][fold]
        previous = data[PRIMARY.get(fold, BASE)][fold]
        ids, y = current["subjects"], current["truth"]
        m = {name: margin(data[method][fold], condition) for name, (method, condition) in columns.items()}
        prev_margin = margin(previous, "full")
        scope = selectors(fold, ids)
        for j, patient in enumerate(ids):
            case, label = int(patient), int(y[j])
            exposed = fold in FOLDS and bool(scope["all_exposed"][j])
            new = exposed and scope["newly_added"][j]
            old_pred = int(previous["full"]["prediction"][j])
            new_pred = int(current["full"]["prediction"][j])
            off_pred = int(current["off"]["prediction"][j])
            kind = "unchanged_fold" if fold not in FOLDS else (
                "newly_added" if new else "historical_pair" if exposed else "held_out_82")
            row = {"fold": fold, "patient_id_1based": case, "true_label": label, "scope": kind,
                   "in_current_training": int(exposed), "is_new_training_member": int(new),
                   "primary_reference": PRIMARY.get(fold, BASE), "prediction_reference": old_pred,
                   "p_asd_reference": float(previous["full"]["probability"][j, 0]),
                   "true_margin_reference": float(prev_margin[j])}
            for key, (method, condition) in columns.items():
                item = data[method][fold][condition]
                row.update({"prediction_" + key: int(item["prediction"][j]), "p_asd_" + key: float(item["probability"][j, 0]),
                            "true_margin_" + key: float(m[key][j])})
            row.update({"corrected": int(old_pred != label and new_pred == label), "harmed": int(old_pred == label and new_pred != label),
                        "residual_corrected": int(off_pred != label and new_pred == label),
                        "residual_harmed": int(off_pred == label and new_pred != label),
                        "margin_change_vs_reference": float(m["current_full"][j] - prev_margin[j]),
                        "margin_gain_off_to_full": float(m["current_full"][j] - m["current_off"][j])})
            residual = current.get("residual", {})
            row.update({k: float(residual[k][j]) if k in residual else "" for k in
                        ("delta_l2", "delta_parallel_l2", "delta_perp_l2", "cam_ratio")})
            gradient = certificate[str(fold)]["gradients"][str(case)]["best"] if exposed else {}
            row.update({"gradient_" + k: gradient.get(k, "") for k in (
                "native_nll_logp_l2", "full_ce_delta_l2", "prompt_l2", "control_l2",
                "spectral_candidate_l2", "complement_candidate_l2")})
            row.update({"best_sha_reference": previous["best_sha"], "best_sha_current": current["best_sha"]})
            rows.append(row)
            if exposed:
                summaries[str(case)] = row
    h.require(len(rows) == len({r["patient_id_1based"] for r in rows}) == 871 and len(summaries) == 10 and
              sum(x["is_new_training_member"] for x in rows) == 8, "Incorrect CV or exposure accounting")
    return rows, summaries


def stats(scores, skip_affected=False):
    out = {}
    for source in (DARG, BASE, PRIOR, CURRENT):
        rows = [r for r in scores if r["method"] == source and r["scope"] == "all" and
                (not skip_affected or r["fold"] not in FOLDS)]
        h.require(len(rows) == (8 if skip_affected else 10), "Historical and current folds are incomplete")
        out[source] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                           "population_sd": float(np.std([r[k] for r in rows], ddof=0))} for k in METRICS}
    return out


def metrics_table(summary):
    return table(("模型", *HEADER), [(model, *(f"{100*result[k]['mean']:.4f} ± {100*result[k]['population_sd']:.4f}"
                                               for k in METRICS)) for model, result in summary.items()])


def patient_details(items):
    def cls(x):
        return "ASD" if x == 0 else "TD"

    return table(("Fold / 患者", "真值", "原参考 → Off → Full", "p(ASD) 原参考 / Off / Full",
                  "真类 Margin 原参考 / Off / Full", "残差 L2 联合 / 谱内 / 谱外", "纠正 / 误伤"), [
        (f"{r['fold']} / {r['patient_id_1based']}", cls(r["true_label"]),
         " → ".join(cls(r[k]) for k in ("prediction_reference", "prediction_current_off", "prediction_current_full")),
         " / ".join(f"{r[k]:.6f}" for k in ("p_asd_reference", "p_asd_current_off", "p_asd_current_full")),
         " / ".join(f"{r[k]:+.6f}" for k in ("true_margin_reference", "true_margin_current_off", "true_margin_current_full")),
         " / ".join(f"{r[k]:.6g}" for k in ("delta_l2", "delta_parallel_l2", "delta_perp_l2")),
         f"{r['corrected']} / {r['harmed']}") for r in items])


SCOPE_NAMES = {"all": "全部 87 人", "newly_added": "本轮新增", "historical_pair": "既有 377、407",
               "all_exposed": "全部训练重叠", "held_out_82": "其余 82 人", "ten_fold_mean": "十折均值",
               "unchanged_eight_mean": "八折均值"}


def comparisons_table(rows):
    return table(("Fold", "范围", "比较", "纠正 / 误伤 / 净值", "ΔACC pp", "ΔBA pp", "ΔProb-AUC pp",
                  "原错 Margin 提高", "原对 Margin 提高"), [
        (r["fold"] or "—", SCOPE_NAMES[r["scope"]], r["comparison"],
         f"{r['corrected']} / {r['harmed']} / {r['net']:+d}", f"{r['delta_acc_pp']:+.4f}",
         f"{r['delta_hard_auc_pp']:+.4f}", f"{r['delta_prob_auc_pp']:+.4f}",
         f"{r['wrong_positive_margin_gain']} / {r['baseline_wrong']}",
         f"{r['right_positive_margin_gain']} / {r['baseline_right']}") for r in rows])


def append_handoff(body):
    marker = ("## " + CONFIG["experiment"] + "\n").encode()
    block = ("\n\n" + body.rstrip() + "\n").encode()

    def sha(raw):
        return hashlib.sha256(raw).hexdigest()

    with HANDOFF.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        original = stream.read()
        if marker in original:
            offset = original.find(block)
            h.require(offset >= 0 and original.count(marker) == 1, "Conflicting experiment handoff")
            prefix, expected, appended = original[:offset], original, False
        else:
            prefix, expected, appended = original, original + block, True
            stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        readback = HANDOFF.read_bytes()
        h.require(readback == expected and readback[:len(prefix)] == prefix and
                  readback[len(prefix):len(prefix) + len(block)] == block, "Independent append readback failed")
    return {"path": str(HANDOFF), "prefix_sha256": sha(prefix), "prefix_bytes": len(prefix), "append_bytes": len(block),
            "append_sha256": sha(block), "file_sha256_at_readback": sha(readback), "historical_prefix_exact": True,
            "new_section_exact": True, "new_append": appended}


def publish(manifest, darg, base, prior, variants, audit):
    current = {**base, **variants}
    data = {DARG: darg, BASE: base, PRIOR: prior, CURRENT: current}
    scores, pairs = folds_and_pairs(data)
    patients, exposed = patient_table(data, audit)
    population, eight = stats(scores), stats(scores, skip_affected=True)
    unchanged = {}
    for k in h.FOLDS:
        if k not in FOLDS:
            src = base[k]
            unchanged[str(k)] = {key: src[key] for key in (
                "best_epoch", "best_sha", "best_path", "certificate_sha", "certificate_path",
                "inference_sha", "reference_sha", "partition_sha")}
            unchanged[str(k)]["prediction_probability_and_metrics_exact"] = True
    h.require(eight[BASE] == eight[PRIOR] == eight[CURRENT] and
              len(unchanged) == 8 and all(
                p["prediction_original_g2"] == p["prediction_current_full"] and p["p_asd_original_g2"] == p["p_asd_current_full"]
                for p in patients if p["scope"] == "unchanged_fold"), "Eight reused folds changed")

    members = {k: cert["membership"] for k, cert in audit.items()}
    roster = table(("Fold", "本轮新增，依追加顺序", "历史训练 → 本轮", "新增监督 → 本轮", "测试人数", "训练—测试重叠"), [
        (k, ", ".join(map(str, m["added_1based_in_order"])), f"{m['before_train']} → {m['after_train']}",
         f"{m['before_legal']} → {m['after_legal']}", m["test"], ", ".join(map(str, m["overlap_1based"]))) for k, m in members.items()])
    choices = [r for r in pairs if r["fold"] in FOLDS and
               (r["comparison"] in ("primary_reference_to_full", "current_off_to_full") or
                (r["scope"] == "all" and r["comparison"] in (DARG + "_to_full", "DARG_to_current_off", "82.4.6.1_off_to_full")))]
    current_scores = [r for r in scores if r["fold"] in FOLDS and r["scope"] == "all"]
    f2f3 = table(("模型", "Fold", "Best epoch", *HEADER), [
        (r["method"], r["fold"], r["best_epoch"], *(f"{100*r[k]:.4f}" for k in METRICS)) for r in current_scores])
    stage_rows = [r for r in pairs if r["scope"] == "ten_fold_mean" and r["comparison"] in
                  (DARG + "_to_full", BASE + "_to_full", PRIOR + "_to_full")]
    preservation = table(("Fold", "Best epoch", "终止 epoch", "RNG 回放轮数", "与 DARG / 原 G2 共有轮数",
                          "EMA 精确轮数", "最大实际共享梯度"), [
        (k, cert["best_epoch"], cert["end_epoch"], cert["rng"]["replayed_epochs"],
         f"{cert['rng']['darg']['common_epochs']} / {cert['rng']['original_g2']['common_epochs']}",
         cert["journal"]["ema_bitwise_epochs"],
         f"{100*max(cert['journal']['budget_maximum_ratio'][family + '.full'] for family in ('gfr','darg','cam','head')):.7g}%")
        for k, cert in audit.items()])
    methods = ("复用 E82.4.6-G2 SCMR-DA-SRPE、E74.4-G2 DARG 及 E82.4.6.1 的实现，从各自 E61 epoch 10 "
               "归档恢复历史模型、优化器和 RNG。沿用 E61 顺序关系采样与固定 ASD=0、TD=1 的逆频率权重。每折 4096 对，"
               "同类和异类各 2048。Original O U16、中心、白化尺度和八条 RN50 Prompt 保持原身份；五折 Gaussian 银行"
               "按新成员重建，每个训练样本从自身使用的统计银行留出。推理银行含全部 789 名训练成员。"
               "EMA 保持先评估后更新，扩充的标签与关系对允许完整历史损失和首步更新变化。")
    verification = (f"2/2 Best、Latest 与下一步训练及评估重载达到逐位一致。两折均保有六次历史 Dropout，"
                    f"共有 epoch 的训练前、训练后和评估后 RNG 一致。Fold 2 与 E82.4.6.1 的 "
                    f"{audit['2']['rng'][PRIOR]['common_epochs']} 个共有 epoch 也完全一致。"
                    "两折各余 82 人的标签翻转探针、端到端监督梯度、Jacobian 两空间正交及 8% 残差限幅通过审计。"
                    f"{manifest['guard_after']['files']} 份历史保护文件 SHA 前后一致。")
    extension_rows = [r for r in choices if r["comparison"] == "primary_reference_to_full" and
                      r["scope"] in ("newly_added", "held_out_82")]
    change = " ".join(f"Fold {r['fold']} {SCOPE_NAMES[r['scope']]}纠正 {r['corrected']} 人、误伤 {r['harmed']} 人，"
                      f"净 {r['net']:+d}。" for r in extension_rows)
    order = [r for r in exposed.values()]
    block = [f"## {CONFIG['experiment']}", "", "**COMPLETE_AUDITED**", "",
             f"权威实验目录 `{ROOT}`。冻结源码 SHA `{manifest['implementation_sha']}`。", "",
             LIMIT, "", roster, "", "十折指标为百分比及 population SD。", "", metrics_table(population), "",
             change, "", comparisons_table(choices), "", patient_details(order), "", verification, "", preservation]
    report = [f"# {CONFIG['experiment']}", "", "**COMPLETE_AUDITED**", "", LIMIT, "",
              "## 成员、模型与训练", "", roster, "", methods, "",
              f"首选位置返回 errno 5，使用唯一权威目录 `{ROOT}`。物理 GPU0 运行 {manifest['worker_plan']['concurrent']} 个训练 worker。"
              f"归档保护清单、来源身份和冻结 SHA 见 manifest.json。", "",
              "## 十折指标", "", "百分比，均值 ± population SD。Hard-AUC 为 BA，Prob-AUC 使用连续 p(ASD)。", "",
              metrics_table(population), "", comparisons_table(stage_rows), "",
              "## 移除 Fold 2 和 Fold 3 的八折结果", "", metrics_table(eight), "",
              "原 G2、E82.4.6.1 与本轮的八折逐患者概率、预测和全部指标完全一致。其他八折只读重载，文件 SHA 见 manifest.json。", "",
              "## 两折表现", "", f2f3, "", "## 配对纠错与真实类别 Margin", "",
              "Fold 2 的主参考为 E82.4.6.1，Fold 3 的主参考为 E82.4.6-G2。主参考到本轮 Full 的差值包含"
              "成员、训练和 Best epoch 的变化；同一 Best 的 Off→Full 直接衡量残差。Margin 提高列使用严格正增量。", "",
              change, "", comparisons_table(choices), "", "## 十名重叠患者", "", patient_details(order), "",
              "## 独立审计", "", verification, "", preservation, "",
              "Fold 2 和 Fold 3 的 871 节点转导图、CV 前 RFE 和原 val=test checkpoint 选择均照历史协议执行。"]
    save_rows("fold_metrics.csv", scores)
    save_rows("paired_comparisons.csv", pairs)
    save_rows("patient_level.csv", patients)
    h.write_json(RESULTS / "membership_audit.json", {"status": "PASS", "affected_folds": members,
                 "newly_added": 8, "total_train_test_overlap": 10, "unaffected_folds": unchanged})
    handoff = append_handoff("\n".join(block))
    commit(RESULTS / "report.md", ("\n".join(report) + "\n").encode())
    h.write_json(RESULTS / "audit_status.json", {"status": "COMPLETE_AUDITED", "best_reloaded": 2, "latest_reloaded": 2,
                 "latest_next_step_exact": 2, "protected_sha256": manifest["guard_after"], "folds": audit,
                 "unchanged_eight": unchanged, "ten_fold": population, "eight_fold": eight,
                 "exposed_predictions": exposed, "handoff": handoff, "independent_external_performance": False})
    manifest.update(status="COMPLETE_AUDITED", unchanged_fold_sources=unchanged, handoff=handoff,
                    ten_fold=population, unchanged_eight=eight,
                    result_files={str(p.relative_to(ROOT)): h.sha_file(p) for p in sorted(RESULTS.iterdir()) if p.is_file()})
    manifest.pop("last_error", None)
    h.write_json(MANIFEST, manifest)
    print("COMPLETE_AUDITED | Fold 2 and 3 Best/Latest/next 2/2 | eight untouched folds exact | handoff verified", flush=True)
