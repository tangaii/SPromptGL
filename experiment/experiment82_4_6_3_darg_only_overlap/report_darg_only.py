"""Publish paired audit results and append a verified historical entry."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import os
from pathlib import Path

from darg_only import CFG, CONFIG, EXPOSED, FOLDS, MANIFEST, RESULTS, ROOT, f, formats, h

import numpy as np

DARG, BASE, PRIOR, FULL = formats.DARG, formats.BASE, formats.PRIOR, formats.CURRENT
CURRENT = "E82.4.6.3"
METRICS = formats.METRICS
PRIMARY = "e82_4_6_2_full_to_e82_4_6_3_full"
DIRECT = "e82_4_6_3_off_to_full"
QUERY = "anonymous_to_medical"
OFF_PAIR = "e82_4_6_2_off_to_e82_4_6_3_off"
PREVIOUS_PATTERN = "pre_extension_to_e82_4_6_2_full"
LIMIT = ("本轮 10 名训练—测试重叠患者的标签继续进入 DARG 分类和关系训练。"
         "共享主干和转导患者图仍可间接传递相关信息。受影响折与十折汇总只用于标签暴露诊断，"
         "不能作为独立泛化结果或 MICCAI 创新模块 GO 的依据。")


def save_csv(name, rows):
    stream = io.StringIO(newline="")
    sink = csv.DictWriter(stream, fieldnames=list(rows[0]))
    sink.writeheader()
    sink.writerows(rows)
    formats.commit(RESULTS / name, stream.getvalue().encode())


def paired(old, new, selected, *, start, finish, name, fold, scope):
    rec = formats.prediction_pair(old, new, selected, start=start, finish=finish, name=name, fold=fold, scope=scope)
    change = (formats.margin(new, finish) - formats.margin(old, start))[selected]
    good = (old[start]["prediction"] == old["truth"])[selected]
    rec.update(wrong_negative_margin_gain=int((~good & (change < 0)).sum()),
               right_negative_margin_gain=int((good & (change < 0)).sum()),
               margin_positive=int((change > 0).sum()), margin_negative=int((change < 0).sum()),
               margin_equal=int((change == 0).sum()))
    return rec


def summary(rows, exclude=()):
    out = {}
    for method in (DARG, BASE, PRIOR, FULL, CURRENT):
        current = [r for r in rows if r["method"] == method and r["scope"] == "all" and r["fold"] not in exclude]
        h.require(len(current) == 10 - len(exclude), "A fold metric source is missing")
        out[method] = {key: {"mean": float(np.mean([r[key] for r in current])),
                             "population_sd": float(np.std([r[key] for r in current], ddof=0))} for key in METRICS}
    return out


def pool(items, condition):
    return {"truth": np.concatenate([rec["truth"] for rec in items]),
            "subjects": np.concatenate([rec["subjects"] for rec in items]),
            condition: {key: np.concatenate([rec[condition][key] for rec in items])
                        for key in ("logits", "probability", "prediction")}}


def fold_results(data):
    pairs, results = [], []
    comparisons = {
        PRIMARY: (FULL, "full", CURRENT, "full"),
        "darg_to_current_off": (DARG, "full", CURRENT, "off"),
        "darg_to_current_full": (DARG, "full", CURRENT, "full"),
        "e82_4_6_g2_full_to_current_full": (BASE, "full", CURRENT, "full"),
        "e82_4_6_1_full_to_current_full": (PRIOR, "full", CURRENT, "full"),
        OFF_PAIR: (FULL, "off", CURRENT, "off"),
        "e82_4_6_2_off_to_full": (FULL, "off", FULL, "full"),
        DIRECT: (CURRENT, "off", CURRENT, "full"),
        QUERY: (CURRENT, "anonymous", CURRENT, "full"),
    }

    def resolve(name, fold):
        if name == PREVIOUS_PATTERN:
            old = PRIOR if fold == 2 else BASE
            return data[old][fold], "full", data[FULL][fold], "full"
        left, start, right, finish = comparisons[name]
        return data[left][fold], start, data[right][fold], finish

    for fold in h.FOLDS:
        ids = data[BASE][fold]["subjects"]
        for scope, selected in formats.selectors(fold, ids).items():
            for method, saved in data.items():
                item = saved[fold]
                results.append({"method": method, "fold": fold, "scope": scope,
                                "best_epoch": item["best_epoch"], "best_sha": item["best_sha"],
                                "best_path": item["best_path"], **formats.evaluate(item, "full", selected)})
            for name in (*comparisons, PREVIOUS_PATTERN):
                if name == QUERY and fold not in FOLDS:
                    continue
                left, start, right, finish = resolve(name, fold)
                pairs.append(paired(left, right, selected, start=start, finish=finish, name=name, fold=fold, scope=scope))

    for name in (*comparisons, PREVIOUS_PATTERN):
        resolved = [resolve(name, fold) for fold in FOLDS]
        start, finish = resolved[0][1::2]
        left, right = pool([r[0] for r in resolved], start), pool([r[2] for r in resolved], finish)
        overlap = np.isin(left["subjects"], [i + 1 for rows in EXPOSED.values() for i in rows])
        for scope, select, size in (("pooled_174", np.ones(len(overlap), bool), 174),
                                    ("pooled_10", overlap, 10), ("pooled_164", ~overlap, 164)):
            h.require(int(select.sum()) == size, "Unexpected pooled comparison count")
            row = paired(left, right, select, start=start, finish=finish, name=name, fold="", scope=scope)
            row["folds"] = len(FOLDS)
            pairs.append(row)
        combined = [r for r in pairs if r["comparison"] == name and r["scope"] == "all"]
        if len(combined) == 10:
            for scope, use in (("ten_fold_mean", combined), ("unchanged_eight", [r for r in combined if r["fold"] not in FOLDS])):
                avg = formats.fold_mean(use, scope)
                for key in ("wrong_negative_margin_gain", "right_negative_margin_gain", "margin_positive", "margin_negative", "margin_equal"):
                    avg[key] = sum(r[key] for r in use)
                pairs.append(avg)
    return results, pairs


def patients(data, certificates):
    columns = (("darg", DARG, "full"), ("original_g2", BASE, "full"), ("e82_4_6_1", PRIOR, "full"),
               ("e82_4_6_2_off", FULL, "off"), ("e82_4_6_2_full", FULL, "full"),
               ("off", CURRENT, "off"), ("full", CURRENT, "full"), ("anonymous", CURRENT, "anonymous"))
    rows = []
    for fold, main in sorted(data[CURRENT].items()):
        exposed = {i + 1 for i in EXPOSED.get(fold, ())}
        gradients = certificates.get(str(fold), {}).get("scope_gradient_best", {}).get("patients", {})
        quantities = {key: (data[method][fold].get(mode),
                            formats.margin(data[method][fold], mode) if mode in data[method][fold] else None)
                      for key, method, mode in columns}
        for index, identifier in enumerate(main["subjects"]):
            i, y = int(identifier), int(main["truth"][index])
            row = {"fold": fold, "patient_id_1based": i, "patient_index_0based": i - 1, "true_label": y,
                   "in_darg_training": i in exposed, "in_scmr_legal": False,
                   "residual_masked": i in exposed, "changed_fold": fold in FOLDS,
                   "best_sha_e82_4_6_2": data[FULL][fold]["best_sha"], "best_sha_e82_4_6_3": main["best_sha"]}
            for key, (prediction, margin) in quantities.items():
                row["prediction_" + key] = int(prediction["prediction"][index]) if prediction is not None else ""
                row["p_asd_" + key] = float(prediction["probability"][index, 0]) if prediction is not None else ""
                row["true_margin_" + key] = float(margin[index]) if margin is not None else ""
            old, off, new = (row["prediction_" + c] for c in ("e82_4_6_2_full", "off", "full"))
            row.update(corrected=old != y and new == y, harmed=old == y and new != y,
                       residual_corrected=off != y and new == y, residual_harmed=off == y and new != y,
                       true_margin_gain_vs_e82_4_6_2=row["true_margin_full"] - row["true_margin_e82_4_6_2_full"],
                       true_margin_gain_off_to_full=row["true_margin_full"] - row["true_margin_off"])
            for key in ("delta_l2", "delta_parallel_l2", "delta_perp_l2", "cam_ratio"):
                row[key] = float(main["residual"][key][index]) if "residual" in main else ""
            row.update({key: gradients.get(str(i), {}).get(key, "") for key in
                        ("native_nll_logp_l2", "native_relation_gradient_l2_sum", "full_ce_delta_l2")})
            rows.append(row)
    h.require(len(rows) == len({row["patient_id_1based"] for row in rows}) == CFG["subjects"],
              "Patient export does not cover precisely the historical subjects")
    masked = [p for p in rows if p["residual_masked"]]
    h.require(len(masked) == 10 and all(p["delta_l2"] == p["delta_parallel_l2"] == p["delta_perp_l2"] ==
                                      p["full_ce_delta_l2"] == 0 and p["p_asd_off"] == p["p_asd_full"] and
                                      p["true_margin_off"] == p["true_margin_full"] for p in masked),
              "Masked patient artifact is inconsistent with checkpoints")
    return rows


def append_handoff(body):
    target = Path(CONFIG["handoff"])
    h.require(target.is_file(), "Historical handoff missing")
    block = ("\n\n" + body.rstrip() + "\n").encode()
    marker = ("## Experiment " + CONFIG["experiment"] + "\n").encode()
    with target.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        previous = stream.read()
        if marker in previous:
            begin = previous.find(block)
            h.require(begin >= 0 and previous.count(marker) == 1, "Conflicting 82.4.6.3 handoff section")
            prefix, expected, appended = previous[:begin], previous, False
        else:
            prefix, expected, appended = previous, previous + block, True
        before = hashlib.sha256(prefix).hexdigest()
        if appended:
            stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        read = target.read_bytes()
        h.require(read == expected and hashlib.sha256(read[:len(prefix)]).hexdigest() == before and
                  read[len(prefix):len(prefix) + len(block)] == block, "Handoff independent readback differs")
    return {"path": str(target), "prefix_sha256": before, "prefix_bytes": len(prefix), "append_bytes": len(block),
            "append_sha256": hashlib.sha256(block).hexdigest(), "file_sha256_at_readback": hashlib.sha256(read).hexdigest(),
            "historical_prefix_exact": True, "new_section_exact": True, "appended": appended}


def pair_table(rows):
    names = {PRIMARY: ".2 Full → .3 Full", DIRECT: ".3 Off → Full", QUERY: "匿名 → 医学 Query",
             PREVIOUS_PATTERN: "扩充前 → .2 Full", OFF_PAIR: ".2 Off → .3 Off",
             "darg_to_current_full": "DARG → .3 Full"}
    scopes = {"pooled_10": "重叠 10 人", "pooled_164": "未加入 164 人", "pooled_174": "全部 174 人",
              "all": "全部 87 人", "held_out_82": "未加入 82 人"}
    return formats.table(("比较", "范围", "n", "纠正", "误伤", "净纠错", "ΔACC (pp)", "ΔBA (pp)",
                          "原错 Margin 升/降", "原错平均 ΔMargin", "原对 Margin 升/降"),
                         [(names[x["comparison"]], (f"Fold {x['fold']} " if x["fold"] else "") + scopes[x["scope"]], x["n"],
                           x["corrected"], x["harmed"], f"{x['net']:+d}", f"{x['delta_acc_pp']:+.4f}",
                           f"{x['delta_hard_auc_pp']:+.4f}", f"{x['wrong_positive_margin_gain']}/{x['wrong_negative_margin_gain']}",
                           f"{x['wrong_margin_mean_change']:+.5f}" if x["wrong_margin_mean_change"] != "" else "—",
                           f"{x['right_positive_margin_gain']}/{x['right_negative_margin_gain']}")
                          for x in rows])


def publish(manifest, darg, base, prior, full, current, checks):
    data = dict(zip((DARG, BASE, PRIOR, FULL, CURRENT), (darg, base, prior, full, current), strict=True))
    certified_eight = {}
    for fold in sorted(set(base) - set(FOLDS)):
        for condition in ("full", "off"):
            for key, value in base[fold][condition].items():
                f.equal(current[fold][condition][key], value, "Untouched fold " + key)
        certified_eight[str(fold)] = {k: base[fold][k] for k in
                                      ("best_path", "best_sha", "certificate_path", "certificate_sha", "inference_sha")}
    scores, pairs = fold_results(data)
    rows = patients(data, checks)
    ten, eight = summary(scores), summary(scores, FOLDS)
    h.require(all(eight[k] == eight[BASE] for k in (PRIOR, FULL, CURRENT)), "Unaffected metrics differ")
    masks = [row for row in rows if row["residual_masked"]]
    main_rows = [r for r in pairs if r["comparison"] == PRIMARY and
                 (r["scope"].startswith("pooled_") or r["fold"] in FOLDS and r["scope"] in ("all", "held_out_82"))]
    diagnostic = [r for r in pairs if r["comparison"] in (DIRECT, QUERY, PREVIOUS_PATTERN, OFF_PAIR, "darg_to_current_full")
                  and r["scope"] in ("pooled_10", "pooled_164", "pooled_174")]
    key_text = "、".join(f"{p['patient_id_1based']} {'仍判对' if p['prediction_full'] == p['true_label'] else '本轮判错'}"
                       for p in masks if p["patient_id_1based"] in (681, 779, 46))
    change = 100 * (ten[CURRENT]["acc"]["mean"] - ten[FULL]["acc"]["mean"])
    direct_held, query_held, prev_held = (next(r for r in diagnostic if r["comparison"] == name and r["scope"] == "pooled_164")
                                          for name in (DIRECT, QUERY, PREVIOUS_PATTERN))
    main_held = next(r for r in main_rows if r["scope"] == "pooled_164")
    main_exposed = next(r for r in main_rows if r["scope"] == "pooled_10")
    identity_table = formats.table(("Fold", "历史 DARG", "本轮 DARG", "SCMR", "测试", "屏蔽 ID 1-based → 0-based",
                                    "Best", "Latest", "RNG 重放轮数", "与 .2 共同比对轮数"),
                                   [(fold, p["membership"]["before_train"], p["membership"]["after_train"],
                                     p["membership"]["after_legal"], p["membership"]["test"],
                                     ", ".join(f"{i+1}→{i}" for i in EXPOSED[int(fold)]), p["best_epoch"], p["end_epoch"],
                                     p["journal"]["replayed_epochs"], p["rng"]["E82.4.6.2"]["common_epochs"])
                                    for fold, p in checks.items()])
    one_fold = formats.table(("Fold", "方法", "Best", *formats.HEADER),
                             [(r["fold"], r["method"], r["best_epoch"], *(f"{100*r[k]:.4f}" for k in METRICS))
                              for r in scores if r["fold"] in FOLDS and r["scope"] == "all"])
    patient_table = formats.table(("Fold / ID", "真实标签", ".2 预测", ".3 预测", ".2 p(ASD)", ".3 Off p(ASD)",
                                   ".3 Full p(ASD)", ".2 真类 Margin", ".3 真类 Margin", "联合 / 谱内 / 谱外 L2", "纠错/误伤"),
                                  [(f"{p['fold']} / {p['patient_id_1based']}", p["true_label"],
                                    p["prediction_e82_4_6_2_full"], p["prediction_full"],
                                    f"{p['p_asd_e82_4_6_2_full']:.8f}", f"{p['p_asd_off']:.8f}", f"{p['p_asd_full']:.8f}",
                                    f"{p['true_margin_e82_4_6_2_full']:+.6f}", f"{p['true_margin_full']:+.6f}",
                                    "0 / 0 / 0", "纠正" if p["corrected"] else "误伤" if p["harmed"] else "—")
                                   for p in masks])
    query_table = formats.table(("Fold", "联合/谱内/谱外活跃 n", "医学/匿名 ΔCAM n", "Δ概率 n",
                                 "平均 |Δ p(ASD)|", "硬翻转"),
                                [(fold, "/".join(str(p["best_mask"][k + "_active_held_out"]) for k in
                                                  ("delta", "delta_parallel", "delta_perp")),
                                  p["query"]["changed_feature_rows"], p["query"]["changed_probability_rows"],
                                  f"{p['query']['mean_absolute_asd_probability_change']:.8f}", p["query"]["hard_flips"])
                                 for fold, p in checks.items()])
    residual_text = (
        f"未加入的 164 人中，同权重残差纠正 {direct_held['corrected']} 人、误伤 {direct_held['harmed']} 人、"
        f"净纠错 {direct_held['net']:+d}。原错者 Margin 上升 {direct_held['wrong_positive_margin_gain']} / "
        f"{direct_held['baseline_wrong']} 人，下降 {direct_held['wrong_negative_margin_gain']} 人。"
        f"原对者 Margin 上升 {direct_held['right_positive_margin_gain']} / {direct_held['baseline_right']} 人，"
        f"下降 {direct_held['right_negative_margin_gain']} 人。"
    )
    query_text = (f"匿名 Query 换为医学 Query，在 164 人中净纠错 {query_held['net']:+d}，"
                  f"BA 变化 {query_held['delta_hard_auc_pp']:+.4f} 个百分点。")
    method = (
        "两折沿用 E82.4.6.2 的 DARG 成员、类别权重和 E61 局部随机机制生成的 4096 个关系对。"
        "SCMR 沿用 E82.4.6-G2 的 784 行监督、Original O U16、RN50 医学与匿名 Query、"
        "五折 OOF 类别证据、float64 EMA、共享差分梯度和 8% 幅度限制。两折均从各自 E61 epoch 10 恢复。"
        "预检核验了与 E82.4.6.2 的初始参数、四优化器、RNG、图输入、训练和测试成员及关系对一致。"
        "实际模型使用历史 PLNNoPEN 的 CAM144 和原分类头。原论文 Prompt-guided PLN 含 PEN。"
    )
    h.require(len(checks) == 2 and all(c["status"] == "PASS" and c["best_reloaded"] and c["latest_reloaded"] and
                                       c["latest_next_exact"] for c in checks.values()), "Independent checks incomplete")
    audit = {"status": "COMPLETE_AUDITED", "experiment": CONFIG["experiment"], "new_formal_tasks": list(FOLDS),
             "best_reloaded": len(checks), "latest_reloaded": len(checks), "latest_next_exact": len(checks),
             "historical_sha": manifest["guard_after"], "unchanged_folds": certified_eight,
             "preflight": manifest["preflight"], "folds": checks, "test_label_exposure": True,
             "eligible_for_independent_generalization_or_innovation_go": False}
    report = [
        "# Experiment 82.4.6.3 — DARG-only Overlap Supervision", "COMPLETE_AUDITED",
        "在相同的两折 DARG 标签暴露下，测量移除 10 人的 SCMR 直接监督、类别统计和特征残差后，"
        "患者预测与固定权重残差纠错如何变化。", LIMIT, "## 设计与训练身份", method,
        "[SPromptGL 原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) · "
        "[原作者代码](https://github.com/wanxixi11/SPromptGL/tree/" + CONFIG["author_repository"] + ") · "
        "[E82.4.6.2 源码](https://github.com/tangaii/SPromptGL/tree/" + CONFIG["published_e82_4_6_2"] +
        "/experiment/experiment82_4_6_2_g2_full_path_extension)",
        identity_table,
        f"服务器 Git {CONFIG['local_commit']}。冻结 SHA-256 {manifest['implementation_sha']}。"
        f"物理 GPU0 UUID {CONFIG['physical_gpu_uuid']}。首选 OSSFS 建目录返回 errno "
        f"{(CONFIG['storage']['io_failure'] or {}).get('errno', '0')}，权威目录 {ROOT}。",
        "## 十折和单折性能", "分类指标单位为 %。十折为折均值 ± population SD，ASD=0 为阳性，"
        "Hard-AUC 等于 BA，Prob-AUC 使用连续 ASD 概率。", formats.metrics_table(ten),
        f"本轮十折 ACC {100*ten[CURRENT]['acc']['mean']:.4f}%，较 E82.4.6.2 变化 {change:+.4f} 个百分点。",
        one_fold, "### 排除 Fold 2 与 Fold 3", formats.metrics_table(eight),
        "其余八折从 E82.4.6-G2 原 Best 独立重载，逐样本概率、预测和指标与归档完全相同。"
        "检查点通过归档路径及 SHA 引用。",
        "## 同患者配对与关键病例", f"{key_text}。", pair_table(main_rows), patient_table,
        f"E82.4.6.2 的 164 人结果相对各折扩充前基准为 {prev_held['corrected']} 人纠正、"
        f"{prev_held['harmed']} 人误伤、净 {prev_held['net']:+d}。"
        f"本轮相对 E82.4.6.2 在同一 164 人中纠正 {main_held['corrected']} 人、"
        f"误伤 {main_held['harmed']} 人、净 {main_held['net']:+d}。"
        f"10 名重叠患者相对 E82.4.6.2 的净变化为 {main_exposed['net']:+d}。",
        "## 机制归因", pair_table(diagnostic), residual_text, query_table, query_text,
        ".2 Off → .3 Off 比较两次联合训练后的原输出路径，包含共同适配和 Best 轮次差异。"
        "两次训练的 SCMR 监督、统计银行和遮罩同时发生变化。固定权重 Off → Full 直接测量特征残差。"
        "固定权重的医学与匿名 Query 比较测量文本替换效应。该设计无法逐一识别三个训练差异的贡献。",
        "## 独立审计", f"Best、Latest、Latest 下一步训练和评估均精确通过 2/2。"
        f"两折核验训练前、训练后、评估后 RNG 和六次 Dropout，按首次严格最高测试 ACC 复核 Best。"
        f"历史受保护文件 {manifest['guard_after']['files']} 个 SHA-256 前后一致。",
        "Gaussian 初态从原 784 行独立重建。逐轮复算 EMA，逐位一致。重叠患者的 DARG 分类、关系梯度范数均大于零，"
        "SCMR 逐行梯度为零。每轮训练和评估、Best 与 Latest 中，重叠患者的谱内、谱外及联合残差均为零。"
        "Best 和 Latest 的增强 CAM144、logits、概率与 Off 逐元素一致。未屏蔽的两条谱路径保持活跃。"
        "训练保留端到端更新，按实际梯度写入核验 0.1% 共享预算。",
        "历史 val=test 选模、CV 前 RFE/预处理、871 节点转导图和单 seed 均继续限制解释。"
    ]
    save_csv("fold_metrics.csv", scores)
    save_csv("paired_comparisons.csv", pairs)
    save_csv("patient_level.csv", rows)
    h.write_json(RESULTS / "membership_audit.json", {k: {name: c[name] for name in
                                                        ("membership", "gaussian", "initial_parity",
                                                         "scope_gradient_best", "best_mask", "latest_mask")}
                                                       for k, c in checks.items()})
    formats.commit(RESULTS / "report.md", ("\n\n".join(report) + "\n").encode())
    body = "\n\n".join((
        "## Experiment 82.4.6.3", "COMPLETE_AUDITED · DARG-only Overlap Supervision",
        f"权威目录 {ROOT}。服务器 Git {CONFIG['local_commit']}，冻结 SHA-256 {manifest['implementation_sha']}。",
        method, identity_table, formats.metrics_table(ten), key_text + "。", pair_table(main_rows), residual_text, query_text,
        f"Best、Latest、下一步训练与评估均精确通过 2/2。Gaussian 每折 784，10 人无新增监督且各分量残差为零。"
        f"八个沿用折预测及指标与归档完全一致。历史受保护文件 {manifest['guard_after']['files']} 个 SHA-256 未变。",
        LIMIT, f"完整报告 {RESULTS / 'report.md'}。",
    ))
    handoff = append_handoff(body)
    h.write_json(RESULTS / "audit_status.json", {**audit, "handoff": handoff})
    manifest.update(status="COMPLETE_AUDITED", ten_fold=ten, unchanged_eight=eight, unchanged_fold_sources=certified_eight,
                    handoff=handoff, result_files={str(path.relative_to(ROOT)): h.sha_file(path)
                                                for path in sorted(RESULTS.iterdir()) if path.is_file()})
    manifest.pop("last_error", None)
    h.write_json(MANIFEST, manifest)
    print("COMPLETE_AUDITED | 2 Best | 2 Latest | 2 exact next steps | 8 read-only folds | verified handoff", flush=True)
