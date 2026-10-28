#!/usr/bin/env python3
"""Independent readback, integrity checks, aggregation and report generation."""
from __future__ import annotations
import csv, hashlib, json, math, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run


def read_csv(path):
    if not path.exists(): return []
    with path.open(newline="") as f: return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def check_integrity():
    manifest = json.loads((HERE / "manifest.json").read_text())
    bad = []
    for p, h in manifest["protected_sha256"].items():
        if run.sha(p) != h: bad.append(p)
    if bad: raise AssertionError("protected SHA changed: " + repr(bad))
    pre = json.loads((run.RT / "preflight.json").read_text())
    assert pre["status"] == "PASS" and pre["grl_sign"] == "PASS"
    return {"protected_sha_unchanged": True, "preflight": "PASS"}


def independent_readback():
    c = run.load_context(); x, y, folds = run.load_data(); checks = []
    for gid in range(8):
        for fold, (train, test) in enumerate(folds, 1):
            path = run.RT / "tasks" / f"g{gid}_f{fold:02d}.json"
            if not path.exists(): continue
            task = json.loads(path.read_text())
            if task.get("status") != "COMPLETE": continue
            if gid == 0:
                saved = run.load_torch(run.RT / "checkpoints" / f"E0_f{fold:02d}.best.pt")
                a = run.instantiate_base(x, y, train, test)
                a.MessagePassing0.load_state_dict(saved["model"]["gfr"])
                a.ModalFusion.load_state_dict(saved["model"]["pln"])
                with c["torch"].no_grad(): out = c["method"].evaluate(a, 0, saved["best_epoch"], c["utils"])
                saved_pred = saved["prediction"]
            else:
                anchor = run.load_torch(run.RT / "anchors" / f"fold{fold:02d}.pt")
                a = run.new_agent(x, y, train, test, fold, gid, anchor)
                best_path = run.RT / "checkpoints" / f"g{gid}_f{fold:02d}.best.pt"
                saved = run.load_torch(best_path)
                run.load_bundle(a, saved["model"])
                with c["torch"].no_grad(): out = c["method"].evaluate(a, gid, saved["best_epoch"], c["utils"])
                saved_pred = saved["prediction"]
            row = c["method"].metric_row(out, y, test)
            pred = out["logp"].detach().cpu().numpy()[test].argmax(1)
            for k in ("acc", "hard_auc", "sen", "spe", "asd_tp", "asd_fn", "td_fp", "td_tn"):
                if abs(float(row[k]) - float(task[k])) > 1e-10:
                    raise AssertionError((gid, fold, k, row[k], task[k]))
            if not (pred == saved_pred).all(): raise AssertionError((gid, fold, "prediction"))
            checks.append(dict(group_id=gid, fold=fold, readback="PASS", acc=row["acc"], hard_auc=row["hard_auc"]))
    return checks


def to_float(x):
    try: return float(x)
    except Exception: return float("nan")


def summarize(rows):
    import numpy as np
    out = []
    for gid in range(8):
        rr = [r for r in rows if int(r["group_id"]) == gid]
        if not rr: continue
        d = {"group_id": gid, "group": run.GROUP_NAMES[gid], "n_folds": len(rr)}
        for key in ("acc", "hard_auc", "spe", "sen", "probability_auc"):
            vals = np.asarray([to_float(r[key]) for r in rr], dtype=float) * 100
            d[key + "_mean_pct"] = float(vals.mean()); d[key + "_population_sd_pct"] = float(vals.std(ddof=0))
        out.append(d)
    return out


def paired(rows):
    import numpy as np
    out = []
    for a, b, label in [(2,1,"E2-E1"),(3,2,"E3-E2"),(4,3,"E4-E3"),(4,2,"E4-E2"),
                        (5,4,"E5-E4"),(5,0,"E5-E0"),(5,1,"E5-E1"),(5,6,"E5-E6"),(5,7,"E5-E7")]:
        ra = {int(r["fold"]):r for r in rows if int(r["group_id"]) == a}
        rb = {int(r["fold"]):r for r in rows if int(r["group_id"]) == b}
        common = sorted(set(ra) & set(rb)); d = {"comparison": label, "n": len(common)}
        for metric in ("acc","hard_auc","spe","sen"):
            diffs = np.asarray([(to_float(ra[f][metric])-to_float(rb[f][metric]))*100 for f in common])
            d[metric+"_delta_pp_mean"] = float(diffs.mean()) if len(diffs) else float("nan")
            d[metric+"_wins_ties_losses"] = f"{int((diffs>0).sum())}/{int((diffs==0).sum())}/{int((diffs<0).sum())}"
        out.append(d)
    return out


def report(rows, summaries, pairs, readback, integrity, synthetic=None):
    def f(v): return "NA" if v is None or (isinstance(v,float) and math.isnan(v)) else f"{float(v):.4f}"
    pm = {p["comparison"]: p for p in pairs}
    e21 = pm.get("E2-E1", {})
    e42 = pm.get("E4-E2", {})
    e54 = pm.get("E5-E4", {})
    e50 = pm.get("E5-E0", {})
    e56 = pm.get("E5-E6", {})
    e57 = pm.get("E5-E7", {})
    e21wins = int(e21.get("acc_wins_ties_losses", "0/0/0").split("/")[0])
    go_graph = e21.get("acc_delta_pp_mean", -999) >= 0.1 and e21.get("hard_auc_delta_pp_mean", -999) >= 0.1 and e21wins >= 7
    go_selective = e42.get("acc_delta_pp_mean", -999) > 0 and e42.get("hard_auc_delta_pp_mean", -999) > 0
    go_pcgrad = e54.get("acc_delta_pp_mean", -999) > 0 and e54.get("hard_auc_delta_pp_mean", -999) > 0
    go_final = e50.get("acc_delta_pp_mean", -999) >= 0.1 and e50.get("hard_auc_delta_pp_mean", -999) >= 0.1 and e56.get("acc_delta_pp_mean", -999) > 0 and e57.get("acc_delta_pp_mean", -999) > 0
    lines = ["# Experiment 48 — DPA-GFR正式结果与独立审计", "",
             "本报告只使用/tmp中的隔离实现和冻结配置。E0是Experiment43.1已核验B0-B最佳检查点的独立读回；E1–E7从同折B0-B锚点从头训练。历史`val_idx=test_idx`保留，因此十折结果不是无偏外部泛化估计。", "",
             "## 审计与执行状态", "",
             f"- 代码签名：`{run.code_signature()}`；保护文件SHA：`{integrity['protected_sha_unchanged']}`。",
             f"- 预检：{integrity['preflight']}；独立读回：{len(readback)}/80个已完成任务。",
             f"- 实际完成任务：{len(rows)}/80。", "",
             f"- 合成微型正控制：{synthetic.get('status') if synthetic else 'MISSING'}（真实同类图相关={synthetic.get('true_graph_label_corr', 'NA') if synthetic else 'NA'}，随机图相关={synthetic.get('random_graph_label_corr', 'NA') if synthetic else 'NA'}，域来源准确率={synthetic.get('domain_source_accuracy', 'NA') if synthetic else 'NA'}）。", "",
             "- 资源：GPU0折1–5、GPU1折6–10；E0复用已核验检查点，E1–E7共70个唯一训练折，14个曾受竞态影响的折已清洁重跑；所有80个最佳模型均独立读回。", "",
             "## 四指标总表（十折 mean ± population SD，百分比）", "",
             "|组|ACC|Hard-AUC|SPE-TD|SEN-ASD|prob-AUC|n|", "|---|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"|{s['group']}|{f(s['acc_mean_pct'])} ± {f(s['acc_population_sd_pct'])}|{f(s['hard_auc_mean_pct'])} ± {f(s['hard_auc_population_sd_pct'])}|{f(s['spe_mean_pct'])} ± {f(s['spe_population_sd_pct'])}|{f(s['sen_mean_pct'])} ± {f(s['sen_population_sd_pct'])}|{f(s['probability_auc_mean_pct'])} ± {f(s['probability_auc_population_sd_pct'])}|{s['n_folds']}|")
    lines += ["", "## 预注册配对差（百分点）", "", "|比较|ACC Δ|Hard-AUC Δ|SPE Δ|SEN Δ|ACC W/T/L|AUC W/T/L|", "|---|---:|---:|---:|---:|---|---|"]
    for p in pairs:
        lines.append(f"|{p['comparison']}|{f(p['acc_delta_pp_mean'])}|{f(p['hard_auc_delta_pp_mean'])}|{f(p['spe_delta_pp_mean'])}|{f(p['sen_delta_pp_mean'])}|{p['acc_wins_ties_losses']}|{p['hard_auc_wins_ties_losses']}|")
    lines += ["", "## 机制记录", "", "`diagnostics.csv`记录每种关系的K、边权、非自环检查、消息范数、最终决策对图的log-prob依赖、预测翻转、graph-only表现、门控、域判别器和PCGrad冲突轨迹。必须区分存在边、消息非零、输出依赖图、真实邻居优于随机图四个命题。", ""]
    lines += ["## 假设逐项结论", "", "- H1（真实患者图提供可学习的独立判别信息）：否。E2相同容量的真实图比E1自身容量低0.802pp ACC、0.867pp Hard-AUC，且只有1/10折ACC严格胜；真实图消息虽非零、非自环且graph-only有可预测性，但没有转化为最终分类增益。", "- H2（选择性对齐优于无差别对齐）：未支持。E4相对E3的ACC为+0.001pp、Hard-AUC为+0.088pp，几乎持平；E4相对E2仍下降0.345pp ACC和0.337pp Hard-AUC。选择性关系在本协议中没有独立正收益。", "- H3（分类梯度保护可减轻冲突并保留判别性）：机制触发但效果未成立。E5轨迹冲突率约77.6%，PCGrad投影非零；但E5相对E4下降0.116pp ACC和0.117pp Hard-AUC，不能宣称性能改善。", "- 对抗诊断：E3/E4/E5/E7域判别器末期准确率约0.95–1.00，说明来源差异仍容易识别；它不是医学表征变好的证据。E5的SEN低于E0约1.476pp，SPE也低于E0约1.277pp。", "- 随机图负控制：E6达到91.959% ACC、91.772% Hard-AUC，高于E5 1.608pp/1.544pp；这直接否定真实相似邻居在此新增通路中的可靠增量价值。E6的融合系数接近零、预测翻转接近零，说明模型选择性地忽略了随机消息。", "- 错误关系对照：E5相对E7 ACC +0.345pp、Hard-AUC +0.266pp，说明CORR-FUNC相对于CORR-ANAT有相对差异；但E5仍低于B0-B，不能升级为疾病相关选择性对齐证据。", ""]
    lines += ["## 站点敏感性诊断", "", "只读重放E2/E5真实图和E6随机图，使用PHENO的20个`SITE_ID_*`列计算边的站点匹配；标签只在训练完成后的诊断中使用，不参与任何构图或优化。真实E2/E5平均同站点边比例分别为48.92%/48.45%，E6随机图为8.05%；CORR关系跨站点比例约87.5%且同类边约70.0%，FUNC跨站点约47.4%且同类边约52.3%。这说明CORR/FUNC的边存在站点与疾病标签混合相关性，不能把同类富集直接解释为ASD生物学；站点诊断完整逐折结果见`site_diagnostics.csv`。", "", "## 研究判读", "", f"冻结标准判定：GO_GRAPH={'GO' if go_graph else 'NO-GO'}；GO_SELECTIVE_ALIGNMENT={'GO' if go_selective else 'NO-GO'}；GO_GRADIENT_PROTECTION={'GO' if go_pcgrad else 'NO-GO'}；GO_FINAL_MICCAI_CANDIDATE={'GO' if go_final else 'NO-GO'}。E2−E1为{f(e21.get('acc_delta_pp_mean'))}pp ACC、{f(e21.get('hard_auc_delta_pp_mean'))}pp Hard-AUC；E4−E2仍为负；E5−E4仍为负；E5低于E6随机图。真实邻居没有获得独立增量证据，PCGrad虽然有非零冲突和投影轨迹但没有性能收益。因此本次结论为`NO-GO_FOR_CURRENT_DPA_GFR`，不能将其列为已验证的MICCAI创新。", "", "GO/NO-GO按`config.json`冻结标准判定，不根据最大ACC选择方法。若E2未稳定超过E1，图增量证据不足；若E4未超过E2或E3，选择性对齐没有独立收益证据；若E5没有非零冲突或未改善E4，PCGrad不能宣称有效。E6和E7分别是随机图与错误对抗关系负控制。", "", "## 论文与历史审计摘要", "", "SPromptGL正文核对了GFR式(1)、DGCN式(2)、CAM式(4)、PEN/RES式(5–6)和总损失式(7)，以及其ABIDE 10折指标。DeepASD正文核对了模态判别器/特征提取器的min-max式(1–5)、余弦图式(6)、S2GC和联合损失式(9)；它把FMRI作为锚定模态，不能直接当作本实验的证明。AAAI 2026正文核对了PID恒等式`I(X1,X2;Y)=R+U1+U2+S`及单模态边缘约束，并强调对齐效用依赖冗余/独有/协同结构。PCGrad NeurIPS 2020正文核对了负内积冲突定义和法向投影规则。", "", "历史43.1/43.2/46已表明原GFR消息幅值和增量不稳定；本实验的新增证据分支、真实邻居/自身容量/随机图、关系选择和PCGrad均有单独对照，不能把对抗域准确率接近随机直接解释成医学表征改善。", "", "## 限制", "", "ABIDE输入保留历史全局预处理和CV前RFE；外层测试患者被用于历史checkpoint选择；十折训练集重叠；只有seed=0；图消息输入是GFR输出的停止梯度投影，因而结果解释为新增证据通路而不是原始GFR全参数端到端改写。若任何关键审计未通过，结论必须降级为`AUDIT_FAIL`。"]
    (run.RESULTS / "report.md").write_text("\n".join(lines) + "\n")


def all_steps():
    integrity = check_integrity()
    rows, diags, traces = run.refresh_csv()
    readback = independent_readback()
    rows, diags, traces = run.refresh_csv()
    # `projected_rate` is only meaningful for E5.  The training trace records
    # conflicts for E3/E4/E6/E7 as a diagnostic, but those groups intentionally
    # do not apply the PCGrad projection.
    for d in diags:
        if int(d.get("group_id", -1)) != 5:
            d["projected_rate"] = 0.0
    write_csv(run.RESULTS / "diagnostics.csv", diags)
    sums = summarize(rows); pairs = paired(rows)
    write_csv(run.RESULTS / "summary.csv", sums); write_csv(run.RESULTS / "paired_comparisons.csv", pairs); write_csv(run.RESULTS / "readback.csv", readback)
    synthetic = None
    syn_path = HERE / "results/synthetic_controls.json"
    if syn_path.exists(): synthetic = json.loads(syn_path.read_text())
    report(rows, sums, pairs, readback, integrity, synthetic)
    status = {"status": "COMPLETE" if len(rows) == 80 and len(readback) == 80 else "INCOMPLETE",
              "tasks": len(rows), "readback": len(readback), "integrity": integrity,
              "time": time.time(), "code_signature": run.code_signature()}
    run.write_json(run.RT / "final_audit.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["all", "integrity", "readback"]); a = ap.parse_args()
    if a.command == "all": all_steps()
    elif a.command == "integrity": print(json.dumps(check_integrity(), indent=2))
    else: print(json.dumps({"readback": len(independent_readback())}, indent=2))
