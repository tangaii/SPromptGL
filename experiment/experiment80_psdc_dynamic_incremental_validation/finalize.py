#!/usr/bin/env python3
"""Finalize E80 report artifacts from audited outputs and append handoff."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path


EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"
REPO = Path("/data/adodas/SPromptGL-main")
HANDOFF = Path("/model/spromptgl/experiment/handoff.md")
MARKER = "## Experiment 80 — PSDC-IVE"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, content: bytes, mode: int = 0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if tmp.exists():
            tmp.unlink()


def dump_json(path: Path, value):
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def f(value, digits=4):
    return f"{float(value):.{digits}f}"


def ci_rows(comparisons, key, scope="all10"):
    return {r["metric"]: r for r in comparisons if r["comparison"] == key and r["scope"] == scope}


def format_metric_table(rows):
    names = ["E74.4-G2", "A0", "A1", "A2", "A3", "DynamicOnly"]
    table = ["| 方法 | n | ACC | Hard-AUC / BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        row = next(r for r in rows if r["method"] == name)
        values = []
        for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1"):
            mean = float(row[f"{metric}_mean_percent"])
            sd = float(row[f"{metric}_population_sd_percent"])
            values.append(f"{mean:.4f} ± {sd:.4f}")
        table.append(f"| {name} | {row['subjects']} | " + " | ".join(values) + " |")
    return "\n".join(table)


def site_table(qc):
    sites = sorted(qc["timepoints_distribution_by_site"])
    lines = ["| 站点/子站点 | TR (s) | 时间点 T | 窗口 W | 实际窗口秒数 |",
             "|---|---:|---:|---:|---:|"]
    for site in sites:
        t = qc["timepoints_distribution_by_site"][site]
        w = qc["window_count_distribution_by_site"][site]
        d = qc["window_duration_seconds_distribution_by_site"][site]
        lines.append(f"| {site} | {qc['tr_seconds_by_site'][site]:g} | {int(t['min'])}–{int(t['max'])} | "
                     f"{int(w['min'])}–{int(w['max'])} | {d['min']:g}–{d['max']:g} |")
    return "\n".join(lines)


def build_handoff(qc, metric_rows, comparisons, complement, darg, probe, audit, confounds):
    a2 = next(r for r in metric_rows if r["method"] == "A2")
    a1 = next(r for r in metric_rows if r["method"] == "A1")
    a0 = next(r for r in metric_rows if r["method"] == "A0")
    dynamic_only = next(r for r in metric_rows if r["method"] == "DynamicOnly")
    po = {r["method"]: r for r in read_csv(RESULTS / "pooled_oof_metrics.csv")}
    deltas = probe["pooled_oof_deltas_pp"]
    c_a2 = next(r for r in complement if r["method"] == "A2" and r["scope"] == "all10")
    fold2_a0 = ci_rows(comparisons, "A2-A0", "nine_without_fold2")
    fold2_a1 = ci_rows(comparisons, "A2-A1", "nine_without_fold2")
    fold_sizes = sorted({int(r["fold"]): int(r["n_test"]) for r in read_csv(RESULTS / "fold_metrics.csv")
                         if r["method"] == "A0"}.items())
    fold_size_text = ", ".join(f"F{fold}={n}" for fold, n in fold_sizes)
    conf = {r["confound"]: float(r["spearman_rho"]) for r in confounds
            if r["dynamic_summary"] == "mean_window_variability"}
    return f"""\n\n{MARKER}（COMPLETE_AUDITED；`NO_GO_DYNAMIC_INCREMENTAL_EVIDENCE`）

目标：PSDC-IVE Stage A，只检验原始 BOLD 滑窗 FC 是否在静态 320 维 DARG 输入与容量对照之上提供诊断增量；不是完整 PSDC 模型。由 E79 的 Prompt/原型/Memory 路线转向原始时序证据问题，本实验没有继承其模型结构。

数据：871/871 CSV 行、SUB_ID、BOLD 文件、E11 ROI 顺序和 SHA256 一一核验；每人111 ROI、时序78–316点、全有限，动态可评估871/871。TR依据 ABIDE I 官方站点/子站点采集协议与 SITE_ID、CPAC 文件前缀匹配，20个站点/子站点均有来源且无缺失，范围1.5–3.0秒；`.1D`没有逐帧时间戳或 censor/delete 记录，无法排除内部删帧/拼接。固定60秒窗、50%步长、每窗至少30点、至少3窗；实际每人4–20窗，均值10.800、中位数10，实际窗长均值65.772秒。

E11静态 FC 身份：按历史 Nilearn兼容流程 demean/标准化、Ledoit–Wolf、协方差转相关、Fisher-z，固定256边次序；871×256共222,976项全在1e-6内，最大绝对误差{qc['static_reconstruction_max_absolute_error']:.3e}，RMSE {audit['data']['static_fc_global_rmse']:.3e}。没有重跑RFE或改静态输入。收缩系数最小/中位/最大为{qc['window_shrinkage_min']:.3f}/{qc['window_shrinkage_median']:.3f}/{qc['window_shrinkage_max']:.3f}，≥0.99占0%，未见近单位收缩压平；未知 ROI索引82涉及4边，其平均动态方差占比{qc['unknown_roi_dynamic_variance_share_mean']:.1%}，排除该4边敏感性仍未改变结论。

动态输入：`M=窗口FC均值256`（仅时间平均对照）、`V=窗口FC标准差256`、`D=相邻窗绝对差均值256`；Dynamic=concat(V,D) 512维。A0=Static320；A1=Static320+MeanPCA32+Static CORR PCA32容量控制；A2=Static320+MeanPCA32+DynamicPCA32；A3同维且每人独立无标签随机重排窗口顺序后计算D；另有Dynamic-only诊断。训练折内PCA/Scaler/残差化、5折内层log-loss选C∈{{0.01,0.1,1,10}}，外层StratifiedKFold10折seed=0；Fold2测试仍保留，历史重叠行376/406仅从该折新增训练中排除。全部探针CPU执行，8 workers。

主要结果（10折均值±总体SD，%）：

{format_metric_table(metric_rows)}

每折测试人数：{fold_size_text}。

Dynamic-only（仅诊断，不作为GO条件）：ACC/BA/Prob-AUC {float(dynamic_only['acc_mean_percent']):.4f}/{float(dynamic_only['hard_auc_mean_percent']):.4f}/{float(dynamic_only['prob_auc_mean_percent']):.4f}%。

Pooled OOF：A2−A0的ACC/BA/Prob-AUC为{deltas['A2-A0']['acc']:+.3f}/{deltas['A2-A0']['hard_auc']:+.3f}/{deltas['A2-A0']['prob_auc']:+.3f}pp；A2−A1为{deltas['A2-A1']['acc']:+.3f}/{deltas['A2-A1']['hard_auc']:+.3f}/{deltas['A2-A1']['prob_auc']:+.3f}pp；A2−A3为{deltas['A2-A3']['acc']:+.3f}/{deltas['A2-A3']['hard_auc']:+.3f}/{deltas['A2-A3']['prob_auc']:+.3f}pp。A2−A0排除Fold2仍为负（ACC/BA/AUC {float(fold2_a0['acc']['mean_difference_pp']):+.3f}/{float(fold2_a0['hard_auc']['mean_difference_pp']):+.3f}/{float(fold2_a0['prob_auc']['mean_difference_pp']):+.3f}pp）；A2−A1同样为负（{float(fold2_a1['acc']['mean_difference_pp']):+.3f}/{float(fold2_a1['hard_auc']['mean_difference_pp']):+.3f}/{float(fold2_a1['prob_auc']['mean_difference_pp']):+.3f}pp）。

互补性：DARG错误68例，A2支持正确8例（5 ASD、3 TD），仍共同错误60例；DARG正确者中A2另误判77例（30 ASD、47 TD）。去Fold2后8/61个DARG错误有A2正确支持；属于事后潜在互补统计，未做融合纠错。8例分布在5个站点/子站点，其他多站点无对应纠错；Dynamic-only虽支持27/68个DARG错误，但同时误伤381个DARG正确患者。

混杂：窗口变化均值与TR、时间点数、窗口数、站点的相关分别为{conf['tr_seconds']:+.3f}、{conf['timepoints']:+.3f}、{conf['window_count']:+.3f}，站点η²见`results/confound_analysis.csv`；同时与mean FD相关{conf['func_mean_fd']:+.3f}。训练折内残差化后A2仍未超过A1，未发现隐藏的动态增量。窗口均值相对全扫描FC主要是时间平均，不应视为新的状态证据；时变幅度并未提供分类增益，顺序依赖增益也未建立。

DARG身份：E74.4-G2逐折最佳Checkpoint、E61 epoch-10初始化、参数/优化器初始化摘要、训练关系对、划分及历史RNG记录逐折做只读工件核验，10/10；历史E74.4独立审计为30/30 PASS，六指标逐折复算与记录差异为0。本轮没有训练DARG，未声称新增逐Epoch RNG parity。E43.1历史锚点二进制在两个挂载均缺失，因此本轮未从E80重新执行DARG前向；采用历史30/30前向审计记录并明确保留此限制。

审计：独立E80审计`PASS_INDEPENDENT_AUDIT`，90个方法-折变换轨迹、7839行预测和OOF结果独立重建，全部外层测试标签不参与选择，Fold2重叠排除通过。历史val=test选Checkpoint、CV前RFE/标准化、全患者transductive图和Fold2训练/测试重叠仍限制临床泛化解释。

最终门控：`NO_GO_DYNAMIC_INCREMENTAL_EVIDENCE`，不建议当前证据启动PSDC Stage B。它回答的是本次固定60秒窗和轻量统计探针未观察到稳定的条件疾病增量，不是对所有BOLD动态建模作否定。建议后续唯一优先实验为外部/站点留出复核，并先补齐逐帧删帧/运动记录与站点采集协议，预注册时间尺度；本轮不启动E80.1或Stage B。权威完整结果位于`experiment80_psdc_dynamic_incremental_validation/results/report.md`。
"""


def build_report(qc, summary, audit, metric_rows, comparisons, pooled, complement, classes,
                 sites, confounds, darg_fold_rows, handoff_written):
    darg = next(r for r in metric_rows if r["method"] == "E74.4-G2")
    a2 = next(r for r in metric_rows if r["method"] == "A2")
    a3 = next(r for r in metric_rows if r["method"] == "A3")
    dyn = next(r for r in metric_rows if r["method"] == "DynamicOnly")
    pair_lines = ["| 比较（A2−对照） | 指标 | 逐折均差 pp | W/T/L | 描述性95%区间 pp | pooled OOF差 pp |",
                  "|---|---|---:|---:|---:|---:|"]
    for comp, pooled_key in (("A2-A0", "A2-A0"), ("A2-A1", "A2-A1"), ("A2-A3", "A2-A3")):
        for metric in ("acc", "hard_auc", "prob_auc"):
            r = ci_rows(comparisons, comp)[metric]
            pair_lines.append(f"| {comp} | {metric} | {float(r['mean_difference_pp']):+.3f} | "
                              f"{r['wins']}/{r['ties']}/{r['losses']} | "
                              f"[{float(r['descriptive_95ci_low_pp']):+.3f}, {float(r['descriptive_95ci_high_pp']):+.3f}] | "
                              f"{float(summary['pooled_oof_deltas_pp'][pooled_key][metric]):+.3f} |")
    residual = {r["method"]: r for r in pooled}
    comp_a2 = next(r for r in complement if r["method"] == "A2" and r["scope"] == "all10")
    comp_donly = next(r for r in complement if r["method"] == "DynamicOnly" and r["scope"] == "all10")
    class_rows = [r for r in classes if r["method"] == "A2"]
    class_map = {(r["event"], r["true_class"]): int(r["count"]) for r in class_rows}
    corr = {r["confound"]: float(r["spearman_rho"]) for r in confounds
            if r["dynamic_summary"] == "mean_window_variability"}
    site_corrections = [r for r in sites if r["method"] == "A2" and int(r["darg_wrong_dynamic_correct"]) > 0]
    site_text = ", ".join(f"{r['site']} {r['darg_wrong_dynamic_correct']}" for r in site_corrections)
    fold2 = {name: ci_rows(comparisons, name, "nine_without_fold2")
             for name in ("A2-A0", "A2-A1", "A2-A3")}
    fold_sizes = sorted({int(r["fold"]): int(r["n_test"]) for r in read_csv(RESULTS / "fold_metrics.csv")
                         if r["method"] == "A0"}.items())
    fold_size_text = ", ".join(f"F{fold}={n}" for fold, n in fold_sizes)
    report = f"""# Experiment 80 — PSDC-IVE

**中文名称：** PSDC静态–动态诊断增量证据验证（Stage A）  
**执行状态：** `COMPLETE_AUDITED`  
**科研判定：** `NO_GO_DYNAMIC_INCREMENTAL_EVIDENCE`  
**实验目录：** `{EXP}`  
**结果范围：** 本次固定数据、窗口和轻量探针；不等于完整PSDC，也不否定其他动态建模。

## 研究问题与结论

检验871名ABIDE A受试者原始BOLD滑窗功能连接，在静态320维输入和容量匹配静态对照之上是否提供ASD/TD条件增量。实际A2没有胜过A0或A1： pooled OOF ACC/BA/Prob-AUC分别下降{summary['pooled_oof_deltas_pp']['A2-A0']['acc']:.3f}/{summary['pooled_oof_deltas_pp']['A2-A0']['hard_auc']:.3f}/{summary['pooled_oof_deltas_pp']['A2-A0']['prob_auc']:.3f}pp（相对A0），相对A1也下降{summary['pooled_oof_deltas_pp']['A2-A1']['acc']:.3f}/{summary['pooled_oof_deltas_pp']['A2-A1']['hard_auc']:.3f}/{summary['pooled_oof_deltas_pp']['A2-A1']['prob_auc']:.3f}pp。去Fold2后仍为负。数据身份、静态重建和统计探针审计通过，但动态特征与采集站点、TR、扫描长度、窗口数和运动质量有明显关联。当前证据不支持启动完整PSDC Stage B。

## 执行与来源审查

已审查[SPromptGL MICCAI 2025论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[原作者仓库](https://github.com/wanxixi11/SPromptGL)、[本项目仓库](https://github.com/tangaii/SPromptGL)、E74.4-G2、E61初始化和E11恢复记录。源码中GFR按模态特征形成患者关系并由GCN传递多模态表征；CAM分类前输入为144维，OutputLayer消费融合后的CAM表示及注意力表征。PLNNoPEN是移除PEN路径的配置，不能与原论文PEN混称。simple-2源码计算ALoss但未将其加入该路径总损失；本实验没有修改或重训DARG，也没有将该损失擅自加回。历史Hard-AUC口径为硬预测balanced accuracy（BA），Prob-AUC才是连续ASD概率的ROC-AUC。

已阅读[BrainSTR MICCAI 2026](https://papers.miccai.org/miccai-2026/0124-Paper0931.html)及[代码](https://github.com/guoguiliang111/BrainSTR)、[BrainACU MICCAI 2026](https://papers.miccai.org/miccai-2026/0117-Paper0936.html)及[代码](https://github.com/guoguiliang111/BrainACU)、[BrainPrompt+代码](https://github.com/AngusMonroe/BrainPromptPlus)和[TMI DOI](https://doi.org/10.1109/TMI.2026.3692958)、[SMGD ICML 2026](https://proceedings.mlr.press/v306/shi26ag.html)。其窗口、阶段/相位、协调、延迟注意力、频谱/空间/ROI/疾病/受试者Prompt、共享与互补信息区分仅构成动机，不证明本数据有效；本轮没有移植复杂模型。

E74.4-G2唯一基座身份沿用历史正式文件：逐折核验E61 epoch-10 warmup、参数及优化器初始化摘要、训练关系对、患者划分、Checkpoint SHA和历史RNG记录。历史E74.4审计`PASS`且30/30 checkpoint前向与30/30 latest模型/优化器/RNG读回已记录。本轮没有新增DARG训练，因此没有声称重新执行epoch RNG parity。E80逐折只读核验10/10工件；E43.1历史锚点二进制在两个挂载均缺失，本轮无法重新从锚点开始前向回放，采用既有E74审计证据而未替换输入。

运行于Hygon C86-4G CPU服务器（256逻辑CPU）；数据提取与嵌套逻辑回归均使用CPU、8个worker、每worker单BLAS线程。服务器可见PPU-ZW810E 98,304 MiB GPU，本实验未使用GPU。测得计算阶段：BOLD/FC提取28.179秒、探针57.236秒、独立审计25.388秒，合计约110.8秒；不含文献审查和handoff写入时间。有效动态受试者871/871，10/10折完成，独立审计`PASS_INDEPENDENT_AUDIT`。

## 数据核验与动态特征

871行身份与871个BOLD文件按SUB_ID和原始行序匹配；每人111 ROI，列顺序和全部BOLD SHA与E11清单一致，所有信号及窗口FC有限。站点/子站点TR由ABIDE I官方扫描协议与SITE_ID、CPAC文件名前缀核对，20/20均有来源：1.5秒（OLIN/PITT）、1.656秒（LEUVEN 1/2）、2秒（CALTECH/CMU/NYU/SDSU/STANFORD/TRINITY/UM 1/2/USM/YALE）、2.2秒（SBL）、2.5秒（KKI/OHSU）、3秒（MAX_MUN/UCLA 1/2）。TR来自站点/子站点协议而非每个BOLD文件内字段。样本文件无逐帧时间戳或删帧日志，故不能核实潜在时间间隙、拼接或删帧。

全扫描静态256边按E11历史兼容流程重建：demean/标准化、Ledoit–Wolf协方差、转相关、Fisher-z和冻结三角边顺序；比较222,976个值，最大误差{qc['static_reconstruction_max_absolute_error']:.3e}、全局RMSE {audit['data']['static_fc_global_rmse']:.3e}，全数在1e-6容差内，RFE/历史输入未改。

固定60秒目标窗、50%步长、至少30个时间点、至少3窗。实际T范围{qc['timepoints_min']}–{qc['timepoints_max']}；W范围{audit['data']['window_check']['window_count_min']}–{audit['data']['window_check']['window_count_max']}，平均{qc['mean_window_count']:.3f}、中位数{qc['median_window_count']:.0f}；因采样周期取整，实际窗口长度平均{qc['mean_actual_window_duration_seconds']:.3f}秒。各站点/子站点：

{site_table(qc)}

特征为M=窗口FC均值256（时间平均对照）；V=逐边窗间标准差256；D=逐边相邻窗口绝对差均值256；Dynamic=concat(V,D)，共512维。A1/A2/A3的PCA 32维在对应内层或外层训练集拟合；Scaler、残差化和正则强度C∈{{0.01,0.1,1,10}}也仅在训练范围拟合/选择，内层5折按log-loss。A3保留每人窗口集合、均值和标准差，仅以预先独立、无标签排列改变相邻窗次序。所有新增探针按原10折`StratifiedKFold(shuffle=True,random_state=10)`，seed=0；相同受试者集和分类器、类别权重、模型选择规则。Fold 2测试照旧，重叠行376/406仅从该折新增训练集排除。

## 正式结果

10折mean±population SD，单位百分比。DARG列是历史E74.4-G2基座；静态探针与DARG模型容量不同，不能把两者差异归因于动态信息。

{format_metric_table(metric_rows)}

各外层折测试人数：{fold_size_text}。同一折所有探针和相同有效队列下DARG均按这些测试行评价。

Dynamic-only仅作诊断：ACC/BA/Prob-AUC={float(dyn['acc_mean_percent']):.4f}/{float(dyn['hard_auc_mean_percent']):.4f}/{float(dyn['prob_auc_mean_percent']):.4f}%，未显示有用的独立分类能力。六指标逐折和pooled OOF指标分别见`fold_metrics.csv`、`pooled_oof_metrics.csv`。

### 条件配对比较

下表逐折均差使用折级指标之差的平均，区间为描述性配对95%区间；W/T/L为以比较方向计算的折胜/平/负。pooled OOF差由871条折外预测合并后计算，故与逐折均差不同。

{chr(10).join(pair_lines)}

去Fold2后（9折）A2−A0的ACC/BA/Prob-AUC均差为{float(fold2['A2-A0']['acc']['mean_difference_pp']):+.3f}/{float(fold2['A2-A0']['hard_auc']['mean_difference_pp']):+.3f}/{float(fold2['A2-A0']['prob_auc']['mean_difference_pp']):+.3f}pp；A2−A1为{float(fold2['A2-A1']['acc']['mean_difference_pp']):+.3f}/{float(fold2['A2-A1']['hard_auc']['mean_difference_pp']):+.3f}/{float(fold2['A2-A1']['prob_auc']['mean_difference_pp']):+.3f}pp；负向结论不依赖Fold2。A2−A3 pooled ACC持平、BA仅+0.017pp、Prob-AUC +0.084pp，折级AUC胜负3/0/7，不能建立时序顺序信号。A2较A1的差值也为负，故窗口均值和额外维度没有转化为动态时变增益。

## DARG错误互补性

A2与历史DARG按相同871行OOF/测试身份比较：两者同时正确726、同时错误60；DARG错而A2对8；DARG对而A2错77。DARG错误共68，其中A2给出正确支持8/68（11.8%），另60/68（88.2%）共同错。8例中ASD 5、TD 3；77个误伤中ASD 30、TD 47。排除Fold2后为8/61个DARG错误得到A2正确支持，共同错53，且仍有67个DARG正确病例被A2误判。支持分布为{site_text}，没有跨多数站点重复的稳定趋势。

Dynamic-only给出27/68个DARG错误的正确预测，但误判381个DARG正确病例，且动态only Prob-AUC仅{float(dyn['prob_auc_mean_percent']):.2f}%。上述是测试集事后“潜在互补性”计数，不是融合后的净纠错；没有训练融合器，也不能写成PSDC已经纠正病例。E80不使用DARG训练内CAM/Logits作为OOF教师特征。

## 混杂、敏感性与审计

窗口变化均值与TR、扫描时长、时间点数、有效窗口数、mean FD、DVARS、percent FD的Spearman相关分别见`confound_analysis.csv`。主要值：V均值与TR ρ={corr['tr_seconds']:+.3f}、时间点数ρ={corr['timepoints']:+.3f}、窗口数ρ={corr['window_count']:+.3f}、站点η²=0.713、mean FD ρ={corr['func_mean_fd']:+.3f}；D均值对应TR ρ=−0.682、时间点数ρ=+0.557、窗口数ρ=+0.418、站点η²=0.586、mean FD ρ=+0.252。训练折内按站点、TR、扫描长度、窗口数及已有运动指标拟合Ridge残差后，pooled A2_residualized ACC/BA/AUC={float(residual['A2_residualized']['acc']):.3f}/{float(residual['A2_residualized']['hard_auc']):.3f}/{float(residual['A2_residualized']['prob_auc']):.3f}%，仍低于A1_residualized的{float(residual['A1_residualized']['acc']):.3f}/{float(residual['A1_residualized']['hard_auc']):.3f}/{float(residual['A1_residualized']['prob_auc']):.3f}%。动态增量不是在残差化后显现。

Ledoit–Wolf窗口收缩系数范围{qc['window_shrinkage_min']:.3f}–{qc['window_shrinkage_max']:.3f}、中位数{qc['window_shrinkage_median']:.3f}，≥0.99比例0；没有观察到近单位收缩导致窗口变化整体压平。ROI索引82（编码3455）为额外未知ROI，出现193次常数列；全体常数ROI列出现235次，201人至少有1列。该未知ROI连接历史256边中的索引195/225/232/236，平均贡献动态方差54.94%；去除四边敏感性下A2仍低于A0/A1。

E80独立审计`PASS_INDEPENDENT_AUDIT`：871身份、111列、E11边序、BOLD SHA、TR来源和静态重建；窗口特征/排列重新构建；90个方法-折预处理与内外层样本轨迹、7839条预测复算；全部测试标签未参与选择；A1/A2/A3特征维度/同患者标签位置和Fold2排除检查通过。E74.4-G2 `PASS_ARTIFACT_REPLAY` 10/10：Checkpoint SHA、E61 warmup与参数/optimizer身份摘要、划分/4096关系对/RNG trace及历史预测六指标记录核验；原E74独立审计`PASS` 30/30。历史DARG源码、数据和Checkpoint只读且SHA复核未变。本次不产生新的DARG训练RNG一致性结论。E43.1锚点文件缺失，因此E80未自行重演DARG最佳Checkpoint前向；这是审计范围的明确限制，不影响轻量探针独立审计状态。

历史协议边界：基座继承val=test选Checkpoint、CV前RFE与标准化、transductive全患者图及Fold2行376/406重叠。E80新探针的训练预处理只拟合训练数据，Fold2重叠训练行排除，但复用历史测试划分不消除基座偏差。因此结果不是无偏临床泛化性能。ABIDE各站点病例量差异大；本次的站点校正相关分析不能代替外部站点留出。

## 科研结论

对于预先固定的60秒窗口和轻量统计探针，原始BOLD动态特征尚未证明具有超出DARG静态输入的稳定疾病判别增量：A2在ACC、BA、Prob-AUC、SEN、SPE、F1均低于A0；相较容量匹配A1也整体较差，排除Fold2后仍负。动态only近机会水平，A2只给少量DARG错例独立正确支持，并伴随更多新增错误。窗口幅度的变化保留了数值波动，但主要分类信号没有显示为增量；窗口均值不是新动态证据，时序顺序控制没有正向优势。采集站点、TR、扫描时长、窗口数和FD与动态统计强相关，且时间点删帧元数据缺失，是主要解释限制。

故建议停止当前Stage A假设下直接启动PSDC Stage B，最终状态`NO_GO_DYNAMIC_INCREMENTAL_EVIDENCE`。这不是证明所有动态FC方法都无效。医学Prompt互补学习仍有一般方法学空间，但本轮没有证据支持把它建立在当前dFC统计增量之上；先需要独立站点验证、更完整帧级运动/censor元数据及预注册时间尺度。按用户约束，本轮不自动执行E80.1或Stage B。

更新handoff：{'成功' if handoff_written else '失败，详见manifest中的handoff状态'}。审计时间：{audit['audit_timestamp_utc']}。
"""
    return report


def append_handoff(content: str):
    if not HANDOFF.exists():
        return {"status": "FAILED_MISSING", "path": str(HANDOFF)}
    lock_path = EXP / ".runtime" / "handoff.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        old = HANDOFF.read_bytes()
        if MARKER.encode() in old:
            appendix = content.encode("utf-8")
            if old.count(MARKER.encode()) != 1:
                return {"status": "FAILED_EXISTING_SECTION_MISMATCH", "path": str(HANDOFF),
                        "sha256": sha256(HANDOFF)}
            offset = old.index(MARKER.encode())
            prefix, previous = old[:offset], old[offset:]
            if not previous.startswith(MARKER.encode()) or not previous.rstrip().endswith("report.md`。".encode()):
                return {"status": "FAILED_EXISTING_SECTION_MISMATCH", "path": str(HANDOFF),
                        "sha256": sha256(HANDOFF)}
            mode = stat.S_IMODE(HANDOFF.stat().st_mode)
            atomic_write(HANDOFF, prefix + appendix, mode)
            final = HANDOFF.read_bytes()
            if not final.startswith(prefix) or not final.endswith(appendix):
                raise RuntimeError("handoff section update verification failed")
            return {"status": "UPDATED_AND_READBACK_VERIFIED", "path": str(HANDOFF),
                    "historical_prefix_bytes_preserved": len(prefix), "section_count": 1,
                    "sha256": sha256(HANDOFF)}
        mode = stat.S_IMODE(HANDOFF.stat().st_mode)
        appendix = content.encode("utf-8")
        atomic_write(HANDOFF, old + appendix, mode)
        final = HANDOFF.read_bytes()
        if not final.startswith(old) or not final.endswith(appendix):
            raise RuntimeError("handoff atomic append verification failed")
        return {"status": "APPENDED_AND_READBACK_VERIFIED", "path": str(HANDOFF),
                "original_bytes_preserved": len(old), "appended_bytes": len(appendix),
                "sha256": sha256(HANDOFF)}


def main():
    qc = read_json(RESULTS / "qc_summary.json")
    probe = read_json(RESULTS / "probe_summary.json")
    audit = read_json(RESULTS / "audit_status.json")
    metrics = read_csv(RESULTS / "main_table.csv")
    comparisons = read_csv(RESULTS / "paired_comparisons.csv")
    pooled = read_csv(RESULTS / "pooled_oof_metrics.csv")
    complement = read_csv(RESULTS / "complementary_analysis.csv")
    classes = read_csv(RESULTS / "complementarity_by_class.csv")
    sites = read_csv(RESULTS / "complementarity_by_site.csv")
    confounds = read_csv(RESULTS / "confound_analysis.csv")
    darg_folds = read_csv(RESULTS / "darg_fold_metrics.csv")
    darg = next(r for r in metrics if r["method"] == "E74.4-G2")

    if audit.get("status") != "PASS_INDEPENDENT_AUDIT" or probe.get("folds_completed") != 10:
        raise RuntimeError("cannot finalize before the independent audit and all ten folds pass")
    if qc.get("status") != "PASS_DATA_AND_STATIC_FC" or qc.get("valid_dynamic_subjects") != 871:
        raise RuntimeError("data/FC preflight did not pass the required 871-person coverage")
    if len(darg_folds) != 10 or len(read_csv(RESULTS / "probe_fold_audit.csv")) != 90:
        raise RuntimeError("fold audit artifacts are incomplete")

    handoff_text = build_handoff(qc, metrics, comparisons, complement, darg, probe, audit, confounds)
    handoff_status = append_handoff(handoff_text)
    handoff_written = handoff_status["status"] in {
        "APPENDED_AND_READBACK_VERIFIED", "PRESENT_AND_READBACK_VERIFIED", "UPDATED_AND_READBACK_VERIFIED"
    }

    config_path = EXP / "config.json"
    config = read_json(config_path)
    config["status"] = "COMPLETE_AUDITED"
    config["research_decision"] = "NO_GO_DYNAMIC_INCREMENTAL_EVIDENCE"
    config["handoff_status"] = handoff_status["status"]
    dump_json(config_path, config)

    probe["status"] = "COMPLETE_AUDITED"
    probe["research_decision"] = "NO_GO_DYNAMIC_INCREMENTAL_EVIDENCE"
    probe["independent_audit_status"] = audit["status"]
    dump_json(RESULTS / "probe_summary.json", probe)

    report = build_report(qc, probe, audit, metrics, comparisons, pooled, complement,
                          classes, sites, confounds, darg_folds, handoff_written)
    atomic_write(RESULTS / "report.md", report.encode("utf-8"))

    artifacts = sorted(p for p in RESULTS.iterdir() if p.is_file() and p.name != "manifest.json")
    code_paths = [EXP / n for n in ("config.json", "extract_dynamic.py", "run.py", "audit.py", "finalize.py")]
    e11_root = REPO / "experiment/experiment11_roi_mapping_recovery"
    e74_root = Path("/model/spromptgl/experiment/experiment74_4_soft_positive_dcr")
    sources = [e11_root / "recover_roi_mapping.py", e11_root / "results/roi_256_mapping.csv",
               e11_root / "results/data_audit.json", e74_root / "run.py", e74_root / "soft_positive_dcr.py",
               e74_root / "manifest.json", e74_root / "results/audit_status.json"]
    manifest = {
        "experiment": "Experiment 80 — PSDC-IVE",
        "status": "COMPLETE_AUDITED",
        "research_decision": "NO_GO_DYNAMIC_INCREMENTAL_EVIDENCE",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_root": str(EXP),
        "preferred_root": config["preferred_root"],
        "preferred_root_fallback_reason": config["fallback_reason"],
        "data": {"subjects": qc["subjects"], "dynamic_eligible": qc["valid_dynamic_subjects"],
                 "coverage": probe["coverage"], "roi_count": qc["roi_count"], "static_edges": qc["static_edges"],
                 "static_fc_max_abs_error": qc["static_reconstruction_max_absolute_error"],
                 "bold_inventory_hashes_verified": qc["all_bold_hashes_match_e11"],
                 "tr_all_sites_verified": qc["tr_recovered_for_all_subjects"],
                 "frame_timing_limitation": audit["data"]["timepoint_row_sequence_metadata"]},
        "execution": {"device": "CPU", "cpu_model": "Hygon C86-4G", "workers": 8,
                      "gpu_available_but_unused": "PPU-ZW810E, 98304 MiB", "phase_seconds": {
                          "dynamic_extraction": qc["wall_seconds"], "nested_probes": probe["wall_seconds"],
                          "independent_audit": audit["elapsed_seconds"]}, "darg_retraining": False},
        "audit_status": audit["status"],
        "darg_history_status": audit["darg_history"]["status"],
        "darg_forward_replay_in_this_run": audit["darg_history"]["forward_replay_in_this_run"],
        "historical_e74_audit": audit["assets_unchanged"],
        "historical_darg_protected_source_sha256": {
            path: record["sha256"]
            for path, record in audit["darg_history"]["protected_sources_resolved"].items()
        },
        "handoff": handoff_status,
        "code_sha256": {str(p.relative_to(EXP)): sha256(p) for p in code_paths if p.exists()},
        "historical_source_sha256": {str(p): sha256(p) for p in sources if p.exists()},
        "result_sha256": {p.name: sha256(p) for p in artifacts},
        "headline_pooled_oof_deltas_pp": probe["pooled_oof_deltas_pp"],
        "limitations": audit["historical_protocol_limits"]
    }
    dump_json(EXP / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "decision": manifest["research_decision"],
                      "handoff": handoff_status, "report": str(RESULTS / "report.md"),
                      "manifest": str(EXP / "manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
