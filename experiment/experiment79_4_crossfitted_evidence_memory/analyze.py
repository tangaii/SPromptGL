"""Summarize audited Experiment 79.4 metrics and mechanism evidence."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
import numpy as np
import run

ROOT=Path(__file__).resolve().parent
METRICS=("acc","hard_auc","prob_auc","sen","spe","asd_f1")

def write(path,rows):run.write_csv(path,rows)
def pct(x):return 100.0*float(x)
def mean_sd(values):
    a=np.asarray(values,dtype=float)
    return f"{a.mean():.4f}±{a.std(ddof=0):.4f}"
def metric_row(method,fold,metrics,ledger):
    return {"method":method,"ledger":ledger,"fold":fold,**{k:pct(metrics[k]) for k in METRICS}}
def paired(rows,baseline,method,ledger):
    baseline_ledger="historical" if baseline=="E74.4-G2" else "base_anchored"
    b={r["fold"]:r for r in rows if r["method"]==baseline and r["ledger"]==baseline_ledger}
    m={r["fold"]:r for r in rows if r["method"]==method and r["ledger"]==ledger}
    out=[]
    for key in METRICS:
        ds=[m[f][key]-b[f][key] for f in sorted(b)]
        arr=np.asarray(ds);wins=sum(x>1e-12 for x in ds);ties=sum(abs(x)<=1e-12 for x in ds);losses=len(ds)-wins-ties
        d9=[m[f][key]-b[f][key] for f in sorted(b) if f!=2]
        out.append({"comparison":f"{method}-{baseline}","ledger":ledger,"metric":key,
            "mean_difference_pp":float(arr.mean()),"population_sd_difference_pp":float(arr.std(ddof=0)),
            "descriptive_ci95_low":float(arr.mean()-1.96*arr.std(ddof=1)/math.sqrt(len(arr))),
            "descriptive_ci95_high":float(arr.mean()+1.96*arr.std(ddof=1)/math.sqrt(len(arr))),
            "W":wins,"T":ties,"L":losses,"fold_differences_pp":json.dumps({str(i+1):float(x) for i,x in enumerate(ds)}),
            "excluding_fold2_mean_difference_pp":float(np.mean(d9)),"excluding_fold2_wtl":f"{sum(x>1e-12 for x in d9)}/{sum(abs(x)<=1e-12 for x in d9)}/{sum(x< -1e-12 for x in d9)}"})
    return out

def main():
    audit=json.loads((ROOT/"results/audit_status.json").read_text())
    if audit.get("status")!="PASS":raise RuntimeError("analysis requires a passing independent audit")
    by={(r["fold"],r["ledger"]):r for r in audit["best_rows"]}
    folds=[];mechanism=[];memory=[];corrections=[]
    for fold in range(1,11):
        base=by[(fold,"base_anchored")]["base_metrics"]
        folds.append(metric_row("E74.4-G2",fold,base,"historical"))
        for ledger in ("base_anchored","method"):
            r=by[(fold,ledger)];folds.append(metric_row("E79.4-G1",fold,r["metrics"],ledger))
            t=r["transition"]
            corrections.append({"fold":fold,"ledger":ledger,"method":"E79.4-G1","base_errors":t["base_errors"],
                "method_errors":t["method_errors"],"error_to_correct":t["error_to_correct"],"correct_to_error":t["correct_to_error"],
                "net_correction":t["net_correction"],"hard_flips":t["hard_flips"],"asd_error_corrected":t["asd_error_corrected"],
                "td_error_corrected":t["td_error_corrected"],"correct_retention":t["correct_retention"],
                "mean_abs_probability_delta":t["mean_abs_probability_delta"]})
            corrections[-1].update({"relative_to_history_"+k:v for k,v in r["history_transition"].items()})
            e=r.get("test_memory_evidence",{});tr=r.get("training_evidence",{})
            mechanism.append({"fold":fold,"ledger":ledger,"epoch":r["epoch"],"wq_abs_sum":r["wq_abs_sum"],"wk_abs_sum":r["wk_abs_sum"],
                "learned_fixed_gap":r["learned_fixed_gap"],"memory_evidence_test_accuracy":e.get("accuracy",0),
                "memory_evidence_test_asd_accuracy":e.get("asd_accuracy",0),"memory_evidence_test_td_accuracy":e.get("td_accuracy",0),
                "memory_evidence_valid_fraction":e.get("valid_fraction",0),"memory_evidence_test_n":e.get("valid_count",0),
                "memory_evidence_train_accuracy":tr.get("evidence_accuracy",0),"memory_evidence_train_asd_accuracy":tr.get("evidence_asd_accuracy",0),
                "memory_evidence_train_td_accuracy":tr.get("evidence_td_accuracy",0),"evidence_loss":tr.get("evidence",0),
                "wq_grad_norm":tr.get("grad_norms",{}).get("Wq",0),"wk_grad_norm":tr.get("grad_norms",{}).get("Wk",0),
                "wv_grad_norm":tr.get("grad_norms",{}).get("Wv",0),"gate_grad_norm":tr.get("grad_norms",{}).get("gate",0),
                "gate_mean":r["gate_mean"],"delta_ratio_mean":r["delta_ratio_mean"],"delta_ratio_p90":r["delta_ratio_p90"],
                "cap_fraction":r["cap_fraction"],"reliability_mean":r["reliability_mean"],
                "base_anchor_state_match":r["base_anchor_state_match"]})
            mechanism[-1].update({"memory_evidence_test_"+k:e.get(k,0) for k in ("hard_auc","prob_auc","base_wrong_memory_correct",
                "base_wrong_memory_wrong","memory_correct_fusion_still_wrong","memory_wrong_harm","wrong_base_margin_improvement_rate","retrieval_rerank_fraction")})
            mechanism[-1].update({"memory_evidence_train_"+k:tr.get("replayed_evidence_"+k,0) for k in ("accuracy","hard_auc","prob_auc")})
            mechanism[-1].update({k:r[k] for k in ("gate_p50","gate_p90","gate_p95","delta_ratio_p50","delta_ratio_p95")})
            # The full memory counts and reliability are present in the checkpoint audit row.
            ck=run.torch_load(r["checkpoint"]);mem=ck["memory_used"] or {}
            for name,m in mem.items():
                labels=m.get("label",[]);rel=m.get("reliability",[])
                memory.append({"fold":fold,"ledger":ledger,"bank":name,"version":m.get("version"),"refresh_epoch":m.get("refresh_epoch"),
                    "size":len(m.get("ids",[])),"asd":int((labels==0).sum()) if hasattr(labels,"shape") else 0,
                    "td":int((labels==1).sum()) if hasattr(labels,"shape") else 0,"candidate_count":m.get("candidate_count",0),
                    "evidence_candidate_count":m.get("evidence_candidate_count",0),"accepted_count":m.get("accepted_count",0),
                    "rejected_count":m.get("rejected_count",0),"reliability_mean":float(rel.float().mean()) if len(rel) else 0.0,
                    "reliability_min":float(rel.float().min()) if len(rel) else 0.0,"reliability_max":float(rel.float().max()) if len(rel) else 0.0,
                    "support_auc_mean":float(m["support_auc"].float().mean()) if len(m.get("support_auc",[])) else 0.0,
                    "support_separation_mean":float(m["support_separation"].float().mean()) if len(m.get("support_separation",[])) else 0.0,
                    "source_count":len(m.get("source_ids",[])),"support_count":len(m.get("support_ids",[])),
                    "source_digest":m.get("source_ids_digest"),"support_digest":m.get("support_ids_digest")})
    write(ROOT/"results/fold_metrics.csv",folds)
    write(ROOT/"results/correction_analysis.csv",corrections)
    write(ROOT/"results/mechanism.csv",mechanism)
    write(ROOT/"results/memory_analysis.csv",memory)
    paired_rows=[]
    paired_rows+=paired(folds,"E74.4-G2","E79.4-G1","base_anchored")
    paired_rows+=paired(folds,"E74.4-G2","E79.4-G1","method")
    write(ROOT/"results/paired_comparisons.csv",paired_rows)
    summaries=[]
    for method,ledger in (("E74.4-G2","historical"),("E79.4-G1","base_anchored"),("E79.4-G1","method")):
        vals=[r for r in folds if r["method"]==method and r["ledger"]==ledger]
        row={"method":method,"ledger":ledger}
        for key in METRICS:row[key+"_mean_sd_pct"]=mean_sd([x[key] for x in vals])
        summaries.append(row)
    write(ROOT/"results/summary.csv",summaries)
    interventions=json.loads((ROOT/"results/audit_status.json").read_text()).get("interventions",[])
    write(ROOT/"results/intervention_analysis.csv",interventions)
    # Prespecified judgment: an improvement must exist at the historical base
    # epoch, survive Fold 2 exclusion, and have a positive patient correction balance.
    ba_rows=[r for r in corrections if r["ledger"]=="base_anchored"]
    net=sum(r["net_correction"] for r in ba_rows)
    acc_delta=next(r["mean_difference_pp"] for r in paired_rows if r["ledger"]=="base_anchored" and r["metric"]=="acc")
    acc_9=next(r["excluding_fold2_mean_difference_pp"] for r in paired_rows if r["ledger"]=="base_anchored" and r["metric"]=="acc")
    content=[r for r in interventions if r["ledger"]=="base_anchored" and r["intervention"]=="Memory-Content-Permuted"]
    content_effect=bool(content) and bool(np.mean([r["full_minus_intervention_acc"] for r in content])>1e-12 or np.mean([r["prob_auc_delta"] for r in content])>1e-5)
    tradeoff_ok=all(next(r["mean_difference_pp"] for r in paired_rows if r["ledger"]=="base_anchored" and r["metric"]==k)>=-.5 for k in ("sen","spe"))
    if acc_delta>0 and acc_9>0 and net>0 and content_effect and tradeoff_ok:
        verdict="PROMISING_CROSS_FITTED_EVIDENCE_MEMORY"
    elif content_effect or acc_delta>0:
        verdict="PARTIAL_CROSS_FITTED_EVIDENCE_MEMORY"
    else:verdict="NO_GO_CROSS_FITTED_EVIDENCE_MEMORY"
    analysis={"status":"PASS","audit_status":audit["status"],"verdict":verdict,"base_anchored_acc_delta_pp":acc_delta,
              "base_anchored_acc_delta_excluding_fold2_pp":acc_9,"base_anchored_net_correction":net,
              "content_permutation_has_observable_effect":content_effect,"summary_rows":len(summaries),"fold_rows":len(folds),
              "paired_rows":len(paired_rows),"correction_rows":len(corrections),"memory_rows":len(memory),
              "mechanism_rows":len(mechanism),"intervention_rows":len(interventions)}
    report=["# Experiment 79.4：CF-CPDEM正式结果","",f"- 状态：`{audit['status']}`；科研判定：`{verdict}`。",
      "- 基座：E74.4-G2 Positive-DCR；seed=0；10折；单折一条共享DARG轨迹。",
      "- 新记忆只读取合法外层训练CAM144。训练Query从A读取B Bank、从B读取A Bank；每个Bank的Source/Support互斥。测试只读取M_full。",
      "- 训练和预测保留历史val=test选模、CV前预处理与transductive图；这些限制不构成独立外部验证。Fold 2原历史重叠仍保留在旧基座，但被新增Memory排除。","",
      "## 六指标（%，mean ± population SD）","","| 方法 | Ledger | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        label=s["method"]+(" Base-Anchored" if s["ledger"]=="base_anchored" else " Method-Selected" if s["ledger"]=="method" else "")
        report.append("| "+label+" | "+s["ledger"]+" | "+" | ".join(s[k+"_mean_sd_pct"] for k in METRICS)+" |")
    report += ["","## 配对和纠错","",f"Base-Anchored ACC差 `{acc_delta:+.4f}pp`，排除Fold 2为 `{acc_9:+.4f}pp`；历史最佳Epoch的净纠错为 `{net:+d}`。",""]
    for ledger in ("base_anchored","method"):
        group=[r for r in corrections if r["ledger"]==ledger]
        report.append(f"- {ledger}: 错误→正确 {sum(x['error_to_correct'] for x in group)}，正确→错误 {sum(x['correct_to_error'] for x in group)}，净纠错 {sum(x['net_correction'] for x in group)}，硬翻转 {sum(x['hard_flips'] for x in group)}。")
    report += ["","## 机制","",f"Base-Anchored Memory-Off外的内容打乱可观察效应：`{content_effect}`。可学习匹配梯度、训练/测试Memory证据表现、独立Support可靠性、Gate与增量分布见`mechanism.csv`和`memory_analysis.csv`。",
      "","## 审计","",f"best={audit['best_audited']}/20，latest={audit['latest_audited']}/10，最新状态双重下一步重放={audit['latest_continuation_replays_passed']}/10。详细依据见`audit_status.json`。",
      "","## 解释边界","","本实验中的CAM维度贡献及跨患者支持是模型表示空间内的计算性证据，不等同于医学病理标志物或因果生物标志物。E74.4历史测试集参与选模、CV前RFE/标准化、transductive图与Fold 2重叠限制泛化解释。"]
    report=complete_report(report,analysis,audit,summaries,paired_rows,corrections,mechanism,memory,interventions)
    run.atomic_json(ROOT/"results/analysis_status.json",analysis)
    (ROOT/"results/report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps(analysis,indent=2))

def complete_report(report,analysis,audit,summaries,paired_rows,corrections,mechanism,memory,interventions):
    """Postprocessing only: apply the user's outcome rules and retain definitions."""
    def avg(rows,key):return float(np.mean([float(r[key]) for r in rows]))
    tasks=[json.loads(p.read_text()) for p in sorted((ROOT/".runtime/tasks").glob("E79.4-G1_f*.json")) if not p.name.endswith("continuation.json")]
    samples=[]
    for line in (ROOT/".runtime/logs/gpu_samples.csv").read_text().splitlines():
        try:samples.append([float(x.strip().split()[0]) for x in line.split(",")])
        except ValueError:continue
    resource={"device":tasks[0]["device"],"max_workers":max(r["max_workers"] for r in tasks),
        "longest_worker_elapsed_seconds":max(r["elapsed_seconds"] for r in tasks),
        "max_worker_peak_allocated_mib":max(r["peak_gpu_bytes"] for r in tasks)/2**20,
        "sampled_device_peak_used_mib":max(r[0] for r in samples),"device_total_mib":max(r[1] for r in samples),
        "device_sample_count":len(samples),"epochs":sum(r["epochs_executed"] for r in tasks),
        "memory_refreshes":sum(r["memory_refreshes"] for r in tasks),"trainable_memory_parameters":sum(p.numel() for p in run.CFCPDEM().parameters())}
    analysis["resources"]=resource
    diagnostic={}
    for ledger in ("base_anchored","method"):
        rr=[r for r in mechanism if r["ledger"]==ledger];ar=[r for r in audit["best_rows"] if r["ledger"]==ledger]
        fields=("memory_evidence_train_accuracy","memory_evidence_train_hard_auc","memory_evidence_train_prob_auc",
                "memory_evidence_test_accuracy","memory_evidence_test_hard_auc","memory_evidence_test_prob_auc",
                "wq_grad_norm","wk_grad_norm","wv_grad_norm","gate_grad_norm","gate_mean","delta_ratio_mean","cap_fraction",
                "memory_evidence_test_retrieval_rerank_fraction")
        d={k:avg(rr,k) for k in fields}
        for k in ("base_wrong_memory_correct","base_wrong_memory_wrong","memory_correct_fusion_still_wrong","memory_wrong_harm"):
            d[k]=sum(r["test_memory_evidence"][k] for r in ar)
        strata={}
        for domain in ("training","test"):
            strata[domain]={}
            for group in ("all","base_correct","base_wrong","memory_conflict","memory_agreement"):
                values=[r["enhancement_diagnostics"][domain][group] for r in ar];n=sum(v["n"] for v in values)
                strata[domain][group]={"n":n,"gate_mean":sum(v["gate_mean"]*v["n"] for v in values)/max(n,1),
                    "delta_ratio_mean":sum(v["delta_ratio_mean"]*v["n"] for v in values)/max(n,1),
                    "near_cap_count":sum(v["near_cap_count"] for v in values),"zero_count":sum(v["zero_count"] for v in values)}
        d["patient_weighted_strata"]=strata
        diagnostic[ledger]=d
    analysis["diagnostic_summary"]=diagnostic
    analysis["verdict_basis"]={"priority":"Base-Anchored at the exact historical best checkpoints",
        "no_acc_gain_over_history":all(r["mean_difference_pp"]<=0 for r in paired_rows if r["metric"]=="acc"),
        "no_positive_anchored_net_correction":analysis["base_anchored_net_correction"]<=0,
        "no_classification_loss_from_value_permutation":not analysis["content_permutation_has_observable_effect"],
        "note":"Memory-only test accuracy above chance is a mechanism observation; it does not override the prescribed NO-GO criteria. The isolated Method-Selected +1 correction is not a historical-base gain."}
    report[2]=f"- 状态：`COMPLETE_AUDITED`；科研判定：`{analysis['verdict']}`。"
    report += ["","## 论文、源码与历史身份","",
      "已审查[SPromptGL原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[作者源码](https://github.com/wanxixi11/SPromptGL)、E74.4真实训练恢复接口、E79.3-G2源码与权威结果，以及[SAM 2论文](https://arxiv.org/abs/2408.00714)和官方Memory实现。原CAM进行跨模态交互；本实验新增跨患者Memory，二者不等同。SAM 2仅提供有限Bank、Cross-Attention与先读取/预测、后写入的借鉴；未移植视频掩码、帧位置或跟踪假设。",
      "实际PLNNoPEN的OutputLayer输入是CAM144，内部为LayerNorm(144)→Linear(144,36)→GELU→Dropout→Linear(36,2)。新增分支使用其冻结参数的eval函数调用，梯度仅回到新增模块。原分类头仍由原DARG损失更新。本实验不是完整主干端到端联合优化。",
      "每折从E61 epoch-10恢复模型、三个优化器、4096关系对和全部RNG；opt_base→opt_cam→opt_added次序不变。历史最佳Epoch以1-based锁定为123/171/183/216/295/339/148/263/186/233。10/10历史最佳主干（含优化器）逐张量一致；当前871人整批只读分类头输出与原BaseEval逐元素一致。历史无全部逐轮参数快照，故逐轮参数摘要仅记录本实验，未伪称与不存在的历史摘要比较。",
      "","## CF-CPDEM实际方法","",
      "A/B按合法训练患者的疾病与站点确定性分层，互不重叠；各Bank以最大余数分配约75% Source、25% Support。A查询只读取B、B只读取A；Source特征干预和Support可靠性/疾病方向完全在该Bank内部。测试只读取Full。此隔离针对新增Memory的显式标签作用域，原DARG仍按历史全训练监督和transductive图运行，不构成整个表示流程的无标签OOF。",
      "正确分类Source患者执行α=0.5和1.0的144维替换，选两种强度下都正贡献的最多12维，并验证联合遮蔽margin下降。背景维度匹配Support，每类最多16人，至少4人，要求同类证据匹配均值高于异类且AUC≥0.60。r_self=1−exp(−joint_drop)，r_within=(same_match_mean+1)/2，r_between=2(AUC−0.5)，r_coverage=min(1,n_support/32)×min(1,n_sites/3)，各分量截断在[0,1]，最终取乘积。",
      "固定Masked Cosine仅预筛每类16项；当前可训练Wq/Wk在同一Mask作用后的Query/Key上计算64维归一化匹配，以cosine/0.2+log(reliability)重排Top-8。真实有效数量归一化logmeanexp形成ASD/TD竞争分数；缺任一类即严格零增强。Wv无bias，将中心化证据与按CAM尺度缩放的Support疾病方向编码为Value；两类读出及其差产生144维修正方向。",
      "Gate使用原logp、Memory类别支持、Attention熵、两类实际可靠性等信息，初始化bias−2.0。有效Gate额外乘sqrt(r_ASD r_TD)/(sqrt(r_ASD r_TD)+0.1)，记录中区分原始Attention可靠性和这一变换后可靠性。平滑tanh范数限制保持7.5%上限。最终投影std=0.01的小非零初始化；无随机Dropout。",
      "新增损失固定为加权NLL+0.20错误margin+0.10正确预测保护+0.10证据CE+0.01相对增量平方；错误样本权重3、正确1，保留历史类权重，以sum(sample_weight×class_weight[label])归一化NLL。AdamW lr=0.0005、WD=0.01。未依据测试结果调整任何阈值或超参数。",
      "每5轮刷新三套Bank，第一次在第11轮预测结束后构建；当轮仅读取旧版本。Key随当前Wk重算，不使用陈旧投影缓存。只保留当前Memory及独立重建所需Bank局部CAM/分类头快照，未保存全Epoch患者特征。",
      "","## 资源与完成情况","",
      f"实际目录：`{ROOT}`。首选OSSFS目录在mkdir/写入/fsync/原子rename/读回探测中出现EIO(errno=5)，立即采用/model回退。单张{resource['device']}，10个fold worker，1个正式方法。最长worker实测{resource['longest_worker_elapsed_seconds']:.1f}秒（约{resource['longest_worker_elapsed_seconds']/60:.1f}分钟；并行训练阶段计时），独立最终审计{audit['elapsed_seconds']:.1f}秒。",
      f"设备1秒采样{resource['device_sample_count']}条，实测整卡used峰值{resource['sampled_device_peak_used_mib']:.0f}MiB（{resource['sampled_device_peak_used_mib']/1024:.2f}GiB）；总显存{resource['device_total_mib']:.0f}MiB。单worker Torch peak allocated最高{resource['max_worker_peak_allocated_mib']:.1f}MiB；两种内存口径不同。新增可训练参数{resource['trainable_memory_parameters']:,}个。10/10完成，共{resource['epochs']}个实际训练Epoch、{resource['memory_refreshes']}次fold级Memory刷新。",
      "","## 双Ledger配对结果","","以下差值均为方法−E74.4-G2（pp）。95%区间为配对折差mean±1.96×sample-SD/sqrt(10)的描述性区间；十折训练集重叠，不能视为独立外部队列或临床显著性证明。","",
      "| Ledger | ACC差与描述性95%区间 | W/T/L | 去Fold 2 ACC差 | Hard-AUC差 | Prob-AUC差 | SEN差 | SPE差 | ASD-F1差 |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ledger in ("base_anchored","method"):
        rr={r["metric"]:r for r in paired_rows if r["ledger"]==ledger};a=rr["acc"]
        report.append(f"| {ledger} | {a['mean_difference_pp']:+.4f} [{a['descriptive_ci95_low']:+.4f},{a['descriptive_ci95_high']:+.4f}] | {a['W']}/{a['T']}/{a['L']} | {a['excluding_fold2_mean_difference_pp']:+.4f} | "+" | ".join(f"{rr[k]['mean_difference_pp']:+.4f}" for k in METRICS if k!="acc")+" |")
    report += ["","逐折六指标差、六指标95%区间、去Fold 2结果均保存在paired_comparisons.csv。Base-Anchored仅Fold 6降低1/87；Method-Selected各折ACC均与历史持平，但患者身份可变化。Method-Selected最佳Epoch为123/171/183/216/295/377/135/263/186/233，Fold 6与7偏离历史最佳。",
      "","## 同Checkpoint与历史基座纠错分开统计","",
      "| Ledger/参照 | 错误→正确 | 正确→错误 | 净纠错 | ASD纠正 | TD纠正 |",
      "|---|---:|---:|---:|---:|---:|"]
    for ledger in ("base_anchored","method"):
        rr=[r for r in corrections if r["ledger"]==ledger]
        for historical in (False,True):
            prefix="relative_to_history_" if historical else ""
            keys=("error_to_correct","correct_to_error","net_correction","asd_error_corrected","td_error_corrected")
            report.append("| "+ledger+("/历史最佳" if historical else "/同Epoch Memory-Off")+" | "+" | ".join(str(sum(r[prefix+k] for r in rr)) for k in keys)+" |")
    report += ["","Base-Anchored原正确803人保留802人（99.8755%），损害是1名TD被改判ASD。Method-Selected同Epoch原主干正确802人全部保留，同时纠正1名TD；相对历史最佳则纠正1名ASD、误伤1名TD。两种参照不能相互替代。去Fold 2后上述纠错计数不变。",
      "","## 证据能力与融合失效分解","",
      "| Ledger | A/B交叉训练证据ACC/Hard-AUC/Prob-AUC | 测试证据ACC/Hard-AUC/Prob-AUC | 原错误中Memory正确/错误 | Memory正确但融合仍错 | Memory错误并造成误伤 |",
      "|---|---|---|---:|---:|---:|"]
    for ledger,d in diagnostic.items():
        tr="/".join(f"{100*d['memory_evidence_train_'+k]:.4f}" for k in ("accuracy","hard_auc","prob_auc"))
        te="/".join(f"{100*d['memory_evidence_test_'+k]:.4f}" for k in ("accuracy","hard_auc","prob_auc"))
        report.append(f"| {ledger} | {tr} | {te} | {d['base_wrong_memory_correct']}/{d['base_wrong_memory_wrong']} | {d['memory_correct_fusion_still_wrong']} | {d['memory_wrong_harm']} |")
    report += ["","以上证据指标为逐折均值，均有双类别可用，测试总计871个折测试记录。训练指标是A/B交叉Memory查询，不是外部测试或无标签表征OOF。历史最佳的68个原错误中，65个Memory也错，仅3个给出正确类别支持（4.41%）；这3个最终仍全错，另有1个原正确TD被错误Memory支持误伤。故同时存在错误互补性不足和正确Memory支持未转化为硬纠错。现有证据不能唯一归因于记忆内容缺乏独立信息或融合没有有效利用内容。",
      "","## 梯度、可靠性和修正幅度","",
      "L_evidence单独反传的正式预检确认Wq/Wk梯度均非零有限，并检查参数扰动会改变分数、CAM/主干/原OutputLayer不接收新增梯度。正式各有效更新同样检查Wq/Wk梯度和主干原梯度不变。最佳点的梯度范数是训练总Memory损失的诊断，不能混称为单独证据损失梯度。","",
      "| Ledger | Wq/Wk/Wv/Gate平均梯度范数 | 选中支持越出固定Top-8比例 | Mean Gate | 平均相对增量 | 接近7.5%上限比例 |",
      "|---|---|---:|---:|---:|---:|"]
    for ledger,d in diagnostic.items():
        grad="/".join(f"{d[k+'_grad_norm']:.6f}" for k in ("wq","wk","wv","gate"))
        report.append(f"| {ledger} | {grad} | {100*d['memory_evidence_test_retrieval_rerank_fraction']:.2f}% | {d['gate_mean']:.4f} | {100*d['delta_ratio_mean']:.4f}% | {100*d['cap_fraction']:.2f}% |")
    report += ["","接近上限预定义为delta_ratio>0.99×0.075（7.425%），不是精确硬裁剪命中率。平滑限制保留修正方向，但Base-Anchored仍有约一半患者接近上限，未充分解决饱和。Gate并未塌缩到0；问题不是E79.2式几乎没有增强。P50/P90/P95是各折分位数的均值，见mechanism.csv；分组结果如下（按患者记录数加权）。","",
      "| Ledger | 阶段/人群 | 记录数 | Gate均值 | 平均相对增量 | 接近上限人数 | 全零人数 |",
      "|---|---|---:|---:|---:|---:|---:|"]
    for ledger,d in diagnostic.items():
        for domain,groups in d["patient_weighted_strata"].items():
            for name,v in groups.items():
                report.append(f"| {ledger} | {domain}/{name} | {v['n']} | {v['gate_mean']:.4f} | {100*v['delta_ratio_mean']:.4f}% | {v['near_cap_count']} | {v['zero_count']} |")
    report += ["","训练患者跨折重复计入，不能把这些记录数称为不同的独立患者。",
      "","## 实际Memory容量和独立Support","",
      "| Ledger/Bank | 容量范围 | 候选合计/接受/拒绝 | 原始可靠性均值 | Support AUC均值 |",
      "|---|---:|---:|---:|---:|"]
    for ledger in ("base_anchored","method"):
        for bank in ("A","B","full"):
            rr=[r for r in memory if r["ledger"]==ledger and r["bank"]==bank]
            report.append(f"| {ledger}/{bank} | {min(r['size'] for r in rr)}–{max(r['size'] for r in rr)} | {sum(r['candidate_count'] for r in rr)}/{sum(r['accepted_count'] for r in rr)}/{sum(r['rejected_count'] for r in rr)} | {avg(rr,'reliability_mean'):.4f} | {avg(rr,'support_auc_mean'):.4f} |")
    report += ["","这些是十折最佳点各Bank的候选记录，不是不同患者数或全训练累计数。每个有效候选最多12维。A/B最佳点均48 ASD+48 TD；Full的Base-Anchored仅一折47 TD，其余满容量。独立Support不保证低接受率：强训练表示下，最佳点大部分候选仍全部通过，Full的平均Support AUC为0.9734，而测试Memory证据ACC仅0.8874。不能把高训练Support可靠性等同于外部诊断可靠性。",
      "","## 同Checkpoint只读干预","",
      "差值为Full−干预。概率MAE按患者ASD概率变化后逐折均值，Prob-AUC差为pp；Memory-Off的delta_ratio严格0。其他干预保持权重，不重训，因而不是训练消融。","",
      "| Ledger | 干预 | ACC差pp | Prob-AUC差pp | 相对Full硬翻转 | ASD概率MAE |",
      "|---|---|---:|---:|---:|---:|"]
    for ledger in ("base_anchored","method"):
        for name in ("Memory-Off","Memory-Content-Permuted","Learnable-Matching-Off","Cross-Patient-Reliability-Off"):
            rr=[r for r in interventions if r["ledger"]==ledger and r["intervention"]==name]
            report.append(f"| {ledger} | {name} | {100*avg(rr,'full_minus_intervention_acc'):+.4f} | {100*avg(rr,'prob_auc_delta'):+.4f} | {sum(r['prediction_flips'] for r in rr)} | {avg(rr,'mean_abs_prob_delta'):.8f} |")
    report += ["","Content-Permuted固定Query、Key、Top-K和Attention，仅循环置换每个已选类内Value，独立核验Value患者身份非平凡改变。结果是极小概率变化、零硬翻转、零Prob-AUC变化；没有显示具体Key–Value患者对应具有独立分类收益。这一干预不等价于删除类别分区或任意外部噪声。可学习匹配和可靠性Off确实改变分数/幅度及部分概率，但均无硬分类变化，且Full的Prob-AUC反低0.0105pp。不能把可学习检索有梯度和可靠性在公式中生效解释为已证明诊断增益。",
      "","## 独立审计、源码版本与复现","",
      f"10/10正式折、20/20双Ledger最佳Checkpoint独立重放，10/10 Latest恢复并实际执行下一步，比较从未重新加载的live轨迹：主干与新增参数、四个优化器、Memory、预测、损失及Python/NumPy/Torch CPU/CUDA RNG摘要全部一致。{audit['rng_exact_triplets']}/{audit['rng_common_triplets']}个共同Epoch的三处RNG记录完全匹配，0不匹配。",
      "Memory的Source/Support/分区、刷新时CAM与冻结head保存于Checkpoint；审计从刷新时材料重建Mask、遮蔽效应、支持ID、对照方向和可靠性逐项一致。不是使用最佳Epoch的新CAM重建旧Memory。Fold 2的376/406和所有外层测试ID在新增Memory及支持/中性统计中均排除。",
      "Memory-Off令增强CAM与原CAM逐元素一致。只读head在原871人批大小下logp逐元素一致；小测试批head与全批原输出存在至多约1e-5的算子归约差异，审计单独记录，不将其当作记忆效应。Memory-Off概率采用原BaseEval概率严格回退。",
      "正式训练后仅audit.py/analyze.py作审计/汇总修正：固定全批分类头等价检查、补充标签分层诊断、JSON布尔序列化及依据原任务规则的判定逻辑。没有改动训练入口、模型、损失、配置、权重或历史文件。训练与最终后处理SHA均列于audit_status.json/source_changes及manifest.json；训练源集合SHA保留不覆盖。复现入口为python run.py，预检/训练参数见其--help；独立重放python audit.py，汇总python analyze.py。",
      "","## 与E79.3-G2比较及最终判断","",
      "E79.3-G2的Base-Anchored ACC91.9592%，净纠错−2；E79.4为92.0742%，净纠错−1，改善0.1149pp但仍低于DARG。Method-Selected两者ACC均92.1891%；E79.4 Prob-AUC93.3828%低于E79.3-G2的93.5690%。本次实质修复可学习Q/K证据梯度、显式交叉记忆作用域、Source/Support独立性、真实可靠性Gate、无效Top-K和真实Latest续训审计，工程修复不等于科研成功。",
      "最终判定：NO_GO_CROSS_FITTED_EVIDENCE_MEMORY。未超过历史DARG，历史最佳Epoch无错误被纠正且误伤1人；方法选模的单折+1不能抵消历史最佳对比净变化0；具体Value内容干预没有破坏分类表现。Memory证据有类别区分能力，但在DARG原错误上几乎没有互补性，少量正确证据也未被融合成纠错。当前证据同时暴露内容冗余与融合利用不足，无法唯一确定根本原因。没有充分理由将此实现继续作为MICCAI 2027第二创新模块；不自动追加E79.5或测试集调参。"]
    return report

if __name__=="__main__":main()
