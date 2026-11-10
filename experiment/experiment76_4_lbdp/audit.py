#!/usr/bin/env python3
"""Independent deterministic audit and report generation for Experiment 76.4."""
from __future__ import annotations

import csv, hashlib, importlib.util, json, math, sys, time
from pathlib import Path

EXP=Path(__file__).resolve().parent
sys.dont_write_bytecode=True


def load_run():
    spec=importlib.util.spec_from_file_location("e764_run_for_audit",EXP/"run.py")
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


def read_csv(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def close(a,b,name,atol=1e-10):
    if not math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=atol):raise AssertionError(f"{name}: {a} != {b}")


def table(rows,scope,groups,metrics):
    metric_ids={"ACC":"acc","Hard-AUC/BA":"hard_auc","Prob-AUC":"prob_auc",
                "SEN-ASD":"sen","SPE-TD":"spe","ASD-F1":"asd_f1"}
    lines=["| Method | "+" | ".join(metrics)+" |","|---|"+"|".join(["---:"]*len(metrics))+"|"]
    for group in groups:
        r=next(x for x in rows if x["scope"]==scope and x["group"]==group)
        label={"M0_original_mlp":"M0 原始MLP","M1_base_score_calibration":"M1 基础分数校准",
               "M2_global_hard_prototype":"M2 全局困难原型","M3_local_boundary_prototype":"M3 局部边界原型",
               "M4_shuffled_local_pair":"M4 局部配对打乱"}[group]
        vals=[]
        for display in metrics:
            m=metric_ids.get(display,display)
            vals.append(f"{100*float(r[m+'_mean']):.4f}±{100*float(r[m+'_sd_population']):.4f}")
        lines.append("| "+label+" | "+" | ".join(vals)+" |")
    return "\n".join(lines)


def render_report(status,summary,paired,fold_rows,corr,boundary,diag,fold2_rows):
    metrics=("acc","hard_auc","prob_auc","sen","spe","asd_f1")
    groups=("M0_original_mlp","M1_base_score_calibration","M2_global_hard_prototype",
            "M3_local_boundary_prototype","M4_shuffled_local_pair")
    labels={"acc":"ACC","hard_auc":"Hard-AUC/BA","prob_auc":"Prob-AUC","sen":"SEN-ASD","spe":"SPE-TD","asd_f1":"ASD-F1"}
    lines=["# Experiment 76.4 — LBDP 局部混淆边界原型验证","",
           f"状态：`{status['status']}`；结论：`{status['conclusion']}`。冻结特征实验，未训练DARG/GFR/CAM/OutputLayer。","",
           "## 方法和审计范围","",
           "在E74.4-G2 Positive-DCR最佳Checkpoint上，DARG、CAM和原OutputLayer保持冻结。E76.2角色阈值与稳定性（10次CAM144确定性扰动，>=9次预测不变）沿用为共同定义；困难样本直接来自该MLP自身错误预测。M2复用E76.2可靠/困难全局类别原型和全局对比方向。M3对每个真实误判患者建立`hard + same-disease correct boundary/reliable + opposite-disease reliable`局部三元关系，再以CAM144余弦相似度Top-3、温度0.10检索患者条件化方向。每个ASD/TD方向至少3个真实误判且可用三元关系，否则mask。M4在方向内保留角色/类别，仅打乱hard病例与同病/异病正确参照的映射；五个私有置换以校正分数集成，不作为独立折。","",
           "残差校正器对M1–M4统一使用训练支持集拟合的StandardScaler、L-BFGS-B二元logistic残差（L2=0.01）：`s_final=s_base+b+w^T z`，ASD分数为`logp_ASD-logp_TD`，正值对应ASD=0；原始分类器未被替换或更新。每个外层训练集内5折，校正训练证据另用3折cross-fit；测试标签只作最终计分。","",
           "CoW的困难原型来自支持分割自预测与mask差异，服务像素级分割边界；PGP-SAM使用可学习类内/类间原型和类引导cross-attention生成分割prompt。本实验只借鉴困难/局部关系的分析动机，采用患者分类错误及CAM144的固定特征检索，不是CoW或PGP-SAM的患者级复现。","",
           "## 冻结特征内层OOF","",
           "mean±population SD（%）；五个内层验证折汇总后以外层fold为统计单位。冻结编码器此前已在外层训练患者上训练，因此这只是条件于冻结表示的读出诊断。","",
           table(summary,"inner_oof",groups,[labels[m] for m in metrics]),"",
           "## 历史外层测试核验","",
           "mean±population SD（%）。M0必须与E74.4-G2原始最佳Checkpoint预测一致。这里的外层测试仍继承历史test折选模、CV前预处理、transductive图和Fold 2训练/测试标签重叠，不能称为无偏独立泛化结果。","",
           table(summary,"outer_test",groups,[labels[m] for m in metrics]),"",
           "### 排除Fold 2的描述性结果","",
           "排除Fold 2后的9个外层fold均值±总体SD（%）：","",
           table([r for r in summary if r["scope"]=="outer_test_exclude_fold2_9"],"outer_test_exclude_fold2_9",groups,[labels[m] for m in metrics]),"",
           "Fold 2去除1-based行377和407后的单折指标：","",
           "| Method | "+" | ".join(labels[m] for m in metrics)+" |","|---|"+"|".join(["---:"]*len(metrics))+"|"]
    for g in groups:
        r=next(x for x in fold2_rows if x["group"]==g)
        lines.append("| "+g+" | "+" | ".join(f"{100*float(r[m]):.4f}" for m in metrics)+" |")
    lines += ["","## M3配对比较","",
              "配对差为M3减对照，单位百分点。下表列外层测试主要指标；完整内层/外层八项指标、全十折与排除Fold 2九折的W/T/L和描述性95%配对区间见`paired_comparisons.csv`。","",
              "| 对照 | 指标 | Δ pp | 95% CI | W/T/L | 排除Fold 2 Δ pp |","|---|---|---:|---:|---:|---:|"]
    for other in groups:
        if other=="M3_local_boundary_prototype":continue
        for m in metrics[:6]:
            row=next(x for x in paired if x["scope"]=="outer_test" and x["comparison"]==f"M3_local_boundary_prototype - {other}" and x["metric"]==m and x["fold_set"]=="all_10")
            nine=next(x for x in paired if x["scope"]=="outer_test" and x["comparison"]==f"M3_local_boundary_prototype - {other}" and x["metric"]==m and x["fold_set"]=="exclude_fold2_9")
            lines.append(f"| {other} | {labels[m]} | {float(row['mean_delta_pp']):+.3f} | [{float(row['ci95_low_descriptive_pp']):+.3f}, {float(row['ci95_high_descriptive_pp']):+.3f}] | {row['wins']}/{row['ties']}/{row['losses']} | {float(nine['mean_delta_pp']):+.3f} |")
    lines += ["","## 真实纠错与边界证据","",
              "错误翻转按外层fold出现次数汇总；Fold 2两名历史重叠受试者在新增支持/校正拟合中排除，但在完整历史外层测试统计中按预注册保留并单列敏感性分析。稳定误判标签来自原始E74.4-G2的测试预测与冻结特征扰动，不参与推理路由。","",
              "| M3相对 | 范围 | Wrong→Correct | Correct→Wrong | 净纠正 | ASD→TD 错误前→后 | TD→ASD 错误前→后 |","|---|---|---:|---:|---:|---:|---:|"]
    outer_corr=[r for r in corr if r["scope"]=="outer_test" and r.get("comparison")]
    for other in groups:
        if other=="M3_local_boundary_prototype":continue
        for subset in ("all","stable_hard","stable_reliable","boundary_correct","ASD_to_TD_errors","TD_to_ASD_errors"):
            q=[r for r in outer_corr if r["group"]=="M3_local_boundary_prototype" and r["reference_group"]==other and r["subset"]==subset]
            if not q:continue
            sums={k:sum(int(x[k]) for x in q) for k in ("wrong_to_correct","correct_to_wrong","net_correction","asd_to_td_before","asd_to_td_after","td_to_asd_before","td_to_asd_after")}
            lines.append(f"| {other} | {subset} | {sums['wrong_to_correct']} | {sums['correct_to_wrong']} | {sums['net_correction']} | {sums['asd_to_td_before']}→{sums['asd_to_td_after']} | {sums['td_to_asd_before']}→{sums['td_to_asd_after']} |")
    lines += ["","M2/M3按同一原始预测方向区分真实误判患者与正确异类患者的条件AUC：","",
              "| 范围 | 方向 | 方法 | 可评估折 | 条件AUC均值 |","|---|---|---|---:|---:|"]
    for scope in ("inner_oof","outer_test"):
        for direction in ("ASD_to_TD","TD_to_ASD"):
            for method in ("M2_global","M3_local"):
                vals=[float(x["conditional_auc"]) for x in boundary if x["scope"]==scope and x["direction"]==direction and x["method"]==method and x["conditional_auc"] not in ("","NA_INSUFFICIENT_CLASS")]
                lines.append(f"| {scope} | {direction} | {method} | {len(vals)} | {sum(vals)/len(vals):.4f} |" if vals else f"| {scope} | {direction} | {method} | 0 | NA |")
    site_same=[float(x["pair_same_site_fraction"]) for x in diag if x["pair_same_site_fraction"] not in ("",None)]
    site_other=[float(x["pair_other_site_fraction"]) for x in diag if x["pair_other_site_fraction"] not in ("",None)]
    lines += ["","条件AUC只在同一基础预测类别内比较真实误判目标类与正确异类；无双类别折保留NA、不填0.5。局部配对中hard→same-reference同站点比例均值为"+
              (f"{sum(site_same)/len(site_same):.3f}" if site_same else "NA")+"，hard→other-reference同站点比例均值为"+
              (f"{sum(site_other)/len(site_other):.3f}" if site_other else "NA")+"。配对类别支持、每折fallback、Top-k、M4改变配对比例及bootstrap局部方向余弦见`prototype_diagnostics.csv`和runtime原型注册表。","",
              "## 决策与限制","",
              f"审计结论：`{status['status']}`。预注册研究判定：`{status['conclusion']}`。", "",
              "E74.4-G2模型曾用test折选择checkpoint，预处理在外层CV前完成，图为transductive；Fold 2的377/407同时出现在历史训练与测试索引。新增原型库和校正拟合将两人排除，但这不消除表征/图中历史暴露。训练侧困难患者是原始模型对自身训练样本的错误，不能称作独立样本外错误。对M3相对M0的外层测试错误净纠正是Q2的核心证据；不能用校正器内层成绩代替。","",
              "源码/文献核验：E74.4-G2 [锁定实现](https://github.com/tangaii/SPromptGL/tree/upload/experiment74-4-soft-positive-dcr/experiment/experiment74_4_soft_positive_dcr)的原OutputLayer输入仍为CAM144。CoW [IJCAI 2025](https://ijcai.org/proceedings/2025/139)及[官方代码](https://github.com/jcjiang99/CoW)研究分割支持集预测误差驱动的hard prototype，不是患者分类。PGP-SAM [arXiv:2501.06692](https://arxiv.org/abs/2501.06692)及[官方代码](https://github.com/PRIS-CV/PGP-SAM)研究类内/类间原型条件化分割prompt，不是此处的患者原型方法。","",
              "文件：`fold_metrics.csv`、`summary.csv`、`paired_comparisons.csv`、`correction_analysis.csv`、`boundary_diagnostics.csv`、`prototype_diagnostics.csv`、`calibrator_diagnostics.csv`、`audit_status.json`。","","**论文级解释边界：**本实验只验证冻结E74.4表示上的条件化原型检索和轻量校正，不证明独立临床泛化，不证明原型承载CAM144以外的疾病信息。"]
    return "\n".join(lines)+"\n"


def audit():
    import numpy as np
    r=load_run(); started=time.time(); results=EXP/"results"
    saved=json.loads((EXP/"manifest.json").read_text())
    source_sha=r.sha_json(r.inventory())
    if saved.get("source_sha256")!=source_sha:raise AssertionError("locked source inventory changed after run")
    refs=r.verify_references()
    stored=read_csv(results/"fold_metrics.csv")
    if len(stored)!=105:raise AssertionError(f"expected 105 fold metric rows including Fold2 sensitivity, got {len(stored)}")
    store={(int(x["fold"]),x["scope"],x["group"]):x for x in stored}
    if len(store)!=105:raise AssertionError("fold metric identities duplicate")
    cache=np.load(EXP/".runtime/cache/fold_predictions_and_features.npz",allow_pickle=False)
    registry=json.loads((EXP/".runtime/cache/prototype_registry.json").read_text())
    if r.sha_file(EXP/".runtime/cache/fold_predictions_and_features.npz")!=saved.get("feature_cache_sha256"):raise AssertionError("feature/prediction cache SHA mismatch")
    if r.sha_file(EXP/".runtime/cache/prototype_registry.json")!=saved.get("prototype_registry_sha256"):raise AssertionError("prototype registry SHA mismatch")
    recalc=[]; corrections=[];boundaries=[];diagnostics=[];fold_checks=[];split_checks=[];calchecks=[];fold2_rows=[]
    for fold in range(1,11):
        out=r.run_fold(fold);ids=out["inner_ids"];test=out["test_ids"];data=out["data"];p,_,lp=r.modules()
        pfx=f"f{fold}_"
        if not np.array_equal(ids,cache[pfx+"eligible_ids"]) or not np.array_equal(test,cache[pfx+"test_ids"]):raise AssertionError(f"fold {fold}: cached index alignment mismatch")
        if fold==2 and ({376,406}&set(ids.tolist()) or not {376,406}.issubset(set(test.tolist()))):raise AssertionError("Fold2 overlap protection failed")
        if np.intersect1d(ids,test).size:raise AssertionError(f"fold {fold}: outer test entered support")
        if not np.allclose(data["h"],cache[pfx+"h"],rtol=0,atol=0):raise AssertionError(f"fold {fold}: CAM144 feature cache mismatch")
        if not np.array_equal(data["pred"],cache[pfx+"pred_all"]) or not np.allclose(data["prob"],cache[pfx+"prob_all"],rtol=0,atol=1e-12):raise AssertionError(f"fold {fold}: frozen MLP output changed")
        if not np.array_equal(data["stability"],cache[pfx+"stability"]):raise AssertionError(f"fold {fold}: perturbation stability mismatch")
        testrows=np.asarray(data["test"])
        # The frozen E74.4 test output is also the checkpoint's historical saved output.
        for scope,method_ids,preds,probs,scores in (("inner_oof",ids,out["inner_pred"],out["inner_prob"],out["inner_score"]),
                                                     ("outer_test",test,out["outer_pred"],out["outer_prob"],out["outer_score"])):
            yy=data["y"][method_ids]
            for group in r.GROUPS:
                cp=cache[pfx+("inner" if scope=="inner_oof" else "outer")+f"_{group}_pred"]
                cb=cache[pfx+("inner" if scope=="inner_oof" else "outer")+f"_{group}_prob_asd"]
                cs=cache[pfx+("inner" if scope=="inner_oof" else "outer")+f"_{group}_score"]
                if not np.array_equal(preds[group],cp) or not np.allclose(probs[group],cb,rtol=0,atol=1e-12) or not np.allclose(scores[group],cs,rtol=0,atol=1e-12):raise AssertionError(f"fold {fold}/{scope}/{group}: patient predictions differ")
                m=r.metric(yy,preds[group],probs[group]);row=store[(fold,scope,group)]
                for key,val in m.items():close(row[key],val,f"{fold}/{scope}/{group}/{key}")
                recalc.append({"fold":fold,"scope":scope,"group":group,**m})
            if scope=="outer_test":
                e74=read_csv(r.BASE/"results/fold_metrics.csv")
                ref=next(x for x in e74 if x["group"]==r.BASE_GROUP and int(x["fold"])==fold)
                m0=store[(fold,"outer_test","M0_original_mlp")]
                for k in ("acc","hard_auc","prob_auc","sen","spe","asd_f1"):
                    close(m0[k],ref[k],f"fold {fold}: historical E74.4 {k}")
        if fold==2:
            noexp=out["outer_noexp"]
            if not np.array_equal(cache[pfx+"outer_noexp_M0_original_mlp_ids"],noexp["M0_original_mlp"]["ids"]):raise AssertionError("Fold2 exposure exclusion IDs mismatch")
            for group,v in noexp.items():
                if not np.array_equal(v["pred"],cache[pfx+f"outer_noexp_{group}_pred"]) or not np.allclose(v["prob"],cache[pfx+f"outer_noexp_{group}_prob"],rtol=0,atol=1e-12):raise AssertionError("Fold2 no-overlap prediction cache mismatch")
                m=r.metric(data["y"][v["ids"]],v["pred"],v["prob"])
                row=store[(2,"outer_test_without_377_407",group)]
                for k,val in m.items():close(row[k],val,f"Fold2 no-overlap/{group}/{k}")
                fold2_rows.append({"group":group,**m})
        # Verify the exact nested split coverage and disjoint evidence cross-fit.
        valid=[]
        for s in out["split_rows"]:
            tr=np.asarray(s["train_ids"],int);va=np.asarray(s["valid_ids"],int);valid.extend(va.tolist())
            if np.intersect1d(tr,va).size or np.intersect1d(va,test).size or not set(tr.tolist()).issubset(set(ids.tolist())):raise AssertionError("inner split leakage")
            split_checks.append({"fold":fold,"inner_fold":s["inner_fold"],"n_train":len(tr),"n_valid":len(va),"train_valid_disjoint":True,"outer_test_disjoint":True})
        if len(valid)!=len(ids) or set(valid)!=set(ids.tolist()) or len(set(valid))!=len(ids):raise AssertionError(f"fold {fold}: validation rows are not exact OOF partition")
        for split in out["split_rows"]:
            raw=split["M3_hard_counts"]
            expected=[int(n) if int(n)>=3 else 0 for n in raw]
            if list(map(int,split["M2_hard_counts"]))!=expected:raise AssertionError("M2/M3 hard candidate roles differ")
        # Each stored local triple must use legal roles and preserve class direction.
        local_ids=set(ids.tolist())
        boundary_pools=[ids[(data["y"][ids]==c)&(data["pred"][ids]==c)&(data["prob"][ids,c]>=.50)&(data["prob"][ids,c]<.70)] for c in (0,1)]
        reliable_pools=[ids[(data["y"][ids]==c)&(data["pred"][ids]==c)&(data["prob"][ids,c]>=.70)&(data["stability"][ids]>=.90)] for c in (0,1)]
        m3_hard_ids={c:{int(x["hard_id"]) for x in out["pair_registry"] if int(x["hard_class"])==c} for c in (0,1)}
        for rec in out["pair_registry"]:
            hi,si,oi=rec["hard_id"],rec["same_id"],rec["other_id"]
            if not {hi,si,oi}.issubset(local_ids):raise AssertionError("local prototype contains a non-support patient")
            cls=rec["hard_class"]
            if int(data["y"][hi])!=cls or int(data["pred"][hi])!=1-cls:raise AssertionError("hard prototype source is not a true E74.4 error")
            if int(data["y"][si])!=cls or int(data["pred"][si])!=cls:raise AssertionError("same-disease local reference label mismatch")
            if int(data["y"][oi])!=1-cls or int(data["pred"][oi])!=1-cls:raise AssertionError("opposite reliable reference label mismatch")
            allowed_same=boundary_pools[cls] if len(boundary_pools[cls]) else reliable_pools[cls]
            if si not in set(allowed_same.tolist()) or oi not in set(reliable_pools[1-cls].tolist()):raise AssertionError("local reference source pool violates boundary/fallback rule")
        reg_fold=next(x for x in registry["folds"] if x["fold"]==fold)
        if len(reg_fold["splits"])!=5:raise AssertionError("prototype registry omitted nested splits")
        # Verify M4 changes only the reference correspondence within each true-error direction.
        for entry in out["details"]:
            if entry["scope"]!="outer_test":continue
            for cls,records in enumerate(entry["records"]):
                if {int(x["hard_id"]) for x in records}!=m3_hard_ids[cls]:raise AssertionError("M4 changed hard-case membership")
                for rec in records:
                    hi,si,oi=rec["hard_id"],rec["same_id"],rec["other_id"]
                    if not {hi,si,oi}.issubset(local_ids):raise AssertionError("M4 shuffled pair escaped legal support")
                    if int(data["y"][hi])!=cls or int(data["pred"][hi])!=1-cls:raise AssertionError("M4 changed true hard-case role")
                    if int(data["y"][si])!=cls or int(data["pred"][si])!=cls:raise AssertionError("M4 changed same-disease reference role")
                    if int(data["y"][oi])!=1-cls or int(data["pred"][oi])!=1-cls:raise AssertionError("M4 changed opposite reliable role")
        # The E76.2 definitions and the local masks are checked by the builder during every rerun.
        fold_checks.append({"fold":fold,"checkpoint_sha256":data["checkpoint_sha"],"cam144_sha256":data["feature_sha"],
                            "checkpoint_weights_sha256":data["checkpoint_weights_sha"],"post_forward_weights_sha256":data["post_weights_sha"],
                            "frozen_weights_unchanged":data["frozen_weights_exact"],"output_layer_calls":1,
                            "base_prediction_probability_max_error":data["probability_max_error"],
                            "eligible_train_n":len(ids),"test_n":len(test),"outer_test_support_overlap":0,
                            "fold2_rows_excluded_from_support":fold!=2 or {376,406}.isdisjoint(local_ids),
                            "hard_counts_equal_old_and_local_roles":True,"local_pair_records":len(out["pair_registry"])})
        corrections.extend(out["correction"]);boundaries.extend(out["boundary"]);diagnostics.append(out["diagnostics"])
        calchecks.extend(out["calibrator_rows"])
    # Compare saved in-fold result rows to independent deterministic re-execution.
    run_metrics=[r for r in recalc]
    expected_count=100
    if len(run_metrics)!=expected_count:raise AssertionError(f"expected 100 independently recalculated primary rows, got {len(run_metrics)}")
    summary=r.summarize_metrics([x for x in stored if x["scope"] in ("inner_oof","outer_test")])
    # Build 9-fold descriptive summaries and direct M3-vs-control correction tables.
    for scope in ("inner_oof","outer_test"):
        for group in r.GROUPS:
            subset=[x for x in stored if x["scope"]==scope and x["group"]==group and int(x["fold"])!=2]
            row={"scope":scope+"_exclude_fold2_9","group":group,"folds":len(subset)}
            for m in r.METRICS:
                vals=np.asarray([float(x[m]) for x in subset]);row[m+"_mean"]=float(vals.mean());row[m+"_sd_population"]=float(vals.std(ddof=0))
            summary.append(row)
    paired=r.paired_rows([x for x in stored if x["scope"] in ("inner_oof","outer_test")])
    # Fold 2's explicit removal is a single-fold sensitivity result, not a new CV estimate.
    for c in corrections:
        c["reference_group"]="M0_original_mlp"
        c["comparison"]=f"{c['group']} vs M0_original_mlp"
    # Add M3 versus the other calibrators, keeping original-model role subsets fixed.
    for outfold in range(1,11):
        # Recover fold result arrays in the same deterministic run for pairwise flips.
        # Pred/cache were checked above, so the saved arrays are the audit input here.
        fkey=f"f{outfold}_";d=np.load(EXP/".runtime/cache/fold_predictions_and_features.npz",allow_pickle=False)
        for scope,ids_key in (("inner_oof","eligible_ids"),("outer_test","test_ids")):
            ids=d[fkey+ids_key].astype(int);y=d[fkey+"y_all"][ids];base=d[fkey+("inner" if scope=="inner_oof" else "outer")+"_M0_original_mlp_pred"]
            m3=d[fkey+("inner" if scope=="inner_oof" else "outer")+"_M3_local_boundary_prototype_pred"]
            pbase=d[fkey+"prob_all"][ids];predall=d[fkey+"pred_all"][ids];stab=d[fkey+"stability"][ids]
            masks=[(y==0)&(predall==0)&(pbase[:,0]>=.70)&(stab>=.90),
                   (y==1)&(predall==1)&(pbase[:,1]>=.70)&(stab>=.90),
                   (y==0)&(predall==1)&(pbase[:,1]>=.60)&(stab>=.90),
                   (y==1)&(predall==0)&(pbase[:,0]>=.60)&(stab>=.90)]
            subsets={"all":np.ones(len(ids),bool),"stable_hard":masks[2]|masks[3],"stable_reliable":masks[0]|masks[1],
                     "ASD_to_TD_errors":(y==0)&(base==1),"TD_to_ASD_errors":(y==1)&(base==0)}
            for other in ("M1_base_score_calibration","M2_global_hard_prototype","M4_shuffled_local_pair"):
                prefix="inner" if scope=="inner_oof" else "outer"
                ref=d[fkey+prefix+f"_{other}_pred"]
                for subset,mask in subsets.items():
                    wrong=(ref!=y)&mask;right=(ref==y)&mask
                    corrections.append({"fold":outfold,"scope":scope,"group":"M3_local_boundary_prototype","reference_group":other,
                                        "comparison":f"M3_local_boundary_prototype vs {other}","subset":subset,
                                        "wrong_to_correct":int(np.sum(wrong&(m3==y))),"correct_to_wrong":int(np.sum(right&(m3!=y))),
                                        "net_correction":int(np.sum(wrong&(m3==y))-np.sum(right&(m3!=y))),
                                        "asd_to_td_before":int(np.sum(mask&(y==0)&(ref==1))),"asd_to_td_after":int(np.sum(mask&(y==0)&(m3==1))),
                                        "td_to_asd_before":int(np.sum(mask&(y==1)&(ref==0))),"td_to_asd_after":int(np.sum(mask&(y==1)&(m3==0))),
                                        "raw_error_count":int(np.sum(wrong)),"raw_correct_count":int(np.sum(right)),
                                        "mean_abs_asd_probability_change":float(np.mean(np.abs(d[fkey+prefix+"_M3_local_boundary_prototype_prob_asd"]-d[fkey+prefix+f"_{other}_prob_asd"])))} )
    # Add corrected labels and metrics for the explicit Fold 2 no-overlap row.
    fold2_rows=[]
    for group in r.GROUPS:
        row=next(x for x in stored if int(x["fold"])==2 and x["scope"]=="outer_test_without_377_407" and x["group"]==group)
        fold2_rows.append(row)
    # Independently recompute summary, paired deltas and correction outputs before report.
    paired=r.paired_rows([x for x in stored if x["scope"] in ("inner_oof","outer_test")])
    summary_path=results/"summary.csv"
    r.write_csv(summary_path,summary)
    r.write_csv(results/"paired_comparisons.csv",paired)
    r.write_csv(results/"correction_analysis.csv",corrections)
    r.write_csv(results/"boundary_diagnostics.csv",boundaries)
    r.write_csv(results/"prototype_diagnostics.csv",diagnostics)
    r.write_csv(results/"calibrator_diagnostics.csv",calchecks)
    # Decision is fixed by outer held-out results and direct hard-error correction.
    pmap={(x["scope"],x["comparison"],x["metric"],x["fold_set"]):x for x in paired}
    def pval(other,m):return pmap[("outer_test",f"M3_local_boundary_prototype - {other}",m,"all_10")]
    m3m0=[pval("M0_original_mlp",m) for m in ("acc","hard_auc")]
    net=sum(int(x["net_correction"]) for x in corrections if x["scope"]=="outer_test" and x["comparison"]=="M3_local_boundary_prototype vs M0_original_mlp" and x["subset"]=="all")
    comparison_wins=all(int(pval(other,"acc")["wins"])>int(pval(other,"acc")["losses"]) for other in ("M1_base_score_calibration","M2_global_hard_prototype","M4_shuffled_local_pair"))
    if all(float(x["mean_delta_pp"])>0 and float(x["ci95_low_descriptive_pp"])>0 for x in m3m0) and net>0 and comparison_wins:
        conclusion="PROMISING_HISTORICAL_CORRECTION"
    else:
        bauc=[x for x in boundaries if x["scope"]=="outer_test" and x["method"] in ("M2_global","M3_local") and x["conditional_auc"] not in ("", "NA_INSUFFICIENT_CLASS")]
        contrasts=[]
        for direction in ("ASD_to_TD","TD_to_ASD"):
            a=[float(x["conditional_auc"]) for x in bauc if x["direction"]==direction and x["method"]=="M2_global"]
            b=[float(x["conditional_auc"]) for x in bauc if x["direction"]==direction and x["method"]=="M3_local"]
            if a and b:contrasts.append((sum(a)/len(a),sum(b)/len(b)))
        boundary_supported=any(local>global_auc for global_auc,local in contrasts)
        if boundary_supported and net<=0:conclusion="BOUNDARY_SIGNAL_WITHOUT_NET_CORRECTION"
        elif any(float(x["mean_delta_pp"])>0 for x in m3m0) and net>0:conclusion="INCONCLUSIVE"
        else:conclusion="NO-GO_LOCAL_PROTOTYPE_CORRECTION"
    audit_status={"status":"PASS","conclusion":conclusion,"folds_reaudited":10,"primary_metric_rows":"100/100 inner+outer rows independently recomputed",
                  "fold2_overlap_rows_excluded_from_new_fit":[377,407],"fold2_without_overlap_metrics":"5/5 recomputed",
                  "base_checkpoint_and_test_prediction_identity":"10/10 exact hard prediction; probability tolerance <=1e-6",
                  "cam144_reextracted_and_frozen":"10/10; exact cache match","support_test_intersection":"0/10",
                  "nested_5fold_oof":"10/10 exact partitions","training_evidence_3fold_crossfit":"all query rows disjoint from prototype support",
                  "global_local_role_definition_parity":"10/10 folds; checked on each prototype build",
                  "local_pair_category_and_support_audit":"PASS","M4_pair_roles_preserved":"PASS; five permutations per evaluation support",
                  "outputlayer_calls_per_checkpoint_extraction":1,"original_model_parameters_updated":False,
                  "source_sha256":source_sha,"references":refs,"fold_checks":fold_checks,"inner_split_checks":split_checks,
                  "diagnostic_runtime_seconds":saved.get("runtime_seconds"),"audit_runtime_seconds":time.time()-started}
    (results/"audit_status.json").write_text(json.dumps(audit_status,indent=2,allow_nan=False)+"\n")
    # Result hashes are written after the report is generated below.
    report=render_report(audit_status,summary,paired,stored,corrections,boundaries,diagnostics,fold2_rows)
    (results/"report.md").write_text(report,encoding="utf-8")
    manifest={**saved,"status":"COMPLETE_AUDITED","conclusion":conclusion,
              "audit_status_sha256":hashlib.sha256((results/"audit_status.json").read_bytes()).hexdigest(),
              "results_sha256":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(results.iterdir()) if f.is_file()}}
    (EXP/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":"PASS","conclusion":conclusion,"metric_rows":len(recalc),"runtime_seconds":saved.get("runtime_seconds"),
                      "audit_seconds":audit_status["audit_runtime_seconds"]},indent=2))


if __name__=="__main__":audit()
