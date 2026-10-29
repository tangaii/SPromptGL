#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
import run

HERE=Path(__file__).resolve().parent;RT=HERE/".runtime";RES=HERE/"results"
CFG=json.loads((HERE/"config.json").read_text())

def read_csv(path):
    with Path(path).open(newline="") as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    if not rows:return
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def evaluate_checkpoint(gid,fold,x,y,tr,te):
    c=run.context();m=c["method"];task=json.loads((RT/"tasks"/f"E{gid}_f{fold:02d}.json").read_text())
    anchor=run.load(RT/"anchors"/f"fold{fold:02d}.pt")
    a=run.initialize_agent(gid,fold,x,y,tr,te,anchor);a.srd_fit_ids,a.srd_cal_ids=m.partition_srd_patients(tr,y,481100+fold)
    ckpath=RT/"checkpoints"/f"E{gid}_f{fold:02d}.best.pt";b=run.load(ckpath);m.load_bundle(a,b["model"])
    with c["torch"].no_grad():full=m.evaluate(a,6,c["utils"])
    metrics=m.metrics(full,y,te);pred=full["logp"][te].argmax(1).detach().cpu().numpy()
    pred_sha=hashlib.sha256(pred.tobytes()).hexdigest()
    assert pred_sha==task["prediction_sha256"]==hashlib.sha256(b["prediction"].tobytes()).hexdigest()
    assert run.sha(ckpath)==task["best_sha"]
    for key,value in metrics.items():
        if key in task:assert np.isclose(value,task[key],atol=1e-12,rtol=0),(gid,fold,key,value,task[key])
    full_correct=pred==y[te];alpha=float(full["alpha"].detach());orig_alpha=a.APCR.alpha
    try:
        a.APCR.alpha=lambda: c["torch"].zeros((),device=c["device"])
        with c["torch"].no_grad():off=m.evaluate(a,6,c["utils"])
    finally:a.APCR.alpha=orig_alpha
    off_metrics=m.metrics(off,y,te);off_pred=off["logp"][te].argmax(1).detach().cpu().numpy();off_correct=off_pred==y[te]
    residual=full["fused"]-full["base"]
    intervention={"group_id":gid,"group":run.GROUPS[gid]["name"],"fold":fold,**off_metrics,
        "delta_acc_vs_full_pp":100*(off_metrics["acc"]-metrics["acc"]),
        "delta_hard_auc_vs_full_pp":100*(off_metrics["hard_auc"]-metrics["hard_auc"]),
        "flips":int((off_pred!=pred).sum()),"newly_correct":int((~full_correct&off_correct).sum()),"newly_wrong":int((full_correct&~off_correct).sum()),
        "mean_abs_probability_change":float((off["logp"][te].exp()[:,1]-full["logp"][te].exp()[:,1]).abs().mean()),
        "mean_abs_log_probability_change":float((off["logp"][te]-full["logp"][te]).abs().mean()),
        "alpha_best":alpha,"relative_residual_norm":float(residual.norm()/(full["base"].norm()+1e-8)),
        "residual_norm_by_modality":json.dumps([float((alpha*g.unsqueeze(-1)*d).norm()) for d,g in zip(full["deltas"],full["gates"])]),
        "gate_means":json.dumps([float(g.mean()) for g in full["gates"]])}
    row={"group_id":gid,"group":run.GROUPS[gid]["name"],"fold":fold,"best_epoch":task["best_epoch"],"epochs_executed":task["epochs_executed"],"prediction_sha256":pred_sha,"best_sha":task["best_sha"],"independent_readback":True,**metrics}
    return row,intervention,task

def main():
    x,y,folds,mapping=run.data();tasks=[json.loads(p.read_text()) for p in sorted((RT/"tasks").glob("E*_f*.json"))]
    assert len(tasks)==20 and all(t["status"]=="COMPLETE" for t in tasks)
    assert len({(t["group_id"],t["fold"]) for t in tasks})==20
    sigs={t["signature"] for t in tasks};assert len(sigs)==1 and next(iter(sigs))==json.loads((HERE/"manifest.json").read_text())["code_signature"]
    rows=[];interventions=[];task_map={(t["group_id"],t["fold"]):t for t in tasks}
    for gid in (1,2):
        for fold,(tr,te) in enumerate(folds,1):
            row,off,task=evaluate_checkpoint(gid,fold,x,y,tr,te);rows.append(row);interventions.append(off)
            print("READBACK",gid,fold,flush=True)
    by={(r["group_id"],r["fold"]):r for r in rows}
    comparisons=[]
    old48=read_csv(RT/"48_4_fold_metrics.csv");old483=read_csv(RT/"48_3_fold_metrics.csv")
    old_groups={
        "B0B_REFERENCE_48_4":{int(r["fold"]):r for r in old48 if int(r["group_id"])==0},
        "FULL_APCR_GFR_48_3":{int(r["fold"]):r for r in old483 if int(r["group_id"])==6},
        "FULL_VERIFIED_APCR_48_4":{int(r["fold"]):r for r in old48 if int(r["group_id"])==6},
    }
    contrasts=[(2,1,"E2_LEARNABLE_MINUS_E1_FIXED")]
    for gid in (1,2):
        for ref in old_groups:contrasts.append((gid,ref,f"E{gid}_MINUS_{ref}"))
    for left,right,label in contrasts:
        d={"comparison":label}
        for key in ("acc","hard_auc","spe","sen","probability_auc"):
            if isinstance(right,int):vals=np.asarray([(by[(left,f)][key]-by[(right,f)][key])*100 for f in range(1,11)])
            else:vals=np.asarray([(float(by[(left,f)][key])-float(old_groups[right][f][key]))*100 for f in range(1,11)])
            d[key+"_delta_pp"]=float(vals.mean());d[key+"_w_t_l"]=f"{int((vals>0).sum())}/{int((vals==0).sum())}/{int((vals<0).sum())}"
        comparisons.append(d)
    summaries=[]
    for gid in (1,2):
        rr=[r for r in rows if r["group_id"]==gid];s={"group_id":gid,"group":run.GROUPS[gid]["name"],"n_folds":len(rr)}
        for key in ("acc","hard_auc","spe","sen","probability_auc"):
            v=np.asarray([r[key] for r in rr])*100;s[key+"_mean_pct"]=float(v.mean());s[key+"_sd_pct"]=float(v.std(ddof=0))
        summaries.append(s)
    write_csv(RES/"fold_metrics.csv",rows);write_csv(RES/"summary.csv",summaries);write_csv(RES/"paired_comparisons.csv",comparisons);write_csv(RES/"apcr_off_interventions.csv",interventions)
    make_report(summaries,comparisons,interventions,task_map)
    manifest=json.loads((HERE/"manifest.json").read_text());protected_after={p:run.sha(p) for p in manifest["protected_sha256"]}
    assert protected_after==manifest["protected_sha256"]
    alpha_diag={"fixed_alpha_exact_all_steps":all(all(abs(float(v)-.0022)<1e-7 for v in task_map[(1,f)]["diagnostics"]["alpha_trace"]) for f in range(1,11)),"learnable_alpha_updated_all_folds":all(task_map[(2,f)]["diagnostics"]["alpha_grad_nonzero_steps"]>0 and abs(task_map[(2,f)]["diagnostics"]["alpha_best"]-.0022)>1e-10 for f in range(1,11)),"initial_bundle_sha_match_all_folds":all(task_map[(1,f)]["initial_model_bundle_sha"]==task_map[(2,f)]["initial_model_bundle_sha"] for f in range(1,11))}
    alpha_diag["initial_gfr_sha_match_all_folds"]=all(task_map[(1,f)]["initial_gfr_sha"]==task_map[(2,f)]["initial_gfr_sha"] for f in range(1,11))
    alpha_diag["initial_cam_sha_match_all_folds"]=all(task_map[(1,f)]["initial_cam_sha"]==task_map[(2,f)]["initial_cam_sha"] for f in range(1,11))
    alpha_diag["initial_apcr_sha_match_all_folds"]=all(task_map[(1,f)]["initial_apcr_sha"]==task_map[(2,f)]["initial_apcr_sha"] for f in range(1,11))
    alpha_diag["fixed_alpha_exact_all_steps"]=all(all(abs(float(v)-.0022)<1e-7 for v in task_map[(1,f)]["diagnostics"]["alpha_trace"]) for f in range(1,11))
    assert alpha_diag["fixed_alpha_exact_all_steps"] and alpha_diag["learnable_alpha_updated_all_folds"] and all(alpha_diag[k] for k in ("initial_bundle_sha_match_all_folds","initial_gfr_sha_match_all_folds","initial_cam_sha_match_all_folds","initial_apcr_sha_match_all_folds"))
    complete_times=[(RT/"tasks"/f"E{g}_f{f:02d}.json").stat().st_mtime for g in (1,2) for f in range(1,11)]
    manifest.update({"status":"COMPLETE_AUDITED","tasks":{"total":20,"complete":20},"trained_folds":20,"independent_readback":"20/20_PASS","training_signatures":list(sigs),"protected_sha256_after":protected_after,"protected_sha_unchanged":True,"alpha_audit":alpha_diag,"elapsed_wall_seconds":max(complete_times)-min(complete_times),"training_seconds_by_group":{str(g):sum(float(t["wall_seconds"]) for t in tasks if t["group_id"]==g) for g in (1,2)},"preflight_sha256":run.sha(RT/"preflight_output.json"),"parent_48_4_source_signature":manifest["parent_48_4_code_signature"],"results_sha256":{str(p.relative_to(HERE)):run.sha(p) for p in RES.iterdir() if p.is_file()}})
    run.atomic_json(HERE/"manifest.json",manifest)
    print(json.dumps({"status":"AUDIT_PASS","tasks":"20/20","learnable_alpha_updated":alpha_diag["learnable_alpha_updated_all_folds"],"screened_signature":list(sigs)},indent=2))

def make_report(summaries,pairs,interventions,tasks):
    s={q["group_id"]:q for q in summaries};p={q["comparison"]:q for q in pairs}["E2_LEARNABLE_MINUS_E1_FIXED"]
    old48=json.loads((RT/"48_4_manifest.json").read_text())
    old48_rows=read_csv(RT/"48_4_fold_metrics.csv")
    old48_e0=[r for r in old48_rows if int(r["group_id"])==0];old48_e6=[r for r in old48_rows if int(r["group_id"])==6]
    old483=read_csv(RT/"48_3_summary.csv")
    def legacy(rows,key):
        vals=np.asarray([float(r[key]) for r in rows])*100
        return float(vals.mean()),float(vals.std(ddof=0))
    b0={k:legacy(old48_e0,k) for k in ("acc","hard_auc","spe","sen","probability_auc")}
    p48={k:legacy(old48_e6,k) for k in ("acc","hard_auc","spe","sen","probability_auc")}
    p483=next((r for r in old483 if "E6" in r.get("group","") or int(r.get("group_id",-1))==6),None)
    lines=["# Experiment 48.5 — APCR Alpha Ablation","","## 执行状态","",f"20/20训练折完成；E1/E2共享fold锚点、APCR初始化和epoch随机计划。GPU=单卡，worker数={len(CFG['worker_plan']['folds'])}，训练源码签名=`{next(iter({t['signature'] for t in tasks.values()}))}`。完整读回及指标核验见manifest。", "", "## 正式结果（均值±population SD，%）", "", "|模型|α初始化|α最佳均值|ACC|Hard-AUC|SPE-TD|SEN-ASD|Probability-AUC|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines.append(f"|48.4 E6|0.01000|0.01096|{p48['acc'][0]:.4f} ± {p48['acc'][1]:.4f}|{p48['hard_auc'][0]:.4f} ± {p48['hard_auc'][1]:.4f}|{p48['spe'][0]:.4f} ± {p48['spe'][1]:.4f}|{p48['sen'][0]:.4f} ± {p48['sen'][1]:.4f}|{p48['probability_auc'][0]:.4f} ± {p48['probability_auc'][1]:.4f}|")
    for gid in (1,2):
        q=s[gid];alphas=[float(tasks[(gid,f)]["diagnostics"]["alpha_best"]) for f in range(1,11)]
        lines.append(f"|E{gid} {q['group']}|0.00220|{np.mean(alphas):.6f} ± {np.std(alphas):.6f}|{q['acc_mean_pct']:.4f} ± {q['acc_sd_pct']:.4f}|{q['hard_auc_mean_pct']:.4f} ± {q['hard_auc_sd_pct']:.4f}|{q['spe_mean_pct']:.4f} ± {q['spe_sd_pct']:.4f}|{q['sen_mean_pct']:.4f} ± {q['sen_sd_pct']:.4f}|{q['probability_auc_mean_pct']:.4f} ± {q['probability_auc_sd_pct']:.4f}|")
    lines += ["", "## 配对比较", "",f"E2−E1：ACC {p['acc_delta_pp']:+.4f}pp ({p['acc_w_t_l']}胜/平/负)，Hard-AUC {p['hard_auc_delta_pp']:+.4f}pp ({p['hard_auc_w_t_l']})，SPE {p['spe_delta_pp']:+.4f}pp，SEN {p['sen_delta_pp']:+.4f}pp。", "", "相对历史模型的十折同fold差值（pp；历史组为独立训练，不解释为独立重复）：", "", "|组|参考|ΔACC (胜/平/负)|ΔHard-AUC (胜/平/负)|ΔSPE|ΔSEN|", "|---|---|---:|---:|---:|---:|"]
    for gid in (1,2):
        q=s[gid]
        for ref,label in (("B0-B","B0B_REFERENCE_48_4"),("48.4 E6","FULL_VERIFIED_APCR_48_4"),("48.3 E6","FULL_APCR_GFR_48_3")):
            z={x["comparison"]:x for x in pairs}[f"E{gid}_MINUS_{label}"]
            lines.append(f"|E{gid}|{ref}|{z['acc_delta_pp']:+.4f} ({z['acc_w_t_l']})|{z['hard_auc_delta_pp']:+.4f} ({z['hard_auc_w_t_l']})|{z['spe_delta_pp']:+.4f}|{z['sen_delta_pp']:+.4f}|")
    lines += ["", "## α轨迹与机制", "", "|模型|折|α初值|α最佳|α末值|变化|有梯度步数/总步数|平均|grad α_raw|APCR关闭ΔACC pp|翻转数|残差/GFR范数|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for gid in (1,2):
        for fold in range(1,11):
            task=tasks[(gid,fold)];d=task["diagnostics"];off=next(r for r in interventions if r["group_id"]==gid and r["fold"]==fold)
            lines.append(f"|E{gid}|{fold}|0.002200|{d['alpha_best']:.7f}|{d['alpha_final_training']:.7f}|{d['alpha_best']-.0022:+.7f}|{d['alpha_grad_nonzero_steps']}/{d['alpha_grad_steps']}|{d['alpha_grad_abs_mean']:.3g}|{off['delta_acc_vs_full_pp']:+.4f}|{off['flips']}|{off['relative_residual_norm']:.6f}|")
    lines += ["", "## 结论与限制", "",f"固定E1 ACC={s[1]['acc_mean_pct']:.4f}%；可学习E2 ACC={s[2]['acc_mean_pct']:.4f}%；历史B0-B ACC={b0['acc'][0]:.4f}%；48.3 E6 ACC=91.9592%。", "历史CV前RFE/全局预处理、transductive图、`val=test` checkpoint选择和单seed协议均保留，结果不能解释为无偏外部泛化性能。α固定与可学习的差异是本实验唯一干预；任何后续修改需另行预注册。"]
    (RES/"report.md").write_text("\n".join(lines)+"\n")

if __name__=="__main__":main()
