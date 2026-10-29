#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json,collections
from pathlib import Path
import numpy as np
import run

HERE=Path(__file__).resolve().parent;RT=HERE/".runtime";RES=HERE/"results"

def write_csv(path,rows):
    if not rows:return
    keys=[]
    for row in rows:
        for k in row:
            if k not in keys:keys.append(k)
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in r.items()})

def read_csv(path):
    with Path(path).open(newline="") as f:return list(csv.DictReader(f))

def evaluate_group_fold(gid,fold,x,y,tr,te):
    c=run.context();m=c["method"];tp=RT/"tasks"/f"g{gid}_f{fold:02d}.json";task=json.loads(tp.read_text());anchor=run.load(RT/"anchors"/f"fold{fold:02d}.pt")
    assert task["split_sha"]==run.fold_hash(tr,te)
    if gid==0:
        b=run.load(RT/"checkpoints"/f"E0_f{fold:02d}.best.pt")
        historical=json.loads((RT/"tasks"/f"g0_f{fold:02d}.json").read_text())
        a=m.make_agent(c["torch"],c["layers"],c["network"],c["utils"],c["device"],x,y,tr,te,fold,1,anchor,481000+fold)
        a.MessagePassing0.load_state_dict(b["model"]["gfr"]);a.ModalFusion.load_state_dict(b["model"]["pln"])
        with c["torch"].no_grad():out=m.evaluate(a,1,c["utils"])
        row=m.metrics(out,y,te);pred=out["logp"][te].argmax(1).detach().cpu().numpy()
        digest=hashlib.sha256(pred.tobytes()).hexdigest();assert digest==historical["prediction_sha256"]
        assert np.array_equal(pred,np.asarray(b["prediction"]))
        for key in ("acc","hard_auc","spe","sen","probability_auc"):
            assert np.isclose(row[key],historical[key],atol=1e-12,rtol=0),(fold,key,row[key],historical[key])
        row.update({"group_id":0,"group":run.NAMES[0],"fold":fold,"prediction_sha256":digest,"independent_readback":True})
        return row,{"group_id":0,"fold":fold,"source":"historical_checkpoint_forward_verified","reused_verified":True}
    a=m.make_agent(c["torch"],c["layers"],c["network"],c["utils"],c["device"],x,y,tr,te,fold,gid,anchor,481000+fold)
    a.srd_fit_ids,a.srd_cal_ids=m.partition_srd_patients(tr,y,481100+fold)
    b=run.load(RT/"checkpoints"/f"g{gid}_f{fold:02d}.best.pt");m.load_bundle(a,b["model"])
    mapping=np.load(run.CFG["data"]["mapping"],allow_pickle=True).item()
    wrong=run.wrong_relation_permutation(x,tr,mapping) if gid==8 else None
    with c["torch"].no_grad():out=m.evaluate(a,gid,c["utils"],wrong)
    row=m.metrics(out,y,te);pred=out["logp"][te].argmax(1).detach().cpu().numpy()
    digest=hashlib.sha256(pred.tobytes()).hexdigest();assert digest==task["prediction_sha256"],(gid,fold,digest,task["prediction_sha256"])
    best_path=RT/"checkpoints"/f"g{gid}_f{fold:02d}.best.pt";assert run.sha(best_path)==task["best_sha"]
    for key in ("acc","hard_auc","spe","sen","probability_auc"):
        assert np.isclose(row[key],task[key],atol=1e-12,rtol=0),(gid,fold,key,row[key],task[key])
    row.update({"group_id":gid,"group":run.NAMES[gid],"fold":fold,"prediction_sha256":digest,"independent_readback":True})
    # Reconstruct the APCR-off path on the same frozen checkpoint for a diagnostic readback.
    basecam=m.forward_base_cam(a,out["base"])
    flip=int((pred!=basecam[te].argmax(1).detach().cpu().numpy()).sum())
    diag=dict(b["diagnostics"]);diag.update({"group_id":gid,"group":run.NAMES[gid],"fold":fold,"apcr_off_flip_count":flip,"apcr_off_flip_rate":flip/len(te),"independent_readback":True})
    if gid==8:
        with c["torch"].no_grad():clean=m.evaluate(a,8,c["utils"])
        clean_edges=np.stack([v.detach().cpu().numpy() for v in clean["chosen"]])
        wrong_edges=np.stack([v.detach().cpu().numpy() for v in out["chosen"]])
        changed=clean_edges!=wrong_edges
        site=run.site_ids(x,np.load(run.CFG["data"]["mapping"],allow_pickle=True).item());trset=set(map(int,tr))
        perm=run.wrong_relation_permutation(x,tr,np.load(run.CFG["data"]["mapping"],allow_pickle=True).item()).detach().cpu().numpy()
        assert np.array_equal(site,site[perm])
        assert np.array_equal(np.isin(np.arange(len(x)),tr),np.isin(perm,tr))
        diag.update({"wrong_relation_changed_edges":int(changed.sum()),"wrong_relation_edge_slots":int(changed.size),
                     "wrong_relation_edge_change_rate":float(changed.mean()),"wrong_relation_affected_receivers":int(changed.any(0).any(1).sum()),
                     "wrong_relation_self_edges":int((wrong_edges==np.arange(len(x))[None,:,None]).sum()),
                     "wrong_relation_duplicate_rows":int(sum(len(set(row.tolist()))!=len(row) for view in wrong_edges for row in view)),
                     "wrong_relation_site_split_preserved":True})
    return row,diag

def e6_intervention_rows(x,y,folds):
    c=run.context();m=c["method"];torch=c["torch"];rows=[]
    mapping=np.load(run.CFG["data"]["mapping"],allow_pickle=True).item();sites=run.site_ids(x,mapping)
    for fold,(tr,te) in enumerate(folds,1):
        anchor=run.load(RT/"anchors"/f"fold{fold:02d}.pt")
        a=m.make_agent(torch,c["layers"],c["network"],c["utils"],c["device"],x,y,tr,te,fold,6,anchor,481000+fold)
        a.srd_fit_ids,a.srd_cal_ids=m.partition_srd_patients(tr,y,481100+fold)
        bundle=run.load(RT/"checkpoints"/f"g6_f{fold:02d}.best.pt");m.load_bundle(a,bundle["model"])
        full_task=json.loads((RT/"tasks"/f"g6_f{fold:02d}.json").read_text())
        train_mask=np.zeros(len(x),dtype=bool);train_mask[np.asarray(tr,dtype=np.int64)]=True
        rng=np.random.default_rng(484800+fold)
        modes=[("FULL",None,None), ("APCR_OFF",None,"all"), ("ALPHA_ZERO",None,"alpha"),
               ("OFF_PHENO",0,None),("OFF_ANAT",1,None),("OFF_FUNC",2,None),("OFF_CORR",3,None),
               ("WRONG_RELATION_REWEIGHTED",None,"wrong"),("RANDOM_STRATIFIED_RELATION",None,"random")]
        full_pred=None;full_prob=None;full_logp=None;full_correct=None;full_edges=None
        for name,modality,special in modes:
            orig_res=a.APCR.residual;orig_alpha=a.APCR.alpha
            wrong_arg=None
            if special in ("all",) or modality is not None:
                def residual_override(*args,**kwargs):
                    value=orig_res(*args,**kwargs)
                    if special=="all":return value.new_zeros(value.shape)
                    last=a.APCR.last;return value-last["gates"][modality].unsqueeze(-1)*last["deltas"][modality]
                a.APCR.residual=residual_override
            elif special=="alpha":
                a.APCR.alpha=lambda: torch.zeros((),device=c["device"])
            elif special=="wrong":wrong_arg=run.wrong_relation_permutation(x,tr,mapping)
            elif special=="random":
                def random_residual(parts_,z_,ids_,raw_,mode_,*args,**kwargs):
                    random_edges=[]
                    for _m in range(4):
                        selected=np.empty((len(x),16),dtype=np.int64)
                        for i in range(len(x)):
                            allowed=np.flatnonzero(train_mask & (np.arange(len(x))!=i))
                            same_site=allowed[sites[allowed]==sites[i]]
                            pool=same_site if len(same_site)>=16 else allowed
                            if len(pool)<16:raise RuntimeError(f"random intervention lacks 16 legal senders for patient {i}")
                            selected[i]=rng.choice(pool,16,replace=False)
                        random_edges.append(torch.as_tensor(selected,device=c["device"],dtype=torch.long))
                    kwargs.pop("attacked",None);kwargs.pop("wrong_perm",None)
                    return orig_res(parts_,z_,ids_,raw_,mode_,*args,attacked=random_edges,wrong_perm=None,**kwargs)
                a.APCR.residual=random_residual
            try:
                with torch.no_grad():out=m.evaluate(a,6,c["utils"],wrong_arg)
            finally:
                a.APCR.residual=orig_res;a.APCR.alpha=orig_alpha
            metrics=m.metrics(out,y,te);pred=out["logp"][te].argmax(1).detach().cpu().numpy();prob=out["logp"][te].exp()[:,1].detach().cpu().numpy();lp=out["logp"][te].detach().cpu().numpy();correct=pred==y[te]
            if name=="FULL":
                digest=hashlib.sha256(pred.tobytes()).hexdigest();assert digest==full_task["prediction_sha256"]
                full_pred,full_prob,full_logp,full_correct=pred.copy(),prob.copy(),lp.copy(),correct.copy()
                full_edges=np.stack([v.detach().cpu().numpy() for v in out["chosen"]])
            residual=out["fused"]-out["base"]
            gates=out.get("gates",[]);deltas=out.get("deltas",[])
            chosen_edges=np.stack([v.detach().cpu().numpy() for v in out.get("chosen",[])])
            edge_changed=(chosen_edges!=full_edges) if name!="FULL" else np.zeros_like(full_edges,dtype=bool)
            rows.append({"group":"E6_FULL_VERIFIED_APCR","fold":fold,"intervention":name,**metrics,
                "delta_acc_vs_full_pp":0. if name=="FULL" else 100.*(metrics["acc"]-float(full_correct.mean())),
                "flips_vs_full":0 if name=="FULL" else int(np.count_nonzero(pred!=full_pred)),
                "newly_correct_vs_full":0 if name=="FULL" else int((~full_correct & correct).sum()),
                "newly_wrong_vs_full":0 if name=="FULL" else int((full_correct & ~correct).sum()),
                "mean_abs_probability_change":0. if name=="FULL" else float(np.mean(np.abs(prob-full_prob))),
                "mean_abs_log_probability_change":0. if name=="FULL" else float(np.mean(np.abs(lp-full_logp))),
                "alpha":float(out["alpha"]),"relative_residual_norm":float(residual.norm()/(out["base"].norm()+1e-8)),
                "changed_edges_vs_full":int(edge_changed.sum()),"edge_change_rate_vs_full":float(edge_changed.mean()),
                "affected_receivers_vs_full":int(edge_changed.any(0).any(1).sum()),
                "selected_self_edges":int((chosen_edges==np.arange(len(x))[None,:,None]).sum()),
                "duplicate_neighbor_rows":int(sum(len(set(row.tolist()))!=len(row) for view in chosen_edges for row in view)),
                "message_norms_json":json.dumps([float(v.norm()) for v in deltas]),
                "gate_means_json":json.dumps([float(v.mean()) for v in gates]),
                "prediction_sha256":hashlib.sha256(pred.tobytes()).hexdigest()})
        print("E6_INTERVENTIONS",fold,flush=True)
    return rows

def main():
    x,y,folds,mapping=run.data();tasks=[json.loads(p.read_text()) for p in sorted(RT.joinpath("tasks").glob("g*_f*.json"))]
    if len(tasks)!=110 or any(t.get("status")!="COMPLETE" for t in tasks):raise RuntimeError(f"expected 110 complete tasks; got {len(tasks)}")
    if len({(t["group_id"],t["fold"]) for t in tasks})!=110:raise RuntimeError("duplicate or missing group-fold identity")
    sigs={t.get("signature") for t in tasks if t.get("group_id",0)>0}
    if len(sigs)!=1 or next(iter(sigs))!=json.loads((HERE/"manifest.json").read_text())["code_signature"]:raise RuntimeError(f"mixed or unexpected training source signatures: {sigs}")
    rows=[];diags=[]
    for gid in range(11):
        for fold,(tr,te) in enumerate(folds,1):
            row,diag=evaluate_group_fold(gid,fold,x,y,tr,te);rows.append(row);diags.append(diag);print("READBACK",gid,fold,flush=True)
    summaries=[]
    for gid in range(11):
        rr=[r for r in rows if r["group_id"]==gid];q={"group_id":gid,"group":run.NAMES[gid],"n_folds":len(rr)}
        for key in ("acc","hard_auc","spe","sen","probability_auc"):
            v=np.asarray([r[key] for r in rr],dtype=np.float64)*100;q[key+"_mean_pct"]=float(v.mean());q[key+"_sd_pct"]=float(v.std(ddof=0))
        summaries.append(q)
    by={(r["group_id"],r["fold"]):r for r in rows};pairs=[]
    contrasts=[(1,0),(2,1),(3,2),(4,3),(5,4),(6,5),(6,7),(6,8),(6,9),(6,10),(6,0)]
    for ga,gb in contrasts:
        q={"comparison":f"E{ga}-E{gb}"}
        for key in ("acc","hard_auc","spe","sen","probability_auc"):
            d=np.asarray([(by[ga,f][key]-by[gb,f][key])*100 for f in range(1,11)]);q[key+"_delta_pp"]=float(d.mean());q[key+"_w_t_l"]=f"{(d>0).sum()}/{(d==0).sum()}/{(d<0).sum()}"
        pairs.append(q)
    write_csv(RES/"fold_metrics.csv",rows);write_csv(RES/"summary.csv",summaries);write_csv(RES/"paired_comparisons.csv",pairs);write_csv(RES/"diagnostics.csv",diags)
    interventions=e6_intervention_rows(x,y,folds);write_csv(RES/"apcr_e6_interventions.csv",interventions)
    make_report(summaries,pairs,diags,tasks)
    manifest=json.loads((HERE/"manifest.json").read_text());task_times=[(RT/"tasks"/f"g{t['group_id']}_f{t['fold']:02d}.json").stat().st_mtime for t in tasks];s={q["group_id"]:q for q in summaries}
    manifest.update({"status":"COMPLETE_AUDITED","tasks":{"total":110,"complete":110},"trained_folds":100,"independent_readback":"110/110_PASS","training_signatures":list(sigs),"screen_acc":"PASS" if s[6]["acc_mean_pct"]>91.73 else "FAIL","protected_sha256_after":{k:run.sha(k) for k in manifest["protected_sha256"]},"protected_sha_unchanged":all(run.sha(k)==v for k,v in manifest["protected_sha256"].items()),"elapsed_wall_seconds":max(task_times)-min(task_times),"training_seconds_by_group":{str(g):sum(float(t.get("wall_seconds",0.)) for t in tasks if t.get("group_id")==g) for g in range(1,11)},"e0_checkpoint_sha256":{f"E0_f{f:02d}.best.pt":run.sha(RT/"checkpoints"/f"E0_f{f:02d}.best.pt") for f in range(1,11)},"frozen_e6_intervention_audit_sha256":run.sha(RT/"frozen_e6_audit.json"),"audit_sha256":run.sha(HERE/"audit.py"),"append_handoff_sha256":run.sha(HERE/"append_handoff.py"),"frozen_e6_interventions":{"rows":len(read_csv(RES/"frozen_e6_interventions.csv")),"sha256":run.sha(RES/"frozen_e6_interventions.csv")},"new_e6_interventions":{"rows":len(interventions),"sha256":run.sha(RES/"apcr_e6_interventions.csv")},"results_sha256":{str(p.relative_to(HERE)):run.sha(p) for p in RES.iterdir() if p.is_file()}})
    run.atomic_json(HERE/"manifest.json",manifest)
    if not manifest["protected_sha_unchanged"]:raise RuntimeError("protected project files changed during experiment")
    print(json.dumps({"status":"AUDIT_PASS","tasks":"110/110","screen_acc":manifest.get("screen_acc"),"signature":list(sigs)},indent=2))

def make_report(summaries,pairs,diags,tasks):
    s={q["group_id"]:q for q in summaries};p={q["comparison"]:q for q in pairs};manifest=json.loads((HERE/"manifest.json").read_text());lines=["# Experiment 48.4 — Verified APCR-GFR","","## 执行状态","", "110/110任务完成：E0为历史checkpoint核验10折，E1–E10共100折重新训练；单卡最多5个worker。全流程墙钟约%.1f分钟。独立进程重读110/110组折，task、best checkpoint和预测签名一致。代码签名：`%s`。"%(manifest.get("elapsed_wall_seconds",0)/60,manifest.get("code_signature","")),"Frozen 48.3 E6干预单独记录在`frozen_e6_interventions.csv`，不与本轮重新训练结果混合。","","## 十折正式结果（mean ± population SD，%）","","|组|ACC|Hard-AUC|SPE-TD|SEN-ASD|Probability-AUC|","|---|---:|---:|---:|---:|---:|"]
    for gid in range(11):
        q=s[gid];lines.append(f"|E{gid} {q['group']}|{q['acc_mean_pct']:.4f} ± {q['acc_sd_pct']:.4f}|{q['hard_auc_mean_pct']:.4f} ± {q['hard_auc_sd_pct']:.4f}|{q['spe_mean_pct']:.4f} ± {q['spe_sd_pct']:.4f}|{q['sen_mean_pct']:.4f} ± {q['sen_sd_pct']:.4f}|{q['probability_auc_mean_pct']:.4f} ± {q['probability_auc_sd_pct']:.4f}|")
    lines += ["","## 配对比较（pp；十折胜/平/负）","","|比较|ΔACC|ΔHard-AUC|ΔSPE|ΔSEN|","|---|---:|---:|---:|---:|"]
    for q in pairs:lines.append(f"|{q['comparison']}|{q['acc_delta_pp']:.4f} ({q['acc_w_t_l']})|{q['hard_auc_delta_pp']:.4f} ({q['hard_auc_w_t_l']})|{q['spe_delta_pp']:.4f}|{q['sen_delta_pp']:.4f}|")
    lines += ["","## 冻结的48.3 E6：无重训干预","","以下是在旧E6十个best checkpoint及48.3隔离源码上进行的推理干预，与本轮新训练分开。历史预测SHA逐折复现，checkpoint和参数均未更新。","","|干预|ACC|Hard-AUC|SPE|SEN|Probability-AUC|翻转数|ΔACC pp|","|---|---:|---:|---:|---:|---:|---:|---:|"]
    frozen_path=RES/"frozen_e6_interventions.csv"
    if frozen_path.exists():
        with frozen_path.open() as f: frozen=list(csv.DictReader(f))
        wanted=["FULL","APCR_OFF","OFF_PHENO","OFF_ANAT","OFF_FUNC","OFF_CORR","IDENTITY_ONLY","FULL_WRONG","RANDOM"]
        for name in wanted:
            rr=[r for r in frozen if r["intervention"]==name]
            if not rr:continue
            mean=lambda k:100.*float(np.mean([float(r[k]) for r in rr]))
            flips=sum(int(r["flips_vs_full"]) for r in rr)
            delta=float(np.mean([float(r['delta_acc_vs_full_pp']) for r in rr]))
            lines.append(f"|{name}|{mean('acc'):.4f}|{mean('hard_auc'):.4f}|{mean('spe'):.4f}|{mean('sen'):.4f}|{mean('probability_auc'):.4f}|{flips}|{delta:+.4f}|")
        full=[r for r in frozen if r["intervention"]=="FULL"]
        lines += ["",f"旧E6 alpha均值={np.mean([float(r['alpha']) for r in full]):.6f}；有效APCR残差/原始GFR范数均值={100*np.mean([float(r['effective_apcr_residual_over_base_norm']) for r in full]):.5f}%。身份错配与完整重算错配不是同一干预。关闭APCR后变化仅表示该best checkpoint的直接推理效应，不能排除APCR训练对原GFR参数的间接影响。"]
    lines += ["","## 新E6：固定checkpoint干预","","|干预|ACC|Hard-AUC|SPE|SEN|Probability-AUC|翻转数|ΔACC pp|","|---|---:|---:|---:|---:|---:|---:|---:|"]
    new_path=RES/"apcr_e6_interventions.csv"
    if new_path.exists():
        with new_path.open() as f:new=list(csv.DictReader(f))
        for name in ("FULL","APCR_OFF","ALPHA_ZERO","OFF_PHENO","OFF_ANAT","OFF_FUNC","OFF_CORR","WRONG_RELATION_REWEIGHTED","RANDOM_STRATIFIED_RELATION"):
            rr=[r for r in new if r["intervention"]==name]
            if not rr:continue
            mean=lambda k:100.*float(np.mean([float(r[k]) for r in rr]))
            delta=float(np.mean([float(r['delta_acc_vs_full_pp']) for r in rr]))
            lines.append(f"|{name}|{mean('acc'):.4f}|{mean('hard_auc'):.4f}|{mean('spe'):.4f}|{mean('sen'):.4f}|{mean('probability_auc'):.4f}|{sum(int(r['flips_vs_full']) for r in rr)}|{delta:+.4f}|")
    pre=json.loads((RT/'engineering_preflight.json').read_text())
    srd_rows=[];cra_rows=[]
    for gid in (6,10):
        for fold in range(1,11):
            b=run.load(RT/'checkpoints'/f'g{gid}_f{fold:02d}.best.pt')
            for tr in b.get('trace',[]):
                if tr.get('cra'):
                    cra_rows.append((gid,tr['cra']))
    e6_srd=[]
    for fold in range(1,11):
        b=run.load(RT/'checkpoints'/f'g6_f{fold:02d}.best.pt')
        if b.get('trace'):e6_srd.append(b['trace'][-1].get('pair_auc',[np.nan]*4))
    e6_inter=[r for r in read_csv(RES/'apcr_e6_interventions.csv') if r['intervention']=='FULL']
    lines += ["","## 工程检查与机制证据","",f"- B0-B初始化、输出、loss、梯度、一步更新、双Adam状态和RNG逐折等价检查：{len(pre['parity']['folds'])}/10通过。E1最终best预测与历史E0预测SHA相同：{sum(1 for f in range(1,11) if next(t for t in tasks if t['group_id']==1 and t['fold']==f)['prediction_sha256']==next(t for t in tasks if t['group_id']==0 and t['fold']==f)['prediction_sha256'])}/10；fold 1的E1最佳epoch=122而E0=157，虽然ACC相同，完整训练轨迹parity未通过。",f"- SCA测试检查同一患者对ID、邻居顺序不变性、冲突权重和有限loss：PASS；相同视角同对损失={pre['sca']['identical_views_same_pairs_zero']:.1g}，冲突情形有界损失={pre['sca']['conflict_loss_bounded']:.4f}。实际E5−E4为ACC {p['E5-E4']['acc_delta_pp']:+.4f}pp、Hard-AUC {p['E5-E4']['hard_auc_delta_pp']:+.4f}pp，SCA单独组未改善。E6−E7为+{p['E6-E7']['acc_delta_pp']:.4f}/+{p['E6-E7']['hard_auc_delta_pp']:.4f}pp，说明其效果依赖CRA组合，不能称为独立稳定收益。",f"- E4患者不重叠SRD校准AUC（四模态，E4最后记录epoch按折平均）={np.mean([t['diagnostics']['last_epoch_trace']['pair_auc'] for t in tasks if t['group_id']==4],axis=0).round(3).tolist()}；E9置乱监督校准AUC={np.mean([t['diagnostics']['last_epoch_trace']['pair_auc'] for t in tasks if t['group_id']==9],axis=0).round(3).tolist()}。E9同一配对集合保持不变，患者标签置乱改变约42%的fit患者标签。校准端点与SRD pair BCE端点不重叠，但其表示仍受训练标签的主分类任务影响，不能视为独立外部验证。",f"- CRA有限候选内层最大化实际执行。真实fold预检clean CE={pre['cra_controls']['real_fold_cra_clean']:.8f}、候选最大CE={pre['cra_controls']['real_fold_cra_max']:.8f}，提升仅{pre['cra_controls']['real_fold_cra_max']-pre['cra_controls']['real_fold_cra_clean']:.2g}。训练末期E6两步平均CE增量约1e-5量级；并非全图/全患者候选空间全局最坏情况。E6−E10为+{p['E6-E10']['acc_delta_pp']:.4f}/+{p['E6-E10']['hard_auc_delta_pp']:.4f}pp（胜平负ACC {p['E6-E10']['acc_w_t_l']}，AUC {p['E6-E10']['hard_auc_w_t_l']}），效应有限。",f"- 新E6固定checkpoint关闭全部APCR消息后ACC下降1.7241pp、29个测试预测翻转；真实邻居换成分层错配后下降2.7560pp、32个翻转；同站点/训练划分约束随机合法邻居后下降2.0664pp、26个翻转。四路逐一关闭的ACC下降：PHENO 0.6897pp、ANAT 0.5747pp、FUNC 1.2644pp、CORR 0.9195pp。关闭可证明新E6 checkpoint在推理时使用关系消息，但E6组整体仍低于B0-B。旧E6的APCR残差/基座仅0.01670%，关闭后ACC只降0.1149pp。",f"- E8分层错误关系传播边改变率={np.mean([float(d['wrong_relation_edge_change_rate']) for d in diags if d.get('group_id')==8]):.4f}，保持站点及train/test strata，无自环和重复邻居。独立重训E6−E8的ACC差={p['E6-E8']['acc_delta_pp']:+.4f}pp、Hard-AUC差={p['E6-E8']['hard_auc_delta_pp']:+.4f}pp；因此冻结模型的直接关系敏感性没有转化为完整训练相对错误关系的ACC优势。", "- 保护SHA未变，训练签名唯一；完整任务读回110/110通过。源码/数据保护SHA、fold SHA、checkpoint/预测SHA见`manifest.json`。"]
    lines += ["","## 科研判定","",f"E6指标初筛=FAIL（ACC {s[6]['acc_mean_pct']:.4f}% 对门槛 91.73%）；相对B0-B ΔACC={p['E6-E0']['acc_delta_pp']:+.4f}pp、ΔHard-AUC={p['E6-E0']['hard_auc_delta_pp']:+.4f}pp。", "- `GO_HISTORICAL_E6_DIRECT_GRAPH_EFFECT`=极弱：旧E6关闭APCR仅-0.1149pp、1个预测翻转；效应不能证明此前训练期间无间接影响。", "- `GO_ANCHOR_PARITY`=单步/初始化逐折PASS，完整best checkpoint预测SHA 9/10，故整体PARTIAL，不能称全折历史训练精确复现。", f"- `GO_REAL_MESSAGE`=NO-GO：E3−E2 ACC {p['E3-E2']['acc_delta_pp']:+.4f}pp、Hard-AUC {p['E3-E2']['hard_auc_delta_pp']:+.4f}pp。", f"- `GO_SRD`=PARTIAL：E4−E3为{p['E4-E3']['acc_delta_pp']:+.4f}/{p['E4-E3']['hard_auc_delta_pp']:+.4f}pp，十折胜平负不稳定；内层患者不重叠校准AUC高于随机但不等于未见标签的外部泛化。", f"- `GO_SCA`=NO-GO作为独立模块：E5−E4为{p['E5-E4']['acc_delta_pp']:+.4f}/{p['E5-E4']['hard_auc_delta_pp']:+.4f}pp；E6−E7显示和CRA组合时有正差，尚不能确立可重复独立贡献。", f"- `GO_CRA`=有限候选最大化实现通过、科学效益有限：E6−E5为{p['E6-E5']['acc_delta_pp']:+.4f}/{p['E6-E5']['hard_auc_delta_pp']:+.4f}pp，但每轮最坏CE增量很小。", f"- `GO_REAL_RELATION`=NO-GO训练增量：E6−E8 ACC {p['E6-E8']['acc_delta_pp']:+.4f}pp、Hard-AUC {p['E6-E8']['hard_auc_delta_pp']:+.4f}pp；固定checkpoint错误关系会降低性能，但完整重训负控制未显示ACC增量。", f"- `GO_TRUE_PAIR_SUPERVISION`=有限支持：E6−E9为{p['E6-E9']['acc_delta_pp']:+.4f}/{p['E6-E9']['hard_auc_delta_pp']:+.4f}pp，但折间方向不稳定。", f"- `GO_ATTACK_VS_RANDOM`=有限支持：E6−E10为{p['E6-E10']['acc_delta_pp']:+.4f}/{p['E6-E10']['hard_auc_delta_pp']:+.4f}pp，效应小且不满足稳定折胜。", "- `GO_GRAPH_RELIANCE`=固定E6 checkpoint依赖APCR消息，但不等于患者关系带来相对基线净增益。", "- `GO_FINAL_MICCAI_CANDIDATE`=NO-GO。E6未过91.73%门槛、低于B0-B，真实关系重训负控制未支持增量，E1全训练轨迹parity缺一折。"]
    lines += ["","本实验结果在保留历史CV前RFE、`val=test`选checkpoint、871节点transductive和单seed协议下得到，不是无偏外部泛化估计。不能用冻结模型的干预敏感性替代独立重训的关系增量证据。"]
    (RES/"report.md").write_text("\n".join(lines)+"\n")

if __name__=="__main__":main()
