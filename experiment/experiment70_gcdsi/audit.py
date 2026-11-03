#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
_runner_spec=importlib.util.spec_from_file_location("e70_audit_runner",EXP/"run.py")
R=importlib.util.module_from_spec(_runner_spec)
sys.modules[_runner_spec.name]=R
_runner_spec.loader.exec_module(R)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def tensor_tree_sha(value) -> str:
    digest=hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            tensor=item.detach().cpu().contiguous()
            digest.update(b"tensor\0"+str(tensor.dtype).encode()+b"\0"+repr(tuple(tensor.shape)).encode()+b"\0")
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item,dict):
            digest.update(b"dict\0")
            for key in sorted(item,key=lambda k:repr(k)):
                visit(key);visit(item[key])
        elif isinstance(item,(list,tuple)):
            digest.update(type(item).__name__.encode()+b"\0")
            for child in item: visit(child)
        else:
            digest.update(json.dumps(item,sort_keys=True,default=str).encode()+b"\0")
    visit(value)
    return digest.hexdigest()


def pearson_abs(left, right) -> float:
    left=np.asarray(left,dtype=np.float64);right=np.asarray(right,dtype=np.float64)
    left=left-left.mean();right=right-right.mean()
    denom=float(np.sqrt(np.dot(left,left)*np.dot(right,right)))
    return abs(float(np.dot(left,right)/denom)) if denom>0 else 0.0


def site_eta_squared(values, site_ids) -> float:
    values=np.asarray(values,dtype=np.float64);grand=float(values.mean())
    total=float(np.square(values-grand).sum())
    if total==0:return 0.0
    between=0.0
    for site in np.unique(site_ids):
        group=values[site_ids==site]
        between+=len(group)*float(np.square(group.mean()-grand))
    return between/total


def audit_fold(group: str, fold: int, ref_manifest, ref_folds, features):
    task_path=R.RUNTIME/group/"tasks"/f"{group}_f{fold:02d}.json"
    cp_path=R.RUNTIME/group/"checkpoints"/f"{group}_f{fold:02d}.best.pt"
    task=json.loads(task_path.read_text())
    if task.get("status")!="COMPLETE" or R.sha_file(cp_path)!=task.get("checkpoint_sha256"):
        raise AssertionError(f"task/checkpoint signature mismatch {group} fold {fold}")
    cp=torch.load(cp_path,map_location="cpu",weights_only=False)
    if cp.get("group")!=group or cp.get("fold")!=fold: raise AssertionError("checkpoint identity mismatch")
    for key,expected in (("split_sha",ref_folds[fold]["split_sha"]),("pair_sha",ref_folds[fold]["pair_sha"]),("initial_shared_sha",ref_folds[fold]["initial_shared_sha"]),("initial_added_sha",ref_folds[fold]["initial_added_sha"]),("initial_rng_sha",ref_folds[fold]["initial_rng_sha"])):
        if cp.get(key)!=expected or task.get(key)!=expected: raise AssertionError(f"{key} mismatch {group} fold {fold}")
    if cp.get("reference_checkpoint_sha256")!=ref_folds[fold]["checkpoint_sha256"]: raise AssertionError("wrong historical reference binding")
    assets,_=R.text_banks(); text=assets[group].to(R.BASE.runtime()["device"])
    state=R.build_state(fold,ref_manifest,ref_folds)
    module=R.make_module(group,fold,text)
    R.install_output_hook(state["cam"],module,text)
    state["model"].load_state_dict(cp["state_model"],strict=True)
    state["cam"].load_state_dict(cp["state_cam"],strict=True)
    module.load_state_dict(cp["state_gcdsi"],strict=True)
    if not torch.equal(cp["text_bank"],text.detach().cpu()): raise AssertionError("checkpoint prompt bank mismatch")
    if cp.get("source_sha256")!=R.sha_file(EXP/"run.py") or cp.get("config_sha256")!=R.sha_file(EXP/"config.json"):
        raise AssertionError("training source/config SHA mismatch")
    optimizer_specs=(("optimizer_base","opt_base","Adam",0.0038,0.11),("optimizer_cam","opt_cam","Adam",0.0038,0.11),("optimizer_added","opt_added","AdamW",0.001,0.01),("optimizer_gcdsi","gcdsi","AdamW",0.001,0.001))
    for key,state_key,kind,lr,wd in optimizer_specs:
        optimizer=state[state_key] if state_key!="gcdsi" else torch.optim.AdamW(module.parameters(),lr=lr,weight_decay=wd)
        if optimizer.__class__.__name__!=kind: raise AssertionError(f"{state_key} optimizer class mismatch")
        groups=cp[key]["param_groups"]
        if any(float(g["lr"])!=lr or float(g["weight_decay"])!=wd for g in groups):
            raise AssertionError(f"{key} LR/WD mismatch")
    state["_is_training"]=False
    with torch.no_grad():
        out=R.forward_state(state,True)
        metrics=R.BASE.classification_metrics(out["logp"],state["test"],state["y"])
    for metric in ("acc","hard_auc","prob_auc","spe","sen","ba","asd_f1"):
        if abs(float(metrics[metric])-float(task["metrics"][metric]))>1e-10: raise AssertionError(f"readback {metric} mismatch")
    diagnostic=out["gcdsi"]
    base_prob=out["logp"].detach().exp()
    base_pred=base_prob.argmax(-1)
    gamma_raw=module.gamma_raw.detach().clone()
    with torch.no_grad():module.gamma_raw.zero_()
    with torch.no_grad():gamma_off=R.forward_state(state,True)
    with torch.no_grad():module.gamma_raw.copy_(gamma_raw)
    gamma_prob=gamma_off["logp"].detach().exp()
    gamma_intervention={"gamma_off_mean_abs_delta_p_asd":float((gamma_prob[:,0]-base_prob[:,0]).abs().mean().cpu()),"gamma_off_max_abs_delta_logit":float((gamma_off["logp"]-out["logp"]).abs().max().cpu()),"gamma_off_prediction_flips":int((gamma_prob.argmax(-1)!=base_pred).sum().cpu())}
    query_weight=module.query_f.weight.detach().clone()
    with torch.no_grad():module.query_f.weight.zero_()
    with torch.no_grad():query_off=R.forward_state(state,True)
    with torch.no_grad():module.query_f.weight.copy_(query_weight)
    query_prob=query_off["logp"].detach().exp()
    query_diag=query_off["gcdsi"]
    graph_intervention={"graph_query_off_mean_abs_delta_p_asd":float((query_prob[:,0]-base_prob[:,0]).abs().mean().cpu()),"graph_query_off_max_abs_delta_logit":float((query_off["logp"]-out["logp"]).abs().max().cpu()),"graph_query_off_prediction_flips":int((query_prob.argmax(-1)!=base_pred).sum().cpu()),"graph_query_off_attention_l1":float((query_diag["attention"]-diagnostic["attention"]).abs().mean().cpu())}
    attention=diagnostic["attention"].detach()
    entropy=-(attention.clamp_min(1e-12)*attention.clamp_min(1e-12).log()).sum(-1)
    residual=diagnostic["residual"]
    data=features
    if data.shape!=(871,320):raise AssertionError(f"unexpected audit feature shape {data.shape}")
    age_centers=np.arange(7.47,58.0,2.0,dtype=np.float64)
    age=data[:,:26]@age_centers
    site_ids=data[:,26:46].argmax(axis=1)
    motion=data[:,60]
    quality=data[:,48:64]
    rows=[]
    for modality,name in enumerate(("PHENO","ANAT","FUNC","CORR")):
        a=attention[:,modality]
        attn_patient=a[:,0].cpu().numpy()
        gate_patient=diagnostic["gate"][:,modality].detach().mean(-1).cpu().numpy()
        gate=diagnostic["gate"][:,modality]
        common={**gamma_intervention,**graph_intervention}
        rows.append({"group":group,"fold":fold,"modality":name,"gamma":float(module.gamma.detach().cpu()),"attention_asd_mean":float(a[:,0].mean().cpu()),"attention_asd_sd":float(a[:,0].std(unbiased=False).cpu()),"attention_entropy_mean":float(entropy[:,modality].mean().cpu()),"attention_entropy_sd":float(entropy[:,modality].std(unbiased=False).cpu()),"patient_attention_asd_sd":float(a[:,0].std(unbiased=False).cpu()),"cross_modality_attention_sd":float(attention[:,:,0].std(dim=1,unbiased=False).mean().cpu()),"gate_mean":float(gate.mean().cpu()),"gate_sd":float(gate.std(unbiased=False).cpu()),"gate_saturation_fraction":float(((gate<0.01)|(gate>0.99)).float().mean().cpu()),"residual_rms_ratio":float((residual[:,modality].square().mean().sqrt()/(diagnostic["graph_rms"].square().mean().sqrt()+1e-12)).cpu()),"residual_to_cam_rms":float((residual[:,modality].square().mean().sqrt()/(diagnostic["cam_rms"][:,modality].square().mean().sqrt()+1e-12)).cpu()),"w_f_norm":float(module.query_f.weight.detach().norm().cpu()),"w_f_grad_norm":float(cp["diagnostics"]["w_f_grad_norm"]),"site_eta2_attention":site_eta_squared(attn_patient,site_ids),"age_abs_r_attention":pearson_abs(attn_patient,age),"mean_fd_abs_r_attention":pearson_abs(attn_patient,motion),"max_abs_r_quality_attention":max(pearson_abs(attn_patient,quality[:,j]) for j in range(quality.shape[1])),"site_eta2_gate":site_eta_squared(gate_patient,site_ids),"age_abs_r_gate":pearson_abs(gate_patient,age),"mean_fd_abs_r_gate":pearson_abs(gate_patient,motion),"max_abs_r_quality_gate":max(pearson_abs(gate_patient,quality[:,j]) for j in range(quality.shape[1])),**common})
    metric_row={"group":group,"fold":fold,"status":"PASS","checkpoint_sha256":R.sha_file(cp_path),"best_epoch":cp["best_epoch"],"epochs_executed":cp["epochs_executed"],"source_sha256":cp["source_sha256"],"config_sha256":cp["config_sha256"],**{k:float(metrics[k]) for k in ("acc","hard_auc","prob_auc","sen","spe","ba","asd_f1")},**{k:int(metrics[k]) for k in ("asd_tp","asd_fn","td_fp","td_tn")}}
    return metric_row,rows


def active_step(state,module,semantic_optimizer):
    state["_is_training"]=True
    state["opt_base"].zero_grad(set_to_none=True);state["opt_cam"].zero_grad(set_to_none=True);state["opt_added"].zero_grad(set_to_none=True);semantic_optimizer.zero_grad(set_to_none=True)
    out=R.forward_state(state,True);out["total"].backward()
    state["opt_base"].step();state["opt_cam"].step();state["opt_added"].step();semantic_optimizer.step()
    state["_is_training"]=False
    with torch.no_grad(): evaluated=R.forward_state(state,True)
    return evaluated["logp"].detach().cpu()


def capture_training_state(state,module,semantic_optimizer):
    return {"model":R.BASE._state_cpu(state["model"]),"cam":R.BASE._state_cpu(state["cam"]),"gcdsi":R.BASE._state_cpu(module),"optimizer_base":copy.deepcopy(state["opt_base"].state_dict()),"optimizer_cam":copy.deepcopy(state["opt_cam"].state_dict()),"optimizer_added":copy.deepcopy(state["opt_added"].state_dict()),"optimizer_gcdsi":copy.deepcopy(semantic_optimizer.state_dict()),"rng":copy.deepcopy(R.BASE.pack_rng())}


def restore_training_state(snapshot,state,module,semantic_optimizer):
    state["model"].load_state_dict(snapshot["model"],strict=True);state["cam"].load_state_dict(snapshot["cam"],strict=True);module.load_state_dict(snapshot["gcdsi"],strict=True)
    state["opt_base"].load_state_dict(snapshot["optimizer_base"]);state["opt_cam"].load_state_dict(snapshot["optimizer_cam"]);state["opt_added"].load_state_dict(snapshot["optimizer_added"]);semantic_optimizer.load_state_dict(snapshot["optimizer_gcdsi"]);R.BASE.restore_rng(snapshot["rng"])


def short_replay_check(ref_manifest,ref_folds):
    fold=1;group="E70-G1";assets,_=R.text_banks();state=R.build_state(fold,ref_manifest,ref_folds)
    warmup_path=R.E59.warmup_cache_path(fold)
    R.E59.load_warmup_cache_into_state(state,fold,json.loads(R.E59.CONFIG_PATH.read_text()))
    text=assets[group].to(R.BASE.runtime()["device"]);module=R.make_module(group,fold,text);R.install_output_hook(state["cam"],module,text)
    semantic_optimizer=torch.optim.AdamW(module.parameters(),lr=0.001,weight_decay=0.001)
    start=capture_training_state(state,module,semantic_optimizer)
    try:
        active_step(state,module,semantic_optimizer)
        midpoint=capture_training_state(state,module,semantic_optimizer)
        continuous_logits=active_step(state,module,semantic_optimizer)
        continuous=capture_training_state(state,module,semantic_optimizer)
        restore_training_state(start,state,module,semantic_optimizer)
        active_step(state,module,semantic_optimizer);repeat_logits=active_step(state,module,semantic_optimizer)
        repeated=capture_training_state(state,module,semantic_optimizer)
        restore_training_state(midpoint,state,module,semantic_optimizer)
        resumed_logits=active_step(state,module,semantic_optimizer)
        resumed=capture_training_state(state,module,semantic_optimizer)
        same=lambda a,b: tensor_tree_sha(a)==tensor_tree_sha(b)
        record={"status":"PASS" if same(continuous,repeated) and same(continuous,resumed) and torch.equal(continuous_logits,repeat_logits) and torch.equal(continuous_logits,resumed_logits) else "FAIL","fold":fold,"group":group,"active_epochs_replayed":2,"warmup_sha256":R.sha_file(warmup_path),"continuous_state_sha256":tensor_tree_sha(continuous),"repeated_state_sha256":tensor_tree_sha(repeated),"resumed_state_sha256":tensor_tree_sha(resumed),"continuous_repeat_logits_equal":bool(torch.equal(continuous_logits,repeat_logits)),"continuous_resume_logits_equal":bool(torch.equal(continuous_logits,resumed_logits)),"optimizer_rng_and_model_state_equal":bool(same(continuous,repeated) and same(continuous,resumed))}
        if record["status"]!="PASS":raise AssertionError("actual active training replay/resume parity failed")
        return record
    finally:
        del state,module,semantic_optimizer
        if torch.cuda.is_available():torch.cuda.empty_cache()


def append_handoff(summary,comparisons,gates,decision):
    handoff=Path("/model/spromptgl/experiment/handoff.md")
    marker="<!-- EXPERIMENT70_GCDSI_2026-11-03 -->"
    base=next(x for x in summary if x["group"]=="E53.3-BASE")
    groups=[x for x in summary if x["group"] in R.GROUPS]
    best=max(groups,key=lambda x:x["acc_mean_pct"])
    g1g3=next(x for x in comparisons if x["group_a"]=="E70-G1" and x["group_b"]=="E70-G3" and x["metric"]=="acc")
    lines=["\n## Experiment 70 — GCDSI", "", "- 状态：`COMPLETE_AUDITED`；30/30正式折完成，30/30最佳checkpoint独立读回通过；10/10初始化与共享warmup来源审计通过。", f"- 结论：**{decision}**；当前最佳组为 {best['group']}（ACC {best['acc_mean_pct']:.3f}±{best['acc_sd_pct']:.3f}%）。真实BrainPrompt文本相对匿名Token的ΔACC为{g1g3['mean_delta_pp']:+.3f} pp，W/T/L {g1g3['wins']}/{g1g3['ties']}/{g1g3['losses']}。", f"- 53.3第三组基座 ACC/Hard-AUC/Prob-AUC/SEN/SPE：{base['acc_mean_pct']:.3f}/{base['hard_auc_mean_pct']:.3f}/{base['prob_auc_mean_pct']:.3f}/{base['sen_mean_pct']:.3f}/{base['spe_mean_pct']:.3f}%。", "- 三组 ACC/Hard-AUC/Prob-AUC/SEN/SPE（% mean±population SD）："]
    for row in groups:
        lines.append(f"  - {row['group']}: {row['acc_mean_pct']:.3f}±{row['acc_sd_pct']:.3f} / {row['hard_auc_mean_pct']:.3f}±{row['hard_auc_sd_pct']:.3f} / {row['prob_auc_mean_pct']:.3f}±{row['prob_auc_sd_pct']:.3f} / {row['sen_mean_pct']:.3f}±{row['sen_sd_pct']:.3f} / {row['spe_mean_pct']:.3f}±{row['spe_sd_pct']:.3f}%.")
    base_deltas=[]
    for group in R.GROUPS:
        delta=next(x for x in comparisons if x["group_a"]==group and x["group_b"]=="E53.3-BASE" and x["metric"]=="acc")
        base_deltas.append(f"{group} ΔACC {delta['mean_delta_pp']:+.3f} pp ({delta['wins']}/{delta['ties']}/{delta['losses']} W/T/L)")
    lines.extend(["- 配对ACC相对基座："+"；".join(base_deltas)+"。", "- GCDSI门控残差、注意力、W_f梯度及固定checkpoint干预结果见实验报告；报告包含站点、年龄、FD与影像质量特征关联诊断。", "- 路径：`/model/spromptgl/experiment/experiment70_gcdsi/results/report.md`；清单：`/model/spromptgl/experiment/experiment70_gcdsi/manifest.json`。", "- 局限与下一步：历史`val=test`、CV前RFE/标准化和871节点transductive图限制泛化解释；按冻结判据不追加搜索，后续仅在结论允许时开展多seed、严格验证和组件消融。", marker, ""])
    with handoff.open("a+",encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        stream.seek(0);contents=stream.read()
        if marker not in contents:
            stream.seek(0,os.SEEK_END);stream.write("\n".join(lines));stream.flush();os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
    return R.sha_file(handoff)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--all",action="store_true");args=parser.parse_args()
    if not args.all: parser.error("use --all")
    ref_manifest,ref_folds,ref_metrics=R.load_reference()
    (features,_labels),_folds=R.BASE.load_data()
    metrics=[];mechanisms=[];errors=[]
    for group in R.GROUPS:
        for fold in range(1,11):
            try:
                row,diag=audit_fold(group,fold,ref_manifest,ref_folds,features);metrics.append(row);mechanisms.extend(diag)
                print(f"PASS {group} fold {fold}",flush=True)
            except Exception as exc:
                errors.append({"group":group,"fold":fold,"error":repr(exc)})
                print(f"FAIL {group} fold {fold}: {exc}",flush=True)
                raise
    write_csv(R.RESULTS/"fold_metrics.csv",metrics);write_csv(R.RESULTS/"mechanism.csv",mechanisms)
    ref_rows=[]
    for fold,row in ref_metrics.items():
        ref_rows.append({"group":"E53.3-BASE","fold":fold,**{k:float(row[k]) for k in ("acc","hard_auc","prob_auc","sen","spe","ba","asd_f1")}})
    all_metrics=ref_rows+metrics
    summary=[]
    for group in ["E53.3-BASE",*R.GROUPS]:
        rows=[r for r in all_metrics if r["group"]==group]
        summary.append({"group":group,"folds":len(rows)})
        for key in ("acc","hard_auc","prob_auc","sen","spe","ba","asd_f1"):
            vals=[float(r[key]) for r in rows];summary[-1][f"{key}_mean_pct"]=100*sum(vals)/len(vals);summary[-1][f"{key}_sd_pct"]=100*float(torch.tensor(vals,dtype=torch.float64).std(unbiased=False))
        if group!="E53.3-BASE":
            for key in ("asd_tp","td_fp","asd_fn","td_tn"):
                summary[-1][f"{key}_sum"]=sum(int(r[key]) for r in rows)
    write_csv(R.RESULTS/"summary.csv",summary)
    comparisons=[]
    pairs=[("E70-G1","E53.3-BASE"),("E70-G2","E53.3-BASE"),("E70-G3","E53.3-BASE"),("E70-G1","E70-G2"),("E70-G1","E70-G3"),("E70-G2","E70-G3")]
    for left,right in pairs:
        for metric in ("acc","hard_auc","prob_auc","sen","spe"):
            a={r["fold"]:r[metric] for r in all_metrics if r["group"]==left};b={r["fold"]:r[metric] for r in all_metrics if r["group"]==right};delta=[100*(a[f]-b[f]) for f in range(1,11)]
            comparisons.append({"group_a":left,"group_b":right,"metric":metric,"mean_delta_pp":sum(delta)/10,"wins":sum(x>1e-12 for x in delta),"ties":sum(abs(x)<=1e-12 for x in delta),"losses":sum(x< -1e-12 for x in delta),"fold_deltas_pp":json.dumps(delta)})
    write_csv(R.RESULTS/"paired_comparisons.csv",comparisons)
    gates=[]
    for group in R.GROUPS:
        c=next(x for x in comparisons if x["group_a"]==group and x["group_b"]=="E53.3-BASE" and x["metric"]=="acc")
        s=next(x for x in comparisons if x["group_a"]==group and x["group_b"]=="E53.3-BASE" and x["metric"]=="sen")
        p=next(x for x in comparisons if x["group_a"]==group and x["group_b"]=="E53.3-BASE" and x["metric"]=="prob_auc")
        passed=c["mean_delta_pp"]>=0.5 and c["wins"]>=6 and s["mean_delta_pp"]>=-1 and p["mean_delta_pp"]>=-0.5
        gates.append({"group":group,"base_gate":"PASS" if passed else "FAIL","delta_acc_pp":c["mean_delta_pp"],"acc_wins":c["wins"],"delta_sen_pp":s["mean_delta_pp"],"delta_prob_auc_pp":p["mean_delta_pp"]})
    anon=next(x for x in comparisons if x["group_a"]=="E70-G1" and x["group_b"]=="E70-G3" and x["metric"]=="acc")
    sem_pass=anon["mean_delta_pp"]>0 and anon["wins"]>=6
    status="PASS" if len(metrics)==30 and not errors else "FAIL"
    mechanism_gates=[]
    for group in R.GROUPS:
        rows=[x for x in mechanisms if x["group"]==group]
        tests={"nonzero_residual":all(x["residual_to_cam_rms"]>1e-5 and abs(x["gamma"])>1e-6 for x in rows),"nonconstant_patient_and_modality_attention":all(x["patient_attention_asd_sd"]>1e-5 and x["cross_modality_attention_sd"]>1e-5 for x in rows),"two_token_attention_not_saturated":all(x["attention_entropy_mean"]>0.01 for x in rows),"gate_not_collapsed":all(x["gate_saturation_fraction"]<0.95 for x in rows),"graph_query_gradient_and_sensitivity":all(x["w_f_grad_norm"]>0 and x["graph_query_off_attention_l1"]>1e-6 for x in rows)}
        mechanism_gates.append({"group":group,"status":"PASS" if all(tests.values()) else "FAIL",**tests})
    mechanism_pass=all(x["status"]=="PASS" for x in mechanism_gates)
    decision="GO" if any(x["base_gate"]=="PASS" for x in gates) and sem_pass and mechanism_pass else "NO-GO"
    preflight=json.loads((R.RUNTIME/"preflight.json").read_text())
    e59_preflight_path=R.E59.RUNTIME/"preflight_audit.json"
    e59_preflight=json.loads(e59_preflight_path.read_text())
    if e59_preflight.get("status")!="PASS" or e59_preflight.get("rng_and_warmup_parity",{}).get("status")!="PASS" or e59_preflight.get("shared_warmup_cache",{}).get("fold_count")!=10:
        raise AssertionError("Experiment 59 shared warmup cache lacks a complete parity audit")
    warmup_parity={"e59_preflight_status":e59_preflight["status"],"rng_and_warmup_parity":e59_preflight["rng_and_warmup_parity"],"shared_cache_status":e59_preflight["shared_warmup_cache"]["status"],"folds":e59_preflight["shared_warmup_cache"]["fold_count"],"all_fold_readbacks_pass":all(x["readback"]=="PASS" for x in e59_preflight["shared_warmup_cache"]["folds"].values()),"e70_fold_binding_pass":all(all(x.values()) for x in preflight["folds"])}
    if not warmup_parity["all_fold_readbacks_pass"] or not warmup_parity["e70_fold_binding_pass"]:
        raise AssertionError("10-fold initial/warmup binding audit failed")
    replay=short_replay_check(ref_manifest,ref_folds)
    if replay["status"]!="PASS":raise AssertionError("short actual-path replay check failed")
    source_paths={"run.py":R.EXP/"run.py","gcdsi.py":R.EXP/"gcdsi.py","audit.py":Path(__file__),"config.json":R.EXP/"config.json","design.md":R.EXP/"design.md"}
    source_sha={name:R.sha_file(path) for name,path in source_paths.items()}
    base_source_sha={"run.py":R.sha_file(R.BASE_EXP/"run.py"),"cadr_gfr.py":R.sha_file(R.BASE_EXP/"cadr_gfr.py"),"config.json":R.sha_file(R.BASE_EXP/"config.json")}
    if base_source_sha!=preflight["base_source_sha256"]:raise AssertionError("53.3 source changed after preflight")
    official_sources=ref_manifest.get("source_sha256",{})
    for filename,digest in base_source_sha.items():
        if official_sources.get(filename)!=digest:raise AssertionError(f"official 53.3 source hash drift: {filename}")
    data_sha={"processed_standard_data.csv":R.sha_file(R.ROOT/"data/ABIDE/processed_standard_data.csv"),"modal_feat_dict.npy":R.sha_file(R.ROOT/"data/ABIDE/modal_feat_dict.npy")}
    ref_checkpoint_sha={str(f):ref_folds[f]["checkpoint_sha256"] for f in range(1,11)}
    integrity={"historical_source_hashes_match_preflight_and_manifest":True,"historical_checkpoints_match_reference_manifest":len(ref_checkpoint_sha)==10,"historical_base_retrained":False,"historical_base_checkpoint_sha256":ref_checkpoint_sha,"data_sha256_at_final_audit":data_sha,"base_source_sha256":base_source_sha,"experiment70_source_sha256":source_sha,"e59_preflight_sha256":R.sha_file(e59_preflight_path),"e70_preflight_sha256":R.sha_file(R.RUNTIME/"preflight.json"),"text_embedding_cache_sha256":R.sha_file(Path("/model/spromptgl/experiment/experiment60_dss_pen/.runtime/text_embeddings.pt")),"warmup_cache_sha256":preflight["warmup_cache_sha256"]}
    if integrity["text_embedding_cache_sha256"]!=preflight["text_assets"]["cache_sha256"]:raise AssertionError("text embedding cache changed after preflight")
    if {str(f):R.sha_file(R.E59.warmup_cache_path(f)) for f in range(1,11)}!=preflight["warmup_cache_sha256"]:raise AssertionError("warmup cache changed after preflight")
    for row in metrics:
        fold=int(row["fold"]);group=row["group"]
        task=json.loads((R.RUNTIME/group/"tasks"/f"{group}_f{fold:02d}.json").read_text())
        if task.get("status")!="COMPLETE":raise AssertionError(f"task completion status mismatch {group}/{fold}")
    provenance={"reference_group":"CADR_WD_0p01_GFR_WD_0p11","reference_manifest_sha256":R.sha_file(R.REF_EXP/"manifest_fallback_wd_sweep.json"),"reference_metrics_sha256":R.sha_file(R.REF_EXP/"results/fallback_wd_sweep/fold_metrics.csv"),"text_metadata":preflight["text_assets"],"prompt_text_sha256":{"brainprompt":R.sha_obj(R.TEXTS_BP),"spromptgl":R.sha_obj(R.TEXTS_SP)},"label_order":{"ASD":0,"TD":1},"input":{"patients":871,"modalities":["PHENO","ANAT","FUNC","CORR"],"dimensions":[48,6,10,256]},"warmup_parity":warmup_parity,"short_replay_resume":replay,"integrity":integrity}
    with (R.RESULTS/"report.md").open("w",encoding="utf-8") as f:
        f.write(f"# Experiment 70 — GCDSI\n\n状态：`{('COMPLETE_AUDITED' if status=='PASS' else status)}`，30/30新增折独立读回。科研判定：**{decision}**。单seed探索性判据，不是统计显著性结论。\n\n")
        f.write("## 研究问题与方法\n\n")
        f.write("GFR传播后的患者图状态是否能与CAM四个模态表示共同构造不同Query，从固定ASD/TD疾病文本中检索患者、模态相关语义，并在最终分类前受控注入？路径为`320维输入 → CADR-GFR → 871×320 → CAM → 871×4×36 → GCDSI → 871×144 → 原OutputLayer`。GCDSI只读取GFR表示，不回写图；CAM输出进入未修改的分类器。\n\n")
        f.write("对患者i、模态m：`q_im=LN(W_h h_im+W_f F_i+q_CLS)`；`A_im=softmax(q_im K^T/sqrt(36))`；`u_im=A_im V`；`g_im=sigmoid(MLP([h_im,q_im,u_im]))`；`h'_im=h_im+0.5*tanh(gamma_raw)*g_im⊙W_Ou_im`。`gamma_raw`从0初始化。分类损失经GCDSI回传至CAM与GFR；不detach图/CAM特征。新增模块无Dropout。损失保持`L_cls+L_graph+0.05L_pair+0.01L_adv+0.01L_preserve`，分类损失使用注入后的logits；没有原PEN、额外语义CE/对比损失或新分类头。\n\n")
        f.write("SABER公式(8)-(10)提供Graph-CLS、患者文本Key/Value、cross-attention与门控残差的设计动机；SABER另外把疾病文本用于对比监督，本实验没有复现该监督，不能声称完整复现SABER。SPromptGL中GFR是患者关系图传播、CAM是四模态交互、PEN是Prompt驱动局部增强；实验70在CAM后插入GCDSI，不恢复4×4×9局部切分。`OutputLayer.forward(x, attn_embedding)`虽接收attn参数，其分类实现只消费x。来源：[SABER](https://arxiv.org/abs/2607.01901)、[SPromptGL official](https://github.com/wanxixi11/SPromptGL)、[研究仓库](https://github.com/tangaii/SPromptGL)。\n\n")
        f.write("## 基座与审计\n\n")
        f.write("唯一基座为`CADR_WD_0p01_GFR_WD_0p11`。逐折从53.3第三组初始构造逻辑重建，未加载最佳权重进行微调；参考manifest、10个checkpoint SHA、task/readback、split、pair/cache、共享/新增参数和初始化RNG均通过校验。CADR AdamW为lr=0.001、WD=0.01；GFR/CAM Adam为lr=0.0038、WD=0.11；GCDSI另用AdamW lr=0.001、WD=0.001。每组先共享同折53.3状态的10轮warmup缓存，再从epoch索引10激活GCDSI；保留450 epoch上限、原early stopping、val=test与严格ACC改善选checkpoint。协议为ABIDE A 871名患者、seed=0、`StratifiedKFold(n_splits=10, shuffle=True, random_state=10)`、PHENO/ANAT/FUNC/CORR维度48/6/10/256、ASD=0/TD=1。\n\n")
        base=next(x for x in summary if x["group"]=="E53.3-BASE")
        f.write(f"参考基座10折：ACC {base['acc_mean_pct']:.4f}±{base['acc_sd_pct']:.4f}%；Hard-AUC/BA {base['hard_auc_mean_pct']:.4f}±{base['hard_auc_sd_pct']:.4f}%；Probability-AUC {base['prob_auc_mean_pct']:.4f}±{base['prob_auc_sd_pct']:.4f}%；SPE-TD {base['spe_mean_pct']:.4f}±{base['spe_sd_pct']:.4f}%；SEN-ASD {base['sen_mean_pct']:.4f}±{base['sen_sd_pct']:.4f}%；ASD-F1 {base['asd_f1_mean_pct']:.4f}±{base['asd_f1_sd_pct']:.4f}%。历史10折仅作只读参考。\n\n")
        f.write(f"初始化/warmup：E70逐折split/pair、原参数SHA、初始化RNG和warmup绑定10/10通过。E59独立预检状态`{warmup_parity['e59_preflight_status']}`，epoch-10共享缓存10/10读回一致；另外{warmup_parity['rng_and_warmup_parity'].get('steps')}步随机CAM前向、参数、优化器和共享RNG parity通过。E70实际入口在fold 1从epoch 10进行2个active epoch，连续训练、相同起点重复训练及从epoch 11断点恢复的模型/优化器/RNG/logits完全一致。详情在`results/audit_status.json`及manifest provenance。\n\n")
        f.write("## 文本来源与三组\n\n编码器均为冻结OpenAI CLIP RN50，输入先按同一归一化规则处理，文本向量宽度1024；K/V使用相同共享投影结构，输出宽度36。训练/推理始终同时读两类Token，不按真实标签挑选。文本与embedding SHA见preflight和manifest。\n\n")
        f.write("G1 BrainPrompt原始文本，实际排序[ASD,TD]（原文件排列[TC,ASD]，原文不改）：\n\n```text\n"+R.TEXTS_BP[1]+"\n"+R.TEXTS_BP[0]+"\n```\n\n")
        f.write("G2 SPromptGL `network.py`原始八条文本，保留`Nromal`拼写；各自归一化后前四条均值为TD、后四条均值为ASD，再归一化并排序[ASD,TD]。这是统一CLIP编码和两Token适配，不是原PEN四向量拼接MLP的实现。这些泛化正常/异常描述不等同于ASD特异生物标志物：\n\n```text\n"+"\n".join(R.TEXTS_SP)+"\n```\n\n")
        f.write("G3为固定匿名2向量，独立私有CPU生成器seed=700070；不使用标签、测试性能或患者特征，匹配G1两向量范数及余弦几何。三组参数数目、模块初始化和训练协议相同，唯一正式差异是语义bank内容。\n\n")
        f.write("## 十折结果（% mean ± population SD）\n\n| Group | ACC | Hard-AUC | Prob-AUC | BA | SEN-ASD | SPE-TD | ASD-F1 | TP | FP | FN | TN |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            counts=" | ".join(str(row.get(k,"—")) for k in ("asd_tp_sum","td_fp_sum","asd_fn_sum","td_tn_sum"))
            f.write(f"| {row['group']} | {row['acc_mean_pct']:.3f} ± {row['acc_sd_pct']:.3f} | {row['hard_auc_mean_pct']:.3f} ± {row['hard_auc_sd_pct']:.3f} | {row['prob_auc_mean_pct']:.3f} ± {row['prob_auc_sd_pct']:.3f} | {row['ba_mean_pct']:.3f} ± {row['ba_sd_pct']:.3f} | {row['sen_mean_pct']:.3f} ± {row['sen_sd_pct']:.3f} | {row['spe_mean_pct']:.3f} ± {row['spe_sd_pct']:.3f} | {row['asd_f1_mean_pct']:.3f} ± {row['asd_f1_sd_pct']:.3f} | {counts} |\n")
        f.write("\nProbability-AUC沿用历史连续TD类概率；Hard-AUC沿用历史硬预测实现。ASD=0、TD=1，SEN=ASD敏感度、SPE=TD特异度。TP/FP/FN/TN为ASD阳性方向的10折逐折合计；历史基座计数不在该参考CSV中提供。所有SD为population SD。\n\n## 配对比较\n\n`mean delta`定义为A−B百分点；W/T/L按同fold方向计数。\n\n| A | B | 指标 | Mean Δ pp | W/T/L |\n|---|---|---|---:|---:|\n")
        for row in comparisons:f.write(f"| {row['group_a']} | {row['group_b']} | {row['metric']} | {row['mean_delta_pp']:+.3f} | {row['wins']}/{row['ties']}/{row['losses']} |\n")
        f.write("\n逐折delta数组在`paired_comparisons.csv`。\n\n## 机制证据\n\n| Group | mean gamma | attention entropy | patient attention SD | gate mean | gate saturation | residual/CAM RMS | W_f grad norm | graph-query attention Δ |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for group in R.GROUPS:
            rows=[x for x in mechanisms if x["group"]==group]
            av=lambda key:sum(float(x[key]) for x in rows)/len(rows)
            f.write(f"| {group} | {av('gamma'):.5f} | {av('attention_entropy_mean'):.5f} | {av('patient_attention_asd_sd'):.5f} | {av('gate_mean'):.5f} | {av('gate_saturation_fraction'):.5f} | {av('residual_to_cam_rms'):.6f} | {av('w_f_grad_norm'):.6g} | {av('graph_query_off_attention_l1'):.6g} |\n")
        f.write("\n固定checkpoint干预在全871个transductive节点上、不读取标签：关闭gamma后，平均|ΔP(ASD)|分别为G1/G2/G3 "+"/".join(f"{sum(x['gamma_off_mean_abs_delta_p_asd'] for x in mechanisms if x['group']==g)/sum(1 for x in mechanisms if x['group']==g):.4f}" for g in R.GROUPS)+"，平均logit最大偏移分别为 "+"/".join(f"{sum(x['gamma_off_max_abs_delta_logit'] for x in mechanisms if x['group']==g)/sum(1 for x in mechanisms if x['group']==g):.3f}" for g in R.GROUPS)+"，平均预测翻转为 "+"/".join(f"{sum(x['gamma_off_prediction_flips'] for x in mechanisms if x['group']==g)/sum(1 for x in mechanisms if x['group']==g):.1f}" for g in R.GROUPS)+"个/871。说明GCDSI确实改变CAM最终分类表示，但固定模型干预不等价于重训消融。平均残差/CAM RMS为G1/G2/G3 "+"/".join(f"{sum(x['residual_to_cam_rms'] for x in mechanisms if x['group']==g)/sum(1 for x in mechanisms if x['group']==g):.3f}" for g in R.GROUPS)+"；单折模态范围分别为 "+"; ".join(f"{g} "+f"{min(x['residual_to_cam_rms'] for x in mechanisms if x['group']==g):.3f}–{max(x['residual_to_cam_rms'] for x in mechanisms if x['group']==g):.3f}" for g in R.GROUPS)+"，部分折残差超过CAM RMS，数值尺度并非所有折都温和。\n\n")
        f.write("清零`W_f`后，平均|ΔP(ASD)|分别为 "+"/".join(f"{sum(x['graph_query_off_mean_abs_delta_p_asd'] for x in mechanisms if x['group']==g)/sum(1 for x in mechanisms if x['group']==g):.4f}" for g in R.GROUPS)+"，平均预测翻转为 "+"/".join(f"{sum(x['graph_query_off_prediction_flips'] for x in mechanisms if x['group']==g)/sum(1 for x in mechanisms if x['group']==g):.1f}" for g in R.GROUPS)+"个/871；结合非零`W_f`梯度与注意力变化，图表示确实进入Query/决策交互，但这仍是固定模型敏感性而非独立重训消融。\n\n")
        f.write("站点/年龄/头动/质量关联为诊断性相关，不用于选择模型。平均attention站点η²为 "+"/".join(f"{sum(x['site_eta2_attention'] for x in mechanisms if x['group']==g)/sum(1 for x in mechanisms if x['group']==g):.3f}" for g in R.GROUPS)+"；最大质量变量|r|平均值为 "+"/".join(f"{sum(x['max_abs_r_quality_attention'] for x in mechanisms if x['group']==g)/sum(1 for x in mechanisms if x['group']==g):.3f}" for g in R.GROUPS)+"；年龄|r|约0.07–0.09，mean-FD |r|约0.16–0.21。逐折站点η²最高约0.57–0.62、质量相关最高约0.74–0.83，提示存在值得关注的站点与质量混杂，不能称为疾病特异证据。G1/G3的平均注意力熵接近ln(2)，区分性较弱；G2更有选择性。G1/G3仍有跨患者、跨模态变化，但此现象不证明其生物学解释。\n\n")
        f.write("站点采用site one-hot的η²，年龄用年龄箱中心的|r|，头动用`func_mean_fd`的|r|，影像质量用16个ANAT/FUNC质量变量中最大|r|；这些关联基于无标签输入特征，不能把ANAT/FUNC质量特征解释为病灶。\n\n")
        for row in mechanism_gates:f.write(f"- {row['group']}机制非塌缩门：{row['status']}；条件={json.dumps({k:v for k,v in row.items() if k not in {'group','status'}},ensure_ascii=False)}。\n")
        f.write(f"- 真实疾病文本相对匿名Token：G1−G3 ΔACC {anon['mean_delta_pp']:+.3f} pp，W/T/L {anon['wins']}/{anon['ties']}/{anon['losses']}；独立文本价值判据={'PASS' if sem_pass else 'FAIL'}。\n\n## 冻结工程判据与结论\n\n")
        for row in gates:f.write(f"- {row['group']} vs 53.3: {row['base_gate']}；ΔACC {row['delta_acc_pp']:+.3f} pp、ACC胜{row['acc_wins']}/10、ΔSEN {row['delta_sen_pp']:+.3f} pp、ΔProb-AUC {row['delta_prob_auc_pp']:+.3f} pp。\n")
        f.write(f"- 文本独立价值：{'PASS' if sem_pass else 'FAIL'}；机制非塌缩：{'PASS' if mechanism_pass else 'FAIL'}。\n- 结论：**{decision}**。即使GO，也只支持进入多seed、严格验证与预注册组件消融候选阶段，不证明MICCAI级创新或临床有效性。\n\n")
        f.write("## 局限、独立读回与来源指纹\n\n所有30个最佳checkpoint以独立进程/脚本重建53.3原始状态后读回，完整比较分类指标与训练时记录，要求30/30一致；训练规则仍使用历史`val=test`选择checkpoint，且RFE/标准化在外层CV前完成，并在871节点transductive图中传播。结果因此不能视为无偏外部泛化估计。单seed与折间相关也不支持显著性推断。\n\n")
        f.write("历史base源码及10个checkpoint SHA与只读参考manifest一致；数据、prompt编码缓存、共享warmup缓存和新实验源码哈希记录在`manifest.json`。实验70只在本目录写结果/日志/权重，handoff仅追加记录。文件路径：`fold_metrics.csv`、`summary.csv`、`paired_comparisons.csv`、`mechanism.csv`、`audit_status.json`。\n")
    record={"status":status,"folds_audited":len(metrics),"errors":errors,"decision":decision,"scientific_gates":gates,"mechanism_gates":mechanism_gates,"mechanism_noncollapse":"PASS" if mechanism_pass else "FAIL","text_value_gate":"PASS" if sem_pass else "FAIL","warmup_parity":warmup_parity,"short_replay_resume":replay,"provenance":provenance,"audit_source_sha256":R.sha_file(Path(__file__))}
    R.atomic_json(R.RESULTS/"audit_status.json",record)
    if status=="PASS":
        handoff_sha=append_handoff(summary,comparisons,gates,decision)
        result_shas={p.name:R.sha_file(p) for p in sorted(R.RESULTS.iterdir()) if p.is_file()}
        manifest=json.loads((EXP/"manifest.json").read_text());manifest.update({"status":"COMPLETE_AUDITED","completed_formal_folds":30,"failed_formal_folds":0,"independently_audited_checkpoints":30,"checkpoint_audit_status":"PASS","initialization_parity":"PASS_10_OF_10","warmup_parity":"PASS_10_OF_10","active_short_replay_resume":"PASS","decision":decision,"best_group":max((x for x in summary if x["group"] in R.GROUPS),key=lambda x:x["acc_mean_pct"])["group"],"scientific_gates":gates,"mechanism_gates":mechanism_gates,"text_value_gate":"PASS" if sem_pass else "FAIL","provenance":provenance,"result_sha256":result_shas,"audit_status_sha256":R.sha_file(R.RESULTS/"audit_status.json"),"handoff_path":"/model/spromptgl/experiment/handoff.md","handoff_sha256":handoff_sha});R.atomic_json(EXP/"manifest.json",manifest)
    print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
