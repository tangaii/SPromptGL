"""Independent CF-CPDEM checkpoint replay and continuation audit."""
from __future__ import annotations
import copy,hashlib,json,time,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run
from evidence_memory import build_crossfit_memories,_build_bank
from memory_common import output_layer_logp

def _metrics(x):return {k:float(v) for k,v in x.items() if k not in ("prob","pred")}
def _close(a,b,atol=1e-7):return bool(np.allclose(np.asarray(a),np.asarray(b),atol=atol,rtol=1e-7))

def _memory_validity(memory,eligible,test_ids,fold,parts,labels):
    legal=set(map(int,eligible));test=set(map(int,test_ids));excluded=run.FOLD2_EXCLUDE if fold==2 else set();checks={}
    for name,m in memory.items():
        bank=set(map(int,m.get("bank_ids",[])));src=set(map(int,m.get("source_ids",[])));sup=set(map(int,m.get("support_ids",[])))
        ids=set(map(int,m.get("ids",[])))
        checks[name]={"bank_subset_eligible":bank<=legal,"no_test":not(bank&test),"no_fold2_overlap":not(bank&excluded),
                      "source_support_disjoint":not(src&sup),"source_support_cover":src|sup==bank,
                      "source_75_support_25":len(sup)==int(round(len(bank)*.25)),
                      "memory_ids_from_source":ids<=src,"support_records_valid":True}
        for pid,same,opp in zip(m.get("ids",[]),m.get("support_same_ids",[]),m.get("support_opposite_ids",[])):
            cls=int(labels[int(pid)])
            if int(pid) in same or int(pid) in opp or set(same)&set(opp) or not set(same+opp)<=sup or len(same)<4 or len(opp)<4:
                checks[name]["support_records_valid"]=False
            if any(int(labels[int(q)])!=cls for q in same) or any(int(labels[int(q)])==cls for q in opp):
                checks[name]["support_records_valid"]=False
        if checks[name]["support_records_valid"]:
            if len(m.get("support_same_ids",[]))!=len(m.get("ids",[])) or len(m.get("support_opposite_ids",[]))!=len(m.get("ids",[])):
                checks[name]["support_records_valid"]=False
    checks["crossfit"]={"A_B_bank_disjoint":not(set(parts["A"]["bank_ids"])&set(parts["B"]["bank_ids"])),
                         "A_reads_B":not(set(parts["A"]["query_ids"])&set(memory["B"]["bank_ids"])),
                         "B_reads_A":not(set(parts["B"]["query_ids"])&set(memory["A"]["bank_ids"]))}
    return checks

def _memory_rebuild_match(saved,recomputed):
    if not saved:return not recomputed,{}
    fields=("ids","label","reference","mask","direction","reliability","r_self","r_within","r_between","r_coverage",
            "support_auc","support_separation","joint_drop","support_same_ids","support_opposite_ids","source_ids","support_ids","bank_ids",
            "source_ids_digest","support_ids_digest","version","refresh_epoch","individual_margin_effects",
            "candidate_count","accepted_count","rejected_count")
    detail={}
    for name in ("A","B","full"):
        a=saved[name];b=recomputed[name];ok=True;checks={}
        for key in fields:
            av=a.get(key);bv=b.get(key)
            if torch.is_tensor(av) or torch.is_tensor(bv):
                value=torch.is_tensor(av) and torch.is_tensor(bv) and av.shape==bv.shape and torch.allclose(av.float().cpu(),bv.float().cpu(),atol=1e-6,rtol=1e-6)
            else:value=av==bv
            checks[key]=bool(value);ok=ok and bool(value)
        detail[name]={"match":ok,"fields":checks}
    return all(x["match"] for x in detail.values()),detail

def _rebuild_at_refresh(saved,parts,state):
    results={};device=state["y"].device
    for name,m in saved.items():
        raw=m["build_inputs"];ids=m["bank_ids"];ix=torch.tensor(ids,dtype=torch.long,device=device)
        cam=torch.zeros((871,144),device=device);lp=torch.zeros((871,2),device=device)
        labels=torch.full((871,),-1,dtype=torch.long,device=device);sites=torch.full_like(labels,-1)
        cam[ix]=raw["cam"].to(device);lp[ix]=raw["logp"].to(device)
        labels[ix]=raw["labels"].to(device);sites[ix]=raw["sites"].to(device)
        head=copy.deepcopy(state["cam"].Outputlayer);head.load_state_dict(raw["head"]);head.eval()
        results[name]=_build_bank(cam,labels,lp,parts[name],sites,head,version=m["version"],refresh_epoch=m["refresh_epoch"],config=run.CFG)
        actual=hashlib.sha256(raw["cam"].contiguous().numpy().tobytes()).hexdigest()
        if actual!=m["source_cam_digest"]:raise RuntimeError(f"refresh CAM digest mismatch: {name}")
    return results

def _replay_branch(module,state,ids,cam,base_lp,memory,intervention=None):
    module.eval();ix=torch.tensor(ids,dtype=torch.long,device=cam.device)
    with torch.no_grad():
        enhanced,diag=module(ids,cam[ix].detach(),base_lp[ix].detach(),memory,intervention=intervention,collect_retrieval=True)
        lp=output_layer_logp(state["cam"].Outputlayer,enhanced)
        # The intervention definition is an exact return to the saved original
        # probability. A smaller head batch can differ in last-bit reduction
        # order even for bitwise-identical CAM input; report that separately.
        if memory is None:
            diag["off_head_max_logp_error"]=(lp-base_lp[ix]).abs().max()
            lp=base_lp[ix].clone()
    return lp,enhanced,diag

def _one_step_signature(fold,payload):
    state,warm,identity,task,ckpt=run.prepare_fold(fold);runtime=run.legacy.history_runtime();hist=run.legacy.load_base().load_history()
    hist.load_bundle(state,payload["base_state"]);runtime.restore_rng(payload["rng"])
    module=run.init_module(state["y"].device);module.load_state_dict(payload["module"])
    optimizer=torch.optim.AdamW(module.parameters(),lr=run.CFG["optimizer"]["lr"],weight_decay=run.CFG["optimizer"]["weight_decay"])
    optimizer.load_state_dict(payload["optimizer"])
    return run.continuation_signature(state,module,optimizer,payload["memories"],payload["partitions"],
        run.eligible_ids(state,fold),payload["next_epoch_index"],payload["memory_version"])

def _continue_replay(fold,payload):
    runtime=run.legacy.history_runtime();original=runtime.pack_rng()
    try:
        a=_one_step_signature(fold,payload)
        b=json.loads((run.TASKS/f"E79.4-G1_f{fold:02d}.continuation.json").read_text())
        checks={key:a.get(key)==value for key,value in b.items()}
        return {"pass":all(checks.values()),"checks":checks,"live_reference":b,"reloaded_observed":a,
                "base_parameters_equal":checks["base_state_digest"],"memory_parameters_equal":checks["module_digest"],
                "optimizer_equal":checks["optimizer_digest"],"rng_equal":checks["rng_after_memory"]}
    finally:runtime.restore_rng(original)

def _enhancement_summary(h,base_lp,diag,y):
    """Read-only patient strata; labels are used only after checkpoint loading."""
    correct=base_lp.argmax(1).eq(y)
    conflict=diag["evidence_logits"].argmax(1).ne(base_lp.argmax(1))&diag["evidence_valid"]
    ratio=diag["delta_ratio"];out={}
    for name,mask in (("all",torch.ones_like(correct)),("base_correct",correct),("base_wrong",~correct),
                      ("memory_conflict",conflict),("memory_agreement",~conflict)):
        n=int(mask.sum());values=ratio[mask]
        out[name]={"n":n,"gate_mean":float(diag["gate"][mask].mean()) if n else 0.,
            "delta_ratio_mean":float(values.mean()) if n else 0.,
            "delta_ratio_p50":float(torch.quantile(values,.5)) if n else 0.,
            "delta_ratio_p90":float(torch.quantile(values,.9)) if n else 0.,
            "delta_ratio_p95":float(torch.quantile(values,.95)) if n else 0.,
            "near_cap_count":int((values>.99*.075).sum()),"zero_count":int((values==0).sum())}
    raw=diag["raw_delta"].double();delta=diag["delta_h"].double()
    nonzero=raw.norm(dim=1).gt(1e-12)&delta.norm(dim=1).gt(1e-12)
    cosine=(raw[nonzero]*delta[nonzero]).sum(1)/(raw[nonzero].norm(dim=1)*delta[nonzero].norm(dim=1)).clamp_min(1e-30)
    out["raw_final_direction_cosine_mean"]=float(cosine.mean()) if bool(nonzero.any()) else 0.
    out["attention_reliability_asd_mean"]=float(diag["reliability_asd"].mean())
    out["attention_reliability_td_mean"]=float(diag["reliability_td"].mean())
    return out


def audit_best(fold,ledger,state,hist_best,eligible,test_ids,y,sites,parts):
    p=ROOT/f".runtime/checkpoints/E79.4-G1_f{fold:02d}.{ledger}.pt";rec=run.torch_load(p);hist=run.legacy.load_base().load_history()
    hist.load_bundle(state,rec["base_state"])
    with torch.no_grad():evaluated,cam=run.hook_forward(state,False)
    base_lp=evaluated["logp"].detach();base_metrics=run.legacy.extended_metrics(base_lp,test_ids,y)
    module=run.init_module(y.device);module.load_state_dict(rec["module"]);module.eval()
    memory=rec["memory_used"]
    rebuilt=_rebuild_at_refresh(memory,parts,state) if memory else {}
    rebuilt_ok,rebuilt_detail=_memory_rebuild_match(memory,rebuilt)
    validity=_memory_validity(memory,eligible,test_ids,fold,parts,y.detach().cpu()) if memory else {"empty":{"before_first_read":rec["best_epoch_1based"]<=11}}
    for name,m in (memory or {}).items():
        validity[name].update({"read_old_memory":m["refresh_epoch"]<rec["best_epoch_1based"],
            "capacity":all(int(m["label"].eq(c).sum())<=48 for c in (0,1)),
            "reliability_product":torch.allclose(m["reliability"],m["r_self"]*m["r_within"]*m["r_between"]*m["r_coverage"],atol=1e-7),
            "support_auc_threshold":bool((m["support_auc"]>=.6).all()),"support_separation_positive":bool((m["support_separation"]>0).all()),
            "mask_dimensions":bool((m["mask"].sum(1)<=12).all()),"no_graph":all(not v.requires_grad for v in m.values() if torch.is_tensor(v))})
    full_memory=memory["full"] if memory else None
    lp,enh,diag=_replay_branch(module,state,test_ids,cam,base_lp,full_memory)
    full=base_lp.clone();ti=torch.tensor(test_ids,dtype=torch.long,device=y.device);full[ti]=lp
    metrics=run.legacy.extended_metrics(full,test_ids,y)
    probability_ok=_close(metrics["prob"],rec["probability"]) and _close(base_metrics["prob"],rec["base_probability"])
    prediction_ok=np.array_equal(np.asarray(metrics["pred"],int),np.asarray(rec["prediction"],int))
    base_prediction_ok=np.array_equal(np.asarray(base_metrics["pred"],int),np.asarray(rec["base_prediction"],int))
    anchor_ok=True
    if ledger=="base_anchored":anchor_ok=run.same_state(rec["base_state"],hist_best["best_state"])
    off_lp,off_h,off_diag=_replay_branch(module,state,test_ids,cam,base_lp,None)
    off_exact=torch.equal(off_h,cam[ti]) and torch.equal(off_lp,base_lp[ti])
    with torch.no_grad():full_batch_head=output_layer_logp(state["cam"].Outputlayer,cam)
    full_head_match=torch.equal(full_batch_head,base_lp)
    interventions=[{"fold":fold,"ledger":ledger,"method":run.GROUP,"epoch":rec["best_epoch_1based"],
      "intervention":"Memory-Off","metrics":_metrics(base_metrics),
      "full_minus_intervention_acc":float(metrics["acc"]-base_metrics["acc"]),
      "prob_auc_delta":float(metrics["prob_auc"]-base_metrics["prob_auc"]),
      "prediction_flips":int(np.sum(np.asarray(metrics["pred"])!=np.asarray(base_metrics["pred"]))),
      "mean_abs_prob_delta":float(np.abs(np.asarray(metrics["prob"])-np.asarray(base_metrics["prob"])).mean()),
      "delta_ratio_mean":0.0,"learned_fixed_gap":0.0}]
    content_checks=[];content_value_changes=[];content_original_ids=[];content_permuted_ids=[]
    for name in ("Memory-Content-Permuted","Learnable-Matching-Off","Cross-Patient-Reliability-Off"):
        ilp,ih,idg=_replay_branch(module,state,test_ids,cam,base_lp,full_memory,name)
        if name=="Memory-Content-Permuted":
            for a,b in zip(diag["retrieval"],idg["retrieval"]):
                for side in ("asd","td"):
                    content_checks.append([(z["patient_id"],z["attention"]) for z in a[side]]==[(z["patient_id"],z["attention"]) for z in b[side]])
                    content_value_changes.extend(z["value_patient_id"]!=z["patient_id"] for z in b[side])
                    content_original_ids.extend(z["patient_id"] for z in b[side])
                    content_permuted_ids.extend(z["value_patient_id"] for z in b[side])
        f=base_lp.clone();f[ti]=ilp;im=run.legacy.extended_metrics(f,test_ids,y)
        interventions.append({"fold":fold,"ledger":ledger,"method":run.GROUP,"epoch":rec["best_epoch_1based"],
          "intervention":name,"metrics":_metrics(im),"full_minus_intervention_acc":float(metrics["acc"]-im["acc"]),
          "prob_auc_delta":float(metrics["prob_auc"]-im["prob_auc"]),
          "prediction_flips":int(np.sum(np.asarray(metrics["pred"])!=np.asarray(im["pred"]))),
          "mean_abs_prob_delta":float(np.abs(np.asarray(metrics["prob"])-np.asarray(im["prob"])).mean()),
          "delta_ratio_mean":float(idg["delta_ratio"].mean()),"learned_fixed_gap":float(idg["learned_fixed_gap"].mean()),
          "mean_abs_evidence_logit_delta":float((diag["evidence_logits"]-idg["evidence_logits"]).abs().mean()),
          "mean_abs_gate_delta":float((diag["gate"]-idg["gate"]).abs().mean()),
          "retrieved_patient_lists_changed":sum(any([z["patient_id"] for z in a[side]]!=[z["patient_id"] for z in b[side]] for side in ("asd","td")) for a,b in zip(diag["retrieval"],idg["retrieval"])),
          "probability_asd":np.asarray(im["prob"])[:,0].tolist(),"transition":run.correction_stats(base_lp,f,test_ids,y)})
    content_raw_changed=content_encoded_changed=False;content_max_raw_delta=content_max_encoded_delta=0.
    if full_memory and content_original_ids:
        mi={pid:i for i,pid in enumerate(full_memory["ids"])}
        original_indices=torch.tensor([mi[p] for p in content_original_ids],dtype=torch.long)
        permuted_indices=torch.tensor([mi[p] for p in content_permuted_ids],dtype=torch.long)
        values=(full_memory["reference"]-full_memory["neutral_mean"])*full_memory["mask"]+.5*full_memory["direction"]
        encoded=values@module.Wv.weight.detach().cpu().T
        content_max_raw_delta=float((values[original_indices]-values[permuted_indices]).abs().max())
        content_max_encoded_delta=float((encoded[original_indices]-encoded[permuted_indices]).abs().max())
        content_raw_changed=content_max_raw_delta>0.;content_encoded_changed=content_max_encoded_delta>0.
    transition=run.correction_stats(base_lp,full,test_ids,y)
    history_lp=base_lp.clone();history_lp[ti]=torch.as_tensor(hist_best["test_probability"],dtype=base_lp.dtype,device=y.device).clamp_min(1e-30).log()
    history_transition=run.correction_stats(history_lp,full,test_ids,y)
    retrieval_ids=[int(z["patient_id"]) for row in diag["retrieval"] for side in (row["asd"],row["td"]) for z in side]
    bad=set(retrieval_ids)&(set(test_ids)|set(run.FOLD2_EXCLUDE if fold==2 else set()))
    ev=diag["evidence_logits"].detach();evy=y[ti];evmask=diag["evidence_valid"].detach();evpred=ev.argmax(1)
    ev_metrics={"valid_fraction":float(evmask.float().mean()),
        "accuracy":float(evpred[evmask].eq(evy[evmask]).float().mean()) if bool(evmask.any()) else 0.0,
        "asd_accuracy":float(evpred[evmask&evy.eq(0)].eq(0).float().mean()) if bool((evmask&evy.eq(0)).any()) else 0.0,
        "td_accuracy":float(evpred[evmask&evy.eq(1)].eq(1).float().mean()) if bool((evmask&evy.eq(1)).any()) else 0.0,
        "valid_count":int(evmask.sum())}
    from sklearn.metrics import balanced_accuracy_score,roc_auc_score
    if bool(evmask.any()):
        yy=evy[evmask].cpu().numpy();ep=evpred[evmask].cpu().numpy();prob=torch.softmax(ev,1)[evmask,0].cpu().numpy()
        ev_metrics["hard_auc"]=float(balanced_accuracy_score(yy,ep))
        ev_metrics["prob_auc"]=float(roc_auc_score(yy==0,prob)) if len(set(yy))==2 else 0.
    bc=base_lp[ti].argmax(1).eq(evy);mc=evpred.eq(evy)&evmask;fc=lp.argmax(1).eq(evy)
    ev_metrics.update({"base_wrong_memory_correct":int((~bc&mc).sum()),"base_wrong_memory_wrong":int((~bc&~mc).sum()),
        "memory_correct_fusion_still_wrong":int((~bc&mc&~fc).sum()),"memory_wrong_harm":int((bc&~mc&~fc).sum())})
    t=(1-2*evy).float();margin_gain=t*((lp[:,0]-lp[:,1])-(base_lp[ti,0]-base_lp[ti,1]))
    ev_metrics["wrong_base_margin_improvement_rate"]=float((margin_gain[~bc]>0).float().mean()) if bool((~bc).any()) else 0.
    retrieval=[z for row in diag["retrieval"] for side in (row["asd"],row["td"]) for z in side]
    ev_metrics["retrieval_rerank_fraction"]=float(np.mean([z["fixed_rank"]>8 for z in retrieval])) if retrieval else 0.
    training_evidence=copy.deepcopy(rec.get("train_diagnostics",{}))
    if memory:
        all_scores=[];all_valid=[];all_y=[];all_base=[];all_h=[];all_diag=[]
        for query_bank,read_bank in (("A","B"),("B","A")):
            qids=parts[query_bank]["query_ids"];qi=torch.tensor(qids,dtype=torch.long,device=y.device)
            with torch.no_grad():_,qd=module(qids,cam[qi].detach(),base_lp[qi].detach(),memory[read_bank])
            all_scores.append(qd["evidence_logits"]);all_valid.append(qd["evidence_valid"]);all_y.append(y[qi])
            all_h.append(cam[qi]);all_base.append(base_lp[qi]);all_diag.append(qd)
        es=torch.cat(all_scores);valid=torch.cat(all_valid);ey=torch.cat(all_y);ep=es.argmax(1)
        if bool(valid.any()):
            yy=ey[valid].cpu().numpy();pp=ep[valid].cpu().numpy();prob=torch.softmax(es,1)[valid,0].cpu().numpy()
            training_evidence.update({"replayed_evidence_accuracy":float((yy==pp).mean()),
                "replayed_evidence_hard_auc":float(balanced_accuracy_score(yy,pp)),
                "replayed_evidence_prob_auc":float(roc_auc_score(yy==0,prob)) if len(set(yy))==2 else 0.})
        joined={key:torch.cat([d[key] for d in all_diag]) for key in
                ("gate","delta_ratio","evidence_logits","evidence_valid","raw_delta","delta_h","reliability_asd","reliability_td")}
        train_enhancement=_enhancement_summary(torch.cat(all_h),torch.cat(all_base),joined,ey)
    else:train_enhancement={}
    return {"row":{"fold":fold,"ledger":ledger,"method":run.GROUP,"checkpoint":str(p),"epoch":rec["best_epoch_1based"],
      "checkpoint_sha256":run.sha_file(p),"training_source_sha256":rec["source_sha256"],"probability_replay_match":bool(probability_ok),
      "method_prediction_match":bool(prediction_ok),"base_prediction_match":bool(base_prediction_ok),"base_anchor_state_match":bool(anchor_ok),
      "cam_shape":list(cam.shape),"empty_memory_exact":bool(off_exact),"memory_count":len(full_memory["ids"]) if full_memory else 0,
      "off_head_max_logp_error":float(off_diag["off_head_max_logp_error"]),
      "off_head_numerically_equivalent":bool(full_head_match),
      "off_head_full_batch_max_logp_error":float((full_batch_head-base_lp).abs().max()),
      "crossfit_memory_rebuild_match":bool(rebuilt_ok),"memory_rebuild":rebuilt_detail,"memory_validity":validity,
      "content_keys_attention_fixed":all(content_checks),"content_value_association_changed":any(content_value_changes),
      "content_raw_values_numerically_changed":content_raw_changed,"content_encoded_values_numerically_changed":content_encoded_changed,
      "content_max_raw_value_delta":content_max_raw_delta,"content_max_encoded_value_delta":content_max_encoded_delta,
      "content_permutation_applicable":any(len(row[side])>1 for row in diag["retrieval"] for side in ("asd","td")),
      "bad_retrieved_patient_ids":sorted(bad),"transition":transition,"gate_mean":float(diag["gate"].mean()),
      "history_transition":history_transition,
      "delta_ratio_mean":float(diag["delta_ratio"].mean()),"delta_ratio_p90":float(torch.quantile(diag["delta_ratio"],.9)),
      "cap_fraction":float(diag["cap_fraction"].mean()),"reliability_mean":float(diag["memory_reliability"].mean()),
      "wq_abs_sum":float(module.Wq.weight.detach().abs().sum()),"wk_abs_sum":float(module.Wk.weight.detach().abs().sum()),
      "learned_fixed_gap":float(diag["learned_fixed_gap"].mean()),"test_memory_evidence":ev_metrics,
      "gate_p50":float(torch.quantile(diag["gate"],.5)),"gate_p90":float(torch.quantile(diag["gate"],.9)),"gate_p95":float(torch.quantile(diag["gate"],.95)),
      "delta_ratio_p50":float(torch.quantile(diag["delta_ratio"],.5)),"delta_ratio_p95":float(torch.quantile(diag["delta_ratio"],.95)),
      "enhancement_diagnostics":{"test":_enhancement_summary(cam[ti],base_lp[ti],diag,evy),"training":train_enhancement},
      "training_evidence":training_evidence,"metrics":_metrics(metrics),"base_metrics":_metrics(base_metrics)},
      "interventions":interventions}

def main():
    started=time.time();best=[];interventions=[];latest_rows=[];resume_rows=[];rng_total=rng_exact=0
    pre=json.loads((ROOT/"results/preflight.json").read_text())
    current=run.source_snapshot()
    audit_source_path=str((ROOT/"audit.py").resolve())
    postprocess_paths={audit_source_path,str((ROOT/"analyze.py").resolve())}
    source_changes={p:{"training_sha256":value,"current_sha256":current.get(p),
        "scope":"postprocessing_only" if p in postprocess_paths else "training_or_history"}
        for p,value in pre["source_hashes"].items() if current.get(p)!=value}
    sources_unchanged=all(change["scope"]=="postprocessing_only" for change in source_changes.values())
    for fold in range(1,11):
        state,warm,identity,hist_task,hist_ckpt=run.prepare_fold(fold);hist_best=run.torch_load(hist_ckpt)
        eligible=run.eligible_ids(state,fold);test_ids=list(map(int,state["test_np"]));y=state["y"].long();sites=run.site_tensor(y.device)
        parts=run.make_partitions(eligible,y.detach().cpu(),sites.detach().cpu())
        for ledger in ("base_anchored","method"):
            result=audit_best(fold,ledger,state,hist_best,eligible,test_ids,y,sites,parts);best.append(result["row"]);interventions.extend(result["interventions"])
            print(f"replayed fold={fold} ledger={ledger} prediction={result['row']['probability_replay_match']} memory={result['row']['crossfit_memory_rebuild_match']}",flush=True)
        latest_path=ROOT/f".runtime/checkpoints/E79.4-G1_f{fold:02d}.latest.pt";latest=run.torch_load(latest_path)
        trace=latest.get("trace",[]);expected=int(trace[-1]["epoch"]) if trace else -1
        history_trace={r["epoch"]:r for r in hist_task["rng_trace"]}
        for r in trace:
            rng_total+=1;rng_exact+=int(all(r[k]==history_trace.get(r["epoch"],{}).get(k) for k in ("rng_before","rng_after_train","rng_after_eval")))
        basic=(expected==int(latest.get("next_epoch_index",-2)) and set(latest.get("memories",{}))=={"A","B","full"}
               and set(latest.get("partitions",{}))=={"A","B","full"} and set(latest.get("rng",{}))=={"python","numpy","cpu","cuda"}
               and latest.get("identity")==identity and latest.get("source_sha256")==best[-1]["training_source_sha256"])
        latest_rows.append({"fold":fold,"status":"PASS" if basic else "FAIL","next_epoch_index":latest.get("next_epoch_index"),
                            "expected_next_epoch_index":expected,"memory_version":latest.get("memory_version"),"required_state_present":bool(basic)})
        resume=_continue_replay(fold,latest);resume["fold"]=fold;resume_rows.append(resume)
        print(f"live versus loaded next step fold={fold}: {resume['pass']}",flush=True)
    best_ok=all(r["probability_replay_match"] and r["method_prediction_match"] and r["base_prediction_match"] and
                r["base_anchor_state_match"] and r["empty_memory_exact"] and r["off_head_numerically_equivalent"] and r["crossfit_memory_rebuild_match"] and
                r["content_keys_attention_fixed"] and (not r["content_permutation_applicable"] or
                  (r["content_value_association_changed"] and r["content_raw_values_numerically_changed"] and r["content_encoded_values_numerically_changed"])) and
                not r["bad_retrieved_patient_ids"] and all(all(v for k,v in check.items() if isinstance(v,bool)) for check in r["memory_validity"].values()) for r in best)
    latest_ok=all(x["status"]=="PASS" for x in latest_rows) and len(resume_rows)==10 and all(x["pass"] for x in resume_rows)
    status={"status":"PASS" if best_ok and latest_ok and pre.get("status")=="PASS" and sources_unchanged and rng_total==rng_exact else "FAIL",
            "best_expected":20,"best_audited":len(best),"base_anchored_audited":sum(x["ledger"]=="base_anchored" for x in best),
            "method_selected_audited":sum(x["ledger"]=="method" for x in best),"latest_expected":10,"latest_audited":len(latest_rows),
            "latest_continuation_replays_expected":10,"latest_continuation_replays_passed":sum(x["pass"] for x in resume_rows),
            "best_pass":bool(best_ok),"latest_pass":bool(latest_ok),"preflight_pass":pre.get("status")=="PASS",
            "source_unchanged":sources_unchanged,"rng_common_triplets":rng_total,"rng_exact_triplets":rng_exact,"rng_mismatches":rng_total-rng_exact,
            "training_audit_source_sha256":pre["source_hashes"][audit_source_path],"independent_audit_source_sha256":current[audit_source_path],
            "source_changes":source_changes,"locked_training_and_history_files_unchanged":sources_unchanged,
            "best_rows":best,"latest_rows":latest_rows,"latest_resume_replay":resume_rows,"interventions":interventions,
            "elapsed_seconds":time.time()-started}
    run.atomic_json(ROOT/"results/audit_status.json",status)
    fields=("fold","ledger","method","epoch","intervention","full_minus_intervention_acc","prob_auc_delta","prediction_flips","mean_abs_prob_delta","delta_ratio_mean","learned_fixed_gap")
    run.write_csv(ROOT/"results/intervention_analysis.csv",interventions)
    print(json.dumps({"status":status["status"],"best":len(best),"latest":len(latest_rows),
                      "latest_replay_pass":status["latest_continuation_replays_passed"],"elapsed":status["elapsed_seconds"]},indent=2))

if __name__=="__main__":main()
