from __future__ import annotations
import json,importlib.util,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("e488_run",HERE/"run.py")
run=importlib.util.module_from_spec(spec);sys.modules[spec.name]=run;spec.loader.exec_module(run)
c=run.context();m=c["method"];t=c["torch"];x,y,folds,mapping=run.data();tr,te=folds[0]

def check_srd(a,parts):
    z=a.APCR.representations(parts)
    assert all(t.allclose(v.norm(dim=-1),t.ones(v.shape[0],device=v.device),atol=1e-5) for v in z)
    zi,zj=z[0][:32],z[0][32:64];co=(zi*zj).sum(-1)
    q1=t.sigmoid(a.APCR.scorer[0](zi,zj,co));q2=t.sigmoid(a.APCR.scorer[0](zj,zi,co))
    assert t.allclose(q1,q2,atol=1e-7)
    return z

def main():
    results=[]
    a0=run.initialize_agent(1,1,x,y,tr,te,mapping)
    with t.no_grad():parts0,_,_,_,_=m.base_forward(a0,c["utils"])
    z0=a0.APCR.representations(parts0);raw0=m.graph_scores(a0);ids0,_=a0.APCR.candidates(raw0)
    _,_,pair_mix=m.fit_srd_candidate(a0,z0,ids0,a0.sites,17,y)
    assert pair_mix["global_pairs"]==1024 and pair_mix["candidate_pairs_used"]==1024
    assert pair_mix["global_candidate_unique_pairs"]
    assert pair_mix["global_category_counts"]==[256,256,256,256]
    assert pair_mix["candidate_category_counts"]==[256,256,256,256]
    results.append({"check":"candidate_srd_half_global_half_candidate_unique_balanced","status":"PASS",**pair_mix})
    for gid in range(11):
        a=run.initialize_agent(gid,1,x,y,tr,te,mapping)
        a.MessagePassing0.eval();a.ModalFusion.eval();a.APCR.eval()
        with t.no_grad():parts,base,adjs,graph,_=m.base_forward(a,c["utils"])
        z=check_srd(a,parts);raw=m.graph_scores(a);ids,rv=a.APCR.candidates(raw)
        pairs,pq,support=a.APCR.union_pair_scores(z,ids,t.isin(t.arange(871,device=a.dev),t.as_tensor(a.srd_fit_ids,device=a.dev)))
        assert pairs.shape[0]==pq[0].numel() and support.shape==(len(pairs),4)
        targets=np.asarray(a.srd_fit_ids[:32],dtype=np.int64)
        chosen=[]
        q=a.APCR.score_candidate_edges(z,ids,rv)
        for k in range(4):chosen.append(ids[k].gather(1,q[k].topk(16,1).indices))
        atk,_=m.candidate_attacks_targets(a,chosen,ids,targets,901000+gid)
        for k in range(4):
            ix=t.as_tensor(targets,device=a.dev,dtype=t.long)
            rows=atk[0][k].index_select(0,ix)
            assert not bool((rows==ix[:,None]).any())
            assert bool(((rows!=chosen[k].index_select(0,ix)).sum(1)<=2).all())
        # A bounded reliable SCA zero fallback must remain finite and differentiable.
        loss,n,_=m.reliable_teacher_sca(a,pq,support,pairs,y,[.5,.5,.5,.5])
        assert t.isfinite(loss) and float(loss.detach())==0
        a.variant=run.GROUPS[gid]["variant"]
        # One deterministic training step exercises every configured loss path,
        # including CRA, randomized attacks and no-edge pseudo attacks.
        stat=m.train_step(a,6,90,c["utils"],a.sites,y)
        assert np.isfinite(stat["loss"]) and np.isfinite(stat["main"])
        assert np.isfinite(stat["srd"]) and np.isfinite(stat["sca"]) and np.isfinite(stat["robust"])
        if gid in (3,5,6,7,8,9,10):assert stat["cra"]
        if gid in (3,5,6,7,8):
            for step,choice_key in (("step1","step1_selected_index"),("step2","selected_index")):
                vals=np.asarray(stat["cra"]["candidate_target_nll"][step])
                assert int(stat["cra"][choice_key])==int(vals.argmax())
        if gid==10:
            assert stat["cra"]["changed_edges"]==0
            assert stat["cra"]["kl_target"]==0
            # E10 is a no-edge-change control, but its declared clean-graph
            # extra CE must still be differentiable.
            assert any(p.grad is not None and bool(t.isfinite(p.grad).all()) and float(p.grad.abs().sum())>0 for p in a.APCR.parameters())
        results.append({"group":run.GROUPS[gid]["name"],"finite_step":True,
                        "pairs":len(pairs),"srd":stat["srd"],"sca":stat["sca"],
                        "cra":stat["cra"]})
        print(results[-1]["group"],"PASS",flush=True)
    run.atomic_json(run.RT/"mechanism_preflight.json",{"status":"PASS","groups":results})
    return results
if __name__=="__main__":main()
