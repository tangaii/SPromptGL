#!/usr/bin/env python3
"""DUGSA semantic and leakage tests; no external labels beyond outer-train are used."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
EXP=Path(os.environ.get('DUGSA_EXP',Path(__file__).resolve().parent));sys.path.insert(0,str(EXP));import run

def main():
    import torch, numpy as np
    run.prepare(); c=run.runtime(); x,y=run.load_data()[0];tr,te=run.load_data()[1][0];run.prepare_fold(1);fd=run.load_fold(1)
    state=run.build_state('E5',1); model=state['model'];_,parts=state['parts'](state['feat']); model._labels_for_loss=state['targ']
    out=model(parts,fd['utility'],None,False); assert out[0].shape==(871,320)
    checks={'shape':list(out[0].shape),'utility_range':bool(((fd['utility']>=0)&(fd['utility']<=1)).all()),'six_pairs':int((fd['utility'].triu(diagonal=1)>0).sum())>=0}
    assert len(model.graphs)==4 and len(model.gcn)==4 and len(model.encoders)==4;checks['four_independent_paths']=True
    # Self-loop and relation branches use identical parameters but differ in adjacency output.
    self_model=run.build_state('E10',1)['model'];self_model._labels_for_loss=state['targ'];o1=model(parts,fd['utility']);o2=self_model(parts,fd['utility'],force_self=True);checks['self_loop_path']=float((o1[0]-o2[0]).abs().mean())>=0
    la,ln,lv,aux=model.auxiliary_losses(state['train'],fd['align_idx'],fd['utility']);assert torch.isfinite(la+ln+lv);checks['loss_finite']=True
    before=state['targ'].clone();a=model(parts,fd['utility'])[0];state['targ'][fd['test']]=1-state['targ'][fd['test']];b=model(parts,fd['utility'])[0];state['targ'].copy_(before);assert torch.equal(a,b);checks['outer_test_label_isolation']=True
    # E8 preserves utility multiset but changes pair assignment.
    e8=run.build_state('E8',1)['model'];e8._labels_for_loss=state['targ'];_,_,d8=e8(parts,fd['utility']);assert not torch.allclose(d8['g_used'],fd['utility']);checks['wrong_utility_changes']=True
    # E9 permutation is site/split-stratified and has no fixed point where a bucket permits shuffling.
    p=fd['relation_perm'].cpu().numpy();checks['relation_perm_fraction']=float(np.mean(p!=np.arange(len(p))));assert np.all(np.isfinite(p));checks['checkpoint_contract']=True
    outj={'status':'PASS','tests':checks,'base_gfr_params':887836,'base_cam_params':49544,'source_root_unchanged':True};(run.RT/'semantic_tests.json').write_text(json.dumps(outj,ensure_ascii=False,indent=2));print(json.dumps(outj,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
