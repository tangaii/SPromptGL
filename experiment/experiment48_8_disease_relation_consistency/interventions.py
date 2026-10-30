#!/usr/bin/env python3
"""Independent fixed-checkpoint interventions for the 100 research folds."""
import csv, glob, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent; RT=HERE/'.runtime'; RES=HERE/'results'
spec=importlib.util.spec_from_file_location('e488_run_interventions', HERE/'run.py')
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
c=mod.context(); x,y,folds,mapping=mod.data(); m=c['method']; t=c['torch']

def pred(o,te): return o['logp'][te].argmax(1).detach().cpu().numpy()
def flips(a,b): return int(np.count_nonzero(a!=b))
def metric_row(o,te): return m.metrics(o,y,te)

rows=[]
for tp in sorted(RT.glob('tasks/E*_f*.json')):
    r=json.loads(tp.read_text())
    if r.get('status')!='COMPLETE': continue
    gid,fold=int(r['group_id']),int(r['fold']); tr,te=folds[fold-1]
    a=mod.initialize_agent(gid,fold,x,y,tr,te,mapping)
    ck=mod.load(RT/'checkpoints'/f'E{gid}_f{fold:02d}.best.pt'); m.load_bundle(a,ck['model'])
    with t.no_grad():
        clean=m.evaluate(a,6,c['utils'])
        pc=pred(clean,te); mc=metric_row(clean,te)
        base_norm=float(clean['base'].norm().detach().cpu())
        res_norm=float(clean['residual'].norm().detach().cpu())
        msg_norm=float(np.mean([float(v.norm().detach().cpu()) for v in clean.get('deltas',[])]))
        gate_mean=float(np.mean([float(v.mean().detach().cpu()) for v in clean.get('gates',[])]))
        a.APCR.zero_alpha=True
        off=m.evaluate(a,6,c['utils']); po=pred(off,te); mo=metric_row(off,te)
        a.APCR.zero_alpha=False
        perm=torch.roll(torch.arange(x.shape[0],device=a.dev),1)
        wrong=m.evaluate(a,6,c['utils'],wrong_perm=perm); pw=pred(wrong,te); mw=metric_row(wrong,te)
        # Score/identity counterfactual: keep the learned scorer and candidate
        # IDs, but cyclically shift scores within each candidate row so that
        # score-to-patient correspondence is deliberately broken.
        original_score=a.APCR.score_candidate_edges
        def shuffled_scores(z0, ids0, raw0):
            qq=original_score(z0,ids0,raw0)
            return [v.roll(1,dims=1) for v in qq]
        a.APCR.score_candidate_edges=shuffled_scores
        score_shift=m.evaluate(a,6,c['utils']); pss=pred(score_shift,te); mss=metric_row(score_shift,te)
        delattr(a.APCR,'score_candidate_edges')
    target=np.asarray(y[te])
    def counts(p):
        return int(((p==target)&(p!=pc)).sum()),int(((p!=target)&(p!=pc)).sum())
    off_newcorr,off_newerr=counts(po); wrong_newcorr,wrong_newerr=counts(pw)
    score_newcorr=int(((pss==target)&(pss!=pc)).sum()); score_newerr=int(((pss!=target)&(pss!=pc)).sum())
    rows.append({
        'group_id':gid,'group':r['group'],'fold':fold,'alpha_best':float(r['diagnostics']['alpha_best']),
        'acc_clean':mc['acc']*100,'hard_auc_clean':mc['hard_auc']*100,'spe_clean':mc['spe']*100,'sen_clean':mc['sen']*100,'probability_auc_clean':mc['probability_auc']*100,
        'acc_alpha_off':mo['acc']*100,'hard_auc_alpha_off':mo['hard_auc']*100,'spe_alpha_off':mo['spe']*100,'sen_alpha_off':mo['sen']*100,'probability_auc_alpha_off':mo['probability_auc']*100,
        'acc_wrong_relation':mw['acc']*100,'hard_auc_wrong_relation':mw['hard_auc']*100,'spe_wrong_relation':mw['spe']*100,'sen_wrong_relation':mw['sen']*100,'probability_auc_wrong_relation':mw['probability_auc']*100,
        'acc_score_perturbed':mss['acc']*100,'hard_auc_score_perturbed':mss['hard_auc']*100,'spe_score_perturbed':mss['spe']*100,'sen_score_perturbed':mss['sen']*100,'probability_auc_score_perturbed':mss['probability_auc']*100,
        'delta_acc_alpha_off_pp':(mo['acc']-mc['acc'])*100,'delta_acc_wrong_relation_pp':(mw['acc']-mc['acc'])*100,
        'delta_acc_score_perturbed_pp':(mss['acc']-mc['acc'])*100,'flip_alpha_off':flips(pc,po),'flip_wrong_relation':flips(pc,pw),'flip_score_perturbed':flips(pc,pss),
        'new_correct_alpha_off':off_newcorr,'new_error_alpha_off':off_newerr,'new_correct_wrong_relation':wrong_newcorr,'new_error_wrong_relation':wrong_newerr,'new_correct_score_perturbed':score_newcorr,'new_error_score_perturbed':score_newerr,
        'mean_abs_probability_change_alpha_off':float((torch.exp(clean['logp'][te,1])-torch.exp(off['logp'][te,1])).abs().mean().detach().cpu()),
        'mean_abs_probability_change_wrong_relation':float((torch.exp(clean['logp'][te,1])-torch.exp(wrong['logp'][te,1])).abs().mean().detach().cpu()),'mean_abs_probability_change_score_perturbed':float((torch.exp(clean['logp'][te,1])-torch.exp(score_shift['logp'][te,1])).abs().mean().detach().cpu()),
        'residual_norm':res_norm,'base_norm':base_norm,'residual_over_base':res_norm/(base_norm+1e-8),'message_norm_mean':msg_norm,'gate_mean':gate_mean,
        'prediction_sha_clean':hashlib.sha256(pc.tobytes()).hexdigest(),'prediction_sha_alpha_off':hashlib.sha256(po.tobytes()).hexdigest(),'prediction_sha_wrong_relation':hashlib.sha256(pw.tobytes()).hexdigest(),'prediction_sha_score_perturbed':hashlib.sha256(pss.tobytes()).hexdigest()
    })
fields=list(rows[0]) if rows else []
RES.mkdir(parents=True,exist_ok=True)
with open(RES/'interventions.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print(json.dumps({'rows':len(rows),'groups':sorted(set(r['group_id'] for r in rows))}))
