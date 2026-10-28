"""Stage A: corrected source-labelled scale audit and unified loss checks."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np, pandas as pd, torch
import integration as I

def load_original(fold,group,device='cuda:0'):
    a,s=I.B.load_anchor_agent(fold,device,group='T1')
    q=torch.load(I.RT/'checkpoints'/f'{group}_f{fold:02}.best.pt',map_location='cpu',weights_only=False)
    I.B.set_model(a,{'gfr':q['model']['gfr'],'cam':q['model'].get('cam',q['model'].get('pln'))})
    a.gfr.eval();a.cam.eval();return a,q

def metric_row(group,fold,source,alpha,centered,part,lp,hid,a,raw,M):
    idx=a.train_idx if part=='train' else a.test_idx
    m=I.metric_indices(lp,a,idx)
    nll=I.weighted_nll(lp,a,idx)
    return {'group':group,'fold':fold,'source':source,'alpha':alpha,'centered':centered,'partition':part,
            'acc':m['acc'],'hard_auc':m['hard_auc'],'spe':m['spe'],'sen':m['sen'],'nll':nll,
            'M_norm':float(M.norm()),'X_norm':float(raw.norm()),'M_over_X':float(M.norm()/(raw.norm()+1e-12)),
            'patient_M_sd':float((M-M[a.train_idx].mean(0)).std()),'hidden_norm':float(hid.norm())}

def run_scale():
    rows=[]
    for fold in range(1,11):
        for group in ('g0','T1'):
            a,q=load_original(fold,group)
            with torch.no_grad():
                o=I.B.forward_components(a);ps,raw=I.raw_parts(a);bs=I.branch_outputs(a,o['adjs'],raw);M=bs[1]['scaled']+bs[2]['scaled']+bs[3]['scaled']
                mu=M[a.train_idx].mean(0);mc=M-mu
                # Verify the original GFR/CAM output is the alpha=1 path.
                original_err=float((o['feat']-(raw+M)).abs().max())
                for alpha in (0.,.01,.1,.25,.5,1.):
                    lp,h=a.cam(raw+alpha*M)
                    for part in ('train','test'): 
                        r=metric_row(f'{group}_ALPHA',fold,group,alpha,0,part,lp,h,a,raw,M);r['original_forward_max_abs_error']=original_err;rows.append(r)
                lp,h=a.cam(raw+mc)
                for part in ('train','test'):
                    r=metric_row(f'{group}_CENTERED',fold,group,1.,1,part,lp,h,a,raw,M);r['original_forward_max_abs_error']=original_err;rows.append(r)
    p=I.RESULTS/'corrected_scale_audit.csv';p.parent.mkdir(exist_ok=True)
    with p.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return pd.DataFrame(rows)

def unified_loss_audit():
    rows=[]
    # Original GFR historical best and T1 are evaluated in exactly the same
    # eval/no-dropout weighted-NLL path.
    for fold in range(1,11):
        for group in ('g0','T1'):
            a,q=load_original(fold,group)
            with torch.no_grad(): o=I.B.forward_components(a)
            tr,te=I.train_metrics(o['logp'],a)
            rows.append({'group':group,'fold':fold,'train_acc':tr['acc'],'test_acc':te['acc'],'train_nll':tr['nll'],'test_nll':te['nll'],'best_epoch':q.get('best_epoch')})
        # Existing D1 checkpoint uses the verified 47.2 scaled path.
        a,s=I.load_anchor_agent(fold,'cuda:0',fixed=True);q=torch.load(I.RT/'checkpoints'/f'D1_f{fold:02}.best.pt',map_location='cpu',weights_only=False);I.load_payload(a,q['model']);a.eval_mode=False;a.gfr.eval();a.cam.eval();a.fold=fold
        c=I.load_controls(fold,'cuda:0')
        with torch.no_grad(): lp,h,_,_,_,aux=I.scaled_forward(a,'D1',c)
        tr,te=I.train_metrics(lp,a)
        rows.append({'group':'D1','fold':fold,'train_acc':tr['acc'],'test_acc':te['acc'],'train_nll':tr['nll'],'test_nll':te['nll'],'best_epoch':q.get('best_epoch')})
    p=I.RESULTS/'unified_loss_audit.csv';pd.DataFrame(rows).to_csv(p,index=False);return pd.DataFrame(rows)

if __name__=='__main__':
    d=run_scale();u=unified_loss_audit();print(d.groupby(['group','partition'])[['acc','nll']].mean().to_string());print(u.groupby('group')[['train_acc','test_acc','train_nll','test_nll']].mean().to_string())
