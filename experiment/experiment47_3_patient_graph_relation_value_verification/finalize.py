"""Reload all completed checkpoints and produce paired relation analyses."""
from __future__ import annotations
import json, hashlib, itertools
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import confusion_matrix
import integration as I

GROUPS=('D1','D2','D3','D4','D5')
KNOWN={'B0-B':(91.7294,91.6046,93.1360,90.0732),
       'CAM-only':(91.7294,91.4880,94.6346,88.3415)}

def load_ck(group,fold,device='cuda:0'):
    p=I.RT/'checkpoints'/f'{group}_f{fold:02}.best.pt'; return torch.load(p,map_location='cpu',weights_only=False)

def fold_table():
    rows=[]
    for g in GROUPS:
        for f in range(1,11):
            q=load_ck(g,f)
            m=q['metrics'];tm=q.get('train_metrics',{})
            rows.append({'group':g,'fold':f,'seed':0,'best_epoch':q.get('best_epoch'),
                         'epochs_executed':q.get('epochs_executed'),
                         'acc':m['acc'],'hard_auc':m['hard_auc'],'spe':m['spe'],'sen':m['sen'],
                         'asd_tp':m['asd_tp'],'asd_fn':m['asd_fn'],'td_fp':m['td_fp'],'td_tn':m['td_tn'],
                         'train_acc':tm.get('acc'),'train_nll':tm.get('nll'),'test_nll':m.get('nll'),
                         'device_recorded':q.get('device'),'rho':I.rho})
    d=pd.DataFrame(rows);d.to_csv(I.RESULTS/'fold_metrics.csv',index=False);return d

def summary(d):
    cols=['acc','hard_auc','spe','sen'];rows=[]
    for g in GROUPS:
        q=d[d.group==g]; rows.append({'group':g,**{f'{c}_mean_pct':100*q[c].mean() for c in cols},**{f'{c}_sd_pct':100*q[c].std(ddof=0) for c in cols}})
    for g,v in KNOWN.items(): rows.append({'group':g,**{f'{c}_mean_pct':v[i] for i,c in enumerate(cols)},**{f'{c}_sd_pct':np.nan for c in cols}})
    s=pd.DataFrame(rows);s.to_csv(I.RESULTS/'summary_metrics.csv',index=False);return s

def paired(d):
    cols=['acc','hard_auc','spe','sen'];rows=[]
    for g1,g2 in [('D1','D2'),('D1','D3'),('D1','D4'),('D1','D5')]:
        a=d[d.group==g1].sort_values('fold');b=d[d.group==g2].sort_values('fold')
        r={'left':g1,'right':g2}
        for c in cols:
            diff=a[c].to_numpy()-b[c].to_numpy(); r[f'{c}_diff_pp']=100*diff.mean();r[f'{c}_wins']=int((diff>0).sum());r[f'{c}_ties']=int((diff==0).sum());r[f'{c}_losses']=int((diff<0).sum())
        rows.append(r)
    p=pd.DataFrame(rows);p.to_csv(I.RESULTS/'paired_analysis.csv',index=False);return p

def _pred(q): return np.asarray(q['metrics']['pred'],dtype=np.int64)

def corrections(d):
    x,y,folds,_=I.data();rows=[]
    for g in ('D2','D3','D4','D5'):
        r={'left':'D1','right':g,'left_wrong_right_correct':0,'left_correct_right_wrong':0,'net_correction':0,'n_test':0}
        for f,(_,te) in enumerate(folds,1):
            p1=_pred(load_ck('D1',f));p2=_pred(load_ck(g,f));yy=y[te]
            c1=p1==yy;c2=p2==yy;r['left_wrong_right_correct']+=int((~c1&c2).sum());r['left_correct_right_wrong']+=int((c1&~c2).sum());r['n_test']+=len(yy)
        r['net_correction']=r['left_wrong_right_correct']-r['left_correct_right_wrong'];rows.append(r)
    out=pd.DataFrame(rows);out.to_csv(I.RESULTS/'patient_corrections.csv',index=False);return out

def _load_for_eval(group,fold,device):
    a,s=I.load_anchor_agent(fold,device,fixed=True);a.fold=fold;q=load_ck(group,fold,device);I.load_payload(a,q['model']);a.gfr.eval();a.cam.eval();c=I.load_controls(fold,device);return a,q,c

def interventions():
    # Fixed-checkpoint dependence audit for the three central trained models.
    rows=[];device='cuda:0'; evals=('D1','D2','D3','D4','D5','ZERO_MESSAGE')
    for model_group in ('D1','D4','D5'):
        for f in range(1,11):
            a,q,c=_load_for_eval(model_group,f,device)
            with torch.no_grad():
                base_group=model_group
                base=I.scaled_forward(a,base_group,c); base_lp=base[0]; base_m=I.metric_indices(base_lp,a,a.test_idx); base_pred=base_m['pred']; y=a.y_np[a.test_idx]
                for eg in evals:
                    if eg=='ZERO_MESSAGE':
                        _,raw=I.raw_parts(a); lp,h=a.cam(raw); aux={'M_norm':torch.zeros((),device=raw.device),'Z_norm':raw.norm()}
                    else:
                        lp,h,_,_,_,aux=I.scaled_forward(a,eg,c)
                    m=I.metric_indices(lp,a,a.test_idx);pred=m['pred'];
                    true_lp=float(lp[a.test_idx,torch.as_tensor(y,device=lp.device)].mean().cpu())
                    base_true_lp=float(base_lp[a.test_idx,torch.as_tensor(y,device=lp.device)].mean().cpu())
                    rows.append({'model_group':model_group,'fold':f,'intervention':eg,
                                 'acc':m['acc'],'hard_auc':m['hard_auc'],'spe':m['spe'],'sen':m['sen'],
                                 'logp_true_mean':true_lp,'delta_logp_true':true_lp-base_true_lp,
                                 'flip_count':int((pred!=base_pred).sum()),
                                 'base_acc':base_m['acc'],'message_norm':float(aux['M_norm'].detach().cpu()),
                                 'Z_norm':float(aux['Z_norm'].detach().cpu())})
    out=pd.DataFrame(rows);out.to_csv(I.RESULTS/'interventions.csv',index=False);return out

def main():
    d=fold_table();s=summary(d);p=paired(d);c=corrections(d);iv=interventions()
    print(s.to_string(index=False));print('\nPAIRED\n',p.to_string(index=False));print('\nCORRECTIONS\n',c.to_string(index=False));
    print('\nINTERVENTION MEANS\n',iv.groupby(['model_group','intervention'])[['acc','hard_auc','flip_count','delta_logp_true']].mean().to_string())

if __name__=='__main__':main()
