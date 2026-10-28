"""Stage A: read-only baseline, message-scale and fixed checkpoint diagnostics."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np
import torch
import integration as I

def _load_best(fold, group, device='cuda:0'):
    a,s=I.load_anchor_agent(fold,device,fixed=False)
    p=I.RT/'checkpoints'/f'{group}_f{fold:02}.best.pt'
    q=torch.load(p,map_location='cpu',weights_only=False)
    I.set_model(a,{'gfr':q['model']['gfr'],'cam':q['model'].get('cam',q['model'].get('pln'))})
    a.gfr.eval();a.cam.eval();return a,q

def _cam_only(fold,device='cuda:0'):
    # Physical CAM-only checkpoint has a single cam state and no GFR keys.
    a=I.make_agent(fold,device)
    q=torch.load(I.RT/'cam_only_checkpoints'/f'g0_f{fold:02}.best.pt',map_location='cpu',weights_only=False)
    a.cam.load_state_dict(q['model']['cam']);a.cam.eval();a.gfr=None
    with torch.no_grad(): logp,hid=a.cam(a.x)
    tr=I.metric_indices(logp,a,a.train_idx);te=I.metric_indices(logp,a,a.test_idx)
    return {'fold':fold,'train':tr,'test':te,'hidden':hid.detach(),'logp':logp.detach()}

def _row(group,fold,part,m,logp,hid,a,aux):
    mm=I.metric_indices(logp,a,a.train_idx if part=='train' else a.test_idx)
    return {'group':group,'fold':fold,'partition':part,**{k:v for k,v in mm.items() if k!='pred'},
            'nll':I.weighted_nll(logp,a,a.train_idx if part=='train' else a.test_idx),
            'logit_min':float(logp.detach().min()),'logit_max':float(logp.detach().max()),
            'raw_norm':float(aux['raw_norm']),'M_norm':float(aux['M_norm']),
            'Mc_norm':float(aux['Mc_norm']),'M_mean_norm':float(aux['train_M_mean_norm']),
            'patient_M_sd':float(aux['patient_var']),
            'Z_norm':float(aux['Z_norm']),'hidden_norm':float(hid.detach().norm()),
            'hidden_patient_sd':float(hid.detach()[a.train_idx if part=='train' else a.test_idx].std()),
            'M_over_X':float(aux['M_norm']/(aux['raw_norm']+1e-12)),
            'Mc_over_X':float(aux['Mc_norm']/(aux['raw_norm']+1e-12)),
            'message_mean_fraction':float(aux['train_M_mean_norm']/(aux['M_norm']+1e-12))}

def run_stage_a():
    I.seed_for_task(0); br=[];sc=[];camrows=[]
    alphas=[0.,.01,.1,.25,.5,1.]
    for fold in range(1,11):
        for group in ('g0','T1'):
            a,q=_load_best(fold,group)
            with torch.no_grad():
                ps,raw=I.raw_parts(a);o=I.forward_components(a);adjs=o['adjs']
                bs=I.branch_outputs(a,adjs,raw);M=bs[1]['scaled']+bs[2]['scaled']+bs[3]['scaled']
                tr=torch.as_tensor(a.train_idx,device=raw.device);mu_m=M[tr].mean(0);mc=M-mu_m
                aux={'raw_norm':raw.norm(),'M_norm':M.norm(),'Mc_norm':mc.norm(),'train_M_mean_norm':M[tr].mean(0).norm(),'patient_var':(M[tr]-M[tr].mean(0)).pow(2).mean().sqrt(),'Z_norm':(raw+mc).norm()}
                # Baseline is the trained original GFR output, not the manually
                # reconstructed scale path. It is the historical comparison.
                logp,hid=o['logp'],o['hidden']
                for part in ('train','test'): br.append(_row(group,fold,part,{},logp,hid,a,aux))
                # The alpha path is a fixed-checkpoint intervention and is not
                # used to choose a training setting.
                for alpha in alphas:
                    z=raw+float(alpha)*M; lp,h=a.cam(z)
                    for part in ('train','test'):
                        rr=_row('T1_ALPHA',fold,part,{},lp,h,a,aux);rr['alpha']=alpha;rr['centered']=0;sc.append(rr)
                zc=raw+mc;lp,h=a.cam(zc)
                for part in ('train','test'):
                    rr=_row('T1_CENTERED',fold,part,{},lp,h,a,aux);rr['alpha']=1.;rr['centered']=1;sc.append(rr)
        # CAM-only validation and hidden representation isolation.
        c=_cam_only(fold)
        for part,d in [('train',c['train']),('test',c['test'])]:
            camrows.append({'group':'CAM_ONLY','fold':fold,'partition':part,**{k:v for k,v in d.items() if k!='pred'},
                            'hidden_norm':float(c['hidden'].norm()),'parameter_count':sum(p.numel() for p in I.make_cam().parameters())})
    def write(path,rows):
        path=I.RESULTS/path;path.parent.mkdir(exist_ok=True)
        keys=[]
        for r in rows:
            for k in r:
                if k not in keys:keys.append(k)
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    write('baseline_diagnostics.csv',br);write('scale_diagnostics.csv',sc);write('cam_only_validation.csv',camrows)
    return br,sc,camrows

if __name__=='__main__':
    b,s,c=run_stage_a();print(json.dumps({'baseline_rows':len(b),'scale_rows':len(s),'cam_rows':len(c)},indent=2))
