"""Scale-controlled D0--D3 training for the pre-registered pilot/extension."""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import integration as I

EXP=I.EXP; RT=I.RT; RES=I.RESULTS
(RT/'checkpoints').mkdir(exist_ok=True); (RT/'tasks').mkdir(exist_ok=True)

SAMPLE={0,1,2,5,10,20,50,100,150,200}

def _state(a): return I.state_payload(a)

def _norm_grad(a):
    vals={}
    for i,mod in enumerate(I.MODALITIES):
        for nm,p in [('W1',getattr(a.gfr,f'MessagePassing{i}').gcn1.W),('W2',getattr(a.gfr,f'MessagePassing{i}').gcn2.W)]:
            vals[f'{mod}_{nm}_grad']=0. if p.grad is None else float(p.grad.detach().norm().cpu())
    vals['cam_grad']=float(torch.sqrt(sum((p.grad.detach().square().sum() for p in a.cam.parameters() if p.grad is not None),torch.zeros((),device=a.x.device))).cpu())
    return vals

def _forward_eval(a,group,fixed):
    wasg,wasc=a.gfr.training,a.cam.training;a.gfr.eval();a.cam.eval()
    with torch.no_grad(): out=I.scaled_forward(a,group,fixed_adjs=fixed,compute_cam=True)
    if wasg:a.gfr.train()
    if wasc:a.cam.train()
    return out

def _metric_row(a,group,fold,epoch,logp,hid,aux,train_m,test_m,gl,cl,grad):
    return {'group':group,'fold':fold,'epoch':epoch,
            'train_nll':train_m['nll'],'train_acc':train_m['acc'],'test_nll':test_m['nll'],'test_acc':test_m['acc'],
            'train_hard_auc':train_m['hard_auc'],'test_hard_auc':test_m['hard_auc'],
            'train_spe':train_m['spe'],'test_spe':test_m['spe'],'train_sen':train_m['sen'],'test_sen':test_m['sen'],
            'classification_loss':float(cl.detach().cpu()),'graph_loss':float(gl.detach().cpu()),
            'raw_norm':float(aux['raw_norm'].detach().cpu()),'M_norm':float(aux['M_norm'].detach().cpu()),
            'Mc_norm':float(aux['Mc_norm'].detach().cpu()),'scale':float(aux['scale'].detach().cpu()),
            'Z_norm':float(aux['Z_norm'].detach().cpu()),'Xc_train_norm':float(aux['Xc_train_norm'].detach().cpu()),
            'Mc_train_norm':float(aux['Mc_train_norm'].detach().cpu()),
            'M_mean_norm':float(aux['train_M_mean_norm'].detach().cpu()),
            'patient_M_sd':float(aux['patient_var'].detach().cpu()),
            'M_over_X':float((aux['M_norm']/(aux['raw_norm']+1e-12)).detach().cpu()),
            'Mc_over_X':float((aux['Mc_norm']/(aux['raw_norm']+1e-12)).detach().cpu()),
            'cam_hidden_norm':float(hid.detach().norm().cpu()),**grad}

def run_task(group,fold,device='cuda:0',max_epoch=450):
    I.seed_for_task(0)
    fixed=None
    if I.train_group_is_fixed(group):
        fixed=I.fixed_true_graphs(fold,device='cpu')[{'D1':'true','D2':'rewire','D3':'self'}[group]]
    a,anchor=I.load_anchor_agent(fold,device,fixed=I.train_group_is_fixed(group))
    a.opt_gfr=I.make_optimizer(a)
    a.opt_cam=torch.optim.Adam(a.cam.parameters(),lr=a.hp.lr,weight_decay=a.hp.reg)
    # Restore the anchor RNG after all model/optimizer construction.  New
    # groups deliberately share the same initialization and dropout stream.
    I.restore(anchor['rng'])
    start=time.monotonic();best_acc=-1.;best_epoch=-1;wait=0;best=None;best_test=None;best_train=None
    traj=[];best_aux=None
    for epoch in range(max_epoch):
        a.gfr.train();a.cam.train();a.opt_gfr.zero_grad(set_to_none=True);a.opt_cam.zero_grad(set_to_none=True)
        lp,hid,cl,gl,adjs,aux=I.scaled_forward(a,group,fixed_adjs=fixed,compute_cam=True)
        total=cl+gl if group=='D0' else cl
        total.backward();grad=_norm_grad(a)
        a.opt_gfr.step();a.opt_cam.step()
        # Checkpoint selection follows the historical val=test rule. The gate
        # never reads this test accuracy; it reads only train_nll.
        out=_forward_eval(a,group,fixed); elp,eh,_,egl,_,eaux=out
        tm,vm=I.train_metrics(elp,a);acc=vm['acc']
        if acc>best_acc:
            best_acc=acc;best_epoch=epoch;wait=0;best=_state(a);best_test=vm;best_train=tm;best_aux={k:(v.detach().cpu().clone() if torch.is_tensor(v) else v) for k,v in eaux.items()}
        else: wait+=1
        if epoch in SAMPLE or epoch%10==0:
            # use eval metrics and the post-update auxiliary quantities; this
            # does not consume dropout RNG because the model is eval().
            traj.append(_metric_row(a,group,fold,epoch,elp,eh,eaux,tm,vm,egl,cl,grad))
        if wait>200: break
    if best is None: raise RuntimeError('no checkpoint')
    I.load_payload(a,best); out=_forward_eval(a,group,fixed);elp,eh,_,egl,_,eaux=out;tm,vm=I.train_metrics(elp,a)
    pred=vm['pred']
    payload={'group':group,'fold':fold,'device':device,'best_epoch':best_epoch,'epochs_executed':epoch+1,
             'best_acc':best_acc,'model':best,'metrics':vm,'train_metrics':tm,'graph_sha':I.graph_sha(fixed) if fixed is not None else None,
             'anchor_sha':anchor.get('parameter_sha'),'rng':I.rng(),'trajectory':traj}
    outp=RT/'checkpoints'/f'{group}_f{fold:02}.best.pt';torch.save(payload,outp)
    row={'group':group,'fold':fold,'seed':0,'best_epoch':best_epoch,'epochs_executed':epoch+1,
         'acc':vm['acc'],'hard_auc':vm['hard_auc'],'spe':vm['spe'],'sen':vm['sen'],
         'asd_tp':vm['asd_tp'],'asd_fn':vm['asd_fn'],'td_fp':vm['td_fp'],'td_tn':vm['td_tn'],
         'train_acc':tm['acc'],'train_nll':tm['nll'],'test_nll':vm['nll'],'best_sha':hashlib.sha256(torch.cat([q.detach().cpu().flatten() for q in best['cam'].values()]).numpy().tobytes()).hexdigest(),
         'wall_seconds':time.monotonic()-start,'status':'COMPLETE','device':device,
         'rho':I.rho,'graph_sha':payload['graph_sha']}
    pref=RES/f'pilot_{group}_f{fold:02}'
    _write(pref.with_suffix('.trajectory.csv'),traj);_write(pref.with_suffix('.metrics.csv'),[row])
    print('TASK_DONE',group,fold,row['acc'],row['train_nll'],best_epoch,flush=True)
    return row

def _write(p,rows):
    if not rows:return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['D0','D1','D2','D3'],required=True);ap.add_argument('--fold',type=int,required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--epochs',type=int,default=450);a=ap.parse_args();run_task(a.group,a.fold,a.device,a.epochs)
