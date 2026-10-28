"""Train only the pre-registered D4/D5 fixed-graph controls."""
from __future__ import annotations
import argparse, csv, hashlib, time
from pathlib import Path
import torch
import integration as I

RT,RES=I.RT,I.RESULTS
(RT/'checkpoints').mkdir(exist_ok=True); (RT/'tasks').mkdir(exist_ok=True)
SAMPLE={0,1,2,5,10,20,50,100,150,200}

def _grad_summary(a):
    out={}
    for i,mod in enumerate(I.MODALITIES):
        for nm,p in [('W1',getattr(a.gfr,f'MessagePassing{i}').gcn1.W),('W2',getattr(a.gfr,f'MessagePassing{i}').gcn2.W)]:
            out[f'{mod}_{nm}_grad']=0. if p.grad is None else float(p.grad.detach().norm().cpu())
    c=sum((p.grad.detach().square().sum() for p in a.cam.parameters() if p.grad is not None),torch.zeros((),device=a.x.device))
    out['cam_grad']=float(c.sqrt().cpu()); return out

def _eval(a,group,c):
    wasg,wasc=a.gfr.training,a.cam.training; a.gfr.eval();a.cam.eval()
    with torch.no_grad(): out=I.scaled_forward(a,group,c)
    if wasg:a.gfr.train()
    if wasc:a.cam.train()
    return out

def _row(group,fold,epoch,out,tm,vm,cl,grad):
    lp,hid,_,gl,_,aux=out
    return {'group':group,'fold':fold,'epoch':epoch,
            'train_nll':tm['nll'],'train_acc':tm['acc'],'test_nll':vm['nll'],'test_acc':vm['acc'],
            'train_hard_auc':tm['hard_auc'],'test_hard_auc':vm['hard_auc'],
            'train_spe':tm['spe'],'test_spe':vm['spe'],'train_sen':tm['sen'],'test_sen':vm['sen'],
            'classification_loss':float(cl.detach().cpu()),'graph_loss':0.,
            'raw_norm':float(aux['raw_norm'].detach().cpu()),'M_norm':float(aux['M_norm'].detach().cpu()),
            'Mc_norm':float(aux['Mc_norm'].detach().cpu()),'scale':float(aux['scale'].detach().cpu()),
            'Z_norm':float(aux['Z_norm'].detach().cpu()),'Xc_train_norm':float(aux['Xc_train_norm'].detach().cpu()),
            'Mc_train_norm':float(aux['Mc_train_norm'].detach().cpu()),'M_mean_norm':float(aux['train_M_mean_norm'].detach().cpu()),
            'patient_M_sd':float(aux['patient_var'].detach().cpu()),'M_over_X':float((aux['M_norm']/(aux['raw_norm']+1e-12)).detach().cpu()),
            'Mc_over_X':float((aux['Mc_norm']/(aux['raw_norm']+1e-12)).detach().cpu()),
            'cam_hidden_norm':float(hid.detach().norm().cpu()),**grad}

def _write(p,rows):
    if not rows:return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def run_task(group,fold,device='cuda:0',max_epoch=450):
    if group not in ('D4','D5'): raise ValueError(group)
    I.seed_for_task(0)
    a,anchor=I.load_anchor_agent(fold,device,fixed=True);a.fold=fold
    c=I.load_controls(fold,device)
    # D4/D5 share the exact fresh-anchor model and optimizer construction used
    # by the fixed-graph D1/D2/D3 controls.  GraphLearn is frozen and never
    # enters the forward path for these groups.
    a.opt_gfr=I.make_optimizer(a)
    a.opt_cam=torch.optim.Adam(a.cam.parameters(),lr=a.hp.lr,weight_decay=a.hp.reg)
    I.restore(anchor['rng'])
    start=time.monotonic();best_acc=-1.;best_epoch=-1;wait=0;best=None;best_vm=None;best_tm=None;traj=[]
    for epoch in range(max_epoch):
        a.gfr.train();a.cam.train();a.opt_gfr.zero_grad(set_to_none=True);a.opt_cam.zero_grad(set_to_none=True)
        lp,hid,cl,gl,adjs,aux=I.scaled_forward(a,group,c)
        cl.backward();grad=_grad_summary(a);a.opt_gfr.step();a.opt_cam.step()
        out=_eval(a,group,c);elp,eh,_,egl,_,eaux=out;tm,vm=I.train_metrics(elp,a)
        if vm['acc']>best_acc:
            best_acc=vm['acc'];best_epoch=epoch;wait=0;best=I.state_payload(a);best_vm=vm;best_tm=tm
        else: wait+=1
        if epoch in SAMPLE or epoch%10==0:
            traj.append(_row(group,fold,epoch,out,tm,vm,cl,grad))
        if wait>200: break
    if best is None: raise RuntimeError('no best checkpoint')
    I.load_payload(a,best);out=_eval(a,group,c);elp,eh,_,egl,_,eaux=out;tm,vm=I.train_metrics(elp,a)
    payload={'group':group,'fold':fold,'device':device,'best_epoch':best_epoch,'epochs_executed':epoch+1,
             'best_acc':best_acc,'model':best,'metrics':vm,'train_metrics':tm,
             'graph_sha':I.graph_sha(c['D4'] if group=='D4' else c['true']),
             'anchor_sha':anchor.get('parameter_sha'),'rng':I.rng(),
             'optimizer_gfr':a.opt_gfr.state_dict(),'optimizer_cam':a.opt_cam.state_dict(),'trajectory':traj}
    outp=RT/'checkpoints'/f'{group}_f{fold:02}.best.pt';torch.save(payload,outp)
    row={'group':group,'fold':fold,'seed':0,'best_epoch':best_epoch,'epochs_executed':epoch+1,
         'acc':vm['acc'],'hard_auc':vm['hard_auc'],'spe':vm['spe'],'sen':vm['sen'],
         'asd_tp':vm['asd_tp'],'asd_fn':vm['asd_fn'],'td_fp':vm['td_fp'],'td_tn':vm['td_tn'],
         'train_acc':tm['acc'],'train_nll':tm['nll'],'test_nll':vm['nll'],
         'best_sha':hashlib.sha256(torch.cat([q.detach().cpu().flatten() for q in best['cam'].values()]).numpy().tobytes()).hexdigest(),
         'wall_seconds':time.monotonic()-start,'status':'COMPLETE','device':device,'rho':I.rho,
         'graph_sha':payload['graph_sha']}
    pref=RES/f'{group.lower()}_f{fold:02}';_write(pref.with_suffix('.trajectory.csv'),traj);_write(pref.with_suffix('.metrics.csv'),[row])
    print('TASK_DONE',group,fold,row['acc'],row['train_nll'],best_epoch,flush=True);return row

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['D4','D5'],required=True);ap.add_argument('--fold',type=int,required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--epochs',type=int,default=450)
    run_task(ap.parse_args().group,ap.parse_args().fold,ap.parse_args().device,ap.parse_args().epochs)
