"""Independent audits, gate evaluation, and fixed-checkpoint interventions."""
from __future__ import annotations
import argparse,csv,glob,json,math
from pathlib import Path
import numpy as np, pandas as pd, torch
import integration as I

def _load(group,fold,device):
    if group in ('D1','D2','D3'):
        fixed=I.fixed_true_graphs(fold,'cpu')[{'D1':'true','D2':'rewire','D3':'self'}[group]]
    else:
        fixed=None
    a,anchor=I.load_anchor_agent(fold,device,fixed=group in ('D1','D2','D3'))
    q=torch.load(I.RT/'checkpoints'/f'{group}_f{fold:02}.best.pt',map_location='cpu',weights_only=False);I.load_payload(a,q['model']);a.gfr.eval();a.cam.eval()
    return a,fixed,q

def _forward(a,group,mode,fixed):
    ps,raw=I.raw_parts(a)
    if mode=='strict_zero':
        lp,h=a.cam(raw)
        return lp,raw,raw.new_zeros(())
    # The graph replacement modes are deliberate fixed-model interventions.
    adjs=fixed
    if mode=='true_graph': adjs=I.fixed_true_graphs(a.fold if hasattr(a,'fold') else 1,'cpu')['true']
    if mode=='rewire_graph': adjs=I.fixed_true_graphs(a.fold if hasattr(a,'fold') else 1,'cpu')['rewire']
    if mode=='self_loop': adjs=I.fixed_true_graphs(a.fold if hasattr(a,'fold') else 1,'cpu')['self']
    if mode=='original':
        lp,h,_,_,adjs,aux=I.scaled_forward(a,group,fixed_adjs=fixed,compute_cam=True)
        return lp,aux['Z'],aux['M']
    if adjs is None:
        adjs=[getattr(a.gfr,f'GraphConstruct{i}')(ps[i]) for i in range(4)]
    else:
        adjs=[q.to(raw.device) for q in adjs]
    bs=I.branch_outputs(a,adjs,raw)
    drop={'drop_anat':1,'drop_func':2,'drop_corr':3}.get(mode)
    M=raw.new_zeros(raw.shape)
    for i in [1,2,3]:
        if i!=drop:M=M+bs[i]['scaled']
    tr=torch.as_tensor(a.train_idx,device=raw.device);mu=M[tr].mean(0).detach();xc=raw[tr]-raw[tr].mean(0).detach();mc=M[tr]-mu;scale=(torch.tensor(I.rho,device=raw.device)*xc.norm()/(mc.norm()+I.eps)).detach();z=raw+scale*(M-mu)
    if mode=='center_off': z=raw+scale*M
    lp,h=a.cam(z)
    return lp,z,M

def run_fold(fold,groups,device):
    rows=[]
    for group in groups:
        a,fixed,q=_load(group,fold,device);a.fold=fold
        modes=['original','strict_zero','self_loop','true_graph','rewire_graph','drop_anat','drop_func','drop_corr','center_off']
        base=None
        with torch.no_grad():
            for mode in modes:
                lp,z,M=_forward(a,group,mode,fixed)
                tr,te=I.train_metrics(lp,a)
                if base is None: base=lp.detach().clone()
                p=te['pred']; flips=int((lp.detach().argmax(1)!=base.argmax(1)).cpu().numpy()[a.test_idx].sum())
                delta=(lp.detach()-base).abs().mean().item() if base is not None else 0.
                rows.append({'group':group,'fold':fold,'intervention':mode,'acc':te['acc'],'hard_auc':te['hard_auc'],'spe':te['spe'],'sen':te['sen'],'train_acc':tr['acc'],'train_nll':tr['nll'],'test_nll':te['nll'],'logprob_abs_delta':delta,'test_flips':flips,'Z_norm':float(z.norm()),'M_norm':float(M.norm())})
    p=I.RESULTS/f'interventions_f{fold:02}.csv';
    with p.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows

def merge():
    ps=sorted(glob.glob(str(I.RESULTS/'interventions_f*.csv')))
    if ps:
        d=pd.concat([pd.read_csv(p) for p in ps],ignore_index=True);d.to_csv(I.RESULTS/'interventions.csv',index=False);return d

def gates():
    # Use only training-side epoch-100 values; no external test metric enters.
    hist=pd.read_csv(I.RT/'history47_1'/'training_trajectory.csv')
    t1=hist[(hist.group=='T1')&(hist.epoch==100)].set_index('fold')['cls_loss']
    out={}
    for g in ('D0','D1','D2','D3'):
        vals=[]
        for f in (1,6):
            p=I.RESULTS/f'pilot_{g}_f{f:02}.trajectory.csv';d=pd.read_csv(p);r=d[d.epoch==100].iloc[0]
            ratio=float(r.scale*r.Mc_train_norm/r.Xc_train_norm)
            vals.append({'fold':f,'train_nll':float(r.train_nll),'t1_nll':float(t1.loc[f]),'relative':float(r.train_nll/t1.loc[f]),'ratio':ratio,'valid':bool(np.isfinite(r.train_nll) and abs(ratio-I.rho)<=.1*I.rho)})
        out[g]=vals
    gateA=all(v['valid'] and v['train_nll']<=.8*v['t1_nll'] for v in out['D0'])
    gateB=all(all(v['valid'] and np.isfinite(v['train_nll']) for v in out[g]) for g in ('D1','D2','D3')) and all(v['train_nll']<=.8*v['t1_nll'] for v in out['D1'])
    result={'gate_A':gateA,'gate_B':gateB,'details':out,'expansion':('ALL' if gateA and gateB else 'D0' if gateA else 'D1_D2_D3' if gateB else 'STOP')}
    (I.RT/'gates.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--folds',default='1-10');ap.add_argument('--device',default='cuda:0');ap.add_argument('--intervene',action='store_true');args=ap.parse_args()
    if args.intervene:
        a,b=map(int,args.folds.split('-'))
        for f in range(a,b+1): print('INTERVENE',f,len(run_fold(f,['D0','D1','D2','D3'],args.device)),flush=True)
        merge()
    else: print(json.dumps(gates(),indent=2))
