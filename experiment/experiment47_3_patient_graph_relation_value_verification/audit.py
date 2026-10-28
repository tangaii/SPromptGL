"""Read-only relation-control effectiveness audit and final paired analysis.

The first part of this file is deliberately label-free: D2/D4/D5 are judged
from graph operators and propagated messages before any new training is
considered.  Labels are used only for post-hoc diagnostic summaries.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import integration as I

MODS = I.MODALITIES
EFFECTIVE = (1, 2, 3)


def _norm_adj(a):
    u = I.modules()['utils']
    eye = torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    return u.normalize_adj(a + eye)


def _offdiag(a):
    n = a.shape[0]
    return a.clone().fill_diagonal_(0)


def _row_cos(a, b):
    aa = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    bb = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    return (aa * bb).sum(1)


def _part_stats(v, idx):
    if idx is None:
        q = v
    else:
        q = v[idx]
    return float(q.norm()), float(q.std()), float(q.abs().mean())


def _graph_stats(a, b, sites, tr, te):
    an, bn = _norm_adj(a), _norm_adj(b)
    aoff, boff = _offdiag(a), _offdiag(b)
    support = (aoff > 0) | (boff > 0)
    changed = ((aoff > 0) != (boff > 0))
    denom = max(1, int((aoff > 0).sum().item()))
    return {
        'operator_distance': float((an - bn).norm() / (an.norm() + 1e-8)),
        'operator_abs': float((an - bn).norm()),
        'support_change': float(changed.sum().item() / denom),
        'row_cos_mean': float(_row_cos(an, bn).mean()),
        'row_cos_p05': float(torch.quantile(_row_cos(an, bn), .05)),
        'weight_fro_distance': float((a-b).norm()/(a.norm()+1e-8)),
        'degree_abs_mean': float((aoff.sum(1)-boff.sum(1)).abs().mean()),
        'degree_corr': float(np.corrcoef(aoff.sum(1).cpu().numpy(), boff.sum(1).cpu().numpy())[0,1]),
        'edge_count_true': int((aoff>0).sum().item()),
        'edge_count_alt': int((boff>0).sum().item()),
    }


def _message_rows(M0, M1, tr, te, label):
    # Centre each candidate according to its own training-fold mean.  This is
    # the same rule used by the fixed-graph training path; it avoids a hidden
    # test-set calibration.
    out=[]
    mu0=M0[tr].mean(0); mu1=M1[tr].mean(0)
    V0=M0-mu0; V1=M1-mu1
    for part,idx in [('all',None),('train',tr),('test',te)]:
        q0=V0 if idx is None else V0[idx]; q1=V1 if idx is None else V1[idx]
        den=float(q0.norm())
        d=float((q1-q0).norm()/(den+1e-8))
        absd=float((q1-q0).norm())
        row={
            'control':label,'modality':'AGGREGATE','partition':part,
            'message_distance':d,'message_abs_distance':absd,
            'true_centered_norm':float(q0.norm()),
            'alt_centered_norm':float(q1.norm()),
            'true_patient_sd':float(q0.std()),
            'alt_patient_sd':float(q1.std()),
            'true_mean_norm':float(mu0.norm()),'alt_mean_norm':float(mu1.norm()),
            'message_row_abs_mean':float((q1-q0).norm(dim=1).mean()),
            'message_row_abs_p95':float(torch.quantile((q1-q0).norm(dim=1),.95)),
        }
        out.append(row)
    return out


def _posthoc_same_class(a, y, sites):
    # Weighted support is evaluated only after graph construction and is not
    # used for any control.  It is included to distinguish structural and
    # diagnostic effects in the report.
    out=[]
    for i,A in enumerate(a):
        B=_offdiag(A); nz=B>0
        den=B.sum(1)+1e-8
        same=(torch.as_tensor(y,device=B.device)[:,None]==torch.as_tensor(y,device=B.device)[None,:]).float()
        out.append({'modality':MODS[i], 'weighted_same_class':float((B*same).sum()/B.sum().clamp_min(1e-8)),
                    'weighted_same_site':float((B*(torch.as_tensor(sites,device=B.device)[:,None]==torch.as_tensor(sites,device=B.device)[None,:]).float()).sum()/B.sum().clamp_min(1e-8)),
                    'mean_same_class_row':float(((B*same).sum(1)/den).mean()),
                    'isolated':int((B.sum(1)==0).sum())})
    return out


def audit_fold(fold, device='cuda:0'):
    x,y,folds,_=I.data(); tr,te=folds[fold-1]
    a,s=I.load_anchor_agent(fold,device,fixed=True); a.fold=fold
    q=torch.load(I.RT/'checkpoints'/f'D1_f{fold:02}.best.pt',map_location='cpu',weights_only=False)
    I.load_payload(a,q['model']); a.gfr.eval(); a.cam.eval()
    c=I.load_controls(fold,device)
    # D1 is the fixed true relation model used for all null audits.
    with torch.no_grad():
        ps,raw=I.raw_parts(a)
        bs_true=I.branch_outputs(a,c['true'],raw)
        bs_re=I.branch_outputs(a,torch.load(I.RT/'fixed_graphs'/f'fold{fold:02}.pt',map_location=device,weights_only=False)['rewire'],raw)
        bs_d4=I.branch_outputs(a,c['D4'],raw)
        M_true=sum((bs_true[i]['scaled'] for i in EFFECTIVE))
        M_re=sum((bs_re[i]['scaled'] for i in EFFECTIVE))
        M_d4=sum((bs_d4[i]['scaled'] for i in EFFECTIVE))
        M_d5=M_true[c['perm']]
        rows_graph=[]; rows_msg=[]
        sites=c['sites'];
        adj_sets={'D2':(c['true'],torch.load(I.RT/'fixed_graphs'/f'fold{fold:02}.pt',map_location=device,weights_only=False)['rewire']),
                  'D4':(c['true'],c['D4'])}
        for label,(aa,bb) in adj_sets.items():
            for i in EFFECTIVE:
                r=_graph_stats(aa[i],bb[i],sites,tr,te);r.update({'fold':fold,'control':label,'modality':MODS[i]});rows_graph.append(r)
        # D5 does not change a graph operator; it changes the receiver-message
        # correspondence while preserving the complete message marginal.
        for i in EFFECTIVE:
            Mi_true=bs_true[i]['scaled']; Mi_re_i=bs_re[i]['scaled']; Mi_d4=bs_d4[i]['scaled']; Mi_d5=Mi_true[c['perm']]
            for label,Mi in [('D2',Mi_re_i),('D4',Mi_d4),('D5',Mi_d5)]:
                rows_msg.extend([{**z,'fold':fold,'modality':MODS[i]} for z in _message_rows(Mi_true,Mi,tr,te,label)])
        # Explicit aggregate rows for the D2/D4/D5 effective-null gate.
        for label,Malt in [('D2',M_re),('D4',M_d4),('D5',M_d5)]:
            for z in _message_rows(M_true,Malt,tr,te,label):
                z.update({'fold':fold,'modality':'AGGREGATE'}); rows_msg.append(z)
        # Post-hoc graph diagnostics.
        for label,adjs in [('TRUE',c['true']),('D4',c['D4'])]:
            for z in _posthoc_same_class(adjs,y,sites): z.update({'fold':fold,'control':label}); rows_graph.append(z)
    return rows_graph,rows_msg


def run_effectiveness():
    gs=[];ms=[]
    for f in range(1,11):
        g,m=audit_fold(f,'cuda:0');gs.extend(g);ms.extend(m);print('audited fold',f,flush=True)
    pd.DataFrame(gs).to_csv(I.RESULTS/'original_D2_effectiveness.csv',index=False)
    pd.DataFrame(ms).to_csv(I.RESULTS/'strong_null_effectiveness.csv',index=False)
    return pd.DataFrame(gs),pd.DataFrame(ms)


def gates(ms=None):
    if ms is None: ms=pd.read_csv(I.RESULTS/'strong_null_effectiveness.csv')
    # One aggregate row per fold/control/partition is retained; the gate uses
    # the training partition, with all/test values reported separately.
    q=ms[(ms.modality=='AGGREGATE')&(ms.partition=='train')].copy()
    rows=[]; result={}
    for control in ('D4','D5'):
        z=q[q.control==control].drop_duplicates(['fold','control'])
        valid=z[np.isfinite(z.message_distance)&(z.message_distance>=.25)]
        no_nan=bool(np.isfinite(z[['message_distance','message_abs_distance']].to_numpy()).all())
        ok=int(len(valid))>=8 and no_nan
        result[control]={'valid_folds':int(len(valid)),'total_folds':int(len(z)),'threshold':.25,'no_nan':no_nan,'pass':ok,
                         'dist_mean':float(z.message_distance.mean()),'dist_median':float(z.message_distance.median()),
                         'abs_mean':float(z.message_abs_distance.mean())}
        rows.append({'control':control,**result[control]})
    out={'D4':result['D4'],'D5':result['D5'],'pass_all':bool(result['D4']['pass'] and result['D5']['pass'])}
    (I.RT/'null_gates.json').write_text(json.dumps(out,indent=2))
    pd.DataFrame(rows).to_csv(I.RESULTS/'null_gates.csv',index=False)
    print(json.dumps(out,indent=2));return out


def main():
    _,ms=run_effectiveness();gates(ms)

if __name__=='__main__': main()
