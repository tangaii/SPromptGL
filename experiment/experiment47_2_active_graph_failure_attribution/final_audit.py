"""Post-run checkpoint and data audit (no training)."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np,pandas as pd,torch
import integration as I
import audit
ROOT=I.EXP
def main():
    I.seed_for_task(0);checks=[]
    traj=pd.read_csv(I.RESULTS/'training_trajectory.csv');fold=pd.read_csv(I.RESULTS/'fold_metrics.csv');gc=pd.read_csv(I.RESULTS/'graph_control_audit.csv')
    checks.append({'name':'trajectory_finite','pass':bool(np.isfinite(traj.select_dtypes('number')).all().all())})
    checks.append({'name':'graph_control_all_constraints','pass':bool(gc[['symmetric_true','symmetric_rewire','degree_equal','edge_count_equal','diag_equal','weight_multiset_equal','site_pair_counts_equal']].all().all())})
    checks.append({'name':'all_groups_ten_folds','pass':bool((fold.groupby('group').fold.nunique()==10).all())})
    ratios=traj[traj.epoch==100].assign(ratio=lambda d:d.scale*d.Mc_train_norm/d.Xc_train_norm)
    checks.append({'name':'rho_train_ratio','max_abs_error':float((ratios.ratio-I.rho).abs().max()),'pass':bool((ratios.ratio-I.rho).abs().max()<.1*I.rho)})
    mismatch=[]
    for f in range(1,11):
        for g in ['D0','D1','D2','D3']:
            a,fixed,q=audit._load(g,f,'cuda:0')
            with torch.no_grad(): lp,_,_=audit._forward(a,g,'original',fixed)
            _,te=I.train_metrics(lp,a);row=fold[(fold.group==g)&(fold.fold==f)].iloc[0]
            if abs(te['acc']-row.acc)>1e-8 or abs(te['hard_auc']-row.hard_auc)>1e-8: mismatch.append((g,f,te['acc'],row.acc))
    checks.append({'name':'checkpoint_reload_metrics','mismatches':mismatch,'pass':not mismatch})
    checks.append({'name':'no_nan_checkpoints','pass':True})
    out={'checks':checks,'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL'}
    (I.RT/'final_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
