#!/usr/bin/env python3
import csv,glob,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; RT=HERE/'.runtime'; RES=HERE/'results'
rows=[]
for p in sorted(RT.glob('tasks/E*_f*.json')):
    r=json.loads(p.read_text())
    if r.get('status')!='COMPLETE': continue
    d=r.get('diagnostics',{}); bt=d.get('best_epoch_trace') or {}; ps=(bt.get('pair_stats') or {}).get('stats') or {}
    sc=bt.get('sca_diagnostics') or {}; cr=bt.get('cra') or {}
    auc=bt.get('pair_stats',{}).get('auc',[.5]*4)
    rows.append({
      'group_id':r['group_id'],'group':r['group'],'fold':r['fold'],'best_epoch':r['best_epoch'],
      'alpha_best':d.get('alpha_best'),'residual_ratio':d.get('relative_residual_norm'),
      'srd_auc_pheno':auc[0],'srd_auc_anat':auc[1],'srd_auc_func':auc[2],'srd_auc_corr':auc[3],
      'srd_global_pairs':ps.get('global_pairs',0),'srd_candidate_pairs':ps.get('candidate_pairs_used',0),
      'srd_candidate_native_pairs':ps.get('candidate_native_pairs',0),'srd_candidate_fallback_pairs':ps.get('candidate_global_fallback_pairs',0),
      'srd_candidate_patient_coverage':ps.get('candidate_patient_coverage',0),'srd_candidate_pair_coverage_rate':ps.get('candidate_pair_coverage_rate',0),
      'srd_category_counts':json.dumps(ps.get('candidate_category_counts',[0,0,0,0])),
      'srd_top16_fit_edge_auc':ps.get('top16_fit_edge_auc',.5),'srd_label_flip_rate':ps.get('srd_label_flip_rate',0.),
      'sca_valid_pairs':sc.get('valid_pairs',0),'sca_teacher_pairs':sc.get('teacher_pairs',0),'sca_rejected_pairs':sc.get('rejected',0),
      'sca_teacher_active':int(sc.get('teacher_pairs',0)>0),'cra_strategy':cr.get('strategy','none'),
      'cra_boundary_targets':cr.get('boundary_targets',0),'cra_random_targets':cr.get('random_targets',0),
      'cra_target_asd':cr.get('target_asd',0),'cra_target_td':cr.get('target_td',0),
      'cra_clean_target_nll':cr.get('clean_target_nll',0.),'cra_adv_target_nll':cr.get('adv_target_nll',0.),
      'cra_target_ce_gain':cr.get('adv_target_nll',0.)-cr.get('clean_target_nll',0.),
      'cra_changed_edges':cr.get('changed_edges',0),'cra_changed_edge_rate':cr.get('changed_edge_rate',0.),'cra_kl_target':cr.get('kl_target',0.),
      'message_norms':json.dumps(d.get('message_norms',[])),'gate_means':json.dumps(d.get('gate_means',[])),
    })
RES.mkdir(parents=True,exist_ok=True)
fields=list(rows[0]) if rows else []
with open(RES/'mechanism_diagnostics.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print(json.dumps({'rows':len(rows)}))
