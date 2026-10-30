"""Independent checkpoint readback for all formal folds."""
import json,csv,sys
from pathlib import Path
import numpy as np
import run

def main():
 run.runtime(); rows=[]; failures=[]
 for g in ['E0','E1']+[f'E{i}' for i in range(2,13)]:
  for f in range(1,11):
   try:
    if g in ('E0','E1'):
     continue
    d=run.torch_load(run.CK/f'{g}_f{f:02d}.best.pt'); st=run.build_state(g,f); st['model'].load_state_dict(d['model']); st['cam'].load_state_dict(d['cam']); fd=run.load_fd(f)
    import torch
    with torch.no_grad(): out=run.forward(st,fd,False,epoch=int(d['best_epoch']))
    met=run.metrics(out['lp'],st['targ'].detach().cpu().numpy(),st['test'])
    saved=d['metrics']; err=max(abs(float(met[k])-float(saved[k])) for k in ('acc','hard_auc','prob_auc','spe','sen','ba','asd_f1'))
    rows.append({'group':g,'fold':f,'saved_best_epoch':d['best_epoch'],'readback_max_abs_error':err,'status':'PASS' if err<1e-8 else 'FAIL','checkpoint_sha':run.sha_file(run.CK/f'{g}_f{f:02d}.best.pt')})
    if err>=1e-8: failures.append((g,f,err))
   except Exception as e: failures.append((g,f,repr(e)))
 p=run.RES/'checkpoint_audit.csv'; fields=sorted({k for r in rows for k in r})
 with p.open('w',newline='') as fh:w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows(rows)
 m=json.loads((run.EXP/'manifest.json').read_text());m.update(checkpoint_audit={'rows':len(rows),'passed':len(rows)-len(failures),'failures':failures,'max_error':max([r['readback_max_abs_error'] for r in rows] or [0]),'sha':run.sha_file(p)});run.atomic_json(run.EXP/'manifest.json',m)
 print('AUDIT',len(rows),len(failures))
 if failures: print(failures[:5]); raise SystemExit(1)
if __name__=='__main__': main()
