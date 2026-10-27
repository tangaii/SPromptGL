"""Independent delivery audit for Experiment 44.1."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/data/adodas/SPromptGL-main'); EXP=Path(__file__).resolve().parent; RT=EXP/'.runtime'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 m=json.loads((EXP/'manifest.json').read_text()); checks=[]
 for n,h in m['protected_sha256'].items(): checks.append({'file':n,'before':h,'after':sha(ROOT/n),'pass':h==sha(ROOT/n)})
 assert all(c['pass'] for c in checks)
 pf=m['stageA']; assert pf['status']=='PREFLIGHT_SCIENTIFIC_NO_GO'; assert len(pf['folds'])==2; assert pf['teacher_tasks']==6
 for f in (1,6):
  p=RT/'candidates'/f'fold{f:02}.npz'; z=np.load(p); c=z['candidates']; assert c.shape==(871,4,16); assert np.all(c[:, :, :] >= -1)
  t=RT/'teachers'/f'outer_fold{f:02}.pt'; assert t.exists(); o=__import__('torch').load(t,map_location='cpu',weights_only=False); assert o['candidate_hash']==sha(p)
 out={'status':'PASS','protected':checks,'stageA_status':pf['status'],'teacher_tasks':pf['teacher_tasks'],'formal_new_folds':0,'historical_rows':sum(1 for _ in __import__('csv').DictReader((EXP/'results/fold_metrics.csv').open()))}
 (RT/'delivery_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); m['delivery_audit']=out; (EXP/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
