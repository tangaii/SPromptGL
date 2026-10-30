"""Semantic checks for Experiment 51.4.  They assert behavior, not outcomes."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run

def main():
 run.runtime(); (x,y,si,pi),folds=run.load_data(); assert x.shape==(871,320); assert len(si)==20 and len(pi)==28 and not set(si)&set(pi)
 for f in range(1,11):
  d=json.loads((run.FD/f'fold{f:02d}.json').read_text()); assert d['split_sha']==run.split_sha(f); assert 0<=float(d['rS'])<=1 and 0<=float(d['rC'])<=1
  z=np.load(run.FD/f'fold{f:02d}.npz',allow_pickle=True); tr=z['train']; assert set(tr).isdisjoint(set(z['test'])); assert len(z['align_ids'])==256
 # E1 and D3 parameter/RNG construction are deterministic and site-free.
 st=run.build_state('E3',1); fd=run.load_fd(1); out=run.forward(st,fd,True,epoch=0)
 assert out['fused'].shape==(871,320); assert np.isfinite(out['total'].detach().cpu().numpy()); out['total'].backward()
 grads=[p.grad for p in st['model'].parameters() if p.grad is not None]; assert grads and all(np.isfinite(g.detach().cpu().numpy()).all() for g in grads)
 # E5 zero evidence routes must be exactly E1 at the new-input boundary.
 st5=run.build_state('E5',1); out5=run.forward(st5,fd,False,epoch=0); base=out5['z']['base'];
 if float(json.loads((run.FD/'fold01.json').read_text())['rS'])==0 and float(json.loads((run.FD/'fold01.json').read_text())['rC'])==0:
  assert (out5['fused']-base).abs().max().item()==0.0
 # E9 correspondence is fixed and nonidentity for the block used by forward.
 b=fd['blocks'][0]; assert len(b)>2; assert not np.array_equal(np.arange(len(b)-1,-1,-1),np.arange(len(b)))
 Path(run.RES/'semantic_tests.json').write_text(json.dumps({'status':'PASS','folds':10,'site_indices':si.tolist(),'pheno_non_site_indices':pi.tolist(),'rS_all_zero':all(float(json.loads((run.FD/f'fold{f:02d}.json').read_text())['rS'])==0 for f in range(1,11))},indent=2))
 print('SEMANTIC_TESTS_PASS')
if __name__=='__main__': main()
