"""Compare the old 47.2 D1 path with the current read-only D1 path."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

EXP=Path(__file__).resolve().parent
OLD=Path('/tmp/SPromptGL-E47/experiment47_2_active_graph_failure_attribution')

OLD_CODE=r'''
import json, hashlib
import integration as I
fold=1; device='cuda:0'; I.seed_for_task(0)
a,s=I.load_anchor_agent(fold,device,fixed=True); a.fold=fold
a.gfr.train(); a.cam.train(); c=I.fixed_true_graphs(fold,device)
lp,h,cl,gl,adjs,aux=I.scaled_forward(a,'D1',c['true']); cl.backward()
def sh(m):
 h=hashlib.sha256()
 for n,p in sorted(m.state_dict().items()): h.update(n.encode()+p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def gn(m): return {n:float(p.grad.detach().norm().cpu()) for n,p in m.named_parameters() if p.grad is not None}
print(json.dumps({'loss':float(cl.detach().cpu()),'z_norm':float(aux['Z_norm']),'m_norm':float(aux['M_norm']),'grad':gn(a.gfr),'cam_grad':gn(a.cam),'before_gfr':sh(a.gfr),'before_cam':sh(a.cam)}))
a.opt_gfr.step(); a.opt_cam.step(); print(json.dumps({'after_gfr':sh(a.gfr),'after_cam':sh(a.cam)}))
'''

NEW_CODE=r'''
import json, hashlib
import integration as I
fold=1; device='cuda:0'; I.seed_for_task(0)
a,s=I.load_anchor_agent(fold,device,fixed=True); a.fold=fold
a.gfr.train(); a.cam.train(); c=I.load_controls(fold,device)
lp,h,cl,gl,adjs,aux=I.scaled_forward(a,'D1',c); cl.backward()
def sh(m):
 h=hashlib.sha256()
 for n,p in sorted(m.state_dict().items()): h.update(n.encode()+p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def gn(m): return {n:float(p.grad.detach().norm().cpu()) for n,p in m.named_parameters() if p.grad is not None}
print(json.dumps({'loss':float(cl.detach().cpu()),'z_norm':float(aux['Z_norm']),'m_norm':float(aux['M_norm']),'grad':gn(a.gfr),'cam_grad':gn(a.cam),'before_gfr':sh(a.gfr),'before_cam':sh(a.cam)}))
a.opt_gfr.step(); a.opt_cam.step(); print(json.dumps({'after_gfr':sh(a.gfr),'after_cam':sh(a.cam)}))
'''

def _run(code, cwd):
    out=subprocess.check_output([sys.executable,'-c',code],cwd=str(cwd),text=True,stderr=subprocess.STDOUT)
    lines=[x for x in out.splitlines() if x.startswith('{')]
    return json.loads(lines[-2]),json.loads(lines[-1]),out

def main():
    old0,old1,oldout=_run(OLD_CODE,OLD); new0,new1,newout=_run(NEW_CODE,EXP)
    vals=[]
    for k in ('loss','z_norm','m_norm'): vals.append(abs(float(old0[k])-float(new0[k])))
    for k in ('grad','cam_grad'):
        keys=set(old0[k])|set(new0[k]); vals.extend(abs(float(old0[k].get(x,0))-float(new0[k].get(x,0))) for x in keys)
    result={'old':old0,'new':new0,'old_step':old1,'new_step':new1,
            'max_forward_backward_abs_diff':max(vals),
            'state_digest_equal':bool(old1==new1 and old0['before_gfr']==new0['before_gfr'] and old0['before_cam']==new0['before_cam']),
            'old_stdout_tail':oldout[-1000:],'new_stdout_tail':newout[-1000:]}
    (EXP/'.runtime'/'parity_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('old','new','old_step','new_step','old_stdout_tail','new_stdout_tail')},indent=2))

if __name__=='__main__': main()
