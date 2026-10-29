#!/usr/bin/env python3
"""Engineering semantic checks for Experiment 51.2."""
import json, os, sys
from pathlib import Path
os.environ.setdefault('PDC_EXP', str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import run

def main():
    run.prepare_fold(1); c=run.runtime(); x,y=run.load_data()[0]
    assert x.shape==(871,320) and (y==0).sum()==403 and (y==1).sum()==468
    e1=run.build_state('E1',1); e6=run.build_state('E6',1); fd=run.load_fold(1)
    assert e1["initial_rng_sha"]==e6["initial_rng_sha"] and e1["initial_parameter_sha"]==e6["initial_parameter_sha"]
    o=run.forward_state(e6,fd,False); d=o['details']; assert o['fused'].shape==(871,320)
    assert len(d['g'])==4 and len(d['s_graph'])==4 and len(d['c'])==4
    assert all(g.shape==(871,871) for g in d['g'])
    assert all(torch.isfinite(g).all() for g in d['g'])
    # Pair scores are symmetric and unsupported edges have zero selection weight.
    for lg,g,a in zip(d['logits'],d['g'],d['adjs']):
        assert torch.allclose(lg,lg.t(),atol=1e-5,rtol=1e-5)
        assert float((g[a<=0]).abs().max())==0.0
        assert float((g-g.t()).abs().max())<1e-5
    # Graph/self branches use shared GCN parameters and differ on supported graphs.
    assert max(float((d['graph_states'][m]-d['self_states'][m]).norm()) for m in range(4))>1e-5
    # New residual is finite, nonzero, and output remains 320D.
    residual=torch.cat(d['residuals'],1); assert residual.shape==(871,320) and torch.isfinite(residual).all() and float(residual.norm())>0
    # Pair and alignment gradients are reachable.
    e6['opt_base'].zero_grad(); e6['opt_new'].zero_grad(); e6['opt_cam'].zero_grad(); t=run.forward_state(e6,fd,True); t['total'].backward()
    assert run.grad_norm(e6['model'])>0 and run.grad_norm(e6['cam'])>0
    assert any(p.grad is not None and float(p.grad.abs().sum())>0 for p in e6['model'].selectors.parameters())
    # Test labels do not affect p-conditioned forward.
    p1=run.forward_state(e6,fd,False)['prob'].detach(); old=e6['targ'].clone(); te=torch.as_tensor(e6['test'],device='cuda'); e6['targ'][te]=1-e6['targ'][te]; p2=run.forward_state(e6,fd,False)['prob'].detach(); e6['targ'].copy_(old); assert torch.equal(p1,p2)
    # E7 uses support-uniform weights; E8 changes geometry correspondence; E9 changes only added graph branch.
    e7=run.build_state('E7',1); o7=run.forward_state(e7,fd,False); assert o7['details']['g'][0].shape==(871,871)
    e8=run.build_state('E8',1); o8=run.forward_state(e8,fd,False); assert abs(float(o8['align_raw'])-float(o7['align_raw']))>1e-10
    e9=run.build_state('E9',1); o9=run.forward_state(e9,fd,False); assert torch.equal(o9['details']['base_fused'],o['details']['base_fused']); assert float((o9['fused']-o['fused']).norm())>1e-6
    result={'status':'PASS','shape':list(x.shape),'label_counts':[int((y==0).sum()),int((y==1).sum())],'initial_rng_sha':e1["initial_rng_sha"],'initial_parameter_sha':e1["initial_parameter_sha"],'max_graph_self_delta':max(float((d['graph_states'][m]-d['self_states'][m]).norm()) for m in range(4)),'residual_norm':float(residual.norm()),'pair_raw':float(o['pair_raw']),'align_raw':float(o['align_raw']),'test_label_isolation':True,'alignment_correspondence_changed':True,'wrong_graph_changed':True}
    Path(run.RT/'semantic_tests.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
