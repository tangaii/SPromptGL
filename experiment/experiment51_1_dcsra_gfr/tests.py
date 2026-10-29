#!/usr/bin/env python3
"""Semantic tests for Experiment 51.1. These are engineering checks only."""
import json, os, sys
from pathlib import Path
os.environ.setdefault("DCSRA_EXP", str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import run


def main():
    run.prepare_fold(1)
    c=run.runtime(); x,y=run.load_data()[0]
    assert x.shape==(871,320) and y.shape==(871,) and (y==0).sum()==403 and (y==1).sum()==468
    state=run.build_state("E7",1); fd=run.load_fold(1)
    out=run.forward_state(state,fd,False)
    d=out["details"]
    assert out["fused"].shape==(871,320)
    assert len(d["branches"])==4 and all(b["sg"].shape==(871,8) and b["si"].shape==(871,8) and b["c"].shape==(871,8) for b in d["branches"])
    assert d["gates"].shape==(871,4) and torch.isfinite(out["total"])
    # Graph and self use the same branch parameters, but their operators differ.
    delta=[float((b["hg"]-b["hi"]).norm()) for b in d["branches"]]
    assert max(delta)>1e-6
    # E1 base path remains the 320-dimensional four-graph aggregation.
    assert d["base_fused"].shape==(871,320) and len(d["base_msgs"])==4
    # Non-zero bounded residual at initialization and finite gradients.
    assert float(torch.cat(d["residuals"],1).norm())>0
    state["opt_base"].zero_grad(); state["opt_new"].zero_grad(); state["opt_cam"].zero_grad()
    tr=run.forward_state(state,fd,True); tr["total"].backward()
    grads=[p.grad for p in state["model"].parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in state["model"].branches.parameters())
    # Test labels cannot affect the forward path.
    p1=run.forward_state(state,fd,False)["prob"].detach()
    original=state["targ"].clone(); te=torch.as_tensor(state["test"],device="cuda"); state["targ"][te]=1-state["targ"][te]
    p2=run.forward_state(state,fd,False)["prob"].detach(); state["targ"].copy_(original)
    assert torch.equal(p1,p2)
    # Wrong-new-graph permutation changes the added branch while preserving the E1 base.
    o10=run.forward_state(state,fd,False)
    pstate=dict(state); pstate["group"]="E10"; pstate["mode"]="E10_WRONG_GRAPH"; o10w=run.forward_state(pstate,fd,False)
    assert torch.equal(o10["details"]["base_fused"],o10w["details"]["base_fused"])
    assert float((o10["fused"]-o10w["fused"]).norm())>1e-6
    # Alignment correspondence permutation is nontrivial and changes the geometry term.
    fd9=dict(fd); fd9["align_perm"]=fd["align_perm"]
    s9=dict(state); s9["group"]="E9"; s9["mode"]="E9_WRONG_ALIGN"
    out9=run.forward_state(s9,fd9,False); a1=out9["aux"]["align_raw"]
    # Force identity correspondences and recompute with the same state.
    ident=torch.arange(871,device="cuda").repeat(4,1); fdid=dict(fd); fdid["align_perm"]=ident
    outid=run.forward_state(s9,fdid,False); a0=outid["aux"]["align_raw"]
    assert float((a1-a0).abs())>1e-8
    # E8 changes alignment weights but not gates or graph outputs.
    s8=dict(state); s8["group"]="E8"; s8["mode"]="E8_UNIFORM"; o8=run.forward_state(s8,fd,False)
    assert torch.equal(out["details"]["gates"],o8["details"]["gates"])
    result={"status":"PASS","shape":list(x.shape),"label_counts":[int((y==0).sum()),int((y==1).sum())],"graph_self_delta":delta,"new_residual_norm":float(torch.cat(d["residuals"],1).norm()),"test_label_isolation":True,"wrong_graph_delta":float((o10["fused"]-o10w["fused"]).norm()),"alignment_permutation_delta":float((a1-a0).abs())}
    Path(run.RT/"semantic_tests.json").parent.mkdir(parents=True,exist_ok=True); (run.RT/"semantic_tests.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,ensure_ascii=False))

if __name__=="__main__": main()
