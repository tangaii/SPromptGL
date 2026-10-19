"""Parameter-free, class-identity-preserving prompt evidence from ORIGINAL PLN tokens.

Label convention: ASD=0, TD=1; original P0 first 4 sentences are
NORMAL/TD and last 4 are ALTERED/ASD. Only the prompt/text branch changes;
GFR, CAM, RES, original SPGS max and classifier remain unchanged.
"""
import torch
import torch.nn.functional as F
from e26_contract import GAMMA,LAMBDA_AUX,CONF_CENTER,CONF_TAU

def evidence(token_norm,prototype,selective=True,fc_only=False):
    assert token_norm.ndim==4 and token_norm.shape[1:]==(4,4,9)
    assert prototype.shape==(2,9)
    # Direction preserves the class identity (ASD=altered minus TD=normal).
    td,asd=F.normalize(prototype,dim=-1,eps=1e-8)
    delta=asd-td
    delta=F.normalize(delta,dim=0,eps=1e-5)
    signed=(token_norm*delta.view(1,1,1,9)).sum(-1)
    per_modal=signed.mean(dim=-1) # N,4 signed class evidence
    if selective:
        conf=torch.sigmoid((per_modal.detach().abs()-CONF_CENTER)/CONF_TAU)
    else:conf=torch.ones_like(per_modal)
    selected=per_modal*conf
    # Same scale in full vs FC-only: zero out other slots, don't renormalize.
    if fc_only:
        m=torch.zeros_like(selected)
        m[:,3]=selected[:,3]
        selected=m
    score=selected.sum(-1)/4.
    logits=torch.stack((score,-score),dim=-1)
    return logits,dict(score=score,by_modality=per_modal,confidence=conf,
                       prototype_cos=F.cosine_similarity(td[None],asd[None],dim=-1)[0],
                       prompt_distance=torch.linalg.vector_norm(asd-td))

def apply(base_log_probs,ev_logits,readout=True):
    assert base_log_probs.shape==ev_logits.shape
    if not readout:return base_log_probs
    return F.log_softmax(base_log_probs+GAMMA*ev_logits,dim=-1)

def aux_loss(ev_logits,targets,train,auxiliary=True):
    if not auxiliary:return ev_logits.sum()*0.
    return LAMBDA_AUX*F.cross_entropy(ev_logits[train],targets[train])

def self_test():
    torch.manual_seed(26)
    part=torch.randn(13,4,4,9,requires_grad=True)
    norm=F.normalize(part,dim=-1)
    prototype=torch.randn(2,9,requires_grad=True)
    for selective in (True,False):
        for fc_only in (True,False):
            e,d=evidence(norm,prototype,selective,fc_only)
            assert e.shape==(13,2) and bool(torch.isfinite(e).all())
            assert torch.allclose(e.sum(-1),torch.zeros(13),atol=1e-7)
            if fc_only:
                assert d['by_modality'].shape==(13,4)
                # Output ignores other modalities (gradient zero on non-FC tokens).
                g=torch.autograd.grad(e[:,0].sum(),part,retain_graph=True)[0]
                assert torch.count_nonzero(g[:,:3])==0
            base=torch.log_softmax(torch.randn(13,2),dim=-1)
            assert torch.equal(apply(base,e,False),base)
            assert bool(torch.isfinite(apply(base,e,True)).all())
            y=torch.randint(0,2,(13,))
            loss=aux_loss(e,y,torch.arange(8))
            assert loss.isfinite().all()
            # Swap class labels in the prototypes: new evidence must reverse sign.
            e2,_=evidence(norm,prototype.flip(0),selective,fc_only)
            assert torch.allclose(e,-e2,rtol=1e-5,atol=1e-5)
    # Original SPGS max is invariant to label order by construction.
    logits=torch.randn(17,2)
    p=torch.softmax(logits,dim=-1).max(-1)[0]
    q=torch.softmax(logits.flip(-1),dim=-1).max(-1)[0]
    assert torch.equal(p,q)
    print('E26_EVIDENCE_SELF_TEST_PASS')
if __name__=='__main__':self_test()
