"""Fold-local readout configuration; fixed prototype caches are created in preflight."""
from pathlib import Path
import numpy as np
import torch
from e26_contract import DESIGNS
from e26_evidence import evidence,apply,aux_loss
_CURRENT=None
_CACHE=None

def configure(config,cache_path):
    global _CURRENT,_CACHE
    assert config in DESIGNS
    _CURRENT=DESIGNS[config]
    if _CACHE is None:
        with np.load(str(cache_path)) as z:
            _CACHE={k:np.asarray(z[k],dtype=np.float32).copy() for k in ('real','unrelated','isometric')}
        for v in _CACHE.values():assert v.shape==(2,4096) and np.isfinite(v).all()

def prototype(original):
    assert _CURRENT is not None
    if _CURRENT.prompt=='real':return original # preserves original network RNG/maths
    return torch.as_tensor(_CACHE[_CURRENT.prompt],device=original.device,dtype=original.dtype)

def signed_project(mlp,proto4096):
    # Reuse EXACT pretrained-text projection parameters, but remove final ReLU
    # only for signed evidence. Original P0/SPGS MLP remains untouched.
    return mlp.t2(torch.relu(mlp.t1(proto4096)))

def forward(pln,base_log_probs,tokens,projected_text):
    assert _CURRENT is not None
    ev,details=evidence(tokens,projected_text,_CURRENT.selective,_CURRENT.fc_only)
    pln._e26_logits=ev
    pln._e26_details=details
    pln._e26_last=dict(base=base_log_probs.detach(),tokens=tokens.detach(),text=projected_text.detach())
    return apply(base_log_probs,ev,_CURRENT.readout)

def loss(pln,labels,train):
    assert _CURRENT is not None and hasattr(pln,'_e26_logits')
    return aux_loss(pln._e26_logits,labels,train,_CURRENT.auxiliary)

def counterfactual(pln,base_logits,tokens,txt,mode):
    # Read-only after checkpoint; never evaluate held-out label during routing.
    if mode=='REAL':return forward(pln,base_logits,tokens,txt)
    if mode=='SWAP':return forward(pln,base_logits,tokens,txt.flip(0))
    if mode=='ZERO':return base_logits
    raise ValueError(mode)
