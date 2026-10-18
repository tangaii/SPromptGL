"""No params, no optimizers, no different evaluation path, no test labels."""
import numpy as np
from e24_align import CARA

CONTEXT=None

def configure(config,fold,sites):
    global CONTEXT
    CONTEXT=dict(config=config,fold=fold,sites=np.asarray(sites,dtype=int))

def install(agent):
    assert CONTEXT is not None
    if CONTEXT['config']=='A0':
        agent._e24=None
        return
    agent._e24=CARA(CONTEXT['config'],agent.trn_idx,
                    agent.targ.detach().cpu().numpy(),CONTEXT['sites'],CONTEXT['fold'])

