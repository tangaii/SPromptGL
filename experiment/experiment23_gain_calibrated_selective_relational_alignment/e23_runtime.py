"""The sole link between the accepted SPromptGL EvalHelper and experiment23.

One isolated GPU worker = one process. `configure` before creating any agent.
This module is imported by experiment-local model.py, never the project root.
"""
from __future__ import annotations
import torch
from e23_bridge import GainRelationalBridge, CONDITIONS

_CONTEXT=None

def configure(*, config, fold, evidence, pheno_indices):
    global _CONTEXT
    assert config in CONDITIONS and fold in range(1,11)
    _CONTEXT=dict(config=config,fold=fold,evidence=evidence,pheno_indices=pheno_indices)


def install_branch23(agent):
    """Called AFTER EvalHelper's ModalFusion.apply(my_weight_init).

    Model state + Adam state now include branch. No new GFR optimizer groups.
    """
    assert _CONTEXT is not None,'configure runtime before creating agent'
    data=_CONTEXT
    if data['config']=='G0':
        assert not hasattr(agent.ModalFusion,'bridge23')
        return
    assert not hasattr(agent.ModalFusion,'bridge23')
    modalities=agent.dataLayer(agent.feat)[1]
    assert [tuple(x.shape) for x in modalities] == [(871,48),(871,6),(871,10),(871,256)]
    evidence=data['evidence']
    assert (evidence['outer_train']==agent.trn_idx).all()
    assert (evidence['outer_test']==agent.tst_idx).all()
    # Construct on current CUDA device, but don't advance historical RNG stream.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(230000 + int(agent.hyperpm.seed))
        branch=GainRelationalBridge(data['config'],modalities,agent.trn_idx,
                                    data['pheno_indices'],evidence).to(agent.dev)
    agent.ModalFusion.bridge23=branch
    agent.optimizer_MF.add_param_group(dict(params=branch.parameters(),
                                             lr=agent.hyperpm.lr,weight_decay=0.))
    # Network is a component of ModalFusion, so .train()/.eval() synchronize.
    original={id(p) for p in agent.MessagePassing0.parameters()}
    original.update(id(p) for n,p in agent.ModalFusion.named_parameters() if not n.startswith('bridge23.'))
    params=[p for g in agent.optimizer_MF.param_groups+agent.optimizer_MP0.param_groups
            for p in g['params']]
    assert len(params)==len(set(map(id,params)))
    assert set(map(id,params))=={id(p) for p in agent.MessagePassing0.parameters()} | {id(p) for p in agent.ModalFusion.parameters()}


def intervention(agent, condition):
    """Use inside same checkpoint, restore condition=REAL after every call."""
    b=getattr(agent.ModalFusion,'bridge23',None)
    assert b is not None
    assert condition in ('REAL','GATE_OFF','GATE_CONSTANT','RELATION_OFF')
    b.condition=condition


def check_partition_independence(agent,evidence):
    assert not set(evidence['outer_train']).intersection(evidence['outer_test'])
    assert len(evidence['outer_train'])+len(evidence['outer_test'])==871
    # Teacher/gate is all on outer train; test labels are never in runtime evidence.
    assert not any(k in evidence for k in ('test_label','test_y','y_test'))

