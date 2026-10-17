"""Frozen Exp22 registrations and runtime branch installer.

Everything here is independent of earlier experiment *modules*. The runner
copies audited source texts/datasets and uses baseline Experiment9 patch.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Spec:
    base: str             # P0 or E9: experimental P3+D (no GRL)
    condition: str        # REAL, NUMERIC, NEUTRAL, WRONG_ANATOMY, ISOMETRIC
    graph: bool
    variant: str          # SIGNED, UNSIGNED, SHARED
    complement: bool
    explanation: str

SPECS = {
    'M00': Spec('P0','REAL',False,'SIGNED',False,'P0 + disease-concept signed local'),
    'M01': Spec('P0','NUMERIC',False,'SIGNED',False,'P0 + numerical signed local'),
    'M02': Spec('P0','NEUTRAL',False,'SIGNED',False,'P0 + neutral signed local'),
    'M03': Spec('P0','WRONG_ANATOMY',False,'SIGNED',False,'P0 + wrong-ROI signed local'),
    'M04': Spec('P0','ISOMETRIC',False,'SIGNED',False,'P0 + isometric random signed local'),
    'M05': Spec('P0','REAL',False,'UNSIGNED',False,'P0 + real text; omit FC direction'),
    'M06': Spec('P0','REAL',False,'SHARED',False,'P0 + real text; one shared domain adaptation'),
    'M07': Spec('P0','REAL',True,'SIGNED',False,'P0 + true prior + signed evidence + magnitude-preserving graph'),
    'M08': Spec('P0','NUMERIC',True,'SIGNED',False,'P0 + numerical signed graph, no semantic factor'),
    'M09': Spec('P0','NEUTRAL',True,'SIGNED',False,'P0 + neutral signed magnitude graph'),
    'M10': Spec('E9','REAL',False,'SIGNED',False,'E9 + true signed local'),
    'M11': Spec('E9','NUMERIC',False,'SIGNED',False,'E9 + numerical signed local'),
    'M12': Spec('E9','NEUTRAL',False,'SIGNED',False,'E9 + neutral signed local'),
    'M13': Spec('E9','REAL',False,'SIGNED',True,'E9 + true signed local with orthogonal CORR residual'),
    'M14': Spec('E9','REAL',True,'SIGNED',False,'E9 + true signed magnitude graph'),
    'M15': Spec('E9','NUMERIC',True,'SIGNED',False,'E9 + numerical signed graph'),
    'M16': Spec('E9','NEUTRAL',True,'SIGNED',False,'E9 + neutral signed graph'),
    'M17': Spec('E9','REAL',True,'SIGNED',True,'E9 + true signed graph with orthogonal CORR residual'),
}
assert list(SPECS)==[f'M{i:02}' for i in range(18)]
TASKS=[(name,seed,(i*3+seed)%2) for i,name in enumerate(SPECS) for seed in (0,1,2)]
assert len(TASKS)==54
assert sum(g==0 for _,_,g in TASKS)==sum(g==1 for _,_,g in TASKS)==27
HISTORICAL_REFERENCES={
    'P0':'U00', 'E9':'U01', 'old_real_local':'U06',
    'old_real_graph':'U11','old_e9_local':'U14','old_e9_graph':'U16',
}
PAIRS=[
    ('M00','M01'),('M00','M02'),('M00','M03'),('M00','M04'),
    ('M00','M05'),('M00','M06'),('M00','P0'),('M00','old_real_local'),
    ('M07','M08'),('M07','M09'),('M07','M00'),('M07','old_real_graph'),
    ('M10','M11'),('M10','M12'),('M10','E9'),('M10','old_e9_local'),
    ('M13','M10'),('M13','E9'),
    ('M14','M15'),('M14','M16'),('M14','E9'),('M14','old_e9_graph'),
    ('M17','M14'),('M17','E9'),
]
CF_GROUPS=['M00','M07','M10','M14','M13','M17']
CF_ACTIONS=[
    'REAL','NEUTRAL','ISOMETRIC','WRONG_ANATOMY','NUMERIC_ONLY',
    'ADAPT_OFF','SIGN_OFF','SHARED_DOMAIN','GRAPH_SEMANTIC_OFF',
    'GRAPH_STRENGTH_OFF','GRAPH_OFF','BRANCH_OFF','FC_BRANCH_SHUFFLED',
]
# local groups have 3 not applicable graph actions; graph groups have 13.
EXPECTED_CF_ROWS=len(CF_GROUPS)*3*10*len(CF_ACTIONS)
EXPECTED_CF_NA=2*3*10*3 # M00/M10 local graph interventions; M13 is also local -> fix below
EXPECTED_CF_NA=3*3*10*3 # M00/M10/M13 are local
assert EXPECTED_CF_ROWS==2340 and EXPECTED_CF_NA==270


def install_evidence22(agent, *, cache, train_idx, raw_modalities, device, config, seed):
    """Invoke from EvalHelper.__init__ only AFTER Exp9 initialization.

    This function does not create any host cache/CLIP context. Runtime passes
    only outer train labels to the branch, never complete label arrays.
    """
    import torch
    from signed_evidence22 import SignedConceptEvidence22
    spec=SPECS[config]
    train_t=torch.as_tensor(train_idx,dtype=torch.long,device=agent.targ.device)
    y_train=agent.targ.index_select(0,train_t).detach()
    assert len(raw_modalities)==4 and tuple(raw_modalities[3].shape)==(871,256)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if agent.dev.type=='cuda' else []):
        torch.manual_seed(220000+int(seed))
        module=SignedConceptEvidence22(
            condition=spec.condition,use_graph=spec.graph,cache=cache,
            raw_corr=raw_modalities[3],train_idx=train_idx,train_labels=y_train,
            complement=spec.complement,variant=spec.variant,kl_weight=0.002,
        ).to(device)
    expected=20525 if spec.graph else 19437
    assert sum(p.numel() for p in module.parameters())==expected
    agent.ModalFusion.evidence22=module
    agent._e22_raw=raw_modalities
    agent.optimizer_MF.add_param_group(dict(params=list(module.parameters()),lr=agent.hyperpm.lr,weight_decay=0.))
    return module

