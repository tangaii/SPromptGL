"""Frozen configs, stage-0 feasibility and balanced GPU task assignment."""
import itertools
import json
import numpy as np
from e23_bridge import CONDITIONS

SEEDS=(0,1,2)
FOLDS=tuple(range(1,11))
CONFIGS=tuple(CONDITIONS)
METRICS=('acc','ba','hard_auc','probability_auc','sensitivity_asd','specificity_td','f1_asd')
CONTRASTS=(('G4','G0'),('G4','G1'),('G4','G2'),('G4','G3'),('G4','G5'),('G4','G6'),('G4','G7'))
INTERVENTIONS=('REAL','GATE_OFF','GATE_CONSTANT','RELATION_OFF')


def tasks():
    # Every task contains continuous original 10-fold RNG stream.
    result=[(c,s) for c in CONFIGS for s in SEEDS]
    return [(c,s,i%2) for i,(c,s) in enumerate(result)]


def assert_plan():
    ts=tasks()
    assert len(ts)==24 and len(set((c,s) for c,s,g in ts))==24
    assert sum(g==0 for c,s,g in ts)==sum(g==1 for c,s,g in ts)==12
    assert len(ts)*10==240
    assert len(CONTRASTS)==7


def feasibility_gate(rows):
    """Pre-registered FAST-FUTILITY: all four must be true to stop.

    Rows are train-only OOF observations. This is a planning heuristic,
    not a statistical claim of zero conditional information.
    """
    assert len(rows)==10
    mean_delta=float(np.mean([r['mean_oof_delta'] for r in rows]))
    high20=float(np.mean([r['high20_meta_oof_delta'] for r in rows]))
    rho=float(np.mean([r['meta_oof_spearman'] for r in rows]))
    sd=float(np.mean([r['gain_gate_train_std'] for r in rows]))
    stop=bool(mean_delta<=0 and high20<=0 and rho<=.02 and sd<.02)
    return dict(early_stop=stop,average_mean_delta=mean_delta,
                average_high20_delta=high20,average_meta_oof_spearman=rho,
                average_train_gate_std=sd,
                reason=('PRE-REGISTERED NO-UTILITY CONDITION' if stop else 'RUN EIGHT FIXED CONFIGS'))

if __name__=='__main__':
    assert_plan(); print('Contract PASS: 24 tasks 240 folds 120 each GPU')

