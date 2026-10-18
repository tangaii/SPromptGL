"""Freeze before training. Pilot is first 2 folds of each uninterrupted (config,seed) task."""
from dataclasses import dataclass
SEEDS=(0,1,2)
FOLDS=tuple(range(1,11))
@dataclass(frozen=True)
class Design:
    name:str
    sup:bool=False
    alignment:str='none' # none/uniform/selective
    safety:bool=False
    shuffle:bool=False

DESIGNS={
 'A0':Design('historical P0'),
 'A1':Design('FC label-graph supervision only',True),
 'A2':Design('FC graph supervision + uniform relational alignment',True,'uniform'),
 'A3':Design('FC graph supervision + agreement-selective alignment',True,'selective'),
 'A4':Design('FC graph supervision + conflict-safe selective alignment',True,'selective',True),
 'A5':Design('FC graph supervision + shuffled auxiliary relations',True,'selective',True,True),
 'A6':Design('selective relational alignment without label graph supervision',False,'selective'),
 'A7':Design('FC graph supervision + conflict-safe uniform alignment',True,'uniform',True),
}
LAMBDA_SUP=0.05
LAMBDA_ALIGN=0.10
PILOT_FOLDS=2
CONTRASTS=[('A4',k) for k in ('A0','A1','A2','A3','A5','A6','A7')]
METRICS=('acc','ba','probability_auc','sen_asd','spe_td','f1_asd')

def tasks():
    res=[(k,s) for k in DESIGNS for s in SEEDS]
    assert len(res)==24
    return [(k,s,i%2) for i,(k,s) in enumerate(res)]

def assert_contract():
    assert set(DESIGNS)=={f'A{i}' for i in range(8)}
    assert len(tasks())*PILOT_FOLDS==48
    assert len(tasks())*len(FOLDS)==240
    assert sum(g==0 for _,_,g in tasks())==sum(g==1 for _,_,g in tasks())==12

if __name__=='__main__':
    assert_contract();print('contract PASS')

