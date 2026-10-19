"""Experiment26 fixed 6 new arms; historical 30-fold P0/E9 are readonly."""
from dataclasses import dataclass
SEEDS=(0,1,2)
FOLDS=tuple(range(1,11))
GAMMA=.30
LAMBDA_AUX=.10
CONF_CENTER=.20
CONF_TAU=.10
@dataclass(frozen=True)
class Design:
    name: str
    prompt: str = 'real'
    selective: bool = True
    auxiliary: bool = True
    readout: bool = True
    fc_only: bool = False
DESIGNS={
    'D1':Design('class-signed uniform evidence','real',False,True,True,False),
    'D2':Design('full class-signed selective evidence','real',True,True,True,False),
    'D3':Design('auxiliary alignment only, no evidence readout','real',True,True,False,False),
    'D4':Design('unrelated text matched capacity','unrelated',True,True,True,False),
    'D5':Design('isometric random text matched Gram','isometric',True,True,True,False),
    'D6':Design('FC-only prompt evidence','real',True,True,True,True),
}
METRICS=('acc','ba','probability_auc','sen_asd','spe_td','f1_asd')
TEXT_UNRELATED=(
    'A quiet coastal lighthouse','A stone bridge over a river',
    'A wooden chair near a window','A green bicycle outside a shop',
    'A red umbrella on a sidewalk','A metal kettle on a table',
    'A glass bottle beside a notebook','A small boat on a lake',
)
def tasks():
    out=[(c,s,i%2) for i,(c,s) in enumerate((c,s) for c in DESIGNS for s in SEEDS)]
    assert len(out)==18 and sum(g==0 for _,_,g in out)==sum(g==1 for _,_,g in out)==9
    return out
def validate():
    assert list(DESIGNS)==['D1','D2','D3','D4','D5','D6']
    assert all(x.prompt in ('real','unrelated','isometric') for x in DESIGNS.values())
    assert len(tasks())*len(FOLDS)==180
