"""Frozen Experiment25: six NEW methods; historical P0 is read only."""
from dataclasses import dataclass
SEEDS=(0,1,2)
FOLDS=tuple(range(1,11))
RHO=0.20
@dataclass(frozen=True)
class Config:
    title:str
    kind:str
    site_adjust:bool=True
    shuffle:bool=False

DESIGNS={
    'B0':Config('Support-mean constant attenuation','constant'),
    'B1':Config('ANAT quality-conditioned FC edges','anat'),
    'B2':Config('FUNC quality-conditioned FC edges','func'),
    'B3':Config('Site-adjusted ANAT+FUNC reliability alignment','dual'),
    'B4':Config('Within-site shuffled ANAT+FUNC','dual',shuffle=True),
    'B5':Config('Unadjusted ANAT+FUNC','dual',site_adjust=False),
}
METRICS=('acc','ba','probability_auc','sen_asd','spe_td','f1_asd')

def tasks():
    xs=[(k,s,i%2) for i,(k,s) in enumerate((k,s) for k in DESIGNS for s in SEEDS)]
    assert len(xs)==18 and sum(g==0 for _,_,g in xs)==sum(g==1 for _,_,g in xs)==9
    return xs

def validate():
    assert len(DESIGNS)==6
    assert len(FOLDS)==10
    assert len(tasks())*10==180
    assert RHO>0 and RHO<=.25

