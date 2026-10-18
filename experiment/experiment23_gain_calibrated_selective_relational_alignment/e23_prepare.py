"""Build just one 10-fold evidence.npz + stage-0 feasibility.json.

Usage: python e23_prepare.py /data/adodas/SPromptGL-main EXP23_DIR
"""
from __future__ import annotations
from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from e23_contract import feasibility_gate,assert_plan
from e23_evidence import build_fold,readonly_evidence_digest


def execute(root,exp):
    root,exp=Path(root),Path(exp)
    assert_plan()
    table=pd.read_csv(root/'data/ABIDE/processed_standard_data.csv')
    mapping=np.load(root/'data/ABIDE/modal_feat_dict.npy',allow_pickle=True).item()
    assert list(mapping)==['PHENO','ANAT','FUNC','Correlation']
    assert [len(v) for v in mapping.values()]==[48,6,10,256]
    x=table.iloc[:,:-1].to_numpy(dtype=np.float64)
    y=table.iloc[:,-1].to_numpy(dtype=np.int64)-1
    assert x.shape==(871,320) and (y==0).sum()==403 and (y==1).sum()==468
    ph=[i for i,n in enumerate(mapping['PHENO']) if str(n).startswith(('AGE_AT_SCAN_','SEX_'))]
    sites=[i for i,n in enumerate(mapping['PHENO']) if str(n).startswith('SITE_ID_')]
    assert len(ph)==28 and len(sites)==20 and not set(ph).intersection(sites)
    folds=list(StratifiedKFold(10,random_state=10,shuffle=True).split(x,y))
    dest={}; reports=[];digests={}
    for fold,(tr,te) in enumerate(folds,1):
        e=build_fold(x,y,tr,te,ph,seed=23000+fold)
        r=e.pop('report');r['fold']=fold
        reports.append(r)
        digests[str(fold)]=readonly_evidence_digest(e)
        for k,v in e.items():
            assert isinstance(v,np.ndarray)
            dest[f'fold{fold}_{k}']=v
    exp.mkdir(parents=True,exist_ok=True)
    target=exp/'evidence.npz'
    temp=exp/'evidence.npz.tmp'
    with temp.open('wb') as f:np.savez_compressed(f,**dest)
    temp.replace(target)
    verdict=feasibility_gate(reports)
    data={'folds':reports,'decision':verdict,'evidence_digest_by_fold':digests,
          'dataset_sha256':hashlib.sha256((root/'data/ABIDE/processed_standard_data.csv').read_bytes()).hexdigest(),
          'training_labels_only':True,'historic_preprocessing_untouched':True,
          'meta_oof_limit':'First-level OOF predictions overlap across meta folds; exploratory only.'}
    out=exp/'results';out.mkdir(exist_ok=True)
    (out/'feasibility.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print('Stage-0 decision:',verdict)
    return verdict


def load_fold(path,fold):
    with np.load(path,allow_pickle=False) as z:
        prefix=f'fold{fold}_'
        values={k[len(prefix):]:z[k].copy() for k in z.files if k.startswith(prefix)}
    assert len(values)>=10
    return values

if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('Usage: python e23_prepare.py ROOT EXP_DIR')
    execute(sys.argv[1],sys.argv[2])

