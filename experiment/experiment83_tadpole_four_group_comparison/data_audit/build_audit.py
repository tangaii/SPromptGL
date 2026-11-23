import hashlib, json, os, platform, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

E83=Path('/model/spromptgl/experiment/experiment83_tadpole_four_group_comparison')
RAW=Path('/model/spromptgl/Tadpole/tadpole_challenge')
PRE=E83/'preprocessing'

def sha256(path, chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def stat(path):
    p=Path(path)
    return {'path':str(p.resolve()),'size_bytes':p.stat().st_size,'sha256':sha256(p)}

raw_files=['TADPOLE_D1_D2.csv','TADPOLE_D1_D2_Dict.csv','TADPOLE_D3.csv','TADPOLE_D4_corr.csv','TADPOLE_readme.txt']
raw_manifest={f:stat(RAW/f) for f in raw_files}
proc=PRE/'processed_standard_data.csv'
modal_path=PRE/'modal_feat_dict.npy'
sample_path=PRE/'sample_ids.csv'
data=pd.read_csv(proc)
modal=np.load(modal_path,allow_pickle=True).item()
sample=pd.read_csv(sample_path)
y=(data['LABEL'].to_numpy(dtype=int)-1)
X=data.drop(columns=['LABEL']).to_numpy(dtype=np.float64)
assert X.shape==(598,365),X.shape
assert sample.shape[0]==598
assert np.array_equal(sample['LABEL'].to_numpy(dtype=int),data['LABEL'].to_numpy(dtype=int))
assert np.isfinite(X).all()
assert sorted(np.unique(y).tolist())==[0,1,2]
assert np.bincount(y).tolist()==[209,315,74]
modal_dims={k:len(v) for k,v in modal.items()}
assert list(modal.keys())==['UCSFFSX','UCBERKELEYAV45','UPENNBIOMK9','RISK_FACTOR','COGNITIVE_TEST','ROI_AVERAGE'],list(modal)
assert sum(modal_dims.values())==365
skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=10)
splits=[]
for fold,(tr,te) in enumerate(skf.split(X,y),1):
    splits.append({'fold':fold,'train_idx':tr.tolist(),'test_idx':te.tolist(),'train_count':int(len(tr)),'test_count':int(len(te)),'train_class_counts':np.bincount(y[tr],minlength=3).tolist(),'test_class_counts':np.bincount(y[te],minlength=3).tolist(),'val_equals_test':True})
(E83/'data_audit'/'split_indices.json').write_text(json.dumps({'protocol':'StratifiedKFold(n_splits=10,shuffle=True,random_state=10)','training_seed':0,'n_samples':598,'n_classes':3,'splits':splits},indent=2))
(E83/'data_audit'/'class_distribution.json').write_text(json.dumps({'dataset':'TADPOLE','n_samples':598,'label_column':'LABEL','raw_label_values':{'NC':1,'MCI':2,'AD':3},'model_label_values':{'NC':0,'MCI':1,'AD':2},'counts':{'NC':209,'MCI':315,'AD':74}},indent=2))
(E83/'data_audit'/'modality_manifest.json').write_text(json.dumps({'modality_order':list(modal.keys()),'modality_dimensions':modal_dims,'feature_total':sum(modal_dims.values()),'features':modal,'source':'MMGL Data_preprocessing_TADPOLE (2).ipynb','pre_cv_rfe':True,'rfe_features_requested':150,'post_missing_filter_dimensions':modal_dims},indent=2))
manifest={
 'dataset':'TADPOLE','raw_source_root':str(RAW.resolve()),'preprocessing_script':str((PRE/'preprocess_tadpole_mmgl.py').resolve()),
 'protocol':'MMGL Data_preprocessing_TADPOLE (2).ipynb, stable NC/sMCI/AD DXCHANGE 1/2/3, PMCI_selection=False, baseline VISCODE=bl, full-data RFE and StandardScaler before outer CV, mean imputation',
 'raw_files':raw_manifest,
 'processed_files':{'processed_standard_data.csv':stat(proc),'modal_feat_dict.npy':stat(modal_path),'sample_ids.csv':stat(sample_path)},
 'n_samples':int(len(data)),'n_features':int(X.shape[1]),'label_distribution':{'NC':209,'MCI':315,'AD':74},
 'label_encoding':{'raw':{'NC':1,'MCI':2,'AD':3},'model':{'NC':0,'MCI':1,'AD':2}},
 'modalities':modal_dims,'sample_id_column':'RID','sample_id_sha256':sha256(sample_path),
 'finite_numeric':bool(np.isfinite(X).all()),'duplicate_rid_count':int(sample['RID'].duplicated().sum()),
 'split_file':str((E83/'data_audit'/'split_indices.json').resolve()),'split_protocol':'StratifiedKFold(n_splits=10,shuffle=True,random_state=10)','seed':0,
}
(E83/'data_audit'/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'shape':X.shape,'counts':np.bincount(y,minlength=3).tolist(),'modalities':modal_dims,'processed_sha256':manifest['processed_files']['processed_standard_data.csv']['sha256']},indent=2))
