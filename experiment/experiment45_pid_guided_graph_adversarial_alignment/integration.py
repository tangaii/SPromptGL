"""Read-only data, anchor and isolated source integration for Experiment 45."""
from __future__ import annotations
import hashlib, json, os, random, shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path('/data/adodas/SPromptGL-main')
EXP = Path(__file__).resolve().parent
RT = EXP / '.runtime'
HIST = ROOT / 'experiment/experiment43_2_gfr_incremental_value_audit'
DIMS = (48, 6, 10, 256)
MODS = ('PHENO', 'ANAT', 'FUNC', 'CORR')
SLICES = ((0, 48), (48, 54), (54, 64), (64, 320))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def read_data():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    base = ROOT / 'data/ABIDE'
    frame = pd.read_csv(base / 'processed_standard_data.csv')
    mapping = np.load(base / 'modal_feat_dict.npy', allow_pickle=True).item()
    assert list(mapping) == ['PHENO', 'ANAT', 'FUNC', 'Correlation']
    assert [len(v) for v in mapping.values()] == list(DIMS)
    assert frame.shape == (871, 321)
    assert list(frame.columns[:-1]) == [str(v) for k in mapping for v in mapping[k]]
    x = frame.iloc[:, :-1].to_numpy(np.float32)
    y = frame.iloc[:, -1].to_numpy(np.int64) - 1
    assert x.shape == (871, 320) and np.isfinite(x).all()
    assert np.bincount(y).tolist() == [403, 468]
    folds = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y))
    return x, y, folds, mapping, list(frame.columns[:-1])

def fold_sha(train, test):
    return hashlib.sha256(np.asarray(train, np.int64).tobytes() + np.asarray(test, np.int64).tobytes()).hexdigest()

def source_copy():
    src = RT / 'sources'; src.mkdir(parents=True, exist_ok=True)
    old = HIST / '.runtime/sources'
    for name in ('network.py', 'layers.py', 'utils.py'):
        shutil.copy2(old / name, src / name)
    return {name: sha(src / name) for name in ('network.py', 'layers.py', 'utils.py')}

def copy_anchors():
    src = HIST / '.runtime/anchors'; dst = RT / 'anchors'; dst.mkdir(parents=True, exist_ok=True)
    for f in range(1, 11): shutil.copy2(src / f'fold{f:02}.pt', dst / f'fold{f:02}.pt')
    shutil.copy2(src / 'source_index.json', dst / 'source_index.json')
    return {str(f): sha(dst / f'fold{f:02}.pt') for f in range(1, 11)}

def make_hp(cfg):
    p = cfg['protocol']
    return SimpleNamespace(datadir=str(ROOT / 'data') + '/', datname='ABIDE', cpu=False, gpu='-1',
        nepoch=p['nepoch'], early=p['early'], lr=p['lr'], reg=p['reg'], dropout=p['dropout'],
        nlayer=p['nlayer'], n_hidden=p['n_hidden'], n_head=p['n_head'], n_iter=10, nmodal=4,
        th=p['threshold'], GC_mode='weighted-cosine', MP_mode='GCN', MF_mode='concat', alpha=.5,
        theta_smooth=p['theta_smooth'], theta_degree=p['theta_degree'], theta_sparsity=p['theta_sparsity'],
        nclass=p['nclass'], mode='simple-2', seed=0)

def protected_hashes():
    names = ['main.py', 'model.py', 'network.py', 'layers.py', 'utils.py',
             'ABIDE-simple-2-concat-weighted-cosine.sh',
             'data/ABIDE/processed_standard_data.csv', 'data/ABIDE/modal_feat_dict.npy']
    return {str(ROOT / n): sha(ROOT / n) for n in names}
