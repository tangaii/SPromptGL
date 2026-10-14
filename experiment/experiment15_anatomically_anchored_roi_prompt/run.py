#!/usr/bin/env python3
"""AARSP: eight frozen configurations, original continuous ten-fold protocol."""
import argparse
import contextlib
import csv
import difflib
import fcntl
import hashlib
import io
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

EXP = Path(__file__).resolve().parent
ROOT = EXP.parent.parent
STATE = Path('/tmp') / ('spromptgl15-state-' + hashlib.sha256(str(EXP).encode()).hexdigest()[:12])
SOURCES = STATE / 'sources'
MANIFEST = EXP / 'manifest.json'
RESULTS = EXP / 'results'
LOGS = EXP / 'logs'
CORE = ['main.py', 'model.py', 'network.py', 'layers.py', 'utils.py', 'ABIDE-simple-2-concat-weighted-cosine.sh']
MAPPING = ROOT / 'experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv'
LABELS = ROOT / 'experiment/experiment11_roi_mapping_recovery/results/roi_111_labels.csv'
CONFIGS = ['A' + str(i) for i in range(8)]
TASKS = [(c, s, (i * 3 + s) % 2) for i, c in enumerate(CONFIGS) for s in range(3)]
NAMES = ['原始P0', '无SPGS保留RES', '数值图/常量文本', '匿名ROI身份',
         '真实HO名称', '真实HO名称+功能', '固定语义错配', '真实文本/仅自身消息']
PROTOCOL = dict(datadir=str(ROOT / 'data') + '/', datname='ABIDE', cpu=False,
                gpu='-1', nepoch=450, early=200, lr=.0038, reg=.11, dropout=.35,
                nlayer=1, n_hidden=18, n_head=2, n_iter=10, nmodal=4, th=.9,
                GC_mode='weighted-cosine', MP_mode='GCN', MF_mode='concat',
                alpha=.5, theta_smooth=1., theta_degree=.5, theta_sparsity=0.,
                nclass=2, mode='simple-2')
METRICS = ['acc', 'ba', 'hard_auc', 'probability_auc', 'sensitivity_asd', 'specificity_td']
PAIRS = [('A1','A0'),('A2','A1'),('A3','A2'),('A4','A3'),('A5','A4'),
         ('A5','A6'),('A5','A7'),('A5','A0'),('A5','A3'),('A5','A2')]
_CTX = None
_SIGNATURE = None


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with open(temp, 'wb') as f:
        f.write(content.encode() if isinstance(content, str) else content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def save_json(path, obj):
    atomic(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_json(path):
    for attempt in range(6):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            if attempt == 5:
                raise
            time.sleep(.15)


def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    f = io.StringIO(newline='')
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, f.getvalue())


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def save_checkpoint(path, obj):
    import torch
    f = io.BytesIO()
    torch.save(obj, f)
    data = f.getvalue()
    atomic(path, b'SP15' + hashlib.sha256(data).hexdigest().encode() + data)


def checkpoint(path):
    import torch
    b = Path(path).read_bytes()
    assert b[:4] == b'SP15' and hashlib.sha256(b[68:]).hexdigest().encode() == b[4:68]
    return torch.load(io.BytesIO(b[68:]), map_location='cpu', weights_only=False)


def seed_rng(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def rng_hash(state):
    h = hashlib.sha256(pickle.dumps((state['python'], state['numpy']), protocol=4))
    for value in [state['cpu']] + state['cuda']:
        h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def input_data():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    data = pd.read_csv(ROOT / 'data/ABIDE/processed_standard_data.csv').values
    dims = [len(v) for v in np.load(ROOT / 'data/ABIDE/modal_feat_dict.npy', allow_pickle=True).item().values()]
    assert data.shape == (871, 321) and dims == [48,6,10,256] and np.isfinite(data).all()
    x, y = data[:, :-1], data[:, -1].astype(int) - 1
    assert set(y) == {0,1}
    return x, y, list(StratifiedKFold(10, shuffle=True, random_state=10).split(x,y))


def verified_graph():
    import pandas as pd
    import numpy as np
    labels = pd.read_csv(LABELS)
    rows = labels[labels.name_verified & labels.column_order_verified & ~labels.is_extra_roi]
    rows = rows.sort_values('roi_index_zero_based')
    assert len(rows) == 110 and rows.roi_name.nunique() == 110
    lookup = {int(r.roi_index_zero_based): i for i,r in enumerate(rows.itertuples())}
    mapping = pd.read_csv(MAPPING).sort_values('correlation_feature_index')
    assert mapping.correlation_feature_index.tolist() == list(range(256))
    valid = mapping[mapping.usable_for_roi_semantic_prompt]
    assert len(valid) == 252 and (valid.verification_status == 'VERIFIED_NUMERIC_UNIQUE').all()
    assert valid.n_subjects_verified.eq(871).all() and valid.max_absolute_error.max() < 1e-12
    indices = valid.correlation_feature_index.to_numpy(dtype='int64')
    ends = np.array([[lookup[int(r.roi_i_zero_based)], lookup[int(r.roi_j_zero_based)]] for r in valid.itertuples()])
    assert np.all(ends[:,0] != ends[:,1]) and len(set(map(tuple, ends))) == 252
    unknown = mapping[~mapping.usable_for_roi_semantic_prompt].correlation_feature_index.tolist()
    assert unknown == [195,225,232,236]
    return rows.to_dict('records'), indices, ends, unknown


def build_prompts():
    import ast
    import numpy as np
    regions, _, _, _ = verified_graph()
    roles = {
        'Thalamus': 'a subcortical structure involved in relaying and integrating neural information',
        'Caudate': 'a basal ganglia structure contributing to action selection and learning',
        'Putamen': 'a basal ganglia structure contributing to movement and action learning',
        'Pallidum': 'a basal ganglia structure contributing to regulation of movement',
        'Hippocampus': 'a medial temporal structure contributing to memory and spatial processing',
        'Amygdala': 'a medial temporal structure contributing to emotional learning',
        'Accumbens': 'a ventral striatal structure contributing to motivation and reward processing',
        'Frontal Pole': 'an anterior frontal region contributing to higher-order cognitive processing',
        'Insular Cortex': 'a cortical region contributing to integration of bodily and sensory information',
        'Superior Frontal Gyrus': 'a frontal region contributing to cognitive and motor control',
        'Middle Frontal Gyrus': 'a frontal association region contributing to cognitive control',
        'Inferior Frontal Gyrus, pars triangularis': 'a lateral frontal region contributing to language and cognitive processing',
        'Inferior Frontal Gyrus, pars opercularis': 'a lateral frontal region contributing to language and action processing',
        'Precentral Gyrus': 'a cortical region involved in voluntary motor control',
        'Temporal Pole': 'a temporal association region involved in integrating sensory and conceptual information',
        'Superior Temporal Gyrus, anterior division': 'a temporal region contributing to auditory and association processing',
        'Superior Temporal Gyrus, posterior division': 'a temporal region contributing to auditory and language processing',
        'Middle Temporal Gyrus, anterior division': 'a temporal association region contributing to conceptual processing',
        'Middle Temporal Gyrus, posterior division': 'a temporal association region contributing to sensory and conceptual processing',
        'Middle Temporal Gyrus, temporooccipital part': 'a temporal-occipital association region contributing to visual information processing',
        'Inferior Temporal Gyrus, anterior division': 'a temporal association region contributing to visual object processing',
        'Inferior Temporal Gyrus, posterior division': 'a temporal region contributing to visual object processing',
        'Inferior Temporal Gyrus, temporooccipital part': 'a temporal-occipital region contributing to visual object processing',
        'Postcentral Gyrus': 'a cortical region involved in somatosensory processing',
        'Superior Parietal Lobule': 'a parietal association region contributing to spatial and sensorimotor integration',
        'Supramarginal Gyrus, anterior division': 'a parietal association region contributing to sensory integration',
        'Supramarginal Gyrus, posterior division': 'a parietal association region contributing to sensory and language processing',
        'Angular Gyrus': 'a parietal association region contributing to multimodal information integration',
        'Lateral Occipital Cortex, superior division': 'an occipital region contributing to visual information processing',
        'Lateral Occipital Cortex, inferior division': 'an occipital region contributing to visual object processing',
        'Intracalcarine Cortex': 'a cortical region involved in early visual processing',
        'Frontal Medial Cortex': 'a medial frontal region contributing to integrative cognitive processing',
        'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)': 'a medial frontal region contributing to motor planning and coordination',
        'Subcallosal Cortex': 'a medial cortical region contributing to affective processing',
        'Paracingulate Gyrus': 'a medial frontal region contributing to cognitive and affective processing',
        'Cingulate Gyrus, anterior division': 'a medial cortical region contributing to cognitive and affective processing',
        'Cingulate Gyrus, posterior division': 'a medial cortical region contributing to memory and integrative processing',
        'Precuneous Cortex': 'a medial parietal region contributing to visuospatial and integrative processing',
        'Cuneal Cortex': 'an occipital region contributing to visual processing',
        'Frontal Orbital Cortex': 'a frontal region contributing to valuation and decision processing',
        'Parahippocampal Gyrus, anterior division': 'a medial temporal region contributing to contextual memory processing',
        'Parahippocampal Gyrus, posterior division': 'a medial temporal region contributing to contextual and scene processing',
        'Lingual Gyrus': 'an occipital region contributing to visual processing',
        'Temporal Fusiform Cortex, anterior division': 'a ventral temporal region contributing to visual recognition',
        'Temporal Fusiform Cortex, posterior division': 'a ventral temporal region contributing to visual recognition',
        'Temporal Occipital Fusiform Cortex': 'a ventral visual region contributing to object recognition',
        'Occipital Fusiform Gyrus': 'a ventral occipital region contributing to visual recognition',
        'Frontal Operculum Cortex': 'an opercular region contributing to oral sensorimotor and language processing',
        'Central Opercular Cortex': 'an opercular region contributing to sensory and motor integration',
        'Parietal Operculum Cortex': 'an opercular region contributing to somatosensory processing',
        'Planum Polare': 'a temporal region contributing to auditory processing',
        "Heschl's Gyrus (includes H1 and H2)": 'a temporal cortical region involved in auditory processing',
        'Planum Temporale': 'a temporal cortical region contributing to auditory processing',
        'Supracalcarine Cortex': 'an occipital region contributing to visual processing',
        'Occipital Pole': 'an occipital region involved in visual processing'}
    banks = {'CONSTANT': ['A brain region in a neuroimaging atlas.']}
    banks['ANONYMOUS'] = [f'Brain region ID {i+1:03d} in the atlas.' for i in range(110)]
    banks['NAME'] = [f"{r['roi_name']} is a region in the Harvard-Oxford brain atlas." for r in regions]
    banks['FUNCTION'] = []
    for r in regions:
        name = r['roi_name']
        base = name.split(' ',1)[1]
        assert base in roles, base
        banks['FUNCTION'].append(f'{name} is {roles[base]}.')
    gen = np.random.default_rng(1506)
    perm = gen.permutation(110)
    while np.any(perm == np.arange(110)):
        perm = gen.permutation(110)
    tree = ast.parse((ROOT / 'network.py').read_text())
    p0 = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='attr_words' for t in n.targets))
    records = [{k:r[k] for k in ('roi_index_zero_based','atlas_integer_value','roi_name','hemisphere','anatomical_category')} for r in regions]
    obj = dict(atlas='Verified ABIDE PCP Harvard-Oxford; extra 3455 excluded',
               regions=records, banks=banks, a6_permutation=perm.tolist(),
               a6_mapping_rule='node i receives FUNCTION[a6_permutation[i]]',
               p0=p0, freeze_rule='No diagnosis, patient state, or online generation.',
               function_scope='Conservative regional associations, not exclusive roles or network assignments.',
               function_sources=['https://www.nature.com/articles/nn.4269',
                    'https://pubmed.ncbi.nlm.nih.gov/18331903/',
                    'https://www.fmrib.ox.ac.uk/primers/intro_primer/ExBox21/IntroBox21.html',
                    'https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf'])
    save_json(EXP / 'prompts.json', obj)
    return obj


def patched_sources():
    original = {name:(ROOT/name).read_text() for name in ['network.py','model.py']}
    n = original['network.py']
    begin = n.index('_prompt_text_features = None')
    end = n.index('\nclass MLP',begin)
    n = n[:begin] + '''from aarsp_runtime import get_frozen_p0, make_roi_encoder, inject_roi

def get_prompt_text_features():
    return get_frozen_p0()
''' + n[end:]
    begin = n.index('        text_features = self.prompt_text_features\n')
    end = n.index('        # 对最后一个维度进行平均池化', begin)
    old = n[begin:end]
    n = n[:begin] + "        if os.environ.get('AARSP_CONFIG', 'A0') == 'A0':\n" + ''.join('    '+line if line.strip() else line for line in old.splitlines(True)) + "        else:\n            part_x = x.view(871, 4, 4, 9)\n" + n[end:]
    m = original['model.py']
    old = '        self.ModalFusion.apply(my_weight_init)\n'
    added = old + '''        if os.environ.get('AARSP_CONFIG', 'A0') in ('A2','A3','A4','A5','A6','A7'):
            # Register after all original initialization; preserve its RNG stream.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(150000 + int(hyperpm.seed))
                self.ModalFusion.roi_encoder = make_roi_encoder().to(dev)
            self.optimizer_MF.add_param_group({'params': self.ModalFusion.roi_encoder.parameters(),
                                             'lr': hyperpm.lr, 'weight_decay': hyperpm.reg})
'''
    assert m.count(old) == 1
    m = m.replace(old, added)
    for line in ['            feat, adj, ALoss = self.MessagePassing0(feat1,model_num, self.targ, self.trn_idx)\n',
                 '        feat ,_,_= self.MessagePassing0(feat1,model_num, self.targ, self.trn_idx)\n']:
        assert m.count(line) == 1
        indent = line[:len(line)-len(line.lstrip())]
        m = m.replace(line,line+indent+'feat = inject_roi(self, feat)\n')
    modified = {'network.py':n,'model.py':m}
    patch = ''.join(''.join(difflib.unified_diff(original[k].splitlines(True),modified[k].splitlines(True),fromfile='a/'+k,tofile='b/'+k)) for k in original)
    return modified, patch


def make_sources():
    SOURCES.mkdir(parents=True, exist_ok=True)
    for name in CORE:
        if not (SOURCES/name).exists():
            shutil.copy2(ROOT/name,SOURCES/name)
    for name in ['CLIP','data']:
        if not (SOURCES/name).exists():
            (SOURCES/name).symlink_to(ROOT/name,target_is_directory=True)
    for name in ['attn','graph']:
        (SOURCES/name).mkdir(exist_ok=True)
    modified,_ = patched_sources()
    for name,text in modified.items():
        if not (SOURCES/name).exists() or (SOURCES/name).read_text() != text:
            atomic(SOURCES/name,text)
    link = EXP/'state'
    if not link.exists():
        link.symlink_to(STATE,target_is_directory=True)


def geometry(features):
    import numpy as np
    e = features.numpy().astype('float64')
    cos = (e @ e.T)[np.triu_indices(len(e),1)]
    def rank(a):
        s = np.linalg.svd(a,compute_uv=False)
        if s.sum() < 1e-9:
            return 0.
        p = s/s.sum()
        return float(np.exp(-(p[p>0]*np.log(p[p>0])).sum()))
    return dict(pair_cos_mean=float(cos.mean()),pair_cos_p05=float(np.quantile(cos,.05)),
                pair_cos_p50=float(np.quantile(cos,.5)),pair_cos_p95=float(np.quantile(cos,.95)),
                effective_rank=rank(e),centered_effective_rank=rank(e-e.mean(0)),
                exact_duplicate_rows=int(len(e)-len(np.unique(e,axis=0))))


def prepare():
    import torch
    from torch.nn import functional as F
    if MANIFEST.exists():
        m = read_json(MANIFEST)
        if m.get('run_status') == 'COMPLETE':
            integrity(full=True)
            return m
        if (STATE/'static.pt').exists():
            integrity()
            return m
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 1, 'GPU required; no CPU fallback'
    STATE.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    p = build_prompts()
    _,patch = patched_sources()
    atomic(EXP/'integration_patch.diff',patch)
    x,y,folds = input_data()
    _,indices,ends,unknown = verified_graph()
    from roi_semantic import patient_graph
    fixed = patient_graph(x[:,64:320],indices,ends)
    sys.path.insert(0,str(ROOT))
    from CLIP.clip import clip
    encoder,_ = clip.load('RN50','cuda')
    encoder.eval().requires_grad_(False)
    lengths = []
    def encode(sentences):
        sizes = [len(clip._tokenizer.encode(t))+2 for t in sentences]
        assert max(sizes) <= 77
        lengths.extend(sizes)
        with torch.no_grad():
            # P0 is deliberately encoded in its original batch of eight.
            return encoder.encode_text(clip.tokenize(sentences,truncate=False).cuda()).float().detach().cpu()
    p0 = encode(p['p0'])
    encoded = {k:F.normalize(encode(v),dim=-1) for k,v in p['banks'].items()}
    encoded['CONSTANT'] = encoded['CONSTANT'].expand(110,-1).clone()
    encoded['SHUFFLED'] = encoded['FUNCTION'][p['a6_permutation']].clone()
    history = ROOT/'experiment/experiment12_roi_semantic_alignment_feasibility/cache/roi_text_embeddings.pt'
    if history.exists():
        h = torch.load(history,map_location='cpu',weights_only=False)
        assert torch.equal(p0,h['p0_features']), 'P0 cached embedding parity failed'
    save_checkpoint(STATE/'static.pt',dict(p0=p0,banks=encoded,fixed=fixed))
    del encoder
    torch.cuda.empty_cache()
    source_paths = CORE + ['data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy',
                          str(MAPPING.relative_to(ROOT)),str(LABELS.relative_to(ROOT)),
                          'CLIP/clip/clip.py','CLIP/clip/model.py','CLIP/clip/simple_tokenizer.py']
    source = {p:sha(ROOT/p) for p in source_paths}
    protected_path = Path('/tmp/spromptgl15-audit/protected.json')
    protected = read_json(protected_path) if protected_path.exists() else {
        str(f.relative_to(ROOT)):sha(f) for d in ROOT.joinpath('experiment').glob('experiment*')
        if d != EXP for f in d.rglob('*') if f.is_file()}
    github_path = Path('/tmp/spromptgl15-audit/github_audit.json')
    github = read_json(github_path) if github_path.exists() else dict(status='not available; local source recorded')
    m = dict(experiment='AARSP Anatomically Anchored ROI Semantic Prompt',run_status='PREPARED',
             prepared_utc=utc(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             github_audit=github,source_data_sha256=source,protected_history_sha256=protected,
             protected_handoff_sha256=sha(ROOT/'handoff.md'),protected_handoff_bytes=(ROOT/'handoff.md').stat().st_size,
             clip_weight_sha256=sha(Path.home()/'.cache/clip/RN50.pt'),
             paper_sha256={p.name:sha(p) for p in ROOT.glob('*.pdf')},
             prompt_sha256=sha(EXP/'prompts.json'),patch_sha256=sha(EXP/'integration_patch.diff'),
             code_sha256={n:sha(EXP/n) for n in ['run.py','roi_semantic.py']},
             protocol=PROTOCOL,split_sha256=canonical([[a.tolist(),b.tolist()] for a,b in folds]),
             class_order={'0':'ASD','1':'TD'},dimensions=[48,6,10,256],
             roi_graph=dict(nodes=110,edges=252,unknown_features=unknown,
                            mean_degree=float(fixed['degree'].mean()),min_degree=int(fixed['degree'].min()),
                            max_degree=int(fixed['degree'].max()),edge_weight='tanh(Fisher-z), signed degree-normalized'),
             text_audit=dict(max_tokens=max(lengths),encoded_sentences=339,
                             geometry={k:geometry(v) for k,v in encoded.items()},encoder='RN50 frozen eval',
                             training_encode_calls=0),
             base_parameters=1462957,additional_parameters=38242,
             new_optimizer=dict(Adam=True,lr=.0038,weight_decay=.11),
             rng_policy='Each config independently preserves original continuous 10-fold RNG. No fold reseeding or A0 anchor injection. Extra initialization fork_rng(150000+seed). Identical entry => identical trunk; later entries may differ after different stopping lengths.',
             gpu_hardware=subprocess.check_output(['nvidia-smi'],text=True),
             cpu_cgroup_cores=32,memory_cgroup_gib=128,worker_threads=2,workers_per_gpu=4,
             tasks={f'{c}_seed{s}':dict(config=c,seed=s,physical_gpu=g,status='PENDING',folds=0) for c,s,g in TASKS},
             completed_folds=0,preflight=None)
    save_json(MANIFEST,m)
    make_sources()
    write_readme(m)
    return m


def integrity(full=False):
    m = read_json(MANIFEST)
    for p,h in m['source_data_sha256'].items():
        assert sha(ROOT/p) == h, ('source/data changed',p)
    assert sha(EXP/'prompts.json') == m['prompt_sha256']
    assert sha(EXP/'integration_patch.diff') == m['patch_sha256']
    for p,h in m['code_sha256'].items():
        assert sha(EXP/p) == h, ('code changed after freeze',p)
    if full:
        for p,h in m['protected_history_sha256'].items():
            assert sha(ROOT/p) == h, ('protected history changed',p)
        assert sha(Path.home()/'.cache/clip/RN50.pt') == m['clip_weight_sha256']
    return m


def signature():
    global _SIGNATURE
    if _SIGNATURE is not None:
        return _SIGNATURE
    m = read_json(MANIFEST)
    _SIGNATURE = canonical({k:m[k] for k in ['source_data_sha256','prompt_sha256','code_sha256',
                                           'patch_sha256','protocol','split_sha256','rng_policy']})
    return _SIGNATURE


def context():
    global _CTX
    if _CTX is not None:
        return _CTX
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'Worker must have exactly one visible GPU'
    sys.modules['aarsp_runtime'] = sys.modules[__name__]
    make_sources()
    sys.path.insert(0,str(SOURCES))
    import network
    import model
    x,y,folds = input_data()
    static = checkpoint(STATE/'static.pt')
    _CTX = dict(torch=torch,network=network,model=model,x=x,y=y,folds=folds,static=static,clip_loaded=False)
    return _CTX


def get_frozen_p0():
    c = context()
    if not c['clip_loaded']:
        # Preserve the original first lazy CLIP construction and CPU RNG use.
        encoder,_ = c['network'].clip.load('RN50','cuda')
        del encoder
        c['clip_loaded'] = True
    return c['static']['p0'].cuda()


def make_roi_encoder():
    from roi_semantic import ROIEncoder
    cfg = os.environ['AARSP_CONFIG']
    key = dict(A2='CONSTANT',A3='ANONYMOUS',A4='NAME',A5='FUNCTION',A6='SHUFFLED',A7='FUNCTION')[cfg]
    c = context()
    return ROIEncoder(c['static']['fixed'],c['static']['banks'][key],self_only=cfg=='A7')


def inject_roi(agent, h):
    module = getattr(agent.ModalFusion,'roi_encoder',None)
    agent._aarsp_base_fc = h[:,64:320].detach()
    if module is None:
        return h
    delta = module()
    new = h.clone()
    new[:,64:320] = new[:,64:320] + delta
    return new


def agent_for(config,seed,fold,original=False):
    c = context()
    os.environ['AARSP_CONFIG'] = config
    cls = c['model'].EvalHelper
    if original:
        module = types.ModuleType('aarsp_original_network')
        sys.modules[module.__name__] = module
        exec(compile((ROOT/'network.py').read_text(),'original_network.py','exec'),module.__dict__)
        original_model = types.ModuleType('aarsp_original_model')
        src = (ROOT/'model.py').read_text().replace('from network import *','from aarsp_original_network import *')
        exec(compile(src,'original_model.py','exec'),original_model.__dict__)
        cls = original_model.EvalHelper
    tr,te = c['folds'][fold-1]
    hp = SimpleNamespace(**PROTOCOL,seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        a = cls([48,6,10,256],c['x'],c['y'],hp,tr,te)
    a.MF_sav.close()
    a.GCMP_sav.close()
    assert (a.val_idx==a.tst_idx).all()
    return a


def model_state(a):
    return {name:{k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
            for name,module in [('gfr',a.MessagePassing0),('pln',a.ModalFusion)]}


def load_state(a,state):
    a.MessagePassing0.load_state_dict(state['gfr'])
    a.ModalFusion.load_state_dict(state['pln'])


def weight_hash(a,branch=False):
    h = hashlib.sha256()
    for name,module in [('gfr',a.MessagePassing0),('pln',a.ModalFusion)]:
        for key,value in sorted(module.named_parameters()):
            if key.startswith('roi_encoder.') == branch:
                h.update((name+'.'+key).encode()+value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def optimizer_audit(a,config):
    groups = a.optimizer_MP0.param_groups+a.optimizer_MF.param_groups
    params = [p for g in groups for p in g['params']]
    actual = [p for m in [a.MessagePassing0,a.ModalFusion] for p in m.parameters()]
    assert len(set(map(id,params)))==len(params) and set(map(id,params))==set(map(id,actual))
    assert all(g['lr']==.0038 and g['weight_decay']==.11 for g in groups)
    count = sum(p.numel() for p in actual)
    assert count==1462957+(38242 if config in CONFIGS[2:] else 0)
    return count


def forward(a):
    import torch
    a.MessagePassing0.eval()
    a.ModalFusion.eval()
    with torch.no_grad():
        n,parts = a.dataLayer(a.feat)
        h,_,_ = a.MessagePassing0(parts,n,a.targ,a.trn_idx)
        h = inject_roi(a,h)
        out,_,_ = a.ModalFusion(h)
        assert torch.isfinite(out).all()
        return out.clone()


def metrics(out,indices):
    import numpy as np
    from sklearn.metrics import confusion_matrix,roc_auc_score
    y = context()['y'][indices]
    p = out.exp().cpu().numpy()[indices]
    pred = p.argmax(1)
    cm = confusion_matrix(y,pred,labels=[0,1])
    sen,spe = float(cm[0,0]/cm[0].sum()),float(cm[1,1]/cm[1].sum())
    return dict(acc=float((pred==y).mean()),ba=(sen+spe)/2,
                hard_auc=float(roc_auc_score(np.eye(2)[y],np.eye(2)[pred])),
                probability_auc=float(roc_auc_score(y==0,p[:,0])),
                sensitivity_asd=sen,specificity_td=spe)


def gradients(a):
    import torch
    module = getattr(a.ModalFusion,'roi_encoder',None)
    if module is None:
        return {}
    components = dict(numerical=module.numerical.parameters(),query=module.query.parameters(),
                      message=module.message.parameters(),self_transform=module.self_transform.parameters(),
                      decoder=module.decoder.parameters(),alpha=[module.alpha])
    norms = []
    for params in components.values():
        grads = [p.grad for p in params if p.grad is not None]
        norms.append(torch.stack([g.square().sum() for g in grads]).sum().sqrt() if grads else module.alpha.new_zeros(()))
    values = torch.stack(norms).detach().cpu().tolist()
    assert all(math.isfinite(v) for v in values), 'Nonfinite ROI gradient'
    return dict(zip(components,values))


def train_step(a):
    a.run_epoch('simple-2')
    return gradients(a)


def counterfactual(a,seed,fold,real):
    import torch
    module = a.ModalFusion.roi_encoder
    idx = a.tst_idx
    baseline = metrics(real,idx)
    real_delta = module.last['delta'].clone()
    real_response = module.last['response'].clone()
    original_hash = weight_hash(a)+weight_hash(a,True)
    text = module.text_features.clone()
    rows = []
    for condition in ['REAL','SHUFFLED','CONSTANT','OFF','SELF_ONLY']:
        module.text_override = context()['static']['banks'][condition].cuda() if condition in ['SHUFFLED','CONSTANT'] else None
        module.off = condition=='OFF'
        module.self_only_override = True if condition=='SELF_ONLY' else None
        out = forward(a)
        if condition=='REAL':
            assert torch.equal(out,real)
        met = metrics(out,idx)
        row = dict(config='A5',seed=seed,fold=fold,condition=condition,**met,
                   flip_rate=float((out[idx].argmax(1)!=real[idx].argmax(1)).float().mean()),
                   probability_abs_change=float((out[idx,0].exp()-real[idx,0].exp()).abs().mean()),
                   residual_abs_mean=float(module.last['delta'][idx].abs().mean()),
                   residual_abs_change=float((module.last['delta'][idx]-real_delta[idx]).abs().mean()),
                   roi_response_abs_change=float((module.last['response'][idx]-real_response[idx]).abs().mean()),
                   **{k+'_delta':met[k]-baseline[k] for k in METRICS})
        rows.append(row)
    module.text_override = None
    module.off = False
    module.self_only_override = None
    assert torch.equal(forward(a),real) and torch.equal(module.text_features,text)
    assert weight_hash(a)+weight_hash(a,True)==original_hash
    return rows


def diagnostics(a,config,seed,fold,out,tracker):
    import torch
    import numpy as np
    row = dict(config=config,seed=seed,fold=fold,prediction_finite=bool(torch.isfinite(out).all()),
               enabled=config in CONFIGS[2:],spgs_enabled=config=='A0',res_kept=True)
    module = getattr(a.ModalFusion,'roi_encoder',None)
    if module is None:
        return row
    idx = a.tst_idx
    last = module.last
    delta = last['delta'][idx]
    score,response = last['score'][idx],last['response'][idx]
    base = a._aarsp_base_fc[idx]
    valid = module.feature_indices
    common = module.text_features.mean(0,keepdim=True).expand(110,-1)
    common_score,_ = module.similarities(last['h'][idx],common)
    expected_text = context()['static']['banks'][dict(A2='CONSTANT',A3='ANONYMOUS',A4='NAME',A5='FUNCTION',A6='SHUFFLED',A7='FUNCTION')[config]].cuda()
    assert torch.equal(module.text_features,expected_text)
    assert torch.count_nonzero(delta[:,[195,225,232,236]])==0
    row.update(mean_degree=float(module.degree.mean()),input_numeric_abs_mean=float(module.stats[idx].abs().mean()),
               roi_message_abs_mean=float(last['message'][idx].abs().mean()),
               roi_updated_abs_mean=float(last['updated'][idx].abs().mean()),
               residual_abs_mean=float(delta.abs().mean()),
               residual_valid_abs_mean=float(delta[:,valid].abs().mean()),
               residual_max_abs=float(delta.abs().max()),
               residual_relative_l2=float((delta.norm(dim=1)/(base.norm(dim=1)+1e-12)).mean()),
               residual_nonzero_fraction=float(delta[:,valid].abs().gt(1e-8).float().mean()),
               residual_common_edge_fraction=float((delta[:,valid].mean(1).square().mean()/(delta[:,valid].square().mean()+1e-12)).sqrt()),
               score_mean=float(score.mean()),score_std=float(score.std(unbiased=False)),
               score_between_roi_std=float(score.std(dim=1,unbiased=False).mean()),
               score_roi_mean_std=float(score.mean(0).std(unbiased=False)),
               response_mean=float(response.mean()),response_std=float(response.std(unbiased=False)),
               response_roi_mean_std=float(response.mean(0).std(unbiased=False)),
               response_near_constant_fraction=float(response.std(dim=1,unbiased=False).lt(.001).float().mean()),
               common_score_rms=float(common_score.square().mean().sqrt()),
               distinctive_score_rms=float((score-common_score).square().mean().sqrt()),
               query_norm_mean=float(last['query_norm'][idx].mean()),alpha=float(module.alpha),
               beta=float(.3*module.alpha.tanh()),unknown_direct_delta_max=0.,frozen_text_unchanged=True)
    key = dict(A2='CONSTANT',A3='ANONYMOUS',A4='NAME',A5='FUNCTION',A6='SHUFFLED',A7='FUNCTION')[config]
    row.update({'clip_'+k:v for k,v in read_json(MANIFEST)['text_audit']['geometry'][key].items()})
    row.update({k+'_grad_max':v for k,v in tracker.items()})
    row.update({k+'_grad_valid':v>0 for k,v in tracker.items()})
    row.update(query_parameter_norm=float(module.query.weight.norm()),
               message_parameter_norm=float(module.message.weight.norm()),
               all_roi_values_finite=all(bool(torch.isfinite(v).all()) for v in last.values()))
    row.update(dict(zip(['score_p05','score_p50','score_p95'],
                        map(float,np.quantile(score.cpu().numpy(),[.05,.5,.95])))))
    assert all(not isinstance(v,float) or math.isfinite(v) for v in row.values())
    return row


def forbid_encoding():
    from CLIP.clip.model import CLIP
    def forbidden(*args,**kwargs):
        raise RuntimeError('Text encoding is forbidden during formal training')
    CLIP.encode_text = forbidden


def preflight():
    import torch
    import numpy as np
    from torch.nn import functional as F
    m = integrity(full=True)
    c = context()
    seed_rng(0)
    entry = rng()
    original = agent_for('A0',0,1,original=True)
    original_out = forward(original)
    assert torch.equal(original.ModalFusion.prompt_text_features.cpu(),c['static']['p0'])
    original_hash = weight_hash(original)
    original_rng = rng_hash(rng())
    forbid_encoding()
    counts, errors = {}, {}
    for config in CONFIGS:
        restore_rng(entry)
        c['clip_loaded'] = False
        a = agent_for(config,0,1)
        assert weight_hash(a)==original_hash and rng_hash(rng())==original_rng
        counts[config] = optimizer_audit(a,config)
        out = forward(a)
        if config=='A0':
            errors['A0_original'] = float((out-original_out).abs().max())
            assert torch.equal(out,original_out)
        elif config=='A1':
            no_spgs = out
            a1 = a
        else:
            errors[config+'_A1'] = float((out-no_spgs).abs().max())
            assert torch.equal(out,no_spgs) and float(a.ModalFusion.roi_encoder.alpha)==0.
            assert torch.count_nonzero(a.ModalFusion.roi_encoder.last['delta'])==0
        if config!='A1':
            del a
    # Other seeds: same entry produces the same trunk and branch, without
    # claiming identical later fold entries under independent early stopping.
    for seed in [0,1,2]:
        seed_rng(seed)
        state = rng()
        common,extra = None,None
        for config in CONFIGS:
            restore_rng(state)
            c['clip_loaded'] = False
            a = agent_for(config,seed,1)
            if common is None:
                common,after = weight_hash(a),rng_hash(rng())
            assert common==weight_hash(a) and after==rng_hash(rng())
            if config in CONFIGS[2:]:
                if extra is None:
                    extra = weight_hash(a,True)
                assert extra==weight_hash(a,True)
                # Train-mode zero residual parity including dropout RNG.
                a.ModalFusion.train()
                a1.ModalFusion.load_state_dict({k:v for k,v in a.ModalFusion.state_dict().items() if not k.startswith('roi_encoder.')})
                a1.ModalFusion.train()
                drop_rng = rng()
                with torch.no_grad():
                    h = a.feat.clone()
                    expected = a1.ModalFusion(h)[0]
                    restore_rng(drop_rng)
                    actual = a.ModalFusion(inject_roi(a,h))[0]
                assert torch.equal(expected,actual)
            del a
    c['clip_loaded'] = True
    gradient_checks = {}
    for config in CONFIGS[2:]:
        seed_rng(0)
        a = agent_for(config,0,1)
        tracked = {k:0. for k in gradients(a)}
        for step in range(2):
            with contextlib.redirect_stdout(io.StringIO()):
                grad = train_step(a)
            if step==0:
                assert grad['alpha']>0 and all(grad[k]==0 for k in grad if k!='alpha')
            for k,v in grad.items():
                tracked[k] = max(tracked[k],v)
        assert all(v>0 for v in tracked.values()), (config,tracked)
        out = forward(a)
        module = a.ModalFusion.roi_encoder
        direct = (F.normalize(module.query(module.last['h'][:8]),dim=-1)*module.text_features[None]).sum(-1)
        err = float((direct-module.last['score'][:8]).abs().max())
        assert err<1e-6, err
        grad_audit = diagnostics(a,config,0,1,out,tracked)
        assert grad_audit['unknown_direct_delta_max']==0 and grad_audit['all_roi_values_finite']
        gradient_checks[config] = dict(first_alpha_valid=True,two_step_max=tracked,
                                       cosine_algebra_max_error=err)
        if config=='A5':
            tested = a
        else:
            del a
    # Exact continuation of model + both Adam instances + all RNG.
    a = tested
    restore = dict(model=model_state(a),mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),rng=rng())
    path = STATE/'preflight_resume.pt'
    save_checkpoint(path,restore)
    with contextlib.redirect_stdout(io.StringIO()):
        train_step(a)
    expected = weight_hash(a)+weight_hash(a,True)
    expected_rng,expected_out = rng_hash(rng()),forward(a)
    saved = checkpoint(path)
    load_state(a,saved['model'])
    a.optimizer_MP0.load_state_dict(saved['mp'])
    a.optimizer_MF.load_state_dict(saved['mf'])
    restore_rng(saved['rng'])
    with contextlib.redirect_stdout(io.StringIO()):
        train_step(a)
    assert expected==weight_hash(a)+weight_hash(a,True) and expected_rng==rng_hash(rng())
    assert torch.equal(expected_out,forward(a))
    path.unlink()
    cf = counterfactual(a,0,1,forward(a))
    assert len(cf)==5 and cf[0]['flip_rate']==cf[0]['probability_abs_change']==0
    # Held-out labels cannot influence the original GFR or the ROI inputs.
    native = forward(a)
    labels = a.targ.clone()
    a.targ[a.tst_idx] = 1-a.targ[a.tst_idx]
    assert torch.equal(native,forward(a))
    a.targ.copy_(labels)
    module = a.ModalFusion.roi_encoder
    native_delta = module().clone()
    ends = module.endpoints.clone()
    module.endpoints.copy_(ends.flip(-1))
    assert torch.equal(module(),native_delta)
    module.endpoints.copy_(ends)
    h = a.feat.clone()
    injected = inject_roi(a,h)
    assert torch.equal(injected[:,:64],h[:,:64]) and torch.equal(injected[:,64:320][:,[195,225,232,236]],h[:,64:320][:,[195,225,232,236]])
    # A text perturbation propagates via a neighbor to a nonincident edge.
    original_delta = module().clone()
    adjacency = (c['static']['fixed']['adjacency'][0].numpy()!=0)
    chain = next((u,v,w) for u in range(110) for v in np.flatnonzero(adjacency[u])
                 for w in np.flatnonzero(adjacency[v]) if u!=w)
    u,v,w = chain
    target = np.flatnonzero(((ends.cpu().numpy()==[v,w]).all(1)) | ((ends.cpu().numpy()==[w,v]).all(1)))[0]
    modified = module.text_features.clone()
    modified[u] = modified[(u+37)%110]
    module.text_override = modified
    changed = module()
    propagation = float((changed[:,module.feature_indices[target]]-original_delta[:,module.feature_indices[target]]).abs().max())
    assert propagation>1e-10, ('neighbor propagation missing',chain,propagation)
    module.text_override = None
    assert torch.equal(module(),original_delta)
    checks = dict(status='PASS',parameter_counts=counts,initial_output_errors=errors,
                  identical_entry_trunk_and_new_initialization_all_seeds=True,
                  new_initialization_rng_isolated=True,train_mode_zero_residual_parity=True,
                  first_lazy_RN50_rng_preserved=True,gradient_checks=gradient_checks,
                  model_Adam_rng_resume_exact=True,unknown_residual_zero=True,first64_unmodified=True,
                  endpoint_order_invariant=True,neighbor_text_propagation_max=propagation,
                  counterfactual_REAL_exact=True,counterfactual_weights_unchanged=True,
                  held_out_labels_do_not_change_forward=True,
                  A6_derangement=all(i!=v for i,v in enumerate(read_json(EXP/'prompts.json')['a6_permutation'])),
                  A5_A6_text_set_exact=True,CLIP_frozen=True,preflight_utc=utc())
    integrity(full=True)
    assert sha(ROOT/'handoff.md')==m['protected_handoff_sha256']
    m['preflight'] = checks
    m['run_status'] = 'PREFLIGHT_PASS'
    save_json(MANIFEST,m)
    print('PREFLIGHT_PASS '+json.dumps({'initial_errors':errors,'parameters':counts,'neighbor_propagation':propagation}),flush=True)
    return checks


def lock(path):
    f = open(path,'a+')
    fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    return f


def worker(config,seed):
    import torch
    m = integrity()
    assert m['preflight']['status']=='PASS'
    task = f'{config}_seed{seed}'
    physical = int(os.environ['AARSP_PHYSICAL_GPU'])
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(physical)
    task_lock = lock(STATE/(task+'.lock'))
    c = context()
    forbid_encoding()
    result_path,save_path = STATE/(task+'.json'),STATE/(task+'.pt')
    sig = signature()
    if result_path.exists() and read_json(result_path)['status']=='COMPLETE':
        print('SKIP_COMPLETE '+task,flush=True)
        return
    saved = checkpoint(save_path) if save_path.exists() else None
    if saved:
        assert saved['signature']==sig
        result = saved['result']
        restore_rng(saved['rng'])
        c['clip_loaded'] = True
        start_fold = int(saved['fold'])+(1 if saved['phase']=='BETWEEN_FOLDS' else 0)
        print(f'RESUME {task} phase={saved["phase"]} fold={start_fold}',flush=True)
    else:
        seed_rng(seed)
        result = dict(signature=sig,task=task,config=config,seed=seed,physical_gpu=physical,
                      status='RUNNING',metrics=[],diagnostics=[],counterfactual=[],started_utc=utc(),seconds=0.)
        start_fold = 1
    print('WORKER '+task+' physical='+str(physical)+' logical=cuda:0 device='+torch.cuda.get_device_name(0),flush=True)
    for fold in range(start_fold,11):
        entry = rng()
        a = agent_for(config,seed,fold)
        optimizer_audit(a,config)
        initial,branch_initial = weight_hash(a),weight_hash(a,True)
        best_acc,best_epoch,best_model,wait,start = 0.,-1,None,0,0
        tracked = {k:0. for k in gradients(a)}
        elapsed_before = 0.
        if saved and saved['phase']=='EPOCH' and int(saved['fold'])==fold:
            load_state(a,saved['model'])
            a.optimizer_MP0.load_state_dict(saved['mp'])
            a.optimizer_MF.load_state_dict(saved['mf'])
            restore_rng(saved['rng'])
            best_acc,best_epoch,best_model,wait,start = saved['best_acc'],saved['best_epoch'],saved['best_model'],saved['wait'],saved['epoch']+1
            tracked = saved['tracker']
            entry,initial,branch_initial = saved['entry_rng'],saved['initial'],saved['branch_initial']
            elapsed_before = saved['fold_seconds']
        saved = None
        t0 = time.monotonic()
        print(f'FOLD_START {fold} entry_rng={rng_hash(entry)} initial={initial}',flush=True)
        last = start-1
        for epoch in range(start,450):
            print(f'EPOCH {fold}:{epoch} ',end='')
            grad = train_step(a)
            for k,v in grad.items():
                tracked[k] = max(tracked[k],v)
            # Original two eval forwards are identical in eval mode; reuse one.
            out = forward(a)
            val = float(out[a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean())
            print(f' val-acc={val:.12f}',flush=True)
            if val>best_acc:
                best_acc,best_epoch,best_model,wait = val,epoch,model_state(a),0
            else:
                wait += 1
            last = epoch
            stop = wait>200 or epoch==449
            if (epoch+1)%25==0 or stop:
                save_checkpoint(save_path,dict(signature=sig,phase='EPOCH',fold=fold,epoch=epoch,
                    model=model_state(a),mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),
                    rng=rng(),best_acc=best_acc,best_epoch=best_epoch,best_model=best_model,wait=wait,
                    tracker=tracked,entry_rng=entry,initial=initial,branch_initial=branch_initial,
                    fold_seconds=elapsed_before+time.monotonic()-t0,result=result))
            if stop:
                break
        assert best_model is not None
        continuation = rng()
        load_state(a,best_model)
        out = forward(a)
        met = metrics(out,a.tst_idx)
        assert abs(met['acc']-best_acc)<1e-12 and abs(met['ba']-met['hard_auc'])<1e-12
        seconds = elapsed_before+time.monotonic()-t0
        row = dict(config=config,seed=seed,fold=fold,physical_gpu=physical,**met,
                   best_epoch=best_epoch,epochs_executed=last+1,seconds=seconds,
                   selected_val_acc=best_acc,train_n=len(a.trn_idx),test_n=len(a.tst_idx),
                   initial_trunk_sha256=initial,initial_branch_sha256=branch_initial if config in CONFIGS[2:] else '',
                   entry_rng_sha256=rng_hash(entry),additional_parameters=38242 if config in CONFIGS[2:] else 0)
        result['metrics'].append(row)
        result['diagnostics'].append(diagnostics(a,config,seed,fold,out,tracked))
        if config=='A5':
            result['counterfactual'].extend(counterfactual(a,seed,fold,out))
        assert config not in CONFIGS[2:] or all(v>0 for v in tracked.values())
        restore_rng(continuation)
        result['seconds'] = sum(r['seconds'] for r in result['metrics'])
        result['status'] = 'COMPLETE' if fold==10 else 'RUNNING'
        result['updated_utc'] = utc()
        save_checkpoint(save_path,dict(signature=sig,phase='BETWEEN_FOLDS',fold=fold,rng=continuation,result=result))
        save_json(result_path,result)
        print('FOLD_COMPLETE '+json.dumps(row,ensure_ascii=False),flush=True)
        if fold==5 and sum(r['acc'] for r in result['metrics'])/5<.6:
            raise RuntimeError('Original fifth-fold low-ACC guard triggered')
        del a
        if fold==10:
            save_path.unlink(missing_ok=True)
    print('TASK_COMPLETE '+task,flush=True)
    task_lock.close()


def task_results():
    out = []
    for c,s,g in TASKS:
        p = STATE/f'{c}_seed{s}.json'
        if p.exists():
            r = read_json(p)
            assert r['signature']==signature() and r['physical_gpu']==g
            out.append(r)
    return out


def live_workers():
    out = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p/'cmdline').read_bytes().decode().split('\0')
            if str(Path(__file__).resolve()) in args and '--worker' in args:
                i = args.index('--worker')
                out[f'{args[i+1]}_seed{args[i+2]}'] = int(p.name)
        except (OSError,UnicodeError,ValueError,IndexError):
            pass
    return out


def refresh():
    m = read_json(MANIFEST)
    if m['run_status']=='COMPLETE' and not STATE.exists():
        return m
    results = task_results()
    folds = [r for t in results for r in t['metrics']]
    diag = [r for t in results for r in t['diagnostics']]
    cf = [r for t in results for r in t['counterfactual']]
    assert len(set((r['config'],r['seed'],r['fold']) for r in folds))==len(folds)
    for t in results:
        task = m['tasks'][t['task']]
        task.update(status=t['status'],folds=len(t['metrics']),seconds=t['seconds'])
    m['completed_folds'] = len(folds)
    m['updated_utc'] = utc()
    for name,rows in [('fold_metrics',folds),('diagnostics',diag),('counterfactual',cf)]:
        save_csv(RESULTS/(name+'.csv'),rows)
    save_json(MANIFEST,m)
    return m


def worker_env(gpu):
    return dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),
                AARSP_PHYSICAL_GPU=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1',
                PYTHONDONTWRITEBYTECODE='1')


def resources():
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,utilization.gpu',
                                   '--format=csv,noheader'],text=True)
    cards = []
    for line in raw.strip().splitlines():
        i,name,uuid,bus,total,used,util = [v.strip() for v in line.split(',')]
        cards.append(dict(index=int(i),name=name,uuid=uuid,pci_bus=bus,
                          total_mib=int(total.split()[0]),used_mib=int(used.split()[0]),utilization=int(util.split()[0])))
    stats = dict(line.split() for line in Path('/sys/fs/cgroup/memory/memory.stat').read_text().splitlines())
    return dict(utc=utc(),monotonic=time.monotonic(),gpu=cards,
                cgroup_rss_gib=int(stats['rss'])/2**30,cgroup_cache_gib=int(stats['cache'])/2**30,
                cgroup_memory_gib=int(Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text())/2**30,
                cpu_usage_ns=int(Path('/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage').read_text()))


def schedule(workers_per_gpu=4):
    if MANIFEST.exists() and read_json(MANIFEST).get('run_status')=='COMPLETE':
        validate_complete()
        print('ALREADY_COMPLETE: 24 tasks / 240 folds; no training restarted',flush=True)
        return
    prepare()
    scheduler_lock = lock(STATE/'scheduler.lock')
    if read_json(MANIFEST)['preflight'] is None:
        print('PREFLIGHT_START',flush=True)
        with open(LOGS/'preflight.log','a') as f:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'preflight'],env=worker_env(0),cwd=EXP,stdout=f,stderr=subprocess.STDOUT,check=True)
    m = read_json(MANIFEST)
    assert m['preflight']['status']=='PASS'
    m.update(run_status='RUNNING',workers_per_gpu=workers_per_gpu)
    m.setdefault('started_unix',time.time())
    m.setdefault('started_utc',utc())
    m['launch_resource_snapshot'] = subprocess.check_output(['nvidia-smi'],text=True)
    save_json(MANIFEST,m)
    attempts = {}
    active = {t:dict(pid=pid,process=None,gpu=m['tasks'][t]['physical_gpu']) for t,pid in live_workers().items()}
    print('SCHEDULER_START '+utc(),flush=True)
    last_refresh = 0.
    last_resource = None
    while True:
        for task,info in list(active.items()):
            process = info['process']
            if process is None:
                alive = Path('/proc') .joinpath(str(info['pid'])).exists()
                code = None if alive else 0
            else:
                code = process.poll()
            if code is not None:
                if 'log' in info:
                    info['log'].close()
                del active[task]
                if code!=0:
                    attempts[task] = attempts.get(task,0)+1
                    print(f'WORKER_RESTART {task} exit={code}; resume committed checkpoint',flush=True)
                    if attempts[task]>=3:
                        raise RuntimeError('Repeated worker failure: '+task)
        done = {t['task'] for t in task_results() if t['status']=='COMPLETE'}
        for gpu in [0,1]:
            for config,seed,assigned in TASKS:
                task = f'{config}_seed{seed}'
                if assigned!=gpu or task in done or task in active:
                    continue
                if sum(v['gpu']==gpu for v in active.values())>=workers_per_gpu:
                    break
                log = open(LOGS/(task+'.log'),'a')
                log.write('\nATTEMPT_START '+utc()+'\n')
                log.flush()
                process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',config,str(seed)],
                    cwd=EXP,env=worker_env(gpu),stdout=log,stderr=subprocess.STDOUT)
                active[task] = dict(pid=process.pid,process=process,gpu=gpu,log=log)
                print(f'LAUNCH {task} GPU{gpu} pid={process.pid}',flush=True)
        if time.monotonic()-last_refresh>20 or not active:
            m = refresh()
            if last_resource is None or time.monotonic()-last_resource['monotonic']>=60:
                sample = resources()
                if last_resource:
                    sample['cpu_cores_used'] = (sample['cpu_usage_ns']-last_resource['cpu_usage_ns'])/1e9/(sample['monotonic']-last_resource['monotonic'])
                peaks = m.setdefault('runtime_resource_peaks',dict(rss_gib=0.,gpu0_mib=0,gpu1_mib=0,cpu_cores_used=0.))
                peaks['rss_gib'] = max(peaks['rss_gib'],sample['cgroup_rss_gib'])
                peaks['cpu_cores_used'] = max(peaks['cpu_cores_used'],sample.get('cpu_cores_used',0.))
                for card in sample['gpu']:
                    key = 'gpu'+str(card['index'])+'_mib'
                    peaks[key] = max(peaks[key],card['used_mib'])
                m['gpu_physical_mapping'] = [{k:v for k,v in card.items() if k not in ['used_mib','utilization']} for card in sample['gpu']]
                save_json(MANIFEST,m)
                print('RESOURCE '+json.dumps(sample),flush=True)
                last_resource = sample
            print(f'PROGRESS tasks={len(done)}/24 folds={m["completed_folds"]}/240 active={len(active)}',flush=True)
            last_refresh = time.monotonic()
        if len(done)==24 and not active:
            break
        time.sleep(2)
    m = refresh()
    m.update(completed_utc=utc(),wall_seconds=time.time()-m['started_unix'],
             failures_and_restarts=attempts,run_status='TRAINING_COMPLETE')
    save_json(MANIFEST,m)
    validate_complete()
    report()
    append_handoff()
    assert not live_workers()
    integrity(full=True)
    m = read_json(MANIFEST)
    link = EXP/'state'
    if link.is_symlink():
        link.unlink()
    scheduler_lock.close()
    shutil.rmtree(STATE)
    m.update(run_status='COMPLETE',temporary_state_cleaned=True,no_training_processes=True,
             handoff_sha256=sha(ROOT/'handoff.md'),report_sha256=sha(RESULTS/'report.md'),
             results_sha256={n:sha(RESULTS/(n+'.csv')) for n in ['fold_metrics','diagnostics','counterfactual']})
    save_json(MANIFEST,m)
    print('COMPLETE 24/24 tasks 240/240 folds; temporary state cleaned',flush=True)


def validate_complete():
    import numpy as np
    m = integrity(full=True)
    rows = read_csv(RESULTS/'fold_metrics.csv')
    ds = read_csv(RESULTS/'diagnostics.csv')
    cf = read_csv(RESULTS/'counterfactual.csv')
    keys = {(c,s,f) for c in CONFIGS for s in range(3) for f in range(1,11)}
    row_keys = {(r['config'],int(r['seed']),int(r['fold'])) for r in rows}
    assert len(rows)==240 and row_keys==keys
    assert len(ds)==240 and {(r['config'],int(r['seed']),int(r['fold'])) for r in ds}==keys
    assert len(cf)==150 and {(int(r['seed']),int(r['fold']),r['condition']) for r in cf}=={(s,f,k) for s in range(3) for f in range(1,11) for k in ['REAL','SHUFFLED','CONSTANT','OFF','SELF_ONLY']}
    for r in rows+cf:
        for k in METRICS:
            assert 0<=float(r[k])<=1 and math.isfinite(float(r[k]))
        assert abs(float(r['ba'])-float(r['hard_auc']))<1e-12
    for r in rows:
        assert abs(float(r['selected_val_acc'])-float(r['acc']))<1e-12
        assert 0<=int(r['best_epoch'])<int(r['epochs_executed'])<=450
    for r in ds:
        assert str(r['prediction_finite']).lower()=='true' and str(r['res_kept']).lower()=='true'
        if r['config'] in CONFIGS[2:]:
            assert float(r['unknown_direct_delta_max'])==0
            assert all(str(r[k+'_grad_valid']).lower()=='true' for k in ['numerical','query','message','self_transform','decoder','alpha'])
    for gpu in [0,1]:
        assert sum(int(r['physical_gpu'])==gpu for r in rows)==120
        assert sum(t['physical_gpu']==gpu for t in m['tasks'].values())==12
    # A0 is an actual retraining, compared to both historical native baselines.
    parity = {}
    for number,config in [(13,'F0'),(14,'E00')]:
        folder = next(ROOT.joinpath('experiment').glob('experiment'+str(number)+'_*'))
        reference = {(int(r['seed']),int(r['fold'])):r for r in read_csv(folder/'results/fold_metrics.csv') if r['config']==config}
        errors = {}
        for k in METRICS:
            historical_key = 'balanced_accuracy' if number==14 and k=='ba' else k
            errors[k] = max(abs(float(r[k])-float(reference[(int(r['seed']),int(r['fold']))][historical_key])) for r in rows if r['config']=='A0')
        parity['experiment'+str(number)] = dict(max_errors=errors,exact=all(v<1e-12 for v in errors.values()))
    initial_pairs = [[r for r in rows if int(r['seed'])==s and int(r['fold'])==f] for s in range(3) for f in range(1,11)]
    initialized_equal = sum(len({r['initial_trunk_sha256'] for r in group})==1 for group in initial_pairs)
    # No failure/partial attempt contributes duplicate or uncommitted metrics.
    assert not live_workers()
    m.update(validation=dict(status='PASS',folds=240,diagnostics=240,counterfactual=150,
                             gpu_folds={'0':120,'1':120},common_initialization_equal_pairs=initialized_equal,
                             continuous_rng_no_fold_reseed=True,baseline_parity=parity))
    save_json(MANIFEST,m)
    return rows,ds,cf


def aggregates(rows):
    import numpy as np
    out = {}
    for config in CONFIGS:
        out[config] = {}
        for k in METRICS:
            means = [float(np.mean([float(r[k]) for r in rows if r['config']==config and int(r['seed'])==s])) for s in range(3)]
            out[config][k] = dict(mean=float(np.mean(means)),sd=float(np.std(means)),seeds=means)
    return out


def paired(rows,left,right,key):
    import numpy as np
    index = {(r['config'],int(r['seed']),int(r['fold'])):r for r in rows}
    diffs = [float(index[(left,s,f)][key])-float(index[(right,s,f)][key]) for s in range(3) for f in range(1,11)]
    return dict(mean=float(np.mean(diffs)),sd=float(np.std([np.mean(diffs[i*10:(i+1)*10]) for i in range(3)])),
                seeds=[float(np.mean(diffs[i*10:(i+1)*10])) for i in range(3)],
                win=sum(d>1e-12 for d in diffs),tie=sum(abs(d)<=1e-12 for d in diffs),loss=sum(d<-1e-12 for d in diffs))


def scientific_judgment(rows):
    a = aggregates(rows)
    means = {k:paired(rows,'A5','A0',k)['mean'] for k in ['acc','ba','probability_auc']}
    improved = all(v>1e-12 for v in means.values())
    def robust(left,right):
        return all(all(v>1e-12 for v in paired(rows,left,right,k)['seeds']) for k in ['acc','ba','probability_auc'])
    semantic = all(robust('A5',c) for c in ['A2','A3','A6'])
    structural = robust('A5','A7') and robust('A2','A1')
    return dict(mean_classification_improved=improved,classification_seed_consistent=robust('A5','A0'),
                semantic_support='DESCRIPTIVE_SUPPORT' if semantic else 'NOT_SUPPORTED',
                structural_support='DESCRIPTIVE_SUPPORT' if structural else 'NOT_SUPPORTED',
                advance_global_alignment=improved and robust('A5','A0') and semantic and structural,
                best={k:[c for c in CONFIGS if abs(a[c][k]['mean']-max(a[v][k]['mean'] for v in CONFIGS))<1e-12] for k in ['acc','ba','probability_auc']})


def result_table(rows):
    a = aggregates(rows)
    lines = ['|组|设计|ACC %|BA %|Hard-AUC %|Prob-AUC %|SEN-ASD %|SPE-TD %|',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for i,c in enumerate(CONFIGS):
        lines.append('|'+c+'|'+NAMES[i]+'|'+'|'.join(f"{a[c][k]['mean']*100:.4f}±{a[c][k]['sd']*100:.4f}" for k in METRICS)+'|')
    return lines


def report():
    import numpy as np
    rows,ds,cf = validate_complete()
    m = read_json(MANIFEST)
    a = aggregates(rows)
    judgment = scientific_judgment(rows)
    lines = ['# 实验十五：AARSP 解剖锚定ROI局部语义提示学习','',
             '工程验收：PASS。8组、24任务、240折；240条最佳checkpoint诊断、150条A5同checkpoint反事实。全部结果来自本轮真实训练，没有复制历史指标。','',
             '## 全部正式结果','',
             '各seed先求10折均值，再对三个seed求均值与population SD（ddof=0）。ASD=0；连续AUC使用exp(log-softmax)第0列；原one-hot硬预测AUC保留，二分类下等于BA。','']
    lines += result_table(rows)
    lines += ['', '## 每seed十折均值','', '|组|seed|ACC %|BA %|Hard-AUC %|Prob-AUC %|SEN %|SPE %|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in CONFIGS:
        for s in range(3):
            lines.append('|'+c+'|'+str(s)+'|'+'|'.join(f"{a[c][k]['seeds'][s]*100:.5f}" for k in METRICS)+'|')
    lines += ['', '## 事先规定的配对比较','',
              '差值单位为百分点；SD为三个seed差值的population SD。相同seed/fold配对，但后续折保留各配置连续随机流，因此不声称所有配对模型均有同一初始化。胜/平/负以逐折差值和1e-12平局容差统计。','',
              '|比较|指标|均值±seed SD pp|seed0/1/2 pp|胜/平/负|',
              '|---|---|---:|---|---|']
    for left,right in PAIRS:
        for k in METRICS:
            v = paired(rows,left,right,k)
            lines.append(f"|{left}−{right}|{k}|{v['mean']*100:+.5f}±{v['sd']*100:.5f}|"+'/'.join(f'{x*100:+.5f}' for x in v['seeds'])+f"|{v['win']}/{v['tie']}/{v['loss']}|")
    lines += ['', '## ROI机制诊断','',
              '统计最佳checkpoint的测试折人员；梯度为整个训练过程的最大有限范数，不能解释为临床有效性。原型始终冻结，不存在训练文本投影；查询、数值编码及图层仍可形成共有方向响应。','',
              '|组|alpha|beta|有效delta绝对均值|相对L2|消息绝对均值|跨ROI score SD|ROI均值response SD|共同/差异score RMS|非零比例|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for c in CONFIGS[2:]:
        selected = [r for r in ds if r['config']==c]
        val = lambda k:float(np.mean([float(r[k]) for r in selected]))
        lines.append(f"|{c}|{val('alpha'):+.6f}|{val('beta'):+.6f}|{val('residual_valid_abs_mean'):.6e}|{val('residual_relative_l2'):.6f}|{val('roi_message_abs_mean'):.6e}|{val('score_between_roi_std'):.6f}|{val('response_roi_mean_std'):.6f}|{val('common_score_rms'):.6f}/{val('distinctive_score_rms'):.6f}|{val('residual_nonzero_fraction'):.6f}|")
    lines += ['', '|文本bank|CLIP平均cos|cos 5/50/95%|有效秩|中心化有效秩|完全重复行数|', '|---|---:|---|---:|---:|---:|']
    for key,g in m['text_audit']['geometry'].items():
        lines.append(f"|{key}|{g['pair_cos_mean']:.6f}|{g['pair_cos_p05']:.6f}/{g['pair_cos_p50']:.6f}/{g['pair_cos_p95']:.6f}|{g['effective_rank']:.3f}|{g['centered_effective_rank']:.3f}|{g['exact_duplicate_rows']}|")
    near_zero = sum(float(r.get('residual_valid_abs_mean') or 0)<1e-8 for r in ds if r['config'] in CONFIGS[2:])
    near_constant = sum(float(r.get('score_between_roi_std') or 0)<.001 for r in ds if r['config'] in CONFIGS[2:])
    lines += ['',f'180个ROI分支折中，近零残差（有效delta绝对均值<1e-8）为{near_zero}折，跨ROI score SD<.001为{near_constant}折。所有数值编码、查询、消息、自身变换、解码和alpha均有非零有限梯度；四条未知连接直接delta恒为0。CLIP余弦固定，低有效秩或共同score强不能自动称为医学语义无效/有效。','',
              '## A5同checkpoint反事实','',
              '每折五个条件使用同一最佳checkpoint，参数、受试者间GFR、原始输入和checkpoint选择不变。SHUFFLED复用A6置换；CONSTANT复用A2文本；SELF_ONLY把邻居传播改成同参数自身映射；OFF仅令新增delta为0。REAL逐值等于正常测试，全部条件后恢复REAL且权重未变。','',
              '|条件|ACC %|BA %|Prob-AUC %|翻转率 %|ASD概率绝对变化|delta绝对变化|ROI响应绝对变化|ACC/BA/Prob-AUC差 pp|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for cond in ['REAL','SHUFFLED','CONSTANT','OFF','SELF_ONLY']:
        rr = [r for r in cf if r['condition']==cond]
        val = lambda k:float(np.mean([float(r[k]) for r in rr]))
        lines.append(f"|{cond}|{val('acc')*100:.5f}|{val('ba')*100:.5f}|{val('probability_auc')*100:.5f}|{val('flip_rate')*100:.5f}|{val('probability_abs_change'):.6e}|{val('residual_abs_change'):.6e}|{val('roi_response_abs_change'):.6e}|"+'/'.join(f"{val(k+'_delta')*100:+.5f}" for k in ['acc','ba','probability_auc'])+'|')
    lines += ['', '## 科学判断','',f'最高均值：ACC {"/".join(judgment["best"]["acc"])}；BA {"/".join(judgment["best"]["ba"])}；Probability-AUC {"/".join(judgment["best"]["probability_auc"])}。', '',
              '分类与医学归因分别判断：A5相对A0的ACC/BA/Probability-AUC均值差为'+'/'.join(f"{paired(rows,'A5','A0',k)['mean']*100:+.5f}pp" for k in ['acc','ba','probability_auc'])+'。',
              f'三指标均值同时提高：{judgment["mean_classification_improved"]}；三个seed方向全部一致：{judgment["classification_seed_consistent"]}。',
              f'正确ROI语义相对常量A2、匿名A3和错配A6的共同证据：{judgment["semantic_support"]}；数值图/邻居结构必要性：{judgment["structural_support"]}。',
              '判定规则预先保守采用ACC、BA、连续AUC三项及三seed方向，属于指定协议下的描述性证据，不代表统计显著性或总体效应为零。A5−A4额外功能文本的收益、A5−A6正确对应和A5−A7邻居传播应分别查看，不能按单一ACC冠军替代。',
              '若正确功能文本不能稳定超过错配、匿名或常量控制，任何A5−P0的提升仍可能来自普通ROI图、身份编码、数值调制和新增容量。若文本替换改变预测，也只说明敏感，不说明医学语义正确。',
              ('建议：有描述性证据支持另行立项研究ROI条件化全局疾病对齐，仍需独立验证。' if judgment['advance_global_alignment'] else '建议：本轮证据不足以进入全局ASD/TD疾病语义对齐；保留全部对照与负结果，不自动扩展模型或改写文本。'), '',
              '## 实现、论文与历史审计','',readme_audit(),'',
              '## 运行、复现与限制','',
              f'每卡12任务/120折，最多4 workers、每worker2线程。正式墙钟时间{m.get("wall_seconds",0)/60:.2f}分钟。资源与GPU物理映射见manifest及scheduler.log；启动时其他GPU任务只作资源记录，未操作其他进程。',
              f'CPU cgroup上限32核，内存128 GiB；审计时约11 GiB RSS与约115 GiB可回收文件缓存。实际设备为PPU-ZW810E，每卡98304 MiB，使用其CUDA兼容PyTorch接口，不声称是NVIDIA显卡。',
              f'失败/恢复：{json.dumps(m.get("failures_and_restarts",{}),ensure_ascii=False)}。每25 epoch及结束保存包含model、两Adam、best model、早停计数和完整RNG的带SHA256校验checkpoint；每折提交后不重训。最新完整状态唯一，CSV无重复/失败attempt混入。',
              f'A0历史parity：{json.dumps(m["validation"]["baseline_parity"],ensure_ascii=False)}。相同seed/fold八组正式初始化均一致的配对数为{m["validation"]["common_initialization_equal_pairs"]}/30；其余分叉由各组训练与早停消耗不同RNG导致，不每折播种，不恢复A0入口。',
              '原val=test、共享标准化和历史RFE按任务保留，测试参与checkpoint选择；不是独立无泄漏泛化估计。三seed共享相同fold，30条记录不是30个独立数据集。A2与其他文本bank几何分布不同，A7改变传播规则，控制并非所有方面完全单因素隔离。',
              '固定252条已选连接构成稀疏图，不是完整HO全脑连接；ANAT/FUNC为质量指标。保守功能描述非专家临床认证，CLIP为通用图文编码器；ROI响应不能当成ASD病灶或因果生物标志物。',
              '根源码与实验一至十四逐文件SHA256核验保持不变。源码副本、CLIP缓存及checkpoint只在临时state，完成后清理；run.py、roi_semantic.py、可应用diff、原版本/数据/映射哈希足以重建。未执行git commit/push。', '',
              f'重跑：`python {EXP}/run.py all`；恢复：`python {EXP}/run.py resume`；刷新：`python {EXP}/run.py refresh`；报告：`python {EXP}/run.py report`。']
    atomic(RESULTS/'report.md','\n'.join(lines)+'\n')
    m = read_json(MANIFEST)
    m['scientific_judgment'] = judgment
    save_json(MANIFEST,m)
    return judgment


def readme_audit():
    return '''实际阅读三份本地论文正文、架构图、关键公式与消融：SPromptGL的GFR为受试者间四关系图，CAM对应VariLengthInputLayer与EncodeLayer；SPGS为MLP文本原型与4×4×9潜在Token的cosine→softmax→max；RES为局部特征与AvgPool全局特征的逐维softmax交互后加回x。停止SPGS只跳过该标量加权，保留整个RES段，门控等价1，绝非0.5。

SPromptGL Fig.2 ABIDE从86.48%经GFR到91.73%、RES到92.08%、SPGS到92.30%；主要增量来自GFR，SPGS约+0.22pp ACC。BrainPrompt Eq.2将真实AAL116 ROI身份/功能的冻结LLM编码加到节点后再消息传播；其ABIDE ROI单模块约+1.67pp ACC，但用了另一队列、图谱和验证协议。SABER Eq.1在AAL节点加文本投影再全局注意力，Eq.8–13在图级用Graph-CLS检索患者文本、残差注入及疾病监督；表I full/lite的ABIDE ACC为71.58/69.82%，AUC为70.40/72.23%，语义并非全指标提升。本文只迁移真实节点锚定和邻居传播思想，不复制其AAL、LLM、超图、患者或疾病头。来源：[SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[BrainPrompt](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf)、本地SABER/arXiv:2607.01901v1。

本地六核心源码与GitHub main 9de3b1561a6ac107b0f4dad8ce01df74638c0a4c逐字节一致；本地HEAD仍为1167dd392837a02eb4f83481ae49f02607a9dae5。实际保留原GFR中combined_feat覆盖分支以及ALoss未加入simple-2最终损失的行为，不顺手修复旧模型。原加权NLL与四图GraphConstructLoss平均值、两Adam更新顺序、450/200、ACC严格提高选checkpoint和第5折低ACC防护均保留。

实验十一旧映射表roi_i_name/roi_j_name仍含历史UNKNOWN，实际名称必须按零基端点关联新版roi_111_labels.csv，并核验name_verified/column_order_verified。110节点、252边、平均度4.581818，度范围1–12；未知82列/atlas3455对应特征195/225/232/236，不进入ROI图且直接delta恒0。连接是默认LedoitWolf收缩相关后的Fisher-z，消息权重固定tanh(z)，不误称原始Pearson r。

实验六signed门控、七双主题潜在Token残差没有稳定超过P1；实验十二逐边残差为独立验证单fold，不能与本轮直接比较历史ACC；实验十三原协议下的静态和患者LOW/MID/HIGH逐边对照均未支持正确语义；实验十四文本改写/对应关系差异极小，疾病头与原预测相关约.961；实验九E9疾病改善经实验十ADHD/中性文本控制未证明ASD语义必要性。本轮没有重做这些数学结构，也没有疾病/对比/正交损失。

实现入口在隔离model.py中GFR之后、PLN之前，训练与评价均调用inject_roi；只有H[:,64:320]增加ROI delta。ROI节点先融合所有关联真实边的五个统计，经文本条件化后跨真实邻居传播，节点更新再用于所有相连边，已验证单ROI文本影响非直接关联连接，因此不是独立逐边重加权。CLIP原型冻结在1024维，学习32→1024数值查询；利用WqᵀWq计算归一化分母，避免巨大查询张量，预检核对与直接cosine公式相同。

“所有折同初始化”与“各配置原始连续十折随机流”在早停轮数不同后不能同时满足。遵从本任务明确保留连续随机流的要求：每个config×seed仅起始播种一次，新增初始化在fork_rng隔离；预检证明相同入口下全部旧参数一致，新参数在A2–A7一致；不把随后分叉隐瞒为同初始化实验。与实验十三/十四的A0入口锚定策略明确区分。'''


def write_readme(m):
    text = ['# 实验十五 AARSP','',
            '研究问题：真实HO ROI解剖功能文本与患者内连接传播是否优于常量、匿名、错配和self-only控制？仅研究局部ROI语义，无全局疾病头。','',
            '|组|定义|','|---|---|']
    text += [f'|{c}|{NAMES[i]}|' for i,c in enumerate(CONFIGS)]
    text += ['', '## 冻结数学机制','',
             'z为原CSV的252个可信Fisher-z连接；110个ROI按实验十一已验证顺序。每人每节点统计为mean(z)、mean(abs(z))、population std(z)、max(abs(z))、degree/max(degree)。单边std=0。原始数值不按标签加工。','',
             '```text',
             'h_i = GELU(Linear32(GELU(Linear5→32(stat_i))))',
             's_i = normalize(Wq h_i) · frozen_unit_CLIP_i',
             'r_i = 0.5*tanh(s_i/0.2);  htilde_i = (1+r_i)*h_i',
             'A_ij = tanh(z_ij)/degree_i on the 252 undirected edges',
             'm_i = sum_j A_ij Wmessage htilde_j (A7: m_i=Wmessage htilde_i)',
             'g_i = GELU(Wself htilde_i + bself + m_i)',
             'edge_context = [g_i+g_j, abs(g_i-g_j), z_ij]',
             'd_ij = Linear32→1(GELU(Linear65→32(edge_context)))',
             'beta = 0.3*tanh(alpha), alpha initialized 0',
             'delta_ij = beta*tanh(d_ij); scatter 252 to 256, unknown four = 0',
             'Hnew[:,64:320] = H[:,64:320] + delta; Hnew[:,:64]=H[:,:64]',
             '```','',
             'A2–A7新增38,242参数：数值MLP1,248；查询32,768；消息1,024；自身变换1,056；边解码2,145；alpha 1。原参数1,462,957全部注册保留。全部Adam lr .0038、weight_decay .11；新组与原组设置相同，没有额外正则化搜索。原my_weight_init完成后追加ROI模块，fork_rng(150000+seed)隔离，只有alpha零初始化；不同时零化解码器导致断梯度。','',
             '缓存静态患者节点统计和邻接，冻结RN50文本只编码一次；前向仅张量化MLP/矩阵乘法，无患者×边Python循环。相似度采用等价Gram公式，不含可学习文本MLP；温度与幅值预先固定，不根据测试调整。A6复用A5全部向量的无固定点置换。A7保留消息映射参数，替换为自身传播。','',
             '## 论文、源码和历史审计','',readme_audit(),'',
             '## 运行与恢复','',
             f'`python {EXP}/run.py all` 完成prepare→预检→双GPU调度→240折→150反事实→中文报告→handoff→清理。',
             f'`python {EXP}/run.py resume` 恢复完整model/Adam/早停/RNG状态；完成折不重训。',
             f'`python {EXP}/run.py prepare`；`python {EXP}/run.py preflight`；`python {EXP}/run.py refresh`；`python {EXP}/run.py report`。','',
             '24个独立config×seed进程，每个按原顺序连续10折；交错分配两卡各12任务/120折，最多4并行/卡、2线程/进程，不回退CPU。保存完整任务日志；仅最新原子提交状态进入CSV，临时state位于本机/tmp，完成删除。已COMPLETE后all/resume只验收，不启动新训练。','',
             '全部八组固定，原StratifiedKFold(10,shuffle=True,random_state=10)，val=test、原ACC checkpoint、450 epoch/200 patience、原全图训练监督边界。保持指定历史协议，不替换成新的严格CV。已知共享标准化/RFE及val=test限制在最终报告单列；新文本不是个体诊断或ASD病理声明。','',
             '临时源码由根目录六文件复制；仅network.py/model.py应用真实integration_patch.diff，运行时辅助函数来自run.py，核心ROI实现来自roi_semantic.py。复核必须同时读基线、diff、两个新源码和manifest，不能仅看diff。原目录与历史目录均只读。','',
             '最终文件：本README、run.py、roi_semantic.py、prompts.json、integration_patch.diff、manifest.json；results三份逐折CSV与report.md；24任务日志、scheduler/preflight日志。无需永久源码副本、权重或受试者级预测。未自动git commit/push。']
    atomic(EXP/'README.md','\n'.join(text)+'\n')


def append_handoff():
    m = read_json(MANIFEST)
    rows = read_csv(RESULTS/'fold_metrics.csv')
    judgment = scientific_judgment(rows)
    marker = '## 实验十五：AARSP 解剖锚定ROI局部语义提示学习'
    path = ROOT/'handoff.md'
    old = path.read_text()
    if marker in old:
        old = old.split('\n'+marker,1)[0].rstrip()+'\n'
    else:
        assert sha(path)==m['protected_handoff_sha256'], 'handoff changed externally; preserve and inspect'
    chapter = [marker,'',f'状态COMPLETE；8组24任务240折，240诊断、150个A5同checkpoint反事实。GPU0/GPU1各12任务120折，4 workers/card、2线程/worker；正式耗时{m["wall_seconds"]/60:.2f}分钟，完成时间{m["completed_utc"]}。','',
               '目的：把原无真实ROI身份的局部SPGS替换为HO患者内图语义；GFR输出后、PLN输入前只增加Correlation段残差。110节点、252可信连接，未知特征195/225/232/236直接增量为0。数值统计→32维节点→冻结1024维CLIP与可学习查询匹配→一层连接权重消息→对称边解码；beta=.3*tanh(alpha)、alpha0。A2–A7各新增38,242参数，Adam .0038/.11，原参数与CAM/RES/分类器保持原计算。','',
               '区别：实验十二/十三为独立逐边静态/LOW-MID-HIGH残差；十四为潜在Token与全局疾病头。本轮用真实HO节点聚合并跨邻居传播，未复制AAL、超图、全局疾病/额外损失或措辞搜索。八组A0原P0、A1关闭SPGS保留RES、A2常量数值图、A3匿名身份、A4真名称、A5真名称+功能、A6固定无固定点错配、A7自身消息。','',
               '协议：原871人/48-6-10-256、固定RFE/输入、10折seed0/1/2、val=test、450/200、ACC严格提高选checkpoint及原Hard-AUC。每config×seed起始播种一次并连续十折，新增初始化fork_rng；相同入口下主干一致，后续各组因早停自然分叉，不恢复A0折入口。根源码与实验一至十四494个保护文件哈希均保持不变。','',
               f'本地HEAD `{m["git_head"]}`，GitHub核验main `{m["github_audit"].get("remote_main","unavailable")}`；代码、三论文、数据、映射、CLIP、Prompt、补丁与split SHA256见`{MANIFEST}`。','']
    chapter += result_table(rows)
    chapter += ['', '关键比较（ACC/BA/Hard-AUC/Prob-AUC/SEN/SPE，百分点）：']
    for control in ['A0','A2','A3','A4','A6','A7']:
        chapter.append('A5−'+control+'：'+'/'.join(f"{paired(rows,'A5',control,k)['mean']*100:+.5f}" for k in METRICS)+'pp；三seed及胜平负见report。')
    chapter += ['', '反事实：']
    cf = read_csv(RESULTS/'counterfactual.csv')
    for condition in ['SHUFFLED','CONSTANT','OFF','SELF_ONLY']:
        rr = [r for r in cf if r['condition']==condition]
        mean = lambda k:sum(float(r[k]) for r in rr)/len(rr)
        chapter.append(f"{condition}：翻转{mean('flip_rate')*100:.5f}%，ASD概率绝对变化{mean('probability_abs_change'):.6e}；ACC/BA/Prob-AUC相对REAL差"+'/'.join(f"{mean(k+'_delta')*100:+.5f}" for k in ['acc','ba','probability_auc'])+'pp。')
    chapter += ['', f'科学判断：三分类指标均值同时提高={judgment["mean_classification_improved"]}，三个seed方向一致={judgment["classification_seed_consistent"]}；医学语义必要性={judgment["semantic_support"]}；结构必要性={judgment["structural_support"]}。非零梯度、原型几何和替换敏感性不能代替正确/错配、匿名和常量的重训对照。',
                ('建议另行研究ROI条件化全局疾病对齐，当前证据仍为原协议下描述性结果。' if judgment['advance_global_alignment'] else '暂不建议进入ROI条件化全局ASD/TD疾病对齐；保留完整结果，不自动扩展或调参。'),
                f'失败与恢复：{json.dumps(m.get("failures_and_restarts",{}),ensure_ascii=False)}；已完成折不重训，model/两Adam/完整RNG恢复预检通过，无失败attempt/重复指标混入；完成后清理临时源码、文本缓存和checkpoint，无残留本实验训练进程。',
                '限制：val=test、共享上游标准化/RFE及三seed复用fold保留；A2文本几何、A7传播规则不同；文本功能描述保守且非临床专家认证，ROI响应不是ASD因果脑区。',
                f'目录 `{EXP}`；报告 `{RESULTS}/report.md`；manifest `{MANIFEST}`。运行 `python {EXP}/run.py all`，恢复 `python {EXP}/run.py resume`。未执行git commit/push。']
    atomic(path,old+'\n\n'+'\n'.join(chapter)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',default='all',choices=['prepare','preflight','all','resume','refresh','report'])
    parser.add_argument('--worker',nargs=2,metavar=('CONFIG','SEED'))
    parser.add_argument('--workers-per-gpu',type=int,default=4)
    args = parser.parse_args()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
    os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key] = '2'
    if args.worker:
        return worker(args.worker[0],int(args.worker[1]))
    if args.command=='prepare':
        prepare()
        print('PREPARED',flush=True)
    elif args.command=='preflight':
        prepare()
        preflight()
    elif args.command in ['all','resume']:
        schedule(args.workers_per_gpu)
    elif args.command=='refresh':
        m = refresh()
        print(m['run_status'],m['completed_folds'])
    else:
        report()


if __name__=='__main__':
    main()
