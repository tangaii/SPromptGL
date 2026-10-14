#!/usr/bin/env python3
"""RSPR: ten frozen configurations, original continuous ten-fold protocol."""
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
REFERENCE15 = ROOT/'experiment/experiment15_anatomically_anchored_roi_prompt'
PROMPTS = REFERENCE15/'prompts.json'
CF_CONDITIONS = ['REAL','SHUFFLED','ANONYMOUS','CONSTANT','SELF_ONLY','OFF']
COUNTS = {f'B{i}':0 if i==0 else 38242 if i<5 else 39201 for i in range(10)}
STATE = Path('/tmp') / ('spromptgl16-state-' + hashlib.sha256(str(EXP).encode()).hexdigest()[:12])
SOURCES = STATE / 'sources'
MANIFEST = EXP / 'manifest.json'
RESULTS = EXP / 'results'
LOGS = EXP / 'logs'
CORE = ['main.py', 'model.py', 'network.py', 'layers.py', 'utils.py', 'ABIDE-simple-2-concat-weighted-cosine.sh']
MAPPING = ROOT / 'experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv'
LABELS = ROOT / 'experiment/experiment11_roi_mapping_recovery/results/roi_111_labels.csv'
CONFIGS = ['B' + str(i) for i in range(10)]
TASKS = [(c, s, (i * 3 + s) % 2) for i, c in enumerate(CONFIGS) for s in range(3)]
NAMES = ['原始P0','旧ROI/WD.11/FC直通','旧ROI/WD0/FC直通','旧ROI/WD.11/关闭FC直通',
         '旧ROI/WD0/关闭FC直通','新耦合/FUNCTION','新耦合/CONSTANT','新耦合/ANONYMOUS',
         '新耦合/SHUFFLED','新耦合/SELF_ONLY']
PROTOCOL = dict(datadir=str(ROOT / 'data') + '/', datname='ABIDE', cpu=False,
                gpu='-1', nepoch=450, early=200, lr=.0038, reg=.11, dropout=.35,
                nlayer=1, n_hidden=18, n_head=2, n_iter=10, nmodal=4, th=.9,
                GC_mode='weighted-cosine', MP_mode='GCN', MF_mode='concat',
                alpha=.5, theta_smooth=1., theta_degree=.5, theta_sparsity=0.,
                nclass=2, mode='simple-2')
METRICS = ['acc', 'ba', 'hard_auc', 'probability_auc', 'sensitivity_asd', 'specificity_td']
PAIRS = [('B2','B1'),('B3','B1'),('B4','B2'),('B4','B3'),('B5','B4'),
         ('B5','B6'),('B5','B7'),('B5','B8'),('B5','B9'),('B5','B0'),('B5','B1')]
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
    atomic(path, b'SP16' + hashlib.sha256(data).hexdigest().encode() + data)


def checkpoint(path):
    import torch
    b = Path(path).read_bytes()
    assert b[:4] == b'SP16' and hashlib.sha256(b[68:]).hexdigest().encode() == b[4:68]
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
    p=read_json(PROMPTS)
    regions,_,_,_=verified_graph()
    assert [r['roi_name'] for r in regions]==[r['roi_name'] for r in p['regions']]
    assert len(p['a6_permutation'])==110 and sorted(p['a6_permutation'])==list(range(110))
    assert all(i!=j for i,j in enumerate(p['a6_permutation']))
    assert len(p['banks']['FUNCTION'])==len(p['banks']['ANONYMOUS'])==110
    assert all(not any(t in line.lower() for t in ['autism','patient','asd diagnosis']) for line in p['banks']['FUNCTION'])
    return p


def patched_sources():
    original={name:(ROOT/name).read_text() for name in ['network.py','model.py']}
    n=original['network.py'];begin=n.index('_prompt_text_features = None');end=n.index('\nclass MLP',begin)
    n=n[:begin]+('from roi_repair_runtime import get_frozen_p0,make_roi_encoder,inject_roi,roi_weight_decay\n\n'
                  'def get_prompt_text_features():\n    return get_frozen_p0()\n')+n[end:]
    begin=n.index('        text_features = self.prompt_text_features\n');end=n.index('        # 对最后一个维度进行平均池化',begin);old=n[begin:end]
    n=n[:begin]+"        if os.environ.get('RSPR_CONFIG','B0') == 'B0':\n"+''.join('    '+line if line.strip() else line for line in old.splitlines(True))+"        else:\n            part_x = x.view(871,4,4,9)\n"+n[end:]
    m=original['model.py'];old='        self.ModalFusion.apply(my_weight_init)\n'
    add=("        if os.environ.get('RSPR_CONFIG','B0') in ('B1','B2','B3','B4','B5','B6','B7','B8','B9'):\n"
         "            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):\n"
         "                torch.manual_seed(150000 + int(hyperpm.seed))\n"
         "                self.ModalFusion.roi_encoder = make_roi_encoder().to(dev)\n"
         "            self.optimizer_MF.add_param_group({'params': self.ModalFusion.roi_encoder.parameters(),\n"
         "                                             'lr': hyperpm.lr,'weight_decay': roi_weight_decay()})\n")
    assert m.count(old)==1;m=m.replace(old,old+add)
    for line in ['            feat, adj, ALoss = self.MessagePassing0(feat1,model_num, self.targ, self.trn_idx)\n','        feat ,_,_= self.MessagePassing0(feat1,model_num, self.targ, self.trn_idx)\n']:
        assert m.count(line)==1;indent=line[:len(line)-len(line.lstrip())];m=m.replace(line,line+indent+'feat = inject_roi(self, feat)\n')
    modified={'network.py':n,'model.py':m}
    patch=''.join(''.join(difflib.unified_diff(original[k].splitlines(True),modified[k].splitlines(True),fromfile='a/'+k,tofile='b/'+k)) for k in original)
    return modified,patch


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
    existing=read_json(MANIFEST) if MANIFEST.exists() else {}
    if existing.get('run_status')=='COMPLETE':
        integrity(full=True);return existing
    if (STATE/'static.pt').exists() and existing.get('source_data_sha256'):
        integrity();return existing
    assert torch.cuda.is_available() and torch.cuda.device_count()>=1,'GPU required; no CPU fallback'
    STATE.mkdir(exist_ok=True);RESULTS.mkdir(exist_ok=True);LOGS.mkdir(exist_ok=True)
    p=build_prompts();_,patch=patched_sources();atomic(EXP/'integration_patch.diff',patch)
    x,y,folds=input_data();_,indices,ends,unknown=verified_graph()
    from roi_pathway import patient_graph
    fixed=patient_graph(x[:,64:320],indices,ends)
    sys.path.insert(0,str(ROOT));from CLIP.clip import clip
    encoder,_=clip.load('RN50','cuda');encoder.eval().requires_grad_(False)
    lengths=[]
    def encode(sentences):
        sizes=[len(clip._tokenizer.encode(t))+2 for t in sentences];assert max(sizes)<=77;lengths.extend(sizes)
        with torch.no_grad():return encoder.encode_text(clip.tokenize(sentences,truncate=False).cuda()).float().detach().cpu()
    # Preserve experiment15's identical per-bank batches, including its unused
    # NAME bank. No new sentence or batch arrangement is introduced.
    p0=encode(p['p0']);encoded={k:F.normalize(encode(v),dim=-1) for k,v in p['banks'].items()}
    encoded['CONSTANT']=encoded['CONSTANT'].expand(110,-1).clone()
    encoded['SHUFFLED']=encoded['FUNCTION'][p['a6_permutation']].clone()
    history=ROOT/'experiment/experiment12_roi_semantic_alignment_feasibility/cache/roi_text_embeddings.pt'
    if history.exists():assert torch.equal(p0,torch.load(history,map_location='cpu',weights_only=False)['p0_features'])
    save_checkpoint(STATE/'static.pt',dict(p0=p0,banks=encoded,fixed=fixed))
    del encoder;torch.cuda.empty_cache()
    source_paths=CORE+['data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy',str(MAPPING.relative_to(ROOT)),str(LABELS.relative_to(ROOT)),'CLIP/clip/clip.py','CLIP/clip/model.py','CLIP/clip/simple_tokenizer.py']
    source={name:sha(ROOT/name) for name in source_paths}
    source.update({str((REFERENCE15/name).relative_to(ROOT)):sha(REFERENCE15/name) for name in ['run.py','roi_semantic.py','prompts.json','manifest.json','integration_patch.diff','results/fold_metrics.csv','results/diagnostics.csv','results/counterfactual.csv','results/report.md']})
    protected={str(f.relative_to(ROOT)):sha(f) for d in ROOT.joinpath('experiment').glob('experiment*') if d!=EXP for f in d.rglob('*') if f.is_file()}
    g={k:geometry(v) for k,v in encoded.items()}
    historical=read_json(REFERENCE15/'manifest.json')['text_audit']['geometry']
    assert all(abs(g[k][v]-historical[k][v])<1e-8 for k in g for v in g[k]),'Frozen bank geometry differs from experiment15'
    m=dict(experiment='RSPR ROI semantic pathway repair',run_status='PREPARED',prepared_utc=utc(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),github_audit=existing.get('github_audit',{}),
        source_data_sha256=source,protected_history_sha256=protected,protected_handoff_sha256=sha(ROOT/'handoff.md'),protected_handoff_bytes=(ROOT/'handoff.md').stat().st_size,
        clip_weight_sha256=sha(Path.home()/'.cache/clip/RN50.pt'),paper_sha256={p.name:sha(p) for p in ROOT.glob('*.pdf')},
        prompt_reference=str(PROMPTS.relative_to(ROOT)),prompt_sha256=sha(PROMPTS),patch_sha256=sha(EXP/'integration_patch.diff'),
        code_sha256={n:sha(EXP/n) for n in ['run.py','roi_pathway.py']},protocol=PROTOCOL,
        split_sha256=canonical([[a.tolist(),b.tolist()] for a,b in folds]),class_order={'0':'ASD','1':'TD'},dimensions=[48,6,10,256],
        roi_graph=dict(nodes=110,edges=252,unknown_features=unknown,mean_degree=float(fixed['degree'].mean()),
            legacy_weight='tanh(z)/degree',coupled_weight='positive/negative tanh(z), each normalized by sum(abs weights); epsilon1e-12; zero channel=0'),
        text_audit=dict(max_tokens=max(lengths),encoded_sentences=339,geometry=g,encoder='RN50 frozen eval',training_encode_calls=0,
            bank_sha256={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in encoded.items()}),
        base_parameters=1462957,additional_parameters=COUNTS,
        configurations={c:dict(name=NAMES[i],roi_wd=.11 if c in ['B1','B3'] else 0.,fc_direct=c in ['B1','B2'],bank='CONSTANT' if c=='B6' else 'ANONYMOUS' if c=='B7' else 'SHUFFLED' if c=='B8' else 'FUNCTION',self_only=c=='B9') for i,c in enumerate(CONFIGS)},
        rng_policy='Each config independently preserves native continuous tenfold stream, seed once. Isolated new init fork_rng(150000+seed); no fold reseeding or baseline anchors. Same entry=>same trunk; later entries may diverge after different stopping lengths.',
        gpu_hardware=subprocess.check_output(['nvidia-smi'],text=True),cpu_cgroup_cores=32,memory_cgroup_gib=128,worker_threads=2,workers_per_gpu=4,
        tasks={f'{c}_seed{s}':dict(config=c,seed=s,physical_gpu=g,status='PENDING',folds=0) for c,s,g in TASKS},completed_folds=0,preflight=None,
        design_notes=['Coupled decoder output bias would cancel in zero-reference subtraction, omitted equally for B5-B9; 39201 new parameters.',
            'B3/B4 retain all 38242 parameters and 65 decoder inputs; only final FC input is zero.',
            'Same-shape zero decoder reference; no detach; exact zero input produces exact zero output.',
            'No successful-response requirement during formal training; an unused semantic branch remains a valid negative result.'])
    save_json(MANIFEST,m);make_sources();write_readme(m);return m


def integrity(full=False):
    m = read_json(MANIFEST)
    for p,h in m['source_data_sha256'].items():
        assert sha(ROOT/p) == h, ('source/data changed',p)
    assert sha(PROMPTS) == m['prompt_sha256']
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
                                           'patch_sha256','protocol','split_sha256','rng_policy','configurations']})
    return _SIGNATURE


def context():
    global _CTX
    if _CTX is not None:
        return _CTX
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'Worker must have exactly one visible GPU'
    sys.modules['roi_repair_runtime'] = sys.modules[__name__]
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


def roi_weight_decay():
    return .11 if os.environ.get('RSPR_CONFIG') in ['B1','B3'] else 0.


def make_roi_encoder():
    from roi_pathway import LegacyROIEncoder,CoupledROIEncoder
    cfg=os.environ['RSPR_CONFIG'];c=context()
    key={'B6':'CONSTANT','B7':'ANONYMOUS','B8':'SHUFFLED'}.get(cfg,'FUNCTION')
    if cfg in ['B1','B2','B3','B4']:
        return LegacyROIEncoder(c['static']['fixed'],c['static']['banks'][key],disable_fc=cfg in ['B3','B4'])
    return CoupledROIEncoder(c['static']['fixed'],c['static']['banks'][key],self_only=cfg=='B9')


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
    os.environ['RSPR_CONFIG'] = config
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
    groups=a.optimizer_MP0.param_groups+a.optimizer_MF.param_groups
    params=[p for g in groups for p in g['params']]
    actual=[p for module in [a.MessagePassing0,a.ModalFusion] for p in module.parameters()]
    assert len(set(map(id,params)))==len(params) and set(map(id,params))==set(map(id,actual))
    assert all(g['lr']==.0038 for g in groups)
    assert a.optimizer_MP0.param_groups[0]['weight_decay']==a.optimizer_MF.param_groups[0]['weight_decay']==.11
    count=sum(p.numel() for p in actual);expected=COUNTS.get(config,0)
    if expected:
        assert len(a.optimizer_MF.param_groups)==2 and a.optimizer_MF.param_groups[1]['weight_decay']==(.11 if config in ['B1','B3'] else 0.)
        assert sum(p.numel() for p in a.optimizer_MF.param_groups[1]['params'])==expected
    assert count==1462957+expected
    return count


def forward(a):
    import torch
    a.MessagePassing0.eval();a.ModalFusion.eval()
    def capture(module,inputs,output):a._repair_logits=output.detach().clone()
    hook=a.ModalFusion.Outputlayer.classifier.register_forward_hook(capture)
    try:
        with torch.no_grad():
            n,parts=a.dataLayer(a.feat);h,_,_=a.MessagePassing0(parts,n,a.targ,a.trn_idx)
            out,_,_=a.ModalFusion(inject_roi(a,h));assert torch.isfinite(out).all()
            return out.clone()
    finally:hook.remove()


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


def components(a):
    module=getattr(a.ModalFusion,'roi_encoder',None)
    if module is None:return {}
    d=dict(numerical=module.numerical,query=module.query,self_transform=module.self_transform,decoder=module.decoder)
    if hasattr(module,'message'):d['message']=module.message
    else:d.update(positive_map=module.positive_map,negative_map=module.negative_map)
    return d


def parameter_norms(a):
    import torch
    d={k:float(torch.stack([p.detach().square().sum() for p in v.parameters()]).sum().sqrt()) for k,v in components(a).items()}
    module=getattr(a.ModalFusion,'roi_encoder',None)
    if module is not None:d['alpha']=abs(float(module.alpha.detach()))
    return d


def gradients(a):
    import torch
    module=getattr(a.ModalFusion,'roi_encoder',None)
    if module is None:return {}
    comps={k:list(v.parameters()) for k,v in components(a).items()};comps['alpha']=[module.alpha]
    norms=[]
    for params in comps.values():
        gg=[p.grad for p in params if p.grad is not None]
        norms.append(torch.stack([v.square().sum() for v in gg]).sum().sqrt() if gg else module.alpha.new_zeros(()))
    values=torch.stack(norms).detach().cpu().tolist();assert all(math.isfinite(v) for v in values)
    return dict(zip(comps,values))


def train_step(a):
    a.run_epoch('simple-2')
    return gradients(a)


def snapshot(a):
    module=a.ModalFusion.roi_encoder
    return {**{k:v.clone() for k,v in module.last.items()},'logits':a._repair_logits.clone()}


def differences(out,real,now,base,idx):
    values={}
    def record(prefix,t):
        v=t.detach().abs();values[prefix+'_abs_mean']=float(v.mean());values[prefix+'_abs_max']=float(v.max())
    record('probability_change',out[idx,0].exp()-real[idx,0].exp())
    record('log_probability_change',out[idx]-real[idx])
    for name in ['score','centered_score','response','semantic_node','m_pos','m_neg','pos_path','neg_path','self_path','message','updated','edge_signal','delta','logits']:
        if name in now and name in base:record(name+'_change',now[name][idx]-base[name][idx])
    values['flip_rate']=float((out[idx].argmax(1)!=real[idx].argmax(1)).float().mean())
    return values


def intervention(a,condition):
    module=a.ModalFusion.roi_encoder
    module.text_override=context()['static']['banks'][condition].cuda() if condition in ['FUNCTION','SHUFFLED','ANONYMOUS','CONSTANT'] else None
    module.off=condition=='OFF';module.self_only_override=True if condition=='SELF_ONLY' else None
    return forward(a)


def restore_real(a,real):
    module=a.ModalFusion.roi_encoder;module.text_override=None;module.off=False;module.self_only_override=None
    assert context()['torch'].equal(forward(a),real)


def counterfactual(a,seed,fold,real):
    import torch
    base=snapshot(a);baseline=metrics(real,a.tst_idx)
    original_hash=weight_hash(a)+weight_hash(a,True);text=a.ModalFusion.roi_encoder.text_features.clone();original_rng=rng_hash(rng())
    rows=[]
    for condition in CF_CONDITIONS:
        out=intervention(a,condition)
        if condition=='REAL':assert torch.equal(out,real)
        met=metrics(out,a.tst_idx)
        row=dict(config='B5',seed=seed,fold=fold,condition=condition,**met,
            **differences(out,real,snapshot(a),base,a.tst_idx),**{k+'_delta':met[k]-baseline[k] for k in METRICS})
        rows.append(row)
    restore_real(a,real)
    assert torch.equal(a.ModalFusion.roi_encoder.text_features,text)
    assert weight_hash(a)+weight_hash(a,True)==original_hash and rng_hash(rng())==original_rng
    return rows


def diagnostics(a,config,seed,fold,out,tracker,initial_norms,path_tracker):
    import torch
    import numpy as np
    row=dict(config=config,seed=seed,fold=fold,prediction_finite=bool(torch.isfinite(out).all()),enabled=config!='B0',spgs_enabled=config=='B0',res_kept=True)
    module=getattr(a.ModalFusion,'roi_encoder',None)
    if module is None:return row
    idx=a.tst_idx;last=module.last;base=snapshot(a);delta=last['delta'][idx];valid=module.feature_indices;fc=a._aarsp_base_fc[idx]
    score=last['score'][idx];centered=last['centered_score'][idx]
    bank={'B6':'CONSTANT','B7':'ANONYMOUS','B8':'SHUFFLED'}.get(config,'FUNCTION')
    assert torch.equal(module.text_features,context()['static']['banks'][bank].cuda())
    def amplitude(v):return float(v[idx].abs().mean())
    energy=delta[:,valid].square().mean(1)
    common=delta[:,valid].mean(1).square()/torch.where(energy>0,energy,torch.ones_like(energy))
    row.update(mean_degree=float(module.degree.mean()),roi_wd=.11 if config in ['B1','B3'] else 0.,direct_fc=config in ['B1','B2'],bank=bank,
        input_numeric_abs_mean=float(module.stats[idx].abs().mean()),alpha=float(module.alpha.detach()),beta=float(.3*module.alpha.detach().tanh()),
        query_vector_norm_mean=float(last['query_norm'][idx].mean()),query_vector_norm_min=float(last['query_norm'][idx].min()),
        score_mean=float(score.mean()),score_sd=float(score.std(unbiased=False)),score_roi_sd=float(score.std(1,unbiased=False).mean()),
        centered_score_mean=float(centered.mean()),centered_score_sd=float(centered.std(unbiased=False)),
        response_roi_sd=float(last['response'][idx].std(1,unbiased=False).mean()),response_roi_mean_sd=float(last['response'][idx].mean(0).std(unbiased=False)),
        numerical_node_abs_mean=amplitude(last['h']),semantic_node_abs_mean=amplitude(last['semantic_node']),
        self_path_abs_mean=amplitude(last['self_path']),message_abs_mean=amplitude(last['message']),updated_abs_mean=amplitude(last['updated']),
        node_between_roi_sd=float(last['updated'][idx].std(1,unbiased=False).mean()),
        edge_signal_abs_mean=amplitude(last['edge_signal']),edge_signal_edge_sd=float(last['edge_signal'][idx].std(1,unbiased=False).mean()),
        delta_abs_mean=float(delta.abs().mean()),delta_abs_max=float(delta.abs().max()),delta_valid_abs_mean=float(delta[:,valid].abs().mean()),
        delta_relative_l2=float((delta.norm(dim=1)/(fc.norm(dim=1)+1e-12)).mean()),delta_common_edge_energy_fraction=float(common.clamp(0,1).mean()),
        delta_nonzero_fraction=float(delta[:,valid].abs().gt(1e-8).float().mean()),unknown_direct_delta_max=float(delta[:,[195,225,232,236]].abs().max()),
        message_over_self_l2=float((last['message'][idx].flatten(1).norm(dim=1)/(last['self_path'][idx].flatten(1).norm(dim=1)+1e-12)).mean()),
        frozen_text_unchanged=True,all_roi_values_finite=all(bool(torch.isfinite(v).all()) for v in last.values()))
    common_text=module.text_features.mean(0,keepdim=True).expand(110,-1)
    common_score,_=module.similarities(last['h'][idx],common_text)
    row.update(common_score_rms=float(common_score.square().mean().sqrt()),distinctive_score_rms=float((score-common_score).square().mean().sqrt()))
    if hasattr(module,'positive_map'):
        for k in ['m_pos','m_neg','pos_path','neg_path']:row[k+'_abs_mean']=amplitude(last[k])
        pos,neg=last['pos_path'][idx].flatten(1),last['neg_path'][idx].flatten(1)
        row['message_cancellation']=float((1-(pos+neg).norm(1,dim=1)/(pos.norm(1,dim=1)+neg.norm(1,dim=1)+1e-12)).clamp(0,1).mean())
        row['positive_zero_channel_fraction']=float(module.positive.sum(-1).eq(0).float().mean())
        row['negative_zero_channel_fraction']=float(module.negative.sum(-1).eq(0).float().mean())
        row['zero_input_decoder_signal']=float(module.decode_edges(torch.zeros_like(last['updated'])).abs().max())
    else:
        zero=module.decoder(module.valid_fc.new_zeros(1,65)).squeeze()
        row['zero_input_decoder_signal']=float(zero.detach())
    norms=parameter_norms(a)
    for k,v in norms.items():
        row[k+'_parameter_norm']=v;row[k+'_initial_parameter_norm']=initial_norms[k]
        if k!='alpha':row[k+'_norm_ratio_to_init']=v/(initial_norms[k]+1e-12)
    row.update(path_tracker)
    row.update({k+'_grad_max':v for k,v in tracker.items()});row.update({k+'_grad_valid':v>0 for k,v in tracker.items()})
    row.update({'clip_'+k:v for k,v in read_json(MANIFEST)['text_audit']['geometry'][bank].items()})
    original_hash=weight_hash(a)+weight_hash(a,True);original_rng=rng_hash(rng())
    alternate='FUNCTION' if config=='B8' else 'SHUFFLED'
    row['text_intervention_bank']=alternate
    for condition,prefix in [(alternate,'text'),('SELF_ONLY','neighbor'),('OFF','off')]:
        changed=intervention(a,condition)
        change=differences(changed,out,snapshot(a),base,idx)
        row.update({prefix+'_'+k:v for k,v in change.items()})
    if hasattr(module,'disable_fc'):
        old=module.disable_fc;module.disable_fc=True
        changed=intervention(a,'REAL');row.update({'fc_off_'+k:v for k,v in differences(changed,out,snapshot(a),base,idx).items()});module.disable_fc=old
    restore_real(a,out)
    assert original_hash==weight_hash(a)+weight_hash(a,True) and original_rng==rng_hash(rng())
    assert row['unknown_direct_delta_max']==0 and all(not isinstance(v,float) or math.isfinite(v) for v in row.values())
    return row


def forbid_encoding():
    from CLIP.clip.model import CLIP
    def forbidden(*args,**kwargs):
        raise RuntimeError('Text encoding is forbidden during formal training')
    CLIP.encode_text = forbidden


def historical15_agent(seed,fold):
    import importlib.util
    c=context()
    spec=importlib.util.spec_from_file_location('rr15_reference_run',REFERENCE15/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    spec=importlib.util.spec_from_file_location('rr15_reference_roi',REFERENCE15/'roi_semantic.py');roi=importlib.util.module_from_spec(spec);spec.loader.exec_module(roi)
    shim=types.ModuleType('aarsp_runtime');shim.get_frozen_p0=get_frozen_p0;shim.inject_roi=inject_roi
    oldfixed={k:v for k,v in c['static']['fixed'].items() if k not in ['positive','negative']}
    shim.make_roi_encoder=lambda:roi.ROIEncoder(oldfixed,c['static']['banks']['FUNCTION'])
    sys.modules['aarsp_runtime']=shim
    sources,_=old.patched_sources()
    n=types.ModuleType('rr15_reference_network');sys.modules[n.__name__]=n
    exec(compile(sources['network.py'],'history15/network.py','exec'),n.__dict__)
    mm=types.ModuleType('rr15_reference_model');sys.modules[mm.__name__]=mm
    src=sources['model.py'].replace('from network import *','from rr15_reference_network import *')
    exec(compile(src,'history15/model.py','exec'),mm.__dict__)
    os.environ['AARSP_CONFIG']='A5';tr,te=c['folds'][fold-1]
    with contextlib.redirect_stdout(io.StringIO()):a=mm.EvalHelper([48,6,10,256],c['x'],c['y'],SimpleNamespace(**PROTOCOL,seed=seed),tr,te)
    a.MF_sav.close();a.GCMP_sav.close()
    return a


def preflight():
    import torch
    import numpy as np
    from torch.nn import functional as F
    from roi_pathway import CoupledROIEncoder,patient_graph
    m=integrity(full=True);c=context();seed_rng(0);entry=rng()
    original=agent_for('B0',0,1,original=True);native=forward(original)
    assert torch.equal(original.ModalFusion.prompt_text_features.cpu(),c['static']['p0'])
    native_hash=weight_hash(original);native_rng=rng_hash(rng());forbid_encoding()
    restore_rng(entry);c['clip_loaded']=False;off=agent_for('NO_SPGS',0,1);off_out=forward(off)
    errors={};counts={};hashes={}
    for config in CONFIGS:
        restore_rng(entry);c['clip_loaded']=False;a=agent_for(config,0,1)
        assert weight_hash(a)==native_hash and rng_hash(rng())==native_rng
        counts[config]=optimizer_audit(a,config);out=forward(a)
        expected=native if config=='B0' else off_out
        errors[config]=float((out-expected).abs().max());assert torch.equal(out,expected)
        if config!='B0':assert float(a.ModalFusion.roi_encoder.alpha)==0 and torch.count_nonzero(a.ModalFusion.roi_encoder.last['delta'])==0
        del a
    for seed in range(3):
        seed_rng(seed);entry=rng();common=None;new={}
        for config in CONFIGS:
            restore_rng(entry);c['clip_loaded']=False;a=agent_for(config,seed,1)
            if common is None:common,after=weight_hash(a),rng_hash(rng())
            assert weight_hash(a)==common and rng_hash(rng())==after
            if config!='B0':
                family='legacy' if int(config[1:])<5 else 'coupled';value=weight_hash(a,True)
                if family not in new:new[family]=value
                assert value==new[family]
            del a
        hashes[str(seed)]=dict(trunk=common,branches=new)
    # Original experiment15 constructor, formulas, gradient and two Adam steps.
    seed_rng(0);entry=rng();c['clip_loaded']=False;ref=historical15_agent(0,1)
    ref_init=weight_hash(ref)+weight_hash(ref,True);ref_after=rng()
    ref_out=forward(ref);ref_delta=ref.ModalFusion.roi_encoder.last['delta'].clone()
    states=[]
    for step in range(2):
        with contextlib.redirect_stdout(io.StringIO()):train_step(ref)
        states.append(dict(weights=weight_hash(ref)+weight_hash(ref,True),rng=rng_hash(rng()),out=forward(ref),
            gradients={k:p.grad.detach().clone() for k,p in ref.ModalFusion.named_parameters() if p.grad is not None}))
    restore_rng(entry);c['clip_loaded']=False;b1=agent_for('B1',0,1)
    assert ref_init==weight_hash(b1)+weight_hash(b1,True) and rng_hash(ref_after)==rng_hash(rng())
    assert torch.equal(forward(b1),ref_out) and torch.equal(b1.ModalFusion.roi_encoder.last['delta'],ref_delta)
    for expected in states:
        with contextlib.redirect_stdout(io.StringIO()):train_step(b1)
        assert expected['weights']==weight_hash(b1)+weight_hash(b1,True) and expected['rng']==rng_hash(rng())
        assert torch.equal(forward(b1),expected['out'])
        for k,v in b1.ModalFusion.named_parameters():
            if v.grad is not None:assert torch.equal(v.grad,expected['gradients'][k]),k
    del ref
    # Save/load both Adam optimizers, model and full RNG for legacy and coupled.
    recovery={};step_checks={};c['clip_loaded']=True
    for config in ['B1','B2','B3','B4','B5','B6','B7','B8','B9']:
        seed_rng(0);a=agent_for(config,0,1);optimizer_audit(a,config);tracked={k:0. for k in gradients(a)}
        for step in range(2):
            with contextlib.redirect_stdout(io.StringIO()):g=train_step(a)
            if step==0:assert g['alpha']>0 and all(v==0 for k,v in g.items() if k!='alpha'),(config,g)
            for k,v in g.items():tracked[k]=max(tracked[k],v)
        assert all(v>0 for v in tracked.values()),(config,tracked)
        step_checks[config]=tracked
        if config in ['B1','B5']:
            path=STATE/'preflight_restore.pt';save_checkpoint(path,dict(model=model_state(a),mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),rng=rng()))
            with contextlib.redirect_stdout(io.StringIO()):train_step(a)
            expected=weight_hash(a)+weight_hash(a,True);expected_rng=rng_hash(rng());expected_out=forward(a)
            saved=checkpoint(path);del a;a=agent_for(config,0,1);optimizer_audit(a,config);load_state(a,saved['model']);a.optimizer_MP0.load_state_dict(saved['mp']);a.optimizer_MF.load_state_dict(saved['mf']);restore_rng(saved['rng'])
            with contextlib.redirect_stdout(io.StringIO()):train_step(a)
            assert expected==weight_hash(a)+weight_hash(a,True) and expected_rng==rng_hash(rng()) and torch.equal(expected_out,forward(a))
            recovery[config]=True;path.unlink()
        if config=='B5':selected=a
        else:del a
    # Deterministic synthetic graph and probe; no evaluation ACC is used.
    rng_before=rng();synthetic=np.random.default_rng(160016).normal(0,.7,(5,256)).astype('float32')
    fixed=patient_graph(synthetic,c['static']['fixed']['feature_indices'].numpy(),c['static']['fixed']['endpoints'].numpy())
    probe={};generator=torch.Generator(device='cuda').manual_seed(160016)
    for config in CONFIGS[5:]:
        seed_rng(150000);bank={'B6':'CONSTANT','B7':'ANONYMOUS','B8':'SHUFFLED'}.get(config,'FUNCTION')
        mod=CoupledROIEncoder(fixed,c['static']['banks'][bank],self_only=config=='B9').cuda()
        assert torch.count_nonzero(mod())==0
        with torch.no_grad():mod.alpha.fill_(.2)
        real=mod();base={k:v.clone() for k,v in mod.last.items()};assert float(real.abs().max())>0,(config,'permanent zero branch')
        loss=(real*torch.randn(real.shape,device='cuda',generator=generator)).mean();loss.backward()
        grads={k:float(sum(float(v.grad.square().sum()) for v in vv.parameters() if v.grad is not None)**.5) for k,vv in dict(numerical=mod.numerical,query=mod.query,self_transform=mod.self_transform,positive_map=mod.positive_map,negative_map=mod.negative_map,decoder=mod.decoder).items()}
        grads['alpha']=float(mod.alpha.grad.abs());assert all(math.isfinite(v) and v>0 for v in grads.values()),(config,grads)
        assert not mod.text_features.requires_grad and mod.text_features.grad is None
        alternate='FUNCTION' if config=='B8' else 'SHUFFLED';mod.text_override=c['static']['banks'][alternate].cuda();changed=mod()
        stages={k:float((mod.last[k]-base[k]).abs().max()) for k in ['score','semantic_node','updated','edge_signal','delta']}
        assert all(v>0 for v in stages.values()),(config,stages)
        mod.text_override=None;mod.self_only_override=(not mod.self_only);neighbor=mod();assert float((neighbor-real).abs().max())>0
        mod.self_only_override=None;assert torch.equal(mod(),real)
        zeros=torch.zeros_like(base['semantic_node']);updated,_=mod.update_nodes(zeros);assert torch.count_nonzero(updated)==0 and torch.count_nonzero(mod.decode_edges(updated))==0
        fc=mod.valid_fc.clone();mod.valid_fc.add_(7);assert torch.equal(mod(),real);mod.valid_fc.copy_(fc)
        ends=mod.endpoints.clone();mod.endpoints.copy_(ends.flip(-1));assert torch.equal(mod(),real);mod.endpoints.copy_(ends)
        assert torch.count_nonzero(real[:,[195,225,232,236]])==0
        h=torch.randn(2,110,32,device='cuda',generator=generator,requires_grad=True)
        gram,_=mod.similarities(h,mod.text_features);direct=(F.normalize(mod.query(h),dim=-1)*mod.text_features[None]).sum(-1)
        value_err=float((gram-direct).abs().max());assert value_err<1e-6
        weights=torch.randn(gram.shape,device='cuda',generator=generator)
        gg=torch.autograd.grad((gram*weights).mean(),[h,mod.query.weight],retain_graph=True)
        gd=torch.autograd.grad((direct*weights).mean(),[h,mod.query.weight])
        grad_err=max(float((x-y).abs().max()) for x,y in zip(gg,gd));assert grad_err<1e-6
        edge=torch.randn(2,252,64,device='cuda',generator=generator)
        full=mod.decoder(edge)-mod.decoder(torch.zeros_like(edge));broadcast=mod.decoder(edge)-mod.decoder(edge.new_zeros(1,64))
        zero_err=float((full-broadcast).abs().max());assert zero_err<1e-6
        gf=torch.autograd.grad(full.square().mean(),list(mod.decoder.parameters()),retain_graph=True)
        gb=torch.autograd.grad(broadcast.square().mean(),list(mod.decoder.parameters()))
        zero_grad_err=max(float((x-y).abs().max()) for x,y in zip(gf,gb));assert zero_grad_err<1e-6
        # Full original PLN forward on random probe H, without patient labels.
        original_model=selected.ModalFusion;original_model.eval();H=torch.randn(871,320,device='cuda',generator=generator)
        d1=real[torch.arange(871,device='cuda')%5];d2=changed[torch.arange(871,device='cuda')%5]
        with torch.no_grad():
            H1=H.clone();H2=H.clone();H1[:,64:320]+=d1;H2[:,64:320]+=d2
            out1=original_model(H1)[0];out2=original_model(H2)[0]
        prediction=float((out1-out2).abs().max());assert prediction>0
        probe[config]=dict(stage_max_changes=stages,synthetic_output_change_max=prediction,gradient_norms=grads,
            gram_value_max_error=value_err,gram_gradient_max_error=grad_err,zero_reference_value_max_error=zero_err,zero_reference_gradient_max_error=zero_grad_err,
            nonzero_constant_path=config=='B6',zero_node_zero_output=True,no_direct_fc=True,endpoint_symmetric=True)
        del mod
    restore_rng(rng_before)
    real=forward(selected);cf=counterfactual(selected,0,1,real);assert len(cf)==6
    assert all(cf[0][k]==0 for k in cf[0] if k.endswith('_change_abs_mean') or k.endswith('_change_abs_max'))
    y=selected.targ.clone();selected.targ[selected.tst_idx]=1-selected.targ[selected.tst_idx];assert torch.equal(forward(selected),real);selected.targ.copy_(y)
    h=selected.feat.clone();changed=inject_roi(selected,h);assert torch.equal(changed[:,:64],h[:,:64]) and torch.equal(changed[:,64:320][:,[195,225,232,236]],h[:,64:320][:,[195,225,232,236]])
    # Real patch application check against unmodified root source.
    patchcheck=STATE/'patchcheck';patchcheck.mkdir(exist_ok=True)
    for name in ['network.py','model.py']:shutil.copy2(ROOT/name,patchcheck/name)
    subprocess.run(['git','apply','--check',str(EXP/'integration_patch.diff')],cwd=patchcheck,check=True);shutil.rmtree(patchcheck)
    checks=dict(status='PASS',parameter_counts=counts,initial_output_errors=errors,all_seed_initialization_hashes=hashes,
        original_B0_forward_exact=True,legacy_B1_initialization_forward_gradients_two_Adam_steps_exact=True,
        baseline_CLIP_rng_preserved=True,new_initialization_rng_isolated=True,optimizer_groups_covered=True,
        two_step_gradient_checks=step_checks,resume_model_both_Adam_rng_exact=recovery,synthetic_pathway_probes=probe,
        B6_not_permanently_zero=True,zero_input_no_common_bias=True,unknown_delta_zero=True,first64_unchanged=True,
        held_out_labels_no_forward_effect=True,counterfactual_REAL_and_restore_exact=True,counterfactual_rng_and_weights_unchanged=True,
        patch_applies_to_root=True,preflight_utc=utc())
    integrity(full=True);assert sha(ROOT/'handoff.md')==m['protected_handoff_sha256']
    m['preflight']=checks;m['run_status']='PREFLIGHT_PASS';m['code_freeze_utc']=utc();save_json(MANIFEST,m)
    print('PREFLIGHT_PASS '+json.dumps({'parameters':counts,'initial_errors':errors,'legacy_exact':True,'B6_probe':probe['B6']},ensure_ascii=False),flush=True)
    return checks


def lock(path):
    f = open(path,'a+')
    fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    return f



def assert_historical_fold(row):
    if row['config'] not in ['B0','B1']:return
    cfg='A0' if row['config']=='B0' else 'A5'
    reference=next(r for r in read_csv(REFERENCE15/'results/fold_metrics.csv') if r['config']==cfg and int(r['seed'])==row['seed'] and int(r['fold'])==row['fold'])
    for k in METRICS:assert abs(row[k]-float(reference[k]))<1e-12,('historical parity',row['config'],row['seed'],row['fold'],k,row[k],reference[k])
    for k in ['initial_trunk_sha256','entry_rng_sha256']:
        assert row[k]==reference[k],('historical initialization/RNG',row['config'],k)
    if row['config']=='B1':assert row['initial_branch_sha256']==reference['initial_branch_sha256']
    for k in ['best_epoch','epochs_executed']:assert row[k]==int(reference[k]),('historical selection',row['config'],k)


def worker(config,seed):
    import torch
    m = integrity()
    assert m['preflight']['status']=='PASS'
    task = f'{config}_seed{seed}'
    physical = int(os.environ['RSPR_PHYSICAL_GPU'])
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
        # Rehydrate the committed JSON after an interruption between the
        # atomic BETWEEN_FOLDS checkpoint and its JSON commit, including fold10.
        if saved['phase']=='BETWEEN_FOLDS':save_json(result_path,result)
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
        initial_norms = parameter_norms(a)
        best_acc,best_epoch,best_model,wait,start = 0.,-1,None,0,0
        tracked = {k:0. for k in gradients(a)}
        path_tracker={}
        elapsed_before = 0.
        if saved and saved['phase']=='EPOCH' and int(saved['fold'])==fold:
            load_state(a,saved['model'])
            a.optimizer_MP0.load_state_dict(saved['mp'])
            a.optimizer_MF.load_state_dict(saved['mf'])
            restore_rng(saved['rng'])
            best_acc,best_epoch,best_model,wait,start = saved['best_acc'],saved['best_epoch'],saved['best_model'],saved['wait'],saved['epoch']+1
            tracked = saved['tracker']
            entry,initial,branch_initial = saved['entry_rng'],saved['initial'],saved['branch_initial']
            initial_norms = saved['initial_norms'];path_tracker=saved['path_tracker']
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
            if config!='B0' and (epoch==0 or (epoch+1)%25==0 or wait>200 or epoch==449):
                ll=a.ModalFusion.roi_encoder.last
                vv={'delta_abs_max':ll['delta'].abs().max(),'message_abs_mean':ll['message'].abs().mean(),'semantic_node_abs_mean':ll['semantic_node'].abs().mean(),'query_vector_norm_mean':ll['query_norm'].mean()}
                for k,v in vv.items():path_tracker['training_max_'+k]=max(path_tracker.get('training_max_'+k,0.),float(v))
            last = epoch
            stop = wait>200 or epoch==449
            if (epoch+1)%25==0 or stop:
                save_checkpoint(save_path,dict(signature=sig,phase='EPOCH',fold=fold,epoch=epoch,
                    model=model_state(a),mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),
                    rng=rng(),best_acc=best_acc,best_epoch=best_epoch,best_model=best_model,wait=wait,
                    tracker=tracked,entry_rng=entry,initial=initial,branch_initial=branch_initial,initial_norms=initial_norms,path_tracker=path_tracker,
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
                   initial_trunk_sha256=initial,initial_branch_sha256=branch_initial if config in CONFIGS[1:] else '',
                   entry_rng_sha256=rng_hash(entry),additional_parameters=COUNTS[config])
        assert_historical_fold(row)
        result['metrics'].append(row)
        result['diagnostics'].append(diagnostics(a,config,seed,fold,out,tracked,initial_norms,path_tracker))
        if config=='B5':
            result['counterfactual'].extend(counterfactual(a,seed,fold,out))
        assert all(math.isfinite(v) for v in tracked.values())
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
        task.update(status=t['status'],folds=len(t['metrics']),seconds=t['seconds'],started_utc=t['started_utc'],updated_utc=t.get('updated_utc'))
    m['completed_folds'] = len(folds)
    m['updated_utc'] = utc()
    for name,rows in [('fold_metrics',folds),('diagnostics',diag),('counterfactual',cf)]:
        save_csv(RESULTS/(name+'.csv'),rows)
    save_json(MANIFEST,m)
    return m


def worker_env(gpu):
    return dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),
                RSPR_PHYSICAL_GPU=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
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
        print('ALREADY_COMPLETE: 30 tasks / 300 folds; no training restarted',flush=True)
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
    attempts = dict(m.get('failures_and_restarts',{}))
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
                    m=read_json(MANIFEST);m['failures_and_restarts']=attempts;save_json(MANIFEST,m)
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
            print(f'PROGRESS tasks={len(done)}/30 folds={m["completed_folds"]}/300 active={len(active)}',flush=True)
            last_refresh = time.monotonic()
        if len(done)==30 and not active:
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
    print('COMPLETE 30/30 tasks 300/300 folds; temporary state cleaned',flush=True)


def validate_complete():
    m=integrity(full=True)
    rows=read_csv(RESULTS/'fold_metrics.csv');ds=read_csv(RESULTS/'diagnostics.csv');cf=read_csv(RESULTS/'counterfactual.csv')
    keys={(c,s,f) for c in CONFIGS for s in range(3) for f in range(1,11)}
    assert len(rows)==300 and {(r['config'],int(r['seed']),int(r['fold'])) for r in rows}==keys
    assert len(ds)==300 and {(r['config'],int(r['seed']),int(r['fold'])) for r in ds}==keys
    assert len(cf)==180 and {(r['config'],int(r['seed']),int(r['fold']),r['condition']) for r in cf}=={('B5',s,f,k) for s in range(3) for f in range(1,11) for k in CF_CONDITIONS}
    assert m['preflight']['status']=='PASS' and all(t['status']=='COMPLETE' and t['folds']==10 for t in m['tasks'].values())
    for r in rows+cf:
        assert all(0<=float(r[k])<=1 and math.isfinite(float(r[k])) for k in METRICS)
        assert abs(float(r['ba'])-float(r['hard_auc']))<1e-12
    for r in rows:
        assert abs(float(r['selected_val_acc'])-float(r['acc']))<1e-12
        assert 0<=int(r['best_epoch'])<int(r['epochs_executed'])<=450
        assert int(r['additional_parameters'])==COUNTS[r['config']]
        assert_historical_fold({**r,'seed':int(r['seed']),'fold':int(r['fold']),**{k:float(r[k]) for k in METRICS},'best_epoch':int(r['best_epoch']),'epochs_executed':int(r['epochs_executed'])})
    for r in ds:
        assert r['prediction_finite'].lower()=='true' and r['res_kept'].lower()=='true'
        if r['config']!='B0':
            assert r['all_roi_values_finite'].lower()=='true' and float(r['unknown_direct_delta_max'])==0
            assert all(math.isfinite(float(v)) for k,v in r.items() if v and (k.endswith('_grad_max') or k.endswith('_parameter_norm')))
        if r['config'] in CONFIGS[5:]:assert float(r['zero_input_decoder_signal'])==0
    for r in cf:
        if r['condition']=='REAL':assert all(float(v)==0 for k,v in r.items() if k.endswith('_change_abs_mean') or k.endswith('_change_abs_max') or k.endswith('_delta') or k=='flip_rate')
    for gpu in [0,1]:
        assert sum(int(r['physical_gpu'])==gpu for r in rows)==150
        assert sum(t['physical_gpu']==gpu for t in m['tasks'].values())==15
    parity={}
    for number,config in [(13,'F0'),(14,'E00'),(15,'A0')]:
        folder=next(ROOT.joinpath('experiment').glob('experiment'+str(number)+'_*'))
        ref={(int(r['seed']),int(r['fold'])):r for r in read_csv(folder/'results/fold_metrics.csv') if r['config']==config}
        errors={k:max(abs(float(r[k])-float(ref[int(r['seed']),int(r['fold'])]['balanced_accuracy' if number==14 and k=='ba' else k])) for r in rows if r['config']=='B0') for k in METRICS}
        assert all(v<1e-12 for v in errors.values()),('B0 historical parity',number,errors)
        parity['experiment'+str(number)]=dict(max_errors=errors,exact=True)
    initial_groups=[[r for r in rows if int(r['seed'])==s and int(r['fold'])==f] for s in range(3) for f in range(1,11)]
    for s in range(3):
        for family in [CONFIGS[1:5],CONFIGS[5:]]:
            assert len({r['initial_branch_sha256'] for r in rows if int(r['seed'])==s and r['config'] in family})==1
    assert not live_workers()
    m['validation']=dict(status='PASS',folds=300,diagnostics=300,counterfactual=180,gpu_folds={'0':150,'1':150},gpu_tasks={'0':15,'1':15},
        common_initialization_equal_pairs=sum(len({r['initial_trunk_sha256'] for r in g})==1 for g in initial_groups),
        new_initialization_equal_within_family=True,continuous_rng_no_fold_reseed=True,baseline_parity=parity,legacy_B1_all_six_metrics_selection_rng_initialization_exact=True,
        formal_near_zero_branches_retained=True,protected_history_files=len(m['protected_history_sha256']))
    save_json(MANIFEST,m)
    return rows,ds,cf


def aggregates(rows):
    import numpy as np
    out={}
    for config in CONFIGS:
        out[config]={}
        for k in METRICS:
            means=[float(np.mean([float(r[k]) for r in rows if r['config']==config and int(r['seed'])==s])) for s in range(3)]
            out[config][k]=dict(mean=float(np.mean(means)),sd=float(np.std(means)),seeds=means)
    return out


def contrast(rows,terms,key):
    import numpy as np
    index={(r['config'],int(r['seed']),int(r['fold'])):r for r in rows}
    diffs=[sum(coef*float(index[c,s,f][key]) for c,coef in terms.items()) for s in range(3) for f in range(1,11)]
    seeds=[float(np.mean(diffs[i*10:(i+1)*10])) for i in range(3)]
    return dict(mean=float(np.mean(seeds)),sd=float(np.std(seeds)),seeds=seeds,win=sum(v>1e-12 for v in diffs),tie=sum(abs(v)<=1e-12 for v in diffs),loss=sum(v<-1e-12 for v in diffs))


def paired(rows,left,right,key):
    return contrast(rows,{left:1,right:-1},key)


def factorial():
    return {'WD主效应(0−.11)':{'B2':.5,'B4':.5,'B1':-.5,'B3':-.5},
            'FC直通关闭主效应':{'B3':.5,'B4':.5,'B1':-.5,'B2':-.5},
            '交互(B4−B3)−(B2−B1)':{'B4':1,'B3':-1,'B2':-1,'B1':1}}


def mean_rows(rows,key):
    values=[float(r[key]) for r in rows if r.get(key) not in [None,'']]
    return sum(values)/len(values) if values else 0.


def scientific_judgment(rows,ds,cf):
    a=aggregates(rows);primary=['acc','ba','probability_auc']
    def robust(left,right):return all(all(v>1e-12 for v in paired(rows,left,right,k)['seeds']) for k in primary)
    real=[r for r in ds if r['config']=='B5'];swap=[r for r in cf if r['condition']=='SHUFFLED']
    active=sum(float(r['text_delta_change_abs_max'])>0 and float(r['text_probability_change_abs_max'])>0 for r in real)
    substantive=sum(float(r['text_probability_change_abs_mean'])>1e-8 for r in real)
    neighbor_active=sum(float(r['neighbor_probability_change_abs_max'])>0 for r in real)
    semantic_training=all(robust('B5',c) for c in ['B6','B7','B8'])
    # REAL-minus-SHUFFLED uses the same selected checkpoint; contrast is descriptive.
    cf_seed={k:[-mean_rows([r for r in swap if int(r['seed'])==s],k+'_delta') for s in range(3)] for k in primary}
    semantic_cf=all(all(v>1e-12 for v in values) for values in cf_seed.values())
    classification=robust('B5','B0')
    return dict(mean_classification_improved=all(paired(rows,'B5','B0',k)['mean']>1e-12 for k in primary),classification_seed_consistent=classification,
        pathway_text_prediction_sensitive_folds=active,pathway_text_probability_change_gt_1e_minus8_folds=substantive,
        neighbor_prediction_sensitive_folds=neighbor_active,numerical_pathway_repaired=substantive>=24,
        correct_matching_ACC_BA_all_seeds_positive=all(all(v>1e-12 for v in paired(rows,'B5','B8',k)['seeds']) for k in ['acc','ba']),
        correct_matching_probability_AUC_all_seeds_positive=all(v>1e-12 for v in paired(rows,'B5','B8','probability_auc')['seeds']),
        semantic_training_controls_consistent=semantic_training,semantic_counterfactual_consistent=semantic_cf,
        semantic_support='DESCRIPTIVE_SUPPORT' if semantic_training and semantic_cf else 'NOT_SUPPORTED',
        neighbor_performance_support=robust('B5','B9'),counterfactual_REAL_minus_SHUFFLED_seeds=cf_seed,
        advance_global_alignment=classification and semantic_training and semantic_cf and robust('B5','B9') and substantive>=24,
        best={k:[c for c in CONFIGS if abs(a[c][k]['mean']-max(a[v][k]['mean'] for v in CONFIGS))<1e-12] for k in primary})


def result_table(rows):
    a=aggregates(rows)
    lines=['|组|设计|ACC %|BA %|Hard-AUC %|Prob-AUC %|SEN-ASD %|SPE-TD %|','|---|---|---:|---:|---:|---:|---:|---:|']
    for i,c in enumerate(CONFIGS):lines.append('|'+c+'|'+NAMES[i]+'|'+'|'.join(f"{a[c][k]['mean']*100:.4f}±{a[c][k]['sd']*100:.4f}" for k in METRICS)+'|')
    return lines


def interpretation(rows,ds,cf,j):
    b=[r for r in ds if r['config']=='B5']
    near_zero=sum(float(r['delta_valid_abs_mean'])<1e-8 for r in b)
    wd=paired(rows,'B2','B1','acc');fc=paired(rows,'B3','B1','acc');both=paired(rows,'B4','B1','acc')
    lines=[f"数值通路：B5在{j['pathway_text_prediction_sensitive_folds']}/30个最佳checkpoint上对错配文本产生非零delta及预测变化；其中概率平均变化>1e-8有{j['pathway_text_probability_change_gt_1e_minus8_folds']}/30折。邻居替换引起非零预测变化有{j['neighbor_prediction_sensitive_folds']}/30折。有效delta均值<1e-8的B5折数为{near_zero}/30。实验十五A5正确/错配/常量/SELF_ONLY的delta和预测变化曾逐值为0；据此区分数值连接恢复与医学必要性。",
        f"旧分支2×2：取消WD的B2−B1 ACC为{wd['mean']*100:+.5f}pp；仅关FC直通B3−B1为{fc['mean']*100:+.5f}pp；两项同时改变B4−B1为{both['mean']*100:+.5f}pp。必须结合参数范数、消息幅度、文本替换影响及交互项，不将单一分类变化等同于因果机制。",
        f"分类：B5−B0的ACC/BA/Prob-AUC为"+'/'.join(f"{paired(rows,'B5','B0',k)['mean']*100:+.5f}" for k in ['acc','ba','probability_auc'])+f"pp；三指标均值同时提高={j['mean_classification_improved']}，各指标三个seed均为正={j['classification_seed_consistent']}。",
        f"医学语义必要性：B5稳定超过B6常量、B7匿名和B8错配（三指标、三seed全正）={j['semantic_training_controls_consistent']}；同checkpoint REAL−SHUFFLED满足同一方向条件={j['semantic_counterfactual_consistent']}。合并判断={j['semantic_support']}。任何非零梯度或文本敏感性本身不证明正确医学语义有效。",
        f"邻居必要性：B5稳定超过B9={j['neighbor_performance_support']}；即使SELF_ONLY干预显著改变预测，也可能说明分布变化敏感，不能代替B9重训对照。",
        ('下一步：支持另行立项研究ROI条件化疾病语义对齐，仍需独立外部验证。' if j['advance_global_alignment'] else '下一步：当前证据不足以进入全局疾病语义对齐。保留这次通路修复及负结果，不自动增加Prompt、损失、模型或超参数版本。')]
    lines+=['', '逐层因果证据（均为30折平均，最大变化取跨fold最大）：']
    for c in ['B1','B2','B3','B4']:
        rr=[r for r in ds if r['config']==c]
        v=lambda k:mean_rows(rr,k)
        maximum=lambda k:max(float(r[k]) for r in rr)
        sensitive=sum(float(r['text_probability_change_abs_mean'])>1e-8 for r in rr)
        lines.append(f"{c}：query范数/初始={v('query_norm_ratio_to_init'):.6e}，数值编码范数/初始={v('numerical_norm_ratio_to_init'):.6e}；消息幅度={v('message_abs_mean'):.6e}；edge signal绝对均值={v('edge_signal_abs_mean'):.6e}；零输入decoder输出绝对均值={sum(abs(float(r['zero_input_decoder_signal'])) for r in rr)/30:.6e}；共同边delta能量={v('delta_common_edge_energy_fraction'):.6f}；文本替换Δedge signal均值={v('text_edge_signal_change_abs_mean'):.6e}，Δdelta均值/最大={v('text_delta_change_abs_mean'):.6e}/{maximum('text_delta_change_abs_max'):.6e}，Δ概率均值/最大={v('text_probability_change_abs_mean'):.6e}/{maximum('text_probability_change_abs_max'):.6e}，有实质概率变化（均值>1e−8）={sensitive}/30折。")
    lines.append('以上证实WD与参数收缩有关：B1/B3远小于初始范数，B2/B4保留范数。然而取消WD没有普遍恢复最终文本依赖；中间edge signal仍可变化，而delta变化被显著压低。B2/B4较大的edge signal与末端tanh饱和一致，共同边能量说明大量空间差异已消失。关闭FC直通也未普遍消除退化，因此不能把FC直通唯一认定为根因。独立偏置效应与(1+r)绕行效应没有单独重训因子，仍未唯一证实。')
    rr=[r for r in ds if r['config']=='B5'];v=lambda k:mean_rows(rr,k)
    lines.append(f"B5修复状态：query范数/初始={v('query_norm_ratio_to_init'):.6f}，原始正/负邻居消息幅度={v('m_pos_abs_mean'):.6e}/{v('m_neg_abs_mean'):.6e}，映射后正/负消息={v('pos_path_abs_mean'):.6e}/{v('neg_path_abs_mean'):.6e}，抵消比例={v('message_cancellation'):.6f}，共同边delta能量={v('delta_common_edge_energy_fraction'):.6f}，delta相对GFR Correlation特征L2={v('delta_relative_l2'):.6f}。零节点残差严格为0，并未强制所有受试者残差为0。")
    matching={k:paired(rows,'B5','B8',k) for k in ['acc','ba','probability_auc']}
    lines.append('正确对应关系的有限正证据：B5−B8 ACC/BA/Prob-AUC='+'/'.join(f"{matching[k]['mean']*100:+.5f}" for k in matching)+'pp；ACC三个seed='+ '/'.join(f'{v*100:+.5f}' for v in matching['acc']['seeds'])+'pp，BA三个seed='+ '/'.join(f'{v*100:+.5f}' for v in matching['ba']['seeds'])+'pp；ACC/BA三seed方向一致，但Prob-AUC三seed='+ '/'.join(f'{v*100:+.5f}' for v in matching['probability_auc']['seeds'])+'pp，仍不一致。同checkpoint REAL−SHUFFLED三个seed的三项指标均为正，说明正确已学习对应关系对当前模型有作用。')
    lines.append('因此不能概括为完全没有任何对应关系证据；但B5低于原始P0和常量控制，真实文本对匿名控制及邻居传播优势也不稳定。预先规定的全部控制×三指标×三seed联合标准未通过。可支持“对应关系有局部作用”，尚不能支持“完整医学知识产生稳定独立诊断增益”。')
    lines.append('直接回答：优化收缩是否缓解与数值通路是否恢复，按上述参数范数及干预链分别判断；正确医学语义是否增加诊断信息，必须用B5−B6/B7/B8及同checkpoint REAL−SHUFFLED联合判定。没有把“文本进入分类器”包装成“临床知识正确”。')
    lines.append('OFF导致已训练模型性能下降，只说明已训练主干/分类器与残差共同适应；它不能解释为相对重新训练P0的净增益。B5的相对L2为每fold受试者均值，不能据其小于1保证每个受试者的残差均不大于原特征。新方案仍有部分共同边偏移和末端tanh饱和，恢复文本敏感性不等于所有数值退化均已消除。')
    return lines


def report():
    rows,ds,cf=validate_complete();m=read_json(MANIFEST);a=aggregates(rows);j=scientific_judgment(rows,ds,cf)
    lines=['# 实验十六：ROI语义通路修复与医学语义必要性验证','',
        '工程验收PASS：10组、30任务、300折真实训练；300条逐fold最佳checkpoint诊断；B5同checkpoint六条件180条反事实。B0/B1本轮重训并逐折核对历史，不复制历史指标。','',
        '## 全部正式结果','',
        '每个seed先求10折均值，再计算三个seed均值及population SD(ddof=0)。ASD=0、TD=1；probability-AUC采用exp(log-softmax)第0列；Hard-AUC保留原硬one-hot口径，二分类下与BA相等。','']
    lines+=result_table(rows)
    lines+=['','## 每seed十折均值','','|组|seed|ACC %|BA %|Hard-AUC %|Prob-AUC %|SEN %|SPE %|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in CONFIGS:
        for s in range(3):lines.append('|'+c+'|'+str(s)+'|'+'|'.join(f"{a[c][k]['seeds'][s]*100:.5f}" for k in METRICS)+'|')
    lines+=['','## 规定的配对比较','','同seed/fold配对，差值为pp；seed SD不把30折视为30个独立样本。胜/平/负容差1e-12。连续十折随机流因早停不同自然分叉；除同入口预检外，不声称每fold主干初始化都相同。','','|比较|指标|均值±seed SD pp|seed0/1/2 pp|胜/平/负|','|---|---|---:|---|---|']
    for left,right in PAIRS:
        for k in METRICS:
            v=paired(rows,left,right,k);lines.append(f"|{left}−{right}|{k}|{v['mean']*100:+.5f}±{v['sd']*100:.5f}|"+'/'.join(f'{x*100:+.5f}' for x in v['seeds'])+f"|{v['win']}/{v['tie']}/{v['loss']}|")
    lines+=['','## B1–B4的2×2主效应与交互','','WD主效应=[(B2−B1)+(B4−B3)]/2；关闭FC直通主效应=[(B3−B1)+(B4−B2)]/2；交互=(B4−B3)−(B2−B1)。注册参数一致，B3/B4末列FC输入置0，末列权重的有效输入容量也因此降低。','','|效应|指标|均值±seed SD pp|seed0/1/2 pp|正/零/负折|','|---|---|---:|---|---|']
    for name,terms in factorial().items():
        for k in METRICS:
            v=contrast(rows,terms,k);lines.append(f"|{name}|{k}|{v['mean']*100:+.5f}±{v['sd']*100:.5f}|"+'/'.join(f'{x*100:+.5f}' for x in v['seeds'])+f"|{v['win']}/{v['tie']}/{v['loss']}|")
    lines+=['','## 最佳checkpoint数值通路','','梯度列为整个训练过程最大有限范数；training_max列为epoch0、每25epoch和结束时的训练后前向抽样峰值。两者与最佳checkpoint的文本/邻居/OFF依赖分别记录，不将曾经学习等同于最终预测依赖。','','|组|query参数/初始|数值参数/初始|query向量norm|语义节点幅度|self幅度|消息幅度|delta均值|相对L2|共同边能量比例|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in CONFIGS[1:]:
        rr=[r for r in ds if r['config']==c]
        ks=['query_norm_ratio_to_init','numerical_norm_ratio_to_init','query_vector_norm_mean','semantic_node_abs_mean','self_path_abs_mean','message_abs_mean','delta_valid_abs_mean','delta_relative_l2','delta_common_edge_energy_fraction']
        lines.append('|'+c+'|'+'|'.join(f'{mean_rows(rr,k):.6e}' for k in ks)+'|')
    lines+=['','|组|跨ROI score SD|中心化score SD|response跨ROI SD|updated跨ROI SD|edge signal边间SD|pos原消息|neg原消息|pos映射幅度|neg映射幅度|抵消比例|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in CONFIGS[1:]:
        rr=[r for r in ds if r['config']==c]
        ks=['score_roi_sd','centered_score_sd','response_roi_sd','node_between_roi_sd','edge_signal_edge_sd','m_pos_abs_mean','m_neg_abs_mean','pos_path_abs_mean','neg_path_abs_mean','message_cancellation']
        lines.append('|'+c+'|'+'|'.join(f'{mean_rows(rr,k):.6e}' if any(r.get(k) for r in rr) else '—' for k in ks)+'|')
    lines+=['','抵消比例为1−||pos_path+neg_path||₁/(||pos_path||₁+||neg_path||₁+1e−12)，先按受试者计算再平均。共同边能量=mean(delta)²/mean(delta²)，delta为0时定义0；不是借助epsilon把微小delta强行判为不共同。','',
        '|组|训练query梯度峰值|训练消息/正映射梯度峰值|训练delta最大值峰值|最佳delta最大值|text Δdelta均值/最大|text Δ概率均值/最大|neighbor Δ概率均值/最大|OFF Δ概率均值/最大|','|---|---:|---:|---:|---:|---|---|---|---|']
    for c in CONFIGS[1:]:
        rr=[r for r in ds if r['config']==c];g='message_grad_max' if c in CONFIGS[1:5] else 'positive_map_grad_max'
        v=lambda k:mean_rows(rr,k);mx=lambda k:max(float(r[k]) for r in rr)
        line=f"|{c}|{v('query_grad_max'):.6e}|{v(g):.6e}|{v('training_max_delta_abs_max'):.6e}|{mx('delta_abs_max'):.6e}|"
        line+='|'.join(f"{v(k+'_abs_mean'):.6e}/{mx(k+'_abs_max'):.6e}" for k in ['text_delta_change','text_probability_change','neighbor_probability_change','off_probability_change'])+'|';lines.append(line)
    lines+=['','所有最大变化列报告30折中的最大值；均值为先每fold测试人员平均再平均30折。未保存受试者级预测。B1–B4的额外FC_OFF同checkpoint通路诊断保存在diagnostics.csv；其干预均值/最大值用于区分语义残差与原数值直通。','',
        '|旧组|FC_OFF Δdelta均值/最大|FC_OFF Δ概率均值/最大|零输入decoder输出|','|---|---|---|---:|']
    for c in CONFIGS[1:5]:
        rr=[r for r in ds if r['config']==c]
        lines.append('|'+c+'|'+'|'.join(f"{mean_rows(rr,k+'_abs_mean'):.6e}/{max(float(r[k+'_abs_max']) for r in rr):.6e}" for k in ['fc_off_delta_change','fc_off_probability_change'])+f"|{mean_rows(rr,'zero_input_decoder_signal'):.6e}|")
    lines+=['','机制层2×2效应（原单位，含seed差值）：','','|效应|诊断项|均值±seed SD|seed0/1/2|','|---|---|---:|---|']
    for name,terms in factorial().items():
        for k in ['query_norm_ratio_to_init','query_vector_norm_mean','message_abs_mean','delta_valid_abs_mean','delta_common_edge_energy_fraction','text_delta_change_abs_mean','text_probability_change_abs_mean','neighbor_probability_change_abs_mean']:
            v=contrast(ds,terms,k);lines.append(f"|{name}|{k}|{v['mean']:+.6e}±{v['sd']:.6e}|"+'/'.join(f'{x:+.6e}' for x in v['seeds'])+'|')
    lines+=['','## B5六条件同checkpoint反事实','','仅改变ROI分支：REAL/FUNCTION，SHUFFLED固定实验十五置换，ANONYMOUS，CONSTANT，SELF_ONLY及OFF。所有参数、GFR、输入、fold与checkpoint保持不变，不再训练。最终恢复REAL，检查权重及RNG完全相同；logits取原OutputLayer.classifier真实输出，概率由log-softmax.exp取得。','','|条件|ACC %|BA %|Prob-AUC %|翻转 %|Δ概率均值/最大|Δdelta均值/最大|Δlogits均值/最大|相对REAL ACC/BA/Prob-AUC pp|','|---|---:|---:|---:|---:|---|---|---|---|']
    for cond in CF_CONDITIONS:
        rr=[r for r in cf if r['condition']==cond];v=lambda k:mean_rows(rr,k)
        lines.append(f"|{cond}|{v('acc')*100:.5f}|{v('ba')*100:.5f}|{v('probability_auc')*100:.5f}|{v('flip_rate')*100:.5f}|"+'|'.join(f"{v(k+'_abs_mean'):.6e}/{max(float(r[k+'_abs_max']) for r in rr):.6e}" for k in ['probability_change','delta_change','logits_change'])+'|'+'/'.join(f"{v(k+'_delta')*100:+.5f}" for k in ['acc','ba','probability_auc'])+'|')
    lines+=['','完整反事实通路（fold均值/跨fold最大）：','','|条件|score|centered score|semantic node|updated|edge signal|','|---|---|---|---|---|---|']
    for cond in CF_CONDITIONS:
        rr=[r for r in cf if r['condition']==cond]
        lines.append('|'+cond+'|'+'|'.join(f"{mean_rows(rr,k+'_change_abs_mean'):.6e}/{max(float(r[k+'_change_abs_max']) for r in rr):.6e}" for k in ['score','centered_score','semantic_node','updated','edge_signal'])+'|')
    lines+=['','|条件|指标|干预−REAL均值±seed SD pp|seed0/1/2 pp|REAL优/平/劣折|','|---|---|---:|---|---|']
    import numpy as np
    for cond in CF_CONDITIONS[1:]:
        rr=[r for r in cf if r['condition']==cond]
        for k in METRICS:
            vals=[float(r[k+'_delta']) for r in rr];seeds=[mean_rows([r for r in rr if int(r['seed'])==s],k+'_delta') for s in range(3)]
            lines.append(f"|{cond}|{k}|{np.mean(seeds)*100:+.5f}±{np.std(seeds)*100:.5f}|"+'/'.join(f'{x*100:+.5f}' for x in seeds)+f"|{sum(v<-1e-12 for v in vals)}/{sum(abs(v)<=1e-12 for v in vals)}/{sum(v>1e-12 for v in vals)}|")
    lines+=['','## 分类、机制及医学归因','','最高均值：'+', '.join(k+'='+','.join(j['best'][k]) for k in j['best'])+'。','']+interpretation(rows,ds,cf,j)
    lines+=['','上述“稳定”预先采用ACC、BA、Prob-AUC三指标及三seed全为正的保守描述规则；通路恢复阈值为至少24/30折Δ概率均值>1e−8。该阈值不用于选模型或调参数，也不替代统计显著性。','',
        '实验十五的退化不能仅归咎于Prompt措辞：旧query参数收缩、数值消息幅度极小、decoder FC直通及bias可产生共同残差，而(1+r)允许数值分支不依赖文本。2×2只识别WD与直通的条件效应；B5相对B4同时改动中心化、乘法耦合、正负归一化和零参考解码，新增959参数，不能把其结果唯一归给某一项修复。','',
        '## 历史比较','','B0对实验十三F0/十四E00/十五A0六指标逐折完全一致；B1对十五A5六指标、初始化、入口RNG和best/总epoch逐折完全一致。实验九E9只作历史描述，不能当作相同初始化严格对照。','']
    old=next(ROOT.joinpath('experiment').glob('experiment9_*'));historical=[r for r in read_csv(old/'results/fold_metrics.csv') if r['config']=='E9']
    lines+=['|配置|ACC %|BA %|Hard-AUC %|Prob-AUC %|SEN %|SPE %|','|---|---:|---:|---:|---:|---:|---:|']
    keys9=['acc','balanced_accuracy','hard_auc','probability_auc','sensitivity_asd','specificity_asd']
    lines.append('|历史E9|'+'|'.join(f'{mean_rows(historical,k)*100:.5f}' for k in keys9)+'|')
    for c in ['B0','B1','B5']:lines.append('|'+c+'|'+'|'.join(f"{a[c][k]['mean']*100:.5f}" for k in METRICS)+'|')
    lines+=['','## 论文、源码与历史审计','',readme_audit(),'',
        '## 运行与限制','',f"正式墙钟{m.get('wall_seconds',0)/60:.2f}分钟，GPU0/GPU1各15任务150折，最多4worker/卡，2线程/worker；资源峰值与物理UUID见manifest。启动时GPU0存在其他训练，未中断或修改其他进程。", 
        f"失败/恢复：{json.dumps(m.get('failures_and_restarts',{}),ensure_ascii=False)}。每25epoch及折边界保存SHA256完整状态（模型、两Adam、早停、best及RNG），仅原子提交记录进入汇总；完成任务跳过。",
        f"同seed/fold所有配置主干初始化相同的正式配对为{m['validation']['common_initialization_equal_pairs']}/30；两种新分支各自家族初始化均一致。根源码及实验一至十五{len(m['protected_history_sha256'])}个历史文件保持SHA256不变。",
        '原val=test、共享标准化/RFE及checkpoint选取均按要求保留；测试参与选择，因此不是独立无泄漏泛化估计。三seed复用相同fold。固定252条边不是完整HO全脑图；文本不是个体诊断或ASD病理标签，ROI响应不能解释为因果脑区发现。',
        'B6常量仍是文本数值调制；不同节点查询可产生不同score。B7为身份标记，几何分布与FUNCTION不同；B8保留全部FUNCTION向量、仅改变对应，因而B5−B8联合REAL−SHUFFLED最有判别力。B9仍有三个映射且全参与，只将传播规则改成自身I。',
        'run.py、roi_pathway.py、integration_patch.diff与manifest能重建临时源码；冻结文本来自实验十五prompts.json。最终清除临时CLIP、源码及checkpoint，无受试者级预测或永久权重。未执行git commit/push。']
    atomic(RESULTS/'report.md','\n'.join(lines)+'\n');m=read_json(MANIFEST);m['scientific_judgment']=j
    if m['run_status']=='COMPLETE':m['report_sha256']=sha(RESULTS/'report.md')
    save_json(MANIFEST,m);return j


def readme_audit():
    return """实际阅读三份本地PDF正文、公式及架构图。SPromptGL的GFR为受试者间关系图；CAM的局部4×4×9是潜在Token而非ROI；SPGS是cosine→softmax→max标量门控，RES保留。其ABIDE消融由86.48经GFR达91.73、RES92.08、SPGS92.30，主要提升来自GFR。

BrainPrompt Eq.2把真实AAL116 ROI身份/功能LLM表示注入真实节点再传播。SABER Eq.1为节点语义，Eq.8–13含Graph-CLS、患者文本交互、门控及疾病监督；full/lite并非所有指标同向。两文数据、图谱及评估与本项目不同。本轮借鉴真实节点与传播，不复制AAL、LLM、超图、疾病头或额外损失。可核对本地PDF（哈希见manifest）、[SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[BrainPrompt](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf)、SABER arXiv:2607.01901v1。

本地六核心文件与GitHub main 52f9ae9cbe0a3b49ded4a78e51a358dda9632604逐字节一致；本地HEAD1167dd392837a02eb4f83481ae49f02607a9dae5。保留原GFR combined_feat覆盖、simple-2不加入ALoss、加权NLL+四图损失平均、Adam顺序、450/200、严格ACC提高选checkpoint及第五折低ACC保护。

实验十一端点用零基编号关联已核验roi_111_labels.csv，不使用旧映射的UNKNOWN名称：110可信HO ROI、252无向边，未知4个feature位置195/225/232/236直接delta恒0。统计基于默认LedoitWolf相关的Fisher-z，边权为tanh(z)，不是把标准化数据直接称作原Pearson系数。

实验六标量门控、七潜在双主题、十二/十三逐边静态与患者状态、十四层级与疾病头均未给出稳定语义必要性。实验九E9提升经十的疾病内容对照未证明ASD语义独特性。实验十五首次使用真实HO节点与邻居，A5 ACC91.806%，相对P0仅约+0.115pp；训练后正确/错配/常量/SELF_ONLY的delta与预测完全相同，OFF才改变概率，28/30近共同边偏移。源码逐行审计显示query在WD.11下收缩、signed消息微小、数值FC直通及decoder bias可绕过文本，(1+r)也保留无语义数值路径。

本轮B1准确复现十五A5；B2/B3/B4构成WD×FC直通2×2，保留注册参数数。B5–B9采用中心化score→纯乘法语义节点→正负归一化邻居→bias-free映射→对称边解码及未detach的零参考。仅在GFR与PLN之间增加Correlation residual，其余64维不变。没有标签进入编码器，也没有新Prompt或疾病语义头。

每config×seed只播种一次并连续十fold；新增参数fork_rng(150000+seed)隔离。在相同入口下旧参数相同；不同早停造成后续入口分叉，不能同时承诺每fold同初始化与原连续流。预检核对Gram与直接cosine的值/梯度、B1与十五A5两次Adam逐值一致、恢复模型/两Adam/RNG、合成非零语义与预测通路以及常量不会被永久零化。"""


def write_readme(m):
    text=['# 实验十六：ROI语义通路修复与医学语义必要性验证','',
        '固定10组，不搜索超参数。B0/P0与B1/实验十五A5均真实重训；B1–B4辨别正则化与FC直通；B5–B9检验修复后的医学语义与邻居必要性。','',
        '|组|定义|新增参数|新分支WD|','|---|---|---:|---:|']
    text+=[f"|{c}|{NAMES[i]}|{COUNTS[c]}|{.11 if c in ['B1','B3'] else 0}|" for i,c in enumerate(CONFIGS)]
    text+=['','## 固定机制','',
        '原数值输入871×320、48/6/10/256。110节点统计为关联252可信边的mean(z)、mean(abs(z))、population sd(z)、max(abs(z))、degree/maxdegree；度1的sd为0。冻结实验十五原始全部文本、RN50及固定无固定点置换，不另存或改写Prompt。','',
        '```text','B1–B4: 实验十五A5原模块；B3/B4只把decoder第65维FC输入置0。',
        'B5–B9:', 'h = GELU(Linear32(GELU(Linear5→32(stats))))',
        'score_i = cosine(Wq h_i, frozen_unit_CLIP_i)',
        'centered_i = score_i − mean_110_ROI(score)', 'response_i = tanh(centered_i/0.2)',
        'semantic_i = h_i * response_i (无1+response数值直通)',
        'A+ = max(tanh(z),0) / max(row_sum,1e−12)',
        'A− = max(−tanh(z),0) / max(row_sum,1e−12)',
        'm+ = A+ semantic; m− = A− semantic (B9两者均为I*semantic)',
        'updated = tanh(Wself semantic + Wpos m+ + Wneg m−); 三映射无bias',
        'context_ij = [updated_i+updated_j, abs(updated_i−updated_j)] (64维，无原FC)',
        'edge_signal = decoder(context) − decoder(zeros(context)); reference不detach',
        'decoder = Linear64→32 + GELU + Linear32→1(bias=False)',
        'delta = .3*tanh(alpha)*tanh(edge_signal); alpha0；252scatter到256，未知4维0',
        'Hnew[:,64:320] = H[:,64:320]+delta，前64维不变；原PLN中关闭SPGS，保留CAM/RES/分类器','```','',
        '零参考最后bias会恒等相消，B5–B9统一省略该冗余标量，新增39,201参数（比旧38,242多959）；其余decoder bias保留。参考使用同形状零输入且不detach，保证零节点严格零输出；广播参考只在预检中核对值与梯度误差。新分支只有alpha为0，数值/查询/图/解码按固定初始化；两个Adam步骤后检查梯度，不为梯度审计改变初始化。B6相同文本仍可因不同查询产生不同score，并非强制零响应。','',
        '原参数1,462,957完全注册保留；原两Adam组lr.0038/WD.11，新分支lr.0038、WD按表。dropout.35、n_hidden18、n_head2、nlayer1、simple-2损失及训练协议不变。B0保留原P0路径，B1–B9按十五A5关闭SPGS且完整保留RES。','',
        '## 预检与科学判定','',
        '预检核对原模型、B1历史前向与两步Adam、同入口旧参数、新分支家族初始化、零残差、参数组全覆盖、两步有效梯度、模型/两Adam/RNG恢复、Gram值与梯度、对称边、未知维、零输入零输出、无FC直通、常量可响应和不使用label。合成probe仅用于连通性，不比较测试性能或选参数。',
        '正式诊断不强制非零梯度/响应；近零使用属于真实负结果。记录训练梯度与抽样幅度峰值，最佳checkpoint各路径均值、最大值、参数范数及文本/SELF_ONLY/OFF依赖。B5六条件反事实180行，其余组同checkpoint轻量通路诊断只进每fold单行。',
        '“稳定收益”预先定义ACC、BA、Prob-AUC三个指标的三个seed差值全正；医学支持同时要求B5超过B6/B7/B8并且REAL−SHUFFLED同向；通路数值恢复预先采用≥24/30折文本替换Δ概率均值>1e−8。仅为保守描述判定，不是显著性检验，也不用于训练调参。','',
        '## 审计','',readme_audit(),'',
        '## 运行、恢复与文件','',f'`python {EXP}/run.py all`：prepare→preflight→30任务300fold→300诊断/180反事实→中文报告→追加handoff→清理。',
        f'`python {EXP}/run.py resume`：恢复model/两Adam/早停/best/RNG；`refresh`刷新；`report`重建报告；`prepare`与`preflight`可单独运行。',
        '每config×seed独立进程，10fold在同进程按原顺序，固定两卡各15任务150fold、最多4worker/卡、2线程/worker，不回退CPU。GPU逻辑cuda:0对应worker CUDA_VISIBLE_DEVICES；UUID及物理分配在manifest。每25epoch与折边界原子保存带SHA256完整状态，已完成任务跳过，失败恢复最近已验证状态，最新提交唯一，日志追加保留。完成后all/resume只验收，不重训。',
        '临时状态与源码在/tmp/spromptgl16-state-*，只修改临时network.py/model.py；原six源码及实验一至十五逐文件哈希保护。复核需读原源码+integration_patch.diff+run.py+roi_pathway.py+manifest；Prompt引用十五prompts.json。最终保留README、两个源码、diff、manifest、30任务日志与scheduler/preflight日志、三CSV和report，无重复文本、受试者预测、永久权重或临时源码。',
        '沿用StratifiedKFold10(shuffle=True,random_state10)、val=test、450/200及严格ACCcheckpoint。共享标准化/RFE与val=test限制单列报告。三seed复用fold，30折不视为独立样本；B5−B4同时改变多项机制且参数+959。未执行git commit/push。']
    atomic(EXP/'README.md','\n'.join(text)+'\n')


def append_handoff():
    m=read_json(MANIFEST);rows=read_csv(RESULTS/'fold_metrics.csv');ds=read_csv(RESULTS/'diagnostics.csv');cf=read_csv(RESULTS/'counterfactual.csv');j=scientific_judgment(rows,ds,cf)
    marker='## 实验十六：ROI语义通路修复与医学语义必要性验证'
    path=ROOT/'handoff.md';old=path.read_bytes()
    prefix=old[:m['protected_handoff_bytes']]
    assert hashlib.sha256(prefix).hexdigest()==m['protected_handoff_sha256'],'Existing handoff prefix changed; preserve and inspect'
    suffix=old[m['protected_handoff_bytes']:]
    assert not suffix.strip() or suffix.decode().lstrip().startswith(marker),'External append detected; preserve and inspect'
    chapter=[marker,'',f"COMPLETE：10组30任务300fold，300诊断、B5六条件180条反事实；GPU0/1各15任务150fold，4worker/卡、2线程/worker，正式{m['wall_seconds']/60:.2f}分钟，完成{m['completed_utc']}。",'',
        '前十五轮及既有handoff原字节完整保留。十五A5文本/邻居干预delta与概率逐值不变，query收缩、消息小、FC直通及decoder bias可能绕过文本；本轮不再改Prompt措辞，而检验路径修复。',
        'B0原P0、B1精确十五A5、B2新组WD0、B3关闭FC直通、B4两项；B5新耦合FUNCTION、B6CONSTANT、B7ANONYMOUS、B8SHUFFLED、B9SELF_ONLY。旧分支38,242，新耦合39,201参数；原组WD.11不变，B1/B3新组.11，其余新组0。',
        '新耦合：中心化cosine→tanh(score/.2)×数值节点→正负边权分开归一化→三bias-free映射→tanh→对称64维解码（无FC）→未detach零参考→.3tanh(alpha)残差，alpha0，110HO/252边，四未知delta0。没有新增疾病Prompt、损失、global head或超参数搜索。',
        f"版本：本地{m['git_head']}，GitHub核验{m['github_audit'].get('remote_main')}；六核心、三论文、输入/RFE、映射、CLIP、十五Prompt、源码/patch/split哈希见manifest。保护实验一至十五{len(m['protected_history_sha256'])}文件。", 
        '协议：871人，48/6/10/256；450epoch、wait>200、严格ACC首最佳；val=test、ASD0、Hard-AUC原口径，六指标。每config×seed连续十fold，仅初始播种，fork_rng隔离新模块；早停后自然RNG分叉，不伪称全部fold同初始化。','']
    chapter+=result_table(rows)
    chapter+=['','配对主结果（ACC/BA/Hard-AUC/Prob-AUC/SEN/SPE，pp）：']
    for left,right in PAIRS:chapter.append(left+'−'+right+'：'+'/'.join(f"{paired(rows,left,right,k)['mean']*100:+.5f}" for k in METRICS)+'；各seed及胜平负见report。')
    chapter+=['','2×2主效应与交互（相同六指标，pp）：']
    for name,terms in factorial().items():chapter.append(name+'：'+'/'.join(f"{contrast(rows,terms,k)['mean']*100:+.5f}" for k in METRICS)+'。')
    chapter+=['','B5反事实：']
    for condition in CF_CONDITIONS[1:]:
        rr=[r for r in cf if r['condition']==condition]
        chapter.append(condition+f"：翻转{mean_rows(rr,'flip_rate')*100:.5f}%；Δ概率均值{mean_rows(rr,'probability_change_abs_mean'):.6e}/最大{max(float(r['probability_change_abs_max']) for r in rr):.6e}；Δdelta均值{mean_rows(rr,'delta_change_abs_mean'):.6e}/最大{max(float(r['delta_change_abs_max']) for r in rr):.6e}；ACC/BA/Prob-AUC干预−REAL"+'/'.join(f"{mean_rows(rr,k+'_delta')*100:+.5f}" for k in ['acc','ba','probability_auc'])+'pp。')
    chapter+=['']+interpretation(rows,ds,cf,j)+['',
        'B0与十三F0/十四E00/十五A0六指标逐fold完全一致；B1与十五A5六指标、初始化、入口RNG、best/总epoch完全一致。九E9历史六指标均值见report，不能当作同初始化严格对照。',
        '预检PASS：历史两步Adam、初始化、梯度、Gram/零参考值与梯度、完整恢复、零输入、未知维、对称边与合成语义/邻居预测链；正式近零或负结果原样保留。三seed共用fold，val=test及原上游处理限制仍存在，不能外推独立泛化或ASD因果病灶。',
        f"任务恢复记录{json.dumps(m.get('failures_and_restarts',{}),ensure_ascii=False)}；仅最新原子提交，完成后临时模型/源码/cache清除，无本实验训练进程。目录{EXP}；报告{RESULTS/'report.md'}。未执行git commit/push。"]
    previous=next(ROOT.joinpath('experiment').glob('experiment9_*'))
    hist=[r for r in read_csv(previous/'results/fold_metrics.csv') if r['config']=='E9']
    hist_keys=['acc','balanced_accuracy','hard_auc','probability_auc','sensitivity_asd','specificity_asd']
    hist_means=[mean_rows(hist,k) for k in hist_keys];current=aggregates(rows)['B5']
    chapter+=['', '历史E9描述性六指标（ACC/BA/Hard-AUC/Prob-AUC/SEN/SPE）：'+'/'.join(f'{v*100:.5f}' for v in hist_means)+'%；B5−E9描述性差值'+'/'.join(f"{(current[k]['mean']-v)*100:+.5f}" for k,v in zip(METRICS,hist_means))+'pp。旧E9随机流/源码机制不同，此差值不作为严格配对因果比较。',
        '工程修复：预检中单行零参考广播与批量解码的浮点舍入出现约1e−8差异，改为同形状零输入且不detach，严格零节点零残差校验通过，数学与参数量不变。完成后补齐runner在BETWEEN_FOLDS checkpoint已提交而JSON尚未提交时的恢复同步；模拟最终fold的中断窗口验证通过。没有改动模型、优化器、RNG、指标或任何正式训练fold。']
    atomic(path,prefix+b'\n\n'+('\n'.join(chapter)+'\n').encode())


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
