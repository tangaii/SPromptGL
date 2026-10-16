#!/usr/bin/env python3
"""Frozen 20-arm experiment: historical parity, local evidence, patient edge-line graphs."""
import argparse, ast, copy, contextlib, csv, difflib, fcntl, hashlib, io, json, math, os, pickle
import random, re, shutil, subprocess, sys, tempfile, time, types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode = True
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
if 'preflight' in sys.argv or 'prepare' in sys.argv:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
EXP=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('SP20_ROOT','/data/adodas/SPromptGL-main')).resolve()
STATE=Path('/tmp')/('spromptgl20-state-'+hashlib.sha256(str(EXP).encode()).hexdigest()[:12])
MANIFEST,RESULTS,LOGS=EXP/'manifest.json',EXP/'results',EXP/'logs'
HANDOFF=EXP/'handoff.md'
CORE=['main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh']
AUDIT_COMMIT='f36ed2e9f06b39e65d9ae70c549256dd39cf8e80'
P9=ROOT/'experiment/experiment9_prompt_disease_adversarial_interaction'
P19=ROOT/'experiment/experiment19_contrastive_disease_evidence_alignment'
CONFIGS=[f'T{i:02}' for i in range(20)]
DESIGNS={}
for _i,_c in enumerate(CONFIGS):
    _condition=(['NUMERIC','REAL','NEUTRAL','WRONG_ANATOMY','ISOMETRIC','ANONYMOUS'][_i-4] if 4<=_i<=9 else
                ['NUMERIC','REAL','NEUTRAL','WRONG_ANATOMY'][_i-10] if 10<=_i<=13 else
                ['NUMERIC','REAL','NEUTRAL'][(_i-14)%3] if _i>=14 else None)
    DESIGNS[_c]=dict(prompt='P3' if _i in (1,2) or _i>=14 else 'P0',disease=_i in (2,3) or _i>=14,
                    adv=_i==3,condition=_condition,use_graph=_i in (10,11,12,13,17,18,19))
TASKS=[(c,s,(i*3+s)%2) for i,c in enumerate(CONFIGS) for s in range(3)]
METRICS=['acc','ba','hard_auc','probability_auc','sensitivity_asd','specificity_td']
CONDITIONS=['REAL','NEGATIVE_TEXT','ISOMETRIC','WRONG_ANATOMY','ANONYMOUS_EDGE_NAMES','NUMERIC_ONLY',
            'FC_BRANCH_SHUFFLED','BRANCH_OFF','GRAPH_NUMERIC']
PAIRS=[('T05','T04'),('T05','T06'),('T05','T07'),('T05','T08'),('T05','T09'),
       ('T11','T10'),('T11','T12'),('T11','T13'),('T11','T05'),('T15','T14'),('T15','T16'),('T15','T02'),
       ('T18','T17'),('T18','T19'),('T18','T15'),('T18','T02'),('T18','T00'),('T02','T01'),('T03','T00')]
PROTOCOL=dict(datadir=str(ROOT/'data')+'/',datname='ABIDE',cpu=False,gpu='-1',nepoch=450,early=200,
              lr=.0038,reg=.11,dropout=.35,nlayer=1,n_hidden=18,n_head=2,n_iter=10,nmodal=4,th=.9,
              GC_mode='weighted-cosine',MP_mode='GCN',MF_mode='concat',alpha=.5,
              theta_smooth=1.,theta_degree=.5,theta_sparsity=0.,nclass=2,mode='simple-2')
_CTX=None

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with open(temporary, "wb") as handle:
        handle.write(data.encode() if isinstance(data, str) else data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def save_json(path, value):
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, buf.getvalue())

def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))

def save_checkpoint(path, obj):
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    atomic(path, b"SP20" + hashlib.sha256(data).hexdigest().encode() + data)

def checkpoint(path):
    import torch
    data = Path(path).read_bytes()
    assert data[:4] == b"SP20" and hashlib.sha256(data[68:]).hexdigest().encode() == data[4:68]
    return torch.load(io.BytesIO(data[68:]), map_location="cpu", weights_only=False)

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

def restore_rng(saved):
    import numpy as np
    import torch
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["cpu"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda"]])

def rng_hash(saved):
    h = hashlib.sha256(pickle.dumps((saved["python"], saved["numpy"]), protocol=4))
    for v in [saved["cpu"]] + saved["cuda"]:
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()

def data():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    table = pd.read_csv(ROOT / "data/ABIDE/processed_standard_data.csv")
    mapping = np.load(ROOT / "data/ABIDE/modal_feat_dict.npy", allow_pickle=True).item()
    assert list(mapping) == ["PHENO", "ANAT", "FUNC", "Correlation"]
    assert [len(v) for v in mapping.values()] == [48, 6, 10, 256]
    assert table.columns[:-1].tolist() == [str(v) for fields in mapping.values() for v in fields]
    values = table.values
    assert values.shape == (871, 321) and np.isfinite(values).all()
    x, y = values[:, :-1], values[:, -1].astype(int) - 1
    assert int((y == 0).sum()) == 403 and int((y == 1).sum()) == 468
    pheno = [i for i, n in enumerate(mapping["PHENO"]) if str(n).startswith(("AGE_AT_SCAN_", "SEX_"))]
    sites = [i for i, n in enumerate(mapping["PHENO"]) if str(n).startswith("SITE_ID_")]
    assert len(pheno) == 28 and len(sites) == 20 and not set(pheno).intersection(sites)
    folds = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y))
    audit = dict(shape=list(x.shape), labels={"ASD_0": 403, "TD_1": 468},
                 modality_columns={k: list(map(str, v)) for k, v in mapping.items()},
                 pheno_age_sex_indices=pheno, site_indices=sites)
    return x, y, folds, audit

def assert_equal_tree(a, b):
    import numpy as np
    import torch
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for v, w in zip(a, b):
            assert_equal_tree(v, w)
    else:
        assert a == b

def acquire_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle

def summary(rows):
    import numpy as np
    result = {}
    configs = sorted(set(r["config"] for r in rows), key=lambda v: (v[0], int(v[1:])))
    for config in configs:
        result[config] = {}
        for key in METRICS:
            seeds = [float(np.mean([float(r[key]) for r in rows if r["config"] == config and int(r["seed"]) == seed])) for seed in range(3)]
            result[config][key] = dict(mean=float(np.mean(seeds)), sd=float(np.std(seeds)), seeds=seeds)
    return result

def paired(rows, left, right, key):
    import numpy as np
    lookup = {(r["config"], int(r["seed"]), int(r["fold"])):float(r[key]) for r in rows}
    differences = np.array([[lookup[left, seed, fold] - lookup[right, seed, fold] for fold in range(1, 11)] for seed in range(3)])
    return dict(mean=float(differences.mean()), seeds=differences.mean(1).tolist(),
                wins=int((differences > 1e-12).sum()), ties=int((np.abs(differences) <= 1e-12).sum()), losses=int((differences < -1e-12).sum()))

def correlation(x,y):
    import numpy as np
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    # A constant has no defined correlation; record 0 with a separate SD below.
    return float(np.corrcoef(x,y)[0,1]) if x.std()>1e-10 and y.std()>1e-10 else 0.

def read_json(path):
    # OSS/FUSE can briefly hide the target of an atomic rename.
    for attempt in range(8):
        try:
            return json.loads(Path(path).read_text())
        except (FileNotFoundError,json.JSONDecodeError,OSError):
            if attempt==7: raise
            time.sleep(.15*(attempt+1))


def historical_folders():
    return sorted(p for p in (ROOT/'experiment').iterdir() if p.is_dir() and
                  re.match(r'experiment\d+_',p.name) and 1<=int(p.name.split('_',1)[0][10:])<=19)


def build_baseline_sources():
    """Apply the exact accepted Experiment9 diff to disposable, hash-checked originals."""
    directory=STATE/'oracle'
    directory.mkdir(parents=True,exist_ok=True)
    for name in CORE: atomic(directory/name,(ROOT/name).read_bytes())
    result=subprocess.run(['patch','--batch','--fuzz=0','-p0','-i',str(P9/'alignment_patch.diff')],
                          cwd=directory,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    sources={name:(directory/name).read_text() for name in CORE}
    return sources


def patched_sources():
    src=build_baseline_sources()
    # The independent runner implements main's continuous ten-fold state machine.
    src['main.py']=(ROOT/'main.py').read_text()
    model,network=src['model.py'],src['network.py']
    # Remove only the unused old report writer, preserving all historical training operations.
    lines=model.splitlines(keepends=True)
    for node in reversed(ast.parse(model).body):
        if isinstance(node,ast.ClassDef) and node.name=='EvalHelper':
            for fn in reversed(node.body):
                if isinstance(fn,ast.FunctionDef) and fn.name=='write_pdai_outputs':
                    del lines[fn.lineno-1:fn.end_lineno]
    model=''.join(lines)
    anchor='    def forward(self, x):\n        bs = x.size(0)'
    assert network.count(anchor)==1
    network=network.replace(anchor,'    def forward(self, x, raw_modalities=None):\n        bs = x.size(0)')
    anchor='        text_features = self.prompt_text_features\n'
    assert network.count(anchor)==1
    network=network.replace(anchor,"        if hasattr(self, 'concept_evidence20'):\n            x = self.concept_evidence20(x, raw_modalities)\n"+anchor)
    anchor=next(line for line in model.splitlines() if line.startswith('    def run_epoch('))
    assert model.count(anchor)==1
    model=model.replace(anchor,"        from sp20_runtime import install_branch\n        install_branch(self)\n\n"+anchor)
    # Only substitute fusion forward calls; GFR, original losses, both original Adam steps unchanged.
    model=model.replace('self.ModalFusion(feat)','self.ModalFusion(feat, self._e20_raw)')
    assert 'self.ModalFusion(feat)' not in model
    src['model.py'],src['network.py']=model,network
    return src


def make_sources(task='preflight'):
    directory=STATE/'tasks'/task/'sources'
    directory.mkdir(parents=True,exist_ok=True)
    # Reconstruct only from the unique frozen diff, not from other experiment modules.
    for name in CORE: atomic(directory/name,(ROOT/name).read_bytes())
    result=subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(EXP/'integration_patch.diff')],cwd=directory,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    for name in CORE: assert sha(directory/name)==read_json(MANIFEST)['patched_source_sha256'][name]
    link=directory/'CLIP'
    if not link.exists():link.symlink_to(ROOT/'CLIP',target_is_directory=True)
    work=directory.parent/'work'; work.mkdir(exist_ok=True)
    for name in ('attn','graph'): (work/name).mkdir(exist_ok=True)
    os.chdir(work)
    return directory


def context():
    global _CTX
    if _CTX is None:
        import torch,numpy as np
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        directory=make_sources(os.environ.get('SP20_TASK','preflight'))
        sys.path.insert(0,str(directory));sys.modules['sp20_runtime']=sys.modules[__name__]
        import network,model
        x,y,folds,audit=data()
        with np.load(EXP/'semantic_cache.npz',allow_pickle=False) as packed:cache={k:packed[k].copy() for k in packed.files}
        _CTX=dict(torch=torch,network=network,model=model,x=x,y=y,folds=folds,audit=audit,cache=cache)
    return _CTX


def set_config(config):
    d=DESIGNS[config]
    os.environ.update(PDAI_PROMPT=d['prompt'],PDAI_DISEASE=str(int(d['disease'])),PDAI_ADV=str(int(d['adv'])),
                      PDAI_PROMPTS_FILE=str(P9/'prompts.json'))


def install_branch(a):
    from concept_evidence import ConceptEvidence20
    c=context();a._e20_raw=a.dataLayer(a.feat)[1]
    d=DESIGNS[c['current_config']]
    if d['condition'] is None:return
    torch=c['torch']
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if a.dev.type=='cuda' else []):
        torch.manual_seed(200000+int(a.hyperpm.seed))
        a.ModalFusion.concept_evidence20=ConceptEvidence20(condition=d['condition'],use_graph=d['use_graph'],
            cache=c['cache'],raw_corr=a._e20_raw[3],train_idx=a.trn_idx).to(a.dev)
    a.ModalFusion.concept_evidence20.prepare_graph(a._e20_raw[3])
    a.optimizer_MF.add_param_group(dict(params=list(a.ModalFusion.concept_evidence20.parameters()),lr=a.hyperpm.lr,weight_decay=0.))


def agent_for(config,seed,fold,oracle=False):
    c=context();set_config(config);c.update(current_config=config,current_fold=fold)
    cls=c['model'].EvalHelper
    if oracle:
        directory=STATE/'oracle';directory.mkdir(parents=True,exist_ok=True)
        src=build_baseline_sources()
        # Use oracle layers with exactly the same accepted V capture.
        net=types.ModuleType('sp20_oracle_network');sys.modules[net.__name__]=net
        exec(compile(src['network.py'],'oracle_network.py','exec'),net.__dict__)
        mod=types.ModuleType('sp20_oracle_model');sys.modules[mod.__name__]=mod
        exec(compile(src['model.py'].replace('from network import *','from sp20_oracle_network import *'),'oracle_model.py','exec'),mod.__dict__)
        cls=mod.EvalHelper
    tr,te=c['folds'][fold-1]
    with contextlib.redirect_stdout(io.StringIO()):
        a=cls([48,6,10,256],c['x'],c['y'],SimpleNamespace(**PROTOCOL,seed=seed),tr,te)
    a.MF_sav.close();a.GCMP_sav.close()
    assert (a.val_idx==a.tst_idx).all()
    if oracle:a._e20_raw=a.dataLayer(a.feat)[1]
    return a


def branch(a):return getattr(a.ModalFusion,'concept_evidence20',None)


def model_modules(a):
    modules=[('gfr',a.MessagePassing0),('pln',a.ModalFusion)]
    if a.ModalityDiscriminator is not None:modules.append(('discriminator',a.ModalityDiscriminator))
    return modules


def model_state(a):
    return {name:{k:v.detach().cpu().clone() for k,v in module.state_dict().items()} for name,module in model_modules(a)}


def load_state(a,saved):
    for name,module in model_modules(a):module.load_state_dict(saved[name])


def optim_state(a):
    return copy.deepcopy(dict(mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),
                discriminator=a.optimizer_D.state_dict() if a.optimizer_D else None))


def load_optim(a,saved):
    a.optimizer_MP0.load_state_dict(saved['mp']);a.optimizer_MF.load_state_dict(saved['mf'])
    if a.optimizer_D:a.optimizer_D.load_state_dict(saved['discriminator'])


def graph_state(a):return branch(a).graph_adj.detach().cpu().clone() if branch(a) and branch(a).use_graph else None


def load_graph(a,saved):
    if saved is not None:branch(a).graph_adj=saved.to(a.dev)


def weight_hash(a,auxiliary=False,legacy13=False):
    h=hashlib.sha256()
    for name,module in model_modules(a):
        for key,value in sorted(module.named_parameters()):
            if key.startswith('concept_evidence20.')!=auxiliary:continue
            tag=key if legacy13 else name+'.'+key
            h.update(tag.encode()+value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def optimizer_audit(a,config):
    groups=a.optimizer_MP0.param_groups+a.optimizer_MF.param_groups+(a.optimizer_D.param_groups if a.optimizer_D else [])
    parameters=[p for _,m in model_modules(a) for p in m.parameters()]
    supplied=[p for g in groups for p in g['params']]
    assert len(supplied)==len(set(map(id,supplied))) and set(map(id,supplied))==set(map(id,parameters))
    assert all(g['lr']==.0038 for g in groups)
    assert a.optimizer_MP0.param_groups[0]['weight_decay']==a.optimizer_MF.param_groups[0]['weight_decay']==.11
    assert all(g['weight_decay']==0. for g in a.optimizer_MF.param_groups[1:])
    total=sum(p.numel() for p in parameters)
    assert total==1462957+read_json(MANIFEST)['additional_parameters'][config],(config,total)
    return total


def forward(a,oracle=False):
    import torch
    a.MessagePassing0.eval();a.ModalFusion.eval()
    with torch.no_grad():
        n,parts=a.dataLayer(a.feat)
        hidden,_,_=a.MessagePassing0(parts,n,a.targ,a.trn_idx)
        if oracle:out,_,_=a.ModalFusion(hidden)
        else:out,_,_=a.ModalFusion(hidden,a._e20_raw)
        assert torch.isfinite(out).all()
    return out.clone()


def metrics(out,indices):
    import numpy as np
    from sklearn.metrics import confusion_matrix,roc_auc_score
    target=context()['y'][indices]
    probability=out.exp().detach().cpu().numpy()[indices];pred=probability.argmax(1)
    cm=confusion_matrix(target,pred,labels=[0,1])
    sen,spe=float(cm[0,0]/cm[0].sum()),float(cm[1,1]/cm[1].sum())
    return dict(acc=float((pred==target).mean()),ba=(sen+spe)/2,
                hard_auc=float(roc_auc_score(np.eye(2)[target],np.eye(2)[pred])),
                probability_auc=float(roc_auc_score(target==0,probability[:,0])),sensitivity_asd=sen,specificity_td=spe)


def step(a):
    with contextlib.redirect_stdout(io.StringIO()):a.run_epoch('simple-2')


def gradients(a):
    result={}
    if branch(a):
        b=branch(a)
        for name,pp in [('alpha',[b.alpha]),('encoder',b.encoder.parameters()),('edge_id',b.edge_id.parameters()),
                        ('readout',b.readout.parameters()),('graph_proj',b.graph_proj.parameters() if b.use_graph else [])]:
            grads=[p.grad for p in pp if p.grad is not None]
            assert all(bool(g.isfinite().all()) for g in grads)
            result[name]=math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads))
    return result

AUDIT={
    'gfr':'Four subject graphs; i==1 overwrites modality0 combined_feat; preserve this exact behavior.',
    'aloss':'Uses train labels only to construct its historical mask; ALoss is computed but not backpropagated.',
    'insertion':'After Encoder and FeedForward, [871,4,36], before all original SPGS/global/RES operations.',
    'latent_tokens':'Four 9D local tokens per modality are latent, never ROI nodes.',
    'protocol_limitations':'test=val and test ACC checkpoint selection; global standardization/supervised RFE before outer CV; transductive all-871 subject propagation. Not leakage-free independent generalization.',
    'exp9':'P3 32 sentences, two class prototypes; 36->1024, T=.2, CE=.1, gamma=.2*tanh(alpha); E3 original four-way GRL=.03.',
    'exp10':'ADHD/TD and neutral close to true ASD/TD: class-code capacity is an alternative explanation.',
    'exp11':'252 verified mapped FC edges; missing 195/225/232/236 remain only in original backbone.',
    'exp12_13':'ROI identity/state semantics not supported beyond numeric controls.',
    'exp14':'Small hierarchical gains not stable across BA and probability AUC; no ROI identities on latent tokens.',
    'exp15':'A4 local real-name clues; A5 text/neighbor changes failed to reach predictions.',
    'exp16':'Repaired sensitivity but stronger coupling lowered performance; medical necessity not supported.',
    'exp17_18':'Phenotype/selective alignment did not provide usable additional diagnostic information.',
    'exp19':'Neutral C5 ACC91.845 > REAL C4 91.538; free q@W_key absorbs query rotation; reject NULL/BCE evidence head.',
    'new_interpretation':'Line-graph nodes are retained measured FC connections; shared-ROI edges fixed. Patient FC and frozen text alter weights, not physical brain topology.',
}
LITERATURE={
    'MMGPL':dict(url='https://arxiv.org/pdf/2312.14574',version='author v2, 2024-06-27; MedIA 10.1016/j.media.2024.103225',
       audited='Fig1; Sec3.3 Eq7-10 GPT concepts/text encoding/token similarity/category weighting; Sec3.4 Eq11-14 semantic-profile graph+GCN+classification; Table1; Fig2 ablations.',
       boundary='Concept learning/semantic token graphs are established. B+G still computes semantic similarities; no full correct-vs-neutral/anonymous/isometric/wrong-ROI verification.',
       protocol='1029 ABIDE MRI+FC; 5-fold x3 seeds; frozen BiomedCLIP; ACC72.39%; not comparable to 871-subject SPromptGL.'),
    'SPromptGL':dict(url='https://papers.miccai.org/miccai-2025/0861-Paper5129.html',audited='Local original PDF, methods/figures/ablations; GFR subject graphs and prompt-weighted fusion precede this work.'),
    'MGDR':dict(url='https://papers.miccai.org/miccai-2024/paper/2494_paper.pdf',audited='MGRL and common/private SVD decomposition, contrastive objectives, attention fusion; multimodal graph representation not new.'),
    'BrainPrompt':dict(url='https://arxiv.org/abs/2504.16096',audited='Local 2025 PDF; ROI, subject and disease prompt injection already proposed.'),
}


def prepare():
    import numpy as np,torch
    if MANIFEST.exists():
        m=integrity(full=True)
        if m['experiment_directory']!=str(EXP):
            assert m['completed_folds']==0 and m['completed_tasks']==0,'Only an untrained staging directory may be relocated'
            m.update(experiment_directory=str(EXP),temporary_state=str(STATE),relocated_utc=utc(),run_status='PREPARED')
            m.pop('blocking_reason',None);save_json(MANIFEST,m);write_readme(m)
        return m
    LOGS.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True);STATE.mkdir(parents=True,exist_ok=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    audit_repo=ROOT if subprocess.run(['git','cat-file','-e',AUDIT_COMMIT],cwd=ROOT,capture_output=True).returncode==0 else Path('/tmp/spromptgl19-publish')
    comparison={}
    for name in CORE:
        historical=subprocess.check_output(['git','show',AUDIT_COMMIT+':'+name],cwd=audit_repo)
        current=(ROOT/name).read_bytes()
        comparison[name]=dict(local=hashlib.sha256(current).hexdigest(),audit=hashlib.sha256(historical).hexdigest(),equal=current==historical)
        assert current==historical,('Audit source mismatch; compare before patching',name)
    x,y,folds,audit=data()
    protected={str(p.relative_to(ROOT)):sha(p) for folder in historical_folders() for p in sorted(folder.rglob('*')) if p.is_file()}
    names=CORE+['README.md','data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy','data/ABIDE/data_processing.ipynb',
                'CLIP/clip/clip.py','CLIP/clip/model.py','CLIP/clip/simple_tokenizer.py',
                str((P9/'prompts.json').relative_to(ROOT)),str((P19/'prompts.json').relative_to(ROOT)),
                str((P9/'alignment_patch.diff').relative_to(ROOT)),
                'experiment/experiment11_roi_mapping_recovery/results/roi_111_labels.csv',
                'experiment/experiment11_roi_mapping_recovery/results/roi_256_named_mapping.csv']
    names+=list(dict.fromkeys(str(p.relative_to(ROOT)) for pattern in ('*SPromptGL*.pdf','*BrainPrompt*.pdf','*SABER*.pdf') for p in ROOT.glob(pattern)))
    sources={str(ROOT/n):sha(ROOT/n) for n in names}
    design=dict(configurations=DESIGNS,tasks=[dict(config=c,seed=s,physical_gpu=g) for c,s,g in TASKS],protocol=PROTOCOL,
                pairs=PAIRS,counterfactual_conditions=CONDITIONS,counterfactual_groups=['T05','T11','T15','T18'],
                prompt_files=dict(p3_and_classes=str(P9/'prompts.json'),concepts=str(P19/'prompts.json')),
                evidence=dict(local_parameters=19429,graph_parameters=20517,encoder=[3,32],readout=[136,64,36],alpha_initial=0.,residual_limit=.1,
                              prior_trainable=False,new_weight_decay=0.,initialization_seed='200000+seed inside fork_rng, after all Experiment9 initialization',
                              fold_statistics='all outer-train FC, without class labels',shuffle_seed='202000+fold; test-only derangement'),
                frozen_before_training=True,semantic_gate='Descriptive all-three-seed positive ACC/BA/Prob-AUC against all specified negative controls; no independent-fold p-values.')
    save_json(EXP/'design.json',design)
    from cache_reference import prepare_static_cache
    sys.path.insert(0,str(ROOT));from CLIP.clip import clip
    net,_=clip.load('RN50',device='cuda:0',jit=False);net.eval()
    for p in net.parameters():p.requires_grad_(False)
    encodings=[];token_counts=[];encoded_text_hash=[]
    def encode(texts):
        tokens=clip.tokenize(texts,truncate=False).to('cuda:0')
        assert tokens.shape==(len(texts),77) and int((tokens!=0).sum(-1).max())<=77
        token_counts.append(int((tokens!=0).sum(-1).max()));encoded_text_hash.append(canonical(texts))
        with torch.no_grad():v=net.encode_text(tokens).float().cpu()
        encodings.append(v);return v
    p19=read_json(P19/'prompts.json')
    cache=prepare_static_cache(ROOT,encode,p19['banks']['REAL'],p19['banks']['NEUTRAL'])
    assert len(encodings)==4
    from torch.nn import functional as F
    v=F.normalize(encodings[0],dim=-1).view(4,2,1024);qr=F.normalize(v[:,0]-v[:,1],dim=-1)
    qn=F.normalize(encodings[1],dim=-1).view(4,2,1024);qn=F.normalize(qn[:,0]-qn[:,1],dim=-1)
    raw_cos=(qr@F.normalize(encodings[2],dim=-1).T).numpy()
    b=np.linalg.qr(qr.numpy().astype('float64').T,mode='reduced')[0];rnd=np.random.default_rng(202020).normal(size=(1024,4));rnd-=b@(b.T@rnd)
    rb=np.linalg.qr(rnd,mode='reduced')[0];qiso=((qr.numpy().astype('float64')@b)@rb.T).astype('float32')
    gram_error=float(np.abs(qiso@qiso.T-qr.numpy()@qr.numpy().T).max());assert gram_error<1e-5
    semantic=dict(candidate_counts=cache['masks'].sum(1).tolist(),candidate_overlap=(cache['masks'].astype(int)@cache['masks'].T.astype(int)).tolist(),
        verified_edges=252,excluded=[195,225,232,236],gram_error=gram_error,random_original_subspace_dot_max=float(np.abs(qiso@qr.numpy().T).max()),
        clip_candidate_cos_std=[float(raw_cos[k,cache['masks'][k]].std()) for k in range(4)],
        prior_candidate_std={key:[float(cache[key][k,cache['masks'][k]].std()) for k in range(4)] for key in ('prior_real','prior_neutral','prior_random','prior_anonymous')},
        token_max=max(token_counts),encoded_texts_sha256=encoded_text_hash,edge_texts_changed=encoded_text_hash[2]!=encoded_text_hash[3],
        line_graph_directed_entries=int(cache['line_graph'].sum()),query_gram_only_control=True)
    semantic['status']='SEMANTIC_PRIOR_UNINFORMATIVE' if min(semantic['clip_candidate_cos_std'])<1e-4 else 'NONCONSTANT_PRIOR'
    validate_cache(cache)
    buf=io.BytesIO();np.savez_compressed(buf,**cache);atomic(EXP/'semantic_cache.npz',buf.getvalue())
    del net,encodings;torch.cuda.empty_cache()
    src=patched_sources()
    patch=''.join(''.join(difflib.unified_diff((ROOT/n).read_text().splitlines(True),src[n].splitlines(True),fromfile='a/'+n,tofile='b/'+n)) for n in CORE)
    atomic(EXP/'integration_patch.diff',patch)
    params={c:(36865 if d['disease'] else 0)+(1388 if d['adv'] else 0)+(20517 if d['use_graph'] else 19429 if d['condition'] else 0) for c,d in DESIGNS.items()}
    m=dict(experiment='SABER20 Grounded Concept Graph Alignment',schema_version=1,run_status='PREPARED',prepared_utc=utc(),root=str(ROOT),experiment_directory=str(EXP),
        temporary_state=str(STATE),git_head_at_prepare=head,github_audit_commit=AUDIT_COMMIT,source_comparison=comparison,protocol=PROTOCOL,configurations=DESIGNS,
        audit=AUDIT,literature_audit=LITERATURE,data_audit=audit,semantic_audit=semantic,additional_parameters=params,source_data_sha256=sources,
        protected_history_sha256=protected,patched_source_sha256={n:hashlib.sha256(src[n].encode()).hexdigest() for n in CORE},
        frozen_file_sha256={n:sha(EXP/n) for n in ['design.json','semantic_cache.npz','integration_patch.diff']},
        code_sha256={n:sha(EXP/n) for n in ['run.py','concept_evidence.py','cache_reference.py']},
        clip_weight_sha256=sha(Path.home()/'.cache/clip/RN50.pt'),split_sha256=canonical([dict(train=tr.tolist(),test=te.tolist()) for tr,te in folds]),
        handoff_prefix_bytes=(ROOT/'handoff.md').stat().st_size,handoff_prefix_sha256=sha(ROOT/'handoff.md'),
        preflight=None,completed_tasks=0,completed_folds=0,diagnostic_rows=0,counterfactual_rows=0,
        tasks={f'{c}_seed{s}':dict(config=c,seed=s,physical_gpu=g,status='PENDING',folds=0) for c,s,g in TASKS})
    save_json(MANIFEST,m);write_readme(m)
    print('PREPARED '+json.dumps(dict(parameters=params,semantic_audit=semantic)),flush=True)
    return m


def validate_cache(cache):
    import numpy as np
    assert cache['valid_idx'].shape==(252,) and sorted(set(range(256))-set(cache['valid_idx']))==[195,225,232,236]
    mask=cache['masks'];wrong=cache['masks_wrong'];assert mask.sum(1).tolist()==[72,56,45,32]
    assert np.array_equal(mask.astype(int)@mask.T.astype(int),wrong.astype(int)@wrong.T.astype(int))
    assert (cache['edge_permutation']!=np.arange(252)).all()
    assert np.array_equal(cache['line_graph'],cache['line_graph'].T) and np.diag(cache['line_graph']).all()
    for key in ('prior_real','prior_neutral','prior_random','prior_anonymous'):
        a=cache[key];assert a.shape==(4,252) and np.isfinite(a).all() and abs(a).max()<=1.
        assert all(a[k,mask[k]].std()>1e-7 for k in range(4)),key
    assert all(not np.array_equal(cache['prior_real'],cache[k]) for k in ('prior_neutral','prior_random','prior_anonymous'))


def integrity(full=False):
    m=read_json(MANIFEST)
    for name,h in m['source_data_sha256'].items():assert sha(name)==h,('source/data changed',name)
    for field in ('code_sha256','frozen_file_sha256'):
        for name,h in m[field].items():assert sha(EXP/name)==h,('frozen file changed',name)
    if full:
        current={str(p.relative_to(ROOT)) for f in historical_folders() for p in f.rglob('*') if p.is_file()}
        assert current==set(m['protected_history_sha256'])
        for name,h in m['protected_history_sha256'].items():assert sha(ROOT/name)==h,('history changed',name)
        assert sha(Path.home()/'.cache/clip/RN50.pt')==m['clip_weight_sha256']
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==m['git_head_at_prepare']
    return m


def signature():
    m=read_json(MANIFEST)
    return canonical({k:m[k] for k in ('source_data_sha256','code_sha256','frozen_file_sha256','protocol','configurations','split_sha256','clip_weight_sha256')})


def task_signature(config,seed):return canonical(dict(experiment_signature=signature(),config=config,seed=seed))

@contextlib.contextmanager
def intervention(a,condition,fold):
    """Only the added branch is changed; rebuild and restore every cached graph exactly."""
    import numpy as np,torch
    b=branch(a);assert b is not None
    saved=(b.prior.clone(),b.masks.clone(),b.condition,b.graph_adj,a._e20_raw)
    handle=None
    try:
        cache=context()['cache']
        if condition=='NEGATIVE_TEXT':b.prior.copy_(torch.as_tensor(cache['prior_neutral'],device=a.dev))
        elif condition=='ISOMETRIC':b.prior.copy_(torch.as_tensor(cache['prior_random'],device=a.dev))
        elif condition=='WRONG_ANATOMY':b.masks.copy_(torch.as_tensor(cache['masks_wrong'],device=a.dev))
        elif condition=='ANONYMOUS_EDGE_NAMES':b.prior.copy_(torch.as_tensor(cache['prior_anonymous'],device=a.dev))
        elif condition=='NUMERIC_ONLY':b.prior.zero_();b.condition='NUMERIC'
        elif condition=='FC_BRANCH_SHUFFLED':
            rng_=np.random.default_rng(202000+fold);idx=a.tst_idx;perm=rng_.permutation(idx)
            while np.any(idx==perm):perm=rng_.permutation(idx)
            raw=a._e20_raw[3].clone();raw[idx]=raw[perm]
            a._e20_raw=[*a._e20_raw[:3],raw]
        elif condition=='BRANCH_OFF':
            handle=b.register_forward_hook(lambda module,inputs,output:inputs[0])
        elif condition=='GRAPH_NUMERIC':
            assert b.use_graph
            b.condition='NUMERIC'
        elif condition!='REAL':raise ValueError(condition)
        if b.use_graph:b.prepare_graph(a._e20_raw[3])
        if condition=='GRAPH_NUMERIC':b.condition=saved[2]  # local prior remains REAL
        yield
    finally:
        if handle is not None:handle.remove()
        b.prior.copy_(saved[0]);b.masks.copy_(saved[1]);b.condition=saved[2]
        b.graph_adj=saved[3];a._e20_raw=saved[4]


def counterfactual(a,config,seed,fold,real):
    import torch
    b=branch(a);te=a.tst_idx;rows=[]
    pred=real[te].argmax(1);prob=real[te].exp();values=metrics(real,te)
    parameter_before=weight_hash(a,True)
    state_before=model_state(a)
    for condition in CONDITIONS:
        row=dict(config=config,seed=seed,fold=fold,condition=condition,applicable=True,retrained=False)
        if condition=='GRAPH_NUMERIC' and not b.use_graph:
            row.update(applicable=False,reason='NA: no graph installed')
            rows.append(row);continue
        with intervention(a,condition,fold):
            out=forward(a);m=metrics(out,te)
            row.update(**m,prediction_flip=float((out[te].argmax(1)!=pred).float().mean()),
                       probability_abs_change=float((out[te].exp()-prob).abs().mean()),
                       acc_change=m['acc']-values['acc'],ba_change=m['ba']-values['ba'],
                       probability_auc_change=m['probability_auc']-values['probability_auc'])
        restored=forward(a)
        assert torch.equal(restored,real),('CF REAL restoration failed',condition)
        rows.append(row)
    assert parameter_before==weight_hash(a,True)
    assert_equal_tree(state_before,model_state(a))
    assert len(rows)==9
    return rows


def diagnostics(a,config,seed,fold,real,tracker):
    import numpy as np,torch
    from torch.nn import functional as F
    d=DESIGNS[config];b=branch(a);te=a.tst_idx
    result=dict(config=config,seed=seed,fold=fold,condition=d['condition'] or 'NONE',use_graph=d['use_graph'],
                prediction_nonfinite=False,gradient_norms=json.dumps(tracker,sort_keys=True),
                disease_gamma=a.ModalFusion.last_disease_gamma,
                disease_probability_correlation=correlation(a.ModalFusion.last_disease_logits[te].softmax(-1)[:,0].detach().cpu().numpy(),
                                                           a.ModalFusion.last_base_logprob[te].exp()[:,0].detach().cpu().numpy()) if d['disease'] else '',
                train_reference_label_free=True)
    if not b:return result
    capture={}
    def hook(module,inputs,output):capture.update(base=inputs[0].detach().clone(),new=output.detach().clone())
    handle=b.register_forward_hook(hook)
    try:assert torch.equal(forward(a),real)
    finally:handle.remove()
    base,new=capture['base'],capture['new'];ll={k:v.clone() for k,v in b.last.items()}
    norm=base[:,3].norm(dim=-1);res=ll['delta'][:,3].norm(dim=-1)/(norm+1e-8)
    assert res.max()<=.100001 and ll['delta'][:,:3].count_nonzero()==0
    att=ll['attention'][te]
    entropy=-(att*att.clamp_min(1e-30).log()).sum(-1)
    z,dev,g=b.measure(a._e20_raw[3]);q=(dev/3).tanh();phi=torch.stack((q,q.abs(),z.tanh()),dim=-1)
    with torch.no_grad():
        encoded=b.encoder(phi)
        h=b.edge_norm(encoded+b.edge_id.weight[None]);no_id=b.edge_norm(encoded)
        h_after=b.graph_norm(h+F.gelu(b.graph_proj(torch.bmm(b.graph_adj,h)))) if b.use_graph else h
    mask_union=b.masks.any(0);active=(b.line_graph[mask_union].any(0) if b.use_graph else mask_union)
    domain_top=[];cache=context()['cache']
    for k in range(4):
        sel=b.prior[k].masked_fill(~b.masks[k],-1e4).topk(5).indices
        domain_top.append([int(b.valid_idx[i]) for i in sel.cpu().tolist()])
    result.update(alpha=float(b.alpha.detach()),residual_relative_mean=float(res[te].mean()),residual_relative_max=float(res.max()),
        residual_nonzero_fraction=float((ll['delta'][te,3].norm(dim=-1)>1e-10).float().mean()),noncorr_delta_max=float(ll['delta'][:,:3].abs().max()),
        representation_cosine=float(F.cosine_similarity(base[te,3],new[te,3],dim=-1).mean()),
        train_fc_mean_min=float(b.reference_mean.min()),train_fc_mean_max=float(b.reference_mean.max()),
        train_fc_std_min=float(b.reference_std.min()),train_fc_std_max=float(b.reference_std.max()),
        prior_candidate_std=json.dumps([float(b.prior[k,b.masks[k]].std(unbiased=False)) for k in range(4)]),
        top_feature_indices=json.dumps(domain_top),domain_attention_entropy=json.dumps(entropy.mean(0).cpu().tolist()),
        normalized_attention_entropy=json.dumps((entropy/torch.tensor(b.masks.sum(1).cpu().numpy(),device=a.dev).log()).mean(0).cpu().tolist()),
        domain_coverage=json.dumps(ll['coverage'][te].mean(0).cpu().tolist()),domain_signed_fc=json.dumps(ll['signed'][te].mean(0).cpu().tolist()),
        edge_id_feature_change=float((h[te]-no_id[te]).norm(dim=-1).mean()),edge_id_active_connections=int(active.sum()),
        message_relative_change=float(((h_after[te]-h[te]).norm(dim=-1)/(h[te].norm(dim=-1)+1e-8)).mean()),
        adjacency_patient_variance=float(b.graph_adj[te].var(0,unbiased=False)[b.line_graph].mean()) if b.use_graph else 0.,
        adjacency_outside_line_nonzero=int(b.graph_adj[:,~b.line_graph].count_nonzero()) if b.use_graph else 0,
        actual_gradient_parameter_count=sum(int((p.grad!=0).sum()) for p in b.parameters() if p.grad is not None),
        prior_requires_grad=b.prior.requires_grad,original_base_norm_mean=float(norm[te].mean()),
        delta_norm_mean=float(ll['delta'][te,3].norm(dim=-1).mean()))
    with intervention(a,'BRANCH_OFF',fold):off=forward(a)
    om=metrics(off,te);rm=metrics(real,te)
    result.update(branch_off_acc=om['acc'],branch_off_ba=om['ba'],branch_off_probability_auc=om['probability_auc'],
                  branch_off_probability_change=float((off[te].exp()-real[te].exp()).abs().mean()))
    if b.use_graph:
        b.use_graph=False
        try:out=forward(a);gm=metrics(out,te)
        finally:b.use_graph=True
        result.update(graph_off_acc=gm['acc'],graph_off_ba=gm['ba'],graph_off_probability_auc=gm['probability_auc'],
                      graph_off_probability_change=float((out[te].exp()-real[te].exp()).abs().mean()))
    assert torch.equal(forward(a),real)
    return result


def historical_reference(config,seed,fold):
    map_={'T00':'E0','T01':'E8','T02':'E9','T03':'E3'}
    if config not in map_:return None
    cfg=map_[config]
    rows=read_csv(P9/'results'/('reference.csv' if cfg in ('E0','E8') else 'fold_metrics.csv'))
    row=next(r for r in rows if r['config']==cfg and int(r['seed'])==seed and int(r['fold'])==fold)
    keymap={'ba':'balanced_accuracy','specificity_td':'specificity_asd'}
    result={k:float(row[keymap.get(k,k)]) for k in METRICS if row.get(keymap.get(k,k),'')!=''}
    if config=='T00':
        extra=next(r for r in read_csv(P19/'results/fold_metrics.csv') if r['config']=='C0' and int(r['seed'])==seed and int(r['fold'])==fold)
        result.update({k:float(extra[k]) for k in METRICS})
        result.update(best_epoch=int(extra['best_epoch']),epochs_executed=int(extra['epochs_executed']),initial_trunk_legacy13_sha256=extra['initial_trunk_legacy13_sha256'],entry_rng_sha256=extra['entry_rng_sha256'])
        p13=next((ROOT/'experiment').glob('experiment13_*'))
        f0=next(r for r in read_csv(p13/'results/fold_metrics.csv') if r['config']=='F0' and int(r['seed'])==seed and int(r['fold'])==fold)
        assert all(abs(float(f0[k])-result[k])<1e-12 for k in METRICS),'Inconsistent C0/F0 history'
        assert int(f0['best_epoch'])==result['best_epoch'] and int(f0['epochs_executed'])==result['epochs_executed']
    if cfg=='E8':
        # Exp1 CSV truncated ACC/AUC to four decimals and SEN/SPE to six.
        # Its original log retains full SEN with the original denominator+1e-8.
        # Recover the unique integer confusion counts, then compare exact metrics.
        old=ROOT/'experiment/experiment1_modality_aware_semantic_prompt'
        text=(old/f'logs/P3_seed{seed}.log').read_text()
        sensitivities=[ast.literal_eval(v)[-1] for v in re.findall(r'^sens: (\[.*\])$',text,re.M)]
        assert len(sensitivities)==10
        _,labels,folds,_=data();test=folds[fold-1][1]
        totals=[int((labels[test]==i).sum()) for i in (0,1)]
        correct=[round(value*(n+1e-8)) for value,n in zip(sensitivities[fold-1],totals)]
        assert all(abs(value-count/(n+1e-8))<1e-13 for value,count,n in zip(sensitivities[fold-1],correct,totals))
        sen,spe=[count/n for count,n in zip(correct,totals)]
        exact=dict(acc=sum(correct)/sum(totals),ba=(sen+spe)/2,hard_auc=(sen+spe)/2,sensitivity_asd=sen,specificity_td=spe)
        assert all(f'{exact[k]:.4f}'==f'{result[k]:.4f}' for k in ('acc','hard_auc'))
        assert all(abs(result[k]-v)<.5001e-6 for k,v in zip(('sensitivity_asd','specificity_td'),sensitivities[fold-1]))
        result.update(exact)
    if cfg in ('E3','E9','E8'):
        if cfg!='E8':text=(P9/f'logs/{cfg}_seed{seed}.log').read_text()
        chunks=re.split(r'Counter\(\{.*?\}\)',text)[1:]
        assert len(chunks)==10,(cfg,seed,len(chunks))
        epochs=re.findall(r'\b(\d+)/450\b[^\n]*?val-auc: [\d.]+\s+acc: ([\d.]+)',chunks[fold-1])
        assert epochs,(cfg,seed,fold)
        # Logs round ACC to four decimals; best index is still unambiguous for discrete test accuracy.
        best=-1.;bestepoch=-1
        for e,v in epochs:
            if float(v)>best:best=float(v);bestepoch=int(e)
        result.update(best_epoch=bestepoch,epochs_executed=int(epochs[-1][0])+1)
    return result


def assert_historical_fold(row):
    ref=historical_reference(row['config'],row['seed'],row['fold'])
    if ref is None:return
    for key,value in ref.items():
        if isinstance(value,str):assert row[key]==value,('historical hash',row['config'],row['seed'],row['fold'],key)
        else:assert abs(row[key]-value)<1e-12,('HISTORICAL_PARITY_FAILED',row['config'],row['seed'],row['fold'],key,row[key],value)

def preflight():
    import numpy as np,torch
    from torch.nn import functional as F
    from concept_evidence import ConceptEvidence20
    m=integrity(full=True)
    if m.get('preflight',{} ) and m['preflight']['status']=='PASS':return m['preflight']
    c=context();validate_cache(c['cache']);checks=dict(status='RUNNING',started_utc=utc(),checks={},baseline_parity={})
    # Standalone tests do not train or choose any held-out checkpoint.
    raw=torch.linspace(-2,2,21*256,device='cuda').view(21,256);base=torch.randn(21,4,36,device='cuda');tr=np.arange(16)
    for condition in ('NUMERIC','REAL','NEUTRAL','WRONG_ANATOMY','ISOMETRIC','ANONYMOUS'):
        for graph in (False,True):
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(200000);b=ConceptEvidence20(condition=condition,use_graph=graph,cache=c['cache'],raw_corr=raw,train_idx=tr).cuda()
            b.prepare_graph(raw)
            assert sum(p.numel() for p in b.parameters())==(20517 if graph else 19429)
            parts=[torch.zeros(21,48,device='cuda'),torch.zeros(21,6,device='cuda'),torch.zeros(21,10,device='cuda'),raw]
            assert torch.equal(b(base,parts),base)
            assert b.last['attention'][:,~b.masks].count_nonzero()==0
            assert torch.allclose(b.last['attention'].sum(-1),torch.ones(21,4,device='cuda'),atol=1e-6)
            assert not b.prior.requires_grad and not any(isinstance(mod,torch.nn.Linear) and mod.out_features==1024 for mod in b.modules())
            opt=torch.optim.Adam(b.parameters(),lr=.0038,weight_decay=0.)
            for step_number in range(2):
                opt.zero_grad();o=b(base,parts);loss=(o*torch.arange(36,device='cuda')).sum();loss.backward()
                if step_number==0:assert b.alpha.grad is not None and float(b.alpha.grad.abs())>0
                else:
                    for name,mod in [('encoder',b.encoder),('edge_id',b.edge_id),('readout',b.readout)]+([('graph_proj',b.graph_proj)] if graph else []):
                        gg=[p.grad for p in mod.parameters()];assert all(g is not None and torch.isfinite(g).all() for g in gg) and sum(float(g.abs().sum()) for g in gg)>0,(name,condition)
                opt.step()
            with torch.no_grad():b.alpha.fill_(100);o=b(base,parts)
            delta=o-base;assert delta[:,:3].count_nonzero()==0
            assert float((delta[:,3].norm(dim=-1)/(base[:,3].norm(dim=-1)+1e-8)).max())<=.100001
    checks['checks']['twelve_reference_module_cases']='PASS: sizes, zero-alpha, masks, gradients, graph row sums, residual bound, no free Key'
    # Three updates of every historical path, including two original Adams and E3 discriminator.
    oracle_initial={}
    for config in CONFIGS[:4]:
        c['network']._prompt_text_features={};c['network']._pdai_disease_features=None
        seed_rng(0);entry=rng();native=agent_for(config,0,1,oracle=True)
        native_init=model_state(native);native_rng=rng();native_out=forward(native,oracle=True)
        expected=[]
        for _ in range(3):step(native);expected.append((model_state(native),optim_state(native),rng(),forward(native,oracle=True)))
        c['network']._prompt_text_features={};c['network']._pdai_disease_features=None;restore_rng(entry)
        a=agent_for(config,0,1);optimizer_audit(a,config)
        assert_equal_tree(native_init,model_state(a));assert rng_hash(native_rng)==rng_hash(rng())
        assert torch.equal(native_out,forward(a))
        oracle_initial[config]=(native_init,native_rng,native_out,entry)
        if config=='T03':
            recovery_path=STATE/'e3_preflight.pt'
            save_checkpoint(recovery_path,dict(model=model_state(a),optim=optim_state(a),rng=rng()))
        for target_model,target_opt,target_rng,target_out in expected:
            step(a);assert_equal_tree(target_model,model_state(a));assert_equal_tree(target_opt,optim_state(a))
            assert rng_hash(target_rng)==rng_hash(rng()) and torch.equal(target_out,forward(a)),(config,'three original Adam steps')
        if config=='T03':
            point=checkpoint(recovery_path);load_state(a,point['model']);load_optim(a,point['optim']);restore_rng(point['rng'])
            for sm,so,sr,out in expected:
                step(a);assert_equal_tree(sm,model_state(a));assert_equal_tree(so,optim_state(a));assert rng_hash(sr)==rng_hash(rng());assert torch.equal(out,forward(a))
            recovery_path.unlink()
        checks['baseline_parity'][config]=dict(initial_parameters_exact=True,initial_predictions_exact=True,entry_and_post_rng_exact=True,three_optimizer_steps_exact=True)
        del a,native,expected;torch.cuda.empty_cache()
    # Every actual configuration keeps the corresponding backbone and complete lazy-CLIP RNG.
    first_branches={}
    for config in CONFIGS[4:]:
        family='T02' if DESIGNS[config]['disease'] else 'T00'
        native_state,native_rng,native_out,entry=oracle_initial[family]
        c['network']._prompt_text_features={};c['network']._pdai_disease_features=None;restore_rng(entry)
        a=agent_for(config,0,1);optimizer_audit(a,config);b=branch(a)
        saved=model_state(a);saved['pln']={k:v for k,v in saved['pln'].items() if not k.startswith('concept_evidence20.')}
        assert_equal_tree(native_state,saved)
        assert rng_hash(native_rng)==rng_hash(rng()) and torch.equal(forward(a),native_out),(config,'zero-residual historical parity')
        key=b.use_graph;h=weight_hash(a,True)
        if key not in first_branches:first_branches[key]=h
        assert h==first_branches[key],(config,'auxiliary parameter initialization')
        del a;torch.cuda.empty_cache()
    checks['checks']['twenty_paths_zero_initialization']='PASS; common tensors and RNG exact; local/graph family auxiliary parameter hashes exact'
    # Full-data graph model three-step checkpoint restoration, all RNG/Adam/buffers/adjacency.
    c['network']._prompt_text_features={};c['network']._pdai_disease_features=None;seed_rng(0)
    a=agent_for('T18',0,1);step(a)
    assert gradients(a)['alpha']>0,'Integrated alpha must receive first-step gradient'
    restore_point=dict(model=model_state(a),optim=optim_state(a),rng=rng(),graph=graph_state(a))
    target=[]
    for _ in range(3):
        step(a);gg=gradients(a)
        assert all(gg[k]>0 for k in ('encoder','edge_id','readout','graph_proj')),'Integrated second-step gradients'
        target.append((model_state(a),optim_state(a),rng(),forward(a)))
    cp=STATE/'preflight_recovery.pt';save_checkpoint(cp,restore_point);restored=checkpoint(cp)
    load_state(a,restored['model']);load_optim(a,restored['optim']);load_graph(a,restored['graph']);restore_rng(restored['rng'])
    for sm,so,sr,out in target:
        step(a);assert_equal_tree(sm,model_state(a));assert_equal_tree(so,optim_state(a));assert rng_hash(sr)==rng_hash(rng());assert torch.equal(out,forward(a))
    cp.unlink();checks['checks']['three_step_atomic_recovery']='PASS: all models, original Adams, RNG, buffers, graph_adj; discriminator separately verified above'
    # New branch never receives labels; full original train loss and gradients unchanged under held-out label permutation.
    start=dict(model=model_state(a),optim=optim_state(a),rng=rng(),graph=graph_state(a));original_labels=a.targ.clone()
    step(a);expected_state=model_state(a);expected_opt=optim_state(a);expected_rng=rng()
    load_state(a,start['model']);load_optim(a,start['optim']);load_graph(a,start['graph']);restore_rng(start['rng'])
    perm=np.random.default_rng(202020).permutation(a.tst_idx);a.targ[a.tst_idx]=original_labels[perm]
    branch(a).prepare_graph(a._e20_raw[3]);assert torch.equal(branch(a).graph_adj,start['graph'].to(a.dev))
    step(a);assert_equal_tree(expected_state,model_state(a));assert_equal_tree(expected_opt,optim_state(a));assert rng_hash(expected_rng)==rng_hash(rng())
    a.targ.copy_(original_labels);checks['checks']['heldout_label_noninterference']='PASS: train update and cached graph exactly unchanged, test labels never used by new statistics'
    real=forward(a);cf=counterfactual(a,'T18',0,1,real);assert len(cf)==9 and all(r['applicable'] for r in cf)
    checks['checks']['counterfactual_cache_and_parameter_restoration']='PASS: all 9 conditions, REAL exact after every intervention; parameters unchanged'
    checks.update(status='PASS',completed_utc=utc(),formal_fold_training=False,historical_full_fold_parity='Pending actual T00-T03 120 formal folds; scheduler gates other groups until all pass.')
    m=integrity(full=True);m['preflight']=checks;save_json(MANIFEST,m)
    print('PREFLIGHT_PASS '+json.dumps(checks,ensure_ascii=False),flush=True)
    return checks


def worker(config,seed):
    import torch
    m=integrity();task=f'{config}_seed{seed}';gpu=int(os.environ['SP20_PHYSICAL_GPU'])
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)==str(m['tasks'][task]['physical_gpu'])
    lock=acquire_lock(STATE/(task+'.lock'));path=STATE/(task+'.json');cp=STATE/(task+'.pt');sig=task_signature(config,seed)
    if path.exists() and read_json(path)['status']=='COMPLETE':
        result=read_json(path);assert result['signature']==sig and len(result['metrics'])==10;print('SKIP_COMPLETE '+task,flush=True);return
    c=context();set_config(config)
    saved=checkpoint(cp) if cp.exists() else None
    if saved:
        assert saved['signature']==sig;result=saved['result']
        # Warm exactly the historical process caches, then restore its uninterrupted RNG.
        c['network'].get_prompt_text_features()
        if DESIGNS[config]['disease']:c['network'].get_pdai_disease_prototypes()
        restore_rng(saved['rng']);start_fold=saved['fold']+(saved['phase']=='BETWEEN_FOLDS')
        if saved['phase']=='BETWEEN_FOLDS':save_json(path,result)
        print(f'RESUME {task} fold={start_fold} phase={saved["phase"]}',flush=True)
    else:
        seed_rng(seed);start_fold=1
        result=dict(signature=sig,task=task,config=config,seed=seed,physical_gpu=gpu,status='RUNNING',metrics=[],diagnostics=[],counterfactual=[],started_utc=utc(),seconds=0.)
    print(f'WORKER {task} physical_gpu={gpu} logical=cuda:0 signature={sig}',flush=True)
    for fold in range(start_fold,11):
        entry=rng();a=agent_for(config,seed,fold);optimizer_audit(a,config)
        initial,legacy,aux=weight_hash(a),weight_hash(a,legacy13=True),weight_hash(a,True)
        tracker={k:0. for k in gradients(a)};best,best_epoch,best_model,wait,start,prior_seconds=0.,-1,None,0,0,0.
        if saved and saved['phase']=='EPOCH' and saved['fold']==fold:
            load_state(a,saved['model']);load_optim(a,saved['optim']);load_graph(a,saved['graph']);restore_rng(saved['rng'])
            best,best_epoch,best_model,wait,start=[saved[k] for k in ('best_acc','best_epoch','best_model','wait','next_epoch')]
            tracker,entry,initial,legacy,aux,prior_seconds=[saved[k] for k in ('tracker','entry_rng','initial','legacy','aux','fold_seconds')]
        saved=None;started=time.monotonic();last=start-1
        for epoch in (range(start,450) if wait<=200 else []):
            step(a)
            if epoch in (0,1) or (epoch+1)%25==0:
                for key,value in gradients(a).items():tracker[key]=max(tracker[key],value)
            out=forward(a);val=float(out[a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean())
            if val>best:best,best_epoch,best_model,wait=val,epoch,model_state(a),0
            else:wait+=1
            last=epoch;stop=wait>200 or epoch==449
            if (epoch+1)%25==0 or stop:
                save_checkpoint(cp,dict(signature=sig,phase='EPOCH',fold=fold,next_epoch=epoch+1,model=model_state(a),optim=optim_state(a),graph=graph_state(a),rng=rng(),
                    best_acc=best,best_epoch=best_epoch,best_model=best_model,wait=wait,tracker=tracker,entry_rng=entry,initial=initial,legacy=legacy,aux=aux,
                    fold_seconds=prior_seconds+time.monotonic()-started,result=result))
                print(f'EPOCH fold={fold} epoch={epoch} best_acc={best:.12f} wait={wait}',flush=True)
            if stop:break
        assert best_model is not None
        continuation=rng();load_state(a,best_model);out=forward(a);values=metrics(out,a.tst_idx)
        assert abs(values['acc']-best)<1e-12 and abs(values['ba']-values['hard_auc'])<1e-12
        row=dict(config=config,seed=seed,fold=fold,physical_gpu=gpu,task_signature=sig,**values,best_epoch=best_epoch,epochs_executed=last+1,
                 selected_val_acc=best,train_n=len(a.trn_idx),test_n=len(a.tst_idx),val_equals_test=True,initial_trunk_sha256=initial,
                 initial_trunk_legacy13_sha256=legacy,initial_branch_sha256=aux if branch(a) else '',entry_rng_sha256=rng_hash(entry),additional_parameters=m['additional_parameters'][config])
        assert_historical_fold(row)
        diagnostic=diagnostics(a,config,seed,fold,out,tracker);diagnostic['task_signature']=sig
        cf=counterfactual(a,config,seed,fold,out) if config in ('T05','T11','T15','T18') else []
        for r in cf:r['task_signature']=sig
        row['seconds']=prior_seconds+time.monotonic()-started
        result['metrics'].append(row);result['diagnostics'].append(diagnostic);result['counterfactual'].extend(cf)
        restore_rng(continuation)
        result.update(seconds=sum(r['seconds'] for r in result['metrics']),status='COMPLETE' if fold==10 else 'RUNNING',updated_utc=utc())
        save_checkpoint(cp,dict(signature=sig,phase='BETWEEN_FOLDS',fold=fold,rng=continuation,result=result));save_json(path,result)
        print('FOLD_COMPLETE '+json.dumps(row),flush=True)
        del a,best_model;torch.cuda.empty_cache()
    cp.unlink(missing_ok=True);shutil.rmtree(STATE/'tasks'/task,ignore_errors=True)
    print('TASK_COMPLETE '+task,flush=True);lock.close()

def live_workers():
    import psutil
    result={}
    for p in psutil.process_iter(['pid','cmdline','status']):
        try:
            args=p.info['cmdline'] or []
            if str(EXP/'run.py') in args and '--worker' in args and p.info['status']!='zombie':
                i=args.index('--worker');result[f'{args[i+1]}_seed{args[i+2]}']=p.pid
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return result


def task_results():
    out=[]
    for c,s,g in TASKS:
        path=STATE/f'{c}_seed{s}.json'
        if not path.exists():continue
        t=read_json(path);assert t['signature']==task_signature(c,s) and t['physical_gpu']==g
        assert [r['fold'] for r in t['metrics']]==list(range(1,len(t['metrics'])+1))
        assert len(t['diagnostics'])==len(t['metrics'])
        out.append(t)
    if not out and read_json(MANIFEST)['completed_folds']==600:
        # Finalization may be interrupted after deleting temporary checkpoints. Never retrain completed CV.
        metrics_=read_csv(RESULTS/'fold_metrics.csv');diag_=read_csv(RESULTS/'diagnostics.csv');cf_=read_csv(RESULTS/'counterfactual.csv')
        for c,s,g in TASKS:
            select=lambda rr:[r for r in rr if r['config']==c and int(r['seed'])==s]
            rr=select(metrics_);assert len(rr)==10
            for r in rr:
                for k in ('seed','fold','physical_gpu','best_epoch','epochs_executed'):r[k]=int(r[k])
                for k in METRICS+['seconds']:r[k]=float(r[k])
            out.append(dict(task=f'{c}_seed{s}',config=c,seed=s,physical_gpu=g,signature=task_signature(c,s),status='COMPLETE',metrics=rr,diagnostics=select(diag_),counterfactual=select(cf_),seconds=sum(r['seconds'] for r in rr),updated_utc=utc()))
    return out


def refresh():
    lock=acquire_lock(EXP/'logs/.refresh.lock')
    try:
        m=read_json(MANIFEST)
        if m['run_status']=='COMPLETE' and not STATE.exists():return m
        completed=task_results();rows=[r for t in completed for r in t['metrics']]
        ds=[r for t in completed for r in t['diagnostics']];cf=[r for t in completed for r in t['counterfactual']]
        assert len(set((r['config'],r['seed'],r['fold']) for r in rows))==len(rows)
        for t in completed:m['tasks'][t['task']].update(status=t['status'],signature=t['signature'],folds=len(t['metrics']),seconds=t['seconds'],updated_utc=t['updated_utc'])
        m.update(completed_tasks=sum(t['status']=='COMPLETE' for t in completed),completed_folds=len(rows),diagnostic_rows=len(ds),counterfactual_rows=len(cf),updated_utc=utc())
        for name,v in [('fold_metrics',rows),('diagnostics',ds),('counterfactual',cf)]:save_csv(RESULTS/(name+'.csv'),v)
        save_json(MANIFEST,m);return m
    finally:lock.close()


def worker_env(gpu,task='preflight'):
    return dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),SP20_PHYSICAL_GPU=str(gpu),SP20_TASK=task,
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')


def resources():
    import psutil
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
    cards=[]
    for line in raw.strip().splitlines():
        index,name,total,used,utilization=[v.strip() for v in line.split(',')]
        cards.append(dict(index=int(index),name=name,total_mib=int(total.split()[0]),used_mib=int(used.split()[0]),utilization=int(utilization.split()[0])))
    rss=0
    for pid in live_workers().values():
        try:rss+=psutil.Process(pid).memory_info().rss
        except psutil.NoSuchProcess:pass
    cg=Path('/sys/fs/cgroup')
    if (cg/'memory.max').exists():
        limit=(cg/'memory.max').read_text().strip();used=int((cg/'memory.current').read_text())
        memstat=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
        reclaimable=int(memstat.get('inactive_file',0));cgroup_version=2
    else:
        cg=cg/'memory';limit=(cg/'memory.limit_in_bytes').read_text().strip();used=int((cg/'memory.usage_in_bytes').read_text())
        memstat=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
        reclaimable=int(memstat.get('total_inactive_file',memstat.get('inactive_file',0)));cgroup_version=1
    return dict(utc=utc(),gpu=cards,worker_rss_gib=rss/2**30,host_available_gib=psutil.virtual_memory().available/2**30,
                cgroup_available_gib=(int(limit)-used+reclaimable)/2**30 if limit!='max' else 999.,active_workers=len(live_workers()),
                cgroup_version=cgroup_version,cpu_load_1m=os.getloadavg()[0],cpu_affinity=len(os.sched_getaffinity(0)))


def schedule(workers_per_gpu=4):
    assert 1<=workers_per_gpu<=4
    m=prepare()
    if m['run_status']=='COMPLETE':validate_complete();print('ALREADY_COMPLETE 60 tasks / 600 folds',flush=True);return
    lock=acquire_lock(STATE/'scheduler.lock')
    if not m.get('preflight') or m['preflight']['status']!='PASS':
        with open(LOGS/'preflight.log','a') as out:subprocess.run([sys.executable,str(EXP/'run.py'),'preflight'],env=worker_env(0),stdout=out,stderr=subprocess.STDOUT,check=True)
    m=read_json(MANIFEST);m.update(run_status='RUNNING',workers_per_gpu=workers_per_gpu,formal_training_started=True,staging_only=False)
    m.setdefault('started_unix',time.time());m.setdefault('started_utc',utc());m['launch_resources']=resources();save_json(MANIFEST,m)
    active={t:dict(pid=p,process=None,gpu=m['tasks'][t]['physical_gpu']) for t,p in live_workers().items()}
    failures=dict(m.get('failures_and_restarts',{}));limits={0:workers_per_gpu,1:workers_per_gpu};last_refresh=last_resources=0.
    print('SCHEDULER_START '+utc(),flush=True)
    try:
        while True:
            for task,info in list(active.items()):
                code=info['process'].poll() if info['process'] else (None if Path('/proc',str(info['pid'])).exists() else 0)
                if code is None:continue
                if 'log' in info:info['log'].close()
                del active[task]
                if code!=0:
                    failures[task]=failures.get(task,0)+1
                    text=(LOGS/(task+'.log')).read_text()[-10000:]
                    if 'HISTORICAL_PARITY_FAILED' in text or failures[task]>=3:raise RuntimeError('Task failed and formal run stopped: '+task+'; '+text[-1200:])
                    print('RECOVER '+task+' from atomic checkpoint',flush=True)
            done={t['task'] for t in task_results() if t['status']=='COMPLETE'}
            baseline_done=all(f'{c}_seed{s}' in done for c in CONFIGS[:4] for s in range(3))
            if baseline_done and not read_json(MANIFEST).get('historical_parity_120'):
                m=read_json(MANIFEST);m['historical_parity_120']=dict(status='PASS',folds=120,groups=CONFIGS[:4],metrics='All available historical metrics and E3/E9 best/executed epochs',utc=utc());save_json(MANIFEST,m)
                print('HISTORICAL_PARITY_PASS 120/120; release T04-T19',flush=True)
            # Fixed GPU task assignment; long graph tasks first once historical paths are verified.
            pending=sorted(TASKS,key=lambda t:(not DESIGNS[t[0]]['use_graph'],t[0],t[1])) if baseline_done else TASKS
            for gpu in (0,1):
                for config,seed,assigned in pending:
                    task=f'{config}_seed{seed}'
                    if assigned!=gpu or task in done or task in active or (not baseline_done and config not in CONFIGS[:4]):continue
                    if sum(v['gpu']==gpu for v in active.values())>=limits[gpu]:break
                    output=open(LOGS/(task+'.log'),'a');output.write('\nATTEMPT_START '+utc()+'\n');output.flush()
                    proc=subprocess.Popen([sys.executable,str(EXP/'run.py'),'--worker',config,str(seed)],env=worker_env(gpu,task),cwd=EXP,stdout=output,stderr=subprocess.STDOUT)
                    active[task]=dict(pid=proc.pid,process=proc,gpu=gpu,log=output);print(f'LAUNCH {task} GPU{gpu} pid={proc.pid}',flush=True)
            now=time.monotonic()
            if now-last_refresh>=20 or not active:
                m=refresh();m['failures_and_restarts']=failures
                if now-last_resources>=60:
                    r=resources();peaks=m.setdefault('resource_peaks',dict(worker_rss_gib=0.,gpu0_mib=0,gpu1_mib=0))
                    peaks['worker_rss_gib']=max(peaks['worker_rss_gib'],r['worker_rss_gib'])
                    for card in r['gpu']:
                        g=card['index'];peaks[f'gpu{g}_mib']=max(peaks[f'gpu{g}_mib'],card['used_mib'])
                        newlimit=1 if card['used_mib']>85000 or r['cgroup_available_gib']<12 else (2 if r['cgroup_available_gib']<24 else workers_per_gpu)
                        if limits[g]!=newlimit:m.setdefault('resource_throttling',[]).append(dict(utc=utc(),gpu=g,new_limit=newlimit));limits[g]=newlimit
                    last_resources=now;print('RESOURCE '+json.dumps(r),flush=True)
                save_json(MANIFEST,m);last_refresh=now
                print(f'PROGRESS tasks={len(done)}/60 folds={m["completed_folds"]}/600 active={len(active)} baseline_gate={baseline_done}',flush=True)
            if len(done)==60 and not active:break
            time.sleep(2)
        m=refresh();m.update(run_status='TRAINING_COMPLETE',training_completed_utc=utc(),training_wall_seconds=time.time()-m['started_unix']);save_json(MANIFEST,m)
        validate_complete();report();assert not live_workers();integrity(full=True)
        shutil.rmtree(STATE)
        m=read_json(MANIFEST);m.update(run_status='COMPLETE',completed_utc=utc(),temporary_state_cleaned=True,no_training_processes=True,
            root_handoff_sha256=sha(ROOT/'handoff.md'),results_sha256={n:sha(RESULTS/(n+'.csv')) for n in ['fold_metrics','diagnostics','counterfactual']})
        save_json(MANIFEST,m);report();append_handoff()
        m=read_json(MANIFEST);m['report_sha256']=sha(RESULTS/'report.md');save_json(MANIFEST,m)
        for p in LOGS.glob('.*.lock'):p.unlink(missing_ok=True)
        print('COMPLETE 60/60 tasks; 600/600 folds; 600 diagnostics; 1080 counterfactual rows',flush=True)
    except Exception as exc:
        for info in active.values():
            if info['process'] and info['process'].poll() is None:info['process'].terminate()
        for info in active.values():
            if info['process']:
                try:info['process'].wait(timeout=25)
                except subprocess.TimeoutExpired:info['process'].kill();info['process'].wait()
        m=refresh();m.update(run_status='BLOCKED',blocking_reason=str(exc),failures_and_restarts=failures);save_json(MANIFEST,m)
        report();append_handoff();raise
    finally:lock.close()

def validate_complete():
    m=integrity(full=True);rows=read_csv(RESULTS/'fold_metrics.csv');ds=read_csv(RESULTS/'diagnostics.csv');cf=read_csv(RESULTS/'counterfactual.csv')
    expected={(c,s,f) for c in CONFIGS for s in range(3) for f in range(1,11)}
    keys=lambda rr:{(r['config'],int(r['seed']),int(r['fold'])) for r in rr}
    assert len(rows)==len(ds)==600 and keys(rows)==keys(ds)==expected
    assert all(math.isfinite(float(r[k])) for r in rows for k in METRICS)
    assert all(sum(int(r['physical_gpu'])==g for r in rows)==300 for g in (0,1))
    assert len(cf)==1080 and len({(r['config'],int(r['seed']),int(r['fold']),r['condition']) for r in cf})==1080
    assert sum(r['applicable']=='False' for r in cf)==60
    assert m['completed_tasks']==60 and m['historical_parity_120']['status']=='PASS'
    for r in rows:
        if r['config'] in CONFIGS[:4]:
            typed={k:(float(v) if k in METRICS else int(v) if k in ('seed','fold','best_epoch','epochs_executed') else v) for k,v in r.items()}
            assert_historical_fold(typed)
    assert all(float(r.get('residual_relative_max') or 0)<=.100001 and int(r.get('adjacency_outside_line_nonzero') or 0)==0 for r in ds)
    return m


def write_readme(m):
    lines=['# SABER Experiment20：固定疾病概念锚定与患者级证据图联合验证','',
        '冻结20组×3seed×10fold=600fold。基于原始P0及实验九P3+D/E9，新增模块只在Encoder→FeedForward之后、SPGS之前作用于CORR的36维潜在特征。根源码与实验一至十九只读。','',
        '文本完全来自实验九与十九；HO连接映射来自实验十一经过数值、标签、列序核验的两表。252个连接是线图节点，共享HO ROI定义固定连边；患者FC只调节边权，不改变物理脑连接。','',
        '`d=clip((FC-train_mean)/train_std,-6,6)`；参考统计仅使用当前训练折全体，不使用标签。四域固定prior直接进入候选softmax，禁止自由Key或额外证据诊断头。`3→32 + edgeID`编码，136→64→36读出；`0.1*tanh(alpha)*||base_CORR||*normalize(readout)`残差；alpha=0，绝对相对范数≤10%。','',
        'Local新增19429参数；Graph新增20517参数；其中没有进入候选读出或一跳邻居的ID参数可能无梯度，诊断报告实际可达连接。E9另加36865，E3判别器另加1388。新参数Adam lr=.0038、weight_decay=0；原两Adam和原始损失保留。','',
        f'按用户最新授权，正式实验目录为 `{EXP}`。`/data` 仅作为已审计源码和数据的只读来源；完整更新后的handoff保存在 `{HANDOFF}`，保留根handoff原有全部内容。','',
        '|配置|Prompt|D|GRL|先验|患者线图|','|---|---|---|---|---|---|']
    for c,d in DESIGNS.items():lines.append(f"|{c}|{d['prompt']}|{d['disease']}|{d['adv']}|{d['condition'] or '无'}|{d['use_graph']}|")
    lines+=['','全部超参数、任务分卡、核心比较、输入文件与数值定义冻结在design.json；版本及哈希在manifest.json。正式训练前原实验九四路径初值/输出/三步Adam逐值一致，并以正式T00–T03的120折验证历史指标及可获得的epoch，过关后才释放新配置。T00还比对实验19 C0、实验13 F0。','',
        '每个seed进程内连续十折，只在任务开始播种；不同组早停造成后续折随机流分叉，不把相同fold误称全程同初始化。checkpoint每25epoch及折边界原子保存模型、原两Adam、E3判别器Adam、所有RNG与图邻接，任务结束删权重。','',
        '反事实只对T05/T11/T15/T18同一最佳checkpoint执行九条件；图条件每次重建邻接并逐值检查REAL恢复。Local的GRAPH_NUMERIC保留NA行，不把它算一次有效评估。','',
        '**历史协议局限**：val=test且用test ACC选checkpoint；全队列标准化与监督RFE先于外层CV；GFR在871人上transductive传播。结果有乐观估计风险，不能称独立无泄漏泛化。三个seed复用相同十折，不进行虚假的30独立样本检验。','',
        'MMGPL概念加权与语义图、MGDR多模态图分解、BrainPrompt多层语义注入均已有先例；本轮只能检验固定医学对应关系是否超越数值/中性/匿名/同Gram随机/错解剖控制，不能声称CLIP概念或动态图首次提出。MMGPL使用1029人MRI+FC、BiomedCLIP、5折×3seed，其72.39%不与本协议91.69%横比。','',
        '```bash',f'cd {ROOT}',f'python {EXP}/run.py prepare',f'python {EXP}/run.py preflight',f'python {EXP}/run.py all',f'python {EXP}/run.py resume',f'python {EXP}/run.py refresh',f'python {EXP}/run.py report','```','',
        '首次prepare从经核验的根核心源码在临时目录应用实验九已验收diff，生成本实验唯一integration_patch.diff；任务各自重新复制根源码并应用本diff。语义缓存一次编码，训练新分支不加载CLIP；历史P3/疾病头仍按原实验九首次惰性编码顺序执行，保留其随机流。','',
        f"审计参照提交：{AUDIT_COMMIT}；服务器HEAD：{m['git_head_at_prepare']}。不执行git commit/push。"]
    atomic(EXP/'README.md','\n'.join(lines)+'\n')


def stable_support(rows,left,right):
    return all(all(v>0 for v in paired(rows,left,right,k)['seeds']) for k in ('acc','ba','probability_auc'))


def report():
    import numpy as np
    m=read_json(MANIFEST);rows=read_csv(RESULTS/'fold_metrics.csv');ds=read_csv(RESULTS/'diagnostics.csv');cf=read_csv(RESULTS/'counterfactual.csv')
    lines=['# 实验二十：固定疾病概念锚定与患者级证据图联合验证','',
        f"状态：**{m['run_status']}**；生成：{utc()}。{m['completed_tasks']}/60任务，{len(rows)}/600fold，{len(ds)}/600诊断，{len(cf)}/1080反事实记录。",'',
        '本轮预先冻结局部证据、患者连接线图、与P3/E9协同三方向及20组阴性控制。医学成功依赖正确先验优于中性、匿名、同Gram随机及错误解剖，而不是只看最高ACC。','',
        '源码审计：原GFR的i==1覆盖、ALoss不反传、两个原Adam、全部SPGS/global/RES及输出分类器保留；接入点是Encoder→FeedForward后的[871,4,36]，潜在token不解释为脑区。E9和E3依照实验九补丁原数学与加载顺序。','',
        f"服务器HEAD `{m['git_head_at_prepare']}`；GitHub参照 `{AUDIT_COMMIT}`；六核心文件逐字节一致。历史保护 {len(m['protected_history_sha256'])} 文件；CLIP SHA256 `{m['clip_weight_sha256']}`；缓存 SHA256 `{m['frozen_file_sha256']['semantic_cache.npz']}`。其余代码/数据/文本哈希见manifest。",'',
        '历史动机：实验10的错疾病/中性近似，11恢复真实ROI，12–14未证明语义独立增益，15局部ROI名称有线索但A5文本变化未传至预测，16加强耦合反降，17/18表型对齐无可用补充，19中性C5优于真实C4且自由Key可重参数化。本轮不再训练Key、不增加NULL/BCE头、不改词或调参。','',
        'MMGPL作者版本Fig1、§3.3 Eq7–10和§3.4 Eq11–14已有概念加权及语义关系图；Fig2中B+G仍计算语义相似度，其消融没有替代本轮完整的错误语义验证。ABIDE1029人MRI+FC、BiomedCLIP、5折×3seed、ACC72.39%；输入和协议不同，不与本项目横比。[原文](https://arxiv.org/pdf/2312.14574)。',
        'SPromptGL已提出GFR/Prompt融合；MGDR已研究多模态图分解；BrainPrompt已有ROI/受试者/疾病多层Prompt。只可将严格固定解剖—概念对应的增量验证作为待检验假设，不把语义图或CLIP概念本身当作首次创新。','',
        f"静态映射：252条可信连接，排除[195,225,232,236]，候选数{m['semantic_audit']['candidate_counts']}；查询Gram最大误差{m['semantic_audit']['gram_error']:.3g}；先验状态{m['semantic_audit']['status']}。同Gram只约束query，不能声称最终prior分布相同。",'',
        '训练统计仅由outer train全体FC估计，d是相对训练集偏差而不是病理分数。线图节点是252条FC连接；共享已核验HO ROI决定固定支持集，患者数值和先验只调节权重，不是111 ROI的新物理拓扑。四条无语义身份连接继续存在原主干中。','',
        '主指标按每seed十折均值，再三seed mean±population SD。SEN以ASD(0)为阳性，SPE为TD(1)；Hard-AUC来自离散预测，Prob-AUC用exp(logprob)的ASD连续概率。','',
        '|配置|具体机制|ACC|BA|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|','|---|---|---:|---:|---:|---:|---:|---:|']
    complete=[c for c in CONFIGS if sum(r['config']==c for r in rows)==30]
    stats=summary([r for r in rows if r['config'] in complete]) if complete else {}
    for c,d in DESIGNS.items():
        mechanism=d['prompt']+('+D' if d['disease'] else '')+('+GRL' if d['adv'] else '')+(' + '+d['condition']+(' graph' if d['use_graph'] else ' local') if d['condition'] else '')
        values=[f"{100*stats[c][k]['mean']:.3f}±{100*stats[c][k]['sd']:.3f}" for k in METRICS] if c in stats else ['未完成']*6
        lines.append('|'+c+'|'+mechanism+'|'+'|'.join(values)+'|')
    if m.get('blocking_reason'):lines+=['',f"阻塞原因：{m['blocking_reason']}"]
    import numpy as np
    with np.load(EXP/'semantic_cache.npz',allow_pickle=False) as packed:
        mask=packed['masks'];valid=packed['valid_idx'];endpoints=packed['roi_endpoints']
        atlas_rows=read_csv(ROOT/'experiment/experiment11_roi_mapping_recovery/results/roi_111_labels.csv')
        atlas={int(r['roi_index_zero_based']):r['roi_name'] for r in atlas_rows}
        top={}
        for bank in ('real','neutral','random','anonymous'):
            v=packed['prior_'+bank];top[bank]=[np.where(mask[k])[0][np.argsort(v[k,mask[k]])[-5:][::-1]].tolist() for k in range(4)]
        for k in range(4):
            names=[f"{int(valid[e])}: {atlas[int(endpoints[e,0])]} — {atlas[int(endpoints[e,1])]}" for e in top['real'][k]]
            lines.append(f"静态域{k+1} REAL Top5: "+'; '.join(names))
            lines.append('中性/同Gram随机/匿名Top5索引与REAL的交集：'+'; '.join(bank+'='+str([int(valid[e]) for e in top[bank][k]])+f" (overlap {len(set(top[bank][k])&set(top['real'][k]))}/5)" for bank in ('neutral','random','anonymous')))
    lines.append('上述静态连接排序没有读取任何患者标签，也不是已验证ASD病灶。')

    lines+=['','原始协议保留test=val、test ACC选模、CV前全局监督RFE/标准化及全871人transductive传播，存在乐观估计风险，不是独立无泄漏泛化。三个seed共享fold，不把30折当独立数据集，不报告不当p值。相同seed/fold只表示相同索引；连续折随机流可能因早停分叉。','',
        '每seed均值（ACC/BA/Hard-AUC/Prob-AUC/SEN/SPE，%）：']
    for c in complete:
        for s in range(3):lines.append(f"- {c} seed{s}: "+' / '.join(f"{100*stats[c][k]['seeds'][s]:.5f}" for k in METRICS))
    lines+=['','预注册全部配对差：单位pp；[]内seed0/1/2；括号内逐折胜/平/负。']
    for left,right in PAIRS:
        if left not in stats or right not in stats:continue
        parts=[]
        for k in METRICS:
            p=paired(rows,left,right,k);parts.append(f"{k} {100*p['mean']:+.5f} [{'/'.join(f'{100*x:+.5f}' for x in p['seeds'])}] ({p['wins']}/{p['ties']}/{p['losses']})")
        lines.append(f"- {left}−{right}: "+'; '.join(parts))
    if len(complete)==20:
        winners={k:max(stats,key=lambda c:stats[c][k]['mean']) for k in ('acc','ba','probability_auc')}
        local=all(stable_support(rows,'T05',r) for r in ('T04','T06','T07','T08','T09'))
        graph=all(stable_support(rows,'T11',r) for r in ('T10','T12','T13'))
        synergy=all(stable_support(rows,'T18',r) for r in ('T02','T17','T19','T15')) and all(stable_support(rows,'T15',r) for r in ('T02','T14','T16'))
        lines+=['',f'最佳均值配置：{json.dumps(winners,ensure_ascii=False)}。最高配置是描述性结果，20组均保留。',
            f'疾病概念局部语义严格门槛：{"SUPPORTED" if local else "NOT_SUPPORTED"}；概念图严格门槛：{"SUPPORTED" if graph else "NOT_SUPPORTED"}；E9互补严格门槛：{"SUPPORTED" if synergy else "NOT_SUPPORTED"}。门槛为ACC/BA/Prob-AUC每个seed均正的描述性标准，不能当统计显著性。',
            '图与Local参数数不同，T11−T05仅说明组合变化；医学图归因以等图容量T11−T10/12/13和T18−T17/19为主。若真实图不超过数值图，数值建图足以解释本轮观察，不能宣称医学语义建图创新。']
        historical=read_csv(P19/'results/fold_metrics.csv');old=summary([r for r in historical if r['config']=='C5'])['C5']
        lines.append('T18与历史E9（本轮T02逐折复刻）及实验19 C5的描述性比较；C5不称相同初始化配对：')
        for k in METRICS:lines.append(f"- {k}: T18−E9={100*(stats['T18'][k]['mean']-stats['T02'][k]['mean']):+.5f}pp；T18−旧C5={100*(stats['T18'][k]['mean']-old[k]['mean']):+.5f}pp。")
        lines+=['',f"MICCAI2027第一创新点判断：{'达到本轮内部机制支持门槛，仍需无泄漏外部验证及创新范围核实' if local and graph and synergy else '当前不足；工程成功、非零梯度、注意力合理或某组ACC上涨均不能替代语义阴性控制和E9增量证据'}。",'',
                '后续唯一建议：'+('冻结当前方法，在嵌套选模、训练折内RFE/标准化与独立站点上验证，不据本测试折改词。' if local and graph and synergy else '保留成熟P0/E9作为参照；停止本套静态文本的自动追分，在严格分离选模和评估的协议中仅验证已经观察到的数值建图假设。')]
    lines+=['','逐组最佳checkpoint机制（四域顺序沿实验19；attention是候选连接权重，不能当症状真值）：']
    for c in CONFIGS:
        sub=[r for r in ds if r['config']==c and r.get('alpha','')!='']
        if not sub:continue
        fields=['alpha','residual_relative_mean','residual_relative_max','residual_nonzero_fraction','representation_cosine','adjacency_patient_variance',
                'message_relative_change','edge_id_feature_change','edge_id_active_connections','branch_off_acc','branch_off_probability_auc','branch_off_probability_change',
                'graph_off_acc','graph_off_probability_auc','graph_off_probability_change','disease_probability_correlation']
        parts=[f"{k}={np.mean([float(r[k]) for r in sub if r.get(k,'')!='']):.7g}" for k in fields if any(r.get(k,'')!='' for r in sub)]
        for k in ('domain_attention_entropy','domain_coverage','domain_signed_fc','prior_candidate_std'):
            parts.append(k+'='+json.dumps(np.mean([json.loads(r[k]) for r in sub],axis=0).tolist()))
        grads=[json.loads(r['gradient_norms']) for r in sub]
        parts.append('maximum_sampled_gradient_norm_means='+json.dumps({k:float(np.mean([g.get(k,0.) for g in grads])) for k in grads[0]}))
        parts.append('train_reference_ranges='+json.dumps({k:[min(float(r[k]) for r in sub),max(float(r[k]) for r in sub)] for k in ('train_fc_mean_min','train_fc_mean_max','train_fc_std_min','train_fc_std_max')}))
        lines.append('- '+c+': '+', '.join(parts))
    lines+=['','反事实汇总：REAL与每次恢复逐值相同；固定同一checkpoint，全部参数和原始P3/E9/GFR输入保持不变。FC_SHUFFLED仅改新分支test患者FC，BRANCH_OFF是分布外消融，不等于重训对照。']
    for c in ('T05','T11','T15','T18'):
        for condition in CONDITIONS:
            sub=[r for r in cf if r['config']==c and r['condition']==condition]
            if not sub:continue
            valid=[r for r in sub if str(r['applicable'])=='True']
            if not valid:lines.append(f'- {c} {condition}: NA ×{len(sub)}，未评估。');continue
            fields=['acc','ba','hard_auc','probability_auc','sensitivity_asd','specificity_td','acc_change','prediction_flip','probability_abs_change']
            lines.append(f'- {c} {condition} ({len(valid)} folds): '+', '.join(f'{k}={np.mean([float(r[k]) for r in valid]):.8f}' for k in fields))
    lines+=['','数值/医学解释边界：FC是LedoitWolf相关+Fisher-z经RFE保留的稀疏连接；当前输入已全局标准化，不是完整111×111全脑图。四域候选只是解剖与文献先验，未用患者症状量表验证。只能写解剖与文献先验+训练标签驱动的弱监督对齐，不能声称病灶、因果脑区或患者症状监督。',
        f"运行壁钟：{m.get('training_wall_seconds','未完成')}秒；GPU0/GPU1预分配各30任务、300fold，最多每卡4worker，每worker2CPU线程；实际资源见manifest。无commit/push。状态未完成时不生成虚假分数。"]
    atomic(RESULTS/'report.md','\n'.join(lines)+'\n')
    return lines


def append_handoff():
    m=read_json(MANIFEST);path=HANDOFF;lock=acquire_lock(LOGS/'.handoff.lock')
    try:
        assert sha(ROOT/'handoff.md')==m['handoff_prefix_sha256'],'root handoff changed'
        original=path.read_bytes() if path.exists() else (ROOT/'handoff.md').read_bytes()
        prefix=original[:m['handoff_prefix_bytes']]
        assert hashlib.sha256(prefix).hexdigest()==m['handoff_prefix_sha256'],'handoff history changed'
        marker='## 实验二十：固定疾病概念锚定与患者级证据图联合验证'
        if m.get('handoff_recorded_status')==m['run_status']:return
        text='\n\n'+marker+'\n\n'+f"时间：{utc()}；状态：{m['run_status']}。实验目录：`{EXP}`。\n\n"
        text+='实验10–19表明类别码容量、近零或共有语义响应、自由Key重参数化都是替代解释。本轮冻结REAL/NEUTRAL、真实252连接、4域候选[72,56,45,32]，从Encoder→FeedForward后以零初始化≤10% CORR残差加入局部证据及共享ROI患者线图，并原样保留P0/P3/E9/E3。\n\n'
        text+='20配置：'+json.dumps(DESIGNS,ensure_ascii=False)+'\n\n'
        text+=f"工程：{m['completed_tasks']}/60任务，{m['completed_folds']}/600fold，{m['diagnostic_rows']}/600诊断，{m['counterfactual_rows']}/1080反事实；preflight={m.get('preflight',{} ) and m['preflight']['status']}。Git HEAD {m['git_head_at_prepare']}，参照 {AUDIT_COMMIT}；代码/数据/文本/权重/先验哈希见manifest。\n\n"
        if m.get('blocking_reason'):text+='阻塞：'+m['blocking_reason']+'；未完成部分不作科学成败判定。\n\n'
        rows=read_csv(RESULTS/'fold_metrics.csv')
        if len(rows)==600:
            stats=summary(rows)
            text+='主结果，ACC/BA/Hard-AUC/Prob-AUC/SEN-ASD/SPE-TD（%，三seed mean±population SD）：\n\n'
            for c in CONFIGS:text+='- '+c+': '+' / '.join(f"{100*stats[c][k]['mean']:.5f}±{100*stats[c][k]['sd']:.5f}" for k in METRICS)+'\n'
            for l,r in [('T05','T06'),('T05','T08'),('T05','T07'),('T11','T10'),('T11','T12'),('T18','T19'),('T18','T17'),('T18','T02')]:
                text+=f"- {l}−{r}: "+' / '.join(f"{k} {100*paired(rows,l,r,k)['mean']:+.5f}pp" for k in ('acc','ba','probability_auc'))+'\n'
            text+='\n严格三seed三主要指标门槛（不等于显著性）：局部='+str(all(stable_support(rows,'T05',r) for r in ('T04','T06','T07','T08','T09')))+'；图='+str(all(stable_support(rows,'T11',r) for r in ('T10','T12','T13')))+'；组合='+str(all(stable_support(rows,'T18',r) for r in ('T02','T17','T19','T15')))+'。\n'
        text+=f"\n运行壁钟：{m.get('training_wall_seconds','未启动正式CV')}；GPU各30任务/300fold，最多4worker×2CPU线程。反事实为同checkpoint九条件；NA不算有效评估。全部结果和唯一后续建议见 `{RESULTS/'report.md'}`。\n"
        text+='原test=val选模、全局监督RFE/标准化与transductive协议存在乐观风险，不称无泄漏泛化或医学因果。固定语义未优于阴性控制时停止追分。没有修改原源码或实验1–19，未commit/push。\n'
        atomic(path,original+text.encode())
        m=read_json(MANIFEST);m.update(handoff_recorded_status=m['run_status'],handoff_sha256=sha(path),
            handoff_file=str(path),handoff_update=dict(status='WRITTEN_COPY',root_unchanged=True,path=str(path)));save_json(MANIFEST,m)
    finally:lock.close()


def main():
    if '--worker' in sys.argv:
        i=sys.argv.index('--worker');worker(sys.argv[i+1],int(sys.argv[i+2]));return
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','preflight','all','resume','report','refresh']);parser.add_argument('--workers-per-gpu',type=int,default=4)
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='preflight':preflight()
    elif args.command in ('all','resume'):schedule(args.workers_per_gpu)
    elif args.command=='refresh':refresh()
    elif args.command=='report':report()

if __name__=='__main__':main()
