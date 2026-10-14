#!/usr/bin/env python3
"""DS3A-v2: frozen 12-group experiment with continuous-fold recovery."""
import argparse, contextlib, csv, difflib, fcntl, hashlib, io, json, math, os
import pickle, random, shutil, subprocess, sys, time, types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from ds3a import (DS3A, DESIGNS, CONFIGURATION_TABLE, LAMBDA_ALIGN,
                 LAMBDA_PRIVATE, PRIVATE_COS_MARGIN, RESIDUAL_LIMIT)
EXP = Path(__file__).resolve().parent
ROOT = EXP.parent.parent
STATE = Path('/tmp') / ('spromptgl17-state-' + hashlib.sha256(str(EXP).encode()).hexdigest()[:12])
SOURCES = STATE/'sources'
MANIFEST, RESULTS, LOGS = EXP/'manifest.json', EXP/'results', EXP/'logs'
CORE = ['main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh']
CONFIGS = list(DESIGNS)
TASKS = [(c,s,(i*3+s)%2) for i,c in enumerate(CONFIGS) for s in range(3)]
CF_CONDITIONS = ['REAL','NEUTRAL','UNRELATED','SWAPPED','SKIP','NO_QC','NO_PRIVATE','RESIDUAL_OFF']
METRICS = ['acc','ba','hard_auc','probability_auc','sensitivity_asd','specificity_td']
PAIRS = [('D2','D1'),('D3','D2'),('D3','D1'),('D11','D3'),('D4','D3'),('D5','D4'),
         ('D5','D9'),('D5','D6'),('D5','D7'),('D5','D8'),('D5','D10'),('D5','D0'),('D5','D1')]
PROTOCOL = dict(datadir=str(ROOT/'data')+'/',datname='ABIDE',cpu=False,gpu='-1',
    nepoch=450,early=200,lr=.0038,reg=.11,dropout=.35,nlayer=1,n_hidden=18,n_head=2,
    n_iter=10,nmodal=4,th=.9,GC_mode='weighted-cosine',MP_mode='GCN',MF_mode='concat',
    alpha=.5,theta_smooth=1.,theta_degree=.5,theta_sparsity=0.,nclass=2,mode='simple-2')
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
    atomic(path, b'SP17' + hashlib.sha256(data).hexdigest().encode() + data)

def checkpoint(path):
    import torch
    b = Path(path).read_bytes()
    assert b[:4] == b'SP17' and hashlib.sha256(b[68:]).hexdigest().encode() == b[4:68]
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

def model_state(a):
    return {name:{k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
            for name,module in [('gfr',a.MessagePassing0),('pln',a.ModalFusion)]}

def load_state(a,state):
    a.MessagePassing0.load_state_dict(state['gfr'])
    a.ModalFusion.load_state_dict(state['pln'])

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

def lock(path):
    f = open(path,'a+')
    fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    return f

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


def data_audit():
    import numpy as np
    import pandas as pd
    mapping = np.load(ROOT/'data/ABIDE/modal_feat_dict.npy', allow_pickle=True).item()
    columns = pd.read_csv(ROOT/'data/ABIDE/processed_standard_data.csv', nrows=0).columns.tolist()
    assert list(mapping) == ['PHENO','ANAT','FUNC','Correlation']
    assert columns[:-1] == [str(c) for values in mapping.values() for c in values]
    pheno = [str(c) for c in mapping['PHENO']]
    indices = [i for i,c in enumerate(pheno) if c.startswith(('AGE_AT_SCAN_', 'SEX_'))]
    site = [i for i,c in enumerate(pheno) if c.startswith('SITE_ID_')]
    age = [i for i,c in enumerate(pheno) if c.startswith('AGE_AT_SCAN_')]
    sex = [i for i,c in enumerate(pheno) if c.startswith('SEX_')]
    assert (len(age),len(site),len(sex)) == (26,20,2)
    assert sorted(indices+site) == list(range(48))
    x,y,splits = input_data()
    return dict(shape=list(x.shape), modality_columns={k:list(map(str,v)) for k,v in mapping.items()},
                pheno_age_sex_indices=indices,site_indices=site,age_indices=age,sex_indices=sex,
                labels={'ASD_0':int((y==0).sum()),'TD_1':int((y==1).sum())})


def replace_once(source, old, new):
    assert source.count(old)==1, ('patch anchor count',old,source.count(old))
    return source.replace(old,new,1)


def patched_sources():
    sources = {p:(ROOT/p).read_text() for p in ['network.py','model.py']}
    n = sources['network.py']
    start, end = n.index('def get_prompt_text_features():'), n.index('class MLP')
    n = n[:start] + '''def get_prompt_text_features():
    from ds3a_runtime import get_frozen_p0
    return get_frozen_p0()

''' + n[end:]
    n = replace_once(n,'    def forward(self, x):\n        bs = x.size(0)',
                       '    def forward(self, x, raw_modalities=None):\n        bs = x.size(0)')
    n = replace_once(n,'        text_features = self.prompt_text_features',
        '        if hasattr(self, "ds3a"):\n            x = self.ds3a(x, raw_modalities)\n        text_features = self.prompt_text_features')
    sources['network.py'] = n
    s = sources['model.py']
    s = replace_once(s, '        self.ModalFusion.apply(my_weight_init)',
        '''        self.ModalFusion.apply(my_weight_init)
        from ds3a_runtime import install_branch
        install_branch(self)
'''.rstrip())
    s = replace_once(s, '            prob,_,_ = self.ModalFusion(feat)',
        '            prob,_,_ = self.ModalFusion(feat, self._ds3a_raw_modalities)')
    s = replace_once(s, '            loss = cls_loss + all_graph_loss',
        '''            loss = cls_loss + all_graph_loss
            self._ds3a_original_losses = (cls_loss.detach(), all_graph_loss.detach())
            if hasattr(self.ModalFusion, "ds3a"):
                loss = loss + self.ModalFusion.ds3a.loss(self.targ, self.trn_idx)
'''.rstrip())
    sources['model.py'] = s
    return sources


def make_sources():
    SOURCES.mkdir(parents=True,exist_ok=True)
    for name in CORE:
        shutil.copy2(ROOT/name,SOURCES/name)
    for name,content in patched_sources().items():
        atomic(SOURCES/name,content)
    for name in ['CLIP','data']:
        link = SOURCES/name
        if not link.exists(): link.symlink_to(ROOT/name,target_is_directory=True)
    for name in ['attn','graph']:
        (SOURCES/name).mkdir(exist_ok=True)
    sys.path.insert(0,str(SOURCES)) if str(SOURCES) not in sys.path else None
    os.chdir(SOURCES)


def encode_static():
    import torch
    from torch.nn import functional as F
    sys.path.insert(0,str(ROOT))
    from CLIP.clip import clip
    prompts = read_json(EXP/'prompts.json')
    encoder,_ = clip.load('RN50','cuda')
    encoder.eval()
    with torch.no_grad():
        p0 = encoder.encode_text(clip.tokenize(prompts['p0']).cuda()).float().detach().cpu()
        banks,details = {},{}
        for key,sentences in prompts['banks'].items():
            flat = [s for group in sentences for s in group]
            tokens = clip.tokenize(flat)
            features = encoder.encode_text(tokens.cuda()).float()
            proto = F.normalize(F.normalize(features,dim=-1).view(2,4,1024).mean(1),dim=-1).cpu()
            banks[key] = proto
            details[key] = dict(sentence_sha256=canonical(flat),word_counts=[len(s.split()) for s in flat],
                token_lengths=tokens.ne(0).sum(1).tolist(),
                raw_embedding_sha256=hashlib.sha256(features.cpu().numpy().tobytes()).hexdigest(),
                prototype_sha256=hashlib.sha256(proto.numpy().tobytes()).hexdigest(),
                prototype_cosine=float((proto[0]*proto[1]).sum()))
    del encoder
    torch.cuda.empty_cache()
    save_checkpoint(STATE/'static.pt',dict(p0=p0,banks=banks,pheno_indices=data_audit()['pheno_age_sex_indices']))
    return details,hashlib.sha256(p0.numpy().tobytes()).hexdigest()


def history_refs():
    return [(13,'F0'),(14,'E00'),(15,'A0'),(16,'B0')]


def prepare():
    if MANIFEST.exists():
        m = integrity()
        if m['run_status']=='COMPLETE': return m
        if not (STATE/'static.pt').exists():
            details,p0hash = encode_static()
            assert details == m['text_encoding'] and p0hash == m['p0_embedding_sha256']
        make_sources()
        return m
    STATE.mkdir(parents=True,exist_ok=True)
    RESULTS.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True)
    data = data_audit()
    details,p0hash = encode_static()
    diff = ''.join(''.join(difflib.unified_diff((ROOT/n).read_text().splitlines(True),v.splitlines(True),
                      fromfile='a/'+n,tofile='b/'+n)) for n,v in patched_sources().items())
    atomic(EXP/'integration_patch.diff',diff)
    protected = {str(p.relative_to(ROOT)):sha(p) for n in range(1,17)
                 for directory in (ROOT/'experiment').glob(f'experiment{n}_*')
                 for p in directory.rglob('*') if p.is_file()}
    audited = {str(ROOT/p):sha(ROOT/p) for p in CORE+[
        'data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy',
        'data/ABIDE/data_processing.ipynb','CLIP/clip/clip.py','CLIP/clip/model.py','CLIP/clip/simple_tokenizer.py']}
    x,y,splits = input_data()
    audit = Path('/tmp/spromptgl17-audit')
    github = read_json(audit/'github.json')
    papers = read_json(audit/'papers_parsed.json')
    papers['CR-mCCAR'] = dict(url='https://pmc.ncbi.nlm.nih.gov/articles/PMC12869378/',
                            parsed_text_sha256=sha(audit/'CR-mCCAR.txt'),method_section_read=True)
    mmgl_history = {str(p):sha(p) for d in (Path('/data/adodas/MMGL-main')/'experiment').iterdir()
                    if d.is_dir() and (d.name=='alignment_validation' or d.name.startswith(('exp2_','exp3_','exp4_')))
                    for p in d.glob('**/report.md')}
    counts = {c:0 if c=='D0' else sum(p.numel() for p in DS3A(c,data['pheno_age_sex_indices'],
                 checkpoint(STATE/'static.pt')['banks'],0).parameters()) for c in CONFIGS}
    m = dict(schema_version=1,experiment='DS3A-v2',run_status='PREPARED',prepared_utc=utc(),
        root=str(ROOT),experiment_directory=str(EXP),protocol=PROTOCOL,
        configurations=CONFIGURATION_TABLE,additional_parameters=counts,
        fixed_design=dict(shared_dim=32,private_dim=32,lambda_align=LAMBDA_ALIGN,
            lambda_private=LAMBDA_PRIVATE,private_cos_margin=PRIVATE_COS_MARGIN,
            residual_relative_l2_limit=RESIDUAL_LIMIT,new_weight_decay=0.,gate_weight_stop_gradient=True,
            prototype_scope='train fold only; same class-conditional loss across all alignment groups',
            no_unconditional_training_control='fixed12-group matrix: unclassified distances are diagnostic only'),
        data_audit=data,source_data_sha256=audited,protected_history_sha256=protected,
        github_audit=github,literature_audit=papers,mmgl_audit_report_sha256=mmgl_history,
        instruction_prompt_sha256=sha(ROOT/'prompt'),
        clip_weight_sha256=sha(Path.home()/'.cache/clip/RN50.pt'),
        prompt_sha256=sha(EXP/'prompts.json'),text_encoding=details,p0_embedding_sha256=p0hash,
        patch_sha256=sha(EXP/'integration_patch.diff'),
        code_sha256={n:sha(EXP/n) for n in ['run.py','ds3a.py']},
        split_sha256=canonical([dict(train=tr.tolist(),test=te.tolist()) for tr,te in splits]),
        rng_policy='seed once per config; continuous ten folds; branch init fork_rng; exact full RNG checkpoint',
        git_head_at_prepare=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        handoff_prefix_sha256=sha(ROOT/'handoff.md'),handoff_prefix_bytes=(ROOT/'handoff.md').stat().st_size,
        preflight=None,completed_folds=0,tasks={f'{c}_seed{s}':dict(config=c,seed=s,physical_gpu=g,status='PENDING',folds=0) for c,s,g in TASKS})
    save_json(MANIFEST,m)
    make_sources()
    write_readme(m)
    return m


def integrity(full=False):
    m = read_json(MANIFEST)
    for path,h in m['source_data_sha256'].items(): assert sha(path)==h, ('protected source/data changed',path)
    for name,h in m['code_sha256'].items(): assert sha(EXP/name)==h, ('frozen code changed',name)
    assert sha(EXP/'prompts.json')==m['prompt_sha256']
    assert sha(EXP/'integration_patch.diff')==m['patch_sha256']
    if full:
        for name,h in m['protected_history_sha256'].items(): assert sha(ROOT/name)==h, ('history changed',name)
        assert sha(Path.home()/'.cache/clip/RN50.pt')==m['clip_weight_sha256']
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==m['git_head_at_prepare']
    return m


def signature():
    global _SIGNATURE
    if _SIGNATURE is None:
        m = read_json(MANIFEST)
        _SIGNATURE = canonical({k:m[k] for k in ['source_data_sha256','code_sha256','prompt_sha256',
            'patch_sha256','protocol','configurations','fixed_design','split_sha256','rng_policy']})
        if 'execution_code_sha256' in m:
            payload={k:m[k] for k in ['source_data_sha256','code_sha256','prompt_sha256',
                'patch_sha256','protocol','configurations','fixed_design','split_sha256','rng_policy']}
            payload['code_sha256']=m['execution_code_sha256']
            _SIGNATURE=canonical(payload)
    return _SIGNATURE


def context():
    global _CTX
    if _CTX is None:
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        sys.modules['ds3a_runtime'] = sys.modules[__name__]
        make_sources()
        import network, model
        x,y,folds = input_data()
        _CTX = dict(torch=torch,network=network,model=model,x=x,y=y,folds=folds,
                    static=checkpoint(STATE/'static.pt'),clip_loaded=False)
    return _CTX


def get_frozen_p0():
    c = context()
    if not c['clip_loaded']:
        # The original first lazy CLIP.load consumes CPU RNG; retain that event.
        encoder,_ = c['network'].clip.load('RN50','cuda')
        del encoder
        c['clip_loaded'] = True
    return c['static']['p0'].cuda()


def install_branch(a):
    c = context()
    a._ds3a_raw_modalities = a.dataLayer(a.feat)[1]
    config = os.environ['DS3A_CONFIG']
    if config=='D0': return
    # Create AFTER original apply(my_weight_init), preserving all old streams.
    with c['torch'].random.fork_rng(devices=[0]):
        c['torch'].manual_seed(170000+int(a.hyperpm.seed))
        a.ModalFusion.ds3a = DS3A(config,c['static']['pheno_indices'],c['static']['banks'],a.hyperpm.seed).cuda()
    a.optimizer_MF.add_param_group(dict(params=a.ModalFusion.ds3a.parameters(),
                                       lr=a.hyperpm.lr,weight_decay=0.))


def agent_for(config,seed,fold,original=False):
    c = context(); os.environ['DS3A_CONFIG'] = config
    cls = c['model'].EvalHelper
    if original:
        module = types.ModuleType('ds3a_original_network'); sys.modules[module.__name__]=module
        exec(compile((ROOT/'network.py').read_text(),'original_network.py','exec'),module.__dict__)
        original_model = types.ModuleType('ds3a_original_model')
        src = (ROOT/'model.py').read_text().replace('from network import *','from ds3a_original_network import *')
        exec(compile(src,'original_model.py','exec'),original_model.__dict__)
        cls = original_model.EvalHelper
    tr,te = c['folds'][fold-1]
    with contextlib.redirect_stdout(io.StringIO()):
        a = cls([48,6,10,256],c['x'],c['y'],SimpleNamespace(**PROTOCOL,seed=seed),tr,te)
    a.MF_sav.close(); a.GCMP_sav.close()
    assert (a.val_idx==a.tst_idx).all()
    return a


def weight_hash(a,branch=False,exclude_learned=False,legacy13=False):
    h = hashlib.sha256()
    for name,module in [('gfr',a.MessagePassing0),('pln',a.ModalFusion)]:
        for key,value in sorted(module.named_parameters()):
            if key.startswith('ds3a.') != branch: continue
            if exclude_learned and key=='ds3a.learned_prototypes': continue
            tag = key if legacy13 else name+'.'+key
            # Experiment13 used a distinct serialization prefix; verified below.
            h.update(tag.encode()+value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def optimizer_audit(a,config):
    m = read_json(MANIFEST)
    groups = a.optimizer_MP0.param_groups+a.optimizer_MF.param_groups
    actual = [p for module in [a.MessagePassing0,a.ModalFusion] for p in module.parameters()]
    supplied = [p for group in groups for p in group['params']]
    assert len(supplied)==len(set(map(id,supplied))) and set(map(id,supplied))==set(map(id,actual))
    assert all(g['lr']==.0038 for g in groups)
    assert a.optimizer_MP0.param_groups[0]['weight_decay']==a.optimizer_MF.param_groups[0]['weight_decay']==.11
    total = sum(p.numel() for p in actual)
    assert total==1462957+m['additional_parameters'][config]
    if config!='D0':
        assert len(a.optimizer_MF.param_groups)==2
        assert a.optimizer_MF.param_groups[1]['weight_decay']==0.
        assert sum(p.numel() for p in a.optimizer_MF.param_groups[1]['params'])==m['additional_parameters'][config]
    return total


def forward(a):
    import torch
    a.MessagePassing0.eval(); a.ModalFusion.eval()
    def capture(module,inputs,output): a._ds3a_logits = output.detach().clone()
    hook = a.ModalFusion.Outputlayer.classifier.register_forward_hook(capture)
    try:
        with torch.no_grad():
            n,parts = a.dataLayer(a.feat)
            h,_,_ = a.MessagePassing0(parts,n,a.targ,a.trn_idx)
            out,_,_ = a.ModalFusion(h,a._ds3a_raw_modalities) if hasattr(a.ModalFusion,'ds3a') else a.ModalFusion(h)
            assert torch.isfinite(out).all()
            return out.clone()
    finally:
        hook.remove()


def branch(a):
    return getattr(a.ModalFusion,'ds3a',None)


def parameter_norms(a):
    import torch
    b = branch(a)
    if b is None: return {}
    return {k:float(torch.stack([p.detach().square().sum() for p in params]).sum().sqrt())
            for k,params in b.component_parameters().items()}


def capacity_audit(b):
    names={'shared_corr','private_corr','shared_pheno','private_pheno','fusion','alpha'}
    if b.design.selective:names.add('align_skip')
    if b.design.prompt!='OFF':names.update(['query_corr','query_pheno'])
    if b.design.quality:names.update(['quality_encoder','quality_confidence'])
    if b.design.four_modalities:names.update(['shared_anat','shared_func'])
    if b.design.prompt=='LEARNED':names.add('learned_prototypes')
    counts={k:sum(p.numel() for p in pp) for k,pp in b.component_parameters().items()}
    active=sum(counts[k] for k in names)
    # Permanently zero inputs disconnect exact first-layer weight columns.
    if b.design.selective:
        active-=16*(4*(b.design.prompt=='OFF')+16*(not b.design.quality))
    if not b.design.fuse_private:active-=64*64
    return dict(registered_parameters=sum(counts.values()),structurally_active_parameters=active,
        active_components=sorted(names),component_registered_counts=counts,
        description='structural connectivity; observed sampled nonzero gradients may be fewer')


def gradients(a):
    import torch
    b = branch(a)
    if b is None: return {}
    comps = b.component_parameters()
    vv = []
    for pp in comps.values():
        gg = [p.grad for p in pp if p.grad is not None]
        vv.append(torch.stack([g.square().sum() for g in gg]).sum().sqrt() if gg else b.alpha.new_zeros(()))
    values = torch.stack(vv).detach().cpu().tolist()
    assert all(math.isfinite(v) for v in values), ('nonfinite auxiliary gradients',values)
    return dict(zip(comps,values))


def train_step(a):
    a.run_epoch('simple-2')
    return gradients(a)


def forbid_encoding():
    from CLIP.clip.model import CLIP
    def forbidden(*args,**kwargs): raise RuntimeError('CLIP encoding is forbidden during formal training')
    CLIP.encode_text = forbidden


def snapshot(a):
    b = branch(a)
    return {k:v.detach().clone() for k,v in b.last.items() if hasattr(v,'detach')} if b else {}


def cf_changes(a,out,real,now,base,idx,real_logits):
    import torch
    p,q = out.exp()[idx,0],real.exp()[idx,0]
    result = dict(prediction_flip_rate=float(out[idx].argmax(1).ne(real[idx].argmax(1)).float().mean()),
        mean_abs_asd_probability_change=float((p-q).abs().mean()),
        mean_abs_logit_change=float((a._ds3a_logits[idx]-real_logits[idx]).abs().mean()))
    for key in ['align','shared_corr','shared_pheno','private_corr','private_pheno','delta','quality','fusion_input']:
        result[key+'_mean_abs_change'] = float((now[key][idx]-base[key][idx]).abs().mean())
    result['align_mean'] = float(now['align'][idx].mean())
    result['residual_relative_mean'] = float((now['delta'][idx].norm(dim=-1)/(now['base'][idx].norm(dim=-1)+1e-8)).mean())
    return result


def counterfactual(a,seed,fold,real):
    import torch
    b = branch(a); b.condition = 'REAL'
    baseline = snapshot(a); real_logits = a._ds3a_logits.clone()
    model_hash = (weight_hash(a),weight_hash(a,True)); before_rng = rng_hash(rng())
    real_metrics = metrics(real,a.tst_idx)
    rows = []
    try:
        for condition in CF_CONDITIONS:
            b.condition = condition
            out = forward(a); now = snapshot(a)
            assert torch.equal(now['base'],baseline['base'])
            for key in ['shared_corr','shared_pheno','private_corr','private_pheno']:
                assert torch.equal(now[key],baseline[key])
            if condition=='REAL': assert torch.equal(out,real)
            values = metrics(out,a.tst_idx)
            changes = cf_changes(a,out,real,now,baseline,a.tst_idx,real_logits)
            rows.append(dict(config='D5',seed=seed,fold=fold,condition=condition,**values,
                **{'real_'+k:v for k,v in real_metrics.items()},
                **{'delta_'+k:values[k]-real_metrics[k] for k in METRICS},**changes,
                same_checkpoint=True,base_tokens_exact=True,parameters_unchanged=True))
    finally:
        b.condition = 'REAL'; restored = forward(a)
    assert torch.equal(restored,real)
    assert (weight_hash(a),weight_hash(a,True))==model_hash
    assert rng_hash(rng())==before_rng
    return rows


def stat_values(row,prefix,values):
    import numpy as np
    v = np.asarray(values,dtype=float).reshape(-1)
    assert np.isfinite(v).all()
    for key,value in [('mean',v.mean()),('sd',v.std()),('min',v.min()),('p05',np.quantile(v,.05)),('p10',np.quantile(v,.1)),
                       ('p50',np.median(v)),('p90',np.quantile(v,.9)),('p95',np.quantile(v,.95)),('max',v.max())]:
        row[prefix+'_'+key] = float(value)


def representation_diagnostics(row,ll,tr,te,labels):
    import numpy as np
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import balanced_accuracy_score,roc_auc_score
    from sklearn.metrics.pairwise import cosine_similarity
    for suffix,indices in [('train',tr),('test',te)]:
        align = ll['align'][indices,0]
        stat_values(row,'align_'+suffix,align)
        row['skip_lt_0_1_'+suffix] = float((align<.1).mean())
        row['skip_lt_0_5_'+suffix] = float((align<.5).mean())
        row['align_gt_0_9_'+suffix] = float((align>.9).mean())
        row['align_entropy_'+suffix] = float((-(align*np.log(np.maximum(align,1e-12))+
            (1-align)*np.log(np.maximum(1-align,1e-12)))).mean())
        for cls,name in [(0,'asd'),(1,'td')]:
            row[f'align_{name}_{suffix}'] = float(align[labels[indices]==cls].mean())
        z = [ll[k][indices] for k in ['shared_corr','shared_pheno']]
        norms = [np.linalg.norm(v,axis=-1,keepdims=True) for v in z]
        unit = [v/np.maximum(n,1e-8) for v,n in zip(z,norms)]
        row['shared_instance_cosine_'+suffix] = float((unit[0]*unit[1]).sum(-1).mean())
        row['unconditional_prototype_distance_'+suffix] = float(np.square(
            unit[0].mean(0)/max(np.linalg.norm(unit[0].mean(0)),1e-8)-
            unit[1].mean(0)/max(np.linalg.norm(unit[1].mean(0)),1e-8)).sum())
        pair_protos=[]
        for modality,values in [('corr',z[0]),('pheno',z[1])]:
            protos=[]
            for cls in [0,1]:
                mask = labels[indices]==cls
                weights = align[mask]
                proto = (values[mask]*weights[:,None]).sum(0)/max(weights.sum(),1e-8)
                proto = proto/max(np.linalg.norm(proto),1e-8)
                protos.append(proto)
            row[f'{modality}_class_prototype_cosine_{suffix}'] = float(np.dot(*protos))
            row[f'{modality}_class_prototype_collapsed_{suffix}'] = bool(np.dot(*protos)>.99)
            pair_protos.append(np.array(protos))
        row['class_prototype_distance_'+suffix] = float(np.square(pair_protos[0]-pair_protos[1]).sum(-1).mean())
        pc,pp = [ll[k][indices] for k in ['private_corr','private_pheno']]
        upc=pc/(np.linalg.norm(pc,axis=1,keepdims=True)+1e-8)
        upp=pp/(np.linalg.norm(pp,axis=1,keepdims=True)+1e-8)
        row['private_instance_distance_'+suffix] = float(np.square(upc-upp).sum(-1).mean())
    for modality in ['corr','pheno']:
        for scope in ['shared','private']:
            values = ll[f'{scope}_{modality}']
            row[f'{scope}_{modality}_norm_mean'] = float(np.linalg.norm(values,axis=1).mean())
            row[f'{scope}_{modality}_feature_variance'] = float(values[tr].var(0).mean())
            centered = values[tr]-values[tr].mean(0)
            singular = np.linalg.svd(centered,compute_uv=False)
            prob = singular/max(singular.sum(),1e-8)
            row[f'{scope}_{modality}_effective_rank'] = float(np.exp(-(prob*np.log(np.maximum(prob,1e-12))).sum()))
            classifier = RidgeClassifier(alpha=1.).fit(values[tr],labels[tr])
            pred = classifier.predict(values[te])
            score = -classifier.decision_function(values[te])
            row[f'{scope}_{modality}_probe_ba'] = float(balanced_accuracy_score(labels[te],pred))
            row[f'{scope}_{modality}_probe_prob_auc'] = float(roc_auc_score(labels[te]==0,score))
        s,p = ll['shared_'+modality],ll['private_'+modality]
        cosine = (s*p).sum(-1)/(np.linalg.norm(s,axis=-1)*np.linalg.norm(p,axis=-1)+1e-8)
        row[f'shared_private_cosine_{modality}'] = float(cosine.mean())
        row[f'shared_private_abs_cosine_{modality}'] = float(np.abs(cosine).mean())
        row[f'shared_private_gt_0_99_{modality}'] = float((np.abs(cosine)>.99).mean())


def diagnostics(a,config,seed,fold,out,tracker,initial_norms,path_tracker):
    import numpy as np
    import torch
    from sklearn.linear_model import RidgeClassifier
    b = branch(a)
    row = dict(config=config,seed=seed,fold=fold,branch_enabled=b is not None,
        prediction_nonfinite=False,**path_tracker,
        parameter_norms=json.dumps(parameter_norms(a),sort_keys=True),
        sampled_gradient_max=json.dumps(tracker,sort_keys=True))
    if b is None:
        row.update(align_test_mean=0.,align_test_sd=0.,skip_lt_0_5_test=0.,
                   residual_relative_pheno_mean=0.,residual_relative_corr_mean=0.,residual_relative_max=0.)
        return row
    state = snapshot(a)
    ll = {k:v.cpu().numpy() for k,v in state.items()}
    assert all(np.isfinite(v).all() for v in ll.values())
    row['capacity_audit']=json.dumps(capacity_audit(b),sort_keys=True)
    row['sampled_nonzero_component_parameter_upper_count']=sum(sum(p.numel() for p in pp)
        for k,pp in b.component_parameters().items() if tracker.get(k,0.)>0)
    row.update(weighted_auxiliary_loss=float(b.loss(a.targ,a.trn_idx).detach()),
        private_margin_loss=float(b.last['private_loss']),
        training_prototype_alignment_loss=float(b.last['align_loss']),
        zero_prototype_rows=int((np.linalg.norm(ll['text'],axis=1)<1e-8).sum()),
        text_prototype_cosine=float(np.dot(*ll['text'])),alpha_pheno=float(b.alpha[0].detach()),
        alpha_corr=float(b.alpha[1].detach()),
        bound_verified=True,pheno_site_excluded=True)
    norms = parameter_norms(a)
    for k,v in norms.items():
        row['parameter_norm_'+k] = v
        row['parameter_norm_ratio_'+k] = v/max(initial_norms.get(k,0.),1e-8)
        row['gradient_'+k+'_sampled_max'] = tracker.get(k,0.)
    representation_diagnostics(row,ll,a.trn_idx,a.tst_idx,context()['y'])
    relative = np.linalg.norm(ll['delta'],axis=-1)/(np.linalg.norm(ll['base'],axis=-1)+1e-8)
    assert relative.max()<=RESIDUAL_LIMIT+1e-6
    for m,name in enumerate(['pheno','anat','func','corr']):
        stat_values(row,'residual_relative_'+name,relative[:,m])
        row['base_norm_'+name] = float(np.linalg.norm(ll['base'][:,m],axis=-1).mean())
        row['delta_norm_'+name] = float(np.linalg.norm(ll['delta'][:,m],axis=-1).mean())
        row['delta_nonzero_'+name] = float((np.linalg.norm(ll['delta'][:,m],axis=-1)>1e-8).mean())
        row['delta_base_cosine_'+name] = float((ll['delta'][:,m]*ll['base'][:,m]).sum(-1).__truediv__(
            np.linalg.norm(ll['delta'][:,m],axis=-1)*np.linalg.norm(ll['base'][:,m],axis=-1)+1e-8).mean())
    row['residual_relative_max'] = float(relative.max())
    row['residual_near_zero_selected_fraction'] = float((relative[:,[0,3]]<1e-5).mean())
    original_text=a.ModalFusion.prompt_text_features
    p0_projected=a.ModalFusion.MLP(torch.cat([original_text[:4].reshape(1,-1),original_text[4:].reshape(1,-1)],0)).detach()
    q=torch.nn.functional.normalize(state['enhanced'].view(871,4,4,9),dim=-1)
    gate=torch.einsum('nmtf,cf->nmtc',q,torch.nn.functional.normalize(p0_projected,dim=-1)).clamp(-5,5).softmax(-1).max(-1).values
    row['original_spgs_weight_mean']=float(gate.mean())
    row['original_p0_projection_norm_mean']=float(p0_projected.norm(dim=-1).mean())
    row['original_p0_projection_nonfinite']=not bool(torch.isfinite(p0_projected).all())
    for modality in ['corr','pheno']:
        sem,q = ll['semantic_'+modality],ll['query_'+modality]
        row['semantic_'+modality+'_asd_mean'] = float(sem[:,0].mean())
        row['semantic_'+modality+'_td_mean'] = float(sem[:,1].mean())
        stat_values(row,'semantic_'+modality+'_asd_minus_td',sem[:,0]-sem[:,1])
        stat_values(row,'query_'+modality+'_norm',np.linalg.norm(q,axis=-1))
        unit = q/(np.linalg.norm(q,axis=-1,keepdims=True)+1e-8)
        row['query_'+modality+'_centered_variance'] = float(unit[a.trn_idx].var(0).mean())
        row['query_'+modality+'_mean_pair_cosine'] = float(((unit.sum(0)**2).sum()-(unit**2).sum())/max(len(unit)*(len(unit)-1),1))
    stat_values(row,'quality_confidence_proxy',ll['confidence'])
    data = read_json(MANIFEST)['data_audit']; raw = context()['x']
    sites = raw[:,data['site_indices']].argmax(1)
    row['align_by_site'] = json.dumps({str(site):float(ll['align'][sites==site].mean()) for site in sorted(set(sites))},sort_keys=True)
    ages = raw[:,data['age_indices']].argmax(1)
    sexes = raw[:,data['sex_indices']].argmax(1)
    row['align_age_bin_correlation'] = float(np.corrcoef(ages,ll['align'][:,0])[0,1]) if ll['align'].std()>1e-8 else 0.
    row['align_by_sex'] = json.dumps({str(s):float(ll['align'][sexes==s].mean()) for s in sorted(set(sexes))})
    for key in ['shared_corr','shared_pheno','private_corr','private_pheno']:
        model = RidgeClassifier(alpha=1.).fit(ll[key][a.trn_idx],sites[a.trn_idx])
        row[key+'_site_probe_acc'] = float((model.predict(ll[key][a.tst_idx])==sites[a.tst_idx]).mean())
    # Lightweight same-checkpoint path checks for every auxiliary group.
    real_logits = a._ds3a_logits.clone()
    for condition in ['RESIDUAL_OFF','SKIP','NO_PRIVATE','NO_QC'] + (['NEUTRAL','UNRELATED','SWAPPED'] if b.design.prompt!='OFF' else []):
        b.condition = condition; altered = forward(a); now = snapshot(a)
        row[condition.lower()+'_probability_abs_change'] = float((altered.exp()[a.tst_idx,0]-out.exp()[a.tst_idx,0]).abs().mean())
        row[condition.lower()+'_align_abs_change'] = float((now['align']-state['align']).abs().mean())
    b.condition = 'REAL'; restored = forward(a)
    assert torch.equal(out,restored)
    assert torch.equal(a._ds3a_logits,real_logits)
    return row


def assert_historical_fold(row):
    if row['config']!='D0': return
    for number,cfg in history_refs():
        folder = next((ROOT/'experiment').glob(f'experiment{number}_*'))
        ref = next(r for r in read_csv(folder/'results/fold_metrics.csv')
                   if r['config']==cfg and int(r['seed'])==int(row['seed']) and int(r['fold'])==int(row['fold']))
        for key in METRICS:
            refkey = 'balanced_accuracy' if number==14 and key=='ba' else key
            assert abs(float(row[key])-float(ref[refkey]))<1e-12, ('history metrics',number,row['seed'],row['fold'],key)
        for key in ['best_epoch','epochs_executed']:
            assert int(row[key])==int(ref[key]), ('history checkpoint',number,key,row[key],ref[key])
        initkey = 'initial_trunk_legacy13_sha256' if number==13 else 'initial_trunk_sha256'
        assert row[initkey]==ref['initial_trunk_sha256'], ('history init',number,row['seed'],row['fold'])
        rngkey = 'fold_entry_rng_sha256' if number==14 else 'entry_rng_sha256'
        if rngkey in ref: assert row['entry_rng_sha256']==ref[rngkey], ('history RNG',number,row['seed'],row['fold'])


def preflight():
    import torch
    import numpy as np
    from torch.nn import functional as F
    c = context(); integrity(full=True)
    checks = dict(status='RUNNING',started_utc=utc(),configurations={},
        data=data_audit(),original_parameter_count=1462957,clip_encoding_during_training=False)
    # Native source, without any experiment module, provides the numerical oracle.
    for seed in range(3):
        seed_rng(seed); entry = rng()
        native = agent_for('D0',seed,1,original=True)
        initial = weight_hash(native); native_rng = rng()
        native_initial = model_state(native)
        native_out = forward(native)
        assert torch.equal(native.ModalFusion.prompt_text_features.cpu(),c['static']['p0'])
        common_branch = None
        for cfg in CONFIGS:
            restore_rng(entry); c['clip_loaded']=False
            a = agent_for(cfg,seed,1); count = optimizer_audit(a,cfg)
            assert weight_hash(a)==initial and rng_hash(rng())==rng_hash(native_rng)
            outputs = forward(a); assert torch.equal(outputs,native_out), ('initial forward differs',seed,cfg)
            assert weight_hash(a,legacy13=True)==next(r['initial_trunk_sha256'] for r in read_csv(
                ROOT/'experiment/experiment13_patient_specific_fc_prompt/results/fold_metrics.csv')
                if r['config']=='F0' and int(r['seed'])==seed and r['fold']=='1')
            if cfg!='D0':
                b = branch(a)
                assert b.alpha.count_nonzero()==0 and b.last['delta'].count_nonzero()==0
                new_hash = weight_hash(a,True,exclude_learned=True)
                if common_branch is None: common_branch=new_hash
                assert new_hash==common_branch
                assert all(v.shape==(871,32) for k,v in b.last.items() if k.startswith(('shared_','private_')))
                assert b.text_banks.shape==(3,2,1024) and not b.text_banks.requires_grad
                assert b.last['align'].shape==(871,1) and b.last['base'].shape==(871,4,36)
            checks['configurations'][f'{cfg}_seed{seed}'] = dict(total_parameters=count,
                original_init_sha256=initial,branch_init_sha256=weight_hash(a,True) if cfg!='D0' else '',
                new_common_init_sha256=common_branch or '',entry_rng_sha256=rng_hash(entry),
                initial_output_exact=True,existing_parameter_rng_exact=True)
            del a
        # Two updates verify D0 retains classification, graph loss and optimizer math.
        restore_rng(native_rng)
        native_audit=[]
        for step in range(2):
            native.run_epoch('simple-2')
            native_audit.append((weight_hash(native),rng_hash(rng()),forward(native)))
        restore_rng(entry); c['clip_loaded']=False
        d0 = agent_for('D0',seed,1)
        for step in range(2):
            d0.run_epoch('simple-2')
            oracle = native_audit[step]
            assert weight_hash(d0)==oracle[0] and rng_hash(rng())==oracle[1]
            assert torch.equal(forward(d0),oracle[2])
        del d0,native,native_audit,native_initial
    forbid_encoding()
    seed_rng(0); c['clip_loaded']=False
    a = agent_for('D5',0,1)
    shapes = {}
    def hook_shape(name):
        def hook(module,inputs,out):
            value = out[0] if isinstance(out,tuple) else out
            shapes[name] = list(value.shape)
        return hook
    hooks=[a.MessagePassing0.register_forward_hook(hook_shape('gfr')),
           a.ModalFusion.InputLayer.register_forward_hook(hook_shape('pln_input')),
           a.ModalFusion.FeedForward[-1].register_forward_hook(hook_shape('cam'))]
    forward(a)
    for h in hooks: h.remove()
    assert shapes==dict(gfr=[871,320],pln_input=[871,4,36],cam=[871,4,36])
    checks['tensor_shapes']=shapes
    grads=[]
    for step in range(2): grads.append(train_step(a))
    for key in ['shared_corr','private_corr','shared_pheno','private_pheno',
                'query_corr','query_pheno','quality_encoder','quality_confidence','align_skip','fusion','alpha']:
        assert max(d[key] for d in grads)>0, ('disconnected',key,grads)
    checks['two_step_gradients']=grads
    # Restore full model, both optimizers and all streams, then repeat a third step.
    for cfg in ['D0','D5']:
        seed_rng(1); c['clip_loaded']=False; aa = agent_for(cfg,1,1)
        train_step(aa); train_step(aa)
        saved = dict(model=model_state(aa),mp=aa.optimizer_MP0.state_dict(),
                     mf=aa.optimizer_MF.state_dict(),rng=rng())
        probe = STATE/'preflight-recovery.pt'; save_checkpoint(probe,saved)
        train_step(aa); expected=(weight_hash(aa),weight_hash(aa,True),rng_hash(rng()),forward(aa))
        recovered=checkpoint(probe);load_state(aa,recovered['model'])
        aa.optimizer_MP0.load_state_dict(recovered['mp']);aa.optimizer_MF.load_state_dict(recovered['mf'])
        restore_rng(recovered['rng']);train_step(aa)
        assert (weight_hash(aa),weight_hash(aa,True),rng_hash(rng()))==expected[:3]
        assert torch.equal(forward(aa),expected[3])
        probe.unlink();del aa
    checks['full_model_optimizer_rng_recovery_exact']=True
    # Fixed probe, no performance selection. Open alpha only in this disposable model.
    seed_rng(2); c['clip_loaded']=False; a=agent_for('D5',2,1); b=branch(a)
    with torch.no_grad(): b.alpha.fill_(.2)
    out=forward(a); cf=counterfactual(a,2,1,out)
    for condition in ['NEUTRAL','UNRELATED','SKIP','NO_PRIVATE','NO_QC','RESIDUAL_OFF']:
        row=next(r for r in cf if r['condition']==condition)
        assert row['mean_abs_asd_probability_change']>0, ('intervention disconnected',condition)
    checks['probe_counterfactual']={r['condition']:{k:r[k] for k in ['mean_abs_asd_probability_change','align_mean_abs_change']} for r in cf}
    ll=snapshot(a)
    # Labels have no forward signature. Held-out labels also cannot alter auxiliary loss.
    real_loss=b.loss(a.targ,a.trn_idx).detach().clone()
    perturbed=a.targ.clone();perturbed[a.tst_idx]=1-perturbed[a.tst_idx]
    b._train_indices=None;b._class_weights=None
    assert torch.equal(real_loss,b.loss(perturbed,a.trn_idx).detach())
    b._train_indices=None;b._class_weights=None
    raw=[x.clone() for x in a.dataLayer(a.feat)[1]]
    raw[0][:,read_json(MANIFEST)['data_audit']['site_indices']]*=-3
    changed=b(ll['base'],raw)
    assert torch.equal(changed,ll['enhanced']), 'site entered auxiliary path'
    checks['test_labels_excluded_from_loss']=True; checks['site_auxiliary_forward_invariance']=True
    # Exercise the complete post-training record writer on the disposable probe.
    out=forward(a)
    diagnostic_probe=diagnostics(a,'D5',2,1,out,gradients(a),parameter_norms(a),{})
    assert all(math.isfinite(v) for v in diagnostic_probe.values() if isinstance(v,(int,float)))
    checks['diagnostic_writer_finite']=True
    checks['auxiliary_capacity']={cfg:capacity_audit(DS3A(cfg,c['static']['pheno_indices'],c['static']['banks'],0))
                                 for cfg in CONFIGS[1:]}
    # Loss scale audit uses training labels and gradients; no validation scores.
    a.MessagePassing0.train();a.ModalFusion.train()
    n,raw=a.dataLayer(a.feat);feat,adj,_=a.MessagePassing0(raw,n,a.targ,a.trn_idx)
    outputs,_,_=a.ModalFusion(feat,raw)
    cls=c['model'].ClsLoss(outputs,a.targ,a.trn_idx,a.weight)
    graph=sum(c['model'].GraphConstructLoss(raw[i],adj[i],1.,.5,0.) for i in range(n))/4
    b.loss(a.targ,a.trn_idx)
    allparams=[p for mm in [a.MessagePassing0,a.ModalFusion] for p in mm.parameters()]
    scales={}
    for name,loss in [('classification',cls),('original_graph',graph),
                     ('alignment_unweighted',b.last['align_loss']),('private_unweighted',b.last['private_loss'])]:
        gg=torch.autograd.grad(loss,allparams,allow_unused=True,retain_graph=True)
        norm=float(torch.stack([g.square().sum() for g in gg if g is not None]).sum().sqrt())
        assert math.isfinite(float(loss)) and math.isfinite(norm)
        scales[name]=dict(value=float(loss.detach()),gradient_l2=norm)
    checks['fixed_coefficient_gradient_scales']=scales
    checks['residual_bound']=True
    # Reconstructing the patch from tracked roots must apply cleanly to a disposable copy.
    patchcheck=STATE/'patch-check';patchcheck.mkdir(exist_ok=True)
    for name in ['model.py','network.py']:shutil.copy2(ROOT/name,patchcheck/name)
    subprocess.run(['git','apply','--check',str(EXP/'integration_patch.diff')],cwd=patchcheck,check=True)
    shutil.rmtree(patchcheck)
    integrity(full=True)
    checks.update(status='PASS',completed_utc=utc(),original_two_adam_steps_exact=True,
                  patch_applies=True,root_and_history_unchanged=True)
    m=read_json(MANIFEST);m['preflight']=checks;save_json(MANIFEST,m)
    print('PREFLIGHT_PASS '+json.dumps({k:v for k,v in checks.items() if k not in ['configurations','data']},ensure_ascii=False),flush=True)


def worker(config,seed):
    import torch
    m=integrity();assert m['preflight']['status']=='PASS'
    task=f'{config}_seed{seed}';physical=int(os.environ['DS3A_PHYSICAL_GPU'])
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(physical)
    assert m['tasks'][task]['physical_gpu']==physical
    task_lock=lock(STATE/(task+'.lock'));c=context();forbid_encoding()
    result_path,save_path=STATE/(task+'.json'),STATE/(task+'.pt')
    sig=signature()
    if result_path.exists() and read_json(result_path)['status']=='COMPLETE':
        old=read_json(result_path);assert old['signature']==sig and len(old['metrics'])==10
        print('SKIP_COMPLETE '+task,flush=True);return
    saved=checkpoint(save_path) if save_path.exists() else None
    if saved:
        assert saved['signature']==sig
        result=saved['result']
        if saved['phase']=='BETWEEN_FOLDS':save_json(result_path,result)
        restore_rng(saved['rng']);c['clip_loaded']=True
        start_fold=int(saved['fold'])+(1 if saved['phase']=='BETWEEN_FOLDS' else 0)
        print(f'RESUME {task} phase={saved["phase"]} fold={start_fold}',flush=True)
    else:
        seed_rng(seed)
        result=dict(signature=sig,task=task,config=config,seed=seed,physical_gpu=physical,
            status='RUNNING',metrics=[],diagnostics=[],counterfactual=[],started_utc=utc(),seconds=0.)
        start_fold=1
    print(f'WORKER {task} physical={physical} logical=cuda:0 device={torch.cuda.get_device_name(0)}',flush=True)
    for fold in range(start_fold,11):
        entry=rng();a=agent_for(config,seed,fold);optimizer_audit(a,config)
        initial,legacy13,branch_initial=weight_hash(a),weight_hash(a,legacy13=True),weight_hash(a,True)
        initial_norms=parameter_norms(a);tracker={k:0. for k in gradients(a)};path_tracker={}
        best_acc,best_epoch,best_model,wait,start,elapsed_before=0.,-1,None,0,0,0.
        if saved and saved['phase']=='EPOCH' and int(saved['fold'])==fold:
            load_state(a,saved['model']);a.optimizer_MP0.load_state_dict(saved['mp']);a.optimizer_MF.load_state_dict(saved['mf'])
            restore_rng(saved['rng'])
            best_acc,best_epoch,best_model,wait,start=(saved[k] for k in ['best_acc','best_epoch','best_model','wait','next_epoch'])
            tracker,entry,initial,legacy13,branch_initial=(saved[k] for k in ['tracker','entry_rng','initial','legacy13','branch_initial'])
            initial_norms,path_tracker,elapsed_before=(saved[k] for k in ['initial_norms','path_tracker','fold_seconds'])
        saved=None;t0=time.monotonic();last=start-1
        print(f'FOLD_START {fold} entry_rng={rng_hash(entry)} initial={initial}',flush=True)
        for epoch in range(start,450):
            print(f'EPOCH {fold}:{epoch} ',end='');a.run_epoch('simple-2')
            sample=epoch in [0,1] or (epoch+1)%25==0
            training_align_loss=float(branch(a).last['align_loss'].detach()) if sample and branch(a) is not None else 0.
            if sample:
                for k,v in gradients(a).items():tracker[k]=max(tracker[k],v)
            out=forward(a)
            val=float(out[a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean())
            print(f' val-acc={val:.12f}',flush=True)
            if val>best_acc:best_acc,best_epoch,best_model,wait=val,epoch,model_state(a),0
            else:wait+=1
            last=epoch;stop=wait>200 or epoch==449
            if sample or stop:
                if branch(a) is not None:
                    ll=branch(a).last
                    ratios=ll['delta'].norm(dim=-1)/(ll['base'].norm(dim=-1)+1e-8)
                    assert float(ratios.max())<=RESIDUAL_LIMIT+1e-6
                    for key,value in [('residual_relative_max',float(ratios.max())),
                                      ('align_sd',float(ll['align'].std(unbiased=False))),
                                      ('align_loss',training_align_loss)]:
                        path_tracker['training_sampled_max_'+key]=max(path_tracker.get('training_sampled_max_'+key,0.),value)
                    assert all(math.isfinite(v) for v in path_tracker.values())
            if (epoch+1)%25==0 or stop:
                save_checkpoint(save_path,dict(signature=sig,phase='EPOCH',fold=fold,next_epoch=epoch+1,
                    model=model_state(a),mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),
                    rng=rng(),best_acc=best_acc,best_epoch=best_epoch,best_model=best_model,wait=wait,
                    tracker=tracker,entry_rng=entry,initial=initial,legacy13=legacy13,branch_initial=branch_initial,
                    initial_norms=initial_norms,path_tracker=path_tracker,
                    fold_seconds=elapsed_before+time.monotonic()-t0,result=result))
            if stop:break
        assert best_model is not None
        continuation=rng();load_state(a,best_model);out=forward(a);met=metrics(out,a.tst_idx)
        assert abs(met['acc']-best_acc)<1e-12 and abs(met['ba']-met['hard_auc'])<1e-12
        row=dict(config=config,seed=seed,fold=fold,physical_gpu=physical,**met,
            best_epoch=best_epoch,epochs_executed=last+1,seconds=elapsed_before+time.monotonic()-t0,
            selected_val_acc=best_acc,train_n=len(a.trn_idx),test_n=len(a.tst_idx),val_equals_test=True,
            initial_trunk_sha256=initial,initial_trunk_legacy13_sha256=legacy13,
            initial_branch_sha256=branch_initial if config!='D0' else '',entry_rng_sha256=rng_hash(entry),
            additional_parameters=m['additional_parameters'][config])
        assert_historical_fold(row)
        result['metrics'].append(row)
        result['diagnostics'].append(diagnostics(a,config,seed,fold,out,tracker,initial_norms,path_tracker))
        if config=='D5':result['counterfactual'].extend(counterfactual(a,seed,fold,out))
        restore_rng(continuation)
        result.update(seconds=sum(r['seconds'] for r in result['metrics']),
                      status='COMPLETE' if fold==10 else 'RUNNING',updated_utc=utc())
        save_checkpoint(save_path,dict(signature=sig,phase='BETWEEN_FOLDS',fold=fold,rng=continuation,result=result))
        save_json(result_path,result)
        print('FOLD_COMPLETE '+json.dumps(row,ensure_ascii=False),flush=True)
        del a,best_model
        if fold==10:save_path.unlink(missing_ok=True)
    print('TASK_COMPLETE '+task,flush=True);task_lock.close()


def task_results():
    results=[]
    for c,s,g in TASKS:
        path=STATE/f'{c}_seed{s}.json'
        if path.exists():
            result=read_json(path)
            assert result['signature']==signature() and result['physical_gpu']==g
            keys=[int(r['fold']) for r in result['metrics']]
            assert keys==list(range(1,len(keys)+1))
            assert len(result['diagnostics'])==len(keys)
            if result['status']=='COMPLETE':assert len(keys)==10
            results.append(result)
    return results


def refresh():
    m=read_json(MANIFEST)
    if m['run_status']=='COMPLETE' and not STATE.exists():return m
    tasks=task_results()
    rows=[r for t in tasks for r in t['metrics']]
    ds=[r for t in tasks for r in t['diagnostics']]
    cf=[r for t in tasks for r in t['counterfactual']]
    assert len({(r['config'],r['seed'],r['fold']) for r in rows})==len(rows)
    for t in tasks:
        m['tasks'][t['task']].update(status=t['status'],folds=len(t['metrics']),seconds=t['seconds'],
                started_utc=t['started_utc'],updated_utc=t.get('updated_utc'))
    m.update(completed_folds=len(rows),completed_tasks=sum(t['status']=='COMPLETE' for t in tasks),
             diagnostic_rows=len(ds),counterfactual_rows=len(cf),updated_utc=utc())
    for name,items in [('fold_metrics',rows),('diagnostics',ds),('counterfactual',cf)]:save_csv(RESULTS/(name+'.csv'),items)
    save_json(MANIFEST,m)
    return m


def worker_env(gpu):
    return dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),
        DS3A_PHYSICAL_GPU=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')


def schedule(workers_per_gpu=4):
    assert 1<=workers_per_gpu<=4
    if MANIFEST.exists() and read_json(MANIFEST).get('run_status')=='COMPLETE':
        validate_complete();print('ALREADY_COMPLETE 36 tasks / 360 folds',flush=True);return
    prepare();scheduler_lock=lock(STATE/'scheduler.lock')
    if read_json(MANIFEST)['preflight'] is None:
        print('PREFLIGHT_START',flush=True)
        with open(LOGS/'preflight.log','a') as f:
            subprocess.run([sys.executable,str(EXP/'run.py'),'preflight'],cwd=EXP,env=worker_env(0),
                stdout=f,stderr=subprocess.STDOUT,check=True)
    m=read_json(MANIFEST);assert m['preflight']['status']=='PASS'
    m.update(run_status='RUNNING',workers_per_gpu=workers_per_gpu)
    m.setdefault('started_unix',time.time());m.setdefault('started_utc',utc())
    m['launch_resources']=resources();save_json(MANIFEST,m)
    attempts=dict(m.get('failures_and_restarts',{}))
    active={t:dict(pid=pid,process=None,gpu=m['tasks'][t]['physical_gpu']) for t,pid in live_workers().items()}
    last_refresh=0.;last_resource=None
    effective_limits={0:workers_per_gpu,1:workers_per_gpu}
    print('SCHEDULER_START '+utc(),flush=True)
    while True:
        for task,info in list(active.items()):
            proc=info['process']
            code=proc.poll() if proc else (None if Path('/proc',str(info['pid'])).exists() else 0)
            if code is not None:
                if 'log' in info:info['log'].close()
                del active[task]
                if code!=0:
                    attempts[task]=attempts.get(task,0)+1
                    m=read_json(MANIFEST);m['failures_and_restarts']=attempts;save_json(MANIFEST,m)
                    print(f'RESTART {task} exit={code} from atomic checkpoint',flush=True)
                    if attempts[task]>=3:raise RuntimeError('Repeated task failure: '+task)
        done={t['task'] for t in task_results() if t['status']=='COMPLETE'}
        for gpu in [0,1]:
            for cfg,seed,assigned in TASKS:
                task=f'{cfg}_seed{seed}'
                if assigned!=gpu or task in done or task in active:continue
                if sum(v['gpu']==gpu for v in active.values())>=effective_limits[gpu]:break
                log=open(LOGS/(task+'.log'),'a');log.write('\nATTEMPT_START '+utc()+'\n');log.flush()
                proc=subprocess.Popen([sys.executable,str(EXP/'run.py'),'--worker',cfg,str(seed)],
                    cwd=EXP,env=worker_env(gpu),stdout=log,stderr=subprocess.STDOUT)
                active[task]=dict(pid=proc.pid,process=proc,gpu=gpu,log=log)
                print(f'LAUNCH {task} GPU{gpu} pid={proc.pid}',flush=True)
        if time.monotonic()-last_refresh>20 or not active:
            m=refresh()
            if last_resource is None or time.monotonic()-last_resource['monotonic']>=60:
                sample=resources()
                if last_resource:
                    sample['cpu_cores_used']=(sample['cpu_usage_ns']-last_resource['cpu_usage_ns'])/1e9/(sample['monotonic']-last_resource['monotonic'])
                peaks=m.setdefault('runtime_resource_peaks',dict(rss_gib=0.,gpu0_mib=0,gpu1_mib=0,cpu_cores_used=0.))
                peaks['rss_gib']=max(peaks['rss_gib'],sample['cgroup_rss_gib'])
                peaks['cpu_cores_used']=max(peaks['cpu_cores_used'],sample.get('cpu_cores_used',0.))
                for card in sample['gpu']:peaks['gpu'+str(card['index'])+'_mib']=max(peaks['gpu'+str(card['index'])+'_mib'],card['used_mib'])
                # Resource pressure controls only future starts; never kills another job.
                for card in sample['gpu']:
                    gpu=card['index']
                    new_limit=1 if sample['cgroup_rss_gib']>110 or card['used_mib']>85000 else workers_per_gpu
                    if new_limit!=effective_limits[gpu]:
                        m.setdefault('resource_throttling',[]).append(dict(utc=utc(),gpu=gpu,limit=new_limit))
                        effective_limits[gpu]=new_limit
                save_json(MANIFEST,m);print('RESOURCE '+json.dumps(sample),flush=True);last_resource=sample
            print(f'PROGRESS tasks={len(done)}/36 folds={m["completed_folds"]}/360 active={len(active)}',flush=True)
            last_refresh=time.monotonic()
        if len(done)==36 and not active:break
        time.sleep(2)
    m=refresh();m.update(run_status='TRAINING_COMPLETE',completed_utc=utc(),
        wall_seconds=m.get('training_wall_seconds',time.time()-m['started_unix']),failures_and_restarts=attempts);save_json(MANIFEST,m)
    validate_complete();report();append_handoff();assert not live_workers();integrity(full=True)
    scheduler_lock.close();shutil.rmtree(STATE)
    m=read_json(MANIFEST);m.update(run_status='COMPLETE',temporary_state_cleaned=True,no_training_processes=True,
        handoff_sha256=sha(ROOT/'handoff.md'),report_sha256=sha(RESULTS/'report.md'),
        results_sha256={n:sha(RESULTS/(n+'.csv')) for n in ['fold_metrics','diagnostics','counterfactual']})
    save_json(MANIFEST,m);print('COMPLETE 36/36 tasks 360/360 folds; temporary state cleaned',flush=True)


def validate_complete():
    m=integrity(full=True);rows=read_csv(RESULTS/'fold_metrics.csv')
    ds=read_csv(RESULTS/'diagnostics.csv');cf=read_csv(RESULTS/'counterfactual.csv')
    keys={(c,str(s),str(f)) for c in CONFIGS for s in range(3) for f in range(1,11)}
    assert len(rows)==len(ds)==360
    assert {(r['config'],r['seed'],r['fold']) for r in rows}==keys
    assert {(r['config'],r['seed'],r['fold']) for r in ds}==keys
    assert len(cf)==240 and {(r['seed'],r['fold'],r['condition']) for r in cf}=={
        (str(s),str(f),cond) for s in range(3) for f in range(1,11) for cond in CF_CONDITIONS}
    assert [sum(int(r['physical_gpu'])==g for r in rows) for g in [0,1]]==[180,180]
    assert all(abs(float(r['selected_val_acc'])-float(r['acc']))<1e-12 for r in rows)
    for r in rows:assert_historical_fold(r)
    for r in ds:
        if r['config']!='D0':assert float(r['residual_relative_max'])<=.100001
    assert all(t['status']=='COMPLETE' and t['folds']==10 for t in m['tasks'].values())
    assert m['preflight']['status']=='PASS'
    audit = dict(validation_status='PASS',groups=12,tasks=36,folds=360,diagnostics=360,counterfactual=240,
        gpu_folds=[180,180],historical_p0_parity_folds=30,complete_seed_fold_keys=True,
        original_source_history_hashes_verified=True,validated_utc=utc())
    # Later folds legitimately diverge as continuous RNG consumes different epoch counts.
    lookup={(r['config'],int(r['seed']),int(r['fold'])):r for r in rows}
    audit['original_initialization_equal_to_D0_by_config']={c:sum(lookup[(c,s,f)]['initial_trunk_sha256']==lookup[('D0',s,f)]['initial_trunk_sha256']
        for s in range(3) for f in range(1,11)) for c in CONFIGS}
    assert all(lookup[(c,s,1)]['initial_trunk_sha256']==lookup[('D0',s,1)]['initial_trunk_sha256'] for c in CONFIGS for s in range(3))
    m['final_validation']=audit;save_json(MANIFEST,m)
    return audit


def aggregates(rows):
    import numpy as np
    return {c:{key:dict(seed=[float(np.mean([float(r[key]) for r in rows if r['config']==c and int(r['seed'])==s])) for s in range(3)],
        mean=float(np.mean([float(r[key]) for r in rows if r['config']==c])),
        sd=float(np.std([np.mean([float(r[key]) for r in rows if r['config']==c and int(r['seed'])==s]) for s in range(3)])))
        for key in METRICS} for c in CONFIGS}


def paired(rows,left,right,key):
    import numpy as np
    lookup={(r['config'],int(r['seed']),int(r['fold'])):float(r[key]) for r in rows}
    values=[lookup[(left,s,f)]-lookup[(right,s,f)] for s in range(3) for f in range(1,11)]
    return dict(mean=float(np.mean(values)),seed=[float(np.mean(values[s*10:(s+1)*10])) for s in range(3)],
        win=sum(v>1e-12 for v in values),tie=sum(abs(v)<=1e-12 for v in values),loss=sum(v< -1e-12 for v in values))


def mean_field(rows,key):
    values=[float(r[key]) for r in rows if r.get(key) not in ['',None]]
    assert values, ('missing diagnostic',key)
    return sum(values)/len(values)


def result_table(rows):
    agg=aggregates(rows)
    lines=['|组别|ACC|BA|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|','|---|---:|---:|---:|---:|---:|---:|']
    for c in CONFIGS:
        lines.append('|'+c+' '+DESIGNS[c].name+'|'+'|'.join(f'{100*agg[c][k]["mean"]:.4f}±{100*agg[c][k]["sd"]:.4f}' for k in METRICS)+'|')
    return '\n'.join(lines)


def scientific_judgment(rows,ds,cf):
    agg=aggregates(rows);d5=[r for r in ds if r['config']=='D5']
    stable=lambda l,r:all(paired(rows,l,r,k)['mean']>0 and all(v>0 for v in paired(rows,l,r,k)['seed'])
                          for k in ['acc','ba','probability_auc'])
    textrows=[r for r in cf if r['condition'] in ['NEUTRAL','UNRELATED','SWAPPED']]
    return dict(best={k:max(CONFIGS,key=lambda c:agg[c][k]['mean']) for k in ['acc','ba','probability_auc']},
        shared_alignment_three_metric_stable=stable('D3','D1'),
        shared_over_all_alignment_stable=stable('D3','D2'),
        selective_stable=stable('D4','D3'),quality_stable=stable('D5','D9'),private_stable=stable('D5','D10'),
        real_text_over_all_controls_stable=all(stable('D5',r) for r in ['D6','D7','D8']),
        d5_over_p0_stable=stable('D5','D0'),d5_over_capacity_stable=stable('D5','D1'),
        d5_align_mean=mean_field(d5,'align_test_mean'),d5_align_subject_sd=mean_field(d5,'align_test_sd'),
        d5_skip_below_half=mean_field(d5,'skip_lt_0_5_test'),
        d5_residual_pheno=mean_field(d5,'residual_relative_pheno_mean'),
        d5_residual_corr=mean_field(d5,'residual_relative_corr_mean'),
        text_probability_change=mean_field(textrows,'mean_abs_asd_probability_change'),
        text_change_folds=sum(any(float(r['mean_abs_asd_probability_change'])>1e-7 for r in textrows
            if int(r['seed'])==s and int(r['fold'])==f) for s in range(3) for f in range(1,11)),
        near_zero_residual_fraction=mean_field(d5,'residual_near_zero_selected_fraction'))


def report():
    rows=read_csv(RESULTS/'fold_metrics.csv');ds=read_csv(RESULTS/'diagnostics.csv');cf=read_csv(RESULTS/'counterfactual.csv')
    assert len(rows)==360 and len(ds)==360 and len(cf)==240
    m=read_json(MANIFEST);agg=aggregates(rows);j=scientific_judgment(rows,ds,cf)
    lines=['# 实验十七 DS³A-v2：疾病语义与质量条件化的选择性共享—特异对齐',
        '',f'完成时间：{m["completed_utc"]}；正式训练墙钟 {m["wall_seconds"]/60:.2f} 分钟。36/36连续十折任务、360/360 folds、360条诊断、240条D5反事实。GPU0/GPU1各18任务、180 folds；每卡最多4进程。',
        '', '## 结果与评价口径', '',
        '先计算每个seed的10折均值，再报告三个seed均值±population SD，单位为%。ASD=0；概率取log-softmax输出exp()的第0列。Hard-AUC由硬标签one-hot计算，二分类时等于BA，不能作为独立的排序性能证据。',
        '',result_table(rows),'',
        '三项最佳：'+ '；'.join(f'{k}：{c} ({agg[c][k]["mean"]*100:.4f}%)' for k,c in j['best'].items())+'。',
        '', '## 每个seed的十折均值', '', '|组别|seed|ACC|BA|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in CONFIGS:
        for s in range(3):lines.append(f'|{c}|{s}|'+'|'.join(f'{100*agg[c][k]["seed"][s]:.4f}' for k in METRICS)+'|')
    lines+=['','## 预设配对比较与D5全部对照','','差值为百分点；W/T/L为同seed、同fold的胜/平/负。连续随机流因训练epoch不同而在后续fold自然分叉，配对数据划分一致，并非全部fold初始化相同。30次训练共享10套划分，不视为30个独立数据集，不做独立样本显著性宣称。',
        '', '|比较|指标|均值差(pp)|seed0/1/2差(pp)|W/T/L|','|---|---|---:|---|---|']
    comparisons=list(dict.fromkeys(PAIRS+[('D5',c) for c in CONFIGS if c!='D5']))
    for left,right in comparisons:
        for key in METRICS:
            p=paired(rows,left,right,key)
            lines.append(f'|{left}−{right}|{key}|{p["mean"]*100:+.4f}|'+
                '/'.join(f'{v*100:+.4f}' for v in p['seed'])+f'|{p["win"]}/{p["tie"]}/{p["loss"]}|')
    lines+=['','## Shared/Private、Align/Skip与残差','','|组别|训练Shared原型距离|测试Shared原型距离|Private逐样本距离|Shared/Private绝对cos CORR/PHENO|Align均值/受试者SD|Skip<0.5|PHENO/CORR残差%|',
        '|---|---:|---:|---:|---|---|---:|---|']
    for c in CONFIGS[1:]:
        d=[r for r in ds if r['config']==c]
        lines.append(f'|{c}|{mean_field(d,"class_prototype_distance_train"):.5f}|{mean_field(d,"class_prototype_distance_test"):.5f}|{mean_field(d,"private_instance_distance_test"):.5f}|'+
            f'{mean_field(d,"shared_private_abs_cosine_corr"):.4f}/{mean_field(d,"shared_private_abs_cosine_pheno"):.4f}|'+
            f'{mean_field(d,"align_test_mean"):.4f}/{mean_field(d,"align_test_sd"):.4f}|{mean_field(d,"skip_lt_0_5_test")*100:.2f}%|'+
            f'{mean_field(d,"residual_relative_pheno_mean")*100:.4f}/{mean_field(d,"residual_relative_corr_mean")*100:.4f}|')
    d5=[r for r in ds if r['config']=='D5']
    lines+=['',f'D5 Align 5%/50%/95%分位数的折均值：{mean_field(d5,"align_test_p05"):.4f}/{mean_field(d5,"align_test_p50"):.4f}/{mean_field(d5,"align_test_p95"):.4f}；<0.1比例{mean_field(d5,"skip_lt_0_1_test")*100:.2f}%，>0.9比例{mean_field(d5,"align_gt_0_9_test")*100:.2f}%。ASD/TD事后均值{mean_field(d5,"align_asd_test"):.4f}/{mean_field(d5,"align_td_test"):.4f}；QC proxy均值{mean_field(d5,"quality_confidence_proxy_mean"):.4f}，它没有真实质量标签校准。',
        '',f'D5训练CORR/PHENO类别原型cosine折均值{mean_field(d5,"corr_class_prototype_cosine_train"):.4f}/{mean_field(d5,"pheno_class_prototype_cosine_train"):.4f}。原型对齐距离下降可以由类别坍塌导致；不能单独证明共享诊断信息增加。Shared的ASD线性probe BA（CORR/PHENO）{mean_field(d5,"shared_corr_probe_ba")*100:.2f}%/{mean_field(d5,"shared_pheno_probe_ba")*100:.2f}%，Private为{mean_field(d5,"private_corr_probe_ba")*100:.2f}%/{mean_field(d5,"private_pheno_probe_ba")*100:.2f}%。这些固定Ridge(alpha=1)训练折拟合、测试折评估的probe仅用于描述表示，不参与选模。',
        '',f'D5站点probe ACC（Shared CORR/PHENO）为{mean_field(d5,"shared_corr_site_probe_acc")*100:.2f}%/{mean_field(d5,"shared_pheno_site_probe_acc")*100:.2f}%。新增分支未直接读取站点，但年龄、性别、FC与站点仍可能相关；原主干仍含站点，不能宣称消除站点混杂。',
        '',f'D5查询向量中心化方向方差（CORR/PHENO）{mean_field(d5,"query_corr_centered_variance"):.8f}/{mean_field(d5,"query_pheno_centered_variance"):.8f}；平均两受试者查询cosine{mean_field(d5,"query_corr_mean_pair_cosine"):.4f}/{mean_field(d5,"query_pheno_mean_pair_cosine"):.4f}。冻结真实文本原型cosine为{m["text_encoding"]["REAL"]["prototype_cosine"]:.6f}；中性/无关原型为{m["text_encoding"]["NEUTRAL"]["prototype_cosine"]:.6f}/{m["text_encoding"]["UNRELATED"]["prototype_cosine"]:.6f}。文本几何与医学内容同时变化，阴性文本对照不能完全隔离语言几何因素。',
        '',f'全部辅助模型最大单受试者、单目标Token相对L2残差为{max(float(r["residual_relative_max"]) for r in ds if r["config"]!="D0")*100:.4f}%，低于预注册10%上限。D5目标Token近零残差(<1e-5)比例为{j["near_zero_residual_fraction"]*100:.2f}%；PHENO/CORR位置分别为{j["d5_residual_pheno"]*100:.4f}%/{j["d5_residual_corr"]*100:.4f}%，ANAT/FUNC残差恒为0。该位置已经经过CAM交互，不能把位置贡献当作纯模态因果贡献，更不能解释为ROI。',
        '', '## D5同一最佳checkpoint反事实', '',
        '不重训、不重新选epoch、不重新初始化；所有条件使用同一受试者和checkpoint，原主干参数及注入前CAM Token逐值不变，训练后均恢复REAL。SKIP同时令共享融合输入为0，Private仍可响应；它不能回撤训练期对齐损失。Shared/Private编码本身在全部干预中逐值不变，改变的是gate及融合输入。',
        '', '|条件|ΔACC(pp)|ΔBA(pp)|ΔProb-AUC(pp)|翻转率%|ASD概率绝对变化|Align绝对变化|delta绝对变化|logit绝对变化|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cond in CF_CONDITIONS:
        rr=[r for r in cf if r['condition']==cond]
        lines.append(f'|{cond}|'+'|'.join(f'{mean_field(rr,k)*scale:.8f}' for k,scale in [
            ('delta_acc',100),('delta_ba',100),('delta_probability_auc',100),('prediction_flip_rate',100),
            ('mean_abs_asd_probability_change',1),('align_mean_abs_change',1),('delta_mean_abs_change',1),('mean_abs_logit_change',1)])+'|')
    lines+=['',f'文本干预在{j["text_change_folds"]}/30个checkpoint产生>1e-7的ASD概率变化。平均文本干预概率变化{j["text_probability_change"]:.9f}。计算图敏感性与正确临床语义的独立诊断价值是两项不同证据；后者还须D5同时优于D6/D7/D8。',
        '', '## 五项科学判断', '',
        f'**A 对齐价值**：D3相对D1在ACC/BA/Prob-AUC均为正且三个seed均为正：{j["shared_alignment_three_metric_stable"]}；Shared-only相对Shared+Private满足同标准：{j["shared_over_all_alignment_stable"]}。详见配对差值，不以原型距离作为分类收益替代。',
        '',f'**B 选择性价值**：D4相对D3满足三指标、三seed同向标准：{j["selective_stable"]}。D5权重均值{j["d5_align_mean"]:.4f}，受试者SD{j["d5_align_subject_sd"]:.4f}；Skip<0.5比例{j["d5_skip_below_half"]*100:.2f}%。低均值或低方差可能是总体收缩，而非有意义的患者特异选择；该阈值只作描述。',
        '',f'**C 疾病语义价值**：D5同时相对中性、无关文本、可学习类别原型达到三指标及三seed同向标准：{j["real_text_over_all_controls_stable"]}。若False，本轮不能证明ASD医学语言具有超过任意向量或新增容量的独立贡献。',
        '',f'**D 数据角色价值**：D5相对D9质量条件的稳定收益：{j["quality_stable"]}；相对D10 Private融合的稳定收益：{j["private_stable"]}。D11−D3用于检验质量指标强制对齐，但不能单由D5−D11分离QC、语义、选择性等多个因素。年龄/性别不是临床症状；Shared与Private probe和原型坍塌诊断仅提供有限的共享信息证据。',
        '',f'**E 分类价值**：D5相对原P0、容量控制的三指标三seed同向改善分别为{j["d5_over_p0_stable"]}/{j["d5_over_capacity_stable"]}。SEN/SPE也完整列出；不能只挑ACC。',
        '', '## 与历史E9及既有方法的关系', '',
        '历史实验九E9：ACC 91.8826%、BA/Hard-AUC 91.6915%、Prob-AUC 93.8227%、SEN 89.2480%、SPE 94.1351%。它更换P3且连续随机轨迹不同，仅作描述性比较。',
        '', 'D5−E9描述性差值(pp)：'+ '；'.join(f'{k} {100*agg["D5"][k]["mean"]-v:+.4f}' for k,v in [
            ('acc',91.88262),('ba',91.69151),('probability_auc',93.82272),('sensitivity_asd',89.24797),('specificity_td',94.13506)])+'。',
        '', '本轮三项工程改进：辅助编码器读取真实原始模态并排除直接站点输入；ANAT/FUNC主要作QC条件而非疾病共享目标；零初始化且相对L2≤10%的残差保留原判别路径。Common/Private、类原型对齐及语义融合均有既有工作，不可称作通用组件首创。文献审计和具体方法差异见README。',
        '', '## 复现、限制与下一阶段', '',
        f'D0全部30fold与实验13F0、14E00、15A0、16B0六指标、best/executed epochs逐值一致；初始化采用各历史版本对应序列化格式核对，14–16入口RNG逐值一致。原始参数1462957；新增参数{m["additional_parameters"]["D5"]}，D8再多2048。所有注册参数均进入优化器，关闭组件允许无梯度；实际参与情况见diagnostics的sampled_gradient_max。完整model/双optimizer/全部RNG恢复已通过两步后第三步逐值复演。',
        '',f'正式运行资源峰值：{json.dumps(m.get("runtime_resource_peaks",{}),ensure_ascii=False)}。失败/恢复记录：{json.dumps(m.get("failures_and_restarts",{}))}。不同组入口初始化与D0相同fold数：{json.dumps(m["final_validation"]["original_initialization_equal_to_D0_by_config"])}。',
        '', '限制：val=test，并以测试折ACC选checkpoint；全体871人transductive图；沿用预先标准化与RFE，而非本轮训练折内重算。结果是历史协议内比较，不能声称独立无泄漏泛化。固定12组没有无类别对齐训练对照，无法证明类别条件对齐的增量价值；D2另对Private加同系数对齐，合计对齐预算更大，D11激活704个QC共享编码参数，D8多2048，均不隐瞒有效容量差异。停止梯度避免gate直接利用小权重规避损失，但不证明gate等于真实互信息或可靠性。',
        '', '下一阶段应先检验真实临床症状与FC是否存在共享诊断信号，并在训练内选模、独立站点验证的协议中复核；如要隔离类条件对齐或QC交互，另预注册单因素对照。当前不自动修改Prompt、不继续超参数搜索，也不因小幅均值提升宣称达到MICCAI 2027标准。',
        '', '根源码、实验1–16全部哈希验收通过；handoff仅追加；无自动commit/push。完整参数/数据/文本/源码哈希和配置位于manifest.json，temporary sources由integration_patch.diff重建，正式完成后清理临时checkpoint。']
    atomic(RESULTS/'report.md','\n'.join(lines)+'\n')
    m['scientific_judgment']=j;save_json(MANIFEST,m)


def append_handoff():
    m=read_json(MANIFEST);path=ROOT/'handoff.md'
    with open(ROOT/'handoff.md.lock','a+') as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
        content=path.read_bytes();n=m['handoff_prefix_bytes']
        assert hashlib.sha256(content[:n]).hexdigest()==m['handoff_prefix_sha256']
        marker=b'<!-- experiment17_disease_semantic_selective_alignment -->'
        if marker not in content:
            block='\n\n'+marker.decode()+'\n\n## 实验十七（DS3A-v2）正式完成\n\n'
            block+='MICCAI 2027第一创新模块候选：疾病语义与质量条件化的选择性Shared/Private对齐。下文保留完整正式统计和负结果，未执行额外Prompt或超参数搜索。\n\n'
            block+=f'代码版本：本地HEAD {m["git_head_at_prepare"]}；项目远端审计 {m["github_audit"]["tangaii/SPromptGL"]["head"]}。源/数据哈希：`{EXP}/manifest.json`。\n\n'
            block+=f'正式实验目录：`{EXP}`；报告：`{RESULTS}/report.md`；运行器：`{EXP}/run.py`；源码扩展：`{EXP}/ds3a.py`。\n\n'
            block+=(RESULTS/'report.md').read_text().replace('# 实验十七 DS³A-v2：疾病语义与质量条件化的选择性共享—特异对齐','### 实验十七验收记录',1)
            atomic(path,content+block.encode())


def write_readme(m):
    text='''# DS³A-v2：实验十七

本实验在原SPromptGL P0之外增加读取真实原始模态的选择性Shared/Private辅助分支。正式范围固定12组×3 seeds×连续10 folds；不改变根源码、数据、既有实验，不自动commit/push。结果仅用于本项目历史协议内比较。

## 源码及数据审计

官方仓库`wanxixi11/SPromptGL`与本地的差异主要为CLIP文本缓存、标签one-hot设备适配；项目远端`tangaii/SPromptGL`六个核心文件与本地逐字节一致。审计提交、六文件哈希和所有数据/CLIP哈希见manifest。已逐项读取原论文Fig.1–3、Section2.1/2.2与Eq.(1)–(7)，本地PDF解析文本与官方相同。

原GFR在四个关系图上处理拼接320维，输出[871,320]已经混合，且循环中的已有覆盖行为保持不变；不能把输出分段视为纯模态。原`simple-2`反传`L_cls+L_graph`，GFR的ALoss虽计算并打印却未进入loss。PLN InputLayer/CAM产生[871,4,36]；此次注入在最后FeedForward之后、`text_features = self.prompt_text_features`之前。P0的8×1024→2×4096→MLP(4096,128,9)，产生两个非负9维原型；原cosine→softmax→max标量加权[871,4,4,9]，global平均池化与局部channel-softmax响应相加，分类器输出log-softmax。该实现不能直接解释成真实ROI或论文所有公式的完整实现。

CSV实测871×321，320个输入+标签；ASD原1→0(403人)，TD原2→1(468人)。PHENO48=26年龄分箱+20站点+2性别，新增PHENO仅由列名前缀提取28个年龄/性别字段；站点仍存在原主干。ANAT6=cnr/efc/fber/fwhm/qi1/snr，FUNC10=efc/fber/fwhm/dvars/outlier/quality/mean_fd/num_fd/perc_fd/gsr，均为预计算影像QC，不是脑体积或完整时间序列；`func_gsr`是Ghost-to-Signal Ratio，不能称为全局信号回归。QC方向不统一，不手设“越大越好”，sigmoid只称quality-conditioned confidence proxy。CORR256来自HO ROI相关矩阵、上三角Fisher-z及既有RFE，原数据处理用ConnectivityMeasure相关估计；当前不重算。列顺序、有限性、映射数量由prepare程序断言，详见data_audit。

## 历史实验复核

1：P1仅小幅超过P0；2：对比状态文本未稳定提高；3：固定/替换gate、语义投影诊断显示内容作用弱；4：新增ASD特定FC文本无稳定收益；5：取消ReLU/减衰减的MLP调整未优于P1；6：方向标量门控无稳定收益；7：双主题特征残差未证明正确主题优于容量控制；8：疾病Shared支路与原预测高度冗余；9：E9表现较好(ACC91.8826、Prob-AUC93.8227)，语言内容独立作用未证明；10：真实疾病文本相对中性优势很小；11：256FC列核实252具名、4未知，不能给潜在Token强加ROI名；12：ROI语义可实现但小规模试验不足；13：患者FC提示未稳定超过基线；14：层级文本投影趋同且敏感性弱；15：语义通路在强衰减下退化，反事实变化接近0；16：恢复通路有文本敏感性，但残差约32.33%、ACC下降。报告/真实逐折CSV与关键patch均在创建本模块前审计，全部既有文件哈希作为只读保护。

另读取`/data/adodas/MMGL-main/experiment`的alignment_validation、exp2_selective_alignment、exp3_disentangled_alignment、exp4_pairadaptive及exp4_legacy。最早严格基线ACC84.62，统一hybrid对齐86.34；后三轮严格3seed中selective-medium84.69未稳定超过uniform-strong84.88，disentangled-medium84.96但强对齐下降，pair-adaptive BA略增而Prob-AUC不增、权重近均匀。MMGL验证划分/训练协议不同，不是本轮性能基线。该目录无独立exp5，不能虚构“实验五MMGL”记录。

## 文献方法与本研究边界

- [SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)：关系图、CAM和语义门控/残差；本轮以实际P0源码为不可变主路径，不声称补丁复现所有论文公式。
- [MGDR](https://papers.miccai.org/miccai-2024/paper/2494_paper.pdf)：关系图表示、共性空间分解与特异表示/融合；本轮不是SVD或多模态特异注意力重建。
- [BrainPrompt](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf)：真实ROI层、个体文本建图、疾病文本对齐；我们的年龄/性别/QC与潜在Token没有同等症状及ROI身份。
- [PhenoBrain](https://raw.githubusercontent.com/mlresearch/v306/main/assets/meng26f/meng26f.pdf)：症状表型条件的低秩路由、多跳脑连接与表示学习；其症状表型不等于本项目人口学背景。
- [To Align or Not to Align](https://arxiv.org/pdf/2511.12121)：对齐强度应随冗余/特异信息变化，并通过可控信息分布验证；本轮匹配分数只是代理，不称精确互信息。
- [Class-Aware Semantic Decoupling](https://papers.miccai.org/miccai-2026/paper/1452_paper.pdf)：类别共享/特异、语义权重、实例和分布对齐与类感知融合；这些通用组件已有。本轮用全局32维Shared/Private、小型训练类别原型损失和可拒绝门控，研究不同输入角色，不等同于其类别特定分解。
- [CALM](https://arxiv.org/pdf/2607.01656)：非配对影像/遗传的类别对齐和独立配对验证；本项目没有遗传或独立配对验证，不能借用其证据。
- [CR-mCCAR](https://pmc.ncbi.nlm.nih.gov/articles/PMC12869378/)：联合优化跨模态典型相关、临床参考相关并压低协变量相关；本轮QC条件不等于混杂回归或因果校正。
- [SABER](https://arxiv.org/pdf/2607.01901)：实际ROI语义、多尺度脑图、个体条件残差与疾病对齐；本轮不加ROI图、复杂超图或新文本分类损失。

以上是文献已有方法；“CORR与年龄/性别有可对齐诊断信息”“QC有助于选择或拒绝”是本轮待检验假设，允许负结果。新贡献候选是明确数据角色、患者特异可拒绝决策与保留原判别路径的联合证据，不把Common/Private或类原型对齐单独称首创。

## 固定数学方案

所有辅助编码器Linear→GELU→LayerNorm；CORR256与PHENO28各有32维Shared和Private，QC16→16。冻CLIP RN50每条文本先归一化，每类4句平均再归一化，得2×1024原型；原P0八句不变。真实/中性文本逐字复用历史9/10，UNRELATED骨关节炎/2型糖尿病文本仅是背景控制；具体句子、长度、token数和编码哈希可核查prompts.json/manifest。

Shared单位向量经无bias的32→1024查询映射，与ASD/TD原型取cosine得到4个分数。Gate输入29维：Shared cosine、平滑单位表示差、两个范数/均值/方差(8维)+文本分数4维+QC16维+QC proxy1维；Gate Linear29→16→2+softmax生成Align/Skip。关闭文本时分数0，关闭QC时embedding0且proxy0.5，统一对齐组Align=1。

训练原型由本折train_idx按ASD/TD分组、用detach(Align)加权求均值后归一化；L_align是跨模态同类原型平方距离平均，D2另加Private原型距离，D11平均四模态六对距离。无效类别/极小分母妥善处理；forward不接受标签。Shared/Private去冗余为mean(relu(abs(cos)−0.5)^2)，统一系数0.01，不要求硬正交。L_total=L_original+0.1 L_align+0.01 L_private。D1的L_align=0，仍保留相同去冗余，故它是统一正则下容量控制，不能称完全无辅助正则。

融合输入[Align×s_corr,p_corr,Align×s_pheno,p_pheno,Align]共129维→64→72；输出两个36维方向，对PHENO位置0和CORR位置3增强。每个方向d归一化为d/sqrt(sum(d²)+1e-8)，delta=0.1*tanh(alpha)*detach(||base||)*direction。alpha为2个全零参数，故初始前向与P0逐值相同，每人每目标位置相对L2≤10%；这是预注册工程上限不是最优值。注入后完整执行原SPGS/RES及分类器，ANAT/FUNC位置不注入。

新增模块在原PLN完全注册并执行原my_weight_init之后，在fork_rng中统一seed=170000+seed创建；原网络/CPU/CUDA随机轨迹不变，无新dropout。原参数Adam lr0.0038/WD0.11；新增参数同lr/WD0固定。D8的两个随机单位类别向量用独立局部Generator初始化，不包含CLIP先验，多2048个可学习参数。

## 十二组与容量审计

CONFIG_TABLE

所有D1–D11注册相同基础模块，D8追加2048参数。关闭组件仍进入优化器但可无梯度；D11激活另注册的ANAT/FUNC Shared编码器704参数，仅进入对齐损失。D10 Private仍参与固定去冗余但不进入增强融合；不能解释为删除整个原主干私有信息。D2同时Shared与Private对齐，总对齐项更多；D5−D4同时增加疾病语义和QC，不能独立归因。固定12组无无类别对齐训练对照，unconditional距离仅诊断，类别条件对齐的额外价值待后续独立预注册检验。

## 训练、恢复和复核

871人原始transductive四图；StratifiedKFold(10,shuffle=True,random_state=10)，seeds0/1/2，一次seed后同一进程连续10fold。最多450epoch，wait>200停止，严格测试折ACC提升才保存首次最佳；val_idx=tst_idx，不能宣称独立无泄漏泛化。既有全局预处理/RFE不重算，原GFR/CAM/SPGS/RES/分类头/损失和全部超参数保留。

```bash
python experiment/experiment17_disease_semantic_selective_alignment/run.py prepare
python experiment/experiment17_disease_semantic_selective_alignment/run.py all
python experiment/experiment17_disease_semantic_selective_alignment/run.py resume
python experiment/experiment17_disease_semantic_selective_alignment/run.py refresh
```

运行器以绝对目录定位，复制六个根文件至`/tmp/spromptgl17-state-<pathhash>/sources`，仅应用integration_patch.diff，CLIP/data用只读访问链接。不会import过去实验runner。临时源码清理后可按同一patch重建。调度器/任务独占锁、36个固定config×seed任务，两卡各18任务且每卡最多4并发；每worker两CPU线程。每25epoch及fold边界原子保存model/两个Adam/RNG/best-model/wait/已提交指标，带checksum；失败从完整checkpoint恢复，完成task跳过，任务完成文件签名隔离，汇总不混入未提交fold。完成后清理临时权重和源码，只保留日志、三份CSV、中文报告。

预检比较原始源码与D0两Adam步的权重/输出/RNG；3seeds×12初始化检查原路径与P0输出、参数、RNG相同；检查两步梯度、文本/QC/Skip/Private干预、训练标签隔离、站点辅助路径不变、10%界限、完整checkpoint第三步复演。仅为实现验证，不选模型或超参数。D0训练后的30fold与历史13/14/15/16六指标、初始化、14–16RNG与最佳epoch逐项比对；不同历史哈希格式都保留对应序列化。

D5每个最佳checkpoint做REAL/NEUTRAL/UNRELATED/SWAPPED/SKIP/NO_QC/NO_PRIVATE/RESIDUAL_OFF共240行。不重训、不选epoch，保持主干与注入前Token不变，最终恢复REAL；概率来自exp(log_softmax)。完整诊断每fold一行；Ridge probe仅用训练折拟合，作为训练后描述，不参与训练或选模。公开报告不保存大规模患者预测或永久权重。所有源码、原数据、CLIP、Prompt和历史文件SHA256以manifest为准。
'''
    table=['|组别|名称|对齐|文本|选择性|QC|Private融合|新增参数|','|---|---|---|---|---|---|---|---:|']
    for c,d in DESIGNS.items():
        table.append(f'|{c}|{d.name}|{d.alignment}|{d.prompt}|{d.selective}|{d.quality}|{d.fuse_private}|{m["additional_parameters"][c]}|')
    atomic(EXP/'README.md',text.replace('CONFIG_TABLE','\n'.join(table)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',nargs='?',choices=['prepare','preflight','all','resume','refresh','report','validate'])
    parser.add_argument('--worker',nargs=2,metavar=('CONFIG','SEED'))
    parser.add_argument('--workers-per-gpu',type=int,default=4)
    args=parser.parse_args()
    if args.worker:worker(args.worker[0],int(args.worker[1]));return
    if args.command=='prepare':prepare()
    elif args.command=='preflight':preflight()
    elif args.command in ['all','resume']:schedule(args.workers_per_gpu)
    elif args.command=='refresh':print(json.dumps({k:v for k,v in refresh().items() if k in ['run_status','completed_folds','completed_tasks']},ensure_ascii=False))
    elif args.command=='report':report()
    elif args.command=='validate':print(json.dumps(validate_complete(),ensure_ascii=False))
    else:parser.error('choose prepare/all/resume/refresh')


if __name__=='__main__':
    main()
