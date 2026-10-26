#!/usr/bin/env python3
"""E43.2: paired physical B0-B full GFR versus directly trained raw-input PLN."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
sys.dont_write_bytecode=True
import argparse,copy,csv,fcntl,hashlib,json,pickle,random,subprocess,time,traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path('/data/adodas/SPromptGL-main')
EXP=Path(__file__).resolve().parent
RT=EXP/'.runtime';RESULTS=EXP/'results'
HISTORY=ROOT/'experiment/experiment43_1_b0b_gfr_mechanism_retest'
CFG=json.loads((EXP/'config.json').read_text());PROTOCOL=CFG['protocol'];INPUT_DIMS=[48,6,10,256];GROUPS=CFG['groups']
CTX=None
def data():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    base = ROOT / "data/ABIDE"
    frame = pd.read_csv(base / "processed_standard_data.csv")
    mapping = np.load(base / "modal_feat_dict.npy", allow_pickle=True).item()
    assert list(mapping) == ["PHENO", "ANAT", "FUNC", "Correlation"]
    assert [len(v) for v in mapping.values()] == INPUT_DIMS
    assert list(frame.columns[:-1]) == [str(x) for xs in mapping.values() for x in xs]
    x = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = frame.iloc[:, -1].to_numpy(dtype=np.int64) - 1
    assert x.shape == (871, 320) and np.isfinite(x).all()
    assert np.bincount(y).tolist() == [403, 468]
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10).split(x, y))
    return x, y, folds

def seed_everything(torch, seed=0):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def digest_state(agent):
    h = hashlib.sha256()
    for prefix, module in (("gfr", agent.MessagePassing0), ("pln", agent.ModalFusion)):
        for name, p in sorted(module.named_parameters()):
            h.update((prefix + "." + name).encode() + b"\0")
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def make_pln_without_pen(torch, layers, dims, hp):
    import torch.nn as nn

    class PLNNoPEN(nn.Module):
        def __init__(self):
            super().__init__()
            self.hyperpm = hp
            self.input_data_dims = dims
            self.d_q = self.d_k = self.d_v = hp.n_hidden
            self.n_head = hp.n_head
            self.dropout = hp.dropout
            self.n_layer = hp.nlayer
            self.modal_num = hp.nmodal
            self.n_class = hp.nclass
            self.d_out = self.d_v * self.n_head * self.modal_num
            self.fusion_weight = nn.Parameter(torch.ones([6, 1]))
            self.InputLayer = layers.VariLengthInputLayer(dims, self.d_k, self.d_v, self.n_head, self.dropout)
            self.Encoder = []
            self.FeedForward = []
            for index in range(self.n_layer):
                encoder = layers.EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v,
                                             self.n_head, self.dropout)
                self.add_module("encode_%d" % index, encoder)
                self.Encoder.append(encoder)
                feedforward = layers.FeedForwardLayer(self.d_v * self.n_head,
                                                       self.d_v * self.n_head, dropout=self.dropout)
                self.add_module("feed_%d" % index, feedforward)
                self.FeedForward.append(feedforward)
            self.Outputlayer = layers.OutputLayer(self.d_v * self.n_head * self.modal_num,
                                                   self.d_v * self.n_head, self.n_class,
                                                   self.modal_num, self.dropout)

        def forward(self, x):
            bs = x.size(0)
            attn_map = []
            x, input_attention = self.InputLayer(x)
            attn = input_attention.mean(dim=1)
            attn_map.append(attn.detach().cpu().numpy())
            for index in range(self.n_layer):
                x, layer_attention = self.Encoder[index](q=x, k=x, v=x, modal_num=self.modal_num)
                attn = layer_attention.mean(dim=1)
                x = self.FeedForward[index](x)
                attn_map.append(attn.detach().cpu().numpy())
            output, hidden = self.Outputlayer(x.view(bs, -1), attn.view(bs, -1))
            return output, hidden, attn_map

    module = PLNNoPEN()
    assert not hasattr(module, "MLP")
    assert not hasattr(module, "prompt_text_features")
    assert not hasattr(module, "avg_pool")
    return module

def make_agent(torch, np, layers, network, utils, hp, x, y, train, test):
    from torch import optim

    class Agent:
        pass

    agent = Agent()
    agent.dev = torch.device("cuda")
    agent.hyperpm = hp
    agent.th, agent.GC_mode = hp.th, hp.GC_mode
    agent.MP_mode, agent.MF_mode = hp.MP_mode, hp.MF_mode
    agent.d_v, agent.modal_num, agent.n_class = hp.n_hidden, hp.nmodal, hp.nclass
    agent.dropout, agent.alpha, agent.n_head = hp.dropout, hp.alpha, hp.n_head
    agent.feat = torch.from_numpy(x).float().to(agent.dev)
    agent.targ = torch.from_numpy(y).long().to(agent.dev)
    agent.trn_idx = np.asarray(train)
    agent.tst_idx = agent.val_idx = np.asarray(test)
    trn_label = agent.targ[agent.trn_idx].cpu().numpy()
    counter = Counter(trn_label)
    weights = len(trn_label) / np.asarray(list(counter.values())) / agent.n_class
    agent.weight = torch.from_numpy(weights).float().to(agent.dev)
    agent.dataLayer = layers.InitialDataLayer(INPUT_DIMS)
    agent.MessagePassing0 = network.GFR(INPUT_DIMS, INPUT_DIMS, th=agent.th, mode=agent.GC_mode).to(agent.dev)
    agent.ModalFusion = make_pln_without_pen(torch, layers, INPUT_DIMS, hp).to(agent.dev)
    assert sum(p.numel() for p in agent.ModalFusion.parameters()) == 49544
    assert not any(name.startswith("MLP.") for name, _ in agent.ModalFusion.named_parameters())
    agent.optimizer_MP0 = optim.Adam(agent.MessagePassing0.parameters(), lr=hp.lr, weight_decay=hp.reg)
    agent.optimizer_MF = optim.Adam(agent.ModalFusion.parameters(), lr=hp.lr, weight_decay=hp.reg)
    agent.ModalFusion.apply(utils.my_weight_init)
    opt_ids = [id(p) for opt in (agent.optimizer_MP0, agent.optimizer_MF)
               for group in opt.param_groups for p in group["params"]]
    model_ids = [id(p) for module in (agent.MessagePassing0, agent.ModalFusion) for p in module.parameters()]
    assert len(opt_ids) == len(set(opt_ids)) and set(opt_ids) == set(model_ids)
    assert np.array_equal(agent.val_idx, agent.tst_idx)
    return agent

def metric_row(output, y, test):
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_auc_score

    prediction = output.detach().cpu().numpy()[test].argmax(1)
    target = y[test]
    tp, fn, fp, tn = map(int, confusion_matrix(target, prediction, labels=[0, 1]).ravel())
    hard_auc = float(roc_auc_score(np.eye(2)[target], np.eye(2)[prediction]))
    return dict(acc=float((prediction == target).mean()), hard_auc=hard_auc,
                sen=float(tp / (tp + fn)), spe=float(tn / (tn + fp)),
                asd_tp=tp, asd_fn=fn, td_fp=fp, td_tn=tn)

def write(path,value):
    p=Path(path);t=p.with_name(p.name+f'.tmp.{os.getpid()}')
    with t.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)

def save(path,value):
    import torch
    p=Path(path);t=p.with_name(p.name+f'.tmp.{os.getpid()}')
    with t.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(t,p)

def load(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=False)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text())

def csvread(path):
    if not Path(path).exists():return []
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def csvwrite(path,rows):
    p=Path(path);t=p.with_name(p.name+f'.tmp.{os.getpid()}');fields=list(dict.fromkeys(k for r in rows for k in r)) or ['group','seed','fold']
    with t.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(t,p)

def rng():
    t=runtime()['torch']
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=t.get_rng_state(),cuda=t.cuda.get_rng_state_all())

def restore(r):
    t=runtime()['torch'];random.setstate(r['python']);np.random.set_state(r['numpy']);t.set_rng_state(r['cpu'].cpu());t.cuda.set_rng_state_all([v.cpu() for v in r['cuda']])

def rngsha(r):
    h=hashlib.sha256(pickle.dumps((r['python'],r['numpy']),protocol=4))
    for x in [r['cpu']]+r['cuda']:h.update(x.cpu().numpy().tobytes())
    return h.hexdigest()

def legacy_rngsha(r):return hashlib.sha256(r['cpu'].numpy().tobytes()+r['cuda'][0].numpy().tobytes()).hexdigest()

def model(a):return {k:{n:p.detach().cpu().clone() for n,p in m.state_dict().items()} for k,m in [('gfr',a.MessagePassing0),('pln',a.ModalFusion)]}

def setmodel(a,s):a.MessagePassing0.load_state_dict(s['gfr']);a.ModalFusion.load_state_dict(s['pln'])

def equal(a,b):
    t=runtime()['torch']
    if isinstance(a,t.Tensor):return t.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def original_step(agent, torch, utils):
    agent.MessagePassing0.train()
    agent.ModalFusion.train()
    agent.optimizer_MP0.zero_grad()
    agent.optimizer_MF.zero_grad()
    model_num, parts = agent.dataLayer(agent.feat)
    features, adjs, alignment_loss = agent.MessagePassing0(parts, model_num, agent.targ, agent.trn_idx)
    graph_loss = sum(utils.GraphConstructLoss(parts[i], adjs[i], agent.hyperpm.theta_smooth,
                                               agent.hyperpm.theta_degree, agent.hyperpm.theta_sparsity)
                     for i in range(model_num)) / 4
    logp, _, _ = agent.ModalFusion(features)
    cls_loss = utils.ClsLoss(logp, agent.targ, agent.trn_idx, agent.weight)
    loss = cls_loss + graph_loss
    loss.backward()
    agent.optimizer_MP0.step()
    agent.optimizer_MF.step()
    return float(cls_loss.detach()), float(graph_loss.detach()), float(alignment_loss.detach())

def runtime():
    global CTX
    if CTX is None:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        torch.cuda.set_device(0);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        sys.path.insert(0,str(RT/'sources'))
        import layers,network,utils
        assert not hasattr(network,'PLN') and not any(n=='CLIP' or n.startswith('CLIP.') for n in sys.modules)
        CTX=dict(torch=torch,layers=layers,network=network,utils=utils)
    return CTX

def statesha(value):
    t=runtime()['torch'];h=hashlib.sha256()
    def add(v):
        if isinstance(v,t.Tensor):
            a=v.detach().cpu().contiguous().numpy();h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
        elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=repr):h.update(repr(k).encode());add(v[k])
        elif isinstance(v,(tuple,list)):
            for x in v:add(x)
        else:h.update(repr(v).encode())
    add(value);return h.hexdigest()

def signature():
    paths=[EXP/n for n in ['run.py','integration.py','config.json']]+list((RT/'sources').glob('*.py'))
    return hashlib.sha256(json.dumps({str(p.relative_to(EXP)):sha(p) for p in paths},sort_keys=True).encode()).hexdigest()

def foldsha(fold):
    tr,te=data()[2][fold-1]
    return hashlib.sha256(tr.astype(np.int64).tobytes()+te.astype(np.int64).tobytes()).hexdigest()

def history():
    a={int(r['fold']):r for r in csvread(HISTORY/'results/fold_metrics.csv') if int(r['group_id'])==0}
    b={int(r['fold']):r for r in csvread(ROOT/'experiment/b0_b_seed0_full10fold/results/fold_metrics.csv')}
    assert len(a)==len(b)==10
    return a,b

def fresh(fold):
    c=runtime();x,y,folds=data();tr,te=folds[fold-1]
    return make_agent(c['torch'],np,c['layers'],c['network'],c['utils'],SimpleNamespace(**PROTOCOL),x,y,tr,te)

def fromanchor(group,fold):
    anchor=load(RT/'anchors'/f'fold{fold:02}.pt');ref,_=history();r=ref[fold]
    assert anchor['signature']==CFG['baseline_source_signature']
    assert anchor['fold']==fold and anchor['split_sha']==foldsha(fold)==r['split_sha']
    assert sha(RT/'anchors'/f'fold{fold:02}.pt')==r['anchor_sha']
    assert legacy_rngsha(anchor['entry_rng'])==r['initial_rng_sha256']
    assert rngsha(anchor['entry_rng'])==r['entry_full_rng_sha256']
    assert rngsha(anchor['rng'])==r['post_initial_rng_sha256']
    restore(anchor['entry_rng']);a=fresh(fold)
    assert sum(p.numel() for p in a.MessagePassing0.parameters())==887836
    assert sum(p.numel() for p in a.ModalFusion.parameters())==49544
    assert digest_state(a)==anchor['parameter_sha']==r['initial_parameter_sha256']
    assert equal(model(a),anchor['model']) and equal(rng(),anchor['rng'])
    assert equal(a.optimizer_MP0.state_dict(),anchor['mp']) and equal(a.optimizer_MF.state_dict(),anchor['mf'])
    assert not anchor['mp']['state'] and not anchor['mf']['state']
    a.optimizer_MP0.load_state_dict(copy.deepcopy(anchor['mp']));a.optimizer_MF.load_state_dict(copy.deepcopy(anchor['mf']))
    restore(anchor['rng'])
    assert runtime()['torch'].equal(runtime()['torch'].cat(a.dataLayer(a.feat)[1],dim=1),a.feat)
    a.group=group;a.gfr_calls=0;a.graph_loss_calls=0;a.gcn_calls=0
    def count_gfr(module,args):
        a.gfr_calls+=1
        assert a.group==0,'RAW group called GFR'
    def count_gcn(module,args):
        a.gcn_calls+=1
        assert a.group==0,'RAW group called GCN'
    a.MessagePassing0.register_forward_pre_hook(count_gfr)
    for i in range(4):getattr(a.MessagePassing0,f'MessagePassing{i}').register_forward_pre_hook(count_gcn)
    if group==1:a.MessagePassing0.requires_grad_(False)
    a.initial_gfr_sha=statesha(a.MessagePassing0.state_dict());a.initial_mp_sha=statesha(a.optimizer_MP0.state_dict())
    return a,anchor

def forward(a):
    c=runtime();t=c['torch']
    if a.group==0:
        n,parts=a.dataLayer(a.feat);feat,adjs,aloss=a.MessagePassing0(parts,n,a.targ,a.trn_idx)
        gl=sum(c['utils'].GraphConstructLoss(parts[i],adjs[i],PROTOCOL['theta_smooth'],PROTOCOL['theta_degree'],PROTOCOL['theta_sparsity']) for i in range(4))/4
        a.graph_loss_calls+=4
    else:
        feat=a.feat;gl=feat.new_zeros(());adjs=[];aloss=None
    out=a.ModalFusion(feat)[0];cl=c['utils'].ClsLoss(out,a.targ,a.trn_idx,a.weight)
    return out,cl,gl,adjs,aloss

def evaluate(a):
    a.MessagePassing0.eval();a.ModalFusion.eval()
    with runtime()['torch'].no_grad():return forward(a)

def grads(a):
    return {k:{n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in m.named_parameters()} for k,m in [('gfr',a.MessagePassing0),('pln',a.ModalFusion)]}

def trainstep(a):
    t=runtime()['torch'];a.ModalFusion.train();a.MessagePassing0.train(a.group==0)
    a.optimizer_MP0.zero_grad();a.optimizer_MF.zero_grad()
    out,cl,gl,_,_=forward(a);loss=cl+gl if a.group==0 else cl
    assert t.isfinite(out).all() and t.isfinite(loss)
    loss.backward()
    assert all(t.isfinite(p.grad).all() for m in [a.MessagePassing0,a.ModalFusion] for p in m.parameters() if p.grad is not None)
    if a.group==0:a.optimizer_MP0.step()
    else:
        assert a.gfr_calls==a.gcn_calls==a.graph_loss_calls==0
        assert all(p.grad is None for p in a.MessagePassing0.parameters())
    a.optimizer_MF.step()
    return dict(cls=float(cl.detach()),graph=float(gl.detach()),total=float(loss.detach()))

def history_differences(row):
    result={}
    for label,source in zip(['E43.1','historical_B0B'],history()):
        ref=source[row['fold']];diff={}
        for k in ['acc','hard_auc','spe','sen','asd_tp','asd_fn','td_fp','td_tn','best_epoch','epochs_executed','initial_parameter_sha256','initial_rng_sha256']:
            bad=(abs(float(ref[k])-float(row[k]))>1e-12 if k not in ['initial_parameter_sha256','initial_rng_sha256'] else ref[k]!=row[k])
            if bad:diff[k]=[ref[k],row[k]]
        if label=='E43.1':
            for k in ['prediction_sha256','entry_full_rng_sha256','post_initial_rng_sha256']:
                if row[k]!=ref[k]:diff[k]=[ref[k],row[k]]
        result[label]=diff
    return result

def verify_sources():
    for p,h in read(EXP/'manifest.json')['protected_sha256'].items():assert sha(p)==h,'Protected file modified: '+p

def prepare():
    from integration import prepare as generate
    generate(ROOT,EXP)
    old=read(HISTORY/'manifest.json')
    paths=['run.py','method.py','integration.py','integration.patch','config.json']
    historical_signature=hashlib.sha256(json.dumps({n:sha(HISTORY/n) for n in paths},sort_keys=True).encode()).hexdigest()
    assert historical_signature==CFG['baseline_source_signature']==old['code_signature']
    for n in ['network.py','layers.py','utils.py']:
        assert sha(RT/'sources'/n)==sha(HISTORY/'.runtime/sources'/n)
    index=read(RT/'anchors/source_index.json')['anchors'];refs,_=history()
    for z in index:
        f=z['fold'];assert sha(RT/'anchors'/f'fold{f:02}.pt')==z['anchor_sha256']==refs[f]['anchor_sha']
        assert foldsha(f)==z['split_sha256']==refs[f]['split_sha']
        for p,h in z['inputs'].items():assert sha(p)==h
    verify_sources();m=read(EXP/'manifest.json');m.update(status='PREPARED',code_signature=signature(),historical_anchor_verification='PASS_10_FILE_SHA_SPLIT_SOURCE',isolated_source_sha256={n:sha(RT/'sources'/n) for n in ['network.py','layers.py','utils.py']});write(EXP/'manifest.json',m)
    print('PREPARE_PASS: source historical signature and ten anchors verified',flush=True)

def preflight(gpu):
    t=runtime()['torch'];folds=CFG['gpu_folds'][str(gpu)];checks={};started=time.monotonic()
    for f in folds:
        a,aa=fromanchor(0,f);b,bb=fromanchor(1,f)
        assert equal(model(a),model(b)) and equal(aa['rng'],bb['rng'])
        assert equal(a.optimizer_MP0.state_dict(),b.optimizer_MP0.state_dict()) and equal(a.optimizer_MF.state_dict(),b.optimizer_MF.state_dict())
        checks[f'fold{f:02}_initial_pair']=True
    f=folds[0];a,anchor=fromanchor(0,f);post=copy.deepcopy(anchor['rng']);ref=fresh(f);setmodel(ref,anchor['model'])
    restore(post);original_step(ref,t,runtime()['utils']);expected=model(ref);expected_grads=grads(ref);ops=copy.deepcopy([ref.optimizer_MP0.state_dict(),ref.optimizer_MF.state_dict()]);rr=copy.deepcopy(rng())
    restore(post);trainstep(a)
    assert equal(expected,model(a)) and equal(expected_grads,grads(a)) and equal(ops,[a.optimizer_MP0.state_dict(),a.optimizer_MF.state_dict()]) and equal(rr,rng())
    checks['E00_original_step_gradients_both_adams_rng_bitwise']=True
    b,_=fromanchor(1,f);before=model(b);op=copy.deepcopy(b.optimizer_MP0.state_dict());restore(post)
    old=runtime()['utils'].GraphConstructLoss
    def forbidden(*args,**kwargs):raise AssertionError('RAW group called graph regularizer')
    runtime()['utils'].GraphConstructLoss=forbidden
    try:trainstep(b)
    finally:runtime()['utils'].GraphConstructLoss=old
    assert equal(before['gfr'],model(b)['gfr']) and equal(op,b.optimizer_MP0.state_dict())
    assert not equal(before['pln'],model(b)['pln']) and all(p.grad is None for p in b.MessagePassing0.parameters())
    assert equal(rr,rng())
    checks['E01_no_GFR_no_graphloss_no_update_equal_dropout_rng']=True
    for group in (0,1):
        z,_=fromanchor(group,f);out=evaluate(z)[0];target=z.targ.clone();z.targ[z.tst_idx]=1-z.targ[z.tst_idx];assert t.equal(out,evaluate(z)[0]);z.targ=target
        assert all(t.isfinite(p).all() for m in (z.MessagePassing0,z.ModalFusion) for p in m.parameters())
        checks[f'g{group}_heldout_label_permutation_output_bitwise']=True
        trainstep(z);trainstep(z);tmp=RT/'checkpoints'/f'preflight_gpu{gpu}_g{group}.pt'
        save(tmp,dict(model=model(z),mp=z.optimizer_MP0.state_dict(),mf=z.optimizer_MF.state_dict(),rng=rng()))
        trainstep(z);expected=model(z);ops=copy.deepcopy([z.optimizer_MP0.state_dict(),z.optimizer_MF.state_dict()]);rr=copy.deepcopy(rng())
        q,_=fromanchor(group,f);ss=load(tmp);setmodel(q,ss['model']);q.optimizer_MP0.load_state_dict(ss['mp']);q.optimizer_MF.load_state_dict(ss['mf']);restore(ss['rng']);trainstep(q)
        assert equal(expected,model(q)) and equal(ops,[q.optimizer_MP0.state_dict(),q.optimizer_MF.state_dict()]) and equal(rr,rng());tmp.unlink()
        checks[f'g{group}_disk_resume_two_then_third_exact']=True
    checks.update(status='PASS',signature=signature(),physical_gpu=gpu,planned_folds=10,model_parameters=dict(GFR=887836,PLN=49544),hardware=str(t.cuda.get_device_properties(0)),peak_allocated_bytes=t.cuda.max_memory_allocated(),seconds=time.monotonic()-started)
    write(RT/f'preflight_gpu{gpu}.json',checks);print('PREFLIGHT_PASS',json.dumps(checks),flush=True)

def completed(group,fold):
    p=RT/'tasks'/f'g{group}_f{fold:02}.json'
    if not p.exists():return False
    r=read(p)
    assert r['signature']==signature() and r['split_sha']==foldsha(fold)
    if r['status']!='COMPLETE':return False
    assert r['anchor_sha']==sha(RT/'anchors'/f'fold{fold:02}.pt')
    assert r['best_sha']==sha(RT/'checkpoints'/f'g{group}_f{fold:02}.best.pt')
    return True

def execute_fold(group,fold,gpu):
    t=runtime()['torch'];key=f'g{group}_f{fold:02}';task=RT/'tasks'/f'{key}.json'
    lock=(RT/'tasks'/f'{key}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if completed(group,fold):print('SKIP_VERIFIED',key,flush=True);return
    assert fold in CFG['gpu_folds'][str(gpu)]
    a,anchor=fromanchor(group,fold);started=time.monotonic();latest=RT/'checkpoints'/f'{key}.latest.pt';bestpath=RT/'checkpoints'/f'{key}.best.pt'
    base=dict(group=GROUPS[group],group_id=group,seed=0,fold=fold,physical_gpu=gpu,signature=signature(),split_sha=foldsha(fold),anchor_sha=sha(RT/'anchors'/f'fold{fold:02}.pt'),initial_parameter_sha256=digest_state(a),initial_rng_sha256=legacy_rngsha(anchor['entry_rng']),entry_full_rng_sha256=rngsha(anchor['entry_rng']),post_initial_rng_sha256=rngsha(anchor['rng']),initial_pln_sha=statesha(anchor['model']['pln']),initial_gfr_sha=a.initial_gfr_sha,initial_mp_sha=a.initial_mp_sha)
    begin=0;bestacc=0.;bestepoch=-1;wait=0;best=None;elapsed=0.;rng_trace=[];loss_trace=[]
    if latest.exists():
        s=load(latest);assert s['signature']==signature() and s['split_sha']==base['split_sha'] and s['anchor_sha']==base['anchor_sha']
        setmodel(a,s['model']);a.optimizer_MP0.load_state_dict(s['mp']);a.optimizer_MF.load_state_dict(s['mf']);restore(s['rng']);begin=s['epoch']+1;bestacc=s['bestacc'];bestepoch=s['bestepoch'];wait=s['wait'];best=s['best'];elapsed=s['elapsed'];rng_trace=s['rng_trace'];loss_trace=s['loss_trace'];a.gfr_calls=s['gfr_calls'];a.gcn_calls=s['gcn_calls'];a.graph_loss_calls=s['graph_loss_calls']
        print('RESUME',key,begin,flush=True)
    write(task,dict(**base,status='RUNNING',epoch=begin));epoch=begin-1
    for epoch in (range(begin,PROTOCOL['nepoch']) if wait<=PROTOCOL['early'] else []):
        values=trainstep(a);out=evaluate(a)[0];acc=float(out[a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean().item())
        if acc>bestacc:
            bestacc=acc;bestepoch=epoch;wait=0
            best=dict(model=model(a),mp=copy.deepcopy(a.optimizer_MP0.state_dict()),mf=copy.deepcopy(a.optimizer_MF.state_dict()),rng=copy.deepcopy(rng()),epoch=epoch,bestacc=acc)
        else:wait+=1
        rng_trace.append(rngsha(rng()))
        if epoch%10==0 or epoch==449:loss_trace.append(dict(epoch=epoch,**values))
        if (epoch+1)%CFG['checkpoint_every']==0 or wait>PROTOCOL['early'] or epoch==449:
            save(latest,dict(**base,model=model(a),mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),rng=rng(),epoch=epoch,bestacc=bestacc,bestepoch=bestepoch,wait=wait,best=best,elapsed=elapsed+time.monotonic()-started,rng_trace=rng_trace,loss_trace=loss_trace,gfr_calls=a.gfr_calls,gcn_calls=a.gcn_calls,graph_loss_calls=a.graph_loss_calls))
            write(task,dict(**base,status='RUNNING',epoch=epoch,best_epoch=bestepoch))
        if wait>PROTOCOL['early']:break
    assert best is not None
    terminal_rng=copy.deepcopy(rng());setmodel(a,best['model']);out=evaluate(a)[0];metrics=metric_row(out,data()[1],a.tst_idx)
    prediction=out[a.tst_idx].argmax(1).detach().cpu().numpy().astype(np.int64)
    if group==1:
        assert statesha(a.MessagePassing0.state_dict())==a.initial_gfr_sha and statesha(a.optimizer_MP0.state_dict())==a.initial_mp_sha
        assert a.gfr_calls==a.gcn_calls==a.graph_loss_calls==0 and all(p.grad is None for p in a.MessagePassing0.parameters())
    artifact=dict(**base,**best,prediction=prediction,test_indices=a.tst_idx,best_epoch=bestepoch,epochs_executed=epoch+1,metrics=metrics,wait=0)
    save(bestpath,artifact);saved=load(bestpath);assert equal(saved['model'],best['model']) and equal(saved['prediction'],prediction) and saved['metrics']==metrics
    row=dict(**base,**metrics,best_epoch=bestepoch,epochs_executed=epoch+1,prediction_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),model_sha256=statesha(best['model']),best_sha=sha(bestpath),wall_seconds=elapsed+time.monotonic()-started,status='COMPLETE',gfr_calls=a.gfr_calls,gcn_calls=a.gcn_calls,graph_loss_calls=a.graph_loss_calls,final_gfr_sha=statesha(a.MessagePassing0.state_dict()),final_mp_sha=statesha(a.optimizer_MP0.state_dict()),rng_trace=rng_trace,loss_trace=loss_trace,terminal_rng_sha=rngsha(terminal_rng))
    if group==0:
        row['historical_differences']=history_differences(row)
        if any(row['historical_differences'].values()):write(task,dict(**row,status='ANCHOR_REPLAY_FAIL'));raise AssertionError(row['historical_differences'])
    else:
        br=read(RT/'tasks'/f'g0_f{fold:02}.json');common=min(len(br['rng_trace']),len(rng_trace))
        assert br['rng_trace'][:common]==rng_trace[:common],'Paired dropout RNG divergence'
        row['rng_common_epochs_match']=common
    write(task,row);latest.unlink(missing_ok=True)
    print('FOLD_DONE',key,'gpu',gpu,'ACC',metrics['acc'],'best_epoch',bestepoch,'epochs',epoch+1,'seconds',round(row['wall_seconds'],2),flush=True)

def refresh():
    rows=[]
    for p in sorted((RT/'tasks').glob('g*_f*.json')):
        r=read(p)
        if r['status']=='COMPLETE':
            assert completed(r['group_id'],r['fold']);rows.append(r)
    assert len({(r['group_id'],r['seed'],r['fold']) for r in rows})==len(rows)
    rows.sort(key=lambda r:(r['group_id'],r['fold']))
    csvwrite(RESULTS/'fold_metrics.csv',[{k:v for k,v in r.items() if k not in ['rng_trace','loss_trace','historical_differences']} for r in rows])
    return rows

def launch(command,gpu,label):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));p=RT/'logs'/f'{label}.log';log=p.open('a',buffering=1)
    child=subprocess.Popen([sys.executable,'-B','-u',str(EXP/'run.py'),*command,'--gpu',str(gpu)],cwd=EXP,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return child,log,label

def dispatch(group):
    live=[launch(['worker','--group',str(group)],gpu,f'g{group}_gpu{gpu}') for gpu in (0,1)]
    while live:
        time.sleep(3)
        for item in live[:]:
            p,log,label=item
            if p.poll() is not None:
                log.close();live.remove(item)
                if p.returncode:
                    for q,_,_ in live:q.terminate()
                    raise RuntimeError('Worker failed '+label+'; see .runtime/logs/'+label+'.log')
        refresh()

def controller():
    lock=(RT/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for gpu in (0,1):
        pre=read(RT/f'preflight_gpu{gpu}.json');assert pre['status']=='PASS' and pre['signature']==signature()
    verify_sources();m=read(EXP/'manifest.json');m.update(status='E00_RUNNING',formal_started_unix=m.get('formal_started_unix',time.time()));write(EXP/'manifest.json',m)
    dispatch(0);rows=refresh();base=[r for r in rows if r['group_id']==0];assert len(base)==10 and all(not any(r['historical_differences'].values()) for r in base)
    m=read(EXP/'manifest.json');m.update(status='ANCHOR_REPLAY_PASS',baseline_replay='PASS_10_OF_10',groups_complete=1,folds_complete=len(rows));write(EXP/'manifest.json',m)
    print('ANCHOR_REPLAY_PASS 10/10; starting paired RAW_INPUT_NO_GFR',flush=True)
    dispatch(1);rows=refresh();assert len(rows)==20
    m=read(EXP/'manifest.json');m.update(status='TRAINING_COMPLETE_AUDIT_PENDING',groups_complete=2,folds_complete=20,formal_finished_unix=time.time());m['formal_wall_seconds']=m['formal_finished_unix']-m['formal_started_unix'];write(EXP/'manifest.json',m)
    print('TRAINING_COMPLETE 20/20; independent audit required',flush=True)

def diagnostics(a,fold):
    t=runtime()['torch'];group=a.group;before=statesha(model(a));ops=statesha([a.optimizer_MP0.state_dict(),a.optimizer_MF.state_dict()])
    if group==1:
        out=evaluate(a)[0]
        assert statesha(a.MessagePassing0.state_dict())==a.initial_gfr_sha
        assert statesha(a.optimizer_MP0.state_dict())==a.initial_mp_sha
        assert a.gfr_calls==a.graph_loss_calls==a.gcn_calls==0
        return [dict(group=GROUPS[group],fold=fold,modality='RAW_INPUT',input_norm=float(a.feat.norm()),pln_input_equals_raw=True,gfr_calls=0,gcn_calls=0,graph_loss_calls=0,gfr_gradient_count=sum(p.grad is not None for p in a.MessagePassing0.parameters()),gfr_parameter_unchanged=True,gfr_optimizer_unchanged=True)]
    g=a.MessagePassing0;capture={};disabled=set();handles=[]
    for i in range(4):
        def hook(module,inputs,output,i=i):
            capture[i]=output.detach().clone()
            return t.zeros_like(output) if i in disabled else output
        handles.append(getattr(g,f'MessagePassing{i}').register_forward_hook(hook))
    handles.append(g.register_forward_hook(lambda module,inputs,output:capture.update(features=output[0].detach().clone())))
    try:
        out,cl,gl,adjs,_=evaluate(a);messages=[capture[i].clone() for i in range(4)];features=capture['features'].clone()
        composed=t.sigmoid(g.fusion_weight[1])*messages[1]+a.feat
        for i in (2,3):composed=t.sigmoid(g.fusion_weight[i])*messages[i]+composed
        assert t.equal(features,composed),'Original overwrite arithmetic changed'
        a.ModalFusion.eval();g.eval();_,cls,graph,_,_=forward(a)
        ps=[p for i in range(4) for p in getattr(g,f'MessagePassing{i}').parameters()]
        clsgr=t.autograd.grad(cls,ps,retain_graph=True,allow_unused=True)
        graphps=[p for i in range(4) for p in getattr(g,f'GraphConstruct{i}').p.parameters()]
        graphgr=t.autograd.grad(graph,graphps,allow_unused=True)
        norms=lambda gs:float(sum((z.square().sum() for z in gs if z is not None),a.feat.new_zeros(())).sqrt())
        rows=[]
        for i,name in enumerate(['PHENO','ANAT','FUNC','CORR']):
            disabled.add(i);off=evaluate(a)[0];disabled.clear();fw=float(t.sigmoid(g.fusion_weight[i]));mask=adjs[i]>0;mask.fill_diagonal_(False)
            rows.append(dict(group=GROUPS[group],fold=fold,modality=name,input_norm=float(a.feat.norm()),gfr_output_norm=float(features.norm()),gfr_minus_raw_norm=float((features-a.feat).norm()),gfr_minus_raw_relative=float((features-a.feat).norm()/a.feat.norm()),message_norm=float(messages[i].norm()),fusion_sigmoid=fw,effective_contribution_norm=fw*float(messages[i].norm()) if i else 0.,effective_contribution_input_ratio=fw*float(messages[i].norm()/a.feat.norm()) if i else 0.,classification_branch_gradient_norm=norms(clsgr[2*i:2*i+2]),graph_projection_gradient_norm=norms(graphgr[2*i:2*i+2]),branch_off_logp_mean=float((out[a.tst_idx]-off[a.tst_idx]).abs().mean()),branch_off_flip_count=int((out[a.tst_idx].argmax(1)!=off[a.tst_idx].argmax(1)).sum()),directed_nonself_edges=int(mask.sum()),mean_neighbors=float(mask.sum(1).float().mean()),pheno_overwrite_exact=True))
        disabled.update(range(4));off=evaluate(a)[0];disabled.clear()
        for row in rows:row.update(joint_off_logp_mean=float((out[a.tst_idx]-off[a.tst_idx]).abs().mean()),joint_off_flip_count=int((out[a.tst_idx].argmax(1)!=off[a.tst_idx].argmax(1)).sum()))
        assert rows[0]['classification_branch_gradient_norm']==0 and rows[0]['branch_off_flip_count']==0 and rows[0]['branch_off_logp_mean']==0
    finally:
        for h in handles:h.remove()
    assert before==statesha(model(a)) and ops==statesha([a.optimizer_MP0.state_dict(),a.optimizer_MF.state_dict()])
    return rows

def audit_worker(gpu):
    from sklearn.metrics import confusion_matrix,roc_auc_score
    t=runtime()['torch'];checks=[];diags=[]
    for fold in CFG['gpu_folds'][str(gpu)]:
        pairs=[]
        for group in (0,1):
            assert completed(group,fold)
            row=read(RT/'tasks'/f'g{group}_f{fold:02}.json');p=RT/'checkpoints'/f'g{group}_f{fold:02}.best.pt';s=load(p)
            a,anchor=fromanchor(group,fold);setmodel(a,s['model']);out=evaluate(a)[0]
            assert t.isfinite(out).all() and all(t.isfinite(p).all() for module in [a.MessagePassing0,a.ModalFusion] for p in module.parameters())
            prediction=out[a.tst_idx].argmax(1).cpu().numpy().astype(np.int64);target=data()[1][a.tst_idx]
            tp,fn,fp,tn=map(int,confusion_matrix(target,prediction,labels=[0,1]).ravel())
            independently_recomputed=dict(acc=float((prediction==target).mean()),hard_auc=float(roc_auc_score(np.eye(2)[target],np.eye(2)[prediction])),spe=float(tn/(tn+fp)),sen=float(tp/(tp+fn)),asd_tp=tp,asd_fn=fn,td_fp=fp,td_tn=tn)
            assert independently_recomputed==s['metrics']
            assert all(row[k]==v for k,v in independently_recomputed.items())
            assert equal(s['prediction'],prediction) and equal(s['test_indices'],a.tst_idx)
            assert hashlib.sha256(prediction.tobytes()).hexdigest()==row['prediction_sha256']
            assert statesha(s['model'])==row['model_sha256']
            original=a.targ.clone();a.targ[a.tst_idx]=1-a.targ[a.tst_idx];assert t.equal(out,evaluate(a)[0]);a.targ=original
            if group==0:
                assert not any(history_differences(row).values())
                hist=load(HISTORY/'.runtime/checkpoints'/f'g0_f{fold:02}.best.pt')
                assert equal(s['model'],hist['model']),'Replayed best weights differ from E43.1'
            else:
                assert row['initial_gfr_sha']==row['final_gfr_sha']==statesha(s['model']['gfr'])
                assert row['initial_mp_sha']==row['final_mp_sha']==statesha(s['mp'])
                assert row['gfr_calls']==row['gcn_calls']==row['graph_loss_calls']==0
            diags.extend(diagnostics(a,fold));checks.append(dict(group=GROUPS[group],fold=fold,status='PASS',best_sha=sha(p),prediction_sha=row['prediction_sha256'],heldout_label_output_bitwise=True));pairs.append(row)
            del a;t.cuda.empty_cache()
        for k in ['split_sha','anchor_sha','initial_parameter_sha256','entry_full_rng_sha256','post_initial_rng_sha256','initial_pln_sha']:
            assert pairs[0][k]==pairs[1][k]
        common=min(len(r['rng_trace']) for r in pairs)
        assert pairs[0]['rng_trace'][:common]==pairs[1]['rng_trace'][:common]
    write(RT/f'final_audit_gpu{gpu}.json',dict(status='PASS',signature=signature(),checks=checks,diagnostics=diags,physical_gpu=gpu,peak_allocated_bytes=t.cuda.max_memory_allocated()))
    print('INDEPENDENT_READBACK_PASS',len(checks),'gpu',gpu,flush=True)

def audit_controller():
    live=[launch(['audit-worker'],g,f'final_audit_gpu{g}') for g in (0,1)]
    while live:
        time.sleep(2)
        for item in live[:]:
            p,log,label=item
            if p.poll() is not None:
                log.close();live.remove(item)
                if p.returncode:
                    for q,_,_ in live:q.terminate()
                    raise RuntimeError('Independent audit failed '+label)
    verify_sources();diags=[]
    for g in (0,1):
        a=read(RT/f'final_audit_gpu{g}.json');assert a['status']=='PASS' and a['signature']==signature() and len(a['checks'])==10;diags.extend(a['diagnostics'])
    csvwrite(RESULTS/'diagnostics.csv',diags)
    m=read(EXP/'manifest.json');m.update(independent_readback='PASS_20_OF_20',baseline_best_weights_exact='PASS_10_OF_10_E43.1',protected_files_unchanged=True,paired_epoch_rng='PASS_EVERY_COMMON_EPOCH',status='INDEPENDENT_AUDIT_PASS');write(EXP/'manifest.json',m)

def append_handoff(text):
    p=ROOT/'handoff.md';marker='<!-- EXPERIMENT43_2_GFR_INCREMENTAL_VALUE -->';pending=EXP/'handoff_append_pending.md';m=read(EXP/'manifest.json')
    try:
        with p.open('r+b') as f:
            fcntl.flock(f,fcntl.LOCK_EX);before=f.read();initial=m['handoff_prefix'];assert hashlib.sha256(before[:initial['bytes']]).hexdigest()==initial['sha256']
            h=hashlib.sha256(before).hexdigest()
            if marker.encode() in before:return dict(status='ALREADY_APPENDED',prefix_sha256=h)
            f.seek(0,2);f.write(text.encode());f.flush();os.fsync(f.fileno());f.seek(0);after=f.read();assert after[:len(before)]==before and hashlib.sha256(after[:len(before)]).hexdigest()==h
        pending.unlink(missing_ok=True);return dict(status='APPENDED',prefix_bytes=len(before),prefix_sha256=h,after_sha256=hashlib.sha256(after).hexdigest())
    except Exception as e:
        pending.write_text(text+'\nAppend failure: '+repr(e)+'\n');return dict(status='PENDING',error=repr(e))

def report(final=True):
    rows=refresh();m=read(EXP/'manifest.json')
    if final:assert len(rows)==20 and m.get('independent_readback')=='PASS_20_OF_20'
    summaries=[];by={g:{r['fold']:r for r in rows if r['group_id']==g} for g in (0,1)}
    lines=['# Experiment 43.2 — GFR增量价值审计','',f'正式完成：{len(rows)}/20折。仅seed0；四指标为十折mean ± population SD，单位%。','', '|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|','|---|---:|---:|---:|---:|']
    for g,name in enumerate(GROUPS):
        rr=list(by[g].values())
        if len(rr)!=10:continue
        stat={}
        for k in CFG['metrics']:
            v=100*np.array([r[k] for r in rr]);stat[k+'_mean']=float(v.mean());stat[k+'_sd']=float(v.std(ddof=0))
        summaries.append(dict(group=name,folds=10,**stat));lines.append('|'+name+'|'+'|'.join(f"{stat[k+'_mean']:.4f} ± {stat[k+'_sd']:.4f}" for k in CFG['metrics'])+'|')
    if len(summaries)!=2:
        (RESULTS/'report.md').write_text('\n'.join(lines)+'\n');return
    diffs={k:[100*(by[0][f][k]-by[1][f][k]) for f in range(1,11)] for k in CFG['metrics']};paired={k:float(np.mean(v)) for k,v in diffs.items()}
    wtl={k:[sum(v>1e-10 for v in diffs[k]),sum(abs(v)<=1e-10 for v in diffs[k]),sum(v< -1e-10 for v in diffs[k])] for k in ['acc','hard_auc']}
    go=paired['acc']>=.1 and paired['hard_auc']>=.1 and wtl['acc'][0]>=7 and wtl['hard_auc'][0]>=7;conclusion='GO_GFR_INCREMENTAL_EVIDENCE' if go else 'NO-GO / 增量价值未获充分支持'
    summaries[0].update({f'delta_vs_raw_{k}_pp':v for k,v in paired.items()},acc_wtl='/'.join(map(str,wtl['acc'])),hard_auc_wtl='/'.join(map(str,wtl['hard_auc'])),conclusion=conclusion);csvwrite(RESULTS/'summary.csv',summaries)
    lines+=['','## 同折配对 E00−E01','','|fold|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|E00 best/epochs|E01 best/epochs|','|---|---:|---:|---:|---:|---|---|']
    for f in range(1,11):
        a,b=by[0][f],by[1][f];lines.append(f'|{f}|'+'|'.join(f'{diffs[k][f-1]:+.4f}' for k in CFG['metrics'])+f"|{a['best_epoch']}/{a['epochs_executed']}|{b['best_epoch']}/{b['epochs_executed']}|")
    lines+=['|平均|'+'|'.join(f'{paired[k]:+.4f}' for k in CFG['metrics'])+'|—|—|','',f"ACC W/T/L：{'/'.join(map(str,wtl['acc']))}；Hard-AUC W/T/L：{'/'.join(map(str,wtl['hard_auc']))}。",'',f'预注册判定：**{conclusion}**。门槛为两项均≥+0.1pp，且各自≥7折严格胜。未达门槛时不等价于证明所有患者图无效。','', '## 历史复现与初始化审计','','E00重新完整训练10折；所有可用历史混淆、四指标、best_epoch、epochs_executed、初始化参数和CPU/CUDA入口RNG逐折匹配。E43.1另有完整四类RNG及预测SHA，本轮逐折匹配；独立读回进一步比较十折最佳模型所有张量，完全一致。历史B0-B没有保存全部预测和完整四类RNG，未伪称核验不存在的历史字段。','十个未训练锚点按E43.1原signature验证，未将旧signature误要求为新代码signature。共同PLN初始化、双Adam初态和完整RNG逐值相同；每折E00/E01的共同训练epoch后完整RNG SHA全部匹配。两组停止时间不同允许后续未共有epoch不作比较。E01全程不调用GFR/GCN/图损失，GFR参数与Adam状态SHA保持初始化值；从头训练的是同一PLN，而非最佳GFR模型微调。','', '## 每折混淆：TP-ASD/FN-ASD/FP-TD/TN-TD','','|fold|E00|E01|','|---|---|---|']
    for f in range(1,11):lines.append(f'|{f}|'+'|'.join('/'.join(str(by[g][f][k]) for k in ['asd_tp','asd_fn','td_fp','td_tn']) for g in (0,1))+'|')
    diags=csvread(RESULTS/'diagnostics.csv')
    if diags:
        dd=[r for r in diags if r['group']==GROUPS[0]];relative=[float(r['gfr_minus_raw_relative']) for r in dd if r['modality']=='PHENO']
        lines+=['','## GFR机制','','|模态|分类消息梯度范数|图构建正则梯度范数|有效消息/输入范数|关闭分支logp变化|关闭类别翻转总数|','|---|---:|---:|---:|---:|---:|']
        for mod in ['PHENO','ANAT','FUNC','CORR']:
            d=[r for r in dd if r['modality']==mod];lines.append('|'+mod+'|'+'|'.join(f"{np.mean([float(r[k]) for r in d]):.8g}" for k in ['classification_branch_gradient_norm','graph_projection_gradient_norm','effective_contribution_input_ratio','branch_off_logp_mean'])+'|'+str(sum(int(r['branch_off_flip_count']) for r in d))+'|')
        one=[r for r in dd if r['modality']=='PHENO'];lines+=[f"\n最佳模型GFR−raw相对范数：十折均值{np.mean(relative):.8g}，范围[{min(relative):.8g}, {max(relative):.8g}]。同时关闭四支消息类别翻转{sum(int(r['joint_off_flip_count']) for r in one)}/871；平均绝对logp变化{np.mean([float(r['joint_off_logp_mean']) for r in one]):.8g}。",'PHENO中间消息虽可非零，最终直接贡献与分类消息梯度均为0，因为i==1覆盖i==0。图构建正则仍可更新PHENO图学习参数。固定模型关闭消息只解释依赖，不能替代本轮E01独立重训。']
    counts={str(g):sum(r['physical_gpu']==g for r in rows) for g in (0,1)}
    elapsed=time.time()-m['started_unix'];formal=m.get('formal_wall_seconds',0.)
    lines+=['','## 工程验收、时间与限制','',f'GPU0={counts["0"]}/10折（fold1–5两组），GPU1={counts["1"]}/10折（fold6–10两组）。正式训练墙钟{formal:.2f}秒；本次存储保护记录起至报告{elapsed:.2f}秒。每GPU一个worker、每worker2 CPU线程；不调用CLIP/GPT、不下载模型、未Git提交/推送。','20个最佳checkpoint独立重载：预测、四指标、混淆、模型/预测SHA及split/anchor全通过。预检含两组两步后磁盘恢复第三步逐值复演、E00原始步梯度/双Adam/RNG等价、E01输入恒等和GFR零调用、测试标签置换前向不变。受保护根源码、原数据、E43.1及历史基线文件SHA保持不变。','原val=test选首次最佳、CV前全数据标准化/RFE、全节点transductive和重叠训练集保持。Hard-AUC由硬标签计算，不能当作概率ROC-AUC；不额外报BA等第五指标。该配对比较不构成独立泛化、显著性或临床因果证据。差异归于整体GFR包（传播和必要的图正则），本轮未隔离它们的单独作用。','',f'产物：`{EXP}`。训练恢复：`python -B run.py resume`；独立复核：`python -B run.py audit`；报告重建：`python -B run.py report`。']
    text='\n'.join(lines)+'\n';(RESULTS/'report.md').write_text(text)
    m.update(scientific='COMPLETE',status='SCIENTIFIC_COMPLETE',groups_complete=2,folds_complete=20,gpu_folds=counts,conclusion=conclusion,summary=summaries,paired_e00_minus_e01_pp=paired,paired_wtl=wtl,paired_per_fold_pp=diffs,total_wall_seconds=elapsed,code_signature=signature())
    write(EXP/'manifest.json',m)
    if final:
        append='\n\n<!-- EXPERIMENT43_2_GFR_INCREMENTAL_VALUE -->\n## Experiment 43.2 — GFR Incremental Value Audit\n\n'+text.split('\n',1)[1]
        handoff=append_handoff(append);m.update(handoff=handoff,delivery='COMPLETE' if handoff['status']!='PENDING' else 'PARTIAL',status='COMPLETE' if handoff['status']!='PENDING' else 'SCIENTIFIC_COMPLETE_DELIVERY_PARTIAL');write(EXP/'manifest.json',m)
        with (RESULTS/'report.md').open('a') as f:f.write('\nHandoff状态：'+handoff['status']+'；交付状态：'+m['delivery']+'。\n')
    print(text[:3500],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','preflight','worker','all','resume','audit','audit-worker','report']);p.add_argument('--gpu',type=int,default=0);p.add_argument('--group',type=int,choices=[0,1],default=0);args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='preflight':preflight(args.gpu)
    elif args.command=='worker':
        runtime()
        for fold in CFG['gpu_folds'][str(args.gpu)]:execute_fold(args.group,fold,args.gpu)
    elif args.command in ['all','resume']:controller()
    elif args.command=='audit-worker':audit_worker(args.gpu)
    elif args.command=='audit':audit_controller()
    elif args.command=='report':report(True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        traceback.print_exc();write(RT/'tasks'/f'failure_{os.getpid()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time(),command=sys.argv));raise
