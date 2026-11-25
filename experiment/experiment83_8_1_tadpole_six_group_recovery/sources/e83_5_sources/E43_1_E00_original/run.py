#!/usr/bin/env python3
"""E43.1 immutable B0-B anchors, continuous historical streams, paired GFR tests."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
sys.dont_write_bytecode=True
import argparse,contextlib,copy,csv,fcntl,hashlib,io,json,pickle,random,subprocess,time,traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path('/data/adodas/SPromptGL-main')
EXP=Path(__file__).resolve().parent
RT=EXP/'.runtime';RESULTS=EXP/'results'
CFG=json.loads((EXP/'config.json').read_text());PROTOCOL=CFG['protocol'];INPUT_DIMS=[48,6,10,256];GROUPS=CFG['groups']
CTX=None;DATA=None
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
def history():return {int(r['fold']):r for r in csvread(ROOT/'experiment/b0_b_seed0_full10fold/results/fold_metrics.csv')}
def runtime():
    global CTX
    if CTX is None:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        torch.cuda.set_device(0);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        sys.path.insert(0,str(RT/'sources'))
        import layers,network,utils,method
        assert 'CLIP.clip.clip' not in sys.modules and not hasattr(network,'PLN')
        CTX=dict(torch=torch,np=np,layers=layers,network=network,utils=utils,method=method)
    return CTX

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

def signature():
    files=['run.py','method.py','integration.py','integration.patch','config.json']
    return hashlib.sha256(json.dumps({n:sha(EXP/n) for n in files},sort_keys=True).encode()).hexdigest()
def foldsha(fold):
    tr,te=data()[2][fold-1]
    return hashlib.sha256(tr.astype(np.int64).tobytes()+te.astype(np.int64).tobytes()).hexdigest()
def fresh(fold):
    c=runtime();x,y,folds=data();tr,te=folds[fold-1]
    return make_agent(c['torch'],np,c['layers'],c['network'],c['utils'],SimpleNamespace(**PROTOCOL),x,y,tr,te)
def fromanchor(group,fold):
    c=runtime();anchor=load(RT/'anchors'/f'fold{fold:02}.pt');assert anchor['signature']==signature() and anchor['split_sha']==foldsha(fold)
    restore(anchor['entry_rng']);a=fresh(fold)
    assert digest_state(a)==anchor['parameter_sha'] and equal(model(a),anchor['model'])
    assert rngsha(rng())==rngsha(anchor['rng'])
    a.optimizer_MP0.load_state_dict(copy.deepcopy(anchor['mp']));a.optimizer_MF.load_state_dict(copy.deepcopy(anchor['mf']))
    restore(anchor['rng']);before=rngsha(rng());c['method'].install(a,group)
    assert before==rngsha(rng()) and digest_state(a)==anchor['parameter_sha']
    a.sup_weight=None;a.sup_calibration={}
    return a,anchor

def forward(a):
    c=runtime();n,parts=a.dataLayer(a.feat);feat,adjs,aloss=a.MessagePassing0(parts,n,a.targ,a.trn_idx)
    gl=sum(c['utils'].GraphConstructLoss(parts[i],adjs[i],1.,.5,0.) for i in range(4))/4
    out=a.ModalFusion(feat)[0];cl=c['utils'].ClsLoss(out,a.targ,a.trn_idx,a.weight)
    return out,cl,gl,adjs,aloss

def evaluate(a):
    a.MessagePassing0.eval();a.ModalFusion.eval()
    with runtime()['torch'].no_grad():return forward(a)
def norms(loss,params):
    t=runtime()['torch'];gs=t.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    v=sum((g.square().sum() for g in gs if g is not None),t.tensor(0.,device='cuda'))
    return float(v.sqrt()),gs

def trainstep(a,group,audit=False):
    t=runtime()['torch'];a.MessagePassing0.train();a.ModalFusion.train();a.optimizer_MP0.zero_grad();a.optimizer_MF.zero_grad()
    out,cl,gl,adj,al=forward(a);sup=cl.new_zeros(());stats={}
    if group==2:
        sup=runtime()['method'].supervised(adj,a.trn_idx,a.targ[a.trn_idx])
        if a.sup_weight is None:
            params=[p for i in range(4) for p in getattr(a.MessagePassing0,f'GraphConstruct{i}').p.parameters()]
            cn,cg=norms(cl,params);sn,sg=norms(sup,params)
            a.sup_weight=min(1.,.1*cn/(sn+1e-12));a.sup_calibration=dict(cls_projection_norm=cn,sup_projection_norm=sn,lambda_sup=a.sup_weight,scaled_gradient_ratio=a.sup_weight*sn/(cn+1e-12),sup_permod_projection_grad=[float(sum((p.square().sum() for p in sg[2*i:2*i+2] if p is not None),cl.new_zeros(())).sqrt()) for i in range(4)])
            assert cn>0 and sn>0 and all(v>0 for v in a.sup_calibration['sup_permod_projection_grad'])
        stats.update(a.sup_calibration)
    if audit:
        ps=[p for i in range(4) for p in getattr(a.MessagePassing0,f'MessagePassing{i}').parameters()];_,gs=norms(cl,ps)
        for i in range(4):stats[f'cls_branch_grad_{i}']=float(sum((v.square().sum() for v in gs[2*i:2*i+2] if v is not None),cl.new_zeros(())).sqrt())
    loss=cl+gl+(a.sup_weight*sup if group==2 else 0.)
    assert t.isfinite(loss);loss.backward()
    assert all(t.isfinite(p.grad).all() for m in [a.MessagePassing0,a.ModalFusion] for p in m.parameters() if p.grad is not None)
    a.optimizer_MP0.step();a.optimizer_MF.step()
    return {**dict(cls=float(cl.detach()),graph=float(gl.detach()),sup=float(sup.detach()),lambda_sup=float(a.sup_weight or 0.),total=float(loss.detach())),**stats}

def diagnostics(a,group,fold):
    t=runtime()['torch'];out,cl,gl,adj,_=evaluate(a);g=a.MessagePassing0
    msgs=[g.messages431[i].clone() for i in range(4)];inputs=[g.inputs431[i] for i in range(4)]
    prop=g.prop431 if group else adj;rows=[]
    gradparams=[p for i in range(4) for p in getattr(g,f'MessagePassing{i}').parameters()]
    # eval-mode, classification-only gradient, leaving model and optimizers unchanged.
    oo,cc,_,_,_=forward(a);_,gs=norms(cc,gradparams)
    for i in range(4):
        ar,ap=adj[i],prop[i];mask=ap>0;off=mask.clone();off.fill_diagonal_(False);deg=off.sum(1).float();fw=float(t.sigmoid(g.fusion_weight[i]));msg=msgs[i]
        start,stop=runtime()['method'].SLICES[i];excluded=t.ones(320,device='cuda',dtype=t.bool);excluded[start:stop]=False
        g.off431=i;changed=evaluate(a)[0];g.off431=None
        row=dict(group=GROUPS[group],seed=0,fold=fold,modality=runtime()['method'].MODALITIES[i],directed_nonself_edges=int(off.sum()),density=float(off.sum()/(871*870)),degree_mean=float(deg.mean()),degree_min=int(deg.min()),degree_max=int(deg.max()),degree_sd=float(deg.std(unbiased=False)),isolated=int((deg==0).sum()),symmetry_max=float((ap-ap.T).abs().max()),diagonal_min=float(ap.diag().min()),diagonal_max=float(ap.diag().max()),support_changed=int(((ar>0)!=(ap>0)).sum()),diagonal_changed=int((ar.diag()!=ap.diag()).sum()),prop_weight_mean=float(ap[off].mean()) if off.any() else 0.,raw_weight_mean=float(ar[off].mean()) if off.any() else 0.,message_norm=float(msg.norm()),fusion_sigmoid=fw,contribution_input_ratio=float(msg.norm())*fw/float(a.feat.norm()) if not(group==0 and i==0) else 0.,branch_off_logp_mean=float((out[a.tst_idx]-changed[a.tst_idx]).abs().mean()),branch_off_class_flip=float((out[a.tst_idx].argmax(1)!=changed[a.tst_idx].argmax(1)).float().mean()),cls_branch_grad=float(sum((v.square().sum() for v in gs[2*i:2*i+2] if v is not None),cc.new_zeros(())).sqrt()),outside_input_max=float(inputs[i][:,excluded].abs().max()),outside_output_max=float(msg[:,excluded].abs().max()),lambda_sup=float(a.sup_weight or 0.))
        if group==4:assert row['outside_input_max']==row['outside_output_max']==0.
        assert row['support_changed']==row['diagonal_changed']==0
        if group==2:
            sup=runtime()['method'].supervised(adj,a.trn_idx,a.targ[a.trn_idx]);row['sup_best']=float(sup);row.update({k:v for k,v in a.sup_calibration.items() if not isinstance(v,list)});row['sup_projection_grad_initial']=a.sup_calibration['sup_permod_projection_grad'][i]
        rows.append(row)
    return rows

def check_history(row):
    ref=history()[row['fold']];fields=['acc','hard_auc','spe','sen','asd_tp','asd_fn','td_fp','td_tn','best_epoch','epochs_executed','initial_parameter_sha256','initial_rng_sha256']
    diff={k:[ref[k],row[k]] for k in fields if (abs(float(ref[k])-float(row[k]))>1e-12 if k not in ['initial_parameter_sha256','initial_rng_sha256'] else ref[k]!=row[k])}
    return diff

def completed(group,fold):
    p=RT/'tasks'/f'g{group}_f{fold:02}.json'
    if not p.exists():return False
    r=read(p)
    assert r['signature']==signature() and r['split_sha']==foldsha(fold)
    if r['status']!='COMPLETE':return False
    assert sha(RT/'checkpoints'/f'g{group}_f{fold:02}.best.pt')==r['best_sha']
    return True

def execute_fold(group,fold,gpu,baseline_agent=None,entry=None):
    t=runtime()['torch'];started=time.monotonic();key=f'g{group}_f{fold:02}';task=RT/'tasks'/f'{key}.json';latest=RT/'checkpoints'/f'{key}.latest.pt';bestpath=RT/'checkpoints'/f'{key}.best.pt'
    lock=(RT/'tasks'/f'{key}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if completed(group,fold):return read(task)
    if group==0:
        a=baseline_agent;anchor=load(RT/'anchors'/f'fold{fold:02}.pt')
    else:a,anchor=fromanchor(group,fold)
    base=dict(group=GROUPS[group],group_id=group,seed=0,fold=fold,physical_gpu=gpu,signature=signature(),split_sha=foldsha(fold),initial_parameter_sha256=anchor['parameter_sha'],initial_rng_sha256=legacy_rngsha(anchor['entry_rng']),entry_full_rng_sha256=rngsha(anchor['entry_rng']),post_initial_rng_sha256=rngsha(anchor['rng']),anchor_sha=sha(RT/'anchors'/f'fold{fold:02}.pt'))
    begin=0;bestacc=0.;bestepoch=-1;wait=0;best=None;trace=[];elapsed=0.
    if latest.exists():
        s=load(latest);assert s['signature']==signature() and s['split_sha']==foldsha(fold)
        setmodel(a,s['model']);a.optimizer_MP0.load_state_dict(s['mp']);a.optimizer_MF.load_state_dict(s['mf']);restore(s['rng']);begin=s['epoch']+1;bestacc=s['bestacc'];bestepoch=s['bestepoch'];wait=s['wait'];best=s['best'];trace=s['trace'];elapsed=s['elapsed'];a.sup_weight=s['sup_weight'];a.sup_calibration=s['sup_calibration']
        print('RESUME',key,begin,flush=True)
    write(task,dict(**base,status='RUNNING',epoch=begin))
    epoch=begin-1
    for epoch in (range(begin,PROTOCOL['nepoch']) if wait<=PROTOCOL['early'] else []):
        values=trainstep(a,group);out=evaluate(a)[0];acc=float(out[a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean().item())
        if acc>bestacc:bestacc=acc;bestepoch=epoch;wait=0;best=model(a)
        else:wait+=1
        if epoch%10==0 or epoch==449:trace.append(dict(group=GROUPS[group],seed=0,fold=fold,epoch=epoch,**values))
        if (epoch+1)%10==0 or wait>PROTOCOL['early'] or epoch==449:
            save(latest,dict(**base,model=model(a),mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),rng=rng(),epoch=epoch,bestacc=bestacc,bestepoch=bestepoch,wait=wait,best=best,trace=trace,elapsed=elapsed+time.monotonic()-started,sup_weight=a.sup_weight,sup_calibration=a.sup_calibration))
            write(task,dict(**base,status='RUNNING',epoch=epoch,best_epoch=bestepoch))
        if wait>PROTOCOL['early']:break
    assert best is not None
    boundary=copy.deepcopy(rng());setmodel(a,best);out=evaluate(a)[0];metrics=metric_row(out,data()[1],a.tst_idx)
    prediction=out[a.tst_idx].argmax(1).detach().cpu().numpy().astype(np.int64)
    save(bestpath,dict(**base,model=best,prediction=prediction,test_indices=a.tst_idx,best_epoch=bestepoch,epochs_executed=epoch+1,metrics=metrics,sup_weight=a.sup_weight,sup_calibration=a.sup_calibration))
    row=dict(**base,**metrics,best_epoch=bestepoch,epochs_executed=epoch+1,prediction_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),best_sha=sha(bestpath),wall_seconds=elapsed+time.monotonic()-started,status='COMPLETE')
    if group==0:
        row['historical_differences']=check_history(row)
        if row['historical_differences']:
            write(task,dict(**row,status='ANCHOR_REPLAY_FAIL'));raise AssertionError(row['historical_differences'])
    # Best-checkpoint diagnosis is read-only and every RNG is restored afterward.
    before=digest_state(a);optbefore=copy.deepcopy([a.optimizer_MP0.state_dict(),a.optimizer_MF.state_dict()])
    row['diagnostics']=diagnostics(a,group,fold);assert digest_state(a)==before and equal(optbefore,[a.optimizer_MP0.state_dict(),a.optimizer_MF.state_dict()]);restore(boundary)
    row['loss_trace']=trace
    if group==0:save(RT/'anchors'/f'boundary{fold:02}.pt',dict(rng=boundary,signature=signature()))
    # Result is committed last, after validated best state and replay comparison.
    saved=load(bestpath);assert equal(saved['prediction'],prediction) and saved['metrics']==metrics
    write(task,row);latest.unlink(missing_ok=True)
    print('FOLD_DONE',key,'gpu',gpu,'ACC',metrics['acc'],'epoch',bestepoch,'seconds',round(row['wall_seconds'],2),flush=True)
    return row

def baseline_worker(gpu):
    c=runtime();data();seed_everything(c['torch'],0)
    for fold in (range(1,6) if gpu==0 else range(6,11)):
        if completed(0,fold):restore(load(RT/'anchors'/f'boundary{fold:02}.pt')['rng']);continue
        entry=copy.deepcopy(rng());assert legacy_rngsha(entry)==history()[fold]['initial_rng_sha256'],f'entry RNG mismatch fold {fold}'
        a=fresh(fold);assert digest_state(a)==history()[fold]['initial_parameter_sha256'],f'initial model mismatch fold {fold}'
        anchor=dict(model=model(a),mp=copy.deepcopy(a.optimizer_MP0.state_dict()),mf=copy.deepcopy(a.optimizer_MF.state_dict()),entry_rng=entry,rng=copy.deepcopy(rng()),parameter_sha=digest_state(a),split_sha=foldsha(fold),signature=signature(),fold=fold)
        p=RT/'anchors'/f'fold{fold:02}.pt'
        if p.exists():assert equal(load(p),anchor)
        else:save(p,anchor)
        c['method'].install(a,0);a.sup_weight=None;a.sup_calibration={}
        execute_fold(0,fold,gpu,a,entry);del a;c['torch'].cuda.empty_cache()

def preflight():
    c=runtime();t=c['torch'];data();checks={};seed_everything(t,0);entry=copy.deepcopy(rng());a=fresh(1);initial=model(a);post=copy.deepcopy(rng())
    assert digest_state(a)==history()[1]['initial_parameter_sha256'];assert legacy_rngsha(entry)==history()[1]['initial_rng_sha256']
    assert sum(p.numel() for p in a.MessagePassing0.parameters())==887836
    assert not any(hasattr(a.ModalFusion,n) for n in ['MLP','prompt_text_features','avg_pool']);assert len(list(a.ModalFusion.named_parameters()))==35
    checks['physical_PEN_removal_no_CLIP_and_history_initial_sha_rng']=True
    # Independent original physical runner step versus instrumented step.
    b=fresh(1);setmodel(b,initial);c['method'].install(b,0);b.sup_weight=None;b.sup_calibration={}
    restore(post);original_step(a,t,c['utils']);expected=model(a);op=[copy.deepcopy(a.optimizer_MP0.state_dict()),copy.deepcopy(a.optimizer_MF.state_dict())];after=copy.deepcopy(rng())
    restore(post);trainstep(b,0)
    assert equal(expected,model(b)) and equal(op,[b.optimizer_MP0.state_dict(),b.optimizer_MF.state_dict()]) and equal(after,rng())
    checks['original_loss_gradients_two_adams_first_step_bitwise']=True
    # Two steps, actual disk checkpoint, exact third step including optimizers/RNG.
    trainstep(b,0);p=RT/'checkpoints/preflight.pt';save(p,dict(model=model(b),mp=b.optimizer_MP0.state_dict(),mf=b.optimizer_MF.state_dict(),rng=rng()))
    trainstep(b,0);expected=model(b);ops=copy.deepcopy([b.optimizer_MP0.state_dict(),b.optimizer_MF.state_dict()]);rr=copy.deepcopy(rng())
    ss=load(p);setmodel(a,ss['model']);a.optimizer_MP0.load_state_dict(ss['mp']);a.optimizer_MF.load_state_dict(ss['mf']);a.sup_weight=None;a.sup_calibration={};restore(ss['rng']);trainstep(a,0)
    assert equal(expected,model(a)) and equal(ops,[a.optimizer_MP0.state_dict(),a.optimizer_MF.state_dict()]) and equal(rr,rng());p.unlink();checks['disk_resume_third_update_all_states_exact']=True
    for group in range(1,6):
        z=fresh(1);setmodel(z,initial);restore(post);before=rngsha(rng());c['method'].install(z,group);z.sup_weight=None;z.sup_calibration={};assert before==rngsha(rng())
        audit=trainstep(z,group,True);assert all(audit[f'cls_branch_grad_{i}']>0 for i in range(4));checks[f'group{group}_first_update']=audit
        out,_,_,adj,_=evaluate(z)
        if group in (3,5):assert all(t.equal(raw>0,prop>0) and t.equal(raw.diag(),prop.diag()) for raw,prop in zip(adj,z.MessagePassing0.prop431))
        if group==4:assert all(float(z.MessagePassing0.messages431[i][:,np.r_[0:s,stop:320]].abs().max())==0 for i,(s,stop) in enumerate(c['method'].SLICES))
        if group==2:
            loss=c['method'].supervised(adj,z.trn_idx,z.targ[z.trn_idx]);mut=z.targ.clone();mut[z.tst_idx]=1-mut[z.tst_idx];assert t.equal(loss,c['method'].supervised(adj,z.trn_idx,mut[z.trn_idx]))
            original=z.targ;z.targ=mut;changed=evaluate(z)[0];z.targ=original;assert t.equal(out,changed)
            checks['E02_no_heldout_label_in_loss_or_forward']=True
    checks['E03_E05_same_registered_math']=True
    z=fresh(1);setmodel(z,initial);restore(post);c['method'].install(z,3);z.sup_weight=None;z.sup_calibration={};trainstep(z,3);expected=model(z);rr=copy.deepcopy(rng())
    z=fresh(1);setmodel(z,initial);restore(post);c['method'].install(z,5);z.sup_weight=None;z.sup_calibration={};trainstep(z,5);assert equal(expected,model(z)) and equal(rr,rng())
    checks['E03_E05_first_update_exact']=True
    checks['hardware']=str(t.cuda.get_device_properties(0));checks['peak_memory_bytes']=t.cuda.max_memory_allocated();checks['status']='PASS';write(RT/'preflight.json',checks);print('PREFLIGHT_PASS',json.dumps(checks),flush=True)

def verify_sources():
    m=read(EXP/'manifest.json')
    for p,h in m['protected_sha256'].items():assert sha(p)==h,'Protected file modified: '+p

def refresh():
    rows=[];diags=[];traces=[]
    for p in sorted((RT/'tasks').glob('g*_f*.json')):
        r=read(p)
        if r['status']=='COMPLETE':
            assert completed(r['group_id'],r['fold']);diags+=r['diagnostics'];traces+=r['loss_trace'];rows.append({k:v for k,v in r.items() if k not in ['diagnostics','loss_trace','historical_differences']})
    assert len({(r['group_id'],r['seed'],r['fold']) for r in rows})==len(rows)
    rows.sort(key=lambda r:(r['group_id'],r['fold']));csvwrite(RESULTS/'fold_metrics.csv',rows);csvwrite(RESULTS/'diagnostics.csv',diags);csvwrite(RESULTS/'loss_trace.csv',traces)
    return rows,diags

def launch(args,gpu,label):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),E431_GPU=str(gpu))
    p=RT/'logs'/f'{label}.log';log=p.open('a',buffering=1)
    child=subprocess.Popen([sys.executable,'-B','-u',str(EXP/'run.py'),*args],cwd=str(EXP),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return child,log,label

def dispatch(jobs):
    pending=list(jobs);live=[];started=time.monotonic()
    while pending or live:
        for gpu in (0,1):
            while sum(1 for q in live if q[0]==gpu)<2:
                hit=next((i for i,j in enumerate(pending) if j[0]==gpu),None)
                if hit is None:break
                gp,args,label=pending.pop(hit);live.append((gp,*launch(args,gp,label)))
        time.sleep(3)
        for item in live[:]:
            gp,p,l,label=item
            if p.poll() is not None:
                l.close();live.remove(item)
                if p.returncode:
                    for _,other,_,_ in live:other.terminate()
                    raise RuntimeError(f'Worker failed {label}: see .runtime/logs/{label}.log')
                print('TASK_COMPLETE',label,'elapsed',round(time.monotonic()-started,1),flush=True)
        refresh()

def controller():
    lock=(RT/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_sources();m=read(EXP/'manifest.json');assert (RT/'preflight.json').exists() and read(RT/'preflight.json')['status']=='PASS'
    m.update(status='BASELINE_RUNNING',code_signature=signature());write(EXP/'manifest.json',m)
    dispatch([(g,['baseline','--gpu',str(g)],f'baseline_gpu{g}') for g in (0,1)])
    rows,_=refresh();base=[r for r in rows if r['group_id']==0];assert len(base)==10
    assert all(not read(RT/'tasks'/f'g0_f{f:02}.json')['historical_differences'] for f in range(1,11))
    m=read(EXP/'manifest.json');m.update(status='ANCHOR_REPLAY_PASS',anchor_replay='PASS',anchor_checks=10);write(EXP/'manifest.json',m);print('ANCHOR_REPLAY_PASS 10/10; starting 50 paired folds',flush=True)
    jobs=[]
    # Ten-fold group subsets stay on each assigned device; E05 split is explicit to meet 25/25.
    for gpu in (0,1):
        assignment=CFG[f'innovation_gpu{gpu}']
        for group in sorted({g for g,f in assignment}):
            folds=[f for g,f in assignment if g==group]
            jobs.append((gpu,['worker','--group',str(group),'--folds',','.join(map(str,folds)),'--gpu',str(gpu)],f'group{group}_gpu{gpu}'))
    dispatch(jobs);report(True)

def paired(rows,group):
    base={r['fold']:r for r in rows if r['group_id']==0};other={r['fold']:r for r in rows if r['group_id']==group}
    assert len(base)==len(other)==10
    diffs={k:[100*(other[f][k]-base[f][k]) for f in range(1,11)] for k in CFG['metrics']}
    out={f'delta_{k}_pp':float(np.mean(v)) for k,v in diffs.items()}
    for k in ['acc','hard_auc']:
        a=np.array(diffs[k]);out[f'{k}_wtl']=[int((a>1e-10).sum()),int((abs(a)<=1e-10).sum()),int((a<-1e-10).sum())]
    out['per_fold_pp']=diffs
    return out

def append_handoff(text):
    p=ROOT/'handoff.md';pending=EXP/'handoff_append_pending.md';marker='<!-- EXPERIMENT43_1_B0B_GFR_RETEST -->'
    try:
        with p.open('r+b') as f:
            fcntl.flock(f,fcntl.LOCK_EX);before=f.read();h=hashlib.sha256(before).hexdigest()
            if marker.encode() in before:return dict(status='ALREADY_APPENDED',prefix_sha256=h)
            f.seek(0,2);f.write(text.encode());f.flush();os.fsync(f.fileno());f.seek(0);after=f.read();assert after[:len(before)]==before and hashlib.sha256(after[:len(before)]).hexdigest()==h
        pending.unlink(missing_ok=True)
        return dict(status='APPENDED',prefix_bytes=len(before),prefix_sha256=h,after_sha256=hashlib.sha256(after).hexdigest())
    except Exception as e:
        pending.write_text(text+'\nAppend failure: '+repr(e)+'\n');return dict(status='PENDING',error=repr(e))

def report(final=False):
    rows,diags=refresh();summaries=[];pairs={};lines=['# Experiment 43.1 — B0-B GFR机制复验','',f'完成 {len(rows)}/60 唯一正式折。正式四指标为十折 mean ± population SD，单位%。','', '|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|','|---|---:|---:|---:|---:|']
    for g,name in enumerate(GROUPS):
        rr=[r for r in rows if r['group_id']==g]
        if len(rr)!=10:continue
        stats={}
        for k in CFG['metrics']:v=np.array([r[k] for r in rr])*100;stats[f'{k}_mean']=float(v.mean());stats[f'{k}_sd']=float(v.std(ddof=0))
        summary=dict(group=name,seed=0,folds=10,**stats)
        if g and len([r for r in rows if r['group_id']==0])==10:
            p=paired(rows,g);pairs[name]=p;summary.update({k:v for k,v in p.items() if k!='per_fold_pp'})
            mechanism=all(r['support_changed']==r['diagonal_changed']==0 and r['cls_branch_grad']>0 for r in diags if r['group']==name)
            if g==2:mechanism=mechanism and all(r['lambda_sup']>0 and r['scaled_gradient_ratio']<=.100001 and r['sup_projection_grad_initial']>0 for r in diags if r['group']==name)
            if g==4:mechanism=mechanism and all(r['outside_input_max']==r['outside_output_max']==0 for r in diags if r['group']==name)
            summary['go']=p['delta_acc_pp']>=.1 and p['delta_hard_auc_pp']>=.1 and p['acc_wtl'][0]>=7 and p['hard_auc_wtl'][0]>=7 and mechanism
        summaries.append(summary);lines.append('|'+name+'|'+'|'.join(f"{stats[k+'_mean']:.4f} ± {stats[k+'_sd']:.4f}" for k in CFG['metrics'])+'|')
    lines+=['','## 与同折 B0-B 配对','', '|组别−B0-B|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|ACC W/T/L|AUC W/T/L|判定|','|---|---:|---:|---:|---:|---|---|---|']
    for s in summaries[1:]:
        p=pairs[s['group']];lines.append('|'+s['group']+'|'+'|'.join(f"{p['delta_'+k+'_pp']:+.4f}" for k in CFG['metrics'])+'|'+'/'.join(map(str,p['acc_wtl']))+'|'+'/'.join(map(str,p['hard_auc_wtl']))+'|'+('GO_CANDIDATE' if s['go'] else 'NO-GO')+'|')
    lines+=['','## 基线身份及设计边界','','B0-B物理不构造PEN/CLIP；PLN49,544参数，GFR887,836参数。原PLN fusion_weight的6个闲置参数是原有注册，不属于PEN。本轮未使用任何B0-A/E40锚点。B0-B历史为两条seed0连续随机流：GPU0折1–5，GPU1折6–10。保存新基线的首次更新前模型、双Adam与完整四种RNG；其余组逐折恢复这些未训练锚点。','历史未保留预测向量或完整Python/NumPy RNG，不能宣称历史预测SHA或完整四类历史RNG匹配；逐折实际核验所有历史已有的参数SHA、CPU/CUDA入口SHA、混淆、四指标、best/执行epoch。本轮另存完整RNG与预测向量及SHA。','E03与E05按用户定义完全同构；保留两组正式训练作为重复性核验，不能解释成两个独立改进或组合协同。','E02是仓库非平方Frobenius ALoss的训练诱导子图安全实现；论文式(1)为平方范数，故不声称数值复现论文损失。预测与同类目标均先限制train后加I归一化；原对角约1，加I约2；标签同类对角为1，加I为2。lambda只由首个同一次训练前向的四个p.weight/p.bias梯度计算，冻结不调参。原非优化ALoss继续不进入基础损失。','E03/E05校准仅作用传播图，原四GraphConstructLoss继续使用原始GraphLearn输出；每次forward逐元素验证支持、对角不变，没有mutual/topk/新边。E04输入和输出两端均限制对应模态块，参数形状仍320→320。','','## 机制诊断','', '|组别|模态|邻居均值|分类消息梯度|贡献/输入范数|关闭分支log-prob变化|关闭翻转率|','|---|---|---:|---:|---:|---:|---:|']
    for g,name in enumerate(GROUPS):
        for mod in ['PHENO','ANAT','FUNC','CORR']:
            dd=[d for d in diags if d['group']==name and d['modality']==mod]
            if len(dd)==10:lines.append('|'+name+'|'+mod+'|'+'|'.join(f"{np.mean([d[k] for d in dd]):.7g}" for k in ['degree_mean','cls_branch_grad','contribution_input_ratio','branch_off_logp_mean','branch_off_class_flip'])+'|')
    baseline=[r for r in rows if r['group_id']==0]
    lines+=['','## 基线逐折审计','','|fold|参数SHA|入口CPU/CUDA RNG|混淆/四指标|best epoch|epochs|','|---|---|---|---|---|---|']
    for r in baseline:lines.append(f"|{r['fold']}|MATCH|MATCH|MATCH|{r['best_epoch']}|{r['epochs_executed']}|")
    lines+=['','## 统计边界与结论','','历史B0-A（仅次级参照，非本轮锚点）：ACC92.0742、Hard-AUC91.8546、SPE94.6300、SEN89.0793%。若仅超过B0-B而未超过B0-A，只能声称相对B0-B的候选改善。','原val=test严格首次最佳、CV前全队列RFE/标准化、全871节点transductive与重叠训练集保留；不构成无偏独立泛化、统计显著性、医学语义或临床因果证据。本轮不调用GPT/CLIP，不下载模型。','所有支持/对角/局部传播及首步梯度等工程证据见.runtime/preflight.json，逐折图统计与分支关闭响应见diagnostics.csv，监督权重及损失见loss_trace.csv。']
    if final:
        assert len(rows)==60;verify_sources();assert len(baseline)==10
        e3={r['fold']:r for r in rows if r['group_id']==3};e5={r['fold']:r for r in rows if r['group_id']==5}
        for f in range(1,11):
            assert e3[f]['prediction_sha256']==e5[f]['prediction_sha256'] and all(e3[f][k]==e5[f][k] for k in ['best_epoch','epochs_executed',*CFG['metrics']])
            assert equal(load(RT/'checkpoints'/f'g3_f{f:02}.best.pt')['model'],load(RT/'checkpoints'/f'g5_f{f:02}.best.pt')['model'])
        conclusion='GO_CANDIDATE' if any(s.get('go') for s in summaries) else 'NO-GO';m=read(EXP/'manifest.json');elapsed=time.time()-m['started_unix'];counts={str(g):sum(r['physical_gpu']==g for r in rows) for g in (0,1)}
        lines+=['',f'科研结论：{conclusion}。E03/E05十折最佳模型、预测、epoch与四指标逐值相同。GPU0/GPU1正式折数：{counts}；其中创新折分别25/25，基线各5。总墙钟{elapsed:.2f}秒。']
        text='\n'.join(lines)+'\n';(RESULTS/'report.md').write_text(text);csvwrite(RESULTS/'summary.csv',summaries)
        append='\n\n<!-- EXPERIMENT43_1_B0B_GFR_RETEST -->\n## Experiment 43.1 — B0-B GFR五组优化验证\n\n'+text.split('\n',1)[1]+f'\n产物：`{EXP}`；复现：`python -B run.py all`，精确续跑：`python -B run.py resume`。\n'
        delivery=append_handoff(append);m.update(status='COMPLETE' if delivery['status']!='PENDING' else 'COMPLETE_WITH_HANDOFF_BLOCKED',scientific='COMPLETE',delivery='COMPLETE' if delivery['status']!='PENDING' else 'PARTIAL',conclusion=conclusion,anchor_replay='PASS',folds_complete=60,gpu_folds=counts,total_wall_seconds=elapsed,summary=summaries,paired=pairs,protected_files_unchanged=True,duplicate_E03_E05_best_models_exact=True,handoff=delivery,code_signature=signature());write(EXP/'manifest.json',m)
    else:
        (RESULTS/'report.md').write_text('\n'.join(lines)+'\n');csvwrite(RESULTS/'summary.csv',summaries)
    print('\n'.join(lines[:12]),flush=True)

def prepare():
    from integration import prepare as generate
    generate(ROOT,EXP);verify_sources();m=read(EXP/'manifest.json');m.update(code_signature=signature(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),stage='PREPARED');write(EXP/'manifest.json',m)
    for name in ['fold_metrics.csv','diagnostics.csv','loss_trace.csv']:
        if not (RESULTS/name).exists():csvwrite(RESULTS/name,[])

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','preflight','all','resume','baseline','worker','report']);p.add_argument('--gpu',type=int,default=0);p.add_argument('--group',type=int);p.add_argument('--folds');a=p.parse_args()
    try:
        if a.command=='prepare':prepare()
        elif a.command=='preflight':preflight()
        elif a.command in ['all','resume']:controller()
        elif a.command=='baseline':baseline_worker(a.gpu)
        elif a.command=='worker':
            assert read(EXP/'manifest.json').get('anchor_replay')=='PASS'
            for f in map(int,a.folds.split(',')):execute_fold(a.group,f,a.gpu)
        elif a.command=='report':report(False)
    except Exception as e:
        traceback.print_exc()
        if a.command in ['all','resume','preflight']:
            m=read(EXP/'manifest.json');m.update(status='FAIL',failure=repr(e),failed_command=a.command);write(EXP/'manifest.json',m)
            (RESULTS/'report.md').write_text('# Experiment 43.1 — FAIL\n\n'+repr(e)+'\nSee .runtime/logs and manifest. No missing results imputed.\n')
            text='\n\n<!-- EXPERIMENT43_1_B0B_GFR_RETEST_FAIL -->\n## Experiment 43.1 — B0-B GFR复验执行阻塞\n\n'+repr(e)+f'\n路径：{EXP}；未完成折不生成指标；修复后由resume继续。\n'
            append_handoff(text)
        raise


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

if __name__=="__main__":main()
