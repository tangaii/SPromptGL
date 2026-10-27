#!/usr/bin/env python3
"""E43.3 locked H/P × A/B × G/raw reproduction. No imports of historical Python."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
sys.dont_write_bytecode=True
import argparse,ast,copy,csv,fcntl,hashlib,io,json,pickle,random,subprocess,time,traceback,contextlib,difflib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path('/data/adodas/SPromptGL-main');EXP=Path(__file__).resolve().parent
RT=EXP/'.runtime';RESULTS=EXP/'results';INPUT_DIMS=[48,6,10,256]
CFG=json.loads((EXP/'config.json').read_text());GROUPS={x['id']:x for x in CFG['groups']}
CTX=DATA=None
HA=ROOT/'experiment/experiment43_corr_anchored_gfr_audit'
HB=ROOT/'experiment/experiment43_2_gfr_incremental_value_audit'
E32=ROOT/'experiment/experiment32_disease_contrastive_evidence_routing'
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

def grads(a):
    return {k:{n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in m.named_parameters()} for k,m in [('gfr',a.MessagePassing0),('pln',a.ModalFusion)]}

def retry(fn):
    for attempt in range(5):
        try:return fn()
        except OSError as e:
            if e.errno not in (2,5,116) or attempt==4:raise
            time.sleep(CFG['retry_delays_seconds'][attempt])

def bytesread(p):return retry(lambda:Path(p).read_bytes())
def sha(p):return hashlib.sha256(bytesread(p)).hexdigest()
def read(p):return json.loads(bytesread(p))
def atomic(p,b):
    p=Path(p);temp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    def commit():
        with temp.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
        os.replace(temp,p)
        assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(b).digest()
    retry(commit)
def write(p,v):atomic(p,(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def save(p,v):
    import torch
    stream=io.BytesIO();torch.save(v,stream);atomic(p,stream.getvalue())
def load(p):
    import torch
    return torch.load(io.BytesIO(bytesread(p)),map_location='cpu',weights_only=False)
def csvread(p):return list(csv.DictReader(io.StringIO(bytesread(p).decode())))
def csvwrite(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['group','fold']
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows);atomic(p,stream.getvalue().encode())
def data():
    global DATA
    if DATA is None:
        import pandas as pd
        from sklearn.model_selection import StratifiedKFold
        p=ROOT/'data/ABIDE';frame=pd.read_csv(p/'processed_standard_data.csv');mapping=np.load(p/'modal_feat_dict.npy',allow_pickle=True).item()
        assert list(mapping)==['PHENO','ANAT','FUNC','Correlation'] and [len(v) for v in mapping.values()]==INPUT_DIMS
        assert list(frame.columns[:-1])==[str(x) for v in mapping.values() for x in v]
        x=frame.iloc[:,:-1].to_numpy(np.float32);y=frame.iloc[:,-1].to_numpy(np.int64)-1
        assert x.shape==(871,320) and np.bincount(y).tolist()==[403,468] and np.isfinite(x).all()
        DATA=x,y,list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y))
    return DATA
def foldsha(f):
    tr,te=data()[2][f-1];return hashlib.sha256(tr.astype(np.int64).tobytes()+te.astype(np.int64).tobytes()).hexdigest()
def signature():
    paths=[EXP/n for n in ['run.py','integration.py','integration.patch','config.json']]+sorted((RT/'sources').glob('*.py'))
    return hashlib.sha256(json.dumps({str(p.relative_to(EXP)):sha(p) for p in paths},sort_keys=True).encode()).hexdigest()
def runtime():
    global CTX
    if CTX is None:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        torch.cuda.set_device(0);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        sys.path.insert(0,str(RT/'sources'))
        import layers,network,utils
        CTX=dict(torch=torch,layers=layers,network=network,utils=utils)
    return CTX
def anchorpath(group,f):
    z=GROUPS[group];return RT/'anchors'/f"{z['protocol']}_{z['base']}_fold{f:02}.pt"
def histsha(a):
    h=hashlib.sha256()
    for key,m in [('gfr',a.MessagePassing0),('pln',a.ModalFusion)]:
        for n,p in sorted(m.named_parameters()):h.update(key.encode()+b'/'+n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def fresh(group,f):
    from integration import make_agent
    c=runtime();x,y,folds=data();tr,te=folds[f-1];z=GROUPS[group]
    a=make_agent(c['torch'],np,c['layers'],c['network'],c['utils'],SimpleNamespace(**CFG['protocols'][z['protocol']]),x,y,tr,te,z['base'])
    a.group=group;a.gfr=bool(z['gfr']);a.gfr_calls=0;a.gcn_calls=0;a.graph_calls=0
    def count_gfr(m,args):a.gfr_calls+=1;assert a.gfr,'No-GFR group invoked GFR'
    def count_gcn(m,args):a.gcn_calls+=1;assert a.gfr,'No-GFR group invoked GCN'
    a.MessagePassing0.register_forward_pre_hook(count_gfr)
    for i in range(4):getattr(a.MessagePassing0,f'MessagePassing{i}').register_forward_pre_hook(count_gcn)
    if not a.gfr:a.MessagePassing0.requires_grad_(False)
    return a
def fromanchor(group,f):
    s=load(anchorpath(group,f));assert s['split_sha']==foldsha(f) and s['protocol']==GROUPS[group]['protocol'] and s['base']==GROUPS[group]['base']
    c=runtime()
    if s['base']=='A':c['network']._prompt_text_features=s['model']['pln']['prompt_text_features'].to('cuda')
    restore(s['entry_rng']);a=fresh(group,f)
    # H snapshots transport the historical cold-CLIP construction state exactly;
    # they are untrained states, not a trained checkpoint or a fresh seed reset.
    setmodel(a,s['model']);a.optimizer_MP0.load_state_dict(copy.deepcopy(s['mp']));a.optimizer_MF.load_state_dict(copy.deepcopy(s['mf']));restore(s['rng'])
    assert digest_state(a)==s['parameter_sha'] and equal(model(a),s['model']) and rngsha(rng())==rngsha(s['rng'])
    assert not s['mp']['state'] and not s['mf']['state']
    a.initial_gfr_sha=statesha(a.MessagePassing0.state_dict());a.initial_mp_sha=statesha(a.optimizer_MP0.state_dict())
    return a,s
def forward(a):
    c=runtime();t=c['torch']
    if a.gfr:
        n,parts=a.dataLayer(a.feat);feat,adjs,aloss=a.MessagePassing0(parts,n,a.targ,a.trn_idx)
        gl=sum(c['utils'].GraphConstructLoss(parts[i],adjs[i],a.hyperpm.theta_smooth,a.hyperpm.theta_degree,a.hyperpm.theta_sparsity) for i in range(4))/4;a.graph_calls+=4
    else:feat=a.feat;gl=feat.new_zeros(());adjs=[];aloss=None
    out=a.ModalFusion(feat)[0];cl=c['utils'].ClsLoss(out,a.targ,a.trn_idx,a.weight)
    return out,cl,gl,adjs,aloss,feat
def evaluate(a):
    a.MessagePassing0.eval();a.ModalFusion.eval()
    with runtime()['torch'].no_grad():return forward(a)
def trainstep(a):
    t=runtime()['torch'];a.MessagePassing0.train(a.gfr);a.ModalFusion.train();a.optimizer_MP0.zero_grad();a.optimizer_MF.zero_grad()
    out,cl,gl,_,_,_=forward(a);loss=cl+gl if a.gfr else cl
    assert t.isfinite(out).all() and t.isfinite(loss);loss.backward()
    assert all(t.isfinite(p.grad).all() for m in [a.MessagePassing0,a.ModalFusion] for p in m.parameters() if p.grad is not None)
    if a.gfr:a.optimizer_MP0.step()
    else:assert a.gfr_calls==a.gcn_calls==a.graph_calls==0 and all(p.grad is None for p in a.MessagePassing0.parameters())
    a.optimizer_MF.step()
    return dict(cls=float(cl.detach()),graph=float(gl.detach()),train_acc=float(out[a.trn_idx].argmax(1).eq(a.targ[a.trn_idx]).float().mean()))
def verify_protected():
    for p,h in read(EXP/'manifest.json')['protected_sha256'].items():assert sha(p)==h,'PROTECTED FILE DRIFT '+p
def setup_legacy_loader():
    c=runtime();t=c['torch'];sys.path.insert(0,str(ROOT));from CLIP.clip import clip
    words=['Nromal brain volume','Uniform cortical thickness','Symmetrical brain structure','Normal ventricular morphology','Altered brain volume','Abnormal functional connectivity','White matter integrity issues','Different activation patterns']
    tokens=clip.tokenize(words).to('cuda')
    def loader():
        m,_=clip.load('RN50','cuda')
        with t.no_grad():value=m.encode_text(tokens).float().detach()
        del m
        return value
    c['network']._prompt_text_features=None;c['network']._legacy_loader=loader

def prepare():
    from integration import prepare as generate
    generate(ROOT,EXP);protected={}
    files=[ROOT/n for n in ['README.md','main.py','model.py','network.py','layers.py','utils.py','ABIDE-simple-2-concat-weighted-cosine.sh','data/ABIDE/processed_standard_data.csv','data/ABIDE/modal_feat_dict.npy']]
    for h in [E32,HA,HB,ROOT/'experiment/experiment43_1_b0b_gfr_mechanism_retest',ROOT/'experiment/b0_b_seed0_full10fold',ROOT/'experiment/b0_pen_physical_removal_equivalence_audit']:
        for n in ['run.py','integration.py','config.json','manifest.json','results/fold_metrics.csv','results/report.md']:
            if (h/n).exists():files.append(h/n)
    for p in files:protected[str(p)]=sha(p)
    m=read(EXP/'manifest.json');m.update(status='PREPARED',protected_sha256=protected,source_signature=signature(),official_source=read(RT/'official/source_audit.json'));write(EXP/'manifest.json',m)
    print('PREPARED',len(protected),'protected files',flush=True)

def normalize_historical_anchors():
    t=runtime()['torch'];refs={int(r['fold']):r for r in csvread(E32/'results/fold_metrics.csv') if r['config']=='B0'}
    for base in ['A','B']:
        for f in range(1,11):
            source=HA/'.runtime/anchors'/f'anchor_fold{f:02}.pt' if base=='A' else HB/'.runtime/anchors'/f'fold{f:02}.pt'
            v=load(source);s=dict(protocol='H',base=base,fold=f,split_sha=foldsha(f),model=v['model'],mp=v['mp'],mf=v['mf'],entry_rng=v['entry_rng'],rng=v['post_rng'] if base=='A' else v['rng'],source=str(source),source_sha=sha(source))
            if base=='A':
                assert v['initial_trunk_sha256']==refs[f]['initial_trunk_sha256'] and v['initial_rng_sha256']==refs[f]['initial_rng_sha256']==rngsha(s['entry_rng'])
                runtime()['network']._prompt_text_features=s['model']['pln']['prompt_text_features'].to('cuda')
            restore(s['entry_rng']);a=fresh(f'H-{base}-G',f);setmodel(a,s['model']);s['parameter_sha']=digest_state(a)
            if base=='A':assert histsha(a)==refs[f]['initial_trunk_sha256']
            else:assert digest_state(a)==v['parameter_sha'] and equal(rng(),s['rng'])
            assert not s['mp']['state'] and not s['mf']['state'];path=anchorpath(f'H-{base}-G',f)
            if path.exists():assert equal(load(path),s)
            else:save(path,s)
    print('HISTORICAL_UNTRAINED_ANCHORS_VERIFIED 20/20',flush=True)

def reuse(gpu):
    t=runtime()['torch'];checks=[];diags=[]
    refs32={int(r['fold']):r for r in csvread(E32/'results/fold_metrics.csv') if r['config']=='B0'}
    hrows={int(r['fold']):r for r in csvread(HA/'results/fold_metrics.csv') if r['group']=='E00_B0A_REPLAY'}
    for f in (range(1,6) if gpu==0 else range(6,11)):
        for group in ['H-A-G','H-B-G','H-B-R']:
            path=HA/'.runtime/checkpoints'/f'E00_B0A_REPLAY_seed0_fold{f:02}_best.pt' if group=='H-A-G' else HB/'.runtime/checkpoints'/f'g{0 if group.endswith("G") else 1}_f{f:02}.best.pt'
            old=load(path);a,s=fromanchor(group,f);setmodel(a,old['model']);out=evaluate(a)[0];metrics=metric_row(out,data()[1],a.tst_idx);pred=out[a.tst_idx].argmax(1).cpu().numpy().astype(np.int64)
            if group=='H-A-G':
                r=hrows[f];rr=refs32[f]
                for k,z in [('acc','acc'),('hard_auc','auc'),('spe','spe_td'),('sen','sen_asd')]:assert metrics[k]==float(rr[z])==float(r[k if k in r else z])
                for k in ['asd_tp','asd_fn','td_fp','td_tn','best_epoch','epochs_executed']:assert int(r[k])==int(rr[k])
                assert s['source_sha']==r['anchor_sha'] and np.array_equal(pred,old['pred']) and sha(path)==r['best_checkpoint_sha256']
                assert rngsha(s['entry_rng'])==r['initial_rng_sha256'] and histsha(a)!=rr['initial_trunk_sha256']
                # Historical E43 retained best models but not best Adam state;
                # the paired entry two-Adam state is fully available and empty.
                best_epoch=int(r['best_epoch']);epochs=int(r['epochs_executed']);history_task=HA/'.runtime/tasks'/f'E00_B0A_REPLAY_seed0_fold{f:02}.json'
                history=dict(source=str(history_task),metrics_csv=str(HA/'results/fold_metrics.csv'),historical_e32=str(E32/'results/fold_metrics.csv'),best_adam_available=False)
            else:
                r=read(HB/'.runtime/tasks'/f'g{0 if group.endswith("G") else 1}_f{f:02}.json');assert r['status']=='COMPLETE' and sha(path)==r['best_sha']
                assert metrics==old['metrics'] and all(metrics[k]==r[k] for k in metrics) and np.array_equal(pred,old['prediction'])
                assert rngsha(s['entry_rng'])==r['entry_full_rng_sha256'] and s['parameter_sha']==r['initial_parameter_sha256']
                best_epoch=int(r['best_epoch']);epochs=int(r['epochs_executed']);history=dict(source=str(HB/'.runtime/tasks'/f'g{0 if group.endswith("G") else 1}_f{f:02}.json'),source_sha=sha(HB/'.runtime/tasks'/f'g{0 if group.endswith("G") else 1}_f{f:02}.json'),best_adam_available=True)
                if group.endswith('R'):
                    assert equal(old['model']['gfr'],s['model']['gfr']) and equal(old['mp'],s['mp']) and r['gfr_calls']==r['gcn_calls']==r['graph_loss_calls']==0
            row=dict(group=group,protocol='H',base=GROUPS[group]['base'],gfr=GROUPS[group]['gfr'],seed=0,fold=f,origin='REUSED_VERIFIED',**metrics,best_epoch=best_epoch,epochs_executed=epochs,anchor_sha=sha(anchorpath(group,f)),split_sha=foldsha(f),initial_parameter_sha256=s['parameter_sha'],entry_full_rng_sha256=rngsha(s['entry_rng']),post_initial_rng_sha256=rngsha(s['rng']),initial_pln_sha=statesha(s['model']['pln']),best_checkpoint=str(path),best_sha=sha(path),prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),prediction=pred.tolist(),physical_gpu=int(r['physical_gpu']),status='COMPLETE',independent_acceptance='PASS',source_signature=signature(),source_history=history)
            write(RT/'tasks'/f'{group}_f{f:02}.json',row);diags.extend(diagnostics(a,f));checks.append(dict(group=group,fold=f,status='PASS'))
            print('REUSED_VERIFIED',group,f,flush=True)
    write(RT/f'reuse_gpu{gpu}.json',dict(status='PASS',checks=checks,diagnostics=diags))

def preflight(gpu):
    t=runtime()['torch'];checks={};started=time.monotonic();f=1 if gpu==0 else 6
    for base in ['A','B']:
        for protocol in ['H','P']:
            gg=f'{protocol}-{base}-G';rr=f'{protocol}-{base}-R'
            if protocol=='H':a,s=fromanchor(gg,f);b,_=fromanchor(rr,f)
            else:
                if base=='A':runtime()['network']._prompt_text_features=load(anchorpath('H-A-G',f))['model']['pln']['prompt_text_features'].to('cuda')
                seed_everything(t,0);entry=copy.deepcopy(rng());a=fresh(gg,f);s=dict(model=model(a),mp=copy.deepcopy(a.optimizer_MP0.state_dict()),mf=copy.deepcopy(a.optimizer_MF.state_dict()),rng=copy.deepcopy(rng()))
                restore(entry);b=fresh(rr,f);assert equal(model(b),s['model']) and equal(rng(),s['rng'])
            assert equal(model(a),model(b)) and equal(a.optimizer_MF.state_dict(),b.optimizer_MF.state_dict())
            checks[gg+'_parameters']={'GFR':sum(p.numel() for p in a.MessagePassing0.parameters()),'PLN':sum(p.numel() for p in a.ModalFusion.parameters()),'CAM_width':a.hyperpm.n_hidden*a.hyperpm.n_head,'CAM_flat':4*a.hyperpm.n_hidden*a.hyperpm.n_head}
            restore(s['rng']);out=evaluate(b)[0];assert equal(model(b),s['model'])
            seen=[];hook=b.ModalFusion.InputLayer.register_forward_pre_hook(lambda m,args:seen.append(t.equal(args[0],b.feat)))
            restore(s['rng']);trainstep(b);hook.remove();assert all(seen) and equal(model(b)['gfr'],s['model']['gfr']) and equal(b.optimizer_MP0.state_dict(),s['mp'])
            assert b.gfr_calls==b.gcn_calls==b.graph_calls==0 and not equal(model(b)['pln'],s['model']['pln'])
            if base=='A':assert all(p.grad is None for p in b.ModalFusion.MLP.parameters())
            checks[rr+'_raw_identity_no_gfr_calls_gradients_updates']=True
            for z in [a,b]:
                out=evaluate(z)[0];labels=z.targ.clone();z.targ[z.tst_idx]=1-z.targ[z.tst_idx];assert t.equal(out,evaluate(z)[0]);z.targ=labels
                trainstep(z);trainstep(z);ss=dict(model=model(z),mp=copy.deepcopy(z.optimizer_MP0.state_dict()),mf=copy.deepcopy(z.optimizer_MF.state_dict()),rng=copy.deepcopy(rng()))
                path=RT/'checkpoints'/f'preflight_{gpu}.pt';save(path,ss);trainstep(z);expected=model(z);ops=copy.deepcopy([z.optimizer_MP0.state_dict(),z.optimizer_MF.state_dict()]);end=copy.deepcopy(rng())
                q=fresh(z.group,f);saved=load(path);setmodel(q,saved['model']);q.optimizer_MP0.load_state_dict(saved['mp']);q.optimizer_MF.load_state_dict(saved['mf']);restore(saved['rng']);trainstep(q)
                assert equal(expected,model(q)) and equal(ops,[q.optimizer_MP0.state_dict(),q.optimizer_MF.state_dict()]) and equal(end,rng());path.unlink()
            checks[gg+'_and_RAW_disk_resume_third_step_bitwise']=True
    # Forward, all gradients, both Adam steps against a direct transcription of root simple-2.
    a,s=fromanchor('H-A-G',f);b,_=fromanchor('H-A-G',f)
    restore(s['rng']);trainstep(a);expected=model(a);gradient=grads(a);end=copy.deepcopy(rng())
    restore(s['rng']);b.MessagePassing0.train();b.ModalFusion.train();b.optimizer_MP0.zero_grad();b.optimizer_MF.zero_grad();n,parts=b.dataLayer(b.feat);feat,adjs,al=b.MessagePassing0(parts,n,b.targ,b.trn_idx);out=b.ModalFusion(feat)[0]
    cl=runtime()['utils'].ClsLoss(out,b.targ,b.trn_idx,b.weight);gl=sum(runtime()['utils'].GraphConstructLoss(parts[i],adjs[i],1.,.5,0.) for i in range(4))/4;(cl+gl).backward();b.optimizer_MP0.step();b.optimizer_MF.step()
    assert equal(expected,model(b)) and equal(gradient,grads(b)) and equal(a.optimizer_MP0.state_dict(),b.optimizer_MP0.state_dict()) and equal(a.optimizer_MF.state_dict(),b.optimizer_MF.state_dict()) and equal(end,rng())
    checks['H_A_G_root_simple2_grad_both_adam_rng_bitwise']=True
    checks.update(status='PASS',source_signature=signature(),gpu=gpu,hardware=str(t.cuda.get_device_properties(0)),seconds=time.monotonic()-started,peak_allocated_bytes=t.cuda.max_memory_allocated())
    write(RT/f'preflight_gpu{gpu}.json',checks);print('PREFLIGHT_PASS',gpu,flush=True)

def completed(group,f):
    p=RT/'tasks'/f'{group}_f{f:02}.json'
    if not p.exists():return False
    r=read(p)
    if r.get('status')!='COMPLETE':return False
    assert r['source_signature']==signature() and r['split_sha']==foldsha(f) and sha(r['best_checkpoint'])==r['best_sha']
    return True
def execute(group,f,gpu,constructed=None):
    t=runtime()['torch'];key=f'{group}_f{f:02}';path=RT/'tasks'/f'{key}.json';lock=(RT/'tasks'/f'{key}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if completed(group,f):return load(RT/'checkpoints'/f'{key}.best.pt')['terminal_rng']
    a,s=constructed if constructed is not None else fromanchor(group,f);hp=a.hyperpm;start=time.monotonic();latest=RT/'checkpoints'/f'{key}.latest.pt';bestpath=RT/'checkpoints'/f'{key}.best.pt'
    base=dict(group=group,protocol=GROUPS[group]['protocol'],base=GROUPS[group]['base'],gfr=a.gfr,seed=0,fold=f,physical_gpu=gpu,origin='NEWLY_TRAINED',source_signature=signature(),split_sha=foldsha(f),anchor_sha=sha(anchorpath(group,f)),initial_parameter_sha256=s['parameter_sha'],entry_full_rng_sha256=rngsha(s['entry_rng']),post_initial_rng_sha256=rngsha(s['rng']),initial_pln_sha=statesha(s['model']['pln']))
    begin=0;best=None;bestacc=0.;bestepoch=-1;wait=0;elapsed=0.;trace=[];rngtrace=[];resume_count=0
    if latest.exists():
        v=load(latest);assert v['base']==base;setmodel(a,v['model']);a.optimizer_MP0.load_state_dict(v['mp']);a.optimizer_MF.load_state_dict(v['mf']);restore(v['rng']);begin=v['epoch']+1;best=v['best'];bestacc=v['bestacc'];bestepoch=v['bestepoch'];wait=v['wait'];elapsed=v['elapsed'];trace=v['trace'];rngtrace=v['rngtrace'];resume_count=v['resume_count']+1
        a.gfr_calls,a.gcn_calls,a.graph_calls=v['calls'];print('RESUME',key,begin,flush=True)
    write(path,dict(**base,status='RUNNING',epoch=begin));epoch=begin-1
    for epoch in (range(begin,hp.nepoch) if wait<=hp.early else []):
        losses=trainstep(a);out=evaluate(a)[0];acc=float(out[a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean())
        if acc>bestacc:
            bestacc=acc;bestepoch=epoch;wait=0;best=dict(model=model(a),mp=copy.deepcopy(a.optimizer_MP0.state_dict()),mf=copy.deepcopy(a.optimizer_MF.state_dict()),rng=copy.deepcopy(rng()),epoch=epoch)
        else:wait+=1
        rngtrace.append(rngsha(rng()))
        if epoch%10==0 or wait>hp.early or epoch==449:trace.append(dict(epoch=epoch,**losses,test_acc=acc,best_acc=bestacc))
        if (epoch+1)%CFG['checkpoint_every']==0 or wait>hp.early or epoch==449:
            save(latest,dict(base=base,model=model(a),mp=a.optimizer_MP0.state_dict(),mf=a.optimizer_MF.state_dict(),rng=rng(),epoch=epoch,best=best,bestacc=bestacc,bestepoch=bestepoch,wait=wait,elapsed=elapsed+time.monotonic()-start,trace=trace,rngtrace=rngtrace,resume_count=resume_count,calls=[a.gfr_calls,a.gcn_calls,a.graph_calls]))
        if wait>hp.early:break
    assert best is not None;terminal=copy.deepcopy(rng());setmodel(a,best['model']);out=evaluate(a)[0];metrics=metric_row(out,data()[1],a.tst_idx);prediction=out[a.tst_idx].argmax(1).cpu().numpy().astype(np.int64)
    if not a.gfr:
        assert equal(model(a)['gfr'],s['model']['gfr']) and equal(a.optimizer_MP0.state_dict(),s['mp']) and a.gfr_calls==a.gcn_calls==a.graph_calls==0
    artifact=dict(**base,**best,terminal_rng=terminal,prediction=prediction,test_indices=a.tst_idx,metrics=metrics,best_epoch=bestepoch,epochs_executed=epoch+1,trace=trace,rng_trace=rngtrace)
    save(bestpath,artifact);v=load(bestpath);assert equal(v['model'],best['model']) and np.array_equal(v['prediction'],prediction)
    row=dict(**base,**metrics,best_epoch=bestepoch,epochs_executed=epoch+1,best_checkpoint=str(bestpath),best_sha=sha(bestpath),prediction_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),prediction=prediction.tolist(),model_sha=statesha(best['model']),status='COMPLETE',independent_acceptance='PENDING',wall_seconds=elapsed+time.monotonic()-start,early_stop_reason='wait>200' if wait>hp.early else '450 epoch cap',resumes=resume_count,gfr_calls=a.gfr_calls,gcn_calls=a.gcn_calls,graph_calls=a.graph_calls,rng_trace=rngtrace,loss_trace=trace)
    write(path,row);latest.unlink(missing_ok=True);restore(terminal);print('FOLD_DONE',key,'gpu',gpu,'ACC',metrics['acc'],'best',bestepoch,'executed',epoch+1,'seconds',round(row['wall_seconds'],1),flush=True)
    return terminal
def worker(group,gpu):
    assert read(RT/f'preflight_gpu{gpu}.json')['status']=='PASS'
    if group=='H-A-R':
        for f in (range(1,6) if gpu==0 else range(6,11)):execute(group,f,gpu)
        return
    assert group in ['P-A-G','P-B-G','P-A-R','P-B-R']
    if group.endswith('R'):
        for f in range(1,11):execute(group,f,gpu)
        return
    t=runtime()['torch']
    if GROUPS[group]['base']=='A':setup_legacy_loader()
    seed_everything(t,0)
    for f in range(1,11):
        if completed(group,f):
            if GROUPS[group]['base']=='A':runtime()['network']._prompt_text_features=load(anchorpath(group,f))['model']['pln']['prompt_text_features'].to('cuda')
            restore(load(RT/'checkpoints'/f'{group}_f{f:02}.best.pt')['terminal_rng']);continue
        ap=anchorpath(group,f)
        if ap.exists():a,s=fromanchor(group,f)
        else:
            entry=copy.deepcopy(rng());a=fresh(group,f);s=dict(protocol='P',base=GROUPS[group]['base'],fold=f,split_sha=foldsha(f),entry_rng=entry,rng=copy.deepcopy(rng()),model=model(a),mp=copy.deepcopy(a.optimizer_MP0.state_dict()),mf=copy.deepcopy(a.optimizer_MF.state_dict()),parameter_sha=digest_state(a));save(ap,s)
        execute(group,f,gpu,(a,s))

def diagnostics(a,f):
    t=runtime()['torch'];group=a.group
    if not a.gfr:return [dict(group=group,fold=f,modality='RAW',gfr_calls=0,gcn_calls=0,graph_loss_calls=0,input_norm=float(a.feat.norm()),pln_input_equals_raw=True)]
    g=a.MessagePassing0;capture={};disabled=set();hooks=[]
    for i in range(4):
        def hook(m,inp,out,i=i):capture[i]=out.detach().clone();return t.zeros_like(out) if i in disabled else out
        hooks.append(getattr(g,f'MessagePassing{i}').register_forward_hook(hook))
    try:
        out,_,_,adjs,_,feat=evaluate(a);messages=[capture[i].clone() for i in range(4)];features=feat.detach().clone()
        comp=t.sigmoid(g.fusion_weight[1])*messages[1]+a.feat
        for i in [2,3]:comp=t.sigmoid(g.fusion_weight[i])*messages[i]+comp
        assert t.equal(comp,features)
        _,cl,gl,_,_,_=forward(a);ps=[p for i in range(4) for p in getattr(g,f'MessagePassing{i}').parameters()];gps=[p for i in range(4) for p in getattr(g,f'GraphConstruct{i}').p.parameters()]
        cg=t.autograd.grad(cl,ps,retain_graph=True,allow_unused=True);gg=t.autograd.grad(gl,gps,allow_unused=True)
        norms=lambda v:float(sum((q.square().sum() for q in v if q is not None),a.feat.new_zeros(())).sqrt());rows=[];N=len(a.feat)
        for i,name in enumerate(['PHENO','ANAT','FUNC','CORR']):
            disabled.add(i);off=evaluate(a)[0];disabled.clear();mask=adjs[i]>0;mask.fill_diagonal_(False);degree=mask.sum(1).float();fw=float(t.sigmoid(g.fusion_weight[i]));effective=fw*float(messages[i].norm()) if i else 0.
            rows.append(dict(group=group,fold=f,modality=name,input_norm=float(a.feat.norm()),gfr_output_norm=float(features.norm()),gfr_minus_raw_relative=float((features-a.feat).norm()/a.feat.norm()),message_norm=float(messages[i].norm()),fusion_sigmoid=fw,effective_contribution_input_ratio=effective/float(a.feat.norm()),classification_gcn_grad_norm=norms(cg[2*i:2*i+2]),graph_projection_grad_norm=norms(gg[2*i:2*i+2]),edge_density=float(mask.sum())/(N*(N-1)),mean_neighbors=float(degree.mean()),min_neighbors=int(degree.min()),max_neighbors=int(degree.max()),branch_off_logp_abs_mean=float((out[a.tst_idx]-off[a.tst_idx]).abs().mean()),branch_off_flip_count=int((out[a.tst_idx].argmax(1)!=off[a.tst_idx].argmax(1)).sum()),overwrite_verified=True))
        disabled.update(range(4));off=evaluate(a)[0]
        for r in rows:r.update(all_off_logp_abs_mean=float((out[a.tst_idx]-off[a.tst_idx]).abs().mean()),all_off_flip_count=int((out[a.tst_idx].argmax(1)!=off[a.tst_idx].argmax(1)).sum()))
        assert rows[0]['classification_gcn_grad_norm']==rows[0]['branch_off_logp_abs_mean']==0
    finally:
        for h in hooks:h.remove()
    return rows
def audit_worker(gpu):
    t=runtime()['torch'];diags=[];checks=[]
    for z in CFG['groups']:
        group=z['id']
        if z['origin']=='REUSED_VERIFIED':continue
        fs=(range(1,6) if gpu==0 else range(6,11)) if group=='H-A-R' else (range(1,11) if (0 if z['base']=='A' else 1)==gpu else [])
        for f in fs:
            assert completed(group,f);path=RT/'tasks'/f'{group}_f{f:02}.json';r=read(path);best=load(r['best_checkpoint']);a,s=fromanchor(group,f);setmodel(a,best['model']);a.optimizer_MP0.load_state_dict(best['mp']);a.optimizer_MF.load_state_dict(best['mf']);out=evaluate(a)[0];pred=out[a.tst_idx].argmax(1).cpu().numpy().astype(np.int64)
            assert metric_row(out,data()[1],a.tst_idx)==best['metrics'] and all(r[k]==best['metrics'][k] for k in best['metrics'])
            assert np.array_equal(pred,best['prediction']) and hashlib.sha256(pred.tobytes()).hexdigest()==r['prediction_sha256'] and statesha(best['model'])==r['model_sha']
            if not a.gfr:assert equal(best['model']['gfr'],s['model']['gfr']) and equal(best['mp'],s['mp']) and r['gfr_calls']==r['gcn_calls']==r['graph_calls']==0
            diags.extend(diagnostics(a,f));r['independent_acceptance']='PASS';write(path,r);checks.append(dict(group=group,fold=f,status='PASS'));print('INDEPENDENT_ACCEPT',group,f,flush=True)
    write(RT/f'final_audit_gpu{gpu}.json',dict(status='PASS',checks=checks,diagnostics=diags,peak_allocated_bytes=t.cuda.max_memory_allocated()))
def launch(command,gpu,label):
    p=RT/'logs'/f'{label}.log';log=p.open('a',buffering=1)
    child=subprocess.Popen([sys.executable,'-B','-u',str(EXP/'run.py'),*command,'--gpu',str(gpu)],cwd=EXP,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return child,log,label
def wait_jobs(jobs):
    while jobs:
        time.sleep(3)
        for item in jobs[:]:
            p,log,label=item
            if p.poll() is not None:
                log.close();jobs.remove(item)
                if p.returncode:raise RuntimeError('Failed '+label+'; other isolated workers can finish safely; inspect .runtime/logs')
def controller():
    lock=(RT/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_protected()
    for g in [0,1]:assert read(RT/f'preflight_gpu{g}.json')['status']=='PASS' and read(RT/f'reuse_gpu{g}.json')['status']=='PASS'
    m=read(EXP/'manifest.json');assert m['source_signature']==signature();m.update(status='H_A_R_RUNNING',formal_started_unix=m.get('formal_started_unix',time.time()));write(EXP/'manifest.json',m)
    wait_jobs([launch(['worker','--group','H-A-R'],g,f'H-A-R_gpu{g}') for g in [0,1]])
    m=read(EXP/'manifest.json');m.update(status='P_G_RUNNING',new_folds_complete=10,reused_folds_complete=30);write(EXP/'manifest.json',m)
    wait_jobs([launch(['worker','--group',f'P-{b}-G'],g,f'P-{b}-G') for g,b in enumerate(['A','B'])])
    m=read(EXP/'manifest.json');m.update(status='P_R_RUNNING',new_folds_complete=30);write(EXP/'manifest.json',m)
    wait_jobs([launch(['worker','--group',f'P-{b}-R'],g,f'P-{b}-R') for g,b in enumerate(['A','B'])])
    m=read(EXP/'manifest.json');m.update(status='INDEPENDENT_AUDIT',new_folds_complete=50,formal_finished_unix=time.time());m['formal_wall_seconds']=m['formal_finished_unix']-m['formal_started_unix'];write(EXP/'manifest.json',m)
    wait_jobs([launch(['audit-worker'],g,f'final_audit_gpu{g}') for g in [0,1]])
    verify_protected();m=read(EXP/'manifest.json');m.update(status='ALL_80_VERIFIED',scientific='COMPLETE',groups_complete=8,folds_complete=80,new_folds_complete=50,reused_folds_complete=30,gpu_new_folds={'0':25,'1':25});write(EXP/'manifest.json',m)
    print('ALL_80_VERIFIED NEW50 REUSED30 GPU25_25',flush=True)

def append_handoff(text):
    p=ROOT/'handoff.md';pending=EXP/'handoff_append_pending.md';marker='<!-- EXPERIMENT43_3_PAPER_GFR_ABLATION -->';m=read(EXP/'manifest.json')
    try:
        with p.open('r+b') as f:
            fcntl.flock(f,fcntl.LOCK_EX);before=f.read();prefix=m['handoff_prefix']
            assert hashlib.sha256(before[:prefix['bytes']]).hexdigest()==prefix['sha256'],'Historical handoff prefix changed'
            oldsha=hashlib.sha256(before).hexdigest()
            if marker.encode() in before:return dict(status='ALREADY_APPENDED',prefix_sha256=oldsha)
            f.seek(0,2);f.write(text.encode());f.flush();os.fsync(f.fileno());f.seek(0);after=f.read()
            assert after[:len(before)]==before and hashlib.sha256(after[:len(before)]).hexdigest()==oldsha
        pending.unlink(missing_ok=True);return dict(status='APPENDED',prefix_bytes=len(before),prefix_sha256=oldsha,after_sha256=hashlib.sha256(after).hexdigest())
    except Exception as e:
        atomic(pending,(text+'\nAppend failure: '+repr(e)+'\n').encode());return dict(status='PENDING',error=repr(e))

def report():
    m=read(EXP/'manifest.json');rows=[]
    for z in CFG['groups']:
        for f in range(1,11):
            p=RT/'tasks'/f"{z['id']}_f{f:02}.json"
            if p.exists():
                r=read(p)
                if r.get('status')=='COMPLETE':assert completed(z['id'],f);rows.append(r)
    assert len({(r['group'],r['seed'],r['fold']) for r in rows})==len(rows)
    complete=len(rows)==80 and all(r['independent_acceptance']=='PASS' for r in rows)
    csvwrite(RESULTS/'fold_metrics.csv',[{k:v for k,v in r.items() if k not in ['rng_trace','loss_trace','source_history','prediction']} for r in rows])
    summaries=[];by={z['id']:{r['fold']:r for r in rows if r['group']==z['id']} for z in CFG['groups']}
    for z in CFG['groups']:
        rr=list(by[z['id']].values())
        if len(rr)!=10:continue
        s=dict(group=z['id'],origin=z['origin'],folds=10)
        for k in CFG['metrics']:
            v=100*np.array([r[k] for r in rr]);s[k+'_mean']=float(v.mean());s[k+'_sd']=float(v.std(ddof=0))
        for k in ['asd_tp','asd_fn','td_fp','td_tn']:s[k+'_total']=sum(r[k] for r in rr)
        s['correct_total']=s['asd_tp_total']+s['td_tn_total'];s['weighted_micro_acc']=100*s['correct_total']/871
        s['best_epochs']=','.join(str(r['best_epoch']) for r in sorted(rr,key=lambda r:r['fold']));s['executed_epochs']=','.join(str(r['epochs_executed']) for r in sorted(rr,key=lambda r:r['fold']))
        summaries.append(s)
    csvwrite(RESULTS/'summary.csv',summaries);stats={r['group']:r for r in summaries};paired=[]
    pairs=[(f'{p}-{b}-G',f'{p}-{b}-R','GFR') for p in ['H','P'] for b in ['A','B']]+[(f'{p}-A-{g}',f'{p}-B-{g}','BASE') for p in ['H','P'] for g in ['G','R']]
    for lhs,rhs,kind in pairs:
        if len(by[lhs])!=10 or len(by[rhs])!=10:continue
        ds={k:np.array([100*(by[lhs][f][k]-by[rhs][f][k]) for f in range(1,11)]) for k in CFG['metrics']}
        q=dict(lhs=lhs,rhs=rhs,kind=kind)
        for k,v in ds.items():q['delta_'+k+'_pp']=float(v.mean());q[k+'_wins']=int((v>1e-10).sum());q[k+'_ties']=int((np.abs(v)<=1e-10).sum());q[k+'_losses']=int((v< -1e-10).sum());q[k+'_fold_deltas']=','.join(f'{x:.8f}' for x in v)
        if kind=='GFR':
            for f in range(1,11):
                l,r=by[lhs][f],by[rhs][f]
                for k in ['anchor_sha','initial_parameter_sha256','initial_pln_sha','entry_full_rng_sha256','post_initial_rng_sha256','split_sha']:assert l[k]==r[k],(lhs,rhs,f,k)
                if 'rng_trace' in l and 'rng_trace' in r:
                    common=min(len(l['rng_trace']),len(r['rng_trace']));assert l['rng_trace'][:common]==r['rng_trace'][:common]
            q['paired_initialization']='PASS_10_OF_10'
            q['decision']='GO' if q['delta_acc_pp']>=.1 and q['delta_hard_auc_pp']>=.1 and q['acc_wins']>=7 and q['hard_auc_wins']>=7 else 'NO-GO'
        else:q['decision']='DESCRIPTIVE_CONSTRUCTION_RNG_DIFFERENCE'
        paired.append(q)
    csvwrite(RESULTS/'paired_comparisons.csv',paired);diags=[]
    for name in ['reuse_gpu0.json','reuse_gpu1.json','final_audit_gpu0.json','final_audit_gpu1.json']:
        if (RT/name).exists():diags.extend(read(RT/name)['diagnostics'])
    csvwrite(RESULTS/'mechanism_diagnostics.csv',diags)
    tab=['|组别|来源|ACC|Hard-AUC|SPE-TD|SEN-ASD|','|---|---|---:|---:|---:|---:|']
    for s in summaries:tab.append('|'+s['group']+'|'+s['origin']+'|'+'|'.join(f"{s[k+'_mean']:.4f} ± {s[k+'_sd']:.4f}" for k in CFG['metrics'])+'|')
    lines=['# Experiment 43.3 — 原论文GFR消融专项复现核查','',f"状态：{'SCIENTIFIC_COMPLETE' if complete else 'PARTIAL'}；实际{len(rows)}/80条唯一折记录。新增{sum(r['origin']=='NEWLY_TRAINED' for r in rows)}，核验复用{sum(r['origin']=='REUSED_VERIFIED' for r in rows)}。",'', '正式四指标：seed0十折mean ± population SD，单位%；ACC为折均值，并非全871人加权micro。后者单独列在summary.csv。','',*tab,'','## 原论文与双基座比较','','|来源/协议|基座|无GFR ACC|有GFR ACC|ΔACC|无GFR AUC|有GFR AUC|ΔAUC|','|---|---|---:|---:|---:|---:|---:|---:|','|Figure 2图中标注|作者未公开完整消融实现|86.48|91.73|5.25|86.40|91.54|5.14|']
    for p in ['H','P']:
        for b in ['A','B']:
            g,r=f'{p}-{b}-G',f'{p}-{b}-R'
            if g in stats and r in stats:
                x,y=stats[g],stats[r];lines.append(f"|{p}|B0-{b}|{y['acc_mean']:.4f}|{x['acc_mean']:.4f}|{x['acc_mean']-y['acc_mean']:+.4f}|{y['hard_auc_mean']:.4f}|{x['hard_auc_mean']:.4f}|{x['hard_auc_mean']-y['hard_auc_mean']:+.4f}|")
    lines+=['','Figure 2(c) +RES：ACC92.08、AUC91.93；(d) +PEN：ACC92.30、AUC92.22。以上全部来自PDF第7页原图数字标签，非本轮结果；本轮不训练(c)/(d)。正文Sec3.3明确baseline只用CAM。作者公开树中没有Figure2专用消融实现、消融初始化/训练顺序/种子。标记：PAPER_ABLATION_IMPLEMENTATION_UNAVAILABLE。','', '## 同协议同基座GFR配对','','|比较|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|ACC W/T/L|AUC W/T/L|决策|','|---|---:|---:|---:|---:|---|---|---|']
    for q in paired:
        lines.append('|'+q['lhs']+' − '+q['rhs']+'|'+'|'.join(f"{q['delta_'+k+'_pp']:+.4f}" for k in CFG['metrics'])+f"|{q['acc_wins']}/{q['acc_ties']}/{q['acc_losses']}|{q['hard_auc_wins']}/{q['hard_auc_ties']}/{q['hard_auc_losses']}|{q['decision']}|")
    lines+=['','GFR准入要求：两项增益均≥0.1pp且两项各≥7/10严格胜。A/B差别含构造与初始化随机路径，不能解释为PEN语义收益；H/P同时改变多个参数，不能归因单一参数。训练折相互重叠，这些胜平负不是独立数据集显著性检验。','', '## 参数映射与确定源码证据','', '见paper_ablation_audit.json逐项paper_text_original/repo_argument/mapping_evidence/confidence/ambiguity。H为官方Shell运行参数；P=dropout .75、lr .0023、n_hidden10、heads4、nlayer4，明确为PAPER_TEXT_MAPPED_PROXY。原文“number of hidden layers10”与代码n_hidden表示Q/K/V每头宽度存在语义歧义；nlayer实际控制Encoder+FeedForward重复次数。P的alpha记为1以对应原文，但代码该参数仅存储，simple-2中原始特征加法硬编码为1；H默认alpha .5与P1均不改变该路径。reg .11、th .9、450/200及其它未明参数继承Shell/default，不是论文确认。','', '官方main/model/layers/shell与本地逐字相同。GraphLearn/GCN/GFR的AST逐值一致；均含i==1覆盖PHENO消息。simple-2的ALoss计算而不优化。官方network把RN50作为ViT_model注册在PLN，且在MLP之前加载；用户network先构造MLP再调用一次文本缓存，缓存8条1024维buffer，不注册整个RN50。这改变构造与apply(my_weight_init)所遍历的参数/RNG。官方utils.one_hot在CPU创建eye，本地改到索引device以兼容新PyTorch；数学指标口径相同。上述是确定实现/配置差别，尚不能单独解释原作者私有消融的5.25pp。','', 'B0-A复用E43已逐折精确匹配E32的10个未训练状态、完整RNG与best；本轮又重新加载这些best并计算预测/四指标/混淆。B0-B复用E43.2已独立验收模型并再次重算。新增无GFR组只恢复未训练入口，不加载有GFR best作为初始化。历史A best只保留模型，best时Adam未保存；入口两个Adam空状态完整存在。该历史限制明确保留，不补造缺失状态。','', '## GFR机制','','|有GFR组|输出−输入相对范数(均值)|有效消息/输入ANAT,FUNC,CORR|图密度PHENO,ANAT,FUNC,CORR|关闭全部消息logp绝对变化|类别翻转(871)|','|---|---:|---|---|---:|---:|']
    for group in ['H-A-G','H-B-G','P-A-G','P-B-G']:
        rr=[r for r in diags if r['group']==group and r['modality']!='RAW']
        if not rr:continue
        per=lambda mod:[r for r in rr if r['modality']==mod]
        ratios=[float(np.mean([r['effective_contribution_input_ratio'] for r in per(mod)])) for mod in ['ANAT','FUNC','CORR']]
        dens=[float(np.mean([r['edge_density'] for r in per(mod)])) for mod in ['PHENO','ANAT','FUNC','CORR']]
        one=per('PHENO');lines.append(f"|{group}|{np.mean([r['gfr_minus_raw_relative'] for r in one]):.8g}|{', '.join(f'{x:.6g}' for x in ratios)}|{', '.join(f'{x:.4f}' for x in dens)}|{np.mean([r['all_off_logp_abs_mean'] for r in one]):.7g}|{sum(r['all_off_flip_count'] for r in one)}|")
    lines+=['','逐分支消息范数、分类GCN梯度、图投影梯度、有效融合、密度/度数及单支关闭logp/翻转见mechanism_diagnostics.csv。PHENO消息虽计算但被覆盖，分类GCN梯度为0；图正则仍优化PHENO GraphLearn。数值有消息、固定模型依赖消息、从头训练带来分类增益是三件事。','', '## 论文数值兼容性与核心回答','']
    for p in ['H','P']:
        for b in ['A','B']:
            g,r=f'{p}-{b}-G',f'{p}-{b}-R'
            if g not in stats or r not in stats:continue
            x,y=stats[g],stats[r];delta=x['acc_mean']-y['acc_mean'];lines.append(f"- {p}/B0-{b}：CAM-only距86.48为{y['acc_mean']-86.48:+.4f}pp，GFR距91.73为{x['acc_mean']-91.73:+.4f}pp，GFR增量距+5.25为{delta-5.25:+.4f}pp。")
    lines+=['','数值接近不等于严格重现作者私有消融；没有原始消融实现与完整初始化记录，不能断言论文错误或作者不当行为。B0-A/B0-B只能在各自协议内作配对增量；跨基座最好/最差拼接禁止。根据四个预注册增量与机制审计判断是否值得继续以GFR作创新基座：若证据不足，优先保留它作为历史兼容比较，并以CAM-only为必要强对照，不以论文5.25pp作为已证实前提。','', '## 训练、恢复与独立验收','']
    if complete:
        for group in GROUPS:
            rr=list(by[group].values());lines.append(f"- {group}：best_epoch={[r['best_epoch'] for r in rr]}；epochs_executed={[r['epochs_executed'] for r in rr]}；来源{GROUPS[group]['origin']}。")
        lines.append(f"- 新训GPU0={sum(r['origin']=='NEWLY_TRAINED' and r['physical_gpu']==0 for r in rows)}折，GPU1={sum(r['origin']=='NEWLY_TRAINED' and r['physical_gpu']==1 for r in rows)}折；正式墙钟{m.get('formal_wall_seconds',0):.2f}s；本轮新训fold wall总和{sum(r.get('wall_seconds',0) for r in rows):.2f}s（并行和I/O，不等于墙钟）。")
    lines+=['- 新训每10epochs原子保存model、两Adam、best、当前epoch/wait、四类RNG、源/数据/配置/anchor/fold签名；完成后独立进程重载best重新预测及指标，并检查原输入/no-GFR零调用零梯度零更新。','- loss/训练ACC/testACC轨迹在新训task的loss_trace，10epoch抽样；历史保留的可读轨迹作为历史证据，不虚构未保存训练曲线。strict first best与wait>200均未更改。','- 原始代码/数据/历史文件protected SHA前后核对；无Git commit/push。','', '## 限制','', 'val=test参与checkpoint选择、全队列标准化及CV前RFE、全871节点transductive协议保持，仅用于内部历史可比；不能视为无泄漏独立临床泛化。单seed与重叠训练折不支持显著性或因果结论。ANAT/FUNC是扫描质量/运动摘要；图是患者图，CAM潜token不是ROI。','']
    text='\n'.join(lines)+'\n';atomic(RESULTS/'report.md',text.encode())
    if complete:
        verify_protected();handoff='\n\n<!-- EXPERIMENT43_3_PAPER_GFR_ABLATION -->\n## Experiment 43.3 — 原论文GFR消融复现专项核查\n\n'+text.split('\n',1)[1]
        result=append_handoff(handoff);m=read(EXP/'manifest.json');m.update(handoff=result,status='COMPLETE' if result['status'] in ['APPENDED','ALREADY_APPENDED'] else 'COMPLETE_WITH_DELIVERY_BLOCKED',delivery='COMPLETE' if result['status'] in ['APPENDED','ALREADY_APPENDED'] else 'PARTIAL',finished_unix=time.time(),protected_files_unchanged=True,scientific='COMPLETE',summary=summaries,paired=paired);m['total_wall_seconds']=m['finished_unix']-m['started_unix'];write(EXP/'manifest.json',m)
    print(text,flush=True)

def cli():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','anchors','reuse','preflight','all','resume','worker','audit-worker','report']);ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--group');args=ap.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='anchors':normalize_historical_anchors()
    elif args.command=='reuse':reuse(args.gpu)
    elif args.command=='preflight':preflight(args.gpu)
    elif args.command in ['all','resume']:controller()
    elif args.command=='worker':worker(args.group,args.gpu)
    elif args.command=='audit-worker':audit_worker(args.gpu)
    elif args.command=='report':report()
if __name__=='__main__':cli()
