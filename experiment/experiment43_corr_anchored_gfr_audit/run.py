#!/usr/bin/env python3
"""E43 prepare/preflight/all/resume/worker/report, paired immutable initialization."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import argparse, contextlib, copy, csv, datetime, fcntl, hashlib, io, json, pickle, random
import signal, subprocess, time, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/data/adodas/SPromptGL-main')
RT, RESULTS = HERE/'.runtime', HERE/'results'
CFG = json.loads((HERE/'config.json').read_text())
GROUPS = CFG['groups']
CTX = DATA = None


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(b):return hashlib.sha256(b).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path, value):
    p=Path(path);tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    with tmp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def save(path,value):
    import torch
    p=Path(path);tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def load(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=False)
def csvread(path):
    if not Path(path).exists():return []
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def csvwrite(path,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r)) or ['group','seed','fold']
    p=Path(path);tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def data():
    global DATA
    if DATA is None:
        import pandas as pd
        from sklearn.model_selection import StratifiedKFold
        p=ROOT/'data/ABIDE';frame=pd.read_csv(p/'processed_standard_data.csv')
        mapping=np.load(p/'modal_feat_dict.npy',allow_pickle=True).item()
        assert list(mapping)==['PHENO','ANAT','FUNC','Correlation']
        assert [len(v) for v in mapping.values()]==[48,6,10,256]
        assert list(frame.columns[:-1])==[str(n) for values in mapping.values() for n in values]
        x=frame.iloc[:,:-1].to_numpy(np.float32);y=frame.iloc[:,-1].to_numpy(np.int64)-1
        assert x.shape==(871,320) and np.bincount(y).tolist()==[403,468] and np.isfinite(x).all()
        ends=np.cumsum([0]+[len(v) for v in mapping.values()]);slices=list(zip(ends[:-1].tolist(),ends[1:].tolist()))
        pidx={k:[i for i,n in enumerate(mapping['PHENO']) if n.startswith(prefix)] for k,prefix in [('age','AGE_AT_SCAN_'),('site','SITE_ID_'),('sex','SEX_')]}
        assert [len(pidx[k]) for k in ['age','site','sex']]==[26,20,2]
        for cols in pidx.values():assert np.allclose(x[:,cols].sum(1),1.)
        folds=list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y))
        DATA=dict(x=x,y=y,mapping=mapping,slices=slices,pheno_indices=pidx,folds=folds,
                  split_sha=digest(b''.join(np.concatenate([a,b]).astype(np.int64).tobytes() for a,b in folds)),
                  age=x[:,pidx['age']].argmax(1),site=x[:,pidx['site']].argmax(1),sex=x[:,pidx['sex']].argmax(1))
    return DATA


def rng():
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore(r):
    import torch
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu'].cpu());torch.cuda.set_rng_state_all([t.cpu() for t in r['cuda']])
def rngsha(r):
    h=hashlib.sha256(pickle.dumps((r['python'],r['numpy']),protocol=4))
    for v in [r['cpu']]+r['cuda']:h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()
def equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def states(agent):
    return {k:{n:p.detach().cpu().clone() for n,p in m.state_dict().items()} for k,m in [('gfr',agent.MessagePassing0),('pln',agent.ModalFusion)]}
def setstates(agent,s):agent.MessagePassing0.load_state_dict(s['gfr']);agent.ModalFusion.load_state_dict(s['pln'])
def trunksha(agent):
    h=hashlib.sha256()
    for key,m in [('gfr',agent.MessagePassing0),('pln',agent.ModalFusion)]:
        for n,p in sorted(m.named_parameters()):h.update(key.encode()+b'/'+n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def codehash():return {n:sha(HERE/n) for n in ['run.py','method.py','integration.py','config.json','integration.patch']}
def verify_sources():
    m=read(HERE/'manifest.json')
    for p,h in m['source_sha'].items():assert sha(p)==h,'protected source drift: '+p


def runtime():
    global CTX
    if CTX is None:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        sys.path.insert(0,str(RT/'sources'))
        import model,network,utils
        CTX=dict(torch=torch,model=model,network=network,utils=utils)
    return CTX


def historical():
    return {int(r['fold']):r for r in csvread(ROOT/'experiment/experiment32_disease_contrastive_evidence_routing/results/fold_metrics.csv') if r['config']=='B0'}


def reliability(fold):
    from method import estimate_reliability
    p=RT/'cache'/f'reliability_{fold:02}.json'
    lock=(RT/'cache'/f'reliability_{fold:02}.lock').open('a+')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if p.exists():
            result=read(p);assert result['config_sha']==sha(HERE/'config.json');return result
        d=data();tr,te=d['folds'][fold-1]
        result=estimate_reliability(d['x'],tr,d['y'][tr],fold,d['slices'],d['pheno_indices'])
        assert not set(te)&{n for r in result['inner_records'] for n in r['fit_subjects']+r['validation_subjects']}
        result['config_sha']=sha(HERE/'config.json');write(p,result);return result


def make(group,fold,with_hooks=True):
    import torch
    import method
    c=runtime();d=data();a=load(RT/'anchors'/f'anchor_fold{fold:02}.pt');tr,te=d['folds'][fold-1]
    assert a['fold_hash']==digest(np.concatenate([tr,te]).astype(np.int64).tobytes())
    assert a['initial_trunk_sha256']==historical()[fold]['initial_trunk_sha256']
    assert a['initial_rng_sha256']==historical()[fold]['initial_rng_sha256']==rngsha(a['entry_rng'])
    # The historical legacy buffer is immutable and never encoded or executed here.
    c['network']._prompt_text_features=a['model']['pln']['prompt_text_features'].to('cuda')
    restore(a['entry_rng']);entry=rngsha(rng())
    with contextlib.redirect_stdout(io.StringIO()):
        agent=c['model'].EvalHelper([48,6,10,256],d['x'],d['y'],SimpleNamespace(**CFG['protocol']),tr,te)
    agent.MF_sav.close();agent.GCMP_sav.close()
    setstates(agent,a['model'])
    agent.optimizer_MP0.load_state_dict(copy.deepcopy(a['mp']));agent.optimizer_MF.load_state_dict(copy.deepcopy(a['mf']))
    restore(a['post_rng'])
    assert trunksha(agent)==a['initial_trunk_sha256'] and equal(states(agent),a['model'])
    assert rngsha(rng())==rngsha(a['post_rng']) and entry==a['initial_rng_sha256']
    # Reliability fitting is a separate deterministic CPU procedure. Preserve every
    # training RNG even if a dependency consumes a global stream internally.
    paired_rng=copy.deepcopy(rng())
    q=reliability(fold)['q'] if group in (6,8) else None
    restore(paired_rng)
    before=rngsha(rng())
    if with_hooks:method.install(agent,group,fold,q,d['slices'])
    assert before==rngsha(rng()),'extra initialization polluted RNG'
    return agent,a


def forward(agent,loss=True):
    count,parts=agent.dataLayer(agent.feat)
    feat,graphs,aloss=agent.MessagePassing0(parts,count,agent.targ,agent.trn_idx)
    logp,hidden,attn=agent.ModalFusion(feat)
    graph_loss=0.
    if loss:
        for i in range(4):graph_loss=graph_loss+runtime()['utils'].GraphConstructLoss(parts[i],graphs[i],1.,.5,0.)
        graph_loss=graph_loss/4
    return logp,graph_loss,graphs,feat,aloss
def evaluate(agent):
    import torch
    agent.MessagePassing0.eval();agent.ModalFusion.eval()
    with torch.no_grad():return forward(agent,False)
def metric(agent,out):
    from sklearn.metrics import roc_auc_score,confusion_matrix
    te=agent.tst_idx;y=data()['y'][te];pred=out[te].argmax(1).detach().cpu().numpy()
    tp,fn,fp,tn=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    return dict(acc=float((pred==y).mean()),hard_auc=float(roc_auc_score(np.eye(2)[y],np.eye(2)[pred])),spe_td=tn/(tn+fp),sen_asd=tp/(tp+fn),asd_tp=tp,asd_fn=fn,td_fp=fp,td_tn=tn),pred


def gradients(loss,params,retain=True):
    import torch
    grads=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=retain)
    return [float(g.detach().norm()) if g is not None else 0. for g in grads]


def training_step(agent,group,audit=False):
    import torch
    from method import safe_supervision
    agent.MessagePassing0.train();agent.ModalFusion.train()
    agent.optimizer_MP0.zero_grad();agent.optimizer_MF.zero_grad()
    out,graph,_,_,_=forward(agent)
    cls=runtime()['utils'].ClsLoss(out,agent.targ,agent.trn_idx,agent.weight)
    scores=[getattr(agent.MessagePassing0,f'GraphConstruct{i}').last_score43 for i in range(4)]
    sup=safe_supervision(scores,agent.trn_idx,agent.targ[agent.trn_idx]) if group==2 else out.new_tensor(0.)
    diag={'cls_loss':float(cls.detach()),'graph_loss':float(graph.detach()),'sup_loss':float(sup.detach())}
    if group==2 or audit:
        params=[getattr(agent.MessagePassing0,f'GraphConstruct{i}').p.weight for i in range(4)]
        for name,l in [('cls',cls),('graph',graph)]+([('sup',sup)] if group==2 else []):
            for m,v in enumerate(gradients(l,params)):diag[f'{name}_projection_grad_{m}']=v
    loss=cls+graph+sup
    assert torch.isfinite(loss),'nonfinite total loss'
    loss.backward()
    if audit:
        for name,p in agent.MessagePassing0.named_parameters():
            diag['grad/'+name]=float(p.grad.norm()) if p.grad is not None else 0.
    agent.optimizer_MP0.step();agent.optimizer_MF.step()
    return diag


def compare_anchor(row):
    b=historical()[int(row['fold'])];diff=[]
    for k in ['asd_tp','asd_fn','td_fp','td_tn','best_epoch','epochs_executed']:
        if int(row[k])!=int(b[k]):diff.append(k)
    for k in ['initial_trunk_sha256','initial_rng_sha256']:
        if row[k]!=b[k]:diff.append(k)
    for a,z in [('acc','acc'),('hard_auc','auc'),('spe_td','spe_td'),('sen_asd','sen_asd')]:
        if float(row[a])!=float(b[z]):diff.append(a)
    return diff


def diagnostics(agent,group,fold):
    import torch
    from method import graph_audit,MODALITIES
    d=data();tr,te=d['folds'][fold-1]
    out,_,graphs,feat,_=evaluate(agent);pred=out.argmax(1)
    g=agent.MessagePassing0
    rows=[]
    for i,a in enumerate(graphs):
        row=dict(group=GROUPS[group],seed=0,fold=fold,modality=MODALITIES[i],**graph_audit(a,tr,d['y'][tr],d['age'],d['sex'],d['site']))
        msg=g.e43_messages[i];fw=torch.sigmoid(g.fusion_weight[i]).item()
        active=not(group in (0,2,3) and i==0)
        row.update(fusion_sigmoid=fw,message_norm=float(msg.norm()),contribution_norm=float(msg.norm())*fw if active else 0.,
                   contribution_input_ratio=float(msg.norm())*fw/float(agent.feat.norm()) if active else 0.)
        g.e43_off=i;changed=evaluate(agent)[0];g.e43_off=None
        row.update(branch_off_abs_logit=float((out[te]-changed[te]).abs().mean()),branch_off_class_flip=float((pred[te]!=changed[te].argmax(1)).float().mean()))
        if group>=5:
            rel=reliability(fold);row.update(reliability_delta=json.dumps(rel['delta']),reliability_q=json.dumps(rel['q']))
        if group>=6:
            support=graphs[3]>0;other=a>0
            assert torch.equal(support,other),'auxiliary introduced/dropped CORR support'
            ratio=a[support]/graphs[3][support]
            row.update(corr_support_changed=0,aux_multiplier_mean=float(ratio.mean()),aux_multiplier_min=float(ratio.min()),aux_multiplier_max=float(ratio.max()),actual_q=float(g.e43_q[i]) if i<3 else 0.)
            if i<3:assert float(ratio.max())<=np.exp(.25*float(g.e43_q[i]))+1e-5 and float(ratio.min())>=np.exp(-.25*float(g.e43_q[i]))-1e-5
        rows.append(row)
    for i in range(4):
        for j in range(i+1,4):
            a,b=(graphs[k]>0 for k in [i,j]);inter=int((a&b).sum());union=int((a|b).sum())
            rows[i][f'jaccard_with_{MODALITIES[j]}']=inter/union if union else 1.
    # Classification-only gradients distinguish branch overwrite from graph regularization.
    agent.MessagePassing0.zero_grad();agent.ModalFusion.zero_grad()
    out,gl,_,_,_=forward(agent)
    cls=runtime()['utils'].ClsLoss(out,agent.targ,tr,agent.weight)
    cls.backward()
    for i in range(4):
        mp=getattr(g,f'MessagePassing{i}');gc=getattr(g,f'GraphConstruct{i}')
        rows[i].update(cls_message_grad=float(sum((p.grad.norm()**2 for p in mp.parameters() if p.grad is not None),torch.tensor(0.,device='cuda')).sqrt()),
                       cls_projection_grad=0. if gc.p.weight.grad is None else float(gc.p.weight.grad.norm()),
                       cls_fusion_grad=0. if g.fusion_weight.grad is None else float(g.fusion_weight.grad[i].norm()),
                       combined_weight_grad_none=g.combined_weight.grad is None,
                       graph_w_grad_none=gc.w.weight.grad is None,graph_t_grad_none=gc.t.grad is None,graph_threshold_grad_none=gc.threshold.grad is None)
    return rows


def worker(group,fold,gpu):
    import torch
    m=read(HERE/'manifest.json');assert m['preflight']=='PASS' and m['code_sha']==codehash()
    if group:assert m['anchor_replay']=='ANCHOR_REPLAY_PASS'
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    key=f'{GROUPS[group]}_seed0_fold{fold:02}';task=RT/'tasks'/f'{key}.json';latest=RT/'checkpoints'/f'{key}_latest.pt';bestpath=RT/'checkpoints'/f'{key}_best.pt'
    lock=(RT/'tasks'/f'{key}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if task.exists():
        t=read(task)
        if t['status']=='COMPLETE':assert bestpath.exists() and sha(bestpath)==t['metric']['best_checkpoint_sha256'];return
    started=time.monotonic();agent,a=make(group,fold)
    signature=dict(code=m['code_sha'],input_sha=m['source_sha'][str(ROOT/'data/ABIDE/processed_standard_data.csv')],anchor_sha=sha(RT/'anchors'/f'anchor_fold{fold:02}.pt'),split_sha=data()['split_sha'])
    best=None;bestacc=0.;bestepoch=-1;wait=0;epoch_start=0;history=[];prior=0.;resumes=0
    if latest.exists():
        saved=load(latest);assert saved['signature']==signature
        setstates(agent,saved['model']);agent.optimizer_MP0.load_state_dict(copy.deepcopy(saved['mp']));agent.optimizer_MF.load_state_dict(copy.deepcopy(saved['mf']));restore(saved['rng'])
        best,bestacc,bestepoch,wait=saved['best'],saved['bestacc'],saved['bestepoch'],saved['wait']
        epoch_start=saved['epoch']+1;history=saved['history'];prior=saved['seconds'];resumes=saved['resumes']+1
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    for epoch in range(epoch_start,450):
        history.append(dict(epoch=epoch,**training_step(agent,group,audit=(epoch==0))))
        out=evaluate(agent)[0];te=agent.tst_idx
        acc=int(out[te].argmax(1).eq(agent.targ[te]).sum())/len(te)
        if acc>bestacc:best,bestacc,bestepoch,wait=states(agent),acc,epoch,0
        else:wait+=1
        if (epoch+1)%10==0 or wait>200 or epoch==449 or stop[0]:
            save(latest,dict(model=states(agent),mp=agent.optimizer_MP0.state_dict(),mf=agent.optimizer_MF.state_dict(),rng=rng(),best=best,bestacc=bestacc,bestepoch=bestepoch,wait=wait,epoch=epoch,signature=signature,history=history,seconds=prior+time.monotonic()-started,resumes=resumes))
        if (epoch+1)%100==0:print('PROGRESS',key,epoch+1,bestepoch,bestacc,flush=True)
        if wait>200 or stop[0]:break
    if stop[0]:print('SAFE_STOP',key,flush=True);return
    assert best is not None
    setstates(agent,best);out=evaluate(agent)[0];mt,pred=metric(agent,out)
    assert mt['acc']==bestacc
    row=dict(group=GROUPS[group],seed=0,fold=fold,physical_gpu=gpu,**mt,best_epoch=bestepoch,epochs_executed=epoch+1,
             initial_trunk_sha256=trunksha_from_anchor(a),initial_rng_sha256=a['initial_rng_sha256'],anchor_sha=signature['anchor_sha'],split_sha=signature['split_sha'],
             prediction_sha256=digest(pred.astype(np.int64).tobytes()),resumes=resumes)
    if group==0:
        diff=compare_anchor(row);assert not diff,f'ANCHOR_REPLAY_FAIL fold {fold} {diff}'
        original=load(Path(CFG['anchor']['reference_predictions'].replace('NN',f'{fold:02}')))
        current=states(agent);setstates(agent,original['model']);expected=evaluate(agent)[0][agent.tst_idx].argmax(1).cpu().numpy();setstates(agent,current)
        assert np.array_equal(expected,pred),'ANCHOR_REPLAY_FAIL prediction identity'
    ds=diagnostics(agent,group,fold)
    save(bestpath,dict(model=best,signature=signature,group=GROUPS[group],fold=fold,best_epoch=bestepoch,pred=pred))
    assert equal(load(bestpath)['model'],best),'best checkpoint readback failure'
    row.update(best_checkpoint=str(bestpath),best_checkpoint_sha256=sha(bestpath),seconds=prior+time.monotonic()-started)
    write(task,dict(status='COMPLETE',key=key,metric=row,diagnostics=ds,history=history,finished=now()))
    # Only this fold's validated best and immutable anchor are retained after commit.
    latest.unlink(missing_ok=True)
    print('FOLD_DONE',key,gpu,mt,bestepoch,row['seconds'],flush=True)


def trunksha_from_anchor(a):return a['initial_trunk_sha256']


def prepare():
    import integration
    integration.emit(ROOT,RT/'sources',HERE/'integration.patch')
    d=data();m=read(HERE/'manifest.json');verify_sources()
    m.update(code_sha=codehash(),config_sha=sha(HERE/'config.json'),split_sha=d['split_sha'],source_slices=d['slices'],pheno_indices=d['pheno_indices'])
    write(HERE/'manifest.json',m)
    for n in ['fold_metrics.csv','summary.csv','diagnostics.csv','loss_trace.csv']:
        if not (RESULTS/n).exists():csvwrite(RESULTS/n,[])
    print('PREPARED',m['split_sha'],flush=True)


def preflight():
    import torch
    from method import safe_supervision,refined_graphs,mutual_support
    runtime();checks={};start=time.monotonic()
    for f in range(1,11):
        agent,a=make(0,f);assert trunksha(agent)==historical()[f]['initial_trunk_sha256'];del agent
    checks['all10_immutable_entry_model_rng_split']=True
    a,anchor=make(0,1);b,_=make(0,1,with_hooks=False);seed=copy.deepcopy(anchor['post_rng'])
    for t in range(2):
        restore(seed);d1=training_step(a,0,True);s1=states(a);p1=copy.deepcopy(a.optimizer_MP0.state_dict());p2=copy.deepcopy(a.optimizer_MF.state_dict());post=rng()
        restore(seed);d2=training_step(b,0,True)
        assert equal(s1,states(b)) and equal(p1,b.optimizer_MP0.state_dict()) and equal(p2,b.optimizer_MF.state_dict()) and d1==d2 and equal(post,rng())
        seed=post
    checks['instrumented_original_forward_losses_gradients_two_adams_two_steps_exact']=True
    original,_=make(0,1,with_hooks=False);instrumented,_=make(0,1)
    original_seed=copy.deepcopy(anchor['post_rng'])
    for _ in range(2):
        restore(original_seed)
        with contextlib.redirect_stdout(io.StringIO()):original.run_epoch('simple-2')
        original_post=copy.deepcopy(rng())
        restore(original_seed);training_step(instrumented,0)
        assert equal(states(original),states(instrumented))
        assert equal(original.optimizer_MP0.state_dict(),instrumented.optimizer_MP0.state_dict())
        assert equal(original.optimizer_MF.state_dict(),instrumented.optimizer_MF.state_dict())
        assert equal(original_post,rng())
        original_seed=original_post
    checks['root_original_simple2_two_updates_and_rng_bitwise_equivalent']=True
    restore(seed)
    disk=RT/'cache/preflight_resume.pt';save(disk,dict(model=states(a),mp=copy.deepcopy(a.optimizer_MP0.state_dict()),mf=copy.deepcopy(a.optimizer_MF.state_dict()),rng=copy.deepcopy(rng())))
    training_step(a,0);expected=states(a);post=rng();saved=load(disk)
    setstates(b,saved['model']);b.optimizer_MP0.load_state_dict(copy.deepcopy(saved['mp']));b.optimizer_MF.load_state_dict(copy.deepcopy(saved['mf']));restore(saved['rng']);training_step(b,0)
    assert equal(expected,states(b)) and equal(a.optimizer_MP0.state_dict(),b.optimizer_MP0.state_dict()) and equal(a.optimizer_MF.state_dict(),b.optimizer_MF.state_dict()) and equal(post,rng())
    disk.unlink();checks['disk_resume_third_update_model_two_adams_rng_exact']=True
    agent,anchor=make(0,1);base=training_step(agent,0,True)
    assert base['grad/MessagePassing0.gcn1.W']==0 and base['grad/fusion_weight']>0
    checks['original_branch0_classification_and_message_dead']=True
    agent,anchor=make(1,1);audit=training_step(agent,1,True)
    assert all(audit[f'grad/MessagePassing{i}.gcn1.W']>0 for i in range(4))
    checks['aggregation_fix_all4_message_gradients_nonzero']=True
    agent,_=make(2,1);audit=training_step(agent,2,True)
    assert all(audit[f'sup_projection_grad_{i}']>0 for i in range(4))
    scores=[getattr(agent.MessagePassing0,f'GraphConstruct{i}').last_score43 for i in range(4)]
    s=safe_supervision(scores,agent.trn_idx,agent.targ[agent.trn_idx]);mut=agent.targ.clone();mut[agent.tst_idx]=1-mut[agent.tst_idx]
    assert torch.equal(s,safe_supervision(scores,agent.trn_idx,mut[agent.trn_idx]))
    checks['safe_supervision_all4_projection_gradients_and_test_label_invariance']=True
    agent,_=make(3,1);out,_,graphs,_,_=evaluate(agent)
    assert all(torch.all(g>=0) and torch.all(g.diag()==0) and torch.all((g>0).sum(1)<=96) and torch.allclose(g,g.T,atol=1e-6) for g in graphs)
    repeat=evaluate(agent)[2];assert all(torch.equal(a,b) for a,b in zip(graphs,repeat))
    checks['density_symmetry_budget_nonnegative_no_self_deterministic']=True
    # Train-only OOF reliability is a mechanism proxy, not a fifth classification metric.
    rel=reliability(1);checks['reliability_fold1_q']=rel['q'];checks['reliability_fold1_delta']=rel['delta']
    for g in [4,5,6,7,8,9]:
        agent,_=make(g,1);audit=training_step(agent,g,True);assert all(np.isfinite(v) for v in audit.values())
        if g>=6:
            gg=evaluate(agent)[2];assert all(torch.equal(v>0,gg[3]>0) for v in gg)
        if g==9:assert all(audit[f'grad/GraphConstruct{i}.p.weight']==0 for i in range(3))
    checks['refinement_corr_support_only_bounded_e09_aux_no_gradient']=True
    agent,_=make(0,1);t=time.monotonic();training_step(agent,0);checks['baseline_step_seconds']=time.monotonic()-t
    agent,_=make(3,1);t=time.monotonic();training_step(agent,3);checks['density_step_seconds']=time.monotonic()-t
    checks['gpu_memory_allocated_peak']=int(torch.cuda.max_memory_allocated());checks['hardware']=str(torch.cuda.get_device_properties(0))
    checks['seconds']=time.monotonic()-start
    write(RT/'preflight.json',checks);m=read(HERE/'manifest.json');m.update(preflight='PASS',preflight_evidence=checks,status='PREFLIGHT_PASS',code_sha=codehash());write(HERE/'manifest.json',m)
    print('PREFLIGHT_PASS',checks,flush=True)


def refresh():
    rows=[];diags=[];traces=[];tasks={}
    for p in sorted((RT/'tasks').glob('*.json')):
        t=read(p)
        if t.get('status')!='COMPLETE':continue
        r=t['metric'];best=Path(r['best_checkpoint'])
        assert best.is_file() and sha(best)==r['best_checkpoint_sha256'],'invalid completed artifact '+str(p)
        rows.append(r);diags.extend(t['diagnostics']);tasks[t['key']]={'status':'COMPLETE','best_checkpoint_sha256':r['best_checkpoint_sha256'],'gpu':r['physical_gpu']}
        if r['group']==GROUPS[2]:
            traces.extend(dict(group=r['group'],seed=0,fold=r['fold'],**h) for h in t['history'])
    keys=[(r['group'],r['seed'],r['fold']) for r in rows];assert len(set(keys))==len(keys)
    rows.sort(key=lambda r:(r['group'],int(r['fold'])))
    csvwrite(RESULTS/'fold_metrics.csv',rows);csvwrite(RESULTS/'diagnostics.csv',diags);csvwrite(RESULTS/'loss_trace.csv',traces)
    m=read(HERE/'manifest.json');m.update(tasks=tasks,completed_folds=len(rows),gpu_counts={str(g):sum(int(r['physical_gpu'])==g for r in rows) for g in [0,1]},refreshed=now())
    write(HERE/'manifest.json',m);return rows,diags


def child(args,gpu,label):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='2'
    f=(RT/'logs'/f'{label}.log').open('a')
    p=subprocess.Popen([sys.executable,'-u',str(HERE/'run.py')]+args,cwd=HERE,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    f.close();return p


def dispatch(jobs):
    from collections import deque
    queues={g:deque(j for j in jobs if j[2]==g) for g in [0,1]};running={};started=time.monotonic()
    # Memory measurement in preflight determines the conservative cap.
    m=read(HERE/'manifest.json');cap=min(4,int(m.get('workers_per_gpu',4)))
    while any(queues.values()) or running:
        for gpu in [0,1]:
            while queues[gpu] and sum(j[2]==gpu for j in running.values())<cap:
                group,fold,_=queues[gpu].popleft();label=f'{GROUPS[group]}_fold{fold:02}_gpu{gpu}'
                p=child(['worker','--group',str(group),'--fold',str(fold),'--gpu',str(gpu)],gpu,label)
                running[p]=(group,fold,gpu);print('LAUNCH',label,p.pid,flush=True)
        for p,j in list(running.items()):
            code=p.poll()
            if code is None:continue
            del running[p]
            if code:
                for other in running:
                    other.send_signal(signal.SIGTERM)
                for other in running:
                    try:other.wait(timeout=45)
                    except subprocess.TimeoutExpired:other.kill();other.wait()
                refresh();raise RuntimeError(f'Formal task failed group={j[0]} fold={j[1]} gpu={j[2]} exit={code}; see worker log')
            rows,_=refresh();print('COMMITTED',len(rows),'/100',GROUPS[j[0]],j[1],flush=True)
        time.sleep(1.)
    return time.monotonic()-started


def controller():
    lock=(RT/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        prepare();m=read(HERE/'manifest.json')
        if m['preflight']!='PASS':
            p=child(['preflight'],0,'preflight');assert p.wait()==0,'PREFLIGHT_FAIL; see preflight.log'
        m=read(HERE/'manifest.json')
        m['workers_per_gpu']=4;m['formal_started']=m.get('formal_started',now());m['formal_started_unix']=m.get('formal_started_unix',time.time());write(HERE/'manifest.json',m)
        if m['anchor_replay']!='ANCHOR_REPLAY_PASS':
            dispatch([(0,f,(f-1)%2) for f in range(1,11)])
            rows,_=refresh();base=[r for r in rows if r['group']==GROUPS[0]]
            assert len(base)==10 and all(not compare_anchor(r) for r in base),'ANCHOR_REPLAY_FAIL'
            m=read(HERE/'manifest.json');m.update(anchor_replay='ANCHOR_REPLAY_PASS',anchor_verified=now(),status='ANCHOR_REPLAY_PASS');write(HERE/'manifest.json',m)
            print('ANCHOR_REPLAY_PASS 10/10, original initialization/RNG/epoch/confusion/predictions exact',flush=True)
        jobs=[(g,f,((f-1)+(g%2))%2) for g in range(1,10) for f in range(1,11)]
        dispatch(jobs)
        m=read(HERE/'manifest.json');m.update(status='COMPLETE',formal_finished=now(),formal_wall_seconds=time.time()-m['formal_started_unix']);write(HERE/'manifest.json',m)
    except Exception as e:
        m=read(HERE/'manifest.json');m.update(status='FAIL',failure={'error':repr(e),'utc':now(),'traceback':traceback.format_exc()});write(HERE/'manifest.json',m)
        print(traceback.format_exc(),flush=True)
    finally:
        report(finalize=True)


def paired(rows,left,right):
    a={int(r['fold']):r for r in rows if r['group']==GROUPS[left]};b={int(r['fold']):r for r in rows if r['group']==GROUPS[right]}
    if len(a)!=10 or len(b)!=10:return None
    out={'left':GROUPS[left],'right':GROUPS[right]}
    for metric in ['acc','hard_auc','spe_td','sen_asd']:
        vals=[(float(a[f][metric])-float(b[f][metric]))*100 for f in range(1,11)]
        out[metric]={'delta_pp':float(np.mean(vals)),'per_fold_pp':vals,'wins':sum(v>1e-8 for v in vals),'ties':sum(abs(v)<=1e-8 for v in vals),'losses':sum(v< -1e-8 for v in vals)}
    return out


def append_handoff(text):
    prefix=read(RT/'handoff_prefix.json');pending=HERE/'handoff_append_pending.md';pending.write_text(text)
    marker='<!-- EXPERIMENT43_CORR_ANCHORED_GFR_FINAL -->'
    target=ROOT/'handoff.md'
    try:
        with target.open('ab') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            before=target.read_bytes();assert digest(before[:prefix['bytes']])==prefix['sha256'],'historical handoff prefix drift'
            if marker.encode() in before:return {'status':'ALREADY_APPENDED','original_prefix_preserved':True}
            f.write(text.encode());f.flush();os.fsync(f.fileno());after=target.read_bytes()
            assert after==before+text.encode() and after[:len(before)]==before
        pending.unlink();return {'status':'APPENDED_FSYNC_PREFIX_VERIFIED','before_bytes':len(before),'before_sha256':digest(before),'after_sha256':digest(after),'original_prefix_preserved':True}
    except Exception as e:return {'status':'FAILED_PENDING','error':repr(e),'path':str(pending)}


def report(finalize=False):
    rows,diags=refresh();m=read(HERE/'manifest.json');summary=[]
    for group in GROUPS:
        rs=[r for r in rows if r['group']==group]
        if len(rs)!=10:continue
        r={'group':group,'folds':10}
        for key in ['acc','hard_auc','spe_td','sen_asd']:
            vals=np.array([float(x[key])*100 for x in rs]);r[key+'_mean_percent']=float(vals.mean());r[key+'_sd_population_percent']=float(vals.std(ddof=0))
        summary.append(r)
    csvwrite(RESULTS/'summary.csv',summary)
    comparisons=[(1,0),(2,0),(3,0),(4,1),(5,1),(5,3),(6,0),(6,5),(6,7),(6,8),(6,9)]
    pairs=[p for a,b in comparisons if (p:=paired(rows,a,b)) is not None]
    go=False;checks={}
    if len(rows)==100:
        for c in [0,5,7,8,9]:
            p=paired(rows,6,c);checks[GROUPS[c]]=all(p[k]['delta_pp']>.1 and p[k]['wins']>=7 for k in ['acc','hard_auc'])
        active=[r for r in diags if r['group']==GROUPS[6] and r['modality']!='CORR']
        checks['bounded_true_auxiliary_response']=any(abs(float(r.get('aux_multiplier_mean',1.))-1.)>1e-8 and float(r['branch_off_abs_logit'])>1e-8 for r in active) and all(int(r.get('corr_support_changed',1))==0 for r in active)
        go=all(checks.values())
    decision='GO_INTERNAL_CANDIDATE' if go else 'NO-GO' if len(rows)==100 else 'NOT_EVALUABLE_INCOMPLETE'
    m.update(summary=summary,paired=pairs,decision=decision,go_checks=checks,total_wall_seconds=time.time()-m['started_unix'])
    title=['# Experiment 43 — GFR结构审计、密度校准与CORR锚定可靠性精炼','',
           f"状态：{m['status']}；完成{len(rows)}/100唯一正式折；GPU0={m['gpu_counts']['0']}、GPU1={m['gpu_counts']['1']}；B0-A={m['anchor_replay']}。",
           f"科研判定：**{decision}**。正式调度墙钟{m.get('formal_wall_seconds',0.):.2f}秒；总执行{m['total_wall_seconds']:.2f}秒。",'',
           '所有组seed0；四指标为十折mean±population SD，单位%。未完成十折的组不生成均值。',
           '|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|','|---|---:|---:|---:|---:|']
    for r in summary:title.append('|'+r['group']+'|'+'|'.join(f"{r[k+'_mean_percent']:.4f} ± {r[k+'_sd_population_percent']:.4f}" for k in ['acc','hard_auc','spe_td','sen_asd'])+'|')
    title+=['','## 预注册配对','', '|左减右|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|ACC W/T/L|Hard-AUC W/T/L|','|---|---:|---:|---:|---:|---:|---:|']
    for p in pairs:title.append('|'+p['left']+' − '+p['right']+'|'+'|'.join(f"{p[k]['delta_pp']:+.4f}" for k in ['acc','hard_auc','spe_td','sen_asd'])+'|'+'|'.join(f"{p[k]['wins']}/{p[k]['ties']}/{p[k]['losses']}" for k in ['acc','hard_auc'])+'|')
    title+=['','完整逐fold四指标差见manifest.paired；混淆、best/执行epoch、预测hash、入口SHA/RNG、best checkpoint SHA见fold_metrics.csv。完成状态要求任务原子提交和best文件读回SHA，不能只数CSV行。',
           '','## 原始结构、损失和机制核验','',
           '原始GFR i==1覆盖i==0；每支传播全320维是论文设计而非代码错误。GraphLearn.p实际参与cosine；w/t/threshold和combined_weight未进入有效前向。原simple-2只有分类＋四图GraphConstructLoss平均，ALoss不优化。E01只修聚合；E02只加train-train无序对MSE，lambda=1；E03只改mutual96与训练pair median/IQR；E04相对E01只保留对应模态输入块；E05为聚合修复＋密度校准。',
           'E06–E09全部以CORR mutual支持为上限；辅助只在exp(±.25*q)内改边权。可靠性来自原始特征空间、subject-disjoint内层逻辑回归的同诊断pair AUC增量；正式精炼使用当前GraphLearn投影空间。这个pair AUC是机制代理，不是第五项分类指标。所有新增标签计算只接收outer-train标签。E09 q=0的辅助GraphLearn实际可能没有梯度，须结合E07/E08，不能单据E06>E09宣称医学互补。',
           '','|组|模态|平均邻居|分类消息梯度均值|分支关闭logit变化|分支关闭翻转|','|---|---|---:|---:|---:|---:|']
    for group in GROUPS:
        for mod in ['PHENO','ANAT','FUNC','CORR']:
            rr=[r for r in diags if r['group']==group and r['modality']==mod]
            if rr:title.append(f"|{group}|{mod}|"+'|'.join(f"{np.mean([float(r[k]) for r in rr]):.7g}" for k in ['degree_mean','cls_message_grad','branch_off_abs_logit','branch_off_class_flip'])+'|')
    title+=['','## 可靠性门控（训练内估计）','', '|fold|PHENO Δpair-AUC|ANAT Δpair-AUC|FUNC Δpair-AUC|q(PHENO/ANAT/FUNC)|','|---|---:|---:|---:|---|']
    for fold in range(1,11):
        p=RT/'cache'/f'reliability_{fold:02}.json'
        if p.exists():
            rel=read(p);title.append(f"|{fold}|"+'|'.join(f'{v:+.7f}' for v in rel['delta'])+'|'+str([round(v,6) for v in rel['q']])+'|')
    title+=['','## 四类问题的独立结论','']
    for label,comparisons_here in [('分支覆盖修正',[(1,0)]),('论文监督意图',[(2,0)]),('密度及传播策略',[(3,0),(4,1),(5,1),(5,3)]),('CORR锚定辅助可靠性',[(6,0),(6,5),(6,7),(6,8),(6,9)])]:
        pp=[paired(rows,a,b) for a,b in comparisons_here];pp=[p for p in pp if p]
        title.append('- '+label+'：'+ ('；'.join(f"{p['left']}−{p['right']} ΔACC={p['acc']['delta_pp']:+.4f}pp、ΔHard-AUC={p['hard_auc']['delta_pp']:+.4f}pp" for p in pp) if pp else '尚无完整配对，不作性能判断')+'。')
    title+=['','预注册E06门槛：'+json.dumps(checks,ensure_ascii=False)+'。实现修复、监督项恢复或稀疏图收益分别解释，不自动计为疾病语义创新。',
           '','## 初始化、恢复、文献与局限','',
           '本轮没有调用GPT-4、CLIP或编码新文本。保留原PLN完整参数注册与旧8条buffer，以E40已验证的E32训练前初始模型/两个Adam/四种RNG重建每fold；缓存替代原CLIP构造的随机消耗后显式恢复post-construction RNG。加载的是未训练fold入口，不是历史最佳模型微调。E00按相同入口分两卡各五折重训；只在10/10指标、混淆、best/执行epoch、初始化/RNG/split及最佳预测全匹配后启动90折。',
           '原始simple-2/双Adam/450epochs/early>200/lr.0038/wd.11/dropout.35/hidden18/head2/th.9均沿用。阈值前score额外缓存不改变E00返回值；数值/梯度/两Adam两步一致与磁盘恢复后第三步一致见.runtime/preflight.json。每fold独立task/lock/原子checkpoint，10epoch覆盖保存，完成后仅保留可追溯best和入口。',
           '论文来源：[SPromptGL MICCAI2025](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)；对照[作者源码](https://github.com/wanxixi11/SPromptGL)。论文本身已有标签监督图及全模态传播；本轮E02是安全的阈值前MSE实现，不声称与原ALoss数值等价。网络论文没有证明ANAT/FUNC QC是病理信号。',
           '历史val=test选模、CV前全队列RFE/标准化、871节点transductive和重叠训练集使结果只适合内部配对筛选；不作独立泛化、临床因果、显著性或SOTA主张。',
           '', '失败/恢复：'+json.dumps(m.get('failure',{}),ensure_ascii=False),
           '','源码、数据、官方来源、Prompt文件指令、入口和结果hash见manifest。原保护文件需在结束时逐字节SHA仍相同。']
    text='\n'.join(title)+'\n';(RESULTS/'report.md').write_text(text)
    m['result_sha']={p.name:sha(p) for p in RESULTS.iterdir() if p.is_file()}
    if finalize:
        try:verify_sources();m['protected_sources']='UNCHANGED_SHA256'
        except Exception as e:m['protected_sources']=repr(e);m['status']='FAIL'
        handoff='\n\n<!-- EXPERIMENT43_CORR_ANCHORED_GFR_FINAL -->\n## Experiment 43 — CORR-anchored GFR Audit\n\n'+text+'\n产物目录：`'+str(HERE)+'`；复现：`python run.py all`，恢复：`python run.py resume`，报告：`python run.py report`。\n源码SHA：'+json.dumps(codehash(),ensure_ascii=False)+'\n输入SHA：'+m['source_sha'][str(ROOT/'data/ABIDE/processed_standard_data.csv')]+'\n'
        m['handoff']=append_handoff(handoff)
        m['delivery']='FORMAL_DIRECTORY_COMPLETE' if m['status']=='COMPLETE' and m['handoff']['status'] in ['APPENDED_FSYNC_PREFIX_VERIFIED','ALREADY_APPENDED'] else 'PARTIAL'
    write(HERE/'manifest.json',m)
    print('REPORT',m['status'],len(rows),m['gpu_counts'],decision,'handoff',m.get('handoff'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','preflight','all','resume','worker','report','reliability']);p.add_argument('--group',type=int);p.add_argument('--fold',type=int);p.add_argument('--gpu',type=int)
    a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='preflight':preflight()
    elif a.command in ['all','resume']:controller()
    elif a.command=='worker':worker(a.group,a.fold,a.gpu)
    elif a.command=='reliability':print(reliability(a.fold))
    elif a.command=='report':report()


if __name__=='__main__':main()
