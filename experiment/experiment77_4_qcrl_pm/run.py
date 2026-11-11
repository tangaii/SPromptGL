#!/usr/bin/env python3
"""Thirty resumable formal QCRL-PM tasks with an unchanged independent base path."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse, copy, csv, json, subprocess, sys, time, traceback
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import torch
from path_adapter import load_base_runner, BASE, E773, HERE, REPO, sha_file
from memory import Memory, identity_split, audit_bank, cpu_bank
from qcrl_pm import build, base_forward

CFG = json.loads((HERE/'config.json').read_text())
GROUPS = tuple(CFG['groups'])
BASE_GROUP = 'E74.4-G2-Positive-DCR'
METRICS = ('acc','hard_auc','prob_auc','sen','spe','asd_f1')
E74 = None

def base():
    global E74
    if E74 is None:
        E74 = load_base_runner()
    return E74

def runtime():
    return base().load_history().load_e69().load_e61().load_base()

def digest(value):
    return base().load_history().sha_object(value)

def save_json(path,value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def save_torch(path,value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def write_csv(path,rows):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def cpu_state(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}

def source_map():
    return {n:sha_file(HERE/n) for n in ('run.py','qcrl_pm.py','memory.py','path_adapter.py','config.json','runtime/text_embeddings.pt')}

def source_sha():
    return digest(source_map())

def initialize(fold,group):
    s,w,identity = base().prepare(fold)
    assert identity == json.loads((BASE/'.runtime/tasks'/f'{BASE_GROUP}_f{fold:02d}.json').read_text())['identity']
    r = runtime();before = r.rng_sha(r.pack_rng())
    text = torch.load(HERE/'runtime/text_embeddings.pt',map_location='cpu',weights_only=False)
    assert sha_file(HERE/'runtime/text_embeddings.pt') == CFG['text_source_sha256']
    assert text['medical_text'] == CFG['medical_prompts'] and text['anonymous_text'] == CFG['anonymous_prompts']
    module = build(CFG['method'],fold,text['texts'],s['x'].device)
    optimizer = torch.optim.AdamW(module.parameters(),lr=.001,weight_decay=.01)
    allowed = np.setdiff1d(s['train_np'],s['test_np'])
    split = identity_split(allowed,s['labels_np'],fold)
    memory = Memory(CFG['method'],allowed,s['labels_np'],split,fold)
    assert r.rng_sha(r.pack_rng()) == before == w['rng_sha']
    initial = dict(identity=identity,module_parameters_sha=digest(dict(module.named_parameters())),
        split_sha=digest(split),new_optimizer_sha=digest(optimizer.state_dict()),
        public_rng_sha=before,parameter_count=sum(p.numel() for p in module.parameters()))
    return s,module,optimizer,memory,initial

def public_bundle(s):
    return base().load_history().state_bundle(s)

def snapshot(s,module,optimizer,memory):
    return dict(base_state=public_bundle(s),module_state=cpu_state(module),new_optimizer=copy.deepcopy(optimizer.state_dict()),
        memory_state=memory.state_dict(),rng=runtime().pack_rng())

def restore(s,module,optimizer,memory,cp):
    base().load_history().load_bundle(s,cp['base_state'])
    module.load_state_dict(cp['module_state'],strict=True)
    optimizer.load_state_dict(cp['new_optimizer'])
    memory.load_state_dict(cp['memory_state'])
    runtime().restore_rng(cp['rng'])

def gradnorm(parameters):
    tensors = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.sqrt(torch.stack(tensors).sum())) if tensors else 0.

def metric_values(logp,indices,y):
    if torch.is_tensor(indices):
        indices = indices.detach().cpu().numpy()
    return runtime().classification_metrics(logp,indices,y)

def compact_metrics(m):
    return {k:v for k,v in m.items() if k not in ('prob','pred')}

def public_update(s):
    base().load_history().zero_optimizers(s)
    out,h = base_forward(base(),s,True)
    losses = {k:float(out[k].detach()) for k in ('cls','graph','pair','adv','preserve','total')}
    out['total'].backward()
    for name in ('opt_base','opt_cam','opt_added'):
        s[name].step()
    del out,h
    return losses

def step(s,module,optimizer,memory,group,local):
    r = runtime();before = r.rng_sha(r.pack_rng())
    losses = public_update(s)
    after_train = r.rng_sha(r.pack_rng())
    # Exactly the original one eval forward per epoch, no additional base call.
    with torch.no_grad():
        out,h = base_forward(base(),s,False)
        logp = out['logp'].detach()
        base_metrics = metric_values(logp,s['test'],s['y'])
    after_eval = r.rng_sha(r.pack_rng())
    del out
    query = np.flatnonzero(memory.split == (local-1)%3)
    support = np.setdiff1d(memory.allowed,query)
    hard = group != 'E77.4-G1' and local>CFG['method']['warmup']
    text_index = int(group == 'E77.4-G3')
    hc = h.cpu().numpy();rawc = s['parts'][3].detach().cpu().numpy()
    memory.observe_predictions(logp.argmax(1).cpu().numpy(),local)
    memory_stats = memory.update_relations(hc,rawc,support,hard)
    bank = memory.bank(hc,rawc,support,h.device)
    isolation = audit_bank(bank,query,s['test_np'],s['labels_np'])
    qi = torch.as_tensor(query,device=h.device)
    module.train();optimizer.zero_grad(set_to_none=True)
    z = module(h[qi],logp[qi],s['parts'][3][qi],qi,bank,hard=hard,text_index=text_index)
    total,aux = module.losses(z,s['y'][qi])
    assert torch.isfinite(total)
    public = tuple(p for mm in (s['model'],s['cam']) for p in mm.parameters() if p.requires_grad)
    leak = torch.autograd.grad(total,public,allow_unused=True,retain_graph=True)
    assert all(g is None for g in leak), 'New loss reached DARG-CAM'
    total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
    gradients = {name:gradnorm(getattr(module,name).parameters()) for name in
        ('cam_typical','cam_relation','corr_value','query_key','pair_key','pair_difference','text_projection','typical_gate','confusion_gate')}
    gradients['public_new_gradient'] = 0.
    with torch.no_grad():
        yq = s['y'][qi]
        bp,fp = logp[qi].argmax(1),z['logp'].argmax(1)
        query_stats = dict(count=len(query),asd=int((yq==0).sum()),td=int((yq==1).sum()),
            base_errors=int((bp!=yq).sum()),corrected=int(((bp!=yq)&(fp==yq)).sum()),
            harmed=int(((bp==yq)&(fp!=yq)).sum()),valid_relation=int(z['has_pairs'].sum()),
            relation_direction_correct=int((((1-2*yq)*z['confusion'])>0).sum()),
            positive=int((z['confusion']>0).sum()),negative=int((z['confusion']<0).sum()),
            zero=int((z['confusion']==0).sum()),support_pairs=len(bank['pair_ids']))
    optimizer.step()
    # Full legal outer-training memory is used only for unknown-label test Query.
    eval_bank = memory.bank(hc,rawc,memory.allowed,h.device)
    audit_bank(eval_bank,s['test_np'],s['test_np'],s['labels_np'])
    module.eval()
    ti = torch.as_tensor(s['test_np'],device=h.device)
    with torch.no_grad():
        evaluated = module(h[ti],logp[ti],s['parts'][3][ti],ti,eval_bank,hard=hard,text_index=text_index)
        metrics = metric_values(evaluated['logp'],torch.arange(len(ti),device=h.device),s['y'][ti])
    memory.eval_bank = cpu_bank(eval_bank)
    after_memory = r.rng_sha(r.pack_rng())
    assert after_memory == after_eval, 'New branch consumed public RNG'
    return dict(base_logp=logp,test_evidence=evaluated,metrics=metrics,base_metrics=base_metrics,
        losses=losses|{k:float(v.detach()) for k,v in aux.items()},gradients=gradients,
        query=query_stats,isolation=isolation,memory=memory_stats,
        rng=dict(rng_before_train=before,rng_after_train=after_train,rng_after_eval=after_eval,rng_after_memory=after_memory))

def paths(group,fold):
    return tuple(HERE/p/f'{group}_f{fold:02d}{ext}' for p,ext in
        [('runtime/tasks','.json'),('checkpoints','.best.pt'),('checkpoints','.latest.pt'),('runtime/logs','.jsonl')])

def references(fold):
    task = json.loads((BASE/'.runtime/tasks'/f'{BASE_GROUP}_f{fold:02d}.json').read_text())
    losses = {v['epoch']:v for line in (BASE/'.runtime/logs'/f'{BASE_GROUP}_f{fold:02d}.jsonl').read_text().splitlines() if (v:=json.loads(line))}
    bp = BASE/'.runtime/checkpoints'/f'{BASE_GROUP}_f{fold:02d}.best.pt'
    lp = BASE/'.runtime/checkpoints'/f'{BASE_GROUP}_f{fold:02d}.latest.pt'
    assert sha_file(bp) == task['checkpoint_sha256']
    b = torch.load(bp,map_location='cpu',weights_only=False)
    l = torch.load(lp,map_location='cpu',weights_only=False)
    states = {b['best_epoch_1based']:digest(b['best_state']),l['next_epoch']:digest(l['current'])}
    return dict(task=task,rng={v['epoch']:v for v in task['rng_trace']},log=losses,states=states,
        best_epoch=b['best_epoch_1based'],best_losses=b['best_losses'],best_probability=b['test_probability'])

def train_task(group,fold,expected):
    taskpath,bestpath,latestpath,logpath = paths(group,fold)
    if taskpath.exists():
        t = json.loads(taskpath.read_text())
        if t['status']=='COMPLETE':
            assert t['source_sha']==expected and sha_file(bestpath)==t['checkpoint_sha256']
            return t
    assert source_sha()==expected
    s,module,optimizer,memory,initial = initialize(fold,group)
    ref = references(fold)
    trace,signatures,reference_checks = [],[],[]
    start_epoch,best_acc,best_epoch,wait = 10,-1.,-1,0
    if latestpath.exists():
        saved = torch.load(latestpath,map_location='cpu',weights_only=False)
        assert saved['source_sha']==expected and saved['initial']==initial
        restore(s,module,optimizer,memory,saved)
        start_epoch,best_acc,best_epoch,wait = (saved[k] for k in ('next_epoch','best_acc','best_epoch','wait'))
        trace,signatures,reference_checks = saved['trace'],saved['warmup_signatures'],saved['reference_checks']
        # Preserve a crashed process's log tail and deterministically replay it.
        if logpath.exists():
            records = [json.loads(line) for line in logpath.read_text().splitlines()]
            tail = [v for v in records if v['epoch']>start_epoch]
            if tail:
                save_json(HERE/'runtime/recovery'/f'{group}_f{fold:02d}_after{start_epoch}.json',tail)
                logpath.write_text(''.join(json.dumps(v)+'\n' for v in records if v['epoch']<=start_epoch))
        print(f'RESUME {group} f{fold} epoch={start_epoch+1}',flush=True)
    start = time.time()
    for epoch in range(start_epoch,CFG['max_epoch']):
        local = epoch-9
        z = step(s,module,optimizer,memory,group,local)
        rr = z['rng'];historical = ref['rng'].get(epoch+1)
        if historical is not None:
            assert (rr['rng_before_train'],rr['rng_after_train'],rr['rng_after_eval']) == tuple(historical[k] for k in ('rng_before','rng_after_train','rng_after_eval')), (group,fold,epoch,'RNG mismatch')
        public_sha = digest({k:cpu_state(s[k]) for k in ('model','cam')})
        if epoch+1 in ref['log']:
            assert z['losses']['total']==ref['log'][epoch+1]['loss'], (group,fold,epoch,'base loss mismatch')
        if epoch+1 in ref['states']:
            assert digest(public_bundle(s))==ref['states'][epoch+1], (group,fold,epoch,'base model/optimizer mismatch')
            reference_checks.append(dict(epoch=epoch+1,base_model_optimizer_exact=True))
        if epoch+1 == ref['best_epoch']:
            assert all(z['losses'][k]==v for k,v in ref['best_losses'].items())
            assert np.array_equal(np.asarray(z['base_metrics']['prob']),np.asarray(ref['best_probability']))
        if local<=CFG['method']['warmup']:
            signatures.append(dict(epoch=local,public=digest(public_bundle(s)),
                parameters=digest(dict(module.named_parameters())),optimizer=digest(optimizer.state_dict()),
                bank=digest(memory.eval_bank),rng=rr))
        improved = z['metrics']['acc']>best_acc
        if improved:
            best_acc,best_epoch,wait = float(z['metrics']['acc']),epoch+1,0
            payload = snapshot(s,module,optimizer,memory)|dict(group=group,fold=fold,source_sha=expected,
                initial=initial,best_epoch=epoch+1,local_epoch=local,metrics=compact_metrics(z['metrics']),
                base_metrics=compact_metrics(z['base_metrics']),test_probability=z['metrics']['prob'].tolist(),
                test_prediction=z['metrics']['pred'].tolist(),base_probability=z['base_metrics']['prob'].tolist(),
                base_prediction=z['base_metrics']['pred'].tolist(),losses=z['losses'],gradients=z['gradients'],
                query=z['query'],isolation=z['isolation'],base_public_sha=public_sha,
                text_cache_sha256=sha_file(HERE/'runtime/text_embeddings.pt'))
            save_torch(bestpath,payload)
        else:
            wait += 1
        trace.append(dict(epoch=epoch+1,public_sha=public_sha,**rr))
        record = dict(epoch=epoch+1,local_epoch=local,best_acc=best_acc,best_epoch=best_epoch,
            losses=z['losses'],gradients=z['gradients'],memory=z['memory'],query=z['query'],
            isolation=z['isolation'],rng=rr,public_sha=public_sha,
            mean_abs_delta=float(z['test_evidence']['delta'].abs().mean()))
        logpath.parent.mkdir(parents=True,exist_ok=True)
        with logpath.open('a') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n')
        if (epoch+1)%10==0 or wait>200 or epoch==449:
            save_torch(latestpath,snapshot(s,module,optimizer,memory)|dict(group=group,fold=fold,source_sha=expected,
                initial=initial,next_epoch=epoch+1,best_acc=best_acc,best_epoch=best_epoch,wait=wait,
                trace=trace,warmup_signatures=signatures,reference_checks=reference_checks))
            print(f'{group} fold={fold} epoch={epoch+1} best={best_acc:.6f}@{best_epoch} pairs={len(memory.pairs)}',flush=True)
        del z
        if wait>200:
            break
    cp = torch.load(bestpath,map_location='cpu',weights_only=False)
    result = dict(status='COMPLETE',group=group,fold=fold,source_sha=expected,initial=initial,
        metrics=cp['metrics'],base_metrics=cp['base_metrics'],best_epoch=best_epoch,epochs_executed=epoch+1,
        checkpoint_sha256=sha_file(bestpath),trace=trace,warmup_signatures=signatures,
        reference_checks=reference_checks,historical_reference_last_epoch=max(ref['rng']),wall_seconds=time.time()-start)
    save_json(taskpath,result)
    print(f'COMPLETE {group} fold={fold}',flush=True)
    return result

def worker(fold,expected):
    try:
        tasks = [train_task(group,fold,expected) for group in GROUPS]
        assert tasks[0]['warmup_signatures']==tasks[1]['warmup_signatures']
        assert tasks[0]['initial']==tasks[1]['initial']==tasks[2]['initial']
        for x in tasks[1:]:
            for a,b in zip(tasks[0]['trace'],x['trace']):
                assert a==b, ('cross-group public parity',fold,a['epoch'])
        save_json(HERE/'results'/f'parity_f{fold:02d}.json',dict(status='PASS',fold=fold,
            warmup_epochs=15,initial_exact=True,public_common_epochs=min(len(t['trace']) for t in tasks),
            public_parameter_rng_exact=True,new_gradient_leaks=0))
    except Exception:
        save_json(HERE/'runtime/failures'/f'fold{fold:02d}.json',dict(traceback=traceback.format_exc(),source_sha=expected))
        raise

def run_all(workers):
    assert 1<=workers<=10
    pre = json.loads((HERE/'results/preflight.json').read_text())
    expected = source_sha()
    assert pre['status']=='PASS' and pre['source_sha']==expected
    save_json(HERE/'manifest.json',dict(experiment=CFG['experiment'],status='FORMAL_RUNNING',source_sha=expected,
        frozen_sources=source_map(),formal_tasks=30,workers=workers,preflight_sha256=sha_file(HERE/'results/preflight.json')))
    pending = list(range(1,11));active = {};finished=[];start=time.time()
    while pending or active:
        while pending and len(active)<workers:
            fold = pending.pop(0)
            path = HERE/'runtime/logs'/f'worker_f{fold:02d}.log';path.parent.mkdir(parents=True,exist_ok=True)
            f = path.open('a')
            p = subprocess.Popen([sys.executable,str(HERE/'run.py'),'--worker',str(fold),'--expected',expected],stdout=f,stderr=subprocess.STDOUT,cwd=REPO)
            active[fold]=(p,f)
        for fold,(p,f) in list(active.items()):
            if p.poll() is not None:
                f.close();del active[fold]
                finished.append(dict(fold=fold,returncode=p.returncode))
                if p.returncode:
                    # Already-running tasks are allowed to checkpoint; no task is
                    # silently restarted or initialized over an existing latest.
                    pending=[]
        save_json(HERE/'runtime/scheduler.json',dict(active=list(active),pending=pending,finished=finished,
            started_unix=start,elapsed=time.time()-start))
        if active:
            time.sleep(2)
    assert all(t['returncode']==0 for t in finished) and len(finished)==10, finished
    from audit import finalize
    finalize()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--preflight',action='store_true');p.add_argument('--all',action='store_true')
    p.add_argument('--audit',action='store_true');p.add_argument('--worker',type=int)
    p.add_argument('--expected');p.add_argument('--workers',type=int,default=10)
    args=p.parse_args()
    if args.preflight:
        from audit import preflight
        preflight()
    elif args.worker:
        worker(args.worker,args.expected)
    elif args.all:
        run_all(args.workers)
    elif args.audit:
        from audit import finalize
        finalize()
    else:
        p.error('choose --preflight, --all, --audit, or --worker')

if __name__=='__main__':
    main()
