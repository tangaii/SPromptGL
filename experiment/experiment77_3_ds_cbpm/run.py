#!/usr/bin/env python3
"""Exact-base, three-group Experiment 77.3 trainer and resumable scheduler."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse, copy, csv, hashlib, json, subprocess, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
from path_adapter import load_base_runner, BASE, sha_file, ANCHORS, LEGACY, REPO
from ds_cbpm import build, forward, prepare_texts
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'config.json').read_text())
GROUPS=('E77.3-G1','E77.3-G2','E77.3-G3')
METRICS=('acc','hard_auc','prob_auc','sen','spe','ba','asd_f1')
BASE_GROUP='E74.4-G2-Positive-DCR'
E74=None

def base():
    global E74
    if E74 is None:
        E74=load_base_runner()
    return E74

def runtime():
    return base().load_history().load_e69().load_e61().load_base()

def digest(v):
    return base().load_history().sha_object(v)

def save_json(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp')
    with q.open('w') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(q,p)

def save_torch(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp')
    with q.open('wb') as f:
        torch.save(v,f);f.flush();os.fsync(f.fileno())
    os.replace(q,p)

def write_csv(p,rs):
    keys=list(dict.fromkeys(k for v in rs for k in v))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)

def cpu_state(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}

def source_map():
    return {str(HERE/n):sha_file(HERE/n) for n in ('run.py','ds_cbpm.py','memory.py','path_adapter.py','config.json','runtime/text_embeddings.pt')}

def source_sha():
    return digest(source_map())

def initialize(fold,group):
    s,w,identity=base().prepare(fold)
    task=json.loads((BASE/'.runtime/tasks'/f'{BASE_GROUP}_f{fold:02d}.json').read_text())
    assert task['identity']==identity
    r=runtime();before=r.rng_sha(r.pack_rng())
    texts=prepare_texts(HERE,CFG)['texts']
    module=build(CFG['method'],fold,texts,s['parts'][3])
    assert r.rng_sha(r.pack_rng())==before==w['rng_sha']
    old_groups=len(s['opt_added'].param_groups)
    s['opt_added'].add_param_group({'params':list(module.parameters()),'lr':.001,'weight_decay':.01})
    s['fold']=fold;s['rudag_identity']=identity
    allowed=torch.tensor(np.setdiff1d(s['train_np'],s['test_np']),device=s['x'].device,dtype=torch.long)
    return s,module,allowed,identity,w,old_groups

def public_bundle(s):
    return base().load_history().state_bundle(s)

def restore(s,module,saved):
    base().load_history().load_bundle(s,saved['base_state'])
    module.load_state_dict(saved['module_state'],strict=True)
    runtime().restore_rng(saved['rng'])

def snapshot(s,module):
    return dict(base_state=public_bundle(s),module_state=cpu_state(module),rng=runtime().pack_rng())

def start_memory(s,module,allowed,group):
    before=runtime().rng_sha(runtime().pack_rng())
    with torch.no_grad():
        _,capture=forward(base(),s,module,group,0,False,enabled=False)
        module.memory.update_class(capture['h'],s['parts'][3],s['y'],allowed)
    assert runtime().rng_sha(runtime().pack_rng())==before

def gradnorm(params):
    pieces=[p.grad.detach().square().sum() for p in params if p.grad is not None]
    return float(torch.sqrt(torch.stack(pieces).sum())) if pieces else 0.

def step(s,module,allowed,group,local_epoch):
    r=runtime(); before=r.rng_sha(r.pack_rng())
    base().load_history().zero_optimizers(s)
    out,cap=forward(base(),s,module,group,local_epoch,True)
    boundary=group!='E77.3-G1' and local_epoch>CFG['method']['warmup']
    aux,rank=module.losses(cap['evidence'],s['y'],allowed,1 if group=='E77.3-G3' else 0,boundary)
    total=out['total']+CFG['method']['class_loss_weight']*aux+CFG['method']['ranking_loss_weight']*rank
    assert torch.isfinite(total)
    total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for mm in (s['model'],s['cam'],module) for p in mm.parameters())
    norms=dict(module=gradnorm(module.parameters()),corr_projection=gradnorm(module.corr_projection.parameters()),
               text_projection=gradnorm(module.text_projection.parameters()),base=gradnorm(s['model'].parameters()))
    for opt in ('opt_base','opt_cam','opt_added'):
        s[opt].step()
    after_train=r.rng_sha(r.pack_rng())
    with torch.no_grad():
        evaluated,evcap=forward(base(),s,module,group,local_epoch,False)
        metrics=r.classification_metrics(evaluated['logp'],s['test'],s['y'])
    after_eval=r.rng_sha(r.pack_rng())
    return dict(out=out,evaluated=evaluated,capture=evcap,metrics=metrics,
                losses={k:float(out[k].detach()) for k in ('cls','graph','pair','adv','preserve','total')} |
                       {'typical_aux':float(aux.detach()),'boundary_rank':float(rank.detach()),'total_with_aux':float(total.detach())},
                gradients=norms,rng={'rng_before_train':before,'rng_after_train':after_train,'rng_after_eval':after_eval})

def update_memory(s,module,allowed,cap,group,local_epoch):
    m=module.memory
    result=m.update_errors_and_pairs(cap['base_logp'].detach(),cap.get('final_logp',cap['base_logp']).detach(),
          cap['h'].detach(),s['parts'][3],s['y'],allowed,local_epoch,
          group!='E77.3-G1' and local_epoch>=CFG['method']['warmup'])
    m.update_class(cap['h'].detach(),s['parts'][3],s['y'],allowed)
    return result

def paths(group,fold):
    return tuple(HERE/p/f'{group}_f{fold:02d}{ext}' for p,ext in
                 [('runtime/tasks','.json'),('checkpoints','.best.pt'),('checkpoints','.latest.pt'),('runtime/logs','.jsonl')])

def train_task(group,fold,expected):
    taskpath,bestpath,latestpath,logpath=paths(group,fold)
    if taskpath.exists():
        done=json.loads(taskpath.read_text())
        if done['status']=='COMPLETE' and done['source_sha']==expected and sha_file(bestpath)==done['checkpoint_sha256']:
            return done
    assert source_sha()==expected
    s,module,allowed,identity,w,old_groups=initialize(fold,group)
    r=runtime(); initial_params=digest(dict(module.named_parameters()))
    start_memory(s,module,allowed,group)
    trace=[];signatures=[];best_acc=-1.;best_epoch=-1;wait=0;start_epoch=10
    if latestpath.exists():
        saved=torch.load(latestpath,map_location='cpu',weights_only=False)
        assert saved['source_sha']==expected and saved['identity']==identity
        restore(s,module,saved)
        start_epoch=saved['next_epoch'];trace=saved['rng_trace'];signatures=saved['warmup_signatures']
        best_acc,best_epoch,wait=saved['best_acc'],saved['best_epoch'],saved['wait']
        print(f'RESUME {group} fold{fold} next_epoch={start_epoch+1}',flush=True)
    history={t['epoch']:t for t in json.loads((BASE/'.runtime/tasks'/f'{BASE_GROUP}_f{fold:02d}.json').read_text())['rng_trace']}
    start=time.time()
    for epoch in range(start_epoch,450):
        local=epoch-9
        z=step(s,module,allowed,group,local)
        z['capture']['final_logp']=z['evaluated']['logp']
        rr=z['rng'];hist=history.get(epoch+1)
        if hist is not None:
            assert (rr['rng_before_train'],rr['rng_after_train'],rr['rng_after_eval'])==(hist['rng_before'],hist['rng_after_train'],hist['rng_after_eval']), (group,fold,epoch,'public RNG mismatch')
        if local<=CFG['method']['warmup']:
            signatures.append(dict(epoch=local,logp=digest(z['evaluated']['logp'].detach()),
               shared=digest(public_bundle(s)),parameters=digest(dict(module.named_parameters())),
               class_memory=digest({k:v for k,v in module.memory.state_dict().items() if k in ('cam_sum','raw_sum','counts','assignment','cam_cache','raw_cache')}),rng=rr.copy()))
        improved=z['metrics']['acc']>best_acc
        if improved:
            best_acc=float(z['metrics']['acc']);best_epoch=epoch;wait=0
            # Exact memory used for this evaluation, before next epoch's update.
            payload=snapshot(s,module) | dict(group=group,fold=fold,identity=identity,source_sha=expected,
                best_epoch=epoch+1,local_epoch=local,metrics={k:v for k,v in z['metrics'].items() if k not in ('prob','pred')},
                test_probability=z['metrics']['prob'].tolist(),test_prediction=z['metrics']['pred'].tolist(),
                losses=z['losses'],gradients=z['gradients'],text_cache_sha256=sha_file(HERE/'runtime/text_embeddings.pt'),
                initial_module_parameters_sha=initial_params,old_added_groups=old_groups,
                evaluation_memory_epoch=int(module.memory.epoch))
            save_torch(bestpath,payload)
        else:
            wait+=1
        memory=update_memory(s,module,allowed,z['capture'],group,local)
        rr['rng_after_memory']=r.rng_sha(r.pack_rng())
        assert rr['rng_after_memory']==rr['rng_after_eval']
        trace.append(dict(epoch=epoch+1,local_epoch=local,**rr))
        logpath.parent.mkdir(parents=True,exist_ok=True)
        record=dict(epoch=epoch+1,local_epoch=local,best_acc=best_acc,best_epoch=best_epoch+1,
                    losses=z['losses'],gradients=z['gradients'],memory=memory,rng=rr,
                    typical_abs=float(z['capture']['evidence']['typical'].abs().mean()),
                    boundary_abs=float(z['capture']['evidence']['boundary'].abs().mean()),
                    correction_abs=float(z['capture']['evidence']['margin_correction'].abs().mean()))
        with logpath.open('a') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n')
        if (epoch+1)%10==0 or wait>200 or epoch==449:
            save_torch(latestpath,snapshot(s,module) | dict(group=group,fold=fold,source_sha=expected,identity=identity,
               next_epoch=epoch+1,wait=wait,best_acc=best_acc,best_epoch=best_epoch,rng_trace=trace,warmup_signatures=signatures))
            print(f'{group} fold={fold} epoch={epoch+1} best={best_acc:.6f}@{best_epoch+1} pairs={int(module.memory.pair_count)}',flush=True)
        del z
        if wait>200:
            break
    cp=torch.load(bestpath,map_location='cpu',weights_only=False)
    result=dict(status='COMPLETE',group=group,fold=fold,source_sha=expected,identity=identity,
                metrics=cp['metrics'],best_epoch=best_epoch+1,epochs_executed=epoch+1,
                checkpoint_sha256=sha_file(bestpath),initial_module_parameters_sha=initial_params,
                rng_trace=trace,warmup_signatures=signatures,wall_seconds=time.time()-start)
    save_json(taskpath,result)
    print(f'COMPLETE {group} fold{fold}',flush=True)
    return result

def worker(fold,expected):
    for group in GROUPS:
        train_task(group,fold,expected)
    a=json.loads(paths(GROUPS[0],fold)[0].read_text())['warmup_signatures']
    b=json.loads(paths(GROUPS[1],fold)[0].read_text())['warmup_signatures']
    assert len(a)==len(b)==15 and a==b, f'Warmup parity failed fold{fold}'
    save_json(HERE/'results'/f'warmup_parity_f{fold:02d}.json',{'fold':fold,'status':'PASS','epochs':15})

def schedule(workers):
    from audit import preflight, finalize
    if not (HERE/'results/preflight.json').exists():
        preflight()
    pre=json.loads((HERE/'results/preflight.json').read_text());assert pre['status']=='PASS'
    expected=source_sha();assert expected==pre['source_sha']
    pending=list(range(1,11));active=[];attempts={i:0 for i in pending};failures=[]
    while pending or active:
        while pending and len(active)<workers:
            fold=pending.pop(0);attempts[fold]+=1
            log=HERE/'runtime/logs'/f'fold{fold:02d}.stdout.log';log.parent.mkdir(parents=True,exist_ok=True)
            f=log.open('a')
            p=subprocess.Popen([sys.executable,'-B',str(HERE/'run.py'),'--worker','--fold',str(fold),'--source',expected],cwd=HERE,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
            active.append((fold,p,f))
        remaining=[]
        for fold,p,f in active:
            code=p.poll()
            if code is None:
                remaining.append((fold,p,f));continue
            f.close()
            if code:
                failures.append(dict(fold=fold,attempt=attempts[fold],exit_code=code))
                if attempts[fold]>=2:
                    save_json(HERE/'results/execution.json',{'failures':failures,'status':'FAILED'})
                    raise RuntimeError(f'Fold{fold} failed twice; see log')
                pending.append(fold)
            else:
                print(f'FOLD {fold} ALL GROUPS COMPLETE; active={len(remaining)} pending={len(pending)}',flush=True)
        active=remaining
        save_json(HERE/'results/execution.json',{'failures':failures,'attempts':attempts,'active':[x[0] for x in active],'pending':pending,'worker_limit':workers})
        if active:
            time.sleep(3)
    finalize()

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--audit',action='store_true');p.add_argument('--all',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--fold',type=int);p.add_argument('--source');p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if a.worker:
        worker(a.fold,a.source)
    elif a.preflight:
        from audit import preflight
        preflight()
    elif a.audit:
        from audit import finalize
        finalize()
    elif a.all:
        assert 1<=a.workers<=10
        schedule(a.workers)
    else:p.print_help()
if __name__=='__main__':main()
