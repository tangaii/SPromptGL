#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, csv, fcntl, hashlib, importlib.util, json, os, pickle, random, shutil, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

EXP=Path(__file__).resolve().parent; ROOT=Path('/data/adodas/SPromptGL-main'); RT=EXP/'.runtime'; RES=EXP/'results'; HIST=ROOT/'experiment/experiment43_2_gfr_incremental_value_audit'
CFG=json.loads((EXP/'config.json').read_text()); CTX=None
sys.path.insert(0,str(EXP))
from integration import read_data, fold_sha, sha, source_copy, copy_anchors, protected_hashes, make_hp
from pid_estimator import estimate_pid, synthetic_audit
from method import make_agent, close_agent, original_gfr_step, train_step, evaluate, metric_row, state_hash

def read(p): return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+f'.tmp.{os.getpid()}')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); os.replace(t,p)
def save(p,v):
    import torch
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+f'.tmp.{os.getpid()}')
    with t.open('wb') as f: torch.save(v,f); f.flush(); os.fsync(f.fileno())
    os.replace(t,p)
def load(p):
    import torch
    return torch.load(p,map_location='cpu',weights_only=False)
def csvread(p):
    if not Path(p).exists(): return []
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+f'.tmp.{os.getpid()}')
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['fold']
    with t.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def rng():
    import torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore(r):
    import torch
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])
def rng_hash(r):
    h=hashlib.sha256(pickle.dumps((r['python'],r['numpy']),protocol=4))
    for x in [r['cpu']]+r['cuda']:h.update(x.cpu().numpy().tobytes())
    return h.hexdigest()
def model_state(a): return {'gfr':{k:v.detach().cpu().clone() for k,v in a.gfr.state_dict().items()},'pln':{k:v.detach().cpu().clone() for k,v in a.pln.state_dict().items()},'augment':{k:v.detach().cpu().clone() for k,v in a.augment.state_dict().items()}}
def set_model(a,s): a.gfr.load_state_dict(s['gfr']);a.pln.load_state_dict(s['pln']);a.augment.load_state_dict(s['augment'])
def equal(a,b):
    import torch
    if isinstance(a,torch.Tensor): return torch.equal(a,b)
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def signature():
    files=[EXP/n for n in ['run.py','integration.py','pid_estimator.py','method.py','config.json']]+list((RT/'sources').glob('*.py'))
    return hashlib.sha256(json.dumps({str(p.relative_to(EXP)):sha(p) for p in files},sort_keys=True).encode()).hexdigest()
def data(): return read_data()[:3]
def anchor(f): return load(RT/'anchors'/f'fold{f:02}.pt')
def runtime():
    global CTX
    if CTX is None:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        torch.cuda.set_device(0);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        sys.path.insert(0,str(RT/'sources'))
        import layers,network,utils
        assert not any(x in sys.modules for x in ('CLIP','CLIP.clip'))
        CTX={'torch':torch,'layers':layers,'network':network,'utils':utils}
    return CTX
def build_from_anchor(fold,group,pid_lambdas):
    c=runtime(); x,y,folds=data(); tr,te=folds[fold-1]; hp=make_hp(CFG); a=make_agent(c['torch'],np,c['layers'],c['network'],c['utils'],hp,x,y,tr,te,anchor(fold),fold,group,CFG)
    restore(anchor(fold)['rng']); a.pid_lambdas=pid_lambdas; return a
def historical_rows():
    rows=csvread(HIST/'results/fold_metrics.csv'); return {int(r['fold']):r for r in rows if int(r['group_id'])==0}
def pid_rows(): return csvread(RES/'pid_weights.csv')
def pid_for_fold(f):
    rr=[r for r in pid_rows() if int(r['fold'])==f and r['k']=='4']
    return {r['pair']:float(r['lambda']) for r in rr}
def prepare():
    src=source_copy(); ah=copy_anchors(); x,y,folds,mapping,columns=read_data(); refs=historical_rows();
    assert len(refs)==10
    for f in range(1,11):
        a=anchor(f); assert a['fold']==f and a['split_sha']==fold_sha(*folds[f-1]); assert sha(RT/'anchors'/f'fold{f:02}.pt')==refs[f]['anchor_sha']
    hb=(ROOT/'handoff.md').read_bytes(); m=read(EXP/'manifest.json') if (EXP/'manifest.json').exists() else {}
    m.update(status='PREPARED',started_unix=m.get('started_unix',time.time()),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha=src,anchor_sha=ah,protected_sha256=protected_hashes(),data_shape=list(x.shape),label_counts=np.bincount(y).tolist(),fold_sha={str(f):fold_sha(*folds[f-1]) for f in range(1,11)},history_code_signature=read(HIST/'manifest.json').get('code_signature'),history_summary=refs,handoff_prefix={'bytes':len(hb),'sha256':hashlib.sha256(hb).hexdigest()},columns_sha=hashlib.sha256(json.dumps(columns).encode()).hexdigest(),pid_estimator='I_min_two_source_discrete_pid_v1',cvxpy_available=False)
    write(EXP/'manifest.json',m); print('PREPARE_PASS',json.dumps({'source':src,'anchors':len(ah),'shape':x.shape,'labels':np.bincount(y).tolist()}),flush=True)
def preflight(gpu):
    c=runtime(); x,y,folds=data(); f=CFG['gpu_folds'][str(gpu)][0]; a=build_from_anchor(f,0,{'AF':0.,'AC':0.,'FC':0.}); checks={}
    assert sum(p.numel() for p in a.gfr.parameters())==887836 and sum(p.numel() for p in a.pln.parameters())==49544; checks['parameter_counts']=True
    # The physical PEN-free PLN still contains the original OutputLayer MLP
    # head; only PEN-specific prompt/CLIP/SPGS/RES/avg_pool artifacts must be
    # absent.  Banning the generic ``mlp`` token would incorrectly reject the
    # classifier head and was a preflight-only false failure.
    banned=('prompt','clip','avg_pool','spgs','res.')
    assert all(not any(tok in n.lower() for tok in banned) for n in list(a.pln.state_dict())+list(a.gfr.state_dict())); checks['physical_pen_free']=True
    out0,feat,adjs=evaluate(c['torch'],a); assert c['torch'].isfinite(out0).all() and c['torch'].equal(c['torch'].cat(a.dataLayer(a.feat)[1],dim=1),a.feat);checks['forward_finite_raw_concat']=True
    old_t=c['torch']; before=model_state(a); beforemp=copy.deepcopy(a.mp.state_dict()); beforemf=copy.deepcopy(a.mf.state_dict()); r0=rng(); original_gfr_step(old_t,c['utils'],a); assert old_t.isfinite(a.feat).all(); checks['one_step_finite']=True
    # Test the independent augmentation path and held-out-label invariance.
    b=build_from_anchor(f,1,{'AF':0.01,'AC':0.02,'FC':0.03}); o1=evaluate(old_t,b)[0]; old=b.targ.clone(); b.targ[b.tst_idx]=1-b.targ[b.tst_idx]; o2=evaluate(old_t,b)[0]; b.targ=old; assert old_t.equal(o1,o2); checks['heldout_label_invariance']=True
    total,vals=__import__('method').forward_train(old_t,c['utils'],b,CFG,[.01,.02,.03],0); total.backward();
    assert any(p.grad is not None and old_t.isfinite(p.grad).all() and float(p.grad.abs().sum())>0 for p in b.augment.parameters()); checks['new_module_gradient']=True
    for m in [a,b]: close_agent(m)
    z=synthetic_audit(n=2048); checks['synthetic_pid']=True
    m=read(EXP/'manifest.json')
    entry={'status':'PASS','gpu':gpu,'checks':checks,'synthetic':z,'signature':signature(),'seconds':time.time()-m['started_unix']}
    write(RT/f'preflight_gpu{gpu}.json',entry)
    # Keep per-device records without assigning the same dictionary back into
    # itself (which made the manifest JSON circular on the first preflight).
    m.setdefault('preflight_by_gpu',{})[str(gpu)]=entry
    m['preflight_status']='PASS'
    write(EXP/'manifest.json',m);print('PREFLIGHT_PASS',gpu,flush=True)
def _fit_labels(x,train,k,seed):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    out=[]
    for m,(lo,hi) in enumerate(((0,48),(48,54),(54,64),(64,320))):
        z=StandardScaler().fit_transform(x[train,lo:hi]);
        out.append(KMeans(n_clusters=k,n_init=10,random_state=seed+m).fit_predict(z))
    return out
def _pid_on(labels,y,idx,m,n,k):
    # _fit_labels is fit on x[train] and therefore returns train-local rows;
    # only the target labels need indexing by the outer-train indices.
    return estimate_pid(labels[m],labels[n],y[idx],k,k)
def pid_phase():
    x,y,folds=data(); pairs=[('PA',0,1),('PF',0,2),('PC',0,3),('AF',1,2),('AC',1,3),('FC',2,3)]; primary=[]; diag=[]; weights=[]; all_sens=[]
    for fold,(tr,te) in enumerate(folds,1):
        for k in CFG['pid']['cluster_k_sensitivity']+[CFG['pid']['cluster_k_primary']]:
            labels=_fit_labels(x,tr,k,45000+fold*100+k); 
            for pair,i,j in pairs:
                z=_pid_on(labels,y,tr,i,j,k); row={'fold':fold,'k':k,'pair':pair,'R':z['R'],'U1':z['U1'],'U2':z['U2'],'S':z['S'],'I1':z['I1'],'I2':z['I2'],'Ijoint':z['Ijoint'],'identity_residual':z['identity_residual'],'minimum_component':z['minimum_component'],'redundancy_fraction':z['redundancy_fraction'],'solver_success':z['solver']['success'],'solver_status':z['solver']['status'],'marginal_error':z['solver']['marginal_error'],'n_train':len(tr)}; primary.append(row)
                if k==4 and pair in ('AF','AC','FC'):
                    rng=np.random.default_rng(45000+fold*1000+sum(ord(c) for c in pair)); boots=[]; perms=[]; siteperms=[]
                    site=np.argmax(x[tr,:26+20],axis=1) # age/site summary; site-specific result is a sensitivity, not a feature
                    for b in range(CFG['pid']['bootstrap_replicates']):
                        ix=rng.integers(0,len(tr),len(tr)); q=_pid_on(labels,y,tr[ix],i,j,4); boots.append(q['R'])
                    for b in range(CFG['pid']['permutation_replicates']):
                        yp=y[tr].copy(); rng.shuffle(yp); q=estimate_pid(labels[i],labels[j],yp,4,4); perms.append(q['R'])
                        ys=yp.copy()
                        for s in np.unique(site):
                            where=np.where(site==s)[0]; ys[where]=ys[rng.permutation(where)] if len(where)>1 else ys[where]
                        q2=estimate_pid(labels[i],labels[j],ys,4,4); siteperms.append(q2['R'])
                    qval=float(np.clip((np.quantile(boots,.10)-np.quantile(perms,.95))/(np.median(boots)+1e-12),0,1)); base=z['R']+z['U1']+z['U2']+z['S']
                    diag.append({'fold':fold,'pair':pair,'k':4,'R_boot_q10':float(np.quantile(boots,.10)),'R_boot_median':float(np.median(boots)),'R_boot_q90':float(np.quantile(boots,.90)),'R_perm_q95':float(np.quantile(perms,.95)),'R_site_perm_q95':float(np.quantile(siteperms,.95)),'q':qval,'bootstrap_n':len(boots),'perm_n':len(perms),'site_perm_n':len(siteperms),'R_stable':float(np.std(boots)),'I_identity_residual':z['identity_residual'],'base_Ijoint':base,'solver_failures':sum(not np.isfinite(v) for v in boots+perms+siteperms)})
                    all_sens.append((fold,pair,z,qval))
    csvwrite(RES/'pid_pairs.csv',primary); csvwrite(RES/'pid_diagnostics.csv',diag)
    by={(f,p):q for f,p,z,q in all_sens};
    for f in range(1,11):
        vals={p:by[(f,p)] for p in ('AF','AC','FC')}; bar=sum(vals.values())/3
        for p in ('AF','AC','FC'):
            # A zero-q relation remains explicitly zero; no hand-selected nonzero floor.
            weights.append({'fold':f,'k':'4','pair':p,'q':vals[p],'r':next(z['redundancy_fraction'] for ff,pp,z,q in all_sens if ff==f and pp==p),'lambda':CFG['protocol']['lambda_max']*vals[p]*next(z['redundancy_fraction'] for ff,pp,z,q in all_sens if ff==f and pp==p),'lambda_bar':CFG['protocol']['lambda_max']*bar*sum(next(z['redundancy_fraction'] for ff,pp,z,q in all_sens if ff==f and pp==p2) for p2 in ('AF','AC','FC'))/3})
    csvwrite(RES/'pid_weights.csv',weights)
    valid=[r for r in primary if int(r['k'])==4]; failures=[r for r in valid if not str(r['solver_success']).lower()=='true' or float(r['identity_residual'])>CFG['pid']['reliability']['max_identity_residual_bit']]
    lam=[float(r['lambda']) for r in weights]; positive=sum(v>0 for v in lam); heterogeneous=(max(lam)-min(lam))>CFG['pid']['reliability']['min_lambda_range']
    reliability='PID_RELIABLE_AND_HETEROGENEOUS' if not failures and positive>=CFG['pid']['reliability']['min_positive_lambda_relations'] and heterogeneous else ('PID_RELIABLE_BUT_NONDISCRIMINATIVE' if not failures else 'PID_UNRELIABLE')
    m=read(EXP/'manifest.json');m.update(pid={'status':reliability,'primary_rows':len(valid),'diagnostic_rows':len(diag),'positive_lambdas':positive,'lambda_range':max(lam)-min(lam),'failures':len(failures)});m['status']='PID_PASS' if reliability=='PID_RELIABLE_AND_HETEROGENEOUS' else 'PID_NO_GO';write(EXP/'manifest.json',m)
    print('PID_STATUS',reliability,'positive',positive,'range',max(lam)-min(lam),'failures',len(failures),flush=True)
    return reliability
def prepare_e0_rows():
    out=[]
    for r in historical_rows().values():
        x=dict(r); x['group_id']=0;x['group']='E0_B0B';x['status']='REUSED_VERIFIED';x['source']='E43.2';out.append(x)
    return out
def completed(g,f):
    p=RT/'tasks'/f'g{g}_f{f:02}.json';
    if not p.exists():return False
    q=read(p); return q.get('status')=='COMPLETE' and q.get('signature')==signature() and q.get('best_sha')==sha(RT/'checkpoints'/f'g{g}_f{f:02}.best.pt')
def execute_fold(g,f,gpu):
    if completed(g,f): print('SKIP_VERIFIED',g,f,flush=True);return
    x,y,folds=data(); tr,te=folds[f-1]; lamb=pid_for_fold(f); a=build_from_anchor(f,g,lamb); started=time.time(); best=None;bestacc=0.;wait=0;bestep=-1;trace=[];begin=0;key=f'g{g}_f{f:02}'; latest=RT/'checkpoints'/f'{key}.latest.pt'; bestp=RT/'checkpoints'/f'{key}.best.pt';
    base={'group_id':g,'group':CFG['groups'][str(g)]['name'],'seed':0,'fold':f,'physical_gpu':gpu,'signature':signature(),'split_sha':fold_sha(tr,te),'anchor_sha':sha(RT/'anchors'/f'fold{f:02}.pt'),'initial_gfr_sha':state_hash(runtime()['torch'],a.gfr.state_dict()),'initial_pln_sha':state_hash(runtime()['torch'],a.pln.state_dict()),'augment_init_sha':state_hash(runtime()['torch'],a.augment.state_dict()),'pid_lambdas':lamb}
    if latest.exists():
        s=load(latest); assert s['base']==base; set_model(a,s['model']);a.mp.load_state_dict(s['mp']);a.mf.load_state_dict(s['mf']);a.newopt.load_state_dict(s['newopt']);restore(s['rng']);begin=s['epoch']+1;best=s['best'];bestacc=s['bestacc'];bestep=s['bestep'];wait=s['wait'];trace=s['trace']
    for ep in range(begin,CFG['protocol']['nepoch']):
        total,vals=train_step(runtime()['torch'],runtime()['utils'],a,CFG,[lamb.get(p,0.) for p in ('AF','AC','FC')],ep); out=evaluate(runtime()['torch'],a)[0]; acc=float(out[a.val_idx].argmax(1).eq(a.targ[a.val_idx]).double().mean().item())
        if acc>bestacc: bestacc=acc;bestep=ep;wait=0;best={'model':model_state(a),'mp':copy.deepcopy(a.mp.state_dict()),'mf':copy.deepcopy(a.mf.state_dict()),'newopt':copy.deepcopy(a.newopt.state_dict()),'rng':copy.deepcopy(rng()),'epoch':ep,'bestacc':acc}
        else:wait+=1
        if ep%10==0 or ep==449:trace.append({'epoch':ep,'acc':acc,'bestacc':bestacc,**vals})
        if (ep+1)%10==0 or wait>CFG['protocol']['early'] or ep==449: save(latest,{'base':base,'model':model_state(a),'mp':a.mp.state_dict(),'mf':a.mf.state_dict(),'newopt':a.newopt.state_dict(),'rng':rng(),'epoch':ep,'best':best,'bestacc':bestacc,'bestep':bestep,'wait':wait,'trace':trace})
        if wait>CFG['protocol']['early']: break
    assert best is not None; set_model(a,best['model']); out=evaluate(runtime()['torch'],a)[0]; met=metric_row(out,y,te); pred=np.asarray(met.pop('prediction'),np.int64); best_art={'base':base,**best,'metrics':met,'prediction':pred.tolist(),'test_indices':te.tolist(),'epochs_executed':ep+1,'trace':trace,'best_epoch':bestep}
    save(bestp,best_art); row={**base,**met,'best_epoch':bestep,'epochs_executed':ep+1,'prediction_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),'best_sha':sha(bestp),'status':'COMPLETE','wall_seconds':time.time()-started,'trace':trace}; write(RT/'tasks'/f'{key}.json',row); latest.unlink(missing_ok=True);close_agent(a);print('FOLD_DONE',key,'gpu',gpu,'acc',row['acc'],'best',bestep,flush=True)
def worker(gpu):
    assert read(EXP/'manifest.json')['status']=='PID_PASS';
    for g in range(1,7):
        for f in CFG['gpu_folds'][str(gpu)]: execute_fold(g,f,gpu)
def launch(cmd,gpu,label):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)); pth=RT/'logs'/f'{label}.log'; log=pth.open('a',buffering=1); p=__import__('subprocess').Popen([sys.executable,'-B',str(EXP/'run.py'),*cmd,'--gpu',str(gpu)],cwd=EXP,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);return p,log,label
def train_controller():
    m=read(EXP/'manifest.json');assert m['status']=='PID_PASS';m['status']='TRAINING';m['training_started_unix']=time.time();write(EXP/'manifest.json',m)
    live=[launch(['worker'],g,f'worker_gpu{g}') for g in (0,1)]
    while live:
        time.sleep(5)
        for item in live[:]:
            p,log,label=item
            if p.poll() is not None:
                log.close();live.remove(item)
                if p.returncode:
                    for q,_,_ in live:q.terminate()
                    raise RuntimeError(label+' failed')
    rows=refresh(); assert len([r for r in rows if int(r['group_id'])>=1])==60
    m=read(EXP/'manifest.json');m.update(status='TRAINING_COMPLETE',training_finished_unix=time.time(),new_folds=60,gpu_folds_actual={'0':30,'1':30});write(EXP/'manifest.json',m)
def refresh():
    rows=prepare_e0_rows()
    for p in sorted((RT/'tasks').glob('g*_f*.json')):
        q=read(p)
        if q.get('status')=='COMPLETE': assert completed(q['group_id'],q['fold']);rows.append(q)
    assert len({(str(r.get('group_id')),int(r['fold'])) for r in rows})==len(rows)
    rows.sort(key=lambda r:(int(r['group_id']),int(r['fold']))); csvwrite(RES/'fold_metrics.csv',rows);return rows
def audit():
    c=runtime(); rows=refresh(); assert len(rows)==70; checks=[];diags=[];x,y,folds=data()
    for q in rows:
        g=int(q['group_id']);f=int(q['fold'])
        if g==0: checks.append({'group':g,'fold':f,'status':'PASS_REUSED_VERIFIED'});continue
        ck=load(RT/'checkpoints'/f'g{g}_f{f:02}.best.pt'); a=build_from_anchor(f,g,pid_for_fold(f)); set_model(a,ck['model']); out,feat,adjs=evaluate(c['torch'],a); met=metric_row(out,y,folds[f-1][1]); assert all(abs(float(met[k])-float(q[k]))<1e-12 for k in ('acc','hard_auc','spe','sen')); pred=np.asarray(met['prediction'],np.int64); assert hashlib.sha256(pred.tobytes()).hexdigest()==q['prediction_sha256']; assert c['torch'].isfinite(out).all()
        a.capture.clear(); a.gfr.eval();
        with c['torch'].no_grad():
            n,parts=a.dataLayer(a.feat); f2,ad,al=a.gfr(parts,n,a.targ,a.trn_idx)
        msg={m:float(a.capture[m].norm().item()) for m in (1,2,3)}
        diags.append({'group':CFG['groups'][str(g)]['name'],'fold':f,'gfr_output_norm':float(f2.norm().item()),'raw_input_norm':float(a.feat.norm().item()),'gfr_minus_raw_relative':float((f2-a.feat).norm().item()/a.feat.norm().item()),'anat_message_norm':msg[1],'func_message_norm':msg[2],'corr_message_norm':msg[3],'pid_lambda_af':q['pid_lambdas']['AF'],'pid_lambda_ac':q['pid_lambdas']['AC'],'pid_lambda_fc':q['pid_lambdas']['FC'],'finite':True})
        checks.append({'group':g,'fold':f,'status':'PASS','best_sha':sha(RT/'checkpoints'/f'g{g}_f{f:02}.best.pt')});close_agent(a)
    for path,h in read(EXP/'manifest.json')['protected_sha256'].items(): assert sha(path)==h
    write(RT/'audit.json',{'status':'PASS','count':len(checks),'checks':checks});csvwrite(RES/'mechanism_diagnostics.csv',diags);m=read(EXP/'manifest.json');m.update(independent_readback='PASS_70_ROWS',protected_files_unchanged=True,status='AUDIT_PASS');write(EXP/'manifest.json',m);print('AUDIT_PASS',len(checks),flush=True)
def report():
    rows=refresh(); m0=read(EXP/'manifest.json'); expected=70 if m0.get('status') in ('TRAINING_COMPLETE','AUDIT_PASS','SCIENTIFIC_COMPLETE') else 10
    assert len(rows)==expected
    rel=m0.get('pid',{}).get('status','UNKNOWN')
    lines=['# Experiment 45 — PID-DGAA','',f'状态：E0复用10折，E1–E6新训练{max(0,len(rows)-10)}/60折；PID准入={rel}。','', '|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|','|---|---:|---:|---:|---:|']; summaries=[]
    for g in range(7):
        rr=[r for r in rows if int(r['group_id'])==g];
        if len(rr)!=10:continue
        d={'group':CFG['groups'][str(g)]['name'],'folds':10};
        for k in CFG['formal_metrics']:
            v=100*np.asarray([float(r[k]) for r in rr]); d[k+'_mean']=float(v.mean());d[k+'_sd']=float(v.std(ddof=0))
        summaries.append(d);lines.append('|'+d['group']+'|'+'|'.join(f"{d[k+'_mean']:.4f} ± {d[k+'_sd']:.4f}" for k in CFG['formal_metrics'])+'|')
    csvwrite(RES/'summary.csv',summaries)
    base={int(r['group_id']):{int(r['fold']):r for r in rows if int(r['group_id'])==int(r['group_id'])} for r in rows}
    e4={int(r['fold']):r for r in rows if int(r['group_id'])==4}
    lines+=['','## E4逐折配对','','|比较|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|ACC W/T/L|AUC W/T/L|','|---|---:|---:|---:|---:|---|---|']
    paired=[]
    for g in (0,1,3,5,6):
        if not e4 or not any(int(r['group_id'])==g for r in rows):
            continue
        rr={int(r['fold']):r for r in rows if int(r['group_id'])==g}; ds={k:[100*(float(e4[f][k])-float(rr[f][k])) for f in range(1,11)] for k in CFG['formal_metrics']}; w=lambda z:'/'.join(map(str,[sum(v>1e-10 for v in z),sum(abs(v)<=1e-10 for v in z),sum(v< -1e-10 for v in z)])); lines.append(f"|E4−{CFG['groups'][str(g)]['name']}|"+'|'.join(f'{np.mean(ds[k]):+.4f}' for k in CFG['formal_metrics'])+f'|{w(ds["acc"])}|{w(ds["hard_auc"])}|'); paired.append({'comparison':f'E4-E{g}',**{f'delta_{k}_pp':float(np.mean(ds[k])) for k in CFG['formal_metrics']},'acc_wtl':w(ds['acc']),'hard_auc_wtl':w(ds['hard_auc'])})
    csvwrite(RES/'paired_comparisons.csv',paired)
    pdiag=csvread(RES/'pid_diagnostics.csv'); pw=csvread(RES/'pid_weights.csv');
    lines+=['','## PID主结果','', '|关系|平均R|平均U1|平均U2|平均S|平均冗余占比|平均q|平均lambda|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for pair in ('PA','PF','PC','AF','AC','FC'):
        pp=[r for r in csvread(RES/'pid_pairs.csv') if r['pair']==pair and int(r['k'])==4]; ww=[r for r in pw if r['pair']==pair] if pair in ('AF','AC','FC') else []
        if pp: lines.append('|'+pair+'|'+'|'.join(f'{np.mean([float(r[k]) for r in pp]):.6f}' for k in ('R','U1','U2','S','redundancy_fraction'))+'|'+(f'{np.mean([float(r["q"]) for r in ww]):.6f}|{np.mean([float(r["lambda"]) for r in ww]):.6f}|' if ww else '—|—|'))
    lines+=['','## 机制与限制','PID使用训练折聚类、I_min离散分解、12次bootstrap和19次标签置乱；未使用外层测试标签。E0来自实验43.2 verified B0-B，新增组从同折未训练anchor构造。图节点为患者，ANAT/FUNC是质量/头动摘要，PHENO分支的原始 i==1 覆盖行为保持。历史 val=test、CV前全局RFE/标准化、transductive图、单seed和PID有限样本估计误差意味着这些结果只支持内部机制比较。','', '## 判定', 'E4只有在同时超过E0、E3、E5且图消息对主分类有可测依赖时才满足预注册GO；否则报告NO-GO，不以辅助正则或单次均值提升替代PID关系证据。', '', f'产物：`{EXP}`。']
    if rel!='PID_RELIABLE_AND_HETEROGENEOUS':
        lines += ['', '## 正式训练闸门', f'PID准入未达到 `PID_RELIABLE_AND_HETEROGENEOUS`（实际为 `{rel}`），因此按预注册协议未启动E1–E6的60折训练；这不是缺失结果，而是PID关系没有通过非零且异质的训练内可用性门槛。']
    (RES/'report.md').write_text('\n'.join(lines)+'\n');m=read(EXP/'manifest.json');m.update(summary=summaries,paired=paired,status='SCIENTIFIC_COMPLETE');write(EXP/'manifest.json',m);print('\n'.join(lines[:60]),flush=True)
def handoff():
    p=ROOT/'handoff.md';m=read(EXP/'manifest.json');marker='<!-- EXPERIMENT45_PID_DGAA -->';text='\n\n'+marker+'\n## Experiment 45 — PID-DGAA\n\n'+(RES/'report.md').read_text()
    try:
        with p.open('r+b') as f:
            fcntl.flock(f,fcntl.LOCK_EX);before=f.read();pr=m['handoff_prefix'];assert hashlib.sha256(before[:pr['bytes']]).hexdigest()==pr['sha256']; old=hashlib.sha256(before).hexdigest()
            if marker.encode() not in before:f.seek(0,2);f.write(text.encode());f.flush();os.fsync(f.fileno())
            f.seek(0);after=f.read();assert after[:len(before)]==before
        m['handoff']={'status':'APPENDED','prefix_sha256':old,'after_sha256':hashlib.sha256(after).hexdigest()};m['delivery']='COMPLETE';write(EXP/'manifest.json',m);print('HANDOFF_APPENDED',flush=True)
    except Exception as e:
        (EXP/'handoff_append_pending.md').write_text(text+'\nERROR '+repr(e));m['handoff']={'status':'PENDING','error':repr(e)};m['delivery']='PARTIAL';write(EXP/'manifest.json',m);print('HANDOFF_PENDING',repr(e),flush=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=['prepare','preflight','pid','train','worker','audit','report','handoff','all']); ap.add_argument('--gpu',type=int,default=0); a=ap.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='preflight':preflight(a.gpu)
    elif a.command=='pid':pid_phase()
    elif a.command=='train':train_controller()
    elif a.command=='worker':worker(a.gpu)
    elif a.command=='audit':audit()
    elif a.command=='report':report()
    elif a.command=='handoff':handoff()
    elif a.command=='all':
        prepare();preflight(0);preflight(1);rel=pid_phase()
        if rel=='PID_RELIABLE_AND_HETEROGENEOUS':train_controller();audit();report();handoff()
        else: report();handoff()
if __name__=='__main__':
    try: main()
    except Exception: traceback.print_exc(); raise
