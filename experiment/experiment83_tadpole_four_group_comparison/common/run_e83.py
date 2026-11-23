from __future__ import annotations
import argparse, csv, hashlib, json, os, random, sys, time, traceback
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

ROOT=Path('/model/spromptgl/experiment/experiment83_tadpole_four_group_comparison')
sys.path.insert(0,str(ROOT/'common'))
from tadpole_model import E83Model
from metrics import calculate_metrics, summarize, CLASS_NAMES

GROUPS=['G1_E28_E0','G2_E43_1_E00','G3_E74_4_G2','G4_E82_4_6_10_G1']
DIMS=[145,148,3,36,24,9]
NCLASS=3
SEED=0
MAX_EPOCHS=450
PATIENCE=200
PROMPT_PATH=str(ROOT/'sources/E82_dependencies/prompts.pt')
METRIC_FIELDS=['ACC','AUC_original','AUC_probability','SPE_macro','SEN_macro','F1_macro','F1_weighted','Balanced_Accuracy','NC_precision','NC_recall','NC_f1','MCI_precision','MCI_recall','MCI_f1','AD_precision','AD_recall','AD_f1']

def seed_everything(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_data():
    df=pd.read_csv(ROOT/'preprocessing/processed_standard_data.csv')
    ids=pd.read_csv(ROOT/'preprocessing/sample_ids.csv')
    X=df.drop(columns=['LABEL']).to_numpy(np.float32)
    y=df.LABEL.to_numpy(np.int64)-1
    assert X.shape==(598,365) and len(ids)==598
    splits=json.loads((ROOT/'data_audit/split_indices.json').read_text())['splits']
    return X,y,ids,splits

def make_pairs(train_idx, y, seed):
    rng=np.random.default_rng(seed)
    train_idx=np.asarray(train_idx,dtype=int)
    pos=[]; neg=[]
    for p,i in enumerate(train_idx):
        for j in train_idx[p+1:]:
            (pos if y[i]==y[j] else neg).append((int(i),int(j)))
    rng.shuffle(pos); rng.shuffle(neg)
    k=min(2048,len(pos),len(neg))
    selected=pos[:k]+neg[:k]; rng.shuffle(selected)
    pi=torch.tensor([a for a,b in selected],dtype=torch.long)
    pj=torch.tensor([b for a,b in selected],dtype=torch.long)
    pt=torch.tensor([1.0 if y[a]==y[b] else 0.0 for a,b in selected],dtype=torch.float32)
    return pi,pj,pt,{'positive_pairs':k,'negative_pairs':k,'total_pairs':len(selected),'rng_seed':int(seed)}

def optimizer_for(model, group):
    if group in ('G1_E28_E0','G2_E43_1_E00'):
        return torch.optim.Adam(model.parameters(),lr=0.012,weight_decay=0.0036), {'base_lr':0.012,'base_weight_decay':0.0036}
    darg=[]; dpsl=[]; base=[]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        if name.startswith('dpsl.'):
            dpsl.append(p)
        elif name.startswith('gfr.relation_') or name.startswith('gfr.darg_gate'):
            darg.append(p)
        else: base.append(p)
    groups=[{'params':base,'lr':0.012,'weight_decay':0.0036},{'params':darg,'lr':0.001,'weight_decay':0.01}]
    if dpsl: groups.append({'params':dpsl,'lr':0.0005,'weight_decay':0.01})
    return torch.optim.AdamW(groups), {'base_lr':0.012,'base_weight_decay':0.0036,'darg_lr':0.001,'darg_weight_decay':0.01,'dpsl_lr':0.0005,'dpsl_weight_decay':0.01}

def row_from_metrics(group,fold,split,best_epoch,best_val_acc,ckpt,metrics,seed=0,epoch_count=0):
    row={'group':group,'fold':fold,'seed':seed,'train_count':len(split['train_idx']),'test_count':len(split['test_idx']),
         'best_epoch':best_epoch,'best_val_acc':best_val_acc,'checkpoint_path':str(ckpt),'epoch_count':epoch_count,'val_equals_test':True}
    row.update({k:metrics[k] for k in METRIC_FIELDS}); return row

def write_csv(path, rows):
    if not rows:return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def train_fold(group,fold,X,y,ids,split,device,log):
    gdir=ROOT/group
    for sub in ['best_checkpoints','predictions','confusion_matrices','audits']:(gdir/sub).mkdir(parents=True,exist_ok=True)
    ckpt=gdir/'best_checkpoints'/f'fold{fold:02d}.pt'
    foldcsv=gdir/'fold_metrics.csv'
    # A complete, finite fold row plus a checkpoint is the resume contract.
    if (not os.environ.get('E83_FORCE_RERUN')) and ckpt.exists() and foldcsv.exists():
        try:
            old=pd.read_csv(foldcsv); row=old[old.fold==fold]
            if len(row)==1 and bool(np.isfinite(row[METRIC_FIELDS].to_numpy(float)).all()):
                log.write(f'fold {fold:02d} already complete, resume skip\n'); log.flush(); return row.iloc[0].to_dict()
        except Exception: pass
    seed_everything(SEED)
    tr=np.asarray(split['train_idx'],dtype=int); te=np.asarray(split['test_idx'],dtype=int)
    if group.startswith(('G3','G4')):
        pair_cpu,pj_cpu,pt_cpu,pair_info=make_pairs(tr,y,532108+fold)
    else:
        pair_cpu=torch.empty(0,dtype=torch.long); pj_cpu=torch.empty(0,dtype=torch.long); pt_cpu=torch.empty(0,dtype=torch.float32); pair_info={'positive_pairs':0,'negative_pairs':0,'total_pairs':0,'rng_seed':None,'used_for_training':False}
    model=E83Model(DIMS,NCLASS,10,4,2,0.0,group,PROMPT_PATH,threshold=0.9).to(device)
    opt,opt_cfg=optimizer_for(model,group)
    tensor_x=torch.from_numpy(X).to(device); tensor_y=torch.from_numpy(y).long().to(device); train_t=torch.from_numpy(tr).long().to(device); test_t=torch.from_numpy(te).long().to(device)
    pair=(pair_cpu.to(device),pj_cpu.to(device),pt_cpu.to(device)) if group.startswith(('G3','G4')) else None
    counts=np.bincount(y[tr],minlength=NCLASS).astype(float); weights=torch.tensor(len(tr)/(np.maximum(counts,1)*NCLASS),dtype=torch.float32,device=device)
    best=-1.0; best_epoch=-1; wait=0; best_state=None; last_stats={}; start_time=time.time(); history=[]; first_grad={}
    for epoch in range(1,MAX_EPOCHS+1):
        model.train(); opt.zero_grad(set_to_none=True)
        logp,aux=model(tensor_x,tensor_y,train_t,pair)
        cls=F.nll_loss(logp[train_t],tensor_y[train_t],weight=weights)
        total=cls+aux['graph_loss']+0.05*aux['pair_loss']+0.01*aux['preserve_loss']+aux['dpsl_loss']
        if not torch.isfinite(total): raise RuntimeError(f'non-finite loss group={group} fold={fold} epoch={epoch}')
        total.backward()
        if epoch==1:
            first_grad={'relation_grad_norm':float(sum(p.grad.detach().norm() for n,p in model.named_parameters() if p.grad is not None and n.startswith('gfr.relation_'))),
                        'dpsl_grad_norm':float(sum(p.grad.detach().norm() for n,p in model.named_parameters() if p.grad is not None and n.startswith('dpsl.')))}
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        model.eval()
        with torch.no_grad(): eval_logp,eval_aux=model(tensor_x,tensor_y,train_t,pair)
        pred=eval_logp.argmax(1); val_acc=float((pred[test_t]==tensor_y[test_t]).float().mean().item())
        history.append({'epoch':epoch,'loss':float(total.detach()),'cls_loss':float(cls.detach()),'graph_loss':float(aux['graph_loss'].detach()),'pair_loss':float(aux['pair_loss'].detach()),'dpsl_loss':float(aux['dpsl_loss'].detach()),'val_acc':val_acc})
        if epoch==1 or epoch%10==0: log.write(f'group={group} fold={fold:02d} epoch={epoch:03d} loss={float(total):.6f} val_acc={val_acc:.6f}\n'); log.flush()
        if val_acc>best:
            best=val_acc; best_epoch=epoch; wait=0
            best_state={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best_val_acc':best,'group':group,'fold':fold,'seed':SEED,'val_idx_equals_test_idx':True,'split_sha256':sha(ROOT/'data_audit/split_indices.json'),'config':{'dropout':0.0,'lr':0.012,'n_hidden':10,'nlayer':2,'n_head':4,'alpha':1.0,'max_epochs':MAX_EPOCHS,'early':PATIENCE,'modal_dims':DIMS,'nclass':3}}
            torch.save(best_state,ckpt)
        else:
            wait+=1
            if wait>PATIENCE: break
    if best_state is None: raise RuntimeError('no checkpoint created')
    # Reload best checkpoint, then produce all requested predictions from that exact state.
    load=torch.load(ckpt,map_location=device,weights_only=False); model.load_state_dict(load['model']); model.eval()
    with torch.no_grad(): lp,aux=model(tensor_x,tensor_y,train_t,pair)
    probs=lp.exp().detach().cpu().numpy(); preds=probs.argmax(1); metric=calculate_metrics(y[te],preds[te],probs[te],NCLASS)
    pred_path=gdir/'predictions'/f'fold{fold:02d}.csv'
    pred_df=pd.DataFrame({'fold':fold,'seed':SEED,'sample_index':te,'sample_id':ids.iloc[te]['RID'].to_numpy(),'true_label':y[te],'true_label_name':[CLASS_NAMES[i] for i in y[te]],'pred_label':preds[te],'pred_label_name':[CLASS_NAMES[i] for i in preds[te]],'prob_NC':probs[te,0],'prob_MCI':probs[te,1],'prob_AD':probs[te,2]})
    pred_df.to_csv(pred_path,index=False)
    (gdir/'confusion_matrices'/f'fold{fold:02d}.json').write_text(json.dumps({'fold':fold,'labels':list(CLASS_NAMES),'matrix':metric['confusion_matrix']},indent=2))
    # Mechanism audit includes direct evidence that each requested branch participated.
    with torch.enable_grad():
        model.train(); opt.zero_grad(set_to_none=True); lp2,a2=model(tensor_x,tensor_y,train_t,pair); l2=F.nll_loss(lp2[train_t],tensor_y[train_t],weight=weights)+a2['graph_loss']+0.05*a2['pair_loss']+0.01*a2['preserve_loss']+a2['dpsl_loss']; l2.backward()
    relation_delta=float(sum((a-b).norm().detach().cpu() for a,b in zip(a2['effective_adjacencies'],a2['adjacencies']))) if model.use_darg else 0.0
    dpsl_ratio=float(a2['dpsl']['residual_ratio'].mean().detach().cpu()) if model.use_dpsl else 0.0
    mech={'group':group,'fold':fold,'seed':SEED,'pen_present':bool(model.pen is not None),'darg_active':bool(model.use_darg),'dpsl_active':bool(model.use_dpsl),'relation_update_norm':relation_delta,'darg_relation_grad_norm':float(sum(p.grad.detach().norm() for n,p in model.named_parameters() if p.grad is not None and n.startswith('gfr.relation_'))),'dpsl_gradient_norm':float(sum(p.grad.detach().norm() for n,p in model.named_parameters() if p.grad is not None and n.startswith('dpsl.'))),'dpsl_residual_ratio_mean':dpsl_ratio,'dpsl_residual_ratio_max':float(a2['dpsl']['residual_ratio'].max().detach().cpu()) if model.use_dpsl else 0.0,'pair_info':pair_info,'predicted_class_counts':np.bincount(preds[te],minlength=3).tolist(),'test_class_counts':np.bincount(y[te],minlength=3).tolist(),'ad_recall':metric['AD_recall'],'best_epoch':best_epoch,'best_checkpoint_criterion':'val/test ACC','checkpoint':str(ckpt)}
    (gdir/'audits'/f'fold{fold:02d}.json').write_text(json.dumps(mech,indent=2))
    row=row_from_metrics(group,fold,split,best_epoch,best,ckpt,metric,SEED,epoch)
    row['train_seconds']=time.time()-start_time; row['prediction_path']=str(pred_path); row['mechanism_audit_path']=str(gdir/'audits'/f'fold{fold:02d}.json')
    # Update fold CSV atomically enough for resume. Existing successful rows are retained.
    existing=[]
    if foldcsv.exists():
        try: existing=pd.read_csv(foldcsv).to_dict('records')
        except Exception: existing=[]
    existing=[r for r in existing if int(r.get('fold',-1))!=fold]; existing.append(row); existing.sort(key=lambda r:int(r['fold']))
    write_csv(foldcsv,existing)
    (gdir/f'train_history_fold{fold:02d}.json').write_text(json.dumps(history,indent=2))
    log.write(f'COMPLETE group={group} fold={fold:02d} best_epoch={best_epoch} best_val_acc={best:.6f} ACC={metric["ACC"]:.6f} AUC_original={metric["AUC_original"]:.6f} seconds={time.time()-start_time:.1f}\n'); log.flush()
    return row

def run_group(group):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X,y,ids,splits=load_data(); gdir=ROOT/group; gdir.mkdir(exist_ok=True)
    (gdir/'config.json').write_text(json.dumps({'group':group,'dataset':'TADPOLE','seed':0,'n_splits':10,'cv_random_state':10,'n_samples':598,'nclass':3,'nmodal':6,'modal_dims':DIMS,'dropout':0.0,'lr':0.012,'n_hidden':10,'nlayer':2,'n_head':4,'alpha':1.0,'max_epochs':MAX_EPOCHS,'early':PATIENCE,'checkpoint_criterion':'val=test ACC','rfe':'full data before outer CV','device':str(device),'actual_concurrency':1,'worker_limit':10,'optimizer_protocol':'base Adam 0.012/0.0036, E74 DARG AdamW 0.001/0.01, E82 DPSL AdamW 0.0005/0.01','adapter_version':'E83-DPSL-v2 true CAM-margin gradient constraint and medical-anonymous prompt differential'},indent=2))
    log=open(gdir/'train.log','a',buffering=1)
    log.write(f'BEGIN group={group} device={device} seed={SEED} folds=10\n')
    rows=[]
    try:
        for split in splits:
            rows.append(train_fold(group,int(split['fold']),X,y,ids,split,device,log))
        rows=[]
        rows=pd.read_csv(gdir/'fold_metrics.csv').to_dict('records')
        summary=summarize(rows,METRIC_FIELDS)
        summary.update({'group':group,'dataset':'TADPOLE','seed':SEED,'folds_complete':len(rows),'checkpoint_protocol':'val_idx == test_idx, best ACC','rfe_protocol':'full-data RFE before outer CV','data_sha256':sha(ROOT/'preprocessing/processed_standard_data.csv'),'split_sha256':sha(ROOT/'data_audit/split_indices.json')})
        (gdir/'summary.json').write_text(json.dumps(summary,indent=2))
        log.write(f'GROUP COMPLETE {group} folds={len(rows)}\n')
    except Exception as exc:
        log.write('ERROR '+repr(exc)+'\n'+traceback.format_exc()); log.flush(); raise
    finally: log.close()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('group',choices=GROUPS+['all']); args=ap.parse_args()
    groups=GROUPS if args.group=='all' else [args.group]
    for g in groups: run_group(g)
if __name__=='__main__': main()
