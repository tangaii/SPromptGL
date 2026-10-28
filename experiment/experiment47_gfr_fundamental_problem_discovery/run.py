#!/usr/bin/env python3
"""Experiment 47 Stage A/B.

This program performs read-only fixed-checkpoint audits.  It has no training
loop for GFR and no new model module.  Each fold writes one compact JSON file;
the final process merges those files into the required CSV artefacts.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, sys, time
from pathlib import Path
import numpy as np

EXP=Path(__file__).resolve().parent
RT=EXP/'.runtime'; RES=EXP/'results'; OUT=RT/'fold_outputs'
ROOT=Path('/data/adodas/SPromptGL-main')
sys.path.insert(0,str(EXP))
import integration as I

def jdump(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(_clean(obj),ensure_ascii=False,indent=2,allow_nan=False,default=_json)+'\n')
    os.replace(tmp,path)
def _json(x):
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,)): return float(x)
    if isinstance(x,np.ndarray): return x.tolist()
    raise TypeError(type(x).__name__)
def _clean(x):
    if isinstance(x,dict): return {k:_clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [_clean(v) for v in x]
    if isinstance(x,(float,np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x,np.ndarray): return _clean(x.tolist())
    if isinstance(x,(np.integer,)): return int(x)
    return x
def fnum(x):
    x=float(x); return x if math.isfinite(x) else None

def row_metric(logp,y,test):
    return I.metric_np(np.asarray(logp),y,test)
def pred_stats(base_pred,new_pred,y,test):
    b=np.asarray(base_pred); p=np.asarray(new_pred); t=np.asarray(y)[np.asarray(test)]
    return dict(flip=float(np.mean(b!=p)),base_correct_new_wrong=int(np.sum((b==t)&(p!=t)),),
                base_wrong_new_correct=int(np.sum((b!=t)&(p==t))),
                base_correct_new_correct=int(np.sum((b==t)&(p==t))),
                base_wrong_new_wrong=int(np.sum((b!=t)&(p!=t))))

def graph_components(a):
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        q=np.asarray(a).copy(); np.fill_diagonal(q,0); q=(q>0)|(q.T>0)
        n,c=connected_components(csr_matrix(q.astype(np.int8)),directed=False,return_labels=True)
        return int(n),float(np.bincount(c).max()/len(c))
    except Exception:
        return -1,float('nan')

def row_stats(a, y, site, train):
    a=np.asarray(a); n=a.shape[0]; tr=np.asarray(train); b=a[np.ix_(tr,tr)]
    whole=I.adjacency_stats(a,y,site); trst=I.adjacency_stats(b,np.asarray(y)[tr],np.asarray(site)[tr])
    bl=np.asarray(y)[tr]; bs=np.asarray(site)[tr]; bo=b.copy(); np.fill_diagonal(bo,0)
    same=(bs[:,None]==bs[None,:]); cross=~same; same &= (bo>0); cross &= (bo>0)
    labpair=(bl[:,None]==bl[None,:])
    whole['train_within_site_homophily']=float((bo*labpair*same).sum()/(bo*same).sum()) if (bo*same).any() else np.nan
    whole['train_cross_site_homophily']=float((bo*labpair*cross).sum()/(bo*cross).sum()) if (bo*cross).any() else np.nan
    whole['train_label_prevalence_null']=float((np.mean(bl==0)**2+np.mean(bl==1)**2))
    c,largest=graph_components(a); tc,tl=graph_components(b)
    whole.update(dict(train_edges=trst['edges'],train_density=trst['density'],train_degree_mean=trst['degree_mean'],
                      train_homophily_unweighted=trst.get('homophily_unweighted',np.nan),
                      train_homophily_weighted=trst.get('homophily_weighted',np.nan),
                      train_same_site_edge_rate=trst.get('same_site_edge_rate',np.nan),
                      train_same_site_weighted=trst.get('same_site_weighted',np.nan),
                      components=c,largest_component=largest,train_components=tc,train_largest_component=tl))
    # A degree-preserving row randomization provides an explicit null rather
    # than comparing homophily with 0.5.  It is diagnostic only.
    rng=np.random.default_rng(470000+int(train[0]) if len(train) else 470000)
    null=[]; site_null=[]
    allowed=[np.arange(n) for _ in range(n)]
    for _ in range(20):
        r=I.row_random_adjacency(a,allowed,rng)
        rb=r[np.ix_(tr,tr)]
        null.append(I.adjacency_stats(rb,np.asarray(y)[tr],np.asarray(site)[tr])['homophily_weighted'])
        rs=I.row_random_adjacency(a,allowed,rng,same_site=True,site=site)
        rsb=rs[np.ix_(tr,tr)]
        site_null.append(I.adjacency_stats(rsb,np.asarray(y)[tr],np.asarray(site)[tr])['homophily_weighted'])
    whole['null_homophily_mean']=float(np.mean(null)); whole['null_homophily_sd']=float(np.std(null))
    whole['site_null_homophily_mean']=float(np.mean(site_null)); whole['site_null_homophily_sd']=float(np.std(site_null))
    whole['homophily_null_z']=float((trst['homophily_weighted']-whole['null_homophily_mean'])/(whole['null_homophily_sd']+1e-12))
    whole['homophily_site_null_z']=float((trst['homophily_weighted']-whole['site_null_homophily_mean'])/(whole['site_null_homophily_sd']+1e-12))
    return whole

def exact_graph_losses(ctx, feat, adjs):
    c=I.modules(); t=c['torch']; u=c['utils']
    parts=I.split_features(ctx['xt'])
    gl=sum(u.GraphConstructLoss(parts[i],adjs[i],1.,.5,0.) for i in range(4))/4
    # ALoss follows the original GFR definition and is intentionally not in
    # simple-2's optimized loss.
    lab=ctx['yt']; idx=ctx['train']; n=len(lab); nl=lab.clone(); nl[nl==0]=-1
    nl=torch_index=nl.detach().clone()
    mask=np.zeros(n,dtype=bool); mask[np.asarray(idx)]=True
    nl[~c['torch'].from_numpy(mask).to(lab.device)]=0
    l1=nl.unsqueeze(1).repeat(1,n); l2=nl.t().repeat(n,1); aa=l1.eq(l2)
    target=u.normalize_adj(aa.float()+t.eye(n,device=lab.device))
    al=sum(t.linalg.norm(u.normalize_adj(a+t.eye(n,device=a.device))-target) for a in adjs)/4
    return gl,al

def gradient_row(ctx, fold):
    c=I.modules(); t=c['torch']; u=c['utils']; g=ctx['gfr']; p=ctx['cam'];
    g.zero_grad(set_to_none=True); p.zero_grad(set_to_none=True)
    parts=I.split_features(ctx['xt']); raw=t.cat(parts,1)
    feat,adjs,msgs,outs=I.gfr_extract(ctx)
    logp,hidden=p(feat)
    tr=t.as_tensor(ctx['train'],device=ctx['xt'].device)
    weights=t.bincount(ctx['yt'][tr],minlength=2).float(); weights=weights.sum()/(weights*2)
    cls=t.nn.functional.nll_loss(logp[tr],ctx['yt'][tr],weight=weights)
    gl,al=exact_graph_losses(ctx,feat,adjs)
    def vec(loss,params):
        gs=t.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        flat=[]; norms=[]
        for par,q in zip(params,gs):
            if q is not None: flat.append(q.detach().reshape(-1)); norms.append(float(q.detach().norm()))
            else: flat.append(t.zeros(par.numel(),device=ctx['xt'].device)); norms.append(0.)
        return (t.cat(flat) if flat else t.zeros(1,device=ctx['xt'].device)),sum(v*v for v in norms)**.5
    gp=list(g.parameters()); pp=list(p.parameters());
    vc,nc=vec(cls,gp); vg,ng=vec(gl,gp); va,na=vec(al,gp); vp,np_=vec(cls,pp)
    def cos(a,b): return float((a@b/(a.norm()*b.norm()+1e-12)).detach().cpu())
    rows=[dict(fold=fold,loss_cls=float(cls.detach()),loss_graph=float(gl.detach()),loss_aloss=float(al.detach()),
               gfr_cls_grad_norm=float(nc),gfr_graph_grad_norm=float(ng),gfr_aloss_grad_norm=float(na),
               cam_cls_grad_norm=float(np_),gfr_cls_graph_cosine=cos(vc,vg),gfr_cls_aloss_cosine=cos(vc,va))]
    for i in range(4):
        gp_i=list(getattr(g,f'GraphConstruct{i}').parameters())
        if gp_i:
            _,a1=vec(cls,gp_i); _,a2=vec(gl,gp_i); _,a3=vec(al,gp_i)
        else: a1=a2=a3=0.
        mp_i=list(getattr(g,f'MessagePassing{i}').parameters()); _,b1=vec(cls,mp_i)
        rows.append(dict(fold=fold,loss_cls=float(cls.detach()),loss_graph=float(gl.detach()),loss_aloss=float(al.detach()),
                         modality=I.MODALITIES[i],graphlearn_cls_grad_norm=float(a1),graphlearn_graph_grad_norm=float(a2),
                         graphlearn_aloss_grad_norm=float(a3),gcn_cls_grad_norm=float(b1),
                         graphlearn_w_grad_norm=0.,graphlearn_t_grad_norm=0.,graphlearn_threshold_grad_norm=0.))
    # Record the actual per-branch messages and layer contraction.
    for i,a in enumerate(adjs):
        nax=I.norm_adj(a); l1=t.relu(g.__getattr__(f'MessagePassing{i}').gcn1(raw,nax)); l2=g.__getattr__(f'MessagePassing{i}').gcn2(l1,nax)
        rows.append(dict(fold=fold,modality=I.MODALITIES[i],message_norm=float(msgs[i].norm().detach()),
                         raw_norm=float(raw.norm().detach()),layer1_norm=float(l1.norm().detach()),layer2_norm=float(l2.norm().detach()),
                         layer1_raw_ratio=float(l1.norm().detach()/(raw.norm().detach()+1e-12)),
                         layer2_raw_ratio=float(l2.norm().detach()/(raw.norm().detach()+1e-12))))
    g.zero_grad(set_to_none=True); p.zero_grad(set_to_none=True)
    return rows

def intervene(ctx, fold, base, adjs, mode, new_adjs=None, x_override=None):
    c=I.modules(); t=c['torch'];
    if new_adjs is None: new_adjs=adjs
    with t.no_grad(): q=I.original_forward(ctx,new_adjs,x_override)
    met=I.metric_np(q['logp'].detach().cpu().numpy(),ctx['y'],ctx['test'])
    bmet=I.metric_np(base['logp'].detach().cpu().numpy(),ctx['y'],ctx['test'])
    ps=pred_stats(bmet['pred'],met['pred'],ctx['y'],ctx['test'])
    trueidx=np.asarray(ctx['test']); base_lp=base['logp'].detach().cpu().numpy()[trueidx,np.asarray(ctx['y'])[trueidx]]; new_lp=q['logp'].detach().cpu().numpy()[trueidx,np.asarray(ctx['y'])[trueidx]]
    row=dict(fold=fold,mode=mode,acc=met['acc'],hard_auc=met['hard_auc'],spe=met['spe'],sen=met['sen'],
             asd_tp=met['asd_tp'],asd_fn=met['asd_fn'],td_fp=met['td_fp'],td_tn=met['td_tn'],
             true_logp_delta=float(np.mean(new_lp-base_lp)),abs_logp_delta=float(np.mean(np.abs(new_lp-base_lp))),**ps)
    row['message_norm']=float(q['feat'].detach().norm().cpu())
    return row

def residual_probe_space(ctx, adjs, fold, feature_matrix, space):
    """Conditional information probe with separate self/residual scaling."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    c=I.modules(); t=c['torch']; x=np.asarray(feature_matrix,dtype=np.float64); y=ctx['y']; n=len(x); tr=np.asarray(ctx['train']); te=np.asarray(ctx['test']); site=ctx['meta']['site']
    base=np.asarray(adjs)
    def summaries(graphs, sender_x=x):
        out=[]
        for a in graphs:
            w=np.asarray(a).copy(); np.fill_diagonal(w,0); den=w.sum(1,keepdims=True); den[den<1e-12]=1.; out.append((w@sender_x)/den)
        return np.concatenate(out,1)
    true=summaries(base)
    rng=np.random.default_rng(470700+fold); allowed=[np.arange(n) for _ in range(n)]
    global_r=[]; site_r=[]
    for i,a in enumerate(base):
        global_r.append(I.row_random_adjacency(a,allowed,rng))
        site_r.append(I.row_random_adjacency(a,allowed,rng,same_site=True,site=site))
    random_s=summaries(global_r); site_s=summaries(site_r)
    perm=rng.permutation(n)
    if np.any(perm==np.arange(n)):
        # deterministic cyclic derangement
        perm=np.roll(np.arange(n),1)
    perm_s=summaries(base,x[perm])
    # Residualize each neighbor summary using only outer-train receivers.
    xm=x[tr]; mu=xm.mean(0); xc=xm-mu; reg=1.0*np.eye(x.shape[1])
    def residual(s):
        sm=s[tr].mean(0); B=np.linalg.solve(xc.T@xc+reg,xc.T@(s[tr]-sm))
        return s-((x-mu)@B+sm)
    rs={k:residual(v) for k,v in [('TRUE',true),('SITE_RANDOM',site_s),('RANDOM',random_s),('PERMUTED',perm_s)]}
    # Separate standardization prevents a large raw self vector from coupling
    # the neighbor scale through a single LayerNorm.
    sm=xm.mean(0); ss=xm.std(0); ss[ss<1e-8]=1.
    out=[]
    def metric_prob(prob,idx):
        pred=np.asarray(prob).argmax(1); tar=y[idx]; return dict(acc=float(np.mean(pred==tar)),hard_auc=float(roc_auc_score(np.eye(2)[tar],np.eye(2)[pred])),
            spe=float(np.sum((pred==1)&(tar==1))/max(1,np.sum(tar==1))),sen=float(np.sum((pred==0)&(tar==0))/max(1,np.sum(tar==0))))
    conditions=['SELF','TRUE','SITE_RANDOM','RANDOM','PERMUTED']
    # Inner OOF is a patient-level check; it is not used for any architecture choice.
    from sklearn.model_selection import StratifiedKFold
    inner=list(StratifiedKFold(5,shuffle=True,random_state=470+fold).split(tr,y[tr]))
    for cond in conditions:
        rr=np.zeros((n,true.shape[1])) if cond=='SELF' else rs[cond]
        rz=(rr[...]-rr[tr].mean(0))/np.maximum(rr[tr].std(0),1e-8)
        selfz=(x-sm)/ss
        z=np.concatenate([selfz,rz],1)
        # liblinear is a deterministic, fixed-capacity linear probe and is
        # substantially cheaper than a high-dimensional lbfgs path here.
        clf=LogisticRegression(C=1.0,solver='liblinear',max_iter=200,class_weight='balanced',random_state=0)
        clf.fit(z[tr],y[tr]); p=clf.predict_proba(z[te]); m=metric_prob(p,te)
        out.append(dict(fold=fold,space=space,split='outer_test',condition=cond,**m,train_n=len(tr),test_n=len(te),feature_dim=z.shape[1]))
        oo=[]
        for a,b in inner:
            ia,ib=tr[a],tr[b]; cc=LogisticRegression(C=1.0,solver='liblinear',max_iter=200,class_weight='balanced',random_state=0)
            cc.fit(z[ia],y[ia]); oo.append(metric_prob(cc.predict_proba(z[ib]),ib))
        out.append(dict(fold=fold,space=space,split='outer_train_oof',condition=cond,**{k:float(np.mean([q[k] for q in oo])) for k in ['acc','hard_auc','spe','sen']},
                        train_n=len(tr),test_n=len(tr),feature_dim=z.shape[1]))
    return out

def synthetic_control():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rng=np.random.default_rng(470999); n=600; d=8; y=rng.integers(0,2,n); selfx=rng.normal(size=(n,d)); neigh=rng.normal(size=(n,d)); neigh[:,0]+=2*(y-.5)
    def fit(z):
        c=LogisticRegression(C=1.,solver='lbfgs',max_iter=500,class_weight='balanced').fit(z[:400],y[:400]); p=c.predict_proba(z[400:]).argmax(1); yy=y[400:]; return float(np.mean(p==yy))
    return [dict(fold=0,split='synthetic_positive_control',condition='SELF',acc=fit(selfx),hard_auc=0,spe=0,sen=0,train_n=400,test_n=200,feature_dim=d),
            dict(fold=0,split='synthetic_positive_control',condition='TRUE',acc=fit(np.concatenate([selfx,neigh],1)),hard_auc=0,spe=0,sen=0,train_n=400,test_n=200,feature_dim=2*d)]

def process_fold(fold,gpu):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    I.seed_all(0); start=time.time(); ctx=I.load_fold(fold,'cuda:0',cam=False); x,y,folds,meta=I.load_data();
    c=I.modules(); t=c['torch']; tr=ctx['train']; te=ctx['test']; site=meta['site'];
    with t.no_grad(): base=I.original_forward(ctx)
    adjs=[a.detach().clone() for a in base['adjs']]; arrays=[a.detach().cpu().numpy() for a in adjs]
    graph_rows=[]; interventions=[]; grads=[]; probes=[]
    base_m=I.metric_np(base['logp'].detach().cpu().numpy(),y,te)
    interventions.append(dict(fold=fold,mode='B0B_BASELINE',acc=base_m['acc'],hard_auc=base_m['hard_auc'],spe=base_m['spe'],sen=base_m['sen'],asd_tp=base_m['asd_tp'],asd_fn=base_m['asd_fn'],td_fp=base_m['td_fp'],td_tn=base_m['td_tn']))
    for i,a in enumerate(arrays):
        # raw score is captured by the same GraphLearn forward path used for
        # the actual adjacency; no separate audit feature slice is used.
        raw_score=getattr(ctx['gfr'],f'GraphConstruct{i}').score431.detach().cpu().numpy()
        st=row_stats(a,y,site,tr); st.update(dict(fold=fold,modality=I.MODALITIES[i],raw_score_mean=float(raw_score.mean()),raw_score_sd=float(raw_score.std()),raw_score_p90=float(np.quantile(raw_score,.9)),threshold=.9))
        graph_rows.append(st)
        # Message norms, including layer-wise contraction and final fusion.
        g=ctx['gfr']; raw=t.cat(I.split_features(ctx['xt']),1)
        with t.no_grad():
            na=I.norm_adj(adjs[i]); l1=t.relu(getattr(g,f'MessagePassing{i}').gcn1(raw,na)); l2=getattr(g,f'MessagePassing{i}').gcn2(l1,na)
            graph_rows[-1].update(dict(gcn1_norm=float(l1.norm().cpu()),gcn2_norm=float(l2.norm().cpu()),
                                       raw_input_norm=float(raw.norm().cpu()),weighted_message_norm=float(base['msgs'][i].norm().cpu()),
                                       message_to_input=float(base['msgs'][i].norm().cpu()/(raw.norm().cpu()+1e-12)),
                                       final_feat_input_delta=float((base['feat']-raw).norm().cpu()/(raw.norm().cpu()+1e-12))))
    for i in range(4):
        for j in range(i+1,4):
            mi=arrays[i]>0; mj=arrays[j]>0; union=mi|mj
            vi=arrays[i][mi & mj]; vj=arrays[j][mi & mj]
            graph_rows.append(dict(fold=fold,modality='PAIR',pair=f'{I.MODALITIES[i]}-{I.MODALITIES[j]}',
                                   support_jaccard=float((mi&mj).sum()/max(1,union.sum())),
                                   shared_edges=int((mi&mj).sum()),weight_corr=float(np.corrcoef(vi,vj)[0,1]) if len(vi)>2 else np.nan,
                                   message_cosine=float((base['msgs'][i].flatten()@base['msgs'][j].flatten()).detach().cpu()/(base['msgs'][i].norm().detach().cpu()*base['msgs'][j].norm().detach().cpu()+1e-12))))
    # Fixed-model relation and graph controls.  All retain the same row degree
    # and sampled weight scale; site controls use PHENO site only.
    rng=np.random.default_rng(471000+fold); allowed=[np.arange(len(x)) for _ in range(len(x))]
    for mode,aa in [('ALL_MESSAGES_OFF',[np.zeros_like(a) for a in arrays]),
                    ('PHENO_OFF',[np.zeros_like(arrays[0]),arrays[1],arrays[2],arrays[3]]),
                    ('ANAT_OFF',[arrays[0],np.zeros_like(arrays[1]),arrays[2],arrays[3]]),
                    ('FUNC_OFF',[arrays[0],arrays[1],np.zeros_like(arrays[2]),arrays[3]]),
                    ('CORR_OFF',[arrays[0],arrays[1],arrays[2],np.zeros_like(arrays[3])])]:
        interventions.append(intervene(ctx,fold,base,adjs,mode,[t.as_tensor(a,device=ctx['xt'].device).float() for a in aa]))
    for i,a in enumerate(arrays):
        glob=I.row_random_adjacency(a,allowed,rng); sitea=I.row_random_adjacency(a,allowed,rng,same_site=True,site=site)
        for name,aa in [('GLOBAL_DEGREE_RANDOM',glob),('SITE_MATCHED_RANDOM',sitea),('WITHIN_SITE_ONLY',a*(site[:,None]==site[None,:])),('CROSS_SITE_ONLY',a*(site[:,None]!=site[None,:]))]:
            aa=aa.astype(np.float32); alt=list(arrays); alt[i]=aa
            interventions.append(intervene(ctx,fold,base,adjs,f'{name}_{I.MODALITIES[i]}',[t.as_tensor(z,device=ctx['xt'].device).float() for z in alt]))
    # Cohort dependence: same graph and model, but other outer-test inputs are
    # permuted.  This is a fixed-model diagnostic, not a changed training rule.
    xo=ctx['xt'].detach().clone(); test=np.asarray(te); perm=np.roll(test,1); xo[test]=xo[perm]
    # This broad permutation includes each target's own feature and is kept as
    # a stress diagnostic only.  The target-held-out rows below are the cohort
    # dependence test: a target is unchanged while only its other test peers
    # are permuted.  A second row recomputes the transductive graph, whereas
    # the first keeps the learned graph fixed.
    interventions.append(intervene(ctx,fold,base,adjs,'ALL_TEST_ROWS_PERMUTED',None,x_override=xo))
    for ti in [0, len(test)//2, len(test)-1]:
        target=int(test[ti]); peers=np.asarray([z for z in test if int(z)!=target]); shifted=np.roll(peers,1)
        xh=ctx['xt'].detach().clone(); xh[peers]=xh[shifted]
        with t.no_grad(): qh=I.original_forward(ctx,adjs,xh); qr=I.original_forward(ctx,None,xh)
        for mode,qz in [('OTHER_TEST_ONLY_TARGET_HELD_FIXED_GRAPH',qh),('OTHER_TEST_ONLY_TARGET_HELD_RECOMPUTE_GRAPH',qr)]:
            old=base['logp'][target].detach().cpu().numpy(); new=qz['logp'][target].detach().cpu().numpy();
            interventions.append(dict(fold=fold,mode=mode,target=target,true_logp_delta=float(new[y[target]]-old[y[target]]),abs_logp_delta=float(np.abs(new-old).mean()),
                                      flip=int(new.argmax()!=old.argmax()),message_norm=float(qz['feat'][target].detach().norm().cpu())))
    # CAM-only identity check is performed through the separately frozen CAM
    # checkpoint; it does not construct GFR.
    camctx=I.load_fold(fold,'cuda:0',cam=True)
    with t.no_grad(): camout=I.cam_forward(camctx)
    cm=I.metric_np(camout['logp'].detach().cpu().numpy(),y,te)
    interventions.append(dict(fold=fold,mode='CAM_ONLY_RECHECK',acc=cm['acc'],hard_auc=cm['hard_auc'],spe=cm['spe'],sen=cm['sen'],asd_tp=cm['asd_tp'],asd_fn=cm['asd_fn'],td_fp=cm['td_fp'],td_tn=cm['td_tn']))
    # Fixed representation separability summaries on train and test.
    with t.no_grad():
        raw=t.cat(I.split_features(ctx['xt']),1); gfeat=base['feat']; hidden=base['hidden']
    for name,z in [('RAW_INPUT',raw),('GFR_OUTPUT',gfeat),('CAM_HIDDEN',hidden)]:
        zz=z.detach().cpu().numpy();
        for split,idx in [('train',tr),('test',te)]:
            ii=np.asarray(idx); a0=zz[ii][y[ii]==0]; a1=zz[ii][y[ii]==1]
            if len(a0) and len(a1):
                c0=a0.mean(0); c1=a1.mean(0); within=(np.mean(np.sum((a0-c0)**2,1))+np.mean(np.sum((a1-c1)**2,1)))/2; between=float(np.sum((c0-c1)**2))
                graph_rows.append(dict(fold=fold,modality='REPRESENTATION',representation=name,split=split,between_distance=float(between),within_distance=float(within),separation_ratio=float(between/(within+1e-12)),effective_rank=float(np.linalg.matrix_rank(zz[ii]-zz[ii].mean(0)))))
    grads=gradient_row(ctx,fold)
    # The primary probe uses the original 320-dimensional features.  The
    # same residualized, separately standardized design is repeated in the
    # frozen B0-B CAM hidden (36-dimensional) space as an auxiliary check.
    probes=residual_probe_space(ctx,arrays,fold,ctx['x'],'RAW320')
    probes.extend(residual_probe_space(ctx,arrays,fold,camout['hidden'].detach().cpu().numpy(),'CAM36'))
    if fold==1:
        probes.extend(synthetic_control())
    out=dict(fold=fold,gpu=gpu,elapsed_seconds=time.time()-start,graph_statistics=graph_rows,interventions=interventions,gradient_diagnostics=grads,conditional_probe=probes)
    jdump(OUT/f'fold{fold:02}.json',out)
    print(json.dumps({'fold':fold,'gpu':gpu,'elapsed':out['elapsed_seconds'],'baseline_acc':base_m['acc'],'cam_acc':cm['acc']},ensure_ascii=False),flush=True)

def merge():
    rows={'graph_statistics':[],'interventions':[],'gradient_diagnostics':[],'conditional_probe':[]}
    for p in sorted(OUT.glob('fold*.json')):
        d=json.loads(p.read_text())
        for k in rows: rows[k].extend(d.get(k,[]))
    RES.mkdir(parents=True,exist_ok=True)
    for k,v in rows.items():
        if not v: continue
        fields=[]
        for r in v:
            for key in r:
                if key not in fields: fields.append(key)
        with (RES/({'graph_statistics':'graph_statistics.csv','interventions':'interventions.csv','gradient_diagnostics':'gradient_diagnostics.csv','conditional_probe':'conditional_probe.csv'}[k])).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader();
            for r in v: w.writerow({q:_json(z) if isinstance(z,(np.generic,np.ndarray)) else z for q,z in r.items()})
    # a compact per-fold status manifest
    done=[p for p in OUT.glob('fold*.json') if p.stem[4:].isdigit()]
    jdump(RT/'stage_status.json',dict(folds=sorted(int(p.stem[4:]) for p in done),count=len(done),merged=True))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--folds',default='1-10'); ap.add_argument('--gpu',default='0'); ap.add_argument('--merge',action='store_true'); args=ap.parse_args()
    if args.merge: merge(); return
    if '-' in args.folds:
        a,b=map(int,args.folds.split('-')); fs=range(a,b+1)
    else: fs=[int(z) for z in args.folds.split(',')]
    for f in fs:
        try: process_fold(f,int(args.gpu))
        except Exception as e:
            import traceback; traceback.print_exc(); jdump(OUT/f'fold{f:02}.error.json',dict(fold=f,error=repr(e),traceback=traceback.format_exc())); raise

if __name__=='__main__': main()
