"""Fold-local, label-safe disease-validity estimates for D3-SAG.

The estimators are deliberately frozen before model training.  They use only
outer-train observations and inner five-fold out-of-fold probabilities.  The
result is a reproducible routing prior, not a claim of causal PID recovery.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss

EPS = 1e-8

def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def _safe_prob(p):
    return np.clip(np.asarray(p, float), 1e-6, 1-1e-6)

def _fit_predict_oof(X, y, inner, seed):
    out = np.zeros(len(y), float)
    for k, (a,b) in enumerate(inner):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=seed+k))
        clf.fit(X[a], y[a]); out[b] = clf.predict_proba(X[b])[:,1]
    return out

def _site_residual(X, site, tr):
    """Remove a training-fitted site main effect from every row."""
    Z = np.column_stack([np.ones(len(site)), site.astype(float)])
    a = np.linalg.pinv(Z[np.asarray(tr)])
    beta = a @ X[np.asarray(tr)]
    return X - Z @ beta

def linear_cka(X, Y):
    X = np.asarray(X,float); Y=np.asarray(Y,float)
    X = X - X.mean(0,keepdims=True); Y = Y-Y.mean(0,keepdims=True)
    xx=X.T@X; yy=Y.T@Y; xy=X.T@Y
    den=np.sqrt((xx*xx).sum()*(yy*yy).sum())
    return float((xy*xy).sum()/max(den,EPS))

def _bootstrap_lcb(diff, seed, n=200):
    rng=np.random.default_rng(seed); diff=np.asarray(diff,float); N=len(diff)
    if N==0: return 0.0
    vals=np.empty(n,float)
    for i in range(n): vals[i]=diff[rng.integers(0,N,N)].mean()
    return float(np.quantile(vals,0.05))

def _u(lcb):
    z=max(0.0,float(lcb)); return float(np.clip(z/(0.02+z),0,1))

def _site_perm(site, rng):
    p=np.arange(len(site),dtype=np.int64)
    for s in np.unique(site):
        ids=np.where(site==s)[0]
        if len(ids)>1:
            sh=ids.copy(); rng.shuffle(sh)
            if np.all(sh==ids): sh=np.roll(ids,1)
            p[ids]=sh
    return p

def compute_evidence(X, y, tr, pheno_idx, site_idx, q_raw, fold, cache_path=None):
    """Compute and optionally persist all fold-local routing statistics."""
    tr=np.asarray(tr,dtype=np.int64); y=np.asarray(y,dtype=np.int64)
    site=np.argmax(X[:,site_idx],axis=1).astype(np.int64)
    # inner folds and only outer-train data are used by all teachers
    inner=list(StratifiedKFold(5,shuffle=True,random_state=514100+fold).split(tr,y[tr]))
    yy=y[tr]
    site_oh=X[:,site_idx]
    F=X[:,64:320]; P=X[:,pheno_idx]
    # raw views are standardized inside each teacher by the pipeline
    ftr=F[tr]; ptr=P[tr]; str_=site_oh[tr]
    # fixed low-rank interaction fitted only on outer train
    pca_f=PCA(n_components=min(8,ftr.shape[1]),random_state=514200+fold).fit(StandardScaler().fit_transform(ftr))
    pca_p=PCA(n_components=min(8,ptr.shape[1]),random_state=514300+fold).fit(StandardScaler().fit_transform(ptr))
    Sf=pca_f.transform(StandardScaler().fit(ftr).transform(ftr)); Sp=pca_p.transform(StandardScaler().fit(ptr).transform(ptr))
    # Refit scalers once for interaction on the complete outer train; all are frozen.
    sfsc=StandardScaler().fit(ftr); spsc=StandardScaler().fit(ptr)
    Sf=pca_f.transform(sfsc.transform(ftr)); Sp=pca_p.transform(spsc.transform(ptr))
    inter=(Sf[:,:,None]*Sp[:,None,:]).reshape(len(tr),-1)
    X0=str_; XF=np.concatenate([str_,ftr],1); XP=np.concatenate([str_,ptr],1); XJ=np.concatenate([str_,ftr,ptr,inter],1)
    preds={k:_fit_predict_oof(v,yy,inner,514400+fold*10+i) for i,(k,v) in enumerate((('site',X0),('fc',XF),('pheno',XP),('joint',XJ)))}
    losses={k:np.where(yy==1,-np.log(_safe_prob(p)), -np.log(1-_safe_prob(p))) for k,p in preds.items()}
    risks={k:float(v.mean()) for k,v in losses.items()}
    dF=losses['site']-losses['fc']; dP=losses['site']-losses['pheno']; dC=losses['fc']-losses['joint']
    lF=_bootstrap_lcb(dF,514500+fold); lP=_bootstrap_lcb(dP,514600+fold); lC=_bootstrap_lcb(dC,514700+fold)
    uF,uP,uC=map(_u,(lF,lP,lC))
    # site residualization and a fixed within-site null distribution
    f_res=_site_residual(ftr,site[tr],np.arange(len(tr)))
    p_res=_site_residual(ptr,site[tr],np.arange(len(tr)))
    cka_obs=linear_cka(f_res,p_res); null=[]; rng=np.random.default_rng(514800+fold)
    for _ in range(39):
        perm=_site_perm(site[tr],rng); null.append(linear_cka(f_res,p_res[perm]))
    q95=float(np.quantile(null,0.95)); rgeom=float(np.clip((cka_obs-q95)/(1-q95+EPS),0,1)); rS=float(min(uF,uP)*rgeom); rC=float(uC)
    # label-free robust reliability from ANAT/FUNC quality proxies
    q=np.asarray(q_raw,float); med=np.median(q[tr],0); mad=np.median(np.abs(q[tr]-med),0)*1.4826+1e-3
    anom=np.mean(np.abs((q-med)/mad),1); qrel=np.exp(-np.clip(anom,0,8)); qrel=np.clip(qrel,0,1)
    out={'fold':int(fold),'train':tr,'site':site,'qrel':qrel.astype(np.float32),'risks':risks,
         'dF':float(risks['site']-risks['fc']),'dP':float(risks['site']-risks['pheno']),'dC':float(risks['fc']-risks['joint']),
         'lcbF':lF,'lcbP':lP,'lcbC':lC,'uF':uF,'uP':uP,'uC':uC,'cka_obs':cka_obs,'cka_q95':q95,
         'rgeom':rgeom,'rS':rS,'rC':rC,'null_cka':null,'oof':{k:v.tolist() for k,v in preds.items()},
         'site_idx':np.asarray(site_idx,dtype=np.int64),'pheno_idx':np.asarray(pheno_idx,dtype=np.int64),
         'sha_train':_sha(tr),'sha_q':_sha(qrel)}
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache_path, qrel=qrel, train=tr, site=site, rS=np.array([rS]), rC=np.array([rC]),
                            dF=np.array([out['dF']]), dP=np.array([out['dP']]), dC=np.array([out['dC']]),
                            lcbF=np.array([lF]),lcbP=np.array([lP]),lcbC=np.array([lC]),cka_obs=np.array([cka_obs]),cka_q95=np.array([q95]),rgeom=np.array([rgeom]))
        small={}
        for k,v in out.items():
            if k in ('train','site','qrel','oof','null_cka'): continue
            if isinstance(v,np.ndarray):
                if v.size==1: small[k]=float(v.reshape(-1)[0])
                else: small[k]=v.tolist()
            elif isinstance(v,(np.floating,np.integer)): small[k]=v.item()
            else: small[k]=v
        Path(str(cache_path)+'.json').write_text(json.dumps(small,ensure_ascii=False,indent=2,default=float))
    return out
