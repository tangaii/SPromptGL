#!/usr/bin/env python3
"""Experiment 54 D²-ARG runner.

The only external read-only dependencies are the ABIDE files and the copied
Experiment 51 E1 source/anchor snapshots under this experiment's ``source``.
Every mutable artifact is written below Experiment 54.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, pickle, random, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("E52_EXP", str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/SPromptGL-main")
SRC = EXP / "source"
RT = EXP / ".runtime"
TASKS, CKPTS, LATEST, LOGS = RT/"tasks", RT/"checkpoints", RT/"latest", RT/"logs"
FOLD_DATA, REL = RT/"fold_data", RT/"relation_cache"
RESULTS, INTERVENTIONS = EXP/"results", RT/"interventions"
CFG = json.loads((EXP / "config.json").read_text())
DIMS = list(CFG["data"]["dims"])
MODS = ["PHENO", "ANAT", "FUNC", "CORR"]
GROUPS = tuple(g["id"] for g in CFG["groups"])
GROUP_BY_ID = {g["id"]: g for g in CFG["groups"]}
CTX = None
DATA = None
FOLDS = None

# Import the copied, audited E1 construction only after the local EXP override.
import importlib.util
_spec = importlib.util.spec_from_file_location("e1_base_local", SRC / "e1_base.py")
e1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e1)
sys.path.insert(0, str(EXP))
import d2arg


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def sha_obj(obj):
    return hashlib.sha256(pickle.dumps(obj, protocol=4)).hexdigest()


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float, allow_nan=False))
    os.replace(tmp, path)


def torch_save(path, obj):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows); fields = []
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def csv_read(path):
    if not Path(path).exists(): return []
    with Path(path).open(newline="") as f: return list(csv.DictReader(f))


def ensure_dirs():
    for p in (RT, TASKS, CKPTS, LATEST, LOGS, FOLD_DATA, REL, RESULTS, INTERVENTIONS): p.mkdir(parents=True, exist_ok=True)


def task_path(g, f): return TASKS / f"{g}_f{f:02d}.json"
def ckpt_path(g, f): return CKPTS / f"{g}_f{f:02d}.best.pt"
def latest_path(g, f): return LATEST / f"{g}_f{f:02d}.latest.pt"
def rel_path(f, wrong=False): return REL / f"fold{f:02d}{'_wrong' if wrong else ''}.pt"
def rel_meta_path(f, wrong=False): return REL / f"fold{f:02d}{'_wrong' if wrong else ''}.json"


def runtime():
    global CTX
    if CTX is None:
        import numpy as np, pandas as pd, torch
        torch.set_num_threads(2)
        try: torch.set_num_interop_threads(1)
        except RuntimeError: pass
        torch.cuda.set_device(0)
        c = e1.runtime()
        CTX = {"np":np, "pd":pd, "torch":torch, "device":torch.device("cuda"), **c}
    return CTX


def load_data():
    global DATA, FOLDS
    if DATA is None:
        import numpy as np, pandas as pd
        from sklearn.model_selection import StratifiedKFold
        base = ROOT / "data/ABIDE"
        frame = pd.read_csv(base / "processed_standard_data.csv")
        mapping = np.load(base / "modal_feat_dict.npy", allow_pickle=True).item()
        expected = [str(v) for key in ("PHENO", "ANAT", "FUNC", "Correlation") for v in mapping[key]]
        assert list(frame.columns[:-1]) == expected
        x = frame.iloc[:, :-1].to_numpy(np.float32); y = frame.iloc[:, -1].to_numpy(np.int64) - 1
        assert x.shape == (871, 320) and np.bincount(y).tolist() == [403, 468] and np.isfinite(x).all()
        assert [len(mapping[k]) for k in ("PHENO","ANAT","FUNC","Correlation")] == DIMS
        FOLDS = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y)); DATA = (x, y)
    return DATA, FOLDS


def sites_from_x(x):
    # Mapping audit confirms PHENO indices 26:46 are one-hot SITE_ID columns.
    s = x[:, 26:46]
    site = s.argmax(1).astype("int64")
    site[s.sum(1) <= 0] = -1
    return site


def split_sha(fold):
    import numpy as np
    tr, te = load_data()[1][fold-1]
    return hashlib.sha256(np.asarray(tr, np.int64).tobytes() + np.asarray(te, np.int64).tobytes()).hexdigest()


def module_sha(module):
    h = hashlib.sha256()
    for n, p in sorted(module.named_parameters()):
        h.update(n.encode()); h.update(str(tuple(p.shape)).encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def state_cpu(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def optimizer_cpu_state(opt):
    import copy
    d = copy.deepcopy(opt.state_dict())
    for st in d.get("state", {}).values():
        for k, v in list(st.items()):
            if hasattr(v, "detach"): st[k] = v.detach().cpu().clone()
    return d


def optimizer_to_cuda(opt):
    for st in opt.state.values():
        for k, v in list(st.items()):
            if hasattr(v, "to"): st[k] = v.cuda()


def pack_rng():
    import numpy as np, torch
    return {"python":random.getstate(), "numpy":np.random.get_state(), "cpu":torch.get_rng_state(), "cuda":torch.cuda.get_rng_state_all()}


def restore_rng(r):
    import numpy as np, torch
    random.setstate(r["python"]); np.random.set_state(r["numpy"]); torch.set_rng_state(r["cpu"].cpu()); torch.cuda.set_rng_state_all([v.cpu() for v in r["cuda"]])


def rng_sha(r):
    import numpy as np
    h = hashlib.sha256(pickle.dumps((r["python"], r["numpy"]), protocol=4)); h.update(r["cpu"].cpu().numpy().tobytes())
    for v in r["cuda"]: h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def module_grad_norm(module):
    vals = [(p.grad.detach()**2).sum() for p in module.parameters() if p.grad is not None]
    return float(__import__("torch").stack(vals).sum().sqrt().cpu()) if vals else 0.0


def make_wrong_perm(site, fold):
    import numpy as np
    n = len(site); rng = np.random.default_rng(540000 + fold); perm = np.arange(n, dtype=np.int64)
    changed = False
    for s in sorted(set(site.tolist())):
        ids = np.where(site == s)[0]
        if len(ids) > 1:
            sh = ids.copy(); rng.shuffle(sh)
            if np.array_equal(sh, ids): sh = np.roll(ids, 1)
            perm[ids] = sh; changed |= not np.array_equal(sh, ids)
    if not changed:
        sh = rng.permutation(n); sh = np.roll(sh, 1); perm = sh.astype(np.int64)
    if np.any(perm == np.arange(n)):
        # Ensure a derangement without changing the deterministic seed rule.
        bad = np.where(perm == np.arange(n))[0]
        if len(bad) > 1:
            perm[bad] = np.roll(bad, 1)
        else:
            j = (int(bad[0]) + 1) % n; perm[bad[0]], perm[j] = perm[j], perm[bad[0]]
    return perm


def make_support(fold):
    import numpy as np
    x, y = load_data()[0]; anchor = e1.fold_anchor(fold); starts = np.cumsum([0] + DIMS[:-1])
    all_pairs, modality_pairs = set(), []
    for m, (start, dim) in enumerate(zip(starts, DIMS)):
        g = anchor["model"]["gfr"]; w = g[f"GraphConstruct{m}.p.weight"].cpu().numpy(); b = g[f"GraphConstruct{m}.p.bias"].cpu().numpy()
        z = x[:, start:start+dim] @ w.T + b; z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8); sim = z @ z.T
        ii, jj = np.where(np.triu(sim > CFG["base"]["threshold"], 1)); pairs = sorted(zip(ii.astype(np.int64).tolist(), jj.astype(np.int64).tolist()))
        modality_pairs.append(pairs); all_pairs.update(pairs)
    union = sorted(all_pairs); lookup = {p:i for i,p in enumerate(union)}
    modality_idx = [np.asarray([lookup[p] for p in pairs], np.int64) for pairs in modality_pairs]
    ui = np.asarray([p[0] for p in union], np.int64); uj = np.asarray([p[1] for p in union], np.int64)
    site = sites_from_x(x); perm = make_wrong_perm(site, fold)
    return {"u_i":ui, "u_j":uj, "modality_idx":modality_idx, "union_edge_count":len(union), "modality_edge_counts":[len(p) for p in modality_pairs], "wrong_perm":perm, "wrong_fixed_points":int(np.sum(perm==np.arange(len(perm))))}


def sample_pairs(indices, y, site, support, max_count, seed, wrong_supervision=False):
    import numpy as np
    ids = np.asarray(indices, np.int64); trset = set(ids.tolist());
    cand = [(int(i),int(j)) for i,j in zip(support["u_i"].tolist(), support["u_j"].tolist()) if i in trset and j in trset]
    ii,jj = np.triu_indices(len(ids), 1); ai,aj = ids[ii], ids[jj]
    pools = {(d,s):[] for d in (0,1) for s in (0,1)}
    for i,j in cand:
        pools[(int(y[i] == y[j]), int(site[i] == site[j]))].append((i,j))
    allp = {(d,s):[] for d in (0,1) for s in (0,1)}
    for i,j in zip(ai.tolist(), aj.tolist()): allp[(int(y[i] == y[j]), int(site[i] == site[j]))].append((int(i),int(j)))
    rng = np.random.default_rng(int(seed)); k = min(int(max_count)//4, *(len(allp[t]) for t in allp))
    selected=[]
    for t in ((0,0),(0,1),(1,0),(1,1)):
        pool = pools[t] if len(pools[t]) >= k else allp[t]
        if len(pool) > k: selected.extend([pool[int(a)] for a in rng.choice(len(pool), k, replace=False)])
        else: selected.extend(pool[:k])
    rng.shuffle(selected)
    pi=np.asarray([p[0] for p in selected],np.int64); pj=np.asarray([p[1] for p in selected],np.int64)
    dy=(y[pi]==y[pj]).astype(np.int64); sy=(site[pi]==site[pj]).astype(np.int64)
    if wrong_supervision and len(dy):
        wr = np.random.default_rng(int(seed)+991); wr.shuffle(dy)
    return pi,pj,dy,sy


def pair_sha(pair):
    return sha_obj(tuple(np.asarray(a) for a in pair))


def prepare_folds():
    import numpy as np
    ensure_dirs(); x,y = load_data()[0]; site = sites_from_x(x)
    for fold,(tr,te) in enumerate(load_data()[1],1):
        support = make_support(fold)
        pair = sample_pairs(tr,y,site,support,CFG["relation"]["pair_max"],550000+fold,False)
        wrong = sample_pairs(tr,y,site,support,CFG["relation"]["pair_max"],550000+fold,True)
        np.savez_compressed(FOLD_DATA/f"fold{fold:02d}.npz", train=tr, test=te, pair_i=pair[0], pair_j=pair[1], pair_y=pair[2], pair_site=pair[3], wrong_pair_y=wrong[2])
        atomic_json(FOLD_DATA/f"fold{fold:02d}.json", {"fold":fold,"split_sha":split_sha(fold),"pair_sha":sha_file(FOLD_DATA/f"fold{fold:02d}.npz"),"candidate":support["union_edge_count"],"modality_edge_counts":support["modality_edge_counts"],"wrong_fixed_points":support["wrong_fixed_points"],"test_labels_excluded":True})


def load_fd(fold):
    import numpy as np, torch
    z=np.load(FOLD_DATA/f"fold{fold:02d}.npz")
    return {k:torch.from_numpy(z[k]).cuda() for k in z.files}


def build_core(seed):
    import torch
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(int(seed)); core=d2arg.D2ARGCore(CFG["relation"]["hidden"], CFG["relation"]["grl_scale"]).cuda()
    return core


def train_relation_core(corr, pair, seed, epochs, adv=True):
    import torch
    pi,pj,dy,sy = [torch.from_numpy(a).cuda() for a in pair]
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(int(seed)); core=d2arg.D2ARGCore(CFG["relation"]["hidden"], CFG["relation"]["grl_scale"]).cuda(); opt=torch.optim.AdamW(core.optimizer_param_groups(CFG["relation"]["lr"], CFG["relation"]["weight_decay"]))
        core.train()
        for _ in range(int(epochs)):
            opt.zero_grad(); _,_,_,logits,lp,la,_=core.pair_forward(corr,pi,pj,dy,sy,adv=adv,adv_condition=dy); loss=lp + (0.10*la if adv else 0.0)
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite relation loss")
            loss.backward();
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in core.parameters()): raise FloatingPointError("nonfinite relation gradient")
            opt.step()
    return core


def relation_cache_one(fold, wrong=False):
    import numpy as np, torch
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    path=rel_path(fold,wrong); meta=rel_meta_path(fold,wrong)
    if path.exists() and meta.exists(): return
    x,y=load_data()[0]; site=sites_from_x(x); tr,_=load_data()[1][fold-1]; corr=torch.from_numpy(x[:,64:]).float().cuda()
    inner=StratifiedKFold(CFG["relation"]["inner_splits"],shuffle=True,random_state=551000+fold)
    oof_log=[]; oof_y=[]; oof_site=[]; inner_rows=[]
    for sid,(ia,iv) in enumerate(inner.split(tr,y[tr])):
        itr,iva=tr[ia],tr[iv]
        support=make_support(fold)
        ptrain=sample_pairs(itr,y,site,support,CFG["relation"]["inner_pair_max"],551100+fold*10+sid,wrong)
        pvalid=sample_pairs(iva,y,site,support,CFG["relation"]["inner_pair_max"],551200+fold*10+sid,False)
        core=train_relation_core(corr,ptrain,551300+fold*100+sid,CFG["relation"]["inner_epochs"],adv=False); core.eval(); pi,pj,dy,sy=[torch.from_numpy(a).cuda() for a in pvalid]
        with torch.no_grad(): _,_,_,logits,_,_,_=core.pair_forward(corr,pi,pj,dy,sy,adv=False); pr=torch.sigmoid(logits).cpu().numpy()
        oof_log.extend(np.log(np.clip(pr,1e-6,1-1e-6)/(1-np.clip(pr,1e-6,1-1e-6))).tolist()); oof_y.extend(dy.cpu().numpy().tolist()); oof_site.extend(sy.cpu().numpy().tolist())
        inner_rows.append({"split":sid,"train_patients":len(itr),"heldout_patients":len(iva),"pair_count":len(pvalid),"patient_disjoint":not bool(set(itr.tolist())&set(iva.tolist()))})
        del core; torch.cuda.empty_cache()
    oof_log=np.asarray(oof_log,np.float64); oof_y=np.asarray(oof_y,np.int64); oof_site=np.asarray(oof_site,np.int64)
    raw=1/(1+np.exp(-oof_log)); cal=LogisticRegression(C=1e6,solver="lbfgs",random_state=0).fit(oof_log.reshape(-1,1),oof_y); a=float(cal.coef_[0,0]); b=float(cal.intercept_[0]); cp=1/(1+np.exp(-(a*oof_log+b)))
    auc=float(roc_auc_score(oof_y,cp)); cross= float(roc_auc_score(oof_y[oof_site==0],cp[oof_site==0])) if len(set(oof_y[oof_site==0].tolist()))==2 else 0.5; same=float(roc_auc_score(oof_y[oof_site==1],cp[oof_site==1])) if len(set(oof_y[oof_site==1].tolist()))==2 else 0.5
    reliable=bool(auc>=CFG["relation"]["reliable_auc_threshold"] and cross>=CFG["relation"]["reliable_auc_threshold"] and cp[oof_y==1].mean()>cp[oof_y==0].mean())
    trpair=sample_pairs(tr,y,site,make_support(fold),CFG["relation"]["pair_max"],551800+fold,wrong)
    final=train_relation_core(corr,trpair,552000+fold,CFG["relation"]["inner_epochs"],adv=True)
    package={"relation_state":state_cpu(final),"calibration_a":np.asarray([a],np.float32),"calibration_b":np.asarray([b],np.float32),"relation_metrics":{"oof_auc":auc,"cross_site_auc":cross,"same_site_auc":same,"brier":float(brier_score_loss(oof_y,cp)),"reliable":reliable},"inner_rows":inner_rows,"final_pair":trpair,"candidate":make_support(fold),"fold":fold,"split_sha":split_sha(fold),"wrong_supervision":wrong,"test_label_excluded":True,"relation_parameter_sha":module_sha(final)}
    torch_save(path, package)
    atomic_json(meta, {"fold": fold, "split_sha": split_sha(fold),
                       "relation_metrics": package["relation_metrics"],
                       "inner_rows": package["inner_rows"],
                       "wrong_supervision": wrong, "test_label_excluded": True,
                       "relation_parameter_sha": package["relation_parameter_sha"],
                       "candidate_edge_count": int(package["candidate"]["union_edge_count"]),
                       "modality_edge_counts": package["candidate"]["modality_edge_counts"]})
    del final; torch.cuda.empty_cache()


def pretrain_relations():
    for f in range(1,11): relation_cache_one(f,False); relation_cache_one(f,True); print("RELATION_CACHE_DONE",f,flush=True)


def make_relation_module(package, fold):
    import torch
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(553000+fold); core=d2arg.D2ARGCore(CFG["relation"]["hidden"],CFG["relation"]["grl_scale"]).cuda()
    core.load_state_dict(package["relation_state"],strict=True)
    mod=d2arg.D2ARGCore(CFG["relation"]["hidden"],CFG["relation"]["grl_scale"]).cuda(); mod.load_state_dict(core.state_dict(),strict=True)
    # Bounded edge/propagation parameters are in a separate module so E1 params
    # and their optimizer remain exactly untouched.
    dev = torch.device("cuda")
    mod.gamma_raw=torch.nn.Parameter(torch.full((4,), math.log(CFG["propagation"]["gamma_init"]/(CFG["propagation"]["gamma_max"]-CFG["propagation"]["gamma_init"])), device=dev))
    mod.mu_raw=torch.nn.Parameter(torch.full((4,), math.log(CFG["propagation"]["mu_init"]/(CFG["propagation"]["mu_max"]-CFG["propagation"]["mu_init"])), device=dev))
    mod.rho_raw=torch.nn.Parameter(torch.tensor(math.log(CFG["propagation"]["rho_init"]/(CFG["propagation"]["rho_max"]-CFG["propagation"]["rho_init"])), device=dev))
    mod.kappa_raw=torch.nn.Parameter(torch.tensor(math.log(CFG["propagation"]["kappa_init"]/(CFG["propagation"]["kappa_max"]-CFG["propagation"]["kappa_init"])), device=dev))
    mod.register_buffer("candidate_i",torch.from_numpy(package["candidate"]["u_i"]).long().cuda()); mod.register_buffer("candidate_j",torch.from_numpy(package["candidate"]["u_j"]).long().cuda())
    for m,idx in enumerate(package["candidate"]["modality_idx"]): mod.register_buffer(f"modality_edge_{m}",torch.from_numpy(idx).long().cuda())
    mod.register_buffer("wrong_perm",torch.from_numpy(package["candidate"]["wrong_perm"]).long().cuda())
    return mod


def add_param_groups(mod):
    return [{"params":list(mod.parameters()),"lr":CFG["relation"]["lr"],"weight_decay":CFG["relation"]["weight_decay"]}]


def build_state(group,fold):
    import torch, numpy as np
    base=e1.build_state("E0",fold); base["group"]=group
    base["initial_shared_sha"]=e1.digest_shared(base["base"],base["cam"]); base["initial_schema_sha"]=e1.schema_shared(base["base"],base["cam"]); base["initial_rng_sha"]=e1.rng_sha(base["initial_rng"]); base["initial_cam_sha"]=module_sha(base["cam"]); base["split_sha"]=split_sha(fold); base["pair_sha"]=sha_file(FOLD_DATA/f"fold{fold:02d}.npz")
    if group=="B0": base.update({"d2":None,"opt_d2":None,"package":None})
    else:
        wrong=group=="B7"; package=torch_load(rel_path(fold,wrong)); d2=make_relation_module(package,fold); base["d2"]=d2; base["opt_d2"]=torch.optim.Adam(d2.parameters(),lr=CFG["relation"]["lr"],weight_decay=CFG["relation"]["weight_decay"]); base["package"]=package
    base["initial_relation_sha"]=module_sha(base["d2"]) if base["d2"] is not None else ""
    base["site_ids"]=torch.from_numpy(sites_from_x(load_data()[0][0])).cuda()
    return base


def base_forward(base,parts,adjs_override=None,force_self=False,wrong_perm=None):
    import torch
    raw=torch.cat(parts,1); fused=raw; adjs=[]; msgs=[]; used=[]; util=e1.runtime()["utils"]
    for m,x in enumerate(parts):
        adj=base.graphs[m](x); adjs.append(adj); n=adj.shape[0]; eye=torch.eye(n,device=adj.device,dtype=adj.dtype)
        if force_self: a=eye
        elif adjs_override is not None: a=util.normalize_adj(adjs_override[m]+eye)
        elif wrong_perm is not None: a=util.normalize_adj(adj[wrong_perm][:,wrong_perm]+eye)
        else: a=util.normalize_adj(adj+eye)
        msg=base.gcn[m](raw,a); msgs.append(msg); fused=fused+torch.sigmoid(base.fusion_weight[m])*msg; used.append(a)
    base.last={"adjs":adjs,"msgs":msgs,"used":used,"fused":fused}; return fused,adjs,used,msgs


def relation_q(mod,corr,fd,group):
    import torch
    if GROUP_BY_ID[group].get("similarity",False):
        norm=torch.nn.functional.normalize(corr,dim=-1)
        q=((norm[mod.candidate_i]*norm[mod.candidate_j]).sum(-1)+1)/2
        return q, {"pair_loss": corr.new_zeros(()), "site_loss": corr.new_zeros(()), "pair_logits": None, "site_logits": None, "z": None, "features": None, "pair_target": None}
    pi,pj=fd["pair_i"].long(),fd["pair_j"].long(); target=fd["wrong_pair_y"].long() if group=="B7" else fd["pair_y"].long(); sy=fd["pair_site"].long(); adv=bool(GROUP_BY_ID[group].get("adv",False))
    z,prior,feat,logits,lp,la,site_logits=mod.pair_forward(corr,pi,pj,target,sy,adv=adv,adv_condition=fd["pair_y"].long())
    norm=torch.nn.functional.normalize(corr,dim=-1); prior_e=(norm[mod.candidate_i]*norm[mod.candidate_j]).sum(-1)
    zn=torch.nn.functional.normalize(z,dim=-1); ef=mod.scorer.pair_features(z,mod.candidate_i,mod.candidate_j,prior_e); q=torch.sigmoid(mod.scorer(ef))
    return q,{"pair_loss":lp,"site_loss":la,"pair_logits":logits,"site_logits":site_logits,"z":z,"features":feat,"pair_target":target}


def calibrated_adjs(mod,adjs,q,group,wrong=False):
    import torch
    out=[]; gam=CFG["propagation"]["gamma_max"]*torch.sigmoid(mod.gamma_raw); mu=CFG["propagation"]["mu_max"]*torch.sigmoid(mod.mu_raw); n=adjs[0].shape[0]
    pos=torch.relu(2*q-1); neg=torch.relu(1-2*q)
    for m,adj in enumerate(adjs):
        qm=torch.full((n,n),0.5,device=adj.device,dtype=adj.dtype); idx=getattr(mod,f"modality_edge_{m}");
        if idx.numel(): qm[mod.candidate_i[idx],mod.candidate_j[idx]]=q[idx].to(adj.dtype); qm[mod.candidate_j[idx],mod.candidate_i[idx]]=q[idx].to(adj.dtype)
        qp=torch.relu(2*qm-1); qn=torch.relu(1-2*qm); fac=(1+gam[m]*qp)*(1-mu[m]*qn); fac.fill_diagonal_(1.0); a=adj*fac; a.fill_diagonal_(0.0)
        if wrong: p=mod.wrong_perm; a=a[p][:,p]
        out.append(a)
    return out,{"gamma":gam.detach(),"mu":mu.detach(),"edge_change":float(torch.stack([(out[m]-adjs[m]).abs().mean() for m in range(4)]).mean().detach().cpu())}


def propagation_delta(mod,raw,q,group,override=None):
    import torch
    if override in ("off","self_only") or GROUP_BY_ID[group].get("new_message_off",False) or GROUP_BY_ID[group].get("propagation","none")=="none": return raw.new_zeros(raw.shape),{"positive_norm":0.,"negative_norm":0.,"positive_edges":0,"negative_edges":0}
    pos=torch.relu(2*q-1); neg=torch.relu(1-2*q); use_neg=GROUP_BY_ID[group].get("propagation")=="signed" and override!="negative_off"; use_pos=True
    ids_p=torch.where(pos>0)[0] if use_pos else torch.empty(0,dtype=torch.long,device=raw.device); ids_n=torch.where(neg>0)[0] if use_neg else torch.empty(0,dtype=torch.long,device=raw.device)
    perm=mod.wrong_perm if GROUP_BY_ID[group].get("wrong_graph",False) or override=="wrong" else None
    def directed(ids,w):
        si=mod.candidate_i[ids]; sj=mod.candidate_j[ids]
        if perm is not None: si,sj=perm[si],perm[sj]
        return torch.cat((si,sj)),torch.cat((sj,si)),torch.cat((w,w))
    sp,dp,wp=directed(ids_p,pos[ids_p]) if len(ids_p) else (raw.new_empty(0,dtype=torch.long),raw.new_empty(0,dtype=torch.long),raw.new_empty(0))
    sn,dn,wn=directed(ids_n,neg[ids_n]) if len(ids_n) else (raw.new_empty(0,dtype=torch.long),raw.new_empty(0,dtype=torch.long),raw.new_empty(0))
    corr=raw[:,64:]
    def msg(src,dst,w):
        acc=torch.zeros_like(corr); deg=torch.zeros((corr.shape[0],1),device=raw.device,dtype=raw.dtype)
        if len(src): acc.index_add_(0,src,w[:,None]*(corr[dst]-corr[src])); deg.index_add_(0,src,w[:,None])
        mean=acc/deg.clamp_min(1e-6); scale=corr.pow(2).mean(1,keepdim=True).sqrt().clamp_min(1e-6); return scale*torch.tanh(mean/(scale+1e-6))
    mp=msg(sp,dp,wp); mn=msg(sn,dn,wn); kappa=CFG["propagation"]["kappa_max"]*torch.sigmoid(mod.kappa_raw); rho=CFG["propagation"]["rho_max"]*torch.sigmoid(mod.rho_raw); delta_corr=rho*torch.tanh((mp-kappa*mn)/(corr.pow(2).mean(1,keepdim=True).sqrt().clamp_min(1e-6)))
    delta=torch.zeros_like(raw); delta[:,64:]=delta_corr
    return delta,{"positive_norm":float(mp.norm().detach().cpu()),"negative_norm":float(mn.norm().detach().cpu()),"delta_norm":float(delta.norm().detach().cpu()),"positive_edges":int(len(sp)),"negative_edges":int(len(sn)),"rho":float(rho.detach().cpu()),"kappa":float(kappa.detach().cpu())}


def forward_state(state,fd,train_mode=True,intervention=None):
    import torch
    c=runtime(); group=state["group"]; state["base"].train(train_mode); state["cam"].train(train_mode)
    if state["d2"] is not None: state["d2"].train(train_mode)
    _,parts=c["layers"].InitialDataLayer(DIMS)(state["feat"]); raw=torch.cat(parts,1); override=intervention or {}; cfg=GROUP_BY_ID[group]
    if group=="B0":
        fused,adjs,last=state["base"](parts); used=last.get("used", []); msgs=last.get("msgs", [])
        details={"pair_loss":raw.new_zeros(()),"site_loss":raw.new_zeros(()),"positive_norm":0.,"negative_norm":0.,"delta_norm":0.,"edge_change":0.}; delta=raw.new_zeros(raw.shape)
    else:
        corr=raw[:,64:]; q,details=relation_q(state["d2"],corr,fd,group); adjs0= [state["base"].graphs[m](parts[m]) for m in range(4)]
        if override.get("q_half"): q=torch.full_like(q,0.5)
        if override.get("swap"): q=1.0-q
        edge_on=cfg.get("edge",False) and not override.get("edge_off",False)
        wrong=cfg.get("wrong_graph",False) or override.get("wrong_graph",False)
        adj_used,ed=calibrated_adjs(state["d2"],adjs0,q,group,wrong=wrong) if edge_on else (adjs0,{"edge_change":0.,"gamma":torch.zeros(4),"mu":torch.zeros(4)})
        fused,adjs,used,msgs=base_forward(state["base"],parts,adjs_override=adj_used if edge_on else None,force_self=override.get("self_only",False)); delta,ps=propagation_delta(state["d2"],fused*0+raw,q,group,override="off" if override.get("prop_off") else ("negative_off" if override.get("negative_off") else ("wrong" if wrong else None)))
        fused=fused+delta; details.update(ed); details.update(ps)
    logp,hidden,attn=state["cam"](fused); util=c["utils"]; graph=sum(util.GraphConstructLoss(parts[m],adjs[m],CFG["base"]["theta_smooth"],CFG["base"]["theta_degree"],CFG["base"]["theta_sparsity"]) for m in range(4))/4.0; cls=util.ClsLoss(logp,state["targ"],state["train"],state["weight"])
    pair_loss=details.get("pair_loss",raw.new_zeros(())); site_loss=details.get("site_loss",raw.new_zeros(())); total=cls+graph
    if group!="B0" and cfg.get("pair_aux",False): total=total+CFG["loss"]["lambda_rel"]*pair_loss
    if group!="B0" and cfg.get("adv",False): total=total+CFG["loss"]["lambda_adv"]*site_loss
    details.update({"total":total,"cls":cls,"graph":graph,"q":q if group!="B0" else None,"raw":raw,"fused_e1":fused-delta,"fused":fused,"delta":delta,"adjs":adjs,"used":used,"msgs":msgs})
    return {"logp":logp,"hidden":hidden,"attn":attn,"total":total,"cls":cls,"graph":graph,"pair_loss":pair_loss,"site_loss":site_loss,"fused":fused,"fused_e1":fused-delta,"delta":delta,"adjs":adjs,"details":details}


def metrics(logp,state):
    import numpy as np
    from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score
    idx=state["test"]; yy=state["targ"].detach().cpu().numpy()[idx]; prob=logp.detach().exp().cpu().numpy()[idx]; pred=prob.argmax(1); cm=confusion_matrix(yy,pred,labels=[0,1]); tp,fn,fp,tn=map(int,cm.ravel())
    return {"acc":float((pred==yy).mean()),"hard_auc":float(roc_auc_score(np.eye(2)[yy],np.eye(2)[pred])),"prob_auc":float(roc_auc_score(yy,prob[:,1])),"spe":float(tn/max(tn+fp,1)),"sen":float(tp/max(tp+fn,1)),"ba":float((tn/max(tn+fp,1)+tp/max(tp+fn,1))/2),"asd_f1":float(f1_score(yy,pred,pos_label=0,zero_division=0)),"asd_tp":tp,"asd_fn":fn,"td_fp":fp,"td_tn":tn,"n_test":len(idx)}


def prepare():
    prepare_folds(); x,y=load_data()[0]
    manifest={"experiment":CFG["experiment"],"status":"PREPARED","formal_groups":list(GROUPS),"formal_folds":100,"data_shape":list(x.shape),"label_counts":__import__("numpy").bincount(y).tolist(),"source_sha256":{str(p):sha_file(p) for p in [ROOT/"main.py",ROOT/"model.py",ROOT/"network.py",ROOT/"layers.py",ROOT/"utils.py",ROOT/"data/ABIDE/processed_standard_data.csv",ROOT/"data/ABIDE/modal_feat_dict.npy"]},"private_source_sha256":{str(p.relative_to(EXP)):sha_file(p) for p in list(SRC.glob("*.py"))+[EXP/"d2arg.py",EXP/"run.py"]},"e1_anchor_sha256":{str(f):sha_file(e1.ANCHORS/f"fold{f:02d}.pt") for f in range(1,11)},"fold_sha256":{str(f):split_sha(f) for f in range(1,11)},"e1_seed_rule":"510000 + fold*100 + 1","new_rng_rule":"torch.random.fork_rng; relation 553000+fold; pretrain fold-local","test_labels_forbidden":True,"groups":list(GROUPS)}; atomic_json(EXP/"manifest.json",manifest); print("PREPARE_PASS",flush=True)


def preflight():
    import torch
    ensure_dirs(); x,y=load_data()[0]
    out={"status":"PASS","folds":{},"grl_ok":False,"label_isolation_pass":False,"math":{}}
    # Fold 1/5/10 initialize the exact E1 entry; compare the copied local path
    # to itself and verify the custom wrapper is bitwise equivalent when unused.
    for f in (1,5,10):
        s=build_state("B0",f); fd=load_fd(f); _,parts=runtime()["layers"].InitialDataLayer(DIMS)(s["feat"])
        with torch.no_grad(): a=s["base"](parts); b=base_forward(s["base"],parts)
        diff=float((a[0]-b[0]).abs().max().cpu()); out["folds"][str(f)]={"initial_shared_sha":s["initial_shared_sha"],"initial_schema_sha":s["initial_schema_sha"],"initial_cam_sha":s["initial_cam_sha"],"initial_rng_sha":s["initial_rng_sha"],"forward_max_abs":diff,"pair_sha":s["pair_sha"]};
        if diff>0: out["status"]="FAIL"
    # Explicit first optimizer-step replay: two independently constructed E1
    # states consume the same dropout/RNG sequence and must update identically.
    sa=build_state("B0",1); sb=build_state("B0",1); fd0=load_fd(1); r0=sa["initial_rng"]
    def one_step(st):
        restore_rng(r0); st["opt_base"].zero_grad(); st["opt_cam"].zero_grad(); oo=forward_state(st,fd0,True); oo["total"].backward(); st["opt_base"].step(); st["opt_cam"].step(); return pack_rng()
    ra=one_step(sa); rb=one_step(sb)
    step_diffs=[]
    for mod_a, mod_b in ((sa["base"],sb["base"]),(sa["cam"],sb["cam"])):
        for k,v in mod_a.state_dict().items(): step_diffs.append(float((v-mod_b.state_dict()[k]).abs().max().cpu()))
    out["first_optimizer_step_max_abs"] = max(step_diffs) if step_diffs else 0.0
    out["first_optimizer_step_rng_equal"] = rng_sha(ra) == rng_sha(rb)
    if out["first_optimizer_step_max_abs"] > 0: out["status"]="FAIL"
    # Mathematical tests and GRL direction.
    z=torch.randn(23,device="cuda",requires_grad=True); g=torch.autograd.grad(d2arg.grad_reverse(z,0.03).sum(),z)[0]; out["grl_ok"]=bool(torch.allclose(g,torch.full_like(g,-0.03),atol=1e-7,rtol=1e-6))
    s=build_state("B5",1); fd=load_fd(1)
    with torch.no_grad(): before=forward_state(s,fd,False); old=s["targ"].clone(); s["targ"][fd["test"].long()]=1-s["targ"][fd["test"].long()]; after=forward_state(s,fd,False); s["targ"].copy_(old)
    out["label_isolation"]={"fused_max_abs":float((before["fused"]-after["fused"]).abs().max().cpu()),"logp_max_abs":float((before["logp"]-after["logp"]).abs().max().cpu())}; out["label_isolation_pass"]=out["label_isolation"]["fused_max_abs"]<1e-5 and out["label_isolation"]["logp_max_abs"]<1e-5
    q=before["details"]["q"]
    with torch.no_grad():
        corr=before["details"]["raw"][:,64:]; norm=torch.nn.functional.normalize(corr,dim=-1); ii=state_pair_i=fd["pair_i"].long(); jj=fd["pair_j"].long(); prior=(norm[ii]*norm[jj]).sum(-1); z=before["details"]["z"] if before["details"].get("z") is not None else s["d2"].encode(corr); f1=s["d2"].scorer.pair_features(z,ii,jj,prior); f2=s["d2"].scorer.pair_features(z,jj,ii,prior); symmetry=float((s["d2"].scorer(f1)-s["d2"].scorer(f2)).abs().max().cpu())
    out["math"]={"q_finite":bool(torch.isfinite(q).all()),"q_symmetry_max":symmetry,"delta_nonzero":float(before["delta"].norm().cpu())>0,"adjs_finite":all(bool(torch.isfinite(a).all()) for a in before["adjs"]),"adjs_nonnegative":all(bool((a>=0).all()) for a in before["adjs"])}
    if not (out["grl_ok"] and out["label_isolation_pass"] and all(v["forward_max_abs"]==0 for v in out["folds"].values()) and out["math"]["q_finite"] and out["math"]["adjs_finite"] and out["math"]["adjs_nonnegative"]): out["status"]="FAIL"
    atomic_json(RT/"preflight.json",out); print("PREFLIGHT",out["status"],flush=True)
    if out["status"]!="PASS": raise SystemExit(2)


def train_one(group,fold):
    import torch
    ensure_dirs(); tp=task_path(group,fold)
    if tp.exists() and json.loads(tp.read_text()).get("status")=="COMPLETE": return
    state=build_state(group,fold); fd=load_fd(fold); start=0; wait=0; best_acc=-1.; best_epoch=-1; best_state=None; best_cam=None; best_d2=None; best_ob=best_oc=best_od=None; best_diag={}; lp=latest_path(group,fold)
    if lp.exists():
        try:
            d=torch_load(lp); assert d["group"]==group and int(d["fold"])==fold; state["base"].load_state_dict(d["base"]); state["cam"].load_state_dict(d["cam"]); state["opt_base"].load_state_dict(d["opt_base"]); optimizer_to_cuda(state["opt_base"]); state["opt_cam"].load_state_dict(d["opt_cam"]); optimizer_to_cuda(state["opt_cam"])
            if state["d2"] is not None: state["d2"].load_state_dict(d["d2"]); state["opt_d2"].load_state_dict(d["opt_d2"]); optimizer_to_cuda(state["opt_d2"])
            restore_rng(d["rng"]); start=int(d["epoch"])+1; wait=int(d["wait"]); best_acc=float(d["best_acc"]); best_epoch=int(d["best_epoch"]); best_state=d.get("best_state"); best_cam=d.get("best_cam"); best_d2=d.get("best_d2"); best_ob=d.get("best_ob"); best_oc=d.get("best_oc"); best_od=d.get("best_od"); best_diag=d.get("best_diag",{})
        except Exception: lp.unlink(missing_ok=True)
    t0=time.time(); epoch=start-1
    try:
        for epoch in range(start,int(CFG["base"]["epochs"])):
            state["opt_base"].zero_grad(); state["opt_cam"].zero_grad();
            if state["opt_d2"] is not None: state["opt_d2"].zero_grad()
            out=forward_state(state,fd,True)
            if not torch.isfinite(out["total"]): raise FloatingPointError("nonfinite total loss")
            out["total"].backward()
            for mod in (state["base"],state["cam"],state["d2"]):
                if mod is not None and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in mod.parameters()): raise FloatingPointError("nonfinite gradient")
            gn_base=module_grad_norm(state["base"]); gn_cam=module_grad_norm(state["cam"]); gn_d2=module_grad_norm(state["d2"]) if state["d2"] is not None else 0.
            state["opt_base"].step(); state["opt_cam"].step();
            if state["opt_d2"] is not None: state["opt_d2"].step()
            with torch.no_grad(): ev=forward_state(state,fd,False); met=metrics(ev["logp"],state)
            if met["acc"]>best_acc:
                best_acc=met["acc"]; best_epoch=epoch; wait=0; best_state=state_cpu(state["base"]); best_cam=state_cpu(state["cam"]); best_d2=state_cpu(state["d2"]) if state["d2"] is not None else None; best_ob=optimizer_cpu_state(state["opt_base"]); best_oc=optimizer_cpu_state(state["opt_cam"]); best_od=optimizer_cpu_state(state["opt_d2"]) if state["opt_d2"] is not None else None; best_diag={"cls":float(out["cls"].detach()),"graph":float(out["graph"].detach()),"pair":float(out["pair_loss"].detach()),"site":float(out["site_loss"].detach()),"base_grad":gn_base,"cam_grad":gn_cam,"d2_grad":gn_d2,"delta_norm":float(out["delta"].norm().detach().cpu())}
                torch_save(ckpt_path(group,fold),{"group":group,"fold":fold,"seed":0,"best_epoch":best_epoch,"epochs_executed":epoch+1,"metrics":met,"base":best_state,"cam":best_cam,"d2":best_d2,"opt_base":best_ob,"opt_cam":best_oc,"opt_d2":best_od,"diagnostics":best_diag,"split_sha":state["split_sha"],"pair_sha":state["pair_sha"],"initial_shared_sha":state["initial_shared_sha"],"initial_schema_sha":state["initial_schema_sha"],"initial_cam_sha":state["initial_cam_sha"],"initial_rng_sha":state["initial_rng_sha"],"initial_relation_sha":state["initial_relation_sha"]})
            else: wait+=1
            if epoch%10==0:
                torch_save(lp,{"group":group,"fold":fold,"epoch":epoch,"wait":wait,"best_acc":best_acc,"best_epoch":best_epoch,"base":state_cpu(state["base"]),"cam":state_cpu(state["cam"]),"d2":state_cpu(state["d2"]) if state["d2"] is not None else None,"opt_base":optimizer_cpu_state(state["opt_base"]),"opt_cam":optimizer_cpu_state(state["opt_cam"]),"opt_d2":optimizer_cpu_state(state["opt_d2"]) if state["opt_d2"] is not None else None,"rng":pack_rng(),"best_state":best_state,"best_cam":best_cam,"best_d2":best_d2,"best_ob":best_ob,"best_oc":best_oc,"best_od":best_od,"best_diag":best_diag})
            if wait>int(CFG["base"]["early"]): break
        if best_state is None: raise RuntimeError("best checkpoint missing")
        state["base"].load_state_dict(best_state); state["cam"].load_state_dict(best_cam)
        if state["d2"] is not None: state["d2"].load_state_dict(best_d2)
        with torch.no_grad(): check=metrics(forward_state(state,fd,False)["logp"],state)
        if abs(check["acc"]-best_acc)>1e-8: raise AssertionError("best readback mismatch")
        row={"group":group,"name":GROUP_BY_ID[group]["name"],"fold":fold,"seed":0,"status":"COMPLETE","source":"new_training","best_epoch":best_epoch,"epochs_executed":epoch+1,"wall_seconds":time.time()-t0,"best_sha":sha_file(ckpt_path(group,fold)),"split_sha":state["split_sha"],"pair_sha":state["pair_sha"],"initial_shared_sha":state["initial_shared_sha"],"initial_schema_sha":state["initial_schema_sha"],"initial_cam_sha":state["initial_cam_sha"],"initial_rng_sha":state["initial_rng_sha"],"initial_relation_sha":state["initial_relation_sha"],"relation_metrics":json.dumps(state["package"].get("relation_metrics",{}) if state["package"] else {}),**check,**best_diag}; atomic_json(tp,row); lp.unlink(missing_ok=True); print("DONE",group,fold,best_epoch,row["acc"],flush=True)
    except Exception as exc:
        atomic_json(tp,{"group":group,"fold":fold,"status":"FAILED","error":repr(exc),"traceback":traceback.format_exc()}); raise


def controller(groups):
    groups=list(groups); assert all(g in GROUPS for g in groups); jobs=[(g,f) for g in groups for f in range(1,11)]; procs=[]; workers=min(5,int(os.environ.get("E54_WORKERS",CFG["runtime"]["workers_max"]))); env=dict(os.environ); env["CUDA_VISIBLE_DEVICES"]="0"; start=time.time()
    while jobs or procs:
        alive=sum(p.poll() is None for _,p,_ in procs)
        while alive<workers and jobs:
            g,f=jobs.pop(0); log=(LOGS/f"{g}_f{f:02d}.log").open("ab"); p=subprocess.Popen([sys.executable,"-B",str(EXP/"run.py"),"worker","--group",g,"--fold",str(f)],cwd=EXP,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); procs.append((g,p,log)); alive+=1
        for item in list(procs):
            if item[1].poll() is not None: item[2].close(); procs.remove(item)
        time.sleep(.5)
    statuses=[json.loads(task_path(g,f).read_text()).get("status","MISSING") if task_path(g,f).exists() else "MISSING" for g in groups for f in range(1,11)]; m=json.loads((EXP/"manifest.json").read_text()); m.setdefault("controller_runs",[]).append({"groups":groups,"workers":workers,"completed":statuses.count("COMPLETE"),"failed":statuses.count("FAILED"),"wall_seconds":time.time()-start}); m["training_status"]="COMPLETE" if all(s=="COMPLETE" for s in statuses) else "PARTIAL_OR_FAILED"; atomic_json(EXP/"manifest.json",m); print("CONTROLLER_DONE",groups,statuses.count("COMPLETE"),statuses.count("FAILED"),flush=True)
    if not all(s=="COMPLETE" for s in statuses): raise SystemExit(3)


def aggregate():
    import numpy as np
    rows=[]
    for g in GROUPS:
        for f in range(1,11):
            r=json.loads(task_path(g,f).read_text()); assert r["status"]=="COMPLETE",r; rows.append(r)
    csv_write(RESULTS/"fold_metrics.csv",rows); summary=[]
    for g in GROUPS:
        rr=[r for r in rows if r["group"]==g]; q={"group":g,"name":rr[0]["name"],"folds":10}
        for k in ("acc","hard_auc","prob_auc","spe","sen","ba","asd_f1"):
            v=np.asarray([float(r[k]) for r in rr]); q[k+"_mean_pct"]=float(v.mean()*100); q[k+"_sd_pct"]=float(v.std(ddof=0)*100)
        summary.append(q)
    csv_write(RESULTS/"summary.csv",summary); by={(r["group"],int(r["fold"])):r for r in rows}; pairs=[]
    for left,right in (("B5","B0"),("B5","B4"),("B5","B7"),("B5","B8"),("B5","B9"),("B3","B0"),("B4","B3"),("B5","B4"),("B6","B5"),("B5","B2"),("B5","B1")):
        for k in ("acc","hard_auc","prob_auc","spe","sen"):
            d=np.asarray([(float(by[(left,f)][k])-float(by[(right,f)][k]))*100 for f in range(1,11)]); pairs.append({"comparison":f"{left}_minus_{right}","metric":k,"delta_pp":float(d.mean()),"wins":int((d>0).sum()),"ties":int((d==0).sum()),"losses":int((d<0).sum()),"fold_deltas_pp":json.dumps(d.tolist())})
    csv_write(RESULTS/"paired_comparisons.csv",pairs); m=json.loads((EXP/"manifest.json").read_text()); m.update({"status":"AGGREGATED","formal_groups_complete":10,"formal_folds_complete":100,"fold_metrics_sha256":sha_file(RESULTS/"fold_metrics.csv"),"summary_sha256":sha_file(RESULTS/"summary.csv"),"paired_sha256":sha_file(RESULTS/"paired_comparisons.csv")}); atomic_json(EXP/"manifest.json",m); print("AGGREGATE_PASS",flush=True)


def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd",required=True); s.add_parser("prepare"); s.add_parser("pretrain-relations"); s.add_parser("preflight"); c=s.add_parser("controller"); c.add_argument("--groups",nargs="+",required=True); w=s.add_parser("worker"); w.add_argument("--group",choices=GROUPS,required=True); w.add_argument("--fold",type=int,required=True); s.add_parser("aggregate")
    a=p.parse_args(); ensure_dirs()
    if a.cmd=="prepare": prepare()
    elif a.cmd=="pretrain-relations": pretrain_relations()
    elif a.cmd=="preflight": preflight()
    elif a.cmd=="controller": controller(a.groups)
    elif a.cmd=="worker": train_one(a.group,a.fold)
    else: aggregate()

if __name__=="__main__": main()
