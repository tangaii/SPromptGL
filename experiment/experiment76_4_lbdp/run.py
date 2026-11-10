#!/usr/bin/env python3
"""Frozen E74.4-G2 local-boundary prototype and residual correction probe."""
from __future__ import annotations

import csv, hashlib, importlib.util, json, math, os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text())
BASE = Path(CFG["base"]["path"])
E762 = Path(CFG["references"]["e762_path"])
E763 = Path(CFG["references"]["e763_path"])
RESULTS, CACHE = EXP / "results", EXP / ".runtime/cache"
BASE_GROUP = CFG["base"]["name"]
GROUPS = ("M0_original_mlp", "M1_base_score_calibration", "M2_global_hard_prototype",
          "M3_local_boundary_prototype", "M4_shuffled_local_pair")
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1", "log_loss", "brier")
P, E3, LP = None, None, None


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules():
    global P, E3, LP
    if P is None:
        P = load_module("e764_locked_e762_probe", E762 / "probe.py")
        E3 = load_module("e764_locked_e763_run", E763 / "run.py")
        LP = load_module("e764_local_prototypes", EXP / "local_prototypes.py")
    return P, E3, LP


def inventory():
    p, _, _ = modules()
    paths = {
        "config": EXP / "config.json", "run": EXP / "run.py", "local_prototypes": EXP / "local_prototypes.py",
        "audit": EXP / "audit.py", "e762_probe": E762 / "probe.py", "e762_audit": E762 / "audit.py",
        "e762_config": E762 / "config.json", "e762_manifest": E762 / "manifest.json",
        "e762_report": E762 / "results/report.md", "e762_metrics": E762 / "results/fold_metrics.csv",
        "e762_status": E762 / "results/audit_status.json", "e762_cache": E762 / ".runtime/cache/oof_predictions.npz",
        "e763_run": E763 / "run.py", "e763_audit": E763 / "audit.py", "e763_config": E763 / "config.json",
        "e763_manifest": E763 / "manifest.json", "e763_report": E763 / "results/report.md",
        "e763_metrics": E763 / "results/fold_metrics.csv", "e763_status": E763 / "results/audit_status.json",
        "e763_cache": E763 / ".runtime/cache/oof_predictions.npz",
        "e744_run": BASE / "run.py", "e744_model": BASE / "soft_positive_dcr.py",
        "e744_config": BASE / "config.json", "e744_manifest": BASE / "manifest.json",
        "e744_metrics": BASE / "results/fold_metrics.csv", "e744_status": BASE / "results/audit_status.json",
        "processed": Path("/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv"),
        "mapping": Path("/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy"),
        "site_decoder": Path("/model/spromptgl/experiment/experiment72_4_dual_level_semantic_coordination/dual_prompt.py")
    }
    paths.update({f"best_{f}": BASE / f".runtime/checkpoints/{BASE_GROUP}_f{f:02d}.best.pt" for f in range(1, 11)})
    paths.update({f"task_{f}": BASE / f".runtime/tasks/{BASE_GROUP}_f{f:02d}.json" for f in range(1, 11)})
    paths.update({f"warmup_{f}": Path(CFG["base"]["warmup_path"]) / f"fold{f:02d}.pt" for f in range(1, 11)})
    missing = [str(x) for x in paths.values() if not x.is_file()]
    if missing:
        raise FileNotFoundError("locked reference files missing: " + ", ".join(missing))
    return {k: sha_file(v) for k, v in paths.items()}


def verify_references():
    p, e3, _ = modules()
    e3.verify_references()
    remote={}
    branch_locks=(("base",CFG["base"]["branch"],CFG["base"]["commit"]),
                  ("e762",CFG["references"]["e762_branch"],CFG["references"]["e762_commit"]),
                  ("e763",CFG["references"]["e763_branch"],CFG["references"]["e763_commit"]))
    for key,branch,expected in branch_locks:
        out = subprocess.run(["git", "ls-remote", "https://github.com/tangaii/SPromptGL.git",
                              f"refs/heads/{branch}"], check=True, capture_output=True, text=True, timeout=45)
        rows = [line.split() for line in out.stdout.splitlines() if line.strip()]
        if len(rows) != 1 or not rows[0][0].startswith(expected):
            raise RuntimeError(f"{key} GitHub branch does not match locked commit {expected}: {rows}")
        remote[key]=rows[0][0]
    e74 = json.loads((BASE / "manifest.json").read_text())
    e74_audit = json.loads((BASE / "results/audit_status.json").read_text())
    if e74.get("status") != "COMPLETE_AUDITED" or e74_audit.get("status") != "PASS" or e74_audit.get("audited_checkpoints") != 30:
        raise RuntimeError("E74.4 base audit is not PASS")
    return {"e74_audit": "PASS_30_CHECKPOINTS", "e762_audit": "PASS", "e763_audit": "PASS",
            "e744_commit": remote["base"], "e762_commit": remote["e762"], "e763_commit": remote["e763"]}


def extract_fold(fold):
    """Use the audited E76.2 extractor; retain its cached perturbation stability."""
    p, _, _ = modules()
    data = p.extract_fold(fold)
    old = __import__("numpy").load(E762 / ".runtime/cache/oof_predictions.npz", allow_pickle=False)
    prefix = f"f{fold}_"
    eligible = __import__("numpy").asarray(old[prefix + "eligible"], dtype=int)
    if not __import__("numpy").array_equal(data["eligible"], eligible):
        raise RuntimeError(f"fold {fold}: eligible training IDs differ from E76.2 cache")
    if not __import__("numpy").array_equal(data["y"][eligible], old[prefix + "labels"]):
        raise RuntimeError(f"fold {fold}: E76.2 cached labels mismatch")
    if not __import__("numpy").array_equal(data["pred"][eligible], old[prefix + "base_pred"]):
        raise RuntimeError(f"fold {fold}: frozen MLP predictions mismatch E76.2")
    if __import__("numpy").max(__import__("numpy").abs(data["prob"][eligible] - old[prefix + "base_prob"])) > 1e-6:
        raise RuntimeError(f"fold {fold}: frozen MLP probabilities mismatch E76.2")
    np = __import__("numpy")
    if not np.allclose(data["stability"][eligible], old[prefix + "stability"], rtol=0, atol=1e-12):
        raise RuntimeError(f"fold {fold}: recomputed stability differs from the audited E76.2 cache")
    if fold == 2 and ({376, 406} & set(eligible.tolist())):
        raise RuntimeError("Fold 2 known exposed subjects entered the eligible support")
    return data


def sigmoid(x):
    import numpy as np
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def fit_calibrator(base_score, evidence, y):
    """Fit deterministic residual logistic calibration, with base score offset."""
    import numpy as np
    from scipy.optimize import minimize
    from sklearn.preprocessing import StandardScaler
    base_score = np.asarray(base_score, dtype=float).reshape(-1)
    evidence = np.asarray(evidence, dtype=float)
    X = np.c_[base_score, evidence]
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    target = (np.asarray(y, dtype=int) == 0).astype(float)
    lam = float(CFG["evaluation"]["l2"])

    def objective(theta):
        b, w = theta[0], theta[1:]
        logits = base_score + b + Z @ w
        loss = np.logaddexp(0.0, logits) - target * logits
        val = float(np.mean(loss) + .5 * lam * np.dot(w, w))
        residual = sigmoid(logits) - target
        grad = np.r_[residual.mean(), Z.T @ residual / len(target) + lam * w]
        return val, grad

    result = minimize(objective, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B",
                      options={"maxiter": int(CFG["evaluation"]["max_iterations"]), "ftol": 1e-12, "gtol": 1e-8, "maxls": 50})
    if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
        raise RuntimeError("residual calibrator returned non-finite parameters")
    if not result.success and np.linalg.norm(result.jac) > 1e-5:
        raise RuntimeError(f"residual calibrator failed convergence: {result.message}; grad={np.linalg.norm(result.jac)}")
    return {"theta": result.x, "scaler": scaler, "success": bool(result.success),
            "iterations": int(result.nit), "objective": float(result.fun),
            "gradient_norm": float(np.linalg.norm(result.jac))}


def score_calibrator(model, base_score, evidence):
    import numpy as np
    base_score = np.asarray(base_score, dtype=float).reshape(-1)
    X = np.c_[base_score, np.asarray(evidence, dtype=float)]
    Z = model["scaler"].transform(X)
    logits = base_score + model["theta"][0] + Z @ model["theta"][1:]
    return logits, sigmoid(logits)


def metric(y, pred, p_asd):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, log_loss, brier_score_loss
    y = np.asarray(y, dtype=int); pred = np.asarray(pred, dtype=int); p_asd = np.asarray(p_asd, dtype=float)
    return {"acc": float(accuracy_score(y, pred)), "hard_auc": float(balanced_accuracy_score(y, pred)),
            "prob_auc": float(roc_auc_score(y == 0, p_asd)),
            "sen": float(np.sum((y == 0) & (pred == 0)) / max(1, np.sum(y == 0))),
            "spe": float(np.sum((y == 1) & (pred == 1)) / max(1, np.sum(y == 1))),
            "asd_f1": float(f1_score(y == 0, pred == 0, zero_division=0)),
            "log_loss": float(log_loss((y == 0).astype(int), p_asd, labels=[0, 1])),
            "brier": float(brier_score_loss((y == 0).astype(int), p_asd))}


def global_ev(p, data, support, query_ids, include_hard=True):
    mem = p.build_memory(data, support)
    return p.evidence(data["h"][query_ids], mem, include_hard), mem


def local_ev(lp, data, support, query_ids, sites, shuffle_seed=None):
    p, _, _ = modules()
    mem = lp.build_local_memory(data, support, data["h"], sites, p.role_masks, shuffle_seed)
    ev = lp.local_evidence(data["h"][query_ids], mem, data["h"],
                           CFG["roles"]["local_temperature"], CFG["roles"]["local_top_k"])
    return ev, mem


def crossfit_ev(kind, p, lp, data, ids, sites, permutation=None, split_tag=0):
    """3-fold training evidence, each query excluded from its prototype support."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    ids = np.asarray(ids, dtype=int); y = data["y"]
    out = np.zeros((len(ids), 12), dtype=float)
    cv = StratifiedKFold(n_splits=3, shuffle=True,
                         random_state=int(CFG["evaluation"]["train_evidence_random_state_base"]) + int(data["fold"]))
    for subfold, (spos, qpos) in enumerate(cv.split(ids, y[ids])):
        support, query = ids[spos], ids[qpos]
        if np.intersect1d(support, query).size:
            raise RuntimeError("training evidence crossfit self-match")
        if kind == "M2":
            out[qpos], _ = global_ev(p, data, support, query, True)
        else:
            seed = None if permutation is None else int(CFG["evaluation"]["pair_permutation_seed_base"] + data["fold"]*100000 + split_tag*1000 + permutation*10 + subfold)
            out[qpos], _ = local_ev(lp, data, support, query, sites, seed)
    return out


def fit_predict(kind, data, train_ids, valid_ids, train_ev, valid_ev):
    import numpy as np
    y = data["y"]
    if kind == "M1":
        train_ev = np.zeros((len(train_ids), 12)); valid_ev = np.zeros((len(valid_ids), 12))
    model = fit_calibrator(data["base_score"][train_ids], train_ev, y[train_ids])
    logits, probs = score_calibrator(model, data["base_score"][valid_ids], valid_ev)
    pred = np.where(logits >= 0.0, 0, 1)
    return pred.astype(int), probs.astype(float), logits.astype(float), model


def boundary_rows(fold, scope, y, base_pred, global_ev_arr, local_ev_arr, sites, ids):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    rows = []
    for hard_cls, opposite_pred, slot in ((0, 1, 2), (1, 0, 3)):
        pos = np.asarray(ids)[(np.asarray(y) == hard_cls) & (np.asarray(base_pred) == opposite_pred)]
        neg = np.asarray(ids)[(np.asarray(y) == 1-hard_cls) & (np.asarray(base_pred) == opposite_pred)]
        if len(pos) and len(neg):
            for method, ev in (("M2_global", global_ev_arr), ("M3_local", local_ev_arr)):
                score = np.r_[ev[np.isin(ids, pos), slot], ev[np.isin(ids, neg), slot]]
                target = np.r_[np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)]
                auc = float(roc_auc_score(target, score)) if len(np.unique(target)) == 2 else ""
                psim = ev[np.isin(ids, pos), slot]; nsim = ev[np.isin(ids, neg), slot]
                rows.append({"fold": fold, "scope": scope, "direction": "ASD_to_TD" if hard_cls == 0 else "TD_to_ASD",
                             "method": method, "n_hard_errors": len(pos), "n_correct_opposite": len(neg),
                             "conditional_auc": auc, "mean_hard_evidence": float(np.mean(psim)),
                             "mean_correct_opposite_evidence": float(np.mean(nsim)),
                             "mean_difference": float(np.mean(psim)-np.mean(nsim)),
                             "hard_same_site_fraction": ""})
        else:
            for method in ("M2_global", "M3_local"):
                rows.append({"fold": fold, "scope": scope, "direction": "ASD_to_TD" if hard_cls == 0 else "TD_to_ASD",
                             "method": method, "n_hard_errors": len(pos), "n_correct_opposite": len(neg),
                             "conditional_auc": "NA_INSUFFICIENT_CLASS", "mean_hard_evidence": "",
                             "mean_correct_opposite_evidence": "", "mean_difference": "", "hard_same_site_fraction": ""})
    return rows


def correction_rows(fold, scope, data, ids, predictions, scores):
    import numpy as np
    p, _, _ = modules()
    ids = np.asarray(ids, dtype=int); y = data["y"][ids]; base = data["pred"][ids]
    masks = p.role_masks(data, ids)
    boundary_asd=(y==0)&(base==0)&(data["prob"][ids,0]>=.50)&(data["prob"][ids,0]<.70)
    boundary_td=(y==1)&(base==1)&(data["prob"][ids,1]>=.50)&(data["prob"][ids,1]<.70)
    base_p_asd=data["prob"][ids,0]
    base_p_true=np.where(y==0,base_p_asd,1-base_p_asd)
    base_conf=np.maximum(base_p_asd,1-base_p_asd)
    subsets = {"all": np.ones(len(ids), dtype=bool),
               "stable_hard": masks[2][ids] | masks[3][ids],
               "stable_reliable": masks[0][ids] | masks[1][ids],
               "boundary_correct": boundary_asd | boundary_td,
               "ASD_to_TD_errors": (y==0)&(base==1),
               "TD_to_ASD_errors": (y==1)&(base==0),
               "low_confidence_errors": (base!=y)&(base_conf<.70),
               "high_confidence_errors": (base!=y)&(base_conf>=.70)}
    rows=[]
    for group in GROUPS[1:]:
        pred=np.asarray(predictions[group],dtype=int); prob=np.asarray(scores[group],dtype=float)
        for subset,mask in subsets.items():
            if not np.any(mask):
                continue
            wrong=(base!=y)&mask; right=(base==y)&mask
            rows.append({"fold":fold,"scope":scope,"group":group,"subset":subset,
                         "wrong_to_correct":int(np.sum(wrong&(pred==y))),
                         "correct_to_wrong":int(np.sum(right&(pred!=y))),
                         "net_correction":int(np.sum(wrong&(pred==y))-np.sum(right&(pred!=y))),
                         "asd_to_td_before":int(np.sum(mask&(y==0)&(base==1))),
                         "asd_to_td_after":int(np.sum(mask&(y==0)&(pred==1))),
                         "td_to_asd_before":int(np.sum(mask&(y==1)&(base==0))),
                         "td_to_asd_after":int(np.sum(mask&(y==1)&(pred==0))),
                         "raw_error_count":int(np.sum(wrong)),"raw_correct_count":int(np.sum(right)),
                         "original_correct_case_retention":float(np.sum(right&(pred==y))/max(1,np.sum(right))),
                         "original_error_correction_rate":float(np.sum(wrong&(pred==y))/max(1,np.sum(wrong))),
                         "mean_abs_asd_probability_change":float(np.mean(np.abs(prob-data["prob"][ids,0])))} )
    return rows


def run_fold(fold, keep_details=True):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    p, _, lp = modules()
    data = extract_fold(fold)
    sites, site_names, site_meta = p.site_data()
    ids, test = np.asarray(data["eligible"], int), np.asarray(data["test"], int)
    y, h = data["y"], data["h"]
    data["base_score"] = np.log(np.clip(data["prob"][:,0],1e-12,1.0)) - np.log(np.clip(data["prob"][:,1],1e-12,1.0))
    if np.intersect1d(ids, test).size or (fold==2 and ({376,406}&set(ids.tolist()))):
        raise RuntimeError("outer test or known overlap row entered support")
    inner_cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=10)
    inner_pred={g:np.full(len(ids),-1,dtype=int) for g in GROUPS}
    inner_prob={g:np.full(len(ids),np.nan,dtype=float) for g in GROUPS}
    inner_score={g:np.full(len(ids),np.nan,dtype=float) for g in GROUPS}
    inner_global=np.full((len(ids),12),np.nan); inner_local=np.full((len(ids),12),np.nan)
    split_rows=[]; calibrator_rows=[]; detail_pairs=[]
    for inner,(trpos,vapos) in enumerate(inner_cv.split(ids,y[ids]),start=1):
        tr,va=ids[trpos],ids[vapos]
        if np.intersect1d(tr,va).size or np.intersect1d(va,test).size:
            raise RuntimeError(f"fold {fold} inner {inner}: split leakage")
        # Base model is a fixed historical prediction, not refit on inner support.
        inner_pred["M0_original_mlp"][vapos]=data["pred"][va]
        inner_prob["M0_original_mlp"][vapos]=data["prob"][va,0]
        inner_score["M0_original_mlp"][vapos]=data["base_score"][va]
        tr_m2=crossfit_ev("M2",p,lp,data,tr,sites,split_tag=inner)
        va_m2,mem2=global_ev(p,data,tr,va,True)
        tr_m3=crossfit_ev("M3",p,lp,data,tr,sites,split_tag=inner)
        va_m3,mem3=local_ev(lp,data,tr,va,sites)
        inner_global[vapos]=va_m2; inner_local[vapos]=va_m3
        model_ev={"M1":(np.zeros((len(tr),12)),np.zeros((len(va),12))),
                  "M2":(tr_m2,va_m2),"M3":(tr_m3,va_m3)}
        for kind,group in (("M1","M1_base_score_calibration"),("M2","M2_global_hard_prototype"),("M3","M3_local_boundary_prototype")):
            pr,prb,sc,cal=fit_predict(kind,data,tr,va,*model_ev[kind])
            inner_pred[group][vapos]=pr; inner_prob[group][vapos]=prb; inner_score[group][vapos]=sc
            calibrator_rows.append({"fold":fold,"scope":"inner","inner_fold":inner,"group":group,
                                    "success":cal["success"],"iterations":cal["iterations"],"gradient_norm":cal["gradient_norm"]})
        m4_scores=[];m4_probs=[]
        for perm in range(5):
            tr_ev=crossfit_ev("M4",p,lp,data,tr,sites,permutation=perm,split_tag=inner)
            seed=764000+fold*100000+inner*1000+perm
            va_ev,mem4=local_ev(lp,data,tr,va,sites,seed)
            pr,prb,sc,cal=fit_predict("M4",data,tr,va,tr_ev,va_ev)
            m4_scores.append(sc);m4_probs.append(prb)
            calibrator_rows.append({"fold":fold,"scope":"inner","inner_fold":inner,"group":"M4_shuffled_local_pair",
                                    "permutation":perm,"success":cal["success"],"iterations":cal["iterations"],"gradient_norm":cal["gradient_norm"]})
            if inner==1 and keep_details:
                detail_pairs.append({"scope":"inner","inner_fold":inner,"permutation":perm,
                                     "records":mem4["records"]})
        s4=np.mean(m4_scores,axis=0); p4=sigmoid(s4)
        inner_score["M4_shuffled_local_pair"][vapos]=s4
        inner_prob["M4_shuffled_local_pair"][vapos]=p4
        inner_pred["M4_shuffled_local_pair"][vapos]=np.where(s4>=0,0,1)
        split_rows.append({"fold":fold,"inner_fold":inner,"train_ids":tr.tolist(),"valid_ids":va.tolist(),
                           "n_train":len(tr),"n_valid":len(va),"M2_hard_counts":mem2["counts"][2:4],
                           "M3_hard_counts":mem3["hard_raw_counts"],"M3_pair_counts":[len(x) for x in mem3["records"]],
                           "M3_fallback_counts":mem3["fallback_counts"]})
    if np.any(inner_pred["M0_original_mlp"]<0) or np.any(~np.isfinite(inner_prob["M4_shuffled_local_pair"])):
        raise RuntimeError(f"fold {fold}: inner OOF did not cover all eligible subjects")

    # Frozen local/global evidence diagnostics on each patient's inner validation prediction.
    boundary_inner=boundary_rows(fold,"inner_oof",y[ids],data["pred"][ids],inner_global,inner_local,sites,ids)

    # Historical outer test: fit only on eligible outer train; the test cohort is query-only.
    train_m2=crossfit_ev("M2",p,lp,data,ids,sites,split_tag=900)
    test_m2,full_m2=global_ev(p,data,ids,test,True)
    train_m3=crossfit_ev("M3",p,lp,data,ids,sites,split_tag=900)
    test_m3,full_m3=local_ev(lp,data,ids,test,sites)
    outer_pred={"M0_original_mlp":data["pred"][test].astype(int)}
    outer_prob={"M0_original_mlp":data["prob"][test,0].astype(float)}
    outer_score={"M0_original_mlp":data["base_score"][test].astype(float)}
    for kind,group,tev,qev in (("M1","M1_base_score_calibration",None,None),
                                ("M2","M2_global_hard_prototype",train_m2,test_m2),
                                ("M3","M3_local_boundary_prototype",train_m3,test_m3)):
        if kind=="M1": tev=np.zeros((len(ids),12));qev=np.zeros((len(test),12))
        pr,prb,sc,cal=fit_predict(kind,data,ids,test,tev,qev)
        outer_pred[group]=pr;outer_prob[group]=prb;outer_score[group]=sc
        calibrator_rows.append({"fold":fold,"scope":"outer_test","group":group,"success":cal["success"],
                                "iterations":cal["iterations"],"gradient_norm":cal["gradient_norm"]})
    m4_scores=[];m4_probs=[];m4_records=[]
    for perm in range(5):
        tr_ev=crossfit_ev("M4",p,lp,data,ids,sites,permutation=perm,split_tag=900)
        seed=764000+fold*100000+900000+perm
        q_ev,mem4=local_ev(lp,data,ids,test,sites,seed)
        pr,prb,sc,cal=fit_predict("M4",data,ids,test,tr_ev,q_ev)
        m4_scores.append(sc);m4_probs.append(prb);m4_records.append(mem4["records"])
        calibrator_rows.append({"fold":fold,"scope":"outer_test","group":"M4_shuffled_local_pair",
                                "permutation":perm,"success":cal["success"],"iterations":cal["iterations"],"gradient_norm":cal["gradient_norm"]})
        if keep_details: detail_pairs.append({"scope":"outer_test","permutation":perm,"records":mem4["records"]})
    s4=np.mean(m4_scores,axis=0);p4=sigmoid(s4)
    outer_score["M4_shuffled_local_pair"]=s4;outer_prob["M4_shuffled_local_pair"]=p4
    outer_pred["M4_shuffled_local_pair"]=np.where(s4>=0,0,1)

    fold_metrics=[]; correction=[]
    for scope,scope_ids,preds,probs,scores in (("inner_oof",ids,inner_pred,inner_prob,inner_score),
                                               ("outer_test",test,outer_pred,outer_prob,outer_score)):
        yy=y[scope_ids]
        for group in GROUPS:
            met=metric(yy,preds[group],probs[group])
            fold_metrics.append({"fold":fold,"scope":scope,"group":group,"n":len(scope_ids),**met})
        correction.extend(correction_rows(fold,scope,data,scope_ids,preds,probs))
    # Fold 2 sensitivity drops the two known exposed rows from final test scoring only.
    outer_noexp={}
    if fold==2:
        keep=~np.isin(test,[376,406]); sens_ids=test[keep]
        for group in GROUPS:
            outer_noexp[group]={"ids":sens_ids,"pred":outer_pred[group][keep],"prob":outer_prob[group][keep]}
            met=metric(y[sens_ids],outer_noexp[group]["pred"],outer_noexp[group]["prob"])
            fold_metrics.append({"fold":fold,"scope":"outer_test_without_377_407","group":group,"n":len(sens_ids),**met})

    full_diag=lp.local_diagnostics(data,ids,full_m3,h,sites,seed=764000+fold)
    pair_registry=[]
    for c,rs in enumerate(full_m3["records"]):
        direction="ASD_to_TD" if c==0 else "TD_to_ASD"
        for rec in rs:
            pair_registry.append({"fold":fold,"direction":direction,**rec,
                                 "hard_class":int(c),"same_class":int(c),"other_class":int(1-c)})
    full_role=p.role_masks(data,ids)
    site_counts={str(site_names[s]):int(np.sum(sites[ids]==s)) for s in np.unique(sites[ids])}
    role_counts={"R_ASD":len(full_role[0][np.asarray(full_role[0]).nonzero()[0]]),
                 "R_TD":len(full_role[1][np.asarray(full_role[1]).nonzero()[0]]),
                 "H_ASD_to_TD":len(full_role[2][np.asarray(full_role[2]).nonzero()[0]]),
                 "H_TD_to_ASD":len(full_role[3][np.asarray(full_role[3]).nonzero()[0]])}
    # Direct site-match diagnostics for the local triples.
    pair_site=[]
    for c,records in enumerate(full_m3["records"]):
        for rec in records:
            pair_site.append({"fold":fold,"direction":"ASD_to_TD" if c==0 else "TD_to_ASD",
                              "same_site":rec["same_site"],"other_site":rec["other_site"],
                              "same_site_id":int(sites[rec["hard_id"]])==int(sites[rec["same_id"]]),
                              "other_site_id":int(sites[rec["hard_id"]])==int(sites[rec["other_id"]])})
    perm_change=[]
    for perm, records in enumerate(m4_records):
        for c, group_records in enumerate(records):
            perm_change.append({"permutation":perm,"direction":"ASD_to_TD" if c==0 else "TD_to_ASD",
                                "changed_fraction":float(np.mean([r.get("permutation_changed_n",0)/max(1,r.get("permutation_total_n",1)) for r in group_records])) if group_records else 0.0,
                                "pair_count":len(group_records)})
    diagnostics={"fold":fold,"eligible_n":len(ids),"test_n":len(test),"role_counts":role_counts,
                 "boundary_correct_counts":[len(full_m3["boundary_ids"][0]),len(full_m3["boundary_ids"][1])],
                 "local_pairs":[len(x) for x in full_m3["records"]],"fallback_counts":full_m3["fallback_counts"],
                 "role_available":full_m3["hard_available"],"sites":site_counts,
                 "pair_same_site_fraction":float(np.mean([r["same_site"] for r in pair_site])) if pair_site else "",
                 "pair_other_site_fraction":float(np.mean([r["other_site"] for r in pair_site])) if pair_site else "",
                 "bootstrap":full_diag,"permutation_changed_fraction_by_direction":perm_change,
                 "feature_sha256":data["feature_sha"],"checkpoint_sha256":data["checkpoint_sha"],
                 "checkpoint_weights_sha256":data["checkpoint_weights_sha"],"post_weights_sha256":data["post_weights_sha"],
                 "base_test_probability_max_error":data["probability_max_error"],"device":data["device"],
                 "output_layer_calls":1,"zero_perturb_exact":data["zero_perturb_exact"],
                 "frozen_weights_exact":data["frozen_weights_exact"],"site_source":site_meta}
    boundary_outer=boundary_rows(fold,"outer_test",y[test],data["pred"][test],test_m2,test_m3,sites,test)
    return {"fold":fold,"data":data,"inner_ids":ids,"test_ids":test,"fold_metrics":fold_metrics,
            "inner_pred":inner_pred,"inner_prob":inner_prob,"inner_score":inner_score,
            "outer_pred":outer_pred,"outer_prob":outer_prob,"outer_score":outer_score,
            "outer_noexp":outer_noexp,"correction":correction,"boundary":boundary_inner+boundary_outer,
            "diagnostics":diagnostics,"split_rows":split_rows,"calibrator_rows":calibrator_rows,
            "pair_registry":pair_registry,"inner_global":inner_global,"inner_local":inner_local,
            "outer_global":test_m2,"outer_local":test_m3,"sites":sites,"site_names":site_names,
            "details":detail_pairs}


def summarize_metrics(fold_rows):
    import numpy as np
    rows=[]
    for scope in ("inner_oof","outer_test"):
        for group in GROUPS:
            selected=[r for r in fold_rows if r["scope"]==scope and r["group"]==group]
            row={"scope":scope,"group":group,"folds":len(selected)}
            for m in METRICS:
                vals=np.asarray([float(x[m]) for x in selected],dtype=float)
                row[m+"_mean"]=float(np.mean(vals));row[m+"_sd_population"]=float(np.std(vals,ddof=0))
            rows.append(row)
    return rows


def paired_rows(fold_rows):
    import numpy as np
    from scipy.stats import t
    lookup={(r["scope"],int(r["fold"]),r["group"]):r for r in fold_rows}
    out=[]
    for scope in ("inner_oof","outer_test"):
        for other in GROUPS:
            if other=="M3_local_boundary_prototype": continue
            cmp=f"M3_local_boundary_prototype - {other}"
            for m in METRICS:
                ds=np.asarray([100*(float(lookup[(scope,f,"M3_local_boundary_prototype")][m])-float(lookup[(scope,f,other)][m])) for f in range(1,11)])
                for fold_set, arr in (("all_10",ds),("exclude_fold2_9",np.delete(ds,1))):
                    n=len(arr); mean=float(np.mean(arr));sd=float(np.std(arr,ddof=1));half=float(t.ppf(.975,n-1)*sd/math.sqrt(n))
                    out.append({"scope":scope,"comparison":cmp,"metric":m,"fold_set":fold_set,"n_folds":n,
                                "mean_delta_pp":mean,"ci95_low_descriptive_pp":mean-half,"ci95_high_descriptive_pp":mean+half,
                                "wins":int(np.sum(arr>1e-12)),"ties":int(np.sum(np.abs(arr)<=1e-12)),"losses":int(np.sum(arr< -1e-12)),
                                "fold_deltas_pp":json.dumps(arr.tolist())})
    return out


def write_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fields=[]
    for row in rows:
        for k in row:
            if k not in fields:fields.append(k)
    tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def run_all():
    import numpy as np
    start=time.time(); (RESULTS).mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    source=inventory(); refs=verify_references();p,_,_=modules()
    sites,site_names,site_meta=p.site_data()
    outputs=[run_fold(f) for f in range(1,11)]
    fold_rows=[r for o in outputs for r in o["fold_metrics"]]
    summary=summarize_metrics(fold_rows);paired=paired_rows(fold_rows)
    corr=[r for o in outputs for r in o["correction"]]
    boundary=[r for o in outputs for r in o["boundary"]]
    diag=[o["diagnostics"] for o in outputs]
    split_rows=[r for o in outputs for r in o["split_rows"]]
    cal_rows=[r for o in outputs for r in o["calibrator_rows"]]
    # Save all prediction data required to independently recalculate every table.
    cache_obj={}
    for o in outputs:
        f=o["fold"];d=o["data"];cache_obj[f"f{f}_eligible_ids"]=o["inner_ids"];cache_obj[f"f{f}_test_ids"]=o["test_ids"]
        cache_obj[f"f{f}_y_all"]=d["y"];cache_obj[f"f{f}_h"]=d["h"];cache_obj[f"f{f}_prob_all"]=d["prob"]
        cache_obj[f"f{f}_pred_all"]=d["pred"];cache_obj[f"f{f}_stability"]=d["stability"];cache_obj[f"f{f}_sites"]=o["sites"]
        for prefix,preds,probs,scores in (("inner",o["inner_pred"],o["inner_prob"],o["inner_score"]),
                                          ("outer",o["outer_pred"],o["outer_prob"],o["outer_score"])):
            for g in GROUPS:
                cache_obj[f"f{f}_{prefix}_{g}_pred"]=preds[g]
                cache_obj[f"f{f}_{prefix}_{g}_prob_asd"]=probs[g]
                cache_obj[f"f{f}_{prefix}_{g}_score"]=scores[g]
        if o["outer_noexp"]:
            for g,v in o["outer_noexp"].items():
                cache_obj[f"f{f}_outer_noexp_{g}_ids"]=v["ids"]
                cache_obj[f"f{f}_outer_noexp_{g}_pred"]=v["pred"]
                cache_obj[f"f{f}_outer_noexp_{g}_prob"]=v["prob"]
    cache_path=CACHE/"fold_predictions_and_features.npz";np.savez_compressed(cache_path,**cache_obj)
    registry={"source_sha256":sha_json(source),"folds":[{"fold":o["fold"],"pairs":o["pair_registry"],
               "permutations":o["details"],"splits":o["split_rows"]} for o in outputs]}
    reg_path=CACHE/"prototype_registry.json";reg_path.write_text(json.dumps(registry,indent=2,allow_nan=False)+"\n")
    write_csv(RESULTS/"fold_metrics.csv",fold_rows);write_csv(RESULTS/"paired_comparisons.csv",paired)
    write_csv(RESULTS/"correction_analysis.csv",corr);write_csv(RESULTS/"boundary_diagnostics.csv",boundary)
    write_csv(RESULTS/"prototype_diagnostics.csv",diag);write_csv(RESULTS/"calibrator_diagnostics.csv",cal_rows)
    write_csv(RESULTS/"summary.csv",summary)
    preflight={"status":"PASS","references":refs,"source_sha256":sha_json(source),"folds":10,
               "frozen_best_checkpoint_extractions":10,"E74.4_outputlayer_input":"[871,144]","optimizer_updates":False,
               "base_prediction_cache_reproduced":True,"stability_cache":"E76.2 audited frozen perturbation stability",
               "fold2_overlap_excluded_from_new_support":[377,407],"inner_cv":"5-fold stratified random_state=10",
               "training_evidence":"3-fold crossfit random_state=76300+fold","calibrator":"residual logistic L-BFGS-B, L2=0.01",
               "M4_pair_permutations":5,"site_source_sha256":site_meta,"device_by_fold":[o["diagnostics"]["device"] for o in outputs]}
    (RESULTS/"preflight.json").write_text(json.dumps(preflight,indent=2,allow_nan=False)+"\n")
    manifest={"experiment":CFG["experiment"],"status":"COMPLETE_PENDING_AUDIT","folds":10,"groups":list(GROUPS),
              "source_sha256":sha_json(source),"source_inventory":source,"reference_commits":refs,
              "feature_cache_sha256":sha_file(cache_path),"prototype_registry_sha256":sha_file(reg_path),
              "runtime_seconds":time.time()-start,"metric_rows":len(fold_rows)}
    (EXP/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":"RUN_COMPLETE","folds":10,"metric_rows":len(fold_rows),
                      "seconds":time.time()-start,"source_sha256":manifest["source_sha256"]},indent=2))


if __name__=="__main__":
    run_all()
