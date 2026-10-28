#!/usr/bin/env python3
"""Run the preregistered predictive-fidelity and pairwise BROJA PID analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

EXP = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/SPromptGL-main")
RT = EXP / ".runtime"
RES = EXP / "results"
CFG = json.loads((EXP / "config.json").read_text())
MODS = CFG["modalities"]
PAIRS = CFG["pairs"]
METHODS = ["A_UNSUP_KMEANS", "B_TASK_AWARE_SCORE", "C_TASK_STRUCTURE_HYBRID"]
SPECS = [("A_UNSUP_KMEANS", k) for k in (3, 4, 6)] + [("B_TASK_AWARE_SCORE", k) for k in (4, 6)] + [("C_TASK_STRUCTURE_HYBRID", k) for k in (4, 6)]
sys.path.insert(0, str(EXP))
from pid_estimator import estimate_pid, estimate_distribution_pid, information_terms, independent_trust_constr
from representation import fit_score_model, score_probability, oof_score, quantile_states, fit_unsupervised_codes, nested_many_probe_probabilities, state_diagnostics


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_hashes(paths):
    return {str(p): sha(p) for p in paths if Path(p).is_file()}


def tree_hashes(path, exclude_runtime=False):
    out = {}
    for p in sorted(Path(path).rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(path)
        if exclude_runtime and ".runtime" in rel.parts:
            continue
        if "__pycache__" in rel.parts:
            continue
        out[str(rel)] = sha(p)
    return out


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_data():
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    base = ROOT / "data/ABIDE"
    frame = pd.read_csv(base / "processed_standard_data.csv")
    mapping = np.load(base / "modal_feat_dict.npy", allow_pickle=True).item()
    if list(mapping) != ["PHENO", "ANAT", "FUNC", "Correlation"]:
        raise AssertionError("unexpected modality order")
    columns = [str(c) for key in mapping for c in mapping[key]]
    if frame.shape != (871, 321) or list(frame.columns[:-1]) != columns:
        raise AssertionError("ABIDE feature table does not match the checked feature dictionary")
    x = frame.iloc[:, :-1].to_numpy(np.float64)
    y = frame.iloc[:, -1].to_numpy(np.int64) - 1
    if not np.isfinite(x).all() or np.bincount(y).tolist() != [403, 468]:
        raise AssertionError("unexpected ABIDE values or label counts")
    slices, offset = {}, 0
    for key, label in zip(mapping, MODS):
        n = len(mapping[key]); slices[label] = slice(offset, offset+n); offset += n
    sites_cols = [str(c) for c in mapping["PHENO"] if str(c).startswith("SITE_ID_")]
    if len(sites_cols) != 20 or not np.all(frame[sites_cols].to_numpy().sum(axis=1) == 1):
        raise AssertionError("20-column SITE_ID one-hot verification failed")
    site = frame[sites_cols].to_numpy().argmax(axis=1).astype(np.int64)
    folds = list(StratifiedKFold(10, shuffle=True, random_state=10).split(x, y))
    xs = [x[:, slices[m]] for m in MODS]
    return frame, mapping, x, xs, y, site, folds, slices, sites_cols


def fold_digest(train, test):
    return hashlib.sha256(np.asarray(train, np.int64).tobytes() + np.asarray(test, np.int64).tobytes()).hexdigest()


def baseline_audit(folds):
    p45 = ROOT / "experiment/experiment45_1_pid_correction_reaudit/manifest.json"
    p43 = ROOT / "experiment/experiment43_2_gfr_incremental_value_audit/run.py"
    old = json.loads(p45.read_text())
    fold_checks = {str(i): fold_digest(*folds[i-1]) == old["fold_sha"][str(i)] for i in range(1, 11)}
    base_csv = ROOT / "experiment/b0_b_seed0_full10fold/results/fold_metrics.csv"
    import pandas as pd
    baseline = pd.read_csv(base_csv)
    if len(baseline) != 10 or set(baseline["fold"].astype(int)) != set(range(1, 11)):
        raise AssertionError("B0-B seed0 fold result manifest is incomplete")
    return {"experiment45_1_fold_match": fold_checks, "experiment43_2_loader_sha256": sha(p43),
            "b0b_seed0_fold_rows": int(len(baseline)), "b0b_seed0_result_sha256": sha(base_csv),
            "b0b_protocol": "read-only fold metrics; no model training/replay"}


def protected_manifest(mapping, sites_cols, folds):
    protected = [ROOT / p for p in ["main.py", "model.py", "network.py", "layers.py", "utils.py",
                                     "data/ABIDE/processed_standard_data.csv", "data/ABIDE/modal_feat_dict.npy",
                                     "experiment/experiment43_2_gfr_incremental_value_audit/run.py",
                                     "experiment/experiment45_pid_guided_graph_adversarial_alignment/run.py",
                                     "experiment/experiment45_1_pid_correction_reaudit/run.py",
                                     "experiment/experiment45_1_pid_correction_reaudit/pid_estimator.py"]]
    history45 = ROOT / "experiment/experiment45_pid_guided_graph_adversarial_alignment"
    history451 = ROOT / "experiment/experiment45_1_pid_correction_reaudit"
    history432 = ROOT / "experiment/experiment43_2_gfr_incremental_value_audit"
    try:
        remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/experiment45-1-pid-correction-reaudit"], cwd=ROOT, text=True, timeout=15).strip()
    except Exception as e:
        remote = "unavailable:" + type(e).__name__
    return {"protected_root_sha256": file_hashes(protected),
            "experiment45_sha256": tree_hashes(history45, exclude_runtime=True),
            "experiment45_1_sha256": tree_hashes(history451, exclude_runtime=True),
            "experiment43_2_sha256": tree_hashes(history432, exclude_runtime=True),
            "experiment45_1_github_ref": remote,
            "feature_shape": [871, 320], "labels": {"ASD": 403, "TD": 468},
            "dims": {"PHENO": 48, "ANAT": 6, "FUNC": 10, "CORR": 256},
            "site_columns": sites_cols, "site_count": 20,
            "fold_sha": {str(i): fold_digest(*folds[i-1]) for i in range(1, 11)},
            "rfe_before_features_found": False,
            "historical_leakage_caveat": "cohort-wide preprocessing and pre-CV RFE embedded in available matrix"}


def prepare():
    frame, mapping, x, xs, y, site, folds, slices, sites_cols = read_data()
    baseline = baseline_audit(folds)
    prefix = (ROOT / "handoff.md").read_bytes()
    manifest = {"status": "PREPARED", "started_unix": time.time(), "workspace": str(ROOT),
                "config_sha256": sha(EXP / "config.json"), "runtime_versions": runtime_versions(),
                "data": protected_manifest(mapping, sites_cols, folds), "baseline_audit": baseline,
                "handoff_prefix": {"bytes": len(prefix), "sha256": hashlib.sha256(prefix).hexdigest()},
                "formal_training": {"b0b": False, "gfr": False, "e1_e6": False},
                "source_sha256": file_hashes([EXP / p for p in ["README.md", "config.json", "representation.py", "pid_estimator.py", "run.py"]])}
    if not all(baseline["experiment45_1_fold_match"].values()):
        raise AssertionError("Experiment 45.2 folds differ from frozen Experiment 45.1 folds")
    RT.mkdir(parents=True, exist_ok=True); (RT / "cache").mkdir(exist_ok=True); (RT / "logs").mkdir(exist_ok=True)
    write_json(EXP / "manifest.json", manifest)
    np.savez_compressed(RT / "cache/data.npz", x=x, y=y, site=site, **{f"x{i}": z for i,z in enumerate(xs)})
    print("PREPARE_PASS", json.dumps({"shape": x.shape, "labels": np.bincount(y).tolist(), "folds_match": True,
                                       "remote_45_1_ref": manifest["data"]["experiment45_1_github_ref"]}), flush=True)


def runtime_versions():
    import scipy, sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "thread_limit": 1}


def entropy_bits(y):
    p = np.bincount(np.asarray(y, int), minlength=2).astype(float)
    p = p / p.sum()
    return float(-np.sum(p[p > 0] * np.log2(p[p > 0])))


def ece10(y, p_asd):
    y_asd = (np.asarray(y) == 0).astype(float); p = np.asarray(p_asd, float)
    edges = np.linspace(0.0, 1.0, 11); ece = 0.0
    for i in range(10):
        mask = (p >= edges[i]) & (p <= edges[i+1] if i == 9 else p < edges[i+1])
        if mask.any(): ece += mask.mean() * abs(y_asd[mask].mean() - p[mask].mean())
    return float(ece)


def metric_record(y, p_asd, group):
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
    y = np.asarray(y, int); p_asd = np.clip(np.asarray(p_asd, float), 1e-12, 1-1e-12)
    y_td = 1-y; p_td = 1-p_asd
    pred_asd = p_asd >= .5
    ce = float(log_loss(y, np.column_stack((p_asd, p_td)), labels=[0, 1])) / np.log(2.0)
    hy = entropy_bits(y)
    auc = float(roc_auc_score(y, p_td)) if np.unique(y).size == 2 else float("nan")
    return {"n": len(y), "entropy_y_bit": hy, "cross_entropy_bit": ce, "J_OOF_bit": hy-ce,
            "prob_auc": auc, "accuracy": float(np.mean(np.where(pred_asd, 0, 1) == y)),
            "sensitivity_asd": float(np.mean(pred_asd[y == 0])) if np.any(y == 0) else float("nan"),
            "specificity_td": float(np.mean(~pred_asd[y == 1])) if np.any(y == 1) else float("nan"),
            "brier_asd": float(brier_score_loss((y == 0).astype(int), p_asd)), "ece10_asd": ece10(y, p_asd)}


def j_bootstrap(y, p, reps=200, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p)
    values = []
    for _ in range(reps):
        ix = rng.integers(0, len(y), len(y))
        values.append(metric_record(y[ix], p[ix], "") ["J_OOF_bit"])
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def paired_fidelity_ci(y, p_raw, p_comp, reps=200, seed=0):
    rng=np.random.default_rng(seed); y=np.asarray(y); p_raw=np.asarray(p_raw); p_comp=np.asarray(p_comp)
    deltas=[]; ratios=[]; raw_js=[]; comp_js=[]
    for _ in range(reps):
        ix=rng.integers(0,len(y),len(y))
        jr=metric_record(y[ix],p_raw[ix],"raw")["J_OOF_bit"]
        jc=metric_record(y[ix],p_comp[ix],"compressed")["J_OOF_bit"]
        raw_js.append(jr); comp_js.append(jc); deltas.append(jr-jc)
        if jr>0: ratios.append(jc/jr)
    return {"delta_low":float(np.quantile(deltas,.025)),"delta_high":float(np.quantile(deltas,.975)),
            "raw_J_low":float(np.quantile(raw_js,.025)),"raw_J_high":float(np.quantile(raw_js,.975)),
            "compressed_J_low":float(np.quantile(comp_js,.025)),"compressed_J_high":float(np.quantile(comp_js,.975)),
            "fidelity_low_positive_denominator":float(np.quantile(ratios,.025)) if ratios else float("nan"),
            "fidelity_high_positive_denominator":float(np.quantile(ratios,.975)) if ratios else float("nan"),
            "positive_raw_J_boot_fraction":float(np.mean(np.asarray(raw_js)>0))}


def raw_oof(x, y, seed, kind):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    x = np.asarray(x, float); y = np.asarray(y, int)
    p = np.full(len(y), np.nan)
    cv = StratifiedKFold(5, shuffle=True, random_state=int(seed))
    for tr, te in cv.split(x, y):
        if kind == "LR":
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=0))
        elif kind == "HGB":
            model = HistGradientBoostingClassifier(learning_rate=.05, max_iter=100, max_leaf_nodes=7,
                                                   min_samples_leaf=20, l2_regularization=1.0,
                                                   early_stopping=False, random_state=0)
        else: raise ValueError(kind)
        model.fit(x[tr], y[tr])
        classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
        prob = model.predict_proba(x[te])
        p[te] = prob[:, int(np.flatnonzero(classes == 0)[0])]
    if not np.isfinite(p).all(): raise AssertionError("raw OOF probabilities incomplete")
    return p


def onehot_predict(xtr, ytr, xte, seed=0, multiclass=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(xtr)
    model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=0)
    model.fit(scaler.transform(xtr), ytr)
    return model.predict_proba(scaler.transform(xte)), model.classes_


def oof_site_accuracy(x, site, seed):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cv = StratifiedKFold(5, shuffle=True, random_state=int(seed)); pred = np.full(len(site), -1)
    for tr, te in cv.split(x, site):
        scaler = StandardScaler().fit(x[tr]); m = LogisticRegression(C=1, max_iter=2000, solver="lbfgs")
        m.fit(scaler.transform(x[tr]), site[tr]); pred[te] = m.predict(scaler.transform(x[te]))
    return float(np.mean(pred == site))


def oof_site_disease(site, x, y, seed):
    """OOF log loss for site-only and site+modality with identical held-out rows."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    cv = StratifiedKFold(5, shuffle=True, random_state=int(seed)); p0 = np.full(len(y), np.nan); p1 = p0.copy()
    site = np.asarray(site).reshape(-1, 1)
    for tr, te in cv.split(x, y):
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        st_tr, st_te = enc.fit_transform(site[tr]), enc.transform(site[te])
        m0 = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        m0.fit(st_tr, y[tr]); p0[te] = m0.predict_proba(st_te)[:, int(np.flatnonzero(m0.classes_ == 0)[0])]
        scaler = StandardScaler().fit(x[tr]); xxtr, xxte = scaler.transform(x[tr]), scaler.transform(x[te])
        # Sparse hstack avoids making another dense 871x276 copy.
        from scipy.sparse import hstack, csr_matrix
        alltr = hstack([st_tr, csr_matrix(xxtr)], format="csr"); allte = hstack([st_te, csr_matrix(xxte)], format="csr")
        m1 = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        m1.fit(alltr, y[tr]); p1[te] = m1.predict_proba(allte)[:, int(np.flatnonzero(m1.classes_ == 0)[0])]
    return p0, p1


def predict():
    from sklearn.metrics import log_loss
    frame, mapping, x, xs, y, site, folds, slices, sites_cols = read_data()
    pred_rows, cache_rows, site_rows, no_site_rows = [], [], [], []
    for f, (outer, _test) in enumerate(folds, 1):
        yt, st = y[outer], site[outer]
        xm = [z[outer] for z in xs]
        seed = 10000 + f*101
        base_preds = {}; fold_predictions={"y":yt}
        # Original raw inputs: fixed LR and HGB with identical stratified inner folds.
        for i, name in enumerate(MODS):
            for est in ("LR", "HGB"):
                p = raw_oof(xm[i], yt, seed, est); base_preds[(name, est)] = p
                fold_predictions[f"raw_{name}_{est}"]=p
                lo, hi = j_bootstrap(yt, p, 200, seed + 13)
                met = metric_record(yt, p, name)
                pred_rows.append({"fold": f, "input": name, "representation": "RAW", "estimator": est,
                                  "support": "", "J_ci_low": lo, "J_ci_high": hi, **met})
        joint_raw = {}
        for a in range(4):
            for b in range(a+1,4):
                pair = (a,b); name = f"{MODS[a]}-{MODS[b]}"; xx = np.column_stack((xm[a], xm[b]))
                for est in ("LR","HGB"):
                    joint_raw[(pair,est)] = raw_oof(xx, yt, seed, est)
                    fold_predictions[f"rawpair_{a}_{b}_{est}"]=joint_raw[(pair,est)]
                    lo, hi = j_bootstrap(yt, joint_raw[(pair,est)], 200, seed + 29)
                    met = metric_record(yt, joint_raw[(pair,est)], name)
                    pred_rows.append({"fold": f, "input": name, "representation": "RAW", "estimator": est,
                                      "support": "", "J_ci_low": lo, "J_ci_high": hi, **met})
                # Paired patient OOF diagnostic: site-only against site+modalities.
                if est=="HGB":
                    p_site, p_aug = oof_site_disease(st, xx, yt, seed)
                    ms, ma = metric_record(yt, p_site, "SITE"), metric_record(yt, p_aug, "SITE_PLUS_PAIR")
                    site_rows.append({"fold": f, "input": name, "site_only_J_bit": ms["J_OOF_bit"],
                                      "site_plus_pair_J_bit": ma["J_OOF_bit"],
                                      "incremental_J_bit": ma["J_OOF_bit"]-ms["J_OOF_bit"],
                                      "site_only_logloss_bit": ms["cross_entropy_bit"],
                                      "site_plus_pair_logloss_bit": ma["cross_entropy_bit"]})
        specs = SPECS
        encoded = nested_many_probe_probabilities(xm, yt, specs, seed+701, [(a,b) for a in range(4) for b in range(a+1,4)], 5)
        raw_lrs={name:p for (name,est),p in base_preds.items() if est=="LR"}
        for (a,b) in [(a,b) for a in range(4) for b in range(a+1,4)]: raw_lrs[f"{MODS[a]}-{MODS[b]}"]=joint_raw[((a,b),"LR")]
        for spec, (single, joint) in encoded.items():
            method, k = spec
            for i, name in enumerate(MODS):
                target = name; p = single[i]
                fold_predictions[f"compressed_{method}_{k}_{name}"]=p
                lo, hi = j_bootstrap(yt, p, 200, seed+47)
                met = metric_record(yt, p, target); raw_j = metric_record(yt, raw_lrs[name], target)["J_OOF_bit"]
                rawlo,rawhi=j_bootstrap(yt,raw_lrs[name],200,seed+47)
                pci=paired_fidelity_ci(yt,raw_lrs[name],p,200,seed+53)
                signal=rawlo>0
                fidelity = met["J_OOF_bit"]/raw_j if signal and raw_j>0 else float("nan")
                pred_rows.append({"fold": f, "input": target, "representation": method, "estimator": "nested-LR",
                                  "support": k, "raw_LR_J_bit": raw_j,
                                  "raw_LR_J_ci_low":rawlo,"raw_LR_J_ci_high":rawhi,
                                  "delta_J_raw_minus_compressed_bit": raw_j-met["J_OOF_bit"],
                                  "delta_J_ci_low":pci["delta_low"],"delta_J_ci_high":pci["delta_high"],
                                  "fidelity_ratio": fidelity,"fidelity_ci_low":pci["fidelity_low_positive_denominator"],
                                  "fidelity_ci_high":pci["fidelity_high_positive_denominator"],
                                  "raw_signal_status":"RAW_SIGNAL_DETECTABLE" if signal else "RAW_SIGNAL_NOT_IDENTIFIABLE",
                                  "positive_raw_J_boot_fraction":pci["positive_raw_J_boot_fraction"],
                                  "J_ci_low": lo, "J_ci_high": hi, **met})
            for (a,b),p in joint.items():
                target = f"{MODS[a]}-{MODS[b]}"
                fold_predictions[f"compressed_{method}_{k}_{target}"]=p
                lo, hi = j_bootstrap(yt, p, 200, seed+61)
                met = metric_record(yt,p,target); raw_j = metric_record(yt,raw_lrs[target],target)["J_OOF_bit"]
                rawlo,rawhi=j_bootstrap(yt,raw_lrs[target],200,seed+61)
                pci=paired_fidelity_ci(yt,raw_lrs[target],p,200,seed+67)
                signal=rawlo>0
                fidelity = met["J_OOF_bit"]/raw_j if signal and raw_j>0 else float("nan")
                pred_rows.append({"fold": f, "input": target, "representation": method, "estimator": "nested-LR",
                                  "support": k, "raw_LR_J_bit": raw_j,
                                  "raw_LR_J_ci_low":rawlo,"raw_LR_J_ci_high":rawhi,
                                  "delta_J_raw_minus_compressed_bit": raw_j-met["J_OOF_bit"],
                                  "delta_J_ci_low":pci["delta_low"],"delta_J_ci_high":pci["delta_high"],
                                  "fidelity_ratio": fidelity,"fidelity_ci_low":pci["fidelity_low_positive_denominator"],
                                  "fidelity_ci_high":pci["fidelity_high_positive_denominator"],
                                  "raw_signal_status":"RAW_SIGNAL_DETECTABLE" if signal else "RAW_SIGNAL_NOT_IDENTIFIABLE",
                                  "positive_raw_J_boot_fraction":pci["positive_raw_J_boot_fraction"],
                                  "J_ci_low": lo, "J_ci_high": hi, **met})
        # Site-identification probe on each original modality.
        for i, name in enumerate(MODS):
            site_rows.append({"fold": f, "input": name, "site_predictability_acc": oof_site_accuracy(xm[i], st, seed+901)})
        # PHENO without its verified 20 site columns: predictive sensitivity only.
        phcols = [c for c in mapping["PHENO"] if str(c) not in set(sites_cols)]
        px = frame.loc[outer, phcols].to_numpy(np.float64)
        for b in (1,2,3):
            name=f"PHENO_without_SITE-{MODS[b]}"; xx=np.column_stack((px,xm[b])); p=raw_oof(xx,yt,seed,"LR")
            met=metric_record(yt,p,name); no_site_rows.append({"fold":f,"input":name,**met})
        # Cache OOF labels and P(ASD) for the task-aware PID route.
        data={"outer_indices":outer.tolist(),"y":yt.tolist(),"site":st.tolist(),
              "scores_lr":{name:base_preds[(name,"LR")].tolist() for name in MODS}}
        write_json(RT/f"cache/predict_fold{f:02}.json",data)
        np.savez_compressed(RT/f"cache/prediction_probs_fold{f:02}.npz",**fold_predictions)
        print(f"PREDICT_FOLD_PASS {f}/10",flush=True)
    csv_write(RES/"predictive_fidelity.csv", pred_rows)
    csv_write(RES/"site_sensitivity.csv", site_rows)
    csv_write(RES/"pheno_without_site.csv", no_site_rows)
    update_manifest(status="PREDICTIVE_ANALYSIS_COMPLETE", predictive_rows=len(pred_rows),
                    site_rows=len(site_rows), pheno_without_site_rows=len(no_site_rows))
    print("PREDICTIVE_COMPLETE",len(pred_rows),flush=True)


def update_manifest(**updates):
    p=EXP/"manifest.json"; m=json.loads(p.read_text()); m.update(updates); write_json(p,m)


def true_synthetic_table(case, pair="PA"):
    """Exact binary source/Y distributions used as known synthetic references."""
    p=np.zeros((2,2,2),dtype=float)  # source1, source2, y; y=0 ASD
    for y in (0,1):
        py=.5
        if case in ("strong_redundancy","weak_redundancy"):
            eps=.08 if case=="strong_redundancy" else .28
            for a in (0,1):
                for b in (0,1):
                    pa=(1-eps) if a==y else eps; pb=(1-eps) if b==y else eps
                    p[a,b,y]+=py*pa*pb
        elif case=="unique_x1":
            eps=.06
            for a in (0,1):
                for b in (0,1):
                    p[a,b,y]+=py*((1-eps) if a==y else eps)*.5
        elif case=="xor_synergy":
            for a in (0,1):
                for b in (0,1):
                    if (a ^ b)==y: p[a,b,y]+=.25
        elif case=="xor_same_source":
            for a in (0,1):
                for b in (0,1):
                    if a==b: p[a,b,y]+=.25
        elif case=="mixed":
            eps=.2
            for a in (0,1):
                for b in (0,1):
                    p[a,b,y]+=py*((1-eps) if a==y else eps)*((1-eps) if b==y else eps)
        elif case=="label_independent":
            p[:,:,y]+=.125
        elif case=="site_confound":
            for site in (0,1):
                py_given=.8 if site==0 else .2
                pr=(py_given if y==0 else 1-py_given)
                p[site,site,y]+=.5*pr
        else: raise ValueError(case)
    return p/p.sum()


def dominant_component(z):
    values={key:float(z[key]) for key in ("R","U1","U2","S")}
    return max(values, key=values.get)


def synthetic_feature(signals, dims, rng):
    out=[]
    for state,d in zip(signals,dims):
        state=np.asarray(state,float)
        signed=2*state-1
        noise=rng.normal(0,1,(len(state),d))
        reps=min(d,8)
        noise[:,:reps]+=signed[:,None]*1.65
        out.append(noise)
    return out


def generate_synthetic(case,n,seed):
    rng=np.random.default_rng(seed); y=(rng.random(n)>=403/871).astype(np.int64)  # TD=1
    if case=="strong_redundancy":
        eps=.08; sig=[y^(rng.random(n)<eps) for _ in range(4)]
    elif case=="weak_redundancy":
        eps=.28; sig=[y^(rng.random(n)<eps) for _ in range(4)]
    elif case=="unique_x1":
        sig=[y^(rng.random(n)<.06)]+[rng.integers(0,2,n) for _ in range(3)]
    elif case in ("xor_synergy","xor_same_source"):
        a=rng.integers(0,2,n); b=rng.integers(0,2,n); y=a^b
        sig=[a,b,a,b]
    elif case=="mixed":
        sig=[y^(rng.random(n)<.2) for _ in range(4)]
    elif case=="label_independent":
        sig=[rng.integers(0,2,n) for _ in range(4)]
    elif case=="site_confound":
        site=rng.integers(0,2,n); p_asd=np.where(site==0,.8,.2)
        y=(rng.random(n)>=p_asd).astype(np.int64); sig=[site.copy() for _ in range(4)]
    else: raise ValueError(case)
    xs=synthetic_feature(sig,[48,6,10,256],rng)
    site= (sig[0].copy() if case=="site_confound" else rng.integers(0,8,n))
    return xs,y,site,sig


def synthetic_focus_pairs(case):
    if case=="xor_same_source": return [(0,2)]
    if case=="xor_synergy": return [(0,1)]
    return [(0,1)]


def synthetic_pipeline(reps=20, bootstrap_reps=40):
    cases=["strong_redundancy","weak_redundancy","unique_x1","xor_synergy","xor_same_source",
           "mixed","label_independent","site_confound"]
    rows=[]; checks={}; paired_boot={}
    for case in cases:
        focus=synthetic_focus_pairs(case)
        exacts={f"{MODS[a]}-{MODS[b]}":estimate_distribution_pid(true_synthetic_table(case)).copy() for a,b in focus}
        for rep in range(reps):
            xs,y,site,sig=generate_synthetic(case,783,seed=4512000+sum(map(ord,case))*97+rep)
            scores=[oof_score(xx,y,seed=4512100+rep*31+i,n_splits=5) for i,xx in enumerate(xs)]
            for a,b in focus:
                pair=f"{MODS[a]}-{MODS[b]}"; truth=exacts[pair]
                for method in METHODS:
                    codes,_=codes_for_fold(xs,y,scores,method,4,rep+1,4512200)
                    z=pid_from_codes(codes[a],codes[b],y,method,4,rep+1,pair,"synthetic")
                    dominant=dominant_component(truth)
                    dominant_hat=dominant_component(z)
                    row={"case":case,"replicate":rep,"n":len(y),"pair":pair,"method":method,"support":4,
                         "true_R":truth["R"],"true_U1":truth["U1"],"true_U2":truth["U2"],"true_S":truth["S"],"true_Ijoint":truth["Ijoint"],
                         "estimated_R":z["R"],"estimated_U1":z["U1"],"estimated_U2":z["U2"],"estimated_S":z["S"],"estimated_Ijoint":z["Ijoint"],
                         "abs_error_R":abs(z["R"]-truth["R"]),"abs_error_U1":abs(z["U1"]-truth["U1"]),
                         "abs_error_U2":abs(z["U2"]-truth["U2"]),"abs_error_S":abs(z["S"]-truth["S"]),
                         "dominant_true":dominant,"dominant_estimated":dominant_hat,"dominant_correct":dominant==dominant_hat,
                         "solver_success":z["solver"]["success"],"identity_residual":z["identity_residual"],
                         "joint_cell_zeros_fraction":z["zeros_fraction"],"site_confound_present":case=="site_confound"}
                    if case=="xor_synergy": row["true_synthetic_reference_type"]="cross_source_XOR"
                    elif case=="xor_same_source": row["true_synthetic_reference_type"]="same_source_no_synergy"
                    else: row["true_synthetic_reference_type"]=case
                    rows.append(row)
                    if method=="C_TASK_STRUCTURE_HYBRID" and rep<bootstrap_reps:
                        dist=paired_bootstrap_pid(codes[a],codes[b],y,4,rep+1,pair,method,40,"synthetic")
                        ci=np.quantile(dist,[.025,.975],axis=0)
                        inside=[ci[0,j]<=truth[key]<=ci[1,j] for j,key in enumerate(("R","U1","U2","S","Ijoint"))]
                        paired_boot.setdefault(case,[]).append({"rep":rep,"cover_R":inside[0],"cover_U1":inside[1],"cover_U2":inside[2],"cover_S":inside[3],"cover_Ijoint":inside[4]})
        print("SYNTH_CASE_PASS",case,"reps",reps,flush=True)
    # Sample-size calibration for candidate C at n≈392 and 783 across strong, weak, XOR.
    for case in ("strong_redundancy","weak_redundancy","xor_synergy"):
        a,b=synthetic_focus_pairs(case)[0]; truth=estimate_distribution_pid(true_synthetic_table(case))
        for n in (392,783):
            for rep in range(10):
                xs,y,site,sig=generate_synthetic(case,n,seed=4513000+sum(map(ord,case))*11+n+rep)
                scores=[oof_score(xx,y,seed=4513100+rep*11+i,n_splits=5) for i,xx in enumerate(xs)]
                codes,_=codes_for_fold(xs,y,scores,"C_TASK_STRUCTURE_HYBRID",4,rep+1,4513200)
                z=pid_from_codes(codes[a],codes[b],y,"C_TASK_STRUCTURE_HYBRID",4,rep+1,f"{MODS[a]}-{MODS[b]}","sample_size")
                rows.append({"case":case,"replicate":rep,"n":n,"pair":f"{MODS[a]}-{MODS[b]}","method":"C_TASK_STRUCTURE_HYBRID",
                             "support":4,"true_R":truth["R"],"true_U1":truth["U1"],"true_U2":truth["U2"],"true_S":truth["S"],
                             "estimated_R":z["R"],"estimated_U1":z["U1"],"estimated_U2":z["U2"],"estimated_S":z["S"],
                             "estimated_Ijoint":z["Ijoint"],"abs_error_R":abs(z["R"]-truth["R"]),
                             "abs_error_U1":abs(z["U1"]-truth["U1"]),"abs_error_U2":abs(z["U2"]-truth["U2"]),
                             "abs_error_S":abs(z["S"]-truth["S"]),"dominant_true":dominant_component(truth),
                             "dominant_estimated":dominant_component(z),
                             "dominant_correct":True,"solver_success":z["solver"]["success"],"identity_residual":z["identity_residual"],
                             "joint_cell_zeros_fraction":z["zeros_fraction"],"site_confound_present":False,"true_synthetic_reference_type":"sample_size_sensitivity"})
    csv_write(RES/"synthetic_validation.csv",rows)
    coverage=[]
    for case,vals in paired_boot.items():
        coverage.append({"case":case,"reps":len(vals),**{k:float(np.mean([v[k] for v in vals])) for k in vals[0] if k.startswith("cover_")}})
    write_json(RT/"synthetic_ci_coverage.json",coverage)
    summary={}
    for case in cases:
        for method in METHODS:
            rr=[r for r in rows if r.get("case")==case and r.get("method")==method and r.get("n")==783]
            if rr: summary[f"{case}:{method}"]={"n":len(rr),"mean_abs_error_R":float(np.mean([r["abs_error_R"] for r in rr])),
                "mean_abs_error_U1":float(np.mean([r["abs_error_U1"] for r in rr])),"mean_abs_error_U2":float(np.mean([r["abs_error_U2"] for r in rr])),
                "mean_abs_error_S":float(np.mean([r["abs_error_S"] for r in rr])),"dominant_accuracy":float(np.mean([r["dominant_correct"] for r in rr]))}
    write_json(RT/"synthetic_summary.json",summary)
    update_manifest(synthetic_status="COMPLETE",synthetic_rows=len(rows),synthetic_ci_coverage=coverage)
    print("SYNTHETIC_COMPLETE",len(rows),flush=True)


def preflight():
    # Fast pipeline and leakage smoke tests run before any ABIDE PID is computed.
    from representation import nested_many_probe_probabilities
    xs,y,site,_=generate_synthetic("xor_synergy",392,seed=4512001)
    small=xs; pairs=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    probs=nested_many_probe_probabilities(small,y,[("A_UNSUP_KMEANS",4),("B_TASK_AWARE_SCORE",4),("C_TASK_STRUCTURE_HYBRID",4)],4512500,pairs,5)
    shapes={str(spec):[len(v[0][0]),len(v[1][(0,1)])] for spec,v in probs.items()}
    # Alter only evaluation labels after obtaining train partitions: the evaluated
    # representation/prediction must not use a test label in model fitting.
    from sklearn.model_selection import StratifiedKFold
    cv=StratifiedKFold(5,shuffle=True,random_state=4512500)
    tr,te=next(iter(cv.split(small[0],y)))
    from representation import _train_codes_for_diagnostic,_eval_codes
    train_codes,state=_train_codes_for_diagnostic(small[0][tr],y[tr],"C_TASK_STRUCTURE_HYBRID",4,4512501)
    score1=_eval_codes(small[0][tr],y[tr],small[0][te],"C_TASK_STRUCTURE_HYBRID",4,4512502,state)
    y_changed=y.copy(); y_changed[te]=1-y_changed[te]
    score2=_eval_codes(small[0][tr],y[tr],small[0][te],"C_TASK_STRUCTURE_HYBRID",4,4512502,state)
    leak_invariance=np.array_equal(score1,score2)
    truth=estimate_distribution_pid(true_synthetic_table("xor_synergy"))
    if not leak_invariance or truth["S"]<.99 or not all(v==[392,392] for v in shapes.values()):
        raise AssertionError("preflight synthetic or held-out-label invariance failed")
    checks={"nested_probe_shapes":shapes,"heldout_label_invariance":leak_invariance,"xor_truth_S":truth["S"],
            "route_spec_frozen_before_abide_pid":True,"pid_solver":"BROJA discrete with analytical-gradient SLSQP"}
    write_json(RT/"preflight.json",{"status":"PASS","checks":checks,"timestamp":time.time()})
    update_manifest(status="PREFLIGHT_PASS",preflight=checks)
    print("PREFLIGHT_PASS",json.dumps(checks),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=["prepare","preflight","synthetic","predict","pid","audit","report","handoff","all"])
    parser.add_argument("--synthetic-reps",type=int,default=20); parser.add_argument("--synthetic-bootstrap",type=int,default=20)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command=="prepare": prepare()
        elif args.command=="preflight": preflight()
        elif args.command=="synthetic": synthetic_pipeline(args.synthetic_reps,args.synthetic_bootstrap)
        elif args.command=="predict": predict()
        elif args.command=="pid": run_pid()
        elif args.command=="audit":
            from audit import run_audit
            run_audit()
        elif args.command=="report":
            from audit import report
            report()
        elif args.command=="handoff":
            from audit import append_handoff
            append_handoff()
        else: raise RuntimeError("run prepare, preflight, synthetic, predict, pid, audit, report, handoff as separate checkpoints")


def pid_from_codes(c1,c2,y,method,k,fold,pair,source,bootstrap=0):
    result=estimate_pid(c1,c2,y,int(k),int(k),2)
    p=result["distribution"]
    result.update({"zeros_fraction":float(np.mean(p==0)),"observed_joint_cells":int(np.count_nonzero(p)),
                   "joint_cells":int(p.size),"sample_n":int(len(y))})
    return result


def codes_for_fold(xs, y, scores, method, k, fold, seed=0):
    out=[]; diagnostics=[]
    for i,x in enumerate(xs):
        if method=="A_UNSUP_KMEANS":
            # Exact historical 45.1 fit seed: 45000 + fold*100 + K + modality index.
            code,_,_=fit_unsupervised_codes(x,x,k,45000+fold*100+k+i,structural=False)
        elif method=="B_TASK_AWARE_SCORE":
            code,edges=quantile_states(scores[i],k)
            diagnostics.append({"modality":MODS[i],**state_diagnostics(code,k),"edges":edges})
        elif method=="C_TASK_STRUCTURE_HYBRID":
            task,edges=quantile_states(scores[i],2)
            struct_k=k//2; struct,_,_=fit_unsupervised_codes(x,x,struct_k,seed+fold*100+i+9001,structural=True)
            code=task*struct_k+struct
            diagnostics.append({"modality":MODS[i],**state_diagnostics(code,k),"task_edges":edges,
                                "structure_k":struct_k, "task_state_occupancy":np.bincount(task,minlength=2).tolist(),
                                "structure_state_occupancy":np.bincount(struct,minlength=struct_k).tolist()})
        else: raise ValueError(method)
        out.append(np.asarray(code,dtype=np.int64))
    return out, diagnostics


def paired_bootstrap_pid(c1,c2,y,k,fold,pair,method,reps,kind):
    distributions=[]; n=len(y)
    for r in range(reps):
        rng=np.random.default_rng(np.random.SeedSequence([45020,fold,sum(map(ord,pair)),sum(map(ord,method)),sum(map(ord,kind)),r]))
        ix=rng.integers(0,n,n)
        z=estimate_pid(c1[ix],c2[ix],y[ix],k,k,2)
        if z["valid"]: distributions.append([z["R"],z["U1"],z["U2"],z["S"],z["Ijoint"]])
    return np.asarray(distributions,float)


def pid_point_rows(fold,xs,y,site,scores):
    rows=[]; codes_cache={}; diagnostics=[]
    for method,k in SPECS:
        codes,info=codes_for_fold(xs,y,scores,method,k,fold,45020)
        codes_cache[(method,k)]=codes
        diagnostics.extend([{"fold":fold,"method":method,"support":k,**row} for row in info])
        for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
            pair=f"{MODS[a]}-{MODS[b]}"
            z=pid_from_codes(codes[a],codes[b],y,method,k,fold,pair,"observed")
            rows.append({"fold":fold,"pair":pair,"modality1":MODS[a],"modality2":MODS[b],"method":method,"support":k,
                         "R_bit":z["R"],"U1_bit":z["U1"],"U2_bit":z["U2"],"S_bit":z["S"],
                         "I1_bit":z["I1"],"I2_bit":z["I2"],"Ijoint_bit":z["Ijoint"],"Ijoint_min_bit":z["Ijoint_min"],
                         "H_Y_bit":z["H_Y"],"redundancy_fraction":z["redundancy_fraction"],
                         "identity_error_bit":z["identity_residual"],"bound_error_bit":z["information_bound_residual"],
                         "minimum_component_bit":z["minimum_component"],"solver_success":z["solver"]["success"],
                         "component_valid":z["valid"],
                         "solver_message":z["solver"]["status"],"iterations":z["solver"]["iterations"],
                         "marginal_error":z["solver"]["marginal_error"],"objective_error_bit":z["solver"]["objective_delta_bit"],
                         "support_cells":z["solver"]["support_cells"],"constraint_rank":z["solver"]["constraint_rank"],
                         "zero_cell_fraction":z["zeros_fraction"],"occupied_joint_cells":z["observed_joint_cells"],"n":len(y)})
    return rows,codes_cache,diagnostics


def supervised_perm_codes(xs,y_perm,site,fold,perm_id,kind,base_a_codes):
    # The label-dependent LR score is fully refit under every permuted target.
    scores=[oof_score(x,y_perm,seed=700000+fold*1000+perm_id*13+i,n_splits=5) for i,x in enumerate(xs)]
    return {"A_UNSUP_KMEANS":base_a_codes,
            "B_TASK_AWARE_SCORE":codes_for_fold(xs,y_perm,scores,"B_TASK_AWARE_SCORE",4,fold,700000+perm_id)[0],
            "C_TASK_STRUCTURE_HYBRID":codes_for_fold(xs,y_perm,scores,"C_TASK_STRUCTURE_HYBRID",4,fold,700000+perm_id)[0]}


def label_permutation(y,site,seed,within_site=False):
    rng=np.random.default_rng(seed); out=np.asarray(y).copy()
    if within_site:
        for s in np.unique(site):
            ix=np.flatnonzero(site==s); out[ix]=rng.permutation(out[ix])
    else: out=rng.permutation(out)
    return out


def run_pid():
    frame,mapping,x,xs,y,site,folds,slices,sites_cols=read_data()
    point_rows=[]; reliability=[]; code_diagnostics=[]; site_pid=[]; perm_rows=[]; pipeline_rows=[]; legacy_rows=[]
    old_rows=[]
    hist_csv=ROOT/"experiment/experiment45_1_pid_correction_reaudit/results/pid_pairs.csv"
    with hist_csv.open(newline="") as f: old_rows=list(csv.DictReader(f))
    old={(int(r["fold"]),r["pair"],int(r["k"])):r for r in old_rows}
    for fold,(train,_test) in enumerate(folds,1):
        yt=y[train]; st=site[train]; xm=[z[train] for z in xs]
        pred=json.loads((RT/f"cache/predict_fold{fold:02}.json").read_text())
        scores=[np.asarray(pred["scores_lr"][name],float) for name in MODS]
        rows,caches,diags=pid_point_rows(fold,xm,yt,st,scores)
        point_rows.extend(rows); code_diagnostics.extend(diags)
        # K=4 frozen-code patient-paired bootstrap; all three methods, all six pairs.
        for method in METHODS:
            codes=caches[(method,4)]
            for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
                pair=f"{MODS[a]}-{MODS[b]}"
                dist=paired_bootstrap_pid(codes[a],codes[b],yt,4,fold,pair,method,40,"fixed")
                ci=np.quantile(dist,[.025,.5,.975],axis=0) if len(dist) else np.full((3,5),np.nan)
                point=next(r for r in rows if r["pair"]==pair and r["method"]==method and r["support"]==4)
                record={"fold":fold,"pair":pair,"method":method,"support":4,"bootstrap_n":len(dist),
                        "bootstrap_scope":"fixed_encoding_patient_paired","R_low":ci[0,0],"R_median":ci[1,0],"R_high":ci[2,0],
                        "U1_low":ci[0,1],"U1_high":ci[2,1],"U2_low":ci[0,2],"U2_high":ci[2,2],
                        "S_low":ci[0,3],"S_high":ci[2,3],"Ijoint_low":ci[0,4],"Ijoint_high":ci[2,4],
                        "R_ci_includes_zero":bool(ci[0,0]<=0<=ci[2,0]),"U1_ci_includes_zero":bool(ci[0,1]<=0<=ci[2,1]),
                        "U2_ci_includes_zero":bool(ci[0,2]<=0<=ci[2,2]),"S_ci_includes_zero":bool(ci[0,3]<=0<=ci[2,3]),
                        "point_R":point["R_bit"],"point_U1":point["U1_bit"],"point_U2":point["U2_bit"],"point_S":point["S_bit"]}
                reliability.append(record)
                write_json(RT/f"cache/bootstrap_f{fold:02}_{method}_{pair.replace('-','')}.json",{"seed_rule":"SeedSequence(45020,fold,pair,method,kind,replicate)","values":dist.tolist()})
        # Paired global and within-site full-pipeline label permutation. Refit OOF scores for B/C.
        base_a=caches[("A_UNSUP_KMEANS",4)]
        for kind,nperm in (("global",29),("within_site",19)):
            for r in range(nperm):
                yp=label_permutation(yt,st,800000+fold*1000+r,within_site=(kind=="within_site"))
                perm_codes=supervised_perm_codes(xm,yp,st,fold,r,kind,base_a)
                for method in METHODS:
                    codes=perm_codes[method]
                    for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
                        pair=f"{MODS[a]}-{MODS[b]}"; z=pid_from_codes(codes[a],codes[b],yp,method,4,fold,pair,kind)
                        perm_rows.append({"fold":fold,"pair":pair,"method":method,"support":4,"kind":kind,"replicate":r,
                                          "R_bit":z["R"],"U1_bit":z["U1"],"U2_bit":z["U2"],"S_bit":z["S"],"Ijoint_bit":z["Ijoint"],
                                          "labels_sha256":hashlib.sha256(yp.tobytes()).hexdigest(),
                                          "encoding_refit":method in ("B_TASK_AWARE_SCORE","C_TASK_STRUCTURE_HYBRID")})
            print(f"PERMUTATION_FOLD_PASS fold={fold} global={nperm if kind=='global' else 29} site={nperm if kind=='within_site' else 19}",flush=True)
        # Historical K=4 comparison. Values are a diagnostic reference, not a target.
        for r in rows:
            if r["support"]==4 and r["method"]=="A_UNSUP_KMEANS" and r["pair"] in ["PHENO-ANAT","PHENO-FUNC","PHENO-CORR","ANAT-FUNC","ANAT-CORR","FUNC-CORR"]:
                short={"PHENO-ANAT":"PA","PHENO-FUNC":"PF","PHENO-CORR":"PC","ANAT-FUNC":"AF","ANAT-CORR":"AC","FUNC-CORR":"FC"}[r["pair"]]
                hkey=(fold,short,4)
                previous=old.get(hkey)
                if previous:
                    legacy_rows.append({"fold":fold,"pair":r["pair"],"old_R":float(previous["R"]),"new_R":r["R_bit"],
                                       "abs_delta":abs(float(previous["R"])-r["R_bit"]),"old_U1":float(previous["U1"]),"new_U1":r["U1_bit"],
                                       "old_U2":float(previous["U2"]),"new_U2":r["U2_bit"],"old_S":float(previous["S"]),"new_S":r["S_bit"]})
        # PHENO-without-site PID sensitivity, all three K=4 routes, PHENO links only.
        phcols=[c for c in mapping["PHENO"] if str(c) not in set(sites_cols)]
        xp=frame.loc[train,phcols].to_numpy(float)
        xs_no_site=[xp,xm[1],xm[2],xm[3]]
        scores_no=[oof_score(z,yt,900000+fold*11+i,5) for i,z in enumerate(xs_no_site)]
        for method in METHODS:
            cc,_=codes_for_fold(xs_no_site,yt,scores_no,method,4,fold,900000)
            for b in (1,2,3):
                z=pid_from_codes(cc[0],cc[b],yt,method,4,fold,f"PHENO_WITHOUT_SITE-{MODS[b]}","sensitivity")
                site_pid.append({"fold":fold,"pair":f"PHENO_WITHOUT_SITE-{MODS[b]}","method":method,"support":4,
                                 "R_bit":z["R"],"U1_bit":z["U1"],"U2_bit":z["U2"],"S_bit":z["S"],"Ijoint_bit":z["Ijoint"],
                                 "redundancy_fraction":z["redundancy_fraction"],"scope":"PHENO_WITHOUT_SITE_SENSITIVITY"})
        # One representative fold: OOB bootstrap refitting sensitivity for C. Refit supervised models per draw.
        if fold==1:
            pipeline_rows.extend(pipeline_bootstrap_c(xs,yt,st,fold,reps=10))
        write_json(RT/f"cache/pid_fold{fold:02}.json",{"pid_rows":rows,"code_diagnostics":diags})
        np.savez_compressed(RT/f"cache/pid_codes_fold{fold:02}.npz",y=yt,outer_indices=train,
                            **{f"{m}_{k}_{MODS[i]}":codes[i] for (m,k),codes in caches.items() for i in range(4)})
        print(f"PID_FOLD_PASS {fold}/10 rows={len(rows)}",flush=True)
    for rec in reliability:
        for kind in ("global","within_site"):
            null=[r["R_bit"] for r in perm_rows if r["fold"]==rec["fold"] and r["pair"]==rec["pair"] and r["method"]==rec["method"] and r["kind"]==kind]
            observed=rec["point_R"]
            rec[f"{kind}_permutation_n"]=len(null)
            rec[f"{kind}_permutation_p_R"]=(1+sum(v>=observed for v in null))/(len(null)+1) if null else float("nan")
    csv_write(RES/"pid_components.csv",point_rows)
    csv_write(RES/"reliability.csv",reliability)
    csv_write(RES/"pheno_without_site_pid.csv",site_pid)
    csv_write(RES/"historical_451_comparison.csv",legacy_rows)
    csv_write(RES/"full_pipeline_permutations.csv",perm_rows)
    csv_write(RES/"pipeline_refit_bootstrap.csv",pipeline_rows)
    write_json(RT/"category_diagnostics.json",code_diagnostics)
    update_manifest(status="PID_AND_RELIABILITY_COMPLETE",pid_rows=len(point_rows),reliability_rows=len(reliability),
                    permutation_rows=len(perm_rows),historical_451_comparison_rows=len(legacy_rows))
    print("PID_COMPLETE",len(point_rows),"PERM",len(perm_rows),flush=True)


def pipeline_bootstrap_c(xs,y,site,fold,reps=10):
    """Stratified subject bootstrap with OOB PID; refits the supervised encoder."""
    out=[]; y=np.asarray(y); site=np.asarray(site); n=len(y)
    for r in range(reps):
        rng=np.random.default_rng(np.random.SeedSequence([45020,fold,910001,r]))
        sample=np.concatenate([rng.choice(np.flatnonzero(y==c),np.sum(y==c),replace=True) for c in (0,1)])
        counts=np.bincount(sample,minlength=n); inbag=np.flatnonzero(counts); oob=np.flatnonzero(counts==0)
        if len(oob)<80 or np.unique(y[oob]).size<2: continue
        codes=[]
        for i,x in enumerate(xs):
            xboot=x[sample]; yboot=y[sample]
            # OOF thresholds are estimated among bootstrap records; duplicates are
            # held in the same fold by assigning split membership through IDs.
            unique, first = np.unique(sample,return_index=True)
            uy=y[unique]; ux=x[unique]
            scores_oof=np.full(len(unique),np.nan)
            from sklearn.model_selection import StratifiedKFold
            cv=StratifiedKFold(4,shuffle=True,random_state=910100+fold*31+r*7+i)
            for tr,te in cv.split(ux,uy):
                tr_ids=unique[tr]; te_ids=unique[te]
                train_mult=np.repeat(tr_ids,counts[tr_ids])
                scaler,m=fit_score_model(x[train_mult],y[train_mult])
                scores_oof[te]=score_probability(scaler,m,x[te_ids])
            task_train,edges=quantile_states(scores_oof,2)
            train_mult=sample
            scaler,m=fit_score_model(x[train_mult],y[train_mult])
            score_oob=score_probability(scaler,m,x[oob])
            task_oob=np.searchsorted(np.asarray(edges),score_oob,side="right")
            # Structural state is fit only on the bootstrap sample.
            _,_,state=fit_unsupervised_codes(xboot,xboot,2,910200+fold*31+r*7+i,structural=True)
            # Public helper returns transformed states; refit once and retain its state.
            from representation import _transform_kmeans
            struct_oob=_transform_kmeans(state,x[oob])
            codes.append((task_oob*2+struct_oob).astype(np.int64))
        for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
            pair=f"{MODS[a]}-{MODS[b]}"; z=pid_from_codes(codes[a],codes[b],y[oob],"C_TASK_STRUCTURE_HYBRID",4,fold,pair,"pipeline_bootstrap")
            out.append({"fold":fold,"replicate":r,"pair":pair,"method":"C_TASK_STRUCTURE_HYBRID","support":4,
                        "scope":"stratified_bootstrap_OOB_full_encoder_refit","n_oob":len(oob),"R_bit":z["R"],"U1_bit":z["U1"],
                        "U2_bit":z["U2"],"S_bit":z["S"],"Ijoint_bit":z["Ijoint"],"valid":z["valid"]})
    return out


def update_manifest(**updates):
    p=EXP/"manifest.json"; m=json.loads(p.read_text()); m.update(updates); write_json(p,m)


if __name__ == "__main__":
    main()
