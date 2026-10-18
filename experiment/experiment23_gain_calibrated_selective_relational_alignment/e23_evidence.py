"""Experiment 23: outer-train-only, label-auditable OOF gain evidence.

No test labels are accepted as arguments in build_fold. All feature scalers,
base teachers and meta-regressors use strictly the indices passed as `train`.
Historical preprocessing/RFE are an input to this module, NOT reimplemented.
"""
from __future__ import annotations
import hashlib
import json
import warnings
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6


def _prob0(model, x):
    classes = model.steps[-1][1].classes_
    col = np.flatnonzero(classes == 0)
    assert len(col) == 1
    return np.clip(model.predict_proba(x)[:, int(col[0])], EPS, 1 - EPS)


def _teacher():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1., solver='lbfgs', max_iter=1000))


def _loss_per_person(p_asd, y):
    p = np.clip(np.asarray(p_asd, np.float64), EPS, 1-EPS)
    return np.where(np.asarray(y) == 0, -np.log(p), -np.log(1-p))


def _score_features(pfc, pfj, xqc):
    pfc = np.asarray(pfc, np.float64)
    pfj = np.asarray(pfj, np.float64)
    xqc = np.asarray(xqc, np.float64)
    assert pfc.ndim == pfj.ndim == 1 and len(pfc) == len(pfj) == len(xqc)
    raw = np.column_stack([
        pfc, pfj, abs(pfc - pfj), 2 * abs(pfc - .5), 2 * abs(pfj - .5),
        -(pfc * np.log(pfc) + (1-pfc)*np.log(1-pfc)),
        -(pfj * np.log(pfj) + (1-pfj)*np.log(1-pfj)),
        xqc,
    ])
    assert np.isfinite(raw).all()
    return raw


def _ridge():
    # Highly regularized on purpose: gain target is noisy per subject.
    return make_pipeline(StandardScaler(), Ridge(alpha=150.))


def _budget_match(raw, mean_target, eligible, low=.02, high=.98):
    """Monotone transform to match target mean with fixed, training-derived threshold.
    `eligible` is ONLY the outer train set, even when `raw` spans all subjects.
    """
    raw = np.asarray(raw, float)
    lo, hi = -40., 40.
    z = np.clip(raw, -30., 30.)
    mean_target = float(np.clip(mean_target, low, high))
    for _ in range(75):
        mid = (lo + hi) / 2
        means = (low + (high - low) * expit(z - mid))[eligible].mean()
        if means > mean_target:
            lo = mid
        else:
            hi = mid
    result = low + (high - low) * expit(z - (lo + hi)/2)
    assert abs(result[eligible].mean() - mean_target) < 1e-6
    return result


def _derange(subset, seed, n):
    """Random cycle (no identity mapping), strictly within each partition."""
    subset = np.asarray(subset, dtype=int)
    assert len(subset) >= 2
    rng = np.random.default_rng(seed)
    order = rng.permutation(subset)
    index = np.arange(n)
    index[order] = np.roll(order, -1)
    assert np.all(index[subset] != subset)
    return index


def build_fold(x, y, train, test, pheno28, seed=23000):
    """Never access y[test]. y passed for indexability but only y[train] read.

    Returns N-length deterministic gates, fold evidence and source OOF stats.
    Teacher OOF predictions exclude that subject from teacher fitting. Meta-gate
    5-fold is exploratory due overlapping first-level OOF teacher training pools.
    """
    x = np.asarray(x, np.float64)
    train, test = np.asarray(train, int), np.asarray(test, int)
    y_train = np.asarray(y)[train].astype(int)
    n = len(x)
    assert x.shape == (871, 320) and sorted(np.unique(y_train).tolist()) == [0, 1]
    assert len(set(train).intersection(test)) == 0 and len(train)+len(test) == n
    pheno28 = np.asarray(pheno28, int)
    assert len(pheno28) == 28 and np.all(pheno28 < 48)
    xf = x[:, 64:320]
    xp = x[:, pheno28]
    xfp = np.concatenate([xf, xp], axis=1)
    qc = x[:, 48:64]
    # Site category is excluded from teachers/gate. The original SPromptGL remains unchanged.
    oof_f, oof_fp = np.full(n, np.nan), np.full(n, np.nan)
    inner_splits = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(x[train], y_train))
    visits = np.zeros(len(train), dtype=np.int32)
    for a, b in inner_splits:
        visits[b] += 1
        for data, output in [(xf, oof_f), (xfp, oof_fp)]:
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                est = _teacher()
                est.fit(data[train[a]], y_train[a])
            output[train[b]] = _prob0(est, data[train[b]])
    assert np.all(visits == 1)
    test_f, test_fp = np.empty(len(test)), np.empty(len(test))
    for data, dest in [(xf, test_f), (xfp, test_fp)]:
        model = _teacher()
        model.fit(data[train], y_train)
        dest[:] = _prob0(model, data[test])
    p_f = oof_f.copy(); p_fp = oof_fp.copy()
    p_f[test], p_fp[test] = test_f, test_fp
    assert np.isfinite(p_f).all() and np.isfinite(p_fp).all()
    # Positive means fusion improved held-out teacher log-loss for THAT train subject.
    delta = _loss_per_person(oof_f[train], y_train) - _loss_per_person(oof_fp[train], y_train)
    target = np.clip(delta, -1.5, 1.5)
    score = _score_features(p_f, p_fp, qc)
    # No tuning or evaluation on test labels; this OOF meta score is EXPLORATORY.
    meta_oof = np.zeros(len(train), dtype=np.float64)
    for a,b in KFold(5, shuffle=True, random_state=seed + 71).split(train):
        reg = _ridge(); reg.fit(score[train[a]], target[a])
        meta_oof[b] = reg.predict(score[train[b]])
    reg = _ridge(); reg.fit(score[train], target)
    pred_gain = np.clip(reg.predict(score), -1.5, 1.5)
    # Construct absolute gain gate from predicted loss change and training-only scale.
    gain_scale = max(float(target.std()), .05)
    gate = .02 + .96 * expit(pred_gain / gain_scale)
    # Confidence comparator budget matched on outer TRAIN; not the test distribution.
    raw_conf = 2 * (0.5 - np.abs(p_f - 0.5))
    conf = _budget_match(4*(raw_conf - raw_conf[train].mean()), gate[train].mean(), train)
    permute = np.arange(n)
    for group, s in [(train, seed+491), (test, seed+509)]:
        partial = _derange(group, s, n)
        permute[group] = partial[group]
    shuffled = gate[permute]
    # Mean-matched within train and test separately: permutation preserves each distribution.
    assert np.allclose(np.sort(shuffled[train]), np.sort(gate[train]))
    assert np.allclose(np.sort(shuffled[test]), np.sort(gate[test]))
    # Training-only feasibility metrics; never call metrics on test ground truth here.
    pf = oof_f[train]; pj = oof_fp[train]
    high = meta_oof >= np.quantile(meta_oof, .8)
    rho = spearmanr(meta_oof, delta).statistic if meta_oof.std() > 1e-9 and delta.std()>1e-9 else 0.
    if not np.isfinite(rho): rho=0.
    report = dict(
        train_n=int(len(train)), test_n=int(len(test)),
        fc_oof_logloss=float(_loss_per_person(pf,y_train).mean()),
        joint_oof_logloss=float(_loss_per_person(pj,y_train).mean()),
        mean_oof_delta=float(delta.mean()),
        positive_delta_fraction=float((delta > .02).mean()),
        high20_meta_oof_delta=float(delta[high].mean()),
        meta_oof_spearman=float(rho),
        gain_gate_train_mean=float(gate[train].mean()),
        gain_gate_train_std=float(gate[train].std()),
        confidence_gate_train_mean=float(conf[train].mean()),
        site_in_auxiliary=False, subject_label_used_for_gate_test=False,
        meta_oof_is_nested_independent=False,
        fc_oof_ba=float(balanced_accuracy_score(y_train, (pf<.5).astype(int))),
        fp_oof_ba=float(balanced_accuracy_score(y_train, (pj<.5).astype(int))),
        fc_oof_auc=float(roc_auc_score(y_train==0, pf)),
        fp_oof_auc=float(roc_auc_score(y_train==0, pj)),
    )
    return dict(gain_gate=gate.astype(np.float32), constant_gate=np.full(n,gate[train].mean(),np.float32),
                confidence_gate=conf.astype(np.float32), shuffled_gate=shuffled.astype(np.float32),
                pheno_permutation=permute.astype(np.int64),
                pfc=p_f.astype(np.float32), pjoint=p_fp.astype(np.float32),
                outer_train=train.astype(np.int64), outer_test=test.astype(np.int64),
                gain_meta_oof=meta_oof.astype(np.float32), delta_train=delta.astype(np.float32),
                report=report)


def readonly_evidence_digest(obj):
    d = hashlib.sha256()
    for k,v in sorted(obj.items()):
        if isinstance(v,np.ndarray):d.update(k.encode());d.update(np.ascontiguousarray(v).tobytes())
    return d.hexdigest()


def self_test():
    rng=np.random.default_rng(23)
    x=rng.normal(size=(871,320))
    y=(rng.random(871)>.47).astype(int)
    tr=np.arange(780);te=np.arange(780,871)
    out=build_fold(x,y,tr,te,np.arange(28),seed=23001)
    assert np.isfinite(out['gain_gate']).all()
    altered=y.copy(); altered[te]=1-altered[te]
    other=build_fold(x,altered,tr,te,np.arange(28),seed=23001)
    for k in ('gain_gate','constant_gate','confidence_gate','shuffled_gate','pfc','pjoint'):
        assert np.array_equal(out[k],other[k]), 'TEST LABEL LEAK: '+k
    print('Evidence self-test PASS, no test-label use')

if __name__=='__main__':self_test()

