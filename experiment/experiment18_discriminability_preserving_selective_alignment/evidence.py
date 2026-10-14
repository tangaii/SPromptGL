"""Outer-train-only OOF teachers and joint-correctness reliability proxy."""
import io
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TEACHER_SETTINGS = dict(C=1.0, solver="lbfgs", max_iter=1000)
FEATURE_SETS = ("FC", "PHENO", "FC_PHENO", "SITE", "FC_SITE", "FC_PHENO_SITE")


def fit_teacher(x, y):
    model = make_pipeline(StandardScaler(), LogisticRegression(**TEACHER_SETTINGS))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y)
    notices = [str(w.message) for w in caught]
    return model, notices


def probability_asd(model, x):
    position = np.flatnonzero(model.classes_ == 0)
    assert len(position) == 1
    return model.predict_proba(x)[:, int(position[0])]


def evidence_features(fc_probability, pheno_probability):
    return np.column_stack([fc_probability, pheno_probability,
                            2 * np.abs(fc_probability - .5), 2 * np.abs(pheno_probability - .5),
                            np.abs(fc_probability - pheno_probability),
                            (fc_probability >= .5) == (pheno_probability >= .5)]).astype(np.float64)


def derangement(indices, generator):
    assert len(indices) > 1
    order = generator.permutation(indices)
    result = np.arange(871)
    result[order] = np.roll(order, -1)
    assert np.all(result[indices] != indices)
    return result


def build_outer(x, y, train, test, fold, pheno_indices, site_indices):
    """All supervised fitting gets y[train] only; outer test predictions are unlabeled."""
    train, test = np.asarray(train), np.asarray(test)
    train_y = np.asarray(y)[train]
    inputs = {
        "FC": x[:, 64:], "PHENO": x[:, pheno_indices], "SITE": x[:, site_indices],
        "FC_PHENO": x[:, np.r_[np.arange(64, 320), pheno_indices]],
        "FC_SITE": x[:, np.r_[np.arange(64, 320), site_indices]],
        "FC_PHENO_SITE": x[:, np.r_[np.arange(64, 320), pheno_indices, site_indices]],
    }
    inner = list(StratifiedKFold(5, shuffle=True, random_state=18000 + fold).split(x[train], train_y))
    inner_fold = np.full(871, -1, dtype=np.int16)
    counts = np.zeros(len(train), dtype=np.int16)
    for inner_number, (fit_idx, held_idx) in enumerate(inner):
        assert not np.intersect1d(fit_idx, held_idx).size
        counts[held_idx] += 1
        inner_fold[train[held_idx]] = inner_number
    assert np.all(counts == 1)
    probs, rows = {}, []
    all_notices = []
    for name in FEATURE_SETS:
        features = inputs[name]
        p = np.full(871, np.nan)
        for fit_idx, held_idx in inner:
            model, notices = fit_teacher(features[train[fit_idx]], train_y[fit_idx])
            all_notices.extend(notices)
            p[train[held_idx]] = probability_asd(model, features[train[held_idx]])
        # Only the two actual teachers need outer-test inference.
        if name in ("FC", "PHENO"):
            model, notices = fit_teacher(features[train], train_y)
            all_notices.extend(notices)
            p[test] = probability_asd(model, features[test])
        probs[name] = p
        pred = np.where(p[train] >= .5, 0, 1)
        rows.append(dict(fold=fold,feature_set=name,train_n=len(train),test_n=len(test),
                         dimensions=features.shape[1],oof_acc=float(accuracy_score(train_y, pred)),
                         oof_ba=float(balanced_accuracy_score(train_y, pred)),
                         oof_probability_auc=float(roc_auc_score(train_y == 0, p[train])),
                         asd_fraction=float((train_y == 0).mean()),
                         predicted_asd_fraction=float((pred == 0).mean()),
                         oof_exclusion_verified=True,outer_test_labels_used=False))
    fc, ph = probs["FC"], probs["PHENO"]
    joint = np.full(871, np.nan)
    joint[train] = ((np.where(fc[train] >= .5, 0, 1) == train_y) &
                    (np.where(ph[train] >= .5, 0, 1) == train_y)).astype(float)
    ef = evidence_features(fc, ph)
    if len(np.unique(joint[train])) == 1:
        reliability = np.full(871, joint[train].mean())
        constant_prior = True
        reliability_model = {}
    else:
        model, notices = fit_teacher(ef[train], joint[train].astype(int))
        all_notices.extend(notices)
        reliability = model.predict_proba(ef)[:, np.flatnonzero(model.classes_ == 1)[0]]
        constant_prior = False
        scaler, lr = model.steps[0][1], model.steps[1][1]
        reliability_model = dict(scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                                 lr_coef=lr.coef_, lr_intercept=lr.intercept_)
    identity = np.arange(871)
    reliability_perm = derangement(train, np.random.default_rng(18600 + fold))
    class_perm = identity.copy()
    gen = np.random.default_rng(18900 + fold)
    for category in (0, 1):
        group = train[train_y == category]
        class_perm[group] = derangement(group, gen)[group]
    cf_perm = identity.copy()
    gen = np.random.default_rng(18800 + fold)
    for group in (train, test):
        cf_perm[group] = derangement(group, gen)[group]
    assert np.isnan(joint[test]).all()
    assert np.isfinite(ef).all() and np.isfinite(reliability).all()
    assert np.all(y[class_perm[train]] == train_y)
    assert np.all(class_perm[train] != train)
    for row in rows:
        reference = next(r for r in rows if r["feature_set"] == ("FC_SITE" if row["feature_set"] == "FC_PHENO_SITE" else "FC"))
        for key in ("oof_acc", "oof_ba", "oof_probability_auc"):
            row["increment_" + key] = row[key] - reference[key]
        row.update(fc_pheno_agreement=float(((fc[train] >= .5) == (ph[train] >= .5)).mean()),
                   fc_pheno_probability_correlation=float(np.corrcoef(fc[train], ph[train])[0, 1]),
                   joint_correct_fraction=float(joint[train].mean()),
                   reliability_mean=float(reliability[train].mean()),
                   reliability_sd=float(reliability[train].std()),
                   reliability_constant_prior=constant_prior)
    result = dict(train_indices=train,test_indices=test,train_mask=np.isin(identity,train),
                  test_mask=np.isin(identity,test),inner_fold=inner_fold,
                  teacher_fc_probability=fc,teacher_pheno_probability=ph,
                  joint_correct=joint,reliability=reliability,evidence_features=ef,
                  reliability_permutation=reliability_perm,class_pair_permutation=class_perm,
                  cf_permutation=cf_perm,identity=identity,
                  constant_prior=np.asarray(constant_prior), **reliability_model)
    for name in FEATURE_SETS:
        result["oof_" + name] = probs[name]
    return result, rows, all_notices


def pack_evidence(all_folds):
    data = {f"fold{fold}_{k}":v for fold,values in all_folds.items() for k,v in values.items()}
    buf = io.BytesIO()
    np.savez_compressed(buf, **data)
    return buf.getvalue()


def load_evidence(path, fold):
    prefix = f"fold{fold}_"
    with np.load(path, allow_pickle=False) as npz:
        return {k[len(prefix):]:npz[k].copy() for k in npz.files if k.startswith(prefix)}
