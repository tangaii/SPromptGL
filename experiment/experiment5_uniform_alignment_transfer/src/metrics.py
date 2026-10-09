"""Metrics used by both Experiment 5 protocols.

The repository uses internal labels 0=ASD and 1=NC.  Reporting metrics use
ASD as the positive class, while ``legacy_hard_auc`` deliberately preserves
the original repository's one-hot hard-prediction AUC.
"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def _as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def safe_auc(y_true, score):
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def legacy_hard_auc(labels_internal, pred_internal, nclass=2):
    labels_internal = np.asarray(labels_internal, dtype=np.int64)
    pred_internal = np.asarray(pred_internal, dtype=np.int64)
    target = np.eye(nclass, dtype=np.float64)[labels_internal]
    prediction = np.eye(nclass, dtype=np.float64)[pred_internal]
    return float(roc_auc_score(target, prediction))


def metrics_from_arrays(labels_internal, log_prob, pred_internal=None):
    labels_internal = np.asarray(labels_internal, dtype=np.int64)
    log_prob = _as_numpy(log_prob).astype(np.float64)
    probabilities = np.exp(log_prob)
    if pred_internal is None:
        pred_internal = np.argmax(log_prob, axis=1)
    pred_internal = np.asarray(pred_internal, dtype=np.int64)

    y_asd = (labels_internal == 0).astype(np.int64)
    pred_asd = (pred_internal == 0).astype(np.int64)
    tp = int(np.sum((y_asd == 1) & (pred_asd == 1)))
    tn = int(np.sum((y_asd == 0) & (pred_asd == 0)))
    fp = int(np.sum((y_asd == 0) & (pred_asd == 1)))
    fn = int(np.sum((y_asd == 1) & (pred_asd == 0)))
    sensitivity = float(tp / (tp + fn)) if tp + fn else 0.0
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    return {
        "ACC": float(np.mean(pred_internal == labels_internal)),
        "BA": 0.5 * (sensitivity + specificity),
        "probability_ASD_AUC": safe_auc(y_asd, probabilities[:, 0]),
        "legacy_hard_AUC": legacy_hard_auc(labels_internal, pred_internal),
        "ASD_F1": float(f1_score(y_asd, pred_asd, zero_division=0)),
        "ASD_sensitivity": sensitivity,
        "NC_specificity": specificity,
        "p_asd": probabilities[:, 0],
        "pred_internal": pred_internal,
        "pred_asd": pred_asd,
    }


def evaluate_prob(log_prob, labels, indices):
    """Evaluate a full-cohort log-probability tensor on ``indices``."""
    indices = _as_numpy(indices).astype(np.int64)
    labels_np = _as_numpy(labels).astype(np.int64)
    log_prob_np = _as_numpy(log_prob)
    return metrics_from_arrays(labels_np[indices], log_prob_np[indices])

