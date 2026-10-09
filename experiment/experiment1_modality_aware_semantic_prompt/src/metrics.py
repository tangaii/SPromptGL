from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def classification_metrics(y_true, probability) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape != (len(y_true), 2):
        raise ValueError(f"expected probabilities [N, 2], got {probability.shape}")
    if not np.isfinite(probability).all():
        raise ValueError("non-finite predicted probabilities")
    pred = probability.argmax(axis=1)
    asd_positive = (y_true == 0).astype(np.int64)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "probability_auc_asd_positive": float(roc_auc_score(asd_positive, probability[:, 0])),
        "accuracy": float(accuracy_score(y_true, pred)),
        "asd_sensitivity": float(np.mean(pred[y_true == 0] == 0)),
        "nc_specificity": float(np.mean(pred[y_true == 1] == 1)),
        "f1_asd_positive": float(f1_score(asd_positive, (pred == 0).astype(np.int64), zero_division=0)),
    }
