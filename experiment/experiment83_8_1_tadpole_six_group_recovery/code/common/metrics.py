"""Official E83.1 metrics only."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def one_hot(labels: np.ndarray, nclass: int) -> np.ndarray:
    out = np.zeros((len(labels), nclass), dtype=np.float64)
    out[np.arange(len(labels)), labels.astype(int)] = 1.0
    return out


def acc_auc(y_true: np.ndarray, prediction: np.ndarray, nclass: int = 3) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    hard_auc = roc_auc_score(one_hot(y_true, nclass), one_hot(prediction, nclass),
                             average="macro", multi_class="ovr")
    return {"ACC": float((y_true == prediction).mean()), "AUC": float(hard_auc)}

