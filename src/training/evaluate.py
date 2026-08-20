"""src/training/evaluate.py

Helper module to calculate evaluation metrics.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    
    return {
        "PR_AUC": float(average_precision_score(y_true, probs)),
        "ROC_AUC": float(roc_auc_score(y_true, probs)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "F1": float(f1_score(y_true, preds, zero_division=0)),
        "Brier": float(brier_score_loss(y_true, probs)),
        "LogLoss": float(log_loss(y_true, probs)),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }
