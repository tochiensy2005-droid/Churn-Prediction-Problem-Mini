"""src/monitoring/prediction_drift.py

Calculates summary statistics on predicted probability distributions.
"""

from __future__ import annotations

import numpy as np


def calculate_prediction_drift(probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    return {
        "mean_prob": float(np.mean(probs)),
        "median_prob": float(np.median(probs)),
        "p90_prob": float(np.percentile(probs, 90)),
        "p95_prob": float(np.percentile(probs, 95)),
        "p99_prob": float(np.percentile(probs, 99)),
        "predicted_churn_rate": float(np.mean(preds))
    }
