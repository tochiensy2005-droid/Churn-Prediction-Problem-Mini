"""src/monitoring/performance_drift.py

Evaluates performance degradation (PR-AUC, ROC-AUC, F1) when ground truth labels become available.
"""

from __future__ import annotations

import numpy as np
from src.training.evaluate import evaluate_predictions


def calculate_performance_drift(probs: np.ndarray, threshold: float, actuals: np.ndarray) -> dict:
    metrics = evaluate_predictions(actuals, probs, threshold)
    metrics["actual_churn_rate"] = float(np.mean(actuals))
    return metrics
