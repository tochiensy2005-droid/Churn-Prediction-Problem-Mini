"""src/training/calibrate_model.py

Calibrates raw probability outputs using Platt Scaling (Logistic Regression).
"""

from __future__ import annotations

import logging
import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def fit_platt_calibration(val_probs: np.ndarray, y_val: np.ndarray) -> LogisticRegression:
    logger.info("Fitting Platt Scaling probability calibrator...")
    platt = LogisticRegression(C=1.0)
    platt.fit(val_probs.reshape(-1, 1), y_val)
    return platt
