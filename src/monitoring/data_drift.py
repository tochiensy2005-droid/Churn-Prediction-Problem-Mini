"""src/monitoring/data_drift.py

Calculates Population Stability Index (PSI) to detect feature distribution drift.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """Computes Population Stability Index (PSI) between two 1D distributions."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    # Deduplicate percentiles to avoid zero-width bins in highly skewed features
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    bins = np.unique(bins)
    if len(bins) < 2:
        # Fallback to standard min/max scaling range if percentiles yield single value
        bins = np.linspace(expected.min() - 1e-5, expected.max() + 1e-5, num_bins + 1)
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
            
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    eps = 1e-4
    expected_pct = np.clip(expected_pct, eps, 1.0)
    actual_pct = np.clip(actual_pct, eps, 1.0)
    
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)
