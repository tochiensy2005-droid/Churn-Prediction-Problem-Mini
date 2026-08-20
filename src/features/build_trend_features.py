"""src/features/build_trend_features.py

Generates trend statistics (1-month change, slope) for monthly snapshots.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def estimate_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 3:
        return 0.0
    x = np.arange(n)
    cov = np.cov(x, y)[0, 1]
    var = np.var(x, ddof=1)
    return float(cov / var) if var > 0 else 0.0


def build_trend_features(base: pd.DataFrame, behavioral_cols: list[str]) -> pd.DataFrame:
    logger.info("Computing trend features...")
    df = base.sort_values(["customer_id", "snapshot_date"]).copy()
    grouped = df.groupby("customer_id", sort=False)
    
    trend_df = df.copy()
    for col in behavioral_cols:
        trend_df[f"{col}_change_1m"] = grouped[col].diff(1)
        trend_df[f"{col}_pct_change_1m"] = grouped[col].pct_change(1).replace([np.inf, -np.inf], np.nan)
        trend_df[f"{col}_slope_3m"] = grouped[col].rolling(window=3, min_periods=3).apply(estimate_slope, raw=True).reset_index(level=0, drop=True)
        
    trend_cols = [c for c in trend_df.columns if "_change_" in c or "_slope_" in c]
    out_cols = ["customer_id", "snapshot_date"] + trend_cols
    df_trend = trend_df[out_cols].copy()
    return df_trend
