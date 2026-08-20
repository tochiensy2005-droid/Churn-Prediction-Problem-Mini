"""src/features/build_lag_features.py

Generates historical lags for monthly snapshots.
"""

from __future__ import annotations

import logging
import pandas as pd
from src import config

logger = logging.getLogger(__name__)


def build_lag_features(base: pd.DataFrame, behavioral_cols: list[str]) -> pd.DataFrame:
    logger.info("Computing lag features...")
    df = base.sort_values(["customer_id", "snapshot_date"]).copy()
    grouped = df.groupby("customer_id", sort=False)
    
    lagged = df.copy()
    for col in behavioral_cols:
        for lag in (1, 2, 3):
            lagged[f"{col}_lag_{lag}"] = grouped[col].shift(lag)
            
    lag_cols = [c for c in lagged.columns if "_lag_" in c]
    out_cols = ["customer_id", "snapshot_date"] + behavioral_cols + lag_cols
    df_lag = lagged[out_cols].copy()
    
    df_lag.to_parquet(config.OUTPUT_DIR / "lag_features.parquet", index=False)
    logger.info(f"Lag features parquet written to {config.OUTPUT_DIR / 'lag_features.parquet'}. Shape: {df_lag.shape}")
    return df_lag
