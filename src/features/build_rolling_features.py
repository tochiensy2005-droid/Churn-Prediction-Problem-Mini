"""src/features/build_rolling_features.py

Generates rolling statistics for monthly snapshots.
"""

from __future__ import annotations

import logging
import pandas as pd
from src import config

logger = logging.getLogger(__name__)


def build_rolling_features(base: pd.DataFrame, behavioral_cols: list[str]) -> pd.DataFrame:
    logger.info("Computing rolling features...")
    df = base.sort_values(["customer_id", "snapshot_date"]).copy()
    grouped = df.groupby("customer_id", sort=False)
    
    rolling_df = df.copy()
    for col in behavioral_cols:
        for w in (1, 3, 6):
            rolling = grouped[col].rolling(window=w, min_periods=1)
            rolling_df[f"{col}_rolling_sum_{w}m"] = rolling.sum().reset_index(level=0, drop=True)
            rolling_df[f"{col}_rolling_mean_{w}m"] = rolling.mean().reset_index(level=0, drop=True)
            rolling_df[f"{col}_rolling_std_{w}m"] = rolling.std(ddof=0).reset_index(level=0, drop=True)
            rolling_df[f"{col}_rolling_min_{w}m"] = rolling.min().reset_index(level=0, drop=True)
            rolling_df[f"{col}_rolling_max_{w}m"] = rolling.max().reset_index(level=0, drop=True)
            
    roll_cols = [c for c in rolling_df.columns if "_rolling_" in c]
    # Keep behavioral columns and lag columns out of rolling parquet to follow refactored pipeline structure
    out_cols = ["customer_id", "snapshot_date"] + roll_cols
    df_roll = rolling_df[out_cols].copy()
    
    df_roll.to_parquet(config.OUTPUT_DIR / "rolling_features.parquet", index=False)
    logger.info(f"Rolling features parquet written to {config.OUTPUT_DIR / 'rolling_features.parquet'}. Shape: {df_roll.shape}")
    return df_roll
