"""generate_features.py

Orchestrates the temporal feature extraction, recency calculations,
forward labeling, and base parquet dataset generation.
"""

import logging
from pathlib import Path
import pandas as pd

from src import config
from src.features.build_temporal_base import build_temporal_base_features
from src.features.build_lag_features import build_lag_features
from src.features.build_rolling_features import build_rolling_features
from src.features.build_trend_features import build_trend_features
from src.features.build_recency_features import build_recency_features
from src.labels.build_churn_labels import build_churn_labels

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Temporal Feature Engineering Pipeline...")
    
    # 1. Base temporal features (Monthly aggregation)
    base = build_temporal_base_features()
    behavioral = [c for c in base.columns if c not in {"customer_id", "snapshot_date"}]
    
    # 2. Lag features
    df_lag = build_lag_features(base, behavioral)
    lag_cols = [c for c in df_lag.columns if "_lag_" in c]
    
    # 3. Rolling features
    df_roll = build_rolling_features(base, behavioral)
    roll_cols = [c for c in df_roll.columns if "_rolling_" in c]
    
    # 4. Trend features
    df_trend = build_trend_features(base, behavioral)
    trend_cols = [c for c in df_trend.columns if "_change_" in c or "_slope_" in c]
    
    # 5. Recency features
    df_rec = build_recency_features(base)
    rec_cols = [c for c in df_rec.columns if "days_since_last_" in c]
    
    # 6. Churn labels
    churn_labels = build_churn_labels(base)
    
    # 7. Merge all features and labels
    logger.info("Merging features and labels into final dataset...")
    merged = base.merge(df_lag[["customer_id", "snapshot_date"] + lag_cols], on=["customer_id", "snapshot_date"], how="left")
    merged = merged.merge(df_roll, on=["customer_id", "snapshot_date"], how="left")
    merged = merged.merge(df_trend, on=["customer_id", "snapshot_date"], how="left")
    merged = merged.merge(df_rec, on=["customer_id", "snapshot_date"], how="left")
    merged["churn_next_30d"] = churn_labels
    
    # Save output
    output_path = config.OUTPUT_DIR / "churn_temporal_dataset.parquet"
    merged.to_parquet(output_path, index=False)
    logger.info(f"Temporal dataset successfully generated at {output_path}. Shape: {merged.shape}")

if __name__ == "__main__":
    main()
