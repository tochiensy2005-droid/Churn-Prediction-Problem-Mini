"""src/inference/predict_churn.py

Executes batch churn predictions for a given snapshot date.
Validates input schemas, imputes missing values, predicts raw probabilities,
calibrates predictions, and outputs parquet predictions.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pandas as pd
import joblib

from src import config
from src.features.build_temporal_base import build_temporal_base_features
from src.features.build_lag_features import build_lag_features
from src.features.build_rolling_features import build_rolling_features
from src.features.build_trend_features import build_trend_features
from src.features.build_recency_features import build_recency_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def validate_inference_schema(df: pd.DataFrame, expected_features: list[str]) -> None:
    logger.info("Validating schema before inference...")
    
    # 1. Check missing features
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        raise ValueError(f"Schema mismatch: missing {len(missing)} expected features. Missing features: {missing[:10]}")
        
    # 2. Check NaN rates
    nan_rates = df[expected_features].isna().mean()
    high_nan = nan_rates[nan_rates > 0.90].index.tolist()
    if high_nan:
        logger.warning(f"High NaN rate (>90%) detected in features: {high_nan[:10]}")
        
    # 3. Check dtypes (numeric features must be float/int)
    for col in expected_features:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Dtype mismatch: expected numeric type for column {col}, found {df[col].dtype}")
            
    logger.info("Schema validation successful. Fail-fast checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production batch churn inference.")
    parser.add_argument("--snapshot-date", type=str, required=True, help="Snapshot date in YYYY-MM-DD format")
    args = parser.parse_args()
    
    target_date = pd.Timestamp(args.snapshot_date)
    logger.info(f"Running inference for snapshot date: {target_date.date()}")
    
    # Load model bundle
    artifact_path = config.ARTIFACTS_DIR / "temporal_churn_model.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Production model bundle not found at {artifact_path}. Please train a model first.")
        
    bundle = joblib.load(artifact_path)
    model = bundle["model"]
    calibrator = bundle["calibrator"]
    features = bundle["selected_features"]
    imputer = bundle["imputer"]
    threshold = bundle["threshold"]
    
    # Prepare features
    # Check if precomputed features are available locally first
    precomputed_path = config.OUTPUT_DIR / "churn_temporal_dataset.parquet"
    if precomputed_path.exists():
        logger.info("Precomputed features parquet found. Loading and slicing snapshot...")
        df_all = pd.read_parquet(precomputed_path)
        df_all["snapshot_date"] = pd.to_datetime(df_all["snapshot_date"])
        snap_df = df_all[df_all["snapshot_date"] == target_date].copy()
    else:
        logger.info("Precomputed features not found. Building temporal features on the fly from silver tables...")
        # 1. Base temporal features
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
        
        # Merge all features
        logger.info("Merging generated feature tables...")
        merged = base.merge(df_lag[["customer_id", "snapshot_date"] + lag_cols], on=["customer_id", "snapshot_date"], how="left")
        merged = merged.merge(df_roll, on=["customer_id", "snapshot_date"], how="left")
        merged = merged.merge(df_trend, on=["customer_id", "snapshot_date"], how="left")
        merged = merged.merge(df_rec, on=["customer_id", "snapshot_date"], how="left")
        
        snap_df = merged[merged["snapshot_date"] == target_date].copy()
        
    if len(snap_df) == 0:
        raise ValueError(f"No customer records found for snapshot date {args.snapshot_date} in the feature database.")
        
    logger.info(f"Loaded {len(snap_df)} customer records for inference.")
    
    # Validate schema
    validate_inference_schema(snap_df, features)
    
    # Run Imputer
    X_snap = snap_df[features].copy()
    X_snap_imp = imputer.transform(X_snap)
    
    # Predict raw probabilities
    raw_probs = model.predict_proba(X_snap_imp)[:, 1]
    
    # Apply Calibration
    calibrated_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    
    # Predict labels
    predictions = (calibrated_probs >= threshold).astype(int)
    
    # Build output dataframe
    output_df = pd.DataFrame({
        "customer_id": snap_df["customer_id"].values,
        "snapshot_date": snap_df["snapshot_date"].dt.date.values,
        "churn_probability": calibrated_probs,
        "churn_prediction": predictions
    })
    
    # Save predictions
    out_dir = config.OUTPUT_DIR / "predictions" / args.snapshot_date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "churn_predictions.parquet"
    output_df.to_parquet(out_file, index=False)
    
    logger.info(f"Inference complete. Batch predictions saved to {out_file}.")
    print(output_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
