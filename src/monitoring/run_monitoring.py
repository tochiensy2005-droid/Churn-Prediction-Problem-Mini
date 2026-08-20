"""src/monitoring/run_monitoring.py

Orchestrates the data, prediction, target drift, and model performance checks
for a specific prediction snapshot date.
"""

from __future__ import annotations

import argparse
import logging
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src import config
from src.monitoring.data_drift import calculate_psi
from src.monitoring.prediction_drift import calculate_prediction_drift
from src.monitoring.performance_drift import calculate_performance_drift
from src.labels.build_churn_labels import build_churn_labels

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

CRITICAL_FEATURES = ["usage_rolling_mean_6m", "payment_count_rolling_min_6m"]


def append_metrics_to_csv(filepath: Path, row_data: dict) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row_data])
    if filepath.exists():
        df_old = pd.read_csv(filepath)
        df_merged = pd.concat([df_old, df_new], ignore_index=True)
        df_merged = df_merged.drop_duplicates(subset=["snapshot_date"], keep="last")
        df_merged.to_csv(filepath, index=False)
    else:
        df_new.to_csv(filepath, index=False)
    logger.info(f"Appended monitoring metrics to {filepath}")


def check_persistence_drift(history_file: Path, current_snapshot: str, psi_threshold: float = 0.10) -> bool:
    """Checks if data drift (mean_psi > psi_threshold) was observed in the previous snapshot."""
    if not history_file.exists():
        return False
    try:
        df_hist = pd.read_csv(history_file)
        if len(df_hist) < 2:
            return False
        df_hist = df_hist.sort_values("snapshot_date").reset_index(drop=True)
        # Find index of current snapshot if it was already appended
        match_idx = df_hist[df_hist["snapshot_date"] == current_snapshot].index
        if len(match_idx) > 0:
            target_idx = match_idx[0] - 1
        else:
            target_idx = len(df_hist) - 1
            
        if target_idx >= 0:
            prev_row = df_hist.iloc[target_idx]
            prev_psi = prev_row.get("mean_psi", 0.0)
            if prev_psi > psi_threshold:
                logger.info(f"Persistent drift detected. Previous snapshot {prev_row['snapshot_date']} had mean PSI of {prev_psi:.4f} (> {psi_threshold})")
                return True
    except Exception as e:
        logger.warning(f"Error checking persistence drift: {e}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production model monitoring.")
    parser.add_argument("--snapshot-date", type=str, required=True, help="Snapshot date in YYYY-MM-DD format")
    args = parser.parse_args()
    
    target_date = pd.Timestamp(args.snapshot_date)
    logger.info(f"Running model monitoring for snapshot date: {target_date.date()}")
    
    # 1. Load predictions
    pred_path = config.OUTPUT_DIR / "predictions" / args.snapshot_date / "churn_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found at {pred_path}. Please run inference first.")
        
    pred_df = pd.read_parquet(pred_path)
    probs = pred_df["churn_probability"].to_numpy()
    
    # 2. Load model bundle
    artifact_path = config.ARTIFACTS_DIR / "temporal_churn_model.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model bundle not found at {artifact_path}.")
        
    bundle = joblib.load(artifact_path)
    features = bundle["selected_features"]
    threshold = bundle["threshold"]
    val_pr_auc_baseline = bundle.get("validation_PR_AUC", 0.1478)
    
    # 3. Load feature data for PSI (expected vs actual)
    dataset_path = config.OUTPUT_DIR / "churn_temporal_dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Temporal dataset not found at {dataset_path}.")
        
    df_all = pd.read_parquet(dataset_path)
    df_all["snapshot_date"] = pd.to_datetime(df_all["snapshot_date"])
    
    # Baseline: Training date range from bundle
    train_start = pd.Timestamp(bundle["training_start"])
    train_end = pd.Timestamp(bundle["training_end"])
    train_df = df_all[(df_all["snapshot_date"] >= train_start) & (df_all["snapshot_date"] <= train_end)]
    
    # Actual feature slice
    snap_features_df = df_all[df_all["snapshot_date"] == target_date]
    
    psi_avg = np.nan
    psi_median = np.nan
    psi_max = np.nan
    count_psi_010 = 0
    count_psi_025 = 0
    top_10_drift = []
    critical_drift_detected = False
    reasons = []
    recommend_retrain = False
    
    if len(snap_features_df) == 0:
        logger.warning(f"Feature table has no record for snapshot {args.snapshot_date} to calculate PSI. Skipping.")
    else:
        logger.info("Computing PSI for Top 100 features...")
        psi_records = []
        for feature in features:
            expected = train_df[feature].to_numpy()
            actual = snap_features_df[feature].to_numpy()
            psi_val = calculate_psi(expected, actual)
            
            # Check critical feature drift
            if any(cf in feature for cf in CRITICAL_FEATURES) and psi_val > 0.25:
                critical_drift_detected = True
                reasons.append(f"Critical Feature Drift: '{feature}' PSI = {psi_val:.4f} (> 0.25)")
                
            psi_records.append({"feature": feature, "PSI": psi_val})
            
        df_psi = pd.DataFrame(psi_records).sort_values("PSI", ascending=False)
        drift_dir = config.OUTPUT_DIR / "monitoring" / "drift"
        drift_dir.mkdir(parents=True, exist_ok=True)
        psi_file = drift_dir / f"data_drift_{args.snapshot_date}.csv"
        df_psi.to_csv(psi_file, index=False)
        
        # Calculate stats
        psi_avg = float(df_psi["PSI"].mean())
        psi_median = float(df_psi["PSI"].median())
        psi_max = float(df_psi["PSI"].max())
        count_psi_010 = int((df_psi["PSI"] > 0.10).sum())
        count_psi_025 = int((df_psi["PSI"] > 0.25).sum())
        top_10_drift = df_psi.head(10).to_dict("records")
        
        logger.info(f"PSI stats: Mean={psi_avg:.4f} | Median={psi_median:.4f} | Max={psi_max:.4f}")
        
    # 4. Calculate prediction drift
    logger.info("Calculating prediction drift statistics...")
    pred_drift = calculate_prediction_drift(probs, threshold)
    pred_drift["snapshot_date"] = args.snapshot_date
    pred_drift["mean_psi"] = psi_avg
    pred_drift["median_psi"] = psi_median
    pred_drift["max_psi"] = psi_max
    
    # Check persistence of drift (mean PSI > 0.10 for consecutive months)
    history_file = config.OUTPUT_DIR / "monitoring" / "drift" / "prediction_drift.csv"
    persistent_drift = check_persistence_drift(history_file, args.snapshot_date, 0.10)
    
    # Append prediction metrics
    append_metrics_to_csv(history_file, pred_drift)
    
    # Print prediction drift statistics
    print("\n============================================================")
    print("MONITORING REPORT: PREDICTION & DATA DRIFT")
    print("============================================================")
    print(f"Mean PSI:             {psi_avg:.6f}")
    print(f"Median PSI:           {psi_median:.6f}")
    print(f"Max PSI:              {psi_max:.6f}")
    print(f"Features PSI > 0.10:  {count_psi_010}")
    print(f"Features PSI > 0.25:  {count_psi_025}")
    print(f"Mean Predicted Prob:  {pred_drift['mean_prob']:.6f}")
    print(f"Predicted Churn Rate: {pred_drift['predicted_churn_rate']:.4%}")
    print("\nTop 10 Drifted Features:")
    for i, r in enumerate(top_10_drift, 1):
        print(f"  {i:2d}. {r['feature']:45s} | PSI: {r['PSI']:.6f}")
        
    # 5. Calculate target drift and performance if labels are available
    label_complete = (target_date + pd.Timedelta(days=30)) <= pd.Timestamp("2026-07-28")
    
    if label_complete:
        logger.info("Ground truth target window is complete. Evaluating performance degradation...")
        grid = pred_df[["customer_id"]].copy()
        grid["snapshot_date"] = target_date
        
        actual_labels = build_churn_labels(grid)
        y_true = actual_labels.to_numpy()
        
        perf_metrics = calculate_performance_drift(probs, threshold, y_true)
        perf_metrics["snapshot_date"] = args.snapshot_date
        
        # Check performance degradation: PR-AUC drop > 20% relative to validation baseline
        pr_auc_ratio = perf_metrics["PR_AUC"] / val_pr_auc_baseline
        if pr_auc_ratio < 0.80:
            recommend_retrain = True
            reasons.append(
                f"Performance Degradation: PR-AUC fell by {100*(1-pr_auc_ratio):.2f}% "
                f"below validation baseline (Current: {perf_metrics['PR_AUC']:.6f} vs Baseline: {val_pr_auc_baseline:.6f})"
            )
            
        # Target drift: actual churn rate out of tolerance
        # Expected baseline validation churn rate is ~0.45%. If actual churn is > 1.5% or < 0.1%, it's out of bounds.
        if perf_metrics["actual_churn_rate"] > 0.015:
            recommend_retrain = True
            reasons.append(f"Target Drift: Actual churn rate {perf_metrics['actual_churn_rate']:.4%} exceeds tolerance limit (1.5%)")
            
        append_metrics_to_csv(config.OUTPUT_DIR / "monitoring" / "drift" / "performance_drift.csv", perf_metrics)
        
        print("\nModel Performance & Target Drift:")
        print(f"  PR-AUC:             {perf_metrics['PR_AUC']:.6f} (Baseline: {val_pr_auc_baseline:.6f})")
        print(f"  ROC-AUC:            {perf_metrics['ROC_AUC']:.6f}")
        print(f"  F1 Score:           {perf_metrics['F1']:.4%}")
        print(f"  Actual Churn Rate:  {perf_metrics['actual_churn_rate']:.4%}")
    else:
        logger.info(f"Target window is incomplete for snapshot date {args.snapshot_date}. Skipping performance checks.")
        
    # Retraining Policy Flags
    if psi_avg > 0.25:
        recommend_retrain = True
        reasons.append(f"High Data Drift: Mean PSI of {psi_avg:.4f} exceeds 0.25")
        
    if psi_max > 0.25:
        recommend_retrain = True
        reasons.append(f"High Data Drift: Max PSI of {psi_max:.4f} exceeds 0.25 (Feature: {df_psi.iloc[0]['feature']})")
        
    if critical_drift_detected:
        recommend_retrain = True
        
    if persistent_drift:
        recommend_retrain = True
        reasons.append("Persistent Drift: Data drift was observed in the previous snapshot as well")
        
    print("\n============================================================")
    print("RETRAINING RECOMMENDATION STATUS")
    print("============================================================")
    print(f"Recommend Retrain: {recommend_retrain}")
    if recommend_retrain:
        print("Reasons:")
        for r in reasons:
            print(f"  - {r}")
    else:
        print("  - Pipeline is stable. No retraining recommended.")
    print("============================================================")


if __name__ == "__main__":
    main()
