"""src/training/train_processed_xgb.py

Trains a GPU-accelerated XGBoost classifier on the preprocessed churn dataset.
Performs chronological splitting, Platt probability calibration, threshold search,
and serializes the production model bundle.
"""

from __future__ import annotations

import logging
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import f1_score

from src import config
from src.training.calibrate_model import fit_platt_calibration
from src.training.evaluate import evaluate_predictions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    # 1. Load dataset
    data_path = Path(config.DATA_DIR) / "churn_feature_dataset_processed.csv"
    logger.info(f"Loading processed dataset from {data_path}...")
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    logger.info(f"Dataset loaded. Shape: {df.shape}")
    
    # 2. Chronological Splitting
    # Train Set: snapshot_month <= "2025-08"
    # Validation Set: "2025-09" <= snapshot_month <= "2026-02"
    # Test Set: snapshot_month >= "2026-03"
    train_mask = df["snapshot_month"] <= "2025-08"
    val_mask = (df["snapshot_month"] >= "2025-09") & (df["snapshot_month"] <= "2026-02")
    test_mask = df["snapshot_month"] >= "2026-03"
    
    train_count = train_mask.sum()
    val_count = val_mask.sum()
    test_count = test_mask.sum()
    
    logger.info(f"Data Split:")
    logger.info(f"  Train:      {train_count} rows")
    logger.info(f"  Validation: {val_count} rows")
    logger.info(f"  Test:       {test_count} rows")
    
    if train_count == 0 or val_count == 0 or test_count == 0:
        raise ValueError("One of the splits is empty. Verify snapshot_month values.")
        
    # 3. Identify features
    meta_cols = ["customer_id", "snapshot_month", "snapshot_month_ord", "label_churn"]
    selected_features = [c for c in df.columns if c not in meta_cols]
    logger.info(f"Number of training features: {len(selected_features)}")
    
    # Extract matrices
    X_train = df.loc[train_mask, selected_features].copy()
    y_train = df.loc[train_mask, "label_churn"].to_numpy()
    
    X_val = df.loc[val_mask, selected_features].copy()
    y_val = df.loc[val_mask, "label_churn"].to_numpy()
    
    X_test = df.loc[test_mask, selected_features].copy()
    y_test = df.loc[test_mask, "label_churn"].to_numpy()
    
    # 4. Train XGBoost on GPU
    logger.info("Initializing GPU XGBoost Classifier...")
    seed = 42
    sw = (len(y_train) - y_train.sum()) / y_train.sum()
    
    clf = xgb.XGBClassifier(
        random_state=seed,
        n_estimators=400,
        max_depth=4,
        learning_rate=0.01,
        scale_pos_weight=sw,
        tree_method="hist",
        device="cuda",
        eval_metric="logloss"
    )
    
    logger.info("Fitting XGBoost model on GPU...")
    clf.fit(X_train, y_train)
    logger.info("Model fitting complete.")
    
    # 5. Probability Calibration (Fit Platt Scaling on Validation raw probs)
    logger.info("Running probability calibration on validation raw probabilities...")
    val_probs_raw = clf.predict_proba(X_val)[:, 1]
    calibrator = fit_platt_calibration(val_probs_raw, y_val)
    
    # 6. Optimize Threshold on Calibrated Validation probabilities
    val_probs_cal = calibrator.predict_proba(val_probs_raw.reshape(-1, 1))[:, 1]
    
    best_f1 = -1.0
    best_thresh = config.DEFAULT_THRESHOLD
    
    # Sweep threshold strictly on Calibrated Validation
    for th in np.linspace(0.01, 0.99, 99):
        preds = (val_probs_cal >= th).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(th)
            
    logger.info(f"Optimal validation threshold selected: {best_thresh:.2f} (Val F1: {best_f1:.4%})")
    
    # 7. Evaluate on Validation & Test
    train_probs_raw = clf.predict_proba(X_train)[:, 1]
    train_probs_cal = calibrator.predict_proba(train_probs_raw.reshape(-1, 1))[:, 1]
    
    test_probs_raw = clf.predict_proba(X_test)[:, 1]
    test_probs_cal = calibrator.predict_proba(test_probs_raw.reshape(-1, 1))[:, 1]
    
    train_metrics = evaluate_predictions(y_train, train_probs_cal, best_thresh)
    val_metrics = evaluate_predictions(y_val, val_probs_cal, best_thresh)
    test_metrics = evaluate_predictions(y_test, test_probs_cal, best_thresh)
    
    # 8. Save Model Artifact Bundle
    bundle = {
        "model": clf,
        "calibrator": calibrator,
        "selected_features": selected_features,
        "threshold": best_thresh,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "random_seed": seed,
        "created_at": datetime.datetime.now().isoformat()
    }
    
    artifact_path = config.ARTIFACTS_DIR / "processed_churn_model.joblib"
    joblib.dump(bundle, artifact_path)
    logger.info(f"Production model bundle saved to {artifact_path}")
    
    # 9. Audit Report
    print("\n============================================================")
    print("PROCESSED DATASET GPU TRAINING REPORT: XGBOOST")
    print("============================================================")
    print(f"Random Seed:            {seed}")
    print(f"Features trained:       {len(selected_features)}")
    print(f"Train row count:        {train_count}")
    print(f"Val row count:          {val_count}")
    print(f"Test row count:         {test_count}")
    print(f"Optimized Threshold:    {best_thresh:.2f}")
    print("------------------------------------------------------------")
    print("VALIDATION METRICS:")
    print(f"  PR-AUC:               {val_metrics['PR_AUC']:.6f}")
    print(f"  ROC-AUC:              {val_metrics['ROC_AUC']:.6f}")
    print(f"  Precision:            {val_metrics['precision']:.4%}")
    print(f"  Recall:               {val_metrics['recall']:.4%}")
    print(f"  F1-Score:             {val_metrics['F1']:.4%}")
    print("------------------------------------------------------------")
    print("TEST METRICS:")
    print(f"  PR-AUC:               {test_metrics['PR_AUC']:.6f}")
    print(f"  ROC-AUC:              {test_metrics['ROC_AUC']:.6f}")
    print(f"  Precision:            {test_metrics['precision']:.4%}")
    print(f"  Recall:               {test_metrics['recall']:.4%}")
    print(f"  F1-Score:             {test_metrics['F1']:.4%}")
    print("============================================================")


if __name__ == "__main__":
    main()
