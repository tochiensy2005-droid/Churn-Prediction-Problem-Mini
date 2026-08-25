"""train_lightgbm_v2.py

Trains LightGBM model on All379 features using Churn Labels v2.
Performs probability calibration, threshold optimization, and model serialization.
"""

import logging
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score

from src import config
from src.data.load_silver import load_silver_table
from src.training.calibrate_model import fit_platt_calibration
from src.training.evaluate import evaluate_predictions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def get_max_data_date() -> pd.Timestamp:
    """Finds the maximum event date present in the raw silver tables."""
    orders = load_silver_table("churn_orders")
    order_date = next(c for c in ("order_date", "created_at") if c in orders.columns)
    max_order = pd.to_datetime(orders[order_date]).max()
    
    payments = load_silver_table("churn_payments")
    pay_date = next(c for c in ("payment_date", "created_at") if c in payments.columns)
    max_pay = pd.to_datetime(payments[pay_date]).max()
    
    max_date = max(max_order, max_pay)
    if getattr(max_date, "tz", None) is not None:
        max_date = max_date.tz_localize(None)
    return max_date


def main() -> None:
    # 1. Load dataset v2
    data_path = config.OUTPUT_DIR / "churn_temporal_dataset_v2.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Temporal dataset v2 not found at {data_path}. Please run generate_features_v2.py first.")
        
    logger.info(f"Loading v2 dataset from {data_path}...")
    df = pd.read_parquet(data_path)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    
    # Check max data date & label completeness
    max_data_date = get_max_data_date()
    logger.info(f"Latest database event date: {max_data_date.date()}")
    
    # 2. Chronological Splitting
    # Train Set (Rolling 12M): 2024-09-01 to 2025-08-01
    # Validation Set (3M calibration window + 3M validation): 2025-09-01 to 2026-02-01
    # Test Set (Clean Test): 2026-03-01 to 2026-06-01
    train_start = pd.Timestamp("2024-09-01")
    train_end = pd.Timestamp("2025-08-01")
    
    val_start = pd.Timestamp("2025-09-01")
    val_end = pd.Timestamp("2026-02-01")
    
    test_start = pd.Timestamp("2026-03-01")
    test_end = pd.Timestamp("2026-06-01")
    
    # Maturity checks
    # Each snapshot t requires t + 30 days <= max_data_date
    for name, end_date in [("Train", train_end), ("Validation", val_end), ("Test", test_end)]:
        maturity_date = end_date + pd.Timedelta(days=30)
        if maturity_date > max_data_date:
            raise ValueError(
                f"FAIL-SAFE: Cannot train model. Labels for {name} end snapshot {end_date.date()} "
                f"are immature! Require data up to {maturity_date.date()} but database only has events up to {max_data_date.date()}."
            )
    logger.info("Label maturity checks PASSED for Train, Validation, and Test splits.")
    
    train_mask = (df["snapshot_date"] >= train_start) & (df["snapshot_date"] <= train_end)
    val_mask = (df["snapshot_date"] >= val_start) & (df["snapshot_date"] <= val_end)
    test_mask = (df["snapshot_date"] >= test_start) & (df["snapshot_date"] <= test_end)
    
    train_rows = int(train_mask.sum())
    val_rows = int(val_mask.sum())
    test_rows = int(test_mask.sum())
    
    train_churn = int(df.loc[train_mask, "churn_next_30d"].sum())
    val_churn = int(df.loc[val_mask, "churn_next_30d"].sum())
    test_churn = int(df.loc[test_mask, "churn_next_30d"].sum())
    
    train_rate = train_churn / train_rows if train_rows > 0 else 0.0
    val_rate = val_churn / val_rows if val_rows > 0 else 0.0
    test_rate = test_churn / test_rows if test_rows > 0 else 0.0
    
    print("\n============================================================")
    print("CHRONOLOGICAL SPLITS DIAGNOSTICS")
    print("============================================================")
    print(f"Train Window:       {train_start.date()} to {train_end.date()}")
    print(f"Train Rows:         {train_rows:,}")
    print(f"Train Churn Count:  {train_churn:,}")
    print(f"Train Churn Rate:   {train_rate:.6%}")
    print("------------------------------------------------------------")
    print(f"Val Window:         {val_start.date()} to {val_end.date()}")
    print(f"Val Rows:           {val_rows:,}")
    print(f"Val Churn Count:    {val_churn:,}")
    print(f"Val Churn Rate:     {val_rate:.6%}")
    print("------------------------------------------------------------")
    print(f"Test Window:        {test_start.date()} to {test_end.date()}")
    print(f"Test Rows:          {test_rows:,}")
    print(f"Test Churn Count:   {test_churn:,}")
    print(f"Test Churn Rate:    {test_rate:.6%}")
    print("============================================================")
    
    # 3. Identify features (exclude all metadata, target and audit columns)
    meta_cols = {
        "customer_id", "snapshot_date", "churn_next_30d",
        "rule1_closed", "rule2_downgrade_to_free_inactive", "rule3_free_at_snapshot_inactive",
        "churn_reason", "tier_at_snapshot", "closed_date", "label_complete"
    }
    selected_features = [c for c in df.columns if c not in meta_cols and not c.startswith("future_") and not c.endswith("_future")]
    logger.info(f"Number of training features: {len(selected_features)}")
    
    # Check for any leakage columns
    leakage_cols = [c for c in selected_features if c in meta_cols or "future" in c.lower() or "rule" in c.lower()]
    if leakage_cols:
        raise ValueError(f"LEAKAGE DETECTED! The following columns must not be in features: {leakage_cols}")
        
    X_train = df.loc[train_mask, selected_features].copy()
    y_train = df.loc[train_mask, "churn_next_30d"].to_numpy()
    
    X_val = df.loc[val_mask, selected_features].copy()
    y_val = df.loc[val_mask, "churn_next_30d"].to_numpy()
    
    X_test = df.loc[test_mask, selected_features].copy()
    y_test = df.loc[test_mask, "churn_next_30d"].to_numpy()
    
    # 4. Preprocessing (Simple Median Imputation)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)
    
    # 5. Fit LightGBM Classifier
    seed = 42
    np.random.seed(seed)
    sw = (len(y_train) - y_train.sum()) / y_train.sum()
    
    hyperparams = {
        "num_leaves": 15,
        "max_depth": 4,
        "learning_rate": 0.01,
        "n_estimators": 400,
        "min_child_samples": 50,
        "feature_fraction": 0.7,
        "bagging_fraction": 1.0,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0
    }
    
    logger.info("Fitting LightGBM classifier on Train split...")
    clf = lgb.LGBMClassifier(
        random_state=seed,
        scale_pos_weight=sw,
        verbose=-1,
        **hyperparams
    )
    clf.fit(X_train_imp, y_train)
    
    # 6. Fit Platt Scaling Calibration on Validation split
    logger.info("Calibrating model on Validation split probabilities...")
    val_probs_raw = clf.predict_proba(X_val_imp)[:, 1]
    calibrator = fit_platt_calibration(val_probs_raw, y_val)
    
    # 7. Optimize threshold on calibrated Validation probabilities to maximize F1
    val_probs_cal = calibrator.predict_proba(val_probs_raw.reshape(-1, 1))[:, 1]
    
    best_f1 = -1.0
    best_thresh = config.DEFAULT_THRESHOLD
    for th in np.linspace(0.01, 0.99, 99):
        preds = (val_probs_cal >= th).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(th)
            
    logger.info(f"Optimal validation threshold selected: {best_thresh:.2f} (Val F1: {best_f1:.4%})")
    
    # Evaluate calibrated probabilities
    train_probs_raw = clf.predict_proba(X_train_imp)[:, 1]
    train_probs_cal = calibrator.predict_proba(train_probs_raw.reshape(-1, 1))[:, 1]
    
    test_probs_raw = clf.predict_proba(X_test_imp)[:, 1]
    test_probs_cal = calibrator.predict_proba(test_probs_raw.reshape(-1, 1))[:, 1]
    
    train_metrics = evaluate_predictions(y_train, train_probs_cal, best_thresh)
    val_metrics = evaluate_predictions(y_val, val_probs_cal, best_thresh)
    test_metrics = evaluate_predictions(y_test, test_probs_cal, best_thresh)
    
    # 8. Save Model Artifact Bundle (v2)
    bundle = {
        "model": clf,
        "calibrator": calibrator,
        "selected_features": selected_features,
        "imputer": imputer,
        "threshold": best_thresh,
        "business_rule_version": "v2",
        "label_definition": "v2 (Rule 1: Closed, Rule 2: Downgrade to Free + Inactive, Rule 3: Already Free + Inactive)",
        "training_start": str(train_start.date()),
        "training_end": str(train_end.date()),
        "feature_count": len(selected_features),
        "random_seed": seed,
        "created_at": datetime.datetime.now().isoformat()
    }
    
    artifact_path = config.ARTIFACTS_DIR / "temporal_churn_model_v2.joblib"
    joblib.dump(bundle, artifact_path)
    logger.info(f"Model v2 bundle successfully saved to {artifact_path}")
    
    # Print training report
    print("\n============================================================")
    print("FINAL PRODUCTION AUDIT REPORT: MODEL TRAINING (V2)")
    print("============================================================")
    print(f"Random Seed:          {seed}")
    print(f"Features Count:       {len(selected_features)}")
    print(f"Calibrated Threshold: {best_thresh:.2f}")
    print("------------------------------------------------------------")
    print("VALIDATION METRICS:")
    print(f"Validation PR-AUC:    {val_metrics['PR_AUC']:.6f}")
    print(f"Validation ROC-AUC:   {val_metrics['ROC_AUC']:.6f}")
    print(f"Validation Precision: {val_metrics['precision']:.4%}")
    print(f"Validation Recall:    {val_metrics['recall']:.4%}")
    print(f"Validation F1:        {val_metrics['F1']:.4%}")
    print(f"Validation Brier Src: {val_metrics['Brier']:.6f}")
    print("------------------------------------------------------------")
    print("CLEAN TEST METRICS:")
    print(f"Test PR-AUC:          {test_metrics['PR_AUC']:.6f}")
    print(f"Test ROC-AUC:         {test_metrics['ROC_AUC']:.6f}")
    print(f"Test Precision:       {test_metrics['precision']:.4%}")
    print(f"Test Recall:          {test_metrics['recall']:.4%}")
    print(f"Test F1:              {test_metrics['F1']:.4%}")
    print(f"Test Brier Score:     {test_metrics['Brier']:.6f}")
    conf_mtx = [[test_metrics['TN'], test_metrics['FP']], [test_metrics['FN'], test_metrics['TP']]]
    print(f"Test Confusion Mtx:\n{conf_mtx}")
    print("============================================================")


if __name__ == "__main__":
    main()
