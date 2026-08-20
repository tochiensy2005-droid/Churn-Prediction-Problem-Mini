"""src/training/train_lightgbm.py

Fits the final production LightGBM model on a rolling 12-month window relative to snapshot_date.
Performs feature selection, probability calibration, threshold search,
enforces maturity guardrails, and serializes the production model bundle.
"""

from __future__ import annotations

import argparse
import logging
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import xgboost as xgb
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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


def perform_feature_selection(df: pd.DataFrame, train_mask: pd.Series, seed: int) -> list[str]:
    logger.info("Performing feature selection on Train split...")
    train_df = df[train_mask]
    
    meta_cols = ["customer_id", "snapshot_date", "churn_next_30d"]
    all_features = [c for c in df.columns if c not in meta_cols]
    
    # 1. Zero/near-zero variance check
    variances = train_df[all_features].var()
    low_var_cols = variances[variances <= 1e-5].index.tolist()
    logger.info(f"Dropped {len(low_var_cols)} features due to near-zero variance.")
    features_step1 = [f for f in all_features if f not in low_var_cols]
    
    # 2. High Correlation check (r > 0.98)
    corr_matrix = train_df[features_step1].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    for col in upper.columns:
        high_corr = upper[col][upper[col] > 0.98].index.tolist()
        if high_corr:
            col_nans = train_df[col].isna().sum()
            for hc_col in high_corr:
                hc_nans = train_df[hc_col].isna().sum()
                if col_nans > hc_nans:
                    to_drop.add(col)
                else:
                    to_drop.add(hc_col)
                    
    logger.info(f"Dropped {len(to_drop)} features due to collinearity (|r| > 0.98).")
    selected_features = [f for f in features_step1 if f not in to_drop]
    
    # 3. XGBoost Ranking strictly on Train split
    X = train_df[selected_features].copy()
    y = train_df["churn_next_30d"].to_numpy()
    
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    
    scale_weight = (len(y) - y.sum()) / y.sum()
    model = xgb.XGBClassifier(
        random_state=seed,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=4,
        scale_pos_weight=scale_weight
    )
    model.fit(X_imp, y)
    
    importances = model.feature_importances_
    rank_df = pd.DataFrame({
        "feature": selected_features,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    ranked_features = rank_df["feature"].tolist()
    top_100 = ranked_features[:100]
    logger.info(f"Feature selection complete. Kept Top 100 features out of {len(ranked_features)} candidates.")
    return top_100


def main() -> None:
    parser = argparse.ArgumentParser(description="Train production LightGBM model dynamically.")
    parser.add_argument("--snapshot-date", type=str, default="2026-06-01", help="Reference snapshot date in YYYY-MM-DD format")
    args = parser.parse_args()
    
    t = pd.Timestamp(args.snapshot_date)
    logger.info(f"Training reference snapshot date: {t.date()}")
    
    # Find maximum available event timestamp
    max_data_date = get_max_data_date()
    logger.info(f"Latest database event date detected: {max_data_date.date()}")
    
    # Determine training and calibration windows
    # Latest mature snapshot at inference time t is t - 1 month
    t_mature = t - pd.offsets.MonthBegin(1)
    
    # Calibration window: 3 months ending at t_mature
    calibration_end = t_mature
    calibration_start = t_mature - pd.offsets.MonthBegin(2)
    
    # Training window: 12 months ending right before calibration window starts
    train_end = calibration_start - pd.offsets.MonthBegin(1)
    train_start = train_end - pd.offsets.MonthBegin(11)
    
    logger.info(f"Computed dynamic timeline for snapshot {t.date()}:")
    logger.info(f"  Training Window (12M):    {train_start.date()} to {train_end.date()}")
    logger.info(f"  Calibration Window (3M): {calibration_start.date()} to {calibration_end.date()}")
    
    # Assert label maturity: calibration_end + 30 days <= max_data_date
    label_maturity_date = calibration_end + pd.Timedelta(days=30)
    logger.info(f"Label maturity guardrail check: Requires complete data up to {label_maturity_date.date()}")
    if label_maturity_date > max_data_date:
        raise ValueError(
            f"FAIL-SAFE: Cannot train model. Labels for calibration snapshot {calibration_end.date()} "
            f"are immature! Require data up to {label_maturity_date.date()} but database only has events up to {max_data_date.date()}."
        )
    logger.info("Label maturity check PASSED.")
    
    # Load dataset
    data_path = config.OUTPUT_DIR / "churn_temporal_dataset.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Temporal dataset not found at {data_path}. Please build features first.")
        
    df = pd.read_parquet(data_path)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    
    # Split
    train_mask = (df["snapshot_date"] >= train_start) & (df["snapshot_date"] <= train_end)
    val_mask = (df["snapshot_date"] >= calibration_start) & (df["snapshot_date"] <= calibration_end)
    
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError("Dynamic training split is empty. Verify that your dataset contains records for the computed dates.")
        
    seed = 42
    np.random.seed(seed)
    
    # Perform feature selection strictly on Train
    selected_features = perform_feature_selection(df, train_mask, seed)
    
    # Prepare matrices
    train_df = df[train_mask]
    val_df = df[val_mask]
    
    X_train = train_df[selected_features].copy()
    y_train = train_df["churn_next_30d"].to_numpy()
    
    X_val = val_df[selected_features].copy()
    y_val = val_df["churn_next_30d"].to_numpy()
    
    # Fit Imputer on Train
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    
    # Fit LightGBM
    logger.info("Fitting LightGBM classifier...")
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
    
    clf = lgb.LGBMClassifier(
        random_state=seed,
        scale_pos_weight=sw,
        verbose=-1,
        **hyperparams
    )
    clf.fit(X_train_imp, y_train)
    
    # Probability Calibration (Fit on Validation)
    val_probs_raw = clf.predict_proba(X_val_imp)[:, 1]
    calibrator = fit_platt_calibration(val_probs_raw, y_val)
    
    # Optimize Threshold on Calibrated Validation probabilities
    val_probs_cal = calibrator.predict_proba(val_probs_raw.reshape(-1, 1))[:, 1]
    
    best_f1 = -1.0
    best_thresh = config.DEFAULT_THRESHOLD
    from sklearn.metrics import f1_score
    
    # Sweep threshold strictly on Calibrated Validation
    for th in np.linspace(0.01, 0.99, 99):
        preds = (val_probs_cal >= th).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(th)
            
    logger.info(f"Optimal validation threshold selected: {best_thresh:.2f} (Val F1: {best_f1:.4%})")
    
    # Print metrics
    train_probs_raw = clf.predict_proba(X_train_imp)[:, 1]
    train_probs_cal = calibrator.predict_proba(train_probs_raw.reshape(-1, 1))[:, 1]
    
    train_metrics = evaluate_predictions(y_train, train_probs_cal, best_thresh)
    val_metrics = evaluate_predictions(y_val, val_probs_cal, best_thresh)
    
    # Save Model Artifact Bundle with extensive metadata
    bundle = {
        "model": clf,
        "calibrator": calibrator,
        "selected_features": selected_features,
        "imputer": imputer,
        "threshold": best_thresh,
        "training_start": str(train_start.date()),
        "training_end": str(train_end.date()),
        "calibration_start": str(calibration_start.date()),
        "calibration_end": str(calibration_end.date()),
        "validation_PR_AUC": float(val_metrics["PR_AUC"]),
        "validation_F1": float(val_metrics["F1"]),
        "random_seed": seed,
        "model_version": config.MODEL_VERSION,
        "feature_version": config.FEATURE_VERSION,
        "created_at": datetime.datetime.now().isoformat()
    }
    
    artifact_path = config.ARTIFACTS_DIR / "temporal_churn_model.joblib"
    joblib.dump(bundle, artifact_path)
    logger.info(f"Production model bundle saved to {artifact_path}")
    
    # Print report
    print("\n============================================================")
    print("FINAL PRODUCTION AUDIT REPORT: MODEL TRAINING")
    print("============================================================")
    print(f"Random Seed:          {seed}")
    print(f"Training Start:       {train_start.date()}")
    print(f"Training End:         {train_end.date()}")
    print(f"Calibration Start:    {calibration_start.date()}")
    print(f"Calibration End:      {calibration_end.date()}")
    print(f"Raw Threshold Proxy:  {config.DEFAULT_THRESHOLD}")
    print(f"Calibrated Threshold: {best_thresh:.2f}")
    print(f"Validation PR-AUC:    {val_metrics['PR_AUC']:.6f}")
    print(f"Validation ROC-AUC:   {val_metrics['ROC_AUC']:.6f}")
    print(f"Validation Precision: {val_metrics['precision']:.4%}")
    print(f"Validation Recall:    {val_metrics['recall']:.4%}")
    print(f"Validation F1:        {val_metrics['F1']:.4%}")
    print("============================================================")


if __name__ == "__main__":
    main()
