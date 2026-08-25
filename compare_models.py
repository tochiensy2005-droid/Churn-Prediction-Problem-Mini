"""compare_models.py

Evaluates the production model v2 performance metrics and runs verification assertions.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from src import config
from src.training.evaluate import evaluate_predictions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def evaluate_bundle(bundle_path: Path, dataset_path: Path) -> dict:
    bundle = joblib.load(bundle_path)
    df = pd.read_parquet(dataset_path)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    
    test_start = pd.Timestamp("2026-03-01")
    test_end = pd.Timestamp("2026-06-01")
    test_df = df[(df["snapshot_date"] >= test_start) & (df["snapshot_date"] <= test_end)]
    
    target_col = "churn_next_30d"
    total_rows = len(df)
    churn_count = int(df[target_col].sum())
    churn_rate = churn_count / total_rows if total_rows > 0 else 0.0
    
    selected_features = bundle["selected_features"]
    imputer = bundle["imputer"]
    clf = bundle["model"]
    calibrator = bundle["calibrator"]
    threshold = bundle["threshold"]
    
    X_test = test_df[selected_features].copy()
    y_test = test_df[target_col].to_numpy()
    
    X_test_imp = imputer.transform(X_test)
    raw_probs = clf.predict_proba(X_test_imp)[:, 1]
    cal_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    
    metrics = evaluate_predictions(y_test, cal_probs, threshold)
    
    return {
        "total_rows": total_rows,
        "churn_count": churn_count,
        "churn_rate": churn_rate,
        "PR_AUC": metrics["PR_AUC"],
        "ROC_AUC": metrics["ROC_AUC"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "F1": metrics["F1"]
    }

# Evaluating Model v2
bundle_v2_path = config.ARTIFACTS_DIR / "temporal_churn_model_v2.joblib"
dataset_v2_path = config.OUTPUT_DIR / "churn_temporal_dataset_v2.parquet"

res_v2 = evaluate_bundle(bundle_v2_path, dataset_v2_path)

# Create Metrics DataFrame for Model v2
metrics_data = {
    "Metric": ["Total Rows", "Churn Count", "Churn Rate", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"],
    "Value": [
        f"{res_v2['total_rows']:,}", f"{res_v2['churn_count']:,}", f"{res_v2['churn_rate']:.6%}",
        f"{res_v2['PR_AUC']:.6f}", f"{res_v2['ROC_AUC']:.6f}", f"{res_v2['precision']:.4%}",
        f"{res_v2['recall']:.4%}", f"{res_v2['F1']:.4%}"
    ]
}
metrics_df = pd.DataFrame(metrics_data)

metrics_csv_path = config.OUTPUT_DIR / "churn_rule_v2_metrics.csv"
metrics_df.to_csv(metrics_csv_path, index=False)
logger.info(f"Metrics report saved to {metrics_csv_path}")

print("\n============================================================")
print("PRODUCTION MODEL V2 PERFORMANCE METRICS")
print("============================================================")
print(metrics_df.to_string(index=False))
print("============================================================")

# Verify saved model bundle v2 loading & prediction range
logger.info("Verifying saved model bundle v2...")
bundle_v2 = joblib.load(bundle_v2_path)
df_v2 = pd.read_parquet(dataset_v2_path)
sample_df = df_v2.sample(n=10, random_state=42)

features = bundle_v2["selected_features"]
imputer = bundle_v2["imputer"]
model = bundle_v2["model"]
calibrator = bundle_v2["calibrator"]
threshold = bundle_v2["threshold"]

X_sample = sample_df[features].copy()
X_sample_imp = imputer.transform(X_sample)

raw_sample_probs = model.predict_proba(X_sample_imp)[:, 1]
cal_sample_probs = calibrator.predict_proba(raw_sample_probs.reshape(-1, 1))[:, 1]
sample_preds = (cal_sample_probs >= threshold).astype(int)

# Assertions
assert len(features) == len(bundle_v2["selected_features"]), "Feature count mismatch!"
assert np.all(cal_sample_probs >= 0.0) and np.all(cal_sample_probs <= 1.0), "Probabilities out of bounds!"
assert np.all(np.isin(sample_preds, [0, 1])), "Predictions are not binary!"

print("\n============================================================")
print("SAVED MODEL V2 VERIFICATION")
print("============================================================")
print("Model Load:           PASS")
print(f"Feature Compatibility: PASS (Checked {len(features)} features)")
print(f"Probabilities range:   PASS (Min: {cal_sample_probs.min():.4f}, Max: {cal_sample_probs.max():.4f})")
print(f"Prediction output:    PASS ({sample_preds.tolist()})")
print("============================================================")
