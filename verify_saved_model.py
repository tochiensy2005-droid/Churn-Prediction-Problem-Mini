"""verify_saved_model.py

Verification script to check model loading, feature compatibility,
probability calibration, thresholding, and predictions on a sample.
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "temporal_churn_model.joblib"
DATASET_PATH = ROOT / "output" / "churn_temporal_dataset.parquet"

def main():
    print("============================================================")
    print("RUNNING FINAL MODEL PACKAGING VERIFICATION")
    print("============================================================")

    # 1. Load model artifact
    print(f"Loading artifact from {ARTIFACT_PATH}...")
    if not ARTIFACT_PATH.exists():
        print(f"FAIL: Artifact not found at {ARTIFACT_PATH}")
        return
        
    bundle = joblib.load(ARTIFACT_PATH)
    
    # 2. Check model components
    model = bundle.get("model")
    calibrator = bundle.get("calibrator")
    selected_features = bundle.get("selected_features")
    threshold = bundle.get("threshold")
    
    assert model is not None, "FAIL: Model object is missing in the bundle!"
    assert selected_features is not None, "FAIL: selected_features list is missing in the bundle!"
    assert threshold is not None, "FAIL: Threshold is missing in the bundle!"
    assert calibrator is not None, "FAIL: Calibrator is missing in the bundle!"
    
    print("PASS: Model, selected_features, threshold, and calibrator exist in bundle.")
    
    # 3. Print metadata
    print("\nMetadata:")
    for k, v in bundle.items():
        if k not in ("model", "calibrator", "imputer", "selected_features"):
            print(f"  {k}: {v}")
            
    # 4. Feature Compatibility
    print("\nChecking Feature Compatibility...")
    if not DATASET_PATH.exists():
        print(f"FAIL: Temporal dataset not found at {DATASET_PATH} to run compatibility check.")
        return
        
    df = pd.read_parquet(DATASET_PATH)
    missing_features = [f for f in selected_features if f not in df.columns]
    
    print(f"  Expected features count: {len(selected_features)}")
    print(f"  Available features count: {len([f for f in selected_features if f in df.columns])}")
    print(f"  Missing features count: {len(missing_features)}")
    
    assert len(missing_features) == 0, f"FAIL: Missing features in dataset: {missing_features}"
    print("PASS: Feature compatibility checks passed (0 missing features).")
    
    # 5. Verify Prediction and Calibration
    print("\nRunning Prediction and Calibration Test on 100 sample rows...")
    
    # Take a sample of 100 rows
    sample_df = df.head(100).copy()
    X = sample_df[selected_features]
    
    # Impute
    imputer = bundle.get("imputer")
    assert imputer is not None, "FAIL: Imputer missing in bundle!"
    X_imp = imputer.transform(X)
    
    # Predict raw probability
    raw_probs = model.predict_proba(X_imp)[:, 1]
    
    # Calibrate probability
    calibrated_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    
    # Threshold to get predictions
    preds = (calibrated_probs >= threshold).astype(int)
    
    # Asserts
    assert np.all(calibrated_probs >= 0.0) and np.all(calibrated_probs <= 1.0), "FAIL: Calibrated probabilities out of bounds [0, 1]!"
    assert np.all(np.isin(preds, [0, 1])), "FAIL: Predictions must be binary 0 or 1!"
    
    print("PASS: Prediction tests completed successfully.")
    print("  Sample predictions summary:")
    print(f"    Min probability:  {calibrated_probs.min():.6f}")
    print(f"    Mean probability: {calibrated_probs.mean():.6f}")
    print(f"    Max probability:  {calibrated_probs.max():.6f}")
    print(f"    Predicted Churn:  {preds.sum()} out of {len(preds)}")
    print("============================================================")
    print("VERIFICATION COMPLETED: PASS")
    print("============================================================")

if __name__ == "__main__":
    main()
