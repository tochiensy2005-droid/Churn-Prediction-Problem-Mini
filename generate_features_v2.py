"""generate_features_v2.py

Generates version 2 churn labels and builds features from raw Silver tables.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

from src import config
from src.features.build_temporal_base import build_temporal_base_features
from src.features.build_lag_features import build_lag_features
from src.features.build_rolling_features import build_rolling_features
from src.features.build_trend_features import build_trend_features
from src.features.build_recency_features import build_recency_features
from src.labels.build_churn_labels import build_churn_labels_v2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Building temporal feature engineering from raw Silver tables...")

# 1. Base temporal features (Monthly aggregation)
base = build_temporal_base_features()
# Let base["customer_id"] keep its natural type (int32) during feature extraction to match other Silver tables
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

# 6. Recompute churn labels using v2 rules
res_labels = build_churn_labels_v2(base)

# 7. Merge all features and labels
logger.info("Merging features and labels into final dataset...")
merged = base.merge(df_lag[["customer_id", "snapshot_date"] + lag_cols], on=["customer_id", "snapshot_date"], how="left")
merged = merged.merge(df_roll, on=["customer_id", "snapshot_date"], how="left")
merged = merged.merge(df_trend, on=["customer_id", "snapshot_date"], how="left")
merged = merged.merge(df_rec, on=["customer_id", "snapshot_date"], how="left")

# Cast merged["customer_id"] to int64 to align with res_labels["customer_id"] (which is int64)
merged["customer_id"] = merged["customer_id"].astype("int64")
df_v2 = merged.merge(res_labels, on=["customer_id", "snapshot_date"], how="inner")

# 8. Chronological Preprocessing (Median Imputation)
logger.info("Applying chronological preprocessing (median imputation strictly fit on Train)...")
train_end = pd.Timestamp("2025-08-01")
meta_cols = {
    "customer_id", "snapshot_date", "churn_next_30d",
    "rule1_closed", "rule2_downgrade_to_free_inactive", "rule3_free_at_snapshot_inactive",
    "churn_reason", "tier_at_snapshot", "closed_date", "label_complete"
}
feature_cols = [c for c in df_v2.columns if c not in meta_cols]

# Compute medians strictly on Train split (<= 2025-08-01)
train_mask = pd.to_datetime(df_v2["snapshot_date"]) <= train_end
train_medians = df_v2.loc[train_mask, feature_cols].median()

# Impute the entire merged dataset
df_v2[feature_cols] = df_v2[feature_cols].fillna(train_medians)
df_v2[feature_cols] = df_v2[feature_cols].fillna(0.0) # Fallback for any all-NaN columns on Train

# Save output parquet
output_path = config.OUTPUT_DIR / "churn_temporal_dataset_v2.parquet"
df_v2.to_parquet(output_path, index=False)
logger.info(f"Preprocessed temporal dataset v2 successfully generated at {output_path}. Shape: {df_v2.shape}")

# Save output CSV
output_csv = config.OUTPUT_DIR / "churn_temporal_dataset_v2.csv"
df_v2.to_csv(output_csv, index=False)
logger.info(f"Preprocessed temporal dataset v2 successfully saved to CSV at {output_csv}")

# ==========================================
# 9. LABEL AUDIT
# ==========================================
logger.info("Generating Label Audit Report...")
total_rows = len(res_labels)
total_churn = int(res_labels["churn_next_30d"].sum())
non_churn = total_rows - total_churn
churn_rate = total_churn / total_rows if total_rows > 0 else 0.0

r1 = res_labels["rule1_closed"] == 1
r2 = res_labels["rule2_downgrade_to_free_inactive"] == 1
r3 = res_labels["rule3_free_at_snapshot_inactive"] == 1

r1_count = int(r1.sum())
r2_count = int(r2.sum())
r3_count = int(r3.sum())

r1_only = int((r1 & ~r2 & ~r3).sum())
r2_only = int((~r1 & r2 & ~r3).sum())
r3_only = int((~r1 & ~r2 & r3).sum())

r1_r2 = int((r1 & r2 & ~r3).sum())
r1_r3 = int((r1 & ~r2 & r3).sum())
r2_r3 = int((~r1 & r2 & r3).sum())
all_3 = int((r1 & r2 & r3).sum())

audit_data = {
    "metric": [
        "Total customer-snapshots",
        "Total churn",
        "Non-churn",
        "Churn rate",
        "Rule 1 count",
        "Rule 2 count",
        "Rule 3 count",
        "Rule1 only",
        "Rule2 only",
        "Rule3 only",
        "Rule1 + Rule2",
        "Rule1 + Rule3",
        "Rule2 + Rule3",
        "All 3"
    ],
    "value": [
        total_rows,
        total_churn,
        non_churn,
        churn_rate,
        r1_count,
        r2_count,
        r3_count,
        r1_only,
        r2_only,
        r3_only,
        r1_r2,
        r1_r3,
        r2_r3,
        all_3
    ]
}
audit_df = pd.DataFrame(audit_data)
audit_csv_path = config.OUTPUT_DIR / "churn_rule_v2_audit.csv"
audit_df.to_csv(audit_csv_path, index=False)
logger.info(f"Audit report saved to {audit_csv_path}")

print("\n============================================================")
print("NEW CHURN RULE AUDIT SUMMARY")
print("============================================================")
for idx, row in audit_df.iterrows():
    val = row['value']
    if isinstance(val, float):
        print(f"{row['metric'] + ':':<35} {val:.6%}")
    else:
        print(f"{row['metric'] + ':':<35} {val:,}")
print("============================================================")

# ==========================================
# 10. LABEL DISTRIBUTION BY SNAPSHOT
# ==========================================
logger.info("Generating Label Distribution by Snapshot...")
snap_grouped = res_labels.groupby("snapshot_date").agg(
    customers=("churn_next_30d", "count"),
    rule1_count=("rule1_closed", "sum"),
    rule2_count=("rule2_downgrade_to_free_inactive", "sum"),
    rule3_count=("rule3_free_at_snapshot_inactive", "sum"),
    churn_count=("churn_next_30d", "sum")
).reset_index()
snap_grouped["churn_rate"] = snap_grouped["churn_count"] / snap_grouped["customers"]

snap_csv_path = config.OUTPUT_DIR / "churn_rule_v2_by_snapshot.csv"
snap_grouped.to_csv(snap_csv_path, index=False)
logger.info(f"Distribution by snapshot saved to {snap_csv_path}")

print("\n============================================================")
print("LABEL DISTRIBUTION BY SNAPSHOT DATE")
print("============================================================")
print(snap_grouped.to_string(index=False))
print("============================================================")
logger.info("Pipeline generate_features_v2 completed successfully.")
