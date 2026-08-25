"""src/labels/build_churn_labels.py

Computes churn target labels (churn_next_30d) based on future event logs.
"""

from __future__ import annotations

import logging
import pandas as pd
from src.data.load_silver import load_silver_table

logger = logging.getLogger(__name__)


def datetime_ns(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").astype("datetime64[ns]")


def next_event_after_snapshots(grid: pd.DataFrame, events: pd.DataFrame, date_column: str) -> pd.Series:
    left = grid[["customer_id", "snapshot_date"]].sort_values(["snapshot_date", "customer_id"])
    right = events[["customer_id", date_column]].dropna().sort_values([date_column, "customer_id"])
    left["snapshot_date"] = datetime_ns(left["snapshot_date"])
    right[date_column] = datetime_ns(right[date_column])
    joined = pd.merge_asof(
        left, right, left_on="snapshot_date", right_on=date_column,
        by="customer_id", direction="forward", allow_exact_matches=True,
    )
    return pd.Series(joined[date_column].to_numpy(), index=left.index).reindex(grid.index)


def build_churn_labels(base: pd.DataFrame) -> pd.Series:
    logger.info("Computing churn labels...")
    customers = load_silver_table("churn_customers")
    orders = load_silver_table("churn_orders")
    usage = load_silver_table("churn_product_usage")
    payments = load_silver_table("churn_payments")
    subscriptions = load_silver_table("churn_subscriptions")
    
    order_date = next(c for c in ("order_date", "created_at") if c in orders.columns)
    usage_date = next(c for c in ("event_date", "usage_date", "created_at") if c in usage.columns)
    payment_date = next(c for c in ("payment_date", "created_at") if c in payments.columns)
    subscription_date = next(c for c in ("start_date", "change_date", "created_at") if c in subscriptions.columns)
    
    orders[order_date] = pd.to_datetime(orders[order_date])
    usage[usage_date] = pd.to_datetime(usage[usage_date])
    payments[payment_date] = pd.to_datetime(payments[payment_date])
    subscriptions[subscription_date] = pd.to_datetime(subscriptions[subscription_date])
    customers["closed_date"] = pd.to_datetime(customers["closed_date"])
    
    # timezone naïve normalization
    for df in (orders, usage, payments, subscriptions, customers):
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if getattr(df[col].dt, "tz", None) is not None:
                    df[col] = df[col].dt.tz_localize(None)
                    
    order_status = next(c for c in ("status", "order_status") if c in orders.columns)
    payment_status = next(c for c in ("status", "payment_status") if c in payments.columns)
    orders["_is_completed"] = orders[order_status].astype(str).str.strip().str.lower().eq("completed").astype("int8")
    payments["_is_success"] = payments[payment_status].astype(str).str.strip().str.lower().eq("success").astype("int8")
    
    horizon = base["snapshot_date"] + pd.Timedelta(days=30)
    closed = customers[["customer_id", "closed_date"]].drop_duplicates("customer_id")
    df_labels = base.merge(closed, on="customer_id", how="left")
    
    closed_in_window = df_labels["closed_date"].ge(df_labels["snapshot_date"]) & df_labels["closed_date"].lt(horizon)
    
    subscriptions["change_type"] = subscriptions["change_type"].fillna("")
    downgrade_events = subscriptions.loc[
        subscriptions["change_type"].astype(str).str.strip().str.lower().eq("downgrade"), ["customer_id", subscription_date]
    ]
    next_downgrade = next_event_after_snapshots(base, downgrade_events, subscription_date)
    downgrade_in_window = next_downgrade.lt(horizon)
    
    completed_orders = orders.loc[orders["_is_completed"].eq(1), ["customer_id", order_date]]
    successful_payments = payments.loc[payments["_is_success"].eq(1), ["customer_id", payment_date]]
    
    activity_events = pd.concat([
        usage[["customer_id", usage_date]].rename(columns={usage_date: "activity_date"}),
        completed_orders.rename(columns={order_date: "activity_date"}),
        successful_payments.rename(columns={payment_date: "activity_date"}),
    ], ignore_index=True)
    next_activity = next_event_after_snapshots(base, activity_events, "activity_date")
    inactive_in_window = next_activity.isna() | next_activity.ge(horizon)
    
    churn_series = (closed_in_window | (downgrade_in_window & inactive_in_window)).astype("int8")
    return churn_series


def build_churn_labels_v2(base: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing Churn Labels v2...")
    customers = load_silver_table("churn_customers")
    orders = load_silver_table("churn_orders")
    usage = load_silver_table("churn_product_usage")
    payments = load_silver_table("churn_payments")
    subscriptions = load_silver_table("churn_subscriptions")
    
    # Cast customer_id to int64 for all tables to avoid merge type issues
    base = base.copy()
    for df in (orders, usage, payments, subscriptions, customers, base):
        if "customer_id" in df.columns:
            df["customer_id"] = df["customer_id"].astype("int64")
            
    order_date = next(c for c in ("order_date", "created_at") if c in orders.columns)
    usage_date = next(c for c in ("event_date", "usage_date", "created_at") if c in usage.columns)
    payment_date = next(c for c in ("payment_date", "created_at") if c in payments.columns)
    subscription_date = next(c for c in ("start_date", "change_date", "created_at") if c in subscriptions.columns)
    
    orders[order_date] = pd.to_datetime(orders[order_date])
    usage[usage_date] = pd.to_datetime(usage[usage_date])
    payments[payment_date] = pd.to_datetime(payments[payment_date])
    subscriptions[subscription_date] = pd.to_datetime(subscriptions[subscription_date])
    customers["closed_date"] = pd.to_datetime(customers["closed_date"])
    
    # timezone naïve normalization
    for df in (orders, usage, payments, subscriptions, customers, base):
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if getattr(df[col].dt, "tz", None) is not None:
                    df[col] = df[col].dt.tz_localize(None)
                    
    order_status = next(c for c in ("status", "order_status") if c in orders.columns)
    payment_status = next(c for c in ("status", "payment_status") if c in payments.columns)
    orders["_is_completed"] = orders[order_status].astype(str).str.strip().str.lower().eq("completed").astype("int8")
    payments["_is_success"] = payments[payment_status].astype(str).str.strip().str.lower().eq("success").astype("int8")
    
    # Standardize and clean subscription plan tiers
    subscriptions["plan_tier_clean"] = subscriptions["plan_tier"].astype(str).str.strip()
    def clean_tier(val):
        v = str(val).strip().lower()
        if "free" in v:
            return "Free"
        elif "premium" in v:
            return "Premium"
        elif "plus" in v:
            return "Plus"
        return val
    subscriptions["plan_tier_clean"] = subscriptions["plan_tier_clean"].apply(clean_tier)
    
    # Sort subscriptions chronologically
    subscriptions = subscriptions.sort_values(["customer_id", subscription_date, "subscription_id"])
    
    # 1. Determine tier_at_snapshot
    grid_sub = base[["customer_id", "snapshot_date"]].copy()
    grid_sub["snapshot_date"] = datetime_ns(grid_sub["snapshot_date"])
    
    sub_lookup = subscriptions[["customer_id", subscription_date, "plan_tier_clean"]].copy()
    sub_lookup[subscription_date] = datetime_ns(sub_lookup[subscription_date])
    sub_lookup = sub_lookup.sort_values([subscription_date, "customer_id"])
    
    joined_tier = pd.merge_asof(
        grid_sub.sort_values(["snapshot_date", "customer_id"]),
        sub_lookup,
        left_on="snapshot_date",
        right_on=subscription_date,
        by="customer_id",
        direction="backward"
    )
    
    joined_tier = joined_tier.set_index(["customer_id", "snapshot_date"])
    base_indexed = base.set_index(["customer_id", "snapshot_date"])
    base["tier_at_snapshot"] = base_indexed.index.map(joined_tier["plan_tier_clean"])
    
    # 2. Determine downgrade to Free in next 30 days
    subscriptions["change_type_clean"] = subscriptions["change_type"].fillna("").astype(str).str.strip().str.lower()
    downgrade_to_free_events = subscriptions[
        (subscriptions["change_type_clean"] == "downgrade") & 
        (subscriptions["plan_tier_clean"] == "Free")
    ][["customer_id", subscription_date]].rename(columns={subscription_date: "downgrade_date"})
    
    next_downgrade_to_free = next_event_after_snapshots(base, downgrade_to_free_events, "downgrade_date")
    next_downgrade_to_free = pd.to_datetime(next_downgrade_to_free)
    
    horizon = base["snapshot_date"] + pd.Timedelta(days=30)
    downgrade_to_free_in_next_30d = (
        next_downgrade_to_free.notna() & 
        next_downgrade_to_free.ge(base["snapshot_date"]) & 
        next_downgrade_to_free.lt(horizon)
    )
    
    # 3. Determine inactivity in next 30 days
    completed_orders = orders[orders["_is_completed"] == 1][["customer_id", order_date]]
    successful_payments = payments[payments["_is_success"] == 1][["customer_id", payment_date]]
    activity_events = pd.concat([
        usage[["customer_id", usage_date]].rename(columns={usage_date: "activity_date"}),
        completed_orders.rename(columns={order_date: "activity_date"}),
        successful_payments.rename(columns={payment_date: "activity_date"}),
    ], ignore_index=True)
    
    next_activity = next_event_after_snapshots(base, activity_events, "activity_date")
    next_activity = pd.to_datetime(next_activity)
    inactive_in_window = next_activity.isna() | next_activity.ge(horizon)
    
    # 4. Compute Rule Flags
    # Rule 1: Closed account in prediction window
    closed_df = customers[["customer_id", "closed_date"]].drop_duplicates("customer_id")
    df_labels = base.merge(closed_df, on="customer_id", how="left")
    rule1_closed = (
        df_labels["closed_date"].ge(df_labels["snapshot_date"]) & 
        df_labels["closed_date"].lt(horizon)
    ).astype("int8")
    
    # Rule 2: Downgrade to Free + Inactive
    rule2_downgrade_to_free_inactive = (
        (base["tier_at_snapshot"].notna()) &
        (base["tier_at_snapshot"] != "Free") &
        (downgrade_to_free_in_next_30d == True) &
        (inactive_in_window == True)
    ).astype("int8")
    
    # Rule 3: Already Free + Inactive
    rule3_free_at_snapshot_inactive = (
        (base["tier_at_snapshot"] == "Free") &
        (downgrade_to_free_in_next_30d == False) &
        (inactive_in_window == True)
    ).astype("int8")
    
    # Churn next 30d
    churn_next_30d = (
        (rule1_closed == 1) |
        (rule2_downgrade_to_free_inactive == 1) |
        (rule3_free_at_snapshot_inactive == 1)
    ).astype("int8")
    
    # Determine reason
    reasons = []
    for r1, r2, r3 in zip(rule1_closed, rule2_downgrade_to_free_inactive, rule3_free_at_snapshot_inactive):
        sub_reasons = []
        if r1 == 1:
            sub_reasons.append("Closed")
        if r2 == 1:
            sub_reasons.append("DowngradeToFree_Inactive")
        if r3 == 1:
            sub_reasons.append("FreeTier_Inactive")
        if not sub_reasons:
            reasons.append("Active")
        else:
            reasons.append("+".join(sub_reasons))
            
    res_df = pd.DataFrame({
        "customer_id": base["customer_id"],
        "snapshot_date": base["snapshot_date"],
        "churn_next_30d": churn_next_30d,
        "rule1_closed": rule1_closed,
        "rule2_downgrade_to_free_inactive": rule2_downgrade_to_free_inactive,
        "rule3_free_at_snapshot_inactive": rule3_free_at_snapshot_inactive,
        "churn_reason": reasons,
        "tier_at_snapshot": base["tier_at_snapshot"].fillna("Unknown")
    })
    return res_df
