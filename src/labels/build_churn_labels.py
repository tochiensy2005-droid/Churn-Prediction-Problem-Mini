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
