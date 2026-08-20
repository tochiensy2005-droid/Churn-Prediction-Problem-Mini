"""src/features/build_recency_features.py

Generates recency statistics (days since last event) using chronological joins.
"""

from __future__ import annotations

import logging
import pandas as pd
from src.data.load_silver import load_silver_table

logger = logging.getLogger(__name__)


def datetime_ns(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").astype("datetime64[ns]")


def build_recency_features(base: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing recency features...")
    orders = load_silver_table("churn_orders")
    usage = load_silver_table("churn_product_usage")
    payments = load_silver_table("churn_payments")
    subscriptions = load_silver_table("churn_subscriptions")
    tickets = load_silver_table("churn_support_tickets")
    
    order_date = next(c for c in ("order_date", "created_at") if c in orders.columns)
    usage_date = next(c for c in ("event_date", "usage_date", "created_at") if c in usage.columns)
    payment_date = next(c for c in ("payment_date", "created_at") if c in payments.columns)
    subscription_date = next(c for c in ("start_date", "change_date", "created_at") if c in subscriptions.columns)
    ticket_date = next(c for c in ("created_at", "ticket_date", "opened_at") if c in tickets.columns)
    
    orders[order_date] = pd.to_datetime(orders[order_date])
    usage[usage_date] = pd.to_datetime(usage[usage_date])
    payments[payment_date] = pd.to_datetime(payments[payment_date])
    subscriptions[subscription_date] = pd.to_datetime(subscriptions[subscription_date])
    tickets[ticket_date] = pd.to_datetime(tickets[ticket_date])
    
    # timezone naïve normalization
    for df in (orders, usage, payments, subscriptions, tickets):
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if getattr(df[col].dt, "tz", None) is not None:
                    df[col] = df[col].dt.tz_localize(None)
                    
    # Identify downgrades
    change_type = next(c for c in ("change_type", "type") if c in subscriptions.columns)
    subscriptions["_is_downgrade"] = subscriptions[change_type].astype(str).str.strip().str.lower().eq("downgrade").astype("int8")
    downgrades = subscriptions.loc[subscriptions["_is_downgrade"].eq(1)].copy()
    
    events = {
        "usage": (usage, usage_date),
        "order": (orders, order_date),
        "payment": (payments, payment_date),
        "support_ticket": (tickets, ticket_date),
        "downgrade": (downgrades, subscription_date),
    }
    
    result = base[["customer_id", "snapshot_date"]].copy()
    left = result.sort_values(["snapshot_date", "customer_id"])
    left["snapshot_date"] = datetime_ns(left["snapshot_date"])
    
    for feature, (events_df, date_col) in events.items():
        right = events_df[["customer_id", date_col]].dropna().sort_values([date_col, "customer_id"])
        right[date_col] = datetime_ns(right[date_col])
        joined = pd.merge_asof(
            left, right, left_on="snapshot_date", right_on=date_col,
            by="customer_id", direction="backward", allow_exact_matches=False,
        )
        values = (joined["snapshot_date"] - joined[date_col]).dt.days.to_numpy()
        result[f"days_since_last_{feature}"] = pd.Series(values, index=left.index).reindex(result.index)
        
    return result
