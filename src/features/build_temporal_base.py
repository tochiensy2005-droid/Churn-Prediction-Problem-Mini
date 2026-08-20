"""src/features/build_temporal_base.py

Aggregates raw event databases into customer x monthly snapshot tables.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from src.data.load_silver import load_silver_table
from src import config

logger = logging.getLogger(__name__)


def first_existing(df: pd.DataFrame, candidates: tuple[str, ...], table: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"{table} needs one of {candidates}; found {list(df.columns)}")


def optional_existing(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((col for col in candidates if col in df.columns), None)


def monthly_aggregate(df: pd.DataFrame, date_col: str, aggs: dict) -> pd.DataFrame:
    frame = df.dropna(subset=["customer_id", date_col]).copy()
    frame["month"] = frame[date_col].dt.to_period("M").dt.to_timestamp()
    return frame.groupby(["customer_id", "month"], as_index=False).agg(**aggs)


def make_snapshot_grid(customers: pd.DataFrame, snapshot_dates: pd.DatetimeIndex) -> pd.DataFrame:
    ids = customers[["customer_id", "signup_date", "closed_date"]].drop_duplicates("customer_id")
    grid = ids.merge(pd.DataFrame({"snapshot_date": snapshot_dates}), how="cross")
    return grid.loc[
        grid["signup_date"].le(grid["snapshot_date"])
        & (grid["closed_date"].isna() | grid["closed_date"].ge(grid["snapshot_date"]))
    ].reset_index(drop=True)


def build_temporal_base_features() -> pd.DataFrame:
    logger.info("Starting base temporal aggregation...")
    
    customers = load_silver_table("churn_customers")
    orders = load_silver_table("churn_orders")
    usage = load_silver_table("churn_product_usage")
    payments = load_silver_table("churn_payments")
    subscriptions = load_silver_table("churn_subscriptions")
    tickets = load_silver_table("churn_support_tickets")
    marketing = load_silver_table("churn_marketing_interactions")
    
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    customers["closed_date"] = pd.to_datetime(customers["closed_date"])
    
    order_date = first_existing(orders, ("order_date", "created_at"), "churn_orders")
    usage_date = first_existing(usage, ("event_date", "usage_date", "created_at"), "churn_product_usage")
    payment_date = first_existing(payments, ("payment_date", "created_at"), "churn_payments")
    subscription_date = first_existing(subscriptions, ("start_date", "change_date", "created_at"), "churn_subscriptions")
    ticket_date = first_existing(tickets, ("created_at", "ticket_date", "opened_at"), "churn_support_tickets")
    marketing_date = first_existing(marketing, ("event_date", "interaction_date", "created_at", "sent_at"), "churn_marketing_interactions")
    
    orders[order_date] = pd.to_datetime(orders[order_date])
    usage[usage_date] = pd.to_datetime(usage[usage_date])
    payments[payment_date] = pd.to_datetime(payments[payment_date])
    subscriptions[subscription_date] = pd.to_datetime(subscriptions[subscription_date])
    tickets[ticket_date] = pd.to_datetime(tickets[ticket_date])
    marketing[marketing_date] = pd.to_datetime(marketing[marketing_date])
    
    # Timezone normalization
    for df in (orders, usage, payments, subscriptions, tickets, marketing):
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if getattr(df[col].dt, "tz", None) is not None:
                    df[col] = df[col].dt.tz_localize(None)
                    
    order_status = first_existing(orders, ("status", "order_status"), "churn_orders")
    payment_status = first_existing(payments, ("status", "payment_status"), "churn_payments")
    
    orders["_is_completed"] = orders[order_status].astype(str).str.strip().str.lower().eq("completed").astype("int8")
    payments["_is_success"] = payments[payment_status].astype(str).str.strip().str.lower().eq("success").astype("int8")
    payments["_is_failure"] = payments[payment_status].astype(str).str.strip().str.lower().isin({"failed", "failure", "declined"}).astype("int8")
    
    subscriptions["change_type"] = subscriptions["change_type"].fillna("")
    subscriptions["_is_downgrade"] = subscriptions["change_type"].astype(str).str.strip().str.lower().eq("downgrade").astype("int8")
    subscriptions["_is_upgrade"] = subscriptions["change_type"].astype(str).str.strip().str.lower().eq("upgrade").astype("int8")
    
    event_dates = pd.concat([orders[order_date], usage[usage_date], payments[payment_date], subscriptions[subscription_date], tickets[ticket_date], marketing[marketing_date]]).dropna()
    min_snapshot = event_dates.min().to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
    max_snapshot = (event_dates.max() - pd.Timedelta(days=30)).to_period("M").to_timestamp()
    
    snapshots = pd.date_range(min_snapshot, max_snapshot, freq="MS")
    grid = make_snapshot_grid(customers, snapshots)
    
    order_amount = optional_existing(orders, ("amount", "order_amount", "total_amount", "total"))
    order_aggs = {"orders": ("customer_id", "size"), "completed_orders": ("_is_completed", "sum")}
    if order_amount:
        order_aggs["spend"] = (order_amount, "sum")
        
    usage_monthly = monthly_aggregate(usage, usage_date, {"usage": ("customer_id", "size"), "active_days": (usage_date, "nunique")})
    order_monthly = monthly_aggregate(orders, order_date, order_aggs)
    payment_monthly = monthly_aggregate(payments, payment_date, {"payment_count": ("customer_id", "size"), "payment_success": ("_is_success", "sum"), "payment_failure": ("_is_failure", "sum")})
    
    ticket_aggs = {"support_ticket": ("customer_id", "size")}
    csat = optional_existing(tickets, ("csat", "csat_score", "satisfaction_score"))
    if csat:
        ticket_aggs["csat"] = (csat, "mean")
    ticket_monthly = monthly_aggregate(tickets, ticket_date, ticket_aggs)
    
    marketing_click = optional_existing(marketing, ("clicked", "is_clicked"))
    marketing_type = optional_existing(marketing, ("interaction_type", "event_type", "action", "type"))
    if marketing_click:
        marketing["_is_click"] = marketing[marketing_click].fillna(False).astype(bool).astype("int8")
        marketing_monthly = monthly_aggregate(marketing, marketing_date, {"marketing_interaction": ("customer_id", "size"), "marketing_click": ("_is_click", "sum")})
    elif marketing_type:
        marketing["_is_click"] = marketing[marketing_type].astype(str).str.strip().str.lower().eq("click").astype("int8")
        marketing_monthly = monthly_aggregate(marketing, marketing_date, {"marketing_interaction": ("customer_id", "size"), "marketing_click": ("_is_click", "sum")})
    else:
        marketing_monthly = monthly_aggregate(marketing, marketing_date, {"marketing_interaction": ("customer_id", "size")})
        
    subscription_monthly = monthly_aggregate(subscriptions, subscription_date, {"downgrade": ("_is_downgrade", "sum"), "upgrade": ("_is_upgrade", "sum"), "subscription_change": ("customer_id", "size")})
    
    base = grid.drop(columns=["signup_date", "closed_date"]).copy()
    for table in (order_monthly, usage_monthly, payment_monthly, ticket_monthly, marketing_monthly, subscription_monthly):
        base["month"] = base["snapshot_date"] - pd.offsets.MonthBegin(1)
        base = base.merge(table, on=["customer_id", "month"], how="left")
        
    behavioral = [col for col in base.columns if col not in {"customer_id", "signup_date", "closed_date", "snapshot_date", "month"}]
    base[behavioral] = base[behavioral].fillna(0.0)
    
    base["payment_success_rate"] = np.where(base["payment_count"].gt(0), base["payment_success"] / base["payment_count"], np.nan)
    if "marketing_click" in base:
        base["marketing_click_rate"] = np.where(base["marketing_interaction"].gt(0), base["marketing_click"] / base["marketing_interaction"], np.nan)
        
    base = base.drop(columns=["month"])
    base.to_parquet(config.OUTPUT_DIR / "temporal_base.parquet", index=False)
    logger.info(f"Temporal base parquet written to {config.OUTPUT_DIR / 'temporal_base.parquet'}. Shape: {base.shape}")
    return base
