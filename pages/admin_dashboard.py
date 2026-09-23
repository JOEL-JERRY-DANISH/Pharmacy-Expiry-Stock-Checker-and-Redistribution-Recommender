# pages/admin_dashboard.py
import streamlit as st
import pandas as pd
import os

# ── Authentication guard ─────────────────────────────────────────────────────
if not st.session_state.get("logged_in", False):
    st.error("🔒 You must be logged in to view this page.")
    st.info("Please return to the main page and log in.")
    st.stop()

current_user = st.session_state.get("current_user", {})
if current_user.get("branch") != "All branches":
    st.error("🚫 Access denied. This page is restricted to admin users only.")
    st.stop()
# ─────────────────────────────────────────────────────────────────────────────

st.title("📊 Admin Dashboard")
st.caption("Summary of all pharmacy stock decisions, waste reduction, and inventory expiry risk.")
st.sidebar.success(f"Logged in as: {current_user.get('name', '')}")
st.sidebar.caption(f"Branch: {current_user.get('branch', '')}")

from database import load_decisions, load_stock
from recommender import generate_recommendations, calculate_baseline, days_to_expiry, urgency_label

# Load operational decision data
df_decisions = load_decisions()
if not df_decisions.empty and "timestamp" in df_decisions.columns:
    df_decisions["timestamp"] = pd.to_datetime(df_decisions["timestamp"])
    df_decisions["date"] = df_decisions["timestamp"].dt.date

# Load operational inventory data
stock_df = load_stock()

# Compute stock risk & recommendations based on actual inventory
if not stock_df.empty:
    stock_df["dte"] = stock_df["expiry_date"].apply(days_to_expiry)
    stock_df["urgency"] = stock_df["dte"].apply(urgency_label)
    stock_df["stock_value"] = (stock_df["quantity"] * stock_df["unit_cost_gbp"]).round(2)
    at_risk_df = stock_df[stock_df["dte"].between(0, 30)]
    stock_val_at_risk = calculate_baseline(stock_df)
    recs = generate_recommendations(stock_df)
    transfer_recs = [r for r in recs if r["action"] == "TRANSFER"]
    transfer_qty = sum(r["quantity"] for r in transfer_recs)
    meds_at_risk_count = len(at_risk_df)
    unique_meds_at_risk = at_risk_df["medicine_name"].nunique() if not at_risk_df.empty else 0
    critical_count    = int((stock_df["urgency"] == "critical").sum())
    near_expiry_count = int((stock_df["urgency"] == "near-expiry").sum())
else:
    at_risk_df        = pd.DataFrame()
    stock_val_at_risk = 0.0
    transfer_qty      = 0
    meds_at_risk_count = 0
    unique_meds_at_risk = 0
    transfer_recs     = []
    critical_count    = 0
    near_expiry_count = 0

# ─────────────────────────────────────────────────────────────────────────────
# Network Health Summary
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🩺 Network Health Summary")
h1, h2, h3, h4 = st.columns(4)
h1.metric("🔴 Critical (0–7 d)",   critical_count,
          delta=f"{critical_count} batches" if critical_count else "None ✅",
          delta_color="inverse" if critical_count else "off")
h2.metric("🟠 Near-Expiry (8–30 d)", near_expiry_count,
          delta=f"{near_expiry_count} batches" if near_expiry_count else "None ✅",
          delta_color="inverse" if near_expiry_count else "off")
h3.metric("💷 Value at Risk",     f"£{stock_val_at_risk:,.2f}")
h4.metric("🚚 Qty for Transfer",  f"{transfer_qty:,} units",
          delta=f"{len(transfer_recs)} recommendations",
          delta_color="off")

# Branch-level risk table
if not at_risk_df.empty and "branch_name" in at_risk_df.columns:
    _br = (
        at_risk_df.groupby("branch_name")
        .agg(
            Batches=("batch_id", "count"),
            Value_at_Risk=("stock_value", "sum"),
        )
        .round({"Value_at_Risk": 2})
        .reset_index()
        .rename(columns={
            "branch_name":   "Branch",
            "Value_at_Risk": "Value at Risk (£)",
        })
        .sort_values("Value at Risk (£)", ascending=False)
    )
    with st.expander("🏥 Branch-Level Expiry Risk (≤30 days)", expanded=True):
        st.dataframe(
            _br.style.format({"Value at Risk (£)": "£{:,.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Decision Summary Metrics
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📋 Decision Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total decisions", len(df_decisions))
c2.metric("Confirmed transfers", len(df_decisions[df_decisions["action"] == "CONFIRMED"]) if not df_decisions.empty else 0)
c3.metric("Overrides", len(df_decisions[df_decisions["action"] == "OVERRIDDEN"]) if not df_decisions.empty else 0)
c4.metric("Manual reviews", len(df_decisions[df_decisions["action"] == "MANUALLY_REVIEWED"]) if not df_decisions.empty else 0)

st.subheader("⚠️ Inventory & Expiry Risk")
r1, r2, r3 = st.columns(3)
r1.metric("Medicines at expiry risk", f"{meds_at_risk_count} batches", f"{unique_meds_at_risk} unique medicines (≤30d)")
r2.metric("Quantity recommended for transfer", f"{transfer_qty:,} units", f"{len(transfer_recs)} recommended transfers")
r3.metric("Stock value at risk", f"£{stock_val_at_risk:,.2f}", "Expiring within 30 days")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Analytics & Trends
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📈 Analytics & Trends")

# Charts row 1: Decisions over time | Confirmed vs overridden
v_col1, v_col2 = st.columns(2)

with v_col1:
    st.subheader("📅 Decisions over time")
    if not df_decisions.empty and "date" in df_decisions.columns:
        daily = df_decisions.groupby("date").size().reset_index(name="Decisions")
        st.bar_chart(daily.set_index("date"))
    else:
        st.info("No decision records available yet.")

with v_col2:
    st.subheader("⚖️ Confirmed vs overridden")
    if not df_decisions.empty and "action" in df_decisions.columns:
        action_counts = df_decisions["action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Decisions"]
        st.bar_chart(action_counts.set_index("Action"))
    else:
        st.info("No decision records available yet.")

# Charts row 2: Transfers by destination | Expiry-risk distribution
v_col3, v_col4 = st.columns(2)

with v_col3:
    st.subheader("🚚 Transfers by destination")
    if not df_decisions.empty and "destination" in df_decisions.columns:
        dest_data = df_decisions[df_decisions["destination"] != ""]
        if not dest_data.empty:
            dest_counts = dest_data["destination"].value_counts().reset_index()
            dest_counts.columns = ["Destination Branch", "Transfers"]
            st.bar_chart(dest_counts.set_index("Destination Branch"))
        else:
            st.info("No transfers with destinations recorded yet.")
    else:
        st.info("No transfer destination records available.")

with v_col4:
    st.subheader("⏳ Expiry-risk distribution")
    if not stock_df.empty and "urgency" in stock_df.columns:
        urgency_order = ["critical", "near-expiry", "watch", "safe", "expired"]
        urg_counts = (
            stock_df["urgency"].value_counts()
            .reindex(urgency_order).fillna(0).astype(int).reset_index()
        )
        urg_counts.columns = ["Urgency Status", "Batches"]
        st.bar_chart(urg_counts.set_index("Urgency Status"))
    else:
        st.info("No inventory data available.")

# Chart row 3: Stock value at risk by branch
st.subheader("🏥 Stock value at risk by branch (£)")
if not at_risk_df.empty and "branch_name" in at_risk_df.columns:
    branch_risk = (
        at_risk_df.groupby("branch_name")["stock_value"]
        .sum().round(2).reset_index()
    )
    branch_risk.columns = ["Branch", "Value at Risk (£)"]
    st.bar_chart(branch_risk.set_index("Branch"))
else:
    st.info("No inventory currently expiring within 30 days.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Override reasons
# ─────────────────────────────────────────────────────────────────────────────
if not df_decisions.empty:
    overrides = df_decisions[df_decisions["action"] == "OVERRIDDEN"]
    if not overrides.empty:
        st.subheader("📝 Override reasons")
        cols = [c for c in [
            "timestamp", "user", "medicine", "batch_id",
            "source_branch", "destination", "quantity", "override_reason"
        ] if c in overrides.columns]
        st.dataframe(overrides[cols], use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Full decision log
# ─────────────────────────────────────────────────────────────────────────────
if not df_decisions.empty:
    with st.expander("📁 View full decision log"):
        st.dataframe(df_decisions, use_container_width=True)
        st.download_button(
            "⬇️ Download as CSV",
            df_decisions.to_csv(index=False),
            file_name="full_decision_log.csv"
        )
