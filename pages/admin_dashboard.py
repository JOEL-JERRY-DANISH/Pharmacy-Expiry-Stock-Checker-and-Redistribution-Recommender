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

from ui_theme import (
    inject_custom_css,
    render_page_header,
    render_sidebar_account,
    render_sidebar_nav,
    render_section_header,
    render_empty_state,
)

inject_custom_css()

render_sidebar_account(current_user)
render_sidebar_nav(current_page="admin_dashboard", user=current_user)

render_page_header(
    "Admin Dashboard",
    "Operational overview of pharmacy stock decisions, waste reduction, and inventory expiry risk.",
    icon="📊",
    context_info="🛡️ Scope: All Branches",
)

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
# Network Health Summary (Executive KPI Banner)
# ─────────────────────────────────────────────────────────────────────────────
render_section_header("Network Health Summary", "Executive real-time inventory and expiry exposure across all branches", icon="🩺")
h1, h2, h3, h4 = st.columns(4)
h1.metric(
    "🔴 Critical (0–7 days)",
    critical_count,
    delta=f"{critical_count} batches require immediate action" if critical_count else "Zero Critical ✅",
    delta_color="inverse" if critical_count else "off"
)
h2.metric(
    "🟠 Near-Expiry (8–30 days)",
    near_expiry_count,
    delta=f"{near_expiry_count} batches in transfer window" if near_expiry_count else "Zero Imminent ✅",
    delta_color="inverse" if near_expiry_count else "off"
)
h3.metric(
    "💷 Stock Value at Risk",
    f"£{stock_val_at_risk:,.2f}",
    delta=f"{meds_at_risk_count} batches ({unique_meds_at_risk} unique meds)" if meds_at_risk_count else "Zero Exposure ✅",
    delta_color="off"
)
h4.metric(
    "🚚 Recommended Transfer Qty",
    f"{transfer_qty:,} units",
    delta=f"{len(transfer_recs)} recommended transfers",
    delta_color="off"
)

# ─────────────────────────────────────────────────────────────────────────────
# Prominent Network Warnings & Alerts
# ─────────────────────────────────────────────────────────────────────────────
if critical_count > 0:
    st.error(
        f"🚨 **Critical Expiry Alert:** {critical_count} inventory batches are expiring within 7 days. "
        f"Immediate dispensary dispatch or express redistribution transfer is required to prevent stock write-off."
    )
elif near_expiry_count > 0:
    st.warning(
        f"⚠️ **Near-Expiry Warning:** {near_expiry_count} batches are within 8–30 days of expiration, "
        f"representing £{stock_val_at_risk:,.2f} in financial exposure across {unique_meds_at_risk} unique medicines."
    )
else:
    st.success(
        "✅ **Network Status Optimal:** No inventory batches across the pharmacy network are currently within 30 days of expiration."
    )

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
            "Batches":       "At-Risk Batches (≤30d)",
        })
        .sort_values("Value at Risk (£)", ascending=False)
    )
    with st.expander("🏥 Branch-Level Expiry Risk Breakdown (≤30 days)", expanded=True):
        st.dataframe(
            _br.style.format({
                "Value at Risk (£)": "£{:,.2f}",
                "At-Risk Batches (≤30d)": "{:,}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Network exposure total: £{stock_val_at_risk:,.2f} across {len(_br)} branches.")
else:
    render_empty_state("No Imminent Expiry Risk", "No batches across the network are currently within 30 days of expiry.", "✅")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Decision Summary Metrics & Operational Governance
# ─────────────────────────────────────────────────────────────────────────────
render_section_header("Operational Governance & Decision Audit", "Executive overview of confirmed, overridden, and manually reviewed actions", icon="📋")

total_decisions = len(df_decisions)
confirmed_count = len(df_decisions[df_decisions["action"] == "CONFIRMED"]) if not df_decisions.empty else 0
overrides_count = len(df_decisions[df_decisions["action"] == "OVERRIDDEN"]) if not df_decisions.empty else 0
manual_count    = len(df_decisions[df_decisions["action"] == "MANUALLY_REVIEWED"]) if not df_decisions.empty else 0

conf_delta = f"{int(confirmed_count / total_decisions * 100)}% confirmed" if total_decisions > 0 else None
ov_delta = f"{int(overrides_count / total_decisions * 100)}% override rate" if total_decisions > 0 else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Decisions Logged", total_decisions)
c2.metric("Confirmed Transfers", confirmed_count, delta=conf_delta, delta_color="normal")
c3.metric("Pharmacist Overrides", overrides_count, delta=ov_delta, delta_color="off")
c4.metric("Manual Reviews", manual_count, delta=f"{manual_count} items reviewed" if manual_count else None, delta_color="off")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Analytics & Trends (Organized in Logical Tabs)
# ─────────────────────────────────────────────────────────────────────────────
render_section_header("Analytics & Regional Trends", "Visual distribution of operational decisions, actions, and regional transfer routes", icon="📈")

tab_decisions, tab_regional, tab_distribution = st.tabs([
    "📊 Decisions & Operational Audit",
    "🏥 Regional Exposure & Transfers",
    "⏳ Expiry Risk Distribution",
])

with tab_decisions:
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.markdown("##### 📅 Decisions Over Time")
        if not df_decisions.empty and "date" in df_decisions.columns:
            daily = df_decisions.groupby("date").size().reset_index(name="Decisions")
            st.bar_chart(daily.set_index("date"), height=260)
            st.caption("Daily volume of confirmed, overridden, and reviewed decisions.")
        else:
            render_empty_state("No Decisions Recorded Yet", "Operational activity will display here once decisions are recorded.", "📅")

    with v_col2:
        st.markdown("##### ⚖️ Confirmed vs Overridden Actions")
        if not df_decisions.empty and "action" in df_decisions.columns:
            action_counts = df_decisions["action"].value_counts().reset_index()
            action_counts.columns = ["Action", "Decisions"]
            st.bar_chart(action_counts.set_index("Action"), height=260)
            st.caption("Distribution of confirmed transfers, overrides, and manual reviews.")
        else:
            render_empty_state("No Action Data", "Action distribution will display here once decisions are recorded.", "⚖️")

with tab_regional:
    v_col3, v_col4 = st.columns(2)
    with v_col3:
        st.markdown("##### 🏥 Stock Value at Risk by Branch (£)")
        if not at_risk_df.empty and "branch_name" in at_risk_df.columns:
            branch_risk = (
                at_risk_df.groupby("branch_name")["stock_value"]
                .sum().round(2).reset_index()
            )
            branch_risk.columns = ["Branch", "Value at Risk (£)"]
            st.bar_chart(branch_risk.set_index("Branch"), height=260)
            st.caption("Financial valuation of inventory expiring within 30 days per branch.")
        else:
            render_empty_state("Zero Regional Risk", "No inventory is currently expiring within 30 days.", "🏥")

    with v_col4:
        st.markdown("##### 🚚 Transfers by Destination Branch")
        if not df_decisions.empty and "destination" in df_decisions.columns:
            dest_data = df_decisions[df_decisions["destination"] != ""]
            if not dest_data.empty:
                dest_counts = dest_data["destination"].value_counts().reset_index()
                dest_counts.columns = ["Destination Branch", "Transfers"]
                st.bar_chart(dest_counts.set_index("Destination Branch"), height=260)
                st.caption("Volume of confirmed transfers received by each destination branch.")
            else:
                render_empty_state("No Transfers Logged", "Confirmed redistribution destinations will be visualized here.", "🚚")
        else:
            render_empty_state("No Destination Records", "No transfer destination records available in audit log.", "🚚")

with tab_distribution:
    st.markdown("##### ⏳ Network Expiry-Risk Distribution")
    if not stock_df.empty and "urgency" in stock_df.columns:
        urgency_order = ["critical", "near-expiry", "watch", "safe", "expired"]
        urg_counts = (
            stock_df["urgency"].value_counts()
            .reindex(urgency_order).fillna(0).astype(int).reset_index()
        )
        urg_counts.columns = ["Urgency Status", "Batches"]
        st.bar_chart(urg_counts.set_index("Urgency Status"), height=280)
        st.caption("Batch count across semantic risk tiers: Critical (≤7d), Near-Expiry (8–30d), Watch (31–60d), Safe (>60d).")
    else:
        render_empty_state("No Inventory Data", "Inventory stock data is required to calculate expiry urgency distribution.", "⏳")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Override reasons
# ─────────────────────────────────────────────────────────────────────────────
if not df_decisions.empty:
    overrides = df_decisions[df_decisions["action"] == "OVERRIDDEN"]
    if not overrides.empty:
        render_section_header("Override Reasons", "Clinical justifications logged by pharmacists when overriding recommendations", icon="📝", badge_text=f"{len(overrides)} Overrides")
        cols = [c for c in [
            "timestamp", "user", "medicine", "batch_id",
            "source_branch", "destination", "quantity", "reason_code", "override_reason"
        ] if c in overrides.columns]
        renamed_override_cols = {
            "timestamp": "Timestamp",
            "user": "Pharmacist User",
            "medicine": "Medicine",
            "batch_id": "Batch ID",
            "source_branch": "Source Branch",
            "destination": "Destination",
            "quantity": "Quantity",
            "reason_code": "Reason Code",
            "override_reason": "Clinical Justification",
        }
        filtered_overrides = overrides
        if "reason_code" in overrides.columns:
            codes = sorted([c for c in overrides["reason_code"].dropna().unique() if c])
            if len(codes) > 1:
                filter_col, _ = st.columns([1, 2])
                with filter_col:
                    selected_code = st.selectbox("Filter by Reason Code", ["All Reason Codes"] + codes, key="admin_filter_reason_code")
                    if selected_code != "All Reason Codes":
                        filtered_overrides = overrides[overrides["reason_code"] == selected_code]

        st.dataframe(
            filtered_overrides[cols].rename(columns={k: v for k, v in renamed_override_cols.items() if k in cols}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        render_section_header("Override Reasons", "Clinical justifications logged by pharmacists when overriding recommendations", icon="📝")
        render_empty_state("Zero Overrides Logged", "All processed recommendations were accepted without overrides.", "🛡️")

# ─────────────────────────────────────────────────────────────────────────────
# Full decision log
# ─────────────────────────────────────────────────────────────────────────────
if not df_decisions.empty:
    with st.expander("📁 View Complete Decision Audit Log", expanded=False):
        st.dataframe(df_decisions, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download Decision Log as CSV",
            df_decisions.to_csv(index=False),
            file_name="full_decision_log.csv",
            mime="text/csv",
        )
