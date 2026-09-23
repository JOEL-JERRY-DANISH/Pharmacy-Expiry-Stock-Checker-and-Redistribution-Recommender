import streamlit as st
import pandas as pd
import hashlib
from datetime import datetime, timedelta
from recommender import generate_recommendations, calculate_baseline
from log_manager import save_entry, load_log, get_summary
from auth_config import CREDENTIALS, authenticate_user

st.set_page_config(
    page_title="Pharmacy Expiry Stock Checker",
    page_icon="💊",
    layout="wide"
)

# ============================================================
# LOGIN SECTION — paste everything below here, after imports
# ============================================================

def check_password(username, password):
    """Check username and password against stored credentials."""
    user = authenticate_user(username, password)
    if user:
        return True, user
    return False, None


def _credentials_configured() -> bool:
    """Return True if at least one user has a non-empty password hash."""
    users = CREDENTIALS["usernames"]
    return any(u.get("password", "") for u in users.values())


# --- Login screen ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in  = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "login_time" not in st.session_state:
    st.session_state.login_time = None

if not st.session_state.logged_in:
    st.title("💊 Pharmacy Stock Checker")
    st.subheader("Please log in to continue")

    # Warn the operator when no passwords have been configured at all.
    if not _credentials_configured():
        st.warning(
            "⚠️ **Authentication is not configured.** "
            "No password hashes were found in `.streamlit/secrets.toml` "
            "or environment variables.  "
            "Copy `.streamlit/secrets.toml.example` to "
            "`.streamlit/secrets.toml` and fill in the SHA-256 hashes, "
            "or set the `PHARMACIST1_PASSWORD` / `PHARMACIST2_PASSWORD` / "
            "`ADMIN_PASSWORD` environment variables."
        )

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")

        if submitted:
            valid, user_info = check_password(username, password)
            if valid:
                st.session_state.logged_in    = True
                st.session_state.current_user = user_info
                st.session_state.login_time   = datetime.now()
                st.rerun()
            else:
                st.error("❌ Incorrect username or password. Please try again.")
    st.stop()

# --- Session timeout (8 hours) ---
if st.session_state.login_time is not None:
    elapsed = datetime.now() - st.session_state.login_time
    if elapsed > timedelta(hours=8):
        st.session_state.logged_in    = False
        st.session_state.current_user = None
        st.session_state.login_time   = None
        st.warning("⏰ Your session has expired. Please log in again.")
        st.rerun()

# --- Show logged-in user ---
user = st.session_state.current_user
if user is not None:
    st.sidebar.success(f"Logged in as: {user['name']}")
    st.sidebar.caption(f"Branch: {user['branch']}")

if st.sidebar.button("Log Out"):
    st.session_state.logged_in    = False
    st.session_state.current_user = None
    st.session_state.login_time   = None
    st.session_state.pop("confirmed", None)
    st.session_state.pop("overridden", None)
    st.session_state.pop("barcode_registry", None)
    st.rerun()

st.title("💊 Pharmacy Expiry Stock Checker")
st.caption(
    "This tool finds medicines that are about to expire and "
    "recommends which branch to send them to before they are wasted."
)

from database import load_stock, initialise_database, save_decision, invalidate_stock_cache
initialise_database()

def dual_save(batch_id, medicine, action,
              destination="", override_reason="", user="",
              source_branch="", quantity=0, system_recommendation=""):
    """Record an operational decision in SQLite (the sole authoritative audit log)."""
    save_entry(batch_id=batch_id, medicine=medicine, action=action,
               destination=destination, override_reason=override_reason,
               user=user, source_branch=source_branch, quantity=quantity,
               system_recommendation=system_recommendation,
               final_decision=action)
    invalidate_stock_cache()

def load_data(force_reload=False):
    """
    Load live operational stock data directly from SQLite.
    Never returns stale cached inventory.
    """
    if force_reload:
        invalidate_stock_cache()
    return load_stock()

# Provide .clear method for backwards compatibility with st.cache_data usage
load_data.clear = invalidate_stock_cache

raw_df = load_data()

# --- Sidebar ---
st.sidebar.header("🔍 Filter Stock")
branches = ["All branches"] + \
           sorted(raw_df["branch_name"].unique().tolist())
selected = st.sidebar.selectbox("Show stock from:", branches)
if selected != "All branches":
    df = raw_df[raw_df["branch_name"] == selected]
else:
    df = raw_df

# --- Session state — restored from persistent log on first load ---
if "log" not in st.session_state:
    st.session_state.log = []
if "confirmed" not in st.session_state or "overridden" not in st.session_state:
    # Load once and derive both sets to avoid calling load_log() twice.
    _existing_log = load_log()
    _confirmed_actions = {"CONFIRMED", "MANUALLY_REVIEWED"}
    if "confirmed" not in st.session_state:
        st.session_state.confirmed = set(
            _existing_log.loc[
                _existing_log["action"].isin(_confirmed_actions), "batch_id"
            ].tolist()
        ) if not _existing_log.empty else set()
    if "overridden" not in st.session_state:
        st.session_state.overridden = set(
            _existing_log.loc[
                _existing_log["action"] == "OVERRIDDEN", "batch_id"
            ].tolist()
        ) if not _existing_log.empty else set()

# --- Generate recommendations ---
# Pass raw_df as all_df so destination branches across the network are always discovered
recs         = generate_recommendations(df, all_df=raw_df)
baseline     = calculate_baseline(df)
transfer_val = sum(r["stock_value"] for r in recs
                   if r["action"] == "TRANSFER")

# --- Email alert section ---
from alert_manager import send_alert

critical_recs = [r for r in recs if r["urgency"] == "critical"]
if critical_recs:
    st.sidebar.warning(
        f"⚠️ {len(critical_recs)} critical items found"
    )
    if st.sidebar.button("📧 Send Email Alert"):
        success = send_alert(critical_recs)
        if success:
            st.sidebar.success("Email sent successfully!")
        else:
            st.sidebar.error(
                "Email failed. Check alert_manager.py settings."
            )
# --- Email alert section ends here ---

if not recs:
    st.success("✅ No near-expiry stock needs action today.")
    st.stop()

# --- Summary metrics ---
critical = sum(1 for r in recs if r["urgency"] == "critical")
near_exp = sum(1 for r in recs if r["urgency"] == "near-expiry")
total_val = sum(r["stock_value"] for r in recs)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🔴 Urgent (0–7 days)",     critical,
          delta=f"{critical} need immediate action" if critical else None,
          delta_color="inverse")
c2.metric("🟠 Act Soon (8–30 days)",  near_exp,
          delta=f"{near_exp} batches" if near_exp else None,
          delta_color="off")
c3.metric("💷 Total Stock at Risk",   f"£{total_val:,.2f}")
c4.metric("✅ Covered by Transfers",  f"£{transfer_val:,.2f}")

# --- Expiry risk distribution chart ---
_urgency_order = ["critical", "near-expiry", "watch"]
_urg_counts = (
    raw_df.copy()
    .assign(dte=raw_df["expiry_date"].apply(
        lambda d: (pd.Timestamp(d).date() - pd.Timestamp.today().date()).days
    ))
    .assign(urgency=lambda x: x["dte"].map(
        lambda d: "critical" if d <= 7 else ("near-expiry" if d <= 30 else "watch")
    ))
    [lambda x: x["urgency"].isin(_urgency_order)]
    .groupby("urgency")
    .size()
    .reindex(_urgency_order)
    .fillna(0)
    .astype(int)
    .reset_index()
)
_urg_counts.columns = ["Urgency", "Batches"]
if not _urg_counts.empty and _urg_counts["Batches"].sum() > 0:
    with st.expander("📊 Expiry Risk Distribution", expanded=True):
        st.bar_chart(_urg_counts.set_index("Urgency"), height=200)

# --- Branch-level risk summary ---
_branch_risk = (
    raw_df.copy()
    .assign(dte=raw_df["expiry_date"].apply(
        lambda d: (pd.Timestamp(d).date() - pd.Timestamp.today().date()).days
    ))
    .assign(stock_value=raw_df["quantity"] * raw_df["unit_cost_gbp"])
    .pipe(lambda x: x[x["dte"].between(0, 30)])
    .groupby("branch_name")
    .agg(
        at_risk_batches=("batch_id", "count"),
        at_risk_value=("stock_value", "sum"),
    )
    .round({"at_risk_value": 2})
    .reset_index()
    .rename(columns={
        "branch_name": "Branch",
        "at_risk_batches": "At-Risk Batches (≤30d)",
        "at_risk_value": "Value at Risk (£)",
    })
    .sort_values("Value at Risk (£)", ascending=False)
)
if not _branch_risk.empty:
    with st.expander("🏥 Branch-Level Risk Summary", expanded=False):
        st.dataframe(
            _branch_risk.style.format({"Value at Risk (£)": "£{:,.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

# --- Colour guide ---
st.markdown("""
**Colour guide — colour AND text used together:**
🔴 **Red** = act within 7 days |
🟠 **Orange** = act within 30 days |
🔵 **Blue** = cannot match automatically, review manually
""")

# --- Evaluation panel ---
with st.expander("📊 View Evaluation Results"):
    improvement = ((transfer_val / baseline) * 100
                   ) if baseline > 0 else 0
    st.markdown(f"""
| Metric | Value |
|---|---|
| Baseline stock at risk | £{baseline:,.2f} |
| Covered by transfer recommendations | £{transfer_val:,.2f} |
| Improvement over baseline | {improvement:.1f}% |
| Target | 60% |
| Target achieved | {'✅ YES' if improvement >= 60 else '❌ Not yet'} |
    """)

st.divider()

# --- ADD THE BARCODE SCANNER CODE HERE ---
st.subheader("🔍 Scan a Barcode")
st.caption("Type or scan a barcode to instantly look up a medicine batch.")

barcode_input = st.text_input(
    "Enter barcode number:",
    placeholder="e.g. 5000112345678",
    key="barcode_scan"
)

if barcode_input.strip():
    from barcode_lookup import lookup_barcode
    from barcode_registry import BarcodeRegistry

    # Cache the registry for the session — avoids reading CSV on every scan
    if "barcode_registry" not in st.session_state:
        st.session_state.barcode_registry = BarcodeRegistry()

    result = lookup_barcode(barcode_input.strip(),
                            registry=st.session_state.barcode_registry)

    if result["found"]:
        urgency = result["urgency"]
        if urgency == "critical":
            st.error(
                f"🔴 URGENT — {result['medicine_name']} | "
                f"Expires in {result['dte']} days | "
                f"Worth £{result['stock_value']:.2f}"
            )
        elif urgency == "near-expiry":
            st.warning(
                f"🟠 ACT SOON — {result['medicine_name']} | "
                f"Expires in {result['dte']} days | "
                f"Worth £{result['stock_value']:.2f}"
            )
        elif urgency == "expired":
            st.error(
                f"❌ EXPIRED — {result['medicine_name']} | "
                f"Expired {abs(result['dte'])} days ago"
            )
        else:
            st.success(
                f"✅ SAFE — {result['medicine_name']} | "
                f"Expires in {result['dte']} days"
            )

        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Batch:** {result['batch_id']}")
        col2.markdown(f"**Branch:** {result['branch_name']}")
        col3.markdown(f"**Quantity:** {result['quantity']} units")

        col4, col5 = st.columns(2)
        col4.markdown(f"**Risk Score:** {result.get('score', 0)}")
        col5.markdown(f"**Stock Value:** £{result.get('stock_value', 0):.2f}")

        if result.get("recommendation"):
            rec = result["recommendation"]
            if rec["action"] == "TRANSFER":
                st.info(f"💡 **Recommended Action:** Transfer {rec['suggested_quantity']} units to **{rec['destination_branch']}**.")
            else:
                st.info(f"💡 **Recommended Action:** Flag for review — {rec['reason']}")

        if result["status"] == "superseded":
            st.warning(
                "⚠️ This barcode was updated. "
                "The system resolved it to the current batch."
            )
    else:
        st.error(f"❌ {result['message']}")

st.divider()
# --- BARCODE SCANNER CODE ENDS HERE ---

st.subheader(f"📋 {len(recs)} items need attention")

# --- Render each recommendation ---
for i, rec in enumerate(recs):
    uid = rec["batch_id"]
    already_confirmed  = uid in st.session_state.confirmed
    already_overridden = uid in st.session_state.overridden

    urgency = rec["urgency"]
    if urgency == "critical":
        border_color = "#E24B4A"
        label        = "🔴 URGENT"
    elif urgency == "near-expiry":
        border_color = "#F5A623"
        label        = "🟠 ACT SOON"
    else:
        border_color = "#4A90D9"
        label        = "🔵 REVIEW"

    st.markdown(
        f"<div style='border-left:6px solid {border_color};"
        f"padding:12px;margin-bottom:4px;border-radius:4px'>",
        unsafe_allow_html=True
    )

    # Plain English header
    if urgency == "critical":
        st.error(
            f"{label} — {rec['medicine_name']} | "
            f"{rec['quantity']} units | "
            f"Expires in {rec['dte']} days | "
            f"Worth £{rec['stock_value']:.2f}"
        )
    elif urgency == "near-expiry":
        st.warning(
            f"{label} — {rec['medicine_name']} | "
            f"{rec['quantity']} units | "
            f"Expires in {rec['dte']} days | "
            f"Worth £{rec['stock_value']:.2f}"
        )
    else:
        st.info(
            f"{label} — {rec['medicine_name']} | "
            f"{rec['quantity']} units | "
            f"Expires in {rec['dte']} days"
        )

    # Details row
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.markdown(f"**Batch:** {rec['batch_id']}")
    col_b.markdown(f"**Branch:** {rec['branch_name']}")
    col_c.markdown(f"**Expiry:** {rec['expiry_date']}")
    col_d.markdown(f"**Confidence:** {rec['confidence']}")

    # Reason — always visible
    st.markdown(f"**Why:** {rec['reason']}")

    if "ml_risk_class" in rec:
        prob_pct = int(rec.get("ml_risk_probability", 0.0) * 100)
        ml_badge = "🔴" if rec["ml_risk_class"] == "High" else ("🟡" if rec["ml_risk_class"] == "Medium" else "🟢")
        st.caption(f"{ml_badge} **ML Expiry Risk Assessment:** {rec['ml_risk_class']} ({prob_pct}% probability)")

    if "decision_factors" in rec:
        with st.expander("📊 View Decision Factors"):
            df_fac = rec["decision_factors"]
            fcol1, fcol2, fcol3, fcol4 = st.columns(4)
            fcol1.markdown(f"• **Days to expiry:** {df_fac['days_to_expiry']}d")
            fcol2.markdown(f"• **Current stock:** {df_fac['current_stock']} units")
            fcol3.markdown(f"• **Source demand:** {df_fac['demand']} units/wk")
            dest_dem_str = f"{df_fac['destination_demand']} units/wk" if df_fac['destination_demand'] is not None else "N/A"
            fcol4.markdown(f"• **Dest demand:** {dest_dem_str}")

            fcol5, fcol6, fcol7, fcol8 = st.columns(4)
            cap_str = f"{df_fac['available_capacity']} units" if df_fac['available_capacity'] is not None else "N/A"
            fcol5.markdown(f"• **Capacity:** {cap_str}")
            fcol6.markdown(f"• **Transfer time:** ~{df_fac['transfer_time_days']}d")
            fcol7.markdown(f"• **Medicine value:** £{df_fac['medicine_value_gbp']:,.2f}")
            fcol8.markdown(f"• **Risk score:** {df_fac['risk_score']}")

            if "score_components" in df_fac:
                sc = df_fac["score_components"]
                st.caption(
                    f"🎯 **Score Breakdown:** Urgency: {sc['urgency_score']:.1f} pts | "
                    f"Quantity Volume: {sc['quantity_score']:.1f} pts | "
                    f"Financial Exposure (£{sc['stock_value']:,.2f}): {sc['value_score']:.1f} pts"
                )

            if "ml_risk_class" in df_fac:
                fcol9, fcol10 = st.columns(2)
                fcol9.markdown(f"• **ML Risk Class:** {df_fac['ml_risk_class']}")
                fcol10.markdown(f"• **ML Risk Probability:** {int(df_fac['ml_risk_probability'] * 100)}%")

    # Action buttons
    if already_confirmed:
        st.success("✅ Transfer confirmed for this session.")
    elif already_overridden:
        st.info("↩️ This recommendation was overridden.")
    elif rec["action"] == "TRANSFER" and rec["destinations"]:
        is_split = rec.get("is_split", False)
        split_dests = rec.get("split_destinations", [])
        if not split_dests:
            split_dests = [d for d in rec["destinations"] if d.get("transfer_quantity", 0) > 0]
            is_split = len(split_dests) > 1

        if is_split:
            st.markdown("🔀 **Split Redistribution Plan (Multi-Branch Allocation):**")
            split_cols_header = st.columns([3, 2, 2, 2, 2])
            split_cols_header[0].markdown("**Destination**")
            split_cols_header[1].markdown("**Transfer Qty**")
            split_cols_header[2].markdown("**Weekly Demand**")
            split_cols_header[3].markdown("**Current Cover**")
            split_cols_header[4].markdown("**Capacity**")

            for sd in split_dests:
                scols = st.columns([3, 2, 2, 2, 2])
                bname = sd.get("branch_name") or sd.get("dest_branch_name")
                tqty = sd.get("transfer_quantity", 0)
                dem = sd.get("demand") if sd.get("demand") is not None else sd.get("dest_demand_per_week", 0)
                woc = sd.get("weeks_of_cover") if sd.get("weeks_of_cover") is not None else sd.get("dest_weeks_of_cover", 0.0)
                cap = sd.get("capacity") if sd.get("capacity") is not None else sd.get("dest_capacity", 0)

                scols[0].markdown(f"🏥 **{bname}**")
                scols[1].markdown(f"**{tqty} units**")
                scols[2].markdown(f"{dem} /wk")
                scols[3].markdown(f"{woc} wks")
                scols[4].markdown(f"{cap} units")

            dest_display = ", ".join(f"{sd.get('branch_name') or sd.get('dest_branch_name')} ({sd.get('transfer_quantity', 0)}u)" for sd in split_dests)
        else:
            best      = rec["destinations"][0]
            dest_name = best["dest_branch_name"]
            dest_display = dest_name
            st.markdown(
                f"**Suggested transfer:** {rec['quantity']} units "
                f"→ **{dest_name}**"
            )

        if rec["requires_confirmation"]:
            st.caption(
                "⚠️ This is a large transfer (high-impact). "
                "Please confirm or override below."
            )
            btn_col, reason_col = st.columns([1, 2])
            with btn_col:
                confirm_label = "✅ Confirm Split Transfer" if is_split else "✅ Confirm Transfer"
                if st.button(confirm_label,
                             key=f"confirm_{i}",
                             type="primary"):
                    st.session_state.confirmed.add(uid)
                    if is_split:
                        for sd in split_dests:
                            bname = sd.get("branch_name") or sd.get("dest_branch_name")
                            tqty = sd.get("transfer_quantity", 0)
                            dual_save(
                                batch_id=uid,
                                medicine=rec["medicine_name"],
                                action="CONFIRMED",
                                destination=f"{bname} ({tqty} units)",
                                override_reason="",
                                user=st.session_state.current_user["name"],
                                source_branch=rec.get("branch_name", ""),
                                quantity=tqty,
                                system_recommendation="TRANSFER (SPLIT)",
                            )
                    else:
                        dual_save(
                            batch_id=uid,
                            medicine=rec["medicine_name"],
                            action="CONFIRMED",
                            destination=dest_display,
                            override_reason="",
                            user=st.session_state.current_user["name"],
                            source_branch=rec.get("branch_name", ""),
                            quantity=rec.get("quantity", 0),
                            system_recommendation=rec.get("action", "TRANSFER"),
                        )
                    st.rerun()
            with reason_col:
                reason_text = st.text_input(
                    "Override reason (required before rejecting):",
                    key=f"reason_{i}",
                    placeholder="e.g. Branch already has enough stock"
                )
                if st.button("↩️ Override / Reject",
                             key=f"override_{i}"):
                    if reason_text.strip():
                        st.session_state.overridden.add(uid)
                        dual_save(
                            batch_id=uid,
                            medicine=rec["medicine_name"],
                            action="OVERRIDDEN",
                            destination=dest_display,
                            override_reason=reason_text,
                            user=st.session_state.current_user["name"],
                            source_branch=rec.get("branch_name", ""),
                            quantity=rec.get("quantity", 0),
                            system_recommendation=rec.get("action", "TRANSFER"),
                        )
                        st.rerun()
                    else:
                        st.error(
                            "Please type a reason before overriding."
                        )
        else:
            action_button_label = f"✅ Confirm Split Transfer ({len(split_dests)} branches)" if is_split else f"✅ Transfer to {dest_display}"
            if st.button(action_button_label,
                         key=f"go_{i}"):
                st.session_state.confirmed.add(uid)
                if is_split:
                    for sd in split_dests:
                        bname = sd.get("branch_name") or sd.get("dest_branch_name")
                        tqty = sd.get("transfer_quantity", 0)
                        dual_save(
                            batch_id=uid,
                            medicine=rec["medicine_name"],
                            action="CONFIRMED",
                            destination=f"{bname} ({tqty} units)",
                            override_reason="",
                            user=st.session_state.current_user["name"],
                            source_branch=rec.get("branch_name", ""),
                            quantity=tqty,
                            system_recommendation="TRANSFER (SPLIT)",
                        )
                else:
                    dual_save(
                        batch_id=uid,
                        medicine=rec["medicine_name"],
                        action="CONFIRMED",
                        destination=dest_display,
                        override_reason="",
                        user=st.session_state.current_user["name"],
                        source_branch=rec.get("branch_name", ""),
                        quantity=rec.get("quantity", 0),
                        system_recommendation=rec.get("action", "TRANSFER"),
                    )
                st.rerun()

    elif rec["action"] == "FLAG_FOR_REVIEW":
        st.warning(
            "⚠️ Could not automatically match this item to a "
            "receiving branch. Please review manually."
        )
        if st.button("📋 Mark as Reviewed", key=f"review_{i}"):
            st.session_state.confirmed.add(uid)
            dual_save(
                batch_id=uid,
                medicine=rec["medicine_name"],
                action="MANUALLY_REVIEWED",
                destination="",
                override_reason="",
                user=st.session_state.current_user["name"],
                source_branch=rec.get("branch_name", ""),
                quantity=rec.get("quantity", 0),
                system_recommendation=rec.get("action", "FLAG_FOR_REVIEW"),
            )
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()

# --- Decision log (loaded from SQLite — the sole authoritative audit log) ---
st.subheader("📁 Decision Log (Audit Trail)")
_persistent_log = load_log()
if not _persistent_log.empty:
    st.dataframe(_persistent_log, use_container_width=True)
    st.download_button(
        "⬇️ Download Decision Log as CSV",
        _persistent_log.to_csv(index=False),
        file_name="decision_log.csv",
        mime="text/csv"
    )
else:
    st.caption(
        "No decisions recorded yet. "
        "Confirm or override a recommendation above."
    )
