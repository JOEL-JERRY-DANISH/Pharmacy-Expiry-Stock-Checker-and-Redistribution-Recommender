import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from recommender import generate_recommendations, calculate_baseline
from log_manager import load_log
from auth_config import CREDENTIALS, authenticate_user

st.set_page_config(
    page_title="Pharmacy Expiry Stock Checker",
    page_icon="💊",
    layout="wide"
)

from ui_theme import (
    inject_custom_css,
    render_page_header,
    render_sidebar_account,
    render_sidebar_nav,
    render_section_header,
    get_urgency_theme,
    render_empty_state,
    format_urgency_badge,
    format_shelf_life,
    format_risk_badge,
)

inject_custom_css()

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
    render_page_header(
        "Pharmacy Stock Checker",
        "Clinical decision support system — Please log in to continue",
        icon="💊"
    )

    # Warn the operator when no passwords have been configured at all.
    if not _credentials_configured():
        st.warning(
            "⚠️ **Authentication is not configured.** "
            "No password hashes were found in `.streamlit/secrets.toml` "
            "or environment variables.  "
            "Copy `.streamlit/secrets.toml.example` to "
            "`.streamlit/secrets.toml` and fill in the PBKDF2-HMAC-SHA256 password hashes, "
            "or set the `PHARMACIST1_PASSWORD` / `PHARMACIST2_PASSWORD` / "
            "`ADMIN_PASSWORD` environment variables."
        )

    with st.form("login_form"):
        st.markdown("#### 🔐 Authorized Sign-In")
        st.caption("Enter your assigned pharmacy credentials to access expiry monitoring and redistribution tools.")
        username = st.text_input(
            "Username *",
            placeholder="e.g. pharmacist_central or admin",
            help="Clinical staff username assigned by Central Office."
        )
        password = st.text_input(
            "Password *",
            type="password",
            placeholder="Enter secure password",
            help="Case-sensitive account password."
        )
        submitted = st.form_submit_button("Sign In to Portal", type="primary", use_container_width=True)

        if submitted:
            valid, user_info = check_password(username, password)
            if valid:
                st.session_state.logged_in    = True
                st.session_state.current_user = user_info
                st.session_state.login_time   = datetime.now()
                st.rerun()
            else:
                st.error("❌ Authentication Failed: Incorrect username or password. Please verify your credentials and try again.")
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

# --- Show logged-in user & sidebar navigation ---
user = st.session_state.current_user
if user is not None:
    render_sidebar_account(user)
    render_sidebar_nav(current_page="recommender", user=user)

context_badge = f"📍 Scope: {user.get('branch', 'All branches')}" if user else None
render_page_header(
    "Pharmacy Expiry Stock Checker",
    "Identify near-expiry inventory and allocate redistribution transfers across branches before stock is wasted.",
    icon="💊",
    context_info=context_badge,
)

from database import (
    load_stock,
    initialise_database,
    invalidate_stock_cache,
    save_decision,
    update_stock_quantity,
    DatabaseSaveError,
    validate_override,
    OVERRIDE_REASON_CODES,
)
initialise_database()

DECISION_SAVE_ERROR_MESSAGE = (
    "The decision could not be saved. The action was not confirmed. "
    "Please try again."
)

def dual_save(batch_id, medicine, action,
              destination="", override_reason="", user="",
              source_branch="", quantity=0, system_recommendation="",
              reason_code=""):
    """Record an operational decision in SQLite (the sole authoritative audit log)."""
    ok = save_decision(
        batch_id=batch_id,
        medicine=medicine,
        action=action,
        destination=destination,
        override_reason=override_reason,
        user=user,
        source_branch=source_branch,
        quantity=quantity,
        system_recommendation=system_recommendation,
        final_decision=action,
        reason_code=reason_code,
    )
    if ok:
        invalidate_stock_cache()
    return bool(ok)

def record_recommendation_action(rec, action, user_name, destination="",
                                 override_reason="", reason_code="", is_split=False,
                                 split_dests=None) -> bool:
    """
    Centralize recording of operational decisions (single or split transfers,
    overrides, or manual reviews) to SQLite audit log and invalidate stock cache.
    Returns True if all decision records were persisted successfully, False otherwise.
    """
    uid = rec["batch_id"]
    source_branch = rec.get("branch_name", "")
    medicine = rec["medicine_name"]
    sys_rec = rec.get("action", "TRANSFER")

    if action == "CONFIRMED" and is_split and split_dests:
        # Note on split-transfer architecture limitation:
        # Currently, database.save_decision() commits each branch allocation
        # sequentially in individual SQLite transactions. If one destination record
        # succeeds and a subsequent destination record fails, the prior record remains
        # committed. The operation is not fully atomic across all split destinations.
        # To handle this safely, we stop processing on the first failure and return False,
        # ensuring the batch is NOT marked as confirmed in st.session_state.
        for sd in split_dests:
            bname = sd.get("branch_name") or sd.get("dest_branch_name")
            tqty = sd.get("transfer_quantity", 0)
            ok = dual_save(
                batch_id=uid,
                medicine=medicine,
                action="CONFIRMED",
                destination=f"{bname} ({tqty} units)",
                override_reason="",
                user=user_name,
                source_branch=source_branch,
                quantity=tqty,
                system_recommendation="TRANSFER (SPLIT)",
            )
            if not ok:
                return False
        return True
    else:
        return dual_save(
            batch_id=uid,
            medicine=medicine,
            action=action,
            destination=destination,
            override_reason=override_reason,
            user=user_name,
            source_branch=source_branch,
            quantity=rec.get("quantity", 0),
            system_recommendation=sys_rec,
            reason_code=reason_code,
        )

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
st.sidebar.markdown("<div class='sidebar-section-title'>🔍 Filter Inventory</div>", unsafe_allow_html=True)
branches = ["All branches"] + \
           sorted(raw_df["branch_name"].unique().tolist())
selected = st.sidebar.selectbox("Show stock from:", branches)
if selected != "All branches":
    df = raw_df[raw_df["branch_name"] == selected]
else:
    df = raw_df

search_filter = st.sidebar.text_input(
    "Search Medicine / Batch:",
    placeholder="e.g. Amoxicillin, BATCH-101",
    key="sidebar_search_filter"
)
status_options = [
    "All Statuses",
    "🔴 Urgent (0–7 days)",
    "🟠 Act Soon (8–30 days)",
    "🟡 Watch (31–60 days)",
    "🟢 Safe (>60 days)",
    "❌ Expired (<0 days)",
]
selected_status_filter = st.sidebar.selectbox(
    "Filter by Expiry Status:",
    status_options,
    index=0,
    key="sidebar_status_filter"
)

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
    st.sidebar.markdown("<div class='sidebar-section-title'>🔔 Notifications</div>", unsafe_allow_html=True)
    st.sidebar.warning(
        f"⚠️ {len(critical_recs)} critical items found"
    )
    if st.sidebar.button("📧 Send Email Alert", use_container_width=True):
        success = send_alert(critical_recs)
        if success:
            st.sidebar.success("Email sent successfully!")
        else:
            st.sidebar.error(
                "Email failed. Check alert_manager.py settings."
            )
# --- Email alert section ends here ---

# --- Recommendations check ---
if not recs:
    st.info("ℹ️ No near-expiry stock currently requires immediate redistribution in this scope.")

# --- Summary metrics ---
render_section_header("Network Overview", "High-level inventory status for the selected scope", icon="📊")
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

# --- Clinical Expiry & Risk Framework Guide ---
with st.expander("ℹ️ Clinical Expiry Horizons & Risk Assessment Guide", expanded=False):
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.markdown("""
**Operational Expiry Horizons (Dual-Encoded Semantic Tiers):**
- 🔴 **Critical / Urgent (0–7 days):** Immediate dispensation priority or express transfer.
- 🟠 **Warning / Act Soon (8–30 days):** Active redistribution window to prevent stock expiration.
- 🟡 **Watch (31–60 days):** Approaching horizon; monitoring branch dispensary velocity.
- 🟢 **Safe (>60 days):** Routine shelf life; standard inventory circulation.
- 🔵 **Manual Review:** Automatic transfer match unavailable; clinical pharmacist review required.
- ⚪ **Unavailable / Error:** Advisory feature data unavailable or parsing error.
""")
    with gcol2:
        st.markdown("""
**Risk Assessment & Advisory Signal Framework:**
- **Deterministic Heuristic Score (0–100+):** Objective operational score combining:
  1. *Urgency Factor:* Proximity to expiration.
  2. *Quantity Volume Factor:* Unit count requiring redistribution.
  3. *Financial Exposure Factor:* Monetary value at risk of expiration (£).
- **Supplementary Advisory ML Signal:** Random Forest statistical prediction estimating expiry wastage probability based on inventory attributes.
  *(Advisory only — does not supersede clinical judgment or rule-based safety criteria).*
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

# --- INVENTORY DIRECTORY & EXPLORER ---
st.divider()
render_section_header(
    "Inventory Directory",
    "Comprehensive branch stock batches with real-time quantities, shelf life, and risk status",
    icon="📦",
    badge_text=f"{len(df)} Batches Total"
)

# Compute enriched stock view
today = pd.Timestamp.today().date()
inv_df = df.copy()
inv_df["dte"] = inv_df["expiry_date"].apply(
    lambda d: (pd.Timestamp(d).date() - today).days
)
inv_df["stock_value"] = (inv_df["quantity"] * inv_df["unit_cost_gbp"]).round(2)

def _calc_status_badge(days: int) -> str:
    if days < 0:
        return "❌ Expired"
    elif days <= 7:
        return "🔴 Critical (≤7d)"
    elif days <= 30:
        return "🟠 Act Soon (8–30d)"
    elif days <= 60:
        return "🟡 Watch (31–60d)"
    else:
        return "🟢 Safe (>60d)"

inv_df["stock_status"] = inv_df["dte"].apply(_calc_status_badge)
inv_df["shelf_life"] = inv_df["dte"].apply(format_shelf_life)

# Filter by search_filter if provided
filtered_inv = inv_df
if search_filter.strip():
    q = search_filter.strip().lower()
    filtered_inv = filtered_inv[
        filtered_inv["medicine_name"].astype(str).str.lower().str.contains(q) |
        filtered_inv["batch_id"].astype(str).str.lower().str.contains(q)
    ]

# Filter by selected_status_filter if provided
if selected_status_filter != "All Statuses":
    if "Critical" in selected_status_filter:
        filtered_inv = filtered_inv[(filtered_inv["dte"] >= 0) & (filtered_inv["dte"] <= 7)]
    elif "Act Soon" in selected_status_filter:
        filtered_inv = filtered_inv[(filtered_inv["dte"] > 7) & (filtered_inv["dte"] <= 30)]
    elif "Watch" in selected_status_filter:
        filtered_inv = filtered_inv[(filtered_inv["dte"] > 30) & (filtered_inv["dte"] <= 60)]
    elif "Safe" in selected_status_filter:
        filtered_inv = filtered_inv[filtered_inv["dte"] > 60]
    elif "Expired" in selected_status_filter:
        filtered_inv = filtered_inv[filtered_inv["dte"] < 0]

if filtered_inv.empty:
    render_empty_state(
        "No Matching Inventory Batches",
        "No inventory records match your current branch, search query, or status filter.",
        "🔍"
    )
else:
    display_cols = [
        "stock_status",
        "medicine_name",
        "batch_id",
        "branch_name",
        "quantity",
        "expiry_date",
        "shelf_life",
        "unit_cost_gbp",
        "stock_value",
    ]
    renamed_cols = {
        "stock_status": "Risk Level",
        "medicine_name": "Medicine",
        "batch_id": "Batch ID",
        "branch_name": "Branch Location",
        "quantity": "Quantity (Units)",
        "expiry_date": "Expiry Date",
        "shelf_life": "Shelf Life Remaining",
        "unit_cost_gbp": "Unit Cost (£)",
        "stock_value": "Stock Value (£)",
    }
    view_table = filtered_inv[display_cols].rename(columns=renamed_cols)
    st.dataframe(
        view_table.style.format({
            "Quantity (Units)": "{:,}",
            "Unit Cost (£)": "£{:,.2f}",
            "Stock Value (£)": "£{:,.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

# --- STOCK LEVEL MANAGEMENT (COUNT RECONCILIATION) ---
with st.expander("✏️ Update Batch Stock Quantity (Physical Count Reconciliation)", expanded=False):
    st.markdown("Adjust recorded stock quantities in SQLite following a physical stock count or manual dispensing.")
    u_col1, u_col2 = st.columns([2, 1])
    with u_col1:
        batch_options = filtered_inv["batch_id"].tolist() if not filtered_inv.empty else inv_df["batch_id"].tolist()
        batch_to_update = st.selectbox(
            "Select Batch to Update *",
            options=batch_options,
            key="stock_update_batch_select",
            help="Select the batch identifier verified during the physical inventory count."
        ) if batch_options else None
    with u_col2:
        current_batch_qty = 0
        if batch_to_update:
            matched_rows = inv_df[inv_df["batch_id"] == batch_to_update]
            if not matched_rows.empty:
                current_batch_qty = int(matched_rows.iloc[0]["quantity"])
                med_name = matched_rows.iloc[0]["medicine_name"]
                st.caption(f"**Medicine:** {med_name} | **Current Recorded:** {current_batch_qty} units")

        new_stock_qty = st.number_input(
            "New Physical Stock Quantity (units) *",
            min_value=0,
            step=1,
            value=current_batch_qty,
            key="stock_update_new_qty",
            help="Enter the verified physical count on hand."
        )

    if st.button("💾 Save Updated Stock Quantity", type="primary", key="btn_confirm_stock_update"):
        if not batch_to_update:
            st.error("⚠️ Validation Error: Please select a valid batch identifier to update.")
        else:
            try:
                updated = update_stock_quantity(batch_to_update, new_stock_qty)
                if updated:
                    st.success(f"✅ Batch '{batch_to_update}' stock quantity successfully updated to {new_stock_qty} units.")
                    st.rerun()
                else:
                    st.error(f"❌ Batch Not Found: Batch '{batch_to_update}' was not found in the database.")
            except ValueError as ve:
                st.error(f"❌ Validation Error: {ve}")
            except DatabaseSaveError as dbe:
                st.error(f"❌ Database Save Error: {dbe}")
            except Exception as ex:
                st.error(f"❌ Unexpected Error: {ex}")

st.divider()

# --- ADD THE BARCODE SCANNER CODE HERE ---
render_section_header("Scan Barcode", "Type or scan a barcode to instantly look up a medicine batch", icon="🔍")

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
        urg_badge = format_urgency_badge(urgency)
        if urgency == "critical":
            st.error(
                f"{urg_badge} — {result['medicine_name']} | "
                f"{result['quantity']} units | "
                f"Expires in {result['dte']} days | "
                f"Worth £{result['stock_value']:.2f}"
            )
        elif urgency == "near-expiry":
            st.warning(
                f"{urg_badge} — {result['medicine_name']} | "
                f"{result['quantity']} units | "
                f"Expires in {result['dte']} days | "
                f"Worth £{result['stock_value']:.2f}"
            )
        elif urgency == "expired":
            st.error(
                f"❌ EXPIRED — {result['medicine_name']} | "
                f"{result['quantity']} units | "
                f"Expired {abs(result['dte'])} days ago"
            )
        else:
            st.success(
                f"{urg_badge} — {result['medicine_name']} | "
                f"{result['quantity']} units | "
                f"Expires in {result['dte']} days"
            )

        col1, col2, col3 = st.columns(3)
        col1.markdown(f"🏷️ **Batch ID:** `{result['batch_id']}`")
        col2.markdown(f"🏥 **Branch:** {result['branch_name']}")
        col3.markdown(f"📦 **Quantity:** {result['quantity']} units")

        col4, col5, col6 = st.columns(3)
        col4.markdown(f"📅 **Expiry Date:** {result.get('expiry_date', 'N/A')}")
        col5.markdown(f"⏳ **Shelf Life:** {format_shelf_life(result['dte'])}")
        col6.markdown(f"💷 **Stock Value:** £{result.get('stock_value', 0):.2f}")

        st.caption(f"🎯 **Deterministic Heuristic Risk Score:** {result.get('score', 0)} pts")

        _b_ml_class = result.get("ml_risk_class", "Unavailable")
        _b_ml_prob = result.get("ml_risk_prob")
        if _b_ml_class != "Unavailable" and _b_ml_prob is not None:
            _b_badge = format_risk_badge(
                "critical" if _b_ml_class == "High" else ("warning" if _b_ml_class == "Medium" else "safe"),
                f"ADVISORY ML SIGNAL: {_b_ml_class.upper()} RISK ({int(_b_ml_prob * 100)}% probability)"
            )
            st.caption(f"🤖 {_b_badge} *(Statistical advisory estimate; does not supersede clinical validation)*")

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

        with st.expander(f"✏️ Quick Quantity Update for Batch {result['batch_id']}", expanded=False):
            scan_new_qty = st.number_input(
                "Verified Count (units) *",
                min_value=0,
                step=1,
                value=int(result["quantity"]),
                key=f"scan_qty_input_{result['batch_id']}",
                help="Adjust current count on hand directly following barcode verification."
            )
            if st.button("💾 Save Verified Quantity", type="primary", key=f"btn_save_scan_{result['batch_id']}"):
                try:
                    upd = update_stock_quantity(result["batch_id"], scan_new_qty)
                    if upd:
                        st.success(f"✅ Batch '{result['batch_id']}' quantity successfully updated to {scan_new_qty} units.")
                        st.rerun()
                    else:
                        st.error(f"❌ Batch Not Found: Batch '{result['batch_id']}' was not found in the database.")
                except ValueError as ve:
                    st.error(f"❌ Validation Error: {ve}")
                except DatabaseSaveError as dbe:
                    st.error(f"❌ Database Error: {dbe}")
                except Exception as ex:
                    st.error(f"❌ Unexpected Error: {ex}")
    else:
        st.error(f"❌ Barcode Lookup Failed: {result['message']}")

st.divider()
# --- BARCODE SCANNER CODE ENDS HERE ---

# Filter actionable batches according to search and status filters if set
displayed_recs = recs
if search_filter.strip():
    q_recs = search_filter.strip().lower()
    displayed_recs = [
        r for r in displayed_recs
        if q_recs in str(r.get("medicine_name", "")).lower() or q_recs in str(r.get("batch_id", "")).lower()
    ]
if selected_status_filter != "All Statuses":
    if "Critical" in selected_status_filter:
        displayed_recs = [r for r in displayed_recs if r.get("urgency") == "critical"]
    elif "Act Soon" in selected_status_filter:
        displayed_recs = [r for r in displayed_recs if r.get("urgency") == "near-expiry"]
    elif "Watch" in selected_status_filter:
        displayed_recs = [r for r in displayed_recs if r.get("urgency") == "watch"]
    elif "Safe" in selected_status_filter:
        displayed_recs = [r for r in displayed_recs if r.get("urgency") == "safe"]
    elif "Expired" in selected_status_filter:
        displayed_recs = [r for r in displayed_recs if r.get("urgency") == "expired"]

render_section_header("Actionable Batches", "Priority inventory requiring redistribution transfers or clinical review", icon="📋", badge_text=f"{len(displayed_recs)} Batches")

if not displayed_recs:
    if not recs:
        render_empty_state("No Near-Expiry Stock Needs Action Today", f"All inventory in '{selected}' is currently within safe expiry limits, or all pending transfers have already been processed.", "✅")
    else:
        render_empty_state("No Actionable Batches Match Filter", "Try clearing or adjusting your search query and status filter criteria.", "🔍")

# --- Render each recommendation ---
for i, rec in enumerate(displayed_recs):
    uid = rec["batch_id"]
    already_confirmed  = uid in st.session_state.confirmed
    already_overridden = uid in st.session_state.overridden

    urgency = rec["urgency"]
    theme = get_urgency_theme(urgency)
    border_color = theme["color"]
    label        = format_urgency_badge(urgency)

    st.markdown(
        f"<div style='border-left:6px solid {border_color};"
        f"padding:14px;margin-bottom:12px;border-radius:8px;"
        f"background-color:#FFFFFF;border-top:1px solid #E2E8F0;"
        f"border-right:1px solid #E2E8F0;border-bottom:1px solid #E2E8F0;"
        f"box-shadow:0 1px 3px rgba(0,0,0,0.04)'>",
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

    # Details row - highly scannable
    col_a, col_b, col_c, col_d, col_e = st.columns([1.4, 1.4, 1.2, 1.3, 1.5])
    col_a.markdown(f"🏷️ **Batch:** `{rec['batch_id']}`")
    col_b.markdown(f"🏥 **Branch:** {rec['branch_name']}")
    col_c.markdown(f"📦 **Quantity:** {rec['quantity']} units")
    col_d.markdown(f"📅 **Expiry:** {rec['expiry_date']}")
    col_e.markdown(f"⏳ **Shelf Life:** {format_shelf_life(rec['dte'])}")

    st.caption(f"💷 **Stock Value at Risk:** £{rec.get('stock_value', 0):,.2f} | 🎯 **Deterministic Heuristic Risk Score:** {rec.get('score', 0)} pts")

    # Reason — always visible
    st.markdown(f"**Clinical Redistribution Reason:** {rec['reason']}")

    if "ml_risk_class" in rec:
        _ml_prob = rec.get("ml_risk_probability")
        _ml_class = rec.get("ml_risk_class", "Unavailable")
        if _ml_class == "Unavailable" or _ml_prob is None:
            ml_badge = format_risk_badge("unavailable", "ADVISORY ML SIGNAL: UNAVAILABLE")
            prob_note = "Model prediction not available for this batch profile."
        elif _ml_class == "High":
            ml_badge = format_risk_badge("critical", f"ADVISORY ML SIGNAL: HIGH WASTAGE RISK ({int(_ml_prob * 100)}% probability)")
            prob_note = "Estimated statistical probability of expiry based on historical dispensary velocity and batch attributes."
        elif _ml_class == "Medium":
            ml_badge = format_risk_badge("warning", f"ADVISORY ML SIGNAL: MODERATE WASTAGE RISK ({int(_ml_prob * 100)}% probability)")
            prob_note = "Estimated statistical probability of expiry based on historical dispensary velocity and batch attributes."
        else:
            ml_badge = format_risk_badge("safe", f"ADVISORY ML SIGNAL: LOW WASTAGE RISK ({int(_ml_prob * 100)}% probability)")
            prob_note = "Estimated statistical probability of expiry based on historical dispensary velocity and batch attributes."

        st.markdown(
            f"<div class='advisory-ml-box'>"
            f"<strong>🤖 {ml_badge}</strong><br/>"
            f"<span style='color:#64748B; font-size:0.80rem;'>{prob_note} "
            f"<em>(Supplementary statistical advisory only — does not supersede clinical judgment or rule-based safety criteria)</em></span>"
            f"</div>",
            unsafe_allow_html=True
        )

    if "decision_factors" in rec:
        with st.expander("📊 View Decision Factors & Heuristic Score Breakdown"):
            df_fac = rec["decision_factors"]
            fcol1, fcol2, fcol3, fcol4 = st.columns(4)
            fcol1.markdown(f"• **Days to Expiry:** {df_fac['days_to_expiry']} days")
            fcol2.markdown(f"• **Current Stock:** {df_fac['current_stock']} units")
            fcol3.markdown(f"• **Source Demand:** {df_fac['demand']} units/wk")
            dest_dem_str = f"{df_fac['destination_demand']} units/wk" if df_fac['destination_demand'] is not None else "N/A"
            fcol4.markdown(f"• **Dest Demand:** {dest_dem_str}")

            fcol5, fcol6, fcol7, fcol8 = st.columns(4)
            cap_str = f"{df_fac['available_capacity']} units" if df_fac['available_capacity'] is not None else "N/A"
            fcol5.markdown(f"• **Dest Capacity:** {cap_str}")
            fcol6.markdown(f"• **Transit Time:** ~{df_fac['transfer_time_days']} days")
            fcol7.markdown(f"• **Medicine Value:** £{df_fac['medicine_value_gbp']:,.2f}")
            fcol8.markdown(f"• **Composite Score:** **{df_fac['risk_score']} pts**")

            if "score_components" in df_fac:
                sc = df_fac["score_components"]
                st.info(
                    f"🎯 **Heuristic Score Breakdown ({df_fac['risk_score']} total pts):**\n"
                    f"- ⏳ **Urgency Factor:** {sc['urgency_score']:.1f} pts (proportional to proximity to expiration)\n"
                    f"- 📦 **Quantity Volume Factor:** {sc['quantity_score']:.1f} pts (proportional to batch size needing redistribution)\n"
                    f"- 💷 **Financial Exposure Factor:** {sc['value_score']:.1f} pts (based on £{sc['stock_value']:,.2f} stock value)\n\n"
                    f"*Higher composite scores indicate batches that face imminent expiration with larger quantities and higher monetary exposure.*"
                )

            if "ml_risk_class" in df_fac:
                _fac_ml_class = df_fac.get("ml_risk_class", "Unavailable")
                _fac_ml_prob = df_fac.get("ml_risk_probability")
                _fac_prob_str = f"{int(_fac_ml_prob * 100)}%" if _fac_ml_prob is not None else "N/A"
                _fac_badge = format_risk_badge(
                    "critical" if _fac_ml_class == "High" else ("warning" if _fac_ml_class == "Medium" else ("safe" if _fac_ml_class == "Low" else "unavailable")),
                    f"{_fac_ml_class} ({_fac_prob_str} probability)"
                )
                st.caption(f"🤖 **Advisory ML Assessment:** {_fac_badge} *(Statistical advisory estimate)*")

    # Action buttons and decision controls
    if already_confirmed:
        st.success("✅ **Transfer Confirmed:** The redistribution action was confirmed by the pharmacist and committed to the SQLite audit log.")
    elif already_overridden:
        st.info("↩️ **Transfer Overridden:** Pharmacist rejected this recommendation with clinical justification.")
    elif rec["action"] == "TRANSFER" and rec["destinations"]:
        is_split = rec.get("is_split", False)
        split_dests = rec.get("split_destinations", [])
        if not split_dests:
            split_dests = [d for d in rec["destinations"] if d.get("transfer_quantity", 0) > 0]
            is_split = len(split_dests) > 1

        if is_split:
            st.markdown("#### 🔀 Proposed Split Redistribution Plan")
            st.caption(
                f"The system recommends dividing **{rec['quantity']} units** across "
                f"**{len(split_dests)} destination branches** to respect network capacity limits and prevent over-saturation:"
            )
            split_cols_header = st.columns([3, 2, 2, 2, 2])
            split_cols_header[0].markdown("**🏥 Destination Branch**")
            split_cols_header[1].markdown("**📦 Transfer Qty**")
            split_cols_header[2].markdown("**📈 Weekly Demand**")
            split_cols_header[3].markdown("**⏱️ Stock Cover**")
            split_cols_header[4].markdown("**📥 Net Capacity**")

            for sd in split_dests:
                scols = st.columns([3, 2, 2, 2, 2])
                bname = sd.get("branch_name") or sd.get("dest_branch_name")
                tqty = sd.get("transfer_quantity", 0)
                share_pct = int((tqty / rec["quantity"] * 100)) if rec["quantity"] > 0 else 0
                dem = sd.get("demand") if sd.get("demand") is not None else sd.get("dest_demand_per_week", 0)
                woc = sd.get("weeks_of_cover") if sd.get("weeks_of_cover") is not None else sd.get("dest_weeks_of_cover", 0.0)
                cap = sd.get("capacity") if sd.get("capacity") is not None else sd.get("dest_capacity", 0)

                scols[0].markdown(f"🏥 **{bname}**")
                scols[1].markdown(f"**{tqty} units** `({share_pct}%)`")
                scols[2].markdown(f"{dem} units/wk")
                scols[3].markdown(f"{woc:.1f} wks")
                scols[4].markdown(f"{cap} units")

            dest_display = ", ".join(f"{sd.get('branch_name') or sd.get('dest_branch_name')} ({sd.get('transfer_quantity', 0)}u)" for sd in split_dests)
        else:
            best      = rec["destinations"][0]
            dest_name = best["dest_branch_name"]
            dest_display = dest_name
            st.markdown("#### 🚚 Proposed Single-Branch Transfer")
            st.markdown(
                f"The system recommends transferring **{rec['quantity']} units** "
                f"from **{rec['branch_name']}** ➡️ **{dest_name}**."
            )
            r_col1, r_col2, r_col3 = st.columns(3)
            r_col1.markdown(f"📦 **Transfer Quantity:** {rec['quantity']} units")
            dest_dem = best.get("demand", best.get("dest_demand_per_week", "N/A"))
            r_col2.markdown(f"📈 **Destination Demand:** {dest_dem} units/wk")
            dest_cap = best.get("capacity", best.get("dest_capacity", "N/A"))
            r_col3.markdown(f"📥 **Destination Capacity:** {dest_cap} units")

        st.markdown("---")
        st.markdown("##### ⚖️ Pharmacist Action Required")
        st.caption("Advisory recommendation. No inventory will be transferred until explicitly confirmed by the pharmacist.")

        if rec["requires_confirmation"]:
            st.warning(
                f"⚠️ **High-Impact Transfer Threshold:** This transfer exceeds high-impact criteria "
                f"(£{rec['stock_value']:.2f} value or {rec['quantity']} units). Explicit pharmacist confirmation or override reason required."
            )
            btn_col, reason_col = st.columns([1, 1.8])
            with btn_col:
                confirm_label = f"✅ Confirm Split Transfer ({rec['quantity']}u)" if is_split else f"✅ Confirm Transfer to {dest_name}"
                if st.button(confirm_label,
                             key=f"confirm_{i}",
                             type="primary"):
                    saved = record_recommendation_action(
                        rec=rec,
                        action="CONFIRMED",
                        user_name=st.session_state.current_user["name"],
                        destination=dest_display,
                        is_split=is_split,
                        split_dests=split_dests,
                    )
                    if saved:
                        st.session_state.confirmed.add(uid)
                        st.rerun()
                    else:
                        st.error(DECISION_SAVE_ERROR_MESSAGE)
            with reason_col:
                reason_cat = st.selectbox(
                    "Reason Category (required) *",
                    options=[""] + list(OVERRIDE_REASON_CODES.keys()),
                    format_func=lambda c: f"{OVERRIDE_REASON_CODES[c]} ({c})" if c else "-- Select reason category * --",
                    key=f"reason_cat_{i}",
                    help="Select a standardized machine-readable reason code."
                )
                reason_text = st.text_input(
                    "Clinical Justification (required) *",
                    key=f"reason_{i}",
                    placeholder="e.g. Destination refrigerator is under maintenance",
                    help="Mandatory clinical justification recorded into the audit trail."
                )
                if st.button("↩️ Override / Reject",
                             key=f"override_{i}"):
                    if not reason_cat:
                        st.error("⚠️ Reason Category Required: Please select a reason category before overriding this recommendation.")
                    elif not reason_text.strip():
                        st.error("⚠️ Justification Required: Please type a clinical reason before overriding this recommendation.")
                    else:
                        saved = record_recommendation_action(
                            rec=rec,
                            action="OVERRIDDEN",
                            user_name=st.session_state.current_user["name"],
                            destination=dest_display,
                            override_reason=reason_text.strip(),
                            reason_code=reason_cat,
                        )
                        if saved:
                            st.session_state.overridden.add(uid)
                            st.rerun()
                        else:
                            st.error(DECISION_SAVE_ERROR_MESSAGE)
        else:
            btn_col, reason_col = st.columns([1, 1.8])
            with btn_col:
                action_button_label = f"✅ Confirm Split Transfer ({len(split_dests)} branches)" if is_split else f"✅ Transfer to {dest_display}"
                if st.button(action_button_label,
                             key=f"go_{i}",
                             type="primary"):
                    saved = record_recommendation_action(
                        rec=rec,
                        action="CONFIRMED",
                        user_name=st.session_state.current_user["name"],
                        destination=dest_display,
                        is_split=is_split,
                        split_dests=split_dests,
                    )
                    if saved:
                        st.session_state.confirmed.add(uid)
                        st.rerun()
                    else:
                        st.error(DECISION_SAVE_ERROR_MESSAGE)
            with reason_col:
                with st.expander("↩️ Reject / Override this Recommendation"):
                    std_reason_cat = st.selectbox(
                        "Reason Category (required) *",
                        options=[""] + list(OVERRIDE_REASON_CODES.keys()),
                        format_func=lambda c: f"{OVERRIDE_REASON_CODES[c]} ({c})" if c else "-- Select reason category * --",
                        key=f"std_reason_cat_{i}",
                        help="Select a standardized machine-readable reason code."
                    )
                    std_reason_text = st.text_input(
                        "Clinical Justification (required) *",
                        key=f"std_reason_{i}",
                        placeholder="e.g. Destination refrigerator is under maintenance",
                        help="Enter clinical justification to override this automated recommendation."
                    )
                    if st.button("Confirm Clinical Override", key=f"std_override_{i}"):
                        if not std_reason_cat:
                            st.error("⚠️ Reason Category Required: Please select a reason category before overriding this recommendation.")
                        elif not std_reason_text.strip():
                            st.error("⚠️ Justification Required: Please type a clinical reason before overriding this recommendation.")
                        else:
                            saved = record_recommendation_action(
                                rec=rec,
                                action="OVERRIDDEN",
                                user_name=st.session_state.current_user["name"],
                                destination=dest_display,
                                override_reason=std_reason_text.strip(),
                                reason_code=std_reason_cat,
                            )
                            if saved:
                                st.session_state.overridden.add(uid)
                                st.rerun()
                            else:
                                st.error(DECISION_SAVE_ERROR_MESSAGE)

    elif rec["action"] == "FLAG_FOR_REVIEW":
        st.warning(
            "⚠️ **Manual Clinical Review Required:** The redistribution engine evaluated potential destination branches "
            "across the network but could not discover a viable automated match (receiving branches may be at capacity, report zero dispensing demand, or contain insufficient shelf-life absorption buffer)."
        )
        st.info(f"💡 **Review Justification:** {rec['reason']}")
        st.markdown("##### ⚖️ Pharmacist Action Required")
        st.caption("Review batch manually and confirm local dispensing or disposal protocol.")
        if st.button("📋 Mark as Reviewed", key=f"review_{i}", type="primary"):
            saved = record_recommendation_action(
                rec=rec,
                action="MANUALLY_REVIEWED",
                user_name=st.session_state.current_user["name"],
            )
            if saved:
                st.session_state.confirmed.add(uid)
                st.rerun()
            else:
                st.error(DECISION_SAVE_ERROR_MESSAGE)

    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()

# --- Decision log (loaded from SQLite — the sole authoritative audit log) ---
render_section_header("Decision Log", "Authoritative SQLite audit trail of all operational confirmations and overrides", icon="📁", badge_text="Audit Trail")
_persistent_log = load_log()
if not _persistent_log.empty:
    st.dataframe(_persistent_log, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download Decision Log as CSV",
        _persistent_log.to_csv(index=False),
        file_name="decision_log.csv",
        mime="text/csv"
    )
else:
    render_empty_state(
        "No Decisions Recorded Yet",
        "Operational decisions (confirmed transfers and overrides) will appear here in the audit trail once recorded.",
        "📁"
    )
