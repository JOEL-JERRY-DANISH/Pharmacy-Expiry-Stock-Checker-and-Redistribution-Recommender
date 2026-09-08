import streamlit as st                                    # ← line 1
import pandas as pd                                       # ← line 2
from datetime import datetime                             # ← line 3
from recommender import generate_recommendations, calculate_baseline
from log_manager import save_entry, load_log, get_summary
from auth_config import CREDENTIALS                       # ← add this

st.set_page_config(
    page_title="Pharmacy Expiry Stock Checker",
    page_icon="💊",
    layout="wide"
)

# ============================================================
# LOGIN SECTION — paste everything below here, after imports
# ============================================================

def check_password(username, password):
    """Check if username and password are correct."""
    users = CREDENTIALS["usernames"]
    if username not in users:
        return False, None
    if users[username]["password"] == password:
        return True, users[username]
    return False, None

# --- Login screen ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

if not st.session_state.logged_in:
    st.title("💊 Pharmacy Stock Checker")
    st.subheader("Please log in to continue")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")

        if submitted:
            valid, user_info = check_password(username, password)
            if valid:
                st.session_state.logged_in    = True
                st.session_state.current_user = user_info
                st.rerun()
            else:
                st.error("Incorrect username or password. "
                         "Please try again.")
    st.stop()

# --- Show logged in user ---
user = st.session_state.current_user
if user is not None:
    st.sidebar.success(f"Logged in as: {user['name']}")
    st.sidebar.caption(f"Branch: {user['branch']}")

if st.sidebar.button("Log Out"):
    st.session_state.logged_in    = False
    st.session_state.current_user = None
    st.rerun()

# ============================================================
# REST OF YOUR APP STARTS HERE — st.set_page_config etc
# ============================================================


st.title("💊 Pharmacy Expiry Stock Checker")
st.caption(
    "This tool finds medicines that are about to expire and "
    "recommends which branch to send them to before they are wasted."
)

from database import load_stock, initialise_database
initialise_database()

@st.cache_data
def load_data():
    return load_stock()

df = load_data()

# --- Sidebar ---
st.sidebar.header("🔍 Filter Stock")
branches = ["All branches"] + \
           sorted(df["branch_name"].unique().tolist())
selected = st.sidebar.selectbox("Show stock from:", branches)
if selected != "All branches":
    df = df[df["branch_name"] == selected]

# --- Session state ---
if "log" not in st.session_state:
    st.session_state.log = []
if "confirmed" not in st.session_state:
    st.session_state.confirmed = set()
if "overridden" not in st.session_state:
    st.session_state.overridden = set()

# --- Generate recommendations ---
recs         = generate_recommendations(df)
baseline     = calculate_baseline(df)
transfer_val = sum(r["stock_value"] for r in recs
                   if r["action"] == "TRANSFER")
# --- Generate recommendations ---
recs         = generate_recommendations(df)
baseline     = calculate_baseline(df)
transfer_val = sum(r["stock_value"] for r in recs
                   if r["action"] == "TRANSFER")

# --- Email alert section --- ← ADD EVERYTHING BELOW HERE
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
c1.metric("🔴 Urgent (0–7 days)",     critical)
c2.metric("🟠 Act Soon (8–30 days)",  near_exp)
c3.metric("💷 Total Stock at Risk",   f"£{total_val:,.2f}")
c4.metric("✅ Covered by Transfers",  f"£{transfer_val:,.2f}")

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
    result = lookup_barcode(barcode_input.strip())

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

    # Action buttons
    if already_confirmed:
        st.success("✅ Transfer confirmed for this session.")
    elif already_overridden:
        st.info("↩️ This recommendation was overridden.")
    elif rec["action"] == "TRANSFER" and rec["destinations"]:
        best      = rec["destinations"][0]
        dest_name = best["dest_branch_name"]

        if rec["requires_confirmation"]:
            st.markdown(
                f"**Suggested transfer:** {rec['quantity']} units "
                f"→ **{dest_name}**"
            )
            st.caption(
                "⚠️ This is a large transfer. "
                "Please confirm or override below."
            )
            btn_col, reason_col = st.columns([1, 2])
            with btn_col:
                if st.button("✅ Confirm Transfer",
                             key=f"confirm_{i}",
                             type="primary"):
                    st.session_state.confirmed.add(uid)
                    save_entry(
                        batch_id=uid,
                        medicine=rec["medicine_name"],
                        action="CONFIRMED",
                        destination=dest_name,
                        override_reason="",
                        user="pharmacist"
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
                        save_entry(
                            batch_id=uid,
                            medicine=rec["medicine_name"],
                            action="OVERRIDDEN",
                            destination=dest_name,
                            override_reason=reason_text,
                            user="pharmacist"
                        )
                        st.rerun()
                    else:
                        st.error(
                            "Please type a reason before overriding."
                        )
        else:
            st.markdown(
                f"**Suggested transfer:** {rec['quantity']} units "
                f"→ **{dest_name}**"
            )
            if st.button(f"✅ Transfer to {dest_name}",
                         key=f"go_{i}"):
                st.session_state.confirmed.add(uid)
                save_entry(
                    batch_id=uid,
                    medicine=rec["medicine_name"],
                    action="CONFIRMED",
                    destination=dest_name,
                    override_reason="",
                    user="pharmacist"
                )
                st.rerun()

    elif rec["action"] == "FLAG_FOR_REVIEW":
        st.warning(
            "⚠️ Could not automatically match this item to a "
            "receiving branch. Please review manually."
        )
        if st.button("📋 Mark as Reviewed", key=f"review_{i}"):
            st.session_state.confirmed.add(uid)
            save_entry(
                batch_id=uid,
                medicine=rec["medicine_name"],
                action="MANUALLY_REVIEWED",
                destination="",
                override_reason="",
                user="pharmacist"
            )
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()

# --- Decision log ---
st.subheader("📁 Decision Log (Audit Trail)")
if st.session_state.log:
    log_df = pd.DataFrame(st.session_state.log)
    st.dataframe(log_df, use_container_width=True)
    st.download_button(
        "⬇️ Download Decision Log as CSV",
        log_df.to_csv(index=False),
        file_name="decision_log.csv",
        mime="text/csv"
    )
else:
    st.caption(
        "No decisions recorded yet. "
        "Confirm or override a recommendation above."
    )