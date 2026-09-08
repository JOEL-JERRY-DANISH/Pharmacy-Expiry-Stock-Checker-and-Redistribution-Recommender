# pages/admin_dashboard.py
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Admin Dashboard", layout="wide")
st.title("📊 Admin Dashboard")
st.caption("Summary of all pharmacy stock decisions and waste reduction.")

LOG_PATH = "data/decision_log.csv"

if not os.path.exists(LOG_PATH):
    st.warning("No decision log found yet. "
               "Make some decisions in the main app first.")
    st.stop()

df = pd.read_csv(LOG_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"]      = df["timestamp"].dt.date

# --- Summary metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total decisions",   len(df))
c2.metric("Transfers confirmed",
          len(df[df["action"] == "CONFIRMED"]))
c3.metric("Overrides",
          len(df[df["action"] == "OVERRIDDEN"]))
c4.metric("Manual reviews",
          len(df[df["action"] == "MANUALLY_REVIEWED"]))

st.divider()

# --- Decisions over time ---
st.subheader("Decisions over time")
daily = df.groupby("date").size().reset_index(name="count")
st.bar_chart(daily.set_index("date"))

# --- Action breakdown ---
st.subheader("Action breakdown")
action_counts = df["action"].value_counts().reset_index()
action_counts.columns = ["Action", "Count"]
st.bar_chart(action_counts.set_index("Action"))

# --- Override reasons ---
overrides = df[df["action"] == "OVERRIDDEN"]
if not overrides.empty:
    st.subheader("Override reasons")
    st.dataframe(
        overrides[["timestamp","medicine",
                   "destination","override_reason"]],
        use_container_width=True
    )

# --- Branch performance ---
st.subheader("Transfers by destination branch")
if "destination" in df.columns:
    branch_counts = df[
        df["destination"] != ""
    ]["destination"].value_counts().reset_index()
    branch_counts.columns = ["Branch", "Transfers"]
    st.bar_chart(branch_counts.set_index("Branch"))

# --- Full log ---
with st.expander("View full decision log"):
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Download as CSV",
        df.to_csv(index=False),
        file_name="full_decision_log.csv"
    )