# log_manager.py
import pandas as pd
import os
from datetime import datetime

LOG_PATH = "data/decision_log.csv"

COLUMNS = [
    "timestamp", "batch_id", "medicine",
    "action", "destination", "override_reason", "user"
]

def load_log():
    """Load existing log from disk."""
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=COLUMNS)

def save_entry(batch_id, medicine, action,
               destination="", override_reason="", user="pharmacist"):
    """Append one decision to the persistent log."""
    df = load_log()
    new_row = {
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "batch_id":        batch_id,
        "medicine":        medicine,
        "action":          action,
        "destination":     destination,
        "override_reason": override_reason,
        "user":            user,
    }
    df = pd.concat([df, pd.DataFrame([new_row])],
                   ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    return df

def get_summary():
    """Return summary stats from the log."""
    df = load_log()
    if df.empty:
        return {"total": 0, "confirmed": 0,
                "overridden": 0, "reviewed": 0}
    return {
        "total":     len(df),
        "confirmed": len(df[df["action"] == "CONFIRMED"]),
        "overridden":len(df[df["action"] == "OVERRIDDEN"]),
        "reviewed":  len(df[df["action"] == "MANUALLY_REVIEWED"]),
    }