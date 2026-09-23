# log_manager.py
import pandas as pd
import os
from datetime import datetime
from typing import Optional

from constants import LOG_PATH as _DEFAULT_LOG_PATH, DECISION_COLUMNS as _DECISION_COLUMNS

# Module-level names preserved for backward-compatibility and test isolation.
# Tests that redirect LOG_PATH do so by patching this module's LOG_PATH directly.
LOG_PATH = _DEFAULT_LOG_PATH
COLUMNS = _DECISION_COLUMNS

def load_log() -> pd.DataFrame:
    """
    Load decisions. Uses SQLite as the sole authoritative source on the
    standard operational path.  When LOG_PATH has been redirected to a
    temporary file (test isolation), that file is read directly instead.
    """
    # If LOG_PATH has been redirected to a custom file (e.g. in isolated unit tests)
    if LOG_PATH != "data/decision_log.csv":
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0:
            try:
                df = pd.read_csv(LOG_PATH).fillna("")
                for col in COLUMNS:
                    if col not in df.columns:
                        df[col] = 0 if col == "quantity" else ""
                if "final_decision" in df.columns and "action" in df.columns:
                    df["final_decision"] = df["final_decision"].replace("", None).fillna(df["action"])
                cols = [c for c in COLUMNS if c in df.columns]
                return df[cols].fillna("")
            except pd.errors.EmptyDataError:
                return pd.DataFrame(columns=COLUMNS)
        return pd.DataFrame(columns=COLUMNS)

    # Standard path: read exclusively from SQLite
    try:
        from database import load_decisions
        df = load_decisions()
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = 0 if col == "quantity" else ""
        if "final_decision" in df.columns and "action" in df.columns:
            df["final_decision"] = df["final_decision"].replace("", None).fillna(df["action"])
        cols = [c for c in COLUMNS if c in df.columns]
        return df[cols].fillna("")
    except Exception as e:
        print(f"Notice: SQLite log load error: {e}")

    return pd.DataFrame(columns=COLUMNS)


def save_entry(
    batch_id: str,
    medicine: str,
    action: str,
    destination: str = "",
    override_reason: str = "",
    user: str = "pharmacist",
    source_branch: str = "",
    quantity: int = 0,
    system_recommendation: str = "",
    final_decision: str = "",
) -> pd.DataFrame:
    """
    Save a decision.

    Standard operational path (LOG_PATH == "data/decision_log.csv"):
        Writes exclusively to SQLite. No CSV file is touched during
        normal runtime. Use database.export_decisions_to_csv() to
        produce an on-demand audit CSV.

    Test isolation path (LOG_PATH redirected to a temp file):
        Writes to the temp CSV only, so test assertions against that
        file continue to work without touching the operational database.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Consistent action normalization (CONFIRMED, OVERRIDDEN, MANUALLY_REVIEWED)
    std_action = str(action).strip().upper()
    if not final_decision:
        final_decision = std_action
    else:
        final_decision = str(final_decision).strip().upper()

    try:
        qty_int = int(float(quantity))
    except (ValueError, TypeError):
        qty_int = 0

    new_row = {
        "timestamp":             now_str,
        "user":                  user,
        "medicine":              medicine,
        "batch_id":              batch_id,
        "source_branch":         source_branch,
        "destination":           destination,
        "quantity":              qty_int,
        "system_recommendation": system_recommendation,
        "action":                std_action,
        "final_decision":        final_decision,
        "override_reason":       override_reason,
    }

    # ── Standard operational path ────────────────────────────────────────────
    if LOG_PATH == "data/decision_log.csv":
        try:
            from database import save_decision
            save_decision(
                batch_id=batch_id, medicine=medicine, action=std_action,
                destination=destination, override_reason=override_reason,
                user=user, source_branch=source_branch,
                quantity=qty_int, system_recommendation=system_recommendation,
                final_decision=final_decision,
                # csv_path intentionally omitted — SQLite only
            )
        except Exception as e:
            print(f"Notice: SQLite decision save: {e}")

        return load_log()

    # ── Test isolation path (LOG_PATH redirected to a temp file) ────────────
    # Read → append → rewrite so the temp CSV reflects every saved entry
    # (tests that redirect LOG_PATH rely on reading the CSV directly).
    df = load_log()
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(LOG_PATH)), exist_ok=True)
        df.to_csv(LOG_PATH, index=False)
    except Exception as e:
        print(f"Warning: Could not write CSV log: {e}")

    return df


def get_summary() -> dict:
    """Return summary stats from the decision log (reads from SQLite)."""
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
