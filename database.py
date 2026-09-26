# database.py
import sqlite3
import os
import logging
import pandas as pd
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

from constants import (
    DB_PATH,
    MEDICINES_CSV,
    BARCODES_CSV,
    DECISIONS_CSV,
    REQUIRED_STOCK_COLUMNS,
    DECISION_COLUMNS,
)

# Re-export so existing imports from database continue to work.
__all__ = [
    "DB_PATH", "MEDICINES_CSV", "BARCODES_CSV", "DECISIONS_CSV",
    "REQUIRED_STOCK_COLUMNS", "DECISION_COLUMNS",
    "get_connection", "initialise_database", "load_stock", "save_decision",
    "load_decisions", "import_stock_from_csv", "invalidate_stock_cache",
    "update_stock_quantity", "export_decisions_to_csv",
    "DatabaseLoadError", "get_last_stock_load_error",
    "DatabaseSaveError", "get_last_decision_save_error",
    "atomic_update_barcode",
]


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Open an SQLite connection with directory creation and timeout.

    Args:
        db_path: Path to the SQLite file. Defaults to :data:`DB_PATH`.

    Returns:
        An open :class:`sqlite3.Connection` with a 10-second busy timeout.
    """
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return sqlite3.connect(path, timeout=10)

def initialise_database(db_path: Optional[str] = None, seed: bool = True) -> None:
    """
    Create tables and indexes if missing, and automatically seed initial data from CSV
    if database tables are newly created or empty and seed is True.
    """
    path = db_path or DB_PATH
    conn = get_connection(path)
    cursor = conn.cursor()

    # 1. Stock table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock (
            batch_id TEXT PRIMARY KEY,
            medicine_name TEXT,
            category TEXT,
            branch_id TEXT,
            branch_name TEXT,
            quantity INTEGER,
            expiry_date TEXT,
            unit_cost_gbp REAL,
            demand_per_week INTEGER,
            branch_capacity_remaining INTEGER
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_med ON stock(medicine_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_branch ON stock(branch_id)")

    # 2. Barcode registry table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS barcodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            barcode TEXT,
            batch_id TEXT,
            medicine_name TEXT,
            registered_date TEXT,
            superseded_date TEXT,
            reason_for_change TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_barcodes_bc ON barcodes(barcode)")
    # Enforce at-most-one active (non-superseded) row per barcode value at the DB level.
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_barcodes_active
        ON barcodes(barcode)
        WHERE superseded_date IS NULL
    """)

    # 3. Decision audit log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user TEXT,
            medicine TEXT,
            batch_id TEXT,
            source_branch TEXT,
            destination TEXT,
            quantity INTEGER,
            system_recommendation TEXT,
            action TEXT,
            final_decision TEXT,
            override_reason TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_batch ON decisions(batch_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_action ON decisions(action)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_user ON decisions(user)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_time ON decisions(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_medicine ON decisions(medicine)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_source ON decisions(source_branch)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_user_action ON decisions(user, action)")

    # Safe schema migration for existing databases
    cursor.execute("PRAGMA table_info(decisions)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    migrations = [
        ("user", "TEXT"),
        ("source_branch", "TEXT"),
        ("quantity", "INTEGER"),
        ("system_recommendation", "TEXT"),
        ("final_decision", "TEXT"),
    ]
    for col_name, col_type in migrations:
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE decisions ADD COLUMN {col_name} {col_type}")

    conn.commit()

    # Automatic initial import from CSV if tables are empty
    if seed:
        _seed_if_empty(conn)

    conn.close()

def _seed_if_empty(conn):
    """Import CSV data into tables if they are empty."""
    cursor = conn.cursor()

    # Seed stock if empty using validated safe row insertion
    cursor.execute("SELECT COUNT(*) FROM stock")
    if cursor.fetchone()[0] == 0 and os.path.exists(MEDICINES_CSV):
        try:
            df_stock = pd.read_csv(MEDICINES_CSV)
            missing = [c for c in REQUIRED_STOCK_COLUMNS if c not in df_stock.columns]
            if not missing:
                for _, row in df_stock.iterrows():
                    cursor.execute("""
                        INSERT OR IGNORE INTO stock (
                            batch_id, medicine_name, category, branch_id,
                            branch_name, quantity, expiry_date, unit_cost_gbp,
                            demand_per_week, branch_capacity_remaining
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(row["batch_id"]).strip(),
                        str(row["medicine_name"]).strip(),
                        str(row.get("category", "General")),
                        str(row["branch_id"]).strip(),
                        str(row.get("branch_name", row["branch_id"])),
                        int(row["quantity"]),
                        str(row["expiry_date"]).strip(),
                        float(row["unit_cost_gbp"]),
                        int(row["demand_per_week"]),
                        int(row["branch_capacity_remaining"]),
                    ))
                conn.commit()
        except Exception as e:
            print(f"Warning: Could not seed stock from {MEDICINES_CSV}: {e}")

    # Seed barcodes if empty
    cursor.execute("SELECT COUNT(*) FROM barcodes")
    if cursor.fetchone()[0] == 0 and os.path.exists(BARCODES_CSV):
        try:
            df_bc = pd.read_csv(BARCODES_CSV, dtype={"barcode": str})
            df_bc.to_sql("barcodes", conn, if_exists="append", index=False)
            conn.commit()
        except Exception as e:
            print(f"Warning: Could not seed barcodes from {BARCODES_CSV}: {e}")

    # Seed historical decisions if table is empty and CSV exists
    cursor.execute("SELECT COUNT(*) FROM decisions")
    if cursor.fetchone()[0] == 0 and os.path.exists(DECISIONS_CSV) and os.path.getsize(DECISIONS_CSV) > 0:
        try:
            df_dec = pd.read_csv(DECISIONS_CSV)
            if not df_dec.empty:
                for col in DECISION_COLUMNS:
                    if col not in df_dec.columns:
                        df_dec[col] = 0 if col == "quantity" else ""
                if "final_decision" in df_dec.columns and "action" in df_dec.columns:
                    df_dec["final_decision"] = df_dec["final_decision"].fillna(df_dec["action"])
                cols = [c for c in DECISION_COLUMNS if c in df_dec.columns]
                df_dec[cols].to_sql("decisions", conn, if_exists="append", index=False)
                conn.commit()
        except Exception as e:
            print(f"Warning: Could not seed decisions from {DECISIONS_CSV}: {e}")

class DatabaseLoadError(RuntimeError):
    """Raised when loading inventory from SQLite fails due to corruption, lock, or query error."""
    pass


LAST_STOCK_LOAD_ERROR: Optional[str] = None


def get_last_stock_load_error() -> Optional[str]:
    """Return the last error encountered during stock loading, or None if successful."""
    return LAST_STOCK_LOAD_ERROR


def load_stock(db_path=None, raise_on_error=False):
    """
    Load stock data from SQLite as the primary operational data store.

    Distinguishes between:
      CASE A: First-time setup (database file does not exist yet). Automatically
              initialises and seeds from CSV if seed=True.
      CASE B: Database exists but loading fails due to SQLite corruption, lock,
              schema error, or query failure. In Case B, does NOT silently fall
              back to stale CSV data; surfaces the error and returns an explicit
              failure state (or raises DatabaseLoadError if raise_on_error=True).
    """
    global LAST_STOCK_LOAD_ERROR
    LAST_STOCK_LOAD_ERROR = None
    path = db_path or DB_PATH

    stock_columns = [
        "batch_id", "medicine_name", "category", "branch_id",
        "branch_name", "quantity", "expiry_date", "unit_cost_gbp",
        "demand_per_week", "branch_capacity_remaining"
    ]

    # CASE A: First-time setup — database file does not exist on disk yet
    if not os.path.exists(path):
        try:
            initialise_database(path, seed=True)
        except Exception as e:
            err_msg = f"Failed to initialise database at '{path}': {e}"
            LAST_STOCK_LOAD_ERROR = err_msg
            print(f"ERROR: {err_msg}")
            if raise_on_error:
                raise DatabaseLoadError(err_msg) from e
            failed_df = pd.DataFrame(columns=stock_columns)
            failed_df.attrs["error"] = err_msg
            failed_df.attrs["load_failed"] = True
            return failed_df

    # Query SQLite database
    conn = None
    try:
        conn = get_connection(path)
        df = pd.read_sql("SELECT * FROM stock", conn)

        # If the default database was initialized empty, attempt re-seeding
        if df.empty and path == DB_PATH:
            initialise_database(path, seed=True)
            df = pd.read_sql("SELECT * FROM stock", conn)

        return df

    except Exception as e:
        # CASE B failure: Corrupted DB, locked DB, missing tables, invalid schema
        err_msg = f"SQLite load_stock error for '{path}': {e}"
        LAST_STOCK_LOAD_ERROR = err_msg
        print(f"ERROR: {err_msg}")

        if raise_on_error:
            raise DatabaseLoadError(err_msg) from e

        # Explicit failure state: Empty DataFrame with stock schema and error metadata.
        # DO NOT silently fall back to MEDICINES_CSV to prevent operating on stale inventory.
        failed_df = pd.DataFrame(columns=stock_columns)
        failed_df.attrs["error"] = err_msg
        failed_df.attrs["load_failed"] = True
        return failed_df

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

def invalidate_stock_cache():
    """
    Invalidate any active inventory caches (e.g. Streamlit cache_data or in-memory caches).
    Safe to call in both Streamlit runtime and non-Streamlit environments (tests/CLI).
    """
    try:
        import streamlit as st
        if hasattr(st, "cache_data") and hasattr(st.cache_data, "clear"):
            st.cache_data.clear()
    except Exception:
        pass

clear_stock_cache = invalidate_stock_cache


def update_stock_quantity(batch_id, new_quantity, db_path=None):
    """
    Update the inventory quantity for an existing batch in SQLite.
    Validates the quantity (rejects negative numbers) and invalidates caches.

    Parameters
    ----------
    batch_id : str
        The unique batch identifier.
    new_quantity : int or float
        The new quantity (must be >= 0).
    db_path : str, optional
        Path to SQLite database file.

    Returns
    -------
    bool
        True if an existing record was updated, False otherwise.
    """
    if batch_id is None or not str(batch_id).strip():
        raise ValueError("batch_id cannot be empty")

    try:
        qty_int = int(float(new_quantity))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid quantity: {new_quantity}")

    if qty_int < 0:
        raise ValueError("Quantity cannot be negative")

    path = db_path or DB_PATH
    conn = get_connection(path)
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE stock SET quantity = ? WHERE batch_id = ?",
            (qty_int, str(batch_id).strip())
        )
        conn.commit()
        updated = cur.rowcount > 0
    finally:
        conn.close()

    if updated:
        invalidate_stock_cache()

    return updated

class DatabaseSaveError(RuntimeError):
    """Raised when writing an operational audit decision to SQLite fails."""
    pass


LAST_DECISION_SAVE_ERROR: Optional[str] = None


def get_last_decision_save_error() -> Optional[str]:
    """Return the last error encountered during decision saving, or None if successful."""
    return LAST_DECISION_SAVE_ERROR


def save_decision(batch_id, medicine, action,
                  destination="", override_reason="",
                  user="pharmacist", source_branch="",
                  quantity=0, system_recommendation="",
                  final_decision="", db_path=None, csv_path=None,
                  raise_on_error=False) -> bool:
    """
    Save an operational decision to SQLite (the sole authoritative audit log).

    SQLite is the only persistent store for runtime decisions.
    CSV files are NOT written during normal operations; use
    export_decisions_to_csv() to produce an audit export on demand.

    Parameters
    ----------
    batch_id : str
        The unique batch identifier.
    medicine : str
        Medicine name.
    action : str
        Decision action (CONFIRMED, OVERRIDDEN, MANUALLY_REVIEWED).
    destination : str, optional
        Destination label.
    override_reason : str, optional
        Reason provided for an override.
    user : str, optional
        Name/username of pharmacist.
    source_branch : str, optional
        Source branch name.
    quantity : int, float, or str, optional
        Quantity involved.
    system_recommendation : str, optional
        System's original recommendation.
    final_decision : str, optional
        Final action decided.
    db_path : str, optional
        Path to SQLite database file.
    csv_path : str or None, optional
        If provided, the decision is also written to this CSV file (test isolation only).
    raise_on_error : bool, optional
        If True, raises DatabaseSaveError on failure instead of returning False.

    Returns
    -------
    bool
        True if the audit decision was successfully persisted, False otherwise.
    """
    global LAST_DECISION_SAVE_ERROR
    LAST_DECISION_SAVE_ERROR = None

    path = db_path or DB_PATH
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

    conn = None
    # Primary write to SQLite — sole source of truth
    try:
        if not os.path.exists(path):
            initialise_database(path)

        conn = get_connection(path)
        with conn:
            conn.execute("""
                INSERT INTO decisions
                (timestamp, user, medicine, batch_id, source_branch,
                 destination, quantity, system_recommendation, action,
                 final_decision, override_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_str, user, medicine, batch_id, source_branch,
                destination, qty_int, system_recommendation, std_action,
                final_decision, override_reason
            ))

        # Only write to CSV if primary SQLite write succeeded (prevents inconsistent state)
        if csv_path:
            try:
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
                df_row = pd.DataFrame([new_row])
                file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
                df_row.to_csv(csv_path, mode="a", header=not file_exists, index=False)
            except Exception as e:
                print(f"Warning: Could not write decision to {csv_path}: {e}")

        invalidate_stock_cache()
        return True

    except Exception as e:
        err_msg = f"Failed to persist decision for batch '{batch_id}' to SQLite at '{path}': {e}"
        LAST_DECISION_SAVE_ERROR = err_msg
        logger.error(err_msg)
        print(f"ERROR: {err_msg}")

        if raise_on_error:
            raise DatabaseSaveError(err_msg) from e
        return False

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

def load_decisions(db_path=None):
    """
    Load all operational decisions from SQLite — the sole authoritative audit log.

    Returns a DataFrame with all DECISION_COLUMNS. Returns an empty
    DataFrame (with correct columns) when no decisions have been recorded.
    """
    path = db_path or DB_PATH
    try:
        if not os.path.exists(path):
            initialise_database(path)

        conn = get_connection(path)
        df = pd.read_sql("SELECT * FROM decisions ORDER BY id ASC", conn)
        conn.close()

        for col in DECISION_COLUMNS:
            if col not in df.columns:
                df[col] = 0 if col == "quantity" else ""
        if "final_decision" in df.columns and "action" in df.columns:
            df["final_decision"] = df["final_decision"].fillna(df["action"])
        cols = [c for c in DECISION_COLUMNS if c in df.columns]
        return df[cols].fillna("")
    except Exception as e:
        print(f"Warning reading decisions from SQLite: {e}")

    return pd.DataFrame(columns=DECISION_COLUMNS)


def export_decisions_to_csv(csv_path=None, db_path=None):
    """
    Export all decisions from SQLite to a CSV file on demand.

    This is the only sanctioned way to produce a CSV audit export.
    Normal runtime operations never write to CSV directly.

    Parameters
    ----------
    csv_path : str
        Destination CSV file path. Defaults to DECISIONS_CSV.
    db_path : str or None
        SQLite database path. Defaults to DB_PATH.

    Returns
    -------
    str
        Absolute path of the written CSV file.
    """
    out_path = csv_path or DECISIONS_CSV
    df = load_decisions(db_path=db_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False)
    return os.path.abspath(out_path)

def validate_stock_row(row, existing_db_batches=None, seen_in_file=None, update_existing=False):
    """
    Validate a single row (dict-like or Series) for stock table import.

    Returns:
      (is_valid: bool, error_message: str | None, cleaned_data: dict | None, is_update: bool)
    """
    errors = []

    # 1. batch_id
    raw_batch = row.get("batch_id")
    if raw_batch is None or pd.isna(raw_batch) or not str(raw_batch).strip():
        errors.append("batch_id cannot be empty")
        batch_id = ""
    else:
        batch_id = str(raw_batch).strip()
        if seen_in_file is not None and batch_id in seen_in_file:
            errors.append(f"duplicate batch_id '{batch_id}' found in CSV")

    is_update = False
    if batch_id and existing_db_batches is not None and batch_id in existing_db_batches:
        if update_existing:
            is_update = True
        else:
            errors.append(
                f"batch_id '{batch_id}' already exists in database (use update_existing=True to overwrite)"
            )

    # 2. medicine_name
    raw_med = row.get("medicine_name")
    if raw_med is None or pd.isna(raw_med) or not str(raw_med).strip():
        errors.append("medicine_name cannot be empty")
        medicine_name = ""
    else:
        medicine_name = str(raw_med).strip()

    # 3. branch_id
    raw_branch = row.get("branch_id")
    if raw_branch is None or pd.isna(raw_branch) or not str(raw_branch).strip():
        errors.append("branch_id cannot be empty")
        branch_id = ""
    else:
        branch_id = str(raw_branch).strip()

    # 4. quantity
    raw_qty = row.get("quantity") if "quantity" in row else row.get("current_stock")
    quantity = None
    if raw_qty is None or pd.isna(raw_qty) or (isinstance(raw_qty, str) and not raw_qty.strip()):
        errors.append("quantity cannot be empty")
    else:
        try:
            f_qty = float(raw_qty)
            if not f_qty.is_integer():
                errors.append("quantity must be an integer")
            elif f_qty < 0:
                errors.append("quantity cannot be negative")
            else:
                quantity = int(f_qty)
        except (ValueError, TypeError):
            errors.append(f"invalid quantity '{raw_qty}': must be a non-negative integer")

    # 5. expiry_date
    raw_exp = row.get("expiry_date")
    expiry_date = ""
    if raw_exp is None or pd.isna(raw_exp) or not str(raw_exp).strip():
        errors.append("expiry_date cannot be empty")
    else:
        s_exp = str(raw_exp).strip()
        try:
            parsed = datetime.strptime(s_exp, "%Y-%m-%d")
            expiry_date = parsed.strftime("%Y-%m-%d")
        except ValueError:
            errors.append(f"invalid expiry_date '{s_exp}': must be YYYY-MM-DD")

    # 6. unit_cost_gbp
    raw_cost = row.get("unit_cost_gbp") if "unit_cost_gbp" in row else row.get("unit_cost")
    unit_cost_gbp = None
    if raw_cost is None or pd.isna(raw_cost) or (isinstance(raw_cost, str) and not raw_cost.strip()):
        errors.append("unit_cost_gbp cannot be empty")
    else:
        try:
            f_cost = float(raw_cost)
            if f_cost < 0:
                errors.append("unit_cost_gbp cannot be negative")
            else:
                unit_cost_gbp = round(f_cost, 4)
        except (ValueError, TypeError):
            errors.append(f"invalid unit_cost_gbp '{raw_cost}': must be a non-negative number")

    # 7. demand_per_week
    raw_dem = row.get("demand_per_week")
    demand_per_week = None
    if raw_dem is None or pd.isna(raw_dem) or (isinstance(raw_dem, str) and not raw_dem.strip()):
        errors.append("demand_per_week cannot be empty")
    else:
        try:
            f_dem = float(raw_dem)
            if not f_dem.is_integer():
                errors.append("demand_per_week must be an integer")
            elif f_dem < 0:
                errors.append("demand_per_week cannot be negative")
            else:
                demand_per_week = int(f_dem)
        except (ValueError, TypeError):
            errors.append(f"invalid demand_per_week '{raw_dem}': must be a non-negative integer")

    # 8. branch_capacity_remaining
    raw_cap = row.get("branch_capacity_remaining") if "branch_capacity_remaining" in row else row.get("capacity")
    branch_capacity_remaining = None
    if raw_cap is None or pd.isna(raw_cap) or (isinstance(raw_cap, str) and not raw_cap.strip()):
        errors.append("branch_capacity_remaining cannot be empty")
    else:
        try:
            f_cap = float(raw_cap)
            if not f_cap.is_integer():
                errors.append("branch_capacity_remaining must be an integer")
            elif f_cap < 0:
                errors.append("branch_capacity_remaining cannot be negative")
            else:
                branch_capacity_remaining = int(f_cap)
        except (ValueError, TypeError):
            errors.append(f"invalid branch_capacity_remaining '{raw_cap}': must be a non-negative integer")

    if errors:
        return False, "; ".join(errors), None, is_update

    category = str(row.get("category", "General")).strip() if not pd.isna(row.get("category")) else "General"
    branch_name = str(row.get("branch_name", branch_id)).strip() if not pd.isna(row.get("branch_name")) else branch_id

    cleaned_data = {
        "batch_id": batch_id,
        "medicine_name": medicine_name,
        "category": category,
        "branch_id": branch_id,
        "branch_name": branch_name,
        "quantity": quantity,
        "expiry_date": expiry_date,
        "unit_cost_gbp": unit_cost_gbp,
        "demand_per_week": demand_per_week,
        "branch_capacity_remaining": branch_capacity_remaining,
    }
    return True, None, cleaned_data, is_update


def import_stock_from_csv(csv_path=None, db_path=None, update_existing=False, force=False, strict=False):
    """
    Safely import stock records from CSV into SQLite with column and row-level validation.

    Parameters:
      csv_path: Path to CSV file (defaults to MEDICINES_CSV)
      db_path: Path to SQLite DB (defaults to DB_PATH)
      update_existing: If True, updates records whose batch_id already exists in SQLite.
                       If False, preserves existing records and rejects duplicate batch_ids.
      force: If True, clears existing stock table prior to import within the transaction.
      strict: If True, any validation rejection causes the entire import to abort without writing.

    Returns:
      Dictionary containing import summary:
      {
          "success": bool,
          "records_processed": int,
          "records_inserted": int,
          "records_updated": int,
          "records_rejected": int,
          "errors": list[str],
          "warnings": list[str],
          "processed": int,
          "inserted": int,
          "updated": int,
          "rejected": int,
      }
    """
    summary = {
        "success": False,
        "records_processed": 0,
        "records_inserted": 0,
        "records_updated": 0,
        "records_rejected": 0,
        "errors": [],
        "warnings": [],
        "processed": 0,
        "inserted": 0,
        "updated": 0,
        "rejected": 0,
    }

    csv_file = csv_path or MEDICINES_CSV
    if not os.path.exists(csv_file):
        summary["errors"].append(f"CSV file not found: {csv_file}")
        return summary

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        summary["errors"].append(f"Failed to read CSV file: {e}")
        return summary

    # 1. Validate required CSV columns
    missing_cols = [col for col in REQUIRED_STOCK_COLUMNS if col not in df.columns]
    if missing_cols:
        summary["errors"].append(f"Missing required column(s): {', '.join(sorted(missing_cols))}")
        return summary

    path = db_path or DB_PATH
    initialise_database(path, seed=False)

    # 2. Fetch existing DB batch IDs
    conn = get_connection(path)
    try:
        cur = conn.cursor()
        if force:
            existing_db_batches = set()
        else:
            cur.execute("SELECT batch_id FROM stock")
            existing_db_batches = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()

    # 3. Validate rows
    to_insert = []
    to_update = []
    seen_in_file = set()

    for idx, row in df.iterrows():
        summary["records_processed"] += 1
        is_valid, err, cleaned, is_update = validate_stock_row(
            row,
            existing_db_batches=existing_db_batches,
            seen_in_file=seen_in_file,
            update_existing=update_existing,
        )
        if not is_valid:
            summary["records_rejected"] += 1
            summary["errors"].append(f"Row {idx + 1}: {err}")
        else:
            b_id = cleaned["batch_id"]
            seen_in_file.add(b_id)
            if is_update:
                to_update.append(cleaned)
            else:
                to_insert.append(cleaned)
                existing_db_batches.add(b_id)

    if strict and summary["records_rejected"] > 0:
        summary["errors"].append(
            f"Strict mode aborted import due to {summary['records_rejected']} validation failure(s)."
        )
        summary["processed"] = summary["records_processed"]
        summary["rejected"] = summary["records_rejected"]
        return summary

    # 4. Database Transaction
    conn = get_connection(path)
    try:
        with conn:
            cur = conn.cursor()
            if force:
                cur.execute("DELETE FROM stock")

            for item in to_insert:
                cur.execute("""
                    INSERT INTO stock (
                        batch_id, medicine_name, category, branch_id,
                        branch_name, quantity, expiry_date, unit_cost_gbp,
                        demand_per_week, branch_capacity_remaining
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item["batch_id"], item["medicine_name"], item["category"],
                    item["branch_id"], item["branch_name"], item["quantity"],
                    item["expiry_date"], item["unit_cost_gbp"],
                    item["demand_per_week"], item["branch_capacity_remaining"]
                ))
                summary["records_inserted"] += 1

            for item in to_update:
                cur.execute("""
                    UPDATE stock SET
                        medicine_name = ?,
                        category = ?,
                        branch_id = ?,
                        branch_name = ?,
                        quantity = ?,
                        expiry_date = ?,
                        unit_cost_gbp = ?,
                        demand_per_week = ?,
                        branch_capacity_remaining = ?
                    WHERE batch_id = ?
                """, (
                    item["medicine_name"], item["category"], item["branch_id"],
                    item["branch_name"], item["quantity"], item["expiry_date"],
                    item["unit_cost_gbp"], item["demand_per_week"],
                    item["branch_capacity_remaining"], item["batch_id"]
                ))
                summary["records_updated"] += 1

        summary["success"] = True
        if summary["records_inserted"] > 0 or summary["records_updated"] > 0:
            invalidate_stock_cache()
    except Exception as exc:
        summary["success"] = False
        summary["errors"].append(f"Database transaction error: {exc}")
        summary["records_inserted"] = 0
        summary["records_updated"] = 0
    finally:
        conn.close()

    summary["processed"] = summary["records_processed"]
    summary["inserted"] = summary["records_inserted"]
    summary["updated"] = summary["records_updated"]
    summary["rejected"] = summary["records_rejected"]
    return summary


def import_from_csv(db_path=None, csv_path=None, force=False, update_existing=False):
    """
    Explicitly import CSV data into SQLite tables with validation and transactions.
    Preserves existing records unless update_existing=True or force=True.

    If db_path looks like a CSV file and csv_path is None, swaps them for convenience.
    """
    if db_path and str(db_path).lower().endswith(".csv") and csv_path is None:
        csv_path = db_path
        db_path = None

    path = db_path or DB_PATH
    initialise_database(path)

    # 1. Safely import stock using validated transactional pipeline
    summary = import_stock_from_csv(
        csv_path=csv_path or MEDICINES_CSV,
        db_path=path,
        update_existing=update_existing,
        force=force,
    )

    # 2. Safely import barcodes if barcodes table needs populating or force=True
    if os.path.exists(BARCODES_CSV):
        conn = get_connection(path)
        try:
            with conn:
                cur = conn.cursor()
                if force:
                    cur.execute("DELETE FROM barcodes")
                cur.execute("SELECT COUNT(*) FROM barcodes")
                if cur.fetchone()[0] == 0:
                    df_bc = pd.read_csv(BARCODES_CSV, dtype={"barcode": str})
                    for _, row in df_bc.iterrows():
                        bc = str(row.get("barcode", "")).strip()
                        if bc:
                            cur.execute("""
                                INSERT OR IGNORE INTO barcodes
                                (barcode, batch_id, medicine_name, registered_date, superseded_date, reason_for_change)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (
                                bc,
                                str(row.get("batch_id", "")).strip(),
                                str(row.get("medicine_name", "")).strip(),
                                str(row.get("registered_date", "")).strip(),
                                None if pd.isna(row.get("superseded_date")) or not str(row.get("superseded_date")).strip() else str(row.get("superseded_date")).strip(),
                                str(row.get("reason_for_change", "")).strip(),
                            ))
        except Exception as e:
            summary["warnings"].append(f"Barcode import notice: {e}")
        finally:
            conn.close()

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Barcode-specific helpers (used by BarcodeRegistry as SQLite backend)
# ─────────────────────────────────────────────────────────────────────────────

def register_barcode(barcode, batch_id, medicine_name,
                     reason="initial", db_path=None):
    """
    Insert a new active barcode row into SQLite.

    Raises ValueError if:
      - barcode is empty / None
      - an active row for *barcode* already exists for a different batch_id

    If an identical active row already exists (same barcode AND same batch_id)
    the call is idempotent — returns without raising.

    Uses parameterized SQL; no user input is ever interpolated into a query.
    """
    if not barcode or not str(barcode).strip():
        raise ValueError("Barcode cannot be empty")

    s_barcode = str(barcode).strip()
    path = db_path or DB_PATH
    conn = get_connection(path)
    try:
        cursor = conn.cursor()
        # Check for an existing active row first
        cursor.execute(
            "SELECT batch_id FROM barcodes "
            "WHERE barcode = ? AND superseded_date IS NULL",
            (s_barcode,),
        )
        existing = cursor.fetchone()
        if existing is not None:
            if existing[0] == batch_id:
                return  # idempotent — already registered to the same batch
            raise ValueError(
                f"Barcode {s_barcode!r} is already active for batch "
                f"{existing[0]!r}; cannot reassign to {batch_id!r}"
            )

        today = datetime.today().strftime("%Y-%m-%d")
        cursor.execute(
            "INSERT INTO barcodes "
            "(barcode, batch_id, medicine_name, "
            " registered_date, superseded_date, reason_for_change) "
            "VALUES (?, ?, ?, ?, NULL, ?)",
            (s_barcode, batch_id, medicine_name, today, reason),
        )
        conn.commit()
    finally:
        conn.close()


def supersede_barcode(barcode, superseded_date=None, db_path=None):
    """
    Mark the active row for *barcode* as superseded.
    Raises ValueError if no active row is found.
    """
    s_barcode = str(barcode).strip()
    today = superseded_date or datetime.today().strftime("%Y-%m-%d")
    path = db_path or DB_PATH
    conn = get_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE barcodes SET superseded_date = ? "
            "WHERE barcode = ? AND superseded_date IS NULL",
            (today, s_barcode),
        )
        if cursor.rowcount == 0:
            raise ValueError(
                f"Barcode {s_barcode!r} not found as an active entry"
            )
        conn.commit()
    finally:
        conn.close()


def atomic_update_barcode(old_barcode, new_barcode, reason, db_path=None):
    """
    Atomically supersede *old_barcode* and register *new_barcode* for the same
    batch in a single SQLite transaction.

    All three operations — fetching the medicine name, marking the old barcode
    as superseded, and inserting the new active row — are performed on one
    connection within a single ``with conn:`` block.  SQLite's connection
    context manager commits when the block exits normally and rolls back
    automatically on any exception, so the database can never be left in a
    partially updated state.

    Parameters
    ----------
    old_barcode : str
        The currently active barcode to supersede.  Must exist and be active.
    new_barcode : str
        The replacement barcode to register as active.  Must not already be
        active for a different batch.
    reason : str
        Reason for the change (stored in ``reason_for_change``).
    db_path : str or None
        Path to the SQLite database file.  Defaults to :data:`DB_PATH`.

    Raises
    ------
    ValueError
        If *old_barcode* is not currently active, or if *new_barcode* is
        already active for a different batch.
    """
    s_old = str(old_barcode).strip()
    s_new = str(new_barcode).strip()
    today = datetime.today().strftime("%Y-%m-%d")
    path = db_path or DB_PATH

    conn = get_connection(path)
    try:
        with conn:  # commits on exit; rolls back on any exception
            cur = conn.cursor()

            # 1. Verify the old barcode is active and fetch medicine_name
            cur.execute(
                "SELECT batch_id, medicine_name FROM barcodes "
                "WHERE barcode = ? AND superseded_date IS NULL",
                (s_old,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(
                    f"Barcode {s_old!r} not found as an active entry"
                )
            batch_id, medicine_name = row[0], row[1] or ""

            # 2. Guard: new barcode must not already be active for a different batch
            cur.execute(
                "SELECT batch_id FROM barcodes "
                "WHERE barcode = ? AND superseded_date IS NULL",
                (s_new,),
            )
            existing_new = cur.fetchone()
            if existing_new is not None:
                if existing_new[0] != batch_id:
                    raise ValueError(
                        f"Barcode {s_new!r} is already active for batch "
                        f"{existing_new[0]!r}; cannot reassign to {batch_id!r}"
                    )
                # idempotent: new barcode already active for the same batch;
                # still supersede the old one if not already done

            # 3. Supersede the old barcode
            cur.execute(
                "UPDATE barcodes SET superseded_date = ? "
                "WHERE barcode = ? AND superseded_date IS NULL",
                (today, s_old),
            )
            # rowcount == 0 is fine here only in the idempotent case above;
            # in all other cases the SELECT already confirmed an active row exists

            # 4. Insert the new active barcode row (skip if idempotent)
            if existing_new is None:
                cur.execute(
                    "INSERT INTO barcodes "
                    "(barcode, batch_id, medicine_name, "
                    " registered_date, superseded_date, reason_for_change) "
                    "VALUES (?, ?, ?, ?, NULL, ?)",
                    (s_new, batch_id, medicine_name, today, reason),
                )
        # Transaction committed successfully
    finally:
        conn.close()


def resolve_barcode(barcode, db_path=None):
    """
    Return (batch_id, status) for *barcode* from the SQLite barcodes table.

    status: "active" | "superseded" | "unknown"
    Returns (None, "unknown") for empty / None input.
    """
    if barcode is None or not str(barcode).strip():
        return None, "unknown"

    s_barcode = str(barcode).strip()
    path = db_path or DB_PATH
    conn = get_connection(path)
    try:
        cursor = conn.cursor()
        # Active row first
        cursor.execute(
            "SELECT batch_id FROM barcodes "
            "WHERE barcode = ? AND superseded_date IS NULL",
            (s_barcode,),
        )
        row = cursor.fetchone()
        if row is not None:
            return row[0], "active"
        # Most-recently-superseded row
        cursor.execute(
            "SELECT batch_id FROM barcodes "
            "WHERE barcode = ? "
            "ORDER BY superseded_date DESC LIMIT 1",
            (s_barcode,),
        )
        row = cursor.fetchone()
        if row is not None:
            return row[0], "superseded"
        return None, "unknown"
    finally:
        conn.close()


def load_barcodes(db_path=None):
    """Return the full barcodes table as a DataFrame."""
    path = db_path or DB_PATH
    _cols = [
        "id", "barcode", "batch_id", "medicine_name",
        "registered_date", "superseded_date", "reason_for_change",
    ]
    try:
        conn = get_connection(path)
        try:
            return pd.read_sql("SELECT * FROM barcodes", conn)
        finally:
            conn.close()
    except Exception as exc:
        print(f"Warning: load_barcodes error: {exc}")
        return pd.DataFrame(columns=_cols)


def get_all_barcode_batch_ids(db_path=None):
    """Return the set of all distinct batch_id values in the barcodes table."""
    path = db_path or DB_PATH
    conn = get_connection(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT batch_id FROM barcodes")
        return {row[0] for row in cursor.fetchall()}
    finally:
        conn.close()


if __name__ == "__main__":
    initialise_database()
    print("Database verified and ready at data/pharmacy.db")
