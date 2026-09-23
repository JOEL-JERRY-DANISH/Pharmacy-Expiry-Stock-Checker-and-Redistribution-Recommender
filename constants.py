# constants.py
"""
Central constants for the Pharmacy Expiry Stock Checker.

Importing from this module keeps shared values in one place.
Modules that previously defined these constants locally still export
them as module-level names so that existing imports are not broken.
"""

# ── File paths ───────────────────────────────────────────────────────────────
DB_PATH: str = "data/pharmacy.db"
MEDICINES_CSV: str = "data/medicines.csv"
BARCODES_CSV: str = "data/barcode_history.csv"
DECISIONS_CSV: str = "data/decision_log.csv"
LOG_PATH: str = "data/decision_log.csv"

# ── Database column definitions ──────────────────────────────────────────────
REQUIRED_STOCK_COLUMNS: list[str] = [
    "batch_id",
    "medicine_name",
    "branch_id",
    "quantity",
    "expiry_date",
    "unit_cost_gbp",
    "demand_per_week",
    "branch_capacity_remaining",
]

DECISION_COLUMNS: list[str] = [
    "timestamp",
    "user",
    "medicine",
    "batch_id",
    "source_branch",
    "destination",
    "quantity",
    "system_recommendation",
    "action",
    "final_decision",
    "override_reason",
]

# ── Recommender thresholds ───────────────────────────────────────────────────
#: Batches with fewer than this many units are ignored by the recommender.
MIN_QTY: int = 10

#: Batches with a total stock value above this threshold (£) require
#: explicit pharmacist confirmation before a transfer is logged.
HIGH_VALUE: float = 50.0

#: Batches with more units than this threshold also require confirmation.
HIGH_QTY: int = 200

#: Estimated transit time (days) used when calculating weeks of cover at
#: the destination after a hypothetical transfer.
DEFAULT_TRANSFER_DAYS: int = 1

# ── Application URL (used by alert_manager for email links) ─────────────────
import os as _os
APP_URL: str = _os.getenv("APP_URL", "http://localhost:8501")
