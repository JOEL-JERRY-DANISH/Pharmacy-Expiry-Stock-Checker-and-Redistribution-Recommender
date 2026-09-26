# barcode_registry.py
#
# BarcodeRegistry is now backed entirely by SQLite.
# CSV files are no longer read or written at runtime; they exist only for
# initial seeding and export.
#
# Backward-compatibility note
# ---------------------------
# The constructor used to accept a CSV path:
#   BarcodeRegistry("data/barcode_history.csv")
# It now accepts a SQLite db path or a CSV path.  When a .csv path is
# supplied, the class transparently derives a sibling .db path so that
# existing test code that passes a temp CSV file continues to work without
# any changes to test_edge_cases.py.

import os
from database import (
    initialise_database,
    register_barcode,
    supersede_barcode,
    resolve_barcode,
    get_all_barcode_batch_ids,
    get_connection,
    DB_PATH,
    atomic_update_barcode,
)


def _db_path_from_arg(path):
    """
    Convert a constructor path argument to a SQLite db path.

    Rules:
      - None / omitted  → use the project default (database.DB_PATH)
      - ends with .db   → use as-is
      - ends with .csv  → replace extension with .db (same directory)
      - anything else   → use as-is (assumed to be a db path already)
    """
    if path is None:
        return None  # let database helpers use their own default
    if path.endswith(".csv"):
        return os.path.splitext(path)[0] + ".db"
    return path


class BarcodeRegistry:
    """
    Maps barcodes to stable batch identities using SQLite as the backing store.

    A barcode can change (supersession / repackaging / label correction).
    A batch_id never changes.

    All reads and writes go through the five parameterized helper functions in
    database.py.  No CSV is touched at runtime.
    """

    def __init__(self, path=None):
        """
        Parameters
        ----------
        path : str or None
            Path to a SQLite database file.  May also be a CSV path for
            backward compatibility with older test code — the .csv extension
            is transparently converted to .db in the same directory.
            Defaults to the project database (data/pharmacy.db).
        """
        self._db_path = _db_path_from_arg(path)
        # Ensure the schema exists (idempotent — safe to call multiple times)
        initialise_database(self._db_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve(self, barcode):
        """
        Return (batch_id, status) for any barcode.

        status: "active" | "superseded" | "unknown"
        Returns (None, "unknown") for empty / None input.
        """
        return resolve_barcode(barcode, db_path=self._db_path)

    def register(self, barcode, batch_id, medicine_name, reason="initial"):
        """
        Register a new active barcode.

        Raises ValueError if:
          - barcode is empty / None
          - an active row for *barcode* already exists for a different batch_id

        Idempotent: calling with the same (barcode, batch_id) twice is safe.
        """
        if barcode is None or not str(barcode).strip():
            raise ValueError("Barcode cannot be empty")
        register_barcode(
            barcode=barcode,
            batch_id=batch_id,
            medicine_name=medicine_name,
            reason=reason,
            db_path=self._db_path,
        )

    def update_barcode(self, old_bc, new_bc, reason):
        """
        Atomically supersede *old_bc* and register *new_bc* for the same batch.

        The supersede and insert are performed inside a single SQLite
        transaction via :func:`database.atomic_update_barcode`.  If the new
        barcode registration fails for any reason (e.g. duplicate barcode,
        DB error), the supersede of the old barcode is automatically rolled
        back, leaving the database in its original state.

        Raises ValueError if old_bc is not currently active, or if new_bc is
        already active for a different batch.
        """
        s_old = str(old_bc).strip()
        s_new = str(new_bc).strip()
        atomic_update_barcode(s_old, s_new, reason, db_path=self._db_path)

    def get_all_batches(self):
        """Return the set of all distinct batch_id values in the registry."""
        return get_all_barcode_batch_ids(db_path=self._db_path)
