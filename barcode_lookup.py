# barcode_lookup.py
import pandas as pd
from barcode_registry import BarcodeRegistry
from recommender import days_to_expiry, urgency_label

def lookup_barcode(barcode):
    """
    Scan a barcode and return full batch information.
    Returns a dict with batch details or an error message.
    """
    registry = BarcodeRegistry()
    batch_id, status = registry.resolve(barcode)

    if batch_id is None:
        return {
            "found": False,
            "message": f"Barcode {barcode} not recognised. "
                       f"Please check the label or register "
                       f"this barcode manually.",
        }

    # Load stock data
    try:
        df = pd.read_csv("data/medicines.csv")
    except FileNotFoundError:
        return {"found": False,
                "message": "Stock data file not found."}

    batch = df[df["batch_id"] == batch_id]
    if batch.empty:
        return {
            "found": False,
            "message": f"Batch {batch_id} found in barcode "
                       f"registry but not in stock data.",
        }

    row = batch.iloc[0]
    dte = days_to_expiry(row["expiry_date"])
    urgency = urgency_label(dte)

    return {
        "found":         True,
        "status":        status,
        "batch_id":      batch_id,
        "barcode":       barcode,
        "medicine_name": row["medicine_name"],
        "branch_name":   row["branch_name"],
        "quantity":      int(row["quantity"]),
        "expiry_date":   row["expiry_date"],
        "dte":           dte,
        "urgency":       urgency,
        "stock_value":   round(
            row["quantity"] * row["unit_cost_gbp"], 2),
        "demand_per_week": int(row["demand_per_week"]),
    }