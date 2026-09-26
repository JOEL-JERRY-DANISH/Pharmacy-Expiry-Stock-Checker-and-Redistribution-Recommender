# barcode_lookup.py
import pandas as pd
from barcode_registry import BarcodeRegistry
from recommender import days_to_expiry, urgency_label, score_batch, find_destinations
from database import load_stock

def lookup_barcode(barcode, registry=None, stock_df=None):
    """
    Scan a barcode and return full batch information.
    Returns a dict with batch details, risk calculations, and recommendation if applicable.
    Handles invalid/empty input safely and avoids crashing on unknown barcodes.
    """
    # 4. Invalid/empty input handling
    if barcode is None or not str(barcode).strip():
        return {
            "found":   False,
            "status":  "invalid",
            "message": "Invalid barcode: barcode input cannot be empty.",
        }

    clean_barcode = str(barcode).strip()

    if registry is None:
        registry = BarcodeRegistry()

    # 1, 2, 3. Resolve barcode (handles valid, unknown, and superseded)
    batch_id, status = registry.resolve(clean_barcode)

    if batch_id is None or status == "unknown":
        return {
            "found":   False,
            "status":  "unknown",
            "barcode": clean_barcode,
            "message": f"Barcode {clean_barcode} not recognised. "
                       f"Please check the label or register "
                       f"this barcode manually.",
        }

    # Load stock from SQLite (or provided dataframe)
    if stock_df is not None:
        df = stock_df
    else:
        try:
            df = load_stock()
        except Exception as e:
            return {
                "found":   False,
                "status":  "error",
                "message": f"Could not load stock data: {e}",
            }

    batch = df[df["batch_id"] == batch_id]
    if batch.empty:
        return {
            "found":    False,
            "status":   "not_in_stock",
            "batch_id": batch_id,
            "barcode":  clean_barcode,
            "message":  f"Batch {batch_id} found in barcode "
                        f"registry but not in stock data.",
        }

    row = batch.iloc[0]
    dte = days_to_expiry(row["expiry_date"])
    urgency = urgency_label(dte)

    # Risk calculation
    row_dict = row.to_dict()
    row_dict["dte"]     = dte
    row_dict["urgency"] = urgency
    score = score_batch(row_dict)
    stock_value = round(float(row["quantity"] * row["unit_cost_gbp"]), 2)

    # ML Expiry Risk Prediction
    try:
        from ml_expiry_model import get_ml_predictor
        ml_res = get_ml_predictor().predict_batch(row_dict)
        ml_risk_prob = ml_res["expiry_risk_probability"]
        ml_risk_class = ml_res["risk_class"]
    except Exception as _ml_exc:
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "ML expiry prediction unavailable for batch %s: %s",
            batch_id,
            type(_ml_exc).__name__,
        )
        ml_risk_prob = None
        ml_risk_class = "Unavailable"

    # Allow recommendation generation for actionable near-expiry stock
    recommendation = None
    if urgency in ["critical", "near-expiry"] and int(row["quantity"]) >= 10:
        dests, rec_status, msg = find_destinations(row_dict, df)
        high_impact = (stock_value > 50.0 or int(row["quantity"]) > 200)

        if rec_status == "OK" and dests:
            best = dests[0]
            recommendation = {
                "recommended_action":    "TRANSFER",
                "action":                "TRANSFER",
                "source_branch":         row["branch_name"],
                "destination_branch":    best["dest_branch_name"],
                "suggested_quantity":    int(row["quantity"]),
                "risk_urgency":          urgency,
                "score":                 score,
                "reason":                best["reason"],
                "confidence":            best["confidence"],
                "is_high_impact":        high_impact,
                "requires_confirmation": high_impact,
                "destinations":          dests,
            }
        else:
            recommendation = {
                "recommended_action":    "FLAG_FOR_REVIEW",
                "action":                "FLAG_FOR_REVIEW",
                "source_branch":         row["branch_name"],
                "destination_branch":    None,
                "suggested_quantity":    int(row["quantity"]),
                "risk_urgency":          urgency,
                "score":                 score,
                "reason":                msg,
                "confidence":            "LOW",
                "is_high_impact":        high_impact,
                "requires_confirmation": high_impact,
                "destinations":          [],
            }

    return {
        "found":                     True,
        "status":                    status,
        "batch_id":                  batch_id,
        "barcode":                   clean_barcode,
        "medicine_name":             row["medicine_name"],
        "category":                  row.get("category", "General"),
        "branch_name":               row["branch_name"],
        "branch_id":                 row.get("branch_id", ""),
        "quantity":                  int(row["quantity"]),
        "current_stock":             int(row["quantity"]),
        "expiry_date":               row["expiry_date"],
        "dte":                       dte,
        "days_to_expiry":            dte,
        "urgency":                   urgency,
        "risk_urgency":              urgency,
        "score":                     score,
        "risk_score":                score,
        "unit_cost_gbp":             float(row["unit_cost_gbp"]),
        "stock_value":               stock_value,
        "demand_per_week":           int(row.get("demand_per_week", 0)),
        "branch_capacity_remaining": int(row.get("branch_capacity_remaining", 0)),
        "ml_risk_probability":       ml_risk_prob,
        "ml_risk_class":             ml_risk_class,
        "recommendation":            recommendation,
    }
