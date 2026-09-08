import pandas as pd
from datetime import datetime

MIN_QTY    = 10
HIGH_VALUE = 50.0
HIGH_QTY   = 200

def days_to_expiry(expiry_str):
    exp = datetime.strptime(expiry_str, "%Y-%m-%d")
    return (exp - datetime.today()).days

def urgency_label(dte):
    if dte < 0:    return "expired"
    if dte <= 7:   return "critical"
    if dte <= 30:  return "near-expiry"
    if dte <= 90:  return "watch"
    return "safe"

def score_batch(row):
    """
    RULE: score = urgency + quantity_weight + value_weight
    critical=100, near-expiry=50, watch=10
    quantity: 0-30 pts  |  value: 0-20 pts
    """
    urg = {"critical":100,"near-expiry":50,"watch":10}.get(
           row["urgency"], 0)
    if urg == 0:
        return 0
    qty_score = min(row["quantity"] / 500, 1.0) * 30
    val_score = min(row["unit_cost_gbp"] / 5.0, 1.0) * 20
    return round(urg + qty_score + val_score, 1)

def find_destinations(source_row, all_df):
    """
    Find best receiving branches.
    Rules: same medicine, different branch, demand>0, capacity>=qty
    Ranked by demand descending (highest need first)
    """
    candidates = all_df[
        (all_df["medicine_name"] == source_row["medicine_name"]) &
        (all_df["branch_id"]     != source_row["branch_id"])
    ].copy()

    if candidates.empty:
        return [], "NO_OTHER_BRANCHES", \
               "No other branches carry this medicine."

    viable = candidates[
        (candidates["demand_per_week"] > 0) &
        (candidates["branch_capacity_remaining"] >=
         source_row["quantity"])
    ].copy()

    if viable.empty:
        if candidates["demand_per_week"].max() == 0:
            msg = "All other branches report zero demand."
        else:
            msg = ("All receiving branches are at capacity. "
                   "Consider splitting the batch.")
        return [], "NO_VIABLE_DEST", msg

    viable = viable.sort_values("demand_per_week", ascending=False)
    results = []
    for _, dest in viable.head(3).iterrows():
        current_woc = round(
            dest["quantity"] / dest["demand_per_week"], 1)
        added_woc = round(
            source_row["quantity"] / dest["demand_per_week"], 1)
        reason = (
            f"{dest['branch_name']} uses "
            f"{int(dest['demand_per_week'])} units/week. "
            f"Current cover: {current_woc} weeks. "
            f"Transfer adds {added_woc} weeks before expiry "
            f"({source_row['dte']} days away)."
        )
        confidence = "HIGH" if dest["demand_per_week"] >= 20 \
                     else "MEDIUM"
        results.append({
            "dest_branch_id":       dest["branch_id"],
            "dest_branch_name":     dest["branch_name"],
            "dest_demand_per_week": int(dest["demand_per_week"]),
            "reason":               reason,
            "confidence":           confidence,
        })
    return results, "OK", ""

def generate_recommendations(df):
    df = df.copy()
    df["dte"]         = df["expiry_date"].apply(days_to_expiry)
    df["urgency"]     = df["dte"].apply(urgency_label)
    df["score"]       = df.apply(score_batch, axis=1)
    df["stock_value"] = (df["quantity"] * df["unit_cost_gbp"]).round(2)

    actionable = df[
        (df["urgency"].isin(["critical","near-expiry"])) &
        (df["quantity"] >= MIN_QTY)
    ].sort_values("score", ascending=False)

    recs = []
    for _, row in actionable.iterrows():
        dests, status, msg = find_destinations(row, df)
        high_impact = (row["stock_value"] > HIGH_VALUE or
                       row["quantity"]    > HIGH_QTY)
        if status != "OK" or not dests:
            recs.append({
                "batch_id":             row["batch_id"],
                "medicine_name":        row["medicine_name"],
                "category":             row["category"],
                "branch_name":          row["branch_name"],
                "branch_id":            row["branch_id"],
                "quantity":             int(row["quantity"]),
                "expiry_date":          row["expiry_date"],
                "dte":                  int(row["dte"]),
                "urgency":              row["urgency"],
                "score":                row["score"],
                "stock_value":          float(row["stock_value"]),
                "action":               "FLAG_FOR_REVIEW",
                "reason":               msg,
                "confidence":           "LOW",
                "destinations":         [],
                "is_high_impact":       high_impact,
                "requires_confirmation":False,
            })
        else:
            best = dests[0]
            recs.append({
                "batch_id":             row["batch_id"],
                "medicine_name":        row["medicine_name"],
                "category":             row["category"],
                "branch_name":          row["branch_name"],
                "branch_id":            row["branch_id"],
                "quantity":             int(row["quantity"]),
                "expiry_date":          row["expiry_date"],
                "dte":                  int(row["dte"]),
                "urgency":              row["urgency"],
                "score":                row["score"],
                "stock_value":          float(row["stock_value"]),
                "action":               "TRANSFER",
                "reason":               best["reason"],
                "confidence":           best["confidence"],
                "destinations":         dests,
                "is_high_impact":       high_impact,
                "requires_confirmation":high_impact,
            })
    return recs

def calculate_baseline(df):
    df = df.copy()
    df["dte"]         = df["expiry_date"].apply(days_to_expiry)
    df["stock_value"] = df["quantity"] * df["unit_cost_gbp"]
    at_risk = df[df["dte"].between(0, 30)]
    return round(at_risk["stock_value"].sum(), 2)