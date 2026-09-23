import pandas as pd
from datetime import datetime
from typing import Any, Optional

from constants import (
    MIN_QTY,
    HIGH_VALUE,
    HIGH_QTY,
    DEFAULT_TRANSFER_DAYS,
)

# Re-export so that code importing directly from recommender continues to work.
__all__ = [
    "MIN_QTY", "HIGH_VALUE", "HIGH_QTY", "DEFAULT_TRANSFER_DAYS",
    "days_to_expiry", "urgency_label", "score_batch", "get_score_components",
    "calculate_need_score", "calculate_destination_need", "find_destinations",
    "generate_recommendations", "calculate_baseline",
]

def days_to_expiry(expiry_str: str) -> int:
    """Calculate integer days from today until the expiry date."""
    exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    today = datetime.today().date()
    return (exp - today).days

def urgency_label(dte: int) -> str:
    """Categorize expiry urgency based on days to expiry."""
    if dte < 0:    return "expired"
    if dte <= 7:   return "critical"
    if dte <= 30:  return "near-expiry"
    if dte <= 90:  return "watch"
    return "safe"

class BatchScore(float):
    """
    Float-compatible score representation that also encapsulates individual
    score components as attributes and dictionary keys for UI explainability.
    """
    def __new__(cls, total, components=None):
        instance = super().__new__(cls, total)
        instance.components = components or {}
        instance.urgency_score = float(instance.components.get("urgency_score", 0.0))
        instance.quantity_score = float(instance.components.get("quantity_score", 0.0))
        instance.value_score = float(instance.components.get("value_score", 0.0))
        instance.stock_value = float(instance.components.get("stock_value", 0.0))
        return instance

    def __getitem__(self, key):
        return self.components[key]

    def get(self, key, default=None):
        return self.components.get(key, default)


def score_batch(row, return_components=False):
    """
    Transparent, explainable, and mathematically consistent batch risk score (0 to 150):

    1. Expiry Urgency Score (0 - 100 pts):
       - 'critical' (0 to 7 days to expiry): 100.0 pts
       - 'near-expiry' (8 to 30 days to expiry): 50.0 pts
       - 'watch' (31 to 90 days to expiry): 10.0 pts
       - 'safe' / 'expired' (<= 0 or > 90 days): 0.0 pts (Safe batches score 0)

    2. Stock Quantity Weight (0 - 30 pts):
       - Reflects operational volume risk: min(quantity / 500.0, 1.0) * 30.0 pts
       - Normalization ceiling: 500 units

    3. Total Stock Value Weight (0 - 20 pts):
       - Reflects actual financial exposure of the batch:
         stock_value = quantity * unit_cost_gbp
         value_score = min(stock_value / 1250.0, 1.0) * 20.0 pts
       - Normalization ceiling: £1,250.00 (e.g. 500 units @ £2.50/unit)
       - Ensures very large quantities or prices do not disproportionately distort the score.

    Total score = Urgency + Quantity Weight + Value Weight (Bounded between 0.0 and 150.0)

    Parameters:
      row: dict-like or Series containing batch attributes
      return_components: If True, returns tuple (total_score, components_dict)

    Returns:
      BatchScore (float subclass with .components, .urgency_score, .quantity_score, .value_score)
      or tuple (total_score, components_dict) if return_components is True.
    """
    urg_map = {"critical": 100.0, "near-expiry": 50.0, "watch": 10.0}
    urg = urg_map.get(str(row.get("urgency", "")).strip().lower(), 0.0)

    qty = max(0.0, float(row.get("quantity", 0) or 0))
    cost = max(0.0, float(row.get("unit_cost_gbp", 0.0) or 0.0))
    stock_value = round(qty * cost, 2)

    if urg == 0.0:
        total = 0.0
        components = {
            "urgency_score": 0.0,
            "quantity_score": 0.0,
            "value_score": 0.0,
            "total_score": 0.0,
            "stock_value": stock_value,
            "urgency_label": str(row.get("urgency", "safe")),
        }
        if return_components:
            return 0.0, components
        return BatchScore(0.0, components)

    # 2. Quantity Weight: 0 to 30 pts (normalized at 500 units)
    qty_score = round(min(qty / 500.0, 1.0) * 30.0, 2)

    # 3. Stock Value Weight: 0 to 20 pts (normalized at £1,250.00 total stock value)
    val_score = round(min(stock_value / 1250.0, 1.0) * 20.0, 2)

    total_score = round(urg + qty_score + val_score, 1)

    components = {
        "urgency_score": float(urg),
        "quantity_score": float(qty_score),
        "value_score": float(val_score),
        "total_score": float(total_score),
        "stock_value": float(stock_value),
        "urgency_label": str(row.get("urgency", "")),
    }

    if return_components:
        return total_score, components

    return BatchScore(total_score, components)


def get_score_components(row: Any) -> dict:
    """Return dictionary of individual score components for a batch."""
    _, components = score_batch(row, return_components=True)
    return components


def calculate_need_score(dest_row, source_quantity):
    """
    Calculate an explainable destination shortage/need score (0 to 100):

    1. Stock Coverage Need (0 - 50 pts):
       - Measures urgency of replenishment based on current weeks of cover (WOC).
       - WOC = current_quantity / demand_per_week
       - WOC <= 1.0 wk: Critical shortage risk (40 - 50 pts)
       - 1.0 < WOC <= 4.0 wks: Moderate replenishment need (15 - 40 pts)
       - WOC >= 6.0 wks: Ample stock, minimal need (0 pts)
       - Formula: max(0.0, (6.0 - min(woc, 6.0)) / 6.0) * 50.0 pts

    2. Demand Absorption Capacity (0 - 30 pts):
       - Higher weekly demand absorbs stock faster, mitigating re-expiry risk.
       - Formula: min(demand_per_week / 50.0, 1.0) * 30.0 pts

    3. Capacity Headroom Feasibility (0 - 20 pts):
       - Measures ease of storing incoming batch without capacity strain.
       - ratio = capacity_remaining / source_quantity
       - Formula: min(ratio / 5.0, 1.0) * 20.0 pts

    Total need_score = round(coverage_score + demand_score + capacity_score, 1)
    """
    qty = max(0.0, float(dest_row.get("quantity", 0) or 0))
    demand = max(0.0, float(dest_row.get("demand_per_week", 0) or 0))
    cap = max(0.0, float(dest_row.get("branch_capacity_remaining", 0) or 0))
    src_qty = max(1.0, float(source_quantity))

    if demand <= 0:
        return 0.0, {
            "coverage_score": 0.0,
            "demand_score": 0.0,
            "capacity_score": 0.0,
            "total_need_score": 0.0,
            "weeks_of_cover": 0.0,
            "dest_quantity": int(qty),
            "dest_demand_per_week": int(demand),
            "dest_capacity_remaining": int(cap),
        }

    woc = round(qty / demand, 2)

    # 1. Coverage score (up to 50 pts) - lower WOC -> higher need
    cov_score = round(max(0.0, (6.0 - min(woc, 6.0)) / 6.0) * 50.0, 2)

    # 2. Demand score (up to 30 pts) - higher demand -> faster absorption
    dem_score = round(min(demand / 50.0, 1.0) * 30.0, 2)

    # 3. Capacity score (up to 20 pts) - headroom ratio
    cap_ratio = cap / src_qty
    cap_score = round(min(cap_ratio / 5.0, 1.0) * 20.0, 2)

    total_score = round(cov_score + dem_score + cap_score, 1)

    components = {
        "coverage_score": float(cov_score),
        "demand_score": float(dem_score),
        "capacity_score": float(cap_score),
        "total_need_score": float(total_score),
        "weeks_of_cover": float(woc),
        "dest_quantity": int(qty),
        "dest_demand_per_week": int(demand),
        "dest_capacity_remaining": int(cap),
    }
    return total_score, components


def calculate_destination_need(dest_row, usable_days=None, target_cover_weeks=8):
    """
    Calculate reasonable destination need based on weekly demand, current stock,
    and target stock coverage (default 8 weeks) or consumption potential before expiry.

    Guarantees:
    - Zero demand branches have 0 need.
    - Branches already well-stocked (>= target cover) have 0 need.
    - Never returns a negative need.
    """
    demand = float(dest_row.get("demand_per_week", 0))
    if demand <= 0:
        return 0

    current_stock = float(dest_row.get("quantity", 0))
    target_stock = demand * target_cover_weeks

    if usable_days is not None and usable_days > 0:
        consumption_potential = (demand / 7.0) * usable_days
        target_stock = max(target_stock, consumption_potential)

    need = int(round(target_stock - current_stock))
    return max(0, need)


def find_destinations(source_row, all_df, transfer_days=DEFAULT_TRANSFER_DAYS):
    """
    Find and rank candidate receiving branches for stock redistribution based on actual need.
    Supports dividing a source batch across multiple destination branches when single
    destinations have limited capacity or multiple branches demonstrate stock shortage.

    Guarantees:
    1. Days to expiry (dte) > transfer_days. Expired or transit-infeasible batches are rejected.
    2. Zero-demand branches are strictly excluded.
    3. Source branch is strictly excluded.
    4. Candidates ranked by need score (urgency, coverage, demand, capacity).
    5. Each destination is allocated a transfer quantity that never exceeds:
       - remaining source quantity
       - destination available capacity
       - reasonable destination need
    6. Total recommended transfer quantity never exceeds source quantity.
    7. If total capacity across viable branches is less than source quantity,
       retains FLAG_FOR_REVIEW.
    8. Top 3 viable destinations are returned.
    """
    dte = source_row.get("dte")
    if dte is None:
        exp_str = source_row.get("expiry_date")
        dte = days_to_expiry(exp_str) if exp_str else 0

    # Safety Rule 1: Never transfer expired stock
    if dte < 0:
        return [], "EXPIRED_STOCK", "Expired stock cannot be transferred."

    # Safety Rule 2: Transit time feasibility
    if dte <= transfer_days:
        return [], "TRANSIT_INFEASIBLE", (
            f"Estimated transit time ({transfer_days} day{'s' if transfer_days != 1 else ''}) "
            f"leaves insufficient shelf life before expiry ({dte} days)."
        )

    # Candidate branches carrying the same medicine (excluding source branch)
    candidates = all_df[
        (all_df["medicine_name"] == source_row["medicine_name"]) &
        (all_df["branch_id"]     != source_row["branch_id"])
    ].copy()

    if candidates.empty:
        return [], "NO_OTHER_BRANCHES", "No other branches carry this medicine."

    # Safety Rule 3: Exclude zero-demand and zero-capacity branches
    viable = candidates[
        (candidates["demand_per_week"] > 0) &
        (candidates["branch_capacity_remaining"] > 0)
    ].copy()

    if viable.empty:
        if candidates["demand_per_week"].max() == 0:
            msg = "All other branches report zero demand."
        else:
            msg = ("All receiving branches are at capacity. "
                   "Consider splitting the batch.")
        return [], "NO_VIABLE_DEST", msg

    source_qty = int(source_row["quantity"])
    total_viable_capacity = int(viable["branch_capacity_remaining"].sum())

    usable_days = max(0, dte - transfer_days)

    # Check if any viable branch can hold the entire batch alone
    full_capacity_candidates = viable[viable["branch_capacity_remaining"] >= source_qty].copy()

    if not full_capacity_candidates.empty:
        # A single destination can accommodate the entire batch.
        # Score and rank full-capacity candidates to find the highest-need destination.
        scored_candidates = []
        for _, dest in full_capacity_candidates.iterrows():
            score, comps = calculate_need_score(dest, source_qty)
            scored_candidates.append({
                "dest": dest,
                "need_score": score,
                "components": comps,
            })

        scored_candidates.sort(
            key=lambda item: (
                -item["need_score"],
                item["components"]["weeks_of_cover"],
                -item["components"]["dest_demand_per_week"],
                str(item["dest"]["branch_id"]),
            )
        )
        top_candidates = scored_candidates[:3]
        allocations = {str(top_candidates[0]["dest"]["branch_id"]): source_qty}
        for cand in top_candidates[1:]:
            allocations[str(cand["dest"]["branch_id"])] = 0
    else:
        # No single destination can hold the entire batch.
        # Safety Rule 4: If total capacity across all viable branches is less than source quantity,
        # it cannot accommodate the batch
        if total_viable_capacity < source_qty:
            return [], "NO_VIABLE_DEST", (
                f"All receiving branches lack sufficient available capacity for {source_qty} units "
                f"(total capacity across branches: {total_viable_capacity} units). "
                f"Flagged for manual pharmacist review."
            )

        # Batch must be divided across multiple destination branches.
        # Score and rank all viable candidates by need score
        scored_candidates = []
        for _, dest in viable.iterrows():
            score, comps = calculate_need_score(dest, source_qty)
            scored_candidates.append({
                "dest": dest,
                "need_score": score,
                "components": comps,
            })

        scored_candidates.sort(
            key=lambda item: (
                -item["need_score"],
                item["components"]["weeks_of_cover"],
                -item["components"]["dest_demand_per_week"],
                str(item["dest"]["branch_id"]),
            )
        )
        top_candidates = scored_candidates[:3]

        # Calculate recommended transfer quantity for each destination:
        # Constraints:
        # - never transfer more than source quantity
        # - never transfer more than destination capacity
        # - never transfer more than reasonable destination need
        # - total recommended transfer quantity never exceeds source quantity
        remaining_to_allocate = source_qty
        allocations = {}

        # Pass 1: Allocate by need and capacity
        for cand in top_candidates:
            dest = cand["dest"]
            bid = str(dest["branch_id"])
            cap = int(dest["branch_capacity_remaining"])
            need = calculate_destination_need(dest, usable_days=usable_days)

            allocated = min(remaining_to_allocate, cap, need)
            allocations[bid] = allocated
            remaining_to_allocate -= allocated

        # Pass 2: If remaining units exist and destinations have remaining capacity,
        # allocate remainder up to destination capacity
        if remaining_to_allocate > 0:
            for cand in top_candidates:
                if remaining_to_allocate <= 0:
                    break
                dest = cand["dest"]
                bid = str(dest["branch_id"])
                cap = int(dest["branch_capacity_remaining"])
                room = cap - allocations[bid]
                if room > 0:
                    add_qty = min(remaining_to_allocate, room)
                    allocations[bid] += add_qty
                    remaining_to_allocate -= add_qty

    results = []
    source_score = source_row.get("score", score_batch(source_row))
    source_val   = float(source_row.get("stock_value", source_row["quantity"] * source_row.get("unit_cost_gbp", 1.0)))
    source_dem   = int(source_row.get("demand_per_week", 0))

    for cand in top_candidates:
        dest = cand["dest"]
        bid = str(dest["branch_id"])
        score = cand["need_score"]
        comps = cand["components"]
        current_woc = comps["weeks_of_cover"]
        transfer_qty = allocations.get(bid, 0)
        dest_need = calculate_destination_need(dest, usable_days=usable_days)

        # Expected consumption during remaining shelf life after transit
        expected_demand = (dest["demand_per_week"] / 7.0) * usable_days
        effective_qty = transfer_qty if transfer_qty > 0 else source_qty
        absorption_pct = min(100, int(round((expected_demand / max(1, effective_qty)) * 100)))

        reason = (
            f"The medicine is approaching expiry ({dte} days left, risk score: {source_score}), "
            f"the source branch ({source_row['branch_name']}) has excess stock ({source_qty} units, "
            f"worth £{source_val:.2f}, demand: {source_dem} units/wk), and the destination branch ({dest['branch_name']}) "
            f"has high need (need score: {score:.1f}, current stock: {int(dest['quantity'])} units with only {current_woc} wks cover, "
            f"demand: {int(dest['demand_per_week'])} units/wk, recommended transfer: {transfer_qty} units) "
            f"and sufficient available capacity ({int(dest['branch_capacity_remaining'])} units remaining). "
            f"(~{transfer_days}d transit, current cover {current_woc} wks, ~{absorption_pct}% expected absorption)."
        )

        confidence = "HIGH" if dest["demand_per_week"] >= 20 else "MEDIUM"

        results.append({
            "dest_branch_id":       dest["branch_id"],
            "dest_branch_name":     dest["branch_name"],
            "dest_demand_per_week": int(dest["demand_per_week"]),
            "dest_capacity":        int(dest["branch_capacity_remaining"]),
            "dest_current_stock":   int(dest["quantity"]),
            "dest_weeks_of_cover":  float(current_woc),
            "dest_need_score":      float(score),
            "need_score":           float(score),
            "need_score_components": comps,
            "absorption_pct":       absorption_pct,
            "transfer_quantity":    int(transfer_qty),
            "allocated_quantity":   int(transfer_qty),
            "dest_need":            int(dest_need),
            "reason":               reason,
            "confidence":           confidence,
        })

    return results, "OK", ""


def generate_recommendations(df, transfer_days=DEFAULT_TRANSFER_DAYS, all_df=None):
    """
    Generate explainable redistribution recommendations for near-expiry inventory.

    Parameters
    ----------
    df : pd.DataFrame
        Inventory batches to evaluate for redistribution.
    transfer_days : int, default DEFAULT_TRANSFER_DAYS
        Estimated transfer days between branches.
    all_df : pd.DataFrame, optional
        Complete inventory pool across all branches to discover viable destinations.
        If None, defaults to df.
    """
    df = df.copy()
    candidate_pool = all_df if all_df is not None else df
    df["dte"]         = df["expiry_date"].apply(days_to_expiry)
    df["urgency"]     = df["dte"].apply(urgency_label)
    df["score"]       = df.apply(score_batch, axis=1)
    df["stock_value"] = (df["quantity"] * df["unit_cost_gbp"]).round(2)

    # ML Expiry Risk Prediction step (supporting the rule-based engine)
    try:
        from ml_expiry_model import get_ml_predictor
        predictor = get_ml_predictor()
        df = predictor.predict_dataframe(df)
    except Exception:
        df["ml_risk_probability"] = 0.0
        df["ml_risk_class"] = "Low"

    # Actionable: Near-expiry or critical items with sufficient quantity
    # Never recommends expired stock (urgency == 'expired' is excluded)
    actionable = df[
        (df["urgency"].isin(["critical", "near-expiry"])) &
        (df["quantity"] >= MIN_QTY)
    ].sort_values("score", ascending=False)

    recs = []
    for _, row in actionable.iterrows():
        dests, status, msg = find_destinations(row, candidate_pool, transfer_days=transfer_days)
        high_impact = (row["stock_value"] > HIGH_VALUE or
                       row["quantity"]    > HIGH_QTY)

        source_demand = int(row.get("demand_per_week", 0))
        ml_prob = float(row.get("ml_risk_probability", 0.0))
        ml_class = str(row.get("ml_risk_class", "Low"))
        score_comp = get_score_components(row)

        if status != "OK" or not dests:
            if "zero demand" in msg.lower():
                clean_reason = (
                    f"The medicine is approaching expiry ({int(row['dte'])} days left, risk score: {row['score']}), "
                    f"but all other branches report zero demand. Flagged for manual pharmacist review."
                )
            elif "capacity" in msg.lower():
                clean_reason = (
                    f"The medicine is approaching expiry ({int(row['dte'])} days left, risk score: {row['score']}), "
                    f"but all receiving branches lack sufficient available capacity for {int(row['quantity'])} units. "
                    f"Flagged for manual pharmacist review."
                )
            elif "transit time" in msg.lower() or "shelf life" in msg.lower():
                clean_reason = (
                    f"The medicine is approaching expiry ({int(row['dte'])} days left), "
                    f"leaving insufficient shelf life for safe transit time (~{transfer_days}d transit). "
                    f"Flagged for urgent local dispensing or disposal."
                )
            else:
                clean_reason = msg

            decision_factors = {
                "days_to_expiry":     int(row["dte"]),
                "current_stock":      int(row["quantity"]),
                "demand":             source_demand,
                "destination_demand": None,
                "available_capacity": None,
                "transfer_time_days": transfer_days,
                "medicine_value_gbp": float(row["stock_value"]),
                "risk_score":         float(row["score"]),
                "score_components":   score_comp,
                "urgency_score":      score_comp["urgency_score"],
                "quantity_score":     score_comp["quantity_score"],
                "value_score":        score_comp["value_score"],
                "ml_risk_probability":   ml_prob,
                "ml_risk_class":         ml_class,
            }

            explanation = (
                f"Recommended Action: FLAG_FOR_REVIEW\n\n"
                f"Source: {row['branch_name']}\n"
                f"Destination: None\n"
                f"Quantity: {int(row['quantity'])}\n\n"
                f"Reason:\n"
                f"{clean_reason}\n\n"
                f"Score Breakdown: Urgency: {score_comp['urgency_score']:.1f}, "
                f"Quantity: {score_comp['quantity_score']:.1f}, Value: {score_comp['value_score']:.1f}\n\n"
                f"ML Expiry Risk Prediction: {ml_class} ({int(ml_prob * 100)}% probability)"
            )

            rec = {
                # Core required return fields
                "recommended_action":    "FLAG_FOR_REVIEW",
                "action":                "FLAG_FOR_REVIEW",
                "source_branch":         row["branch_name"],
                "source_branch_id":      row["branch_id"],
                "destination_branch":    None,
                "destination_branch_id": None,
                "suggested_quantity":    int(row["quantity"]),
                "risk_urgency":          row["urgency"],
                "score":                 float(row["score"]),
                "score_components":      score_comp,
                "urgency_score":         score_comp["urgency_score"],
                "quantity_score":        score_comp["quantity_score"],
                "value_score":           score_comp["value_score"],
                "reason":                clean_reason,
                "explanation":           explanation,
                "decision_factors":      decision_factors,
                "ml_risk_probability":   ml_prob,
                "ml_risk_class":         ml_class,

                # Detailed metadata & backwards compatibility
                "batch_id":              row["batch_id"],
                "medicine_name":         row["medicine_name"],
                "category":              row["category"],
                "branch_name":           row["branch_name"],
                "branch_id":             row["branch_id"],
                "quantity":              int(row["quantity"]),
                "expiry_date":           row["expiry_date"],
                "dte":                   int(row["dte"]),
                "days_to_expiry":        int(row["dte"]),
                "urgency":               row["urgency"],
                "stock_value":           float(row["stock_value"]),
                "unit_cost_gbp":         float(row["unit_cost_gbp"]),
                "confidence":            "LOW",
                "destinations":          [],
                "is_high_impact":        high_impact,
                "requires_confirmation": False,
                "is_feasible":           False,
                "transfer_time_days":    transfer_days,
            }
            recs.append(rec)
        else:
            for d in dests:
                d["branch_id"] = d["dest_branch_id"]
                d["branch_name"] = d["dest_branch_name"]
                d["demand"] = d["dest_demand_per_week"]
                d["weeks_of_cover"] = d["dest_weeks_of_cover"]
                d["capacity"] = d["dest_capacity"]

            split_dests = [d for d in dests if d.get("transfer_quantity", 0) > 0]
            is_split = len(split_dests) > 1

            best = split_dests[0] if split_dests else dests[0]
            dest_demand   = int(best["dest_demand_per_week"])
            dest_capacity = int(best["dest_capacity"])
            dest_transfer_qty = int(best.get("transfer_quantity", row["quantity"]))

            decision_factors = {
                "days_to_expiry":     int(row["dte"]),
                "current_stock":      int(row["quantity"]),
                "demand":             source_demand,
                "destination_demand": dest_demand,
                "available_capacity": dest_capacity,
                "destination_weeks_of_cover": best.get("dest_weeks_of_cover"),
                "destination_need_score":     best.get("dest_need_score"),
                "destination_need_components": best.get("need_score_components"),
                "transfer_time_days": transfer_days,
                "medicine_value_gbp": float(row["stock_value"]),
                "risk_score":         float(row["score"]),
                "score_components":   score_comp,
                "urgency_score":      score_comp["urgency_score"],
                "quantity_score":     score_comp["quantity_score"],
                "value_score":        score_comp["value_score"],
                "ml_risk_probability":   ml_prob,
                "ml_risk_class":         ml_class,
                "is_split":           is_split,
                "split_destinations": split_dests,
            }

            if not is_split:
                explanation = (
                    f"Recommended Action: TRANSFER\n\n"
                    f"Source: {row['branch_name']}\n"
                    f"Source Quantity: {int(row['quantity'])}\n"
                    f"Destination: {best['dest_branch_name']}\n"
                    f"Quantity: {int(row['quantity'])}\n"
                    f"Transfer Quantity: {dest_transfer_qty}\n"
                    f"Destination Demand: {dest_demand} units/wk\n"
                    f"Current Stock Coverage: {best.get('dest_weeks_of_cover')} wks\n"
                    f"Available Capacity: {dest_capacity} units\n\n"
                    f"Reason:\n"
                    f"{best['reason']}\n\n"
                    f"Score Breakdown: Urgency: {score_comp['urgency_score']:.1f}, "
                    f"Quantity: {score_comp['quantity_score']:.1f}, Value: {score_comp['value_score']:.1f}\n\n"
                    f"ML Expiry Risk Prediction: {ml_class} ({int(ml_prob * 100)}% probability)"
                )
                dest_branch_str = best["dest_branch_name"]
                dest_branch_id_str = best["dest_branch_id"]
                reason_str = best["reason"]
                confidence_str = best["confidence"]
            else:
                split_lines = []
                for d in split_dests:
                    split_lines.append(
                        f"• {d['dest_branch_name']}: Transfer {d['transfer_quantity']} units | "
                        f"Demand: {d['dest_demand_per_week']} units/wk | "
                        f"Current Coverage: {d['dest_weeks_of_cover']} wks | "
                        f"Available Capacity: {d['dest_capacity']} units | "
                        f"Need Score: {d['dest_need_score']:.1f}"
                    )
                split_summary_str = "\n".join(split_lines)

                explanation = (
                    f"Recommended Action: TRANSFER (SPLIT)\n\n"
                    f"Source: {row['branch_name']}\n"
                    f"Source Quantity: {int(row['quantity'])}\n"
                    f"Destination: Split across {len(split_dests)} branches\n"
                    f"Quantity: {int(row['quantity'])}\n\n"
                    f"Split Breakdown:\n"
                    f"{split_summary_str}\n\n"
                    f"Reason:\n"
                    f"The medicine is approaching expiry ({int(row['dte'])} days left, risk score: {row['score']}). "
                    f"To respect destination capacity and match local pharmacy need, the batch of {int(row['quantity'])} units "
                    f"from {row['branch_name']} is divided across {len(split_dests)} receiving branches: "
                    + "; ".join(f"{d['dest_branch_name']} ({d['transfer_quantity']} units)" for d in split_dests) + ".\n\n"
                    f"Score Breakdown: Urgency: {score_comp['urgency_score']:.1f}, "
                    f"Quantity: {score_comp['quantity_score']:.1f}, Value: {score_comp['value_score']:.1f}\n\n"
                    f"ML Expiry Risk Prediction: {ml_class} ({int(ml_prob * 100)}% probability)"
                )
                dest_branch_str = ", ".join(f"{d['dest_branch_name']} ({d['transfer_quantity']})" for d in split_dests)
                dest_branch_id_str = ", ".join(str(d["dest_branch_id"]) for d in split_dests)
                reason_str = (
                    f"Divided transfer of {int(row['quantity'])} units across {len(split_dests)} branches: "
                    + "; ".join(
                        f"{d['dest_branch_name']} (+{d['transfer_quantity']} units, demand: {d['dest_demand_per_week']}/wk, "
                        f"cover: {d['dest_weeks_of_cover']} wks, cap: {d['dest_capacity']})"
                        for d in split_dests
                    )
                )
                confidence_str = "HIGH" if all(d.get("confidence") == "HIGH" for d in split_dests) else "MEDIUM"

            rec = {
                # Core required return fields
                "recommended_action":    "TRANSFER",
                "action":                "TRANSFER",
                "source_branch":         row["branch_name"],
                "source_branch_id":      row["branch_id"],
                "destination_branch":    dest_branch_str,
                "destination_branch_id": dest_branch_id_str,
                "destination_weeks_of_cover": best.get("dest_weeks_of_cover"),
                "destination_need_score":     best.get("dest_need_score"),
                "dest_need_score":            best.get("dest_need_score"),
                "need_score":                 best.get("dest_need_score"),
                "suggested_quantity":    int(row["quantity"]),
                "risk_urgency":          row["urgency"],
                "score":                 float(row["score"]),
                "score_components":      score_comp,
                "urgency_score":         score_comp["urgency_score"],
                "quantity_score":        score_comp["quantity_score"],
                "value_score":           score_comp["value_score"],
                "reason":                reason_str,
                "explanation":           explanation,
                "decision_factors":      decision_factors,
                "ml_risk_probability":   ml_prob,
                "ml_risk_class":         ml_class,

                # Detailed metadata & backwards compatibility
                "batch_id":              row["batch_id"],
                "medicine_name":         row["medicine_name"],
                "category":              row["category"],
                "branch_name":           row["branch_name"],
                "branch_id":             row["branch_id"],
                "quantity":              int(row["quantity"]),
                "expiry_date":           row["expiry_date"],
                "dte":                   int(row["dte"]),
                "days_to_expiry":        int(row["dte"]),
                "urgency":               row["urgency"],
                "stock_value":           float(row["stock_value"]),
                "unit_cost_gbp":         float(row["unit_cost_gbp"]),
                "confidence":            confidence_str,
                "destinations":          dests,
                "is_split":              is_split,
                "split_destinations":    split_dests,
                "is_high_impact":        high_impact,
                "requires_confirmation": high_impact,
                "is_feasible":           True,
                "transfer_time_days":    transfer_days,
            }
            recs.append(rec)

    return recs

def calculate_baseline(df: pd.DataFrame) -> float:
    """
    Calculate baseline value of stock at risk (expiring within 0–30 days).

    Returns the total £ value of all inventory rows whose expiry date falls
    between today and 30 days from now (inclusive).
    """
    df = df.copy()
    df["dte"]         = df["expiry_date"].apply(days_to_expiry)
    df["stock_value"] = df["quantity"] * df["unit_cost_gbp"]
    at_risk = df[df["dte"].between(0, 30)]
    return round(at_risk["stock_value"].sum(), 2)
