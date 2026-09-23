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
    "validate_numeric_field", "validate_batch_numerics",
]

def days_to_expiry(expiry_str: Any) -> Optional[int]:
    """Calculate integer days from today until the expiry date.
    Returns None if the expiry date is missing, malformed, or invalid.
    """
    if expiry_str is None or pd.isna(expiry_str):
        return None
    try:
        if not isinstance(expiry_str, str):
            expiry_str = str(expiry_str)
        cleaned = expiry_str.strip()
        if not cleaned:
            return None
        exp = datetime.strptime(cleaned, "%Y-%m-%d").date()
        today = datetime.today().date()
        return (exp - today).days
    except (ValueError, TypeError, AttributeError):
        return None

def urgency_label(dte: Optional[int]) -> str:
    """Categorize expiry urgency based on days to expiry."""
    if dte is None or (isinstance(dte, float) and pd.isna(dte)):
        return "invalid"
    if dte < 0:    return "expired"
    if dte <= 7:   return "critical"
    if dte <= 30:  return "near-expiry"
    if dte <= 90:  return "watch"
    return "safe"

def validate_numeric_field(val: Any, field_name: str, is_integer: bool = True) -> tuple[bool, Optional[float], Optional[str]]:
    """
    Validate a single numeric value.
    Returns (is_valid, parsed_value, error_message).
    Rejects strings ('abc', 'hello', ''), None, NaN, inf, negatives, and non-integers if required.
    """
    if val is None or pd.isna(val):
        return False, None, f"{field_name} cannot be empty or missing"
    if isinstance(val, str) and not val.strip():
        return False, None, f"{field_name} cannot be empty"
    try:
        f = float(val)
        if pd.isna(f) or f == float("inf") or f == float("-inf"):
            return False, None, f"invalid {field_name} '{val}': not a finite number"
        if f < 0:
            return False, None, f"{field_name} cannot be negative ({val})"
        if is_integer and not f.is_integer():
            return False, None, f"{field_name} must be an integer ({val})"
        return True, (int(f) if is_integer else f), None
    except (ValueError, TypeError):
        return False, None, f"invalid {field_name} '{val}': not a valid number"

def validate_batch_numerics(row: Any) -> tuple[bool, dict, list[str]]:
    """
    Validate numeric fields of an inventory row.
    Returns (is_valid, cleaned_dict, error_messages).
    """
    errors = []
    cleaned = {}

    # 1. Quantity / current_stock
    raw_qty = row.get("quantity") if "quantity" in row else row.get("current_stock")
    v_qty, val_qty, err_qty = validate_numeric_field(raw_qty, "quantity", is_integer=True)
    if not v_qty:
        errors.append(err_qty)
    else:
        cleaned["quantity"] = val_qty

    # 2. Unit cost
    raw_cost = row.get("unit_cost_gbp") if "unit_cost_gbp" in row else row.get("unit_cost", 0.0)
    v_cost, val_cost, err_cost = validate_numeric_field(raw_cost, "unit_cost_gbp", is_integer=False)
    if not v_cost:
        errors.append(err_cost)
    else:
        cleaned["unit_cost_gbp"] = val_cost

    # 3. Demand per week
    if "demand_per_week" in row:
        raw_dem = row.get("demand_per_week")
        v_dem, val_dem, err_dem = validate_numeric_field(raw_dem, "demand_per_week", is_integer=False)
        if not v_dem:
            errors.append(err_dem)
        else:
            cleaned["demand_per_week"] = int(val_dem)
    else:
        cleaned["demand_per_week"] = 0

    # 4. Branch capacity remaining
    if "branch_capacity_remaining" in row or "capacity" in row:
        raw_cap = row.get("branch_capacity_remaining") if "branch_capacity_remaining" in row else row.get("capacity")
        v_cap, val_cap, err_cap = validate_numeric_field(raw_cap, "branch_capacity_remaining", is_integer=False)
        if not v_cap:
            errors.append(err_cap)
        else:
            cleaned["branch_capacity_remaining"] = int(val_cap)
    else:
        cleaned["branch_capacity_remaining"] = 500

    is_valid = len(errors) == 0
    return is_valid, cleaned, errors

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

    raw_qty = row.get("quantity") if "quantity" in row else row.get("current_stock", 0)
    raw_cost = row.get("unit_cost_gbp") if "unit_cost_gbp" in row else row.get("unit_cost", 0.0)

    v_qty, qty_val, _ = validate_numeric_field(raw_qty, "quantity", is_integer=False)
    v_cost, cost_val, _ = validate_numeric_field(raw_cost, "unit_cost_gbp", is_integer=False)

    qty = qty_val if (v_qty and qty_val is not None) else 0.0
    cost = cost_val if (v_cost and cost_val is not None) else 0.0
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
    v_src, src_qty_val, _ = validate_numeric_field(source_quantity, "source_quantity", is_integer=False)
    if not v_src or src_qty_val is None or src_qty_val <= 0:
        return 0.0, {
            "coverage_score": 0.0, "demand_score": 0.0, "capacity_score": 0.0,
            "total_need_score": 0.0, "weeks_of_cover": 0.0, "dest_quantity": 0,
            "dest_demand_per_week": 0, "dest_capacity_remaining": 0,
        }

    raw_qty = dest_row.get("quantity") if "quantity" in dest_row else dest_row.get("current_stock", 0)
    v_qty, qty_val, _ = validate_numeric_field(raw_qty, "quantity", is_integer=False)

    raw_dem = dest_row.get("demand_per_week", 0)
    v_dem, dem_val, _ = validate_numeric_field(raw_dem, "demand_per_week", is_integer=False)

    raw_cap = dest_row.get("branch_capacity_remaining") if "branch_capacity_remaining" in dest_row else dest_row.get("capacity", 0)
    v_cap, cap_val, _ = validate_numeric_field(raw_cap, "branch_capacity_remaining", is_integer=False)

    if not v_dem or dem_val is None or dem_val <= 0 or not v_cap or cap_val is None or cap_val <= 0 or not v_qty or qty_val is None:
        return 0.0, {
            "coverage_score": 0.0, "demand_score": 0.0, "capacity_score": 0.0,
            "total_need_score": 0.0, "weeks_of_cover": 0.0,
            "dest_quantity": int(qty_val) if (v_qty and qty_val is not None) else 0,
            "dest_demand_per_week": int(dem_val) if (v_dem and dem_val is not None) else 0,
            "dest_capacity_remaining": int(cap_val) if (v_cap and cap_val is not None) else 0,
        }

    qty = float(qty_val)
    demand = float(dem_val)
    cap = float(cap_val)
    src_qty = max(1.0, float(src_qty_val))

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
    and target stock coverage (default 8 weeks) bounded by remaining shelf life.

    Guarantees:
    - Zero demand branches have 0 need.
    - Branches already well-stocked (>= target cover) have 0 need.
    - When usable_days is specified, demand coverage is capped by usable shelf life:
        usable_weeks = usable_days / 7.0
        weeks_to_cover = min(target_cover_weeks, usable_weeks)
        target_stock = demand * weeks_to_cover
    - When usable_days <= 0 or invalid, need is 0.
    - Never returns a negative need.
    - Never returns a negative demand.
    """
    raw_dem = dest_row.get("demand_per_week", 0)
    v_dem, dem_val, _ = validate_numeric_field(raw_dem, "demand_per_week", is_integer=False)
    if not v_dem or dem_val is None or dem_val <= 0:
        return 0

    raw_qty = dest_row.get("quantity") if "quantity" in dest_row else dest_row.get("current_stock", 0)
    v_qty, qty_val, _ = validate_numeric_field(raw_qty, "quantity", is_integer=False)
    if not v_qty or qty_val is None:
        return 0

    demand = float(dem_val)
    current_stock = float(qty_val)

    if usable_days is not None:
        try:
            u_days = float(usable_days)
        except (ValueError, TypeError):
            u_days = 0.0
        if u_days <= 0:
            return 0
        usable_weeks = u_days / 7.0
        weeks_to_cover = min(float(target_cover_weeks), usable_weeks)
    else:
        weeks_to_cover = float(target_cover_weeks)

    target_stock = demand * weeks_to_cover
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
    # Safety Rule -1: Source quantity validation
    raw_qty = source_row.get("quantity") if "quantity" in source_row else source_row.get("current_stock")
    v_qty, source_qty, err_qty = validate_numeric_field(raw_qty, "quantity", is_integer=True)
    if not v_qty or source_qty is None:
        return [], "INVALID_NUMERIC_DATA", f"Invalid source quantity '{raw_qty}': must be a positive integer. Flagged for manual pharmacist review."
    if source_qty <= 0:
        return [], "ZERO_QUANTITY", f"Source quantity is {source_qty}. No stock to transfer."

    dte = source_row.get("dte")
    if dte is None or (isinstance(dte, float) and pd.isna(dte)):
        exp_str = source_row.get("expiry_date")
        dte = days_to_expiry(exp_str) if exp_str is not None else None

    # Safety Rule 0: Invalid or missing expiry date requires manual review
    if dte is None or (isinstance(dte, float) and pd.isna(dte)):
        return [], "INVALID_EXPIRY", "Invalid or missing expiry date. Flagged for manual pharmacist review."

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

    # Safety Rule 3: Exclude zero-demand, zero-capacity, and corrupt numeric branches
    valid_dest_rows = []
    for _, dest in candidates.iterrows():
        is_val, clean, _ = validate_batch_numerics(dest)
        if is_val and clean["demand_per_week"] > 0 and clean["branch_capacity_remaining"] > 0:
            d_copy = dest.copy()
            d_copy["quantity"] = clean["quantity"]
            d_copy["demand_per_week"] = clean["demand_per_week"]
            d_copy["branch_capacity_remaining"] = clean["branch_capacity_remaining"]
            valid_dest_rows.append(d_copy)

    if not valid_dest_rows:
        return [], "NO_VIABLE_DEST", "All receiving branches are at capacity, report zero demand, or contain invalid inventory data."

    viable = pd.DataFrame(valid_dest_rows)

    total_viable_capacity = int(viable["branch_capacity_remaining"].sum())

    usable_days = max(0, dte - transfer_days)

    # Safety Rule 4: If total capacity across all viable branches is less than source quantity,
    # it cannot accommodate the batch
    if total_viable_capacity < source_qty:
        return [], "NO_VIABLE_DEST", (
            f"All receiving branches lack sufficient available capacity for {source_qty} units "
            f"(total capacity across branches: {total_viable_capacity} units). "
            f"Flagged for manual pharmacist review."
        )

    # Score and evaluate all viable candidate branches
    scored_all = []
    for _, dest in viable.iterrows():
        score, comps = calculate_need_score(dest, source_qty)
        need = calculate_destination_need(dest, usable_days=usable_days)
        cap = int(dest.get("branch_capacity_remaining", 0) or 0)
        scored_all.append({
            "dest": dest,
            "need_score": score,
            "components": comps,
            "need": need,
            "cap": cap,
            "can_absorb_full": (cap >= source_qty and need >= source_qty),
        })

    # Sort deterministically by:
    # 1. Highest need score
    # 2. Lowest weeks of cover
    # 3. Highest weekly demand
    # 4. Branch ID (string tie-breaker)
    scored_all.sort(
        key=lambda item: (
            -item["need_score"],
            item["components"]["weeks_of_cover"],
            -item["components"]["dest_demand_per_week"],
            str(item["dest"]["branch_id"]),
        )
    )

    # If branches exist that can accommodate the entire batch alone (both capacity >= source_qty
    # and destination need >= source_qty), prioritize them at the front of candidates.
    full_candidates = [c for c in scored_all if c["can_absorb_full"]]
    other_candidates = [c for c in scored_all if not c["can_absorb_full"]]
    top_candidates = (full_candidates + other_candidates)[:3]

    # Calculate recommended transfer quantity for each destination:
    # Constraints:
    # - transfer_quantity = min(remaining_source_quantity, destination_need, destination_capacity)
    # - never transfer more than source quantity
    # - never transfer more than destination capacity
    # - never transfer more than reasonable destination need
    # - total recommended transfer quantity never exceeds source quantity
    # - allocation is never negative
    # - zero-need destinations receive zero
    # - zero-capacity destinations receive zero
    remaining_to_allocate = max(0, source_qty)
    allocations = {}

    # Pass 1: Allocate by need and capacity
    for cand in top_candidates:
        dest = cand["dest"]
        bid = str(dest["branch_id"])
        cap = max(0, cand["cap"])
        need = max(0, cand["need"])

        allocated = max(0, min(remaining_to_allocate, cap, need))
        allocations[bid] = allocated
        remaining_to_allocate -= allocated

    # Pass 2: If remaining units exist and destinations have remaining capacity and need,
    # allocate remainder respecting remaining capacity and remaining destination need
    if remaining_to_allocate > 0:
        for cand in top_candidates:
            if remaining_to_allocate <= 0:
                break
            dest = cand["dest"]
            bid = str(dest["branch_id"])
            cap = max(0, cand["cap"])
            need = max(0, cand["need"])

            already_allocated = max(0, allocations.get(bid, 0))
            remaining_capacity = max(0, cap - already_allocated)
            remaining_need = max(0, need - already_allocated)

            add_qty = max(0, min(remaining_to_allocate, remaining_capacity, remaining_need))
            if add_qty > 0:
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
        if transfer_qty <= 0:
            absorption_pct = 0
        else:
            absorption_pct = max(0, min(100, int(round((expected_demand / float(transfer_qty)) * 100))))

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

    # Safe stock_value computation
    qty_series = pd.to_numeric(df.get("quantity", df.get("current_stock", 0)), errors="coerce")
    cost_series = pd.to_numeric(df.get("unit_cost_gbp", df.get("unit_cost", 0)), errors="coerce")
    valid_val_mask = qty_series.notna() & (qty_series >= 0) & cost_series.notna() & (cost_series >= 0)
    df["stock_value"] = (qty_series * cost_series).round(2).where(valid_val_mask, 0.0)

    # Validate each row's numeric integrity
    validations = [validate_batch_numerics(row) for _, row in df.iterrows()]
    df["is_numeric_valid"] = [v[0] for v in validations]
    df["cleaned_numerics"] = [v[1] for v in validations]
    df["numeric_errors"]   = [v[2] for v in validations]

    # ML Expiry Risk Prediction step (supporting the rule-based engine)
    try:
        from ml_expiry_model import get_ml_predictor
        predictor = get_ml_predictor()
        df = predictor.predict_dataframe(df)
    except Exception:
        df["ml_risk_probability"] = None
        df["ml_risk_class"] = "Unavailable"

    # Actionable items:
    # 1. Invalid numerics (corrupted data -> must be reviewed, never transferred)
    # 2. Invalid expiry (must be reviewed, never transferred)
    # 3. Valid near-expiry or critical items with sufficient quantity (>= MIN_QTY)
    def row_is_actionable(r):
        if not r["is_numeric_valid"]:
            return True
        if r["urgency"] == "invalid":
            return True
        if r["urgency"] in ["critical", "near-expiry"]:
            qty_val = r["cleaned_numerics"].get("quantity", 0)
            return qty_val >= MIN_QTY
        return False

    actionable_mask = df.apply(row_is_actionable, axis=1)
    actionable = df[actionable_mask].sort_values("score", ascending=False)

    recs = []
    for _, row in actionable.iterrows():
        # Corrupted numeric input check
        if not row.get("is_numeric_valid", True):
            num_errs = row.get("numeric_errors", ["Invalid numeric inventory data"])
            clean_reason = f"Corrupted numeric inventory data: {'; '.join(num_errs)}. Flagged for manual pharmacist review."
            dte_val = int(row["dte"]) if (row.get("dte") is not None and not pd.isna(row.get("dte"))) else None
            score_comp = get_score_components(row)
            explanation = (
                f"Recommended Action: FLAG_FOR_REVIEW\n\n"
                f"Source: {row.get('branch_name')}\n"
                f"Destination: None\n"
                f"Quantity: {row.get('quantity')}\n\n"
                f"Reason:\n{clean_reason}\n\n"
                f"ML Expiry Risk Prediction: Unavailable"
            )
            rec = {
                "recommended_action":    "FLAG_FOR_REVIEW",
                "action":                "FLAG_FOR_REVIEW",
                "source_branch":         row.get("branch_name"),
                "source_branch_id":      row.get("branch_id"),
                "destination_branch":    None,
                "destination_branch_id": None,
                "suggested_quantity":    0,
                "risk_urgency":          "invalid",
                "score":                 0.0,
                "score_components":      score_comp,
                "urgency_score":         0.0,
                "quantity_score":        0.0,
                "value_score":           0.0,
                "reason":                clean_reason,
                "explanation":           explanation,
                "decision_factors":      {
                    "days_to_expiry":     dte_val,
                    "current_stock":      row.get("quantity"),
                    "demand":             row.get("demand_per_week"),
                    "destination_demand": None,
                    "available_capacity": None,
                    "transfer_time_days": transfer_days,
                    "medicine_value_gbp": 0.0,
                    "risk_score":         0.0,
                    "score_components":   score_comp,
                    "urgency_score":      0.0,
                    "quantity_score":     0.0,
                    "value_score":        0.0,
                    "ml_risk_probability": None,
                    "ml_risk_class":      "Unavailable",
                },
                "ml_risk_probability":   None,
                "ml_risk_class":         "Unavailable",
                "batch_id":              row.get("batch_id"),
                "medicine_name":         row.get("medicine_name"),
                "category":              row.get("category"),
                "branch_name":           row.get("branch_name"),
                "branch_id":             row.get("branch_id"),
                "quantity":              row.get("quantity"),
                "expiry_date":           row.get("expiry_date"),
                "dte":                   dte_val,
                "days_to_expiry":        dte_val,
                "urgency":               "invalid",
                "stock_value":           0.0,
                "unit_cost_gbp":         row.get("unit_cost_gbp"),
                "confidence":            "LOW",
                "destinations":          [],
                "is_high_impact":        False,
                "requires_confirmation": False,
                "is_feasible":           False,
                "transfer_time_days":    transfer_days,
            }
            recs.append(rec)
            continue

        dests, status, msg = find_destinations(row, candidate_pool, transfer_days=transfer_days)
        high_impact = (row["stock_value"] > HIGH_VALUE or
                       row["quantity"]    > HIGH_QTY)

        source_demand = int(row.get("demand_per_week", 0))

        raw_prob = row.get("ml_risk_probability")
        if raw_prob is not None and not pd.isna(raw_prob):
            try:
                ml_prob = float(raw_prob)
            except (ValueError, TypeError):
                ml_prob = None
        else:
            ml_prob = None

        raw_class = row.get("ml_risk_class")
        if raw_class is not None and not pd.isna(raw_class):
            ml_class = str(raw_class)
        else:
            ml_class = "Unavailable"

        if ml_prob is None and ml_class not in ["Low", "Medium", "High"]:
            ml_class = "Unavailable"

        # If expiry date is invalid, ML model cannot make a valid expiry prediction
        if row.get("dte") is None or (isinstance(row.get("dte"), float) and pd.isna(row.get("dte"))):
            ml_prob = None
            ml_class = "Unavailable"

        ml_summary = (
            f"ML Expiry Risk Prediction: {ml_class} ({int(ml_prob * 100)}% probability)"
            if ml_prob is not None
            else f"ML Expiry Risk Prediction: {ml_class}"
        )

        score_comp = get_score_components(row)

        if status != "OK" or not dests:
            dte_val = int(row["dte"]) if (row.get("dte") is not None and not pd.isna(row.get("dte"))) else None
            dte_str = f"{dte_val} days left" if dte_val is not None else "Unknown days left"
            if status == "INVALID_EXPIRY" or "invalid" in msg.lower():
                clean_reason = msg
            elif "zero demand" in msg.lower():
                clean_reason = (
                    f"The medicine is approaching expiry ({dte_str}, risk score: {row['score']}), "
                    f"but all other branches report zero demand. Flagged for manual pharmacist review."
                )
            elif "capacity" in msg.lower():
                clean_reason = (
                    f"The medicine is approaching expiry ({dte_str}, risk score: {row['score']}), "
                    f"but all receiving branches lack sufficient available capacity for {int(row['quantity'])} units. "
                    f"Flagged for manual pharmacist review."
                )
            elif "transit time" in msg.lower() or "shelf life" in msg.lower():
                clean_reason = (
                    f"The medicine is approaching expiry ({dte_str}), "
                    f"leaving insufficient shelf life for safe transit time (~{transfer_days}d transit). "
                    f"Flagged for urgent local dispensing or disposal."
                )
            else:
                clean_reason = msg

            decision_factors = {
                "days_to_expiry":     dte_val,
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
                f"{ml_summary}"
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
                "dte":                   dte_val,
                "days_to_expiry":        dte_val,
                "urgency":               row["urgency"],
                "stock_value":           float(row["stock_value"]),
                "unit_cost_gbp":         float(row["unit_cost_gbp"]),
                "confidence":            "LOW",
                "destinations":          [],
                "is_high_impact":        high_impact,
                "requires_confirmation": high_impact,
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

            dte_val = int(row["dte"]) if (row.get("dte") is not None and not pd.isna(row.get("dte"))) else None

            decision_factors = {
                "days_to_expiry":     dte_val,
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
                    f"{ml_summary}"
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
                    f"{ml_summary}"
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
                "dte":                   dte_val,
                "days_to_expiry":        dte_val,
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
    df["dte"] = df["expiry_date"].apply(days_to_expiry)
    qty_num = pd.to_numeric(df.get("quantity", df.get("current_stock", 0)), errors="coerce")
    cost_num = pd.to_numeric(df.get("unit_cost_gbp", df.get("unit_cost", 0)), errors="coerce")
    valid_val_mask = qty_num.notna() & (qty_num >= 0) & cost_num.notna() & (cost_num >= 0)
    df["stock_value"] = (qty_num * cost_num).where(valid_val_mask, 0.0)
    at_risk = df[df["dte"].between(0, 30) & valid_val_mask]
    return round(float(at_risk["stock_value"].sum()), 2)
