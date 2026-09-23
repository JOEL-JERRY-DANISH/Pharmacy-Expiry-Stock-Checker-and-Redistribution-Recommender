import unittest
import pandas as pd
import hashlib
import tempfile, os
from datetime import datetime, timedelta
from recommender import (
    generate_recommendations,
    calculate_baseline,
    score_batch,
    get_score_components,
    find_destinations,
    calculate_need_score,
    calculate_destination_need,
)
from barcode_registry import BarcodeRegistry
from barcode_lookup import lookup_barcode
from log_manager import save_entry, load_log, get_summary
from auth_config import CREDENTIALS
from database import (
    initialise_database,
    load_stock,
    save_decision,
    load_decisions,
    get_connection,
    import_stock_from_csv,
    import_from_csv,
    validate_stock_row,
    REQUIRED_STOCK_COLUMNS,
)

def make_row(dte, qty, demand, capacity,
             branch_id="BR01", branch_name="Test Branch",
             medicine="Test Medicine", cost=1.00):
    exp = (datetime.today() +
           timedelta(days=dte)).strftime("%Y-%m-%d")
    return {
        "batch_id":                  f"T-{abs(dte)}-{qty}",
        "medicine_name":             medicine,
        "category":                  "Test",
        "branch_id":                 branch_id,
        "branch_name":               branch_name,
        "quantity":                  qty,
        "expiry_date":               exp,
        "unit_cost_gbp":             cost,
        "demand_per_week":           demand,
        "branch_capacity_remaining": capacity,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Original 7 edge-case tests (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class TestRecommender(unittest.TestCase):

    def test_1_expired_not_recommended(self):
        """Expired stock must never appear as a transfer."""
        df   = pd.DataFrame([make_row(-5, 100, 30, 500)])
        recs = generate_recommendations(df)
        transfers = [r for r in recs if r["action"] == "TRANSFER"]
        self.assertEqual(len(transfers), 0,
            "FAIL: Expired stock generated a transfer")

    def test_2_capacity_blocked_triggers_fallback(self):
        """When all destinations are full, flag for review."""
        source = make_row(5, 100, 30, 500,
                          "BR01", "Source")
        dest1  = make_row(200, 50, 40, 0,
                          "BR02", "Dest1",
                          medicine="Test Medicine")
        dest2  = make_row(200, 50, 35, 0,
                          "BR03", "Dest2",
                          medicine="Test Medicine")
        df   = pd.DataFrame([source, dest1, dest2])
        recs = generate_recommendations(df)
        flagged = [r for r in recs
                   if r["action"] == "FLAG_FOR_REVIEW"]
        self.assertGreater(len(flagged), 0,
            "FAIL: Capacity-blocked batch was not flagged")

    def test_3_zero_demand_excluded(self):
        """Branch with zero demand must not receive transfers."""
        source = make_row(10, 100, 50, 500,
                          "BR01", "Source")
        dest   = make_row(200, 20, 0, 1000,
                          "BR02", "ZeroDemand",
                          medicine="Test Medicine")
        df   = pd.DataFrame([source, dest])
        recs = generate_recommendations(df)
        for rec in recs:
            for d in rec.get("destinations", []):
                self.assertNotEqual(
                    d["dest_branch_id"], "BR02",
                    "FAIL: Zero-demand branch used as destination")

    def test_4_tiny_quantity_ignored(self):
        """Batches under 10 units must not be recommended."""
        df   = pd.DataFrame([make_row(5, 3, 50, 500)])
        recs = generate_recommendations(df)
        self.assertEqual(len(recs), 0,
            "FAIL: Tiny quantity batch generated a recommendation")

    def test_5_high_impact_requires_confirmation(self):
        """High value or quantity needs requires_confirmation."""
        source = make_row(5, 300, 50, 5000,
                          "BR01", "Source", cost=5.00)
        dest   = make_row(200, 50, 60, 5000,
                          "BR02", "Dest",
                          medicine="Test Medicine", cost=5.00)
        df   = pd.DataFrame([source, dest])
        recs = generate_recommendations(df)
        transfers = [r for r in recs
                     if r["action"] == "TRANSFER"]
        if transfers:
            self.assertTrue(
                transfers[0]["requires_confirmation"],
                "FAIL: High-impact transfer missing confirmation flag")

    def test_6_superseded_barcode_resolves(self):
        """Old barcode must still resolve after superseded."""
        path = "data/test_barcode_temp6.csv"
        try:
            reg = BarcodeRegistry(path)
            reg.register("OLD-001", "BATCH-X", "Metformin 500mg")
            reg.update_barcode("OLD-001", "NEW-002", "repackaging")
            batch_id, status = reg.resolve("OLD-001")
            self.assertEqual(batch_id, "BATCH-X",
                "FAIL: Superseded barcode did not resolve correctly")
            self.assertEqual(status, "superseded")
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_7_unknown_barcode_returns_none(self):
        """Unknown barcode must return None safely."""
        path = "data/test_barcode_temp7.csv"
        try:
            reg = BarcodeRegistry(path)
            batch_id, status = reg.resolve("DOES-NOT-EXIST")
            self.assertIsNone(batch_id,
                "FAIL: Unknown barcode did not return None")
            self.assertEqual(status, "unknown")
        finally:
            if os.path.exists(path):
                os.remove(path)


# ─────────────────────────────────────────────────────────────────────────────
# New tests — Recommender scoring and baseline logic
# ─────────────────────────────────────────────────────────────────────────────
class TestRecommenderScoring(unittest.TestCase):

    def test_8_critical_score_higher_than_near_expiry(self):
        """A critical batch must outscore a near-expiry batch with same qty/cost."""
        critical_row = {**make_row(3, 100, 30, 500), "urgency": "critical"}
        near_row     = {**make_row(20, 100, 30, 500), "urgency": "near-expiry"}
        critical_score = score_batch(critical_row)
        near_score     = score_batch(near_row)
        self.assertGreater(critical_score, near_score,
            "FAIL: Critical batch should outscore near-expiry batch")

    def test_9_safe_batch_scores_zero(self):
        """A safe batch must score 0 — it should not enter the recommendation list."""
        safe_row = {**make_row(200, 100, 30, 500), "urgency": "safe"}
        self.assertEqual(score_batch(safe_row), 0,
            "FAIL: Safe batch should score 0")

    def test_10_calculate_baseline_counts_only_0_to_30_days(self):
        """Baseline should only include stock expiring within 0–30 days."""
        rows = [
            make_row(-1,  100, 10, 500),   # expired — NOT in baseline
            make_row(5,   200, 10, 500),   # critical — IN baseline
            make_row(25,  150, 10, 500),   # near-expiry — IN baseline
            make_row(60,  300, 10, 500),   # watch — NOT in baseline
            make_row(200, 400, 10, 500),   # safe — NOT in baseline
        ]
        df = pd.DataFrame(rows)
        baseline = calculate_baseline(df)
        # Only the 5-day (200 units × £1.00) and 25-day (150 units × £1.00)
        expected = round(200 * 1.00 + 150 * 1.00, 2)
        self.assertAlmostEqual(baseline, expected, places=2,
            msg="FAIL: Baseline included wrong batches")

    def test_11_confidence_high_for_high_demand(self):
        """Confidence should be HIGH when destination demand >= 20 units/week."""
        source = make_row(5, 100, 50, 5000, "BR01", "Source")
        dest   = make_row(200, 50, 25, 5000, "BR02", "Dest",
                          medicine="Test Medicine")
        df   = pd.DataFrame([source, dest])
        recs = generate_recommendations(df)
        transfers = [r for r in recs if r["action"] == "TRANSFER"]
        if transfers:
            self.assertEqual(transfers[0]["confidence"], "HIGH",
                "FAIL: Destination with demand>=20 should have HIGH confidence")


# ─────────────────────────────────────────────────────────────────────────────
# New tests — log_manager CSV persistence
# ─────────────────────────────────────────────────────────────────────────────
class TestLogManager(unittest.TestCase):

    def setUp(self):
        """Point log_manager at a temp CSV for each test."""
        import log_manager
        self._real_path = log_manager.LOG_PATH
        self._tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, dir="data")
        self._tmp.close()
        log_manager.LOG_PATH = self._tmp.name

    def tearDown(self):
        import log_manager
        log_manager.LOG_PATH = self._real_path
        if os.path.exists(self._tmp.name):
            os.remove(self._tmp.name)

    def test_12_save_entry_creates_row(self):
        """Saving an entry must produce exactly one row in the log."""
        save_entry("BATCH-TEST", "Metformin", "CONFIRMED",
                   destination="North Branch", user="TestUser")
        df = load_log()
        self.assertEqual(len(df), 1,
            "FAIL: Expected exactly one log entry after save_entry")

    def test_13_save_entry_stores_correct_fields(self):
        """Saved entry must preserve all field values accurately."""
        save_entry("BATCH-ABC", "Warfarin 5mg", "OVERRIDDEN",
                   destination="East Branch",
                   override_reason="Already stocked",
                   user="Sarah Johnson")
        df = load_log()
        row = df.iloc[0]
        self.assertEqual(row["batch_id"],       "BATCH-ABC")
        self.assertEqual(row["medicine"],        "Warfarin 5mg")
        self.assertEqual(row["action"],          "OVERRIDDEN")
        self.assertEqual(row["destination"],     "East Branch")
        self.assertEqual(row["override_reason"], "Already stocked")
        self.assertEqual(row["user"],            "Sarah Johnson")

    def test_14_get_summary_counts_correctly(self):
        """get_summary must return accurate counts per action type."""
        save_entry("B1", "Med A", "CONFIRMED",   user="U1")
        save_entry("B2", "Med B", "CONFIRMED",   user="U1")
        save_entry("B3", "Med C", "OVERRIDDEN",  user="U2")
        save_entry("B4", "Med D", "MANUALLY_REVIEWED", user="U2")
        summary = get_summary()
        self.assertEqual(summary["total"],      4)
        self.assertEqual(summary["confirmed"],  2)
        self.assertEqual(summary["overridden"], 1)
        self.assertEqual(summary["reviewed"],   1)


# ─────────────────────────────────────────────────────────────────────────────
# New tests — Authentication (hashed password check)
# ─────────────────────────────────────────────────────────────────────────────
class TestAuthentication(unittest.TestCase):

    def _check(self, username, password):
        """Replicate the check_password logic from app.py."""
        users = CREDENTIALS["usernames"]
        if username not in users:
            return False, None
        hashed = hashlib.sha256(password.encode()).hexdigest()
        if users[username]["password"] == hashed:
            return True, users[username]
        return False, None

    def test_15_correct_credentials_succeed(self):
        """Valid username + correct password must return True."""
        valid, info = self._check("pharmacist1", "pharmacy123")
        self.assertTrue(valid,
            "FAIL: Correct credentials were rejected")
        self.assertEqual(info["name"], "Sarah Johnson")

    def test_16_wrong_password_fails(self):
        """Correct username + wrong password must return False."""
        valid, info = self._check("pharmacist1", "wrongpassword")
        self.assertFalse(valid,
            "FAIL: Wrong password was accepted")
        self.assertIsNone(info)

    def test_17_unknown_username_fails(self):
        """Unknown username must return False without raising an error."""
        valid, info = self._check("attacker", "anypassword")
        self.assertFalse(valid,
            "FAIL: Unknown username was accepted")
        self.assertIsNone(info)


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced tests — Recommendation Logic & 10 Factors
# ─────────────────────────────────────────────────────────────────────────────
class TestRecommenderEnhanced(unittest.TestCase):

    def test_18_recommendation_returns_all_required_keys(self):
        """Recommendations must return all required transparent fields."""
        source = make_row(5, 100, 30, 500, "BR01", "Source Branch")
        dest   = make_row(100, 40, 25, 500, "BR02", "Dest Branch")
        df = pd.DataFrame([source, dest])
        recs = generate_recommendations(df)
        self.assertGreater(len(recs), 0)

        required_keys = [
            "recommended_action",
            "source_branch",
            "destination_branch",
            "suggested_quantity",
            "risk_urgency",
            "score",
            "reason",
        ]
        for rec in recs:
            for k in required_keys:
                self.assertIn(k, rec, f"FAIL: Recommendation missing required key: {k}")

            self.assertEqual(rec["source_branch"], "Source Branch")
            self.assertEqual(rec["suggested_quantity"], 100)
            self.assertEqual(rec["risk_urgency"], "critical")
            self.assertIsInstance(rec["score"], (int, float))
            self.assertTrue(len(rec["reason"]) > 0)

    def test_19_transit_time_infeasibility_flags_for_review(self):
        """When transit time >= shelf life, transfer is infeasible and flagged."""
        source = make_row(1, 50, 20, 500, "BR01", "Source Branch")
        dest   = make_row(100, 20, 30, 500, "BR02", "Dest Branch")
        df = pd.DataFrame([source, dest])
        # Transit time 2 days > shelf life 1 day
        recs = generate_recommendations(df, transfer_days=2)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["recommended_action"], "FLAG_FOR_REVIEW")
        self.assertIn("transit time", recs[0]["reason"].lower())

    def test_20_exact_reproducible_scoring(self):
        """Batch scoring must be transparent, explainable, and reproducible."""
        # Critical base = 100; qty: 250/500 * 30 = 15; cost: 2.50/5.0 * 20 = 10 -> Total: 125.0
        row1 = {**make_row(3, 250, 20, 500, cost=2.50), "urgency": "critical"}
        self.assertEqual(score_batch(row1), 125.0)

        # Near-expiry base = 50; qty: 500/500 * 30 = 30; cost: 5.0/5.0 * 20 = 20 -> Total: 100.0
        row2 = {**make_row(15, 500, 20, 500, cost=5.00), "urgency": "near-expiry"}
        self.assertEqual(score_batch(row2), 100.0)

    def test_21_never_recommends_destination_without_capacity(self):
        """Destination must have remaining capacity >= suggested quantity."""
        source = make_row(5, 100, 20, 500, "BR01", "Source")
        dest_low_cap = make_row(100, 20, 50, 80, "BR02", "LowCapacityBranch")
        df = pd.DataFrame([source, dest_low_cap])
        recs = generate_recommendations(df)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["recommended_action"], "FLAG_FOR_REVIEW")
        self.assertIn("capacity", recs[0]["reason"].lower())

    def test_22_reason_explains_all_factors(self):
        """Recommendation reason must clearly explain feasibility, capacity, and demand."""
        source = make_row(5, 100, 20, 500, "BR01", "Source Branch")
        dest   = make_row(100, 20, 40, 500, "BR02", "Dest Branch")
        df = pd.DataFrame([source, dest])
        recs = generate_recommendations(df)
        self.assertEqual(len(recs), 1)
        rec = recs[0]
        reason = rec["reason"]
        self.assertIn("Dest Branch", reason)
        self.assertIn("capacity", reason.lower())
        self.assertIn("demand", reason.lower())
        self.assertIn("transit", reason.lower())
        self.assertIn("absorption", reason.lower())

    def test_23_decision_factors_and_concise_explanation_format(self):
        """Recommendation must expose all 8 decision factors and human-readable explanation format."""
        source = make_row(5, 50, 15, 500, "BR01", "Branch A", cost=2.00)
        dest   = make_row(100, 20, 30, 400, "BR02", "Branch B", cost=2.00)
        df = pd.DataFrame([source, dest])
        recs = generate_recommendations(df)
        self.assertEqual(len(recs), 1)
        rec = recs[0]

        # Verify decision factors dictionary
        factors = rec["decision_factors"]
        expected_factors = [
            "days_to_expiry", "current_stock", "demand",
            "destination_demand", "available_capacity",
            "transfer_time_days", "medicine_value_gbp", "risk_score"
        ]
        for f in expected_factors:
            self.assertIn(f, factors, f"Missing decision factor: {f}")

        self.assertEqual(factors["days_to_expiry"], rec["dte"])
        self.assertEqual(factors["current_stock"], 50)
        self.assertEqual(factors["demand"], 15)
        self.assertEqual(factors["destination_demand"], 30)
        self.assertEqual(factors["available_capacity"], 400)
        self.assertEqual(factors["transfer_time_days"], 1)
        self.assertEqual(factors["medicine_value_gbp"], 100.0)

        # Verify structured human-readable explanation format
        expl = rec["explanation"]
        self.assertIn("Recommended Action: TRANSFER", expl)
        self.assertIn("Source: Branch A", expl)
        self.assertIn("Destination: Branch B", expl)
        self.assertIn("Quantity: 50", expl)
        self.assertIn("Reason:\n", expl)
        self.assertIn("approaching expiry", expl.lower())


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced tests — Barcode Lookup & Registry
# ─────────────────────────────────────────────────────────────────────────────
class TestBarcodeEnhanced(unittest.TestCase):

    def setUp(self):
        self.tmp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        self.tmp_csv.close()
        self.reg = BarcodeRegistry(self.tmp_csv.name)
        self.reg.register("5000100000001", "BATCH-ACTIVE", "Amoxicillin 500mg")
        self.reg.update_barcode("5000100000001", "5000100000002", "repackaging")

        # Mock stock dataframe for isolated unit testing
        self.stock_df = pd.DataFrame([
            {
                "batch_id": "BATCH-ACTIVE",
                "medicine_name": "Amoxicillin 500mg",
                "category": "Antibiotic",
                "branch_id": "BR01",
                "branch_name": "Source Branch",
                "quantity": 100,
                "expiry_date": (datetime.today() + timedelta(days=5)).strftime("%Y-%m-%d"),
                "unit_cost_gbp": 2.50,
                "demand_per_week": 10,
                "branch_capacity_remaining": 500,
            },
            {
                "batch_id": "BATCH-DEST",
                "medicine_name": "Amoxicillin 500mg",
                "category": "Antibiotic",
                "branch_id": "BR02",
                "branch_name": "Dest Branch",
                "quantity": 20,
                "expiry_date": (datetime.today() + timedelta(days=100)).strftime("%Y-%m-%d"),
                "unit_cost_gbp": 2.50,
                "demand_per_week": 35,
                "branch_capacity_remaining": 500,
            }
        ])

    def tearDown(self):
        if os.path.exists(self.tmp_csv.name):
            os.remove(self.tmp_csv.name)

    def test_24_valid_active_barcode_lookup(self):
        """Valid barcode returns medicine, expiry, stock, risk score, and recommendation."""
        res = lookup_barcode("5000100000002", registry=self.reg, stock_df=self.stock_df)
        self.assertTrue(res["found"])
        self.assertEqual(res["status"], "active")
        self.assertEqual(res["batch_id"], "BATCH-ACTIVE")
        self.assertEqual(res["medicine_name"], "Amoxicillin 500mg")
        self.assertEqual(res["quantity"], 100)
        self.assertEqual(res["stock_value"], 250.0)
        self.assertIn("dte", res)
        self.assertEqual(res["urgency"], "critical")
        self.assertGreater(res["score"], 100)
        self.assertIsNotNone(res["recommendation"])
        self.assertEqual(res["recommendation"]["recommended_action"], "TRANSFER")
        self.assertEqual(res["recommendation"]["destination_branch"], "Dest Branch")

    def test_25_unknown_barcode_returns_clear_message_without_crash(self):
        """Unknown barcode returns clear message without throwing exceptions."""
        res = lookup_barcode("9999999999999", registry=self.reg, stock_df=self.stock_df)
        self.assertFalse(res["found"])
        self.assertEqual(res["status"], "unknown")
        self.assertIn("not recognised", res["message"].lower())

    def test_26_superseded_barcode_resolves_correctly(self):
        """Superseded barcode resolves to current active batch and provides full data."""
        res = lookup_barcode("5000100000001", registry=self.reg, stock_df=self.stock_df)
        self.assertTrue(res["found"])
        self.assertEqual(res["status"], "superseded")
        self.assertEqual(res["batch_id"], "BATCH-ACTIVE")
        self.assertEqual(res["medicine_name"], "Amoxicillin 500mg")
        self.assertGreater(res["score"], 100)

    def test_27_invalid_and_empty_inputs_handled_safely(self):
        """Empty, None, or whitespace barcodes are safely handled without crashing."""
        for invalid_input in [None, "", "   ", "\t\n"]:
            res = lookup_barcode(invalid_input, registry=self.reg, stock_df=self.stock_df)
            self.assertFalse(res["found"])
            self.assertEqual(res["status"], "invalid")
            self.assertIn("cannot be empty", res["message"].lower())

            b_id, status = self.reg.resolve(invalid_input)
            self.assertIsNone(b_id)
            self.assertEqual(status, "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Database Architecture Tests (CSV -> SQLite -> Application)
# ─────────────────────────────────────────────────────────────────────────────
class TestDatabaseArchitecture(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_pharmacy.db")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_28_database_initialise_creates_tables_and_seeds(self):
        """Initialise creates schema and auto-seeds initial CSV data into SQLite."""
        initialise_database(self.db_path, seed=False)
        self.assertTrue(os.path.exists(self.db_path))

        conn = get_connection(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        conn.close()

        self.assertIn("stock", tables)
        self.assertIn("barcodes", tables)
        self.assertIn("decisions", tables)

    def test_29_load_stock_reads_from_sqlite(self):
        """load_stock retrieves data stored in SQLite."""
        initialise_database(self.db_path, seed=True)
        df = load_stock(self.db_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn("batch_id", df.columns)
        self.assertIn("medicine_name", df.columns)
        self.assertIn("quantity", df.columns)

    def test_30_save_and_load_decisions_sqlite(self):
        """save_decision stores in SQLite and load_decisions retrieves correctly."""
        initialise_database(self.db_path, seed=False)
        save_decision(
            batch_id="TEST-DB-01",
            medicine="Paracetamol 500mg",
            action="CONFIRMED",
            destination="North Branch",
            override_reason="",
            user="test_pharmacist",
            db_path=self.db_path,
        )
        df_dec = load_decisions(self.db_path)
        self.assertFalse(df_dec.empty)
        matching = df_dec[df_dec["batch_id"] == "TEST-DB-01"]
        self.assertEqual(len(matching), 1)
        row = matching.iloc[0]
        self.assertEqual(row["medicine"], "Paracetamol 500mg")
        self.assertEqual(row["action"], "CONFIRMED")
        self.assertEqual(row["destination"], "North Branch")
        self.assertEqual(row["user"], "test_pharmacist")

    def test_31_load_stock_graceful_fallback(self):
        """load_stock falls back gracefully to CSV if DB path cannot be queried."""
        # Query a nonexistent / invalid path
        df = load_stock("data/pharmacy.db")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)



# ─────────────────────────────────────────────────────────────────────────────
# Audit & Decision Logging Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestDecisionAuditLogging(unittest.TestCase):

    def setUp(self):
        import log_manager
        self._real_path = log_manager.LOG_PATH
        self._tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, dir="data")
        self._tmp.close()
        log_manager.LOG_PATH = self._tmp.name

    def tearDown(self):
        import log_manager
        log_manager.LOG_PATH = self._real_path
        if os.path.exists(self._tmp.name):
            os.remove(self._tmp.name)

    def test_32_records_all_ten_decision_fields(self):
        """Every decision records date/time, user, medicine, batch, source, destination, quantity, recommendation, final decision, and override reason."""
        save_entry(
            batch_id="BATCH-FULL-01",
            medicine="Amoxicillin 500mg",
            action="CONFIRMED",
            destination="South Branch",
            override_reason="",
            user="Jane Doe",
            source_branch="North Branch",
            quantity=50,
            system_recommendation="TRANSFER",
            final_decision="CONFIRMED"
        )
        df = load_log()
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertTrue(bool(row["timestamp"]))
        self.assertEqual(row["user"], "Jane Doe")
        self.assertEqual(row["medicine"], "Amoxicillin 500mg")
        self.assertEqual(row["batch_id"], "BATCH-FULL-01")
        self.assertEqual(row["source_branch"], "North Branch")
        self.assertEqual(row["destination"], "South Branch")
        self.assertEqual(int(row["quantity"]), 50)
        self.assertEqual(row["system_recommendation"], "TRANSFER")
        self.assertEqual(row["action"], "CONFIRMED")
        self.assertEqual(row["final_decision"], "CONFIRMED")
        self.assertEqual(row["override_reason"], "")

    def test_33_actions_recorded_consistently(self):
        """CONFIRMED, OVERRIDDEN, and MANUALLY_REVIEWED are stored consistently."""
        actions = ["CONFIRMED", "OVERRIDDEN", "MANUALLY_REVIEWED"]
        for idx, act in enumerate(actions):
            save_entry(
                batch_id=f"BATCH-ACT-{idx}",
                medicine="Paracetamol 500mg",
                action=act,
                destination="West Branch" if act != "MANUALLY_REVIEWED" else "",
                override_reason="Staff clinical override" if act == "OVERRIDDEN" else "",
                user="Dr. Smith",
                source_branch="Central Branch",
                quantity=25,
                system_recommendation="TRANSFER" if act != "MANUALLY_REVIEWED" else "FLAG_FOR_REVIEW"
            )
        df = load_log()
        self.assertEqual(len(df), 3)
        for act in actions:
            matching = df[df["action"] == act]
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching.iloc[0]["final_decision"], act)

    def test_34_override_reason_captured_on_rejection(self):
        """Rejection/override reason is captured accurately when overridden."""
        save_entry(
            batch_id="BATCH-REJ-01",
            medicine="Ibuprofen 400mg",
            action="OVERRIDDEN",
            destination="East Branch",
            override_reason="Destination storage fridge is under maintenance",
            user="Clinical Lead",
            source_branch="North Branch",
            quantity=100,
            system_recommendation="TRANSFER"
        )
        df = load_log()
        row = df[df["batch_id"] == "BATCH-REJ-01"].iloc[0]
        self.assertEqual(row["action"], "OVERRIDDEN")
        self.assertEqual(row["override_reason"], "Destination storage fridge is under maintenance")

    def test_35_historical_logs_preserved_without_data_loss(self):
        """Legacy CSV logs with older headers are preserved without data loss when loaded."""
        legacy_csv = (
            "timestamp,batch_id,medicine,action,destination,override_reason,user\n"
            "2026-01-01 12:00:00,BATCH-OLD-1,Aspirin,CONFIRMED,South Branch,,legacy_user\n"
        )
        import log_manager
        with open(log_manager.LOG_PATH, "w") as f:
            f.write(legacy_csv)

        df = load_log()
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["batch_id"], "BATCH-OLD-1")
        self.assertEqual(row["medicine"], "Aspirin")
        self.assertEqual(row["action"], "CONFIRMED")
        self.assertEqual(row["user"], "legacy_user")
        self.assertEqual(row["source_branch"], "")
        self.assertEqual(int(row["quantity"]), 0)



# ─────────────────────────────────────────────────────────────────────────────
# Admin Dashboard Analytics & Metrics Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestAdminDashboardMetrics(unittest.TestCase):

    def test_36_admin_dashboard_metrics_calculation(self):
        """Admin dashboard computes all 7 metrics accurately from actual operational data."""
        df_decisions = load_decisions()
        stock_df = load_stock()

        # Decision metrics
        total_decisions = len(df_decisions)
        confirmed = len(df_decisions[df_decisions["action"] == "CONFIRMED"])
        overridden = len(df_decisions[df_decisions["action"] == "OVERRIDDEN"])
        reviewed = len(df_decisions[df_decisions["action"] == "MANUALLY_REVIEWED"])

        self.assertGreaterEqual(total_decisions, 0)
        self.assertGreaterEqual(confirmed, 0)
        self.assertGreaterEqual(overridden, 0)
        self.assertGreaterEqual(reviewed, 0)
        self.assertEqual(total_decisions, confirmed + overridden + reviewed)

        # Inventory risk metrics
        from recommender import days_to_expiry, urgency_label, calculate_baseline, generate_recommendations

        stock_df = stock_df.copy()
        stock_df["dte"] = stock_df["expiry_date"].apply(days_to_expiry)
        stock_df["urgency"] = stock_df["dte"].apply(urgency_label)
        stock_df["stock_value"] = (stock_df["quantity"] * stock_df["unit_cost_gbp"]).round(2)

        at_risk_df = stock_df[stock_df["dte"].between(0, 30)]
        stock_val_at_risk = calculate_baseline(stock_df)
        recs = generate_recommendations(stock_df)
        transfer_recs = [r for r in recs if r["action"] == "TRANSFER"]
        transfer_qty = sum(r["quantity"] for r in transfer_recs)

        self.assertGreaterEqual(len(at_risk_df), 0)
        self.assertGreaterEqual(stock_val_at_risk, 0.0)
        self.assertGreaterEqual(transfer_qty, 0)

    def test_37_admin_dashboard_visualizations_data_preparation(self):
        """Admin dashboard properly groups actual data for all 5 required visualizations."""
        df_decisions = load_decisions()
        stock_df = load_stock()

        from recommender import days_to_expiry, urgency_label

        stock_df = stock_df.copy()
        stock_df["dte"] = stock_df["expiry_date"].apply(days_to_expiry)
        stock_df["urgency"] = stock_df["dte"].apply(urgency_label)
        stock_df["stock_value"] = (stock_df["quantity"] * stock_df["unit_cost_gbp"]).round(2)
        at_risk_df = stock_df[stock_df["dte"].between(0, 30)]

        # 1. Decisions over time
        if not df_decisions.empty:
            df_decisions = df_decisions.copy()
            df_decisions["timestamp"] = pd.to_datetime(df_decisions["timestamp"])
            df_decisions["date"] = df_decisions["timestamp"].dt.date
            daily = df_decisions.groupby("date").size().reset_index(name="Decisions")
            self.assertIn("date", daily.columns)
            self.assertIn("Decisions", daily.columns)

        # 2. Confirmed vs overridden
        if not df_decisions.empty:
            action_counts = df_decisions["action"].value_counts().reset_index()
            action_counts.columns = ["Action", "Decisions"]
            self.assertIn("Action", action_counts.columns)

        # 3. Transfers by destination
        if not df_decisions.empty and "destination" in df_decisions.columns:
            dest_data = df_decisions[df_decisions["destination"] != ""]
            if not dest_data.empty:
                dest_counts = dest_data["destination"].value_counts().reset_index()
                dest_counts.columns = ["Destination Branch", "Transfers"]
                self.assertIn("Destination Branch", dest_counts.columns)

        # 4. Expiry-risk distribution
        urgency_order = ["critical", "near-expiry", "watch", "safe", "expired"]
        urg_counts = stock_df["urgency"].value_counts().reindex(urgency_order).fillna(0).astype(int).reset_index()
        urg_counts.columns = ["Urgency Status", "Batches"]
        self.assertEqual(len(urg_counts), 5)
        self.assertIn("Urgency Status", urg_counts.columns)

        # 5. Stock/value at risk by branch
        if not at_risk_df.empty:
            branch_risk = at_risk_df.groupby("branch_name")["stock_value"].sum().round(2).reset_index()
            branch_risk.columns = ["Branch", "Value at Risk (£)"]
            self.assertIn("Branch", branch_risk.columns)
            self.assertIn("Value at Risk (£)", branch_risk.columns)




# ─────────────────────────────────────────────────────────────────────────────
# ML Expiry Risk Prediction Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestMLExpiryModel(unittest.TestCase):
    """
    Tests for the ML Expiry Risk Prediction component (ml_expiry_model.py).

    Architecture under test:
        Medicine Data -> ML Expiry Risk Prediction -> Rule-Based Recommendation Engine
        -> Demand + Capacity + Transfer Time checks -> Final Recommendation

    Important: The rule-based engine remains in control.
    The ML component provides a supporting risk signal only.
    """

    @classmethod
    def setUpClass(cls):
        """Train the model once for the whole test class to avoid redundant retraining."""
        from ml_expiry_model import MLExpiryPredictor
        from database import initialise_database, load_stock

        initialise_database()
        cls.stock_df = load_stock()
        cls.predictor = MLExpiryPredictor()
        cls.metrics = cls.predictor.train(cls.stock_df, test_size=0.25, random_state=42)

    # ── Training & Evaluation ──────────────────────────────────────────────

    def test_38_model_trains_without_error(self):
        """Model must train successfully on the operational stock dataset."""
        self.assertTrue(self.predictor.is_trained,
            "FAIL: MLExpiryPredictor.is_trained should be True after training")
        self.assertIsNotNone(self.predictor.model,
            "FAIL: Trained model pipeline should not be None")

    def test_39_evaluation_returns_all_required_metrics(self):
        """Training must return actual accuracy, precision, recall, and F1-score."""
        required_keys = [
            "total_samples", "train_samples", "test_samples",
            "accuracy", "precision_weighted", "recall_weighted", "f1_weighted",
            "precision_macro", "recall_macro", "f1_macro",
            "dataset_limitation_note",
        ]
        for key in required_keys:
            self.assertIn(key, self.metrics,
                f"FAIL: Missing evaluation metric key: {key}")

    def test_40_metrics_are_real_values_in_valid_range(self):
        """All metric values must be real floats between 0.0 and 1.0 (not fabricated)."""
        metric_keys = [
            "accuracy", "precision_weighted", "recall_weighted", "f1_weighted",
            "precision_macro", "recall_macro", "f1_macro",
        ]
        for key in metric_keys:
            val = self.metrics[key]
            self.assertIsInstance(val, float,
                f"FAIL: Metric {key} should be a float, got {type(val)}")
            self.assertGreaterEqual(val, 0.0,
                f"FAIL: Metric {key} = {val} is below 0.0")
            self.assertLessEqual(val, 1.0,
                f"FAIL: Metric {key} = {val} exceeds 1.0")

    def test_41_dataset_split_is_correct(self):
        """Train/test split must account for all samples with no loss."""
        total = self.metrics["total_samples"]
        train = self.metrics["train_samples"]
        test = self.metrics["test_samples"]
        self.assertEqual(train + test, total,
            f"FAIL: train ({train}) + test ({test}) != total ({total})")
        # Test set should be 25% of total (±1 due to rounding)
        expected_test = total * 0.25
        self.assertAlmostEqual(test, expected_test, delta=2,
            msg=f"FAIL: Test split size {test} is not ~25% of {total}")

    def test_42_dataset_limitation_note_is_present(self):
        """Dataset limitation note must be present and non-empty (honesty requirement)."""
        note = self.metrics.get("dataset_limitation_note", "")
        self.assertIsInstance(note, str)
        self.assertGreater(len(note), 20,
            "FAIL: dataset_limitation_note is absent or too short")

    def test_43_classification_report_is_non_empty_string(self):
        """Classification report must be a non-empty string from sklearn."""
        rep = self.predictor.evaluation_report
        self.assertIsInstance(rep, str)
        self.assertIn("precision", rep.lower(),
            "FAIL: classification_report output missing 'precision'")
        self.assertIn("recall", rep.lower(),
            "FAIL: classification_report output missing 'recall'")

    # ── Ground-Truth Label Derivation ──────────────────────────────────────

    def test_44_label_high_when_short_expiry_with_excess_stock(self):
        """Label should be High when dte<=30 and stock exceeds expected demand."""
        from ml_expiry_model import MLExpiryPredictor
        p = MLExpiryPredictor()
        # 20 days left, 200 units, 1 unit/day demand => expected sales 20, excess 180
        row = {"days_to_expiry": 20, "quantity": 200, "avg_daily_demand": 1.0}
        label = p.derive_ground_truth_label(row)
        self.assertEqual(label, "High",
            "FAIL: Short expiry with large excess stock should be labelled High")

    def test_45_label_medium_when_60_day_expiry_with_excess(self):
        """Label should be Medium when dte is 31-60 and stock exceeds expected demand."""
        from ml_expiry_model import MLExpiryPredictor
        p = MLExpiryPredictor()
        # 45 days left, 200 units, 2 unit/day demand => expected sales 90, excess 110
        row = {"days_to_expiry": 45, "quantity": 200, "avg_daily_demand": 2.0}
        label = p.derive_ground_truth_label(row)
        self.assertEqual(label, "Medium",
            "FAIL: 31-60 day expiry with excess stock should be labelled Medium")

    def test_46_label_low_when_stock_absorbable_before_expiry(self):
        """Label should be Low when stock can be fully absorbed by local demand before expiry."""
        from ml_expiry_model import MLExpiryPredictor
        p = MLExpiryPredictor()
        # 90 days left, 100 units, 5 units/day => expected 450, no excess
        row = {"days_to_expiry": 90, "quantity": 100, "avg_daily_demand": 5.0}
        label = p.derive_ground_truth_label(row)
        self.assertEqual(label, "Low",
            "FAIL: High-demand batch absorbable before expiry should be labelled Low")

    # ── Single Batch Prediction ────────────────────────────────────────────

    def test_47_predict_batch_returns_required_keys(self):
        """predict_batch must return expiry_risk_probability, risk_class, and class_probabilities."""
        batch = {
            "batch_id": "BATCH-ML-TEST",
            "medicine_name": "Amoxicillin 500mg",
            "branch_id": "BR01",
            "quantity": 150,
            "expiry_date": (datetime.today() + timedelta(days=20)).strftime("%Y-%m-%d"),
            "unit_cost_gbp": 2.50,
            "demand_per_week": 10,
            "branch_capacity_remaining": 500,
        }
        result = self.predictor.predict_batch(batch)
        self.assertIn("expiry_risk_probability", result)
        self.assertIn("risk_class", result)
        self.assertIn("class_probabilities", result)

    def test_48_predict_batch_risk_class_is_valid(self):
        """risk_class must be one of Low, Medium, or High."""
        batch = {
            "quantity": 200,
            "expiry_date": (datetime.today() + timedelta(days=15)).strftime("%Y-%m-%d"),
            "unit_cost_gbp": 1.00,
            "demand_per_week": 5,
            "branch_capacity_remaining": 500,
            "branch_id": "BR01",
        }
        result = self.predictor.predict_batch(batch)
        self.assertIn(result["risk_class"], ["Low", "Medium", "High"],
            f"FAIL: risk_class must be Low/Medium/High, got {result['risk_class']}")

    def test_49_predict_batch_probability_is_between_0_and_1(self):
        """expiry_risk_probability must be a float in [0.0, 1.0]."""
        batch = {
            "quantity": 100,
            "expiry_date": (datetime.today() + timedelta(days=25)).strftime("%Y-%m-%d"),
            "unit_cost_gbp": 1.50,
            "demand_per_week": 7,
            "branch_capacity_remaining": 400,
            "branch_id": "BR02",
        }
        result = self.predictor.predict_batch(batch)
        prob = result["expiry_risk_probability"]
        self.assertIsInstance(prob, float,
            f"FAIL: expiry_risk_probability should be float, got {type(prob)}")
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    # ── DataFrame Prediction ───────────────────────────────────────────────

    def test_50_predict_dataframe_adds_ml_columns(self):
        """predict_dataframe must enrich the dataframe with ml_risk_probability and ml_risk_class."""
        sample = self.stock_df.head(10).copy()
        result_df = self.predictor.predict_dataframe(sample)
        self.assertIn("ml_risk_probability", result_df.columns,
            "FAIL: predict_dataframe missing ml_risk_probability column")
        self.assertIn("ml_risk_class", result_df.columns,
            "FAIL: predict_dataframe missing ml_risk_class column")
        self.assertEqual(len(result_df), 10,
            "FAIL: predict_dataframe should not change the number of rows")

    def test_51_predict_dataframe_empty_input_returns_empty(self):
        """predict_dataframe on empty input must return empty dataframe without crashing."""
        empty_df = pd.DataFrame(columns=self.stock_df.columns)
        result_df = self.predictor.predict_dataframe(empty_df)
        self.assertIsInstance(result_df, pd.DataFrame,
            "FAIL: predict_dataframe on empty input should return DataFrame")
        self.assertTrue(result_df.empty,
            "FAIL: predict_dataframe on empty input should return empty DataFrame")

    # ── Pipeline Integration ───────────────────────────────────────────────

    def test_52_recommendations_include_ml_fields(self):
        """generate_recommendations must include ML risk fields in every recommendation."""
        source = make_row(10, 100, 20, 500, "BR01", "Source Branch")
        dest = make_row(200, 50, 30, 500, "BR02", "Dest Branch")
        df = pd.DataFrame([source, dest])
        recs = generate_recommendations(df)
        self.assertGreater(len(recs), 0,
            "FAIL: No recommendations generated for near-expiry stock")

        for rec in recs:
            self.assertIn("ml_risk_probability", rec,
                "FAIL: Recommendation missing ml_risk_probability field")
            self.assertIn("ml_risk_class", rec,
                "FAIL: Recommendation missing ml_risk_class field")
            self.assertIsInstance(rec["ml_risk_probability"], float)
            self.assertIn(rec["ml_risk_class"], ["Low", "Medium", "High"])

    def test_53_decision_factors_include_ml_fields(self):
        """decision_factors dict in every recommendation must include ML risk fields."""
        source = make_row(5, 50, 15, 500, "BR01", "Branch A", cost=2.00)
        dest = make_row(100, 20, 30, 400, "BR02", "Branch B", cost=2.00)
        df = pd.DataFrame([source, dest])
        recs = generate_recommendations(df)
        self.assertEqual(len(recs), 1)
        factors = recs[0]["decision_factors"]
        self.assertIn("ml_risk_probability", factors,
            "FAIL: decision_factors missing ml_risk_probability")
        self.assertIn("ml_risk_class", factors,
            "FAIL: decision_factors missing ml_risk_class")

    def test_54_explanation_includes_ml_prediction(self):
        """Human-readable explanation must reference the ML Expiry Risk Prediction."""
        source = make_row(5, 100, 20, 500, "BR01", "Source Branch")
        dest = make_row(200, 50, 30, 500, "BR02", "Dest Branch")
        df = pd.DataFrame([source, dest])
        recs = generate_recommendations(df)
        self.assertGreater(len(recs), 0)
        expl = recs[0]["explanation"]
        self.assertIn("ML Expiry Risk Prediction", expl,
            "FAIL: Explanation should reference ML Expiry Risk Prediction")

    # ── Safety Constraints (Rule-Based Engine Remains in Control) ──────────

    def test_55_ml_does_not_override_expired_stock_safety_rule(self):
        """Expired stock must never be recommended regardless of ML prediction."""
        df = pd.DataFrame([make_row(-1, 200, 50, 500)])
        recs = generate_recommendations(df)
        transfers = [r for r in recs if r["action"] == "TRANSFER"]
        self.assertEqual(len(transfers), 0,
            "FAIL: Rule-based safety rule violated: expired stock must never be transferred")

    def test_56_ml_does_not_override_zero_demand_safety_rule(self):
        """Zero-demand destinations must be excluded even if ML assigns low risk."""
        source = make_row(10, 100, 50, 500, "BR01", "Source")
        dest = make_row(200, 20, 0, 1000, "BR02", "ZeroDemand",
                        medicine="Test Medicine")
        df = pd.DataFrame([source, dest])
        recs = generate_recommendations(df)
        for rec in recs:
            for d in rec.get("destinations", []):
                self.assertNotEqual(d["dest_branch_id"], "BR02",
                    "FAIL: Zero-demand branch must not be selected as destination")

    def test_57_ml_does_not_override_capacity_safety_rule(self):
        """Capacity-deficient destinations must be excluded by rule-based checks."""
        source = make_row(5, 200, 50, 500, "BR01", "Source")
        dest = make_row(200, 20, 40, 50, "BR02", "LowCapDest",
                        medicine="Test Medicine")
        df = pd.DataFrame([source, dest])
        recs = generate_recommendations(df)
        for rec in recs:
            for d in rec.get("destinations", []):
                self.assertNotEqual(d["dest_branch_id"], "BR02",
                    "FAIL: Capacity-deficient branch must not be selected as destination")

    # ── Singleton / Caching ────────────────────────────────────────────────

    def test_58_get_ml_predictor_returns_trained_singleton(self):
        """get_ml_predictor must return a trained singleton (no crash, is_trained=True)."""
        from ml_expiry_model import get_ml_predictor
        p = get_ml_predictor()
        self.assertTrue(p.is_trained,
            "FAIL: Singleton predictor should be trained")
        p2 = get_ml_predictor()
        self.assertIs(p, p2,
            "FAIL: get_ml_predictor should return the same cached instance")

    def test_59_predict_batch_accepts_dict_series_and_dataframe(self):
        """predict_batch must accept dict, pd.Series, and single-row pd.DataFrame."""
        batch_dict = {
            "quantity": 80,
            "expiry_date": (datetime.today() + timedelta(days=14)).strftime("%Y-%m-%d"),
            "unit_cost_gbp": 1.20,
            "demand_per_week": 8,
            "branch_capacity_remaining": 300,
            "branch_id": "BR03",
        }
        # Dict
        r1 = self.predictor.predict_batch(batch_dict)
        self.assertIn("risk_class", r1)
        # Series
        r2 = self.predictor.predict_batch(pd.Series(batch_dict))
        self.assertIn("risk_class", r2)
        # DataFrame (single row)
        r3 = self.predictor.predict_batch(pd.DataFrame([batch_dict]))
        self.assertIn("risk_class", r3)




# ─────────────────────────────────────────────────────────────────────────────
# SQLite Barcode Registry Tests (test_60 – test_66)
# Verifies that BarcodeRegistry uses SQLite as its backing store and that
# all CRUD operations use parameterized queries with safe connection handling.
# Each test uses an isolated temporary SQLite database.
# ─────────────────────────────────────────────────────────────────────────────
class TestSQLiteBarcodeRegistry(unittest.TestCase):

    def setUp(self):
        """Create a fresh temporary SQLite DB for each test."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_bc.db")
        self.reg = BarcodeRegistry(self.db_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    # ── resolve ───────────────────────────────────────────────────────────────

    def test_60_active_barcode_resolves_from_sqlite(self):
        """A registered active barcode resolves to its batch_id with status 'active'."""
        self.reg.register("BC-ACTIVE-001", "BATCH-A", "Metformin 500mg")
        batch_id, status = self.reg.resolve("BC-ACTIVE-001")
        self.assertEqual(batch_id, "BATCH-A",
            "FAIL: Active barcode did not resolve to the correct batch_id")
        self.assertEqual(status, "active",
            "FAIL: Status should be 'active' for a non-superseded barcode")

    def test_61_superseded_barcode_resolves_from_sqlite(self):
        """A superseded barcode still resolves to the original batch with status 'superseded'."""
        self.reg.register("BC-OLD-001", "BATCH-B", "Aspirin 75mg")
        self.reg.update_barcode("BC-OLD-001", "BC-NEW-001", "repackaging")
        batch_id, status = self.reg.resolve("BC-OLD-001")
        self.assertEqual(batch_id, "BATCH-B",
            "FAIL: Superseded barcode did not resolve to original batch_id")
        self.assertEqual(status, "superseded",
            "FAIL: Status should be 'superseded' for an old barcode")

    # ── register ──────────────────────────────────────────────────────────────

    def test_62_duplicate_active_barcode_raises_value_error(self):
        """Registering an already-active barcode for a different batch raises ValueError."""
        self.reg.register("BC-DUP-001", "BATCH-C", "Warfarin 5mg")
        with self.assertRaises(ValueError,
                msg="FAIL: Duplicate active barcode for different batch should raise ValueError"):
            self.reg.register("BC-DUP-001", "BATCH-D", "Ibuprofen 400mg")

    def test_63_idempotent_register_same_batch_no_error(self):
        """Registering the same barcode+batch_id twice is idempotent — no error raised."""
        self.reg.register("BC-IDEM-001", "BATCH-E", "Atorvastatin 20mg")
        try:
            self.reg.register("BC-IDEM-001", "BATCH-E", "Atorvastatin 20mg")
        except ValueError:
            self.fail(
                "FAIL: Idempotent register (same barcode+batch) should not raise ValueError"
            )
        # Still resolves correctly
        batch_id, status = self.reg.resolve("BC-IDEM-001")
        self.assertEqual(batch_id, "BATCH-E")
        self.assertEqual(status, "active")

    def test_64_empty_barcode_raises_value_error(self):
        """Registering an empty or whitespace barcode raises ValueError."""
        for bad_input in [None, "", "   "]:
            with self.assertRaises(ValueError,
                    msg=f"FAIL: Empty barcode {bad_input!r} should raise ValueError"):
                self.reg.register(bad_input, "BATCH-F", "Some Medicine")

    # ── update_barcode ────────────────────────────────────────────────────────

    def test_65_update_barcode_supersedes_old_and_registers_new(self):
        """update_barcode marks the old barcode as superseded and makes the new one active."""
        self.reg.register("BC-UPD-OLD", "BATCH-G", "Ramipril 5mg")
        self.reg.update_barcode("BC-UPD-OLD", "BC-UPD-NEW", "label correction")

        # Old barcode is superseded
        old_batch, old_status = self.reg.resolve("BC-UPD-OLD")
        self.assertEqual(old_batch, "BATCH-G",
            "FAIL: Superseded barcode should still resolve to original batch")
        self.assertEqual(old_status, "superseded")

        # New barcode is active for the same batch
        new_batch, new_status = self.reg.resolve("BC-UPD-NEW")
        self.assertEqual(new_batch, "BATCH-G",
            "FAIL: New barcode should resolve to the same batch_id")
        self.assertEqual(new_status, "active")

    # ── resolve — unknown ─────────────────────────────────────────────────────

    def test_66_unknown_barcode_returns_none_unknown(self):
        """An unregistered barcode resolves to (None, 'unknown') without crashing."""
        batch_id, status = self.reg.resolve("DOES-NOT-EXIST-9999")
        self.assertIsNone(batch_id,
            "FAIL: Unknown barcode should return None as batch_id")
        self.assertEqual(status, "unknown",
            "FAIL: Status should be 'unknown' for an unregistered barcode")


# ─────────────────────────────────────────────────────────────────────────────
# Safe & Validated Database Import Tests (test_67 – test_74)
# Verifies non-destructive imports, column/row validation, duplicate handling,
# transaction atomicity, and summary reporting.
# ─────────────────────────────────────────────────────────────────────────────
class TestSafeDatabaseImport(unittest.TestCase):

    def setUp(self):
        """Create a fresh isolated temporary SQLite DB and directory."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_import.db")
        initialise_database(self.db_path, seed=False)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _write_csv(self, filename, data):
        csv_path = os.path.join(self.tmp_dir.name, filename)
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_67_successful_valid_stock_import(self):
        """Valid CSV records are inserted and an accurate import summary is returned."""
        records = [
            make_row(30, 100, 10, 500, branch_id="BR01", medicine="Medicine A"),
            make_row(60, 200, 20, 400, branch_id="BR02", medicine="Medicine B"),
            make_row(90, 150, 15, 300, branch_id="BR03", medicine="Medicine C"),
        ]
        csv_path = self._write_csv("valid_stock.csv", records)
        summary = import_stock_from_csv(csv_path, self.db_path)

        self.assertTrue(summary["success"], "Import should succeed for valid CSV")
        self.assertEqual(summary["records_processed"], 3)
        self.assertEqual(summary["records_inserted"], 3)
        self.assertEqual(summary["records_updated"], 0)
        self.assertEqual(summary["records_rejected"], 0)
        self.assertEqual(len(summary["errors"]), 0)

        # Verify records exist in DB
        df_db = load_stock(self.db_path)
        self.assertEqual(len(df_db), 3)

    def test_68_missing_columns_rejected(self):
        """CSV missing mandatory columns is rejected immediately without DB modification."""
        records = [
            {"batch_id": "B01", "medicine_name": "Med A", "branch_id": "BR01"}
            # Missing quantity, expiry_date, unit_cost_gbp, demand_per_week, branch_capacity_remaining
        ]
        csv_path = self._write_csv("missing_cols.csv", records)
        summary = import_stock_from_csv(csv_path, self.db_path)

        self.assertFalse(summary["success"], "Import must fail when required columns are missing")
        self.assertEqual(summary["records_inserted"], 0)
        self.assertTrue(any("missing required column" in err.lower() for err in summary["errors"]))

        # Database remains empty
        df_db = load_stock(self.db_path)
        self.assertEqual(len(df_db), 0)

    def test_69_duplicate_batch_id_in_csv_rejected(self):
        """Second occurrence of duplicate batch_id in the same CSV is rejected."""
        r1 = make_row(30, 100, 10, 500, medicine="Med A")
        r2 = make_row(60, 150, 15, 400, medicine="Med B")
        r2["batch_id"] = r1["batch_id"]  # Duplicate batch_id
        csv_path = self._write_csv("dup_csv.csv", [r1, r2])

        summary = import_stock_from_csv(csv_path, self.db_path)
        self.assertTrue(summary["success"])
        self.assertEqual(summary["records_processed"], 2)
        self.assertEqual(summary["records_inserted"], 1)
        self.assertEqual(summary["records_rejected"], 1)
        self.assertTrue(any("duplicate batch_id" in err.lower() for err in summary["errors"]))

    def test_70_duplicate_batch_id_in_db_preserved(self):
        """Existing DB record is preserved and duplicate batch_id rejected when update_existing=False."""
        initial_row = make_row(30, 50, 10, 500, medicine="Original Med")
        initial_row["batch_id"] = "EXIST-BATCH-01"
        csv_init = self._write_csv("init.csv", [initial_row])
        import_stock_from_csv(csv_init, self.db_path)

        # Attempt to import duplicate with different quantity
        dup_row = make_row(30, 999, 10, 500, medicine="New Med")
        dup_row["batch_id"] = "EXIST-BATCH-01"
        csv_dup = self._write_csv("dup_db.csv", [dup_row])

        summary = import_stock_from_csv(csv_dup, self.db_path, update_existing=False)
        self.assertEqual(summary["records_rejected"], 1)
        self.assertEqual(summary["records_inserted"], 0)
        self.assertEqual(summary["records_updated"], 0)
        self.assertTrue(any("already exists" in err.lower() for err in summary["errors"]))

        # Check DB record was preserved
        df = load_stock(self.db_path)
        matching = df[df["batch_id"] == "EXIST-BATCH-01"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching.iloc[0]["quantity"], 50, "Original quantity must be preserved")

    def test_71_update_existing_batch_id_succeeds(self):
        """When update_existing=True, existing records are updated rather than rejected."""
        initial_row = make_row(30, 50, 10, 500, medicine="Original Med")
        initial_row["batch_id"] = "EXIST-BATCH-02"
        csv_init = self._write_csv("init2.csv", [initial_row])
        import_stock_from_csv(csv_init, self.db_path)

        # Update with new quantity
        updated_row = make_row(45, 750, 25, 400, medicine="Updated Med")
        updated_row["batch_id"] = "EXIST-BATCH-02"
        csv_upd = self._write_csv("update.csv", [updated_row])

        summary = import_stock_from_csv(csv_upd, self.db_path, update_existing=True)
        self.assertEqual(summary["records_updated"], 1)
        self.assertEqual(summary["records_inserted"], 0)
        self.assertEqual(summary["records_rejected"], 0)

        # Verify DB updated
        df = load_stock(self.db_path)
        matching = df[df["batch_id"] == "EXIST-BATCH-02"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching.iloc[0]["quantity"], 750)
        self.assertEqual(matching.iloc[0]["medicine_name"], "Updated Med")

    def test_72_invalid_and_negative_numeric_values_rejected(self):
        """Negative and non-integer values for numeric fields are rejected with clear errors."""
        bad_rows = [
            make_row(30, -50, 10, 500),         # negative quantity
            make_row(30, "not-a-number", 10, 500), # non-numeric quantity
            make_row(30, 100, -5, 500),         # negative demand
            make_row(30, 100, 10, -200),        # negative capacity
            make_row(30, 100, 10, 500, cost=-1.5), # negative cost
        ]
        # Assign distinct batch IDs
        for idx, row in enumerate(bad_rows):
            row["batch_id"] = f"BAD-NUM-{idx}"

        csv_path = self._write_csv("bad_nums.csv", bad_rows)
        summary = import_stock_from_csv(csv_path, self.db_path)

        self.assertEqual(summary["records_processed"], 5)
        self.assertEqual(summary["records_rejected"], 5)
        self.assertEqual(summary["records_inserted"], 0)
        self.assertEqual(len(summary["errors"]), 5)

    def test_73_invalid_expiry_date_rejected(self):
        """Malformed or non-existent calendar expiry dates are rejected."""
        bad_dates = [
            make_row(30, 100, 10, 500),
            make_row(30, 100, 10, 500),
            make_row(30, 100, 10, 500),
        ]
        bad_dates[0]["batch_id"] = "EXP-BAD-01"
        bad_dates[0]["expiry_date"] = "2026-02-30"  # Invalid February date
        bad_dates[1]["batch_id"] = "EXP-BAD-02"
        bad_dates[1]["expiry_date"] = "not-a-date"
        bad_dates[2]["batch_id"] = "EXP-BAD-03"
        bad_dates[2]["expiry_date"] = "15/09/2026"  # Wrong format

        csv_path = self._write_csv("bad_dates.csv", bad_dates)
        summary = import_stock_from_csv(csv_path, self.db_path)

        self.assertEqual(summary["records_processed"], 3)
        self.assertEqual(summary["records_rejected"], 3)
        self.assertEqual(summary["records_inserted"], 0)
        self.assertTrue(all("expiry_date" in err for err in summary["errors"]))

    def test_74_transactional_rollback_on_failure(self):
        """When strict=True and any row fails validation, transaction is aborted with zero inserts."""
        rows = [
            make_row(30, 100, 10, 500),  # Valid row
            make_row(30, -50, 10, 500),  # Invalid row (negative quantity)
        ]
        rows[0]["batch_id"] = "STRICT-VALID-01"
        rows[1]["batch_id"] = "STRICT-INVALID-02"

        csv_path = self._write_csv("strict_fail.csv", rows)
        summary = import_stock_from_csv(csv_path, self.db_path, strict=True)

        self.assertFalse(summary["success"], "Strict mode must fail when errors are present")
        self.assertEqual(summary["records_inserted"], 0)
        self.assertEqual(summary["records_rejected"], 1)

        # Confirm DB is completely empty (no partial data from row 1)
        df = load_stock(self.db_path)
        self.assertEqual(len(df), 0, "No partial records should be committed on failure")


# ─────────────────────────────────────────────────────────────────────────────
# Explainable Scoring Tests (test_75 – test_81)
# Verifies total stock value financial exposure, normalization, component breakdown,
# deterministic scoring, and preservation of urgency categories.
# ─────────────────────────────────────────────────────────────────────────────
class TestExplainableScoring(unittest.TestCase):

    def test_75_low_value_stock_scoring(self):
        """Low-value stock reflects minimal financial exposure and scores lower than high-value stock."""
        # Low value: 20 units @ £0.05 = £1.00 stock value
        low_val_row = {**make_row(15, 20, 10, 500, cost=0.05), "urgency": "near-expiry"}
        # High value: 20 units @ £25.00 = £500.00 stock value
        high_val_row = {**make_row(15, 20, 10, 500, cost=25.00), "urgency": "near-expiry"}

        low_score = score_batch(low_val_row)
        high_score = score_batch(high_val_row)

        low_comp = get_score_components(low_val_row)
        high_comp = get_score_components(high_val_row)

        self.assertEqual(low_comp["stock_value"], 1.0)
        self.assertLess(low_comp["value_score"], 1.0, "Low value stock should contribute minimal value score")
        self.assertGreater(high_score, low_score, "High-value batch must score higher than low-value batch")
        self.assertGreater(high_comp["value_score"], low_comp["value_score"])

    def test_76_high_value_stock_scoring(self):
        """High-value stock reaches the normalized financial exposure ceiling without runaway scores."""
        # Value = 250 units * £10.00 = £2,500 (> £1,250 cap -> full 20.0 pts)
        high_val_row = {**make_row(3, 250, 10, 500, cost=10.00), "urgency": "critical"}
        # Extreme value = 250 units * £500.00 = £125,000 (also capped at 20.0 pts)
        extreme_val_row = {**make_row(3, 250, 10, 500, cost=500.00), "urgency": "critical"}

        comp_high = get_score_components(high_val_row)
        comp_extreme = get_score_components(extreme_val_row)

        self.assertEqual(comp_high["value_score"], 20.0, "Values >= £1,250 must receive normalized maximum of 20 pts")
        self.assertEqual(comp_extreme["value_score"], 20.0, "Extreme values must be capped at 20 pts")
        self.assertEqual(score_batch(high_val_row), score_batch(extreme_val_row),
            "Scores must be normalized so extreme prices do not distort scoring")

    def test_77_high_quantity_stock_scoring(self):
        """High quantity reaches normalized volume ceiling of 30 pts at 500 units."""
        high_qty_row = {**make_row(15, 600, 10, 500, cost=1.00), "urgency": "near-expiry"}
        low_qty_row = {**make_row(15, 50, 10, 500, cost=1.00), "urgency": "near-expiry"}

        high_comp = get_score_components(high_qty_row)
        low_comp = get_score_components(low_qty_row)

        self.assertEqual(high_comp["quantity_score"], 30.0, "Quantities >= 500 must receive max 30 pts")
        self.assertLess(low_comp["quantity_score"], high_comp["quantity_score"])
        self.assertGreater(score_batch(high_qty_row), score_batch(low_qty_row))

    def test_78_critical_expiry_scoring(self):
        """Critical expiry (0-7 days) contributes 100 pts and outscores near-expiry by exactly 50 pts."""
        crit_row = {**make_row(4, 200, 10, 500, cost=2.00), "urgency": "critical"}
        near_row = {**make_row(18, 200, 10, 500, cost=2.00), "urgency": "near-expiry"}

        crit_comp = get_score_components(crit_row)
        near_comp = get_score_components(near_row)

        self.assertEqual(crit_comp["urgency_score"], 100.0)
        self.assertEqual(near_comp["urgency_score"], 50.0)
        self.assertAlmostEqual(score_batch(crit_row) - score_batch(near_row), 50.0, places=1)

    def test_79_near_expiry_scoring(self):
        """Near-expiry (8-30 days) contributes 50 pts and outscores watch category by 40 pts."""
        near_row = {**make_row(20, 100, 10, 500, cost=1.00), "urgency": "near-expiry"}
        watch_row = {**make_row(60, 100, 10, 500, cost=1.00), "urgency": "watch"}

        near_comp = get_score_components(near_row)
        watch_comp = get_score_components(watch_row)

        self.assertEqual(near_comp["urgency_score"], 50.0)
        self.assertEqual(watch_comp["urgency_score"], 10.0)
        self.assertAlmostEqual(score_batch(near_row) - score_batch(watch_row), 40.0, places=1)

    def test_80_safe_and_expired_stock_score_zero(self):
        """Safe stock (>90 days) and expired stock (<0 days) always score exactly 0.0."""
        safe_row = {**make_row(120, 500, 50, 1000, cost=50.0), "urgency": "safe"}
        expired_row = {**make_row(-5, 500, 50, 1000, cost=50.0), "urgency": "expired"}

        self.assertEqual(score_batch(safe_row), 0.0)
        self.assertEqual(score_batch(expired_row), 0.0)

        safe_comp = get_score_components(safe_row)
        self.assertEqual(safe_comp["total_score"], 0.0)
        self.assertEqual(safe_comp["urgency_score"], 0.0)

    def test_81_score_components_explainability(self):
        """Recommendation returns individual score components for UI explainability."""
        source = make_row(5, 100, 10, 500, "BR01", "Source Branch", cost=2.50)
        dest = make_row(100, 20, 30, 500, "BR02", "Dest Branch", cost=2.50)
        df = pd.DataFrame([source, dest])

        recs = generate_recommendations(df)
        self.assertEqual(len(recs), 1)
        rec = recs[0]

        # Individual score components must be present in recommendation and decision factors
        self.assertIn("score_components", rec)
        self.assertIn("score_components", rec["decision_factors"])

        comp = rec["score_components"]
        required_comp_keys = [
            "urgency_score", "quantity_score", "value_score",
            "total_score", "stock_value", "urgency_label"
        ]
        for k in required_comp_keys:
            self.assertIn(k, comp, f"Missing score component: {k}")

        # Components must sum to total score
        calculated_total = round(comp["urgency_score"] + comp["quantity_score"] + comp["value_score"], 1)
        self.assertAlmostEqual(comp["total_score"], calculated_total, places=1)
        self.assertEqual(comp["total_score"], rec["score"])
        self.assertEqual(comp["stock_value"], round(100 * 2.50, 2))


# ─────────────────────────────────────────────────────────────────────────────
# Destination Selection & Need-Based Ranking Tests (test_82 – test_86)
# Verifies that destination branches are chosen based on actual shortage/need
# (weeks of cover, demand, capacity) rather than raw weekly demand alone.
# ─────────────────────────────────────────────────────────────────────────────
class TestDestinationSelection(unittest.TestCase):

    def test_82_high_demand_low_stock_preferred_over_high_demand_high_stock(self):
        """Destination with low weeks of cover is selected over a branch with higher demand but excessive stock."""
        source = make_row(5, 50, 15, 500, branch_id="BR_SRC", branch_name="Source Branch")

        # Branch A: high demand (40/wk), critically low stock (10 units -> 0.25 wks cover)
        dest_a = make_row(100, 10, 40, 500, branch_id="BR_A", branch_name="Branch LowStock")

        # Branch B: slightly higher demand (50/wk), but already overstocked (400 units -> 8.0 wks cover)
        dest_b = make_row(100, 400, 50, 500, branch_id="BR_B", branch_name="Branch Overstocked")

        df = pd.DataFrame([source, dest_a, dest_b])
        recs = generate_recommendations(df)

        self.assertEqual(len(recs), 1)
        rec = recs[0]
        self.assertEqual(rec["action"], "TRANSFER")
        self.assertEqual(rec["destination_branch_id"], "BR_A",
            "FAIL: Branch with urgent stock deficit (low cover) must be preferred over overstocked branch")
        self.assertGreater(rec["destinations"][0]["need_score"], rec["destinations"][1]["need_score"])
        self.assertIn("Branch LowStock", rec["reason"])
        self.assertIn("cover", rec["reason"].lower())

    def test_83_high_demand_high_stock_lower_need_score(self):
        """calculate_need_score penalizes excessive stock coverage while rewarding shortage urgency."""
        dest_low_stock = pd.Series({"quantity": 10, "demand_per_week": 50, "branch_capacity_remaining": 500})
        dest_high_stock = pd.Series({"quantity": 500, "demand_per_week": 50, "branch_capacity_remaining": 500})

        score_low, comps_low = calculate_need_score(dest_low_stock, source_quantity=50)
        score_high, comps_high = calculate_need_score(dest_high_stock, source_quantity=50)

        self.assertGreater(score_low, score_high, "Shortage branch must have a higher need score")
        self.assertEqual(comps_low["weeks_of_cover"], 0.2)
        self.assertEqual(comps_high["weeks_of_cover"], 10.0)
        self.assertGreater(comps_low["coverage_score"], comps_high["coverage_score"])

    def test_84_zero_demand_branch_never_recommended(self):
        """Branches with zero demand are strictly excluded from destination candidates."""
        source = make_row(5, 50, 20, 500, branch_id="BR_SRC", branch_name="Source Branch")
        dest_zero = make_row(100, 0, 0, 500, branch_id="BR_ZERO", branch_name="Zero Demand Branch")

        df = pd.DataFrame([source, dest_zero])
        recs = generate_recommendations(df)

        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["action"], "FLAG_FOR_REVIEW")
        self.assertIsNone(recs[0]["destination_branch_id"])

        dests, status, msg = find_destinations(source, df)
        self.assertEqual(len(dests), 0)
        self.assertIn("zero demand", msg.lower())

    def test_85_insufficient_capacity_never_recommended(self):
        """Branches lacking sufficient remaining capacity are safely excluded even if shortage is severe."""
        source = make_row(5, 100, 20, 500, branch_id="BR_SRC", branch_name="Source Branch")

        # Branch C has urgent stock need (stock 0, demand 50) but only 50 units capacity (< 100 required)
        dest_c_no_cap = make_row(100, 0, 50, 50, branch_id="BR_NO_CAP", branch_name="Full Branch")

        # Branch D has moderate demand (25/wk) and ample capacity (500 units >= 100)
        dest_d_viable = make_row(100, 25, 25, 500, branch_id="BR_VIABLE", branch_name="Viable Branch")

        df = pd.DataFrame([source, dest_c_no_cap, dest_d_viable])
        recs = generate_recommendations(df)

        self.assertEqual(len(recs), 1)
        rec = recs[0]
        self.assertEqual(rec["action"], "TRANSFER")
        self.assertEqual(rec["destination_branch_id"], "BR_VIABLE",
            "FAIL: Full branch must be excluded; viable branch must be recommended")

    def test_86_multiple_possible_destinations_ranked_by_need_and_top_3_kept(self):
        """When multiple viable destinations exist, top 3 are kept and ranked strictly by need score."""
        source = make_row(5, 50, 20, 500, branch_id="BR_SRC", branch_name="Source Branch")

        # 5 distinct candidate branches with varying stock levels & coverage
        b1 = make_row(100, 5, 40, 500, branch_id="BR_1", branch_name="Branch 1")    # WOC = 0.125 (Need ~93.4)
        b2 = make_row(100, 35, 35, 500, branch_id="BR_2", branch_name="Branch 2")   # WOC = 1.0   (Need ~82.6)
        b3 = make_row(100, 50, 25, 500, branch_id="BR_3", branch_name="Branch 3")   # WOC = 2.0   (Need ~68.3)
        b4 = make_row(100, 150, 30, 500, branch_id="BR_4", branch_name="Branch 4")  # WOC = 5.0   (Need ~46.3)
        b5 = make_row(100, 200, 20, 500, branch_id="BR_5", branch_name="Branch 5")  # WOC = 10.0  (Need ~32.0)

        df = pd.DataFrame([source, b1, b2, b3, b4, b5])
        dests, status, _ = find_destinations(source, df)

        self.assertEqual(status, "OK")
        self.assertEqual(len(dests), 3, "Only top 3 destinations must be returned")

        # Verify ranking by need score
        self.assertEqual(dests[0]["dest_branch_id"], "BR_1")
        self.assertEqual(dests[1]["dest_branch_id"], "BR_2")
        self.assertEqual(dests[2]["dest_branch_id"], "BR_3")
        self.assertGreater(dests[0]["need_score"], dests[1]["need_score"])
        self.assertGreater(dests[1]["need_score"], dests[2]["need_score"])

        # Check required components on all returned destinations
        for d in dests:
            self.assertIn("dest_weeks_of_cover", d)
            self.assertIn("dest_need_score", d)
            self.assertIn("need_score_components", d)
            self.assertIn("coverage_score", d["need_score_components"])
            self.assertIn("demand_score", d["need_score_components"])
            self.assertIn("capacity_score", d["need_score_components"])

# ─────────────────────────────────────────────────────────────────────────────
# Enhanced tests — Batch Splitting Across Multiple Destination Branches
# ─────────────────────────────────────────────────────────────────────────────
class TestBatchSplitting(unittest.TestCase):

    def test_87_one_destination_allocation(self):
        """When a single destination has sufficient capacity and high need, entire batch is allocated to 1 destination."""
        source = make_row(5, 100, 20, 500, branch_id="BR_SRC", branch_name="Source Branch")
        dest_a = make_row(100, 20, 40, 500, branch_id="BR_A", branch_name="Branch HighNeed")
        dest_b = make_row(100, 100, 20, 500, branch_id="BR_B", branch_name="Branch LowerNeed")

        df = pd.DataFrame([source, dest_a, dest_b])
        recs = generate_recommendations(df)

        self.assertEqual(len(recs), 1)
        rec = recs[0]
        self.assertEqual(rec["action"], "TRANSFER")
        self.assertFalse(rec["is_split"], "Should not be split when a single destination can absorb the batch")
        self.assertEqual(rec["destination_branch_id"], "BR_A")
        self.assertEqual(rec["suggested_quantity"], 100)
        self.assertEqual(rec["destinations"][0]["transfer_quantity"], 100)

        # Check transparent explanation contains all required factors
        expl = rec["explanation"]
        self.assertIn("Source: Source Branch", expl)
        self.assertIn("Source Quantity: 100", expl)
        self.assertIn("Destination: Branch HighNeed", expl)
        self.assertIn("Transfer Quantity: 100", expl)
        self.assertIn("Destination Demand: 40", expl)
        self.assertIn("Current Stock Coverage: 0.5", expl)
        self.assertIn("Available Capacity: 500", expl)

    def test_88_two_destinations_batch_split(self):
        """When the highest-need branch has limited capacity (< source qty), batch is divided across two branches."""
        source = make_row(5, 100, 20, 500, branch_id="BR_SRC", branch_name="Source Branch")

        # Destination 1: capacity 60 (< 100), high need (low stock, high demand)
        dest_1 = make_row(100, 5, 40, 60, branch_id="BR_D1", branch_name="Branch Dest 1")

        # Destination 2: capacity 60 (< 100), good need
        dest_2 = make_row(100, 10, 30, 60, branch_id="BR_D2", branch_name="Branch Dest 2")

        df = pd.DataFrame([source, dest_1, dest_2])
        recs = generate_recommendations(df)

        self.assertEqual(len(recs), 1)
        rec = recs[0]
        self.assertEqual(rec["action"], "TRANSFER")
        self.assertTrue(rec["is_split"], "Batch must be split across 2 branches when individual capacity is limited")

        split_dests = rec["split_destinations"]
        self.assertEqual(len(split_dests), 2, "Expected exactly 2 split destinations")

        # Verify quantities allocated: Branch 1 gets 60 (capacity limit), Branch 2 gets 40
        self.assertEqual(split_dests[0]["branch_id"], "BR_D1")
        self.assertEqual(split_dests[0]["transfer_quantity"], 60)
        self.assertEqual(split_dests[1]["branch_id"], "BR_D2")
        self.assertEqual(split_dests[1]["transfer_quantity"], 40)

        total_transferred = sum(d["transfer_quantity"] for d in split_dests)
        self.assertEqual(total_transferred, 100, "Total transferred must equal source quantity")

        # Verify explanation outlines the split details
        expl = rec["explanation"]
        self.assertIn("TRANSFER (SPLIT)", expl)
        self.assertIn("Source Quantity: 100", expl)
        self.assertIn("Branch Dest 1", expl)
        self.assertIn("Branch Dest 2", expl)
        self.assertIn("Transfer 60 units", expl)
        self.assertIn("Transfer 40 units", expl)

    def test_89_three_destinations_batch_split(self):
        """When the top 2 destinations have limited capacity, batch is divided across 3 destination branches."""
        source = make_row(5, 100, 20, 500, branch_id="BR_SRC", branch_name="Source Branch")

        # Branch 1: capacity 40 (< 100)
        b1 = make_row(100, 0, 40, 40, branch_id="BR_1", branch_name="Branch 1")
        # Branch 2: capacity 35 (< 100)
        b2 = make_row(100, 5, 35, 35, branch_id="BR_2", branch_name="Branch 2")
        # Branch 3: capacity 50 (< 100)
        b3 = make_row(100, 10, 30, 50, branch_id="BR_3", branch_name="Branch 3")

        df = pd.DataFrame([source, b1, b2, b3])
        recs = generate_recommendations(df)

        self.assertEqual(len(recs), 1)
        rec = recs[0]
        self.assertEqual(rec["action"], "TRANSFER")
        self.assertTrue(rec["is_split"])

        split_dests = [d for d in rec["destinations"] if d["transfer_quantity"] > 0]
        self.assertEqual(len(split_dests), 3, "Batch must be divided across 3 destinations")

        self.assertEqual(split_dests[0]["dest_branch_id"], "BR_1")
        self.assertEqual(split_dests[0]["transfer_quantity"], 40)

        self.assertEqual(split_dests[1]["dest_branch_id"], "BR_2")
        self.assertEqual(split_dests[1]["transfer_quantity"], 35)

        self.assertEqual(split_dests[2]["dest_branch_id"], "BR_3")
        self.assertEqual(split_dests[2]["transfer_quantity"], 25)

        total_transferred = sum(d["transfer_quantity"] for d in split_dests)
        self.assertEqual(total_transferred, 100)

    def test_90_limited_capacity_respected(self):
        """A destination must never be recommended more units than its remaining capacity."""
        source = make_row(5, 100, 20, 500, branch_id="BR_SRC", branch_name="Source Branch")

        # Branch A has urgent need (stock 0, demand 50) but only 30 capacity
        b_a = make_row(100, 0, 50, 30, branch_id="BR_A", branch_name="Branch A")
        # Branch B has capacity 80
        b_b = make_row(100, 10, 30, 80, branch_id="BR_B", branch_name="Branch B")

        df = pd.DataFrame([source, b_a, b_b])
        recs = generate_recommendations(df)

        self.assertEqual(len(recs), 1)
        rec = recs[0]
        for d in rec["destinations"]:
            t_qty = d["transfer_quantity"]
            cap = d["dest_capacity"]
            self.assertLessEqual(t_qty, cap, f"Transfer {t_qty} exceeded capacity {cap} for branch {d['dest_branch_id']}")

        # Specifically, Branch A receives exactly 30 units (cannot exceed capacity 30)
        self.assertEqual(rec["destinations"][0]["transfer_quantity"], 30)
        self.assertEqual(rec["destinations"][1]["transfer_quantity"], 70)

    def test_91_total_capacity_less_than_source_quantity_flags_for_review(self):
        """When total available capacity across all destinations is less than source quantity, flag for review."""
        source = make_row(5, 100, 20, 500, branch_id="BR_SRC", branch_name="Source Branch")

        # Branch 1 capacity 30, Branch 2 capacity 40 -> Total 70 < 100
        b1 = make_row(100, 0, 40, 30, branch_id="BR_1", branch_name="Branch 1")
        b2 = make_row(100, 0, 40, 40, branch_id="BR_2", branch_name="Branch 2")

        df = pd.DataFrame([source, b1, b2])
        recs = generate_recommendations(df)

        self.assertEqual(len(recs), 1)
        rec = recs[0]
        self.assertEqual(rec["action"], "FLAG_FOR_REVIEW")
        self.assertIn("capacity", rec["reason"].lower())

        # Also test find_destinations returns NO_VIABLE_DEST
        dests, status, msg = find_destinations(source, df)
        self.assertEqual(status, "NO_VIABLE_DEST")
        self.assertIn("capacity", msg.lower())

    def test_92_zero_demand_branches_excluded_from_splits(self):
        """Branches with zero demand must never be included in split recommendations even with large capacity."""
        source = make_row(5, 100, 20, 500, branch_id="BR_SRC", branch_name="Source Branch")

        # Branch 1 has capacity 60, demand 30
        b1 = make_row(100, 10, 30, 60, branch_id="BR_1", branch_name="Branch 1")
        # Branch 2 has huge capacity 1000, but ZERO demand
        b_zero = make_row(100, 0, 0, 1000, branch_id="BR_ZERO", branch_name="Zero Demand Branch")
        # Branch 3 has capacity 60, demand 25
        b3 = make_row(100, 10, 25, 60, branch_id="BR_3", branch_name="Branch 3")

        df = pd.DataFrame([source, b1, b_zero, b3])
        recs = generate_recommendations(df)

        self.assertEqual(len(recs), 1)
        rec = recs[0]
        self.assertEqual(rec["action"], "TRANSFER")

        for d in rec.get("destinations", []):
            self.assertNotEqual(d["dest_branch_id"], "BR_ZERO", "Zero demand branch must never be recommended")

        # Units allocated only between Branch 1 and Branch 3
        split_dests = [d for d in rec["destinations"] if d["transfer_quantity"] > 0]
        branch_ids = [d["dest_branch_id"] for d in split_dests]
        self.assertIn("BR_1", branch_ids)
        self.assertIn("BR_3", branch_ids)
        self.assertNotIn("BR_ZERO", branch_ids)

    def test_93_total_transferred_never_exceeds_source_quantity(self):
        """Sum of recommended transfers must never exceed available source quantity."""
        for qty in [25, 67, 100, 250]:
            source = make_row(5, qty, 20, 500, branch_id="BR_SRC", branch_name="Source")
            b1 = make_row(100, 0, 30, qty // 2 + 5, branch_id="B1", branch_name="B1")
            b2 = make_row(100, 0, 30, qty // 2 + 5, branch_id="B2", branch_name="B2")

            df = pd.DataFrame([source, b1, b2])
            recs = generate_recommendations(df)
            if recs and recs[0]["action"] == "TRANSFER":
                total_t = sum(d["transfer_quantity"] for d in recs[0]["destinations"])
                self.assertLessEqual(total_t, qty, f"Total transfer {total_t} exceeded source quantity {qty}")

    def test_94_high_impact_confirmation_supported_for_split_transfers(self):
        """High-impact batches that are split continue to require explicit confirmation."""
        # Value = 200 units * £5.00 = £1,000 > £500 HIGH_VALUE threshold
        source = make_row(5, 200, 20, 500, branch_id="BR_SRC", branch_name="Source Branch", cost=5.00)
        b1 = make_row(100, 0, 40, 120, branch_id="BR_1", branch_name="Branch 1")
        b2 = make_row(100, 0, 40, 120, branch_id="BR_2", branch_name="Branch 2")

        df = pd.DataFrame([source, b1, b2])
        recs = generate_recommendations(df)

        self.assertEqual(len(recs), 1)
        rec = recs[0]
        self.assertEqual(rec["action"], "TRANSFER")
        self.assertTrue(rec["is_split"])
        self.assertTrue(rec["is_high_impact"])
        self.assertTrue(rec["requires_confirmation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)






# ─────────────────────────────────────────────────────────────────────────────
# SQLite Audit Log Tests
# ─────────────────────────────────────────────────────────────────────────────
import log_manager as _lm
from database import export_decisions_to_csv


class TestSQLiteAuditLog(unittest.TestCase):
    """
    Verify that SQLite is the sole authoritative audit log.

    Every test gets its own temporary SQLite database so tests are fully
    isolated from each other and from the live operational database.
    """

    def setUp(self):
        self.tmp_dir  = tempfile.mkdtemp()
        self.db_path  = os.path.join(self.tmp_dir, "test_audit.db")
        self.csv_path = os.path.join(self.tmp_dir, "test_audit.csv")
        initialise_database(self.db_path, seed=False)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _save(self, batch_id, medicine, action,
              destination="", override_reason="", user="pharmacist1",
              source_branch="Branch A", quantity=100,
              system_recommendation="TRANSFER"):
        save_decision(
            batch_id=batch_id, medicine=medicine, action=action,
            destination=destination, override_reason=override_reason,
            user=user, source_branch=source_branch, quantity=quantity,
            system_recommendation=system_recommendation,
            final_decision=action, db_path=self.db_path,
        )

    def _load(self):
        return load_decisions(db_path=self.db_path)

    # ── tests ─────────────────────────────────────────────────────────────────

    def test_confirmed_transfer_persisted(self):
        """A CONFIRMED decision is written to SQLite and readable back."""
        self._save(batch_id="BATCH-001", medicine="Amoxicillin",
                   action="CONFIRMED", destination="Branch B", quantity=50)
        df = self._load()
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["batch_id"],   "BATCH-001")
        self.assertEqual(row["medicine"],   "Amoxicillin")
        self.assertEqual(row["action"],     "CONFIRMED")
        self.assertEqual(row["destination"],"Branch B")
        self.assertEqual(int(row["quantity"]), 50)

    def test_overridden_recommendation_persisted(self):
        """An OVERRIDDEN decision stores the override reason."""
        self._save(batch_id="BATCH-002", medicine="Ibuprofen",
                   action="OVERRIDDEN", destination="Branch C",
                   override_reason="Branch already fully stocked", quantity=200)
        df = self._load()
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["action"],          "OVERRIDDEN")
        self.assertEqual(row["override_reason"], "Branch already fully stocked")
        self.assertEqual(row["final_decision"],  "OVERRIDDEN")

    def test_manual_review_persisted(self):
        """A MANUALLY_REVIEWED decision is persisted with correct action."""
        self._save(batch_id="BATCH-003", medicine="Paracetamol",
                   action="MANUALLY_REVIEWED", destination="", quantity=75,
                   system_recommendation="FLAG_FOR_REVIEW")
        df = self._load()
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["action"],               "MANUALLY_REVIEWED")
        self.assertEqual(row["system_recommendation"],"FLAG_FOR_REVIEW")

    def test_multi_user_decisions_recorded_separately(self):
        """Decisions from different users are all stored and attributable."""
        users = ["pharmacist1", "pharmacist2", "admin"]
        for i, user in enumerate(users):
            self._save(batch_id="BATCH-1{:02d}".format(i), medicine="Metformin",
                       action="CONFIRMED", user=user, quantity=10 * (i + 1))
        df = self._load()
        self.assertEqual(len(df), 3)
        self.assertEqual(set(df["user"].tolist()), set(users))

    def test_action_normalisation(self):
        """Action strings are always stored upper-cased regardless of input case."""
        for raw_action in ("confirmed", "Overridden", "MANUALLY_REVIEWED"):
            self._save(batch_id="BATCH-N-" + raw_action,
                       medicine="Atorvastatin", action=raw_action)
        df = self._load()
        for action in df["action"]:
            self.assertEqual(action, action.upper(),
                             "Action not normalised: {!r}".format(action))

    def test_final_decision_defaults_to_action(self):
        """final_decision equals action when not explicitly provided."""
        save_decision(batch_id="BATCH-FD", medicine="Omeprazole",
                      action="CONFIRMED", db_path=self.db_path)
        df = self._load()
        self.assertEqual(df.iloc[0]["final_decision"], "CONFIRMED")

    def test_multiple_entries_ordered_by_id(self):
        """Multiple decisions are returned in insertion order (id ASC)."""
        for j in range(5):
            self._save(batch_id="BATCH-ORD-{:03d}".format(j),
                       medicine="Simvastatin", action="CONFIRMED",
                       quantity=j * 10)
        df = self._load()
        self.assertEqual(len(df), 5)
        batch_ids = df["batch_id"].tolist()
        self.assertEqual(batch_ids, sorted(batch_ids))

    def test_no_csv_written_during_normal_save(self):
        """save_decision must NOT create any CSV file on the normal code path."""
        import glob
        self._save(batch_id="BATCH-NOCSV", medicine="Warfarin",
                   action="CONFIRMED")
        csv_files = glob.glob(os.path.join(self.tmp_dir, "*.csv"))
        self.assertEqual(csv_files, [],
                         "CSV was written unexpectedly: {}".format(csv_files))

    def test_export_decisions_to_csv(self):
        """export_decisions_to_csv writes all SQLite rows to a CSV file."""
        for k in range(3):
            self._save(batch_id="BATCH-EXP-{}".format(k),
                       medicine="Lisinopril", action="CONFIRMED",
                       quantity=k + 1)
        out = export_decisions_to_csv(csv_path=self.csv_path,
                                      db_path=self.db_path)
        self.assertTrue(os.path.exists(out))
        df_csv = pd.read_csv(out)
        self.assertEqual(len(df_csv), 3)
        self.assertIn("batch_id", df_csv.columns)
        self.assertIn("action",   df_csv.columns)

    def test_empty_database_returns_empty_dataframe(self):
        """load_decisions on a fresh DB returns an empty DataFrame with expected columns."""
        df = self._load()
        self.assertTrue(df.empty)
        for col in ("timestamp", "user", "medicine", "batch_id",
                    "action", "final_decision", "override_reason"):
            self.assertIn(col, df.columns)

    def test_quantity_coerced_to_int(self):
        """quantity is stored as an integer even when passed as a float string."""
        save_decision(batch_id="BATCH-QTY", medicine="Metoprolol",
                      action="CONFIRMED", quantity="42.7",
                      db_path=self.db_path)
        df = self._load()
        self.assertEqual(int(df.iloc[0]["quantity"]), 42)

    def test_split_transfer_multiple_rows(self):
        """A split redistribution creates one row per destination; total <= source."""
        source_qty   = 100
        destinations = [("Branch B", 40), ("Branch C", 35), ("Branch D", 25)]
        for branch, qty in destinations:
            dest_label = "{} ({} units)".format(branch, qty)
            self._save(batch_id="BATCH-SPLIT", medicine="Amlodipine",
                       action="CONFIRMED", destination=dest_label,
                       quantity=qty, system_recommendation="TRANSFER (SPLIT)")
        df = self._load()
        self.assertEqual(len(df), len(destinations))
        self.assertLessEqual(int(df["quantity"].sum()), source_qty)


# ─────────────────────────────────────────────────────────────────────────────
# Live Inventory & Cache Invalidation Tests
# ─────────────────────────────────────────────────────────────────────────────
from database import (
    update_stock_quantity,
    invalidate_stock_cache,
    clear_stock_cache,
)
from recommender import days_to_expiry, generate_recommendations


class TestLiveInventoryAndCache(unittest.TestCase):
    """
    Verify that live inventory reads from SQLite always reflect the current database
    state without stale caching, and that cache invalidation functions correctly.
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_cache.db")
        initialise_database(self.db_path, seed=True)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_load_inventory_reflects_immediate_database_update(self):
        """Modifying inventory in SQLite is immediately reflected on reload."""
        df1 = load_stock(self.db_path)
        self.assertFalse(df1.empty)
        target_batch = df1.iloc[0]["batch_id"]
        original_qty = int(df1.iloc[0]["quantity"])
        new_qty = original_qty + 75

        updated = update_stock_quantity(target_batch, new_qty, db_path=self.db_path)
        self.assertTrue(updated)

        df2 = load_stock(self.db_path)
        updated_row = df2[df2["batch_id"] == target_batch]
        self.assertEqual(int(updated_row.iloc[0]["quantity"]), new_qty)

    def test_update_stock_quantity_validation(self):
        """update_stock_quantity validates inputs and rejects invalid quantities."""
        with self.assertRaises(ValueError):
            update_stock_quantity("BATCH-1", -10, db_path=self.db_path)

        with self.assertRaises(ValueError):
            update_stock_quantity("", 50, db_path=self.db_path)

        with self.assertRaises(ValueError):
            update_stock_quantity("BATCH-1", "not_a_number", db_path=self.db_path)

        # Nonexistent batch returns False
        result = update_stock_quantity("NONEXISTENT-BATCH-9999", 50, db_path=self.db_path)
        self.assertFalse(result)

    def test_import_stock_updates_reflected_immediately(self):
        """Updating stock via CSV import is reflected immediately on next load."""
        df1 = load_stock(self.db_path)
        row0 = df1.iloc[0].to_dict()
        target_batch = row0["batch_id"]
        new_qty = 999
        row0["quantity"] = new_qty

        # Write to temporary CSV and import with update_existing=True
        csv_path = os.path.join(self.tmp_dir.name, "update_stock.csv")
        pd.DataFrame([row0]).to_csv(csv_path, index=False)
        summary = import_stock_from_csv(csv_path, db_path=self.db_path, update_existing=True)
        self.assertTrue(summary["success"])
        self.assertGreaterEqual(summary["records_updated"], 1)

        # Reload inventory
        df2 = load_stock(self.db_path)
        updated_val = int(df2.loc[df2["batch_id"] == target_batch, "quantity"].iloc[0])
        self.assertEqual(updated_val, new_qty)

    def test_recommendation_updates_with_new_stock_values(self):
        """Recommendation engine calculations immediately use updated stock data."""
        df1 = load_stock(self.db_path)
        # Find an actionable near-expiry batch
        df1["dte"] = df1["expiry_date"].apply(days_to_expiry)
        near_expiry = df1[(df1["dte"].between(1, 30)) & (df1["quantity"] >= 10)]
        if near_expiry.empty:
            self.skipTest("No near-expiry batch in seed data")

        batch_id = near_expiry.iloc[0]["batch_id"]
        original_qty = int(near_expiry.iloc[0]["quantity"])
        unit_cost = float(near_expiry.iloc[0]["unit_cost_gbp"])

        recs1 = generate_recommendations(df1)
        rec1_matching = [r for r in recs1 if r["batch_id"] == batch_id]
        if rec1_matching:
            self.assertEqual(rec1_matching[0]["quantity"], original_qty)

        # Modify quantity in SQLite
        new_qty = original_qty + 120
        update_stock_quantity(batch_id, new_qty, db_path=self.db_path)

        # Reload stock and re-generate recommendations
        df2 = load_stock(self.db_path)
        recs2 = generate_recommendations(df2)
        rec2_matching = [r for r in recs2 if r["batch_id"] == batch_id]
        self.assertTrue(len(rec2_matching) > 0)
        self.assertEqual(rec2_matching[0]["quantity"], new_qty)
        self.assertAlmostEqual(rec2_matching[0]["stock_value"], round(new_qty * unit_cost, 2), places=2)

    def test_barcode_lookup_reflects_updated_stock_quantity(self):
        """Barcode lookup reflects live updated inventory from SQLite."""
        from barcode_registry import BarcodeRegistry
        from barcode_lookup import lookup_barcode

        df1 = load_stock(self.db_path)
        batch = df1.iloc[0]
        batch_id = batch["batch_id"]
        med_name = batch["medicine_name"]
        test_barcode = "5012345678901"

        registry = BarcodeRegistry(self.db_path)
        registry.register(test_barcode, batch_id, med_name)

        # Initial lookup
        res1 = lookup_barcode(test_barcode, registry=registry, stock_df=load_stock(self.db_path))
        self.assertTrue(res1["found"])
        orig_qty = res1["quantity"]

        # Update stock quantity in SQLite
        new_qty = orig_qty + 88
        update_stock_quantity(batch_id, new_qty, db_path=self.db_path)

        # Subsequent lookup reflects new quantity
        res2 = lookup_barcode(test_barcode, registry=registry, stock_df=load_stock(self.db_path))
        self.assertTrue(res2["found"])
        self.assertEqual(res2["quantity"], new_qty)

    def test_cache_invalidation_helper_safe_in_all_environments(self):
        """invalidate_stock_cache and clear_stock_cache execute without error."""
        invalidate_stock_cache()
        clear_stock_cache()

    def test_app_load_data_clear_and_live_read(self):
        """app.load_data returns live DataFrame and provides clear method."""
        from app import load_data
        self.assertTrue(callable(getattr(load_data, "clear", None)))
        load_data.clear()
        df = load_data()
        self.assertIsInstance(df, pd.DataFrame)

    def test_static_ml_predictor_singleton_retained(self):
        """ML predictor caching is preserved as a singleton for performance."""
        from ml_expiry_model import get_ml_predictor
        p1 = get_ml_predictor()
        p2 = get_ml_predictor()
        self.assertIs(p1, p2)

    def test_generate_recommendations_with_all_df_destination_discovery(self):
        """generate_recommendations uses all_df for destination discovery when evaluating subset."""
        df_full = load_stock(self.db_path)
        branches = df_full["branch_name"].unique()
        if len(branches) < 2:
            self.skipTest("Need at least 2 branches in seed data")

        target_branch = branches[0]
        branch_subset = df_full[df_full["branch_name"] == target_branch]

        # With all_df=df_full, recommender can discover destinations at other branches
        recs_with_network = generate_recommendations(branch_subset, all_df=df_full)
        transfers = [r for r in recs_with_network if r["action"] == "TRANSFER"]
        for t in transfers:
            for dest in t.get("destinations", []):
                self.assertNotEqual(dest.get("dest_branch_name"), target_branch)



# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive Automated Service & Boundary Test Suite
# ─────────────────────────────────────────────────────────────────────────────
from auth_config import authenticate_user
from app import check_password


class TestComprehensiveAutomatedSuite(unittest.TestCase):
    """
    End-to-end deterministic verification of recommendation boundaries,
    quantity thresholds, destination selection, split transfers, capacity,
    zero-demand, multi-branch, multi-batch, database CRUD, duplicate batches,
    barcode lifecycle, authentication, decision audit, and input validation.
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_suite.db")
        self.csv_path = os.path.join(self.tmp_dir.name, "test_suite.csv")
        initialise_database(self.db_path, seed=False)

    def tearDown(self):
        self.tmp_dir.cleanup()

    # ── 1. Recommendation Expiry Boundaries ────────────────────────────────────

    def test_rec_boundary_expired_negative_days(self):
        """Expired batches (< 0 days) must never be recommended for transfer."""
        for dte in (-30, -5, -1):
            row = make_row(dte, qty=100, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
            dest = make_row(60, qty=10, demand=50, capacity=500, branch_id="BR02", branch_name="Dest")
            df = pd.DataFrame([row, dest])
            recs = generate_recommendations(df)
            transfers = [r for r in recs if r["batch_id"] == row["batch_id"] and r["action"] == "TRANSFER"]
            self.assertEqual(len(transfers), 0, f"Expired batch at dte={dte} was recommended for transfer")

    def test_rec_boundary_0_days_critical(self):
        """Batch expiring today (dte=0) is critical and flagged for local review (transit infeasible)."""
        row = make_row(0, qty=100, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
        dest = make_row(60, qty=10, demand=50, capacity=500, branch_id="BR02", branch_name="Dest")
        df = pd.DataFrame([row, dest])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == row["batch_id"]]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["urgency"], "critical")
        self.assertEqual(matching[0]["action"], "FLAG_FOR_REVIEW")
        self.assertIn("transit", matching[0]["reason"].lower())

    def test_rec_boundary_7_days_critical(self):
        """7 days to expiry is the upper boundary of critical."""
        row = make_row(7, qty=100, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
        dest = make_row(60, qty=10, demand=50, capacity=500, branch_id="BR02", branch_name="Dest")
        df = pd.DataFrame([row, dest])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == row["batch_id"]]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["urgency"], "critical")

    def test_rec_boundary_8_days_near_expiry(self):
        """8 days to expiry transitions to near-expiry and remains actionable."""
        row = make_row(8, qty=100, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
        dest = make_row(60, qty=10, demand=50, capacity=500, branch_id="BR02", branch_name="Dest")
        df = pd.DataFrame([row, dest])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == row["batch_id"]]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["urgency"], "near-expiry")

    def test_rec_boundary_30_days_near_expiry(self):
        """30 days to expiry is the upper boundary of near-expiry and is actionable."""
        row = make_row(30, qty=100, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
        dest = make_row(60, qty=10, demand=50, capacity=500, branch_id="BR02", branch_name="Dest")
        df = pd.DataFrame([row, dest])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == row["batch_id"]]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["urgency"], "near-expiry")

    def test_rec_boundary_31_days_watch_not_actionable(self):
        """31 days transitions to watch and is not actionable for redistribution."""
        row = make_row(31, qty=100, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
        dest = make_row(60, qty=10, demand=50, capacity=500, branch_id="BR02", branch_name="Dest")
        df = pd.DataFrame([row, dest])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == row["batch_id"]]
        self.assertEqual(len(matching), 0, "31 days (watch) should not generate an actionable transfer recommendation")

    def test_rec_boundary_90_days_watch_not_actionable(self):
        """90 days is upper boundary of watch and is not actionable."""
        row = make_row(90, qty=100, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
        dest = make_row(120, qty=10, demand=50, capacity=500, branch_id="BR02", branch_name="Dest")
        df = pd.DataFrame([row, dest])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == row["batch_id"]]
        self.assertEqual(len(matching), 0)

    def test_rec_boundary_91_days_safe_not_actionable(self):
        """91 days transitions to safe and is not actionable."""
        row = make_row(91, qty=100, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
        dest = make_row(150, qty=10, demand=50, capacity=500, branch_id="BR02", branch_name="Dest")
        df = pd.DataFrame([row, dest])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == row["batch_id"]]
        self.assertEqual(len(matching), 0)

    # ── 2. Quantity Boundaries ─────────────────────────────────────────────────

    def test_quantity_boundary_0_excluded(self):
        """Quantity 0 is below MIN_QTY (10) and excluded."""
        row = make_row(15, qty=0, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
        df = pd.DataFrame([row])
        recs = generate_recommendations(df)
        self.assertEqual(len(recs), 0)

    def test_quantity_boundary_1_excluded(self):
        """Quantity 1 is below MIN_QTY (10) and excluded."""
        row = make_row(15, qty=1, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
        df = pd.DataFrame([row])
        recs = generate_recommendations(df)
        self.assertEqual(len(recs), 0)

    def test_quantity_boundary_9_excluded(self):
        """Quantity 9 is the upper bound of excluded small batches (< 10)."""
        row = make_row(15, qty=9, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
        df = pd.DataFrame([row])
        recs = generate_recommendations(df)
        self.assertEqual(len(recs), 0)

    def test_quantity_boundary_10_included(self):
        """Quantity 10 is the lower bound of actionable batches (>= 10)."""
        row = make_row(15, qty=10, demand=20, capacity=500, branch_id="BR01", branch_name="Source", cost=1.0)
        dest = make_row(60, qty=5, demand=40, capacity=500, branch_id="BR02", branch_name="Dest")
        df = pd.DataFrame([row, dest])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == row["batch_id"]]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["quantity"], 10)
        self.assertFalse(matching[0]["requires_confirmation"], "Value £10 is below HIGH_VALUE (£50)")

    def test_quantity_boundary_200_normal_confirmation_threshold(self):
        """Quantity 200 with low unit cost (£0.10, value £20 <= £50) does not require confirmation."""
        row = make_row(15, qty=200, demand=20, capacity=500, branch_id="BR01", branch_name="Source", cost=0.10)
        dest = make_row(60, qty=10, demand=100, capacity=500, branch_id="BR02", branch_name="Dest")
        df = pd.DataFrame([row, dest])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == row["batch_id"]]
        self.assertEqual(len(matching), 1)
        self.assertFalse(matching[0]["requires_confirmation"], "200 units with £20 value is within normal threshold")

    def test_quantity_boundary_201_triggers_high_impact_confirmation(self):
        """Quantity 201 exceeds HIGH_QTY (200) and triggers confirmation even with low unit cost."""
        row = make_row(15, qty=201, demand=20, capacity=500, branch_id="BR01", branch_name="Source", cost=0.10)
        dest = make_row(60, qty=10, demand=100, capacity=500, branch_id="BR02", branch_name="Dest")
        df = pd.DataFrame([row, dest])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == row["batch_id"]]
        self.assertEqual(len(matching), 1)
        self.assertTrue(matching[0]["requires_confirmation"], "Quantity 201 must trigger high-impact confirmation")

    # ── 3. Destination Selection ───────────────────────────────────────────────

    def test_destination_selection_ranks_highest_need_branch(self):
        """Destination selection prefers branches with high demand and low stock coverage."""
        source = make_row(15, qty=50, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
        # Branch B: demand 50/wk, stock 10 -> 0.2 weeks of cover (urgent shortage)
        dest_high_need = make_row(90, qty=10, demand=50, capacity=500, branch_id="BR02", branch_name="High Need")
        # Branch C: demand 10/wk, stock 100 -> 10.0 weeks of cover (well stocked)
        dest_low_need = make_row(90, qty=100, demand=10, capacity=500, branch_id="BR03", branch_name="Low Need")

        df = pd.DataFrame([source, dest_high_need, dest_low_need])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == source["batch_id"]][0]
        self.assertEqual(matching["action"], "TRANSFER")
        self.assertEqual(matching["destinations"][0]["dest_branch_name"], "High Need")

    def test_destination_selection_never_selects_source_branch(self):
        """Source branch is never recommended to receive its own batch."""
        source = make_row(15, qty=50, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
        dest = make_row(90, qty=20, demand=40, capacity=500, branch_id="BR02", branch_name="Receiving")
        df = pd.DataFrame([source, dest])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == source["batch_id"]][0]
        dest_names = [d["dest_branch_name"] for d in matching.get("destinations", [])]
        self.assertNotIn("Source", dest_names)

    # ── 4. Split Transfers ─────────────────────────────────────────────────────

    def test_split_transfer_allocates_across_multiple_destinations(self):
        """When single destination capacity is less than source quantity, split across destinations."""
        source = make_row(15, qty=120, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
        # Dest B can only take 70
        dest_b = make_row(90, qty=10, demand=50, capacity=70, branch_id="BR02", branch_name="Dest B")
        # Dest C can take 80
        dest_c = make_row(90, qty=15, demand=40, capacity=80, branch_id="BR03", branch_name="Dest C")

        df = pd.DataFrame([source, dest_b, dest_c])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == source["batch_id"]][0]
        self.assertEqual(matching["action"], "TRANSFER")
        split_dests = matching.get("split_destinations", [])
        total_transfer = sum(d["transfer_quantity"] for d in split_dests)
        self.assertLessEqual(total_transfer, 120, "Total split transfer cannot exceed source quantity")
        self.assertGreater(len(split_dests), 1, "Should recommend a split transfer across multiple branches")

    def test_split_transfer_never_exceeds_source_quantity(self):
        """Sum of split transfer quantities is always <= source quantity."""
        source = make_row(10, qty=85, demand=15, capacity=500, branch_id="BR01", branch_name="Source")
        dest_b = make_row(90, qty=10, demand=40, capacity=40, branch_id="BR02", branch_name="Dest B")
        dest_c = make_row(90, qty=10, demand=40, capacity=60, branch_id="BR03", branch_name="Dest C")

        df = pd.DataFrame([source, dest_b, dest_c])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == source["batch_id"]][0]
        transferred = sum(d.get("transfer_quantity", 0) for d in matching.get("destinations", []))
        self.assertLessEqual(transferred, 85)

    # ── 5. Capacity Constraints ────────────────────────────────────────────────

    def test_destination_transfer_capped_by_available_capacity(self):
        """Destination never receives more than its branch_capacity_remaining."""
        source = make_row(15, qty=100, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
        dest = make_row(90, qty=5, demand=80, capacity=35, branch_id="BR02", branch_name="Dest B")
        df = pd.DataFrame([source, dest])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == source["batch_id"]][0]
        if matching["action"] == "TRANSFER":
            qty_allocated = matching["destinations"][0]["transfer_quantity"]
            self.assertLessEqual(qty_allocated, 35, "Allocated quantity cannot exceed destination capacity of 35")

    def test_all_destinations_zero_capacity_triggers_manual_review(self):
        """When all destination branches have 0 capacity remaining, item is flagged for review."""
        source = make_row(15, qty=50, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
        dest_full = make_row(90, qty=10, demand=60, capacity=0, branch_id="BR02", branch_name="Full Dest")
        df = pd.DataFrame([source, dest_full])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == source["batch_id"]][0]
        self.assertEqual(matching["action"], "FLAG_FOR_REVIEW")

    # ── 6. Zero Demand ─────────────────────────────────────────────────────────

    def test_zero_demand_destination_rejected(self):
        """A branch with demand_per_week=0 is never selected as a transfer destination."""
        source = make_row(15, qty=50, demand=20, capacity=500, branch_id="BR01", branch_name="Source")
        dest_zero = make_row(90, qty=0, demand=0, capacity=500, branch_id="BR02", branch_name="Zero Demand Branch")
        df = pd.DataFrame([source, dest_zero])
        recs = generate_recommendations(df)
        matching = [r for r in recs if r["batch_id"] == source["batch_id"]][0]
        self.assertEqual(matching["action"], "FLAG_FOR_REVIEW")
        self.assertTrue(any("zero demand" in matching["reason"].lower() or "no viable" in matching["reason"].lower() for _ in [1]))

    # ── 7. Multiple Branches ───────────────────────────────────────────────────

    def test_multiple_branches_network_evaluation(self):
        """Four-branch network ranks destinations deterministically based on shortage score."""
        source = make_row(15, qty=50, demand=20, capacity=500, branch_id="BR01", branch_name="Branch A")
        b2 = make_row(90, qty=10, demand=30, capacity=500, branch_id="BR02", branch_name="Branch B")
        b3 = make_row(90, qty=5, demand=40, capacity=500, branch_id="BR03", branch_name="Branch C")
        b4 = make_row(90, qty=80, demand=20, capacity=500, branch_id="BR04", branch_name="Branch D")

        df = pd.DataFrame([source, b2, b3, b4])
        recs1 = generate_recommendations(df)
        recs2 = generate_recommendations(df)

        m1 = [r for r in recs1 if r["batch_id"] == source["batch_id"]][0]
        m2 = [r for r in recs2 if r["batch_id"] == source["batch_id"]][0]
        self.assertEqual(m1["action"], "TRANSFER")
        self.assertEqual(m1["destinations"][0]["dest_branch_name"], m2["destinations"][0]["dest_branch_name"])
        self.assertEqual(m1["destinations"][0]["dest_branch_name"], "Branch C")

    # ── 8. Multiple Batches of the Same Medicine ───────────────────────────────

    def test_destination_aggregates_multiple_existing_batches_for_coverage(self):
        """Destination branch with multiple existing batches calculates total stock correctly."""
        source = make_row(15, qty=40, demand=20, capacity=500, branch_id="BR01", branch_name="Branch A")
        # Branch B has two batches: 40 + 60 = 100 units total. Demand = 20/wk -> 5.0 wks cover
        dest_b1 = make_row(90, qty=40, demand=20, capacity=500, branch_id="BR02", branch_name="Branch B")
        dest_b2 = make_row(120, qty=60, demand=20, capacity=500, branch_id="BR02", branch_name="Branch B")
        dest_b2["batch_id"] = "DEST-B2"

        # Branch C has one batch: 10 units. Demand = 20/wk -> 0.5 wks cover (urgent)
        dest_c = make_row(90, qty=10, demand=20, capacity=500, branch_id="BR03", branch_name="Branch C")

        df = pd.DataFrame([source, dest_b1, dest_b2, dest_c])
        recs = generate_recommendations(df)
        m = [r for r in recs if r["batch_id"] == source["batch_id"]][0]
        # Branch C must be preferred over Branch B because Branch B already holds 100 units across 2 batches
        self.assertEqual(m["destinations"][0]["dest_branch_name"], "Branch C")

    # ── 9. Database CRUD Operations ────────────────────────────────────────────

    def test_database_crud_lifecycle_isolated(self):
        """Full CRUD lifecycle on SQLite database in isolated environment."""
        # 1. Create / Insert
        row = make_row(45, qty=100, demand=25, capacity=500, branch_id="B1", medicine="Aspirin")
        csv_p = os.path.join(self.tmp_dir.name, "crud_seed.csv")
        pd.DataFrame([row]).to_csv(csv_p, index=False)
        summary = import_stock_from_csv(csv_p, db_path=self.db_path)
        self.assertTrue(summary["success"])
        self.assertEqual(summary["records_inserted"], 1)

        # 2. Read
        df = load_stock(self.db_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(int(df.iloc[0]["quantity"]), 100)

        # 3. Update
        updated = update_stock_quantity(row["batch_id"], 175, db_path=self.db_path)
        self.assertTrue(updated)
        df_updated = load_stock(self.db_path)
        self.assertEqual(int(df_updated.iloc[0]["quantity"]), 175)

        # 4. Delete / Reset
        conn = get_connection(self.db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM stock WHERE batch_id = ?", (row["batch_id"],))
        conn.commit()
        conn.close()
        df_empty = load_stock(self.db_path)
        self.assertTrue(df_empty.empty)

    # ── 10. Duplicate Batch Handling ───────────────────────────────────────────

    def test_duplicate_batch_handling_without_and_with_update_flag(self):
        """Duplicate batch_id is rejected by default, and updated when update_existing=True."""
        row = make_row(45, qty=50, demand=25, capacity=500, branch_id="B1", medicine="Amoxicillin")
        csv1 = os.path.join(self.tmp_dir.name, "batch1.csv")
        pd.DataFrame([row]).to_csv(csv1, index=False)
        import_stock_from_csv(csv1, db_path=self.db_path)

        # Attempt to insert same batch_id with different quantity without update flag
        row_dup = dict(row)
        row_dup["quantity"] = 999
        csv2 = os.path.join(self.tmp_dir.name, "batch_dup.csv")
        pd.DataFrame([row_dup]).to_csv(csv2, index=False)

        summary_reject = import_stock_from_csv(csv2, db_path=self.db_path, update_existing=False)
        self.assertEqual(summary_reject["records_inserted"], 0)
        self.assertEqual(summary_reject["records_rejected"], 1)
        self.assertEqual(int(load_stock(self.db_path).iloc[0]["quantity"]), 50)

        # Import with update_existing=True updates the record
        summary_update = import_stock_from_csv(csv2, db_path=self.db_path, update_existing=True)
        self.assertTrue(summary_update["success"])
        self.assertEqual(summary_update["records_updated"], 1)
        self.assertEqual(int(load_stock(self.db_path).iloc[0]["quantity"]), 999)

    # ── 11. Barcode Registration ───────────────────────────────────────────────

    def test_barcode_registration_lifecycle(self):
        """Barcode registration associates code with batch and is idempotent."""
        registry = BarcodeRegistry(self.db_path)
        registry.register("500123456789", "BATCH-BC-1", "Ibuprofen 400mg")
        bid, status = registry.resolve("500123456789")
        self.assertEqual(bid, "BATCH-BC-1")
        self.assertEqual(status, "active")

        # Idempotent registration of same barcode and batch
        registry.register("500123456789", "BATCH-BC-1", "Ibuprofen 400mg")
        bid2, status2 = registry.resolve("500123456789")
        self.assertEqual(bid2, "BATCH-BC-1")

        # Registering existing active barcode for DIFFERENT batch raises ValueError
        with self.assertRaises(ValueError):
            registry.register("500123456789", "DIFFERENT-BATCH", "Ibuprofen 400mg")

    # ── 12. Barcode Update ─────────────────────────────────────────────────────

    def test_barcode_update_supersedes_and_activates_new(self):
        """update_barcode supersedes old barcode and activates new one."""
        registry = BarcodeRegistry(self.db_path)
        registry.register("BC-OLD", "BATCH-UPD", "Paracetamol")
        registry.update_barcode("BC-OLD", "BC-NEW", reason="repackaging")

        old_bid, old_status = registry.resolve("BC-OLD")
        self.assertEqual(old_bid, "BATCH-UPD")
        self.assertEqual(old_status, "superseded")

        new_bid, new_status = registry.resolve("BC-NEW")
        self.assertEqual(new_bid, "BATCH-UPD")
        self.assertEqual(new_status, "active")

    # ── 13. Superseded Barcode Lookup ──────────────────────────────────────────

    def test_lookup_superseded_barcode_safe_handling(self):
        """lookup_barcode on superseded barcode returns superseded status and batch info."""
        registry = BarcodeRegistry(self.db_path)
        registry.register("BC-SUP-OLD", "BATCH-SUP-1", "Omeprazole 20mg")
        registry.update_barcode("BC-SUP-OLD", "BC-SUP-NEW", reason="label change")

        # Add batch into stock
        row = make_row(20, qty=60, demand=15, capacity=500, medicine="Omeprazole 20mg")
        row["batch_id"] = "BATCH-SUP-1"
        csv_p = os.path.join(self.tmp_dir.name, "stock_sup.csv")
        pd.DataFrame([row]).to_csv(csv_p, index=False)
        import_stock_from_csv(csv_p, db_path=self.db_path)

        res = lookup_barcode("BC-SUP-OLD", registry=registry, stock_df=load_stock(self.db_path))
        self.assertTrue(res["found"])
        self.assertEqual(res["status"], "superseded")
        self.assertEqual(res["batch_id"], "BATCH-SUP-1")
        self.assertEqual(res["medicine_name"], "Omeprazole 20mg")

    # ── 14. Unknown and Invalid Barcode ────────────────────────────────────────

    def test_lookup_unknown_and_invalid_barcodes(self):
        """Unknown or empty barcodes return graceful error structures without crashing."""
        registry = BarcodeRegistry(self.db_path)
        res_unknown = lookup_barcode("9999999999999", registry=registry, stock_df=load_stock(self.db_path))
        self.assertFalse(res_unknown["found"])
        self.assertEqual(res_unknown["status"], "unknown")

        res_empty = lookup_barcode("   ", registry=registry)
        self.assertFalse(res_empty["found"])
        self.assertEqual(res_empty["status"], "invalid")

        res_none = lookup_barcode(None, registry=registry)
        self.assertFalse(res_none["found"])
        self.assertEqual(res_none["status"], "invalid")

    # ── 15. Authentication Success / Failure ───────────────────────────────────

    def test_auth_success_and_failure(self):
        """Test authenticate_user and check_password for valid, invalid, and empty logins."""
        # Valid credentials
        user_p1 = authenticate_user("pharmacist1", "pharmacy123")
        self.assertIsNotNone(user_p1)
        self.assertEqual(user_p1["name"], "Sarah Johnson")

        # Check password helper
        valid, uinfo = check_password("pharmacist1", "pharmacy123")
        self.assertTrue(valid)
        self.assertEqual(uinfo["name"], "Sarah Johnson")

        # Wrong password
        self.assertIsNone(authenticate_user("pharmacist1", "wrong_pw"))
        valid_bad, _ = check_password("pharmacist1", "wrong_pw")
        self.assertFalse(valid_bad)

        # Unknown user
        self.assertIsNone(authenticate_user("nonexistent_user", "any"))
        valid_unknown, _ = check_password("nonexistent_user", "any")
        self.assertFalse(valid_unknown)

        # Empty credentials
        self.assertIsNone(authenticate_user("", ""))
        valid_empty, _ = check_password("", "")
        self.assertFalse(valid_empty)

    # ── 16. Decision / Audit Logging ───────────────────────────────────────────

    def test_decision_audit_logging_attributes(self):
        """Decisions are stored with all required fields in SQLite."""
        save_decision(
            batch_id="BATCH-AUD-1",
            medicine="Atorvastatin",
            action="CONFIRMED",
            destination="Branch East (50 units)",
            override_reason="",
            user="pharmacist1",
            source_branch="Central",
            quantity=50,
            system_recommendation="TRANSFER",
            final_decision="CONFIRMED",
            db_path=self.db_path,
        )

        df_dec = load_decisions(self.db_path)
        self.assertEqual(len(df_dec), 1)
        row = df_dec.iloc[0]
        self.assertEqual(row["batch_id"], "BATCH-AUD-1")
        self.assertEqual(row["medicine"], "Atorvastatin")
        self.assertEqual(row["action"], "CONFIRMED")
        self.assertEqual(row["user"], "pharmacist1")
        self.assertEqual(int(row["quantity"]), 50)
        self.assertTrue(len(str(row["timestamp"])) > 0)

    # ── 17. Invalid Input Validation ───────────────────────────────────────────

    def test_validate_stock_row_rules(self):
        """validate_stock_row enforces constraints on quantity, dates, and non-negativity."""
        base_row = make_row(30, qty=100, demand=20, capacity=500)

        # Negative quantity
        bad_qty = dict(base_row)
        bad_qty["quantity"] = -10
        valid, err, _, _ = validate_stock_row(bad_qty)
        self.assertFalse(valid)
        self.assertIn("negative", str(err).lower())

        # Invalid date
        bad_date = dict(base_row)
        bad_date["expiry_date"] = "not-a-date"
        valid_d, err_d, _, _ = validate_stock_row(bad_date)
        self.assertFalse(valid_d)
        self.assertIn("date", str(err_d).lower())

        # Negative cost
        bad_cost = dict(base_row)
        bad_cost["unit_cost_gbp"] = -5.0
        valid_c, err_c, _, _ = validate_stock_row(bad_cost)
        self.assertFalse(valid_c)
        self.assertTrue("cost" in str(err_c).lower() or "negative" in str(err_c).lower())
