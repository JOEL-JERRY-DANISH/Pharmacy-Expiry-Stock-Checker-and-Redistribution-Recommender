import unittest
import pandas as pd
import tempfile, os
from datetime import datetime, timedelta
from recommender import generate_recommendations
from barcode_registry import BarcodeRegistry

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

if __name__ == "__main__":
    unittest.main(verbosity=2)