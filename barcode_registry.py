import pandas as pd
from datetime import datetime

class BarcodeRegistry:
    """Maps barcodes to stable batch identities.
    A barcode can change — a batch_id never does."""

    def __init__(self, path="data/barcode_history.csv"):
        self.path = path
        try:
            self.df = pd.read_csv(path)
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=[
                "barcode","batch_id","medicine_name",
                "registered_date","superseded_date","reason_for_change"
            ])

    def resolve(self, barcode):
        """Return (batch_id, status) for any barcode."""
        active = self.df[
            (self.df["barcode"] == barcode) &
            (self.df["superseded_date"].isna())
        ]
        if not active.empty:
            return active.iloc[0]["batch_id"], "active"

        old = self.df[self.df["barcode"] == barcode]
        if not old.empty:
            row = old.sort_values("superseded_date",ascending=False).iloc[0]
            return row["batch_id"], "superseded"

        return None, "unknown"

    def register(self, barcode, batch_id, medicine_name,
                 reason="initial"):
        today = datetime.today().strftime("%Y-%m-%d")
        existing = self.df[
            (self.df["barcode"] == barcode) &
            (self.df["superseded_date"].isna())
        ]
        if not existing.empty:
            if existing.iloc[0]["batch_id"] != batch_id:
                raise ValueError(
                    f"Barcode {barcode} already assigned to "
                    f"{existing.iloc[0]['batch_id']}")
            return
        new_row = {
            "barcode": barcode, "batch_id": batch_id,
            "medicine_name": medicine_name,
            "registered_date": today,
            "superseded_date": None,
            "reason_for_change": reason,
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_row])],
                            ignore_index=True)
        self.df.to_csv(self.path, index=False)

    def update_barcode(self, old_bc, new_bc, reason):
        today = datetime.today().strftime("%Y-%m-%d")
        mask = ((self.df["barcode"] == old_bc) &
                (self.df["superseded_date"].isna()))
        if self.df[mask].empty:
            raise ValueError(f"Barcode {old_bc} not found")
        batch_id      = self.df[mask].iloc[0]["batch_id"]
        medicine_name = self.df[mask].iloc[0]["medicine_name"]
        self.df.loc[mask, "superseded_date"] = today
        self.register(new_bc, batch_id, medicine_name, reason)

    def get_all_batches(self):
        return set(self.df["batch_id"].unique())