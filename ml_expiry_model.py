# ml_expiry_model.py
"""
Machine Learning Expiry & Wastage Risk Prediction Component.

Predicts the probability and risk class ('Low', 'Medium', 'High') of a medicine
batch expiring before it can be consumed locally.

Architecture:
Medicine Data -> ML Expiry Risk Prediction -> Existing Rule-Based Engine -> Final Recommendation
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

NUMERIC_FEATURES = [
    "days_to_expiry",
    "quantity",
    "avg_daily_demand",
    "stock_to_demand_ratio",
    "stock_value",
    "unit_cost_gbp",
    "branch_capacity_remaining",
]

CATEGORICAL_FEATURES = ["branch_id"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class MLExpiryPredictor:
    """
    Explainable Random Forest model to predict expiry and inventory wastage risk.
    """

    def __init__(self):
        self.is_trained = False
        self.model = None
        self.evaluation_metrics = {}
        self.evaluation_report = ""

    def _engineer_features(self, df):
        """Extract and engineer required ML features from raw inventory data."""
        data = df.copy()

        # Days to expiry
        if "days_to_expiry" not in data.columns:
            if "expiry_date" in data.columns:
                now = datetime.today()
                data["days_to_expiry"] = (
                    pd.to_datetime(data["expiry_date"], errors="coerce") - now
                ).dt.days.fillna(0).astype(int)
            elif "dte" in data.columns:
                data["days_to_expiry"] = data["dte"].fillna(0).astype(int)
            else:
                data["days_to_expiry"] = 0

        # Quantity
        data["quantity"] = pd.to_numeric(data.get("quantity", 0), errors="coerce").fillna(0).astype(float)

        # Demand
        weekly_dem = pd.to_numeric(data.get("demand_per_week", 0), errors="coerce").fillna(0).astype(float)
        data["avg_daily_demand"] = (weekly_dem / 7.0).round(3)
        data["stock_to_demand_ratio"] = (data["quantity"] / np.maximum(weekly_dem, 0.1)).round(2)

        # Unit cost and stock value
        unit_cost = pd.to_numeric(data.get("unit_cost_gbp", 1.0), errors="coerce").fillna(1.0).astype(float)
        data["unit_cost_gbp"] = unit_cost
        data["stock_value"] = (data["quantity"] * unit_cost).round(2)

        # Branch capacity
        data["branch_capacity_remaining"] = pd.to_numeric(
            data.get("branch_capacity_remaining", 500), errors="coerce"
        ).fillna(500).astype(float)

        # Branch ID
        data["branch_id"] = data.get("branch_id", "BR01").astype(str).fillna("BR01")

        return data

    @staticmethod
    def derive_ground_truth_label(row):
        """
        Derive ground-truth expiry risk class for evaluation based on clinical demand-absorption:
        - High: Expiry within 30 days and excess stock exceeding remaining local demand.
        - Medium: Expiry within 60 days with stock exceeding remaining local demand.
        - Low: Stock can be fully absorbed by local demand before expiry, or expiry > 60 days.
        """
        dte = float(row.get("days_to_expiry", 0))
        qty = float(row.get("quantity", 0))
        daily_dem = float(row.get("avg_daily_demand", 0))

        # Expected local sales before expiry
        expected_sales = daily_dem * max(dte, 0)
        excess = qty - expected_sales

        if dte <= 30 and excess > 0:
            return "High"
        elif dte <= 60 and excess > 0:
            return "Medium"
        else:
            return "Low"

    def train(self, df=None, test_size=0.25, random_state=42):
        """
        Train the explainable Random Forest classifier with cross-validation/test split.
        Computes actual accuracy, precision, recall, and F1-score.
        """
        if df is None:
            from database import load_stock
            df = load_stock()

        if df.empty:
            raise ValueError("Cannot train ML model on empty stock dataset")

        prepared = self._engineer_features(df)
        prepared["target_risk"] = prepared.apply(self.derive_ground_truth_label, axis=1)

        X = prepared[ALL_FEATURES]
        y = prepared["target_risk"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ]
        )

        # Explainable, non-deep learning classifier
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=50,
                        max_depth=5,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Actual evaluation metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec_w = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec_w = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1_w = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        rep = classification_report(y_test, y_pred, zero_division=0)

        self.model = pipeline
        self.is_trained = True
        self.evaluation_metrics = {
            "total_samples": len(prepared),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": round(acc, 4),
            "precision_weighted": round(prec_w, 4),
            "recall_weighted": round(rec_w, 4),
            "f1_weighted": round(f1_w, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "dataset_limitation_note": (
                f"Evaluation is conducted on operational sample data (N={len(prepared)}, test N={len(X_test)}). "
                "While precision/accuracy are representative for this distribution, real-world deployment requires "
                "continuous validation against multi-year longitudinal pharmacy dispensing and waste records."
            ),
        }
        self.evaluation_report = rep
        return self.evaluation_metrics

    def predict_batch(self, batch_data):
        """
        Predict expiry risk for a single batch.
        Returns:
            dict with:
                - expiry_risk_probability (float 0.0 - 1.0)
                - risk_class ('Low', 'Medium', 'High')
                - class_probabilities (dict)
        """
        if not self.is_trained or self.model is None:
            self.train()

        if isinstance(batch_data, dict):
            row_df = pd.DataFrame([batch_data])
        elif isinstance(batch_data, pd.Series):
            row_df = pd.DataFrame([batch_data.to_dict()])
        elif isinstance(batch_data, pd.DataFrame):
            row_df = batch_data.iloc[[0]]
        else:
            raise TypeError("batch_data must be dict, pd.Series, or pd.DataFrame")

        prepared = self._engineer_features(row_df)
        X = prepared[ALL_FEATURES]

        classes = list(self.model.classes_)
        probs = self.model.predict_proba(X)[0]
        prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}

        pred_class = str(self.model.predict(X)[0])

        # Risk probability: probability that batch is at High or Medium wastage risk
        # P(wastage risk) = P(High) + 0.5 * P(Medium) or 1.0 - P(Low)
        p_low = prob_dict.get("Low", 0.0)
        risk_prob = round(float(1.0 - p_low), 3)

        return {
            "expiry_risk_probability": max(0.0, min(1.0, risk_prob)),
            "risk_class": pred_class,
            "class_probabilities": {k: round(v, 3) for k, v in prob_dict.items()},
        }

    def predict_dataframe(self, df):
        """Enrich an entire dataframe of inventory with ML predictions."""
        if not self.is_trained or self.model is None:
            self.train()

        if df.empty:
            df_out = df.copy()
            df_out["ml_risk_probability"] = []
            df_out["ml_risk_class"] = []
            return df_out

        prepared = self._engineer_features(df)
        X = prepared[ALL_FEATURES]

        classes = list(self.model.classes_)
        all_probs = self.model.predict_proba(X)
        preds = self.model.predict(X)

        low_idx = classes.index("Low") if "Low" in classes else None

        if low_idx is not None:
            risk_probs = (1.0 - all_probs[:, low_idx]).round(3)
        else:
            risk_probs = np.ones(len(df))

        df_out = df.copy()
        df_out["ml_risk_probability"] = risk_probs
        df_out["ml_risk_class"] = preds
        return df_out


# Global cached singleton instance
_GLOBAL_PREDICTOR = None


def get_ml_predictor():
    """Retrieve or initialize the global ML expiry predictor singleton."""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None or not _GLOBAL_PREDICTOR.is_trained:
        predictor = MLExpiryPredictor()
        predictor.train()
        _GLOBAL_PREDICTOR = predictor
    return _GLOBAL_PREDICTOR
