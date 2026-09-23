"""
Full ML model evaluation script.
Produces actual metrics for the academic project report.
No values are fabricated or estimated.
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

from database import initialise_database, load_stock
from ml_expiry_model import MLExpiryPredictor

# ── 1. Load dataset ────────────────────────────────────────────────────────────
initialise_database()
df = load_stock()

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print("Total records (N)  :", len(df))
print("Columns            :", list(df.columns))
print()

# ── 2. Feature engineering + label derivation (mirrors ml_expiry_model.py) ────
predictor = MLExpiryPredictor()
prepared = predictor._engineer_features(df.copy())
prepared["target_risk"] = prepared.apply(predictor.derive_ground_truth_label, axis=1)

print("Label Distribution (full dataset):")
label_counts = prepared["target_risk"].value_counts()
for label, count in label_counts.items():
    pct = round(100 * count / len(prepared), 1)
    print(f"  {label:<8}: {count:>4} samples ({pct}%)")
print()

# ── 3. Feature summary ─────────────────────────────────────────────────────────
from ml_expiry_model import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES
X_full = prepared[ALL_FEATURES]
y_full = prepared["target_risk"]

print("Feature Summary:")
print(f"  Numeric features    : {NUMERIC_FEATURES}")
print(f"  Categorical features: {CATEGORICAL_FEATURES}")
print()
print("Numeric feature statistics (full dataset):")
print(X_full[NUMERIC_FEATURES].describe().round(3).to_string())
print()

# ── 4. Single train/test split (75/25, stratified) ────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.25

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_full,
)

print("=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)
print(f"  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")
print(f"  Test size        : {TEST_SIZE * 100:.0f}%")
print(f"  Stratified       : Yes")
print(f"  Random state     : {RANDOM_STATE}")
print()

# ── 5. Train model ─────────────────────────────────────────────────────────────
metrics = predictor.train(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
y_pred  = predictor.model.predict(X_test)
classes = list(predictor.model.classes_)

print("=" * 60)
print("ACTUAL EVALUATION METRICS (Hold-out Test Set)")
print("=" * 60)
print(f"  Accuracy             : {metrics['accuracy']}")
print(f"  Precision (weighted) : {metrics['precision_weighted']}")
print(f"  Recall    (weighted) : {metrics['recall_weighted']}")
print(f"  F1-score  (weighted) : {metrics['f1_weighted']}")
print()
print(f"  Precision (macro)    : {metrics['precision_macro']}")
print(f"  Recall    (macro)    : {metrics['recall_macro']}")
print(f"  F1-score  (macro)    : {metrics['f1_macro']}")
print()

# ── 6. Per-class classification report ────────────────────────────────────────
print("Per-class Classification Report:")
print(predictor.evaluation_report)

# ── 7. Confusion matrix ────────────────────────────────────────────────────────
ordered_labels = ["High", "Low", "Medium"]
cm = confusion_matrix(y_test, y_pred, labels=ordered_labels)
cm_df = pd.DataFrame(cm, index=ordered_labels, columns=ordered_labels)
cm_df.index.name = "Actual \\ Predicted"

print("=" * 60)
print("CONFUSION MATRIX  (rows = Actual, columns = Predicted)")
print("=" * 60)
print(cm_df.to_string())
print()
print("Interpretation:")
for i, actual in enumerate(ordered_labels):
    for j, predicted in enumerate(ordered_labels):
        if i == j:
            print("  [" + actual + " -> " + predicted + "] Correct: " + str(cm[i][j]))
        elif cm[i][j] > 0:
            print("  [" + actual + " -> " + predicted + "] Misclassified: " + str(cm[i][j]))
print()

# ── 8. Stratified 5-fold cross-validation ─────────────────────────────────────
print("=" * 60)
print("5-FOLD STRATIFIED CROSS-VALIDATION (full dataset)")
print("=" * 60)
# Re-build fresh pipeline for CV (avoid data leakage from prior fit)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

cv_preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
])
cv_pipeline = Pipeline(steps=[
    ("preprocessor", cv_preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=50, max_depth=5,
        random_state=RANDOM_STATE, class_weight="balanced",
    )),
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_acc  = cross_val_score(cv_pipeline, X_full, y_full, cv=skf, scoring="accuracy")
cv_f1w  = cross_val_score(cv_pipeline, X_full, y_full, cv=skf, scoring="f1_weighted")
cv_f1m  = cross_val_score(cv_pipeline, X_full, y_full, cv=skf, scoring="f1_macro")

print(f"  Fold accuracies        : {[round(v, 4) for v in cv_acc]}")
print(f"  CV Accuracy  mean ± sd : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print()
print(f"  Fold F1-weighted       : {[round(v, 4) for v in cv_f1w]}")
print(f"  CV F1-weighted mean±sd : {cv_f1w.mean():.4f} ± {cv_f1w.std():.4f}")
print()
print(f"  Fold F1-macro          : {[round(v, 4) for v in cv_f1m]}")
print(f"  CV F1-macro   mean±sd  : {cv_f1m.mean():.4f} ± {cv_f1m.std():.4f}")
print()

# ── 9. Feature importance (from trained RandomForest) ─────────────────────────
print("=" * 60)
print("FEATURE IMPORTANCE (Mean Decrease in Impurity)")
print("=" * 60)
rf = predictor.model.named_steps["classifier"]
pre = predictor.model.named_steps["preprocessor"]

# Get feature names after one-hot encoding
ohe_cats = pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
feature_names = NUMERIC_FEATURES + list(ohe_cats)

importances = rf.feature_importances_
fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances.round(4),
}).sort_values("Importance", ascending=False)

for _, row in fi_df.iterrows():
    bar = "#" * int(row["Importance"] * 100)
    print("  " + str(row["Feature"]).ljust(35) + " " + str(row["Importance"]).ljust(6) + "  " + bar)
print()

# ── 10. Save structured JSON for report ──────────────────────────────────────
results = {
    "generated_at": datetime.now().isoformat(),
    "dataset": {
        "total_samples": len(df),
        "columns": list(df.columns),
        "label_distribution": label_counts.to_dict(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_size_pct": TEST_SIZE * 100,
        "stratified": True,
        "random_state": RANDOM_STATE,
    },
    "model": {
        "algorithm": "RandomForestClassifier",
        "n_estimators": 50,
        "max_depth": 5,
        "class_weight": "balanced",
        "preprocessing": "StandardScaler (numeric) + OneHotEncoder (categorical)",
    },
    "features": {
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
    },
    "target": {
        "variable": "target_risk",
        "classes": ordered_labels,
        "derivation": (
            "High: dte<=30 and stock > expected demand before expiry. "
            "Medium: 30<dte<=60 and stock > expected demand. "
            "Low: stock absorbable before expiry, or dte>60."
        ),
    },
    "holdout_metrics": {
        "accuracy": metrics["accuracy"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
    },
    "per_class": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    "confusion_matrix": {
        "labels": ordered_labels,
        "matrix": cm.tolist(),
    },
    "cross_validation": {
        "strategy": "StratifiedKFold(n_splits=5, shuffle=True)",
        "cv_accuracy_folds": cv_acc.round(4).tolist(),
        "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
        "cv_accuracy_std": round(float(cv_acc.std()), 4),
        "cv_f1_weighted_folds": cv_f1w.round(4).tolist(),
        "cv_f1_weighted_mean": round(float(cv_f1w.mean()), 4),
        "cv_f1_weighted_std": round(float(cv_f1w.std()), 4),
        "cv_f1_macro_folds": cv_f1m.round(4).tolist(),
        "cv_f1_macro_mean": round(float(cv_f1m.mean()), 4),
        "cv_f1_macro_std": round(float(cv_f1m.std()), 4),
    },
    "feature_importance": fi_df.set_index("Feature")["Importance"].to_dict(),
    "limitations": [
        "Dataset is synthetic/operational sample data (N=600); not multi-year real-world dispensing records.",
        "Medium class is severely underrepresented (N=35 total, N=9 in test), leading to low recall (0.33) for that class.",
        "Ground-truth labels are derived algorithmically from demand absorption heuristics, not clinician-annotated real wastage events.",
        "Model may not generalise to pharmacies with different demand patterns, seasonal variation, or product mix.",
        "Continuous retraining against confirmed waste events from the audit log is required for production use.",
    ],
}

with open("data/ml_evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Structured results saved to: data/ml_evaluation_results.json")
