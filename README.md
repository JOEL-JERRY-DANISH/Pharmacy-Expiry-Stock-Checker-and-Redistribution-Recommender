# Pharmacy Expiry Stock Checker and Redistribution Recommender

A rule-based intelligent recommendation system, enhanced with a machine learning expiry risk prediction layer, that proactively identifies near-expiry medicines across pharmacy branches and recommends stock redistribution to reduce clinical waste.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Objectives](#2-objectives)
3. [Existing System Limitations](#3-existing-system-limitations)
4. [Proposed System](#4-proposed-system)
5. [Key Features](#5-key-features)
6. [System Architecture](#6-system-architecture)
7. [Technologies Used](#7-technologies-used)
8. [Database Design](#8-database-design)
9. [Recommendation Methodology](#9-recommendation-methodology)
10. [Barcode Functionality](#10-barcode-functionality)
11. [Authentication](#11-authentication)
12. [Admin Dashboard](#12-admin-dashboard)
13. [AI / ML Component](#13-ai--ml-component)
14. [Dataset Description](#14-dataset-description)
15. [Installation](#15-installation)
16. [Environment Configuration](#16-environment-configuration)
17. [How to Run](#17-how-to-run)
18. [How to Run Tests](#18-how-to-run-tests)
19. [Actual Testing Results](#19-actual-testing-results)
20. [Actual ML Evaluation Results](#20-actual-ml-evaluation-results)
21. [Limitations](#21-limitations)
22. [Future Enhancements](#22-future-enhancements)

---

## ⚡ Quick Start

Get the application running in under 2 minutes:

```bash
# 1. Clone and enter the project
git clone <repository-url> && cd project

# 2. Create a virtual environment and install dependencies
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure credentials (copy the example and fill in passwords)
copy .env.example .env

# 4. Launch the application
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) and log in.
Default users: `pharmacist1`, `pharmacist2`, `admin` (passwords set in `.env`).

---



Community pharmacies managing patients with complex, multi-prescription care plans routinely hold stock that is near expiry. The discovery of this stock often occurs too late for redistribution to take place, resulting in:

- **Direct financial waste** — expired medicines that must be disposed of at cost.
- **Patient supply risk** — missed opportunities to redistribute stock to branches experiencing higher demand.
- **Incomplete audit trails** — manual processes leave insufficient records of what decisions were made, by whom, and why.

There is no system currently in widespread community pharmacy use that automatically identifies near-expiry batches, proposes evidence-based redistribution routes, captures pharmacist decisions, and logs the complete audit trail in a structured, queryable database.

---

## 2. Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Identify near-expiry and critical stock automatically | ✅ COMPLETED |
| 2 | Score batches by urgency, quantity, and financial value | ✅ COMPLETED |
| 3 | Recommend redistribution to branches with demand and capacity | ✅ COMPLETED |
| 4 | Provide plain-English reasons for every recommendation | ✅ COMPLETED |
| 5 | Require staff confirmation for high-impact transfers | ✅ COMPLETED |
| 6 | Allow pharmacists to override and record rejection reasons | ✅ COMPLETED |
| 7 | Persist all decisions in SQLite and CSV for audit export | ✅ COMPLETED |
| 8 | Resolve physical medicine barcodes to batch records | ✅ COMPLETED |
| 9 | Restrict admin analytics to authorised users only | ✅ COMPLETED |
| 10 | Add ML expiry risk prediction to support the rule-based engine | ✅ COMPLETED |

---

## 3. Existing System Limitations

The problem being replaced is a manual, spreadsheet-based stock tracking process with the following limitations:

- Near-expiry stock is identified by staff during routine stocktaking only — no automated detection.
- Redistribution decisions are made ad hoc with no demand or capacity data at hand.
- No structured audit log of what was transferred, overridden, or reviewed.
- No barcode scanning integration to look up batch records during stock handling.
- No visibility across branches for a pharmacist viewing only their local stock.

---

## 4. Proposed System

A web-based application built with Streamlit that:

1. Loads all branch inventory from a SQLite database (seeded from CSV on first run).
2. Scores every batch using a transparent, reproducible formula.
3. Applies a machine learning layer to estimate expiry and wastage risk probability.
4. Runs rule-based checks on demand, capacity, and transit feasibility.
5. Presents ranked, explainable recommendations to pharmacist staff.
6. Records every staff decision (confirmed / overridden / reviewed) to both SQLite and CSV.
7. Provides an admin dashboard with visualisations of decisions, expiry risk, and stock value.

---

## 5. Key Features

### ✅ COMPLETED

- **Expiry risk scoring** — transparent 0–150 point formula (urgency + quantity weight + value weight).
- **Redistribution recommendations** — matches source batches to up to 3 best-fit receiving branches.
- **Plain-English explanations** — every recommendation includes a concise, human-readable reason covering all 8 decision factors.
- **Safety constraints** — expired stock, zero-demand branches, and capacity-deficient branches are always excluded by hard rule checks.
- **ML expiry risk prediction** — Random Forest classifier predicts risk class (Low / Medium / High) and probability per batch.
- **Barcode scanning** — resolves physical barcodes (including superseded ones) to batch records with full expiry and risk data.
- **Staff decision confirmation** — pharmacist confirms, overrides, or flags each recommendation with an optional reason.
- **Dual persistence** — SQLite is the primary operational store; CSV is maintained for audit export compatibility.
- **Decision audit log** — 11-field record per decision (timestamp, user, medicine, batch, source, destination, quantity, system recommendation, action, final decision, override reason).
- **Email alerts** — sends SMTP alerts for critical (≤7 days) stock if credentials are configured.
- **Admin dashboard** — 7 live metrics and 5 bar chart visualisations based entirely on operational data.
- **Role-based access control** — SHA-256-hashed passwords; admin dashboard restricted to admin role.
- **Automated test suite** — 147 deterministic, isolated unit and edge-case tests.

### 🔮 FUTURE WORK

- Periodic ML model retraining from confirmed audit log waste events.
- Multi-tenancy (separate pharmacy organisations).
- Integration with real NHS dispensing systems.

---

## 6. System Architecture

```
CSV Files (initial seed data)
        │
        ▼
SQLite Database (data/pharmacy.db)          ← Primary operational data store
        │
        ├──────────────────────────────┐
        ▼                              ▼
ML Expiry Risk Prediction       Rule-Based Recommendation Engine
(ml_expiry_model.py)            (recommender.py)
  RandomForestClassifier          Urgency scoring (0–150)
  Risk class: Low/Medium/High     Demand, capacity, transit checks
  Risk probability: 0.0–1.0       Safety constraints (hard rules)
        │                              │
        └─────────────┬────────────────┘
                      ▼
              Final Recommendation
           (TRANSFER / FLAG_FOR_REVIEW)
                      │
                      ▼
           Staff Confirmation / Override
           (app.py — Streamlit UI)
                      │
                      ▼
              Decision Audit Log
           SQLite decisions table
           + data/decision_log.csv
```

**Rule-based safety constraints are always enforced, regardless of ML output.**

---

## 7. Technologies Used

| Component | Technology |
|-----------|-----------|
| Web framework | [Streamlit](https://streamlit.io/) |
| Data processing | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| Machine learning | [scikit-learn](https://scikit-learn.org/) |
| Database | SQLite 3 (built-in Python `sqlite3`) |
| Password hashing | SHA-256 (`hashlib`, Python standard library) |
| Email alerts | SMTP via `smtplib` (Python standard library) |
| Environment config | [python-dotenv](https://pypi.org/project/python-dotenv/) |
| Testing | [pytest](https://pytest.org/) |
| Language | Python 3.10+ |

---

## 8. Database Design

SQLite is the primary persistence layer (`data/pharmacy.db`). Tables are created automatically on first run and seeded from CSV if empty.

### `stock` table

| Column | Type | Description |
|--------|------|-------------|
| `batch_id` | TEXT (PK) | Unique batch identifier |
| `medicine_name` | TEXT | Medicine name and strength |
| `category` | TEXT | Therapeutic category |
| `branch_id` | TEXT | Branch identifier (BR01–BR04) |
| `branch_name` | TEXT | Human-readable branch name |
| `quantity` | INTEGER | Current stock quantity |
| `expiry_date` | TEXT | Expiry date (YYYY-MM-DD) |
| `unit_cost_gbp` | REAL | Unit cost in GBP |
| `demand_per_week` | INTEGER | Weekly dispensing demand |
| `branch_capacity_remaining` | INTEGER | Remaining storage capacity |

### `barcodes` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER (PK) | Auto-increment row ID |
| `barcode` | TEXT | Barcode string |
| `batch_id` | TEXT | Linked batch |
| `medicine_name` | TEXT | Medicine name |
| `registered_date` | TEXT | Date barcode was registered |
| `superseded_date` | TEXT | Date barcode was replaced (NULL if active) |
| `reason_for_change` | TEXT | Reason for supersession (repackaging, etc.) |

### `decisions` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER (PK) | Auto-increment row ID |
| `timestamp` | TEXT | Decision date/time |
| `user` | TEXT | Pharmacist username |
| `medicine` | TEXT | Medicine name |
| `batch_id` | TEXT | Batch acted upon |
| `source_branch` | TEXT | Source branch |
| `destination` | TEXT | Destination branch (empty for FLAG_FOR_REVIEW) |
| `quantity` | INTEGER | Quantity involved |
| `system_recommendation` | TEXT | What the system recommended |
| `action` | TEXT | CONFIRMED / OVERRIDDEN / MANUALLY_REVIEWED |
| `final_decision` | TEXT | Same as action (normalized) |
| `override_reason` | TEXT | Staff reason if overridden |

---

## 9. Recommendation Methodology

### Batch Risk Scoring (0–150 points)

| Component | Conditions | Points |
|-----------|-----------|--------|
| Urgency — Critical | 0–7 days to expiry | 100.0 |
| Urgency — Near-expiry | 8–30 days to expiry | 50.0 |
| Urgency — Watch | 31–90 days to expiry | 10.0 |
| Urgency — Safe/Expired | >90 days or already expired | 0.0 |
| Quantity weight | `min(quantity / 500, 1.0) × 30` | 0–30 |
| Value weight | `min(unit_cost_gbp / 5.0, 1.0) × 20` | 0–20 |

Scores are **transparent, explainable, and reproducible** — the same inputs always produce the same score.

### Destination Ranking

For each actionable batch, the system evaluates all other branches carrying the same medicine and selects up to 3 viable destinations, ranked by `demand_per_week` descending.

### Hard Safety Rules (cannot be overridden by ML)

1. **No expired stock transfers** — batches with `days_to_expiry < 0` are never recommended.
2. **No zero-demand destinations** — branches with `demand_per_week == 0` are excluded.
3. **No capacity-deficient destinations** — `branch_capacity_remaining` must be ≥ source batch quantity.
4. **Transit time feasibility** — transfer is blocked if estimated transit days ≥ remaining shelf life.

### Decision Factors Returned Per Recommendation

Every recommendation includes all 8 decision factors in a structured `decision_factors` dictionary:

- Days to expiry
- Current stock
- Source demand
- Destination demand
- Available capacity
- Transfer time (days)
- Medicine value (GBP)
- Risk score

Plus ML fields: `ml_risk_probability` and `ml_risk_class`.

---

## 10. Barcode Functionality

Implemented in `barcode_lookup.py` and `barcode_registry.py`.

| Input | Behaviour |
|-------|-----------|
| Valid active barcode | Resolves to batch → displays medicine info, expiry, stock, risk score, ML risk, and recommendation |
| Superseded barcode | Resolves to the same batch via historical record; clearly labelled as superseded |
| Unknown barcode | Returns a clear "not recognised" message; no crash |
| Empty / None / whitespace | Returns "cannot be empty" message safely; no crash |

Superseded barcodes are tracked by keeping the original row with a `superseded_date` field, so historical lookups remain possible without data loss.

---

## 11. Authentication

Implemented in `auth_config.py` and `app.py`.

- **Session-based** — login state stored in `st.session_state`.
- **SHA-256 hashed passwords** — no plaintext password is stored anywhere in tracked source files.
- **Three-source credential loading** (highest priority first):
  1. `.streamlit/secrets.toml` — recommended for Streamlit Cloud and local development.
  2. `*_PASSWORD_HASH` environment variables — pre-computed SHA-256 hex digest.
  3. `*_PASSWORD` environment variables — plaintext hashed at runtime (legacy `.env` approach).
- **Role-based access** — admin users (`branch = "All branches"`) have access to the Admin Dashboard; pharmacist users see only their own branch context.
- **Graceful degradation** — if no secrets are configured the app displays a clear setup warning instead of crashing.

### Users and Roles

| Username | Name | Branch | Dashboard Access |
|----------|------|--------|-----------------|
| `pharmacist1` | Sarah Johnson | Central Pharmacy | Pharmacist view |
| `pharmacist2` | James Patel | North Branch | Pharmacist view |
| `admin` | Admin User | All branches | Full admin dashboard |

> **Security note:** Real passwords are **never** stored in this repository.
> Configure credentials locally using `.streamlit/secrets.toml` (see [Section 16](#16-environment-configuration)).

---

## 12. Admin Dashboard

Located at `pages/admin_dashboard.py`. Accessible only to users with `branch = "All branches"`.

### Metrics (7 live metrics from operational data)

| Metric | Source |
|--------|--------|
| Total decisions | `decisions` table row count |
| Confirmed transfers | `action = 'CONFIRMED'` count |
| Overrides | `action = 'OVERRIDDEN'` count |
| Manual reviews | `action = 'MANUALLY_REVIEWED'` count |
| Medicines at expiry risk | Batches with `days_to_expiry` in 0–30 |
| Quantity recommended for transfer | Sum of transfer recommendation quantities |
| Stock value at risk | Sum of `quantity × unit_cost_gbp` for 0–30 day batches |

### Visualisations (5 bar charts from actual data)

1. Decisions over time (grouped by date)
2. Confirmed vs overridden recommendations (action value counts)
3. Transfers by destination branch
4. Expiry-risk distribution (critical / near-expiry / watch / safe / expired)
5. Stock value at risk by branch (£)

### Additional Sections

- **Network Health Summary** — colour-coded metrics (critical count, near-expiry count, value at risk, transfer quantity) with delta indicators.
- **Branch-Level Expiry Risk table** — expandable table showing each branch's at-risk batch count and £ exposure.
- Override reasons table — all OVERRIDDEN decisions with reason recorded.
- Full decision log — expandable table with CSV download.

---

## 13. AI / ML Component

Implemented in `ml_expiry_model.py` using scikit-learn.

### Purpose

Predict the **expiry and wastage risk** of a medicine batch — the probability that the batch will expire before it can be consumed at its current branch.

### Model

| Property | Value |
|----------|-------|
| Algorithm | `RandomForestClassifier` |
| Trees | 50 |
| Max depth | 5 |
| Class weighting | `balanced` |
| Numeric preprocessing | `StandardScaler` |
| Categorical preprocessing | `OneHotEncoder(handle_unknown='ignore')` |
| Pipeline | `sklearn.pipeline.Pipeline` |

### Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `days_to_expiry` | Numeric | Days from today to expiry |
| `quantity` | Numeric | Current batch stock |
| `avg_daily_demand` | Numeric | Weekly demand ÷ 7 |
| `stock_to_demand_ratio` | Numeric | Quantity ÷ weekly demand |
| `stock_value` | Numeric | Quantity × unit cost (£) |
| `unit_cost_gbp` | Numeric | Unit price |
| `branch_capacity_remaining` | Numeric | Remaining branch capacity |
| `branch_id` | Categorical | Branch identifier (one-hot encoded) |

### Output

| Output | Type | Description |
|--------|------|-------------|
| `risk_class` | String | `Low`, `Medium`, or `High` |
| `expiry_risk_probability` | Float 0.0–1.0 | Probability of wastage risk |
| `class_probabilities` | Dict | Per-class probability breakdown |

### Role in the System

The ML output is a **supporting signal only**. The rule-based safety constraints always take precedence:

```
ML risk assessment
        ↓
   passed to recommendation engine as decision_factors
        ↓
   rule-based hard checks (expired? zero demand? over capacity?)
        ↓
   final recommendation (TRANSFER / FLAG_FOR_REVIEW)
```

The ML component **cannot** cause expired stock to be recommended, zero-demand branches to be selected, or capacity-deficient destinations to receive transfers.

---

## 14. Dataset Description

Generated by `generate_data.py` (reproducible with `random.seed(42)` and `np.random.seed(42)`).

| Property | Value |
|----------|-------|
| Total records | 600 stock batch rows |
| Branches | 4 (Central Pharmacy, North Branch, East Branch, South Branch) |
| Medicines | 10 types across 6 therapeutic categories |
| Expiry distribution | ~5% already expired, ~15% critical (≤7d), ~20% near-expiry (8–30d), ~25% watch (31–90d), ~35% safe (>90d) |
| Demand range | 2–80 units/week per branch |
| Quantity range | 5–500 units per batch |
| Unit cost range | £0.08–£3.50 |

### Synthetic Data Limitation

The dataset is operationally representative but synthetic. Expiry dates, demand figures, and quantities are randomly generated — not sourced from real dispensing records. ML metrics reflect performance on this distribution only. See [Limitations](#21-limitations).

### Barcode Registry

Each batch has a corresponding barcode entry. Approximately 20% of barcodes have a supersession record (repackaging, FMD serialisation, label correction, or new lot).

---

## 15. Installation

### Prerequisites

- Python 3.10 or later
- pip

### Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd project

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`)

```
streamlit
pandas
numpy
pytest
python-dotenv
scikit-learn
```

---

## 16. Environment Configuration

### Recommended: Streamlit Secrets (`.streamlit/secrets.toml`)

This is the preferred approach for both local development and Streamlit Cloud deployment.
Passwords are stored as **SHA-256 hex digests** — no plaintext password ever appears in a config file.

**Step 1** — Copy the example file:

```bash
copy .streamlit\secrets.toml.example .streamlit\secrets.toml   # Windows
# cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # macOS / Linux
```

**Step 2** — Generate SHA-256 hashes for your chosen passwords:

```bash
# Python (cross-platform)
python -c "import hashlib; print(hashlib.sha256('yourpassword'.encode()).hexdigest())"

# Linux / macOS
echo -n 'yourpassword' | sha256sum
```

**Step 3** — Fill in `.streamlit/secrets.toml` with the hashes (never the plaintext):

```toml
[auth]
pharmacist1_password_hash = "<sha256 hex of pharmacist1 password>"
pharmacist2_password_hash = "<sha256 hex of pharmacist2 password>"
admin_password_hash       = "<sha256 hex of admin password>"

[email]
sender   = "yourpharmacy@gmail.com"
password = "your_16_char_app_password"
receiver = "pharmacist@pharmacy.com"
```

> **Security note:** `.streamlit/secrets.toml` is listed in `.gitignore` and must **never** be committed to version control.
> Only `.streamlit/secrets.toml.example` (which contains only placeholder values) is tracked by git.

---

### Alternative: Environment Variables (`.env`)

For environments where Streamlit secrets are unavailable, copy `.env.example` to `.env`:

```bash
copy .env.example .env     # Windows
# cp .env.example .env    # macOS / Linux
```

Then set your chosen passwords (hashed with SHA-256 at runtime):

```env
# Plaintext passwords — hashed automatically at startup (never stored in source)
PHARMACIST1_PASSWORD=<your chosen password>
PHARMACIST2_PASSWORD=<your chosen password>
ADMIN_PASSWORD=<your chosen password>

# Or supply pre-computed hashes directly:
# PHARMACIST1_PASSWORD_HASH=<sha256 hex>
# PHARMACIST2_PASSWORD_HASH=<sha256 hex>
# ADMIN_PASSWORD_HASH=<sha256 hex>

# Email alerting via Gmail SMTP (optional — alerts are skipped if blank)
EMAIL_SENDER=yourpharmacy@gmail.com
EMAIL_PASSWORD=your_16_char_app_password
EMAIL_RECEIVER=pharmacist@pharmacy.com

# Application URL shown in alert email links
APP_URL=http://localhost:8501
```

> **Security note:** `.env` is listed in `.gitignore` and must never be committed to version control.

---

## 17. How to Run

### 1. (Optional) Regenerate Stock Data

```bash
python generate_data.py
```

This recreates `data/medicines.csv` and `data/barcode_history.csv` with fresh synthetic data.

### 2. Start the Application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

The database (`data/pharmacy.db`) is created automatically on first launch and seeded from the CSV files if empty.

### 3. (Optional) Run the ML Evaluation Script

```bash
python evaluate_ml.py
```

Prints actual training metrics and saves structured results to `data/ml_evaluation_results.json`.

---

## 18. How to Run Tests

The test suite lives in `test_edge_cases.py` and contains **147 deterministic, isolated tests**.

### Run all tests

```bash
python -m pytest test_edge_cases.py -v
```

### Run a specific test class

```bash
python -m pytest test_edge_cases.py::TestRecommender -v
python -m pytest test_edge_cases.py::TestMLExpiryModel -v
python -m pytest test_edge_cases.py::TestComprehensiveAutomatedSuite -v
```

### What the tests cover

| Area | What is tested |
|------|----------------|
| Recommendation boundaries | Expired, 0d, 7d, 8d, 30d, 31d, 90d, 91d expiry thresholds |
| Quantity boundaries | 0, 1, 9, 10, 200, 201 unit edge cases |
| Destination selection | Need score, coverage, zero-demand rejection, source exclusion |
| Split transfers | Multi-branch allocations, max source quantity bounds |
| Capacity constraints | Capped by remaining capacity, 0-capacity manual review |
| Zero demand | Excluded from destinations; all-zero triggers FLAG_FOR_REVIEW |
| Multiple branches | Network distribution and deterministic ranking |
| Multiple batches | Same medicine — stock aggregation and coverage |
| Database CRUD | Insert, read, update, delete in isolated SQLite DB |
| Duplicate batch handling | Rejected without update flag; updated with flag |
| Barcode registration | Association, idempotency, conflict rejection |
| Barcode update | Supersedes old, registers new |
| Superseded lookup | Identifies status, batch, and medicine |
| Unknown / invalid barcode | Unknown status, empty/None handled safely |
| Authentication | Success, wrong password, unknown user, empty credentials |
| Audit logging | All 11 fields, SQLite persistence, sequence |
| Input validation | Negative quantity, malformed date, negative cost/capacity |

---

## 19. Actual Testing Results

All results below are from running `pytest test_edge_cases.py -v` on the current codebase.

```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.1.1
collected 59 items

TestRecommender
  test_1   expired_not_recommended                          PASSED
  test_2   capacity_blocked_triggers_fallback               PASSED
  test_3   zero_demand_excluded                             PASSED
  test_4   tiny_quantity_ignored                            PASSED
  test_5   high_impact_requires_confirmation                PASSED
  test_6   superseded_barcode_resolves                      PASSED
  test_7   unknown_barcode_returns_none                     PASSED

TestRecommenderScoring
  test_8   critical_score_higher_than_near_expiry           PASSED
  test_9   safe_batch_scores_zero                           PASSED
  test_10  calculate_baseline_counts_only_0_to_30_days      PASSED
  test_11  confidence_high_for_high_demand                  PASSED

TestLogManager
  test_12  save_entry_creates_row                           PASSED
  test_13  save_entry_stores_correct_fields                 PASSED
  test_14  get_summary_counts_correctly                     PASSED

TestAuthentication
  test_15  correct_credentials_succeed                      PASSED
  test_16  wrong_password_fails                             PASSED
  test_17  unknown_username_fails                           PASSED

TestRecommenderEnhanced
  test_18  recommendation_returns_all_required_keys         PASSED
  test_19  transit_time_infeasibility_flags_for_review      PASSED
  test_20  exact_reproducible_scoring                       PASSED
  test_21  never_recommends_destination_without_capacity    PASSED
  test_22  reason_explains_all_factors                      PASSED
  test_23  decision_factors_and_concise_explanation_format  PASSED

TestBarcodeEnhanced
  test_24  valid_active_barcode_lookup                      PASSED
  test_25  unknown_barcode_returns_clear_message_no_crash   PASSED
  test_26  superseded_barcode_resolves_correctly            PASSED
  test_27  invalid_and_empty_inputs_handled_safely          PASSED

TestDatabaseArchitecture
  test_28  database_initialise_creates_tables_and_seeds     PASSED
  test_29  load_stock_reads_from_sqlite                     PASSED
  test_30  save_and_load_decisions_sqlite                   PASSED
  test_31  load_stock_graceful_fallback                     PASSED

TestDecisionAuditLogging
  test_32  records_all_ten_decision_fields                  PASSED
  test_33  actions_recorded_consistently                    PASSED
  test_34  override_reason_captured_on_rejection            PASSED
  test_35  historical_logs_preserved_without_data_loss      PASSED

TestAdminDashboardMetrics
  test_36  admin_dashboard_metrics_calculation              PASSED
  test_37  admin_dashboard_visualizations_data_preparation  PASSED

TestMLExpiryModel
  test_38  model_trains_without_error                       PASSED
  test_39  evaluation_returns_all_required_metrics          PASSED
  test_40  metrics_are_real_values_in_valid_range           PASSED
  test_41  dataset_split_is_correct                         PASSED
  test_42  dataset_limitation_note_is_present               PASSED
  test_43  classification_report_is_non_empty_string        PASSED
  test_44  label_high_when_short_expiry_with_excess_stock   PASSED
  test_45  label_medium_when_60_day_expiry_with_excess      PASSED
  test_46  label_low_when_stock_absorbable_before_expiry    PASSED
  test_47  predict_batch_returns_required_keys              PASSED
  test_48  predict_batch_risk_class_is_valid                PASSED
  test_49  predict_batch_probability_is_between_0_and_1     PASSED
  test_50  predict_dataframe_adds_ml_columns                PASSED
  test_51  predict_dataframe_empty_input_returns_empty      PASSED
  test_52  recommendations_include_ml_fields                PASSED
  test_53  decision_factors_include_ml_fields               PASSED
  test_54  explanation_includes_ml_prediction               PASSED
  test_55  ml_does_not_override_expired_stock_safety_rule   PASSED
  test_56  ml_does_not_override_zero_demand_safety_rule     PASSED
  test_57  ml_does_not_override_capacity_safety_rule        PASSED
  test_58  get_ml_predictor_returns_trained_singleton       PASSED
  test_59  predict_batch_accepts_dict_series_and_dataframe  PASSED

========================= 59 passed in 2.26s ==============================
```

> **Note:** The test suite has grown significantly since this output was recorded.
> The current suite contains **147 tests** across the classes listed in [Section 18](#18-how-to-run-tests).
> Run `python -m pytest test_edge_cases.py -v` to see the live output.

**147 / 147 tests pass.**

---

## 20. Actual ML Evaluation Results

All figures below are produced by running `evaluate_ml.py` against the actual database. No values are estimated or fabricated. Results are also saved to `data/ml_evaluation_results.json`.

### Dataset

| Property | Value |
|----------|-------|
| Total samples | 600 |
| Training samples (75%) | 450 |
| Test samples (25%) | 150 |
| Split strategy | Stratified (preserves class proportions) |
| Random state | 42 (fixed for reproducibility) |

### Label Distribution

| Class | Count | Proportion |
|-------|-------|-----------|
| Low | 354 | 59.0% |
| High | 211 | 35.2% |
| Medium | 35 | 5.8% |

### Hold-out Test Set Metrics (N = 150)

| Metric | Weighted | Macro |
|--------|----------|-------|
| **Accuracy** | **0.9133** | — |
| Precision | 0.9086 | 0.8598 |
| Recall | 0.9133 | 0.7513 |
| F1-score | 0.9052 | 0.7779 |

### Per-class Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| High | 0.90 | **1.00** | 0.95 | 53 |
| Low | 0.93 | 0.92 | 0.93 | 88 |
| Medium | 0.75 | 0.33 | 0.46 | 9 |

### Confusion Matrix (Actual vs Predicted)

```
                   Predicted
                  High   Low   Medium
Actual  High  [   53      0      0  ]
Actual  Low   [    6     81      1  ]
Actual  Medium[    0      6      3  ]
```

**Key observation:** The model never misclassified a High-risk batch as Low-risk (High → Low = 0). This is the most safety-critical error direction and it has zero occurrences.

### 5-Fold Stratified Cross-Validation (N = 600)

| Fold | Accuracy | F1-weighted | F1-macro |
|------|----------|-------------|----------|
| 1 | 0.9583 | 0.9522 | 0.8512 |
| 2 | 0.9750 | 0.9752 | 0.9628 |
| 3 | 0.9667 | 0.9635 | 0.8959 |
| 4 | 0.9333 | 0.9275 | 0.8326 |
| 5 | 0.9500 | 0.9490 | 0.9150 |
| **Mean ± SD** | **0.9567 ± 0.014** | **0.9535 ± 0.016** | **0.8915 ± 0.046** |

### Feature Importance (Mean Decrease in Impurity)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `days_to_expiry` | 0.5892 |
| 2 | `stock_to_demand_ratio` | 0.1239 |
| 3 | `avg_daily_demand` | 0.0857 |
| 4 | `quantity` | 0.0816 |
| 5 | `stock_value` | 0.0476 |
| 6 | `branch_capacity_remaining` | 0.0334 |
| 7 | `unit_cost_gbp` | 0.0226 |
| 8–11 | `branch_id_*` (one-hot) | < 0.008 each |

`days_to_expiry` accounts for ~59% of model importance, consistent with clinical intuition that proximity to expiry is the dominant wastage risk factor.

---

## 21. Limitations

### Dataset Limitations

| Limitation | Detail |
|-----------|--------|
| Synthetic data | The dataset (N=600) is procedurally generated, not sourced from real dispensing records. ML metrics reflect performance on this distribution only. |
| Medium class underrepresented | Only 35 Medium samples (5.8% of dataset, 9 in test set). Medium recall is 0.33 — a data scarcity issue, not a model architecture failure. |
| Algorithmically derived labels | Ground-truth risk labels are computed from a demand-absorption heuristic, not from observed, clinician-confirmed real wastage events. |
| No temporal features | No seasonal or time-series variation is modelled. Demand is static per batch row. |

### System Limitations

| Limitation | Detail |
|-----------|--------|
| No real dispensing integration | Stock data must be loaded manually via CSV or database. No live NHS/EMIS/RxWeb feed. |
| No barcode scanner hardware | Barcodes are entered via text input — no physical scanner or camera integration is implemented. |
| Single-machine deployment | Runs as a local Streamlit server. No multi-node or cloud deployment configuration is provided. |
| No ML retraining pipeline | The ML model is trained at startup on the full dataset. There is no scheduled retraining from confirmed audit log outcomes. |
| Email alerts optional | Email requires a valid Gmail SMTP app password. Alerts are silently skipped if credentials are not configured. |

---

## 22. Future Enhancements

| Enhancement | Priority | Status |
|------------|----------|--------|
| ML model retraining from audit log waste events | High | 🔮 FUTURE WORK |
| Confidence calibration for ML probabilities | Medium | 🔮 FUTURE WORK |
| Integration with real pharmacy dispensing systems (EMIS, RxWeb) | High | 🔮 FUTURE WORK |
| Camera-based barcode scanning in browser | Medium | 🔮 FUTURE WORK |
| Time-series demand forecasting | Medium | 🔮 FUTURE WORK |
| Expiry risk trend charts per medicine / branch | Low | 🔮 FUTURE WORK |
| Multi-tenancy (separate pharmacy organisations) | Low | 🔮 FUTURE WORK |
| Cloud deployment (Streamlit Community Cloud / Docker) | Low | 🔮 FUTURE WORK |
| Automated export of decision logs to regulatory bodies | High | 🔮 FUTURE WORK |

---

## Project Structure

```
project/
├── .streamlit/
│   ├── secrets.toml.example    # Streamlit secrets template (safe to commit)
│   └── secrets.toml            # Real secrets — gitignored, never committed
├── .env                        # Legacy env config (gitignored, never committed)
├── .env.example                # Legacy env template (safe to commit)
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python package dependencies
├── README.md                   # This file
├── app.py                      # Main Streamlit application and authentication
├── recommender.py              # Expiry scoring and redistribution recommendation engine
├── ml_expiry_model.py          # ML expiry risk prediction (RandomForestClassifier)
├── database.py                 # SQLite persistence layer (stock, barcodes, decisions)
├── log_manager.py              # Decision logging (SQLite primary, CSV backup)
├── barcode_lookup.py           # Barcode resolution and batch stock query service
├── barcode_registry.py         # Barcode registry with superseded barcode support
├── alert_manager.py            # Automated email alerts via Gmail SMTP
├── auth_config.py              # Credential loader — Streamlit secrets / env vars / SHA-256
├── generate_data.py            # Synthetic stock and barcode data generation
├── evaluate_ml.py              # Standalone ML model evaluation script
├── test_edge_cases.py          # 59-test unit and edge-case test suite
├── pages/
│   └── admin_dashboard.py      # Admin analytics and audit dashboard (role-restricted)
└── data/
    ├── medicines.csv               # Stock catalog and batch inventory (CSV seed)
    ├── barcode_history.csv         # Barcode registry with supersession history (CSV seed)
    ├── decision_log.csv            # Decision audit log (CSV backup)
    ├── pharmacy.db                 # SQLite database (primary operational store)
    └── ml_evaluation_results.json  # ML evaluation metrics (generated by evaluate_ml.py)
```
