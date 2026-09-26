# Pharmacy Expiry Stock Checker and Redistribution Recommender

An intelligent clinical pharmacy decision-support system that proactively identifies near-expiry medicines across pharmacy branches and recommends stock redistributions to reduce clinical waste and help prevent localized medicine shortages.

The system combines a **deterministic, explainable rule-based recommendation engine** with a **supporting machine learning expiry risk classification layer** (Random Forest). All clinical allocations enforce strict safety bounds—including destination demand, storage capacity, shelf-life, and transit feasibility. Pharmacists retain full decision authority with mandatory justification for clinical overrides, while a transactional audit log records decisions in SQLite and CSV.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Objectives](#2-objectives)
3. [Existing System Limitations vs Proposed System](#3-existing-system-limitations-vs-proposed-system)
4. [Key Features](#4-key-features)
5. [System Architecture](#5-system-architecture)
6. [Technology Stack](#6-technology-stack)
7. [Recommendation Methodology & Hard Safety Constraints](#7-recommendation-methodology--hard-safety-constraints)
8. [Database Design & Reliability](#8-database-design--reliability)
9. [Barcode Management](#9-barcode-management)
10. [Authentication & Security](#10-authentication--security)
11. [Audit Logging & Governance](#11-audit-logging--governance)
12. [Admin Dashboard](#12-admin-dashboard)
13. [AI / ML Component](#13-ai--ml-component)
14. [Dataset Description & Clinical Disclaimer](#14-dataset-description--clinical-disclaimer)
15. [Installation & Configuration](#15-installation--configuration)
16. [Running the Application](#16-running-the-application)
17. [Testing & Verification (Actual Testing Results)](#17-testing--verification-actual-testing-results)
18. [Machine Learning Evaluation](#18-machine-learning-evaluation)
19. [Limitations](#19-limitations)
20. [Future Enhancements](#20-future-enhancements)
21. [Project Structure](#21-project-structure)

---

## ⚡ Quick Start

Get the application running locally with the following steps:

```bash
# 1. Clone the repository and enter the directory
git clone https://github.com/JOEL-JERRY-DANISH/Pharmacy-Expiry-Stock-Checker-and-Redistribution-Recommender.git
cd Pharmacy-Expiry-Stock-Checker-and-Redistribution-Recommender

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Configure local environment variables (or copy example)
copy .env.example .env       # Windows
# cp .env.example .env       # macOS / Linux

# 5. Launch the Streamlit web application
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.
- **Demo Pharmacist Users:** `pharmacist1` (Central Pharmacy), `pharmacist2` (North Branch)
- **Demo Administrator:** `admin` (Full network visibility & admin analytics)
- *Passwords are configured locally through `.env` or `.streamlit/secrets.toml` and should never be committed to the repository.*

---

## 1. Problem Statement

Community pharmacies managing complex dispensing workflows may hold medicines that approach expiry before local demand can absorb them. Traditional stock management relies on periodic physical shelf checks and spreadsheet records. By the time near-expiry batches are discovered:

- **Direct Financial Waste:** Expired medicines generally require disposal through applicable pharmaceutical or clinical waste procedures, creating financial and operational costs for the pharmacy network.
- **Patient Supply Disruption:** While one branch disposes of excess near-expiry stock, a neighboring branch often experiences shortages of the exact same medicine.
- **Transit Feasibility Window Lost:** Medicines discovered with minimal shelf life cannot safely be packaged, shipped, received, and dispensed before expiration.
- **Absence of Governance & Audit Trails:** Manual ad-hoc transfers lack structured documentation detailing who authorized a redistribution, why an alternative branch was selected, or why a recommended transfer was rejected.

There is a need for an automated decision-support system that analyzes branch inventory, calculates absorption capacity across the network, generates explainable redistribution recommendations, and maintains a transactional audit trail.

---

## 2. Objectives

| # | Objective | Implementation Status |
|---|-----------|-----------------------|
| 1 | Identify critical (≤7d), near-expiry (8–30d), and watch (31–90d) batches | ✅ COMPLETED |
| 2 | Calculate deterministic, explainable risk scores (0–150 points) based on urgency, volume, and cost | ✅ COMPLETED |
| 3 | Recommend redistribution routes to branches with verified dispensing demand and storage capacity | ✅ COMPLETED |
| 4 | Support split allocations across multiple receiving branches without exceeding available stock | ✅ COMPLETED |
| 5 | Enforce hard clinical safety constraints (exclude expired stock, zero-demand, and capacity-deficient branches) | ✅ COMPLETED |
| 6 | Provide human-readable, multi-factor clinical explanations for every recommendation | ✅ COMPLETED |
| 7 | Enforce explicit confirmation workflows for high-impact redistributions | ✅ COMPLETED |
| 8 | Capture pharmacist clinical decisions, overrides, and mandatory rejection reasons | ✅ COMPLETED |
| 9 | Persist all decisions transactionally in SQLite with full rollback safety and synchronized CSV export | ✅ COMPLETED |
| 10 | Prevent silent fallback to stale CSV data when operational database errors occur | ✅ COMPLETED |
| 11 | Resolve physical GTIN barcodes to batch records, including historical tracking of superseded barcodes | ✅ COMPLETED |
| 12 | Secure user authentication using salted PBKDF2-HMAC-SHA256 with legacy migration support | ✅ COMPLETED |
| 13 | Provide role-restricted administrative analytics with current network health and financial exposure metrics | ✅ COMPLETED |
| 14 | Integrate a supporting Random Forest ML model to predict expiry risk without overriding safety rules | ✅ COMPLETED |
| 15 | Verify implemented functionality through comprehensive automated tests (233 passing tests) | ✅ COMPLETED |

---

## 3. Existing System Limitations vs Proposed System

| Dimension | Legacy Manual Process | Proposed Intelligent Recommender |
|-----------|------------------------|----------------------------------|
| **Detection Timing** | Reactive during monthly/quarterly stock counts; often too late for transfer. | Automated during inventory analysis; evaluates shelf life against branch dispensing rates. |
| **Destination Matching** | Informal telephone calls or spreadsheet guessing without capacity visibility. | Deterministic matching based on remaining shelf life, weekly demand, and available storage. |
| **Allocation Logic** | Single-branch subjective guesswork. | Multi-branch split allocation strictly bounded by source quantity, branch demand, and capacity. |
| **Safety Enforcement** | Human-error prone; expired stock may inadvertently be transferred. | Hard rule enforcement; expired stock, zero-demand, and zero-capacity destinations are strictly blocked. |
| **Machine Learning** | None. | Random Forest classifier providing supplementary expiry risk probabilities with safe rule-based fallback. |
| **Barcode Integration** | Manual lookup or separate standalone POS lookup. | Native barcode resolution tracking active and superseded barcodes (repackaging, serialization). |
| **Audit Logging** | Paper transfer slips or untracked verbal agreements. | 11-field transactional SQLite audit log + CSV export with mandatory override justification. |
| **Database Reliability** | Vulnerable to stale data or unhandled corruption. | Transactional ACID updates; database failures are surfaced immediately rather than serving stale CSVs. |
| **Security** | Shared passwords or unhashed local text files. | PBKDF2-HMAC-SHA256 password hashing (100,000 rounds, 128-bit salt, `hmac.compare_digest`). |

---

## 4. Key Features

- **Explainable 0–150 Point Risk Scoring:** Transparent composite scoring factoring urgency (0–100 pts), batch quantity weighting (0–30 pts), and unit financial cost weighting (0–20 pts).
- **Proportional Redistribution Engine:** Allocates stock to up to 3 viable destination branches ranked by need, strictly respecting source inventory limits, destination capacity, and shelf-life absorption constraints.
- **Hard Clinical Safety Constraints:** Hard-coded algorithmic guards that ML cannot bypass:
  - Expired batches (`days_to_expiry < 0`) are never transferred.
  - Branches with zero dispensing demand (`demand_per_week == 0`) are excluded.
  - Branches with zero storage capacity (`branch_capacity_remaining == 0`) are excluded.
  - Transfers exceeding destination capacity or expected demand before expiry are prevented.
  - Infeasible transfers where estimated transit days exceed shelf life are automatically flagged for local review (`FLAG_FOR_REVIEW`).
- **Pharmacist Clinical Governance:** Pharmacists can confirm, override (with mandatory clinical justification reason), or flag recommendations. High-impact recommendations require explicit confirmation.
- **Machine Learning Advisory Layer:** Pre-trained Random Forest classifier predicting risk class (`Low`, `Medium`, `High`) and expiry probability. If ML predictions are unavailable or fail, the system falls back gracefully to pure rule-based execution without crashing or mislabeling batches.
- **Barcode Registry & Supersession:** Resolves barcodes to batch records. Full historical tracking for superseded barcodes (e.g., Falsified Medicines Directive serialization, repackaging, batch updates).
- **Hardened PBKDF2-HMAC-SHA256 Authentication:** 100,000 iterations of SHA-256 with a 128-bit cryptographically secure salt, constant-time `hmac.compare_digest` verification, automatic in-memory upgrade of legacy SHA-256 hashes, and complete elimination of plaintext password caching.
- **Transactional Persistence & SQLite Safety:** Atomic database writes with rollback protection during decision saving. If SQLite encounters an error, it is surfaced to the user; the system strictly avoids falling back to stale seed CSVs.
- **Role-Based Admin Analytics:** Real-time visibility into network stock value at risk, expiry timelines, destination transfer distribution, and clinical override logs.

---

## 5. System Architecture

```
                  ┌────────────────────────────────────────┐
                  │       Seed Inventory CSV Files         │
                  │   (Initial First-Time Setup Only)      │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    SQLite Database (data/pharmacy.db)  │ ◄── Primary Operational Store
                  │  - stock | barcodes | decisions tables │     (ACID Compliant)
                  └───────────────────┬────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
┌───────────────────────────────┐           ┌───────────────────────────────────┐
│ Machine Learning Risk Model   │           │ Rule-Based Recommendation Engine  │
│ (ml_expiry_model.py)          │           │ (recommender.py)                  │
│ • Random Forest Classifier    │           │ • 0–150 Point Risk Scoring        │
│ • Predicts: Low / Med / High  │           │ • Need & Capacity Calculations    │
│ • Supporting advisory signal  │           │ • Multi-branch Split Allocation   │
│ • Graceful fallback on error  │           │ • Hard Clinical Safety Rules      │
└─────────────┬─────────────────┘           └─────────────────┬─────────────────┘
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │     Composite Decision Generator       │
                  │ • Action: TRANSFER or FLAG_FOR_REVIEW  │
                  │ • Structured Factors & Explanation     │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │    Streamlit Web Interface (app.py)    │
                  │ • Pharmacist Review / Confirm / Override│
                  │ • High-Impact Confirmation Guard       │
                  │ • Barcode Scanning & Resolution        │
                  │ • Role-Restricted Admin Dashboard      │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │     Transactional Audit Logging        │
                  │ • Atomic SQLite Insert with Rollback   │
                  │ • Synchronized data/decision_log.csv   │
                  └────────────────────────────────────────┘
```

> **Safety Architecture Rule:** Rule-based safety constraints take precedence over machine learning outputs. Machine learning outputs serve strictly as an advisory signal and cannot override the defined safety boundaries.

---

## 6. Technology Stack

| Layer | Technology | Version / Standard | Architectural Role |
|-------|------------|--------------------|--------------------|
| **UI Framework** | [Streamlit](https://streamlit.io/) | Modern Web App | Multi-page clinical interface, pharmacist workflow, and admin analytics |
| **Language** | [Python](https://www.python.org/) | 3.10+ | Core application logic and algorithmic pipeline |
| **Data Processing** | [pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) | High-performance | Tabular inventory manipulation, vector math, and feature extraction |
| **Machine Learning** | [scikit-learn](https://scikit-learn.org/) | Pipelines & Models | `RandomForestClassifier`, `StandardScaler`, `OneHotEncoder`, evaluation |
| **Database** | [SQLite 3](https://www.sqlite.org/) | Python built-in | Primary operational relational database with atomic transactions |
| **Authentication** | [hashlib](https://docs.python.org/3/library/hashlib.html) & [secrets](https://docs.python.org/3/library/secrets.html) | Python standard library | PBKDF2-HMAC-SHA256 (100k rounds, 128-bit salt, `hmac.compare_digest`) |
| **Alerting** | [smtplib](https://docs.python.org/3/library/smtplib.html) | Python standard library | Automated SMTP notifications for critical near-expiry batches |
| **Configuration** | [python-dotenv](https://pypi.org/project/python-dotenv/) | Standards-compliant | Environment variable configuration and Streamlit secrets management |
| **Testing** | [pytest](https://pytest.org/) | Automated Suite | 233 deterministic unit, boundary, integration, and security tests |

---

## 7. Recommendation Methodology & Hard Safety Constraints

### 7.1 Batch Risk Scoring (0–150 Points)

Every batch is evaluated using a deterministic, explainable scoring formula:

$$\text{Score} = \text{Urgency Points} + \text{Quantity Weight} + \text{Financial Value Weight}$$

| Component | Condition / Calculation | Points |
|-----------|-------------------------|--------|
| **Urgency: Critical** | $\text{Days to Expiry} \in [0, 7]$ | **100.0** |
| **Urgency: Near-Expiry** | $\text{Days to Expiry} \in [8, 30]$ | **50.0** |
| **Urgency: Watch** | $\text{Days to Expiry} \in [31, 90]$ | **10.0** |
| **Urgency: Safe / Expired** | $\text{Days to Expiry} > 90$ or $< 0$ | **0.0** |
| **Quantity Weight** | $\min\left(\frac{\text{quantity}}{500}, 1.0\right) \times 30.0$ | **0.0 – 30.0** |
| **Financial Value Weight** | $\min\left(\frac{\text{unit\_cost\_gbp}}{5.0}, 1.0\right) \times 20.0$ | **0.0 – 20.0** |

### 7.2 Destination Selection & Need Calculation

For each actionable batch, the engine identifies potential receiving branches carrying the same medicine and calculates **Destination Need** based on remaining shelf life:

$$\text{Expected Demand Before Expiry} = \left(\frac{\text{demand\_per\_week}}{7}\right) \times \text{Remaining Shelf Life (Days)}$$

$$\text{Destination Need} = \max\left(0, \lfloor\text{Expected Demand}\rfloor - \text{Destination Current Stock}\right)$$

Branches with higher calculated need scores are prioritized. Up to 3 destinations can receive split allocations.

### 7.3 Multi-Branch Split Allocation Constraints

When allocating stock from a source batch across destination branches:

1. **Individual Allocation Cap:** An individual transfer to branch $i$ cannot exceed the available stock, destination need, or destination capacity:
   $$\text{Transfer}_i \le \min(\text{Source Quantity Remaining}, \text{Destination Need}_i, \text{Destination Capacity}_i)$$
2. **Total Allocation Cap:** The sum of all allocated transfers across all destinations cannot exceed the source batch quantity:
   $$\sum_{i} \text{Transfer}_i \le \text{Source Quantity}$$
3. **Absorption Percentage:** Calculated per destination to quantify clinical utility:
   $$\text{Absorption Pct} = \begin{cases} 0.0\% & \text{if } \text{Transfer}_i \le 0 \\ \min\left(100.0, \frac{\text{Expected Demand}_i}{\text{Transfer}_i} \times 100.0\right) & \text{if } \text{Transfer}_i > 0 \end{cases}$$

### 7.4 Hard Safety Rules (Enforced by the Implementation)

The recommendation engine strictly enforces the following non-negotiable boundaries:

1. **No Expired Stock Redistribution:** Batches with `days_to_expiry < 0` are immediately excluded from transfer recommendations.
2. **Zero-Demand Destination Rejection:** Branches with `demand_per_week == 0` never receive stock.
3. **Zero-Capacity Destination Rejection:** Branches with `branch_capacity_remaining == 0` never receive stock.
4. **Transit Infeasibility Quarantine:** If estimated transit time meets or exceeds remaining shelf life ($\text{Transit Days} \ge \text{Shelf Life}$), the batch is marked as infeasible for transfer and assigned `FLAG_FOR_REVIEW` for local expedited dispensing or safe disposal.
5. **No Negative Quantities:** All inventory quantities, demands, costs, and capacities are validated; negative values are rejected or sanitized safely.

---

## 8. Database Design & Reliability

### 8.1 SQLite Schema (`data/pharmacy.db`)

SQLite serves as the primary operational store, ensuring ACID compliance and transaction isolation.

#### `stock` Table
```sql
CREATE TABLE stock (
    batch_id TEXT PRIMARY KEY,
    medicine_name TEXT NOT NULL,
    category TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    branch_name TEXT NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity >= 0),
    expiry_date TEXT NOT NULL,
    unit_cost_gbp REAL NOT NULL CHECK (unit_cost_gbp >= 0),
    demand_per_week INTEGER NOT NULL CHECK (demand_per_week >= 0),
    branch_capacity_remaining INTEGER NOT NULL CHECK (branch_capacity_remaining >= 0)
);
```

#### `barcodes` Table
```sql
CREATE TABLE barcodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    barcode TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    medicine_name TEXT NOT NULL,
    registered_date TEXT NOT NULL,
    superseded_date TEXT,
    reason_for_change TEXT
);
```

#### `decisions` Table
```sql
CREATE TABLE decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    user TEXT NOT NULL,
    medicine TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    source_branch TEXT NOT NULL,
    destination TEXT,
    quantity INTEGER NOT NULL,
    system_recommendation TEXT NOT NULL,
    action TEXT NOT NULL,
    final_decision TEXT NOT NULL,
    override_reason TEXT
);
```

### 8.2 Database Reliability & Error Handling

- **No Silent Fallback on Database Failure:**
  - **First-Time Setup (Case A):** When the database does not exist or has not been initialized, `initialise_database(db_path, seed=True)` legitimately populates the schema and initial operational inventory from the seed CSV files.
  - **Database Error / Corruption (Case B):** If the SQLite database exists but encounters a failure (file lock, disk I/O failure, schema error, or corruption), `load_stock()` raises `DatabaseLoadError` and records the diagnostic error. The application **does NOT silently fall back to stale CSV data**, preventing clinical decisions from being made on outdated stock. Instead, an explicit error notification is surfaced in the UI.
- **Transactional Decision Saving:**
  `save_decision()` executes all database modifications inside an atomic SQLite transaction (`with conn:`). Both the decision record and audit log are committed together. If an error occurs during audit persistence, the transaction rolls back completely to prevent inconsistent or orphaned records, and the error is returned or raised to inform the caller.

---

## 9. Barcode Management

Implemented across [`barcode_registry.py`](barcode_registry.py) and [`barcode_lookup.py`](barcode_lookup.py):

- **Active Barcode Resolution:** Scanning or entering an active GTIN barcode instantly retrieves medicine details, batch ID, current branch stock, expiry date, urgency scoring, and real-time redistribution recommendations.
- **Superseded Barcode Lifecycle:** Pharmaceutical supply chains frequently re-label batches due to repackaging, regulatory serialization (EU Falsified Medicines Directive), or manufacturer updates. When a barcode is updated:
  - The previous barcode is preserved with a `superseded_date` timestamp and `reason_for_change`.
  - A lookup of the superseded barcode still successfully identifies the correct batch and medicine, prominently notifying the pharmacist that the barcode has been superseded.
- **Defensive Error Handling:** Malformed, blank, or unrecognized barcodes return clear status dictionaries (`{"found": False, "status": "not_found" | "invalid"}`) without raising exceptions or interrupting application flow.

---

## 10. Authentication & Security

Implemented in [`auth_config.py`](auth_config.py) and [`app.py`](app.py):

### 10.1 PBKDF2-HMAC-SHA256 Password Hashing

The authentication architecture uses key derivation via PBKDF2-HMAC-SHA256:

- **100,000 Iterations:** Password derivation uses 100,000 PBKDF2-HMAC-SHA256 iterations.
- **128-Bit Cryptographically Secure Salt:** Each password hash utilizes 16 bytes (32 hex characters) of cryptographically strong random salt generated via `secrets.token_hex(16)`.
- **Constant-Time Verification:** Hash comparison is performed using `hmac.compare_digest()` to eliminate timing-attack vulnerabilities.
- **Standardized Hash Serialization:** Stored in the standard modular format:
  `pbkdf2:sha256:100000$<salt_hex>$<derived_key_hex>`

### 10.2 Transparent Legacy SHA-256 Migration

Existing installations or legacy test credentials with standard 64-character SHA-256 hex digests authenticate seamlessly:
1. When a user logs in, `verify_password()` detects whether the stored credential is a legacy 64-character SHA-256 hash or a modern PBKDF2 string.
2. If legacy SHA-256 verification succeeds, `needs_rehash()` triggers an automatic in-memory upgrade to PBKDF2-HMAC-SHA256 (`_MIGRATED_HASHES`), without requiring manual database migration or downtime.

### 10.3 Plaintext Password Cache Elimination

In accordance with strict security standards, `_PLAINTEXT_HASH_CACHE` has been completely removed. Environment variables containing plaintext passwords (used in test/local setups) are hashed on demand via PBKDF2 and are never stored or cached in plaintext in memory.

### 10.4 Credential Resolution Hierarchy

`auth_config.get_credentials()` checks credentials in priority order:
1. **Streamlit Secrets** (`.streamlit/secrets.toml` under `[auth]`)
2. **Environment Variable Hashes** (`*_PASSWORD_HASH`)
3. **Environment Variable Plaintext** (`*_PASSWORD`, hashed on demand)

### 10.5 Role-Based Access Control (RBAC) — Demo Accounts

The names below are fictional demonstration identities.


| Username | Name | Branch Assignment | Permitted Interfaces |
|----------|------|-------------------|----------------------|
| `pharmacist1` | Sarah Johnson | Central Pharmacy | Pharmacist review, transfers, barcode lookup |
| `pharmacist2` | James Patel | North Branch | Pharmacist review, transfers, barcode lookup |
| `admin` | Admin User | All branches | Pharmacist workflow + Role-restricted Admin Dashboard |

---

## 11. Audit Logging & Governance

Implemented in [`log_manager.py`](log_manager.py) and [`database.py`](database.py):

Every recorded clinical decision creates an 11-field audit record:

```
[Timestamp] [User] [Medicine] [Batch ID] [Source Branch] [Destination Branch]
[Quantity] [System Recommendation] [Action] [Final Decision] [Override Reason]
```

- **Mandatory Clinical Justification:** When a pharmacist overrides a system recommendation (`OVERRIDDEN`), the UI requires a clinical justification reason, which is committed to the audit log.
- **Dual Persistence:** Decisions are written transactionally to the SQLite `decisions` table and synchronized to `data/decision_log.csv` for inspection and structured export.
- **Credential Protection:** Secrets, passwords, session tokens, and password hashes are strictly excluded from logs, error messages, and audit tables.

---

## 12. Admin Dashboard

Accessible at `pages/admin_dashboard.py` exclusively to users with `branch = "All branches"`.

### Operational Metrics (7 Real-Time Data Points)
1. **Total Decisions Logged:** Total records in the audit trail.
2. **Confirmed Transfers:** Volume of accepted system recommendations.
3. **Overrides Recorded:** Count of pharmacist-rejected recommendations.
4. **Manual Reviews:** Batches flagged for local action.
5. **Medicines at Expiry Risk:** Count of batches within the 0–30 day critical/near-expiry window.
6. **Recommended Transfer Units:** Total medicine units proposed for redistribution.
7. **Stock Value at Risk (£):** Cumulative GBP value of inventory expiring within 30 days.

### Visualizations (5 Dynamic Charts)
- **Decisions Over Time:** Trend line of clinical review activity.
- **Action Breakdown:** Proportion of confirmed vs. overridden vs. reviewed decisions.
- **Redistribution Destinations:** Volume of stock received per branch.
- **Network Expiry Risk Distribution:** Proportions of Critical, Near-Expiry, Watch, Safe, and Expired batches.
- **Financial Risk by Branch:** Cumulative value (£) of at-risk inventory categorized by branch.

---

## 13. AI / ML Component

Implemented in [`ml_expiry_model.py`](ml_expiry_model.py):

### 13.1 Purpose & Role

The ML model provides a **supplementary advisory signal** estimating an expiry-risk probability based on the inventory features used by the model. It evaluates multidimensional inventory attributes to output:
- **Predicted Risk Class:** `Low`, `Medium`, or `High`
- **Expiry Risk Probability:** Continuous float from `0.0` to `1.0`

### 13.2 Model Pipeline & Specification

```python
Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", StandardScaler(), [
            "days_to_expiry", "quantity", "avg_daily_demand",
            "stock_to_demand_ratio", "stock_value", "unit_cost_gbp",
            "branch_capacity_remaining"
        ]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["branch_id"])
    ])),
    ("classifier", RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ))
])
```

### 13.3 Safe Failure & Resilience

If the machine learning predictor fails to load, encounters corrupt input, or throws an unhandled error:
- The system catches the error safely and assigns `"ml_risk_class": "Unavailable"` and `"ml_risk_probability": None`.
- The recommendation engine continues executing normally using its deterministic rule-based algorithms.
- **ML failure never crashes the application and never falsely defaults to `Low` risk.**

---

## 14. Dataset Description & Clinical Disclaimer

The dataset consists of **600 synthetic medicine batch records** across 4 branches and 10 representative pharmaceutical products:

| Property | Dataset Value |
|----------|---------------|
| Total Batch Records | 600 rows |
| Participating Branches | 4 (Central Pharmacy, North Branch, East Branch, South Branch) |
| Therapeutic Categories | 6 categories (Cardiovascular, Antibiotics, Analgesics, Respiratory, etc.) |
| Expiry Distribution | ~5% Expired (<0d), ~15% Critical (≤7d), ~20% Near-Expiry (8–30d), ~25% Watch (31–90d), ~35% Safe (>90d) |
| Weekly Demand Range | 2 to 80 units/week |
| Batch Quantity Range | 5 to 500 units |
| Unit Cost Range | £0.08 to £3.50 per unit |

### ⚠️ Synthetic Data & Clinical Disclaimer

> **IMPORTANT:** The dataset included in this repository was procedurally generated using `generate_data.py` (fixed random seed) to simulate community pharmacy inventory dynamics for research and demonstration purposes.
>
> Ground-truth labels are derived algorithmically from mathematical demand-absorption heuristics, not from clinician-annotated historical disposal logs. The machine learning metrics reported below demonstrate algorithmic learning and validation on this synthetic distribution; **they do not establish real-world clinical performance or regulatory approval**. Production deployment in a healthcare setting requires validation against multi-year real-world dispensing data.

---

## 15. Installation & Configuration

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/JOEL-JERRY-DANISH/Pharmacy-Expiry-Stock-Checker-and-Redistribution-Recommender.git
cd Pharmacy-Expiry-Stock-Checker-and-Redistribution-Recommender

# 2. Set up virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration Option A: Streamlit Secrets (Recommended)

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`:

```bash
copy .streamlit\secrets.toml.example .streamlit\secrets.toml   # Windows
# cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # macOS / Linux
```

Generate secure PBKDF2 password hashes using Python:
```bash
python -c "import auth_config; print(auth_config.hash_password('your_chosen_password'))"
```

> **Security:** Never commit `.streamlit/secrets.toml` or `.env` files containing real passwords, API keys, email credentials, or other secrets. Use the provided example files as templates only.

Configure `.streamlit/secrets.toml`:
```toml
[auth]
pharmacist1_password_hash = "pbkdf2:sha256:100000$..."
pharmacist2_password_hash = "pbkdf2:sha256:100000$..."
admin_password_hash       = "pbkdf2:sha256:100000$..."

[email]
sender   = "yourpharmacy@gmail.com"
password = "your_16_char_app_password"
receiver = "pharmacist@pharmacy.com"
```

### Configuration Option B: Environment Variables (`.env`)

Copy `.env.example` to `.env`:
```bash
copy .env.example .env     # Windows
# cp .env.example .env     # macOS / Linux
```

Configure `.env` with passwords (hashed on the fly via PBKDF2) or pre-computed hashes:
```env
PHARMACIST1_PASSWORD=your_password_here
PHARMACIST2_PASSWORD=your_password_here
ADMIN_PASSWORD=your_password_here

# Or use pre-computed PBKDF2 hashes:
# PHARMACIST1_PASSWORD_HASH=pbkdf2:sha256:100000$...

EMAIL_SENDER=yourpharmacy@gmail.com
EMAIL_PASSWORD=your_16_char_app_password
EMAIL_RECEIVER=pharmacist@pharmacy.com
```

---

## 16. Running the Application

### Launch Streamlit Interface
```bash
streamlit run app.py
```
Access the application at [http://localhost:8501](http://localhost:8501).

### (Optional) Regenerate Synthetic Inventory
```bash
python generate_data.py
```
Recreates `data/medicines.csv` and `data/barcode_history.csv` with fresh synthetic data.

### (Optional) Run ML Evaluation
```bash
python evaluate_ml.py
```
Evaluates the Random Forest model on the dataset and writes metrics to `data/ml_evaluation_results.json`.

---

## 17. Testing & Verification (Actual Testing Results)

The repository contains an automated, deterministic test suite in [`test_edge_cases.py`](test_edge_cases.py). The suite has evolved from the initial foundational suite to **233 passing tests** covering functional boundaries, edge cases, and defined safety constraints across the project.

### Run the Full Test Suite

```bash
python -m pytest test_edge_cases.py -q
# or simply:
pytest -q
```

### Verified Test Suite Execution Output (Actual Testing Results)

```
........................................................................ [ 30%]
........................................................................ [ 61%]
........................................................................ [ 92%]
.................                                                        [100%]
233 passed
```

### Test Suite Architecture (31 Test Classes, 233 Tests)

| Test Class | Focus Area | Test Count |
|------------|------------|:----------:|
| `TestRecommender` | Foundational expiry, capacity, and zero-demand exclusion | 7 |
| `TestRecommenderScoring` | Base score differentiation, baseline windowing, and confidence | 4 |
| `TestLogManager` | Decision log file creation, field serialization, summary counts | 3 |
| `TestAuthentication` | PBKDF2 login success, invalid password rejection, unknown user rejection | 3 |
| `TestRecommenderEnhanced` | Transparent keys, transit infeasibility, reproducible scores, capacity check | 6 |
| `TestBarcodeEnhanced` | Active barcode resolution, unknown handling, supersession tracking | 4 |
| `TestDatabaseArchitecture` | SQLite table creation, schema seeding, decision persistence, error isolation | 4 |
| `TestDecisionAuditLogging` | 11 audit fields, consistent action logging, override justification | 4 |
| `TestAdminDashboardMetrics` | Live operational metric aggregation and chart data prep | 2 |
| `TestMLExpiryModel` | Model training, holdout evaluation, feature importance, safe fallback | 22 |
| `TestSQLiteBarcodeRegistry` | Relational barcode CRUD, duplicate rejection, historical query | 7 |
| `TestBarcodeUpdateAtomicity` | Atomic barcode supersession and replacement rollback transactions | 9 |
| `TestSafeDatabaseImport` | Schema-validated CSV import, required columns, duplicate protection | 8 |
| `TestExplainableScoring` | Mathematical accuracy of 0–150 composite scoring formula | 7 |
| `TestDestinationSelection` | Need-score formula, multi-branch ranking, capacity limits | 5 |
| `TestBatchSplitting` | Proportional split allocation across multiple destinations | 8 |
| `TestSQLiteAuditLog` | Relational audit log persistence, ordering, and retrieval | 12 |
| `TestLiveInventoryAndCache` | Cache invalidation, live stock updates, stock deduction | 9 |
| `TestComprehensiveAutomatedSuite` | End-to-end integration across all system services | 32 |
| `TestPhase1DestinationAllocation` | Destination capacity, need limits, zero-demand and split boundaries | 6 |
| `TestPhase2MLFailureHandling` | Safe ML failure recovery, `"Unavailable"` status, no crashes | 5 |
| `TestBarcodeLookupMLFailure` | Safe ML failure handling in barcode lookup, `"Unavailable"` status | 6 |
| `TestPhase3InvalidExpiryHandling` | Malformed date strings, partial dates, safe validation | 10 |
| `TestPhase4InvalidNumericHandling` | Negative quantities, invalid costs, non-numeric strings | 8 |
| `TestPhase5DestinationNeedShelfLife` | Shelf-life-adjusted destination demand calculation | 8 |
| `TestPhase6AbsorptionPercentage` | Zero transfer absorption, partial transfer, over-demand absorption | 5 |
| `TestPhase7HighImpactConfirmationConsistency` | Explicit confirmation state machine for high-impact recommendations | 6 |
| `TestPhase8PasswordSecurity` | PBKDF2 verification, salt randomness, timing resistance, legacy migration | 5 |
| `TestPhase12DatabaseFallback` | Prevention of silent CSV fallback on SQLite database failure | 5 |
| `TestPhase13SaveDecisionAudit` | Transactional rollback and error surfacing on decision save failure | 5 |
| `TestDecisionSaveFailureHandling` | Session state protection and error surfacing on decision save failure | 8 |
| **Total Verified Tests** | **Deterministic, isolated unit and edge-case tests** | **233 Passed** |

---

## 18. Machine Learning Evaluation

All metrics below are generated directly from the hold-out test set ($N = 150$) and 5-fold cross-validation ($N = 600$) using [`evaluate_ml.py`](evaluate_ml.py) and stored in `data/ml_evaluation_results.json`.

### Dataset Split & Distribution
- **Total Samples:** 600 batches
- **Training Set (75%):** 450 batches (Stratified)
- **Hold-Out Test Set (25%):** 150 batches (Stratified)
- **Class Distribution:** Low Risk: 354 (59.0%), High Risk: 211 (35.2%), Medium Risk: 35 (5.8%)

### Hold-Out Test Set Performance ($N = 150$)

| Evaluation Metric | Weighted Average | Macro Average |
|-------------------|:----------------:|:-------------:|
| **Accuracy** | **0.9133** | — |
| **Precision** | 0.9086 | 0.8598 |
| **Recall** | 0.9133 | 0.7513 |
| **F1-Score** | **0.9052** | 0.7779 |

### Per-Class Detailed Classification Report

| Risk Class | Precision | Recall | F1-Score | Support |
|------------|:---------:|:------:|:--------:|:-------:|
| **High Risk** | 0.898 | **1.000** | **0.946** | 53 |
| **Low Risk** | 0.931 | 0.920 | 0.926 | 88 |
| **Medium Risk** | 0.750 | 0.333 | 0.462 | 9 |

> **Critical Safety Finding:** The model achieved **100% recall on High-Risk batches** (53 out of 53 detected). Zero High-risk batches were misclassified as Low-risk. In clinical waste prevention, High $\to$ Low is the most dangerous error mode; achieving zero occurrences validates the model's safety-aligned class weighting.

### Test Set Confusion Matrix

```
                    Predicted Class
                  High   Low   Medium
Actual  High   [   53      0      0   ]   <-- 100% High Risk Recall in This Test Set
Actual  Low    [    6     81      1   ]
Actual  Medium [    0      6      3   ]
```

### 5-Fold Stratified Cross-Validation ($N = 600$)

| Fold | Accuracy | F1-Score (Weighted) | F1-Score (Macro) |
|:----:|:--------:|:-------------------:|:----------------:|
| 1 | 0.9583 | 0.9522 | 0.8512 |
| 2 | 0.9750 | 0.9752 | 0.9628 |
| 3 | 0.9667 | 0.9635 | 0.8959 |
| 4 | 0.9333 | 0.9275 | 0.8326 |
| 5 | 0.9500 | 0.9490 | 0.9150 |
| **Mean ± SD** | **0.9567 ± 0.0143** | **0.9535 ± 0.0159** | **0.8915 ± 0.0464** |

### Feature Importance (Mean Decrease in Impurity)

| Rank | Feature | Importance | Interpretation |
|:----:|---------|:----------:|-------------------------|
| 1 | `days_to_expiry` | **0.5892** | Proximity to expiry had the highest feature importance in this model |
| 2 | `stock_to_demand_ratio` | 0.1239 | Higher stock-to-demand ratios had greater feature importance in this model |
| 3 | `avg_daily_demand` | 0.0857 | Local branch dispensing velocity |
| 4 | `quantity` | 0.0816 | Physical units at risk |
| 5 | `stock_value` | 0.0476 | Financial exposure (£) |
| 6 | `branch_capacity_remaining`| 0.0334 | Physical shelf space available |
| 7 | `unit_cost_gbp` | 0.0226 | Per-unit medication price |
| 8–11 | `branch_id_*` (One-Hot) | < 0.008 each | Minimal branch-specific geographic bias |

---

## 19. Limitations

### Dataset Limitations
1. **Synthetic Data:** The dataset ($N=600$) is procedurally generated. Real-world dispensing logs may exhibit non-linear seasonal shifts, holiday demand spikes, or local demographic variations not represented in this data.
2. **Medium Risk Scarcity:** The Medium Risk class accounts for only 5.8% of the dataset (35 samples total, 9 in the test set), leading to lower recall (0.33) for that specific class due to sample scarcity.
3. **Algorithmic Labels:** Training labels are derived mathematically from demand absorption heuristics rather than historical clinical disposal outcomes.

### System & Infrastructure Limitations
1. **No Direct PMR/EHR Integration:** The system operates standalone and does not currently integrate with live electronic prescribing, pharmacy management, or electronic health-record systems.
2. **Barcode Input:** Barcode resolution is demonstrated via text input. Hardware scanner support or camera-based WebRTC scanning is not natively bundled.
3. **Single-Node Execution:** Designed for local or single-instance Streamlit deployment. Multi-node cloud clustering requires an external PostgreSQL/MySQL database configuration.

---

## 20. Future Enhancements

| Enhancement | Clinical / Architectural Value | Priority |
|-------------|--------------------------------|:--------:|
| **Continuous Model Retraining** | Retrain Random Forest models automatically using confirmed disposal events from the audit log | High |
| **EHR / PMR System API** | Real-time HL7 / FHIR integration with NHS Electronic Prescription Service | High |
| **Time-Series Demand Forecasting** | Incorporate Prophet / ARIMA models for seasonal prescription fluctuations | High |
| **Camera-Based Barcode Scanning** | Integrated in-browser HTML5 barcode scanner using device camera | Medium |
| **Probability Calibration** | Platt scaling / isotonic regression for calibrated confidence scores | Medium |
| **Multi-Tenancy** | Organization-level isolation for regional pharmacy groups and hospital trusts | Medium |
| **Automated Regulatory Reporting** | Configurable export of waste-reduction and audit information for review against applicable regulatory requirements | Low |

---

## 21. Project Structure

```
Pharmacy-Expiry-Stock-Checker-and-Redistribution-Recommender/
├── .streamlit/
│   └── secrets.toml.example        # Template for Streamlit secrets (safe to commit)
├── data/
│   ├── medicines.csv               # Seed catalog and branch inventory data (CSV)
│   ├── barcode_history.csv         # Seed barcode registry with supersession history (CSV)
│   ├── decision_log.csv            # Dual-persisted decision audit log (CSV export)
│   ├── pharmacy.db                 # Primary operational SQLite database (created on first run)
│   └── ml_evaluation_results.json  # Actual ML evaluation metrics generated by evaluate_ml.py
├── pages/
│   └── admin_dashboard.py          # Role-restricted administrative analytics dashboard
├── .env.example                    # Template for environment configuration
├── .gitignore                      # Git exclusion rules (ignores secrets, databases, venvs)
├── alert_manager.py                # Automated SMTP email alert dispatcher for critical stock
├── app.py                          # Main Streamlit web application & clinical interface
├── auth_config.py                  # PBKDF2-HMAC-SHA256 authentication & credential resolver
├── barcode_lookup.py               # Barcode query service resolving barcodes to batch stock
├── barcode_registry.py             # SQLite barcode registry managing active & superseded barcodes
├── constants.py                    # Centralized system constants and threshold definitions
├── database.py                     # Primary persistence layer, schema validation, & transactions
├── evaluate_ml.py                  # Standalone ML model evaluation and validation script
├── generate_data.py                # Deterministic synthetic data generator
├── log_manager.py                  # Clinical decision logging and summary statistics service
├── ml_expiry_model.py              # Scikit-learn Random Forest expiry risk predictor pipeline
├── problem_analysis.md             # Clinical requirements analysis and domain documentation
├── README.md                       # Comprehensive system documentation (this file)
├── recommender.py                  # Deterministic scoring, need calculation, & allocation engine
├── requirements.txt                # Python package dependencies
└── test_edge_cases.py              # Automated test suite (233 deterministic unit/boundary tests)
```
