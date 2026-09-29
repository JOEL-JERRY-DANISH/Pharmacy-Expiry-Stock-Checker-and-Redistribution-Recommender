# Pharmacy Expiry Stock Checker and Redistribution Recommender

A Streamlit-based pharmacy inventory decision-support system for identifying medicines at expiry risk, evaluating redistribution opportunities, and supporting pharmacist-approved stock transfers between branches.

> **Clinical safety notice:** This project is an academic/software engineering decision-support system. It does not replace pharmacist or clinical judgment. All redistribution recommendations require human review and confirmation.

---

## 📌 Project Overview

Pharmacies can lose stock when medicines approach expiry while another branch has stronger demand for the same product. This project provides a safety-first workflow for:

* Detecting medicines approaching expiry.
* Calculating an explainable expiry-risk score.
* Checking inventory and demand across branches.
* Ranking redistribution destinations using a formal multi-criteria objective.
* Supporting split redistribution when one destination cannot absorb the full quantity.
* Recording pharmacist approvals, overrides, and rejections.
* Providing structured reason codes for human overrides.
* Maintaining an auditable SQLite decision history.
* Using barcode lookup and lifecycle management.
* Providing an ML advisory signal for expiry-risk classification.
* Testing boundary conditions and failure scenarios using Pytest.

The system is designed around a **human-in-the-loop** model: the software recommends, while the pharmacist makes the final operational decision.

---

## 🎯 Objectives

| Objective                            | Implementation                                   |
| ------------------------------------ | ------------------------------------------------ |
| Identify expiry risk                 | Explainable 0–150 risk score                     |
| Detect redistribution opportunities  | Safety-first destination eligibility             |
| Formalize destination ranking        | Normalized multi-criteria objective              |
| Balance inventory value and demand   | Demand, protected value, and transit criteria    |
| Prevent unsafe transfers             | Capacity, demand, stock, and transit constraints |
| Support limited destination capacity | Split redistribution                             |
| Capture pharmacist decisions         | SQLite audit log                                 |
| Explain overrides/rejections         | Structured reason-code taxonomy                  |
| Protect authentication data          | PBKDF2-HMAC-SHA256 password hashing              |
| Support barcode workflows            | Lookup, update, and validation                   |
| Provide ML assistance                | Expiry-risk classification advisory              |
| Verify reliability                   | Automated Pytest regression suite                |
| Provide reproducible test data       | Deterministic synthetic dataset                  |

---

## ✨ Key Features

### 1. Expiry-Risk Assessment

Each medicine batch receives an explainable risk score based on its remaining shelf life and inventory characteristics.

The system categorizes batches into:

* **Critical:** 0–7 days remaining
* **Near Expiry:** 8–30 days remaining
* **Watch:** 31–90 days remaining
* **Safe:** more than 90 days remaining
* **Expired:** fewer than 0 days remaining

The risk score is designed to make the recommendation process understandable to pharmacy staff.

---

### 2. Safety-First Redistribution

Before ranking a destination, the system checks hard safety constraints.

A destination must satisfy conditions including:

* Valid inventory and demand values.
* Positive destination demand.
* Positive storage capacity.
* Sufficient remaining shelf life for transportation.
* Available destination capacity.
* Ability to absorb the required stock.
* Source quantity must be valid and non-negative.

A destination that fails a mandatory safety condition is excluded before objective scoring.

---

### 3. Formal Multi-Criteria Destination Ranking

The redistribution recommender uses a normalized objective instead of relying only on qualitative rules.

For every viable destination, the system evaluates:

1. **Demand velocity**

   * Destination weekly demand.

2. **Stock value protected**

   * Estimated inventory value that can be protected from expiry/waste.
   * When explicit protected value is unavailable, the system estimates shortage against a demand-based target stock level.

3. **Transit feasibility**

   * Remaining shelf-life margin after accounting for transit time.

The criteria are normalized across viable destinations and combined using:

| Criterion             |   Weight |
| --------------------- | -------: |
| Demand                | **0.35** |
| Protected Stock Value | **0.45** |
| Transit Feasibility   | **0.20** |

The resulting destination score is:

```text
Destination Score =
    0.35 × Normalized Demand
  + 0.45 × Normalized Protected Value
  + 0.20 × Normalized Transit Feasibility
```

This allows the system to balance:

* Where the medicine is needed.
* How much stock value can be protected.
* Whether the transfer has sufficient shelf-life margin.

---

### 4. Deterministic Tie-Breaking

When destinations have similar objective scores, the system uses deterministic tie-breaking:

1. Higher destination score.
2. Higher need score.
3. Lower weeks of stock cover.
4. Higher destination weekly demand.
5. Branch ID.

This makes recommendation results reproducible and easier to test.

---

### 5. Split Redistribution

If a single branch cannot accept the complete source quantity, the system can distribute the stock across multiple eligible branches.

Split allocation respects:

```text
Transfer quantity ≤ Source quantity
Transfer quantity ≤ Destination capacity
Transfer quantity ≤ Destination need
Transfer quantity ≥ 0
```

The total recommended quantity cannot exceed the available source stock.

---

## 🧑‍⚕️ Human-in-the-Loop Decision Making

The pharmacist remains the final decision-maker.

The workflow is:

```text
System Recommendation
        ↓
Pharmacist Review
        ↓
Approve / Override / Reject
        ↓
Reason Code
        ↓
Clinical Justification
        ↓
Audit Record
```

For overrides and rejections, the system records a structured reason code.

### Override / Rejection Reason Codes

| Code                      | Meaning                                                 |
| ------------------------- | ------------------------------------------------------- |
| `LOCAL_STOCK_BUFFER`      | Local stock must be retained as a safety buffer         |
| `STORAGE_UNAVAILABLE`     | Destination storage conditions/capacity are unavailable |
| `REVISED_CLINICAL_DEMAND` | Expected demand has changed                             |
| `COLD_CHAIN_MAINTENANCE`  | Cold-chain requirements prevent the transfer            |
| `TRANSIT_RISK`            | Transit conditions introduce unacceptable risk          |
| `OTHER`                   | Another documented reason                               |

When `OTHER` is selected, additional justification is required.

This provides machine-readable audit data while preserving human clinical judgment.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │     Streamlit UI    │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
       Authentication      Inventory        Admin Dashboard
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Recommendation     │
                    │      Engine         │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
       Expiry Risk       Destination       ML Advisory
        Scoring            Ranking
             │                 │
             └─────────────────┼─────────────────┘
                               │
                               ▼
                    Pharmacist Decision
                               │
                               ▼
                    ┌─────────────────────┐
                    │      SQLite DB      │
                    │   Audit History     │
                    └─────────────────────┘
```

---

## 🔄 Recommendation Safety Flow

```text
Medicine Batch
      │
      ▼
Calculate Days to Expiry
      │
      ▼
Calculate Expiry Risk
      │
      ▼
Check Source Stock
      │
      ▼
Find Candidate Branches
      │
      ▼
Apply Hard Safety Constraints
      │
      ├── Unsafe → Exclude
      │
      ▼
Normalize Viable Candidates
      │
      ▼
Calculate Multi-Criteria Score
      │
      ▼
Deterministic Ranking
      │
      ▼
Generate Recommendation
      │
      ▼
Pharmacist Review
      │
      ├── Approve
      ├── Override
      └── Reject
      │
      ▼
Persist Audit Record
```

---

## 🧮 Expiry-Risk Methodology

The application uses a risk score on a **0–150 scale**.

The score considers factors such as:

* Remaining shelf life.
* Quantity at risk.
* Inventory value.
* Demand-related information.

The system presents the resulting urgency level to the pharmacist rather than treating the score as an automatic clinical decision.

### Risk Categories

| Condition  | Category    |
| ---------- | ----------- |
| Expired    | Expired     |
| 0–7 days   | Critical    |
| 8–30 days  | Near Expiry |
| 31–90 days | Watch       |
| >90 days   | Safe        |

---

## 📊 Destination Scoring

For every viable destination:

### Demand Velocity

The destination's weekly demand represents how quickly the medicine is expected to be consumed.

```text
Demand Velocity = Destination Weekly Demand
```

### Protected Stock Value

The system estimates the value protected by sending stock to a branch with a shortage.

A demand-based target stock level is used when explicit protected-stock information is unavailable.

```text
Target Stock = Weekly Demand × Planning Horizon
Shortage = max(0, Target Stock − Current Stock)

Protected Value =
    Shortage × Unit Cost
```

### Transit Feasibility

Transit feasibility considers the remaining shelf life after transportation.

```text
Shelf-Life Margin =
    Days to Expiry − Transit Days
```

A destination is excluded if its transit time is not safely compatible with the remaining shelf life.

### Normalization

Each criterion is normalized across viable candidates so that different units can be compared.

```text
Normalized Metric =
    (Value − Minimum)
    ------------------
    (Maximum − Minimum)
```

If all candidates have the same value for a criterion, a neutral normalized value is used.

---

## 🗄️ Database and Reliability

SQLite is the authoritative operational store for decision records.

Important information includes:

* Medicine details.
* Batch information.
* Branch inventory.
* Demand information.
* Recommendation results.
* Pharmacist decisions.
* Override/rejection reason codes.
* Audit timestamps.

### Transaction Safety

Database operations are designed to prevent partial updates.

For stock quantity updates:

```text
Validate
   ↓
Begin Transaction
   ↓
Update Database
   ↓
Commit
   ↓
Invalidate Cache
```

If the database operation fails:

```text
Database Error
      ↓
Rollback
      ↓
Raise DatabaseSaveError
      ↓
Keep Cache Consistent
```

Cache invalidation occurs only after a successful database commit.

---

## 📝 Audit Logging

The SQLite decision log provides an auditable record of recommendation outcomes.

A decision can contain:

* Source branch.
* Destination branch.
* Medicine/batch.
* Recommended quantity.
* Final quantity.
* Pharmacist decision.
* Reason code.
* Clinical justification.
* Timestamp.

The `reason_code` field provides structured data for future reporting and analysis.

CSV export is available when explicitly requested through the application's export functionality.

---

## 📦 Barcode Management

The application supports barcode-based medicine workflows.

Features include:

* Barcode lookup.
* Inventory identification.
* Barcode updates.
* Barcode normalization.
* Validation against duplicate/identical updates.

An update where the normalized old barcode and new barcode are identical is rejected.

---

## 🔐 Authentication and Security

Authentication uses password hashing rather than storing plaintext passwords.

### Password Hashing

The application uses:

```text
PBKDF2-HMAC-SHA256
100,000 iterations
16-byte random salt
```

Password comparisons use constant-time comparison.

Legacy SHA-256 password hashes can be migrated when users authenticate successfully.

Plaintext password caching has been removed.

> Demo credentials, where provided by the academic project, are fictional and should not be reused for real systems.

---

## 🤖 Machine Learning Advisory

The project includes an ML model that provides an additional expiry-risk classification signal.

The ML component is used as an **advisory signal**, not as an automatic replacement for the rule-based safety system or pharmacist decision.

### Evaluation Dataset

The synthetic dataset contains:

* **600 medicine batch records**
* **150-record holdout evaluation set**
* **5-fold cross-validation**
* Multiple demand profiles
* Multiple expiry-risk conditions

### Holdout Evaluation

| Metric             |     Result |
| ------------------ | ---------: |
| Weighted Accuracy  | **0.9133** |
| Weighted Precision | **0.9086** |
| Macro Precision    | **0.8598** |
| Weighted Recall    | **0.9133** |
| Macro Recall       | **0.7513** |
| Weighted F1        | **0.9052** |
| Macro F1           | **0.7779** |

### Per-Class Results

| Class  | Precision | Recall |    F1 | Support |
| ------ | --------: | -----: | ----: | ------: |
| High   |     0.898 |  1.000 | 0.946 |      53 |
| Low    |     0.931 |  0.920 | 0.926 |      88 |
| Medium |     0.750 |  0.333 | 0.462 |       9 |

The relatively small Medium-class support should be considered when interpreting the class-specific metrics.

### 5-Fold Cross-Validation

| Fold | Accuracy |
| ---- | -------: |
| 1    |   0.9583 |
| 2    |   0.9750 |
| 3    |   0.9667 |
| 4    |   0.9333 |
| 5    |   0.9500 |

Summary:

```text
Accuracy:     0.9567 ± 0.0143
Weighted F1:  0.9535 ± 0.0159
Macro F1:     0.8915 ± 0.0464
```

---

## 🧪 Synthetic Dataset

The project includes deterministic synthetic data for development, testing, and evaluation.

The dataset contains approximately:

* **600 records**
* **4 branches**
* **10 medicine products**
* Weekly demand between **2 and 80 units**
* Batch quantities between **5 and 500 units**
* Unit costs between **£0.08 and £3.50**
* Approximately:

  * 5% expired
  * 15% critical
  * 20% near expiry
  * 25% watch
  * 35% safe

A fixed random seed is used to make dataset generation reproducible.

This provides varied demand profiles and inventory conditions for testing the recommendation engine.

---

## 🧪 Automated Testing

The project uses **Pytest** for automated testing.

The latest verified test suite contains:

```text
255 passed
```

The tests cover areas including:

* Expiry calculations.
* Risk scoring.
* Destination ranking.
* Normalization.
* Tie-breaking.
* Zero storage capacity.
* Negative stock.
* Expired batches.
* Transit constraints.
* Split redistribution.
* Barcode validation.
* Database failures.
* Transaction rollback.
* Cache invalidation.
* Authentication.
* Password migration.
* Reason-code validation.
* Decision persistence.
* CSV export.
* Admin dashboard filtering.
* Edge cases and regression scenarios.

### Running Tests

```bash
pytest -q
```

Expected result:

```text
255 passed
```

---

## 🛠️ Technology Stack

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Core application        |
| Streamlit    | Web interface           |
| SQLite       | Persistent data storage |
| Pytest       | Automated testing       |
| Pandas       | Data processing         |
| Scikit-learn | Machine learning        |
| NumPy        | Numerical processing    |
| Plotly       | Data visualization      |
| Git/GitHub   | Version control         |

---

## 📁 Project Structure

```text
Pharmacy-Expiry-Stock-Checker-and-Redistribution-Recommender/
│
├── app.py
│
├── constants.py
├── database.py
├── recommender.py
├── log_manager.py
├── ui_theme.py
├── generate_data.py
│
├── pages/
│   └── admin_dashboard.py
│
├── tests/
│   └── test_edge_cases.py
│
├── data/
│   └── synthetic datasets / generated data
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/JOEL-JERRY-DANISH/Pharmacy-Expiry-Stock-Checker-and-Redistribution-Recommender.git
```

Move into the project directory:

```bash
cd Pharmacy-Expiry-Stock-Checker-and-Redistribution-Recommender
```

---

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser.

Typical workflow:

```text
Login
  ↓
Medicine / Barcode Lookup
  ↓
View Batch Details
  ↓
Expiry Risk Assessment
  ↓
ML Advisory Signal
  ↓
Redistribution Recommendation
  ↓
Review Destination Score
  ↓
Approve / Override / Reject
  ↓
Enter Reason Code
  ↓
Confirm Decision
  ↓
Audit Record
```

---

## 📊 Admin Dashboard

The administrator dashboard provides visibility into operational decisions.

It supports information such as:

* Inventory records.
* Recommendation history.
* Pharmacist decisions.
* Override/rejection reason codes.
* Decision filtering.
* Audit information.
* Export functionality.

This helps make the system's recommendation history inspectable and auditable.

---

## 🔍 Explainability

The recommendation interface exposes decision factors instead of displaying only a final destination.

Important factors include:

* Expiry urgency.
* Source quantity.
* Destination demand.
* Destination stock.
* Destination capacity.
* Transit time.
* Shelf-life margin.
* Protected inventory value.
* Normalized objective components.
* Final destination score.
* ML advisory signal.

This supports pharmacist review and makes the system easier to demonstrate and test.

---

## ⚠️ Safety Constraints

The system is designed to reject or exclude unsafe redistribution candidates.

Examples include:

```text
Negative source stock
        ↓
Invalid

Zero destination capacity
        ↓
Invalid

Zero destination demand
        ↓
Invalid

Transit time >= remaining shelf life
        ↓
Invalid

Insufficient total viable capacity
        ↓
No viable redistribution
```

The system therefore prioritizes safety constraints before optimization.

---

## 🧑‍💻 Development Principles

The project follows several important software engineering principles:

### Safety First

Hard constraints are evaluated before objective scoring.

### Explainability

Recommendations expose the factors contributing to the result.

### Determinism

Tie-breaking rules make results reproducible.

### Human Oversight

Pharmacists retain control over final decisions.

### Auditability

Decisions and override reasons are persisted.

### Reliability

Database failures are handled using rollback and explicit error handling.

### Testability

Boundary conditions and regression cases are covered with automated tests.

---

## 📈 Future Enhancements

Potential future improvements include:

* Time-series demand forecasting.
* Continuous ML model retraining.
* More detailed cold-chain constraints.
* Real-time branch inventory synchronization.
* Automated notification for critical expiry batches.
* More advanced transfer optimization.
* Role-based access control.
* Expanded audit analytics.
* Historical demand trend visualization.
* Integration with real pharmacy inventory systems.

---

## ⚠️ Limitations

This project uses synthetic data and is intended as an academic/software engineering prototype.

It does not currently represent:

* Real-time pharmacy inventory systems.
* Real clinical decision-making.
* Regulatory approval workflows.
* Real-world cold-chain infrastructure.
* Real pharmaceutical supply-chain contracts.
* Production-scale authentication infrastructure.

The ML model is an advisory component and should not independently determine pharmacy operations.

---

## 📋 Example Recommendation Logic

A simplified example:

```text
Source Branch
    Medicine: Product A
    Quantity: 100
    Days to Expiry: 20

Candidate Branch A
    Demand: 60/week
    Stock: 10
    Capacity: 100
    Transit: 2 days

Candidate Branch B
    Demand: 30/week
    Stock: 5
    Capacity: 100
    Transit: 5 days
```

The system first checks safety.

Then it calculates:

```text
Demand Score
Value Protection Score
Transit Feasibility Score
```

These are normalized and combined:

```text
Final Score =
    0.35 × Demand
  + 0.45 × Protected Value
  + 0.20 × Transit
```

The candidates are then ranked deterministically.

The pharmacist reviews the recommendation before any transfer is recorded as approved.

---

## 📌 Project Status

Current implementation includes:

* ✅ Expiry-risk scoring
* ✅ Barcode lookup and validation
* ✅ Multi-branch inventory management
* ✅ Formal multi-criteria destination ranking
* ✅ Normalized objective function
* ✅ Deterministic destination ranking
* ✅ Split redistribution
* ✅ Safety constraints
* ✅ Human-in-the-loop decision workflow
* ✅ Structured override/rejection reason codes
* ✅ SQLite audit logging
* ✅ Transaction-safe inventory updates
* ✅ PBKDF2 password hashing
* ✅ Legacy password migration
* ✅ ML advisory classification
* ✅ Synthetic evaluation dataset
* ✅ Automated Pytest suite
* ✅ Admin dashboard
* ✅ CSV decision export
* ✅ Regression and boundary-condition testing

---

## 👥 Academic Project

This project demonstrates the application of:

* Software engineering
* Database management
* Machine learning
* Optimization
* Human-in-the-loop systems
* Explainable decision support
* Automated testing
* Security engineering
* Data analysis
* Web application development

The project is intended to demonstrate how software can assist pharmacy inventory management while keeping final decisions under appropriate human oversight.

---

## 📄 License

Add the project's applicable license here if one is required by your institution or repository policy.

---

## 🔗 Repository

[GitHub Repository](https://github.com/JOEL-JERRY-DANISH/Pharmacy-Expiry-Stock-Checker-and-Redistribution-Recommender?utm_source=chatgpt.com)
