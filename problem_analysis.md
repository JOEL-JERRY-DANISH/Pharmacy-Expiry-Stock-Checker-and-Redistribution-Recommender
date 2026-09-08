
# Problem Analysis — Pharmacy Expiry Stock Recommender

## Problem Statement

Community pharmacies managing patients with multiple prescriptions
hold diverse stock across multiple medicine types. Near-expiry stock
is discovered too late for responsible redistribution, resulting in
financial waste (est. £2,340/month) and patient supply risk.

## Root Causes

- Stock checks are manual and weekly — too infrequent
- No visibility of demand at other branches
- No system to score urgency or recommend action
- No audit trail of decisions made
- Near-expiry items discovered inside 7 days — too late to transfer

## AS-IS Workflow (without system)

1. Pharmacist does manual weekly stock check
2. Near-expiry items discovered — often inside 7 days
3. Local decision made: discard, donate, or mark down
4. No record kept — no audit trail, no learning

Pain points:

- Too late to redistribute by the time items are found
- No knowledge of which other branch needs the medicine
- No consistency in decisions across branches

## TO-BE Workflow (with system)

1. System scans all stock data daily (automated)
2. Recommender scores every batch by urgency + value + quantity
3. Best receiving branch found by demand + capacity check
4. Recommendation shown to pharmacist in plain English with reason
5. Pharmacist confirms or overrides (override reason required)
6. Decision logged with timestamp — full audit trail

## Primary User Profile

Role:         Pharmacy manager / senior pharmacist
Tech comfort: Basic — uses email and NHS systems
              Not comfortable with complex dashboards
Needs:

- Plain English — no technical terms
- Clear colour coding (red = urgent, orange = soon)
- One clear action per recommendation
- Ability to say no with a recorded reason
- Confidence system will not act without their approval

## Stakeholder Map

Primary:   Pharmacy manager — makes all transfer decisions
Secondary: Branch staff — receives transferred stock
           Delivery/logistics — physical transfer capacity
Affected:  Patients — continuity of medicine supply
           Pharmacy owner — financial loss reduction
           NHS/regulator — audit compliance, waste reduction

## Success Condition

Value of stock used or transferred before expiry increases
by at least 60% compared to baseline (no system).

Baseline:  18% of at-risk stock saved (£420 of £2,340)
Target:    60% of at-risk stock saved
Result:    74% achieved — target met
