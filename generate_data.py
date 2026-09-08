import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random, os

random.seed(42)
np.random.seed(42)

MEDICINES = [
    {"name": "Metformin 500mg",     "category": "Diabetes",       "unit_cost": 0.12},
    {"name": "Atorvastatin 20mg",   "category": "Cholesterol",    "unit_cost": 0.18},
    {"name": "Amlodipine 5mg",      "category": "Blood Pressure", "unit_cost": 0.09},
    {"name": "Omeprazole 20mg",     "category": "Gastric",        "unit_cost": 0.22},
    {"name": "Salbutamol Inhaler",  "category": "Respiratory",    "unit_cost": 3.50},
    {"name": "Ramipril 5mg",        "category": "Heart",          "unit_cost": 0.15},
    {"name": "Levothyroxine 50mcg", "category": "Thyroid",        "unit_cost": 0.08},
    {"name": "Amoxicillin 500mg",   "category": "Antibiotic",     "unit_cost": 0.25},
    {"name": "Warfarin 5mg",        "category": "Anticoagulant",  "unit_cost": 0.11},
    {"name": "Sertraline 50mg",     "category": "Mental Health",  "unit_cost": 0.14},
]

BRANCHES = [
    {"id":"BR01","name":"Central Pharmacy","capacity":5000},
    {"id":"BR02","name":"North Branch",    "capacity":3000},
    {"id":"BR03","name":"East Branch",     "capacity":2500},
    {"id":"BR04","name":"South Branch",    "capacity":2000},
]

def make_expiry():
    today = datetime.today()
    bucket = random.choices([0,1,2,3,4],
                            weights=[0.05,0.15,0.20,0.25,0.35])[0]
    days = {0:-random.randint(1,30), 1:random.randint(1,7),
            2:random.randint(8,30),  3:random.randint(31,90),
            4:random.randint(91,730)}[bucket]
    return (today + timedelta(days=days)).strftime("%Y-%m-%d")

def make_barcode():
    return f"50001{random.randint(10000000,99999999)}"

os.makedirs("data", exist_ok=True)
stock_rows, barcode_rows = [], []

for i in range(600):
    med    = random.choice(MEDICINES)
    branch = random.choice(BRANCHES)
    batch_id = f"BATCH-{10000+i}"
    stock_rows.append({
        "batch_id":                  batch_id,
        "medicine_name":             med["name"],
        "category":                  med["category"],
        "branch_id":                 branch["id"],
        "branch_name":               branch["name"],
        "quantity":                  random.randint(5,500),
        "expiry_date":               make_expiry(),
        "unit_cost_gbp":             med["unit_cost"],
        "demand_per_week":           random.randint(2,80),
        "branch_capacity_remaining": random.randint(0,branch["capacity"]),
    })
    original_bc = make_barcode()
    barcode_rows.append({
        "barcode":           original_bc,
        "batch_id":          batch_id,
        "medicine_name":     med["name"],
        "registered_date":   "2026-01-01",
        "superseded_date":   None,
        "reason_for_change": "initial",
    })
    if random.random() < 0.20:
        new_bc = make_barcode()
        barcode_rows[-1]["superseded_date"] = "2026-03-15"
        barcode_rows.append({
            "barcode":           new_bc,
            "batch_id":          batch_id,
            "medicine_name":     med["name"],
            "registered_date":   "2026-03-15",
            "superseded_date":   None,
            "reason_for_change": random.choice([
                "repackaging","FMD serialisation",
                "label correction","new lot"]),
        })

pd.DataFrame(stock_rows).to_csv("data/medicines.csv",        index=False)
pd.DataFrame(barcode_rows).to_csv("data/barcode_history.csv", index=False)
print(f"Generated {len(stock_rows)} stock records   -> data/medicines.csv")
print(f"Generated {len(barcode_rows)} barcode records -> data/barcode_history.csv")
print("Done! Now create barcode_registry.py")