# database.py
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "data/pharmacy.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def initialise_database():
    """Create all tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Stock table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock (
            batch_id TEXT PRIMARY KEY,
            medicine_name TEXT,
            category TEXT,
            branch_id TEXT,
            branch_name TEXT,
            quantity INTEGER,
            expiry_date TEXT,
            unit_cost_gbp REAL,
            demand_per_week INTEGER,
            branch_capacity_remaining INTEGER
        )
    """)

    # Barcode table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS barcodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            barcode TEXT,
            batch_id TEXT,
            medicine_name TEXT,
            registered_date TEXT,
            superseded_date TEXT,
            reason_for_change TEXT
        )
    """)

    # Decision log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            batch_id TEXT,
            medicine TEXT,
            action TEXT,
            destination TEXT,
            override_reason TEXT,
            user TEXT
        )
    """)

    conn.commit()
    conn.close()

def load_stock():
    """Load stock data as a pandas DataFrame."""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM stock", conn)
    conn.close()
    return df

def save_decision(batch_id, medicine, action,
                  destination="", override_reason="",
                  user="pharmacist"):
    """Save a decision to the database."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO decisions
        (timestamp, batch_id, medicine, action,
         destination, override_reason, user)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        batch_id, medicine, action,
        destination, override_reason, user
    ))
    conn.commit()
    conn.close()

def load_decisions():
    """Load all decisions as a pandas DataFrame."""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM decisions", conn)
    conn.close()
    return df

def import_from_csv():
    """Import existing CSV data into the database."""
    conn = get_connection()

    stock_df = pd.read_csv("data/medicines.csv")
    stock_df.to_sql("stock", conn,
                    if_exists="replace", index=False)

    barcode_df = pd.read_csv("data/barcode_history.csv")
    barcode_df.to_sql("barcodes", conn,
                      if_exists="replace", index=False)

    conn.commit()
    conn.close()
    print("CSV data imported into database successfully")

if __name__ == "__main__":
    initialise_database()
    import_from_csv()
    print("Database ready at data/pharmacy.db")