import sqlite3
import csv
import os
from datetime import datetime
from .config import DATABASE_PATH

LOG_FILE = "data/logs/events.csv"

os.makedirs("data/logs", exist_ok=True)


def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        time        TEXT,
        name        TEXT,
        event_type  TEXT,
        confidence  REAL,
        image       TEXT
    )
    """)
    conn.commit()
    conn.close()


def insert_event(time, name, event_type, confidence, image):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute(
        "INSERT INTO events(time,name,event_type,confidence,image) VALUES(?,?,?,?,?)",
        (time, name, event_type, confidence, image)
    )
    conn.commit()
    conn.close()


def log_event(name: str, event: str, image: str = "", confidence: float = 0):
    """
    Log to CSV.
    event must be one of: ENTRY, EXIT, UNKNOWN
    - ENTRY   → known person recognized
    - UNKNOWN → unrecognized face detected
    Never logs a known person as UNKNOWN or vice versa.
    """
    # Guard: known people must use ENTRY/EXIT, never UNKNOWN
    if name != "Unknown" and event == "UNKNOWN":
        event = "ENTRY"

    # Guard: Unknown faces must use UNKNOWN, never ENTRY
    if name == "Unknown" and event == "ENTRY":
        event = "UNKNOWN"

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            name,
            event,
            round(confidence, 2),
            image
        ])