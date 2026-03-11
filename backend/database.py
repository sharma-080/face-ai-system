import sqlite3
import csv
from datetime import datetime
from .config import DATABASE_PATH

LOG_FILE = "data/logs/events.csv"


def init_db():

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT,
        name TEXT,
        event_type TEXT,
        confidence REAL,
        image TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_event(time, name, event_type, confidence, image):

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO events(time,name,event_type,confidence,image) VALUES(?,?,?,?,?)",
        (time, name, event_type, confidence, image)
    )

    conn.commit()
    conn.close()


# -------------------------
# Simple CSV logging
# -------------------------

def log_event(name, event, image):

    with open(LOG_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            name,
            event,
            datetime.now(),
            image
        ])