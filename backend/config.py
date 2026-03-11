import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

KNOWN_DIR = os.path.join(DATA_DIR, "known_faces")
UNKNOWN_DIR = os.path.join(DATA_DIR, "unknown_faces")
LOG_DIR = os.path.join(DATA_DIR, "logs")

DATABASE_PATH = os.path.join(DATA_DIR, "events.db")

os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)