import cv2
import time
from pathlib import Path
from .n8n import send_unknown_alert
from .database import log_event

UNKNOWN_DIR = Path("data/unknown_faces")

# -------------------------
# Agent memory
# -------------------------
people_state = {}  # track_id -> {name, enter, last_seen}


def handle_person(track_id, name):
    """
    Track entry/exit for a person by track_id.
    Returns "ENTRY" when someone first appears.
    """
    now = time.time()

    if track_id not in people_state:
        people_state[track_id] = {
            "name": name,
            "enter": now,
            "last_seen": now
        }
        return "ENTRY"
    else:
        people_state[track_id]["last_seen"] = now
        return None


def cleanup():
    """
    Check for people who have left (no update for >5 sec)
    Returns list of exited people as (track_id, data)
    """
    now = time.time()
    exited = []

    for pid, data in list(people_state.items()):
        if now - data["last_seen"] > 5:
            exited.append((pid, data))
            del people_state[pid]

    return exited


# -------------------------
# Detection handler
# -------------------------
def handle_detection(person):
    """
    Logs detection and unknown faces.
    """
    name = person["name"]
    face = person.get("face")  # may not exist if only bounding box
    track_id = person.get("id")  # optional

    # Use agent memory if track_id exists
    entry_status = None
    if track_id is not None:
        entry_status = handle_person(track_id, name)

    timestamp = int(time.time())

    if name == "Unknown" and face is not None:
        path = UNKNOWN_DIR / f"{timestamp}.jpg"
        cv2.imwrite(str(path), face)
        send_unknown_alert(str(path))

        # Log only if first entry or always?
        log_event("Unknown", "ENTRY", str(path))

    elif entry_status == "ENTRY":
        # Only log real people on first entry
        log_event(name, "ENTRY", "")