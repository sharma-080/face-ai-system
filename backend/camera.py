import cv2
import time
import os
from datetime import datetime

from .recognition import recognize_all_faces, load_index
from .config import UNKNOWN_DIR
from .n8n import send_event
from .database import log_event

# ── Camera ──
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

DETECTION_SCALE  = 0.75
RECOG_EVERY_N    = 3
UNKNOWN_COOLDOWN = 5   # seconds between saving same unknown
ENTRY_COOLDOWN   = 30  # seconds before re-logging same known person

_frame_count  = 0
_last_results: list = []
last_unknown: dict[str, float] = {}  # cell → last save time
last_entry:   dict[str, float] = {}  # name → last log time


def draw_label(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def generate_frames():
    global _frame_count, _last_results

    load_index()

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.05)
            continue

        _frame_count += 1
        now = time.time()

        # ── Run recognition every N frames ──
        if _frame_count % RECOG_EVERY_N == 0:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (int(w * DETECTION_SCALE), int(h * DETECTION_SCALE)))
            raw   = recognize_all_faces(small)

            scale = 1.0 / DETECTION_SCALE
            _last_results = [{
                "name":       r["name"],
                "confidence": r["confidence"],
                "box": (
                    int(r["box"][0] * scale),
                    int(r["box"][1] * scale),
                    int(r["box"][2] * scale),
                    int(r["box"][3] * scale),
                )
            } for r in raw]

        # ── Draw + log ──
        for person in _last_results:
            name       = person["name"]
            confidence = person["confidence"]
            x, y, bw, bh = person["box"]

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + bw)
            y2 = min(frame.shape[0], y + bh)

            color = (50, 220, 50) if name != "Unknown" else (30, 30, 220)
            label = f"{name}  {confidence}%" if confidence > 0 else "Unknown"
            draw_label(frame, x1, y1, x2, y2, label, color)

            if name == "Unknown":
                # ── Unknown person: save image + log as UNKNOWN ──
                cell = f"{x // 120}_{y // 120}"
                if now - last_unknown.get(cell, 0) > UNKNOWN_COOLDOWN:
                    face = frame[y1:y2, x1:x2]
                    if face.size > 0:
                        filename = f"{int(now * 1000)}.jpg"
                        path = os.path.join(UNKNOWN_DIR, filename)
                        cv2.imwrite(path, face)
                        last_unknown[cell] = now

                        # Log as UNKNOWN — never ENTRY
                        log_event("Unknown", "UNKNOWN", path, confidence)
                        print(f"⚠ Unknown saved → {filename}")

                        try:
                            send_event(
                                {"event": "unknown_person",
                                 "time": datetime.now().isoformat(),
                                 "confidence": confidence},
                                path
                            )
                        except Exception as e:
                            print(f"⚠ n8n failed: {e}")

            else:
                # ── Known person: log as ENTRY (throttled) ──
                if now - last_entry.get(name, 0) > ENTRY_COOLDOWN:
                    last_entry[name] = now
                    log_event(name, "ENTRY", "", confidence)
                    print(f"✅ ENTRY logged: {name} ({confidence}%)")

        # ── Stream ──
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"