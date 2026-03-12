"""
Run from face_ai_system/ root:
  export CLOUD_URL="https://faceai-cloud-production.up.railway.app"
  export EDGE_TOKEN="abc123"
  export CLOUDINARY_CLOUD_NAME="your-cloud-name"
  export CLOUDINARY_API_KEY="your-api-key"
  export CLOUDINARY_API_SECRET="your-api-secret"
  python3 edge_pusher.py
"""
import os
import time
import threading
import requests
import cv2

CLOUD_URL   = os.environ.get("CLOUD_URL", "").rstrip("/")
EDGE_TOKEN  = os.environ.get("EDGE_TOKEN", "abc123")
HEADERS     = {"x-token": EDGE_TOKEN}

CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_API_KEY    = os.environ.get("CLOUDINARY_API_KEY", "")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET", "")

FRAME_INTERVAL = 1.0
EVENT_INTERVAL = 3.0

last_event_time   = ""
last_unknown_seen = set()   # track which unknown files already uploaded

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def upload_to_cloudinary(image_path: str) -> str:
    """Upload image to Cloudinary, return public URL."""
    if not CLOUDINARY_CLOUD_NAME:
        return ""
    try:
        import cloudinary
        import cloudinary.uploader
        cloudinary.config(
            cloud_name=CLOUDINARY_CLOUD_NAME,
            api_key=CLOUDINARY_API_KEY,
            api_secret=CLOUDINARY_API_SECRET
        )
        result = cloudinary.uploader.upload(
            image_path,
            folder="faceai_unknowns",
            resource_type="image"
        )
        return result.get("secure_url", "")
    except Exception as e:
        print(f"⚠ Cloudinary upload error: {e}")
        return ""


def push_frames():
    print(f"📡 Pushing frames → {CLOUD_URL}")
    while True:
        try:
            ok, frame = cap.read()
            if ok:
                _, buf = cv2.imencode(".jpg", frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 75])
                requests.post(
                    f"{CLOUD_URL}/push/frame",
                    data=buf.tobytes(),
                    headers={**HEADERS, "Content-Type": "image/jpeg"},
                    timeout=8
                )
        except Exception as e:
            print(f"⚠ Frame error: {e}")
        time.sleep(FRAME_INTERVAL)


def push_events():
    global last_event_time, last_unknown_seen
    print(f"📡 Pushing events → {CLOUD_URL}")

    unknown_dir = "data/unknown_faces"

    while True:
        try:
            # ── Push new unknown face images ──
            if os.path.exists(unknown_dir):
                files = sorted(os.listdir(unknown_dir))
                for fname in files:
                    if fname in last_unknown_seen:
                        continue
                    if not fname.endswith(".jpg"):
                        continue

                    path      = os.path.join(unknown_dir, fname)
                    image_url = upload_to_cloudinary(path)
                    last_unknown_seen.add(fname)

                    # Push as UNKNOWN event with image URL
                    requests.post(
                        f"{CLOUD_URL}/push/event",
                        json={
                            "time":       fname.replace(".jpg", ""),
                            "name":       "Unknown",
                            "event":      "UNKNOWN",
                            "confidence": 0,
                            "image_url":  image_url,
                        },
                        headers=HEADERS,
                        timeout=5
                    )
                    if image_url:
                        print(f"📸 Unknown uploaded → {image_url}")

            # ── Push known person events from CSV ──
            events = requests.get(
                "http://localhost:8000/api/events?limit=20",
                timeout=5
            ).json()

            for event in reversed(events):
                t = str(event.get("time", ""))
                if t and t > last_event_time:
                    last_event_time = t
                    if event.get("event") == "ENTRY":
                        requests.post(
                            f"{CLOUD_URL}/push/event",
                            json={
                                "time":       t,
                                "name":       event.get("name"),
                                "event":      "ENTRY",
                                "confidence": float(event.get("confidence") or 0),
                                "image_url":  "",
                            },
                            headers=HEADERS,
                            timeout=5
                        )

        except Exception as e:
            print(f"⚠ Event error: {e}")

        time.sleep(EVENT_INTERVAL)


if __name__ == "__main__":
    if not CLOUD_URL:
        print("❌ Set CLOUD_URL first")
        exit(1)

    print(f"🚀 Edge pusher → {CLOUD_URL}")
    if CLOUDINARY_CLOUD_NAME:
        print(f"☁ Cloudinary enabled → unknown faces will be uploaded")
    else:
        print(f"⚠ Cloudinary not set → unknown face images won't show in cloud dashboard")

    threading.Thread(target=push_frames, daemon=True).start()
    threading.Thread(target=push_events, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")