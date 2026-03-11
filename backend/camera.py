import cv2
import time
import os
from datetime import datetime

from .recognition import recognize_face
from .config import UNKNOWN_DIR
from .n8n import send_event

# -------------------------
# Face detector
# -------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------
# Camera
# -------------------------
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# -------------------------
# Control unknown alerts
# -------------------------
last_unknown_save = 0
UNKNOWN_COOLDOWN = 3  # seconds


# -------------------------
# Frame generator
# -------------------------
def generate_frames():

    global last_unknown_save

    while True:

        success, frame = cap.read()

        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:

            # expand bounding box
            pad = 20

            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # -------------------------
            # Face recognition
            # -------------------------
            name, confidence = recognize_face(face)

            label = f"{name} {confidence}%"

            color = (0, 255, 0)

            # -------------------------
            # UNKNOWN PERSON
            # -------------------------
            if name == "Unknown":

                color = (0, 0, 255)

                # Prevent saving every frame
                if time.time() - last_unknown_save > UNKNOWN_COOLDOWN:

                    filename = f"{int(time.time()*1000)}.jpg"

                    path = os.path.join(UNKNOWN_DIR, filename)

                    cv2.imwrite(path, face)

                    last_unknown_save = time.time()

                    print("⚠ Unknown person detected")

                    # -------------------------
                    # Send event to n8n
                    # -------------------------
                    try:

                        event = {
                            "event": "unknown_person",
                            "time": datetime.now().isoformat(),
                            "confidence": confidence
                        }

                        send_event(event, path)

                    except Exception as e:
                        print("⚠ n8n send failed:", e)

            # -------------------------
            # Draw bounding box
            # -------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # -------------------------
        # Encode frame for streaming
        # -------------------------
        ret, buffer = cv2.imencode(".jpg", frame)

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )