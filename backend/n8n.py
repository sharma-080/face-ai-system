import base64
import requests
import socket

WEBHOOK_URL = ""
DEVICE_ID = socket.gethostname()


def send_event(event, image_path):

    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        payload = {
            "device_id": DEVICE_ID,
            "decision": event.get("event"),
            "time": event.get("time"),
            "confidence": event.get("confidence"),
            "image": encoded
        }

        response = requests.post(WEBHOOK_URL, json=payload, timeout=10)

        print("📤 Alert sent:", response.status_code)

    except Exception as e:
        print("⚠ n8n webhook failed:", e)