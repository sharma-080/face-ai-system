import os
import csv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .camera import generate_frames
from .recognition import build_index
from .database import init_db

app = FastAPI(title="FaceAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
KNOWN_DIR = os.path.join(DATA_DIR, "known_faces")
UNKNOWN_DIR = os.path.join(DATA_DIR, "unknown_faces")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
LOG_FILE = os.path.join(DATA_DIR, "logs", "events.csv")

os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "logs"), exist_ok=True)

# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
def startup():
    init_db()
    build_index()  # load FAISS from disk


# -------------------------
# Live video stream
# -------------------------
@app.get("/video")
def video():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# -------------------------
# Add known face
# -------------------------
@app.post("/api/add-known-face")
async def add_known_face(
    name: str = Form(...),
    images: list[UploadFile] = File(...),
):
    from .face_manager import save_known_face

    imgs = [await img.read() for img in images]
    paths = save_known_face(name, imgs)
    build_index()  # reload into memory
    return {"saved": len(paths), "person": name}


# -------------------------
# Recent unknown faces
# -------------------------
@app.get("/api/unknown_faces")
def unknown_faces():
    if not os.path.exists(UNKNOWN_DIR):
        return []
    files = sorted(os.listdir(UNKNOWN_DIR), reverse=True)[:12]
    return files


# -------------------------
# Recent events from CSV
# -------------------------
@app.get("/api/events")
def get_events(limit: int = 50):
    if not os.path.exists(LOG_FILE):
        return []
    rows = []
    try:
        with open(LOG_FILE, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    rows.append({
                        "name": row[0],
                        "event": row[1],
                        "time": row[2],
                        "image": row[3] if len(row) > 3 else ""
                    })
    except Exception:
        pass
    return list(reversed(rows))[:limit]


# -------------------------
# Known people list
# -------------------------
@app.get("/api/known_people")
def known_people():
    if not os.path.exists(KNOWN_DIR):
        return []
    return [
        p for p in os.listdir(KNOWN_DIR)
        if os.path.isdir(os.path.join(KNOWN_DIR, p))
    ]


# -------------------------
# Static files
# -------------------------
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")