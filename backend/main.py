import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .camera import generate_frames
from .recognition import build_index  # <-- import build_index to rebuild embeddings

app = FastAPI()

# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
KNOWN_DIR = os.path.join(DATA_DIR, "known_faces")
UNKNOWN_DIR = os.path.join(DATA_DIR, "unknown_faces")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# -----------------------------
# VIDEO STREAM
# -----------------------------
@app.get("/video")
def video():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# -----------------------------
# ADD KNOWN FACE
# -----------------------------
@app.post("/api/add-known-face")
async def add_known_face(
    name: str = Form(...),
    images: list[UploadFile] = File(...)
):
    from .face_manager import save_known_face

    imgs = []

    for image in images:
        content = await image.read()
        imgs.append(content)

    # Save faces (cropped + aligned)
    paths = save_known_face(name, imgs)

    # Rebuild FAISS index so new faces are immediately recognized
    build_index()

    print(f"Saved {len(paths)} images for {name} and rebuilt index")

    return {"saved": len(paths)}

# -----------------------------
# UNKNOWN FACE GALLERY
# -----------------------------
@app.get("/api/unknown_faces")
def unknown_faces():
    if not os.path.exists(UNKNOWN_DIR):
        return []

    files = sorted(os.listdir(UNKNOWN_DIR), reverse=True)[:8]
    return files

# -----------------------------
# STATIC FILES
# -----------------------------
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")