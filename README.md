# FaceAI — Local Setup & Deployment

## Folder structure
```
face_ai_system/
├── backend/              ← YOUR EXISTING BRAIN (don't touch)
│   ├── __init__.py
│   ├── main.py
│   ├── camera.py
│   ├── recognition.py
│   ├── face_manager.py
│   ├── database.py
│   ├── config.py
│   ├── n8n.py
│   └── agent.py
│
├── cloud/                ← NEW FOLDER (just add this)
│   ├── server.py
│   ├── dashboard.html
│   ├── requirements.txt  ← CLOUD ONLY (fastapi + uvicorn, nothing else)
│   └── Dockerfile
│
├── edge_pusher.py        ← NEW FILE (put at root, runs on Mac)
│
├── frontend/
│   └── dashboard.html
├── data/
├── install.sh
├── start.sh
└── requirements.txt      ← MAC/JETSON (don't touch this)

## First-time setup
```bash
cd face_ai_system
python -m venv venv
source venv/bin/activate
bash install.sh
```

## Run
```bash
source venv/bin/activate
bash start.sh
# Open http://localhost:8000
```

## Register a face
1. Open http://localhost:8000
2. Type a name → click **CAPTURE** → stay in frame for 40 frames
3. Index rebuilds automatically

## Delete old index (if recognition seems wrong)
```bash
rm data/faiss_index.bin data/faiss_labels.pkl
# Re-register faces from dashboard
```

## Accuracy notes
- Threshold: 0.55 cosine similarity (raise in recognition.py → CONFIRM_THRESHOLD if strangers match)
- Smooth frames: 6 (raise in camera.py → SMOOTH_FRAMES to reduce flicker more)
- Good lighting and frontal face = best results

## Event log
- `data/logs/events.csv` — columns: time, name, event, confidence, image
- Events: ENTRY (known person), UNKNOWN (unrecognized face)
