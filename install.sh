#!/bin/bash
# Run this once to set up the environment correctly
set -e
echo "🔧 Installing FaceAI dependencies..."

pip install tensorflow==2.16.2 tf-keras==2.16.0
pip install numpy==1.26.4 --force-reinstall
pip install opencv-python-headless==4.9.0.80
pip install fastapi==0.115.0 "uvicorn[standard]==0.30.6" python-multipart==0.0.9
pip install faiss-cpu==1.8.0
pip install retina-face mtcnn --no-deps
pip install Pillow tqdm pandas requests cloudinary
pip install deepface==0.0.93 --no-deps
pip install numpy==1.26.4 --force-reinstall
pip uninstall opencv-python -y 2>/dev/null || true

echo "✅ Done. Run: uvicorn backend.main:app --host 0.0.0.0 --port 8000"