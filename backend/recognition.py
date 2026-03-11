import cv2
import numpy as np
from deepface import DeepFace
import faiss
import os
import pickle

INDEX_PATH = "data/faiss_index.bin"
LABELS_PATH = "data/faiss_labels.pkl"

index = None
labels = []


# -------------------------
# Load FAISS index
# -------------------------
def build_index():
    global index, labels

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "rb") as f:
            labels = pickle.load(f)

    if index is not None:
        print(f"FAISS index loaded with {len(labels)} faces")


# -------------------------
# Face recognition
# -------------------------
def recognize_face(face):

    global index, labels

    if index is None:
        build_index()

    if index is None or len(labels) == 0:
        return "Unknown", 0

    try:
        face_resized = cv2.resize(face, (224, 224))

        embedding = DeepFace.represent(
            img_path=face_resized,
            model_name="ArcFace",
            detector_backend="opencv",
            enforce_detection=False
        )[0]["embedding"]

        emb = np.array([embedding]).astype("float32")

        D, I = index.search(emb, 1)

        distance = float(D[0][0])
        idx = int(I[0][0])

        print("MATCH DEBUG:", distance, idx, labels[idx] if idx < len(labels) else "no label")

        # Correct ArcFace threshold
        if distance < 7:

            name = labels[idx]

            confidence = max(0, 100 - distance * 10)

            return name, round(confidence, 2)

    except Exception as e:
        print("Recognition error:", e)

    return "Unknown", 0