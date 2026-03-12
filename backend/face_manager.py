import os
import cv2
import numpy as np
import faiss
import pickle
from deepface import DeepFace

KNOWN_DIR = "data/known_faces"
INDEX_PATH = "data/faiss_index.bin"
LABELS_PATH = "data/faiss_labels.pkl"

MODEL_NAME = "FaceNet512"
EMBEDDING_DIM = 512


def save_known_face(name: str, images: list[bytes]) -> list[str]:
    """
    Save raw image bytes for a person and rebuild the FAISS index.
    """
    person_dir = os.path.join(KNOWN_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    paths = []

    for i, img_bytes in enumerate(images):
        path = os.path.join(person_dir, f"{i}.jpg")
        with open(path, "wb") as f:
            f.write(img_bytes)
        paths.append(path)

    print(f"💾 Saved {len(paths)} images for '{name}'")

    build_and_save_index()

    return paths


def build_and_save_index():
    """
    Rebuild FAISS index from all known faces using FaceNet512.
    Saves index + labels to disk.
    """
    print(f"🔨 Rebuilding FAISS index with {MODEL_NAME}...")

    embeddings = []
    labels = []

    for person in sorted(os.listdir(KNOWN_DIR)):
        person_dir = os.path.join(KNOWN_DIR, person)

        if not os.path.isdir(person_dir):
            continue

        for img_file in sorted(os.listdir(person_dir)):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(person_dir, img_file)

            try:
                result = DeepFace.represent(
                    img_path=path,
                    model_name=MODEL_NAME,
                    detector_backend="opencv",
                    enforce_detection=False,
                    align=True
                )
                emb = result[0]["embedding"]
                embeddings.append(emb)
                labels.append(person)
                print(f"  ✔ {person}/{img_file}")

            except Exception as e:
                print(f"  ✗ Skipped {path}: {e}")

    if not embeddings:
        print("❌ No embeddings — check known_faces directory.")
        return

    arr = np.array(embeddings).astype("float32")
    faiss.normalize_L2(arr)  # cosine similarity via inner product

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(arr)

    faiss.write_index(index, INDEX_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(labels, f)

    print(f"✅ Index built: {len(labels)} embeddings, {len(set(labels))} people")