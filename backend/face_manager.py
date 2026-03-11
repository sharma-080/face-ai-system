import os
import cv2
import numpy as np
import faiss
from deepface import DeepFace

KNOWN_DIR = "data/known_faces"

index = None
labels = []

def save_known_face(name, images):

    person_dir = os.path.join(KNOWN_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    paths = []

    for i, img_bytes in enumerate(images):

        path = os.path.join(person_dir, f"{i}.jpg")

        with open(path, "wb") as f:
            f.write(img_bytes)

        paths.append(path)

    print(f"Saved {len(paths)} images for {name}")

    build_index()

    return paths


def build_index():

    global index, labels

    print("Building FAISS index...")

    embeddings = []
    labels = []

    for person in os.listdir(KNOWN_DIR):

        person_dir = os.path.join(KNOWN_DIR, person)

        if not os.path.isdir(person_dir):
            continue

        for img in os.listdir(person_dir):

            path = os.path.join(person_dir, img)

            try:

                emb = DeepFace.represent(
                    img_path=path,
                    model_name="ArcFace",
                    detector_backend="opencv",
                    enforce_detection=False
                )[0]["embedding"]

                embeddings.append(emb)
                labels.append(person)

            except Exception as e:
                print("Skipping image:", path, e)

    if len(embeddings) == 0:
        print("No embeddings found")
        return

    dim = len(embeddings[0])

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    print("FAISS index built with", len(labels), "faces")