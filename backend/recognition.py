import cv2
import numpy as np
from deepface import DeepFace
import faiss
import os
import pickle

INDEX_PATH  = "data/faiss_index.bin"
LABELS_PATH = "data/faiss_labels.pkl"

MODEL_NAME    = "Facenet512"
EMBEDDING_DIM = 512

# ── Threshold tuning guide ──
# After L2 normalization, inner product = cosine similarity (0→1)
# Same person good lighting:  0.55–0.85
# Same person poor/distance:  0.35–0.55
# Different people:           0.05–0.30
# 0.35 is aggressive (fewer missed detections, more false positives)
# Raise to 0.50 if strangers are being recognized as known people
COSINE_THRESHOLD = 0.35

index  = None
labels = []
_model_loaded = False


def _warmup_model():
    global _model_loaded
    if _model_loaded:
        return
    print("⏳ Loading Facenet512 model...")
    dummy = np.zeros((160, 160, 3), dtype=np.uint8)
    try:
        DeepFace.represent(
            img_path=dummy,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
        )
    except Exception:
        pass
    _model_loaded = True
    print("✅ Facenet512 ready")


def load_index():
    global index, labels
    _warmup_model()
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        print(f"✅ FAISS index loaded ({index.ntotal} vectors)")
    else:
        print("⚠ No FAISS index — add known faces first.")
        return
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "rb") as f:
            labels = pickle.load(f)
        print(f"✅ People: {list(set(labels))}")


def build_index(known_dir="data/known_faces"):
    """
    Build FAISS index from known_faces directory.
    Uses multiple detector backends per image for robustness.
    """
    global index, labels
    _warmup_model()

    print(f"🔨 Building index from '{known_dir}'...")
    embeddings = []
    new_labels = []

    for person in sorted(os.listdir(known_dir)):
        person_dir = os.path.join(known_dir, person)
        if not os.path.isdir(person_dir):
            continue

        count = 0
        for img_file in sorted(os.listdir(person_dir)):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(person_dir, img_file)

            # Try multiple backends — opencv is fastest, skip is fallback
            emb = None
            for backend in ["opencv", "skip"]:
                try:
                    result = DeepFace.represent(
                        img_path=path,
                        model_name=MODEL_NAME,
                        detector_backend=backend,
                        enforce_detection=False,
                        align=True,
                    )
                    if result and len(result[0]["embedding"]) == EMBEDDING_DIM:
                        emb = result[0]["embedding"]
                        break
                except Exception:
                    continue

            if emb is not None:
                embeddings.append(emb)
                new_labels.append(person)
                count += 1

        print(f"  ✔ {person}: {count} embeddings")

    if not embeddings:
        print("❌ No embeddings built.")
        return

    labels = new_labels
    arr = np.array(embeddings).astype("float32")
    faiss.normalize_L2(arr)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(arr)

    faiss.write_index(index, INDEX_PATH)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(labels, f)

    print(f"✅ Index built: {len(labels)} embeddings, {len(set(labels))} people")


def get_embedding(face_img):
    """
    Extract FaceNet512 embedding from a cropped face image.
    Tries multiple strategies for robustness at distance/angle.
    """
    # Try at different sizes — larger = better for distant faces
    for size in [(160, 160), (224, 224)]:
        resized = cv2.resize(face_img, size)

        # Enhance contrast for distant/dark faces
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        resized = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        for backend in ["opencv", "skip"]:
            try:
                result = DeepFace.represent(
                    img_path=resized,
                    model_name=MODEL_NAME,
                    detector_backend=backend,
                    enforce_detection=False,
                    align=True,
                )
                if result and len(result[0]["embedding"]) == EMBEDDING_DIM:
                    return result[0]["embedding"]
            except Exception:
                continue

    return None


def recognize_face(face_img):
    """
    Recognize a single cropped face. Returns (name, confidence_percent).
    """
    global index, labels

    if index is None:
        load_index()

    if index is None or index.ntotal == 0 or len(labels) == 0:
        return "Unknown", 0

    emb_vec = get_embedding(face_img)
    if emb_vec is None:
        return "Unknown", 0

    emb = np.array([emb_vec]).astype("float32")
    faiss.normalize_L2(emb)

    # Search top-3 matches and vote
    k = min(3, index.ntotal)
    scores, indices = index.search(emb, k)

    best_score = float(scores[0][0])
    best_idx   = int(indices[0][0])
    matched    = labels[best_idx] if best_idx < len(labels) else "?"

    print(f"🔍 score={best_score:.4f}  match={matched}  threshold={COSINE_THRESHOLD}")

    if best_score >= COSINE_THRESHOLD and best_idx < len(labels):
        # Majority vote across top-3 for stability
        votes: dict[str, float] = {}
        for s, i in zip(scores[0], indices[0]):
            if i < len(labels) and float(s) >= COSINE_THRESHOLD:
                name = labels[i]
                votes[name] = votes.get(name, 0) + float(s)

        if votes:
            winner = max(votes, key=lambda n: votes[n])
            confidence = round(best_score * 100, 1)
            return winner, confidence

    return "Unknown", 0


def recognize_all_faces(frame):
    """
    Detect AND recognize ALL faces in a frame.
    Returns list of dicts: {name, confidence, box:(x,y,w,h)}
    
    Use this in camera.py instead of calling recognize_face per detected box,
    for proper multi-face support.
    """
    global index, labels

    if index is None:
        load_index()

    results = []

    try:
        # Use DeepFace to detect all faces in frame at once
        detected = DeepFace.represent(
            img_path=frame,
            model_name=MODEL_NAME,
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
        )

        for face_data in detected:
            emb_vec = face_data.get("embedding")
            region  = face_data.get("facial_area", {})

            if not emb_vec or len(emb_vec) != EMBEDDING_DIM:
                continue

            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("w", 0)
            h = region.get("h", 0)

            if index is None or index.ntotal == 0 or len(labels) == 0:
                results.append({"name": "Unknown", "confidence": 0, "box": (x, y, w, h)})
                continue

            emb = np.array([emb_vec]).astype("float32")
            faiss.normalize_L2(emb)

            k = min(3, index.ntotal)
            scores, indices_arr = index.search(emb, k)

            best_score = float(scores[0][0])
            best_idx   = int(indices_arr[0][0])

            print(f"🔍 face@({x},{y}) score={best_score:.4f} → {labels[best_idx] if best_idx < len(labels) else '?'}")

            if best_score >= COSINE_THRESHOLD and best_idx < len(labels):
                votes: dict[str, float] = {}
                for s, i in zip(scores[0], indices_arr[0]):
                    if i < len(labels) and float(s) >= COSINE_THRESHOLD:
                        name = labels[i]
                        votes[name] = votes.get(name, 0) + float(s)

                if votes:
                    winner = max(votes, key=lambda n: votes[n])
                    results.append({
                        "name": winner,
                        "confidence": round(best_score * 100, 1),
                        "box": (x, y, w, h)
                    })
                    continue

            results.append({"name": "Unknown", "confidence": 0, "box": (x, y, w, h)})

    except Exception as e:
        print(f"⚠ recognize_all_faces error: {e}")

    return results


if __name__ == "__main__":
    load_index()
    if index is None:
        print("No index. Add known faces first.")
        exit()
    print(f"\nIndex: {index.ntotal} vectors for {list(set(labels))}")
    print(f"Threshold: {COSINE_THRESHOLD}\n")
    known_dir = "data/known_faces"
    for person in sorted(os.listdir(known_dir)):
        person_dir = os.path.join(known_dir, person)
        if not os.path.isdir(person_dir):
            continue
        imgs = [f for f in sorted(os.listdir(person_dir)) if f.endswith(".jpg")]
        if not imgs:
            continue
        img = cv2.imread(os.path.join(person_dir, imgs[0]))
        name, conf = recognize_face(img)
        status = "✅" if name == person else "❌"
        print(f"  {status} {person} → '{name}' ({conf}%)")