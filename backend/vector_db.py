import faiss
import numpy as np

index = faiss.IndexFlatL2(512)

names = []


def add_face(embedding, name):

    global index, names

    index.add(np.array([embedding]).astype("float32"))

    names.append(name)


def search_face(embedding):

    if index.ntotal == 0:
        return "Unknown", 0

    D, I = index.search(np.array([embedding]).astype("float32"), 1)

    distance = D[0][0]
    idx = I[0][0]

    if distance < 0.8:
        return names[idx], 1 - distance

    return "Unknown", 0