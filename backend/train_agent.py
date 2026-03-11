from deepface import DeepFace
import os

def rebuild_database():
    """
    Rebuilds the face database embeddings for all known faces
    """
    DeepFace.find(
        img_path="data/known_faces",
        db_path="data/known_faces",
        enforce_detection=False
    )