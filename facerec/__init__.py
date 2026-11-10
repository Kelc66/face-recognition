"""facerec: small, training-free face recognition with OpenCV YuNet + SFace."""
from .engine import (  # noqa: F401
    COSINE_THRESHOLD, DEFAULT_DB_PATH, DEFAULT_KNOWN_DIR, DEFAULT_MODELS_DIR, IMAGE_EXTS, UNKNOWN,
    Detection, FaceRecognizer, PhotoResult, build_database, load_photo_results, camera_backend, detect_faces, draw_label, embed_face,
    ensure_models, list_cameras,
    identify, iter_people, largest_face, load_database, load_image, load_models, save_database,
)

__version__ = "0.1.0"
