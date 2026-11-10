"""
Face detection + recognition with OpenCV's YuNet (detector) and SFace (recognizer).

Pipeline:
    image -> YuNet detects faces (box + 5 landmarks)
          -> SFace aligns & crops each face to 112x112
          -> SFace turns the crop into a 128-d embedding vector
          -> embeddings are compared with cosine similarity

No training is involved: "enrolling" a person just means storing embeddings
computed from a few photos of them.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_KNOWN_DIR = ROOT / "known_faces"
DEFAULT_DB_PATH = ROOT / "face_db.npz"

# Recommended threshold for SFace cosine similarity (from the OpenCV docs).
# Higher = stricter (fewer false matches, more "Unknown").
COSINE_THRESHOLD = 0.363
EMBEDDING_DIM = 128
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UNKNOWN = "Unknown"

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "detector": {
        "file": "face_detection_yunet_2023mar.onnx",
        "url": f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    },
    "recognizer": {
        "file": "face_recognition_sface_2021dec.onnx",
        "url": f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    },
}


# --------------------------------------------------------------------------- models
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_models(models_dir: Path | str = DEFAULT_MODELS_DIR, log: Callable[[str], None] = print,
                  models: dict | None = None) -> dict:
    """Download ONNX models if they are missing and verify their SHA-256.

    `models` defaults to the detector + recognizer (OpenCV model zoo).
    Returns {key: Path}, e.g. {"detector": Path, "recognizer": Path}.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, info in (MODELS if models is None else models).items():
        path = models_dir / info["file"]
        if not path.exists():
            log(f"Downloading {info['file']} ...")
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(info["url"], tmp)
            digest = _sha256(tmp)
            if digest != info["sha256"]:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Checksum mismatch for {info['file']} (got {digest}). "
                    f"Download it manually from {info['url']} into {models_dir}."
                )
            tmp.replace(path)
        paths[key] = path
    return paths


def load_models(models_dir: Path | str = DEFAULT_MODELS_DIR, score_threshold: float = 0.8):
    """Create (detector, recognizer), downloading the models first if needed."""
    paths = ensure_models(models_dir)
    detector = cv2.FaceDetectorYN.create(
        str(paths["detector"]), "", (320, 320),
        score_threshold=score_threshold, nms_threshold=0.3, top_k=5000,
    )
    recognizer = cv2.FaceRecognizerSF.create(str(paths["recognizer"]), "")
    return detector, recognizer


# --------------------------------------------------------------------------- cameras
def camera_backend() -> int:
    """OpenCV capture backend used for webcams (DirectShow on Windows: fast to open, stable indices)."""
    return cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY


def list_cameras(max_probe: int = 6) -> list[tuple[int, str]]:
    """Return [(opencv_index, name), ...] for the cameras on this machine.

    Uses the optional `cv2-enumerate-cameras` package for real device names. Without it,
    falls back to opening indices 0..max_probe-1 (slower, names are just "Camera N").
    """
    try:
        from cv2_enumerate_cameras import enumerate_cameras
        backend = camera_backend()
        cams = enumerate_cameras(backend) if backend != cv2.CAP_ANY else enumerate_cameras()
        seen, result = set(), []
        for c in cams:
            if c.index not in seen:
                seen.add(c.index)
                result.append((c.index, c.name or f"Camera {c.index}"))
        if result:
            return sorted(result)
    except Exception:  # package missing or enumeration failed -> probe instead
        pass
    result = []
    for i in range(max_probe):
        cap = cv2.VideoCapture(i, camera_backend())
        ok = cap.isOpened()
        cap.release()
        if ok:
            result.append((i, f"Camera {i}"))
        elif result or i > 1:
            break
    return result


# --------------------------------------------------------------------------- primitives
def load_image(path: Path | str, max_side: int = 1280):
    """Read an image (handles non-ASCII Windows paths) and downscale huge photos.

    Returns a BGR image or None if it could not be read.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    scale = max_side / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def detect_faces(detector, image) -> np.ndarray:
    """Return an (N, 15) array of faces: x, y, w, h, 5 landmarks (x, y), score."""
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    return faces if faces is not None else np.empty((0, 15), dtype=np.float32)


def largest_face(faces: np.ndarray):
    """The face with the biggest box, or None."""
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def embed_face(recognizer, image, face):
    """Align + crop one detected face and return (embedding, aligned_crop)."""
    aligned = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned)
    return feature.flatten().copy(), aligned


def _normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def identify(embedding, names, embeddings, threshold: float = COSINE_THRESHOLD):
    """Compare one embedding against the database.

    Returns (name or "Unknown", best cosine similarity).
    """
    if len(embeddings) == 0:
        return UNKNOWN, 0.0
    sims = _normalize(np.asarray(embeddings)) @ _normalize(np.asarray(embedding))
    best = int(np.argmax(sims))
    score = float(sims[best])
    return (names[best] if score >= threshold else UNKNOWN), score


COLOR_KNOWN = (0, 200, 0)       # BGR green
COLOR_UNKNOWN = (0, 0, 255)     # red


def draw_label(frame, face, label: str, score: float):
    """Draw a box + "name (score)" tag on a BGR frame, in place."""
    x, y, w, h = face[:4].astype(int)
    color = COLOR_KNOWN if label != UNKNOWN else COLOR_UNKNOWN
    # Scale line width + text with the image so tags stay readable on big photos.
    k = max(1.0, max(frame.shape[:2]) / 720)
    thick, fs = max(2, int(round(2 * k))), 0.6 * k
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thick)
    text = f"{label} ({score:.2f})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
    pad = int(5 * k)
    y_top = max(y - th - 2 * pad, 0)
    cv2.rectangle(frame, (x, y_top), (x + tw + 2 * pad, y_top + th + 2 * pad), color, -1)
    cv2.putText(frame, text, (x + pad, y_top + th + pad), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick)


# --------------------------------------------------------------------------- database
def iter_people(known_dir: Path | str) -> Iterable[tuple[str, list[Path]]]:
    """Yield (person_name, [image paths]) for each sub-folder of known_dir."""
    known_dir = Path(known_dir)
    if not known_dir.exists():
        return
    for person_dir in sorted(p for p in known_dir.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        images = sorted(p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        yield person_dir.name, images


# Enrollment detection: faces scoring >= DETECT_SCORE are used directly. Harder photos (turned
# head, odd angle, partly covered) are retried down to ENROLL_MIN_SCORE and kept only if the face
# looks like the same person as their clear photos (mean cosine similarity >= MIN_CONSISTENCY).
# This adds useful extra angles without letting cut-off or wrong faces into the database.
DETECT_SCORE = 0.8
ENROLL_MIN_SCORE = 0.5
MIN_CONSISTENCY = 0.45

# PhotoResult.status values
OK, OK_HARD, SKIP_NO_FACE, SKIP_UNCLEAR, SKIP_UNREADABLE = "ok", "ok_hard", "no_face", "unclear", "unreadable"


@dataclass
class PhotoResult:
    """What happened to one enrollment photo during build_database()."""
    key: str            # "Person/file.jpg"
    status: str         # OK | OK_HARD | SKIP_NO_FACE | SKIP_UNCLEAR | SKIP_UNREADABLE
    note: str           # human-readable explanation
    det_score: float = 0.0
    consistency: float | None = None

    @property
    def used(self) -> bool:
        return self.status in (OK, OK_HARD)


def consistency(embedding, references) -> float:
    """Mean cosine similarity of one embedding to a set of reference embeddings."""
    return float(np.mean(_normalize(np.asarray(references)) @ _normalize(np.asarray(embedding))))


def build_database(known_dir, detector, recognizer, verbose: bool = True,
                   progress: Callable[[int, int, str], None] | None = None,
                   min_score: float = ENROLL_MIN_SCORE, min_consistency: float = MIN_CONSISTENCY):
    """Walk known_dir/<PersonName>/*.jpg and compute one embedding per usable image.

    The largest confidently-detected face in each image is used. If none is found, the photo is
    retried with a lower detection threshold, and that face is kept only if it is consistent
    with the person's clear photos (see MIN_CONSISTENCY).
    progress(done, total, message) is called after each image, if given.

    Returns: names (list[str]), embeddings (N x 128 float32), crops (list of BGR images),
             results (list[PhotoResult], one per photo, used or not).
    """
    people = list(iter_people(known_dir))
    total = sum(len(imgs) for _, imgs in people)
    names, embeddings, crops, results = [], [], [], []
    done = 0

    def report(msg):
        if verbose:
            print(msg)
        if progress:
            progress(done, total, msg)

    old_threshold = detector.getScoreThreshold()
    detector.setScoreThreshold(min(min_score, old_threshold))
    try:
        for person, images in people:
            clear, hard = [], []   # (PhotoResult, embedding, crop)
            for img_path in images:
                done += 1
                key = f"{person}/{img_path.name}"
                img = load_image(img_path)
                if img is None:
                    results.append(PhotoResult(key, SKIP_UNREADABLE, "Not used: this file could not be read as an image."))
                    report(f"  [skip] {key}: could not read the file")
                    continue
                faces = detect_faces(detector, img)
                confident = faces[faces[:, 14] >= DETECT_SCORE] if len(faces) else faces
                face = largest_face(confident if len(confident) else faces)
                if face is None:
                    results.append(PhotoResult(key, SKIP_NO_FACE, "Not used: no face was found in this photo."))
                    report(f"  [skip] {key}: no face found")
                    continue
                emb, crop = embed_face(recognizer, img, face)
                res = PhotoResult(key, OK, "Used.", float(face[14]))
                if len(confident) > 1:
                    res.note = f"Used. {len(confident)} faces in this photo; the largest one was used."
                    report(f"  [note] {key}: {len(confident)} faces, used the largest")
                (clear if len(confident) else hard).append((res, emb, crop))
                if progress and len(confident):
                    progress(done, total, f"  [ok] {key}")

            # Hard photos: keep only if they look like the same person as the clear ones.
            refs = [e for _, e, _ in clear]
            for res, emb, crop in hard:
                if refs:
                    res.consistency = consistency(emb, refs)
                    keep = res.consistency >= min_consistency
                else:
                    keep = True   # nothing to compare against
                if keep:
                    res.status = OK_HARD
                    res.note = f"Used. The face was hard to detect (score {res.det_score:.2f})"
                    res.note += (f" but matches {person}'s other photos ({res.consistency:.2f})."
                                 if res.consistency is not None else ".")
                    clear.append((res, emb, crop))
                    report(f"  [added] {res.key}: hard to detect, but matches the other photos")
                else:
                    res.status = SKIP_UNCLEAR
                    res.note = (f"Not used: a face was found but it is too unclear. It matches {person}'s "
                                f"other photos only {res.consistency:.2f} (needs {min_consistency:.2f}). "
                                "Usually the face is cut off, turned too far or covered.")
                    results.append(res)
                    report(f"  [skip] {res.key}: face too unclear (cut off, turned away or covered)")
            for res, emb, crop in clear:
                names.append(person)
                embeddings.append(emb)
                crops.append(crop)
                results.append(res)
    finally:
        detector.setScoreThreshold(old_threshold)

    emb_arr = np.array(embeddings, dtype=np.float32) if embeddings else np.empty((0, EMBEDDING_DIM), np.float32)
    return names, emb_arr, crops, results


def save_database(path, names, embeddings, results: list[PhotoResult] | None = None):
    """Save embeddings (+ optional per-photo results, so the app can show why photos were skipped)."""
    results = results or []
    np.savez(path, names=np.array(names, dtype=str), embeddings=np.asarray(embeddings, dtype=np.float32),
             photo_keys=np.array([r.key for r in results], dtype=str),
             photo_status=np.array([r.status for r in results], dtype=str),
             photo_notes=np.array([r.note for r in results], dtype=str))


def load_database(path):
    """Return (names, embeddings); empty if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return [], np.empty((0, EMBEDDING_DIM), np.float32)
    with np.load(path) as data:
        return [str(n) for n in data["names"]], data["embeddings"]


def load_photo_results(path) -> dict[str, PhotoResult]:
    """Per-photo results saved by save_database(), keyed by "Person/file.jpg" (empty if none)."""
    path = Path(path)
    if not path.exists():
        return {}
    with np.load(path) as data:
        if "photo_keys" not in data:
            return {}
        return {str(k): PhotoResult(str(k), str(st), str(n))
                for k, st, n in zip(data["photo_keys"], data["photo_status"], data["photo_notes"])}


# --------------------------------------------------------------------------- high-level
@dataclass
class Detection:
    box: tuple[int, int, int, int]   # x, y, w, h
    name: str
    score: float
    face: np.ndarray = field(repr=False)  # raw 15-value YuNet row

    @property
    def known(self) -> bool:
        return self.name != UNKNOWN


class FaceRecognizer:
    """Convenience wrapper: models + database + threshold in one object."""

    def __init__(self, models_dir=DEFAULT_MODELS_DIR, threshold: float = COSINE_THRESHOLD,
                 score_threshold: float = 0.8):
        self.detector, self.recognizer = load_models(models_dir, score_threshold)
        self.threshold = threshold
        self.names: list[str] = []
        self.embeddings = np.empty((0, EMBEDDING_DIM), np.float32)

    # database ------------------------------------------------------------
    def build(self, known_dir=DEFAULT_KNOWN_DIR, db_path=DEFAULT_DB_PATH, verbose=False, progress=None):
        names, emb, crops, results = build_database(known_dir, self.detector, self.recognizer,
                                                    verbose=verbose, progress=progress)
        self.names, self.embeddings = names, emb
        if db_path is not None:
            save_database(db_path, names, emb, results)
        return crops, results

    def load(self, db_path=DEFAULT_DB_PATH) -> bool:
        self.names, self.embeddings = load_database(db_path)
        return len(self.names) > 0

    @property
    def people(self) -> list[str]:
        return sorted(set(self.names))

    # inference -----------------------------------------------------------
    def recognize(self, frame) -> list[Detection]:
        """Detect + identify every face in a BGR frame."""
        results = []
        for face in detect_faces(self.detector, frame):
            emb, _ = embed_face(self.recognizer, frame, face)
            name, score = identify(emb, self.names, self.embeddings, self.threshold)
            x, y, w, h = (int(v) for v in face[:4])
            results.append(Detection((x, y, w, h), name, score, face))
        return results

    @staticmethod
    def annotate(frame, detections: list[Detection]):
        for d in detections:
            draw_label(frame, d.face, d.name, d.score)
        return frame
