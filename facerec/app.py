"""
Desktop app for facerec (PySide6).

    python app.py          # or: python -m facerec

Left:   people (one folder per person in known_faces/) and their photos.
Center: live camera view (or a still image) with recognized faces boxed.
Right:  faces currently detected + similarity threshold.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QThread, Signal
from PySide6.QtGui import (
    QAction, QColor, QFont, QIcon, QImage, QImageReader, QPainter, QPainterPath, QPalette, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QComboBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QInputDialog,
    QLabel, QListView, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSizePolicy, QSlider, QSplitter, QStackedWidget, QStyledItemDelegate, QVBoxLayout, QWidget,
)

from . import engine as fr

APP_NAME = "Face Recognition"
THUMB = 96
SETTINGS_SIG_KEY = "db_signature"
VIEW_HINT = "Press  Start camera,  or switch to  Test a photo  above."
MODE_LIVE, MODE_PHOTO = "live", "photo"

# =========================================================================== themes
# Each theme: Qt palette roles + a few accent colours used in stylesheets / labels.
THEMES: dict[str, dict[str, str]] = {
    "Dark": dict(window="#1e1f22", base="#2b2d30", alt="#313338", text="#dfe1e5", muted="#8c8f96",
                 button="#393b40", border="#45474d", accent="#4c8dff", good="#4ade80", bad="#f87171",
                 warn="#f0a030", video="#111214", on_accent="#ffffff"),
    "Midnight": dict(window="#0f172a", base="#1e293b", alt="#24324a", text="#e2e8f0", muted="#94a3b8",
                     button="#273449", border="#334155", accent="#38bdf8", good="#34d399", bad="#fb7185",
                     warn="#fbbf24", video="#020617", on_accent="#0f172a"),
    "Dracula": dict(window="#21222c", base="#282a36", alt="#2f3241", text="#f8f8f2", muted="#9aa0c0",
                    button="#383a4a", border="#44475a", accent="#bd93f9", good="#50fa7b", bad="#ff5555",
                    warn="#ffb86c", video="#15161d", on_accent="#21222c"),
    "Light": dict(window="#f3f4f6", base="#ffffff", alt="#f7f8fa", text="#1f2328", muted="#6b7280",
                  button="#ffffff", border="#d0d4da", accent="#2563eb", good="#15803d", bad="#dc2626",
                  warn="#b45309", video="#1a1a1a", on_accent="#ffffff"),
}
DEFAULT_THEME = "Dark"


def apply_theme(app: QApplication, name: str) -> dict[str, str]:
    """Apply a theme (Fusion style + palette + small stylesheet) to the whole app."""
    t = THEMES.get(name, THEMES[DEFAULT_THEME])
    app.setStyle("Fusion")
    pal = QPalette()
    R, c = QPalette.ColorRole, QColor
    for role, key in [(R.Window, "window"), (R.WindowText, "text"), (R.Base, "base"),
                      (R.AlternateBase, "alt"), (R.Text, "text"), (R.Button, "button"),
                      (R.ButtonText, "text"), (R.ToolTipBase, "base"), (R.ToolTipText, "text"),
                      (R.PlaceholderText, "muted"), (R.Highlight, "accent"), (R.Link, "accent")]:
        pal.setColor(role, c(t[key]))
    pal.setColor(R.HighlightedText, c(t["on_accent"]))
    pal.setColor(R.BrightText, c(t["bad"]))
    for role in (R.WindowText, R.Text, R.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, c(t["muted"]))
    app.setPalette(pal)
    app.setStyleSheet(f"""
        QGroupBox {{ border: 1px solid {t['border']}; border-radius: 6px; margin-top: 14px; padding-top: 6px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {t['muted']}; }}
        QPushButton {{ background: {t['button']}; border: 1px solid {t['border']}; border-radius: 5px;
                       padding: 5px 10px; }}
        QPushButton:hover {{ border-color: {t['accent']}; }}
        QPushButton:pressed {{ background: {t['accent']}; color: {t['on_accent']}; }}
        QPushButton:disabled {{ color: {t['muted']}; }}
        QPushButton:checked {{ background: {t['accent']}; color: {t['on_accent']}; border-color: {t['accent']}; }}
        QPushButton#modeButton {{ padding: 6px 16px; font-weight: bold; }}
        QPushButton#backButton {{ background: {t['accent']}; color: {t['on_accent']}; border-color: {t['accent']};
                                 font-weight: bold; }}
        QComboBox {{ background: {t['button']}; border: 1px solid {t['border']}; border-radius: 5px;
                     padding: 3px 8px; }}
        QComboBox:hover {{ border-color: {t['accent']}; }}
        QComboBox:disabled {{ color: {t['muted']}; }}
        /* Dropdown lists: without explicit colours Qt reuses the combo's background for the
           hovered row and paints white text on it (invisible in the Light theme). */
        QComboBox QAbstractItemView {{ background: {t['base']}; color: {t['text']}; outline: 0;
                                       border: 1px solid {t['border']};
                                       selection-background-color: {t['accent']}; selection-color: {t['on_accent']}; }}
        QComboBox QAbstractItemView::item {{ min-height: 24px; padding: 0 8px; }}
        QComboBox QAbstractItemView::item:hover,
        QComboBox QAbstractItemView::item:selected {{ background: {t['accent']}; color: {t['on_accent']}; }}
        QListWidget, QPlainTextEdit {{ border: 1px solid {t['border']}; border-radius: 5px; }}
        QMenuBar {{ background: {t['window']}; color: {t['text']}; }}
        QMenuBar::item {{ background: transparent; padding: 4px 10px; border-radius: 4px; }}
        QMenuBar::item:selected, QMenuBar::item:pressed {{ background: {t['accent']}; color: {t['on_accent']}; }}
        QMenu {{ background: {t['base']}; color: {t['text']}; border: 1px solid {t['border']}; padding: 4px; }}
        QMenu::item {{ padding: 5px 24px 5px 16px; border-radius: 4px; }}
        QMenu::item:selected {{ background: {t['accent']}; color: {t['on_accent']}; }}
        QMenu::item:disabled {{ color: {t['muted']}; }}
        QMenu::separator {{ height: 1px; background: {t['border']}; margin: 4px 8px; }}
        QToolTip {{ background: {t['base']}; color: {t['text']}; border: 1px solid {t['border']}; padding: 4px; }}
        QProgressBar {{ border: 1px solid {t['border']}; border-radius: 4px; text-align: center; }}
        QProgressBar::chunk {{ background: {t['accent']}; border-radius: 3px; }}
        QSplitter::handle {{ background: {t['window']}; }}
    """)
    return t


# =========================================================================== helpers
_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def sanitize_person_name(raw: str) -> str | None:
    """Return a safe folder name for a person, or None if it can't be made safe."""
    name = _INVALID_CHARS.sub("", raw).strip().rstrip(". ")
    name = re.sub(r"\s+", " ", name)
    if not name or name in {".", ".."} or name.startswith((".", "_")):
        return None
    if name.split(".")[0].upper() in _RESERVED or len(name) > 64:
        return None
    return name


def unique_path(folder: Path, filename: str) -> Path:
    """folder/filename, adding _1, _2, ... if it already exists."""
    target = folder / filename
    stem, suffix, i = target.stem, target.suffix, 1
    while target.exists():
        target = folder / f"{stem}_{i}{suffix}"
        i += 1
    return target


def known_faces_signature(known_dir: Path) -> str:
    """Hash of every enrolled photo's path/size/mtime; changes when photos change."""
    h = hashlib.sha1()
    for person, images in fr.iter_people(known_dir):
        for p in images:
            st = p.stat()
            h.update(f"{person}/{p.name}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def open_in_file_manager(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


_thumb_cache: dict[tuple[str, int], QPixmap] = {}


def thumbnail(path: Path, size: int = THUMB) -> QPixmap:
    """Small, EXIF-rotated, cached preview of an image file."""
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return QPixmap()
    if key in _thumb_cache:
        return _thumb_cache[key]
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    src = reader.size()
    if src.isValid() and max(src.width(), src.height()) > size * 2:
        reader.setScaledSize(src.scaled(size * 2, size * 2, Qt.AspectRatioMode.KeepAspectRatio))
    img = reader.read()
    pix = QPixmap.fromImage(img).scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation) if not img.isNull() else QPixmap()
    _thumb_cache[key] = pix
    return pix


def with_warning_badge(pix: QPixmap, color: str) -> QPixmap:
    """Copy of a thumbnail, dimmed, with a warning triangle in the top-right corner."""
    if pix.isNull():
        return pix
    out = QPixmap(pix)
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.fillRect(out.rect(), QColor(0, 0, 0, 110))          # dim: "not used"
    s = max(18, min(out.width(), out.height()) // 3)
    x0, y0 = out.width() - s - 3, 3
    tri = QPainterPath()
    tri.moveTo(x0 + s / 2, y0)
    tri.lineTo(x0 + s, y0 + s * 0.9)
    tri.lineTo(x0, y0 + s * 0.9)
    tri.closeSubpath()
    p.setPen(QPen(QColor(0, 0, 0, 160), 1.5))
    p.setBrush(QColor(color))
    p.drawPath(tri)
    f = QFont()
    f.setBold(True)
    f.setPixelSize(int(s * 0.6))
    p.setFont(f)
    p.setPen(QColor("#1b1b1b"))
    p.drawText(int(x0), int(y0 + s * 0.22), int(s), int(s * 0.68), Qt.AlignmentFlag.AlignCenter, "!")
    p.end()
    return out


# =========================================================================== engine (thread-safe)
class SharedEngine:
    """A FaceRecognizer guarded by a lock so the camera thread and UI thread can share it."""

    def __init__(self, recognizer: fr.FaceRecognizer):
        self.rec = recognizer
        self.lock = threading.Lock()

    def recognize(self, frame: np.ndarray) -> list[fr.Detection]:
        with self.lock:
            return self.rec.recognize(frame)

    def set_threshold(self, value: float) -> None:
        with self.lock:
            self.rec.threshold = value

    def set_database(self, names: list[str], embeddings: np.ndarray) -> None:
        with self.lock:
            self.rec.names, self.rec.embeddings = list(names), embeddings

    def load(self, db_path: Path) -> bool:
        with self.lock:
            return self.rec.load(db_path)


# =========================================================================== workers
class ModelLoader(QThread):
    loaded = Signal(object)   # fr.FaceRecognizer
    failed = Signal(str)

    def __init__(self, models_dir: Path):
        super().__init__()
        self.models_dir = models_dir

    def run(self):
        try:
            self.loaded.emit(fr.FaceRecognizer(self.models_dir))
        except Exception as e:  # network errors, checksum mismatch, bad OpenCV build...
            self.failed.emit(f"{type(e).__name__}: {e}")


class CameraScanner(QThread):
    """Lists available cameras off the UI thread (probing can take a second per device)."""
    found = Signal(object)   # list[(index, name)]

    def run(self):
        try:
            self.found.emit(fr.list_cameras())
        except Exception:
            self.found.emit([])


class BuildWorker(QThread):
    """Rebuilds the embedding database with its own model instances (no lock contention)."""
    progress = Signal(int, int, str)
    done = Signal(object, object, object)   # names, embeddings, results (list[PhotoResult])
    failed = Signal(str)

    def __init__(self, models_dir: Path, known_dir: Path, db_path: Path):
        super().__init__()
        self.models_dir, self.known_dir, self.db_path = models_dir, known_dir, db_path

    def run(self):
        try:
            detector, recognizer = fr.load_models(self.models_dir)
            names, emb, _crops, results = fr.build_database(
                self.known_dir, detector, recognizer, verbose=False,
                progress=lambda d, t, m: self.progress.emit(d, t, m))
            fr.save_database(self.db_path, names, emb, results)
            self.done.emit(names, emb, results)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class CameraWorker(QThread):
    frame_ready = Signal(QImage, object, float)   # annotated frame, detections, fps
    failed = Signal(str)

    def __init__(self, index: int, engine: SharedEngine | None):
        super().__init__()
        self.index, self.engine = index, engine
        self._running = True
        # Paused = camera stays open (instant resume) but frames are not processed or shown.
        self.paused = False

        self._last_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()

    def stop(self):
        self._running = False

    def last_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return None if self._last_frame is None else self._last_frame.copy()

    def run(self):
        cap = cv2.VideoCapture(self.index, fr.camera_backend())
        try:
            if not cap.isOpened():
                self.failed.emit(f"Could not open camera {self.index}.")
                return
            prev, fps, misses = time.perf_counter(), 0.0, 0
            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    misses += 1
                    if misses > 30:
                        self.failed.emit(f"Camera {self.index} stopped delivering frames.")
                        return
                    self.msleep(30)
                    continue
                misses = 0
                with self._frame_lock:
                    self._last_frame = frame.copy()
                if self.paused:
                    prev, fps = time.perf_counter(), 0.0
                    self.msleep(15)
                    continue
                detections = self.engine.recognize(frame) if self.engine else []
                fr.FaceRecognizer.annotate(frame, detections)
                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6)) if fps else 1.0 / max(now - prev, 1e-6)
                prev = now
                self.frame_ready.emit(bgr_to_qimage(frame), detections, fps)
        finally:
            cap.release()


# =========================================================================== widgets
class VideoView(QLabel):
    """Shows a pixmap scaled to fit, keeping aspect ratio."""

    def __init__(self):
        super().__init__(VIEW_HINT)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.set_colors("#111214", "#888888")
        self._pix: QPixmap | None = None

    def set_colors(self, bg: str, fg: str):
        self.setStyleSheet(f"background:{bg}; color:{fg}; font-size:14px; border-radius:6px;")

    def set_image(self, img: QImage):
        self._pix = QPixmap.fromImage(img)
        self._rescale()

    def clear_image(self, text: str = ""):
        self._pix = None
        self.clear()
        self.setText(text or VIEW_HINT)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))


class DropListWidget(QListWidget):
    """QListWidget that accepts dropped image files: files_dropped(paths, item_under_cursor)."""
    files_dropped = Signal(list, object)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)

    @staticmethod
    def _image_paths(event) -> list[Path]:
        if not event.mimeData().hasUrls():
            return []
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        return [p for p in paths if p.is_file() and p.suffix.lower() in fr.IMAGE_EXTS]

    def dragEnterEvent(self, e):
        e.acceptProposedAction() if self._image_paths(e) else e.ignore()

    def dragMoveEvent(self, e):
        e.acceptProposedAction() if self._image_paths(e) else e.ignore()

    def dropEvent(self, e):
        paths = self._image_paths(e)
        if paths:
            self.files_dropped.emit(paths, self.itemAt(e.position().toPoint()))
            e.acceptProposedAction()


# =========================================================================== main window
class MainWindow(QMainWindow):
    models_ready = Signal()
    build_finished = Signal(bool)

    def __init__(self, known_dir: Path = fr.DEFAULT_KNOWN_DIR, db_path: Path = fr.DEFAULT_DB_PATH,
                 models_dir: Path = fr.DEFAULT_MODELS_DIR):
        super().__init__()
        self.known_dir, self.db_path, self.models_dir = Path(known_dir), Path(db_path), Path(models_dir)
        self.known_dir.mkdir(parents=True, exist_ok=True)
        self.settings = QSettings("facerec", "FaceRecognition")
        self.engine: SharedEngine | None = None
        self.camera: CameraWorker | None = None
        self.builder: BuildWorker | None = None
        self.loader: ModelLoader | None = None
        self._dirty = False
        self.scanner: CameraScanner | None = None
        self._last_detections: list[fr.Detection] | None = None
        try:
            self._photo_results: dict[str, fr.PhotoResult] = fr.load_photo_results(self.db_path)
        except Exception:
            self._photo_results = {}
        self.mode = MODE_LIVE
        self._photo_path: Path | None = None
        self.theme: dict[str, str] = THEMES[DEFAULT_THEME]

        self.setWindowTitle(APP_NAME)
        self.resize(1400, 850)
        self._build_ui()
        self.set_theme(str(self.settings.value("theme", DEFAULT_THEME)))
        self.refresh_people()
        self.refresh_cameras()
        self._set_engine_enabled(False)
        self._load_models()

    # ------------------------------------------------------------------ UI construction
    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_people_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([320, 800, 280])
        self.setCentralWidget(splitter)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)

        menu = self.menuBar().addMenu("&File")
        for text, slot, shortcut in [("Test recognition on a photo...", self.choose_photo, "Ctrl+O"),
                                     ("Back to live camera", self.enter_live_mode, "Ctrl+L"),
                                     ("Open known_faces folder", lambda: open_in_file_manager(self.known_dir), None),
                                     ("Quit", self.close, "Ctrl+Q")]:
            act = QAction(text, self)
            act.triggered.connect(slot)
            if shortcut:
                act.setShortcut(shortcut)
            menu.addAction(act)

        # Parent + attribute keep the widget alive: PySide does not take ownership in setCornerWidget.
        self._corner = corner = QWidget(self.menuBar())
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(0, 2, 8, 2)
        cl.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(THEMES))
        self.theme_combo.setToolTip("Colour theme")
        self.theme_combo.currentTextChanged.connect(self.set_theme)
        cl.addWidget(self.theme_combo)
        self.menuBar().setCornerWidget(corner, Qt.Corner.TopRightCorner)
        self._fix_combo_popups()

    def _fix_combo_popups(self):
        # QComboBox's default popup delegate ignores the ::item:hover/:selected stylesheet rules
        # (the hovered row came out white-on-white in the Light theme). A standard delegate honours them.
        for combo in self.findChildren(QComboBox):
            combo.setItemDelegate(QStyledItemDelegate(combo))

    def _build_people_panel(self) -> QWidget:
        box = QGroupBox("People")
        lay = QVBoxLayout(box)

        self.people_list = DropListWidget()
        self.people_list.setIconSize(QSize(48, 48))
        self.people_list.setToolTip("Drop image files on a person to add them")
        self.people_list.currentItemChanged.connect(lambda *_: self.refresh_photos())
        self.people_list.files_dropped.connect(self._on_people_drop)
        lay.addWidget(self.people_list, 2)

        row1, row2 = QHBoxLayout(), QHBoxLayout()
        self.btn_add_person = self._button("Add person", self.add_person, row1)
        self.btn_rename = self._button("Rename", self.rename_person, row1)
        self.btn_delete = self._button("Delete", self.delete_person, row1)
        self.btn_add_photos = self._button("Add photos...", self.add_photos, row2)
        self.btn_capture = self._button("Capture", self.capture_photo, row2,
                                        tip="Save the current camera frame to the selected person")
        self.btn_open_folder = self._button("Open folder", self.open_person_folder, row2)
        lay.addLayout(row1)
        lay.addLayout(row2)

        lay.addWidget(QLabel("Photos (drop images here):"))
        self.photo_list = DropListWidget()
        self.photo_list.setViewMode(QListView.ViewMode.IconMode)
        self.photo_list.setIconSize(QSize(THUMB, THUMB))
        self.photo_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.photo_list.setMovement(QListView.Movement.Static)
        self.photo_list.setSpacing(4)
        self.photo_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.photo_list.files_dropped.connect(lambda paths, _item: self._copy_photos(self._current_person(), paths))
        self.photo_list.itemDoubleClicked.connect(
            lambda it: self.enter_photo_mode(Path(it.data(Qt.ItemDataRole.UserRole))))
        self.photo_list.setToolTip("Double-click a photo to test recognition on it")
        lay.addWidget(self.photo_list, 3)

        row3 = QHBoxLayout()
        self.btn_remove_photo = self._button("Remove selected photos", self.remove_photos, row3)
        lay.addLayout(row3)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        lay.addWidget(sep)
        self.btn_rebuild = self._button("Rebuild database", self.start_rebuild, lay)
        self.dirty_label = QLabel()
        self.dirty_label.setWordWrap(True)
        self.dirty_label.setStyleSheet("color:#c77700;")
        lay.addWidget(self.dirty_label)
        box.setMinimumWidth(280)
        return box

    def _build_center_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        # Mode switch: live webcam vs. testing a single photo.
        modes = QHBoxLayout()
        modes.setSpacing(10)
        self.btn_mode_live = QPushButton("\u25cf  Live camera")
        self.btn_mode_photo = QPushButton("\U0001f5bc  Test a photo")
        self.btn_mode_live.setToolTip("Recognize faces in the webcam feed (Ctrl+L)")
        self.btn_mode_photo.setToolTip("Run recognition on a single photo file. The camera is paused, not closed (Ctrl+O)")
        self.mode_group = QButtonGroup(self)
        for b in (self.btn_mode_live, self.btn_mode_photo):
            b.setObjectName("modeButton")
            b.setCheckable(True)
            self.mode_group.addButton(b)
            modes.addWidget(b)
        self.btn_mode_live.setChecked(True)
        self.btn_mode_live.clicked.connect(self.enter_live_mode)
        self.btn_mode_photo.clicked.connect(lambda: self.enter_photo_mode())
        modes.addStretch(1)
        self.fps_label = QLabel("")
        modes.addWidget(self.fps_label)
        lay.addLayout(modes)

        # Per-mode controls.
        self.mode_bars = QStackedWidget()
        live_bar = QWidget()
        controls = QHBoxLayout(live_bar)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(QLabel("Camera:"))
        self.cam_combo = QComboBox()
        self.cam_combo.setMinimumWidth(200)
        self.cam_combo.setToolTip("Cameras found on this computer. Switching while running restarts the feed.")
        self.cam_combo.activated.connect(self._on_camera_picked)
        controls.addWidget(self.cam_combo)
        self.btn_cam_refresh = self._button("\u27f3", self.refresh_cameras, controls, tip="Search for cameras again")
        self.btn_cam_refresh.setFixedWidth(34)
        self.btn_camera = self._button("Start camera", self.toggle_camera, controls)
        controls.addStretch(1)
        self.mode_bars.addWidget(live_bar)

        photo_bar = QWidget()
        pb = QHBoxLayout(photo_bar)
        pb.setContentsMargins(0, 0, 0, 0)
        self.btn_choose_photo = self._button("Choose photo...", self.choose_photo, pb,
                                             tip="Pick another photo to test (Ctrl+O)")
        self.photo_label = QLabel("")
        self.photo_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        pb.addWidget(self.photo_label, 1)
        self.btn_back_live = self._button("\u25c0  Back to live camera", self.enter_live_mode, pb,
                                          tip="Return to the webcam feed (Ctrl+L)")
        self.btn_back_live.setObjectName("backButton")
        self.mode_bars.addWidget(photo_bar)
        self.mode_bars.setFixedHeight(live_bar.sizeHint().height() + 4)
        lay.addWidget(self.mode_bars)

        self.view = VideoView()
        lay.addWidget(self.view, 1)
        return w

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        det_box = QGroupBox("Detected faces")
        dl = QVBoxLayout(det_box)
        self.det_list = QListWidget()
        dl.addWidget(self.det_list)
        lay.addWidget(det_box, 2)

        thr_box = QGroupBox("Match threshold")
        tl = QVBoxLayout(thr_box)
        self.thr_slider = QSlider(Qt.Orientation.Horizontal)
        self.thr_slider.setRange(200, 700)
        saved = float(self.settings.value("threshold", fr.COSINE_THRESHOLD))
        self.thr_slider.setValue(int(round(saved * 1000)))
        self.thr_label = QLabel()
        self.thr_slider.valueChanged.connect(self._on_threshold)
        tl.addWidget(self.thr_slider)
        tl.addWidget(self.thr_label)
        hint = QLabel("Higher = stricter (more \"Unknown\"). Default 0.363.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:gray;")
        tl.addWidget(hint)
        reset = QPushButton("Reset to default")
        reset.clicked.connect(lambda: self.thr_slider.setValue(int(round(fr.COSINE_THRESHOLD * 1000))))
        tl.addWidget(reset)
        lay.addWidget(thr_box)

        log_box = QGroupBox("Log")
        ll = QVBoxLayout(log_box)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        ll.addWidget(self.log)
        lay.addWidget(log_box, 2)
        self._on_threshold(self.thr_slider.value())
        w.setMinimumWidth(240)
        return w

    @staticmethod
    def _button(text, slot, layout, tip: str | None = None) -> QPushButton:
        b = QPushButton(text)
        b.clicked.connect(slot)
        if tip:
            b.setToolTip(tip)
        layout.addWidget(b)
        return b

    # ------------------------------------------------------------------ small utilities
    def status(self, msg: str, timeout: int = 5000):
        self.statusBar().showMessage(msg, timeout)

    def log_line(self, msg: str):
        self.log.appendPlainText(msg)

    def _set_engine_enabled(self, on: bool):
        for b in (self.btn_camera, self.btn_mode_photo, self.btn_choose_photo, self.btn_rebuild):
            b.setEnabled(on)

    def _current_person(self) -> str | None:
        it = self.people_list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it else None

    def _require_person(self) -> str | None:
        name = self._current_person()
        if not name:
            QMessageBox.information(self, APP_NAME, "Select a person first (or add one).")
        return name

    # ------------------------------------------------------------------ models
    def _load_models(self):
        self.status("Loading models (first run downloads ~39 MB from the OpenCV model zoo)...", 0)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.loader = ModelLoader(self.models_dir)
        self.loader.loaded.connect(self._on_models_loaded)
        self.loader.failed.connect(self._on_models_failed)
        self.loader.start()

    def _on_models_loaded(self, rec: fr.FaceRecognizer):
        rec.threshold = self.thr_slider.value() / 1000
        self.engine = SharedEngine(rec)
        self.progress.hide()
        self._set_engine_enabled(True)
        has_db = self.engine.load(self.db_path)
        if has_db:
            self.log_line(f"Loaded database: {len(rec.names)} embeddings, {len(rec.people)} people")
        self.status("Models loaded")
        self.models_ready.emit()
        self._check_dirty(auto_build=True)

    def _on_models_failed(self, err: str):
        self.progress.hide()
        self.status("Model loading failed", 0)
        self.log_line(f"[error] {err}")
        QMessageBox.critical(
            self, APP_NAME,
            "Could not load the face models.\n\n"
            f"{err}\n\nCheck your internet connection (they are downloaded on first run), "
            f"or place the .onnx files in:\n{self.models_dir}")

    # ------------------------------------------------------------------ people / photos
    def refresh_people(self, select: str | None = None):
        select = select or self._current_person()
        self.people_list.blockSignals(True)
        self.people_list.clear()
        for name, images in fr.iter_people(self.known_dir):
            label = f"{name}  ({len(images)} photo{'s' if len(images) != 1 else ''}"
            skipped = sum(1 for p in images if not self._photo_used(name, p))
            label += f",  \u26a0 {skipped} not used)" if skipped else ")"
            it = QListWidgetItem(label)
            it.setData(Qt.ItemDataRole.UserRole, name)
            if skipped:
                it.setToolTip(f"{skipped} photo(s) of {name} could not be used. "
                              "They are marked with \u26a0 in the photo grid; hover over one to see why.")
            if images:
                it.setIcon(QIcon(thumbnail(images[0], 48)))
            self.people_list.addItem(it)
            if name == select:
                self.people_list.setCurrentItem(it)
        if self.people_list.currentItem() is None and self.people_list.count():
            self.people_list.setCurrentRow(0)
        self.people_list.blockSignals(False)
        self.refresh_photos()
        self._check_dirty()

    def refresh_photos(self):
        self.photo_list.clear()
        name = self._current_person()
        has = name is not None
        for b in (self.btn_rename, self.btn_delete, self.btn_add_photos, self.btn_open_folder,
                  self.btn_remove_photo, self.btn_capture):
            b.setEnabled(has)
        if not has:
            return
        for p in sorted(q for q in (self.known_dir / name).iterdir() if q.suffix.lower() in fr.IMAGE_EXTS):
            res = self._photo_results.get(f"{name}/{p.name}")
            pix = thumbnail(p)
            if res is not None and not res.used:
                pix = with_warning_badge(pix, self.theme["warn"])
                tip = f"\u26a0  {res.note}\n\nTip: use a clear, front-facing photo, or remove this one."
            elif res is not None and res.status == fr.OK_HARD:
                tip = res.note
            elif res is None and self.db_path.exists():
                tip = "Not in the database yet. Press Rebuild database to include it."
            else:
                tip = "Used."
            it = QListWidgetItem(QIcon(pix), p.name)
            it.setData(Qt.ItemDataRole.UserRole, str(p))
            it.setToolTip(f"<b>{p.name}</b><br>{tip.replace(chr(10), '<br>')}"
                          "<br><br><i>Double-click to test recognition on this photo.</i>")
            if res is not None and not res.used:
                it.setForeground(QColor(self.theme["warn"]))
            self.photo_list.addItem(it)

    def _photo_used(self, person: str, path: Path) -> bool:
        """False only if the last build looked at this photo and could not use it."""
        res = self._photo_results.get(f"{person}/{path.name}")
        return res is None or res.used

    def add_person(self):
        raw, ok = QInputDialog.getText(self, "Add person", "Name:")
        if not ok or not raw.strip():
            return
        name = sanitize_person_name(raw)
        if not name:
            QMessageBox.warning(self, APP_NAME, f"'{raw}' is not a valid name.")
            return
        folder = self.known_dir / name
        if folder.exists():
            QMessageBox.information(self, APP_NAME, f"'{name}' already exists.")
        else:
            folder.mkdir(parents=True)
            self.log_line(f"Added person '{name}'")
        self.refresh_people(select=name)

    def rename_person(self):
        old = self._require_person()
        if not old:
            return
        raw, ok = QInputDialog.getText(self, "Rename person", "New name:", text=old)
        if not ok or raw.strip() == old:
            return
        new = sanitize_person_name(raw)
        if not new:
            QMessageBox.warning(self, APP_NAME, f"'{raw}' is not a valid name.")
            return
        target = self.known_dir / new
        if target.exists() and target.resolve() != (self.known_dir / old).resolve():
            QMessageBox.warning(self, APP_NAME, f"'{new}' already exists.")
            return
        try:
            (self.known_dir / old).rename(target)
        except OSError as e:
            QMessageBox.warning(self, APP_NAME, f"Rename failed: {e}")
            return
        self.log_line(f"Renamed '{old}' -> '{new}'")
        self.refresh_people(select=new)

    def delete_person(self):
        name = self._require_person()
        if not name:
            return
        folder = self.known_dir / name
        n = sum(1 for p in folder.iterdir() if p.is_file())
        if QMessageBox.question(self, "Delete person",
                                f"Delete '{name}' and their folder with {n} file(s)?\nThis cannot be undone."
                                ) != QMessageBox.StandardButton.Yes:
            return
        try:
            shutil.rmtree(folder)
        except OSError as e:
            QMessageBox.warning(self, APP_NAME, f"Delete failed: {e}")
            return
        self.log_line(f"Deleted '{name}'")
        self.people_list.setCurrentItem(None)
        self.refresh_people()

    def add_photos(self):
        name = self._require_person()
        if not name:
            return
        exts = " ".join(f"*{e}" for e in sorted(fr.IMAGE_EXTS))
        files, _ = QFileDialog.getOpenFileNames(self, f"Add photos of {name}", "", f"Images ({exts})")
        if files:
            self._copy_photos(name, [Path(f) for f in files])

    def _on_people_drop(self, paths: list[Path], item):
        name = item.data(Qt.ItemDataRole.UserRole) if item else self._current_person()
        if not name:
            QMessageBox.information(self, APP_NAME, "Drop the photos onto a person (add one first).")
            return
        self._copy_photos(name, paths)

    def _copy_photos(self, name: str | None, paths: list[Path]):
        if not name:
            QMessageBox.information(self, APP_NAME, "Select a person first.")
            return
        folder = self.known_dir / name
        copied = 0
        for src in paths:
            if src.suffix.lower() not in fr.IMAGE_EXTS or not src.is_file():
                continue
            if src.resolve().parent == folder.resolve():
                continue  # already there
            try:
                shutil.copy2(src, unique_path(folder, src.name))
                copied += 1
            except OSError as e:
                self.log_line(f"[error] could not copy {src.name}: {e}")
        self.log_line(f"Added {copied} photo(s) to '{name}'")
        self.refresh_people(select=name)

    def remove_photos(self):
        items = self.photo_list.selectedItems()
        if not items:
            QMessageBox.information(self, APP_NAME, "Select one or more photos first.")
            return
        if QMessageBox.question(self, "Remove photos", f"Delete {len(items)} photo(s) from disk?"
                                ) != QMessageBox.StandardButton.Yes:
            return
        for it in items:
            try:
                Path(it.data(Qt.ItemDataRole.UserRole)).unlink()
            except OSError as e:
                self.log_line(f"[error] {e}")
        self.refresh_people()

    def capture_photo(self):
        name = self._require_person()
        if not name:
            return
        if self.mode != MODE_LIVE:
            QMessageBox.information(self, APP_NAME, "Switch back to Live camera to capture a photo.")
            return
        frame = self.camera.last_frame() if self.camera and self.camera.isRunning() else None
        if frame is None:
            QMessageBox.information(self, APP_NAME, "Start the camera first, then press Capture.")
            return
        path = unique_path(self.known_dir / name, time.strftime("capture_%Y%m%d_%H%M%S.jpg"))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            QMessageBox.warning(self, APP_NAME, "Could not encode the frame.")
            return
        buf.tofile(str(path))  # tofile handles non-ASCII paths on Windows
        self.log_line(f"Captured {path.name} for '{name}'")
        self.status(f"Saved {path.name}")
        self.refresh_people(select=name)

    def open_person_folder(self):
        name = self._require_person()
        if name:
            open_in_file_manager(self.known_dir / name)

    # ------------------------------------------------------------------ database
    def _check_dirty(self, auto_build: bool = False):
        sig = known_faces_signature(self.known_dir)
        has_photos = any(imgs for _, imgs in fr.iter_people(self.known_dir))
        # A database from an older version has no per-photo results: rebuild it once.
        self._dirty = has_photos and (not self.db_path.exists() or not self._photo_results
                                      or self.settings.value(SETTINGS_SIG_KEY) != sig)
        if not self.db_path.exists() and not has_photos:
            self.dirty_label.setText("Add a person and some photos, then rebuild the database.")
        elif self._dirty:
            self.dirty_label.setText("Photos changed since the last build. Rebuild to apply.")
        else:
            self.dirty_label.setText("")
        self.btn_rebuild.setStyleSheet(f"font-weight:bold; color:{self.theme['warn']};" if self._dirty else "")
        if auto_build and self._dirty and self.engine is not None:
            self.log_line("Database is out of date: rebuilding automatically")
            self.start_rebuild()

    def start_rebuild(self):
        if self.builder and self.builder.isRunning():
            return
        self._build_sig = known_faces_signature(self.known_dir)
        self.btn_rebuild.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.status("Rebuilding database...", 0)
        self.builder = BuildWorker(self.models_dir, self.known_dir, self.db_path)
        self.builder.progress.connect(self._on_build_progress)
        self.builder.done.connect(self._on_build_done)
        self.builder.failed.connect(self._on_build_failed)
        self.builder.start()

    def _on_build_progress(self, done: int, total: int, msg: str):
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        if "[skip]" in msg or "[note]" in msg or "[added]" in msg:
            self.log_line(msg.strip())

    def _on_build_done(self, names, emb, results):
        if self.engine:
            self.engine.set_database(names, emb)
        self._photo_results = {r.key: r for r in results}
        skipped = [r for r in results if not r.used]
        self.settings.setValue(SETTINGS_SIG_KEY, self._build_sig)
        self.progress.hide()
        self.btn_rebuild.setEnabled(True)
        people = sorted(set(names))
        msg = f"Database rebuilt: {len(names)} embeddings for {len(people)} people"
        if skipped:
            msg += f" ({len(skipped)} photo(s) not used, marked \u26a0 in the photo grid)"
        self.log_line(msg)
        self.status(msg)
        self.refresh_people()
        self._refresh_photo()
        self.build_finished.emit(True)

    def _on_build_failed(self, err: str):
        self.progress.hide()
        self.btn_rebuild.setEnabled(True)
        self.log_line(f"[error] rebuild failed: {err}")
        QMessageBox.critical(self, APP_NAME, f"Rebuilding the database failed:\n{err}")
        self.build_finished.emit(False)

    # ------------------------------------------------------------------ recognition views
    def _on_threshold(self, v: int):
        t = v / 1000
        self.thr_label.setText(f"Threshold: {t:.3f}")
        self.settings.setValue("threshold", t)
        if self.engine:
            self.engine.set_threshold(t)
            self._refresh_photo()

    def _show_detections(self, detections: list[fr.Detection]):
        self._last_detections = detections
        self.det_list.clear()
        for d in sorted(detections, key=lambda d: -d.score):
            it = QListWidgetItem(f"{d.name}   {d.score:.2f}")
            it.setForeground(QColor(self.theme["good"] if d.known else self.theme["bad"]))
            self.det_list.addItem(it)
        if not detections:
            self.det_list.addItem(QListWidgetItem("(no faces)"))

    # ------------------------------------------------------------------ theme
    def set_theme(self, name: str):
        if name not in THEMES:
            name = DEFAULT_THEME
        self.theme = apply_theme(QApplication.instance(), name)
        self.settings.setValue("theme", name)
        if self.theme_combo.currentText() != name:
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentText(name)
            self.theme_combo.blockSignals(False)
        self.view.set_colors(self.theme["video"], self.theme["muted"])
        self.dirty_label.setStyleSheet(f"color:{self.theme['warn']};")
        self.btn_rebuild.setStyleSheet(f"font-weight:bold; color:{self.theme['warn']};" if self._dirty else "")
        if self._last_detections is not None:
            self._show_detections(self._last_detections)
        if self.people_list.count():
            self.refresh_photos()

    # ------------------------------------------------------------------ camera list
    def refresh_cameras(self):
        if self.scanner and self.scanner.isRunning():
            return
        self.cam_combo.clear()
        self.cam_combo.addItem("Searching for cameras...", None)
        self.cam_combo.setEnabled(False)
        self.btn_cam_refresh.setEnabled(False)
        self.scanner = CameraScanner()
        self.scanner.found.connect(self._on_cameras_found)
        self.scanner.start()

    def _on_cameras_found(self, cams: list[tuple[int, str]]):
        self.cam_combo.clear()
        self.btn_cam_refresh.setEnabled(not (self.camera and self.camera.isRunning()))
        if not cams:
            self.cam_combo.addItem("No camera found", None)
            self.log_line("[camera] no cameras found")
            return
        for idx, name in cams:
            self.cam_combo.addItem(f"{name}  ({idx})", idx)
        self.cam_combo.setEnabled(True)
        # Re-select the last used camera: by name first (indices can shift when devices change).
        last_name = str(self.settings.value("camera_name", ""))
        last_idx = self.settings.value("camera_index")
        pos = self.cam_combo.findText(last_name) if last_name else -1
        if pos < 0 and last_idx is not None:
            pos = self.cam_combo.findData(int(last_idx))
        self.cam_combo.setCurrentIndex(max(pos, 0))
        self.log_line("[camera] found: " + ", ".join(n for _, n in cams))

    def _selected_camera(self) -> int | None:
        return self.cam_combo.currentData()

    def _on_camera_picked(self, _pos: int):
        """Clicking a camera while the feed runs switches to it immediately."""
        if self.camera and self.camera.isRunning() and self._selected_camera() != self.camera.index:
            self.start_camera()

    def toggle_camera(self):
        if self.camera and self.camera.isRunning():
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        self.stop_camera()
        idx = self._selected_camera()
        if idx is None:
            QMessageBox.information(self, APP_NAME, "No camera found. Connect one and press \u27f3.")
            return
        self.settings.setValue("camera_index", idx)
        self.settings.setValue("camera_name", self.cam_combo.currentText())
        self.camera = CameraWorker(idx, self.engine)
        self.camera.paused = self.mode != MODE_LIVE
        self.camera.frame_ready.connect(self._on_frame)
        self.camera.failed.connect(self._on_camera_failed)
        self.camera.finished.connect(self._on_camera_finished)
        self.camera.start()
        self.btn_camera.setText("Stop camera")
        self.btn_cam_refresh.setEnabled(False)
        if self.mode == MODE_LIVE:
            self.view.clear_image("Starting camera...")
        self.status(f"{self.cam_combo.currentText()} starting...")

    def stop_camera(self):
        had_camera = self.camera is not None
        if self.camera:
            self.camera.stop()
            self.camera.wait(3000)
            self.camera = None
        self.btn_camera.setText("Start camera")
        self.btn_cam_refresh.setEnabled(True)
        self.fps_label.setText("")
        if had_camera and self.mode == MODE_LIVE:
            self.view.clear_image("Camera stopped. Press  Start camera  to resume.")
            self.det_list.clear()

    def _on_frame(self, img: QImage, detections, fps: float):
        if self.mode != MODE_LIVE:   # a frame queued just before we switched to photo mode
            return
        self.view.set_image(img)
        self._show_detections(detections)
        self.fps_label.setText(f"FPS {fps:.0f}")

    def _on_camera_failed(self, err: str):
        self.log_line(f"[camera] {err}")
        QMessageBox.warning(self, APP_NAME, f"{err}\n\nPick another camera from the list.")

    def _on_camera_finished(self):
        if self.camera and not self.camera.isRunning():
            self.camera = None
            self.btn_camera.setText("Start camera")
            self.btn_cam_refresh.setEnabled(True)
            self.fps_label.setText("")

    # ------------------------------------------------------------------ modes
    def _set_mode_ui(self, mode: str):
        self.mode = mode
        self.mode_bars.setCurrentIndex(0 if mode == MODE_LIVE else 1)
        (self.btn_mode_live if mode == MODE_LIVE else self.btn_mode_photo).setChecked(True)

    def choose_photo(self) -> bool:
        """Ask for a photo and show it in Test-a-photo mode. Returns False if cancelled."""
        if self.engine is None:
            return False
        exts = " ".join(f"*{e}" for e in sorted(fr.IMAGE_EXTS))
        start = str(self._photo_path.parent) if self._photo_path else str(self.settings.value("photo_dir", ""))
        file, _ = QFileDialog.getOpenFileName(self, "Test recognition on a photo", start, f"Images ({exts})")
        if not file:
            return False
        self.settings.setValue("photo_dir", str(Path(file).parent))
        return self.enter_photo_mode(Path(file))

    def enter_photo_mode(self, path: Path | None = None) -> bool:
        """Switch to Test-a-photo mode. The camera keeps running in the background, paused."""
        if self.engine is None:
            return False
        if path is None:
            if self._photo_path is None or not self._photo_path.exists():
                if not self.choose_photo():
                    self._set_mode_ui(self.mode)   # dialog cancelled: stay where we were
                    return False
                return True
            path = self._photo_path
        img = fr.load_image(path)
        if img is None:
            QMessageBox.warning(self, APP_NAME, f"Could not read {path.name}.")
            self._set_mode_ui(self.mode)
            return False
        self._photo_path = path
        self._photo_img = img
        if self.camera:
            self.camera.paused = True
        self._set_mode_ui(MODE_PHOTO)
        self.fps_label.setText("")
        self._refresh_photo()
        return True

    def _refresh_photo(self):
        """(Re)run recognition on the current photo, e.g. after a threshold change."""
        if self.mode != MODE_PHOTO or self.engine is None or getattr(self, "_photo_img", None) is None:
            return
        img = self._photo_img.copy()
        detections = self.engine.recognize(img)
        self.view.set_image(bgr_to_qimage(fr.FaceRecognizer.annotate(img, detections)))
        self._show_detections(detections)
        n = len(detections)
        self.photo_label.setText(f"  {self._photo_path.name}   \u2014   {n} face{'s' if n != 1 else ''}")
        self.photo_label.setToolTip(str(self._photo_path))

    def enter_live_mode(self):
        """Back to the webcam feed: resume the paused camera, or start it."""
        self._set_mode_ui(MODE_LIVE)
        self.det_list.clear()
        if self.camera and self.camera.isRunning():
            self.camera.paused = False
            self.view.clear_image("Resuming camera...")
        elif self.engine is not None and self._selected_camera() is not None:
            self.view.clear_image("Starting camera...")
            self.start_camera()
        else:
            self.view.clear_image()

    # ------------------------------------------------------------------ shutdown
    def closeEvent(self, e):
        self.stop_camera()
        for worker in (self.builder, self.loader, self.scanner):
            if worker and worker.isRunning():
                worker.wait(10000)
        super().closeEvent(e)


def main(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    apply_theme(app, str(QSettings("facerec", "FaceRecognition").value("theme", DEFAULT_THEME)))
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
