# Face Recognition

Training-free face recognition for your webcam, with a desktop app and a test notebook.
It is built on OpenCV's [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) face detector and [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) face recognizer. It runs on the CPU, needs no GPU, and stays fully offline after the first model download.

## How it works

1. **Detect**: YuNet finds each face plus 5 landmarks (eyes, nose, mouth corners).
2. **Embed**: SFace aligns the face using those landmarks and turns it into a 128-number vector (an *embedding*).
3. **Match**: the embedding is compared to the embeddings of your enrolled photos with cosine similarity. If the best match is above the threshold (default `0.363`), the face gets that person's name. Otherwise it is **Unknown**.

Adding a person means dropping a few photos of them into a folder. There is no training step.

## Setup

```bash
git clone https://github.com/Kelc66/face-recognition.git
cd face-recognition
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is required. The two ONNX models (~39 MB) download into `models/` on first run, and their checksums are verified.

## Desktop app

```bash
python app.py          # or: python -m facerec
```

- **People** (left): add, rename or delete people. To add photos, use the file picker, drag and drop image files onto a person, or press **Capture** to save the current camera frame.
- **Rebuild database**: computes embeddings for all photos. The button turns orange when your photos have changed since the last build.
- **Live camera / Test a photo** (center, two modes). Faces are boxed **green with a name** when recognized and **red** when unknown.
  - **Live camera**: choose a camera from the list. Cameras are found automatically; press ⟳ to search again. Then press Start camera. Choosing another camera while the feed runs switches to it.
  - **Test a photo** (Ctrl+O): runs recognition on a single image file. You can also double-click an enrolled photo on the left. The camera pauses in the background, so **◀ Back to live camera** (Ctrl+L) resumes it instantly. Threshold changes re-run recognition on the photo.
- **Theme** (top right): Dark, Midnight, Dracula or Light. Your choice is remembered.
- **Detections and threshold** (right): live list of recognized faces and their scores. Raise the threshold if strangers get matched, lower it if you are not recognized.

## Test notebook

`face_recognition.ipynb` walks through the same pipeline step by step:

1. Load the models.
2. Build the database and view the aligned face crops.
3. Look at a pairwise similarity matrix.
4. Test on a single photo.
5. Run live webcam recognition.

Use it to check your enrollment photos and tune the threshold.

## Adding people manually

```
known_faces/
    Alice/
        1.jpg
        2.jpg
    Bob/
        photo_a.png
```

Use 3 to 10 photos per person, with different angles and lighting and the face clearly visible. When a photo contains several faces, the largest one is used.

Photos where the face is hard to detect, such as a turned head, are retried and kept only if the face matches that person's clear photos. Photos that can't be used (no face, or a face that is cut off or covered) are marked **⚠** in the app's photo grid. Hover over one to see why.

## Using it from code

```python
import cv2
import facerec as fr

rec = fr.FaceRecognizer()
rec.build("known_faces")              # or rec.load() to reuse face_db.npz
frame = fr.load_image("group.jpg")
for d in rec.recognize(frame):
    print(d.name, round(d.score, 3), d.box)
cv2.imwrite("out.jpg", rec.annotate(frame, rec.recognize(frame)))
```

## Privacy

- `known_faces/`, `face_db.npz` (embeddings are biometric data), all image files and the model weights are listed in `.gitignore`.
- The notebook must be committed **without outputs**, because outputs include face crops. `pytest` fails if they are present (*Kernel → Restart & Clear Output* before committing).
- Only recognize people who have agreed to it, and follow the privacy laws that apply to you (e.g. GDPR).

## Tests

```bash
pytest
```

## Project layout

```
app.py                  # launches the desktop app
facerec/engine.py       # detection, embeddings, matching, database
facerec/app.py          # PySide6 desktop UI
face_recognition.ipynb  # step-by-step test notebook
known_faces/            # your photos (git-ignored)
tests/                  # fast unit tests (no models or photos needed)
```

## License

MIT. See [LICENSE](LICENSE). The YuNet model is MIT-licensed and the SFace model is Apache-2.0, both from the [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo). They are downloaded at runtime and are not part of this repo.
