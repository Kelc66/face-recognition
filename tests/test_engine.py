"""Fast tests that need no models, camera, or personal photos."""
import json
from pathlib import Path

import numpy as np

import facerec as fr

ROOT = Path(__file__).resolve().parent.parent


def test_identify_matches_closest_embedding():
    rng = np.random.default_rng(0)
    db = rng.normal(size=(3, 128)).astype(np.float32)
    names = ["Alice", "Bob", "Carol"]
    query = db[1] + 0.05 * rng.normal(size=128)
    name, score = fr.identify(query, names, db)
    assert name == "Bob" and score > 0.9


def test_identify_unknown_below_threshold():
    db = np.eye(128, dtype=np.float32)[:2]
    query = np.eye(128, dtype=np.float32)[5]  # orthogonal -> similarity 0
    assert fr.identify(query, ["A", "B"], db)[0] == fr.UNKNOWN


def test_identify_empty_database():
    assert fr.identify(np.ones(128), [], np.empty((0, 128))) == (fr.UNKNOWN, 0.0)


def test_database_roundtrip(tmp_path):
    names = ["Alice", "Bob"]
    emb = np.random.default_rng(1).normal(size=(2, 128)).astype(np.float32)
    results = [fr.PhotoResult("Alice/a.jpg", "ok", "Used."),
               fr.PhotoResult("Bob/x.jpg", "no_face", "Not used: no face.")]
    path = tmp_path / "db.npz"
    fr.save_database(path, names, emb, results)
    loaded_names, loaded_emb = fr.load_database(path)
    assert loaded_names == names
    np.testing.assert_allclose(loaded_emb, emb)
    loaded = fr.load_photo_results(path)
    assert loaded["Alice/a.jpg"].used and not loaded["Bob/x.jpg"].used
    assert loaded["Bob/x.jpg"].note == "Not used: no face."


def test_consistency_check_separates_same_and_different_faces():
    from facerec.engine import consistency
    rng = np.random.default_rng(2)
    person = rng.normal(size=128)
    refs = person + 0.3 * rng.normal(size=(5, 128))
    same = person + 0.3 * rng.normal(size=128)
    other = rng.normal(size=128)
    assert consistency(same, refs) > fr.engine.MIN_CONSISTENCY > consistency(other, refs)


def test_load_missing_database_is_empty(tmp_path):
    names, emb = fr.load_database(tmp_path / "nope.npz")
    assert names == [] and emb.shape == (0, 128)


def test_iter_people_skips_non_images(tmp_path):
    (tmp_path / "Alice").mkdir()
    (tmp_path / "Alice" / "a.jpg").write_bytes(b"")
    (tmp_path / "Alice" / "notes.txt").write_text("x")
    (tmp_path / "README.md").write_text("x")
    people = list(fr.iter_people(tmp_path))
    assert [(n, [p.name for p in imgs]) for n, imgs in people] == [("Alice", ["a.jpg"])]


def test_notebook_has_no_outputs():
    """Notebook outputs can contain face crops; they must be cleared before committing."""
    nb = json.loads((ROOT / "face_recognition.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert not cell.get("outputs"), "Clear notebook outputs before committing"
            assert cell.get("execution_count") is None

