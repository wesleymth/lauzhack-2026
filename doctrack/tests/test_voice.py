"""Voice notes: pointer -> document coords, sentence placement, PDF export."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from doctrack.tracker import DocumentTracker
from doctrack.voice import Note, PointerLog, Recording, VoiceNotes, export_pdf, to_view

pypdf = pytest.importorskip("pypdf")


def locked_tracker(corners):
    tr = DocumentTracker(detector=None)
    tr._lock = SimpleNamespace(corners=np.asarray(corners, np.float32), scan_size=(420, 594))
    return tr


QUAD = [[100, 100], [500, 120], [520, 700], [80, 680]]   # TL TR BR BL in the image


def test_to_doc_corners_and_outside():
    tr = locked_tracker(QUAD)
    for (x, y), uv in zip(QUAD, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        got = tr.to_doc(x + (0.01 if uv[0] == 0 else -0.01), y + (0.01 if uv[1] == 0 else -0.01))
        assert got == pytest.approx(uv, abs=1e-3)
    assert tr.to_doc(10, 10) is None
    assert DocumentTracker(detector=None).to_doc(1, 1) is None


def test_view_transform_follows_rotate_and_mirror():
    tr = locked_tracker(QUAD)
    uv = tr.to_doc(200, 200)
    img_pt = np.array([200.0, 200.0])
    for op in ["rotate", "rotate", "mirror", "rotate"]:
        tr.rotate(1) if op == "rotate" else tr.toggle_mirror()
        # the same image point, measured in the CURRENT oriented corners, must match
        oriented = tr._lock.corners[tr._perm]
        import cv2
        H = cv2.getPerspectiveTransform(oriented, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32))
        p = H @ np.array([*img_pt, 1.0])
        assert to_view(uv, tr.view_transform()) == pytest.approx((p[0] / p[2], p[1] / p[2]), abs=1e-4)


def test_export_pdf_annotations(tmp_path):
    scan = np.full((594, 420, 3), 255, np.uint8)
    notes = [Note(1, "Fix this title.", time.time(), time.time() + 1, (0.25, 0.1)),
             Note(2, "Unplaced remark.", time.time(), time.time() + 1, None)]
    path = export_pdf(scan, notes, tmp_path / "out.pdf", paper="a4")
    page = pypdf.PdfReader(str(path)).pages[0]
    w, h = float(page.mediabox.width), float(page.mediabox.height)
    assert (w, h) == pytest.approx((595.3, 841.9), abs=1.5)        # A4 in points
    annots = [a.get_object() for a in page["/Annots"]]
    assert [a["/Subtype"] for a in annots] == ["/Text", "/Text"]
    assert annots[0]["/Contents"] == "Fix this title."
    x0, y0, x1, y1 = [float(v) for v in annots[0]["/Rect"]]
    assert ((x0 + x1) / 2, (y0 + y1) / 2) == pytest.approx((0.25 * w, 0.9 * h), abs=0.5)
    assert annots[1]["/Contents"].startswith("(no pointer)")
    assert (tmp_path / "out.json").exists()


def test_recording_pipeline_with_fake_mic_and_model(tmp_path, monkeypatch):
    """Recording -> chunks -> stt.transcriber_worker -> notes, without audio hardware."""
    sr = 16000

    class FakeStream:
        def __init__(self, callback, **kw):
            self.cb = callback

        def start(self):
            block = np.zeros((800, 1), np.int16)
            for _ in range(40):                    # 2 s of audio
                self.cb(block, 800, None, None)

        def stop(self):
            pass

        def close(self):
            pass

    class Seg:
        def __init__(self, words):
            self.words = words

    class FakeModel:
        def transcribe(self, audio, **kw):
            W = SimpleNamespace
            return [Seg([W(word=" Hello", start=0.2, end=0.5), W(word=" there.", start=0.6, end=0.9),
                         W(word=" Second", start=1.2, end=1.5), W(word=" one.", start=1.5, end=1.8)])], None

    import sounddevice
    monkeypatch.setattr(sounddevice, "InputStream", lambda **kw: FakeStream(**kw))
    vn = VoiceNotes(tmp_path, chunk_s=10)
    vn._device = {"index": 0, "name": "fake", "default_samplerate": sr, "max_input_channels": 1}
    vn._model.model, vn._model._started = FakeModel(), True
    vn._model.ready.set()
    vn.start()
    t0 = vn.recordings[0].transcript.session_start_epoch
    for k in range(20):                          # held at one spot for 1 s, then at another
        vn.add_pointer(t0 + 0.1 * k, (0.1, 0.2) if k < 10 else (0.7, 0.8))
    vn.stop()
    assert vn.wait(timeout=10)
    notes = vn.notes()
    assert [n.text for n in notes] == ["Hello there.", "Second one."]
    assert [n.uv for n in notes] == [pytest.approx((0.1, 0.2)), pytest.approx((0.7, 0.8))]
    assert (vn.session_dir / "rec_01" / "session.wav").stat().st_size > 2 * sr
    assert isinstance(vn.recordings[0], Recording)


def fake_hand(label, tip, pointing=True, reason="index not extended"):
    return SimpleNamespace(handedness=label, tip=np.asarray(tip, float), pointing_any_dir=pointing,
                           reason_any_dir="ok" if pointing else reason)


def test_pick_pointer_label_is_only_a_priority_and_direction_is_ignored():
    from doctrack.voice import pick_pointer
    to_doc = lambda x, y: (x / 100, y / 100) if 0 <= x <= 100 and 0 <= y <= 100 else None
    assert pick_pointer([], to_doc)[2] == "no hand seen"
    # right hand mislabelled "Left" by MediaPipe: still used
    uv, h, st = pick_pointer([fake_hand("Left", (50, 20))], to_doc, "Right")
    assert (uv, st) == ((0.5, 0.2), "ok")
    # both pointing on the page: the labelled hand wins
    uv, h, _ = pick_pointer([fake_hand("Left", (10, 10)), fake_hand("Right", (90, 90))], to_doc, "Right")
    assert h.handedness == "Right" and uv == (0.9, 0.9)
    # reasons
    assert pick_pointer([fake_hand("Right", (500, 5))], to_doc)[2] == "fingertip off the page"
    assert pick_pointer([fake_hand("Right", (5, 5), pointing=False)], to_doc)[2] == "index not extended"
    assert pick_pointer([fake_hand("Right", (5, 5), pointing=False)], to_doc, any_pose=True)[2] == "ok"


def track(*legs, fps=10):
    """Fingertip samples: legs of (t0, t1, (u, v)), with small jitter, None = finger off the page."""
    rng = np.random.default_rng(0)
    out = []
    for t0, t1, uv in legs:
        for t in np.arange(t0, t1, 1 / fps):
            if uv is not None:
                out.append((float(t), (uv[0] + rng.normal(0, 0.006), uv[1] + rng.normal(0, 0.006))))
    return out


def notes_from(words, samples, **kw):
    """words: [(t, "word"), ...] spoken; samples: [(t, (u, v)), ...] fingertip."""
    vn = VoiceNotes("/nonexistent", **kw)
    vn.aspect = 1.0
    vn.recordings = [SimpleNamespace(words=lambda: words, transcript=SimpleNamespace(sentences=[]))]
    for t, uv in samples:
        vn.pointer.add(t, *uv)
    return vn.notes()


def said(text, t0, dt=0.3):
    return [(t0 + i * dt, w) for i, w in enumerate(text.split())]


def test_new_comment_starts_at_the_first_word_spoken_at_a_new_spot():
    # ONE sentence while moving from the heading (0.2, 0.1) to a number (0.7, 0.6)
    words = said("this heading is wrong and this number should be 2000", 10.0)   # 10.0 .. 12.7
    notes = notes_from(words, track((9.5, 11.4, (0.2, 0.1)), (11.4, 11.6, (0.45, 0.35)),
                                    (11.6, 14.0, (0.7, 0.6))))
    assert [n.text for n in notes] == ["this heading is wrong and", "this number should be 2000"]
    assert notes[0].uv == pytest.approx((0.2, 0.1), abs=0.01)
    assert notes[1].uv == pytest.approx((0.7, 0.6), abs=0.01)


def test_small_moves_and_jitter_stay_one_comment():
    words = said("first remark here.", 1.0) + said("second remark same place.", 3.0)
    # wanders within 3% of the page: below the 5% radius
    notes = notes_from(words, track((0.5, 2.0, (0.50, 0.50)), (2.0, 3.0, (0.52, 0.51)),
                                    (3.0, 5.0, (0.49, 0.52))))
    assert len(notes) == 1 and notes[0].text == "first remark here. second remark same place."


def test_passing_over_a_spot_without_stopping_is_not_an_anchor():
    words = said("one two three four five six seven eight", 1.0)               # 1.0 .. 3.1
    notes = notes_from(words, track((0.5, 1.9, (0.2, 0.2)), (1.9, 2.1, (0.5, 0.5)),  # 0.2 s: transit
                                    (2.1, 4.0, (0.8, 0.8))))
    assert [n.uv for n in notes] == [pytest.approx((0.2, 0.2), abs=0.01), pytest.approx((0.8, 0.8), abs=0.01)]


def test_words_without_finger_stay_with_the_last_stop():
    words = said("look here.", 1.0) + said("and also this, finger lifted.", 4.0)
    notes = notes_from(words, track((0.5, 2.0, (0.3, 0.3))))
    assert len(notes) == 1 and notes[0].uv == pytest.approx((0.3, 0.3), abs=0.01)


def test_words_before_any_finger_are_one_unpinned_comment():
    words = said("intro words.", 1.0) + said("now pointing.", 5.0)
    notes = notes_from(words, track((5.0, 7.0, (0.6, 0.4))))
    assert [(n.text, n.uv is None) for n in notes] == [("intro words.", True), ("now pointing.", False)]


def test_no_finger_at_all_gives_one_unpinned_comment():
    notes = notes_from(said("A. B.", 1.0), [])
    assert [(n.text, n.uv) for n in notes] == [("A. B.", None)]
