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


def test_pointer_log_picks_sample_nearest_sentence_start():
    log = PointerLog()
    for t, u in [(10.0, 0.1), (11.0, 0.2), (12.0, 0.3), (20.0, 0.9)]:
        log.add(t, u, 0.5)
    assert log.at(11.1, 13.0) == (0.2, 0.5)
    assert log.at(15.5, 19.8) == (0.9, 0.5)        # finger arrived during the sentence
    assert log.at(30.0, 31.0) is None              # finger never on the page
    assert PointerLog().at(0, 1) is None


def test_export_pdf_annotations(tmp_path):
    scan = np.full((594, 420, 3), 255, np.uint8)
    notes = [Note(1, "Fix this title.", time.time(), time.time() + 1, (0.25, 0.1)),
             Note(2, "Unplaced remark.", time.time(), time.time() + 1, None)]
    path = export_pdf(scan, notes, tmp_path / "out.pdf", paper="a4")
    page = pypdf.PdfReader(str(path)).pages[0]
    w, h = float(page.mediabox.width), float(page.mediabox.height)
    assert (w, h) == pytest.approx((595.3, 841.9), abs=1.5)        # A4 in points
    every = [a.get_object() for a in page["/Annots"]]
    assert [a["/Subtype"] for a in every] == ["/Text", "/Popup", "/Text", "/Popup"]
    annots = [a for a in every if a["/Subtype"] == "/Text"]
    for a in annots:   # viewers like Preview need the linked popup + spec /F flags to show the text
        assert a["/Popup"].get_object()["/Parent"].get_object() is a or \
            a["/Popup"].get_object()["/Parent"].idnum == a.indirect_reference.idnum
        assert a["/F"] & 4 and "/Flags" not in a
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
    vn.add_pointer(t0 + 0.2, (0.1, 0.2))
    vn.add_pointer(t0 + 1.2, (0.7, 0.8))
    vn.stop()
    assert vn.wait(timeout=10)
    notes = vn.notes()
    assert [n.text for n in notes] == ["Hello there.", "Second one."]
    assert [n.uv for n in notes] == [(0.1, 0.2), (0.7, 0.8)]
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


def notes_from(sentences, samples):
    vn = VoiceNotes("/nonexistent")
    vn.recordings = [SimpleNamespace(sentences=lambda: sentences)]
    for t, uv in samples:
        vn.pointer.add(t, *uv)
    return vn.notes()


def test_sentence_without_finger_joins_the_last_comment():
    notes = notes_from([(10, 12, "Look here."), (13, 15, "Still about it."), (20, 22, "New point.")],
                       [(10.0, (0.2, 0.2)), (20.1, (0.8, 0.8))])
    assert [(n.text, n.uv) for n in notes] == [("Look here. Still about it.", (0.2, 0.2)),
                                             ("New point.", (0.8, 0.8))]
    assert notes[0].end == 15


def test_finger_lifted_before_talking_starts_a_comment_at_its_last_spot():
    # pointed at (0.7, 0.3) at t=16, lifted, then spoke at t=20 with no finger on the page
    notes = notes_from([(10, 12, "First."), (20, 22, "Second.")], [(10.0, (0.1, 0.1)), (16.0, (0.7, 0.3))])
    assert [(n.text, n.uv) for n in notes] == [("First.", (0.1, 0.1)), ("Second.", (0.7, 0.3))]


def test_no_finger_at_all_gives_one_unpinned_comment():
    notes = notes_from([(1, 2, "A."), (3, 4, "B.")], [])
    assert [(n.text, n.uv) for n in notes] == [("A. B.", None)]
