"""Tests for the Apple Vision locked-document tracker (macOS only)."""
import cv2
import numpy as np
import pytest

from doctrack import DocumentTracker, OpenCVDocumentDetector
from doctrack.vision_track import vision_available

from test_doctrack import HAND, QA, QB, _img, _scene, _track_scene

pytestmark = pytest.mark.skipif(not vision_available(), reason="Apple Vision not available")
S = (1920, 1080)


def _locked(q1=QA, q2=QB):
    tr = DocumentTracker(OpenCVDocumentDetector(), lock_backend="vision")
    img = _scene([q1, q2], size=S)
    _track_scene(lambda i: img, 12, tr)
    assert tr.select_at(*np.float32(q1).mean(0)) is not None
    return tr


def test_lock_snaps_to_page():
    tr = _locked()
    assert np.abs(_img(tr.selected.corners) - QA).max() < 8   # the detector's accuracy here: a few px


def test_still_page_gives_still_outline():
    tr = _locked()
    hist = []
    for k in range(40):                                   # fresh sensor noise every frame
        tr.update(_scene([QA, QB], size=S, seed=100 + k), t=1 + k / 15)
        hist.append(tr.selected.corners.copy())
    steps = np.abs(np.diff(np.array(hist[3:]), axis=0)).max()
    assert steps == 0.0


def test_follows_page_moving():
    tr = _locked()
    for k in range(1, 21):
        tr.update(_scene([QA + [6 * k, 2 * k], QB], size=S, seed=k), t=1 + k / 15)
    for k in range(6):
        tr.update(_scene([QA + [120, 40], QB], size=S, seed=40 + k), t=3 + k / 15)
    assert np.abs(_img(tr.selected.corners) - (QA + [120, 40])).max() < 8


def test_hand_on_page_holds_then_refinds():
    tr = _locked()
    before = tr.selected.corners.copy()
    hand = cv2.ellipse2Poly((380, 470), (160, 220), 0, 0, 360, 20).astype(np.float32)
    for k in range(10):                                   # hand on the page, page pushed meanwhile
        s = _scene([QA + [8 * k, 0], QB], size=S, seed=k)
        cv2.fillPoly(s, [(hand + [8 * k, 0]).astype(np.int32)], HAND)
        tr.update(s, t=1 + k / 15, occluders=[hand + [8 * k, 0]])
        assert tr.selected.state == "covered"
    np.testing.assert_array_equal(tr.selected.corners, before)   # held exactly while covered
    for k in range(8):                                    # hand gone: page found where it is now
        tr.update(_scene([QA + [72, 0], QB], size=S, seed=30 + k), t=2 + k / 15)
    assert tr.selected.state == "tracking"
    assert np.abs(_img(tr.selected.corners) - (QA + [72, 0])).max() < 8


def test_does_not_jump_to_other_page():
    """Page removed: it must be searched for, not replaced by another sheet
    (here a blank one: same size and shape, different content)."""
    tr = _locked()
    for k in range(15):
        tr.update(_scene([QB], size=S, seed=k, ink=(235, 235, 235)), t=1 + k / 15)
    assert tr.selected.state == "searching"
    assert np.abs(_img(tr.selected.corners) - QA).max() < 8   # still where it was last seen
