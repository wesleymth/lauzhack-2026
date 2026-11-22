import cv2
import numpy as np
import pytest

from doctrack import (DocumentTracker, OpenCVDocumentDetector, estimate_aspect, order_corners, paper_size,
                      point_in_quad, quad_iou, refine_quad, warp_document)


WOOD, WHITE, GREY = (40, 70, 110), (238, 238, 238), (200, 200, 200)


def _scene(quads, size=(1280, 720), desk=WOOD, paper=(235, 235, 235), ink=(40, 40, 40), seed=0,
           doc_up="down"):
    """Desk with 'pages' carrying a title bar + fake text lines, plus mild sensor
    noise. Quads are given in image order (TL, TR, BR, BL on screen); with
    doc_up="down" the page is drawn upside down (top towards the image bottom)."""
    w, h = size
    img = np.full((h, w, 3), desk, np.uint8)
    for q in quads:
        q = np.asarray(q, np.float32)
        page = np.full((297, 210, 3), paper, np.uint8)
        cv2.rectangle(page, (20, 8), (190, 18), ink, -1)  # title bar at the top of the page
        words = np.random.default_rng(42)  # same "text" on every page / frame
        for y in range(34, 270, 14):
            x = 20
            while x < 185:
                L = int(words.integers(8, 30))
                cv2.line(page, (x, y), (min(x + L, 190), y), ink, 2)
                x += L + int(words.integers(4, 8))
        dst = np.roll(q, 2, axis=0) if doc_up == "down" else q
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [209, 0], [209, 296], [0, 296]]), dst)
        warped = cv2.warpPerspective(page, M, (w, h))
        mask = cv2.warpPerspective(np.full((297, 210), 255, np.uint8), M, (w, h))
        img[mask > 0] = warped[mask > 0]
    noise = np.random.default_rng(seed).normal(0, 3, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _img(c):
    """Corners back in on-screen order, to compare with the image-order fixtures."""
    return order_corners(c, "up")


Q1 = [[100, 100], [400, 120], [380, 540], [90, 520]]
Q2 = [[700, 150], [1000, 140], [1020, 560], [720, 580]]


@pytest.mark.parametrize("desk,paper,ink", [
    (WOOD, (235, 235, 235), (40, 40, 40)),        # white paper, dark desk
    (WHITE, (35, 35, 35), (230, 230, 230)),       # black paper, white desk
    (WHITE, (230, 200, 150), (40, 40, 40)),       # light-blue paper, white desk
    (GREY, (90, 200, 250), (40, 40, 40)),         # yellow paper, grey desk
    (WHITE, (60, 60, 200), (240, 240, 240)),      # red paper, white desk
], ids=["white-on-wood", "black-on-white", "blue-on-white", "yellow-on-grey", "red-on-white"])
def test_detect_any_colour(desk, paper, ink):
    img = _scene([Q1, Q2], desk=desk, paper=paper, ink=ink)
    dets = OpenCVDocumentDetector().detect(img)
    assert len(dets) == 2, [d.info for d in dets]
    for d in dets:
        ref = Q1 if d.corners[0, 0] < 600 else Q2
        assert np.abs(_img(d.corners) - np.float32(ref)).max() < 12


@pytest.mark.parametrize("desk", [WOOD, WHITE], ids=["wood", "white"])
def test_rejects_non_documents(desk):
    img = _scene([], desk=desk)
    rng = np.random.default_rng(1)
    # "photo / screen": busy random content in a paper-shaped rectangle
    img[100:500, 100:380] = rng.integers(0, 255, (400, 280, 3), dtype=np.uint8)
    # square coaster
    cv2.rectangle(img, (700, 200), (900, 400), (30, 30, 220), -1)
    assert OpenCVDocumentDetector().detect(img) == []


def test_order_corners():
    pts = np.float32([[100, 0], [0, 0], [0, 100], [100, 100]])
    np.testing.assert_array_equal(order_corners(pts), np.float32([[0, 0], [100, 0], [100, 100], [0, 100]]))
    # page top towards the image bottom: the doc's TL is the on-screen bottom-right
    np.testing.assert_array_equal(order_corners(pts, "down"),
                                  np.float32([[100, 100], [0, 100], [0, 0], [100, 0]]))
    np.testing.assert_array_equal(order_corners(pts, "left")[0], np.float32([0, 100]))


def test_quad_iou():
    a = np.float32([[0, 0], [10, 0], [10, 10], [0, 10]])
    b = a + np.float32([5, 0])
    assert abs(quad_iou(a, a) - 1.0) < 1e-6
    assert abs(quad_iou(a, b) - 50 / 150) < 1e-3


def test_warp_size():
    img = np.zeros((200, 200, 3), np.uint8)
    out = warp_document(img, np.float32([[10, 10], [110, 10], [110, 160], [10, 160]]))
    assert out.shape[:2] == (150, 100)


def _track_scene(frame_fn, n, tracker, t0=0.0):
    for i in range(n):
        tracker.update(frame_fn(i), t=t0 + i / 30.0)


def test_browse_then_lock_ignores_others():
    tr = DocumentTracker(OpenCVDocumentDetector(), lock_backend="classic")
    img = _scene([Q1, Q2])
    _track_scene(lambda i: img, 5, tr)
    assert tr.tracks == []                           # not shown until confident for a while
    _track_scene(lambda i: img, 10, tr, 5 / 30)
    assert len(tr.tracks) == 2
    t = tr.select_at(250, 300)
    assert t is not None and tr.locked
    assert tr.scan_size == (778, 1100)               # A4 portrait, fixed
    _track_scene(lambda i: img, 5, tr, 1.0)
    assert [x.id for x in tr.tracks] == [t.id]       # other document no longer tracked
    assert tr.select_at(850, 300).id == t.id         # clicks don't switch while locked
    assert tr.scan(img).shape[:2] == (1100, 778)
    tr.clear_selection()
    _track_scene(lambda i: img, 15, tr, 2.0)
    assert len(tr.tracks) == 2


def test_browse_shows_at_most_three():
    quads = [np.float32(q) for q in ([[60, 60], [300, 60], [300, 400], [60, 400]],
                                     [[360, 60], [600, 60], [600, 400], [360, 400]],
                                     [[660, 60], [900, 60], [900, 400], [660, 400]],
                                     [[960, 60], [1200, 60], [1200, 400], [960, 400]])]
    tr = DocumentTracker(OpenCVDocumentDetector(), lock_backend="classic")
    img = _scene(quads)
    _track_scene(lambda i: img, 15, tr)
    shown = [t.id for t in tr.tracks]
    assert len(shown) == 3
    _track_scene(lambda i: _scene(quads, seed=i), 15, tr, 1.0)
    assert [t.id for t in tr.tracks] == shown        # the shown set doesn't flicker


def test_scan_is_upright_for_reader():
    """Page lies with its top towards the image bottom: the scan must be upright."""
    tr = _locked_tracker(QA, QB)
    scan = cv2.cvtColor(tr.scan(_scene([QA, QB], size=(1920, 1080))), cv2.COLOR_BGR2GRAY)
    h = scan.shape[0]
    # the title bar is at the top of the page -> dark band in the top 7 % of the scan
    assert scan[int(0.02 * h):int(0.07 * h)].mean() < scan[int(0.93 * h):int(0.98 * h)].mean() - 30


def _locked_tracker(q1, q2, size=(1920, 1080)):
    tr = DocumentTracker(OpenCVDocumentDetector(), lock_backend="classic")
    img = _scene([q1, q2], size=size)
    _track_scene(lambda i: img, 12, tr)
    c = np.float32(q1).mean(0)
    assert tr.select_at(*c) is not None
    return tr


QA = np.float32([[150, 150], [600, 180], [570, 810], [135, 780]])
QB = np.float32([[1050, 225], [1500, 210], [1530, 840], [1080, 870]])
HAND = (120, 160, 210)


def test_locked_is_steady_under_sensor_noise():
    tr = _locked_tracker(QA, QB)
    hist = []
    for i in range(40):
        tr.update(_scene([QA, QB], size=(1920, 1080), seed=100 + i), t=1 + i / 30)
        hist.append(_img(tr.selected.corners))
    hist = np.array(hist[10:])
    assert hist.std(axis=0).max() < 0.5                       # sub-pixel jitter
    assert np.abs(hist.mean(axis=0) - QA).max() < 3


@pytest.mark.parametrize("covered", [[2], [1, 2], [0, 1, 2], [0, 1, 2, 3]],
                         ids=["1-corner", "2-corners", "3-corners", "4-corners"])
def test_covered_corners_keep_last_position(covered):
    """A hand over 1..4 corners: covered corners stay exactly where they were last
    seen, the others keep tracking; once uncovered, they are found again."""
    tr = _locked_tracker(QA, QB)
    for i in range(5):
        tr.update(_scene([QA, QB], size=(1920, 1080), seed=200 + i), t=1 + i / 30)
    before = tr.selected.corners.copy()
    img_idx = [int(np.argmin(np.linalg.norm(_img(before) - before[j], axis=1))) for j in range(4)]
    for k in range(20):
        s = _scene([QA, QB], size=(1920, 1080), seed=k)
        for c in covered:                                     # covered = image-order corner index
            x, y = QA[c].astype(int)
            cv2.circle(s, (x, y), 70, HAND, -1)
        tr.update(s, t=2 + k / 30)
    sel = tr.selected
    hidden_img = sorted(img_idx[j] for j in range(4) if not sel.visible[j])
    assert hidden_img == sorted(covered)
    for j in range(4):
        if not sel.visible[j]:
            np.testing.assert_array_equal(sel.corners[j], before[j])   # frozen exactly
    assert sel.state == ("searching" if len(covered) == 4 else "partial")
    for k in range(5):                                        # hand lifted
        tr.update(_scene([QA, QB], size=(1920, 1080), seed=50 + k), t=3 + k / 30)
    assert tr.selected.visible.all() and tr.selected.state == "tracking"
    assert np.abs(_img(tr.selected.corners) - QA).max() < 3


def test_covered_corner_refound_after_page_moved():
    tr = _locked_tracker(QA, QB)
    size0 = tr.scan_size
    for k in range(1, 16):  # page slides a little while a hand covers its bottom-right corner
        s = _scene([QA + [2 * k, k], QB], size=(1920, 1080), seed=k)
        cv2.circle(s, (570 + 2 * k, 810 + k), 80, HAND, -1)
        tr.update(s, t=1 + k / 30)
    assert tr.selected.state == "partial"
    for k in range(6):      # hand lifted: corner found again by searching outward
        tr.update(_scene([QA + [30, 15], QB], size=(1920, 1080), seed=50 + k), t=2 + k / 30)
    assert tr.selected.state == "tracking"
    assert np.abs(_img(tr.selected.corners) - (QA + [30, 15])).max() < 3
    assert tr.scan_size == size0


def test_locked_reacquires_after_full_occlusion_and_move():
    tr = _locked_tracker(QA, QB)
    tid = tr.selected.id
    for k in range(15):     # page fully covered
        s = _scene([QA, QB], size=(1920, 1080), seed=k)
        cv2.rectangle(s, (80, 80), (700, 900), HAND, -1)
        tr.update(s, t=1 + k / 30)
    assert tr.locked and tr.selected.id == tid and tr.selected.state == "searching"
    moved = QA + [200, 50]  # reappears somewhere else
    for k in range(15):
        tr.update(_scene([moved, QB], size=(1920, 1080), seed=30 + k), t=2 + k / 30)
    assert tr.selected.state == "tracking"
    assert np.abs(_img(tr.selected.corners) - moved).max() < 3


def test_refine_quad_snaps_to_edges():
    img = _scene([QA], size=(1920, 1080))
    rng = np.random.default_rng(3)
    noisy = QA + rng.uniform(-6, 6, QA.shape).astype(np.float32)
    ref, n = refine_quad(img, noisy)
    assert n == 4 and np.abs(ref - QA).max() < 1.5


def test_estimate_aspect_under_perspective():
    f, W, H = 1100.0, 1920, 1080
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    P = np.array([[0, 0, 0], [210, 0, 0], [210, 297, 0], [0, 297, 0]], float) - [105, 148, 0]
    for rvec in ([0.5, 0.2, 0.1], [0.8, 0.2, 0.3], [-0.6, 0.2, 1.2]):
        R, _ = cv2.Rodrigues(np.array(rvec))
        X = (R @ P.T).T + [30, -20, 600]
        x = (K @ X.T).T
        q = x[:, :2] / x[:, 2:]
        assert abs(estimate_aspect(q, (W, H)) - 210 / 297) < 0.01
        assert paper_size(estimate_aspect(q, (W, H))) == (778, 1100)


# --------------------------------------------------------------------------- #
# hand selection (fake hands: no camera / MediaPipe needed)
# --------------------------------------------------------------------------- #
from doctrack.hands import DwellSelector, Hand  # noqa: E402


def _hand(tip, pointing=True):
    lm = np.tile(np.float32(tip), (21, 1)) + np.float32([0, -150])  # wrist side above the tip
    lm[8] = tip
    lm[0] = np.float32(tip) + [40, -250]
    return Hand(lm.astype(np.float32), "Right", pointing)


def _browse_ready():
    tr = DocumentTracker(OpenCVDocumentDetector(), lock_backend="classic")
    img = _scene([Q1, Q2])
    _track_scene(lambda i: img, 15, tr)
    assert len(tr.tracks) == 2
    return tr, img


def _run(tr, sel, img, tip, t0, secs, pointing=True, fps=30):
    fired = None
    for i in range(int(secs * fps)):
        t = t0 + i / fps
        hands = [_hand(tip, pointing)] if tip is not None else []
        tr.update(img, t, occluders=[h.hull for h in hands])
        fired = sel.update(tr, hands, t) or fired
    return fired


def test_point_one_second_selects_document():
    tr, img = _browse_ready()
    sel = DwellSelector()
    target = next(t.id for t in tr.tracks if point_in_quad(t.corners, 250, 300))
    assert _run(tr, sel, img, (250, 300), 1.0, 0.8) is None and not tr.locked   # not yet
    assert _run(tr, sel, img, (250, 300), 1.8, 0.4) == target
    assert tr.locked and tr.selected.id == target


def test_non_pointing_hand_does_not_select():
    tr, img = _browse_ready()
    sel = DwellSelector()
    assert _run(tr, sel, img, (250, 300), 1.0, 2.0, pointing=False) is None
    assert not tr.locked


def test_short_dropout_keeps_timer_moving_finger_resets():
    tr, img = _browse_ready()
    sel = DwellSelector()
    _run(tr, sel, img, (250, 300), 1.0, 0.6)
    _run(tr, sel, img, None, 1.6, 0.15)                     # hand lost for 0.15 s
    assert _run(tr, sel, img, (250, 300), 1.75, 0.4) is not None
    tr.clear_selection()
    _track_scene(lambda i: img, 15, tr, 3.0)
    sel2 = DwellSelector()
    _run(tr, sel2, img, (250, 300), 4.0, 0.6)
    assert _run(tr, sel2, img, (860, 350), 4.6, 0.6) is None  # moved to the other doc: restarts


def test_no_reselect_until_finger_leaves_after_release():
    tr, img = _browse_ready()
    sel = DwellSelector()
    assert _run(tr, sel, img, (250, 300), 1.0, 1.2) is not None
    _run(tr, sel, img, (250, 300), 2.2, 0.2)                # locked; finger still there
    tr.clear_selection()
    _track_scene(lambda i: img, 15, tr, 2.5)
    assert _run(tr, sel, img, (250, 300), 3.0, 2.0) is None  # still resting: no re-lock
    _run(tr, sel, img, None, 5.0, 0.5)                       # finger leaves
    assert _run(tr, sel, img, (250, 300), 5.5, 1.2) is not None


def test_document_under_hand_stays_shown():
    tr, img = _browse_ready()
    tid = next(t.id for t in tr.tracks if point_in_quad(t.corners, 250, 300))
    covered = img.copy()
    arm = np.int32([[180, 0], [330, 0], [300, 330], [210, 330]])  # arm over the doc's top edge
    cv2.fillConvexPoly(covered, arm, HAND)
    before = next(t for t in tr.tracks if t.id == tid).corners.copy()
    for i in range(90):                                       # 3 s with the hand on it
        tr.update(covered, 2 + i / 30, occluders=[arm.astype(np.float32)])
    t = next((t for t in tr.tracks if t.id == tid), None)
    assert t is not None and np.abs(t.corners - before).max() < 3


def test_rotate_and_mirror_scan_and_corners():
    tr = _locked_tracker(QA, QB)
    img = _scene([QA, QB], size=(1920, 1080))
    base = tr.scan(img).astype(np.float32)
    c0 = tr.selected.corners.copy()
    tr.rotate()                                              # 90 deg clockwise
    assert tr.scan_size == (1100, 778)
    rot = tr.scan(img).astype(np.float32)
    assert rot.shape[:2] == (778, 1100)
    assert np.abs(rot - np.rot90(base, k=-1)).mean() < 3
    np.testing.assert_array_equal(tr.selected.corners[0], c0[3])   # new TL = old BL
    tr.rotate(3)                                             # back to the start
    np.testing.assert_array_equal(tr.selected.corners, c0)
    tr.toggle_mirror()
    assert tr.mirrored and np.abs(tr.scan(img).astype(np.float32) - base[:, ::-1]).mean() < 3
    tr.update(img, 5.0)                                      # survives tracking updates
    assert np.abs(tr.selected.corners - c0[[1, 0, 3, 2]]).max() < 2
    tr.clear_selection()
    assert not tr.mirrored and tr.scan_size is None


def test_scan_any_size():
    tr = _locked_tracker(QA, QB)
    img = _scene([QA, QB], size=(1920, 1080))
    assert tr.scan(img, size=(640, 300)).shape[:2] == (300, 640)
