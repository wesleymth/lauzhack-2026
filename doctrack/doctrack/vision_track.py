"""Locked-document tracker built on Apple's pretrained document detector.

Apple Vision (macOS) ships a neural document detector
(VNDetectDocumentSegmentationRequest) that returns the 4 corners of the most
prominent document in an image, in ~5 ms on Apple silicon. On its own it
returns ONE document per image and gets confused by hands and by other
objects, so it is used like this:

  * measure: the detector runs on a crop around the selected page (~1.5x its
    size), so other objects on the desk stay out of view;
  * trust: a result is used only if it is confident, plausible (convex,
    similar size) and its content still looks like this page (the flattened
    page is compared with a picture of it stored when it was first seen
    clean);
  * covered: when the hand tracker sees a hand on the page, the detector is
    not trusted and the outline holds still;
  * lost: no usable result for a few frames with no hand on the page -> search
    with wider crops, the left / right half and the full frame; candidates
    must pass the same size and content checks;
  * stable output: changes smaller than the detector's noise are ignored, so
    a still page gives a still scan; real motion is followed at once.

Same interface as `locked.LockedTracker` so `DocumentTracker` can use either.
"""

import sys
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .filters import OneEuroFilter
from .geometry import estimate_aspect, paper_size, quad_area, quad_size, warp_document

TRACKING, PARTIAL, COVERED, SEARCHING = "tracking", "partial", "covered", "searching"


def vision_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import Vision  # noqa: F401
        return hasattr(Vision, "VNDetectDocumentSegmentationRequest")
    except Exception:
        return False


class VisionDocumentDetector:
    """Apple Vision document detector: image -> (corners (4, 2) image px, confidence)."""
    min_conf, strict_conf = 0.8, 0.9      # confidence scale of this model
    full_frame = True                     # can find a page anywhere in a large image
    square_crops = False
    trust_under_hand = False              # hands confuse it: hold still instead

    def __init__(self) -> None:
        import objc
        import Vision
        from Foundation import NSData
        self._objc, self._Vision, self._NSData = objc, Vision, NSData

    def __call__(self, img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        h, w = img_bgr.shape[:2]
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return None, 0.0
        V = self._Vision
        with self._objc.autorelease_pool():
            data = self._NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
            handler = V.VNImageRequestHandler.alloc().initWithData_options_(data, None)
            req = V.VNDetectDocumentSegmentationRequest.alloc().init()
            handler.performRequests_error_([req], None)
            res = req.results() or []
            if not res:
                return None, 0.0
            o = res[0]
            pts = [o.topLeft(), o.topRight(), o.bottomRight(), o.bottomLeft()]
            quad = np.float32([[p.x * w, (1.0 - p.y) * h] for p in pts])   # Vision: origin bottom-left
            return quad, float(o.confidence())


class VisionLockedTracker:
    def __init__(self, frame: np.ndarray, corners: np.ndarray, t: float, paper: str = "a4",
                 scan_long_side: int = 1100, occluders: Sequence[np.ndarray] = (),
                 detector=None, min_conf: Optional[float] = None) -> None:
        H, W = frame.shape[:2]
        self._k = W / 1920.0                              # pixel scale relative to 1080p
        self.model = detector or VisionDocumentDetector()
        self.min_conf = min_conf if min_conf is not None else self.model.min_conf
        self._proposer = None                             # page detector used to propose where to look
        corners = np.asarray(corners, np.float32)
        self.aspect = estimate_aspect(corners, (W, H))
        self.scan_size: Tuple[int, int] = paper_size(self.aspect, paper, scan_long_side)
        self.raw = corners.copy()                         # last accepted corners (doc order)
        self._filter = OneEuroFilter(min_cutoff=1.0, beta=0.05)
        self.corners = self._filter(corners, t)
        self._area = quad_area(corners)                   # reference size of the page in the image
        self._occ: List[np.ndarray] = _polys(occluders)
        self._misses = 0                                  # frames without a usable result (no hand)
        self._search_step = 0
        self._still = 0
        self.state = TRACKING
        self.visible = np.ones(4, bool)
        self._app_size = (36, 50) if self.scan_size[0] < self.scan_size[1] else (50, 36)
        self._app: Optional[np.ndarray] = None            # picture of the page (content check)
        self._ref_small: Optional[np.ndarray] = None      # frame (1/4 size) at the last outline update
        self._since_check = 0
        self._drift: Optional[np.ndarray] = None
        if not any(_overlaps(corners, o) for o in self._occ):
            m = self._measure(frame, self.raw)            # snap to the model's corners right away
            if m is not None:
                self.raw = m
                self._filter.reset()
                self.corners = self._filter(m, t)
            self._app = self._appearance(frame, self.raw)
            self._ref_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=0.25, fy=0.25,
                                         interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------ #
    @property
    def corner_state(self) -> List[str]:
        if self.state == SEARCHING:
            return ["lost"] * 4
        return ["occluded" if self._hand_near(c) else "visible" for c in self.raw]

    def update(self, frame: np.ndarray, t: float, detector=None,
               occluders: Sequence[np.ndarray] = ()) -> np.ndarray:
        self._occ = _polys(occluders)
        hand_on_page = any(_overlaps(self.raw, o) for o in self._occ)
        moved = False
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=0.25, fy=0.25,
                           interpolation=cv2.INTER_AREA)
        self._proposer = detector
        if hand_on_page:
            self.state = COVERED
            if getattr(self.model, "trust_under_hand", False):
                # the model estimates covered corners too: accept it under stricter rules
                m = self._measure(frame, self.raw, strict=True)
                if m is not None and float(np.linalg.norm(m - self.raw, axis=1).max()) > 4.0 * self._k:
                    moved = self._accept(m)
                    self._ref_small = small
        elif self._ref_small is not None and self._unchanged(small) and self._still_check():
            self._misses = 0                              # page did not move: nothing to do
            self.state = TRACKING
        else:
            m = self._measure(frame, self.raw)
            if m is None and self._misses >= 2:
                m = self._search(frame)                   # page not where it was: look further
            hand_near = any(_overlaps(_grow(self.raw, 1.5), o) for o in self._occ)
            if m is not None and self._ref_small is not None and (hand_near or self._unchanged(small)):
                # still page, or a hand next to it (it disturbs the model): only
                # clear, repeated corrections
                m = self._drift_gate(m)
            if m is not None:
                self._misses, self._search_step = 0, 0
                moved = self._accept(m)
                self._ref_small = small
                if self._app is None:
                    self._app = self._appearance(frame, self.raw)   # first clean view
                self.state = TRACKING
            else:
                self._misses += 1
                self.state = SEARCHING if self._misses > 3 else self.state

        self._still = 0 if moved else self._still + 1
        out = self._filter(self.raw, t)
        if self._still >= 3 and float(np.abs(out - self.corners).max()) < 0.05:
            out = self.corners                            # still page: the output does not move
        self.corners = out
        cs = self.corner_state
        self.visible = np.array([c == "visible" for c in cs])
        if self.state == TRACKING and not self.visible.all():
            self.state = PARTIAL
        return self.corners

    def scan(self, frame: np.ndarray) -> np.ndarray:
        return warp_document(frame, self.corners, size=self.scan_size)

    # ------------------------------------------------------------------ #
    def _unchanged(self, small: np.ndarray) -> bool:
        """Did the page stay where it was? Compares the page area (hands masked
        out) with the frame of the last outline update."""
        mask = np.zeros(small.shape, np.uint8)
        cv2.fillConvexPoly(mask, (self.raw * 0.25).astype(np.int32), 255)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))       # include the page edges
        for o in self._occ:
            cv2.fillPoly(mask, [(o * 0.25).astype(np.int32)], 0)
        if (mask > 0).sum() < 200:
            return False
        d = (small.astype(np.float32) - self._ref_small.astype(np.float32))[mask > 0]
        d = np.abs(d - np.median(d))                      # ignore global brightness / exposure drift
        return float(np.mean(d)) < 2.5 and float(np.percentile(d, 98)) < 25

    def _still_check(self) -> bool:
        """Still page: skip the model, except every ~2 s to catch slow drift."""
        self._since_check += 1
        if self._since_check >= 30:
            self._since_check = 0
            return False
        return True

    def _drift_gate(self, m: np.ndarray) -> Optional[np.ndarray]:
        """On a still page, correct only a clear offset seen twice in a row."""
        if float(np.linalg.norm(m - self.raw, axis=1).max()) < max(6.0 * self._k, 0.012 * _diag(self.raw)):
            self._drift = None
            return None
        if self._drift is not None and np.linalg.norm(m - self._drift, axis=1).max() < 4.0 * self._k:
            self._drift = None
            return m
        self._drift = m
        return None

    def _accept(self, m: np.ndarray) -> bool:
        """Update the corners from a measurement; returns True if the page moved."""
        d = float(np.linalg.norm(m - self.raw, axis=1).max())
        if d < 2.0 * self._k:
            return False                                  # detector noise: keep still
        if d < 6.0 * self._k:
            self.raw = (self.raw + 0.5 * (m - self.raw)).astype(np.float32)   # small: settle gently
        else:
            if d > 0.15 * _diag(self.raw):
                self._filter.reset()                      # re-found elsewhere: jump there
            self.raw = m.astype(np.float32)
        self._area = 0.9 * self._area + 0.1 * quad_area(self.raw)
        return True

    def _measure(self, frame: np.ndarray, near: np.ndarray, scale: float = 1.5,
                 strict: bool = False, far: bool = False) -> Optional[np.ndarray]:
        """Run the model on a crop around `near`; return verified corners or None."""
        H, W = frame.shape[:2]
        x, y, w, h = cv2.boundingRect(near.astype(np.int32))
        cx, cy = x + w / 2, y + h / 2
        half = 0.5 * scale * max(w, h)
        if getattr(self.model, "square_crops", False):
            # a square crop centred on the page (padded where it leaves the frame)
            return self._detect_in(frame, (int(cx - half), int(cy - half), int(cx + half), int(cy + half)),
                                   strict, far)
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
        return self._detect_in(frame, (x0, y0, x1, y1), strict, far)

    def _detect_in(self, frame: np.ndarray, box, strict: bool = False, far: bool = False) -> Optional[np.ndarray]:
        x0, y0, x1, y1 = box
        if x1 - x0 < 32 or y1 - y0 < 32:
            return None
        H, W = frame.shape[:2]
        crop = frame[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            crop = cv2.copyMakeBorder(crop, max(0, -y0), max(0, y1 - H), max(0, -x0), max(0, x1 - W),
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
        q, conf = self.model(crop)
        if q is None or conf < (getattr(self.model, "strict_conf", 0.9) if strict else self.min_conf):
            return None
        q = (q + np.float32([x0, y0])).astype(np.float32)
        q = _align(self.raw, q)                           # keep the document's corner order
        if not cv2.isContourConvex(q.reshape(-1, 1, 2)):
            return None
        r = quad_area(q) / max(self._area, 1.0)
        if not ((0.85 < r < 1.18) if strict else (0.7 < r < 1.4)):
            return None                                   # not the same page (size)
        w0, h0 = quad_size(self.raw)
        w1, h1 = quad_size(q)
        if abs(np.log((max(w1, h1) / min(w1, h1)) / (max(w0, h0) / min(w0, h0)))) > (0.12 if strict else 0.25):
            return None                                   # not the same page (shape)
        if strict:
            if getattr(self.model, "trust_under_hand", False):
                return q                                  # the model infers covered corners
            # a corner under the hand is only the model's guess
            return None if any(self._hand_near(c) for c in q) else q
        if self._app is not None and self._similarity(frame, q) < (0.5 if far else 0.35):
            return None                                   # not the same page (content)
        return q

    def _search(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Look further away: wider crops, then halves of the frame, then all of it."""
        if not getattr(self.model, "full_frame", True):
            return self._search_proposals(frame)
        H, W = frame.shape[:2]
        boxes = [None, None, (0, 0, W // 2 + W // 8, H), (W // 2 - W // 8, 0, W, H), (0, 0, W, H)]
        i = self._search_step % len(boxes)
        self._search_step += 1
        if i == 0:
            return self._measure(frame, self.raw, scale=2.5)
        if i == 1:
            return self._measure(frame, self.raw, scale=4.0, far=True)
        return self._detect_in(frame, boxes[i], far=True)      # far away: needs a strong content match

    def _search_proposals(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """For models that need the page to fill the image: a wider crop around
        the last position, then crops around pages the page detector proposes
        (nearest first); each must pass the size / shape / content checks."""
        self._search_step += 1
        if self._search_step % 2 == 1:
            return self._measure(frame, self.raw, scale=2.2)
        if self._proposer is None:
            return None
        c0 = self.raw.mean(0)
        props = sorted(self._proposer.detect(frame), key=lambda d: float(np.linalg.norm(d.corners.mean(0) - c0)))
        for d in props[:3]:
            far = float(np.linalg.norm(d.corners.mean(0) - c0)) > 0.5 * _diag(self.raw)
            m = self._measure(frame, d.corners, scale=1.4, far=far)
            if m is not None:
                return m
        return None

    def _appearance(self, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        w, h = self._app_size
        g = cv2.cvtColor(warp_document(frame, quad.astype(np.float32), size=(w * 4, h * 4)), cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)

    def _similarity(self, frame: np.ndarray, quad: np.ndarray) -> float:
        a, b = self._app, self._appearance(frame, quad)
        if a.std() < 6.0:                                 # blank page: compare plainly
            return float(np.clip(1.0 - abs(a.mean() - b.mean()) / 40.0, 0, 1) *
                         np.clip(1.0 - (b.std() - 6.0) / 20.0, 0, 1))
        a0, b0 = a - a.mean(), b - b.mean()
        den = float(np.sqrt((a0 * a0).sum() * (b0 * b0).sum()))
        return float((a0 * b0).sum()) / den if den > 1e-6 else 0.0

    def _hand_near(self, p) -> bool:
        pt = (float(p[0]), float(p[1]))
        return any(cv2.pointPolygonTest(o.reshape(-1, 1, 2), pt, True) > -40.0 * self._k for o in self._occ)


def _align(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Cyclically rotate `new` so each corner keeps its identity."""
    k = min(range(4), key=lambda k: float(np.linalg.norm(np.roll(new, -k, 0) - old, axis=1).sum()))
    return np.roll(new, -k, 0)


def _grow(q: np.ndarray, f: float) -> np.ndarray:
    c = q.mean(0)
    return (c + (q - c) * f).astype(np.float32)


def _diag(q: np.ndarray) -> float:
    return float(np.linalg.norm(q[2] - q[0]))


def _overlaps(quad: np.ndarray, poly: np.ndarray) -> bool:
    poly = np.asarray(poly, np.float32).reshape(-1, 2)
    if len(poly) < 3:
        return False
    hull = cv2.convexHull(poly).reshape(-1, 2)
    inter, _ = cv2.intersectConvexConvex(np.asarray(quad, np.float32), hull)
    return inter > 0


def _polys(occluders) -> List[np.ndarray]:
    return [np.asarray(o, np.float32).reshape(-1, 2) for o in occluders if len(o) >= 3]
