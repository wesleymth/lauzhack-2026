"""Document tracker: two modes.

* browse (nothing selected): documents are detected, given a stable ID and
  smoothed heavily (a desk is a slow scene). Only documents seen confidently
  for a while are shown, at most `max_docs` of them. Use `select(id)` /
  `select_at(x, y)` to pick one.
* locked (one selected): ONLY that document is tracked, with a dedicated
  planar tracker (see locked.py). Other documents are ignored and the selected
  one is never dropped. Covered corners keep their last position (state
  "partial", or "searching" when all 4 are covered) until they reappear. `clear_selection()` returns to browse mode.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .filters import OneEuroFilter
from .geometry import point_in_quad, quad_area, quad_iou, warp_document
from .locked import LockedTracker
from .docaligner import DocAlignerModel, docaligner_available
from .vision_track import VisionLockedTracker


@dataclass
class Track:
    id: int
    corners: np.ndarray        # (4, 2) float32, TL TR BR BL (smoothed)
    score: float
    hits: int = 1
    misses: int = 0
    state: str = "tracking"    # "tracking" | "partial" | "searching"
    visible: np.ndarray = field(default_factory=lambda: np.ones(4, bool))  # per corner
    shown: bool = False
    _filter: OneEuroFilter = field(
        default_factory=lambda: OneEuroFilter(min_cutoff=0.25, beta=0.006), repr=False)

    def to_dict(self) -> Dict:
        return {"id": self.id, "corners": self.corners.round(1).tolist(),
                "visible": self.visible.tolist(), "score": round(self.score, 3),
                "state": self.state}


class DocumentTracker:
    def __init__(self, detector, paper: str = "a4", scan_long_side: int = 1100,
                 max_docs: int = 3, min_hits: int = 10, min_score: float = 0.6,
                 new_score: float = 0.45, match_iou: float = 0.3, max_misses: int = 30,
                 lock_backend: str = "auto") -> None:
        """lock_backend: how a selected document is tracked. "docaligner" =
        pretrained DocAligner corner model (onnxruntime), "vision" = Apple's
        document detector (macOS), "classic" = the OpenCV tracker, "auto" =
        docaligner when onnxruntime is available."""
        self.detector = detector
        if lock_backend == "auto":
            lock_backend = "docaligner" if docaligner_available() else "classic"
        self.lock_backend = lock_backend
        self._last_occluders: List[np.ndarray] = []
        self._model = None                                # corner model, loaded on first lock
        self.max_docs = max_docs
        self.min_score = min_score
        self.new_score = new_score
        self.paper = paper
        self.scan_long_side = scan_long_side
        self.match_iou = match_iou
        self.min_hits = min_hits
        self.max_misses = max_misses
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._lock: Optional[LockedTracker] = None
        self._lock_track: Optional[Track] = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_t = 0.0
        # user correction of the locked document's orientation: which tracked
        # corner is reported as TL, TR, BR, BL (rotation / mirror)
        self._perm = np.arange(4)
        self._quarter_turns = 0
        self.mirrored = False

    # ------------------------------ selection API ------------------------- #
    @property
    def locked(self) -> bool:
        return self._lock is not None

    @property
    def tracks(self) -> List[Track]:
        """What to display: the locked document only, or all confirmed documents."""
        if self._lock_track is not None:
            return [self._lock_track]
        return [t for t in self._tracks.values() if t.shown]

    @property
    def selected(self) -> Optional[Track]:
        return self._lock_track

    @property
    def scan_size(self) -> Optional[Tuple[int, int]]:
        """Default (w, h) of the scan: the paper size, swapped after a 90/270 deg rotation."""
        if self._lock is None:
            return None
        w, h = self._lock.scan_size
        return (h, w) if self._quarter_turns % 2 else (w, h)

    def rotate(self, quarter_turns: int = 1) -> None:
        """Correct the document's orientation by 90 deg clockwise steps: the corner
        labels (and so the scan) turn with it."""
        for _ in range(quarter_turns % 4):
            self._perm = self._perm[[3, 0, 1, 2]]      # new TL = old BL
            self._quarter_turns = (self._quarter_turns + 1) % 4
        self._sync_lock_track()

    def toggle_mirror(self) -> None:
        """Mirror the document left-right (TL <-> TR, BL <-> BR)."""
        self._perm = self._perm[[1, 0, 3, 2]]
        self.mirrored = not self.mirrored
        self._sync_lock_track()

    def _sync_lock_track(self) -> None:
        if self._lock is not None and self._lock_track is not None:
            self._lock_track.corners = self._lock.corners[self._perm].copy()
            self._lock_track.visible = self._lock.visible[self._perm].copy()

    def select(self, track_id: int) -> bool:
        t = self._tracks.get(track_id)
        if t is None or self._last_frame is None:
            return False
        if self.lock_backend in ("vision", "docaligner"):
            if self._model is None:
                self._model = DocAlignerModel() if self.lock_backend == "docaligner" else None
            self._lock = VisionLockedTracker(self._last_frame, t.corners, self._last_t, paper=self.paper,
                                             scan_long_side=self.scan_long_side,
                                             occluders=self._last_occluders, detector=self._model)
        else:
            self._lock = LockedTracker(self._last_frame, t.corners, self._last_t, paper=self.paper,
                                       scan_long_side=self.scan_long_side)
        self._lock_track = Track(t.id, self._lock.corners.copy(), t.score, t.hits)
        self._reset_orientation()
        self._tracks.clear()  # stop tracking everything else
        return True

    def select_at(self, x: float, y: float) -> Optional[Track]:
        if self._lock is not None:
            return self._lock_track  # locked: clicks don't switch documents
        hits = [t for t in self.tracks if point_in_quad(t.corners, x, y)]  # shown docs only
        if not hits:
            return None
        t = min(hits, key=lambda t: quad_area(t.corners))  # innermost document wins
        return self._lock_track if self.select(t.id) else None

    def clear_selection(self) -> None:
        self._lock = None
        self._lock_track = None
        self._tracks.clear()
        self._reset_orientation()

    def _reset_orientation(self) -> None:
        self._perm, self._quarter_turns, self.mirrored = np.arange(4), 0, False

    def scan(self, frame: np.ndarray, size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """Flattened view of the locked document (with the orientation
        correction applied), at `size` = (w, h), or `scan_size` by default."""
        if self._lock is None:
            return None
        return warp_document(frame, self._lock.corners[self._perm], size=size or self.scan_size)

    # ------------------------------ update -------------------------------- #
    def update(self, frame: np.ndarray, t: Optional[float] = None,
               occluders: Optional[List[np.ndarray]] = None) -> List[Track]:
        """`occluders`: polygons (N, 2) of things known to be in front of the
        documents (hands). In browse mode a document under one is treated as
        covered, not gone: it keeps its position, ID and confidence."""
        t = time.perf_counter() if t is None else t
        self._last_frame, self._last_t = frame, t
        self._last_occluders = list(occluders or [])
        if self._lock is not None:
            if self.lock_backend in ("vision", "docaligner"):
                self._lock.update(frame, t, self.detector, occluders=self._last_occluders)
            else:
                self._lock.update(frame, t, self.detector)
            self._lock_track.state = self._lock.state
            self._sync_lock_track()
            return [self._lock_track]
        self._update_browse(frame, self.detector.detect(frame), t, occluders or [])
        return self.tracks

    def _update_browse(self, frame: np.ndarray, detections, t: float, occluders) -> None:
        tracks = list(self._tracks.values())
        covered = {tr.id for tr in tracks if any(_overlaps(tr.corners, o) for o in occluders)}
        pairs = sorted(((quad_iou(tr.corners, d.corners), ti, di)
                        for ti, tr in enumerate(tracks) for di, d in enumerate(detections)),
                       reverse=True)
        used_t, used_d = set(), set()
        for iou, ti, di in pairs:
            if iou < self.match_iou:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr, d = tracks[ti], detections[di]
            new = _align(tr.corners, d.corners)
            diag = float(np.linalg.norm(tr.corners[2] - tr.corners[0])) + 1e-6
            if tr.id in covered and float(np.linalg.norm(new - tr.corners, axis=1).max()) > 0.03 * diag:
                tr.misses = 0                 # a hand distorts the outline: keep the old one
                continue
            if float(np.linalg.norm(new - tr.corners, axis=1).mean()) > 0.25 * diag:
                tr._filter.reset()
            tr.corners = tr._filter(new, t)
            tr.score = 0.9 * tr.score + 0.1 * d.score
            tr.hits += 1
            tr.misses = 0
        for ti, tr in enumerate(tracks):
            if ti not in used_t:
                if tr.id in covered:
                    tr.misses = 0             # under a hand: covered, not gone
                    continue
                tr.misses += 1                # keep last position through short dropouts
                tr.score *= 0.97
                if tr.misses > self.max_misses:
                    del self._tracks[tr.id]
        for di, d in enumerate(detections):
            if di not in used_d and d.score >= self.new_score:
                tr = Track(self._next_id, d.corners.copy(), d.score)
                tr.corners = tr._filter(d.corners, t)
                self._tracks[tr.id] = tr
                self._next_id += 1
        # show only confident, established documents; at most max_docs, and
        # documents already shown are favoured so the set doesn't flicker
        cands = [tr for tr in self._tracks.values()
                 if tr.hits >= self.min_hits and tr.score >= (self.min_score - (0.1 if tr.shown else 0))]
        cands.sort(key=lambda tr: tr.score + (0.15 if tr.shown else 0.0), reverse=True)
        top = {tr.id for tr in cands[:self.max_docs]}
        for tr in self._tracks.values():
            tr.shown = tr.id in top


def _overlaps(quad: np.ndarray, poly: np.ndarray) -> bool:
    poly = np.asarray(poly, np.float32).reshape(-1, 2)
    if len(poly) < 3:
        return False
    inter, _ = cv2.intersectConvexConvex(np.asarray(quad, np.float32), poly)
    return inter > 0


def _align(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Cyclically rotate `new` so each corner keeps its identity."""
    k = min(range(4), key=lambda k: float(np.linalg.norm(np.roll(new, -k, 0) - old, axis=1).sum()))
    return np.roll(new, -k, 0)
