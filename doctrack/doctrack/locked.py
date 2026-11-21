"""Single-document tracker used once a document is selected ("locked").

Each of the 4 corners is tracked on its own:
  * visible  -> the two paper sides meeting at it are found near its last
               position (clean straight boundary, paper colour inside) and the
               corner is updated (One-Euro smoothed);
  * covered  -> (hand, pen, sleeve...) its LAST position is kept unchanged and
               used for the warp, for 1, 2, 3 or all 4 corners;
  * re-finding a covered corner starts at its last position and searches
    outward: wider side search, then appearance (patch) matching in growing
    windows, each candidate verified by the side fit and confirmed over 2
    frames. A move is only accepted if the page keeps a sane shape.
If the whole page was moved far while fully hidden, an ORB match against a
snapshot taken at lock time (or, for blank pages, the detector) re-acquires it.

The scan size (a standard paper ratio) is fixed at lock time.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .corners import ColourModel, fit_corner
from .filters import OneEuroFilter
from .geometry import (estimate_aspect, interior_angles, paper_size, quad_area, quad_size, refine_quad,
                       warp_document)

TRACKING, PARTIAL, SEARCHING = "tracking", "partial", "searching"


class LockedTracker:
    def __init__(self, frame: np.ndarray, corners: np.ndarray, t: float, paper: str = "a4",
                 scan_long_side: int = 1100, small_width: int = 960) -> None:
        H, W = frame.shape[:2]
        self._k = W / 1920.0                       # pixel scale relative to 1080p
        corners = np.asarray(corners, np.float32)
        ref, n = refine_quad(frame, corners)
        if n >= 3:
            corners = ref
        self.raw = corners.copy()                  # last known corner positions (unsmoothed)
        self.corners = corners.copy()              # output (smoothed, frozen when covered)
        self.visible = np.ones(4, bool)
        self._filters = [OneEuroFilter() for _ in range(4)]
        for i in range(4):
            self._filters[i](self.raw[i], t)
        self.aspect = estimate_aspect(corners, (W, H))
        self.scan_size: Tuple[int, int] = paper_size(self.aspect, paper, scan_long_side)
        self.state = TRACKING
        self.model = ColourModel.from_frame(frame, corners)
        self._sides0 = _sides(corners)
        self._s = min(1.0, small_width / float(W))
        self._patch_half = 16
        self._patches: List[Optional[np.ndarray]] = [None] * 4
        self._pending: List[Optional[Tuple[np.ndarray, int]]] = [None] * 4
        self._all_hidden = 0
        self._frame_idx = 0
        self._orb = cv2.ORB_create(nfeatures=1500, fastThreshold=12)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        gray = self._gray(frame)
        for i in range(4):
            self._store_patch(gray, i)
        self._init_template(frame)

    # ------------------------------------------------------------------ #
    def update(self, frame: np.ndarray, t: float, detector=None) -> np.ndarray:
        self._frame_idx += 1
        gray = self._gray(frame)
        k = self._k
        props: List[Optional[np.ndarray]] = [None] * 4
        local = np.zeros(4, bool)
        for i in range(4):
            seg = self._seg(i)
            for r in (16 * k, 32 * k):              # small motion: local side fit
                c = fit_corner(frame, self.raw, i, self.raw[i], r, seg, self.model)
                if c is not None:
                    props[i], local[i] = c, True
                    break
            if props[i] is None and self._frame_idx % 3 == i % 3:
                # covered or moved: search outward (every 3rd frame per corner, it's costlier)
                props[i] = self._confirm(i, self._search_outward(frame, gray, i))
            else:
                self._pending[i] = None

        new, vis = self._consolidate(props)
        if not vis.any():
            self._all_hidden += 1
            if self._all_hidden > 20 and self._frame_idx % 10 == 0:
                found = self._reacquire(frame, gray, detector)
                if found is not None:
                    new, vis = found, np.ones(4, bool)
        else:
            self._all_hidden = 0

        for i in range(4):
            if not vis[i]:
                continue                            # covered: keep last position as is
            if np.linalg.norm(new[i] - self.raw[i]) > 12 * k:
                self._filters[i].reset()            # re-found elsewhere: jump, don't glide
            self.raw[i] = new[i]
            self.corners[i] = self._filters[i](new[i], t)
        # hidden corners moved along with the page (see _consolidate)
        for i in np.where(~vis & (np.linalg.norm(new - self.raw, axis=1) > 0.5))[0]:
            self.raw[i] = self.corners[i] = new[i]
            self._filters[i].reset()
            self._filters[i](new[i], t)

        self.visible = vis
        self.state = TRACKING if vis.all() else (SEARCHING if not vis.any() else PARTIAL)
        if self._frame_idx % 15 == 0:
            for i in np.where(local & vis)[0]:
                self._store_patch(gray, i)
        return self.corners

    def scan(self, frame: np.ndarray) -> np.ndarray:
        return warp_document(frame, self.corners, size=self.scan_size)

    # ------------------------------------------------------------------ #
    def _seg(self, i: int) -> float:
        s = _sides(self.raw)
        return float(np.clip(0.3 * min(s[i - 1], s[i]), 25 * self._k, 140 * self._k))

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._s < 1.0:
            g = cv2.resize(g, None, fx=self._s, fy=self._s, interpolation=cv2.INTER_AREA)
        return g

    def _store_patch(self, gray: np.ndarray, i: int) -> None:
        h = self._patch_half
        x, y = np.rint(self.raw[i] * self._s).astype(int)
        if h <= x < gray.shape[1] - h and h <= y < gray.shape[0] - h:
            self._patches[i] = gray[y - h:y + h + 1, x - h:x + h + 1].copy()

    def _search_outward(self, frame: np.ndarray, gray: np.ndarray, i: int) -> Optional[np.ndarray]:
        """Look for corner i in growing windows around its last position, using
        its appearance patch; every candidate is verified by the side fit."""
        patch = self._patches[i]
        if patch is None:
            return None
        h = self._patch_half
        cx, cy = self.raw[i] * self._s
        seg = self._seg(i)
        for R in (40, 80, 160):
            x0, y0 = max(int(cx) - R - h, 0), max(int(cy) - R - h, 0)
            x1, y1 = min(int(cx) + R + h + 1, gray.shape[1]), min(int(cy) + R + h + 1, gray.shape[0])
            win = gray[y0:y1, x0:x1]
            if win.shape[0] <= patch.shape[0] or win.shape[1] <= patch.shape[1]:
                continue
            res = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
            peaks = []
            for _ in range(3):
                _, v, _, loc = cv2.minMaxLoc(res)
                if v < 0.6:
                    break
                peaks.append(np.array([loc[0] + h + x0, loc[1] + h + y0], np.float32) / self._s)
                cv2.rectangle(res, (loc[0] - h, loc[1] - h), (loc[0] + h, loc[1] + h), -1.0, -1)
            peaks.sort(key=lambda p: float(np.linalg.norm(p - self.raw[i])))
            for p in peaks:
                c = fit_corner(frame, self.raw, i, p, 16 * self._k, seg, self.model)
                if c is not None:
                    return c
        return None

    def _confirm(self, i: int, cand: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """A corner found far from its last position must be seen twice."""
        if cand is None:
            self._pending[i] = None
            return None
        prev = self._pending[i]
        n = prev[1] + 1 if prev is not None and np.linalg.norm(prev[0] - cand) < 4 * self._k else 1
        self._pending[i] = (cand, n)
        return cand if n >= 2 else None

    def _sane(self, q: np.ndarray) -> bool:
        if not cv2.isContourConvex(q.astype(np.float32).reshape(-1, 1, 2)):
            return False
        ang = interior_angles(q)
        if ang.min() < 35 or ang.max() > 145:
            return False
        ratio = _sides(q) / self._sides0
        return bool(ratio.min() > 0.65 and ratio.max() < 1.5)

    def _consolidate(self, props) -> Tuple[np.ndarray, np.ndarray]:
        """Combine per-corner proposals into one sane page. Covered corners keep
        their last position; if that is impossible because the visible ones
        moved a lot (page pushed while partly covered), the covered ones follow
        the visible ones' rigid motion."""
        found = [i for i in range(4) if props[i] is not None]
        new = self.raw.copy()
        for i in found:
            new[i] = props[i]
        if self._sane(new):
            return new, np.array([p is not None for p in props])
        if len(found) >= 2:
            A, _ = cv2.estimateAffinePartial2D(self.raw[found], new[found])
            if A is not None:
                moved = cv2.transform(self.raw.reshape(-1, 1, 2), A).reshape(4, 2)
                moved[found] = new[found]
                if self._sane(moved):
                    return moved.astype(np.float32), np.array([p is not None for p in props])
        # drop the most-displaced proposals until the page is sane again
        order = sorted(found, key=lambda i: -float(np.linalg.norm(new[i] - self.raw[i])))
        keep = set(found)
        for i in order:
            keep.discard(i)
            new[i] = self.raw[i]
            self._pending[i] = None
            if self._sane(new):
                break
        return new, np.array([i in keep for i in range(4)])

    # ------------------------- whole-page re-acquisition ------------------- #
    def _init_template(self, frame: np.ndarray) -> None:
        sw, sh = self.scan_size
        k = 640.0 / max(sw, sh)
        self._tsize = (int(sw * k), int(sh * k))
        tmpl = cv2.cvtColor(warp_document(frame, self.raw, size=self._tsize), cv2.COLOR_BGR2GRAY)
        self._tkp, self._tdesc = self._orb.detectAndCompute(tmpl, None)
        self._app_size = (36, 50) if sw < sh else (50, 36)
        self._app_tmpl = self._small(frame, self.raw)
        self._app_textured = float(self._app_tmpl.std()) > 6.0
        tw, th = self._tsize
        self._tcorners = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])

    def _small(self, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(warp_document(frame, quad, size=(self._app_size[0] * 4, self._app_size[1] * 4)),
                         cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, self._app_size, interpolation=cv2.INTER_AREA).astype(np.float32)

    def _appearance(self, frame: np.ndarray, quad: np.ndarray) -> float:
        """Similarity (0..1) between the page content now and at lock time."""
        a, b = self._app_tmpl, self._small(frame, quad)
        if self._app_textured:
            a0, b0 = a - a.mean(), b - b.mean()
            den = float(np.sqrt((a0 * a0).sum() * (b0 * b0).sum()))
            return max(0.0, float((a0 * b0).sum()) / den) if den > 1e-6 else 0.0
        return float(np.clip(1.0 - abs(a.mean() - b.mean()) / 40.0, 0, 1) *
                     np.clip(1.0 - (b.std() - 6.0) / 20.0, 0, 1))

    def _reacquire(self, frame: np.ndarray, gray: np.ndarray, detector) -> Optional[np.ndarray]:
        quad = self._match_template(gray)
        if quad is None and detector is not None:
            quad = self._pick_detection(detector.detect(frame))
        if quad is None or self._appearance(frame, quad) < 0.35:
            return None
        ref, n = refine_quad(frame, quad)
        return ref if n >= 3 else quad

    def _match_template(self, gray: np.ndarray) -> Optional[np.ndarray]:
        if self._tdesc is None or len(self._tkp) < 15:
            return None  # blank page: nothing to match
        kp, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or len(kp) < 15:
            return None
        good = [m for m, *rest in self._matcher.knnMatch(self._tdesc, desc, k=2)
                if rest and m.distance < 0.8 * rest[0].distance]
        if len(good) < 15:
            return None
        src = np.float32([self._tkp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if Hm is None or inl.sum() < 12 or inl.mean() < 0.3:
            return None
        quad = cv2.perspectiveTransform(self._tcorners.reshape(-1, 1, 2), Hm).reshape(4, 2) / self._s
        quad = quad.astype(np.float32)
        return quad if self._sane(quad) else None

    def _pick_detection(self, dets) -> Optional[np.ndarray]:
        """Blank page: the detection closest to the last position with a similar size and shape."""
        c0, a0 = self.raw.mean(0), quad_area(self.raw)
        w0, h0 = quad_size(self.raw)
        best, best_d = None, np.inf
        for d in dets:
            a = quad_area(d.corners)
            w, h = quad_size(d.corners)
            if not (0.5 * a0 < a < 2.0 * a0):
                continue
            if abs(np.log((max(w, h) / min(w, h)) / (max(w0, h0) / min(w0, h0)))) > 0.25:
                continue
            dist = float(np.linalg.norm(d.corners.mean(0) - c0))
            if dist < best_d:
                best, best_d = d.corners, dist
        if best is None:
            return None
        k = min(range(4), key=lambda k: float(np.linalg.norm(np.roll(best, -k, 0) - self.raw, axis=1).sum()))
        return np.roll(best, -k, 0).astype(np.float32)


def _sides(q: np.ndarray) -> np.ndarray:
    """Lengths of sides i -> i+1 (index i = side starting at corner i)."""
    q = np.asarray(q, np.float32)
    return np.linalg.norm(np.roll(q, -1, 0) - q, axis=1)
