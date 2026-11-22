"""Locked-document tracker that follows the page's CONTENT (no learned model, no training).

The page is treated as a rigid plane. At lock time it is flattened into a reference
image ("page space"). Every frame:

  1. track: a few hundred feature points on the page are followed with pyramidal
     Lucas-Kanade optical flow (forward-backward checked). Each point keeps its
     fixed position in page space, so a robust homography fit (MAGSAC) from page
     space to the image gives the page's pose, and the 4 corners follow from it.
     Only a handful of points need to stay visible: covered corners, or a hand
     sliding the page while resting on it, don't stop the tracking. Points under
     a hand (the hand tracker's outline) are ignored, and points that move with
     the hand are rejected as outliers.
  2. replenish: lost points are replaced by new ones on the visible page.
  3. correct drift: every few frames the current view is matched against the
     lock-time reference with ORB features; a clear disagreement re-anchors the
     pose and the points.
  4. recover: when the flow loses the page, ORB matching against the reference
     looks for it in the whole frame, so the page is found again after being
     covered, lifted, or moved out of view.

A blank page has too little texture to follow: then this tracker hands over to
the DocAligner tracker (vision_track.VisionLockedTracker), whose behaviour is
unchanged.

Same interface as the other lock backends: .corners, .state, .visible,
.scan_size, update(frame, t, detector, occluders).
"""

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import estimate_aspect, paper_size, quad_area, warp_document

TRACKING, PARTIAL, SEARCHING = "tracking", "partial", "searching"
REF_LONG = 640            # long side of the page-space reference image (px)
MIN_TEXTURE = 60          # fewer ORB features than this on the page -> blank page: fall back


class FlowLockedTracker:
    def __init__(self, frame: np.ndarray, corners: np.ndarray, t: float, paper: str = "a4",
                 scan_long_side: int = 1100, occluders: Sequence[np.ndarray] = (), detector=None,
                 fallback=None) -> None:
        H, W = frame.shape[:2]
        self._k = W / 1920.0
        corners = np.asarray(corners, np.float32)
        self.aspect = estimate_aspect(corners, (W, H))
        self.scan_size: Tuple[int, int] = paper_size(self.aspect, paper, scan_long_side)
        sw, sh = self.scan_size
        s = REF_LONG / float(max(sw, sh))
        self._ref_size = (max(8, int(sw * s)), max(8, int(sh * s)))
        rw, rh = self._ref_size
        self._page_rect = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]])
        self._occ = _polys(occluders)
        self._area0 = quad_area(corners)

        self.H = cv2.getPerspectiveTransform(self._page_rect, corners)   # page space -> image
        self.corners = corners.copy()
        self.state = TRACKING
        self.visible = np.ones(4, bool)
        self._fallback = None
        self._frame_i = 0
        self._lost = 0

        gray = _gray(frame)
        self._ref = cv2.warpPerspective(gray, np.linalg.inv(self.H), (rw, rh), flags=cv2.INTER_AREA)
        ref_mask = self._page_mask_ref(occluders)
        self._orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8, fastThreshold=10)
        self._ref_kp, self._ref_des = self._orb.detectAndCompute(self._ref, ref_mask)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        if self._ref_des is None or len(self._ref_kp) < MIN_TEXTURE:
            # blank page: nothing to follow -> the corner-model tracker does the job
            self._fallback = fallback() if fallback else None
            if self._fallback is not None:
                self._sync_from_fallback()
            return
        self._ref_pts = np.float32([kp.pt for kp in self._ref_kp])
        self._prev_gray = gray
        self._pts_img = np.zeros((0, 2), np.float32)    # tracked points, image px
        self._pts_page = np.zeros((0, 2), np.float32)   # the same points, page space
        self._replenish(gray, target=300)

    # ------------------------------------------------------------------ #
    @property
    def uses_fallback(self) -> bool:
        return self._fallback is not None

    def update(self, frame: np.ndarray, t: float, detector=None,
               occluders: Sequence[np.ndarray] = ()) -> np.ndarray:
        if self._fallback is not None:
            self._fallback.update(frame, t, detector, occluders=occluders)
            self._sync_from_fallback()
            return self.corners
        self._occ = _polys(occluders)
        self._frame_i += 1
        gray = _gray(frame)
        ok = self._track(gray) if self.state != SEARCHING else False
        if ok and (self._frame_i % 10 == 0):
            self._correct_drift(gray)                    # periodic check against the reference
        if not ok:
            self._lost += 1
            if self._recover(gray, whole_frame=self._lost > 2):
                self._lost = 0
                ok = True
        if ok:
            self._lost = 0
            if len(self._pts_img) < 180 or self._frame_i % 15 == 0:
                self._replenish(gray, target=300)
            new = cv2.perspectiveTransform(self._page_rect[None], self.H)[0]
            if float(np.abs(new - self.corners).max()) > 0.25 * self._k:   # sub-pixel deadband: still page stays still
                self.corners = new.astype(np.float32)
            self.state = PARTIAL if self._hand_on_page() else TRACKING
        else:
            self.state = SEARCHING
        self.visible = np.array([self.state != SEARCHING and not self._hand_near(c) for c in self.corners])
        self._prev_gray = gray
        return self.corners

    def scan(self, frame: np.ndarray) -> np.ndarray:
        return warp_document(frame, self.corners, size=self.scan_size)

    # ------------------------------------------------------------------ #
    def _track(self, gray: np.ndarray) -> bool:
        if len(self._pts_img) < 8:
            return False
        p0 = self._pts_img.reshape(-1, 1, 2)
        lk = dict(winSize=(21, 21), maxLevel=4,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None, **lk)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, p1, None, **lk)
        fb = np.linalg.norm(p0r.reshape(-1, 2) - self._pts_img, axis=1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
        p1 = p1.reshape(-1, 2)
        good &= ~self._under_hand(p1)
        img, page = p1[good], self._pts_page[good]
        if len(img) < 12:
            self._pts_img, self._pts_page = img, page
            return False
        Hn, inl = cv2.findHomography(page, img, cv2.USAC_MAGSAC, 2.0 * max(self._k, 0.5))
        if Hn is None or inl is None or int(inl.sum()) < 12 or not self._plausible(Hn):
            self._pts_img, self._pts_page = img, page
            return False
        inl = inl.ravel().astype(bool)
        self._pts_img, self._pts_page = img[inl], page[inl]
        self.H = Hn
        return True

    def _replenish(self, gray: np.ndarray, target: int) -> None:
        """Add feature points on the visible page (not under a hand, not next to existing points)."""
        need = target - len(self._pts_img)
        if need < 20:
            return
        mask = np.zeros(gray.shape, np.uint8)
        q = self._shrink(cv2.perspectiveTransform(self._page_rect[None], self.H)[0], 0.97)
        cv2.fillConvexPoly(mask, q.astype(np.int32), 255)
        for o in self._occ:
            cv2.fillPoly(mask, [_grow(o, 1.15).astype(np.int32)], 0)
        for p in self._pts_img:
            cv2.circle(mask, (int(p[0]), int(p[1])), int(10 * self._k) + 4, 0, -1)
        new = cv2.goodFeaturesToTrack(gray, maxCorners=need, qualityLevel=0.01,
                                      minDistance=int(10 * self._k) + 4, mask=mask, blockSize=5)
        if new is None:
            return
        new = new.reshape(-1, 2).astype(np.float32)
        new = cv2.cornerSubPix(gray, new.reshape(-1, 1, 2), (5, 5), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)).reshape(-1, 2)
        page = cv2.perspectiveTransform(new[None], np.linalg.inv(self.H))[0]
        self._pts_img = np.vstack([self._pts_img, new]).astype(np.float32)
        self._pts_page = np.vstack([self._pts_page, page]).astype(np.float32)

    def _orb_pose(self, gray: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> Optional[np.ndarray]:
        """Page pose (page space -> image) from ORB matches with the reference, or None."""
        H_img, W_img = gray.shape
        if roi is not None:
            x0, y0, x1, y1 = roi
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W_img, x1), min(H_img, y1)
            if x1 - x0 < 64 or y1 - y0 < 64:
                return None
            sub, off = gray[y0:y1, x0:x1], np.float32([x0, y0])
        else:
            sub, off = gray, np.float32([0, 0])
        mask = np.full(sub.shape, 255, np.uint8)
        for o in self._occ:
            cv2.fillPoly(mask, [(_grow(o, 1.1) - off).astype(np.int32)], 0)
        orb = cv2.ORB_create(nfeatures=3000 if roi is None else 2000, scaleFactor=1.2, nlevels=8, fastThreshold=10)
        kp, des = orb.detectAndCompute(sub, mask)
        if des is None or len(kp) < 20:
            return None
        pairs = self._matcher.knnMatch(self._ref_des, des, k=2)
        good = [m for m, *n in pairs if n and m.distance < 0.78 * n[0].distance]
        if len(good) < 20:
            return None
        src = self._ref_pts[[m.queryIdx for m in good]]
        dst = np.float32([kp[m.trainIdx].pt for m in good]) + off
        Hn, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0 * max(self._k, 0.5))
        if Hn is None or inl is None or int(inl.sum()) < 20 or not self._plausible(Hn):
            return None
        return Hn

    def _correct_drift(self, gray: np.ndarray) -> None:
        Hn = self._orb_pose(gray, self._roi(1.2))
        if Hn is None:
            return
        a = cv2.perspectiveTransform(self._page_rect[None], self.H)[0]
        b = cv2.perspectiveTransform(self._page_rect[None], Hn)[0]
        if float(np.linalg.norm(a - b, axis=1).max()) > 2.0 * self._k:     # clear drift: re-anchor
            self.H = Hn
            self._reseed(gray)

    def _recover(self, gray: np.ndarray, whole_frame: bool) -> bool:
        Hn = self._orb_pose(gray, None if whole_frame else self._roi(2.0))
        if Hn is None:
            return False
        self.H = Hn
        self._reseed(gray)
        return True

    def _reseed(self, gray: np.ndarray) -> None:
        self._pts_img = np.zeros((0, 2), np.float32)
        self._pts_page = np.zeros((0, 2), np.float32)
        self._replenish(gray, target=300)

    # ------------------------------------------------------------------ #
    def _plausible(self, Hn: np.ndarray) -> bool:
        q = cv2.perspectiveTransform(self._page_rect[None], Hn)[0]
        if not cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.float32)):
            return False
        r = quad_area(q) / max(self._area0, 1.0)
        return 0.4 < r < 2.5

    def _roi(self, scale: float):
        q = cv2.perspectiveTransform(self._page_rect[None], self.H)[0]
        x, y, w, h = cv2.boundingRect(q.astype(np.int32))
        cx, cy, half_w, half_h = x + w / 2, y + h / 2, scale * w / 2, scale * h / 2
        return int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h)

    def _page_mask_ref(self, occluders) -> np.ndarray:
        rw, rh = self._ref_size
        m = np.full((rh, rw), 255, np.uint8)
        border = max(4, int(0.02 * max(rw, rh)))
        cv2.rectangle(m, (0, 0), (rw - 1, rh - 1), 0, border)              # ignore the page edge itself
        Hinv = np.linalg.inv(self.H)
        for o in _polys(occluders):
            po = cv2.perspectiveTransform(_grow(o, 1.15)[None], Hinv)[0]
            cv2.fillPoly(m, [po.astype(np.int32)], 0)
        return m

    def _under_hand(self, pts: np.ndarray) -> np.ndarray:
        out = np.zeros(len(pts), bool)
        for o in self._occ:
            hull = cv2.convexHull(_grow(o, 1.15)).reshape(-1, 1, 2)
            out |= np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0 for x, y in pts])
        return out

    def _hand_on_page(self) -> bool:
        return any(cv2.intersectConvexConvex(self.corners.astype(np.float32),
                                             cv2.convexHull(o).reshape(-1, 2))[0] > 0 for o in self._occ)

    def _hand_near(self, p) -> bool:
        pt = (float(p[0]), float(p[1]))
        return any(cv2.pointPolygonTest(cv2.convexHull(o).reshape(-1, 1, 2), pt, True) > -40.0 * self._k
                   for o in self._occ)

    @staticmethod
    def _shrink(q: np.ndarray, f: float) -> np.ndarray:
        c = q.mean(0)
        return (c + (q - c) * f).astype(np.float32)

    def _sync_from_fallback(self) -> None:
        fb = self._fallback
        self.corners, self.state, self.visible = fb.corners, fb.state, fb.visible
        self.scan_size = fb.scan_size


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def _grow(q: np.ndarray, f: float) -> np.ndarray:
    q = np.asarray(q, np.float32).reshape(-1, 2)
    c = q.mean(0)
    return (c + (q - c) * f).astype(np.float32)


def _polys(occluders) -> List[np.ndarray]:
    return [np.asarray(o, np.float32).reshape(-1, 2) for o in occluders if len(o) >= 3]
