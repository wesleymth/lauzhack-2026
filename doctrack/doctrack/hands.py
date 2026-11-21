"""Hand-based document selection: point at a document for 1 s to lock it.

Hand landmarks come from MediaPipe Hand Landmarker, using the gesture logic of
the team's hand tracker (../Hand_tracker/hand_realtime.py: `pointing_check`):
index finger extended and straight, other fingers curled, thumb away from the
index (so holding a pen / writing does not count). It runs on the same frames
as the document tracker; only one program can open the camera.
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import cv2
import numpy as np

from .geometry import point_in_quad, quad_area

HAND_DIR = Path(__file__).resolve().parents[2] / "Hand_tracker"
INDEX_TIP = 8


@dataclass
class Hand:
    landmarks: np.ndarray              # (21, 2) float32, full-frame pixels
    handedness: str
    pointing: bool
    reason: str = ""

    @property
    def tip(self) -> np.ndarray:
        return self.landmarks[INDEX_TIP]

    @property
    def hull(self) -> np.ndarray:
        """Convex outline of the hand, used to tell the tracker what it covers."""
        return cv2.convexHull(self.landmarks.astype(np.float32)).reshape(-1, 2)


class HandTracker:
    def __init__(self, model_path: Optional[str] = None, max_hands: int = 2, width: int = 640,
                 point_dir: str = "down", dir_tolerance: float = 60.0,
                 straight_deg: float = 150.0, pinch_ratio: float = 0.45) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if str(HAND_DIR) not in sys.path:
            sys.path.insert(0, str(HAND_DIR))
        import hand_realtime as hr  # reuse the team's gesture logic and model download

        self._mp = mp
        self._hr = hr
        self.connections = hr.CONNECTIONS
        model = model_path or str(HAND_DIR / "hand_landmarker.task")
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=hr.ensure_model(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self.width = width
        self._gesture = SimpleNamespace(point_dir=point_dir, dir_tolerance=dir_tolerance,
                                        straight_deg=straight_deg, pinch_ratio=pinch_ratio)
        self._t0 = time.perf_counter()
        self._last_ms = -1

    def detect(self, frame: np.ndarray, t: Optional[float] = None) -> List[Hand]:
        h, w = frame.shape[:2]
        small = frame
        if self.width and w > self.width:
            small = cv2.resize(frame, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        t = time.perf_counter() if t is None else t
        ms = max(int((t - self._t0) * 1000), self._last_ms + 1)  # must strictly increase
        self._last_ms = ms
        res = self._landmarker.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ms)
        hands = []
        for lm, hd in zip(res.hand_landmarks, res.handedness):
            pts = np.array([[p.x * w, p.y * h] for p in lm], np.float32)
            pointing, reason = self._hr.pointing_check(lm, w, h, self._gesture)
            hands.append(Hand(pts, hd[0].category_name, bool(pointing), reason))
        return hands

    def close(self) -> None:
        self._landmarker.close()


@dataclass
class DwellSelector:
    """Locks the document a fingertip stays on for `dwell_s` seconds.

    * only documents shown by the tracker can be selected;
    * with `require_pointing`, only a pointing hand counts (not one writing);
    * short tracking dropouts (< `grace_s`) don't reset the timer;
    * after a release, the finger must leave the documents before it can select
      again (otherwise a finger still resting there would re-lock at once).
    """
    dwell_s: float = 1.0
    grace_s: float = 0.3
    require_pointing: bool = True
    target: Optional[int] = None       # track id under the finger
    tip: Optional[np.ndarray] = None   # fingertip being timed (for drawing)
    _start: float = 0.0
    _last_seen: float = 0.0
    _armed: bool = True
    _was_locked: bool = field(default=False, repr=False)

    def progress(self, t: float) -> float:
        if self.target is None:
            return 0.0
        return float(np.clip((t - self._start) / self.dwell_s, 0.0, 1.0))

    def update(self, tracker, hands: List[Hand], t: float) -> Optional[int]:
        """Feed this frame's hands. Returns the track id if a document was just locked."""
        if tracker.locked:
            self._reset()
            self._was_locked = True
            return None
        if self._was_locked:               # just released: wait for the finger to leave
            self._was_locked, self._armed = False, False

        target, tip = None, None
        docs = sorted(tracker.tracks, key=lambda d: quad_area(d.corners))  # innermost first
        for h in hands:
            if self.require_pointing and not h.pointing:
                continue
            for d in docs:
                if point_in_quad(d.corners, *h.tip):
                    target, tip = d.id, h.tip
                    break
            if target is not None:
                break

        if not self._armed:
            if target is None:
                self._armed = True
            return None
        if target is None:
            if self.target is not None and t - self._last_seen > self.grace_s:
                self._reset()
            return None
        if target != self.target:
            self.target, self._start = target, t
        self._last_seen, self.tip = t, tip
        if t - self._start >= self.dwell_s and tracker.select(target):
            self._reset()
            self._armed = False
            return target
        return None

    def _reset(self) -> None:
        self.target, self.tip = None, None
