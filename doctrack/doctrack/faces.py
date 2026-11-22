"""Face recognition: OpenCV YuNet (detect) + SFace (128-d embedding). No extra dependency.
Enrolled embeddings are stored locally in faces.npz."""

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models"
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
MODELS = {
    "yunet.onnx": _ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface.onnx": _ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def _ensure(name: str) -> str:
    path = MODEL_DIR / name
    if not path.exists():
        MODEL_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MODELS[name], path)
    return str(path)


@dataclass
class Face:
    box: tuple          # (x, y, w, h), full-frame pixels
    name: str           # enrolled name or "unknown"
    score: float        # cosine similarity to the best match


class FaceRecognizer:
    def __init__(self, db_path: str = "faces.npz", width: int = 320, every: int = 5,
                 threshold: float = 0.363) -> None:  # 0.363 = SFace's recommended cosine threshold
        self._det = cv2.FaceDetectorYN.create(_ensure("yunet.onnx"), "", (width, width), 0.7)
        self._rec = cv2.FaceRecognizerSF.create(_ensure("sface.onnx"), "")
        self.width, self.every, self.threshold = width, every, threshold
        self.db_path = Path(db_path)
        self.db: Dict[str, np.ndarray] = {}            # name -> (k, 128) unit vectors
        if self.db_path.exists():
            with np.load(self.db_path) as z:
                self.db = {k: z[k] for k in z.files}
        self._n = 0
        self._last: List[Face] = []

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        s = min(1.0, self.width / w)
        small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame
        self._det.setInputSize((small.shape[1], small.shape[0]))
        _, rows = self._det.detect(small)
        if rows is None:
            return np.zeros((0, 15), np.float32)
        rows = rows.copy()
        rows[:, :14] /= s                              # box + 5 landmarks back to full-frame pixels
        return rows

    def _embed(self, frame: np.ndarray, row: np.ndarray) -> np.ndarray:
        f = self._rec.feature(self._rec.alignCrop(frame, row)).ravel()
        return f / (np.linalg.norm(f) + 1e-9)

    def _nearest(self, box):
        """Previous face at about the same place (same person, one frame earlier)."""
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        best, best_d = None, 0.6 * max(box[2], box[3])
        for p in self._last:
            d = np.hypot(p.box[0] + p.box[2] / 2 - cx, p.box[1] + p.box[3] / 2 - cy)
            if d < best_d:
                best, best_d = p, d
        return best

    def identify(self, frame: np.ndarray) -> List[Face]:
        """Boxes are tracked every frame; the (costlier) recognition runs every `every` frames,
        or at once for a face that just appeared."""
        self._n += 1
        refresh = self.every <= 1 or self._n % self.every == 1
        out = []
        for row in self._detect(frame):
            box = np.array(row[:4], np.float32)
            prev = self._nearest(tuple(box))
            if prev is not None:                       # smooth the box so the thumbnail doesn't jitter
                box = 0.6 * box + 0.4 * np.array(prev.box, np.float32)
            box = tuple(int(v) for v in box)
            if prev is not None and not refresh:
                out.append(Face(box, prev.name, prev.score))
                continue
            e = self._embed(frame, row)
            name, best = "unknown", -1.0
            for n, embs in self.db.items():
                s = float((embs @ e).max())
                if s > best:
                    best, name = s, n
            if best < self.threshold:
                name = "unknown"
            out.append(Face(box, name, best))
        self._last = out
        return out

    def enroll(self, name: str, frame: np.ndarray) -> bool:
        """Add the largest face in `frame` under `name`. Call a few times (angles, light)."""
        rows = self._detect(frame)
        if not len(rows):
            return False
        row = max(rows, key=lambda r: r[2] * r[3])
        e = self._embed(frame, row)[None]
        self.db[name] = np.vstack([self.db[name], e]) if name in self.db else e
        np.savez(self.db_path, **self.db)
        return True


def draw(frame: np.ndarray, faces: List[Face]) -> None:
    for f in faces:
        x, y, w, h = f.box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 3)
        #cv2.putText(frame, f.name, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        
        
def draw_thumbs(vis: np.ndarray, frame: np.ndarray, faces: List[Face],
                size: float = 0.2, max_n: int = 3, pad: float = 0.45, min_px: int = 140) -> None:
    """Small live crop of each IDENTIFIED face, stacked up from the bottom-right corner of `vis`.
    The crop is cut from `frame`, which may have a different size than `vis` (e.g. the scan)."""
    fH, fW = frame.shape[:2]          # where the crop comes from
    H, W = vis.shape[:2]              # where it is drawn
    s, m = min(max(int(size * W), min_px), W - 2 * 12, H - 2 * 12), 12
    if s < 16:
        return
    y = H - m
    for f in faces[:max_n]:
        x0, y0, w, h = f.box
        half = max(w, h) // 2 + int(pad * max(w, h))
        cx, cy = x0 + w // 2, y0 + h // 2
        xa, ya, xb, yb = cx - half, cy - half, cx + half, cy + half
        crop = frame[max(ya, 0):min(yb, fH), max(xa, 0):min(xb, fW)]
        if crop.size == 0:
            continue
        # pad where the square leaves the frame, so the face is never stretched
        crop = cv2.copyMakeBorder(crop, max(-ya, 0), max(yb - fH, 0), max(-xa, 0), max(xb - fW, 0),
                                  cv2.BORDER_CONSTANT, value=(40, 40, 40))
        thumb = cv2.resize(crop, (s, s), interpolation=cv2.INTER_AREA)
        y -= s
        if y < 0:
            break
        x = W - m - s
        vis[y:y + s, x:x + s] = thumb
        y -= m