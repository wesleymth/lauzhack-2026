"""Outline the page you point at, with a pretrained point-prompted segmentation model.

MobileSAM (Segment Anything, mobile version; MIT licence), as ready-made ONNX files
(https://huggingface.co/Acly/MobileSAM), run with onnxruntime. Nothing is trained.

Given the frame and the point you pointed at / clicked, it returns the mask of the
object under that point (the page), from which the 4 page corners are fitted and
then snapped onto the page's real edges. It runs once, when a page is locked
(~0.2-0.5 s on a laptop CPU), not every frame. It replaces the classical browse
detector's corners at lock time, and makes it possible to lock a page that the
browse detector never found (low contrast, cluttered desk).

The two model files (~45 MB) are downloaded on first use into doctrack/models/.
"""

import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .geometry import order_corners, quad_area, refine_quad

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
_HF = "https://huggingface.co/Acly/MobileSAM/resolve/main/"
FILES = ("mobile_sam_image_encoder.onnx", "sam_mask_decoder_single.onnx")
SIDE = 1024                                   # SAM's input long side


def ensure_models() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = MODELS_DIR / name
        if not path.exists():
            print(f"downloading MobileSAM {name} to {path} ...")
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(_HF + name, tmp)
            if tmp.stat().st_size < 1_000_000:
                tmp.unlink()
                raise RuntimeError(f"could not download {name}; put it in {MODELS_DIR} manually")
            tmp.rename(path)
    return MODELS_DIR


def sam_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


class PageSegmenter:
    def __init__(self) -> None:
        import onnxruntime as ort
        d = ensure_models()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.encoder = ort.InferenceSession(str(d / FILES[0]), so, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(d / FILES[1]), so, providers=["CPUExecutionProvider"])

    def mask(self, frame_bgr: np.ndarray, point) -> Optional[np.ndarray]:
        """Binary mask (H, W) of the object under `point` (x, y), or None."""
        H, W = frame_bgr.shape[:2]
        s = SIDE / float(max(H, W))
        small = cv2.resize(frame_bgr, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
        emb = self.encoder.run(None, {"input_image": rgb})[0]
        # one positive point + SAM's "no box" padding point
        coords = np.float32([[[point[0] * s, point[1] * s], [0.0, 0.0]]])
        labels = np.float32([[1, -1]])
        masks, iou, _ = self.decoder.run(None, {
            "image_embeddings": emb, "point_coords": coords, "point_labels": labels,
            "mask_input": np.zeros((1, 1, 256, 256), np.float32), "has_mask_input": np.zeros(1, np.float32),
            "orig_im_size": np.float32([H, W])})
        m = masks[0, 0] > 0.0
        if float(iou.ravel()[0]) < 0.6 or m.sum() < 0.005 * H * W:
            return None
        return m

    def page_quad(self, frame_bgr: np.ndarray, point, min_area_frac: float = 0.01) -> Optional[np.ndarray]:
        """4 page corners (TL TR BR BL, image px) of the page under `point`, or None."""
        m = self.mask(frame_bgr, point)
        if m is None:
            return None
        m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        pt = (float(point[0]), float(point[1]))
        inside = [c for c in cnts if cv2.pointPolygonTest(c, pt, False) >= 0]
        c = max(inside or cnts, key=cv2.contourArea)
        H, W = frame_bgr.shape[:2]
        if cv2.contourArea(c) < min_area_frac * H * W:
            return None
        hull = cv2.convexHull(c)
        quad = None
        peri = cv2.arcLength(hull, True)
        for eps in (0.01, 0.02, 0.03, 0.05):               # simplest 4-sided outline of the hull
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(np.float32)
                break
        if quad is None:                                   # fall back to the tightest rotated box
            quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
        if quad_area(quad) < 0.6 * cv2.contourArea(hull):
            return None
        quad = order_corners(quad)
        quad, _ = refine_quad(frame_bgr, quad)             # snap the sides onto the real edges
        return np.asarray(quad, np.float32)
