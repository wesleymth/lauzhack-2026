"""Document detectors.

`Detector` is the interface the tracker and app depend on; any implementation
(classical, or later a learned keypoint model) just returns a list of
`Detection`s with corners in full-frame pixel coordinates, ordered TL, TR, BR, BL.

The OpenCV detector makes no assumption about colours: white paper on a dark
desk, dark paper on a white desk and coloured paper all work. A document is
recognised by its *structure* instead:
  * a 4-sided outline whose every side is backed by an image edge,
  * an interior dominated by one colour (the paper, whatever its colour)
    with sparse marks on it (text), i.e. not a photo, screen or keyboard,
  * a paper-like aspect ratio.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np

from .geometry import (interior_angles, order_corners, quad_area, quad_iou, quad_size, refine_quad,
                       warp_document)


@dataclass
class Detection:
    corners: np.ndarray  # (4, 2) float32, TL TR BR BL, full-frame pixels
    score: float
    info: Dict[str, float] = field(default_factory=dict)


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]: ...


@dataclass
class DetectorConfig:
    work_width: int = 640              # detection runs on a frame downscaled to this width
    min_area_frac: float = 0.015       # of the frame
    max_area_frac: float = 0.90
    min_angle: float = 45.0            # interior angle bounds (perspective tolerant)
    max_angle: float = 135.0
    min_rectangularity: float = 0.85   # contour area / quad area
    min_aspect: float = 1.12           # long/short side; A4 = 1.414, Letter = 1.294
    max_aspect: float = 3.2            # receipts
    # edge support: fraction of points along each side lying on an image edge
    edge_grad_thr: float = 10.0        # Lab gradient magnitude counted as an edge
    min_side_support: float = 0.45     # weakest side (a hand may cover part of one)
    min_mean_support: float = 0.70
    # interior: one dominant colour + sparse marks
    fill_delta: float = 16.0           # Lab distance to the paper colour still counted as paper
    min_fill: float = 0.55             # fraction of the interior that is paper colour
    max_edge_density: float = 0.35     # keyboards / busy photos are denser than text
    score_threshold: float = 0.35
    nms_iou: float = 0.5
    nested_delta: float = 20.0         # inner quad with this paper colour = part of outer doc
    patch_long_side: int = 128
    refine: bool = True                # snap corners to the edges at full resolution
    doc_up: str = "down"               # where the document's top points in the image


class OpenCVDocumentDetector:
    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.cfg = config or DetectorConfig()
        self.debug: Dict[str, np.ndarray] = {}
        self._k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self._k7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    # ------------------------------------------------------------------ #
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        cfg = self.cfg
        H, W = frame_bgr.shape[:2]
        scale = min(1.0, cfg.work_width / float(W))
        small = cv2.resize(frame_bgr, (int(W * scale), int(H * scale)),
                           interpolation=cv2.INTER_AREA) if scale < 1.0 else frame_bgr
        h, w = small.shape[:2]
        frame_area = float(h * w)

        lab_raw = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        lab = cv2.GaussianBlur(lab_raw, (5, 5), 0)
        grad = self._gradient(lab)
        edges = self._edge_map(lab)
        masks = self._region_masks(lab)
        self.debug = {"edges": edges, "gradient": np.clip(grad * 4, 0, 255).astype(np.uint8),
                      "regions": masks[0] if masks else np.zeros_like(edges)}
        # tolerate +-1-2 px misalignment between the fitted side and the true edge
        grad_d = cv2.dilate(grad, self._k5)

        contours: List[np.ndarray] = []
        contours.extend(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0])
        for m in masks:
            contours.extend(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])

        min_a, max_a = cfg.min_area_frac * frame_area, cfg.max_area_frac * frame_area
        cands: List[Tuple[np.ndarray, float]] = []
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a < min_a or a > max_a:
                continue
            quad = self._fit_quad(cnt, cfg.doc_up)
            if quad is None:
                continue
            qa = quad_area(quad)
            if qa < min_a or qa > max_a:
                continue
            rect = min(a / qa, qa / a)
            if rect < cfg.min_rectangularity:
                continue
            ang = interior_angles(quad)
            if ang.min() < cfg.min_angle or ang.max() > cfg.max_angle:
                continue
            cands.append((quad, rect))

        # many contours give near-identical quads: dedupe before the costlier scoring
        uniq: List[Tuple[np.ndarray, float]] = []
        for q, r in cands:
            if all(quad_iou(q, u) < 0.9 for u, _ in uniq):
                uniq.append((q, r))

        dets: List[Detection] = []
        for quad, rect in uniq:
            score, info = self._document_score(small, lab_raw, grad_d, quad)
            info["rect"] = rect
            if score >= cfg.score_threshold:
                dets.append(Detection(corners=(quad / scale).astype(np.float32),
                                      score=score, info=info))
        dets = self._nms(dets)
        if cfg.refine:
            for d in dets:
                ref, n = refine_quad(frame_bgr, d.corners)
                if n >= 3:
                    d.corners = ref
        return dets

    # --------------------------- candidates ----------------------------- #
    @staticmethod
    def _gradient(lab: np.ndarray) -> np.ndarray:
        """Max over L, a, b of the Sobel magnitude: an edge in brightness OR colour."""
        out = None
        for c in cv2.split(lab):
            gx = cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)
            m = cv2.magnitude(gx, gy) / 4.0  # ~ per-pixel intensity step
            out = m if out is None else np.maximum(out, m)
        return out

    def _edge_map(self, lab: np.ndarray) -> np.ndarray:
        L, a, b = cv2.split(lab)
        e = cv2.Canny(L, 25, 75)
        # chroma channels have a smaller range: lower thresholds
        e |= cv2.Canny(a, 12, 36)
        e |= cv2.Canny(b, 12, 36)
        return cv2.dilate(e, self._k3, iterations=1)

    def _region_masks(self, lab: np.ndarray) -> List[np.ndarray]:
        """Otsu foreground/background splits on each Lab channel, both polarities.
        Light-on-dark, dark-on-light and colour-on-neutral all appear in one of them."""
        masks = []
        for i, c in enumerate(cv2.split(lab)):
            if i > 0 and float(c.std()) < 4.0:   # no colour contrast in this channel
                continue
            _, m = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            for mm in (m, cv2.bitwise_not(m)):
                mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, self._k7)  # fill in text
                mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, self._k3)
                masks.append(mm)
        return masks

    @staticmethod
    def _fit_quad(cnt: np.ndarray, up: str = "up") -> Optional[np.ndarray]:
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        for eps in (0.015, 0.02, 0.03, 0.04, 0.05, 0.06):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                return order_corners(approx.reshape(4, 2).astype(np.float32), up)
            if len(approx) < 4:
                return None
        return None

    # ----------------------------- scoring ------------------------------ #
    def _side_support(self, grad_d: np.ndarray, quad: np.ndarray, n: int = 40) -> np.ndarray:
        h, w = grad_d.shape
        t = np.linspace(0.1, 0.9, n, dtype=np.float32)[:, None]  # skip the corners themselves
        sup = []
        for i in range(4):
            p0, p1 = quad[i], quad[(i + 1) % 4]
            pts = p0 + t * (p1 - p0)
            xs = np.clip(pts[:, 0].round().astype(int), 0, w - 1)
            ys = np.clip(pts[:, 1].round().astype(int), 0, h - 1)
            sup.append(float((grad_d[ys, xs] > self.cfg.edge_grad_thr).mean()))
        return np.array(sup)

    def _document_score(self, small: np.ndarray, lab: np.ndarray, grad_d: np.ndarray,
                        quad: np.ndarray) -> Tuple[float, Dict[str, float]]:
        cfg = self.cfg
        # 1) every side is a real edge
        sup = self._side_support(grad_d, quad)
        edge_s = _ramp(sup.min(), cfg.min_side_support - 0.2, cfg.min_side_support) * \
            _ramp(sup.mean(), cfg.min_mean_support - 0.2, cfg.min_mean_support)

        # 2) interior: a dominant "paper" colour (any colour) with sparse marks
        qw, qh = quad_size(quad)
        s = cfg.patch_long_side / max(qw, qh, 1.0)
        pw, ph = max(int(qw * s), 16), max(int(qh * s), 16)
        patch = warp_document(lab, quad, size=(pw, ph)).astype(np.float32)
        m = max(2, int(0.06 * min(pw, ph)))
        patch = patch[m:ph - m, m:pw - m]
        paper = np.median(patch.reshape(-1, 3), axis=0)
        d = patch - paper
        d[..., 0] *= 0.5  # tolerate shading / uneven light across the sheet
        fill = float((np.sqrt((d * d).sum(axis=2)) < cfg.fill_delta).mean())
        fill_s = _ramp(fill, cfg.min_fill - 0.2, cfg.min_fill)

        L = cv2.GaussianBlur(patch[..., 0].astype(np.uint8), (3, 3), 0)
        edge_density = float((cv2.Canny(L, 40, 120) > 0).mean())
        texture_s = _ramp(-edge_density, -(cfg.max_edge_density + 0.12), -cfg.max_edge_density)

        # 3) paper-like shape (soft: perspective distorts it)
        aspect = max(qw, qh) / max(min(qw, qh), 1.0)
        if aspect < cfg.min_aspect:   # squares (coasters, sticky notes, boxes) are rarely pages
            aspect_s = 0.15 + 0.85 * _ramp(aspect, 1.02, cfg.min_aspect)
        else:                         # very elongated: soft penalty, receipts exist
            aspect_s = 0.3 + 0.7 * _ramp(-aspect, -(cfg.max_aspect + 0.8), -cfg.max_aspect)

        score = edge_s * fill_s * texture_s * aspect_s
        info_paper = {"paper_L": float(paper[0]), "paper_a": float(paper[1]), "paper_b": float(paper[2])}
        return score, {**info_paper, "support_min": float(sup.min()), "support_mean": float(sup.mean()),
                       "fill": fill, "edges": edge_density, "aspect": aspect,
                       "edge_s": edge_s, "fill_s": fill_s, "texture_s": texture_s}

    def _nms(self, dets: List[Detection]) -> List[Detection]:
        # A quad inside another one with the same paper colour is a text block /
        # margin of that document, not a separate document: drop it.
        dets.sort(key=lambda d: quad_area(d.corners), reverse=True)
        outer: List[Detection] = []
        for d in dets:
            if not any(self._nested_same_paper(d, o) for o in outer):
                outer.append(d)
        outer.sort(key=lambda d: d.score, reverse=True)
        kept: List[Detection] = []
        for d in outer:
            if all(quad_iou(d.corners, k.corners) < self.cfg.nms_iou for k in kept):
                kept.append(d)
        return kept

    def _nested_same_paper(self, inner: Detection, outer: Detection) -> bool:
        inter, _ = cv2.intersectConvexConvex(inner.corners, outer.corners)
        if inter < 0.9 * quad_area(inner.corners):
            return False
        c = np.array([inner.info[k] - outer.info[k] for k in ("paper_L", "paper_a", "paper_b")])
        c[0] *= 0.5
        return float(np.linalg.norm(c)) < self.cfg.nested_delta


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 below lo, 1 above hi, linear in between."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
