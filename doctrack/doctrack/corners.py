"""Local, per-corner document corner locator (used while a document is locked).

A corner is "visible" only if BOTH sides meeting at it can be found as clean,
straight paper boundaries near it: strong edge, paper colour on the inside and
background colour on the outside. A hand or pen over the corner breaks that, so
the corner is reported as covered and the caller keeps its last position.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .geometry import warp_document

_LW = np.array([0.5, 1.0, 1.0], np.float32)  # Lab weights: tolerate shading (L) more than hue


def lab_dist(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    d = (np.asarray(x, np.float32) - np.asarray(c, np.float32)) * _LW
    return np.sqrt((d * d).sum(-1))


@dataclass
class ColourModel:
    paper: np.ndarray          # Lab colour of the paper
    desk: np.ndarray           # Lab colour just outside the paper
    use: bool                  # paper and desk distinguishable by colour?
    paper_tol: float = 32.0

    @classmethod
    def from_frame(cls, frame: np.ndarray, quad: np.ndarray) -> "ColourModel":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        patch = cv2.cvtColor(warp_document(frame, quad, size=(64, 90)), cv2.COLOR_BGR2LAB)
        paper = np.median(patch[8:-8, 8:-8].reshape(-1, 3), axis=0).astype(np.float32)
        q = np.asarray(quad, np.float32)
        c = q.mean(0)
        diag = float(np.linalg.norm(q[2] - q[0]))

        def grow(px):
            d = q - c
            return (c + d * (1 + px / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6))).astype(np.int32)

        m_out = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillConvexPoly(m_out, grow(max(20.0, 0.06 * diag)), 255)
        cv2.fillConvexPoly(m_out, grow(6.0), 0)
        ring = lab[m_out > 0]
        desk = np.median(ring, axis=0).astype(np.float32) if len(ring) > 50 else paper.copy()
        return cls(paper=paper, desk=desk, use=bool(lab_dist(paper, desk) > 15.0))


def _sample(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    xi = np.clip(np.rint(pts[..., 0]).astype(int), 0, w - 1)
    yi = np.clip(np.rint(pts[..., 1]).astype(int), 0, h - 1)
    return img[yi, xi]


def fit_corner(img: np.ndarray, quad: np.ndarray, i: int, center: np.ndarray, r: float,
               seg: float, model: ColourModel, min_strength: float = 8.0,
               min_support: float = 0.6, ns: int = 20) -> Optional[np.ndarray]:
    """Locate corner `i` of `quad` near `center` (full-res pixels).

    The two sides adjacent to the corner are searched within +-r px of their
    expected position, along `seg` px from the corner; for every sample the
    valid paper boundary *closest to the expected position* is kept (search
    goes outward from the last known position). Returns the intersection of
    the two fitted lines, or None if the corner is not cleanly visible."""
    q = np.asarray(quad, np.float32)
    c = np.asarray(center, np.float32)
    cen = q.mean(0) + (c - q[i])
    H, W = img.shape[:2]
    k = W / 1920.0
    R = int(seg + r + 14 * max(k, 0.5))
    x0, y0 = max(int(c[0]) - R, 0), max(int(c[1]) - R, 0)
    x1, y1 = min(int(c[0]) + R + 1, W), min(int(c[1]) + R + 1, H)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    lab8 = cv2.GaussianBlur(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2LAB), (3, 3), 0)
    gx = cv2.Sobel(lab8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lab8, cv2.CV_32F, 0, 1, ksize=3)
    lab = lab8.astype(np.float32)
    o0 = np.array([x0, y0], np.float32)
    cl, cenl = c - o0, cen - o0
    ri = int(round(r))
    offs = np.arange(-ri, ri + 1, dtype=np.float32)
    m = len(offs)
    ar = np.arange(ns)
    d_in = (max(3.0, 5 * k), max(5.0, 9 * k))

    lines = []
    for j in ((i - 1) % 4, (i + 1) % 4):
        u = q[j] - q[i]
        u = u / (np.linalg.norm(u) + 1e-9)
        n = np.array([-u[1], u[0]], np.float32)
        if np.dot(n, cenl - cl) > 0:
            n = -n                                       # outward normal of this side
        s = np.linspace(min(6.0, 0.2 * seg), seg, ns, dtype=np.float32)
        base = cl + s[:, None] * u                       # (ns, 2)
        pts = base[:, None, :] + offs[None, :, None] * n  # (ns, m, 2)
        g = np.abs(_sample(gx, pts) * n[0] + _sample(gy, pts) * n[1]).max(-1) / 4.0
        lm = np.ones_like(g, bool)
        lm[:, 1:] &= g[:, 1:] >= g[:, :-1]
        lm[:, :-1] &= g[:, :-1] >= g[:, 1:]
        valid = (g > min_strength) & lm
        if model.use:
            for dd in d_in:
                ins, outs = _sample(lab, pts - dd * n), _sample(lab, pts + dd * n)
                dp_in = lab_dist(ins, model.paper)
                valid &= (dp_in < model.paper_tol) & (dp_in < lab_dist(ins, model.desk))
                valid &= lab_dist(outs, model.desk) < lab_dist(outs, model.paper)
        # the real paper edge is the OUTERMOST paper->background boundary: inner
        # ones are title bars, borders, text lines, or a hand resting on the page
        cost = np.where(valid, -offs[None, :], np.inf)
        kk = cost.argmin(1)
        has = np.isfinite(cost[ar, kk])
        if has.mean() < min_support:
            return None
        # sub-pixel: parabola through the gradient peak
        kc = np.clip(kk, 1, m - 2)
        g0, g1, g2 = g[ar, kc - 1], g[ar, kc], g[ar, kc + 1]
        den = g0 - 2 * g1 + g2
        delta = np.where(np.abs(den) > 1e-6, 0.5 * (g0 - g2) / np.where(den == 0, 1, den), 0.0)
        off = offs[kk] + np.where(kk == kc, np.clip(delta, -0.5, 0.5), 0.0)
        P = (base[has] + off[has][:, None] * n).astype(np.float32)
        vx, vy, px, py = cv2.fitLine(P, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        resid = np.abs((P[:, 0] - px) * vy - (P[:, 1] - py) * vx)
        inl = resid < 2.0 * max(k, 0.5)
        if inl.mean() < 0.75:                            # not straight: a hand, not paper
            return None
        vx, vy, px, py = cv2.fitLine(P[inl], cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines.append((np.array([px, py]), np.array([vx, vy])))

    (p, d), (pp, e) = lines
    cosang = abs(float(np.dot(d, e)))
    if cosang > 0.77:                                    # sides must meet at ~40..140 deg
        return None
    A = np.array([[d[0], -e[0]], [d[1], -e[1]]])
    a, _ = np.linalg.solve(A, pp - p)
    corner = (p + a * d + o0).astype(np.float32)
    if float(np.linalg.norm(corner - c)) > 1.5 * r + 3:
        return None
    return corner
