"""Quad helpers. A quad is a (4, 2) float32 array ordered TL, TR, BR, BL."""

from typing import Optional, Tuple

import cv2
import numpy as np


# Direction (in the image) that the TOP of the document points to.
UP_VECTORS = {"up": (0.0, -1.0), "down": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}


def order_corners(pts: np.ndarray, up: str = "up") -> np.ndarray:
    """Order 4 points as the document's TL, TR, BR, BL (clockwise on the page).

    `up` is where the top of the document points in the image: "up" for a page
    read from the camera's side, "down" when the page faces a user sitting at
    the bottom of the camera view (top of the page towards their keyboard)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # y points down, so increasing angle = clockwise on screen
    u = np.array(UP_VECTORS[up], np.float32)
    left = np.array([u[1], -u[0]], np.float32)  # document's left, in image coordinates
    start = int(np.argmax((pts - c) @ (u + left)))
    return np.roll(pts, -start, axis=0)


def quad_area(q: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(q, np.float32).reshape(-1, 1, 2))))


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float32).reshape(-1, 2)
    b = np.asarray(b, np.float32).reshape(-1, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    if inter <= 0:
        return 0.0
    union = quad_area(a) + quad_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def point_in_quad(q: np.ndarray, x: float, y: float) -> bool:
    cnt = np.asarray(q, np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0


def quad_size(q: np.ndarray) -> Tuple[float, float]:
    """(width, height) of the flattened quad, from the longer opposite edges."""
    tl, tr, br, bl = np.asarray(q, np.float32)
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    return float(w), float(h)


def interior_angles(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float32)
    out = []
    for i in range(4):
        p, a, b = q[i], q[i - 1], q[(i + 1) % 4]
        v1, v2 = a - p, b - p
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        out.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return np.array(out)


def warp_document(frame: np.ndarray, q: np.ndarray,
                  size: Optional[Tuple[int, int]] = None,
                  max_side: int = 1200) -> np.ndarray:
    """Perspective-correct the quad region into a flat top-down image."""
    q = np.asarray(q, np.float32)
    if size is None:
        w, h = quad_size(q)
        s = min(1.0, max_side / max(w, h, 1.0))
        size = (max(int(w * s), 8), max(int(h * s), 8))
    w, h = size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- #
# Sub-pixel corner refinement: snap each side to the strongest nearby edge
# --------------------------------------------------------------------------- #

def _intersect(p, d, q, e) -> Optional[np.ndarray]:
    """Intersection of lines p + a*d and q + b*e."""
    A = np.array([[d[0], -e[0]], [d[1], -e[1]]], np.float64)
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    a, _ = np.linalg.solve(A, np.asarray(q, np.float64) - np.asarray(p, np.float64))
    return np.asarray(p, np.float64) + a * np.asarray(d, np.float64)


def refine_quad(img_bgr: np.ndarray, quad: np.ndarray, search: Optional[float] = None,
                n: int = 32, min_strength: float = 8.0,
                min_frac: float = 0.4) -> Tuple[np.ndarray, int]:
    """Move each side of `quad` onto the strongest brightness-or-colour edge
    within +-`search` px along its normal (robust line fit, sub-pixel), then
    intersect adjacent sides. Sides without enough edge support (e.g. covered
    by a hand) keep their input position. Returns (corners, n_sides_refined)."""
    q = np.asarray(quad, np.float32).reshape(4, 2)
    H, W = img_bgr.shape[:2]
    diag = float(np.linalg.norm(q[2] - q[0]))
    r = int(round(search if search is not None else np.clip(0.02 * diag, 5, 16)))
    x0, y0 = np.floor(q.min(0)).astype(int) - r - 3
    x1, y1 = np.ceil(q.max(0)).astype(int) + r + 4
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return q.copy(), 0
    roi = cv2.GaussianBlur(cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2LAB), (3, 3), 0)
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    h, w = roi.shape[:2]
    off0 = np.array([x0, y0], np.float32)
    t = np.linspace(0.08, 0.92, n, dtype=np.float32)
    offs = np.arange(-r, r + 1, dtype=np.float32)
    m = len(offs)
    ar = np.arange(n)

    lines, ok = [], 0
    for i in range(4):
        p0, p1 = q[i] - off0, q[(i + 1) % 4] - off0
        d = p1 - p0
        u = d / (np.linalg.norm(d) + 1e-9)
        nrm = np.array([-u[1], u[0]], np.float32)
        if np.dot(nrm, (q.mean(0) - off0) - (p0 + 0.5 * d)) > 0:
            nrm = -nrm                                                # outward normal
        base = p0 + t[:, None] * d                                    # (n, 2)
        pts = base[:, None, :] + offs[None, :, None] * nrm            # (n, m, 2)
        xi = np.clip(np.rint(pts[..., 0]).astype(int), 0, w - 1)
        yi = np.clip(np.rint(pts[..., 1]).astype(int), 0, h - 1)
        g = np.abs(gx[yi, xi] * nrm[0] + gy[yi, xi] * nrm[1]).max(-1) / 4.0  # (n, m)
        # strong local maxima; take the OUTERMOST (not simply the strongest: a
        # title bar or border line inside the page can be stronger than its edge)
        lm = np.ones_like(g, bool)
        lm[:, 1:] &= g[:, 1:] >= g[:, :-1]
        lm[:, :-1] &= g[:, :-1] >= g[:, 1:]
        strong = lm & (g > np.maximum(min_strength, 0.4 * g.max(1, keepdims=True)))
        k = np.where(strong, -offs[None, :], np.inf).argmin(1)
        k = np.clip(k, 1, m - 2)
        g0, g1, g2 = g[ar, k - 1], g[ar, k], g[ar, k + 1]
        den = g0 - 2 * g1 + g2
        delta = np.where(np.abs(den) > 1e-6, 0.5 * (g0 - g2) / np.where(den == 0, 1, den), 0.0)
        off = offs[k] + np.clip(delta, -0.5, 0.5)
        good = g1 > min_strength
        if good.mean() < min_frac:
            lines.append((p0, d))
            continue
        P = (base[good] + off[good][:, None] * nrm).astype(np.float32)
        vx, vy, cx, cy = cv2.fitLine(P, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([cx, cy]), np.array([vx, vy])))
        ok += 1

    out = q.copy()
    for i in range(4):
        (p, d), (pp, e) = lines[i - 1], lines[i]
        c = _intersect(p, d, pp, e)
        if c is not None:
            out[i] = c + off0
    if float(np.linalg.norm(out - q, axis=1).max()) > 2.5 * r:
        return q.copy(), 0
    return out.astype(np.float32), ok


# --------------------------------------------------------------------------- #
# True aspect ratio of a rectangle seen in perspective, and paper sizes
# --------------------------------------------------------------------------- #

PAPER_RATIOS = {"a4": 2 ** 0.5, "letter": 11 / 8.5}  # long side / short side


def estimate_aspect(quad: np.ndarray, image_size: Tuple[int, int]) -> float:
    """Real width/height (TL->TR is the width) of the planar rectangle imaged as
    `quad`, compensating perspective (Zhang & He 2007, principal point at the
    image centre, focal length estimated). Falls back to the image-space
    ratio when the view is (near) fronto-parallel."""
    q = np.asarray(quad, np.float64).reshape(4, 2)
    W, H = image_size
    tl, tr, br, bl = q - np.array([W / 2.0, H / 2.0])
    m1, m2, m3, m4 = (np.array([p[0], p[1], 1.0]) for p in (tl, tr, bl, br))
    qw, qh = quad_size(q)
    simple = qw / max(qh, 1e-6)
    try:
        k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
        k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
    except ZeroDivisionError:
        return simple
    n2, n3 = k2 * m2 - m1, k3 * m3 - m1
    if abs(n2[2]) < 1e-6 or abs(n3[2]) < 1e-6:
        return simple
    f2 = -(n2[0] * n3[0] + n2[1] * n3[1]) / (n2[2] * n3[2])
    if not np.isfinite(f2) or f2 <= 0 or not (0.3 * W < np.sqrt(f2) < 8 * W):
        return simple
    num = (n2[0] ** 2 + n2[1] ** 2) / f2 + n2[2] ** 2
    den = (n3[0] ** 2 + n3[1] ** 2) / f2 + n3[2] ** 2
    r = float(np.sqrt(num / den))
    return r if np.isfinite(r) and 0.1 < r < 10 else simple


def paper_size(aspect_wh: float, paper: str = "a4", long_side: int = 1100) -> Tuple[int, int]:
    """Fixed (w, h) scan size: a standard paper ratio, oriented like `aspect_wh`.
    paper: 'a4', 'letter', or 'auto' (closest of the two)."""
    long_short = max(aspect_wh, 1.0 / aspect_wh)
    if paper == "auto":
        ratio = min(PAPER_RATIOS.values(), key=lambda r: abs(np.log(long_short / r)))
    else:
        ratio = PAPER_RATIOS[paper]
    short = int(round(long_side / ratio))
    return (long_side, short) if aspect_wh >= 1.0 else (short, long_side)
