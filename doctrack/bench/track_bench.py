"""Synthetic benchmark for the locked-document tracker.

Renders a textured A4 page (text lines + a logo) on a wood desk at 1080p, moves it
along known paths, adds a hand (also passed to the tracker as an occluder, like
the real hand tracker does), sensor noise and a slight exposure flicker, then
measures how far the tracked corners are from the true ones.

    cd doctrack
    uv run python -m bench.track_bench                       # every backend found
    uv run python -m bench.track_bench --backend docaligner  # one backend
    uv run python -m bench.track_bench --save out/           # also write videos

Scenarios: still (jitter), slide (move + rotate), corner (hand over a corner,
page still), drag (hand on the page while it slides).
Metrics, after the lock: mean / max corner error (px), share of frames whose
error is above 15 px ("off"), mean step-to-step wobble on still frames (px),
and ms per frame of the tracker alone.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from doctrack import DocumentTracker, OpenCVDocumentDetector

W, H = 1920, 1080
PAGE = (850, 1202)                      # A4 page texture, px


def page_texture(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((PAGE[1], PAGE[0], 3), 238, np.uint8)
    cv2.putText(img, "Quarterly report", (70, 120), cv2.FONT_HERSHEY_DUPLEX, 1.6, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.circle(img, (720, 105), 50, (40, 90, 200), -1)                         # a logo
    cv2.rectangle(img, (690, 75), (750, 135), (240, 240, 240), 6)
    y = 220
    words = ["the", "budget", "camera", "notes", "page", "results", "team", "plan", "data", "risk"]
    while y < PAGE[1] - 80:
        x = 70
        while x < PAGE[0] - 140:
            w = str(rng.choice(words))
            cv2.putText(img, w, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (45, 45, 45), 2, cv2.LINE_AA)
            x += 22 * len(w) + 20
        y += int(rng.integers(38, 52))
        if rng.random() < 0.12:
            y += 40                                                            # paragraph break
    return img


def desk(seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros((H, W, 3), np.float32)
    base[:] = (60, 95, 140)                                                    # wood (BGR)
    grain = cv2.resize(rng.normal(0, 1, (H // 8, 1)).astype(np.float32), (W, H))
    base += grain[..., None] * 18
    base += rng.normal(0, 4, (H, W, 1))
    return np.clip(base, 0, 255).astype(np.uint8)


def quad_at(cx, cy, angle_deg, scale=0.62, tilt=0.06):
    """Page corners (TL TR BR BL) for a page centred at (cx, cy), rotated, with mild perspective."""
    w, h = PAGE[0] * scale, PAGE[1] * scale
    pts = np.float32([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    pts[:2, 0] *= (1 - tilt)                                                   # far edge a bit shorter
    a = np.deg2rad(angle_deg)
    R = np.float32([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (pts @ R.T + np.float32([cx, cy])).astype(np.float32)


def hand_poly(tip, angle_deg=200.0, size=1.0):
    """A hand: palm + pointing index finger ending at `tip` (polygon, image px)."""
    a = np.deg2rad(angle_deg)
    d = np.float32([np.cos(a), np.sin(a)])                                     # tip -> wrist direction
    n = np.float32([-d[1], d[0]])
    tip = np.float32(tip)
    s = 160 * size
    pts = [tip + n * 14, tip + d * s * 0.9 + n * 20, tip + d * s * 1.0 + n * 90,
           tip + d * s * 2.2 + n * 95, tip + d * s * 2.4 - n * 60, tip + d * s * 1.1 - n * 80,
           tip + d * s * 0.9 - n * 18, tip - n * 14]
    return np.float32(pts)


def render(bg, tex, quad, hand=None, rng=None, gain=1.0):
    src = np.float32([[0, 0], [PAGE[0], 0], [PAGE[0], PAGE[1]], [0, PAGE[1]]])
    M = cv2.getPerspectiveTransform(src, quad)
    page = cv2.warpPerspective(tex, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full(tex.shape[:2], 255, np.uint8), M, (W, H))
    img = bg.copy()
    img[mask > 127] = page[mask > 127]
    if hand is not None:
        cv2.fillPoly(img, [hand.astype(np.int32)], (120, 150, 205), cv2.LINE_AA)
    img = img.astype(np.float32) * gain
    if rng is not None:
        img += rng.normal(0, 3.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def scenario(name, n=120):
    """Yields (quad, hand_polygon_or_None) per frame."""
    c0 = np.float32([900, 560])
    for i in range(n):
        f = i / (n - 1)
        if name == "still":
            yield quad_at(*c0, 8), None
        elif name == "slide":
            yield quad_at(c0[0] + 260 * np.sin(np.pi * f), c0[1] + 60 * f, 8 + 14 * f), None
        elif name == "corner":
            q = quad_at(*c0, 8)
            on = 0.2 < f < 0.8
            yield q, (hand_poly(q[2] + np.float32([-20, -30]), 30) if on else None)
        elif name == "drag":
            g = np.clip((f - 0.15) / 0.7, 0, 1)
            q = quad_at(c0[0] + 220 * g, c0[1] - 40 * g, 8 - 10 * g)
            tip = q.mean(0) + np.float32([60, 120])
            yield q, (hand_poly(tip, 60) if 0.1 < f < 0.95 else None)


def corner_error(est, true):
    """Mean distance between corresponding corners, best over the 4 cyclic orders
    (the tracker may relabel corners for the reader's orientation)."""
    return min(float(np.linalg.norm(np.roll(est, -k, 0) - true, axis=1).mean()) for k in range(4))


def run(backend, name, tex, bg, save=None):
    rng = np.random.default_rng(3)
    tr = DocumentTracker(OpenCVDocumentDetector(), lock_backend=backend)
    frames = list(scenario(name))
    q0 = frames[0][0]
    t = 0.0
    for _ in range(14):                                     # browse until the page is shown, then lock
        tr.update(render(bg, tex, q0, rng=rng), t)
        t += 1 / 30
    if tr.select_at(*q0.mean(0)) is None:
        return None
    errs, ms, still_steps, prev = [], [], [], None
    writer = None
    for i, (q, hand) in enumerate(frames):
        gain = 1.0 + 0.04 * np.sin(i / 9.0)                 # slight exposure flicker
        img = render(bg, tex, q, hand, rng, gain)
        t0 = time.perf_counter()
        tr.update(img, t, occluders=[] if hand is None else [hand])
        ms.append(1000 * (time.perf_counter() - t0))
        t += 1 / 30
        est = tr.selected.corners
        errs.append(corner_error(est, q))
        if prev is not None and np.abs(frames[i - 1][0] - q).max() < 1e-3:
            still_steps.append(float(np.linalg.norm(est - prev, axis=1).mean()))
        prev = est.copy()
        if save:
            vis = img.copy()
            cv2.polylines(vis, [q.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.polylines(vis, [est.astype(np.int32)], True, (0, 0, 255), 2)
            if writer is None:
                Path(save).mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(Path(save) / f"{backend}_{name}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 30, (W // 2, H // 2))
            writer.write(cv2.resize(vis, (W // 2, H // 2)))
    if writer:
        writer.release()
    e = np.array(errs)
    return {"mean": e.mean(), "max": e.max(), "off": (e > 15).mean(),
            "wobble": float(np.mean(still_steps)) if still_steps else 0.0, "ms": float(np.median(ms))}


def backends():
    out = ["classic"]
    try:
        from doctrack.docaligner import docaligner_available
        if docaligner_available():
            out.append("docaligner")
    except Exception:
        pass
    try:
        import doctrack.flow_track  # noqa: F401   (track/flow branch)
        out.append("flow")
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", action="append", help="lock backend(s) to test (default: all found)")
    ap.add_argument("--scenario", action="append", choices=["still", "slide", "corner", "drag"])
    ap.add_argument("--save", help="folder for videos (green = truth, red = tracked)")
    args = ap.parse_args()
    tex, bg = page_texture(), desk()
    print(f"{'backend':<12}{'scenario':<9}{'mean px':>9}{'max px':>9}{'off %':>8}{'wobble':>9}{'ms':>7}")
    for b in args.backend or backends():
        for s in args.scenario or ["still", "slide", "corner", "drag"]:
            r = run(b, s, tex, bg, args.save)
            if r is None:
                print(f"{b:<12}{s:<9}  could not lock the page")
                continue
            print(f"{b:<12}{s:<9}{r['mean']:9.1f}{r['max']:9.1f}{100 * r['off']:8.0f}{r['wobble']:9.2f}"
                  f"{r['ms']:7.0f}", flush=True)


if __name__ == "__main__":
    main()
