"""Live desk view: detect + track documents, click one to get a flat scan.

    uv run python -m doctrack.app                 # auto-find the LogiInsight RGB camera
    uv run python -m doctrack.app --device 0      # any webcam by index
    uv run python -m doctrack.app --video desk.mp4

Select a document by POINTING at it with your index finger for 1 s (hand
tracking, ../Hand_tracker), or with a left click. Right click releases.
While locked, only that document is tracked and it is never dropped.
Scan window: freely resizable, its size IS the warp size. Toolbar buttons (or
keys) to fix the document orientation: Rotate 90 (o), Mirror left-right (m).
Keys: q/Esc quit, r release, d toggle debug masks, s save scan PNG.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .detector import DetectorConfig, OpenCVDocumentDetector
from .tracker import DocumentTracker

CAM_WIN = "doctrack - camera"
SCAN_WIN = "doctrack - scan"
DEBUG_WIN = "doctrack - debug"
CORNER_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # doc TL TR BR BL (BGR)
CORNER_NAMES = ["TL", "TR", "BR", "BL"]
STATE_COLORS = {"tracking": (0, 255, 0), "partial": (0, 220, 255), "searching": (0, 120, 255)}
SDK_DIR = Path(__file__).resolve().parents[2] / "logiinsight_api"


def open_source(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"Cannot open video {args.video}")
        return cap
    try:
        sys.path.insert(0, str(SDK_DIR))
        import camera_utils as cu  # LogiInsight SDK helpers
        if args.device is None:
            idx, backend_name = cu.find_camera("rgb")
            backend = cu.backend_id(backend_name)
        else:
            idx, backend = args.device, None
        cap, info = cu.open_camera(idx, backend=backend, width=args.width, height=args.height)
        print("Camera:", cu.describe(info))
        return cap
    except ImportError:
        cap = cv2.VideoCapture(args.device or 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            sys.exit("Cannot open camera")
        return cap


def draw(frame, tracker, fps):
    out = frame.copy()
    locked = tracker.locked
    for t in tracker.tracks:
        color = STATE_COLORS.get(t.state, (0, 255, 0)) if locked else (255, 200, 0)
        q = t.corners.astype(np.int32)
        cv2.polylines(out, [q], True, color, 3 if locked else 2, cv2.LINE_AA)
        for p, c, name, vis in zip(q, CORNER_COLORS, CORNER_NAMES, t.visible):
            p = tuple(int(v) for v in p)
            # filled = seen this frame; hollow = covered, last known position kept
            cv2.circle(out, p, 8, c, -1 if vis else 2, cv2.LINE_AA)
            cv2.putText(out, name, (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        cx, cy = q.mean(axis=0).astype(int)
        hidden = int((~t.visible).sum())
        label = f"#{t.id}" + (f" {hidden} covered" if hidden else "")
        cv2.putText(out, label, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    if locked:
        w, h = tracker.scan_size
        status = f"{fps:5.1f} fps | LOCKED #{tracker.selected.id} ({w}x{h}) | r / right-click to release"
    else:
        status = f"{fps:5.1f} fps | {len(tracker.tracks)} doc(s) | point at a document (1 s) or click"
    cv2.putText(out, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_hands(out, hands, connections, selector, t):
    for h in hands:
        pts = [tuple(int(v) for v in p) for p in h.landmarks]
        color = (0, 255, 0) if h.pointing else (180, 180, 180)
        for a, b in connections:
            cv2.line(out, pts[a], pts[b], color, 2, cv2.LINE_AA)
        cv2.circle(out, pts[8], 7, color, -1, cv2.LINE_AA)
    if selector is not None and selector.tip is not None:
        # progress ring around the fingertip: full circle = selected
        c = tuple(int(v) for v in selector.tip)
        cv2.circle(out, c, 28, (80, 80, 80), 3, cv2.LINE_AA)
        cv2.ellipse(out, c, (28, 28), -90, 0, 360 * selector.progress(t), (0, 255, 0), 6, cv2.LINE_AA)


TOOLBAR_H = 44
BUTTONS = [("rotate", "Rotate 90", 8, 138), ("mirror", "Mirror", 146, 256)]  # (id, label, x0, x1)


def toolbar(width, tracker):
    bar = np.full((TOOLBAR_H, width, 3), 50, np.uint8)
    enabled = tracker.locked
    for bid, label, x0, x1 in BUTTONS:
        active = bid == "mirror" and tracker.mirrored
        fill = (90, 140, 60) if active else ((95, 95, 95) if enabled else (65, 65, 65))
        cv2.rectangle(bar, (x0, 6), (x1, TOOLBAR_H - 7), fill, -1)
        cv2.rectangle(bar, (x0, 6), (x1, TOOLBAR_H - 7), (150, 150, 150), 1)
        cv2.putText(bar, label, (x0 + 12, TOOLBAR_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255) if enabled else (130, 130, 130), 1, cv2.LINE_AA)
    return bar


def scan_window_size(default):
    """Current content size (w, h) of the scan window; the user can resize it freely."""
    try:
        _, _, w, h = cv2.getWindowImageRect(SCAN_WIN)
    except cv2.error:
        return default
    return (w, h) if w > 50 and h > TOOLBAR_H + 50 else default


def placeholder(w, h, text="point at / click a document"):
    img = np.full((h, w, 3), 40, np.uint8)
    cv2.putText(img, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=None, help="camera index (default: auto-find)")
    ap.add_argument("--video", help="read from a video file instead of a camera")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--paper", choices=["a4", "letter", "auto"], default="a4",
                    help="scan aspect ratio (orientation is detected); auto = closest of a4/letter")
    ap.add_argument("--scan-size", type=int, default=1100, help="long side of the scan in px")
    ap.add_argument("--doc-up", choices=["down", "up", "left", "right"], default="down",
                    help="where the top of the document points in the camera image "
                         "(default down: page faces the user, top towards the keyboard)")
    ap.add_argument("--display-width", type=int, default=1280, help="camera window width")
    ap.add_argument("--scan-display-height", type=int, default=720, help="scan window height")
    ap.add_argument("--no-hands", action="store_true", help="disable hand tracking (click only)")
    ap.add_argument("--dwell", type=float, default=1.0, help="seconds to point at a document to select it")
    ap.add_argument("--any-finger-pose", action="store_true",
                    help="any index fingertip on a document counts, not only the pointing gesture")
    ap.add_argument("--point-dir", choices=["any", "down", "up", "left", "right"], default="down",
                    help="required pointing direction in the image (as in Hand_tracker; default down)")
    args = ap.parse_args(argv)

    cap = open_source(args)
    tracker = DocumentTracker(OpenCVDocumentDetector(DetectorConfig(doc_up=args.doc_up)), paper=args.paper,
                              scan_long_side=args.scan_size)
    state = {"scale": 1.0, "debug": False}
    hand_tracker, selector = None, None
    if not args.no_hands:
        try:
            from .hands import DwellSelector, HandTracker
            hand_tracker = HandTracker(point_dir=args.point_dir)
            selector = DwellSelector(dwell_s=args.dwell, require_pointing=not args.any_finger_pose)
            print(f"hand tracking on: point at a document for {args.dwell:g}s to select it")
        except Exception as e:  # mediapipe / model missing: fall back to clicking
            print(f"hand tracking unavailable ({e}); click to select")

    def on_mouse(event, x, y, _flags, _param):
        fx, fy = x / state["scale"], y / state["scale"]
        if event == cv2.EVENT_LBUTTONDOWN and not tracker.locked:
            t = tracker.select_at(fx, fy)
            if t:
                print(f"locked #{t.id}, scan {tracker.scan_size[0]}x{tracker.scan_size[1]}")
            else:
                print("no document there")
        elif event == cv2.EVENT_RBUTTONDOWN:
            tracker.clear_selection()

    def fit_scan_window():
        """Size the scan window to the paper shape at --scan-display-height."""
        w, h = tracker.scan_size or (int(args.scan_display_height / 2 ** 0.5), args.scan_display_height)
        k = args.scan_display_height / float(h)
        state["scan_win"] = (int(w * k), int(h * k) + TOOLBAR_H)
        cv2.resizeWindow(SCAN_WIN, *state["scan_win"])

    def rotate():
        if tracker.locked:
            w, h = scan_window_size(state["scan_win"])
            tracker.rotate(1)
            state["scan_win"] = (h - TOOLBAR_H, w + TOOLBAR_H)   # content turns 90 deg
            cv2.resizeWindow(SCAN_WIN, *state["scan_win"])

    def mirror():
        if tracker.locked:
            tracker.toggle_mirror()

    def on_scan_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or y >= TOOLBAR_H:
            return
        for bid, _label, x0, x1 in BUTTONS:
            if x0 <= x <= x1:
                rotate() if bid == "rotate" else mirror()

    cv2.namedWindow(CAM_WIN, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(SCAN_WIN, cv2.WINDOW_NORMAL)   # freely resizable: its size is the warp size
    fit_scan_window()
    cv2.setMouseCallback(CAM_WIN, on_mouse)
    cv2.setMouseCallback(SCAN_WIN, on_scan_mouse)
    was_locked = False

    fps, last = 0.0, time.perf_counter()
    scan = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if args.video:
                    break
                continue
            now = time.perf_counter()
            hands = []
            if hand_tracker is not None and not tracker.locked:   # only needed to select
                hands = hand_tracker.detect(frame, now)
            tracker.update(frame, now, occluders=[h.hull for h in hands])
            if selector is not None:
                tid = selector.update(tracker, hands, now)
                if tid is not None:
                    print(f"locked #{tid} by pointing, scan {tracker.scan_size[0]}x{tracker.scan_size[1]}")
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now

            vis = draw(frame, tracker, fps)
            if hands:
                draw_hands(vis, hands, hand_tracker.connections, selector, now)
            state["scale"] = min(1.0, args.display_width / float(vis.shape[1]))
            if state["scale"] < 1.0:
                vis = cv2.resize(vis, None, fx=state["scale"], fy=state["scale"],
                                 interpolation=cv2.INTER_AREA)
            cv2.imshow(CAM_WIN, vis)

            if tracker.locked and not was_locked:
                fit_scan_window()                  # new document: start at its paper shape
            was_locked = tracker.locked
            ww, wh = scan_window_size(state["scan_win"])
            state["scan_win"] = (ww, wh)
            scan = tracker.scan(frame, size=(ww, wh - TOOLBAR_H))
            body = scan if scan is not None else placeholder(ww, wh - TOOLBAR_H)
            cv2.imshow(SCAN_WIN, np.vstack([toolbar(ww, tracker), body]))

            if state["debug"] and tracker.detector.debug:
                cv2.imshow(DEBUG_WIN, np.hstack(list(tracker.detector.debug.values())))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                tracker.clear_selection()
            elif key == ord("o"):
                rotate()
            elif key == ord("m"):
                mirror()
            elif key == ord("d"):
                state["debug"] = not state["debug"]
                if not state["debug"]:
                    cv2.destroyWindow(DEBUG_WIN)
            elif key == ord("s") and scan is not None:
                name = f"scan_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(name, scan)
                print(f"saved {name} ({scan.shape[1]}x{scan.shape[0]})")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if hand_tracker is not None:
            hand_tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
