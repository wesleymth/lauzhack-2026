"""Live desk view: detect + track documents, click one to get a flat scan.

    uv run python -m doctrack.app                 # auto-find the LogiInsight RGB camera
    uv run python -m doctrack.app --device 0      # any webcam by index
    uv run python -m doctrack.app --video desk.mp4

Select a document by POINTING at it with your index finger for 1 s (hand
tracking, ../Hand_tracker), or with a left click. Right click or r releases.
While locked, only that document is tracked and it is never dropped.
Scan window: freely resizable; the warp is recomputed at the window's size, so
the scan always fills it. Buttons (or keys): Rotate 90 (o), Mirror (m).
Keys (any window): Esc/q quit, r release, o rotate, m mirror, s save scan PNG, d debug.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .detector import DetectorConfig, OpenCVDocumentDetector
from .ui import UI
from .tracker import DocumentTracker

CORNER_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # doc TL TR BR BL (BGR)
CORNER_NAMES = ["TL", "TR", "BR", "BL"]
STATE_COLORS = {"tracking": (0, 255, 0), "partial": (0, 220, 255), "covered": (0, 220, 255), "searching": (0, 120, 255)}
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

    def click(x, y):
        if tracker.locked:
            return
        t = tracker.select_at(x / state["scale"], y / state["scale"])
        if t:
            print(f"locked #{t.id}, scan {tracker.scan_size[0]}x{tracker.scan_size[1]}")
        else:
            print("no document there")

    def release():
        if tracker.locked:
            tracker.clear_selection()
            print("released")

    def fit_scan_window():
        """Start a newly locked document at its paper shape, --scan-display-height tall."""
        w, h = tracker.scan_size or (int(args.scan_display_height / 2 ** 0.5), args.scan_display_height)
        k = args.scan_display_height / float(h)
        ui.scan.set_content_size(int(w * k), int(h * k))

    def rotate():
        if tracker.locked:
            w, h = ui.scan.size()
            tracker.rotate(1)
            ui.scan.set_content_size(h, w)       # content turns 90 deg: swap the window shape

    def mirror():
        if tracker.locked:
            tracker.toggle_mirror()

    def save():
        if state.get("scan") is not None:
            name = f"scan_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(name, state["scan"])
            print(f"saved {name} ({state['scan'].shape[1]}x{state['scan'].shape[0]})")

    def toggle_debug():
        state["debug"] = not state["debug"]
        if not state["debug"]:
            ui.show_debug(None)

    actions = dict(click=click, release=release, rotate=rotate, mirror=mirror, save=save,
                   debug=toggle_debug, quit=lambda: state.update(quit=True))
    ui = UI(actions)
    fit_scan_window()
    was_locked = False

    fps, last = 0.0, time.perf_counter()
    try:
        while not state.get("quit") and not ui.closed:
            ok, frame = cap.read()
            if not ok or frame is None:
                if args.video:
                    break
                ui.update()
                continue
            now = time.perf_counter()
            hands = []
            if hand_tracker is not None:          # select by pointing; "covered" while locked
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
            ui.show_camera(vis)

            if tracker.locked and not was_locked:
                fit_scan_window()                  # new document: start at its paper shape
            was_locked = tracker.locked
            # the warp is recomputed at the window's current size: it always fills the window
            ww, wh = ui.scan.size()
            scan = tracker.scan(frame, size=(ww, wh))
            state["scan"] = scan
            ui.scan.show(scan if scan is not None else placeholder(ww, wh))
            ui.scan.set_state(tracker.locked, tracker.mirrored)

            if state["debug"] and tracker.detector.debug:
                ui.show_debug(np.hstack(list(tracker.detector.debug.values())))
            ui.update()                            # handles clicks, buttons and keys
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if hand_tracker is not None:
            hand_tracker.close()
        ui.close()


if __name__ == "__main__":
    main()
