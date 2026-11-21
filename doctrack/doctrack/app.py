"""Live desk view: detect + track documents, click one to get a flat scan.

    uv run python -m doctrack.app                 # auto-find the LogiInsight RGB camera
    uv run python -m doctrack.app --device 0      # any webcam by index
    uv run python -m doctrack.app --video desk.mp4

Mouse (camera window): left click = lock onto a document, right click = release.
While locked, only that document is tracked and it is never dropped.
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
        status = f"{fps:5.1f} fps | {len(tracker.tracks)} doc(s) | click a document to lock"
    cv2.putText(out, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def fit_height(img, h):
    s = h / float(img.shape[0])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def placeholder(h, text="click a document"):
    img = np.full((h, int(h / 2 ** 0.5), 3), 40, np.uint8)
    cv2.putText(img, text, (30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
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
    args = ap.parse_args(argv)

    cap = open_source(args)
    tracker = DocumentTracker(OpenCVDocumentDetector(DetectorConfig(doc_up=args.doc_up)), paper=args.paper,
                              scan_long_side=args.scan_size)
    state = {"scale": 1.0, "debug": False}

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

    cv2.namedWindow(CAM_WIN, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(SCAN_WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(CAM_WIN, on_mouse)

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
            tracker.update(frame, now)
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now

            vis = draw(frame, tracker, fps)
            state["scale"] = min(1.0, args.display_width / float(vis.shape[1]))
            if state["scale"] < 1.0:
                vis = cv2.resize(vis, None, fx=state["scale"], fy=state["scale"],
                                 interpolation=cv2.INTER_AREA)
            cv2.imshow(CAM_WIN, vis)

            scan = tracker.scan(frame)
            cv2.imshow(SCAN_WIN, fit_height(scan, args.scan_display_height) if scan is not None
                       else placeholder(args.scan_display_height))

            if state["debug"] and tracker.detector.debug:
                cv2.imshow(DEBUG_WIN, np.hstack(list(tracker.detector.debug.values())))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                tracker.clear_selection()
            elif key == ord("d"):
                state["debug"] = not state["debug"]
                if not state["debug"]:
                    cv2.destroyWindow(DEBUG_WIN)
            elif key == ord("s") and scan is not None:
                name = f"scan_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(name, scan)
                print("saved", name)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
