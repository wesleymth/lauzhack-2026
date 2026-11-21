#!/usr/bin/env python3
"""Real-time full-hand tracking (all 21 joints per hand) that saves a JSON every N seconds.

Reads the camera directly with OpenCV and runs MediaPipe Hand Landmarker (Tasks API) in
VIDEO mode (tracks hands between frames), on a downscaled copy of each frame.

IMPORTANT: close the Hand Tracker app first -- only one program can use the camera.

    pip install mediapipe opencv-python numpy

    python hand_realtime.py --list-cameras           # find the camera index
    python hand_realtime.py --show                   # camera 1 (IR), live preview, q to quit
    python hand_realtime.py --camera 1 --save-every 1 --out captures
    python hand_realtime.py --video clip.mp4         # test without a camera

Every --save-every seconds it writes into --out:
    frame_YYYYmmdd_HHMMSS_mmm.json   all joints (x, y normalized, y bottom->top like the tracker)

Images are NOT saved by default. Add:
    --save-images   to also save frame_*.jpg (the frame with the skeleton drawn on it)
    --save-raw      to also save frame_*_raw.jpg (the untouched frame)

TO RUN:
uv run python hand_realtime.py --camera 2 --show --mirror --fourcc MJPG --cam-width 1920 --cam-height 1080
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
# finger -> (mcp, pip, tip). Extended = tip farther from the wrist than pip (works at any rotation).
FINGERS = {"index": (5, 6, 8), "middle": (9, 10, 12), "ring": (13, 14, 16), "pinky": (17, 18, 20)}
COLORS = {"Left": (255, 160, 0), "Right": (0, 200, 255), "Unknown": (200, 200, 200)}


# ----------------------------------------------------------------------------- model / camera
def ensure_model(path: str) -> str:
    if not os.path.exists(path):
        print(f"# downloading model to {path} ...", file=sys.stderr, flush=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class LatestFrame:
    """Reads the source in a background thread so we always process the NEWEST frame (no lag)."""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.video:
            self.cap = cv2.VideoCapture(args.video)
            self.is_file = True
        else:
            self.cap = cv2.VideoCapture(args.camera, pick_backend(args.backend))
            self.is_file = False
            if args.fourcc and args.fourcc.lower() != "none":
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc.upper()))
            if args.cam_width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
            if args.cam_height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(15):  # warm-up: some cameras return black frames at first
                self.cap.read()
                time.sleep(0.05)
        if not self.cap.isOpened():
            raise SystemExit("cannot open the video source (is the Hand Tracker app still running? "
                             "try --list-cameras)")
        print(f"# camera opened at {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}", file=sys.stderr, flush=True)
        self.frame, self.stamp, self.seq = None, 0.0, 0
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                self.stopped = True
                break
            with self.lock:
                self.frame, self.stamp, self.seq = frame, time.time(), self.seq + 1
            if self.is_file:
                time.sleep(1 / 30)

    def read(self, last_seq: int):
        """Return (frame, seq) if there is a frame newer than last_seq, else (None, last_seq)."""
        with self.lock:
            if self.frame is None or self.seq == last_seq:
                return None, last_seq
            return self.frame.copy(), self.seq

    def close(self) -> None:
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()


def pick_backend(name: str) -> int:
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    if name == "any":
        return cv2.CAP_ANY
    return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY  # auto


def list_cameras(backend_name: str, max_index: int = 8) -> None:
    """Open each index, warm up, and report the frame size and brightness (0 = black)."""
    for i in range(max_index):
        cap = cv2.VideoCapture(i, pick_backend(backend_name))
        if cap.isOpened():
            frame = None
            for _ in range(15):
                ok, f = cap.read()
                time.sleep(0.05)
                if ok:
                    frame = f
            if frame is None:
                print(f"camera {i}: opens, but no frame")
            else:
                print(f"camera {i}: opens, {frame.shape[1]}x{frame.shape[0]}, "
                      f"mean brightness {frame.mean():.1f}{'  <- BLACK' if frame.mean() < 3 else ''}")
        cap.release()


# ----------------------------------------------------------------------------- hand logic
def finger_states(lm) -> dict:
    wx, wy = lm[0].x, lm[0].y
    out = {}
    for name, (_mcp, pip, tip) in FINGERS.items():
        d_tip = (lm[tip].x - wx) ** 2 + (lm[tip].y - wy) ** 2
        d_pip = (lm[pip].x - wx) ** 2 + (lm[pip].y - wy) ** 2
        out[name] = bool(d_tip > d_pip * 1.15)
    return out


DIRS = {"down": (0, 1), "up": (0, -1), "left": (-1, 0), "right": (1, 0)}


def _angle(a, b, c) -> float:
    """Angle at b (degrees) between segments b->a and b->c."""
    v1, v2 = np.subtract(a, b), np.subtract(c, b)
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))


def pointing_check(lm, w: int, h: int, args) -> tuple:
    """Return (is_pointing, reason). Everything is computed in pixel space (aspect-correct)."""
    P = [np.array([p.x * w, p.y * h]) for p in lm]
    size = float(np.linalg.norm(P[0] - P[9])) + 1e-9          # wrist -> middle knuckle = hand size
    ext = finger_states(lm)
    if not ext["index"]:
        return False, "index not extended"
    if ext["middle"] or ext["ring"] or ext["pinky"]:
        return False, "other fingers not curled"
    # straight index finger (a pen grip bends the index)
    if min(_angle(P[5], P[6], P[7]), _angle(P[6], P[7], P[8])) < args.straight_deg:
        return False, "index bent"
    # thumb touching the index = pinch / holding a pen = writing
    if float(np.linalg.norm(P[4] - P[8])) < args.pinch_ratio * size:
        return False, "thumb near index (writing/pinch)"
    # direction of the finger in the image
    if args.point_dir != "any":
        v = P[8] - P[5]
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return False, "no direction"
        d = np.array(DIRS[args.point_dir], dtype=float)
        ang = float(np.degrees(np.arccos(np.clip(np.dot(v, d) / n, -1, 1))))
        if ang > args.dir_tolerance:
            return False, f"not pointing {args.point_dir} ({ang:.0f} deg off)"
    return True, "ok"


def hand_to_dict(lm, label: str, w: int, h: int, args) -> dict:
    ext = finger_states(lm)
    pointing, reason = pointing_check(lm, w, h, args)
    return {
        "handedness": label,
        "pointing": pointing,
        "extended": ext,
        "landmarks": {n: {"x": round(p.x, 4), "y": round(1 - p.y, 4), "z": round(p.z, 4)}
                      for n, p in zip(NAMES, lm)},
    }


def draw(frame, hands_lm, hands_info, fps: float):
    h, w = frame.shape[:2]
    for lm, info in zip(hands_lm, hands_info):
        color = COLORS.get(info["handedness"], COLORS["Unknown"])
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            cv2.circle(frame, pt, 5 if i in (4, 8, 12, 16, 20) else 3, (255, 255, 255), -1, cv2.LINE_AA)
        label = info["handedness"] + (" POINTING" if info["pointing"] else "")
        cv2.putText(frame, label, (pts[0][0] - 20, pts[0][1] + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
        if info["pointing"]:
            cv2.circle(frame, pts[8], 14, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.0f} fps  hands: {len(hands_lm)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


# ----------------------------------------------------------------------------- main loop
def run(args: argparse.Namespace) -> None:
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model(args.model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    source = LatestFrame(args)
    os.makedirs(args.out, exist_ok=True)
    print(f"# saving JSON to {os.path.abspath(args.out)} every {args.save_every:g}s "
          f"(images: {'on' if args.save_images or args.save_raw else 'off'}; Ctrl+C to stop)",
          file=sys.stderr, flush=True)

    last_seq, last_save, t0 = 0, 0.0, time.time()
    frames, fps, fps_t = 0, 0.0, time.time()
    last_ts_ms = -1
    streak: dict = {}
    track_f = open(args.track_file, "a", encoding="utf-8") if args.track_file else None
    if track_f:
        print(f"# logging {args.track_hand} index tip to {os.path.abspath(args.track_file)}",
              file=sys.stderr, flush=True)
    try:
        while True:
            frame, last_seq_new = source.read(last_seq)
            if frame is None:
                if source.stopped:
                    break
                time.sleep(0.002)
                continue
            last_seq = last_seq_new
            if args.mirror:
                frame = cv2.flip(frame, 1)

            # detect on a downscaled copy for speed; landmarks are normalized so they map back
            h, w = frame.shape[:2]
            if args.width and w > args.width:
                small = cv2.resize(frame, (args.width, int(h * args.width / w)), interpolation=cv2.INTER_AREA)
            else:
                small = frame
            rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            ts_ms = max(int((time.time() - t0) * 1000), last_ts_ms + 1)  # must strictly increase
            last_ts_ms = ts_ms
            res = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)

            hands_lm = res.hand_landmarks
            swap = {"Left": "Right", "Right": "Left"}
            hands_info = [
                hand_to_dict(lm, hd[0].category_name if args.keep_labels
                             else swap.get(hd[0].category_name, hd[0].category_name), w, h, args)
                for lm, hd in zip(hands_lm, res.handedness)
            ]
            # debounce: must be pointing for N consecutive frames, so a passing pose doesn't count
            seen = set()
            for info in hands_info:
                key = info["handedness"]
                seen.add(key)
                streak[key] = streak.get(key, 0) + 1 if info["pointing"] else 0
                info["pointing"] = streak[key] >= args.hold_frames
            for key in list(streak):
                if key not in seen:
                    streak[key] = 0

            # log the index-tip position of the chosen hand (one line per frame it is visible)
            if track_f:
                for lm, info in zip(hands_lm, hands_info):
                    if info["handedness"] == args.track_hand:
                        tip = lm[8]
                        t = datetime.now()
                        track_f.write(json.dumps({
                            "time": t.isoformat(timespec="milliseconds"),
                            "timestamp_ms": int(t.timestamp() * 1000),
                            "hand": args.track_hand,
                            "x": round(tip.x, 4),            # 0..1, origin top-left, grows right
                            "y": round(tip.y, 4),            # 0..1, origin top-left, grows DOWN
                            "x_px": int(tip.x * w),
                            "y_px": int(tip.y * h),
                            "image": {"width": w, "height": h},
                            "pointing": info["pointing"],
                        }) + "\n")
                        track_f.flush()
                        break

            frames += 1
            now = time.time()
            if now - fps_t >= 1.0:
                fps, frames, fps_t = frames / (now - fps_t), 0, now

            save_due = now - last_save >= args.save_every

            # only draw the skeleton when something needs it: the preview or a saved annotated image
            annotated = None
            if args.show or (save_due and args.save_images):
                annotated = draw(frame.copy(), hands_lm, hands_info, fps)

            if save_due:
                last_save = now
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                base = os.path.join(args.out, f"frame_{stamp}")
                if args.save_images:
                    cv2.imwrite(base + ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if args.save_raw:
                    cv2.imwrite(base + "_raw.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                with open(base + ".json", "w", encoding="utf-8") as f:
                    json.dump({"timestamp_ms": int(now * 1000), "image": {"width": w, "height": h},
                               "hands": hands_info}, f)
                pointing = [i["handedness"] for i in hands_info if i["pointing"]]
                print(f"{datetime.now().strftime('%H:%M:%S')}  {fps:4.1f} fps  hands={len(hands_info)}"
                      f"  pointing={pointing or '-'}  saved {os.path.basename(base)}.json", flush=True)

            if args.show:
                # shrink for display (keeps aspect ratio) so a 1080p frame fits on screen
                vw = args.view_width
                if vw and annotated.shape[1] > vw:
                    view = cv2.resize(annotated, (vw, int(annotated.shape[0] * vw / annotated.shape[1])),
                                      interpolation=cv2.INTER_AREA)
                else:
                    view = annotated
                cv2.imshow("hand_realtime (q to quit)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if track_f:
            track_f.close()
        source.close()
        detector.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time full-hand tracking, saves a JSON every N seconds.")
    p.add_argument("--camera", type=int, default=1, help="camera index (default: 1 = LogiInsight IR camera)")
    p.add_argument("--video", help="use a video file instead of a camera (for testing)")
    p.add_argument("--list-cameras", action="store_true")
    p.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"], default="auto",
                   help="OpenCV capture backend; on Windows try dshow, then msmf (default: auto)")
    p.add_argument("--cam-width", type=int, default=1920, help="request camera width (default: 1920)")
    p.add_argument("--cam-height", type=int, default=1080, help="request camera height (default: 1080)")
    p.add_argument("--fourcc", default="MJPG",
                   help="pixel format to request (MJPG, YUY2, NV12 or none). Default: MJPG, needed for "
                        "1920x1080 on the RGB camera. Use 'none' (with --cam-width 0 --cam-height 0) "
                        "for the IR camera.")
    p.add_argument("--point-dir", choices=["any", "down", "up", "left", "right"], default="down",
                   help="required direction of the index finger in the image (default: down)")
    p.add_argument("--dir-tolerance", type=float, default=60.0,
                   help="allowed angle off --point-dir, in degrees (default: 60)")
    p.add_argument("--straight-deg", type=float, default=150.0,
                   help="min angle at the index joints to count as straight (default: 150)")
    p.add_argument("--pinch-ratio", type=float, default=0.45,
                   help="thumb-index distance below this x hand size = writing/pinch (default: 0.45)")
    p.add_argument("--hold-frames", type=int, default=4,
                   help="consecutive frames needed before POINTING is reported (default: 4)")
    p.add_argument("--track-file", default="right_index.jsonl",
                   help="file that logs the index-tip position of one hand, one JSON object per line "
                        "(default: right_index.jsonl; empty string disables)")
    p.add_argument("--track-hand", choices=["Right", "Left"], default="Right",
                   help="which hand to log (default: Right)")
    p.add_argument("--keep-labels", action="store_true",
                   help="do NOT swap Left/Right (by default they are swapped, because this camera is "
                        "not mirrored and MediaPipe assumes a mirrored selfie image)")
    p.add_argument("--width", type=int, default=640, help="detection width in px (smaller = faster; 0 = full)")
    p.add_argument("--max-hands", type=int, default=2)
    p.add_argument("--mirror", action="store_true", help="flip the image horizontally (selfie view)")
    p.add_argument("--save-every", type=float, default=1.0, help="seconds between saved JSON files (default 1)")
    p.add_argument("--save-images", action="store_true",
                   help="also save the annotated frame as .jpg (default: off, JSON only)")
    p.add_argument("--save-raw", action="store_true",
                   help="also save the untouched frame as _raw.jpg (default: off)")
    p.add_argument("--out", default="captures", help="output folder (default: captures)")
    p.add_argument("--show", action="store_true", help="show a live preview window")
    p.add_argument("--view-width", type=int, default=960,
                   help="preview window width in px, aspect ratio kept (0 = native size; default 960)")
    p.add_argument("--model", default="hand_landmarker.task", help="path to hand_landmarker.task")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        if args.list_cameras:
            list_cameras(args.backend)
        else:
            run(args)
    except KeyboardInterrupt:
        pass