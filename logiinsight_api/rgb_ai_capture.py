#!/usr/bin/env python3
"""
Copyright (c) 2026 Logitech Europe S.A.
SPDX-License-Identifier: MIT

LogiInsight -- Capture RGB frames + AI data and save annotated PNG/JSON pairs.

Two modes, selected at the command line:

  Default (JSON-only):
      Open the RGB camera with cv2.VideoCapture, fetch the AI JSON via SFTP per
      frame, draw overlays from the JSON, save a PNG+JSON pair.
      Pros: simple, low overhead.
      Caveat: the video frame and the AI JSON travel over two separate paths
      (USB video and SSH/SFTP) and the AI runs a little behind the video, so
      the boxes/landmarks in the JSON can lag the frame you see, most visibly
      when you move. This is expected.

  --device-snapshot:
      Ask the firmware to also dump a YUV file paired with each JSON. The
      host fetches the (BGR image, JSON) pair together, so overlays align
      to the same NPU tick as the source frame. Slow (about 2 s per snapshot
      at 1920x1080) but exact alignment. The picture is the frame the AI
      analysed, which is not necessarily the frame the video showed at that
      moment.

All coordinates in the AI JSON (boxes and landmarks, face mesh included) are
absolute pixels of the RGB stream you opened -- not sensor pixels -- so they
are drawn as-is, with no remapping.

Requirements (Windows, macOS, Linux; no conda):
    pip install -r requirements.txt        # or: uv sync

The SSH user name and password (printed on the card in your kit) are read from
--user / --password or the LOGIWEBCAM_USER / LOGIWEBCAM_PASSWORD environment
variables.

Usage:
    python rgb_ai_capture.py                                  # 10 frames, face detection
    python rgb_ai_capture.py --frames 20 --face-mesh          # + 478-point face mesh
    python rgb_ai_capture.py --format NV12                    # RGB as NV12 (or YUY2); default MJPG
    python rgb_ai_capture.py --device-snapshot --face-mesh    # exactly paired snapshots
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np  # noqa: F401  (kept available for student extensions)

from logiwebcam import (
    AiStalenessMonitor,
    LogiWebcamClient,
    firmware_matches_sdk,
    parse_firmware_label,
)
from logiwebcam import __version__ as LOGIWEBCAM_VERSION

# RGB stream formats the camera offers; the on-camera AI runs on all three
# (firmware 1.5.2 or newer; firmware 1.5.1 runs it on MJPG only).
RGB_FORMATS = ("MJPG", "YUY2", "NV12")

# ── Face mesh landmark regions (MediaPipe 478-point indices) ────────────────
FACE_REGIONS = {
    "jaw":       (list(range(172, 200)) + [0],                   (180, 180, 180)),
    "l_eyebrow": (list(range(70, 80)),                           (180, 130, 255)),
    "r_eyebrow": (list(range(300, 310)),                         (180, 130, 255)),
    "l_eye":     (list(range(130, 160)),                          (255, 180, 100)),
    "r_eye":     (list(range(360, 390)),                          (255, 180, 100)),
    "nose":      (list(range(1, 20)),                             (0, 200, 255)),
    "lips":      (list(range(61, 96)) + list(range(185, 213)),    (80, 80, 255)),
    "l_iris":    (list(range(468, 473)),                          (80, 230, 80)),
    "r_iris":    (list(range(473, 478)),                          (80, 230, 80)),
}

def _lm_xy(lm):
    """Extract (x, y) from a landmark in either [x,y,z] or {"x":..} form."""
    if isinstance(lm, dict):
        return lm["x"], lm["y"]
    return lm[0], lm[1]


# ── Drawing helpers ─────────────────────────────────────────────────────────

def draw_face_bbox(frame, face, frame_w, frame_h):
    bbox = face.get("bbox", [])
    if len(bbox) < 4:
        return
    bx, by, bw, bh = bbox
    x1 = max(0, int(bx))
    y1 = max(0, int(by))
    x2 = min(frame_w - 1, int(bx + bw))
    y2 = min(frame_h - 1, int(by + bh))
    conf = face.get("confidence", 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"face {conf:.2f}", (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def draw_face_landmarks(frame, landmarks, frame_w, frame_h):
    """Draw the face mesh. Landmarks are already absolute pixels of the
    stream (same space as `bbox`), so they are drawn without any mapping."""
    if not landmarks:
        return
    n = len(landmarks)
    for lm in landmarks:
        lx, ly = _lm_xy(lm)
        x, y = int(lx), int(ly)
        if 0 <= x < frame_w and 0 <= y < frame_h:
            cv2.circle(frame, (x, y), 1, (100, 100, 100), -1)
    for region_name, (indices, color) in FACE_REGIONS.items():
        for idx in indices:
            if idx >= n:
                continue
            lx, ly = _lm_xy(landmarks[idx])
            x, y = int(lx), int(ly)
            if 0 <= x < frame_w and 0 <= y < frame_h:
                r = 3 if "iris" in region_name else 2
                cv2.circle(frame, (x, y), r, color, -1)
    cv2.putText(frame, f"Face Mesh: {n} pts", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_overlays(frame, ai_data, frame_w, frame_h, want_face_mesh):
    """Apply all enabled overlays on `frame` in-place.

    The firmware draws nothing on the video, so every box and landmark you
    see comes from the JSON and is drawn here. Both modes (live frame and
    --device-snapshot) therefore look the same.
    """
    if not ai_data:
        return
    for face in ai_data.get("faces", []):
        draw_face_bbox(frame, face, frame_w, frame_h)
        if want_face_mesh:
            draw_face_landmarks(frame, face.get("landmarks", []), frame_w, frame_h)


# ── UVC helpers ─────────────────────────────────────────────────────────────

def uvc_backends():
    """Capture backends to try, best-first, for the current platform.

    OpenCV's default backend selection is unreliable for MJPG + explicit
    resolution, so we name one. DSHOW is listed before MSMF on Windows
    because MSMF often ignores CAP_PROP_FOURCC and silently falls back to
    an uncompressed format, which caps 1080p well below 30fps.
    """
    if sys.platform.startswith("linux"):
        return [("V4L2", cv2.CAP_V4L2)]
    if sys.platform == "win32":
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
    if sys.platform == "darwin":
        return [("AVFoundation", cv2.CAP_AVFOUNDATION)]
    return [("default", cv2.CAP_ANY)]


def _decode_fourcc(v):
    v = int(v)
    if v <= 0:
        return f"raw={v}"
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4))


def device_label(idx):
    """Human-readable name for a camera index on this platform."""
    return f"/dev/video{idx}" if sys.platform.startswith("linux") else f"camera index {idx}"


def parse_device_arg(dev_arg):
    """Accept a Linux device path (/dev/video2) or a bare index (2)."""
    text = dev_arg.strip()
    if text.startswith("/dev/video"):
        text = text[len("/dev/video"):]
    try:
        return int(text)
    except ValueError:
        print(f"  ERROR: Cannot parse --device {dev_arg!r}. "
              f"Use a camera index (e.g. 0) or /dev/video0 on Linux.")
        sys.exit(1)


def _try_open(idx, backend):
    """Open one (index, backend) pair, or return None. Never leaks a capture object."""
    cap = cv2.VideoCapture(idx, backend)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def find_uvc_device():
    """Auto-detect the first working UVC device. Returns (index, backend).

    NOTE: this is the first camera that delivers a frame. On a laptop that is
    usually the built-in webcam -- pass --device to pick the LogiInsight RGB
    camera. (camera_utils.find_camera() can pick by name on Linux.)
    """
    for name, backend in uvc_backends():
        for idx in range(10):
            # On Linux the /dev node check is a cheap filter that avoids
            # 10 open() attempts. Other platforms have no such path, so
            # probing the index is the only option.
            if sys.platform.startswith("linux") and not os.path.exists(f"/dev/video{idx}"):
                continue
            cap = _try_open(idx, backend)
            if cap is None:
                continue
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print(f"  Found UVC device: {device_label(idx)} ({name})")
                return idx, backend
    return -1, None


def open_uvc(width, height, dev_arg, fourcc="MJPG"):
    if dev_arg:
        idx = parse_device_arg(dev_arg)
        cap = backend = None
        for name, candidate in uvc_backends():
            cap = _try_open(idx, candidate)
            if cap is not None:
                backend = name
                break
        if cap is None:
            print(f"  ERROR: Cannot open {device_label(idx)}")
            sys.exit(1)
    else:
        idx, chosen = find_uvc_device()
        if idx < 0:
            print("  ERROR: No UVC device found. Check USB connection and OS camera permissions.")
            sys.exit(1)
        cap = _try_open(idx, chosen)
        if cap is None:
            print(f"  ERROR: Cannot re-open {device_label(idx)}")
            sys.exit(1)
        backend = next(n for n, b in uvc_backends() if b == chosen)

    # Resolution MUST be set before FOURCC -- at least on DSHOW, setting
    # FOURCC first silently negotiates YUY2 instead of the format you asked
    # for (verified empirically).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fourcc = _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Camera: {device_label(idx)} [{backend}]  "
          f"{actual_w}x{actual_h} @ {fps:.0f}fps ({actual_fourcc})")
    if (actual_w, actual_h) != (width, height):
        print(f"  NOTE: requested {width}x{height}, backend gave "
              f"{actual_w}x{actual_h}. Overlay scaling uses the actual size.")
    if actual_fourcc != fourcc:
        print(f"  NOTE: requested {fourcc}, the OS delivered {actual_fourcc!r}. "
              f"The AI works with MJPG, YUY2 and NV12 on firmware 1.5.2 or newer "
              f"(older firmware: MJPG only). The format decides the frame rate and "
              f"whether the IR stream can run next to it. (macOS AVFoundation and "
              f"Windows MSMF may choose the format themselves; on Windows you can "
              f"try another backend / camera index.)")
    return cap, actual_w, actual_h


# ── Per-frame summary printer ───────────────────────────────────────────────

def _print_frame_summary(idx, total, base, ai_data, extra=""):
    parts = [f"[{idx + 1:3d}/{total}]  {base}.png"]
    if ai_data:
        ts = ai_data.get("timestamp_ms", "N/A")
        parts.append(f"ts={ts}")
        faces = ai_data.get("faces", [])
        if faces:
            bbox = faces[0].get("bbox", [])
            if bbox:
                parts.append(f"face_bbox={bbox}")
            n_lm = len(faces[0].get("landmarks", []))
            if n_lm:
                parts.append(f"face_lm={n_lm}")
    if extra:
        parts.append(extra)
    print("  " + "  ".join(parts))


# ── Mode 1 -- JSON-only (default) ───────────────────────────────────────────

def run_naive(args, cam):
    print("[2/3] Opening UVC camera ...")
    cap, w, h = open_uvc(args.width, args.height, args.device, args.format)

    # Warm up so the camera is delivering frames before we start saving.
    for _ in range(5):
        cap.read()

    monitor = AiStalenessMonitor()
    print(f"\n[3/3] Capturing {args.frames} frames (json-only) ...\n"
          f"      Press Ctrl+C to stop early.\n")
    try:
        for i in range(args.frames):
            t0 = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"  [{i + 1:3d}/{args.frames}] FAILED to read frame")
                time.sleep(args.interval)
                continue

            ai_data = cam.get_latest_ai_data()
            warning = monitor.update(ai_data)
            if warning:
                print(f"  WARNING: {warning}")
            annotated = frame.copy()
            draw_overlays(annotated, ai_data, w, h, args.face_mesh)

            base = f"frame_{i:04d}"
            cv2.imwrite(os.path.join(args.output, f"{base}.png"), annotated)
            with open(os.path.join(args.output, f"{base}.json"), "w") as f:
                json.dump({
                    "frame_index": i,
                    "host_timestamp": time.time(),
                    "host_time_iso": datetime.now().isoformat(),
                    "image_file": f"{base}.png",
                    "image_resolution": [w, h],
                    "device_ai_data": ai_data,
                    "capture_latency_ms": round((time.time() - t0) * 1000, 1),
                }, f, indent=2)

            _print_frame_summary(
                i, args.frames, base, ai_data,
                extra=f"latency={round((time.time() - t0) * 1000, 1)}ms",
            )

            wait = max(0, args.interval - (time.time() - t0))
            if wait > 0 and i < args.frames - 1:
                time.sleep(wait)
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ── Mode 2 -- device-side atomic snapshot ───────────────────────────────────

def run_device_snapshot(args, cam):
    """The firmware's NPU pipeline only streams while a UVC consumer is
    active, so we still open cv2.VideoCapture. The cv2 frames themselves
    are discarded; the BGR image we save is the one the firmware dumped
    together with the JSON. Run cv2.read() once per iteration to keep
    the device-side pipeline awake (and naturally pace the loop)."""
    logging.basicConfig(level=logging.WARNING, format="    [diag] %(message)s")

    print("[2/3] Opening UVC camera (keeps the device pipeline awake) ...")
    cap, _, _ = open_uvc(args.width, args.height, args.device, args.format)

    print("  Warming up pipeline (10 cv2.read() calls) ...")
    for _ in range(10):
        cap.read()

    print(f"\n[3/3] Pulling {args.frames} device-side YUV+JSON snapshots ...\n"
          f"      Press Ctrl+C to stop early.\n")
    diagnosed = False
    try:
        for i in range(args.frames):
            cap.read()  # keep-alive + paces the loop to the camera's frame rate.

            t0 = time.time()
            bgr, ai_data = cam.get_device_snapshot(verbose=True)
            elapsed_ms = round((time.time() - t0) * 1000, 1)
            base = f"frame_{i:04d}"

            if bgr is None or ai_data is None:
                status = cam.get_yuv_status()
                print(f"  [{i + 1:3d}/{args.frames}] {base}: snapshot unavailable "
                      f"elapsed={elapsed_ms}ms")
                if status:
                    print(f"      device-status: {status}")
                if not diagnosed:
                    diagnosed = True
                    cam.diagnose()
                if i < args.frames - 1:
                    time.sleep(args.interval)
                continue

            h, w = bgr.shape[:2]
            annotated = bgr.copy()
            draw_overlays(annotated, ai_data, w, h, args.face_mesh)

            cv2.imwrite(os.path.join(args.output, f"{base}.png"), annotated)
            with open(os.path.join(args.output, f"{base}.json"), "w") as f:
                json.dump({
                    "frame_index": i,
                    "host_timestamp": time.time(),
                    "host_time_iso": datetime.now().isoformat(),
                    "image_file": f"{base}.png",
                    "image_resolution": [w, h],
                    "device_ai_data": ai_data,
                    "fetch_latency_ms": elapsed_ms,
                }, f, indent=2)

            _print_frame_summary(
                i, args.frames, base, ai_data,
                extra=f"size={w}x{h}  fetch={elapsed_ms}ms",
            )

            wait = max(0, args.interval - (time.time() - t0))
            if wait > 0 and i < args.frames - 1:
                time.sleep(wait)
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ── Main ────────────────────────────────────────────────────────────────────

def add_connection_args(parser):
    """--ip / --user / --password, shared with rgb_ai_live.py."""
    parser.add_argument("--ip", type=str, default="192.168.1.10",
                        help="Device IP (default: 192.168.1.10)")
    parser.add_argument("--user", type=str, default=None,
                        help="SSH/SFTP user name from the card in your kit. Default: "
                             "the LOGIWEBCAM_USER environment variable.")
    parser.add_argument("--password", type=str, default=None,
                        help="SSH/SFTP password from the card in your kit. Default: "
                             "the LOGIWEBCAM_PASSWORD environment variable.")


def add_format_arg(parser):
    """--format MJPG|YUY2|NV12, shared with rgb_ai_live.py."""
    parser.add_argument("--format", type=str.upper, choices=RGB_FORMATS, default="MJPG",
                        help="RGB stream format to request (default: MJPG). The AI works "
                             "with all three on firmware 1.5.2 or newer. Use NV12 or YUY2 "
                             "when you also need the IR stream at full rate.")


def connect_client(args):
    """Create and connect the SSH client, or exit with a readable error."""
    try:
        cam = LogiWebcamClient(ip=args.ip, username=args.user, password=args.password)
        cam.connect()
        print(f"  Connected to {args.ip}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Check: the camera's USB network adapter has an address on 192.168.1.x "
              "(see README), and LOGIWEBCAM_USER / LOGIWEBCAM_PASSWORD are set. "
              "Run check_connection.py to find the problem step by step.")
        sys.exit(1)

    fw = cam.get_firmware_version()
    if fw:
        print(f"  Firmware: {fw}")
        if firmware_matches_sdk(fw) is False:
            print(f"  WARNING: this SDK is v{LOGIWEBCAM_VERSION} but the camera firmware "
                  f"reports a different release. Use the matching SDK download.")
        label = parse_firmware_label(fw)
        if label is not None and label < (1, 5, 2):
            print(f"  NOTE: firmware {'.'.join(map(str, label))} runs the AI on MJPG "
                  f"streams only. Use --format MJPG (the default); YUY2 and NV12 need "
                  f"firmware 1.5.2 or newer.")
    return cam


def main():
    parser = argparse.ArgumentParser(
        description="LogiInsight -- capture RGB frames + AI data")
    parser.add_argument("--device", type=str, default=None,
                        help="Camera index (e.g. 0), or /dev/video0 on Linux. "
                             "Auto-detect (first working camera) if omitted.")
    parser.add_argument("--frames", type=int, default=10,
                        help="Number of frames to capture (default: 10)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between captures (default: 1.0)")
    parser.add_argument("--output", type=str, default="captures",
                        help="Output directory (default: ./captures)")
    parser.add_argument("--width",  type=int, default=1920,
                        help="Frame width  (default: 1920)")
    parser.add_argument("--height", type=int, default=1080,
                        help="Frame height (default: 1080)")
    parser.add_argument("--face-mesh", action="store_true",
                        help="Enable 478-point face mesh landmarks")
    add_format_arg(parser)
    add_connection_args(parser)
    parser.add_argument("--device-snapshot", action="store_true",
                        help="Use the device-side YUV+JSON path. "
                             "Saved PNGs are decoded from the firmware's YUV "
                             "dump, so overlays align to the same NPU tick.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"LogiInsight Python API v{LOGIWEBCAM_VERSION}")
    print("Copyright (c) 2026 Logitech Europe S.A.\n")

    print("[1/3] Connecting to LogiInsight AI ...")
    cam = connect_client(args)

    cam.set_ai_config(
        enable_face_detection=True,
        enable_face_mesh=args.face_mesh,
        enable_yuv_snapshot=args.device_snapshot,
    )
    print(f"  AI config: face_mesh={args.face_mesh}, yuv_snapshot={args.device_snapshot}")
    time.sleep(1.0)  # firmware re-reads the INI about every 0.5 s while streaming

    try:
        if args.device_snapshot:
            run_device_snapshot(args, cam)
        else:
            run_naive(args, cam)
    finally:
        # Always disable the YUV dump on exit so the device isn't writing
        # a large file per frame to its small config partition forever.
        if args.device_snapshot:
            cam.set_ai_config(
                enable_face_detection=True,
                enable_face_mesh=args.face_mesh,
                enable_yuv_snapshot=False,
            )
        cam.disconnect()

    print(f"\nDone! Output in {os.path.abspath(args.output)}/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C) -- shutting down cleanly.")
        sys.exit(130)
