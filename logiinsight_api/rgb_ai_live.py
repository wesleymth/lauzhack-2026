#!/usr/bin/env python3
"""
Copyright (c) 2026 Logitech Europe S.A.
SPDX-License-Identifier: MIT

LogiInsight -- live preview window with real-time AI overlays.

Unlike rgb_ai_capture.py (which saves PNG/JSON pairs to disk), this opens a
cv2.imshow() window and continuously draws the latest face AI data
on top of the live RGB stream. Press 'q' or ESC (or Ctrl+C) to quit.

The boxes and landmarks come from the AI JSON, which travels separately from
the video and runs slightly behind it, so they can lag the picture when you
move. --device-snapshot shows exactly paired frames instead, at a much lower
frame rate.

Requirements (Windows, macOS, Linux; no conda):
    pip install -r requirements.txt        # or: uv sync
    (needs the GUI build of OpenCV, i.e. NOT opencv-python-headless)

If --device is not given, the connected cameras are listed and you are
prompted to pick one. The SSH user name and password (printed on the card in
your kit) are read from --user / --password or the LOGIWEBCAM_USER /
LOGIWEBCAM_PASSWORD environment variables.

Usage:
    python rgb_ai_live.py                       # prompts for camera, face detection only
    python rgb_ai_live.py --face-mesh           # + 478-point face mesh
    python rgb_ai_live.py --device-snapshot     # exactly paired snapshots (slow)
    python rgb_ai_live.py --device 0            # skip the prompt, use camera 0
    python rgb_ai_live.py --no-overlay          # raw picture, nothing drawn on it
    python rgb_ai_live.py --format NV12         # RGB as NV12 (or YUY2); default MJPG
"""

import argparse
import os
import sys
import time

import cv2

from logiwebcam import AiStalenessMonitor
from logiwebcam import __version__ as LOGIWEBCAM_VERSION
from rgb_ai_capture import (
    _try_open,
    add_connection_args,
    add_format_arg,
    connect_client,
    device_label,
    draw_overlays,
    open_uvc,
    uvc_backends,
)

WINDOW_NAME = "LogiInsight -- live preview (q/ESC to quit)"


def list_uvc_devices(max_index=10):
    """Scan for all working UVC devices (index, backend_name, width, height).

    Unlike rgb_ai_capture.py's find_uvc_device() (which stops at the first hit),
    this enumerates every candidate on the first backend that yields any
    working device, so the user has a real list to choose from.
    """
    devices = []
    for name, backend in uvc_backends():
        for idx in range(max_index):
            if sys.platform.startswith("linux") and not os.path.exists(f"/dev/video{idx}"):
                continue
            cap = _try_open(idx, backend)
            if cap is None:
                continue
            ret, frame = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if ret and frame is not None:
                devices.append((idx, name, w, h))
        if devices:
            break  # this backend works on this platform; no need to try the rest
    return devices


def select_camera_interactive():
    """Print all detected cameras and prompt the user to pick one.

    Returns a --device-style string (camera index). Falls back to the
    first detected camera if stdin isn't interactive (e.g. run from a
    script/CI), and exits if no camera is found at all.
    """
    print("Scanning for cameras ...")
    devices = list_uvc_devices()
    if not devices:
        print("  ERROR: No UVC camera found. Check USB connection and OS camera permissions.")
        sys.exit(1)

    print("\nAvailable cameras (the LogiInsight RGB stream is landscape; the near-infrared "
          "'CPD' stream is portrait):")
    for i, (idx, name, w, h) in enumerate(devices):
        print(f"  [{i}] {device_label(idx)}  ({name}, {w}x{h})")

    if len(devices) == 1:
        idx = devices[0][0]
        print(f"\nOnly one camera found, using {device_label(idx)}.\n")
        return str(idx)

    while True:
        try:
            choice = input(f"\nSelect a camera [0-{len(devices) - 1}]: ").strip()
        except EOFError:
            idx = devices[0][0]
            print(f"\nNo interactive input available, defaulting to {device_label(idx)}.\n")
            return str(idx)
        if not choice.isdigit() or not (0 <= int(choice) < len(devices)):
            print(f"  Please enter a number between 0 and {len(devices) - 1}.")
            continue
        idx = devices[int(choice)][0]
        print(f"Using {device_label(idx)}.\n")
        return str(idx)


def _draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def run_live_naive(args, cam):
    """JSON-only: cv2.VideoCapture frame + latest AI JSON, drawn live."""
    print("[2/2] Opening UVC camera ...")
    cap, w, h = open_uvc(args.width, args.height, args.device, args.format)

    for _ in range(5):
        cap.read()

    print("\nStreaming ... press 'q' or ESC in the window to quit.\n")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    t_prev = time.time()
    fps = 0.0
    monitor = AiStalenessMonitor()
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            ai_data = cam.get_latest_ai_data()
            warning = monitor.update(ai_data)
            if warning:
                print(f"WARNING: {warning}")
            annotated = frame.copy()
            if args.no_overlay:
                # Raw camera picture; only a counter shows that the AI is running.
                n_faces = len(ai_data.get("faces", [])) if ai_data else 0
                cv2.putText(annotated, f"AI faces: {n_faces}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                draw_overlays(annotated, ai_data, w, h, args.face_mesh)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            _draw_fps(annotated, fps)

            cv2.imshow(WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_live_device_snapshot(args, cam):
    """Device-side YUV+JSON pair, drawn live (slower, pixel-perfect)."""
    print("[2/2] Opening UVC camera (keeps the device pipeline awake) ...")
    cap, _, _ = open_uvc(args.width, args.height, args.device, args.format)

    for _ in range(10):
        cap.read()

    print("\nStreaming (device-snapshot) ... press 'q' or ESC in the window to quit.\n")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    t_prev = time.time()
    fps = 0.0
    try:
        while True:
            cap.read()  # keep-alive so the device NPU pipeline keeps running

            bgr, ai_data = cam.get_device_snapshot(retries=1)
            if bgr is None or ai_data is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            h, w = bgr.shape[:2]
            annotated = bgr.copy()
            draw_overlays(annotated, ai_data, w, h, args.face_mesh)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            _draw_fps(annotated, fps)

            cv2.imshow(WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="LogiInsight -- live preview window with real-time AI overlays")
    parser.add_argument("--device", type=str, default=None,
                        help="Camera index (e.g. 0), or /dev/video0 on Linux. "
                             "If omitted, you'll be prompted to pick from a list.")
    parser.add_argument("--width",  type=int, default=1920,
                        help="Frame width  (default: 1920)")
    parser.add_argument("--height", type=int, default=1080,
                        help="Frame height (default: 1080)")
    parser.add_argument("--face-mesh", action="store_true",
                        help="Enable 478-point face mesh landmarks")
    add_format_arg(parser)
    add_connection_args(parser)
    parser.add_argument("--no-overlay", action="store_true",
                        help="Show the raw camera picture without drawing boxes or "
                             "landmarks (only an 'AI faces: N' counter). Handy to see "
                             "what the camera really sends. Ignored with --device-snapshot.")
    parser.add_argument("--device-snapshot", action="store_true",
                        help="Use the device-side YUV+JSON path instead "
                             "of the default UVC+JSON path.")
    args = parser.parse_args()

    print(f"LogiInsight Python API v{LOGIWEBCAM_VERSION}")
    print("Copyright (c) 2026 Logitech Europe S.A.\n")

    if args.device is None:
        args.device = select_camera_interactive()

    print("[1/2] Connecting to LogiInsight AI ...")
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
            run_live_device_snapshot(args, cam)
        else:
            run_live_naive(args, cam)
    finally:
        if args.device_snapshot:
            cam.set_ai_config(
                enable_face_detection=True,
                enable_face_mesh=args.face_mesh,
                enable_yuv_snapshot=False,
            )
        cam.disconnect()

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C) -- shutting down cleanly.")
        sys.exit(130)
