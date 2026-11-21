#!/usr/bin/env python3
"""
Copyright (c) 2026 Logitech Europe S.A.
SPDX-License-Identifier: MIT

LogiInsight -- live near-infrared (IR) stream with lightweight processing.

Opens the camera's "CPD" stream: the monochrome near-infrared sensor, delivered
as MJPG video that every OS and OpenCV backend can read. Frames are PORTRAIT
(256x320, 512x640 or 1024x1280). The sensor is not colour and responds to a wide
range of near-infrared wavelengths (not one narrow band), and the camera's own
NIR illuminator lights the scene, so the image looks different from the RGB one:
skin is bright, many dark fabrics are bright, screens and sunlight behave
unexpectedly.

No SSH / network setup is needed for this script: it only reads the video.

Processing modes (cycle with the 'm' key, or pass --mode):
    raw        the decoded grey image
    stretch    contrast stretch between the 1st and 99th percentile
    clahe      adaptive histogram equalisation (local contrast)
    threshold  fixed threshold ('+' / '-' change it)
    otsu       automatic (Otsu) threshold

Keys (click the window first): m mode, + / - threshold, s save PNG, q or ESC quit.
Ctrl+C in the terminal also quits; the camera is always released.

Usage:
    python example_IR.py                       # auto-detect the IR camera, 512x640
    python example_IR.py --width 1024 --height 1280
    python example_IR.py --mode clahe
    python example_IR.py --device 2            # choose the camera index yourself
    python example_IR.py --frames 300 --no-display   # headless smoke test

Notes:
  * The IR stream can run next to the RGB stream. If the RGB stream is MJPG
    the IR stream drops to about 1 fps (the two share the camera's JPEG encoder);
    with RGB as NV12 or YUY2 both run at about 30 fps and the AI keeps running.
    Open the RGB stream first; if a stream does not start or the frame rate
    drops, close both and open them in the other order.
  * The firmware's AI JSON describes the RGB stream only; there are no AI
    results in IR pixel coordinates.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from camera_utils import (
    CameraNotFoundError,
    backend_id,
    camera_hint,
    describe,
    find_camera,
    install_stop_flag,
    is_grayscale,
    open_camera,
    parse_device_arg,
)

MODES = ["raw", "stretch", "clahe", "threshold", "otsu"]
CPD_SIZES = [(256, 320), (512, 640), (1024, 1280)]
MAX_CONSECUTIVE_FAILURES = 90   # about 3 s at 30 fps


# --------------------------------------------------------------------------- #
# Processing (pure functions on uint8 images, easy to reuse / test)
# --------------------------------------------------------------------------- #

def to_gray(frame: np.ndarray) -> np.ndarray:
    """Single-channel uint8 image from whatever the backend returned."""
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def contrast_stretch(gray: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(gray, (low_pct, high_pct))
    if hi - lo < 1:
        return gray.copy()
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_enhance(gray: np.ndarray, clip: float = 3.0, tiles: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(gray)


def fixed_threshold(gray: np.ndarray, thresh: int) -> np.ndarray:
    return cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)[1]


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def process(gray: np.ndarray, mode: str, thresh: int = 128) -> np.ndarray:
    """Apply one of MODES to a grey image and return a grey image."""
    if mode == "raw":
        return gray
    if mode == "stretch":
        return contrast_stretch(gray)
    if mode == "clahe":
        return clahe_enhance(gray)
    if mode == "threshold":
        return fixed_threshold(gray, thresh)
    if mode == "otsu":
        return otsu_threshold(gray)
    raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")


# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #

def compose_view(raw_gray, processed_gray, mode, thresh, fps, max_height=800):
    """Raw and processed side by side, with a status line, scaled to fit the screen."""
    left = cv2.cvtColor(raw_gray, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)
    view = np.hstack([left, right])
    if view.shape[0] > max_height:
        s = max_height / view.shape[0]
        view = cv2.resize(view, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    label = f"{mode}" + (f" t={thresh}" if mode == "threshold" else "") + f"  {fps:.1f} fps"
    cv2.putText(view, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(view, "raw", (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
    cv2.putText(view, "processed", (view.shape[1] // 2 + 10, view.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return view


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run(args) -> int:
    stop = install_stop_flag()

    # ---- find / open the IR camera ----------------------------------------
    try:
        if args.device is not None:
            index, backend = parse_device_arg(args.device), None
        else:
            print("Looking for the near-infrared (CPD) camera ...")
            index, backend_name = find_camera("ir")
            backend = backend_id(backend_name)
        cap, info = open_camera(index, backend, args.width, args.height, "MJPG", args.fps)
    except CameraNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"Camera: {describe(info)}")
    if (info["width"], info["height"]) != (args.width, args.height):
        print(f"  NOTE: requested {args.width}x{args.height}, the OS gave "
              f"{info['width']}x{info['height']}.")
    print("  Note: if another program has the RGB stream open as MJPG, the IR frame rate "
          "drops to about 1 fps; with RGB as NV12 or YUY2 both run at about 30 fps.")

    out_dir = args.output
    mode = args.mode
    thresh = args.threshold
    prev_gray = None
    warned_colour = warned_landscape = False
    frames = failures = saved = 0
    fps = 0.0
    t_prev = time.time()
    t_start = t_prev

    if not args.no_display:
        cv2.namedWindow("LogiInsight IR", cv2.WINDOW_NORMAL)
        print("Streaming. Keys: m=mode  +/-=threshold  s=save  q/ESC=quit  (Ctrl+C also quits)")

    try:
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    print("ERROR: the camera stopped delivering frames (unplugged, or opened "
                          "by another program?).\n" + camera_hint(), file=sys.stderr)
                    return 3
                time.sleep(0.03)
                continue
            failures = 0
            frames += 1

            if not warned_colour and not is_grayscale(frame):
                warned_colour = True
                print("  WARNING: this image has colour, so it is probably not the IR stream. "
                      "Pass --device <index> (try the other indexes).")
            if not warned_landscape and frame.shape[1] > frame.shape[0]:
                warned_landscape = True
                print("  WARNING: frame is landscape; the IR (CPD) stream is portrait. "
                      "Probably the wrong camera -- use --device.")

            gray = to_gray(frame)
            if args.flicker_filter:
                # If lit and unlit frames alternate, the per-pixel max of two
                # consecutive frames is always the lit one.
                shown_src = gray if prev_gray is None or prev_gray.shape != gray.shape \
                    else np.maximum(gray, prev_gray)
                prev_gray = gray
            else:
                shown_src = gray

            result = process(shown_src, mode, thresh)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6)) if frames > 1 else 0.0
            t_prev = now

            if args.no_display:
                if frames % 30 == 0:
                    print(f"  {frames} frames, {fps:.1f} fps, mean={float(shown_src.mean()):.1f}")
            else:
                cv2.imshow("LogiInsight IR", compose_view(shown_src, result, mode, thresh, fps))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("m"):
                    mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
                    print(f"  mode: {mode}")
                elif key in (ord("+"), ord("=")):
                    thresh = min(255, thresh + 5)
                    print(f"  threshold: {thresh}")
                elif key in (ord("-"), ord("_")):
                    thresh = max(0, thresh - 5)
                    print(f"  threshold: {thresh}")
                elif key == ord("s"):
                    os.makedirs(out_dir, exist_ok=True)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(os.path.join(out_dir, f"ir_{stamp}_{saved:03d}_raw.png"), shown_src)
                    cv2.imwrite(os.path.join(out_dir, f"ir_{stamp}_{saved:03d}_{mode}.png"), result)
                    saved += 1
                    print(f"  saved to {os.path.abspath(out_dir)}")

            if args.frames and frames >= args.frames:
                break
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C) -- shutting down cleanly.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        elapsed = max(time.time() - t_start, 1e-6)
        print(f"Camera released. {frames} frames in {elapsed:.1f}s ({frames / elapsed:.1f} fps avg).")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="LogiInsight -- near-infrared (IR) live view")
    p.add_argument("--device", type=str, default=None,
                   help="Camera index (e.g. 2). Default: auto-detect the IR (CPD) camera.")
    p.add_argument("--width", type=int, default=512,
                   help="Frame width, one of 256/512/1024 (default: 512)")
    p.add_argument("--height", type=int, default=640,
                   help="Frame height, one of 320/640/1280 (default: 640)")
    p.add_argument("--fps", type=float, default=30.0, help="Requested frame rate (default: 30)")
    p.add_argument("--mode", choices=MODES, default="stretch",
                   help="Processing mode shown on the right (default: stretch)")
    p.add_argument("--threshold", type=int, default=128,
                   help="Fixed threshold value 0-255 for 'threshold' mode (default: 128)")
    p.add_argument("--flicker-filter", action="store_true",
                   help="Per-pixel max of two consecutive frames (use if the image flickers "
                        "between lit and unlit frames).")
    p.add_argument("--output", type=str, default="ir_captures",
                   help="Folder for 's' key screenshots (default: ./ir_captures)")
    p.add_argument("--frames", type=int, default=0,
                   help="Stop after N frames (default: 0 = run until you quit)")
    p.add_argument("--no-display", action="store_true",
                   help="No window; print a status line every 30 frames (headless / testing)")
    args = p.parse_args()

    if (args.width, args.height) not in CPD_SIZES:
        print(f"NOTE: the camera advertises {CPD_SIZES}; {args.width}x{args.height} "
              f"may be rejected or replaced by the OS.")
    print("LogiInsight near-infrared example")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
