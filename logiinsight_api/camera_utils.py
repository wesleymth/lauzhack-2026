"""
Copyright (c) 2026 Logitech Europe S.A.
SPDX-License-Identifier: MIT

LogiInsight PC SDK -- cross-platform camera helpers (Windows / macOS / Linux).

Used by example_IR.py. No SSH is involved here: this
module only deals with the UVC video side of the camera.

What the camera shows up as on the host (one USB device, several UVC streams):

    "<product> Camera"       RGB sensor, MJPG / YUY2 / NV12
    "<product> CPD Camera"   near-infrared sensor, MJPG, PORTRAIT frames
    "<product> Hello Camera" Windows Hello IR stream (L8_IR 480x480). Reserved
                             for Windows Hello; this SDK does not use it.

<product> is "LVNext" by default. Camera *indexes* are not stable between
operating systems or between plug-ins, so cameras are picked by name where the
OS exposes names (Linux) and by probing otherwise.
"""

import os
import signal
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2

IR_NAME_TOKEN = "CPD"          # "<product> CPD Camera" = MJPG near-infrared stream
HELLO_NAME_TOKEN = "Hello"     # "<product> Hello Camera" = Windows Hello, not used
IR_PROBE_SIZE = (512, 640)     # a CPD size; portrait, so it can't be mistaken for RGB


class CameraNotFoundError(RuntimeError):
    """No usable camera. The message includes OS-specific things to check."""


# --------------------------------------------------------------------------- #
# Backends / opening
# --------------------------------------------------------------------------- #

def uvc_backends() -> List[Tuple[str, int]]:
    """Capture backends to try, best-first, for the current platform.

    DSHOW is listed before MSMF on Windows because MSMF often ignores
    CAP_PROP_FOURCC and silently falls back to an uncompressed format.
    """
    if sys.platform.startswith("linux"):
        return [("V4L2", cv2.CAP_V4L2)]
    if sys.platform == "win32":
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
    if sys.platform == "darwin":
        return [("AVFoundation", cv2.CAP_AVFOUNDATION)]
    return [("default", cv2.CAP_ANY)]


def backend_id(name: Optional[str]) -> Optional[int]:
    """OpenCV backend constant for a name returned by find_camera()/list_cameras()."""
    for n, b in uvc_backends():
        if n == name:
            return b
    return None


def decode_fourcc(value: float) -> str:
    v = int(value)
    if v <= 0:
        return f"raw={v}"
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4))


def device_label(index: int) -> str:
    return f"/dev/video{index}" if sys.platform.startswith("linux") else f"camera index {index}"


def parse_device_arg(text: str) -> int:
    """Accept a bare index ("2") or, on Linux, a device path ("/dev/video2")."""
    t = text.strip()
    if t.startswith("/dev/video"):
        t = t[len("/dev/video"):]
    try:
        return int(t)
    except ValueError:
        raise CameraNotFoundError(
            f"Cannot parse --device {text!r}. Use a camera index such as 0 "
            f"(or /dev/video0 on Linux).") from None


def open_camera(
    index: int,
    backend: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fourcc: Optional[str] = "MJPG",
    fps: Optional[float] = None,
) -> Tuple["cv2.VideoCapture", Dict[str, Any]]:
    """Open one camera. Returns (capture, info).

    The order is size, then fps, then FOURCC: on DirectShow, setting the FOURCC
    before the size, or the fps after it, silently negotiates YUY2 instead of MJPG. The negotiated values
    are read back into `info` (width, height, fourcc, fps, backend) so callers
    can detect that the OS ignored a request. Raises CameraNotFoundError.
    """
    candidates = [(n, b) for n, b in uvc_backends() if backend is None or b == backend]
    if backend is not None and not candidates:
        candidates = [("custom", backend)]
    cap = None
    used = ""
    for name, be in candidates:
        c = cv2.VideoCapture(index, be)
        if c.isOpened():
            cap, used = c, name
            break
        c.release()
    if cap is None:
        raise CameraNotFoundError(f"Cannot open {device_label(index)}.\n{camera_hint()}")

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # FPS goes before FOURCC: on DirectShow, requesting an fps after the FOURCC
    # silently renegotiates the stream to YUY2 (seen on Windows with the
    # RGB camera at 1920x1080), which would disable the on-device AI.
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    info = {
        "index": index,
        "backend": used,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fourcc": decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
    }
    return cap, info


def describe(info: Dict[str, Any]) -> str:
    return (f"{device_label(info['index'])} [{info['backend']}]  "
            f"{info['width']}x{info['height']} @ {info['fps']:.0f}fps ({info['fourcc']})")


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def _linux_names() -> Dict[int, str]:
    """{video index: V4L2 card name} from sysfs (Linux only)."""
    names: Dict[int, str] = {}
    base = "/sys/class/video4linux"
    try:
        entries = os.listdir(base)
    except OSError:
        return names
    for entry in entries:
        if not entry.startswith("video"):
            continue
        try:
            idx = int(entry[len("video"):])
            with open(os.path.join(base, entry, "name"), encoding="utf-8", errors="replace") as f:
                names[idx] = f.read().strip()
        except (ValueError, OSError):
            continue
    return names


def is_grayscale(frame, tolerance: int = 2) -> bool:
    """True if all three channels of a BGR frame are (nearly) identical.

    An IR stream decoded from MJPG by OpenCV is a 3-channel image with equal
    channels, so this is a cheap 'is this the IR camera?' hint.
    """
    if frame is None or frame.ndim == 2:
        return frame is not None
    if frame.shape[2] < 3:
        return True
    b = frame[::8, ::8, 0].astype("int16")
    g = frame[::8, ::8, 1].astype("int16")
    r = frame[::8, ::8, 2].astype("int16")
    return bool(abs(b - g).max() <= tolerance and abs(g - r).max() <= tolerance)


def list_cameras(max_index: int = 10, probe_size: Optional[Tuple[int, int]] = None
                 ) -> List[Dict[str, Any]]:
    """Probe camera indexes and return those that deliver a frame.

    Each entry: index, backend, name (Linux only, else ""), width, height,
    fourcc, gray (bool). `probe_size=(w, h)` asks for that size (MJPG) first.
    Opening a camera briefly starts its stream on the device.
    """
    names = _linux_names() if sys.platform.startswith("linux") else {}
    found: List[Dict[str, Any]] = []
    for _be_name, be in uvc_backends():
        for idx in range(max_index):
            if sys.platform.startswith("linux") and not os.path.exists(f"/dev/video{idx}"):
                continue
            try:
                if probe_size:
                    cap, info = open_camera(idx, be, probe_size[0], probe_size[1], "MJPG")
                else:
                    cap, info = open_camera(idx, be, fourcc=None)
            except CameraNotFoundError:
                continue
            try:
                ok, frame = cap.read()
            finally:
                cap.release()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            info.update({"name": names.get(idx, ""), "width": w, "height": h,
                         "gray": is_grayscale(frame)})
            found.append(info)
        if found:
            break  # this backend works on this platform; don't try the others
    return found


def find_camera(kind: str = "rgb") -> Tuple[int, str]:
    """Locate the RGB ("rgb") or near-infrared CPD ("ir") stream.

    Returns (index, backend_name). Raises CameraNotFoundError.

    Linux: matches the V4L2 card name ("... CPD Camera" = IR; a "... Camera"
    that is neither CPD nor Hello = RGB).
    Windows / macOS: the OS gives no names through OpenCV, so cameras are
    probed. A frame taller than wide (CPD frames are portrait) or a perfectly
    grey one is treated as IR; anything else as RGB. If the PC has a built-in
    webcam this can pick the wrong camera -- pass --device in that case.
    """
    if kind not in ("rgb", "ir"):
        raise ValueError("kind must be 'rgb' or 'ir'")

    if sys.platform.startswith("linux"):
        names = _linux_names()
        want = []
        for idx in sorted(names):
            n = names[idx]
            if IR_NAME_TOKEN in n and HELLO_NAME_TOKEN not in n:
                is_ir = True
            elif HELLO_NAME_TOKEN in n:
                continue
            else:
                is_ir = False
            if (kind == "ir") == is_ir and (is_ir or "Camera" in n):
                want.append(idx)
        for idx in want:
            # each UVC function also exposes a metadata node: keep only
            # nodes that really deliver frames
            try:
                cap, _ = open_camera(idx, fourcc=None)
            except CameraNotFoundError:
                continue
            try:
                ok, frame = cap.read()
            finally:
                cap.release()
            if ok and frame is not None:
                return idx, "V4L2"
        if want:
            raise CameraNotFoundError(
                f"Found {kind.upper()} camera node(s) {[device_label(i) for i in want]} "
                f"but none delivered a frame.\n{camera_hint()}")
        # fall through to probing when the product name is not the default

    cams = list_cameras(probe_size=IR_PROBE_SIZE if kind == "ir" else None)
    if kind == "ir":
        ir = [c for c in cams if c["height"] > c["width"]] or [c for c in cams if c["gray"]]
        if ir:
            return ir[0]["index"], ir[0]["backend"]
        raise CameraNotFoundError("No near-infrared (CPD) camera found.\n" + camera_hint())
    rgb = [c for c in cams if c["width"] >= c["height"] and not c["gray"]]
    if not rgb:
        raise CameraNotFoundError("No RGB camera found.\n" + camera_hint())
    if len(rgb) > 1:
        print("  NOTE: several RGB-like cameras found: "
              + ", ".join(device_label(c["index"]) for c in rgb)
              + f". Using {device_label(rgb[0]['index'])}; pass --device to choose another "
                "(a laptop's built-in webcam is also listed).")
    return rgb[0]["index"], rgb[0]["backend"]


def camera_hint() -> str:
    """OS-specific checklist shown when no camera can be opened."""
    common = ("  - Is the camera plugged in with a USB 3 cable and recognised by the OS?\n"
              "  - Is another program (Teams, Zoom, browser, camera app) using it?\n")
    if sys.platform == "win32":
        specific = ("  - Windows: Settings > Privacy & security > Camera > allow desktop apps.\n"
                    "  - Check Device Manager > Cameras for the LVNext devices.\n")
    elif sys.platform == "darwin":
        specific = ("  - macOS: System Settings > Privacy & Security > Camera > allow your "
                    "terminal / IDE, then restart it.\n"
                    "  - macOS often ignores the MJPG request; frames still arrive but as raw video.\n")
    else:
        specific = ("  - Linux: `ls -l /dev/video*` and `v4l2-ctl --list-devices`; your user "
                    "must be in the `video` group (then log out and in).\n")
    return "Things to check:\n" + common + specific


# --------------------------------------------------------------------------- #
# Clean shutdown
# --------------------------------------------------------------------------- #

class StopFlag:
    """Set by Ctrl+C / SIGTERM so capture loops can leave cleanly."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)


def install_stop_flag() -> StopFlag:
    """Make Ctrl+C (and SIGTERM / Ctrl+Break) set a flag instead of raising.

    Loops should test `flag.is_set()` and leave through their `finally:` block
    (cap.release(); cv2.destroyAllWindows()). A SECOND Ctrl+C raises
    KeyboardInterrupt so a loop stuck inside a blocking read can still be
    aborted.
    """
    flag = StopFlag()

    def _on_signal(signum, _frame):
        if flag.is_set():
            raise KeyboardInterrupt
        print("\nStopping (Ctrl+C again to force) ...", flush=True)
        flag.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):  # not in main thread / unsupported
            pass
    return flag
