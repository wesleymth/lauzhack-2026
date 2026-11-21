"""
Copyright (c) 2026 Logitech Europe S.A.
SPDX-License-Identifier: MIT

LogiInsight Python API -- Host-side client for the LogiInsight smart webcam.

Two ways to consume AI data:

1. JSON-only (default, v1 compatible):
   - cv2.VideoCapture for frames, cam.get_latest_ai_data() for AI JSON.
   - Lightweight; the AI bbox/landmarks may lag the visual frame (the video
     and the AI results travel over two separate paths, and the AI runs
     slightly behind the video).

2. Device-side snapshot (v2):
   - cam.set_ai_config(enable_yuv_snapshot=True) then cam.get_device_snapshot().
   - Returns a (BGR image, ai_data) pair captured from the same NPU tick on
     the device, so overlays align exactly. Slow (about 2 s per snapshot at
     1920x1080).

Requirements:
    pip install paramiko
    pip install opencv-python numpy   # only needed for get_device_snapshot()

Connection settings (never put them in code): export LOGIWEBCAM_USER and
LOGIWEBCAM_PASSWORD, the values printed on the card in your kit.
"""

import io
import json
import logging
import os
import re
import time
from typing import Dict, Optional, Tuple

import paramiko

__version__ = "1.5.2"

logger = logging.getLogger(__name__)

# Device file paths -- expressed in the SFTP user's chrooted view, so they
# resolve directly via paramiko without needing any path translation.
_AI_CONFIG_PATH = "/files/logi_conf/logiwebcam_ai.ini"
_AI_OUTPUT_PATH = "/files/logi_conf/logiwebcam_ai_output.json"
_AI_YUV_STATUS  = "/files/logi_conf/logiwebcam_frame.status"
_VERSION_PATH   = "/files/logi_version.txt"
_INI_SECTION    = "logiwebcam_ai"

# The firmware version text written into logi_version.txt looks like
# "... hackathon-1.5.2 <variant> <commit>"; only the major.minor part has to
# match this SDK.
_FW_LABEL_RE = re.compile(r"hackathon-(\d+)\.(\d+)\.(\d+)")


class LogiWebcamConnectionError(Exception):
    """Raised when the SSH connection to the device fails."""


class LogiWebcamDataError(Exception):
    """Raised when AI output data cannot be read or parsed."""


class AiStalenessMonitor:
    """Tells you when the AI results stop updating.

    Call ``update(data)`` with every result from ``get_latest_ai_data()``. It
    returns a warning text **once** when ``timestamp_ms`` has not changed for
    ``max_age_s`` seconds (and again only after the results have resumed and
    stopped once more). It returns None the rest of the time.

    The AI only runs while a program reads the RGB camera (as MJPG, YUY2 or
    NV12 on firmware 1.5.2 or newer; firmware 1.5.1 needs MJPG), so a frozen
    timestamp usually means: the camera is not opened or nobody reads it. (With
    the AI switched off by set_ai_config() the timestamp still advances, but
    the faces list stays empty.)
    """

    MESSAGE = (
        "AI results are not updating (timestamp_ms is frozen). They only update "
        "while the RGB camera is open and read. Check that your program reads the "
        "RGB camera. (The AI works with MJPG, YUY2 and NV12 on firmware 1.5.2 or "
        "newer; older firmware needs MJPG.)"
    )

    def __init__(self, max_age_s: float = 2.0) -> None:
        self._max_age_s = max_age_s
        self._last_ts: Optional[int] = None
        self._last_change = time.monotonic()
        self._warned = False

    def update(self, data: Optional[Dict]) -> Optional[str]:
        now = time.monotonic()
        ts = None if data is None else data.get("timestamp_ms")
        if ts is not None and ts != self._last_ts:
            self._last_ts = ts
            self._last_change = now
            self._warned = False
            return None
        if not self._warned and now - self._last_change > self._max_age_s:
            self._warned = True
            return self.MESSAGE
        return None


def parse_firmware_label(text: Optional[str]) -> Optional[Tuple[int, int, int]]:
    """Return (major, minor, patch) from a logi_version.txt text, or None."""
    if not text:
        return None
    m = _FW_LABEL_RE.search(text)
    return tuple(int(g) for g in m.groups()) if m else None


def firmware_matches_sdk(text: Optional[str]) -> Optional[bool]:
    """True if the firmware's major.minor equals this SDK's, None if unknown."""
    fw = parse_firmware_label(text)
    if fw is None:
        return None
    sdk = tuple(int(x) for x in __version__.split(".")[:2])
    return fw[:2] == sdk


class LogiWebcamClient:
    """Client for controlling LogiInsight AI models over USB-NCM SSH/SFTP.

    The user name and password are on the card in your kit. Export them as
    LOGIWEBCAM_USER and LOGIWEBCAM_PASSWORD first (or pass them as arguments).

    Example (v1 style):
        >>> with LogiWebcamClient() as cam:
        ...     cam.set_ai_config(enable_face_detection=True, enable_face_mesh=True)
        ...     data = cam.get_latest_ai_data()
        ...     print(data["faces"])

    Example (v2 device-side snapshot):
        >>> with LogiWebcamClient() as cam:
        ...     cam.set_ai_config(enable_face_detection=True, enable_yuv_snapshot=True)
        ...     bgr, data = cam.get_device_snapshot()         # bgr is np.ndarray
        ...     cam.set_ai_config(enable_yuv_snapshot=False)  # stop tmp writes
    """

    def __init__(
        self,
        ip: str = "192.168.1.10",
        username: Optional[str] = None,
        password: Optional[str] = None,
        port: int = 22,
        timeout: int = 10,
    ) -> None:
        # No user name or password is stored in the repository. Pass them
        # explicitly or export LOGIWEBCAM_USER / LOGIWEBCAM_PASSWORD.
        if username is None:
            username = os.environ.get("LOGIWEBCAM_USER")
        if password is None:
            password = os.environ.get("LOGIWEBCAM_PASSWORD")
        if not username or password is None:
            raise ValueError(
                "SSH user name and password for the device are required (see the "
                "card in your kit): pass username=... and password=..., or set the "
                "LOGIWEBCAM_USER and LOGIWEBCAM_PASSWORD environment variables"
            )
        self._ip = ip
        self._username = username
        self._password = password
        self._port = port
        self._timeout = timeout
        self._ssh: Optional[paramiko.SSHClient] = None
        # Persistent SFTP channel for low-latency JSON / YUV reads. Each
        # `cat` over a fresh `exec_command` costs ~50-200ms; SFTP over an
        # already-open channel is ~5ms.
        self._sftp: Optional[paramiko.SFTPClient] = None

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        """Open the SSH connection. Idempotent."""
        if self._ssh is not None:
            return
        try:
            self._ssh = paramiko.SSHClient()
            self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self._ssh.connect(
                hostname=self._ip,
                port=self._port,
                username=self._username,
                password=self._password,
                look_for_keys=False,
                timeout=self._timeout,
            )
            logger.info("Connected to LogiInsight at %s:%d", self._ip, self._port)
        except Exception as e:
            self._ssh = None
            raise LogiWebcamConnectionError(
                f"Failed to connect to {self._ip}:{self._port}: {e}"
            ) from e

    def disconnect(self) -> None:
        """Close the SFTP channel and the SSH connection. Idempotent."""
        if self._sftp is not None:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._ssh is not None:
            try:
                self._ssh.close()
            except Exception:
                pass
            self._ssh = None
            logger.info("Disconnected from LogiInsight")

    def __enter__(self) -> "LogiWebcamClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        if self._ssh is None:
            return False
        transport = self._ssh.get_transport()
        return transport is not None and transport.is_active()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_sftp(self) -> paramiko.SFTPClient:
        if self._ssh is None:
            raise LogiWebcamConnectionError("Not connected. Call connect() first.")
        if self._sftp is None:
            self._sftp = self._ssh.open_sftp()
        return self._sftp

    def _exec(self, command: str, timeout: int = 10) -> str:
        if self._ssh is None:
            raise LogiWebcamConnectionError("Not connected. Call connect() first.")
        _, stdout, _ = self._ssh.exec_command(command, timeout=timeout)
        return stdout.read().decode().strip()

    # ------------------------------------------------------------------ #
    # AI configuration
    # ------------------------------------------------------------------ #

    def set_ai_config(
        self,
        enable_face_detection: bool = True,
        enable_face_mesh: bool = False,
        enable_debug_timestamp: bool = False,
        enable_yuv_snapshot: bool = False,
    ) -> bool:
        """Update AI model configuration on the device.

        Writes the values into the device's INI file; the firmware re-reads it
        about twice a second while streaming. The two optional flags default to
        False.

        Args:
            enable_face_detection: 1 = run face bounding-box inference.
            enable_face_mesh:      1 = also run 478-point face mesh
                                       (requires enable_face_detection=1).
            enable_debug_timestamp: kept for compatibility; the firmware no
                                        longer draws anything on the video, so
                                        this has no visible effect.
            enable_yuv_snapshot:   1 = also dump a paired NV12 YUV file per
                                       NPU cycle for get_device_snapshot().

        Returns:
            True on success.
        """
        ini_content = (
            f"[{_INI_SECTION}]\n"
            f"enable_face_detection={int(enable_face_detection)}\n"
            f"enable_face_mesh={int(enable_face_mesh)}\n"
            f"enable_debug_timestamp={int(enable_debug_timestamp)}\n"
            f"enable_yuv_snapshot={int(enable_yuv_snapshot)}\n"
        )
        try:
            self._exec(f"echo '{ini_content}' > {_AI_CONFIG_PATH}")
            logger.info(
                "AI config: face_det=%s face_mesh=%s debug_ts=%s yuv=%s",
                enable_face_detection, enable_face_mesh,
                enable_debug_timestamp, enable_yuv_snapshot,
            )
            return True
        except Exception as e:
            logger.error("Failed to set AI config: %s", e)
            return False

    # ------------------------------------------------------------------ #
    # AI data fetch
    # ------------------------------------------------------------------ #

    def get_latest_ai_data(self) -> Optional[Dict]:
        """Fetch the latest AI tracking JSON from the device.

        Returns a dict with the canonical keys:
            api_version      -- JSON schema version (2)
            plugin_loaded    -- False means "no AI results" (empty faces)
            timestamp_ms     -- device wall-clock ms (32-bit, can wrap or jump) taken
                                when the frame reached the AI pipeline; use it only to
                                compare with other values from the same device
            faces            -- list of {bbox, confidence, landmarks, ...}

        All bbox / landmark coordinates are pixels of the RGB stream (not sensor
        pixels), so they can be drawn directly on the frame from cv2.VideoCapture.

        When set_ai_config(enable_yuv_snapshot=True) is active, additional
        ``yuv_*`` keys appear (see get_device_snapshot for details). Returns
        None if the file is missing, empty, or malformed (the firmware writes
        atomically, so a torn read on the next call simply succeeds).
        """
        try:
            sftp = self._ensure_sftp()
            buf = io.BytesIO()
            sftp.getfo(_AI_OUTPUT_PATH, buf)
            raw = buf.getvalue().decode("utf-8", errors="replace").strip()
            if not raw:
                return None
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Malformed JSON from device, skipping read")
            return None
        except (FileNotFoundError, IOError):
            return None
        except Exception as e:
            logger.warning("Failed to read AI data: %s", e)
            self._sftp = None
            return None

    def get_firmware_version(self) -> Optional[str]:
        """Read the firmware version text from the device (first line).

        Looks like ``VERSIONMAGIC<...>Hemei <date> hackathon-1.5.2 <variant> <commit>``.
        Use ``firmware_matches_sdk(text)`` to compare it with this SDK.
        Returns None if the file cannot be read.
        """
        try:
            sftp = self._ensure_sftp()
            buf = io.BytesIO()
            sftp.getfo(_VERSION_PATH, buf)
            text = buf.getvalue().decode("utf-8", errors="replace").strip()
            return text.splitlines()[0].strip() if text else None
        except (FileNotFoundError, IOError):
            return None
        except Exception as e:
            logger.debug("get_firmware_version failed: %s", e)
            self._sftp = None
            return None

    def get_yuv_status(self) -> Optional[str]:
        """Read the firmware's one-line YUV-write status file.

        Format::

            ok ts_ms=12345 size=460800 count=42
            err stage=fwrite rc=-4 errno=28 No space left on device

        Returns None if the status file is absent. Useful when
        get_device_snapshot keeps returning None and you want to know why.
        """
        try:
            sftp = self._ensure_sftp()
            buf = io.BytesIO()
            sftp.getfo(_AI_YUV_STATUS, buf)
            return buf.getvalue().decode("utf-8", errors="replace").strip() or None
        except (FileNotFoundError, IOError):
            return None
        except Exception as e:
            logger.debug("get_yuv_status failed: %s", e)
            self._sftp = None
            return None

    def get_device_snapshot(
        self, retries: int = 3, verbose: bool = False
    ) -> Tuple[Optional["numpy.ndarray"], Optional[Dict]]:  # noqa: F821
        """Atomically fetch a (BGR image, AI JSON) pair from the device.

        Bypasses the host's own UVC pipeline -- the firmware writes the YUV
        file and the JSON from the same NPU iteration, so the returned image
        and AI data refer to the *same* sensor frame. Requires
        ``enable_yuv_snapshot=True`` in the AI config.

        Returns:
            ``(bgr, ai_data)`` on success.
            ``(None, ai_data)`` when the JSON exists but has no yuv_* fields
                (yuv_snapshot is disabled).
            ``(None, None)`` if even the JSON cannot be fetched after
                retries (no AI running, or torn writes).

        Requires ``numpy`` and ``opencv-python``.
        """
        # Local imports so callers that never use this method don't pay the
        # numpy/cv2 import cost.
        import cv2
        import numpy as np

        last_reason = "no attempts"
        for attempt in range(max(1, retries)):
            data = self.get_latest_ai_data()
            if data is None:
                last_reason = "JSON fetch returned None"
                continue
            yuv_path = data.get("yuv_path")
            yuv_ts   = data.get("yuv_capture_timestamp_ms")
            yuv_w    = data.get("yuv_width")
            yuv_h    = data.get("yuv_height")
            sy       = data.get("yuv_stride_y")
            suv      = data.get("yuv_stride_uv")
            hdr_size = data.get("yuv_header_size", 8)
            if not (yuv_path and yuv_ts and yuv_w and yuv_h and sy and suv):
                # Firmware didn't dump a YUV this cycle -- enable_yuv_snapshot
                # is off. Surface the device-side
                # status file so the user can tell which.
                status = self.get_yuv_status()
                last_reason = ("JSON has no yuv_* fields. "
                               "Set enable_yuv_snapshot=True or check the firmware.")
                if status:
                    last_reason += f" (device status: {status})"
                if verbose:
                    logger.warning("get_device_snapshot: %s", last_reason)
                return None, data

            try:
                sftp = self._ensure_sftp()
                buf = io.BytesIO()
                sftp.getfo(yuv_path, buf)
                raw = buf.getvalue()
            except (FileNotFoundError, IOError) as e:
                last_reason = f"YUV file missing at {yuv_path}: {e}"
                if verbose:
                    logger.warning("get_device_snapshot: %s", last_reason)
                continue
            except Exception as e:
                last_reason = f"YUV fetch failed: {e}"
                if verbose:
                    logger.warning("get_device_snapshot: %s", last_reason)
                self._sftp = None
                continue

            if len(raw) < hdr_size:
                last_reason = f"YUV file too short: {len(raw)} bytes < {hdr_size}"
                if verbose:
                    logger.warning("get_device_snapshot: %s", last_reason)
                continue

            file_ts = int.from_bytes(raw[:hdr_size], "little", signed=False)
            if file_ts != int(yuv_ts):
                last_reason = (f"YUV/JSON timestamp mismatch on attempt "
                               f"{attempt + 1}/{retries}: file={file_ts} json={yuv_ts}")
                if verbose:
                    logger.warning("get_device_snapshot: %s", last_reason)
                continue

            payload  = raw[hdr_size:]
            y_bytes  = sy * yuv_h
            uv_bytes = suv * (yuv_h // 2)
            if len(payload) < y_bytes + uv_bytes:
                last_reason = (f"YUV payload truncated: "
                               f"{len(payload)} < {y_bytes + uv_bytes}")
                if verbose:
                    logger.warning("get_device_snapshot: %s", last_reason)
                continue

            # NV12 buffers from the device may be row-padded to stride_y /
            # stride_uv (>= width). Strip the padding before passing to cv2,
            # which expects a tightly-packed (h*3/2, w) NV12 buffer.
            y_padded  = np.frombuffer(payload[:y_bytes],
                                      dtype=np.uint8).reshape(yuv_h, sy)
            uv_padded = np.frombuffer(payload[y_bytes:y_bytes + uv_bytes],
                                      dtype=np.uint8).reshape(yuv_h // 2, suv)
            nv12 = np.concatenate([
                y_padded[:, :yuv_w].reshape(-1),
                uv_padded[:, :yuv_w].reshape(-1),
            ]).reshape((yuv_h * 3) // 2, yuv_w)
            bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
            return bgr, data

        if verbose:
            logger.warning("get_device_snapshot: gave up after %d retries: %s",
                           retries, last_reason)
        return None, None

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def diagnose(self) -> Dict[str, str]:
        """One-shot health check, runs as the SFTP user.

        Probes only the files visible to the SDK (paths inside the user's
        chrooted view -- ps/df/dmesg are not available there). Prints and
        returns the dict so a quick ``cam.diagnose()`` is enough.
        """
        conf_dir = _AI_CONFIG_PATH.rsplit("/", 1)[0]
        probes = {
            "logi_conf_dir": f"ls -la {conf_dir}/ 2>&1",
            "ini_contents":  f"cat {_AI_CONFIG_PATH} 2>&1",
            "json_size":     f"wc -c {_AI_OUTPUT_PATH} 2>&1",
            "json_head":     f"head -c 800 {_AI_OUTPUT_PATH} 2>&1",
            "yuv_status":    f"cat {_AI_YUV_STATUS} 2>&1",
        }
        out: Dict[str, str] = {}
        print("=== LogiInsight diagnose() ===")
        for label, cmd in probes.items():
            try:
                out[label] = self._exec(cmd, timeout=5) or "(empty)"
            except Exception as e:
                out[label] = f"(probe failed: {e})"
            print(f"--- {label} ---")
            print(out[label])
            print()
        print("=== end diagnose() ===\n")
        return out
