"""
Copyright (c) 2026 Logitech Europe S.A.
SPDX-License-Identifier: MIT

LogiInsight PC SDK -- USB microphone helpers (Windows / macOS / Linux).

Used by example_audio.py. The camera's microphone shows up on
the host as a standard USB Audio Class (UAC1) input device, next to the UVC video
streams. Nothing on the device has to be configured: this module only records
from that input and looks at the samples.

What the audio is (see the Audio section of docs/API.md):

    format      signed 16-bit PCM, 48 kHz (16 kHz is also advertised)
    channels    2 are exposed by the current firmware (see EXPECTED_CHANNELS)
    processing  none: no noise suppression, echo cancellation, AGC or filtering
    name        "<product> Microphone"; <product> is "LVNext" by default

Samples are returned exactly as the OS delivers them: a NumPy int16 array of shape
(num_samples, num_channels), one column per USB channel, no mixing or resampling.

PortAudio (through the `sounddevice` package) does the capture. It is imported
lazily so that the maths helpers below work without it. On Linux it needs the
system library: `sudo apt install libportaudio2`.
"""

import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile

USB_VID = 0x046D
USB_PIDS = (0x094F, 0x0950, 0x0951)   # the PID follows the channel count: 2, 4, 5
AUDIO_NAME_TOKENS = ("LVNext", "Hello", "Logi")   # best-first
DEFAULT_SAMPLERATE = 48000
EXPECTED_CHANNELS = 2                 # what the current firmware exposes (see docs/API.md, Audio)

# Windows exposes one USB microphone through several PortAudio host APIs, and they can
# disagree on the channel count (seen on Windows: WASAPI 2, MME / DirectSound / WDM-KS 6). This
# order only breaks ties between endpoints with the same number of channels.
_HOSTAPI_RANK = ("Windows WASAPI", "Windows DirectSound", "Windows WDM-KS", "MME",
                 "Core Audio", "ALSA")


class AudioNotAvailableError(RuntimeError):
    """PortAudio / sounddevice cannot be used. The message says how to fix it."""


class AudioDeviceNotFoundError(RuntimeError):
    """No usable microphone. The message lists the input devices that were seen."""


class AudioStreamError(RuntimeError):
    """The input stream could not be opened or stopped delivering samples."""


_SD = None


def _sd():
    """Import sounddevice on first use (it loads the PortAudio library)."""
    global _SD
    if _SD is None:
        try:
            import sounddevice
        except ImportError as e:
            raise AudioNotAvailableError(
                "The 'sounddevice' package is missing. Run: uv sync   "
                "(or: pip install -r requirements.txt)") from e
        except OSError as e:  # PortAudio shared library not found
            raise AudioNotAvailableError(
                f"PortAudio library not found: {e}\n" + audio_hint()) from e
        _SD = sounddevice
    return _SD


# --------------------------------------------------------------------------- #
# Device discovery
# --------------------------------------------------------------------------- #

def list_audio_devices() -> List[Dict[str, Any]]:
    """Every input-capable audio device: index, name, hostapi, max_input_channels,
    default_samplerate. The same physical microphone can appear once per host API."""
    sd = _sd()
    hostapis = sd.query_hostapis()
    out = []
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            out.append({
                "index": index,
                "name": dev["name"],
                "hostapi": hostapis[dev["hostapi"]]["name"],
                "max_input_channels": int(dev["max_input_channels"]),
                "default_samplerate": float(dev["default_samplerate"]),
            })
    return out


def format_devices(devices: List[Dict[str, Any]]) -> str:
    """One line per device, for messages and `example_audio.py --list`."""
    if not devices:
        return "  (no input devices found)"
    return "\n".join(
        f"  [{d['index']:>2}] {d['name']}  |  {d['hostapi']}  |  "
        f"{d['max_input_channels']} ch  |  {d['default_samplerate']:.0f} Hz"
        for d in devices)


def _linux_usb_cards() -> List[int]:
    """ALSA card numbers whose USB id is the camera's (046d:094f / 0950 / 0951)."""
    cards = []
    base = "/proc/asound"
    try:
        names = os.listdir(base)
    except OSError:
        return cards
    for name in names:
        m = re.fullmatch(r"card(\d+)", name)
        if not m:
            continue
        try:
            with open(os.path.join(base, name, "usbid"), encoding="ascii") as f:
                usb_id = f.read().strip().lower()
        except OSError:
            continue
        for pid in USB_PIDS:
            if usb_id == f"{USB_VID:04x}:{pid:04x}":
                cards.append(int(m.group(1)))
    return cards


def _name_rank(name: str) -> Optional[int]:
    low = name.lower()
    for rank, token in enumerate(AUDIO_NAME_TOKENS):
        if token.lower() in low:
            return rank
    return None


def _hostapi_rank(hostapi: str) -> int:
    for rank, name in enumerate(_HOSTAPI_RANK):
        if name.lower() in hostapi.lower():
            return rank
    return len(_HOSTAPI_RANK)


def _sort_key(dev: Dict[str, Any], token_rank: int) -> Tuple[int, int, int, int]:
    # most channels first (host APIs differ: one test showed 6 on DirectSound, 2 on WASAPI),
    # then name token, then host API
    return (-dev["max_input_channels"], token_rank, _hostapi_rank(dev["hostapi"]),
            dev["index"])


def find_audio_device(device=None, min_channels: int = 1) -> Dict[str, Any]:
    """Pick the camera's microphone.

    device          None = auto-detect; an int (or digit string) = PortAudio device index;
                    any other string = case-insensitive substring of the device name.
    min_channels    ignore devices that expose fewer channels than this.

    Auto-detection matches the product name ("LVNext" ...), and on Linux also the
    USB id of the ALSA card, then prefers the endpoint with the most channels and the
    best host API. Raises AudioDeviceNotFoundError (with the device list) if nothing fits.
    """
    devices = list_audio_devices()
    usable = [d for d in devices if d["max_input_channels"] >= min_channels]

    if isinstance(device, str) and device.strip().isdigit():
        device = int(device.strip())

    if isinstance(device, int):
        for d in devices:
            if d["index"] == device:
                if d["max_input_channels"] < min_channels:
                    raise AudioDeviceNotFoundError(
                        f"Device {device} ({d['name']}, {d['hostapi']}) exposes only "
                        f"{d['max_input_channels']} channel(s), fewer than {min_channels}.\n"
                        f"Input devices:\n{format_devices(devices)}")
                return d
        raise AudioDeviceNotFoundError(
            f"Device index {device} is not an input device.\n"
            f"Input devices:\n{format_devices(devices)}")

    if isinstance(device, str):
        want = device.lower()
        hits = [d for d in usable if want in d["name"].lower()]
        if not hits:
            raise AudioDeviceNotFoundError(
                f"No input device with {device!r} in its name"
                f"{f' and >= {min_channels} channels' if min_channels > 1 else ''}.\n"
                f"Input devices:\n{format_devices(devices)}")
        return min(hits, key=lambda d: _sort_key(d, _name_rank(d["name"]) or 0))

    cards = _linux_usb_cards() if sys.platform.startswith("linux") else []
    candidates = []
    for d in usable:
        rank = _name_rank(d["name"])
        if rank is None:
            for card in cards:
                if re.search(rf"hw:{card}\b", d["name"]):
                    rank = 0
                    break
        if rank is not None:
            candidates.append((d, rank))
    if not candidates:
        raise AudioDeviceNotFoundError(
            "No camera microphone found (looked for "
            f"{', '.join(AUDIO_NAME_TOKENS)} in the device names"
            f"{' and for USB id 046d:094f' if cards or sys.platform.startswith('linux') else ''}"
            ").\n" + f"Input devices:\n{format_devices(devices)}\n" + audio_hint())
    return min(candidates, key=lambda c: _sort_key(c[0], c[1]))[0]


def audio_hint() -> str:
    """OS-specific checklist for 'no microphone / no audio' problems."""
    common = ("  - Is the camera plugged in? Its microphone is an input named "
              "'<product> Microphone'.\n"
              "  - Pick one yourself: python example_audio.py --list, then --device <index>.\n")
    if sys.platform == "win32":
        return ("Check:\n" + common +
                "  - Settings > Privacy & security > Microphone: allow desktop apps.\n"
                "  - Sound settings > Input: the device must be enabled, not 'Disabled'.\n"
                "  - The same microphone is listed once per host API. In our test WASAPI "
                "showed 2 channels and MME / DirectSound / WDM-KS showed 6 (channels 3-6 "
                "silent with this firmware).\n")
    if sys.platform == "darwin":
        return ("Check:\n" + common +
                "  - System Settings > Privacy & Security > Microphone: allow your "
                "terminal / IDE, then restart it.\n")
    return ("Check:\n" + common +
            "  - PortAudio is needed: sudo apt install libportaudio2\n"
            "  - arecord -l should list the camera (USB id 046d:094f).\n"
            "  - Another program (PulseAudio / PipeWire) may hold the device; "
            "select its entry or stop it.\n")


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #

class AudioRingRecorder:
    """Records continuously into a ring buffer that always holds the latest `seconds`.

    The PortAudio callback only copies samples into preallocated memory (two slice
    copies under a short lock): no allocation, no I/O, no FFT. Everything heavy
    (snapshot, spectrum, saving) is done by the caller on its own thread.

        rec = AudioRingRecorder(find_audio_device(), seconds=10)
        rec.start()
        ...
        data = rec.snapshot()          # (N, C) int16, oldest sample first
        rec.stop()
    """

    def __init__(self, device_info: Dict[str, Any], samplerate: Optional[int] = None,
                 channels: Optional[int] = None, seconds: float = 10.0):
        self.device_info = device_info
        self.requested_samplerate = samplerate
        self.requested_channels = channels
        self.seconds = float(seconds)
        self.samplerate = 0
        self.channels = 0
        self._stream = None
        self._lock = threading.Lock()
        self._ring = None
        self._cap = 0
        self._pos = 0
        self._filled = 0
        self.frames_total = 0
        self.callbacks = 0
        self.xruns = 0
        self.started_monotonic = None
        self.last_callback_monotonic = None

    # -- buffer ------------------------------------------------------------- #

    def prepare(self, samplerate: int, channels: int) -> None:
        """Allocate the ring for this format. start() calls it; tests may call it directly."""
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self._cap = max(1, int(math.ceil(self.seconds * self.samplerate)))
        self._ring = np.zeros((self._cap, self.channels), dtype=np.int16)
        self._pos = 0
        self._filled = 0
        self.frames_total = self.callbacks = self.xruns = 0
        self.last_callback_monotonic = None

    def _callback(self, indata, frames, time_info, status) -> None:
        """PortAudio callback (real-time thread). Keep it tiny."""
        n = len(indata)
        cap = self._cap
        with self._lock:
            ring = self._ring
            if n >= cap:
                ring[:] = indata[n - cap:]
                self._pos = 0
                self._filled = cap
            else:
                end = self._pos + n
                if end <= cap:
                    ring[self._pos:end] = indata
                else:
                    k = cap - self._pos
                    ring[self._pos:] = indata[:k]
                    ring[:end - cap] = indata[k:]
                self._pos = end % cap
                self._filled = min(cap, self._filled + n)
            self.frames_total += n
            self.callbacks += 1
            if (getattr(status, "input_overflow", False)
                    or getattr(status, "input_underflow", False)):
                self.xruns += 1
            self.last_callback_monotonic = time.monotonic()

    def snapshot(self) -> np.ndarray:
        """Copy of the latest recorded audio, shape (N, C), oldest sample first.
        N is at most `seconds * samplerate` (less until the ring has filled once)."""
        with self._lock:
            if self._ring is None:
                return np.zeros((0, max(self.channels, 1)), dtype=np.int16)
            if self._filled < self._cap:
                return self._ring[:self._filled].copy()
            return np.concatenate((self._ring[self._pos:], self._ring[:self._pos]))

    # -- stream ------------------------------------------------------------- #

    def start(self) -> None:
        """Open and start the input stream. Raises AudioStreamError / AudioNotAvailableError."""
        sd = _sd()
        channels = self.requested_channels or self.device_info["max_input_channels"]
        if self.requested_samplerate:
            rates = [int(self.requested_samplerate)]
        else:  # 48 kHz is what the firmware advertises; fall back to the OS mix rate
            rates = [DEFAULT_SAMPLERATE]
            default = int(self.device_info.get("default_samplerate") or 0)
            if default and default not in rates:
                rates.append(default)
        last_error = None
        for rate in rates:
            self.prepare(rate, channels)
            try:
                self._stream = sd.InputStream(
                    device=self.device_info["index"], channels=channels, samplerate=rate,
                    dtype="int16", blocksize=0, callback=self._callback)
                self._stream.start()
                self.started_monotonic = time.monotonic()
                return
            except Exception as e:  # noqa: BLE001 - PortAudioError and friends
                last_error = e
                self._close_stream()
        raise AudioStreamError(
            f"Cannot open {self.device_info['name']!r} ({self.device_info['hostapi']}) with "
            f"{channels} channel(s) at {'/'.join(str(r) for r in rates)} Hz: {last_error}")

    @property
    def active(self) -> bool:
        return self._stream is not None and bool(self._stream.active)

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:  # noqa: BLE001
                pass
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        """Stop and close the stream; safe to call twice."""
        self._close_stream()

    def stats(self) -> Dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            last = self.last_callback_monotonic
            return {
                "device": self.device_info["name"],
                "hostapi": self.device_info["hostapi"],
                "samplerate": self.samplerate,
                "channels": self.channels,
                "frames": self.frames_total,
                "callbacks": self.callbacks,
                "xruns": self.xruns,
                "seconds_since_callback": None if last is None else now - last,
                "elapsed_s": None if self.started_monotonic is None
                else now - self.started_monotonic,
            }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


def record_audio(duration_s: float = 10.0, device=None, samplerate: Optional[int] = None,
                 channels: Optional[int] = None, stop_event=None
                 ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Blocking recorder. Returns (audio, samplerate, stats).

    audio   int16 array of shape (N, C): one column per USB channel, untouched.
    stats   xruns, duration_actual_s, frames, samplerate, channels, device, hostapi,
            interrupted (True if stop_event ended the recording early).

    `device` is an index, a name substring, an already-resolved device dict, or None
    (auto-detect). `stop_event` is any object with is_set() / wait(timeout), e.g.
    camera_utils.install_stop_flag(). A recording cut short is returned as is.
    """
    info = device if isinstance(device, dict) else find_audio_device(device)
    rec = AudioRingRecorder(info, samplerate=samplerate, channels=channels, seconds=duration_s)
    interrupted = False
    rec.start()
    try:
        target = int(math.ceil(duration_s * rec.samplerate))
        t_start = time.monotonic()
        while rec.frames_total < target:
            if stop_event is not None:
                if stop_event.wait(0.1):
                    interrupted = True
                    break
            else:
                time.sleep(0.1)
            if time.monotonic() - t_start > duration_s + 5.0:
                raise AudioStreamError(
                    f"The microphone stopped delivering samples ({rec.frames_total} of "
                    f"{target} after {time.monotonic() - t_start:.1f} s). Was it unplugged?")
        data = rec.snapshot()
        stats = rec.stats()
    finally:
        rec.stop()
    stats.update({
        "duration_actual_s": len(data) / rec.samplerate,
        "interrupted": interrupted,
    })
    return data, rec.samplerate, stats


# --------------------------------------------------------------------------- #
# Analysis and saving
# --------------------------------------------------------------------------- #

def _as_float(audio: np.ndarray) -> np.ndarray:
    """(N,) or (N, C) samples -> float64 (N, C) scaled to [-1, 1) for integer input."""
    a = np.asarray(audio)
    if a.ndim == 1:
        a = a[:, None]
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(np.float64) / (np.iinfo(a.dtype).max + 1.0)
    return a.astype(np.float64)


def compute_power_spectrum(audio: np.ndarray, samplerate: int, nperseg: int = 1024
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel power spectral density (Welch, Hann window) from 0 Hz to fs/2.

    Returns (freqs_hz, psd_db): freqs_hz has shape (F,), psd_db has shape (F, C) in
    dBFS/Hz (0 dB = a full-scale constant). F = nperseg // 2 + 1.
    """
    x = _as_float(audio)
    if x.shape[0] < 2:
        raise ValueError("need at least 2 samples to compute a spectrum")
    freqs, psd = signal.welch(x, fs=samplerate, nperseg=min(nperseg, x.shape[0]), axis=0,
                              scaling="density")
    return freqs, 10.0 * np.log10(np.maximum(psd, 1e-20))


def channel_stats(audio: np.ndarray, samplerate: int, freqs: Optional[np.ndarray] = None,
                  psd_db: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Per-channel numbers for a quick look: rms_dbfs, peak_dbfs, dominant_hz, plus two
    integrity flags: `silent` (every sample is 0) and `duplicate_of` (1-based number of an
    earlier channel with bit-identical samples, else None). dBFS is relative to amplitude 1.0,
    so a full-scale sine reads -3.0 dB RMS and 0 dB peak.
    """
    raw = np.asarray(audio)
    if raw.ndim == 1:
        raw = raw[:, None]
    x = _as_float(raw)
    if freqs is None or psd_db is None:
        freqs, psd_db = compute_power_spectrum(raw, samplerate)
    out = []
    for c in range(x.shape[1]):
        col = x[:, c]
        rms = float(np.sqrt(np.mean(col * col))) if col.size else 0.0
        peak = float(np.max(np.abs(col))) if col.size else 0.0
        band = freqs > 20.0
        dominant = float(freqs[band][int(np.argmax(psd_db[band, c]))]) if band.any() else 0.0
        silent = not bool(np.any(raw[:, c]))
        duplicate = None
        if not silent:
            for earlier in range(c):
                if np.array_equal(raw[:, earlier], raw[:, c]):
                    duplicate = earlier + 1
                    break
        out.append({
            "channel": c + 1,
            "rms_dbfs": 20.0 * math.log10(max(rms, 1e-9)),
            "peak_dbfs": 20.0 * math.log10(max(peak, 1e-9)),
            "dominant_hz": dominant,
            "silent": silent,
            "duplicate_of": duplicate,
        })
    return out


def save_power_spectrum_plot(freqs_hz: np.ndarray, psd_db: np.ndarray, out_path,
                             title: str = "Per-Channel Power Spectrum",
                             labels: Optional[List[str]] = None) -> Path:
    """Draw one PSD curve per channel and save it as a PNG (no GUI needed: Agg canvas)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    psd_db = np.asarray(psd_db)
    if psd_db.ndim == 1:
        psd_db = psd_db[:, None]
    fig = Figure(figsize=(10, 5.5), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    for c in range(psd_db.shape[1]):
        ax.plot(freqs_hz[1:], psd_db[1:, c], linewidth=1.0,
                label=labels[c] if labels else f"ch{c + 1}")
    ax.set_xscale("log")
    ax.set_xlim(max(float(freqs_hz[1]), 1.0), float(freqs_hz[-1]))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (dBFS/Hz)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(str(out_path))
    return out_path


def save_wav(path, audio: np.ndarray, samplerate: int) -> Path:
    """Write a lossless multi-channel PCM .wav (int16 stays int16)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), int(samplerate), np.asarray(audio))
    return path


def timestamp() -> str:
    """File-name friendly local time, e.g. 20261113_153045."""
    return time.strftime("%Y%m%d_%H%M%S")
