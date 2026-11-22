#!/usr/bin/env python3
"""
Local speech-to-text with word and sentence timestamps.

Records from a selectable input device until Ctrl+C. While recording, audio is
written to disk continuously and transcribed in ~25 s chunks (cut at a quiet
point) by a background thread, so transcript.json grows during the session.

Usage:
    python stt.py --list               # list input devices
    python stt.py                      # system default input
    python stt.py --camera             # LogiInsight camera microphone
    python stt.py --device 3           # by index (or part of the name)

Output: sessions/<YYYY-MM-DD_HH-MM-SS>/{session.wav, transcript.json}
Times in transcript.json are seconds since session_start_epoch.
"""

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
import wave
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "logiinsight_api"))
from audio_utils import (  # noqa: E402
    AudioDeviceNotFoundError,
    find_audio_device,
    format_devices,
    list_audio_devices,
)

MODEL = "small.en"
TARGET_SR = 16000
CHUNK_S = 25.0          # nominal chunk length sent to the transcriber
SEARCH_S = 5.0          # look for a quiet cut point in the last SEARCH_S seconds of a chunk
WIN_S = 0.2             # energy window for finding the quiet point
PROMPT_CHARS = 200      # previous text passed as context to the next chunk
SENTENCE_END = (".", "?", "!")


# --------------------------------------------------------------------------- #
# Device selection
# --------------------------------------------------------------------------- #

def pick_device(args):
    import sounddevice as sd
    if args.camera:
        return find_audio_device(None)
    if args.device is not None:
        return find_audio_device(args.device)
    idx = sd.default.device[0]
    if idx is None or idx < 0:
        raise AudioDeviceNotFoundError("No default input device.\n"
                                       + format_devices(list_audio_devices()))
    return find_audio_device(int(idx))


def channel_selection(info):
    """Channels to open and which ones to average. The camera exposes its 2 real mics as
    channels 0 and 3 when the OS offers a 6-channel format (see logiinsight_api docs §11)."""
    n = info["max_input_channels"]
    if n == 6 and any(t in info["name"] for t in ("LVNext", "Logi")):
        return n, [0, 3]
    n = min(n, 2)
    return n, list(range(n))


# --------------------------------------------------------------------------- #
# Transcript handling
# --------------------------------------------------------------------------- #

def fmt(t):
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"


class Transcript:
    """Collects words, groups them into sentences, writes transcript.json atomically."""

    def __init__(self, out_dir: Path):
        self.path = out_dir / "transcript.json"
        self.session_start_epoch = None
        self.sentences = []
        self.pending = []      # words of a sentence not finished yet (may continue next chunk)

    def add_words(self, words, final=False):
        for w in words:
            self.pending.append(w)
            if w["word"].rstrip("\"')]").endswith(SENTENCE_END):
                self._close_sentence()
        if final and self.pending:
            self._close_sentence()
        self.write()

    def _close_sentence(self):
        words, self.pending = self.pending, []
        s = {
            "start": words[0]["time"],
            "end": words[-1]["end"],
            "text": " ".join(w["word"] for w in words),
            "words": [{"word": w["word"], "time": w["time"]} for w in words],
        }
        self.sentences.append(s)
        print(f"[{fmt(s['start'])}–{fmt(s['end'])}] {s['text']}", flush=True)

    def text_tail(self, n=PROMPT_CHARS):
        text = " ".join(s["text"] for s in self.sentences[-10:])
        text += " " + " ".join(w["word"] for w in self.pending)
        return text.strip()[-n:] or None

    def write(self):
        data = {"session_start_epoch": self.session_start_epoch, "sentences": self.sentences}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        os.replace(tmp, self.path)


def transcriber_worker(model, jobs: "queue.Queue", transcript: Transcript):
    while True:
        job = jobs.get()
        if job is None:
            break
        audio16k, offset_s, final = job
        segments, _ = model.transcribe(
            audio16k, language="en", beam_size=5, word_timestamps=True,
            vad_filter=True, condition_on_previous_text=False,
            initial_prompt=transcript.text_tail())
        words = []
        for seg in segments:
            for w in seg.words or []:
                word = w.word.strip()
                if word:
                    words.append({"word": word,
                                  "time": round(offset_s + w.start, 2),
                                  "end": round(offset_s + w.end, 2)})
        transcript.add_words(words, final=final)
        if jobs.qsize() > 1:
            print(f"  (transcriber is {jobs.qsize()} chunks behind)", flush=True)


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #

def quiet_cut(buf: np.ndarray, sr: int) -> int:
    """Index in buf to cut at: centre of the lowest-energy window in the last SEARCH_S s."""
    win = int(WIN_S * sr)
    start = max(0, len(buf) - int(SEARCH_S * sr))
    tail = buf[start:]
    if len(tail) < win * 2:
        return len(buf)
    energy = np.convolve(tail.astype(np.float32) ** 2, np.ones(win, np.float32), "valid")
    return start + int(np.argmin(energy)) + win // 2


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    x = x.astype(np.float32) / 32768.0
    if sr == TARGET_SR:
        return x
    g = gcd(TARGET_SR, sr)
    return resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)


def record(args):
    import sounddevice as sd
    from faster_whisper import WhisperModel

    info = pick_device(args)
    sr = int(info["default_samplerate"])
    n_ch, use_ch = channel_selection(info)
    print(f"Input: [{info['index']}] {info['name']} ({info['hostapi']}), "
          f"{n_ch} ch @ {sr} Hz, using channels {use_ch}")

    out_dir = HERE / "sessions" / time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {MODEL} (first run downloads it) ...", flush=True)
    model = WhisperModel(MODEL, device="cpu", compute_type="int8")

    transcript = Transcript(out_dir)
    jobs: "queue.Queue" = queue.Queue()
    worker = threading.Thread(target=transcriber_worker, args=(model, jobs, transcript),
                              daemon=True)
    worker.start()

    blocks: "queue.Queue" = queue.Queue()
    status_msgs = []

    def callback(indata, frames, time_info, status):
        if status:
            status_msgs.append(str(status))
        if transcript.session_start_epoch is None:
            transcript.session_start_epoch = round(time.time() - frames / sr, 3)
        blocks.put(indata[:, use_ch].mean(axis=1).astype(np.int16))

    wav = wave.open(str(out_dir / "session.wav"), "wb")
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sr)

    buf = np.zeros(0, np.int16)   # native-rate samples not yet sent to the transcriber
    consumed = 0                  # native-rate samples already sent (gives chunk offsets)
    chunk_len = int(CHUNK_S * sr)

    stream = sd.InputStream(device=info["index"], channels=n_ch, samplerate=sr,
                            dtype="int16", blocksize=int(0.05 * sr), callback=callback)
    print("Recording. Press Ctrl+C to stop.\n", flush=True)
    try:
        with stream:
            while True:
                try:
                    block = blocks.get(timeout=0.5)
                except queue.Empty:
                    continue
                wav.writeframes(block.tobytes())   # header is patched on every write
                buf = np.concatenate([buf, block])
                if len(buf) >= chunk_len:
                    cut = quiet_cut(buf, sr)
                    jobs.put((to_16k(buf[:cut], sr), consumed / sr, False))
                    consumed += cut
                    buf = buf[cut:]
                while status_msgs:
                    print(f"  (audio: {status_msgs.pop(0)})", flush=True)
    except KeyboardInterrupt:
        print("\nStopping, transcribing the last part ...", flush=True)

    while not blocks.empty():                       # drain what the callback still queued
        block = blocks.get()
        wav.writeframes(block.tobytes())
        buf = np.concatenate([buf, block])
    wav.close()

    jobs.put((to_16k(buf, sr), consumed / sr, True) if len(buf) else None)
    if len(buf):
        jobs.put(None)
    try:
        worker.join()
    except KeyboardInterrupt:
        print("Interrupted again; transcript saved up to the last finished chunk.")
        return 130
    if not len(buf):
        transcript.add_words([], final=True)
    print(f"\nSaved {out_dir.relative_to(HERE)}/ "
          f"({len(transcript.sentences)} sentences, {(consumed + len(buf)) / sr:.1f} s audio)")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list input devices and exit")
    p.add_argument("--device", help="input device index or part of its name")
    p.add_argument("--camera", action="store_true", help="use the LogiInsight camera mic")
    args = p.parse_args()

    def _stop(*_):
        raise KeyboardInterrupt
    # Ctrl+C and SIGTERM both stop cleanly (even if SIGINT was inherited as ignored).
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        if args.list:
            print(format_devices(list_audio_devices()))
            return 0
        return record(args)
    except AudioDeviceNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
