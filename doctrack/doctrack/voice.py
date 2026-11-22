"""Voice notes pinned to the locked document, exported as a commented PDF.

While a document is locked, `v` starts/stops recording. Speech is transcribed
locally with the team's speech-to-text (../speech/stt.py: faster-whisper, word
timestamps). Each SENTENCE becomes one note, pinned where the right index
finger pointed (pointing pose, any direction) on the page when the sentence
began. A sentence spoken with no fingertip on the page joins the comment of the
last fingertip position (VoiceNotes.notes). `p` exports
the flattened scan as a PDF with one sticky-note comment per sentence.

Pointer positions are stored in normalised document coordinates (u, v in
[0, 1], origin TL) in the lock-time corner order, so rotating / mirroring the
scan afterwards still puts the notes in the right place (see
DocumentTracker.to_doc / view_transform).

Output: voice_notes/<date_time>/
    rec_01/session.wav, rec_01/transcript.json   one folder per v ... v recording
    pointer.jsonl                                 fingertip samples (epoch, u, v)
    notes_<time>.pdf, notes_<time>.json           one pair per export
"""

import io
from collections import Counter
import json
import queue
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import numpy as np

SPEECH_DIR = Path(__file__).resolve().parents[2] / "speech"
UNIT = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)  # TL TR BR BL


def _stt():
    """The team's speech-to-text module (helpers only: no recording loop runs on import)."""
    if str(SPEECH_DIR) not in sys.path:
        sys.path.insert(0, str(SPEECH_DIR))
    import stt
    return stt


# --------------------------------------------------------------------------- #
# Fingertip positions over time
# --------------------------------------------------------------------------- #

class PointerLog:
    """Fingertip samples (wall-clock epoch, u, v) in document coordinates."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.t = []
        self.uv = []
        self._file = open(path, "a", buffering=1) if path else None   # line-buffered

    def add(self, t: float, u: float, v: float) -> None:
        self.t.append(t)
        self.uv.append((u, v))
        if self._file:
            self._file.write(json.dumps({"t": round(t, 3), "u": round(u, 4), "v": round(v, 4)}) + "\n")

    def at(self, start: float, end: float, before: float = 1.0,
           after: float = 0.5) -> Optional[Tuple[float, float]]:
        """Where the finger was when a sentence spoken in [start, end] began: the sample
        closest to `start`, looking from `before` s earlier to `after` s past the end."""
        if not self.t:
            return None
        t = np.asarray(self.t)
        ok = (t >= start - before) & (t <= end + after)
        if not ok.any():
            return None
        i = np.flatnonzero(ok)[np.argmin(np.abs(t[ok] - start))]
        return self.uv[i]

    def last_before(self, t0: float) -> Optional[Tuple[float, Tuple[float, float]]]:
        """(time, uv) of the last sample at or before t0, or None."""
        i = int(np.searchsorted(np.asarray(self.t), t0, side="right")) - 1
        return (self.t[i], self.uv[i]) if i >= 0 else None

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


def pick_pointer(hands, to_doc, hand: str = "Right", any_pose: bool = False):
    """Which fingertip places the notes this frame: (uv, hand, status).

    Hands labelled `hand` are tried first, then the others: MediaPipe's Left/Right
    guess is unreliable from a top-down desk camera, so the label only sets priority.
    The pointing pose is required (unless `any_pose`) but NOT its direction: you point
    at a spot from wherever your hand is. `status` says why nothing was picked.
    """
    if not hands:
        return None, None, "no hand seen"
    ordered = sorted(hands, key=lambda h: hand != "any" and h.handedness != hand)
    reasons = []
    for h in ordered:
        uv = to_doc(*h.tip)
        if uv is None:
            reasons.append("fingertip off the page")
        elif not any_pose and not h.pointing_any_dir:
            reasons.append(h.reason_any_dir or "not pointing")
        else:
            return uv, h, "ok"
    return None, None, reasons[0]


@dataclass
class Note:
    index: int                              # 1-based, in speaking order
    text: str
    start: float                            # wall-clock epoch of the first word
    end: float
    uv: Optional[Tuple[float, float]]       # lock-frame doc coords; None = finger not on the page

    def to_dict(self, uv_view=None) -> dict:
        uv = uv_view if uv_view is not None else self.uv
        return {"index": self.index, "text": self.text,
                "start_epoch": round(self.start, 3), "end_epoch": round(self.end, 3),
                "u": None if uv is None else round(float(uv[0]), 4),
                "v": None if uv is None else round(float(uv[1]), 4)}


# --------------------------------------------------------------------------- #
# One recording (v ... v): mic -> wav + chunked transcription
# --------------------------------------------------------------------------- #

class _Model:
    """faster-whisper model, loaded once in the background on the first recording."""

    def __init__(self) -> None:
        self.model = None
        self.error: Optional[Exception] = None
        self.ready = threading.Event()
        self._started = False

    def load(self) -> None:
        if self._started:
            return
        self._started = True

        def run():
            try:
                from faster_whisper import WhisperModel
                stt = _stt()
                print(f"voice: loading model {stt.MODEL} (first run downloads it) ...", flush=True)
                self.model = WhisperModel(stt.MODEL, device="cpu", compute_type="int8")
                print("voice: model ready", flush=True)
            except Exception as e:  # noqa: BLE001  (reported, recording keeps the wav)
                self.error = e
                print(f"voice: cannot load the speech model: {e}", flush=True)
            self.ready.set()
        threading.Thread(target=run, daemon=True).start()


class Recording:
    """Same pipeline as stt.record(), but started/stopped by the app instead of Ctrl+C."""

    def __init__(self, out_dir: Path, device_info: dict, model: _Model, chunk_s: float) -> None:
        stt = _stt()
        self._stt = stt
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.transcript = stt.Transcript(out_dir)
        self._model = model
        self._chunk_s = chunk_s
        self._info = device_info
        self._sr = int(device_info["default_samplerate"])
        self._n_ch, self._use_ch = stt.channel_selection(device_info)
        self._blocks: "queue.Queue" = queue.Queue()
        self._jobs: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self.done = threading.Event()        # set once everything is transcribed and saved
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd
        sr = self._sr

        def callback(indata, frames, time_info, status):
            if self.transcript.session_start_epoch is None:
                self.transcript.session_start_epoch = round(time.time() - frames / sr, 3)
            self._blocks.put(indata[:, self._use_ch].mean(axis=1).astype(np.int16))

        self._stream = sd.InputStream(device=self._info["index"], channels=self._n_ch, samplerate=sr,
                                      dtype="int16", blocksize=int(0.05 * sr), callback=callback)
        self._stream.start()
        threading.Thread(target=self._record, daemon=True).start()
        threading.Thread(target=self._transcribe, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def recording(self) -> bool:
        return self._stream is not None and not self._stop.is_set()

    def _record(self) -> None:
        stt, sr = self._stt, self._sr
        wav = wave.open(str(self.out_dir / "session.wav"), "wb")
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        buf = np.zeros(0, np.int16)   # native-rate samples not yet sent to the transcriber
        consumed = 0                  # samples already sent (gives each chunk's time offset)
        chunk_len = int(self._chunk_s * sr)
        try:
            while not self._stop.is_set():
                try:
                    block = self._blocks.get(timeout=0.2)
                except queue.Empty:
                    continue
                wav.writeframes(block.tobytes())
                buf = np.concatenate([buf, block])
                if len(buf) >= chunk_len:
                    cut = stt.quiet_cut(buf, sr)
                    self._jobs.put((stt.to_16k(buf[:cut], sr), consumed / sr, False))
                    consumed += cut
                    buf = buf[cut:]
        finally:
            self._stream.stop()
            self._stream.close()
            while not self._blocks.empty():
                block = self._blocks.get()
                wav.writeframes(block.tobytes())
                buf = np.concatenate([buf, block])
            wav.close()
            if len(buf):
                self._jobs.put((stt.to_16k(buf, sr), consumed / sr, True))
            self._jobs.put(None)

    def _transcribe(self) -> None:
        self._model.ready.wait()
        if self._model.model is not None:
            try:
                self._stt.transcriber_worker(self._model.model, self._jobs, self.transcript)
            except Exception as e:  # noqa: BLE001
                print(f"voice: transcription failed: {e}", flush=True)
        if self.transcript.pending:                 # nothing final was transcribed: flush
            self.transcript.add_words([], final=True)
        self.done.set()

    def sentences(self) -> List[Tuple[float, float, str]]:
        """(start epoch, end epoch, text) of the sentences transcribed so far."""
        t0 = self.transcript.session_start_epoch
        if t0 is None:
            return []
        return [(t0 + s["start"], t0 + s["end"], s["text"]) for s in list(self.transcript.sentences)]


# --------------------------------------------------------------------------- #
# All notes for the locked document
# --------------------------------------------------------------------------- #

class VoiceNotes:
    def __init__(self, out_root: Path, mic: Optional[str] = None, camera_mic: bool = False,
                 chunk_s: float = 10.0) -> None:
        self.out_root = Path(out_root)
        self._mic = SimpleNamespace(device=mic, camera=camera_mic)
        self._chunk_s = chunk_s
        self._model = _Model()
        self._device = None
        self.session_dir: Optional[Path] = None
        self.pointer = PointerLog()
        self.recordings: List[Recording] = []
        self.misses: Counter = Counter()      # why no fingertip was taken, per frame
        self.frames = 0

    @property
    def recording(self) -> bool:
        return bool(self.recordings) and self.recordings[-1].recording

    @property
    def busy(self) -> bool:
        """Recording, or still transcribing."""
        return any(not r.done.is_set() for r in self.recordings)

    def reset(self) -> None:
        """New document locked: start a fresh, empty set of notes."""
        self.stop()
        self.pointer.close()
        self.session_dir = None
        self.pointer = PointerLog()
        self.recordings = []

    def _ensure_session(self) -> Path:
        if self.session_dir is None:
            self.session_dir = self.out_root / time.strftime("%Y-%m-%d_%H-%M-%S")
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self.pointer = PointerLog(self.session_dir / "pointer.jsonl")
        return self.session_dir

    def start(self) -> None:
        if self.recording:
            return
        if self._device is None:
            stt = _stt()
            self._device = stt.pick_device(self._mic)
            print(f"voice: mic [{self._device['index']}] {self._device['name']}", flush=True)
        self._model.load()
        rec = Recording(self._ensure_session() / f"rec_{len(self.recordings) + 1:02d}",
                        self._device, self._model, self._chunk_s)
        rec.start()
        self.recordings.append(rec)
        self.misses, self.frames = Counter(), 0
        print("voice: recording (v to stop) - point at the page while you talk", flush=True)

    def stop(self) -> None:
        if self.recording:
            self.recordings[-1].stop()
            print("voice: stopped, transcribing the last part ...", flush=True)
            self._report_pointer()

    def _report_pointer(self) -> None:
        if not self.frames:
            return
        hit = self.frames - sum(self.misses.values())
        why = ", ".join(f"{k} {100 * n / self.frames:.0f}%" for k, n in self.misses.most_common(3))
        print(f"voice: fingertip on the page in {100 * hit / self.frames:.0f}% of frames"
              + (f" (missed: {why})" if why else ""), flush=True)

    def toggle(self) -> bool:
        """Start or stop recording; returns True if now recording."""
        if self.recording:
            self.stop()
        else:
            self.start()
        return self.recording

    def add_pointer(self, t: float, uv: Optional[Tuple[float, float]], status: str = "ok") -> None:
        """This frame's fingertip (None = not usable, `status` says why). Kept while recording."""
        if not self.recording:
            return
        self.frames += 1
        if uv is None:
            self.misses[status] += 1
        else:
            self.pointer.add(t, float(uv[0]), float(uv[1]))

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for all recordings to be transcribed. False on timeout."""
        deadline = None if timeout is None else time.time() + timeout
        for r in self.recordings:
            left = None if deadline is None else max(0.0, deadline - time.time())
            if not r.done.wait(left):
                return False
        return True

    def notes(self, after: float = 0.5) -> List[Note]:
        """Comments, in speaking order. A sentence starts a new comment where the fingertip was
        when it began (PointerLog.at). If the finger wasn't on the page then:
        * finger not seen since the previous comment -> appended to that comment;
        * finger seen since (e.g. pointed, then lifted before talking) -> new comment at the
          last fingertip position."""
        out: List[Note] = []
        for rec in self.recordings:
            for start, end, text in rec.sentences():
                uv = self.pointer.at(start, end)
                if uv is None:
                    last = self.pointer.last_before(start)
                    prev = out[-1] if out else None
                    if prev is not None and (last is None or last[0] <= prev.end + after):
                        prev.text += " " + text
                        prev.end = end
                        continue
                    uv = last[1] if last else None
                out.append(Note(len(out) + 1, text, start, end, uv))
        return out


# --------------------------------------------------------------------------- #
# PDF export
# --------------------------------------------------------------------------- #

PAGE_LONG_PT = {"a4": 841.89, "letter": 792.0}


def page_long_side(paper: str, w: int, h: int) -> float:
    """PDF page long side in points: the paper's, or for 'auto' the one closest in aspect."""
    if paper in PAGE_LONG_PT:
        return PAGE_LONG_PT[paper]
    r = max(w, h) / float(min(w, h))
    return PAGE_LONG_PT["a4"] if abs(r - 2 ** 0.5) < abs(r - 11 / 8.5) else PAGE_LONG_PT["letter"]


def export_pdf(scan_bgr: np.ndarray, notes: Sequence[Note], path: Path,
               view: Optional[np.ndarray] = None, paper: str = "a4") -> Path:
    """Write the scan as a 1-page PDF with one sticky-note comment per note.

    `view`: 3x3 map from lock-frame doc coords to the scan's current orientation
    (DocumentTracker.view_transform()); identity if None. Notes without a pointer
    position are stacked in the top-left margin, marked "(no pointer)".
    """
    import cv2
    from PIL import Image
    from pypdf import PdfReader, PdfWriter
    from pypdf.annotations import Text
    from pypdf.generic import ArrayObject, FloatObject, NameObject, TextStringObject

    path = Path(path)
    h, w = scan_bgr.shape[:2]
    long_pt = page_long_side(paper, w, h)
    dpi = max(w, h) * 72.0 / long_pt
    pw, ph = w * 72.0 / dpi, h * 72.0 / dpi

    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(scan_bgr, cv2.COLOR_BGR2RGB)).save(buf, "PDF", resolution=dpi)
    buf.seek(0)
    writer = PdfWriter(clone_from=PdfReader(buf))

    records = []
    unplaced = 0
    for n in notes:
        uv = n.uv
        text = n.text
        if uv is not None:
            uv = to_view(uv, view)
        else:
            uv = (0.03, 0.03 + 0.05 * unplaced)
            unplaced += 1
            text = "(no pointer) " + text
        x, y = float(uv[0]) * pw, (1.0 - float(uv[1])) * ph   # PDF origin is bottom-left
        s = 10.0
        annot = Text(rect=(x - s, y - s, x + s, y + s), text=text, open=False)
        annot[NameObject("/T")] = TextStringObject(f"Voice note {n.index} · {time.strftime('%H:%M:%S', time.localtime(n.start))}")
        annot[NameObject("/Name")] = NameObject("/Comment")
        annot[NameObject("/C")] = ArrayObject([FloatObject(1.0), FloatObject(0.8), FloatObject(0.0)])
        writer.add_annotation(page_number=0, annotation=annot)
        records.append(n.to_dict(uv_view=None if n.uv is None else uv))

    with open(path, "wb") as f:
        writer.write(f)
    path.with_suffix(".json").write_text(json.dumps(
        {"page_pt": [round(pw, 2), round(ph, 2)], "notes": records}, indent=1, ensure_ascii=False))
    return path


def to_view(uv: Tuple[float, float], view: Optional[np.ndarray]) -> Tuple[float, float]:
    if view is None:
        return float(uv[0]), float(uv[1])
    p = view @ np.array([uv[0], uv[1], 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])
