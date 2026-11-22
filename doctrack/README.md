# doctrack

Real-time document tracking from a desk camera. It finds sheets of paper (documents only, any colour, on any
desk) and gives their **4 corners** (TL, TR, BR, BL), like handpose landmarks. Click a document to **lock** onto
it and get a flattened scan with a fixed A4/Letter shape. Runs locally with classical OpenCV in about 11 ms per 1080p frame.

```bash
uv sync
uv run python -m doctrack.app                  # auto-finds the LogiInsight RGB camera (../logiinsight_api)
uv run python -m doctrack.app --device 0       # any webcam
uv run python -m doctrack.app --paper letter   # scan ratio: a4 (default) | letter | auto
uv run python -m pytest -q
```

**Orientation:** the document's top is assumed to face the bottom of the camera image (towards the
keyboard / the person writing). Corner labels TL/TR/BR/BL and the scan are in the *document's* frame, so the
scan is upright for the reader. Change with `--doc-up {down,up,left,right}`.

**Selecting with your hand:** point at a document with your index finger for **1 s**. A green ring fills
around your fingertip, and when it closes the document is locked. This uses MediaPipe and the team's gesture check
from `../Hand_tracker/hand_realtime.py`: index straight, other fingers curled, thumb away from the index. A hand
holding a pen doesn't select anything. After a release, take your finger off the page before pointing again.
Options: `--dwell 1.5`, `--point-dir any` (the default `down` matches Hand_tracker), `--any-finger-pose`,
`--no-hands` (click only). Hand tracking runs only while browsing. A page under a hand is treated as covered,
not gone, so it doesn't vanish while you point at it.

**Browse mode:** calm on purpose (a desk is a slow scene). A document is only shown after it has been
detected confidently for about 1/3 s. At most 3 are shown, and they are smoothed heavily. **Left click** one to lock.
**Locked mode:** only that document is tracked and it is never dropped. Each corner is handled on its own: a
corner covered by a hand or pen (hollow circle) keeps its **last position**, and that position is used for the warp.
This works for 1, 2, 3 or all 4 corners. When uncovered, it is searched for starting at its last position and going
outward. Outline: green = all corners seen, yellow = some covered, orange = all covered. The scan has a fixed size
(A4 1:√2 by default). **`r` or right click** releases it.
**Windows** are Tk (OpenCV's macOS windows can't report a user-resized size, and mixing them with Tk loses key
presses). **Scan window:** freely
resizable. The warp is recomputed every frame at the window's current size, so the scan always fills it and `s`
saves exactly that. It opens at the paper shape (A4) when a document is locked. Buttons: **Rotate 90°** (`o`)
turns the document's orientation clockwise and swaps the window shape. **Mirror** (`m`) flips left-right. Both
relabel the corners, so `tr.selected.corners` stays consistent with the scan.
Keys work in every window: **Esc**/`q` quit, `r` release, `o` rotate, `m` mirror, `s` save scan PNG, `d` debug masks, `v` voice notes on/off, `p` export the PDF with the notes.

## Voice notes → commented PDF

While a document is locked, press **`v`** to start recording, then talk and point at the page with your
**right index finger** (pointing pose, as for selecting). Press **`v`** again to stop. Each spoken **sentence**
becomes one note. It is pinned where your fingertip was on the page when the sentence began (word timestamps
from `../speech/stt.py`, faster-whisper `small.en`, run locally). The scan window shows a numbered pin per
note, your live fingertip, and a REC badge. Press **`p`** to export the flattened scan as a PDF (A4/Letter
page) with one **sticky-note comment** per sentence. These open in any PDF viewer's comments panel. If
recording is still running, `p` stops it and waits for transcription to finish first. A sentence spoken
while your finger was off the page is still exported, stacked in the top-left corner and marked "(no pointer)".

```bash
uv run python -m doctrack.app                 # v / p work out of the box (system default mic)
uv run python -m doctrack.app --camera-mic    # LogiInsight camera microphone (or --mic <index|name>)
uv run python -m doctrack.app --voice-chunk 6 # notes appear sooner (default: transcribe every ~10 s)
```

Output goes to `voice_notes/<date_time>/`: `rec_NN/session.wav` and `rec_NN/transcript.json` for each recording,
`pointer.jsonl` with fingertip samples, and `notes_<time>.pdf` with a `.json` alongside (note text + u, v
on the page). Positions are kept in document coordinates, so rotating or mirroring the scan before
exporting still puts the notes in the right place. Notes are kept after a release and cleared when a
new document is locked. Options: `--note-hand Left`, `--keep-hand-labels` (don't swap MediaPipe's
Left/Right, like Hand_tracker), `--no-voice`. On macOS, allow microphone access for your terminal.

## Library use (for another frontend)

```python
from doctrack import OpenCVDocumentDetector, DocumentTracker
tr = DocumentTracker(OpenCVDocumentDetector(), paper="a4")
tracks = tr.update(frame)            # browse: all docs | locked: [the locked doc]
tr.select(track_id) / tr.select_at(x, y) / tr.clear_selection()
tr.selected.corners                  # (4, 2) float32, TL TR BR BL
tr.selected.state                    # "tracking" | "partial" | "searching"
tr.selected.visible                  # per-corner bool: False = covered, last position kept
tr.scan(frame, size=(w, h))          # flattened doc (default size: tr.scan_size)
tr.rotate() / tr.toggle_mirror()     # fix the document orientation
tr.selected.to_dict()                # JSON-ready
```

## How it works
- `detector.py`: **no colour assumptions**. Candidate quads come from Canny edges on all Lab channels plus
  Otsu foreground masks in both polarities. Each quad is scored by: every side backed by an image edge ×
  interior dominated by one colour (the paper, whatever it is) × not-too-busy texture × paper-like aspect
  (squares penalised). Text blocks nested inside a page are merged into it, then NMS, then corners are
  snapped to the real edges at full resolution (`geometry.refine_quad`, sub-pixel line fits).
- `hands.py`: MediaPipe hand landmarks + `DwellSelector` (fingertip-on-document timer).
- `tracker.py`, browse mode: IoU matching for stable IDs, heavy One-Euro smoothing (`filters.py`),
  confidence + age gating, at most `max_docs` shown (with hysteresis so the set doesn't flicker).
- `locked.py` + `corners.py`, locked mode: each corner is located locally by fitting the two page sides that meet
  at it (strong edge, paper colour inside, background outside, straight; the outermost such boundary wins, so
  title bars or text don't count). If that fails, the corner is covered: it keeps its last position. Re-finding
  starts at the last position and goes outward: a wider side search, then patch matching in growing windows,
  with every candidate verified and confirmed over 2 frames. A move is accepted only if the page keeps a sane
  shape. If the page was pushed while partly covered, the covered corners follow the visible ones' rigid
  motion. A fully hidden page that was moved far is re-acquired by ORB matching against a lock-time snapshot
  (or the detector, for blank pages).
- Scan shape: the true aspect ratio is recovered from the perspective quad (Zhang & He), only its
  orientation is used, and the size is snapped to the paper standard and fixed for the whole lock.
