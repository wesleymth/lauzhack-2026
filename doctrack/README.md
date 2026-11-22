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
**Locked mode, macOS (default):** the selected page is tracked with Apple's pretrained document detector
(Vision, about 5 ms per frame) run on a crop around the page. A result is used only if it is confident and has the
page's size, shape and content. With a hand on the page (from the hand tracker), the outline holds still. If the
page area hasn't changed, the outline doesn't move at all. If the page disappears, it is searched for with wider
crops and halves of the frame. See `doctrack/vision_track.py`. Use `DocumentTracker(..., lock_backend="classic")`
for the OpenCV tracker described below (used automatically where Vision isn't available).

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
**right index finger** (pointing pose: index straight, other fingers curled, in any direction). Press **`v`**
again to stop. Comments are anchored at the **word** level (word timestamps from `../speech/stt.py`,
faster-whisper `small.en`, run locally; fingertip samples are timed at frame capture). A new comment starts
at the first word spoken after your finger **stops at a new spot**: it moved more than 5% of the page width
(`--anchor-radius`) and stayed there 0.4 s (`--anchor-hold`). Jitter and small moves stay in the same
comment, and passing over the page on the way somewhere doesn't create one. Each comment is pinned at the
median position of its stop. Words spoken with no fingertip on the page stay in the **same comment as the
last fingertip position**. Words spoken before you first point form one comment without a pin.
MediaPipe's Left/Right label is unreliable from above, so it only sets priority: if the right hand isn't
pointing at the page, the other hand is used. While locked, each hand is labelled with why it's not pointing,
the status line shows `finger: on page` or the reason, and stopping a recording prints how often the
fingertip was found. The scan window shows your live fingertip
and a REC badge (the notes themselves only appear in the exported PDF). Press **`p`** to export the flattened scan as a PDF (A4/Letter
page) with one **sticky-note comment** per sentence. These open in any PDF viewer's comments panel. If
recording is still running, `p` stops it and waits for transcription to finish first. If the finger was never seen
on the page, the text is exported in the top-left corner, marked "(no pointer)".

```bash
uv run python -m doctrack.app                 # v / p work out of the box (system default mic)
uv run python -m doctrack.app --camera-mic    # LogiInsight camera microphone (or --mic <index|name>)
uv run python -m doctrack.app --voice-chunk 6 # notes appear sooner (default: transcribe every ~10 s)
```

**Clean-up at export (OCR + local LLM):** before writing the PDF, the scan is OCR'd (`../ocr`, PaddleOCR
PP-OCRv6 tiny). Parts of a note that only read the page out ("Budget 1200 CHF for hardware, that's too low"
becomes "that's too low") are removed by word matching against the OCR text, and a note with nothing new
is dropped. Then **`qwen3:0.6b`** in Ollama writes a **Summary** comment (top-right of the page, also in the
PDF's Subject). It is unloaded right after (`keep_alive: 0`). If Ollama isn't running, the export still
works, without a summary. Setup: `ollama pull qwen3:0.6b`. Options: `--llm-model <name>`, `--no-llm` (notes as
spoken, no OCR), `--ollama-url`. The spoken text stays in the `.json` (`original`), and the OCR text and
summary are saved in `notes_<time>.llm.json`.

While it exports, a **progress popup** shows the page image it captured when you pressed `p` (that exact image is the
PDF background, also saved as `notes_<time>.png`; it has the scan window's current shape, at full
resolution) and the current step (transcribing the last words, OCR, summary,
writing the PDF) with a progress bar. When it's done, it shows where the file was saved and closes itself
after 2 s (on an error it stays open). An **Export debug** window, terminal style, with the LLM's name on
top, shows the OCR text, which comments were kept or dropped, and the summary as the model writes it. Pressing `p` again while an export is running does nothing.
The PDF is also copied to **`~/Downloads`** as `voice-notes_<date_time>_<time>.pdf` (`name (1).pdf` if the name is
taken, like a browser download). Change it with `--download-dir <folder>`, or turn it off with `--download-dir ''`.
Output goes to `voice_notes/<date_time>/`: `rec_NN/session.wav` and `rec_NN/transcript.json` for each recording,
`pointer.jsonl` with fingertip samples, and `notes_<time>.pdf` with a `.json` alongside (note text + u, v
on the page). Positions are kept in document coordinates, so rotating or mirroring the scan before
exporting still puts the notes in the right place. Notes are kept after a release and cleared when a
new document is locked. Options: `--note-hand Left|any`, `--note-any-pose` (any index fingertip on the page counts), `--keep-hand-labels` (don't swap MediaPipe's
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
