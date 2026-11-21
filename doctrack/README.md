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
**Scan window:** freely resizable. Its size *is* the warp size (what you see is what `s` saves), and it opens
at the paper shape. Toolbar: **Rotate 90** (`o`) turns the document's orientation clockwise. **Mirror** (`m`)
flips it left-right. Both relabel the corners, so `tr.selected.corners` stays consistent with the scan.
Keys: `q`/Esc quit, `d` debug masks, `s` save scan PNG.

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
