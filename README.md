# lauzhack-2026: DocTrack

**Talk over a printed page, point at what you mean, get a commented PDF.**

DocTrack watches a desk through a camera. Point at a sheet of paper and it locks onto it and shows
a flat, upright scan of it (the *student view*). Press record and talk while pointing with your
index finger: every remark is transcribed and pinned to the spot you were pointing at. One key
press exports the scan as a PDF with your remarks as comments, cleaned up against the page's text
(OCR), plus a short summary written by a small local LLM. Everything runs **locally** on a laptop
CPU: no cloud service is used.

Built at LauzHack 2026 with the hackathon's LogiInsight webcam and a Logitech MX Creative keypad
as a remote control.

## How it works

```
camera ──► document tracker ──► locked page ──► flat scan (student view)
   │                                   │
   └──► hand tracker ── index fingertip ┴─► position on the page ─┐
                                                                   ├─► comments pinned per word
microphone ──► speech-to-text (word timestamps) ───────────────────┘
                                                                   │
export (p) ─► OCR of the scan ─► drop what only repeats the page ─► LLM summary ─► PDF
```

1. **Find and lock a page.** OpenCV finds sheets of paper on the desk. Point at one for 1 s (or
   click it) to lock it; it is then tracked on its own, even when hands or pens cover a corner.
2. **Record and point.** `v` starts recording. Speech is transcribed with faster-whisper, which
   gives every word a timestamp. The right index fingertip (MediaPipe) is mapped onto the page.
3. **Anchor the comments.** A new comment starts at the first word spoken after your finger stops
   at a new spot (moved more than 5% of the page width and stayed 0.4 s). Small moves and jitter
   stay in the same comment; words spoken with no finger on the page join the last comment.
4. **Export.** `p` freezes the current scan, OCRs it, removes the parts of each comment that only
   read the page out, asks the local LLM for a summary, and writes a PDF with one sticky-note
   comment per remark. The PDF is copied to `~/Downloads` and opened.

## Repository

| Folder | What it is |
|---|---|
| [`doctrack/`](doctrack/) | **The app.** Document tracking, hand pointing, voice notes, OCR/LLM clean-up, PDF export, face recognition, the teacher / student windows, and a local control server for the keypad. |
| [`speech/`](speech/) | Local speech-to-text with sentence and word timestamps (faster-whisper `small.en`). Used by doctrack; also runs on its own. |
| [`ocr/`](ocr/) | Real-time OCR on a video stream (PaddleOCR PP-OCRv6 tiny, ONNX Runtime). doctrack reuses its setup to read the scanned page. |
| [`Hand_tracker/`](Hand_tracker/) | MediaPipe hand tracking and the pointing-gesture check (index straight, other fingers curled) that doctrack uses. |
| [`logiinsight_api/`](logiinsight_api/) | Python SDK for the LogiInsight hackathon webcam: RGB / IR video, microphones, on-device AI. doctrack uses it to find the camera and mic. |
| [`ExamplePlugin/`](ExamplePlugin/) | Logi Actions plugin (C#) for the MX Creative keypad: buttons and dial that drive doctrack. |

## Quick start (macOS)

Requirements: [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com/) for the summary,
a webcam (ideally the LogiInsight camera looking down at the desk) and a microphone.

```bash
ollama pull qwen3:0.6b          # the summary model (~0.5 GB); optional
cd doctrack
uv sync                         # first run also downloads the speech model (~250 MB)
uv run python -m doctrack.app
```

Allow camera and microphone access for your terminal when macOS asks. On startup, the teacher
view (camera with tracking overlays) fills the left two thirds of the screen and the student view
(the scan) the right third.

Useful options: `--device 0` (another camera), `--faces` (face recognition), `--camera-mic`
(LogiInsight microphone), `--no-llm` (export the comments as spoken, no OCR or LLM),
`--no-open-pdf`. `uv run python -m doctrack.app --help` lists them all.

## Using it

| Key | Action |
|---|---|
| point 1 s / click | lock the document under your finger |
| `r` / right click | release the document |
| `v` | start / stop recording voice notes |
| `p` | export the commented PDF |
| `o` / `m` | rotate 90° / mirror the scan |
| `s` | save the scan as a PNG |
| `f` | enroll the face in view (face recognition) |
| `d` | show the detector's debug masks |
| `q` / Esc | quit |

While recording, the teacher view shows a blinking REC badge. During an export a progress popup
shows each step, and the terminal prints the trace: the OCR text, which comments were kept or
dropped, and the summary as the LLM writes it. When it is done, the PDF opens.

Outputs go to `doctrack/voice_notes/<date_time>/`: the audio and transcript of each recording, the
fingertip samples, and each export (`notes_<time>.pdf`, the exact scan used as `.png`, and the
notes and LLM input/output as `.json`).

## MX Creative keypad

The `ExamplePlugin` sends commands to doctrack over a local HTTP server (`POST
http://127.0.0.1:8765/<command>`, started by the app): mirror, release, rotate (dial: turn to
rotate, press to mirror), scan format (auto / A4 portrait / A4 landscape / 4:5 photo), face on/off,
audio on/off, share PDF, share PDF with notes, and stop.

```bash
cd ExamplePlugin
dotnet build src/ExamplePlugin.csproj    # builds, links and reloads it in the Logi Plugin Service
```

Then assign the **DocTrack** actions to keys in Logi Options+. The app must be running for the
buttons to do anything.

## Notes

- **Intel Macs** are supported: the doctrack environment pins Python 3.12 (uv-managed, with Tk) and
  picks mediapipe / onnxruntime versions that still ship Intel-Mac wheels.
- **Performance:** with hand tracking, document tracking, transcription and face recognition all
  on the CPU, the app can drop to 1–2 frames per second while recording; turning faces off helps.
- **Privacy:** audio, transcripts, faces and documents never leave the machine. Face recognition
  stores face embeddings locally in `doctrack/faces.npz`.

Each folder has its own README with the details.
