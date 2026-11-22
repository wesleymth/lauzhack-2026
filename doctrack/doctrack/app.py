"""Live desk view: detect + track documents, click one to get a flat scan.

    uv run python -m doctrack.app                 # auto-find the LogiInsight RGB camera
    uv run python -m doctrack.app --device 0      # any webcam by index
    uv run python -m doctrack.app --video desk.mp4

Select a document by POINTING at it with your index finger for 1 s (hand
tracking, ../Hand_tracker), or with a left click. Right click or r releases.
While locked, only that document is tracked and it is never dropped.
Scan window: freely resizable; the warp is recomputed at the window's size, so
the scan always fills it. Buttons (or keys): Rotate 90 (o), Mirror (m).
Keys (any window): Esc/q quit, r release, o rotate, m mirror, s save scan PNG, d debug.

Voice notes (locked document): v starts/stops recording. Each spoken sentence
becomes a note pinned where your RIGHT index finger points (pointing pose) on
the page when the sentence begins. p exports the scan as a PDF with one
comment per note (voice_notes/<date_time>/). See voice.py.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .detector import DetectorConfig, OpenCVDocumentDetector
from .ui import UI
from .tracker import DocumentTracker

CORNER_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # doc TL TR BR BL (BGR)
CORNER_NAMES = ["TL", "TR", "BR", "BL"]
STATE_COLORS = {"tracking": (0, 255, 0), "partial": (0, 220, 255), "covered": (0, 220, 255), "searching": (0, 120, 255)}
SDK_DIR = Path(__file__).resolve().parents[2] / "logiinsight_api"


def open_source(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"Cannot open video {args.video}")
        return cap
    try:
        sys.path.insert(0, str(SDK_DIR))
        import camera_utils as cu  # LogiInsight SDK helpers
        if args.device is None:
            idx, backend_name = cu.find_camera("rgb")
            backend = cu.backend_id(backend_name)
        else:
            idx, backend = args.device, None
        cap, info = cu.open_camera(idx, backend=backend, width=args.width, height=args.height)
        print("Camera:", cu.describe(info))
        return cap
    except ImportError:
        cap = cv2.VideoCapture(args.device or 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            sys.exit("Cannot open camera")
        return cap


def draw(frame, tracker, fps, extra=""):
    out = frame.copy()
    locked = tracker.locked
    for t in tracker.tracks:
        color = STATE_COLORS.get(t.state, (0, 255, 0)) if locked else (255, 200, 0)
        q = t.corners.astype(np.int32)
        cv2.polylines(out, [q], True, color, 3 if locked else 2, cv2.LINE_AA)
        for p, c, name, vis in zip(q, CORNER_COLORS, CORNER_NAMES, t.visible):
            p = tuple(int(v) for v in p)
            # filled = seen this frame; hollow = covered, last known position kept
            cv2.circle(out, p, 8, c, -1 if vis else 2, cv2.LINE_AA)
            cv2.putText(out, name, (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        cx, cy = q.mean(axis=0).astype(int)
        hidden = int((~t.visible).sum())
        label = f"#{t.id}" + (f" {hidden} covered" if hidden else "")
        cv2.putText(out, label, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    if locked:
        w, h = tracker.scan_size
        status = f"{fps:5.1f} fps | LOCKED #{tracker.selected.id} ({w}x{h}) | r / right-click to release{extra}"
    else:
        status = f"{fps:5.1f} fps | {len(tracker.tracks)} doc(s) | point at a document (1 s) or click"
    cv2.putText(out, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_hands(out, hands, connections, selector, t, pointer_hand=None, labels=False):
    for h in hands:
        pts = [tuple(int(v) for v in p) for p in h.landmarks]
        if labels:   # locked (voice notes): green = this fingertip places the notes
            used = h is pointer_hand
            color = (0, 255, 0) if used else (180, 180, 180)
        else:
            color = (0, 255, 0) if h.pointing else (180, 180, 180)
        for a, b in connections:
            cv2.line(out, pts[a], pts[b], color, 2, cv2.LINE_AA)
        cv2.circle(out, pts[8], 7, color, -1, cv2.LINE_AA)
        if labels:
            text = h.handedness + ": " + ("pointing" if h.pointing_any_dir else h.reason_any_dir)
            org = (pts[0][0] - 60, pts[0][1] + 30)
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    if selector is not None and selector.tip is not None:
        # progress ring around the fingertip: full circle = selected
        c = tuple(int(v) for v in selector.tip)
        cv2.circle(out, c, 28, (80, 80, 80), 3, cv2.LINE_AA)
        cv2.ellipse(out, c, (28, 28), -90, 0, 360 * selector.progress(t), (0, 255, 0), 6, cv2.LINE_AA)


def draw_notes(scan, notes, view, pointer_uv, recording):
    """Numbered pins for the notes so far, the live fingertip, and a REC badge (display only)."""
    from .voice import to_view
    out = scan.copy()
    h, w = out.shape[:2]
    for n in notes:
        if n.uv is None:
            continue
        u, v = to_view(n.uv, view)
        c = (int(u * (w - 1)), int(v * (h - 1)))
        cv2.circle(out, c, 13, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(out, c, 13, (0, 0, 0), 2, cv2.LINE_AA)
        label = str(n.index)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(out, label, (c[0] - tw // 2, c[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2, cv2.LINE_AA)
    if pointer_uv is not None:
        u, v = to_view(pointer_uv, view)
        c = (int(u * (w - 1)), int(v * (h - 1)))
        cv2.circle(out, c, 9, (0, 0, 255) if recording else (0, 200, 0), 3, cv2.LINE_AA)
    if recording:
        cv2.circle(out, (22, 22), 9, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(out, "REC", (38, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return out


class ExportProgress:
    """Export state shared between the export thread (calls `stage`) and the UI (`status`).

    Each stage has a weight in the bar and a typical duration: within a stage the bar
    advances with elapsed time up to 95% of the stage, so it keeps moving on slow steps.
    Skipped stages (no recording to wait for, no LLM) make the bar jump forward."""

    STAGES = [  # name, label, weight, typical seconds
        ("wait", "Transcribing the last words", 0.30, 12.0),
        ("ocr", "Reading the page (OCR)", 0.35, 10.0),
        ("summary", "Writing the summary", 0.30, 8.0),
        ("pdf", "Writing the PDF", 0.05, 1.0),
    ]

    def __init__(self, path: Path) -> None:
        self.path, self.download = path, None
        self.t0 = time.time()
        self.current, self.t_stage = None, self.t0
        self.done, self.error = False, None
        self.notes, self.summary = 0, False

    def stage(self, name: str) -> None:
        self.current, self.t_stage = name, time.time()

    def status(self):
        """(label, fraction 0..1, detail) for the popup."""
        names = [s[0] for s in self.STAGES]
        if self.current not in names:
            return "Starting ...", 0.0, ""
        i = names.index(self.current)
        _, label, weight, typical = self.STAGES[i]
        base = sum(s[2] for s in self.STAGES[:i])
        in_stage = min(0.95, (time.time() - self.t_stage) / typical)
        detail = (f"step {i + 1} of {len(self.STAGES)} - {time.time() - self.t_stage:.0f} s"
                  f" (total {time.time() - self.t0:.0f} s)")
        return label + " ...", base + weight * in_stage, detail


def to_downloads(pdf: Path, folder: Path) -> Path:
    """Copy the exported PDF into the Downloads folder like a browser download:
    `name.pdf`, or `name (1).pdf`, `name (2).pdf` ... if it already exists."""
    import shutil
    folder.mkdir(parents=True, exist_ok=True)
    stem = "voice-notes_" + pdf.parent.name + "_" + pdf.stem.replace("notes_", "")
    dest, i = folder / f"{stem}.pdf", 1
    while dest.exists():
        dest, i = folder / f"{stem} ({i}).pdf", i + 1
    shutil.copyfile(pdf, dest)
    return dest


def placeholder(w, h, text="point at / click a document"):
    img = np.full((h, w, 3), 40, np.uint8)
    cv2.putText(img, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=None, help="camera index (default: auto-find)")
    ap.add_argument("--video", help="read from a video file instead of a camera")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--paper", choices=["a4", "letter", "auto"], default="a4",
                    help="scan aspect ratio (orientation is detected); auto = closest of a4/letter")
    ap.add_argument("--scan-size", type=int, default=1100, help="long side of the scan in px")
    ap.add_argument("--doc-up", choices=["down", "up", "left", "right"], default="down",
                    help="where the top of the document points in the camera image "
                         "(default down: page faces the user, top towards the keyboard)")
    ap.add_argument("--display-width", type=int, default=1280, help="camera window width")
    ap.add_argument("--scan-display-height", type=int, default=720, help="scan window height")
    ap.add_argument("--no-hands", action="store_true", help="disable hand tracking (click only)")
    ap.add_argument("--dwell", type=float, default=1.0, help="seconds to point at a document to select it")
    ap.add_argument("--any-finger-pose", action="store_true",
                    help="any index fingertip on a document counts, not only the pointing gesture")
    ap.add_argument("--point-dir", choices=["any", "down", "up", "left", "right"], default="down",
                    help="required pointing direction in the image (as in Hand_tracker; default down)")
    ap.add_argument("--no-voice", action="store_true", help="disable voice notes (v / p)")
    ap.add_argument("--mic", help="microphone index or part of its name (default: system input; "
                                  "list them with: cd ../speech && uv run python stt.py --list)")
    ap.add_argument("--camera-mic", action="store_true", help="use the LogiInsight camera microphone")
    ap.add_argument("--voice-chunk", type=float, default=10.0,
                    help="seconds of audio per transcription chunk (smaller = notes appear sooner)")
    ap.add_argument("--note-hand", choices=["Right", "Left", "any"], default="Right",
                    help="hand tried first to place the voice notes (MediaPipe's label; the other "
                         "hand is used if this one isn't pointing at the page)")
    ap.add_argument("--note-any-pose", action="store_true",
                    help="any index fingertip on the page places notes, not only the pointing pose")
    ap.add_argument("--keep-hand-labels", action="store_true",
                    help="do NOT swap MediaPipe's Left/Right (as Hand_tracker --keep-labels)")
    ap.add_argument("--notes-dir", default="voice_notes", help="where recordings and PDFs go")
    ap.add_argument("--llm-model", default="qwen3:0.6b",
                    help="Ollama model for the summary. At export the page is OCR'd, note parts that "
                         "repeat it are removed, and this model writes a summary ('' = notes as spoken)")
    ap.add_argument("--no-llm", dest="llm_model", action="store_const", const="",
                    help="same as --llm-model ''")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--download-dir", default="~/Downloads",
                    help="the exported PDF is also copied here ('' = don't)")
    ap.add_argument("--faces", action="store_true",
                    help="recognise enrolled faces (f = enroll the face in view)")
    args = ap.parse_args(argv)

    cap = open_source(args)
    tracker = DocumentTracker(OpenCVDocumentDetector(DetectorConfig(doc_up=args.doc_up)), paper=args.paper,
                              scan_long_side=args.scan_size)
    state = {"scale": 1.0, "debug": False}
    hand_tracker, selector = None, None
    if not args.no_hands:
        try:
            from .hands import DwellSelector, HandTracker
            hand_tracker = HandTracker(point_dir=args.point_dir, swap_labels=not args.keep_hand_labels)
            selector = DwellSelector(dwell_s=args.dwell, require_pointing=not args.any_finger_pose)
            print(f"hand tracking on: point at a document for {args.dwell:g}s to select it")
        except Exception as e:  # mediapipe / model missing: fall back to clicking
            print(f"hand tracking unavailable ({e}); click to select")

    voice = None
    
    face_rec = None
    if args.faces:
        try:
            from .faces import FaceRecognizer
            face_rec = FaceRecognizer()
            print(f"faces on: {len(face_rec.db)} enrolled, f = enroll")
        except Exception as e:  # model download failed / old OpenCV: keep running without
            print(f"faces unavailable ({e})")
            
    if not args.no_voice:
        from .voice import VoiceNotes
        voice = VoiceNotes(Path(args.notes_dir), mic=args.mic, camera_mic=args.camera_mic,
                           chunk_s=args.voice_chunk)
        if hand_tracker is None:
            print("voice notes: no hand tracking, notes will not be pinned to a spot")

    def click(x, y):
        if tracker.locked:
            return
        t = tracker.select_at(x / state["scale"], y / state["scale"])
        if t:
            print(f"locked #{t.id}, scan {tracker.scan_size[0]}x{tracker.scan_size[1]}")
        else:
            print("no document there")

    def release():
        if tracker.locked:
            if voice is not None:
                voice.stop()             # notes are kept until another document is locked
            tracker.clear_selection()
            print("released")

    def fit_scan_window():
        """Start a newly locked document at its paper shape, --scan-display-height tall."""
        w, h = tracker.scan_size or (int(args.scan_display_height / 2 ** 0.5), args.scan_display_height)
        k = args.scan_display_height / float(h)
        ui.scan.set_content_size(int(w * k), int(h * k))

    def rotate():
        if tracker.locked:
            w, h = ui.scan.size()
            tracker.rotate(1)
            ui.scan.set_content_size(h, w)       # content turns 90 deg: swap the window shape

    def mirror():
        if tracker.locked:
            tracker.toggle_mirror()

    def save():
        if state.get("scan") is not None:
            name = f"scan_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(name, state["scan"])
            print(f"saved {name} ({state['scan'].shape[1]}x{state['scan'].shape[0]})")

    def toggle_voice():
        if voice is None:
            return
        if not tracker.locked and not voice.recording:
            print("voice: lock a document first")
            return
        try:
            voice.toggle()
        except Exception as e:  # no mic / no access: keep the app running
            print(f"voice: cannot record ({e})")

    def export():
        if voice is None:
            return
        frame = state.get("frame")
        if not tracker.locked or frame is None:
            print("export: lock a document first")
            return
        running = state.get("export")
        if running is not None and not (running.done or running.error):
            print("export: already running")
            return
        scan = tracker.scan(frame)                     # full paper-size scan, current orientation
        view = tracker.view_transform()
        voice.stop()
        if not voice.recordings:
            print("export: no voice notes yet (v to record)")
        out_dir = voice.session_dir or Path(args.notes_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"notes_{time.strftime('%H-%M-%S')}.pdf"

        progress = ExportProgress(path)
        state["export"] = progress

        def run():
            from .voice import export_pdf
            try:
                if voice.busy:
                    progress.stage("wait")
                    print("export: waiting for transcription to finish ...", flush=True)
                voice.wait()
                notes = voice.notes()
                summary = None
                if args.llm_model:
                    try:
                        from .refine import refine_notes
                        summary, notes = refine_notes(scan, notes, view, model=args.llm_model,
                                                      url=args.ollama_url, log_path=path.with_suffix(".llm.json"),
                                                      on_stage=progress.stage)
                    except Exception as e:  # OCR failed...: plain notes
                        print(f"refine: skipped ({type(e).__name__}: {e}); exporting the notes as spoken",
                              flush=True)
                progress.stage("pdf")
                export_pdf(scan, notes, path, view=view, paper=args.paper, summary=summary)
                placed = sum(n.uv is not None for n in notes)
                print(f"exported {path} ({len(notes)} notes, {placed} pinned)", flush=True)
                if args.download_dir:
                    try:
                        progress.download = to_downloads(path, Path(args.download_dir).expanduser())
                        print(f"saved to {progress.download}", flush=True)
                    except OSError as e:
                        print(f"could not copy the PDF to {args.download_dir}: {e}", flush=True)
                progress.notes, progress.summary = len(notes), bool(summary)
                progress.done = True
            except Exception as e:  # shown in the popup instead of dying silently in the thread
                progress.error = f"{type(e).__name__}: {e}"
                print(f"export failed: {progress.error}", flush=True)
        threading.Thread(target=run, daemon=True).start()

    def toggle_debug():
        state["debug"] = not state["debug"]
        if not state["debug"]:
            ui.show_debug(None)

    def enroll():
        if face_rec is None:
            return
        frame = state.get("frame")
        if frame is None:
            return
        frame = frame.copy()                 # the dialog blocks the loop: keep this frame
        from tkinter import simpledialog
        name = simpledialog.askstring("Enroll face", "Name?", parent=ui.root)
        if not name:
            return
        print(f"enrolled {name}" if face_rec.enroll(name.strip(), frame) else "enroll: no face seen")

    actions = dict(click=click, release=release, rotate=rotate, mirror=mirror, save=save,
                   debug=toggle_debug, voice=toggle_voice, export=export, enroll=enroll,
                   quit=lambda: state.update(quit=True))
    ui = UI(actions)
    fit_scan_window()
    was_locked = False

    fps, last = 0.0, time.perf_counter()
    try:
        while not state.get("quit") and not ui.closed:
            ok, frame = cap.read()
            if not ok or frame is None:
                if args.video:
                    break
                ui.update()
                continue
            now = time.perf_counter()
            state["frame"] = frame
            hands = []
            # select by pointing; "covered" while locked; voice notes: the pointing fingertip
            if hand_tracker is not None:
                hands = hand_tracker.detect(frame, now)
            tracker.update(frame, now, occluders=[h.hull for h in hands])
            if selector is not None:
                tid = selector.update(tracker, hands, now)
                if tid is not None:
                    print(f"locked #{tid} by pointing, scan {tracker.scan_size[0]}x{tracker.scan_size[1]}")
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now

            if tracker.locked and not was_locked and voice is not None:
                voice.reset()                      # new document: new set of notes
            pointer_uv, pointer_hand, pointer_status = None, None, ""
            if voice is not None and tracker.locked:
                from .voice import pick_pointer
                pointer_uv, pointer_hand, pointer_status = pick_pointer(
                    hands, tracker.to_doc, args.note_hand, args.note_any_pose)
                voice.add_pointer(time.time(), pointer_uv, pointer_status)

            extra = ""
            if voice is not None and tracker.locked:
                n = len(voice.notes())
                extra = (" | REC (v stop)" if voice.recording else " | v record") + f" | {n} note(s), p export"
                extra += " | finger: " + ("on page" if pointer_uv is not None else pointer_status)
            vis = draw(frame, tracker, fps, extra)
            if hands:
                draw_hands(vis, hands, hand_tracker.connections, selector, now,
                           pointer_hand=pointer_hand, labels=tracker.locked)
            faces = []
            if face_rec is not None:
                from .faces import draw as draw_faces
                faces = face_rec.identify(frame)
                draw_faces(vis, faces)
            state["scale"] = min(1.0, args.display_width / float(vis.shape[1]))
            if state["scale"] < 1.0:
                vis = cv2.resize(vis, None, fx=state["scale"], fy=state["scale"],
                                 interpolation=cv2.INTER_AREA)
            ui.show_camera(vis)

            if tracker.locked and not was_locked:
                fit_scan_window()                  # new document: start at its paper shape
            was_locked = tracker.locked
            # the warp is recomputed at the window's current size: it always fills the window
            ww, wh = ui.scan.size()
            scan = tracker.scan(frame, size=(ww, wh))
            state["scan"] = scan
            if scan is not None and voice is not None:
                scan = draw_notes(scan, voice.notes(), tracker.view_transform(), pointer_uv, voice.recording)
            if scan is not None and face_rec is not None:
                from .faces import draw_thumbs
                scan = scan.copy()                 # keep state["scan"] clean: the s key saves it
                draw_thumbs(scan, frame, faces, size=0.2)
            ui.scan.show(scan if scan is not None else placeholder(ww, wh))
            ui.scan.set_state(tracker.locked, tracker.mirrored)

            if state["debug"] and tracker.detector.debug:
                ui.show_debug(np.hstack(list(tracker.detector.debug.values())))
            ui.show_export(state.get("export"))   # progress popup (Tk: main thread only)
            ui.update()                            # handles clicks, buttons and keys
    except KeyboardInterrupt:
        pass
    finally:
        if voice is not None and voice.recordings:
            voice.stop()
            print("voice: saving recordings ...")
            voice.wait(timeout=60)
        cap.release()
        if hand_tracker is not None:
            hand_tracker.close()
        ui.close()


if __name__ == "__main__":
    main()
