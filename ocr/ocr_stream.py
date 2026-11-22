"""Real-time OCR on a cv2 video stream using PP-OCRv6 tiny (ONNX Runtime backend).

Capture/display runs at camera speed in the main thread; OCR runs in a worker
thread on the newest frame only (older frames are dropped), so the video never
stalls on OCR latency. Detections are drawn over the live feed.

    uv run ocr_stream.py                       # webcam 0
    uv run ocr_stream.py -s video.mp4          # file
    uv run ocr_stream.py -s rtsp://host/stream # network stream
    uv run ocr_stream.py --no-display          # headless, prints text to stdout

Keys: q / ESC quit.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from paddleocr import PaddleOCR


@dataclass
class OcrState:
    """Latest OCR output shared between worker and display loop."""

    polys: list[np.ndarray] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    stamp: float = 0.0  # time.time() when the result was produced
    latency: float = 0.0  # seconds spent in predict()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-s", "--source", default="0", help="camera index, file path or stream URL (default: 0)")
    p.add_argument("--width", type=int, default=0, help="requested capture width (cameras only)")
    p.add_argument("--height", type=int, default=0, help="requested capture height (cameras only)")
    p.add_argument("--min-score", type=float, default=0.5, help="drop recognitions below this confidence")
    p.add_argument("--max-side", type=int, default=736,
                   help="detector input is resized so the longer side is at most this (lower = faster)")
    p.add_argument("--textline-orientation", action="store_true",
                   help="enable text-line orientation classifier (slower, handles upside-down text)")
    p.add_argument("--hold", type=float, default=1.0,
                   help="seconds to keep drawing the last OCR result before it is considered stale")
    p.add_argument("--no-display", action="store_true", help="don't open a window; print recognized text only")
    return p.parse_args()


def build_ocr(args: argparse.Namespace) -> PaddleOCR:
    return PaddleOCR(
        text_detection_model_name="PP-OCRv6_tiny_det",
        text_recognition_model_name="PP-OCRv6_tiny_rec",
        engine="onnxruntime",
        # Disable the document pre-processing stages: not needed for a camera feed.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=args.textline_orientation,
        text_det_limit_side_len=args.max_side,
        text_det_limit_type="max",
        text_rec_score_thresh=args.min_score,
    )


def ocr_worker(ocr: PaddleOCR, shared: dict, state: OcrState, lock: threading.Lock,
               stop: threading.Event, new_frame: threading.Event, print_text: bool) -> None:
    last_printed: list[str] = []
    while not stop.is_set():
        if not new_frame.wait(timeout=0.1):
            continue
        new_frame.clear()
        with lock:
            frame = shared.get("frame")
        if frame is None:
            continue

        t0 = time.perf_counter()
        results = ocr.predict(frame)  # numpy BGR image, as returned by cv2
        dt = time.perf_counter() - t0

        res = results[0]
        texts = list(res["rec_texts"])
        scores = [float(s) for s in res["rec_scores"]]
        polys = [np.asarray(p, dtype=np.int32) for p in res["rec_polys"]]

        with lock:
            state.polys, state.texts, state.scores = polys, texts, scores
            state.stamp, state.latency = time.time(), dt

        if print_text and texts != last_printed:
            print(f"[{dt * 1000:5.0f} ms] " + " | ".join(texts), flush=True)
            last_printed = texts


def draw(frame: np.ndarray, state: OcrState, hold: float, fps: float) -> np.ndarray:
    out = frame.copy()
    fresh = (time.time() - state.stamp) <= hold
    if fresh:
        for poly, text, score in zip(state.polys, state.texts, state.scores):
            cv2.polylines(out, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            # cv2.putText only renders ASCII; non-ASCII glyphs become '?'.
            label = f"{text.encode('ascii', 'replace').decode()} {score:.2f}"
            x, y = int(poly[:, 0].min()), int(poly[:, 1].min())
            cv2.putText(out, label, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    hud = f"video {fps:4.1f} fps | OCR {state.latency * 1000:4.0f} ms"
    cv2.putText(out, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()

    print("Loading PP-OCRv6 tiny (first run downloads the ONNX models)...", flush=True)
    ocr = build_ocr(args)
    # Warm-up so the first live frame doesn't pay one-time init costs.
    ocr.predict(np.zeros((320, 320, 3), dtype=np.uint8))

    source: int | str = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source!r}")
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    state, lock = OcrState(), threading.Lock()
    shared: dict = {"frame": None}
    stop, new_frame = threading.Event(), threading.Event()
    worker = threading.Thread(
        target=ocr_worker,
        args=(ocr, shared, state, lock, stop, new_frame, args.no_display),
        daemon=True,
    )
    worker.start()

    fps, t_prev = 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of file / stream dropped

            with lock:
                shared["frame"] = frame  # worker always grabs the newest frame
            new_frame.set()

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6)) if fps else 1.0 / max(now - t_prev, 1e-6)
            t_prev = now

            if not args.no_display:
                with lock:
                    snapshot = OcrState(list(state.polys), list(state.texts), list(state.scores),
                                        state.stamp, state.latency)
                cv2.imshow("PP-OCRv6 tiny", draw(frame, snapshot, args.hold, fps))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()