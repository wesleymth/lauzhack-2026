"""Make the voice notes complementary to the page, with a summary, before the PDF export.

1. OCR the flattened scan with the team's PaddleOCR setup (../ocr/ocr_stream.py,
   PP-OCRv6 tiny on ONNX Runtime).
2. Drop what only repeats the page: each note is split into clauses, and a clause
   whose content words are (almost) all in the OCR text is removed ("Budget 1200
   CHF for hardware, that's too low" -> "that's too low"). This is done with plain
   word matching: a 1.7B model can't make that call reliably (tested: it either
   kept everything or dropped everything). A note left with nothing is dropped.
3. A tiny local LLM (Ollama, default qwen3:0.6b, thinking off) writes only the
   summary of the document and the notes. (Asked to tidy the notes too, qwen3:1.7b
   returned them unchanged, so that was dropped: it cost time for nothing.)

Everything runs locally. The model is unloaded right after the call
(keep_alive 0) to give the RAM back.
"""

import json
import re
import sys
import time
import urllib.request
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .voice import Note, to_view

OCR_DIR = Path(__file__).resolve().parents[2] / "ocr"
MAX_PAGE_CHARS = 6000       # keeps the prompt well inside num_ctx on a slow CPU

SYSTEM = """You summarize a printed document and the comments a reader made on it.
Write 2 to 4 sentences of plain text: what the document is about, then the main points of the reader's comments.
Use only the information given. No title, no list, no preamble."""

# Words that carry no content of their own (and speech filler), for the repeat check.
STOP = set("""a an the and or but so if then than that this these those there here it its it's is are was were be
been being am do does did done have has had i i'm me my we our us you your he she they them their what which who
whom when where why how all any some no not only just very too also of in on at to for from by with about as into
over under up down out off again more most such own same can could should would will shall may might must
let let's um uh oh okay ok yeah yes like well really actually basically think mean say said see look know want
going gonna get got one's s t""".split())
NUMBERS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
           "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
           "twenty": "20", "thirty": "30", "fifty": "50", "hundred": "100", "thousand": "1000"}


def content_words(text: str) -> List[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace(",", "")):
        w = NUMBERS.get(w, w)
        if w in STOP or (len(w) < 2 and not w.isdigit()):
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]                                   # crude plural stemming: risks -> risk
        out.append(w)
    return out


def strip_repeats(said: str, page: str, threshold: float = 0.75) -> str:
    """`said` without the clauses that only restate `page` (or are pure filler)."""
    vocab = set(content_words(page))
    kept = []
    for clause in re.split(r"(?<=[.,;:!?])\s+", said.strip()):
        words = content_words(clause)
        if not words:
            continue                                     # filler: "So, um,"
        if vocab and sum(w in vocab for w in words) / len(words) >= threshold:
            continue                                     # reads the page out
        kept.append(clause)
    text = " ".join(kept).strip(" ,;:")
    return text[:1].upper() + text[1:] if text else ""


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #

def ocr_page(scan_bgr: np.ndarray, min_score: float = 0.5) -> List[Tuple[str, np.ndarray]]:
    """(text, box (4, 2) in normalised page coords) per line, in reading order."""
    if str(OCR_DIR) not in sys.path:
        sys.path.insert(0, str(OCR_DIR))
    import ocr_stream

    h, w = scan_bgr.shape[:2]
    ocr = ocr_stream.build_ocr(SimpleNamespace(textline_orientation=False, max_side=max(h, w),
                                               min_score=min_score))
    res = ocr.predict(scan_bgr)[0]
    lines = []
    for text, poly in zip(res["rec_texts"], res["rec_polys"]):
        text = str(text).strip()
        if text:
            lines.append((text, np.asarray(poly, np.float32).reshape(-1, 2) / [w, h]))
    if not lines:
        return []
    # reading order: rows (by centre y, in units of the median line height), then x
    lh = float(np.median([np.ptp(b[:, 1]) for _, b in lines])) or 0.02
    return sorted(lines, key=lambda l: (round(l[1][:, 1].mean() / lh), l[1][:, 0].min()))


def page_text(lines: Sequence[Tuple[str, np.ndarray]]) -> str:
    """Lines joined, with a line break between rows."""
    out, last_y = [], None
    for text, box in lines:
        y = box[:, 1].mean()
        h = max(float(np.ptp(box[:, 1])), 1e-3)
        if last_y is not None:
            out.append("\n" if y - last_y > 0.5 * h else " ")
        out.append(text)
        last_y = y
    return "".join(out)


def text_near(lines, uv, k: int = 2, max_dist: float = 0.12) -> str:
    """The k OCR lines closest to a pin (distance to the line's box)."""
    if uv is None or not lines:
        return ""
    p = np.asarray(uv, np.float32)
    dists = []
    for text, box in lines:
        lo, hi = box.min(axis=0), box.max(axis=0)
        dists.append(float(np.linalg.norm(np.maximum(0, np.maximum(lo - p, p - hi)))))
    order = [i for i in np.argsort(dists)[:k] if dists[i] <= max_dist]
    return " / ".join(lines[i][0] for i in sorted(order, key=lambda i: dists[i]))


# --------------------------------------------------------------------------- #
# LLM (Ollama)
# --------------------------------------------------------------------------- #

def ask_ollama(page: str, comments: List[dict], model: str, url: str, timeout: float = 300) -> str:
    """The summary, as plain text."""
    said = "\n".join(f"- ({c['near']}) {c['said']}" if c["near"] else f"- {c['said']}"
                     for c in comments) or "(none)"
    user = ("DOCUMENT TEXT:\n" + (page[:MAX_PAGE_CHARS] or "(no text found)")
            + "\n\nREADER'S COMMENTS (in brackets: the document text they point at):\n" + said)
    body = {
        "model": model, "stream": False, "think": False, "keep_alive": 0,
        "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": 200},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
    }
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = json.loads(r.read())["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def refine_notes(scan_bgr: np.ndarray, notes: Sequence[Note], view: Optional[np.ndarray] = None,
                 model: str = "qwen3:0.6b", url: str = "http://localhost:11434",
                 log_path: Optional[Path] = None,
                 on_stage: Optional[Callable[[str], None]] = None) -> Tuple[Optional[str], List[Note]]:
    """(summary, notes): notes reduced to what the page doesn't already say; the spoken text
    is kept in Note.original. Raises if OCR fails; an LLM failure only loses the summary.
    `on_stage("ocr" | "summary")` is called as each step starts (progress display)."""
    stage = on_stage or (lambda name: None)
    stage("ocr")
    t0 = time.time()
    lines = ocr_page(scan_bgr)
    page = page_text(lines)
    print(f"refine: OCR {len(lines)} lines in {time.time() - t0:.1f}s", flush=True)

    kept = []
    for n in notes:
        text = strip_repeats(n.text, page)
        if text:
            kept.append(replace(n, text=text, original=n.text))
    if len(kept) < len(notes):
        print(f"refine: {len(notes) - len(kept)} note(s) only repeated the page, dropped", flush=True)

    comments = [{"id": n.index, "near": text_near(lines, None if n.uv is None else to_view(n.uv, view)),
                 "said": n.text} for n in kept]
    summary = None
    if model:
        stage("summary")
        t1 = time.time()
        print(f"refine: asking {model} for a summary ...", flush=True)
        try:
            summary = ask_ollama(page, comments, model, url) or None
            print(f"refine: {model} answered in {time.time() - t1:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001  Ollama down / model missing
            print(f"refine: {model} failed ({type(e).__name__}: {e}); no summary", flush=True)

    if log_path is not None:
        Path(log_path).write_text(json.dumps(
            {"model": model, "page_text": page, "comments": comments, "summary": summary},
            indent=1, ensure_ascii=False))
    return summary, kept
