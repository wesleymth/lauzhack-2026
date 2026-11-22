"""Tk user interface: camera window, scan window, optional debug window.

Everything is in ONE Tk app on purpose: mixing OpenCV windows with Tk on macOS
lets Tk swallow the keyboard events, and OpenCV windows can't report a
user-resized size. The app drives it from its own loop with `update()`.

Keys work in every window: r release, o rotate, m mirror, s save, d debug,
v voice note recording on/off, p export PDF with the notes, q / Esc quit. Camera window: left click = select, right click = release.
Scan window: freely resizable (its size is the warp size), Rotate / Mirror buttons.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

KEYS = {"r": "release", "o": "rotate", "m": "mirror", "s": "save", "d": "debug",
        "v": "voice", "p": "export", "q": "quit", "Escape": "quit", "f": "enroll",}
BG = "#282828"


def _photo(img_bgr: np.ndarray) -> ImageTk.PhotoImage:
    if img_bgr.ndim == 2:
        return ImageTk.PhotoImage(Image.fromarray(img_bgr))
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))


class UI:
    def __init__(self, actions: Dict[str, Callable], scan_size: Tuple[int, int] = (509, 720)) -> None:
        # keep THIS dict: the app may fill it after creating the UI
        self.actions = actions
        self.closed = False
        self.root = tk.Tk()
        self.root.title("doctrack - camera")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.cam_view = tk.Label(self.root, bd=0, bg=BG, highlightthickness=0)
        self.cam_view.pack()
        self.cam_view.bind("<Button-1>", lambda e: self._call("click", e.x, e.y))
        for b in ("<Button-2>", "<Button-3>", "<Control-Button-1>"):  # right click (mac/other)
            self.cam_view.bind(b, lambda e: self._call("release"))
        self.root.bind_all("<KeyPress>", self._on_key)
        self._cam_photo = None

        self.scan = ScanPane(tk.Toplevel(self.root), self._call)
        self.scan.top.protocol("WM_DELETE_WINDOW", self._quit)
        self.scan.set_content_size(*scan_size)

        self._export: Optional["ExportPopup"] = None
        self._export_debug: Optional["ExportDebugWindow"] = None
        self._debug_top: Optional[tk.Toplevel] = None
        self._debug_view = None
        self._debug_photo = None
        self.update()

    # ------------------------------------------------------------------ #
    def _call(self, name: str, *a) -> None:
        fn = self.actions.get(name)
        if fn is not None:
            fn(*a)

    def _on_key(self, e) -> None:
        name = KEYS.get(e.keysym) or KEYS.get(e.char)
        if name:
            self._call(name)

    def _quit(self) -> None:
        self.closed = True
        self._call("quit")

    def show_camera(self, img_bgr: np.ndarray) -> None:
        self._cam_photo = _photo(img_bgr)
        self.cam_view.configure(image=self._cam_photo)

    def show_debug(self, img: Optional[np.ndarray]) -> None:
        if img is None:
            if self._debug_top is not None:
                self._debug_top.destroy()
                self._debug_top = None
            return
        if self._debug_top is None:
            self._debug_top = tk.Toplevel(self.root)
            self._debug_top.title("doctrack - debug")
            self._debug_view = tk.Label(self._debug_top, bd=0)
            self._debug_view.pack()
        self._debug_photo = _photo(img)
        self._debug_view.configure(image=self._debug_photo)

    def show_export(self, progress) -> None:
        """Draw the export progress popup (`progress`: app.ExportProgress, or None = nothing to show)."""
        if progress is None:
            return
        if self._export is None or self._export.progress is not progress:   # a new export
            if self._export is not None:
                self._export.close()
            if self._export_debug is not None:
                self._export_debug.close()
            self._export_debug = ExportDebugWindow(self.root, progress.model)
            self._export = ExportPopup(self.root)
            self._export.progress = progress
        if self._export_debug.alive:               # closed by the user: stays closed
            self._export_debug.render(progress)
        if self._export.alive:
            self._export.render(progress)

    def update(self) -> None:
        if self.closed:
            return
        try:
            self.root.update()
        except tk.TclError:
            self.closed = True

    def close(self) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass


class ExportPopup:
    """Small always-on-top window: thumbnail of the captured page, stage text, progress bar,
    elapsed time. When the export is done it says where the PDF is and closes itself after
    CLOSE_AFTER_S; on an error it stays open with a Close button."""

    CLOSE_AFTER_S = 2.0

    def __init__(self, root: tk.Tk) -> None:
        self.top = tk.Toplevel(root)
        self.top.title("Exporting PDF")
        self.top.resizable(False, False)
        self.top.attributes("-topmost", True)
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        outer = tk.Frame(self.top, padx=14, pady=14)
        outer.pack(fill="both", expand=True)
        left = tk.Frame(outer)
        left.pack(side="left", anchor="n", padx=(0, 14))
        self._thumb = tk.Label(left, bd=1, relief="solid")     # the page frozen at p
        self._thumb.pack()
        self._thumb_caption = tk.Label(left, text="", fg="#777777", font=("TkDefaultFont", 10))
        self._thumb_caption.pack()
        self._thumb_photo = None
        frame = tk.Frame(outer)
        frame.pack(side="left", fill="both", expand=True)
        self._title = tk.Label(frame, text="Exporting PDF ...", font=("TkDefaultFont", 13, "bold"),
                               anchor="w")
        self._title.pack(fill="x")
        self._stage = tk.Label(frame, text="", anchor="w", width=46)
        self._stage.pack(fill="x", pady=(6, 4))
        self._bar = ttk.Progressbar(frame, length=360, maximum=100, mode="determinate")
        self._bar.pack(fill="x")
        self._detail = tk.Label(frame, text="", anchor="w", fg="#777777", wraplength=360, justify="left")
        self._detail.pack(fill="x", pady=(4, 0))
        self._buttons = tk.Frame(frame)
        self._buttons.pack(fill="x", pady=(10, 0))
        self._shown_done = False
        self.alive = True
        self.progress = None
        # centre on the screen
        self.top.update_idletasks()
        w, h = self.top.winfo_reqwidth(), self.top.winfo_reqheight()
        x = (self.top.winfo_screenwidth() - w) // 2
        y = (self.top.winfo_screenheight() - h) // 3
        self.top.geometry(f"+{x}+{y}")

    def render(self, p) -> None:
        if not self.alive or self._shown_done:
            return
        if self._thumb_photo is None and p.snapshot is not None:
            h = 150
            w = max(1, int(p.snapshot.shape[1] * h / p.snapshot.shape[0]))
            self._thumb_photo = _photo(cv2.resize(p.snapshot, (w, h), interpolation=cv2.INTER_AREA))
            self._thumb.configure(image=self._thumb_photo)
            self._thumb_caption.configure(text=f"captured {p.captured_at}")
        label, frac, detail = p.status()
        if p.done or p.error:
            self._finish(p)
            return
        self._stage.configure(text=label)
        self._bar.configure(value=100 * frac)
        self._detail.configure(text=detail)

    def _finish(self, p) -> None:
        self._shown_done = True
        self._bar.configure(value=100)
        if p.error:
            self._title.configure(text="Export failed")
            self._stage.configure(text=p.error)
            self._detail.configure(text="")
        else:
            path = p.download or p.path
            self._title.configure(text="PDF exported")
            self._stage.configure(text=f"{p.notes} comment(s){' + summary' if p.summary else ''}")
            self._detail.configure(text=f"Saved to {path}")
            self.top.after(int(self.CLOSE_AFTER_S * 1000), self.close)   # done: the popup goes away
            return
        ttk.Button(self._buttons, text="Close", command=self.close).pack(side="right")  # error: stays

    def close(self) -> None:
        self.alive = False
        try:
            self.top.destroy()
        except tk.TclError:
            pass


class ExportDebugWindow:
    """Terminal-style log of the export: OCR text, kept / dropped comments, and the LLM's
    answer streamed as it is written, under a header with the model's name and state."""

    COLORS = {"header": "#58a6ff", "info": "#8b949e", "ocr": "#e6edf3", "keep": "#7ee787",
              "drop": "#ff7b72", "llm": "#d2a8ff"}
    STATE = {None: "starting", "wait": "waiting for the transcription", "ocr": "idle - OCR running",
             "summary": "working - writing the summary", "pdf": "done"}

    def __init__(self, root: tk.Tk, model: str) -> None:
        self.top = tk.Toplevel(root)
        self.model = model or "no LLM (--no-llm)"
        self.top.title(f"Export debug - {self.model}")
        self.top.configure(bg="#0d1117")
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self.top.geometry("640x560+40+60")
        bar = tk.Frame(self.top, bg="#161b22", padx=12, pady=8)
        bar.pack(side="top", fill="x")
        tk.Label(bar, text=self.model, bg="#161b22", fg="#d2a8ff",
                 font=("Menlo", 16, "bold")).pack(side="left")
        self._state = tk.Label(bar, text="", bg="#161b22", fg="#8b949e", font=("Menlo", 12))
        self._state.pack(side="right")
        body = tk.Frame(self.top, bg="#0d1117")
        body.pack(fill="both", expand=True)
        scroll = tk.Scrollbar(body)
        scroll.pack(side="right", fill="y")
        self.text = tk.Text(body, bg="#0d1117", fg="#c9d1d9", font=("Menlo", 12), wrap="word",
                            bd=0, padx=12, pady=10, highlightthickness=0, yscrollcommand=scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        scroll.configure(command=self.text.yview)
        for tag, color in self.COLORS.items():
            self.text.tag_configure(tag, foreground=color)
        self.text.tag_configure("header", font=("Menlo", 12, "bold"))
        self.text.configure(state="disabled")
        self._n = 0
        self._dots = 0
        self.alive = True

    def render(self, p) -> None:
        if not self.alive:
            return
        state = "error" if p.error else "done" if p.done else self.STATE.get(p.current, p.current)
        if p.current == "summary" and not (p.done or p.error):
            self._dots = (self._dots + 1) % 24
            state += "." * (1 + self._dots // 8)                # simple activity indicator
        self._state.configure(text=state, fg="#7ee787" if p.done else "#ff7b72" if p.error else "#8b949e")
        new = p.lines[self._n:]
        if not new:
            return
        at_end = self.text.yview()[1] > 0.98
        self.text.configure(state="normal")
        for text, tag in new:
            self.text.insert("end", text, tag)
        self.text.configure(state="disabled")
        self._n += len(new)
        if at_end:
            self.text.see("end")                                 # follow, unless you scrolled up

    def close(self) -> None:
        self.alive = False
        try:
            self.top.destroy()
        except tk.TclError:
            pass


class ScanPane:
    """Scan window: toolbar + an image area that the scan is warped to fit exactly."""

    def __init__(self, top: tk.Toplevel, call: Callable) -> None:
        self.top = top
        top.title("doctrack - scan")
        top.configure(bg=BG)
        bar = tk.Frame(top, bg="#323232", padx=6, pady=4)
        bar.pack(side="top", fill="x")
        self._rotate_btn = tk.Button(bar, text="↻ Rotate 90°", command=lambda: call("rotate"))
        self._rotate_btn.pack(side="left", padx=(0, 6))
        self._mirror_btn = tk.Button(bar, text="⇆ Mirror", command=lambda: call("mirror"))
        self._mirror_btn.pack(side="left")
        self._info = tk.Label(bar, text="", fg="#cfcfcf", bg="#323232")
        self._info.pack(side="right")
        self._bar = bar
        self.view = tk.Label(top, bd=0, bg=BG, highlightthickness=0)
        self.view.pack(side="top", fill="both", expand=True)
        self._photo = None

    def size(self) -> Tuple[int, int]:
        """Current (w, h) of the image area = the size to warp the scan to."""
        return max(self.view.winfo_width(), 16), max(self.view.winfo_height(), 16)

    def set_content_size(self, w: int, h: int) -> None:
        self.top.update_idletasks()
        bar_h = max(self._bar.winfo_reqheight(), self._bar.winfo_height())
        self.top.geometry(f"{int(w)}x{int(h) + bar_h}")

    def set_state(self, enabled: bool, mirrored: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._rotate_btn.configure(state=state)
        self._mirror_btn.configure(state=state, relief="sunken" if mirrored else "raised",
                                   text="⇆ Mirror: on" if mirrored else "⇆ Mirror")
        w, h = self.size()
        self._info.configure(text=f"{w} x {h} px" if enabled else "no document locked")

    def show(self, img_bgr: np.ndarray) -> None:
        """Show an image of exactly `size()` (so the window never grows by itself)."""
        self._photo = _photo(img_bgr)
        self.view.configure(image=self._photo)
