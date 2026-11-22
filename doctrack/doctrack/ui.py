"""Tk user interface: camera window, scan window, optional debug window.

Everything is in ONE Tk app on purpose: mixing OpenCV windows with Tk on macOS
lets Tk swallow the keyboard events, and OpenCV windows can't report a
user-resized size. The app drives it from its own loop with `update()`.

Keys work in every window: r release, o rotate, m mirror, s save, d debug,
v voice note recording on/off, p export PDF with the notes, q / Esc quit. Camera window: left click = select, right click = release.
Scan window: freely resizable (its size is the warp size), Rotate / Mirror buttons.
"""

import tkinter as tk
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
