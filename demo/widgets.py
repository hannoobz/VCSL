import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from demo.pipeline import Segment, fmt_ts

BG = "#12141a"
FG = "#e6e8ee"
MUTED = "#8b93a7"
ACCENT = "#3ddc97"
ACCENT2 = "#ff8c42"
GRID = "#2a2f3d"

MAX_GRAB_SKIP = 60


class _DecodeWorker(threading.Thread):
    def __init__(self, path: str):
        super().__init__(daemon=True)
        self.path = path
        self._cond = threading.Condition()
        self._req: Optional[Tuple[int, int, int]] = None 
        self._out: Optional[Tuple[int, np.ndarray, float]] = None 
        self._out_lock = threading.Lock()
        self._stop = False
        self.frame_count = 0

    def request(self, frame: int, w: int, h: int) -> None:
        with self._cond:
            self._req = (frame, w, h)
            self._cond.notify()

    def take(self) -> Optional[Tuple[int, np.ndarray, float]]:
        with self._out_lock:
            out, self._out = self._out, None
            return out

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()

    def run(self) -> None:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            return
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cur = -1
        raw: Optional[np.ndarray] = None
        last_size = (0, 0)

        while True:
            with self._cond:
                while self._req is None and not self._stop:
                    self._cond.wait(0.2)
                if self._stop:
                    break
                target, w, h = self._req
                self._req = None

            if target != cur or raw is None:
                frame = self._read_at(cap, cur, target)
                if frame is not None:
                    raw, cur = frame, target
            elif (w, h) == last_size:
                continue 

            if raw is None:
                continue
            small, scale = _fit(raw, w, h)
            last_size = (w, h)
            with self._out_lock:
                self._out = (cur, cv2.cvtColor(small, cv2.COLOR_BGR2RGB), scale)
        cap.release()

    @staticmethod
    def _read_at(cap, cur: int, target: int) -> Optional[np.ndarray]:
        delta = target - cur
        if cur >= 0 and 0 < delta <= MAX_GRAB_SKIP:
            for _ in range(delta - 1): 
                if not cap.grab():
                    return None
            ok, frame = cap.read()
            return frame if ok else None
        cap.set(cv2.CAP_PROP_POS_FRAMES, target) 
        ok, frame = cap.read()
        return frame if ok else None


def _fit(frame: np.ndarray, cw: int, ch: int) -> Tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = min(max(1, cw) / w, max(1, ch) / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(frame, (nw, nh), interpolation=interp), scale


class VideoPane(ttk.Frame):
    def __init__(self, master, title: str, min_h: int = 200, **kw):
        super().__init__(master, **kw)
        self.title_var = tk.StringVar(value=title)
        self.info_var = tk.StringVar(value="")

        head = ttk.Frame(self)
        head.pack(fill="x")
        ttk.Label(head, textvariable=self.title_var, style="PaneTitle.TLabel").pack(side="left")
        ttk.Label(head, textvariable=self.info_var, style="Muted.TLabel").pack(side="right")

        self.canvas = tk.Canvas(self, bg="#000000", highlightthickness=1,
                                highlightbackground=GRID, height=min_h)
        self.canvas.pack(fill="both", expand=True, pady=(4, 0))
        self._img_id = self.canvas.create_image(0, 0, anchor="center")
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._photo_size: Tuple[int, int] = (0, 0)

        self._worker: Optional[_DecodeWorker] = None
        self.fps: float = 30.0
        self.frame_count: int = 0
        self._cur: int = -1
        self.dropped: int = 0

    def open(self, path: str, fps: float, frame_count: int) -> None:
        self.close()
        self.fps = fps or 30.0
        self.frame_count = frame_count
        self._cur = -1
        self.dropped = 0
        self._worker = _DecodeWorker(path)
        self._worker.start()
        self.title_var.set(Path(path).name)

    def close(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        self._cur = -1

    def show_time(self, t_sec: float, boxes: Sequence[Tuple[int, int, int, int]] = (),
                  force: bool = False) -> None:
        if self._worker is None:
            return
        target = int(round(t_sec * self.fps))
        count = self.frame_count or self._worker.frame_count
        target = max(0, min(target, count - 1) if count else target)

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self._worker.request(target, cw, ch)

        got = self._worker.take()
        if got is not None:
            idx, rgb, scale = got
            if self._cur >= 0 and idx > self._cur + 1:
                self.dropped += idx - self._cur - 1
            self._cur = idx
            self._blit(rgb, scale, boxes)
        self.info_var.set(f"{fmt_ts(t_sec)}  ·  f{max(0, self._cur)}")

    def _blit(self, rgb: np.ndarray, scale: float,
              boxes: Sequence[Tuple[int, int, int, int]]) -> None:
        if boxes:
            rgb = rgb.copy()
            for (x1, y1, x2, y2) in boxes:
                p1 = (int(x1 * scale), int(y1 * scale))
                p2 = (int(x2 * scale), int(y2 * scale))
                cv2.rectangle(rgb, p1, p2, (61, 220, 151), 2)
                cv2.putText(rgb, "PiP", (p1[0] + 4, max(14, p1[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (61, 220, 151), 1, cv2.LINE_AA)

        img = Image.fromarray(rgb)
        if self._photo is not None and self._photo_size == img.size:
            self._photo.paste(img) 
        else:
            self._photo = ImageTk.PhotoImage(img)
            self._photo_size = img.size
            self.canvas.itemconfigure(self._img_id, image=self._photo)
        self.canvas.coords(self._img_id,
                           max(1, self.canvas.winfo_width()) // 2,
                           max(1, self.canvas.winfo_height()) // 2)


class HeatmapView(ttk.Frame):
    PAD_L, PAD_R, PAD_T, PAD_B = 56, 14, 26, 34

    def __init__(self, master, on_seek: Optional[Callable[[float, float], None]] = None, **kw):
        super().__init__(master, **kw)
        self.on_seek = on_seek
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=1, highlightbackground=GRID)
        self.canvas.pack(fill="both", expand=True)

        self._sim: Optional[np.ndarray] = None
        self._segments: List[Segment] = []
        self._active: int = 0
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._base: Optional[Image.Image] = None
        self._rect: Tuple[int, int, int, int] = (0, 0, 1, 1)
        self._redraw_job = None
        self._stretch = False

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_click)

    def set_map(self, sim: np.ndarray, segments: List[Segment], active: int = 0) -> None:
        self._sim = sim
        self._segments = list(segments)
        self._active = active
        self._base = self._render_base(sim, self._stretch)
        self.redraw()

    def set_active(self, idx: int) -> None:
        self._active = idx
        self.redraw()

    def set_stretch(self, on: bool) -> None:
        self._stretch = bool(on)
        if self._sim is not None:
            self._base = self._render_base(self._sim, self._stretch)
            self.redraw()

    @staticmethod
    def _render_base(sim: np.ndarray, stretch: bool = False) -> Image.Image:
        from scripts.visualize_similarity_maps import render_similarity_map

        return render_similarity_map(sim, vmin=-1.0, vmax=1.0, per_image_normalize=stretch)

    def _plot_rect(self) -> Tuple[int, int, int, int]:
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        x0, y0 = self.PAD_L, self.PAD_T
        x1, y1 = max(x0 + 1, cw - self.PAD_R), max(y0 + 1, ch - self.PAD_B)
        return x0, y0, x1, y1

    def sec_to_px(self, q_sec: float, r_sec: float) -> Tuple[float, float]:
        x0, y0, x1, y1 = self._rect
        if self._sim is None:
            return x0, y0
        h, w = self._sim.shape
        x = x0 + (r_sec / max(1, w)) * (x1 - x0)
        y = y0 + (q_sec / max(1, h)) * (y1 - y0)
        return x, y

    def px_to_sec(self, px: float, py: float) -> Tuple[float, float]:
        x0, y0, x1, y1 = self._rect
        if self._sim is None:
            return 0.0, 0.0
        h, w = self._sim.shape
        r = (px - x0) / max(1, (x1 - x0)) * w
        q = (py - y0) / max(1, (y1 - y0)) * h
        return float(np.clip(q, 0, h - 1)), float(np.clip(r, 0, w - 1))

    def _on_resize(self, _evt=None) -> None:
        if self._redraw_job:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(60, self.redraw)

    def _on_click(self, evt) -> None:
        if self._sim is None or not self.on_seek:
            return
        x0, y0, x1, y1 = self._rect
        if not (x0 <= evt.x <= x1 and y0 <= evt.y <= y1):
            return
        q, r = self.px_to_sec(evt.x, evt.y)
        self.on_seek(q, r)

    def redraw(self) -> None:
        self._redraw_job = None
        c = self.canvas
        c.delete("layer")
        if self._sim is None or self._base is None:
            return

        self._rect = self._plot_rect()
        x0, y0, x1, y1 = self._rect
        pw, ph = max(1, x1 - x0), max(1, y1 - y0)

        img = self._base.resize((pw, ph), Image.NEAREST)
        self._photo = ImageTk.PhotoImage(img)
        c.create_image(x0, y0, anchor="nw", image=self._photo, tags="layer")
        c.create_rectangle(x0, y0, x1, y1, outline=GRID, tags="layer")

        h, w = self._sim.shape
        c.create_text((x0 + x1) // 2, y1 + 18, text="original content (s) →",
                      fill=MUTED, font=("TkDefaultFont", 8), tags="layer")
        c.create_text(14, (y0 + y1) // 2, text="reaction (s)", fill=MUTED, angle=90,
                      font=("TkDefaultFont", 8), tags="layer")

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            xs, _ = self.sec_to_px(0, frac * w)
            _, ys = self.sec_to_px(frac * h, 0)
            c.create_text(xs, y1 + 6, text=fmt_ts(frac * w), fill=MUTED, anchor="n",
                          font=("TkDefaultFont", 7), tags="layer")
            c.create_text(x0 - 8, ys, text=fmt_ts(frac * h), fill=MUTED, anchor="e",
                          font=("TkDefaultFont", 7), tags="layer")

        scale_txt = "min–max stretch" if self._stretch else "cos −1 … 1"
        c.create_text(x1, y0 - 8, text=f"inferno · {scale_txt}", fill=MUTED, anchor="se",
                      font=("TkDefaultFont", 7), tags="layer")

        for i, seg in enumerate(self._segments):
            sx, sy = self.sec_to_px(seg.q_start, seg.r_start)
            ex, ey = self.sec_to_px(seg.q_end, seg.r_end)
            active = i == self._active
            c.create_rectangle(sx, sy, ex, ey,
                               outline=ACCENT if active else "#ffffff",
                               width=2 if active else 1,
                               dash=() if active else (3, 3),
                               tags="layer")
            if active:
                c.create_line(sx, sy, ex, ey, fill=ACCENT, width=1, dash=(2, 4), tags="layer")

        if not self._segments:
            c.create_text((x0 + x1) // 2, (y0 + y1) // 2, text="no segment found",
                          fill="#ffffff", font=("TkDefaultFont", 10, "bold"), tags="layer")

        self._draw_marker_items()

    def set_marker(self, q_sec: float, r_sec: float) -> None:
        self._marker = (q_sec, r_sec)
        self._draw_marker_items()

    def _draw_marker_items(self) -> None:
        c = self.canvas
        c.delete("marker")
        pos = getattr(self, "_marker", None)
        if pos is None or self._sim is None:
            return
        x0, y0, x1, y1 = self._rect
        x, y = self.sec_to_px(*pos)
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return
        c.create_line(x0, y, x1, y, fill=ACCENT2, width=1, tags="marker")
        c.create_line(x, y0, x, y1, fill=ACCENT2, width=1, tags="marker")
        c.create_oval(x - 4, y - 4, x + 4, y + 4, outline="#ffffff", fill=ACCENT2,
                      width=1, tags="marker")
