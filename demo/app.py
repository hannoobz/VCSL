import argparse
import queue
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""): 
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from demo.pipeline import (
    CROPPED,
    STAGES,
    UNCROPPED,
    DemoConfig,
    DemoResult,
    Pipeline,
    Segment,
    export_result,
    fmt_ts,
)
from demo.repo_env import REPO_ROOT, abspath
from demo.widgets import ACCENT, ACCENT2, BG, FG, GRID, MUTED, HeatmapView, VideoPane

CARD = "#1a1e28"
VIDEO_EXTS = [("Video", "*.mp4 *.mkv *.mov *.avi *.webm *.flv *.m4v"), ("All files", "*.*")]
TICK_MS = 33


def install_style(root: tk.Tk) -> None:
    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except tk.TclError:
        pass
    root.configure(bg=BG)
    st.configure(".", background=BG, foreground=FG, fieldbackground=CARD, bordercolor=GRID)
    st.configure("TFrame", background=BG)
    st.configure("Card.TFrame", background=CARD)
    st.configure("TLabel", background=BG, foreground=FG)
    st.configure("Card.TLabel", background=CARD, foreground=FG)
    st.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("TkDefaultFont", 8))
    st.configure("CardMuted.TLabel", background=CARD, foreground=MUTED, font=("TkDefaultFont", 8))
    st.configure("H1.TLabel", background=BG, foreground=FG, font=("TkDefaultFont", 16, "bold"))
    st.configure("H2.TLabel", background=BG, foreground=FG, font=("TkDefaultFont", 11, "bold"))
    st.configure("PaneTitle.TLabel", background=BG, foreground=ACCENT,
                 font=("TkDefaultFont", 9, "bold"))
    st.configure("Value.TLabel", background=CARD, foreground=ACCENT,
                 font=("TkDefaultFont", 10, "bold"))
    st.configure("TButton", background="#252b38", foreground=FG, borderwidth=0, padding=6)
    st.map("TButton", background=[("active", "#313949"), ("disabled", "#1d2029")],
           foreground=[("disabled", MUTED)])
    st.configure("Accent.TButton", background=ACCENT, foreground="#06210f",
                 font=("TkDefaultFont", 10, "bold"), padding=8)
    st.map("Accent.TButton", background=[("active", "#57e8a9"), ("disabled", "#2a3b33")])
    st.configure("TCheckbutton", background=BG, foreground=FG)
    st.map("TCheckbutton", background=[("active", BG)])
    st.configure("Card.TCheckbutton", background=CARD, foreground=MUTED)
    st.map("Card.TCheckbutton", background=[("active", CARD)])
    st.configure("TEntry", fieldbackground=CARD, foreground=FG, insertcolor=FG)
    st.configure("TCombobox", fieldbackground=CARD, background=CARD, foreground=FG,
                 arrowcolor=FG, selectbackground=CARD, selectforeground=FG)
    st.map("TCombobox",
           fieldbackground=[("readonly", CARD), ("disabled", "#1d2029")],
           foreground=[("readonly", FG), ("disabled", MUTED)],
           selectbackground=[("readonly", CARD)],
           selectforeground=[("readonly", FG)],
           arrowcolor=[("disabled", MUTED)])
    st.configure("Card.TCombobox", fieldbackground=BG, background=BG)
    root.option_add("*TCombobox*Listbox.background", CARD)
    root.option_add("*TCombobox*Listbox.foreground", FG)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", "#06210f")
    st.configure("TSpinbox", fieldbackground=CARD, background="#252b38", foreground=FG,
                 arrowcolor=FG)
    st.configure("TLabelframe", background=BG, foreground=MUTED, bordercolor=GRID)
    st.configure("TLabelframe.Label", background=BG, foreground=MUTED)
    st.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=CARD, bordercolor=GRID)
    st.configure("TScale", background=BG, troughcolor="#2f3646")
    st.configure("Horizontal.TScale", background=ACCENT, troughcolor="#2f3646")


class SetupScreen(ttk.Frame):
    def __init__(self, master, app: "DemoApp"):
        super().__init__(master, padding=24)
        self.app = app
        cfg = app.cfg

        self.query = tk.StringVar(value=cfg.query_path)
        self.ref = tk.StringVar(value=cfg.ref_path)
        self.yolo = tk.StringVar(value=cfg.yolo_weights or self._guess_yolo())
        self.backbone = tk.StringVar(value="ISC21" if cfg.backbone == "isc" else "DINO")
        self.method = tk.StringVar(value=cfg.method)
        self.conf = tk.DoubleVar(value=cfg.pip_conf)
        self.device = tk.StringVar(value=cfg.device)
        self.spd_weights = tk.StringVar(value=cfg.spd_weights)
        self.batch = tk.IntVar(value=cfg.batch_size)
        self.min_sim = tk.DoubleVar(value=cfg.min_sim)
        self.min_length = tk.IntVar(value=cfg.min_length)
        self.crop_mode = tk.StringVar(value=cfg.crop_mode)
        self.spd_conf = tk.DoubleVar(value=cfg.spd_conf)

        ttk.Label(self, text="demo", style="H1.TLabel").pack(anchor="w")
        ttk.Label(
            self,
            text="Runs the full pipeline live on two files you pick — YOLO PiP detection on the "
                 "reaction video, crop, ISC/DINO features at 1 fps, similarity map, temporal "
                 "alignment — uncropped and cropped, so you can compare them side by side. ",
            style="Muted.TLabel", wraplength=760, justify="left",
        ).pack(anchor="w", pady=(4, 18))

        files = ttk.LabelFrame(self, text=" input videos ", padding=14)
        files.pack(fill="x")
        files.columnconfigure(1, weight=1)
        self._file_row(files, 0, "Reaction video", self.query, "reaction")
        self._file_row(files, 1, "Original content", self.ref, "original")

        opts = ttk.LabelFrame(self, text=" pipeline ", padding=14)
        opts.pack(fill="x", pady=(14, 0))
        opts.columnconfigure(1, weight=1)
        opts.columnconfigure(3, weight=1)

        ttk.Label(opts, text="Feature extractor").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Combobox(opts, textvariable=self.backbone, values=["ISC21", "DINO"],
                     state="readonly", width=16).grid(row=0, column=1, sticky="w", padx=(10, 24))

        ttk.Label(opts, text="Alignment method").grid(row=0, column=2, sticky="w", pady=6)
        mbox = ttk.Combobox(opts, textvariable=self.method, values=["TN", "SPD"],
                            state="readonly", width=16)
        mbox.grid(row=0, column=3, sticky="w", padx=(10, 0))
        mbox.bind("<<ComboboxSelected>>", lambda _e: self._sync_spd_state())

        ttk.Label(opts, text="PiP detector conf").grid(row=1, column=0, sticky="w", pady=6)
        row = ttk.Frame(opts)
        row.grid(row=1, column=1, columnspan=3, sticky="ew", padx=(10, 0))
        row.columnconfigure(0, weight=1)
        ttk.Scale(row, from_=0.05, to=0.95, variable=self.conf, orient="horizontal",
                  command=lambda _v: self._conf_label()).grid(row=0, column=0, sticky="ew")
        self.conf_lbl = ttk.Label(row, text="", width=6)
        self.conf_lbl.grid(row=0, column=1, padx=(10, 0))
        self._conf_label()

        self._file_row(opts, 2, "YOLO PiP weights", self.yolo, "weights", span=3)
        ttk.Label(opts, text="confidence of the PiP box detector, applied when cropping "
                             "reaction frames — unrelated to SPD's threshold below",
                  style="Muted.TLabel").grid(row=3, column=1, columnspan=3, sticky="w",
                                             padx=(10, 0))

        adv = ttk.LabelFrame(self, text=" advanced ", padding=14)
        adv.pack(fill="x", pady=(14, 0))
        for i in range(6):
            adv.columnconfigure(i, weight=1 if i % 2 else 0)
        ttk.Label(adv, text="Device").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(adv, textvariable=self.device, values=["auto", "cpu", "cuda:0"],
                     state="readonly", width=10).grid(row=0, column=1, sticky="w", padx=(8, 20))
        ttk.Label(adv, text="Batch").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(adv, from_=1, to=256, textvariable=self.batch, width=6).grid(
            row=0, column=3, sticky="w", padx=(8, 20))
        ttk.Label(adv, text="min_sim").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(adv, from_=0.0, to=1.0, increment=0.05, textvariable=self.min_sim,
                    width=6).grid(row=0, column=5, sticky="w", padx=(8, 0))

        ttk.Label(adv, text="crop mode").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Combobox(adv, textvariable=self.crop_mode, state="readonly", width=16,
                     values=["pip_only", "main_minus_pip"]).grid(row=2, column=1, sticky="w",
                                                                 padx=(8, 20))

        ttk.Label(adv, text="min_length").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Spinbox(adv, from_=1, to=120, textvariable=self.min_length, width=6).grid(
            row=1, column=1, sticky="w", padx=(8, 20))
        self.spd_lbl = ttk.Label(adv, text="SPD .pt")
        self.spd_lbl.grid(row=1, column=2, sticky="w")
        self.spd_entry = ttk.Entry(adv, textvariable=self.spd_weights)
        self.spd_entry.grid(row=1, column=3, columnspan=2, sticky="ew", padx=(8, 8))
        self.spd_btn = ttk.Button(adv, text="…", width=3, command=self._pick_spd)
        self.spd_btn.grid(row=1, column=5, sticky="w")

        self.spd_conf_lbl = ttk.Label(adv, text="SPD conf")
        self.spd_conf_lbl.grid(row=3, column=0, sticky="w", pady=4)
        self.spd_conf_box = ttk.Spinbox(adv, from_=0.01, to=0.99, increment=0.05,
                                        textvariable=self.spd_conf, width=6)
        self.spd_conf_box.grid(row=3, column=1, sticky="w", padx=(8, 20))
        ttk.Label(adv, text="segment-box confidence for SPD on the similarity map "
                            "(repo default 0.1) — not the PiP threshold",
                  style="Muted.TLabel").grid(row=3, column=2, columnspan=4, sticky="w")

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(20, 0))
        self.hint = ttk.Label(bar, text="", style="Muted.TLabel")
        self.hint.pack(side="left")
        ttk.Button(bar, text="Run pipeline  ▸", style="Accent.TButton",
                   command=self.run).pack(side="right")

        self._sync_spd_state() 

    def _guess_yolo(self) -> str:
        for cand in ("pip_yolo.pt", "yolo_pip.pt", "best.pt", "pip.pt"):
            p = abspath(cand)
            if p.exists():
                return str(p)
        return ""

    def _file_row(self, parent, row: int, label: str, var: tk.StringVar, kind: str, span: int = 1):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6)
        ent = ttk.Entry(parent, textvariable=var)
        ent.grid(row=row, column=1, columnspan=span, sticky="ew", padx=(10, 8))
        ttk.Button(parent, text="Browse…", command=lambda: self._pick(var, kind)).grid(
            row=row, column=1 + span, sticky="e")

    def _pick(self, var: tk.StringVar, kind: str) -> None:
        if kind == "weights":
            path = filedialog.askopenfilename(title="Pretrained YOLO PiP detector (.pt)",
                                              filetypes=[("PyTorch weights", "*.pt"),
                                                         ("All files", "*.*")])
        else:
            path = filedialog.askopenfilename(
                title=f"Pick the {kind} video", filetypes=VIDEO_EXTS)
        if path:
            var.set(path)

    def _pick_spd(self) -> None:
        path = filedialog.askopenfilename(title="SPD checkpoint (.pt)",
                                          filetypes=[("PyTorch weights", "*.pt")])
        if path:
            self.spd_weights.set(path)

    def _conf_label(self) -> None:
        self.conf_lbl.configure(text=f"{self.conf.get():.2f}")

    def _sync_spd_state(self) -> None:
        state = "normal" if self.method.get() == "SPD" else "disabled"
        for w in (self.spd_entry, self.spd_btn, self.spd_conf_box):
            w.configure(state=state)
        default = abspath(f"./{'isc' if self.backbone.get() == 'ISC21' else 'dino'}.pt")
        self.spd_lbl.configure(text="SPD .pt" if state == "normal" else "SPD .pt (n/a)")
        self.spd_conf_lbl.configure(text="SPD conf" if state == "normal" else "SPD conf (n/a)")
        if state == "normal" and not self.spd_weights.get():
            self.hint.configure(text=f"SPD defaults to {default}")

    def run(self) -> None:
        cfg = self.app.cfg
        cfg.query_path = self.query.get().strip()
        cfg.ref_path = self.ref.get().strip()
        cfg.yolo_weights = self.yolo.get().strip()
        cfg.backbone = "isc" if self.backbone.get() == "ISC21" else "dino"
        cfg.method = self.method.get()
        cfg.pip_conf = round(float(self.conf.get()), 2)
        cfg.device = self.device.get()
        cfg.spd_weights = self.spd_weights.get().strip()
        cfg.batch_size = int(self.batch.get())
        cfg.min_sim = float(self.min_sim.get())
        cfg.min_length = int(self.min_length.get())
        cfg.crop_mode = self.crop_mode.get()
        cfg.spd_conf = float(self.spd_conf.get())

        problems = []
        if not cfg.query_path or not Path(cfg.query_path).exists():
            problems.append("pick a reaction video")
        if not cfg.ref_path or not Path(cfg.ref_path).exists():
            problems.append("pick the original content video")
        if not cfg.yolo_weights or not Path(cfg.yolo_weights).exists():
            problems.append("pick the pretrained YOLO PiP weights (.pt)")
        if cfg.method == "SPD" and not cfg.resolved_spd_weights().exists():
            problems.append(f"SPD checkpoint not found: {cfg.resolved_spd_weights()}")
        if problems:
            messagebox.showwarning("Missing input", "· " + "\n· ".join(problems))
            return
        self.app.start_run()


class ProcessingScreen(ttk.Frame):
    def __init__(self, master, app: "DemoApp"):
        super().__init__(master, padding=24)
        self.app = app
        self.rows = {}

        ttk.Label(self, text="Running pipeline", style="H1.TLabel").pack(anchor="w")
        self.sub = ttk.Label(self, text="", style="Muted.TLabel")
        self.sub.pack(anchor="w", pady=(4, 16))

        card = ttk.Frame(self, style="Card.TFrame", padding=16)
        card.pack(fill="x")
        card.columnconfigure(1, weight=1)
        for i, (key, title) in enumerate(STAGES):
            dot = ttk.Label(card, text="○", style="Card.TLabel", width=2)
            dot.grid(row=i, column=0, sticky="w", pady=3)
            name = ttk.Label(card, text=title, style="Card.TLabel")
            name.grid(row=i, column=1, sticky="w")
            detail = ttk.Label(card, text="", style="CardMuted.TLabel")
            detail.grid(row=i, column=2, sticky="e")
            self.rows[key] = (dot, name, detail)

        self.bar = ttk.Progressbar(self, mode="determinate", maximum=1000)
        self.bar.pack(fill="x", pady=(16, 6))
        self.pct = ttk.Label(self, text="0%", style="Muted.TLabel")
        self.pct.pack(anchor="e")

        ttk.Label(self, text="log", style="Muted.TLabel").pack(anchor="w", pady=(10, 2))
        wrap = ttk.Frame(self)
        wrap.pack(fill="both", expand=True)
        self.log = tk.Text(wrap, bg=CARD, fg=MUTED, insertbackground=FG, height=12,
                           relief="flat", wrap="word", font=("TkFixedFont", 9))
        sb = ttk.Scrollbar(wrap, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(12, 0))
        ttk.Button(bar, text="Cancel", command=self.app.cancel_run).pack(side="right")

    def reset(self, subtitle: str) -> None:
        self.sub.configure(text=subtitle)
        for dot, name, detail in self.rows.values():
            dot.configure(text="○", foreground=MUTED)
            name.configure(foreground=MUTED)
            detail.configure(text="")
        self.bar["value"] = 0
        self.pct.configure(text="0%")
        self.log.delete("1.0", "end")

    def on_stage(self, key: str, state: str, detail: str, frac) -> None:
        if key not in self.rows:
            return
        dot, name, dlabel = self.rows[key]
        if state == "running":
            dot.configure(text="◉", foreground=ACCENT2)
            name.configure(foreground=FG)
        elif state == "done":
            dot.configure(text="●", foreground=ACCENT)
            name.configure(foreground=FG)
        if detail:
            dlabel.configure(text=detail)

        idx = [k for k, _ in STAGES].index(key)
        f = frac if isinstance(frac, (int, float)) else (1.0 if state == "done" else 0.35)
        overall = (idx + min(1.0, max(0.0, f))) / len(STAGES)
        self.bar["value"] = overall * 1000
        self.pct.configure(text=f"{overall * 100:.0f}%")

    def on_log(self, msg: str) -> None:
        self.log.insert("end", msg + "\n")
        self.log.see("end")


class ResultScreen(ttk.Frame):
    def __init__(self, master, app: "DemoApp"):
        super().__init__(master, padding=14)
        self.app = app
        self.result: DemoResult | None = None
        self.seg_index = 0
        self.playing = False
        self._pos = 0.0
        self._last_tick: float | None = None
        self._fps_ema = 0.0
        self._scrubbing = False

        self.pip_on = tk.BooleanVar(value=True)
        self.loop = tk.BooleanVar(value=True)
        self.stretch = tk.BooleanVar(value=False)
        self.stretch_ref = tk.BooleanVar(value=True)
        self.speed = tk.StringVar(value="1.0×")
        self.seg_choice = tk.StringVar()
        self.scrub = tk.DoubleVar(value=0.0)

        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=3)
        self.rowconfigure(2, weight=4)

        head = ttk.Frame(self)
        head.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.title = ttk.Label(head, text="", style="H2.TLabel")
        self.title.pack(side="left")
        ttk.Button(head, text="◂ New run", command=self.app.show_setup).pack(side="right")
        ttk.Button(head, text="Export run", command=self.export).pack(side="right", padx=6)
        self.pip_btn = ttk.Checkbutton(head, text="PiP crop: ON", variable=self.pip_on,
                                       command=self.on_toggle_pip)
        self.pip_btn.pack(side="right", padx=16)

        vids = ttk.Frame(self)
        vids.grid(row=1, column=0, columnspan=2, sticky="nsew")
        vids.columnconfigure(0, weight=1)
        vids.columnconfigure(1, weight=1)
        vids.rowconfigure(0, weight=1)
        self.q_pane = VideoPane(vids, "reaction")
        self.q_pane.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.r_pane = VideoPane(vids, "original content")
        self.r_pane.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        self.heat = HeatmapView(self, on_seek=self.on_heatmap_seek)
        self.heat.grid(row=2, column=0, sticky="nsew", pady=(10, 0))

        info = ttk.Frame(self, style="Card.TFrame", padding=12)
        info.grid(row=2, column=1, sticky="nsew", padx=(10, 0), pady=(10, 0))
        info.columnconfigure(0, weight=1)
        ttk.Label(info, text="PREDICTED SEGMENT", style="CardMuted.TLabel").pack(anchor="w")
        self.seg_box = ttk.Combobox(info, textvariable=self.seg_choice, state="readonly")
        self.seg_box.pack(fill="x", pady=(4, 10))
        self.seg_box.bind("<<ComboboxSelected>>", self.on_pick_segment)

        self.ts_q = ttk.Label(info, text="—", style="Value.TLabel")
        self.ts_q.pack(anchor="w")
        ttk.Label(info, text="reaction", style="CardMuted.TLabel").pack(anchor="w")
        ttk.Label(info, text="↕", style="Card.TLabel").pack(anchor="w")
        self.ts_r = ttk.Label(info, text="—", style="Value.TLabel")
        self.ts_r.pack(anchor="w")
        ttk.Label(info, text="original content", style="CardMuted.TLabel").pack(anchor="w",
                                                                                pady=(0, 10))
        ttk.Checkbutton(info, text="auto-stretch heatmap contrast", variable=self.stretch,
                        style="Card.TCheckbutton",
                        command=lambda: self.heat.set_stretch(self.stretch.get())).pack(anchor="w")
        ttk.Separator(info).pack(fill="x", pady=6)
        self.stats = ttk.Label(info, text="", style="Card.TLabel", justify="left",
                               font=("TkFixedFont", 9))
        self.stats.pack(anchor="w")
        ttk.Separator(info).pack(fill="x", pady=6)
        self.delta = ttk.Label(info, text="", style="CardMuted.TLabel", justify="left",
                               wraplength=240)
        self.delta.pack(anchor="w")

        bar = ttk.Frame(self)
        bar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        bar.columnconfigure(3, weight=1)
        self.play_btn = ttk.Button(bar, text="▶  Play", width=10, command=self.toggle_play)
        self.play_btn.grid(row=0, column=0)
        ttk.Button(bar, text="⟲", width=3, command=self.restart).grid(row=0, column=1, padx=6)
        ttk.Checkbutton(bar, text="loop", variable=self.loop).grid(row=0, column=2, padx=(0, 8))
        ttk.Combobox(bar, textvariable=self.speed, state="readonly", width=6,
                     values=["0.25×", "0.5×", "1.0×", "1.5×", "2.0×"]).grid(row=0, column=5,
                                                                            padx=(10, 0))
        ttk.Checkbutton(bar, text="stretch reference to segment",
                        variable=self.stretch_ref,
                        command=self.render).grid(row=0, column=6, padx=(10, 0))
        sc = ttk.Scale(bar, from_=0.0, to=1.0, variable=self.scrub, orient="horizontal",
                       length=600, command=self.on_scrub)
        sc.grid(row=0, column=3, sticky="ew")
        sc.bind("<ButtonPress-1>", lambda _e: setattr(self, "_scrubbing", True))
        sc.bind("<ButtonRelease-1>", self.on_scrub_end)
        self.clock = ttk.Label(bar, text="00:00 / 00:00", style="Muted.TLabel", width=22,
                               anchor="e")
        self.clock.grid(row=0, column=4, padx=(10, 0))

    def load(self, result: DemoResult) -> None:
        self.result = result
        cfg = result.config
        self.title.configure(
            text=f"{Path(cfg.query_path).name}   ×   {Path(cfg.ref_path).name}      "
                 f"[{cfg.backbone.upper()} · {cfg.method} · PiP conf {cfg.pip_conf:.2f}"
                 + (f" · SPD conf {cfg.spd_conf:.2f}" if cfg.method == "SPD" else "")
                 + f" · {result.elapsed:.0f}s]"
        )
        self.q_pane.open(result.query.path, result.query.fps, result.query.frame_count)
        self.r_pane.open(result.reference.path, result.reference.fps, result.reference.frame_count)
        self.pip_on.set(True)
        self.apply_variant(reset=True)

    def unload(self) -> None:
        self.playing = False
        self.q_pane.close()
        self.r_pane.close()

    @property
    def variant(self):
        return self.result.variant(self.pip_on.get())

    def segments(self) -> list[Segment]:
        segs = self.variant.segments
        if segs:
            return segs
        return [Segment(0, 0, int(self.result.query.duration), int(self.result.reference.duration))]

    @property
    def segment(self) -> Segment:
        segs = self.segments()
        return segs[min(self.seg_index, len(segs) - 1)]

    def apply_variant(self, reset: bool = False) -> None:
        if not self.result:
            return
        var = self.variant
        self.pip_btn.configure(text=f"PiP crop: {'ON' if self.pip_on.get() else 'OFF'}")
        if reset or self.seg_index >= len(self.segments()):
            self.seg_index = 0

        labels = []
        for i, s in enumerate(self.segments()):
            labels.append(f"{i + 1}.  {fmt_ts(s.q_start)}–{fmt_ts(s.q_end)}  ↔  "
                          f"{fmt_ts(s.r_start)}–{fmt_ts(s.r_end)}   ({s.score:+.2f})")
        self.seg_box.configure(values=labels)
        if labels:
            self.seg_choice.set(labels[self.seg_index])

        self.heat.set_map(var.sim, self.variant.segments, self.seg_index)
        self.update_info()
        self.seek(0.0)

    def on_pick_segment(self, _evt=None) -> None:
        idx = self.seg_box.current()
        if idx >= 0:
            self.seg_index = idx
        self.heat.set_active(self.seg_index)
        self.update_info()
        self.seek(0.0)

    def on_toggle_pip(self) -> None:
        was_playing = self.playing
        self.apply_variant(reset=True)
        if was_playing:
            self.start_clock()

    def update_info(self) -> None:
        seg = self.segment
        var = self.variant
        other = self.result.variants[UNCROPPED if self.pip_on.get() else CROPPED]
        self.ts_q.configure(text=f"{fmt_ts(seg.q_start)} – {fmt_ts(seg.q_end)}")
        self.ts_r.configure(text=f"{fmt_ts(seg.r_start)} – {fmt_ts(seg.r_end)}")

        has = bool(var.segments)
        self.stats.configure(
            text=(
                f"variant     {var.name}\n"
                f"note        {var.note}\n"
                f"segments    {len(var.segments)}\n"
                f"path score  {seg.score:+.3f}\n"
                f"contrast    {seg.contrast:+.3f}\n"
                f"map mean    {var.mean_sim:+.3f}\n"
                f"pip frames  {var.pip_frames}\n"
                f"length      {seg.q_len}s ↔ {seg.r_len}s\n"
                f"ref speed   {(seg.r_len / max(1, seg.q_len)):.2f}× reaction"
                + ("" if has else "\n\n(no segment found —\n free playback)")
            )
        )
        ob = other.best
        if ob and var.segments:
            d = seg.score - ob.score
            arrow = "↑" if d > 0 else "↓"
            self.delta.configure(
                text=f"vs {other.name}: path score {arrow} {abs(d):.3f}  "
                     f"({ob.score:+.3f} → {seg.score:+.3f}), "
                     f"{len(other.segments)} → {len(var.segments)} segment(s)."
            )
        elif ob and not var.segments:
            self.delta.configure(text=f"{other.name} found {len(other.segments)} segment(s); "
                                      f"this variant found none.")
        else:
            self.delta.configure(text=f"{other.name} found no segment at all.")

    @property
    def speed_mult(self) -> float:
        try:
            return float(self.speed.get().rstrip("×"))
        except ValueError:
            return 1.0

    def seg_times(self, pos: float) -> tuple[float, float]:
        seg = self.segment
        q_t = seg.q_start + pos
        if not self.stretch_ref.get():
            return q_t, seg.r_start + pos
        frac = min(1.0, max(0.0, pos / max(1e-3, float(seg.q_len))))
        return q_t, seg.r_start + frac * seg.r_len

    def seek(self, pos: float) -> None:
        self._pos = max(0.0, min(pos, float(self.segment.q_len)))
        self._last_tick = None
        self.render()

    def restart(self) -> None:
        self.seek(0.0)

    def toggle_play(self) -> None:
        if self.playing:
            self.playing = False
            self.play_btn.configure(text="▶  Play")
        else:
            self.start_clock()

    def start_clock(self) -> None:
        self.playing = True
        self.play_btn.configure(text="❚❚ Pause")
        self._last_tick = None

    def on_scrub(self, _v=None) -> None:
        if self._scrubbing:
            self._pos = float(self.scrub.get()) * float(self.segment.q_len)
            self.render()

    def on_scrub_end(self, _e=None) -> None:
        self._scrubbing = False
        self.seek(float(self.scrub.get()) * float(self.segment.q_len))

    def on_heatmap_seek(self, q_sec: float, r_sec: float) -> None:
        seg = self.segment
        self.seek(q_sec - seg.q_start)

    def tick(self) -> None:
        if not self.result:
            return
        now = time.perf_counter()
        dt = 0.0 if self._last_tick is None else min(0.25, now - self._last_tick)
        self._last_tick = now
        if dt > 0:
            self._fps_ema = 0.85 * self._fps_ema + 0.15 / dt

        if self.playing and not self._scrubbing:
            self._pos += dt * self.speed_mult
            if self._pos >= self.segment.q_len:
                if self.loop.get():
                    self.seek(0.0)
                    return
                self._pos = float(self.segment.q_len)
                self.playing = False
                self.play_btn.configure(text="▶  Play")
        self.render()

    def render(self) -> None:
        if not self.result:
            return
        q_t, r_t = self.seg_times(self._pos)
        boxes = []
        if self.pip_on.get():
            boxes = [(d.x1, d.y1, d.x2, d.y2) for d in self.result.boxes_at(int(q_t))]
        self.q_pane.show_time(q_t, boxes)
        self.r_pane.show_time(r_t)
        self.heat.set_marker(q_t, r_t)
        if not self._scrubbing:
            self.scrub.set(self._pos / max(1e-3, float(self.segment.q_len)))
        self.clock.configure(
            text=f"{fmt_ts(self._pos)} / {fmt_ts(self.segment.q_len)}   {self._fps_ema:4.0f} fps")

    def export(self) -> None:
        if not self.result:
            return
        out = abspath(Path("output") / "demo_runs" /
                      datetime.now().strftime("%Y%m%d-%H%M%S"))
        try:
            path = export_result(self.result, out)
        except Exception as exc: 
            messagebox.showerror("Export failed", str(exc))
            return
        messagebox.showinfo("Exported", f"similarity maps (.npy/.png) + result.json\n\n{path}")


class DemoApp(tk.Tk):
    def __init__(self, cfg: DemoConfig):
        super().__init__()
        self.title("demo")
        self.geometry("1360x900")
        self.minsize(1100, 760)
        install_style(self)

        self.cfg = cfg
        self.events: "queue.Queue[dict]" = queue.Queue()
        self.pipeline: Pipeline | None = None
        self.worker: threading.Thread | None = None

        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)
        self.setup = SetupScreen(self.container, self)
        self.processing = ProcessingScreen(self.container, self)
        self.result = ResultScreen(self.container, self)
        self.current = None
        self.show(self.setup)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(80, self.drain_events)
        self.after(TICK_MS, self.tick)

    def show(self, frame: ttk.Frame) -> None:
        if self.current is frame:
            return
        if self.current is not None:
            self.current.pack_forget()
        frame.pack(fill="both", expand=True)
        self.current = frame

    def show_setup(self) -> None:
        self.result.unload()
        self.show(self.setup)

    def start_run(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        cfg = self.cfg
        self.processing.reset(
            f"{Path(cfg.query_path).name}  ×  {Path(cfg.ref_path).name}   ·   "
            f"{cfg.backbone.upper()} · {cfg.method} · PiP conf {cfg.pip_conf:.2f}"
            + (f" · SPD conf {cfg.spd_conf:.2f}" if cfg.method == "SPD" else "")
        )
        self.show(self.processing)
        self.pipeline = Pipeline(cfg, on_event=self.events.put)

        def work():
            try:
                res = self.pipeline.run()
                self.events.put({"type": "done", "result": res})
            except Exception as exc:
                self.events.put({"type": "error", "msg": str(exc),
                                 "trace": traceback.format_exc()})

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def cancel_run(self) -> None:
        if self.pipeline:
            self.pipeline.cancel()
        self.show_setup()

    def drain_events(self) -> None:
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev.get("type")
                if kind == "stage":
                    self.processing.on_stage(ev["key"], ev["state"], ev.get("detail", ""),
                                             ev.get("frac"))
                elif kind == "log":
                    self.processing.on_log(ev["msg"])
                elif kind == "done":
                    self.result.load(ev["result"])
                    self.show(self.result)
                elif kind == "error":
                    self.processing.on_log("\nERROR: " + ev["msg"])
                    self.processing.on_log(ev.get("trace", ""))
                    if "cancelled" not in ev["msg"]:
                        messagebox.showerror("Pipeline failed", ev["msg"])
        except queue.Empty:
            pass
        self.after(80, self.drain_events)

    def tick(self) -> None:
        if self.current is self.result:
            try:
                self.result.tick()
            except Exception:
                traceback.print_exc()
        self.after(TICK_MS, self.tick)

    def on_close(self) -> None:
        if self.pipeline:
            self.pipeline.cancel()
        self.result.unload()
        self.destroy()


def parse_args(argv=None) -> DemoConfig:
    ap = argparse.ArgumentParser(description="VCSL PiP demo (Tkinter)")
    ap.add_argument("--query", "--reaction", dest="query", default="", help="reaction video")
    ap.add_argument("--ref", "--original", dest="ref", default="", help="original content video")
    ap.add_argument("--yolo-weights", default="", help="pretrained YOLO PiP detector .pt")
    ap.add_argument("--backbone", choices=["isc", "dino"], default="isc")
    ap.add_argument("--method", choices=["TN", "SPD"], default="TN")
    ap.add_argument("--pip-conf", type=float, default=0.25,
                    help="PiP box detector confidence (YOLO on reaction frames)")
    ap.add_argument("--spd-conf", type=float, default=0.1,
                    help="SPD segment confidence (YOLO on the similarity map) -- separate knob")
    ap.add_argument("--spd-weights", default="", help="defaults to ./<backbone>.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--crop-mode", choices=["pip_only", "main_minus_pip"], default="pip_only")
    ap.add_argument("--batch-size", type=int, default=DemoConfig.batch_size)
    a = ap.parse_args(argv)
    return DemoConfig(
        query_path=a.query,
        ref_path=a.ref,
        yolo_weights=a.yolo_weights,
        backbone=a.backbone,
        method=a.method,
        pip_conf=a.pip_conf,
        spd_conf=a.spd_conf,
        spd_weights=a.spd_weights,
        device=a.device,
        crop_mode=a.crop_mode,
        batch_size=a.batch_size,
    )


def main(argv=None) -> None:
    DemoApp(parse_args(argv)).mainloop()


if __name__ == "__main__":
    main()
