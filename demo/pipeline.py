
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from demo.pip_detector import Detection, PiPDetector
from demo.repo_env import abspath, bootstrap, enable_yolov5_unpickling, pick_device

bootstrap()

from config import CROP, FEAT, PATHS 
from crop.pip_cropper import crop_frame 

BASE_DETECT_CONF = 0.05

UNCROPPED = "uncropped"
CROPPED = "cropped"

STAGES: List[Tuple[str, str]] = [
    ("sample", "Sampling frames (1 fps)"),
    ("detect", "Detecting PiP"),
    ("crop", "Cropping"),
    ("feat", "Extracting features"),
    ("sim", "Building similarity map"),
    ("align", "Aligning"),
]


@dataclass
class VideoMeta:
    path: str
    key: str
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int

    @property
    def name(self) -> str:
        return Path(self.path).name


@dataclass
class Segment:
    q_start: int
    r_start: int
    q_end: int
    r_end: int
    score: float = 0.0
    contrast: float = 0.0

    @property
    def q_len(self) -> int:
        return max(0, self.q_end - self.q_start)

    @property
    def r_len(self) -> int:
        return max(0, self.r_end - self.r_start)

    def as_list(self) -> List[int]:
        return [self.q_start, self.r_start, self.q_end, self.r_end]


@dataclass
class Variant:
    name: str
    sim: np.ndarray
    segments: List[Segment]
    mean_sim: float = 0.0
    pip_frames: int = 0
    note: str = ""

    @property
    def best(self) -> Optional[Segment]:
        return self.segments[0] if self.segments else None


@dataclass
class DemoConfig:
    query_path: str = "" 
    ref_path: str = ""  
    backbone: str = "isc" 
    method: str = "TN" 
    pip_conf: float = 0.25
    pip_iou: float = 0.45 
    yolo_weights: str = ""
    spd_weights: str = "" 
    device: str = "auto"
    crop_mode: str = "pip_only" 
    batch_size: int = FEAT.batch_size
    min_sim: float = 0.3
    min_length: int = 5
    max_iou: float = 0.3 
    tn_top_k: int = 3
    tn_max_step: int = 10
    max_path: int = 10
    spd_conf: float = 0.1
    spd_iou: float = 0.3

    def resolved_spd_weights(self) -> Path:
        if self.spd_weights:
            return Path(self.spd_weights)
        return abspath(f"./{self.backbone}.pt")


@dataclass
class DemoResult:
    config: DemoConfig
    query: VideoMeta
    reference: VideoMeta
    variants: Dict[str, Variant]
    query_dets: List[List[Detection]] = field(default_factory=list)
    elapsed: float = 0.0
    backend: str = ""

    def variant(self, pip_on: bool) -> Variant:
        return self.variants[CROPPED if pip_on else UNCROPPED]

    def boxes_at(self, second: int) -> List[Detection]:
        if 0 <= second < len(self.query_dets):
            return [d for d in self.query_dets[second] if d.conf >= self.config.pip_conf]
        return []


def video_key(path: Path) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def probe_video(path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 1e-3:
        fps = 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return VideoMeta(
        path=str(path),
        key=video_key(path),
        fps=float(fps),
        frame_count=count,
        duration=(count / fps) if count else 0.0,
        width=w,
        height=h,
    )


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def cache_root() -> Path:
    root = abspath(PATHS.work_dir) / "demo_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _extractor(backbone: str, device: str):
    if not hasattr(_extractor, "_cache"):
        _extractor._cache = {}
    ck = (backbone, device)
    if ck not in _extractor._cache:
        if backbone == "isc":
            from features.isc_extractor import ISCExtractor

            _extractor._cache[ck] = ISCExtractor(device=device)
        elif backbone == "dino":
            from features.dino_extractor import DINOExtractor

            _extractor._cache[ck] = DINOExtractor(device=device)
        else:
            raise ValueError(f"unknown backbone {backbone}")
    return _extractor._cache[ck]


def _spd_model(weights: Path, device: str):
    if not hasattr(_spd_model, "_cache"):
        _spd_model._cache = {}
    ck = (str(weights), device)
    if ck not in _spd_model._cache:
        enable_yolov5_unpickling()
        from vcsl.yolov5 import attempt_load

        if not Path(weights).exists():
            raise FileNotFoundError(
                f"SPD checkpoint not found: {weights}\n"
                f"the repo's Makefile expects ./<backbone>.pt (isc.pt / dino.pt), "
                f"see vendor/spd_models.txt"
            )
        _spd_model._cache[ck] = attempt_load(str(weights), map_location=device)
    return _spd_model._cache[ck]


class Pipeline:
    def __init__(self, cfg: DemoConfig, on_event: Optional[Callable[[dict], None]] = None):
        self.cfg = cfg
        self._cb = on_event or (lambda ev: None)
        self.device = pick_device(cfg.device)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _check_cancel(self) -> None:
        if self._cancelled:
            raise RuntimeError("cancelled")

    def stage(self, key: str, state: str, detail: str = "", frac: Optional[float] = None) -> None:
        self._cb({"type": "stage", "key": key, "state": state, "detail": detail, "frac": frac})

    def log(self, msg: str) -> None:
        self._cb({"type": "log", "msg": msg})

    def sample_frames(self, meta: VideoMeta, label: str) -> List[Path]:
        out_dir = cache_root() / meta.key / "frames"
        done_flag = out_dir / ".done"
        if done_flag.exists():
            paths = sorted(out_dir.glob("*.jpg"), key=lambda p: int(p.stem))
            if paths:
                self.log(f"[cache] {label}: {len(paths)} frames already sampled")
                return paths

        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(meta.path)
        paths: List[Path] = []
        idx, next_take, taken = 0, 0.0, 0
        expected = max(1, int(meta.duration) or 1)

        while True:
            self._check_cancel()
            ok = cap.grab()
            if not ok:
                break
            if idx >= next_take:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    p = out_dir / f"{taken:05d}.jpg"
                    cv2.imwrite(str(p), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    paths.append(p)
                    taken += 1
                    if taken % 20 == 0:
                        self.stage("sample", "running", f"{label}: {taken}s", min(0.99, taken / expected))
                next_take += meta.fps
            idx += 1
        cap.release()
        done_flag.write_text(str(len(paths)))
        self.log(f"{label}: sampled {len(paths)} frames @1fps from {idx} decoded frames")
        return paths

    def detect(self, meta: VideoMeta, frames: List[Path], detector: PiPDetector, label: str
               ) -> List[List[Detection]]:
        cache_file = cache_root() / meta.key / "detections.json"
        if cache_file.exists():
            try:
                raw = json.loads(cache_file.read_text())
                if raw.get("n") == len(frames) and raw.get("base_conf") == BASE_DETECT_CONF:
                    self.log(f"[cache] {label}: reusing YOLO detections")
                    return [[Detection.from_list(d) for d in per] for per in raw["dets"]]
            except Exception:
                pass

        def prog(done: int, total: int) -> None:
            self._check_cancel()
            self.stage("detect", "running", f"{label}: {done}/{total} frames", done / max(1, total))

        dets = detector.detect(frames, conf=BASE_DETECT_CONF, progress=prog)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(
                {
                    "n": len(frames),
                    "base_conf": BASE_DETECT_CONF,
                    "backend": detector.backend,
                    "dets": [[d.as_list() for d in per] for per in dets],
                }
            )
        )
        hits = sum(1 for per in dets if any(d.conf >= self.cfg.pip_conf for d in per))
        self.log(f"{label}: PiP found in {hits}/{len(frames)} frames at conf>={self.cfg.pip_conf:.2f}")
        return dets

    def crop(self, meta: VideoMeta, frames: List[Path], dets: List[List[Detection]], label: str
             ) -> Tuple[List[Path], int]:
        conf_tag = f"conf{self.cfg.pip_conf:.2f}"
        out_dir = cache_root() / meta.key / "crops" / self.cfg.crop_mode / conf_tag
        done_flag = out_dir / ".done"
        n_pip = sum(1 for per in dets if any(d.conf >= self.cfg.pip_conf for d in per))

        if done_flag.exists():
            paths = sorted(out_dir.glob("*.jpg"), key=lambda p: int(p.stem))
            if len(paths) == len(frames):
                self.log(f"[cache] {label}: crops already written ({conf_tag})")
                return paths, n_pip

        out_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        for i, src in enumerate(frames):
            self._check_cancel()
            dst = out_dir / f"{i:05d}.jpg"
            if not dst.exists():
                img = cv2.imread(str(src))
                if img is None:
                    continue
                boxes = [d.to_box() for d in dets[i] if d.conf >= self.cfg.pip_conf]
                out = crop_frame(img, boxes, self.cfg.crop_mode)
                if out is None or out.size == 0:
                    out = img
                cv2.imwrite(str(dst), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            paths.append(dst)
            if i % 25 == 0:
                self.stage("crop", "running", f"{label}: {i}/{len(frames)}", i / max(1, len(frames)))
        done_flag.write_text(str(len(paths)))
        return paths, n_pip

    def features(self, meta: VideoMeta, frames: List[Path], mode: str, label: str) -> np.ndarray:
        sub = mode if mode == "full" else f"{mode}/conf{self.cfg.pip_conf:.2f}"
        out_npy = cache_root() / meta.key / "features" / self.cfg.backbone / f"{sub}.npy"
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        if out_npy.exists():
            self.log(f"[cache] {label} [{mode}]: features already extracted")
            return np.load(out_npy)

        self.stage("feat", "running", f"{label} [{mode}]: {len(frames)} frames", None)
        ext = _extractor(self.cfg.backbone, self.device)
        feats = ext.extract_video(frames, out_npy, batch_size=self.cfg.batch_size)
        self.log(f"{label} [{mode}]: {feats.shape[0]}x{feats.shape[1]} {self.cfg.backbone} features")
        return feats

    def align(self, sim: np.ndarray, tag: str) -> List[Segment]:
        from vcsl.vta import spd, tn

        params = (f"min_sim={self.cfg.min_sim}" if self.cfg.method == "TN"
                  else f"spd_conf={self.cfg.spd_conf}")
        self.stage("align", "running",
                   f"{self.cfg.method} [{tag}] on {sim.shape[0]}x{sim.shape[1]} ({params})", None)
        if self.cfg.method == "TN":
            raw = tn(
                sim,
                tn_max_step=self.cfg.tn_max_step,
                tn_top_k=self.cfg.tn_top_k,
                max_path=self.cfg.max_path,
                min_sim=self.cfg.min_sim,
                min_length=self.cfg.min_length,
                max_iou=self.cfg.max_iou,
            )
        elif self.cfg.method == "SPD":
            model = _spd_model(self.cfg.resolved_spd_weights(), self.device)
            raw = spd(
                sim,
                model=model,
                conf_thresh=self.cfg.spd_conf,
                device=self.device,
                iou_thresh=self.cfg.spd_iou,
            )
        else:
            raise ValueError(f"unknown alignment method {self.cfg.method}")

        segs = [score_segment(sim, Segment(int(b[0]), int(b[1]), int(b[2]), int(b[3]))) for b in raw]
        segs = [s for s in segs if s.q_len > 0 and s.r_len > 0]
        segs.sort(key=lambda s: (s.score * min(s.q_len, s.r_len)), reverse=True)
        self.log(f"[{tag}] {self.cfg.method} -> {len(segs)} segment(s)")
        for s in segs[:5]:
            self.log(
                f"    reaction {fmt_ts(s.q_start)}-{fmt_ts(s.q_end)}  <->  "
                f"original {fmt_ts(s.r_start)}-{fmt_ts(s.r_end)}   score={s.score:.3f}"
            )
        return segs

    def run(self) -> DemoResult:
        t0 = time.time()
        cfg = self.cfg
        qp, rp = Path(cfg.query_path), Path(cfg.ref_path)
        for p, what in ((qp, "reaction video"), (rp, "original content")):
            if not p.exists():
                raise FileNotFoundError(f"{what} not found: {p}")

        self.log(f"device={self.device}  backbone={cfg.backbone}  method={cfg.method}")
        q_meta, r_meta = probe_video(qp), probe_video(rp)
        self.log(f"reaction: {q_meta.name} {q_meta.width}x{q_meta.height} "
                 f"{q_meta.fps:.2f}fps {fmt_ts(q_meta.duration)}")
        self.log(f"original: {r_meta.name} {r_meta.width}x{r_meta.height} "
                 f"{r_meta.fps:.2f}fps {fmt_ts(r_meta.duration)}")

        self.stage("sample", "running", "reaction video")
        q_frames = self.sample_frames(q_meta, "reaction")
        self.stage("sample", "running", "original content")
        r_frames = self.sample_frames(r_meta, "original")
        if not q_frames or not r_frames:
            raise RuntimeError("no frames decoded -- is the file a readable video?")
        self.stage("sample", "done", f"{len(q_frames)} + {len(r_frames)} frames")

        self.stage("detect", "running", "loading YOLO")
        detector = PiPDetector(Path(cfg.yolo_weights), device=self.device,
                               iou_thresh=cfg.pip_iou)
        backend = detector.load()
        self.log(f"PiP detector backend: {backend}")
        q_dets = self.detect(q_meta, q_frames, detector, "reaction")
        q_pip = sum(1 for per in q_dets if any(d.conf >= cfg.pip_conf for d in per))
        self.stage("detect", "done", f"reaction {q_pip}/{len(q_frames)} frames with PiP")

        self.stage("crop", "running", "reaction")
        q_crops, _ = self.crop(q_meta, q_frames, q_dets, "reaction")
        self.stage("crop", "done", f"{len(q_crops)} reaction frames cropped")

        q_full = self.features(q_meta, q_frames, "full", "reaction")
        r_full = self.features(r_meta, r_frames, "full", "original")
        q_crop = self.features(q_meta, q_crops, cfg.crop_mode, "reaction") if q_pip else q_full
        if not q_pip:
            self.log("reaction: no PiP above threshold -- cropped run is identical to uncropped")
        self.stage("feat", "done", f"{self.cfg.backbone} features ready "
                                   f"(original extracted once, shared by both variants)")

        self.stage("sim", "running", "cosine similarity (query x reference)")
        sim_full = cosine_map(q_full, r_full)
        sim_crop = cosine_map(q_crop, r_full)
        self.stage("sim", "done", f"{sim_full.shape[0]}x{sim_full.shape[1]} map")

        segs_full = self.align(sim_full, UNCROPPED)
        segs_crop = self.align(sim_crop, CROPPED)
        self.stage("align", "done", f"{cfg.method}: {len(segs_full)} / {len(segs_crop)} segments")

        variants = {
            UNCROPPED: Variant(UNCROPPED, sim_full, segs_full, float(sim_full.mean()), 0,
                               "reaction: whole frame"),
            CROPPED: Variant(CROPPED, sim_crop, segs_crop, float(sim_crop.mean()), q_pip,
                             f"reaction: {cfg.crop_mode} @ conf>={cfg.pip_conf:.2f}"),
        }
        result = DemoResult(
            config=cfg,
            query=q_meta,
            reference=r_meta,
            variants=variants,
            query_dets=q_dets,
            elapsed=time.time() - t0,
            backend=backend,
        )
        self.log(f"finished in {result.elapsed:.1f}s")
        return result


def cosine_map(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    r = np.asarray(r, dtype=np.float32)
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    rn = np.linalg.norm(r, axis=1, keepdims=True)
    q = q / np.clip(qn, 1e-8, None)
    r = r / np.clip(rn, 1e-8, None)
    return np.dot(q, r.T)


def score_segment(sim: np.ndarray, seg: Segment, samples: int = 64) -> Segment:
    h, w = sim.shape
    q0, q1 = np.clip([seg.q_start, seg.q_end], 0, h - 1)
    r0, r1 = np.clip([seg.r_start, seg.r_end], 0, w - 1)
    n = max(2, min(samples, int(max(abs(q1 - q0), abs(r1 - r0))) + 1))
    ys = np.linspace(q0, q1, n).round().astype(int)
    xs = np.linspace(r0, r1, n).round().astype(int)
    path = sim[ys, xs]
    seg.score = float(path.mean())
    seg.contrast = float(path.mean() - sim.mean())
    return seg


def export_result(result: DemoResult, out_dir: Path) -> Path:
    from scripts.visualize_similarity_maps import render_similarity_map

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(result.config),
        "device": result.backend,
        "elapsed_sec": round(result.elapsed, 2),
        "query": asdict(result.query),
        "reference": asdict(result.reference),
        "variants": {},
    }
    for name, var in result.variants.items():
        np.save(out_dir / f"simmap_{name}.npy", var.sim)
        render_similarity_map(var.sim).save(out_dir / f"simmap_{name}.png")
        payload["variants"][name] = {
            "note": var.note,
            "mean_sim": round(var.mean_sim, 4),
            "pip_frames": var.pip_frames,
            "segments": [
                {
                    "segment": s.as_list(),
                    "reaction": [fmt_ts(s.q_start), fmt_ts(s.q_end)],
                    "original": [fmt_ts(s.r_start), fmt_ts(s.r_end)],
                    "score": round(s.score, 4),
                    "contrast": round(s.contrast, 4),
                }
                for s in var.segments
            ],
        }
    out_json = out_dir / "result.json"
    out_json.write_text(json.dumps(payload, indent=2))
    return out_json
