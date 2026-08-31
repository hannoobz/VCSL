
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from demo.repo_env import bootstrap, enable_yolov5_unpickling

bootstrap()

from config import CROP 
from dataset_loader.yolo_utils import Box 


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float

    def to_box(self) -> Box:
        return Box(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)

    def as_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2, round(self.conf, 4)]

    @staticmethod
    def from_list(v: Sequence[float]) -> "Detection":
        return Detection(int(v[0]), int(v[1]), int(v[2]), int(v[3]), float(v[4]))

    def to_yolo_line(self, img_w: int, img_h: int, class_id: int = 0) -> str:
        cx = (self.x1 + self.x2) / 2 / img_w
        cy = (self.y1 + self.y2) / 2 / img_h
        w = (self.x2 - self.x1) / img_w
        h = (self.y2 - self.y1) / img_h
        return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {self.conf:.4f}"


class PiPDetector:
    def __init__(
        self,
        weights: Path,
        device: str = "cpu",
        class_id: int = CROP.pip_class_id,
        imgsz: int = 640,
        iou_thresh: float = 0.45,
        batch_size: int = 16,
    ):
        self.weights = Path(weights)
        self.device = device
        self.class_id = class_id
        self.imgsz = imgsz
        self.iou_thresh = iou_thresh
        self.batch_size = batch_size
        self.backend: Optional[str] = None
        self._model = None

    def load(self) -> str:
        if self._model is not None:
            return self.backend

        if not self.weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")

        try:
            from ultralytics import YOLO 
            self._model = YOLO(str(self.weights))
            self.backend = "ultralytics"
        except Exception as ultra_err:
            try:
                enable_yolov5_unpickling()
                import torch
                from vcsl.yolov5 import attempt_load

                self._model = attempt_load(str(self.weights), map_location=self.device)
                self._model.eval()
                self.backend = "yolov5-vendored"
            except Exception as v5_err:
                raise RuntimeError(
                    f"could not load YOLO weights {self.weights}\n"
                    f"  ultralytics backend: {ultra_err}\n"
                    f"  vendored yolov5 backend: {v5_err}\n"
                    f"install ultralytics (`pip install ultralytics`) if this is a "
                    f"YOLOv8/v11 checkpoint"
                ) from v5_err
        return self.backend

    def detect(
        self,
        frame_paths: Sequence[Path],
        conf: float,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[Detection]]:
        self.load()
        out: List[List[Detection]] = []
        total = len(frame_paths)

        for i in range(0, total, self.batch_size):
            chunk = list(frame_paths[i : i + self.batch_size])
            if self.backend == "ultralytics":
                out.extend(self._detect_ultralytics(chunk, conf))
            else:
                out.extend(self._detect_yolov5(chunk, conf))
            if progress:
                progress(min(i + len(chunk), total), total)
        return out

    def _detect_ultralytics(self, paths: List[Path], conf: float) -> List[List[Detection]]:
        results = self._model.predict(
            source=[str(p) for p in paths],
            conf=conf,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        batch: List[List[Detection]] = []
        for res in results:
            dets: List[Detection] = []
            boxes = getattr(res, "boxes", None)
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                    if self.class_id is not None and int(k) != int(self.class_id):
                        continue
                    dets.append(Detection(int(x1), int(y1), int(x2), int(y2), float(c)))
            batch.append(dets)
        return batch

    def _detect_yolov5(self, paths: List[Path], conf: float) -> List[List[Detection]]:
        import cv2
        import torch

        from vcsl.yolov5 import letterbox, non_max_suppression, scale_coords

        batch: List[List[Detection]] = []
        for p in paths:
            img0 = cv2.imread(str(p))
            if img0 is None:
                batch.append([])
                continue
            img = letterbox(img0, new_shape=self.imgsz, auto=False)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
            x = torch.from_numpy(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = self._model(x, augment=False)[0]
            det = non_max_suppression(pred, conf, self.iou_thresh, agnostic=False)[0]

            dets: List[Detection] = []
            if det is not None and len(det):
                det[:, :4] = scale_coords(x.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, c, k in det.cpu().numpy():
                    if self.class_id is not None and int(k) != int(self.class_id):
                        continue
                    x1, y1, x2, y2 = (int(v) for v in xyxy)
                    dets.append(Detection(x1, y1, x2, y2, float(c)))
            batch.append(dets)
        return batch


def to_boxes(dets: Sequence[Detection], min_conf: float = 0.0) -> List[Box]:
    return [d.to_box() for d in dets if d.conf >= min_conf]
