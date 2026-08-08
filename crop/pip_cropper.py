from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from config import PATHS, CROP
from dataset_loader.yolo_utils import Box, read_yolo_boxes
from dataset_loader.flat_frame_index import index_flat_frames


def crop_frame(img: np.ndarray, boxes: List[Box], mode: str) -> Optional[np.ndarray]:
    h, w = img.shape[:2]

    if mode == "full":
        return img

    if mode == "main_minus_pip":
        if not boxes:
            return img
        out = img.copy()
        for b in boxes:
            out[b.y1:b.y2, b.x1:b.x2] = CROP.fill_value
        return out

    if mode == "pip_only":
        if not boxes:
            return img

        if len(boxes) == 1:
            b = boxes[0]
            px = int((b.x2 - b.x1) * CROP.pad_ratio)
            py = int((b.y2 - b.y1) * CROP.pad_ratio)
            bp = b.pad(px, py, w, h)
            crop = img[bp.y1:bp.y2, bp.x1:bp.x2]
            if crop.size == 0:
                return img
            return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        out = np.full_like(img, CROP.fill_value)
        for b in boxes:
            out[b.y1:b.y2, b.x1:b.x2] = img[b.y1:b.y2, b.x1:b.x2]
        return out

    raise ValueError(f"unknown crop mode: {mode}")


def crop_video(uuid: str, frame_count: int, mode: str,
                frames_root: Path = PATHS.frames_root,
                yolo_root: Path = PATHS.yolo_root,
                yolo_same_folder: bool = PATHS.yolo_same_folder,
                out_root: Optional[Path] = None,
                ext: str = ".jpg") -> Path:
    out_root = out_root or PATHS.crop_dir(mode)
    dst_dir = out_root / uuid
    dst_dir.mkdir(parents=True, exist_ok=True)

    entries = index_flat_frames(uuid, frames_root=frames_root, yolo_root=yolo_root,
                                 yolo_same_folder=yolo_same_folder,
                                 expected_frame_count=frame_count)

    for entry in entries:
        dst = dst_dir / f"{entry.seq_index}{ext}"
        if dst.exists() or entry.image_path is None:
            continue
        img = cv2.imread(str(entry.image_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = read_yolo_boxes(entry.label_path, w, h)
        out = crop_frame(img, boxes, mode)
        if out is not None:
            cv2.imwrite(str(dst), out)

    return dst_dir
