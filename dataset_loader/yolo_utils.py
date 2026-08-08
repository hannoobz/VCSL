from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config import PATHS, CROP


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def pad(self, px: int, py: int, w: int, h: int) -> "Box":
        return Box(
            x1=max(0, self.x1 - px),
            y1=max(0, self.y1 - py),
            x2=min(w, self.x2 + px),
            y2=min(h, self.y2 + py),
        )


def read_yolo_boxes(txt_path: Optional[Path], img_w: int, img_h: int,
                     class_id: int = CROP.pip_class_id) -> List[Box]:
    if txt_path is None or not txt_path.exists():
        return []
    boxes = []
    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            if cls != class_id:
                continue
            cx, cy, w, h = (float(x) for x in parts[1:5])
            boxes.append(Box(
                x1=int((cx - w / 2) * img_w),
                y1=int((cy - h / 2) * img_h),
                x2=int((cx + w / 2) * img_w),
                y2=int((cy + h / 2) * img_h),
            ))
    return boxes