import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from config import PATHS


@dataclass
class FrameEntry:
    seq_index: int 
    orig_frame_num: int 
    image_path: Optional[Path]
    label_path: Optional[Path]


_PATTERN_CACHE: Dict[str, re.Pattern] = {}


def _pattern(frames_root: Path) -> re.Pattern:
    key = str(frames_root)
    if key not in _PATTERN_CACHE:
        _PATTERN_CACHE[key] = re.compile(PATHS.frame_filename_re)
    return _PATTERN_CACHE[key]


def index_flat_frames(uuid: str, frames_root: Path = PATHS.frames_root,
                       yolo_root: Path = PATHS.yolo_root,
                       yolo_same_folder: bool = PATHS.yolo_same_folder,
                       expected_frame_count: Optional[int] = None) -> List[FrameEntry]:
    pat = _pattern(frames_root)
    matches = []
    for p in frames_root.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m and m.group("uuid") == uuid:
            matches.append((int(m.group("frame_num")), p))

    matches.sort(key=lambda t: t[0])

    entries: List[FrameEntry] = []
    for seq_index, (orig_num, img_path) in enumerate(matches):
        label_path = None
        if yolo_same_folder:
            candidate = img_path.with_suffix(".txt")
            label_path = candidate if candidate.exists() else None
        else:
            candidate = yolo_root / f"{img_path.stem}.txt"
            label_path = candidate if candidate.exists() else None
        entries.append(FrameEntry(seq_index=seq_index, orig_frame_num=orig_num,
                                   image_path=img_path, label_path=label_path))

    if expected_frame_count is not None and len(entries) != expected_frame_count:
        raise ValueError(
            f"{uuid}: found {len(entries)} frames in {frames_root}, but "
            f"frames_all.csv says frame_count={expected_frame_count}. "
            f"gt_annotations.json indices won't line up until this matches -- "
            f"check for missing/extra frame files or a naming-pattern mismatch."
        )

    return entries
