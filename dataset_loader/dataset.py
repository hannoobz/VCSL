import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from config import PATHS


@dataclass
class VideoInfo:
    uuid: str
    frame_dir_name: str
    frame_count: int


@dataclass
class Pair:
    query_id: str
    ref_id: str

    @property
    def key(self) -> str:
        return f"{self.query_id}-{self.ref_id}"


def load_frames_all(csv_path: Path = PATHS.frames_all_csv) -> Dict[str, VideoInfo]:
    videos: Dict[str, VideoInfo] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            videos[row["uuid"]] = VideoInfo(
                uuid=row["uuid"],
                frame_dir_name=row["path"],
                frame_count=int(row["frame_count"]),
            )
    return videos


def load_pairs(csv_path: Path) -> List[Pair]:
    pairs = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            pairs.append(Pair(query_id=row["query_id"], ref_id=row["reference_id"]))
    return pairs


def load_gt_annotations(json_path: Path = PATHS.gt_annotations_json) -> Dict[str, List[Tuple[int, int, int, int]]]:
    with open(json_path) as f:
        raw = json.load(f)
    return {k: [tuple(box) for box in v] for k, v in raw.items()}


def load_meta_info(json_path: Path = PATHS.meta_info_json) -> Dict[str, dict]:
    with open(json_path) as f:
        return json.load(f)

def frame_paths(video: VideoInfo, frames_root: Path = PATHS.frames_root,
                 stride: int = 1, ext: str = ".jpg") -> List[Path]:
    from dataset_loader.flat_frame_index import index_flat_frames
    entries = index_flat_frames(video.uuid, frames_root=frames_root,
                                 expected_frame_count=video.frame_count)
    return [e.image_path for e in entries[::stride]]


class Dataset:
    def __init__(self, split: str = "test"):
        self.videos = load_frames_all()
        self.gt = load_gt_annotations()
        self.meta = load_meta_info()
        if split == "test":
            self.pairs = load_pairs(PATHS.pair_file_test_csv)
        else:
            self.pairs = load_pairs(PATHS.pair_file_csv)

    def gt_for(self, pair: Pair) -> List[Tuple[int, int, int, int]]:
        return self.gt.get(pair.key, [])

    def video(self, uuid: str) -> VideoInfo:
        return self.videos[uuid]
