from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    root: Path = Path("./dataset")
    frames_all_csv: Path = root / "frames_all.csv"
    pair_file_csv: Path = root / "pair_file.csv"
    pair_file_test_csv: Path = root / "pair_file_test.csv"
    gt_annotations_json: Path = root / "gt_annotations.json"
    meta_info_json: Path = root / "meta_info.json"
    split_meta_pairs_json: Path = root / "split_meta_pairs.json"

    frames_root: Path = Path("./dataset/images")
    frame_filename_re: str = r"^(?P<uuid>.+)_f(?P<frame_num>\d+)\.(?P<ext>jpg|jpeg|png)$"

    yolo_root: Path = Path("./dataset/labels")
    yolo_same_folder: bool = False

    work_dir: Path = Path("./output/")

    def feat_dir(self, backbone: str, crop_mode: str, label_set: str = "") -> Path:
        parts = [self.work_dir, "features", backbone, crop_mode]
        if label_set:
            parts.append(label_set)
        d = Path(*[str(p) for p in parts])
        d.mkdir(parents=True, exist_ok=True)
        return d

    def crop_dir(self, crop_mode: str, label_set: str = "") -> Path:
        parts = [self.work_dir, "crops", crop_mode]
        if label_set:
            parts.append(label_set)
        d = Path(*[str(p) for p in parts])
        d.mkdir(parents=True, exist_ok=True)
        return d

    def result_dir(self) -> Path:
        d = self.work_dir / "results"
        d.mkdir(parents=True, exist_ok=True)
        return d


@dataclass
class FeatureCfg:
    frame_stride: int = 1
    batch_size: int = 32
    device: str = "cuda"
    isc_weights: str = "isc_ft_v107"
    dino_arch: str = "dino_vitb8"


@dataclass
class CropCfg:
    pip_class_id: int = 0
    fill_value: int = 0
    pad_ratio: float = 0.02


PATHS = Paths()
FEAT = FeatureCfg()
CROP = CropCfg()
