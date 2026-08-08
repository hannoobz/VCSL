import abc
from pathlib import Path
from typing import List

import numpy as np
import torch


class BaseExtractor(abc.ABC):
    name: str = "base"
    dim: int = 0

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

    @abc.abstractmethod
    def preprocess(self, image_paths: List[Path]) -> torch.Tensor:
        """image paths -> a batched, normalized tensor ready for the model."""

    @abc.abstractmethod
    def forward(self, batch: torch.Tensor) -> np.ndarray:
        """batched tensor -> (N, dim) L2-normalized numpy features."""

    def extract_video(self, frame_paths: List[Path], out_npy: Path,
                       batch_size: int = 64, skip_if_exists: bool = True,
                       min_batch_size: int = 1) -> np.ndarray:
        if skip_if_exists and out_npy.exists():
            return np.load(out_npy)

        feats = []
        i = 0
        cur_batch_size = batch_size
        while i < len(frame_paths):
            chunk = [p for p in frame_paths[i:i + cur_batch_size] if p.exists()]
            if not chunk:
                i += cur_batch_size
                continue
            try:
                batch = self.preprocess(chunk)
                with torch.no_grad():
                    f = self.forward(batch)
                feats.append(f)
                i += cur_batch_size
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if cur_batch_size <= min_batch_size:
                    raise
                cur_batch_size = max(min_batch_size, cur_batch_size // 2)
                print(f"[WARN] CUDA OOM at batch_size={cur_batch_size * 2}, "
                      f"retrying this chunk at batch_size={cur_batch_size}")

        feats = np.concatenate(feats, axis=0) if feats else np.zeros((0, self.dim), dtype=np.float32)
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, feats)
        return feats
