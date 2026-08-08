from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import FEAT
from features.base import BaseExtractor


class ISCExtractor(BaseExtractor):
    name = "isc"
    dim = 256

    def __init__(self, device: str = FEAT.device, weights: str = FEAT.isc_weights):
        super().__init__(device)
        try:
            from isc_feature_extractor import create_model
        except ImportError as e:
            raise ImportError(
                "isc_feature_extractor not installed. Run:\n"
                "  pip install git+https://github.com/lyakaap/ISC21-Descriptor-Track-1st "
            ) from e
        self.model, self.preprocess_fn = create_model(weight_name=weights, device=self.device)
        self.model.eval()

    def preprocess(self, image_paths: List[Path]) -> torch.Tensor:
        imgs = [self.preprocess_fn(Image.open(p).convert("RGB")) for p in image_paths]
        return torch.stack(imgs).to(self.device)

    def forward(self, batch: torch.Tensor) -> np.ndarray:
        feat = self.model(batch)
        feat = F.normalize(feat, dim=1)
        return feat.detach().cpu().numpy().astype(np.float32)
