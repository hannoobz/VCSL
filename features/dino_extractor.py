from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import FEAT
from features.base import BaseExtractor

_TFM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_CLS_DIMS = {
    "dino_vits16": 384, "dino_vits8": 384,
    "dino_vitb16": 768, "dino_vitb8": 768,
}


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), x.size(-1)).pow(1.0 / self.p)
        return x.squeeze(-1)


def _patch_forward_with_tokens(vit: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = vit.prepare_tokens(x)
    for blk in vit.blocks:
        x = blk(x)
    x = vit.norm(x)
    return x


class DINOExtractor(BaseExtractor):
    name = "dino"

    def __init__(self, device: str = FEAT.device, arch: str = FEAT.dino_arch):
        super().__init__(device)
        self.arch = arch
        self.model = torch.hub.load("facebookresearch/dino:main", arch).to(self.device).eval()
        self.is_vit = arch in _CLS_DIMS
        if self.is_vit:
            self.gem = GeM().to(self.device).eval()
            self.dim = _CLS_DIMS[arch] * 2
        else:
            self.dim = 2048

    def preprocess(self, image_paths: List[Path]) -> torch.Tensor:
        imgs = [_TFM(Image.open(p).convert("RGB")) for p in image_paths]
        return torch.stack(imgs).to(self.device)

    def forward(self, batch: torch.Tensor) -> np.ndarray:
        if not self.is_vit:
            feat = F.normalize(self.model(batch), dim=1)
            return feat.detach().cpu().numpy().astype(np.float32)

        tokens = _patch_forward_with_tokens(self.model, batch)
        cls_feat = tokens[:, 0]
        patch_feat = self.gem(tokens[:, 1:])
        feat = torch.cat([cls_feat, patch_feat], dim=-1)
        feat = F.normalize(feat, dim=1)
        return feat.detach().cpu().numpy().astype(np.float32)
