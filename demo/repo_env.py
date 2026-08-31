
import sys
from pathlib import Path
from typing import Union

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "vendor"

_bootstrapped = False
_yolov5_ready = False


def bootstrap() -> Path:
    global _bootstrapped
    if not _bootstrapped:
        for p in (VENDOR_ROOT, REPO_ROOT):
            s = str(p)
            if s in sys.path:
                sys.path.remove(s)
            sys.path.insert(0, s)
        _bootstrapped = True
    return REPO_ROOT


def abspath(p: Union[str, Path]) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def enable_yolov5_unpickling() -> None:
    global _yolov5_ready
    if _yolov5_ready:
        return
    bootstrap()

    import torch

    _orig_load = torch.load
    if not getattr(_orig_load, "_vcsl_demo_patched", False):
        def _load_full_unpickle(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)

        _load_full_unpickle._vcsl_demo_patched = True
        torch.load = _load_full_unpickle

    import vcsl.yolov5.models as _models
    import vcsl.yolov5.models.common as _models_common
    import vcsl.yolov5.models.experimental as _models_experimental
    import vcsl.yolov5.models.yolo as _models_yolo
    import vcsl.yolov5.utils as _utils
    import vcsl.yolov5.utils.activations as _utils_activations
    import vcsl.yolov5.utils.datasets as _utils_datasets
    import vcsl.yolov5.utils.general as _utils_general
    import vcsl.yolov5.utils.google_utils as _utils_google
    import vcsl.yolov5.utils.torch_utils as _utils_torch

    for name, mod in {
        "models": _models,
        "models.yolo": _models_yolo,
        "models.common": _models_common,
        "models.experimental": _models_experimental,
        "utils": _utils,
        "utils.general": _utils_general,
        "utils.datasets": _utils_datasets,
        "utils.google_utils": _utils_google,
        "utils.torch_utils": _utils_torch,
        "utils.activations": _utils_activations,
    }.items():
        sys.modules.setdefault(name, mod)

    _yolov5_ready = True


def pick_device(requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
