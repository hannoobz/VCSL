import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
from matplotlib import cm
from PIL import Image
from tqdm import tqdm


def render_similarity_map(sim: np.ndarray, vmin: float = -1.0, vmax: float = 1.0,
                           per_image_normalize: bool = False, cmap_name: str = "inferno") -> Image.Image:
    if per_image_normalize:
        lo, hi = float(sim.min()), float(sim.max())
        if hi > lo:
            normed = (sim - lo) / (hi - lo)
        else:
            normed = np.zeros_like(sim)
    else:
        normed = np.clip((sim - vmin) / (vmax - vmin), 0.0, 1.0)

    try:
        cmap = matplotlib.colormaps[cmap_name]
    except (AttributeError, TypeError):
        cmap = cm.get_cmap(cmap_name)
    rgba = cmap(normed)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", type=str, default=None, help="single .npy similarity map")
    ap.add_argument("--output-file", type=str, default=None, help="output .png path (with --input-file)")
    ap.add_argument("--input-root", type=str, default=None, help="dir of .npy similarity maps")
    ap.add_argument("--output-root", type=str, default=None, help="output dir for .png files (with --input-root)")
    ap.add_argument("--vmin", type=float, default=-1.0, help="fixed colormap lower bound (ignored if --per-image-normalize)")
    ap.add_argument("--vmax", type=float, default=1.0, help="fixed colormap upper bound (ignored if --per-image-normalize)")
    ap.add_argument("--per-image-normalize", action="store_true",
                     help="min-max stretch each map to its own range instead of a fixed [vmin, vmax]")
    ap.add_argument("--cmap", type=str, default="inferno")
    args = ap.parse_args()

    if args.input_file:
        if not args.output_file:
            sys.exit("--output-file is required with --input-file")
        sim = np.load(args.input_file)
        img = render_similarity_map(sim, args.vmin, args.vmax, args.per_image_normalize, args.cmap)
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        print(f"saved -> {out_path}  ({sim.shape[0]}x{sim.shape[1]})")
        return

    if args.input_root:
        if not args.output_root:
            sys.exit("--output-root is required with --input-root")
        in_root = Path(args.input_root)
        out_root = Path(args.output_root)
        out_root.mkdir(parents=True, exist_ok=True)

        npy_files = sorted(in_root.glob("*.npy"))
        for npy_path in tqdm(npy_files, desc="rendering similarity maps"):
            sim = np.load(npy_path)
            img = render_similarity_map(sim, args.vmin, args.vmax, args.per_image_normalize, args.cmap)
            out_path = out_root / (npy_path.stem + ".png")
            img.save(out_path)

        print(f"done: {len(npy_files)} similarity maps -> {out_root}")
        return

    sys.exit("pass either --input-file/--output-file or --input-root/--output-root")

if __name__ == "__main__":
    main()
