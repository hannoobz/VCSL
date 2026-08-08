import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from config import PATHS, FEAT
from dataset_loader.dataset import load_frames_all, frame_paths as get_gt_aligned_frame_paths
from crop.pip_cropper import crop_video
from features.isc_extractor import ISCExtractor
from features.dino_extractor import DINOExtractor


def get_extractor(backbone: str):
    if backbone == "isc":
        return ISCExtractor()
    if backbone == "dino":
        return DINOExtractor()
    raise ValueError(f"unknown backbone {backbone}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["isc", "dino"], required=True)
    ap.add_argument("--crop_mode", choices=["full", "pip_only", "main_minus_pip"], default="full")
    ap.add_argument("--stride", type=int, default=FEAT.frame_stride)
    ap.add_argument("--videos", nargs="*", default=None, help="restrict to these uuids (default: all)")
    ap.add_argument("--batch_size", type=int, default=FEAT.batch_size,
                     help=f"frames per forward pass (default {FEAT.batch_size}). Lower this if you hit "
                          f"CUDA OOM -- dino_vitb8 in particular needs a much smaller batch than "
                          f"dino_vitb16/vits16 on small GPUs (4x more tokens at the same image size, "
                          f"since patch_size=8 vs 16). On a ~4GB card, try 4-8 for vitb8.")

    ap.add_argument("--yolo-root", type=str, default=None,
                     help="dir with the YOLO .txt labels to use for pip_only/main_minus_pip "
                          "cropping (e.g. your ground-truth labels vs your YOLO model's "
                          "predicted labels -- point this at whichever one you want for this "
                          "run). Defaults to config.py's PATHS.yolo_root if not passed. "
                          "Ignored entirely for --crop_mode full.")
    ap.add_argument("--yolo-same-folder", dest="yolo_same_folder", action="store_true", default=None,
                     help="labels live in the same folder as the frame images (same stem, .txt vs .jpg)")
    ap.add_argument("--yolo-separate-folder", dest="yolo_same_folder", action="store_false",
                     help="labels live in a separate folder from --yolo-root (default assumption "
                          "if neither this nor --yolo-same-folder is passed: config.py's default)")
    ap.add_argument("--label-set", type=str, default="",
                     help="short tag naming which label source this run used (e.g. 'gt', 'pred') -- "
                          "namespaces the feature/crop cache so switching label sources never reads "
                          "back another run's cached results for the same --crop_mode. Recommended "
                          "whenever --crop_mode isn't 'full'.")

    args = ap.parse_args()

    yolo_root = Path(args.yolo_root) if args.yolo_root else PATHS.yolo_root
    yolo_same_folder = args.yolo_same_folder if args.yolo_same_folder is not None else PATHS.yolo_same_folder

    if args.crop_mode != "full" and not args.label_set:
        print(f"[WARN] --crop_mode {args.crop_mode} touches YOLO labels but no --label-set was "
              f"given -- crops/features will be cached under an unnamespaced path. If you plan to "
              f"also run this with a different --yolo-root later (e.g. gt vs pred), pass "
              f"--label-set now or you'll silently read back this run's cached results instead of "
              f"recomputing.")

    videos = load_frames_all()
    if args.videos:
        videos = {k: v for k, v in videos.items() if k in args.videos}

    extractor = get_extractor(args.backbone)
    feat_dir = PATHS.feat_dir(args.backbone, args.crop_mode, label_set=args.label_set)

    for uuid, vinfo in tqdm(videos.items(), desc=f"{args.backbone}/{args.crop_mode}"
                                                  + (f"/{args.label_set}" if args.label_set else "")):
        out_npy = feat_dir / f"{uuid}.npy"
        if out_npy.exists():
            continue

        if args.crop_mode == "full":
            ordered_paths = get_gt_aligned_frame_paths(vinfo, stride=args.stride)
        else:
            crop_out_root = PATHS.crop_dir(args.crop_mode, label_set=args.label_set)
            frame_dir = crop_video(uuid, vinfo.frame_count, args.crop_mode,
                                    yolo_root=yolo_root, yolo_same_folder=yolo_same_folder,
                                    out_root=crop_out_root)
            ordered_paths = sorted(frame_dir.glob("*.jpg"), key=lambda p: int(p.stem))[::args.stride]

        if not ordered_paths:
            continue
        extractor.extract_video(ordered_paths, out_npy, batch_size=args.batch_size)

    print(f"done. features at {feat_dir}")


if __name__ == "__main__":
    main()
