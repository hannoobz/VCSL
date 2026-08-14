import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vcsl_stats import extract_per_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-file", required=True, help="path to gt_annotations.json")
    parser.add_argument("--dataset", required=True, help="dir containing pair_file_{split}.csv")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--pred-file", required=True,
                         help="output json from vcsl_run_video_vta.py for ONE scenario, "
                              "e.g. output/vta_out/isc/pip_only/gt/SPD_test.json")
    parser.add_argument("--out", required=True, help="output CSV path")
    args = parser.parse_args()

    df = extract_per_video(args.pred_file, args.anno_file, args.dataset, args.split)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)

    print(f"wrote {len(df)} rows to {args.out}")
    print(f"mean precision={df['precision'].mean():.4f}  "
          f"mean recall={df['recall'].mean():.4f}  "
          f"mean f1={df['f1'].mean():.4f}")


if __name__ == "__main__":
    main()
