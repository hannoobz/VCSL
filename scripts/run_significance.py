import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vcsl_stats import extract_per_video, paired_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["isc", "dino"])
    parser.add_argument("--method", required=True, choices=["TN", "SPD"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--root", default="output", help="output/ dir (contains vta_out/)")
    parser.add_argument("--dataset", default="dataset", help="dataset/ dir")
    parser.add_argument("--out-dir", required=True,
                         help="dir to write per-video CSVs + summary into")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    anno_file = dataset / "gt_annotations.json"
    result_name = f"{args.method}_{args.split}.json"

    scenario_files = {
        "no_crop": root / "vta_out" / args.backbone / "full" / result_name,
        "gt_crop": root / "vta_out" / args.backbone / "pip_only" / "gt" / result_name,
        "pred_crop": root / "vta_out" / args.backbone / "pip_only" / "pred" / result_name,
    }

    missing = [str(p) for p in scenario_files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing prediction file(s), run `make vta` for these first:\n  "
            + "\n  ".join(missing)
        )

    print(f"[{args.backbone} + {args.method}, split={args.split}]")
    dfs = {}
    for scenario, path in scenario_files.items():
        df = extract_per_video(path, anno_file, dataset, args.split)
        dfs[scenario] = df
        csv_path = out_dir / f"{args.backbone}_{scenario}_{args.method}_{args.split}.csv"
        df.to_csv(csv_path)
        print(f"  {scenario:10s} n={len(df):3d}  mean_f1={df['f1'].mean():.4f}  -> {csv_path}")

    comparisons = [
        ("gt_crop", "no_crop"),
        ("pred_crop", "no_crop"),
        ("pred_crop", "gt_crop"),
    ]

    print("\npairwise Wilcoxon signed-rank tests (metric=f1):")
    summary = {"backbone": args.backbone, "method": args.method, "split": args.split,
               "comparisons": []}
    for b_name, a_name in comparisons:
        result = paired_test(dfs[a_name], dfs[b_name], metric="f1",
                              n_boot=args.n_boot, seed=args.seed)
        result["scenario_a"] = a_name
        result["scenario_b"] = b_name
        summary["comparisons"].append(result)

        sig = "significant" if result["significant_at_0.05"] else "not significant"
        print(f"  {b_name} vs {a_name}: "
              f"n={result['n']}  "
              f"mean {a_name}={result['mean_a']:.4f}  mean {b_name}={result['mean_b']:.4f}  "
              f"diff={result['mean_diff']:+.4f} (95% CI [{result['ci_lo']:+.4f}, {result['ci_hi']:+.4f}])  "
              f"p={result['p_value']:.6f} ({sig})")

    summary_path = out_dir / f"{args.backbone}_{args.method}_{args.split}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote summary to {summary_path}")


if __name__ == "__main__":
    main()
