import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vcsl_stats import extract_per_video, normality_test

COMBOS = [("isc", "TN"), ("isc", "SPD"), ("dino", "TN"), ("dino", "SPD")]

COMPARISONS = [
    ("pred_crop", "no_crop"),
    ("pred_crop", "gt_crop"),
]


def load_scenarios(backbone, method, split, root, dataset):
    anno_file = dataset / "gt_annotations.json"
    result_name = f"{method}_{split}.json"

    scenario_files = {
        "no_crop": root / "vta_out" / backbone / "full" / result_name,
        "gt_crop": root / "vta_out" / backbone / "pip_only" / "gt" / result_name,
        "pred_crop": root / "vta_out" / backbone / "pip_only" / "pred" / result_name,
    }

    missing = [str(p) for p in scenario_files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing prediction file(s), run `make vta` for these first:\n  "
            + "\n  ".join(missing)
        )

    return {s: extract_per_video(p, anno_file, dataset, split) for s, p in scenario_files.items()}


def run_one(backbone, method, split, root, dataset):
    dfs = load_scenarios(backbone, method, split, root, dataset)

    print(f"[{backbone} + {method}, split={split}] Shapiro-Wilk normality test on paired F1 diffs")
    results = []
    for b_name, a_name in COMPARISONS:
        result = normality_test(dfs[a_name], dfs[b_name], metric="f1")
        result["backbone"] = backbone
        result["method"] = method
        result["split"] = split
        result["scenario_a"] = a_name
        result["scenario_b"] = b_name
        results.append(result)

        verdict = "cannot reject normality" if result["normal_at_0.05"] else "reject normality"
        w_str = f"{result['shapiro_stat']:.4f}" if result["shapiro_stat"] is not None else "nan"
        p_str = f"{result['p_value']:.16f}" if result["p_value"] is not None else "nan"
        print(f"  {a_name:9s} vs {b_name:9s} (diff={b_name}-{a_name}): "
              f"n={result['n']:3d}  mean_diff={result['mean_diff']:+.4f}  "
              f"W={w_str}  p={p_str}  ({verdict})")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Shapiro-Wilk normality test on paired F1-score differences: "
                     "no_crop vs pred_crop, and gt_crop vs pred_crop."
    )
    parser.add_argument("--backbone", choices=["isc", "dino"])
    parser.add_argument("--method", choices=["TN", "SPD"])
    parser.add_argument("--all", action="store_true",
                         help="run all 4 backbone/method combos (isc/dino x TN/SPD), "
                              "8 tests total, instead of a single --backbone/--method combo")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--root", default="output", help="output/ dir (contains vta_out/)")
    parser.add_argument("--dataset", default="dataset", help="dataset/ dir")
    parser.add_argument("--out-dir", required=True,
                         help="dir to write the normality-test summary json into")
    args = parser.parse_args()

    root = Path(args.root)
    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        combos = COMBOS
    else:
        if not args.backbone or not args.method:
            parser.error("either pass --all, or both --backbone and --method")
        combos = [(args.backbone, args.method)]

    all_results = []
    for backbone, method in combos:
        all_results.extend(run_one(backbone, method, args.split, root, dataset))
        print()

    n_normal = sum(1 for r in all_results if r["normal_at_0.05"])
    n_total = len(all_results)
    print(f"summary: {n_normal}/{n_total} diffs consistent with normality (p >= 0.05)")

    suffix = "all" if args.all else f"{args.backbone}_{args.method}"
    summary_path = out_dir / f"normtest_{suffix}_{args.split}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote summary to {summary_path}")


if __name__ == "__main__":
    main()
