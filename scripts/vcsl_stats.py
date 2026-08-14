import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_metric_module():
    spec = importlib.util.spec_from_file_location(
        "vcsl_metric", REPO_ROOT / "vendor" / "vcsl" / "metric.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_metric = _load_metric_module()
precision_recall = _metric.precision_recall


def load_split_pairs(dataset_dir: Path, split: str) -> set:
    df = pd.read_csv(Path(dataset_dir) / f"pair_file_{split}.csv")
    return set(f"{q}-{r}" for q, r in zip(df.query_id.values, df.reference_id.values))


def extract_per_video(pred_file: Path, anno_file: Path, dataset_dir: Path,
                       split: str) -> pd.DataFrame:
    gt = json.load(open(anno_file))
    pred_dict = json.load(open(pred_file))
    split_pairs = load_split_pairs(dataset_dir, split)

    rows = []
    for key in split_pairs:
        gt_box = np.array(gt.get(key, []))
        pred_box = np.array(pred_dict.get(key, []))
        r = precision_recall(pred_box, gt_box)
        p, rec = r["precision"], r["recall"]
        f1 = 0.0 if (p + rec) == 0 else 2 * p * rec / (p + rec)
        query_id, ref_id = key.split("-")
        rows.append({
            "name": key, "query_id": query_id, "reference_id": ref_id,
            "precision": p, "recall": rec, "f1": f1,
        })

    if not rows:
        raise ValueError(
            f"no pairs found for split={split!r} -- check --dataset / --split / "
            f"that {pred_file} actually contains keys from pair_file_{split}.csv"
        )

    return pd.DataFrame(rows).set_index("name").sort_index()


def paired_test(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str = "f1",
                 n_boot: int = 10000, seed: int = 42) -> dict:
    from scipy.stats import wilcoxon

    common = df_a.index.intersection(df_b.index)
    if len(common) == 0:
        raise ValueError("no overlapping pairs between the two scenarios -- "
                          "did they come from the same split/pair_file?")

    a = df_a.loc[common, metric].to_numpy()
    b = df_b.loc[common, metric].to_numpy()
    diffs = b - a

    if np.allclose(diffs, 0):
        stat, p = float("nan"), 1.0
    else:
        stat, p = wilcoxon(b, a)

    rng = np.random.default_rng(seed)
    boot_means = [rng.choice(diffs, size=len(diffs), replace=True).mean()
                  for _ in range(n_boot)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    return {
        "n": len(common),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diffs.mean()),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "wilcoxon_stat": float(stat) if stat == stat else None,
        "p_value": float(p),
        "significant_at_0.05": bool(p < 0.05),
    }


def normality_test(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str = "f1") -> dict:
    common = df_a.index.intersection(df_b.index)
    if len(common) == 0:
        raise ValueError("no overlapping pairs between the two scenarios -- "
                          "did they come from the same split/pair_file?")

    a = df_a.loc[common, metric].to_numpy()
    b = df_b.loc[common, metric].to_numpy()
    diffs = b - a

    if len(diffs) < 3:
        raise ValueError(f"shapiro-wilk needs at least 3 paired samples, got {len(diffs)}")

    if np.allclose(diffs, diffs[0]):
        stat, p = float("nan"), float("nan")
    else:
        stat, p = shapiro(diffs)

    return {
        "n": len(common),
        "mean_diff": float(diffs.mean()),
        "std_diff": float(diffs.std(ddof=1)),
        "shapiro_stat": float(stat) if stat == stat else None,
        "p_value": float(p) if p == p else None,
        "normal_at_0.05": bool(p >= 0.05) if p == p else None,
    }
