import argparse
import json
import multiprocessing
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))

import pandas as pd
from vcsl import *
from vcsl import build_reader
from loguru import logger


def run_eval(input_dict):
    gt_box = np.array(input_dict["gt"])
    pred_box = np.array(input_dict["pred"])
    result_dict = precision_recall(pred_box, gt_box)
    result_dict["name"] = input_dict["name"]
    return result_dict


if __name__ == '__main__':
    if sys.platform != "win32":
        multiprocessing.set_start_method("fork", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-file",
                         default=str(Path(__file__).resolve().parents[1] / "dataset" / "gt_annotations.json"),
                         type=str)
    parser.add_argument("--pair-group-file",
                         default=str(Path(__file__).resolve().parents[1] / "dataset" / "split_meta_pairs.json"),
                         type=str)
    parser.add_argument("--meta-info",
                         default=str(Path(__file__).resolve().parents[1] / "dataset" / "meta_info.json"),
                         type=str)
    parser.add_argument("--pred-file", type=str, required=True,
                         help="output json from vcsl_run_video_vta.py, shape {'Q0-R0': [[qs,rs,qe,re],...]}")
    parser.add_argument("--pred-store", type=str, default="local")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--pair-file-override", type=str, default=None,
                         help="explicit path to this split's pair_file_{split}.csv, "
                              "skipping the --split-pair-dir lookup below")
    parser.add_argument("--split-pair-dir", type=str,
                         default=str(Path(__file__).resolve().parents[1] / "dataset"),
                         help="dir to look for pair_file_{split}.csv in (NOT the same dir "
                              "as --anno-file/--pair-group-file: the corrected, split-filtered "
                              "pair_file_val.csv/pair_file_test.csv live in this repo's own "
                              "dataset/, not wherever your raw uploads sit)")
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--ratio-pos-neg", type=float, default=1, help="ratio between positive and negative samples")
    args = parser.parse_args()

    if args.split not in ['all', 'train', 'val', 'test']:
        raise ValueError(f"Unknown dataset split {args.split}, must be one of train|val|test")

    config = dict()
    reader = build_reader(args.pred_store, "json", **config)

    logger.info("start loading...")
    gt = json.load(open(args.anno_file))
    key_list = [key for key in gt]

    meta_pairs = json.load(open(args.pair_group_file))

    if args.pair_file_override:
        split_file = args.pair_file_override
    else:
        candidate = os.path.join(args.split_pair_dir, f"pair_file_{args.split}.csv")
        if not os.path.exists(candidate):
            raise FileNotFoundError(
                f"couldn't find {candidate} -- pass --pair-file-override explicitly, "
                f"or --split-pair-dir pointing at wherever pair_file_{args.split}.csv "
                f"actually is (the corrected, split-filtered version, not the stale "
                f"uploads/ copy)."
            )
        split_file = candidate

    df = pd.read_csv(split_file)
    split_pairs = set([f"{q}-{r}" for q, r in zip(df.query_id.values, df.reference_id.values)])
    logger.info("{} contains pairs {}", args.split, len(split_pairs))

    key_list = [key for key in key_list if key in split_pairs]
    logger.info("Copied video data (positive) to evaluate: {}", len(key_list))

    pred_dict = reader.read(args.pred_file)

    eval_list = []
    for key in split_pairs:
        if key in gt:
            eval_list += [{"name": key, "gt": gt[key], "pred": pred_dict.get(key, [])}]
        else:
            eval_list += [{"name": key, "gt": [], "pred": pred_dict.get(key, [])}]

    logger.info("finish loading files, start evaluation...")

    process_pool = Pool(args.pool_size)
    result_list = process_pool.map(run_eval, eval_list)
    result_dict = {i['name']: i for i in result_list}

    if args.split != 'all':
        meta_pairs = meta_pairs[args.split]
    else:
        meta_pairs = {**meta_pairs['train'], **meta_pairs['val'], **meta_pairs['test']}

    try:
        feat, vta = os.path.basename(args.pred_file).split('_')[:2]
    except Exception:
        feat, vta = 'My-FEAT', 'My-VTA'

    r, p, frr, far = evaluate_micro(result_dict, args.ratio_pos_neg)
    logger.info(f"Feature {feat} & VTA {vta}: ")
    logger.info(f"Overall segment-level performance, "
                f"Recall: {r:.2%}, Precision: {p:.2%}, F1: {2 * r * p / (r + p):.2%}")

    r_macro, p_macro, cnt = evaluate_macro(result_dict, meta_pairs)
    logger.info(f"query set cnt {cnt}, query macro-Recall: {r_macro:.2%}, "
                f"query macro-Precision: {p_macro:.2%}, "
                f"F1: {2 * r_macro * p_macro / (r_macro + p_macro):.2%}")
