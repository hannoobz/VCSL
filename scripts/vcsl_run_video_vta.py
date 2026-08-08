import argparse
import os
import sys
from itertools import product, islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))

import pandas as pd
import torch
from vcsl import *
from torch.utils.data import DataLoader
from loguru import logger
import vcsl.yolov5.models as _v_models
import vcsl.yolov5.models.yolo as _v_models_yolo
import vcsl.yolov5.models.common as _v_models_common
import vcsl.yolov5.models.experimental as _v_models_experimental
import vcsl.yolov5.utils as _v_utils
import vcsl.yolov5.utils.general as _v_utils_general
import vcsl.yolov5.utils.datasets as _v_utils_datasets
import vcsl.yolov5.utils.google_utils as _v_utils_google_utils
import vcsl.yolov5.utils.torch_utils as _v_utils_torch_utils
import vcsl.yolov5.utils.activations as _v_utils_activations

if __name__ == '__main__':
    _orig_torch_load = torch.load
    def _torch_load_default_full_unpickle(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _torch_load_default_full_unpickle


    sys.modules.setdefault("models", _v_models)
    sys.modules.setdefault("models.yolo", _v_models_yolo)
    sys.modules.setdefault("models.common", _v_models_common)
    sys.modules.setdefault("models.experimental", _v_models_experimental)
    sys.modules.setdefault("utils", _v_utils)
    sys.modules.setdefault("utils.general", _v_utils_general)
    sys.modules.setdefault("utils.datasets", _v_utils_datasets)
    sys.modules.setdefault("utils.google_utils", _v_utils_google_utils)
    sys.modules.setdefault("utils.torch_utils", _v_utils_torch_utils)
    sys.modules.setdefault("utils.activations", _v_utils_activations)

    parser = argparse.ArgumentParser()

    parser.add_argument("--query-file", "-Q", type=str)
    parser.add_argument("--reference-file", "-G", type=str)
    parser.add_argument("--pair-file", type=str,
                         default=str(Path(__file__).resolve().parents[1] / "dataset" / "pair_file_test.csv"))

    parser.add_argument("--input-store", type=str, default="local")
    parser.add_argument("--input-root", type=str, required=True,
                         help="dir of <ref_id>-<query_id>.npy similarity maps from vcsl_run_video_sim.py")

    parser.add_argument("--oss-config", type=str, default='~/ossutilconfig-copyright')
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--data-workers", type=int, default=4)
    parser.add_argument("--request-workers", type=int, default=4)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--output-store", type=str, default="local")

    parser.add_argument("--alignment-method", type=str, default="TN", help="DTW, DP, TN, HV, SPD")

    parser.add_argument("--min-length", type=int, default=5)
    parser.add_argument("--sum-sim", type=float, default=10.)
    parser.add_argument("--ave-sim", type=float, default=0.3)
    parser.add_argument("--min-sim", type=float, default=0.3)

    parser.add_argument("--max-path", type=int, default=10)
    parser.add_argument("--discontinue", type=int, default=3)
    parser.add_argument("--max-iou", type=float, default=0.3)

    parser.add_argument("--diagonal-thres", type=int, default=10)
    parser.add_argument("--tn-top-K", type=int, default=3)
    parser.add_argument("--tn-max-step", type=int, default=10)

    parser.add_argument("--spd-model-path", type=str, help="path to a YOLOv5 .pt SPD checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0, only used by SPD")
    parser.add_argument("--spd-conf-thres", type=float, default=0.1)

    parser.add_argument("--params-file", type=str)
    parser.add_argument("--result-file", default="pred.json", type=str)

    args = parser.parse_args()

    pairs, files_dict, query, reference = None, None, None, None
    if args.pair_file:
        df = pd.read_csv(args.pair_file)
        pairs = df[['query_id', 'reference_id']].values.tolist()
        data_list = [(f"{p[0]}-{p[1]}", f"{p[0]}-{p[1]}") for p in pairs]
    else:
        query = pd.read_csv(args.query_file)[['uuid']].values.tolist()
        reference = pd.read_csv(args.reference_file)[['uuid']].values.tolist()
        pairs = product(query, reference)
        data_list = [(f"{p[0]}-{p[1]}", f"{p[0]}-{p[1]}") for p in pairs]

    config = dict()
    if args.input_store == 'oss':
        config['oss_config'] = args.oss_config

    dataset = ItemDataset(data_list,
                           store_type=args.input_store,
                           data_type=DataType.NUMPY.type_name,
                           root=args.input_root,
                           trans_key_func=lambda x: x + '.npy',
                           **config)

    logger.info(f"Data to run {len(dataset)}")

    loader_kwargs = dict(collate_fn=lambda x: x, batch_size=args.batch_size,
                          num_workers=args.data_workers)
    if args.data_workers > 0 and sys.platform != "win32":
        loader_kwargs["multiprocessing_context"] = "fork"
    loader = DataLoader(dataset, **loader_kwargs)

    if args.alignment_method.startswith('DTW'):
        model_config = dict(discontinue=args.discontinue, min_sim=args.min_sim,
                             min_length=args.min_length, max_iou=args.max_iou)
    elif args.alignment_method.startswith('TN'):
        model_config = dict(tn_max_step=args.tn_max_step, tn_top_k=args.tn_top_K, max_path=args.max_path,
                             min_sim=args.min_sim, min_length=args.min_length, max_iou=args.max_iou)
    elif args.alignment_method.startswith('DP'):
        model_config = dict(discontinue=args.discontinue, min_sim=args.min_sim, ave_sim=args.ave_sim,
                             min_length=args.min_length, diagonal_thres=args.diagonal_thres)
    elif args.alignment_method.startswith('HV'):
        model_config = dict(min_sim=args.min_sim, iou_thresh=args.max_iou)
    elif args.alignment_method.startswith('SPD'):
        if not args.spd_model_path:
            raise ValueError("SPD requires --spd-model-path pointing at a YOLOv5 .pt checkpoint "
                              "(see vendor/spd_models.txt)")
        model_config = dict(model_path=args.spd_model_path, conf_thresh=args.spd_conf_thres, device=args.device)
    else:
        raise ValueError(f"Unknown VTA method: {args.alignment_method}")

    if args.params_file:
        reader = build_reader(args.input_store, DataType.JSON.type_name, **config)
        param_result = reader.read(args.params_file)
        model_config = param_result['best']['param']
        logger.info("best param {}", model_config)

    model = build_vta_model(method=args.alignment_method, concurrency=args.request_workers, **model_config)

    total_result = dict()
    for batch_data in islice(loader, 0, None):
        logger.info("data cnt: {}, {}", len(batch_data), batch_data[0][0])
        batch_result = model.forward_sim(batch_data)
        logger.info("result cnt: {}", len(batch_result))
        for pair_id, result in batch_result:
            total_result[pair_id] = result

    output_store = args.output_store
    if output_store == 'local' and not os.path.exists(args.output_root):
        os.makedirs(args.output_root, exist_ok=True)
    writer = build_writer(output_store, DataType.JSON.type_name, **config)
    writer.write(os.path.join(args.output_root, args.result_file), total_result)
