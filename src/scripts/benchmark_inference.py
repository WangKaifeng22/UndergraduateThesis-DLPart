import argparse
import gc
import json
import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch

from utils.fourier_model_utils import build_fourier_deeponet_variant, is_original_fourier_deeponet_config


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _read_model_config(model_path: str, model_type: str) -> Tuple[str, Dict[str, Any], Dict[str, Any], bool]:
    model_init_kwargs = None
    data_cfg = None
    model_type_raw = model_type
    is_original = False

    model_config_path = os.path.join(os.path.dirname(model_path), "model_config.json")
    if os.path.exists(model_config_path):
        try:
            with open(model_config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)
            model_type_raw = model_config.get("model_type", model_type)
            model_init_kwargs = model_config.get("model_init_kwargs")
            data_cfg = model_config.get("data", None)
            is_original = is_original_fourier_deeponet_config(model_config)
            print(f"Loaded model config from: {model_config_path}")
            print(f"Auto model_type from config: {model_type_raw}")
            print(f"Auto is_original from config: {is_original}")
        except Exception as exc:
            print(f"Warning: failed to read model_config.json: {exc}")

    model_type_alias = {
        "FourierDeepONet": "FourierDeepONet",
        "BranchTrunkFlower": "BranchTrunkFlower",
        "InversionNet": "InversionNet",
        "NIO": "NIO",
        "NIOUltrasoundCTAbl": "NIO",
    }
    model_type_final = model_type_alias.get(model_type_raw, model_type_raw)
    return model_type_final, model_init_kwargs, data_cfg, is_original


def _load_test_data(
    model_type: str,
    model_init_kwargs: Dict[str, Any],
    data_cfg: Dict[str, Any],
    batch_size: int,
    split_ratio: float,
    total_data_num: int,
    has_ground_truth: bool,
    cache_h5_path: str,
    cache_meta_path: str,
):
    import h5py
    from test.test import _infer_sample_count, _load_full_h5_test_set, _load_full_h5_test_set_nio, _load_h5_meta
    from train.train import samples_per_config as train_samples_per_config

    h5_meta = _load_h5_meta(cache_meta_path)

    if isinstance(h5_meta, dict):
        if cache_h5_path is None and h5_meta.get("cache_h5_path"):
            cache_h5_path = h5_meta.get("cache_h5_path")

    if cache_h5_path is not None:
        if data_cfg is None:
            data_cfg = {}
        data_cfg["cache_h5_path"] = cache_h5_path

    if data_cfg is None:
        raise ValueError("benchmark currently requires data config from model_config.json or --cache-h5-path.")

    if model_type in {"FourierDeepONet", "BranchTrunkFlower"}:
        if total_data_num <= 0 and isinstance(h5_meta, dict):
            total_data_num = int(h5_meta.get("num_samples", total_data_num))
        if total_data_num <= 0 and data_cfg.get("cache_h5_path"):
            with h5py.File(data_cfg["cache_h5_path"], "r") as f:
                total_data_num = int(f["X_branch"].shape[0])
        if has_ground_truth is False:
            data_cfg["split_ratio"] = 0.0

        X_test, y_true_orig = _load_full_h5_test_set(
            data_cfg,
            is_deeponet=True,
            batch_size=batch_size,
            total_data_num=total_data_num,
        )
        split_ratio = data_cfg.get("split_ratio", split_ratio)

    elif model_type == "NIO":
        grid_npy_path = None
        if isinstance(model_init_kwargs, dict):
            grid_npy_path = model_init_kwargs.get("grid_npy_path")
        if grid_npy_path is None:
            grid_npy_path = data_cfg.get("grid_npy_path")
        if not grid_npy_path:
            raise ValueError("NIO benchmark requires 'grid_npy_path' in model_config.json.")

        if total_data_num <= 0 and isinstance(h5_meta, dict):
            total_data_num = int(h5_meta.get("num_samples", total_data_num))
        if total_data_num <= 0 and data_cfg.get("cache_h5_path"):
            with h5py.File(data_cfg["cache_h5_path"], "r") as f:
                total_data_num = int(f["X_branch"].shape[0])
        if has_ground_truth is False:
            data_cfg["split_ratio"] = 0.0

        X_test, y_true_orig = _load_full_h5_test_set_nio(
            data_cfg,
            batch_size=batch_size,
            grid_npy_path=grid_npy_path,
            total_data_num=total_data_num,
        )
        split_ratio = data_cfg.get("split_ratio", split_ratio)

    elif model_type == "InversionNet":
        samples_per_config = train_samples_per_config
        if isinstance(data_cfg, dict):
            samples_per_config = int(data_cfg.get("samples_per_config", samples_per_config))

        if total_data_num <= 0 and isinstance(h5_meta, dict):
            total_data_num = int(h5_meta.get("num_samples", total_data_num))
        if total_data_num <= 0 and data_cfg.get("cache_h5_path"):
            with h5py.File(data_cfg["cache_h5_path"], "r") as f:
                total_data_num = int(f["X_branch"].shape[0])
        if has_ground_truth is False:
            data_cfg["split_ratio"] = 0.0

        X_test, y_true_orig = _load_full_h5_test_set(
            data_cfg,
            is_deeponet=False,
            batch_size=batch_size,
            total_data_num=total_data_num,
        )
        _ = samples_per_config
        split_ratio = data_cfg.get("split_ratio", split_ratio)
    else:
        raise ValueError(f"Unknown model_type={model_type!r}")

    sample_count = len(y_true_orig)
    if sample_count <= 0:
        sample_count = _infer_sample_count(X_test, model_type)

    return X_test, sample_count, split_ratio


def _build_model(model_type: str, model_init_kwargs: Dict[str, Any], X_test, device: torch.device, is_original: bool = False):
    from models.InversionNet import InversionNet
    from models.model_BranchTrunkFlower import BranchTrunkFlower
    from utils.nio_build_utils import (
        extract_nio_build_kwargs,
        resolve_nio_branch_encoder_cls,
        resolve_nio_branch_encoder_kwargs,
    )
    from train.train_NIO import build_nio

    if model_type in {"FourierDeepONet", "BranchTrunkFlower"}:
        if isinstance(model_init_kwargs, dict):
            net = BranchTrunkFlower(**model_init_kwargs) if model_type == "BranchTrunkFlower" else build_fourier_deeponet_variant(model_init_kwargs, original=is_original)
        else:
            trunk_dim = X_test[1].shape[1]
            if model_type == "BranchTrunkFlower":
                net = BranchTrunkFlower(
                    num_parameter=trunk_dim,
                    width=96,
                    Tx=32,
                    Rx=32,
                    T_steps=1900,
                    H=80,
                    W=80,
                    lifting_dim=96,
                    n_levels=4,
                    num_heads=32,
                    boundary_condition_types=["ZEROS"],
                    dropout_rate=0.0,
                    regularization=["l2", 3e-6],
                    channel_lift_first=True,
                )
            else:
                net = build_fourier_deeponet_variant(
                    trunk_dim=trunk_dim,
                    original=is_original,
                    width=64,
                    modes1=16,
                    modes2=16,
                    regularization=["l2", 3e-6],
                    merge_operation="mul",
                )
    elif model_type == "InversionNet":
        if isinstance(model_init_kwargs, dict):
            net = InversionNet(**model_init_kwargs)
        else:
            net = InversionNet(dim0=64, dim1=64, dim2=64, dim3=128, dim4=256, dim5=512, regularization=["l2", 3e-6])
    elif model_type == "NIO":
        seed = 114514
        usct_time_steps = int(X_test[0].shape[-1])
        nio_kwargs = extract_nio_build_kwargs(model_init_kwargs)
        branch_encoder_cls = resolve_nio_branch_encoder_cls(model_init_kwargs)
        branch_encoder_kwargs = resolve_nio_branch_encoder_kwargs(model_init_kwargs)
        if isinstance(model_init_kwargs, dict):
            usct_time_steps = int(model_init_kwargs.get("usct_time_steps", usct_time_steps))
        net = build_nio(
            seed=seed,
            usct_time_steps=usct_time_steps,
            device=device,
            branch_encoder_cls=branch_encoder_cls,
            branch_encoder_kwargs=branch_encoder_kwargs,
            **nio_kwargs,
        )
    else:
        raise ValueError(f"Unknown model_type={model_type!r}")

    net.to(device)
    return net


def _run_benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    model_type, model_init_kwargs, data_cfg, is_original = _read_model_config(args.model_path, args.model_type)
    X_test, sample_count, split_ratio = _load_test_data(
        model_type=model_type,
        model_init_kwargs=model_init_kwargs,
        data_cfg=data_cfg,
        batch_size=args.batch_size,
        split_ratio=args.split_ratio,
        total_data_num=args.total_data_num,
        has_ground_truth=args.has_ground_truth,
        cache_h5_path=args.cache_h5_path,
        cache_meta_path=args.cache_meta_path,
    )

    print(f"Resolved split_ratio: {split_ratio}")
    print(f"Loaded test samples: {sample_count}")

    if sample_count <= 0:
        raise ValueError("No test samples found.")

    net = _build_model(model_type, model_init_kwargs, X_test, device, is_original=is_original)

    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        net.load_state_dict(checkpoint["model_state_dict"])
    else:
        net.load_state_dict(checkpoint)
    net.eval()

    num_batches = sample_count // args.batch_size
    if sample_count % args.batch_size != 0:
        num_batches += 1

    nio_grid_tensor = None
    if model_type == "NIO":
        nio_grid_tensor = torch.as_tensor(X_test[1], dtype=torch.float32, device=device)

    def run_one_iter(iter_idx: int, timing_scope: str):
        batch_idx = iter_idx % num_batches
        start = batch_idx * args.batch_size
        end = min((batch_idx + 1) * args.batch_size, sample_count)

        if model_type in {"FourierDeepONet", "BranchTrunkFlower"}:
            if timing_scope == "end2end":
                _sync_if_cuda(device)
                t0 = time.perf_counter()
                branch_batch = torch.as_tensor(X_test[0][start:end]).to(device)
                trunk_batch = torch.as_tensor(X_test[1][start:end]).to(device)
                outputs = net((branch_batch, trunk_batch))
                _sync_if_cuda(device)
                t1 = time.perf_counter()
            else:
                branch_batch = torch.as_tensor(X_test[0][start:end]).to(device)
                trunk_batch = torch.as_tensor(X_test[1][start:end]).to(device)
                _sync_if_cuda(device)
                t0 = time.perf_counter()
                outputs = net((branch_batch, trunk_batch))
                _sync_if_cuda(device)
                t1 = time.perf_counter()
            del branch_batch, trunk_batch
        elif model_type == "NIO":
            if timing_scope == "end2end":
                _sync_if_cuda(device)
                t0 = time.perf_counter()
                branch_batch = torch.as_tensor(X_test[0][start:end], dtype=torch.float32, device=device)
                outputs = net(branch_batch, nio_grid_tensor)
                _sync_if_cuda(device)
                t1 = time.perf_counter()
            else:
                branch_batch = torch.as_tensor(X_test[0][start:end], dtype=torch.float32, device=device)
                _sync_if_cuda(device)
                t0 = time.perf_counter()
                outputs = net(branch_batch, nio_grid_tensor)
                _sync_if_cuda(device)
                t1 = time.perf_counter()
            del branch_batch
        else:
            if timing_scope == "end2end":
                _sync_if_cuda(device)
                t0 = time.perf_counter()
                x_batch = torch.as_tensor(X_test[start:end]).to(device)
                outputs = net(x_batch)
                _sync_if_cuda(device)
                t1 = time.perf_counter()
            else:
                x_batch = torch.as_tensor(X_test[start:end]).to(device)
                _sync_if_cuda(device)
                t0 = time.perf_counter()
                outputs = net(x_batch)
                _sync_if_cuda(device)
                t1 = time.perf_counter()
            del x_batch

        del outputs
        return (t1 - t0), (end - start)

    with torch.no_grad():
        for i in range(args.warmup_iters):
            run_one_iter(i, args.timing_scope)

        times = []
        samples = []
        for i in range(args.measure_iters):
            dt, n = run_one_iter(i, args.timing_scope)
            times.append(dt)
            samples.append(n)

    times_arr = np.asarray(times, dtype=np.float64)
    samples_arr = np.asarray(samples, dtype=np.float64)

    total_time = float(np.sum(times_arr))
    total_samples = float(np.sum(samples_arr))
    mean_latency_ms = float(np.mean(times_arr) * 1000.0)
    std_latency_ms = float(np.std(times_arr) * 1000.0)
    p50_latency_ms = float(np.percentile(times_arr, 50) * 1000.0)
    p95_latency_ms = float(np.percentile(times_arr, 95) * 1000.0)
    min_latency_ms = float(np.min(times_arr) * 1000.0)
    max_latency_ms = float(np.max(times_arr) * 1000.0)
    throughput = (total_samples / total_time) if total_time > 0 else 0.0

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.model_path), "benchmark_result")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "benchmark_report.txt")

    lines = [
        "Inference Benchmark Report",
        "=" * 30,
        f"Model: {args.model_path}",
        f"Model Type: {model_type}",
        f"Device: {device}",
        f"Timing Scope: {args.timing_scope}",
        f"Batch Size: {args.batch_size}",
        f"Warmup Iters: {args.warmup_iters}",
        f"Measure Iters: {args.measure_iters}",
        f"Mean Latency: {mean_latency_ms:.4f} ms/batch",
        f"Std Latency: {std_latency_ms:.4f} ms/batch",
        f"P50 Latency: {p50_latency_ms:.4f} ms/batch",
        f"P95 Latency: {p95_latency_ms:.4f} ms/batch",
        f"Min Latency: {min_latency_ms:.4f} ms/batch",
        f"Max Latency: {max_latency_ms:.4f} ms/batch",
        f"Throughput: {throughput:.4f} samples/s",
    ]

    print("\n".join(lines))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved benchmark report to: {report_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Benchmark model inference latency and throughput.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .pt model checkpoint")
    parser.add_argument("--model-type", type=str, default="FourierDeepONet", help="Model type override")
    parser.add_argument("--batch-size", type=int, default=32, help="Benchmark batch size")
    parser.add_argument("--split-ratio", type=float, default=0.9, help="Fallback split ratio if needed")
    parser.add_argument("--total-data-num", type=int, default=50000, help="Total dataset size (0 means auto infer)")
    parser.add_argument("--cache-h5-path", type=str, default="/home/wkf/kwave-python/dataset/dataset_shuffle_0.140625-0.453125.h5", help="Optional override for cached test h5")
    parser.add_argument("--cache-meta-path", type=str, default="/home/wkf/kwave-python/dataset/dataset_shuffle_0.140625-0.453125_meta.json", help="Optional path to dataset meta json")
    parser.add_argument("--has-ground-truth", type=str2bool, default=True, help="Whether test set includes ground truth")
    parser.add_argument("--warmup-iters", type=int, default=20, help="Number of warmup iterations")
    parser.add_argument("--measure-iters", type=int, default=100, help="Number of measured iterations")
    parser.add_argument(
        "--timing-scope",
        type=str,
        default="forward",
        choices=["forward", "end2end"],
        help="forward: only net forward on prepared device tensor; end2end: includes per-batch tensor creation/to(device)",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for benchmark report")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    _run_benchmark(args)
