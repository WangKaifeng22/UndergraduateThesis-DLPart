import argparse
import csv
import gc
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

os.environ['DDE_BACKEND'] = 'pytorch'

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from InversionNet import InversionNet
from model_BranchTrunkFlower import BranchTrunkFlower
from model_Unet_CNN import FourierDeepONet
from nio_build_utils import (
    extract_nio_build_kwargs as _extract_nio_build_kwargs,
    resolve_nio_branch_encoder_cls as _resolve_nio_branch_encoder_cls,
    resolve_nio_branch_encoder_kwargs as _resolve_nio_branch_encoder_kwargs,
)
from my_test import (
    _infer_sample_count,
    _load_full_h5_test_set,
    _load_full_h5_test_set_nio,
    _load_h5_meta,
    _metric_mean_std,
    compute_pcc_numpy,
    compute_ssim_numpy,
)
from my_train import samples_per_config as train_samples_per_config
from train_NIO import build_nio
from utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if text in {'0', 'false', 'f', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def _read_model_config(model_path: str, model_type: str):
    model_init_kwargs = None
    data_cfg = None
    model_type_raw = model_type

    model_config_path = os.path.join(os.path.dirname(model_path), 'model_config.json')
    if os.path.exists(model_config_path):
        try:
            with open(model_config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
            model_type_raw = model_config.get('model_type', model_type)
            model_init_kwargs = model_config.get('model_init_kwargs')
            data_cfg = model_config.get('data', None)
            print(f'Loaded model config from: {model_config_path}')
            print(f'Auto model_type from config: {model_type_raw}')
        except Exception as exc:
            print(f'Warning: failed to read model_config.json: {exc}')

    model_type_alias = {
        'FourierDeepONet': 'FourierDeepONet',
        'BranchTrunkFlower': 'BranchTrunkFlower',
        'InversionNet': 'InversionNet',
        'NIO': 'NIO',
        'NIOUltrasoundCTAbl': 'NIO',
    }
    model_type_final = model_type_alias.get(model_type_raw, model_type_raw)
    return model_type_final, model_init_kwargs, data_cfg


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
    h5_meta = _load_h5_meta(cache_meta_path)

    if isinstance(h5_meta, dict):
        if cache_h5_path is None and h5_meta.get('cache_h5_path'):
            cache_h5_path = h5_meta.get('cache_h5_path')

    if cache_h5_path is not None:
        if data_cfg is None:
            data_cfg = {}
        data_cfg['cache_h5_path'] = cache_h5_path

    if data_cfg is None:
        raise ValueError('robustness eval requires data config from model_config.json or --cache-h5-path.')

    if model_type in {'FourierDeepONet', 'BranchTrunkFlower'}:
        if total_data_num <= 0 and isinstance(h5_meta, dict):
            total_data_num = int(h5_meta.get('num_samples', total_data_num))
        if total_data_num <= 0 and data_cfg.get('cache_h5_path'):
            with h5py.File(data_cfg['cache_h5_path'], 'r') as f:
                total_data_num = int(f['X_branch'].shape[0])
        if has_ground_truth is False:
            data_cfg['split_ratio'] = 0.0

        X_test, y_true_orig = _load_full_h5_test_set(
            data_cfg,
            is_deeponet=True,
            batch_size=batch_size,
            total_data_num=total_data_num,
        )
        split_ratio = data_cfg.get('split_ratio', split_ratio)

    elif model_type == 'NIO':
        grid_npy_path = None
        if isinstance(model_init_kwargs, dict):
            grid_npy_path = model_init_kwargs.get('grid_npy_path')
        if grid_npy_path is None:
            grid_npy_path = data_cfg.get('grid_npy_path')
        if not grid_npy_path:
            raise ValueError("NIO robustness eval requires 'grid_npy_path' in model_config.json.")

        if total_data_num <= 0 and isinstance(h5_meta, dict):
            total_data_num = int(h5_meta.get('num_samples', total_data_num))
        if total_data_num <= 0 and data_cfg.get('cache_h5_path'):
            with h5py.File(data_cfg['cache_h5_path'], 'r') as f:
                total_data_num = int(f['X_branch'].shape[0])
        if has_ground_truth is False:
            data_cfg['split_ratio'] = 0.0

        X_test, y_true_orig = _load_full_h5_test_set_nio(
            data_cfg,
            batch_size=batch_size,
            grid_npy_path=grid_npy_path,
            total_data_num=total_data_num,
        )
        split_ratio = data_cfg.get('split_ratio', split_ratio)

    elif model_type == 'InversionNet':
        samples_per_config = train_samples_per_config
        if isinstance(data_cfg, dict):
            samples_per_config = int(data_cfg.get('samples_per_config', samples_per_config))

        if total_data_num <= 0 and isinstance(h5_meta, dict):
            total_data_num = int(h5_meta.get('num_samples', total_data_num))
        if total_data_num <= 0 and data_cfg.get('cache_h5_path'):
            with h5py.File(data_cfg['cache_h5_path'], 'r') as f:
                total_data_num = int(f['X_branch'].shape[0])
        if has_ground_truth is False:
            data_cfg['split_ratio'] = 0.0

        X_test, y_true_orig = _load_full_h5_test_set(
            data_cfg,
            is_deeponet=False,
            batch_size=batch_size,
            total_data_num=total_data_num,
        )
        _ = samples_per_config

    else:
        raise ValueError(f'Unknown model_type={model_type!r}')

    sample_count = len(y_true_orig)
    if sample_count <= 0:
        sample_count = _infer_sample_count(X_test, model_type)

    return X_test, y_true_orig, sample_count, split_ratio


def _build_model(model_type: str, model_init_kwargs: Dict[str, Any], X_test, sosmap_size, device: torch.device):
    if model_type in {'FourierDeepONet', 'BranchTrunkFlower'}:
        if isinstance(model_init_kwargs, dict):
            net = BranchTrunkFlower(**model_init_kwargs) if model_type == 'BranchTrunkFlower' else FourierDeepONet(**model_init_kwargs)
        else:
            trunk_dim = X_test[1].shape[1]
            if model_type == 'BranchTrunkFlower':
                net = BranchTrunkFlower(
                    num_parameter=trunk_dim,
                    width=96,
                    Tx=32,
                    Rx=32,
                    T_steps=1900,
                    H=sosmap_size[0],
                    W=sosmap_size[1],
                    lifting_dim=96,
                    n_levels=4,
                    num_heads=32,
                    boundary_condition_types=['ZEROS'],
                    dropout_rate=0.0,
                    regularization=['l2', 3e-6],
                    channel_lift_first=True,
                )
            else:
                net = FourierDeepONet(
                    num_parameter=trunk_dim,
                    width=64,
                    modes1=16,
                    modes2=16,
                    regularization=['l2', 3e-6],
                    merge_operation='mul',
                )
    elif model_type == 'InversionNet':
        if isinstance(model_init_kwargs, dict):
            net = InversionNet(**model_init_kwargs)
        else:
            net = InversionNet(dim0=64, dim1=64, dim2=64, dim3=128, dim4=256, dim5=512, regularization=['l2', 3e-6])
    elif model_type == 'NIO':
        seed = 114514
        usct_time_steps = int(X_test[0].shape[-1])
        nio_kwargs = _extract_nio_build_kwargs(model_init_kwargs)
        branch_encoder_cls = _resolve_nio_branch_encoder_cls(model_init_kwargs)
        branch_encoder_kwargs = _resolve_nio_branch_encoder_kwargs(model_init_kwargs)
        if isinstance(model_init_kwargs, dict):
            usct_time_steps = int(model_init_kwargs.get('usct_time_steps', usct_time_steps))

        net = build_nio(
            seed=seed,
            usct_time_steps=usct_time_steps,
            device=device,
            branch_encoder_cls=branch_encoder_cls,
            branch_encoder_kwargs=branch_encoder_kwargs,
            **nio_kwargs,
        )
    else:
        raise ValueError(f"Unknown model_type={model_type!r}. Expected 'FourierDeepONet', 'BranchTrunkFlower', 'InversionNet', or 'NIO'.")

    net.to(device)
    return net


def _as_float32(array):
    return np.asarray(array, dtype=np.float32)


def _noise_scale(array: np.ndarray, scale_mode: str) -> np.ndarray:
    if scale_mode == 'unit':
        return np.array(1.0, dtype=np.float32)
    if scale_mode != 'batch_std':
        raise ValueError(f'Unknown scale_mode={scale_mode!r}')
    scale = np.std(array, axis=0, keepdims=True)
    return np.asarray(np.maximum(scale, 1e-8), dtype=np.float32)


def _add_gaussian_noise(array: np.ndarray, sigma: float, rng: np.random.Generator, scale_mode: str, clip_inputs: bool):
    array = _as_float32(array)
    if sigma <= 0:
        return array

    scale = _noise_scale(array, scale_mode)
    noise = rng.normal(loc=0.0, scale=1.0, size=array.shape).astype(np.float32)
    noisy = array + noise * (sigma * scale)

    if clip_inputs:
        arr_min = np.min(array)
        arr_max = np.max(array)
        noisy = np.clip(noisy, arr_min, arr_max)

    return noisy.astype(np.float32, copy=False)


def _scenario_list(model_type: str):
    if model_type in {'FourierDeepONet', 'BranchTrunkFlower'}:
        return ['clean', 'branch_noise', 'trunk_noise', 'branch_plus_trunk_noise']
    return ['clean', 'branch_noise']


def _scenario_supported(model_type: str, scenario: str) -> bool:
    if model_type in {'FourierDeepONet', 'BranchTrunkFlower'}:
        return scenario in {'clean', 'branch_noise', 'trunk_noise', 'branch_plus_trunk_noise'}
    return scenario in {'clean', 'branch_noise'}


def _prepare_batch_inputs(model_type: str, X_test, start: int, end: int, scenario: str, sigma: float, rng, clip_inputs: bool):
    if model_type in {'FourierDeepONet', 'BranchTrunkFlower'}:
        branch_np = _as_float32(X_test[0][start:end])
        trunk_np = _as_float32(X_test[1][start:end])
        if scenario in {'branch_noise', 'branch_plus_trunk_noise'}:
            branch_np = _add_gaussian_noise(branch_np, sigma, rng, scale_mode='batch_std', clip_inputs=clip_inputs)
        if scenario in {'trunk_noise', 'branch_plus_trunk_noise'}:
            trunk_np = _add_gaussian_noise(trunk_np, sigma, rng, scale_mode='batch_std', clip_inputs=clip_inputs)
        return (torch.as_tensor(branch_np, dtype=torch.float32, device=device), torch.as_tensor(trunk_np, dtype=torch.float32, device=device))

    if model_type == 'NIO':
        branch_np = _as_float32(X_test[0][start:end])
        if scenario == 'branch_noise':
            branch_np = _add_gaussian_noise(branch_np, sigma, rng, scale_mode='batch_std', clip_inputs=clip_inputs)
        return torch.as_tensor(branch_np, dtype=torch.float32, device=device)

    x_np = _as_float32(X_test[start:end])
    if scenario == 'branch_noise':
        x_np = _add_gaussian_noise(x_np, sigma, rng, scale_mode='batch_std', clip_inputs=clip_inputs)
    return torch.as_tensor(x_np, dtype=torch.float32, device=device)


def _run_inference(model_type: str, net, X_test, sample_count: int, batch_size: int, scenario: str, sigma: float, seed: int, clip_inputs: bool):
    y_pred_list = []
    num_batches = sample_count // batch_size
    if sample_count % batch_size != 0:
        num_batches += 1

    nio_grid_tensor = None
    if model_type == 'NIO':
        nio_grid_tensor = torch.as_tensor(X_test[1], dtype=torch.float32, device=device)

    rng = np.random.default_rng(seed)

    with torch.no_grad():
        for i in range(num_batches):
            start = batch_size * i
            end = min(batch_size * (i + 1), sample_count)

            if model_type in {'FourierDeepONet', 'BranchTrunkFlower'}:
                inputs = _prepare_batch_inputs(model_type, X_test, start, end, scenario, sigma, rng, clip_inputs)
                outputs = net(inputs)
            elif model_type == 'NIO':
                branch_batch = _prepare_batch_inputs(model_type, X_test, start, end, scenario, sigma, rng, clip_inputs)
                outputs = net(branch_batch, nio_grid_tensor)
            else:
                x_batch = _prepare_batch_inputs(model_type, X_test, start, end, scenario, sigma, rng, clip_inputs)
                outputs = net(x_batch)

            y_pred_list.append(outputs.detach().cpu().numpy())
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    y_pred_norm = np.concatenate(y_pred_list, axis=0).squeeze()
    if y_pred_norm.ndim == 2:
        y_pred_norm = np.expand_dims(y_pred_norm, 0)
    return y_pred_norm


def _evaluate_predictions(y_true_orig, y_pred_norm, has_ground_truth: bool):
    if not has_ground_truth:
        return {}

    y_true_real = minmax_denormalize(y_true_orig, VMIN, VMAX, scale=2)
    y_pred_real = minmax_denormalize(y_pred_norm, VMIN, VMAX, scale=2)

    per_sample_mae = np.mean(np.abs(y_true_real - y_pred_real), axis=(1, 2))
    per_sample_rmse = np.sqrt(np.mean((y_true_real - y_pred_real) ** 2, axis=(1, 2)))

    l2_list = []
    for k in range(len(y_true_real)):
        norm_true = np.linalg.norm(y_true_real[k])
        norm_diff = np.linalg.norm(y_true_real[k] - y_pred_real[k])
        l2_list.append(norm_diff / norm_true if norm_true > 0 else 0)
    l2_rel_list = np.asarray(l2_list, dtype=np.float64)

    ssim_scores = compute_ssim_numpy(y_true_real, y_pred_real, data_range=VMAX - VMIN)
    pcc_scores = compute_pcc_numpy(y_true_real, y_pred_real)

    mae_mean, mae_std = _metric_mean_std(per_sample_mae)
    rmse_mean, rmse_std = _metric_mean_std(per_sample_rmse)
    ssim_mean, ssim_std = _metric_mean_std(ssim_scores)
    pcc_mean, pcc_std = _metric_mean_std(pcc_scores)
    l2_mean, l2_std = _metric_mean_std(l2_rel_list)

    return {
        'mae_mean': mae_mean,
        'mae_std': mae_std,
        'rmse_mean': rmse_mean,
        'rmse_std': rmse_std,
        'ssim_mean': ssim_mean,
        'ssim_std': ssim_std,
        'pcc_mean': pcc_mean,
        'pcc_std': pcc_std,
        'l2_mean': l2_mean,
        'l2_std': l2_std,
    }


def _plot_summary(result_rows: List[Dict[str, Any]], out_dir: str, model_type: str):
    if not result_rows:
        return

    os.makedirs(out_dir, exist_ok=True)

    metrics = ['mae_mean', 'rmse_mean', 'ssim_mean', 'pcc_mean', 'l2_mean']
    metric_titles = {
        'mae_mean': 'MAE',
        'rmse_mean': 'RMSE',
        'ssim_mean': 'SSIM',
        'pcc_mean': 'PCC',
        'l2_mean': 'L2 Relative Error',
    }

    scenario_order = ['clean', 'branch_noise', 'trunk_noise', 'branch_plus_trunk_noise']
    scenarios = [scenario for scenario in scenario_order if scenario in set(row['scenario'] for row in result_rows)]
    sigma_values = sorted(set(float(row['sigma']) for row in result_rows))

    for metric in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.2))
        for scenario in scenarios:
            xs = []
            ys = []
            yerr = []
            for sigma in sigma_values:
                rows = [row for row in result_rows if row['scenario'] == scenario and float(row['sigma']) == float(sigma)]
                if not rows:
                    continue
                xs.append(sigma)
                ys.append(float(np.mean([row[metric] for row in rows])))
                yerr.append(float(np.std([row[metric] for row in rows])))
            if xs:
                ax.errorbar(xs, ys, yerr=yerr, marker='o', capsize=3, linewidth=1.8, label=scenario)

        ax.set_xlabel('Sigma')
        ax.set_ylabel(metric_titles[metric])
        ax.set_title(f'{model_type} robustness: {metric_titles[metric]} vs Sigma')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{metric}.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize_rows(rows: Sequence[Dict[str, Any]]):
    summary = {}
    grouped: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row['scenario'], float(row['sigma'])), []).append(row)

    for (scenario, sigma), items in grouped.items():
        summary[(scenario, sigma)] = {
            'scenario': scenario,
            'sigma': sigma,
            'trials': len(items),
            'mae_mean': float(np.mean([item['mae_mean'] for item in items])),
            'mae_std': float(np.std([item['mae_mean'] for item in items])),
            'rmse_mean': float(np.mean([item['rmse_mean'] for item in items])),
            'rmse_std': float(np.std([item['rmse_mean'] for item in items])),
            'ssim_mean': float(np.mean([item['ssim_mean'] for item in items])),
            'ssim_std': float(np.std([item['ssim_mean'] for item in items])),
            'pcc_mean': float(np.mean([item['pcc_mean'] for item in items])),
            'pcc_std': float(np.std([item['pcc_mean'] for item in items])),
            'l2_mean': float(np.mean([item['l2_mean'] for item in items])),
            'l2_std': float(np.std([item['l2_mean'] for item in items])),
        }
    return list(summary.values())


def main(
    model_path,
    result_dir,
    model_type='FourierDeepONet',
    batch_size=32,
    split_ratio=0.0,
    total_data_num=0,
    sosmap_size=(80, 80),
    cache_h5_path=None,
    cache_meta_path=None,
    has_ground_truth=None,
    sigma_list=None,
    noise_trials=3,
    base_noise_seed=114514,
    clip_inputs=False,
    save_npy=False,
):
    if sigma_list is None:
        sigma_list = [0.0, 0.01, 0.05, 0.1, 0.2]
    sigma_list = [float(sigma) for sigma in sigma_list]
    if 0.0 not in sigma_list:
        sigma_list = [0.0] + sigma_list

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print('--- 1. Loading Config ---')
    model_init_kwargs = None
    model_config_path = os.path.join(os.path.dirname(model_path), 'model_config.json')
    data_cfg = None
    model_type_raw = model_type
    if os.path.exists(model_config_path):
        try:
            with open(model_config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
            model_type_raw = model_config.get('model_type', model_type)
            model_init_kwargs = model_config.get('model_init_kwargs')
            data_cfg = model_config.get('data', None)
            print(f'Loaded model config from: {model_config_path}')
            print(f'Auto model_type from config: {model_type_raw}')
        except Exception as exc:
            print(f'Warning: failed to read model_config.json: {exc}')

    h5_meta = _load_h5_meta(cache_meta_path)
    if isinstance(h5_meta, dict):
        if has_ground_truth is None and 'has_ground_truth' in h5_meta:
            has_ground_truth = bool(h5_meta.get('has_ground_truth'))
        if cache_h5_path is None and h5_meta.get('cache_h5_path'):
            cache_h5_path = h5_meta.get('cache_h5_path')
        if h5_meta.get('y_shape') and (has_ground_truth is False):
            shape_tail = h5_meta.get('y_shape')[1:]
            if len(shape_tail) == 2:
                sosmap_size = (int(shape_tail[0]), int(shape_tail[1]))

    if cache_h5_path is not None:
        if data_cfg is None:
            data_cfg = {}
        data_cfg['cache_h5_path'] = cache_h5_path
    if has_ground_truth is None and isinstance(data_cfg, dict):
        has_ground_truth = bool(data_cfg.get('has_ground_truth', True))
    if has_ground_truth is None:
        has_ground_truth = True
    if not has_ground_truth:
        raise ValueError('Robustness evaluation requires ground truth labels to compute metrics.')

    model_type_alias = {
        'FourierDeepONet': 'FourierDeepONet',
        'BranchTrunkFlower': 'BranchTrunkFlower',
        'InversionNet': 'InversionNet',
        'NIO': 'NIO',
        'NIOUltrasoundCTAbl': 'NIO',
    }
    model_type = model_type_alias.get(model_type_raw, model_type_raw)

    print('--- 2. Loading Data ---')
    X_test, y_true_orig, sample_count, split_ratio = _load_test_data(
        model_type=model_type,
        model_init_kwargs=model_init_kwargs,
        data_cfg=data_cfg,
        batch_size=batch_size,
        split_ratio=split_ratio,
        total_data_num=total_data_num,
        has_ground_truth=has_ground_truth,
        cache_h5_path=cache_h5_path,
        cache_meta_path=cache_meta_path,
    )

    if not has_ground_truth and y_true_orig is not None and y_true_orig.ndim >= 3:
        sosmap_size = (int(y_true_orig.shape[1]), int(y_true_orig.shape[2]))

    print(f'Test Data Shape: {y_true_orig.shape}')
    print('--- 3. Building Model ---')
    net = _build_model(model_type, model_init_kwargs, X_test, sosmap_size, device)

    print(f'--- 4. Loading Weights from {model_path} ---')
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            net.load_state_dict(checkpoint)
        print('Weights loaded successfully.')
    except Exception as exc:
        print(f'Error loading weights: {exc}')
        return

    net.eval()

    scenarios = [scenario for scenario in _scenario_list(model_type) if _scenario_supported(model_type, scenario)]
    if model_type in {'InversionNet', 'NIO'}:
        print('Note: only branch noise is evaluated for InversionNet and NIO.')

    result_rows = []
    out_dir = os.path.join(result_dir, 'robustness')
    os.makedirs(out_dir, exist_ok=True)

    print('--- 5. Robustness Sweep ---')
    scenario_seed_offset = {
        'clean': 0,
        'branch_noise': 1000,
        'trunk_noise': 2000,
        'branch_plus_trunk_noise': 3000,
    }
    for sigma in sigma_list:
        for scenario in scenarios:
            if scenario == 'clean' and float(sigma) != 0.0:
                continue
            if scenario != 'clean' and float(sigma) == 0.0:
                continue
            for trial in range(int(noise_trials)):
                trial_seed = int(base_noise_seed + scenario_seed_offset[scenario] + trial + int(float(sigma) * 100000))
                y_pred_norm = _run_inference(
                    model_type=model_type,
                    net=net,
                    X_test=X_test,
                    sample_count=sample_count,
                    batch_size=batch_size,
                    scenario=scenario,
                    sigma=float(sigma),
                    seed=trial_seed,
                    clip_inputs=clip_inputs,
                )
                metrics = _evaluate_predictions(y_true_orig, y_pred_norm, has_ground_truth)

                row = {
                    'model_type': model_type,
                    'scenario': scenario,
                    'sigma': float(sigma),
                    'trial': int(trial),
                    'sample_count': int(sample_count),
                    'has_ground_truth': bool(has_ground_truth),
                }
                row.update(metrics)
                result_rows.append(row)

                print(
                    f"[{scenario}] sigma={sigma:.6g} trial={trial + 1}/{noise_trials} "
                    f"MAE={row.get('mae_mean', float('nan')):.4f} "
                    f"RMSE={row.get('rmse_mean', float('nan')):.4f} "
                    f"SSIM={row.get('ssim_mean', float('nan')):.4f}"
                )

    summary_rows = _summarize_rows(result_rows)

    csv_path = os.path.join(out_dir, 'robustness_trials.csv')
    summary_csv_path = os.path.join(out_dir, 'robustness_summary.csv')
    _write_csv(csv_path, result_rows)
    _write_csv(summary_csv_path, summary_rows)

    json_path = os.path.join(out_dir, 'robustness_trials.json')
    summary_json_path = os.path.join(out_dir, 'robustness_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_rows, f, ensure_ascii=False, indent=2)
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    _plot_summary(summary_rows, out_dir=out_dir, model_type=model_type)

    if save_npy and has_ground_truth:
        print('Clean baseline was evaluated; raw npy export is intentionally skipped in robustness mode.')

    print(f'All robustness results saved to: {out_dir}')


def _parse_args():
    parser = argparse.ArgumentParser(description='Robustness evaluation with Gaussian noise on branch/trunk inputs.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to .pt checkpoint.')
    parser.add_argument('--result-dir', type=str, required=True, help='Directory to save robustness results.')
    parser.add_argument('--model-type', type=str, default='FourierDeepONet', help='Model type if not provided by model_config.json.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--split-ratio', type=float, default=0.9)
    parser.add_argument('--total-data-num', type=int, default=50000)
    parser.add_argument('--sosmap-size', type=int, nargs=2, default=(80, 80))
    parser.add_argument('--cache-h5-path', type=str, default="/home/wkf/kwave-python/dataset/dataset_shuffle_0.140625-0.453125.h5")
    parser.add_argument('--cache-meta-path', type=str, default="/home/wkf/kwave-python/dataset/dataset_shuffle_0.140625-0.453125_meta.json")
    parser.add_argument('--has-ground-truth', type=str2bool, default=True)
    parser.add_argument('--sigma-list', type=float, nargs='+', default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument('--noise-trials', type=int, default=3)
    parser.add_argument('--base-noise-seed', type=int, default=114514)
    parser.add_argument('--clip-inputs', type=str2bool, default=False)
    parser.add_argument('--save-npy', type=str2bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(
        model_path=args.model_path,
        result_dir=args.result_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        split_ratio=args.split_ratio,
        total_data_num=args.total_data_num,
        sosmap_size=tuple(args.sosmap_size),
        cache_h5_path=args.cache_h5_path,
        cache_meta_path=args.cache_meta_path,
        has_ground_truth=args.has_ground_truth,
        sigma_list=args.sigma_list,
        noise_trials=args.noise_trials,
        base_noise_seed=args.base_noise_seed,
        clip_inputs=args.clip_inputs,
        save_npy=args.save_npy,
    )