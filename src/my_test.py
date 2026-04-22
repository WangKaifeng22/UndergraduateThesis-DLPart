import gc
import os
import json
os.environ['DDE_BACKEND'] = 'pytorch'

import h5py
from matplotlib import colors
from model_Unet_CNN import FourierDeepONet
from InversionNet import InversionNet
from model_BranchTrunkFlower import BranchTrunkFlower
from nio_build_utils import (
    extract_nio_build_kwargs as _extract_nio_build_kwargs,
    resolve_nio_branch_encoder_cls as _resolve_nio_branch_encoder_cls,
    resolve_nio_branch_encoder_kwargs as _resolve_nio_branch_encoder_kwargs,
)
from train_NIO import build_nio
from multi_data import get_dataset as get_multi_dataset
from my_data import get_dataset as get_legacy_dataset
from my_train import samples_per_config as train_samples_per_config
from utils import *
# HDF5 backed dataset (lazy loading)
from h5_dataset import H5DeepONetDataset, H5DatasetConfig
from H5NIODataset import H5NIOConfig, H5NIODataset

# === 直接导入 skimage 用于 NumPy 计算 ===
try:
    from skimage.metrics import structural_similarity as ssim_skimage
except ImportError:
    print("Warning: scikit-image not found. SSIM calculation will be skipped.")
    print("Please install it via: pip install scikit-image")
    ssim_skimage = None

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_ssim_numpy(y_true, y_pred, data_range):
    """
    计算 SSIM (适配 Numpy 数组)
    """
    # 如果没有安装 skimage，直接返回全 0 分数，保证 mean/std 逻辑统一。
    if ssim_skimage is None:
        return np.zeros(len(y_true), dtype=np.float32)

    scores = []
    for i in range(len(y_true)):
        # y_true[i] shape: (sosmap_size[0], sosmap_size[1])
        score = ssim_skimage(
            y_true[i],
            y_pred[i],
            data_range=data_range,
            # 指定 channel_axis=None 表示输入是 (H, W) 的灰度图
            channel_axis=None
        )
        scores.append(score)
    if not scores:
        return np.zeros(1, dtype=np.float32)
    return np.asarray(scores, dtype=np.float32)


def compute_pcc_numpy(y_true, y_pred, eps=1e-8):
    """计算 PCC (皮尔逊相关系数)，返回逐样本分数。"""
    scores = []
    for i in range(len(y_true)):
        true_flat = y_true[i].reshape(-1)
        pred_flat = y_pred[i].reshape(-1)

        true_std = np.std(true_flat)
        pred_std = np.std(pred_flat)

        # 常量图像会导致分母趋近 0；这里做稳定处理。
        if true_std < eps and pred_std < eps:
            scores.append(1.0)
            continue
        if true_std < eps or pred_std < eps:
            scores.append(0.0)
            continue

        corr = np.corrcoef(true_flat, pred_flat)[0, 1]
        if np.isfinite(corr):
            scores.append(corr)

    if not scores:
        return np.zeros(1, dtype=np.float32)
    return np.asarray(scores, dtype=np.float32)


def _metric_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_sample_count(X_test, model_type):
    if model_type in {"FourierDeepONet", "BranchTrunkFlower", "NIO"}:
        return len(X_test[0])
    return len(X_test)


def plot_velocity_comparison(
    y_true,
    y_pred,
    sample_idx=0,
    save_path=None,
    sosmap_size=(80, 80),
    mm_per_pixel=1.0,
    has_ground_truth=True,
):
    """绘制真实声速图与预测声速图的对比"""
    if not has_ground_truth or y_true is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        label_fontsize = 13
        tick_fontsize = 12
        mm_per_pixel = float(mm_per_pixel)
        extent = [0.0, sosmap_size[1] * mm_per_pixel, 0.0, sosmap_size[0] * mm_per_pixel]
        pred_2d = y_pred[sample_idx].reshape(sosmap_size)
        im = ax.imshow(pred_2d, cmap='jet', vmin=VMIN, vmax=VMAX, origin='lower', extent=extent, aspect='auto')
        ax.set_xlabel('Y (mm)', fontsize=label_fontsize)
        ax.set_ylabel('X (mm)', fontsize=label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Velocity (m/s)', fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        #ax.set_title('Predicted SoS')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    label_fontsize = 13
    tick_fontsize = 12

    mm_per_pixel = float(mm_per_pixel)
    extent = [0.0, sosmap_size[1] * mm_per_pixel, 0.0, sosmap_size[0] * mm_per_pixel]

    true_2d = y_true[sample_idx].reshape(sosmap_size)
    pred_2d = y_pred[sample_idx].reshape(sosmap_size)
    error_2d = np.abs(true_2d - pred_2d)

    # 1. 预测值
    im1 = axes[0].imshow(pred_2d, cmap='jet', vmin=VMIN, vmax=VMAX, origin='lower', extent=extent, aspect='auto')
    #axes[0].set_title('Predicted SoS')
    axes[0].set_xlabel('Y (mm)', fontsize=label_fontsize)
    axes[0].set_ylabel('X (mm)', fontsize=label_fontsize)
    axes[0].tick_params(axis='both', labelsize=tick_fontsize)
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Velocity (m/s)', fontsize=label_fontsize)
    cbar1.ax.tick_params(labelsize=tick_fontsize)

    # 2. 真实值
    im2 = axes[1].imshow(true_2d, cmap='jet', vmin=VMIN, vmax=VMAX, origin='lower', extent=extent, aspect='auto')
    #axes[1].set_title(f'Ground Truth (Sample {sample_idx})')
    axes[1].set_xlabel('Y (mm)', fontsize=label_fontsize)
    axes[1].set_ylabel('X (mm)', fontsize=label_fontsize)
    axes[1].tick_params(axis='both', labelsize=tick_fontsize)
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Velocity (m/s)', fontsize=label_fontsize)
    cbar2.ax.tick_params(labelsize=tick_fontsize)

    # 3. 绝对误差
    #mae = np.mean(error_2d)
    #rmse = np.sqrt(np.mean((true_2d - pred_2d) ** 2))
    im3 = axes[2].imshow(error_2d, cmap='inferno', origin='lower', extent=extent, aspect='auto')
    #axes[2].set_title(f'Absolute Error (MAE: {mae:.2f}, RMSE: {rmse:.2f})')
    axes[2].set_xlabel('Y (mm)', fontsize=label_fontsize)
    axes[2].set_ylabel('X (mm)', fontsize=label_fontsize)
    axes[2].tick_params(axis='both', labelsize=tick_fontsize)
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar3.set_label('Absolute Error (m/s)', fontsize=label_fontsize)
    cbar3.ax.tick_params(labelsize=tick_fontsize)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)

def plot_error_distribution(y_true, y_pred, save_path=None, sosmap_size=(80, 80)):
    """绘制误差分布统计"""
    if y_true is None:
        return
    errors = y_true - y_pred
    errors_flat = errors.flatten()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # A. 误差直方图
    axes[0, 0].hist(errors_flat, bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Error (m/s)')
    axes[0, 0].set_ylabel('Frequency')
    #axes[0, 0].set_title('Error Distribution Histogram')
    axes[0, 0].grid(True, alpha=0.3)

    # B. 误差箱线图
    sample_indices = np.random.choice(len(errors_flat), size=min(10000, len(errors_flat)), replace=False)
    axes[0, 1].boxplot(errors_flat[sample_indices])
    axes[0, 1].set_ylabel('Error (m/s)')
    #axes[0, 1].set_title('Error Boxplot (Sampled)')
    axes[0, 1].grid(True, alpha=0.3)

    # C. 真实值 vs 预测值散点图
    flat_true = y_true.flatten()[sample_indices]
    flat_pred = y_pred.flatten()[sample_indices]

    axes[1, 0].scatter(flat_true, flat_pred, alpha=0.2, s=2)
    # 绘制 y=x 参考线
    axes[1, 0].plot([VMIN, VMAX], [VMIN, VMAX], 'r--', alpha=0.8)
    axes[1, 0].set_xlim([VMIN, VMAX])
    axes[1, 0].set_ylim([VMIN, VMAX])
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Predicted Values')
    #axes[1, 0].set_title('True vs Predicted Scatter')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].grid(True, alpha=0.3)

    # D. 平均绝对误差空间分布图
    mean_abs_error = np.mean(np.abs(errors.reshape(-1, sosmap_size[0], sosmap_size[1])), axis=0)
    im = axes[1, 1].imshow(mean_abs_error, cmap='hot', origin='lower')
    #axes[1, 1].set_title('Mean Spatial Error Map')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _load_h5_meta(meta_path):
    if not meta_path or not os.path.exists(meta_path):
        return None
    return _load_json_file(meta_path)


def _load_full_h5_test_set(cfg, is_deeponet, batch_size, seed=114514, total_data_num = 0):
    """Load the full H5 test split in batches to avoid OOM."""
    data = H5DeepONetDataset(
        H5DatasetConfig(h5_path=cfg["cache_h5_path"], split_ratio=cfg["split_ratio"], test_batch_size=batch_size, total_data_num=total_data_num),
        is_deeponet=is_deeponet,
        seed=seed,
    )
    try:
        total_test = len(data.test_indices)
        if total_test <= batch_size:
            return data.test()

        num_batches = (total_test + batch_size - 1) // batch_size
        if is_deeponet:
            xb_list, xt_list, y_list = [], [], []
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, total_test)
                global_idx = data.test_indices[start:end]
                (xb, xt), y = data._get_batch_by_global_indices(global_idx)
                xb_list.append(xb)
                xt_list.append(xt)
                y_list.append(y)
            X_test = (np.concatenate(xb_list, axis=0), np.concatenate(xt_list, axis=0))
        else:
            xb_list, y_list = [], []
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, total_test)
                global_idx = data.test_indices[start:end]
                xb, y = data._get_batch_by_global_indices(global_idx)
                xb_list.append(xb)
                y_list.append(y)
            X_test = np.concatenate(xb_list, axis=0)

        y_true = np.concatenate(y_list, axis=0)
        return X_test, y_true
    finally:
        data.close()


def _load_full_h5_test_set_nio(cfg, batch_size, grid_npy_path, seed=114514, total_data_num=0):
    """Load the full H5 test split for NIO in batches.

    Returns:
        X_test: tuple (X_branch_all, grid_static)
        y_true: ndarray [N, H, W]
    """
    data = H5NIODataset(
        H5NIOConfig(
            h5_path=cfg["cache_h5_path"],
            grid_npy_path=grid_npy_path,
            split_ratio=cfg["split_ratio"],
            test_batch_size=batch_size,
            total_data_num=total_data_num,
            squeeze_y_channel=True,
        ),
        seed=seed,
    )
    try:
        total_test = len(data.test_indices)
        if total_test <= batch_size:
            return data.test()

        num_batches = (total_test + batch_size - 1) // batch_size
        xb_list, y_list = [], []
        static_grid = None
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_test)
            global_idx = data.test_indices[start:end]
            (xb, grid), y = data._get_batch_by_global_indices(global_idx)
            xb_list.append(xb)
            y_list.append(y)
            if static_grid is None:
                static_grid = grid

        X_branch = np.concatenate(xb_list, axis=0)[:total_test]
        y_true = np.concatenate(y_list, axis=0)[:total_test]
        return (X_branch, static_grid), y_true
    finally:
        data.close()


def main(model_path, result_dir, model_type="FourierDeepONet", visualize=True,
         batch_size=4, split_ratio=0.0, total_data_num=100, is_deeponet=True,
         sosmap_size=(80, 80), samples_plot=50, save_npy=False,
         mm_per_pixel=1.0, cache_h5_path=None, cache_meta_path=None,
         has_ground_truth=None):
    """
    主测试函数
    model_path: .pt 文件的完整路径
    model_type: "FourierDeepONet" or "InversionNet"
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"--- 1. Loading Config ---")
    model_init_kwargs = None
    model_config_path = os.path.join(os.path.dirname(model_path), "model_config.json")
    data_cfg = None
    model_type_raw = model_type
    if os.path.exists(model_config_path):
        try:
            with open(model_config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)
            model_type_raw = model_config.get("model_type", model_type)
            model_init_kwargs = model_config.get("model_init_kwargs")
            data_cfg = model_config.get("data", None)
            print(f"Loaded model config from: {model_config_path}")
            print(f"Auto model_type from config: {model_type_raw}")
        except Exception as e:
            print(f"Warning: failed to read model_config.json: {e}")

    h5_meta = _load_h5_meta(cache_meta_path)
    if isinstance(h5_meta, dict):
        if has_ground_truth is None and "has_ground_truth" in h5_meta:
            has_ground_truth = bool(h5_meta.get("has_ground_truth"))
        if cache_h5_path is None and h5_meta.get("cache_h5_path"):
            cache_h5_path = h5_meta.get("cache_h5_path")
        if h5_meta.get("y_shape") and (has_ground_truth is False):
            shape_tail = h5_meta.get("y_shape")[1:]
            if len(shape_tail) == 2:
                sosmap_size = (int(shape_tail[0]), int(shape_tail[1]))

    if cache_h5_path is not None:
        if data_cfg is None:
            data_cfg = {}
        data_cfg["cache_h5_path"] = cache_h5_path
    if has_ground_truth is None and isinstance(data_cfg, dict):
        has_ground_truth = bool(data_cfg.get("has_ground_truth", True))
    if has_ground_truth is None:
        has_ground_truth = True

    model_type_alias = {
        "FourierDeepONet": "FourierDeepONet",
        "BranchTrunkFlower": "BranchTrunkFlower",
        "InversionNet": "InversionNet",
        "NIO": "NIO",
        "NIOUltrasoundCTAbl": "NIO",
    }
    model_type = model_type_alias.get(model_type_raw, model_type_raw)

    print(f"--- 2. Loading Data ---")
    if model_type in {"FourierDeepONet", "BranchTrunkFlower"} and data_cfg is not None:
        if total_data_num <= 0 and isinstance(h5_meta, dict):
            total_data_num = int(h5_meta.get("num_samples", total_data_num))
        if total_data_num <= 0 and data_cfg.get("cache_h5_path"):
            with h5py.File(data_cfg["cache_h5_path"], "r") as f:
                total_data_num = int(f["X_branch"].shape[0])
        if has_ground_truth is False and isinstance(data_cfg, dict):
            data_cfg["split_ratio"] = 0.0
        X_test, y_true_orig = _load_full_h5_test_set(data_cfg, is_deeponet=True, batch_size=batch_size, total_data_num=total_data_num)
        split_ratio = data_cfg.get("split_ratio", split_ratio)
    elif model_type == "NIO" and data_cfg is not None:
        grid_npy_path = None
        if isinstance(model_init_kwargs, dict):
            grid_npy_path = model_init_kwargs.get("grid_npy_path")
        if grid_npy_path is None:
            grid_npy_path = data_cfg.get("grid_npy_path")
        if not grid_npy_path:
            raise ValueError("NIO test requires 'grid_npy_path' in model_config.json.")

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
        if total_data_num <= 0 and data_cfg and data_cfg.get("cache_h5_path"):
            with h5py.File(data_cfg["cache_h5_path"], "r") as f:
                total_data_num = int(f["X_branch"].shape[0])
        if has_ground_truth is False:
            data_cfg["split_ratio"] = 0.0

        X_test, y_true_orig = _load_full_h5_test_set(data_cfg, is_deeponet=False, batch_size=batch_size, total_data_num=total_data_num)

    else:
        _, X_test, _, y_true_orig = get_legacy_dataset(split_ratio=split_ratio, total_data_num=total_data_num, is_deeponet=is_deeponet)

    if not has_ground_truth and y_true_orig is not None and y_true_orig.ndim >= 3:
        sosmap_size = (int(y_true_orig.shape[1]), int(y_true_orig.shape[2]))

    print(f"Test Data Shape: {y_true_orig.shape}")

    sample_count = len(y_true_orig)
    if sample_count <= 0:
        sample_count = _infer_sample_count(X_test, model_type)

    length = sample_count // batch_size
    if sample_count % batch_size != 0:
        length += 1

    print(f"--- 3. Building Model ---")

    if model_type in {"FourierDeepONet", "BranchTrunkFlower"}:
        if isinstance(model_init_kwargs, dict):
            net = BranchTrunkFlower(**model_init_kwargs) if model_type == "BranchTrunkFlower" else FourierDeepONet(**model_init_kwargs)
        else:
            trunk_dim = X_test[1].shape[1]
            if model_type == "BranchTrunkFlower":
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
                    boundary_condition_types=["ZEROS"],
                    dropout_rate=0.0,
                    regularization=["l2", 3e-6],
                    channel_lift_first=True,
                )
            else:
                net = FourierDeepONet(
                    num_parameter=trunk_dim,
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
        nio_kwargs = _extract_nio_build_kwargs(model_init_kwargs)
        branch_encoder_cls = _resolve_nio_branch_encoder_cls(model_init_kwargs)
        branch_encoder_kwargs = _resolve_nio_branch_encoder_kwargs(model_init_kwargs)
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
        raise ValueError(f"Unknown model_type={model_type!r}. Expected 'FourierDeepONet', 'BranchTrunkFlower', 'InversionNet', or 'NIO'.")

    net.to(device)

    print(f"--- 4. Loading Weights from {model_path} ---")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            net.load_state_dict(checkpoint["model_state_dict"])
        else:
            net.load_state_dict(checkpoint)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    net.eval()
    y_pred_list = []
    nio_grid_tensor = None
    if model_type == "NIO":
        nio_grid_tensor = torch.as_tensor(X_test[1], dtype=torch.float32, device=device)

    print(f"--- 5. Inference Loop ---")
    with torch.no_grad():
        for i in range(length):
            start = batch_size * i
            end = min(batch_size * (i + 1), sample_count)

            if model_type in {"FourierDeepONet", "BranchTrunkFlower"}:
                branch_batch = torch.as_tensor(X_test[0][start:end]).to(device)
                trunk_batch = torch.as_tensor(X_test[1][start:end]).to(device)
                inputs = (branch_batch, trunk_batch)
                outputs = net(inputs)
                del branch_batch, trunk_batch
            elif model_type == "NIO":
                branch_batch = torch.as_tensor(X_test[0][start:end], dtype=torch.float32, device=device)
                outputs = net(branch_batch, nio_grid_tensor)
                del branch_batch
            elif model_type == "InversionNet":
                # InversionNet: X_test 是单输入数组/tensor
                x_batch = torch.as_tensor(X_test[start:end]).to(device)
                outputs = net(x_batch)
                del x_batch

            outputs_np = outputs.cpu().numpy()

            if has_ground_truth:
                batch_loss = np.mean(np.abs(outputs_np.squeeze() - y_true_orig[start:end]))
                print(f"Batch {i + 1}/{length} - L1 Loss (Normalized): {batch_loss:.6f}")
            else:
                print(f"Batch {i + 1}/{length} - inference only")

            y_pred_list.append(outputs_np)

            del outputs
            torch.cuda.empty_cache()

    y_pred_norm = np.concatenate(y_pred_list, axis=0).squeeze()
    if y_pred_norm.ndim == 2:
        y_pred_norm = np.expand_dims(y_pred_norm, 0)
    y_pred_norm = y_pred_norm.reshape(-1, sosmap_size[0], sosmap_size[1])

    print(f"--- 6. Denormalizing & Metrics ---")
    y_pred_real = minmax_denormalize(y_pred_norm, VMIN, VMAX, scale=2)
    y_true_real = minmax_denormalize(y_true_orig, VMIN, VMAX, scale=2) if has_ground_truth else None

    mae_mean = mae_std = rmse_mean = rmse_std = ssim_mean = ssim_std = pcc_mean = pcc_std = l2_mean = l2_std = 0.0
    if has_ground_truth:
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

        print(f"\n===== Evaluation Results =====")
        print(f"MAE:  {mae_mean:.4f}±{mae_std:.4f} m/s")
        print(f"RMSE: {rmse_mean:.4f}±{rmse_std:.4f} m/s")
        print(f"SSIM: {ssim_mean:.4f}±{ssim_std:.4f}")
        print(f"PCC:  {pcc_mean:.4f}±{pcc_std:.4f}")
        print(f"L2 Rel: {l2_mean:.4%}±{l2_std:.4%}")
    else:
        print("\n===== Inference Mode =====")
        print("No ground truth available. Metrics are skipped.")

    os.makedirs(result_dir, exist_ok=True)
    if save_npy:
        np.save(os.path.join(result_dir, "y_pred.npy"), y_pred_real)
        if has_ground_truth and y_true_real is not None:
            np.save(os.path.join(result_dir, "y_true.npy"), y_true_real)

    if visualize:
        print(f"--- 7. Plotting ---")
        vis_dir = os.path.join(result_dir, 'plots')
        os.makedirs(vis_dir, exist_ok=True)

        for i in range(min(samples_plot, len(y_pred_real))):
            plot_velocity_comparison(
                y_true_real, y_pred_real, sample_idx=i,
                save_path=os.path.join(vis_dir, f'sample_{i}_comparison.png'),
                sosmap_size=sosmap_size,
                mm_per_pixel=mm_per_pixel,
                has_ground_truth=has_ground_truth,
            )

        if has_ground_truth and y_true_real is not None:
            plot_error_distribution(
                y_true_real, y_pred_real,
                save_path=os.path.join(vis_dir, 'error_analysis.png'),
                sosmap_size=sosmap_size
            )

        with open(os.path.join(vis_dir, 'report.txt'), 'w') as f:
            f.write(f"Evaluation Report\n")
            f.write("=" * 30 + "\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Has Ground Truth: {has_ground_truth}\n")
            f.write(f"Samples: {sample_count}\n")
            if has_ground_truth:
                f.write(f"MAE: {mae_mean:.6f}±{mae_std:.6f} m/s\n")
                f.write(f"RMSE: {rmse_mean:.6f}±{rmse_std:.6f} m/s\n")
                f.write(f"SSIM: {ssim_mean:.6f}±{ssim_std:.6f}\n")
                f.write(f"PCC: {pcc_mean:.6f}±{pcc_std:.6f}\n")
                f.write(f"L2 Relative Error: {l2_mean:.6f}±{l2_std:.6f}\n")
            else:
                f.write("Metrics: skipped because ground truth is unavailable\n")

        print(f"All results saved to: {vis_dir}")


if __name__ == "__main__":
    MODEL_PATH = "/home/wkf/wkf_kwave/src/model_50K_5x2_configs_test1_DFlower_CLF_128width_0.140625-0.453125/model-294000.pt"
    result_dir = "/home/wkf/wkf_kwave/src/model_50K_5x2_configs_test1_DFlower_CLF_128width_0.140625-0.453125/test_result_294000"
    main(model_path=MODEL_PATH, result_dir = result_dir,
     model_type="BranchTrunkFlower", visualize=True, batch_size=32,
         split_ratio=0.9, total_data_num = 50000, is_deeponet=True
         ,sosmap_size=(80, 80), samples_plot=100, mm_per_pixel=0.1,
         cache_h5_path="/home/wkf/kwave-python/dataset/dataset_shuffle_0.140625-0.453125.h5",
         cache_meta_path="/home/wkf/kwave-python/dataset/dataset_shuffle_0.140625-0.453125_meta.json",
         has_ground_truth=True)
