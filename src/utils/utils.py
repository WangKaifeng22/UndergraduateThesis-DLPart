import math
import os
import random

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

try:
    from skimage.metrics import structural_similarity as ssim_skimage
except ImportError:
    print("Warning: scikit-image not found. SSIM calculation will be skipped.")
    print("Please install it via: pip install scikit-image")
    ssim_skimage = None


# 定义物理参数用于(反)归一化
VMIN, VMAX = 1430, 1650
PHYSICAL_LIMIT = 0.04 # 传感器坐标的物理范围 (米)


def _gray(m: int) -> np.ndarray:
    g = np.arange(m, dtype=np.float64) / max(m - 1, 1)
    g = g[:, None]
    return np.hstack([g, g, g])


def _hot(m: int) -> np.ndarray:
    n = int(np.fix(3 / 8 * m))
    r = np.concatenate([np.arange(1, n + 1) / max(n, 1), np.ones(max(m - n, 0))])
    g = np.concatenate([
        np.zeros(n),
        np.arange(1, n + 1) / max(n, 1),
        np.ones(max(m - 2 * n, 0)),
    ])
    denom_b = max(m - 2 * n, 1)
    b = np.concatenate([
        np.zeros(2 * n),
        np.arange(1, m - 2 * n + 1) / denom_b,
    ])
    return np.hstack([r[:, None], g[:, None], b[:, None]])


def _bone(m: int) -> np.ndarray:
    return (7 * _gray(m) + np.fliplr(_hot(m))) / 8


def get_kwave_style_colormap(num_colors: int = 256) -> ListedColormap:
    neg_pad = int(round(48 * num_colors / 256))
    neg = _bone(num_colors // 2 + neg_pad)
    neg = neg[neg_pad:, :]
    pos = np.flipud(_hot(num_colors // 2))
    colors = np.vstack([neg, pos])
    return ListedColormap(colors)


KWAVE_CMAP = get_kwave_style_colormap()
# Backward-compatible alias for existing plotting code.
BRANCH_CMAP = KWAVE_CMAP

# --- 1. 预处理工具函数 ---
class MaxAbsScaler:
    """
    最大绝对值缩放器
    将数据缩放到 [-1, 1] 之间，同时保持 0 仍然是 0。
    """
    def __init__(self):
        self.max_abs = 1.0

    def fit(self, x):
        self.max_abs = np.max(np.abs(x))
        if self.max_abs == 0:
            self.max_abs = 1.0
        return self

    def transform(self, x):
        return x / self.max_abs

    def inverse_transform(self, x):
        return x * self.max_abs


def complex_global_scale_fit(z: np.ndarray, method: str = "max", q: float = 0.999) -> float:
    """为复数数据 z 计算全局缩放因子 s（基于 |z|）。

    目标：让 real/imag 两路共享同一个缩放因子，保持相位关系。

    参数
    - z: 复数数组（任意形状）
    - method: 'max' 或 'quantile'
    - q: method='quantile' 时使用的分位数，比如 0.999

    返回
    - s: >0 的浮点数；若全 0 或异常则返回 1.0
    """
    mag = np.abs(z)
    if method == "quantile":
        s = float(np.quantile(mag, q))
    elif method == "max":
        s = float(np.max(mag))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'max' or 'quantile'.")

    if (not np.isfinite(s)) or s <= 0:
        s = 1.0
    return s


def complex_global_scale_transform(z: np.ndarray, scale: float, clip: float | None = None) -> np.ndarray:
    """对复数数据做全局缩放：z / scale，并可选按幅值 clip。"""
    scale = float(scale)
    if (not np.isfinite(scale)) or scale <= 0:
        scale = 1.0

    z_scaled = z / scale
    if clip is not None:
        clip = float(clip)
        if clip > 0 and np.isfinite(clip):
            # clip 作用在实部/虚部上（等价于对每个分量截断）
            z_scaled = np.clip(z_scaled, -clip, clip)
    return z_scaled


def normalize_branch_from_complex(z: np.ndarray, scale: float, axis_concat: int = 0, clip: float | None = None) -> np.ndarray:
    """把复数频域数据 z 归一化（基于 |z| 的全局缩放），并输出 real/imag 拼接后的实数张量。

    返回形状：concat([real(z/scale), imag(z/scale)], axis=axis_concat)
    """
    z_scaled = complex_global_scale_transform(z, scale=scale, clip=clip)
    return np.concatenate([np.real(z_scaled), np.imag(z_scaled)], axis=axis_concat)


def log_transform(data, k=1, c=0):
    """ 对数变换，用于处理跨度大的频域数据 """
    return (np.log1p(np.abs(k * data) + c)) * np.sign(data)


def minmax_normalize(vid, vmin, vmax, scale=2):
    """ 归一化到 [-1, 1] (scale=2) 或 [0, 1] (scale=1) """
    vid = vid - vmin
    vid = vid / (vmax - vmin)
    if scale == 2:
        return (vid - 0.5) * 2
    return vid


def minmax_denormalize(vid, vmin, vmax, scale=2):
    """ 反归一化 """
    if scale == 2:
        vid = vid / 2 + 0.5
    return vid * (vmax - vmin) + vmin


def prepare_visualization_data(
    data,
    *,
    enabled: bool = True,
    normalize_range: str = "dynamic",
    vmin: float | None = None,
    vmax: float | None = None,
    scale: int = 2,
    log_k: float = 1,
    log_c: float = 0,
):
    """Apply signed log1p compression followed by minmax normalization for plotting."""
    arr = np.asarray(data)
    if not enabled:
        return arr

    arr = log_transform(arr, k=log_k, c=log_c)

    if normalize_range == "none":
        return arr

    if normalize_range == "dynamic":
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return arr
        vmin = float(finite.min())
        vmax = float(finite.max())
    elif normalize_range == "fixed":
        if vmin is None or vmax is None:
            raise ValueError("fixed normalize_range requires vmin and vmax.")
        vmin = float(vmin)
        vmax = float(vmax)
    else:
        raise ValueError("normalize_range must be 'dynamic', 'fixed', or 'none'.")

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(f"Invalid minmax range: vmin={vmin}, vmax={vmax}.")

    return minmax_normalize(arr, vmin, vmax, scale=scale)

def visualize_samples(X_branch, X_trunk, y_data, num_samples=3):
    """
    随机抽样并可视化 Branch, Trunk 和 Output 数据以进行验证。
    """
    batch_size = X_branch.shape[0]
    # 随机选择索引
    indices = np.random.choice(batch_size, min(num_samples, batch_size), replace=False)

    for idx in indices:
        print(f"--- Visualizing Sample Index: {idx} ---")

        # 1. 获取单样本数据
        branch_sample = X_branch[idx]  # Shape: (32, 64, 1024)
        trunk_sample = X_trunk[idx]  # Shape: (64,)
        y_sample = y_data[idx]  # Shape: (384, 384)

        # 2. 反归一化处理
        y_real = minmax_denormalize(y_sample, VMIN, VMAX, scale=2)
        trunk_real = minmax_denormalize(trunk_sample, -PHYSICAL_LIMIT, PHYSICAL_LIMIT, scale=2)

        # 解析 Trunk 坐标
        sensor_x = trunk_real[0::2]
        sensor_y = trunk_real[1::2]

        # 3. 绘图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot A: Ground Truth (Velocity Map)
        im0 = axes[0].imshow(y_real, cmap='jet', origin='lower', extent=[-0.04, 0.04, -0.04, 0.04])
        axes[0].set_title(f"Ground Truth SoS Map\n(Sample {idx})")
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        fig.colorbar(im0, ax=axes[0], label='Speed of Sound (m/s)')

        # Plot B: Trunk (Sensor Geometry)
        axes[1].scatter(sensor_x, sensor_y, c='red', marker='x', label='Receivers')
        axes[1].set_xlim([-PHYSICAL_LIMIT * 1.1, PHYSICAL_LIMIT * 1.1])
        axes[1].set_ylim([-PHYSICAL_LIMIT * 1.1, PHYSICAL_LIMIT * 1.1])
        axes[1].set_title("Trunk Input: Sensor Geometry")
        axes[1].set_xlabel("x (m)")
        axes[1].set_ylabel("y (m)")
        axes[1].grid(True, linestyle='--')
        axes[1].legend()
        axes[1].set_aspect('equal')

        # Plot C: Branch Input (Sensor Data)
        # branch_sample 是(Sources * (real + img), Receivers, Freq)
        # 我们只画第 0 个发射源的数据，形状变为 (Receivers, Freq)
        source_idx_to_plot = 0
        branch_slice = branch_sample[source_idx_to_plot, :, :]  # Shape: (Receivers, Freq)

        im2 = axes[2].imshow(branch_slice, aspect='auto', cmap='viridis')
        axes[2].set_title(f"Branch Input (Source #{source_idx_to_plot})\nShape: {branch_slice.shape}")
        axes[2].set_xlabel("Frequency Points (FFT)")
        axes[2].set_ylabel("Receiver Channels (Real + Imag)")
        fig.colorbar(im2, ax=axes[2], label='Normalized Amplitude')

        plt.tight_layout()
        plt.show()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

def loss_func_L1(y_true, y_pred):
    return torch.nn.L1Loss(reduction="mean")(y_pred, y_true)

def loss_func_L2(y_true, y_pred):
    return torch.nn.MSELoss()(y_pred, y_true)


def preflight_check_xy(
    X_test, y_test, name="test", max_print=12, sample_k=8,
    *,
    visualize: bool = True,
    vis_index: int | None = 0,
    xb_dim0_index: int = 1,
    save_dir: str | None = "test",
    save_path: str | None = "./test.png",
    dpi: int = 500,
    show: bool = False,
    meta_path: str | None = "/home/wkf/wkf_kwave/dataset_meta_timedomain.json",
    denorm_xb_from_meta: bool = False,
):
    """
    训练前数据合理性检查（只用 X_test, y_test）

    新增: 可选从 HDF5 元数据(meta json)里读取 branch_min/branch_max，
    在可视化前对 X_branch 进行反归一化（还原到物理幅值/原始尺度），避免归一化后图像“发绿一片”。

    参数
    - meta_path: str | None
        指向 h5_preprocess 写出的 meta json（包含 branch_min/branch_max）。
        例如: dataset_meta.json
    - denorm_xb_from_meta: bool
        是否在可视化时对 X_branch 做反归一化；默认 True。

    其它参数见函数签名。
    """

    def _as_np(a):
        # deepxde 里通常是 numpy; 这里做一层兜底
        if hasattr(a, "detach"):
            a = a.detach().cpu().numpy()
        return np.asarray(a)

    def _stat(x, tag):
        x = _as_np(x)
        fin = np.isfinite(x)
        nan_cnt = int(np.isnan(x).sum())
        inf_cnt = int(np.isinf(x).sum())
        finite_ratio = float(fin.mean()) if x.size > 0 else 0.0

        # 对全 0/空数组兜底
        if x.size == 0:
            print(f"[{name}] {tag}: EMPTY")
            return

        x_f = x[fin]
        if x_f.size == 0:
            print(f"[{name}] {tag}: NO_FINITE (nan={nan_cnt}, inf={inf_cnt})")
            return

        xmin = float(x_f.min())
        xmax = float(x_f.max())
        mean = float(x_f.mean())
        std = float(x_f.std())
        # 近常数检测：std 很小 或者 max-min 很小
        near_const = (std < 1e-8) or ((xmax - xmin) < 1e-6)

        print(
            f"[{name}] {tag}: shape={x.shape}, dtype={x.dtype}, "
            f"finite={finite_ratio:.4f}, nan={nan_cnt}, inf={inf_cnt}, "
            f"min={xmin:.6g}, max={xmax:.6g}, mean={mean:.6g}, std={std:.6g}, "
            f"near_const={near_const}"
        )

        # 强约束: 不允许 NaN/Inf
        assert nan_cnt == 0 and inf_cnt == 0, f"{tag} contains NaN/Inf"
        # 强约束: 不能是近常数（看任务可放宽）
        assert not near_const, f"{tag} is nearly constant (std/min-max too small)"

    # --- 结构检查 ---
    assert isinstance(X_test, (tuple, list)) and len(X_test) == 2, \
        "X_test must be (X_branch, X_trunk)"
    Xb, Xt = X_test
    Xb = _as_np(Xb)
    Xt = _as_np(Xt)
    y = _as_np(y_test)

    # --- 维度/样本数一致 ---
    assert Xb.shape[0] == Xt.shape[0] == y.shape[0], \
        f"Batch size mismatch: Xb={Xb.shape[0]}, Xt={Xt.shape[0]}, y={y.shape[0]}"
    assert y.ndim >= 2, f"y_test should be at least 2D, got {y.ndim}D"
    assert Xb.ndim >= 2 and Xt.ndim >= 2, \
        f"Unexpected ndims: X_branch={Xb.ndim}, X_trunk={Xt.ndim}"

    # --- dtype 检查（浮点）---
    assert np.issubdtype(Xb.dtype, np.floating), f"X_branch dtype must be float, got {Xb.dtype}"
    assert np.issubdtype(Xt.dtype, np.floating), f"X_trunk dtype must be float, got {Xt.dtype}"
    assert np.issubdtype(y.dtype, np.floating), f"y_test dtype must be float, got {y.dtype}"

    # --- 数值统计 ---
    _stat(Xb, "X_branch")
    _stat(Xt, "X_trunk")
    _stat(y, "y_test")

    # --- 取样本检查 y 的尺度是否离谱/全 0 ---
    n = y.shape[0]
    rng = np.random.default_rng(123)
    k = min(sample_k, n)
    idx = rng.choice(n, size=k, replace=False)

    # 对每个样本计算 L2 范数、均值、std
    y_flat = y.reshape(n, -1)
    norms = np.linalg.norm(y_flat[idx], axis=1)
    means = y_flat[idx].mean(axis=1)
    stds = y_flat[idx].std(axis=1)

    print(f"[{name}] sampled y norms (first {max_print}): {norms[:max_print]}")
    print(f"[{name}] sampled y mean/std (first {max_print}): {list(zip(means[:max_print], stds[:max_print]))}")

    assert np.all(np.isfinite(norms)), "Sampled y norms contain NaN/Inf"
    assert float(np.median(stds)) > 1e-8, "Median sampled y std too small; labels may be (near) constant"

    # --- 可视化（可选） ---
    if visualize:
        # lazy import to avoid overhead if not used
        branch_min = None
        branch_max = None
        if denorm_xb_from_meta and meta_path is not None:
            try:
                import json
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                branch_min = meta.get("branch_min", None)
                branch_max = meta.get("branch_max", None)
                if branch_min is not None:
                    branch_min = float(branch_min)
                if branch_max is not None:
                    branch_max = float(branch_max)
            except Exception as e:
                print(f"[{name}] WARNING: failed to read meta_path={meta_path} for denorm: {e}")
                branch_min = None
                branch_max = None

        if vis_index is None:
            vis_index = int(rng.integers(0, n))
        else:
            vis_index = int(vis_index)
        if not (0 <= vis_index < n):
            raise ValueError(f"vis_index out of range: {vis_index} (n={n})")

        xb0 = int(xb_dim0_index)
        if Xb.ndim < 3:
            raise ValueError(f"X_branch must be at least 3D to slice dim-1, got shape {Xb.shape}")
        if not (0 <= xb0 < Xb.shape[1]):
            raise ValueError(f"xb_dim0_index out of range: {xb0} (Xb.shape[1]={Xb.shape[1]})")

        xb_vis = Xb[vis_index, xb0]
        if xb_vis.ndim != 2:
            xb_vis = np.reshape(xb_vis, (xb_vis.shape[0], -1))

        # Denormalize X_branch for visualization only (keep training arrays untouched)
        if denorm_xb_from_meta and (branch_min is not None) and (branch_max is not None):
            try:
                xb_vis = minmax_denormalize(xb_vis, branch_min, branch_max, scale=2)
                xb_vis = log_transform(xb_vis)
            except Exception as e:
                print(f"[{name}] WARNING: failed to denormalize X_branch for visualization: {e}")

        xt_vis = np.ravel(Xt[vis_index])

        y_vis = y[vis_index]
        if y_vis.ndim > 2:
            y_vis = np.reshape(y_vis, y_vis.shape[-2:])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].imshow(xb_vis, aspect="auto", cmap="viridis")
        title0 = f"X_branch sample={vis_index}, dim1={xb0}\nshape={xb_vis.shape}"
        if denorm_xb_from_meta and (branch_min is not None) and (branch_max is not None):
            title0 += f"\ndenorm=[{branch_min:.4g},{branch_max:.4g}]"
        axes[0].set_title(title0)
        axes[0].set_xlabel("time/freq axis")
        axes[0].set_ylabel("receiver/channel")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        axes[1].plot(xt_vis)
        axes[1].set_title(f"X_trunk sample={vis_index}\nshape={xt_vis.shape}")
        axes[1].set_xlabel("index")
        axes[1].set_ylabel("value")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        im2 = axes[2].imshow(y_vis, cmap="jet", origin="lower")
        axes[2].set_title(f"y sample={vis_index}\nshape={y_vis.shape}")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.suptitle(f"Preflight visualization ({name})", y=1.02)
        plt.tight_layout()

        # ---- save to disk (headless-friendly) ----
        if save_path is None:
            # Default: save into save_dir or current working directory
            out_dir = save_dir if save_dir is not None else os.getcwd()
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"preflight_{name}_idx{vis_index}_xb{xb0}.png")
        else:
            out_dir = os.path.dirname(save_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

        fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight")
        print(f"[{name}] saved preflight figure to: {save_path}")

        if show:
            # 如果有 GUI 环境，这会弹窗；在 headless 环境下一般不会显示
            plt.show()

        plt.close(fig)

    print(f"[{name}] preflight check passed: n={n}")


def large_dataset_schedule(step, total_steps=None, total_epochs=None, start_it = 0):
    step += start_it
    # 阶段1: 预热
    warmup_steps = int(total_steps * 0.05)
    if step < warmup_steps:
        # 线性预热
        return step / warmup_steps
    
    # 阶段2: 高学习率探索
    explore_steps = int(total_steps * 0.45)
    if step < warmup_steps + explore_steps:
        return 1.0  # 保持高学习率
    
    # 阶段3: 余弦退火
    else:
        T_cur = step - (warmup_steps + explore_steps)
        T_max = total_steps - (warmup_steps + explore_steps)
        
        # 余弦退火
        return 0.5 * (1 + math.cos(math.pi * T_cur / T_max))


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