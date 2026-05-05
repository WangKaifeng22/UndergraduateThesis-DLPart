import gc
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

# 导入必要的模块，与 test.py 保持一致
from utils.utils import minmax_denormalize, VMIN, VMAX
from models.InversionNet import InversionNet
from models.model_BranchTrunkFlower import BranchTrunkFlower
from utils.fourier_model_utils import build_fourier_deeponet_variant, is_original_fourier_deeponet_config
from training.train_NIO import build_nio
from utils.nio_build_utils import (
    extract_nio_build_kwargs as _extract_nio_build_kwargs,
    resolve_nio_branch_encoder_cls as _resolve_nio_branch_encoder_cls,
    resolve_nio_branch_encoder_kwargs as _resolve_nio_branch_encoder_kwargs,
)

# 导入 test.py 中的数据加载辅助函数
from evaluation.test import (
    _load_h5_meta,
    _load_full_h5_test_set,
    _load_full_h5_test_set_nio,
    _infer_sample_count
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_predictions(model_info, data_kwargs, max_samples):
    """
    加载单个模型并获取其在测试集上的预测结果。
    通过 max_samples 限制最大预测数量，节省时间和显存。
    """
    model_path = model_info['path']
    model_type = model_info['type']
    batch_size = data_kwargs['batch_size']
    total_data_num = data_kwargs['total_data_num']
    cache_h5_path = data_kwargs['cache_h5_path']
    cache_meta_path = data_kwargs.get('cache_meta_path')
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"\n========== 处理模型: {model_info['name']} ==========")
    model_config_path = os.path.join(os.path.dirname(model_path), "model_config.json")
    model_init_kwargs = None
    is_original = False
    data_cfg = {"cache_h5_path": cache_h5_path, "split_ratio": data_kwargs['split_ratio']}
    
    if os.path.exists(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
        model_init_kwargs = model_config.get("model_init_kwargs")
        is_original = is_original_fourier_deeponet_config(model_config)
        if "data" in model_config:
            data_cfg.update(model_config["data"])

    # --- 1. 数据加载 ---
    if model_type in {"FourierDeepONet", "BranchTrunkFlower"}:
        X_test, y_true_orig = _load_full_h5_test_set(data_cfg, is_deeponet=True, batch_size=batch_size, total_data_num=total_data_num)
    elif model_type == "NIO":
        grid_npy_path = model_init_kwargs.get("grid_npy_path", data_cfg.get("grid_npy_path"))
        X_test, y_true_orig = _load_full_h5_test_set_nio(data_cfg, batch_size=batch_size, grid_npy_path=grid_npy_path, total_data_num=total_data_num)
    elif model_type == "InversionNet":
        X_test, y_true_orig = _load_full_h5_test_set(data_cfg, is_deeponet=False, batch_size=batch_size, total_data_num=total_data_num)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    original_sample_count = len(y_true_orig) if y_true_orig is not None else _infer_sample_count(X_test, model_type)
    # >>> 修改点：只预测前 max_samples 个样本 <<<
    sample_count = min(original_sample_count, max_samples)
    
    sosmap_size = (int(y_true_orig.shape[1]), int(y_true_orig.shape[2]))
    
    # --- 2. 构建模型 ---
    if model_type in {"FourierDeepONet", "BranchTrunkFlower"}:
        if isinstance(model_init_kwargs, dict):
            net = BranchTrunkFlower(**model_init_kwargs) if model_type == "BranchTrunkFlower" else build_fourier_deeponet_variant(model_init_kwargs, original=is_original)
        else:
            trunk_dim = X_test[1].shape[1]
            net = BranchTrunkFlower(num_parameter=trunk_dim, width=96, Tx=32, Rx=32, T_steps=1900, H=sosmap_size[0], W=sosmap_size[1], lifting_dim=96, n_levels=4, num_heads=32, channel_lift_first=True) if model_type == "BranchTrunkFlower" else build_fourier_deeponet_variant(trunk_dim=trunk_dim, original=is_original, width=64, modes1=16, modes2=16)
    elif model_type == "InversionNet":
        net = InversionNet(**model_init_kwargs) if isinstance(model_init_kwargs, dict) else InversionNet(dim0=64, dim1=64, dim2=64, dim3=128, dim4=256, dim5=512)
    elif model_type == "NIO":
        nio_kwargs = _extract_nio_build_kwargs(model_init_kwargs)
        branch_encoder_cls = _resolve_nio_branch_encoder_cls(model_init_kwargs)
        branch_encoder_kwargs = _resolve_nio_branch_encoder_kwargs(model_init_kwargs)
        net = build_nio(seed=114514, usct_time_steps=int(X_test[0].shape[-1]), device=device, branch_encoder_cls=branch_encoder_cls, branch_encoder_kwargs=branch_encoder_kwargs, **nio_kwargs)

    net.to(device)

    # --- 3. 加载权重 ---
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    net.eval()

    # --- 4. 预测推理 ---
    y_pred_list = []
    nio_grid_tensor = torch.as_tensor(X_test[1], dtype=torch.float32, device=device) if model_type == "NIO" else None
    
    # 这里计算的 length 会自动根据限制后的 sample_count 变小
    length = (sample_count + batch_size - 1) // batch_size
    print(f"总计预测 {sample_count} 个样本，分为 {length} 个 Batch。")
    
    with torch.no_grad():
        for i in range(length):
            start = batch_size * i
            end = min(batch_size * (i + 1), sample_count)
            
            if model_type in {"FourierDeepONet", "BranchTrunkFlower"}:
                branch_batch = torch.as_tensor(X_test[0][start:end]).to(device)
                trunk_batch = torch.as_tensor(X_test[1][start:end]).to(device)
                outputs = net((branch_batch, trunk_batch))
            elif model_type == "NIO":
                branch_batch = torch.as_tensor(X_test[0][start:end], dtype=torch.float32, device=device)
                outputs = net(branch_batch, nio_grid_tensor)
            elif model_type == "InversionNet":
                x_batch = torch.as_tensor(X_test[start:end]).to(device)
                outputs = net(x_batch)

            y_pred_list.append(outputs.cpu().numpy())
            torch.cuda.empty_cache()

    y_pred_norm = np.concatenate(y_pred_list, axis=0).squeeze().reshape(-1, sosmap_size[0], sosmap_size[1])
    y_pred_real = minmax_denormalize(y_pred_norm, VMIN, VMAX, scale=2)
    
    # >>> 修改点：保证 y_true 也只返回被预测的那一部分样本 <<<
    y_true_real = minmax_denormalize(y_true_orig[:sample_count], VMIN, VMAX, scale=2)
    
    return y_pred_real, y_true_real, sosmap_size


def plot_multi_model_comparison(y_true, model_preds, model_names, sample_idx, save_path, sosmap_size, mm_per_pixel=1.0):
    """
    绘制多个模型的预测对比图。
    结构：3行 × (N+1)列 网格
    - 前 N 列宽度相同（比例为1），最后 1 列很窄（比例为0.05）专用于放置 Colorbar。
    - 第一行：正中间放 Ground Truth，该行其他位置隐藏。
    - 第二行：各模型的预测图 + 最后一列的 Colorbar。
    - 第三行：各模型的绝对误差图 + 最后一列的 Colorbar。
    """
    N = len(model_preds)
    
    # 增加 1 列专门给 colorbar 用，width_ratios 保证所有预测图一样宽，最后一列窄
    fig, axes = plt.subplots(3, N + 1, figsize=(4 * N + 1, 12),
                             gridspec_kw={'width_ratios': [1] * N + [0.05]})
    
    label_fontsize = 14
    tick_fontsize = 14    # 刻度字体大小 
    title_fontsize = 18   # 标题字体大小

    extent = [0.0, sosmap_size[1] * mm_per_pixel, 0.0, sosmap_size[0] * mm_per_pixel]
    
    true_2d = y_true[sample_idx].reshape(sosmap_size)
    
    # 提前计算所有模型的误差，为了能统一第三行的 Colorbar 的极值范围
    errors = [np.abs(true_2d - pred[sample_idx].reshape(sosmap_size)) for pred in model_preds]
    max_err = max([np.max(err) for err in errors])
    max_err = max(max_err, 1e-5)

    # 1. 第一行：绘制 Ground Truth (放在靠中间列，隐藏其余列)
    gt_col = N // 2
    im_gt = None
    for j in range(N + 1):
        if j == gt_col:
            im_gt = axes[0, j].imshow(true_2d, cmap='jet', vmin=VMIN, vmax=VMAX, origin='lower', extent=extent, aspect='auto')
            axes[0, j].set_title('Ground Truth', fontsize=title_fontsize)
            axes[0, j].set_ylabel('X (mm)', fontsize=label_fontsize)
            axes[0, j].tick_params(axis='both', labelsize=tick_fontsize)
        else:
            axes[0, j].axis('off') # 隐藏空白子图（包括右上角本属于colorbar的位置）
            
    # 2. 第二行（预测图）和第三行（误差图）
    im_pred = None
    im_err = None
    
    for i, (pred_all, name) in enumerate(zip(model_preds, model_names)):
        pred_2d = pred_all[sample_idx].reshape(sosmap_size)
        err_2d = errors[i]
        
        # --- 第二行：预测图 ---
        ax_pred = axes[1, i]
        im_pred = ax_pred.imshow(pred_2d, cmap='jet', vmin=VMIN, vmax=VMAX, origin='lower', extent=extent, aspect='auto')
        ax_pred.set_title(name, fontsize=title_fontsize)
        #ax_pred.set_xticks([]) # 隐藏X轴刻度
        if i == 0:
            ax_pred.set_ylabel('X (mm)', fontsize=label_fontsize)
        else:
            #ax_pred.set_yticks([]) # 隐藏内侧Y轴刻度
            pass
        ax_pred.tick_params(axis='both', labelsize=tick_fontsize)

        # --- 第三行：绝对误差图 ---
        ax_err = axes[2, i]
        im_err = ax_err.imshow(err_2d, cmap='inferno', vmin=0, vmax=max_err, origin='lower', extent=extent, aspect='auto')
        ax_err.set_xlabel('Y (mm)', fontsize=label_fontsize)
        if i == 0:
            ax_err.set_ylabel('X (mm)', fontsize=label_fontsize)
        else:
            #ax_err.set_yticks([])
            pass
        ax_err.tick_params(axis='both', labelsize=tick_fontsize)

    # 3. 布局调整与添加 Colorbar
    plt.tight_layout()
    
    # 直接将最后一列对应的空白坐标轴作为 Colorbar 的绘制区域 (cax)
    # axes[1, N] 即为第二行、最后一列的坐标轴
    cbar_pred = plt.colorbar(im_gt, cax=axes[1, N])
    cbar_pred.set_label('Velocity (m/s)', fontsize=label_fontsize)
    cbar_pred.ax.tick_params(labelsize=tick_fontsize)

    # axes[2, N] 即为第三行、最后一列的坐标轴
    cbar_err = plt.colorbar(im_err, cax=axes[2, N])
    cbar_err.set_label('Absolute Error (m/s)', fontsize=label_fontsize)
    cbar_err.ax.tick_params(labelsize=tick_fontsize)

    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
    plt.close(fig)


def main():
    # =============== 配置区域 ===============
    data_kwargs = {
        "batch_size": 32,
        "split_ratio": 0.9,
        "total_data_num": 50000,
        "cache_h5_path": "/home/wkf/kwave-python/dataset/dataset_shuffle_0.140625-0.453125.h5",
        "cache_meta_path": "/home/wkf/kwave-python/dataset/dataset_shuffle_0.140625-0.453125_meta.json",
    }
    
    # 在这里添加您想要对比的所有模型
    models = [
        {
            "name": "InversionNet", 
            "type": "InversionNet", 
            "path": "/home/wkf/wkf_kwave/src/model_50K_5x2_configs_InversionNet/model-298000.pt"
        },
        {
            "name": "Neural Inverse Operator", 
            "type": "NIO", 
            "path": "/home/wkf/wkf_kwave/src/model_50K_5x2_configs_NIO_test1/model-254000.pt"
        },
        {
            "name": "Fourier-DeepONet-F", 
            "type": "FourierDeepONet", 
            "path": "/home/wkf/wkf_kwave/src/model_50K_5x2_configs_test0_0.140625-0.453125/model-230000.pt"
        }
    ]
    
    result_dir = "/home/wkf/wkf_kwave/Images/multi_model_comparisons_-F"
    samples_plot = 100 # <<< 这个变量现在控制画图数量，同时也限制模型推理数量
    mm_per_pixel = 0.1
    # =======================================

    os.makedirs(result_dir, exist_ok=True)
    
    all_preds = []
    model_names = []
    y_true_shared = None
    sosmap_size_shared = None

    # 1. 依次获取所有模型的预测结果
    for m in models:
        # >>> 将 samples_plot 作为参数传进函数中限制推理数量 <<<
        y_pred, y_true, map_size = get_predictions(m, data_kwargs, max_samples=samples_plot)
        all_preds.append(y_pred)
        model_names.append(m["name"])
        
        # Ground Truth 和 map 尺寸在所有模型间应该是相同的
        if y_true_shared is None:
            y_true_shared = y_true
            sosmap_size_shared = map_size

    # 2. 绘图：将所有模型集成在同一张图内
    print(f"\n========== 开始绘制对比图 ==========")
    # 这里的 len(y_true_shared) 已经被截断成了 samples_plot，直接遍历即可
    for i in range(len(y_true_shared)):
        save_path = os.path.join(result_dir, f'sample_{i}_multi_comparison.png')
        plot_multi_model_comparison(
            y_true=y_true_shared,
            model_preds=all_preds,
            model_names=model_names,
            sample_idx=i,
            save_path=save_path,
            sosmap_size=sosmap_size_shared,
            mm_per_pixel=mm_per_pixel
        )

if __name__ == "__main__":
    main()