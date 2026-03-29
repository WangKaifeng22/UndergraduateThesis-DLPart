import os
import numpy as np
import scipy.io
from numpy import expand_dims as unsqueeze
import matplotlib.pyplot as plt
from utils import *

# --- 核心数据加载函数 ---

def load_kwave_dataset(sos_root_dir, result_root_dir, x_param_list, y_param_list, num_samples_per_config):
    """
    加载多组 K-Wave 仿真数据并合并

    参数:
        sos_root_dir: SoSMap 数据集的**根**目录 (不包含具体参数子文件夹)
        result_root_dir: KwaveResult 结果的**根**目录 (不包含具体参数子文件夹)
        x_param_list: list, 文件夹命名中 x 部分的参数字符串列表 (如 ['3.0e-03', '3.5e-03'])
        y_param_list: list, 文件夹命名中 y 部分的参数字符串列表 (如 ['2.0e-03', '2.0e-03'])
        num_samples_per_config: int, 每组参数配置下读取的样本数量

    注意: x_param_list 和 y_param_list 长度必须一致，函数将成对读取。
    """

    y_list = []  # 存储所有 label
    branch_list = []  # 存储所有 branch input
    trunk_list = []  # 存储所有 trunk input

    # 确保参数列表长度一致
    assert len(x_param_list) == len(y_param_list), "x_param_list 和 y_param_list 长度必须相等"

    total_configs = len(x_param_list)
    print(f"开始加载数据... 共 {total_configs} 组参数配置，每组读取 {num_samples_per_config} 个样本。")

    # 遍历每一组参数配置 (x, y)
    for cfg_idx, (x_val, y_val) in enumerate(zip(x_param_list, y_param_list)):

        # 自动拼接文件夹名称，格式举例: "3.0e-03andInc2.0e-03"
        folder_name = f"{x_val}andInc{y_val}"

        # 拼接完整路径
        current_sos_dir = os.path.join(sos_root_dir, folder_name)
        current_result_dir = os.path.join(result_root_dir, folder_name)

        print(f"正在读取第 {cfg_idx + 1}/{total_configs} 组: 文件夹 [{folder_name}] ...")

        if not os.path.exists(current_sos_dir) or not os.path.exists(current_result_dir):
            print(f"Warning: 文件夹不存在，跳过: \n  SoS: {current_sos_dir} \n  Res: {current_result_dir}")
            continue

        count_loaded = 0
        idx = 0
        # 依次读取该文件夹下的样本
        max_tries = num_samples_per_config * 3

        while count_loaded < num_samples_per_config and idx <= max_tries:
            x_path = os.path.join(current_result_dir, f"sample_KwaveData_{idx:06d}.mat")
            y_path = os.path.join(current_sos_dir, f"sample_{idx:06d}.mat")

            if (not os.path.exists(x_path)) or (not os.path.exists(y_path)):
                if not os.path.exists(x_path):
                    print(f"  File missing: {x_path}, skipping index.")
                if not os.path.exists(y_path):
                    print(f"  File missing: {y_path}, skipping index.")
                idx += 1
                continue

            try:
                x_mat = scipy.io.loadmat(x_path)
                sensor_data_complex = x_mat["freq_data_complex_cat"]
                sensor_data_amp = np.concatenate(
                    [np.real(sensor_data_complex), np.imag(sensor_data_complex)], axis=0
                )

                coords = x_mat["sensor_coords"].astype(np.float32).flatten("F")

                mat_y = scipy.io.loadmat(y_path)
                velocity_map = mat_y["sample_data"].astype(np.float32)

                branch_list.append(unsqueeze(sensor_data_amp, 0))
                trunk_list.append(unsqueeze(coords, 0))
                y_list.append(unsqueeze(velocity_map, 0))

                count_loaded += 1
                idx += 1

            except Exception as e:
                print(f"  Error loading file pair index {idx}: {e}")
                idx += 1

        print(f"  -> 成功从 {folder_name} 读取 {count_loaded} 个样本。")

    if len(y_list) == 0:
        raise ValueError("错误: 没有读取到任何数据，请检查路径配置。")

    # --- C. 数据转换与归一化 (对合并后的数据统一处理) ---

    # 1. 处理 Y (Velocity)
    y_train = np.concatenate(y_list, axis=0).astype(np.float32)
    # 归一化声速 (1430 - 1650)
    y_train = minmax_normalize(y_train, VMIN, VMAX, scale=2)

    # 2. 处理 Branch (Sensor Data)
    X_branch = np.concatenate(branch_list, axis=0).astype(np.float32)

    # 归一化 Branch
    scaler_branch = MaxAbsScaler()
    scaler_branch.fit(X_branch)
    X_branch = scaler_branch.transform(X_branch)

    print(f"Global Branch data range after scaling: [{np.min(X_branch):.4f}, {np.max(X_branch):.4f}]")

    # 3. 处理 Trunk (坐标数据)
    X_trunk = np.concatenate(trunk_list, axis=0).astype(np.float32)
    X_trunk = minmax_normalize(X_trunk, -PHYSICAL_LIMIT, PHYSICAL_LIMIT, scale=2)

    print(f"Total Branch Input Shape: {X_branch.shape}")
    print(f"Total Trunk Input Shape:  {X_trunk.shape}")
    print(f"Total Output Shape:       {y_train.shape}")

    return (X_branch, X_trunk), y_train


def load_h5_dataset(
    h5_path: str,
    start: int | None = None,
    stop: int | None = None,
    indices: np.ndarray | list[int] | None = None,
):
    """Load preprocessed dataset from a single HDF5 file.

    Args:
        h5_path: path to the .h5 cache file.
        start/stop: optional slice range on the first dimension (like Python slicing).
            Example: start=0, stop=1000 loads first 1000 samples.
        indices: optional explicit indices (1D). If provided, start/stop are ignored.
            Example: indices=[0, 5, 7] loads 3 samples.

    Returns:
        (X_branch, X_trunk), y

    Notes:
        - This function loads selected arrays into memory.
        - For very large datasets, prefer the lazy reader `H5DeepONetDataset`.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        Xb = f["X_branch"]
        Xt = f["X_trunk"]
        yds = f["y"]

        n = int(Xb.shape[0])

        if indices is not None:
            idx = np.asarray(indices, dtype=np.int64)
            if idx.ndim != 1:
                raise ValueError("indices must be a 1D array/list")
            if len(idx) == 0:
                # return empty arrays with correct rank
                X_branch = np.empty((0, *Xb.shape[1:]), dtype=np.float32)
                X_trunk = np.empty((0, *Xt.shape[1:]), dtype=np.float32)
                y = np.empty((0, *yds.shape[1:]), dtype=np.float32)
                return (X_branch, X_trunk), y
            if np.any(idx < 0) or np.any(idx >= n):
                raise IndexError(f"indices out of range: valid [0, {n-1}]")

            X_branch = Xb[idx]
            X_trunk = Xt[idx]
            y = yds[idx]
        else:
            s = 0 if start is None else int(start)
            e = n if stop is None else int(stop)

            # handle negative indexing like Python
            if s < 0:
                s = n + s
            if e < 0:
                e = n + e

            s = max(0, min(n, s))
            e = max(0, min(n, e))
            if e < s:
                e = s

            X_branch = Xb[s:e]
            X_trunk = Xt[s:e]
            y = yds[s:e]

        return (X_branch, X_trunk), y


def get_dataset(
    split_ratio=0.96,
    samples_per_config=100,
    is_deeponet=True,
    x_params=None,
    y_params=None,
    cache_h5_path: str | None = None,
    h5_start: int | None = None,
    h5_stop: int | None = None,
    h5_indices: np.ndarray | list[int] | None = None,
    sos_root = None,
    kwave_root = None,
):
    """获取数据集的主入口。

    如果提供 cache_h5_path 且文件存在：优先从 HDF5 cache 读取。
    否则回退到原始 .mat 读取。

    可选：当使用 HDF5 cache 时，可通过 h5_start/h5_stop/h5_indices 只读取一部分数据。
    """
    if cache_h5_path is not None and os.path.exists(cache_h5_path):
        print(f"Loading dataset from HDF5 cache: {cache_h5_path}")
        (X_train_branch, X_train_trunk), y_train = load_h5_dataset(
            cache_h5_path,
            start=h5_start,
            stop=h5_stop,
            indices=h5_indices,
        )
    else:
        # 1. 设置根目录 (注意：这里不要包含具体的参数子文件夹，只要到父目录即可)
        if x_params is None:
            x_params = ["2.0e-03"]
        if y_params is None:
            y_params = ["3.0e-03"]


        # 2. 调用加载函数
        (X_train_branch, X_train_trunk), y_train = load_kwave_dataset(
            sos_root,
            kwave_root,
            x_params,
            y_params,
            samples_per_config,
        )

    total_samples = len(y_train)
    split_idx = int(total_samples * split_ratio)

    if not is_deeponet:
        X_train = X_train_branch[:split_idx]
        X_test = X_train_branch[split_idx:]
    else:
        X_train = (X_train_branch[:split_idx], X_train_trunk[:split_idx])
        X_test = (X_train_branch[split_idx:], X_train_trunk[split_idx:])

    y_train_set = y_train[:split_idx]
    y_test_set = y_train[split_idx:]

    return X_train, X_test, y_train_set, y_test_set


# --- 3. 调用示例 ---

if __name__ == "__main__":
    # 配置根路径
    sos_root = r"C:\Users\Administrator\Documents\Pre_Master_learn\graduationThesis\dataset\SoSMap"
    kwave_root = r"C:\Users\Administrator\Documents\Pre_Master_learn\graduationThesis\dataset\KwaveResult"

    # 想要测试的参数列表
    # 文件夹1: ...\3.0e-03andInc2.0e-03
    x_list = ["3.0e-03"]
    y_list = ["2.0e-03"]

    # 每个文件夹读多少个
    num_per_config = 100

    # 加载数据
    (X_train_branch, X_train_trunk), y_train = load_kwave_dataset(
        sos_root, kwave_root, x_list, y_list, num_per_config
    )

    # 简单的测试集划分
    total_data = len(y_train)
    split_idx = total_data - 5

    X_train = (X_train_branch[:split_idx], X_train_trunk[:split_idx])
    y_train_set = y_train[:split_idx]

    print("\nData Loading Complete.")
    print(f"Train Branch: {X_train[0].shape}")
    print(f"Train Trunk:  {X_train[1].shape}")
    print(f"Train Y:      {y_train_set.shape}")

    # visualize_samples(X_train[0], X_train[1], y_train_set, num_samples=3)