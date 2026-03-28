import os
import scipy.io
from numpy import expand_dims as unsqueeze
from utils import *

# --- 核心数据加载函数 ---

def load_kwave_dataset(data_dir, num_samples, result_dir):
    """
    加载 K-Wave 仿真数据用于 DeepONet 训练

    参数:
        data_dir: 声速图数据集根目录
        num_samples: 样本数量
        result_dir: K-Wave 结果目录

    返回:
        X_train: tuple (X_branch, X_trunk)
        y_train: numpy array
    """

    y_list = []  # 存储声速图 (Label)
    branch_list = []  # 存储传感器数据 (Branch Input)
    trunk_list = []
    print(f"Loading {num_samples} samples from {data_dir}...")

    skip = 0
    for i in range(1, num_samples + 1):
        i = i + skip
        # --- 加载 X_Branch (K-Wave 传感器数据) ---
        x_path = os.path.join(result_dir, f"sample_KwaveData_{i:04d}.mat")
        if not os.path.exists(x_path):
            print(f"Warning: {x_path} not found.")
            skip += 1
            continue
        x_mat = scipy.io.loadmat(x_path)
        # 获取 freq_data (复数)
        sensor_data_complex = x_mat['freq_data_complex_cat']  # shape: (transmitters_num, sensor_num, FFT_points)

        # --- 加载 Y (声速图) ---
        y_path = os.path.join(data_dir, f"sample_{i:04d}.mat")
        if not os.path.exists(y_path):
            print(f"Warning: {y_path} not found.")
            continue

        mat_y = scipy.io.loadmat(y_path)
        velocity_map = mat_y['sample_data'].astype(np.float32)  # shape: (384, 384)
        y_list.append(unsqueeze(velocity_map, 0))

        # 处理复数数据
        sensor_data_amp = np.concatenate([np.real(sensor_data_complex), np.imag(sensor_data_complex)], axis=0)
        branch_list.append(unsqueeze(sensor_data_amp, 0))

        # 获取坐标
        coords = x_mat['sensor_coords'].astype(np.float32)  # shape (2, 32)
        # 展平
        coords = coords.flatten('F')  # shape (64,)
        trunk_list.append(unsqueeze(coords, 0))
    # --- C. 数据转换与归一化 ---

    # 1. 处理 Y (Velocity)
    y_train = np.concatenate(y_list, axis=0).astype(np.float32)  # Shape: (N, 384*384)
    # 归一化声速
    y_train = minmax_normalize(y_train, VMIN, VMAX, scale=2)

    # 2. 处理 Branch (Sensor Data)
    X_branch = np.concatenate(branch_list, axis=0).astype(np.float32)
    #X_branch = log_transform(X_branch)
    #X_branch = minmax_normalize(X_branch, np.min(X_branch), np.max(X_branch), scale=2)
    scaler_branch = MaxAbsScaler()
    scaler_branch.fit(X_branch)
    X_branch = scaler_branch.transform(X_branch)
    print(f"Branch data range after scaling: [{np.min(X_branch):.4f}, {np.max(X_branch):.4f}]")

    print(f"Branch Input Shape: {X_branch.shape}")
    print(f"Output Shape: {y_train.shape}")

    # --- D. 生成 Trunk (坐标数据) ---
    X_trunk = np.concatenate(trunk_list, axis=0).astype(np.float32)  # Shape (N, 64)

    X_trunk = minmax_normalize(X_trunk, -PHYSICAL_LIMIT, PHYSICAL_LIMIT, scale=2)

    print(f"Trunk Input Shape: {X_trunk.shape} (Spatial Coordinates)")

    # 返回 tuple (X_inputs), y_label
    return (X_branch, X_trunk), y_train

def get_dataset(split_ratio=0.96, total_data_num=100, is_deeponet=True):
    # 配置路径
    dataset_path = r"C:\Users\Administrator\Documents\Pre_Master_learn\graduationThesis\dataset\SoSMap\3e-3andInc2e-3"
    result_path = r"C:\Users\Administrator\Documents\Pre_Master_learn\graduationThesis\dataset\KwaveResult"
    # 加载数据
    (X_train_branch, X_train_trunk), y_train = load_kwave_dataset(dataset_path, total_data_num, result_path)

    # 测试集划分
    split_idx = int(total_data_num * split_ratio)

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
    # 配置路径
    dataset_path = r"C:\Users\Administrator\Documents\Pre_Master_learn\graduationThesis\dataset\SoSMap\3e-3andInc2e-3"
    result_path = r"C:\Users\Administrator\Documents\Pre_Master_learn\graduationThesis\dataset\KwaveResult"

    # 建议至少加载 10 个数据进行测试，确保随机性
    total_data_num = 100

    # 加载数据
    (X_train_branch, X_train_trunk), y_train = load_kwave_dataset(dataset_path, total_data_num, result_path)

    # 简单的测试集划分
    split_idx = total_data_num - 2  # 留2个做测试

    # 构建训练集 (Branch, Trunk)
    X_train = (X_train_branch[:split_idx], X_train_trunk[:split_idx])
    y_train_set = y_train[:split_idx]

    # 构建测试集
    X_test = (X_train_branch[split_idx:], X_train_trunk[split_idx:])
    y_test_set = y_train[split_idx:]

    print("\nData Loading Complete.")
    print(f"Train Branch: {X_train[0].shape}")
    print(f"Train Trunk:  {X_train[1].shape}")
    print(f"Train Y:      {y_train_set.shape}")

    # --- 可视化验证 ---
    #print("\nStarting Visualization Check...")
    # 从训练集中随机抽样检查
    #visualize_samples(X_train[0], X_train[1], y_train_set, num_samples=3)