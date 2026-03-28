import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['DDE_BACKEND'] = 'pytorch'
import deepxde as dde
from InversionNet import InversionNet  # 确保 InversionNet.py 在同目录下
from multi_data import get_dataset
from utils import *
from deepxde.callbacks import Callback
from soap import SOAP
from muon import MuonWithAuxAdam
from my_train import LossHistoryCallback
import json

class PlottingCallback(Callback):
    def __init__(self, X_test, y_test, period=1000, save_dir="./training_plots", filename="it", start_iteration=0):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.period = period
        self.save_dir = save_dir
        self.filename = filename
        self.start_iteration = start_iteration

        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        plt.switch_backend('Agg')

    def on_epoch_end(self):
        current_it = self.model.train_state.iteration
        if current_it % self.period == 0 and current_it > 0:
            self.visualize(current_it)

    def visualize(self, iteration):
        iteration += self.start_iteration
        # --- 修改点 1: 适配单输入 ---
        # 取测试集的前1个样本
        X_sample = self.X_test[:1]  # InversionNet 输入直接切片即可
        y_true_sample = self.y_test[:1]

        # DeepXDE 的 model.predict 接受 numpy array
        y_pred_sample = self.model.predict(X_sample)

        # 数据处理
        pred = y_pred_sample[0].squeeze()  # Shape: (384, 384)
        target = y_true_sample[0].squeeze()

        # 反归一化
        pred_real = minmax_denormalize(pred, VMIN, VMAX,2)
        target_real = minmax_denormalize(target, VMIN, VMAX,2)

        error_map = np.abs(pred_real - target_real)

        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Prediction
        im0 = axes[0].imshow(pred_real, cmap='jet', vmin=VMIN, vmax=VMAX, origin='lower')
        axes[0].set_title(f"Pred (Iter {iteration})")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Ground Truth
        im1 = axes[1].imshow(target_real, cmap='jet', vmin=VMIN, vmax=VMAX, origin='lower')
        axes[1].set_title("Ground Truth")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Error
        im2 = axes[2].imshow(error_map, cmap='inferno', origin='lower')
        axes[2].set_title(f"Abs Error (Max: {np.max(error_map):.1f})")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{self.filename}_{iteration:06d}.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"Plot saved to {save_path}")


class Dataset(dde.data.Data):
    def __init__(self, X_train, y_train, X_test, y_test, test_batch_size=200):
        # --- 修改点 2: 假设输入 X_train 已经是单个 tensor/array [N, 64, 32, 1024] ---
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        # BatchSampler 使用样本总数
        self.train_sampler = dde.data.BatchSampler(len(X_train), shuffle=True)
        self.test_batch_size = test_batch_size
        self.test_sampler = dde.data.BatchSampler(len(X_test), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        # --- 修改点 3: 直接返回切片，不再组装 tuple ---
        return self.train_x[indices], self.train_y[indices]

    def test(self):
        if len(self.test_y) <= self.test_batch_size:
            return self.test_x, self.test_y
        indices = self.test_sampler.get_next(self.test_batch_size)
        # --- 修改点 4: 直接返回切片 ---
        X_test_batch = self.test_x[indices]
        y_test_batch = self.test_y[indices]
        return X_test_batch, y_test_batch



def main(dataset, task, resume_training=False, batch_size=8):
    seed = 114514
    set_seed(seed)
    split_ratio = 0.96
    total_data_num = 100

    X_train, X_test, y_train, y_test = get_dataset(split_ratio, total_data_num, is_deeponet = False)

    # 创建Dataset实例
    data = Dataset(X_train, y_train, X_test, y_test, test_batch_size=4)

    # 超参
    dim0 = 64; dim1 = 64; dim2 = 64; dim3 = 128; dim4 = 256; dim5 = 512; regularization = ["l2", 3e-6]

    # --- 修改点 5: 初始化 InversionNet ---
    net = InversionNet(dim0=dim0, dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4, dim5=dim5, regularization=regularization)
    model = dde.Model(data, net)

    path = f'./model_{dataset}_{task}_InversionNet_4'
    os.makedirs(path, exist_ok=True)
    ckpt_dir = os.path.join(path, "model")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- NEW: save model build config before training (no weights) ---
    model_config = {
        "model_type": "InversionNet",
        "model_init_kwargs": {
            "dim0": dim0,
            "dim1": dim1,
            "dim2": dim2,
            "dim3": dim3,
            "dim4": dim4,
            "dim5": dim5,
            "regularization": regularization,
        },
    }
    with open(os.path.join(path, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)

    # Optimizer 配置
    optimizer_name = "adam"

    if optimizer_name == "soap":
        optimizer = SOAP(
            model.parameters(),
            lr=2e-3, betas=(0.95, 0.95), shampoo_beta=0.99, eps=1e-8,
            weight_decay=0.01, precondition_frequency=10, max_precond_dim=4096,  # 调小 dim 省显存
            merge_dims=True, precondition_1d=False, normalize_grads=False,
            data_format="channels_first", correct_bias=True,
        )
    elif optimizer_name == "muon":
        # --- 修改点 6: 通用的 Muon 配置 (自动识别 2D 卷积层) ---
        muon_params = []
        adam_aux_params = []

        for name, p in net.named_parameters():
            if not p.requires_grad:
                continue

            # 策略:
            # 1. Conv2d/ConvTranspose2d 的权重 (ndim >= 4) -> Muon
            # 2. 只有 ndim >= 2 才能用 Muon
            # 3. 所有的 bias, norm, embedding, output head -> AdamW

            # 这里简单判断: 如果是 >= 2D 的权重且不是 Bias，尝试用 Muon
            # 但为了安全，我们优先把 Bias 和 1D 参数放入 AdamW
            if p.ndim < 2:
                adam_aux_params.append(p)
            elif "bias" in name or "norm" in name:
                adam_aux_params.append(p)
            else:
                # 剩下的通常是 Conv weight (4D) 或 Linear weight (2D)
                # InversionNet 主要是 Conv weight
                muon_params.append(p)

        print(f"Muon params: {len(muon_params)}, AdamW params: {len(adam_aux_params)}")

        param_groups = [
            # Muon 组: 较大的学习率
            dict(params=muon_params, use_muon=True, lr=0.02, weight_decay=0.01),
            # AdamW 组: 标准学习率
            dict(params=adam_aux_params, use_muon=False, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
        ]
        optimizer = MuonWithAuxAdam(param_groups)
    elif optimizer_name == "adam" or optimizer_name == "adamw":
        optimizer = optimizer_name
    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} not implemented.")

    model.compile(optimizer=optimizer, lr=1e-3, loss=loss_func_L1, decay=("step", 5000, 0.9),
                  metrics=[lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))])  # MAE

    checker = dde.callbacks.ModelCheckpoint(f"{path}/model", save_better_only=False, period=5000)

    plotter = PlottingCallback(X_test, y_test, period=2500, save_dir=f"{path}/plots",)

    log_period = 500
    start_iteration = 0
    loss_logger = LossHistoryCallback(period=log_period + start_iteration, save_dir=f"{path}/logs")

    total_epoch = 5000
    iterations_per_epoch = (total_data_num + batch_size - 1) // batch_size
    total_iterations = total_epoch * iterations_per_epoch
    remaining_iterations = total_iterations - start_iteration

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_path=None

    losshistory, train_state = model.train(
        iterations=remaining_iterations,
        batch_size=batch_size,
        display_every=log_period,
        callbacks=[checker, plotter, loss_logger],
        model_save_path=f"{path}/model",
        disregard_previous_best=True,
        model_restore_path=model_path
    )

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main(dataset="3e-3", task="Inc2e-3", batch_size=8)