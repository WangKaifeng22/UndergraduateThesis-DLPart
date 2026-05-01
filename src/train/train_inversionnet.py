import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['DDE_BACKEND'] = 'pytorch'
import deepxde as dde
from models.InversionNet import InversionNet  # 确保 InversionNet.py 在同目录下
from utils.multi_data import get_dataset
from utils.utils import *
from deepxde.callbacks import Callback
from optimizer.soap import SOAP
from optimizer.muon import MuonWithAuxAdam
import json
import time
from utils.h5_dataset import H5DeepONetDataset, H5DatasetConfig
from training_callbacks import TensorBoardCallback, SwanLabCallback


class LossHistoryCallback(Callback):
    def __init__(self, period=500, save_dir="./training_logs", filename="loss_history", start_iteration=0):
        super().__init__()
        self.period = period
        self.save_dir = save_dir
        self.filename = filename
        self.start_iteration = start_iteration
        os.makedirs(self.save_dir, exist_ok=True)
        self.filepath = os.path.join(self.save_dir, f"{self.filename}.csv")
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write("iteration,loss_train,loss_test\n")

    def on_epoch_end(self):
        iteration = self.model.train_state.iteration + self.start_iteration
        if iteration % self.period != 0:
            return
        loss_train = self.model.train_state.loss_train
        loss_test = self.model.train_state.loss_test
        loss_train_mean = float(np.mean(loss_train)) if loss_train is not None else np.nan
        loss_test_mean = float(np.mean(loss_test)) if loss_test is not None else np.nan
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(f"{iteration},{loss_train_mean},{loss_test_mean}\n")


class TimingCallback(Callback):
    def __init__(self, period=1, save_dir="./training_logs", filename="iter_time", start_iteration=0, sync_cuda=True):
        super().__init__()
        self.period = period
        self.save_dir = save_dir
        self.filename = filename
        self.start_iteration = start_iteration
        self.sync_cuda = sync_cuda
        self._t0 = None
        os.makedirs(self.save_dir, exist_ok=True)
        self.filepath = os.path.join(self.save_dir, f"{self.filename}.csv")
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write("iteration,wall_time_sec\n")

    def _maybe_sync(self):
        if not self.sync_cuda:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_epoch_begin(self):
        self._maybe_sync()
        self._t0 = time.perf_counter()

    def on_epoch_end(self):
        if self._t0 is None:
            return
        self._maybe_sync()
        dt = time.perf_counter() - self._t0
        iteration = self.model.train_state.iteration + self.start_iteration
        if iteration % self.period != 0:
            return
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(f"{iteration},{dt:.6f}\n")


def preflight_check_single_input(X_test, y_test, name="test", vis_index=0, save_path="./test.png", dpi=300):
    X = np.asarray(X_test)
    y = np.asarray(y_test)

    assert X.ndim >= 2, f"X_test ndim too small: {X.ndim}"
    assert y.ndim >= 2, f"y_test ndim too small: {y.ndim}"
    assert X.shape[0] == y.shape[0], f"Batch mismatch: X={X.shape[0]}, y={y.shape[0]}"

    assert np.isfinite(X).all(), "X_test contains NaN/Inf"
    assert np.isfinite(y).all(), "y_test contains NaN/Inf"

    print(f"[{name}] X_test shape={X.shape}, dtype={X.dtype}, min={X.min():.6g}, max={X.max():.6g}")
    print(f"[{name}] y_test shape={y.shape}, dtype={y.dtype}, min={y.min():.6g}, max={y.max():.6g}")

    idx = int(np.clip(vis_index, 0, X.shape[0] - 1))
    y_vis = minmax_denormalize(y[idx].squeeze(), VMIN, VMAX, 2)
    x_vis = X[idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    im0 = axes[0].imshow(y_vis, cmap="jet", origin="lower")
    axes[0].set_title(f"y sample={idx}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    x_slice = x_vis[0] if x_vis.ndim >= 3 else np.squeeze(x_vis)
    im1 = axes[1].imshow(x_slice, cmap="viridis", aspect="auto")
    axes[1].set_title(f"X sample={idx} slice")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"[{name}] saved preflight figure to: {save_path}")

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
        pred = y_pred_sample[0].squeeze()  
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

from my_train import samples_per_config, x_params, y_params, cache_h5_path, sos_root, kwave_root


def main(
    dataset,
    task,
    batch_size=32,
    lazy: bool = False,
    test=False,
    model_path=None,
    path=None,
    start_iteration=0,
    total_epoch=250,
    enable_timing=True,
    split_ratio=0.8,
    seed=114514,
    optimizer_name="adamw",
    enable_tensorboard: bool = False,
    tensorboard_log_dir=None,
    tensorboard_histograms: bool = False,
    enable_swanlab: bool = False,
    swanlab_project="Kwave-InversionNet",
    swanlab_experiment=None,
    log_period=50,
    #resume_training=False,
):
    if path is None:
        path = f"./model_{dataset}_{task}_InversionNet_4"

    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "model"), exist_ok=True)

    set_seed(seed)

    total_data_num = int(samples_per_config * len(x_params))
    test_batch_size = max(1, int(batch_size * ((1 - split_ratio) / split_ratio)))

    if lazy:
        print(f"Using HDF5 cache dataset: {cache_h5_path}")
        data = H5DeepONetDataset(
            H5DatasetConfig(
                h5_path=cache_h5_path,
                split_ratio=split_ratio,
                test_batch_size=test_batch_size,
                total_data_num=total_data_num,
            ),
            is_deeponet=False,
            seed=seed,
            enable_timing=enable_timing,
        )
        X_test, y_test = data.test()
    else:
        X_train, X_test, y_train, y_test = get_dataset(
            split_ratio,
            samples_per_config,
            False,
            x_params=x_params,
            y_params=y_params,
            cache_h5_path=cache_h5_path,
            h5_start=0,
            h5_stop=total_data_num,
            sos_root=sos_root,
            kwave_root=kwave_root,
        )
        data = Dataset(X_train, y_train, X_test, y_test, test_batch_size=test_batch_size)

    if test:
        preflight_check_single_input(X_test, y_test, name="test", vis_index=0, save_path=f"{path}/preflight_test.png")
        if lazy:
            data.close()
        return

    # 超参
    dim0 = 64; dim1 = 64; dim2 = 64; dim3 = 128; dim4 = 256; dim5 = 512; regularization = ["l2", 3e-6]

    # --- 修改点 5: 初始化 InversionNet ---
    net = InversionNet(dim0=dim0, dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4, dim5=dim5, regularization=regularization)
    model = dde.Model(data, net)

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
        "data": {
            "cache_h5_path": cache_h5_path,
            "split_ratio": split_ratio,
        },
    }
    with open(os.path.join(path, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)

    if optimizer_name == "soap":
        optimizer = SOAP(
            net.parameters(),
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

    iterations_per_epoch = (total_data_num + batch_size - 1) // batch_size
    total_iterations = total_epoch * iterations_per_epoch
    remaining_iterations = total_iterations - start_iteration

    model.compile(
        optimizer=optimizer,
        lr=2.5e-3, #1e-3
        loss=loss_func_L1,
        decay=(
            "lambda",
            lambda step: large_dataset_schedule(
                step=step,
                total_steps=total_iterations,
                total_epochs=total_epoch,
                start_it=start_iteration,
            ),
        ),
        metrics=[
            lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
            lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred) ** 2)),
        ],
    )

    checker = dde.callbacks.ModelCheckpoint(f"{path}/model", save_better_only=True, period=2000)

    plotter = PlottingCallback(X_test, y_test, period=1000, save_dir=f"{path}/plots", start_iteration=start_iteration)

    tensorboard_logger = None
    loss_logger = None
    if enable_tensorboard:
        if tensorboard_log_dir is None:
            tensorboard_log_dir = os.path.join(path, "tensorboard")
        tensorboard_logger = TensorBoardCallback(
            log_dir=tensorboard_log_dir,
            period=log_period,
            start_iteration=start_iteration,
            log_histograms=tensorboard_histograms,
        )
    if not enable_tensorboard and not enable_swanlab:
        loss_logger = LossHistoryCallback(period=log_period, save_dir=f"{path}/logs", start_iteration=start_iteration)

    if enable_tensorboard and enable_swanlab:
        print("Warning: Both TensorBoard and SwanLab logging are enabled. This may lead to redundant logs.")

    swanlab_logger = None
    if enable_swanlab:
        swanlab_logger = SwanLabCallback(
            project=swanlab_project,
            experiment_name=swanlab_experiment or f"exp_{start_iteration}",
            log_dir=os.path.join(path, "swanlog"),
            X_test=(X_test[0][:1], X_test[1][:1]) if isinstance(X_test, tuple) or isinstance(X_test, list) else X_test[:1],
            y_test=y_test[:1],
            plot_period=1000,
            period=log_period,
            start_iteration=start_iteration,
            config=model_config
        )

    timing_logger = None
    if enable_timing:
        timing_logger = TimingCallback(period=1, save_dir=f"{path}/logs", start_iteration=start_iteration, sync_cuda=True)
        remaining_iterations = 25  
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if remaining_iterations > 0:
        callbacks = [checker, plotter]
        if tensorboard_logger is not None:
            callbacks.append(tensorboard_logger)
        if swanlab_logger is not None:
            callbacks.append(swanlab_logger)
        if loss_logger is not None:
            callbacks.append(loss_logger)
        if timing_logger is not None:
            callbacks.append(timing_logger)

        losshistory, train_state = model.train(
            iterations=remaining_iterations,
            batch_size=batch_size,
            display_every=log_period,
            callbacks=callbacks,
            model_save_path=f"{path}/model",
            disregard_previous_best=True,
            model_restore_path=model_path,
        )

        # dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    else:
        print("Training already completed!")

    if lazy:
        data.close()


if __name__ == "__main__":
    dataset = "50K"
    task = "5x2_configs"
    path = f"./model_{dataset}_{task}_InversionNet"
    os.makedirs(path, exist_ok=True)

    main(
        dataset=dataset,
        task=task,
        batch_size=32,
        lazy=True,
        test=False,
        model_path=None,
        path=path,
        start_iteration=0,
        total_epoch=200,
        enable_timing=True,
        split_ratio=0.9,
        seed=114514,
        optimizer_name="adamw",
    )