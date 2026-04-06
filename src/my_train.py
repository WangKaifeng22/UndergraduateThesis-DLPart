from utils import *
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['DDE_BACKEND'] = 'pytorch'
import deepxde as dde
from model_Unet_CNN import FourierDeepONet
from multi_data import get_dataset
from deepxde.callbacks import Callback
from soap import SOAP
from muon import MuonWithAuxAdam
import json
import time

# HDF5 backed dataset (lazy loading)
from h5_dataset import H5DeepONetDataset, H5DatasetConfig


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

        # 设置 Matplotlib 为非交互模式，防止训练中断
        plt.switch_backend('Agg')

    def on_epoch_end(self):
        current_it = self.model.train_state.iteration

        # 只有在达到指定周期时才执行
        if current_it % self.period == 0 and current_it > 0:
            self.visualize(current_it)

    def visualize(self, iteration):
        iteration += self.start_iteration
        # 1. 预测 
        # 我们只取测试集的前几个样本进行预测，避免显存爆炸
        # X_test 是 tuple (Branch, Trunk)
        # 取前1个样本作为切片: (Branch[:1], Trunk[:1])
        X_sample = (self.X_test[0][:1], self.X_test[1][:1])
        y_true_sample = self.y_test[:1]

        y_pred_sample = self.model.predict(X_sample)

        # 2. 数据处理 (Numpy)
        pred = y_pred_sample[0].squeeze()  # Shape: (384, 384)
        target = y_true_sample[0].squeeze()

        # 3. 反归一化 ([-1, 1] -> [1430, 1650])
        pred_real = minmax_denormalize(pred, VMIN, VMAX, 2)
        target_real = minmax_denormalize(target, VMIN, VMAX, 2)

        error_2d = np.abs(pred_real - target_real)

        # 4. 绘图
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
        mae = np.mean(error_2d)
        rmse = np.sqrt(np.mean((target_real - pred_real) ** 2))
        im2 = axes[2].imshow(error_2d, cmap='inferno', origin='lower')
        axes[2].set_title(f"Abs Error (MAE: {mae:.2f}, RMSE: {rmse:.2f})")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # 5. 保存
        save_path = os.path.join(self.save_dir, f"{self.filename}_{iteration:06d}.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Plot saved to {save_path}")


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
        # loss_train/loss_test can be a list (one per component); store their means
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
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

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


class Dataset(dde.data.Data):

    def __init__(self, X_train, y_train, X_test, y_test, test_batch_size=256):
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.train_sampler = dde.data.BatchSampler(len(X_train[0]), shuffle=True)

        self.test_batch_size = test_batch_size
        self.test_sampler = dde.data.BatchSampler(len(X_test[0]), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        return (self.train_x[0][indices], self.train_x[1][indices],), self.train_y[indices]

    def test(self):
        """
        重写 test 方法，防止一次性返回所有数据导致 OOM。
        每次只返回 test_batch_size 大小的数据用于计算 Loss 和 Metrics。
        """
        # 如果测试集很小，直接返回全部
        if len(self.test_y) <= self.test_batch_size:
            return self.test_x, self.test_y

        # 否则，随机采样一部分进行测试
        indices = self.test_sampler.get_next(self.test_batch_size)

        X_test_batch = (self.test_x[0][indices], self.test_x[1][indices])
        y_test_batch = self.test_y[indices]

        return X_test_batch, y_test_batch


samples_per_config = 10000
x_params = ["4.0e-03", "3.5e-03", "3.0e-03", "2.5e-03", "2.0e-03"]
y_params = ["2.0e-03", "2.0e-03","2.0e-03", "2.0e-03", "2.0e-03"]
sos_root = "/home/wkf/kwave-python/dataset/SoSMap"
kwave_root = "/home/wkf/kwave-python/dataset/KwaveResult"
cache_h5_path = "/home/wkf/kwave-python/dataset/dataset_shuffle_0.140625-0.453125.h5"
meta_h5_path = "/home/wkf/kwave-python/dataset/dataset_shuffle_0.140625-0.453125_meta.json"


def main(dataset, task, resume_training=False, batch_size=32, lazy: bool = False, test: bool = False,
         model_path=None, path=None,
         start_iteration=0, total_epoch=250, enable_timing: bool = True, split_ratio = 0.8, seed = 114514):
    set_seed(seed)
    total_data_num = int(samples_per_config * len(x_params))
    test_batch_size = int(batch_size * ((1 - split_ratio) / split_ratio))

    # 如果提供 HDF5 cache，则用 lazy 方式读取，避免 OOM
    if lazy:
        print(f"Using HDF5 cache dataset: {cache_h5_path}")
        data = H5DeepONetDataset(
            H5DatasetConfig(h5_path=cache_h5_path, split_ratio=split_ratio, test_batch_size=test_batch_size,
                            total_data_num=total_data_num),
            is_deeponet=True,
            seed=seed,
            enable_timing=enable_timing,
        )
        trunk_dim = data.trunk_dim
        # 用极小 batch 读取一条用于 PlottingCallback
        X_test, y_test = data.test()
    else:
        X_train, X_test, y_train, y_test = get_dataset(split_ratio, samples_per_config, True, x_params=x_params,
                                                       y_params=y_params
                                                       , cache_h5_path=cache_h5_path, h5_start=0, h5_stop=total_data_num
                                                       , sos_root=sos_root, kwave_root=kwave_root)
        # 创建Dataset实例
        data = Dataset(X_train, y_train, X_test, y_test, test_batch_size=test_batch_size)
        trunk_dim = X_test[1].shape[1]

    if test:
        preflight_check_xy(X_test, y_test, name="test", vis_index=3)
        return

    # 超参
    width = 64; modes1 = 12; modes2 = 20; regularization = ["l2", 3e-6]; merge_operation = "mul"
    use_hfs_block123 = False
    hfs_patch_size = (16, 8, 4)

    net = FourierDeepONet(num_parameter=trunk_dim, width=width, modes1=modes1, modes2=modes2,
                          regularization=regularization, merge_operation=merge_operation,
                          use_hfs_block123=use_hfs_block123, hfs_patch_size=hfs_patch_size)
    model = dde.Model(data, net)

    # --- save model build config before training (no weights) ---
    model_config = {
        "model_type": "FourierDeepONet",
        "model_init_kwargs": {
            "num_parameter": int(trunk_dim),
            "width": width,
            "modes1": modes1,
            "modes2": modes2,
            "regularization": regularization,
            "merge_operation": merge_operation,
            "use_hfs_block123": use_hfs_block123,
            "hfs_patch_size": list(hfs_patch_size),
        },
        "data": {
            "cache_h5_path": cache_h5_path,
            "split_ratio": split_ratio,
        },
    }
    with open(os.path.join(path, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)

    # optimizer: adam, adamw, muon, soap, auon
    optimizer_name = "adamw"
    if optimizer_name == "soap":
        optimizer = SOAP(
            model.parameters(),
            lr=2e-3,
            betas=(0.95, 0.95),
            shampoo_beta=0.99,
            eps=1e-8,
            weight_decay=0.01,
            precondition_frequency=10,
            max_precond_dim=8192,  # 可按显存调
            merge_dims=True,  # 卷积场景建议开
            precondition_1d=False,
            normalize_grads=False,
            data_format="channels_first",
            correct_bias=True,
        )
    elif optimizer_name == "muon":  # muon适用batch size较大时
        # 1. 定义哪些模块属于 "Embedding" (输入层) 和 "Head" (输出层)
        # 这些部分的参数全部使用 AdamW
        embed_modules = [net.branch, net.trunk, net.fusion_layer]
        head_modules = [net.merger.out_conv2]

        # 获取这些模块所有参数的 ID，用于后续从主体中排除
        embed_head_params_ids = set()
        for m in embed_modules + head_modules:
            for p in m.parameters():
                embed_head_params_ids.add(id(p))

        # 2. 分离参数
        muon_params = []  # 用于 Muon (Body 中的权重矩阵)
        adam_aux_params = []  # 用于 AdamW (Body 中的偏置/Norm + Embed + Head)

        for p in net.parameters():
            # 如果参数属于 Embed 或 Head，直接归入 AdamW 组
            if id(p) in embed_head_params_ids:
                adam_aux_params.append(p)
            else:
                # 剩下的属于 Body (net.merger 的主体部分)
                # Muon 只优化 >= 2D 的张量 (卷积核、全连接权重)

                if p.ndim >= 2 and not p.is_complex():
                    muon_params.append(p)
                elif p.ndim >= 2 and p.is_complex():
                    # **建议**：复数权重建议暂时用 AdamW，除非你确认 Muon 实现支持复数牛顿迭代
                    print(f"Info: Complex param shape {p.shape} assigned to AdamW.")
                    adam_aux_params.append(p)
                else:
                    # 偏置 (Bias), 增益 (Gain), 1D 向量 -> AdamW
                    adam_aux_params.append(p)

        # 3. 构建参数组
        param_groups = [
            # Muon 组: 较大的学习率 (通常 0.02 - 0.05), 使用 Muon 优化器
            dict(params=muon_params, use_muon=True,
                 lr=0.02, weight_decay=0.01),

            # AdamW 组: 标准学习率 (3e-4), 使用 AdamW 优化器
            dict(params=adam_aux_params, use_muon=False,
                 lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
        ]

        # 4. 实例化优化器
        optimizer = MuonWithAuxAdam(param_groups)
    # elif optimizer_name == "auon":
    elif optimizer_name == "adam" or optimizer_name == "adamw":
        optimizer = optimizer_name
    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} not implemented.")

    iterations_per_epoch = (total_data_num + batch_size - 1) // batch_size
    total_iterations = total_epoch * iterations_per_epoch
    remaining_iterations = total_iterations - start_iteration

    model.compile(optimizer=optimizer, lr=2.5e-3, loss=loss_func_L1,
                  decay=("lambda", lambda step: large_dataset_schedule(step=step, total_steps=total_iterations,
                                                                       total_epochs=total_epoch,
                                                                       start_it=start_iteration)),
                  metrics=[lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),  # MAE
                           lambda y_true, y_pred: np.sqrt(np.mean(((y_true - y_pred) ** 2)))]  # RMSE
                  , )

    checker = dde.callbacks.ModelCheckpoint(
        f"{path}/model",
        save_better_only=True,
        period=1000
    )

    plotter = PlottingCallback(
        X_test,
        y_test,
        period=1000,
        save_dir=f"{path}/plots",
        start_iteration=start_iteration
    )
    log_period = 50
    loss_logger = LossHistoryCallback(period=log_period, save_dir=f"{path}/logs", start_iteration=start_iteration)

    timing_logger = None
    if enable_timing:
        timing_logger = TimingCallback(period=1, save_dir=f"{path}/logs", start_iteration=start_iteration, sync_cuda=True)
        remaining_iterations = 25
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if remaining_iterations > 0:
        callbacks = [checker, plotter, loss_logger]
        if timing_logger is not None:
            callbacks.append(timing_logger)

        losshistory, train_state = model.train(
            iterations=remaining_iterations,
            batch_size=batch_size,
            display_every=log_period,
            callbacks=callbacks,
            model_save_path=f"{path}/model",
            disregard_previous_best=True,
            model_restore_path=model_path
        )

        #dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    else:
        print("Training already completed!")


if __name__ == "__main__":
    dataset = "50K"
    task = "5x2_configs"
    path = f'./model_{dataset}_{task}_test0_FiLM_0.140625-0.453125'
    os.makedirs(path, exist_ok=True)
    main(dataset=dataset, task=task, batch_size=32, lazy=True, test=False,
         model_path=None, path=path, 
         start_iteration=0, total_epoch=200, enable_timing=False, split_ratio=0.9, seed=114514)

