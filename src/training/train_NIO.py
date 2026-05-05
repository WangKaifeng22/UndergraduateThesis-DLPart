import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from utils.utils import *
os.environ["DDE_BACKEND"] = "pytorch"

import json
import time

import deepxde as dde
import numpy as np
import torch
from deepxde.callbacks import Callback

from utils.H5NIODataset import H5NIOConfig, H5NIODataset
from models.model_NIO import EncoderUSCT, EncoderUSCTHelm2, NIOUltrasoundCTAbl
from training.training_callbacks import TensorBoardCallback, SwanLabCallback


class PlottingCallback(Callback):
    def __init__(self, X_test, y_test, period=1000, save_dir="./training_plots", filename="it", start_iteration=0):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.period = period
        self.save_dir = save_dir
        self.filename = filename
        self.start_iteration = start_iteration

        os.makedirs(self.save_dir, exist_ok=True)
        plt.switch_backend("Agg")

    def on_epoch_end(self):
        current_it = self.model.train_state.iteration
        if current_it % self.period == 0 and current_it > 0:
            self.visualize(current_it)

    def visualize(self, iteration):
        iteration += self.start_iteration

        # NIO trunk input is a fixed grid; only branch uses mini-batch slicing.
        X_sample = (self.X_test[0][:1], self.X_test[1])
        y_true_sample = self.y_test[:1]

        y_pred_sample = self.model.predict(X_sample)

        pred = y_pred_sample[0].squeeze()
        target = y_true_sample[0].squeeze()

        pred_real = minmax_denormalize(pred, VMIN, VMAX, 2)
        target_real = minmax_denormalize(target, VMIN, VMAX, 2)
        error_map = np.abs(pred_real - target_real)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im0 = axes[0].imshow(pred_real, cmap="jet", vmin=VMIN, vmax=VMAX, origin="lower")
        axes[0].set_title(f"Pred (Iter {iteration})")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(target_real, cmap="jet", vmin=VMIN, vmax=VMAX, origin="lower")
        axes[1].set_title("Ground Truth")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(error_map, cmap="inferno", origin="lower")
        axes[2].set_title(f"Abs Error (Max: {np.max(error_map):.1f})")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{self.filename}_{iteration:06d}.png")
        plt.savefig(save_path, dpi=100)
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


from training.train import samples_per_config, x_params, y_params, cache_h5_path

def build_nio(
    seed,
    usct_time_steps,
    device,
    usct_hidden=256,
    trunk_hidden_layers=4,
    trunk_neurons=128,
    trunk_n_basis=256,
    fno_modes=16,
    fno_width=64,
    fno_n_layers=4,
    branch_encoder_cls=EncoderUSCTHelm2,
    branch_encoder_kwargs=None,
):
    if branch_encoder_kwargs is None:
        branch_encoder_kwargs = {}

    network_properties_branch = {}
    network_properties_trunk = {
        "n_hidden_layers": trunk_hidden_layers,
        "neurons": trunk_neurons,
        "act_string": "gelu",
        "retrain": seed,
        "dropout_rate": 0.0,
        "n_basis": trunk_n_basis,
    }
    fno_architecture = {
        "modes": fno_modes,
        "width": fno_width,
        "n_layers": fno_n_layers,
    }

    return NIOUltrasoundCTAbl(
        input_dimensions_trunk=2,
        network_properties_branch=network_properties_branch,
        network_properties_trunk=network_properties_trunk,
        fno_architecture=fno_architecture,
        device=device,
        retrain_seed=seed,
        padding_frac=1 / 4,
        usct_time_steps=usct_time_steps,
        usct_hidden=usct_hidden,
        branch_encoder_cls=branch_encoder_cls,
        branch_encoder_kwargs=branch_encoder_kwargs,
        regularization = ["l2", 3e-6],
    )


def main(
    dataset,
    task,
    grid_npy_path,
    batch_size=32,
    test=False,
    model_path=None,
    path=None,
    start_iteration=0,
    total_epoch=250,
    enable_timing=True,
    split_ratio = 0.8,
    seed = 114514,
    branch_encoder_cls=EncoderUSCTHelm2,
    branch_encoder_kwargs=None,
    enable_tensorboard: bool = False,
    tensorboard_log_dir=None,
    tensorboard_histograms: bool = False,
    enable_swanlab: bool = False,
    swanlab_project="Kwave-NIO",
    swanlab_experiment=None,
    log_period = 50,
):
    set_seed(seed)
    total_data_num = int(samples_per_config * len(x_params))
    test_batch_size = int(batch_size * ((1 - split_ratio) / split_ratio))

    data = H5NIODataset(
        H5NIOConfig(
            h5_path=cache_h5_path,
            grid_npy_path=grid_npy_path,
            split_ratio=split_ratio,
            test_batch_size=test_batch_size,
            total_data_num=total_data_num,
            squeeze_y_channel=True,
            
        ),
        seed=seed,
        enable_timing=enable_timing,
        
    )
    X_test, y_test = data.test()

    if test:
        preflight_check_xy(X_test, y_test, name="test", vis_index=0)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usct_time_steps = int(X_test[0].shape[-1])
    nio_arch = {
        "usct_hidden": 128,
        "trunk_hidden_layers": 4,
        "trunk_neurons": 128,
        "trunk_n_basis": 256,
        "fno_modes": 20,
        "fno_width": 64,
        "fno_n_layers": 4,
    }

    net = build_nio(
        seed=seed,
        usct_time_steps=usct_time_steps,
        device=device,
        branch_encoder_cls=branch_encoder_cls,
        branch_encoder_kwargs=branch_encoder_kwargs,
        **nio_arch,
    )
    model = dde.Model(data, net)

    if branch_encoder_kwargs is None:
        branch_encoder_kwargs = {}

    model_config = {
        "model_type": "NIOUltrasoundCTAbl",
        "model_init_kwargs": {
            "input_dimensions_trunk": 2,
            "usct_time_steps": usct_time_steps,
            **nio_arch,
            "grid_npy_path": grid_npy_path,
            "branch_encoder_cls": getattr(branch_encoder_cls, "__name__", str(branch_encoder_cls)),
            "branch_encoder_kwargs": branch_encoder_kwargs,
        },
        "data": {
            "cache_h5_path": cache_h5_path,
            "split_ratio": split_ratio,
        },
    }
    with open(os.path.join(path, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)

    optimizer_name = "adamw"
    optimizer = optimizer_name if optimizer_name in {"adam", "adamw"} else "adamw"

    iterations_per_epoch = (total_data_num + batch_size - 1) // batch_size
    total_iterations = total_epoch * iterations_per_epoch
    remaining_iterations = total_iterations - start_iteration

    model.compile(
        optimizer=optimizer,
        lr=2.5e-3,
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

    checker = dde.callbacks.ModelCheckpoint(
        f"{path}/model",
        save_better_only=True,
        period=2000,
    )

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
        # NIO trunk input is a fixed grid; only branch uses mini-batch slicing.
        X_sample = (X_test[0][:1], X_test[1]) if isinstance(X_test, (tuple, list)) else X_test[:1]
        swanlab_logger = SwanLabCallback(
            project=swanlab_project,
            experiment_name=swanlab_experiment or f"exp_{start_iteration}",
            log_dir=os.path.join(path, "swanlog"),
            X_test=X_sample,
            y_test=y_test[:1],
            plot_period=1000,
            period=log_period,
            start_iteration=start_iteration,
            config=model_config
        )
        plotter = None
    else:
        plotter = PlottingCallback(
            X_test,
            y_test,
            period=1000,
            save_dir=f"{path}/plots",
            start_iteration=start_iteration,
        )

    timing_logger = None
    if enable_timing:
        timing_logger = TimingCallback(period=1, save_dir=f"{path}/logs", start_iteration=start_iteration, sync_cuda=True)
        remaining_iterations = 25
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if remaining_iterations > 0:
        callbacks: list[Callback] = [checker]
        if plotter is not None:
            callbacks.append(plotter)
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

        #dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    else:
        print("Training already completed!")


if __name__ == "__main__":
    dataset = "50K"
    task = "5x2_configs"
    grid_npy_path = "/home/wkf/kwave-python/temp/grid_xy.npy"
    path = f"./model_{dataset}_{task}_NIO_test1"
    os.makedirs(path, exist_ok=True)

    main(
        dataset=dataset,
        task=task,
        grid_npy_path=grid_npy_path,
        batch_size=32,
        test=False,
        model_path=None,
        path=path,
        start_iteration=0,
        total_epoch=200,
        enable_timing=True,
        split_ratio = 0.9,
        seed = 114514,
        branch_encoder_cls=EncoderUSCTHelm2,
        enable_swanlab=False,
        swanlab_project="Kwave-NIO",
        swanlab_experiment=None,
    )


