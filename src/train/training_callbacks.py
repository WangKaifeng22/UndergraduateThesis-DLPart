from __future__ import annotations

import math
import os

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import torch
from deepxde.callbacks import Callback
from utils.utils import minmax_denormalize

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as exc:  # pragma: no cover - optional dependency guard
    SummaryWriter = None
    _SUMMARY_WRITER_IMPORT_ERROR = exc
else:
    _SUMMARY_WRITER_IMPORT_ERROR = None

try:
    import swanlab
except Exception as exc:
    swanlab = None
    _SWANLAB_IMPORT_ERROR = exc
else:
    _SWANLAB_IMPORT_ERROR = None

import matplotlib.pyplot as plt

class TensorBoardCallback(Callback):
    def __init__(
        self,
        log_dir: str,
        period: int = 50,
        start_iteration: int = 0,
        log_histograms: bool = False,
        log_metrics: bool = True,
        log_learning_rate: bool = True,
    ):
        super().__init__()
        self.log_dir = log_dir
        self.period = max(1, int(period))
        self.start_iteration = int(start_iteration)
        self.log_histograms = log_histograms
        self.log_metrics = log_metrics
        self.log_learning_rate = log_learning_rate
        self.writer = None
        self._last_logged_iteration = None

    def init(self):
        if SummaryWriter is None:
            raise ImportError(
                "TensorBoardCallback requires the tensorboard package. "
                "Install it with `pip install tensorboard`."
            ) from _SUMMARY_WRITER_IMPORT_ERROR

        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    @staticmethod
    def _as_float_array(values):
        array = np.asarray(values, dtype=np.float64)
        return np.reshape(array, (-1,))

    def _log_series(self, tag_prefix: str, values, step: int):
        if values is None:
            return False

        array = self._as_float_array(values)
        if array.size == 0:
            return False

        self.writer.add_scalar(f"{tag_prefix}/mean", float(np.nanmean(array)), step)
        if array.size == 1:
            self.writer.add_scalar(f"{tag_prefix}/value", float(array[0]), step)
        else:
            for index, value in enumerate(array):
                self.writer.add_scalar(f"{tag_prefix}/component_{index}", float(value), step)
        return True

    def _log_learning_rate(self, step: int):
        optimizer = getattr(self.model, "opt", None)
        param_groups = getattr(optimizer, "param_groups", None)
        if not param_groups:
            return False

        wrote_anything = False
        for group_index, group in enumerate(param_groups):
            learning_rate = group.get("lr")
            if learning_rate is None:
                continue
            self.writer.add_scalar(f"optimizer/lr/group_{group_index}", float(learning_rate), step)
            wrote_anything = True
        return wrote_anything

    def _log_gradients(self, step: int):
        net = getattr(self.model, "net", None)
        named_parameters = getattr(net, "named_parameters", None)
        if named_parameters is None:
            return False

        total_squared_norm = 0.0
        has_gradient = False

        for name, parameter in named_parameters():
            gradient = getattr(parameter, "grad", None)
            if gradient is None:
                continue

            gradient_detached = gradient.detach()
            grad_norm = float(gradient_detached.norm().item())
            self.writer.add_scalar(f"gradients/parameter_norms/{name}", grad_norm, step)
            total_squared_norm += grad_norm * grad_norm
            has_gradient = True

            if self.log_histograms:
                self.writer.add_histogram(f"gradients/histograms/{name}", gradient_detached.cpu(), step)
                self.writer.add_histogram(f"weights/histograms/{name}", parameter.detach().cpu(), step)

        if has_gradient:
            self.writer.add_scalar("gradients/global_norm", math.sqrt(total_squared_norm), step)
        return has_gradient

    def _log_current_state(self, force: bool = False):
        iteration = self.model.train_state.iteration + self.start_iteration
        if iteration <= 0:
            return False
        if not force and iteration % self.period != 0:
            return False
        if self._last_logged_iteration == iteration:
            return False

        wrote_anything = False
        wrote_anything |= self._log_series("loss/train", self.model.train_state.loss_train, iteration)
        wrote_anything |= self._log_series("loss/test", self.model.train_state.loss_test, iteration)

        if self.log_metrics and getattr(self.model.train_state, "metrics_test", None) is not None:
            wrote_anything |= self._log_series("metrics/test", self.model.train_state.metrics_test, iteration)

        if self.log_learning_rate:
            wrote_anything |= self._log_learning_rate(iteration)

        wrote_anything |= self._log_gradients(iteration)

        if wrote_anything:
            self.writer.flush()
            self._last_logged_iteration = iteration
        return wrote_anything

    def on_batch_end(self):
        if self.writer is None:
            self.init()
        self._log_current_state(force=False)

    def on_train_end(self):
        if self.writer is None:
            return
        self._log_current_state(force=True)
        self.writer.flush()
        self.writer.close()
        self.writer = None
class SwanLabCallback(Callback):
    def __init__(
        self,
        project: str,
        experiment_name: str,
        log_dir: str,
        X_test=None,
        y_test=None,
        plot_period: int = 1000,
        period: int = 50,
        start_iteration: int = 0,
        log_histograms: bool = False,
        log_metrics: bool = True,
        log_learning_rate: bool = True,
        config: dict = None,
        vmin=1430,
        vmax=1650,
    ):
        super().__init__()
        self.project = project
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.X_test = X_test
        self.y_test = y_test
        self.plot_period = plot_period
        self.period = max(1, int(period))
        self.start_iteration = int(start_iteration)
        self.log_histograms = log_histograms
        self.log_metrics = log_metrics
        self.log_learning_rate = log_learning_rate
        self.config = config
        self.vmin = vmin
        self.vmax = vmax

        self._initialized = False
        self._last_logged_iteration = None

    def init(self):
        if swanlab is None:
            raise ImportError(
                "SwanLabCallback requires the swanlab package. "
                "Install it with `pip install swanlab`."
            ) from _SWANLAB_IMPORT_ERROR

        os.makedirs(self.log_dir, exist_ok=True)
        swanlab.init(
            project=self.project,
            name=self.experiment_name,
            logdir=self.log_dir,
            config=self.config
        )
        self._initialized = True

    @staticmethod
    def _as_float_array(values):
        array = np.asarray(values, dtype=np.float64)
        return np.reshape(array, (-1,))

    def _log_series(self, tag_prefix: str, values, step: int):
        if values is None:
            return False

        array = self._as_float_array(values)
        if array.size == 0:
            return False

        metrics = {f"{tag_prefix}/mean": float(np.nanmean(array))}
        if array.size == 1:
            metrics[f"{tag_prefix}/value"] = float(array[0])
        else:
            for index, value in enumerate(array):
                metrics[f"{tag_prefix}/component_{index}"] = float(value)
        
        swanlab.log(metrics, step=step)
        return True

    def _log_learning_rate(self, step: int):
        optimizer = getattr(self.model, "opt", None)
        param_groups = getattr(optimizer, "param_groups", None)
        if not param_groups:
            return False

        wrote_anything = False
        metrics = {}
        for group_index, group in enumerate(param_groups):
            learning_rate = group.get("lr")
            if learning_rate is None:
                continue
            metrics[f"optimizer/lr/group_{group_index}"] = float(learning_rate)
            wrote_anything = True
            
        if wrote_anything:
            swanlab.log(metrics, step=step)
        return wrote_anything

    def _log_gradients(self, step: int):
        net = getattr(self.model, "net", None)
        named_parameters = getattr(net, "named_parameters", None)
        if named_parameters is None:
            return False

        total_squared_norm = 0.0
        has_gradient = False
        metrics = {}

        for name, parameter in named_parameters():
            gradient = getattr(parameter, "grad", None)
            if gradient is None:
                continue

            gradient_detached = gradient.detach()
            grad_norm = float(gradient_detached.norm().item())
            metrics[f"gradients/parameter_norms/{name}"] = grad_norm
            total_squared_norm += grad_norm * grad_norm
            has_gradient = True

        if has_gradient:
            metrics["gradients/global_norm"] = math.sqrt(total_squared_norm)
            swanlab.log(metrics, step=step)
            
        return has_gradient

    def _log_current_state(self, force: bool = False):
        iteration = self.model.train_state.iteration + self.start_iteration
        if iteration <= 0:
            return False
        if not force and iteration % self.period != 0:
            return False
        if self._last_logged_iteration == iteration:
            return False

        wrote_anything = False
        wrote_anything |= self._log_series("loss/train", self.model.train_state.loss_train, iteration)
        wrote_anything |= self._log_series("loss/test", self.model.train_state.loss_test, iteration)

        if self.log_metrics and getattr(self.model.train_state, "metrics_test", None) is not None:
            wrote_anything |= self._log_series("metrics/test", self.model.train_state.metrics_test, iteration)

        if self.log_learning_rate:
            wrote_anything |= self._log_learning_rate(iteration)

        wrote_anything |= self._log_gradients(iteration)

        if wrote_anything:
            self._last_logged_iteration = iteration
        return wrote_anything
        
    def _plot_and_log_images(self, iteration: int):
        if swanlab is None or self.X_test is None or self.y_test is None:
            return
            
        try:
            X_sample = self.X_test
            y_true_sample = self.y_test
            y_pred_sample = self.model.predict(X_sample)

            pred = y_pred_sample[0].squeeze()
            target = y_true_sample[0].squeeze()

            pred_real = minmax_denormalize(pred, self.vmin, self.vmax, 2)
            target_real = minmax_denormalize(target, self.vmin, self.vmax, 2)

            error_2d = np.abs(pred_real - target_real)

            plt.switch_backend('Agg')
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            im0 = axes[0].imshow(pred_real, cmap='jet', vmin=self.vmin, vmax=self.vmax, origin='lower')
            axes[0].set_title(f"Pred (Iter {iteration})")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(target_real, cmap='jet', vmin=self.vmin, vmax=self.vmax, origin='lower')
            axes[1].set_title("Ground Truth")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            mae = np.mean(error_2d)
            rmse = np.sqrt(np.mean((target_real - pred_real) ** 2))
            im2 = axes[2].imshow(error_2d, cmap='inferno', origin='lower')
            axes[2].set_title(f"Abs Error (MAE: {mae:.2f}, RMSE: {rmse:.2f})")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            plt.tight_layout()

            swanlab.log({"Evaluation/Prediction_vs_GT_vs_Error": swanlab.Image(fig)}, step=iteration)
            plt.close(fig)
        except Exception as e:
            print(f"SwanLabCallback image logging error: {e}")

    def on_batch_end(self):
        if not self._initialized:
            self.init()
        self._log_current_state(force=False)
        
    def on_epoch_end(self):
        iteration = self.model.train_state.iteration + self.start_iteration
        if iteration % self.plot_period == 0 and iteration > 0:
            self._plot_and_log_images(iteration)

    def on_train_end(self):
        if not self._initialized:
            return
        self._log_current_state(force=True)

