from utils import *

os.environ['DDE_BACKEND'] = 'pytorch'
import deepxde as dde
import json

from deepxde.callbacks import Callback

from h5_dataset import H5DeepONetDataset, H5DatasetConfig
from model_BranchTrunkFlower import BranchTrunkFlower
from my_train import PlottingCallback

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
        return (self.train_x[0][indices], self.train_x[1][indices]), self.train_y[indices]

    def test(self):
        if len(self.test_y) <= self.test_batch_size:
            return self.test_x, self.test_y

        indices = self.test_sampler.get_next(self.test_batch_size)
        X_test_batch = (self.test_x[0][indices], self.test_x[1][indices])
        y_test_batch = self.test_y[indices]
        return X_test_batch, y_test_batch


class LossHistoryCallback(Callback):
    def __init__(self, period=50, save_dir="./training_logs", filename="loss_history", start_iteration=0):
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

from my_train import samples_per_config, x_params, y_params, cache_h5_path, sos_root, kwave_root
def main(
    h5_path,
    path,
    split_ratio=0.9,
    seed=114514,
    batch_size=32,
    total_epoch=200,
    start_iteration=0,
    model_path=None,
):
    set_seed(seed)
    os.makedirs(path, exist_ok=True)
    total_data_num = int(samples_per_config * len(x_params))
    test_batch_size = int(batch_size * ((1 - split_ratio) / split_ratio))

    data = H5DeepONetDataset(
        H5DatasetConfig(
            h5_path=h5_path,
            split_ratio=split_ratio,
            test_batch_size=test_batch_size,
            total_data_num=total_data_num,
        ),
        is_deeponet=True,
        seed=seed,
        enable_timing=False,
    )
    X_test, y_test = data.test()

    trunk_dim = data.trunk_dim
    if trunk_dim != 2:
        raise ValueError(f"Expected trunk_dim=2 for (x,z), got {trunk_dim}")

    net = BranchTrunkFlower(
        num_parameter=2,
        width=64,
        Tx=32,
        Rx=32,
        T_steps=1900,
        H=80,
        W=80,
        lifting_dim=96,
        n_levels=4,
        num_heads=32,
        boundary_condition_types=["ZEROS"],
        dropout_rate=0.0,
    )
    model = dde.Model(data, net)

    model_config = {
        "model_type": "BranchTrunkFlower",
        "model_init_kwargs": {
            "num_parameter": 2,
            "width": 64,
            "Tx": 32,
            "Rx": 32,
            "T_steps": 1900,
            "H": 80,
            "W": 80,
            "lifting_dim": 96,
            "n_levels": 4,
            "num_heads": 32,
            "boundary_condition_types": ["ZEROS"],
            "dropout_rate": 0.0,
        },
        "data": {
            "h5_path": h5_path,
            "split_ratio": split_ratio,
            "total_data_num": total_data_num,
        },
    }
    with open(os.path.join(path, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)

    iterations_per_epoch = (total_data_num + batch_size - 1) // batch_size
    total_iterations = total_epoch * iterations_per_epoch
    remaining_iterations = total_iterations - start_iteration

    model.compile(
        optimizer="adamw",
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
            lambda y_true, y_pred: np.sqrt(np.mean(((y_true - y_pred) ** 2))),
        ],
    )

    checker = dde.callbacks.ModelCheckpoint(
        f"{path}/model",
        save_better_only=True,
        period=1000,
    )
    loss_logger = LossHistoryCallback(period=50, save_dir=f"{path}/logs", start_iteration=start_iteration)
    plotter = PlottingCallback(
        X_test,
        y_test,
        period=1000,
        save_dir=f"{path}/plots",
        start_iteration=start_iteration
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if remaining_iterations > 0:
        model.train(
            iterations=remaining_iterations,
            batch_size=batch_size,
            display_every=50,
            callbacks=[checker, plotter, loss_logger],
            model_save_path=f"{path}/model",
            disregard_previous_best=True,
            model_restore_path=model_path,
        )
    else:
        print("Training already completed!")


if __name__ == "__main__":

    save_path = "./model_branch_trunk_flower"
    main(
        h5_path=cache_h5_path,
        path=save_path,
        split_ratio=0.9,
        seed=114514,
        batch_size=32,
        total_epoch=200,
        start_iteration=0,
        model_path=None,
    )
