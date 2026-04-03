from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import deepxde as dde
import threading
import queue


@dataclass
class H5NIOConfig:
    h5_path: str
    grid_npy_path: str
    split_ratio: float = 0.8
    test_batch_size: int = 16
    total_data_num: int = 0
    squeeze_y_channel: bool = True


class H5NIODataset(dde.data.Data):
    """HDF5-backed dataset for NIO training with a static trunk grid.

    Branch input format is the same as Fourier DeepONet (X_branch).
    Trunk input is a fixed grid loaded once from a .npy file: [Nx, Ny, 2].
    """

    def __init__(self, cfg: H5NIOConfig, seed: int = 114514):
        super().__init__()
        self.cfg = cfg
        self.h5_path = cfg.h5_path
        self.n = int(cfg.total_data_num)
        self.squeeze_y_channel = bool(cfg.squeeze_y_channel)

        # Load static reconstruction grid once and keep it in memory.
        grid = np.load(cfg.grid_npy_path)
        grid = np.asarray(grid, dtype=np.float32)
        if grid.ndim != 3 or grid.shape[-1] != 2:
            raise ValueError(f"Expected grid shape [Nx, Ny, 2], got {grid.shape}.")
        self.grid = grid

        with h5py.File(self.h5_path, "r") as f:
            if self.n <= 0:
                self.n = int(f["X_branch"].shape[0])

        self._h5 = None
        self._X_branch = None
        self._y = None

        self.split_idx = int(self.n * cfg.split_ratio)
        indices = range(0, self.n)
        self.train_indices = np.array(indices[:self.split_idx])
        self.test_indices = np.array(indices[self.split_idx:])

        self.train_sampler = dde.data.BatchSampler(len(self.train_indices), shuffle=False)
        self.test_sampler = dde.data.BatchSampler(len(self.test_indices), shuffle=False)
        self.test_batch_size = int(cfg.test_batch_size)

        self.prefetch_queue = queue.Queue(maxsize=2)
        self.prefetch_thread = None
        self._shutdown = False
        self._prefetch_error = None

    def _prefetch_worker(self, batch_size):
        while not self._shutdown:
            try:
                batch = self._get_next_batch_sync(batch_size)
                self.prefetch_queue.put(batch, block=True)
            except Exception as e:
                if not self._shutdown:
                    self._prefetch_error = e
                    self._shutdown = True
                    try:
                        # Sentinel to wake blocked consumer.
                        self.prefetch_queue.put_nowait(None)
                    except queue.Full:
                        pass
                break

    @staticmethod
    def _prepare_h5_indices(global_idx: np.ndarray):
        """Prepare HDF5-safe indices and optional inverse mapping.

        Fast path: already strictly increasing -> return as-is.
        Slow path: sort+deduplicate for HDF5 read, then restore original order
        with inverse map (supports duplicated indices).
        """
        idx = np.asarray(global_idx, dtype=np.int64).reshape(-1)
        if idx.size <= 1:
            return idx, None

        if np.all(idx[1:] > idx[:-1]):
            return idx, None

        sorted_unique, inverse = np.unique(idx, return_inverse=True)
        return sorted_unique, inverse

    def _ensure_file_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(
                self.h5_path,
                "r",
                swmr=True,
                libver="latest",
                rdcc_nbytes=1024 * 1024**2,
                rdcc_nslots=100000,
            )
            self._X_branch = self._h5["X_branch"]
            self._y = self._h5["y"]

    def close(self):
        self._shutdown = True
        self._prefetch_error = None
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            finally:
                self._h5 = None
                self._X_branch = None
                self._y = None

        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except queue.Empty:
                break

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_X_branch"] = None
        state["_y"] = None
        state["_prefetch_error"] = None
        return state

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def _get_batch_by_global_indices(self, global_idx: np.ndarray):
        self._ensure_file_open()

        read_idx, inverse = self._prepare_h5_indices(global_idx)

        Xb = np.array(self._X_branch[read_idx], dtype=np.float32)
        y = np.array(self._y[read_idx], dtype=np.float32)

        if inverse is not None:
            Xb = Xb[inverse]
            y = y[inverse]

        if self.squeeze_y_channel and y.ndim == 4 and y.shape[-1] == 1:
            y = np.squeeze(y, axis=-1)

        return (Xb, self.grid), y

    def _get_next_batch_sync(self, batch_size=None):
        if batch_size is None:
            batch_size = 32

        if len(self.train_indices) == 0:
            return self._get_batch_by_global_indices(self.train_indices)

        local = self.train_sampler.get_next(batch_size)
        global_idx = self.train_indices[local]
        return self._get_batch_by_global_indices(global_idx)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = 32

        if self.prefetch_thread is None or not self.prefetch_thread.is_alive():
            self.prefetch_thread = threading.Thread(
                target=self._prefetch_worker, args=(batch_size,), daemon=True
            )
            self.prefetch_thread.start()

        if self._prefetch_error is not None:
            err = self._prefetch_error
            self._prefetch_error = None
            raise RuntimeError("Prefetch worker failed while loading HDF5 batch") from err

        result = self.prefetch_queue.get(block=True)

        if result is None and self._prefetch_error is not None:
            err = self._prefetch_error
            self._prefetch_error = None
            raise RuntimeError("Prefetch worker failed while loading HDF5 batch") from err

        return result

    def test(self):
        if len(self.test_indices) == 0:
            return self._get_batch_by_global_indices(self.train_indices[:1])

        if len(self.test_indices) <= self.test_batch_size:
            return self._get_batch_by_global_indices(self.test_indices)

        local = self.test_sampler.get_next(self.test_batch_size)
        global_idx = self.test_indices[local]
        return self._get_batch_by_global_indices(global_idx)

