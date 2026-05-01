from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import deepxde as dde
import threading
import queue
import time


@dataclass
class H5DatasetConfig:
    h5_path: str
    split_ratio: float = 0.8
    test_batch_size: int = 16
    total_data_num: int = 0
    


class H5DeepONetDataset(dde.data.Data):
    """DeepXDE dataset backed by an HDF5 file with Multiprocessing support.
    Implements Lazy Loading to ensure pickleability.
    """

    def __init__(self, cfg: H5DatasetConfig, is_deeponet: bool = True, 
                 seed: int = 114514, enable_timing: bool = False): 
        super().__init__()
        self.cfg = cfg
        self.is_deeponet = is_deeponet
        self.h5_path = cfg.h5_path
        self.n = cfg.total_data_num
        
        self._verbose = enable_timing  # 控制是否打印时间信息
        self._read_count = 0
        self._print_interval = 1  # 每n个batch打印一次
        
        with h5py.File(self.h5_path, "r") as f:
            self.trunk_dim = int(f["X_trunk"].shape[1])

        self._h5 = None
        self._X_branch = None
        self._X_trunk = None
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
        """后台线程预取数据"""
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
        Slow path: sort+deduplicate for HDF5 read, and return inverse map
        to restore original order (including duplicated indices).
        """
        idx = np.asarray(global_idx, dtype=np.int64).reshape(-1)
        if idx.size <= 1:
            return idx, None

        if np.all(idx[1:] > idx[:-1]):
            return idx, None

        sorted_unique, inverse = np.unique(idx, return_inverse=True)
        return sorted_unique, inverse

    def _ensure_file_open(self):
        """Lazy load the HDF5 file handle."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True, libver='latest',
                                 rdcc_nbytes=256*1024**2,
                                 rdcc_nslots=89
                                )
            self._X_branch = self._h5["X_branch"]
            self._X_trunk = self._h5["X_trunk"]
            self._y = self._h5["y"]

    def close(self):
        """安全关闭：确保线程终止、文件关闭、内存释放"""
        print("[DataLoader] Closing dataset and releasing resources...")
        
        # 1. 先停止预取线程（防止继续生产数据）
        self._shutdown = True
        self._prefetch_error = None
        
        # 2. 清空队列以解除线程在 put() 上的阻塞
        # 线程可能在 queue.put(block=True) 处卡住，需要消费者取走数据或队列被清空
        cleared_items = 0
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
                cleared_items += 1
            except queue.Empty:
                break
        
        if cleared_items > 0:
            print(f"[DataLoader] Cleared {cleared_items} batches from prefetch queue")
        
        # 3. 等待线程结束（给予足够时间完成当前迭代）
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            # 使用超时 join，避免永久阻塞
            self.prefetch_thread.join(timeout=10.0)
            if self.prefetch_thread.is_alive():
                print("[DataLoader] Warning: Prefetch thread did not terminate in time")
                # 注意：threading.Thread 没有 kill 方法，这里只能等待
        
        # 4. 关闭 HDF5 文件（释放文件句柄和内部缓存）
        if self._h5 is not None:
            try:
                self._h5.close()
                print("[DataLoader] HDF5 file closed")
            except Exception as e:
                print(f"[DataLoader] Error closing HDF5: {e}")
            finally:
                self._h5 = None
                self._X_branch = None
                self._X_trunk = None
                self._y = None
        
        # 5. 强制垃圾回收（帮助 Python 归还内存给 OS）
        import gc
        gc.collect()
        
        # 6. 尝试释放大型 numpy 数组（如果有引用残留）
        if hasattr(self, 'train_indices'):
            self.train_indices = None
        if hasattr(self, 'test_indices'):
            self.test_indices = None

    def __del__(self):
        # 确保析构时调用 close，但 Python 不保证 __del__ 一定被调用
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5'] = None
        state['_X_branch'] = None
        state['_X_trunk'] = None
        state['_y'] = None
        state['_prefetch_error'] = None
        return state

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def _get_batch_by_global_indices(self, global_idx: np.ndarray):
        """实际从HDF5读取数据，带时间统计"""
        t_start = time.perf_counter() if self._verbose else None
        
        self._ensure_file_open()

        read_idx, inverse = self._prepare_h5_indices(global_idx)

        Xb = self._X_branch[read_idx]
        Xt = self._X_trunk[read_idx]
        y = self._y[read_idx]

        if inverse is not None:
            Xb = Xb[inverse]
            Xt = Xt[inverse]
            y = y[inverse]
        
        # 仅在类型不匹配时转换，且尽量使用视图而非拷贝
        if Xb.dtype != np.float32:
            Xb = Xb.astype(np.float32, copy=False)
        if Xt.dtype != np.float32:
            Xt = Xt.astype(np.float32, copy=False)
        if y.dtype != np.float32:
            y = y.astype(np.float32, copy=False)
    
        """Xb = np.array(self._X_branch[global_idx], dtype=np.float32)
        Xt = np.array(self._X_trunk[global_idx], dtype=np.float32)
        y = np.array(self._y[global_idx], dtype=np.float32)"""
        
        # 仅在 verbose=True 时计算和打印时间
        if self._verbose:
            h5_read_time = time.perf_counter() - t_start
            
            if self._read_count % self._print_interval == 0:
                print(f"[DataLoader] H5 Read Time (disk I/O): {h5_read_time*1000:.2f} ms | "
                      f"Batch size: {len(global_idx)} | Shape: {Xb.shape}")
        
        if not self.is_deeponet:
            return Xb, y
        return (Xb, Xt), y

    def _get_next_batch_sync(self, batch_size=None):
        """同步获取batch（预取线程使用）"""
        if batch_size is None:
            batch_size = 32

        if len(self.train_indices) == 0:
            return self._get_batch_by_global_indices(self.train_indices)

        local = self.train_sampler.get_next(batch_size)
        global_idx = self.train_indices[local]
        return self._get_batch_by_global_indices(global_idx)

    def train_next_batch(self, batch_size=None):
        """获取训练batch，统计总等待时间（含队列等待）"""
        if batch_size is None:
            batch_size = 32

        t_start = time.perf_counter() if self._verbose else None
        
        # 启动预取线程（如果不存在）
        if self.prefetch_thread is None or not self.prefetch_thread.is_alive():
            self.prefetch_thread = threading.Thread(
                target=self._prefetch_worker, args=(batch_size,), daemon=True
            )
            self.prefetch_thread.start()

        if self._prefetch_error is not None:
            err = self._prefetch_error
            self._prefetch_error = None
            raise RuntimeError("Prefetch worker failed while loading HDF5 batch") from err

        # 获取已预取的 batch
        result = self.prefetch_queue.get(block=True)

        if result is None and self._prefetch_error is not None:
            err = self._prefetch_error
            self._prefetch_error = None
            raise RuntimeError("Prefetch worker failed while loading HDF5 batch") from err
        
        # 仅在 verbose=True 时计算和打印时间
        if self._verbose:
            total_wait_time = time.perf_counter() - t_start
            self._read_count += 1
            
            if self._read_count % self._print_interval == 0:
                print(f"[DataLoader] Total Wait Time (queue+I/O): {total_wait_time*1000:.2f} ms | "
                      f"Batch: {self._read_count}")
        
        return result

    def test(self):
        """测试集读取（不统计时间，避免干扰训练统计）"""
        if len(self.test_indices) == 0:
            return self._get_batch_by_global_indices(self.train_indices[:1])

        if len(self.test_indices) <= self.test_batch_size:
            return self._get_batch_by_global_indices(self.test_indices)

        local = self.test_sampler.get_next(self.test_batch_size)
        global_idx = self.test_indices[local]
        return self._get_batch_by_global_indices(global_idx)