import os
import json
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Any

import numpy as np
import scipy.io

from utils import (
    VMIN,
    VMAX,
    PHYSICAL_LIMIT,
    minmax_normalize,
    complex_global_scale_fit,
    normalize_branch_from_complex,
    inverse_process_sensor_fft,
    log_transform,
)


@dataclass
class H5PreprocessConfig:
    sos_root_dir: str
    result_root_dir: str
    x_param_list: List[str]
    y_param_list: List[str]
    samples_per_config: int
    out_h5_path: str
    out_meta_path: str
    max_tries_multiplier: int = 2
    dtype: str = "float32"
    compression: str | None = "lzf"  # None / "gzip" / "lzf"
    compression_opts: int | None = None
    shuffle: bool = False  # enable HDF5 shuffle filter for better compression
    shuffle_pairs: bool = False  # shuffle order of samples before writing
    shuffle_seed: int | None = None  # optional seed for deterministic shuffle
    chunk_n: int = 1
    # write HDF5 by batches to avoid per-sample chunk updates.
    # If None, defaults to chunk_n.
    write_batch_n: int | None = None
    # multiprocessing to utilize all CPU cores during MAT loading/preprocess.
    # HDF5 writing remains in the main process (h5py is not safe for concurrent writes).
    num_workers: int = 0  # 0/1 => disabled, >1 => use that many processes
    prefetch_factor: int = 4  # how many batches to keep in-flight (per worker pool)
    # NEW: write X_branch in time domain (inverse FFT from freq_data_complex_cat)
    write_branch_time_domain: bool = True
    # Optional normalized SoS POV bounds: (x_min, x_max, y_min, y_max), each in [0, 1].
    # None means no crop (keep full SoS map, same as current behavior).
    sos_pov_bounds_norm: Tuple[float, float, float, float] | None = None
    # Preview-only mode: visualize first SoS + crop box + cropped result, save and exit.
    sos_crop_preview_enabled: bool = False
    sos_crop_preview_dir: str | None = None
    sos_crop_preview_name: str = "sos_pov_preview.png"
    transducer_mask_path: str | None = None  # Optional path to .npy mask for transducer locations (same shape as SoS map)
    # Inference-only mode: scan a single folder that contains sample_XXXXXX.npz files
    # and write a placeholder y dataset for compatibility with existing H5 readers.
    inference_mode: bool = False
    inference_input_dir: str | None = None
    inference_placeholder_y_shape: Tuple[int, int] | None = None


def _iter_file_pairs(
        sos_root_dir: str,
        result_root_dir: str,
        x_param_list: List[str],
        y_param_list: List[str],
        samples_per_config: int,
        max_tries_multiplier: int = 3,
) -> Iterable[Tuple[str, str]]:
    assert len(x_param_list) == len(y_param_list)

    for x_val, y_val in zip(x_param_list, y_param_list):
        folder_name = f"{x_val}andInc{y_val}"
        current_sos_dir = os.path.join(sos_root_dir, folder_name)
        current_result_dir = os.path.join(result_root_dir, folder_name)
        if (not os.path.exists(current_sos_dir)) or (not os.path.exists(current_result_dir)):
            continue

        count_loaded = 0
        idx = 0
        max_tries = samples_per_config * max_tries_multiplier
        while count_loaded < samples_per_config and idx <= max_tries:
            # Prefer K-Wave .npz (time-domain) + SoS .npy if present, else fall back to .mat
            x_npz = os.path.join(current_result_dir, f"sample_{idx:06d}.npz")
            y_npy = os.path.join(current_sos_dir, f"sample_{idx:06d}.npy")
            x_mat = os.path.join(current_result_dir, f"sample_KwaveData_{idx:06d}.mat")
            y_mat = os.path.join(current_sos_dir, f"sample_{idx:06d}.mat")

            idx += 1
            if os.path.exists(x_npz) and os.path.exists(y_npy):
                yield x_npz, y_npy
                count_loaded += 1
                continue
            if os.path.exists(x_mat) and os.path.exists(y_mat):
                yield x_mat, y_mat
                count_loaded += 1
                continue


def _iter_inference_files(
        inference_input_dir: str,
) -> Iterable[str]:
    if not os.path.exists(inference_input_dir):
        raise FileNotFoundError(f"Inference input dir not found: {inference_input_dir}")

    def _sort_key(path: str):
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        if stem.startswith("sample_"):
            suffix = stem[len("sample_"):]
            if suffix.isdigit():
                return (0, int(suffix), name)
        return (1, name)

    for name in sorted(os.listdir(inference_input_dir), key=_sort_key):
        if not name.endswith(".npz"):
            continue
        if not name.startswith("sample_"):
            continue
        yield os.path.join(inference_input_dir, name)


def _load_x_container(x_path: str) -> tuple[str, Any]:
    ext = os.path.splitext(x_path)[1].lower()
    if ext == ".npz":
        return "npz", np.load(x_path)
    return "mat", scipy.io.loadmat(x_path)


def _load_y_array(y_path: str) -> np.ndarray:
    ext = os.path.splitext(y_path)[1].lower()
    if ext == ".npy":
        return np.load(y_path)
    y_mat = scipy.io.loadmat(y_path)
    return y_mat["SoSMap"]


def _validate_sos_pov_bounds(bounds: Tuple[float, float, float, float] | None) -> None:
    if bounds is None:
        return
    if len(bounds) != 4:
        raise ValueError("sos_pov_bounds_norm must be (x_min, x_max, y_min, y_max).")

    x_min, x_max, y_min, y_max = [float(v) for v in bounds]
    if not (0.0 <= x_min <= 1.0 and 0.0 <= x_max <= 1.0 and 0.0 <= y_min <= 1.0 and 0.0 <= y_max <= 1.0):
        raise ValueError(f"sos_pov_bounds_norm must be within [0,1], got {bounds}.")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            f"sos_pov_bounds_norm must satisfy x_max>x_min and y_max>y_min, got {bounds}."
        )


def _compute_pov_indices(
        shape_2d: tuple[int, int],
        bounds: Tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    """Map normalized bounds (x_min, x_max, y_min, y_max) to pixel indices (r0, r1, c0, c1)."""
    h, w = int(shape_2d[0]), int(shape_2d[1])
    x_min, x_max, y_min, y_max = [float(v) for v in bounds]

    # Use floor/ceil so requested normalized region is fully covered.
    c0 = int(np.floor(x_min * w))
    c1 = int(np.ceil(x_max * w))
    r0 = int(np.floor(y_min * h))
    r1 = int(np.ceil(y_max * h))

    c0 = max(0, min(c0, w - 1))
    c1 = max(c0 + 1, min(c1, w))
    r0 = max(0, min(r0, h - 1))
    r1 = max(r0 + 1, min(r1, h))

    return r0, r1, c0, c1


def _apply_sos_pov_crop(
        velocity_map: np.ndarray,
        pov_indices: tuple[int, int, int, int] | None,
) -> np.ndarray:
    if pov_indices is None:
        return velocity_map

    if velocity_map.ndim != 2:
        raise ValueError(
            f"SoS map must be 2D to apply sos_pov_bounds_norm, got shape {velocity_map.shape}."
        )

    r0, r1, c0, c1 = pov_indices
    return velocity_map[r0:r1, c0:c1]


def _save_sos_crop_preview(
        velocity_map: np.ndarray,
        pov_indices: tuple[int, int, int, int] | None,
        out_path: str,
        bounds: Tuple[float, float, float, float] | None,
        transducer_mask_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    if velocity_map.ndim != 2:
        raise ValueError(f"Preview expects 2D SoS map, got shape {velocity_map.shape}.")

    cropped = _apply_sos_pov_crop(velocity_map, pov_indices)
    transducer_mask = None
    if transducer_mask_path:
        transducer_mask = np.load(transducer_mask_path)
        if transducer_mask.shape != velocity_map.shape:
            raise ValueError(
                "Transducer mask shape must match SoS map shape, "
                f"got mask {transducer_mask.shape} vs SoS {velocity_map.shape}."
            )
        transducer_mask = transducer_mask.astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    ax0 = axes[0]
    im0 = ax0.imshow(velocity_map, cmap="viridis", aspect="auto")
    if transducer_mask is not None:
        transducer_overlay = np.ma.masked_where(~transducer_mask, transducer_mask.astype(np.uint8))
        ax0.imshow(
            transducer_overlay,
            cmap=ListedColormap(["white"]),
            interpolation="nearest",
            aspect="auto",
        )
    ax0.set_title("Original SoS")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    if pov_indices is not None:
        r0, r1, c0, c1 = pov_indices
        rect = Rectangle((c0, r0), c1 - c0, r1 - r0, linewidth=1.5, edgecolor="r", facecolor="none")
        ax0.add_patch(rect)
        if bounds is not None:
            ax0.set_xlabel(f"bounds={tuple(v for v in pov_indices)}")

    ax1 = axes[1]
    im1 = ax1.imshow(cropped, cmap="viridis", aspect="auto")
    ax1.set_title("Cropped SoS")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _extract_time_data(x_kind: str, x_obj: Any) -> np.ndarray:
    if x_kind == "npz":
        return x_obj["time_data_cat"]
    if "time_data_cat" in x_obj:
        return x_obj["time_data_cat"]
    raise KeyError("time_data_cat not found in input")


def _extract_freq_data(x_kind: str, x_obj: Any) -> np.ndarray:
    if x_kind == "npz":
        raise KeyError("freq_data_complex_cat not available in .npz input")
    return x_obj["freq_data_complex_cat"]


def _extract_sensor_coords(x_kind: str, x_obj: Any) -> np.ndarray:
    if x_kind == "npz":
        return x_obj["sensor_coords"]
    return x_obj["sensor_coords"]


def _probe_shapes(
        one_x_path: str,
    one_y_path: str | None,
        *,
        write_branch_time_domain: bool,
        sos_pov_indices: tuple[int, int, int, int] | None,
    inference_placeholder_y_shape: tuple[int, int] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Infer output shapes for HDF5 datasets.

    - If write_branch_time_domain=True:
        * if time_domain_apply_ifft=True: X_branch shape is (..., out_time)
          where out_time = crop_time if provided else n_time.
        * else: use existing last-dim length (ignore crop_time).
    - Else: keep legacy behavior: concat(real, imag) along axis=0.
    """
    x_kind, x_obj = _load_x_container(one_x_path)

    if write_branch_time_domain:
        time_data = _extract_time_data(x_kind, x_obj)
        out_time = int(time_data.shape[-1])
        if out_time <= 0:
            raise ValueError(f"Invalid out_time computed from time_data: {out_time}.")
        leading = time_data.shape[:-1]
        branch_shape = (*leading, out_time)
    else:
        sensor_data_complex = _extract_freq_data(x_kind, x_obj)
        sensor_data_amp = np.concatenate([np.real(sensor_data_complex), np.imag(sensor_data_complex)], axis=0)
        branch_shape = sensor_data_amp.shape

    coords = _extract_sensor_coords(x_kind, x_obj).astype(np.float32).flatten("F")
    trunk_shape = coords.shape

    if one_y_path is not None:
        velocity_map = _load_y_array(one_y_path).astype(np.float32)
        velocity_map = _apply_sos_pov_crop(velocity_map, sos_pov_indices)
        y_shape = velocity_map.shape
    else:
        if inference_placeholder_y_shape is None:
            raise ValueError("inference_placeholder_y_shape must be provided when no ground truth is available")
        y_shape = tuple(int(v) for v in inference_placeholder_y_shape)

    return branch_shape, trunk_shape, y_shape


def _process_one_pair(
        x_path: str,
    y_path: str | None,
        dtype_str: str,
        branch_scale: float,
        *,
        write_branch_time_domain: bool,
        branch_vmin: float | None,
        branch_vmax: float | None,
        branch_log_vmin: float | None,
        branch_log_vmax: float | None,
        sos_pov_indices: tuple[int, int, int, int] | None,
        inference_placeholder_y_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Worker-side: load two files and return normalized (X_branch, X_trunk, y)."""
    dtype = np.dtype(dtype_str)

    x_kind, x_obj = _load_x_container(x_path)
    if y_path is not None:
        velocity_map = _load_y_array(y_path).astype(dtype)
        velocity_map = _apply_sos_pov_crop(velocity_map, sos_pov_indices)
    else:
        if inference_placeholder_y_shape is None:
            raise ValueError("inference_placeholder_y_shape must be provided when y_path is None")
        velocity_map = np.zeros(tuple(int(v) for v in inference_placeholder_y_shape), dtype=dtype)

    if write_branch_time_domain:
        if branch_vmin is None or branch_vmax is None:
            raise ValueError("branch_vmin/branch_vmax must be provided when write_branch_time_domain=True")
        if branch_log_vmin is None or branch_log_vmax is None:
            raise ValueError("branch_log_vmin/branch_log_vmax must be provided when write_branch_time_domain=True")

        time_branch = _extract_time_data(x_kind, x_obj)

        # log transform first, then min-max normalize in log-domain
        time_branch_log = log_transform(time_branch.astype(np.float32))
        amp = minmax_normalize(time_branch_log, float(branch_log_vmin), float(branch_log_vmax), scale=2).astype(dtype)
    else:
        sensor_data_complex = _extract_freq_data(x_kind, x_obj)
        amp = normalize_branch_from_complex(sensor_data_complex, scale=branch_scale, axis_concat=0).astype(dtype)

    coords = _extract_sensor_coords(x_kind, x_obj).astype(dtype).flatten("F")
    coords = minmax_normalize(coords, -PHYSICAL_LIMIT, PHYSICAL_LIMIT, scale=2).astype(dtype)

    velocity_map = minmax_normalize(velocity_map, VMIN, VMAX, scale=2).astype(dtype)

    return amp, coords, velocity_map


def preprocess_to_h5(cfg: H5PreprocessConfig) -> None:
    """Stream MAT->HDF5 without keeping the whole dataset in RAM.

    Normalization is applied BEFORE writing:
      - X_branch:
          * if cfg.write_branch_time_domain:
              inverse FFT (optional) -> log_transform -> minmax_normalize (log-domain) to [-1,1]
          * else: global scaling based on complex magnitude |z| (real/imag share one scale)
      - X_trunk:  minmax_normalize to [-1,1] (scale=2) using [-PHYSICAL_LIMIT, +PHYSICAL_LIMIT]
      - y:        minmax_normalize to [-1,1] (scale=2) using [VMIN, VMAX]

    This keeps phase relationships between real/imag components (freq mode),
    and ensures consistent scaling across the dataset (time mode).
    """
    import h5py  # local import to keep requirements optional
    import os

    # Use all logical CPUs by default if num_workers is negative
    if cfg.num_workers < 0:
        cfg.num_workers = os.cpu_count() or 1

    if not cfg.out_h5_path:
        raise ValueError("cfg.out_h5_path must be a non-empty path")
    if not cfg.out_meta_path:
        raise ValueError("cfg.out_meta_path must be a non-empty path")

    out_dir = os.path.dirname(os.path.abspath(cfg.out_h5_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    _validate_sos_pov_bounds(cfg.sos_pov_bounds_norm)

    if cfg.inference_mode:
        if not cfg.inference_input_dir:
            raise ValueError("cfg.inference_input_dir must be provided when inference_mode=True")
        x_paths = list(_iter_inference_files(cfg.inference_input_dir))
        if len(x_paths) == 0:
            raise ValueError(f"No sample_*.npz files found in inference_input_dir: {cfg.inference_input_dir}")
        if cfg.samples_per_config > 0:
            x_paths = x_paths[: cfg.samples_per_config]
        pairs = [(x_path, None) for x_path in x_paths]
    else:
        pairs = list(
            _iter_file_pairs(
                cfg.sos_root_dir,
                cfg.result_root_dir,
                cfg.x_param_list,
                cfg.y_param_list,
                cfg.samples_per_config,
                cfg.max_tries_multiplier,
            )
        )
        if len(pairs) == 0:
            raise ValueError("No valid (x,y) .mat pairs found. Check paths and folder naming.")

    # Compute crop indices once.
    # In paired mode, infer from the first GT SoS map.
    # In inference mode, infer from placeholder y shape.
    if cfg.inference_mode:
        if cfg.inference_placeholder_y_shape is None:
            raise ValueError("cfg.inference_placeholder_y_shape must be provided when inference_mode=True")
        ph_h, ph_w = (int(cfg.inference_placeholder_y_shape[0]), int(cfg.inference_placeholder_y_shape[1]))
        if ph_h <= 0 or ph_w <= 0:
            raise ValueError(f"Invalid inference_placeholder_y_shape: {cfg.inference_placeholder_y_shape}")
        first_velocity_map = np.zeros((ph_h, ph_w), dtype=np.float32)
        first_velocity_src = "inference_placeholder_y_shape"
    else:
        first_velocity_map = _load_y_array(pairs[0][1]).astype(np.float32)
        first_velocity_src = str(pairs[0][1])

    if first_velocity_map.ndim != 2:
        raise ValueError(f"SoS map must be 2D, got shape {first_velocity_map.shape} from {first_velocity_src}")
    sos_pov_indices = None
    if cfg.sos_pov_bounds_norm is not None:
        sos_pov_indices = _compute_pov_indices(
            (int(first_velocity_map.shape[0]), int(first_velocity_map.shape[1])),
            cfg.sos_pov_bounds_norm,
        )

    if cfg.sos_crop_preview_enabled:
        if not cfg.sos_crop_preview_dir:
            raise ValueError("cfg.sos_crop_preview_dir must be provided when sos_crop_preview_enabled=True")
        if cfg.transducer_mask_path is not None:
            if not str(cfg.transducer_mask_path).lower().endswith(".npy"):
                raise ValueError("cfg.transducer_mask_path must point to a .npy file")
            if not os.path.exists(cfg.transducer_mask_path):
                raise ValueError(f"transducer mask file not found: {cfg.transducer_mask_path}")
        preview_path = os.path.join(cfg.sos_crop_preview_dir, cfg.sos_crop_preview_name)
        _save_sos_crop_preview(
            velocity_map=first_velocity_map,
            pov_indices=sos_pov_indices,
            out_path=preview_path,
            bounds=cfg.sos_pov_bounds_norm,
            transducer_mask_path=cfg.transducer_mask_path,
        )
        print(f"[H5] SoS POV preview saved to: {preview_path}")
        print("[H5] Preview-only mode enabled. Exit without writing HDF5.")
        return

    if cfg.shuffle_pairs:
        rng = np.random.default_rng(cfg.shuffle_seed)
        rng.shuffle(pairs)

    if cfg.write_branch_time_domain:
        pass  # No more IFFT validation needed

    branch_shape, trunk_shape, y_shape = _probe_shapes(
        pairs[0][0],
        pairs[0][1],
        write_branch_time_domain=bool(cfg.write_branch_time_domain),
        sos_pov_indices=sos_pov_indices,
        inference_placeholder_y_shape=cfg.inference_placeholder_y_shape,
    )

    n = len(pairs)
    dtype = np.dtype(cfg.dtype)

    # Pass 1: compute global normalization for branch
    branch_scale = 1.0
    branch_vmin = None
    branch_vmax = None
    branch_log_vmin = None
    branch_log_vmax = None

    if cfg.write_branch_time_domain:
        vmin = float("inf")
        vmax = float("-inf")
        lmin = float("inf")
        lmax = float("-inf")
        valid_count = 0

        for x_path, _ in pairs:
            x_kind, x_obj = _load_x_container(x_path)

            time_branch = _extract_time_data(x_kind, x_obj)

            mn = float(np.nanmin(time_branch))
            mx = float(np.nanmax(time_branch))
            if np.isfinite(mn) and np.isfinite(mx):
                valid_count += 1
                if mn < vmin:
                    vmin = mn
                if mx > vmax:
                    vmax = mx

                time_branch_log = log_transform(time_branch.astype(np.float32))
                lmn = float(np.nanmin(time_branch_log))
                lmx = float(np.nanmax(time_branch_log))
                if np.isfinite(lmn) and np.isfinite(lmx):
                    if lmn < lmin:
                        lmin = lmn
                    if lmx > lmax:
                        lmax = lmx

        if valid_count == 0:
            raise ValueError("All time-domain samples are NaN/Inf. Check time_data_cat or IFFT output.")

        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)):
            raise ValueError(f"Non-finite time-domain min/max: vmin={vmin}, vmax={vmax}")
        if (not np.isfinite(lmin)) or (not np.isfinite(lmax)):
            raise ValueError(f"Non-finite log-domain min/max: lmin={lmin}, lmax={lmax}")

        if vmin == vmax:
            eps = 1e-12 if vmin == 0 else abs(vmin) * 1e-12
            vmin -= eps
            vmax += eps
        if lmin == lmax:
            eps = 1e-12 if lmin == 0 else abs(lmin) * 1e-12
            lmin -= eps
            lmax += eps

        branch_vmin = vmin
        branch_vmax = vmax
        branch_log_vmin = lmin
        branch_log_vmax = lmax

    else:
        for x_path, _ in pairs:
            x_kind, x_obj = _load_x_container(x_path)
            sensor_data = _extract_freq_data(x_kind, x_obj)
            s = complex_global_scale_fit(sensor_data, method="max")
            if s > branch_scale:
                branch_scale = s

        if (not np.isfinite(branch_scale)) or branch_scale <= 0:
            branch_scale = 1.0

    # Write HDF5
    chunk_n = max(1, int(cfg.chunk_n))
    x_chunks = (min(chunk_n, n), *branch_shape)
    t_chunks = (min(chunk_n, n), *trunk_shape)
    y_chunks = (min(chunk_n, n), *y_shape)

    # Batch write size: default to chunk_n (best match for chunked+compressed datasets)
    write_batch_n = cfg.write_batch_n
    if write_batch_n is None:
        write_batch_n = chunk_n
    write_batch_n = max(1, int(write_batch_n))

    # Decide multiprocessing usage
    use_mp = int(cfg.num_workers) > 1

    with h5py.File(cfg.out_h5_path, "w") as f:
        dset_x = f.create_dataset(
            "X_branch",
            shape=(n, *branch_shape),
            dtype=dtype,
            chunks=x_chunks,
            compression=cfg.compression,
            compression_opts=cfg.compression_opts,
            shuffle=bool(cfg.shuffle),
        )
        dset_t = f.create_dataset(
            "X_trunk",
            shape=(n, *trunk_shape),
            dtype=dtype,
            chunks=t_chunks,
            compression=cfg.compression,
            compression_opts=cfg.compression_opts,
            shuffle=bool(cfg.shuffle),
        )
        dset_y = f.create_dataset(
            "y",
            shape=(n, *y_shape),
            dtype=dtype,
            chunks=y_chunks,
            compression=cfg.compression,
            compression_opts=cfg.compression_opts,
            shuffle=bool(cfg.shuffle),
        )

        # NOTE: pre-allocate batch buffers to avoid repeated allocations.
        # These may be large; keep write_batch_n aligned with your RAM.
        batch_x = np.empty((write_batch_n, *branch_shape), dtype=dtype)
        batch_t = np.empty((write_batch_n, *trunk_shape), dtype=dtype)
        batch_y = np.empty((write_batch_n, *y_shape), dtype=dtype)

        batch_fill = 0
        batch_start = 0

        def _flush_batch(fill: int) -> None:
            nonlocal batch_start
            if fill <= 0:
                return
            i0 = batch_start
            i1 = batch_start + fill
            dset_x[i0:i1] = batch_x[:fill]
            dset_t[i0:i1] = batch_t[:fill]
            dset_y[i0:i1] = batch_y[:fill]
            batch_start = i1

        if not use_mp:
            for i, (x_path, y_path) in enumerate(pairs):
                x_kind, x_obj = _load_x_container(x_path)
                if y_path is not None:
                    velocity_map = _load_y_array(y_path).astype(dtype)
                    velocity_map = _apply_sos_pov_crop(velocity_map, sos_pov_indices)
                else:
                    if cfg.inference_placeholder_y_shape is None:
                        raise RuntimeError("inference_placeholder_y_shape must be provided when inference_mode=True")
                    velocity_map = np.zeros(tuple(int(v) for v in cfg.inference_placeholder_y_shape), dtype=dtype)

                if cfg.write_branch_time_domain:
                    if branch_vmin is None or branch_vmax is None or branch_log_vmin is None or branch_log_vmax is None:
                        raise RuntimeError("Internal error: branch min/max should be computed in pass-1")

                    time_branch = _extract_time_data(x_kind, x_obj)

                    time_branch_log = log_transform(time_branch.astype(np.float32))
                    amp = minmax_normalize(
                        time_branch_log,
                        float(branch_log_vmin),
                        float(branch_log_vmax),
                        scale=2,
                    ).astype(dtype)
                else:
                    sensor_data_complex = _extract_freq_data(x_kind, x_obj)
                    amp = normalize_branch_from_complex(sensor_data_complex, scale=branch_scale, axis_concat=0).astype(
                        dtype)

                coords = _extract_sensor_coords(x_kind, x_obj).astype(dtype).flatten("F")
                coords = minmax_normalize(coords, -PHYSICAL_LIMIT, PHYSICAL_LIMIT, scale=2).astype(dtype)

                velocity_map = minmax_normalize(velocity_map, VMIN, VMAX, scale=2).astype(dtype)

                batch_x[batch_fill] = amp
                batch_t[batch_fill] = coords
                batch_y[batch_fill] = velocity_map
                batch_fill += 1

                if batch_fill >= write_batch_n:
                    _flush_batch(batch_fill)
                    batch_fill = 0

                if (i + 1) % 50 == 0:
                    print(f"Prepared {i + 1}/{n} samples...")

            _flush_batch(batch_fill)

        else:
            from concurrent.futures import ProcessPoolExecutor

            print(f"[H5] Multiprocessing enabled: num_workers={cfg.num_workers}")

            max_in_flight = max(1, int(cfg.prefetch_factor) * int(cfg.num_workers))

            def submit(exe: ProcessPoolExecutor, idx: int):
                x_path, y_path = pairs[idx]
                return exe.submit(
                    _process_one_pair,
                    x_path,
                    y_path,
                    cfg.dtype,
                    branch_scale,
                    write_branch_time_domain=bool(cfg.write_branch_time_domain),
                    branch_vmin=branch_vmin,
                    branch_vmax=branch_vmax,
                    branch_log_vmin=branch_log_vmin,
                    branch_log_vmax=branch_log_vmax,
                    sos_pov_indices=sos_pov_indices,
                    inference_placeholder_y_shape=cfg.inference_placeholder_y_shape,
                )

            with ProcessPoolExecutor(max_workers=int(cfg.num_workers)) as exe:
                next_idx = 0
                in_flight: list[tuple[int, Any]] = []

                # prime
                while next_idx < n and len(in_flight) < max_in_flight:
                    in_flight.append((next_idx, submit(exe, next_idx)))
                    next_idx += 1

                # consume in order
                for want_idx in range(n):
                    # find matching future
                    pos = None
                    for j, (idx, fut) in enumerate(in_flight):
                        if idx == want_idx:
                            pos = j
                            break
                    if pos is None:
                        # should not happen
                        raise RuntimeError("Internal error: missing in-flight future")

                    idx, fut = in_flight.pop(pos)
                    amp, coords, velocity_map = fut.result()

                    batch_x[batch_fill] = amp
                    batch_t[batch_fill] = coords
                    batch_y[batch_fill] = velocity_map
                    batch_fill += 1

                    if batch_fill >= write_batch_n:
                        _flush_batch(batch_fill)
                        batch_fill = 0

                    if (want_idx + 1) % 50 == 0:
                        print(f"Prepared {want_idx + 1}/{n} samples...")

                    # refill
                    while next_idx < n and len(in_flight) < max_in_flight:
                        in_flight.append((next_idx, submit(exe, next_idx)))
                        next_idx += 1

                _flush_batch(batch_fill)

    meta = {
        "num_samples": n,
        "branch_shape": [n, *branch_shape],
        "trunk_shape": [n, *trunk_shape],
        "y_shape": [n, *y_shape],
        "has_ground_truth": not bool(cfg.inference_mode),
        "input_mode": "inference_only_npz" if cfg.inference_mode else "paired_sos_kwave",
        "inference_input_dir": cfg.inference_input_dir,
        "inference_placeholder_y_shape": list(cfg.inference_placeholder_y_shape) if cfg.inference_placeholder_y_shape is not None else None,
        "dtype": cfg.dtype,
        "branch_domain": "time" if cfg.write_branch_time_domain else "freq_complex_concat",
        "branch_norm_method": "log_transform+minmax" if cfg.write_branch_time_domain else "max_abs_complex",
        "branch_min": float(branch_vmin) if branch_vmin is not None else None,
        "branch_max": float(branch_vmax) if branch_vmax is not None else None,
        "branch_log_min": float(branch_log_vmin) if branch_log_vmin is not None else None,
        "branch_log_max": float(branch_log_vmax) if branch_log_vmax is not None else None,
        "branch_scale": float(branch_scale),
        "branch_scale_method": "max_abs_complex",
        "shuffle": bool(cfg.shuffle),
        "shuffle_pairs": bool(cfg.shuffle_pairs),
        "shuffle_seed": int(cfg.shuffle_seed) if cfg.shuffle_seed is not None else None,
        "trunk_min": -PHYSICAL_LIMIT,
        "trunk_max": PHYSICAL_LIMIT,
        "y_min": VMIN,
        "y_max": VMAX,
        "normalize_scale": 2,
        "x_param_list": cfg.x_param_list,
        "y_param_list": cfg.y_param_list,
        "samples_per_config": cfg.samples_per_config,
        "h5_chunk_n": int(chunk_n),
        "h5_write_batch_n": int(write_batch_n),
        "compression": cfg.compression,
        "compression_opts": cfg.compression_opts,
        "sos_pov_bounds_norm": list(cfg.sos_pov_bounds_norm) if cfg.sos_pov_bounds_norm is not None else None,
        "sos_pov_indices": list(sos_pov_indices) if sos_pov_indices is not None else None,
        "sos_crop_preview_enabled": bool(cfg.sos_crop_preview_enabled),
        "sos_crop_preview_dir": cfg.sos_crop_preview_dir,
        "sos_crop_preview_name": cfg.sos_crop_preview_name,
    }
    with open(cfg.out_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Done. H5 written to: {cfg.out_h5_path}")
    print(f"Meta written to: {cfg.out_meta_path}")


from my_train import samples_per_config, x_params, y_params, sos_root, kwave_root, cache_h5_path, meta_h5_path

if __name__ == "__main__":
    # Example usage (adjust paths)
    try:
        cfg = H5PreprocessConfig(
            sos_root_dir=sos_root,
            result_root_dir=kwave_root,
            x_param_list=x_params,
            y_param_list=y_params,
            samples_per_config=samples_per_config,
            out_h5_path="/home/wkf/kwave-python/real-worldData/real_world_data.h5",
            out_meta_path="/home/wkf/kwave-python/real-worldData/real_world_data_meta.json",
            num_workers=-1,
            write_branch_time_domain=True,
            shuffle=False,
            shuffle_pairs=False, #打乱顺序
            shuffle_seed=114514,
            sos_pov_bounds_norm=(0.140625, 0.453125, 0.33203125, 0.64453125),  # y范围, x范围（归一化）
            sos_crop_preview_enabled=False,
            sos_crop_preview_dir="./debug_sos_preview",
            sos_crop_preview_name="first_sos_crop_0.140625-0.453125.png",
            transducer_mask_path="/home/wkf/kwave-python/temp/mask.npy",
            inference_mode=True,
            inference_input_dir="/home/wkf/kwave-python/real-worldData/npz_data",
            inference_placeholder_y_shape=(80, 80),
        )
        preprocess_to_h5(cfg)
    except Exception as e:
        print(
            "[h5_preprocess] Example usage failed (this does not affect library usage). "
            f"Reason: {e}"
        )
