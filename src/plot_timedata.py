import argparse
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from h5_preprocess import _load_x_container, _extract_time_data
from utils import KWAVE_CMAP, prepare_visualization_data


def _compute_stats(arr: np.ndarray) -> Tuple[dict, str]:
    arr = np.asarray(arr)
    fin = np.isfinite(arr)
    nan_cnt = int(np.isnan(arr).sum())
    inf_cnt = int(np.isinf(arr).sum())
    finite_ratio = float(fin.mean()) if arr.size > 0 else 0.0

    if arr.size == 0:
        return {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "finite_ratio": finite_ratio,
            "nan": nan_cnt,
            "inf": inf_cnt,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }, "EMPTY"

    arr_f = arr[fin]
    if arr_f.size == 0:
        return {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "finite_ratio": finite_ratio,
            "nan": nan_cnt,
            "inf": inf_cnt,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }, "NO_FINITE"

    stats = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "finite_ratio": float(finite_ratio),
        "nan": nan_cnt,
        "inf": inf_cnt,
        "min": float(arr_f.min()),
        "max": float(arr_f.max()),
        "mean": float(arr_f.mean()),
        "std": float(arr_f.std()),
    }
    return stats, "OK"


def _load_time_data(path: str) -> np.ndarray:
    x_kind, x_obj = _load_x_container(path)
    return _extract_time_data(x_kind, x_obj)


def _flatten_channels(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data[None, :]
    return data.reshape(-1, data.shape[-1])


def _plot_heatmap(
    data: np.ndarray,
    title: str,
    save_path: str | None,
    show: bool,
    transform_enabled: bool,
    normalize_range: str,
    vis_vmin: float | None,
    vis_vmax: float | None,
) -> None:
    if data.ndim == 3:
        data = data[0,:,:]
    channels = _flatten_channels(data)
    channels = prepare_visualization_data(
        channels,
        enabled=transform_enabled,
        normalize_range=normalize_range,
        vmin=vis_vmin,
        vmax=vis_vmax,
        scale=2,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(channels, aspect="auto", origin="lower", cmap=KWAVE_CMAP)
    ax.set_title(title)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Channel")
    colorbar_label = "Transformed amplitude" if transform_enabled else "Amplitude"
    fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_trace(
    data: np.ndarray,
    source_index: int | None,
    receiver_index: int | None,
    title: str,
    save_path: str | None,
    show: bool,
    transform_enabled: bool,
    normalize_range: str,
    vis_vmin: float | None,
    vis_vmax: float | None,
) -> None:
    if data.ndim == 1:
        trace = data
        label = "trace"
    elif data.ndim == 2:
        idx = receiver_index or 0
        trace = data[idx]
        label = f"channel={idx}"
    else:
        src = source_index or 0
        rec = receiver_index or 0
        trace = data[src, rec]
        label = f"source={src}, receiver={rec}"

    trace = prepare_visualization_data(
        trace,
        enabled=transform_enabled,
        normalize_range=normalize_range,
        vmin=vis_vmin,
        vmax=vis_vmax,
        scale=2,
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trace, lw=1.0)
    ax.set_title(f"{title} ({label})")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check and plot time_data_cat from a data file.")
    parser.add_argument("--x-path", required=True, help="Path to .npz or .mat file containing time_data_cat")
    parser.add_argument("--mode", choices=["heatmap", "trace"], default="heatmap", help="Plot mode")
    parser.add_argument("--source-index", type=int, default=None, help="Source index for trace mode")
    parser.add_argument("--receiver-index", type=int, default=None, help="Receiver index for trace mode")
    parser.add_argument("--no-transform", action="store_true", help="Disable log transform and minmax normalization")
    parser.add_argument(
        "--normalize-range",
        choices=["dynamic", "fixed", "none"],
        default="dynamic",
        help="Normalization range mode used after log transform",
    )
    parser.add_argument("--vis-vmin", type=float, default=None, help="Fixed normalization minimum when --normalize-range fixed")
    parser.add_argument("--vis-vmax", type=float, default=None, help="Fixed normalization maximum when --normalize-range fixed")
    parser.add_argument("--save", default=None, help="Optional output image path")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")

    args = parser.parse_args()

    x_path = os.path.abspath(args.x_path)
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Input not found: {x_path}")

    time_data = _load_time_data(x_path)
    stats, status = _compute_stats(time_data)

    print("[time_data_cat] status=", status)
    print(
        "[time_data_cat] shape={shape}, dtype={dtype}, finite={finite_ratio:.6f}, "
        "nan={nan}, inf={inf}, min={min}, max={max}, mean={mean}, std={std}".format(**stats)
    )

    if status != "OK":
        raise ValueError("time_data_cat has no finite values; cannot plot.")

    title = f"{os.path.basename(x_path)}"
    show = not args.no_show
    transform_enabled = not args.no_transform
    normalize_range = args.normalize_range if transform_enabled else "none"

    if args.mode == "heatmap":
        _plot_heatmap(
            time_data,
            title,
            args.save,
            show,
            transform_enabled,
            normalize_range,
            args.vis_vmin,
            args.vis_vmax,
        )
    else:
        _plot_trace(
            time_data,
            args.source_index,
            args.receiver_index,
            title,
            args.save,
            show,
            transform_enabled,
            normalize_range,
            args.vis_vmin,
            args.vis_vmax,
        )


if __name__ == "__main__":
    main()

