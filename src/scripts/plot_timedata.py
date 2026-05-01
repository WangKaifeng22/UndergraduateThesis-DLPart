import argparse
import os
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from utils.h5_preprocess import _load_x_container, _extract_time_data
from utils.utils import KWAVE_CMAP, prepare_visualization_data


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Time‑data loader (original)
# ---------------------------------------------------------------------------
def _load_time_data(path: str) -> np.ndarray:
    x_kind, x_obj = _load_x_container(path)
    return _extract_time_data(x_kind, x_obj)


# ---------------------------------------------------------------------------
# trans_info printing and element_locations extraction
# ---------------------------------------------------------------------------
def _print_and_extract_trans_info(x_obj: dict) -> Optional[np.ndarray]:
    if "trans_info" not in x_obj:
        return None

    ti = x_obj["trans_info"]
    print("\n--- trans_info members ---")

    if not isinstance(ti, np.ndarray) or ti.dtype.names is None:
        print("  (trans_info is not a recognizable MATLAB struct)")
        return None

    coords = None
    for name in ti.dtype.names:
        val = ti[name]

        if name == "element_locations":
            # Extract coordinates, removing redundant struct array packaging if necessary
            coords = val[0, 0] if isinstance(val, np.ndarray) and val.shape == (1, 1) else val

        if isinstance(val, np.ndarray):
            if val.size == 1:
                print(f"  .{name}: {val.item()}")
            elif val.ndim > 0 and (val.dtype.kind in ('U', 'S') or val.dtype.type is np.str_):
                try:
                    s = "".join([str(x) for x in val.flat])
                    print(f"  .{name}: {s}")
                except Exception:
                    print(f"  .{name}: {val[0] if val.size > 0 else ''}")
            else:
                print(f"  .{name}: <array shape={val.shape}, dtype={val.dtype}>")
        else:
            print(f"  .{name}: {val}")
    
    print("--------------------------\n")
    return coords


# ---------------------------------------------------------------------------
# Coordinate normalisation (adapted from plot_sensor_coords_npz.py)
# ---------------------------------------------------------------------------
def _normalize_xy(sensor_coords: np.ndarray) -> np.ndarray:
    """
    Convert sensor coordinates array to (N, 2) for scatter plotting.
    - 1D or (1,N)/(N,1) arrays are treated as x‑coordinates, y=0.
    - 2D arrays with shape (2,N), (N,2), (3,N), (N,3) keep first two dims.
    - No normalisation is applied (raw coordinates are used as‑is).
    """
    coords = np.asarray(sensor_coords)

    # -- 1D input (e.g. (128,)) -------------------------------------------------
    if coords.ndim == 1:
        xy = np.column_stack((coords, np.zeros_like(coords)))
        return xy.astype(float)

    # -- 2D input with a trivial dimension (e.g. (1,128) or (128,1)) ------------
    if coords.ndim == 2:
        if coords.shape[0] == 1 or coords.shape[1] == 1:
            coords = coords.flatten()
            xy = np.column_stack((coords, np.zeros_like(coords)))
            return xy.astype(float)

    # -- Standard 2D/3D multi‑column formats ------------------------------------
    if coords.shape[0] in (2, 3) and coords.shape[1] >= 2:
        xy = coords[:2, :].T                     # (2, N) or (3, N) -> (N, 2)
    elif coords.shape[1] in (2, 3) and coords.shape[0] >= 2:
        xy = coords[:, :2]                       # (N, 2) or (N, 3) -> (N, 2)
    else:
        raise ValueError(
            "Unsupported sensor_coords shape. Expected (2, N), (N, 2), (3, N), "
            f"(N, 3), (1, N), (N, 1), or 1D. Got: {coords.shape}"
        )

    return xy.astype(float)


# ---------------------------------------------------------------------------
# Drawing helpers – reusable for individual and combined figures
# ---------------------------------------------------------------------------
def _flatten_channels(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data[None, :]
    return data.reshape(-1, data.shape[-1])


def _draw_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    transform_enabled: bool,
    normalize_range: str,
    vis_vmin: Optional[float],
    vis_vmax: Optional[float],
    source_index: int = 0,
) -> None:
    # Select source if 3D
    if data.ndim == 3:
        data = data[source_index, :, :]
    channels = _flatten_channels(data)
    channels = prepare_visualization_data(
        channels,
        enabled=transform_enabled,
        normalize_range=normalize_range,
        vmin=vis_vmin,
        vmax=vis_vmax,
        scale=2,
    )
    im = ax.imshow(channels, aspect="auto", origin="lower", cmap=KWAVE_CMAP)
    ax.set_title(title)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Channel")
    colorbar_label = "Transformed amplitude" if transform_enabled else "Amplitude"
    plt.colorbar(im, ax=ax, label=colorbar_label)


def _plot_heatmap(
    data: np.ndarray,
    title: str,
    save_path: Optional[str],
    show: bool,
    transform_enabled: bool,
    normalize_range: str,
    vis_vmin: Optional[float],
    vis_vmax: Optional[float],
    source_index: int = 0,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_heatmap(ax, data, title, transform_enabled, normalize_range,
                  vis_vmin, vis_vmax, source_index)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_trace(
    data: np.ndarray,
    source_index: int,
    receiver_index: int,
    title: str,
    save_path: Optional[str],
    show: bool,
    transform_enabled: bool,
    normalize_range: str,
    vis_vmin: Optional[float],
    vis_vmax: Optional[float],
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


def _draw_sensor_coords(
    ax: plt.Axes,
    xy: np.ndarray,
    title: str = "Element Locations",
    annotate: bool = False,
    marker_size: float = 30.0,          # 稍微减小默认大小，避免线阵上重叠
) -> None:
    ax.scatter(xy[:, 0], xy[:, 1], s=marker_size, c="#1f77b4",
               edgecolors="black", linewidths=0.4)

    if annotate:
        for idx, (x_val, y_val) in enumerate(xy):
            ax.text(x_val, y_val, str(idx), fontsize=7, ha="left", va="bottom")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.35)

    # ---- 如果所有 y 坐标几乎相同（线阵），放宽显示并取消等比例 ----
    y_std = np.std(xy[:, 1])
    if y_std < 1e-6:                     # 所有 y 值相同（或几乎相同）
        y_min, y_max = -0.5, 0.5         # 人为给 y 轴留出空间
        ax.set_ylim(y_min, y_max)
        # 不设置 aspect，让 x 轴可以自由缩放
    else:
        ax.set_aspect("equal", adjustable="box")

# ---------------------------------------------------------------------------
# Combined RF + layout plot
# ---------------------------------------------------------------------------
def _plot_rf_with_layout(
    rf_data: np.ndarray,
    coords: Optional[np.ndarray],
    title: str,
    save_path: Optional[str],
    show: bool,
    transform_enabled: bool,
    normalize_range: str,
    vis_vmin: Optional[float],
    vis_vmax: Optional[float],
    source_index: int = 0,
    marker_size: float = 30.0,
    annotate: bool = False,           # <-- 新增参数
) -> None:
    if coords is not None:
        try:
            xy = _normalize_xy(coords)
            print(f"Element locations loaded: {xy.shape[0]} points.")
        except Exception as e:
            print(f"Warning: could not normalise element_locations – {e}")
            xy = None
    else:
        xy = None

    if xy is not None:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
        _draw_sensor_coords(ax_left, xy, title="Element Locations",
                            marker_size=marker_size, annotate=annotate)   # 传递 annotate
        _draw_heatmap(ax_right, rf_data, title, transform_enabled,
                      normalize_range, vis_vmin, vis_vmax, source_index)
    else:
        print("No valid element locations – displaying heatmap only.")
        fig, ax_right = plt.subplots(figsize=(10, 5))
        _draw_heatmap(ax_right, rf_data, title, transform_enabled,
                      normalize_range, vis_vmin, vis_vmax, source_index)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Check and plot time_data_cat / RF+layout from a data file.")
    parser.add_argument("--x-path", required=True, help="Path to .npz or .mat file containing time_data_cat or rf_data")
    parser.add_argument("--mode", choices=["heatmap", "trace", "rf"], default="heatmap",
                        help="Plot mode: heatmap/trace (time_data_cat) or rf (RF data + element positions)")
    parser.add_argument("--source-index", type=int, default=0, help="Source index for 3D data (rf mode / trace)")
    parser.add_argument("--receiver-index", type=int, default=0, help="Receiver index for trace mode")
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
    parser.add_argument("--marker-size", type=float, default=42.0, help="Scatter marker size for element locations")
    parser.add_argument("--annotate", action="store_true",
                    help="Annotate each element with its index in the layout plot")

    args = parser.parse_args()

    x_path = os.path.abspath(args.x_path)
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Input not found: {x_path}")

    show = not args.no_show
    transform_enabled = not args.no_transform
    normalize_range = args.normalize_range if transform_enabled else "none"

    # ------------------------------------------------------------------
    # New rf mode – load rf_data + trans_info, print trans_info, plot both
    # ------------------------------------------------------------------
    if args.mode == "rf":
        x_kind, x_obj = _load_x_container(x_path)

        if "rf_data" not in x_obj:
            raise KeyError(f"File at {x_path} does not contain 'rf_data'; mode 'rf' requires it.")
        rf_data = np.asarray(x_obj["rf_data"])

        # Print trans_info members and obtain element_locations
        coords = _print_and_extract_trans_info(x_obj)

        stats, status = _compute_stats(rf_data)
        print("[rf_data] status=", status)
        print(
            "[rf_data] shape={shape}, dtype={dtype}, finite={finite_ratio:.6f}, "
            "nan={nan}, inf={inf}, min={min}, max={max}, mean={mean}, std={std}".format(**stats)
        )

        if status != "OK":
            raise ValueError("rf_data has no finite values; cannot plot.")

        title = os.path.basename(x_path)
        _plot_rf_with_layout(
            rf_data=rf_data,
            coords=coords,
            title=title,
            save_path=args.save,
            show=show,
            transform_enabled=transform_enabled,
            normalize_range=normalize_range,
            vis_vmin=args.vis_vmin,
            vis_vmax=args.vis_vmax,
            source_index=args.source_index,
            marker_size=args.marker_size,
            annotate=args.annotate,
        )
        return

    # ------------------------------------------------------------------
    # Original modes (heatmap / trace) for time_data_cat
    # ------------------------------------------------------------------
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
            source_index=args.source_index,
        )
    else:  # trace
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