"""Gradient sanity-check script.

This script builds your FourierDeepONet, pulls ONE batch from either:
  - HDF5-backed dataset (recommended), or
  - in-memory get_dataset()
Then runs forward -> loss -> backward and prints gradient statistics.

Usage examples:

  python src/check_gradients.py --lazy --h5 ..\\cache\\dataset.h5 --batch-size 8
  python src/check_gradients.py --batch-size 4

Notes
- This is *not* a training script. It runs exactly one backward pass.
- It is intended to diagnose issues like:
    * gradients are all zeros
    * gradients are NaN/Inf
    * only a small subset of parameters receive gradients
    * scale is wildly off (huge/small gradient norms)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple
from datetime import datetime

import numpy as np
import torch

# DeepXDE uses this env for backend selection.
os.environ.setdefault("DDE_BACKEND", "pytorch")

from models.model_FourierDeepONetF import FourierDeepONet  # noqa: E402
from utils.utils import loss_func_L1, set_seed  # noqa: E402


@dataclass
class GradStats:
    name: str
    shape: Tuple[int, ...]
    numel: int
    grad_norm: float | None
    grad_abs_mean: float | None
    grad_abs_max: float | None
    grad_min: float | None
    grad_max: float | None
    has_nan: bool
    has_inf: bool
    is_all_zero: bool


class TeeWriter:
    """Write lines to a text file, optionally echoing to stdout.

    Use `writer.write_line(...)` instead of print().
    """

    def __init__(self, file_path: str, echo: bool = True):
        self.file_path = file_path
        self.echo = echo
        os.makedirs(os.path.dirname(os.path.abspath(file_path)) or ".", exist_ok=True)
        self._f = open(file_path, "w", encoding="utf-8")

    def write_line(self, s: str = "") -> None:
        self._f.write(s + "\n")
        if self.echo:
            print(s)

    def close(self) -> None:
        try:
            self._f.flush()
        finally:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def collect_grad_stats(model: torch.nn.Module) -> list[GradStats]:
    stats: list[GradStats] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.grad is None:
            stats.append(
                GradStats(
                    name=name,
                    shape=tuple(p.shape),
                    numel=p.numel(),
                    grad_norm=None,
                    grad_abs_mean=None,
                    grad_abs_max=None,
                    grad_min=None,
                    grad_max=None,
                    has_nan=False,
                    has_inf=False,
                    is_all_zero=False,
                )
            )
            continue

        g = p.grad
        g_np = _np(g)
        has_nan = bool(np.isnan(g_np).any())
        has_inf = bool(np.isinf(g_np).any())
        abs_g = np.abs(g_np)

        grad_norm = float(np.linalg.norm(g_np.reshape(-1)))
        grad_abs_mean = float(abs_g.mean())
        grad_abs_max = float(abs_g.max())
        grad_min = float(g_np.min())
        grad_max = float(g_np.max())
        is_all_zero = bool(np.all(g_np == 0))

        stats.append(
            GradStats(
                name=name,
                shape=tuple(p.shape),
                numel=p.numel(),
                grad_norm=grad_norm,
                grad_abs_mean=grad_abs_mean,
                grad_abs_max=grad_abs_max,
                grad_min=grad_min,
                grad_max=grad_max,
                has_nan=has_nan,
                has_inf=has_inf,
                is_all_zero=is_all_zero,
            )
        )
    return stats


def print_grad_report(stats: list[GradStats], topk: int = 12, writer: TeeWriter | None = None) -> None:
    """Write a human-readable gradient report.

    If writer is provided, output will go to the file (and possibly echoed).
    Otherwise falls back to printing.
    """

    w = writer.write_line if writer is not None else print

    missing = [s for s in stats if s.grad_norm is None]
    bad_nan = [s for s in stats if s.has_nan]
    bad_inf = [s for s in stats if s.has_inf]
    all_zero = [s for s in stats if (s.grad_norm is not None and s.is_all_zero)]

    present = [s for s in stats if s.grad_norm is not None]
    present_sorted = sorted(present, key=lambda s: s.grad_norm or -1, reverse=True)

    w("\n=== Gradient report ===")
    w(f"Params requiring grad: {len(stats)}")
    w(f"Params with grad=None: {len(missing)}")
    w(f"Params with NaN grads : {len(bad_nan)}")
    w(f"Params with Inf grads : {len(bad_inf)}")
    w(f"Params all-zero grads : {len(all_zero)}")

    if bad_nan:
        w("\n[!] Parameters with NaN gradients:")
        for s in bad_nan[:topk]:
            w(f"  - {s.name} {s.shape}")

    if bad_inf:
        w("\n[!] Parameters with Inf gradients:")
        for s in bad_inf[:topk]:
            w(f"  - {s.name} {s.shape}")

    if missing:
        w("\n[!] Parameters with grad=None (no gradient flowed):")
        for s in missing[:topk]:
            w(f"  - {s.name} {s.shape}")
        if len(missing) > topk:
            w(f"  ... and {len(missing) - topk} more")

    if all_zero:
        w("\n[!] Parameters with all-zero gradients:")
        for s in all_zero[:topk]:
            w(f"  - {s.name} {s.shape}")
        if len(all_zero) > topk:
            w(f"  ... and {len(all_zero) - topk} more")

    w(f"\nTop-{min(topk, len(present_sorted))} gradient norms:")
    for s in present_sorted[:topk]:
        w(
            f"  {s.name:60s} | norm={s.grad_norm:.3e} | abs_mean={s.grad_abs_mean:.3e} | abs_max={s.grad_abs_max:.3e}"
        )

    w(f"\nBottom-{min(topk, len(present_sorted))} (smallest non-None) gradient norms:")
    for s in sorted(present_sorted, key=lambda s: s.grad_norm or 0)[:topk]:
        w(
            f"  {s.name:60s} | norm={s.grad_norm:.3e} | abs_mean={s.grad_abs_mean:.3e} | abs_max={s.grad_abs_max:.3e}"
        )


def get_one_batch_from_h5(h5_path: str, split_ratio: float, batch_size: int, seed: int):
    from utils.h5_dataset import H5DeepONetDataset, H5DatasetConfig
    from train.train import samples_per_config, x_params, y_params, sos_root, kwave_root
    total_data_num = samples_per_config * len(x_params)
    data = H5DeepONetDataset(
        H5DatasetConfig(h5_path=h5_path, split_ratio=split_ratio, test_batch_size=batch_size,
        total_data_num=total_data_num),
        is_deeponet=True,
        seed=seed,
        enable_timing=False,
    )
    (Xb, Xt), y = data.train_next_batch(batch_size)
    return (Xb, Xt), y, data.trunk_dim


def get_one_batch_from_mat(batch_size: int, seed: int):
    # Mirrors defaults in my_train.py
    from utils.multi_data import get_dataset
    from train.train import samples_per_config, x_params, y_params, sos_root, kwave_root
    split_ratio = 0.8
    total_data_num = samples_per_config * len(x_params)

    X_train, X_test, y_train, y_test = get_dataset(
        split_ratio,
        total_data_num,
        True,
        x_params=x_params,
        y_params=y_params,
        sos_root=sos_root,
        kwave_root=kwave_root,
        cache_h5_path=None,  # force .mat loading
    )

    # take one batch from train
    n = X_train[0].shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(batch_size, n), replace=False)
    Xb = X_train[0][idx]
    Xt = X_train[1][idx]
    y = y_train[idx]

    trunk_dim = X_train[1].shape[1]
    return (Xb, Xt), y, trunk_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lazy", action="store_true", help="Load one batch from an HDF5 cache dataset")
    parser.add_argument("--h5", type=str, default=None, help="Path to dataset.h5 when using --lazy")
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=114514)

    # model hyperparams (match my_train defaults)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--modes1", type=int, default=12)
    parser.add_argument("--modes2", type=int, default=20)
    parser.add_argument("--merge", type=str, default="mul", choices=["mul", "add"])

    # numeric debug
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # output
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write gradient report to this text file. Default: ./grad_reports/grad_YYYYmmdd_HHMMSS.txt",
    )
    parser.add_argument(
        "--no-echo",
        action="store_true",
        help="Do not echo report to stdout (only write to --out file).",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # derive output path
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join("grad_reports", f"grad_{ts}.txt")

    # Load one batch
    if args.lazy:
        if not args.h5:
            raise SystemExit("--lazy requires --h5 PATH")
        X, y, trunk_dim = get_one_batch_from_h5(args.h5, split_ratio=args.split, batch_size=args.batch_size, seed=args.seed)
    else:
        X, y, trunk_dim = get_one_batch_from_mat(batch_size=args.batch_size, seed=args.seed)

    Xb, Xt = X

    # Convert to torch tensors
    Xb_t = torch.as_tensor(np.asarray(Xb), dtype=torch.float32, device=device)
    Xt_t = torch.as_tensor(np.asarray(Xt), dtype=torch.float32, device=device)
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device)

    # Build model
    net = FourierDeepONet(
        num_parameter=int(trunk_dim),
        width=int(args.width),
        modes1=int(args.modes1),
        modes2=int(args.modes2),
        regularization=["l2", 3e-6],
        merge_operation=args.merge,
    ).to(device)

    net.train()

    # Forward + loss + backward
    # Your DeepONet expects input as (X_branch, X_trunk)
    y_pred = net((Xb_t, Xt_t))

    # DeepXDE's loss_func_L1 expects (y_true, y_pred)
    # Ensure shapes match
    if y_pred.shape != y_t.shape:
        print(f"[Warn] y_pred shape {tuple(y_pred.shape)} != y shape {tuple(y_t.shape)}")

    loss = loss_func_L1(y_t, y_pred)

    net.zero_grad(set_to_none=True)
    loss.backward()

    with TeeWriter(args.out, echo=(not args.no_echo)) as w:
        w.write_line(f"Device: {device}")
        w.write_line(f"Batch size: {Xb_t.shape[0]}")
        w.write_line(f"Loss: {float(loss.detach().cpu().item()):.6g}")

        stats = collect_grad_stats(net)
        print_grad_report(stats, topk=12, writer=w)

        # Extra: overall grad norm
        total_norm_sq = 0.0
        for p in net.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if g.is_complex():
                # complex grad: use magnitude
                total_norm_sq += float((g.real ** 2 + g.imag ** 2).sum().cpu().item())
            else:
                total_norm_sq += float((g ** 2).sum().cpu().item())
        total_norm = float(np.sqrt(total_norm_sq))
        w.write_line(f"\nTotal grad L2 norm (all params): {total_norm:.6g}")

    # Always show where the report is saved
    print(f"\n[check_gradients] Report written to: {args.out}")


if __name__ == "__main__":
    main()

