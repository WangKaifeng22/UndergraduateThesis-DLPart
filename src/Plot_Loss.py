import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_loss_history(
    csv_path: str,
    out_path: str = None,
    title: str = "Loss History",
    smooth_window: int = 0,
    use_logy: bool = False,
    show: bool = False,
):
    # 1) 读取 CSV
    df = pd.read_csv(csv_path)

    # 2) 基础清洗：确保列存在 + 转数值 + 清理 NaN/Inf
    required_cols = {"iteration", "loss_train", "loss_test"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}, 当前列: {list(df.columns)}")

    for c in ["iteration", "loss_train", "loss_test"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["iteration"])
    df = df.sort_values("iteration")

    it = df["iteration"].to_numpy()
    loss_train = df["loss_train"].to_numpy()
    loss_test = df["loss_test"].to_numpy()

    # 3) 可选平滑：滑动平均（只对可用点做）
    def moving_average(y: np.ndarray, window: int) -> np.ndarray:
        if window is None or window <= 1:
            return y
        # 对 NaN 做处理：用 mask 保持 NaN 不参与卷积
        mask = np.isfinite(y).astype(np.float64)
        y0 = np.nan_to_num(y, nan=0.0)
        kernel = np.ones(window, dtype=np.float64)
        num = np.convolve(y0, kernel, mode="same")
        den = np.convolve(mask, kernel, mode="same")
        out = num / np.clip(den, 1.0, None)
        out[den == 0] = np.nan
        return out

    loss_train_plot = moving_average(loss_train, smooth_window)
    loss_test_plot = moving_average(loss_test, smooth_window)

    # 4) 画图
    plt.figure(figsize=(10, 5))
    if use_logy:
        plt.yscale("log")

    plt.plot(it, loss_train_plot, label="train loss", linewidth=1.5)
    plt.plot(it, loss_test_plot, label="test loss", linewidth=1.5)

    plt.xlabel("iteration")
    plt.ylabel("loss")
    if title is not None:
        plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.tight_layout()

    # 5) 保存/显示
    if out_path is None:
        base, _ = os.path.splitext(csv_path)
        out_path = base + "_plot.png"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()
    return out_path


if __name__ == "__main__":
    # 例子：把这里改成你训练输出的 CSV 路径
    csv_file = r"/home/wkf/wkf_kwave/src/model_50K_5x2_configs_test0_original_0.140625-0.453125/logs/loss_history.csv"
    out_file = r"/home/wkf/wkf_kwave/src/model_50K_5x2_configs_test0_original_0.140625-0.453125/logs/loss.png"

    saved = plot_loss_history(
        csv_path=csv_file,
        out_path=out_file,
        title=None,
        smooth_window=1,   # 0 或 1 表示不平滑
        use_logy=False,    # True 表示 y 轴对数
        show=False,
    )
    print(f"Saved plot to: {saved}")
