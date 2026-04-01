import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def get_group_norm(channels):
    # 策略：如果通道数 >= 32，尝试用 32 组；否则用通道数的一半作为组数
    # 或者简单点：找一个能整除 channels 且最接近 32 的数
    # 但对于你的网络(全是2的倍数)，下面的逻辑最稳健：

    # 优先设为 32，如果 channels 小于 32，则设为 channels/2 (最小8组)
    groups = 32
    if channels < 32:
        groups = 8  # 针对 dim0=16 的情况，8组每组2通道

    # 再次确保能整除（防止未来你改了奇数通道）
    if channels % groups != 0:
        groups = 1  # 降级为 LayerNorm 以防报错

    return nn.GroupNorm(num_groups=groups, num_channels=channels)


NORM_LAYERS = {
    'bn': nn.BatchNorm2d,
    'in': nn.InstanceNorm2d,
    'ln': nn.LayerNorm,
    'gn': get_group_norm  # 将 gn 指向这个函数
}

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=(3,3), stride=(1,1), padding=(1,1), norm='gn', relu_slop=0.2, dropout=None):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=(3,3), stride=(1,1), padding=(1,1), norm='gn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=(2,2), stride=(2,2), padding=(0,0), output_padding=0, norm='gn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                                     padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class InversionNet(nn.Module):
    def __init__(self, dim0=32, dim1=64, dim2=64, dim3=128, dim4=256, dim5=512, regularization=None, **kwargs):
        super(InversionNet, self).__init__()
        self.regularizer = regularization

        # ================= Encoder (保持你的修改) =================
        # Input: [B, 32, 32, 1900]

        # Layer 1-3 (长方形特征，不参与 Skip Connection)
        self.convblock1 = ConvBlock(32, dim1, kernel_size=(3, 7), stride=(1, 4), padding=(1, 3))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 3), padding=(1, 1))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 3), padding=(1, 1))

        # Layer 4 (正方形 32x32, 保存特征 e4)
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 3), stride=(1, 4), padding=(1, 4))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 3), padding=(1, 1))

        # Layer 5 (正方形 16x16, 保存特征 e5)
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)

        # Layer 6 (正方形 8x8, 保存特征 e6)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)

        # Layer 7 (正方形 4x4, 保存特征 e7)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)

        # Layer 8 (Bottleneck 1x1)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(4, 4), padding=0)

        # ================= Decoder =================

        # Decoder 1: 1x1 -> 4x4. 拼接 e7 (dim4)
        # Kernel=4, Stride=1, Pad=0 => (1-1)*1 - 0 + 4 = 4
        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=4, stride=1, padding=0)
        # 输入通道: dim5 (上层) + dim4 (e7)
        self.deconv1_2 = ConvBlock(dim5 + dim4, dim5)

        # Decoder 2: 4x4 -> 8x8. 拼接 e6 (dim4)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        # 输入通道: dim4 (上层) + dim4 (e6)
        self.deconv2_2 = ConvBlock(dim4 + dim4, dim4)

        # Decoder 3: 8x8 -> 16x16. 拼接 e5 (dim3)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        # 输入通道: dim3 (上层) + dim3 (e5)
        self.deconv3_2 = ConvBlock(dim3 + dim3, dim3)

        # Decoder 4: 16x16 -> 32x32. 拼接 e4 (dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        # 输入通道: dim2 (上层) + dim3 (e4)
        self.deconv4_2 = ConvBlock(dim2 + dim3, dim2)

        # --- 以下层不进行拼接，负责将 32x32 放大到 80x80 ---

        # Decoder 5: 32x32 -> 40x40
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=9, stride=1, padding=0)
        self.deconv5_2 = ConvBlock(dim1, dim1)

        # Decoder 6: 40x40 -> 80x80
        self.deconv6_1 = DeconvBlock(dim1, dim0, kernel_size=4, stride=2, padding=1)
        self.deconv6_2 = ConvBlock(dim0, dim0)

        # Decoder 7: 80x80 -> 80x80 (keep size)
        self.deconv7_1 = DeconvBlock(dim0, dim0, kernel_size=3, stride=1, padding=1)
        self.deconv7_2 = ConvBlock(dim0, dim0)

        # Output
        #self.final_conv = nn.Conv2d(dim0, 1, kernel_size=3, padding=1)
        self.final_conv = ConvBlock_Tanh(dim0, 1)

    def forward(self, x):
        # --- Encoder ---
        x = self.convblock1(x)
        x = self.convblock2_1(x);x = self.convblock2_2(x)
        x = self.convblock3_1(x);x = self.convblock3_2(x)

        # Layer 4 (Save e4: 32x32)
        x = self.convblock4_1(x);x = self.convblock4_2(x)
        e4 = x

        # Layer 5 (Save e5: 16x16)
        x = self.convblock5_1(x);x = self.convblock5_2(x)
        e5 = x

        # Layer 6 (Save e6: 8x8)
        x = self.convblock6_1(x);x = self.convblock6_2(x)
        e6 = x

        # Layer 7 (Save e7: 4x4)
        x = self.convblock7_1(x);x = self.convblock7_2(x)
        e7 = x

        # Layer 8 (Bottleneck: 1x1)
        x = self.convblock8(x)

        # --- Decoder (With Skip Connections) ---

        # Block 1 (Match e7)
        x = self.deconv1_1(x)  # [B, dim5, 4, 4]
        x = torch.cat([x, e7], dim=1)  # Concat -> [B, dim5+dim4, 4, 4]
        x = self.deconv1_2(x)

        # Block 2 (Match e6)
        x = self.deconv2_1(x)  # [B, dim4, 8, 8]
        x = torch.cat([x, e6], dim=1)  # Concat -> [B, dim4+dim4, 8, 8]
        x = self.deconv2_2(x)

        # Block 3 (Match e5)
        x = self.deconv3_1(x)  # [B, dim3, 16, 16]
        x = torch.cat([x, e5], dim=1)  # Concat -> [B, dim3+dim3, 16, 16]
        x = self.deconv3_2(x)

        # Block 4 (Match e4)
        x = self.deconv4_1(x)  # [B, dim2, 32, 32]
        x = torch.cat([x, e4], dim=1)  # Concat -> [B, dim2+dim3, 32, 32]
        x = self.deconv4_2(x)

        # --- Remaining Blocks (No Skips, just Upsampling) ---
        x = self.deconv5_1(x);x = self.deconv5_2(x)
        x = self.deconv6_1(x);x = self.deconv6_2(x)
        x = self.deconv7_1(x);x = self.deconv7_2(x)

        x = self.final_conv(x)
        return x.squeeze(1)

def plot_grad_flow(named_parameters):
    """
    绘制梯度流动图。
    x轴：网络层（从Encoder到Decoder）
    y轴：梯度幅值的平均值和最大值
    """
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and (p.grad is not None):
            layers.append(n)
            # 这里的 grad.cpu() 是为了防止在 GPU 上绘图报错
            # .abs() 取绝对值，.mean() 取平均
            ave_grads.append(p.grad.abs().mean().cpu().item())
            max_grads.append(p.grad.abs().max().cpu().item())

    # 打印详细数值，方便Debug
    print(f"\n{'Layer Name':<50} | {'Max Grad':<15} | {'Avg Grad':<15}")
    print("-" * 85)
    for i, name in enumerate(layers):
        # 只打印主要层，避免刷屏
        if i % 2 == 0 or "convblock8" in name or "final" in name:
            print(f"{name:<50} | {max_grads[i]:.2e}        | {ave_grads[i]:.2e}")

    plt.figure(figsize=(15, 8))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="c", label='Max Gradient')
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=1, lw=1, color="b", label='Average Gradient')
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))

    # 梯度消失的判断标准：通常小于 1e-6 就很危险了
    plt.ylim(bottom=-0.0001, top=0.02)  # 根据实际情况调整Y轴
    plt.xlabel("Layers")
    plt.ylabel("Gradient magnitude")
    plt.title("Gradient flow along the network")
    plt.legend()
    plt.grid(True, which='both', axis='y', linestyle='--')
    plt.tight_layout()

    # 保存图片
    save_path = "./grad/gradient_flow_check.png"
    plt.savefig(save_path)
    print(f"\nGradient flow plot saved to {save_path}")
    plt.show()


def check_gradients():
    # 1. 准备模型和数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    model = InversionNet().to(device)

    # 模拟输入 (Batch=2, Channels=32, H=32, W=1900)
    x = torch.randn((2, 32, 32, 1900)).to(device)

    # 模拟目标 (Batch=2, H=80, W=80)
    target = torch.randn((2, 80, 80)).to(device)

    # 2. 前向传播
    model.zero_grad()
    output = model(x)

    # 3. 计算 Loss
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print(f"Dummy Loss: {loss.item()}")

    # 4. 反向传播 (此时计算梯度)
    loss.backward()

    # 5. 分析梯度
    print("\nAnalyzing gradients...")
    plot_grad_flow(model.named_parameters())


if __name__ == "__main__":
    check_gradients()

    """# Test
    x = torch.randn((2, 32, 32, 1900))
    model = InversionNet()
    out = model(x)
    print("Input:", x.shape)
    print("Output:", out.shape)

    # Check dims
    assert out.shape[1:] == (80, 80)
    print("Passed!")"""