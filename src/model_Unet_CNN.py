import deepxde as dde
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

class FourierDeepONet(dde.nn.pytorch.NN):
    """MIONet with two input functions for Cartesian product format."""

    def __init__(
            self,
            num_parameter,
            width=64, #Channel
            modes1=20,
            modes2=20,
            regularization=None,
            merge_operation="mul",
    ):
        super().__init__()
        self.num_parameter = num_parameter
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.branch = Branch(self.width)
        self.trunk = Trunk(self.width, self.num_parameter)
        self.merger = decoder(self.modes1, self.modes2, self.width)
        self.b = nn.Parameter(torch.tensor(0.0))
        self.regularizer = regularization
        self.merge_operation = merge_operation
        #self.fusion_layer = nn.Conv2d(self.width * 2, self.width, kernel_size=1)

    def forward(self, inputs):
        x1 = self.branch(inputs[0])
        x2 = self.trunk(inputs[1])

        if self.merge_operation == "add":
            x = x1 + x2
        elif self.merge_operation == "mul":
            x = torch.mul(x1, x2)
        elif self.merge_operation == "concat":
            # Expand & Concat
            target_h, target_w = x1.shape[2], x1.shape[3]
            x2 = x2.expand(-1, -1, target_h, target_w)
            x = torch.cat([x1, x2], dim=1)

            # Fusion
            x = self.fusion_layer(x)
            x = F.gelu(x)

            x = x
        else:
            raise NotImplementedError(
                f"{self.merge_operation} operation to be implimented"
            )
        x = x + self.b
        x = self.merger(x)

        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        # print("out_ft shape:",out_ft.shape)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1,
                                 dropout_rate=dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1,
                                 dropout_rate=dropout_rate)

        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels * 2, output_channels)
        self.deconv0 = self.deconv(input_channels * 2, output_channels)

        self.output_layer = self.output(input_channels * 2, output_channels,
                                        kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)

        return out

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=(kernel_size - 1) // 2)

from torch.utils.checkpoint import checkpoint


class decoder(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(decoder, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # === 核心层定义 ===
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        #self.unet0 = U_net(self.width, self.width, 3, 0)
        self.unet1 = U_net(self.width, self.width, 3, 0)
        self.unet2 = U_net(self.width, self.width, 3, 0)
        self.unet3 = U_net(self.width, self.width, 3, 0)

        self.linear0 = nn.Linear(1900, 1024)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 80)
        self.linear_R_0 = nn.Linear(32, 64)
        self.linear_R_1 = nn.Linear(64, 80)
        #self.linear_R_2 = nn.Linear(128, 192)
        #self.linear_R_3 = nn.Linear(256, 384)

        """self.resize_conv0 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, bias=False),
            nn.GELU(approximate='tanh')
        )
        self.resize_conv1 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, bias=False),
            nn.GELU(approximate='tanh')
        )
        self.resize_conv2 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, bias=False),
            nn.GELU(approximate='tanh')
        )
        self.resize_conv3 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, bias=False),
            nn.GELU(approximate='tanh')
        )"""

        self.fc1 = nn.Linear(self.width, int(self.width * 2))
        self.fc2 = nn.Linear(int(self.width * 2), 1)

        """self.out_conv1 = nn.Conv2d(self.width, int(self.width * 2), kernel_size=3, padding=1, bias = False)
        self.out_conv2 = nn.Conv2d(int(self.width * 2), 1, kernel_size=1, bias = False)"""

        #self.gn_b0 = nn.GroupNorm(num_groups=32, num_channels=self.width)
        self.gn_b1 = nn.GroupNorm(num_groups=32, num_channels=self.width)
        self.gn_b2 = nn.GroupNorm(num_groups=32, num_channels=self.width)
        self.gn_b3 = nn.GroupNorm(num_groups=32, num_channels=self.width)

    def _resize_and_conv(self, x, target_size, conv_layer):
        #x = F.gelu(x, approximate="tanh")
        # 1. 插值调整 H, W 
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        # 2. 卷积处理特征
        x = conv_layer(x)
        return x

    def _linear_sampling(self, x, linear_last_dim, linear_receiver_dim):
        x = linear_last_dim(x)
        x = F.gelu(x, approximate='tanh')
        x = x.permute(0, 1, 3, 2)
        x = linear_receiver_dim(x)
        x = x.permute(0, 1, 3, 2)
        x = F.gelu(x, approximate='tanh')
        return x

    # === 辅助函数：Block 0 ===
    def _forward_block0(self, x):
        x1 = self.conv0(x)
        x2 = self.w0(x)
        #x3 = self.unet0(x)
        x = x1+x2
        #x = self.gn_b0(x)
        #x = self._resize_and_conv(x, (64, 1024), self.resize_conv0)
        x = self._linear_sampling(x, self.linear0, self.linear_R_0)
        return x

    # === 辅助函数：Block 1 ===
    def _forward_block1(self, x):
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x3 = self.unet1(x)
        x = x1 + x2 + x3
        x = self.gn_b1(x)
        #x = self._resize_and_conv(x, (128, 512), self.resize_conv1)
        x = self._linear_sampling(x, self.linear1, self.linear_R_1)
        return x

    # === 辅助函数：Block 2  ===
    def _forward_block2(self, x):
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x3 = self.unet2(x)
        x = x1 + x2 + x3
        x = self.gn_b2(x)
        #x = self._resize_and_conv(x, (256, 384), self.resize_conv2)
        #x = self._linear_sampling(x, self.linear2, self.linear_R_2)
        x = self.linear2(x)
        x = F.gelu(x, approximate='tanh')

        return x
    # === 辅助函数：Block 3 ===
    def _forward_block3(self, x):
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x3 = self.unet3(x)
        x = x1 + x2 + x3
        x = self.gn_b3(x)
        #x = self._resize_and_conv(x, (384, 384), self.resize_conv3)
        x = self.linear3(x)
        x = F.gelu(x, approximate='tanh')

        return x

    # === 辅助函数：Block 4  & Output ===
    def _forward_block4_out(self, x):

        # 2. 通道融合
        x = x.permute(0, 2, 3, 1)  # (Batch, 384, 384, Width)
        x = self.fc1(x)
        x = F.gelu(x, approximate='tanh')
        x = self.fc2(x)  # (Batch, 384, 384, 1)
        x = x.permute(0, 3, 1, 2)  # (Batch, 1, 384, 384)

        """x = self.out_conv1(x)
        x = F.gelu(x, approximate='tanh')
        x = self.out_conv2(x)"""

        return x.squeeze(1)

    def forward(self, x):

        x = self._forward_block0(x)
        x = self._forward_block1(x)
        x = self._forward_block2(x)
        x = self._forward_block3(x)
        x = self._forward_block4_out(x)

        return x

class Branch(nn.Module):
    def __init__(self, width, input_channels = 32):
        super(Branch, self).__init__()
        self.width = width
        """self.net = nn.Sequential(
            # 1. 投影
            nn.Conv2d(input_channels, self.width, kernel_size=1, bias=False),

            # 2. 归一化: 强行把信号拉回标准正态分布，防止梯度消失
            nn.GroupNorm(num_groups=32, num_channels=self.width),

            # 3. 激活: 增加非线性
            #nn.GELU(approximate='tanh')
        )"""
        self.fc = nn.Linear(32, self.width)
    def forward(self, x):
        x = x.permute(0, 3, 2, 1)  # -1, time_steps,R,32
        x = self.fc(x)  # -1, time_steps, R, width
        x = x.permute(0, 3, 2, 1)  # -1, width, R, time_steps
        x = F.gelu(x, approximate="tanh")
        return x


class Trunk(nn.Module):
    def __init__(self, width, num_parameter):
        super(Trunk, self).__init__()
        self.width = width
        self.num_parameter = num_parameter
        self.fc0 = nn.Linear(self.num_parameter, self.width)

    def forward(self, x):
        x = self.fc0(x)
        x = F.gelu(x, approximate="tanh")

        return x[:, :, None, None]

