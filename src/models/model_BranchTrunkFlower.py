import deepxde as dde
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

_FLOWERS_ROOT = "/home/wkf/wkf_kwave/flowers" #https://github.com/t-muser/flowers
if str(_FLOWERS_ROOT) not in sys.path:
    sys.path.append(str(_FLOWERS_ROOT))

from flowers.models.flower import Flower


class BranchTrunkFlower(dde.nn.pytorch.NN):
    """Branch-Trunk-Flower model for ultrasound CT speed-of-sound reconstruction.
    
    Branch: [B, Tx, Rx, T] -> [B, C, H, W] via dual Linear projections
    Trunk:  [B, 2] -> [B, C, 1, 1] via Linear
    Fusion: branch * trunk (broadcast multiply)
    Decoder: Flower U-Net -> [B, H, W]
    """

    def __init__(
            self,
            num_parameter=64,
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
            regularization=None,
            channel_lift_first=True,
    ):
        super().__init__()
        self.num_parameter = num_parameter
        self.width = width
        self.Tx = Tx
        self.Rx = Rx
        self.T_steps = T_steps
        self.H = H
        self.W = W
        self.regularizer = regularization

        self.branch = Branch(width=width, Tx=Tx, Rx=Rx, T_steps=T_steps, H=H, W=W, channel_lift_first=channel_lift_first)
        self.trunk = Trunk(width=width, num_parameter=num_parameter)

        self.flower = Flower(
            dim_in=width,
            dim_out=1,
            n_spatial_dims=2,
            spatial_resolution=(H, W),
            lifting_dim=lifting_dim,
            n_levels=n_levels,
            num_heads=num_heads,
            boundary_condition_types=boundary_condition_types,
            dim_meta=0,
            dropout_rate=dropout_rate,
        )

        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        x1 = self.branch(inputs[0])   # [B, width, H, W]
        x2 = self.trunk(inputs[1])    # [B, width, 1, 1]

        x = torch.mul(x1, x2)         # broadcast -> [B, width, H, W]
        x = x + self.b

        """print(
            f"x1_mean={torch.mean(x1).item():.6f}, x2_mean={torch.mean(x2).item():.6f}\n"
            f"x1_std={torch.std(x1).item():.6f}, x2_std={torch.std(x2).item():.6f}\n"
            f"x_mean={torch.mean(x).item():.6f}, x_std={torch.std(x).item():.6f}, "
            f"x_min={torch.min(x).item():.6f}, x_max={torch.max(x).item():.6f}\n"
        )"""

        x = self.flower(x)            # [B, 1, H, W]
        return x.squeeze(1)


class Branch(nn.Module):
    """Encodes [B, Tx, Rx, T] -> [B, C, H, W] via dual Linear projections."""

    def __init__(self, width=64, Tx=32, Rx=32, T_steps=1900, H=80, W=80, channel_lift_first = False):
        super().__init__()
        self.width = width
        self.Tx = Tx
        self.Rx = Rx
        self.T_steps = T_steps
        self.H = H
        self.W = W

        self.linear_T = nn.Linear(T_steps, W)
        self.linear_Rx = nn.Linear(Rx, H)
        self.channel_lift_first = channel_lift_first
        if self.channel_lift_first:
            self.linear_C_0 = nn.Linear(Tx, int(width*2))
            self.linear_C_1 = nn.Linear(int(width*2), width)
        else:
            self.channel_lift = nn.Conv2d(Tx, width, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        if self.channel_lift_first:
            # Lift Tx axis to channel width
            x = x.permute(0, 3, 2, 1)     # [B, T, Rx, Tx]
            x = self.linear_C_0(x)        # [B, T, Rx, width*2]

            # Project Rx -> H
            x = x.permute(0, 1, 3, 2)     # [B, T, width*2, Rx]
            x = self.linear_Rx(x)         # [B, T, width*2, H]
            x = x.permute(0, 2, 3, 1)     # [B, T, H, width*2]

            x = self.linear_T(x)          # [B, width*2, Rx, Tx]

            x = F.gelu(x, approximate="tanh")
            x = x.permute(0, 3, 2, 1)     # [B, Tx, Rx, width*2]
            x = self.linear_C_1(x)        # [B, Tx, Rx, width]
            x = x.permute(0, 3, 2, 1)     # [B, width, Rx, Tx]
        else:
            # x: [B, Tx, Rx, T]
            # Project T -> W
            x = self.linear_T(x)          # [B, Tx, Rx, W]

            # Project Rx -> H
            x = x.permute(0, 1, 3, 2)     # [B, Tx, W, Rx]
            x = self.linear_Rx(x)         # [B, Tx, W, H]
            x = x.permute(0, 1, 3, 2)     # [B, Tx, H, W]

            # Lift Tx axis to channel width
            x = self.channel_lift(x)      # [B, width, H, W]
            x = F.gelu(x, approximate="tanh")
        return x


class Trunk(nn.Module):
    """Encodes [B, num_params] -> [B, C, 1, 1]."""

    def __init__(self, width, num_parameter):
        super().__init__()
        self.width = width
        self.num_parameter = num_parameter
        self.fc0 = nn.Linear(num_parameter, width)

    def forward(self, x):
        x = self.fc0(x)
        x = F.gelu(x, approximate="tanh")
        return x[:, :, None, None]
