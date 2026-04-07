import deepxde as dde
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_FLOWERS_ROOT = _THIS_DIR.parent.parent / "flowers"
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
            num_parameter=2,
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
    ):
        super().__init__()
        self.num_parameter = num_parameter
        self.width = width
        self.Tx = Tx
        self.Rx = Rx
        self.T_steps = T_steps
        self.H = H
        self.W = W

        self.branch = Branch(width=width, Tx=Tx, Rx=Rx, T_steps=T_steps, H=H, W=W)
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

        x = self.flower(x)            # [B, 1, H, W]
        return x.squeeze(1)


class Branch(nn.Module):
    """Encodes [B, Tx, Rx, T] -> [B, C, H, W] via dual Linear projections."""

    def __init__(self, width=64, Tx=32, Rx=32, T_steps=1900, H=80, W=80):
        super().__init__()
        self.width = width
        self.Tx = Tx
        self.Rx = Rx
        self.T_steps = T_steps
        self.H = H
        self.W = W

        self.linear_T = nn.Linear(T_steps, W)
        self.linear_Rx = nn.Linear(Rx, H)
        self.channel_lift = nn.Conv2d(Tx, width, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
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
