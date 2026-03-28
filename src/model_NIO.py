import torch
import torch.nn as nn

from DeepONetModules import FeedForwardNN, DeepOnetNoBiasOrg
from FNOModules import FNO_WOR


class EncoderUSCT(nn.Module):
    """
    Minimal encoder for USCT measurements:
    Input:  x [B, 32, 32, 1900]
    Output: z [B, L=1024, n_basis]
            where each tx-rx pair is a token (L = 32*32)
    """
    def __init__(self, n_basis: int, time_steps: int = 1900, hidden: int = 256):
        super().__init__()
        self.n_basis = n_basis
        self.time_steps = time_steps

        # MLP over time axis for each (tx, rx) token
        self.time_mlp = nn.Sequential(
            nn.Linear(time_steps, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_basis)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 32, 32, 1900]
        return: [B, 1024, n_basis]
        """
        B, n_tx, n_rx, T = x.shape
        assert n_tx == 32 and n_rx == 32, f"Expected [B,32,32,T], got {x.shape}"
        assert T == self.time_steps, f"Expected T={self.time_steps}, got T={T}"

        # [B, 32, 32, 1900] -> [B, 1024, 1900]
        x = x.reshape(B, n_tx * n_rx, T)

        # token-wise time encoding -> [B, 1024, n_basis]
        z = self.time_mlp(x)
        return z

    def print_size(self):
        nparams = sum(p.numel() for p in self.parameters())
        print(nparams)
        return nparams


class NIOUltrasoundCTAbl(nn.Module):
    """
    NIO for Ultrasound CT (no random subsampling).
    - Branch input: x_meas [B, 32, 32, 1900]
    - Trunk input:  grid [nx, ny, 2] (e.g. [256,256,2])
    - Output:       y [B, nx, ny]
    """
    def __init__(self,
                 input_dimensions_trunk,
                 network_properties_branch,
                 network_properties_trunk,
                 fno_architecture,
                 device,
                 retrain_seed,
                 padding_frac=1/4,
                 usct_time_steps=1900,
                 usct_hidden=256,
                 regularization=None):
        super(NIOUltrasoundCTAbl, self).__init__()

        output_dimensions = network_properties_trunk["n_basis"]  # n_basis
        fno_architecture["retrain_fno"] = retrain_seed
        network_properties_branch["retrain"] = retrain_seed
        network_properties_trunk["retrain"] = retrain_seed

        # Trunk: should take 2D coordinates (x,y), so input_dimensions_trunk should be 2
        self.trunk = FeedForwardNN(input_dimensions_trunk, output_dimensions, network_properties_trunk)

        # Branch: USCT encoder, output [B, 1024, n_basis]
        self.branch = EncoderUSCT(
            n_basis=output_dimensions,
            time_steps=usct_time_steps,
            hidden=usct_hidden
        )

        self.deeponet = DeepOnetNoBiasOrg(self.branch, self.trunk)

        self.fno_layers = fno_architecture["n_layers"]

        # keep the same "2 + aggregated measurement channel" trick as existing NIO code
        self.fc0 = nn.Linear(3, fno_architecture["width"])

        if self.fno_layers != 0:
            self.fno = FNO_WOR(fno_architecture, device=device, padding_frac=padding_frac)
        else:
            self.fno = None

        self.device = device

        self.regularizer = regularization

    def forward(self, x, grid=None):
        """
        x:    [B, 32, 32, 1900]
        grid: [nx, ny, 2]
        """
        # 兼容 DeepXDE 的单参数调用：net((x, grid))
        if grid is None:
            if isinstance(x, (tuple, list)) and len(x) == 2:
                x, grid = x
            else:
                raise TypeError("NIOUltrasoundCTAbl.forward expects (x, grid) or x, grid")

        # 1) Branch encode (NO random subsampling)
        # branch_out: [B, L, n_basis], L=1024
        branch_out = self.branch(x)
        L = branch_out.shape[1]

        # 2) DeepONet query on reconstruction grid
        nx, ny = grid.shape[0], grid.shape[1]
        grid_r = grid.reshape(-1, 2)  # [nx*ny, 2]

        # 避免重复过 branch：branch_out 已是编码后的权重 [B, L, p]
        # DeepONet 核心计算: (weights @ basis^T + b0) / sqrt(p)
        basis = self.trunk(grid_r)  # [nx*ny, p]
        u = (torch.matmul(branch_out, basis.T) + self.deeponet.b0) / (self.deeponet.p ** 0.5)
        u = u.view(u.shape[0], u.shape[1], nx, ny)  # [B, L, nx, ny]

        # 3) Concatenate coordinates as channels
        grid_b = grid.unsqueeze(0).expand(u.shape[0], nx, ny, 2).permute(0, 3, 1, 2)  # [B,2,nx,ny]
        h = torch.cat((grid_b, u), dim=1)  # [B,2+L,nx,ny]

        # 4) Project (2+L) -> width using fc0 expansion trick
        W = self.fc0.weight.data  # [width, 3]
        b = self.fc0.bias.data    # [width]
        W_expand = torch.cat(
            [W[:, :2], W[:, 2].view(-1, 1).repeat(1, L) / L],
            dim=1
        )  # [width, 2+L]

        h = h.permute(0, 2, 3, 1)               # [B,nx,ny,2+L]
        h = torch.matmul(h, W_expand.T) + b     # [B,nx,ny,width]

        # 5) FNO refinement
        if self.fno is not None:
            h = self.fno(h)                     # usually [B,nx,ny,1]
            out = h[..., 0]                     # [B,nx,ny]
        else:
            # fallback if no FNO
            out = h[..., 0]                     # [B,nx,ny]

        return out

    def print_size(self):
        print("Branch params:")
        b_size = self.branch.print_size()
        print("Trunk params:")
        t_size = self.trunk.print_size()

        if self.fno is not None:
            print("FNO params:")
            f_size = self.fno.print_size()
            size = b_size + t_size + f_size
        else:
            print("NO FNO")
            size = b_size + t_size

        print(size)
        return size

    def regularization(self, q):
        reg_loss = 0
        for _, param in self.named_parameters():
            reg_loss = reg_loss + torch.norm(param, q)
        return reg_loss