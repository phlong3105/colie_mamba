# This file provides a robust implementation of the MambaBlock for 2D vision tasks.
# To use this, you will need to install the official Mamba implementation:
# pip install mamba-ssm causal-conv1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """
    A Mamba block adapted for 2D vision tasks.
    The input is expected to be in the shape (B, C, H, W).
    The block flattens the spatial dimensions, processes the sequence with Mamba,
    and then reshapes it back to the original 2D format.
    """

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2):
        """
        Args:
            d_model (int): The feature dimension.
            d_state (int): The state dimension of the SSM.
            d_conv (int): The kernel size of the 1D convolution.
            expand (int): The expansion factor for the hidden dimension.
        """
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

        # Mamba requires the input to be (B, L, D) where L is sequence length and D is dimension.
        self.mambas = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(4)
        ])

    def forward(self, x):
        """
        Forward pass for the 2D Mamba block.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output tensor of the same shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        if C != self.d_model:
            raise ValueError(f"Input channel dimension {C} does not match Mamba d_model {self.d_model}")

        # Pre-normalization
        x_norm = self.norm(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, H, W)

        # --- Multi-Directional Scanning ---

        # 1. Forward scan (top-left to bottom-right)
        x_f = x_norm.flatten(2).transpose(1, 2)  # (B, H*W, C)
        y_f = self.mambas[0](x_f)

        # 2. Reverse scan (bottom-right to top-left)
        x_r = torch.flip(x_f, dims=[1])
        y_r = torch.flip(self.mambas[1](x_r), dims=[1])

        # 3. Transposed-Forward scan (column-wise)
        x_t = x_norm.permute(0, 1, 3, 2).contiguous()  # (B, C, W, H)
        x_tf = x_t.flatten(2).transpose(1, 2)  # (B, W*H, C)
        y_tf = self.mambas[2](x_tf)
        y_t = y_tf.transpose(1, 2).view(B, C, W, H).permute(0, 1, 3, 2).contiguous()  # Reshape back

        # 4. Transposed-Reverse Scan
        x_tr = torch.flip(x_tf, dims=[1])
        y_tr = torch.flip(self.mambas[3](x_tr), dims=[1])
        y_tr = y_tr.transpose(1, 2).view(B, C, W, H).permute(0, 1, 3, 2).contiguous()

        # Combine the outputs from all directions
        y_f = y_f.transpose(1, 2).view(B, C, H, W)
        y_r = y_r.transpose(1, 2).view(B, C, H, W)

        y = y_f + y_r + y_t + y_tr

        return y
