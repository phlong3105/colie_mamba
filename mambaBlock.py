import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    """自定义4方向扫描Mamba块：水平正/反向、垂直正/反向"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model  # 特征通道数
        # 4个方向的Mamba层（保持不变）
        self.mamba_h_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_h_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_v_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_v_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        # 特征融合层（保持定义不变，卷积层需要(B,C,H,W)输入）
        self.fuse = nn.Sequential(
            nn.Conv2d(d_model * 4, d_model, kernel_size=1, padding=0),  # 输入4C，输出C
            nn.LayerNorm(d_model),  # 作用于通道维度C（需通道在最后）
            nn.SiLU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.d_model, f"MambaBlock输入通道数需为{self.d_model}，当前为{C}"

        # 水平正向/反向、垂直正向/反向处理（保持不变）
        # 水平正向处理
        x_h_fwd = x.permute(0, 2, 3, 1).reshape(B * H, W, C)  # (B*H, W, C)
        x_h_fwd = self.mamba_h_fwd(x_h_fwd)
        x_h_fwd = x_h_fwd.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B,C,H,W)

        # 水平反向处理
        x_h_bwd = torch.flip(x, dims=[3])
        x_h_bwd = x_h_bwd.permute(0, 2, 3, 1).reshape(B * H, W, C)
        x_h_bwd = self.mamba_h_bwd(x_h_bwd)
        x_h_bwd = x_h_bwd.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_h_bwd = torch.flip(x_h_bwd, dims=[3])

        # 垂直正向处理
        x_v_fwd = x.permute(0, 3, 2, 1).reshape(B * W, H, C)  # (B*W, H, C)
        x_v_fwd = self.mamba_v_fwd(x_v_fwd)
        x_v_fwd = x_v_fwd.reshape(B, W, H, C).permute(0, 3, 2, 1)  # (B,C,H,W)

        # 垂直反向处理
        x_v_bwd = torch.flip(x, dims=[2])
        x_v_bwd = x_v_bwd.permute(0, 3, 2, 1).reshape(B * W, H, C)
        x_v_bwd = self.mamba_v_bwd(x_v_bwd)
        x_v_bwd = x_v_bwd.reshape(B, W, H, C).permute(0, 3, 2, 1)
        x_v_bwd = torch.flip(x_v_bwd, dims=[2])

        # 融合4方向特征（关键修改：调整维度顺序）
        x_fused = torch.cat([x_h_fwd, x_h_bwd, x_v_fwd, x_v_bwd], dim=1)  # (B, 4C, H, W) → 卷积层需要的格式
        x_fused = self.fuse[0](x_fused)  # 先通过卷积层：(B, 4C, H, W) → (B, C, H, W)

        # 转置维度适配LayerNorm（通道放最后）
        x_fused = x_fused.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_fused = self.fuse[1](x_fused)  # LayerNorm作用于通道维度C
        x_fused = self.fuse[2](x_fused)  # SiLU激活
        x_fused = x_fused.permute(0, 3, 1, 2)  # 恢复为(B, C, H, W)

        return x_fused
