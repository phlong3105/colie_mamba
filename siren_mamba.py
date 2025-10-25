# siren_mamba.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mambaBlock import MambaBlock


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        if not self.is_last:
            self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return nn.Sigmoid()(x) if self.is_last else torch.sin(self.w0 * x)


class EncoderBlock(nn.Module):
    """UNet编码器块：MambaBlock + 下采样"""
    def __init__(self, in_channels, out_channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba_block = MambaBlock(
            d_model=in_channels, d_state=d_state, d_conv=d_conv, expand=expand
        )
        # 修改3：LayerNorm仅作用于通道维度in_channels
        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.SiLU()
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # 残差分支（Mamba处理）
        residual = self.mamba_block(x)  # (B,C,H,W)
        # 修改4：转置维度使通道在最后，归一化后恢复
        residual = residual.permute(0, 2, 3, 1)  # (B,H,W,C)
        residual = self.norm(residual)  # 对通道维度归一化
        residual = residual.permute(0, 3, 1, 2)  # (B,C,H,W)
        residual = self.act(residual)
        # 下采样分支
        down = self.downsample(residual)
        return down, residual


class DecoderBlock(nn.Module):
    """UNet解码器块：上采样 + 跳跃连接 + MambaBlock"""
    def __init__(self, in_channels, skip_channels, out_channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, skip_channels, kernel_size=2, stride=2, padding=0
        )
        self.mamba_block = MambaBlock(
            d_model=skip_channels * 2, d_state=d_state, d_conv=d_conv, expand=expand
        )
        # 修改5：LayerNorm作用于拼接后的通道维度（skip_channels*2）
        self.norm = nn.LayerNorm(skip_channels * 2)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(skip_channels * 2, out_channels, kernel_size=1, padding=0)

    def forward(self, x, skip_x):
        x_up = self.upsample(x)  # (B, skip_channels, H, W)
        x_cat = torch.cat([x_up, skip_x], dim=1)  # (B, skip_channels*2, H, W)
        # Mamba处理
        x_mamba = self.mamba_block(x_cat)
        # 修改6：转置维度使通道在最后，归一化后恢复
        x_mamba = x_mamba.permute(0, 2, 3, 1)  # (B,H,W,2*C)
        x_mamba = self.norm(x_mamba)
        x_mamba = x_mamba.permute(0, 3, 1, 2)  # (B,2*C,H,W)
        x_mamba = self.act(x_mamba)
        # 通道调整
        x_out = self.conv_out(x_mamba)
        return x_out


class INF_MAMBA(nn.Module):
    """基于UNet的INF_MAMBA：用Mamba提取全局上下文"""
    def __init__(self, in_channels=3,
                 base_channels=32,
                 d_state=16,
                 d_conv=4,
                 expand=2):
        super().__init__()
        self.base_channels = base_channels
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        # 输入处理
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        # 修改7：LayerNorm作用于输入通道维度base_channels
        self.input_norm = nn.LayerNorm(base_channels)
        self.input_act = nn.SiLU()

        # UNet编码器
        self.encoder1 = EncoderBlock(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.encoder2 = EncoderBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # 瓶颈层
        self.bottleneck = MambaBlock(
            d_model=base_channels * 4,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        # 修改8：LayerNorm作用于瓶颈层通道维度4*base_channels
        self.bottleneck_norm = nn.LayerNorm(base_channels * 4)
        self.bottleneck_act = nn.SiLU()

        # UNet解码器
        self.decoder1 = DecoderBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.decoder2 = DecoderBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 残差范围约束到[0,1]
        )

    def get_coord_embedding(self, H, W, device):
        """生成坐标嵌入：(B, 2, H, W)"""
        x_coord = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).repeat(1, 1, H, 1)
        y_coord = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).repeat(1, 1, 1, W)
        return torch.cat([x_coord, y_coord], dim=1)

    def forward(self, img_v):
        B, C, H, W = img_v.shape
        device = img_v.device

        # 坐标嵌入拼接
        coord_emb = self.get_coord_embedding(H, W, device).repeat(B, 1, 1, 1)  # (B,2,H,W)
        x_in = torch.cat([img_v, coord_emb], dim=1)  # (B,3,H,W)

        # 输入处理
        x = self.input_conv(x_in)  # (B, base_channels, H, W)
        # 修改9：输入归一化时转置维度
        x = x.permute(0, 2, 3, 1)  # (B,H,W,base_channels)
        x = self.input_norm(x)
        x = x.permute(0, 3, 1, 2)  # 恢复为(B,base_channels,H,W)
        x = self.input_act(x)

        # 编码器前向
        x1, skip1 = self.encoder1(x)
        x2, skip2 = self.encoder2(x1)

        # 瓶颈层
        x_bottleneck = self.bottleneck(x2)  # (B,4C,H/4,W/4)
        # 修改10：瓶颈层归一化时转置维度
        x_bottleneck = x_bottleneck.permute(0, 2, 3, 1)  # (B,H/4,W/4,4C)
        x_bottleneck = self.bottleneck_norm(x_bottleneck)
        x_bottleneck = x_bottleneck.permute(0, 3, 1, 2)  # 恢复为(B,4C,H/4,W/4)
        x_bottleneck = self.bottleneck_act(x_bottleneck)

        # 解码器前向
        x_dec1 = self.decoder1(x_bottleneck, skip2)
        x_dec2 = self.decoder2(x_dec1, skip1)

        # 输出光照残差
        illu_res = self.output_conv(x_dec2)  # (B,1,H,W)

        return illu_res