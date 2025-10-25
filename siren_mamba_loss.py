import torch
import torch.nn as nn
import numpy as np
from mamba_branch import MambaBlock  # Assuming mamba.py with VisionMambaBlock exists


# --- Re-introducing the SirenLayer for the spatial branch ---
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_f, 1 / self.in_f)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_f) / self.w0, np.sqrt(6 / self.in_f) / self.w0)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


# --- U-Net Helper Blocks (Unchanged) ---
class MambaEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mamba_config):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.mamba = MambaBlock(d_model=out_channels, **mamba_config)

    def forward(self, x):
        x = self.downsample(x)
        x_mamba = self.mamba(x)
        return x + x_mamba


class MambaDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, mamba_config):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.mamba = MambaBlock(d_model=out_channels + skip_channels, **mamba_config)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        x = torch.cat([x, skip_connection], dim=1)
        x_after_mamba = self.mamba(x)
        x = x + x_after_mamba
        x = self.conv(x)
        return x


# --- Final Three-Branch Network ---
class CoLIENet(nn.Module):
    def __init__(self, hidden_dim, mamba_d_model, mamba_d_state, mamba_d_conv, mamba_expand, down_size):
        super().__init__()

        #patch_feat_dim = hidden_dim // 4
        spatial_feat_dim = hidden_dim // 4
        mamba_feat_dim = hidden_dim // 2

        # Branch 1: CNN for local features with added BatchNorm
        # self.patch_net = nn.Sequential(
        #     nn.Conv2d(1, patch_feat_dim // 2, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(patch_feat_dim // 2),  # <-- Added Normalization
        #     nn.GELU(),
        #     nn.Conv2d(patch_feat_dim // 2, patch_feat_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(patch_feat_dim),  # <-- Added Normalization
        #     nn.GELU()
        # )

        # Branch 2: SirenLayer-based MLP for spatial coordinates
        self.spatial_net = nn.Sequential(
            SirenLayer(2, hidden_dim, is_first=True),
            SirenLayer(hidden_dim, hidden_dim),
            SirenLayer(hidden_dim, spatial_feat_dim, is_last=True)
        )

        # Branch 3: Hierarchical Mamba U-Net for global context
        mamba_config = {'d_state': mamba_d_state, 'd_conv': mamba_d_conv, 'expand': mamba_expand}
        self.mamba_entry = nn.Sequential(nn.Conv2d(1, mamba_d_model, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(mamba_d_model), nn.GELU())
        self.encoder1 = MambaEncoderBlock(mamba_d_model, mamba_d_model * 2, mamba_config)
        self.encoder2 = MambaEncoderBlock(mamba_d_model * 2, mamba_d_model * 4, mamba_config)
        self.bottleneck = MambaBlock(d_model=mamba_d_model * 4, **mamba_config)
        self.decoder1 = MambaDecoderBlock(mamba_d_model * 4, mamba_d_model * 2, mamba_d_model * 2, mamba_config)
        self.decoder2 = MambaDecoderBlock(mamba_d_model * 2, mamba_d_model, mamba_d_model, mamba_config)
        self.mamba_exit = nn.Conv2d(mamba_d_model, mamba_feat_dim, kernel_size=1)

        # Final Fusion Network with added BatchNorm
        self.fusion_net = nn.Sequential(
            #nn.Conv2d(patch_feat_dim + spatial_feat_dim + mamba_feat_dim, hidden_dim, kernel_size=1),
            nn.Conv2d(spatial_feat_dim + mamba_feat_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),  # <-- Added Normalization
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),  # <-- Added Normalization
            nn.GELU()
        )

        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, image, coords):
        # Flatten coordinates for the spatial_net (MLP)
        coords_flat = coords.view(-1, 2)
        H, W = image.shape[-2], image.shape[-1]

        # Branch 1: Local Features
        #patch_feat_map = self.patch_net(image)

        # Branch 2: Spatial Features
        spatial_feat_flat = self.spatial_net(coords_flat)
        spatial_feat_map = spatial_feat_flat.view(H, W, -1).permute(2, 0, 1).unsqueeze(0)

        # Branch 3: Global Features
        s0 = self.mamba_entry(image)
        s1 = self.encoder1(s0)
        s2 = self.encoder2(s1)
        bottleneck_out = self.bottleneck(s2) + s2
        d1 = self.decoder1(bottleneck_out, s1)
        d2 = self.decoder2(d1, s0)
        global_feat_map = self.mamba_exit(d2)

        # Fusion
        #combined_feat_map = torch.cat((patch_feat_map, spatial_feat_map, global_feat_map), dim=1)
        combined_feat_map = torch.cat((spatial_feat_map, global_feat_map), dim=1)
        fused_map = self.fusion_net(combined_feat_map)

        output = self.output_head(fused_map)
        return output

