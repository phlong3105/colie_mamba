from utils import *
from loss import *
# Make sure to import from the correct network file name
from siren_mamba_loss import CoLIENet
from color import rgb2hsv_torch, hsv2rgb_torch
import random
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.transforms.functional import gaussian_blur

# 在 colie_mamba_loss 基础上修改
class L_texture(nn.Module):
    """
    Structure-Texture Decomposition Loss for Denoising.
    Penalizes high-frequency texture/noise in the reflectance map.
    """
    def __init__(self, kernel_size=3, sigma=1.0):
        super(L_texture, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, x):
        # Create a blurred version of the image to represent the "structure"
        structure = gaussian_blur(x, kernel_size=self.kernel_size, sigma=self.sigma)
        # The "texture" is the difference between the original and the structure
        texture = x - structure
        # Penalize the L1 norm of the texture component
        return torch.mean(torch.abs(texture))


def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser(description='CoLIE with Three Branches and Denoising Loss')
# --- Arguments ---
parser.add_argument('--input_folder', type=str, default='input/dataset/LOLdataset/eval15/low')
parser.add_argument('--output_folder', type=str, default='output/dataset/LOLdataset/eval15/mamba_high')
parser.add_argument('--down_size', type=int, default=256, help='downsampling size')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate')
parser.add_argument('--L', type=float, default=0.5, help='Target exposure level')
# --- Loss Weights ---
parser.add_argument('--alpha', type=float, default=85, help='Weight for fidelity loss (L_spa)')
parser.add_argument('--beta', type=float, default=25, help='Weight for illumination smoothness (L_tv)')
parser.add_argument('--gamma', type=float, default=43, help='Weight for exposure control (L_exp)')
parser.add_argument('--delta', type=float, default=18)
# --- Denoising Loss Weight ---
parser.add_argument('--epsilon', type=float, default=18, help='Weight for reflectance smoothness (denoising)')
# --- Mamba Arguments ---
parser.add_argument('--mamba_d_model', type=int, default=64, help='Base feature dimension for Mamba U-Net')
parser.add_argument('--mamba_d_state', type=int, default=16, help='State dimension for Mamba SSM')
parser.add_argument('--mamba_d_conv', type=int, default=4, help='Kernel size for Mamba convolution')
parser.add_argument('--mamba_expand', type=int, default=2, help='Expansion factor for Mamba block')
parser.add_argument('--seed', type=int, default=42, help='random seed')
opt = parser.parse_args()

set_seed(opt.seed)

if not os.path.exists(opt.input_folder):
    print(f'Input folder: {opt.input_folder} does not exist')
    exit()

if not os.path.exists(opt.output_folder):
    os.makedirs(opt.output_folder)

print(' > Running training...')
for PATH in tqdm(np.sort(os.listdir(opt.input_folder))):
    try:
        img_rgb = get_image(os.path.join(opt.input_folder, PATH))
        img_hsv = rgb2hsv_torch(img_rgb)

        img_v = get_v_component(img_hsv)
        img_v_lr = interpolate_image(img_v, opt.down_size, opt.down_size)

        coords = get_coords(opt.down_size, opt.down_size)

        model = CoLIENet(
            hidden_dim=256,
            mamba_d_model=opt.mamba_d_model,
            mamba_d_state=opt.mamba_d_state,
            mamba_d_conv=opt.mamba_d_conv,
            mamba_expand=opt.mamba_expand,
            down_size=opt.down_size
        )
        model.cuda()

        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=1e-6)

        l_exp = L_exp(16, opt.L)
        l_TV = L_TV()
        l_texture = L_texture()

        for epoch in range(opt.epochs):
            model.train()
            optimizer.zero_grad()

            illu_lr = model(img_v_lr, coords)

            # This is the reflectance R = V / I
            img_v_fixed_lr = (img_v_lr) / (illu_lr + 1e-4)

            # --- Loss Calculation ---
            loss_spa = torch.mean(torch.pow(illu_lr - img_v_lr, 2))
            loss_illum_tv = l_TV(illu_lr)  # Smoothness on Illumination
            loss_exp = torch.mean(l_exp(illu_lr))
            loss_sparsity = torch.mean(img_v_fixed_lr)
            loss_denoise = l_texture(img_v_fixed_lr)

            # --- New Denoising Loss Term ---
            # Enforces smoothness on the final reflectance map to reduce noise
            # loss_reflectance_tv = l_TV(img_v_fixed_lr)

            loss = (loss_spa * opt.alpha +
                    loss_illum_tv * opt.beta +
                    loss_exp * opt.gamma +
                    loss_sparsity * opt.delta +
                    loss_denoise * opt.epsilon)  # Add new loss to the total

            loss.backward()
            optimizer.step()
            scheduler.step()

        # --- Final Image Reconstruction ---
        model.eval()
        with torch.no_grad():
            final_illu_lr = model(img_v_lr, coords)
            final_v_fixed_lr = (img_v_lr) / (final_illu_lr + 1e-4)

        img_v_fixed = filter_up(img_v_lr, final_v_fixed_lr, img_v)
        img_hsv_fixed = replace_v_component(img_hsv, img_v_fixed)
        img_rgb_fixed = hsv2rgb_torch(img_hsv_fixed)
        img_rgb_fixed = torch.clamp(img_rgb_fixed, 0, 1)

        Image.fromarray(
            (torch.movedim(img_rgb_fixed, 1, -1)[0].detach().cpu().numpy() * 255).astype(np.uint8)
        ).save(os.path.join(opt.output_folder, PATH))

    except Exception as e:
        print(f"Failed to process {PATH}: {e}")

print(' > Reconstruction done')

