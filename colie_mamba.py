from utils import *
from loss import *
from siren_mamba import INF_MAMBA  # 导入修改后的INF_MAMBA
from color import rgb2hsv_torch, hsv2rgb_torch
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser(description='CoLIE with INF_MAMBA (Mamba + UNet)')
parser.add_argument('--input_folder', type=str, default='input/dataset/LOLdataset/eval15/low')
parser.add_argument('--output_folder', type=str, default='output/dataset/LOLdataset/eval15/mamba_high_v2')
parser.add_argument('--down_size', type=int, default=256, help='downsampling size for training')
parser.add_argument('--epochs', type=int, default=1000, help='training epochs per image')
# Mamba相关参数
parser.add_argument('--base_channels', type=int, default=32, help='UNet base channels')
parser.add_argument('--d_state', type=int, default=16, help='Mamba d_state')
parser.add_argument('--d_conv', type=int, default=4, help='Mamba d_conv')
parser.add_argument('--expand', type=int, default=2, help='Mamba expand ratio')
# 损失函数权重
parser.add_argument('--alpha', type=float, required=True, help='weight for sparsity loss')
parser.add_argument('--beta', type=float, required=True, help='weight for TV loss')
parser.add_argument('--gamma', type=float, required=True, help='weight for exposure loss')
parser.add_argument('--delta', type=float, required=True, help='weight for fidelity loss')
parser.add_argument('--L', type=float, default=0.5)
# 训练参数
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')

opt = parser.parse_args()

# 检查文件夹
if not os.path.exists(opt.input_folder):
    print(f'Error: Input folder {opt.input_folder} does not exist')
    exit()
if not os.path.exists(opt.output_folder):
    os.makedirs(opt.output_folder)
    print(f'Created output folder {opt.output_folder}')

print('> Starting CoLIE with INF_MAMBA')
print(f'  Downsize: {opt.down_size}, Epochs: {opt.epochs}, LR: {opt.lr}')

# 遍历所有输入图像
for img_name in tqdm(np.sort(os.listdir(opt.input_folder))):
    img_path = os.path.join(opt.input_folder, img_name)
    # -------------------------- 1. 读取图像并转换为HSV --------------------------
    img_rgb = get_image(img_path)  # (1,3,H,W), 归一化到[0,1]
    img_hsv = rgb2hsv_torch(img_rgb)  # (1,3,H,W)
    img_v = get_v_component(img_hsv)  # (1,1,H,W) → 提取V分量
    H_org, W_org = img_v.shape[2], img_v.shape[3]  # 原始分辨率

    # -------------------------- 2. 下采样V分量（降低计算量） --------------------------
    img_v_lr = interpolate_image(img_v, opt.down_size, opt.down_size)  # (1,1,down_size,down_size)
    H_lr, W_lr = img_v_lr.shape[2], img_v_lr.shape[3]

    # -------------------------- 3. 初始化INF_MAMBA模型 --------------------------
    model = INF_MAMBA(
        in_channels=3,  # 1(V) + 2(坐标)
        base_channels=opt.base_channels,
        d_state=opt.d_state,
        d_conv=opt.d_conv,
        expand=opt.expand
    ).cuda()  # 移动到GPU

    # -------------------------- 4. 初始化优化器和损失函数 --------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt.lr,
        betas=(0.9, 0.999),
        weight_decay=opt.weight_decay
    )
    # 损失函数实例化
    l_exp = L_exp(patch_size=16, mean_val=opt.L)  # 曝光损失（patch_size=16，L=0.5默认）
    l_TV = L_TV()  # 平滑损失

    # -------------------------- 5. 单图像训练（Zero-shot） --------------------------
    model.train()
    for epoch in range(opt.epochs):
        optimizer.zero_grad()

        # 前向传播：预测光照残差
        illu_res_lr = model(img_v_lr)  # (1,1,H_lr,W_lr) → 光照残差
        illu_lr = illu_res_lr + img_v_lr  # 预测光照 = V分量 + 残差
        img_v_fixed_lr = img_v_lr / (illu_lr + 1e-4)  # 增强后的V分量（避免除零）

        # 计算各损失项
        loss_fidelity = F.mse_loss(illu_lr, img_v_lr)  # 保真度损失（MSE）
        loss_tv = l_TV(illu_lr)  # 平滑损失（TV）
        loss_exp = l_exp(illu_lr)  # 曝光损失
        loss_sparsity = torch.mean(img_v_fixed_lr)  # 稀疏性损失

        # 总损失
        total_loss = (
            opt.delta * loss_fidelity +  # 原alpha→delta（对应论文L_f）
            opt.beta * loss_tv +         # 原beta→beta（L_s）
            opt.gamma * loss_exp +       # 原gamma→gamma（L_exp）
            opt.alpha * loss_sparsity    # 原delta→alpha（L_spa）
        )

        # 反向传播与优化
        total_loss.backward()
        optimizer.step()

        # 每20轮打印一次损失（可选）
        if (epoch + 1) % 20 == 0:
            tqdm.write(f'  Image: {img_name} | Epoch {epoch+1}/{opt.epochs} | Loss: {total_loss.item():.4f}')

    # -------------------------- 6. 上采样恢复原始分辨率 --------------------------
    # 使用引导滤波上采样增强后的V分量
    img_v_fixed = filter_up(
        x_lr=img_v_lr,    # 低分辨率V分量（引导图）
        y_lr=img_v_fixed_lr,  # 低分辨率增强V分量（待上采样图）
        x_hr=img_v,       # 原始分辨率V分量（目标分辨率）
        r=1               # 引导滤波半径
    )

    # -------------------------- 7. 重构HSV并转换为RGB --------------------------
    img_hsv_fixed = replace_v_component(img_hsv, img_v_fixed)  # 替换V分量
    img_rgb_fixed = hsv2rgb_torch(img_hsv_fixed)  # HSV→RGB
    img_rgb_fixed = img_rgb_fixed / torch.max(img_rgb_fixed)  # 归一化到[0,1]

    # -------------------------- 8. 保存结果 --------------------------
    # 转换为PIL图像格式
    img_np = (torch.movedim(img_rgb_fixed, 1, -1)[0].detach().cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    save_path = os.path.join(opt.output_folder, img_name)
    img_pil.save(save_path)
    tqdm.write(f'  Saved enhanced image to {save_path}')

print('> All images processed successfully!')