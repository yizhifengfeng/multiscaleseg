#!/usr/bin/env python3
"""
MS-CrackSeg 7波段精简版 (适配本地 RTX 2080 8GB + Windows)
==========================================================
基于 7波段3090版 修改, 针对 RTX 2080 (8GB显存) 做以下调整:

★ 相比3090版的参数调整 (共5处) ★

调整1  [路径]        路径改为本地 D 盘 7selectband 目录
调整2  [BASE_DIM]    32 → 16 (模型通道数减半, 参数量降为约1/4, 显存大幅降低)
调整3  [BATCH_SIZE]  16 → 4  (8GB显存安全值, 若不报OOM可试改为6或8)
调整4  [LEARNING_RATE] 1e-3 → 2e-4 (batch变小, 按线性缩放降低LR以保持稳定)
调整5  [NUM_WORKERS] 8 → 0  (Windows多进程极易报错, 设0禁用多进程, 略慢但稳定)
      [persistent_workers] True → False (Windows下必须关闭, 否则报错)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # 无GUI后端, 避免弹窗报错
import matplotlib.pyplot as plt
import json

# ================================================================
# 1. 路径与超参数
# ================================================================

# --- [调整1] 数据路径: 改为本地 D 盘 7波段数据目录 ---
#    (请确认这个路径与你预处理脚本输出的路径一致)
_DEFAULT_DATA = r"D:\deeplearning\MS-CrackSeg-main\reduction_band\7selectband"
train_img_dir  = os.path.join(_DEFAULT_DATA, "img")
train_mask_dir = os.path.join(_DEFAULT_DATA, "masknpy")

# 输出目录也改到本地
OUTPUT_ROOT  = r"D:\deeplearning\MS-CrackSeg-main\reduction_band/output_results_7band"
MODEL_DIR    = os.path.join(OUTPUT_ROOT, "models")
TRAIN_LOG    = os.path.join(OUTPUT_ROOT, "training_log_7band.json")
CHECKPOINT   = os.path.join(MODEL_DIR, "best_model.pth")

IN_CHANNELS  = 7       # 7个最优波段
EPOCHS       = 300

# --- [调整3] 批大小: 16 → 4 ---
#    RTX 2080 显存 8GB, 2D模型+混合精度, batch=4 约占 3~4GB, 安全
#    如果训练过程中没有 OOM(显存溢出), 可以尝试改为 6 或 8 加速训练
BATCH_SIZE   = 4

# --- [调整4] 学习率: 1e-3 → 2e-4 ---
#    经验规则: batch 缩小几倍, LR 也缩小几倍 (16→4 缩4倍, 1e-3→2.5e-4, 取2e-4)
LEARNING_RATE = 2e-4

WEIGHT_DECAY = 1e-4
PATIENCE     = 50

# --- [调整5] 工作线程: 8 → 0 ---
#    Windows 系统下 DataLoader 的多进程 (num_workers>0) 经常因
#    pickle 序列化问题报错, 设为 0 表示主进程加载数据, 速度略慢但不会崩
NUM_WORKERS  = 0

VIZ_BAND_IDX = 3       # 可视化显示第4个波段 (577.8nm, 黄绿光)
PRED_THRESHOLD = 0.22
VIZ_EVERY    = 30
VIZ_SAMPLE_INDICES = [0, 1, 2]
GRAD_CLIP    = 1.0
AUG_PROB     = 0.5
SEED         = 42

POS_WEIGHT   = 80.0
DICE_WEIGHT  = 0.35
BCE_WEIGHT   = 0.35
FOCAL_WEIGHT = 0.30
FOCAL_GAMMA  = 2.5

# --- [调整2] 模型基础通道数: 32 → 16 ---
#    这是控制模型大小最关键的参数:
#    BASE_DIM=32 → 参数量约 40万, 显存约 5~6GB (batch=4时可能OOM)
#    BASE_DIM=16 → 参数量约 10万, 显存约 2~3GB (batch=4 稳定运行)
#    如果 batch=4 训练正常且显存有富余, 可以改回 32 提升精度
BASE_DIM = 16


def set_seed(seed=SEED):
    """固定随机种子, 保证每次运行结果一致"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================================================
# 2. 多尺度空间卷积(MSSK)模块 (2D版本)
# ================================================================
class MSSK(nn.Module):
    """
    多尺度空间卷积模块
    并行 1×1, 3×3, 5×5, 7×7 四个分支 → 拼接 → 1×1融合 → 残差连接
    专门针对细长裂缝设计, 不同尺度卷积捕获不同粗细的裂缝特征
    """
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        n = out_channels // 4  # 每个分支分到的通道数

        # 四个尺度的卷积分支
        self.conv1x1 = nn.Conv2d(in_channels, n, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, n, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, n, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, n, kernel_size=7, padding=3)

        # 融合层: 把4个分支的输出拼起来 (共n*4通道) → 压缩回 out_channels
        self.fusion = nn.Conv2d(n * 4, out_channels, kernel_size=1)
        self.bn     = nn.BatchNorm2d(out_channels)
        self.relu   = nn.ReLU(inplace=True)

        # 残差连接: 如果输入输出通道数不同, 用1×1卷积对齐
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x1 = self.conv1x1(x)   # 捕获点状特征
        x3 = self.conv3x3(x)   # 捕获细裂缝
        x5 = self.conv5x5(x)   # 捕获中等裂缝
        x7 = self.conv7x7(x)   # 捕获粗裂缝+上下文
        out = torch.cat([x1, x3, x5, x7], dim=1)  # 拼接: (B, n*4, H, W)
        out = self.fusion(out)   # 融合: (B, out_channels, H, W)
        out = self.bn(out)
        out += residual         # 残差连接: 帮助梯度稳定回传
        out = self.relu(out)
        return out


# ================================================================
# 3. 空洞注意力模块 (2D版本)
# ================================================================
class DilateAttention(nn.Module):
    """
    空洞注意力模块
    用3种不同膨胀率(dilation=1,2,4)的空洞卷积构造 V/Q/K
    膨胀率越大 → 感受野越大 → 能看到更远的上下文信息
    对裂缝分割很重要: 裂缝周围纹理是判断"这是不是裂缝"的关键线索
    """
    def __init__(self, in_channel, depth):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU()
        # V: 标准卷积, 提取特征值
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        # Q: 膨胀率2, 感受野 5×5
        self.atrous_block2 = nn.Conv2d(in_channel, depth, 3, 1,
                                       padding=2, dilation=2, groups=in_channel)
        # K: 膨胀率4, 感受野 9×9
        self.atrous_block3 = nn.Conv2d(in_channel, depth, 3, 1,
                                       padding=4, dilation=4, groups=in_channel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        v = self.atrous_block1(x)
        q = self.atrous_block2(x)
        k = self.atrous_block3(x)
        temp = k * q           # 注意力权重图
        output = v * self.softmax(temp)  # 加权特征
        output += self.relu(self.batch_norm(x))  # 残差连接
        return output


# ================================================================
# 4. 主模型: 2D编码器-解码器
# ================================================================
class CrackSeg7Band(nn.Module):
    """
    2D U-Net 风格编码器-解码器, 适配7波段多光谱输入

    结构示意 (BASE_DIM=16):
      输入: (B, 7, 224, 224)
        ↓ enc_conv1(7→16) + pool
      e1: (B, 16, 112, 112) ──────────────────┐
        ↓ enc_conv2 + pool                      │ 跳跃连接
      e2: (B, 16, 56, 56) ───────────────┐     │
        ↓ enc_conv3 + pool                │     │
      e3: (B, 16, 28, 28) ──────┐        │     │
        ↓ bottleneck              │        │     │
      b:  (B, 16, 28, 28)       │        │     │
        ↓ up3 + cat(e3) + MSSK  │        │     │
      d3: (B, 16, 56, 56) ←────┘        │     │
        ↓ up2 + cat(e2) + MSSK           │     │
      d2: (B, 16, 112, 112) ←────────────┘     │
        ↓ up1 + conv                         │
      d1: (B, 16, 224, 224) ←───────────────┘
        ↓ final(16→1)
      输出: (B, 1, 224, 224)
    """
    def __init__(self, in_channels=7, classnum=1, base_dim=16):
        super().__init__()
        d = base_dim

        # ========== 编码器 (逐步下采样, 提取多级特征) ==========
        self.enc_conv1 = nn.Conv2d(in_channels, d, kernel_size=7, padding=3)
        self.enc_bn1   = nn.BatchNorm2d(d)
        self.pool1     = nn.MaxPool2d(kernel_size=2, stride=2)
        self.da1       = DilateAttention(d, d)

        self.enc_conv2 = nn.Conv2d(d, d, kernel_size=7, padding=3)
        self.enc_bn2   = nn.BatchNorm2d(d)
        self.pool2     = nn.MaxPool2d(kernel_size=2, stride=2)
        self.da2       = DilateAttention(d, d)

        self.enc_conv3 = nn.Conv2d(d, d, kernel_size=7, padding=3)
        self.enc_bn3   = nn.BatchNorm2d(d)
        self.pool3     = nn.MaxPool2d(kernel_size=2, stride=2)
        self.da3       = DilateAttention(d, d)

        # ========== 瓶颈层 (最深层特征) ==========
        self.bottleneck = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=7, padding=3),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )

        # ========== 解码器 (逐步上采样, 恢复分辨率) ==========
        self.up3       = nn.ConvTranspose2d(d, d, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Conv2d(d, d, kernel_size=7, padding=3)
        self.mssk1     = MSSK(d * 2, d)   # cat后2d通道 → MSSK融合回d通道

        self.up2       = nn.ConvTranspose2d(d, d, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv2d(d, d, kernel_size=7, padding=3)
        self.mssk2     = MSSK(d * 2, d)

        self.up1       = nn.ConvTranspose2d(d, d, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(d, d, kernel_size=7, padding=3)

        # ========== 输出层: d通道 → 1通道 (裂缝概率图) ==========
        self.final = nn.Conv2d(d, classnum, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # ---- 编码器: 逐步缩小分辨率, 提取深层语义特征 ----
        e1 = self.relu(self.enc_bn1(self.enc_conv1(x)))
        e1 = self.pool1(e1)    # 224→112
        e1 = self.da1(e1)

        e2 = self.relu(self.enc_bn2(self.enc_conv2(e1)))
        e2 = self.pool2(e2)    # 112→56
        e2 = self.da2(e2)

        e3 = self.relu(self.enc_bn3(self.enc_conv3(e2)))
        e3 = self.pool3(e3)    # 56→28
        e3 = self.da3(e3)

        # ---- 瓶颈 ----
        b = self.bottleneck(e3)

        # ---- 解码器: 逐步恢复分辨率, 跳跃连接补充细节 ----
        d3 = self.up3(b)                                   # 28→56
        d3 = self.dec_conv3(d3)
        d3 = torch.cat([d3, e3], dim=1)                    # 跳跃拼接
        d3 = self.mssk1(d3)                                # MSSK融合

        d2 = self.up2(d3)                                  # 56→112
        d2 = self.dec_conv2(d2)
        d2 = torch.cat([d2, e2], dim=1)                    # 跳跃拼接
        d2 = self.mssk2(d2)                                # MSSK融合

        d1 = self.up1(d2)                                  # 112→224
        d1 = self.relu(self.dec_conv1(d1))

        out = self.final(d1)                               # → (B, 1, 224, 224)
        return out


# ================================================================
# 5. 数据集
# ================================================================
class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npy")])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".npy")])

        # 通过文件名中的关键标识 (如 r1c1col51row39overlap60) 匹配img和mask
        self.file_mapping = {}
        for img_file in self.img_files:
            key = self._extract_key(img_file)
            self.file_mapping[key] = img_file

        missing = [f for f in self.mask_files
                   if self._extract_key(f) not in self.file_mapping]
        if missing:
            print(f"警告: {len(missing)} 个mask找不到对应img")

        assert len(self.file_mapping) > 0, "没有匹配的img/mask文件!"
        self.valid_keys = list(self.file_mapping.keys())
        print(f"找到 {len(self.valid_keys)} 对文件")

    def _extract_key(self, filename):
        filename = filename.replace('.npy', '')
        for part in filename.split('_'):
            if 'r' in part and 'c' in part and 'col' in part and 'row' in part:
                return part
        return filename

    def __len__(self):
        return len(self.valid_keys)

    def __getitem__(self, idx):
        key = self.valid_keys[idx]
        img_file  = self.file_mapping[key]
        mask_file = f"all_bands_mosaic_crack46all_classic_mask_classic_{key}.npy"

        img_path  = os.path.join(self.img_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        img  = np.load(img_path)    # (7, 224, 224) float
        mask = np.load(mask_path)   # (224, 224) 或 (1, 224, 224) int

        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        # 逐样本归一化到 [0, 1]
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 1e-8:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        # 数据增强 (对所有7个波段同步操作, 保持波段间对应关系)
        if self.augment:
            if random.random() > AUG_PROB:          # 水平翻转
                img = img[:, :, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if random.random() > AUG_PROB:          # 垂直翻转
                img = img[:, ::-1, :].copy()
                mask = mask[::-1, :].copy()
            if random.random() > AUG_PROB:          # 旋转90/180/270°
                k = random.choice([1, 2, 3])
                img  = np.rot90(img,  k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()

        img = img.astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(mask.astype(np.int64)), img_file


# ================================================================
# 6. 损失函数 (Dice + BCE + Focal 三重损失)
# ================================================================
class DiceBCEFocalLoss(nn.Module):
    """
    三重损失组合:
    - BCE (加权): 基础分类损失, pos_weight=80 解决正负样本极度不平衡
    - Dice: 直接优化分割区域重叠度, 对小目标(裂缝)更敏感
    - Focal: 聚焦难分样本, gamma=2.5 让模型更关注分类错误的像素
    """
    def __init__(self, pos_weight=80.0, dice_weight=0.35,
                 bce_weight=0.35, focal_weight=0.30, focal_gamma=2.5):
        super().__init__()
        self.dice_weight  = dice_weight
        self.bce_weight   = bce_weight
        self.focal_weight = focal_weight
        self.focal_gamma  = focal_gamma
        self.register_buffer('pos_weight_tensor', torch.tensor([pos_weight]))

    def forward(self, logits, target):
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        logits_flat = logits.view(-1)
        target_flat = target.view(-1).float()
        probs = torch.sigmoid(logits_flat)

        # 1. 加权BCE
        bce = F.binary_cross_entropy_with_logits(
            logits_flat, target_flat,
            pos_weight=self.pos_weight_tensor.to(logits.device))

        # 2. Dice
        intersection = (probs * target_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + 1e-7) / \
                    (probs.sum() + target_flat.sum() + 1e-7)

        # 3. Focal
        p_clamped = torch.clamp(probs, min=1e-7, max=1 - 1e-7)
        p_t = p_clamped * target_flat + (1 - p_clamped) * (1 - target_flat)
        focal_w = (1 - p_t) ** self.focal_gamma
        focal_loss = (focal_w * F.binary_cross_entropy_with_logits(
            logits_flat, target_flat, reduction='none')).mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss + \
               self.focal_weight * focal_loss


# ================================================================
# 7. 评估与可视化
# ================================================================
@torch.no_grad()
def compute_metrics(model, loader, criterion, device, threshold=PRED_THRESHOLD):
    """计算 IoU, F1, Precision, Recall 四项指标"""
    model.eval()
    total_loss, tp, fp, fn = 0.0, 0.0, 0.0, 0.0

    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        with torch.amp.autocast('cuda'):
            out = model(imgs)
            loss = criterion(out, masks)
        total_loss += loss.item()

        # 注意: squeeze(1) 去掉通道维, 避免 (B,1,H,W) & (B,H,W) 的广播错误
        pred = (torch.sigmoid(out) > threshold).long().squeeze(1)

        tp += ((pred == 1) & (masks == 1)).sum().float()
        fp += ((pred == 1) & (masks == 0)).sum().float()
        fn += ((pred == 0) & (masks == 1)).sum().float()

    eps = 1e-7
    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = (tp + eps) / (tp + fp + fn + eps)

    return total_loss / max(len(loader), 1), {
        'Precision': precision.item(),
        'Recall':    recall.item(),
        'F1':        f1.item(),
        'IoU':       iou.item()
    }


def save_prediction_figure(model, dataset, device, save_path,
                           epoch, indices, band_idx, threshold):
    """保存预测可视化图: 输入波段 | 真实掩码 | 预测结果"""
    model.eval()
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    band_names = ['397.5nm', '456.4nm', '532.7nm', '577.8nm',
                  '903.9nm', '935.1nm', '942.0nm']

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, mask, _ = dataset[idx]
            inp = img.unsqueeze(0).to(device)
            out = model(inp)
            pred = (torch.sigmoid(out).squeeze().cpu().numpy() > threshold) \
                   .astype(np.float32)

            b = min(band_idx, img.shape[0] - 1)
            axes[i, 0].imshow(img[b].numpy(), cmap='gray')
            axes[i, 0].set_title(f'Band {b} ({band_names[b]})')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask.numpy(), cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

    plt.suptitle(f'Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


# ================================================================
# 8. 训练日志
# ================================================================
class TrainingLogger:
    """记录训练过程指标, 支持断点续训时自动读取历史记录"""
    def __init__(self, path):
        self.path = path
        self.h = {'train_loss': [], 'val_loss': [], 'iou': [], 'f1': [],
                  'precision': [], 'recall': [], 'lr': []}
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.h = json.load(f)

    def log(self, ep, tl, vl, m, lr):
        self.h['train_loss'].append(tl)
        self.h['val_loss'].append(vl)
        self.h['iou'].append(m['IoU'])
        self.h['f1'].append(m['F1'])
        self.h['precision'].append(m['Precision'])
        self.h['recall'].append(m['Recall'])
        self.h['lr'].append(lr)
        with open(self.path, 'w') as f:
            json.dump(self.h, f, indent=2)

    def plot(self, path):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].plot(self.h['train_loss'], label='Train')
        axes[0, 0].plot(self.h['val_loss'], label='Val')
        axes[0, 0].set_title('Loss'); axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.h['iou'], label='IoU')
        axes[0, 1].plot(self.h['f1'], label='F1')
        axes[0, 1].set_title('Metrics'); axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(self.h['precision'], label='Precision')
        axes[1, 0].plot(self.h['recall'], label='Recall')
        axes[1, 0].set_title('PR Curve'); axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(self.h['lr'], label='LR', color='red')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_yscale('log'); axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)


# ================================================================
# 9. 主训练循环
# ================================================================
def main():
    set_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---- 数据集划分 (7:3 训练:验证) ----
    full_dataset = CrackDataset(train_img_dir, train_mask_dir, augment=False)
    train_size = int(0.7 * len(full_dataset))
    train_indices, val_indices = random_split(
        range(len(full_dataset)),
        [train_size, len(full_dataset) - train_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # 训练集开启数据增强, 验证集不增强
    train_ds = torch.utils.data.Subset(
        CrackDataset(train_img_dir, train_mask_dir, augment=True),
        train_indices.indices)
    val_ds = torch.utils.data.Subset(full_dataset, val_indices.indices)

    # --- [调整5] persistent_workers=False (Windows必须) ---
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=False)

    # ---- 模型 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrackSeg7Band(in_channels=IN_CHANNELS, classnum=1,
                          base_dim=BASE_DIM).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    print(f"设备: {device}")
    print(f"模型参数量: {total_params:,} ({train_size_mb:.1f} MB)")
    print(f"BASE_DIM={BASE_DIM}, BATCH_SIZE={BATCH_SIZE}, IN_CHANNELS={IN_CHANNELS}")

    # ---- 日志与断点续训 ----
    logger = TrainingLogger(TRAIN_LOG)
    start_epoch = len(logger.h['train_loss']) if logger.h['train_loss'] else 0
    if start_epoch > 0:
        print(f"检测到历史记录, 从第 {start_epoch + 1} 轮继续训练")

    # ---- 损失函数 ----
    criterion = DiceBCEFocalLoss(pos_weight=POS_WEIGHT).to(device)

    # ---- 优化器 + 学习率调度 ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15,
        verbose=True, min_lr=1e-7)

    # 混合精度训练 (RTX 2080 支持FP16, 约省一半显存, 训练更快)
    scaler = torch.amp.GradScaler('cuda')

    trigger = 0
    best_iou = max(logger.h['iou']) if logger.h['iou'] else 0.0

    # ===================== 训练循环 =====================
    for epoch in range(start_epoch, EPOCHS):
        ep = epoch + 1
        model.train()
        t_loss, n_b = 0.0, 0

        for imgs, masks, _ in tqdm(train_loader, desc=f"Epoch {ep:03d}"):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad(set_to_none=True)

            # 混合精度前向传播
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                loss = criterion(out, masks)

            # 反向传播 + 梯度裁剪
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item()
            n_b += 1

        # ---- 验证 ----
        v_loss, metrics = compute_metrics(
            model, val_loader, criterion, device, PRED_THRESHOLD)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step(metrics['IoU'])

        # ---- 保存最优模型 ----
        msg = ""
        if metrics['IoU'] > best_iou:
            best_iou = metrics['IoU']
            trigger = 0
            msg = " >>> Best IoU!"
            torch.save(model.state_dict(), CHECKPOINT)
        else:
            trigger += 1
            msg = f" Wait({trigger}/{PATIENCE})"

        logger.log(ep, t_loss / n_b, v_loss, metrics, lr)

        print(f"Ep{ep:03d} | Tr:{t_loss/n_b:.4f} Va:{v_loss:.4f} | "
              f"IoU:{metrics['IoU']:.4f} F1:{metrics['F1']:.4f} | "
              f"Prec:{metrics['Precision']:.4f} Rec:{metrics['Recall']:.4f} | "
              f"LR:{lr:.2e}{msg}")

        # ---- 可视化 (每30轮保存一次预测图) ----
        if ep % VIZ_EVERY == 0:
            save_prediction_figure(
                model, val_ds, device,
                os.path.join(OUTPUT_ROOT, f"viz_ep{ep:03d}.png"),
                ep, VIZ_SAMPLE_INDICES, VIZ_BAND_IDX, PRED_THRESHOLD)

        # ---- 早停 ----
        if trigger >= PATIENCE:
            print(f"早停: 连续 {PATIENCE} 轮无改善, 训练结束")
            break

    # ---- 训练结束, 绘制曲线 ----
    logger.plot(os.path.join(OUTPUT_ROOT, "training_curves_7band.png"))
    print(f"\n{'='*50}")
    print(f"训练完成! 最佳 IoU: {best_iou:.4f}")
    print(f"模型保存: {CHECKPOINT}")
    print(f"训练曲线: {OUTPUT_ROOT}\\training_curves_7band.png")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
