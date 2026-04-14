#!/usr/bin/env python3
"""
MS-CrackSeg: Multiscale Enhanced Pavement Crack Segmentation Network
====================================================================
Chen et al. (2024), IJAG 128, 103772
带预加载数据到内存和增大Batch Size以提升训练速度
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# ================================================================
# 1. 路径与超参数 (增大Batch Size和调整学习率)
# ================================================================
_DEFAULT_DATA = "/kaggle/input/datasets/zhifengfengyi/gaoguangpu"
train_img_dir = os.path.join(_DEFAULT_DATA, "img")
train_mask_dir = os.path.join(_DEFAULT_DATA, "masknpy")
OUTPUT_ROOT = "/kaggle/working/output_results"
MODEL_DIR = os.path.join(OUTPUT_ROOT, "models")
TRAIN_LOG = os.path.join(OUTPUT_ROOT, "training_log.json")
CHECKPOINT = os.path.join(MODEL_DIR, "best_model.pth")

IN_CHANNELS = 1
BASE_DIM = 16
EPOCHS = 800
BATCH_SIZE = 8  # ★ 增大到8 (原4)
LEARNING_RATE = 2e-4  # ★ Batch Size增大，学习率按比例调整 (原1e-4)
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 15
PATIENCE = 60
NUM_WORKERS = 4  # 数据加载线程数
SLICE_IDX = 88
PRED_THRESHOLD = 0.5
VIZ_EVERY = 20
VIZ_SAMPLE_INDICES = [0, 1, 2]
GRAD_CLIP = 1.0
AUG_PROB = 0.5
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优


# ================================================================
# 2. MSSA 模块
# ================================================================
class MSSA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduction = max(channels // 4, 4)

        self.bn = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.proj_v = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.proj_q = nn.Conv3d(channels, channels, kernel_size=3,
                                padding=2, dilation=2, bias=False)
        self.proj_k = nn.Conv3d(channels, channels, kernel_size=3,
                                padding=4, dilation=4, bias=False)

        self.channel_attn = nn.Sequential(
            nn.Linear(channels * 2, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        x = self.relu(self.bn(x))

        v = self.proj_v(x)
        q = self.proj_q(x)
        k = self.proj_k(x)

        B, C = q.shape[:2]
        q_pool = F.adaptive_avg_pool3d(q, 1).view(B, C)
        k_pool = F.adaptive_avg_pool3d(k, 1).view(B, C)
        attn = self.channel_attn(torch.cat([q_pool, k_pool], dim=1))
        attn = attn.view(B, C, 1, 1, 1)

        return v * attn + identity


# ================================================================
# 3. MS-CrackSeg 主模型
# ================================================================
class MSCrackSeg(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_dim=16):
        super().__init__()
        C = base_dim

        self.enc1_conv = nn.Conv3d(in_channels, C, kernel_size=7, padding=3)
        self.enc1_mssa = MSSA(C)
        self.enc1_pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2_conv = nn.Conv3d(C, C, kernel_size=7, padding=3)
        self.enc2_mssa = MSSA(C)
        self.enc2_pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3_conv = nn.Conv3d(C, C, kernel_size=7, padding=3)
        self.enc3_mssa = MSSA(C)
        self.enc3_pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.dec1_up = nn.ConvTranspose3d(C, C, kernel_size=2, stride=2)
        self.dec1_conv = nn.Sequential(
            nn.Conv3d(C * 2, C, kernel_size=7, padding=3),
            nn.BatchNorm3d(C),
            nn.ReLU(inplace=True),
        )

        self.dec2_up = nn.ConvTranspose3d(C, C, kernel_size=2, stride=2)
        self.dec2_conv = nn.Sequential(
            nn.Conv3d(C * 2, C, kernel_size=7, padding=3),
            nn.BatchNorm3d(C),
            nn.ReLU(inplace=True),
        )

        self.final_up = nn.ConvTranspose3d(C, C, kernel_size=2, stride=2)
        self.final_reduce = nn.Sequential(
            nn.Conv3d(C, C, kernel_size=7, padding=3),
            nn.BatchNorm3d(C),
            nn.ReLU(inplace=True),
            nn.Conv3d(C, 1, kernel_size=7, padding=3),
        )

        self.head = nn.Conv2d(176, num_classes, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                        nonlinearity='relu')

    def forward(self, x):
        e1 = self.enc1_conv(x)
        e1 = self.enc1_mssa(e1)
        skip1 = self.enc1_pool(e1)

        e2 = self.enc2_conv(skip1)
        e2 = self.enc2_mssa(e2)
        skip2 = self.enc2_pool(e2)

        e3 = self.enc3_conv(skip2)
        e3 = self.enc3_mssa(e3)
        bottleneck = self.enc3_pool(e3)

        d1 = self.dec1_up(bottleneck)
        d1 = torch.cat([d1, skip2], dim=1)
        d1 = self.dec1_conv(d1)

        d2 = self.dec2_up(d1)
        d2 = torch.cat([d2, skip1], dim=1)
        d2 = self.dec2_conv(d2)

        d3 = self.final_up(d2)
        d3 = self.final_reduce(d3)
        d3 = d3.squeeze(1)
        out = self.head(d3)
        return out


# ================================================================
# 4. 数据集 (预加载数据到内存)
# ================================================================
class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.img_files = sorted([f for f in os.listdir(img_dir)
                                 if f.endswith(".npy") and not f.startswith(".")])
        self.mask_files = sorted([f for f in os.listdir(mask_dir)
                                  if f.endswith(".npy") and not f.startswith(".")])

        # ★ 预加载所有数据到内存 (减少I/O时间)
        self.images = []
        self.masks = []
        print("Preloading dataset to RAM...")
        for img_file, mask_file in zip(self.img_files, self.mask_files):
            img = np.load(os.path.join(img_dir, img_file))
            mask = np.load(os.path.join(mask_dir, mask_file))

            if img.ndim == 3:
                if img.shape[0] == 176:
                    pass
                elif img.shape[2] == 176:
                    img = img.transpose(2, 0, 1)
                elif 176 in img.shape:
                    axis = list(img.shape).index(176)
                    axes = [a for a in range(3) if a != axis]
                    img = img.transpose(axis, *axes)
                else:
                    raise ValueError(f"Cannot find 176 in shape {img.shape}")

            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)

            self.images.append(img)
            self.masks.append(mask)
        print(f"Dataset preloaded: {len(self.images)} samples")

    def __len__(self):
        return len(self.images)

    def _augment(self, img, mask):
        if random.random() > AUG_PROB:
            img = img[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if random.random() > AUG_PROB:
            img = img[:, ::-1, :].copy()
            mask = mask[::-1, :].copy()
        if random.random() > AUG_PROB:
            k = random.choice([1, 2, 3])
            img = np.rot90(img, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        return img, mask

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        if self.augment:
            img, mask = self._augment(img, mask)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        mask = mask.astype(np.int64)
        return torch.from_numpy(img), torch.from_numpy(mask), self.img_files[idx]


# ================================================================
# 5. 损失函数
# ================================================================
class PaperDiceLoss(nn.Module):
    def __init__(self, eps=1.0, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.eps = eps
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        p = probs.view(-1)
        t = target.view(-1).float()

        bce = F.binary_cross_entropy_with_logits(logits.view(-1), t)

        fg = (p * t).sum() + self.eps
        fg_d = fg / ((p + t).sum() + self.eps)
        bg = ((1 - p) * (1 - t)).sum() + self.eps
        bg_d = bg / ((2 - p - t).sum() + self.eps)

        return self.bce_w * bce + self.dice_w * (1.0 - fg_d - bg_d)


# ================================================================
# 6. 评估指标
# ================================================================
@torch.no_grad()
def compute_metrics(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0.0

    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.amp.autocast('cuda'):
            out = model(imgs)
            loss = criterion(out, masks)
        total_loss += loss.item()
        pred = (torch.sigmoid(out) > threshold).long()
        tp += ((pred == 1) & (masks == 1)).sum().float()
        fp += ((pred == 1) & (masks == 0)).sum().float()
        fn += ((pred == 0) & (masks == 1)).sum().float()
        tn += ((pred == 0) & (masks == 0)).sum().float()

    eps = 1e-7
    pr = (tp + eps) / (tp + fp + eps)
    re = (tp + eps) / (tp + fn + eps)
    f1 = 2 * pr * re / (pr + re + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    n = max(len(loader), 1)
    return total_loss / n, {'Precision': pr.item(), 'Recall': re.item(),
                            'F1': f1.item(), 'IoU': iou.item()}


# ================================================================
# 7. 可视化
# ================================================================
def save_prediction_figure(model, dataset, device, save_path,
                           epoch, indices, slice_idx, threshold):
    model.eval()
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, mask, _ = dataset[idx]
            inp = img.unsqueeze(0).to(device)
            with torch.amp.autocast('cuda'):
                out = model(inp)
            pred = (torch.sigmoid(out).squeeze().cpu().numpy() > threshold).astype(np.float32)
            s = min(slice_idx, img.shape[1] - 1)
            axes[i, 0].imshow(img[0, s].numpy(), cmap='gray')
            axes[i, 0].set_title(f'Band {s}')
            axes[i, 1].imshow(mask.numpy(), cmap='gray')
            axes[i, 1].set_title('GT')
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Pred')
            for ax in axes[i]:
                ax.axis('off')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ================================================================
# 8. 训练日志
# ================================================================
class TrainingLogger:
    def __init__(self, path):
        self.path = path
        self.h = {'train_loss': [], 'val_loss': [], 'iou': [], 'f1': [],
                  'precision': [], 'recall': []}
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.h = json.load(f)

    def log(self, ep, tl, vl, m, lr, msg=''):
        self.h['train_loss'].append(tl)
        self.h['val_loss'].append(vl)
        self.h['iou'].append(m['IoU'])
        self.h['f1'].append(m['F1'])
        self.h['precision'].append(m['Precision'])
        self.h['recall'].append(m['Recall'])
        with open(self.path, 'w') as f:
            json.dump(self.h, f, indent=2)

    def plot(self, path):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(self.h['train_loss'], label='Train')
        axes[0].plot(self.h['val_loss'], label='Val')
        axes[0].set_title('Loss');
        axes[0].legend()
        axes[1].plot(self.h['iou'], label='IoU', color='green')
        axes[1].plot(self.h['f1'], label='F1', color='orange')
        axes[1].set_title('IoU & F1');
        axes[1].legend()
        axes[2].plot(self.h['precision'], label='Precision')
        axes[2].plot(self.h['recall'], label='Recall')
        axes[2].set_title('P & R');
        axes[2].legend()
        plt.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


# ================================================================
# 9. 主训练循环
# ================================================================
def build_scheduler(optimizer, epochs, warmup):
    main = CosineAnnealingLR(optimizer, T_max=epochs - warmup)
    warm = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup)
    return SequentialLR(optimizer, [warm, main], milestones=[warmup])


class AugmentedSubset(Dataset):
    def __init__(self, subset, base_dataset):
        self.subset = subset
        self.base = base_dataset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        real_idx = self.subset.indices[idx]
        img, mask = self.base.images[real_idx], self.base.masks[real_idx]
        img, mask = self.base._augment(img, mask)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        mask = mask.astype(np.int64)
        return torch.from_numpy(img), torch.from_numpy(mask), self.base.img_files[real_idx]


def main():
    set_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("=" * 60)
    print("MS-CrackSeg Training (Paper-Faithful Version)")
    print("=" * 60)

    full_dataset = CrackDataset(train_img_dir, train_mask_dir, augment=False)
    print(f"Total samples: {len(full_dataset)}")

    train_size = int(0.8 * len(full_dataset))
    train_ds, val_ds = random_split(full_dataset,
                                    [train_size, len(full_dataset) - train_size],
                                    generator=torch.Generator().manual_seed(SEED))
    train_ds = AugmentedSubset(train_ds, full_dataset)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ★ 使用pin_memory=True和persistent_workers=True加速数据加载
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSCrackSeg(in_channels=IN_CHANNELS, num_classes=1, base_dim=BASE_DIM)

    if torch.cuda.device_count() > 1:
        print(f"DataParallel: {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    # 检查训练日志，确定起始epoch
    logger = TrainingLogger(TRAIN_LOG)
    start_epoch = 0
    if logger.h and len(logger.h['train_loss']) > 0:
        start_epoch = len(logger.h['train_loss'])
        print(f"Resuming from epoch {start_epoch}")

    # 检查checkpoint
    if start_epoch > 0 and os.path.exists(CHECKPOINT):
        print(f"Loading checkpoint: {CHECKPOINT}")
        sd = torch.load(CHECKPOINT, map_location=device)
        is_dp = isinstance(model, nn.DataParallel)
        has_pre = any(k.startswith('module.') for k in sd.keys())
        if is_dp and not has_pre:
            sd = {'module.' + k: v for k, v in sd.items()}
        elif not is_dp and has_pre:
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd)

    # 优化器与调度器
    criterion = PaperDiceLoss(eps=1.0, bce_weight=0.5, dice_weight=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    # 如果是续训，调整调度器状态
    if start_epoch > 0:
        scheduler.last_epoch = start_epoch - 1

    print(f"LR={LEARNING_RATE}, Epochs={EPOCHS}, BS={BATCH_SIZE}, Dim={BASE_DIM}")
    print("-" * 60)

    trigger = 0
    best_val_loss = float("inf")
    if start_epoch > 0 and logger.h and 'val_loss' in logger.h:
        best_val_loss = min(logger.h['val_loss'])

    for epoch in range(start_epoch, EPOCHS):
        ep = epoch + 1
        model.train()
        t_loss = 0.0
        n_b = 0

        pbar = tqdm(train_loader, desc=f"Epoch {ep:03d}")
        for imgs, masks, _ in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                loss = criterion(out, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item()
            n_b += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_t = t_loss / max(n_b, 1)
        v_loss, metrics = compute_metrics(model, val_loader, criterion, device,
                                          PRED_THRESHOLD)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        msg = ""
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            trigger = 0
            msg = "Best"
            rm = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(rm.state_dict(), CHECKPOINT)
        else:
            trigger += 1
            msg = f"Wait({trigger}/{PATIENCE})"

        logger.log(ep, avg_t, v_loss, metrics, lr, msg)
        print(f"Ep{ep:03d} | Tr:{avg_t:.4f} Va:{v_loss:.4f} | "
              f"IoU:{metrics['IoU']:.4f} F1:{metrics['F1']:.4f} "
              f"P:{metrics['Precision']:.4f} R:{metrics['Recall']:.4f} | "
              f"LR:{lr:.2e} {msg}")

        if ep % VIZ_EVERY == 0:
            save_prediction_figure(model, val_ds, device,
                                   os.path.join(OUTPUT_ROOT, f"viz_ep{ep:03d}.png"),
                                   ep, VIZ_SAMPLE_INDICES, SLICE_IDX, PRED_THRESHOLD)
        if trigger >= PATIENCE:
            print(f"Early stop at epoch {ep}")
            break

    logger.plot(os.path.join(OUTPUT_ROOT, "training_curves.png"))
    v_loss, metrics = compute_metrics(model, val_loader, criterion, device,
                                      PRED_THRESHOLD)
    print(f"\nFinal: IoU={metrics['IoU']:.4f} F1={metrics['F1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
