import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. 路径与全局参数
# -----------------------------------------------------------------------------
_ROOT = "/kaggle/input/datasets/zhifengfengyi"
_DEFAULT_DATA = os.path.join(_ROOT, "gaoguangpu")

train_img_dir = os.path.join(_DEFAULT_DATA, "img")
train_mask_dir = os.path.join(_DEFAULT_DATA, "masknpy")
OUTPUT_ROOT = "/kaggle/working/output_results"
MODEL_DIR = os.path.join(OUTPUT_ROOT, "models")

# 超参数优化：T4 x 2 环境
SLICE_IDX = 88
PRED_THRESHOLD = 0.5
EPOCHS = 300
BATCH_SIZE = 16            # 双卡 16 是比较稳健的选择
NUM_WORKERS = 4            
VIZ_EVERY = 5              
VIZ_SAMPLE_INDICES = [0, 1, 2]

# ===================== 注意力模块（保持原样） =====================
class dilateattention(nn.Module):
    def __init__(self, in_channel, depth):
        super(dilateattention, self).__init__()
        self.batch_norm = nn.BatchNorm3d(in_channel)
        self.relu = nn.ReLU()
        self.atrous_block1 = nn.Conv3d(in_channel, depth, 1, 1)
        self.atrous_block2 = nn.Conv3d(in_channel, depth, 3, 1, padding=2, dilation=2, groups=in_channel)
        self.atrous_block3 = nn.Conv3d(in_channel, depth, 3, 1, padding=4, dilation=4, groups=in_channel)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x_res = self.relu(self.batch_norm(x))
        v = self.atrous_block1(x_res)
        q = self.atrous_block2(x_res)
        k = self.atrous_block3(x_res)
        temp = (k * q) / (k.size(1) ** 0.5)
        attention_map = self.softmax(temp)
        output = v * attention_map + x
        return output

# ===================== 主模型（保持原样） =====================
class mymodel(nn.Module):
    def __init__(self, in_channel, classnum=1):
        super(mymodel, self).__init__()
        self.dim = 8
        self.conv3d1 = nn.Conv3d(in_channel, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dialatiaon1 = dilateattention(self.dim, self.dim)
        self.conv3d2 = nn.Conv3d(self.dim, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dialatiaon2 = dilateattention(self.dim, self.dim)
        self.conv3d3 = nn.Conv3d(self.dim, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dialatiaon3 = dilateattention(self.dim, self.dim)
        self.transpose1 = nn.ConvTranspose3d(self.dim, self.dim, kernel_size=2, stride=2)
        self.conv2d1 = nn.Conv3d(self.dim, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.transpose2 = nn.ConvTranspose3d(self.dim * 2, self.dim, kernel_size=2, stride=2)
        self.conv2d2 = nn.Conv3d(self.dim, self.dim, kernel_size=7, padding=3, stride=1)
        self.transpose3 = nn.ConvTranspose3d(self.dim * 2, self.dim, kernel_size=2, stride=2)
        self.conv2d3 = nn.Conv3d(self.dim, 1, kernel_size=7, padding=3, stride=1)
        self.final = nn.Conv2d(176, classnum, 3, 1, 1)
    def forward(self, x):
        x1 = self.conv3d1(x)
        x1 = self.maxpooling1(x1)
        x1 = self.dialatiaon1(x1)
        x2 = self.conv3d2(x1)
        x2 = self.maxpooling2(x2)
        x2 = self.dialatiaon2(x2)
        x3 = self.conv3d3(x2)
        x3 = self.maxpooling3(x3)
        x3 = self.dialatiaon3(x3)
        x4 = self.transpose1(x3)
        x4 = self.conv2d1(x4)
        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.transpose2(x4)
        x5 = self.conv2d2(x5)
        x6 = torch.cat([x5, x1], dim=1)
        x6 = self.transpose3(x6)
        x6 = self.conv2d3(x6)
        x6 = torch.squeeze(x6, dim=1)
        x6 = self.final(x6)
        return x6

# ===================== 数据集处理（修复核心错误） =====================
class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npy") and not f.startswith(".")])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".npy") and not f.startswith(".")])
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_raw = np.load(os.path.join(self.img_dir, self.img_files[idx]))
        
        # --- 自动校正维度（修复 RuntimeError 的关键） ---
        # 目标形状必须是 (176, H, W)
        if img_raw.shape[0] == 176:
            img = img_raw
        elif img_raw.shape[2] == 176:
            img = img_raw.transpose(2, 0, 1)
        else:
            # 最后的保底措施，如果形状奇怪，手动强制转置
            img = img_raw.transpose(2, 0, 1) if img_raw.ndim == 3 else img_raw
            
        img = np.expand_dims(img, axis=0) # 变为 (1, 176, H, W) 适配 3D 卷积
        mask = np.load(os.path.join(self.mask_dir, self.mask_files[idx]))
        return torch.from_numpy(img).float(), torch.from_numpy(mask).long(), self.img_files[idx]

# ===================== 损失函数与指标 =====================
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, inputs, target, smooth=1e-5):
        bce_loss = F.binary_cross_entropy_with_logits(inputs.view(-1), target.view(-1).float())
        inputs = torch.sigmoid(inputs); inputs = inputs.view(-1); target = target.view(-1)
        intersection = (inputs * target).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + target.sum() + smooth)
        return bce_loss + (1 - dice)

def confusion_to_metrics(tp, fp, fn, tn, eps=1e-7):
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    return {"Dice": dice.item(), "IoU": iou.item()}

@torch.no_grad()
def evaluate_metrics_and_loss(model, loader, criterion, device, threshold: float):
    model.eval()
    tp = fp = fn = tn = 0.0; total_loss = 0.0
    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, masks)
        total_loss += loss.item()
        pred = (torch.sigmoid(outputs) > threshold).long()
        tp += ((pred == 1) & (masks == 1)).sum().float()
        fp += ((pred == 1) & (masks == 0)).sum().float()
        fn += ((pred == 0) & (masks == 1)).sum().float()
        tn += ((pred == 0) & (masks == 0)).sum().float()
    return total_loss / len(loader), confusion_to_metrics(tp, fp, fn, tn)

# ===================== 可视化 =====================
def save_prediction_figure(model, dataset, device, save_path, epoch, sample_indices, slice_idx, threshold):
    model.eval()
    fig, axes = plt.subplots(len(sample_indices), 3, figsize=(15, 5 * len(sample_indices)))
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            img, mask, _ = dataset[idx]
            input_tensor = img.unsqueeze(0).to(device)
            with torch.amp.autocast('cuda'):
                output = model(input_tensor)
            pred = (torch.sigmoid(output).squeeze().cpu().numpy() > threshold).astype(np.float32)
            axes[i, 0].imshow(img[0, slice_idx].numpy(), cmap="gray")
            axes[i, 1].imshow(mask.numpy(), cmap="gray")
            axes[i, 2].imshow(pred, cmap="gray")
            for ax in axes[i]: ax.axis("off")
    plt.tight_layout(); fig.savefig(save_path, dpi=100); plt.close(fig)

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ===================== 核心训练逻辑 =====================
def main():
    ensure_dirs()
    full_dataset = CrackDataset(train_img_dir, train_mask_dir)
    train_size = int(0.8 * len(full_dataset))
    train_ds, val_ds = random_split(full_dataset, [train_size, len(full_dataset)-train_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mymodel(in_channel=1, classnum=1)
    
    if torch.cuda.device_count() > 1:
        print(f"激活双显卡模式！检测到 {torch.cuda.device_count()} 张 T4")
        model = nn.DataParallel(model)
    model = model.to(device)
# === 新增：自动加载断点 ===
    checkpoint_path = os.path.join(MODEL_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"检测到已有存档，正在加载权重继续训练...")
        model.load_state_dict(torch.load(checkpoint_path))
    # ========================
    
    criterion = DiceLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda') # 更新为新版 API

    best_val_loss = float("inf")
    patience = 30; trigger = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, masks, _ in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'): # 更新为新版 API
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss, metrics = evaluate_metrics_and_loss(model, val_loader, criterion, device, PRED_THRESHOLD)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            real_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(real_model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            trigger = 0
            msg = "Saved!"
        else:
            trigger += 1
            msg = f"Wait {trigger}"

        print(f"Epoch {epoch+1:03d} | Train: {train_loss/len(train_loader):.4f} | Val: {val_loss:.4f} | IoU: {metrics['IoU']:.4f} | {msg}")

        if (epoch + 1) % VIZ_EVERY == 0:
            save_prediction_figure(model, val_ds, device, os.path.join(OUTPUT_ROOT, f"viz_{epoch+1}.png"), epoch+1, VIZ_SAMPLE_INDICES, SLICE_IDX, PRED_THRESHOLD)

        if trigger >= patience:
            print("触发早停。")
            break

if __name__ == "__main__":
    main()
