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
# 1. 路径与全局参数优化（适配 Kaggle）
# -----------------------------------------------------------------------------
_ROOT = "/kaggle/input/datasets/zhifengfengyi"  # 请确保你的数据集名称与此路径一致
_DEFAULT_DATA = os.path.join(_ROOT, "gaoguangpu")

train_img_dir = os.path.join(_DEFAULT_DATA, "img")
train_mask_dir = os.path.join(_DEFAULT_DATA, "masknpy")
OUTPUT_ROOT = "/kaggle/working/output_results"
MODEL_DIR = os.path.join(OUTPUT_ROOT, "models")

# --- 针对 T4 x 2 的超参数优化 ---
SLICE_IDX = int(os.environ.get("CRACKSEG_SLICE_IDX", "88"))
PRED_THRESHOLD = float(os.environ.get("CRACKSEG_THRESH", "0.5"))
EPOCHS = 300                # 最大轮数
BATCH_SIZE = 16            # T4显存较大，且有两张卡，将 4 提升至 16 以加快速度
NUM_WORKERS = 4            # Kaggle CPU 为 4 核，增加线程数加快数据读取
VIZ_EVERY = 5              # 每 5 轮保存一次可视化结果
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

# ===================== 数据集处理 =====================
class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # 增加容错：忽略隐藏文件并确保排序
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npy") and not f.startswith(".")])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".npy") and not f.startswith(".")])
        if len(self.img_files) != len(self.mask_files):
            print(f"警告：图片数量({len(self.img_files)})与掩码数量({len(self.mask_files)})不一致！")
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        # 高光谱 (H, W, 176) -> (1, 176, H, W) 适配 3D 卷积
        img = np.load(os.path.join(self.img_dir, self.img_files[idx])).transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        mask = np.load(os.path.join(self.mask_dir, self.mask_files[idx]))
        return torch.from_numpy(img).float(), torch.from_numpy(mask).long(), self.img_files[idx]

# ===================== 损失函数与指标 =====================
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, inputs, target, smooth=1e-5):
        bce_loss = F.binary_cross_entropy_with_logits(inputs.view(-1), target.view(-1).float())
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        target = target.view(-1)
        intersection = (inputs * target).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + target.sum() + smooth)
        dice_loss = 1 - dice
        return bce_loss + dice_loss

def _accumulate_confusion(logits, target, threshold: float):
    pred = (torch.sigmoid(logits) > threshold).long()
    t = target.long()
    tp = ((pred == 1) & (t == 1)).sum().float()
    fp = ((pred == 1) & (t == 0)).sum().float()
    fn = ((pred == 0) & (t == 1)).sum().float()
    tn = ((pred == 0) & (t == 0)).sum().float()
    return tp, fp, fn, tn

def confusion_to_metrics(tp, fp, fn, tn, eps=1e-7):
    iou_fg = (tp + eps) / (tp + fp + fn + eps)
    recall = (tp + eps) / (tp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    return {"Dice": dice.item(), "IoU": iou_fg.item(), "Recall": recall.item()}

@torch.no_grad()
def evaluate_metrics_and_loss(model, loader, criterion, device, threshold: float):
    model.eval()
    tp = fp = fn = tn = 0.0
    total_loss = 0.0
    n_batches = 0
    for imgs, masks, _ in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, masks)
        total_loss += loss.item()
        n_batches += 1
        btp, bfp, bfn, btn = _accumulate_confusion(outputs, masks, threshold)
        tp += btp; fp += bfp; fn += bfn; tn += btn
    avg_loss = total_loss / max(n_batches, 1)
    metrics = confusion_to_metrics(tp, fp, fn, tn)
    return avg_loss, metrics

# ===================== 可视化函数 =====================
def save_prediction_figure(model, dataset, device, save_path: str, epoch_display: int, sample_indices, slice_idx: int, threshold: float):
    model.eval()
    num_samples = len(sample_indices)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1: axes = np.expand_dims(axes, axis=0)
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            if idx >= len(dataset): continue
            img, mask, filename = dataset[idx]
            img_input = img.unsqueeze(0).to(device)
            with torch.cuda.amp.autocast():
                output = model(img_input)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask = (prob_map > threshold).astype(np.float32)
            input_slice = img[0, slice_idx].numpy()
            gt_mask = mask.numpy()
            
            axes[i, 0].imshow(input_slice, cmap="gray")
            axes[i, 0].set_title(f"Sample: {filename}\nInput (Slice {slice_idx})")
            axes[i, 1].imshow(gt_mask, cmap="gray")
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 2].imshow(pred_mask, cmap="gray")
            axes[i, 2].set_title(f"Prediction (Epoch {epoch_display})")
            for ax in axes[i]: ax.axis("off")
            
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ===================== 核心训练逻辑（双显卡适配） =====================
def main():
    ensure_dirs()
    metrics_log_path = os.path.join(OUTPUT_ROOT, "metrics_log.txt")
    
    # 加载数据集
    full_dataset = CrackDataset(train_img_dir, train_mask_dir)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # 数据读取器优化
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- 设备与模型双卡配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mymodel(in_channel=1, classnum=1)
    
    # 关键修改：如果检测到多张显卡，开启 DataParallel
    if torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 张显卡，开启并行训练模式！")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    # -----------------------

    criterion = DiceLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    early_stopping_patience = 30
    trigger_times = 0

    print(f"开始训练 | Batch Size: {BATCH_SIZE} | 并行线程: {NUM_WORKERS}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for imgs, masks, _ in train_bar:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sum += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = train_loss_sum / len(train_loader)
        avg_val, metrics = evaluate_metrics_and_loss(model, val_loader, criterion, device, PRED_THRESHOLD)
        
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step()

        # 保存最优模型
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # 注意：如果用了 DataParallel，保存时要用 model.module
            save_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(save_model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            trigger_times = 0
            status = "Best!"
        else:
            trigger_times += 1
            status = f"Patience {trigger_times}/{early_stopping_patience}"

        log_line = f"Epoch {epoch+1:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | IoU: {metrics['IoU']:.4f} | {status}"
        print(log_line)
        
        with open(metrics_log_path, "a") as f:
            f.write(log_line + "\n")

        if (epoch + 1) % VIZ_EVERY == 0:
            save_prediction_figure(model, val_dataset, device, os.path.join(OUTPUT_ROOT, f"viz_ep_{epoch+1}.png"), epoch+1, VIZ_SAMPLE_INDICES, SLICE_IDX, PRED_THRESHOLD)

        if trigger_times >= early_stopping_patience:
            print("触发早停，训练结束。")
            break

    return 0

if __name__ == "__main__":
    main()
