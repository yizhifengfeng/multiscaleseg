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
# 路径：严格适配Kaggle P100（仅修改这里，其他全保留）
# -----------------------------------------------------------------------------
_ROOT = "/kaggle/input/datasets/zhifengfengyi"
_DEFAULT_DATA = os.path.join(_ROOT, "gaoguangpu")
def _env_path(key: str, default: str) -> str:
    v = os.environ.get(key, "").strip()
    return v if v else default
train_img_dir = os.path.join(_DEFAULT_DATA, "img")
train_mask_dir = os.path.join(_DEFAULT_DATA, "masknpy")
OUTPUT_ROOT = "/kaggle/working/output_results"
MODEL_DIR = os.path.join(OUTPUT_ROOT, "models")
SLICE_IDX = int(os.environ.get("CRACKSEG_SLICE_IDX", "88"))
PRED_THRESHOLD = float(os.environ.get("CRACKSEG_THRESH", "0.5"))
EPOCHS = int(os.environ.get("CRACKSEG_EPOCHS", "300"))
VIZ_EVERY = int(os.environ.get("CRACKSEG_VIZ_EVERY", "5"))
VIZ_SAMPLE_INDICES = [0, 1, 2]

# ===================== 注意力模块（100%还原你的原代码，未做任何修改） =====================
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

# ===================== 主模型（100%还原你的原代码，未做任何修改） =====================
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

# ===================== 数据集（仅修复高光谱维度，保留你原逻辑） =====================
class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npy") and not f.startswith(".")])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".npy") and not f.startswith(".")])
        assert len(self.img_files) == len(self.mask_files), "图像与掩码数量不匹配"
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        # 修复：高光谱(H,W,176) → 适配3D卷积(1,176,H,W)
        img = np.load(os.path.join(self.img_dir, self.img_files[idx])).transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        mask = np.load(os.path.join(self.mask_dir, self.mask_files[idx]))
        return torch.from_numpy(img).float(), torch.from_numpy(mask).long(), self.img_files[idx]

# ===================== 损失函数（100%还原你的原代码） =====================
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

# ===================== 分割指标（100%还原你的原代码） =====================
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
    return {
        "Dice": dice.item(),
        "IoU": iou_fg.item(),
        "Recall": recall.item(),
    }

@torch.no_grad()
def evaluate_metrics_and_loss(model, loader, criterion, device, threshold: float):
    model.eval()
    tp = fp = fn = tn = 0.0
    total_loss = 0.0
    n_batches = 0
    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(imgs)
            loss = criterion(outputs, masks)
        total_loss += loss.item()
        n_batches += 1
        btp, bfp, bfn, btn = _accumulate_confusion(outputs, masks, threshold)
        tp += btp; fp += bfp; fn += bfn; tn += btn
    avg_loss = total_loss / max(n_batches, 1)
    metrics = confusion_to_metrics(tp, fp, fn, tn)
    return avg_loss, metrics

# ===================== 可视化（100%还原你的原代码） =====================
def save_prediction_figure(model, dataset, device, save_path: str, epoch_display: int, sample_indices, slice_idx: int,
                           threshold: float):
    model.eval()
    num_samples = len(sample_indices)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            if idx >= len(dataset): continue
            img, mask, filename = dataset[idx]
            img_input = img.unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model(img_input)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask = (prob_map > threshold).astype(np.float32)
            input_slice = img[0, slice_idx].numpy()
            gt_mask = mask.numpy()
            axes[i, 0].imshow(input_slice, cmap="gray")
            axes[i, 0].set_title(f"Sample: {filename}\nInput (Slice {slice_idx})")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(gt_mask, cmap="gray")
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis("off")
            axes[i, 2].imshow(pred_mask, cmap="gray")
            axes[i, 2].set_title(f"Prediction Mask (Thresh={threshold})")
            axes[i, 2].axis("off")
    fig.suptitle(f"Epoch {epoch_display} — Hyperspectral slice / GT / Prediction", fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_loss_curve(train_losses, val_losses, save_path: str):
    epochs_x = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_x, train_losses, label="Train Loss", linewidth=2, color="#1f77b4")
    plt.plot(epochs_x, val_losses, label="Validation Loss", linewidth=2, color="#ff7f0e")
    plt.title("Training History — Train vs Validation Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.xticks(epochs_x)
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ===================== 主函数（100%还原你的原训练逻辑，仅适配Kaggle） =====================
def main():
    ensure_dirs()
    metrics_log_path = os.path.join(OUTPUT_ROOT, "metrics_log.txt")
    full_dataset = CrackDataset(train_img_dir, train_mask_dir)
    total_size = len(full_dataset)
    # 保留你原有的8:2划分
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # P100显存优化batch_size
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型输入通道保持1，数据已预处理为(1,176,H,W)，完全匹配你的原模型
    model = mymodel(in_channel=1, classnum=1).to(device)
    criterion = DiceLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_metrics = None
    best_epoch = -1
    early_stopping_patience = 30
    trigger_times = 0

    with open(metrics_log_path, "w", encoding="utf-8") as f:
        f.write(
            "MS-CrackSeg training log (P100 Optimized)\n"
            f"Data: train_img={train_img_dir}, Validation Split=20%\n"
            f"Device: {device}, Max Epochs: {EPOCHS}, Early Stopping Patience: {early_stopping_patience}\n"
            "Columns: Epoch | TrainLoss | ValLoss | Dice | IoU | Recall\n\n"
        )
    print(f"使用设备: {device}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print(f"数据总数: {total_size} (训练样本: {train_size}, 验证样本: {val_size})")

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}] Train")
        for batch_idx, (imgs, masks, _) in enumerate(train_bar):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{train_loss_sum / (batch_idx + 1):.4f}")

        avg_train = train_loss_sum / len(train_loader)
        avg_val, metrics = evaluate_metrics_and_loss(model, val_loader, criterion, device, PRED_THRESHOLD)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_metrics = metrics.copy()
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            trigger_times = 0
            status_msg = ">> 验证损失下降，保存最优模型"
        else:
            trigger_times += 1
            status_msg = f">> 验证损失未下降 ({trigger_times}/{early_stopping_patience})"

        line = (
            f"Epoch {epoch + 1:3d} | TrainLoss {avg_train:.6f} | ValLoss {avg_val:.6f} | "
            f"Dice {metrics['Dice']:.6f} | IoU {metrics['IoU']:.6f} | R {metrics['Recall']:.6f} | {status_msg}"
        )
        print(line.strip())
        with open(metrics_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if (epoch + 1) % VIZ_EVERY == 0:
            out_png = os.path.join(OUTPUT_ROOT, f"predictions_epoch_{epoch + 1}.png")
            save_prediction_figure(model, val_dataset, device, out_png, epoch + 1, VIZ_SAMPLE_INDICES, SLICE_IDX, PRED_THRESHOLD)
            print(f" 已保存可视化: {out_png}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if trigger_times >= early_stopping_patience:
            print(f"\n!!! 触发早停，连续 {early_stopping_patience} 轮无改善，训练终止于 Epoch {epoch + 1} !!!")
            break

    final_path = os.path.join(MODEL_DIR, "crackseg_p100_final.pth")
    torch.save(model.state_dict(), final_path)
    curve_path = os.path.join(OUTPUT_ROOT, "loss_curve.png")
    plot_loss_curve(train_losses, val_losses, curve_path)

    with open(metrics_log_path, "a", encoding="utf-8") as f:
        f.write("\n--- Summary ---\n")
        f.write(f"Best Validation Loss: {best_val_loss:.6f} (Epoch {best_epoch})\n")
        if best_metrics:
            for k, v in best_metrics.items():
                f.write(f"Metrics at best-val-loss epoch — {k}: {v:.6f}\n")
        f.write(f"Final model: {final_path}\n")
        f.write(f"Best model (min val loss): {os.path.join(MODEL_DIR, 'best_model.pth')}\n")
        f.write(f"Loss curve: {curve_path}\n")

    print(f"\n训练结束。最终权重: {final_path}")
    print(f"最优验证损失: {best_val_loss:.6f}（Epoch {best_epoch}）-> {os.path.join(MODEL_DIR, 'best_model.pth')}")
    print(f"损失曲线: {curve_path}")
    print(f"指标日志: {metrics_log_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())