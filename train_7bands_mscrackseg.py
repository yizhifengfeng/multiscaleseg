#!/usr/bin/env python3
"""
Generalization-oriented 7-band MS-CrackSeg training script.

This version fixes the main causes of optimistic validation metrics:
1. strict image/mask pairing by spatial key;
2. spatial-block validation split instead of random patch split;
3. train-set per-band normalization applied to train/val/test;
4. lower model capacity and stronger regularization;
5. model selection at fixed threshold, with threshold search only after training;
6. optional final evaluation on testimg/testmasknpy.
"""

import json
import math
import os
import random
import re
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


# Paths
DEFAULT_DATA = Path(os.environ.get("CRACK_DATA_DIR", "/root/autodl-tmp/reduction_band/7selectband"))
TRAIN_IMG_DIR = DEFAULT_DATA / "img"
TRAIN_MASK_DIR = DEFAULT_DATA / "masknpy"
TEST_IMG_DIR = DEFAULT_DATA / "testimg"
TEST_MASK_DIR = DEFAULT_DATA / "testmasknpy"

OUTPUT_ROOT = Path(
    os.environ.get("CRACK_OUTPUT_DIR", "/root/autodl-tmp/reduction_band/output_results_7band/output4_generalized")
)
MODEL_DIR = OUTPUT_ROOT / "models"
CHECKPOINT = MODEL_DIR / "7band_best_model.pth"
LATEST_CHECKPOINT = MODEL_DIR / "latest_checkpoint.pth"
TRAIN_LOG = OUTPUT_ROOT / "training_log_7band_generalized.json"
FINAL_METRICS = OUTPUT_ROOT / "final_metrics_7band_generalized.json"


# Training hyperparameters
IN_CHANNELS = 7
EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 2.0e-4
WEIGHT_DECAY = 1.0e-3
PATIENCE = 30
NUM_WORKERS = 0
SEED = 42
GRAD_CLIP = 1.0
PRED_THRESHOLD = 0.5
MIN_DELTA = 1.0e-4
VAL_RATIO = 0.2
SPATIAL_GROUP_SIZE = 4
RESUME_TRAINING = False


# Loss and regularization
POS_WEIGHT = 4.0
AUTO_POS_WEIGHT = True
MAX_POS_WEIGHT = 6.0
DICE_WEIGHT = 0.65
BCE_WEIGHT = 0.25
FOCAL_WEIGHT = 0.10
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.02

BASE_DIM = 16
DROPOUT = 0.35
MIXUP_ALPHA = 0.10
BAND_DROP_PROB = 0.20
EMA_DECAY = 0.995


# Visualization
VIZ_BAND_IDX = 3
VIZ_EVERY = 20
VIZ_SAMPLE_INDICES = [0, 1, 2]
BAND_NAMES = ["397.5nm", "456.4nm", "532.7nm", "577.8nm", "903.9nm", "935.1nm", "942.0nm"]

KEY_PATTERN = re.compile(r"(r\d+c\d+col\d+row\d+overlap\d+)", re.IGNORECASE)
SPATIAL_PATTERN = re.compile(r"r(\d+)c(\d+)col(\d+)row(\d+)overlap(\d+)", re.IGNORECASE)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_dirs():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "viz").mkdir(parents=True, exist_ok=True)


def extract_key(filename):
    match = KEY_PATTERN.search(Path(filename).stem)
    if match is None:
        raise ValueError(f"Cannot extract spatial key from filename: {filename}")
    return match.group(1).lower()


def read_image(path):
    img = np.load(path).astype(np.float32, copy=False)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    if img.ndim == 4 and 1 in img.shape:
        img = np.squeeze(img)
    if img.ndim != 3:
        raise ValueError(f"Image must be 3D, got {img.shape}: {path}")
    if img.shape[0] == IN_CHANNELS:
        return np.ascontiguousarray(img, dtype=np.float32)
    if img.shape[-1] == IN_CHANNELS:
        return np.ascontiguousarray(np.transpose(img, (2, 0, 1)), dtype=np.float32)
    raise ValueError(f"Image must have {IN_CHANNELS} bands, got {img.shape}: {path}")


def read_mask(path):
    mask = np.load(path)
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D after squeeze, got {mask.shape}: {path}")
    return np.ascontiguousarray((mask > 0).astype(np.float32))


def pair_files(img_dir, mask_dir):
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    images = sorted(img_dir.glob("*.npy"))
    masks = sorted(mask_dir.glob("*.npy"))
    if not images:
        raise RuntimeError(f"No .npy images found in {img_dir}")
    if not masks:
        raise RuntimeError(f"No .npy masks found in {mask_dir}")

    image_by_key = {}
    mask_by_key = {}
    for path in images:
        key = extract_key(path.name)
        if key in image_by_key:
            raise RuntimeError(f"Duplicate image key {key}: {image_by_key[key].name}, {path.name}")
        image_by_key[key] = path
    for path in masks:
        key = extract_key(path.name)
        if key in mask_by_key:
            raise RuntimeError(f"Duplicate mask key {key}: {mask_by_key[key].name}, {path.name}")
        mask_by_key[key] = path

    missing_masks = sorted(set(image_by_key) - set(mask_by_key))
    missing_images = sorted(set(mask_by_key) - set(image_by_key))
    if missing_masks or missing_images:
        raise RuntimeError(
            f"Pairing failed: missing_masks={missing_masks[:5]}, missing_images={missing_images[:5]}"
        )

    return [(key, image_by_key[key], mask_by_key[key]) for key in sorted(image_by_key)]


def spatial_group(key):
    match = SPATIAL_PATTERN.search(key)
    if match is None:
        return key
    r, c, col, row, _ = [int(v) for v in match.groups()]
    return (r, c, col // SPATIAL_GROUP_SIZE, row // SPATIAL_GROUP_SIZE)


def grouped_split(pairs, val_ratio=VAL_RATIO, seed=SEED):
    groups = {}
    for idx, (key, _, _) in enumerate(pairs):
        groups.setdefault(spatial_group(key), []).append(idx)

    group_items = list(groups.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)

    target_val = max(1, int(round(len(pairs) * val_ratio)))
    val_indices = []
    train_indices = []
    for _, indices in group_items:
        if len(val_indices) < target_val:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    if not train_indices or not val_indices:
        indices = list(range(len(pairs)))
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_ratio)))
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]

    return sorted(train_indices), sorted(val_indices)


def compute_band_stats(pairs, indices):
    sums = np.zeros(IN_CHANNELS, dtype=np.float64)
    sq_sums = np.zeros(IN_CHANNELS, dtype=np.float64)
    count = 0
    for idx in tqdm(indices, desc="Computing band mean/std", leave=False):
        img = read_image(pairs[idx][1])
        flat = img.reshape(IN_CHANNELS, -1).astype(np.float64)
        sums += flat.sum(axis=1)
        sq_sums += (flat * flat).sum(axis=1)
        count += flat.shape[1]
    mean = sums / max(count, 1)
    var = sq_sums / max(count, 1) - mean * mean
    std = np.sqrt(np.maximum(var, 1.0e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def compute_pos_weight(pairs, indices):
    positives = 0.0
    total = 0.0
    for idx in tqdm(indices, desc="Computing pos_weight", leave=False):
        mask = read_mask(pairs[idx][2])
        positives += float(mask.sum())
        total += float(mask.size)
    negatives = max(total - positives, 1.0)
    positives = max(positives, 1.0)
    return float(np.clip(negatives / positives, 1.0, MAX_POS_WEIGHT))


class CrackDataset(Dataset):
    def __init__(self, pairs, augment=False, band_mean=None, band_std=None):
        self.pairs = list(pairs)
        self.augment = augment
        self.band_mean = None if band_mean is None else np.asarray(band_mean, dtype=np.float32).reshape(-1, 1, 1)
        self.band_std = None if band_std is None else np.asarray(band_std, dtype=np.float32).reshape(-1, 1, 1)
        if not self.pairs:
            raise RuntimeError("Dataset is empty")

    def __len__(self):
        return len(self.pairs)

    def set_normalization(self, band_mean, band_std):
        self.band_mean = np.asarray(band_mean, dtype=np.float32).reshape(-1, 1, 1)
        self.band_std = np.asarray(band_std, dtype=np.float32).reshape(-1, 1, 1)

    def _normalize(self, img):
        if self.band_mean is None or self.band_std is None:
            lo = np.percentile(img, 1, axis=(1, 2), keepdims=True)
            hi = np.percentile(img, 99, axis=(1, 2), keepdims=True)
            return np.clip((img - lo) / np.maximum(hi - lo, 1.0e-6), 0.0, 1.0)
        img = (img - self.band_mean) / np.maximum(self.band_std, 1.0e-6)
        return np.clip(img, -5.0, 5.0)

    def __getitem__(self, idx):
        key, img_path, mask_path = self.pairs[idx]
        img = read_image(img_path)
        mask = read_mask(mask_path)

        if img.shape[1:] != mask.shape:
            raise ValueError(f"Spatial mismatch for {key}: image={img.shape}, mask={mask.shape}")

        if self.augment:
            if random.random() < 0.5:
                img = img[:, :, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if random.random() < 0.5:
                img = img[:, ::-1, :].copy()
                mask = mask[::-1, :].copy()
            if random.random() < 0.5:
                k = random.choice([1, 2, 3])
                img = np.rot90(img, k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()
            if random.random() < 0.35:
                img = img * np.random.uniform(0.85, 1.15)
            if random.random() < 0.35:
                scale = np.random.uniform(0.90, 1.10, (img.shape[0], 1, 1)).astype(np.float32)
                img = img * scale

        img = self._normalize(img)

        if self.augment:
            if random.random() < 0.25:
                img = img + np.random.normal(0.0, 0.03, img.shape).astype(np.float32)
            if random.random() < BAND_DROP_PROB:
                drop_idx = np.random.choice(img.shape[0], 1, replace=False)
                img = img.copy()
                img[drop_idx] = 0.0

        return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(mask.astype(np.float32)), key


class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def group_norm(channels):
    groups = 8
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class MSSK(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=DROPOUT):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        n = out_channels // 4
        self.conv1x1 = nn.Conv2d(in_channels, n, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, n, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, n, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, n, kernel_size=7, padding=3)
        self.fusion = nn.Conv2d(n * 4, out_channels, kernel_size=1)
        self.norm = group_norm(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = torch.cat([self.conv1x1(x), self.conv3x3(x), self.conv5x5(x), self.conv7x7(x)], dim=1)
        out = self.dropout(self.norm(self.fusion(out)))
        return self.relu(out + residual)


class DilateAttention(nn.Module):
    def __init__(self, in_channel, depth):
        super().__init__()
        self.group_norm = group_norm(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1)
        self.atrous_block2 = nn.Conv2d(in_channel, depth, 3, padding=2, dilation=2)
        self.atrous_block3 = nn.Conv2d(in_channel, depth, 3, padding=4, dilation=4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        v = self.atrous_block1(x)
        q = self.atrous_block2(x)
        k = self.atrous_block3(x)
        return v * self.softmax(k * q) + self.relu(self.group_norm(x))


class CrackSeg7Band(nn.Module):
    def __init__(self, in_channels=7, classnum=1, base_dim=BASE_DIM, dropout=DROPOUT):
        super().__init__()
        d = base_dim
        self.relu = nn.ReLU(inplace=True)

        self.enc_conv1 = nn.Conv2d(in_channels, d, kernel_size=7, padding=3)
        self.enc_gn1 = group_norm(d)
        self.pool1 = nn.MaxPool2d(2)
        self.da1 = DilateAttention(d, d)
        self.drop1 = nn.Dropout2d(dropout)

        self.enc_conv2 = nn.Conv2d(d, d, kernel_size=7, padding=3)
        self.enc_gn2 = group_norm(d)
        self.pool2 = nn.MaxPool2d(2)
        self.da2 = DilateAttention(d, d)
        self.drop2 = nn.Dropout2d(dropout)

        self.enc_conv3 = nn.Conv2d(d, d, kernel_size=7, padding=3)
        self.enc_gn3 = group_norm(d)
        self.pool3 = nn.MaxPool2d(2)
        self.da3 = DilateAttention(d, d)
        self.drop3 = nn.Dropout2d(dropout)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=7, padding=3),
            group_norm(d),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.up3 = nn.ConvTranspose2d(d, d, kernel_size=2, stride=2)
        self.mssk1 = MSSK(d * 2, d, dropout)
        self.up2 = nn.ConvTranspose2d(d, d, kernel_size=2, stride=2)
        self.mssk2 = MSSK(d * 2, d, dropout)
        self.up1 = nn.ConvTranspose2d(d, d, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=3, padding=1),
            group_norm(d),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.final = nn.Conv2d(d, classnum, kernel_size=1)

    def forward(self, x):
        e1 = self.relu(self.enc_gn1(self.enc_conv1(x)))
        e1 = self.drop1(self.da1(self.pool1(e1)))

        e2 = self.relu(self.enc_gn2(self.enc_conv2(e1)))
        e2 = self.drop2(self.da2(self.pool2(e2)))

        e3 = self.relu(self.enc_gn3(self.enc_conv3(e2)))
        e3 = self.drop3(self.da3(self.pool3(e3)))

        b = self.bottleneck(e3)
        d3 = self.mssk1(torch.cat([self.up3(b), e2], dim=1))
        d2 = self.mssk2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.dec_conv1(self.up1(d2))
        return self.final(d1)


class DiceBCEFocalLoss(nn.Module):
    def __init__(self, pos_weight=POS_WEIGHT):
        super().__init__()
        self.register_buffer("pos_weight_tensor", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits, target):
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        logits_flat = logits.view(-1)
        target_flat = target.view(-1).float()
        target_smooth = target_flat * (1.0 - LABEL_SMOOTHING) + LABEL_SMOOTHING / 2.0

        probs = torch.sigmoid(logits_flat)
        bce = F.binary_cross_entropy_with_logits(
            logits_flat, target_smooth, pos_weight=self.pos_weight_tensor.to(logits.device)
        )
        intersection = (probs * target_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (probs.sum() + target_flat.sum() + 1.0)

        p_clamped = torch.clamp(probs, min=1.0e-7, max=1.0 - 1.0e-7)
        p_t = p_clamped * target_flat + (1.0 - p_clamped) * (1.0 - target_flat)
        focal_w = (1.0 - p_t) ** FOCAL_GAMMA
        focal_loss = (
            focal_w * F.binary_cross_entropy_with_logits(logits_flat, target_smooth, reduction="none")
        ).mean()

        return BCE_WEIGHT * bce + DICE_WEIGHT * dice_loss + FOCAL_WEIGHT * focal_loss


@torch.no_grad()
def compute_metrics(model, loader, criterion, device, threshold=PRED_THRESHOLD, desc="Eval"):
    model.eval()
    total_loss = 0.0
    batches = 0
    tp = fp = fn = 0.0
    for imgs, masks, _ in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(imgs)
            loss = criterion(out, masks)
        total_loss += float(loss.item())
        batches += 1
        pred = (torch.sigmoid(out) > threshold).long().squeeze(1)
        labels = (masks > 0.5).long()
        tp += float(((pred == 1) & (labels == 1)).sum().item())
        fp += float(((pred == 1) & (labels == 0)).sum().item())
        fn += float(((pred == 0) & (labels == 1)).sum().item())

    eps = 1.0e-7
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return total_loss / max(batches, 1), {"Precision": precision, "Recall": recall, "F1": f1, "IoU": iou}


@torch.no_grad()
def search_best_threshold(model, loader, device):
    model.eval()
    probs_list = []
    masks_list = []
    for imgs, masks, _ in tqdm(loader, desc="Searching threshold", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(imgs)
        probs = torch.sigmoid(out).cpu().numpy()[:, 0]
        masks_np = masks.numpy()
        for i in range(probs.shape[0]):
            probs_list.append(probs[i])
            masks_list.append(masks_np[i])

    best_thr = PRED_THRESHOLD
    best_iou = -1.0
    for thr in np.arange(0.35, 0.76, 0.05):
        tp = fp = fn = 0.0
        for prob, mask in zip(probs_list, masks_list):
            pred = prob > thr
            label = mask > 0.5
            tp += float((pred & label).sum())
            fp += float((pred & ~label).sum())
            fn += float((~pred & label).sum())
        iou = (tp + 1.0e-7) / (tp + fp + fn + 1.0e-7)
        if iou > best_iou:
            best_iou = iou
            best_thr = float(thr)
    return best_thr, best_iou


def save_prediction_figure(model, dataset, device, save_path, epoch, indices, band_idx, threshold):
    model.eval()
    indices = [idx for idx in indices if idx < len(dataset)]
    if not indices:
        return
    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5 * len(indices)))
    if len(indices) == 1:
        axes = axes[np.newaxis, :]
    with torch.no_grad():
        for row, idx in enumerate(indices):
            img, mask, key = dataset[idx]
            inp = img.unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(inp)
            pred = (torch.sigmoid(out).squeeze().cpu().numpy() > threshold).astype(np.float32)
            b = min(band_idx, img.shape[0] - 1)
            band = img[b].numpy()
            lo, hi = np.percentile(band, [2, 98])
            band = np.clip((band - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)
            axes[row, 0].imshow(band, cmap="gray")
            axes[row, 0].set_title(f"{key}\nBand {b} ({BAND_NAMES[b]})", fontsize=8)
            axes[row, 1].imshow(mask.numpy(), cmap="gray", vmin=0, vmax=1)
            axes[row, 1].set_title("Ground Truth")
            axes[row, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axes[row, 2].set_title("Prediction")
            for col in range(3):
                axes[row, col].axis("off")
    plt.suptitle(f"Epoch {epoch}, threshold={threshold:.2f}", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


class TrainingLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.h = {
            "train_loss": [],
            "val_loss": [],
            "iou": [],
            "f1": [],
            "precision": [],
            "recall": [],
            "lr": [],
            "threshold": [],
        }

    def log(self, tl, vl, metrics, lr, threshold):
        self.h["train_loss"].append(float(tl))
        self.h["val_loss"].append(float(vl))
        self.h["iou"].append(float(metrics["IoU"]))
        self.h["f1"].append(float(metrics["F1"]))
        self.h["precision"].append(float(metrics["Precision"]))
        self.h["recall"].append(float(metrics["Recall"]))
        self.h["lr"].append(float(lr))
        self.h["threshold"].append(float(threshold))
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.h, f, indent=2)

    def plot(self, path):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes[0, 0].plot(self.h["train_loss"], label="Train")
        axes[0, 0].plot(self.h["val_loss"], label="Val")
        axes[0, 0].set_title("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.h["iou"], label="IoU")
        axes[0, 1].plot(self.h["f1"], label="F1")
        axes[0, 1].set_title("Metrics at Fixed Threshold")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(self.h["precision"], label="Precision")
        axes[1, 0].plot(self.h["recall"], label="Recall")
        axes[1, 0].set_title("Precision / Recall")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(self.h["lr"], label="LR", color="red")
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(self.h["threshold"], label="Threshold", color="green")
        axes[0, 2].set_title("Training Threshold")
        axes[0, 2].grid(True, alpha=0.3)
        axes[1, 2].axis("off")
        plt.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def make_loader(dataset, shuffle, generator=None, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        generator=generator,
        drop_last=drop_last,
        persistent_workers=NUM_WORKERS > 0,
    )


def save_json(path, obj):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    set_seed(SEED)
    ensure_dirs()

    train_pairs = pair_files(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    train_indices, val_indices = grouped_split(train_pairs, VAL_RATIO, SEED)
    train_keys = {train_pairs[i][0] for i in train_indices}
    val_keys = {train_pairs[i][0] for i in val_indices}
    if train_keys & val_keys:
        raise RuntimeError("Train/val split leakage detected")

    band_mean, band_std = compute_band_stats(train_pairs, train_indices)
    pos_weight = compute_pos_weight(train_pairs, train_indices) if AUTO_POS_WEIGHT else POS_WEIGHT

    base_dataset = CrackDataset(train_pairs, augment=False, band_mean=band_mean, band_std=band_std)
    train_ds = Subset(base_dataset, train_indices)
    val_ds = Subset(base_dataset, val_indices)

    test_loader = None
    if TEST_IMG_DIR.is_dir() and TEST_MASK_DIR.is_dir():
        test_pairs = pair_files(TEST_IMG_DIR, TEST_MASK_DIR)
        test_dataset = CrackDataset(test_pairs, augment=False, band_mean=band_mean, band_std=band_std)
        test_loader = make_loader(test_dataset, shuffle=False)
        train_val_keys = train_keys | val_keys
        test_keys = {key for key, _, _ in test_pairs}
        if train_val_keys & test_keys:
            raise RuntimeError("Train/test split leakage detected")
    else:
        test_pairs = []

    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_loader = make_loader(train_ds, shuffle=True, generator=generator, drop_last=False)
    val_loader = make_loader(val_ds, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA: {torch.cuda.get_device_name(0)}")

    model = CrackSeg7Band(in_channels=IN_CHANNELS, classnum=1, base_dim=BASE_DIM, dropout=DROPOUT).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Train/Val/Test samples: {len(train_indices)}/{len(val_indices)}/{len(test_pairs)}")
    print(f"Band mean: {band_mean.tolist()}")
    print(f"Band std:  {band_std.tolist()}")
    print(f"pos_weight: {pos_weight:.4f}")

    criterion = DiceBCEFocalLoss(pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=max(len(train_loader), 1),
        pct_start=0.08,
        div_factor=10,
        final_div_factor=50,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model, EMA_DECAY)

    start_epoch = 0
    best_iou = -1.0
    best_epoch = 0
    trigger = 0
    logger = TrainingLogger(TRAIN_LOG)

    if RESUME_TRAINING and LATEST_CHECKPOINT.exists():
        checkpoint = torch.load(LATEST_CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        ema.shadow = {k: v.to(device) for k, v in checkpoint["ema_shadow"].items()}
        start_epoch = int(checkpoint["epoch"])
        best_iou = float(checkpoint.get("best_iou", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        trigger = int(checkpoint.get("trigger", 0))
        logger.h = checkpoint.get("history", logger.h)
        print(f"Resumed from epoch {start_epoch}, best_iou={best_iou:.4f}")

    config = {
        "data_dir": str(DEFAULT_DATA),
        "output_root": str(OUTPUT_ROOT),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "base_dim": BASE_DIM,
        "dropout": DROPOUT,
        "mixup_alpha": MIXUP_ALPHA,
        "band_drop_prob": BAND_DROP_PROB,
        "ema_decay": EMA_DECAY,
        "val_ratio": VAL_RATIO,
        "spatial_group_size": SPATIAL_GROUP_SIZE,
        "pos_weight": pos_weight,
        "band_mean": band_mean.tolist(),
        "band_std": band_std.tolist(),
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "test_count": len(test_pairs),
    }
    save_json(OUTPUT_ROOT / "config_7band_generalized.json", config)

    for epoch in range(start_epoch, EPOCHS):
        ep = epoch + 1
        base_dataset.augment = True
        model.train()
        train_loss = 0.0
        batches = 0

        for imgs, masks, _ in tqdm(train_loader, desc=f"Epoch {ep:03d}"):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if MIXUP_ALPHA > 0 and random.random() < 0.20 and imgs.size(0) > 1:
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                perm = torch.randperm(imgs.size(0), device=device)
                imgs = lam * imgs + (1.0 - lam) * imgs[perm]
                masks = lam * masks + (1.0 - lam) * masks[perm]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(imgs)
                loss = criterion(out, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update()

            train_loss += float(loss.item())
            batches += 1

        if batches == 0:
            continue

        base_dataset.augment = False
        ema.apply_shadow()
        val_loss, metrics = compute_metrics(model, val_loader, criterion, device, PRED_THRESHOLD, desc="Validate")
        ema.restore()

        avg_train_loss = train_loss / batches
        lr = optimizer.param_groups[0]["lr"]
        logger.log(avg_train_loss, val_loss, metrics, lr, PRED_THRESHOLD)

        improved = metrics["IoU"] > best_iou + MIN_DELTA
        if improved:
            best_iou = metrics["IoU"]
            best_epoch = ep
            trigger = 0
            ema.apply_shadow()
            torch.save(model.state_dict(), CHECKPOINT)
            ema.restore()
            msg = " >>> Best fixed-threshold IoU"
        else:
            trigger += 1
            msg = f" Wait({trigger}/{PATIENCE})"

        torch.save(
            {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "ema_shadow": ema.shadow,
                "best_iou": best_iou,
                "best_epoch": best_epoch,
                "trigger": trigger,
                "history": logger.h,
                "config": config,
            },
            LATEST_CHECKPOINT,
        )

        print(
            f"Ep{ep:03d} | Tr:{avg_train_loss:.4f} Va:{val_loss:.4f} | "
            f"IoU:{metrics['IoU']:.4f} F1:{metrics['F1']:.4f} | "
            f"Prec:{metrics['Precision']:.4f} Rec:{metrics['Recall']:.4f} | "
            f"Thr:{PRED_THRESHOLD:.2f} LR:{lr:.2e}{msg}"
        )

        if ep % VIZ_EVERY == 0:
            ema.apply_shadow()
            save_prediction_figure(
                model,
                val_ds,
                device,
                OUTPUT_ROOT / "viz" / f"viz_ep{ep:03d}.png",
                ep,
                VIZ_SAMPLE_INDICES,
                VIZ_BAND_IDX,
                PRED_THRESHOLD,
            )
            ema.restore()

        if trigger >= PATIENCE:
            print(f"Early stopping: no validation IoU improvement for {PATIENCE} epochs")
            break

    logger.plot(OUTPUT_ROOT / "training_curves_7band_generalized.png")

    if CHECKPOINT.exists():
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
        print(f"Loaded best model from epoch {best_epoch}, fixed-threshold val IoU={best_iou:.4f}")

    best_threshold, val_search_iou = search_best_threshold(model, val_loader, device)
    val_loss, val_metrics = compute_metrics(model, val_loader, criterion, device, best_threshold, desc="Final Val")

    results = {
        "best_epoch": best_epoch,
        "fixed_threshold": PRED_THRESHOLD,
        "selected_threshold": best_threshold,
        "val_threshold_search_iou": val_search_iou,
        "final_val_loss": val_loss,
        "final_val_metrics": val_metrics,
    }

    if test_loader is not None:
        test_loss, test_metrics = compute_metrics(model, test_loader, criterion, device, best_threshold, desc="Final Test")
        results["final_test_loss"] = test_loss
        results["final_test_metrics"] = test_metrics
        print(
            f"Final Test | Loss:{test_loss:.4f} IoU:{test_metrics['IoU']:.4f} "
            f"F1:{test_metrics['F1']:.4f} Prec:{test_metrics['Precision']:.4f} "
            f"Rec:{test_metrics['Recall']:.4f} Thr:{best_threshold:.2f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "selected_threshold": best_threshold,
            "band_mean": band_mean,
            "band_std": band_std,
            "config": config,
            "results": results,
        },
        MODEL_DIR / "7band_best_model_with_preprocess.pth",
    )
    save_json(FINAL_METRICS, results)

    print("=" * 60)
    print(f"Training finished. Outputs: {OUTPUT_ROOT}")
    print(f"Best state_dict: {CHECKPOINT}")
    print(f"Model package with preprocessing: {MODEL_DIR / '7band_best_model_with_preprocess.pth'}")
    print(f"Final metrics: {FINAL_METRICS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
