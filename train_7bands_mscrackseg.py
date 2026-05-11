#!/usr/bin/env python3
"""
Train a 7-selected-band hyperspectral concrete crack segmentation model.

Input:
  7selectband/img/*.npy          image cubes, expected shape (7, H, W)
  7selectband/masknpy/*.npy      binary masks, expected shape (H, W)
  7selectband/testimg/*.npy      held-out test image cubes
  7selectband/testmasknpy/*.npy  held-out test masks

Output:
  7output_results/7best_model.pth
  7output_results/checkpoints/last_checkpoint.pth
  7output_results/logs/metrics_log.txt
  7output_results/logs/final_test_metrics.txt
  7output_results/figures/*.png
  7output_results/predictions/predictions_epoch_*.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
except ModuleNotFoundError as exc:
    missing = exc.name or str(exc)
    raise SystemExit(
        f"Missing dependency: {missing}\n"
        "Please install numpy, matplotlib, tqdm, and a CUDA-enabled PyTorch build if training on GPU."
    ) from exc


@dataclass
class Config:
    # Paths
    base_dir: Path = Path(__file__).resolve().parent
    dataset_dir: Path = base_dir / "7selectband"
    train_img_dir: Path = dataset_dir / "img"
    train_mask_dir: Path = dataset_dir / "masknpy"
    test_img_dir: Path = dataset_dir / "testimg"
    test_mask_dir: Path = dataset_dir / "testmasknpy"
    output_dir: Path = base_dir / "7output_results"
    checkpoint_dir: Path = output_dir / "checkpoints"
    figure_dir: Path = output_dir / "figures"
    prediction_dir: Path = output_dir / "predictions"
    log_dir: Path = output_dir / "logs"
    best_model_path: Path = output_dir / "7best_model.pth"
    last_checkpoint_path: Path = checkpoint_dir / "last_checkpoint.pth"

    # Selected spectral bands from GaiaSky-mini 2, 394-1001 nm.
    selected_wavelengths: Tuple[float, ...] = (942.0, 935.1, 903.9, 577.8, 532.7, 397.5, 456.4)

    # Reproducibility and data
    seed: int = 42
    num_bands: int = 7
    val_ratio: float = 0.2
    threshold: float = 0.5
    num_workers: int = 4
    pin_memory: bool = True

    # Training, kept aligned with train_88bands_mscrackseg.py for controlled comparison.
    epochs: int = 200
    batch_size: int = 2
    learning_rate: float = 1.0e-3
    min_lr: float = 1.0e-6
    warmup_epochs: int = 5
    warmup_start_lr: float = 1.0e-6
    weight_decay: float = 1.0e-4
    dice_loss_weight: float = 0.5
    max_pos_weight: float = 20.0
    grad_clip_norm: float = 1.0
    use_amp: bool = True

    # Augmentation, kept aligned with train_88bands_mscrackseg.py.
    brightness_scale_min: float = 0.90
    brightness_scale_max: float = 1.10
    spectral_scale_min: float = 0.95
    spectral_scale_max: float = 1.05
    spectral_dropout_prob: float = 0.25
    spectral_dropout_max_ratio: float = 0.05

    # Output
    prediction_interval: int = 50
    prediction_samples: int = 4


CFG = Config()
KEY_PATTERN = re.compile(r"(r\d+c\d+col\d+row\d+overlap\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MS-CrackSeg on 7 selected hyperspectral bands.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset root with img/masknpy/testimg/testmasknpy.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--num-workers", type=int, default=None, help="Dataloader workers.")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    parser.add_argument("--fresh", action="store_true", help="Do not resume from last checkpoint.")
    return parser.parse_args()


def apply_args(cfg: Config, args: argparse.Namespace) -> None:
    if args.dataset_dir is not None:
        cfg.dataset_dir = args.dataset_dir.resolve()
        cfg.train_img_dir = cfg.dataset_dir / "img"
        cfg.train_mask_dir = cfg.dataset_dir / "masknpy"
        cfg.test_img_dir = cfg.dataset_dir / "testimg"
        cfg.test_mask_dir = cfg.dataset_dir / "testmasknpy"
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir.resolve()
        cfg.checkpoint_dir = cfg.output_dir / "checkpoints"
        cfg.figure_dir = cfg.output_dir / "figures"
        cfg.prediction_dir = cfg.output_dir / "predictions"
        cfg.log_dir = cfg.output_dir / "logs"
        cfg.best_model_path = cfg.output_dir / "7best_model.pth"
        cfg.last_checkpoint_path = cfg.checkpoint_dir / "last_checkpoint.pth"
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.no_amp:
        cfg.use_amp = False


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_dirs(cfg: Config) -> None:
    for path in [cfg.output_dir, cfg.checkpoint_dir, cfg.figure_dir, cfg.prediction_dir, cfg.log_dir]:
        path.mkdir(parents=True, exist_ok=True)


def has_dataset_layout(path: Path) -> bool:
    return all((path / name).is_dir() for name in ["img", "masknpy", "testimg", "testmasknpy"])


def dataset_layout_report(path: Path) -> str:
    parts = []
    for name in ["img", "masknpy", "testimg", "testmasknpy"]:
        parts.append(f"{name}={'OK' if (path / name).is_dir() else 'MISS'}")
    return f"{path}: " + ", ".join(parts)


def resolve_dataset_paths(cfg: Config) -> None:
    candidates: List[Path] = [
        cfg.dataset_dir,
        cfg.base_dir / "7selectband",
        cfg.base_dir / "7SelectBand",
        cfg.base_dir / "selected_7bands",
        cfg.base_dir / "7bands",
    ]
    for child in sorted(cfg.base_dir.iterdir()) if cfg.base_dir.is_dir() else []:
        if child.is_dir():
            candidates.append(child)

    seen = set()
    unique_candidates = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)

    for path in unique_candidates:
        if has_dataset_layout(path):
            cfg.dataset_dir = path
            cfg.train_img_dir = path / "img"
            cfg.train_mask_dir = path / "masknpy"
            cfg.test_img_dir = path / "testimg"
            cfg.test_mask_dir = path / "testmasknpy"
            print(f"Dataset root: {cfg.dataset_dir}")
            return

    existing = [path for path in unique_candidates if path.exists()]
    reports = "\n".join(dataset_layout_report(path) for path in existing[:30])
    raise FileNotFoundError(
        "Could not find a complete 7-band dataset directory. Expected one folder containing "
        "img, masknpy, testimg, and testmasknpy.\nChecked existing candidates:\n"
        f"{reports if reports else '(no candidate paths exist)'}"
    )


def extract_key(path: Path) -> str:
    match = KEY_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot extract sample key from filename: {path.name}")
    return match.group(1).lower()


def pair_files(img_dir: Path, mask_dir: Path) -> List[Tuple[str, Path, Path]]:
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    img_files = sorted(img_dir.glob("*.npy"))
    mask_files = sorted(mask_dir.glob("*.npy"))
    if not img_files:
        raise RuntimeError(f"No .npy image files found in: {img_dir}")
    if not mask_files:
        raise RuntimeError(f"No .npy mask files found in: {mask_dir}")

    img_by_key: Dict[str, Path] = {}
    mask_by_key: Dict[str, Path] = {}
    for path in img_files:
        key = extract_key(path)
        if key in img_by_key:
            raise RuntimeError(f"Duplicate image key {key}: {img_by_key[key].name}, {path.name}")
        img_by_key[key] = path
    for path in mask_files:
        key = extract_key(path)
        if key in mask_by_key:
            raise RuntimeError(f"Duplicate mask key {key}: {mask_by_key[key].name}, {path.name}")
        mask_by_key[key] = path

    img_keys = set(img_by_key)
    mask_keys = set(mask_by_key)
    missing_masks = sorted(img_keys - mask_keys)
    missing_imgs = sorted(mask_keys - img_keys)
    if missing_masks or missing_imgs:
        report = []
        if missing_masks:
            report.append(f"missing masks for {len(missing_masks)} image keys, first: {missing_masks[:5]}")
        if missing_imgs:
            report.append(f"missing images for {len(missing_imgs)} mask keys, first: {missing_imgs[:5]}")
        raise RuntimeError("; ".join(report))

    return [(key, img_by_key[key], mask_by_key[key]) for key in sorted(img_keys)]


def load_image_array(path: Path, num_bands: int) -> np.ndarray:
    arr = np.load(path).astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.ndim == 4 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Image must be 3D, got shape {arr.shape} for {path.name}")
    if arr.shape[0] == num_bands:
        pass
    elif arr.shape[-1] == num_bands:
        arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Image must contain {num_bands} bands, got shape {arr.shape} for {path.name}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_mask_array(path: Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D after squeeze, got shape {arr.shape} for {path.name}")
    arr = (arr > 0).astype(np.float32)
    return np.ascontiguousarray(arr)


def validate_shapes(pairs: Sequence[Tuple[str, Path, Path]], cfg: Config, name: str) -> None:
    if not pairs:
        raise RuntimeError(f"{name} pair list is empty")
    sample_indices = sorted(set([0, len(pairs) // 2, len(pairs) - 1]))
    for idx in sample_indices:
        key, img_path, mask_path = pairs[idx]
        image = load_image_array(img_path, cfg.num_bands)
        mask = load_mask_array(mask_path)
        if image.shape[1:] != mask.shape:
            raise ValueError(
                f"Image/mask spatial shape mismatch for key {key}: image {image.shape}, mask {mask.shape}"
            )


def split_train_val(
    pairs: Sequence[Tuple[str, Path, Path]], val_ratio: float, seed: int
) -> Tuple[List[Tuple[str, Path, Path]], List[Tuple[str, Path, Path]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(pairs) * val_ratio)))
    val_indices = set(indices[:val_count].tolist())
    train_pairs = [pair for i, pair in enumerate(pairs) if i not in val_indices]
    val_pairs = [pair for i, pair in enumerate(pairs) if i in val_indices]
    return train_pairs, val_pairs


def augment_sample(image: np.ndarray, mask: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image = np.flip(image, axis=2)
        mask = np.flip(mask, axis=1)
    if random.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=0)

    k = random.randint(0, 3)
    if k:
        image = np.rot90(image, k=k, axes=(1, 2))
        mask = np.rot90(mask, k=k, axes=(0, 1))

    if random.random() < 0.5:
        image = np.transpose(image, (0, 2, 1))
        mask = np.transpose(mask, (1, 0))

    brightness = random.uniform(cfg.brightness_scale_min, cfg.brightness_scale_max)
    image = image * brightness

    if random.random() < 0.5:
        band_scale = np.random.uniform(
            cfg.spectral_scale_min, cfg.spectral_scale_max, size=(image.shape[0], 1, 1)
        ).astype(np.float32)
        image = image * band_scale

    if random.random() < cfg.spectral_dropout_prob:
        max_drop = max(1, int(round(image.shape[0] * cfg.spectral_dropout_max_ratio)))
        drop_count = random.randint(1, max_drop)
        drop_idx = np.random.choice(image.shape[0], size=drop_count, replace=False)
        image = image.copy()
        image[drop_idx] = 0.0

    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(mask, dtype=np.float32)


class CrackNpyDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[str, Path, Path]], cfg: Config, augment: bool = False) -> None:
        self.pairs = list(pairs)
        self.cfg = cfg
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        key, img_path, mask_path = self.pairs[index]
        image = load_image_array(img_path, self.cfg.num_bands)
        mask = load_mask_array(mask_path)
        if image.shape[1:] != mask.shape:
            raise ValueError(f"Spatial mismatch for {key}: image {image.shape}, mask {mask.shape}")
        if self.augment:
            image, mask = augment_sample(image, mask, self.cfg)
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # [1, D, H, W]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        return image_tensor, mask_tensor, key


class DilateAttention(nn.Module):
    def __init__(self, in_channel: int, depth: int) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm3d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.atrous_block1 = nn.Conv3d(in_channel, depth, kernel_size=1, stride=1)
        self.atrous_block2 = nn.Conv3d(
            in_channel, depth, kernel_size=3, stride=1, padding=2, dilation=2, groups=in_channel
        )
        self.atrous_block3 = nn.Conv3d(
            in_channel, depth, kernel_size=3, stride=1, padding=4, dilation=4, groups=in_channel
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.atrous_block1(x)
        q = self.atrous_block2(x)
        k = self.atrous_block3(x)
        attention = self.softmax(k * q)
        return v * attention + self.relu(self.batch_norm(x))


class MScrackSeg7(nn.Module):
    """MS-CrackSeg variant for 7 selected bands.

    The 88-band baseline pools along spectral and spatial axes. With only 7
    selected bands, three spectral poolings would collapse the depth dimension.
    This version keeps the spectral axis at 7 and downsamples only H/W, allowing
    controlled comparison while preserving all selected wavelengths.
    """

    def __init__(self, in_channel: int = 1, classnum: int = 1, num_bands: int = 7, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands

        self.conv3d1 = nn.Conv3d(in_channel, dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dilatation1 = DilateAttention(dim, dim)

        self.conv3d2 = nn.Conv3d(dim, dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dilatation2 = DilateAttention(dim, dim)

        self.conv3d3 = nn.Conv3d(dim, dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dilatation3 = DilateAttention(dim, dim)

        self.transpose1 = nn.ConvTranspose3d(dim, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv_up1 = nn.Conv3d(dim, dim, kernel_size=(7, 7, 7), padding=3, stride=1)

        self.transpose2 = nn.ConvTranspose3d(dim * 2, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv_up2 = nn.Conv3d(dim, dim, kernel_size=7, padding=3, stride=1)

        self.transpose3 = nn.ConvTranspose3d(dim * 2, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv_up3 = nn.Conv3d(dim, 1, kernel_size=7, padding=3, stride=1)

        self.final = nn.Conv2d(num_bands, classnum, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv3d1(x)
        x1 = self.maxpooling1(x1)
        x1 = self.dilatation1(x1)

        x2 = self.conv3d2(x1)
        x2 = self.maxpooling2(x2)
        x2 = self.dilatation2(x2)

        x3 = self.conv3d3(x2)
        x3 = self.maxpooling3(x3)
        x3 = self.dilatation3(x3)

        x4 = self.transpose1(x3)
        x4 = self.conv_up1(x4)
        x4 = torch.cat([x4, x2], dim=1)

        x5 = self.transpose2(x4)
        x5 = self.conv_up2(x5)
        x6 = torch.cat([x5, x1], dim=1)

        x6 = self.transpose3(x6)
        x6 = self.conv_up3(x6)
        x6 = torch.squeeze(x6, dim=1)  # [B, D, H, W]
        return self.final(x6)  # [B, 1, H, W]


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, dice_weight: float) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def compute_pos_weight(pairs: Sequence[Tuple[str, Path, Path]], cfg: Config) -> float:
    positives = 0.0
    total = 0.0
    for _, _, mask_path in tqdm(pairs, desc="Computing pos_weight", leave=False):
        mask = load_mask_array(mask_path)
        positives += float(mask.sum())
        total += float(mask.size)
    negatives = max(total - positives, 1.0)
    positives = max(positives, 1.0)
    return float(np.clip(negatives / positives, 1.0, cfg.max_pos_weight))


def learning_rate_for_epoch(epoch: int, cfg: Config) -> float:
    if cfg.warmup_epochs > 0 and epoch <= cfg.warmup_epochs:
        alpha = epoch / float(cfg.warmup_epochs)
        return cfg.warmup_start_lr + alpha * (cfg.learning_rate - cfg.warmup_start_lr)
    if cfg.epochs <= cfg.warmup_epochs:
        return cfg.learning_rate
    progress = (epoch - cfg.warmup_epochs) / float(cfg.epochs - cfg.warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + cosine * (cfg.learning_rate - cfg.min_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def empty_metrics() -> Dict[str, float]:
    return {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}


def update_confusion(stats: Dict[str, float], logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> None:
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    labels = targets >= 0.5
    stats["tp"] += float((preds & labels).sum().item())
    stats["tn"] += float((~preds & ~labels).sum().item())
    stats["fp"] += float((preds & ~labels).sum().item())
    stats["fn"] += float((~preds & labels).sum().item())


def summarize_metrics(stats: Dict[str, float]) -> Dict[str, float]:
    eps = 1.0e-7
    tp, tn, fp, fn = stats["tp"], stats["tn"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    bg_iou = tn / (tn + fp + fn + eps)
    miou = 0.5 * (iou + bg_iou)
    score = 0.5 * (iou + f1)
    return {
        "iou": iou,
        "miou": miou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "score": score,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    cfg: Config,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    use_amp = cfg.use_amp and device.type == "cuda"

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        if cfg.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        running_loss += float(loss.item()) * batch_size
        seen += batch_size
        pbar.set_postfix(loss=running_loss / max(seen, 1))

    return running_loss / max(seen, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
    desc: str,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    seen = 0
    stats = empty_metrics()
    use_amp = cfg.use_amp and device.type == "cuda"

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        update_confusion(stats, logits.detach(), masks.detach(), cfg.threshold)
        batch_size = images.shape[0]
        running_loss += float(loss.item()) * batch_size
        seen += batch_size
        pbar.set_postfix(loss=running_loss / max(seen, 1))

    return running_loss / max(seen, 1), summarize_metrics(stats)


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, [2, 98])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def nearest_band_index(wavelengths: Sequence[float], target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(wavelengths, dtype=np.float32) - target)))


def make_pseudo_rgb(cube: np.ndarray, cfg: Config) -> np.ndarray:
    if cube.ndim == 4:
        cube = cube[0]
    red_idx = nearest_band_index(cfg.selected_wavelengths, 577.8)
    green_idx = nearest_band_index(cfg.selected_wavelengths, 532.7)
    blue_idx = nearest_band_index(cfg.selected_wavelengths, 456.4)
    rgb = np.stack(
        [
            normalize_for_display(cube[red_idx]),
            normalize_for_display(cube[green_idx]),
            normalize_for_display(cube[blue_idx]),
        ],
        axis=-1,
    )
    return rgb


@torch.no_grad()
def save_prediction_figure(model: nn.Module, loader: DataLoader, device: torch.device, cfg: Config, epoch: int) -> None:
    model.eval()
    rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    use_amp = cfg.use_amp and device.type == "cuda"

    for images, masks, keys in loader:
        images_gpu = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images_gpu)
        probs = torch.sigmoid(logits).cpu().numpy()
        images_np = images.numpy()
        masks_np = masks.numpy()
        for i in range(images_np.shape[0]):
            rgb = make_pseudo_rgb(images_np[i], cfg)
            truth = masks_np[i, 0]
            pred = (probs[i, 0] >= cfg.threshold).astype(np.float32)
            rows.append((rgb, truth, pred, str(keys[i])))
            if len(rows) >= cfg.prediction_samples:
                break
        if len(rows) >= cfg.prediction_samples:
            break

    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.2 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (rgb, truth, pred, key) in enumerate(rows):
        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_title(f"Input\n{key}", fontsize=8)
        axes[row_idx, 1].imshow(truth, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].set_title("Ground Truth", fontsize=8)
        axes[row_idx, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].set_title("Prediction", fontsize=8)
        for col in range(3):
            axes[row_idx, col].axis("off")

    fig.tight_layout()
    out_path = cfg.prediction_dir / f"predictions_epoch_{epoch}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def append_metrics_log(path: Path, history: List[Dict[str, float]]) -> None:
    header = [
        "epoch",
        "train_loss",
        "val_loss",
        "iou",
        "miou",
        "precision",
        "recall",
        "f1",
        "score",
        "lr",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in history:
            f.write(
                "\t".join(
                    [
                        str(int(row["epoch"])),
                        f"{row['train_loss']:.8f}",
                        f"{row['val_loss']:.8f}",
                        f"{row['iou']:.8f}",
                        f"{row['miou']:.8f}",
                        f"{row['precision']:.8f}",
                        f"{row['recall']:.8f}",
                        f"{row['f1']:.8f}",
                        f"{row['score']:.8f}",
                        f"{row['lr']:.10f}",
                    ]
                )
                + "\n"
            )


def save_all_curves(history: List[Dict[str, float]], cfg: Config) -> None:
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]

    def values(name: str) -> List[float]:
        return [float(row[name]) for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, values("train_loss"), label="Train Loss", linewidth=1.8)
    ax.plot(epochs, values("val_loss"), label="Validation Loss", linewidth=1.8)
    ax.set_title("Train Loss / Validation Loss")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.figure_dir / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, values("iou"), label="IoU", linewidth=1.8)
    ax.plot(epochs, values("f1"), label="F1", linewidth=1.8)
    ax.plot(epochs, values("score"), label="Score", linewidth=1.8)
    ax.set_title("IoU / F1 / Score")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.figure_dir / "iou_f1_score_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, values("precision"), label="Precision", linewidth=1.8)
    ax.plot(epochs, values("recall"), label="Recall", linewidth=1.8)
    ax.set_title("Precision / Recall")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.figure_dir / "precision_recall_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, values("lr"), label="Learning Rate", linewidth=1.8)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.figure_dir / "lr_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def config_for_save(cfg: Config) -> Dict[str, object]:
    data = asdict(cfg)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
        elif isinstance(value, tuple):
            data[key] = list(value)
    return data


def save_config(path: Path, cfg: Config) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(config_for_save(cfg), f, ensure_ascii=False, indent=2)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_score: float,
    history: List[Dict[str, float]],
    cfg: Config,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
            "config": config_for_save(cfg),
        },
        path,
    )


def load_checkpoint_if_available(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    cfg: Config,
    device: torch.device,
    fresh: bool,
) -> Tuple[int, float, List[Dict[str, float]]]:
    if fresh or not cfg.last_checkpoint_path.is_file():
        return 0, -1.0, []
    checkpoint = torch.load(cfg.last_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])
    start_epoch = int(checkpoint.get("epoch", 0))
    best_score = float(checkpoint.get("best_score", -1.0))
    history = list(checkpoint.get("history", []))
    print(f"Resumed from {cfg.last_checkpoint_path} at epoch {start_epoch}, best_score={best_score:.6f}")
    return start_epoch, best_score, history


def make_loader(dataset: Dataset, cfg: Config, shuffle: bool, generator: torch.Generator | None = None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
        generator=generator,
        persistent_workers=cfg.num_workers > 0,
    )


def write_final_test_metrics(path: Path, test_loss: float, metrics: Dict[str, float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"test_loss\t{test_loss:.8f}\n")
        for key in ["iou", "miou", "precision", "recall", "f1", "score"]:
            f.write(f"{key}\t{metrics[key]:.8f}\n")


def print_dataset_summary(
    train_pairs: Sequence[Tuple[str, Path, Path]],
    val_pairs: Sequence[Tuple[str, Path, Path]],
    test_pairs: Sequence[Tuple[str, Path, Path]],
    cfg: Config,
) -> None:
    train_keys = {key for key, _, _ in train_pairs}
    val_keys = {key for key, _, _ in val_pairs}
    test_keys = {key for key, _, _ in test_pairs}
    train_test_overlap = (train_keys | val_keys) & test_keys
    train_val_overlap = train_keys & val_keys
    print("=" * 72)
    print("Dataset summary")
    print(f"Selected wavelengths (nm): {', '.join(str(v) for v in cfg.selected_wavelengths)}")
    print(f"Train samples: {len(train_pairs)}")
    print(f"Val samples:   {len(val_pairs)}")
    print(f"Test samples:  {len(test_pairs)}")
    print(f"Train/Val overlap:  {len(train_val_overlap)}")
    print(f"Train+Val/Test overlap: {len(train_test_overlap)}")
    print("=" * 72)
    if train_val_overlap:
        raise RuntimeError(f"Train/val leakage detected: {sorted(train_val_overlap)[:5]}")
    if train_test_overlap:
        raise RuntimeError(f"Train/test leakage detected: {sorted(train_test_overlap)[:5]}")


def main() -> None:
    args = parse_args()
    cfg = CFG
    apply_args(cfg, args)
    resolve_dataset_paths(cfg)
    ensure_dirs(cfg)
    save_config(cfg.log_dir / "config.json", cfg)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    all_train_pairs = pair_files(cfg.train_img_dir, cfg.train_mask_dir)
    test_pairs = pair_files(cfg.test_img_dir, cfg.test_mask_dir)
    validate_shapes(all_train_pairs, cfg, "train/val")
    validate_shapes(test_pairs, cfg, "test")

    train_pairs, val_pairs = split_train_val(all_train_pairs, cfg.val_ratio, cfg.seed)
    print_dataset_summary(train_pairs, val_pairs, test_pairs, cfg)

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    train_dataset = CrackNpyDataset(train_pairs, cfg, augment=True)
    val_dataset = CrackNpyDataset(val_pairs, cfg, augment=False)
    test_dataset = CrackNpyDataset(test_pairs, cfg, augment=False)

    train_loader = make_loader(train_dataset, cfg, shuffle=True, generator=generator)
    val_loader = make_loader(val_dataset, cfg, shuffle=False)
    test_loader = make_loader(test_dataset, cfg, shuffle=False)

    pos_weight_value = compute_pos_weight(train_pairs, cfg)
    print(f"Using pos_weight={pos_weight_value:.4f}")
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    model = MScrackSeg7(in_channel=1, classnum=1, num_bands=cfg.num_bands).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and device.type == "cuda")
    criterion = CombinedLoss(pos_weight=pos_weight, dice_weight=cfg.dice_loss_weight)

    start_epoch, best_score, history = load_checkpoint_if_available(
        model, optimizer, scaler, cfg, device, fresh=args.fresh
    )
    metrics_log_path = cfg.log_dir / "metrics_log.txt"

    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        lr = learning_rate_for_epoch(epoch, cfg)
        set_optimizer_lr(optimizer, lr)

        print(f"\nEpoch {epoch}/{cfg.epochs} | lr={lr:.8f}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, cfg, desc="Validate")

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(lr),
            **{k: float(v) for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            "Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f} | "
            "IoU={iou:.4f} mIoU={miou:.4f} F1={f1:.4f} "
            "Precision={precision:.4f} Recall={recall:.4f} Score={score:.4f}".format(
                train_loss=train_loss, val_loss=val_loss, **val_metrics
            )
        )

        append_metrics_log(metrics_log_path, history)
        save_checkpoint(cfg.last_checkpoint_path, model, optimizer, scaler, epoch, best_score, history, cfg)

        if val_metrics["score"] > best_score:
            best_score = float(val_metrics["score"])
            torch.save(
                {
                    "epoch": epoch,
                    "best_score": best_score,
                    "model_state": model.state_dict(),
                    "config": config_for_save(cfg),
                },
                cfg.best_model_path,
            )
            save_checkpoint(cfg.last_checkpoint_path, model, optimizer, scaler, epoch, best_score, history, cfg)
            print(f"Saved new best model to {cfg.best_model_path} (score={best_score:.6f})")

        if epoch % cfg.prediction_interval == 0 or epoch == cfg.epochs:
            save_prediction_figure(model, val_loader, device, cfg, epoch)

    save_all_curves(history, cfg)

    if cfg.best_model_path.is_file():
        best_checkpoint = torch.load(cfg.best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state"])
        print(f"\nLoaded best model from epoch {best_checkpoint.get('epoch', 'unknown')}")
    else:
        print("\nBest model file was not found; evaluating current model.")

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device, cfg, desc="Final Test")
    write_final_test_metrics(cfg.log_dir / "final_test_metrics.txt", test_loss, test_metrics)
    print(
        "Final Test | Loss={loss:.6f} | IoU={iou:.4f} mIoU={miou:.4f} F1={f1:.4f} "
        "Precision={precision:.4f} Recall={recall:.4f} Score={score:.4f}".format(
            loss=test_loss, **test_metrics
        )
    )
    print(f"All outputs saved under: {cfg.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", file=sys.stderr)
        raise
