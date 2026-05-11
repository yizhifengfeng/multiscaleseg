#!/usr/bin/env python3
"""
Screen sliding-window normalized-difference spectral indices for crack detection.

For each candidate band pair (i, j), this script evaluates:
  1. information entropy of the index map,
  2. crack/background contrast using the binary crack mask,
  3. edge strength using mean absolute spatial gradients.

It writes a ranked CSV and exports the Top-K selected index maps as a new
multi-channel dataset that can be fed to a PyTorch segmentation script.

Default output:
  spectral_index_selection/candidate_index_scores.csv
  spectral_index_selection/selected_indices_metadata.json
  spectral_index_selection/img/*.npy       shape (top_k, H, W)
  spectral_index_selection/masknpy/*.npy
  spectral_index_selection/testimg/*.npy
  spectral_index_selection/testmasknpy/*.npy
  spectral_index_selection/previews/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


KEY_PATTERN = re.compile(r"(r\d+c\d+col\d+row\d+overlap\d+)", re.IGNORECASE)


def wavelength_axis(num_bands: int = 176, wl_min: float = 394.0, wl_max: float = 1001.0) -> np.ndarray:
    return np.linspace(wl_min, wl_max, num_bands, dtype=np.float32)


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

    img_by_key = {extract_key(p): p for p in sorted(img_dir.glob("*.npy"))}
    mask_by_key = {extract_key(p): p for p in sorted(mask_dir.glob("*.npy"))}
    if not img_by_key:
        raise RuntimeError(f"No image .npy files found in {img_dir}")
    if not mask_by_key:
        raise RuntimeError(f"No mask .npy files found in {mask_dir}")

    missing_masks = sorted(set(img_by_key) - set(mask_by_key))
    missing_imgs = sorted(set(mask_by_key) - set(img_by_key))
    if missing_masks or missing_imgs:
        raise RuntimeError(
            f"Image/mask mismatch. Missing masks: {missing_masks[:5]}, missing images: {missing_imgs[:5]}"
        )
    return [(key, img_by_key[key], mask_by_key[key]) for key in sorted(img_by_key)]


def load_cube(path: Path, num_bands: int) -> np.ndarray:
    arr = np.load(path).astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 4 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"{path.name}: expected a 3D cube, got {arr.shape}")
    if arr.shape[0] == num_bands:
        return np.ascontiguousarray(arr)
    if arr.shape[-1] == num_bands:
        return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))
    raise ValueError(f"{path.name}: cannot find {num_bands} spectral bands in shape {arr.shape}")


def load_mask(path: Path) -> np.ndarray:
    arr = np.squeeze(np.load(path))
    if arr.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2D mask after squeeze, got {arr.shape}")
    return (arr > 0).astype(bool)


def normalized_difference(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = (a - b) / (a + b + eps)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32)


def make_candidate_pairs(
    num_bands: int,
    window_size: int,
    window_stride: int,
    pair_stride: int,
    min_gap: int,
    max_gap: int,
    exhaustive: bool,
) -> List[Tuple[int, int]]:
    """Create candidate band pairs from a sliding spectral window."""
    if min_gap < 1:
        raise ValueError("min_gap must be >= 1")
    if max_gap < min_gap:
        raise ValueError("max_gap must be >= min_gap")

    pairs = set()
    if exhaustive:
        starts = [0]
        window_size = num_bands
        window_stride = num_bands
    else:
        starts = range(0, num_bands, max(1, window_stride))

    for start in starts:
        end = min(start + window_size, num_bands)
        for i in range(start, end, max(1, pair_stride)):
            low = i + min_gap
            high = min(end, i + max_gap + 1)
            for j in range(low, high, max(1, pair_stride)):
                if 0 <= i < j < num_bands:
                    pairs.add((i, j))
    return sorted(pairs)


def entropy_01(values: np.ndarray, bins: int) -> float:
    hist, _ = np.histogram(values, bins=bins, range=(-1.0, 1.0))
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist.astype(np.float64) / float(total)
    p = p[p > 0.0]
    return float(-np.sum(p * np.log2(p)) / math.log2(bins))


def minmax(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def evaluate_candidates(
    pairs: Sequence[Tuple[int, int]],
    sample_pairs: Sequence[Tuple[str, Path, Path]],
    num_bands: int,
    spatial_stride: int,
    chunk_size: int,
    entropy_bins: int,
) -> Dict[str, np.ndarray]:
    n_candidates = len(pairs)
    a_idx = np.array([p[0] for p in pairs], dtype=np.int32)
    b_idx = np.array([p[1] for p in pairs], dtype=np.int32)

    entropy_sum = np.zeros(n_candidates, dtype=np.float64)
    edge_sum = np.zeros(n_candidates, dtype=np.float64)
    contrast_sum = np.zeros(n_candidates, dtype=np.float64)
    mean_diff_sum = np.zeros(n_candidates, dtype=np.float64)
    contrast_count = np.zeros(n_candidates, dtype=np.float64)

    for _, img_path, mask_path in tqdm(sample_pairs, desc="Scoring candidate index pairs"):
        cube = load_cube(img_path, num_bands)[:, ::spatial_stride, ::spatial_stride]
        mask = load_mask(mask_path)[::spatial_stride, ::spatial_stride]
        if cube.shape[1:] != mask.shape:
            raise ValueError(f"Spatial mismatch for {img_path.name}: cube {cube.shape}, mask {mask.shape}")

        mask_flat = mask.reshape(-1)
        has_two_classes = bool(mask_flat.any() and (~mask_flat).any())

        for start in range(0, n_candidates, chunk_size):
            stop = min(start + chunk_size, n_candidates)
            chunk_a = a_idx[start:stop]
            chunk_b = b_idx[start:stop]
            idx_maps = normalized_difference(cube[chunk_a], cube[chunk_b])
            flat = idx_maps.reshape(idx_maps.shape[0], -1)

            for local_i, values in enumerate(flat):
                entropy_sum[start + local_i] += entropy_01(values, entropy_bins)

            gy = np.abs(np.diff(idx_maps, axis=1)).mean(axis=(1, 2))
            gx = np.abs(np.diff(idx_maps, axis=2)).mean(axis=(1, 2))
            edge_sum[start:stop] += 0.5 * (gx + gy)

            if has_two_classes:
                crack_values = flat[:, mask_flat]
                bg_values = flat[:, ~mask_flat]
                crack_mean = crack_values.mean(axis=1)
                bg_mean = bg_values.mean(axis=1)
                crack_std = crack_values.std(axis=1)
                bg_std = bg_values.std(axis=1)
                mean_diff = crack_mean - bg_mean
                contrast = np.abs(mean_diff) / (crack_std + bg_std + 1e-6)
                contrast_sum[start:stop] += contrast
                mean_diff_sum[start:stop] += mean_diff
                contrast_count[start:stop] += 1.0

    n_images = max(1, len(sample_pairs))
    safe_contrast_count = np.maximum(contrast_count, 1.0)
    return {
        "band_a": a_idx,
        "band_b": b_idx,
        "entropy": entropy_sum / n_images,
        "edge_strength": edge_sum / n_images,
        "crack_background_contrast": contrast_sum / safe_contrast_count,
        "mean_crack_minus_background": mean_diff_sum / safe_contrast_count,
        "valid_contrast_fraction": contrast_count / n_images,
    }


def build_ranked_rows(metrics: Dict[str, np.ndarray], wavelengths: np.ndarray, weights: Tuple[float, float, float]) -> List[Dict[str, float]]:
    entropy_n = minmax(metrics["entropy"])
    contrast_n = minmax(metrics["crack_background_contrast"])
    edge_n = minmax(metrics["edge_strength"])
    valid_n = metrics["valid_contrast_fraction"]
    w_entropy, w_contrast, w_edge = weights
    score = (w_entropy * entropy_n + w_contrast * contrast_n + w_edge * edge_n) * valid_n

    rows: List[Dict[str, float]] = []
    for idx in range(len(score)):
        a = int(metrics["band_a"][idx])
        b = int(metrics["band_b"][idx])
        mean_diff = float(metrics["mean_crack_minus_background"][idx])
        orient_sign = 1 if mean_diff >= 0.0 else -1
        rows.append(
            {
                "rank": 0,
                "score": float(score[idx]),
                "band_a": a,
                "wavelength_a_nm": float(wavelengths[a]),
                "band_b": b,
                "wavelength_b_nm": float(wavelengths[b]),
                "orientation_sign": orient_sign,
                "entropy": float(metrics["entropy"][idx]),
                "crack_background_contrast": float(metrics["crack_background_contrast"][idx]),
                "edge_strength": float(metrics["edge_strength"][idx]),
                "mean_crack_minus_background": mean_diff,
                "valid_contrast_fraction": float(metrics["valid_contrast_fraction"][idx]),
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def write_csv(rows: Sequence[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_masks(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.glob("*.npy")):
        shutil.copy2(src, dst_dir / src.name)


def robust_uint8(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [2.0, 98.0])
    if hi <= lo:
        lo, hi = float(np.min(image)), float(np.max(image))
    scaled = (image - lo) / (hi - lo + 1e-6)
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)


def save_preview(stack: np.ndarray, selected_rows: Sequence[Dict[str, float]], out_path: Path, title: str) -> None:
    max_channels = min(stack.shape[0], 12)
    cols = min(4, max_channels)
    rows = int(math.ceil(max_channels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(-1)
    for ax_i, ax in enumerate(axes_arr):
        ax.axis("off")
        if ax_i >= max_channels:
            continue
        meta = selected_rows[ax_i]
        ax.imshow(robust_uint8(stack[ax_i]), cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"#{meta['rank']} b{meta['band_a']}-b{meta['band_b']}\nscore={meta['score']:.3f}",
            fontsize=8,
        )
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def compute_selected_stack(cube: np.ndarray, selected_rows: Sequence[Dict[str, float]]) -> np.ndarray:
    maps = []
    for row in selected_rows:
        a = int(row["band_a"])
        b = int(row["band_b"])
        sign = int(row["orientation_sign"])
        maps.append(sign * normalized_difference(cube[a], cube[b]))
    return np.stack(maps, axis=0).astype(np.float32)


def export_selected_dataset(
    selected_rows: Sequence[Dict[str, float]],
    dataset_dir: Path,
    output_dir: Path,
    num_bands: int,
    preview_count: int,
) -> None:
    split_specs = [
        ("train", dataset_dir / "img", dataset_dir / "masknpy", output_dir / "img", output_dir / "masknpy"),
        ("test", dataset_dir / "testimg", dataset_dir / "testmasknpy", output_dir / "testimg", output_dir / "testmasknpy"),
    ]

    for split_name, src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir in split_specs:
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        copy_masks(src_mask_dir, dst_mask_dir)
        files = sorted(src_img_dir.glob("*.npy"))
        preview_dir = output_dir / "previews" / split_name
        preview_dir.mkdir(parents=True, exist_ok=True)

        for idx, path in enumerate(tqdm(files, desc=f"Exporting selected {split_name} maps")):
            cube = load_cube(path, num_bands)
            stack = compute_selected_stack(cube, selected_rows)
            np.save(dst_img_dir / path.name, stack)
            if idx < preview_count:
                save_preview(stack, selected_rows, preview_dir / f"{path.stem}_selected.png", path.stem)


def write_metadata(
    selected_rows: Sequence[Dict[str, float]],
    output_dir: Path,
    args: argparse.Namespace,
    wavelengths: np.ndarray,
    candidate_count: int,
    scored_sample_count: int,
) -> None:
    channels = []
    for channel, row in enumerate(selected_rows):
        sign = int(row["orientation_sign"])
        a = int(row["band_a"])
        b = int(row["band_b"])
        pos, neg = (a, b) if sign > 0 else (b, a)
        channels.append(
            {
                "channel": channel,
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "formula": f"(R{wavelengths[pos]:.2f} - R{wavelengths[neg]:.2f}) / "
                f"(R{wavelengths[pos]:.2f} + R{wavelengths[neg]:.2f})",
                "positive_band_index": int(pos),
                "positive_wavelength_nm": float(wavelengths[pos]),
                "negative_band_index": int(neg),
                "negative_wavelength_nm": float(wavelengths[neg]),
                "entropy": float(row["entropy"]),
                "crack_background_contrast": float(row["crack_background_contrast"]),
                "edge_strength": float(row["edge_strength"]),
            }
        )

    metadata = {
        "num_output_channels": len(selected_rows),
        "candidate_count": candidate_count,
        "scored_sample_count": scored_sample_count,
        "wavelength_range_nm": [float(wavelengths[0]), float(wavelengths[-1])],
        "selection_settings": {
            "window_size": args.window_size,
            "window_stride": args.window_stride,
            "pair_stride": args.pair_stride,
            "min_gap": args.min_gap,
            "max_gap": args.max_gap,
            "exhaustive": args.exhaustive,
            "spatial_stride": args.spatial_stride,
            "entropy_bins": args.entropy_bins,
            "weights_entropy_contrast_edge": [args.weight_entropy, args.weight_contrast, args.weight_edge],
        },
        "channel_order": channels,
    }
    (output_dir / "selected_indices_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen sliding-window spectral-index band pairs.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("sampleset"))
    parser.add_argument("--output-dir", type=Path, default=Path("spectral_index_selection"))
    parser.add_argument("--num-bands", type=int, default=176)
    parser.add_argument("--wl-min", type=float, default=394.0)
    parser.add_argument("--wl-max", type=float, default=1001.0)
    parser.add_argument("--window-size", type=int, default=64, help="Sliding spectral window size in bands.")
    parser.add_argument("--window-stride", type=int, default=4, help="Sliding spectral window stride in bands.")
    parser.add_argument("--pair-stride", type=int, default=1, help="Stride used when forming pairs inside each window.")
    parser.add_argument("--min-gap", type=int, default=1, help="Minimum distance between two bands.")
    parser.add_argument("--max-gap", type=int, default=175, help="Maximum distance between two bands.")
    parser.add_argument("--exhaustive", action="store_true", help="Evaluate all i<j band pairs.")
    parser.add_argument("--sample-limit", type=int, default=80, help="Number of train samples used for scoring; 0 means all.")
    parser.add_argument("--include-test-in-scoring", action="store_true")
    parser.add_argument("--spatial-stride", type=int, default=4, help="Spatial downsample stride used only for scoring.")
    parser.add_argument("--chunk-size", type=int, default=256, help="Number of candidate pairs evaluated at once.")
    parser.add_argument("--entropy-bins", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=7, help="Number of selected index channels to export.")
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-entropy", type=float, default=0.25)
    parser.add_argument("--weight-contrast", type=float, default=0.50)
    parser.add_argument("--weight-edge", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.spatial_stride < 1:
        raise ValueError("spatial-stride must be >= 1")
    if args.top_k < 1:
        raise ValueError("top-k must be >= 1")

    wavelengths = wavelength_axis(args.num_bands, args.wl_min, args.wl_max)
    all_candidates = make_candidate_pairs(
        args.num_bands,
        args.window_size,
        args.window_stride,
        args.pair_stride,
        args.min_gap,
        args.max_gap,
        args.exhaustive,
    )
    if not all_candidates:
        raise RuntimeError("No candidate band pairs were generated. Check window/gap settings.")

    train_pairs = pair_files(args.dataset_dir / "img", args.dataset_dir / "masknpy")
    scoring_pairs = list(train_pairs)
    if args.include_test_in_scoring:
        scoring_pairs += pair_files(args.dataset_dir / "testimg", args.dataset_dir / "testmasknpy")

    rng = np.random.default_rng(args.seed)
    if args.sample_limit > 0 and len(scoring_pairs) > args.sample_limit:
        chosen = rng.choice(len(scoring_pairs), size=args.sample_limit, replace=False)
        scoring_pairs = [scoring_pairs[int(i)] for i in sorted(chosen)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Candidate band pairs: {len(all_candidates)}")
    print(f"Samples used for scoring: {len(scoring_pairs)}")
    print(f"Output directory: {args.output_dir}")

    metrics = evaluate_candidates(
        all_candidates,
        scoring_pairs,
        args.num_bands,
        args.spatial_stride,
        args.chunk_size,
        args.entropy_bins,
    )
    rows = build_ranked_rows(
        metrics,
        wavelengths,
        (args.weight_entropy, args.weight_contrast, args.weight_edge),
    )
    write_csv(rows, args.output_dir / "candidate_index_scores.csv")

    selected_rows = rows[: min(args.top_k, len(rows))]
    write_metadata(selected_rows, args.output_dir, args, wavelengths, len(all_candidates), len(scoring_pairs))
    export_selected_dataset(selected_rows, args.dataset_dir, args.output_dir, args.num_bands, args.preview_count)

    print("\nTop selected spectral indices:")
    for row in selected_rows:
        sign = int(row["orientation_sign"])
        a = int(row["band_a"])
        b = int(row["band_b"])
        pos, neg = (a, b) if sign > 0 else (b, a)
        print(
            f"  #{int(row['rank']):02d} score={row['score']:.4f} "
            f"(R{wavelengths[pos]:.1f}nm - R{wavelengths[neg]:.1f}nm) / "
            f"(R{wavelengths[pos]:.1f}nm + R{wavelengths[neg]:.1f}nm), "
            f"contrast={row['crack_background_contrast']:.4f}, "
            f"entropy={row['entropy']:.4f}, edge={row['edge_strength']:.4f}"
        )
    print(f"\nDone. Ranked CSV and selected Top-{len(selected_rows)} dataset are in: {args.output_dir}")


if __name__ == "__main__":
    main()
