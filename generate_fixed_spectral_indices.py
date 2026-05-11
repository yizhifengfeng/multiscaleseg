#!/usr/bin/env python3
"""
Generate fixed spectral-index datasets for hyperspectral concrete crack data.

Input dataset layout:
  sampleset/img/*.npy         hyperspectral cubes, shape (176, H, W) or (H, W, 176)
  sampleset/masknpy/*.npy     binary masks
  sampleset/testimg/*.npy
  sampleset/testmasknpy/*.npy

Output layout:
  fixed_spectral_indices/img/*.npy       shape (5, H, W), order below
  fixed_spectral_indices/masknpy/*.npy
  fixed_spectral_indices/testimg/*.npy
  fixed_spectral_indices/testmasknpy/*.npy
  fixed_spectral_indices/previews/*.png

Indices:
  GNDVI = (R800 - R550) / (R800 + R550)
  NDVI  = (R800 - R670) / (R800 + R670)
  NDWI  = (R550 - R800) / (R550 + R800), McFeeters style for available VNIR bands
  NPCI  = (R680 - R430) / (R680 + R430)
  PRI   = (R531 - R570) / (R531 + R570)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


INDEX_ORDER = ["GNDVI", "NDVI", "NDWI", "NPCI", "PRI"]


def wavelength_axis(num_bands: int = 176, wl_min: float = 394.0, wl_max: float = 1001.0) -> np.ndarray:
    return np.linspace(wl_min, wl_max, num_bands, dtype=np.float32)


def nearest_band(target_nm: float, wavelengths: np.ndarray) -> int:
    return int(np.argmin(np.abs(wavelengths - target_nm)))


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


def normalized_difference(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = (a - b) / (a + b + eps)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32)


def build_index_specs(wavelengths: np.ndarray) -> Dict[str, Tuple[int, int, str]]:
    """Return index name -> (positive band, negative band, formula text)."""
    b430 = nearest_band(430.0, wavelengths)
    b531 = nearest_band(531.0, wavelengths)
    b550 = nearest_band(550.0, wavelengths)
    b570 = nearest_band(570.0, wavelengths)
    b670 = nearest_band(670.0, wavelengths)
    b680 = nearest_band(680.0, wavelengths)
    b800 = nearest_band(800.0, wavelengths)
    return {
        "GNDVI": (b800, b550, "(R800 - R550) / (R800 + R550)"),
        "NDVI": (b800, b670, "(R800 - R670) / (R800 + R670)"),
        "NDWI": (b550, b800, "(R550 - R800) / (R550 + R800)"),
        "NPCI": (b680, b430, "(R680 - R430) / (R680 + R430)"),
        "PRI": (b531, b570, "(R531 - R570) / (R531 + R570)"),
    }


def compute_indices(cube: np.ndarray, specs: Dict[str, Tuple[int, int, str]]) -> np.ndarray:
    maps: List[np.ndarray] = []
    for name in INDEX_ORDER:
        pos_band, neg_band, _ = specs[name]
        maps.append(normalized_difference(cube[pos_band], cube[neg_band]))
    return np.stack(maps, axis=0).astype(np.float32)


def robust_uint8(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [2.0, 98.0])
    if hi <= lo:
        lo, hi = float(np.min(image)), float(np.max(image))
    scaled = (image - lo) / (hi - lo + 1e-6)
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)


def save_preview(stack: np.ndarray, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(INDEX_ORDER), figsize=(14, 3.2), constrained_layout=True)
    for ax, name, image in zip(axes, INDEX_ORDER, stack):
        ax.imshow(robust_uint8(image), cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def copy_masks(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.glob("*.npy")):
        shutil.copy2(src, dst_dir / src.name)


def process_split(
    src_img_dir: Path,
    src_mask_dir: Path,
    dst_img_dir: Path,
    dst_mask_dir: Path,
    preview_dir: Path,
    split_name: str,
    specs: Dict[str, Tuple[int, int, str]],
    num_bands: int,
    preview_count: int,
    save_all_png: bool,
) -> int:
    if not src_img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {src_img_dir}")
    if not src_mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {src_mask_dir}")

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    copy_masks(src_mask_dir, dst_mask_dir)
    files = sorted(src_img_dir.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy files found in {src_img_dir}")

    split_preview_dir = preview_dir / split_name
    split_preview_dir.mkdir(parents=True, exist_ok=True)

    for idx, path in enumerate(tqdm(files, desc=f"Generating {split_name} indices")):
        cube = load_cube(path, num_bands)
        stack = compute_indices(cube, specs)
        np.save(dst_img_dir / path.name, stack)

        if save_all_png or idx < preview_count:
            save_preview(stack, split_preview_dir / f"{path.stem}_indices.png", path.stem)
    return len(files)


def write_metadata(out_dir: Path, specs: Dict[str, Tuple[int, int, str]], wavelengths: np.ndarray) -> None:
    metadata = {
        "index_order": INDEX_ORDER,
        "num_output_channels": len(INDEX_ORDER),
        "source_wavelength_range_nm": [float(wavelengths[0]), float(wavelengths[-1])],
        "bands": {},
    }
    for name in INDEX_ORDER:
        pos_band, neg_band, formula = specs[name]
        metadata["bands"][name] = {
            "formula": formula,
            "positive_band_index": int(pos_band),
            "positive_band_wavelength_nm": float(wavelengths[pos_band]),
            "negative_band_index": int(neg_band),
            "negative_band_wavelength_nm": float(wavelengths[neg_band]),
        }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed VNIR spectral-index maps.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("sampleset"))
    parser.add_argument("--output-dir", type=Path, default=Path("fixed_spectral_indices"))
    parser.add_argument("--num-bands", type=int, default=176)
    parser.add_argument("--wl-min", type=float, default=394.0)
    parser.add_argument("--wl-max", type=float, default=1001.0)
    parser.add_argument("--preview-count", type=int, default=20, help="Number of preview mosaics per split.")
    parser.add_argument("--save-all-png", action="store_true", help="Save preview PNGs for every sample.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wavelengths = wavelength_axis(args.num_bands, args.wl_min, args.wl_max)
    specs = build_index_specs(wavelengths)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(args.output_dir, specs, wavelengths)

    print("Fixed index band mapping:")
    for name in INDEX_ORDER:
        pos, neg, formula = specs[name]
        print(
            f"  {name:5s}: band {pos:3d} ({wavelengths[pos]:7.2f} nm), "
            f"band {neg:3d} ({wavelengths[neg]:7.2f} nm), {formula}"
        )

    n_train = process_split(
        args.dataset_dir / "img",
        args.dataset_dir / "masknpy",
        args.output_dir / "img",
        args.output_dir / "masknpy",
        args.output_dir / "previews",
        "train",
        specs,
        args.num_bands,
        args.preview_count,
        args.save_all_png,
    )
    n_test = process_split(
        args.dataset_dir / "testimg",
        args.dataset_dir / "testmasknpy",
        args.output_dir / "testimg",
        args.output_dir / "testmasknpy",
        args.output_dir / "previews",
        "test",
        specs,
        args.num_bands,
        args.preview_count,
        args.save_all_png,
    )

    print(f"\nDone. Wrote {n_train} train and {n_test} test index stacks to: {args.output_dir}")
    print("Each image array has shape (5, H, W) in order:", ", ".join(INDEX_ORDER))


if __name__ == "__main__":
    main()
