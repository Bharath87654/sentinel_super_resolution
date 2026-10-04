"""
Real Sentinel-2 Patch Extractor for 4× Deployment Inference

Scientific Scope
────────────────
This script extracts a REAL 10m Sentinel-2 patch directly from a raw .SAFE
Level-2A product directory. The resulting (4,64,64) patch represents a
640m × 640m spatial extent.

This patch serves as the input to deployment inference (infer_4x.py),
which will output a 2.5m-scale INFERRED product.

IMPORTANT: The eventual 2.5m output will be model-inferred, not ground-truth
2.5m imagery. Any sub-10m spatial detail is synthesized based on the
synthetic 40m-equivalent LR4x → 10m HR training benchmark.

Isolation Guarantee
────────────────────
Reads from the raw .SAFE directory only. Does NOT modify it.
Does NOT use synthetic LR4x data.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

# ─── Configuration ────────────────────────────────────────────────────────────
BANDS = ["B02", "B03", "B04", "B08"]
RESOLUTION_M = 10
PATCH_SIZE = 64
EXPECTED_SHAPE = (4, PATCH_SIZE, PATCH_SIZE)
RGB_INDICES = [2, 1, 0]  # R=B04, G=B03, B=B02


def _apply_stretch(arr_hwc: np.ndarray) -> np.ndarray:
    """Apply 2nd-98th percentile linear stretch and clip to [0,1]."""
    lo = float(np.percentile(arr_hwc, 2.0))
    hi = float(np.percentile(arr_hwc, 98.0))
    return np.clip((arr_hwc - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def extract_patch():
    parser = argparse.ArgumentParser(
        description="Extract a real Sentinel-2 10m patch for 4× inference."
    )
    parser.add_argument(
        "--safe_dir", type=str, required=True,
        help="Path to an actual Sentinel-2 .SAFE Level-2A directory",
    )
    parser.add_argument(
        "--row", type=int, required=True,
        help="Starting row offset in the 10m raster",
    )
    parser.add_argument(
        "--col", type=int, required=True,
        help="Starting column offset in the 10m raster",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to save the output .npy patch",
    )
    args = parser.parse_args()

    safe_dir = Path(args.safe_dir)
    out_npy_path = Path(args.output)
    out_dir = out_npy_path.parent
    stem = out_npy_path.stem

    if not safe_dir.exists() or not safe_dir.is_dir():
        raise FileNotFoundError(f"SAFE directory not found: {safe_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── Locate Band Files ────────────────────────────────────────────────────
    band_paths = {}
    for band in BANDS:
        pattern = f"GRANULE/*/IMG_DATA/*_{band}_{RESOLUTION_M}m.jp2"
        matches = list(safe_dir.glob(pattern))

        # Some SAFE versions place 10m bands in R10m subdirectory
        if not matches:
            pattern_r10m = f"GRANULE/*/IMG_DATA/R{RESOLUTION_M}m/*_{band}_{RESOLUTION_M}m.jp2"
            matches = list(safe_dir.glob(pattern_r10m))

        if len(matches) == 0:
            raise FileNotFoundError(
                f"Could not find 10m {band} file matching pattern {pattern} in {safe_dir}"
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple matches found for {band} 10m in {safe_dir}. Expected one."
            )
        band_paths[band] = matches[0]

    print("=" * 68)
    print("REAL SENTINEL-2 PATCH EXTRACTION (DEPLOYMENT INFERENCE)")
    print("=" * 68)
    print(f"SAFE Directory : {safe_dir.name}")
    for b in BANDS:
        print(f"{b:4s} Path      : {band_paths[b].relative_to(safe_dir)}")

    # ─── Reading and Validation ───────────────────────────────────────────────
    patch_data = []
    base_crs = None
    base_transform = None
    base_width = None
    base_height = None
    patch_transform = None

    window = Window(args.col, args.row, PATCH_SIZE, PATCH_SIZE)

    for i, band in enumerate(BANDS):
        with rasterio.open(band_paths[band]) as src:
            if i == 0:
                base_crs = src.crs
                base_transform = src.transform
                base_width = src.width
                base_height = src.height
                patch_transform = src.window_transform(window)
            else:
                if src.crs != base_crs:
                    raise ValueError(f"CRS mismatch in {band}: {src.crs} != {base_crs}")
                if src.transform != base_transform:
                    raise ValueError(f"Transform mismatch in {band}")
                if src.width != base_width or src.height != base_height:
                    raise ValueError(f"Raster dimensions mismatch in {band}")

            # Verify bounds
            if (args.row < 0 or args.col < 0 or
                    args.row + PATCH_SIZE > base_width or
                    args.col + PATCH_SIZE > base_height):
                raise ValueError(
                    f"Requested window (row={args.row}, col={args.col}, size={PATCH_SIZE}) "
                    f"is out of bounds for raster ({base_height}, {base_width})."
                )

            # Read raw uint16 window
            arr = src.read(1, window=window)
            patch_data.append(arr)

    patch_np = np.stack(patch_data, axis=0)

    # ─── Patch Validation ─────────────────────────────────────────────────────
    assert patch_np.shape == EXPECTED_SHAPE, f"Shape mismatch: {patch_np.shape}"
    assert patch_np.dtype == np.uint16, f"Dtype mismatch: {patch_np.dtype}"

    if np.isnan(patch_np).any() or np.isinf(patch_np).any():
        raise ValueError("Extracted patch contains NaN or Inf values.")

    if np.all(patch_np == 0):
        raise ValueError("Extracted patch is entirely zero.")

    print("\nPatch Geometry Validation:")
    print(f"  CRS            : {base_crs}")
    print(f"  Raster Size    : {base_width} × {base_height}")
    print(f"  Patch Row/Col  : {args.row}, {args.col}")
    print(f"  Patch Shape    : {patch_np.shape}")
    print(f"  Patch Dtype    : {patch_np.dtype}")

    print("\nBand Statistics (Raw uint16):")
    print(f"  {'Band':<4} | {'Min':>5} | {'Max':>5} | {'Mean':>8}")
    print("  " + "-" * 32)
    for i, band in enumerate(BANDS):
        b_arr = patch_np[i]
        print(f"  {band:<4} | {int(b_arr.min()):>5} | {int(b_arr.max()):>5} | {b_arr.mean():>8.2f}")

    # ─── Save NumPy Patch ─────────────────────────────────────────────────────
    np.save(out_npy_path, patch_np)
    print(f"\nSaved patch                : {out_npy_path}")

    # ─── Save Metadata JSON ───────────────────────────────────────────────────
    meta = {
        "source_safe": safe_dir.name,
        "row_off": args.row,
        "col_off": args.col,
        "crs": str(base_crs),
        "orig_transform": list(base_transform),
        "patch_transform": list(patch_transform),
        "bands": BANDS,
        "dtype": str(patch_np.dtype),
        "shape": list(patch_np.shape),
        "pixel_size_m": RESOLUTION_M,
    }

    out_meta_path = out_dir / f"{stem}_metadata.json"
    with open(out_meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata             : {out_meta_path}")

    # ─── Save Optional RGB Preview ────────────────────────────────────────────
    # Convert uint16 -> reflectances roughly mapped to [0,1] for display stretch calculation
    rgb_chw = patch_np[RGB_INDICES, :, :].astype(np.float32) / 10000.0
    rgb_hwc = rgb_chw.transpose(1, 2, 0)
    rgb_disp = _apply_stretch(rgb_hwc)

    out_rgb_path = out_dir / f"{stem}_rgb.png"
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_disp, interpolation="nearest")
    plt.title(f"Real Sentinel-2 Input (10m)\nRGB (B04-B03-B02)")
    plt.axis("off")
    plt.savefig(out_rgb_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved RGB preview (10m)    : {out_rgb_path}")

    # ─── Final Scientific Note ────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("SCIENTIFIC NOTE")
    print("=" * 68)
    print("This is a real 10m Sentinel-2 input patch for deployment")
    print("inference. The eventual 2.5m output will be model-inferred,")
    print("not ground-truth 2.5m imagery.")
    print("=" * 68)


if __name__ == "__main__":
    extract_patch()
