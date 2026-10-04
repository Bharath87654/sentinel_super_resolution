"""
GeoTIFF Exporter for 4× Deployment Inference

Scientific Scope
────────────────
This script converts the 4× model-inferred product from a NumPy array
into a georeferenced GeoTIFF. It scales the 10m input transform to a
2.5m output transform while preserving the original top-left coordinates,
spatial bounds, and CRS.

IMPORTANT: The output GeoTIFF is a model-inferred 2.5m-scale product,
not true ground-truth imagery. This is explicitly embedded into the
GeoTIFF metadata tags to ensure downstream scientific transparency.

Isolation Guarantee
────────────────────
Does not modify any existing Python files, datasets, or the raw
Sentinel-2 .SAFE product.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

EXPECTED_SHAPE = (4, 256, 256)
EXPECTED_BANDS = ["B02", "B03", "B04", "B08"]
SCALE_FACTOR = 4
INPUT_GSD_M = 10.0
OUTPUT_GSD_M = 2.5


def export_geotiff():
    parser = argparse.ArgumentParser(
        description="Convert 4× SR float32 .npy output to georeferenced GeoTIFF."
    )
    parser.add_argument(
        "--reflectance", type=str, required=True,
        help="Path to the float32 SR reflectance .npy",
    )
    parser.add_argument(
        "--metadata", type=str, required=True,
        help="Path to the metadata .json produced by extract_real_patch_4x.py",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to the output .tif file",
    )
    args = parser.parse_args()

    ref_path = Path(args.reflectance)
    meta_path = Path(args.metadata)
    out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ─── Load and Validate Metadata ───────────────────────────────────────────
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file missing: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Extract required fields from JSON
    patch_transform_list = meta["patch_transform"]
    crs_str = meta["crs"]
    source_safe = meta.get("source_safe", "UNKNOWN")
    source_row = meta.get("row_off", -1)
    source_col = meta.get("col_off", -1)

    # rasterio.Affine expects (a, b, c, d, e, f)
    # The JSON stores the 9-element tuple from rasterio or the 6-element tuple.
    # Take the first 6 elements.
    t_10m = Affine(*patch_transform_list[:6])

    # ─── Derive 2.5m Transform ────────────────────────────────────────────────
    # A 4x resolution enhancement means pixels are 4x smaller.
    # Top-left coordinate (c, f) stays exactly the same.
    # Pixel scale components (a, e) are divided by 4.

    # We can mathematically derive this by scaling the affine matrix.
    t_2_5m = t_10m * Affine.scale(1.0 / SCALE_FACTOR, 1.0 / SCALE_FACTOR)

    assert math.isclose(abs(t_2_5m.a), OUTPUT_GSD_M, rel_tol=1e-3), (
        f"Transform X pixel size is {abs(t_2_5m.a)}, expected ~{OUTPUT_GSD_M}"
    )
    assert math.isclose(abs(t_2_5m.e), OUTPUT_GSD_M, rel_tol=1e-3), (
        f"Transform Y pixel size is {abs(t_2_5m.e)}, expected ~{OUTPUT_GSD_M}"
    )

    # ─── Load and Validate Reflectance .npy ───────────────────────────────────
    if not ref_path.exists():
        raise FileNotFoundError(f"Reflectance file missing: {ref_path}")

    sr_ref = np.load(ref_path)

    if sr_ref.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Invalid shape. Expected {EXPECTED_SHAPE}, got {sr_ref.shape}"
        )
    if sr_ref.dtype != np.float32:
        raise ValueError(
            f"Invalid dtype. Expected float32, got {sr_ref.dtype}"
        )
    if np.isnan(sr_ref).any() or np.isinf(sr_ref).any():
        raise ValueError("Reflectance array contains NaN or Inf values.")

    # ─── Write GeoTIFF ────────────────────────────────────────────────────────
    profile = {
        "driver": "GTiff",
        "height": EXPECTED_SHAPE[1],
        "width": EXPECTED_SHAPE[2],
        "count": EXPECTED_SHAPE[0],
        "dtype": "float32",
        "crs": crs_str,
        "transform": t_2_5m,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
    }

    tags = {
        "MODEL": "MSRResNet4x",
        "SCALE_FACTOR": str(SCALE_FACTOR),
        "INPUT_GSD_M": str(INPUT_GSD_M),
        "OUTPUT_GSD_M": str(OUTPUT_GSD_M),
        "PRODUCT_TYPE": "2.5m-scale inferred",
        "SCIENTIFIC_NOTE": (
            "Model-inferred product; not validated against true 2.5m "
            "ground-truth imagery."
        ),
        "SOURCE_SAFE": source_safe,
        "SOURCE_ROW": str(source_row),
        "SOURCE_COL": str(source_col),
    }

    print("=" * 68)
    print("4× GEOTIFF EXPORT (DEPLOYMENT INFERENCE)")
    print("=" * 68)
    print(f"Input NPY   : {ref_path.name}")
    print(f"Input JSON  : {meta_path.name}")
    print(f"Output TIF  : {out_path.name}")

    with rasterio.open(out_path, "w", **profile) as dst:
        # Write array
        dst.write(sr_ref)

        # Set Band descriptions
        for i, b_name in enumerate(EXPECTED_BANDS, start=1):
            dst.set_band_description(i, b_name)

        # Set Dataset tags
        dst.update_tags(**tags)

    print("\nFile written successfully. Starting quality check...")

    # ─── Quality Check via Re-Reading ─────────────────────────────────────────
    with rasterio.open(out_path, "r") as src:
        chk_crs = str(src.crs)
        chk_transform = src.transform
        chk_width = src.width
        chk_height = src.height
        chk_count = src.count
        chk_dtype = src.dtypes[0]
        chk_bounds = src.bounds
        chk_descs = src.descriptions

        chk_arr = src.read()

        print("\nGeoTIFF Metadata Verification:")
        print(f"  CRS          : {chk_crs}")
        print(f"  Transform    : {chk_transform}")
        print(f"  Bounds       : {chk_bounds}")
        print(f"  Pixel Size   : X={abs(chk_transform.a):.4f}m, Y={abs(chk_transform.e):.4f}m")
        print(f"  Dimensions   : {chk_width} × {chk_height} pixels")
        print(f"  Band Count   : {chk_count}")
        print(f"  Dtype        : {chk_dtype}")
        print(f"  Band Labels  : {chk_descs}")

        # Assertions
        assert chk_crs == crs_str, "CRS verification failed."
        assert chk_width == EXPECTED_SHAPE[2], "Width verification failed."
        assert chk_height == EXPECTED_SHAPE[1], "Height verification failed."
        assert chk_count == EXPECTED_SHAPE[0], "Band count verification failed."
        assert chk_dtype == "float32", "Dtype verification failed."
        assert math.isclose(abs(chk_transform.a), OUTPUT_GSD_M, rel_tol=1e-3)
        assert math.isclose(abs(chk_transform.e), OUTPUT_GSD_M, rel_tol=1e-3)
        assert list(chk_descs) == EXPECTED_BANDS, "Band descriptions failed."
        assert not np.isnan(chk_arr).any() and not np.isinf(chk_arr).any(), "NaN/Inf detected on reload."

        # Float tolerance check
        max_diff = np.abs(chk_arr - sr_ref).max()
        assert max_diff < 1e-6, f"Data corrupted during serialization. Max diff: {max_diff}"

        # Stats
        chk_min = float(chk_arr.min())
        chk_max = float(chk_arr.max())
        chk_mean = float(chk_arr.mean())

        print("\nData Value Verification (Float32 Reflectance):")
        print(f"  Min Value    : {chk_min:.6f}")
        print(f"  Max Value    : {chk_max:.6f}")
        print(f"  Mean Value   : {chk_mean:.6f}")

    print("\n" + "=" * 68)
    print("GEOTIFF EXPORT COMPLETE AND VERIFIED")
    print("=" * 68)


if __name__ == "__main__":
    export_geotiff()
