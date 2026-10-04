"""
Generate 4× Synthetic Low-Resolution (LR4x) Patches

Scientific Scope
────────────────
Training benchmark:  Synthetic 40m-equivalent LR → 10m HR reconstruction.
                     HR (4,256,256) at 10m is the training TARGET.
                     LR4x (4,64,64) is a SYNTHETIC 40m-equivalent degradation
                     produced by 4×4 spatial area averaging of the HR patches.

Later deployment:    The trained model will be applied to REAL 10m Sentinel-2
                     inputs to produce a 2.5m-scale INFERRED output.
                     This is an extrapolation beyond the training domain.

No claim is made:    No sub-10m ground-truth validation is performed.
                     Synthetic benchmark PSNR/SSIM does not constitute
                     validation of genuine 2.5m spatial content.

Degradation Formula
────────────────────
LR4x[c, i, j] = mean(HR[c, 4i:4i+4, 4j:4j+4])

Equivalently, via numpy reshape:
    lr_float = HR.astype(np.float32).reshape(4, 64, 4, 64, 4).mean(axis=(2, 4))
    lr_uint16 = np.round(lr_float).astype(np.uint16)

This is a deterministic 4×4 spatial area average computed in float32 per
spectral channel, then rounded to uint16. This serves as a simple low-pass /
anti-aliasing approximation for 4× downsampling. It is NOT a simulation of
the physical Sentinel-2 sensor response at any resolution.

Inputs
──────
  HR patches:  data/processed/hr/{region}/*.npy  (4,256,256) uint16
  HR index:    data/processed/hr_patches_index.csv
  Split index: data/processed/hr_split_index.csv  (used for verification only)

Outputs
───────
  LR4x patches:  data/processed/lr4x/{region}/*.npy  (4,64,64) uint16
  LR4x index:    data/processed/lr4x_patches_index.csv

Isolation Guarantee
────────────────────
This script reads HR patches and writes ONLY to data/processed/lr4x/.
No existing file is modified:
  - data/processed/hr/         → unchanged
  - data/processed/lr/         → unchanged (2× pipeline)
  - data/processed/hr_*        → unchanged (read-only)
  - data/processed/lr_patches_index.csv → unchanged
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_lr4x_patches():
    # ─── Project Paths ────────────────────────────────────────────────────────
    project_root   = Path(__file__).resolve().parent.parent.parent
    hr_dir         = project_root / "data" / "processed" / "hr"
    lr4x_dir       = project_root / "data" / "processed" / "lr4x"
    hr_index_path  = project_root / "data" / "processed" / "hr_patches_index.csv"
    split_index_path = project_root / "data" / "processed" / "hr_split_index.csv"
    lr4x_index_path = project_root / "data" / "processed" / "lr4x_patches_index.csv"

    # ─── Guard: do not touch existing lr/ directory ───────────────────────────
    lr2x_dir = project_root / "data" / "processed" / "lr"
    assert lr2x_dir.exists(), "Sanity check: 2× lr/ directory must exist."
    # (No writes to lr2x_dir below — this assertion just confirms isolation.)

    # ─── Verify Input Indices Exist ───────────────────────────────────────────
    for p, label in [(hr_index_path, "HR index"), (split_index_path, "Split index")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    hr_df    = pd.read_csv(hr_index_path)
    split_df = pd.read_csv(split_index_path)

    print("=" * 70)
    print("4× SYNTHETIC LR PATCH GENERATION")
    print("=" * 70)
    print(f"HR patches in index:    {len(hr_df)}")
    print(f"Split index entries:    {len(split_df)}")

    # Verify the two indices agree on filenames
    hr_filenames    = set(hr_df["filename"])
    split_filenames = set(split_df["filename"])
    unmatched = hr_filenames.symmetric_difference(split_filenames)
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} filenames differ between HR index and split index."
        )
    print(f"HR index ↔ split index: filenames match exactly")

    # ─── Create LR4x Output Directories ─────────────────────────────────────
    regions = ["forest", "urban", "agriculture", "water"]
    for region in regions:
        (lr4x_dir / region).mkdir(parents=True, exist_ok=True)
    print(f"\nCreated output directories under: {lr4x_dir}")

    # ─── Generation Loop ─────────────────────────────────────────────────────
    lr4x_records         = []
    shapes_seen          = set()
    dtypes_seen          = set()
    nan_count            = 0
    inf_count            = 0
    zero_patch_count     = 0
    unmatched_split      = 0
    region_hr_counts     = hr_df["region"].value_counts().to_dict()
    region_lr4x_counts   = {r: 0 for r in regions}

    print("\nGenerating 4× synthetic LR patches (4×4 area average)...")

    for _, row in tqdm(hr_df.iterrows(), total=len(hr_df), desc="LR4x Generation"):
        filename = row["filename"]
        region   = row["region"]

        # Verify this filename is in the authoritative split index
        if filename not in split_filenames:
            print(f"WARNING: {filename} not in split index — skipping.")
            unmatched_split += 1
            continue

        hr_patch_path   = hr_dir / region / filename
        lr4x_patch_path = lr4x_dir / region / filename

        if not hr_patch_path.exists():
            raise FileNotFoundError(f"HR patch missing: {hr_patch_path}")

        # Load HR patch
        hr_patch = np.load(hr_patch_path)   # (4, 256, 256) uint16

        if hr_patch.shape != (4, 256, 256):
            raise ValueError(
                f"Unexpected HR shape {hr_patch.shape} for {filename}"
            )

        # ── 4×4 Spatial Area Average (deterministic, per channel) ────────────
        # Reshape (4, 256, 256) → (4, 64, 4, 64, 4)
        # Mean over axes (2, 4) collapses each 4×4 spatial block into one value.
        lr4x_float = (
            hr_patch.astype(np.float32)
            .reshape(4, 64, 4, 64, 4)
            .mean(axis=(2, 4))
        )                                   # (4, 64, 64) float32
        lr4x_uint16 = np.round(lr4x_float).astype(np.uint16)

        # ── Per-patch QA ─────────────────────────────────────────────────────
        if lr4x_uint16.shape != (4, 64, 64):
            raise ValueError(
                f"LR4x shape error {lr4x_uint16.shape} for {filename}"
            )
        shapes_seen.add(lr4x_uint16.shape)
        dtypes_seen.add(str(lr4x_uint16.dtype))

        if np.isnan(lr4x_uint16).any():
            nan_count += 1
        if np.isinf(lr4x_uint16).any():
            inf_count += 1
        if np.all(lr4x_uint16 == 0):
            zero_patch_count += 1

        # ── Save ─────────────────────────────────────────────────────────────
        np.save(lr4x_patch_path, lr4x_uint16)
        region_lr4x_counts[region] += 1

        # ── Build index record ───────────────────────────────────────────────
        lr4x_records.append({
            "filename":          filename,
            "region":            region,
            "safe_product":      row.get("safe_product", ""),
            "crs":               row.get("crs", ""),
            "row_off":           row.get("row_off", ""),
            "col_off":           row.get("col_off", ""),
            "patch_size_hr":     256,
            "patch_size_lr4x":   64,
            "scale_factor":      4,
            "channel_order":     row.get("channel_order", "B02,B03,B04,B08"),
            "hr_dtype":          str(hr_patch.dtype),
            "lr4x_dtype":        str(lr4x_uint16.dtype),
            "hr_shape":          str(hr_patch.shape),
            "lr4x_shape":        str(lr4x_uint16.shape),
            "degradation_method": "4x4_area_average",
        })

    # ─── Save LR4x Index ─────────────────────────────────────────────────────
    lr4x_df = pd.DataFrame(lr4x_records)
    lr4x_df.to_csv(lr4x_index_path, index=False)
    print(f"\nSaved LR4x index: {lr4x_index_path}")

    # ─── Disk Consistency Check ───────────────────────────────────────────────
    missing_pairs = []
    extra_lr4x    = []
    for region in regions:
        hr_files   = set(os.listdir(hr_dir / region))
        lr4x_files = set(os.listdir(lr4x_dir / region))
        missing_pairs.extend(
            [f"{region}/{f}" for f in (hr_files - lr4x_files)]
        )
        extra_lr4x.extend(
            [f"{region}/{f}" for f in (lr4x_files - hr_files)]
        )

    # ─── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("4× LR GENERATION SUMMARY")
    print("=" * 70)
    print(f"HR patches (from index):       {len(hr_df)}")
    print(f"LR4x patches generated:        {len(lr4x_df)}")
    print("-" * 70)
    print(f"{'Region':<15} | {'HR Count':>10} | {'LR4x Count':>12} | {'Match':>6}")
    print("-" * 70)
    for r in regions:
        hr_c   = region_hr_counts.get(r, 0)
        lr4x_c = region_lr4x_counts.get(r, 0)
        match  = "YES" if hr_c == lr4x_c else "NO ← MISMATCH"
        print(f"{r:<15} | {hr_c:>10} | {lr4x_c:>12} | {match:>6}")
    print("-" * 70)
    print(f"Unique LR4x shapes:            {list(shapes_seen)}")
    print(f"Unique LR4x dtypes:            {list(dtypes_seen)}")
    print(f"NaN patches:                   {nan_count}")
    print(f"Inf patches:                   {inf_count}")
    print(f"All-zero patches:              {zero_patch_count}")
    print(f"Missing HR→LR4x pairs:         {len(missing_pairs)}")
    print(f"Unexpected extra LR4x files:   {len(extra_lr4x)}")
    print(f"Unmatched split-index files:   {unmatched_split}")
    print("=" * 70)

    # ─── Final Pass/Fail ──────────────────────────────────────────────────────
    all_ok = (
        len(hr_df) == 6456
        and len(lr4x_df) == 6456
        and len(missing_pairs) == 0
        and len(extra_lr4x) == 0
        and nan_count == 0
        and inf_count == 0
        and zero_patch_count == 0
        and unmatched_split == 0
        and list(shapes_seen) == [(4, 64, 64)]
        and list(dtypes_seen) == ["uint16"]
    )

    if all_ok:
        print("RESULT: SUCCESS — all 6456 LR4x patches generated and verified.")
    else:
        print("RESULT: WARNING — one or more checks failed. Review the log above.")

    print("=" * 70)
    print("\nScientific scope reminder:")
    print("  Training benchmark = synthetic 40m-equivalent → 10m reconstruction.")
    print("  Deployment target  = 10m Sentinel-2 → 2.5m-scale inferred output.")
    print("  No sub-10m ground-truth validation is claimed or implied.")
    print("=" * 70)


if __name__ == "__main__":
    generate_lr4x_patches()
