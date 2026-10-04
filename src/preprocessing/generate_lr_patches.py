"""
Generate 2x Synthetic Low-Resolution (LR) Patches

Degradation Pipeline:
- Source: 10m Sentinel-2 HR patches (4, 256, 256), uint16
- Target: 20m-equivalent synthetic LR patches (4, 128, 128), uint16
- Method: 2x2 area averaging (box mean) calculated in float32, rounded to uint16
- Description: Simple low-pass / anti-aliasing approximation for 2x downsampling.

Inputs:
- HR Patches: data/processed/hr/{region}/*.npy
- HR Index: data/processed/hr_patches_index.csv
- Split Index: data/processed/hr_split_index.csv

Outputs:
- LR Patches: data/processed/lr/{region}/*.npy
- LR Index: data/processed/lr_patches_index.csv
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


def generate_lr_patches():
    # Define project paths
    project_root = Path(__file__).resolve().parent.parent.parent
    hr_dir = project_root / "data" / "processed" / "hr"
    lr_dir = project_root / "data" / "processed" / "lr"
    hr_index_path = project_root / "data" / "processed" / "hr_patches_index.csv"
    split_index_path = project_root / "data" / "processed" / "hr_split_index.csv"
    lr_index_path = project_root / "data" / "processed" / "lr_patches_index.csv"

    # Verify input index files exist
    if not hr_index_path.exists():
        raise FileNotFoundError(f"HR index missing: {hr_index_path}")
    if not split_index_path.exists():
        raise FileNotFoundError(f"Split index missing: {split_index_path}")

    # Read index files
    hr_df = pd.read_csv(hr_index_path)
    split_df = pd.read_csv(split_index_path)

    # Build verification set of split filenames
    split_filenames = set(split_df["filename"].tolist())

    print(f"Loaded HR index with {len(hr_df)} entries.")
    print(f"Loaded HR split index with {len(split_df)} entries.")

    # Create LR region subdirectories
    regions = ["forest", "urban", "agriculture", "water"]
    for region in regions:
        (lr_dir / region).mkdir(parents=True, exist_ok=True)

    lr_records = []
    shapes_set = set()
    dtypes_set = set()
    hr_region_counts = hr_df["region"].value_counts().to_dict()
    lr_region_counts = {r: 0 for r in regions}

    nan_count = 0
    inf_count = 0
    zero_patch_count = 0
    unmatched_split_count = 0

    print("\nGenerating 2x synthetic LR patches...")

    for idx, row in tqdm(hr_df.iterrows(), total=len(hr_df), desc="Generating LR Patches"):
        filename = row["filename"]
        region = row["region"]

        # Verify filename exists in split index
        if filename not in split_filenames:
            print(f"WARNING: Filename {filename} not present in authoritative split index!")
            unmatched_split_count += 1
            continue

        hr_patch_path = hr_dir / region / filename
        lr_patch_path = lr_dir / region / filename

        if not hr_patch_path.exists():
            raise FileNotFoundError(f"HR patch file not found: {hr_patch_path}")

        # Read HR patch (4, 256, 256) uint16
        hr_data = np.load(hr_patch_path)

        # 2x2 Spatial Area Averaging in float32
        # Reshape (4, 256, 256) -> (4, 128, 2, 128, 2) and mean over downsampling axes (2, 4)
        lr_float = hr_data.astype(np.float32).reshape(4, 128, 2, 128, 2).mean(axis=(2, 4))
        lr_uint16 = np.round(lr_float).astype(np.uint16)

        # Structural & Data QA Checks
        shapes_set.add(lr_uint16.shape)
        dtypes_set.add(str(lr_uint16.dtype))

        if lr_uint16.shape != (4, 128, 128):
            raise ValueError(f"Invalid LR shape {lr_uint16.shape} for patch {filename}")
        if lr_uint16.dtype != np.uint16:
            raise ValueError(f"Invalid LR dtype {lr_uint16.dtype} for patch {filename}")

        if np.isnan(lr_uint16).any():
            nan_count += 1
        if np.isinf(lr_uint16).any():
            inf_count += 1
        if np.all(lr_uint16 == 0):
            zero_patch_count += 1

        # Save LR patch
        np.save(lr_patch_path, lr_uint16)
        lr_region_counts[region] += 1

        # Construct LR index record
        record = {
            "filename": filename,
            "region": region,
            "safe_product": row["safe_product"],
            "crs": row["crs"],
            "row_off": row["row_off"],
            "col_off": row["col_off"],
            "patch_size_hr": row.get("patch_size", 256),
            "patch_size_lr": 128,
            "scale_factor": 2,
            "channel_order": row.get("channel_order", "B02,B03,B04,B08"),
            "hr_dtype": str(hr_data.dtype),
            "lr_dtype": str(lr_uint16.dtype),
            "hr_shape": str(hr_data.shape),
            "lr_shape": str(lr_uint16.shape),
            "degradation_method": "2x2_area_average",
        }
        lr_records.append(record)

    # Save LR Index CSV
    lr_df = pd.DataFrame(lr_records)
    lr_df.to_csv(lr_index_path, index=False)
    print(f"\nSaved LR index CSV to: {lr_index_path}")

    # Disk consistency checks
    missing_pairs = []
    extra_lr_files = []

    for region in regions:
        hr_files = set(os.listdir(hr_dir / region))
        lr_files = set(os.listdir(lr_dir / region))

        missing = hr_files - lr_files
        extra = lr_files - hr_files

        if missing:
            missing_pairs.extend([f"{region}/{f}" for f in missing])
        if extra:
            extra_lr_files.extend([f"{region}/{f}" for f in extra])

    # Detailed Summary Output
    print("\n" + "=" * 65)
    print("2x SYNTHETIC LR GENERATION & VALIDATION SUMMARY")
    print("=" * 65)
    print(f"Total HR Patches (from index):  {len(hr_df)}")
    print(f"Total LR Patches Generated:     {len(lr_df)}")
    print("-" * 65)
    print("Per-Region Patch Breakdown:")
    print(f"{'Region':<15} | {'HR Count':<10} | {'LR Count':<10} | {'Match':<5}")
    print("-" * 65)
    for region in regions:
        hr_c = hr_region_counts.get(region, 0)
        lr_c = lr_region_counts.get(region, 0)
        match_str = "YES" if hr_c == lr_c else "NO"
        print(f"{region:<15} | {hr_c:<10} | {lr_c:<10} | {match_str:<5}")
    print("-" * 65)
    print(f"Unique LR Shapes:               {list(shapes_set)}")
    print(f"Unique LR Dtypes:               {list(dtypes_set)}")
    print(f"NaN Patches Detected:           {nan_count}")
    print(f"Inf Patches Detected:           {inf_count}")
    print(f"All-Zero Patches:               {zero_patch_count}")
    print(f"Missing HR->LR Pairs:           {len(missing_pairs)}")
    print(f"Unexpected Extra LR Files:      {len(extra_lr_files)}")
    print(f"Unmatched Split Index Files:    {unmatched_split_count}")
    print("=" * 65)

    if (
        len(hr_df) == 6456
        and len(lr_df) == 6456
        and len(missing_pairs) == 0
        and len(extra_lr_files) == 0
        and nan_count == 0
        and inf_count == 0
        and zero_patch_count == 0
        and unmatched_split_count == 0
    ):
        print("SUCCESS: All 6456 LR patches generated cleanly and verified!")
    else:
        print("WARNING: Data validation issues detected! Check logs above.")


if __name__ == "__main__":
    generate_lr_patches()
