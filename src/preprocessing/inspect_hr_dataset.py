"""
src/preprocessing/inspect_hr_dataset.py

HR Dataset Quality Control & Structural Verification Script
------------------------------------------------------------
Inspects extracted HR .npy patches in data/processed/hr/ to verify:
  1. Directory structure and file counts
  2. Index CSV integrity (data/processed/hr_patches_index.csv)
  3. Patch array shape, dtype, NaNs, Infs, and zero-pixel percentages
  4. Per-channel statistics (min, max, mean, std) across classes for:
     Ch 0: B02 (Blue, 10m)
     Ch 1: B03 (Green, 10m)
     Ch 2: B04 (Red, 10m)
     Ch 3: B08 (NIR, 10m)

Usage:
    python src/preprocessing/inspect_hr_dataset.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def inspect_directory_structure(root_dir: Path):
    print("=" * 70)
    print("1. DIRECTORY & FILE STRUCTURE VERIFICATION")
    print("=" * 70)

    # Required scripts check
    src_dir = root_dir / "src" / "preprocessing"
    expected_scripts = [
        "inspect_safe.py",
        "inspect_pixels.py",
        "visualize_sample.py",
        "patch_scanner.py",
        "analyze_patch_metadata.py",
        "validate_patch_selection.py",
        "extract_hr_patches.py",
    ]

    print("\n[+] Checking existing preprocessing scripts in src/preprocessing/:")
    for script in expected_scripts:
        script_path = src_dir / script
        status = "FOUND" if script_path.exists() else "MISSING"
        print(f"  - {script:30s} : [{status}]")

    # HR Patches directory check
    hr_dir = root_dir / "data" / "processed" / "hr"
    regions = ["forest", "urban", "agriculture", "water"]

    print("\n[+] Checking HR patch directories & counts under data/processed/hr/:")
    total_found_patches = 0
    class_counts = {}

    if not hr_dir.exists():
        print(f"  [ERROR] Directory not found: {hr_dir}")
        return False, class_counts

    for region in regions:
        region_path = hr_dir / region
        if region_path.exists():
            files = list(region_path.glob("*.npy"))
            count = len(files)
            class_counts[region] = count
            total_found_patches += count
            print(f"  - {region:15s} : {count} .npy patches")
        else:
            class_counts[region] = 0
            print(f"  - {region:15s} : [DIRECTORY MISSING]")

    print(f"\n  Total HR Patches Found: {total_found_patches}")

    # Index CSV check
    index_csv = root_dir / "data" / "processed" / "hr_patches_index.csv"
    print("\n[+] Checking hr_patches_index.csv:")
    if index_csv.exists():
        df_index = pd.read_csv(index_csv)
        print(f"  - Index CSV found at : {index_csv}")
        print(f"  - Index CSV rows     : {len(df_index)}")
        print(f"  - Columns            : {list(df_index.columns)}")
    else:
        print(f"  [WARNING] Index CSV not found at: {index_csv}")

    # Metadata CSV check
    meta_csv = root_dir / "outputs" / "patches_metadata.csv"
    print("\n[+] Checking outputs/patches_metadata.csv:")
    if meta_csv.exists():
        df_meta = pd.read_csv(meta_csv)
        print(f"  - Metadata CSV found at: {meta_csv}")
        print(f"  - Metadata CSV rows    : {len(df_meta)}")
    else:
        print(f"  - Metadata CSV status  : Not found (Optional)")

    return True, class_counts


def inspect_hr_patches(root_dir: Path, num_samples_per_class: int = 20):
    print("\n" + "=" * 70)
    print("2. HR PATCH STRUCTURAL & QUANTITATIVE INSPECTION")
    print("=" * 70)

    hr_dir = root_dir / "data" / "processed" / "hr"
    regions = ["forest", "urban", "agriculture", "water"]
    channel_names = [
        "B02 (Blue)",
        "B03 (Green)",
        "B04 (Red)",
        "B08 (NIR)",
    ]

    all_shapes = set()
    all_dtypes = set()
    nan_count = 0
    inf_count = 0
    zero_patch_count = 0

    class_stats = {}

    for region in regions:
        region_path = hr_dir / region
        if not region_path.exists():
            continue

        files = sorted(list(region_path.glob("*.npy")))
        if not files:
            continue

        # Select samples evenly across the list
        sample_indices = np.linspace(
            0, len(files) - 1, min(num_samples_per_class, len(files)), dtype=int
        )
        sampled_files = [files[i] for i in sample_indices]

        ch_mins = [[] for _ in range(4)]
        ch_maxs = [[] for _ in range(4)]
        ch_means = [[] for _ in range(4)]
        ch_stds = [[] for _ in range(4)]
        ch_zero_pcts = [[] for _ in range(4)]

        for fpath in sampled_files:
            arr = np.load(fpath)

            all_shapes.add(arr.shape)
            all_dtypes.add(arr.dtype)

            # Check NaNs / Infs
            if np.isnan(arr).any():
                nan_count += 1
            if np.isinf(arr).any():
                inf_count += 1

            # Check all zero patch
            if np.all(arr == 0):
                zero_patch_count += 1

            # Calculate per channel stats
            for c in range(4):
                ch_data = arr[c, :, :]
                ch_mins[c].append(np.min(ch_data))
                ch_maxs[c].append(np.max(ch_data))
                ch_means[c].append(np.mean(ch_data))
                ch_stds[c].append(np.std(ch_data))
                ch_zero_pcts[c].append(np.mean(ch_data == 0) * 100.0)

        # Aggregate stats for region
        class_stats[region] = {
            "min": [float(np.min(ch_mins[c])) for c in range(4)],
            "max": [float(np.max(ch_maxs[c])) for c in range(4)],
            "mean": [float(np.mean(ch_means[c])) for c in range(4)],
            "std": [float(np.mean(ch_stds[c])) for c in range(4)],
            "zero_pct": [float(np.mean(ch_zero_pcts[c])) for c in range(4)],
        }

    print(
        f"\n[+] Unique Array Shapes Found : {list(all_shapes)} (Expected: [(4, 256, 256)])"
    )
    print(
        f"[+] Unique Array Dtypes Found : {list(all_dtypes)} (Expected: [dtype('uint16')])"
    )
    print(f"[+] Total Patches with NaN    : {nan_count}")
    print(f"[+] Total Patches with Inf    : {inf_count}")
    print(f"[+] Total All-Zero Patches    : {zero_patch_count}")

    print("\n[+] Per-Class Spectral Statistics (Sampled Patches):")
    print("-" * 70)

    for region, stats in class_stats.items():
        print(f"\n--- REGION CLASS: {region.upper()} ---")
        for c in range(4):
            print(
                f"  Channel {c} ({channel_names[c]:12s}): "
                f"Min={stats['min'][c]:5.0f} | "
                f"Max={stats['max'][c]:5.0f} | "
                f"Mean={stats['mean'][c]:7.1f} | "
                f"Std={stats['std'][c]:7.1f} | "
                f"Zero={stats['zero_pct'][c]:5.2f}%"
            )

    print("\n" + "=" * 70)
    print("3. QA SUMMARY REPORT")
    print("=" * 70)

    valid_shape = all_shapes == {(4, 256, 256)}
    valid_dtype = all_dtypes == {np.dtype("uint16")}
    no_nan_inf = (nan_count == 0) and (inf_count == 0)

    if valid_shape and valid_dtype and no_nan_inf:
        print("[SUCCESS] HR Patches are STRUCTURALLY VALID and Quantitative!")
    else:
        print(
            "[WARNING] Issues detected! Check shape, dtype, or NaN/Inf counts above."
        )


if __name__ == "__main__":
    # Resolve relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    print(f"Project Root: {project_root}\n")

    inspect_directory_structure(project_root)
    inspect_hr_patches(project_root)
