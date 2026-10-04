"""
LR Dataset Inspection and Pairing Quality Assurance

Read-only diagnostic script to verify:
1. HR/LR 1-to-1 patch correspondence and region directory alignment.
2. Membership of every LR file in data/processed/hr_split_index.csv.
3. Numerical precision of degradation by independently recomputing LR from HR on random samples.
4. Structural integrity across all 6,456 LR patches (shape, dtype, NaN, Inf, zero patches).
5. Global & per-region channel statistics (min, max, mean, std, zero pixel percentage) for HR and LR.
6. Split inheritance and distribution across Train/Val/Test sets.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


def inspect_lr_dataset():
    # Setup project paths
    project_root = Path(__file__).resolve().parent.parent.parent
    hr_dir = project_root / "data" / "processed" / "hr"
    lr_dir = project_root / "data" / "processed" / "lr"
    hr_index_path = project_root / "data" / "processed" / "hr_patches_index.csv"
    lr_index_path = project_root / "data" / "processed" / "lr_patches_index.csv"
    split_index_path = project_root / "data" / "processed" / "hr_split_index.csv"

    print("=" * 75)
    print("STARTING LR NUMERICAL & PAIRING QA INSPECTION")
    print("=" * 75)

    qa_results = {}

    # Read index files
    hr_df = pd.read_csv(hr_index_path)
    lr_df = pd.read_csv(lr_index_path)
    split_df = pd.read_csv(split_index_path)

    regions = ["forest", "urban", "agriculture", "water"]
    channel_names = ["B02", "B03", "B04", "B08"]

    # -------------------------------------------------------------------------
    # CHECK 1 & 3: File Existence & 1-to-1 Region Correspondence
    # -------------------------------------------------------------------------
    print("\n[Check 1 & 3] Verifying HR/LR Patch Existence & Region Correspondence...")
    hr_files_by_region = {r: set(os.listdir(hr_dir / r)) for r in regions}
    lr_files_by_region = {r: set(os.listdir(lr_dir / r)) for r in regions}

    correspondence_pass = True
    for r in regions:
        hr_set = hr_files_by_region[r]
        lr_set = lr_files_by_region[r]
        if hr_set != lr_set:
            correspondence_pass = False
            print(f"  FAIL: Mismatch in region '{r}': {len(hr_set)} HR files vs {len(lr_set)} LR files.")
        else:
            print(f"  PASS: Region '{r:<11}' - exact 1-to-1 match ({len(lr_set)} pairs).")

    qa_results["Check 1 & 3: HR/LR 1-to-1 Region Correspondence"] = "PASS" if correspondence_pass else "FAIL"

    # -------------------------------------------------------------------------
    # CHECK 2: Membership in Split Index
    # -------------------------------------------------------------------------
    print("\n[Check 2] Verifying LR Filename Membership in Split Index...")
    split_filenames = set(split_df["filename"])
    lr_filenames = set(lr_df["filename"])

    missing_in_split = lr_filenames - split_filenames
    if len(missing_in_split) == 0 and len(lr_filenames) == len(split_filenames) == 6456:
        print(f"  PASS: All 6,456 LR filenames exist in hr_split_index.csv.")
        qa_results["Check 2: LR Membership in Split Index"] = "PASS"
    else:
        print(f"  FAIL: {len(missing_in_split)} LR filenames missing from split index!")
        qa_results["Check 2: LR Membership in Split Index"] = "FAIL"

    # -------------------------------------------------------------------------
    # CHECK 4: Random Sample Numerical Exactness Verification
    # -------------------------------------------------------------------------
    print("\n[Check 4] Sampling HR/LR Pairs & Recomputing Degradation...")
    np.random.seed(42)
    sample_size_per_region = 50
    total_sampled = 0
    max_abs_diff_global = 0
    total_mismatched_pixels = 0
    mean_abs_diff_list = []

    for r in regions:
        region_hr_files = sorted(list(hr_files_by_region[r]))
        sampled_files = np.random.choice(region_hr_files, size=sample_size_per_region, replace=False)

        for fname in sampled_files:
            hr_path = hr_dir / r / fname
            lr_path = lr_dir / r / fname

            hr_patch = np.load(hr_path)
            lr_saved = np.load(lr_path)

            # Recompute LR using exact specified degradation formula
            recomputed = (
                hr_patch.astype(np.float32)
                .reshape(4, 128, 2, 128, 2)
                .mean(axis=(2, 4))
            )
            recomputed = np.round(recomputed).astype(np.uint16)

            diff = np.abs(lr_saved.astype(np.int32) - recomputed.astype(np.int32))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            mismatches = np.count_nonzero(diff)

            max_abs_diff_global = max(max_abs_diff_global, max_diff)
            mean_abs_diff_list.append(mean_diff)
            total_mismatched_pixels += mismatches
            total_sampled += 1

    overall_mean_abs_diff = np.mean(mean_abs_diff_list)
    print(f"  Sampled pairs tested:       {total_sampled} ({sample_size_per_region} per region)")
    print(f"  Max Absolute Difference:    {max_abs_diff_global}")
    print(f"  Mean Absolute Difference:   {overall_mean_abs_diff:.6f}")
    print(f"  Total Mismatched Pixels:    {total_mismatched_pixels}")

    numerical_pass = (max_abs_diff_global == 0) and (overall_mean_abs_diff == 0.0) and (total_mismatched_pixels == 0)
    qa_results["Check 4: Sample Degradation Numerical Exactness"] = "PASS" if numerical_pass else "FAIL"

    # -------------------------------------------------------------------------
    # CHECK 5: Full LR Dataset Structural Scan & Accumulator Setup
    # -------------------------------------------------------------------------
    print("\n[Check 5] Full LR Dataset Structural Scan (6,456 Patches)...")
    invalid_shape_count = 0
    invalid_dtype_count = 0
    nan_count = 0
    inf_count = 0
    zero_patch_count = 0

    # Channel-wise accumulators for global HR and LR stats
    hr_channel_stats = {c: {"sum": 0.0, "sq_sum": 0.0, "min": float("inf"), "max": 0, "zeros": 0, "count": 0} for c in range(4)}
    lr_channel_stats = {c: {"sum": 0.0, "sq_sum": 0.0, "min": float("inf"), "max": 0, "zeros": 0, "count": 0} for c in range(4)}

    lr_region_stats = {
        r: {c: {"sum": 0.0, "sq_sum": 0.0, "min": float("inf"), "max": 0, "zeros": 0, "count": 0} for c in range(4)}
        for r in regions
    }

    for idx, row in tqdm(lr_df.iterrows(), total=len(lr_df), desc="Scanning Dataset & Computing Stats"):
        r = row["region"]
        fname = row["filename"]

        lr_path = lr_dir / r / fname
        hr_path = hr_dir / r / fname

        lr_patch = np.load(lr_path)
        hr_patch = np.load(hr_path)

        # Integrity checks
        if lr_patch.shape != (4, 128, 128):
            invalid_shape_count += 1
        if lr_patch.dtype != np.uint16:
            invalid_dtype_count += 1
        if np.isnan(lr_patch).any():
            nan_count += 1
        if np.isinf(lr_patch).any():
            inf_count += 1
        if np.all(lr_patch == 0):
            zero_patch_count += 1

        # Channel statistics accumulation
        for c in range(4):
            hr_c = hr_patch[c].astype(np.float64)
            lr_c = lr_patch[c].astype(np.float64)

            # Accumulate HR
            hr_channel_stats[c]["sum"] += np.sum(hr_c)
            hr_channel_stats[c]["sq_sum"] += np.sum(hr_c ** 2)
            hr_channel_stats[c]["min"] = min(hr_channel_stats[c]["min"], np.min(hr_c))
            hr_channel_stats[c]["max"] = max(hr_channel_stats[c]["max"], np.max(hr_c))
            hr_channel_stats[c]["zeros"] += np.count_nonzero(hr_c == 0)
            hr_channel_stats[c]["count"] += hr_c.size

            # Accumulate LR
            lr_channel_stats[c]["sum"] += np.sum(lr_c)
            lr_channel_stats[c]["sq_sum"] += np.sum(lr_c ** 2)
            lr_channel_stats[c]["min"] = min(lr_channel_stats[c]["min"], np.min(lr_c))
            lr_channel_stats[c]["max"] = max(lr_channel_stats[c]["max"], np.max(lr_c))
            lr_channel_stats[c]["zeros"] += np.count_nonzero(lr_c == 0)
            lr_channel_stats[c]["count"] += lr_c.size

            # Accumulate LR Region
            lr_region_stats[r][c]["sum"] += np.sum(lr_c)
            lr_region_stats[r][c]["sq_sum"] += np.sum(lr_c ** 2)
            lr_region_stats[r][c]["min"] = min(lr_region_stats[r][c]["min"], np.min(lr_c))
            lr_region_stats[r][c]["max"] = max(lr_region_stats[r][c]["max"], np.max(lr_c))
            lr_region_stats[r][c]["zeros"] += np.count_nonzero(lr_c == 0)
            lr_region_stats[r][c]["count"] += lr_c.size

    full_integrity_pass = (
        (invalid_shape_count == 0)
        and (invalid_dtype_count == 0)
        and (nan_count == 0)
        and (inf_count == 0)
        and (zero_patch_count == 0)
    )
    print(f"  Invalid Shapes:             {invalid_shape_count}")
    print(f"  Invalid Dtypes:             {invalid_dtype_count}")
    print(f"  NaN Patches:                {nan_count}")
    print(f"  Inf Patches:                {inf_count}")
    print(f"  All-Zero Patches:           {zero_patch_count}")

    qa_results["Check 5: Full Dataset Integrity Scan"] = "PASS" if full_integrity_pass else "FAIL"

    # -------------------------------------------------------------------------
    # CHECK 6: Basic Statistics & HR vs LR Comparison
    # -------------------------------------------------------------------------
    print("\n[Check 6] Overall & Per-Region Statistics Summary...")

    print("\n  A. Global HR vs LR Channel Statistics:")
    print("-" * 85)
    print(f"{'Channel':<8} | {'Dataset':<8} | {'Min':<6} | {'Max':<6} | {'Mean':<10} | {'Std':<10} | {'Zero Pixel %':<12}")
    print("-" * 85)

    for c, ch_name in enumerate(channel_names):
        # HR
        hr_n = hr_channel_stats[c]["count"]
        hr_mean = hr_channel_stats[c]["sum"] / hr_n
        hr_var = (hr_channel_stats[c]["sq_sum"] / hr_n) - (hr_mean ** 2)
        hr_std = np.sqrt(max(0.0, hr_var))
        hr_zero_pct = (hr_channel_stats[c]["zeros"] / hr_n) * 100.0

        # LR
        lr_n = lr_channel_stats[c]["count"]
        lr_mean = lr_channel_stats[c]["sum"] / lr_n
        lr_var = (lr_channel_stats[c]["sq_sum"] / lr_n) - (lr_mean ** 2)
        lr_std = np.sqrt(max(0.0, lr_var))
        lr_zero_pct = (lr_channel_stats[c]["zeros"] / lr_n) * 100.0

        print(f"{ch_name:<8} | {'HR':<8} | {int(hr_channel_stats[c]['min']):<6} | {int(hr_channel_stats[c]['max']):<6} | {hr_mean:<10.2f} | {hr_std:<10.2f} | {hr_zero_pct:<12.4f}%")
        print(f"{ch_name:<8} | {'LR':<8} | {int(lr_channel_stats[c]['min']):<6} | {int(lr_channel_stats[c]['max']):<6} | {lr_mean:<10.2f} | {lr_std:<10.2f} | {lr_zero_pct:<12.4f}%")
        print("-" * 85)

    print("\n  B. LR Per-Region Channel Statistics:")
    print("-" * 85)
    print(f"{'Region':<12} | {'Channel':<8} | {'Min':<6} | {'Max':<6} | {'Mean':<10} | {'Std':<10} | {'Zero Pixel %':<12}")
    print("-" * 85)

    for r in regions:
        for c, ch_name in enumerate(channel_names):
            n = lr_region_stats[r][c]["count"]
            mean = lr_region_stats[r][c]["sum"] / n
            var = (lr_region_stats[r][c]["sq_sum"] / n) - (mean ** 2)
            std = np.sqrt(max(0.0, var))
            zero_pct = (lr_region_stats[r][c]["zeros"] / n) * 100.0
            print(f"{r:<12} | {ch_name:<8} | {int(lr_region_stats[r][c]['min']):<6} | {int(lr_region_stats[r][c]['max']):<6} | {mean:<10.2f} | {std:<10.2f} | {zero_pct:<12.4f}%")
        print("-" * 85)

    qa_results["Check 6: Dataset Statistics Computation"] = "PASS"

    # -------------------------------------------------------------------------
    # CHECK 7: Split Relationship & Distribution Verification
    # -------------------------------------------------------------------------
    print("\n[Check 7] Verifying Split Relationship & Inheritance...")

    merged_split = pd.merge(lr_df, split_df[["filename", "spatial_block_id", "split"]], on="filename", how="inner")

    if len(merged_split) != 6456:
        print(f"  FAIL: Merged split row count is {len(merged_split)}, expected 6456.")
        split_pass = False
    else:
        filename_split_counts = merged_split.groupby("filename")["split"].nunique()
        dup_splits = (filename_split_counts > 1).sum()

        if dup_splits == 0:
            split_pass = True
            print("  PASS: Every LR patch inherits exactly one split. No filename spans multiple splits.")
        else:
            split_pass = False
            print(f"  FAIL: {dup_splits} filenames appear in multiple splits!")

    split_counts_table = pd.crosstab(merged_split["region"], merged_split["split"], margins=True)
    print("\n  LR Patch Split Counts by Region:")
    print("-" * 55)
    print(split_counts_table.to_string())
    print("-" * 55)

    qa_results["Check 7: Split Inheritance & Uniqueness"] = "PASS" if split_pass else "FAIL"

    # -------------------------------------------------------------------------
    # FINAL QA SUMMARY REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("FINAL QA SUMMARY REPORT")
    print("=" * 75)
    all_passed = True
    for check_name, status in qa_results.items():
        print(f"  [{status:<4}] {check_name}")
        if status != "PASS":
            all_passed = False
    print("=" * 75)

    if all_passed:
        print("OVERALL RESULT: ALL QA CHECKS PASSED SUCCESSFULLY!")
    else:
        print("OVERALL RESULT: ONE OR MORE QA CHECKS FAILED.")
    print("=" * 75)


if __name__ == "__main__":
    inspect_lr_dataset()
