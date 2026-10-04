"""
Calculate Training-Only Normalization Statistics

Computes per-channel normalization statistics (mean, std, min, max, percentiles)
STRICTLY from the training set patches (split == "train") in data/processed/hr_split_index.csv.

Applies scale conversion: float32_value = uint16_value / 10000.0 (no clipping).

Outputs:
- data/processed/normalization_stats.json
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


def calc_stats_from_hist(hist, scale=10000.0):
    """Computes exact statistics from uint16 pixel value frequency histogram."""
    total_pixels = hist.sum()
    if total_pixels == 0:
        return {}

    values = np.arange(len(hist), dtype=np.float64) / scale
    nonzero_indices = np.where(hist > 0)[0]
    min_val = float(nonzero_indices[0] / scale)
    max_val = float(nonzero_indices[-1] / scale)

    counts = hist.astype(np.float64)
    sum_val = np.sum(counts * values)
    mean_val = float(sum_val / total_pixels)

    sum_sq = np.sum(counts * (values**2))
    var_val = max(0.0, (sum_sq / total_pixels) - (mean_val**2))
    std_val = float(np.sqrt(var_val))

    cum_counts = np.cumsum(counts)

    def get_percentile(p):
        target = (p / 100.0) * total_pixels
        idx = np.searchsorted(cum_counts, target)
        idx = min(idx, len(hist) - 1)
        return float(idx / scale)

    return {
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
        "min": round(min_val, 6),
        "max": round(max_val, 6),
        "p1": round(get_percentile(1.0), 6),
        "p50_median": round(get_percentile(50.0), 6),
        "p99": round(get_percentile(99.0), 6),
        "p99_9": round(get_percentile(99.9), 6),
        "total_pixels": int(total_pixels),
    }


def calculate_normalization_stats():
    project_root = Path(__file__).resolve().parent.parent.parent
    split_index_path = project_root / "data" / "processed" / "hr_split_index.csv"
    hr_dir = project_root / "data" / "processed" / "hr"
    lr_dir = project_root / "data" / "processed" / "lr"
    output_json_path = project_root / "data" / "processed" / "normalization_stats.json"

    if not split_index_path.exists():
        raise FileNotFoundError(f"Split index not found: {split_index_path}")

    # Load split index
    split_df = pd.read_csv(split_index_path)
    total_patches = len(split_df)

    # Filter strictly for split == 'train'
    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    train_count = len(train_df)
    val_count = len(val_df)
    test_count = len(test_df)

    print("=" * 75)
    print("CALCULATING TRAIN-ONLY NORMALIZATION STATISTICS")
    print("=" * 75)
    print(f"Total Dataset Patches:    {total_patches}")
    print(f"TRAIN Patches (INCLUDED): {train_count} ({train_count/total_patches*100:.1f}%)")
    print(f"VAL Patches (EXCLUDED):   {val_count} ({val_count/total_patches*100:.1f}%)")
    print(f"TEST Patches (EXCLUDED):  {test_count} ({test_count/total_patches*100:.1f}%)")
    print("=" * 75)

    # Verification assertions
    assert train_count + val_count + test_count == total_patches
    assert set(train_df["filename"]).isdisjoint(set(val_df["filename"]))
    assert set(train_df["filename"]).isdisjoint(set(test_df["filename"]))

    regions = ["forest", "urban", "agriculture", "water"]
    channel_names = ["B02", "B03", "B04", "B08"]

    # Histograms for exact calculation without RAM overhead (uint16 max is 65535)
    hr_global_hist = {c: np.zeros(65536, dtype=np.int64) for c in range(4)}
    lr_global_hist = {c: np.zeros(65536, dtype=np.int64) for c in range(4)}

    hr_region_hist = {r: {c: np.zeros(65536, dtype=np.int64) for c in range(4)} for r in regions}
    lr_region_hist = {r: {c: np.zeros(65536, dtype=np.int64) for c in range(4)} for r in regions}

    print("\nProcessing TRAIN patches (converting uint16 -> float32 / 10000.0)...")

    for idx, row in tqdm(train_df.iterrows(), total=train_count, desc="Computing Stats"):
        r = row["region"]
        fname = row["filename"]

        hr_path = hr_dir / r / fname
        lr_path = lr_dir / r / fname

        hr_patch = np.load(hr_path)  # (4, 256, 256) uint16
        lr_patch = np.load(lr_path)  # (4, 128, 128) uint16

        for c in range(4):
            hr_bincount = np.bincount(hr_patch[c].ravel(), minlength=65536)
            lr_bincount = np.bincount(lr_patch[c].ravel(), minlength=65536)

            hr_global_hist[c] += hr_bincount
            lr_global_hist[c] += lr_bincount

            hr_region_hist[r][c] += hr_bincount
            lr_region_hist[r][c] += lr_bincount

    # Compute Global & Region Statistics from Histograms
    hr_global_stats = {}
    lr_global_stats = {}

    for c, ch_name in enumerate(channel_names):
        hr_global_stats[ch_name] = calc_stats_from_hist(hr_global_hist[c], scale=10000.0)
        lr_global_stats[ch_name] = calc_stats_from_hist(lr_global_hist[c], scale=10000.0)

    hr_region_stats = {}
    lr_region_stats = {}

    for r in regions:
        hr_region_stats[r] = {}
        lr_region_stats[r] = {}
        for c, ch_name in enumerate(channel_names):
            hr_region_stats[r][ch_name] = calc_stats_from_hist(hr_region_hist[r][c], scale=10000.0)
            lr_region_stats[r][ch_name] = calc_stats_from_hist(lr_region_hist[r][c], scale=10000.0)

    # Formulate JSON Output
    stats_output = {
        "metadata": {
            "scale_factor": 10000,
            "scale_formula": "float32_value = uint16_value / 10000.0",
            "channels": channel_names,
            "train_patch_count": train_count,
            "total_dataset_patches": total_patches,
            "val_patch_count_excluded": val_count,
            "test_patch_count_excluded": test_count,
            "source_split_index": "data/processed/hr_split_index.csv",
        },
        "hr_train_stats": hr_global_stats,
        "lr_train_stats": lr_global_stats,
        "hr_train_stats_per_region": hr_region_stats,
        "lr_train_stats_per_region": lr_region_stats,
    }

    # Save to JSON
    with open(output_json_path, "w") as f:
        json.dump(stats_output, f, indent=2)

    print(f"\nSaved normalization statistics to: {output_json_path}")

    # Display Tables
    print("\n" + "=" * 85)
    print("GLOBAL TRAIN-SET NORMALIZATION STATISTICS (Reflectance = uint16 / 10000.0)")
    print("=" * 85)
    print(f"{'Channel':<8} | {'Type':<4} | {'Mean':<8} | {'Std':<8} | {'Min':<6} | {'Max':<6} | {'P1':<6} | {'P50':<6} | {'P99':<6} | {'P99.9':<6}")
    print("-" * 85)

    for ch_name in channel_names:
        hs = hr_global_stats[ch_name]
        ls = lr_global_stats[ch_name]
        print(f"{ch_name:<8} | {'HR':<4} | {hs['mean']:<8.4f} | {hs['std']:<8.4f} | {hs['min']:<6.4f} | {hs['max']:<6.4f} | {hs['p1']:<6.4f} | {hs['p50_median']:<6.4f} | {hs['p99']:<6.4f} | {hs['p99_9']:<6.4f}")
        print(f"{ch_name:<8} | {'LR':<4} | {ls['mean']:<8.4f} | {ls['std']:<8.4f} | {ls['min']:<6.4f} | {ls['max']:<6.4f} | {ls['p1']:<6.4f} | {ls['p50_median']:<6.4f} | {ls['p99']:<6.4f} | {ls['p99_9']:<6.4f}")
        print("-" * 85)

    print("\nPER-REGION HR TRAIN-SET MEAN & STD BREAKDOWN:")
    print("-" * 75)
    print(f"{'Region':<12} | {'Channel':<8} | {'Mean':<10} | {'Std':<10} | {'Min':<8} | {'Max':<8}")
    print("-" * 75)
    for r in regions:
        for ch_name in channel_names:
            s = hr_region_stats[r][ch_name]
            print(f"{r:<12} | {ch_name:<8} | {s['mean']:<10.4f} | {s['std']:<10.4f} | {s['min']:<8.4f} | {s['max']:<8.4f}")
        print("-" * 75)

    print("\nEXCLUSION VERIFICATION:")
    print(f"  Validation patches used: 0 (Checked {val_count} val patches excluded)")
    print(f"  Test patches used:       0 (Checked {test_count} test patches excluded)")
    print("=" * 85)


if __name__ == "__main__":
    calculate_normalization_stats()
