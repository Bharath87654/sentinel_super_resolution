"""
Read-Only QA Script for 4× Synthetic LR Patches

Scientific Scope
────────────────
This script verifies the construction of the SYNTHETIC 40m-equivalent → 10m
training dataset only. It confirms that the 4×4 area-average degradation was
applied correctly to produce LR4x (4,64,64) from HR (4,256,256).

It does NOT validate real 10m → 2.5m super-resolution performance.
It does NOT verify that the model generalizes to real 10m Sentinel-2 inputs.

Checks Performed
────────────────
1.  Patch count: exactly 6,456 LR4x files.
2.  Index membership: every LR4x file is present in lr4x_patches_index.csv.
3.  HR correspondence: every LR4x filename has a matching HR .npy file.
4.  Duplicate filenames: none.
5.  Shape: (4, 64, 64) for all patches.
6.  Dtype: uint16 for all patches.
7.  NaN: zero.
8.  Inf: zero.
9.  All-zero patches: zero.
10. Channel nonzero: all four channels contain nonzero values globally.
11. Region counts: match HR index and split index.
12. Degradation exactness (sampled): recompute 4×4 area average from HR
    and compare to stored LR4x. Expect max_abs_diff = 0 after rounding.
13. Aggregate statistics: per-region and overall mean/std for all four bands.
14. HR↔LR4x variance relationship: confirm LR4x std ≤ HR std per channel,
    as expected from spatial averaging (variance reduction property).

Output
──────
  Terminal: PASS/FAIL summary per check.
  File:     outputs/inspect_lr4x_report.json
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
project_root      = Path(__file__).resolve().parent.parent.parent
hr_dir            = project_root / "data" / "processed" / "hr"
lr4x_dir          = project_root / "data" / "processed" / "lr4x"
hr_index_path     = project_root / "data" / "processed" / "hr_patches_index.csv"
lr4x_index_path   = project_root / "data" / "processed" / "lr4x_patches_index.csv"
split_index_path  = project_root / "data" / "processed" / "hr_split_index.csv"
report_path       = project_root / "outputs" / "inspect_lr4x_report.json"

EXPECTED_COUNT = 6456
EXPECTED_LR4X_SHAPE = (4, 64, 64)
EXPECTED_HR_SHAPE   = (4, 256, 256)
REGIONS = ["forest", "urban", "agriculture", "water"]
CHANNELS = ["B02", "B03", "B04", "B08"]

# Deterministic sample: 50 patches per region (200 total)
SAMPLE_PER_REGION = 50
RNG_SEED = 42


def _check(label: str, result: bool, detail: str = "") -> bool:
    status = "PASS" if result else "FAIL"
    detail_str = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{detail_str}")
    return result


def run_qa():
    print("=" * 72)
    print("LR4x (4×) DATASET QUALITY ASSURANCE — READ ONLY")
    print("=" * 72)

    qa = {}         # will be serialised to JSON
    failures = []   # collect names of failed checks

    # ─── Load Indices ─────────────────────────────────────────────────────────
    for p, label in [
        (hr_index_path,   "HR index"),
        (lr4x_index_path, "LR4x index"),
        (split_index_path,"Split index"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    hr_df    = pd.read_csv(hr_index_path)
    lr4x_df  = pd.read_csv(lr4x_index_path)
    split_df = pd.read_csv(split_index_path)

    print(f"\n  HR index rows:      {len(hr_df)}")
    print(f"  LR4x index rows:    {len(lr4x_df)}")
    print(f"  Split index rows:   {len(split_df)}")

    # ─── Check 1: Patch Count ─────────────────────────────────────────────────
    print("\n[Check 1] Patch count")
    n_lr4x_files = sum(
        len(os.listdir(lr4x_dir / r)) for r in REGIONS
        if (lr4x_dir / r).exists()
    )
    c1 = _check(
        f"Total LR4x files == {EXPECTED_COUNT}",
        n_lr4x_files == EXPECTED_COUNT,
        f"found {n_lr4x_files}",
    )
    qa["check1_patch_count"] = {"pass": c1, "found": n_lr4x_files, "expected": EXPECTED_COUNT}
    if not c1: failures.append("check1_patch_count")

    # ─── Check 2: Index Membership ────────────────────────────────────────────
    print("\n[Check 2] Index membership — every LR4x file in lr4x_patches_index.csv")
    index_filenames = set(lr4x_df["filename"])
    disk_filenames  = set()
    for r in REGIONS:
        for f in os.listdir(lr4x_dir / r):
            disk_filenames.add(f)

    in_index_not_disk = index_filenames - disk_filenames
    on_disk_not_index = disk_filenames  - index_filenames
    c2 = _check(
        "All disk files in index",
        len(on_disk_not_index) == 0,
        f"{len(on_disk_not_index)} extra on disk",
    )
    c2b = _check(
        "All index files on disk",
        len(in_index_not_disk) == 0,
        f"{len(in_index_not_disk)} missing from disk",
    )
    c2_pass = c2 and c2b
    qa["check2_index_membership"] = {
        "pass": c2_pass,
        "extra_on_disk": len(on_disk_not_index),
        "missing_from_disk": len(in_index_not_disk),
    }
    if not c2_pass: failures.append("check2_index_membership")

    # ─── Check 3: HR Correspondence ───────────────────────────────────────────
    print("\n[Check 3] HR correspondence — every LR4x has a matching HR file")
    hr_all_files = set()
    for r in REGIONS:
        for f in os.listdir(hr_dir / r):
            hr_all_files.add(f)

    missing_hr = disk_filenames - hr_all_files
    c3 = _check(
        "All LR4x filenames have HR counterparts",
        len(missing_hr) == 0,
        f"{len(missing_hr)} LR4x files without HR",
    )
    qa["check3_hr_correspondence"] = {"pass": c3, "missing_hr": len(missing_hr)}
    if not c3: failures.append("check3_hr_correspondence")

    # ─── Check 4: Duplicate Filenames ─────────────────────────────────────────
    print("\n[Check 4] Duplicate filenames in LR4x index")
    dup_count = int(lr4x_df["filename"].duplicated().sum())
    c4 = _check("No duplicate filenames", dup_count == 0, f"{dup_count} duplicates")
    qa["check4_no_duplicates"] = {"pass": c4, "duplicates": dup_count}
    if not c4: failures.append("check4_no_duplicates")

    # ─── Check 5: Region Counts ───────────────────────────────────────────────
    print("\n[Check 5] Region counts")
    hr_region_counts    = hr_df["region"].value_counts().to_dict()
    lr4x_region_counts  = lr4x_df["region"].value_counts().to_dict()
    split_region_counts = split_df["region"].value_counts().to_dict()

    region_match = True
    region_detail = {}
    for r in REGIONS:
        hr_c   = hr_region_counts.get(r, 0)
        lr4x_c = lr4x_region_counts.get(r, 0)
        sp_c   = split_region_counts.get(r, 0)
        ok = (hr_c == lr4x_c == sp_c)
        _check(f"Region '{r}' counts match", ok,
               f"HR={hr_c}  LR4x={lr4x_c}  split={sp_c}")
        region_match = region_match and ok
        region_detail[r] = {"hr": hr_c, "lr4x": lr4x_c, "split": sp_c, "match": ok}
    c5 = region_match
    qa["check5_region_counts"] = {"pass": c5, "detail": region_detail}
    if not c5: failures.append("check5_region_counts")

    # ─── Full Scan: Checks 6–10 ───────────────────────────────────────────────
    print("\n[Checks 6–10] Full structural scan (shape/dtype/NaN/Inf/zero) — all 6,456 patches")

    bad_shape   = 0
    bad_dtype   = 0
    nan_patches = 0
    inf_patches = 0
    zero_patches = 0
    channel_zero_global = [True, True, True, True]   # will be flipped if nonzero found

    # Histograms for aggregate statistics (uint16 range = 0..65535)
    lr4x_hist = {r: {c: np.zeros(65536, dtype=np.int64) for c in range(4)} for r in REGIONS}
    hr_hist   = {r: {c: np.zeros(65536, dtype=np.int64) for c in range(4)} for r in REGIONS}

    for _, row in tqdm(lr4x_df.iterrows(), total=len(lr4x_df), desc="Full scan"):
        r      = row["region"]
        fname  = row["filename"]
        lr4x_p = lr4x_dir / r / fname
        hr_p   = hr_dir   / r / fname

        lr4x_patch = np.load(lr4x_p)
        hr_patch   = np.load(hr_p)

        if lr4x_patch.shape != EXPECTED_LR4X_SHAPE:
            bad_shape += 1
        if lr4x_patch.dtype != np.uint16:
            bad_dtype += 1
        if np.isnan(lr4x_patch).any():
            nan_patches += 1
        if np.isinf(lr4x_patch).any():
            inf_patches += 1
        if np.all(lr4x_patch == 0):
            zero_patches += 1

        for c in range(4):
            if np.any(lr4x_patch[c] != 0):
                channel_zero_global[c] = False   # channel c has nonzero data
            lr4x_hist[r][c] += np.bincount(lr4x_patch[c].ravel(), minlength=65536)
            hr_hist[r][c]   += np.bincount(hr_patch[c].ravel(),   minlength=65536)

    c6  = _check("All shapes == (4,64,64)",       bad_shape  == 0, f"{bad_shape} bad")
    c7  = _check("All dtypes == uint16",           bad_dtype  == 0, f"{bad_dtype} bad")
    c8  = _check("NaN patches == 0",              nan_patches == 0, f"{nan_patches}")
    c9  = _check("Inf patches == 0",              inf_patches == 0, f"{inf_patches}")
    c10 = _check("All-zero patches == 0",         zero_patches == 0, f"{zero_patches}")
    c10b= _check("All 4 channels nonzero globally",
                 not any(channel_zero_global),
                 f"zero channels: {[CHANNELS[i] for i,z in enumerate(channel_zero_global) if z]}")

    for chk, name in [
        (c6,  "check6_shape"), (c7, "check7_dtype"),
        (c8,  "check8_nan"),   (c9, "check9_inf"),
        (c10, "check10_zero_patches"), (c10b, "check10b_channel_nonzero"),
    ]:
        qa[name] = {"pass": chk}
        if not chk: failures.append(name)

    # ─── Check 11: Degradation Exactness (Sampled) ────────────────────────────
    print(f"\n[Check 11] Degradation exactness — {SAMPLE_PER_REGION} random samples/region")
    rng = np.random.default_rng(RNG_SEED)

    max_abs_diff_global  = 0
    total_mismatch_px    = 0
    mean_abs_diffs       = []
    total_sampled        = 0

    for r in REGIONS:
        region_files = sorted(lr4x_df.loc[lr4x_df["region"] == r, "filename"].tolist())
        chosen = rng.choice(region_files, size=SAMPLE_PER_REGION, replace=False)

        for fname in chosen:
            lr4x_patch = np.load(lr4x_dir / r / fname)
            hr_patch   = np.load(hr_dir   / r / fname)

            # Independently recompute 4×4 area average
            recomputed = (
                hr_patch.astype(np.float32)
                .reshape(4, 64, 4, 64, 4)
                .mean(axis=(2, 4))
            )
            recomputed_u16 = np.round(recomputed).astype(np.uint16)

            diff = np.abs(
                lr4x_patch.astype(np.int32) - recomputed_u16.astype(np.int32)
            )
            max_abs_diff_global = max(max_abs_diff_global, int(diff.max()))
            total_mismatch_px  += int(np.count_nonzero(diff))
            mean_abs_diffs.append(float(diff.mean()))
            total_sampled += 1

    overall_mean_abs_diff = float(np.mean(mean_abs_diffs))

    print(f"    Sampled pairs:           {total_sampled}")
    print(f"    Max absolute difference: {max_abs_diff_global}")
    print(f"    Mean absolute difference:{overall_mean_abs_diff:.8f}")
    print(f"    Total mismatched pixels: {total_mismatch_px}")

    exact = (max_abs_diff_global == 0 and total_mismatch_px == 0)
    c11 = _check(
        "Degradation exact (max_diff=0, mismatch_px=0)", exact,
        f"max_diff={max_abs_diff_global}  mismatch_px={total_mismatch_px}",
    )
    qa["check11_degradation_exactness"] = {
        "pass": c11,
        "sampled_pairs": total_sampled,
        "max_abs_diff": max_abs_diff_global,
        "mean_abs_diff": overall_mean_abs_diff,
        "total_mismatch_pixels": total_mismatch_px,
    }
    if not c11: failures.append("check11_degradation_exactness")

    # ─── Check 12: Aggregate Statistics & HR↔LR4x Variance Relationship ──────
    print("\n[Check 12] Aggregate statistics & variance reduction check")

    def _stats_from_hist(hist, scale=10000.0):
        total = int(hist.sum())
        if total == 0:
            return {"mean": None, "std": None, "min": None, "max": None, "total_pixels": 0}
        vals = np.arange(len(hist), dtype=np.float64) / scale
        counts = hist.astype(np.float64)
        mean = float(np.sum(counts * vals) / total)
        sq   = float(np.sum(counts * vals ** 2) / total)
        var  = max(0.0, sq - mean ** 2)
        nz   = np.where(hist > 0)[0]
        return {
            "mean":          round(mean, 6),
            "std":           round(float(np.sqrt(var)), 6),
            "min":           round(float(nz[0]  / scale), 6),
            "max":           round(float(nz[-1] / scale), 6),
            "total_pixels":  total,
        }

    stats_lr4x = {}
    stats_hr   = {}
    variance_ok = True   # LR4x std ≤ HR std for all channels in all regions

    # Per-region stats
    for r in REGIONS:
        stats_lr4x[r] = {}
        stats_hr[r]   = {}
        for c, ch in enumerate(CHANNELS):
            sl = _stats_from_hist(lr4x_hist[r][c])
            sh = _stats_from_hist(hr_hist[r][c])
            stats_lr4x[r][ch] = sl
            stats_hr[r][ch]   = sh
            if sl["std"] is not None and sh["std"] is not None:
                if sl["std"] > sh["std"] + 1e-5:   # small tolerance for rounding
                    variance_ok = False

    # Overall stats (sum histograms across regions)
    overall_lr4x_hist = {c: sum(lr4x_hist[r][c] for r in REGIONS) for c in range(4)}
    overall_hr_hist   = {c: sum(hr_hist[r][c]   for r in REGIONS) for c in range(4)}
    stats_lr4x["overall"] = {}
    stats_hr["overall"]   = {}
    for c, ch in enumerate(CHANNELS):
        stats_lr4x["overall"][ch] = _stats_from_hist(overall_lr4x_hist[c])
        stats_hr["overall"][ch]   = _stats_from_hist(overall_hr_hist[c])

    # Print stats table
    print(f"\n  Overall Statistics (reflectance = uint16 / 10000):")
    print(f"  {'Chan':<6} | {'Source':<6} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print(f"  {'-'*60}")
    for ch in CHANNELS:
        sl = stats_lr4x["overall"][ch]
        sh = stats_hr["overall"][ch]
        print(f"  {ch:<6} | {'HR':<6} | {sh['mean']:>8.4f} | {sh['std']:>8.4f} | "
              f"{sh['min']:>8.4f} | {sh['max']:>8.4f}")
        print(f"  {ch:<6} | {'LR4x':<6} | {sl['mean']:>8.4f} | {sl['std']:>8.4f} | "
              f"{sl['min']:>8.4f} | {sl['max']:>8.4f}")
        print(f"  {'-'*60}")

    c12 = _check(
        "LR4x std ≤ HR std for all channels in all regions (variance reduction)",
        variance_ok,
        "" if variance_ok else "spatial averaging should reduce variance",
    )
    qa["check12_statistics"] = {
        "pass": c12,
        "lr4x_stats": stats_lr4x,
        "hr_stats":   stats_hr,
        "variance_reduction_satisfied": variance_ok,
    }
    if not c12: failures.append("check12_variance_reduction")

    # ─── Final PASS/FAIL Summary ──────────────────────────────────────────────
    all_passed = len(failures) == 0

    print("\n" + "=" * 72)
    print("QA SUMMARY")
    print("=" * 72)
    checks_done = [
        "check1_patch_count",  "check2_index_membership",
        "check3_hr_correspondence", "check4_no_duplicates",
        "check5_region_counts", "check6_shape", "check7_dtype",
        "check8_nan", "check9_inf", "check10_zero_patches",
        "check10b_channel_nonzero", "check11_degradation_exactness",
        "check12_variance_reduction",
    ]
    for name in checks_done:
        passed = name not in failures
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    print("=" * 72)
    if all_passed:
        print("OVERALL: ALL CHECKS PASSED")
    else:
        print(f"OVERALL: {len(failures)} CHECK(S) FAILED — {failures}")
    print("=" * 72)
    print("\nScientific scope:")
    print("  This QA verifies synthetic 40m-equivalent → 10m training data only.")
    print("  It does NOT validate real 10m → 2.5m super-resolution performance.")
    print("=" * 72)

    # ─── Save JSON Report ─────────────────────────────────────────────────────
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "scientific_scope": (
            "Verifies synthetic 40m-equivalent LR4x patch construction from 10m HR patches. "
            "Does not validate real 10m → 2.5m super-resolution performance."
        ),
        "overall_pass": all_passed,
        "failed_checks": failures,
        "expected_patch_count": EXPECTED_COUNT,
        "found_patch_count": n_lr4x_files,
        "sample_size_per_region_for_exactness_check": SAMPLE_PER_REGION,
        "rng_seed": RNG_SEED,
        "qa_results": qa,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved QA report: {report_path}")


if __name__ == "__main__":
    run_qa()
