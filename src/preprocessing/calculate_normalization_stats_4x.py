"""
Calculate 4× Pipeline Normalization Statistics (Train Split Only)

Scientific Scope
────────────────
This script computes training-set normalization statistics for the 4× SR
pipeline. The training task is:

  LR4x (4,64,64)  40m-equivalent synthetic  →  HR (4,256,256) 10m target

CANONICAL NORMALIZATION CONVENTION:
  HR TRAIN mean/std are used to Z-score normalize BOTH LR4x inputs and HR
  targets. This is consistent with the approved 2× pipeline design and
  physically justified: 4×4 spatial area averaging is a linear operation
  that preserves per-channel mean. LR4x std is slightly lower (spatial
  averaging reduces intra-patch variance), but the same HR reference
  statistics keep both modalities in the same spectral/physical space.

  LR4x-specific statistics are computed here for AUDITING only.
  They must NOT be used as normalization parameters for training.

IMPORTANT: This script does not validate real 10m → 2.5m SR performance.
           It only characterises the synthetic training data.

Isolation Guarantee
────────────────────
Reads:
  data/processed/hr/                      (HR patches)
  data/processed/lr4x/                    (LR4x patches)
  data/processed/hr_split_index.csv       (split membership)
  data/processed/normalization_stats.json (existing 2× stats — read only)

Writes:
  data/processed/normalization_stats_4x.json  (NEW — only new file created)

No existing file is modified.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
project_root       = Path(__file__).resolve().parent.parent.parent
hr_dir             = project_root / "data" / "processed" / "hr"
lr4x_dir           = project_root / "data" / "processed" / "lr4x"
split_index_path   = project_root / "data" / "processed" / "hr_split_index.csv"
stats2x_path       = project_root / "data" / "processed" / "normalization_stats.json"
output_stats_path  = project_root / "data" / "processed" / "normalization_stats_4x.json"

CHANNELS           = ["B02", "B03", "B04", "B08"]
REFLECTANCE_DIVISOR = 10000.0
SCALE_FACTOR        = 4
EXPECTED_TRAIN_COUNT = 5119

# Cross-validation tolerance: canonical HR stats must agree with 2× stats to
# within this absolute reflectance difference to be considered consistent.
CROSS_VALIDATION_TOLERANCE = 1e-4   # 0.0001 in reflectance units


def _stats_from_hist(hist: np.ndarray, divisor: float = 10000.0) -> dict:
    """
    Compute descriptive statistics from a uint16 pixel-frequency histogram.

    Parameters
    ----------
    hist    : np.ndarray, shape (65536,), dtype int64
              Pixel value counts for a single channel.
    divisor : float
              Scale applied to convert uint16 counts to physical values.

    Returns
    -------
    dict with keys: mean, std, min, max, p1, p50_median, p99, p99_9,
                    total_pixels.
    """
    total = int(hist.sum())
    if total == 0:
        return {k: None for k in
                ["mean", "std", "min", "max", "p1", "p50_median",
                 "p99", "p99_9", "total_pixels"]}

    values  = np.arange(65536, dtype=np.float64) / divisor
    counts  = hist.astype(np.float64)

    mean_val = float(np.sum(counts * values) / total)
    sq_mean  = float(np.sum(counts * (values ** 2)) / total)
    var_val  = max(0.0, sq_mean - mean_val ** 2)
    std_val  = float(np.sqrt(var_val))

    nz_indices = np.where(hist > 0)[0]
    min_val    = float(nz_indices[0]  / divisor)
    max_val    = float(nz_indices[-1] / divisor)

    cum = np.cumsum(counts)

    def _percentile(p: float) -> float:
        target = (p / 100.0) * total
        idx    = int(np.searchsorted(cum, target))
        idx    = min(idx, 65535)
        return round(float(idx / divisor), 6)

    return {
        "mean":         round(mean_val, 6),
        "std":          round(std_val,  6),
        "min":          round(min_val,  6),
        "max":          round(max_val,  6),
        "p1":           _percentile(1.0),
        "p50_median":   _percentile(50.0),
        "p99":          _percentile(99.0),
        "p99_9":        _percentile(99.9),
        "total_pixels": total,
    }


def calculate_normalization_stats_4x():
    print("=" * 72)
    print("4× PIPELINE — TRAIN-ONLY NORMALIZATION STATISTICS")
    print("=" * 72)

    # ─── Verify Required Files ────────────────────────────────────────────────
    for p, label in [
        (split_index_path, "Split index"),
        (stats2x_path,     "Existing 2× normalization stats"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    # ─── Load Split Index & Filter TRAIN Only ─────────────────────────────────
    split_df  = pd.read_csv(split_index_path)
    train_df  = split_df[split_df["split"] == "train"].copy().reset_index(drop=True)
    val_df    = split_df[split_df["split"] == "val"].copy()
    test_df   = split_df[split_df["split"] == "test"].copy()

    n_train = len(train_df)
    n_val   = len(val_df)
    n_test  = len(test_df)
    n_total = len(split_df)

    print(f"\n  Split index total rows: {n_total}")
    print(f"  TRAIN (INCLUDED):       {n_train}  (expected {EXPECTED_TRAIN_COUNT})")
    print(f"  VAL   (EXCLUDED):       {n_val}")
    print(f"  TEST  (EXCLUDED):       {n_test}")

    # ── Hard assertion: correct train count ───────────────────────────────────
    if n_train != EXPECTED_TRAIN_COUNT:
        raise ValueError(
            f"TRAIN count mismatch: got {n_train}, expected {EXPECTED_TRAIN_COUNT}."
        )
    print(f"  PASS: Train count = {n_train}")

    # ── Confirm strict disjointness ───────────────────────────────────────────
    train_fnames = set(train_df["filename"])
    val_fnames   = set(val_df["filename"])
    test_fnames  = set(test_df["filename"])

    assert train_fnames.isdisjoint(val_fnames),  "FAIL: train ∩ val is non-empty"
    assert train_fnames.isdisjoint(test_fnames), "FAIL: train ∩ test is non-empty"
    print("  PASS: Val and test patches are strictly excluded from statistics.")

    # ─── Accumulate Pixel Histograms Over TRAIN Set Only ─────────────────────
    # Uint16 histograms: one per channel per dataset (HR, LR4x)
    hr_hist   = {c: np.zeros(65536, dtype=np.int64) for c in range(4)}
    lr4x_hist = {c: np.zeros(65536, dtype=np.int64) for c in range(4)}

    print(f"\n  Processing {n_train} TRAIN patches...")

    for _, row in tqdm(train_df.iterrows(), total=n_train, desc="Computing stats"):
        region = row["region"]
        fname  = row["filename"]

        hr_patch   = np.load(hr_dir   / region / fname)   # (4,256,256) uint16
        lr4x_patch = np.load(lr4x_dir / region / fname)   # (4,64,64)   uint16

        for c in range(4):
            hr_hist[c]   += np.bincount(hr_patch[c].ravel(),   minlength=65536)
            lr4x_hist[c] += np.bincount(lr4x_patch[c].ravel(), minlength=65536)

    # ─── Compute Statistics from Histograms ───────────────────────────────────
    print("\n  Computing statistics from pixel histograms...")

    hr_stats   = {}
    lr4x_stats = {}
    for c, ch in enumerate(CHANNELS):
        hr_stats[ch]   = _stats_from_hist(hr_hist[c],   REFLECTANCE_DIVISOR)
        lr4x_stats[ch] = _stats_from_hist(lr4x_hist[c], REFLECTANCE_DIVISOR)

    # ─── Print Statistics Table ───────────────────────────────────────────────
    print("\n  Statistics Summary (reflectance = uint16 / 10000):")
    print(f"  {'Chan':<5} | {'Source':<6} | {'Mean':>8} | {'Std':>8} | "
          f"{'Min':>7} | {'Max':>7} | {'P99':>7}")
    print(f"  {'-'*70}")
    for ch in CHANNELS:
        hs = hr_stats[ch]
        ls = lr4x_stats[ch]
        print(f"  {ch:<5} | {'HR':<6} | {hs['mean']:>8.4f} | {hs['std']:>8.4f} | "
              f"{hs['min']:>7.4f} | {hs['max']:>7.4f} | {hs['p99']:>7.4f}")
        print(f"  {ch:<5} | {'LR4x':<6} | {ls['mean']:>8.4f} | {ls['std']:>8.4f} | "
              f"{ls['min']:>7.4f} | {ls['max']:>7.4f} | {ls['p99']:>7.4f}")
        print(f"  {'-'*70}")

    # ─── Cross-Validation Against Existing 2× Statistics ──────────────────────
    print("\n  Cross-validating HR TRAIN statistics against existing 2× stats...")

    with open(stats2x_path, "r") as f:
        stats2x = json.load(f)

    stats2x_hr = stats2x["hr_train_stats"]

    cross_val_results = {}
    cross_val_passed  = True
    tolerance         = CROSS_VALIDATION_TOLERANCE

    for ch in CHANNELS:
        mean_2x = stats2x_hr[ch]["mean"]
        std_2x  = stats2x_hr[ch]["std"]
        mean_4x = hr_stats[ch]["mean"]
        std_4x  = hr_stats[ch]["std"]

        delta_mean = abs(mean_4x - mean_2x)
        delta_std  = abs(std_4x  - std_2x)

        ok_mean = delta_mean <= tolerance
        ok_std  = delta_std  <= tolerance
        ok      = ok_mean and ok_std

        cross_val_results[ch] = {
            "hr_train_mean_2x":    mean_2x,
            "hr_train_mean_4x":    mean_4x,
            "delta_mean":          round(delta_mean, 8),
            "mean_within_tolerance": ok_mean,
            "hr_train_std_2x":     std_2x,
            "hr_train_std_4x":     std_4x,
            "delta_std":           round(delta_std, 8),
            "std_within_tolerance":  ok_std,
            "tolerance":           tolerance,
        }

        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] {ch}: "
              f"Δmean={delta_mean:.2e}  Δstd={delta_std:.2e}  "
              f"(tolerance={tolerance:.0e})")

        if not ok:
            cross_val_passed = False
            print(f"    *** MISMATCH in {ch} ***")
            print(f"        2× mean={mean_2x:.6f}  4× mean={mean_4x:.6f}")
            print(f"        2× std ={std_2x:.6f}   4× std ={std_4x:.6f}")
            print(f"        This may indicate that the 4× train/split differs from 2×.")
            print(f"        Do NOT silently ignore this — investigate before training.")

    if cross_val_passed:
        print("  PASS: HR TRAIN statistics are consistent with the 2× pipeline.")
    else:
        print("  FAIL: HR TRAIN statistics DIVERGE from the 2× pipeline.")
        print("        The normalization_stats_4x.json will be written with both")
        print("        versions recorded. Do NOT proceed to training until resolved.")

    # ─── Canonical Normalization Parameters ───────────────────────────────────
    # These are the values to load in sentinel_sr_dataset_4x.py
    canonical_mean = {ch: hr_stats[ch]["mean"] for ch in CHANNELS}
    canonical_std  = {ch: hr_stats[ch]["std"]  for ch in CHANNELS}

    print("\n  Canonical normalization parameters (HR TRAIN, used for BOTH LR4x & HR):")
    for ch in CHANNELS:
        print(f"    {ch}: mean={canonical_mean[ch]:.6f}  std={canonical_std[ch]:.6f}")

    # ─── Build JSON Output ────────────────────────────────────────────────────
    output = {
        "scientific_scope": (
            "Training statistics for the 4× synthetic SR pipeline. "
            "HR TRAIN mean/std are canonical normalization parameters for "
            "both LR4x inputs and HR targets. "
            "Training task: synthetic 40m-equivalent LR4x → 10m HR. "
            "Deployment task: real 10m Sentinel-2 → 2.5m-scale inferred output. "
            "No sub-10m ground-truth validation is claimed."
        ),
        "metadata": {
            "scale_factor":         SCALE_FACTOR,
            "reflectance_divisor":  REFLECTANCE_DIVISOR,
            "reflectance_formula":  "float32_value = uint16_value / 10000.0",
            "channels":             CHANNELS,
            "channel_order":        "B02, B03, B04, B08",
            "train_patch_count":    n_train,
            "val_patch_count_excluded":  n_val,
            "test_patch_count_excluded": n_test,
            "source_split_index":   "data/processed/hr_split_index.csv",
        },
        "canonical_normalization": {
            "description": (
                "Use these values to Z-score normalize BOTH LR4x inputs and HR "
                "targets. LR4x audit_stats below are for reference only — do NOT "
                "use them as training normalization parameters."
            ),
            "source": "HR TRAIN patches — mean and std",
            "mean":   canonical_mean,
            "std":    canonical_std,
        },
        "cross_validation_against_2x_stats": {
            "tolerance": tolerance,
            "all_within_tolerance": cross_val_passed,
            "per_channel": cross_val_results,
        },
        "hr_train_stats": hr_stats,
        "lr4x_train_stats_audit": {
            "description": (
                "LR4x statistics are provided for auditing and scientific "
                "transparency only. They confirm that spatial averaging "
                "preserves per-channel mean while reducing std. "
                "These values are NOT used as normalization parameters."
            ),
            "stats": lr4x_stats,
        },
    }

    # ─── Write Output JSON ────────────────────────────────────────────────────
    output_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_stats_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {output_stats_path}")

    # ─── Final Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("STAGE 3 SUMMARY")
    print("=" * 72)
    print(f"  Train patches processed:    {n_train} (val/test strictly excluded)")
    print(f"  HR stats cross-val passed:  {cross_val_passed}")
    print(f"  Output JSON written:        {output_stats_path.name}")
    print(f"  Canonical mean (B02..B08):  "
          f"{[canonical_mean[ch] for ch in CHANNELS]}")
    print(f"  Canonical std  (B02..B08):  "
          f"{[canonical_std[ch] for ch in CHANNELS]}")
    print("=" * 72)
    if not cross_val_passed:
        print("  ACTION REQUIRED: HR stat mismatch detected. Investigate before")
        print("  proceeding to Stage 4 (dataset class) or training.")
    else:
        print("  Safe to proceed to Stage 4 (sentinel_sr_dataset_4x.py).")
    print("=" * 72)


if __name__ == "__main__":
    calculate_normalization_stats_4x()