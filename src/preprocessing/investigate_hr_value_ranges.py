"""
investigate_hr_value_ranges.py

Purpose:
    READ-ONLY diagnostic to investigate HR patches with unusually high
    pixel values (max > 20000).  Sentinel-2 L2A uint16 reflectance uses
    a nominal scale of 10000 = 100%, so values above 20000 represent
    >200% apparent reflectance and warrant inspection.

    This script does NOT modify any existing files or data.

    Outputs diagnostic reports and visualizations to:
        outputs/hr_value_diagnostics/

Usage:
    python src/preprocessing/investigate_hr_value_ranges.py
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import time


# ── Thresholds ───────────────────────────────────────────────────────
VALUE_THRESHOLDS = [10000, 15000, 20000]
HIGH_VALUE_CUTOFF = 20000  # patches with max > this are investigated

CHANNEL_NAMES = ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)"]
CHANNEL_INDICES = {"B02": 0, "B03": 1, "B04": 2, "B08": 3}
REGIONS = ["forest", "urban", "agriculture", "water"]


def stretch_to_uint8(band: np.ndarray) -> np.ndarray:
    """2-98 percentile stretch for VISUALIZATION ONLY."""
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    stretched = (band.astype(np.float32) - lo) / (hi - lo + 1e-6)
    return (np.clip(stretched, 0, 1) * 255).astype(np.uint8)


def load_metadata_csv(csv_path: Path) -> dict:
    """
    Load outputs/patches_metadata.csv and index by (region, row_off, col_off)
    for fast lookup.  Returns a dict mapping tuple keys to row dicts.
    """
    if not csv_path.exists():
        print(f"  [WARNING] Metadata CSV not found: {csv_path}")
        return {}

    lookup = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["region"], int(float(row["row_off"])),
                   int(float(row["col_off"])))
            lookup[key] = row
    return lookup


def analyze_high_value_patch(arr: np.ndarray, filename: str,
                              region: str) -> dict:
    """
    Compute detailed diagnostics for a single patch with max > HIGH_VALUE_CUTOFF.
    """
    C, H, W = arr.shape
    total_pixels = H * W  # per channel
    total_all = C * H * W

    # Overall stats
    global_max = int(arr.max())
    global_min = int(arr.min())

    # Find which channel has the global max
    max_ch_idx = int(np.unravel_index(arr.argmax(), arr.shape)[0])
    max_ch_name = CHANNEL_NAMES[max_ch_idx]

    # Per-channel maximums
    per_ch_max = [int(arr[c].max()) for c in range(C)]

    # Per-channel percentage above each threshold
    per_ch_above = {}
    for thresh in VALUE_THRESHOLDS:
        per_ch_above[thresh] = [
            float(np.sum(arr[c] > thresh) / total_pixels * 100)
            for c in range(C)
        ]

    # Overall percentage above each threshold (across all channels)
    overall_above = {}
    for thresh in VALUE_THRESHOLDS:
        overall_above[thresh] = float(
            np.sum(arr > thresh) / total_all * 100
        )

    # Spatial analysis: where are the high pixels (>= HIGH_VALUE_CUTOFF)?
    high_mask = arr.max(axis=0) > HIGH_VALUE_CUTOFF  # (H, W) union across channels
    n_high_pixels = int(high_mask.sum())
    high_pixel_pct = float(n_high_pixels / total_pixels * 100)

    # Spatial coordinates of high pixels
    high_rows, high_cols = np.where(high_mask)

    # Spatial concentration: are they clustered or scattered?
    if n_high_pixels > 0:
        row_span = int(high_rows.max() - high_rows.min() + 1)
        col_span = int(high_cols.max() - high_cols.min() + 1)
        bounding_box_area = row_span * col_span
        # Density = fraction of bounding box actually occupied
        density = n_high_pixels / bounding_box_area if bounding_box_area > 0 else 0
        centroid_row = float(high_rows.mean())
        centroid_col = float(high_cols.mean())
    else:
        row_span = col_span = 0
        density = 0.0
        centroid_row = centroid_col = 0.0

    # Multi-channel analysis: do high values appear in one channel or many?
    channels_with_high = []
    for c in range(C):
        if arr[c].max() > HIGH_VALUE_CUTOFF:
            channels_with_high.append(CHANNEL_NAMES[c])

    return {
        "filename": filename,
        "region": region,
        "global_max": global_max,
        "global_min": global_min,
        "max_channel": max_ch_name,
        "max_channel_idx": max_ch_idx,
        "per_ch_max": per_ch_max,
        "per_ch_above": per_ch_above,
        "overall_above": overall_above,
        "n_high_pixels": n_high_pixels,
        "high_pixel_pct": high_pixel_pct,
        "high_pixel_rows": high_rows,
        "high_pixel_cols": high_cols,
        "row_span": row_span,
        "col_span": col_span,
        "spatial_density": density,
        "centroid": (centroid_row, centroid_col),
        "channels_with_high": channels_with_high,
        "n_channels_affected": len(channels_with_high),
    }


def save_diagnostic_figure(arr: np.ndarray, diag: dict,
                            output_path: Path):
    """
    Create a diagnostic figure with:
      Row 1: RGB preview | NIR channel
      Row 2: mask >10000 | mask >15000 | mask >20000
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)

    # ── Row 1, Col 0: RGB preview ──
    ax_rgb = fig.add_subplot(gs[0, 0])
    rgb = np.stack([
        stretch_to_uint8(arr[2]),  # B04 = Red
        stretch_to_uint8(arr[1]),  # B03 = Green
        stretch_to_uint8(arr[0]),  # B02 = Blue
    ], axis=-1)
    ax_rgb.imshow(rgb)
    ax_rgb.set_title("RGB Preview (B04/B03/B02)\n[2-98% stretch, viz only]",
                     fontsize=9)
    ax_rgb.axis("off")

    # ── Row 1, Col 1: NIR channel ──
    ax_nir = fig.add_subplot(gs[0, 1])
    nir_viz = stretch_to_uint8(arr[3])
    ax_nir.imshow(nir_viz, cmap="gray")
    ax_nir.set_title(f"B08 (NIR)\nmax={diag['per_ch_max'][3]}", fontsize=9)
    ax_nir.axis("off")

    # ── Row 1, Col 2: Info text ──
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.axis("off")
    info_lines = [
        f"File: {diag['filename']}",
        f"Region: {diag['region']}",
        f"Global max: {diag['global_max']}",
        f"Global min: {diag['global_min']}",
        f"Max channel: {diag['max_channel']}",
        f"",
        f"Per-channel max:",
        f"  B02={diag['per_ch_max'][0]}",
        f"  B03={diag['per_ch_max'][1]}",
        f"  B04={diag['per_ch_max'][2]}",
        f"  B08={diag['per_ch_max'][3]}",
        f"",
        f"Channels >20000: {', '.join(diag['channels_with_high'])}",
        f"High pixels: {diag['n_high_pixels']} ({diag['high_pixel_pct']:.3f}%)",
        f"Bounding box: {diag['row_span']}×{diag['col_span']}",
        f"Spatial density: {diag['spatial_density']:.3f}",
    ]
    ax_info.text(0.05, 0.95, "\n".join(info_lines), transform=ax_info.transAxes,
                 fontsize=8, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # ── Row 2: Threshold masks ──
    threshold_labels = [">10000", ">15000", ">20000"]
    for i, thresh in enumerate(VALUE_THRESHOLDS):
        ax = fig.add_subplot(gs[1, i])
        # Union mask across all 4 channels
        mask = np.max(arr, axis=0) > thresh
        n_px = int(mask.sum())
        pct = n_px / (arr.shape[1] * arr.shape[2]) * 100

        # Show RGB underneath with mask overlay
        ax.imshow(rgb, alpha=0.5)
        ax.imshow(mask, cmap="Reds", alpha=0.6, vmin=0, vmax=1)
        ax.set_title(f"Pixels {threshold_labels[i]}\n{n_px} px ({pct:.2f}%)",
                     fontsize=9)
        ax.axis("off")

    fig.suptitle(f"HR Value Diagnostic: {diag['filename']}", fontsize=11,
                 fontweight="bold")
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    hr_dir = project_root / "data" / "processed" / "hr"
    index_csv = project_root / "data" / "processed" / "hr_patches_index.csv"
    metadata_csv = project_root / "outputs" / "patches_metadata.csv"
    output_dir = project_root / "outputs" / "hr_value_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load index ───────────────────────────────────────────────
    print("=" * 70)
    print("HR VALUE RANGE INVESTIGATION")
    print("=" * 70)

    df_index = pd.read_csv(index_csv)
    print(f"  HR index loaded: {len(df_index)} patches")

    # ── Load full metadata for cross-reference ───────────────────
    meta_lookup = load_metadata_csv(metadata_csv)
    print(f"  Metadata CSV loaded: {len(meta_lookup)} records")

    # ── Full scan: find patches with max > HIGH_VALUE_CUTOFF ─────
    print(f"\n  Scanning ALL patches for max > {HIGH_VALUE_CUTOFF}...")
    start = time.time()

    affected_diagnostics = []
    total_scanned = 0
    all_maxes = []  # track max value per patch for overall distribution

    for region in REGIONS:
        region_dir = hr_dir / region
        if not region_dir.exists():
            continue
        npy_files = sorted(region_dir.glob("*.npy"))

        for fpath in npy_files:
            arr = np.load(fpath)
            total_scanned += 1
            patch_max = int(arr.max())
            all_maxes.append(patch_max)

            if patch_max > HIGH_VALUE_CUTOFF:
                diag = analyze_high_value_patch(arr, fpath.name, region)

                # Cross-reference with metadata
                # Parse row_off and col_off from filename pattern: region_rXXX_cYYY.npy
                parts = fpath.stem.split("_")
                row_off_str = [p for p in parts if p.startswith("r")][0]
                col_off_str = [p for p in parts if p.startswith("c")][0]
                row_off = int(row_off_str[1:])
                col_off = int(col_off_str[1:])

                meta_key = (region, row_off, col_off)
                meta_row = meta_lookup.get(meta_key, {})
                diag["meta"] = meta_row

                affected_diagnostics.append(diag)

                # Save diagnostic figure
                fig_name = fpath.stem + "_diagnostic.png"
                save_diagnostic_figure(arr, diag, output_dir / fig_name)

            if total_scanned % 1000 == 0:
                print(f"    ... {total_scanned} patches scanned")

    elapsed = time.time() - start
    print(f"  Scan complete: {total_scanned} patches in {elapsed:.1f}s")
    print(f"  Affected patches: {len(affected_diagnostics)}")

    # ── Detailed report for each affected patch ──────────────────
    print("\n" + "=" * 70)
    print("DETAILED REPORT: PATCHES WITH MAX > 20000")
    print("=" * 70)

    meta_fields_to_report = [
        "bad_pct", "unclassified_pct", "vegetation_pct", "bare_soil_pct",
        "water_pct", "cloud_shadow_pct", "cloud_med_pct", "cloud_high_pct",
        "cirrus_pct",
    ]

    for idx, diag in enumerate(affected_diagnostics):
        print(f"\n{'─' * 60}")
        print(f"  [{idx + 1}/{len(affected_diagnostics)}] {diag['filename']}")
        print(f"{'─' * 60}")
        print(f"  Region           : {diag['region']}")
        print(f"  Global max       : {diag['global_max']}")
        print(f"  Global min       : {diag['global_min']}")
        print(f"  Max channel      : {diag['max_channel']}")
        print(f"  Channels >20000  : {', '.join(diag['channels_with_high'])} "
              f"({diag['n_channels_affected']} of 4)")

        print(f"\n  Per-channel maximums:")
        for c in range(4):
            print(f"    {CHANNEL_NAMES[c]:14s}: max = {diag['per_ch_max'][c]}")

        print(f"\n  Per-channel % above thresholds:")
        print(f"    {'Channel':14s}  {'> 10000':>10s}  {'> 15000':>10s}  {'> 20000':>10s}")
        for c in range(4):
            vals = [f"{diag['per_ch_above'][t][c]:8.4f}%" for t in VALUE_THRESHOLDS]
            print(f"    {CHANNEL_NAMES[c]:14s}  {'  '.join(vals)}")

        print(f"\n  Overall % above thresholds (all channels combined):")
        for thresh in VALUE_THRESHOLDS:
            print(f"    > {thresh}: {diag['overall_above'][thresh]:.4f}%")

        print(f"\n  Spatial analysis (pixels > {HIGH_VALUE_CUTOFF}):")
        print(f"    Count          : {diag['n_high_pixels']} pixels")
        print(f"    % of patch     : {diag['high_pixel_pct']:.4f}%")
        print(f"    Bounding box   : {diag['row_span']} × {diag['col_span']}")
        print(f"    Density in bbox: {diag['spatial_density']:.4f}")
        print(f"    Centroid (r,c) : ({diag['centroid'][0]:.1f}, "
              f"{diag['centroid'][1]:.1f})")

        if diag["n_high_pixels"] > 0 and diag["n_high_pixels"] <= 20:
            coords = list(zip(diag["high_pixel_rows"],
                              diag["high_pixel_cols"]))
            print(f"    Coordinates    : {coords}")

        meta = diag.get("meta", {})
        if meta:
            print(f"\n  SCL metadata (from patches_metadata.csv):")
            for field in meta_fields_to_report:
                val = meta.get(field, "N/A")
                print(f"    {field:22s}: {val}")
        else:
            print(f"\n  [WARNING] No metadata match found for this patch.")

    # ── Per-region summary ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY BY REGION")
    print("=" * 70)

    region_counts = {}
    for diag in affected_diagnostics:
        region_counts.setdefault(diag["region"], []).append(diag)

    region_totals = df_index["region"].value_counts().to_dict()
    for region in REGIONS:
        affected_in_region = region_counts.get(region, [])
        total_in_region = region_totals.get(region, 0)
        pct = (len(affected_in_region) / total_in_region * 100
               if total_in_region > 0 else 0)
        print(f"  {region:15s}: {len(affected_in_region):3d} affected / "
              f"{total_in_region} total ({pct:.2f}%)")
        if affected_in_region:
            maxes = [d["global_max"] for d in affected_in_region]
            print(f"    Max values: {sorted(maxes, reverse=True)}")

    # ── Overall statistics ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    all_maxes_arr = np.array(all_maxes)

    print(f"  Total patches scanned     : {total_scanned}")
    print(f"  Patches with max > 20000  : {len(affected_diagnostics)}")
    print(f"  % of dataset affected     : "
          f"{len(affected_diagnostics) / total_scanned * 100:.3f}%")
    print(f"  Maximum observed value    : {int(all_maxes_arr.max())}")
    print(f"  Median patch-max          : {int(np.median(all_maxes_arr))}")
    print(f"  95th percentile patch-max : {int(np.percentile(all_maxes_arr, 95))}")
    print(f"  99th percentile patch-max : {int(np.percentile(all_maxes_arr, 99))}")

    # Distribution of patch maximums
    print(f"\n  Patch maximum distribution:")
    bins = [(0, 5000), (5000, 8000), (8000, 10000), (10000, 12000),
            (12000, 15000), (15000, 20000), (20000, 30000), (30000, 65536)]
    for lo, hi in bins:
        count = int(np.sum((all_maxes_arr >= lo) & (all_maxes_arr < hi)))
        pct = count / total_scanned * 100
        print(f"    {lo:5d}–{hi:5d}: {count:5d} patches ({pct:.2f}%)")

    # ── Concentration analysis across affected patches ───────────
    print(f"\n  High-value concentration analysis:")
    total_high_pixels = sum(d["n_high_pixels"] for d in affected_diagnostics)
    avg_high_pct = (np.mean([d["high_pixel_pct"] for d in affected_diagnostics])
                    if affected_diagnostics else 0)
    max_high_pct = (max(d["high_pixel_pct"] for d in affected_diagnostics)
                    if affected_diagnostics else 0)

    # Channel distribution: which channels contribute?
    ch_contribution = {name: 0 for name in CHANNEL_NAMES}
    for diag in affected_diagnostics:
        for ch_name in diag["channels_with_high"]:
            ch_contribution[ch_name] += 1

    print(f"    Total high pixels (>20000) across all affected patches: "
          f"{total_high_pixels}")
    print(f"    Avg % of patch that is high-value: {avg_high_pct:.4f}%")
    print(f"    Max % of patch that is high-value: {max_high_pct:.4f}%")
    print(f"\n    Channels exceeding 20000 (how many patches each):")
    for ch_name, count in ch_contribution.items():
        print(f"      {ch_name:14s}: {count} patches")

    multi_ch = sum(1 for d in affected_diagnostics if d["n_channels_affected"] > 1)
    single_ch = sum(1 for d in affected_diagnostics if d["n_channels_affected"] == 1)
    print(f"\n    Single-channel high values: {single_ch} patches")
    print(f"    Multi-channel high values : {multi_ch} patches")

    # ── Classification ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CLASSIFICATION")
    print("=" * 70)

    if not affected_diagnostics:
        print("  [NO CONCERN] No patches exceed the threshold.")
    elif max_high_pct < 0.1:
        print("  [ISOLATED HIGH-VALUE PIXELS]")
        print("  The high values are isolated to a tiny fraction of the affected")
        print("  patches (< 0.1% of pixels per patch). This is consistent with")
        print("  specular reflection (sun glint, metal surfaces) or minor")
        print("  atmospheric correction artifacts. These are expected in")
        print("  Sentinel-2 L2A data and do not indicate data corruption.")
        print("")
        print("  Recommendation: no action required for training. The model")
        print("  should learn to handle the full dynamic range. Consider")
        print("  clipping to a reasonable ceiling (e.g. 10000 or the 99.9th")
        print("  percentile) during normalization if desired, but this is a")
        print("  design choice, not a data quality issue.")
    elif max_high_pct < 5.0:
        print("  [SUBSTANTIAL HIGH-VALUE REGIONS REQUIRING INVESTIGATION]")
        print("  Some patches have non-trivial fractions of very high values.")
        print("  Inspect the diagnostic figures to determine whether these")
        print("  correspond to real bright features or SCL misclassification.")
        print("")
        print("  Recommendation: review the diagnostic images in:")
        print(f"    {output_dir}/")
    else:
        print("  [SUBSTANTIAL HIGH-VALUE REGIONS REQUIRING INVESTIGATION]")
        print("  Multiple patches have large areas of very high values (>5%")
        print("  of pixels). This may indicate cloud/snow leakage past the")
        print("  SCL filter. Careful review of diagnostic images is needed.")
        print("")
        print("  Recommendation: review and consider tightening SCL filters")
        print("  or excluding affected patches.")

    print(f"\n  Diagnostic figures saved to: {output_dir}/")
    print(f"  Total diagnostic images: {len(affected_diagnostics)}")


if __name__ == "__main__":
    main()
