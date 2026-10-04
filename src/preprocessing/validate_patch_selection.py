"""
validate_patch_selection.py

Purpose:
    A PRE-EXTRACTION validation stage. Does not regenerate metadata, does
    not touch raw SAFE data, does not create .npy patches or LR images.

    1. Adds a second, configurable filter (max_unclassified_pct) on top
       of the existing accepted/rejected column, WITHOUT overwriting the
       original CSV -- this answers "how many patches survive if we also
       reject high-uncertainty patches?"
    2. Pulls REAL RGB imagery for a spatially-distributed random sample
       of eligible patches per region, so bare-soil-dominant SCL results
       (like your forest scene) can be visually checked rather than
       trusted on numbers alone. SCL class 5 is never assumed to mean
       "urban" or "not forest" -- we only use it to decide what to look
       at, never to conclude what a scene actually is.
"""

from pathlib import Path
import csv
import random
import numpy as np
import rasterio
import matplotlib.pyplot as plt

VALIDATION_CONFIG = {
    "max_bad_pct": 10.0,          # unchanged, matches scanner
    "max_unclassified_pct": 5.0,  # NEW second filter
    "previews_per_region": 10,
    "spatial_bins_per_side": 3,   # 3x3 meta-grid for spatial spread
}


def find_band_file(safe_folder: Path, band_name: str, resolution: str) -> Path | None:
    pattern = f"GRANULE/*/IMG_DATA/R{resolution}/*_{band_name}_{resolution}.jp2"
    matches = list(safe_folder.glob(pattern))
    return matches[0] if matches else None


def load_metadata(csv_path: Path) -> list[dict]:
    numeric_cols = ["row_off", "col_off", "patch_size", "bad_pct", "valid_pct",
                     "nodata_pct", "defective_pct", "dark_area_pct", "cloud_shadow_pct",
                     "vegetation_pct", "bare_soil_pct", "water_pct", "unclassified_pct",
                     "cloud_med_pct", "cloud_high_pct", "cirrus_pct", "snow_ice_pct"]
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in numeric_cols:
                if col in row and row[col] != "":
                    row[col] = float(row[col])
            row["accepted"] = row["accepted"] in ("True", "true", "1")
            rows.append(row)
    return rows


def unclassified_bin(pct: float) -> str:
    if pct <= 1.0:
        return "0-1%"
    elif pct <= 5.0:
        return "1-5%"
    elif pct <= 10.0:
        return "5-10%"
    elif pct <= 25.0:
        return "10-25%"
    return ">25%"


def apply_validation_filter(rows: list[dict], config: dict) -> dict:
    """
    Returns per-region breakdown of the funnel:
      total_candidates_in_csv -> currently_accepted (existing bad_pct filter)
        -> final_eligible (also passes max_unclassified_pct)
    """
    by_region = {}
    for row in rows:
        by_region.setdefault(row["region"], []).append(row)

    results = {}
    for region, region_rows in by_region.items():
        currently_accepted = [r for r in region_rows if r["accepted"]]
        rejected_by_bad_pct = [r for r in region_rows if not r["accepted"]]

        final_eligible = [r for r in currently_accepted
                           if r["unclassified_pct"] <= config["max_unclassified_pct"]]
        rejected_by_unclassified = [r for r in currently_accepted
                                     if r["unclassified_pct"] > config["max_unclassified_pct"]]

        results[region] = {
            "total_metadata_rows": len(region_rows),
            "currently_accepted": len(currently_accepted),
            "rejected_by_bad_pct": len(rejected_by_bad_pct),
            "rejected_by_unclassified": len(rejected_by_unclassified),
            "final_eligible": final_eligible,
            "final_eligible_count": len(final_eligible),
        }
    return results


def report_funnel(region: str, stats: dict, config: dict):
    print(f"\n{'='*70}\n{region.upper()}\n{'='*70}")
    print(f"  Total metadata rows              : {stats['total_metadata_rows']}")
    print(f"  Currently accepted (bad_pct<={config['max_bad_pct']}%) : {stats['currently_accepted']}")
    print(f"  Rejected by max_bad_pct           : {stats['rejected_by_bad_pct']}")
    print(f"  Rejected by max_unclassified_pct  : {stats['rejected_by_unclassified']} "
          f"(new filter, unclassified>{config['max_unclassified_pct']}%)")
    print(f"  FINAL ELIGIBLE                    : {stats['final_eligible_count']}")
    if stats["currently_accepted"] > 0:
        retained_pct = 100 * stats["final_eligible_count"] / stats["currently_accepted"]
        print(f"  Retained vs currently-accepted    : {retained_pct:.1f}%")


def report_unclassified_distribution(region: str, currently_accepted: list[dict]):
    print(f"\n  Unclassified-pixel distribution among currently-accepted patches:")
    bins = {"0-1%": 0, "1-5%": 0, "5-10%": 0, "10-25%": 0, ">25%": 0}
    for r in currently_accepted:
        bins[unclassified_bin(r["unclassified_pct"])] += 1
    n = len(currently_accepted)
    for label, count in bins.items():
        pct = 100 * count / n if n else 0
        print(f"      {label:7s}: {count:4d} ({pct:.1f}%)")


def select_spatial_sample(eligible: list[dict], n_target: int, bins_per_side: int) -> list[dict]:
    """
    Stratified random sample: divide the row_off/col_off space into a
    bins_per_side x bins_per_side meta-grid, then round-robin randomly
    pick from each populated bin until n_target patches are selected.
    This avoids "first N accepted patches", which would just show one
    corner of the tile.
    """
    if not eligible:
        return []

    row_offs = [r["row_off"] for r in eligible]
    col_offs = [r["col_off"] for r in eligible]
    r_min, r_max = min(row_offs), max(row_offs)
    c_min, c_max = min(col_offs), max(col_offs)
    r_span = max(r_max - r_min, 1)
    c_span = max(c_max - c_min, 1)

    bins = {}
    for r in eligible:
        br = min(int(bins_per_side * (r["row_off"] - r_min) / r_span), bins_per_side - 1)
        bc = min(int(bins_per_side * (r["col_off"] - c_min) / c_span), bins_per_side - 1)
        bins.setdefault((br, bc), []).append(r)

    for bin_list in bins.values():
        random.shuffle(bin_list)

    selected = []
    bin_keys = list(bins.keys())
    idx = 0
    while len(selected) < n_target and any(bins[k] for k in bin_keys):
        key = bin_keys[idx % len(bin_keys)]
        if bins[key]:
            selected.append(bins[key].pop())
        idx += 1
    return selected


def stretch_to_uint8(band: np.ndarray) -> np.ndarray:
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    return (np.clip((band.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def save_preview(safe_folder: Path, patch: dict, output_path: Path):
    row_off, col_off, size = int(patch["row_off"]), int(patch["col_off"]), int(patch["patch_size"])
    b02_path = find_band_file(safe_folder, "B02", "10m")
    b03_path = find_band_file(safe_folder, "B03", "10m")
    b04_path = find_band_file(safe_folder, "B04", "10m")

    with rasterio.open(b02_path) as s2, rasterio.open(b03_path) as s3, rasterio.open(b04_path) as s4:
        window = rasterio.windows.Window(col_off, row_off, size, size)
        b02 = s2.read(1, window=window)
        b03 = s3.read(1, window=window)
        b04 = s4.read(1, window=window)

    rgb = np.stack([stretch_to_uint8(b04), stretch_to_uint8(b03), stretch_to_uint8(b02)], axis=-1)

    plt.figure(figsize=(5, 5))
    plt.imshow(rgb)
    plt.axis("off")
    title = (f"{patch['region']}  r{row_off}_c{col_off}\n"
             f"veg={patch['vegetation_pct']:.1f}% soil={patch['bare_soil_pct']:.1f}% "
             f"water={patch['water_pct']:.1f}%\n"
             f"unclass={patch['unclassified_pct']:.1f}% bad={patch['bad_pct']:.1f}%")
    plt.title(title, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    random.seed(42)  # reproducible sample selection

    csv_path = Path("outputs/patches_metadata.csv")
    rows = load_metadata(csv_path)
    funnel = apply_validation_filter(rows, VALIDATION_CONFIG)

    by_region_accepted = {}
    for row in rows:
        if row["accepted"]:
            by_region_accepted.setdefault(row["region"], []).append(row)

    preview_dir_root = Path("outputs/validation_previews")
    preview_dir_root.mkdir(parents=True, exist_ok=True)
    all_selected_rows = []

    for region, stats in funnel.items():
        report_funnel(region, stats, VALIDATION_CONFIG)
        report_unclassified_distribution(region, by_region_accepted.get(region, []))

        selected = select_spatial_sample(
            stats["final_eligible"], VALIDATION_CONFIG["previews_per_region"],
            VALIDATION_CONFIG["spatial_bins_per_side"],
        )
        print(f"\n  Selected {len(selected)} spatially-distributed preview patches.")

        region_safe_dir = Path("data/raw") / region
        safe_candidates = list(region_safe_dir.glob("*.SAFE")) if region_safe_dir.exists() else []
        if not safe_candidates:
            print(f"  [{region}] raw SAFE folder not found -- skipping preview image generation "
                  f"(metadata funnel stats above are still valid).")
            continue

        safe_folder = safe_candidates[0]
        region_preview_dir = preview_dir_root / region
        region_preview_dir.mkdir(parents=True, exist_ok=True)

        for patch in selected:
            out_name = f"{region}_r{int(patch['row_off'])}_c{int(patch['col_off'])}.png"
            save_preview(safe_folder, patch, region_preview_dir / out_name)
            all_selected_rows.append(patch)

        print(f"  Saved previews to {region_preview_dir}/")

    if all_selected_rows:
        out_csv = preview_dir_root / "selected_previews.csv"
        fieldnames = list(all_selected_rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_selected_rows)
        print(f"\nWrote {len(all_selected_rows)} selected-preview records to {out_csv}")