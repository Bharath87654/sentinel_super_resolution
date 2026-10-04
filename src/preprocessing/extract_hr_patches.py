"""
extract_hr_patches.py

Purpose:
    Physically extract the final eligible patches (accepted AND passing
    the unclassified filter) as quantitative 4-band arrays, saved as .npy.

    IMPORTANT: this does NOT normalize or rescale pixel values. The
    2-98 percentile stretch used in preview PNGs is a VISUALIZATION-ONLY
    trick -- it would destroy quantitative meaning if baked into the
    actual training data. These .npy files keep the original Sentinel-2
    uint16 reflectance values untouched. Normalization is a separate,
    deliberate decision (Step 3), applied at load-time in the PyTorch
    Dataset later -- not baked into the stored files.

    Still does NOT generate LR images and does NOT touch data/raw/.
"""

from pathlib import Path
import csv
import numpy as np
import rasterio

EXTRACT_CONFIG = {
    "max_bad_pct": 10.0,
    "max_unclassified_pct": 5.0,
}
# Fixed channel order for every saved patch -- must stay consistent
# across the whole project, since the model will assume this order.
CHANNEL_ORDER = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR


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


def get_final_eligible(rows: list[dict], config: dict) -> list[dict]:
    """Same two-filter logic as validate_patch_selection.py, recomputed
    here directly from the metadata CSV so this script has no hidden
    dependency on an intermediate file."""
    return [r for r in rows if r["accepted"]
            and r["unclassified_pct"] <= config["max_unclassified_pct"]]


def extract_region(safe_folder: Path, region: str, eligible_patches: list[dict],
                    output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    band_paths = {b: find_band_file(safe_folder, b, "10m") for b in CHANNEL_ORDER}

    index_rows = []
    # Keep all 4 band files open for the whole region -- same reasoning
    # as patch_scanner.py: avoid repeated open/close overhead, and never
    # load more than one 256x256 window per band into memory at a time.
    handles = {b: rasterio.open(p) for b, p in band_paths.items()}
    try:
        for patch in eligible_patches:
            row_off, col_off, size = int(patch["row_off"]), int(patch["col_off"]), int(patch["patch_size"])
            window = rasterio.windows.Window(col_off, row_off, size, size)

            # Stack in FIXED channel order: [B02, B03, B04, B08] = CHW format
            arr = np.stack([handles[b].read(1, window=window) for b in CHANNEL_ORDER], axis=0)
            # dtype stays uint16 -- the native Sentinel-2 storage type.
            # NOT cast to float, NOT rescaled, NOT stretched.
            assert arr.dtype == np.uint16, f"unexpected dtype {arr.dtype}"

            filename = f"{region}_r{row_off}_c{col_off}.npy"
            out_path = output_dir / filename
            np.save(out_path, arr)

            index_rows.append({
                "filename": filename,
                "region": region,
                "safe_product": patch["safe_product"],
                "crs": patch["crs"],
                "row_off": row_off,
                "col_off": col_off,
                "patch_size": size,
                "channel_order": ",".join(CHANNEL_ORDER),
                "dtype": str(arr.dtype),
                "shape": f"{arr.shape}",
                "vegetation_pct": patch["vegetation_pct"],
                "bare_soil_pct": patch["bare_soil_pct"],
                "water_pct": patch["water_pct"],
                "unclassified_pct": patch["unclassified_pct"],
                "bad_pct": patch["bad_pct"],
            })
    finally:
        for h in handles.values():
            h.close()

    return index_rows


def write_index_csv(all_rows: list[dict], output_path: Path):
    if not all_rows:
        print("No rows to write.")
        return
    fieldnames = list(all_rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} HR patch index records to {output_path}")


if __name__ == "__main__":
    metadata_csv = Path("outputs/patches_metadata.csv")
    rows = load_metadata(metadata_csv)

    by_region = {}
    for r in rows:
        by_region.setdefault(r["region"], []).append(r)

    output_root = Path("data/processed/hr")
    all_index_rows = []

    for region, region_rows in by_region.items():
        eligible = get_final_eligible(region_rows, EXTRACT_CONFIG)
        print(f"\n{region}: {len(eligible)} eligible patches to extract")

        region_safe_dir = Path("data/raw") / region
        safe_candidates = list(region_safe_dir.glob("*.SAFE")) if region_safe_dir.exists() else []
        if not safe_candidates:
            print(f"  [{region}] raw SAFE folder not found, skipping extraction")
            continue

        index_rows = extract_region(safe_candidates[0], region, eligible, output_root / region)
        all_index_rows.extend(index_rows)
        print(f"  Extracted {len(index_rows)} .npy files to {output_root / region}/")

    write_index_csv(all_index_rows, Path("data/processed/hr_patches_index.csv"))