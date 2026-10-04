"""
inspect_safe.py

Purpose (and nothing more than this):
    Given a Sentinel-2 .SAFE folder, find B02, B03, B04, B08 and SCL,
    print their paths, and confirm each file's resolution/dimensions.

This script does NOT read pixel data into memory in bulk, does NOT modify
anything, and does NOT create patches. It only locates and validates files.
That's the whole point of this stage: prove Python can navigate the SAFE
structure correctly before we do anything that touches the actual pixels.
"""

from pathlib import Path
import rasterio


# The bands we need, and which resolution folder each one lives in.
# B02/B03/B04/B08 are native 10m. SCL (Scene Classification) is native 20m.
BANDS_WE_NEED = {
    "B02": "10m",
    "B03": "10m",
    "B04": "10m",
    "B08": "10m",
    "SCL": "20m",
}


def find_band_file(safe_folder: Path, band_name: str, resolution: str) -> Path | None:
    """
    Search inside SAFE/GRANULE/*/IMG_DATA/R{resolution}/ for a file matching
    the band name. Returns the path, or None if not found.

    We search with a glob pattern rather than hardcoding the full filename,
    because the filename includes the tile ID and acquisition timestamp,
    which differ for every product you download.
    """
    pattern = f"GRANULE/*/IMG_DATA/R{resolution}/*_{band_name}_{resolution}.jp2"
    matches = list(safe_folder.glob(pattern))
    if not matches:
        return None
    return matches[0]


def inspect_one_safe(safe_folder: Path, label: str = ""):
    """Run the full inspection on one .SAFE folder and print a report."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {label or safe_folder.name}")
    print(f"{'='*60}")

    if not safe_folder.exists():
        print(f"  ERROR: folder does not exist: {safe_folder}")
        return

    found_paths = {}

    for band_name, resolution in BANDS_WE_NEED.items():
        path = find_band_file(safe_folder, band_name, resolution)
        if path is None:
            print(f"  {band_name}: NOT FOUND (expected {resolution} resolution)")
            continue

        found_paths[band_name] = path

        # Open it just to confirm rasterio can read it and report real dimensions.
        # This does NOT load the full pixel array — rasterio.open() reads
        # only the header/metadata until you explicitly call .read().
        with rasterio.open(path) as src:
            print(f"  {band_name}: found")
            print(f"      path       : {path}")
            print(f"      resolution : {resolution}")
            print(f"      dimensions : {src.width} x {src.height}")
            print(f"      dtype      : {src.dtypes[0]}")
            print(f"      crs        : {src.crs}")

    missing = set(BANDS_WE_NEED) - set(found_paths)
    print()
    if missing:
        print(f"  RESULT: INCOMPLETE — missing {sorted(missing)}")
    else:
        print(f"  RESULT: OK — all 5 required files found and readable.")

    return found_paths


if __name__ == "__main__":
    # --------------------------------------------------------------
    # EDIT THIS SECTION for your actual project layout.
    # In your real project this will be data/raw/forest/<SAFE folder>, etc.
    # For now, point RAW_DIR at wherever your data/raw folder lives.
    # --------------------------------------------------------------
    RAW_DIR = Path("data/raw")

    regions = {
        "forest": RAW_DIR / "forest",
        "urban": RAW_DIR / "urban",
        "agriculture": RAW_DIR / "agriculture",
        "water": RAW_DIR / "water",
    }

    for region_name, region_folder in regions.items():
        if not region_folder.exists():
            print(f"\n[{region_name}] folder not found yet: {region_folder} (skipping)")
            continue
        # Each region folder should contain exactly one .SAFE folder inside it.
        safe_candidates = list(region_folder.glob("*.SAFE"))
        if not safe_candidates:
            print(f"\n[{region_name}] no .SAFE folder found inside {region_folder}")
            continue
        inspect_one_safe(safe_candidates[0], label=region_name)