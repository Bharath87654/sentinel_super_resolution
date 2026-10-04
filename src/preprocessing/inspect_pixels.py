"""
inspect_pixels.py

Purpose:
    Read a SMALL window (not the whole image) from B02/B03/B04/B08 and SCL,
    and report what the actual pixel values look like: min, max, mean,
    median, percentiles. Also demonstrate the SCL resampling problem
    (20m categorical labels -> 10m grid) on that same small window.

This script still does NOT create patches, does NOT normalize permanently,
and does NOT touch your raw files. It only reads and prints.

Why a small window and not the whole 10980x10980 image?
    10980 x 10980 x 4 bands x 2 bytes (uint16) ~= 965 MB for ONE region,
    just for the raw arrays, before any processing overhead. Reading that
    for every region right now would be wasteful while we're still just
    trying to understand the data. A 1000x1000 window is enough to see
    real reflectance statistics.
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling


BANDS_10M = ["B02", "B03", "B04", "B08"]
WINDOW_SIZE = 1000  # pixels, at 10m resolution -> 10km x 10km sample


def find_band_file(safe_folder: Path, band_name: str, resolution: str) -> Path | None:
    pattern = f"GRANULE/*/IMG_DATA/R{resolution}/*_{band_name}_{resolution}.jp2"
    matches = list(safe_folder.glob(pattern))
    return matches[0] if matches else None


def find_valid_window(path: Path, window_size: int, step: int = 500):
    """
    Some Sentinel-2 tiles are only partially filled (a swath edge cuts
    diagonally through the tile, like your forest scene). Rather than
    assuming the center has real data, scan across the image at coarse
    resolution first, find a block that's mostly non-zero, then read
    that same location at full resolution.
    """
    with rasterio.open(path) as src:
        # Read a heavily downsampled full-tile overview just to locate
        # where the real data is, without loading the full-res array.
        overview = src.read(
            1,
            out_shape=(src.height // step, src.width // step),
            resampling=Resampling.average,
        )
        rows, cols = np.where(overview > 0)
        if rows.size == 0:
            raise RuntimeError(f"No valid data found anywhere in {path}")

        # Pick a point roughly in the middle of the valid-data cluster.
        r_full = int(np.median(rows)) * step
        c_full = int(np.median(cols)) * step

        half = window_size // 2
        col_off = max(min(c_full - half, src.width - window_size), 0)
        row_off = max(min(r_full - half, src.height - window_size), 0)

        window = rasterio.windows.Window(
            col_off=col_off, row_off=row_off,
            width=window_size, height=window_size,
        )
        return src.read(1, window=window), (col_off, row_off)


def report_band_stats(band_name: str, data: np.ndarray):
    valid = data[data > 0]  # 0 is typically no-data/fill for these bands
    if valid.size == 0:
        print(f"  {band_name}: window is entirely no-data (0). Try a different window.")
        return
    print(f"  {band_name}:")
    print(f"      shape        : {data.shape}")
    print(f"      dtype        : {data.dtype}")
    print(f"      min / max    : {valid.min()} / {valid.max()}")
    print(f"      mean / median: {valid.mean():.1f} / {np.median(valid):.1f}")
    print(f"      1st/99th pct : {np.percentile(valid, 1):.1f} / {np.percentile(valid, 99):.1f}")
    print(f"      zero pixels  : {(data == 0).sum()} / {data.size} "
          f"({100 * (data == 0).mean():.1f}% no-data in this window)")


def report_scl_stats(scl_data: np.ndarray):
    print(f"  SCL classes present in this window:")
    class_names = {
        0: "no data", 1: "defective", 2: "dark area", 3: "cloud shadow",
        4: "vegetation", 5: "bare soil", 6: "water", 7: "unclassified",
        8: "cloud medium prob", 9: "cloud high prob", 10: "cirrus", 11: "snow/ice",
    }
    unique, counts = np.unique(scl_data, return_counts=True)
    for val, count in zip(unique, counts):
        pct = 100 * count / scl_data.size
        name = class_names.get(val, "unknown")
        print(f"      class {val:2d} ({name:18s}): {count:8d} px ({pct:5.1f}%)")


def demonstrate_scl_resample(safe_folder: Path, window_size_10m: int, col_off: int, row_off: int):
    """
    Show the actual resolution mismatch problem and its fix:
    SCL is native 20m, our bands are native 10m, so SCL has HALF the
    width/height of the 10m bands over the same ground area.
    We fix this with nearest-neighbor upsampling because SCL values
    are categorical class IDs, not continuous measurements -- averaging
    or interpolating class IDs would invent nonsense classes (e.g.
    averaging "vegetation"=4 and "water"=6 does NOT mean class 5,
    "bare soil" -- that's a meaningless number here).
    """
    scl_path = find_band_file(safe_folder, "SCL", "20m")
    b02_path = find_band_file(safe_folder, "B02", "10m")

    with rasterio.open(b02_path) as b02_src, rasterio.open(scl_path) as scl_src:
        print(f"\n  B02 (10m native) full size : {b02_src.width} x {b02_src.height}")
        print(f"  SCL (20m native) full size : {scl_src.width} x {scl_src.height}")
        print(f"  -> SCL is exactly half the width/height of the 10m bands.")

        # Read the matching SCL window at 20m, then upsample it to 10m
        # using nearest-neighbor so it lines up pixel-for-pixel with B02.
        # Reuse the same ground location we already confirmed has real data,
        # just expressed in SCL's own (half-resolution) pixel coordinates.
        scl_window_size = window_size_10m // 2
        scl_window = rasterio.windows.Window(
            col_off=col_off // 2,
            row_off=row_off // 2,
            width=scl_window_size,
            height=scl_window_size,
        )

        # out_shape forces rasterio to resample WHILE reading.
        # Resampling.nearest = nearest-neighbor = correct choice for class labels.
        scl_upsampled = scl_src.read(
            1,
            window=scl_window,
            out_shape=(window_size_10m, window_size_10m),
            resampling=Resampling.nearest,
        )
        print(f"  SCL window read at native 20m, then upsampled to "
              f"{scl_upsampled.shape} using NEAREST NEIGHBOR.")
        print(f"  This now lines up 1:1 with a {window_size_10m}x{window_size_10m} "
              f"10m-band window over the same ground area.")
        return scl_upsampled


if __name__ == "__main__":
    RAW_DIR = Path("data/raw")
    FOREST_SAFE = list((RAW_DIR / "forest").glob("*.SAFE"))[0]

    print(f"\n{'='*60}")
    print(f"Inspecting real pixel values: forest region")
    print(f"Window: {WINDOW_SIZE}x{WINDOW_SIZE} pixels, centered in the tile")
    print(f"{'='*60}\n")

    # Find where the real data actually is, using B02 as the reference band.
    b02_path = find_band_file(FOREST_SAFE, "B02", "10m")
    _, (col_off, row_off) = find_valid_window(b02_path, WINDOW_SIZE)
    print(f"Located a valid-data window at col_off={col_off}, row_off={row_off}\n")

    for band_name in BANDS_10M:
        band_path = find_band_file(FOREST_SAFE, band_name, "10m")
        with rasterio.open(band_path) as src:
            window = rasterio.windows.Window(col_off, row_off, WINDOW_SIZE, WINDOW_SIZE)
            data = src.read(1, window=window)
        report_band_stats(band_name, data)

    print(f"\n{'-'*60}")
    print("SCL (Scene Classification Layer)")
    print(f"{'-'*60}")
    scl_path = find_band_file(FOREST_SAFE, "SCL", "20m")
    with rasterio.open(scl_path) as src:
        # SCL is 20m so the same ground area is half the pixel offsets/size
        scl_window = rasterio.windows.Window(
            col_off // 2, row_off // 2, WINDOW_SIZE // 2, WINDOW_SIZE // 2
        )
        scl_data_native = src.read(1, window=scl_window)
    report_scl_stats(scl_data_native)

    print(f"\n{'-'*60}")
    print("Resolution mismatch: resampling SCL from 20m to 10m")
    print(f"{'-'*60}")
    demonstrate_scl_resample(FOREST_SAFE, WINDOW_SIZE, col_off, row_off)