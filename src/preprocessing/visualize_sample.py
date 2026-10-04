"""
visualize_sample.py

Purpose:
    Read a manageable window from a Sentinel-2 SAFE product and produce
    three views side by side:
        1. True Color   (B04=Red, B03=Green, B02=Blue)   - what your eye expects
        2. False Color  (B08=Red, B04=Green, B03=Blue)   - vegetation highlighted
        3. SCL map      - categorical land-cover/cloud classification, colored

    Also prints the land-cover composition (% vegetation, % water, % cloud,
    etc.) for the window, because "which folder it came from" is NOT a
    reliable label -- as you just found with your forest window showing
    94.9% bare soil.

Still does not create patches or modify raw data. This is the last
inspection step before we start Phase 5 (real preprocessing).
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


BANDS_10M = ["B02", "B03", "B04", "B08"]
WINDOW_SIZE = 1000

SCL_CLASS_NAMES = {
    0: "no data", 1: "defective", 2: "dark area", 3: "cloud shadow",
    4: "vegetation", 5: "bare soil", 6: "water", 7: "unclassified",
    8: "cloud (med prob)", 9: "cloud (high prob)", 10: "cirrus", 11: "snow/ice",
}
# Fixed colors per class so the SCL map is always readable the same way
# across every region you visualize (forest/urban/agriculture/water).
SCL_COLORS = {
    0: "#000000", 1: "#ff00ff", 2: "#404040", 3: "#8b4513",
    4: "#00a000", 5: "#d2b48c", 6: "#0000ff", 7: "#a0a0a0",
    8: "#c0c0c0", 9: "#ffffff", 10: "#00ffff", 11: "#ff69b4",
}


def find_band_file(safe_folder: Path, band_name: str, resolution: str) -> Path | None:
    pattern = f"GRANULE/*/IMG_DATA/R{resolution}/*_{band_name}_{resolution}.jp2"
    matches = list(safe_folder.glob(pattern))
    return matches[0] if matches else None


def find_valid_window(path: Path, window_size: int, step: int = 500):
    """Same logic as inspect_pixels.py -- locate real data before reading
    at full resolution, since tiles can be only partially filled."""
    with rasterio.open(path) as src:
        overview = src.read(
            1, out_shape=(src.height // step, src.width // step),
            resampling=Resampling.average,
        )
        rows, cols = np.where(overview > 0)
        if rows.size == 0:
            raise RuntimeError(f"No valid data found anywhere in {path}")
        r_full = int(np.median(rows)) * step
        c_full = int(np.median(cols)) * step
        half = window_size // 2
        col_off = max(min(c_full - half, src.width - window_size), 0)
        row_off = max(min(r_full - half, src.height - window_size), 0)
        return col_off, row_off


def read_band_window(safe_folder: Path, band_name: str, col_off: int, row_off: int,
                      window_size: int) -> np.ndarray:
    path = find_band_file(safe_folder, band_name, "10m")
    with rasterio.open(path) as src:
        window = rasterio.windows.Window(col_off, row_off, window_size, window_size)
        return src.read(1, window=window)


def read_scl_window(safe_folder: Path, col_off: int, row_off: int,
                     window_size: int) -> np.ndarray:
    """Read SCL at native 20m for the matching ground area, then upsample
    to 10m with nearest-neighbor so it aligns with the band windows."""
    path = find_band_file(safe_folder, "SCL", "20m")
    with rasterio.open(path) as src:
        window = rasterio.windows.Window(col_off // 2, row_off // 2,
                                          window_size // 2, window_size // 2)
        return src.read(1, window=window, out_shape=(window_size, window_size),
                         resampling=Resampling.nearest)


def stretch_to_uint8(band: np.ndarray, low_pct=2, high_pct=98) -> np.ndarray:
    """Sentinel-2 reflectance values (uint16, roughly 0-6000+) need contrast
    stretching to display sensibly as an 8-bit image. We clip to the
    2nd-98th percentile of THIS window (not a fixed global constant) so
    each region's visualization uses contrast appropriate to its own values,
    then rescale to 0-255."""
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)
    lo, hi = np.percentile(valid, [low_pct, high_pct])
    stretched = np.clip((band.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    return (stretched * 255).astype(np.uint8)


def build_scl_colormap():
    classes = sorted(SCL_COLORS.keys())
    colors = [SCL_COLORS[c] for c in classes]
    cmap = ListedColormap(colors)
    bounds = classes + [classes[-1] + 1]
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, classes


def report_composition(scl_window: np.ndarray, label: str):
    print(f"\n  Land-cover composition ({label}):")
    unique, counts = np.unique(scl_window, return_counts=True)
    total = scl_window.size
    for val, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        name = SCL_CLASS_NAMES.get(val, "unknown")
        print(f"      class {val:2d} ({name:18s}): {100*count/total:5.1f}%")


def visualize_region(safe_folder: Path, region_label: str, output_path: Path,
                      window_size: int = WINDOW_SIZE):
    b02_path = find_band_file(safe_folder, "B02", "10m")
    col_off, row_off = find_valid_window(b02_path, window_size)

    bands = {b: read_band_window(safe_folder, b, col_off, row_off, window_size)
             for b in BANDS_10M}
    scl = read_scl_window(safe_folder, col_off, row_off, window_size)

    report_composition(scl, region_label)

    true_color = np.stack([
        stretch_to_uint8(bands["B04"]),
        stretch_to_uint8(bands["B03"]),
        stretch_to_uint8(bands["B02"]),
    ], axis=-1)

    false_color = np.stack([
        stretch_to_uint8(bands["B08"]),
        stretch_to_uint8(bands["B04"]),
        stretch_to_uint8(bands["B03"]),
    ], axis=-1)

    cmap, norm, classes = build_scl_colormap()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(true_color)
    axes[0].set_title(f"{region_label}: True Color (B04/B03/B02)")
    axes[1].imshow(false_color)
    axes[1].set_title(f"{region_label}: False Color (B08/B04/B03)\nvegetation = bright red")
    im = axes[2].imshow(scl, cmap=cmap, norm=norm)
    axes[2].set_title(f"{region_label}: SCL classification")
    cbar = plt.colorbar(im, ax=axes[2], ticks=[c + 0.5 for c in classes], fraction=0.046)
    cbar.ax.set_yticklabels([SCL_CLASS_NAMES[c] for c in classes], fontsize=7)

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved visualization to {output_path}")


if __name__ == "__main__":
    RAW_DIR = Path("data/raw")
    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)

    regions = {
        "forest": RAW_DIR / "forest",
        "urban": RAW_DIR / "urban",
        "agriculture": RAW_DIR / "agriculture",
        "water": RAW_DIR / "water",
    }

    for region_name, region_folder in regions.items():
        safe_candidates = list(region_folder.glob("*.SAFE")) if region_folder.exists() else []
        if not safe_candidates:
            print(f"\n[{region_name}] no .SAFE folder found yet, skipping")
            continue
        print(f"\n{'='*60}\n{region_name.upper()}\n{'='*60}")
        visualize_region(safe_candidates[0], region_name,
                          OUTPUT_DIR / f"{region_name}_sample.png")