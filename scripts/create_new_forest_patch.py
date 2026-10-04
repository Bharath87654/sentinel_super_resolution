from pathlib import Path
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
safe = ROOT / "data/raw/forest/S2C_MSIL2A_20260924T050651_N0513_R019_T44QKD_20260924T100421.SAFE"
bands = ["B02", "B03", "B04", "B08"]
patch_size = 64

band_paths = {
    band: next(safe.rglob(f"*_{band}_10m.jp2"))
    for band in bands
}

with rasterio.open(band_paths["B02"]) as reference:
    row = (reference.height - patch_size) // 2
    col = (reference.width - patch_size) // 2
    window = Window(col, row, patch_size, patch_size)

    profile = reference.profile.copy()
    profile.update(
        driver="GTiff",
        width=patch_size,
        height=patch_size,
        count=4,
        dtype="uint16",
        compress="deflate"
    )

    output = ROOT / "data/processed/forest_sep2026_center_64x64.tif"

    with rasterio.open(output, "w", **profile) as dst:
        for index, band in enumerate(bands, start=1):
            with rasterio.open(band_paths[band]) as src:
                data = src.read(1, window=window)
                dst.write(data, index)
                dst.set_band_description(index, band)

print(f"Created safely: {output}")
