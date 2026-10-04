from pathlib import Path
import rasterio

safe = Path(
    "data/raw/forest/"
    "S2C_MSIL2A_20260924T050651_N0513_R019_T44QKD_20260924T100421.SAFE"
)

patterns = {
    "B02": "*B02_10m.jp2",
    "B03": "*B03_10m.jp2",
    "B04": "*B04_10m.jp2",
    "B08": "*B08_10m.jp2",
    "SCL": "*SCL_20m.jp2",
}

for band, pattern in patterns.items():
    matches = list(safe.rglob(pattern))

    if not matches:
        print(f"{band}: NOT FOUND")
        continue

    path = matches[0]
    with rasterio.open(path) as src:
        print(f"{band}: {src.width} x {src.height}, CRS: {src.crs}")
        print(f"  File: {path}")
