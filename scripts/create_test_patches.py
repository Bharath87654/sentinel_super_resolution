from pathlib import Path
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ["agriculture", "urban", "water"]
BANDS = ["B02", "B03", "B04", "B08"]
PATCH_SIZE = 64

out_dir = ROOT / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

for region in REGIONS:
    safe_dirs = list((ROOT / "data" / "raw" / region).glob("*.SAFE"))
    if not safe_dirs:
        raise FileNotFoundError(f"No SAFE folder found for {region}")

    safe_dir = safe_dirs[0]
    band_paths = {}

    for band in BANDS:
        matches = list(safe_dir.rglob(f"*_{band}_10m.jp2"))
        if not matches:
            raise FileNotFoundError(f"{band} 10m band not found in {safe_dir}")
        band_paths[band] = matches[0]

    with rasterio.open(band_paths["B02"]) as reference:
        if reference.width < PATCH_SIZE or reference.height < PATCH_SIZE:
            raise ValueError(f"{region}: image is smaller than {PATCH_SIZE}x{PATCH_SIZE}")

        row = (reference.height - PATCH_SIZE) // 2
        col = (reference.width - PATCH_SIZE) // 2
        window = Window(col, row, PATCH_SIZE, PATCH_SIZE)

        profile = reference.profile.copy()
        profile.update(
            driver="GTiff",
            width=PATCH_SIZE,
            height=PATCH_SIZE,
            count=4,
            dtype="uint16",
            compress="deflate"
        )

        output_path = out_dir / f"{region}_center_64x64.tif"

        with rasterio.open(output_path, "w", **profile) as dst:
            for index, band in enumerate(BANDS, start=1):
                with rasterio.open(band_paths[band]) as src:
                    if (src.width, src.height) != (reference.width, reference.height):
                        raise ValueError(f"{region}: {band} dimensions do not match B02")
                    if src.crs != reference.crs:
                        raise ValueError(f"{region}: {band} CRS does not match B02")

                    data = src.read(1, window=window)
                    dst.write(data, index)
                    dst.set_band_description(index, band)

    print(f"Created: {output_path}")
