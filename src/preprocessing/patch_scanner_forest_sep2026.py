"""
patch_scanner.py  (prototype — inspection only, does NOT save full patch arrays yet)

Purpose:
    Scan an entire Sentinel-2 region tile in non-overlapping 256x256 windows,
    compute the real SCL land-cover composition for EVERY candidate patch,
    accept/reject based on a configurable "bad pixel" threshold, and write
    the results to a metadata CSV.

    This answers the real question raised by the single-window samples:
    "how much of this tile is actually usable, and what does it actually
    contain?" -- across the WHOLE tile, not one arbitrary window.

Design decisions (adjust the CONFIG dict below, nothing else needs changing):
    - patch size 256x256, non-overlapping (stride = patch size)
    - a patch is REJECTED if (no-data + defective + cloud shadow +
      cloud medium + cloud high + cirrus + snow/ice) > max_bad_pct
    - memory: only ONE 256x256 window is ever held in memory per band;
      we never load the full 10980x10980 array
    - we do NOT save full patch pixel arrays yet in this prototype --
      only a handful of ACCEPTED patches get saved as quick-look PNGs
      so you can visually sanity-check what "accepted" actually looks like
"""

from pathlib import Path
import csv
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


CONFIG = {
    "patch_size": 256,
    "max_bad_pct": 10.0,       # reject patch if bad-class % exceeds this
    "sample_previews_per_region": 3,  # how many ACCEPTED patches to save as PNG
}

BAD_CLASSES = {0, 1, 3, 8, 9, 10, 11}  # no-data, defective, cloud shadow, cloud x2, cirrus, snow/ice
SCL_CLASS_NAMES = {
    0: "nodata", 1: "defective", 2: "dark_area", 3: "cloud_shadow",
    4: "vegetation", 5: "bare_soil", 6: "water", 7: "unclassified",
    8: "cloud_med", 9: "cloud_high", 10: "cirrus", 11: "snow_ice",
}


def find_band_file(safe_folder: Path, band_name: str, resolution: str) -> Path | None:
    pattern = f"GRANULE/*/IMG_DATA/R{resolution}/*_{band_name}_{resolution}.jp2"
    matches = list(safe_folder.glob(pattern))
    return matches[0] if matches else None


def read_tile_metadata(safe_folder: Path) -> dict:
    """Pull tile ID, acquisition date, CRS from MTD_TL.xml -- so every
    patch's metadata row can trace back to its exact source product."""
    mtd_files = list(safe_folder.glob("GRANULE/*/MTD_TL.xml"))
    crs_code = None
    if mtd_files:
        tree = ET.parse(mtd_files[0])
        for elem in tree.getroot().iter():
            if elem.tag.split("}")[-1] == "HORIZONTAL_CS_CODE":
                crs_code = elem.text
                break
    return {
        "safe_product": safe_folder.name,
        "crs": crs_code,
    }


def scl_composition(scl_patch: np.ndarray) -> dict:
    total = scl_patch.size
    comp = {}
    for cls_id, name in SCL_CLASS_NAMES.items():
        comp[f"{name}_pct"] = round(100.0 * np.sum(scl_patch == cls_id) / total, 2)
    bad_pct = sum(comp[f"{SCL_CLASS_NAMES[c]}_pct"] for c in BAD_CLASSES)
    comp["bad_pct"] = round(bad_pct, 2)
    comp["valid_pct"] = round(100.0 - bad_pct, 2)
    return comp


def stretch_to_uint8(band: np.ndarray) -> np.ndarray:
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    return (np.clip((band.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def scan_region(safe_folder: Path, region_name: str, output_dir: Path,
                 config: dict) -> list[dict]:
    patch_size = config["patch_size"]
    tile_meta = read_tile_metadata(safe_folder)

    b02_path = find_band_file(safe_folder, "B02", "10m")
    b03_path = find_band_file(safe_folder, "B03", "10m")
    b04_path = find_band_file(safe_folder, "B04", "10m")
    b08_path = find_band_file(safe_folder, "B08", "10m")
    scl_path = find_band_file(safe_folder, "SCL", "20m")

    rows_out = []
    preview_count = 0
    preview_dir = output_dir / "preview_patches"
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Keep file handles open for the whole scan -- opening/closing per patch
    # would be far slower than one open + many windowed reads.
    with rasterio.open(b02_path) as b02_src, \
         rasterio.open(b03_path) as b03_src, \
         rasterio.open(b04_path) as b04_src, \
         rasterio.open(b08_path) as b08_src, \
         rasterio.open(scl_path) as scl_src:

        width, height = b02_src.width, b02_src.height
        n_cols = width // patch_size
        n_rows = height // patch_size
        print(f"  Tile is {width}x{height} -> scanning {n_rows}x{n_cols} = "
              f"{n_rows*n_cols} candidate {patch_size}x{patch_size} patches")

        accepted, rejected = 0, 0

        for r in range(n_rows):
            for c in range(n_cols):
                row_off, col_off = r * patch_size, c * patch_size

                # Read only THIS patch's B02 first -- cheapest possible
                # check to skip empty/no-data regions before reading
                # the other 3 bands and doing the SCL resample.
                b02_patch = b02_src.read(1, window=Window(col_off, row_off, patch_size, patch_size))
                if np.mean(b02_patch > 0) < 0.5:
                    rejected += 1
                    continue  # mostly no-data, skip immediately, don't bother with SCL

                scl_patch = scl_src.read(
                    1,
                    window=Window(col_off // 2, row_off // 2, patch_size // 2, patch_size // 2),
                    out_shape=(patch_size, patch_size),
                    resampling=Resampling.nearest,
                )
                comp = scl_composition(scl_patch)

                accept = comp["bad_pct"] <= config["max_bad_pct"]

                row = {
                    "region": region_name,
                    "safe_product": tile_meta["safe_product"],
                    "crs": tile_meta["crs"],
                    "row_off": row_off,
                    "col_off": col_off,
                    "patch_size": patch_size,
                    "accepted": accept,
                    **comp,
                }
                rows_out.append(row)

                if accept:
                    accepted += 1
                    if preview_count < config["sample_previews_per_region"]:
                        # Only for a HANDFUL of accepted patches, read the
                        # other 3 bands and save a quick-look PNG.
                        b03_patch = b03_src.read(1, window=Window(col_off, row_off, patch_size, patch_size))
                        b04_patch = b04_src.read(1, window=Window(col_off, row_off, patch_size, patch_size))
                        rgb = np.stack([stretch_to_uint8(b04_patch),
                                         stretch_to_uint8(b03_patch),
                                         stretch_to_uint8(b02_patch)], axis=-1)
                        plt.figure(figsize=(4, 4))
                        plt.imshow(rgb)
                        plt.axis("off")
                        plt.title(f"{region_name} r{row_off}_c{col_off}\n"
                                  f"valid={comp['valid_pct']}%", fontsize=8)
                        plt.tight_layout()
                        plt.savefig(preview_dir / f"{region_name}_r{row_off}_c{col_off}.png",
                                    dpi=100, bbox_inches="tight")
                        plt.close()
                        preview_count += 1
                else:
                    rejected += 1

        print(f"  Accepted: {accepted}  Rejected: {rejected}  "
              f"({100*accepted/(accepted+rejected):.1f}% acceptance rate)")

    return rows_out


def write_metadata_csv(all_rows: list[dict], output_path: Path):
    if not all_rows:
        print("No rows to write.")
        return
    fieldnames = list(all_rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} patch records to {output_path}")


if __name__ == "__main__":
    SAFE_FOLDER = (
        Path("data/raw/forest")
        / "S2C_MSIL2A_20260924T050651_N0513_R019_T44QKD_20260924T100421.SAFE"
    )

    OUTPUT_DIR = Path("outputs/forest_sep2026_scan")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SAFE_FOLDER.exists():
        raise FileNotFoundError(f"SAFE folder not found: {SAFE_FOLDER}")

    print("\n" + "=" * 60)
    print("FOREST — SEPTEMBER 2026")
    print("=" * 60)

    rows = scan_region(SAFE_FOLDER, "forest_sep2026", OUTPUT_DIR, CONFIG)

    write_metadata_csv(rows, OUTPUT_DIR / "patches_metadata.csv")

    if rows:
        accepted = sum(1 for row in rows if row["accepted"])
        total = len(rows)
        rejected = total - accepted
        pct = 100 * accepted / total if total else 0

        print("\n" + "=" * 60)
        print("SEPTEMBER 2026 FOREST SUMMARY")
        print("=" * 60)
        print(f"Total recorded patches: {total}")
        print(f"Accepted: {accepted}")
        print(f"Rejected: {rejected}")
        print(f"Acceptance rate: {pct:.1f}%")