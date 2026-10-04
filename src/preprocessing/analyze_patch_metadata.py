"""
analyze_patch_metadata.py

Purpose:
    Read outputs/patches_metadata.csv (produced by patch_scanner.py) and
    generate a full dataset-quality report, WITHOUT touching any raw data
    or generating any training patches.

Important ground rules followed here:
    - SCL class 5 (bare soil) is NEVER interpreted as "urban" -- we report
      the actual SCL percentages only, regardless of which region folder
      a patch came from.
    - Land cover is read from the metadata columns, never inferred from
      the region name.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

# Standard Sentinel-2 L2A 10m tile dimensions. Used only to explain the
# "tile not an exact multiple of patch_size" edge-loss, and to draw the
# full candidate grid (including cells that never became metadata rows
# because they failed the early no-data precheck in the scanner).
TILE_DIM = 10980

CLASS_COLUMNS = [
    "nodata_pct", "defective_pct", "dark_area_pct", "cloud_shadow_pct",
    "vegetation_pct", "bare_soil_pct", "water_pct", "unclassified_pct",
    "cloud_med_pct", "cloud_high_pct", "cirrus_pct", "snow_ice_pct",
]
CLOUD_RELATED_COLUMNS = ["cloud_shadow_pct", "cloud_med_pct", "cloud_high_pct",
                          "cirrus_pct", "snow_ice_pct"]


def load_metadata(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in CLASS_COLUMNS + ["bad_pct", "valid_pct", "row_off", "col_off", "patch_size"]:
                if col in row and row[col] != "":
                    row[col] = float(row[col])
            row["accepted"] = row["accepted"] in ("True", "true", "1")
            rows.append(row)
    return rows


def bad_pct_bin(bad_pct: float) -> str:
    if bad_pct <= 1.0:
        return "0-1%"
    elif bad_pct <= 2.0:
        return "1-2%"
    elif bad_pct <= 5.0:
        return "2-5%"
    elif bad_pct <= 10.0:
        return "5-10%"
    return ">10%"


def report_region(region: str, rows: list[dict], patch_size: int):
    accepted_rows = [r for r in rows if r["accepted"]]
    rejected_rows = [r for r in rows if not r["accepted"]]

    n_side = TILE_DIM // patch_size
    total_grid_cells = n_side * n_side
    edge_loss_px = TILE_DIM - n_side * patch_size

    print(f"\n{'='*70}\n{region.upper()}\n{'='*70}")
    print(f"  Full candidate grid (from tile geometry): {n_side}x{n_side} = {total_grid_cells}")
    print(f"  Metadata records (passed initial no-data precheck): {len(rows)}")
    early_skipped = total_grid_cells - len(rows)
    print(f"  Early-skipped before SCL check (early no-data precheck): {early_skipped} "
          f"({100*early_skipped/total_grid_cells:.1f}% of full grid)")
    print(f"  Accepted: {len(accepted_rows)}   Rejected (after SCL check): {len(rejected_rows)}")
    if rows:
        print(f"  Acceptance rate among metadata records: "
              f"{100*len(accepted_rows)/len(rows):.1f}%")
    print(f"  Tile edge pixels lost to non-multiple-of-{patch_size}: {edge_loss_px}px "
          f"strip along right/bottom (tile is {TILE_DIM}px, {n_side}*{patch_size}="
          f"{n_side*patch_size} used)")

    def composition_stats(subset: list[dict], label: str):
        if not subset:
            print(f"\n  {label}: no rows")
            return
        print(f"\n  {label} composition (mean over {len(subset)} patches):")
        for col in CLASS_COLUMNS:
            vals = [r[col] for r in subset]
            print(f"      {col:18s}: mean={np.mean(vals):6.2f}%  "
                  f"min={np.min(vals):6.2f}%  max={np.max(vals):6.2f}%")
        valid_vals = [r["valid_pct"] for r in subset]
        bad_vals = [r["bad_pct"] for r in subset]
        print(f"      {'valid_pct':18s}: mean={np.mean(valid_vals):6.2f}%  "
              f"min={np.min(valid_vals):6.2f}%  max={np.max(valid_vals):6.2f}%")
        print(f"      {'bad_pct':18s}: mean={np.mean(bad_vals):6.2f}%  "
              f"min={np.min(bad_vals):6.2f}%  max={np.max(bad_vals):6.2f}%")

    composition_stats(rows, "ALL metadata records")
    composition_stats(accepted_rows, "ACCEPTED only")

    def dominant_class_report(subset: list[dict], label: str):
        if not subset:
            return
        n = len(subset)
        veg_dom = sum(1 for r in subset if r["vegetation_pct"] > 50)
        water_dom = sum(1 for r in subset if r["water_pct"] > 50)
        soil_dom = sum(1 for r in subset if r["bare_soil_pct"] > 50)
        cloud_dom = sum(1 for r in subset
                         if sum(r[c] for c in CLOUD_RELATED_COLUMNS) > 50)
        print(f"\n  {label} — dominant land-cover class (>50% of patch):")
        print(f"      vegetation-dominant : {veg_dom:4d} ({100*veg_dom/n:.1f}%)")
        print(f"      water-dominant      : {water_dom:4d} ({100*water_dom/n:.1f}%)")
        print(f"      bare-soil-dominant  : {soil_dom:4d} ({100*soil_dom/n:.1f}%)")
        print(f"      cloud-related-dom.  : {cloud_dom:4d} ({100*cloud_dom/n:.1f}%)")

    dominant_class_report(rows, "ALL metadata records")
    dominant_class_report(accepted_rows, "ACCEPTED only")

    if accepted_rows:
        print(f"\n  Bad-pixel distribution among ACCEPTED patches:")
        bins = {"0-1%": 0, "1-2%": 0, "2-5%": 0, "5-10%": 0, ">10%": 0}
        for r in accepted_rows:
            bins[bad_pct_bin(r["bad_pct"])] += 1
        for bin_label, count in bins.items():
            pct = 100 * count / len(accepted_rows)
            print(f"      {bin_label:6s}: {count:4d} ({pct:.1f}%)")

    return {
        "region": region, "n_side": n_side, "rows": rows,
        "accepted_rows": accepted_rows, "rejected_rows": rejected_rows,
    }


def plot_spatial_grid(region_result: dict, output_path: Path):
    """
    Grid visualization: green = accepted, red = rejected (failed SCL check),
    grey = never became a metadata row (early no-data precheck skip).
    """
    n_side = region_result["n_side"]
    patch_size = None
    grid = np.zeros((n_side, n_side, 3))  # RGB, default black
    grid[:] = [0.6, 0.6, 0.6]  # grey = "no metadata row" (early skip)

    for r in region_result["rows"]:
        row_idx = int(r["row_off"]) // int(r["patch_size"])
        col_idx = int(r["col_off"]) // int(r["patch_size"])
        if row_idx < n_side and col_idx < n_side:
            grid[row_idx, col_idx] = [0.1, 0.7, 0.1] if r["accepted"] else [0.8, 0.1, 0.1]

    plt.figure(figsize=(6, 6))
    plt.imshow(grid)
    plt.title(f"{region_result['region']}: spatial coverage\n"
              f"green=accepted  red=rejected  grey=no real data (early skip)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved spatial coverage map to {output_path}")


def recommend_threshold(all_results: list[dict]):
    print(f"\n{'='*70}\nTHRESHOLD RECOMMENDATION\n{'='*70}")
    for res in all_results:
        accepted = res["accepted_rows"]
        if not accepted:
            continue
        bad_vals = [r["bad_pct"] for r in accepted]
        pct_near_limit = 100 * sum(1 for b in bad_vals if b > 5.0) / len(bad_vals)
        print(f"  {res['region']:12s}: {pct_near_limit:.1f}% of accepted patches have "
              f"bad_pct between 5-10% (near the current 10% limit)")
    print("\n  Read this alongside the per-region printouts above: if most accepted")
    print("  patches cluster in the 0-2% bad_pct bins with only a small tail near")
    print("  10%, the current threshold is reasonable (option A). If a large chunk")
    print("  sits in 5-10%, tightening to 5% (option B) would meaningfully clean the")
    print("  dataset without discarding much. If a region's issue is dominant-class")
    print("  imbalance rather than cloud/no-data (e.g. very few water-dominant")
    print("  patches despite high acceptance), that's option C -- a labeling/AOI")
    print("  problem, not a threshold problem, and changing bad_pct won't fix it.")


if __name__ == "__main__":
    csv_path = Path("outputs/patches_metadata.csv")
    rows = load_metadata(csv_path)

    by_region = {}
    for row in rows:
        by_region.setdefault(row["region"], []).append(row)

    all_results = []
    for region, region_rows in by_region.items():
        patch_size = int(region_rows[0]["patch_size"])
        result = report_region(region, region_rows, patch_size)
        plot_spatial_grid(result, Path("outputs") / f"{region}_spatial_coverage.png")
        all_results.append(result)

    recommend_threshold(all_results)