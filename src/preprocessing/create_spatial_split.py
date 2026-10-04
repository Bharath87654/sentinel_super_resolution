"""
create_spatial_split.py

Purpose:
    Create a spatially-separated train/val/test split of the HR dataset.

    Strategy:
        1. Group patches into spatial blocks of 4x4 patches (1024x1024 px).
        2. Assign ENTIRE blocks to train, val, or test — never split a block.
        3. Assignment is done independently per region (each region is a
           separate Sentinel-2 tile with no inter-region spatial correlation).
        4. Blocks are shuffled (seeded) then greedily assigned to val (~10%),
           test (~10%), and train (remainder) by cumulative patch count.

    This prevents spatial leakage: neighboring patches sharing landscape
    context cannot end up in different splits.

    Outputs:
        data/processed/hr_split_index.csv
            - All columns from hr_patches_index.csv
            - Plus: spatial_block_id, split

        outputs/spatial_split_maps/
            - Per-region scatter plot of blocks colored by split

    This script is READ-ONLY on the HR .npy files. It does not copy,
    move, or modify any patch data.

Usage:
    python src/preprocessing/create_spatial_split.py
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuration ────────────────────────────────────────────────────
SPLIT_CONFIG = {
    "block_size_pixels": 1024,   # 4 patches × 256 px = 1024 px per block side
    "patch_size": 256,
    "val_fraction": 0.10,
    "test_fraction": 0.10,
    "random_seed": 42,
}

REGIONS = ["forest", "urban", "agriculture", "water"]

SPLIT_COLORS = {
    "train": "#2196F3",       # blue
    "val":   "#FF9800",       # orange
    "test":  "#E91E63",       # pink
}


def compute_block_id(row_off: int, col_off: int, block_size: int) -> str:
    """
    Compute a block identifier from patch pixel offsets.
    block_row and block_col are integer indices into the block grid.
    """
    block_row = row_off // block_size
    block_col = col_off // block_size
    return f"blk_r{block_row}_c{block_col}"


def assign_blocks_to_splits(blocks: list[dict], config: dict,
                             rng: random.Random) -> dict[str, str]:
    """
    Given a list of block dicts [{"block_id": ..., "count": ...}, ...],
    shuffle them and greedily assign to val, test, train.

    Returns a dict mapping block_id -> split name.
    """
    total_patches = sum(b["count"] for b in blocks)
    target_val = int(round(total_patches * config["val_fraction"]))
    target_test = int(round(total_patches * config["test_fraction"]))

    # Shuffle blocks with the seeded RNG
    shuffled = list(blocks)
    rng.shuffle(shuffled)

    assignment = {}
    val_count = 0
    test_count = 0

    for block in shuffled:
        bid = block["block_id"]
        n = block["count"]

        if val_count < target_val:
            assignment[bid] = "val"
            val_count += n
        elif test_count < target_test:
            assignment[bid] = "test"
            test_count += n
        else:
            assignment[bid] = "train"

    return assignment


def create_split(project_root: Path, config: dict) -> pd.DataFrame:
    """
    Main split logic. Returns the full split DataFrame.
    """
    index_csv = project_root / "data" / "processed" / "hr_patches_index.csv"
    df = pd.read_csv(index_csv)
    block_size = config["block_size_pixels"]

    print(f"  Loaded {len(df)} patches from HR index")
    print(f"  Block size: {block_size} px "
          f"({block_size // config['patch_size']}×"
          f"{block_size // config['patch_size']} patches)")

    # ── Compute block IDs ────────────────────────────────────────
    df["spatial_block_id"] = df.apply(
        lambda row: f"{row['region']}_{compute_block_id(row['row_off'], row['col_off'], block_size)}",
        axis=1
    )

    # ── Assign blocks per region ─────────────────────────────────
    # Use a separate RNG per region, all derived from the same master seed,
    # so that adding/removing a region doesn't change other regions' splits.
    master_rng = random.Random(config["random_seed"])
    region_seeds = {r: master_rng.randint(0, 2**31) for r in REGIONS}

    df["split"] = ""

    for region in REGIONS:
        region_df = df[df["region"] == region]
        if region_df.empty:
            print(f"  [{region}] No patches, skipping.")
            continue

        # Build block list
        block_groups = region_df.groupby("spatial_block_id").size().reset_index(
            name="count"
        )
        blocks = [
            {"block_id": row["spatial_block_id"], "count": row["count"]}
            for _, row in block_groups.iterrows()
        ]

        region_rng = random.Random(region_seeds[region])
        assignment = assign_blocks_to_splits(blocks, config, region_rng)

        # Apply assignment
        for bid, split_name in assignment.items():
            mask = df["spatial_block_id"] == bid
            df.loc[mask, "split"] = split_name

        # Report
        n_blocks = len(blocks)
        n_patches = len(region_df)
        split_counts = {s: sum(b["count"] for b in blocks if assignment[b["block_id"]] == s)
                        for s in ["train", "val", "test"]}
        split_blocks = {s: sum(1 for b in blocks if assignment[b["block_id"]] == s)
                        for s in ["train", "val", "test"]}

        print(f"\n  {region.upper()}: {n_patches} patches in {n_blocks} blocks")
        for s in ["train", "val", "test"]:
            pct = split_counts[s] / n_patches * 100 if n_patches > 0 else 0
            print(f"    {s:5s}: {split_counts[s]:5d} patches "
                  f"({pct:5.1f}%) in {split_blocks[s]:3d} blocks")

    return df


def verify_split(df: pd.DataFrame):
    """
    Exhaustive verification of split integrity.
    """
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    issues = []

    # 1. Every patch has a split
    unassigned = df[df["split"] == ""]
    if len(unassigned) > 0:
        issues.append(f"{len(unassigned)} patches have no split assignment")
    else:
        print("  [PASS] All patches assigned to a split")

    # 2. Total count
    total = len(df)
    print(f"  [INFO] Total patches: {total}")

    # 3. No duplicate filenames
    dup_files = df[df.duplicated(subset=["filename"], keep=False)]
    if len(dup_files) > 0:
        issues.append(f"{len(dup_files)} duplicate filenames found")
        print(f"  [FAIL] {len(dup_files)} duplicate filenames")
    else:
        print("  [PASS] No duplicate filenames")

    # 4. No spatial block in multiple splits
    block_splits = df.groupby("spatial_block_id")["split"].nunique()
    multi_split_blocks = block_splits[block_splits > 1]
    if len(multi_split_blocks) > 0:
        issues.append(f"{len(multi_split_blocks)} blocks appear in multiple splits")
        print(f"  [FAIL] {len(multi_split_blocks)} blocks in multiple splits:")
        for bid in multi_split_blocks.index[:5]:
            splits = df[df["spatial_block_id"] == bid]["split"].unique()
            print(f"         {bid}: {list(splits)}")
    else:
        n_total_blocks = df["spatial_block_id"].nunique()
        print(f"  [PASS] All {n_total_blocks} blocks are in exactly one split")

    # 5. Per-region, per-split counts
    print(f"\n  Per-region split counts:")
    pivot = df.groupby(["region", "split"]).size().unstack(fill_value=0)
    # Reorder columns
    for col in ["train", "val", "test"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["train", "val", "test"]]
    pivot["total"] = pivot.sum(axis=1)
    print(pivot.to_string(col_space=8))

    # 6. Per-region percentages
    print(f"\n  Per-region split percentages:")
    for region in REGIONS:
        if region not in pivot.index:
            continue
        row = pivot.loc[region]
        total_r = row["total"]
        pcts = {s: row[s] / total_r * 100 if total_r > 0 else 0
                for s in ["train", "val", "test"]}
        print(f"    {region:15s}: train={pcts['train']:5.1f}%  "
              f"val={pcts['val']:5.1f}%  test={pcts['test']:5.1f}%")

    # 7. Overall counts and percentages
    print(f"\n  Overall split counts:")
    overall = df["split"].value_counts()
    for s in ["train", "val", "test"]:
        n = overall.get(s, 0)
        pct = n / total * 100 if total > 0 else 0
        print(f"    {s:5s}: {n:5d} ({pct:.1f}%)")

    # 8. Block-level statistics
    print(f"\n  Block statistics:")
    block_sizes = df.groupby("spatial_block_id").size()
    print(f"    Total blocks      : {len(block_sizes)}")
    print(f"    Min patches/block : {block_sizes.min()}")
    print(f"    Max patches/block : {block_sizes.max()}")
    print(f"    Mean patches/block: {block_sizes.mean():.1f}")
    print(f"    Median            : {block_sizes.median():.0f}")

    # 9. Blocks per split
    block_split_counts = df.groupby("split")["spatial_block_id"].nunique()
    print(f"\n  Blocks per split:")
    for s in ["train", "val", "test"]:
        print(f"    {s:5s}: {block_split_counts.get(s, 0)} blocks")

    if issues:
        print(f"\n  [ISSUES] {len(issues)} problems found:")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print(f"\n  [ALL CHECKS PASSED]")

    return len(issues) == 0


def save_spatial_visualizations(df: pd.DataFrame, output_dir: Path,
                                 config: dict):
    """
    Create one scatter plot per region showing patch positions colored by split.
    Each dot represents a patch; spatial blocks are outlined.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    block_size = config["block_size_pixels"]
    patch_size = config["patch_size"]

    for region in REGIONS:
        rdf = df[df["region"] == region]
        if rdf.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot each patch as a colored square
        for _, row in rdf.iterrows():
            color = SPLIT_COLORS[row["split"]]
            rect = plt.Rectangle(
                (row["col_off"], row["row_off"]),
                patch_size, patch_size,
                facecolor=color, edgecolor="none", alpha=0.7
            )
            ax.add_patch(rect)

        # Draw block grid lines
        max_row = rdf["row_off"].max() + patch_size
        max_col = rdf["col_off"].max() + patch_size
        # Extend grid slightly beyond data extent
        grid_max_row = ((max_row // block_size) + 1) * block_size
        grid_max_col = ((max_col // block_size) + 1) * block_size

        for r in range(0, grid_max_row + 1, block_size):
            ax.axhline(y=r, color="gray", linewidth=0.3, alpha=0.5)
        for c in range(0, grid_max_col + 1, block_size):
            ax.axvline(x=c, color="gray", linewidth=0.3, alpha=0.5)

        ax.set_xlim(-50, max_col + 50)
        ax.set_ylim(max_row + 50, -50)  # invert y so row=0 is at top
        ax.set_aspect("equal")
        ax.set_xlabel("col_off (pixels)", fontsize=10)
        ax.set_ylabel("row_off (pixels)", fontsize=10)

        # Region stats for title
        counts = rdf["split"].value_counts()
        n = len(rdf)
        title_parts = []
        for s in ["train", "val", "test"]:
            c = counts.get(s, 0)
            title_parts.append(f"{s}={c} ({c/n*100:.1f}%)")
        ax.set_title(f"{region.upper()} — Spatial Split\n" +
                     "  |  ".join(title_parts), fontsize=11)

        # Legend
        legend_patches = [
            mpatches.Patch(color=SPLIT_COLORS[s], label=s, alpha=0.7)
            for s in ["train", "val", "test"]
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

        fig_path = output_dir / f"{region}_spatial_split.png"
        plt.savefig(fig_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fig_path.name}")


def main():
    project_root = Path(__file__).resolve().parent.parent.parent

    print("=" * 70)
    print("SPATIAL TRAIN / VAL / TEST SPLIT")
    print("=" * 70)
    print(f"  Block size    : {SPLIT_CONFIG['block_size_pixels']} px "
          f"({SPLIT_CONFIG['block_size_pixels'] // SPLIT_CONFIG['patch_size']}×"
          f"{SPLIT_CONFIG['block_size_pixels'] // SPLIT_CONFIG['patch_size']} patches)")
    print(f"  Target ratios : train ~{(1 - SPLIT_CONFIG['val_fraction'] - SPLIT_CONFIG['test_fraction'])*100:.0f}%  "
          f"val ~{SPLIT_CONFIG['val_fraction']*100:.0f}%  "
          f"test ~{SPLIT_CONFIG['test_fraction']*100:.0f}%")
    print(f"  Random seed   : {SPLIT_CONFIG['random_seed']}")

    # ── Create split ─────────────────────────────────────────────
    df = create_split(project_root, SPLIT_CONFIG)

    # ── Verify ───────────────────────────────────────────────────
    all_ok = verify_split(df)

    # ── Save split index CSV ─────────────────────────────────────
    output_csv = project_root / "data" / "processed" / "hr_split_index.csv"
    if all_ok:
        df.to_csv(output_csv, index=False)
        print(f"\n  Wrote split index: {output_csv}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
    else:
        print(f"\n  [SKIPPED] Not writing CSV due to verification failures.")

    # ── Save spatial visualizations ──────────────────────────────
    print("\n" + "=" * 70)
    print("SPATIAL SPLIT VISUALIZATIONS")
    print("=" * 70)

    viz_dir = project_root / "outputs" / "spatial_split_maps"
    save_spatial_visualizations(df, viz_dir, SPLIT_CONFIG)

    # ── Final summary ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    if all_ok:
        print("  Split index written successfully.")
        print("  Next steps:")
        print("    1. Review the spatial split maps in outputs/spatial_split_maps/")
        print("    2. Proceed to LR generation")
    else:
        print("  Fix issues before proceeding.")


if __name__ == "__main__":
    main()
