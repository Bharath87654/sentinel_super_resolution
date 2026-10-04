"""
verify_hr_checkpoint.py

Purpose:
    Quick targeted verification of the HR dataset checkpoint before
    proceeding to LR generation. Checks:
      1. File counts per region (cross-referenced against hr_patches_index.csv)
      2. Array shape, dtype consistency (ALL patches, not just a sample)
      3. NaN / Inf / all-zero detection
      4. Per-channel statistics (min, max, mean, std) per region
      5. Value range sanity (uint16 Sentinel-2 reflectance)

    This is a READ-ONLY script. It does not modify any files.

Usage:
    python src/preprocessing/verify_hr_checkpoint.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import time


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    hr_dir = project_root / "data" / "processed" / "hr"
    index_csv = project_root / "data" / "processed" / "hr_patches_index.csv"

    regions = ["forest", "urban", "agriculture", "water"]
    channel_names = ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)"]

    # ── 1. Load and check index CSV ──────────────────────────────
    print("=" * 70)
    print("1. INDEX CSV CHECK")
    print("=" * 70)

    if not index_csv.exists():
        print(f"[FAIL] Index CSV not found: {index_csv}")
        return

    df = pd.read_csv(index_csv)
    print(f"  Index CSV rows        : {len(df)}")
    print(f"  Columns               : {list(df.columns)}")
    print(f"  Unique regions        : {sorted(df['region'].unique())}")
    print(f"  Unique dtypes in CSV  : {df['dtype'].unique()}")
    print(f"  Unique shapes in CSV  : {df['shape'].unique()}")
    print(f"  Unique patch_sizes    : {df['patch_size'].unique()}")

    csv_counts = df['region'].value_counts().to_dict()
    print(f"\n  Per-region counts from CSV:")
    for r in regions:
        print(f"    {r:15s}: {csv_counts.get(r, 0)}")

    # ── 2. Cross-reference file counts ───────────────────────────
    print("\n" + "=" * 70)
    print("2. FILE COUNT CROSS-REFERENCE")
    print("=" * 70)

    total_files = 0
    all_ok = True
    for region in regions:
        region_dir = hr_dir / region
        if not region_dir.exists():
            print(f"  [{region}] DIRECTORY MISSING")
            all_ok = False
            continue
        npy_files = sorted(region_dir.glob("*.npy"))
        csv_count = csv_counts.get(region, 0)
        match = len(npy_files) == csv_count
        status = "MATCH" if match else "MISMATCH"
        print(f"  {region:15s}: {len(npy_files)} files on disk, "
              f"{csv_count} in CSV  [{status}]")
        if not match:
            all_ok = False
        total_files += len(npy_files)

    print(f"\n  Total .npy files on disk: {total_files}")
    print(f"  Total rows in CSV      : {len(df)}")

    # ── 3. Full scan of ALL patches ──────────────────────────────
    print("\n" + "=" * 70)
    print("3. FULL PATCH SCAN (all patches)")
    print("=" * 70)

    all_shapes = set()
    all_dtypes = set()
    nan_patches = []
    inf_patches = []
    zero_patches = []
    high_value_patches = []

    # Per-region, per-channel accumulators
    region_stats = {}

    start_time = time.time()
    scanned = 0

    for region in regions:
        region_dir = hr_dir / region
        if not region_dir.exists():
            continue

        npy_files = sorted(region_dir.glob("*.npy"))
        ch_mins = [[] for _ in range(4)]
        ch_maxs = [[] for _ in range(4)]
        ch_means = [[] for _ in range(4)]
        ch_stds = [[] for _ in range(4)]
        ch_zeros = [[] for _ in range(4)]

        for fpath in npy_files:
            arr = np.load(fpath)
            all_shapes.add(arr.shape)
            all_dtypes.add(arr.dtype)
            scanned += 1

            # NaN / Inf checks (cast to float for the check)
            arr_f = arr.astype(np.float32)
            if np.isnan(arr_f).any():
                nan_patches.append(fpath.name)
            if np.isinf(arr_f).any():
                inf_patches.append(fpath.name)
            if np.all(arr == 0):
                zero_patches.append(fpath.name)
            if arr.max() > 20000:
                high_value_patches.append((fpath.name, int(arr.max())))

            for c in range(4):
                ch = arr[c]
                ch_mins[c].append(float(ch.min()))
                ch_maxs[c].append(float(ch.max()))
                ch_means[c].append(float(ch.mean()))
                ch_stds[c].append(float(ch.std()))
                ch_zeros[c].append(float(np.mean(ch == 0) * 100))

            if scanned % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"  ... scanned {scanned} patches ({elapsed:.1f}s)")

        region_stats[region] = {
            "count": len(npy_files),
            "min":  [float(np.min(ch_mins[c])) for c in range(4)],
            "max":  [float(np.max(ch_maxs[c])) for c in range(4)],
            "mean": [float(np.mean(ch_means[c])) for c in range(4)],
            "std":  [float(np.mean(ch_stds[c])) for c in range(4)],
            "zero_pct": [float(np.mean(ch_zeros[c])) for c in range(4)],
        }

    elapsed = time.time() - start_time
    print(f"\n  Scanned {scanned} patches in {elapsed:.1f}s")

    print(f"\n  Unique shapes : {list(all_shapes)}")
    print(f"  Unique dtypes : {[str(d) for d in all_dtypes]}")
    print(f"  NaN patches   : {len(nan_patches)}")
    print(f"  Inf patches   : {len(inf_patches)}")
    print(f"  All-zero patches: {len(zero_patches)}")
    if zero_patches:
        print(f"    First 5: {zero_patches[:5]}")
    print(f"  Patches with max > 20000: {len(high_value_patches)}")
    if high_value_patches:
        print(f"    First 5: {high_value_patches[:5]}")

    # ── 4. Per-region per-channel statistics ─────────────────────
    print("\n" + "=" * 70)
    print("4. PER-REGION PER-CHANNEL STATISTICS")
    print("=" * 70)

    for region in regions:
        if region not in region_stats:
            continue
        s = region_stats[region]
        print(f"\n--- {region.upper()} ({s['count']} patches) ---")
        for c in range(4):
            print(f"  Ch {c} ({channel_names[c]:12s}): "
                  f"Min={s['min'][c]:6.0f}  "
                  f"Max={s['max'][c]:6.0f}  "
                  f"Mean={s['mean'][c]:8.1f}  "
                  f"Std={s['std'][c]:7.1f}  "
                  f"Zero%={s['zero_pct'][c]:5.2f}")

    # ── 5. row_off / col_off range (for spatial split planning) ──
    print("\n" + "=" * 70)
    print("5. SPATIAL EXTENT (row_off / col_off ranges per region)")
    print("=" * 70)

    for region in regions:
        rdf = df[df["region"] == region]
        if rdf.empty:
            continue
        print(f"\n  {region.upper()}:")
        print(f"    row_off: {rdf['row_off'].min()} → {rdf['row_off'].max()}  "
              f"(unique: {rdf['row_off'].nunique()})")
        print(f"    col_off: {rdf['col_off'].min()} → {rdf['col_off'].max()}  "
              f"(unique: {rdf['col_off'].nunique()})")
        print(f"    Grid coverage: {rdf['row_off'].nunique()} rows × "
              f"{rdf['col_off'].nunique()} cols")

    # ── 6. Verdict ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CHECKPOINT VERDICT")
    print("=" * 70)

    shape_ok = all_shapes == {(4, 256, 256)}
    dtype_ok = all_dtypes == {np.dtype("uint16")}
    no_nan = len(nan_patches) == 0
    no_inf = len(inf_patches) == 0
    counts_match = all_ok

    checks = [
        ("Shape = (4, 256, 256)", shape_ok),
        ("Dtype = uint16", dtype_ok),
        ("No NaN patches", no_nan),
        ("No Inf patches", no_inf),
        ("File counts match CSV", counts_match),
    ]

    all_pass = True
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\n  [HR CHECKPOINT VALID] Ready to proceed to LR generation.")
    else:
        print("\n  [ISSUES DETECTED] Fix problems above before proceeding.")


if __name__ == "__main__":
    main()
