#!/usr/bin/env python3


import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- DEFAULT CONFIG ----------
BASE_DIR_DEFAULT   = "../datasets"
N_STEER_BINS       = 40
N_THROTTLE_BINS    = 40
PERC_BIN_CAP       = 60.0   # percentile for per-bin cap (60 ~ a bit more tolerant)
THR_ALPHA_DEFAULT  = 0.5    # aggressiveness of throttle flattening (0.5 strong, 0.9 soft)
# ------------------------------------


# ---------- HELPER: LOAD SPLITS ----------

def collect_split_datasets(base_dir: str, split_name: str) -> pd.DataFrame | None:

    if split_name == "train":
        pattern = os.path.join(base_dir, "Deepracer_BaseMap_*", "dataset.csv")
    elif split_name == "validation":
        pattern = os.path.join(base_dir, "validation", "Deepracer_BaseMap_*", "dataset.csv")
    elif split_name == "test":
        pattern = os.path.join(base_dir, "test", "Deepracer_BaseMap_*", "dataset.csv")
    else:
        raise ValueError(f"Unknown split_name: {split_name}")

    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[WARN] No dataset.csv for split '{split_name}' with pattern: {pattern}")
        return None

    dfs = []
    for csv_path in paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[SKIP] Could not read {csv_path}: {e}")
            continue

        if not {"throttle", "steer"}.issubset(df.columns):
            print(f"[SKIP] {csv_path} missing throttle/steer columns.")
            continue

        d = df.copy()
        d["csv_path"] = csv_path
        d["base_dir"] = os.path.dirname(csv_path)
        d["split"]    = split_name
        dfs.append(d)

    if not dfs:
        print(f"[WARN] No usable dataset.csv for split '{split_name}'.")
        return None

    full = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Split '{split_name}': loaded {len(full)} rows from {len(dfs)} CSVs.")
    return full


# ---------- HELPER: FLATTEN JOINT DENSITY ----------

def flatten_joint_density(df: pd.DataFrame,
                          n_steer_bins: int,
                          n_thr_bins: int,
                          perc: float,
                          seed: int) -> pd.DataFrame:

    if df is None or df.empty:
        print("[WARN] flatten_joint_density: empty DataFrame. Returning as is.")
        return df

    df = df.copy()
    rs = np.random.RandomState(seed)

    steer = pd.to_numeric(df["steer"], errors="coerce").to_numpy()
    thr   = pd.to_numeric(df["throttle"], errors="coerce").to_numpy()

    mask_valid = np.isfinite(steer) & np.isfinite(thr)
    df = df.loc[mask_valid].reset_index(drop=True)
    steer = steer[mask_valid]
    thr   = thr[mask_valid]

    if len(df) == 0:
        print("[WARN] No valid steer/throttle after filtering. Returning empty.")
        return df

    steer_min, steer_max = steer.min(), steer.max()
    thr_min,   thr_max   = thr.min(),   thr.max()

    steer_edges = np.linspace(steer_min, steer_max, n_steer_bins + 1)
    thr_edges   = np.linspace(thr_min,   thr_max,   n_thr_bins + 1)

    steer_bin = np.digitize(steer, steer_edges) - 1
    thr_bin   = np.digitize(thr,   thr_edges)   - 1

    steer_bin = np.clip(steer_bin, 0, n_steer_bins - 1)
    thr_bin   = np.clip(thr_bin,   0, n_thr_bins   - 1)

    df["_steer_bin"] = steer_bin
    df["_thr_bin"]   = thr_bin

    bin_index = steer_bin * n_thr_bins + thr_bin
    n_bins = n_steer_bins * n_thr_bins

    counts = np.bincount(bin_index, minlength=n_bins)
    non_zero = counts[counts > 0]
    if len(non_zero) == 0:
        print("[WARN] No non-empty bins; returning original df.")
        return df.drop(columns=["_steer_bin", "_thr_bin"], errors="ignore")

    bin_cap = int(np.percentile(non_zero, perc))
    if bin_cap < 1:
        bin_cap = 1

    print(f"   [DENSITY] steer range = [{steer_min:.3f}, {steer_max:.3f}]")
    print(f"   [DENSITY] thr   range = [{thr_min:.3f}, {thr_max:.3f}]")
    print(f"   [DENSITY] #bins = {n_bins}, non-empty = {len(non_zero)}")
    print(f"   [DENSITY] per-bin cap (perc {perc}) = {bin_cap}")

    df["_bin_index"] = bin_index
    selected_idx = []

    grouped = df.groupby("_bin_index", sort=False)
    for b, sub in grouped:
        c = len(sub)
        if c <= bin_cap:
            selected_idx.extend(sub.index.tolist())
        else:
            chosen = rs.choice(sub.index.to_numpy(), size=bin_cap, replace=False)
            selected_idx.extend(chosen.tolist())

    df_sel = df.loc[selected_idx].copy().reset_index(drop=True)
    df_sel.drop(columns=["_steer_bin", "_thr_bin", "_bin_index"], inplace=True, errors="ignore")

    print(f"   [DENSITY] kept {len(df_sel)} / {len(df)} rows for this split after 2D flattening.")
    return df_sel


def flatten_throttle_per_state(
    df_in: pd.DataFrame,
    nbins: int = 20,
    alpha: float = 0.7,
    seed: int = 42
) -> pd.DataFrame:

    if df_in is None or df_in.empty:
        return df_in

    df = df_in.copy()
    if "throttle" not in df.columns or "estado" not in df.columns:
        # Nothing to do if we don't have states or throttle
        return df

    # Separate states 1/2/3 from any other rows
    est_all = pd.to_numeric(df["estado"], errors="coerce").astype("Int64")
    mask_123 = est_all.isin([1, 2, 3])

    df_123  = df[mask_123].copy()
    df_rest = df[~mask_123].copy()  # kept untouched (e.g. special states)

    if df_123.empty:
        return df

    # Normalize throttle to numeric and drop NaNs
    df_123["throttle"] = pd.to_numeric(df_123["throttle"], errors="coerce")
    df_123 = df_123.dropna(subset=["throttle"])
    if df_123.empty:
        return df

    thr_min = df_123["throttle"].min()
    thr_max = df_123["throttle"].max()
    eps = 1e-6
    bins = np.linspace(thr_min - eps, thr_max + eps, nbins + 1)

    rng = np.random.default_rng(seed)
    kept_parts = []

    # We recompute states only over df_123 to keep indexes aligned
    est_123 = pd.to_numeric(df_123["estado"], errors="coerce").astype("Int64")

    for s in [1, 2, 3]:
        sub = df_123[est_123 == s].copy()
        if sub.empty:
            continue

        thr_vals = sub["throttle"].to_numpy()
        bin_idx = np.digitize(thr_vals, bins) - 1
        bin_idx = np.clip(bin_idx, 0, nbins - 1)
        sub["thr_bin"] = bin_idx

        counts = sub.groupby("thr_bin").size()
        mean_count = counts.mean()
        target = int(max(1, alpha * mean_count))

        for b, g in sub.groupby("thr_bin"):
            if len(g) <= target:
                kept_parts.append(g)
            else:
                idx_keep = rng.choice(g.index.to_numpy(), size=target, replace=False)
                kept_parts.append(sub.loc[idx_keep])

    if kept_parts:
        df_flat = pd.concat(kept_parts, ignore_index=True)
    else:
        df_flat = df_123.iloc[0:0].copy()

    if "thr_bin" in df_flat.columns:
        df_flat = df_flat.drop(columns=["thr_bin"])

    # Reattach non-1/2/3 rows
    df_out = pd.concat([df_flat, df_rest], ignore_index=True)
    df_out = df_out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df_out


# ---------- HELPER: COMPUTE TARGET COUNTS 70/15/15 ----------

def compute_targets_70_15_15(split_sizes: dict[str, int],
                             ratios: dict[str, float]) -> dict[str, int]:

    active = {k: v for k, v in split_sizes.items() if v > 0}
    if not active:
        return {"train": 0, "validation": 0, "test": 0}

    sum_r = sum(ratios[k] for k in active.keys())
    if sum_r <= 0:
        raise ValueError("Ratios sum is <= 0; invalid configuration.")

    norm_ratios = {k: ratios[k] / sum_r for k in active.keys()}

    possible_T = []
    for k, size in active.items():
        rk = norm_ratios[k]
        possible_T.append(size / rk)
    T_max = int(min(possible_T))

    if T_max <= 0:
        return {k: (0 if k not in active else active[k]) for k in split_sizes.keys()}

    targets = {}
    for k in ["train", "validation", "test"]:
        if k in active:
            t = int(round(norm_ratios[k] * T_max))
            t = min(t, split_sizes[k])
            targets[k] = t
        else:
            targets[k] = 0

    return targets


def write_split_inplace(df_split_final: pd.DataFrame,
                        df_split_original: pd.DataFrame,
                        split_name: str,
                        no_backup: bool,
                        dry_run: bool):

    if df_split_original is None or df_split_original.empty:
        print(f"[INFO] No original data for split '{split_name}'. Skipping write.")
        return

    helper_cols = {"csv_path", "base_dir", "split"}
    cols_out = [c for c in df_split_original.columns if c not in helper_cols]

    grouped = df_split_final.groupby("csv_path", dropna=False)
    orig_paths = sorted(df_split_original["csv_path"].unique())

    print(f"[WRITE] Split '{split_name}': rewriting {len(orig_paths)} dataset.csv files.")

    for p in orig_paths:
        if p in grouped.groups:
            df_out = grouped.get_group(p)[cols_out].copy()
        else:
            # If no final rows for this file, write empty with same header
            try:
                orig_df = pd.read_csv(p)
                df_out = orig_df.iloc[0:0][cols_out].copy()
            except Exception as e:
                print(f"   [{split_name.upper()}] WARN: cannot restore header for empty file {p}: {e}")
                df_out = pd.DataFrame(columns=cols_out)

        print(f"   [{split_name.upper()}] {p} -> {len(df_out)} rows")

        if dry_run:
            continue

        if (not no_backup) and os.path.isfile(p):
            import shutil
            bak = p + ".bak"
            try:
                shutil.copy2(p, bak)
                print(f"      backup: {bak}")
            except Exception as e:
                print(f"      [WARN] could not create backup {bak}: {e}")

        try:
            df_out.to_csv(p, index=False)
        except Exception as e:
            print(f"      [ERROR] saving {p}: {e}")


# ---------- VISUALIZATION ----------

def visualize_densities(base_dir: str):

    splits = {
        "Train":      "train",
        "Validation": "validation",
        "Test":       "test",
    }
    dfs = {}
    for label, split_name in splits.items():
        df = collect_split_datasets(base_dir, split_name)
        if df is not None:
            dfs[label] = df

    if not dfs:
        print("[PLOT] No data to visualize.")
        return

    # ---- STEER ----
    print("[PLOT] Steer density...")
    plt.figure(figsize=(8, 5), dpi=120)

    all_steer = []
    for df in dfs.values():
        all_steer.append(pd.to_numeric(df["steer"], errors="coerce").dropna().to_numpy())
    all_steer = np.concatenate(all_steer)
    xmin, xmax = all_steer.min(), all_steer.max()
    bins = 80

    for label, df in dfs.items():
        vals = pd.to_numeric(df["steer"], errors="coerce").dropna().to_numpy()
        if len(vals) < 2:
            continue
        counts, edges = np.histogram(vals, bins=bins, range=(xmin, xmax), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(centers, counts, label=f"{label} (n={len(vals)})")

    plt.title("STEER – Normalized density after flatten + 70/15/15")
    plt.xlabel("Steer")
    plt.ylabel("Density (area = 1)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- THROTTLE ----
    print("[PLOT] Throttle density...")
    plt.figure(figsize=(8, 5), dpi=120)

    all_thr = []
    for df in dfs.values():
        all_thr.append(pd.to_numeric(df["throttle"], errors="coerce").dropna().to_numpy())
    all_thr = np.concatenate(all_thr)
    tmin, tmax = all_thr.min(), all_thr.max()
    bins = 80

    for label, df in dfs.items():
        vals = pd.to_numeric(df["throttle"], errors="coerce").dropna().to_numpy()
        if len(vals) < 2:
            continue
        counts, edges = np.histogram(vals, bins=bins, range=(tmin, tmax), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(centers, counts, label=f"{label} (n={len(vals)})")

    plt.title("THROTTLE – Normalized density after flatten + 70/15/15")
    plt.xlabel("Throttle")
    plt.ylabel("Density (area = 1)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------- MAIN PIPELINE ----------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Flatten steer/throttle density INSIDE each split (train/val/test), "
            "then downsample each split to meet an overall 70/15/15 ratio, "
            "without mixing samples between splits. Original dataset.csv files are overwritten."
        )
    )
    ap.add_argument("--base-dir", default=BASE_DIR_DEFAULT,
                    help=f"Base directory of datasets (default: {BASE_DIR_DEFAULT})")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility.")
    ap.add_argument("--n-steer-bins", type=int, default=N_STEER_BINS,
                    help="Number of bins for steer axis (2D flatten).")
    ap.add_argument("--n-thr-bins", type=int, default=N_THROTTLE_BINS,
                    help="Number of bins for throttle axis (2D flatten & per-state flatten).")
    ap.add_argument("--perc", type=float, default=PERC_BIN_CAP,
                    help="Percentile for per-bin cap in 2D flattening (default 60).")
    ap.add_argument("--thr-alpha", type=float, default=THR_ALPHA_DEFAULT,
                    help="Aggressiveness for per-state throttle flattening (0.5 strong, 0.9 soft).")
    ap.add_argument("--no-backup", action="store_true",
                    help="Do NOT create .bak backups of original dataset.csv files.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Do NOT write any changes, only print what would be done.")
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    print(f"[INFO] Base dir    : {base_dir}")
    print(f"[INFO] Seed        : {args.seed}")
    print(f"[INFO] perc (2D)   : {args.perc}")
    print(f"[INFO] thr_alpha   : {args.thr_alpha}")

    # 1) Load each split separately
    df_train_orig = collect_split_datasets(base_dir, "train")
    df_val_orig   = collect_split_datasets(base_dir, "validation")
    df_test_orig  = collect_split_datasets(base_dir, "test")

    splits_orig = {
        "train": df_train_orig,
        "validation": df_val_orig,
        "test": df_test_orig,
    }

    # 2) Flatten density inside each split (first joint, then throttle per state)
    if df_train_orig is not None:
        tmp = flatten_joint_density(df_train_orig, args.n_steer_bins, args.n_thr_bins, args.perc, args.seed)
        df_train_flat = flatten_throttle_per_state(tmp, nbins=args.n_thr_bins,
                                                   alpha=args.thr_alpha, seed=args.seed)
    else:
        df_train_flat = None

    if df_val_orig is not None:
        tmp = flatten_joint_density(df_val_orig, args.n_steer_bins, args.n_thr_bins, args.perc, args.seed)
        df_val_flat = flatten_throttle_per_state(tmp, nbins=args.n_thr_bins,
                                                 alpha=args.thr_alpha, seed=args.seed)
    else:
        df_val_flat = None

    if df_test_orig is not None:
        tmp = flatten_joint_density(df_test_orig, args.n_steer_bins, args.n_thr_bins, args.perc, args.seed)
        df_test_flat = flatten_throttle_per_state(tmp, nbins=args.n_thr_bins,
                                                  alpha=args.thr_alpha, seed=args.seed)
    else:
        df_test_flat = None

    splits_flat = {
        "train": df_train_flat,
        "validation": df_val_flat,
        "test": df_test_flat,
    }

    # 3) Compute final target sizes for 70/15/15 (only downsampling)
    sizes_flat = {k: (0 if v is None else len(v)) for k, v in splits_flat.items()}
    print("\n[INFO] Sizes after per-split flattening:")
    for k, s in sizes_flat.items():
        print(f"   {k}: {s} rows")

    ratios = {"train": 0.7, "validation": 0.15, "test": 0.15}
    targets = compute_targets_70_15_15(sizes_flat, ratios)
    print("\n[INFO] Target sizes for 70/15/15 ratio (approx, no mixing):")
    for k in ["train", "validation", "test"]:
        print(f"   {k}: {targets[k]} rows")

    # 4) Downsample each split to its target size
    rng = np.random.RandomState(args.seed)
    splits_final = {}
    for k in ["train", "validation", "test"]:
        df_flat = splits_flat[k]
        if df_flat is None or targets[k] == 0:
            splits_final[k] = None
            continue
        if len(df_flat) <= targets[k]:
            splits_final[k] = df_flat
        else:
            idx = rng.choice(df_flat.index.to_numpy(), size=targets[k], replace=False)
            splits_final[k] = df_flat.loc[idx].copy().reset_index(drop=True)
        print(f"   [FINAL] split {k}: {0 if splits_final[k] is None else len(splits_final[k])} rows")

    # 5) Write back per split IN-PLACE
    for split_name in ["train", "validation", "test"]:
        df_orig  = splits_orig[split_name]
        df_final = splits_final[split_name]
        if df_orig is None or df_final is None:
            print(f"[INFO] Split '{split_name}' has no final data or no original data. Skipping write.")
            continue
        write_split_inplace(
            df_split_final=df_final,
            df_split_original=df_orig,
            split_name=split_name,
            no_backup=args.no_backup,
            dry_run=args.dry_run
        )

    if args.dry_run:
        print("\n[DRY-RUN] No files were modified. Skipping plots.")
        return

    print("\n[DONE] Flatten + per-state throttle flatten + 70/15/15 downsampling applied in-place.")
    print("[PLOT] Reloading modified datasets and plotting densities...")
    visualize_densities(base_dir)


if __name__ == "__main__":
    main()
