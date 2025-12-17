#!/usr/bin/env python3
# Global balanced dataset based on the minimum state:
# 1) In each dataset, throttle distribution is edited to try and have same amount of each data values.
#    using bins and a max cap per bin (flatten_throttle_per_state).
# 2) Search lowest value (state,dataset) among states 1/2/3
# 3) For each dataset and state 1/2/3, only left global_min filas
# 4) Rewrite each dataset

import os
import glob
import argparse
import shutil
import pandas as pd
import numpy as np


def flatten_throttle_per_state(df_123: pd.DataFrame,
                               n_thr_bins: int = 30,
                               thr_alpha: float = 0.7,
                               seed: int = 42) -> pd.DataFrame:

    if df_123.empty or "throttle" not in df_123.columns or "estado" not in df_123.columns:
        return df_123

    rs = np.random.RandomState(seed)
    df_123 = df_123.copy()

    # Global throttle range in this dataset
    thr_all = pd.to_numeric(df_123["throttle"], errors="coerce").dropna().to_numpy()
    if thr_all.size == 0:
        return df_123

    thr_min, thr_max = thr_all.min(), thr_all.max()
    thr_edges = np.linspace(thr_min, thr_max, n_thr_bins + 1)

    selected_idx = []

    for est, sub in df_123.groupby("estado"):
        thr_vals = pd.to_numeric(sub["throttle"], errors="coerce").to_numpy()
        idx_sub  = sub.index.to_numpy()

        mask_valid = np.isfinite(thr_vals)
        if not mask_valid.any():
            continue

        thr_vals = thr_vals[mask_valid]
        idx_sub  = idx_sub[mask_valid]

        # Asign throttle bin
        bin_idx = np.digitize(thr_vals, thr_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_thr_bins - 1)

        counts = np.bincount(bin_idx, minlength=n_thr_bins)
        non_zero = counts[counts > 0]
        if non_zero.size == 0:
            continue

        cap = int(thr_alpha * non_zero.mean())
        if cap < 1:
            cap = 1

        print(f"  [THR-FLAT] state={est} -> cap={cap}, non empty bins={len(non_zero)}")

        # Cut by the bin
        for b in range(n_thr_bins):
            mask_b = (bin_idx == b)
            idx_b  = idx_sub[mask_b]
            c = len(idx_b)
            if c == 0:
                continue
            if c <= cap:
                selected_idx.extend(idx_b.tolist())
            else:
                chosen = rs.choice(idx_b, size=cap, replace=False)
                selected_idx.extend(chosen.tolist())

    if not selected_idx:
        print("  [THR-FLAT] Nothing selected; returning df_123 original.")
        return df_123

    df_flat = df_123.loc[sorted(set(selected_idx))].copy().reset_index(drop=True)
    print(f"  [THR-FLAT] throttle changed: {len(df_flat)} / {len(df_123)} rows.")
    return df_flat


def main():
    parser = argparse.ArgumentParser(
        description=(
            " Global balanced dataset based on the minimum state:"
            "1) In each dataset, throttle distribution is edited to try and have same amount of each data values."
            "using bins and a max cap per bin (flatten_throttle_per_state)."
            "2) Search lowest value (state,dataset) among states 1/2/3"
            "3) For each dataset and state 1/2/3, only left global_min filas"
            "4) Rewrite each dataset"
        )
    )
    parser.add_argument(
        "--pattern",
        default="../datasets/validation/Deepracer_BaseMap_*/dataset.csv",
        help="CSV search pattern (default: ../datasets/Deepracer_BaseMap_*/dataset.csv)."
    )
    parser.add_argument("--seed", type=int, default=42, help="seed.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes, just show them")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak files")
    parser.add_argument("--n-thr-bins", type=int, default=30,
                        help="Number of bins to change throttle per state in each dataset")
    parser.add_argument("--thr-alpha", type=float, default=0.7,
                        help="Factor for cap in the throttle bin (0.5=aggresive, 0.9=smooth).")
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(args.pattern))
    if not csv_paths:
        print("CSV not found with given pattern")
        return

    
    file_info = {}   
    global_min = None  

    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[SKIP] No se pudo leer {p}: {e}")
            continue

        if "estado" not in df.columns:
            print(f"[SKIP] {p} no tiene columna 'estado'.")
            continue

        # Split rows with states 1/2/3
        estado_full = pd.to_numeric(df["estado"], errors="coerce").astype("Int64")
        mask_123 = estado_full.isin([1, 2, 3])

        df_123_raw  = df[mask_123].copy()
        df_rest = df[~mask_123].copy()

        if df_123_raw.empty:
            print(f"[{os.path.basename(p)}] Does not have rows with states 1/2/3.")
            file_info[p] = {
                "df_123": df_123_raw,
                "df_rest": df_rest,
                "counts": {1: 0, 2: 0, 3: 0}
            }
            continue

        print(f"\n[{os.path.basename(p)}] Changing throttle...")
        df_123 = flatten_throttle_per_state(
            df_123_raw,
            n_thr_bins=args.n_thr_bins,
            thr_alpha=args.thr_alpha,
            seed=args.seed
        )

        estado_123 = pd.to_numeric(df_123["estado"], errors="coerce").astype("Int64")
        counts_local = {c: int((estado_123 == c).sum()) for c in [1, 2, 3]}
        print(f"[{os.path.basename(p)}] Local counts after THR-FLAT (1/2/3): "
              f"{counts_local[1]}/{counts_local[2]}/{counts_local[3]}")

        for c in [1, 2, 3]:
            cnt = counts_local[c]
            if cnt > 0:
                if global_min is None or cnt < global_min:
                    global_min = cnt

        file_info[p] = {
            "df_123": df_123,
            "df_rest": df_rest,
            "counts": counts_local
        }

    if not file_info:
        print("No datasets with state column")
        return

    if global_min is None or global_min <= 0:
        print("No state 1/2/3 > 0 in datasets.")
        return

    print(f"\n==Min global value among states 1/2/3 from all datasets (after THR-FLAT): {global_min} ==")

    if args.dry_run:
        print("[DRY-RUN] Only show what would have happened.\n")

    for p in csv_paths:
        if p not in file_info:
            continue

        info = file_info[p]
        df_123  = info["df_123"]
        df_rest = info["df_rest"]

        if df_123.empty:
            df_out = df_rest.copy()
            print(f"[{os.path.basename(p)}] No rows with states 1/2/3. (total={len(df_out)}).")
        else:
            estado_123 = pd.to_numeric(df_123["estado"], errors="coerce").astype("Int64")

            sampled_list = []
            final_counts = {1: 0, 2: 0, 3: 0}

            for c in [1, 2, 3]:
                df_c = df_123[estado_123 == c]
                n_c = len(df_c)
                if n_c == 0:
                    continue

                if n_c > global_min:
                    df_c = df_c.sample(n=global_min, random_state=args.seed)
                    final_counts[c] = global_min
                else:
                    final_counts[c] = n_c

                sampled_list.append(df_c)

            if sampled_list:
                df_bal = pd.concat(sampled_list, ignore_index=True)
            else:
                df_bal = pd.DataFrame(columns=df_123.columns)


            df_out = pd.concat([df_bal, df_rest], ignore_index=True)

            df_out = df_out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

            print(f"[{os.path.basename(p)}] Después del balanceo local "
                  f"(1/2/3) -> {final_counts[1]}/{final_counts[2]}/{final_counts[3]} "
                  f"| total con resto = {len(df_out)}")

        if args.dry_run:
            continue

        try:
            if not args.no_backup and os.path.isfile(p):
                shutil.copy2(p, p + ".bak")
                print(f"  Security copy created: {p}.bak")
        except Exception as e:
            print(f"  [WARN] Unable to create .bak {p}: {e}")

        try:
            df_out.to_csv(p, index=False)
            print(f"  Saved {p}")
        except Exception as e:
            print(f"  [ERROR] Saving {p}: {e}")


if __name__ == "__main__":
    main()
