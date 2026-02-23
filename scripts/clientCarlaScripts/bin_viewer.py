#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "../datasets"
MAX_POINTS_PER_SPLIT = 20000   # máx puntos por split para el plot
N_BINS_STEER = 60             # nº de cuadrados en eje X
N_BINS_THR   = 60             # nº de cuadrados en eje Y
CLIP_PERCENTILE = 98          # recortar histograma al p-ésimo percentil

def load_split(pattern, name):
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No CSV found for {name}.")
        return None

    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Cannot read {p}: {e}")
            continue

        if not {"steer", "throttle"}.issubset(df.columns):
            print(f"Missing steer/throttle in {p}")
            continue

        dfs.append(df[["steer", "throttle"]].copy())

    if not dfs:
        print(f"No usable CSV for {name}.")
        return None

    all_df = pd.concat(dfs, ignore_index=True)
    print(f"[OK] {name}: {len(all_df)} rows.")
    return all_df


def maybe_subsample(df, max_points, seed=42):
    if df is None:
        return None
    n = len(df)
    if n <= max_points:
        return df
    rs = np.random.RandomState(seed)
    idx = rs.choice(df.index.to_numpy(), size=max_points, replace=False)
    return df.loc[idx].reset_index(drop=True)


def heatmap_one_split(df, title, cmap):
    """
      - Y fijo [0,1], Recorte al percentil CLIP_PERCENTILE, Normalización a [0,1] para oscurecer el mapa
    """
    if df is None or df.empty:
        print(f"No data for {title}, skipping.")
        return

    x = pd.to_numeric(df["steer"], errors="coerce")
    y = pd.to_numeric(df["throttle"], errors="coerce")
    mask = x.notna() & y.notna()

    x = x[mask].to_numpy()
    y = y[mask].to_numpy()

    n_valid = len(x)
    if n_valid == 0:
        print(f"No valid steer/throttle for {title}, skipping.")
        return

    x_min, x_max = x.min(), x.max()
    y_min, y_max = 0.0, 1.0

    hist, x_edges, y_edges = np.histogram2d(
        x, y,
        bins=[N_BINS_STEER, N_BINS_THR],
        range=[[x_min, x_max], [y_min, y_max]]
    )

    # ----- recortar y normalizar para oscurecer -----
    nonzero = hist[hist > 0]
    if nonzero.size > 0:
        vmax_clip = np.percentile(nonzero, CLIP_PERCENTILE)
        if vmax_clip <= 0:
            vmax_clip = nonzero.max()
    else:
        vmax_clip = 1.0

    hist_clipped = np.minimum(hist, vmax_clip)
    hist_norm = hist_clipped / vmax_clip

    plt.figure(figsize=(9, 6), dpi=130)
    im = plt.imshow(
        hist_norm.T,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0
    )

    cbar = plt.colorbar(im)
    cbar.set_label(f"Densidad (0–1, clip {CLIP_PERCENTILE}%)")

    plt.title(f"{title} – Heatmap (steer vs throttle)  n={n_valid}")
    plt.xlabel("Steer")
    plt.ylabel("Throttle")
    plt.ylim(y_min, y_max)
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.show()


def main():
    df_train = load_split(os.path.join(BASE_DIR, "Deepracer_BaseMap_*", "dataset.csv"), "TRAIN")
    df_val   = load_split(os.path.join(BASE_DIR, "validation", "Deepracer_BaseMap_*", "dataset.csv"), "VALIDATION")
    df_test  = load_split(os.path.join(BASE_DIR, "test", "Deepracer_BaseMap_*", "dataset.csv"), "TEST")

    df_train = maybe_subsample(df_train, MAX_POINTS_PER_SPLIT)
    df_val   = maybe_subsample(df_val,   MAX_POINTS_PER_SPLIT)
    df_test  = maybe_subsample(df_test,  MAX_POINTS_PER_SPLIT)

    if df_train is None and df_val is None and df_test is None:
        print("[ERROR] No data in any split.")
        return

    heatmap_one_split(df_train, "TRAIN",      "Blues")
    heatmap_one_split(df_val,   "VALIDATION", "Oranges")
    heatmap_one_split(df_test,  "TEST",       "Greens")


if __name__ == "__main__":
    main()
