#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


BASE_DIR = "../datasets"


def cargar_split(pattern, nombre_split):
  
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No se encontraron CSV para {nombre_split}.")
        return None

    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"No se pudo leer {p}: {e}")
            continue

        if not {"throttle", "steer"}.issubset(df.columns):
            print(f"{p} no tiene throttle/steer.")
            continue

        dfs.append(df[["throttle", "steer"]].copy())

    if not dfs:
        print(f"No se pudo usar ningún CSV para {nombre_split}.")
        return None

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"{nombre_split}: {len(df_all)} filas combinadas.")
    return df_all


def main():
    # Patrones
    train_pattern = os.path.join(BASE_DIR, "Deepracer_BaseMap_*", "dataset.csv")
    val_pattern   = os.path.join(BASE_DIR, "validation", "Deepracer_BaseMap_*", "dataset.csv")
    test_pattern  = os.path.join(BASE_DIR, "test", "Deepracer_BaseMap_*", "dataset.csv")

    df_train = cargar_split(train_pattern, "TRAIN")
    df_val   = cargar_split(val_pattern, "VAL")
    df_test  = cargar_split(test_pattern, "TEST")

    # Preparar listas
    splits = []
    labels = []
    colors = []

    if df_train is not None:
        splits.append(df_train)
        labels.append("Train")
        colors.append("tab:blue")

    if df_val is not None:
        splits.append(df_val)
        labels.append("Validation")
        colors.append("tab:orange")

    if df_test is not None:
        splits.append(df_test)
        labels.append("Test")
        colors.append("tab:green")

    if not splits:
        print("No existen datos en ningún split.")
        return

 
    plt.figure(figsize=(9, 6), dpi=130)

    # Rango común en X
    all_steer = np.concatenate([df["steer"].astype(float).to_numpy() for df in splits])
    xs = np.linspace(all_steer.min(), all_steer.max(), 400)

    for df, lab, col in zip(splits, labels, colors):
        steer_vals = df["steer"].astype(float).to_numpy()
        if len(steer_vals) < 2:
            continue

        kde = gaussian_kde(steer_vals)
        ys = kde(xs) * len(steer_vals)  # densidad * número de muestras = densidad absoluta

        # Normalización INDIVIDUAL sobre el máximo de ESTA curva
        m = ys.max()
        if m > 0:
            ys = ys / m

        plt.plot(xs, ys, label=f"{lab} (n={len(steer_vals)})", color=col)

    plt.title("STEER – Densidad absoluta normalizada (máx = 1 por split)")
    plt.xlabel("Steer")
    plt.ylabel("Densidad normalizada")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(9, 6), dpi=130)

    all_thr = np.concatenate([df["throttle"].astype(float).to_numpy() for df in splits])
    xs = np.linspace(all_thr.min(), all_thr.max(), 400)

    for df, lab, col in zip(splits, labels, colors):
        thr_vals = df["throttle"].astype(float).to_numpy()
        if len(thr_vals) < 2:
            continue

        kde = gaussian_kde(thr_vals)
        ys = kde(xs) * len(thr_vals)

        # Normalización INDIVIDUAL sobre el máximo de ESTA curva
        m = ys.max()
        if m > 0:
            ys = ys / m

        plt.plot(xs, ys, label=f"{lab} (n={len(thr_vals)})", color=col)

    plt.title("THROTTLE – Densidad absoluta normalizada (máx = 1 por split)")
    plt.xlabel("Throttle")
    plt.ylabel("Densidad normalizada")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
