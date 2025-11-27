#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ====== BASE DIR ======
BASE_DIR = "../datasets"
# ======================

def cargar_split(pattern, nombre_split):
    """
    Carga y concatena todos los dataset.csv que encajen con el patrón.
    Devuelve DataFrame con throttle/steer o None si no hay nada.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[WARN] No se encontraron CSV para {nombre_split}.")
        return None

    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[SKIP] No se pudo leer {p}: {e}")
            continue

        if not {"throttle", "steer"}.issubset(df.columns):
            print(f"[SKIP] {p} no tiene throttle/steer.")
            continue

        dfs.append(df[["throttle", "steer"]].copy())

    if not dfs:
        print(f"[WARN] No se pudo usar ningún CSV para {nombre_split}.")
        return None

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[OK] {nombre_split}: {len(df_all)} filas combinadas.")
    return df_all


def normalizar_sobre_uno(list_of_arrays):
    """
    Normaliza todas las curvas de densidad sobre el máximo absoluto global.
    Devuelve las curvas normalizadas y el factor usado.
    """
    max_global = max(arr.max() for arr in list_of_arrays if arr is not None)
    if max_global <= 0:
        return list_of_arrays, 1.0
    return [arr / max_global if arr is not None else None for arr in list_of_arrays], max_global


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
        print("[ERROR] No existen datos en ningún split.")
        return

    # ======================================================
    # ====================   STEER   =======================
    # ======================================================

    plt.figure(figsize=(9, 6), dpi=130)

    # Rango común
    all_steer = np.concatenate([df["steer"].astype(float).to_numpy() for df in splits])
    xs = np.linspace(all_steer.min(), all_steer.max(), 400)

    curvas = []
    curvas_labels = []

    for df, lab, col in zip(splits, labels, colors):
        steer_vals = df["steer"].astype(float).to_numpy()
        if len(steer_vals) < 2:
            curvas.append(None)
            curvas_labels.append(lab)
            continue

        kde = gaussian_kde(steer_vals)
        ys = kde(xs)

        # ✔ Convertir a densidad *absoluta*
        ys_abs = ys * len(steer_vals)
        curvas.append(ys_abs)
        curvas_labels.append(lab)

    # ✔ Normalizar sobre 1 global (máximo de todos los splits)
    curvas_norm, max_global = normalizar_sobre_uno(curvas)

    print(f"\n[INFO] STEER max_global = {max_global}")

    # Dibujar
    for ys, lab, col in zip(curvas_norm, curvas_labels, colors):
        if ys is None: 
            continue
        plt.plot(xs, ys, label=lab, color=col)

    plt.title("STEER – Densidad absoluta normalizada (máx = 1)")
    plt.xlabel("Steer")
    plt.ylabel("Densidad absoluta normalizada")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ======================================================
    # ==================   THROTTLE   ======================
    # ======================================================

    plt.figure(figsize=(9, 6), dpi=130)

    all_thr = np.concatenate([df["throttle"].astype(float).to_numpy() for df in splits])
    xs = np.linspace(all_thr.min(), all_thr.max(), 400)

    curvas_thr = []
    curvas_thr_labels = []

    for df, lab, col in zip(splits, labels, colors):
        thr_vals = df["throttle"].astype(float).to_numpy()
        if len(thr_vals) < 2:
            curvas_thr.append(None)
            curvas_thr_labels.append(lab)
            continue

        kde = gaussian_kde(thr_vals)
        ys = kde(xs)

        ys_abs = ys * len(thr_vals)
        curvas_thr.append(ys_abs)
        curvas_thr_labels.append(lab)

    curvas_thr_norm, max_thr = normalizar_sobre_uno(curvas_thr)

    print(f"[INFO] THROTTLE max_global = {max_thr}")

    for ys, lab, col in zip(curvas_thr_norm, curvas_thr_labels, colors):
        if ys is None:
            continue
        plt.plot(xs, ys, label=lab, color=col)

    plt.title("THROTTLE – Densidad absoluta normalizada (máx = 1)")
    plt.xlabel("Throttle")
    plt.ylabel("Densidad absoluta normalizada")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
