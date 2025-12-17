#!/usr/bin/env python3
"""
Balanceo de datasets DeepRacer por (steer, throttle) y ajuste 70/15/15.

- Trabaja por splits: train / validation / test.
- Para cada split, carga todos los dataset.csv:
    train:      base_dir/Deepracer_BaseMap_*/dataset.csv
    validation: base_dir/validation/Deepracer_BaseMap_*/dataset.csv
    test:       base_dir/test/Deepracer_BaseMap_*/dataset.csv

- Dentro de cada SPLIT (no por fichero):
    * Discretiza 'steer' en N bins.
    * Discretiza 'throttle' en M bins.
    * Construye bins 2D (steer_bin, thr_bin).
    * Cuenta muestras por bin.
    * Calcula un "cap" = percentil thr_percentile de ocupación de bins no vacíos.
    * Aplica un factor cap_scale < 1 para hacerlo más agresivo.
    * En cada bin, si hay más de cap muestras, hace downsampling a cap.
    * Resultado: distribución (steer, throttle) mucho más plana en cada split.

- Luego ajusta el tamaño de cada split para aproximar 70/15/15
  (solo downsampling, sin mover datos entre splits).

- Modifica los dataset.csv originales IN-PLACE (con .bak salvo --no-backup).
"""

import os
import glob
import argparse
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd


# -------------------- UTILIDADES BÁSICAS -------------------- #

def select_rows_mode(df: pd.DataFrame, n: int, mode: str, seed: int) -> pd.DataFrame:
    """
    Selecciona n filas del DataFrame df según el modo.
    Se usa como último paso para ajustar al número exacto.
    """
    total = len(df)
    if n <= 0:
        return df.iloc[0:0].copy()
    if total <= n:
        return df.copy()

    if mode == "first":
        return df.iloc[:n].copy()

    if mode == "last":
        return df.iloc[-n:].copy().reset_index(drop=True)

    if mode == "random":
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    if mode == "stride":
        idx = np.linspace(0, total - 1, n)
        idx = np.round(idx).astype(int)
        idx = np.unique(idx)[:n]
        return df.iloc[idx].reset_index(drop=True)

    # por defecto, first
    return df.iloc[:n].copy()


def flatten_by_steer_throttle(
    df: pd.DataFrame,
    n_steer_bins: int,
    n_thr_bins: int,
    perc: float,
    seed: int,
    cap_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Aplana la distribución conjunta (steer, throttle) dentro de df (todo el split):

      - Discretiza 'steer' en n_steer_bins.
      - Discretiza 'throttle' en n_thr_bins.
      - Construye bins 2D (steer_bin, thr_bin).
      - Cuenta muestras en cada bin 2D.
      - Calcula un 'cap' como el percentil 'perc' de ocupación de bins no vacíos.
      - Aplica un factor cap_scale (<1 para ser más agresivo).
      - En cada bin 2D, si count > cap, se muestrea aleatoriamente hasta cap.
      - Devuelve un subconjunto 'aplanado' de df.

    Mantiene todas las columnas originales (incluyendo csv_path, etc.).
    """
    if df.empty or ("throttle" not in df.columns) or ("steer" not in df.columns):
        return df

    df_work = df.copy()
    thr   = pd.to_numeric(df_work["throttle"], errors="coerce").to_numpy()
    steer = pd.to_numeric(df_work["steer"],    errors="coerce").to_numpy()

    valid = np.isfinite(thr) & np.isfinite(steer)
    df_work = df_work.loc[valid].reset_index(drop=True)
    thr   = thr[valid]
    steer = steer[valid]

    if len(df_work) == 0:
        return df.iloc[0:0].copy()

    thr_min, thr_max     = thr.min(), thr.max()
    steer_min, steer_max = steer.min(), steer.max()

    if thr_min == thr_max or steer_min == steer_max:
        # Todos los datos en el mismo valor de throttle o steer -> poco que aplanar
        return df_work

    # Bins en steer y throttle
    steer_edges = np.linspace(steer_min, steer_max, n_steer_bins + 1)
    thr_edges   = np.linspace(thr_min,   thr_max,   n_thr_bins   + 1)

    steer_bin = np.digitize(steer, steer_edges) - 1
    thr_bin   = np.digitize(thr,   thr_edges)   - 1

    steer_bin = np.clip(steer_bin, 0, n_steer_bins - 1)
    thr_bin   = np.clip(thr_bin,   0, n_thr_bins   - 1)

    # Bin 2D -> índice 1D
    df_work["_steer_bin"] = steer_bin
    df_work["_thr_bin"]   = thr_bin

    bin_index = steer_bin * n_thr_bins + thr_bin
    df_work["_bin2d"] = bin_index

    n_bins_2d = n_steer_bins * n_thr_bins
    counts = np.bincount(bin_index, minlength=n_bins_2d)
    non_zero = counts[counts > 0]
    if len(non_zero) == 0:
        return df_work.drop(columns=["_steer_bin", "_thr_bin", "_bin2d"], errors="ignore")

    # ---- cap agresivo ----
    raw_cap = np.percentile(non_zero, perc)
    raw_cap = max(raw_cap, 1.0)
    # cap_scale < 1 -> recorta más fuerte los bins muy poblados
    bin_cap = int(max(1, raw_cap * cap_scale))

    print(f"[DEBUG] flatten: perc={perc} -> raw_cap≈{raw_cap:.1f}, cap_scale={cap_scale} -> bin_cap={bin_cap}")

    rng = np.random.RandomState(seed)
    selected_idx: List[int] = []

    for b in range(n_bins_2d):
        sub = df_work[df_work["_bin2d"] == b]
        c = len(sub)
        if c == 0:
            continue
        if c <= bin_cap:
            selected_idx.extend(sub.index.tolist())
        else:
            chosen = rng.choice(sub.index.to_numpy(), size=bin_cap, replace=False)
            selected_idx.extend(chosen.tolist())

    df_flat = df_work.loc[selected_idx].copy().reset_index(drop=True)
    df_flat.drop(columns=["_steer_bin", "_thr_bin", "_bin2d"], inplace=True, errors="ignore")
    return df_flat


def compute_targets_70_15_15(split_sizes: Dict[str, int],
                             ratios: Dict[str, float]) -> Dict[str, int]:
    """
    Dado el tamaño de cada split tras aplanar, calcula objetivos finales
    para aproximar 70/15/15 sin mover muestras entre splits (solo recorta si hace falta).

    split_sizes: {"train": N_train, "validation": N_val, "test": N_test}
    ratios:      {"train": 0.7, "validation": 0.15, "test": 0.15}
    """
    active = {k: v for k, v in split_sizes.items() if v > 0}
    if not active:
        return {k: 0 for k in split_sizes.keys()}

    sum_r = sum(ratios[k] for k in active.keys())
    if sum_r <= 0:
        raise ValueError("Ratios sum is <= 0; invalid configuration.")

    norm_ratios = {k: ratios[k] / sum_r for k in active.keys()}

    # Máximo total T tal que ratio_k * T <= N_k para cada split activo.
    possible_T = []
    for k, size in active.items():
        rk = norm_ratios[k]
        possible_T.append(size / rk)
    T_max = int(min(possible_T))

    if T_max <= 0:
        return {k: (0 if k not in active else active[k]) for k in split_sizes.keys()}

    targets = {}
    for k in split_sizes.keys():
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
    """
    Escribe en disco los dataset.csv para un split (train/val/test),
    usando df_split_final (aplanado) para cada fichero.

    df_split_final debe contener:
      - columna 'csv_path'
      - columnas de datos originales
    """
    if df_split_original is None or df_split_original.empty:
        print(f"[WRITE] Split '{split_name}': sin datos originales, se omite.")
        return

    helper_cols = {"csv_path", "base_dir", "split"}
    cols_out = [c for c in df_split_original.columns if c not in helper_cols]

    grouped = df_split_final.groupby("csv_path", dropna=False)
    orig_paths = sorted(df_split_original["csv_path"].unique())

    print(f"[WRITE] Split '{split_name}': reescribiendo {len(orig_paths)} dataset.csv")

    for p in orig_paths:
        if p in grouped.groups:
            df_out = grouped.get_group(p)[cols_out].copy()
        else:
            # si un fichero se queda sin muestras, escribir cabecera vacía
            try:
                orig_df = pd.read_csv(p)
                df_out = orig_df.iloc[0:0][cols_out].copy()
            except Exception as e:
                print(f"   [{split_name.upper()}] WARN: no se pudo restaurar header de {p}: {e}")
                df_out = pd.DataFrame(columns=cols_out)

        print(f"   [{split_name.upper()}] {p} -> {len(df_out)} filas")

        if dry_run:
            continue

        # backup
        if (not no_backup) and os.path.isfile(p):
            bak = p + ".bak"
            try:
                shutil.copy2(p, bak)
                print(f"      backup: {bak}")
            except Exception as e:
                print(f"      [WARN] no se pudo crear backup {bak}: {e}")

        # guardar
        try:
            df_out.to_csv(p, index=False)
        except Exception as e:
            print(f"      [ERROR] guardando {p}: {e}")


# -------------------- CARGA POR SPLIT -------------------- #

def collect_split_datasets(base_dir: str, split_name: str) -> pd.DataFrame | None:
    """
    Carga todos los dataset.csv de un split y los concatena:
      train      -> base_dir/Deepracer_BaseMap_*/dataset.csv
      validation -> base_dir/validation/Deepracer_BaseMap_*/dataset.csv
      test       -> base_dir/test/Deepracer_BaseMap_*/dataset.csv

    Añade columnas:
      - csv_path
      - base_dir
      - split

    NO exige columna 'estado', solo 'steer' y 'throttle'.
    """
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
            print(f"[SKIP] {csv_path} no se pudo leer: {e}")
            continue

        if not {"throttle", "steer"}.issubset(df.columns):
            print(f"[SKIP] {csv_path} sin columnas necesarias (throttle/steer).")
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


# -------------------- BALANCEO POR SPLIT -------------------- #

def balance_split_steer_throttle(
    df_split: pd.DataFrame,
    steer_bins: int,
    thr_bins: int,
    thr_percentile: float,
    cap_scale: float,
    seed: int,
) -> pd.DataFrame:
    """
    Para un split (train/validation/test):

      - Toma TODAS las filas del split.
      - Aplana la distribución de (steer, throttle) a nivel global del split
        con flatten_by_steer_throttle().
      - Baraja al final.
    """
    if df_split is None or df_split.empty:
        return df_split

    print(f"\n[Split] Aplanando por (steer, throttle) a nivel global del split...")
    df_flat = flatten_by_steer_throttle(
        df_split,
        n_steer_bins=steer_bins,
        n_thr_bins=thr_bins,
        perc=thr_percentile,
        seed=seed,
        cap_scale=cap_scale,
    )
    print(f"[Split] {len(df_split)} -> {len(df_flat)} filas tras aplanamiento 2D")

    # Barajar
    df_flat = df_flat.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df_flat


# -------------------- MAIN -------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Aplana la distribución 2D (steer, throttle) en train/validation/test por separado "
            "y luego ajusta a un ratio global 70/15/15 (solo downsampling). "
            "NO usa 'estado'."
        )
    )
    ap.add_argument("--base-dir", default="../datasets",
                    help="Directorio base de datasets (por defecto ../datasets).")
    ap.add_argument("--steer-bins", type=int, default=30,
                    help="Número de bins para STEER en el aplanamiento 2D (default=30).")
    ap.add_argument("--thr-bins", type=int, default=30,
                    help="Número de bins para THROTTLE en el aplanamiento 2D (default=30).")
    ap.add_argument("--thr-percentile", type=float, default=50.0,
                    help="Percentil de ocupación por bin 2D (steer,throttle) (default=50).")
    ap.add_argument("--cap-scale", type=float, default=0.4,
                    help="Factor multiplicativo sobre el cap del percentil "
                         "(<1 => aplanamiento más agresivo, default=0.4).")
    ap.add_argument("--keep", choices=["first", "last", "random", "stride"],
                    default="random",
                    help="(Apenas usado ahora) estrategia si quisieras recortar manualmente.")
    ap.add_argument("--seed", type=int, default=42, help="Semilla base.")
    ap.add_argument("--no-backup", action="store_true",
                    help="No crear .bak de los dataset.csv originales.")
    ap.add_argument("--dry-run", action="store_true",
                    help="No escribir cambios, solo mostrar lo que se haría.")
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    print(f"[INFO] Base dir       : {base_dir}")
    print(f"[INFO] steer_bins     : {args.steer_bins}")
    print(f"[INFO] thr_bins       : {args.thr_bins}")
    print(f"[INFO] thr_percentile : {args.thr_percentile}")
    print(f"[INFO] cap_scale      : {args.cap_scale}")

    # 1) Cargar splits
    df_train_orig = collect_split_datasets(base_dir, "train")
    df_val_orig   = collect_split_datasets(base_dir, "validation")
    df_test_orig  = collect_split_datasets(base_dir, "test")

    splits_orig = {
        "train": df_train_orig,
        "validation": df_val_orig,
        "test": df_test_orig,
    }

    # 2) Aplanamiento por split (sin mezclar datos entre splits)
    df_train_bal = balance_split_steer_throttle(
        df_train_orig,
        args.steer_bins, args.thr_bins,
        args.thr_percentile,
        args.cap_scale,
        args.seed
    ) if df_train_orig is not None else None

    df_val_bal = balance_split_steer_throttle(
        df_val_orig,
        args.steer_bins, args.thr_bins,
        args.thr_percentile,
        args.cap_scale,
        args.seed + 1
    ) if df_val_orig is not None else None

    df_test_bal = balance_split_steer_throttle(
        df_test_orig,
        args.steer_bins, args.thr_bins,
        args.thr_percentile,
        args.cap_scale,
        args.seed + 2
    ) if df_test_orig is not None else None

    splits_bal = {
        "train": df_train_bal,
        "validation": df_val_bal,
        "test": df_test_bal,
    }

    # 3) Calcular tamaños y objetivos 70/15/15
    sizes = {k: (0 if v is None else len(v)) for k, v in splits_bal.items()}
    print("\n[INFO] Tamaños tras aplanar (antes 70/15/15):")
    for k, s in sizes.items():
        print(f"   {k}: {s} filas")

    ratios = {"train": 0.7, "validation": 0.15, "test": 0.15}
    targets = compute_targets_70_15_15(sizes, ratios)

    print("\n[INFO] Target sizes para ratio 70/15/15 (aprox, solo downsampling):")
    for k in ["train", "validation", "test"]:
        print(f"   {k}: {targets[k]} filas")

    # 4) Downsampling para cumplir targets (dentro de cada split)
    rng = np.random.RandomState(args.seed)
    splits_final = {}
    for k in ["train", "validation", "test"]:
        df_bal = splits_bal[k]
        t = targets[k]
        if df_bal is None or t == 0:
            splits_final[k] = None
            continue
        if len(df_bal) <= t:
            splits_final[k] = df_bal
        else:
            idx = rng.choice(df_bal.index.to_numpy(), size=t, replace=False)
            splits_final[k] = df_bal.loc[idx].copy().reset_index(drop=True)
        print(f"   [FINAL] split {k}: {0 if splits_final[k] is None else len(splits_final[k])} filas")

    if args.dry_run:
        print("\n[DRY-RUN] No se escribirá ningún archivo.")
        return

    # 5) Escribir de vuelta los CSV de cada split
    for split_name in ["train", "validation", "test"]:
        df_orig  = splits_orig[split_name]
        df_final = splits_final[split_name]
        if df_orig is None or df_final is None:
            print(f"[WRITE] Split '{split_name}' sin datos finales u originales, se omite.")
            continue
        write_split_inplace(
            df_split_final=df_final,
            df_split_original=df_orig,
            split_name=split_name,
            no_backup=args.no_backup,
            dry_run=args.dry_run
        )

    print("\n[DONE] Aplanamiento 2D (steer, throttle) agresivo + ratio 70/15/15 aplicado in-place.")


if __name__ == "__main__":
    main()
