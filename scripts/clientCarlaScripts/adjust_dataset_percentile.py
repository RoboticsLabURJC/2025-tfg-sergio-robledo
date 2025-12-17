#!/usr/bin/env python3
"""
Balanceo de datasets DeepRacer con:
  - Balanceo por estados 1/2/3 usando percentil de conteos (por pista).
  - Aplanamiento de la distribución de throttle en cada (pista, estado).
  - Actúa sobre TRAIN, VALIDATION y TEST por separado.
  - Mantiene aproximadamente un ratio global 70/15/15 entre splits
    mediante downsampling (no se mezclan datos entre splits).
  - Modifica los dataset.csv originales IN-PLACE (con .bak opcional).
"""

import os
import glob
import argparse
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd


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


def flatten_by_throttle(
    df_state: pd.DataFrame,
    n_bins: int,
    perc: float,
    seed: int
) -> pd.DataFrame:
    """
    Aplana la distribución de throttle dentro de un subconjunto df_state (mismo estado y fichero).

    - Divide el throttle en n_bins.
    - Cuenta cuántas muestras hay por bin.
    - Calcula un 'cap' por bin como el percentil 'perc' de esos conteos.
    - En cada bin, si count > cap, se hace muestreo aleatorio hasta cap.
    - Devuelve un subconjunto 'aplanado' de df_state.
    """
    if df_state.empty or "throttle" not in df_state.columns:
        return df_state

    df = df_state.copy()
    thr = pd.to_numeric(df["throttle"], errors="coerce").to_numpy()
    valid = np.isfinite(thr)
    df = df.loc[valid].reset_index(drop=True)
    thr = thr[valid]

    if len(df) == 0:
        return df_state.iloc[0:0].copy()

    thr_min, thr_max = thr.min(), thr.max()
    if thr_min == thr_max:
        # Todo el throttle en el mismo valor -> nada que aplanar
        return df

    edges = np.linspace(thr_min, thr_max, n_bins + 1)
    bin_idx = np.digitize(thr, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    df["_thr_bin"] = bin_idx
    counts = np.bincount(bin_idx, minlength=n_bins)
    non_zero = counts[counts > 0]
    if len(non_zero) == 0:
        return df.drop(columns=["_thr_bin"], errors="ignore")

    bin_cap = int(np.percentile(non_zero, perc))
    if bin_cap < 1:
        bin_cap = 1

    rng = np.random.RandomState(seed)
    selected_idx: List[int] = []

    for b in range(n_bins):
        sub = df[df["_thr_bin"] == b]
        c = len(sub)
        if c == 0:
            continue
        if c <= bin_cap:
            selected_idx.extend(sub.index.tolist())
        else:
            chosen = rng.choice(sub.index.to_numpy(), size=bin_cap, replace=False)
            selected_idx.extend(chosen.tolist())

    df_flat = df.loc[selected_idx].copy().reset_index(drop=True)
    df_flat.drop(columns=["_thr_bin"], inplace=True, errors="ignore")
    return df_flat


def compute_targets_70_15_15(split_sizes: Dict[str, int],
                             ratios: Dict[str, float]) -> Dict[str, int]:
    """
    Dado el tamaño de cada split tras balancear/aplanar, calcula
    objetivos finales para aproximar 70/15/15 sin mover muestras
    entre splits (solo se recorta si hace falta).

    split_sizes: {"train": N_train, "validation": N_val, "test": N_test}
    ratios:      {"train": 0.7,   "validation": 0.15, "test": 0.15}
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
    usando df_split_final (balanceado) para cada fichero.

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

        if "estado" not in df.columns or not {"throttle", "steer"}.issubset(df.columns):
            print(f"[SKIP] {csv_path} sin columnas necesarias (estado/throttle/steer).")
            continue

        d = df.copy()
        d["csv_path"] = csv_path
        d["base_dir"] = os.path.dirname(csv_path)
        d["split"] = split_name
        dfs.append(d)

    if not dfs:
        print(f"[WARN] No usable dataset.csv for split '{split_name}'.")
        return None

    full = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Split '{split_name}': loaded {len(full)} rows from {len(dfs)} CSVs.")
    return full


# -------------------- BALANCEO POR SPLIT -------------------- #

def balance_split_states_and_throttle(
    df_split: pd.DataFrame,
    state_percentile: float,
    thr_bins: int,
    thr_percentile: float,
    keep_mode: str,
    seed: int,
    drop_non123: bool
) -> pd.DataFrame:
    """
    Aplica:
      - Balanceo por estados 1/2/3 usando percentil de conteos (por pista).
      - Aplanamiento de throttle para cada (pista, estado).
    TODO esto SEPARADO por split (train, val o test).
    """
    if df_split is None or df_split.empty:
        return df_split

    df = df_split.copy()
    estado = pd.to_numeric(df["estado"], errors="coerce").astype("Int64")
    mask_123 = estado.isin([1, 2, 3])
    df_123 = df[mask_123].copy()
    df_rest = df[~mask_123].copy()

    if df_123.empty:
        print("[WARN] Split sin estados 1/2/3. No se balancea estados.")
        return df if drop_non123 else df

    # Conteos por fichero y estado
    grouped_by_src = df_123.groupby("csv_path", dropna=False)
    counts_per_file: Dict[int, Dict[str, int]] = {1: {}, 2: {}, 3: {}}
    for src, g in grouped_by_src:
        est_local = pd.to_numeric(g["estado"], errors="coerce").astype("Int64")
        for c in [1, 2, 3]:
            counts_per_file[c][src] = int((est_local == c).sum())

    print("\n[Split] Conteos por fichero y estado (antes de balancear):")
    for src in sorted(counts_per_file[1].keys() | counts_per_file[2].keys() | counts_per_file[3].keys()):
        c1 = counts_per_file[1].get(src, 0)
        c2 = counts_per_file[2].get(src, 0)
        c3 = counts_per_file[3].get(src, 0)
        print(f" - {os.path.basename(src)}: estado1={c1}, estado2={c2}, estado3={c3}")

    # Umbral por estado usando percentil sobre (estado,pista)
    per_state_target: Dict[int, int] = {}
    for c in [1, 2, 3]:
        arr = np.array([cnt for cnt in counts_per_file[c].values() if cnt > 0], dtype=int)
        if arr.size == 0:
            per_state_target[c] = 0
            continue
        global_min = int(arr.min())
        perc_val = int(np.percentile(arr, state_percentile))
        target = max(global_min, perc_val)
        if target < 1:
            target = 1
        per_state_target[c] = target

    print("\n[Split] Umbrales por estado (basados en percentil sobre (estado,pista)):")
    for c in [1, 2, 3]:
        print(f"  estado {c}: target_per_file ≈ {per_state_target[c]}")

    # Seleccionar filas por fichero y estado
    selected_parts: List[pd.DataFrame] = []

    for src, g in grouped_by_src:
        est_local = pd.to_numeric(g["estado"], errors="coerce").astype("Int64")
        for c in [1, 2, 3]:
            n_c = counts_per_file[c].get(src, 0)
            if n_c == 0:
                continue

            target_c = per_state_target[c]
            if target_c <= 0:
                continue

            # no pedimos más de lo que hay
            target_here = min(target_c, n_c)
            if target_here <= 0:
                continue

            g_state = g[est_local == c]
            if g_state.empty:
                continue

            # 1) aplanar throttle
            seed_flat = seed + (hash((src, c, "flat")) % 10_000)
            g_flat = flatten_by_throttle(
                g_state,
                n_bins=thr_bins,
                perc=thr_percentile,
                seed=seed_flat
            )

            # 2) ajustar a target_here
            if len(g_flat) > target_here:
                seed_sel = seed + (hash((src, c, "cap")) % 10_000)
                g_final = select_rows_mode(g_flat, target_here, keep_mode, seed_sel)
            elif len(g_flat) == target_here:
                g_final = g_flat
            else:
                remaining = target_here - len(g_flat)
                used_idx = set(g_flat.index.tolist())
                candidates = g_state.loc[~g_state.index.isin(used_idx)]
                if not candidates.empty:
                    n_extra = min(remaining, len(candidates))
                    seed_extra = seed + (hash((src, c, "extra")) % 10_000)
                    extra = select_rows_mode(candidates, n_extra, "random", seed_extra)
                    g_final = pd.concat([g_flat, extra], ignore_index=True)
                else:
                    g_final = g_flat

            selected_parts.append(g_final)

    if selected_parts:
        df_selected = pd.concat(selected_parts, ignore_index=True)
    else:
        df_selected = df_123.iloc[0:0].copy()

    # Barajar
    df_selected = df_selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Reinyectar resto si no se quieren eliminar
    if drop_non123:
        final = df_selected
    else:
        final = pd.concat([df_selected, df_rest], ignore_index=True)
        final = final.reset_index(drop=True)

    print(f"[Split] Tras balanceo+aplanamiento: {len(df_selected)} filas 1/2/3, "
          f"total (incluyendo resto) = {len(final)}")
    return final


# -------------------- MAIN -------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Balancea estados y throttle en train/validation/test por separado, "
            "y luego ajusta a un ratio global 70/15/15 (solo downsampling)."
        )
    )
    ap.add_argument("--base-dir", default="../datasets",
                    help="Directorio base de datasets (por defecto ../datasets).")
    ap.add_argument("--state-percentile", type=float, default=25.0,
                    help="Percentil sobre conteos (estado,pista) para umbral por estado (default=25).")
    ap.add_argument("--thr-bins", type=int, default=30,
                    help="Número de bins para throttle en el aplanamiento (default=30).")
    ap.add_argument("--thr-percentile", type=float, default=60.0,
                    help="Percentil de ocupación por bin de throttle (default=60).")
    ap.add_argument("--keep", choices=["first", "last", "random", "stride"],
                    default="random",
                    help="Estrategia final de selección si sobra después de aplanar (default=random).")
    ap.add_argument("--seed", type=int, default=42, help="Semilla base.")
    ap.add_argument("--no-backup", action="store_true",
                    help="No crear .bak de los dataset.csv originales.")
    ap.add_argument("--dry-run", action="store_true",
                    help="No escribir cambios, solo mostrar lo que se haría.")
    ap.add_argument("--drop-non123", action="store_true",
                    help="Eliminar filas con estado fuera de {1,2,3} en vez de preservarlas.")
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    print(f"[INFO] Base dir: {base_dir}")
    print(f"[INFO] state_percentile = {args.state_percentile}")
    print(f"[INFO] thr_bins        = {args.thr_bins}")
    print(f"[INFO] thr_percentile  = {args.thr_percentile}")

    # 1) Cargar splits
    df_train_orig = collect_split_datasets(base_dir, "train")
    df_val_orig   = collect_split_datasets(base_dir, "validation")
    df_test_orig  = collect_split_datasets(base_dir, "test")

    splits_orig = {
        "train": df_train_orig,
        "validation": df_val_orig,
        "test": df_test_orig,
    }

    # 2) Balanceo+aplanamiento por split (sin mezclar datos entre splits)
    df_train_bal = balance_split_states_and_throttle(
        df_train_orig, args.state_percentile, args.thr_bins,
        args.thr_percentile, args.keep, args.seed, args.drop_non123
    ) if df_train_orig is not None else None

    df_val_bal = balance_split_states_and_throttle(
        df_val_orig, args.state_percentile, args.thr_bins,
        args.thr_percentile, args.keep, args.seed + 1, args.drop_non123
    ) if df_val_orig is not None else None

    df_test_bal = balance_split_states_and_throttle(
        df_test_orig, args.state_percentile, args.thr_bins,
        args.thr_percentile, args.keep, args.seed + 2, args.drop_non123
    ) if df_test_orig is not None else None

    splits_bal = {
        "train": df_train_bal,
        "validation": df_val_bal,
        "test": df_test_bal,
    }

    # 3) Calcular tamaños y objetivos 70/15/15
    sizes = {k: (0 if v is None else len(v)) for k, v in splits_bal.items()}
    print("\n[INFO] Tamaños tras balanceo+aplanamiento (antes 70/15/15):")
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
        df_orig = splits_orig[split_name]
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

    print("\n[DONE] Balanceo por estados + aplanamiento throttle + ratio 70/15/15 aplicado in-place.")


if __name__ == "__main__":
    main()
