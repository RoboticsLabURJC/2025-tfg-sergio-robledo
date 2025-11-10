#!/usr/bin/env python3
# Recorta cada dataset.csv hasta dejar exactamente N filas.

import os
import glob
import argparse
import shutil
import pandas as pd
import numpy as np

def select_rows(df: pd.DataFrame, n: int, mode: str, seed: int) -> pd.DataFrame:
    total = len(df)
    if total <= n:
        # Ya tiene <=N, lo dejamos igual
        return df.copy()

    if mode == "first":
        return df.iloc[:n].copy()

    if mode == "last":
        return df.iloc[-n:].copy().reset_index(drop=True)

    if mode == "random":
        return df.sample(n=n, random_state=seed).sort_index().reset_index(drop=True)

    if mode == "stride":
        # Muestreo lo más uniforme posible
        idx = np.linspace(0, total - 1, n)
        idx = np.round(idx).astype(int)
        # Por si hay duplicados:
        idx = np.unique(idx)[:n]
        return df.iloc[idx].reset_index(drop=True)

    return df.iloc[:n].copy()

def main():
    ap = argparse.ArgumentParser(description="Recorta dataset.csv para dejar exactamente N filas.")
    ap.add_argument("--pattern",
                    default="../datasets/validation/Deepracer_BaseMap_*/dataset.csv",
                    help="Patrón de búsqueda de CSV (por defecto: ../datasets/Deepracer_BaseMap_*/dataset.csv)")
    ap.add_argument("--target", type=int, default=620,
                    help="Número de filas objetivo por CSV (por defecto: 300)")
    ap.add_argument("--keep", choices=["first", "last", "random", "stride"], default="first",
                    help="Estrategia de selección si hay más filas que target. first|last|random|stride")
    ap.add_argument("--seed", type=int, default=42, help="Semilla cuando --keep random")
    ap.add_argument("--dry-run", action="store_true", help="No escribe cambios, solo muestra acciones.")
    ap.add_argument("--no-backup", action="store_true", help="No crear copia .bak antes de sobrescribir.")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        print("No se encontraron CSVs con el patrón dado.")
        return

    print(f"Objetivo: {args.target} filas | keep={args.keep} | archivos={len(paths)}\n")

    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[SKIP] {p} no se pudo leer: {e}")
            continue

        total = len(df)
        if total == 0:
            print(f"[{os.path.basename(p)}] vacío. Se deja igual.")
            continue

        df_new = select_rows(df, args.target, args.keep, args.seed)
        new_total = len(df_new)

        if total == new_total and total <= args.target:
            print(f"[{os.path.basename(p)}] {total} filas (<= {args.target}). Sin cambios.")
            continue

        print(f"[{os.path.basename(p)}] {total} -> {new_total} filas (manteniendo {args.keep}).")

        if args.dry_run:
            continue

        # Copia de seguridad
        try:
            if not args.no_backup and os.path.isfile(p):
                shutil.copy2(p, p + ".bak")
                print(f"  Copia de seguridad creada: {p}.bak")
        except Exception as e:
            print(f"  WARN: no se pudo crear .bak: {e}")

        # Guardar
        try:
            df_new.to_csv(p, index=False)
            print("  Guardado.")
        except Exception as e:
            print(f"  ERROR al guardar: {e}")

if __name__ == "__main__":
    main()
