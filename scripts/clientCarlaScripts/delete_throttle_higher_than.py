#!/usr/bin/env python3
import os
import glob
import argparse
import shutil
import pandas as pd
import numpy as np

# Estructura:
# base_dir/
#   Deepracer_BaseMap_*/dataset.csv              -> TRAIN
#   validation/Deepracer_BaseMap_*/dataset.csv   -> VALIDATION
#   test/Deepracer_BaseMap_*/dataset.csv         -> TEST

def process_pattern(pattern: str, name: str, thr_max: float, dry_run: bool, no_backup: bool):
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No se encontraron CSV para {name} con patrón: {pattern}")
        return

    print(f"\n=== Procesando {name} ({len(paths)} ficheros) ===")
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"No se pudo leer {p}: {e}")
            continue

        if "throttle" not in df.columns:
            print(f"{p} no tiene columna 'throttle'.")
            continue

        thr = pd.to_numeric(df["throttle"], errors="coerce")
        mask_keep = thr <= thr_max

        before = len(df)
        after = int(mask_keep.sum())
        removed = before - after

        print(f"[{name}] {os.path.basename(p)}: {before} -> {after} filas (eliminadas {removed} con throttle > {thr_max})")

        if dry_run:
            continue

        df_out = df.loc[mask_keep].reset_index(drop=True)

        # Backup
        if not no_backup and os.path.isfile(p):
            bak = p + ".bak"
            try:
                shutil.copy2(p, bak)
                print(f"   backup creado: {bak}")
            except Exception as e:
                print(f"   No se pudo crear backup {bak}: {e}")

        # Guardar
        try:
            df_out.to_csv(p, index=False)
        except Exception as e:
            print(f"   Guardando {p}: {e}")


def main():
    ap = argparse.ArgumentParser(
        description="Elimina filas con throttle > umbral en train/validation/test (dataset.csv)."
    )
    ap.add_argument("--base-dir", default="../datasets",
                    help="Directorio base (por defecto ../datasets).")
    ap.add_argument("--thr-max", type=float, default=0.95,
                    help="Umbral máximo de throttle (se eliminan filas con throttle > thr-max).")
    ap.add_argument("--dry-run", action="store_true",
                    help="No escribe cambios, solo muestra lo que haría.")
    ap.add_argument("--no-backup", action="store_true",
                    help="No crear ficheros .bak antes de sobrescribir.")
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    print(f"Base dir: {base_dir}")
    print(f"Umbral throttle: {args.thr_max}")
    print(f"dry-run: {args.dry_run}, no-backup: {args.no_backup}")

    # Patrones para train / val / test
    train_pattern = os.path.join(base_dir, "Deepracer_BaseMap_*", "dataset.csv")
    val_pattern   = os.path.join(base_dir, "validation", "Deepracer_BaseMap_*", "dataset.csv")
    test_pattern  = os.path.join(base_dir, "test", "Deepracer_BaseMap_*", "dataset.csv")

    process_pattern(train_pattern, "TRAIN", args.thr_max, args.dry_run, args.no_backup)
    process_pattern(val_pattern,   "VALIDATION", args.thr_max, args.dry_run, args.no_backup)
    process_pattern(test_pattern,  "TEST", args.thr_max, args.dry_run, args.no_backup)

    print("\nProceso completado.")


if __name__ == "__main__":
    main()
