#!/usr/bin/env python3
import os
import glob
import argparse
import shutil
import pandas as pd
import numpy as np

# Estructura asumida:
# base_dir/
#   Deepracer_BaseMap_*/dataset.csv              -> TRAIN
#   validation/Deepracer_BaseMap_*/dataset.csv   -> VALIDATION
#   test/Deepracer_BaseMap_*/dataset.csv         -> TEST

def process_pattern(pattern: str, name: str, thr_min: float, dry_run: bool, no_backup: bool):
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[WARN] No se encontraron CSV para {name} con patrón: {pattern}")
        return

    print(f"\n=== Procesando {name} ({len(paths)} ficheros) ===")
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[SKIP] No se pudo leer {p}: {e}")
            continue

        if "throttle" not in df.columns:
            print(f"[SKIP] {p} no tiene columna 'throttle'.")
            continue

        # Convertimos a numérico
        thr = pd.to_numeric(df["throttle"], errors="coerce")
        # Nos quedamos solo con throttle >= thr_min
        mask_keep = thr >= thr_min

        before = len(df)
        after = int(mask_keep.sum())
        removed = before - after

        print(f"[{name}] {os.path.basename(p)}: {before} -> {after} filas "
              f"(eliminadas {removed} con throttle < {thr_min})")

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
                print(f"   [WARN] No se pudo crear backup {bak}: {e}")

        # Guardar
        try:
            df_out.to_csv(p, index=False)
        except Exception as e:
            print(f"   [ERROR] Guardando {p}: {e}")


def main():
    ap = argparse.ArgumentParser(
        description="Elimina filas con throttle < umbral en train/validation/test (dataset.csv)."
    )
    ap.add_argument("--base-dir", default="../datasets",
                    help="Directorio base (por defecto ../datasets).")
    ap.add_argument("--thr-min", type=float, default=0.27,
                    help="Umbral mínimo de throttle (se eliminan filas con throttle < thr-min).")
    ap.add_argument("--dry-run", action="store_true",
                    help="No escribe cambios, solo muestra lo que haría.")
    ap.add_argument("--no-backup", action="store_true",
                    help="No crear ficheros .bak antes de sobrescribir.")
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    print(f"[INFO] Base dir: {base_dir}")
    print(f"[INFO] Umbral mínimo throttle: {args.thr_min}")
    print(f"[INFO] dry-run: {args.dry_run}, no-backup: {args.no_backup}")

    # Patrones para train / val / test
    train_pattern = os.path.join(base_dir, "Deepracer_BaseMap_*", "dataset.csv")
    val_pattern   = os.path.join(base_dir, "validation", "Deepracer_BaseMap_*", "dataset.csv")
    test_pattern  = os.path.join(base_dir, "test", "Deepracer_BaseMap_*", "dataset.csv")

    process_pattern(train_pattern, "TRAIN", args.thr_min, args.dry_run, args.no_backup)
    process_pattern(val_pattern,   "VALIDATION", args.thr_min, args.dry_run, args.no_backup)
    process_pattern(test_pattern,  "TEST", args.thr_min, args.dry_run, args.no_backup)

    print("\n[DONE] Proceso completado.")


if __name__ == "__main__":
    main()
