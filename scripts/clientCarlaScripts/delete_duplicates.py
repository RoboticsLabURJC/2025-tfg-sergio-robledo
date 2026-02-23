#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd


def find_all_datasets(base_dir):
    csvs = []
    for root, _, files in os.walk(base_dir):
        if "dataset.csv" in files:
            csvs.append(os.path.join(root, "dataset.csv"))
    return csvs


def dedupe_dataset_csv(csv_path, dry_run=False):
    print(f"\n=== Procesando {csv_path} ===")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[SKIP] No se pudo leer: {e}")
        return

    if "mask_path" not in df.columns:
        print("[SKIP] No existe columna mask_path")
        return

    n_before = len(df)

    df_clean = df.drop_duplicates(
        subset=["mask_path"], 
        keep="first"
    )

    n_after = len(df_clean)
    removed = n_before - n_after

    if removed == 0:
        print("No hay duplicados")
        return

    print(f"Eliminadas {removed} filas duplicadas")

    if dry_run:
        print("No se escribe el archivo")
        return

    # Backup
    bak = csv_path + ".bak"
    if not os.path.exists(bak):
        os.rename(csv_path, bak)
        print(f"Backup creado: {bak}")

    df_clean.to_csv(csv_path, index=False)
    print("dataset.csv actualizado")


def main():
    ap = argparse.ArgumentParser(
        description="Elimina filas duplicadas por imagen (mask_path) en todos los dataset.csv"
    )
    ap.add_argument("--base-dir", required=True, help="Directorio base de datasets")
    ap.add_argument("--dry-run", action="store_true", help="Solo mostrar, no borrar nada")
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    csvs = find_all_datasets(base_dir)

    if not csvs:
        print("No se encontraron dataset.csv")
        sys.exit(0)

    print(f"Encontrados {len(csvs)} dataset.csv")

    for csv in csvs:
        dedupe_dataset_csv(csv, dry_run=args.dry_run)

    print("\nProceso completado")


if __name__ == "__main__":
    main()
