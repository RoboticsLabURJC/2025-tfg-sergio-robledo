#!/usr/bin/env python3

#------------------------------------
#---Pone en negro las filas [0:rows) de las máscaras de cada dataset.csv encontrado,
#--Realizado para eliminar la parte blanca de las paredes en las mascaras
#------------------------------------


import os
import sys
import argparse
import csv
import pandas as pd
from PIL import Image
import numpy as np

def find_all_datasets(base_dir):

    csv_paths = []
    for root, dirs, files in os.walk(base_dir):
        if "dataset.csv" in files:
            csv_paths.append(os.path.join(root, "dataset.csv"))
    return csv_paths

def get_mask_path_from_row(row, csv_path):

    cols = list(row.index)
    if "mask_path" in cols:
        return row["mask_path"]
    else:
        if len(cols) < 2:
            print(f"{csv_path}: fila sin suficiente columnas para encontrar máscara.")
            return None
        return row[cols[1]]

def process_mask_image(mask_abs_path, rows, dry_run=False):
    
    #Abre la máscara, pone a negro las filas [0:rows) y la guarda.
    #Sobrescribe la imagen original excepto si dry_run=True.
    
    if not os.path.isfile(mask_abs_path):
        print(f"   Mask no encontrada: {mask_abs_path}")
        return

    try:
        img = Image.open(mask_abs_path)
    except Exception as e:
        print(f"   No se pudo abrir {mask_abs_path}: {e}")
        return

    arr = np.array(img)

    if arr.ndim == 2:  # escala de grises
        arr[0:rows, :] = 0
    elif arr.ndim == 3:
        arr[0:rows, :, :] = 0
    else:
        print(f"   Dimensión rara en {mask_abs_path}: shape={arr.shape}")
        return

    if dry_run:
        print(f"   Modificaría: {mask_abs_path}")
        return

    try:
        img_mod = Image.fromarray(arr)
        img_mod.save(mask_abs_path)
    except Exception as e:
        print(f"   Guardando {mask_abs_path}: {e}")

def process_dataset_csv(csv_path, rows, dry_run=False):
    print(f"\nProcesando dataset: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"No se pudo leer {csv_path}: {e}")
        return

    base_dir = os.path.dirname(csv_path)
    n = len(df)
    print(f"   Filas en CSV: {n}")

    for i, row in df.iterrows():
        mask_rel = get_mask_path_from_row(row, csv_path)
        if not isinstance(mask_rel, str) or not mask_rel.strip():
            print(f"   [Fila {i}] Sin mask_path válido, salto.")
            continue

        # Quitar "/" inicial si lo hay
        mask_rel = mask_rel.lstrip("/")
        mask_abs = os.path.join(base_dir, mask_rel)

        print(f"   [Fila {i}] {mask_rel}")
        process_mask_image(mask_abs, rows, dry_run=dry_run)

def parse_args():
    ap = argparse.ArgumentParser(
        description="Pone en negro las filas [0:rows) de las máscaras de cada dataset.csv encontrado."
    )
    ap.add_argument(
        "--base-dir",
        required=True,
        help="Directorio base donde buscar recursivamente dataset.csv"
    )
    ap.add_argument(
        "--rows",
        type=int,
        default=100,
        help="Número de filas desde arriba que se ponen a negro (por defecto 100)"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo mostrar lo que haría, sin modificar imágenes"
    )
    return ap.parse_args()

def main():
    args = parse_args()
    base_dir = os.path.abspath(args.base_dir)

    print(f"Base dir : {base_dir}")
    print(f"Filas a negro desde arriba: 0..{args.rows - 1}")
    print(f"dry-run : {args.dry_run}")

    csv_paths = find_all_datasets(base_dir)
    if not csv_paths:
        print("No se encontraron dataset.csv en ese directorio.")
        sys.exit(0)

    print(f"Encontrados {len(csv_paths)} dataset.csv")

    for csv_path in csv_paths:
        process_dataset_csv(csv_path, rows=args.rows, dry_run=args.dry_run)

    print("\nProceso completado.")

if __name__ == "__main__":
    main()
