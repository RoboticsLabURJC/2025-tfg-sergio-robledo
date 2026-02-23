#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
from PIL import Image
import numpy as np

#------------------------------------
#---Pone en negro las filas [0:100) de las máscaras de cada dataset.csv encontrado,
#--Realizado para eliminar la parte blanca de las paredes en las mascaras.
#--Ademas añade 200 filas negras, asi la imagen pasa a ser 800x800 
#------------------------------------


# -------------------------------------------------
# Buscar dataset.csv

def find_all_datasets(base_dir):
    csv_paths = []
    for root, _, files in os.walk(base_dir):
        if "dataset.csv" in files:
            csv_paths.append(os.path.join(root, "dataset.csv"))
    return csv_paths


# -------------------------------------------------
# Obtener ruta de máscara

def get_mask_path_from_row(row, csv_path):
    cols = list(row.index)
    if "mask_path" in cols:
        return row["mask_path"]
    elif len(cols) >= 2:
        return row[cols[1]]
    else:
        print(f"[WARN] {csv_path}: fila sin columnas suficientes")
        return None


# -------------------------------------------------
# PROCESAR IMAGEN

def process_mask_image(mask_abs_path, dry_run=False):

    #PASO 1: pone a negro las primeras 100 filas de la imagen original
    #PASO 2: añade 200 filas negras arriba
    #RESULTADO FINAL: debe ser 800x800 o el programa termina
 
    if not os.path.isfile(mask_abs_path):
        print(f"No existe: {mask_abs_path}")
        sys.exit(1)

    try:
        img = Image.open(mask_abs_path)
    except Exception as e:
        print(f"No se pudo abrir {mask_abs_path}: {e}")
        sys.exit(1)

    arr = np.array(img)
    h, w = arr.shape[:2]

    # Comprobación inicial
    if (w, h) == (800, 800):
        print(f"Ya procesada (800x800): {mask_abs_path}")
        return

    if (w, h) != (800, 600):
        print(f"Tamaño inesperado: {mask_abs_path} ({w}x{h})")
        sys.exit(1)


    # -------------------------
    # PASO 1: negro filas 0:100

    if arr.ndim == 2:
        arr[0:100, :] = 0
    elif arr.ndim == 3:
        arr[0:100, :, :] = 0
    else:
        print(f"Dimensión no soportada: {arr.shape}")
        sys.exit(1)

    # -------------------------
    # PASO 2: añadir 200 filas arriba
  
    if arr.ndim == 2:
        new_arr = np.zeros((h + 200, w), dtype=arr.dtype)
        new_arr[200:, :] = arr
    else:
        c = arr.shape[2]
        new_arr = np.zeros((h + 200, w, c), dtype=arr.dtype)
        new_arr[200:, :, :] = arr

    final_h, final_w = new_arr.shape[:2]

    # -------------------------
    # COMPROBACIÓN FINAL

    if (final_w, final_h) != (800, 800):
        print(f"Resultado final no es 800x800: {final_w}x{final_h}")
        sys.exit(1)

    if dry_run:
        print(f"{mask_abs_path}: 800x600 → 800x800")
        return

    try:
        Image.fromarray(new_arr).save(mask_abs_path)

    except Exception as e:
        print(f"Guardando {mask_abs_path}: {e}")
        sys.exit(1)


# -------------------------------------------------
# Procesar CSV

def process_dataset_csv(csv_path, dry_run=False):
    print(f"\n=== Procesando {csv_path} ===")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"No se pudo leer CSV: {e}")
        sys.exit(1)

    base_dir = os.path.dirname(csv_path)

    for i, row in df.iterrows():
        mask_rel = get_mask_path_from_row(row, csv_path)
        if not isinstance(mask_rel, str) or not mask_rel.strip():
            print(f"[Fila {i} sin mask_path válido")
            sys.exit(1)

        mask_rel = mask_rel.lstrip("/")
        mask_abs = os.path.join(base_dir, mask_rel)

        process_mask_image(mask_abs, dry_run=dry_run)


# -------------------------------------------------
# Args

def parse_args():
    ap = argparse.ArgumentParser(
        description="Negrea 100 filas superiores y añade 200 filas negras arriba. Resultado obligatorio 800x800."
    )
    ap.add_argument("--base-dir", required=True, help="Directorio base")
    ap.add_argument("--dry-run", action="store_true", help="No modificar imágenes")
    return ap.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.abspath(args.base_dir)

    print(f"Base dir : {base_dir}")
    print("Proceso: negro 100 filas + padding 200 filas")
    print(f"Dry-run : {args.dry_run}")

    csv_paths = find_all_datasets(base_dir)
    if not csv_paths:
        print("No se encontraron dataset.csv")
        sys.exit(1)

    for csv_path in csv_paths:
        process_dataset_csv(csv_path, dry_run=args.dry_run)

    print("\nProceso completado correctamente")


if __name__ == "__main__":
    main()
