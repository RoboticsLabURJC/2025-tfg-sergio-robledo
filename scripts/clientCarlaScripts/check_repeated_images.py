#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
from collections import defaultdict


# -------------------------------------------------
# Buscar todos los dataset.csv

def find_all_datasets(base_dir):
    csv_paths = []
    for root, _, files in os.walk(base_dir):
        if "dataset.csv" in files:
            csv_paths.append(os.path.join(root, "dataset.csv"))
    return csv_paths


# -------------------------------------------------
# Obtener ruta de máscara desde una fila

def get_mask_path_from_row(row, csv_path):
    cols = list(row.index)
    if "mask_path" in cols:
        return row["mask_path"]
    elif len(cols) >= 2:
        return row[cols[1]]
    else:
        return None



def main():
    ap = argparse.ArgumentParser(
        description="Comprueba imágenes repetidas referenciadas desde dataset.csv"
    )
    ap.add_argument("--base-dir", required=True, help="Directorio base del dataset")
    ap.add_argument("--show-all", action="store_true", help="Mostrar también las no repetidas")
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    print(f"Base dir: {base_dir}")

    csv_paths = find_all_datasets(base_dir)
    if not csv_paths:
        print("No se encontraron dataset.csv")
        sys.exit(1)

    print(f"Encontrados {len(csv_paths)} dataset.csv")

    image_counter = defaultdict(list)

    # Recorrer CSVs
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"No se pudo leer {csv_path}: {e}")
            continue

        base_csv_dir = os.path.dirname(csv_path)

        for idx, row in df.iterrows():
            mask_rel = get_mask_path_from_row(row, csv_path)
            if not isinstance(mask_rel, str) or not mask_rel.strip():
                continue

            mask_rel = mask_rel.lstrip("/")
            mask_abs = os.path.abspath(os.path.join(base_csv_dir, mask_rel))

            image_counter[mask_abs].append((csv_path, idx))

    # -------------------------------------------------
    # Mostrar resultados
    repeated = {k: v for k, v in image_counter.items() if len(v) > 1}

    print("\n====================")
    print(f"Total imágenes únicas: {len(image_counter)}")
    print(f"Imágenes repetidas    : {len(repeated)}")

    if not repeated:
        print("\nNo hay imágenes repetidas")
        return

    print("\n[REPETIDAS]")
    for img, refs in repeated.items():
        print(f"\n{img}")
        print(f"  Usada {len(refs)} veces:")
        for csv_path, row_idx in refs:
            print(f"    - {csv_path} (fila {row_idx})")

    if args.show_all:
        print("\n[NO REPETIDAS]")
        for img, refs in image_counter.items():
            if len(refs) == 1:
                csv_path, row_idx = refs[0]
                print(f"{img}  -> {csv_path} (fila {row_idx})")


if __name__ == "__main__":
    main()
