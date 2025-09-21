#!/usr/bin/env python3

#------------------------------------------------
#Codigo que reorganiza el dataset.csv del directorio que se le pase como parametro.
#De esta manera, lee la columna de estado y equilibra el dataset para haber el mismo numero de 1,2 y 3,
#o izquierda centro y derecha.

#Por defecto busca Deepracer_BaseMap_*/dataset.csv 
#------------------------------------------------

import os
import glob
import argparse
import shutil
import pandas as pd


def balance_one_csv(csv_path: str, seed: int, dry_run: bool, backup: bool):
    print(f"\nProcesando: {csv_path}")
    if not os.path.isfile(csv_path):
        print("No existe el archivo.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"No se pudo leer: {e}")
        return

    if 'estado' not in df.columns:
        print("No tiene columna 'estado'.")
        return

    # Convertir 'estado' a numero
    estado = pd.to_numeric(df['estado'], errors='coerce').astype('Int64')

    # Solo existen los estados {1,2,3}
    mask_123 = estado.isin([1, 2, 3])
    df_123 = df[mask_123].copy()
    estado_123 = pd.to_numeric(df_123['estado'], errors='coerce').astype('Int64')

    # Filas (si hubiera) con otros estados o NaN: se mantienen sin tocar
    df_rest = df[~mask_123].copy()

    counts = {c: int((estado_123 == c).sum()) for c in [1, 2, 3]}
    presentes = [c for c in [1, 2, 3] if counts[c] > 0]

    print(f"  Conteo inicial: 1->{counts[1]} | 2->{counts[2]} | 3->{counts[3]}")
    if len(presentes) <= 1:
        print("Solo hay una (o ninguna) clase presente. No se balancea.")
        return

    # Igualar al minimo los 3 estados
    n_target = min(counts[c] for c in presentes)
    print(f"Estados presentes: {presentes} | n_target = {n_target}")

    if dry_run:
        print("No se escribe nada")
        return

    # Recortamos cada estado presente a n_target
    parts = []
    for c in presentes:
        df_c = df_123[estado_123 == c]
        if len(df_c) > n_target:
            df_c = df_c.sample(n=n_target, random_state=seed)
        parts.append(df_c)

    # Unimos balanceado + resto y barajamos
    df_bal = pd.concat(parts + ([df_rest] if not df_rest.empty else []), axis=0)
    df_bal = df_bal.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Copia de seguridad
    if backup:
        shutil.copy2(csv_path, csv_path + ".bak")
        print(f"  Copia de seguridad creada: {csv_path}.bak")

    df_bal.to_csv(csv_path, index=False)

    # Recalcular conteos finales (solo de 1/2/3 para informar)
    est_final = pd.to_numeric(df_bal['estado'], errors='coerce').astype('Int64')
    counts_final = {c: int((est_final == c).sum()) for c in [1, 2, 3]}
    print(f" Guardado. Conteo final (1/2/3): {counts_final[1]} / {counts_final[2]} / {counts_final[3]}")
    if not df_rest.empty:
        print(f"Aviso: {len(df_rest)} filas con 'estado' fuera de {{1,2,3}} se han mantenido sin tocar.")

def main():
    parser = argparse.ArgumentParser(
        description="Balancea dataset.csv en directorios Deepracer_BaseMap_* para igualar el número de estados 1, 2 y 3."
    )
    parser.add_argument(
        "--pattern",
        default="Deepracer_BaseMap_*/dataset.csv",
        help="Patrón de búsqueda de CSV (por defecto: Deepracer_BaseMap_*/dataset.csv)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla para muestreo reproducible.")
    parser.add_argument("--dry-run", action="store_true", help="No escribe cambios, solo muestra lo que haría.")
    parser.add_argument("--no-backup", action="store_true", help="No crear .bak antes de sobrescribir.")
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(args.pattern))
    if not csv_paths:
        print("No se encontraron CSV con el patrón dado.")
        return

    for p in csv_paths:
        balance_one_csv(p, seed=args.seed, dry_run=args.dry_run, backup=not args.no_backup)

if __name__ == "__main__":
    main()
