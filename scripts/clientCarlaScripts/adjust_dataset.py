#!/usr/bin/env python3
# Balanceo GLOBAL de estados (1/2/3) sobre varios dataset.csv y reescritura por archivo.

import os
import glob
import argparse
import shutil
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Balancea GLOBALMENTE estados 1/2/3 entre varios dataset.csv y reescribe cada archivo con su porción balanceada."
    )
    parser.add_argument(
        "--pattern",
        default="../datasets/Deepracer_BaseMap_*/dataset.csv",
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

    #  1) Cargar todo, marcando origen y preservando filas fuera de {1,2,3} por archivo
    dfs_all = []
    rest_by_file = {}  # filas fuera de {1,2,3} o NaN que se dejan intactas por archivo
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[SKIP] No se pudo leer {p}: {e}")
            continue

        if 'estado' not in df.columns:
            print(f"[SKIP] {p} no tiene columna 'estado'.")
            continue

        df['_src'] = p
        # Guardamos un id interno para rastrear filas únicas dentro de su archivo
        df['_rowid'] = range(len(df))

        estado = pd.to_numeric(df['estado'], errors='coerce').astype('Int64')
        mask_123 = estado.isin([1, 2, 3])
        df_123 = df[mask_123].copy()
        df_rest = df[~mask_123].copy()
        dfs_all.append(df_123)

        rest_by_file.setdefault(p, pd.DataFrame(columns=df.columns))
        if not df_rest.empty:
            rest_by_file[p] = pd.concat([rest_by_file[p], df_rest], ignore_index=True)

    if not dfs_all:
        print("No hay datos con estados 1/2/3 en los CSV encontrados.")
        return

    all_123 = pd.concat(dfs_all, ignore_index=True)
    estado_123 = pd.to_numeric(all_123['estado'], errors='coerce').astype('Int64')

    #  2) Conteos globales y target
    counts_global = {c: int((estado_123 == c).sum()) for c in [1, 2, 3]}
    presentes = [c for c in [1, 2, 3] if counts_global[c] > 0]

    print("== Conteo GLOBAL inicial ==")
    print(f"  1 -> {counts_global[1]} | 2 -> {counts_global[2]} | 3 -> {counts_global[3]}")
    if len(presentes) <= 1:
        print("Solo hay una (o ninguna) clase presente globalmente. No se balancea.")
        return

    n_target = min(counts_global[c] for c in presentes)
    print(f"Clases presentes globalmente: {presentes} | n_target = {n_target} por clase")

    #  3) Muestreo global estratificado
    sampled_parts = []
    for c in presentes:
        df_c = all_123[estado_123 == c]
        if len(df_c) > n_target:
            df_c = df_c.sample(n=n_target, random_state=args.seed)
        sampled_parts.append(df_c)

    df_global_bal = pd.concat(sampled_parts, ignore_index=True)

    # Barajar globalmente
    df_global_bal = df_global_bal.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Conteo final global
    est_fin = pd.to_numeric(df_global_bal['estado'], errors='coerce').astype('Int64')
    counts_fin_global = {c: int((est_fin == c).sum()) for c in [1, 2, 3]}
    print("== Conteo GLOBAL balanceado (solo 1/2/3) ==")
    print(f"  1 -> {counts_fin_global[1]} | 2 -> {counts_fin_global[2]} | 3 -> {counts_fin_global[3]}")

    if args.dry_run:
        print("[DRY-RUN] No se escribirá ningún archivo.")
        return

    #  4) Repartir el subset global por archivo y reescribir cada CSV
    # Para cada archivo, tomamos sus filas seleccionadas (por _src) + su df_rest (si hay) y guardamos.
    grouped = df_global_bal.groupby('_src', dropna=False)

    for p in csv_paths:
        # Filas balanceadas que pertenecen a este archivo
        df_sel = grouped.get_group(p).copy() if p in grouped.groups else pd.DataFrame(columns=all_123.columns)

        # Rest a reinyectar
        df_rest = rest_by_file.get(p, pd.DataFrame(columns=df_sel.columns))

        # Unir balanceado (1/2/3) + resto (otros estados)
        cols_out = [col for col in (df_sel.columns if not df_sel.empty else df_rest.columns) if col not in ['_src', '_rowid']]
        df_out = pd.concat([df_sel[cols_out], df_rest[cols_out]], ignore_index=True)

        # Barajar por archivo
        df_out = df_out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        # Copia de seguridad
        try:
            if not args.no_backup and os.path.isfile(p):
                shutil.copy2(p, p + ".bak")
                print(f"[{os.path.basename(p)}] Copia de seguridad creada: {p}.bak")
        except Exception as e:
            print(f"[{os.path.basename(p)}] WARN: no se pudo crear .bak: {e}")

        # Guardar
        try:
            df_out.to_csv(p, index=False)
            # Informe por archivo
            est_local = pd.to_numeric(df_out['estado'], errors='coerce').astype('Int64')
            c_local = {c: int((est_local == c).sum()) for c in [1, 2, 3]}
            print(f"[{os.path.basename(p)}] Guardado. (1/2/3): {c_local[1]} / {c_local[2]} / {c_local[3]}  | total={len(df_out)}")
        except Exception as e:
            print(f"[{os.path.basename(p)}] ERROR al guardar: {e}")

if __name__ == "__main__":
    main()
