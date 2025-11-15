#!/usr/bin/env python3
# Balanceo por "circuito" (parejas Deepracer_BaseMap_*):
# Para cada circuito (2 carpetas), se concatena, se calcula el mínimo conteo por estado (1/2/3)
# y se muestrean n_target por estado. Después se reparte el subset resultante a cada CSV

import os
import glob
import argparse
import shutil
import pandas as pd
from collections import defaultdict

def load_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[SKIP] No se pudo leer {path}: {e}")
        return None

def pairwise(lst, size=2):
    """Agrupa una lista en bloques consecutivos de `size` (2 por defecto).
       Si el último grupo queda incompleto, se ignora y se avisa."""
    pairs = []
    for i in range(0, len(lst), size):
        group = lst[i:i+size]
        if len(group) == size:
            pairs.append(group)
        else:
            print(f"[WARN] Grupo incompleto (se ignora): {group}")
    return pairs

def balancear_circuito(csv_paths, seed=42, dry_run=False, no_backup=False):
    """
    Balancea un circuito (lista de 2 CSVs). Devuelve True si ambos se guardaron,
    False si algo salió mal.
    """
    assert len(csv_paths) == 2, "Cada circuito debe tener 2 CSVs."

    # Cargar y anotar origen
    dfs_src = []
    rest_by_file = {}
    for p in csv_paths:
        df = load_csv_safe(p)
        if df is None:
            return False

        if 'estado' not in df.columns:
            print(f"[SKIP] {p} no tiene columna 'estado'.")
            return False

        df = df.copy()
        df['_src'] = p
        df['_rowid'] = range(len(df))

        estado = pd.to_numeric(df['estado'], errors='coerce').astype('Int64')
        mask_123 = estado.isin([1, 2, 3])

        df_123 = df[mask_123].copy()
        df_rest = df[~mask_123].copy()

        dfs_src.append(df_123)

        # Guardamos el "resto" por archivo para reinyectarlo tal cual
        rest_by_file[p] = df_rest if not df_rest.empty else pd.DataFrame(columns=df.columns)

    # Concatenar ambos para balancear a nivel circuito
    if not dfs_src or all(d.empty for d in dfs_src):
        print("[WARN] El circuito no tiene filas con estado 1/2/3.")
        return False

    all_123 = pd.concat(dfs_src, ignore_index=True)
    estado_123 = pd.to_numeric(all_123['estado'], errors='coerce').astype('Int64')

    # Conteos por estado a nivel circuito
    counts = {c: int((estado_123 == c).sum()) for c in [1, 2, 3]}
    presentes = [c for c in [1, 2, 3] if counts[c] > 0]
    print("  Conteos circuito (1/2/3):", counts)

    if len(presentes) <= 1:
        print("  [INFO] Sólo hay una (o ninguna) clase presente en este circuito. No se balancea.")
        return False

    n_target = min(counts[c] for c in presentes)
    print(f"  n_target por estado en este circuito = {n_target}")

    # Muestreo estratificado a nivel circuito
    sampled_parts = []
    for c in presentes:
        df_c = all_123[estado_123 == c]
        if len(df_c) > n_target:
            df_c = df_c.sample(n=n_target, random_state=seed)
        sampled_parts.append(df_c)

    df_balanced = pd.concat(sampled_parts, ignore_index=True)
    # Barajar globalmente dentro del circuito
    df_balanced = df_balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Repartir el subset por archivo y reinyectar "resto"
    grouped = df_balanced.groupby('_src', dropna=False)
    ok = True

    for p in csv_paths:
        # Filas balanceadas que pertenecen a este archivo (puede no haber si todo el muestreo cayó en el otro)
        if p in grouped.groups:
            df_sel = grouped.get_group(p).copy()
        else:
            df_sel = pd.DataFrame(columns=all_123.columns)

        df_rest = rest_by_file.get(p, pd.DataFrame(columns=df_sel.columns))

        cols_out = [col for col in (df_sel.columns if not df_sel.empty else df_rest.columns) if col not in ['_src', '_rowid']]
        df_out = pd.concat([df_sel[cols_out], df_rest[cols_out]], ignore_index=True)

        # Mezclamos para no dejar el orden por bloques
        df_out = df_out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        if dry_run:
            est_local = pd.to_numeric(df_out['estado'], errors='coerce').astype('Int64')
            c_local = {c: int((est_local == c).sum()) for c in [1, 2, 3]}
            print(f"  [DRY-RUN] {os.path.basename(p)} -> (1/2/3): {c_local[1]}/{c_local[2]}/{c_local[3]}  | total={len(df_out)}")
            continue

        try:
            if not no_backup and os.path.isfile(p):
                shutil.copy2(p, p + ".bak")
                print(f"  Copia de seguridad: {p}.bak")
        except Exception as e:
            print(f"  [WARN] No se pudo crear .bak para {p}: {e}")

        try:
            df_out.to_csv(p, index=False)
            est_local = pd.to_numeric(df_out['estado'], errors='coerce').astype('Int64')
            c_local = {c: int((est_local == c).sum()) for c in [1, 2, 3]}
            print(f"  Guardado {os.path.basename(p)}  (1/2/3): {c_local[1]}/{c_local[2]}/{c_local[3]}  | total={len(df_out)}")
        except Exception as e:
            print(f"  [ERROR] Guardando {p}: {e}")
            ok = False

    return ok

def main():
    parser = argparse.ArgumentParser(
        description="Balancea por circuitos (parejas Deepracer_BaseMap_*) usando el mínimo conteo por estado (1/2/3) de cada circuito."
    )
    parser.add_argument(
        "--pattern",
        default="../datasets/validation/Deepracer_BaseMap_*/dataset.csv",
        help="Patrón de búsqueda de CSV (por defecto: ../datasets/Deepracer_BaseMap_*/dataset.csv)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla para muestreo reproducible.")
    parser.add_argument("--dry-run", action="store_true", help="No escribe cambios, solo muestra lo que haría.")
    parser.add_argument("--no-backup", action="store_true", help="No crear .bak antes de sobrescribir.")
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(args.pattern))
    if not csv_paths:
        print("No se encontraron CSV con el patrón dado.")
        return

    # AGRUPAR DE DOS EN DOS = CIRCUITOS
    circuits = pairwise(csv_paths, size=2)
    if not circuits:
        print("No hay parejas completas (circuitos) con el patrón dado.")
        return

    print(f"Se han detectado {len(circuits)} circuito(s).")
    total_ok = 0
    for ci, pair in enumerate(circuits, start=1):
        print("\n===== Circuito", ci, "=====")
        for p in pair:
            print("  -", p)
        ok = balancear_circuito(pair, seed=args.seed, dry_run=args.dry_run, no_backup=args.no_backup)
        if ok:
            total_ok += 1

    print(f"\nResumen: {total_ok}/{len(circuits)} circuitos procesados correctamente{' (dry-run)' if args.dry_run else ''}.")

if __name__ == "__main__":
    main()
