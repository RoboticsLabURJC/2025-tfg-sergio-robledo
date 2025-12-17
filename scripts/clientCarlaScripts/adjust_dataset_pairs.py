#!/usr/bin/env python3
# Balanceo por "circuito" (parejas Deepracer_BaseMap_*):
# 1) Se calculan las cuentas por estado (1/2/3) en TODOS los circuitos.
# 2) Se obtiene un n_target_global = mínimo de todas esas cuentas.
# 3) Cada circuito se balancea usando ese n_target_global.
# 4) El PRIMER CSV de cada pareja “absorbe” al segundo (fusionado).
# 5) Se COPIAN las imágenes de rgb/ y masks/ del segundo al primero.
# 6) Se BORRA el directorio del SEGUNDO de cada pareja (si no es dry-run).

import os
import glob
import argparse
import shutil
import pandas as pd

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

def copy_images_from_second_to_first(second_dir, first_dir):
    """
    Copia los contenidos de:
      second_dir/rgb  -> first_dir/rgb
      second_dir/masks -> first_dir/masks
    Sin machacar ficheros ya existentes.
    """
    for sub in ["rgb", "masks"]:
        src_sub = os.path.join(second_dir, sub)
        dst_sub = os.path.join(first_dir, sub)

        if not os.path.isdir(src_sub):
            continue

        os.makedirs(dst_sub, exist_ok=True)

        for fname in os.listdir(src_sub):
            src_file = os.path.join(src_sub, fname)
            if not os.path.isfile(src_file):
                continue
            dst_file = os.path.join(dst_sub, fname)

            if os.path.exists(dst_file):
                continue

            try:
                shutil.copy2(src_file, dst_file)
            except Exception as e:
                print(f"  [WARN] No se pudo copiar {src_file} -> {dst_file}: {e}")

# ============================================================
# 1) Cálculo del n_target GLOBAL entre todos los circuitos

def compute_global_target(circuits):
    """
    Recorre TODOS los circuitos y devuelve el mínimo conteo por estado (1/2/3)
    entre todos ellos. Ese valor será el n_target_global.
    """
    global_min = None

    for pair in circuits:
        dfs_src = []

        for p in pair:
            df = load_csv_safe(p)
            if df is None:
                dfs_src = []
                break

            if 'estado' not in df.columns:
                print(f"[SKIP] {p} no tiene columna 'estado'.")
                dfs_src = []
                break

            df = df.copy()
            estado = pd.to_numeric(df['estado'], errors='coerce').astype('Int64')
            mask_123 = estado.isin([1, 2, 3])
            df_123 = df[mask_123].copy()
            if not df_123.empty:
                dfs_src.append(df_123)

        if not dfs_src:
            continue

        all_123 = pd.concat(dfs_src, ignore_index=True)
        estado_123 = pd.to_numeric(all_123['estado'], errors='coerce').astype('Int64')

        counts = {c: int((estado_123 == c).sum()) for c in [1, 2, 3]}
        print(f"[PREVIEW] Circuito {pair} -> (1/2/3): {counts[1]}/{counts[2]}/{counts[3]}")

        positivos = [v for v in counts.values() if v > 0]
        if not positivos:
            continue
        local_min = min(positivos)

        if (global_min is None) or (local_min < global_min):
            global_min = local_min

    return global_min

# ============================================================
# 2) Balanceo de un circuito usando n_target_global
#    y guardando SOLO en el primer CSV de la pareja

def balancear_circuito(csv_paths, n_target_global, circuit_id,
                       seed=42, dry_run=False, no_backup=False):
    """
    Balancea un circuito (lista de 2 CSVs) usando un n_target_global común.
    Fusiona ambos ficheros y guarda el resultado en el PRIMER CSV de la pareja
    (como si el primero absorbiera al segundo).
    """
    assert len(csv_paths) == 2, "Cada circuito debe tener 2 CSVs."

    dfs_src = []

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
        dfs_src.append(df_123)

    if not dfs_src or all(d.empty for d in dfs_src):
        print("[WARN] El circuito no tiene filas con estado 1/2/3.")
        return False

    all_123 = pd.concat(dfs_src, ignore_index=True)
    estado_123 = pd.to_numeric(all_123['estado'], errors='coerce').astype('Int64')

    counts = {c: int((estado_123 == c).sum()) for c in [1, 2, 3]}
    presentes = [c for c in [1, 2, 3] if counts[c] > 0]
    print("  Conteos circuito (1/2/3):", counts)

    if len(presentes) <= 1:
        print("  [INFO] Sólo hay una (o ninguna) clase presente en este circuito. No se balancea.")
        return False

    # Intentamos usar el n_target_global
    if any(counts[c] < n_target_global for c in presentes):
        n_target = min(counts[c] for c in presentes)
        print(f"  [WARN] Este circuito no llega a n_target_global={n_target_global}.")
        print(f"        Se usa n_target LOCAL = {n_target}")
    else:
        n_target = n_target_global

    print(f"  n_target aplicado en este circuito = {n_target}")

    # Muestreo estratificado a nivel circuito
    sampled_parts = []
    for c in presentes:
        df_c = all_123[estado_123 == c]
        if len(df_c) > n_target:
            df_c = df_c.sample(n=n_target, random_state=seed)
        sampled_parts.append(df_c)

    df_balanced = pd.concat(sampled_parts, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Eliminamos columnas auxiliares
    cols_out = [col for col in df_balanced.columns if col not in ['_src', '_rowid']]
    df_out = df_balanced[cols_out].copy()

    # Estadísticas
    est_local = pd.to_numeric(df_out['estado'], errors='coerce').astype('Int64')
    c_local = {c: int((est_local == c).sum()) for c in [1, 2, 3]}

    if dry_run:
        print(f"  [DRY-RUN] Circuito {circuit_id} fusionado -> (1/2/3): "
              f"{c_local[1]}/{c_local[2]}/{c_local[3]}  | total={len(df_out)}")
        return True

    # Guardamos en el PRIMER CSV de la pareja
    first_csv = csv_paths[0]
    print(f"  Guardando fusionado en (primer CSV de la pareja): {first_csv}")

    try:
        if (not no_backup) and os.path.isfile(first_csv):
            shutil.copy2(first_csv, first_csv + ".bak")
            print(f"  Copia de seguridad creada: {first_csv}.bak")
    except Exception as e:
        print(f"  [WARN] No se pudo crear .bak para {first_csv}: {e}")

    try:
        df_out.to_csv(first_csv, index=False)
        print(f"  Guardado OK. (1/2/3): {c_local[1]}/{c_local[2]}/{c_local[3]}  | total={len(df_out)}")
    except Exception as e:
        print(f"  [ERROR] Guardando {first_csv}: {e}")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Balancea por circuitos (parejas Deepracer_BaseMap_*) usando un n_target GLOBAL, "
            "fusiona en el primer CSV, copia imágenes del segundo al primero y borra el directorio del segundo."
        )
    )
    parser.add_argument(
        "--pattern",
        default="../datasets/Deepracer_BaseMap_*/dataset.csv",
        help="Patrón de búsqueda de CSV (por defecto: ../datasets/Deepracer_BaseMap_*/dataset.csv)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla para muestreo reproducible.")
    parser.add_argument("--dry-run", action="store_true",
                        help="No escribe cambios ni borra directorios, solo muestra lo que haría.")
    parser.add_argument("--no-backup", action="store_true", help="No crear .bak antes de sobrescribir.")
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(args.pattern))
    if not csv_paths:
        print("No se encontraron CSV con el patrón dado.")
        return

    circuits = pairwise(csv_paths, size=2)
    if not circuits:
        print("No hay parejas completas (circuitos) con el patrón dado.")
        return

    print(f"Se han detectado {len(circuits)} circuito(s).")

    # PASO 1: calcular n_target_global
    n_target_global = compute_global_target(circuits)
    if n_target_global is None:
        print("No se pudo calcular un n_target_global (quizá no hay estados 1/2/3).")
        return

    print(f"\n[GLOBAL] n_target_global (para TODOS los circuitos y estados) = {n_target_global}\n")

    # PASO 2: balancear cada circuito con ese n_target_global
    total_ok = 0
    for ci, pair in enumerate(circuits, start=1):
        print("\n===== Circuito", ci, "=====")
        for p in pair:
            print("  -", p)

        ok = balancear_circuito(
            pair,
            n_target_global,
            circuit_id=ci,
            seed=args.seed,
            dry_run=args.dry_run,
            no_backup=args.no_backup
        )
        if ok:
            total_ok += 1

            if not args.dry_run:
                first_csv  = pair[0]
                second_csv = pair[1]
                first_dir  = os.path.dirname(first_csv)
                second_dir = os.path.dirname(second_csv)

                # 1) Copiar imágenes de second_dir -> first_dir
                print(f"  [COPY] Copiando imágenes de {second_dir} -> {first_dir}")
                copy_images_from_second_to_first(second_dir, first_dir)

                # 2) Borrar el directorio del segundo de la pareja
                if os.path.isdir(second_dir):
                    print(f"  [CLEANUP] Borrando directorio del segundo de la pareja: {second_dir}")
                    try:
                        shutil.rmtree(second_dir)
                    except Exception as e:
                        print(f"  [WARN] No se pudo borrar {second_dir}: {e}")

    print(f"\nResumen: {total_ok}/{len(circuits)} circuitos procesados correctamente"
          f"{' (dry-run)' if args.dry_run else ''}.")

if __name__ == "__main__":
    main()
