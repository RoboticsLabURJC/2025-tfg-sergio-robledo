#!/usr/bin/env python3
# Recorte GLOBAL: deja un total objetivo repartido uniformemente por estado (1/2/3)
# y reescribe cada dataset.csv con su porción correspondiente.

import os
import glob
import argparse
import shutil
import pandas as pd
import numpy as np
from typing import Dict, List

def select_rows_mode(df: pd.DataFrame, n: int, mode: str, seed: int) -> pd.DataFrame:
    """Selecciona n filas del DataFrame df según el modo."""
    total = len(df)
    if n <= 0:
        return df.iloc[0:0].copy()
    if total <= n:
        return df.copy()

    if mode == "first":
        return df.iloc[:n].copy()

    if mode == "last":
        return df.iloc[-n:].copy().reset_index(drop=True)

    if mode == "random":
        return df.sample(n=n, random_state=seed).sort_index().reset_index(drop=True)

    if mode == "stride":
        idx = np.linspace(0, total - 1, n)
        idx = np.round(idx).astype(int)
        idx = np.unique(idx)[:n]
        return df.iloc[idx].reset_index(drop=True)

    # por defecto, first
    return df.iloc[:n].copy()

def compute_uniform_quotas(counts: Dict[int, int], target_total: int) -> Dict[int, int]:
    """
    Reparte target_total uniformemente entre los estados presentes (1/2/3).
    Si algún estado no tiene suficientes filas, se capea y se reparte el sobrante.
    """
    present_states = [c for c in [1,2,3] if counts.get(c, 0) > 0]
    if not present_states:
        return {1:0, 2:0, 3:0}

    base = target_total // len(present_states)
    rem  = target_total %  len(present_states)

    # Asignación inicial uniforme
    quotas = {c: base for c in present_states}

    # Reparto del resto (1 en 1)
    for c in present_states[:rem]:
        quotas[c] += 1

    # Cap por disponibilidad y redistribución del sobrante
    exhausted = True
    while exhausted:
        exhausted = False
        surplus = 0
        for c in list(quotas.keys()):
            cap = counts.get(c, 0)
            if quotas[c] > cap:
                surplus += quotas[c] - cap
                quotas[c] = cap
                exhausted = True
        if surplus > 0:
            # Repartir sobrante entre estados con hueco
            receivers = [c for c in quotas if quotas[c] < counts.get(c, 0)]
            if not receivers:
                # No hay dónde poner el sobrante
                break
            i = 0
            while surplus > 0 and receivers:
                c = receivers[i % len(receivers)]
                if quotas[c] < counts[c]:
                    quotas[c] += 1
                    surplus -= 1
                i += 1

    # Asegura claves 1/2/3
    for c in [1,2,3]:
        quotas.setdefault(c, 0)
    return quotas

def main():
    ap = argparse.ArgumentParser(
        description="Recorta GLOBALMENTE (sobre todos los CSV) hasta un total objetivo, uniforme por estados 1/2/3, y reescribe cada dataset.csv con su porción."
    )
    ap.add_argument("--pattern",
                    default="../datasets/validation/Deepracer_BaseMap_*/dataset.csv",
                    help="Patrón de búsqueda (por defecto: ../datasets/validation/Deepracer_BaseMap_*/dataset.csv)")
    ap.add_argument("--target-total", type=int, default=1400,
                    help="Número total de filas objetivo (global, sumando todos los CSV).")
    ap.add_argument("--keep", choices=["first", "last", "random", "stride"], default="first",
                    help="Estrategia de selección dentro de cada estado global: first|last|random|stride")
    ap.add_argument("--seed", type=int, default=42, help="Semilla cuando --keep=random")
    ap.add_argument("--dry-run", action="store_true", help="No escribe cambios, solo muestra acciones.")
    ap.add_argument("--no-backup", action="store_true", help="No crear copia .bak antes de sobrescribir.")
    ap.add_argument("--drop-non123", action="store_true",
                    help="Si se indica, se eliminan filas con estado fuera de {1,2,3}. Por defecto se PRESERVAN.")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        print("No se encontraron CSVs con el patrón dado.")
        return

    # Cargar todos con metadatos de origen
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[SKIP] {p} no se pudo leer: {e}")
            continue
        if df.empty:
            print(f"[WARN] {p} vacío.")
            # aún así lo tendremos en cuenta para escribir luego (posible vacío final)
        df["_src"] = p
        df["_rowid"] = range(len(df))
        dfs.append(df)

    if not dfs:
        print("No hay datos legibles.")
        return

    full = pd.concat(dfs, ignore_index=True)

    if "estado" not in full.columns:
        print("[ERROR] Ningún CSV contiene columna 'estado'.")
        return

    # Convertir estado a Int y separar 1/2/3
    estado = pd.to_numeric(full["estado"], errors="coerce").astype("Int64")
    mask_123 = estado.isin([1,2,3])

    df_123 = full[mask_123].copy()
    df_rest = full[~mask_123].copy()  # para preservar si no se usa --drop-non123

    # Conteos globales por estado
    counts = {c: int((pd.to_numeric(df_123["estado"], errors="coerce") == c).sum()) for c in [1,2,3]}
    total_disponible = sum(counts.values())
    print("Conteos globales (1/2/3):", counts, "| total 1/2/3 =", total_disponible)

    if total_disponible == 0:
        print("[WARN] No hay filas con estado 1/2/3. Nada que balancear.")
        return

    # Ajustar target_total si es mayor que el total disponible
    target_total = min(args.target_total, total_disponible)
    if target_total < args.target_total:
        print(f"[INFO] target_total ajustado a {target_total} (no hay suficientes filas 1/2/3).")

    # Calcular cuotas uniformes por estado
    quotas = compute_uniform_quotas(counts, target_total)
    print("Cuotas objetivo por estado:", quotas, "| suma =", sum(quotas.values()))

    # Seleccionar filas por estado
    df_selected_parts: List[pd.DataFrame] = []
    for c in [1,2,3]:
        n_c = quotas.get(c, 0)
        if n_c <= 0:
            continue
        df_c = df_123[pd.to_numeric(df_123["estado"], errors="coerce") == c]

        # Modo de selección global (entre todos los ficheros) para este estado
        df_sel_c = select_rows_mode(df_c, n_c, args.keep, args.seed)
        df_selected_parts.append(df_sel_c)

    if df_selected_parts:
        df_selected = pd.concat(df_selected_parts, ignore_index=True)
    else:
        df_selected = df_123.iloc[0:0].copy()

    # Barajar globalmente (opcional) para no sesgar por orden
    df_selected = df_selected.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Si no se pide eliminar non123, los preservamos íntegros
    if args.drop_non123:
        final = df_selected
    else:
        final = pd.concat([df_selected, df_rest], ignore_index=True)
        # Mantener resto tal cual (opcionalmente, podrías barajar)
        final = final.reset_index(drop=True)

    # Ahora repartimos por archivo de origen y guardamos
    grouped = final.groupby("_src", dropna=False)

    print("\nResumen por archivo (previo a escribir):")
    for p in paths:
        if p in grouped.groups:
            sub = grouped.get_group(p)
            est = pd.to_numeric(sub.get("estado", pd.Series(dtype="float64")), errors="coerce").astype("Int64")
            c1 = int((est == 1).sum()); c2 = int((est == 2).sum()); c3 = int((est == 3).sum())
            print(f" - {os.path.basename(p)}: total={len(sub)} | (1/2/3)= {c1}/{c2}/{c3}")
        else:
            print(f" - {os.path.basename(p)}: total=0 | (1/2/3)= 0/0/0")

    if args.dry_run:
        print("\n[DRY-RUN] No se escribirá ningún archivo.")
        return

    # Escribir cada archivo
    for p in paths:
        if p in grouped.groups:
            out = grouped.get_group(p).copy()
        else:
            # Si no quedó nada para este archivo, escribimos CSV vacío con las mismas columnas que tenía originalmente
            try:
                orig = pd.read_csv(p)
                out = orig.iloc[0:0].copy()
            except Exception:
                # fallback
                out = pd.DataFrame()

        # Quitar columnas auxiliares
        for aux in ["_src", "_rowid"]:
            if aux in out.columns:
                out = out.drop(columns=[aux])

        # Copia de seguridad
        try:
            if not args.no_backup and os.path.isfile(p):
                shutil.copy2(p, p + ".bak")
                print(f"[{os.path.basename(p)}] Copia .bak creada.")
        except Exception as e:
            print(f"[{os.path.basename(p)}] WARN: no se pudo crear .bak: {e}")

        # Guardar
        try:
            out.to_csv(p, index=False)
            print(f"[{os.path.basename(p)}] Guardado ({len(out)} filas).")
        except Exception as e:
            print(f"[{os.path.basename(p)}] ERROR al guardar: {e}")

if __name__ == "__main__":
    main()
