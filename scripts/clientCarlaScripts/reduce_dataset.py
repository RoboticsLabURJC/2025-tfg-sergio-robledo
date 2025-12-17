#!/usr/bin/env python3

import os
import glob
import argparse
import shutil
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

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
        # Aleatorio dentro de este df, con semilla
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    if mode == "stride":
        idx = np.linspace(0, total - 1, n)
        idx = np.round(idx).astype(int)
        idx = np.unique(idx)[:n]
        return df.iloc[idx].reset_index(drop=True)

    # por defecto, first
    return df.iloc[:n].copy()


def compute_uniform_quotas(counts: Dict[int, int], target_total: int) -> Dict[int, int]:
    """
    Calcula cuotas UNIFORMES por estado (1/2/3) sumando target_total en total.
    Es la parte "totalmente plana".
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


def compute_per_file_quotas_for_state(
    counts_per_file: Dict[str, int],
    quota_c: int
) -> Dict[str, int]:
    """
    Reparte la cuota quota_c de un estado concreto entre ficheros,
    de forma PROPORCIONAL a lo que tenía cada fichero.
    """
    # Filtramos solo ficheros con al menos 1 muestra
    valid = {src: cnt for src, cnt in counts_per_file.items() if cnt > 0}
    if not valid or quota_c <= 0:
        return {src: 0 for src in counts_per_file.keys()}

    total_c = sum(valid.values())
    if quota_c >= total_c:
        # No hace falta recortar: nos quedamos TODO
        return {src: cnt for src, cnt in counts_per_file.items()}

    # Asignación base proporcional
    raw = {src: quota_c * (cnt / total_c) for src, cnt in valid.items()}
    base = {src: int(np.floor(v)) for src, v in raw.items()}
    assigned = sum(base.values())
    remaining = quota_c - assigned

    # Ordenamos por parte fraccionaria descendente para repartir el resto
    fracs: List[Tuple[float, str]] = sorted(
        [(raw[src] - base[src], src) for src in valid.keys()],
        reverse=True
    )

    i = 0
    # Repartimos el resto sin superar la capacidad de ningún fichero
    while remaining > 0 and fracs:
        frac, src = fracs[i % len(fracs)]
        if base[src] < valid[src]:
            base[src] += 1
            remaining -= 1
        i += 1
        if i > 10_000:  # por seguridad, no debería llegar aquí
            break

    # Aseguramos que ninguna cuota supere el número real de filas
    for src, cnt in valid.items():
        if base[src] > cnt:
            base[src] = cnt

    # Para ficheros sin muestras de ese estado, cuota 0
    quotas_per_file = {src: 0 for src in counts_per_file.keys()}
    quotas_per_file.update(base)
    return quotas_per_file


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Recorta GLOBALMENTE (sobre todos los CSV) hasta un total objetivo, "
            "con cuotas por estados 1/2/3. La 'fuerza' del aplanamiento se controla "
            "con --flatten-alpha: 1.0 = muy plano, 0.0 = casi como el original."
        )
    )
    ap.add_argument("--pattern",
                    default="../datasets/validation/Deepracer_BaseMap_*/dataset.csv",
                    help="Patrón de búsqueda (por defecto: ../datasets/validation/Deepracer_BaseMap_*/dataset.csv)")
    ap.add_argument("--target-total", type=int, default=1462,
                    help="Número total de filas objetivo (global, sumando todos los CSV, sólo estados 1/2/3).")
    ap.add_argument("--keep", choices=["first", "last", "random", "stride"], default="random",
                    help="Estrategia de selección dentro de cada estado por fichero (por defecto: random).")
    ap.add_argument("--seed", type=int, default=42, help="Semilla base.")
    ap.add_argument("--flatten-alpha", type=float, default=0.5,
                    help=(
                        "Fuerza del aplanamiento global por estado en [0,1]. "
                        "1.0 = completamente plano (todos los estados igual), "
                        "0.0 = nada de aplanamiento (solo recorte global). "
                        "Por defecto: 0.5."
                    ))
    ap.add_argument("--dry-run", action="store_true", help="No escribe cambios, solo muestra acciones.")
    ap.add_argument("--no-backup", action="store_true", help="No crear copia .bak antes de sobrescribir.")
    ap.add_argument("--drop-non123", action="store_true",
                    help="Si se indica, se eliminan filas con estado fuera de {1,2,3}. Por defecto se PRESERVAN.")
    args = ap.parse_args()

    # Clamp por seguridad
    alpha = max(0.0, min(1.0, args.flatten_alpha))
    if alpha != args.flatten_alpha:
        print(f"[INFO] flatten-alpha ajustado a {alpha} (estaba fuera de [0,1]).")

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

    # Conteos globales por estado (originales)
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

    # Cuotas UNIFORMES por estado (la versión "totalmente plana")
    quotas_uniform = compute_uniform_quotas(counts, target_total)

    # Mezcla entre cuotas uniformes y distribución original según alpha
    #   alpha = 1 -> quotas_final = quotas_uniform  (máxima planitud)
    #   alpha = 0 -> quotas_final ≈ counts (sólo se recorta si target_total < total)
    sum_counts = sum(counts.values())
    quotas_final = {}
    for c in [1, 2, 3]:
        # proporción original de este estado
        if sum_counts > 0:
            target_orig_c = counts[c] / sum_counts * target_total
        else:
            target_orig_c = 0.0

        q_uniform = quotas_uniform.get(c, 0)
        q_mix = (1.0 - alpha) * target_orig_c + alpha * q_uniform
        q_mix = int(round(q_mix))

        # No puede superar lo que hay disponible de ese estado
        q_mix = min(q_mix, counts[c])
        quotas_final[c] = q_mix

    print("Cuotas objetivo GLOBAL por estado (mezcla original/plano):", quotas_final,
          "| suma =", sum(quotas_final.values()))
    print(f"[INFO] flatten-alpha = {alpha} (1.0 = muy plano, 0.0 = casi original)")

    grouped_by_src = df_123.groupby("_src", dropna=False)
    # counts_per_file[state][src] = nº filas de ese estado en ese fichero
    counts_per_file: Dict[int, Dict[str, int]] = {1: {}, 2: {}, 3: {}}
    for src, g in grouped_by_src:
        est_local = pd.to_numeric(g["estado"], errors="coerce").astype("Int64")
        for c in [1,2,3]:
            counts_per_file[c][src] = int((est_local == c).sum())

    # ---- Seleccionar filas por fichero y estado según cuotas PROPORCIONALES ----
    selected_parts: List[pd.DataFrame] = []
    for c in [1,2,3]:
        quota_c = quotas_final.get(c, 0)
        if quota_c <= 0:
            continue

        # Cuotas para cada fichero dentro del estado c
        per_file_quota = compute_per_file_quotas_for_state(counts_per_file[c], quota_c)
        print(f"\nEstado {c}: cuota global={quota_c} -> cuotas por fichero:")
        for src in sorted(per_file_quota.keys()):
            print(f"  {os.path.basename(src)}: {per_file_quota[src]} / {counts_per_file[c].get(src,0)}")

        # Selección en cada fichero
        for src, group in grouped_by_src:
            n_q = per_file_quota.get(src, 0)
            if n_q <= 0:
                continue

            g_state = group[pd.to_numeric(group["estado"], errors="coerce").astype("Int64") == c]
            if g_state.empty:
                continue

            # Semilla distinta pero reproducible por (src, estado)
            seed_local = args.seed + (hash((src, c)) % 10_000)

            df_sel = select_rows_mode(g_state, n_q, args.keep, seed_local)
            selected_parts.append(df_sel)

    if selected_parts:
        df_selected = pd.concat(selected_parts, ignore_index=True)
    else:
        df_selected = df_123.iloc[0:0].copy()

    # Barajar globalmente para no sesgar por orden
    df_selected = df_selected.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Si no se pide eliminar non123, los preservamos íntegros
    if args.drop_non123:
        final = df_selected
    else:
        final = pd.concat([df_selected, df_rest], ignore_index=True)
        final = final.reset_index(drop=True)

    # Repartimos por archivo de origen y guardamos
    grouped_final = final.groupby("_src", dropna=False)

    print("\nResumen por archivo (previo a escribir):")
    for p in paths:
        if p in grouped_final.groups:
            sub = grouped_final.get_group(p)
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
        if p in grouped_final.groups:
            out = grouped_final.get_group(p).copy()
        else:
            # Si no quedó nada para este archivo, escribimos CSV vacío con las mismas columnas que tenía originalmente
            try:
                orig = pd.read_csv(p)
                out = orig.iloc[0:0].copy()
            except Exception:
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
