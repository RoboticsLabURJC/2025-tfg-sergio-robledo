#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
import numpy as np
import re

# -------------------------------------------------------
# 1. Parsear nombre de carpeta: track_id, direction, v1/v2

def parse_track_name(folder_name):
    """
    Extrae:
      - track_id   -> número (ej: 3, 14)
      - direction  -> 'C' o 'CC'
      - version    -> 'v1' o 'v2'

    Ejemplos esperados de carpeta:
      Deepracer_BaseMap_3C17648...
      Deepracer_BaseMap_3Cv217648...
      Deepracer_BaseMap_12CC1764...
      Deepracer_BaseMap_12CCv21764...
    """
    pattern = r"BaseMap_(\d+)(CC|C)(v2)?"
    m = re.search(pattern, folder_name)
    if not m:
        return None, None, None

    track_id  = int(m.group(1))
    direction = m.group(2) 
    version   = m.group(3) or "v1"
    return track_id, direction, version


# -------------------------------------------------------
# 2. Cargar info de TODOS los dataset.csv (por carpeta)

def load_datasets_by_folder(base_dir):
    pattern = os.path.join(base_dir, "Deepracer_BaseMap_*", "dataset.csv")
    csv_paths = sorted(glob.glob(pattern))

    rows = []
    if not csv_paths:
        print(f"[ERROR] No se encontró ningún dataset.csv con patrón: {pattern}")
        return pd.DataFrame([])

    for csv_path in csv_paths:
        folder = os.path.basename(os.path.dirname(csv_path))
        track_id, direction, version = parse_track_name(folder)

        if track_id is None:
            print(f"[WARN] No se pudo interpretar el nombre de carpeta: {folder}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] No se pudo leer {csv_path}: {e}")
            continue

        if not {"steer", "throttle"}.issubset(df.columns):
            print(f"[WARN] dataset sin columnas steer/throttle: {csv_path}")
            continue

        n = len(df)
        if n == 0:
            print(f"[WARN] dataset vacío: {csv_path}")
            continue

        steer    = df["steer"].to_numpy(dtype=float)
        throttle = df["throttle"].to_numpy(dtype=float)
        abs_steer = np.abs(steer)

        curves_mask = abs_steer > 0.4     # criterio de curva
        fast_mask   = throttle > 0.6      # criterio de gas fuerte

        rows.append({
            "folder": folder,
            "csv_path": csv_path,
            "track_id": track_id,
            "direction": direction,
            "version": version,
            "samples": n,
            "sum_abs_steer": abs_steer.sum(),
            "cnt_curves": int(curves_mask.sum()),
            "sum_throttle": throttle.sum(),
            "cnt_fast": int(fast_mask.sum()),
        })

    return pd.DataFrame(rows)


# -------------------------------------------------------
# 3. Agrupar v1/v2 por (track_id, direction)

def group_by_track_direction(df_folders):
    groups = []
    grouped = df_folders.groupby(["track_id", "direction"])

    for (track_id, direction), g in grouped:
        samples      = int(g["samples"].sum())
        sum_abs_steer = g["sum_abs_steer"].sum()
        cnt_curves    = int(g["cnt_curves"].sum())
        sum_throttle  = g["sum_throttle"].sum()
        cnt_fast      = int(g["cnt_fast"].sum())

        mean_abs_steer = sum_abs_steer / samples
        frac_curves    = cnt_curves / samples
        mean_throttle  = sum_throttle / samples
        frac_fast      = cnt_fast / samples

        groups.append({
            "track_id": track_id,
            "direction": direction,
            "folders": list(g["folder"]),
            "samples": samples,
            "mean_abs_steer": mean_abs_steer,
            "frac_curves": frac_curves,
            "mean_throttle": mean_throttle,
            "frac_fast": frac_fast,
        })

    return pd.DataFrame(groups)


# -------------------------------------------------------
# 4. Split por cobertura (no ratios, round-robin por score)

def coverage_split(df_tracks, split_names=("train", "val", "test")):
    """
    df_tracks: una fila por (track_id, direction) ya fusionando v1/v2.

    Define un score de “dificultad”:
        score = 0.5 * frac_curves + 0.5 * frac_fast

    Ordena por score desc y reparte por round-robin:
        pista1 -> train
        pista2 -> val
        pista3 -> test
        pista4 -> train
        ...
    """
    df = df_tracks.copy()
    df["score"] = 0.5 * df["frac_curves"] + 0.5 * df["frac_fast"]
    df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)

    assigned = {s: [] for s in split_names}
    counts   = {s: 0 for s in split_names}

    for i, (_, row) in enumerate(df_sorted.iterrows()):
        s = split_names[i % len(split_names)]
        assigned[s].append(row)
        counts[s] += int(row["samples"])

    dfs = {s: pd.DataFrame(rows) if rows else pd.DataFrame(columns=df_sorted.columns)
           for s, rows in assigned.items()}
    return dfs, counts


# -------------------------------------------------------
# 5. Imprimir resultado

def print_split(dfs, counts):
    total = sum(counts.values())

    def split_stats(df):
        if df.empty:
            return dict(samples=0,
                        mean_abs_steer=0.0,
                        frac_curves=0.0,
                        mean_throttle=0.0,
                        frac_fast=0.0)
        w = df["samples"].to_numpy()
        w_norm = w / w.sum()
        return {
            "samples": int(df["samples"].sum()),
            "mean_abs_steer": float(np.sum(df["mean_abs_steer"] * w_norm)),
            "frac_curves": float(np.sum(df["frac_curves"] * w_norm)),
            "mean_throttle": float(np.sum(df["mean_throttle"] * w_norm)),
            "frac_fast": float(np.sum(df["frac_fast"] * w_norm)),
        }

    print("\n===================== PROPUESTA DE DIVISIÓN (por cobertura) =====================")

    for name in ("train", "val", "test"):
        df = dfs[name]
        stats = split_stats(df)

        print(f"\n{name.upper()} — {stats['samples']} muestras "
              f"({100*stats['samples']/max(total,1):.2f}% del total)")
        print(f"  · |steer| medio     : {stats['mean_abs_steer']:.3f}")
        print(f"  · frac curvas >0.4  : {stats['frac_curves']*100:.1f}%")
        print(f"  · throttle medio    : {stats['mean_throttle']:.3f}")
        print(f"  · frac throttle>0.6 : {stats['frac_fast']*100:.1f}%")

        if df.empty:
            continue

        for _, r in df.sort_values(["track_id", "direction"]).iterrows():
            folders_str = ", ".join(r["folders"])
            print(f"    - Track {r.track_id} [{r.direction}] "
                  f"({r.samples} muestras)  "
                  f"curvas={r.frac_curves*100:.1f}%  "
                  f"fast={r.frac_fast*100:.1f}%  "
                  f"score={r.score:.3f}  "
                  f"folders=[{folders_str}]")


    print("\n===================== RESUMEN GLOBAL =====================")
    print(f"Total muestras : {total}")
    for name in ("train", "val", "test"):
        print(f"{name.capitalize():5s}: {counts[name]:6d} muestras ({100*counts[name]/max(total,1):5.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Proponer split Train/Val/Test priorizando cobertura de casos"
    )
    parser.add_argument("--base-dir", default=".",
                        help="Directorio raíz donde están Deepracer_BaseMap_*")
    args = parser.parse_args()

    print(f"[INFO] Base dir: {args.base_dir}")

    df_folders = load_datasets_by_folder(args.base_dir)
    if df_folders.empty:
        print("[ERROR] No se han podido cargar datasets.")
        return

    df_tracks = group_by_track_direction(df_folders)
    print(f"[INFO] Encontradas {len(df_tracks)} pistas lógicas (track_id+direction), agrupando v1/v2.")

    dfs, counts = coverage_split(df_tracks, split_names=("train", "val", "test"))
    print_split(dfs, counts)


if __name__ == "__main__":
    main()
