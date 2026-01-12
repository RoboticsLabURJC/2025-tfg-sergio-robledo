#!/usr/bin/env python3

#---------------------------------
#---Cambia en todos los datasets el umbral que delimita el estado 
#---1/2/3 dependiendo del steer en ese momento
#-------------------------------

import argparse
import glob
import os
from pathlib import Path

import pandas as pd


def update_estado_in_csv(csv_path: Path, umbral: float, dry_run: bool) -> None:
    # Lectura robusta por si algún CSV tiene otra codificación
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError(f"No pude leer {csv_path} (problema de encoding o archivo corrupto)")

    if "steer" not in df.columns:
        raise ValueError(f"[{csv_path}] Falta columna 'steer'. Columnas: {list(df.columns)}")
    if "estado" not in df.columns:
        raise ValueError(f"[{csv_path}] Falta columna 'estado'. Columnas: {list(df.columns)}")

    steer = pd.to_numeric(df["steer"], errors="coerce")

    # Nuevo estado
    estado_new = pd.Series(2, index=df.index, dtype="int64")
    estado_new[steer < -umbral] = 1
    estado_new[steer >  umbral] = 3

    # Conteo cambios
    old = pd.to_numeric(df["estado"], errors="coerce").fillna(-999).astype("int64")
    changed = int((old != estado_new).sum())

    c1 = int((estado_new == 1).sum())
    c2 = int((estado_new == 2).sum())
    c3 = int((estado_new == 3).sum())

    print(f"- {csv_path}")
    print(f"  umbral=±{umbral:.3f} | cambios: {changed}/{len(df)} | nuevos: izq={c1}, centro={c2}, dcha={c3}")

    if dry_run:
        return

    # Backup
    bak = csv_path.with_suffix(csv_path.suffix + ".bak")
    if not bak.exists():
        csv_path.replace(bak)
        df = pd.read_csv(bak, encoding="utf-8", engine="python") if bak.stat().st_size else df

    df["estado"] = estado_new
    df.to_csv(csv_path, index=False, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Recalcula 'estado' usando 'steer' en Deepracer*/dataset.csv (incluye validation/test).")
    ap.add_argument("--root", default=".", help="Directorio raíz donde buscar Deepracer* (por defecto: .)")
    ap.add_argument("--umbral", type=float, default=0.15, help="Umbral para centro: |steer|<=umbral (por defecto 0.15)")
    ap.add_argument("--dry-run", action="store_true", help="No escribe cambios, solo muestra el resumen")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    deepracer_dirs = [Path(p) for p in glob.glob(str(root / "Deepracer*")) if Path(p).is_dir()]
    if not deepracer_dirs:
        print(f"No hay carpetas con la expresion Deepracer* en: {root}")
        return

    targets = []
    for d in deepracer_dirs:
        for rel in ("dataset.csv", os.path.join("validation", "dataset.csv"), os.path.join("test", "dataset.csv")):
            p = d / rel
            if p.exists():
                targets.append(p)

    if not targets:
        print("Existe ruta con la expresion Deepracer* pero no hay dataset.csv en raíz/validation/test.")
        return

    print(f"Encontrados {len(targets)} CSV(s). Procesando con umbral=±{args.umbral:.3f}")
    for p in targets:
        try:
            update_estado_in_csv(p, args.umbral, args.dry_run)
        except Exception as e:
            print(f"  [ERROR] {p}: {e}")

    print("\nListo.")
    if args.dry_run:
        print("Dry-run activado: no se escribió ningún archivo.")
    else:
        print("Backups: dataset.csv.bak (si no existían).")


if __name__ == "__main__":
    main()
