#!/usr/bin/env python3

#---------------------------------
#---Clasifica los tipos de movimientos del coche en base a
#---ciertos umbrales para steer y throttle
#-------------------------------

import os
import glob
import csv
import argparse
from typing import List, Tuple, Optional


# ----------------------------
# Buscar dataset.csv
# ----------------------------
def find_dataset_csvs() -> List[str]:
    patterns = [
        "../datasets/Deepracer_BaseMap_*/dataset.csv",
        "../datasets/validation/Deepracer_BaseMap_*/dataset.csv",
        "../datasets/test/Deepracer_BaseMap_*/dataset.csv",
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(list(dict.fromkeys(files)))


# ----------------------------
# Robustez CSV
# ----------------------------
def sniff_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,")
    except Exception:
        class D(csv.Dialect):
            delimiter = ";"
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        return D


def read_text_try_encodings(path: str, encodings: List[str]) -> Tuple[str, str]:
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return f.read(), enc
        except Exception as e:
            last_err = e
    raise last_err


def _pick_col(fieldnames: List[str], primary: str, contains: List[str], aliases: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in fieldnames}
    if primary.lower() in lower_map:
        return lower_map[primary.lower()]

    for c in fieldnames:
        if any(k in c.lower() for k in contains):
            return c

    for c in fieldnames:
        cl = c.lower()
        if any(a in cl for a in aliases):
            return c
    return None


def pick_steer_column(fieldnames: List[str]) -> Optional[str]:
    return _pick_col(
        fieldnames,
        primary="steer",
        contains=["steer"],
        aliases=["direccion", "ángulo", "angulo", "steering", "turn", "giro"],
    )


def pick_throttle_column(fieldnames: List[str]) -> Optional[str]:
    return _pick_col(
        fieldnames,
        primary="throttle",
        contains=["throttle"],
        aliases=["gas", "aceler", "acelerador", "throt", "accel"],
    )


def to_float(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


# ----------------------------
# REGLAS de la nueva clasificacion
# ----------------------------
def classify_turn_rule_based_relaxed(
    steer: float,
    throttle: Optional[float],
    # Recto: throttle alto y steer muy centrado
    straight_throttle_min: float = 0.70,
    straight_steer_abs_max: float = 0.08,
    steer_deadzone: float = 0.02,
) -> str:
    """
      - La dirección se decide SOLO por el signo del steer:
            steer < -deadzone -> LEFT_*
            steer > +deadzone -> RIGHT_*
      - STRAIGHT si throttle > straight_throttle_min y |steer| <= straight_steer_abs_max

    Clases:
      STRAIGHT
      LEFT_LONG / LEFT_MEDIUM / LEFT_SHARP
      RIGHT_LONG / RIGHT_MEDIUM / RIGHT_SHARP
      UNLABELED (solo si throttle falta o steer ~ 0 dentro deadzone y no es STRAIGHT)
    """
    s = float(steer)
    t = None if throttle is None else float(throttle)

    if t is None:
        return "UNLABELED"

    # STRAIGHT primero
    if (t > straight_throttle_min) and (abs(s) <= straight_steer_abs_max):
        return "STRAIGHT"

    # Si steer está muy cerca de 0, no forzamos izquierda/derecha
    if abs(s) <= steer_deadzone:
        return "UNLABELED"

    # Dirección solo por signo
    direction = "LEFT" if s < 0 else "RIGHT"

    # Magnitud SOLO por throttle
    if 0.77 <= t <= 1.0:
        mag = "LONG"
    elif 0.50 <= t <= 0.75:
        mag = "MEDIUM"
    else:
        mag = "SHARP"

    return f"{direction}_{mag}"


# ----------------------------
# Procesado de CSV
# ----------------------------
def process_one_csv(
    path: str,
    out_col: str,
    overwrite: bool,
    straight_throttle_min: float,
    straight_steer_abs_max: float,
    steer_deadzone: float,
) -> None:
    text, used_enc = read_text_try_encodings(path, ["utf-8-sig", "utf-8", "latin1"])
    dialect = sniff_dialect(text[:4096])

    rows = []
    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    if reader.fieldnames is None:
        raise RuntimeError(f"[{path}] CSV sin cabecera o vacío.")

    fieldnames = list(reader.fieldnames)

    steer_col = pick_steer_column(fieldnames)
    if steer_col is None:
        raise RuntimeError(f"[{path}] No encuentro columna de steer. Columnas: {fieldnames}")

    throttle_col = pick_throttle_column(fieldnames)
    if throttle_col is None:
        raise RuntimeError(f"[{path}] No encuentro columna de throttle. Columnas: {fieldnames}")

    new_fieldnames = fieldnames + [out_col] if out_col not in fieldnames else fieldnames[:]

    for r in reader:
        steer = to_float(r.get(steer_col, ""))
        thr = to_float(r.get(throttle_col, ""))

        if steer is None or thr is None:
            r[out_col] = ""
        else:
            r[out_col] = classify_turn_rule_based_relaxed(
                steer=steer,
                throttle=thr,
                straight_throttle_min=straight_throttle_min,
                straight_steer_abs_max=straight_steer_abs_max,
                steer_deadzone=steer_deadzone,
            )
        rows.append(r)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding=used_enc, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames, dialect=dialect)
        writer.writeheader()
        for r in rows:
            for k in new_fieldnames:
                if k not in r:
                    r[k] = ""
            writer.writerow(r)

    if overwrite:
        os.replace(tmp_path, path)
    else:
        out_path = path.replace("dataset.csv", f"dataset_with_{out_col}.csv")
        os.replace(tmp_path, out_path)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Añade una columna de clasificación de giros (RELAXED: dirección por signo del steer, magnitud por throttle)."
    )
    ap.add_argument("--out_col", default="turn_class", help="Nombre de la columna nueva (default: turn_class)")
    ap.add_argument("--no_overwrite", action="store_true", help="No sobrescribir: crea dataset_with_<col>.csv")

    ap.add_argument("--straight_throttle_min", type=float, default=0.70, help="Throttle mínimo para STRAIGHT (default 0.70)")
    ap.add_argument("--straight_steer_abs_max", type=float, default=0.10, help="|steer| máximo para STRAIGHT (default 0.10)")

    ap.add_argument("--steer_deadzone", type=float, default=0.01,
                    help="Zona muerta: si |steer| <= deadzone y no es STRAIGHT => UNLABELED (default 0.01)")

    args = ap.parse_args()

    files = find_dataset_csvs()
    if not files:
        print("No encontré dataset.csv en ../datasets, ../datasets/validation, ../datasets/test.")
        return

    print(f"Encontrados {len(files)} dataset.csv")
    for p in files:
        try:
            process_one_csv(
                p,
                out_col=args.out_col,
                overwrite=(not args.no_overwrite),
                straight_throttle_min=args.straight_throttle_min,
                straight_steer_abs_max=args.straight_steer_abs_max,
                steer_deadzone=args.steer_deadzone,
            )
            print(f"[OK] {p}")
        except Exception as e:
            print(f"[ERROR] {p} -> {e}")


if __name__ == "__main__":
    main()
