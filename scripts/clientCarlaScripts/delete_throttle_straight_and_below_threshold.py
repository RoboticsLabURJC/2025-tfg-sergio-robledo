#!/usr/bin/env python3
import os
import glob
import csv
import shutil

BASE_DIR = "../datasets"
OUT_REMOVED_CSV = "straight_estado2_throttle_lt_0.6_removed.csv"
THROTTLE_THRESHOLD = 0.6
ESTADO_TARGET = 2
MAKE_BACKUP = True

def find_dataset_csvs():
    patterns = [
        os.path.join(BASE_DIR, "Deepracer_BaseMap_*", "dataset.csv"),
        os.path.join(BASE_DIR, "validation", "Deepracer_BaseMap_*", "dataset.csv"),
        os.path.join(BASE_DIR, "test", "Deepracer_BaseMap_*", "dataset.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(list(dict.fromkeys(files)))

def to_float(x):
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

def to_int(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def sniff_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,")
    except Exception:
        class D(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        return D

def read_text_try_encodings(path, encodings=("utf-8-sig", "utf-8", "latin1")):
    last = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return f.read(), enc
        except Exception as e:
            last = e
    raise last

def pick_col(fieldnames, name):
    lower_map = {c.lower(): c for c in fieldnames}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    for c in fieldnames:
        if name.lower() in c.lower():
            return c
    return None

def ensure_backup(path):
    if not MAKE_BACKUP:
        return
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)

def main():
    files = find_dataset_csvs()
    if not files:
        print("[ERROR] No encontré dataset.csv en las rutas esperadas.")
        return

    removed_rows_all = []
    total_removed = 0
    total_kept = 0
    total_seen = 0
    files_touched = 0

    for path in files:
        try:
            text, used_enc = read_text_try_encodings(path)
            dialect = sniff_dialect(text[:4096])

            reader = csv.DictReader(text.splitlines(), dialect=dialect)
            if reader.fieldnames is None:
                print(f"[SKIP] {path} (sin cabecera)")
                continue

            fieldnames = list(reader.fieldnames)
            col_estado = pick_col(fieldnames, "estado")
            col_thr    = pick_col(fieldnames, "throttle")

            if col_estado is None or col_thr is None:
                print(f"[SKIP] {path} (faltan columnas estado/throttle)")
                continue

            kept = []
            removed = []
            seen = 0

            for r in reader:
                seen += 1
                est = to_int(r.get(col_estado))
                thr = to_float(r.get(col_thr))

                cond_remove = (est == ESTADO_TARGET and thr is not None and thr < THROTTLE_THRESHOLD)
                if cond_remove:
                    rr = dict(r)
                    rr["__source_csv__"] = path
                    removed.append(rr)
                else:
                    kept.append(r)

            total_seen += seen
            total_kept += len(kept)
            total_removed += len(removed)

            if len(removed) == 0:
                print(f"[..] {path} -> 0 borradas / {seen} total")
                continue

            ensure_backup(path)

            tmp_path = path + ".tmp"
            with open(tmp_path, "w", newline="", encoding=used_enc) as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, dialect=dialect)
                w.writeheader()
                for r in kept:
                    for k in fieldnames:
                        if k not in r:
                            r[k] = ""
                    w.writerow(r)

            os.replace(tmp_path, path)
            files_touched += 1

            removed_rows_all.extend(removed)
            print(f"[OK] {path} -> {len(removed)} borradas / {seen} total (reescrito)")

        except Exception as e:
            print(f"[ERROR] {path} -> {e}")

    print("\n====================")
    print(f"Archivos modificados: {files_touched}")
    print(f"Total filas leídas  : {total_seen}")
    print(f"Total borradas      : {total_removed}")
    print(f"Total conservadas   : {total_kept}")
    print("====================\n")


if __name__ == "__main__":
    main()
