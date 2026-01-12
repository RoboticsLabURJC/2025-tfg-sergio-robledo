#!/usr/bin/env python3
# plot_turn_classes.py

#---------------------------------
#---Plot para observar la clasificacion de los 7 estados de giro 
#-------------------------------


import os
import glob
import csv
import argparse
from collections import Counter
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# CSV helpers
# -------------------------
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


def find_dataset_csvs(split: str) -> List[str]:
    if split == "train":
        pat = "../datasets/Deepracer_BaseMap_*/dataset.csv"
    elif split == "val":
        pat = "../datasets/validation/Deepracer_BaseMap_*/dataset.csv"
    elif split == "test":
        pat = "../datasets/test/Deepracer_BaseMap_*/dataset.csv"
    else:
        raise ValueError("split debe ser train/val/test")
    return sorted(glob.glob(pat))

# Tener en cuenta casos de unlabeled
CANON = {
    # Izquierda
    "LEFT_LONG": "LEFT_LONG",
    "LEFT_MEDIUM": "LEFT_MEDIUM",
    "LEFT_SHARP": "LEFT_SHARP",
    # Derecha
    "RIGHT_LONG": "RIGHT_LONG",
    "RIGHT_MEDIUM": "RIGHT_MEDIUM",
    "RIGHT_SHARP": "RIGHT_SHARP",
    # Recto
    "STRAIGHT": "STRAIGHT",
    # Sin etiqueta
    "UNLABELED": "UNLABELED",
    # Variantes típicas
    "(empty)": "(empty)",
    "": "(empty)",
    "NONE": "UNLABELED",
    "N/A": "UNLABELED",
    "NA": "UNLABELED",
}

# Orden de aparicion

PREFERRED_ORDER = [
    "STRAIGHT",
    "LEFT_LONG", "LEFT_MEDIUM", "LEFT_SHARP",
    "RIGHT_LONG", "RIGHT_MEDIUM", "RIGHT_SHARP",
    "UNLABELED",
    "(empty)",
]


def canon_turn(v: str) -> str:
    s = (v or "").strip()
    if s == "":
        return "(empty)"
    key = s.upper().replace(" ", "_")
    return CANON.get(key, s)


# -------------------------
# Counting
# -------------------------
def count_turn_types_in_file(csv_path: str, col: str, canonicalize: bool = True) -> Counter:
    text, _enc = read_text_try_encodings(csv_path, ["utf-8-sig", "utf-8", "latin1"])
    dialect = sniff_dialect(text[:4096])

    rdr = csv.DictReader(text.splitlines(), dialect=dialect)
    if rdr.fieldnames is None:
        return Counter()

    if col not in rdr.fieldnames:
        raise RuntimeError(f"Columna '{col}' no existe. Columnas: {rdr.fieldnames}")

    c = Counter()
    for row in rdr:
        v = (row.get(col, "") or "").strip()
        if canonicalize:
            c[canon_turn(v)] += 1
        else:
            c[v if v != "" else "(empty)"] += 1
    return c


def aggregate_counts(split: str, col: str, canonicalize: bool = True) -> Tuple[Counter, int, int]:
    files = find_dataset_csvs(split)
    total = Counter()
    ok = 0
    bad = 0
    for f in files:
        try:
            total += count_turn_types_in_file(f, col, canonicalize=canonicalize)
            ok += 1
        except Exception as e:
            print(f"[WARN] {split}: {f} -> {e}")
            bad += 1
    return total, ok, bad


# -------------------------
# Plotting
# -------------------------
def _ordered_items(counter: Counter, top_k: int = 0) -> List[Tuple[str, int]]:
    if top_k > 0:
        return counter.most_common(top_k)

    items = []
    seen = set()

    for k in PREFERRED_ORDER:
        if k in counter:
            items.append((k, counter[k]))
            seen.add(k)

    rest = [(k, v) for k, v in counter.items() if k not in seen]
    rest.sort(key=lambda kv: kv[1], reverse=True)
    items.extend(rest)
    return items


def plot_counts(counter: Counter, title: str, out_png: str, top_k: int = 0) -> None:
    items = _ordered_items(counter, top_k=top_k)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(12, 5), dpi=140)
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("count")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_percent(counter: Counter, title: str, out_png: str, top_k: int = 0) -> None:
    total = sum(counter.values())
    if total <= 0:
        return

    items = _ordered_items(counter, top_k=top_k)
    labels = [k for k, _ in items]
    values = [100.0 * v / total for _, v in items]

    plt.figure(figsize=(12, 5), dpi=140)
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("percent (%)")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_train_val_test_side_by_side(train: Counter, val: Counter, test: Counter,
                                     title: str, out_png: str, top_k_union: int = 0) -> None:
    union = Counter()
    union.update(train)
    union.update(val)
    union.update(test)

    classes = [k for k, _ in _ordered_items(union, top_k=top_k_union)] if top_k_union > 0 else [
        k for k, _ in _ordered_items(union, top_k=0)
    ]

    x = list(range(len(classes)))
    tr = [train.get(k, 0) for k in classes]
    va = [val.get(k, 0) for k in classes]
    te = [test.get(k, 0) for k in classes]

    width = 0.28
    plt.figure(figsize=(14, 5), dpi=140)
    plt.bar([i - width for i in x], tr, width=width, label="train")
    plt.bar(x,                 va, width=width, label="val")
    plt.bar([i + width for i in x], te, width=width, label="test")
    plt.title(title)
    plt.ylabel("count")
    plt.xticks(x, classes, rotation=35, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_train_val_test_side_by_side_percent(train: Counter, val: Counter, test: Counter,
                                             title: str, out_png: str, top_k_union: int = 0) -> None:
    union = Counter()
    union.update(train)
    union.update(val)
    union.update(test)

    classes = [k for k, _ in _ordered_items(union, top_k=top_k_union)] if top_k_union > 0 else [
        k for k, _ in _ordered_items(union, top_k=0)
    ]
    x = list(range(len(classes)))

    def to_pct(c: Counter):
        tot = sum(c.values())
        if tot <= 0:
            return [0.0 for _ in classes]
        return [100.0 * c.get(k, 0) / tot for k in classes]

    tr = to_pct(train)
    va = to_pct(val)
    te = to_pct(test)

    width = 0.28
    plt.figure(figsize=(14, 5), dpi=140)
    plt.bar([i - width for i in x], tr, width=width, label="train")
    plt.bar(x,                 va, width=width, label="val")
    plt.bar([i + width for i in x], te, width=width, label="test")
    plt.title(title)
    plt.ylabel("percent (%)")
    plt.xticks(x, classes, rotation=35, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", default="turn_class", help="Columna a analizar (default: turn_class)")
    ap.add_argument("--out_dir", default="turn_class_plots", help="Carpeta salida (default: turn_class_plots)")
    ap.add_argument("--top_k", type=int, default=0, help="Si >0, solo plotea top-k clases (por split).")
    ap.add_argument("--top_k_union", type=int, default=0, help="Si >0, limita clases en comparación train/val/test (sobre unión).")
    ap.add_argument("--no_canon", action="store_true", help="No canonicalizar valores (usa texto tal cual).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    canonicalize = (not args.no_canon)

    train_c, ok_tr, bad_tr = aggregate_counts("train", args.col, canonicalize=canonicalize)
    val_c,   ok_va, bad_va = aggregate_counts("val",   args.col, canonicalize=canonicalize)
    test_c,  ok_te, bad_te = aggregate_counts("test",  args.col, canonicalize=canonicalize)

    print("Resumen lectura:")
    print(f"  train: ok={ok_tr}, bad={bad_tr}, total_rows={sum(train_c.values())}")
    print(f"  val  : ok={ok_va}, bad={bad_va}, total_rows={sum(val_c.values())}")
    print(f"  test : ok={ok_te}, bad={bad_te}, total_rows={sum(test_c.values())}")

    # Plots por split
    plot_counts(train_c, f"TRAIN - {args.col} frequency (count)", os.path.join(args.out_dir, "train_counts.png"), top_k=args.top_k)
    plot_percent(train_c, f"TRAIN - {args.col} frequency (percent)", os.path.join(args.out_dir, "train_percent.png"), top_k=args.top_k)

    plot_counts(val_c, f"VAL - {args.col} frequency (count)", os.path.join(args.out_dir, "val_counts.png"), top_k=args.top_k)
    plot_percent(val_c, f"VAL - {args.col} frequency (percent)", os.path.join(args.out_dir, "val_percent.png"), top_k=args.top_k)

    plot_counts(test_c, f"TEST - {args.col} frequency (count)", os.path.join(args.out_dir, "test_counts.png"), top_k=args.top_k)
    plot_percent(test_c, f"TEST - {args.col} frequency (percent)", os.path.join(args.out_dir, "test_percent.png"), top_k=args.top_k)

    # Comparación lado a lado
    plot_train_val_test_side_by_side(
        train_c, val_c, test_c,
        f"TRAIN vs VAL vs TEST - {args.col} (count)",
        os.path.join(args.out_dir, "train_val_test_counts.png"),
        top_k_union=args.top_k_union,
    )
    plot_train_val_test_side_by_side_percent(
        train_c, val_c, test_c,
        f"TRAIN vs VAL vs TEST - {args.col} (percent)",
        os.path.join(args.out_dir, "train_val_test_percent.png"),
        top_k_union=args.top_k_union,
    )

    # Print clases
    print("\nClases detectadas (train):")
    for k, v in _ordered_items(train_c, top_k=0):
        print(f"  {k:12s} : {v}")

    print("\nClases detectadas (val):")
    for k, v in _ordered_items(val_c, top_k=0):
        print(f"  {k:12s} : {v}")

    print("\nClases detectadas (test):")
    for k, v in _ordered_items(test_c, top_k=0):
        print(f"  {k:12s} : {v}")

    print(f"\n[OK] Plots guardados en: {args.out_dir}/")


if __name__ == "__main__":
    main()
