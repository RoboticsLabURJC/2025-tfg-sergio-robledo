#!/usr/bin/env python3
# turnclass_by_track_live.py
#
# Sin argumentos:
# - Busca dataset.csv en train/val/test
# - Deriva turn_class al vuelo desde steer/throttle
# - Muestra 3 gráficos (train/val/test) en pantalla 

import os
import glob
import csv
from collections import Counter
from typing import Dict, List, Tuple, Optional

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


def read_text_try_encodings(path: str, encodings: List[str]) -> str:
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return f.read()
        except Exception as e:
            last_err = e
    raise last_err


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


def classify_turn_rule_based_relaxed(
    steer: float,
    throttle: Optional[float],
    straight_throttle_min: float = 0.70,
    straight_steer_abs_max: float = 0.08,
    steer_deadzone: float = 0.02,
) -> str:
    s = float(steer)
    t = None if throttle is None else float(throttle)

    if t is None:
        return "UNLABELED"

    if (t > straight_throttle_min) and (abs(s) <= straight_steer_abs_max):
        return "STRAIGHT"

    if abs(s) <= steer_deadzone:
        return "UNLABELED"

    direction = "LEFT" if s < 0 else "RIGHT"

    if 0.77 <= t <= 1.0:
        mag = "LONG"
    elif 0.50 <= t <= 0.75:
        mag = "MEDIUM"
    else:
        mag = "SHARP"

    return f"{direction}_{mag}"


# -------------------------
# Dataset discovery
# -------------------------
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


def track_name_from_csv_path(csv_path: str) -> str:
    return os.path.basename(os.path.dirname(csv_path))


# -------------------------
# Counting
# -------------------------
def count_turnclass_in_csv(csv_path: str, prefer_col: str = "turn_class") -> Counter:
    text = read_text_try_encodings(csv_path, ["utf-8-sig", "utf-8", "latin1"])
    dialect = sniff_dialect(text[:4096])

    rdr = csv.DictReader(text.splitlines(), dialect=dialect)
    if rdr.fieldnames is None:
        return Counter()

    fieldnames = list(rdr.fieldnames)
    has_col = prefer_col in fieldnames

    # Si no está la columna, derivamos con steer/throttle
    if (not has_col) and (("steer" not in fieldnames) or ("throttle" not in fieldnames)):
        raise RuntimeError(
            f"[{csv_path}] No existe '{prefer_col}' y tampoco hay steer/throttle para derivar. Columnas: {fieldnames}"
        )

    c = Counter()
    for row in rdr:
        if has_col:
            v = (row.get(prefer_col, "") or "").strip()
            if v == "":
                v = "(empty)"
            c[v] += 1
        else:
            steer = to_float(row.get("steer"))
            thr = to_float(row.get("throttle"))
            if steer is None:
                c["UNLABELED"] += 1
            else:
                v = classify_turn_rule_based_relaxed(steer=steer, throttle=thr)
                c[v] += 1
    return c


def aggregate_by_track(split: str) -> Dict[str, Counter]:
    files = find_dataset_csvs(split)
    out: Dict[str, Counter] = {}
    for f in files:
        out[track_name_from_csv_path(f)] = count_turnclass_in_csv(f, prefer_col="turn_class")
    return out


def union_classes(*maps: Dict[str, Counter]) -> List[str]:
    all_keys = set()
    for m in maps:
        for c in m.values():
            all_keys |= set(c.keys())

    preferred = [
        "STRAIGHT",
        "LEFT_LONG", "LEFT_MEDIUM", "LEFT_SHARP",
        "RIGHT_LONG", "RIGHT_MEDIUM", "RIGHT_SHARP",
        "UNLABELED",
        "(empty)",
    ]
    ordered = [k for k in preferred if k in all_keys]
    for k in sorted(all_keys):
        if k not in ordered:
            ordered.append(k)
    return ordered


# -------------------------
# Plotting
# -------------------------
def plot_split(track_counters: Dict[str, Counter], split: str, classes: List[str]) -> None:
    if not track_counters:
        print(f"[WARN] No hay pistas en split={split}")
        return

    tracks = sorted(track_counters.keys())
    n_tracks = len(tracks)
    n_classes = len(classes)

    # matriz counts
    M = []
    for t in tracks:
        c = track_counters[t]
        M.append([c.get(k, 0) for k in classes])

    fig_w = max(12, n_tracks * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, 6), dpi=120)

    x = list(range(n_tracks))
    group_width = 0.85
    bar_w = group_width / max(1, n_classes)

    for j, cls in enumerate(classes):
        xs = [i - group_width / 2 + (j + 0.5) * bar_w for i in x]
        ys = [M[i][j] for i in range(n_tracks)]
        ax.bar(xs, ys, width=bar_w, label=cls)

    ax.set_title(f"{split.upper()} - turn_class por pista (counts)")
    ax.set_ylabel("count")
    ax.set_xticks(x)
    ax.set_xticklabels(tracks, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()


def main():
    train_by_track = aggregate_by_track("train")
    val_by_track   = aggregate_by_track("val")
    test_by_track  = aggregate_by_track("test")

    classes = union_classes(train_by_track, val_by_track, test_by_track)

    plot_split(train_by_track, "train", classes)
    plot_split(val_by_track,   "val",   classes)
    plot_split(test_by_track,  "test",  classes)

    print("Clases detectadas:", classes)
    plt.show()


if __name__ == "__main__":
    main()
