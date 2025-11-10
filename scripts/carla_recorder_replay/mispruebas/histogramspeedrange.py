#!/usr/bin/env python3
# ------------------------------------------------
# Histograma de velocidades (columna 'speed') 0..4 m/s en pasos de 0.2
# Agrega todos los CSV que cumplan un patrón y dibuja
# barras: [0.0,0.2), [0.2,0.4), ..., [3.8,4.0].
# Uso:
#   python hist_speed.py --pattern "../datasets/Deepracer_*/dataset.csv" --save "hist_speed.png"
#   (Parámetros opcionales: --col speed --vmin 0 --vmax 4 --step 0.2)
# ------------------------------------------------
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cargar_speed(csv_path: str, col="speed") -> np.ndarray | None:
    """Lee la columna 'speed' y la devuelve como array float"""
    try:
        df = pd.read_csv(csv_path, usecols=[col])
    except ValueError:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[ERROR] No pude leer {csv_path}: {e}")
            return None
    except Exception as e:
        print(f"[ERROR] No pude leer {csv_path}: {e}")
        return None

    if col not in df.columns:
        print(f"[WARN] {csv_path} no tiene columna '{col}'. Omitido.")
        return None

    spd = pd.to_numeric(df[col], errors="coerce").astype(float).dropna().to_numpy()
    return spd

def hist_acumulado(speed_arrays, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Acumula histograma usando bordes explícitos."""
    counts_total = np.zeros(len(edges) - 1, dtype=int)
    for spd in speed_arrays:
        if spd.size == 0:
            continue
        c, _ = np.histogram(spd, bins=edges)
        counts_total += c
    return counts_total, edges

def etiquetas_bins(edges: np.ndarray) -> list[str]:
    labels = []
    for i in range(len(edges)-1):
        a, b = edges[i], edges[i+1]
        labels.append(f"{a:.1f}–{b:.1f}")
    return labels

def plot_barras_speed(counts: np.ndarray, edges: np.ndarray, title: str, out_path: str | None):
    labels = etiquetas_bins(edges)
    total = counts.sum() if counts.sum() > 0 else 1

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts)
    ax.set_title(title)
    ax.set_xlabel("Velocidad (m/s)")
    ax.set_ylabel("Número de muestras")
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    for b, v in zip(bars, counts):
        if v > 0:
            pct = 100.0 * v / total
            ax.annotate(f"{v}  ({pct:.1f}%)",
                        (b.get_x() + b.get_width()/2, v),
                        ha='center', va='bottom', fontsize=8,
                        xytext=(0, 2), textcoords='offset points')

    fig.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"[OK] Guardado: {out_path}")
    plt.show()

def main():
    ap = argparse.ArgumentParser(description="Histograma de velocidades (columna 'speed') en 0..4 m/s paso 0.2")
    ap.add_argument("--pattern", default="/home/sergior/Downloads/pruebas/datasets/Deepracer_*/dataset.csv",
                    help="Patrón glob para localizar CSV. Ej: ../datasets/Deepracer_*/dataset.csv")
    ap.add_argument("--save", default=None,
                    help="Ruta para guardar PNG. Si no se da, solo muestra en pantalla.")
    ap.add_argument("--col", default="speed",
                    help="Nombre de la columna de velocidad (por defecto 'speed').")
    ap.add_argument("--vmin", type=float, default=0.0, help="Mínimo del rango (por defecto 0.0).")
    ap.add_argument("--vmax", type=float, default=4.0, help="Máximo del rango (por defecto 4.0).")
    ap.add_argument("--step", type=float, default=0.2,
                    help="Tamaño del bin (por defecto 0.2).")

    args = ap.parse_args()

    csv_paths = sorted(glob.glob(args.pattern))
    if not csv_paths:
        print(f"[WARN] No se encontraron CSV con el patrón: {args.pattern}")
        return

    speed_arrays = []
    print("[INFO] Leyendo velocidades:")
    for p in csv_paths:
        spd = cargar_speed(p, col=args.col)
        if spd is None:
            continue
        # Filtra al rango deseado (incluye el extremo superior).
        spd = spd[(spd >= args.vmin) & (spd <= args.vmax)]
        print(f"  {os.path.basename(os.path.dirname(p))}: {len(spd)} muestras válidas")
        speed_arrays.append(spd)

    # Sumar un pequeño epsilon para evitar perder el último borde por redondeo.
    edges = np.arange(args.vmin, args.vmax + args.step/2, args.step)

    counts, edges = hist_acumulado(speed_arrays, edges=edges)

    print("\n[RESUMEN] Conteos por bin:")
    labs = etiquetas_bins(edges)
    for lab, c in zip(labs, counts):
        print(f"  {lab} m/s: {c}")

    plot_barras_speed(
        counts, edges,
        title=f"Histograma de velocidades {args.vmin}–{args.vmax} m/s (paso {args.step})",
        out_path=args.save
    )

if __name__ == "__main__":
    main()
