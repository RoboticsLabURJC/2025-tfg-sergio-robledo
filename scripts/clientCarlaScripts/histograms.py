#!/usr/bin/env python3

#------------------------------------------------
#Codigo que muestra un histograma de los datos, concretamente del campo estado de los csv.
#De esta manera, lee la columna de estado y muestra del dataset los frames 1,2 y 3, cada uno en una barra.
# Es decir, la orientacion del coche en cada frame
#Por defecto busca ../datasets/Deepracer_*/dataset.csv y suma los datos de todos los datasets que cumplan este patron
#------------------------------------------------
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def contar_estados(csv_path: str):
    # Devuelve (c1, c2, c3) para 'estado' == 1/2/3 en el CSV dado.
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error al leer {csv_path}: {e}")
        return None

    if 'estado' not in df.columns:
        print(f"{csv_path} no tiene columna 'estado'.")
        return None

    est = pd.to_numeric(df['estado'], errors='coerce').astype('Int64')
    c1 = int((est == 1).sum())
    c2 = int((est == 2).sum())
    c3 = int((est == 3).sum())
    return c1, c2, c3

def plot_total(totales, title: str, out_path: str | None):
    etiquetas = ["1 (izq)", "2 (centro)", "3 (der)"]
    valores = [totales[1], totales[2], totales[3]]
    total = sum(valores) if sum(valores) > 0 else 1

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(etiquetas, valores)
    ax.set_title(title)
    ax.set_xlabel("estado")
    ax.set_ylabel("Número de muestras")

    # Guardar conteo y porcentaje
    for b, v in zip(bars, valores):
        if v > 0:
            porcentaje = 100.0 * v / total
            ax.annotate(f"{v}  ({porcentaje:.1f}%)",
                        (b.get_x() + b.get_width()/2, v),
                        ha='center', va='bottom', fontsize=9,
                        xytext=(0, 2), textcoords='offset points')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Guardado: {out_path}")
    plt.show()

def main():
    ap = argparse.ArgumentParser(
        description="Histograma de estados (1/2/3)"
    )
    ap.add_argument(
        "--pattern",
        default="../datasets/Deepracer_*/dataset.csv",
        help="Patrón de búsqueda de CSV (por defecto: ../datasets/Deepracer_*/dataset.csv)."
    )
    ap.add_argument("--save", default=None,
                    help="Ruta para guardar PNG. Si no se da, solo muestra en pantalla.")
    args = ap.parse_args()

    csv_paths = sorted(glob.glob(args.pattern))
    if not csv_paths:
        print(f"No se encontraron CSV con el patrón: {args.pattern}")
        return

    totales = {1: 0, 2: 0, 3: 0}
    print("Acumulando conteos por archivo:")
    for p in csv_paths:
        counts = contar_estados(p)
        if counts is None:
            continue
        c1, c2, c3 = counts
        name = os.path.basename(os.path.dirname(p))
        print(f"  {name}: estado1={c1}  estado2={c2}  estado3={c3}")
        totales[1] += c1
        totales[2] += c2
        totales[3] += c3

    print("\nTotales agregados en TODOS los directorios:")
    print(f"  estado 1: {totales[1]}")
    print(f"  estado 2: {totales[2]}")
    print(f"  estado 3: {totales[3]}")

    plot_total(totales, "Estados (1/2/3). Total agregado (todos los datasets)", out_path=args.save)

if __name__ == "__main__":
    main()
