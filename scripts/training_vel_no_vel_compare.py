#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def read_pct_csv(path):
    if not os.path.isfile(path):
        print(f"[ERROR] No existe: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    # Columnas exactas
    required = ["train_pct_rmse", "val_pct_rmse", "test_pct_rmse"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas en {path}: {missing}")

    return {
        'train': float(df.loc[0, "train_pct_rmse"]),
        'val':   float(df.loc[0, "val_pct_rmse"]),
        'test':  float(df.loc[0, "test_pct_rmse"]),
    }

def main():
    ap = argparse.ArgumentParser(description="Gráfico comparativo %RMSE (con vs sin velocidad)")
    ap.add_argument("--csv_sin_vel", default="percent_rmse.csv",
                    help="CSV sin velocidad (por defecto: percent_rmse.csv)")
    ap.add_argument("--csv_con_vel", default="/home/sergior/Downloads/pruebas/PilotnetconNuevasEntradasValidationDir/percent_rmse_speed_label.csv",
                    help="CSV con velocidad (por defecto: /PilotnetconNuevasEntradasValidationDir/percent_rmse_speed_label.csv)")
    ap.add_argument("--title", default="%RMSE final - Train/Val/Test (con vs sin velocidad)")
    ap.add_argument("--out", default="comparativa_percent_rmse.png", help="Ruta de salida de la figura")
    args = ap.parse_args()

    sin = read_pct_csv(args.csv_sin_vel)
    con = read_pct_csv(args.csv_con_vel)

    print("==== %RMSE (sin velocidad) ====")
    print(f"Train: {sin['train']:.3f} | Val: {sin['val']:.3f} | Test: {sin['test']:.3f}")
    print("==== %RMSE (con velocidad) ====")
    print(f"Train: {con['train']:.3f} | Val: {con['val']:.3f} | Test: {con['test']:.3f}")

    categorias = ['Train', 'Validation', 'Test']
    y_sin = [sin['train'], sin['val'], sin['test']]
    y_con = [con['train'], con['val'], con['test']]

    x = np.arange(len(categorias))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    bars1 = ax.bar(x - width/2, y_sin, width, label="Sin velocidad")
    bars2 = ax.bar(x + width/2, y_con, width, label="Con velocidad")

    ax.set_title(args.title)
    ax.set_ylabel("% RMSE")
    ax.set_xticks(x, categorias)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Etiquetas numéricas
    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.2f}%",
                        xy=(b.get_x() + b.get_width()/2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"[OK] Figura guardada en: {args.out}")
    plt.show()

if __name__ == "__main__":
    main()
