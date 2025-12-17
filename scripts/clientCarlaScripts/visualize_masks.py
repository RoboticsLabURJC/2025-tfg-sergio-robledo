#!/usr/bin/env python3
import os
import glob
import cv2
import pandas as pd
import argparse
import sys

def parse_args():
    ap = argparse.ArgumentParser(
        description="Muestra todas las máscaras de todos los dataset.csv rápidamente una tras otra."
    )
    ap.add_argument(
        "--base-dir",
        default="../datasets",
        help="Directorio base donde están los datasets (por defecto ../datasets)."
    )
    ap.add_argument(
        "--delay-ms",
        type=int,
        default=30,
        help="Tiempo en ms que se muestra cada máscara (por defecto 30 ms)."
    )
    return ap.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.abspath(args.base_dir)

    print(f"[INFO] Base dir: {base_dir}")
    print(f"[INFO] Delay por imagen: {args.delay_ms} ms")

    # Busca todos los dataset.csv bajo base_dir
    pattern = os.path.join(base_dir, "**", "dataset.csv")
    csv_paths = sorted(glob.glob(pattern, recursive=True))

    if not csv_paths:
        print(f"[WARN] No se encontraron dataset.csv bajo {base_dir}")
        sys.exit(0)

    print(f"[INFO] Encontrados {len(csv_paths)} dataset.csv")

    cv2.namedWindow("mask_view", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("mask_view", 800, 600)

    total_imgs = 0

    for csv_path in csv_paths:
        folder = os.path.dirname(csv_path)
        print(f"\n[DATASET] {csv_path}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [ERROR] No se pudo leer {csv_path}: {e}")
            continue

        if "mask_path" not in df.columns:
            print("  [SKIP] No tiene columna 'mask_path'.")
            continue

        for idx, row in df.iterrows():
            rel_mask = str(row["mask_path"]).lstrip("/")
            abs_mask = os.path.join(folder, rel_mask)

            if not os.path.isfile(abs_mask):
                print(f"  [WARN] Falta máscara: {abs_mask}")
                continue

            img = cv2.imread(abs_mask, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  [WARN] No se pudo leer: {abs_mask}")
                continue

            total_imgs += 1
            # Mostrar en ventana
            cv2.imshow("mask_view", img)

            # Mostrar info en la barra de título
            cv2.setWindowTitle(
                "mask_view",
                f"Mask {total_imgs} - {os.path.basename(abs_mask)} (ESC para salir)"
            )

            key = cv2.waitKey(args.delay_ms) & 0xFF
            if key == 27:  # ESC
                print("\n[INFO] ESC pulsado. Saliendo.")
                cv2.destroyAllWindows()
                return

    print(f"\n[INFO] Fin. Mostradas {total_imgs} máscaras.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
