#!/usr/bin/env python3
import os
import glob
from PIL import Image

ROOT = "../datasets/test"              # Raíz que contiene Deepracer_BaseMap_*/
INPLACE = False                        # True: sobrescribe en rgb/ ; False: guarda en rgb_cropped/
DRY_RUN = False                        # True: no guarda, solo muestra lo que haría
EXTS = ["*.png", "*.jpg", "*.jpeg"]    # Extensiones a procesar

# Opción 1: recorte vertical (toda la anchura)
Y0 = 100
Y1 = 1000

# Opción 2 (tiene prioridad si está completa): recorte por caja (x0, y0, x1, y1)
X0 = None
X1 = None
BOX_Y0 = None
BOX_Y1 = None
# =========================


def build_crop_box(img_w, img_h):
    """Devuelve la caja (x0, y0, x1, y1) para recortar.
       Prioridad: caja completa si está definida; si no, recorte vertical."""
    global X0, X1, BOX_Y0, BOX_Y1, Y0, Y1

    if None not in (X0, X1, BOX_Y0, BOX_Y1):
        x0, x1 = X0, X1
        y0, y1 = BOX_Y0, BOX_Y1
    elif Y0 is not None and Y1 is not None:
        x0, x1 = 0, img_w
        y0, y1 = Y0, Y1
    else:
        raise ValueError("Define Y0/Y1 (recorte vertical) o bien X0/X1/BOX_Y0/BOX_Y1 (caja completa).")

    # Limitar a tamaño de imagen
    x0 = max(0, min(img_w, x0))
    x1 = max(0, min(img_w, x1))
    y0 = max(0, min(img_h, y0))
    y1 = max(0, min(img_h, y1))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Caja inválida tras clamp: ({x0},{y0})-({x1},{y1})")
    return (x0, y0, x1, y1)


def main():
    # Buscar datasets: /datasets/Deepracer_BaseMap_*/
    dataset_dirs = sorted(glob.glob(os.path.join(ROOT, "Deepracer_BaseMap_*")))
    if not dataset_dirs:
        print(f"No se encontraron carpetas Deepracer_BaseMap_* en {ROOT}")
        return

    total = 0
    for dset in dataset_dirs:
        rgb_dir = os.path.join(dset, "rgb")
        if not os.path.isdir(rgb_dir):
            print(f"[AVISO] Sin carpeta rgb: {rgb_dir}")
            continue

        out_dir = rgb_dir if INPLACE else os.path.join(dset, "rgb_cropped")
        if not DRY_RUN:
            os.makedirs(out_dir, exist_ok=True)

        # Reunir imágenes
        img_paths = []
        for patt in EXTS:
            img_paths.extend(glob.glob(os.path.join(rgb_dir, patt)))
        img_paths.sort()

        if not img_paths:
            print(f"[AVISO] No hay imágenes en {rgb_dir}")
            continue

        print(f"Procesando {len(img_paths)} imágenes en {rgb_dir} {'INPLACE' if INPLACE else out_dir}")

        for ip in img_paths:
            try:
                with Image.open(ip) as im:
                    w, h = im.size
                    box = build_crop_box(w, h)
                    cropped = im.crop(box)

                    filename = os.path.basename(ip)
                    out_path = os.path.join(out_dir, filename)

                    if DRY_RUN:
                        print(f"[DRY] {ip} -> {out_path}  box={box}  orig=({w}x{h})  new=({cropped.size[0]}x{cropped.size[1]})")
                    else:
                        cropped.save(out_path)
                    total += 1
            except Exception as e:
                print(f"[ERROR] {ip}: {e}")

    print(f"Listo. Imágenes procesadas: {total}")


if __name__ == "__main__":
    main()
