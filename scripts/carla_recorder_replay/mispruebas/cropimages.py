#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from PIL import Image
import sys

# === Configuración del recorte vertical ===
FROM_Y = 226          # fila inicial (incluida)
TO_Y_INCLUSIVE = 436  # fila final (incluida)

def vertical_crop_keep_width(img: Image.Image, y0: int, y1_inc: int) -> Image.Image:
    w, h = img.size
    if y0 < 0 or y1_inc < 0:
        raise ValueError("Las coordenadas no pueden ser negativas.")
    if y0 > y1_inc:
        raise ValueError("FROM_Y no puede ser mayor que TO_Y_INCLUSIVE.")
    if y1_inc >= h:
        raise ValueError(f"TO_Y_INCLUSIVE ({y1_inc}) excede la altura de la imagen ({h}).")
    # En PIL, el límite inferior es EXCLUSIVO → +1
    box = (0, y0, w, y1_inc + 1)
    return img.crop(box)

def process_png(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            cropped = vertical_crop_keep_width(im, FROM_Y, TO_Y_INCLUSIVE)
            # Guardar sobre el mismo archivo, preservando formato y metadatos básicos
            save_kwargs = {}
            if im.format == "PNG":
                # Evita compresión excesiva por defecto si quisieras ajustar: save_kwargs["optimize"] = True
                pass
            cropped.save(p, **save_kwargs)
        return True
    except Exception as e:
        print(f"[SKIP] {p}: {e}")
        return False

def main():
    root = Path("/home/sergior/Downloads/pruebas/datasets/validation/Deepracer_BaseMap_1758564213539/rgb").resolve()
    if not root.exists():
        print(f"ERROR: No existe el directorio {root}")
        sys.exit(1)

    png_files = list(root.rglob("*.png"))
    if not png_files:
        print("No se encontraron .png en el árbol de ../datasets")
        sys.exit(0)

    total = len(png_files)
    ok = 0
    print(f"Encontrados {total} PNG. Aplicando recorte vertical y sobrescribiendo...")

    for p in png_files:
        if process_png(p):
            ok += 1

    print(f"✅ Hecho. Procesadas correctamente: {ok}/{total}.")
    if ok < total:
        print(f"ℹ️  {total - ok} archivos se omitieron por errores o tamaño insuficiente.")

if __name__ == "__main__":
    main()
