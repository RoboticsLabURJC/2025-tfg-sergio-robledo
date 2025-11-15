#!/usr/bin/env python3
import os
import glob
import math
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from utils.pilotnet import PilotNet 

DATASET_ROOT         = "../datasets"
PATTERN              = os.path.join(DATASET_ROOT, "Deepracer_BaseMap_*", "dataset.csv")

# Modelo SIN velocidad (3 canales-> solo RGB)
MODEL_NO_SPEED_PATH  = "experiments/exp_debug_1763213000/trained_models/pilot_net_model_best_123.pth"
IMAGE_SHAPE_NO_SPEED = (66, 200, 3)

DEVICE               = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Nombre del CSV de salida
OUT_CSV = "rmse_per_state_inference_no_speed.csv"

INFER_TF = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

def load_model_no_speed():
    """Carga el modelo sin velocidad (3 canales RGB)."""
    model_no_speed = PilotNet(IMAGE_SHAPE_NO_SPEED, num_labels=2)
    model_no_speed.load_state_dict(torch.load(MODEL_NO_SPEED_PATH, map_location=DEVICE))
    model_no_speed.eval().to(DEVICE)
    return model_no_speed


def estado_idx(e):
    """Mapea estado 1/2/3 → índice 0/1/2. Si no es 1..3, devuelve None."""
    if e in (1, 2, 3):
        return e - 1
    return None


def main():
    csv_paths = sorted(glob.glob(PATTERN))
    if not csv_paths:
        print("No se encontraron dataset.csv con el patrón:", PATTERN)
        return

    print("Encontrados CSV:")
    for p in csv_paths:
        print("  ", p)

    model_no_speed = load_model_no_speed()
    print("Modelo SIN velocidad cargado.")

    # Acumuladores MSE por estado (solo modelo sin velocidad)
    # idx 0 → estado 1, idx 1 → estado 2, idx 2 → estado 3
    mse_no_speed   = np.zeros(3, dtype=np.float64)
    count_no_speed = np.zeros(3, dtype=np.int64)

    for csv_path in csv_paths:
        base_dir = os.path.dirname(csv_path)
        df = pd.read_csv(csv_path)

        # Comprobamos columnas necesarias
        required_cols = {"rgb_path", "steer", "throttle", "estado"}
        if not required_cols.issubset(df.columns):
            print(f"[SKIP] {csv_path} no tiene columnas necesarias: {required_cols}")
            continue

        print(f"Procesando {csv_path} con {len(df)} filas...")

        for _, row in df.iterrows():
            # Estado
            try:
                e = int(row["estado"])
            except Exception:
                continue

            idx = estado_idx(e)
            if idx is None:
                continue

            # Ruta a la imagen RGB
            rgb_rel = row["rgb_path"]
            rgb_abs = os.path.join(base_dir, rgb_rel.lstrip("/"))
            if not os.path.isfile(rgb_abs):
                continue

            # === Imagen 3 canales ===
            img = Image.open(rgb_abs).convert("RGB")
            x3 = INFER_TF(img).unsqueeze(0).to(DEVICE)  # (1,3,66,200)

            # Ground truth
            gt_steer    = float(row["steer"])
            gt_throttle = float(row["throttle"])
            gt = torch.tensor([[gt_steer, gt_throttle]], dtype=torch.float32, device=DEVICE)

            # MODELO SIN VELOCIDAD
            with torch.no_grad():
                pred_no_speed = model_no_speed(x3)

            err_vec = (pred_no_speed - gt) ** 2
            mse_sample = float(err_vec.mean().item())

            mse_no_speed[idx]   += mse_sample
            count_no_speed[idx] += 1

  
    # Cálculo final de %Error
    estados = [1, 2, 3]
    rmse_pct_no_speed = []
    mse_final_no_speed = []
    counts_final = []

    for i, e in enumerate(estados):
        if count_no_speed[i] > 0:
            mse_no_speed[i] /= count_no_speed[i]
            rmse_no = math.sqrt(mse_no_speed[i]) * 100.0
        else:
            mse_no_speed[i] = float("nan")
            rmse_no = float("nan")

        mse_final_no_speed.append(mse_no_speed[i])
        rmse_pct_no_speed.append(rmse_no)
        counts_final.append(int(count_no_speed[i]))

    print("\n=== (%)Error por estado (modelo SIN velocidad) ===")
    for i, e in enumerate(estados):
        print(f"Estado {e}: count = {counts_final[i]}, MSE = {mse_final_no_speed[i]:.6f}, Error = {rmse_pct_no_speed[i]:.2f}%")

    # ===========================
    # Guardar en CSV
    out_df = pd.DataFrame({
        "estado": estados,
        "count_samples": counts_final,
        "mse_no_speed": mse_final_no_speed,
        "pct_rmse_no_speed": rmse_pct_no_speed
    })

    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Resultados guardados en {OUT_CSV}")


if __name__ == "__main__":
    main()
