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

# Modelo CON velocidad (4 canales: 3 RGB + 1 speed)
MODEL_SPEED_PATH     = "experiments/exp_debug_1764593052/trained_models/pilot_net_model_best_123.pth"
IMAGE_SHAPE_SPEED    = (66, 200, 4)

SPEED_DIVISOR        = 3.5
DEVICE               = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Nombre del CSV de salida
OUT_CSV = "rmse_per_state_inference_speed.csv"

INFER_TF = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])


def load_model_speed():
    """Carga el modelo con velocidad (4 canales)."""
    model_speed = PilotNet(IMAGE_SHAPE_SPEED, num_labels=2)
    model_speed.load_state_dict(torch.load(MODEL_SPEED_PATH, map_location=DEVICE))
    model_speed.eval().to(DEVICE)
    return model_speed


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

    model_speed = load_model_speed()
    print("Modelo con velocidad cargado.")

    # Acumuladores MSE por estado (solo modelo con velocidad)
    # idx 0 → estado 1, idx 1 → estado 2, idx 2 → estado 3
    mse_speed      = np.zeros(3, dtype=np.float64)
    count_speed    = np.zeros(3, dtype=np.int64)

    for csv_path in csv_paths:
        base_dir = os.path.dirname(csv_path)
        df = pd.read_csv(csv_path)

        # Comprobamos columnas necesarias
        required_cols = {"rgb_path", "steer", "throttle", "estado"}
        if not required_cols.issubset(df.columns):
            print(f"[SKIP] {csv_path} no tiene columnas necesarias: {required_cols}")
            continue

        # Columna speed (para normalizar)
        if "speed" not in df.columns:
            print(f"[WARN] {csv_path} no tiene columna 'speed'. Se usará speed=0.0.")
            df["speed"] = 0.0

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

            
            # MODELO CON VELOCIDAD
            speed_raw = float(row["speed"])
            speed_norm = float(np.clip(speed_raw / SPEED_DIVISOR, 0.0, 1.0))

            # plano de velocidad (1,1,66,200)
            speed_plane = torch.full((1, 1, 66, 200), speed_norm,
                                     dtype=x3.dtype, device=x3.device)

            x4 = torch.cat([x3, speed_plane], dim=1)  # (1,4,66,200)

            with torch.no_grad():
                pred_speed = model_speed(x4)

            err_vec_sp = (pred_speed - gt) ** 2
            mse_sample_sp = float(err_vec_sp.mean().item())

            mse_speed[idx]   += mse_sample_sp
            count_speed[idx] += 1

    # Cálculo final de %Error
    estados = [1, 2, 3]
    rmse_pct_speed = []
    mse_final_speed = []
    counts_final = []

    for i, e in enumerate(estados):
        if count_speed[i] > 0:
            mse_speed[i] /= count_speed[i]
            rmse_sp = math.sqrt(mse_speed[i]) * 100.0
        else:
            mse_speed[i] = float("nan")
            rmse_sp = float("nan")

        mse_final_speed.append(mse_speed[i])
        rmse_pct_speed.append(rmse_sp)
        counts_final.append(int(count_speed[i]))

    print("\n=== (%)Error por estado (modelo con velocidad) ===")
    for i, e in enumerate(estados):
        print(f"Estado {e}: count = {counts_final[i]}, MSE = {mse_final_speed[i]:.6f}, %RMSE = {rmse_pct_speed[i]:.2f}%")

    out_df = pd.DataFrame({
        "estado": estados,
        "count_samples": counts_final,
        "mse_speed": mse_final_speed,
        "pct_rmse_speed": rmse_pct_speed
    })

    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Resultados guardados en {OUT_CSV}")


if __name__ == "__main__":
    main()
