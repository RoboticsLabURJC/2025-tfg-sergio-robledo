from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import os, csv
import numpy as np


# ----------------------------
# CSV helpers

def _sniff_delimiter(csv_path: str) -> str:
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        return dialect.delimiter
    except Exception:
        return ";"


def _to_float(x):
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


def _to_int(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None
        
# ----------------------------
# Dataset: RGB + speed plane => 4 channels

class PilotNetDatasetWithEstado(Dataset):
    def __init__(self, folder_paths, mirrored=False, transform=None, speed_norm_div=3.5):
        self.image_paths = []
        self.labels = []
        self.speeds = []
        self.estados = []

        self.transform = transform
        self.mirrored = mirrored
        self.speed_norm_div = float(speed_norm_div)

        # Transform por defecto 
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((66, 200)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ])

        for folder in folder_paths:
            csv_path = os.path.join(folder, "dataset.csv")
            if not os.path.exists(csv_path):
                print(f"[WARN] No se encontró: {csv_path}")
                continue

            delim = _sniff_delimiter(csv_path)

            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f, delimiter=delim)
                if reader.fieldnames is None:
                    print(f"[WARN] CSV vacío/sin cabecera: {csv_path}")
                    continue

                # Nombres en el dataset: mask_path, steer, throttle, speed, estado
                for row in reader:
                    mask_path = row.get("mask_path", "")
                    if not mask_path:
                        continue

                    img_rel = str(mask_path).lstrip("/")
                    img_abs = os.path.join(folder, img_rel)
                    if not os.path.isfile(img_abs):
                        # Si faltan algunas máscaras, las saltamos
                        continue

                    steer = _to_float(row.get("steer"))
                    throttle = _to_float(row.get("throttle"))
                    if steer is None or throttle is None:
                        continue

                    spd = _to_float(row.get("speed"))
                    if spd is None:
                        spd = 0.0
                    spd_norm = float(np.clip(spd / self.speed_norm_div, 0.0, 1.0))

                    est = _to_int(row.get("estado"))
                    if est is None:
                        # si no existe estado, lo ponemos como 2 (centro) por defecto
                        est = 2
                    est = int(np.clip(est, 1, 3))

                    # original
                    self.image_paths.append(img_abs)
                    self.labels.append([steer, throttle])
                    self.speeds.append(spd_norm)
                    self.estados.append(est)

              

        if len(self.image_paths) == 0:
            raise RuntimeError("No se encontraron muestras. Revisa rutas y estructura (dataset.csv + mask_path).")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        entry = self.image_paths[idx]
        mirrored = isinstance(entry, tuple) and entry[1] == "mirror"
        img_path = entry[0] if isinstance(entry, tuple) else entry

        img = Image.open(img_path).convert("RGB")
        if mirrored:
            img = ImageOps.mirror(img)

        img = self.transform(img)  # (3,66,200)

        # 4º canal: speed
        spd_norm = float(self.speeds[idx])
        speed_plane = torch.full((1, 66, 200), spd_norm, dtype=torch.float32)

        img4 = torch.cat([img, speed_plane], dim=0)  # (4,66,200)

        y = torch.tensor(self.labels[idx], dtype=torch.float32)   # (2,)
        est = torch.tensor(self.estados[idx], dtype=torch.long)   # escalar 1/2/3
        return img4, y, est

