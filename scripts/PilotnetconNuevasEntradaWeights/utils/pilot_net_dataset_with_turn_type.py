from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import os, csv
import numpy as np


# ----------------------------
# CSV helpers
# ----------------------------
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


# ----------------------------
# Dataset: RGB + speed plane => 4 channels
# + lee tipo de vuelta (string) por fila
# ----------------------------
class PilotNetDatasetWithTurnType(Dataset):
    """
    Devuelve: (img4, y, turn_id)
      - img4: (4,66,200) RGB + plano speed
      - y: (2,) [steer, throttle]
      - turn_id: escalar long con id para la clase de giro
    """

    # Orden fijo para IDs
    TURN_CLASSES = [
        "STRAIGHT",
        "LEFT_LONG", "LEFT_MEDIUM", "LEFT_SHARP",
        "RIGHT_LONG", "RIGHT_MEDIUM", "RIGHT_SHARP",
        "UNLABELED",
    ]
    TURN_TO_ID = {name: i for i, name in enumerate(TURN_CLASSES)}

    def __init__(
        self,
        folder_paths,
        mirrored=False,
        transform=None,
        speed_norm_div=3.5,
        turn_col="turn_class",
        default_turn="UNLABELED",
        drop_unlabeled=False,
        drop_empty_turn=True,
    ):
        self.image_paths = []
        self.labels = []
        self.speeds = []
        self.turn_ids = []

        self.transform = transform
        self.mirrored = mirrored
        self.speed_norm_div = float(speed_norm_div)

        self.turn_col = str(turn_col)
        self.default_turn = str(default_turn)
        self.drop_unlabeled = bool(drop_unlabeled)
        self.drop_empty_turn = bool(drop_empty_turn)

        # Transform por defecto
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((66, 200)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ])

        # Validación de default_turn
        if self.default_turn not in self.TURN_TO_ID:
            raise ValueError(
                f"default_turn='{self.default_turn}' no está en TURN_CLASSES={self.TURN_CLASSES}"
            )

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

                fieldnames = list(reader.fieldnames)

                if self.turn_col not in fieldnames:
                    alias = "turnclass"
                    found = None
                    if alias in fieldnames:
                        found = alias
                        break
                    if found is None:
                        print(f"[WARN] No existe columna '{self.turn_col}' (ni alias) en {csv_path}. "
                              f"Se usará default_turn='{self.default_turn}'. Columnas: {fieldnames}")
                    else:
                        self.turn_col = found

                for row in reader:
                    mask_path = row.get("mask_path", "")
                    if not mask_path:
                        continue

                    img_rel = str(mask_path).lstrip("/")
                    img_abs = os.path.join(folder, img_rel)
                    if not os.path.isfile(img_abs):
                        continue

                    steer = _to_float(row.get("steer"))
                    throttle = _to_float(row.get("throttle"))
                    if steer is None or throttle is None:
                        continue

                    spd = _to_float(row.get("speed"))
                    if spd is None:
                        spd = 0.0
                    spd_norm = float(np.clip(spd / self.speed_norm_div, 0.0, 1.0))

                    # Leer turn label
                    raw_turn = (row.get(self.turn_col, "") or "").strip()
                    if raw_turn == "":
                        if self.drop_empty_turn:
                            continue
                        raw_turn = self.default_turn

                    turn = raw_turn.upper().replace(" ", "_")

                    if turn not in self.TURN_TO_ID:
                        # si hay labels inesperadas, se van a UNLABELED
                        turn = "UNLABELED"

                    if self.drop_unlabeled and turn == "UNLABELED":
                        continue

                    turn_id = int(self.TURN_TO_ID[turn])

                    # original
                    self.image_paths.append(img_abs)
                    self.labels.append([steer, throttle])
                    self.speeds.append(spd_norm)
                    self.turn_ids.append(turn_id)

                    # mirrored 
                    if self.mirrored:
                        self.image_paths.append((img_abs, "mirror"))
                        self.labels.append([-steer, throttle])  # espejo invierte steer
                        self.speeds.append(spd_norm)
                        self.turn_ids.append(turn_id)

        if len(self.image_paths) == 0:
            raise RuntimeError("No se encontraron muestras. Revisa rutas y estructura (dataset.csv + mask_path + steer/throttle).")

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

        # 4º canal: speed plane
        spd_norm = float(self.speeds[idx])
        speed_plane = torch.full((1, 66, 200), spd_norm, dtype=torch.float32)

        img4 = torch.cat([img, speed_plane], dim=0)  # (4,66,200)

        y = torch.tensor(self.labels[idx], dtype=torch.float32)         # (2,)
        turn_id = torch.tensor(self.turn_ids[idx], dtype=torch.long)    # escalar 0..7
        return img4, y, turn_id
