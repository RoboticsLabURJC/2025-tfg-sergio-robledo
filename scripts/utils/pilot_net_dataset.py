import os
import csv
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

class PilotNetDataset(Dataset):
    def __init__(self, folder_paths, mirrored=False, transform=None, preprocessing=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.preprocessing = preprocessing
        self.mirrored = mirrored

        # Recorre todas las carpetas (cada una tiene su dataset.csv)
        for folder in folder_paths:
            csv_path = os.path.join(folder, "dataset.csv")
            rgb_dir = os.path.join(folder, "rgb")  # base para imágenes

            if not os.path.exists(csv_path):
                print(f"[Warning] No se encontró: {csv_path}")
                continue

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_rel_path = row["rgb_path"].lstrip("/")  # ruta relativa, e.g. rgb/frame_001.png
                    img_path = os.path.join(folder, img_rel_path)  # ruta completa

                    if os.path.isfile(img_path):
                        steer = float(row["steer"])
                        throttle = float(row["throttle"])

                        self.image_paths.append(img_path)
                        self.labels.append([steer, throttle])

                        # Si se quiere imagen espejada (mirroring)
                        if self.mirrored:
                            self.image_paths.append((img_path, 'mirror'))
                            self.labels.append([-steer, throttle])
                    else:
                        print(f"[Warning] Imagen no encontrada: {img_path}")

        self.image_shape = (66, 200, 3)  # height, width, channels
        self.num_labels = 2

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_entry = self.image_paths[idx]
        mirrored = False

        if isinstance(img_entry, tuple):
            img_path, mode = img_entry
            mirrored = (mode == 'mirror')
        else:
            img_path = img_entry

        image = Image.open(img_path).convert("RGB")
        image = image.resize((200, 66))  # tamaño esperado por PilotNet

        if self.transform:
            image = np.array(image)  # PIL → numpy
            image = torch.tensor(image, dtype=torch.float32) / 255.0
            image = image.permute(2, 0, 1)  # [H, W, C] → [C, H, W]
        else:
            image = transforms.ToTensor()(image)  # ⬅️ Convierte a Tensor si no hay transform

        if mirrored:
            image = torch.flip(image, dims=[2])  # horizontal flip

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label
