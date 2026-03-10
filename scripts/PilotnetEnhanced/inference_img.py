#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from utils.pilotnet import PilotNet

parser = argparse.ArgumentParser()
parser.add_argument("--img", required=True, help="Ruta imagen máscara")
parser.add_argument("--model", required=True, help="Ruta modelo .pth")
parser.add_argument("--speed", type=float, default=0.0, help="Velocidad real m/s")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Modelo (4 canales: RGB máscara + speed)
image_shape = (66, 200, 4)
model = PilotNet(image_shape, num_labels=2).to(device)

state = torch.load(args.model, map_location=device)
model.load_state_dict(state)
model.eval()

# Transform 

infer_tf = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])


# Leer imagen tal cual (ya es máscara)

img = cv2.imread(args.img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask_img = Image.fromarray(img)

# Resize + normalize

x = infer_tf(mask_img).unsqueeze(0).to(device)  # (1,3,66,200)

# Canal velocidad 

speed_norm = float(np.clip(args.speed / 3.5, 0.0, 1.0))

speed_plane = torch.full((1,1,66,200),
                         speed_norm,
                         dtype=x.dtype,
                         device=device)

x4 = torch.cat([x, speed_plane], dim=1)  # (1,4,66,200)


# Inferencia

with torch.no_grad():
    out = model(x4)
    steer, throttle = out[0].tolist()

steer    = float(np.clip(steer,    -1.0, 1.0))
throttle = float(np.clip(throttle,  0.0, 0.95))

print("\nRESULTADO")
print(f"Speed real (m/s): {args.speed:.3f}")
print(f"Speed norm:       {speed_norm:.3f}")
print(f"Steer:            {steer:.6f}")
print(f"Throttle:         {throttle:.6f}")
print("----------\n")
