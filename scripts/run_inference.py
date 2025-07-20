import torch
from torchvision import transforms
from PIL import Image
from utils.pilotnet import PilotNet

# === CONFIGURACIÓN ===
MODEL_PATH = "experimentstrained_models/pilot_net_model_best_123.pth" 
IMAGE_PATH = "datasets/Deepracer_BaseMap_1751911588893/rgb/1751904402454_rgb_Deepracer_BaseMap_1751911588893.png"     # imagen real de tu dataset

# === Cargar modelo ===
image_shape = (66, 200, 3)
model = PilotNet(image_shape, num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# === Preprocesar imagen ===
transform = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image).unsqueeze(0)  # [1, 3, 66, 200]

# === Inferencia ===
with torch.no_grad():
    output = model(image)
    steer, throttle = output[0].tolist()

print(f"Predicción: 🧭 steer={steer:.4f}, ⚡ throttle={throttle:.4f}")
