# === INFERENCIA AUTOMÁTICA EN CARLA CON PilotNet ===

import carla
import time
import pygame
import numpy as np
import cv2
import torch
from torchvision import transforms
from utils.pilotnet import PilotNet
from PIL import Image

# === Cargar modelo entrenado ===
MODEL_PATH = "experimentstrained_models/pilot_net_model_best_123.pth"
image_shape = (66, 200, 3)
model = PilotNet(image_shape, num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# === Inicialización de CARLA ===
HOST = '127.0.0.1'
PORT = 2000
VEHICLE_MODEL = 'vehicle.finaldeepracer.aws_deepracer'

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DeepRacer - Control automático")

client = carla.Client(HOST, PORT)
client.set_timeout(5.0)
world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1.0 / 30.0
world.apply_settings(settings)

weather = carla.WeatherParameters(sun_altitude_angle=90.0)
world.set_weather(weather)

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find(VEHICLE_MODEL)

spawn_point = carla.Transform(
    carla.Location(x=3, y=-1, z=0.5),
    carla.Rotation(yaw=-90)
)
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    print("Error al spawnear el vehículo")
    exit()
print(f"Vehículo {VEHICLE_MODEL} spawneado en {spawn_point.location}")

# === Crear cámara frontal ===
camera_rgb_bp = blueprint_library.find('sensor.camera.rgb')
camera_rgb_bp.set_attribute('image_size_x', str(WIDTH))
camera_rgb_bp.set_attribute('image_size_y', str(HEIGHT))
camera_rgb_bp.set_attribute('fov', '140')

transform_front = carla.Transform(carla.Location(x=0.13, z=0.13), carla.Rotation(pitch=-30))
camera_front = world.spawn_actor(camera_rgb_bp, transform_front, attach_to=vehicle)

# === Variables de control ===
current_steer = 0.0
current_throttle = 0.0
image_ready = [None]  # lista mutable como referencia

def camera_callback(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    image_ready[0] = array  # se actualiza la imagen disponible

camera_front.listen(camera_callback)

vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
time.sleep(4)

# === Bucle principal ===
running = True
while running:
    world.tick()

    rgb = image_ready[0]
    if rgb is not None:
        # Preprocesar imagen
        pil_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        pil_img = cv2.resize(pil_img, (200, 66))
        pil_img = Image.fromarray(pil_img)
        tensor_img = transform(pil_img).unsqueeze(0)

        # Inferencia
        with torch.no_grad():
            output = model(tensor_img)
            steer, throttle = output[0].tolist()

        # Aplicar control
        control = carla.VehicleControl(throttle=float(throttle), steer=float(steer))
        vehicle.apply_control(control)

        # Mostrar imagen
        camera_img_front = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        screen.blit(camera_img_front, (0, 0))
        pygame.display.flip()

    # Eventos de teclado
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
            break

# === Limpieza ===
settings = world.get_settings()
settings.synchronous_mode = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)
camera_front.stop()
camera_front.destroy()
vehicle.destroy()
pygame.quit()