#------------------------------------------------
#Codigo para la inferencia automatica utilizando las imagenes de la mascara
# #..........................................

import carla
import time
import pygame
import numpy as np
import cv2
import torch
from torchvision import transforms
from utils.pilotnet import PilotNet
from PIL import Image

MODEL_PATH = "experiments/exp_debug_1758129759/trained_models/pilot_net_model_best_123.pth"
image_shape = (66, 200, 3)
model = PilotNet(image_shape, num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

HOST = '127.0.0.1'
PORT = 2000
VEHICLE_MODEL = 'vehicle.finaldeepracer.aws_deepracer'

pygame.init()

W, H = 800, 600
screen = pygame.display.set_mode((W*2, H))
pygame.display.set_caption("DeepRacer Inferencia MASK)")

client = carla.Client(HOST, PORT); client.set_timeout(5.0)
world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1.0 / 30.0
world.apply_settings(settings)

weather = carla.WeatherParameters(
    cloudiness=80.0,
    precipitation=0.0,
    sun_altitude_angle=90.0,
    fog_density=0.0,
    wetness=0.0
)
world.set_weather(weather)

bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.find(VEHICLE_MODEL)


#Track01
spawn_point = carla.Transform(
   carla.Location(x=3, y=-1, z=0.5),
   carla.Rotation(yaw=-90)
)

vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    raise RuntimeError("Error al spawnear el vehículo")
print(f"Vehículo {VEHICLE_MODEL} spawneado en {spawn_point.location}")

# Cámara frontal con la misma config que captura
cam_bp = bp_lib.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', str(W))
cam_bp.set_attribute('image_size_y', str(H))
cam_bp.set_attribute('fov', '140')

cam_tf = carla.Transform(carla.Location(x=0.13, z=0.13), carla.Rotation(pitch=-30))
camera_front = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

# Compartidos
frame_rgb = [None]      # RGB (como en captura)
frame_mask_rgb = [None] # Máscara coloreada (como en captura)

def camera_callback(image):

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]  # BGRA -> BGR
    bgr = array[:, :, ::-1] 
    rgb = bgr.copy()         

   
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([18, 50, 150])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    mask_class = np.zeros_like(mask_white, dtype=np.uint8)
    mask_class[mask_white > 0]  = 1
    mask_class[mask_yellow > 0] = 2

    mask_rgb = np.zeros_like(rgb)
    mask_rgb[mask_class == 1] = [255, 255, 255]  # blanco
    mask_rgb[mask_class == 2] = [255, 255,   0]  # amarillo

    frame_rgb[0] = rgb
    frame_mask_rgb[0] = mask_rgb


camera_front.listen(camera_callback)

vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
time.sleep(1.0)

running = True
while running:
    world.tick()

    rgb = frame_rgb[0]
    mask_rgb = frame_mask_rgb[0]

    if rgb is not None and mask_rgb is not None:
      
        pil_img = Image.fromarray(mask_rgb)
        tensor_img = transform(pil_img).unsqueeze(0)   # [1,3,66,200] normalizado

        with torch.no_grad():
            output = model(tensor_img)
            steer, throttle = output[0].tolist()

        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 0.8))
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
        print(f"steer={steer:+.3f} | thr={throttle:.3f}")

   
        surf_rgb  = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        surf_mask = pygame.surfarray.make_surface(mask_rgb.swapaxes(0, 1))

        surf_rgb  = pygame.transform.smoothscale(surf_rgb,  (W, H))
        surf_mask = pygame.transform.smoothscale(surf_mask, (W, H))

        screen.blit(surf_rgb,  (0,   0))
        screen.blit(surf_mask, (W,   0))
        pygame.display.flip()

    # Eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
            break

# Limpieza
settings = world.get_settings()
settings.synchronous_mode = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)
camera_front.stop()
camera_front.destroy()
vehicle.destroy()
pygame.quit()
