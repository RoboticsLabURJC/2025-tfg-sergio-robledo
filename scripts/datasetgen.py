import carla
import time
import pygame
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import deque
import os
import csv
from datetime import datetime




current_throttle = 0.5
current_steer = 0.0
current_brake = 0.0


currtime = str(int(time.time() * 1000))

DATASET_ID = "Deepracer_BaseMap_" + currtime
BASE_DIR = "dataset"
RGB_DIR = os.path.join(BASE_DIR, "rgb/rgb" + currtime)
MASK_DIR = os.path.join(BASE_DIR, "masks/mask" + currtime)
CSV_PATH = os.path.join(BASE_DIR, f"dataset_{DATASET_ID}.csv")

os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["rgb_path", "mask_path", "timestamp", "throttle", "steer", "brake", "speed", "heading"])



HOST = '127.0.0.1'
PORT = 2000
VEHICLE_MODEL = 'vehicle.finaldeepracer.aws_deepracer'


pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DeepRacer - RGB y Segmentación Semántica")


client = carla.Client(HOST, PORT)
client.set_timeout(5.0)
world = client.get_world()

#FPS
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
print("Clima establecido en 'Sunset'")


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


camera_rgb_bp = blueprint_library.find('sensor.camera.rgb')
camera_rgb_bp.set_attribute('image_size_x', str(WIDTH))
camera_rgb_bp.set_attribute('image_size_y', str(HEIGHT))
camera_rgb_bp.set_attribute('fov', '120')


transform_front = carla.Transform(carla.Location(x=0.13, z=0.13), carla.Rotation(pitch=-30))
transform_thirdpers = carla.Transform(carla.Location(x=-1, z=0.75))
camera_front = world.spawn_actor(camera_rgb_bp, transform_front, attach_to=vehicle)


def camera_callback(image):
    
    print(f"[Frame {image.frame}] timestamp: {image.timestamp:.5f}")

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    bgr = array[:, :, ::-1]
    rgb = bgr.copy()
    camera_img_front = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

     # === Segmentación por color ===
    lower_yellow = np.array([18, 50, 150])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # === Máscara de clases 0 (fondo), 1 (blanco), 2 (amarillo) ===
    mask_class = np.zeros_like(mask_white, dtype=np.uint8)
    mask_class[mask_white > 0] = 1
    mask_class[mask_yellow > 0] = 2

 
    mask_rgb = np.zeros_like(rgb)
    mask_rgb[mask_class == 1] = [255, 255, 255]  # blanco
    mask_rgb[mask_class == 2] = [255, 255, 0]    # amarillo

    # Recoger datos
    timestamp = int(datetime.utcnow().timestamp() * 1000)
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
    heading = transform.rotation.yaw

    guardar_dato(
        timestamp=timestamp,
        rgb_img=array,
        mask_class_img=mask_rgb,
        accel=current_throttle,
        steer=current_steer,
        brake=current_brake,
        speed=speed,
        heading=heading
    )

def guardar_dato(timestamp, rgb_img, mask_class_img, accel, steer, brake, speed, heading):
    rgb_name = f"{timestamp}_rgb_{DATASET_ID}.png"
    mask_name = f"{timestamp}_mask_{DATASET_ID}.png"

    rgb_path_rel = os.path.join("rgb/rgb" + currtime, rgb_name)
    mask_path_rel = os.path.join("masks/mask" + currtime, mask_name)

    cv2.imwrite(os.path.join(RGB_DIR, rgb_name), rgb_img)
    cv2.imwrite(os.path.join(MASK_DIR, mask_name), cv2.cvtColor(mask_class_img, cv2.COLOR_RGB2BGR))


    with open(CSV_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([rgb_path_rel, mask_path_rel, timestamp, accel, steer, brake, speed, heading])


camera_front.listen(camera_callback)


control = carla.VehicleControl()
running = True


vehicle.apply_control(carla.VehicleControl(throttle=current_throttle, steer=current_steer, brake=current_brake))


time.sleep(4)

initTime = time.time()

while running:
    
    t1 = time.time()
    world.tick()
    t2 = time.time()
    print(f"real time: {t2-t1}")
    if ((time.time() - initTime) > 3): break
    


camera_front.stop()
camera_front.destroy()
vehicle.destroy()
pygame.quit()
