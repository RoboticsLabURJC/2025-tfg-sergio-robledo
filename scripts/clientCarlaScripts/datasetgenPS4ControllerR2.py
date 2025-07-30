import carla
import time
import pygame
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import deque
from threading import Lock
import os
import csv
from datetime import datetime
import socket
import select

# Variables globales
current_steer = 0.0
current_throttle = 0.0
current_brake = 0.0

# Mutex
image_queue = deque(maxlen=1)
control_queue = deque(maxlen=1)

queue_lock = Lock()

MAX_IMAGE_AGE_MS = 150

# Crear carpetas de dataset
currtime = str(int(time.time() * 1000))
DATASET_ID = "Deepracer_BaseMap_" + currtime
BASE_DIR = "dataset"
RGB_DIR = os.path.join(DATASET_ID, "rgb")
MASK_DIR = os.path.join(DATASET_ID, "masks")
CSV_PATH = os.path.join(DATASET_ID, f"dataset.csv")

os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["rgb_path", "mask_path", "timestamp", "throttle", "steer", "brake", "speed", "heading"])

# Socket
HOST = 'localhost'
PORT = 1977
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
print(f"Waiting connection on {HOST}:{PORT}...")
conn, addr = sock.accept()
print(f"Connected from {addr}")

# Carla
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(5.0)
world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1.0 / 30.0
world.apply_settings(settings)

weather = carla.WeatherParameters(cloudiness=80.0, precipitation=0.0, sun_altitude_angle=90.0)
world.set_weather(weather)

bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.find('vehicle.finaldeepracer.aws_deepracer')

# -------------------------TRACK01-----------------------------
# spawn_point = carla.Transform(
#     carla.Location(x=3, y=-1, z=0.5),
#     carla.Rotation(yaw=-90)
# )

#-------------------------TRACK02---------------------------------
# spawn_point = carla.Transform(
#     carla.Location(x=-3.7, y=-4, z=0.5),
#     carla.Rotation(yaw=-120)
# )


#-------------------------TRACK03---------------------------------
spawn_point = carla.Transform(
    carla.Location(x=-7, y=-15, z=0.5),
    carla.Rotation(yaw=-15)
)

#-------------------------TRACK04---------------------------------
# spawn_point = carla.Transform(
#     carla.Location(x=17, y=-4.2, z=0.5),
#     carla.Rotation(yaw=-15)
# )


vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

if not vehicle:
    print("Failed to spawn vehicle")
    exit()

print("Vehicle spawned")

# Cámara frontal (para guardar)
cam_front_bp = bp_lib.find('sensor.camera.rgb')
cam_front_bp.set_attribute('image_size_x', '800')
cam_front_bp.set_attribute('image_size_y', '600')
cam_front_bp.set_attribute('fov', '140')
cam_front_transform = carla.Transform(carla.Location(x=0.13, z=0.13), carla.Rotation(pitch=-30))
camera_front = world.spawn_actor(cam_front_bp, cam_front_transform, attach_to=vehicle)

# Cámara tercera persona (para visualizar)
WIDTH, HEIGHT = 800, 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DeepRacer - Manual Dataset")

cam_third_bp = bp_lib.find('sensor.camera.rgb')
cam_third_bp.set_attribute('image_size_x', str(WIDTH))
cam_third_bp.set_attribute('image_size_y', str(HEIGHT))
cam_third_bp.set_attribute('fov', '90')
cam_third_transform = carla.Transform(carla.Location(x=-1, z=0.5))
camera_third = world.spawn_actor(cam_third_bp, cam_third_transform, attach_to=vehicle)

# Callback imagen
camera_image_front = None
camera_image_third = None

def cam_front_callback(image):
    global camera_image_front
    with queue_lock:
        image_queue.clear()
        image_queue.append((int(time.time() * 1000), image))
        camera_image_front = image

def cam_third_callback(image):
    global camera_image_third
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    array = array[:, :, ::-1]
    camera_image_third = pygame.surfarray.make_surface(array.swapaxes(0, 1))

camera_front.listen(cam_front_callback)
camera_third.listen(cam_third_callback)

# Guardar

def guardar_dato(timestamp, rgb_img, mask_class_img, accel, steer, brake, speed, heading):
    rgb_name = f"{timestamp}_rgb_{DATASET_ID}.png"
    mask_name = f"{timestamp}_mask_{DATASET_ID}.png"
    rgb_path_rel = os.path.join("/rgb", rgb_name)
    mask_path_rel = os.path.join("/masks", mask_name)
    cv2.imwrite(os.path.join(RGB_DIR, rgb_name), rgb_img)
    cv2.imwrite(os.path.join(MASK_DIR, mask_name), cv2.cvtColor(mask_class_img, cv2.COLOR_RGB2BGR))
    with open(CSV_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([rgb_path_rel, mask_path_rel, timestamp, accel, steer, brake, speed, heading])

# Función de escalado

def scale_axis(value, axis_type):
    if axis_type in ('ABS_X', 'ABS_RX'):
        return max(-1.0, min(1.0, value / 32767.0))
    elif axis_type in ('ABS_Y', 'ABS_RY'):
        return max(-1.0, min(1.0, -value / 32767.0))  # <- Se invierte aquí
    return 0.0

# Loop
start_time = time.time() + 10
running = True

while running:
    world.tick()
    with queue_lock:
        if image_queue:
            ts, image = image_queue.popleft()
        else:
            image = None

    if image:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
        bgr = array[:, :, ::-1]
        rgb = bgr.copy()
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        lower_yellow = np.array([18, 50, 150])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        mask_class = np.zeros_like(mask_white)
        mask_class[mask_white > 0] = 1
        mask_class[mask_yellow > 0] = 2

        mask_rgb = np.zeros_like(rgb)
        mask_rgb[mask_class == 1] = [255, 255, 255]
        mask_rgb[mask_class == 2] = [255, 255, 0]

        ready = select.select([conn], [], [], 0)
        if ready[0]:
            try:
                buffer = conn.recv(1024).decode(errors='ignore').strip()
                local_steer, local_throttle = current_steer, current_throttle
                for line in buffer.splitlines():
                    if "[ABS_X]" in line and "[R2]" in line:
                        parts = line.strip().split("[ABS_X]")
                        if len(parts) > 1:
                            vals = parts[1].split("[R2]")
                            if len(vals) == 2:
                                val_x = int(vals[0].strip())
                                val_ry = int(vals[1].strip())

                                # Joystick izquierdo: steer de -1 a 1
                                local_steer = (val_x - 127) / 128.0
                                local_steer = max(-1.0, min(1.0, local_steer))

                                # Joystick derecho (arriba = 255): throttle de 0.0 a 1
                                local_throttle = max(0.0, min(0.5, (val_ry / 255.0)*0.5))

                current_steer = local_steer
                current_throttle = local_throttle

                with queue_lock:
                    control_queue.clear()
                    control_queue.append((current_steer, current_throttle, current_brake))

            except Exception as e:
                print("Joystick read error:", e)




        vehicle.apply_control(carla.VehicleControl(throttle=current_throttle, steer=current_steer))

        timestamp = int(datetime.utcnow().timestamp() * 1000)
        velocity = vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

        heading = vehicle.get_transform().rotation.yaw

        if time.time() > start_time and control_queue:
            with queue_lock:
                steer, throttle, brake = control_queue[0]
            print("Throttle: ",throttle, "Steer: ", steer)
            #guardar_dato(timestamp, array, mask_rgb, throttle, steer, brake, speed, heading)

        if camera_image_third:
            screen.blit(camera_image_third, (0, 0))
            pygame.display.flip()

        cv2.waitKey(1)

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
            break

# Cleanup
camera_front.stop()
camera_front.destroy()
camera_third.stop()
camera_third.destroy()
vehicle.destroy()
sock.close()
pygame.quit()
print("✅ Session finished")
