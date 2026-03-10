#------------------------------------------------
#Codigo para utilizar un controlador PD y desplazar el vehiculo en una pista
# PD para steering y P para throttle para recorrer una pista.
# Se emplea el modo sincrono y se fuerzan 30 Fps 
# #..........................................

import carla
import time
import pygame
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import deque

# --- Inicialización de la gráfica ---
# plt.ion()
# fig, ax = plt.subplots()
# history_len = 100  

# steer_history = deque([0]*history_len, maxlen=history_len)
# throttle_history = deque([0]*history_len, maxlen=history_len)
# line1, = ax.plot(steer_history, label="Steer", color="blue")
# line2, = ax.plot(throttle_history, label="Throttle", color="green")
# ax.set_ylim(-1.1, 1.1)
# ax.legend()
#..........................................
current_steer = 0.0
current_throttle = 0.0

last_error_steer = 0
Kp_steer = 0.1
Kd_steer = 0.00001

last_error_throttle = 0
Kp_throttle = 0.02


HOST = '127.0.0.1'
PORT = 2000
VEHICLE_MODEL = 'vehicle.finaldeepracer.aws_deepracer'


pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DeepRacer RGB y Segmentacion Semantica")


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
    carla.Location(x=3, y=-1, z=0.03),
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
camera_rgb_bp.set_attribute('fov', '140')


transform_front = carla.Transform(carla.Location(x=0.13, z=0.13), carla.Rotation(pitch=-30))
transform_thirdpers = carla.Transform(carla.Location(x=-1, z=0.75))
camera_front = world.spawn_actor(camera_rgb_bp, transform_front, attach_to=vehicle)

camera_image_rgb = None

vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0))
time.sleep(2)

def process_rgb(image):
    global camera_image_rgb
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    array = array[:, :, ::-1]
    camera_image_rgb = pygame.surfarray.make_surface(array.swapaxes(0, 1))

def process_image_front(image):

    print(f"[Frame {image.frame}] timestamp: {image.timestamp:.5f}")

    global current_steer, current_throttle, camera_img_front, last_error_steer, last_error_throttle, camera_img_front
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    bgr = array[:, :, ::-1]
    rgb = bgr.copy()
    camera_img_front = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # Segmentación por color
    lower_yellow = np.array([18, 50, 150])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Máscara de clases 0 (fondo), 1 (blanco), 2 (amarillo)
    mask_class = np.zeros_like(mask_white, dtype=np.uint8)
    mask_class[mask_white > 0] = 1
    mask_class[mask_yellow > 0] = 2

    # Convertir a imagen RGB solo para mostrar
    mask_rgb = np.zeros_like(rgb)
    mask_rgb[mask_class == 1] = [255, 255, 255]  # blanco
    mask_rgb[mask_class == 2] = [255, 255, 0]    # amarillo

   
    y = int(0.53 * image.height)
    row = mask_class[y]
    white_indices = np.where(row == 1)[0]

    center_x = None
    
    if len(white_indices) > 10:
        left = white_indices[0]
        right = white_indices[-1]
        center_x = (left + right) // 2

        if mask_class[y, center_x] != 1:
            cv2.circle(mask_rgb, (center_x, y), 4, (255, 0, 0), -1)

    cv2.line(mask_rgb, (0, y), (image.width - 1, y), (100, 100, 100), 1)
    image_center_x = image.width // 2
    cv2.line(mask_rgb, (image_center_x, 0), (image_center_x, image.height), (128, 128, 128), 1)

    
    if center_x is not None:
  
        error = - 100*(image_center_x -center_x) / image_center_x

        
        derivative = error - last_error_steer
        steer = Kp_steer * error + Kd_steer * derivative
   
        steer = np.clip(steer, -1.0, 1.0)
        last_error_steer = error

        abs_error = abs(error)
        last_error_throttle = abs_error
        throttle = 0.5 - Kp_throttle * abs_error
        throttle = np.clip(throttle, 0.2, 0.5)

    
        current_steer = steer
        current_throttle = throttle
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
        #print(f"[PID px] error={error:.1f}px, steer={steer:.3f}, throttle={throttle:.3f}")
        # v = vehicle.get_velocity()
        # speed = (v.x**2 + v.y**2 + v.z**2) ** 0.5
        # print(f"Velocidad: {speed:.2f} m/s")

    else:
        print("No se detectó centro")
        

    cv2.imshow("Mascara Segmentada", cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

camera_front.listen(lambda image: process_image_front(image))


control = carla.VehicleControl()
running = True


time.sleep(2)

while running:

    t1 = time.time()
    world.tick()
    t2 = time.time()
    print(f"real time: {t2-t1}")

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
            
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

            camera_rgb.destroy()
            vehicle.destroy()
            pygame.quit()


    


