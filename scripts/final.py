import carla
import time
import pygame
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import deque


# Cambbbbiossss 222
# --- Inicialización de la gráfica ---
plt.ion()
fig, ax = plt.subplots()
history_len = 100  

steer_history = deque([0]*history_len, maxlen=history_len)
throttle_history = deque([0]*history_len, maxlen=history_len)
line1, = ax.plot(steer_history, label="Steer", color="blue")
line2, = ax.plot(throttle_history, label="Throttle", color="green")
ax.set_ylim(-1.1, 1.1)
ax.legend()
#..........................................
current_steer = 0.0
current_throttle = 0.0

last_error_steer = 0
Kp_steer = 0.03
Kd_steer = 0.01

last_error_throttle = 0
Kp_throttle = 0.005


# Configuración de conexión con CARLA
HOST = '127.0.0.1'
PORT = 2000
VEHICLE_MODEL = 'vehicle.finaldeepracer.aws_deepracer'

# Inicializa Pygame para mostrar la cámara RGB
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DeepRacer - RGB y Segmentación Semántica")

# Conectar con el servidor de CARLA
client = carla.Client(HOST, PORT)
client.set_timeout(5.0)
world = client.get_world()

# Configurar clima de atardecer 🌅
weather = carla.WeatherParameters(
    cloudiness=80.0,
    precipitation=0.0,
    sun_altitude_angle=90.0,
    fog_density=0.0,
    wetness=0.0
)
world.set_weather(weather)
print("🌅 Clima establecido en 'Sunset' (Atardecer)")

# Obtener el blueprint del vehículo
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find(VEHICLE_MODEL)

# Spawnear el vehículo
spawn_point = carla.Transform(
    carla.Location(x=3, y=-1, z=0.5),
    carla.Rotation(yaw=-90)
)
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    print("❌ Error al spawnear el vehículo")
    exit()
print(f"🚗 Vehículo {VEHICLE_MODEL} spawneado en {spawn_point.location}")

# Blueprint para cámara RGB
camera_rgb_bp = blueprint_library.find('sensor.camera.rgb')
camera_rgb_bp.set_attribute('image_size_x', str(WIDTH))
camera_rgb_bp.set_attribute('image_size_y', str(HEIGHT))
camera_rgb_bp.set_attribute('fov', '90')

transform_front = carla.Transform(carla.Location(x=0.13, z=0.13), carla.Rotation(pitch=-30))
transform_thirdpers = carla.Transform(carla.Location(x=-1, z=0.75))
camera_front = world.spawn_actor(camera_rgb_bp, transform_front, attach_to=vehicle)
time.sleep(1)
# Variables para mostrar imágenes
camera_image_rgb = None


def process_rgb(image):
    global camera_image_rgb
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    array = array[:, :, ::-1]
    camera_image_rgb = pygame.surfarray.make_surface(array.swapaxes(0, 1))

def process_image_front(image):
    global current_steer, current_throttle, camera_img_front, last_error_steer, last_error_throttle, camera_img_front
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

    # === Convertir a imagen RGB solo para mostrar ===
    mask_rgb = np.zeros_like(rgb)
    mask_rgb[mask_class == 1] = [255, 255, 255]  # blanco
    #mask_rgb[mask_class == 2] = [255, 255, 0]    # amarillo

   
    y = int(0.4 * image.height)
    row = mask_class[y]
    white_indices = np.where(row == 1)[0]

    center_x = None
    
    if len(white_indices) > 10:
        left = white_indices[0]
        right = white_indices[-1]
        center_x = (left + right) // 2

        if mask_class[y, center_x] != 1:
            cv2.circle(mask_rgb, (center_x, y), 4, (0, 0, 255), -1)

    cv2.line(mask_rgb, (0, y), (image.width - 1, y), (100, 100, 100), 1)
    image_center_x = image.width // 2
    cv2.line(mask_rgb, (image_center_x, 0), (image_center_x, image.height), (128, 128, 128), 1)

    
    if center_x is not None:
        # Dibuja el centro detectado
        cv2.circle(mask_rgb, (center_x, image.height // 2), 5, (255, 0, 0), -1)

        # Calcular error y aplicar PID
        error = image_center_x - center_x
        error = -error  # invertimos el signo para que derecha = positivo

        print(f"Pixel offset = {error:.2f}")

        # PID simple (solo proporcional aquí)
        derivative = error - last_error_steer
        steer = Kp_steer * error + Kd_steer * derivative
        steer = np.clip(steer, -1.3, 1.0)
        last_error_steer = error

        abs_error = abs(error)
        last_error_throttle = abs_error
        throttle = 0.6 - Kp_throttle * abs_error
        throttle = np.clip(throttle, 0.2, 0.6)


        steer += 0.3
        current_steer = steer
        current_throttle = throttle
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
        print(f"[PID px] error={error:.1f}px, steer={steer:.3f}, throttle={throttle:.3f}")

    else:
        print("No se detectó centro")
        


    cv2.imshow("Máscara Segmentada", cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

# Vincular sensores
camera_front.listen(lambda image: process_image_front(image))
# Control del vehículo

control = carla.VehicleControl()
running = True
clock = pygame.time.Clock()

while running:
    clock.tick(30)
  
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

 
    if camera_image_rgb:
        screen.blit(camera_image_rgb, (0, 0))

    pygame.display.flip()
    steer_history.append(current_steer)
    throttle_history.append(current_throttle)

    line1.set_ydata(steer_history)
    line2.set_ydata(throttle_history)
    line1.set_xdata(range(len(steer_history)))
    line2.set_xdata(range(len(throttle_history)))
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.001)
    

# Cleanup
camera_rgb.destroy()
vehicle.destroy()
pygame.quit()
