import carla
import time
import pygame
import numpy as np
import math
import cv2
import sys

# === Argumento de línea de comandos ===
map_index = 1
if len(sys.argv) > 1:
    try:
        map_index = int(sys.argv[1])
        if map_index < 1 or map_index > 6:
            print("⚠️ Índice fuera de rango. Usando 1 por defecto.")
            map_index = 1
    except ValueError:
        print("⚠️ Argumento inválido. Usando 1 por defecto.")

# === Configuración de conexión con CARLA ===
HOST = '127.0.0.1'
PORT = 2000
VEHICLE_MODEL = 'vehicle.finaldeepracer.aws_deepracer'

# Inicializa Pygame
pygame.init()
WIDTH, HEIGHT = 640, 480
window_front = pygame.display.set_mode((WIDTH, HEIGHT))
#pygame.display.set_caption("Cámara Frontal")

# Conexión a CARLA
client = carla.Client(HOST, PORT)
client.set_timeout(5.0)
world = client.get_world()

# Configurar el clima
weather = carla.WeatherParameters(
    cloudiness=80.0,
    precipitation=0.0,
    sun_altitude_angle=90.0,
    fog_density=0.0,
    wetness=0.0
)
world.set_weather(weather)

# Blueprint
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find(VEHICLE_MODEL)

# Lista de posiciones de spawn
spawn_locations = [
    carla.Transform(carla.Location(x=0.5, y=0, z=0.5), carla.Rotation(yaw=0)),
    carla.Transform(carla.Location(x=6, y=-2, z=0.5), carla.Rotation(yaw=45)),
    carla.Transform(carla.Location(x=13.5, y=-2.1, z=0.5), carla.Rotation(yaw=0)),
    carla.Transform(carla.Location(x=13, y=-12, z=0.5), carla.Rotation(yaw=-30)),
    carla.Transform(carla.Location(x=1.75, y=-11, z=0.5), carla.Rotation(yaw=45)),
    carla.Transform(carla.Location(x=4, y=-19, z=0.5), carla.Rotation(yaw=-40)),
]

# Spawneo del vehículo según índice (1-6)
spawn_point = spawn_locations[map_index - 1]
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    print("❌ No se pudo spawnear el vehículo.")
    exit()
print(f"🚗 Vehículo spawneado en {spawn_point.location}")

# Cámara
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(WIDTH))
camera_bp.set_attribute('image_size_y', str(HEIGHT))
camera_bp.set_attribute('fov', '110')

# Transforms de cámara
transform_front = carla.Transform(carla.Location(x=0.13, z=0.13), carla.Rotation(pitch=-30))
transform_thirdpers = carla.Transform(carla.Location(x=-1, z=0.75))
camera_front = world.spawn_actor(camera_bp, transform_front, attach_to=vehicle)
camera_thirdpers = world.spawn_actor(camera_bp, transform_thirdpers, attach_to=vehicle)

camera_img_front = None
camera_img_thirdpers = None
# funciona mas o menos
# === PID para STEER ===
# PID para STEER (más reactivo en curvas)
# last_error_steer = 0.0
# integral_steer = 0.0
# Kp_steer = 0.10
# Ki_steer = 0.001
# Kd_steer = 0.5

# # PID para THROTTLE (curvas bruscas = menos velocidad, rectas = más aceleración)
# last_error_throttle = 0.0
# Kp_throttle = 0.02
# Kd_throttle = 0.12


# PID para STEER
# PID para STEER
# PID para STEER
# PID para STEER (más reactivo a curvas)
last_error_steer = 0.0
integral_steer = 0.0
Kp_steer = 0.18     # aumenta para reaccionar más directamente al error
Ki_steer = 0.0003   # se mantiene estable
Kd_steer = 0.45     # mayor corrección si cambia rápidamente el error

# PID para THROTTLE (reduce más en curvas pronunciadas)
last_error_throttle = 0.0
Kp_throttle = 0.01   # reduce más velocidad en curvas
Kd_throttle = 0.02   # frena aún más si el error cambia bruscamente



def map_virtual_to_real_throttle(virtual_throttle):
    if virtual_throttle <= 0.6:
        return np.interp(virtual_throttle, [0.0, 0.6], [0.0, 0.14])
    else:
        return np.interp(virtual_throttle, [0.6, 1.0], [0.14, 0.54])

def process_image_front(image):
    global camera_img_front, last_error_steer, integral_steer, last_error_throttle
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    bgr = array[:, :, ::-1]
    rgb = bgr.copy()
    camera_img_front = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    lower_yellow = np.array([18, 50, 150])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    mask_combined = cv2.bitwise_or(mask_yellow, mask_white)
    result = cv2.bitwise_and(rgb, rgb, mask=mask_combined)

    height, width = mask_combined.shape
    third_w = width // 3
    roi_mask = mask_combined[:, third_w:2*third_w]
    nonzero = cv2.findNonZero(roi_mask)

    if nonzero is not None:
        mean = np.mean(nonzero, axis=0)[0]
        line_center_x = mean[0] + third_w
        image_center_x = width / 2
        error = image_center_x - line_center_x

        integral_steer += error
        integral_steer = np.clip(integral_steer, -1, 1)
        derivative_steer = error - last_error_steer
        steer = Kp_steer * error + Ki_steer * integral_steer + Kd_steer * derivative_steer
        steer = np.clip(steer, -1.0, 1.0)
        last_error_steer = error

        abs_error = abs(error)
        derivative_throttle = abs_error - last_error_throttle
        last_error_throttle = abs_error
        virtual_throttle = 1.0 - (Kp_throttle * abs_error + Kd_throttle * derivative_throttle)
        virtual_throttle = np.clip(virtual_throttle, 0.5, 1.0)

        real_throttle = map_virtual_to_real_throttle(virtual_throttle)
        control = carla.VehicleControl(throttle=real_throttle, steer=steer)
        vehicle.apply_control(control)

        print(f"[PID Steer] error={error:.2f}, steer={steer:.3f}, throttle_virtual={virtual_throttle:.2f}, throttle_real={real_throttle:.3f}")
    else:
        print("⚠️ Línea no detectada. Frenando.")
        vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0))
        last_error_steer = 0
        integral_steer = 0

    #cv2.imshow("RGB Frontal", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    #cv2.imshow("Máscara Líneas", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

def process_image_thirdpers(image):
    global camera_img_thirdpers
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    camera_img_thirdpers = array

camera_front.listen(lambda image: process_image_front(image))
camera_thirdpers.listen(lambda image: process_image_thirdpers(image))

running = True
clock = pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    if camera_img_front:
        window_front.blit(camera_img_front, (0, 0))
        pygame.display.update()

    if camera_img_thirdpers is not None:
        big_view = cv2.resize(camera_img_thirdpers, (1200, 960))
        #cv2.imshow("Cámara Tercera Persona", big_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

    clock.tick(30)

print("🛑 Finalizando conducción autónoma.")
vehicle.apply_control(carla.VehicleControl(brake=1.0))
camera_front.destroy()
camera_thirdpers.destroy()
vehicle.destroy()
pygame.quit()
cv2.destroyAllWindows()
