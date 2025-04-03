import carla
import time
import pygame
import numpy as np
import math
import cv2

# Configuración de conexión con CARLA
HOST = '127.0.0.1'
PORT = 2000
VEHICLE_MODEL = 'vehicle.finaldeepracer.aws_deepracer'

# Inicializa Pygame para mostrar la cámara frontal
pygame.init()
WIDTH, HEIGHT = 640, 480
window_front = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cámara Frontal")

# Conexión a CARLA
client = carla.Client(HOST, PORT)
client.set_timeout(5.0)
world = client.get_world()

# Configurar el clima
weather = carla.WeatherParameters(
    cloudiness=10.0,
    precipitation=0.0,
    sun_altitude_angle=10.0,
    fog_density=0.0,
    wetness=0.0
)
world.set_weather(weather)
print("🌅 Clima establecido en 'Sunset' (Atardecer)")

# Blueprint y spawn
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find(VEHICLE_MODEL)
spawn_point = carla.Transform(carla.Location(x=0, y=0, z=0.5))
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    print("Error: No se pudo spawnear el vehículo.")
    exit()
print(f"🚗 Vehículo spawneado en {spawn_point.location}")

# Blueprint de cámara
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(WIDTH))
camera_bp.set_attribute('image_size_y', str(HEIGHT))
camera_bp.set_attribute('fov', '110')

# Cámaras
transform_front = carla.Transform(
    carla.Location(x=0.13, z=0.13),
    carla.Rotation(pitch=-30)  # 30 grados hacia abajo
)

transform_thirdpers = carla.Transform(carla.Location(x=-1, z=0.75))

camera_front = world.spawn_actor(camera_bp, transform_front, attach_to=vehicle)
camera_thirdpers = world.spawn_actor(camera_bp, transform_thirdpers, attach_to=vehicle)

camera_img_front = None
camera_img_thirdpers = None

def process_image_front(image):
    global camera_img_front
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    bgr = array[:, :, ::-1]  # Convertimos de BGR a RGB
    rgb = bgr.copy()

    # Mostrar en Pygame (ventana frontal)
    camera_img_front = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    # 🎨 Rango para amarillo (en RGB)
    lower_yellow = np.array([200, 200, 0])   # R, G altos; B bajo
    upper_yellow = np.array([255, 255, 150])

    # 🎨 Rango para blanco (en RGB)
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])

    # Crear máscaras
    mask_yellow = cv2.inRange(rgb, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(rgb, lower_white, upper_white)

    # Combinar máscaras
    mask_combined = cv2.bitwise_or(mask_yellow, mask_white)

    # Aplicar máscara a imagen original
    result = cv2.bitwise_and(rgb, rgb, mask=mask_combined)

    # 🪟 Mostrar imagen frontal original (convertida a BGR para que se vea bien en OpenCV)
    cv2.imshow("Imagen RGB - Cámara Frontal", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # 🪟 Mostrar la imagen con solo líneas detectadas
    cv2.imshow("Líneas Detectadas (Amarillo y Blanco)", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))



    


def process_image_thirdpers(image):
    global camera_img_thirdpers
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    camera_img_thirdpers = array  # Para OpenCV (BGR)

camera_front.listen(lambda image: process_image_front(image))
camera_thirdpers.listen(lambda image: process_image_thirdpers(image))

# Waypoints definidos por el usuario
waypoints = [

    carla.Location(x=2.54, y=-4.40, z=0.00),
    carla.Location(x=2.77, y=-6.18, z=0.00),
    carla.Location(x=1.67, y=-7.39, z=0.00),
    carla.Location(x=0.10, y=-4.62, z=0.00),
    carla.Location(x=0.29, y=0.53, z=0.00),

]

current_wp_index = 0
TARGET_REACHED_DISTANCE = 1.5

def get_steering(vehicle_transform, target_location):
    vehicle_location = vehicle_transform.location
    vehicle_forward = vehicle_transform.get_forward_vector()
    direction_vector = target_location - vehicle_location

    dot = vehicle_forward.x * direction_vector.x + vehicle_forward.y * direction_vector.y
    det = vehicle_forward.x * direction_vector.y - vehicle_forward.y * direction_vector.x
    angle = math.atan2(det, dot)
    return max(min(angle, 1.0), -1.0)

# Variables para suavizar dirección
previous_steer = 0.0
STEER_SMOOTHING = 0.2  # cuanto más alto, más suave (0.0 a 1.0)

# Distancia más exigente para avanzar al siguiente waypoint
TARGET_REACHED_DISTANCE = 1.0

# Loop principal
running = True
clock = pygame.time.Clock()

while running and current_wp_index < len(waypoints):
    vehicle_transform = vehicle.get_transform()
    loc = vehicle_transform.location
    target = waypoints[current_wp_index]
    distance = loc.distance(target)

    print(f"📍 Ubicación: x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f} | Objetivo: {current_wp_index + 1}/{len(waypoints)} | Distancia: {distance:.2f}m")

    if distance < TARGET_REACHED_DISTANCE and current_wp_index < len(waypoints) - 1:
        current_wp_index += 1
        target = waypoints[current_wp_index]

    # Calcular dirección deseada
    steer_raw = get_steering(vehicle_transform, target)
    steer = previous_steer * (1 - STEER_SMOOTHING) + steer_raw * STEER_SMOOTHING
    previous_steer = steer

    # Aceleración variable según curvatura
    throttle = 0.5 if abs(steer) < 0.3 else 0.35

    # Aplicar control
    control = carla.VehicleControl(throttle=throttle, steer=steer)
    vehicle.apply_control(control)

    # Eventos pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    # Mostrar cámara frontal (pygame)
    if camera_img_front:
        window_front.blit(camera_img_front, (0, 0))
        pygame.display.update()

    # Mostrar cámara tercera persona (OpenCV)
    if camera_img_thirdpers is not None:
        cv2.imshow("Cámara Tercera Persona", camera_img_thirdpers)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

    clock.tick(30)

# Cleanup
print("🛑 Finalizando conducción autónoma.")
vehicle.apply_control(carla.VehicleControl(brake=1.0))
camera_front.destroy()
camera_thirdpers.destroy()
vehicle.destroy()
pygame.quit()
cv2.destroyAllWindows()
