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
transform_front = carla.Transform(carla.Location(x=0.13, z=0.13))
transform_thirdpers = carla.Transform(carla.Location(x=-1, z=0.75))

camera_front = world.spawn_actor(camera_bp, transform_front, attach_to=vehicle)
camera_thirdpers = world.spawn_actor(camera_bp, transform_thirdpers, attach_to=vehicle)

camera_img_front = None
camera_img_thirdpers = None

def process_image_front(image):
    global camera_img_front
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    array = array[:, :, ::-1]
    camera_img_front = pygame.surfarray.make_surface(array.swapaxes(0, 1))

def process_image_thirdpers(image):
    global camera_img_thirdpers
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    camera_img_thirdpers = array  # Para OpenCV (BGR)

camera_front.listen(lambda image: process_image_front(image))
camera_thirdpers.listen(lambda image: process_image_thirdpers(image))

# Waypoints definidos por el usuario
waypoints = [
    carla.Location(x=1.09, y=-0.39, z=0.01),
    carla.Location(x=2.84, y=-0.64, z=0.00),
    carla.Location(x=3.15, y=-1.11, z=0.00),
    carla.Location(x=3.33, y=-1.82, z=0.02),
    carla.Location(x=3.24, y=-2.50, z=0.00),
    carla.Location(x=2.99, y=-3.37, z=0.00),
    carla.Location(x=2.62, y=-4.79, z=0.03),
    carla.Location(x=2.38, y=-6.35, z=0.02),
    carla.Location(x=2.15, y=-6.72, z=0.01),
    carla.Location(x=1.61, y=-6.98, z=0.00),
    carla.Location(x=0.61, y=-6.26, z=0.03),
    carla.Location(x=0.44, y=-5.38, z=0.01),
    carla.Location(x=0.43, y=-4.15, z=0.04),
    carla.Location(x=0.42, y=-1.75, z=0.00),
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

# Loop principal
running = True
clock = pygame.time.Clock()

while running and current_wp_index < len(waypoints):
    vehicle_transform = vehicle.get_transform()
    target = waypoints[current_wp_index]
    loc = vehicle_transform.location
    print(f"📍 Ubicación actual: x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}")

    distance = loc.distance(target)

    if distance < TARGET_REACHED_DISTANCE:
        print(f"✅ Waypoint {current_wp_index + 1} alcanzado.")
        current_wp_index += 1
        continue

    steer = get_steering(vehicle_transform, target)
    control = carla.VehicleControl(throttle=0.4, steer=steer)
    vehicle.apply_control(control)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # Mostrar cámara frontal (pygame)
    if camera_img_front:
        window_front.blit(camera_img_front, (0, 0))
        pygame.display.update()

    # Mostrar cámara trasera (OpenCV)
    if camera_img_thirdpers is not None:
        cv2.imshow("Cámara Trasera", camera_img_thirdpers)

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
