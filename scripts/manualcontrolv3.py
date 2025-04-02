import carla
import time
import pygame
import numpy as np

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
    cloudiness=10.0,
    precipitation=0.0,
    sun_altitude_angle=10.0,
    fog_density=10.0,
    wetness=0.0
)
world.set_weather(weather)
print("🌅 Clima establecido en 'Sunset' (Atardecer)")

# Obtener el blueprint del vehículo
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find(VEHICLE_MODEL)

# Spawnear el vehículo
spawn_point = carla.Transform(carla.Location(x=0, y=0, z=0.5))
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

camera_rgb_transform = carla.Transform(carla.Location(x=-2, z=1))
camera_rgb = world.spawn_actor(camera_rgb_bp, camera_rgb_transform, attach_to=vehicle)


# Variables para mostrar imágenes
camera_image_rgb = None


def process_rgb(image):
    global camera_image_rgb
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    array = array[:, :, ::-1]
    camera_image_rgb = pygame.surfarray.make_surface(array.swapaxes(0, 1))


# Vincular sensores
camera_rgb.listen(lambda image: process_rgb(image))

# Control del vehículo
control = carla.VehicleControl()
running = True
while running:
    keys = pygame.key.get_pressed()
    control.throttle = min(control.throttle + 0.05, 1.0) if keys[pygame.K_w] else 0.0
    control.brake = min(control.brake + 0.1, 1.0) if keys[pygame.K_s] else 0.0
    if keys[pygame.K_a]:
        control.steer = max(control.steer - 0.05, -1.0)
    elif keys[pygame.K_d]:
        control.steer = min(control.steer + 0.05, 1.0)
    else:
        control.steer = 0.0
    control.hand_brake = keys[pygame.K_SPACE]

    vehicle.apply_control(control)

    location = vehicle.get_transform().location
    print(f"📍 Posición: x={location.x:.2f}, y={location.y:.2f}, z={location.z:.2f}")

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    # Mostrar cámara RGB o segmentación (puedes alternar entre ambas si lo deseas)
    if camera_image_rgb:
        screen.blit(camera_image_rgb, (0, 0))
    # Si prefieres mostrar la segmentación en lugar del RGB, usa esta línea:
    pygame.display.flip()
    time.sleep(0.05)

# Cleanup
camera_rgb.destroy()
vehicle.destroy()
pygame.quit()
