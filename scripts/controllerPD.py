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
    cloudiness=80.0,
    precipitation=0.0,
    sun_altitude_angle=90.0,
    fog_density=0.0,
    wetness=0.0
)
world.set_weather(weather)
print("☁️ Clima suave con luz cenital establecido.")

# Blueprint y spawn
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find(VEHICLE_MODEL)
spawn_point = carla.Transform(carla.Location(x=0.5, y=0.5, z=0.5))
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    print("❌ No se pudo spawnear el vehículo.")
    exit()
print(f"🚗 Vehículo spawneado en {spawn_point.location}")

# Blueprint de cámara
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(WIDTH))
camera_bp.set_attribute('image_size_y', str(HEIGHT))
camera_bp.set_attribute('fov', '110')

# Cámaras
transform_front = carla.Transform(carla.Location(x=0.13, z=0.13), carla.Rotation(pitch=-30))
transform_thirdpers = carla.Transform(carla.Location(x=-1, z=0.75))

camera_front = world.spawn_actor(camera_bp, transform_front, attach_to=vehicle)
camera_thirdpers = world.spawn_actor(camera_bp, transform_thirdpers, attach_to=vehicle)

camera_img_front = None
camera_img_thirdpers = None



# PD variables
last_error = 0.0
Kp = 0.03
Kd = 0.15

def process_image_front(image):
    global camera_img_front, last_error

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    rgb = array[:, :, ::-1]

    # Mostrar en Pygame
    camera_img_front = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    # 🎨 Filtro para blanco (RGB)
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])
    mask_white = cv2.inRange(rgb, lower_white, upper_white)

    # ROI (parte baja de la imagen)
    height, width = mask_white.shape
    roi = mask_white[int(height * 0.6):int(height * 0.9), :]

    # Detectar líneas blancas
    nonzero = cv2.findNonZero(roi)

    if nonzero is not None:
        mean = np.mean(nonzero, axis=0)[0]  # (x, y)
        line_center_x = mean[0]             # Promedio de todos los píxeles blancos
        image_center_x = width / 2
        error = image_center_x - line_center_x

        # Control PD
        derivative = error - last_error
        steer = Kp * error + Kd * derivative
        last_error = error

        # Limitar giro
        steer = np.clip(steer, -1.0, 1.0)

        # Aplicar control
        control = carla.VehicleControl(throttle=0.40, steer=steer)
        vehicle.apply_control(control)

        print(f"[PD Carril Blanco] error={error:.2f}, steer={steer:.3f}")
    else:
        print("⚠️ Líneas blancas no detectadas.")
        vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))

    # Mostrar imagen
    cv2.imshow("Líneas Blancas Detectadas", mask_white)
    cv2.imshow("RGB Frontal", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def process_image_thirdpers(image):
    global camera_img_thirdpers
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    camera_img_thirdpers = array

camera_front.listen(lambda image: process_image_front(image))
camera_thirdpers.listen(lambda image: process_image_thirdpers(image))

# Loop principal
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
