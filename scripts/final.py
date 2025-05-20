import carla
import time
import pygame
import numpy as np
import math
import cv2

#Cambiosssss nuevsvs

# Configuración de conexión con CARLA
HOST = '127.0.0.1'
PORT = 2000
VEHICLE_MODEL = 'vehicle.finaldeepracer.aws_deepracer'

# Inicializa Pygame
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

# Blueprint y spawn
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find(VEHICLE_MODEL)
spawn_point = carla.Transform(
    carla.Location(x=3, y=0, z=0.5),
    carla.Rotation(yaw=-90)
)
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    print("❌ No se pudo spawnear el vehículo.")
    exit()
print(f"🚗 Vehículo spawneado en {spawn_point.location}")

# Blueprint de cámara
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(WIDTH))
camera_bp.set_attribute('image_size_y', str(HEIGHT))
camera_bp.set_attribute('fov', '120')

# Cámaras
transform_front = carla.Transform(carla.Location(x=0.13, z=0.13), carla.Rotation(pitch=-30))
transform_thirdpers = carla.Transform(carla.Location(x=-1, z=0.75))
camera_front = world.spawn_actor(camera_bp, transform_front, attach_to=vehicle)
camera_thirdpers = world.spawn_actor(camera_bp, transform_thirdpers, attach_to=vehicle)

camera_img_front = None
camera_img_thirdpers = None

# === PID Steer directo en píxeles ===
last_error_steer = 0
last_error_throttle = 0
last_valid_error  = 0
# PID Steer ajustado para errores en píxeles (más reactivo)
# === PID Steer mucho más reactivo ===
Kp_steer = 1.3
Kd_steer = 1.5
Kp_throttle = 0.02   # proporcional: más error = menos throttle
Kd_throttle = 1.4   # derivativo: suaviza frenazos bruscos
previous_throttle = 0.5  # valor inicial

def process_image_front(image):
    global last_valid_error, camera_img_front, last_error_steer, integral_steer, last_error_throttle, previous_throttle
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

    heights = [int(0.45 * image.height), int(0.5 * image.height), int(0.55 * image.height)]
    #heights = [int(0.2 * image.height), int(0.30 * image.height), int(0.40 * image.height)]
    centers = []
    
    for y in heights:
        row = mask_class[y]
        white_indices = np.where(row == 1)[0]

        if len(white_indices) > 10:
            left = white_indices[0]
            right = white_indices[-1]

            center_x = (left + right) // 2

            if mask_class[y, center_x] != 1:
                centers.append((center_x, y))
                cv2.circle(mask_rgb, (center_x, y), 4, (0, 0, 255), -1)
    
                # Opcional: marcar extremos
                cv2.circle(mask_rgb, (left, y), 3, (0, 255, 0), -1)
                cv2.circle(mask_rgb, (right, y), 3, (0, 255, 0), -1)

        cv2.line(mask_rgb, (0, y), (image.width - 1, y), (100, 100, 100), 1)

    # Línea central vertical (siempre)
    image_center_x = image.width // 2
    cv2.line(mask_rgb, (image_center_x, 0), (image_center_x, image.height), (128, 128, 128), 1)

    # Calcular error si hay centros válidos
    
    mean_x = int(np.mean([pt[0] for pt in centers]))
    cv2.circle(mask_rgb, (mean_x, image.height // 2), 5, (255, 0, 0), -1)
    error = image_center_x - mean_x
    error = -error
    last_valid_error = error
    print(f"Pixel offset = {error:.2f}")

    # PID steer
    derivative_steer = error - last_error_steer
    last_error_steer = error

    steer = Kp_steer * error + Kd_steer * derivative_steer
    steer = np.clip(steer, -1.0, 1.0)

    #-----------------
    abs_error = abs(error)
    derivative_throttle = abs_error - last_error_throttle
    last_error_throttle = abs_error
    #print(raw_throttle)

    # invertir porque asi cuanto mas se separa de centro (curva ) menos velocidad

    throttle = 1 - ( Kp_throttle * abs_error + Kd_throttle * derivative_throttle)

    #alpha = 0.1
    #throttle = alpha * throttle + (1 - alpha) * previous_throttle

    #previous_throttle = throttle
    if (throttle > 0.75):

        throttle = 0.75

    if (throttle < 0.25):
        throttle = 0.25
            
    if centers:
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))


        #vehicle.apply_control(carla.VehicleControl(throttle=raw_throttle, steer=steer))

        print(f"[PID px] error={error:.1f}px, steer={steer:.3f}, throttle={throttle:.3f}")

    else:
        print("No se detectó centro")
        last_valid_error = 2*last_valid_error
  
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))


    cv2.imshow("Máscara Segmentada", cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)





def process_image_thirdpers(image):
    global camera_img_thirdpers
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    camera_img_thirdpers = array

# Listeners
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
        big_view = cv2.resize(camera_img_thirdpers, (1200, 960))
        # cv2.imshow("Cámara 3ª Persona", big_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

    clock.tick(30)

# Finalización
print("🛑 Finalizando conducción autónoma.")
vehicle.apply_control(carla.VehicleControl(brake=1.0))
camera_front.destroy()
camera_thirdpers.destroy()
vehicle.destroy()
pygame.quit()
cv2.destroyAllWindows()
