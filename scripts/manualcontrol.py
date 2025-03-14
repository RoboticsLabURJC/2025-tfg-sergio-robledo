import carla
import time
import pygame
import numpy as np

# Configuración de conexión con CARLA
HOST = '127.0.0.1'
PORT = 2000
VEHICLE_MODEL = 'vehicle.deepracer.aws_deepracer'

# Inicializa Pygame para mostrar la cámara y capturar eventos de teclado
pygame.init()
WIDTH, HEIGHT = 1200, 1000
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DeepRacer - Conducción Manual")

# Conectar con el servidor de CARLA
client = carla.Client(HOST, PORT)
client.set_timeout(5.0)
world = client.get_world()

# Configurar el clima para que sea atardecer 🌅
weather = carla.WeatherParameters(
    cloudiness=10.0,  # Pocas nubes
    precipitation=0.0,  # Sin lluvia
    sun_altitude_angle=10.0,  # Posición del sol baja (atardecer)
    fog_density=10.0,  # Un poco de niebla para efecto de atardecer
    wetness=0.0  # Suelo seco
)
world.set_weather(weather)
print("🌅 Clima establecido en 'Sunset' (Atardecer)")

# Obtener el blueprint del vehículo
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find(VEHICLE_MODEL)

# Crear un spawn point en la ubicación deseada
#spawn_points = world.get_map().get_spawn_points()
#spawn_point = spawn_points[0]
spawn_point = carla.Transform(carla.Location(x=0, y=0, z=0.5))

# Spawnear el vehículo
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    print("Error: No se pudo spawnear el vehículo en la ubicación deseada")
    exit()
print(f"🚗 Vehículo {VEHICLE_MODEL} spawneado en {spawn_point.location}")

# Configurar la cámara en la parte delantera del vehículo
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(WIDTH))
camera_bp.set_attribute('image_size_y', str(HEIGHT))
camera_bp.set_attribute('fov', '110')  # Campo de visión amplio

camera_transform = carla.Transform(carla.Location(x=-3, z=2.0))  # Cámara en la parte delantera
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Variable para almacenar la última imagen capturada
camera_image = None

def process_image(image):
    """ Convierte la imagen de CARLA a formato de Pygame. """
    global camera_image
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]  # Ignorar el canal alfa
    array = array[:, :, ::-1]  # Convertir de BGRA a RGB
    camera_image = pygame.surfarray.make_surface(array.swapaxes(0, 1))

# Vincular la cámara para capturar imágenes
camera.listen(lambda image: process_image(image))

# Configurar control del vehículo
control = carla.VehicleControl()

# Configurar estado de la conducción
running = True
while running:
    keys = pygame.key.get_pressed()  # Obtener teclas presionadas
    
    # Acelerar (W)
    if keys[pygame.K_w]:
        control.throttle = min(control.throttle + 0.05, 1.0)
    else:
        control.throttle = 0.0  # Si no se presiona W, no acelera

    # Frenar (S)
    if keys[pygame.K_s]:
        control.brake = min(control.brake + 0.1, 1.0)
    else:
        control.brake = 0.0  # Si no se presiona S, no frena

    # Girar izquierda (A)
    if keys[pygame.K_a]:
        control.steer = max(control.steer - 0.05, -1.0)
    # Girar derecha (D)
    elif keys[pygame.K_d]:
        control.steer = min(control.steer + 0.05, 1.0)
    else:
        control.steer = 0.0  # Si no se presiona A o D, el coche sigue recto

    # Freno de mano (ESPACIO)
    control.hand_brake = keys[pygame.K_SPACE]

    # Aplicar control al vehículo
    vehicle.apply_control(control)

    # Capturar eventos de salida
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
    
    # Mostrar la imagen de la cámara
    if camera_image:
        screen.blit(camera_image, (0, 0))
    pygame.display.flip()
    time.sleep(0.05)  # Pequeño delay para estabilidad

# Finalizar el script y cerrar CARLA
camera.destroy()
vehicle.destroy()
pygame.quit()

