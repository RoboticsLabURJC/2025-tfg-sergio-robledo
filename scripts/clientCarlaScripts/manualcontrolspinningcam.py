import carla
import time
import pygame
import numpy as np
import matplotlib.pyplot as plt
from collections import deque




# #..........................................
# Codigo para visualizar el coche y girar la camara alrededor de el
# #..........................................




HOST = '127.0.0.1'
PORT = 2000
VEHICLE_MODEL = 'vehicle.finaldeepracer.aws_deepracer'


pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DeepRacer - RGB y Segmentación Semántica")


client = carla.Client(HOST, PORT)
client.set_timeout(5.0)
world = client.get_world()
client.load_world('Town01') 


weather = carla.WeatherParameters(
    cloudiness=80.0,
    precipitation=0.0,
    sun_altitude_angle=90.0,
    fog_density=0.0,
    wetness=0.0
)
world.set_weather(weather)
print("🌅 Clima establecido en 'Sunset' (Atardecer)")

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find(VEHICLE_MODEL)


spawn_point = carla.Transform(carla.Location(x=2.95, y=-3.7, z=0.6),
                carla.Rotation(pitch=0, yaw=-90, roll=0))

vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    print("Error al spawnear el vehículo")
    exit()
print(f"Vehículo {VEHICLE_MODEL} spawneado en {spawn_point.location}")


camera_rgb_bp = blueprint_library.find('sensor.camera.rgb')
camera_rgb_bp.set_attribute('image_size_x', str(WIDTH))
camera_rgb_bp.set_attribute('image_size_y', str(HEIGHT))
camera_rgb_bp.set_attribute('fov', '90')

# camera_rgb_transform = carla.Transform(carla.Location(y=-0.5, z=0.1),
#                             carla.Rotation(pitch=0, yaw=90, roll=0))
camera_rgb_transform = carla.Transform(carla.Location(x=-1, z=1.3))
camera_rgb = world.spawn_actor(camera_rgb_bp, camera_rgb_transform)


camera_image_rgb = None


def process_rgb(image):
    global camera_image_rgb
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    array = array[:, :, ::-1]
    camera_image_rgb = pygame.surfarray.make_surface(array.swapaxes(0, 1))



camera_rgb.listen(lambda image: process_rgb(image))


control = carla.VehicleControl()
running = True

angle = 0.0
radius = 0.4
angular_speed = 0.2 
delta_t = 1.0 / 30.0 


while running:
    
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        control.throttle = min(control.throttle + 0.01, 0.8)
    else:
        control.throttle = 0.0
    control.brake = min(control.brake + 0.1, 1.0) if keys[pygame.K_s] else 0.0
    if keys[pygame.K_a]:
        control.steer = max(control.steer - 0.05, -1.0)
    elif keys[pygame.K_d]:
        control.steer = min(control.steer + 0.05, 1.0)
    else:
        control.steer = 0.0
    control.hand_brake = keys[pygame.K_SPACE]
    vehicle.apply_control(control)

 
    car_location = vehicle.get_transform().location


    angle += angular_speed * delta_t
    cam_x = car_location.x + radius * np.cos(angle)
    cam_y = car_location.y + radius * np.sin(angle)
    cam_z = 0.3


    dx = car_location.x - cam_x
    dy = car_location.y - cam_y
    yaw = np.degrees(np.arctan2(dy, dx))
    pitch = -8.0


    new_transform = carla.Transform(
        carla.Location(x=cam_x, y=cam_y, z=cam_z),
        carla.Rotation(pitch=pitch, yaw=yaw)
    )
    camera_rgb.set_transform(new_transform)


    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    if camera_image_rgb:
        screen.blit(camera_image_rgb, (0, 0))
    pygame.display.flip()

   
    time.sleep(delta_t)


camera_rgb.destroy()
vehicle.destroy()
pygame.quit()