#------------------------------------------------
# Codigo para control manual del Deepracer directamente en Carla, en local.
# Se utiliza simplemente para mover el coche por el mundo y observar sus fisicas y texturas
#-------------------------------------------
#Este codigo permite mover el coche con las teclas del propio teclado en local
#------------------------------------------------

import carla
import time
import pygame
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# #..........................................
# Control manual teleoperado del deepracer con las teclas W A S D 
#     w: Hacia delante
#     A: girar a la izquierda
#     S: frenar
#     D: Girar a la derecha
# #..........................................


# # --- Inicialización de la gráfica ---
# plt.ion()
# fig, ax = plt.subplots()
# history_len = 100

# steer_history = deque([0]*history_len, maxlen=history_len)
# throttle_history = deque([0]*history_len, maxlen=history_len)
# line1, = ax.plot(steer_history, label="Steer", color="blue")
# line2, = ax.plot(throttle_history, label="Throttle", color="green")
# ax.set_ylim(-1.1, 1.1)
# ax.legend()
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
print("Clima establecido en Sunset")

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find(VEHICLE_MODEL)

spawn_point = carla.Transform(carla.Location(x=2.95, y=-3.7, z=0.6),
                carla.Rotation(pitch=0, yaw=-90, roll=0))

vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    print("error al spawnear el vehículo")
    exit()
print(f"Vehículo {VEHICLE_MODEL} spawneado en {spawn_point.location}")

camera_rgb_bp = blueprint_library.find('sensor.camera.rgb')
camera_rgb_bp.set_attribute('image_size_x', str(WIDTH))
camera_rgb_bp.set_attribute('image_size_y', str(HEIGHT))
camera_rgb_bp.set_attribute('fov', '90')

# camera_rgb_transform = carla.Transform(carla.Location(y=-0.5, z=0.1),
#                             carla.Rotation(pitch=0, yaw=90, roll=0))

# camera_rgb_transform = carla.Transform(carla.Location(x= 0.2 ,y=-0.3, z=0.1),
#                             carla.Rotation(pitch=10, yaw=120, roll=0))

camera_rgb_transform = carla.Transform(carla.Location(x=-1, z=0.5))
camera_rgb = world.spawn_actor(camera_rgb_bp, camera_rgb_transform, attach_to=vehicle)


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

    #print(f"steer={control.steer:.2f} throttle={control.throttle:.2f}")
    v = vehicle.get_velocity()
    speed = (v.x**2 + v.y**2 + v.z**2) ** 0.5
    print(f"Velocidad: {speed:.2f} m/s")

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    
    if camera_image_rgb:
        screen.blit(camera_image_rgb, (0, 0))

    pygame.display.flip()


    # steer_history.append(control.steer)
    # throttle_history.append(control.throttle)
    # line1.set_ydata(steer_history)
    # line2.set_ydata(throttle_history)
    # line1.set_xdata(range(len(steer_history)))
    # line2.set_xdata(range(len(throttle_history)))
    # ax.relim()
    # ax.autoscale_view()
    # plt.draw()
    # plt.pause(0.001)
    

camera_rgb.destroy()
vehicle.destroy()
pygame.quit()
