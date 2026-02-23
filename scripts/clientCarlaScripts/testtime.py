#------------------------------------------------
#Codigo para la comprobacion de ls diferencia entre “tiempo simulado” y “tiempo de reloj real”
# #..........................................

import carla
import time
import pygame
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import deque

#Always run the simulator at fixed time-step when using the synchronous mode. 
# Otherwise the physics engine will try to recompute at once all the time spent
# waiting for the client, this usually results in inconsistent or not very 
# realistic physics.
# Fixed time-step The simulation runs as fast as possible, 
# simulating the same time increment on each step. To enable this 
# mode set a fixed delta seconds in the world settings

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

#FPS
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1.0 / 30.0
world.apply_settings(settings)

weather = carla.WeatherParameters(
    cloudiness=80.0,
    precipitation=0.0,
    sun_altitude_angle=90.0,
    fog_density=0.0,
    wetness=0.0
)
world.set_weather(weather)
print("Clima establecido en 'Sunset'")


blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find(VEHICLE_MODEL)


spawn_point = carla.Transform(
    carla.Location(x=3, y=-1, z=0.5),
    carla.Rotation(yaw=-90)
)
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    print("Error al spawnear el vehículo")
    exit()
print(f"Vehículo {VEHICLE_MODEL} spawneado en {spawn_point.location}")


camera_rgb_bp = blueprint_library.find('sensor.camera.rgb')
camera_rgb_bp.set_attribute('image_size_x', str(WIDTH))
camera_rgb_bp.set_attribute('image_size_y', str(HEIGHT))
camera_rgb_bp.set_attribute('fov', '120')


transform_front = carla.Transform(carla.Location(x=0.13, z=0.13), carla.Rotation(pitch=-30))
transform_thirdpers = carla.Transform(carla.Location(x=-1, z=0.75))
camera_front = world.spawn_actor(camera_rgb_bp, transform_front, attach_to=vehicle)


def camera_callback(image):
    print(f"[Frame {image.frame}] timestamp: {image.timestamp:.5f}")



camera_front.listen(camera_callback)


control = carla.VehicleControl()
running = True

while running:

    # Move on to the next iteration with world.tick()
    t1 = time.time()
    world.tick()
    t2 = time.time()
    print(f"real time gap: {t2-t1}")


camera_rgb.destroy()
vehicle.destroy()
pygame.quit()
