import carla
import time
import pygame
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import deque


# #..........................................
# Codigo para seguir una recta y evaluar caracteristicas como
# suspension, velocidad, friccion y otras cosas. Codigo de prueba
# #..........................................


# --- Inicialización de la gráfica ---
plt.ion()
fig, ax = plt.subplots()
history_len = 100  
x_history = deque([0]*history_len, maxlen=history_len)
y_history = deque([0]*history_len, maxlen=history_len)
line1, = ax.plot(x_history, label="X", color="blue")
line2, = ax.plot(y_history, label="Y", color="green")
ax.set_ylim(-5, 5)
ax.legend()
#..........................................


HOST = '127.0.0.1'
PORT = 2000
VEHICLE_MODEL = 'vehicle.finaldeepracer.aws_deepracer'


pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen = pygame.display.set_mode((WIDTH * 2, HEIGHT)) 
pygame.display.set_caption("DeepRacer - RGB y Segmentación Semántica")


client = carla.Client(HOST, PORT)
client.set_timeout(5.0)
world = client.get_world()
#client.load_world('Town01') 


weather = carla.WeatherParameters(
    cloudiness=80.0,
    precipitation=0.0,
    sun_altitude_angle=90.0,
    fog_density=0.0,
    wetness=0.0
)
world.set_weather(weather)
print("Clima establecido en 'Sunset' (Atardecer)")


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

transform_top = carla.Transform(
    carla.Location(x=-1, y=0, z=0.5),
    carla.Rotation(pitch=-10)       
)

# transform_top = carla.Transform(
#     carla.Location(x=0, y=-1, z=0.05),    
#     carla.Rotation(yaw = 90)
# )


camera_top = world.spawn_actor(camera_rgb_bp, transform_top, attach_to=vehicle)
time.sleep(1)


camera_image_rgb = None
camera_image_top = None

def process_top(image):
    global camera_image_top
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    array = array[:, :, ::-1]
    camera_image_top = pygame.surfarray.make_surface(array.swapaxes(0, 1))


def process_rgb(image):
    global camera_image_rgb
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    array = array[:, :, ::-1]
    camera_image_rgb = pygame.surfarray.make_surface(array.swapaxes(0, 1))

def on_collision(event):
    actor_we_hit = event.other_actor
    impulse = event.normal_impulse
    location = event.transform.location

    print(f"[COLLISION] with: {actor_we_hit.type_id}")
    print(f"Impulse: {impulse}")
    print(f"Location: x={location.x:.2f}, y={location.y:.2f}, z={location.z:.2f}")


camera_top.listen(process_top)
camera_front.listen(process_rgb) 



# collision_bp = blueprint_library.find('sensor.other.collision')
# collision_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
# collision_sensor = world.spawn_actor(collision_bp, collision_transform, attach_to=vehicle)
#collision_sensor.listen(on_collision)

control = carla.VehicleControl()
running = True

control = carla.VehicleControl(throttle=1, steer=0)
vehicle.apply_control(control)

while running:

    v = vehicle.get_velocity()
    speed = (v.x**2 + v.y**2 + v.z**2) ** 0.5
    print(f"Velocidad: {speed:.2f} m/s")

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

 
    if camera_image_rgb:
        screen.blit(camera_image_rgb, (0, 0))  
    if camera_image_top:
            screen.blit(camera_image_top, (WIDTH, 0)) 

    pygame.display.flip()
    location = vehicle.get_transform().location
    x_history.append(location.x)
    y_history.append(location.y)
    
    line1.set_ydata(x_history)
    line2.set_ydata(y_history)
    line1.set_xdata(range(len(x_history)))
    line2.set_xdata(range(len(y_history)))
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.001)
    


camera_rgb.destroy()
vehicle.destroy()
camera_top.destroy()
pygame.quit()
