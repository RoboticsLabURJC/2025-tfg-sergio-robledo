import carla
import time
import pygame
import numpy as np
import socket

# Socket config

HOST = 'localhost'
PORT = 1977

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
print(f"Waiting for connection {HOST}:{PORT}...")
conn, addr = s.accept()
print(f"Connected from {addr}")

# Carla config
WIDTH, HEIGHT = 800, 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DeepRacer - Remote control")

client = carla.Client('127.0.0.1', 2000)
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

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.finaldeepracer.aws_deepracer')

spawn_point = carla.Transform(carla.Location(x=2.95, y=-3.7, z=0.6), carla.Rotation(yaw=-90))
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    print("Unable to spawn vehicle")
    exit()
print("Vehicle spawned correctly")

camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(WIDTH))
camera_bp.set_attribute('image_size_y', str(HEIGHT))
camera_bp.set_attribute('fov', '90')
camera_transform = carla.Transform(carla.Location(x=-1, z=0.5))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

camera_image = None
def process_image(image):
    global camera_image
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    array = array[:, :, ::-1]
    camera_image = pygame.surfarray.make_surface(array.swapaxes(0, 1))
camera.listen(process_image)


# Scale from joystick values [-32768, 32767] to [-1.0, 1.0]
def scale_axis(value, axis_type):

    if axis_type in ('ABS_X', 'ABS_RX'):
        return max(-1.0, min(1.0, value / 32767.0))
    elif axis_type in ('ABS_Y', 'ABS_RY'):
        return max(-1.0, min(1.0, value / 32767.0))
    return 0.0

control = carla.VehicleControl()
current_steer = 0.0
current_throttle = 0.0

running = True
while running:
    try:
        buffer = conn.recv(1024).decode(errors='ignore').strip()
        if not buffer:
            continue

        lines = buffer.splitlines()
        for line in lines:
            if "[AXIS]" not in line:
                continue
            if "ABS_X" in line:
                try:
                    val = int(line.split("ABS_X")[1].strip())
                    current_steer = scale_axis(val, 'ABS_X')
                except:
                    continue
            elif "ABS_Y" in line:
                try:
                    val = int(line.split("ABS_Y")[1].strip())
                    scaled = scale_axis(val, 'ABS_Y')
                    current_throttle = 0.6 * (1.0 - scaled) if scaled < 0 else 0.6 * (1.0 - abs(scaled))
                except:
                    continue

        # Apply control
        control.steer = current_steer
        control.throttle = max(0.0, current_throttle)
        control.brake = 0.0
        vehicle.apply_control(control)

        # Show camera
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if camera_image:
            screen.blit(camera_image, (0, 0))
        pygame.display.flip()

        # Speed
        velocity = vehicle.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        print(f"Speed: {speed:.2f} m/s  | Steer: {control.steer:.2f} | Throttle: {control.throttle:.2f}")

    except KeyboardInterrupt:
        print("Interrupted")
        running = False
        break

camera.destroy()
vehicle.destroy()
pygame.quit()
conn.close()
s.close()
