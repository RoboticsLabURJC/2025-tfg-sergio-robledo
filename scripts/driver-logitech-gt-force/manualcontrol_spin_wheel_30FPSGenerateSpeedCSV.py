#!/usr/bin/env python3
import argparse
import carla
import time
import pygame
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import csv
from ControllerProxy import ControllerReceiver


log_filename = "/home/sergior/Downloads/carla_recorder_replay/mispruebas/Track011CC.log"


# ===================== ARGUMENTOS =====================
def parse_args():
    p = argparse.ArgumentParser("Control manual con volante (ControllerReceiver)")
    p.add_argument("--carla-host", default="127.0.0.1", type=str)
    p.add_argument("--carla-port", default=3333, type=int, help="Puerto RPC de CARLA (p.ej., 3010)")
    p.add_argument("--controller-port", default=1977, type=int, help="Puerto donde escucha el receptor del volante")
    p.add_argument("--width", type=int, default=1500)
    p.add_argument("--height", type=int, default=1200)
    p.add_argument("--map", type=str, default=None, help="(Opcional) cargar mapa, p.ej. Town04")
    return p.parse_args()

# ===================== MAIN =====================
def main():
    args = parse_args()

    # ======== Matplotlib para monitorizar steer/throttle ========
    # plt.ion()
    # fig, ax = plt.subplots()
    # history_len = 100
    # steer_history = deque([0.0]*history_len, maxlen=history_len)
    # throttle_history = deque([0.0]*history_len, maxlen=history_len)
    # (line1,) = ax.plot(steer_history, label="Steer", color="blue")
    # (line2,) = ax.plot(throttle_history, label="Throttle", color="green")
    # ax.set_ylim(-1.1, 1.1)
    # ax.legend()

    # ======== Pygame (ventana de cámara) ========
    WIDTH, HEIGHT = args.width, args.height
    FPS = 30
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DeepRacer - Control manual (volante)")
    clock = pygame.time.Clock()

    # ======== Conexión a CARLA ========
    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(10.0)

    if args.map:
        try:
            client.load_world(args.map)
        except Exception as e:
            print(f"[WARN] No se pudo cargar el mapa {args.map}: {e}")

    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    settings.substepping = False
    world.apply_settings(settings)

    # Tiempo de simulación (CSV)
    csv_path = "telemetry.csv"
    csv_fh = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(["sim_time", "speed_m_s"])

    weather = carla.WeatherParameters(
        cloudiness=80.0, precipitation=0.0, sun_altitude_angle=90.0,
        fog_density=0.0, wetness=0.0
    )
    world.set_weather(weather)

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.finaldeepracer.aws_deepracer')

    # ELECCION DE PISTA

    #spawn_point = carla.Transform(carla.Location(x=3, y=-1, z=0.5), carla.Rotation(yaw=-90))

    #spawn_point = carla.Transform(carla.Location(x=-8.5, y=-14.7, z=0.5), carla.Rotation(yaw=-15))

    #spawn_point = carla.Transform(carla.Location(x=17, y=-4.8, z=0.5), carla.Rotation(yaw=-10))

    #spawn_point = carla.Transform(carla.Location(x=-10, y=21.2, z=1), carla.Rotation(yaw=-15))
    
    #spawn_point = carla.Transform(carla.Location(x=-3.7, y=-4, z=0.5), carla.Rotation(yaw=-120))
    
    #gillesvilleneuve
    #spawn_point = carla.Transform(carla.Location(x=-1.5, y=33, z=0.5), carla.Rotation(yaw=0))    

    #interlagosautodromojosecarlospace
    #spawn_point = carla.Transform(carla.Location(x=-1.5, y=71.5, z=0.5), carla.Rotation(yaw=180))
    
    #nurburgring
    #spawn_point = carla.Transform(carla.Location(x=-65, y=17.5, z=0.5), carla.Rotation(yaw=150))
    
    #spafrancorchamps
    #spawn_point = carla.Transform(carla.Location(x=-65, y=94.5, z=0.5), carla.Rotation(yaw=-90))
    
    #silverstone
    #spawn_point = carla.Transform(carla.Location(x=-67, y=228, z=0.5), carla.Rotation(yaw=180-25))
    
    #lagoseco
    spawn_point = carla.Transform(carla.Location(x=-67, y=318, z=0.5), carla.Rotation(yaw=180-25))


    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if not vehicle:
        print("Unable to spawn vehicle")
        return
    print("Vehicle spawned correctly")

    client.start_recorder(log_filename, True)

    # ======== Cámara ========
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(WIDTH))
    camera_bp.set_attribute('image_size_y', str(HEIGHT))
    camera_bp.set_attribute('fov', '100')
    camera_transform = carla.Transform(carla.Location(x=-1, z=0.5))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    camera_surface = None
    def camera_callback(image):
        nonlocal camera_surface

        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = np.reshape(arr, (image.height, image.width, 4))[:, :, :3]
        arr = arr[:, :, ::-1]  # BGR -> RGB
        camera_surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))

    camera.listen(camera_callback)

    # ======== Control vía ControllerReceiver ========
    control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)

    def my_callback(tv_sec, tv_usec, code, value):
        # 0=steer (-1..1), 1=throttle (0..1), 2=brake (0..1)
        if code == 0:
            control.steer = -float(value)
        elif code == 1:
            control.throttle = float(value)
        elif code == 2:
            control.brake = float(value)

    server = ControllerReceiver(args.controller_port, my_callback)
    server.start()
    print(f"ControllerReceiver escuchando en puerto {args.controller_port}")


    DURACION_S = 140
    t0 = time.time() 
    running = True


    while running:

        clock.tick(FPS)


        # Timeout
        if time.time() - t0 >= DURACION_S:
            client.stop_recorder()
            print("Grabacion guardada!")
            camera.stop()
            camera.destroy()
            vehicle.destroy()
            pygame.quit()

            print("Session ended.")

        # Eventos Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        # Aplicar control
        vehicle.apply_control(control)

        snapshot = world.tick()
        sim_time = snapshot.timestamp.elapsed_seconds

        # Mostrar cámara
        if camera_surface:
            screen.blit(camera_surface, (0, 0))
        pygame.display.flip()

        # Velocidad (m/s)
        v = vehicle.get_velocity()
        speed = float(np.linalg.norm([vel.x, vel.y, vel.z]))
        csv_writer.writerow([f"{sim_time:.6f}", f"{speed:.6f}"])

        #print(f"Speed: {speed:.2f} m/s | Steer: {control.steer:+.3f} | Throttle: {control.throttle:.3f} | Brake: {control.brake:.3f}")

        # Actualizar gráficas
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

        world.tick()


    try: client.stop_recorder()
        except: pass
    try:
        camera.stop()
    except Exception:
        pass
    try:
        camera.destroy()
    except Exception:
        pass
    try:
        vehicle.destroy()
    except Exception:
        pass
    try:
        if hasattr(server, "stop"):
            server.stop()
    except Exception:
        pass
    try: csv_fh.flush(); csv_fh.close()
        except: pass
    pygame.quit()
    print("Session ended.")

if __name__ == "__main__":
    main()

