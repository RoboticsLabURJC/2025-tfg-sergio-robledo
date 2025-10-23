#!/usr/bin/env python3
import carla
import time
import pygame
import numpy as np
import cv2
from collections import deque
from threading import Lock
import os
import csv
from datetime import datetime
import argparse
import random
from ControllerProxy import ControllerReceiver

# ===================== Argumentos =====================
def parse_args():
    p = argparse.ArgumentParser("Data recorder (teleop) con máscara por color")
    p.add_argument("--carla-host", default="127.0.0.1", type=str)
    p.add_argument("--carla-port", default=3010, type=int, help="Puerto RPC de CARLA")
    p.add_argument("--controller-port", default=1977, type=int, help="Puerto donde escucha el controlador")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int, default=800)#1250
    p.add_argument("--height", type=int, default=600)#1500
    p.add_argument("--start-delay", type=float, default=15.0, help="Segundos antes de empezar a guardar")
    p.add_argument("--max-duration", type=float, default=120.0, help="Segundos totales de ejecución (auto-stop)")
    p.add_argument("--save-root", type=str, default="dataset", help="Carpeta raíz donde crear el dataset")
    p.add_argument("--map", type=str, default=None, help="cargar mapa, p.ej. Town04")
    return p.parse_args()

# ===================== Estado global control =====================
current_steer = 0.0
current_throttle = 0.0
current_brake = 0.0

# Colas y locks
image_queue = deque(maxlen=1)
queue_lock = Lock()
MAX_IMAGE_AGE_MS = 150

def _estado_from_steer(steer: float) -> int:
    """1 si steer < -0.2, 2 si -0.2 <= steer <= 0.2, 3 si steer > 0.2"""
    if steer < -0.2: return 1
    if steer > 0.2:  return 3
    return 2

def main():
    args = parse_args()
    WIDTH, HEIGHT = args.width, args.height

    # ===================== Setup dataset =====================
    currtime = str(int(time.time() * 1000))
    DATASET_ID = f"Deepracer_BaseMap_{currtime}"
    BASE_DIR = args.save_root
    SAVE_DIR = os.path.join(BASE_DIR, DATASET_ID)
    RGB_DIR = os.path.join(SAVE_DIR, "rgb")
    MASK_DIR = os.path.join(SAVE_DIR, "masks")
    CSV_PATH = os.path.join(SAVE_DIR, "dataset.csv")

    os.makedirs(RGB_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["rgb_path", "mask_path", "timestamp",
                             "throttle", "steer", "brake", "speed", "heading", "estado"])

    # ===================== Pygame (visor) =====================
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DeepRacer - Teleop dataset recorder")

    # ===================== Cliente CARLA =====================
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
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    # Tiempo
    weather = carla.WeatherParameters(
        cloudiness=80.0, precipitation=0.0, sun_altitude_angle=90.0,
        fog_density=0.0, wetness=0.0
    )
    world.set_weather(weather)

    bp_lib = world.get_blueprint_library()

    # Vehículo
    vehicle_bp = bp_lib.find('vehicle.finaldeepracer.aws_deepracer')
    
    
    # Pistas
    # -------------------------TRACK01-----------------------------
    # spawn_point = carla.Transform(
    #     carla.Location(x=3, y=-1, z=0.5),
    #     carla.Rotation(yaw=-90)
    # )


    #-------------------------TRACK03---------------------------------
    # spawn_point = carla.Transform(
    #     carla.Location(x=-8, y=-15, z=0.5),
    #     carla.Rotation(yaw=-15)
    # )

    #-------------------------TRACK02---------------------------------
    # spawn_point = carla.Transform(
    #    carla.Location(x=17, y=-4.8, z=0.5),
    #    carla.Rotation(yaw=-10)
    # )

    #-------------------------TRACK04---------------------------------
    # spawn_point = carla.Transform(
    #     carla.Location(x=-10, y=21.2, z=1),
    #     carla.Rotation(yaw=-15)
    # )

    #-------------------------TRACK05---------------------------------
    # spawn_point = carla.Transform(
    #    carla.Location(x=-3.7, y=-4, z=0.5),
    #    carla.Rotation(yaw=-120)
    # )

    #-------------------TRACK06-gillesvilleneuve----------------------
    spawn_point = carla.Transform(
    carla.Location(carla.Location(x=-1.5, y=33.3, z=0.5)),
    carla.Rotation(yaw=180)
    )

    #-------------TRACK07-interlagosautodromojosecarlospace-----------
    # spawn_point = carla.Transform(
    #    carla.Location(x=-1.5, y=71.5, z=0.5),
    #    carla.Rotation(yaw=0)
    # )


    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if not vehicle:
        print("Error al spawnear el vehículo")
        return
    print(f"Vehículo spawneado en {spawn_point.location}")

    # Cámara RGB frontal
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(WIDTH))
    cam_bp.set_attribute('image_size_y', str(HEIGHT))
    cam_bp.set_attribute('fov', '140')
    cam_tf = carla.Transform(carla.Location(x=0.13, z=0.13), carla.Rotation(pitch=-30))
    camera_front = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

    # ===================== Callback cámara =====================
    def camera_callback(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = np.reshape(arr, (image.height, image.width, 4))[:, :, :3]  # BGR
        with queue_lock:
            image_queue.clear()
            image_queue.append((int(time.time() * 1000), arr))

    camera_front.listen(camera_callback)

    # ===================== Receptor del controlador =====================
    control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)

    def my_callback(tv_sec, tv_usec, code, value):
        # 0=steer, 1=throttle, 2=brake
        nonlocal control
        if code == 0:
            control.steer = - float(value)
        elif code == 1:
            control.throttle = float(value)
        elif code == 2:
            control.brake = float(value)

    server = ControllerReceiver(args.controller_port, my_callback)
    server.start()
    print(f"ControllerReceiver escuchando en puerto {args.controller_port}")
 # ===================== Helpers =====================
    def guardar_dato(timestamp, bgr_img, mask_rgb_img, accel, steer, brake, speed, heading):
        rgb_name = f"{timestamp}_rgb_{DATASET_ID}.png"
        mask_name = f"{timestamp}_mask_{DATASET_ID}.png"

        rgb_path_rel = os.path.join("/rgb", rgb_name)
        mask_path_rel = os.path.join("/masks", mask_name)

        # Guarda BGR
        cv2.imwrite(os.path.join(RGB_DIR, rgb_name), bgr_img)
        # Máscara
        cv2.imwrite(os.path.join(MASK_DIR, mask_name), cv2.cvtColor(mask_rgb_img, cv2.COLOR_RGB2BGR))

        estado = _estado_from_steer(float(steer))

        with open(CSV_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([rgb_path_rel, mask_path_rel, timestamp,
                             accel, steer, brake, speed, heading, estado])

    # ======== Control de tiempos ========
    run_start = time.time()
    start_recording_time = run_start + args.start_delay  
    end_time = run_start + args.max_duration             
    started_msg = False

    # Pre-roll y arranque
    vehicle.apply_control(control)
    time.sleep(1.0)

    running = True
    clock = pygame.time.Clock()
    print(f"Se empezará a guardar tras {args.start_delay:.1f}s y se parará a los {args.max_duration:.1f}s.")

    try:
        while running:
            now = time.time()
            if now >= end_time:
                print("Tiempo máximo alcanzado. Parando.")
                break

            if (not started_msg) and now >= start_recording_time:
                started_msg = True
                print("Comienza la grabación.")

            world.tick()
            vehicle.apply_control(control)

            # lee última imagen reciente
            with queue_lock:
                if image_queue:
                    img_ts, bgr = image_queue[0]
                    age = int(time.time() * 1000) - img_ts
                    if age <= MAX_IMAGE_AGE_MS:
                        bgr = image_queue.popleft()[1]
                    else:
                        bgr = None
                else:
                    bgr = None

            if bgr is not None:
                # === Construcción de RGB y máscara===
                rgb = bgr[:, :, ::-1].copy()

                # Mostrar en pygame
                surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                screen.blit(surf, (0, 0))
                pygame.display.flip()

                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

                lower_yellow = np.array([18, 50, 150])
                upper_yellow = np.array([40, 255, 255])
                mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

                lower_white = np.array([0, 0, 200])
                upper_white = np.array([180, 30, 255])
                mask_white = cv2.inRange(hsv, lower_white, upper_white)

                mask_class = np.zeros_like(mask_white, dtype=np.uint8)
                mask_class[mask_white > 0] = 1
                mask_class[mask_yellow > 0] = 2

                mask_rgb = np.zeros_like(rgb)
                mask_rgb[mask_class == 1] = [255, 255, 255]  # blanco
                mask_rgb[mask_class == 2] = [255, 255, 0]    # amarillo

                # === Heading ===
                y = int(0.53 * HEIGHT)
                row = mask_class[y]
                white_indices = np.where(row == 1)[0]
                center_x = None
                if len(white_indices) > 10:
                    left = white_indices[0]
                    right = white_indices[-1]
                    center_x = (left + right) // 2

                image_center_x = WIDTH // 2
                error_px = (image_center_x - center_x) if center_x is not None else 0
                dy_px = int(HEIGHT - 0.53 * HEIGHT)
                heading_rad = np.arctan2(error_px, dy_px)
                heading = float(np.degrees(heading_rad))

                # === Velocidad ===
                vel = vehicle.get_velocity()
                speed = float(np.linalg.norm([vel.x, vel.y, vel.z]))

                # === Guardado tras delay ===
                if now >= start_recording_time:
                    ts = int(datetime.utcnow().timestamp() * 1000)
                    guardar_dato(
                        timestamp=ts,
                        bgr_img=bgr,
                        mask_rgb_img=mask_rgb,
                        accel=float(control.throttle),
                        steer=float(control.steer),
                        brake=float(control.brake),
                        speed=speed,
                        heading=heading
                    )

            # eventos pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            clock.tick(args.fps)

    finally:
        try:
            camera_front.stop()
        except Exception:
            pass
        try:
            camera_front.destroy()
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

        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        pygame.quit()
        print("Session finished")

if __name__ == "__main__":
    main()