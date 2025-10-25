#!/usr/bin/env python3
import os, re, csv, time
import carla, pygame
import numpy as np
import cv2
from datetime import datetime
import queue
from queue import Queue


LOG_FILE = "/tmp/TrackTestSPEED.log"
WIDTH, HEIGHT = 800, 600
FPS = 30.0
FIXED_DT = 1.0 / FPS
prev_timeglobal_var = 0.0

prev_pos = None
prev_t   = None
ema_v    = None 
ALPHA  = 0.2

currtime   = str(int(time.time() * 1000))
CSV_PATH = "/home/sergior/Downloads/carla_recorder_replay/mispruebas/telemetryfromreplay.csv"

with open(CSV_PATH, "w", newline="") as f:
    csv.writer(f).writerow(["timestamp","speed"])

def get_log_duration(client, path: str) -> float:
    info = client.show_recorder_file_info(path, False)
    m = re.search(r"Duration:\s*([0-9.]+)", info)
    if not m:
        raise RuntimeError("No se pudo leer la duración del log")
    return float(m.group(1))

def wait_for_vehicle(world, filt="vehicle.finaldeepracer.aws_deepracer", timeout_s=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        world.tick()
        actors = world.get_actors().filter(filt)
        if actors:
            return actors[0]

        actors = world.get_actors().filter("vehicle.*deepracer*")
        if actors:
            return actors[0]
    return None


def guardar_dato(timestamp,speed):
    
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([timestamp,speed])

def main():
    global prev_timeglobal_var, prev_pos, ema_v, prev_t
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CARLA Replay")
    clock = pygame.time.Clock()

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)

    # Ver info del log
    print(client.show_recorder_file_info(LOG_FILE, False))

    # Lanzar replay y refrescar world
    client.replay_file(LOG_FILE, 0.0, 0.0, 0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    world.apply_settings(settings)

    weather = carla.WeatherParameters(
        cloudiness=80.0, precipitation=0.0, sun_altitude_angle=90.0,
        fog_density=0.0, wetness=0.0
    )
    world.set_weather(weather)

    ego = wait_for_vehicle(world)
    if ego is None:
        print("No se encontró vehículo en el replay.")
        return
    print(f"Ego id={ego.id}  type={ego.type_id}")


    start_sim = world.get_snapshot().timestamp.elapsed_seconds
    duration  = get_log_duration(client, LOG_FILE)
    end_sim   = start_sim + duration

    try:
        while True:

            world.tick()

            snap = world.get_snapshot()
            sim_time = snap.timestamp.elapsed_seconds


            # Telemetría del ego
            t  = sim_time
            dt = snap.timestamp.delta_seconds or (t - prev_t if prev_t is not None else None)

            loc = ego.get_location()
            cur_pos = np.array([loc.x, loc.y, loc.z], dtype=float)

            if prev_pos is not None and dt and dt > 0:
                # distancia (m) / tiempo (s), velocidad en m/s
                speed = float(np.linalg.norm(cur_pos - prev_pos) / dt)

                # suavizado EMA
                ema_v = speed if ema_v is None else (1-ALPHA)*ema_v + ALPHA*speed
                speed = ema_v

            else:
                speed = 0.0
            
            prev_pos,prev_t = cur_pos, t
        
    
            print("Guardando...")
            guardar_dato(sim_time,speed)
        
            # salir al final del log
            if sim_time >= end_sim:
                print("Replay finalizado.")
                break

            # eventos ventana
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    raise KeyboardInterrupt


    except KeyboardInterrupt:
        print("Interrumpido por el usuario")
    finally:
        try:
            cam.stop(); cam.destroy()
        except:
            pass
        pygame.quit()

if __name__ == "__main__":
    main()
