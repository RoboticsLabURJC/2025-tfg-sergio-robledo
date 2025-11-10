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


# === Dataset paths ===
currtime   = str(int(time.time() * 1000))


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

    # Cámara
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(WIDTH))
    bp.set_attribute("image_size_y", str(HEIGHT))
    bp.set_attribute("fov", "90")
    cam_tf = carla.Transform(carla.Location(x=0.13, z=0.13), carla.Rotation(pitch=-30))
    cam = world.spawn_actor(bp, cam_tf, attach_to=ego)

    
    frame_q = Queue(maxsize=1)   # guardar (rgb, bgr, (w, h))

    def _safe_put(q: Queue, item):
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()   # tira el anterior
            except queue.Empty:
                pass
            q.put_nowait(item)


    def on_image(image: carla.Image):
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        bgr  = bgra[:, :, :3].copy()
        rgb  = bgr[:, :, ::-1]
        _safe_put(frame_q, (rgb, bgr, (image.width, image.height)))


    cam.listen(on_image)

    start_sim = world.get_snapshot().timestamp.elapsed_seconds
    print("start sim: ", start_sim)
    duration  = get_log_duration(client, LOG_FILE)
    end_sim   = start_sim + duration
    print("----------------------------------------")
    try:
        while True:

            clock.tick(FPS)
            world.tick()

            snap = world.get_snapshot()
            sim_time = snap.timestamp.elapsed_seconds

            print("sim_time: ",sim_time)

            try:
                rgb, bgr, (w, h) = frame_q.get_nowait()
            except queue.Empty:
                prev_t = sim_time
            
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        raise KeyboardInterrupt
                continue
    

            # Dibujar si hay frame
        
            surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
    
            # loc = ego.get_location()
            # cur_pos = np.array([loc.x, loc.y, loc.z], dtype=float)

            
            # ctrl = ego.get_control()
            # throttle = float(ctrl.throttle)
            # steer    = max(-1.0, min(1.0, float(ctrl.steer)))
            # brake    = float(ctrl.brake)

  

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