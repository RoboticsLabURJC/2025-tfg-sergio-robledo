#!/usr/bin/env python3
import os, re, csv, time
import carla, pygame
import numpy as np
import cv2
from datetime import datetime

LOG_FILE = "/tmp/Track04C.log"
WIDTH, HEIGHT = 800, 600
FPS = 30.0
FIXED_DT = 1.0 / FPS

# === Dataset paths ===
currtime   = str(int(time.time() * 1000))
DATASET_ID = "Deepracer_BaseMap_" + currtime
SAVE_DIR   = DATASET_ID
RGB_DIR    = os.path.join(SAVE_DIR, "rgb")
MASK_DIR   = os.path.join(SAVE_DIR, "masks")
CSV_PATH   = os.path.join(SAVE_DIR, "dataset.csv")

os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
if not os.path.exists(CSV_PATH):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["rgb_path","mask_path","timestamp","throttle","steer","brake","speed","heading","estado"])

def get_log_duration(client, path: str) -> float:
    info = client.show_recorder_file_info(path, False)
    m = re.search(r"Duration:\s*([0-9.]+)", info)
    if not m:
        raise RuntimeError("No se pudo leer la duración del log")
    return float(m.group(1))

def wait_for_vehicle(world, filt="vehicle.finaldeepracer.aws_deepracer", timeout_s=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        snap = world.wait_for_tick(2.0)
        if snap is None:
            continue
        actors = world.get_actors().filter(filt)
        if actors:
            return actors[0]

        actors = world.get_actors().filter("vehicle.*deepracer*")
        if actors:
            return actors[0]
    return None

def _estado_from_steer(steer: float) -> int:
    return 1 if steer < -0.25 else (3 if steer > 0.25 else 2)

def guardar_dato(timestamp, bgr, mask_rgb, throttle, steer, brake, speed, heading):
    rgb_name  = f"{timestamp}_rgb_{DATASET_ID}.png"
    mask_name = f"{timestamp}_mask_{DATASET_ID}.png"
    cv2.imwrite(os.path.join(RGB_DIR,  rgb_name),  bgr)
    cv2.imwrite(os.path.join(MASK_DIR, mask_name), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([f"/rgb/{rgb_name}", f"/masks/{mask_name}", timestamp,
                                throttle, steer, brake, speed, heading, _estado_from_steer(steer)])

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CARLA Replay")
    #clock = pygame.time.Clock()

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

    # buffers compartidos
    last_rgb = [None]
    last_bgr = [None]
    last_wh  = [None]

    def on_image(image: carla.Image):
        # BGRA -> BGR y RGB
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        bgr  = bgra[:, :, :3].copy()
        rgb  = bgr[:, :, ::-1]
        last_bgr[0] = bgr
        last_rgb[0] = rgb
        last_wh[0]  = (image.width, image.height)

    cam.listen(on_image)

    start_sim = world.get_snapshot().timestamp.elapsed_seconds
    duration  = get_log_duration(client, LOG_FILE)
    end_sim   = start_sim + duration
    start_save_sim_t = start_sim + 7.0   # guardar a partir de +5 s

    try:
        while True:
            world.tick()
            snap = world.get_snapshot()
            sim_time = snap.timestamp.elapsed_seconds

            # Dibujar si hay frame
            if last_rgb[0] is not None:
                surface = pygame.surfarray.make_surface(last_rgb[0].swapaxes(0, 1))
                screen.blit(surface, (0, 0))
                pygame.display.flip()

                # === Segmentación muy simple (opcional) ===
                rgb = last_rgb[0]
                bgr = last_bgr[0]
                w, h = last_wh[0]

                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                mask_y = cv2.inRange(hsv, np.array([18, 50, 150]), np.array([40, 255, 255]))
                mask_w = cv2.inRange(hsv, np.array([0, 0, 200]),  np.array([180, 30, 255]))
                mask_c = np.zeros(mask_w.shape, np.uint8); mask_c[mask_w>0]=1; mask_c[mask_y>0]=2
                mask_rgb = np.zeros_like(rgb); mask_rgb[mask_c==1]=[255,255,255]; mask_rgb[mask_c==2]=[255,255,0]

                # Heading
                y = int(0.53 * h)
                row = mask_c[y]; white_idx = np.where(row==1)[0]
                cx = None
                if len(white_idx) > 10:
                    cx = (white_idx[0] + white_idx[-1]) // 2
                img_cx  = w//2
                err_px  = (img_cx - cx) if cx is not None else 0
                dy_px   = h - y
                heading = float(np.degrees(np.arctan2(err_px, dy_px)))

                # Telemetría del ego
                vel = ego.get_velocity()
                speed = float(np.linalg.norm([vel.x, vel.y, vel.z]))
                ctrl = ego.get_control()
                throttle = float(ctrl.throttle)
                steer    = max(-1.0, min(1.0, float(ctrl.steer)))  # clamp por seguridad
                brake    = float(ctrl.brake)

                # Guardar desde t >= 10 s
                if sim_time >= start_save_sim_t:
                    print("Guardando...")
                    ts = int(datetime.utcnow().timestamp()*1000)
                    guardar_dato(ts, bgr, mask_rgb, throttle, steer, brake, speed, heading)
                else:
                    print("Aun no guarda")

            # salir al final del log
            if sim_time >= end_sim:
                print("Replay finalizado.")
                break

            # eventos ventana
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    raise KeyboardInterrupt

            #clock.tick(FPS)

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
