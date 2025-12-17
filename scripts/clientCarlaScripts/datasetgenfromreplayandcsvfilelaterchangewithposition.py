#!/usr/bin/env python3
import os, re, csv, time
import carla, pygame
import numpy as np
import cv2
from queue import Queue
import pandas as pd

SPEED_CSV = "/home/sergior/Downloads/carla_recorder_replay/mispruebas/fourthrecord/speedtrack09C.csv"
LOG_FILE  = "/tmp/Track09CSpeed.log"
WIDTH, HEIGHT = 800, 600
FPS = 30.0

# === Dataset paths ===
currtime   = str(int(time.time() * 1000))
DATASET_ID = "Deepracer_BaseMap_" + currtime
SAVE_DIR   = DATASET_ID
RGB_DIR    = os.path.join(SAVE_DIR, "rgb")
MASK_DIR   = os.path.join(SAVE_DIR, "masks")
CSV_PATH   = os.path.join(SAVE_DIR, "dataset.csv")

os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

# CSV header (añadimos x,y,z)
if not os.path.exists(CSV_PATH):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "rgb_path","mask_path","timestamp",
            "throttle","steer","brake",
            "speed","heading","estado",
            "x","y","z"
        ])

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

def _estado_from_steer(steer: float) -> int:
    return 1 if steer < -0.20 else (3 if steer > 0.20 else 2)

def guardar_dato(timestamp, bgr, mask_rgb, throttle, steer, brake, speed, heading, x, y, z):
    rgb_name  = f"{timestamp:.6f}_rgb_{DATASET_ID}.png"
    mask_name = f"{timestamp:.6f}_mask_{DATASET_ID}.png"

    cv2.imwrite(os.path.join(RGB_DIR,  rgb_name),  bgr)
    cv2.imwrite(os.path.join(MASK_DIR, mask_name), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))

    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            f"/rgb/{rgb_name}",
            f"/masks/{mask_name}",
            float(timestamp),
            float(throttle),
            float(steer),
            float(brake),
            float(speed),
            float(heading),
            _estado_from_steer(float(steer)),
            float(x), float(y), float(z)
        ])

# ========= SPEED CSV: nearest neighbor en tiempo real =========
def load_speed_lookup(speed_csv: str, time_col="sim_time", speed_col="speed_m_s"):
    if not os.path.isfile(speed_csv):
        raise FileNotFoundError(f"No existe SPEED_CSV: {speed_csv}")

    df = pd.read_csv(speed_csv)
    if time_col not in df.columns or speed_col not in df.columns:
        raise ValueError(f"SPEED_CSV debe tener columnas '{time_col}' y '{speed_col}'. Tiene: {list(df.columns)}")

    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=np.float64)
    v = pd.to_numeric(df[speed_col], errors="coerce").to_numpy(dtype=np.float64)

    m = np.isfinite(t) & np.isfinite(v)
    t = t[m]; v = v[m]
    if len(t) < 2:
        raise RuntimeError("SPEED_CSV no tiene suficientes datos válidos")

    idx = np.argsort(t)
    t = t[idx]; v = v[idx]
    return t, v

def nearest_speed(t_arr, v_arr, t_query: float) -> float:
    """
    Devuelve v más cercana a t_query usando búsqueda binaria.
    """
    i = int(np.searchsorted(t_arr, t_query))
    if i <= 0:
        return float(v_arr[0])
    if i >= len(t_arr):
        return float(v_arr[-1])

    t0, t1 = t_arr[i-1], t_arr[i]
    if abs(t_query - t0) <= abs(t1 - t_query):
        return float(v_arr[i-1])
    return float(v_arr[i])

def main():
    # ---- cargar speed lookup (una vez) ----
    speed_t, speed_v = load_speed_lookup(SPEED_CSV, time_col="sim_time", speed_col="speed_m_s")

    #Offset opcional en caso de tener delay
    SPEED_TIME_OFFSET = 0.0

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CARLA Replay")
    clock = pygame.time.Clock()

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)

    print(client.show_recorder_file_info(LOG_FILE, False))

    # Lanzar replay
    client.replay_file(LOG_FILE, 0.0, 0.0, 0)
    world = client.get_world()

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

    frame_q = Queue(maxsize=1)

    def _safe_put(q: Queue, item):
        try:
            q.put_nowait(item)
        except Exception:
            try:
                q.get_nowait()
            except Exception:
                pass
            q.put_nowait(item)

    def on_image(image: carla.Image):
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        bgr  = bgra[:, :, :3].copy()
        rgb  = bgr[:, :, ::-1]
        _safe_put(frame_q, (rgb, bgr, (image.width, image.height)))

    cam.listen(on_image)

    start_sim = world.get_snapshot().timestamp.elapsed_seconds
    duration  = get_log_duration(client, LOG_FILE)
    end_sim   = start_sim + duration

    t0_sim = 0.0

    try:
        while True:
            clock.tick(FPS)

            snap = world.get_snapshot()
            sim_time = snap.timestamp.elapsed_seconds

            if sim_time >= end_sim:
                print("Replay finalizado.")
                break

            try:
                rgb, bgr, (w, h) = frame_q.get_nowait()
            except Exception:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        raise KeyboardInterrupt
                continue

            if t0_sim == 0.0:
                t0_sim = sim_time

            rel_time = sim_time - t0_sim

            # Dibujar
            surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            # Máscaras / heading
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            mask_y = cv2.inRange(hsv, np.array([18, 50, 150]), np.array([40, 255, 255]))
            mask_w = cv2.inRange(hsv, np.array([0, 0, 200]),  np.array([180, 30, 255]))
            mask_c = np.zeros(mask_w.shape, np.uint8)
            mask_c[mask_w > 0] = 1
            mask_c[mask_y > 0] = 2

            mask_rgb = np.zeros_like(rgb)
            mask_rgb[mask_c == 1] = [255,255,255]
            mask_rgb[mask_c == 2] = [255,255,0]

            yrow = int(0.53 * h)
            row = mask_c[yrow]
            white_idx = np.where(row == 1)[0]
            cx = None
            if len(white_idx) > 10:
                cx = (white_idx[0] + white_idx[-1]) // 2
            img_cx  = w // 2
            err_px  = (img_cx - cx) if cx is not None else 0
            dy_px   = h - yrow
            heading = float(np.degrees(np.arctan2(err_px, dy_px)))

            ctrl = ego.get_control()
            throttle = float(ctrl.throttle)
            steer    = float(np.clip(ctrl.steer, -1.0, 1.0))
            brake    = float(ctrl.brake)

            # ===== speed más cercano (en tiempo real) =====
            speed_mps = nearest_speed(speed_t, speed_v, rel_time + SPEED_TIME_OFFSET)

            # ===== posición mundo =====
            loc = ego.get_location()
            x, y, z = float(loc.x), float(loc.y), float(loc.z)

            # Guardar en CSV + imágenes
            guardar_dato(rel_time, bgr, mask_rgb, throttle, steer, brake, speed_mps, heading, x, y, z)

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
