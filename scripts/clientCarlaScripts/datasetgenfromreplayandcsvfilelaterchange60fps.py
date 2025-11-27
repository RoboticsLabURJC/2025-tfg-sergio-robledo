#!/usr/bin/env python3
import os, re, csv, time
import carla, pygame
import numpy as np
import cv2
from datetime import datetime
import queue
from queue import Queue
import pandas as pd
import os


SPEED_CSV = "/home/sergior/Downloads/carla_recorder_replay/mispruebas/secondrecord/speedtrack03CC.csv"
LOG_FILE = "/tmp/Track03CCSpeed.log"
WIDTH, HEIGHT = 800, 600
FPS = 60.0
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
        # 'timestamp' GUARDARÁ sim_time (float).
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

def guardar_dato(timestamp, bgr, mask_rgb, throttle, steer, brake, speed, heading):
    rgb_name  = f"{timestamp}_rgb_{DATASET_ID}.png"
    mask_name = f"{timestamp}_mask_{DATASET_ID}.png"
    cv2.imwrite(os.path.join(RGB_DIR,  rgb_name),  bgr)
    cv2.imwrite(os.path.join(MASK_DIR, mask_name), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([f"/rgb/{rgb_name}", f"/masks/{mask_name}", timestamp,
                                throttle, steer, brake, speed, heading, _estado_from_steer(steer)])

def align_speed(
    dataset_csv: str,
    speed_csv: str,
    dst_speed_col: str = "speed",
    src_speed_col: str = "speed_m_s",
    src_time_col: str = "sim_time"
):
    import os
    import pandas as pd
    import numpy as np

    if not os.path.isfile(dataset_csv):
        print(f"[ERROR] Dataset {dataset_csv} does not exist")
        return
    if not os.path.isfile(speed_csv):
        print(f"[ERROR] No existe CSV de velocidades: {speed_csv}")
        return

    df_dst = pd.read_csv(dataset_csv)
    df_src = pd.read_csv(speed_csv)

    # Checks
    for col in ["timestamp", dst_speed_col]:
        if col not in df_dst.columns:
            print(f"[ERROR] Dataset does not have col '{col}'.")
            return
    for col in [src_time_col, src_speed_col]:
        if col not in df_src.columns:
            print(f"[ERROR] Speed CSV not containing col'{col}'.")
            return
    if df_dst.empty or df_src.empty:
        print("[WARN] Empty Dataset or speed CSV")
        return

    # Num conversion
    df_dst = df_dst.copy()
    df_src = df_src.copy()

    df_dst["timestamp"] = pd.to_numeric(df_dst["timestamp"], errors="coerce")
    df_src[src_time_col] = pd.to_numeric(df_src[src_time_col], errors="coerce")
    df_src[src_speed_col] = pd.to_numeric(df_src[src_speed_col], errors="coerce")

    df_dst = df_dst.dropna(subset=["timestamp"])
    df_src = df_src.dropna(subset=[src_time_col, src_speed_col])

    if df_dst.empty or df_src.empty:
        print("[WARN] NaN data filtered and no data left.")
        return

    # Time order
    df_dst_sorted = df_dst.sort_values("timestamp").reset_index(drop=False)
    df_src_sorted = df_src.sort_values(src_time_col).reset_index(drop=True)

    # Asign speed using nearest neighbor
    merged = pd.merge_asof(
        df_dst_sorted,
        df_src_sorted[[src_time_col, src_speed_col]],
        left_on="timestamp",
        right_on=src_time_col,
        direction="nearest"
    )

    # Copy speed col to original dataframe
    df_dst.loc[merged["index"], dst_speed_col] = merged[src_speed_col].values

    # Guardar
    df_dst.to_csv(dataset_csv, index=False)

    # Info
    time_diff = np.abs(merged["timestamp"] - merged[src_time_col])
    print(f"  - Nº rows dataset:    {len(df_dst)}")
    print(f"  - Nº rows speed_csv:  {len(df_src)}")
    print(f"  - Av. time dif:   {time_diff.mean():.4f} s")
    print(f"  - Max. diff in time:     {time_diff.max():.4f} s")


def main():
    global prev_pos
    
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

    # (En replay no aplicamos modo síncrono)
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
    duration  = get_log_duration(client, LOG_FILE)
    end_sim   = start_sim + duration


    t0_sim = 0.0

    # When writing speed as recording, initial offset is 2 seconds delay so that
    # the car can spawn

    #START_OFFSET_S = 2.0
    
    try:
        while True:
            clock.tick(FPS) 

            snap = world.get_snapshot()
            sim_time = snap.timestamp.elapsed_seconds

            # salir al final del log
            if sim_time >= end_sim:
                print("Replay finalizado.")
                break

            try:
                rgb, bgr, (w, h) = frame_q.get_nowait()
            except queue.Empty:
             
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        raise KeyboardInterrupt
                continue

            if t0_sim == 0.0:
                t0_sim = sim_time

            rel_time = sim_time - t0_sim

            print(rel_time)
            # if rel_time < START_OFFSET_S:
            #     for e in pygame.event.get():
            #         if e.type == pygame.QUIT:
            #             raise KeyboardInterrupt
            #     continue

            # Dibujar
            surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            # Máscaras / heading
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            mask_y = cv2.inRange(hsv, np.array([18, 50, 150]), np.array([40, 255, 255]))
            mask_w = cv2.inRange(hsv, np.array([0, 0, 200]),  np.array([180, 30, 255]))
            mask_c = np.zeros(mask_w.shape, np.uint8); mask_c[mask_w>0]=1; mask_c[mask_y>0]=2
            mask_rgb = np.zeros_like(rgb); mask_rgb[mask_c==1]=[255,255,255]; mask_rgb[mask_c==2]=[255,255,0]

            y = int(0.53 * h)
            row = mask_c[y]; white_idx = np.where(row==1)[0]
            cx = None
            if len(white_idx) > 10:
                cx = (white_idx[0] + white_idx[-1]) // 2
            img_cx  = w//2
            err_px  = (img_cx - cx) if cx is not None else 0
            dy_px   = h - y
            heading = float(np.degrees(np.arctan2(err_px, dy_px)))

            ctrl = ego.get_control()
            throttle = float(ctrl.throttle)
            steer    = max(-1.0, min(1.0, float(ctrl.steer)))
            brake    = float(ctrl.brake)
            speed = 0.0

            guardar_dato(rel_time, bgr, mask_rgb, throttle, steer, brake, speed, heading)

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
        try:
            align_speed(
                CSV_PATH,
                SPEED_CSV,
                dst_speed_col="speed",
                src_speed_col="speed_m_s"
            )
        except Exception as e:
            print(f"[ERROR] Acople secuencial de velocidades: {e}")



if __name__ == "__main__":
    main()
