#!/usr/bin/env python3
import carla
import time
import math
import argparse
import numpy as np
import pandas as pd
import cv2
from collections import deque

IMG_W, IMG_H = 1660, 1000
FPS = 30.0

# -------------------------
# CSV helpers
# -------------------------
def pick_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def load_csv(path):
    df = pd.read_csv(path)

    tcol = pick_col(df, ["timestamp", "time", "sim_time", "t", "ts"])
    xcol = pick_col(df, ["x", "pos_x", "X"])
    ycol = pick_col(df, ["y", "pos_y", "Y"])
    zcol = pick_col(df, ["z", "pos_z", "Z"])

    if tcol is None:
        raise ValueError(f"[{path}] falta columna de tiempo (timestamp/time/sim_time/t/ts)")
    if xcol is None or ycol is None:
        raise ValueError(f"[{path}] faltan columnas x/y (x,y). Columnas: {list(df.columns)}")

    df = df.copy()
    df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    if zcol is not None:
        df[zcol] = pd.to_numeric(df[zcol], errors="coerce")

    df = df.dropna(subset=[tcol, xcol, ycol]).sort_values(tcol).reset_index(drop=True)

    df["t"] = df[tcol] - df[tcol].iloc[0]
    df["x"] = df[xcol]
    df["y"] = df[ycol]
    if zcol is None or df[zcol].isna().all():
        df["z"] = 0.2
    else:
        df["z"] = df[zcol].ffill().fillna(0.2)

    return df[["t", "x", "y", "z"]]

def interp_xyz(df, t):
    if t < df.t.iloc[0] or t > df.t.iloc[-1]:
        return None

    arr_t = df.t.values
    i = int(np.searchsorted(arr_t, t))

    if i <= 0:
        r = df.iloc[0]
        return float(r.x), float(r.y), float(r.z)
    if i >= len(df):
        r = df.iloc[-1]
        return float(r.x), float(r.y), float(r.z)

    r0 = df.iloc[i-1]
    r1 = df.iloc[i]
    t0, t1 = float(r0.t), float(r1.t)
    a = 0.0 if (t1 - t0) < 1e-9 else (t - t0) / (t1 - t0)

    x = (1-a)*float(r0.x) + a*float(r1.x)
    y = (1-a)*float(r0.y) + a*float(r1.y)
    z = (1-a)*float(r0.z) + a*float(r1.z)
    return x, y, z

# -------------------------
# Projection world -> image
# -------------------------
def build_intrinsics(w, h, fov_deg_h):
    hfov = math.radians(float(fov_deg_h))
    fx = w / (2.0 * math.tan(hfov / 2.0))
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (h / w))
    fy = h / (2.0 * math.tan(vfov / 2.0))
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float32)

def world_to_camera_matrix(cam_actor):
    T_wc = np.array(cam_actor.get_transform().get_matrix(), dtype=np.float32)
    return np.linalg.inv(T_wc)

def project(cam_actor, loc: carla.Location, w, h):
    K = build_intrinsics(w, h, float(cam_actor.attributes["fov"]))
    T_cw = world_to_camera_matrix(cam_actor)

    Pw = np.array([loc.x, loc.y, loc.z, 1.0], dtype=np.float32)
    Pc = T_cw @ Pw
    Xc, Yc, Zc, _ = Pc

    if Xc <= 0.001:
        return None

    uvw = K @ np.array([Yc / Xc, -Zc / Xc, 1.0], dtype=np.float32)
    u, v = float(uvw[0]), float(uvw[1])
    if 0 <= u < w and 0 <= v < h:
        return int(u), int(v)
    return None

# -------------------------
# Your CAM + SPAWN mappings
# -------------------------
def get_spawn_point(cam_index: int) -> carla.Transform:
    if cam_index == 1:
        return carla.Transform(carla.Location(x=3, y=-1, z=0.2), carla.Rotation(yaw=-90))
    if cam_index == 3:
        return carla.Transform(carla.Location(x=-8.5, y=-14.7, z=0.2), carla.Rotation(yaw=-15))
    if cam_index == 2:
        return carla.Transform(carla.Location(x=17, y=-4.2, z=0.2), carla.Rotation(yaw=-30))
    if cam_index == 4:
        return carla.Transform(carla.Location(x=-10, y=21.2, z=1), carla.Rotation(yaw=-15))
    if cam_index == 5:
        return carla.Transform(carla.Location(x=-3.7, y=-4, z=0.2), carla.Rotation(yaw=-120))
    if cam_index == 6:
        return carla.Transform(carla.Location(x=-1.5, y=33, z=0.2), carla.Rotation(yaw=0))
    if cam_index == 7:
        return carla.Transform(carla.Location(x=-1.5, y=71.5, z=0.2), carla.Rotation(yaw=180))
    if cam_index == 8:
        return carla.Transform(carla.Location(x=-65, y=17.5, z=0.2), carla.Rotation(yaw=150))
    if cam_index == 9:
        return carla.Transform(carla.Location(x=-65, y=94.5, z=0.2), carla.Rotation(yaw=55))
    if cam_index == 10:
        return carla.Transform(carla.Location(x=-67, y=228, z=0.2), carla.Rotation(yaw=-25))
    if cam_index == 11:
        return carla.Transform(carla.Location(x=-67, y=318, z=0.2), carla.Rotation(yaw=-25))
    if cam_index == 12:
        return carla.Transform(carla.Location(x=-29.2, y=-12, z=0.2), carla.Rotation(yaw=-120))
    if cam_index == 13:
        return carla.Transform(carla.Location(x=-60.2, y=-15, z=0.2), carla.Rotation(yaw=-120))
    raise ValueError("cam_index inválido (1..13)")

def get_cam_location(cam_index: int) -> carla.Location:
    cam_locations = {
        1: carla.Location(x=1.5,  y=-2.5,  z=2.0),
        2: carla.Location(x=17.0, y=-3.0,  z=2.8),
        3: carla.Location(x=-12,  y=-16.5, z=6.0),
        4: carla.Location(x=-9.5, y=13.5,  z=10.5),
        5: carla.Location(x=-8.9, y=-3.9,  z=4.0),
        6: carla.Location(x=-1.5, y=38.9,  z=15.0),
        7: carla.Location(x=-1.5, y=63,    z=13.0),
        8: carla.Location(x=-65,  y=38.9,  z=30.0),
        9: carla.Location(x=-67,  y=120.9, z=30.0),
        10: carla.Location(x=-67, y=227,   z=33.0),
        11: carla.Location(x=-67, y=317,   z=30.0),
        12: carla.Location(x=-37.9,y=-10,  z=10.0),
        13: carla.Location(x=-68, y=-12,   z=8.0),
    }
    if cam_index not in cam_locations:
        raise ValueError("cam_index inválido (1..13)")
    return cam_locations[cam_index]

def get_cam_rotation(cam_index: int) -> carla.Rotation:
    if cam_index in (1, 9, 10):
        return carla.Rotation(pitch=-90)
    return carla.Rotation(pitch=-90, yaw=-90)

# -------------------------
# Drawing helpers
# -------------------------
def put_shadow_text(img, text, org, scale=0.9, th=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = org
    cv2.putText(img, text, (x, y), font, scale, (0,0,0), th+3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (255,255,255), th, cv2.LINE_AA)

# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_human", required=True)
    ap.add_argument("--csv_inf", required=True)
    ap.add_argument("--cam", type=int, default=3)
    ap.add_argument("--lap_zone", type=float, default=0.8, help="radio (m) para detectar vuelta")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--speed", type=float, default=1.0, help="factor reproducción")
    args = ap.parse_args()

    df_h = load_csv(args.csv_human)
    df_i = load_csv(args.csv_inf)

    T_end = float(max(df_h.t.iloc[-1], df_i.t.iloc[-1]))

    sp = get_spawn_point(args.cam)
    finish_xy = np.array([sp.location.x, sp.location.y], dtype=np.float64)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    weather = carla.WeatherParameters(
        cloudiness=80.0,
        precipitation=0.0,
        sun_altitude_angle=90.0,
        fog_density=0.0,
        wetness=0.0
    )
    world.set_weather(weather)

    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMG_W))
    cam_bp.set_attribute("image_size_y", str(IMG_H))
    cam_bp.set_attribute("fov", "120")
    cam_bp.set_attribute("sensor_tick", "0.0")

    cam_tf = carla.Transform(get_cam_location(args.cam), get_cam_rotation(args.cam))
    camera = world.spawn_actor(cam_bp, cam_tf)

    frame = {"bgr": None}
    def on_image(img: carla.Image):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))[:, :, :3]
        frame["bgr"] = arr

    camera.listen(on_image)

    def init_state():
        return {"laps": 0, "in_zone": True, "last_lap_t": 0.0, "prev_lap": 0.0}
    st_h = init_state()
    st_i = init_state()

    def update_laps(state, xy, tnow):
        dist = float(np.linalg.norm(xy - finish_xy))
        if dist < args.lap_zone and not state["in_zone"]:
            state["laps"] += 1
            lap_time = tnow - state["last_lap_t"]
            state["prev_lap"] = lap_time
            state["last_lap_t"] = tnow
            state["in_zone"] = True
        elif dist >= args.lap_zone:
            state["in_zone"] = False

    # -------------------------
    # ESTELAS
    # -------------------------
    TRAIL_LEN = 100       # sube/baja la longitud
    TRAIL_THICK = 5
    trail_h = deque(maxlen=TRAIL_LEN)
    trail_i = deque(maxlen=TRAIL_LEN)

    t = 0.0
    last_wall = time.time()

    try:
        while True:
            now = time.time()
            dt_wall = now - last_wall
            last_wall = now

            t += dt_wall * float(args.speed)
            if t > T_end:
                t = T_end

            h = interp_xyz(df_h, t)
            i = interp_xyz(df_i, t)

            if frame["bgr"] is not None:
                vis = frame["bgr"].copy()

                # ---- DIBUJAR ESTELAS SIEMPRE (antes de los puntos) ----
                if len(trail_h) >= 2:
                    pts_h = np.array(trail_h, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis, [pts_h], False, (0, 255, 0), TRAIL_THICK)

                if len(trail_i) >= 2:
                    pts_i = np.array(trail_i, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis, [pts_i], False, (0, 0, 255), TRAIL_THICK)

                # finish marker
                pf = project(camera, carla.Location(x=float(finish_xy[0]), y=float(finish_xy[1]), z=0.2), IMG_W, IMG_H)
                if pf:
                    cv2.circle(vis, pf, 16, (255,255,255), 2)
                    cv2.putText(vis, "FINISH", (pf[0]+18, pf[1]+6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

                # HUMAN point
                if h is not None:
                    xh, yh, zh = h
                    update_laps(st_h, np.array([xh, yh], dtype=np.float64), t)
                    ph = project(camera, carla.Location(x=xh, y=yh, z=zh), IMG_W, IMG_H)
                    if ph:
                        trail_h.append(ph)  # <-- guardar para estela
                        cv2.circle(vis, ph, 9, (0,255,0), -1)
                        cv2.putText(vis, "HUMAN", (ph[0]+12, ph[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2, cv2.LINE_AA)

                # INF point
                if i is not None:
                    xi, yi, zi = i
                    update_laps(st_i, np.array([xi, yi], dtype=np.float64), t)
                    pi = project(camera, carla.Location(x=xi, y=yi, z=zi), IMG_W, IMG_H)
                    if pi:
                        trail_i.append(pi)
                        cv2.circle(vis, pi, 9, (0,0,255), -1)
                        cv2.putText(vis, "IA", (pi[0]+12, pi[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2, cv2.LINE_AA)

                # distance between points
                if h is not None and i is not None:
                    d = float(np.linalg.norm(np.array([h[0]-i[0], h[1]-i[1]])))
                    put_shadow_text(vis, f"Dist(HUMAN, IA): {d:.2f} m", (20, 150), 0.85, 2)

                # HUD laps/times
                put_shadow_text(vis, f"t={t:.2f}s / {T_end:.2f}s   (x{args.speed:.2f})", (20, 45), 0.9, 2)
                put_shadow_text(vis, f"HUMAN laps={st_h['laps']}  last={st_h['prev_lap']:.2f}s", (20, 85), 0.9, 2)
                put_shadow_text(vis, f"IA   laps={st_i['laps']}  last={st_i['prev_lap']:.2f}s", (20, 120), 0.9, 2)

                cv2.imshow("CSV overlay on REAL CARLA camera", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            if t >= T_end:
                if frame["bgr"] is not None:
                    vis = frame["bgr"].copy()
                    put_shadow_text(vis, "END (press q)", (IMG_W//2 - 140, IMG_H//2), 1.0, 2)
                    cv2.imshow("CSV overlay on REAL CARLA camera", vis)
                while True:
                    k2 = cv2.waitKey(50) & 0xFF
                    if k2 == ord("q") or k2 == 27:
                        return

    finally:
        try:
            camera.stop()
        except:
            pass
        try:
            camera.destroy()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
