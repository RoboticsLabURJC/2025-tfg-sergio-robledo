import carla, time, pygame, numpy as np, cv2, torch, queue
from torchvision import transforms
from queue import Queue
from utils.pilotnet import PilotNet
from PIL import Image
import sys
from collections import deque
import math
import argparse


MODEL_PATH = "experiments/exp_debug_1769708013/trained_models/pilot_net_model_best_123.pth"
image_shape = (66, 200, 4)        # 4 canales: RGB + speed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PilotNet(image_shape, num_labels=2).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()


VEHICLE_MODEL = 'vehicle.finaldeepracer.aws_deepracer'
WIDTH, HEIGHT = 800, 600
FPS = 30.0
FIXED_DT = 1.0 / FPS

lap_time = 0.0

def _build_intrinsics(w, h, fov_deg_h):
    # FOV de CARLA es horizontal. Calculamos fx, fy, cx, cy."""
    hfov = math.radians(fov_deg_h)
    fx = w / (2.0 * math.tan(hfov / 2.0))
    # obtener vfov y fy con píxel cuadrado
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (h / w))
    fy = h / (2.0 * math.tan(vfov / 2.0))
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,   1]], dtype=np.float32)
    return K

def _world_to_camera_matrix(cam_actor):
    # Extrínseca mundo→cámara (4x4) usando la pose real del sensor.
    T_wc = np.array(cam_actor.get_transform().get_matrix(), dtype=np.float32)  # cámara→mundo
    T_cw = np.linalg.inv(T_wc)                                                # invertimos: mundo→cámara
    return T_cw

def project_world_to_image_precise(cam_actor, world_point, img_w, img_h):
    

    # coord. UE/CARLA: X adelante, Y derecha, Z arriba
    # coord. cámara (CARLA): X adelante, Y derecha, Z arriba
    # imagen: u derecha, v abajo
    
    # intrínsecas desde el FOV horizontal del sensor
    fov_h = float(cam_actor.attributes['fov'])
    K = _build_intrinsics(img_w, img_h, fov_h)

    # extrínseca
    T_cw = _world_to_camera_matrix(cam_actor)

    # punto homogéneo en mundo
    Pw = np.array([world_point.x, world_point.y, world_point.z, 1.0], dtype=np.float32)

    # a coords. cámara
    Pc = T_cw @ Pw
    Xc, Yc, Zc, _ = Pc

    # detrás de la cámara → no proyecta
    if Xc <= 0.001:
        return None

    # pinhole (u~Y/X, v~-Z/X)
    uvw = K @ np.array([Yc / Xc, -Zc / Xc, 1.0], dtype=np.float32)
    u, v = float(uvw[0]), float(uvw[1])

    # fuera de imagen
    if u < -10 or u > img_w + 10 or v < -10 or v > img_h + 10:
        return None

    return int(round(u)), int(round(v))

def main():
    global lap_time

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["trail", "heatmap"], default="trail",
                        help="trail = estela actual | heatmap = mapa de calor por velocidad")
    parser.add_argument("--cam", type=int, default=1, help="indice de camara cenital")
    args, unknown = parser.parse_known_args()

    mode = args.mode
    cam_index = args.cam

    if len(sys.argv) > 1:
        try:
            if cam_index < -1 or cam_index > 13:
                print("Índice fuera de rango. Usando 1 por defecto.")
                cam_index = 1
        except ValueError:
            print("Argumento inválido. Usando 1 por defecto.")

    cam_locations = {
        1: carla.Location(x=1.5,  y=-2.5,  z=2.0),
        2: carla.Location(x=17.0, y=-3.0,  z=2.8),
        3: carla.Location(x=-12,  y=-16.5, z=6.0),
        4: carla.Location(x=-9.5, y=13.5,  z=10.5),
        5: carla.Location(x=-8.9, y=-3.9,  z=4.0),
        6: carla.Location(x=-1.5, y=38.9,  z=15.0),
        7: carla.Location(x=-1.5, y=63,  z=13.0),
        8: carla.Location(x=-65, y=38.9,  z=30.0),
        9: carla.Location(x=-67, y=120.9,  z=30.0),
        10: carla.Location(x=-67, y=227,  z=33.0),
        11: carla.Location(x=-67, y=317,  z=30.0),
        12: carla.Location(x=-37.9, y=-10,  z=10.0),
        13: carla.Location(x=-68, y=-12,  z=8.0),
    }

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    # settings = world.get_settings()
    # settings.synchronous_mode = True
    # settings.fixed_delta_seconds = FIXED_DT
    #world.apply_settings(settings)

    weather = carla.WeatherParameters(
        cloudiness=80.0, precipitation=0.0, sun_altitude_angle=90.0,
        fog_density=0.0, wetness=0.0
    )
    world.set_weather(weather)

    bp = world.get_blueprint_library()
    vehicle_bp = bp.find(VEHICLE_MODEL)

    # Spawn según cámara elegida
    if cam_index == 1:
        spawn_point = carla.Transform(carla.Location(x=3, y=-1, z=0.2), carla.Rotation(yaw=-90))
    if cam_index == 3:
        spawn_point = carla.Transform(carla.Location(x=-8.5, y=-14.7, z=0.2), carla.Rotation(yaw=-15))
    if cam_index == 2:
        spawn_point = carla.Transform(carla.Location(x=17, y=-4.2, z=0.2), carla.Rotation(yaw=-30))
    if cam_index == 4:
        spawn_point = carla.Transform(carla.Location(x=-10, y=21.2, z=1), carla.Rotation(yaw=-15))
    if cam_index == 5:
        spawn_point = carla.Transform(carla.Location(x=-3.7, y=-4, z=0.2), carla.Rotation(yaw=-120))
    if cam_index == 6:
        #gillesvilleneuve
        spawn_point = carla.Transform(carla.Location(x=-1.5, y=33, z=0.2), carla.Rotation(yaw=0))    
    if cam_index == 7:
        #interlagosautodromojosecarlospace
        spawn_point = carla.Transform(carla.Location(x=-1.5, y=71.5, z=0.2), carla.Rotation(yaw=180))
    if cam_index == 8:
        #nurburgring
        spawn_point = carla.Transform(carla.Location(x=-65, y=17.5, z=0.2), carla.Rotation(yaw=150))
    if cam_index == 9:
        #spafrancorchamps
        spawn_point = carla.Transform(carla.Location(x=-65, y=94.5, z=0.2), carla.Rotation(yaw=55))
    if cam_index == 10:
        #silverstone
        spawn_point = carla.Transform(carla.Location(x=-67, y=228, z=0.2), carla.Rotation(yaw=-25))
    if cam_index == 11:
        #lagoseco
        spawn_point = carla.Transform(carla.Location(x=-67, y=318, z=0.2), carla.Rotation(yaw=-25))
    
    if cam_index == 12:
        #track12
        spawn_point = carla.Transform(carla.Location(x=-29.2, y=-12, z=0.2), carla.Rotation(yaw=-120))
    
    if cam_index == 13:
        #track12
        spawn_point = carla.Transform(carla.Location(x=-60.2, y=-15, z=0.2), carla.Rotation(yaw=-120))
    

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if not vehicle:
        print("Error al spawnear el vehículo"); raise SystemExit
    print("Vehículo spawneado")

    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1660')
    camera_bp.set_attribute('image_size_y', '1000')
    camera_bp.set_attribute('fov', '120')         
    cam_bp_net = bp.find('sensor.camera.rgb')
    cam_bp_net.set_attribute('image_size_x', str(WIDTH))
    cam_bp_net.set_attribute('image_size_y', str(HEIGHT))
    cam_bp_net.set_attribute('fov', '90')
    cam_bp_net.set_attribute('sensor_tick', '0.0')

    cam_location = cam_locations[cam_index]
    # Yaw de la cámara cenital (lo usaremos para rotar el plano)
    if cam_index == 1 or cam_index == 9 or cam_index == 10:
        cam_rotation = carla.Rotation(pitch=-90)          # yaw = 0 por defecto
    else:
        cam_rotation = carla.Rotation(pitch=-90, yaw=-90) # yaw = -90

    cam_transform = carla.Transform(cam_location, cam_rotation)
    camera = world.spawn_actor(camera_bp, cam_transform)

    cam_net_tf = carla.Transform(carla.Location(x=0.13, z=0.13), carla.Rotation(pitch=-30))
    cam_net = world.spawn_actor(cam_bp_net, cam_net_tf, attach_to=vehicle)

    rgb_net_q = Queue(maxsize=1)          # FOV 90 para la red
    camera_image = {"data": None}  # frame BGR de la cenital


    # Heatmap (misma resolución que la cámara cenital: 1000x1660)
    HEAT_H = int(camera_bp.get_attribute('image_size_y').as_int())  # 1000
    HEAT_W = int(camera_bp.get_attribute('image_size_x').as_int())  # 1660

    # guardamos la velocidad (normalizada 0..1) por píxel: máximo histórico
    heatmap_max = np.zeros((HEAT_H, HEAT_W), dtype=np.float32)
    heat_brush_radius = 10    # radio del "pincel" en píxeles

    def _safe_put(q: Queue, item):
        try:
            q.put_nowait(item)
        except queue.Full:
            try: q.get_nowait()
            except queue.Empty: pass
            q.put_nowait(item)

    # buffer de estela
    trail_px = deque(maxlen=500)   # guarda últimos puntos proyectados (u,v)

    def cb_net(image: carla.Image):
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        bgr  = bgra[:, :, :3]
        _safe_put(rgb_net_q, bgr[:, :, ::-1])

    def on_image(image: carla.Image):
        # guardamos el frame cenital (BGR)
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
        camera_image["data"] = array

    camera.listen(on_image)
    cam_net.listen(cb_net)

    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
    time.sleep(1.0)

    start_sim = world.get_snapshot().timestamp.elapsed_seconds
    infer_tf = transforms.Compose([
        transforms.Resize((66, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    initial_veh_loc = vehicle.get_location()
    init_xy = np.array([initial_veh_loc.x, initial_veh_loc.y], dtype=float)

    init_time = time.time()
    last_lap_ts = init_time

    lap_counter = 0
    lap_zone = 0.5
    in_lap_zone = True


    last_net = None
    clock = pygame.time.Clock()

    try:
        while True:

            # Inferencia y control del vehículo
            try: last_net   = rgb_net_q.get_nowait()
            except queue.Empty: pass

            if last_net is None:
                continue

            clock.tick(30)
            #print(time.time())
            rgb_net = last_net

            veh_loc = vehicle.get_location()
            cur_pos = np.array([veh_loc.x, veh_loc.y, veh_loc.z], dtype=float)

            #Lap check-----------
            now = time.time()

            veh_loc = vehicle.get_location()
            cur_xy = np.array([veh_loc.x, veh_loc.y], dtype=float)

            dist = np.linalg.norm(cur_xy - init_xy)

            if dist < lap_zone and not in_lap_zone:
                lap_counter += 1
                lap_time = now - last_lap_ts
                last_lap_ts = now
                print(f"Lap Nº: {lap_counter} lap time: {lap_time:.2f}s")
                in_lap_zone = True
            elif dist >= lap_zone:
                in_lap_zone = False

            #-------------------

            vel = vehicle.get_velocity()
            speed = float(np.linalg.norm([vel.x, vel.y, vel.z]))

            # Calcula velocidad (m/s) y escálala igual que en train (÷3.5 y clip 0..1)
            speed_norm = float(np.clip(speed / 3.5, 0.0, 1.0))  # [0,1]

            #  Generar máscara HSV desde la RGB capturada
            hsv = cv2.cvtColor(rgb_net, cv2.COLOR_RGB2HSV)

            lower_white  = np.array([0, 0, 200])
            upper_white  = np.array([180, 50, 255])

            lower_yellow = np.array([15, 70, 70])
            upper_yellow = np.array([35, 255, 255])

            mask_w = cv2.inRange(hsv, lower_white, upper_white)
            mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)

            mask = np.zeros_like(rgb_net)
            mask[mask_w > 0] = (255, 255, 255)   # clase 1
            mask[mask_y > 0] = (255, 255,   0)   # clase 2

            # Pintar en negro la fila 0–100
            mask[0:100, :, :] = 0

            # Convertir máscara a tensor
            mask_img = Image.fromarray(mask)                 # PIL image
            x = infer_tf(mask_img).unsqueeze(0)              # (1,3,66,200)
            cv2.imshow("Mask (entrada red)", cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
            # Canal de velocidad 
            speed_plane = torch.full((1,1,66,200), speed_norm,
                                    dtype=x.dtype, device=x.device)

            x4 = torch.cat([x, speed_plane], dim=1).to(device)  # (1,4,66,200)

            #  Inferencia
            with torch.no_grad():
                out = model(x4)
                steer, throttle = out[0].tolist()

            steer    = float(np.clip(steer,    -1.0, 1.0))
            throttle = float(np.clip(throttle,  0.0, 0.95))
            #print(f"[NET] speed={speed:.2f} m/s (norm={speed_norm:.2f}) | steer={steer:.3f} | throttle={throttle:.3f}")

            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

            #  Obtener posición y proyectar a la imagen cenital

            # Proyección (usar FOV y tamaño de la CÁMARA CENITAL)
            u_v = project_world_to_image_precise(
                cam_actor=camera,
                world_point=veh_loc,
                img_w=int(camera_bp.get_attribute('image_size_x').as_int()),
                img_h=int(camera_bp.get_attribute('image_size_y').as_int())
            )
            if u_v is not None:
                u, v = u_v

                if mode == "trail":
                    trail_px.append((u, v))

                elif mode == "heatmap":
                    # velocidad real para la leyenda
                    vel = vehicle.get_velocity()
                    speed_mps = float(np.linalg.norm([vel.x, vel.y, vel.z]))
                    speed_kmh = speed_mps * 3.6

                    # normalización fija 
                    VMAX_KMH = 11.5 
                    s_norm = float(np.clip(speed_kmh / VMAX_KMH, 0.0, 1.0))

                    # actualiza un círculo: heatmap_max = max(heatmap_max, s_norm) dentro del brush
                    mask = np.zeros((HEAT_H, HEAT_W), dtype=np.uint8)
                    cv2.circle(mask, (u, v), heat_brush_radius, 255, thickness=-1)

                    # “max” por máscara
                    np.maximum(heatmap_max, s_norm, out=heatmap_max, where=(mask > 0))


            # Pintar estela en el frame cenital
            if camera_image["data"] is not None:
                frame = camera_image["data"].copy()  # BGR (1000x1660)

                if mode == "trail":
                    # dibuja líneas entre puntos consecutivos
                    for i in range(1, len(trail_px)):
                        p1 = trail_px[i-1]; p2 = trail_px[i]
                        if (0 <= p1[0] < frame.shape[1] and 0 <= p1[1] < frame.shape[0] and
                            0 <= p2[0] < frame.shape[1] and 0 <= p2[1] < frame.shape[0]):
                            cv2.line(frame, p1, p2, (0, 0, 255), 2)

                    # punto actual
                    if len(trail_px) > 0:
                        u, v = trail_px[-1]
                        if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                            cv2.circle(frame, (u, v), 5, (255, 0, 0), -1)

                    text = f"Laps: {lap_counter} / prev lap time: {lap_time:.2f}s"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 1.2
                    th = 3

                    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, th)
                    x = (frame.shape[1] - text_w) // 2
                    y = 50  # separación desde arriba

                    cv2.putText(frame, text, (x, y), font, scale, (0,0,0), th+3, cv2.LINE_AA)   # borde
                    cv2.putText(frame, text, (x, y), font, scale, (255,255,255), th, cv2.LINE_AA)

                    cv2.imshow("Vista Cenital", frame)

                elif mode == "heatmap":
                    # Pasamos velocidad normalizada 0..1 a 0..255
                    heat_u8 = (heatmap_max * 255.0).astype(np.uint8)

                    # Color map (BGR)
                    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

                    # Solo pintar donde hay “algo” (evita colorear todo el fondo)
                    active = heat_u8 > 0
                    # Oscurecer ligeramente la imagen base para resaltar el heatmap
                    dark_factor = 0.65  # 1.0 = igual, 0.7 = más oscuro, 0.5 = muy oscuro
                    base = cv2.convertScaleAbs(frame, alpha=dark_factor, beta=0)

                    overlay = base.copy()
                    alpha = 0.70

                    # Mezcla solo en píxeles activos
                    overlay[active] = (base[active] * (1.0 - alpha) + heat_color[active] * alpha).astype(np.uint8)

                    # Leyenda
                    # barra vertical 20px ancho x 220px alto
                    bar_h = 220
                    bar_w = 22
                    x0 = frame.shape[1] - (bar_w + 60)
                    y0 = 20

                    # gradiente 0..255 (abajo->arriba)
                    grad = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
                    grad = np.repeat(grad, bar_w, axis=1)
                    grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_JET)

                    # pegar barra
                    overlay[y0:y0+bar_h, x0:x0+bar_w] = grad_color

                    # etiquetas
                    cv2.putText(overlay, f"{VMAX_KMH:.0f} km/h", (x0-5, y0-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                    cv2.putText(overlay, "0 km/h", (x0-5, y0+bar_h+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

                    cv2.putText(overlay, "Speed heatmap", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

                    text = f"Laps: {lap_counter} / prev lap time: {lap_time:.2f}s"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 1.2
                    th = 3

                    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, th)
                    x = (overlay.shape[1] - text_w) // 2
                    y = 50

                    cv2.putText(overlay, text, (x, y), font, scale, (0,0,0), th+3, cv2.LINE_AA)
                    cv2.putText(overlay, text, (x, y), font, scale, (255,255,255), th, cv2.LINE_AA)

                    cv2.imshow("Vista Cenital", overlay)



            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Finalizando...")
        try: camera.stop(); camera.destroy()
        except: pass
        try: cam_net.stop(); cam_net.destroy()
        except: pass
        try: vehicle.destroy()
        except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
