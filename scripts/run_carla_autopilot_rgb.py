# ------------------------------------------------
# Inferencia automática con dos cámaras: PID (FOV 140) + NN (FOV 90)
# ------------------------------------------------
import carla, time, pygame, numpy as np, cv2, torch
from torchvision import transforms
from utils.pilotnet import PilotNet
from PIL import Image

pid_on = False
prev_timeglobal_var = 0.0
last_error_steer = 0.0
Kp_steer, Kd_steer = 0.1, 1e-5
Kp_throttle = 0.02

MODEL_PATH = "experiments/exp_debug_1760371287/trained_models/pilot_net_model_best_123.pth"
image_shape = (66, 200, 3)
model = PilotNet(image_shape, num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

HOST, PORT = '127.0.0.1', 2000
VEHICLE_MODEL = 'vehicle.finaldeepracer.aws_deepracer'
WIDTH, HEIGHT = 800, 600
FPS = 30.0
FIXED_DT = 1.0 / FPS
WARMUP_SEC = 20.0  # tiempo de PID antes de pasar a la red

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DeepRacer - PID 140° -> Red 90°")

client = carla.Client(HOST, PORT)
client.set_timeout(5.0)
world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = FIXED_DT
world.apply_settings(settings)


weather = carla.WeatherParameters(
    cloudiness=80.0,
    precipitation=0.0,
    sun_altitude_angle=90.0,
    fog_density=0.0,
    wetness=0.0
)
world.set_weather(weather)
bp = world.get_blueprint_library()
vehicle_bp = bp.find(VEHICLE_MODEL)
# Pistas
# -------------------------TRACK01-----------------------------
# spawn_point = carla.Transform(
#     carla.Location(x=3, y=-1, z=0.5),
#     carla.Rotation(yaw=180-90)
# )

#-------------------------TRACK---------------------------------
spawn_point = carla.Transform(
   carla.Location(x=-3.7, y=-4, z=0.5),
   carla.Rotation(yaw=-120)
)


#-------------------------TRACK03---------------------------------
# spawn_point = carla.Transform(
#     carla.Location(x=-8, y=-15, z=0.5),
#     carla.Rotation(yaw=-15)
# )

#-------------------------TRACK02---------------------------------
# spawn_point = carla.Transform(
#    carla.Location(x=17, y=-4.8, z=0.5),
#    carla.Rotation(yaw=180-10)
# )

#-------------------------TRACK04---------------------------------
# spawn_point = carla.Transform(
#     carla.Location(x=-10, y=21.2, z=1),
#     carla.Rotation(yaw=-15)
# )

vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    print("Error al spawnear el vehículo"); raise SystemExit
print("Vehículo spawneado")

# --- cámaras ---
cam_bp_pid = bp.find('sensor.camera.rgb')
cam_bp_pid.set_attribute('image_size_x', str(WIDTH))
cam_bp_pid.set_attribute('image_size_y', str(HEIGHT))
cam_bp_pid.set_attribute('fov', '140')
cam_bp_pid.set_attribute('sensor_tick', '0.0')  # 1 frame por tick

cam_bp_net = bp.find('sensor.camera.rgb')
cam_bp_net.set_attribute('image_size_x', str(WIDTH))
cam_bp_net.set_attribute('image_size_y', str(HEIGHT))
cam_bp_net.set_attribute('fov', '90')
cam_bp_net.set_attribute('sensor_tick', '0.0')


cam_bp_thirdperson = bp.find('sensor.camera.rgb')
cam_bp_thirdperson.set_attribute('image_size_x', str(WIDTH))
cam_bp_thirdperson.set_attribute('image_size_y', str(HEIGHT))
cam_bp_thirdperson.set_attribute('fov', '90')
cam_bp_thirdperson.set_attribute('sensor_tick', '0.0')


cam_tf = carla.Transform(carla.Location(x=0.13, z=0.13), carla.Rotation(pitch=-30))
cam_pid = world.spawn_actor(cam_bp_pid, cam_tf, attach_to=vehicle)
cam_net = world.spawn_actor(cam_bp_net, cam_tf, attach_to=vehicle)

cam_tf_third = carla.Transform(
    carla.Location(x=-1.0, y=0.0, z=0.7),
    carla.Rotation(pitch=-12, yaw=0)  
)
cam_thirdperson = world.spawn_actor(cam_bp_thirdperson, cam_tf_third, attach_to=vehicle)

# buffers de frames
rgb_pid_buf = [None]  # FOV 140 para PID
rgb_net_buf = [None]  # FOV 90 para la red
rgb_third_buf = [None] # Tercera persona

def cb_third(image: carla.Image):
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    bgr  = bgra[:, :, :3]
    rgb_third_buf[0] = bgr[:, :, ::-1]

def cb_pid(image: carla.Image):
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    bgr  = bgra[:, :, :3]
    rgb_pid_buf[0] = bgr[:, :, ::-1]

def cb_net(image: carla.Image):
    global prev_timeglobal_var

    timeglobal_var = image.timestamp 
    fps_toprint = timeglobal_var - prev_timeglobal_var 
    print(1/fps_toprint) 
    prev_timeglobal_var = timeglobal_var

    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    bgr  = bgra[:, :, :3]
    rgb_net_buf[0] = bgr[:, :, ::-1]

cam_thirdperson.listen(cb_third)
cam_pid.listen(cb_pid)
cam_net.listen(cb_net)

vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
time.sleep(1.0)

start_sim = world.get_snapshot().timestamp.elapsed_seconds
running = True

infer_tf = transforms.Compose([
            transforms.Resize((66, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])


def draw_with_pip(main_rgb, pip_rgb):
    # main (tercera persona) a tamaño completo
    if main_rgb is not None:
        main_surf = pygame.surfarray.make_surface(main_rgb.swapaxes(0, 1))
        screen.blit(main_surf, (0, 0))

    # PiP (esquina superior derecha)
    if pip_rgb is not None:
        pip_w = WIDTH // 3
        pip_h = HEIGHT // 3
        pip_small = cv2.resize(pip_rgb, (pip_w, pip_h))
        pip_surf = pygame.surfarray.make_surface(pip_small.swapaxes(0, 1))
        screen.blit(pip_surf, (WIDTH - pip_w - 8, 8))

    pygame.display.flip()


while running:
    world.tick()
    sim_t = world.get_snapshot().timestamp.elapsed_seconds

    # cambio por tiempo (fuera de cualquier if de detección)
    if pid_on and (sim_t - start_sim) >= WARMUP_SEC:
        pid_on = False

    # eventos (ESC para salir, SPACE para alternar manualmente)
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE: running = False
            if e.key == pygame.K_SPACE:  pid_on = not pid_on

    if pid_on:
        # ======== PID con la cámara de 140° ========
        rgb_pid = rgb_pid_buf[0]
        if rgb_pid is None:
            continue

        hsv = cv2.cvtColor(rgb_pid, cv2.COLOR_RGB2HSV)
        mask_y = cv2.inRange(hsv, np.array([18, 50, 150]), np.array([40, 255, 255]))
        mask_w = cv2.inRange(hsv, np.array([0, 0, 200]),  np.array([180, 30, 255]))

        mask_c = np.zeros_like(mask_w, dtype=np.uint8)
        mask_c[mask_w > 0] = 1
        mask_c[mask_y > 0] = 2

        h, w = mask_c.shape
        y = int(0.53 * h)
        row = mask_c[y]
        white_idx = np.where(row == 1)[0]

        cx = None
        if len(white_idx) > 10:
            cx = (white_idx[0] + white_idx[-1]) // 2

        # control por defecto
        steer, throttle = 0.0, 0.25

        if cx is not None:
            img_cx = w // 2
            error = -100.0 * (img_cx - cx) / img_cx
            # PID (PD en este caso)
          
            derivative = error - last_error_steer
            steer = np.clip(Kp_steer * error + Kd_steer * derivative, -1.0, 1.0)
            last_error_steer = error

            throttle = np.clip(0.8 - Kp_throttle * abs(error), 0.2, 0.6)

        vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer)))

        # overlay sencillo para ver la línea
        vis = rgb_pid.copy()
        if cx is not None:
            cv2.line(vis, (0, y), (w-1, y), (100,100,100), 1)
            cv2.line(vis, (w//2, 0), (w//2, h), (128,128,128), 1)
            cv2.circle(vis, (cx, y), 4, (255,0,0), -1)

        third = rgb_third_buf[0]
        draw_with_pip(third, vis)

    else:
        # ======== Inferencia con la cámara de 90° ========
        rgb_net = rgb_net_buf[0]
        if rgb_net is None:
            continue

        
        pil_img = Image.fromarray(rgb_net)     # asegúrate de que rgb es realmente RGB
        x = infer_tf(pil_img).unsqueeze(0)

        with torch.no_grad():
            out = model(x)
            steer, throttle = out[0].tolist()

        steer    = float(np.clip(steer,    -1.0, 1.0))
        throttle = float(np.clip(throttle,  0.0, 1.0))
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

        print("Throttle: ",throttle," Steer= ",steer)
 

        # PiP con la cámara en tercera persona (arriba a la derecha)
        third = rgb_third_buf[0]
        if third is not None:
            draw_with_pip(third, rgb_net)


# limpieza
try: cam_pid.stop(); cam_pid.destroy()
except: pass
try: cam_net.stop(); cam_net.destroy()
except: pass
try: cam_third.stop(); cam_third.destroy()
except: pass

try: vehicle.destroy()
except: pass
pygame.quit()
