# ------------------------------------------------
# Inferencia automática NN (FOV 90)
# ------------------------------------------------
import carla, time, pygame, numpy as np, cv2, torch, queue
from queue import Queue
from torchvision import transforms
from utils.pilotnet import PilotNet
from PIL import Image


# estado para velocidad por diferencias 
prev_pos = None          # np.array([x,y,z]) del paso anterior
ema_v    = None          # filtro EMA para suavizar
ALPHA_V  = 0.2           # 0..1, más alto = menos suave


inithelp = True
prev_timeglobal_var = 0.0
last_error_steer = 0.0
Kp_steer, Kd_steer = 0.1, 1e-5
Kp_throttle = 0.02

MODEL_PATH = "experiments/exp_debug_1762700241/trained_models/pilot_net_model_best_123.pth"
image_shape = (66, 200, 4)
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
WARMUP_SEC = 2.0  # tiempo antes de pasar a la red

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DeepRacer")

client = carla.Client(HOST, PORT)
client.set_timeout(5.0)
world = client.get_world()

# settings = world.get_settings()
# settings.synchronous_mode = True
# settings.fixed_delta_seconds = FIXED_DT
#world.apply_settings(settings)


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
#     carla.Rotation(yaw=-90)
# )


#-------------------------TRACK03---------------------------------
# spawn_point = carla.Transform(
#     carla.Location(x=-8, y=-15, z=0.5),
#     carla.Rotation(yaw=-15)
# )

#-------------------------TRACK02---------------------------------
# spawn_point = carla.Transform(
#    carla.Location(x=17, y=-4.5, z=0.5),
#    carla.Rotation(yaw=-10)
# )

#-------------------------TRACK04---------------------------------
# spawn_point = carla.Transform(
#     carla.Location(x=-10, y=21.2, z=1),
#     carla.Rotation(yaw=-15)
# )

#-------------------------TRACK05---------------------------------
spawn_point = carla.Transform(
   carla.Location(x=-3.7, y=-4, z=0.5),
   carla.Rotation(yaw=-120)
)

#-------------------TRACK06-gillesvilleneuve----------------------
# spawn_point = carla.Transform(
#    carla.Location(carla.Location(x=-1.5, y=33.3, z=0.5)),
#    carla.Rotation(yaw=180)
# )

#interlagosautodromojosecarlospace
#spawn_point = carla.Transform(carla.Location(x=-1.5, y=71.5, z=0.5), carla.Rotation(yaw=180))

#nurburgring
#spawn_point = carla.Transform(carla.Location(x=-65, y=17.5, z=0.5), carla.Rotation(yaw=150))

#spafrancorchamps
#spawn_point = carla.Transform(carla.Location(x=-65, y=94.5, z=0.5), carla.Rotation(yaw=-90))

#silverstone
#spawn_point = carla.Transform(carla.Location(x=-67, y=228, z=0.5), carla.Rotation(yaw=-25))

#lagoseco
#spawn_point = carla.Transform(carla.Location(x=-67, y=318, z=0.5), carla.Rotation(yaw=180-25))



vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if not vehicle:
    print("Error al spawnear el vehículo"); raise SystemExit
print("Vehículo spawneado")

# --- cámaras ---

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
cam_net = world.spawn_actor(cam_bp_net, cam_tf, attach_to=vehicle)

cam_tf_third = carla.Transform(
    carla.Location(x=-1.0, y=0.0, z=0.7),
    carla.Rotation(pitch=-12, yaw=0)  
)
cam_thirdperson = world.spawn_actor(cam_bp_thirdperson, cam_tf_third, attach_to=vehicle)

# buffers de frames
rgb_net_q   = Queue(maxsize=1)    # FOV 90 para la red
rgb_third_q = Queue(maxsize=1)    # Tercera persona

def _safe_put(q: Queue, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        try: q.get_nowait()
        except queue.Empty: pass
        q.put_nowait(item)

def cb_third(image: carla.Image):
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    _safe_put(rgb_third_q, bgra[:, :, :3][:, :, ::-1])

def cb_net(image: carla.Image):

    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    _safe_put(rgb_net_q, bgra[:, :, :3][:, :, ::-1])

cam_thirdperson.listen(cb_third)
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

clock = pygame.time.Clock()

while running:

    clock.tick(30) 
    
    snap = world.get_snapshot()
    if snap is None:
        continue
    sim_t = snap.timestamp.elapsed_seconds - start_sim
    dt    = snap.timestamp.delta_seconds
    

    # cambio por tiempo 
    if inithelp and sim_t >= WARMUP_SEC:
        inithelp = False


    # ESC para salir, SPACE para alternar manualmente
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE: running = False
            if e.key == pygame.K_SPACE:  inithelp = not inithelp

    # Coge el último frame disponible (sin bloquear)
    try: last_net   = rgb_net_q.get_nowait()
    except queue.Empty: pass
    try: last_third = rgb_third_q.get_nowait()
    except queue.Empty: pass


    # ======== Inferencia con la cámara de 90° ========
    rgb_net = last_net

    if rgb_net is None:
        continue

    loc = vehicle.get_location()              # carla.Location
    cur_pos = np.array([loc.x, loc.y, loc.z], dtype=float)

    if prev_pos is not None and dt and dt > 0:
                
        v = vehicle.get_velocity()
        speed = float(np.sqrt(v.x**2 + v.y**2 + v.z**2))
        print(speed)
    
    else:
        speed = 0.0
    
    prev_pos = cur_pos

    if inithelp:

        steer = 0.0
        throttle = 0.6

    else:
        # Calcula velocidad (m/s) y escálala igual que en train (÷5.0 y clip 0..1)
        speed_norm = float(np.clip(speed / 3.5, 0.0, 1.0))  # [0,1]

        # x: (1, 3, 66, 200) imagen 3 canales
        x = infer_tf(Image.fromarray(rgb_net)).unsqueeze(0)

        # === Crear el canal de velocidad (1, 1, 66, 200) ===
        speed_plane = torch.full((1, 1, 66, 200), speed_norm, dtype=x.dtype, device=x.device)
        # concatena
        x4 = torch.cat([x, speed_plane], dim=1)

        with torch.no_grad():

            # pasar SOLO un tensor al modelo (este ya reduce 4→3 con el adapter 1×1)
            out = model(x4)  
            steer, throttle = out[0].tolist()

        steer    = float(np.clip(steer,    -1.0, 1.0))

        throttle = float(np.clip(throttle,  0.0, 1.0))

    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

    #print("Throttle: ",throttle," Steer= ",steer)

    # PiP con la cámara en tercera persona (arriba a la derecha)
    draw_with_pip(last_third, rgb_net)


# limpieza
try: cam_net.stop(); cam_net.destroy()
except: pass
try: cam_third.stop(); cam_third.destroy()
except: pass

try: vehicle.destroy()
except: pass
pygame.quit()
