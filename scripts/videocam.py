import carla, time, pygame, numpy as np, cv2, torch
from torchvision import transforms
from utils.pilotnet import PilotNet
from PIL import Image
import sys

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

VEHICLE_MODEL = 'vehicle.finaldeepracer.aws_deepracer'
WIDTH, HEIGHT = 800, 600
FPS = 30.0
FIXED_DT = 1.0 / FPS
prev_timeglobal_var = 0.0


def main():
  
    cam_index = 1
    if len(sys.argv) > 1:
        try:
            cam_index = int(sys.argv[1])
            if cam_index < -1 or cam_index > 8:
                print("Índice fuera de rango. Usando 1 por defecto.")
                cam_index = 1
        except ValueError:
            print("Argumento inválido. Usando 1 por defecto.")

   
    cam_locations = {
        1: carla.Location(x=1.5, y=-2.5, z=2),
        2: carla.Location(x=17, y=-3, z=2.8),
        3: carla.Location(x=-12, y=-16.5, z=6),
        4: carla.Location(x=-9.5, y=13.5, z=10.5),
        5: carla.Location(x=-8.9, y=-3.9, z=4),
    }
    
    
    client = carla.Client('localhost', 2000)
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

    if cam_index == 1:

        # -------------------------TRACK01-----------------------------
        spawn_point = carla.Transform(
            carla.Location(x=3, y=-1, z=0.5),
            carla.Rotation(yaw=-90)
        )

    if cam_index == 5:
        #-------------------------TRACK---------------------------------
        spawn_point = carla.Transform(
        carla.Location(x=-3.7, y=-4, z=0.5),
        carla.Rotation(yaw=-120)
        )

    if cam_index == 3:
        #-------------------------TRACK03---------------------------------
        spawn_point = carla.Transform(
            carla.Location(x=-8.5, y=-14.7, z=0.5),
            carla.Rotation(yaw=-15)
        )
    if cam_index == 2:
        #-------------------------TRACK02---------------------------------
        spawn_point = carla.Transform(
           carla.Location(x=17, y=-4.8, z=0.5),
           carla.Rotation(yaw=-10)
        )

    if cam_index == 4:
        #-------------------------TRACK04---------------------------------
        spawn_point = carla.Transform(
            carla.Location(x=-10, y=21.2, z=1),
            carla.Rotation(yaw=-15)
        )

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
    if  cam_index == 1:
        cam_rotation = carla.Rotation(pitch=-90)
    else:
        cam_rotation = carla.Rotation(pitch=-90,yaw=-90)

    cam_transform = carla.Transform(cam_location, cam_rotation)
    camera = world.spawn_actor(camera_bp, cam_transform)

    cam_net_tf = carla.Transform(carla.Location(x=0.13, z=0.13), carla.Rotation(pitch=-30))
    cam_net = world.spawn_actor(cam_bp_net, cam_net_tf, attach_to=vehicle)

    rgb_net_buf = [None]  # FOV 90 para la red

    camera_image = {"data": None}

    def draw_with_pip(main_rgb):
        # main (tercera persona) a tamaño completo
        if main_rgb is not None:
            main_surf = pygame.surfarray.make_surface(main_rgb.swapaxes(0, 1))
            screen.blit(main_surf, (0, 0))

        pygame.display.flip()


    def cb_net(image: carla.Image):
        global prev_timeglobal_var

        timeglobal_var = image.timestamp 
        fps_toprint = timeglobal_var - prev_timeglobal_var 
        #print(1/fps_toprint) 
        prev_timeglobal_var = timeglobal_var

        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        bgr  = bgra[:, :, :3]
        rgb_net_buf[0] = bgr[:, :, ::-1]
        

    def on_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
        camera_image["data"] = array

    camera.listen(on_image)
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


    while running:
        world.tick()
        sim_t = world.get_snapshot().timestamp.elapsed_seconds

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
        loc = vehicle.get_location()                 # carla.Location
        print(loc.x, loc.y, loc.z) 
        #print("Throttle: ",throttle," Steer= ",steer)


        if camera_image["data"] is not None:
            cv2.imshow("Vista Cenital", camera_image["data"])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Finalizando...")
    camera.stop()
    camera.destroy()
    try: cam_net.stop(); cam_net.destroy()
    except: pass

    try: vehicle.destroy()
    except: pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
