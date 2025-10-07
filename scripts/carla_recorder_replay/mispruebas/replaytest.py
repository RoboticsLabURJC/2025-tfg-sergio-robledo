#!/usr/bin/env python3
import re, sys, time
import carla, pygame, numpy as np

LOG_FILE = "/tmp/Track01CC.log"
WIDTH, HEIGHT = 800, 600
FPS = 30.0
FIXED_DT = 1.0 / FPS

def get_log_duration(client, path: str) -> float:
    info = client.show_recorder_file_info(path, False)
    m = re.search(r"Duration:\s*([0-9.]+)", info)
    if not m:
        raise RuntimeError("No se pudo leer la duración del log")
    return float(m.group(1))

def wait_for_ego(world, filt="vehicle.finaldeepracer.aws_deepracer", timeout_s=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        world.wait_for_tick()  # <- mejor que tick() para esperar a que el replay avance
        actors = world.get_actors().filter(filt)
        if actors:
            return actors[0]
    # fallback
    actors = world.get_actors().filter("vehicle.*")
    return actors[0] if actors else None

def wait_for_vehicle_debug(world, filter_str="vehicle.finaldeepracer.aws_deepracer", timeout_s=10.0):
    t0 = time.time()
    printed_header = False
    while time.time() - t0 < timeout_s:
        # deja avanzar el replay; no fuerces synchronous aquí
        snap = world.wait_for_tick(2.0)
        if snap is None:
            continue

        # 1) prueba con el filtro exacto
        actors = world.get_actors().filter(filter_str)
        if actors:
            return actors[0]

        # 2) si no aparece, lista qué vehículos hay para depurar
        if not printed_header:
            print("Esperando al ego... listando vehículos que existen:")
            printed_header = True

        all_veh = world.get_actors().filter("vehicle.*")
        print(f"  - {len(all_veh)} vehículos ahora mismo")
        for a in all_veh[:10]:  # muestra hasta 10 para no saturar
            rn = a.attributes.get("role_name", "")
            print(f"    id={a.id}  type={a.type_id}  role_name={rn}")

        # prueba también un filtro más laxo por si el blueprint cambia levemente
        dp = world.get_actors().filter("vehicle.*deepracer*")
        if dp:
            return dp[0]

    return None


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CARLA Replay")

    client = carla.Client('127.0.0.1', 3333)
    client.set_timeout(10.0)

    # Info del log (útil para ver el id del ego: normalmente 1)
    print(client.show_recorder_file_info(LOG_FILE, False))

    # (Opcional) NO fuerces synchronous durante replay; el recorder manda.
    world = client.get_world()
    # Si insistes en síncrono:
    # settings = world.get_settings()
    # settings.synchronous_mode = True
    # settings.fixed_delta_seconds = FIXED_DT
    # world.apply_settings(settings)

    # 1) Lanza el replay (sigue al actor 1 si ese es tu ego)
    client.replay_file(LOG_FILE, 0.0, 0.0, 21)

    # 2) Re-obtener el world por si se recargó con el mapa del log
    world = client.get_world()

    # 3) Duración del log (¡ahora sí la definimos!)
    duration = get_log_duration(client, LOG_FILE)

    ego = wait_for_vehicle_debug(world, "vehicle.finaldeepracer.aws_deepracer", timeout_s=15.0)
    if ego is None:
        print("No se encontró ningún vehículo en el replay tras el tiempo de espera.")
        return
    print(f"Encontrado ego id={ego.id}, type={ego.type_id}, role_name={ego.attributes.get('role_name','')}")


    # Cámara propia
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(WIDTH))
    bp.set_attribute("image_size_y", str(HEIGHT))
    bp.set_attribute("fov", "90")
    cam_tf = carla.Transform(carla.Location(x=-1, z=0.5))
    camera = world.spawn_actor(bp, cam_tf, attach_to=ego)

    img_surface = [None]
    def on_image(image: carla.Image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, :3]
        arr = arr[:, :, ::-1]
        img_surface[0] = pygame.surfarray.make_surface(arr.swapaxes(0, 1))

    camera.listen(on_image)

    # Para cortar al final del log
    start_sim_time = world.get_snapshot().timestamp.elapsed_seconds
    end_sim_time = start_sim_time + duration
    clock = pygame.time.Clock()

    try:
        while True:
            # Si tienes synchronous activado, usa world.tick()
            # Si NO, deja que el replay avance y solo pinta:
            # world.wait_for_tick() también vale para “sin sync”
            snap = world.wait_for_tick()
            sim_t = snap.timestamp.elapsed_seconds

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            if img_surface[0] is not None:
                screen.blit(img_surface[0], (0, 0))
                pygame.display.flip()

            if sim_t >= end_sim_time:
                print("✔️ Replay finalizado")
                break

            clock.tick(FPS)
    except KeyboardInterrupt:
        print("Interrumpido por el usuario")
    finally:
        try:
            camera.stop(); camera.destroy()
        except:
            pass
        pygame.quit()

if __name__ == "__main__":
    main()
