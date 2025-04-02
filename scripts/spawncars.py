import carla
import time

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Elegir blueprint de vehículo
    vehicle_bp1 = blueprint_library.filter('vehicle.finaldeepracer.aws_deepracer')[0]

    # Coordenadas para los dos vehículos
    spawn_point_1 = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.5), carla.Rotation())

    # Spawn de vehículos
    vehicle1 = world.spawn_actor(vehicle_bp1, spawn_point_1)

    print(f"Vehículo 1 ID: {vehicle1.id} en {spawn_point_1.location}")

    # Esperar 10 segundos para observar los vehículos en el mapa
    time.sleep(10)

    # Eliminar actores
    vehicle1.destroy()
    print("Vehículos destruidos.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrumpido por el usuario.")
