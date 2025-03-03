import carla
import random
import time

def spawn_vehicle():
    try:
        # Conéctate al simulador CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Obtén el mundo
        world = client.get_world()

        # Obtén el blueprint library para elegir un vehículo
        blueprint_library = world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter('vehicle.deepracer04.aws_deepracer04')

        if not vehicle_blueprints:
            print("No se encontraron vehículos en la biblioteca de CARLA.")
            return
        
        # Selecciona un vehículo aleatorio
        vehicle_bp = random.choice(vehicle_blueprints)
        
        # Definir la ubicación de spawn con la información de la imagen
        spawn_location = carla.Location(x=35.3, y=246.1, z=1.2)  # Se coloca un poco elevado (z=0.5) para evitar colisiones
        spawn_rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)  # Ajusta si es necesario
        spawn_transform = carla.Transform(spawn_location, spawn_rotation)

        # Spawnea el DeepRacer
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
        

        if vehicle:
            print(f"Vehículo {vehicle.type_id} spawneado en {spawn_transform.location}")
            
        else:
            print("No se pudo spawnear el vehículo.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    spawn_vehicle()

