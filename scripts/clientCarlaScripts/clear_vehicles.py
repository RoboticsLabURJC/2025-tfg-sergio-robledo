#------------------------------------------------
#Codigo que elimina todos los vehiculos correctamente
#spawneados en el mundo de carla
#------------------------------------------------

import carla

def clear_all_vehicles():
    try:
        # Conexion simulador CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Obtencion del mundo
        world = client.get_world()

        # Obtencion de actores y de ellos obtener vechiculos
        actors = world.get_actors()
        vehicles = actors.filter('vehicle.*')

        if not vehicles:
            print("No hay vehículos en el mapa.")
            return

        # Destruir cada vehículo
        for vehicle in vehicles:
            vehicle.destroy()
            print(f"Vehículo eliminado: {vehicle.type_id} (ID: {vehicle.id})")

        print(f"Se eliminaron {len(vehicles)} vehículos del mapa.")

    except Exception as e:
        print(f"Error al eliminar vehículos: {e}")

if __name__ == "__main__":
    clear_all_vehicles()
