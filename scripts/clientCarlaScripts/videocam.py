import carla
import numpy as np
import cv2
import sys


# #..........................................
# Codigo para spawnear con parametros una camara en distintos sitios y 
# visualizar el deepracer desde distintos angulos
# Con argumentos de -1 a 7, cada uno para un spawnpoint distinto
# #..........................................

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
        5: carla.Location(x=-8.9, y=-3.9, z=4),
    }
    
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
   
 

    weather = carla.WeatherParameters(cloudiness=80.0, sun_altitude_angle=90.0)
    world.set_weather(weather)


    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1660')
    camera_bp.set_attribute('image_size_y', '1000')
    camera_bp.set_attribute('fov', '120')

    cam_location = cam_locations[cam_index]
    if cam_index == 5 or cam_index == 2 or cam_index == 3:
        cam_rotation = carla.Rotation(pitch=-90,yaw=-90)
    else:
        cam_rotation = carla.Rotation(pitch=-90)
    
    cam_transform = carla.Transform(cam_location, cam_rotation)

    camera = world.spawn_actor(camera_bp, cam_transform)

    camera_image = {"data": None}

    def on_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
        camera_image["data"] = array

    camera.listen(on_image)

    print(f"Cámara {cam_index} activa en {cam_location}. Presiona 'q' para salir.")

    try:
        while True:
            if camera_image["data"] is not None:
                cv2.imshow("Vista Cenital", camera_image["data"])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Finalizando...")
        camera.stop()
        camera.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
