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
            if cam_index < -1 or cam_index > 7:
                print("Índice fuera de rango. Usando 1 por defecto.")
                cam_index = 1
        except ValueError:
            print("Argumento inválido. Usando 1 por defecto.")

   
    cam_locations = {
        1: carla.Location(x=-10, y=-21.2, z=30),
        2: carla.Location(x=8, y=-3, z=6),
        3: carla.Location(x=17, y=-3, z=5),
        4: carla.Location(x=16, y=-12, z=6),
        5: carla.Location(x=4, y=-12, z=6),
        6: carla.Location(x=4, y=-19, z=7),
        7: carla.Location(x=1.6, y=-0.8, z=0.1),
        0: carla.Location(x=2.6, y=0.8, z=1.4),
        -1: carla.Location(x=3, y=-1, z=1),
    }
    
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
   
 

    weather = carla.WeatherParameters(cloudiness=80.0, sun_altitude_angle=90.0)
    world.set_weather(weather)


    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1660')
    camera_bp.set_attribute('image_size_y', '1280')
    camera_bp.set_attribute('fov', '100')

    cam_location = cam_locations[cam_index]

    if cam_index == 3 or cam_index == 4 or cam_index == 5:
        cam_rotation = carla.Rotation(pitch=-90, yaw=90)
    else:
        cam_rotation = carla.Rotation(pitch=-90)

    if cam_index == 0:
        cam_rotation = carla.Rotation(pitch=-50,yaw=-110)

    if cam_index == 7:
        cam_rotation = carla.Rotation(pitch=0)


    cam_transform = carla.Transform(cam_location, cam_rotation)
    
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
