---
title: "Teleoperador"
last_modified_at: 2024-12-14T12:42:00
categories:
  - Blog
tags:
  - Carla
  - Pygame
  - Camera
  - Anaconda
---

Durante las primeras semanas, nuestro objetivo principal ha sido adentrarnos en el simulador CARLA y desarrollar un teleoperador sencillo para controlar un vehículo.

## Índice
- [Anaconda](#anaconda)
- [CARLA](#carla)
- [Interfaz](#interfaz)
- [Manejo de sensores](#manejo-de-sensores)
- [Control](#control)
- [Demo](#demo)

## Anaconda

Para trabajar, crearemos un entorno Anaconda que nos permita utilizar la versión deseada de Python, en este caso, utilizaremos la versión 3.10, ya que es la que necesitaremos para usar la red neuronal de segmentación en futuros apartados.

```bash
conda create -n tfg python=3.10
conda activate tfg
pip install pygame numpy==1.26.4 carla Pillow 
```

## CARLA

Para iniciar el simulador CARLA, usamos el siguiente comando:
```bash
/opt/carla/CarlaUE4.sh -world-port=2000 -RenderOffScreen
```

Hemos estado investigando cómo realizar acciones básicas en CARLA: la apertura de distintos entornos, el desplazamiento del observador y la colocación de uno o varios vehículos con la opción de seleccionar su modelo. Después, nos centramos en definir el vehículo que queríamos controlar, conocido como ***Ego Vehicle*** en CARLA, al que añadiremos los sensores. Para esta funcionalidad hemos integrado dos cámaras: una para simular la perspectiva del conductor y otra para visualizar el vehículo en su entorno.
```python
def setup_carla(port:int=2000, name_world:str='Town01', delta_seconds=0.05, client:carla.Client=None)
def add_one_vehicle(world:carla.World, ego_vehicle:bool=False, vehicle_type:str=None, 
                    transform:carla.Transform=None)
def add_vehicles_randomly(world:carla.World, number:int) # Spawn Points
```

En el modo asíncrono de CARLA, el servidor se ejecuta a máxima velocidad, mientras que en el modo síncrono, el cliente indica al servidor cuándo ejecutar (*world.tick()*) y durante cuanto tiempo **simulado** debe hacerlo (*fixed delta seconds*), luego detiene la simulación hasta un nuevo *tick*. El modo síncrono se emplea comúnmente en entrenamientos de modelos, permitiendo detener la simulación durante el procesamiento. Sin embargo, la inferencia se realiza en modo asíncrono, replicando así condiciones más cercanas a la realidad.

## Interfaz

Para la Interacción Humano-Robot (HRI) hemos utilizado la biblioteca ***Pygame***, creando una pantalla que nos permite visualizar el contenido de ambas cámaras de manera simultánea. Para asegurar el correcto funcionamiento de esta biblioteca, es necesario configurar la siguiente variable de entorno:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

También hemos incluido un par de botones para aumentar o disminuir el *throttle*, es decir, la cantidad de acelerador que se aplica en el coche.
<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/teleoperator/interface.png" alt="">
</figure>

## Manejo de sensores

Hemos creado la clase ***Vehicle_sensors***, la cual nos permite almacenar el vehículo, en nuestro caso *ego vehicle*, y una lista de sus sensores.
```python
class Vehicle_sensors:
    def __init__(self, vehicle:carla.Vehicle, world:carla.World, screen:pygame.Surface)

    def add_sensor(self, sensor_type:str, transform:carla.Transform=carla.Transform())    
    def add_camera_rgb(self, size_rect:tuple[int, int], init:tuple[int, int]=None, 
                       transform:carla.Transform=carla.Transform())

    def destroy(self)
```

Cada sensor pertenece a la clase abstracta ***Sensor***, la cual guarda la instancia del sensor en CARLA y contiene el *callback* que almacena la última medida recogida del entorno en un atributo de la clase. La función ***process_data()*** debe ser implementada en cada subclase de acuerdo al tipo de sensor, permitiéndonos actualizar su información y mostrarla en la pantalla si es necesario.
```python
class Sensor(ABC):
    def __init__(self, sensor):
        self.sensor = sensor
        self.data = None
        self.sensor.listen(lambda data: self.__callback_data(data))

    def __callback_data(self, data):
        self.data = data

    def get_last_data(self):
        return self.data

    @abstractmethod
    def process_data(self):
        pass

class Vehicle_sensors:
    def update_data(self, flip:bool=True):
        for sensor in self.sensors:
            sensor.process_data()
```

Para el manejo de la cámara, hemos desarrollado una clase ***Camera*** que hereda de *Sensor*, la cual incorpora nuevos parámetros en el constructor y sobrescribe la función *process_data()*, que simplemente muestra la imagen capturada. Además, hemos añadido una nueva función ***add_camera_rgb*** en la clase *Vehicle_sensors*, que requiere los parámetros del constructor de esta nueva clase y la ubicación de la cámara respecto al coche. Además, hemos añadido un sensor de colisión para detectar si el coche choca contra algún objeto, en caso de impacto, se eleva una excepción.
```python
class Camera(Sensor):      
    def __init__(self, size:tuple[int, int], init:tuple[int, int], sensor:carla.Sensor, screen:pygame.Surface, text:str=None)
    def process_data(self)

class Collision(Sensor):
    def process_data():
        if self.data != None:
            assert False # El vehículo ha chocado

class Vehicle_sensors:
    def add_camera_rgb(self, size_rect:tuple[int, int]=None, init:tuple[int, int]=None,
                       text:str=None, transform:carla.Transform=carla.Transform())
    def add_collision(self)
```

## Control 

El teleoperador también ha sido desarrollado utilizando *Pygame*. En función de la tecla presionada, el vehículo recibe el correspondiente comando de control: la *w* se utiliza para avanzar, las teclas *a* y *d* para girar el volante, y la *s* para frenar.

Para implementar este modo de funcionamiento, hemos creado la clase ***Teleoperator***.
```python
class Teleoperator:
    def __init__(self, vehicle:carla.Vehicle, steer:float=0.3, throttle:float=0.6, brake:float=1.0)
    def control(self)

    def set_steer(self, steer:float)
    def set_throttle(self, throttle:float)
    def set_brake(self, brake:float)

    def get_steer(self)
    def get_throttle(self)
    def get_brake(self)
```

## Demo

<iframe width="560" height="315" src="https://www.youtube.com/embed/4BGPVyMgDx4?si=Lp-pghJCV8Xc17W7" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
