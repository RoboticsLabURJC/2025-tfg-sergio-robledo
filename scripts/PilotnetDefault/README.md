Directorio **PilotNetDefault** donde se encuentran todo lo necesario para el entrenamiento basico con PilotNet, en el que solo utiliza la imagen como canal de entrada.

- En /experiments se guardan los experimentos realizados
- En /utils los ficheros necesarios para la red, ya sea la manera de generar el dataset, como la arquitectura de PilotNet.

- run_training.sh ->  para ejecutar el entrenamiento (contiene los argumentos que se pasan al codigo del entenamiento)
```
./run_training.sh
```
- train_final -> el codigo que realmente representa el entrenamiento entero, con su train, validacion y testing. Se ejecuta llamandolo desde el script .sh anterior.

- run_carla_autopilot_rgb -> codigo que prueba en inferencia un modelo. La perspectiva es en tercera persona estilo videojuego.
```
python3 run_carla_autopilot_rgb.py
```

- videocam -> codigo que prueba en inferencia un modelo. Alterna entre modo mapa de calor dinamico y una estela roja que sigue al vehiculo. Se le pasa como argumento el numero de circuito.
```
python3 videcam.py 4
```