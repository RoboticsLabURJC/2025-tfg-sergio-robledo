Directorio **PilotNetEnhanced** donde se encuentran todo lo necesario para el entrenamiento con PilotNet, en el que se utiliza la imagen y la velocidad como canales de entrada.

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

- fancyvideocam -> codigo que prueba en inferencia un modelo. Alterna entre modo mapa de calor dinamico (heatmap) y una estela roja (trail) que sigue al vehiculo. Se le pasa como argumento el numero de circuito, para saber que camara emplear.
```
python3 fancyvideocam.py --mode heatmap --cam 3
```

- inference_img -> codigo que prueba en inferencia un modelo sobre una imagen, en concreto la mascara, se le pasa la imagen y velocidad como entrada  y devuelve throttle y steer.
```
python3 inference_img.py --img ../../imagen2.png --model experiments/exp_debug_1769708013/trained_models/pilot_net_model_best_123.pth --speed 0.888
```

- log_gen_from_inference -> codigo que genera en inferencia un dataset y un fichero .log. Empleado para despues compararse con el piloto humano o con otros modelos. Se le pasa como argumento el numero de circuito, para saber que camara emplear. 
```
python3 log_gen_from_inference.py --cam 5
```

- logs_compare_numerical, logs_compare_numerical_speed, logs_compare_numerical_speed_by_states -> codigo que compara a nivel cuantitativo los recorridos de dos csv que se pasan como argumentos. En este caso para diferenciar el tomado por el piloto y el generado en inferencia. El codigo logs_compare_numerical compara posiciones, logs_compare_numerical_speed las velocidades por posicion y logs_compare_numerical_speed_by_states aporta informacion de error por estados.
```
python3 logs_compare_numerical.py --ref logs/Deepracer_BaseMap_5CCv1/dataset.csv --inf logs/infer_log_5CC.csv --plot
```
- logs_compare_visual: comparacion de manera visual, es una "carrera" entre los recorridos de dos csv que se pasan como argumentos
```
python3 logs_compare_visual.py --csv_human logs/Deepracer_BaseMap_5CCv1/dataset.csv --csv_inf logs/infer_log_5CC.csv --cam 5
```

- model_eval -> codigo que prueba en inferencia un modelo. Alterna entre modo mapa de velocidad (heatmap) y una estela roja (trail) que sigue al vehiculo. Se le pasa como argumento el numero de circuito, para saber que camara emplear.
```
python3 model_eval.py --mode heatmap --cam 5
```