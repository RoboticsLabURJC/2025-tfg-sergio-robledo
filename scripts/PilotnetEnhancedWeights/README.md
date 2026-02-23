Directorio **PilotNetEnhancedWeights** donde se encuentran todo lo necesario para el entrenamiento con PilotNet, en el que se utiliza la imagen y la velocidad como canales de entrada y se aplican pesos y otras nuevas estrategias como batch mixing para los entrenamientos.

- En /experiments se guardan los experimentos realizados
- En /utils los ficheros necesarios para la red, ya sea la manera de generar el dataset, como la arquitectura de PilotNet.

- run_weighted_training_estados.sh ->  para ejecutar el entrenamiento (contiene los argumentos que se pasan al codigo del entenamiento)
```
./run_weighted_training_estados.sh
```
- train_weight_estados -> el codigo que realmente representa el entrenamiento entero, con su train, validacion y testing. Se ejecuta llamandolo desde el script .sh anterior.

- train_weights_estados_scheduler -> como el anterior pero scheduler añadido y opcion de WeightedRandomSampler.

- train_weights_estados_scheduler_batch_mix -> como el anterior pero se hace directamente el batch mixing y se balancea la loss.