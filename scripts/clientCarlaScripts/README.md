Directorio **clientCarlaScript** donde se encuentran todos los codigos que de forma directa interactuan con el servidor de carla, ademas de los codigos que moldean caracteristicas de los datasets.

Se parte de un dataset  con una nomenclatura y estructura similares a esta:

```
/datasets/
├── Deepracer_BaseMap_12C
│   
├── Deepracer_BaseMap_12CC
│   
├── Deepracer_BaseMap_12CCv2
│   
├── Deepracer_BaseMap_12Cv2
│   
├── Deepracer_BaseMap_13C
│   
├── Deepracer_BaseMap_13CC
│   ├── 
├── Deepracer_BaseMap_13CCv2
│   ├── 
├── Deepracer_BaseMap_13Cv21
│   ├── 
├── Deepracer_BaseMap_3CC
│   ├──
├── Deepracer_BaseMap_3CCv2
│   ├──
├── Deepracer_BaseMap_4C
│   ├
├── Deepracer_BaseMap_4Cv2
│   ├
├── Deepracer_BaseMap_5C
│   ├──
├── Deepracer_BaseMap_5Cv2
│   ├
├── test
│   ├── Deepracer_BaseMap_14C
│   │  
│   ├── Deepracer_BaseMap_14CC
│   │ 
│   ├── Deepracer_BaseMap_14CCv2
│   │   
│   └── Deepracer_BaseMap_14Cv2
│       
└── validation
    ├── Deepracer_BaseMap_3C
    │   
    ├── Deepracer_BaseMap_3Cv2
    │  
    ├── Deepracer_BaseMap_4CC
    │ 
    ├── Deepracer_BaseMap_4CCv2
    │  
    ├── Deepracer_BaseMap_5CC
    │   
    └── Deepracer_BaseMap_5CCv2
        


```
Cada directorio hace referencia a cada toma de datos con circuitos nombrados entre el 3 y el 14. La nomenclatura C, CC y v2 hacen referencia a :
- C: Clockwise, el recorrido de ese circuito se realiza en el sentido de las agujas del reloj
- CC: CounterClockwise, el recorrido de ese circuito se realiza en el sentido contrario al de las agujas del reloj.
- v2: segunda toma de datos del circuito

Dentro de cada directorio hay una carpeta de imagenes rgb y otra con las correspondientes mascaras. Ademas de un csv llamado dataset.csv por directorio que recoge todos los datos y los paths a cada imagen.

Por orden alfabetico en cada bloque:

## Obtencion y balanceo de datos

- adjust_dataset_final ->  para balancear el dataset de tal manera que se busque el circuito con menores datos de un estado (1,2 o 3 equivale a izquierda centro y derecha) y establecer esa cantidad como umbral para recortar en el resto de datos de circuitos. De esta manera cada estado de cada circuito tiene la misma representacion.
```
python3 adjust_dataset_final.py --valdir ../datasets/validation/
```
- check_repeated_images -> comprobar que tras haber balanceado y tratado los datos, de la manera que sea, no haya datos repetidos ni imagenes repetidas que afecten al entrenamiento posterior.
```
python3 check_repeated_images.py --base-dir ../datasets/validation/
```
- datasetgenfromreplayandcsvfilelaterchange -> Genera un directorio como los mencionados previamente, ej: Deepracer_BaseMap_4C. Utiliza un fichero .log y un csv de velocidades para poder generar el dataset final.
```
python3 datasetgenfromreplayandcsvfilelaterchange.py
```

- datasetgenfromreplayandcsvfilelaterchangewithposition -> Igual pero añadiendo tambien la posicion x,y,z
```
python3 datasetgenfromreplayandcsvfilelaterchangewithposition.py
```

- delete_duplicates -> elimina cualquier tipo de dato que este repetido en cualquier dataset.
```
python3 delete_duplicates.py --base-dir ../datasets/
```

- delete_throttle_higher_than, delete_throttle_lower_than -> elimina todos los datos cuyo throttle sea mayor al de un umbral o menor.
```
python3 delete_throttle_higher_than.py
```

## Visualizar datos

- bin_viewer -> para visualizar por lotes la distribucion de los datos del dataset.

```
python3 bin_viewer.py 
```

- dataset_visualizecsv -> para visualizar un dataset entero, la velocidad, giro, imagen , mascara...

```
python3 dataset_visualizecsv.py --base_path ../datasets/Deepracer_BaseMap_14Cv2/

```
- frequency_histograms_absolute y frequency_histograms. > Grafican la distribucion de datos y su densidad. El absolute muestra la densidad de datos de train, validation y test sin normalizar. Sin embargo el otro codigo normaliza entre 0 y 1 por separado cada uno de los tres conjuntos de datos

```
python3 frequency_histograms_absolute.py
```

- histograms.py -> Hace un grafico de barras de los estados (1,2,3)
```
python3 histograms.py --pattern "../datasets/validation/Deepracer_BaseMap_*/dataset.csv"
```


## Modificar y visualizar mascaras

- turn_black_masks -> Edita los 100 pixeles superiores de todas las imagenes y los vuelve completamente negros para eliminar de esta manera el blanco de los muros.
```
python3 turn_black_masks.py --base-dir ../datasets/
```
- turn_black_top200_and_square_masks -> hace lo mismo que turn_black_masks pero añade otros 200 pixeles negros en la parte superior para que la imagen tenga el mismo largo que ancho. (800x800)

```
python3 turn_black_top200_and_square_masks.py --base-dir ../datasets/
```
- visualize_masks -> visualiza en orden las mascaras de un directorio
```
python3 visualize_masks.py --base-dir ../datasets/Deepracer_BaseMap_12Cv2/
```




## Funcionamiento del vehiculo

- clear_vehicles -> para eliminar todos los vehiculos spawneados en el mundo.

```
python3 clear_vehicles.py 
```

- manualcontrol, manualcontrolspinningcam -> para controlar el vehiculo con las teclas WASD, la version spinningcam hace que la camara gire alrededor del coche.

```
python3 manualcontrol.py 
```

- pdcontroller, pdcontroller30fps -> controlador PID para manetener el vehiculo en la linea central

```
python3 pdcontroller.py 
```

## Control desde mandos, Ps4 y Switch pro

- datasetgenNintendoController, datasetgenPS4Controllerjoysticks y datasetgenPS4ControllerR2-> generan datos pero utilizando los mandos como el metodo de conduccion del vehiculo.

```
python3 datasetgenNintendoController.py 
```

- joystick_client_nintendo, joystick_client_ps4_joysticks y joystick_client_ps4R2-> codigos que conectan con los respectivos mandos para mandar los datos a los codigos de datasetgen*.

```
python3 joystick_client_nintendo.py 
```

- manualcontrolNintendoController, manualcontrolPS4Controller -> para controlar el vehiculo con los mandos.

```
python3 manualcontrolPS4Controller.py 
```


## Otros

- testtime -> Codigo para la comprobacion de ls diferencia entre “tiempo simulado” y “tiempo de reloj real”
```
python3 testtime.py 
```