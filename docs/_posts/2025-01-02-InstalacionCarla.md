---
title: "Instalación Carla"
last_modified_at: 2025-01-02T18:35:00
categories:
  - Blog
tags:
  - Intro
  - Carla
---

## Introducción a Carla

CARLA (Car Learning to Act) es un simulador *Open Source* diseñado para el desarrollo, entrenamiento y validación de algoritmos de conducción autónoma. Fue creado por el Computer Vision Center (CVC) de la Universidad Autónoma de Barcelona, en colaboración con empresas como Intel y Toyota Research.

La instalación del simulador puede ser muy compleja debido a cambios en ciertos paquetes en distintas versiones, además de la ambiguedad de ciertos pasos en la web oficial del simulador.


Aquí muestro como sería la instalación correcta de CARLA, concretamente la versión 0.9.15 para Unreal Engine 4.26.

## Instalación paso a paso

En este [enlace](https://carla.readthedocs.io/en/latest/build_linux/) cuentas con la web oficial y los pasos para la correcta instalación, sin embargo te dejo todos y cada uno de los pasos a seguir en este blog, sin necesidad de acceder a la web oficial.

### Requisitos "físicos"

Antes de nada, es muy importante saber que tenemos un ordenador capaz de soportar este software. Para ello se necesita:

1. Mínimo una versión de Ubuntu 18.04 (En mi caso utilizo la versión 22.04)

2. Alrededor de unos 130 GB de espacio, utilizados para la instalación de Unreal Engine y Carla, además de ciertos paquetes extra para el correcto funcionamiento de ambos.

3. Al menos una GPU de 6 GB.

4. Conexión a internet y los puertos TCP 2000 y 2001. Puede parecer un simple detalle, pero en diversos foros, muchas personas comentan errores que han tenido y no consiguen solucionar que tienen que ver con diversos firewalls que tienen configurados en estos puertos. Si nunca has manipulado nada relacionado con estos puertos, no deberías de tener ningún tipo de problema.

5. Tener cuenta en github y *Epic Games* (el proveedor de *Unreal Engine*).

### Requisitos de software

Para los primeros prerrequisitos, ejecuta el siguiente código en tu terminal:

```bash
sudo apt-get update &&
sudo apt-get install -y wget software-properties-common &&
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test &&
wget -O /usr/share/keyrings/llvm-keyring.gpg https://apt.llvm.org/llvm-snapshot.gpg.key

```

A continuación, para evitar errores de versiones entre *Unreal Engine* y Carla, ejecuta:

```bash
sudo apt-add-repository "deb http://archive.ubuntu.com/ubuntu focal main universe"
sudo apt-get update
sudo apt-get install build-essential clang-10 lld-10 g++-7 cmake ninja-build libvulkan1 python python3 python3-dev python3-pip libpng-dev libtiff5-dev libjpeg-dev tzdata sed curl unzip autoconf libtool rsync libxml2-dev git git-lfs
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/lib/llvm-10/bin/clang++ 180 &&
sudo update-alternatives --install /usr/bin/clang clang /usr/lib/llvm-10/bin/clang 180 &&
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 180
```
Si algún comando presenta errores, quizás alguna versión no esté disponible. De todas maneras, se puede continuar con la instalación y solucionar esto más adelante.


A continuación, se necesita una versión de pip3 superior a 20.3, compruébalo con el comando: 

```bash
pip3 -v
```
(sustituye pip3 por pip si no dispones de la versión de python 3)

En caso de querer mejorar a otra versión: 

```bash
pip3 install --upgrade pip
```

Además es necesario instalar las siguientes dependencias de Python:

```bash
pip install --user setuptools &&
pip3 install --user -Iv setuptools==47.3.1 &&
pip install --user distro &&
pip3 install --user distro &&
pip install --user wheel &&
pip3 install --user wheel auditwheel==4.0.0
```

Antes de la instalación de carla, es necesario instalar *Unreal Engine*:

### Unreal Engine

En primer lugar tenemos que clonar el repositorio especial "forkeado" de *Unreal Engine* que usa Carla.

```bash
git clone --depth 1 -b carla https://github.com/CarlaUnreal/UnrealEngine.git ~/UnrealEngine_4.26

```

Después accedemos al repositorio:

```bash
cd ~/UnrealEngine_4.26
```

Por último, tenemos que realzar el siguiente comando para construir todos los paquetes necesarios. Es **muy importante** tener en cuenta que este proceso puede llegar a tardar algo más de hora y media. 

```bash
  ./Setup.sh && ./GenerateProjectFiles.sh && make
```

Una vez ha finalizado, se puede comprobar que se ha instalado correctamente con el siguiente comando:

```bash
 cd ~/UnrealEngine_4.26/Engine/Binaries/Linux && ./UE4Editor
```

Se abrirá un pequeño menú que tiene diversas opciones para crear proyectos, ya sea de videojuegos o animación. Si aparece este menú, significa que se ha instalado correctamente y se puede cerrar.


### Carla

Para poder usar Carla, hay que vincular la cuenta de *Epic Games(Unreal Engine)* con la de github. En este [enlace](https://www.unrealengine.com/en-US/ue-on-github) se muestra más detallado. Es **muy importante** aceptar la invitación que llega al correo electrónico asociado a la cuenta de git, en caso contrario no dejará clonar el repositorio necesario.


Lo primero que debemos hacer, será descargarnos el repositorio oficial de carla:

```bash
git clone https://github.com/carla-simulator/carla
```
A continuación accedemos al repositorio desde el terminal y cambiamos a la rama *master*.

```bash
git checkout master
```
En ella se encuentra la última versión de Carla. En este momento, a día 1 de febrero de 2025 esta es la versión: 0.9.15. Se puede comprobar, accediendo al repositorio de git desde el navegador. Se cambia la rama a *master* (por defecto se muestra *ue5-dev*) y se observa la versión que se muestra en el *README.md*.

A continuación se tiene que ejecutar en el directorio clonado el siguiente script:

```bash
./Update.sh
```

Si se obtiene un error de este estilo:
```bash
gzip: stdin: not in gzip format
tar: Child returned status 1
tar: Error is not recoverable: exiting now
make: *** [Util/BuildTools/Linux.mk:142: setup] Error 2
```
Te explico [aquí](#error-de-formato-gzip-en-updatesh) como solucionarlo.

De esta manera se descargan los últimos archivos para trabajar con la versión actual de CARLA. En nuestro caso se descomprimen los de la última versión. Si se quisieran otros distintos, viene explicado en la [página de instalación de carla](https://carla.readthedocs.io/en/latest/build_linux/) la metodología a seguir.


Cuando se compila y se lanzan los paquetes de este repositorio, necesitan conocer la ubicación de los archivos de *Unreal Engine*, para ello hay que definir una variabe de entorno:

Abre en un editor *~/.bashrc* y escribe la siguiente línea debajo del todo:
```bash
export UE4_ROOT=~/UnrealEngine_4.26 
```
Acepta los cambios y reinicia el terminal.


Ahora es el momento de compilar carla y lanzarlo, para ello, primero compilamos el cliente:

```bash
make PythonAPI
```

------

Si tienes un error similar al siguiente:

```bash
    raise DistributionNotFound(req, requirers)
pkg_resources.DistributionNotFound: The 'typing>=3.7.4' distribution was not found and is required by typing-extensions
make: *** [Util/BuildTools/Linux.mk:92: PythonAPI] Error 1
```
Te explico como solucionarlo [aquí](#make-pythonapi-falta-del-paquete-typing374)

Si tienes un error similar al siguiente:

```bash
raise DistributionNotFound(req, requirers)
pkg_resources.DistributionNotFound: The 'importlib-resources>=1.3; python_version < "3.9"' distribution was not found and is required by the application
make: *** [Util/BuildTools/Linux.mk:92: PythonAPI] Error 1
```
Te explico como solucionarlo [aquí](#make-pythonapi-importlib-resources-para-python-310)


------

A continuación instalamos el fichero *.whl* con pip3. Este se encuentra en *PythonAPI/carla/dist/*. En mi caso:

```bash
pip3 install PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-linux_x86_64.whl

```
Este fichero es único para el SO.


A continuación se compila el servidor: 

```bash
make launch
```

Hay que tener en cuenta que este proceso es muy lento y requiere muchísimo tiempo, ya que tiene que cargar todas las mallas (*meshes*) y *shaders*. Es muy normal que el ordenador alguna que otra vez se quede bloqueado, deje de responder o incluso se congele. Ante cualquier error de este estilo, vuelve a realizar el comando todas las veces que necesites y ten mucha paciencia. Ten en cuenta que también puede que se muestre: *unreal engine dejó de funcionar*  con las opciones de *forzar salida* y esperar. Ante esta situación, solo queda esperar y no pulsar nada. 

Una vez se consiga lanzar el mundo por defecto (*Town10*), seguirá cargando algunos que otros shaders. Si consigues observar el *Unreal Editor* con una ciudad cargada, significa que has instalado Carla correctamente, ¡¡¡enhorabuena!!!.


## Posibles errores y soluciones

Aquí te dejo diversos errores que puedes encontrar a la hora de instalar el simulador, así como la manera de solucionarlos. Es **importante** tener en cuenta que ante cualquier error, ya sea alguno de los siguientes casos, o no, el simulador cuenta con un canal de *Discord*. Se puede acceder [aquí](https://discord.com/invite/8kqACuC). En él, encontrarás errores de otras personas y cualquier error que tengas, probablemente ya lo haya tenido alguien y esté solucionado. Si no es el caso, puedes dejar el mensaje y te ayudarán a solucionarlo.


### Error de formato gzip en Update.sh

Si se obtiene un error similar a la hora de ejectuar el script Update.sh:

```bash
HTTP request sent, awaiting response... 200 OK
Length: 11534 (11K) [text/html]
Saving to: ‘boost_1_80_0.tar.gz’

boost_1_80_0.tar.gz 100%[===================>]  11.26K  --.-KB/s    in 0s      

2025-01-23 10:52:46 (38.4 MB/s) - ‘boost_1_80_0.tar.gz’ saved [11534/11534]

Setup.sh: Extracting boost for Python 3.

gzip: stdin: not in gzip format
tar: Child returned status 1
tar: Error is not recoverable: exiting now
make: *** [Util/BuildTools/Linux.mk:142: setup] Error 2
```

Se soluciona de la siguiente manera:

1. Abre el archivo Setup.sh, encontrado en el path:

```bash
carla/Util/BuildTools/Setup.sh
```
2. Modifica la lína 94 y cambia este código: 

```bash
wget "https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION}/source/${BOOST_PACKAGE_BASENAME}.tar.gz" || true
```
por este:

```bash
wget "https://archives.boost.io/release/${BOOST_VERSION}/source/${BOOST_PACKAGE_BASENAME}.tar.gz" || true
```

Ahora borra todo el contenido dentro de */build* y vuelve a ejecutar *Update.sh* desde el directorio raíz de Carla ( donde has clonado el repositorio).

### make PythonAPI falta del paquete typing>=3.7.4 

Si se obtiene el siguiente error: 


```bash
  raise DistributionNotFound(req, requirers)
pkg_resources.DistributionNotFound: The 'typing>=3.7.4' distribution was not found and is required by typing-extensions
make: *** [Util/BuildTools/Linux.mk:92: PythonAPI] Error 1
```
El problema es la falta del paquete typing>=3.7.4 requerido por typing-extensions. Esto suele suceder cuando hay problemas de dependencias en el entorno de Python.

Solución:

Actualizar pip y setuptools

```bash
pip3 install --upgrade pip setuptools

```

Instalar typing y typing-extensions

```bash
pip3 install typing typing-extensions
```

### make PythonAPI importlib-resources para Python 3.10

Si se obtiene el siguiente error: 

```bash
raise DistributionNotFound(req, requirers)
pkg_resources.DistributionNotFound: The 'importlib-resources>=1.3; python_version < "3.9"' distribution was not found and is required by the application
make: *** [Util/BuildTools/Linux.mk:92: PythonAPI] Error 1
```

El problema principal es que no tienes instalada la dependencia importlib-resources para Python 3.10:

Instala importlib-resources:
```bash
pip3 install importlib-resources

```
