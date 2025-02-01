---
title: "Introducción"
last_modified_at: 2025-01-02T18:02:00
categories:
  - Blog
tags:
  - Intro
---

## ¡Hola!

Soy Sergio Robledo, estudiante de Ingeniería Robótica Software de la universidad Rey Juan Carlos de Madrid.

Estos blogs van a representar distintas partes que componen el trabajo final, así como diversas explicaciones teóricas, procesos de instalación de ciertos paquetes y códigos explicados.

## Explicación del proyecto

El proyecto que se va a realizar consistirá en la navegación autónoma a lo largo de una pista de carreras con un robot [*AWS DeepRacer*](https://aws.amazon.com/es/deepracer/).

Para lograr que este robot recorra una pista nunca antes vista, en el menor tiempo posible, se empleará una técnica de aprendizaje automático conocida como *Aprendizaje por Imitación* ( en inglés *Imitation Learning*). En ella, un agente aprende a realizar una tarea observando y replicando el comportamiento de un experto humano.

## Proyecto paso a paso

El proyecto contará con una serie de fases:

1. Instalación del simulador [*CARLA*](https://carla.org/) así como la creación de los mapas e implementación del modelo del robot.

2. Recopilación de datos en simulación.

3. Comprobación de efectividad en simulación.

4. Entrenamiento en robot real y combinación de datos.

5. Comprobación de efectividad en escenario real y pruebas de robustez para la precisión del modelo.

--- 

<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/intro_images/carla.jpeg" alt="Simulador Carla">
</figure>

<figure class="align-center" style="max-width: 100%">
  <img src="{{ site.url }}{{ site.baseurl }}/images/intro_images/deepracer.jpg" alt="AWS DeepRacer">
</figure>
