#------------------------------------------------
#Codigo que permite visualizar el dataset deaseado 
#frame a frame mostrando ls valores para cada frame 
#------------------------------------------------

import os
import time
import pandas as pd
import pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg

#Path del directorio a examinar y fichero csv
BASE_PATH = "/home/sergior/Downloads/pruebas/clientCarlaScripts/Deepracer_BaseMap_1761064659946"
CSV_PATH = os.path.join(BASE_PATH, "dataset.csv")
df = pd.read_csv(CSV_PATH)

pygame.init()
screen = pygame.display.set_mode((1900, 1000))
pygame.display.set_caption("Visualizador Dataset DeepRacer")

font = pygame.font.SysFont(None, 26)
font_big = pygame.font.SysFont(None, 48)
clock = pygame.time.Clock()

# Graficas para cada elemento del dataset como throttle, steer, speed y heading
def render_plot(df, index, window=50):
    start = max(0, index - window)
    data_slice = df[start:index + 1]
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    fig.tight_layout(pad=2.0)

    timestamps = data_slice['timestamp']

    axs[0, 0].plot(timestamps, data_slice['throttle'], color='green')
    axs[0, 0].set_title("Throttle")
    axs[0, 0].set_xlim(timestamps.min(), timestamps.max())
    axs[0, 0].set_ylim(0.0, 1.1)

    axs[0, 1].plot(timestamps, data_slice['steer'], color='blue')
    axs[0, 1].set_title("Steer")
    axs[0, 1].set_xlim(timestamps.min(), timestamps.max())
    axs[0, 1].set_ylim(-1.0, 1.0)

    axs[1, 0].plot(timestamps, data_slice['brake'], color='red')
    axs[1, 0].set_title("Brake")
    axs[1, 0].set_xlim(timestamps.min(), timestamps.max())

    axs[1, 1].plot(timestamps, data_slice['speed'], color='orange')
    axs[1, 1].set_title("Speed")
    axs[1, 1].set_xlim(timestamps.min(), timestamps.max())
    axs[1, 1].set_ylim(0, 3)

    axs[2, 0].plot(timestamps, data_slice['heading'], color='purple')
    axs[2, 0].set_title("Heading")
    axs[2, 0].set_xlim(timestamps.min(), timestamps.max())
    axs[2, 0].set_ylim(-25.0, 25.0)

    axs[2, 1].axis('off')

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.buffer_rgba()
    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    plt.close(fig)
    return surf

# Obtener estado: izq,centro o dcha
def estado_to_text(e):
    if e == 1:
        return "IZQUIERDA"
    elif e == 2:
        return "CENTRO / RECTO"
    elif e == 3:
        return "DERECHA"
    return "N/A"

# Representar cada estado de un color
def estado_to_color(e):
    if e == 1:
        return (80, 160, 255)   # azul claro
    elif e == 2:
        return (255, 255, 255)  # blanco
    elif e == 3:
        return (255, 180, 0)    # naranja
    return (200, 200, 200)

index = 0
running = True

while running and index < len(df):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    row = df.loc[index]
    plot_surface = render_plot(df, index)

    screen.fill((20, 20, 20))
    screen.blit(plot_surface, (800, 100)) 

    # Cabecera
    txt = f"Frame: {index} | Timestamp: {int(row['timestamp'])}"
    text_surf = font.render(txt, True, (255, 255, 255))
    screen.blit(text_surf, (50, 10))

    # Cargar imágenes
    imagen_rgb_path = BASE_PATH + row.iloc[0]
    img_rgb = pygame.image.load(imagen_rgb_path).convert_alpha() 
    screen.blit(img_rgb, (0, 40))  

    imagen_mask_path = BASE_PATH + row.iloc[1]
    img_mask = pygame.image.load(imagen_mask_path).convert_alpha()
    screen.blit(img_mask, (0, 500))

    # Mostrar estado
    if 'estado' in df.columns:
        try:
            e_val = int(row['estado'])
        except Exception:
            e_val = 0
        estado_txt = estado_to_text(e_val)
        estado_col = estado_to_color(e_val)

        # marco semitransparente detrás del texto
        label_bg = pygame.Surface((460, 60), pygame.SRCALPHA)
        label_bg.fill((50, 50, 50, 140))
        screen.blit(label_bg, (1220, 20))

        estado_surf = font_big.render(f"ESTADO: {estado_txt}  ({e_val})", True, estado_col)
        screen.blit(estado_surf, (1240, 30))
    else:
        warn_surf = font.render("CSV sin columna 'estado'", True, (255, 100, 100))
        screen.blit(warn_surf, (1220, 30))

    pygame.display.flip()
    time.sleep(1/60)
    index += 1

pygame.quit()
