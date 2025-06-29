import os
import time
import pandas as pd
import pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg

CSV_PATH = "dataset/dataset_Deepracer_BaseMap_1751132511826.csv"
df = pd.read_csv(CSV_PATH)

pygame.init()
screen = pygame.display.set_mode((1750, 1000))
pygame.display.set_caption("Visualizador Dataset DeepRacer (solo gráficas)")

font = pygame.font.SysFont(None, 26)
clock = pygame.time.Clock()

ruta_imagen = "/home/sergior/Downloads/pruebas/dataset/masks/mask1751132511826/1751125319931_mask_Deepracer_BaseMap_1751132511826.png"
imagen = pygame.image.load(ruta_imagen).convert_alpha()


def render_plot(data_slice, current_idx, window_duration=500):
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    fig.tight_layout(pad=2.0)

    current_time = data_slice.loc[current_idx, 'timestamp']
    start_time = current_time - window_duration
    end_time = current_time

   
    data_window = data_slice[(data_slice['timestamp'] >= start_time) & (data_slice['timestamp'] <= end_time)]
    timestamps = data_window['timestamp']

    if data_window.empty:
        data_window = data_slice.iloc[[current_idx]]
        timestamps = data_window['timestamp']

    # Gráfica 1: Throttle
    axs[0, 0].plot(timestamps, data_window['throttle'], color='green')
    axs[0, 0].set_title("Throttle")
    axs[0, 0].set_xlim(start_time, end_time)

    # Gráfica 2: Steer
    axs[0, 1].plot(timestamps, data_window['steer'], color='blue')
    axs[0, 1].set_title("Steer")
    axs[0, 1].set_xlim(start_time, end_time)

    # Gráfica 3: Brake
    axs[0, 2].plot(timestamps, data_window['brake'], color='red')
    axs[0, 2].set_title("Brake")
    axs[0, 2].set_xlim(start_time, end_time)

    # Gráfica 4: Speed
    axs[1, 0].plot(timestamps, data_window['speed'], color='orange')
    axs[1, 0].set_title("Speed")
    axs[1, 0].set_xlim(start_time, end_time)

    # Gráfica 5: Heading
    axs[1, 1].plot(timestamps, data_window['heading'], color='purple')
    axs[1, 1].set_title("Heading")
    axs[1, 1].set_xlim(start_time, end_time)

    axs[1, 2].axis('off')

 
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.buffer_rgba()
    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    plt.close(fig)
    return surf


index = 0
running = True
mid_path_name = "/home/sergior/Downloads/pruebas/dataset/"
while running and index < len(df):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    row = df.loc[index]
    plot_surface = render_plot(df, index)

    screen.fill((20, 20, 20))
    screen.blit(plot_surface, (800, 100)) 

 
    txt = f"Frame: {index} | Timestamp: {int(row['timestamp'])}"
    text_surf = font.render(txt, True, (255, 255, 255))
    screen.blit(text_surf, (50, 10))

    imagen_rgb_path = mid_path_name + row.iloc[0]
    img_rgb = pygame.image.load(imagen_rgb_path).convert_alpha() 
    screen.blit(img_rgb, (0, 40))  

    imagen_mask_path = mid_path_name + row.iloc[1]
    img_mask = pygame.image.load(imagen_mask_path).convert_alpha()
    screen.blit(img_mask, (0, 500))

    pygame.display.flip()
    time.sleep(0.033)
    index += 1

pygame.quit()


