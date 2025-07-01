import os
import time
import pandas as pd
import pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg

CSV_PATH = "dataset/dataset_Deepracer_BaseMap_1751392009964.csv"
df = pd.read_csv(CSV_PATH)

pygame.init()
screen = pygame.display.set_mode((1900, 1000))
pygame.display.set_caption("Visualizador Dataset DeepRacer (solo gráficas)")

font = pygame.font.SysFont(None, 26)
clock = pygame.time.Clock()

def render_plot(df, index, window=50):
    start = max(0, index - window)
    data_slice = df[start:index + 1]
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    fig.tight_layout(pad=2.0)

    timestamps = data_slice['timestamp']

    axs[0, 0].plot(timestamps, data_slice['throttle'], color='green')
    axs[0, 0].set_title("Throttle")
    axs[0, 0].set_xlim(timestamps.min(), timestamps.max())
    axs[0, 0].set_ylim(0.0, 0.8)

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
    axs[1, 1].set_ylim(0, 1.5)

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


