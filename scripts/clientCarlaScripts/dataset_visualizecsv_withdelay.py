#!/usr/bin/env python3
#-------------------------------------------------
# Visualize dataset with speed offset (+0.2 s)
#-------------------------------------------------

import os
import time
import pandas as pd
import pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import argparse
import sys
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize dataset with speed and speed +0.2s offset"
    )
    parser.add_argument(
        "--base_path",
        required=True,
        help="Base directory where dataset.csv is located"
    )
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()


def add_speed_lead_column(df, offset_sec=0.2):
    """
    Crea una columna 'speed_lead_0_2' que contiene la velocidad
    adelantada 'offset_sec' segundos respecto a cada fila.

    Usa búsqueda por timestamp para encontrar el índice futuro.
    """
    if "timestamp" not in df.columns or "speed" not in df.columns:
        print("[WARN] No se encontró 'timestamp' o 'speed' en el CSV. "
              "No se aplicará el offset de velocidad.")
        df["speed_lead_0_2"] = df.get("speed", 0.0)
        return df

    ts = df["timestamp"].values.astype(float)
    sp = df["speed"].values.astype(float)

    # Para cada ts[i], buscamos el índice del primer ts[j] >= ts[i] + offset_sec
    target_ts = ts + offset_sec
    idx_future = np.searchsorted(ts, target_ts, side="left")

    # Clamp de índices que se van fuera
    idx_future[idx_future >= len(ts)] = len(ts) - 1

    speed_lead = sp[idx_future]
    df["speed_lead_0_2"] = speed_lead
    return df


# Plots para throttle, steer, brake, speed y speed_lead
def render_plot(df, index, window=50):
    start = max(0, index - window)
    data_slice = df[start:index + 1]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.tight_layout(pad=2.0)

    timestamps = data_slice['timestamp']

    # Throttle
    axs[0, 0].plot(timestamps, data_slice['throttle'], color='green')
    axs[0, 0].set_title("Throttle [0,1]")
    axs[0, 0].set_xlim(timestamps.min(), timestamps.max())
    axs[0, 0].set_ylim(0.0, 1.1)

    # Steer
    axs[0, 1].plot(timestamps, data_slice['steer'], color='blue')
    axs[0, 1].set_title("Steer [-1,1]")
    axs[0, 1].set_xlim(timestamps.min(), timestamps.max())
    axs[0, 1].set_ylim(-1.1, 1.1)

    # Brake
    if "brake" in data_slice.columns:
        axs[1, 0].plot(timestamps, data_slice['brake'], color='red')
        axs[1, 0].set_title("Brake [0,1]")
        axs[1, 0].set_xlim(timestamps.min(), timestamps.max())
        axs[1, 0].set_ylim(0.0, 1.0)
    else:
        axs[1, 0].set_title("Brake [0,1] (no column)")
        axs[1, 0].set_xlim(timestamps.min(), timestamps.max())

    # Speed + Speed adelantada
    axs[1, 1].plot(timestamps, data_slice['speed'], color='orange', label='Speed (now)')
    if "speed_lead_0_2" in data_slice.columns:
        axs[1, 1].plot(timestamps, data_slice['speed_lead_0_2'],
                       color='magenta', linestyle='--', label='Speed (+0.2 s)')
    axs[1, 1].set_title("Speed (m/s) + offset")
    axs[1, 1].set_xlim(timestamps.min(), timestamps.max())
    axs[1, 1].set_ylim(0, 3.5)
    axs[1, 1].legend(loc="upper left")

    # Pasar figura a surface de pygame
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.buffer_rgba()
    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    plt.close(fig)
    return surf


def main():
    args = parse_args()
    BASE_PATH = args.base_path

    CSV_PATH = os.path.join(BASE_PATH, "dataset.csv")
    if not os.path.isfile(CSV_PATH):
        print(f"[ERROR] Unable to find dataset.csv en: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    # Añadir columna de velocidad adelantada 0.2 s
    df = add_speed_lead_column(df, offset_sec=0.2)

    pygame.init()
    screen = pygame.display.set_mode((1900, 1000))
    pygame.display.set_caption("Visualize Dataset DeepRacer (+0.2s speed)")

    font = pygame.font.SysFont(None, 26)
    clock = pygame.time.Clock()

    index = 0
    running = True

 
    columns = list(df.columns)
    if "rgb_path" in columns:
        rgb_col = "rgb_path"
    else:
        rgb_col = columns[0]

    mask_col = None
    if "mask_path" in columns:
        mask_col = "mask_path"
    elif len(columns) > 1 and "rgb_path" in columns and "mask_path" not in columns:
        mask_col = columns[1]

    while running and index < len(df):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        row = df.loc[index]
        plot_surface = render_plot(df, index)

        screen.fill((20, 20, 20))
        screen.blit(plot_surface, (800, 100)) 

        # Header principal
        ts = float(row['timestamp'])
        speed_now   = float(row['speed'])
        speed_lead  = float(row.get('speed_lead_0_2', speed_now))
        txt = (f"Frame: {index} | Timestamp: {int(ts)} | "
               f"Speed now: {speed_now:.2f} m/s | Speed +0.2s: {speed_lead:.2f} m/s")
        text_surf = font.render(txt, True, (255, 255, 255))
        screen.blit(text_surf, (50, 10))

        # Cargar RGB
        rgb_rel = str(row[rgb_col])
        imagen_rgb_path = os.path.join(BASE_PATH, rgb_rel.lstrip("/"))
        if os.path.isfile(imagen_rgb_path):
            try:
                img_rgb = pygame.image.load(imagen_rgb_path).convert_alpha()
                screen.blit(img_rgb, (0, 40))
            except Exception as e:
                warn = font.render(f"RGB error: {e}", True, (255, 100, 100))
                screen.blit(warn, (0, 40))
        else:
            warn = font.render(f"RGB not found: {imagen_rgb_path}", True, (255, 100, 100))
            screen.blit(warn, (0, 40))

        # Cargar mask
        if mask_col is not None:
            mask_rel = str(row[mask_col])
            imagen_mask_path = os.path.join(BASE_PATH, mask_rel.lstrip("/"))
            if os.path.isfile(imagen_mask_path):
                try:
                    img_mask = pygame.image.load(imagen_mask_path).convert_alpha()
                    screen.blit(img_mask, (0, 500))
                except Exception as e:
                    warn = font.render(f"Mask error: {e}", True, (255, 100, 100))
                    screen.blit(warn, (0, 500))
            else:
                warn = font.render(f"Mask not found: {imagen_mask_path}", True, (255, 100, 100))
                screen.blit(warn, (0, 500))

        pygame.display.flip()
        clock.tick(60)
        index += 1

    pygame.quit()


if __name__ == "__main__":
    main()
