#!/usr/bin/env python3
import re
import sys
import time
import carla
import pygame
import numpy as np

LOG_FILE = "/tmp/testlog3.log"
HOST, PORT = "localhost", 3010
WIDTH, HEIGHT = 800, 600
FPS = 30.0
FIXED_DT = 1.0 / FPS


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CARLA Replay")

    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)

    info = client.show_recorder_file_info(LOG_FILE, False)
    print(info)


if __name__ == "__main__":
    main()
