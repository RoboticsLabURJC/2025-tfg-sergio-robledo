#!/usr/bin/env python3
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_REPLAY     = "telemetryfromreplay.csv"
FILE_RECORDING  = "telemetrywhilerecording.csv"
FILE_ITER5      = "telemetryfromreplay5iter.csv"

WINDOW_SECONDS  = 5.0
REFRESH_SEC     = 0.10
Y_MIN, Y_MAX    = 0.0, 5.0

def load_series(path):

    try:
        if not os.path.isfile(path):
            return None, None
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        if df.empty:
            return None, None

        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols
        if "sim_time" in df.columns and "speed_m_s" in df.columns:
            t = pd.to_numeric(df["sim_time"], errors="coerce")
            v = pd.to_numeric(df["speed_m_s"], errors="coerce")
        else:
            num = df.apply(pd.to_numeric, errors="coerce")
            numeric_cols = [c for c in num.columns if num[c].notna().any()]
            if len(numeric_cols) < 2:
                return None, None
            t = num[numeric_cols[0]]
            v = num[numeric_cols[1]]

        mask = t.notna() & v.notna()
        if not mask.any():
            return None, None

        t = t[mask].astype(float).to_numpy()
        v = v[mask].astype(float).to_numpy()

        order = np.argsort(t)
        t = t[order]
        v = v[order]
        t = t - t[0]

        v = np.clip(v, Y_MIN, Y_MAX)
        return t, v
    except Exception:
        return None, None

def main():
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(0, WINDOW_SECONDS)
    ax.set_xlabel("Tiempo sim (s)")
    ax.set_ylabel("Velocidad (m/s)")
    ax.set_title("Velocidad (ventana deslizante 5 s, reproducción 1×)")

    (line_replay,)    = ax.plot([], [], lw=2, label=f"Speed from replay EMA(alpha 0.2)")
    (line_recording,) = ax.plot([], [], lw=2, label=f"Speed while recording")
    (line_iter5,) = ax.plot([], [], lw=2, label=f"Speed from replay, calculus every 5 sec")

    ax.legend(loc="upper right")
    fig.tight_layout()

    wall_t0 = time.monotonic()

    try:
        while True:
            # Tiempo reproducido a 1× 
            wall_elapsed = time.monotonic() - wall_t0

            # Carga/recarga los CSV en cada iteración 
            t1, v1 = load_series(FILE_REPLAY)
            t2, v2 = load_series(FILE_RECORDING)
            t3, v3 = load_series(FILE_ITER5)

            # Determina hasta dónde hay datos en los CSV
            max_avail = 0.0
            if t1 is not None and len(t1) > 0: max_avail = max(max_avail, float(t1[-1]))
            if t2 is not None and len(t2) > 0: max_avail = max(max_avail, float(t2[-1]))
            if t3 is not None and len(t3) > 0: max_avail = max(max_avail, float(t3[-1]))

            # Fin visible
            vis_end = min(wall_elapsed, max_avail)
            vis_start = max(0.0, vis_end - WINDOW_SECONDS)

            # Actualiza cada línea 
            if t1 is not None and len(t1) > 0:
                m1 = (t1 >= vis_start) & (t1 <= vis_end)
                line_replay.set_data(t1[m1], v1[m1])
            else:
                line_replay.set_data([], [])

            if t2 is not None and len(t2) > 0:
                m2 = (t2 >= vis_start) & (t2 <= vis_end)
                line_recording.set_data(t2[m2], v2[m2])
            else:
                line_recording.set_data([], [])

            if t3 is not None and len(t3) > 0: 
                m3 = (t3 >= vis_start) & (t3 <= vis_end)
                line_iter5.set_data(t3[m3], v3[m3])
            else:
                line_iter5.set_data([], [])

            # Mueve la ventana X
            ax.set_xlim(vis_start, max(vis_start + WINDOW_SECONDS, WINDOW_SECONDS))

            # Dibuja
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(REFRESH_SEC)

    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
