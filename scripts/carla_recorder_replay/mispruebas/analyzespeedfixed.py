#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Y_MIN, Y_MAX = 0.0, 5.0

def load_series(path, rebase=False):

    try:
        if not os.path.isfile(path):
            print(f"[INFO] No existe: {path}")
            return None, None
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        if df.empty:
            print(f"[INFO] Vacío: {path}")
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
                print(f"[INFO] No se encontraron 2 columnas numéricas en {path}")
                return None, None
            t = num[numeric_cols[0]]
            v = num[numeric_cols[1]]

        mask = t.notna() & v.notna()
        if not mask.any():
            print(f"[INFO] Sin datos válidos en {path}")
            return None, None

        t = t[mask].astype(float).to_numpy()
        v = v[mask].astype(float).to_numpy()

        # Ordenar por tiempo
        order = np.argsort(t)
        t = t[order]
        v = v[order]

        # Rebase opcional (alinear a 0 cada CSV)
        if rebase:
            t = t - t[0]

        # Limitar velocidades
        v = np.clip(v, Y_MIN, Y_MAX)
        return t, v
    except Exception as e:
        print(f"[WARN] Error leyendo {path}: {e}")
        return None, None

def parse_args():
    ap = argparse.ArgumentParser(description="Plot estático de velocidades en ventana fija.")
    ap.add_argument("--tstart", type=float, required=True, help="Tiempo de inicio (s) del eje X.")
    ap.add_argument("--tend",   type=float, required=True, help="Tiempo de fin (s) del eje X.")
    ap.add_argument("--rebase", type=int, default=1,
                    help="1=alinear cada CSV a t=0 en su primer dato (por defecto), 0=usar tiempos absolutos.")
    ap.add_argument("--replay", default="telemetryfromreplay.csv",
                    help="CSV de velocidad del replay.")
    ap.add_argument("--record", default="telemetrywhilerecording.csv",
                    help="CSV de velocidad mientras se grababa.")
    ap.add_argument("--iter5",  default="telemetryfromreplay5iter.csv",
                    help="CSV de velocidad del replay (cada 5 iters).")
    ap.add_argument("--title",  default="Velocidad (ventana fija)", help="Título del gráfico.")
    return ap.parse_args()

def report_range(name, t):
    if t is None or len(t) == 0:
        print(f"[{name}] sin datos")
    else:
        print(f"[{name}] rango t = [{t[0]:.3f}, {t[-1]:.3f}] con {len(t)} muestras")

def main():
    args = parse_args()
    t0 = float(args.tstart)
    t1 = float(args.tend)
    if t1 <= t0:
        raise ValueError("--tend debe ser mayor que --tstart")

    rebase = bool(args.rebase)

    # Leer series
    t_replay,  v_replay  = load_series(args.replay, rebase=rebase)
    t_record,  v_record  = load_series(args.record, rebase=rebase)
    t_iter5,   v_iter5   = load_series(args.iter5,  rebase=rebase)

    # Diagnóstico de rangos
    print(f"\n[DIAG] rebase={'ON' if rebase else 'OFF'} | Ventana solicitada = [{t0}, {t1}]")
    report_range("replay",  t_replay)
    report_range("record",  t_record)
    report_range("iter5",   t_iter5)

    # Preparar figura
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.set_title(args.title)
    ax.set_xlabel("Tiempo sim (s)" + (" (rebased)" if rebase else ""))
    ax.set_ylabel("Velocidad (m/s)")
    ax.set_xlim(t0, t1)
    ax.set_ylim(Y_MIN, Y_MAX)

    # Pintar si hay puntos en la ventana
    any_plotted = False

    if t_replay is not None and len(t_replay) > 0:
        m = (t_replay >= t0) & (t_replay <= t1)
        print(f"[replay] puntos en ventana: {int(m.sum())}")
        ax.plot(t_replay[m], v_replay[m], lw=2, label="Replay (EMA 0.2)")
        any_plotted = any_plotted or m.any()

    if t_record is not None and len(t_record) > 0:
        m = (t_record >= t0) & (t_record <= t1)
        print(f"[record] puntos en ventana: {int(m.sum())}")
        ax.plot(t_record[m], v_record[m], lw=2, label="Recording")
        any_plotted = any_plotted or m.any()

    if t_iter5 is not None and len(t_iter5) > 0:
        m = (t_iter5 >= t0) & (t_iter5 <= t1)
        print(f"[iter5 ] puntos en ventana: {int(m.sum())}")
        ax.plot(t_iter5[m], v_iter5[m], lw=2, label="Replay (cada 5 iters)")
        any_plotted = any_plotted or m.any()

    if not any_plotted:
        print("\n[AVISO] No hay puntos dentro de la ventana solicitada.\n"
              " - Prueba con --rebase 1 (por defecto) si los CSV no comparten tiempo absoluto.\n"
              " - O ajusta --tstart/--tend para cubrir los rangos impresos arriba.")

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
