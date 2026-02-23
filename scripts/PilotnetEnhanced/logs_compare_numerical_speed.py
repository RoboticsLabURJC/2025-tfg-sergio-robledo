#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# CSV helpers

def pick_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def to_num(df: pd.DataFrame, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def mse(a: np.ndarray) -> float:
    return float(np.mean(a * a))

def rmse(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a)))

# -------------------------
# Carga REF/INF con posición + velocidad

def load_ref(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    xcol = pick_col(df, ["x", "pos_x", "X"])
    ycol = pick_col(df, ["y", "pos_y", "Y"])
    zcol = pick_col(df, ["z", "pos_z", "Z"])
    scol = pick_col(df, ["speed", "speed_mps", "v", "vel"])

    if xcol is None or ycol is None:
        raise ValueError(f"[REF] faltan columnas x/y. Columnas: {list(df.columns)}")
    if scol is None:
        raise ValueError(f"[REF] falta columna de velocidad (speed). Columnas: {list(df.columns)}")

    df = df.copy()
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    if zcol is not None:
        df[zcol] = pd.to_numeric(df[zcol], errors="coerce")
    df[scol] = pd.to_numeric(df[scol], errors="coerce")

    df = df.dropna(subset=[xcol, ycol, scol]).reset_index(drop=True)

    df["x"] = df[xcol].astype(np.float64)
    df["y"] = df[ycol].astype(np.float64)
    if zcol is None or df[zcol].isna().all():
        df["z"] = 0.2
    else:
        df["z"] = df[zcol].ffill().fillna(0.2).astype(np.float64)

    # REF speed
    df["v_mps"] = df[scol].astype(np.float64)

    return df[["x", "y", "z", "v_mps"]]

def load_inf(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    xcol = pick_col(df, ["x", "pos_x", "X"])
    ycol = pick_col(df, ["y", "pos_y", "Y"])
    zcol = pick_col(df, ["z", "pos_z", "Z"])
    scol = pick_col(df, ["speed_kmh", "speed", "v_kmh", "vel_kmh"])

    if xcol is None or ycol is None:
        raise ValueError(f"[INF] faltan columnas x/y. Columnas: {list(df.columns)}")
    if scol is None:
        raise ValueError(f"[INF] falta columna de velocidad (speed_kmh). Columnas: {list(df.columns)}")

    df = df.copy()
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    if zcol is not None:
        df[zcol] = pd.to_numeric(df[zcol], errors="coerce")
    df[scol] = pd.to_numeric(df[scol], errors="coerce")

    df = df.dropna(subset=[xcol, ycol, scol]).reset_index(drop=True)

    df["x"] = df[xcol].astype(np.float64)
    df["y"] = df[ycol].astype(np.float64)
    if zcol is None or df[zcol].isna().all():
        df["z"] = 0.2
    else:
        df["z"] = df[zcol].ffill().fillna(0.2).astype(np.float64)

    # INF speed viene en km/h -> convertir a m/s
    df["v_mps"] = (df[scol].astype(np.float64) / 3.6)

    return df[["x", "y", "z", "v_mps"]]

# ------------------------------
# Nearest Neighbor por posición

def build_nn_index(P_inf: np.ndarray):
    from scipy.spatial import cKDTree
    tree = cKDTree(P_inf)

    class _SciPyNN:
        def query(self, Q):
            dist, idx = tree.query(Q, k=1, workers=-1)
            return dist.astype(np.float64), idx.astype(np.int64)

    return _SciPyNN(), "scipy.cKDTree"

# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="CSV humano (tiene speed)")
    ap.add_argument("--inf", required=True, help="CSV inferencia (tiene speed_kmh)")
    ap.add_argument("--plot", action="store_true", help="mostrar plots interactivos")
    ap.add_argument("--max_pairs", type=int, default=0, help="limitar nº puntos REF (0=sin limite)")
    ap.add_argument("--nn_max_dist", type=float, default=0.0,
                    help="si >0, descarta pares cuyo NN por posición sea > este umbral (m)")
    args = ap.parse_args()

    df_ref = load_ref(args.ref)
    df_inf = load_inf(args.inf)

    if len(df_ref) < 10 or len(df_inf) < 10:
        raise RuntimeError(f"Pocos puntos. REF={len(df_ref)} INF={len(df_inf)}")

    if args.max_pairs and args.max_pairs > 0 and len(df_ref) > args.max_pairs:
        df_ref = df_ref.iloc[:args.max_pairs].reset_index(drop=True)

    P_ref = df_ref[["x","y","z"]].to_numpy(np.float64)
    P_inf = df_inf[["x","y","z"]].to_numpy(np.float64)

    v_ref = df_ref["v_mps"].to_numpy(np.float64)
    v_inf = df_inf["v_mps"].to_numpy(np.float64)

    nn, method = build_nn_index(P_inf)
    dist_pos, idx = nn.query(P_ref)

    # filtrar emparejamientos malos por distancia espacial
    keep = np.ones_like(dist_pos, dtype=bool)
    if args.nn_max_dist and args.nn_max_dist > 0:
        keep = dist_pos <= float(args.nn_max_dist)

    P_ref_k = P_ref[keep]
    idx_k = idx[keep]
    dist_k = dist_pos[keep]

    v_ref_k = v_ref[keep]
    v_inf_match = v_inf[idx_k]

    # error de velocidad en los puntos emparejados
    err_v = v_inf_match - v_ref_k
    abs_err_v = np.abs(err_v)

    stats = {
        "NN_method": method,
        "N_pairs": int(len(err_v)),
        "MSE_speed": mse(err_v),
        "RMSE_speed": rmse(err_v),
        "mean_signed": float(np.mean(err_v)),
        "mean_abs": float(np.mean(abs_err_v)),
        "p95_abs": float(np.percentile(abs_err_v, 95)),
        "max_abs": float(np.max(abs_err_v)),
        "mean_nn_dist": float(np.mean(dist_k)),
        "p95_nn_dist": float(np.percentile(dist_k, 95)),
        "max_nn_dist": float(np.max(dist_k)),
    }

    print("\n========== COMPARACIÓN VELOCIDAD (NN por POSICIÓN XYZ) ==========")
    print(f"Método NN: {stats['NN_method']}")
    print(f"N emparejamientos usados: {stats['N_pairs']}")
    if args.nn_max_dist and args.nn_max_dist > 0:
        print(f"Filtro NN: se descartaron pares con dist_pos > {args.nn_max_dist:.3f} m")
    print(f"NN dist mean={stats['mean_nn_dist']:.4f} m | p95={stats['p95_nn_dist']:.4f} m | max={stats['max_nn_dist']:.4f} m")
    print(f"Speed MSE={stats['MSE_speed']:.6f} (m/s)^2 | RMSE={stats['RMSE_speed']:.6f} m/s")
    print(f"Mean signed (INF-REF)={stats['mean_signed']:.4f} m/s")
    print(f"Mean abs diff={stats['mean_abs']:.4f} m/s")
    print(f"95% abs diff < {stats['p95_abs']:.4f} m/s")
    print(f"Max abs diff={stats['max_abs']:.4f} m/s")
    print("=================================================================\n")

    if not args.plot:
        return

    # ==========================================================
    ref_i_all = np.arange(len(P_ref), dtype=np.int64)
    ref_i_kept = ref_i_all[keep]  # índice original del ref para cada par válido

    x_ref, y_ref = df_ref["x"].to_numpy(), df_ref["y"].to_numpy()
    x_inf, y_inf = df_inf["x"].to_numpy(), df_inf["y"].to_numpy()

    # --- Figura 1: trayectorias XY ---
    fig_xy, ax_xy = plt.subplots(figsize=(8, 6), dpi=110)
    ax_xy.plot(x_ref, y_ref, linewidth=1.0, label="HUMAN (REF)")
    ax_xy.plot(x_inf, y_inf, linewidth=1.0, alpha=0.9, label="INF")

    sc_ref = ax_xy.scatter(x_ref, y_ref, s=10, alpha=0.6, label="HUMAN points", picker=True)
    sc_inf = ax_xy.scatter(x_inf, y_inf, s=10, alpha=0.25, label="INF points")

    ax_xy.set_title("Trayectorias XY + hover: HUMAN i → INF j (NN por posición)")
    ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
    ax_xy.axis("equal"); ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best")

    hl_ref = ax_xy.scatter([], [], s=140, marker="o", linewidths=2, facecolors="none", zorder=7)
    hl_inf = ax_xy.scatter([], [], s=140, marker="o", linewidths=2, facecolors="none", zorder=7)
    link_line, = ax_xy.plot([], [], linewidth=1.5, alpha=0.9)

    ann = ax_xy.annotate(
        "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w", alpha=0.85),
        arrowprops=dict(arrowstyle="->", alpha=0.6),
    )
    ann.set_visible(False)

    # --- Figura 2: distancia NN por índice HUMAN (solo pares válidos) ---
    fig_d, ax_d = plt.subplots(figsize=(8, 3.8), dpi=110)
    ax_d.plot(ref_i_kept, dist_k, linewidth=1.2, label="distancia NN (m)")
    ax_d.set_title("Distancia al punto más cercano en INF (orden HUMAN)")
    ax_d.set_xlabel("índice HUMAN (i)")
    ax_d.set_ylabel("dist (m)")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(loc="best")

    vline = ax_d.axvline(ref_i_kept[0], linewidth=1.2, alpha=0.7)
    hl_d = ax_d.scatter([ref_i_kept[0]], [dist_k[0]], s=70, zorder=5)

    # --- Figura 3: velocidades emparejadas ---
    fig_v, ax_v = plt.subplots(figsize=(8, 3.8), dpi=110)
    ax_v.plot(ref_i_kept, v_ref_k, linewidth=1.2, label="HUMAN speed (m/s)")
    ax_v.plot(ref_i_kept, v_inf_match, linewidth=1.2, label="INF speed@NN (m/s)")
    ax_v.set_title("Velocidad en puntos emparejados por NN (posición)")
    ax_v.set_xlabel("índice HUMAN (i)")
    ax_v.set_ylabel("m/s")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc="best")

    vline_v = ax_v.axvline(ref_i_kept[0], linewidth=1.2, alpha=0.7)
    hl_v = ax_v.scatter([ref_i_kept[0]], [v_ref_k[0]], s=70, zorder=5)

    # --- Figura 4: error de velocidad ---
    fig_ev, ax_ev = plt.subplots(figsize=(8, 3.8), dpi=110)
    ax_ev.plot(ref_i_kept, err_v, linewidth=1.2, label="err_v = v_INF@NN - v_HUMAN (m/s)")
    ax_ev.plot(ref_i_kept, abs_err_v, linewidth=1.0, alpha=0.8, label="|err_v| (m/s)")
    ax_ev.set_title("Error de velocidad en pares NN")
    ax_ev.set_xlabel("índice HUMAN (i)")
    ax_ev.set_ylabel("m/s")
    ax_ev.grid(True, alpha=0.3)
    ax_ev.legend(loc="best")

    vline_ev = ax_ev.axvline(ref_i_kept[0], linewidth=1.2, alpha=0.7)
    hl_ev = ax_ev.scatter([ref_i_kept[0]], [err_v[0]], s=70, zorder=5)

    ref_to_k = {int(i): int(k) for k, i in enumerate(ref_i_kept)}

    def set_focus(i_global: int):
        i_global = int(np.clip(i_global, 0, len(P_ref) - 1))
        if i_global not in ref_to_k:
            ann.set_visible(False)
            fig_xy.canvas.draw_idle()
            return

        k = ref_to_k[i_global]
        j = int(idx_k[k])

        # XY highlight
        hl_ref.set_offsets(np.array([[x_ref[i_global], y_ref[i_global]]]))
        hl_inf.set_offsets(np.array([[x_inf[j], y_inf[j]]]))
        link_line.set_data([x_ref[i_global], x_inf[j]], [y_ref[i_global], y_inf[j]])

        ann.xy = (x_ref[i_global], y_ref[i_global])
        ann.set_text(
            f"HUMAN i={i_global}\n"
            f"INF j={j}\n"
            f"dist_pos={dist_k[k]:.3f} m\n"
            f"v_H={v_ref_k[k]:.3f} m/s\n"
            f"v_I={v_inf_match[k]:.3f} m/s\n"
            f"err_v={err_v[k]:+.3f} m/s"
        )
        ann.set_visible(True)

        # Dist plot
        vline.set_xdata([i_global, i_global])
        hl_d.set_offsets(np.array([[i_global, dist_k[k]]]))

        # Speed plot
        vline_v.set_xdata([i_global, i_global])
        hl_v.set_offsets(np.array([[i_global, v_ref_k[k]]]))

        # Err plot
        vline_ev.set_xdata([i_global, i_global])
        hl_ev.set_offsets(np.array([[i_global, err_v[k]]]))

        fig_xy.canvas.draw_idle()
        fig_d.canvas.draw_idle()
        fig_v.canvas.draw_idle()
        fig_ev.canvas.draw_idle()

    def on_move_xy(event):
        if event.inaxes != ax_xy:
            return
        cont, info = sc_ref.contains(event)
        if not cont:
            ann.set_visible(False)
            fig_xy.canvas.draw_idle()
            return
        i = int(info["ind"][0])
        set_focus(i)

    def on_move_dist(event):
        if event.inaxes != ax_d or event.xdata is None:
            return
        i = int(round(event.xdata))
        set_focus(i)

    fig_xy.canvas.mpl_connect("motion_notify_event", on_move_xy)
    fig_d.canvas.mpl_connect("motion_notify_event", on_move_dist)

    set_focus(int(ref_i_kept[0]))
    plt.show()

if __name__ == "__main__":
    main()
