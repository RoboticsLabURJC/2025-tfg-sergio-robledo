#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------
#---Compara a nivel numerico dos ficheros csv (inferencia y humano)
#---Para obtener datos reales de las trayectorias y sus diferencias
#-------------------------------


# -------------------------
# CSV helpers
# -------------------------
def pick_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def load_positions_csv(path: str, who: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    xcol = pick_col(df, ["x", "pos_x", "X"])
    ycol = pick_col(df, ["y", "pos_y", "Y"])
    zcol = pick_col(df, ["z", "pos_z", "Z"])

    if xcol is None or ycol is None:
        raise ValueError(f"[{who}] faltan columnas x/y. Columnas: {list(df.columns)}")

    df = df.copy()
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    if zcol is not None:
        df[zcol] = pd.to_numeric(df[zcol], errors="coerce")

    df = df.dropna(subset=[xcol, ycol]).reset_index(drop=True)

    df["x"] = df[xcol].astype(np.float64)
    df["y"] = df[ycol].astype(np.float64)
    if zcol is None or df[zcol].isna().all():
        df["z"] = 0.2
    else:
        df["z"] = df[zcol].ffill().fillna(0.2).astype(np.float64)

    return df

def to_num(df: pd.DataFrame, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def robust_clip_speed_mps(s: pd.Series, vmax=20.0) -> pd.Series:
    return s.where((s >= 0.0) & (s <= vmax))

# ------------------------------
# Nearest Neighbor por posicion
# -----------------------------
def build_nn_index(P_inf: np.ndarray):
    """
    Devuelve un objeto con método query(P_ref)->(dist, idx)
    Emplea scipy
    """
    #try:
    from scipy.spatial import cKDTree
    tree = cKDTree(P_inf)

    class _SciPyNN:
        def query(self, Q):
            dist, idx = tree.query(Q, k=1, workers=-1)
            return dist.astype(np.float64), idx.astype(np.int64)

    return _SciPyNN(), "scipy.cKDTree"

# -------------------------
def mse(a: np.ndarray) -> float:
    return float(np.mean(a * a))

def rmse(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="dataset.csv (GT humano / grabado)")
    ap.add_argument("--inf", required=True, help="infer_log_*.csv (inferencia)")
    ap.add_argument("--plot", action="store_true", help="mostrar gráficas interactivas")
    ap.add_argument("--thr_min", type=float, default=0.25, help="usar solo HUMAN con throttle > umbral (si existe)")
    ap.add_argument("--v_min", type=float, default=0.01, help="usar solo HUMAN con speed > umbral (si existe)")
    ap.add_argument("--speed_vmax", type=float, default=20.0, help="clip speed en HUMAN para outliers")
    ap.add_argument("--max_pairs", type=int, default=0, help="limita nº de puntos HUMAN (0 = sin límite)")
    args = ap.parse_args()

    df_ref = load_positions_csv(args.ref, "REF")
    df_inf = load_positions_csv(args.inf, "INF")

    # --- filtros SOLO en REF/Human por throttle/speed  ---
    df_ref = to_num(df_ref, ["throttle", "speed", "steer"])
    if "speed" in df_ref.columns:
        df_ref["speed"] = robust_clip_speed_mps(df_ref["speed"], vmax=args.speed_vmax)

    if "throttle" in df_ref.columns:
        df_ref = df_ref[df_ref["throttle"] > args.thr_min]
    if "speed" in df_ref.columns:
        df_ref = df_ref[df_ref["speed"] > args.v_min]

    df_ref = df_ref.dropna(subset=["x", "y", "z"]).reset_index(drop=True)
    df_inf = df_inf.dropna(subset=["x", "y", "z"]).reset_index(drop=True)

    if len(df_ref) < 10 or len(df_inf) < 10:
        raise RuntimeError(f"Demasiados pocos puntos tras filtrar. REF={len(df_ref)} INF={len(df_inf)}")

    if args.max_pairs and args.max_pairs > 0 and len(df_ref) > args.max_pairs:
        df_ref = df_ref.iloc[:args.max_pairs].reset_index(drop=True)

    # matrices Nx3
    P_ref = df_ref[["x", "y", "z"]].to_numpy(dtype=np.float64)
    P_inf = df_inf[["x", "y", "z"]].to_numpy(dtype=np.float64)

    nn, method = build_nn_index(P_inf)
    dist, idx = nn.query(P_ref)

    P_match = P_inf[idx]
    dxyz = P_match - P_ref
    err_x, err_y, err_z = dxyz[:, 0], dxyz[:, 1], dxyz[:, 2]

    # ----------------------------------------------------------
    # "MSE por punto" (error cuadrático por muestra)
    # ----------------------------------------------------------
    se_x = err_x ** 2
    se_y = err_y ** 2
    se_z = err_z ** 2

    mse_point_xyz = (se_x + se_y + se_z) / 3.0   # (N,)
    mse_point_xy  = (se_x + se_y) / 2.0          # (N,)


    stats = {
        "NN_method": method,
        "N_pairs": int(len(P_ref)),
        "MSE_x": mse(err_x),
        "RMSE_x": rmse(err_x),
        "MSE_y": mse(err_y),
        "RMSE_y": rmse(err_y),
        "MSE_z": mse(err_z),
        "RMSE_z": rmse(err_z),
        "MSE_xyz_mean": float((mse(err_x) + mse(err_y) + mse(err_z)) / 3.0),
        "RMSE_dist": rmse(dist),
        "MSE_dist": mse(dist),
        "mean_dist": float(np.mean(dist)),
        "max_dist": float(np.max(dist)),
        "p95_dist": float(np.percentile(dist, 95)),
    }

    print("\n========== COMPARACIÓN POR POSICIÓN (nearest neighbor en XYZ) ==========")
    print(f"Método NN: {stats['NN_method']}")
    print(f"N emparejamientos: {stats['N_pairs']}")
    print(f"X   MSE={stats['MSE_x']:.6f}  RMSE={stats['RMSE_x']:.6f} (m)")
    print(f"Y   MSE={stats['MSE_y']:.6f}  RMSE={stats['RMSE_y']:.6f} (m)")
    print(f"Z   MSE={stats['MSE_z']:.6f}  RMSE={stats['RMSE_z']:.6f} (m)")
    print(f"XYZ mean MSE={stats['MSE_xyz_mean']:.6f}")
    print(f"Euclidean DIST MSE={stats['MSE_dist']:.6f}  RMSE={stats['RMSE_dist']:.6f} (m)")
    print(f"Euclidean DIST mean={stats['mean_dist']:.3f}")
    print(f"Percentile 95->  95% of the track is less than {stats['p95_dist']*100:.3f}cm from the human ")
    print(f"Max recorded distance between human and inference={stats['max_dist']:.3f} (m)")
    print("=======================================================================\n")

    #LLevar a csv
    # out_pairs = pd.DataFrame({
    #     "ref_i": np.arange(len(P_ref), dtype=np.int64),
    #     "ref_x": P_ref[:, 0], "ref_y": P_ref[:, 1], "ref_z": P_ref[:, 2],
    #     "inf_j": idx.astype(np.int64),
    #     "inf_x": P_match[:, 0], "inf_y": P_match[:, 1], "inf_z": P_match[:, 2],
    #     "err_x": err_x, "err_y": err_y, "err_z": err_z,
    #     "dist_xyz": dist,
    # })
    # out_pairs.to_csv("pairs_by_position.csv", index=False)

    if not args.plot:
        return

    # ==========================================================
    # PLOTS INTERACTIVOS: hover sobre REF(HUMAN) -> resalta match en INF
    # ==========================================================
    x_ref, y_ref = df_ref["x"].to_numpy(), df_ref["y"].to_numpy()
    x_inf, y_inf = df_inf["x"].to_numpy(), df_inf["y"].to_numpy()

    # Figura 1: trayectorias + puntos
    fig_xy, ax_xy = plt.subplots(figsize=(8, 6), dpi=110)
    ax_xy.plot(x_ref, y_ref, linewidth=1.0, label="HUMAN (humano)")
    ax_xy.plot(x_inf, y_inf, linewidth=1.0, alpha=0.9, label="INF (inferencia)")

    # Scatter (para hover)
    sc_ref = ax_xy.scatter(x_ref, y_ref, s=10, alpha=0.6, label="HUMAN points", picker=True)
    sc_inf = ax_xy.scatter(x_inf, y_inf, s=10, alpha=0.25, label="INF points")

    # Marcadores START
    ax_xy.scatter([x_ref[0]], [y_ref[0]], s=130, marker="*", zorder=6)
    ax_xy.text(x_ref[0], y_ref[0], "  START HUMAN", fontsize=9, va="center")

    ax_xy.scatter([x_inf[0]], [y_inf[0]], s=130, marker="*", zorder=6)
    ax_xy.text(x_inf[0], y_inf[0], "  START INF", fontsize=9, va="center")

    ax_xy.set_title("Trayectorias XY + hover: HUMAN i → INF j (nearest neighbor)")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best")

    # Elementos de highlight
    hl_ref = ax_xy.scatter([], [], s=120, marker="o", linewidths=2, facecolors="none", zorder=7)
    hl_inf = ax_xy.scatter([], [], s=120, marker="o", linewidths=2, facecolors="none", zorder=7)
    link_line, = ax_xy.plot([], [], linewidth=1.5, alpha=0.9)

    # Caja tooltip
    ann = ax_xy.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w", alpha=0.85),
        arrowprops=dict(arrowstyle="->", alpha=0.6),
    )
    ann.set_visible(False)

    # Figura 2: distancia NN por índice HUMAN (y hover enlazado)
    fig_d, ax_d = plt.subplots(figsize=(8, 3.8), dpi=110)
    ax_d.plot(dist, linewidth=1.2, label="distancia NN (m)")
    ax_d.set_title("Distancia al punto más cercano en INF (ordenado por índice HUMAN)")
    ax_d.set_xlabel("índice HUMAN (i)")
    ax_d.set_ylabel("dist (m)")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(loc="best")

    # Marcador vertical y punto destacado en dist
    vline = ax_d.axvline(0, linewidth=1.2, alpha=0.7)
    hl_d = ax_d.scatter([0], [dist[0]], s=70, zorder=5)
    ann_d = ax_d.annotate(
        "",
        xy=(0, dist[0]),
        xytext=(12, 12),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w", alpha=0.85),
        arrowprops=dict(arrowstyle="->", alpha=0.6),
    )
    ann_d.set_visible(False)

    # actualizar highlights para un índice i (HUMAN)
    def set_focus(i: int):
        i = int(np.clip(i, 0, len(P_ref) - 1))
        j = int(idx[i])

        # XY highlight
        hl_ref.set_offsets(np.array([[x_ref[i], y_ref[i]]]))
        hl_inf.set_offsets(np.array([[x_inf[j], y_inf[j]]]))
        link_line.set_data([x_ref[i], x_inf[j]], [y_ref[i], y_inf[j]])

        ann.xy = (x_ref[i], y_ref[i])
        ann.set_text(
            f"HUMAN i={i}\n"
            f"INF j={j}\n"
            f"dist={dist[i]:.3f} m\n"
            f"err=(dx={err_x[i]:+.3f}, dy={err_y[i]:+.3f}, dz={err_z[i]:+.3f})"
        )
        ann.set_visible(True)

        # Dist plot highlight
        vline.set_xdata([i, i])
        hl_d.set_offsets(np.array([[i, dist[i]]]))

        ann_d.xy = (i, dist[i])
        ann_d.set_text(f"HUMAN i={i} → INF j={j}\ndist={dist[i]:.3f} m")
        ann_d.set_visible(True)

        fig_xy.canvas.draw_idle()
        fig_d.canvas.draw_idle()

    # Hover en XY: busca punto HUMAN más cercano al cursor
    def on_move_xy(event):
        if event.inaxes != ax_xy:
            return
        cont, info = sc_ref.contains(event)
        if not cont:
            # si no estamos sobre un punto, oculta tooltip
            ann.set_visible(False)
            fig_xy.canvas.draw_idle()
            return

        # info["ind"] trae varios si hay solape: cogemos el primero
        i = int(info["ind"][0])
        set_focus(i)

    # Hover en dist: índice por xdata
    def on_move_dist(event):
        if event.inaxes != ax_d:
            return
        if event.xdata is None:
            return
        i = int(round(event.xdata))
        if 0 <= i < len(dist):
            set_focus(i)

    fig_xy.canvas.mpl_connect("motion_notify_event", on_move_xy)
    fig_d.canvas.mpl_connect("motion_notify_event", on_move_dist)

    # Arranca con i=0
    set_focus(0)

    # Figura 3: errores por eje
    fig_e, ax_e = plt.subplots(figsize=(8, 3.8), dpi=110)
    ax_e.plot(err_x, linewidth=1.0, label="err_x (m)")
    ax_e.plot(err_y, linewidth=1.0, label="err_y (m)")
    ax_e.plot(err_z, linewidth=1.0, label="err_z (m)")
    ax_e.set_title("Errores por eje (INF_nn - HUMAN) en orden HUMAN")
    ax_e.set_xlabel("índice HUMAN (i)")
    ax_e.set_ylabel("m")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(loc="best")


    # Figura 0: MSE por punto (orden HUMAN / REF)
    fig_mse, ax_mse = plt.subplots(figsize=(8, 3.8), dpi=110)
    ax_mse.plot(mse_point_xyz, linewidth=1.2, label="mse_point_xyz = (dx^2+dy^2+dz^2)/3")
  
    ax_mse.set_title("Error cuadrático por punto (ordenado por índice HUMAN/REF)")
    ax_mse.set_xlabel("índice HUMAN (i)")
    ax_mse.set_ylabel("m^2")
    ax_mse.grid(True, alpha=0.3)
    ax_mse.legend(loc="best")


    plt.show()

if __name__ == "__main__":
    main()
