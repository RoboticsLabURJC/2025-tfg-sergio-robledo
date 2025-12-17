#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        df["z"] = df[zcol].fillna(method="ffill").fillna(0.2).astype(np.float64)

    return df

def to_num(df: pd.DataFrame, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def robust_clip_speed_mps(s: pd.Series, vmax=20.0) -> pd.Series:
    return s.where((s >= 0.0) & (s <= vmax))

# -------------------------
# Nearest neighbor (pos)
# -------------------------
def build_nn_index(P_inf: np.ndarray):
    """
    Devuelve un objeto con método query(P_ref)->(dist, idx)
    Intenta scipy, luego sklearn, si no, fallback brute-force.
    """
    # 1) SciPy
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(P_inf)
        class _SciPyNN:
            def query(self, Q):
                dist, idx = tree.query(Q, k=1, workers=-1)
                return dist.astype(np.float64), idx.astype(np.int64)
        return _SciPyNN(), "scipy.cKDTree"
    except Exception:
        pass

    # 2) scikit-learn
    try:
        from sklearn.neighbors import KDTree
        tree = KDTree(P_inf, leaf_size=40)
        class _SkNN:
            def query(self, Q):
                dist, idx = tree.query(Q, k=1)
                return dist[:, 0].astype(np.float64), idx[:, 0].astype(np.int64)
        return _SkNN(), "sklearn.KDTree"
    except Exception:
        pass

    # 3) brute force (por bloques)
    class _BruteNN:
        def __init__(self, P):
            self.P = P
        def query(self, Q):
            P = self.P
            n = Q.shape[0]
            best_dist2 = np.full(n, np.inf, dtype=np.float64)
            best_idx = np.zeros(n, dtype=np.int64)

            # Ajusta chunk según tamaño
            chunk = 5000
            for i0 in range(0, P.shape[0], chunk):
                Pi = P[i0:i0+chunk]  # (m,3)
                # dist2: (n,m) = sum((Q[:,None,:]-Pi[None,:,:])^2)
                d2 = ((Q[:, None, :] - Pi[None, :, :]) ** 2).sum(axis=2)
                j = np.argmin(d2, axis=1)
                d2min = d2[np.arange(n), j]
                better = d2min < best_dist2
                best_dist2[better] = d2min[better]
                best_idx[better] = (i0 + j[better]).astype(np.int64)

            return np.sqrt(best_dist2), best_idx
    return _BruteNN(P_inf), "brute-force"

# -------------------------
def mse(a: np.ndarray) -> float:
    return float(np.mean(a * a))

def rmse(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="dataset.csv (GT humano / grabado)")
    ap.add_argument("--inf", required=True, help="infer_log_*.csv (inferencia)")
    ap.add_argument("--plot", action="store_true", help="mostrar gráficas")
    ap.add_argument("--thr_min", type=float, default=0.25, help="usar solo muestras REF con throttle > umbral (si existe)")
    ap.add_argument("--v_min", type=float, default=0.01, help="usar solo muestras REF con speed > umbral (si existe)")
    ap.add_argument("--speed_vmax", type=float, default=20.0, help="clip speed en REF para outliers")
    ap.add_argument("--max_pairs", type=int, default=0, help="limita nº de puntos REF (0 = sin límite)")
    args = ap.parse_args()

    df_ref = load_positions_csv(args.ref, "REF")
    df_inf = load_positions_csv(args.inf, "INF")

    # --- filtros SOLO en REF por throttle/speed  ---
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
    dist, idx = nn.query(P_ref)  # dist euclídea y índice en INF

    P_match = P_inf[idx]  # (N,3)
    dxyz = P_match - P_ref

    err_x = dxyz[:, 0]
    err_y = dxyz[:, 1]
    err_z = dxyz[:, 2]

    # Métricas
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
        "RMSE_dist": rmse(dist),   # RMSE de la distancia
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
    print(f"DIST MSE={stats['MSE_dist']:.6f}  RMSE={stats['RMSE_dist']:.6f} (m)")
    print(f"DIST mean={stats['mean_dist']:.3f}  p95={stats['p95_dist']:.3f}  max={stats['max_dist']:.3f} (m)")
    print("=======================================================================\n")

    # Si quieres guardar el emparejamiento para inspección
    out_pairs = pd.DataFrame({
        "ref_x": P_ref[:, 0], "ref_y": P_ref[:, 1], "ref_z": P_ref[:, 2],
        "inf_x": P_match[:, 0], "inf_y": P_match[:, 1], "inf_z": P_match[:, 2],
        "err_x": err_x, "err_y": err_y, "err_z": err_z,
        "dist_xyz": dist,
        "inf_index": idx
    })
    # out_pairs.to_csv("pairs_by_position.csv", index=False)

    if args.plot:
        # Trayectorias XY
        plt.figure()
        plt.plot(df_ref["x"], df_ref["y"], label="REF (humano)")
        plt.plot(df_inf["x"], df_inf["y"], label="INF (inferencia)")
        plt.axis("equal")
        plt.grid(True); plt.legend()
        plt.title("Trayectorias XY (sin alinear por tiempo)"); plt.xlabel("x"); plt.ylabel("y")

        # Distancia NN por muestra (en el orden del REF)
        plt.figure()
        plt.plot(dist, label="distancia NN (m)")
        plt.grid(True); plt.legend()
        plt.title("Distancia al punto más cercano en INF (por cada punto REF)"); plt.xlabel("índice REF"); plt.ylabel("m")

        # Errores por eje
        plt.figure()
        plt.plot(err_x, label="err_x")
        plt.plot(err_y, label="err_y")
        plt.plot(err_z, label="err_z")
        plt.grid(True); plt.legend()
        plt.title("Errores por eje (INF_nn - REF)"); plt.xlabel("índice REF"); plt.ylabel("m")

        plt.show()

if __name__ == "__main__":
    main()
