#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

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

def robust_clip_speed_mps(s: pd.Series, vmax=20.0) -> pd.Series:
    return s.where((s >= 0.0) & (s <= vmax))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="dataset.csv (GT humano)")
    ap.add_argument("--inf", required=True, help="infer_log_*.csv (inferencia)")
    ap.add_argument("--plot", action="store_true", help="mostrar gráficas")
    ap.add_argument("--speed_vmax", type=float, default=20.0, help="máximo razonable de speed (m/s) para filtrar outliers")
    ap.add_argument("--max_match_dist", type=float, default=None,
                    help="si lo pones (m), descarta emparejamientos con distancia > este umbral")
    args = ap.parse_args()

    df_ref = load_csv(args.ref)
    df_inf = load_csv(args.inf)

    # ----------------------------
    # Detectar columnas REF
    # ----------------------------
    need_ref = ["timestamp", "steer", "throttle", "speed"]
    for c in need_ref:
        if c not in df_ref.columns:
            raise ValueError(f"[REF] falta columna '{c}'. Columnas REF: {list(df_ref.columns)}")

    xref = pick_col(df_ref, ["x", "pos_x", "X"])
    yref = pick_col(df_ref, ["y", "pos_y", "Y"])
    if xref is None or yref is None:
        raise ValueError(f"[REF] faltan columnas x/y. Columnas REF: {list(df_ref.columns)}")

    # ----------------------------
    # Detectar columnas INF
    # ----------------------------
    # Tiempo opcional, aquí NO lo usamos para alinear, pero lo dejamos por si quieres plot
    t_inf = pick_col(df_inf, ["t", "time", "timestamp", "sim_time", "ts"])

    if "speed_mps" not in df_inf.columns:
        if "speed" in df_inf.columns:
            df_inf = df_inf.rename(columns={"speed": "speed_mps"})
        else:
            raise ValueError(f"[INF] falta columna 'speed_mps' (o 'speed'). Columnas INF: {list(df_inf.columns)}")

    xinf = pick_col(df_inf, ["x", "pos_x", "X"])
    yinf = pick_col(df_inf, ["y", "pos_y", "Y"])
    if xinf is None or yinf is None:
        raise ValueError(f"[INF] faltan columnas x/y. Columnas INF: {list(df_inf.columns)}")

    # ----------------------------
    # Limpieza / numérico
    # ----------------------------
    df_ref = to_num(df_ref, ["timestamp", "steer", "throttle", "speed", xref, yref]).dropna(
        subset=["timestamp", "steer", "throttle", "speed", xref, yref]
    )
    df_inf = to_num(df_inf, ["steer", "throttle", "speed_mps", xinf, yinf] + ([t_inf] if t_inf else [])).dropna(
        subset=["steer", "throttle", "speed_mps", xinf, yinf]
    )

    df_ref = df_ref.sort_values("timestamp").reset_index(drop=True)
    # en INF no importa el orden para nearest, pero lo dejamos estable
    if t_inf:
        df_inf = df_inf.sort_values(t_inf).reset_index(drop=True)
    else:
        df_inf = df_inf.reset_index(drop=True)

    # Filtrar outliers speed en REF (y también en INF si quieres)
    df_ref["speed"] = robust_clip_speed_mps(df_ref["speed"], vmax=args.speed_vmax)
    df_inf["speed_mps"] = robust_clip_speed_mps(df_inf["speed_mps"], vmax=args.speed_vmax)
    df_ref = df_ref.dropna(subset=["speed"])
    df_inf = df_inf.dropna(subset=["speed_mps"])

    if len(df_ref) < 10 or len(df_inf) < 10:
        raise RuntimeError("Muy pocos puntos tras limpieza. Revisa tus CSV.")

    # ----------------------------
    # Emparejar por nearest (x,y)
    # ----------------------------
    ref_xy = df_ref[[xref, yref]].to_numpy(dtype=np.float64)          # (N,2)
    inf_xy = df_inf[[xinf, yinf]].to_numpy(dtype=np.float64)          # (M,2)

    # Para cada ref, distancia a todos los inf: (N,M)
    # Ojo: esto puede ser pesado si tienes MUCHOS puntos (>>50k). Si es tu caso, te paso versión KDTree.
    d2 = ((ref_xy[:, None, :] - inf_xy[None, :, :]) ** 2).sum(axis=2)  # squared distance
    nn_idx = np.argmin(d2, axis=1)
    nn_dist = np.sqrt(d2[np.arange(len(ref_xy)), nn_idx])

    pairs = pd.DataFrame({
        "timestamp_ref": df_ref["timestamp"].to_numpy(),
        "x_ref": ref_xy[:, 0],
        "y_ref": ref_xy[:, 1],
        "steer_ref": df_ref["steer"].to_numpy(),
        "throttle_ref": df_ref["throttle"].to_numpy(),
        "speed_ref": df_ref["speed"].to_numpy(),

        "idx_inf": nn_idx,
        "dist_xy": nn_dist,

        "x_inf": inf_xy[nn_idx, 0],
        "y_inf": inf_xy[nn_idx, 1],
        "steer_inf": df_inf["steer"].to_numpy()[nn_idx],
        "throttle_inf": df_inf["throttle"].to_numpy()[nn_idx],
        "speed_inf": df_inf["speed_mps"].to_numpy()[nn_idx],
    })

    if t_inf:
        pairs["t_inf"] = df_inf[t_inf].to_numpy()[nn_idx]

    # (Opcional) filtrar emparejamientos lejanos
    if args.max_match_dist is not None:
        before = len(pairs)
        pairs = pairs[pairs["dist_xy"] <= float(args.max_match_dist)].reset_index(drop=True)
        after = len(pairs)
        if after < 10:
            raise RuntimeError(f"Tras filtrar por max_match_dist quedaron muy pocos pares ({after}/{before}).")

    # ----------------------------
    # Métricas (MSE)
    # ----------------------------
    pairs["err_steer"] = pairs["steer_inf"] - pairs["steer_ref"]
    pairs["err_throttle"] = pairs["throttle_inf"] - pairs["throttle_ref"]
    pairs["err_speed"] = pairs["speed_inf"] - pairs["speed_ref"]

    def mse(x): return float(np.mean(np.square(x)))
    def rmse(x): return float(np.sqrt(np.mean(np.square(x))))

    stats = {
        "N_pairs": len(pairs),
        "MSE_dist_xy": mse(pairs["dist_xy"]),
        "RMSE_dist_xy": rmse(pairs["dist_xy"]),
        "MSE_steer": mse(pairs["err_steer"]),
        "RMSE_steer": rmse(pairs["err_steer"]),
        "MSE_throttle": mse(pairs["err_throttle"]),
        "RMSE_throttle": rmse(pairs["err_throttle"]),
        "MSE_speed": mse(pairs["err_speed"]),
        "RMSE_speed": rmse(pairs["err_speed"]),
        "MAE_speed": float(np.mean(np.abs(pairs["err_speed"]))),
        "MAX_abs_speed": float(np.max(np.abs(pairs["err_speed"]))),
        "AVG_match_dist": float(np.mean(pairs["dist_xy"])),
        "P95_match_dist": float(np.percentile(pairs["dist_xy"], 95)),
    }

    print("\n========== COMPARACIÓN (emparejado por nearest en XY) ==========")
    print(f"N pares: {stats['N_pairs']}")
    print(f"DIST_XY  MSE={stats['MSE_dist_xy']:.6f}  RMSE={stats['RMSE_dist_xy']:.6f}  AVG={stats['AVG_match_dist']:.3f}m  P95={stats['P95_match_dist']:.3f}m")
    print(f"STEER    MSE={stats['MSE_steer']:.6f}   RMSE={stats['RMSE_steer']:.6f}")
    print(f"THROTT   MSE={stats['MSE_throttle']:.6f} RMSE={stats['RMSE_throttle']:.6f}")
    print(f"SPEED    MSE={stats['MSE_speed']:.6f}   RMSE={stats['RMSE_speed']:.6f}  (m/s)")
    print(f"SPEED    MAE={stats['MAE_speed']:.6f}   MAX|err|={stats['MAX_abs_speed']:.6f} (m/s)")
    print("===============================================================\n")

    # ----------------------------
    # Plots (opcional)
    # ----------------------------
    if args.plot:
        # Para graficar, usamos el tiempo del ref (orden natural)
        t = pairs["timestamp_ref"].to_numpy()

        plt.figure()
        plt.plot(t, pairs["dist_xy"], label="dist_xy (m)")
        plt.grid(True); plt.legend(); plt.title("Distancia espacial (GT -> INF nearest)"); plt.xlabel("timestamp_ref"); plt.ylabel("m")

        plt.figure()
        plt.plot(t, pairs["speed_ref"], label="speed_ref (GT)")
        plt.plot(t, pairs["speed_inf"], label="speed_inf (paired)")
        plt.grid(True); plt.legend(); plt.title("Velocidad (m/s)"); plt.xlabel("timestamp_ref"); plt.ylabel("m/s")

        plt.figure()
        plt.plot(t, pairs["err_speed"], label="err_speed (inf-ref)")
        plt.grid(True); plt.legend(); plt.title("Error velocidad (m/s)"); plt.xlabel("timestamp_ref"); plt.ylabel("m/s")

        plt.figure()
        plt.plot(t, pairs["err_steer"], label="err_steer (inf-ref)")
        plt.grid(True); plt.legend(); plt.title("Error steer"); plt.xlabel("timestamp_ref"); plt.ylabel("steer")

        plt.figure()
        plt.plot(t, pairs["err_throttle"], label="err_throttle (inf-ref)")
        plt.grid(True); plt.legend(); plt.title("Error throttle"); plt.xlabel("timestamp_ref"); plt.ylabel("throttle")

        plt.show()

if __name__ == "__main__":
    main()
