#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Helpers

def pick_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def mse(a: np.ndarray) -> float:
    return float(np.mean(a * a))

def rmse(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a)))

# -------------------------
# Load REF

def load_ref(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    xcol = pick_col(df, ["x","pos_x","X"])
    ycol = pick_col(df, ["y","pos_y","Y"])
    zcol = pick_col(df, ["z","pos_z","Z"])
    scol = pick_col(df, ["speed","speed_mps","v","vel"])

    df["x"] = pd.to_numeric(df[xcol], errors="coerce")
    df["y"] = pd.to_numeric(df[ycol], errors="coerce")
    df["z"] = 0.2 if zcol is None else pd.to_numeric(df[zcol], errors="coerce").fillna(0.2)
    df["v_mps"] = pd.to_numeric(df[scol], errors="coerce")
    df["estado"] = pd.to_numeric(df["estado"], errors="coerce")

    df = df.dropna(subset=["x","y","v_mps","estado"]).reset_index(drop=True)

    return df[["x","y","z","v_mps","estado"]]

# -------------------------
# Load INF

def load_inf(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    xcol = pick_col(df, ["x","pos_x","X"])
    ycol = pick_col(df, ["y","pos_y","Y"])
    zcol = pick_col(df, ["z","pos_z","Z"])
    scol = pick_col(df, ["speed_kmh","speed","v_kmh","vel_kmh"])
    steer_col = pick_col(df, ["steer","angle","steering"])

    df["x"] = pd.to_numeric(df[xcol], errors="coerce")
    df["y"] = pd.to_numeric(df[ycol], errors="coerce")
    df["z"] = 0.2 if zcol is None else pd.to_numeric(df[zcol], errors="coerce").fillna(0.2)
    df["v_mps"] = pd.to_numeric(df[scol], errors="coerce") / 3.6
    df["steer"] = pd.to_numeric(df[steer_col], errors="coerce")

    df = df.dropna(subset=["x","y","v_mps","steer"]).reset_index(drop=True)

    return df[["x","y","z","v_mps","steer"]]

# ------------------------------
# NN

def build_nn_index(P_inf: np.ndarray):
    from scipy.spatial import cKDTree
    tree = cKDTree(P_inf)

    class _NN:
        def query(self, Q):
            dist, idx = tree.query(Q, k=1, workers=-1)
            return dist.astype(np.float64), idx.astype(np.int64)

    return _NN()

# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--inf", required=True)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    df_ref = load_ref(args.ref)
    df_inf = load_inf(args.inf)

    P_ref = df_ref[["x","y","z"]].to_numpy(np.float64)
    P_inf = df_inf[["x","y","z"]].to_numpy(np.float64)

    v_ref = df_ref["v_mps"].to_numpy()
    v_inf = df_inf["v_mps"].to_numpy()

    estado_ref = df_ref["estado"].to_numpy()

    nn = build_nn_index(P_inf)
    _, idx = nn.query(P_ref)

    v_inf_match = v_inf[idx]
    err_v = v_inf_match - v_ref
    abs_err_v = np.abs(err_v)

    print("\n========== GLOBAL SPEED ==========")
    print(f"MSE={mse(err_v):.6f}")
    print(f"RMSE={rmse(err_v):.6f}")
    print("==================================\n")

    # -------------------------
    # Métricas por estado
  
    estados = [1,2,3]
    labels = ["Izquierda","Centro","Derecha"]

    mse_vals = []
    rmse_vals = []
    box_data = []

    print("========== SPEED BY STATE ==========")

    for e in estados:
        mask = estado_ref == e

        if np.sum(mask) == 0:
            mse_vals.append(0)
            rmse_vals.append(0)
            box_data.append([])
            continue

        err_e = err_v[mask]
        abs_e = np.abs(err_e)

        mse_vals.append(mse(err_e))
        rmse_vals.append(rmse(err_e))
        box_data.append(abs_e)

        print(f"\nEstado {e} ({labels[e-1]})")
        print(f"  MSE  = {mse(err_e):.6f}")
        print(f"  RMSE = {rmse(err_e):.6f}")
        print(f"  Mean abs = {np.mean(abs_e):.4f}")

    print("====================================\n")

    if not args.plot:
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    # -------------------------
    # Plot 1: MSE por estado
    # -------------------------
    fig1, ax1 = plt.subplots(figsize=(7,4), dpi=130)
    bars = ax1.bar(labels, mse_vals, color=["#E07A5F","#3D5A80","#81B29A"])
    ax1.set_title("MSE de velocidad por estado")
    ax1.set_ylabel("(m/s)^2")
    ax1.grid(True, axis="y", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x()+bar.get_width()/2, h, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=10)

    # -------------------------
    # Plot 2: RMSE por estado

    fig2, ax2 = plt.subplots(figsize=(7,4), dpi=130)
    bars2 = ax2.bar(labels, rmse_vals, color=["#E07A5F","#3D5A80","#81B29A"])
    ax2.set_title("RMSE de velocidad por estado")
    ax2.set_ylabel("m/s")
    ax2.grid(True, axis="y", alpha=0.3)

    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x()+bar.get_width()/2, h, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=10)

    # -------------------------
    # Plot 3: Boxplot error absoluto

    fig3, ax3 = plt.subplots(figsize=(7,4), dpi=130)
    ax3.boxplot(box_data, labels=labels, showfliers=False)
    ax3.set_title("Distribución del error absoluto por estado")
    ax3.set_ylabel("Error absoluto (m/s)")
    ax3.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
