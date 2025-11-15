#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==================
CSV_NO_SPEED = "PilotnetDefault/rmse_per_state_inference_no_speed.csv"
CSV_SPEED    = "PilotnetconNuevasEntradasValidationDir/rmse_per_state_inference_speed.csv"

SAVE_FIG_RMSE = "rmse_por_modelo_y_estado.png"
SAVE_FIG_MSE  = "mse_por_modelo_y_estado.png"
# ================

def main():
    # Leer ficheros
    df_no = pd.read_csv(CSV_NO_SPEED)
    df_sp = pd.read_csv(CSV_SPEED)

    # Esperamos columnas:
    # df_no: estado, pct_rmse_no_speed
    # df_sp: estado, pct_rmse_speed
    df_no = df_no[["estado", "pct_rmse_no_speed"]]
    df_sp = df_sp[["estado", "pct_rmse_speed"]]

    # Merge por 'estado'
    df = pd.merge(df_no, df_sp, on="estado", how="inner").sort_values("estado")

    # === Calcular MSE a partir del %RMSE ===
    # %RMSE = sqrt(MSE) * 100  =>  MSE = ( %RMSE / 100 )^2
    df["mse_no_speed"] = (df["pct_rmse_no_speed"] / 100.0) ** 2
    df["mse_speed"]    = (df["pct_rmse_speed"]    / 100.0) ** 2

    # Estados presentes
    estados = df["estado"].to_list()  # ej. [1,2,3]

    # Diccionarios estado -> %RMSE
    rmse_no_dict = dict(zip(df["estado"], df["pct_rmse_no_speed"]))
    rmse_sp_dict = dict(zip(df["estado"], df["pct_rmse_speed"]))

    # Diccionarios estado -> MSE
    mse_no_dict = dict(zip(df["estado"], df["mse_no_speed"]))
    mse_sp_dict = dict(zip(df["estado"], df["mse_speed"]))

    # X = modelos: 0 = sin velocidad, 1 = con velocidad
    modelos = ["Sin velocidad", "Con velocidad"]
    x = np.arange(len(modelos))  # [0,1]

    width = 0.2  # ancho de cada barra (tres barras por modelo)

    # PLOT 1: %Error
    fig, ax = plt.subplots(figsize=(7,4), dpi=120)

    # offsets para cada estado
    offsets = {
        estados[0]: -width,  # primer estado
        estados[1]:  0.0,    # segundo estado
        estados[2]:  width   # tercer estado
    }

    colors = {
        estados[0]: "tab:blue",
        estados[1]: "tab:orange",
        estados[2]: "tab:green",
    }

    for e in estados:
        heights = [
            rmse_no_dict.get(e, np.nan),  # sin velocidad
            rmse_sp_dict.get(e, np.nan),  # con velocidad
        ]
        ax.bar(
            x + offsets[e],
            heights,
            width,
            label=f"Estado {e}",
            color=colors.get(e, None)
        )

    ax.set_xticks(x)
    ax.set_xticklabels(modelos)
    ax.set_ylabel("Error % (steer+throttle)")
    ax.set_title("Error % por modelo y estado")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Estados")

    plt.tight_layout()
    plt.savefig(SAVE_FIG_RMSE)
    print(f"Figura %RMSE guardada en {SAVE_FIG_RMSE}")
    plt.show()

    # PLOT 2: MSE
    fig2, ax2 = plt.subplots(figsize=(7,4), dpi=120)

    for e in estados:
        heights_mse = [
            mse_no_dict.get(e, np.nan),  # sin velocidad
            mse_sp_dict.get(e, np.nan),  # con velocidad
        ]
        ax2.bar(
            x + offsets[e],
            heights_mse,
            width,
            label=f"Estado {e}",
            color=colors.get(e, None)
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(modelos)
    ax2.set_ylabel("MSE (steer+throttle)")
    ax2.set_title("MSE por modelo y estado")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(title="Estados")

    plt.tight_layout()
    plt.savefig(SAVE_FIG_MSE)
    print(f"Figura MSE guardada en {SAVE_FIG_MSE}")
    plt.show()


if __name__ == "__main__":
    main()
