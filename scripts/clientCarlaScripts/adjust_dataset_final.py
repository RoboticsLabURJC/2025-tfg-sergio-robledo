#!/usr/bin/env python3

import os
import glob
import argparse
import shutil
import pandas as pd
import numpy as np

TARGET_TRAIN = 21465
VAL_PERCENT = 0.15
TARGET_VAL = int(TARGET_TRAIN * VAL_PERCENT)


def main():
    parser = argparse.ArgumentParser(
        description="Balance validation directory per circuit and per state (1/2/3)"
    )
    parser.add_argument("--valdir", required=True,
                        help="Directorio validation (ej: ../datasets/validation)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    pattern = os.path.join(args.valdir, "Deepracer_BaseMap_*", "dataset.csv")
    csv_paths = sorted(glob.glob(pattern))

    if not csv_paths:
        print("No se encontraron dataset.csv en el directorio.")
        return

    print(f"\nCircuitos encontrados: {len(csv_paths)}")
    print(f"Objetivo total validation: {TARGET_VAL}\n")

    # Calcular target por circuito

    n_circuits = len(csv_paths)
    target_per_circuit = TARGET_VAL // n_circuits
    target_per_state = target_per_circuit // 3

    print(f"Objetivo por circuito: {target_per_circuit}")
    print(f"Objetivo por estado dentro de circuito: {target_per_state}\n")


    # Primero verificar mínimos reales
 
    min_available_state = None

    circuit_data = {}

    for p in csv_paths:
        df = pd.read_csv(p)

        if "estado" not in df.columns:
            print(f"{p} no tiene columna estado")
            continue

        df["estado"] = pd.to_numeric(df["estado"], errors="coerce")
        df = df[df["estado"].isin([1,2,3])]

        counts = df["estado"].value_counts().to_dict()

        for c in [1,2,3]:
            counts.setdefault(c, 0)

        print(f"{os.path.basename(os.path.dirname(p))} -> {counts}")

        local_min = min(counts[1], counts[2], counts[3])

        if min_available_state is None:
            min_available_state = local_min
        else:
            min_available_state = min(min_available_state, local_min)

        circuit_data[p] = df

    if min_available_state is None or min_available_state == 0:
        print("No hay suficientes datos en algún estado.")
        return

    # Ajustar target real por limitación de datos
    final_per_state = min(target_per_state, min_available_state)

    print(f"\nEstado máximo posible común: {min_available_state}")
    print(f"Estado final usado: {final_per_state}\n")

    total_final = final_per_state * 3 * n_circuits
    print(f"Total final validation real: {total_final}\n")


    # Balancear cada circuito

    for p, df in circuit_data.items():

        sampled_list = []

        for c in [1,2,3]:
            df_c = df[df["estado"] == c]
            df_c_sampled = df_c.sample(
                n=final_per_state,
                random_state=args.seed
            )
            sampled_list.append(df_c_sampled)

        df_out = pd.concat(sampled_list, ignore_index=True)
        df_out = df_out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        print(f"{os.path.basename(os.path.dirname(p))} -> "
              f"{final_per_state}/{final_per_state}/{final_per_state} "
              f"= {len(df_out)}")

        if args.dry_run:
            continue

        # Backup
        if not args.no_backup and os.path.isfile(p):
            shutil.copy2(p, p + ".bak")

        df_out.to_csv(p, index=False)

    print("\nBalanceo completado correctamente.")


if __name__ == "__main__":
    main()
