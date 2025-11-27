#!/usr/bin/env python3

import os
import glob
import argparse
import shutil
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description=(
            " Global balanced dataset based on the minimum state:"
            "1) In each dataset, throttle distribution is edited to try and have same amount of each data values."
            "2) Search lowest value (state,dataset) among states 1/2/3
            "3) Rewrite each dataset"
        )
    )
    parser.add_argument(
        "--pattern",
        default="../datasets/Deepracer_BaseMap_*/dataset.csv",
        help="Patrón de búsqueda de CSV (por defecto: ../datasets/validation/Deepracer_BaseMap_*/dataset.csv)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla para muestreo reproducible.")
    parser.add_argument("--dry-run", action="store_true", help="No escribe cambios, solo muestra lo que haría.")
    parser.add_argument("--no-backup", action="store_true", help="No crear .bak antes de sobrescribir.")
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(args.pattern))
    if not csv_paths:
        print("No se encontraron CSV con el patrón dado.")
        return
 
    file_info = {} 
    global_min = None  

    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[SKIP] No se pudo leer {p}: {e}")
            continue

        if "estado" not in df.columns:
            print(f"[SKIP] {p} no tiene columna 'estado'.")
            continue

  
        estado = pd.to_numeric(df["estado"], errors="coerce").astype("Int64")
        mask_123 = estado.isin([1, 2, 3])

        df_123  = df[mask_123].copy()
        df_rest = df[~mask_123].copy()

        
        counts_local = {c: int((estado[mask_123] == c).sum()) for c in [1, 2, 3]}
        print(f"[{os.path.basename(p)}] Conteos locales (1/2/3): {counts_local[1]}/{counts_local[2]}/{counts_local[3]}")

      
        for c in [1, 2, 3]:
            cnt = counts_local[c]
            if cnt > 0:
                if global_min is None or cnt < global_min:
                    global_min = cnt

        file_info[p] = {
            "df_123": df_123,
            "df_rest": df_rest,
            "counts": counts_local
        }

    if not file_info:
        print("No hay datasets válidos con columna 'estado'.")
        return

    if global_min is None or global_min <= 0:
        print("No hay ningún estado 1/2/3 con conteo > 0 en los datasets. No se balancea.")
        return

    print(f"\n== Valor GLOBAL mínimo entre todos los estados 1/2/3 de todos los datasets: {global_min} ==")

    if args.dry_run:
        print("[DRY-RUN] Solo se mostrará lo que se HARÍA, sin escribir en disco.\n")

  
    for p in csv_paths:
        if p not in file_info:

            continue

        info = file_info[p]
        df_123  = info["df_123"]
        df_rest = info["df_rest"]

        if df_123.empty:
            
            df_out = df_rest.copy()
            print(f"[{os.path.basename(p)}] No tiene filas con estados 1/2/3. Se deja solo 'resto' (total={len(df_out)}).")
        else:
            estado_123 = pd.to_numeric(df_123["estado"], errors="coerce").astype("Int64")

        
            sampled_list = []
            final_counts = {1: 0, 2: 0, 3: 0}

            for c in [1, 2, 3]:
                df_c = df_123[estado_123 == c]
                n_c = len(df_c)
                if n_c == 0:
                    
                    continue

                if n_c > global_min:
                    df_c = df_c.sample(n=global_min, random_state=args.seed)
                    final_counts[c] = global_min
                else:
                   
                    final_counts[c] = n_c

                sampled_list.append(df_c)

            if sampled_list:
                df_bal = pd.concat(sampled_list, ignore_index=True)
            else:
                df_bal = pd.DataFrame(columns=df_123.columns)

          
            df_out = pd.concat([df_bal, df_rest], ignore_index=True)

           
            df_out = df_out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

            print(f"[{os.path.basename(p)}] Después del balanceo local "
                  f"(1/2/3) -> {final_counts[1]}/{final_counts[2]}/{final_counts[3]} "
                  f"| total con resto = {len(df_out)}")

        if args.dry_run:
            continue

        try:
            if not args.no_backup and os.path.isfile(p):
                shutil.copy2(p, p + ".bak")
                print(f"  Copia de seguridad creada: {p}.bak")
        except Exception as e:
            print(f"  [WARN] No se pudo crear .bak para {p}: {e}")

        try:
            df_out.to_csv(p, index=False)
            print(f"  Guardado {p}")
        except Exception as e:
            print(f"  [ERROR] Guardando {p}: {e}")

if __name__ == "__main__":
    main()
