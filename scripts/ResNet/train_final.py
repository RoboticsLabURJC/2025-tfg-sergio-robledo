import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from utils.processing import *
from utils.res_net_dataset import ResNetDataset
from utils.resnet import ResNet
from utils.transform_helper import createTransform
import argparse
import json
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import csv
from torch.utils.data import Subset
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset


def steer_to_estado(steer, thr=0.20):
    # steer: tensor (B,)
    estado = torch.full_like(steer, 2, dtype=torch.long)   # centro por defecto
    estado[steer < -thr] = 1
    estado[steer >  thr] = 3
    return estado

@torch.no_grad()
def eval_estado_accuracy(model, loader, device, thr=0.20):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device).float()
        labels = labels.to(device).float()

        outputs = model(images)             # (B,2)
        gt_estado = steer_to_estado(labels[:, 0], thr=thr)
        pr_estado = steer_to_estado(outputs[:, 0], thr=thr)

        correct += (pr_estado == gt_estado).sum().item()
        total += labels.size(0)

    return 100.0 * correct / max(total, 1)



def r2_from_batches(y_true_list, y_pred_list):

    # Entradas: 
    # - y_true_list: lista de tensores con las etiquetas de test 
    #     (cada tensor es un batch: (batch_size, 2)).
    # - y_pred_list: lista de tensores con las predicciones del modelo
    #      (mismas dimensiones).

    # Calcula R² para steer y throttle a partir de listas de tensores (batches).
    # Devuelve un dict con:
    #   - "mean": R² medio de steer y throttle
    #   - "steer": R² sólo para steer
    #   - "throttle": R² sólo para throttle

    # Concatenar todos los batches: (N, 2)
    y_true = torch.cat(y_true_list, dim=0)  # labels
    y_pred = torch.cat(y_pred_list, dim=0)  # outputs

    y_true = y_true.float()
    y_pred = y_pred.float()

    # (SumSquares) SS_res y SS_tot por componente (col 0 = steer, col 1 = throttle)
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)      # shape (2,)
    y_mean = torch.mean(y_true, dim=0)                     # shape (2,)
    ss_tot = torch.sum((y_true - y_mean) ** 2, dim=0)      # shape (2,)

    eps = 1e-8 # evitar divisiones por 0
    r2_vec = 1.0 - ss_res / (ss_tot + eps)                 # shape (2,)

    r2_steer    = float(r2_vec[0].item())
    r2_throttle = float(r2_vec[1].item())
    r2_mean     = float(r2_vec.mean().item())

    return {
        "mean":     r2_mean,
        "steer":    r2_steer,
        "throttle": r2_throttle,
    }

def mse_to_rmse(m):
    """Convierte un MSE escalar en RMSE"""
    return float(m) ** 0.5

def mse_to_pct_rmse(m):
    """Convierte un MSE escalar en %RMSE."""
    return mse_to_rmse(m) * 100.0


def mse_dict_to_percent_rmse(mse_dict):
    # Convierte cada MSE en %RMSE = sqrt(MSE) * 100
    return {k: mse_to_pct_rmse(v) for k, v in mse_dict.items()}


def make_bar_figure_percent(values_dict, title="Summary Error (%)"):
    fig, ax = plt.subplots(figsize=(4,3), dpi=120)
    labels = list(values_dict.keys())
    vals   = [values_dict[k] for k in labels]
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylabel("Error (%)")
    ax.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}%", ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    return fig

def make_bar_figure(values_dict, title="Summary", ylabel="Metric", ylim=None):
    fig, ax = plt.subplots(figsize=(4,3), dpi=120)
    labels = list(values_dict.keys())
    vals   = [values_dict[k] for k in labels]
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    return fig



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', required=True, help="Directory(ies) for Train Data")
    parser.add_argument("--test_dir", action='append', help="Directory(ies) for Test Data")
    parser.add_argument("--val_dir", action='append', required=True, help="Directory(ies) for Validation Data")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Proporción para validación (por defecto 0.2 = 20%)")
    parser.add_argument("--preprocess", action='append', default=None,
                        help="preprocessing info: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--base_dir", type=str, default='/home/sergior/Downloads/pruebas', help="Where to save outputs")
    parser.add_argument("--comment", type=str, default='No augs / no shuffle / no mirror', help="Experiment comment")

    # Hparams
    parser.add_argument("--num_epochs", type=int, default=80, help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--early_stop_patience", type=int, default=8,
                    help="Número de epochs sin mejorar antes de parar")
    parser.add_argument("--early_stop_min_delta", type=float, default=5e-5,
                    help="Mejora mínima requerida en val_mse para resetear paciencia")

    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--save_iter", type=int, default=50, help="Iterations between saves")
    parser.add_argument("--print_terminal", action="store_true", help="Print progress every 10 steps")
    parser.add_argument("--seed", type=int, default=123, help="Seed")

    # --- Label-only augmentation (expande dataset sin tocar imágenes) ---
    parser.add_argument("--label_aug", action="store_true",
                        help="Activa augmentación SOLO en labels (steer/throttle) y expande el dataset.")
    parser.add_argument("--label_aug_ratio", type=float, default=0.8,
                        help="Ratio extra de expansión. 0.8 = +80%% (dataset total = 1.8x).")
    parser.add_argument("--steer_aug_std", type=float, default=0.02,
                        help="STD del ruido gaussiano aplicado a steer en muestras augmentadas.")
    parser.add_argument("--throttle_aug_std", type=float, default=0.02,
                        help="STD del ruido gaussiano aplicado a throttle en muestras augmentadas.")
    parser.add_argument("--label_aug_in_val", action="store_true",
                        help="Si se activa, también aplica label-aug en validation (no recomendado normalmente).")


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    print("TRAIN dirs:", args.data_dir)
    print("TEST dirs :", args.test_dir)
    print("VAL   dirs:", args.val_dir)

    # Convierte args en dict y guarda
    exp_setup = vars(args)

    # Rutas
    base_dir = os.path.join('experiments', args.base_dir)
    model_save_dir = os.path.join(base_dir, 'trained_models')
    log_dir = os.path.join(base_dir, 'log')
    check_path(base_dir); check_path(log_dir); check_path(model_save_dir)

    print("Saving model in:", model_save_dir)

    with open(os.path.join(base_dir, 'args.json'), 'w') as fp:
        json.dump(exp_setup, fp)

    # Hparams
    num_epochs   = args.num_epochs
    batch_size   = args.batch_size
    learning_rate= args.lr
    save_iter    = args.save_iter
    random_seed  = args.seed
    print_terminal = args.print_terminal

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Usando dispositivo: {dev_name}")

    FLOAT = torch.FloatTensor
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # TensorBoard + CSV
    writer = SummaryWriter(log_dir)
    self_path = os.getcwd()
    csv_log_path = os.path.join(self_path, "last_train_data.csv")
    writer_output = csv.writer(open(csv_log_path, "w"))
    writer_output.writerow(["epoch", "val_mse", "val_mae"])

    # ===== Transform común para todo (train/val/test/inferencia) =====
    COMMON_TF = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = ResNetDataset(
        args.data_dir,
        mirrored=False,
        transform=COMMON_TF,
        preprocessing=args.preprocess
    )
    val_dataset = ResNetDataset(
        args.val_dir,
        mirrored=False,
        transform=COMMON_TF,
        preprocessing=args.preprocess
    )

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    #layers=[2,2,2,2] es de resnet 18
    #layers=[3,4,6,3] es de resnet 34
    image_shape = (224, 224, 4)
    model = ResNet(layers=[2,2,2,2], in_channels=image_shape[2], num_outputs=2).to(device)

    #print("DEBUG -> model.num_channels:", model.num_channels)          # debe ser 4
    #print("DEBUG -> ln_1.num_features :", model.ln_1.num_features)     # debe ser 4

        
    ckpt_latest = os.path.join(model_save_dir, f'pilot_net_model_{random_seed}.pth')
    if os.path.isfile(ckpt_latest):
        model.load_state_dict(torch.load(ckpt_latest, map_location=device))
        best_model = deepcopy(model)
        args_json_path = os.path.join(model_save_dir, 'args.json')
        last_epoch = json.load(open(args_json_path))['last_epoch'] + 1 if os.path.isfile(args_json_path) else 0
        print(f"Reanudando desde epoch {last_epoch}")
    else:
        best_model = deepcopy(model)
        last_epoch = 0

    # Pérdidas y optimizador
    criterion_train = nn.MSELoss()
    criterion_mse   = nn.MSELoss()
    criterion_mae   = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    global_iter = 0
    global_val_mse = float('inf')

    #---Early stopping-----
    patience = args.early_stop_patience
    min_delta = args.early_stop_min_delta
    epochs_no_improve = 0
    #----------------------

    best_epoch = None
    best_train_loss = None
    best_val_mse = None

    print("*********** Training Started ************")
    for epoch in range(last_epoch, num_epochs):
        model.train()
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            if i == 0:
                print("Batch shape:", images.shape)

            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)

            outputs = model(images)
            loss = criterion_train(outputs, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_iter % save_iter == 0:
                torch.save(model.state_dict(), ckpt_latest)
            global_iter += 1

            if print_terminal and (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

        # Persistir epoch
        with open(os.path.join(model_save_dir, 'args.json'), 'w') as fp:
            json.dump({'last_epoch': epoch}, fp)

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("performance/train_loss", train_loss/len(train_loader), epoch+1)

        # ===== Validation =====
        model.eval()
        val_mse = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = FLOAT(images).to(device)
                labels = FLOAT(labels.float()).to(device)
                outputs = model(images)
                val_mse += criterion_mse(outputs, labels).item()
                val_mae += criterion_mae(outputs, labels).item()

        val_mse /= len(val_loader)
        val_mae /= len(val_loader)

        # ==== Escalar todo a RMSE y %RMSE para curvas ====
        train_rmse      = mse_to_rmse(avg_train_loss)
        val_rmse        = mse_to_rmse(val_mse)
        train_pct_rmse  = mse_to_pct_rmse(avg_train_loss)
        val_pct_rmse    = mse_to_pct_rmse(val_mse)

        # Curvas en TensorBoard MISMA ESCALA
        writer.add_scalar("performance/train_rmse",     train_rmse,     epoch+1)
        writer.add_scalar("performance/valid_rmse",     val_rmse,       epoch+1)
        writer.add_scalar("performance/train_pct_rmse", train_pct_rmse, epoch+1)
        writer.add_scalar("performance/valid_pct_rmse", val_pct_rmse,   epoch+1)

        writer_output.writerow([epoch+1, val_mse, val_mae])

        if val_mse < global_val_mse:
            global_val_mse = val_mse
            best_model = deepcopy(model)
            epochs_no_improve = 0
            torch.save(best_model.state_dict(), os.path.join(model_save_dir, f'pilot_net_model_best_{random_seed}.pth'))
            mssg = "Model Improved!!"
            best_epoch = epoch + 1
            best_train_loss = avg_train_loss
            best_val_mse = val_mse

        else:
            epochs_no_improve += 1
            mssg = f"Not Improved!! ({epochs_no_improve}/{patience})"

        # ---- EARLY STOPPING ----
        if epochs_no_improve >= patience:
            print(f"\n[EARLY STOP] No mejora en {patience} epochs. Parando entrenamiento.")
            break

        print(f'Epoch [{epoch+1}/{num_epochs}]  Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f}  {mssg}')

        

        # MSE en barras (epoch actual)
        fig_epoch_bars = make_bar_figure(
            {"Train(Loss)": avg_train_loss, "Val(MSE)": val_mse},
            title=f"Epoch {epoch+1} - Train vs Val",
            ylabel="Loss / MSE"
        )
        writer.add_figure("bars/train_val_epoch_mse", fig_epoch_bars, global_step=epoch+1)
        plt.close(fig_epoch_bars)

        # RMSE en barras en %
        percent_rmse_dict = mse_dict_to_percent_rmse({"Train": avg_train_loss, "Val": val_mse})

        fig_epoch_bars_pct = make_bar_figure_percent(percent_rmse_dict, 
            title=f"Epoch {epoch+1} - %Error Train vs Val")

        writer.add_figure("bars/train_val_epoch_percent_rmse", 
        fig_epoch_bars_pct, global_step=epoch+1)
        plt.close(fig_epoch_bars_pct)


    model = best_model

    # ======= VALIDATION (best model) =======
    print("Check performance on validation (best model)")
    model.eval()

    val_mse_eval = 0.0
    val_mae_eval = 0.0
    val_mse_steer_eval = 0.0
    val_mae_steer_eval = 0.0
    val_mse_throttle_eval = 0.0
    val_mae_throttle_eval = 0.0

    val_y_true_batches = []
    val_y_pred_batches = []

    all_val_gt_steer      = []
    all_val_pred_steer    = []
    all_val_gt_throttle   = []
    all_val_pred_throttle = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Valid Eval"):
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            outputs = model(images)

            # pérdidas
            val_mse_eval += criterion_mse(outputs, labels).item()
            val_mae_eval += criterion_mae(outputs, labels).item()

            val_mse_steer_eval    += criterion_mse(outputs[:, 0], labels[:, 0]).item()
            val_mae_steer_eval    += criterion_mae(outputs[:, 0], labels[:, 0]).item()
            val_mse_throttle_eval += criterion_mse(outputs[:, 1], labels[:, 1]).item()
            val_mae_throttle_eval += criterion_mae(outputs[:, 1], labels[:, 1]).item()

            # para R²
            val_y_true_batches.append(labels.cpu())
            val_y_pred_batches.append(outputs.cpu())

            # para scatters
            all_val_gt_steer.extend(labels[:, 0].cpu().numpy())
            all_val_pred_steer.extend(outputs[:, 0].cpu().numpy())

            all_val_gt_throttle.extend(labels[:, 1].cpu().numpy())
            all_val_pred_throttle.extend(outputs[:, 1].cpu().numpy())

    n_val_eval = len(val_loader)
    if n_val_eval > 0:
        val_mse_eval        /= n_val_eval
        val_mae_eval        /= n_val_eval
        val_mse_steer_eval  /= n_val_eval
        val_mae_steer_eval  /= n_val_eval
        val_mse_throttle_eval /= n_val_eval
        val_mae_throttle_eval /= n_val_eval


    acc_val_estado = eval_estado_accuracy(model, val_loader, device, thr=0.20)
    print(f"[VAL] Estado-accuracy (thr=0.20): {acc_val_estado:.2f}%")

    # Escala RMSE y %RMSE para VALIDATION (best model)
    val_rmse_eval       = mse_to_rmse(val_mse_eval)
    val_pct_rmse_eval   = mse_to_pct_rmse(val_mse_eval)

    writer.add_scalar('performance/ValBest_RMSE',      val_rmse_eval)
    writer.add_scalar('performance/ValBest_pct_RMSE',  val_pct_rmse_eval)


    # R² en VALIDATION
    val_r2_dict = r2_from_batches(val_y_true_batches, val_y_pred_batches)
    val_r2_mean     = val_r2_dict["mean"]
    val_r2_steer    = val_r2_dict["steer"]
    val_r2_throttle = val_r2_dict["throttle"]

    # Scalars de validación best model
    # writer.add_scalar('performance/ValBest_MAE', val_mae_eval)
    # writer.add_scalar('performance/ValBest_MSE', val_mse_eval)
    # writer.add_scalar('performance/ValBest_MAE_steer', val_mae_steer_eval)
    # writer.add_scalar('performance/ValBest_MSE_steer', val_mse_steer_eval)
    # writer.add_scalar('performance/ValBest_MAE_throttle', val_mae_throttle_eval)
    # writer.add_scalar('performance/ValBest_MSE_throttle', val_mse_throttle_eval)

    # writer.add_scalar('performance/ValBest_R2_mean',     val_r2_mean)
    # writer.add_scalar('performance/ValBest_R2_steer',    val_r2_steer)
    # writer.add_scalar('performance/ValBest_R2_throttle', val_r2_throttle)

    # ---- Barras R² VALIDATION ----
    r2_dict_val = {
        "Val_R2_mean":     val_r2_mean,
        "Val_R2_steer":    val_r2_steer,
        "Val_R2_throttle": val_r2_throttle,
    }
    fig_val_r2 = make_bar_figure(r2_dict_val, title="Validation R² Summary (best model)", ylabel="R²",ylim=(0.0, 1.0))
    writer.add_figure("bars/val_r2", fig_val_r2)
    plt.close(fig_val_r2)

    # ===== Scatter GT vs Pred: STEER (VALIDATION) =====
    all_val_gt_steer      = np.array(all_val_gt_steer)
    all_val_pred_steer    = np.array(all_val_pred_steer)
    all_val_gt_throttle   = np.array(all_val_gt_throttle)
    all_val_pred_throttle = np.array(all_val_pred_throttle)

    # === Umbral basado en VALIDATION para THROTTLE ===
    val_err_throttle     = all_val_pred_throttle - all_val_gt_throttle       # error signed
    val_abs_err_throttle = np.abs(val_err_throttle)                          # |error|

    # Umbral: percentil 95 (ajusta 90/97/99 según veas)
    tau_throttle = np.percentile(val_abs_err_throttle, 95)
    print(f"[VAL] tau_throttle (percentil 95 abs err) = {tau_throttle:.4f}")


    # ===== Scatter GT vs Pred: STEER (VALIDATION) con banda de umbral =====
   
    # 1) Error y umbral
    val_err_steer     = all_val_pred_steer - all_val_gt_steer
    val_abs_err_steer = np.abs(val_err_steer)

    tau_steer_val = (val_mse_steer_eval ** 0.5) 
    inside_val_steer = val_abs_err_steer <= tau_steer_val
    above_val_steer  = val_err_steer >  tau_steer_val
    below_val_steer  = val_err_steer < -tau_steer_val

    fig_scatter_val_steer_band, ax_vsb = plt.subplots(figsize=(4,4), dpi=120)

    # Puntos dentro de la banda (verde)
    ax_vsb.scatter(
        all_val_gt_steer[inside_val_steer],
        all_val_pred_steer[inside_val_steer],
        alpha=0.3, s=5, c='green', label='inside band'
    )

    # Puntos fuera por arriba (naranja)
    ax_vsb.scatter(
        all_val_gt_steer[above_val_steer],
        all_val_pred_steer[above_val_steer],
        alpha=0.6, s=8, c='orange', label='above band'
    )

    # Puntos fuera por abajo (morado)
    ax_vsb.scatter(
        all_val_gt_steer[below_val_steer],
        all_val_pred_steer[below_val_steer],
        alpha=0.6, s=8, c='purple', label='below band'
    )

    # 2) Diagonal y banda: y = x ± tau_steer_val
    x_line = np.linspace(-1.0, 1.0, 100)
    ax_vsb.plot(x_line, x_line, 'r--', linewidth=1, label='y = x')
    ax_vsb.plot(x_line, x_line + tau_steer_val, 'g--', linewidth=1, label=f'y = x + τ ({tau_steer_val:.3f})')
    ax_vsb.plot(x_line, x_line - tau_steer_val, 'g--', linewidth=1, label=f'y = x - τ')

    ax_vsb.set_xlim(-1.05, 1.05)
    ax_vsb.set_ylim(-1.05, 1.05)
    ax_vsb.set_xlabel("Steer GT (Val)")
    ax_vsb.set_ylabel("Steer Pred (Val)")
    ax_vsb.set_title("GT vs Pred - Steer (Val, banded)")
    ax_vsb.grid(True, alpha=0.3)
    ax_vsb.legend(loc='best', fontsize=7)
    plt.tight_layout()

    writer.add_figure("scatter/val_steer_gt_vs_pred_banded", fig_scatter_val_steer_band)
    plt.close(fig_scatter_val_steer_band)


    # ===== Scatter GT vs Pred: THROTTLE (VALIDATION) con banda =====
    inside_val_thr = val_abs_err_throttle <= tau_throttle
    above_val_thr  = val_err_throttle >  tau_throttle
    below_val_thr  = val_err_throttle < -tau_throttle

    fig_scatter_val_th, ax_vt = plt.subplots(figsize=(4,4), dpi=120)

    # Dentro de la banda -> verde
    ax_vt.scatter(
        all_val_gt_throttle[inside_val_thr],
        all_val_pred_throttle[inside_val_thr],
        alpha=0.3, s=5, c="green", label="Dentro banda"
    )
    # Por encima -> naranja
    ax_vt.scatter(
        all_val_gt_throttle[above_val_thr],
        all_val_pred_throttle[above_val_thr],
        alpha=0.7, s=8, c="orange", label="Por encima banda"
    )
    # Por debajo -> morado
    ax_vt.scatter(
        all_val_gt_throttle[below_val_thr],
        all_val_pred_throttle[below_val_thr],
        alpha=0.7, s=8, c="purple", label="Por debajo banda"
    )

    x_line = np.linspace(0, 1, 200)
    ax_vt.plot(x_line, x_line, 'r--', linewidth=1, label="y = x")
    ax_vt.plot(x_line, x_line + tau_throttle, 'g--', linewidth=1,
               label=f"y = x + {tau_throttle:.2f}")
    ax_vt.plot(x_line, x_line - tau_throttle, 'g--', linewidth=1,
               label=f"y = x - {tau_throttle:.2f}")

    ax_vt.set_xlim(-0.05, 1.05)
    ax_vt.set_ylim(-0.05, 1.05)
    ax_vt.set_xlabel("Throttle GT (Val)")
    ax_vt.set_ylabel("Throttle Pred (Val)")
    ax_vt.set_title("GT vs Pred - Throttle (Validation, banda)")
    ax_vt.grid(True, alpha=0.3)
    ax_vt.legend(fontsize=7)

    plt.tight_layout()
    writer.add_figure("scatter/val_throttle_gt_vs_pred_banded", fig_scatter_val_th)
    plt.close(fig_scatter_val_th)




    # ======= TEST =======
    test_dirs = args.test_dir if args.test_dir is not None else args.data_dir[-1:]
    if args.test_dir is not None:
        overlap = set(test_dirs).intersection(set(args.data_dir))
        if overlap:
            print(f"[WARN] Estas carpetas están en train y test a la vez: {sorted(overlap)}")

    test_set = ResNetDataset(
        test_dirs,
        mirrored=False,
        transform=COMMON_TF,
        preprocessing=args.preprocess
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print("Check performance on testset")
    model.eval()
    test_mse = test_mae = 0.0
    test_mse_steer = test_mae_steer = 0.0
    test_mse_throttle = test_mae_throttle = 0.0

    # listas para R²
    test_y_true_batches = []
    test_y_pred_batches = []
    # Acumuladores para graficar GT vs Predicción
    all_gt_steer      = []
    all_pred_steer    = []
    all_gt_throttle   = []
    all_pred_throttle = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            outputs = model(images)

            test_mse += criterion_mse(outputs, labels).item()
            test_mae += criterion_mae(outputs, labels).item()

            test_mse_steer    += criterion_mse(outputs[:, 0], labels[:, 0]).item()
            test_mae_steer    += criterion_mae(outputs[:, 0], labels[:, 0]).item()
            test_mse_throttle += criterion_mse(outputs[:, 1], labels[:, 1]).item()
            test_mae_throttle += criterion_mae(outputs[:, 1], labels[:, 1]).item()

            #guardar para R²
            test_y_true_batches.append(labels.cpu())
            test_y_pred_batches.append(outputs.cpu())

            # ---- Guardar datos para el scatter GT vs Pred ----
            # labels: (batch, 2) -> [steer, throttle]
            all_gt_steer.extend(labels[:, 0].cpu().numpy())
            all_pred_steer.extend(outputs[:, 0].cpu().numpy())

            all_gt_throttle.extend(labels[:, 1].cpu().numpy())
            all_pred_throttle.extend(outputs[:, 1].cpu().numpy())

    n = len(test_loader)
    test_mse /= n; test_mae /= n
    test_mse_steer /= n; test_mae_steer /= n
    test_mse_throttle /= n; test_mae_throttle /= n


    acc_test_estado = eval_estado_accuracy(model, test_loader, device, thr=0.20)
    print(f"[TEST] Estado-accuracy (thr=0.20): {acc_test_estado:.2f}%")

    # Escala RMSE y %RMSE para TEST
    test_rmse      = mse_to_rmse(test_mse)
    test_pct_rmse  = mse_to_pct_rmse(test_mse)

    writer.add_scalar('performance/Test_RMSE',     test_rmse)
    writer.add_scalar('performance/Test_pct_RMSE', test_pct_rmse)

    # calcular R² en test
    test_r2_dict = r2_from_batches(test_y_true_batches, test_y_pred_batches)
    test_r2_mean     = test_r2_dict["mean"]
    test_r2_steer    = test_r2_dict["steer"]
    test_r2_throttle = test_r2_dict["throttle"]

    # writer.add_scalar('performance/Test_MAE', test_mae)
    # writer.add_scalar('performance/Test_MSE', test_mse)
    # writer.add_scalar('performance/Test_MAE_steer', test_mae_steer)
    # writer.add_scalar('performance/Test_MSE_steer', test_mse_steer)
    # writer.add_scalar('performance/Test_MAE_throttle', test_mae_throttle)
    # writer.add_scalar('performance/Test_MSE_throttle', test_mse_throttle)

    # R² en TensorBoard
    # writer.add_scalar('performance/Test_R2_mean',     test_r2_mean)
    # writer.add_scalar('performance/Test_R2_steer',    test_r2_steer)
    # writer.add_scalar('performance/Test_R2_throttle', test_r2_throttle)

    # ---- Elegir métricas de Train/Val para las figuras finales (best vs last) ----
    if best_train_loss is not None and best_val_mse is not None:
        train_for_plot = best_train_loss
        val_for_plot   = best_val_mse
        print(f"[INFO] Using BEST epoch {best_epoch} for final Train/Val metrics.")
    else:
        # Fallback por si no se actualizó nunca (caso muy raro)
        train_for_plot = avg_train_loss
        val_for_plot   = val_mse
        print("[WARN] best_* metrics not set, using LAST epoch for Train/Val metrics.")

    # MSE EN barras (usando BEST)
    fig_final_mse = make_bar_figure(
        {"Train(best)": train_for_plot, "Val(best)": val_for_plot, "Test": test_mse},
        title="Final MSE/Loss Summary (best model)",
        ylabel="MSE / Loss"
    )
    writer.add_figure("bars/final_mse", fig_final_mse, global_step=num_epochs)
    plt.close(fig_final_mse)

    # RMSE EN % en barras (usando BEST)
    final_pct_dict = mse_dict_to_percent_rmse(
        {"Train(best)": train_for_plot, "Val(best)": val_for_plot, "Test": test_mse}
    )
    fig_final_pct = make_bar_figure_percent(final_pct_dict, title="Final error Summary (best model)")
    writer.add_figure("bars/final_percent_rmse", fig_final_pct, global_step=num_epochs)
    plt.close(fig_final_pct)

    # R2 en barras
    r2_dict_final = {
        "R2_mean":     test_r2_mean,
        "R2_steer":    test_r2_steer,
        "R2_throttle": test_r2_throttle,
    }
    fig_r2 = make_bar_figure(r2_dict_final, title="Final R² Summary Test", ylabel="R²",ylim=(0.0, 1.0))
    writer.add_figure("bars/final_r2", fig_r2, global_step=num_epochs)
    plt.close(fig_r2)

    # Convertir a numpy
    all_gt_steer      = np.array(all_gt_steer)
    all_pred_steer    = np.array(all_pred_steer)
    all_gt_throttle   = np.array(all_gt_throttle)
    all_pred_throttle = np.array(all_pred_throttle)

    # === Clasificación según error de THROTTLE (TEST), usando tau_throttle de VALIDATION ===
    test_err_throttle     = all_pred_throttle - all_gt_throttle
    test_abs_err_throttle = np.abs(test_err_throttle)

    inside_thr = test_abs_err_throttle <= tau_throttle
    above_thr  = test_err_throttle >  tau_throttle
    below_thr  = test_err_throttle < -tau_throttle

    print(f"[TEST] Porcentaje dentro banda throttle: "
          f"{100.0 * inside_thr.mean():.1f}% "
          f"(tau_throttle={tau_throttle:.4f})")


    # ===== Scatter GT vs Pred: STEER (Test) coloreado según error THROTTLE =====
    fig_scatter_steer, ax_s = plt.subplots(figsize=(4,4), dpi=120)

    # Dentro de banda de throttle -> verde
    ax_s.scatter(
        all_gt_steer[inside_thr],
        all_pred_steer[inside_thr],
        alpha=0.3, s=5, c="green", label="Throttle dentro banda"
    )
    # Throttle por encima de banda -> naranja
    ax_s.scatter(
        all_gt_steer[above_thr],
        all_pred_steer[above_thr],
        alpha=0.7, s=8, c="orange", label="Throttle por encima banda"
    )
    # Throttle por debajo de banda -> morado
    ax_s.scatter(
        all_gt_steer[below_thr],
        all_pred_steer[below_thr],
        alpha=0.7, s=8, c="purple", label="Throttle por debajo banda"
    )

    # Sólo la diagonal de referencia (sin bandas verdes)
    ax_s.plot([-1, 1], [-1, 1], 'r--', linewidth=1, label="y = x")
    ax_s.set_xlim(-1.05, 1.05)
    ax_s.set_ylim(-1.05, 1.05)

    ax_s.set_xlabel("Steer GT")
    ax_s.set_ylabel("Steer Pred")
    ax_s.set_title("GT vs Pred - Steer (Test, coloreado por error throttle)")
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(fontsize=7)
    plt.tight_layout()

    writer.add_figure("scatter/test_steer_gt_vs_pred_colored_by_throttle", fig_scatter_steer)
    plt.close(fig_scatter_steer)


    # ===== Scatter GT vs Pred: THROTTLE (Test) con banda y colores =====
    fig_scatter_th, ax_t = plt.subplots(figsize=(4,4), dpi=120)

    ax_t.scatter(
        all_gt_throttle[inside_thr],
        all_pred_throttle[inside_thr],
        alpha=0.3, s=5, c="green", label="Dentro banda"
    )
    ax_t.scatter(
        all_gt_throttle[above_thr],
        all_pred_throttle[above_thr],
        alpha=0.7, s=8, c="orange", label="Por encima banda"
    )
    ax_t.scatter(
        all_gt_throttle[below_thr],
        all_pred_throttle[below_thr],
        alpha=0.7, s=8, c="purple", label="Por debajo banda"
    )

    x_line = np.linspace(0, 1, 200)
    ax_t.plot(x_line, x_line, 'r--', linewidth=1, label="y = x")
    ax_t.plot(x_line, x_line + tau_throttle, 'g--', linewidth=1,
              label=f"y = x + {tau_throttle:.2f}")
    ax_t.plot(x_line, x_line - tau_throttle, 'g--', linewidth=1,
              label=f"y = x - {tau_throttle:.2f}")

    ax_t.set_xlim(-0.05, 1.05)
    ax_t.set_ylim(-0.05, 1.05)
    ax_t.set_xlabel("Throttle GT")
    ax_t.set_ylabel("Throttle Pred")
    ax_t.set_title("GT vs Pred - Throttle (Test, banda)")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=7)
    plt.tight_layout()

    fig_scatter_th.savefig(os.path.join(base_dir, "scatter_throttle_test_banded.png"))
    writer.add_figure("scatter/test_throttle_gt_vs_pred_banded", fig_scatter_th)
    plt.close(fig_scatter_th)


    # Prints
    print(f"Test  -> MAE: {test_mae:.4f} | MSE: {test_mse:.4f}")
    print(f"Steer -> MAE: {test_mae_steer:.4f} | MSE: {test_mse_steer:.4f}")
    print(f"Throt -> MAE: {test_mae_throttle:.4f} | MSE: {test_mse_throttle:.4f}")
    print(f"Test R² -> mean: {test_r2_mean:.4f} | steer: {test_r2_steer:.4f} | throttle: {test_r2_throttle:.4f}")

    # ==== %RMSE finales en csv (usando BEST para Train/Val) ====
    train_pct_rmse_final = (train_for_plot ** 0.5) * 100.0
    val_pct_rmse_final   = (val_for_plot   ** 0.5) * 100.0
    test_pct_rmse_final  = (test_mse       ** 0.5) * 100.0

    final_csv = os.path.join(base_dir, "percent_rmse_speed_label.csv")
    with open(final_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_pct_rmse", "val_pct_rmse", "test_pct_rmse"])
        w.writerow([f"{train_pct_rmse_final:.6f}",
                    f"{val_pct_rmse_final:.6f}",
                    f"{test_pct_rmse_final:.6f}"])

    print(f"[OK] Guardado %Error final en: {final_csv}")


    # Save final + ONNX
    torch.save(model.state_dict(), os.path.join(model_save_dir, f'pilot_net_model_deepracer_{random_seed}.pth'))

    net_file_name = "mynet_deepracer_gpu.onnx" if torch.cuda.is_available() else "mynet_deepracer.onnx"
    dummy_input = torch.randn(1, 4, 224, 224, device=device)
    model = model.to(device)

    torch.onnx.export(
        model,
        dummy_input,
        net_file_name,
        verbose=True,
        export_params=True,
        opset_version=9,
        input_names=['input'],
        output_names=['output']
    )
