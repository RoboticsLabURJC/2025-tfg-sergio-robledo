#!/usr/bin/env python3
# ============================================================
# PilotNetDatasetWithTurnType + pesos por turn_class
# ============================================================

import os
import json
import csv
from copy import deepcopy
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.processing import check_path
from utils.pilotnet import PilotNet
from utils.pilot_net_dataset_with_turn_type import PilotNetDatasetWithTurnType


# ----------------------------------------------------------------------
# Pesos por clase usando IDs del dataset
# ----------------------------------------------------------------------

def print_turn_class_distribution(dataset, name="DATASET"):
    counts = Counter()
    for _, _, turn_id in dataset:
        cname = dataset.TURN_CLASSES[int(turn_id)]
        counts[cname] += 1

    total = sum(counts.values())

    print(f"\n[{name}] TURN_CLASS DISTRIBUTION")
    for k in dataset.TURN_CLASSES:
        v = counts.get(k, 0)
        pct = 100.0 * v / total if total > 0 else 0.0
        print(f"  {k:14s}: {v:6d}  ({pct:5.1f}%)")



def compute_turn_weights_from_dataset(dataset, num_classes, device, eps=1e-8):
    ids = np.array(dataset.turn_ids, dtype=np.int64)
    counts = np.bincount(ids, minlength=num_classes).astype(np.int64)

    counts_t = torch.tensor(counts, dtype=torch.float32)
    freq = counts_t / (counts_t.sum() + eps)

    w = torch.zeros_like(freq)
    nonzero = counts_t > 0
    w[nonzero] = 1.0 / (freq[nonzero] + eps)

    # normaliza SOLO con clases presentes
    w_mean = w[nonzero].mean().clamp_min(eps)
    w[nonzero] = w[nonzero] / w_mean

    w = w.to(device)

    class_names = dataset.TURN_CLASSES
    counts_dict  = {class_names[i]: int(counts[i]) for i in range(num_classes)}
    weights_dict = {class_names[i]: float(w[i].item()) for i in range(num_classes)}
    return w, weights_dict, counts_dict



def weighted_mse_by_turn_id(pred, target, turn_id, w_turn_tensor, w_label=None):
    """
    pred/target: (B,2)
    turn_id:     (B,)  -> {0..K-1}
    w_turn_tensor: (K,)
    w_label: (2,) para ponderar steer/throttle
    """
    mse_2 = (pred - target) ** 2  # (B,2)

    if w_label is not None:
        mse_2 = mse_2 * w_label.view(1, 2)

    cls_idx = turn_id.clamp(0, w_turn_tensor.numel() - 1)
    ws = w_turn_tensor[cls_idx]            # (B,)
    mse_2 = mse_2 * ws.view(-1, 1)         # (B,2)

    return mse_2.mean()


# Métricas y plots

def r2_from_batches(y_true_list, y_pred_list):
    y_true = torch.cat(y_true_list, dim=0).float()
    y_pred = torch.cat(y_pred_list, dim=0).float()

    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    y_mean = torch.mean(y_true, dim=0)
    ss_tot = torch.sum((y_true - y_mean) ** 2, dim=0)

    eps = 1e-8
    r2_vec = 1.0 - ss_res / (ss_tot + eps)

    return {
        "mean": float(r2_vec.mean().item()),
        "steer": float(r2_vec[0].item()),
        "throttle": float(r2_vec[1].item()),
    }


def mse_to_rmse(m): return float(m) ** 0.5
def mse_to_pct_rmse(m): return mse_to_rmse(m) * 100.0
def mse_dict_to_percent_rmse(mse_dict): return {k: mse_to_pct_rmse(v) for k, v in mse_dict.items()}


def make_bar_figure(values_dict, title="Summary", ylabel="Metric", ylim=None):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=120)
    labels = list(values_dict.keys())
    vals = [values_dict[k] for k in labels]
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def make_bar_figure_percent(values_dict, title="Summary Error (%)"):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=120)
    labels = list(values_dict.keys())
    vals = [values_dict[k] for k in labels]
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylabel("Error (%)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", action="append", required=True)
    p.add_argument("--val_dir", action="append", required=True)
    p.add_argument("--test_dir", action="append", default=None)

    p.add_argument("--base_dir", type=str, default="exp_turnclass_weighted_full")
    p.add_argument("--num_epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--save_iter", type=int, default=50)
    p.add_argument("--print_terminal", action="store_true")

    p.add_argument("--early_stop_patience", type=int, default=8)
    p.add_argument("--early_stop_min_delta", type=float, default=5e-5)

    # clase en CSV
    p.add_argument("--turn_col", type=str, default="turn_class")
    p.add_argument("--drop_unlabeled", action="store_true",
                   help="Si se activa, elimina UNLABELED del TRAIN.")

    # etiqueta steer/throttle
    p.add_argument("--w_steer", type=float, default=1.0)
    p.add_argument("--w_throttle", type=float, default=1.0)

    return p.parse_args()


# ----------------------------------------------------------------------
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Usando dispositivo: {dev_name}")

    # Rutas experimento
    base_dir = os.path.join("experiments", args.base_dir)
    model_save_dir = os.path.join(base_dir, "trained_models")
    log_dir = os.path.join(base_dir, "log")
    check_path(base_dir); check_path(model_save_dir); check_path(log_dir)

    with open(os.path.join(base_dir, "args.json"), "w") as fp:
        json.dump(vars(args), fp, indent=2)

    writer = SummaryWriter(log_dir)

    # CSV log simple
    csv_log_path = os.path.join(base_dir, "last_train_data.csv")
    with open(csv_log_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["epoch", "train_loss_weighted", "val_mse_unweighted", "val_mae_unweighted"])

    # Transform común
    COMMON_TF = transforms.Compose([
        transforms.Resize((66, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Datasets
    train_dataset = PilotNetDatasetWithTurnType(
        args.data_dir, mirrored=False, transform=COMMON_TF,
        turn_col=args.turn_col, drop_unlabeled=args.drop_unlabeled
    )
    val_dataset = PilotNetDatasetWithTurnType(
        args.val_dir, mirrored=False, transform=COMMON_TF,
        turn_col=args.turn_col, drop_unlabeled=False
    )

    test_dirs = args.test_dir if (args.test_dir is not None and len(args.test_dir) > 0) else None
    test_dataset = PilotNetDatasetWithTurnType(
        test_dirs, mirrored=False, transform=COMMON_TF,
        turn_col=args.turn_col, drop_unlabeled=False
    ) if test_dirs else None

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset) if test_dataset else 0}")
    print("TURN_CLASSES:", train_dataset.TURN_CLASSES)

    # print_turn_class_distribution(train_dataset, "TRAIN")
    # print_turn_class_distribution(val_dataset,   "VAL")


    # Loaders devuelven (images, labels, turn_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True) if test_dataset else None

    # Modelo
    model = PilotNet(image_shape=(66, 200, 4), num_labels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Pesos por clase (SOLO TRAIN)
    K = len(train_dataset.TURN_CLASSES)
    W_TURN, w_dict, c_dict = compute_turn_weights_from_dataset(train_dataset, num_classes=K, device=device)

    # ======= PESOS A MANO=======
    # OJO: orden exacto = train_dataset.TURN_CLASSES
    #
    # W_TURN = torch.tensor(
    #     [1.0, 2.5, 0.9, 1.8, 1.5, 0.8, 2.7, 0.0],
    #     dtype=torch.float32, device=device
    # )

    # w_dict = {train_dataset.TURN_CLASSES[i]: float(W_TURN[i].item())
    #         for i in range(len(train_dataset.TURN_CLASSES))}
    # ======================================================

    print("[turn_class] counts:", c_dict)
    print("[turn_class] weights:", {k: round(v, 3) for k, v in w_dict.items()})

    # Pesos steer/throttle
    W_LABEL = torch.tensor([args.w_steer, args.w_throttle], dtype=torch.float32, device=device)

    # Criterios UNWEIGHTED para métricas
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    # Checkpoints
    ckpt_latest = os.path.join(model_save_dir, f"pilot_net_model_{args.seed}.pth")
    best_path = os.path.join(model_save_dir, f"pilot_net_model_best_{args.seed}.pth")
    resume_path = os.path.join(model_save_dir, "resume.json")

    best_model = deepcopy(model)
    best_val_mse = float("inf")
    epochs_no_improve = 0
    patience = args.early_stop_patience
    min_delta = args.early_stop_min_delta

    last_epoch = 0
    if os.path.isfile(ckpt_latest):
        model.load_state_dict(torch.load(ckpt_latest, map_location=device))
        best_model = deepcopy(model)
        if os.path.isfile(resume_path):
            last_epoch = int(json.load(open(resume_path, "r"))["last_epoch"]) + 1
        print(f"Reanudando desde epoch {last_epoch}")

    global_iter = 0
    best_epoch = None
    best_train_loss = None
    best_val_mse_for_plot = None

    print("*********** Training Started ************")
    for epoch in range(last_epoch, args.num_epochs):
        model.train()
        train_loss = 0.0

        for i, (images, labels, turn_id) in enumerate(train_loader):
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).float()
            turn_id = turn_id.to(device, non_blocking=True).long()

            outputs = model(images)

            # LOSS ponderada por turn_id
            loss = weighted_mse_by_turn_id(outputs, labels, turn_id, W_TURN, w_label=W_LABEL)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if global_iter % args.save_iter == 0:
                torch.save(model.state_dict(), ckpt_latest)
            global_iter += 1

            if args.print_terminal and (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.6f}")

        avg_train = train_loss / max(1, len(train_loader))
        writer.add_scalar("performance/train_loss_weighted", avg_train, epoch + 1)

        with open(resume_path, "w") as fp:
            json.dump({"last_epoch": epoch}, fp)

        # -------- Validation --------
        model.eval()
        val_mse = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for images, labels, _turn_id in val_loader:
                images = images.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).float()

                out = model(images)
                val_mse += criterion_mse(out, labels).item()
                val_mae += criterion_mae(out, labels).item()

        val_mse /= max(1, len(val_loader))
        val_mae /= max(1, len(val_loader))

        writer.add_scalar("performance/val_mse_unweighted", val_mse, epoch + 1)
        writer.add_scalar("performance/val_mae_unweighted", val_mae, epoch + 1)

        writer.add_scalar("performance/train_rmse", mse_to_rmse(avg_train), epoch + 1)
        writer.add_scalar("performance/val_rmse", mse_to_rmse(val_mse), epoch + 1)
        writer.add_scalar("performance/train_pct_rmse", mse_to_pct_rmse(avg_train), epoch + 1)
        writer.add_scalar("performance/val_pct_rmse", mse_to_pct_rmse(val_mse), epoch + 1)

        with open(csv_log_path, "a", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow([epoch + 1, avg_train, val_mse, val_mae])

        improved = (best_val_mse - val_mse) > min_delta
        if improved:
            best_val_mse = val_mse
            best_model = deepcopy(model)
            torch.save(best_model.state_dict(), best_path)
            epochs_no_improve = 0
            best_epoch = epoch + 1
            best_train_loss = avg_train
            best_val_mse_for_plot = val_mse
            msg = "Model Improved!!"
        else:
            epochs_no_improve += 1
            msg = f"Not Improved!! ({epochs_no_improve}/{patience})"

        print(f"Epoch [{epoch+1}/{args.num_epochs}]  Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f}  {msg}")

        # Figuras epoch (bars)
        fig_epoch_bars = make_bar_figure(
            {"Train(Loss weighted)": avg_train, "Val(MSE unweighted)": val_mse},
            title=f"Epoch {epoch+1} - Train vs Val",
            ylabel="Loss / MSE",
        )
        writer.add_figure("bars/train_val_epoch_mse", fig_epoch_bars, global_step=epoch + 1)
        plt.close(fig_epoch_bars)

        percent_rmse_dict = mse_dict_to_percent_rmse({"Train": avg_train, "Val": val_mse})
        fig_epoch_bars_pct = make_bar_figure_percent(
            percent_rmse_dict, title=f"Epoch {epoch+1} - %Error Train vs Val"
        )
        writer.add_figure("bars/train_val_epoch_percent_rmse", fig_epoch_bars_pct, global_step=epoch + 1)
        plt.close(fig_epoch_bars_pct)

        if epochs_no_improve >= patience:
            print(f"\n[EARLY STOP] No mejora en {patience} epochs (min_delta={min_delta}). Parando.")
            break

    # usar best
    model = best_model
    model.eval()

    # ----------------------------------------------------
    # VALIDATION EVAL (best model) + R2 + Scatters
    # ----------------------------------------------------
    print("Check performance on validation (best model)")

    val_mse_eval = 0.0
    val_mae_eval = 0.0
    val_mse_steer_eval = 0.0
    val_mae_steer_eval = 0.0
    val_mse_throttle_eval = 0.0
    val_mae_throttle_eval = 0.0

    val_y_true_batches = []
    val_y_pred_batches = []

    all_val_gt_steer = []
    all_val_pred_steer = []
    all_val_gt_throttle = []
    all_val_pred_throttle = []
    

    with torch.no_grad():
        for images, labels, _turn_id in tqdm(val_loader, desc="Valid Eval"):
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).float()

            outputs = model(images)

            val_mse_eval += criterion_mse(outputs, labels).item()
            val_mae_eval += criterion_mae(outputs, labels).item()

            val_mse_steer_eval += criterion_mse(outputs[:, 0], labels[:, 0]).item()
            val_mae_steer_eval += criterion_mae(outputs[:, 0], labels[:, 0]).item()

            val_mse_throttle_eval += criterion_mse(outputs[:, 1], labels[:, 1]).item()
            val_mae_throttle_eval += criterion_mae(outputs[:, 1], labels[:, 1]).item()

            val_y_true_batches.append(labels.cpu())
            val_y_pred_batches.append(outputs.cpu())

            all_val_gt_steer.extend(labels[:, 0].detach().cpu().numpy())
            all_val_pred_steer.extend(outputs[:, 0].detach().cpu().numpy())
            all_val_gt_throttle.extend(labels[:, 1].detach().cpu().numpy())
            all_val_pred_throttle.extend(outputs[:, 1].detach().cpu().numpy())

    n_val_eval = max(1, len(val_loader))
    val_mse_eval /= n_val_eval
    val_mae_eval /= n_val_eval
    val_mse_steer_eval /= n_val_eval
    val_mae_steer_eval /= n_val_eval
    val_mse_throttle_eval /= n_val_eval
    val_mae_throttle_eval /= n_val_eval

    val_rmse_eval = mse_to_rmse(val_mse_eval)
    val_pct_rmse_eval = mse_to_pct_rmse(val_mse_eval)

    writer.add_scalar("performance/ValBest_RMSE", val_rmse_eval)
    writer.add_scalar("performance/ValBest_pct_RMSE", val_pct_rmse_eval)

    val_r2_dict = r2_from_batches(val_y_true_batches, val_y_pred_batches)
    val_r2_mean = val_r2_dict["mean"]
    val_r2_steer = val_r2_dict["steer"]
    val_r2_throttle = val_r2_dict["throttle"]

    fig_val_r2 = make_bar_figure(
        {"Val_R2_mean": val_r2_mean, "Val_R2_steer": val_r2_steer, "Val_R2_throttle": val_r2_throttle},
        title="Validation R² Summary (best model)",
        ylabel="R²",
        ylim=(0.0, 1.0),
    )
    writer.add_figure("bars/val_r2", fig_val_r2)
    plt.close(fig_val_r2)

    all_val_gt_steer = np.array(all_val_gt_steer)
    all_val_pred_steer = np.array(all_val_pred_steer)
    all_val_gt_throttle = np.array(all_val_gt_throttle)
    all_val_pred_throttle = np.array(all_val_pred_throttle)

    # tau_throttle: percentil 95 abs error (VAL)
    val_err_throttle = all_val_pred_throttle - all_val_gt_throttle
    val_abs_err_throttle = np.abs(val_err_throttle)
    tau_throttle = float(np.percentile(val_abs_err_throttle, 95))
    print(f"[VAL] tau_throttle (percentil 95 abs err) = {tau_throttle:.4f}")

    # steer band: tau = RMSE(steer) en val
    val_err_steer = all_val_pred_steer - all_val_gt_steer
    val_abs_err_steer = np.abs(val_err_steer)
    tau_steer_val = float(np.sqrt(val_mse_steer_eval))

    inside_val_steer = val_abs_err_steer <= tau_steer_val
    above_val_steer = val_err_steer > tau_steer_val
    below_val_steer = val_err_steer < -tau_steer_val

    fig_scatter_val_steer_band, ax_vsb = plt.subplots(figsize=(4, 4), dpi=120)
    ax_vsb.scatter(all_val_gt_steer[inside_val_steer], all_val_pred_steer[inside_val_steer],
                   alpha=0.3, s=5, c="green", label="inside band")
    ax_vsb.scatter(all_val_gt_steer[above_val_steer], all_val_pred_steer[above_val_steer],
                   alpha=0.6, s=8, c="orange", label="above band")
    ax_vsb.scatter(all_val_gt_steer[below_val_steer], all_val_pred_steer[below_val_steer],
                   alpha=0.6, s=8, c="purple", label="below band")

    x_line = np.linspace(-1.0, 1.0, 100)
    ax_vsb.plot(x_line, x_line, "r--", linewidth=1, label="y = x")
    ax_vsb.plot(x_line, x_line + tau_steer_val, "g--", linewidth=1, label=f"y = x + τ ({tau_steer_val:.3f})")
    ax_vsb.plot(x_line, x_line - tau_steer_val, "g--", linewidth=1, label="y = x - τ")

    ax_vsb.set_xlim(-1.05, 1.05)
    ax_vsb.set_ylim(-1.05, 1.05)
    ax_vsb.set_xlabel("Steer GT (Val)")
    ax_vsb.set_ylabel("Steer Pred (Val)")
    ax_vsb.set_title("GT vs Pred - Steer (Val, banded)")
    ax_vsb.grid(True, alpha=0.3)
    ax_vsb.legend(loc="best", fontsize=7)
    plt.tight_layout()
    writer.add_figure("scatter/val_steer_gt_vs_pred_banded", fig_scatter_val_steer_band)
    plt.close(fig_scatter_val_steer_band)

    # Throttle banded scatter (VAL)
    inside_val_thr = val_abs_err_throttle <= tau_throttle
    above_val_thr = val_err_throttle > tau_throttle
    below_val_thr = val_err_throttle < -tau_throttle

    fig_scatter_val_th, ax_vt = plt.subplots(figsize=(4, 4), dpi=120)
    ax_vt.scatter(all_val_gt_throttle[inside_val_thr], all_val_pred_throttle[inside_val_thr],
                  alpha=0.3, s=5, c="green", label="Dentro banda")
    ax_vt.scatter(all_val_gt_throttle[above_val_thr], all_val_pred_throttle[above_val_thr],
                  alpha=0.7, s=8, c="orange", label="Por encima banda")
    ax_vt.scatter(all_val_gt_throttle[below_val_thr], all_val_pred_throttle[below_val_thr],
                  alpha=0.7, s=8, c="purple", label="Por debajo banda")

    x_line = np.linspace(0.0, 1.0, 200)
    ax_vt.plot(x_line, x_line, "r--", linewidth=1, label="y = x")
    ax_vt.plot(x_line, x_line + tau_throttle, "g--", linewidth=1, label=f"y = x + {tau_throttle:.2f}")
    ax_vt.plot(x_line, x_line - tau_throttle, "g--", linewidth=1, label=f"y = x - {tau_throttle:.2f}")

    ax_vt.set_xlim(-0.05, 1.05)
    ax_vt.set_ylim(-0.05, 1.05)
    ax_vt.set_xlabel("Throttle GT (Val)")
    ax_vt.set_ylabel("Throttle Pred (Val)")
    ax_vt.set_title("GT vs Pred - Throttle (Val, banda)")
    ax_vt.grid(True, alpha=0.3)
    ax_vt.legend(fontsize=7)
    plt.tight_layout()
    writer.add_figure("scatter/val_throttle_gt_vs_pred_banded", fig_scatter_val_th)
    plt.close(fig_scatter_val_th)

    # ----------------------------------------------------
    # TEST EVAL
    # ----------------------------------------------------
    if test_loader is not None:
        print("Check performance on testset (best model)")

        test_mse = 0.0
        test_mae = 0.0
        test_mse_steer = 0.0
        test_mae_steer = 0.0
        test_mse_throttle = 0.0
        test_mae_throttle = 0.0

        test_y_true_batches = []
        test_y_pred_batches = []

        all_gt_steer = []
        all_pred_steer = []
        all_gt_throttle = []
        all_pred_throttle = []

        with torch.no_grad():
            for images, labels, _turn_id in tqdm(test_loader, desc="Test Eval"):
                images = images.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).float()

                outputs = model(images)

                test_mse += criterion_mse(outputs, labels).item()
                test_mae += criterion_mae(outputs, labels).item()

                test_mse_steer += criterion_mse(outputs[:, 0], labels[:, 0]).item()
                test_mae_steer += criterion_mae(outputs[:, 0], labels[:, 0]).item()

                test_mse_throttle += criterion_mse(outputs[:, 1], labels[:, 1]).item()
                test_mae_throttle += criterion_mae(outputs[:, 1], labels[:, 1]).item()

                test_y_true_batches.append(labels.cpu())
                test_y_pred_batches.append(outputs.cpu())

                all_gt_steer.extend(labels[:, 0].detach().cpu().numpy())
                all_pred_steer.extend(outputs[:, 0].detach().cpu().numpy())
                all_gt_throttle.extend(labels[:, 1].detach().cpu().numpy())
                all_pred_throttle.extend(outputs[:, 1].detach().cpu().numpy())

        n_test = max(1, len(test_loader))
        test_mse /= n_test
        test_mae /= n_test
        test_mse_steer /= n_test
        test_mae_steer /= n_test
        test_mse_throttle /= n_test
        test_mae_throttle /= n_test

        writer.add_scalar("performance/test_mse", test_mse)
        writer.add_scalar("performance/test_mae", test_mae)
        writer.add_scalar("performance/test_rmse", mse_to_rmse(test_mse))
        writer.add_scalar("performance/test_pct_rmse", mse_to_pct_rmse(test_mse))

        test_r2_dict = r2_from_batches(test_y_true_batches, test_y_pred_batches)
        test_r2_mean = test_r2_dict["mean"]
        test_r2_steer = test_r2_dict["steer"]
        test_r2_throttle = test_r2_dict["throttle"]

        print(f"[TEST] MSE: {test_mse:.6f} | MAE: {test_mae:.6f}")
        print(f"[TEST] Steer MSE: {test_mse_steer:.6f} | Throttle MSE: {test_mse_throttle:.6f}")
        print(f"[TEST] R² mean: {test_r2_mean:.4f} | steer: {test_r2_steer:.4f} | throttle: {test_r2_throttle:.4f}")

        # Scatters test (coloreado por error throttle usando tau_throttle de VAL)
        all_gt_steer = np.array(all_gt_steer)
        all_pred_steer = np.array(all_pred_steer)
        all_gt_throttle = np.array(all_gt_throttle)
        all_pred_throttle = np.array(all_pred_throttle)

        test_err_throttle = all_pred_throttle - all_gt_throttle
        test_abs_err_throttle = np.abs(test_err_throttle)

        inside_thr = test_abs_err_throttle <= tau_throttle
        above_thr = test_err_throttle > tau_throttle
        below_thr = test_err_throttle < -tau_throttle

        print(f"[TEST] Porcentaje dentro banda throttle: {100.0 * inside_thr.mean():.1f}% (tau_throttle={tau_throttle:.4f})")

        # Steer scatter colored by throttle error
        fig_scatter_steer, ax_s = plt.subplots(figsize=(4, 4), dpi=120)
        ax_s.scatter(all_gt_steer[inside_thr], all_pred_steer[inside_thr],
                     alpha=0.3, s=5, c="green", label="Throttle dentro banda")
        ax_s.scatter(all_gt_steer[above_thr], all_pred_steer[above_thr],
                     alpha=0.7, s=8, c="orange", label="Throttle por encima banda")
        ax_s.scatter(all_gt_steer[below_thr], all_pred_steer[below_thr],
                     alpha=0.7, s=8, c="purple", label="Throttle por debajo banda")

        ax_s.plot([-1, 1], [-1, 1], "r--", linewidth=1, label="y = x")
        ax_s.set_xlim(-1.05, 1.05)
        ax_s.set_ylim(-1.05, 1.05)
        ax_s.set_xlabel("Steer GT")
        ax_s.set_ylabel("Steer Pred")
        ax_s.set_title("GT vs Pred - Steer (Test, colored by throttle error)")
        ax_s.grid(True, alpha=0.3)
        ax_s.legend(fontsize=7)
        plt.tight_layout()
        writer.add_figure("scatter/test_steer_gt_vs_pred_colored_by_throttle", fig_scatter_steer)
        plt.close(fig_scatter_steer)

        # Throttle banded scatter test
        fig_scatter_th, ax_t = plt.subplots(figsize=(4, 4), dpi=120)
        ax_t.scatter(all_gt_throttle[inside_thr], all_pred_throttle[inside_thr],
                     alpha=0.3, s=5, c="green", label="Dentro banda")
        ax_t.scatter(all_gt_throttle[above_thr], all_pred_throttle[above_thr],
                     alpha=0.7, s=8, c="orange", label="Por encima banda")
        ax_t.scatter(all_gt_throttle[below_thr], all_pred_throttle[below_thr],
                     alpha=0.7, s=8, c="purple", label="Por debajo banda")

        x_line = np.linspace(0.0, 1.0, 200)
        ax_t.plot(x_line, x_line, "r--", linewidth=1, label="y = x")
        ax_t.plot(x_line, x_line + tau_throttle, "g--", linewidth=1, label=f"y = x + {tau_throttle:.2f}")
        ax_t.plot(x_line, x_line - tau_throttle, "g--", linewidth=1, label=f"y = x - {tau_throttle:.2f}")

        ax_t.set_xlim(-0.05, 1.05)
        ax_t.set_ylim(-0.05, 1.05)
        ax_t.set_xlabel("Throttle GT")
        ax_t.set_ylabel("Throttle Pred")
        ax_t.set_title("GT vs Pred - Throttle (Test, banded)")
        ax_t.grid(True, alpha=0.3)
        ax_t.legend(fontsize=7)
        plt.tight_layout()
        writer.add_figure("scatter/test_throttle_gt_vs_pred_banded", fig_scatter_th)
        plt.close(fig_scatter_th)

        # Final R² bar (test)
        fig_r2_test = make_bar_figure(
            {"R2_mean": test_r2_mean, "R2_steer": test_r2_steer, "R2_throttle": test_r2_throttle},
            title="Final R² Summary Test",
            ylabel="R²",
            ylim=(0.0, 1.0),
        )
        writer.add_figure("bars/final_r2", fig_r2_test, global_step=args.num_epochs)
        plt.close(fig_r2_test)

    # ---------------
    # Figuras finales 
    # ---------------
    if best_train_loss is not None and best_val_mse_for_plot is not None:
        train_for_plot = best_train_loss
        val_for_plot = best_val_mse_for_plot
        print(f"[INFO] Using BEST epoch {best_epoch} for final Train/Val metrics.")
    else:
        train_for_plot = avg_train
        val_for_plot = val_mse
        print("[WARN] best_* metrics not set, using LAST epoch for Train/Val metrics.")

    test_for_plot = None
    if test_loader is not None:
        try:
            test_for_plot = test_mse
        except NameError:
            test_for_plot = None

    final_mse_dict = {"Train(best)": train_for_plot, "Val(best)": val_for_plot}
    if test_for_plot is not None:
        final_mse_dict["Test"] = test_for_plot

    fig_final_mse = make_bar_figure(
        final_mse_dict,
        title="Final MSE/Loss Summary (best model)",
        ylabel="MSE / Loss",
    )
    writer.add_figure("bars/final_mse", fig_final_mse, global_step=args.num_epochs)
    plt.close(fig_final_mse)

    final_pct_dict = mse_dict_to_percent_rmse(final_mse_dict)
    fig_final_pct = make_bar_figure_percent(final_pct_dict, title="Final error Summary (best model)")
    writer.add_figure("bars/final_percent_rmse", fig_final_pct, global_step=args.num_epochs)
    plt.close(fig_final_pct)

    # Guardar modelo final (best)
    final_pth = os.path.join(model_save_dir, f"pilot_net_model_deepracer_{args.seed}.pth")
    torch.save(model.state_dict(), final_pth)

    # Export ONNX
    dummy_input = torch.randn(1, 4, 66, 200, device=device)
    net_file_name = "mynet_deepracer_gpu.onnx" if torch.cuda.is_available() else "mynet_deepracer.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        net_file_name,
        verbose=False,
        export_params=True,
        opset_version=9,
        input_names=["input"],
        output_names=["output"],
    )

    writer.close()

    print("[OK] Entrenamiento finalizado.")
    print(f"[OK] Best model: {best_path}")
    print(f"[OK] Final model: {final_pth}")
    print(f"[OK] ONNX: {net_file_name}")


if __name__ == "__main__":
    main()
