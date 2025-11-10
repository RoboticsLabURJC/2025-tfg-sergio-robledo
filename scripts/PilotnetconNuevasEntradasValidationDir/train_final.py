import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from utils.processing import *
from utils.pilot_net_dataset import PilotNetDataset
from utils.pilotnet import PilotNet
from utils.transform_helper import createTransform
import argparse
import json
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import csv
from torch.utils.data import Subset
from torchvision import transforms
import matplotlib.pyplot as plt 

def mse_dict_to_percent_rmse(mse_dict):
    # Convierte cada MSE en %RMSE = sqrt(MSE) * 100
    return {k: (float(v) ** 0.5) * 100.0 for k, v in mse_dict.items()}

def make_bar_figure_percent(values_dict, title="Summary %RMSE"):
    fig, ax = plt.subplots(figsize=(4,3), dpi=120)
    labels = list(values_dict.keys())
    vals   = [values_dict[k] for k in labels]
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylabel("% RMSE")
    ax.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}%", ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    return fig

def make_bar_figure(values_dict, title="Summary", ylabel="Metric"):
    fig, ax = plt.subplots(figsize=(4,3), dpi=120)
    labels = list(values_dict.keys())
    vals   = [values_dict[k] for k in labels]
    ax.bar(labels, vals)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', alpha=0.3)
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
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--save_iter", type=int, default=50, help="Iterations between saves")
    parser.add_argument("--print_terminal", action="store_true", help="Print progress every 10 steps")
    parser.add_argument("--seed", type=int, default=123, help="Seed")

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
        transforms.Resize((66, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = PilotNetDataset(
        args.data_dir,
        mirrored=False,
        transform=COMMON_TF,
        preprocessing=args.preprocess
    )
    val_dataset = PilotNetDataset(
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

    pilotModel = PilotNet(image_shape=(66,200,4), num_labels=2).to(device)
    #print("DEBUG -> model.num_channels:", pilotModel.num_channels)          # debe ser 4
    #print("DEBUG -> ln_1.num_features :", pilotModel.ln_1.num_features)     # debe ser 4

        
    ckpt_latest = os.path.join(model_save_dir, f'pilot_net_model_{random_seed}.pth')
    if os.path.isfile(ckpt_latest):
        pilotModel.load_state_dict(torch.load(ckpt_latest, map_location=device))
        best_model = deepcopy(pilotModel)
        args_json_path = os.path.join(model_save_dir, 'args.json')
        last_epoch = json.load(open(args_json_path))['last_epoch'] + 1 if os.path.isfile(args_json_path) else 0
        print(f"Reanudando desde epoch {last_epoch}")
    else:
        best_model = deepcopy(pilotModel)
        last_epoch = 0

    # Pérdidas y optimizador
    criterion_train = nn.MSELoss()
    criterion_mse   = nn.MSELoss()
    criterion_mae   = nn.L1Loss()
    optimizer = torch.optim.Adam(pilotModel.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    global_iter = 0
    global_val_mse = float('inf')

    print("*********** Training Started ************")
    for epoch in range(last_epoch, num_epochs):
        pilotModel.train()
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            if i == 0:
                print("Batch shape:", images.shape)

            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)

            outputs = pilotModel(images)
            loss = criterion_train(outputs, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_iter % save_iter == 0:
                torch.save(pilotModel.state_dict(), ckpt_latest)
            global_iter += 1

            if print_terminal and (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

        # Persistir epoch
        with open(os.path.join(model_save_dir, 'args.json'), 'w') as fp:
            json.dump({'last_epoch': epoch}, fp)

        writer.add_scalar("performance/train_loss", train_loss/len(train_loader), epoch+1)

        # ===== Validation =====
        pilotModel.eval()
        val_mse = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = FLOAT(images).to(device)
                labels = FLOAT(labels.float()).to(device)
                outputs = pilotModel(images)
                val_mse += criterion_mse(outputs, labels).item()
                val_mae += criterion_mae(outputs, labels).item()

        val_mse /= len(val_loader)
        val_mae /= len(val_loader)

        writer.add_scalar("performance/valid_mse", val_mse, epoch+1)
        writer.add_scalar("performance/valid_mae", val_mae, epoch+1)
        writer.add_scalar("performance/valid_loss", val_mse, epoch+1)
        writer_output.writerow([epoch+1, val_mse, val_mae])

        if val_mse < global_val_mse:
            global_val_mse = val_mse
            best_model = deepcopy(pilotModel)
            torch.save(best_model.state_dict(), os.path.join(model_save_dir, f'pilot_net_model_best_{random_seed}.pth'))
            mssg = "Model Improved!!"
        else:
            mssg = "Not Improved!!"

        print(f'Epoch [{epoch+1}/{num_epochs}]  Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f}  {mssg}')

        avg_train_loss = train_loss / len(train_loader)

        # MSE en barras
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
            title=f"Epoch {epoch+1} - %RMSE Train vs Val")

        writer.add_figure("bars/train_val_epoch_percent_rmse", 
        fig_epoch_bars_pct, global_step=epoch+1)
        plt.close(fig_epoch_bars_pct)


    # ======= TEST =======
    pilotModel = best_model

    test_dirs = args.test_dir if args.test_dir is not None else args.data_dir[-1:]
    if args.test_dir is not None:
        overlap = set(test_dirs).intersection(set(args.data_dir))
        if overlap:
            print(f"[WARN] Estas carpetas están en train y test a la vez: {sorted(overlap)}")

    test_set = PilotNetDataset(
        test_dirs,
        mirrored=False,
        transform=COMMON_TF,
        preprocessing=args.preprocess
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print("Check performance on testset")
    pilotModel.eval()
    test_mse = test_mae = 0.0
    test_mse_steer = test_mae_steer = 0.0
    test_mse_throttle = test_mae_throttle = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            outputs = pilotModel(images)

            test_mse += criterion_mse(outputs, labels).item()
            test_mae += criterion_mae(outputs, labels).item()

            test_mse_steer    += criterion_mse(outputs[:, 0], labels[:, 0]).item()
            test_mae_steer    += criterion_mae(outputs[:, 0], labels[:, 0]).item()
            test_mse_throttle += criterion_mse(outputs[:, 1], labels[:, 1]).item()
            test_mae_throttle += criterion_mae(outputs[:, 1], labels[:, 1]).item()

    n = len(test_loader)
    test_mse /= n; test_mae /= n
    test_mse_steer /= n; test_mae_steer /= n
    test_mse_throttle /= n; test_mae_throttle /= n

    writer.add_scalar('performance/Test_MAE', test_mae)
    writer.add_scalar('performance/Test_MSE', test_mse)
    writer.add_scalar('performance/Test_MAE_steer', test_mae_steer)
    writer.add_scalar('performance/Test_MSE_steer', test_mse_steer)
    writer.add_scalar('performance/Test_MAE_throttle', test_mae_throttle)
    writer.add_scalar('performance/Test_MSE_throttle', test_mse_throttle)

    # MSE EN barras
    fig_final_mse = make_bar_figure(
        {"Train(last)": avg_train_loss, "Val(last)": val_mse, "Test": test_mse},
        title="Final MSE/Loss Summary",
        ylabel="MSE / Loss"
    )
    writer.add_figure("bars/final_mse", fig_final_mse, global_step=num_epochs)
    plt.close(fig_final_mse)

    # RMSE EN % en barras
    final_pct_dict = mse_dict_to_percent_rmse({"Train(last)": avg_train_loss, "Val(last)": val_mse, "Test": test_mse})
    fig_final_pct = make_bar_figure_percent(final_pct_dict, title="Final %RMSE Summary")
    writer.add_figure("bars/final_percent_rmse", fig_final_pct, global_step=num_epochs)
    plt.close(fig_final_pct)

    print(f"Test  -> MAE: {test_mae:.4f} | MSE: {test_mse:.4f}")
    print(f"Steer -> MAE: {test_mae_steer:.4f} | MSE: {test_mse_steer:.4f}")
    print(f"Throt -> MAE: {test_mae_throttle:.4f} | MSE: {test_mse_throttle:.4f}")


    # ==== %RMSE finales en csv ====
    train_pct_rmse_final = (avg_train_loss ** 0.5) * 100.0   # del último epoch
    val_pct_rmse_final   = (val_mse        ** 0.5) * 100.0   # del último epoch
    test_pct_rmse_final  = (test_mse       ** 0.5) * 100.0   # del test final

    final_csv = os.path.join(base_dir, "percent_rmse_speed_label.csv")
    with open(final_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_pct_rmse", "val_pct_rmse", "test_pct_rmse"])
        w.writerow([f"{train_pct_rmse_final:.6f}",
                    f"{val_pct_rmse_final:.6f}",
                    f"{test_pct_rmse_final:.6f}"])

    print(f"[OK] Guardado %RMSE final en: {final_csv}")


    # Save final + ONNX
    torch.save(pilotModel.state_dict(), os.path.join(model_save_dir, f'pilot_net_model_deepracer_{random_seed}.pth'))

    net_file_name = "mynet_deepracer_gpu.onnx" if torch.cuda.is_available() else "mynet_deepracer.onnx"
    dummy_input = torch.randn(1, 4, 66, 200, device=device)
    pilotModel = pilotModel.to(device)

    torch.onnx.export(
        pilotModel,
        dummy_input,
        net_file_name,
        verbose=True,
        export_params=True,
        opset_version=9,
        input_names=['input'],
        output_names=['output']
    )
