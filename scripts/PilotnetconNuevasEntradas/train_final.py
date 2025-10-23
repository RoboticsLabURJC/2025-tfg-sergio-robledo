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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', required=True, help="Directory(ies) for Train Data")
    parser.add_argument("--test_dir", action='append', help="Directory(ies) for Test Data")
    #parser.add_argument("--val_dir", action='append', required=True, help="Directory(ies) for Validation Data")
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

    # Avisos de solape
    # if args.test_dir:
    #     print("Overlap TRAIN-TEST:", set(args.data_dir).intersection(set(args.test_dir)))
    # print("Overlap TRAIN-VAL :", set(args.data_dir).intersection(set(args.val_dir)))
    # if args.test_dir:
    #     print("Overlap VAL-TEST :", set(args.test_dir).intersection(set(args.val_dir)))

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

    # ===========================
    # SIN AUGMENTATIONS
    # ===========================
    # transformations_eval = createTransform([])   # ToTensor + Normalize; SIN augs
    # transformations_train = createTransform([])  # igual que eval (sin augs)

    # ===========================
    # Dataset único y split 80/20
    # ===========================

    # ===== Transform común para todo (train/val/test/inferencia) =====
    COMMON_TF = transforms.Compose([
        transforms.Resize((66, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    full_dataset = PilotNetDataset(
        args.data_dir,            
        mirrored=False,
        transform=COMMON_TF,
        preprocessing=args.preprocess,
        speed_scale=5.0,
    )

    N = len(full_dataset)
    rng = np.random.RandomState(args.seed)
    indices = np.arange(N)
    rng.shuffle(indices)

    n_val = int(round(args.val_split * N))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]


    train_dataset = Subset(full_dataset, train_idx)
    val_dataset   = Subset(full_dataset, val_idx)

    print(f"Samples totales: {N}  |  train: {len(train_dataset)}  |  val: {len(val_dataset)}")
    
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Modelo (usa un dataset de probe para obtener shape y #labels)
    probe_ds = PilotNetDataset(args.data_dir, mirrored=False,
                           transform=COMMON_TF,
                           preprocessing=args.preprocess,
                           speed_scale=10.0)

    pilotModel = PilotNet(probe_ds.image_shape, probe_ds.num_labels).to(device)
    
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

        for i, batch in enumerate(train_loader):
            #el dataset devuelve (img, speed, label)
            images, speeds, labels = batch
            images = images.float().to(device)
            speeds = speeds.float().to(device)      # (B,1)
            labels = labels.float().to(device)

            outputs = pilotModel(images,speeds)
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
            for images, speeds, labels in val_loader:
                images = images.float().to(device)
                speeds = speeds.float().to(device)   # (B,1)
                labels = labels.float().to(device)
                outputs = pilotModel(images, speeds) 
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

    # ======= TEST =======
    pilotModel = best_model

    test_dirs = args.test_dir if args.test_dir is not None else args.data_dir[-1:]
    if args.test_dir is not None:
        overlap = set(test_dirs).intersection(set(args.data_dir))
        if overlap:
            print(f"[WARN] Estas carpetas están en train y test a la vez (evítalo): {sorted(overlap)}")

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
        for images, speeds, labels in tqdm(test_loader):
            images = FLOAT(images).to(device)
            speeds = FLOAT(speeds).to(device)
            labels = FLOAT(labels.float()).to(device)
            outputs = pilotModel(images,speeds)

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

    print(f"Test  -> MAE: {test_mae:.4f} | MSE: {test_mse:.4f}")
    print(f"Steer -> MAE: {test_mae_steer:.4f} | MSE: {test_mse_steer:.4f}")
    print(f"Throt -> MAE: {test_mae_throttle:.4f} | MSE: {test_mse_throttle:.4f}")

    # Save final + ONNX
    torch.save(pilotModel.state_dict(), os.path.join(model_save_dir, f'pilot_net_model_deepracer_{random_seed}.pth'))

    net_file_name = "mynet_deepracer_gpu.onnx" if torch.cuda.is_available() else "mynet_deepracer.onnx"
    dummy_img = torch.randn(1, 3, 66, 200, device=device)
    dummy_spd = torch.randn(1, 1, device=device) 
    pilotModel = pilotModel.to(device)

    torch.onnx.export(
        pilotModel, (dummy_img, dummy_spd),
        net_file_name,
        verbose=True,
        export_params=True,
        opset_version=9,
        input_names=['img', 'speed'],
        output_names=['output']
    )
