import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', help="Directory to find Train Data")
    parser.add_argument("--test_dir", action='append', help="Directory to find Test Data")
    parser.add_argument("--val_dir", action='append', default=None, help="Directory to find Validation Data")
    parser.add_argument("--preprocess", action='append', default=None, help="preprocessing information: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--base_dir", type=str, default='/home/sergior/Downloads/pruebas', help="Directory to save everything")
    parser.add_argument("--comment", type=str, default='Random Experiment', help="Comment to know the experiment")
    parser.add_argument("--data_augs", action='append', type=str, default=None, help="Data Augmentations")
    #parser.add_argument("--mirrored_imgs", action='store_true', help="Add mirrored images to the train data")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Policy Net")
    parser.add_argument("--test_split", type=float, default=0.2, help="Train/Val split (frames) when no --val_dir")
    #parser.add_argument("--shuffle", action='store_true', help="Si se barajan los datos de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--save_iter", type=int, default=50, help="Iterations to save the model")
    parser.add_argument("--print_terminal", type=bool, default=False, help="Print progress in terminal")
    parser.add_argument("--seed", type=int, default=123, help="Seed for reproducing")

    args = parser.parse_args()
    return args


if __name__=="__main__":

    # Parse args
    args = parse_args()

    print("TRAIN dirs:", args.data_dir)
    print("VAL dirs  :", getattr(args, "val_dir", None))
    print("TEST dirs :", args.test_dir)

    # Overlap warnings
    if args.test_dir:
        print("Overlap TRAIN-TEST:", set(args.data_dir).intersection(set(args.test_dir)))
    if getattr(args, "val_dir", None):
        print("Overlap TRAIN-VAL :", set(args.data_dir).intersection(set(args.val_dir)))
        print("Overlap VAL-TEST :", set(args.test_dir or []).intersection(set(args.val_dir)))

    # Serialize setup
    exp_setup = vars(args)

    # Paths
    base_dir = os.path.join('experiments', args.base_dir)
    model_save_dir = os.path.join(base_dir, 'trained_models')
    log_dir = os.path.join(base_dir, 'log')
    check_path(base_dir); check_path(log_dir); check_path(model_save_dir)
    print("Saving model in:" + model_save_dir)

    with open(os.path.join(base_dir,'args.json'), 'w') as fp:
        json.dump(exp_setup, fp)

    # Hparams
    augmentations = args.data_augs
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    val_split = args.test_split
    #shuffle_dataset = args.shuffle
    save_iter = args.save_iter
    random_seed = args.seed
    print_terminal = args.print_terminal
    #mirrored_img = args.mirrored_imgs

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Usando dispositivo: {dev_name}")
    FLOAT = torch.FloatTensor

    # TensorBoard
    writer = SummaryWriter(log_dir)
    self_path = os.getcwd()
    writer_output = csv.writer(open(os.path.join(self_path, "last_train_data.csv"), "w"))
    writer_output.writerow(["epoch", "loss"])

    # Transforms
    transform_train = createTransform(augmentations)    # con augs si definiste --data_augs
    transform_eval  = createTransform([])               # sin augs para val/test

    # === TRAIN & VAL ===
    if args.val_dir:
        print("Usando VALIDACIÓN por sesiones (val_dir).")
        train_dataset = PilotNetDataset(
            args.data_dir, 
            transform=transform_train, 
            preprocessing=args.preprocess
        )
        val_dataset = PilotNetDataset(
            args.val_dir,
            transform=transform_eval,
            preprocessing=args.preprocess
        )
    else:
        print("Usando VALIDACIÓN por split de frames (fallback).")

        base_ds_train = PilotNetDataset(
            args.data_dir,
            transform=transform_train, preprocessing=args.preprocess
        )

        dataset_size = len(base_ds_train)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        # if shuffle_dataset:
        #     np.random.seed(random_seed); np.random.shuffle(indices)
        # train_indices, val_indices = indices[split:], indices[:split]

        # 2) TRAIN: si quieres mirroring, créalo aquí, pero usa índices del dataset sin mirroring
        ds_train_for_sampling = PilotNetDataset(
            args.data_dir,transform=transform_train, preprocessing=args.preprocess
        )
        train_dataset = Subset(ds_train_for_sampling, train_indices)

        # 3) VALID: sin mirroring y sin augs
        base_ds_val = PilotNetDataset(
            args.data_dir,
            transform=transform_eval, preprocessing=args.preprocess
        )
        val_dataset = Subset(base_ds_val, val_indices)

    print("len(train_dataset) =", len(train_dataset))
    print("len(val_dataset)   =", len(val_dataset))

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

    # Modelo
    # Nota: usamos base_ds para obtener image_shape/num_labels de forma segura
    probe_ds = PilotNetDataset(args.data_dir, transform=transform_eval, preprocessing=args.preprocess)
    pilotModel = PilotNet(probe_ds.image_shape, probe_ds.num_labels).to(device)

    model_ckpt = os.path.join(model_save_dir, f'pilot_net_model_{random_seed}.pth')
    if os.path.isfile(model_ckpt):
        pilotModel.load_state_dict(torch.load(model_ckpt, map_location=device))
        best_model = deepcopy(pilotModel)
        last_epoch = json.load(open(os.path.join(model_save_dir,'args.json')))['last_epoch']+1
    else:
        best_model = deepcopy(pilotModel)
        last_epoch = 0

    # Loss & opt
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(pilotModel.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    global_iter = 0
    global_val_mse = 0.5

    best_val_loss = float('inf')
    patience = 5
    counter = 0

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
            loss = criterion(outputs, labels)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_iter % save_iter == 0:
                torch.save(pilotModel.state_dict(), model_ckpt)
            global_iter += 1

            if print_terminal and (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # persist last epoch
        with open(os.path.join(model_save_dir,'args.json'), 'w') as fp:
            json.dump({'last_epoch': epoch}, fp)

        writer.add_scalar("performance/train_loss", train_loss/len(train_loader), epoch+1)

        # Validation
        pilotModel.eval()
        with torch.no_grad():
            val_loss = 0.0
            for images, labels in val_loader:
                images = FLOAT(images).to(device)
                labels = FLOAT(labels.float()).to(device)
                outputs = pilotModel(images)
                val_loss += criterion(outputs, labels).item()

            val_loss /= len(val_loader)
            writer.add_scalar("performance/valid_loss", val_loss, epoch+1)
            writer_output.writerow([epoch+1, val_loss])

        if val_loss < global_val_mse:
            global_val_mse = val_loss
            best_model = deepcopy(pilotModel)
            torch.save(best_model.state_dict(), os.path.join(model_save_dir, f'pilot_net_model_best_{random_seed}.pth'))
            mssg = "Model Improved!!"
        else:
            mssg = "Not Improved!!"

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     counter = 0
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print("Early stopping triggered")
        #         break

        print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, val_loss), mssg)

    # Use best model
    pilotModel = best_model

    # === TEST ===
    transformations_val = transform_eval  # sin augs
    test_dirs = args.test_dir if args.test_dir is not None else args.data_dir[-1:]

    if args.test_dir is not None:
        overlap = set(test_dirs).intersection(set(args.data_dir))
        if overlap:
            print(f"[WARN] Estas carpetas están en train y test a la vez (evítalo): {sorted(overlap)}")

    test_set = PilotNetDataset(
        test_dirs,
        transform=transformations_val,
        preprocessing=args.preprocess
    )

    test_loader = DataLoader(test_set, batch_size=batch_size)

    print("Check performance on testset")
    pilotModel.eval()
    with torch.no_grad():
        test_loss = 0.0
        for images, labels in tqdm(test_loader):
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            outputs = pilotModel(images)
            test_loss += criterion(outputs, labels).item()

    test_loss /= len(test_loader)
    writer.add_scalar('performance/Test_MSE', test_loss)
    print(f'Test loss: {test_loss:.6f}')

    # Save final model & export ONNX
    torch.save(pilotModel.state_dict(), os.path.join(model_save_dir, f'pilot_net_model_deepracer_{random_seed}.pth'))

    net_file_name = "mynet_deepracer_gpu.onnx" if torch.cuda.is_available() else "mynet_deepracer.onnx"
    dummy_input = torch.randn(1, 3, 66, 200, device=device)

    pilotModel = pilotModel.to(device)
    dummy_input = dummy_input.to(device)

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
