import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import os
from utils.processing import check_path
from utils.pilot_net_dataset import PilotNetDataset
from utils.pilotnet import PilotNet
from utils.transform_helper import createTransform
import time
import argparse
import json
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dataset_dir", type=str, default="/home/sergior/Downloads/pruebas/datasets", help="Path to main dataset directory with subfolders Deepracer_BaseMap_*")
    parser.add_argument("--base_dir", type=str, default='/home/sergior/Downloads/pruebas', help="Directory to save logs/models")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_iter", type=int, default=100)
    return parser.parse_args()

def load_all_dataset_dirs(base_dir):
    dataset_dirs = []
    for folder in os.listdir(base_dir):
        full_path = os.path.join(base_dir, folder)
        if os.path.isdir(full_path) and folder.startswith("Deepracer_BaseMap_"):
            dataset_dirs.append(full_path)
    return dataset_dirs

if __name__ == "__main__":
    args = parse_args()
    dataset_dirs = load_all_dataset_dirs(args.base_dataset_dir)

    if os.path.isabs(args.base_dir):
        base_dir = os.path.join(args.base_dir, "experiments")
    else:
        base_dir = os.path.join("experiments", args.base_dir)
    model_save_dir = os.path.join(base_dir, 'models')
    log_dir = os.path.join(base_dir, 'log')
    check_path(base_dir)
    check_path(log_dir)
    check_path(model_save_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {torch.cuda.get_device_name(device)}")

    FLOAT = torch.FloatTensor

    # Usado para visualizar las métricas en TensorBoard (entrenamiento, validación, test)
    writer = SummaryWriter(log_dir)
    writer_output = csv.writer(open(base_dir + "/train_log.csv", "w"))
    writer_output.writerow(["epoch", "train_loss", "val_loss"])

    # Load dataset
    transform = createTransform()
    dataset = PilotNetDataset(dataset_dirs, False, transform, preprocessing=["crop", "normal"])

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.test_split * dataset_size))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)

    model = PilotNet(dataset.image_shape, dataset.num_labels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = FLOAT(images).to(device)
            labels = FLOAT(labels).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if global_step % args.save_iter == 0:
                torch.save(model.state_dict(), os.path.join(model_save_dir, f"model_step_{global_step}.pth"))

            global_step += 1

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = FLOAT(images).to(device)
                labels = FLOAT(labels).to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer_output.writerow([epoch + 1, avg_train_loss, avg_val_loss])

        print(f"Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_save_dir, "best_model.pth"))
            print("✅ Best model updated")

    print("✅ Entrenamiento finalizado.")
