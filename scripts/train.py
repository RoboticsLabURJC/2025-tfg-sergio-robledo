import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import os
from utils.processing import *
from utils.pilot_net_dataset import PilotNetDataset
from utils.pilotnet import PilotNet
from utils.transform_helper import createTransform
import time
import argparse
from PIL import Image
import cv2
import json
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import csv

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', help="Directory to find Train Data")
    parser.add_argument("--test_dir", action='append', help="Directory to find Test Data")
    parser.add_argument("--preprocess", action='append', default=None, help="preprocessing information: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--base_dir", type=str, default='/home/sergior/Downloads/pruebas', help="Directory to save everything")
    parser.add_argument("--comment", type=str, default='Random Experiment', help="Comment to know the experiment")
    parser.add_argument("--data_augs", action='append', type=str, default=None, help="Data Augmentations")
    parser.add_argument("--mirrored_imgs", action='store_true', help="Add mirrored images to the train data")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Policy Net")
    parser.add_argument("--test_split", type=float, default=0.2, help="Train test Split")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--save_iter", type=int, default=50, help="Iterations to save the model")
    parser.add_argument("--print_terminal", type=bool, default=False, help="Print progress in terminal")
    parser.add_argument("--seed", type=int, default=123, help="Seed for reproducing")

    args = parser.parse_args()
    return args


if __name__=="__main__":

    # Parsear argumentos
    args = parse_args()

    #Convierte el objeto args en un diccionario
    exp_setup = vars(args)

    # Base Directory
    path_to_data = args.data_dir
    base_dir = os.path.join('experiments', args.base_dir)
    model_save_dir = base_dir + 'trained_models'
    log_dir = base_dir + 'log'
    check_path(base_dir)
    check_path(log_dir)
    check_path(model_save_dir)

    print("Saving model in:" + model_save_dir)

    # Crea un archivo JSON con todos los parámetros del experimento
    #(útil para reproducibilidad o debugging posterior)
    with open(base_dir+'args.json', 'w') as fp:
        json.dump(exp_setup, fp)

    # Hyperparameters
    augmentations = args.data_augs
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    val_split = args.test_split
    shuffle_dataset = args.shuffle
    save_iter = args.save_iter
    random_seed = args.seed
    print_terminal = args.print_terminal
    mirrored_img = args.mirrored_imgs

    # Device Selection (CPU/GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {torch.cuda.get_device_name(device)}")

    FLOAT = torch.FloatTensor

    # Tensorboard Initialization. log_dir: carpeta donde se guardan los archivos .tfevents
    writer = SummaryWriter(log_dir)

    self_path = os.getcwd()
    writer_output = csv.writer(open(self_path + "/last_train_data.csv", "w"))
    writer_output.writerow(["epoch", "loss"])

    # Define data transformations. Llama a una función createTransform(...), 
    #que genera una composición de transformaciones de imagen 
    # (por ejemplo, Resize, ToTensor, Normalize, etc.).
    transformations = createTransform(augmentations)
    
    # === 1. Dataset base sin mirroring para hacer split ===
    dataset = PilotNetDataset(path_to_data, mirrored=False, transform=transformations, preprocessing=args.preprocess)

    # === 2. Crear índices de split ===
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # === 3. Dataset con y sin mirroring ===
    train_dataset = torch.utils.data.Subset(
        PilotNetDataset(path_to_data, mirrored=mirrored_img, transform=transformations, preprocessing=args.preprocess),
        train_indices
    )

    val_dataset = torch.utils.data.Subset(
        PilotNetDataset(path_to_data, mirrored=False, transform=transformations, preprocessing=args.preprocess),
        val_indices
    )

    # === 4. Loaders ===
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)


    # Load Model. Crea una instancia del modelo PilotNet con la forma de imagen (image_shape) 
    # y número de salidas
    pilotModel = PilotNet(dataset.image_shape, dataset.num_labels).to(device)
    if os.path.isfile( model_save_dir + '/pilot_net_model_{}.pth'.format(random_seed)):
        pilotModel.load_state_dict(torch.load(model_save_dir + '/pilot_net_model_{}.pth'.format(random_seed),map_location=device))
        best_model = deepcopy(pilotModel)
        last_epoch = json.load(open(model_save_dir+'/args.json',))['last_epoch']+1
    else:
        last_epoch = 0

    # Loss and optimizer: MSE and Adam
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(pilotModel.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader) #cuántos batches hay por epoch
    global_iter = 0 # contador global de pasos.
    global_val_mse = 0.5 # valor inicial del mejor error de validación para comparar


    print("*********** Training Started ************")

    for epoch in range(last_epoch, num_epochs):
        pilotModel.train()
        # Inicializa la pérdida acumulada de esta época
        train_loss = 0

        # Cada labels es un tensor [batch_size, 2] con steer y throttle y 
        # Cada images es un batch de entrada 
        for i, (images, labels) in enumerate(train_loader):
            if i == 0:
                print("Batch shape:", images.shape)

            #Convierte los tensores a tipo FloatTensor y los pasa a 
            # la GPU o CPU según disponibilidad.
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)


            # Run the forward pass. Se hace una predicción (outputs) 
            # con el modelo dado un lote de images. Luego se calcula la pérdida MSE 
            # comparando outputs vs labels.

            outputs = pilotModel(images)
            loss = criterion(outputs, labels)

            current_loss = loss.item()
            train_loss += current_loss


            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            #Se hace backpropagation (
            loss.backward()
            #Se actualizan los pesos con Adam 
            optimizer.step()

            #Guardar modelo cada X iteraciones
            if global_iter % save_iter == 0:
                torch.save(pilotModel.state_dict(), model_save_dir + '/pilot_net_model_{}.pth'.format(random_seed))
            global_iter += 1

            if print_terminal and (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        
        # add entry of last epoch                            
        with open(model_save_dir+'/args.json', 'w') as fp:
            json.dump({'last_epoch': epoch}, fp)

        writer.add_scalar("performance/train_loss", train_loss/len(train_loader), epoch+1)
        
        # Validation 
        pilotModel.eval()
        with torch.no_grad():
            val_loss = 0 
            for images, labels in val_loader:
                images = FLOAT(images).to(device)
                labels = FLOAT(labels.float()).to(device)
                outputs = pilotModel(images)
                val_loss += criterion(outputs, labels).item()
                
            val_loss /= len(val_loader) # take average
            writer.add_scalar("performance/valid_loss", val_loss, epoch+1)
        
            writer_output.writerow([epoch+1,val_loss])

        # compare
        if val_loss < global_val_mse:
            global_val_mse = val_loss
            best_model = deepcopy(pilotModel)
            torch.save(best_model.state_dict(), model_save_dir + '/pilot_net_model_best_{}.pth'.format(random_seed))
            mssg = "Model Improved!!"
        else:
            mssg = "Not Improved!!"

        print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, val_loss), mssg)
        

    pilotModel = best_model # allot the best model on validation 
    # Test the model
    transformations_val = createTransform([]) 
    # Utiliza la ultima carpeta para test
    test_dirs = args.test_dir if args.test_dir is not None else args.data_dir[-1:]
    test_set = PilotNetDataset(test_dirs, transformations_val, preprocessing=args.preprocess)

    test_loader = DataLoader(test_set, batch_size=batch_size)
    print("Check performance on testset")
    pilotModel.eval()
    with torch.no_grad():
        test_loss = 0
        for images, labels in tqdm(test_loader):
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            outputs = pilotModel(images)
            test_loss += criterion(outputs, labels).item()
    
    writer.add_scalar('performance/Test_MSE', test_loss/len(test_loader))
    print(f'Test loss: {test_loss/len(test_loader)}')
        
    # Save the model and plot
    torch.save(pilotModel.state_dict(), model_save_dir + '/pilot_net_model_deepracer_{}.pth'.format(random_seed))
    
    net_file_name = "mynet_deepracer.onnx"
    
    if torch.cuda.is_available():
        dummy_input = torch.randn(1, 3, 66, 200, device=device)
        net_file_name = "mynet_deepracer_gpu.onnx"
    else:
        dummy_input = torch.randn(1, 3, 66, 200, device=device)

    # ✅ Asegura que el modelo está en el mismo device
    pilotModel = pilotModel.to(device)
    dummy_input = dummy_input.to(device)

    # Exportar
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
