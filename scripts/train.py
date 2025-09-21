import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
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


# ===================== NUEVO: wrapper para redondear etiquetas =====================
class QuantizeLabels(torch.utils.data.Dataset):
    """
    Envuelve un dataset que devuelve (imagen, label) con label[tensor([steer, throttle])].
    Aplica redondeo: steer -> 3 decimales, throttle -> 2 decimales.
    """
    def __init__(self, base_dataset, steer_decimals=3, throttle_decimals=2):
        self.base = base_dataset
        self.s_mult = float(10 ** steer_decimals)
        self.t_mult = float(10 ** throttle_decimals)

        # Reexpone atributos útiles si existen
        self.image_shape = getattr(base_dataset, "image_shape", (66, 200, 3))
        self.num_labels  = getattr(base_dataset, "num_labels", 2)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]  # label: tensor float32 de tamaño [2]
        # Asegura tensor float
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.float32)

        # Redondeo estable con torch
        steer    = torch.round(label[0] * self.s_mult) / self.s_mult
        throttle = torch.round(label[1] * self.t_mult) / self.t_mult
        label_q = torch.stack((steer, throttle)).to(dtype=torch.float32)
        return img, label_q
# ================================================================================



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', help="Directory to find Train Data")
    parser.add_argument("--test_dir", action='append', help="Directory to find Test Data")
    parser.add_argument("--val_dir", action='append', default=None, help="Directory to find Validation Data")
    parser.add_argument("--preprocess", action='append', default=None, help="preprocessing information: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--base_dir", type=str, default='/home/sergior/Downloads/pruebas', help="Directory to save everything")
    parser.add_argument("--comment", type=str, default='Random Experiment', help="Comment to know the experiment")
    parser.add_argument("--data_augs", action='append', type=str, default=None, help="Data Augmentations")
    parser.add_argument("--mirrored_imgs", action='store_true', help="Add mirrored images to the train data")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for Policy Net")
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
    

    print("TRAIN dirs:", args.data_dir)
    print("VAL dirs  :", getattr(args, "val_dir", None))
    print("TEST dirs :", args.test_dir)

    # Avisos de solape
    if args.test_dir:
        print("Overlap TRAIN-TEST:", set(args.data_dir).intersection(set(args.test_dir)))
    if getattr(args, "val_dir", None):
        print("Overlap TRAIN-VAL :", set(args.data_dir).intersection(set(args.val_dir)))
        print("Overlap VAL-TEST :", set(args.test_dir or []).intersection(set(args.val_dir)))

    #Convierte el objeto args en un diccionario
    exp_setup = vars(args)

    # Base Directory
    path_to_data = args.data_dir
    base_dir = os.path.join('experiments', args.base_dir)
    model_save_dir = os.path.join(base_dir, 'trained_models')
    log_dir = os.path.join(base_dir, 'log')
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
    writer_output = csv.writer(open(os.path.join(self_path, "last_train_data.csv"), "w"))
    writer_output.writerow(["epoch", "loss"])

    # Transforms (train con augs; val/test sin augs)
    transformations_train = createTransform(augmentations)
    transformations_eval  = createTransform([])

    # === Datasets: SIEMPRE por carpetas ===
    # TRAIN (puede llevar mirroring)
    _train_dataset = PilotNetDataset(
        args.data_dir,
        mirrored=mirrored_img,
        transform=transformations_train,
        preprocessing=args.preprocess
    )
    # ---- aplicar redondeo de etiquetas (steer 3 dec, throttle 2 dec)
    train_dataset = QuantizeLabels(_train_dataset, steer_decimals=3, throttle_decimals=2)


    # VAL (nunca mirroring)
    _val_dataset = PilotNetDataset(
        args.val_dir,
        mirrored=False,
        transform=transformations_eval,
        preprocessing=args.preprocess
    )

    val_dataset = QuantizeLabels(_val_dataset, steer_decimals=3, throttle_decimals=2)


    print("len(train_dataset) =", len(train_dataset))
    print("len(val_dataset)   =", len(val_dataset))
    if mirrored_img:
        print("⚠️  Mirroring activo en TRAIN (dataset duplicado con steer invertido).")

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=args.shuffle, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Modelo (usa un dataset de probe para obtener shape y #labels)
    probe_ds = PilotNetDataset(args.data_dir, mirrored=False,
                               transform=transformations_eval,
                               preprocessing=args.preprocess)
                               
    # Load Model. Crea una instancia del modelo PilotNet con la forma de imagen (image_shape) 
    # y número de salidas
    pilotModel = PilotNet(probe_ds.image_shape, probe_ds.num_labels).to(device)
    ckpt_latest = os.path.join(model_save_dir, f'pilot_net_model_{random_seed}.pth')
    if os.path.isfile(ckpt_latest):
        pilotModel.load_state_dict(torch.load(ckpt_latest, map_location=device))
        best_model = deepcopy(pilotModel)
        args_json_path = os.path.join(model_save_dir, 'args.json')
        last_epoch = json.load(open(args_json_path))['last_epoch'] + 1 if os.path.isfile(args_json_path) else 0
    else:
        best_model = deepcopy(pilotModel)
        last_epoch = 0

    #------------------PRUEBAS-----------------------------
    #criterion = nn.SmoothL1Loss(beta=0.02)
    # optimizer = torch.optim.AdamW(
    #     pilotModel.parameters(),
    #     lr=1e-4,            # si ahora estás en 1e-5 porque ya estabas al final, al reiniciar prueba 1e-4–3e-4
    #     betas=(0.9, 0.999),
    #     weight_decay=1e-4   # 1e-5–3e-4 es un rango razonable
    # )
    #----------------------------------------

    # Loss and optimizer: MSE and Adam
    # --- Pérdida de entrenamiento y métricas de validación ---
    criterion_train = nn.MSELoss()   # <-- se usa para entrenar (igual que antes)
    criterion_mse   = nn.MSELoss()   # métrica en validación
    criterion_mae   = nn.L1Loss()    # métrica en validación (MAE)
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
            loss = criterion_train(outputs, labels)

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

        # Logs en TensorBoard
        writer.add_scalar("performance/valid_mse", val_mse, epoch + 1)
        writer.add_scalar("performance/valid_mae", val_mae, epoch + 1)

        # (opcional) mantener tu curva antigua de "valid_loss" como MSE
        writer.add_scalar("performance/valid_loss", val_mse, epoch + 1)

        # guardar en CSV si lo usas
        writer_output.writerow([epoch + 1, val_mse, val_mae])


        # compare
        if val_mse < global_val_mse:
            global_val_mse = val_mse
            best_model = deepcopy(pilotModel)
            torch.save(best_model.state_dict(), model_save_dir + '/pilot_net_model_best_{}.pth'.format(random_seed))
            mssg = "Model Improved!!"
        else:
            mssg = "Not Improved!!"

        print(f'Epoch [{epoch+1}/{num_epochs}]  Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f}  {mssg}')

    pilotModel = best_model # allot the best model on validation 
    # Test the model
    transformations_val = createTransform([]) 
    # Utiliza la ultima carpeta para test
    test_dirs = args.test_dir if args.test_dir is not None else args.data_dir[-1:]
    if args.test_dir is not None:
        overlap = set(test_dirs).intersection(set(args.data_dir))
        if overlap:
            print(f"[WARN] Estas carpetas están en train y test a la vez (evítalo): {sorted(overlap)}")

    _test_set = PilotNetDataset(
                test_dirs,
                mirrored=False,
                transform=transformations_val,
                preprocessing=args.preprocess
            )

    # ---- redondeo también en test
    test_set = QuantizeLabels(_test_set, steer_decimals=3, throttle_decimals=2)


    test_mse = test_mae = 0.0
    test_mse_steer = test_mae_steer = 0.0
    test_mse_throttle = test_mae_throttle = 0.0

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print("Check performance on testset")
    pilotModel.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            outputs = pilotModel(images)

            # Global (2 salidas a la vez)
            test_mse += criterion_mse(outputs, labels).item()
            test_mae += criterion_mae(outputs, labels).item()

            # Por componente
            test_mse_steer    += criterion_mse(outputs[:, 0], labels[:, 0]).item()
            test_mae_steer    += criterion_mae(outputs[:, 0], labels[:, 0]).item()
            test_mse_throttle += criterion_mse(outputs[:, 1], labels[:, 1]).item()
            test_mae_throttle += criterion_mae(outputs[:, 1], labels[:, 1]).item()

    
    # Promedios por batch
    n = len(test_loader)
    test_mse /= n; test_mae /= n
    test_mse_steer /= n; test_mae_steer /= n
    test_mse_throttle /= n; test_mae_throttle /= n

    # TensorBoard
    writer.add_scalar('performance/Test_MAE', test_mae)
    writer.add_scalar('performance/Test_MSE', test_mse)
    writer.add_scalar('performance/Test_MAE_steer', test_mae_steer)
    writer.add_scalar('performance/Test_MSE_steer', test_mse_steer)
    writer.add_scalar('performance/Test_MAE_throttle', test_mae_throttle)
    writer.add_scalar('performance/Test_MSE_throttle', test_mse_throttle)

    print(f"Test  -> MAE: {test_mae:.4f} | MSE: {test_mse:.4f}")
    print(f"Steer -> MAE: {test_mae_steer:.4f} | MSE: {test_mse_steer:.4f}")
    print(f"Throt -> MAE: {test_mae_throttle:.4f} | MSE: {test_mse_throttle:.4f}")
            
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
