import os, json, csv, random, argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from utils.processing import *
from utils.pilot_net_dataset import PilotNetDataset
from utils.pilotnet import PilotNet
from utils.transform_helper import createTransform


import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random



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



class RandomRoll:
    """
    Rotación ±max_deg en plano de imagen con prob p.
    Para RGB usamos BILINEAR; para máscaras sería NEAREST.
    """
    def __init__(self, max_deg=1.5, p=0.5, for_masks=False, fill=0.0):
        self.max_deg = float(max_deg)
        self.p = float(p)
        self.for_masks = for_masks
        self.fill = fill  # 0.0 está bien (negro); si prefieres gris neutro usa 0.5

    def __call__(self, img):
        if self.max_deg <= 0 or random.random() > self.p:
            return img
        angle = random.uniform(-self.max_deg, self.max_deg)
        interp = InterpolationMode.NEAREST if self.for_masks else InterpolationMode.BILINEAR
        return TF.rotate(img, angle=angle, interpolation=interp, fill=self.fill)


# ---------------------- ARGS ----------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', help="Directory to find Train Data")
    parser.add_argument("--test_dir", action='append', help="Directory to find Test Data")
    parser.add_argument("--val_dir", action='append', default=None, help="Directory to find Validation Data")
    parser.add_argument("--preprocess", action='append', default=None, help="preprocessing information: crop/nocrop, normal/extreme")
    parser.add_argument("--base_dir", type=str, default='/home/sergior/Downloads/pruebas', help="Directory to save everything")
    parser.add_argument("--comment", type=str, default='Random Experiment', help="Comment to know the experiment")
    parser.add_argument("--data_augs", action='append', type=str, default=None, help="Data Augmentations")

    parser.add_argument("--mirrored_imgs", action='store_true', help="Add mirrored images to the train data")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Policy Net")
    parser.add_argument("--test_split", type=float, default=0.2, help="Train/Val split (frames) when no --val_dir")
    parser.add_argument("--shuffle", action='store_true', help="Shuffle training data (ignored if --use_weighted_sampler)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--save_iter", type=int, default=50, help="Iterations to save the model")
    parser.add_argument("--print_terminal", type=bool, default=False, help="Print progress in terminal")
    parser.add_argument("--seed", type=int, default=123, help="Seed for reproducibility")


    parser.add_argument("--roll_deg", type=float, default=1.5,
                    help="Máximo |roll| (grados) en TRAIN. 0 = desactivado")
    parser.add_argument("--roll_p", type=float, default=0.5,    
                    help="Probabilidad de aplicar roll por imagen en TRAIN")


    # ---- Downsample de rectas ----
    parser.add_argument("--downsample_straight", action='store_true',
                        help="Downsample frames con |steer| < straight_thresh en TRAIN")
    parser.add_argument("--straight_thresh", type=float, default=0.05,
                        help="Umbral de |steer| para considerar 'recta'")
    parser.add_argument("--keep_every", type=int, default=8,
                        help="Mantener 1 de cada K rectas (cuando --downsample_straight)")

    # ---- WeightedRandomSampler ----
    parser.add_argument("--use_weighted_sampler", action='store_true',
                        help="Usar WeightedRandomSampler por bins de |steer|")
    parser.add_argument("--sampler_edges", type=float, nargs=4, default=[0.05, 0.20, 0.50, 1.01],
                        help="Límites de bins para |steer| (4 valores → 4 bins)")
    parser.add_argument("--sampler_target", type=float, nargs=4, default=[0.45, 0.35, 0.15, 0.05],
                        help="Distribución objetivo por bin (suma ~1.0)")

    # ---- Reanudar / volver a epoch ----
    parser.add_argument("--resume_epoch", type=int, default=None,
                        help="Cargar el checkpoint de ESA epoch (1-indexed) y continuar desde ahí")
    parser.add_argument("--reset_optim", action='store_true',
                        help="Si se activa, al reanudar NO carga el estado del optimizador")

    return parser.parse_args()


# ---------------------- HELPERS ----------------------
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_steers_from_dataset(ds):
    """Devuelve np.array con el 'steer' de cada muestra. Asume labels[0] = steer."""
    if isinstance(ds, Subset):
        base, idxs = ds.dataset, ds.indices
        return np.array([float(base[i][1][0]) for i in idxs], dtype=np.float32)
    return np.array([float(ds[i][1][0]) for i in range(len(ds))], dtype=np.float32)


def downsample_straights_on_indices(base_ds_eval, indices, thresh=0.05, keep_every=8):
    """Mantiene todas las curvas y 1/k de las rectas según |steer|."""
    keep, counter = [], 0
    for idx in indices:
        steer = abs(float(base_ds_eval[idx][1][0]))
        if steer < thresh:
            counter += 1
            if counter % keep_every == 0:
                keep.append(idx)
        else:
            keep.append(idx)
    return keep


def make_weighted_sampler(train_ds_or_subset, edges, target):
    """Sampler ponderado por bins de |steer|."""
    steers = np.abs(get_steers_from_dataset(train_ds_or_subset))
    bins = np.digitize(steers, edges)  # 0..3
    counts = np.bincount(bins, minlength=4).astype(np.float64)
    counts[counts == 0] = 1.0
    target = np.array(target, dtype=np.float64); target = target / target.sum()
    w_per_bin = (target / counts)
    weights = w_per_bin[bins]
    # recorte para evitar pesos extremos (estabilidad)
    w_min, w_max = np.percentile(weights, 5), np.percentile(weights, 95)
    weights = np.clip(weights, w_min, w_max)
    return WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True)


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    args = parse_args()
    set_all_seeds(args.seed)

    print("TRAIN dirs:", args.data_dir)
    print("VAL dirs  :", getattr(args, "val_dir", None))
    print("TEST dirs :", args.test_dir)

    # Overlap warnings
    if args.test_dir:
        print("Overlap TRAIN-TEST:", set(args.data_dir).intersection(set(args.test_dir)))
    if getattr(args, "val_dir", None):
        print("Overlap TRAIN-VAL :", set(args.data_dir).intersection(set(args.val_dir)))
        print("Overlap VAL-TEST :", set(args.test_dir or []).intersection(set(args.val_dir)))

    # Paths
    base_dir = os.path.join('experiments', args.base_dir)
    model_save_dir = os.path.join(base_dir, 'trained_models')
    log_dir = os.path.join(base_dir, 'log')
    check_path(base_dir); check_path(log_dir); check_path(model_save_dir)
    print("Saving model in:", model_save_dir)

    with open(os.path.join(base_dir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp)

    # Hparams
    augmentations = args.data_augs
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    val_split = args.test_split
    save_iter = args.save_iter
    random_seed = args.seed
    print_terminal = args.print_terminal
    mirrored_img = args.mirrored_imgs

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Usando dispositivo: {dev_name}")

    # TensorBoard + CSV
    writer = SummaryWriter(log_dir)
    self_path = os.getcwd()
    writer_output = csv.writer(open(os.path.join(self_path, "last_train_data.csv"), "w"))
    writer_output.writerow(["epoch", "val_mse", "val_mae"])

    # Transforms
    transform_train = createTransform(augmentations)    # con augs
    transform_eval  = createTransform([])               # sin augs


        # Añadir roll ligero SOLO en train
    if args.roll_deg > 0:
        roll = RandomRoll(max_deg=args.roll_deg, p=args.roll_p, for_masks=False, fill=0.0)
        transform_train = T.Compose([roll, transform_train])

    # === TRAIN & VAL ===
    if args.val_dir:
        print("Usando VALIDACIÓN por sesiones (val_dir).")
        _train_dataset = PilotNetDataset(
            args.data_dir, mirrored=mirrored_img,
            transform=transform_train, preprocessing=args.preprocess
        )

        # ---- aplicar redondeo de etiquetas (steer 3 dec, throttle 2 dec)
        train_dataset = QuantizeLabels(_train_dataset, steer_decimals=3, throttle_decimals=2)


        _val_dataset = PilotNetDataset(
            args.val_dir, mirrored=False,
            transform=transform_eval, preprocessing=args.preprocess
        )

        val_dataset = QuantizeLabels(_val_dataset, steer_decimals=3, throttle_decimals=2)


    else:
        print("Usando VALIDACIÓN por split de frames.")
        base_ds_eval = PilotNetDataset(args.data_dir, mirrored=False, transform=transform_eval, preprocessing=args.preprocess)

        dataset_size = len(base_ds_eval)
        indices = np.arange(dataset_size)
        np.random.seed(random_seed)              # baraja SIEMPRE para evitar sesgo temporal
        np.random.shuffle(indices)

        split = int(np.floor(val_split * dataset_size))
        val_indices   = indices[:split].tolist()
        train_indices = indices[split:].tolist()

        # Downsample de rectas (opcional)
        if args.downsample_straight:
            before = len(train_indices)
            train_indices = downsample_straights_on_indices(
                base_ds_eval, train_indices,
                thresh=args.straight_thresh, keep_every=args.keep_every
            )
            print(f"[Downsample] Train: {before} -> {len(train_indices)} (th={args.straight_thresh}, keep 1/{args.keep_every})")

        ds_train_full = PilotNetDataset(
            args.data_dir, mirrored=mirrored_img,
            transform=transform_train, preprocessing=args.preprocess
        )
        _train_dataset = Subset(ds_train_full, train_indices)

        train_dataset = QuantizeLabels(_train_dataset, steer_decimals=3, throttle_decimals=2)


        ds_val_full = PilotNetDataset(
            args.data_dir, mirrored=False,
            transform=transform_eval, preprocessing=args.preprocess
        )
        _val_dataset = Subset(ds_val_full, val_indices)

        val_dataset = QuantizeLabels(_val_dataset, steer_decimals=3, throttle_decimals=2)



    print("len(train_dataset) =", len(train_dataset))
    print("len(val_dataset)   =", len(val_dataset))

    # ---------- Loaders ----------
    g = torch.Generator().manual_seed(random_seed)

    if args.use_weighted_sampler:
        sampler = make_weighted_sampler(train_dataset,
                                        edges=tuple(args.sampler_edges),
                                        target=tuple(args.sampler_target))
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  sampler=sampler,     # sampler en vez de shuffle
                                  num_workers=4,
                                  pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=args.shuffle,
                                  generator=g,
                                  num_workers=4,
                                  pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ---------- Modelo ----------
    probe_ds = PilotNetDataset(args.data_dir, mirrored=False, transform=transform_eval, preprocessing=args.preprocess)
    pilotModel = PilotNet(probe_ds.image_shape, probe_ds.num_labels).to(device)

    # ---------- Optim & Loss ----------
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = torch.optim.Adam(pilotModel.parameters(), lr=learning_rate)

    # ---------- Ruta para checkpoint periódico (ligero) ----------
    model_ckpt = os.path.join(model_save_dir, f'pilot_net_model_{random_seed}.pth')


    start_epoch = 0
    best_model = deepcopy(pilotModel)
    best_val_mse = float('inf')

   # ---------- Resume / volver a epoch (fix RNG) ----------
    if args.resume_epoch is not None:
        ckpt_p = os.path.join(model_save_dir, f"ckpt_epoch_{args.resume_epoch:03d}.pth")
        if not os.path.isfile(ckpt_p):
            raise FileNotFoundError(f"No existe el checkpoint {ckpt_p}")

        # Cargar SIEMPRE en CPU para evitar conflictos con el estado del RNG
        ckpt = torch.load(ckpt_p, map_location="cpu")

        # 1) Pesos del modelo
        pilotModel.load_state_dict(ckpt["model_state"])

        # 2) Estado del optimizador (opcional)
        if (not args.reset_optim) and ("optim_state" in ckpt):
            optimizer.load_state_dict(ckpt["optim_state"])

        # 3) Restaurar RNG de forma segura
        rng = ckpt.get("rng_state", None)
        if rng is not None:
            st_torch = rng.get("torch", None)
            if isinstance(st_torch, torch.Tensor):
                # torch.set_rng_state espera un ByteTensor en CPU
                torch.set_rng_state(st_torch.cpu())

            st_cuda = rng.get("cuda", None)
            if torch.cuda.is_available() and st_cuda is not None:
                # Si hay varios dispositivos, aplica el estado por dispositivo
                # (torch.cuda.set_rng_state acepta ByteTensor en CPU)
                for dev_id, state in enumerate(st_cuda):
                    torch.cuda.set_rng_state(state, device=dev_id)

        start_epoch = ckpt.get("epoch", 0)  # continúa a partir de esta epoch
        best_model = deepcopy(pilotModel)
        best_val_mse = ckpt.get("val_mse", float('inf'))
        print(f"[Resume] Loaded epoch {args.resume_epoch} (val_mse={best_val_mse:.4f})")

    elif os.path.isfile(model_ckpt) and os.path.isfile(os.path.join(model_save_dir, 'args.json')):
        # Auto-resume “clásico” (desde último .pth + last_epoch)
        pilotModel.load_state_dict(torch.load(model_ckpt, map_location=device))
        best_model = deepcopy(pilotModel)
        start_epoch = json.load(open(os.path.join(model_save_dir, 'args.json')))['last_epoch'] + 1
        print(f"[Auto-resume] Starting from epoch {start_epoch}")

    total_step = len(train_loader)
    global_iter = 0

    print("*********** Training Started ************")
    for epoch in range(start_epoch, num_epochs):
        pilotModel.train()
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            if i == 0:
                print("Batch shape:", images.shape)

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            outs = pilotModel(images)
            loss = criterion_mse(outs, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_iter > 0 and global_iter % save_iter == 0:
                torch.save(pilotModel.state_dict(), model_ckpt)
            global_iter += 1

            if print_terminal and (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

        # Persist last epoch (auto-resume compat)
        with open(os.path.join(model_save_dir, 'args.json'), 'w') as fp:
            json.dump({'last_epoch': epoch}, fp)

        writer.add_scalar("performance/train_loss_mse", train_loss / len(train_loader), epoch + 1)

        # ---------- Validation ----------
        pilotModel.eval()
        val_mse = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device, dtype=torch.float32)
                lbls = lbls.to(device, dtype=torch.float32)
                pred = pilotModel(imgs)
                val_mse += criterion_mse(pred, lbls).item()
                val_mae += criterion_mae(pred, lbls).item()

        val_mse /= len(val_loader)
        val_mae /= len(val_loader)

        writer.add_scalar("performance/valid_mse", val_mse, epoch + 1)
        writer.add_scalar("performance/valid_mae", val_mae, epoch + 1)
        writer.add_scalar("performance/valid_loss", val_mse, epoch + 1)
        writer_output.writerow([epoch + 1, val_mse, val_mae])

        improved = ""
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model = deepcopy(pilotModel)
            torch.save(best_model.state_dict(), os.path.join(model_save_dir, f'pilot_net_model_best_{random_seed}.pth'))
            improved = "  <-- Model Improved!!"

        print(f'Epoch [{epoch+1}/{num_epochs}]  Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f}{improved}')

        # ---------- Checkpoint por epoch (para volver cuando quieras) ----------
        ckpt_path = os.path.join(model_save_dir, f"ckpt_epoch_{epoch+1:03d}.pth")
        torch.save({
            "epoch": epoch + 1,  # la siguiente epoch a entrenar
            "model_state": pilotModel.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_mse": val_mse,
            "val_mae": val_mae,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }, ckpt_path)

    # Use best model
    pilotModel = best_model

    # ---------- TEST ----------
    transformations_val = transform_eval
    test_dirs = args.test_dir if args.test_dir else args.data_dir[-1:]

    if args.test_dir:
        overlap = set(test_dirs).intersection(set(args.data_dir))
        if overlap:
            print(f"[WARN] Estas carpetas están en train y test: {sorted(overlap)}")

    _test_set = PilotNetDataset(
        test_dirs, mirrored=False,
        transform=transformations_val, preprocessing=args.preprocess
    )

    test_set = QuantizeLabels(_test_set, steer_decimals=3, throttle_decimals=2)


    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print("Check performance on testset")
    pilotModel.eval()
    test_mse = 0.0
    test_mae = 0.0
    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader):
            imgs = imgs.to(device, dtype=torch.float32)
            lbls = lbls.to(device, dtype=torch.float32)
            pred = pilotModel(imgs)
            test_mse += criterion_mse(pred, lbls).item()
            test_mae += criterion_mae(pred, lbls).item()

    test_mse /= len(test_loader)
    test_mae /= len(test_loader)
    writer.add_scalar('performance/Test_MSE', test_mse)
    writer.add_scalar('performance/Test_MAE', test_mae)
    print(f'Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f}')

    # Save final model & export ONNX
    torch.save(pilotModel.state_dict(), os.path.join(model_save_dir, f'pilot_net_model_deepracer_{random_seed}.pth'))

    net_file_name = "mynet_deepracer_gpu.onnx" if torch.cuda.is_available() else "mynet_deepracer.onnx"
    dummy_input = torch.randn(1, 3, 66, 200, device=device)

    torch.onnx.export(
        pilotModel, dummy_input, net_file_name,
        verbose=True, export_params=True, opset_version=9,
        input_names=['input'], output_names=['output']
    )
