"""
DSTFS-adapted.py
Implementación de la rama temporal de DSTFS (tarea única).
"""
from __future__ import annotations
from typing import cast
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# --- CONFIGURACIÓN PRINCIPAL ---
# Reemplaza los argumentos de terminal por un diccionario fácil de editar.
CONFIG = {
    "epochs": 120,
    "patience": 15,
    "min_delta": 1.0,
    "batch_size": 16,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "min_time_s": 0.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "time_scale": 30.0,  # DSTFS: reduce rango numérico del target en la pérdida
}

# --- DATASET Y AUGMENTATION ---
def _build_train_transform() -> transforms.Compose:
    # DSTFS augmentation: Resize(224)->Rot(±10)->Traslación(0-10%)->Flip->Crop(200)->Resize(112)
    return transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10), 
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomCrop(200),
        transforms.Resize((112, 112)), 
        transforms.ToTensor(),
    ])

def _build_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((112, 112)), 
        transforms.ToTensor()
    ])

class ThermalTraceDataset(Dataset):
    """
    DSTFS (Tarea única): 
    - Entrada: imagen térmica BRUTA (thermal_npy), no deltaT, pues el modelo debe separar identidad/tiempo de la imagen original.
    - Normalización: lineal [-1, 1] por imagen, ya que el fondo varía entre capturas absolutas.
    """
    def __init__(self, metadata_csv="processed_data/metadata.csv", is_train=True):
        self.root = Path("processed_data")
        self.transform = _build_train_transform() if is_train else _build_val_transform()
        
        df = pd.read_csv(metadata_csv).dropna(subset=["thermal_path", "sequence_id", "label_time_s"])
        self.df = df[df["label_time_s"].astype(float) > CONFIG["min_time_s"]].reset_index(drop=True)
        self.df = self.df[[self._resolve(p).exists() for p in self.df["thermal_path"]]].reset_index(drop=True)

    def __len__(self) -> int: return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[i]
        thermal = np.load(self._resolve(row["thermal_path"])).astype(np.float32)
        
        # Escalar a uint8 para torchvision
        tmin, tmax = thermal.min(), thermal.max()
        th_u8 = ((thermal - tmin) / (tmax - tmin) * 255).astype(np.uint8) if tmax - tmin > 1e-6 else np.zeros_like(thermal, dtype=np.uint8)
        
        x = cast(torch.Tensor, self.transform(th_u8))
        
        # Normalización [-1, 1] de DSTFS por imagen individual
        xmin, xmax = x.min(), x.max()
        if xmax - xmin > 1e-6: x = (x - xmin) / (xmax - xmin) * 2.0 - 1.0
        else: x = torch.zeros_like(x)

        return x, torch.tensor(float(row["label_time_s"]), dtype=torch.float32)

    def _resolve(self, p: str) -> Path:
        path = Path(str(p).replace("\\", "/"))
        return path if path.is_absolute() else self.root / path

# --- COMPONENTES DEL MODELO (DSTFS) ---

class SoftThresholdPReLU(nn.Module):
    """DSTFS: PReLU con umbral suave adaptativo. Filtra ruido IR débil. Se usa solo si stride=1."""
    def __init__(self, channels: int):
        super().__init__()
        self.prelu = nn.PReLU(channels)
        self.threshold = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_prelu = self.prelu(x)
        return torch.sign(x_prelu) * torch.relu(torch.abs(x_prelu) - torch.relu(self.threshold))

class ChannelAttention(nn.Module):
    """DSTFS: Squeeze-and-Excitation combinando avg-pool y max-pool globales."""
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 16, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False), 
            nn.ReLU(True), 
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True)))

class SpatialAttention(nn.Module):
    """DSTFS: Concatena avg y max espaciales -> Conv 7x7 para capturar difusión térmica amplia."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)))

class ResidualAttentionBlock(nn.Module):
    """DSTFS: Bloque residual ResNet + SPReLU (stride=1) + Atención Dual sobre el residuo."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm2d(out_c), nn.BatchNorm2d(out_c)
        self.act1 = SoftThresholdPReLU(out_c) if stride == 1 else nn.PReLU(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.att_c, self.att_s = ChannelAttention(out_c), SpatialAttention()
        self.act2 = nn.PReLU(out_c)
        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride, bias=False), 
            nn.BatchNorm2d(out_c)
        ) if stride != 1 or in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.act1(self.bn1(self.conv1(x)))
        res = self.att_s(self.att_c(self.bn2(self.conv2(res))))
        return self.act2(res + self.skip(x))

class ThermalDepartureTimeNet(nn.Module):
    """
    DSTFS Temporal: Stem -> 4 Stages (64->128->256->512) -> Head.
    Input: 112x112x1. Softplus en salida garantiza tiempo > 0.
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.PReLU(64), 
            nn.MaxPool2d(3, 2, 1)
        )
        self.stage1 = self._stage(64, 64, 2, 1)
        self.stage2 = self._stage(64, 128, 2, 2)
        self.stage3 = self._stage(128, 256, 2, 2)
        self.stage4 = self._stage(256, 512, 2, 2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(), 
            nn.Dropout(0.25), 
            nn.Linear(512, 64), 
            nn.PReLU(64), 
            nn.Linear(64, 1), 
            nn.Softplus()
        )

    def _stage(self, in_c, out_c, blocks, stride):
        layers = [ResidualAttentionBlock(in_c, out_c, stride)] + [ResidualAttentionBlock(out_c, out_c, 1) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x)))))).squeeze(1)

class SqrtScaledMSELoss(nn.Module):
    """DSTFS: L = MSE(sqrt(pred/s), sqrt(target/s)). Amplifica el error en tiempos pequeños (más difíciles)."""
    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s = CONFIG["time_scale"]
        return nn.functional.mse_loss(torch.sqrt((p / s).clamp_min(1e-6)), torch.sqrt((t / s).clamp_min(1e-6)))

# --- ENTRENAMIENTO ---

def split_by_sequence(df: pd.DataFrame, fraction=0.8, seed=42):
    """Split por secuencias evita data leakage (imágenes del mismo rastro en train y val)."""
    seqs = df["sequence_id"].drop_duplicates().to_numpy()
    np.random.default_rng(seed).shuffle(seqs)
    train_seqs = set(seqs[:max(1, int(len(seqs) * fraction))])
    return df.index[df["sequence_id"].isin(train_seqs)].tolist(), df.index[~df["sequence_id"].isin(train_seqs)].tolist()

def eval_metrics(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float, float]:
    model.eval()
    abs_err, sq_err, n, err60 = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            abs_err += (p - y).abs().sum().item()
            sq_err += ((p - y)**2).sum().item()
            err60 += (p - y).abs().gt(60).sum().item()
            n += x.size(0)
    return abs_err / n, (sq_err / n)**0.5, err60 / n

def main():
    dev = CONFIG["device"]
    train_ds_full = ThermalTraceDataset(is_train=True)
    val_ds_full = ThermalTraceDataset(is_train=False)
    
    t_idx, v_idx = split_by_sequence(train_ds_full.df)
    train_loader = DataLoader(Subset(train_ds_full, t_idx), batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True)
    val_loader = DataLoader(Subset(val_ds_full, v_idx), batch_size=CONFIG["batch_size"], pin_memory=True)
    
    # Para evaluar el error en train
    train_eval_loader = DataLoader(Subset(train_ds_full, t_idx), batch_size=CONFIG["batch_size"], pin_memory=True)

    model = ThermalDepartureTimeNet().to(dev)
    crit = SqrtScaledMSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    print(f"Modelo: {sum(p.numel() for p in model.parameters()):,} params | Dataset: {len(t_idx)} train / {len(v_idx)} val | Device: {dev}")
    
    best_mae, no_imp = float("inf"), 0
    for ep in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
            n += x.size(0)

        v_mae, v_rmse, v_e60 = eval_metrics(model, val_loader, dev)
        t_mae, _, _ = eval_metrics(model, train_eval_loader, dev)
        
        is_best = v_mae < best_mae - CONFIG["min_delta"]
        if is_best: 
            best_mae, no_imp = v_mae, 0
            torch.save(model.state_dict(), "DSTFS_adapted_best.pt")
        else: 
            no_imp += 1

        print(f"Ep {ep:03d} | Loss: {train_loss/n:.4f} | TrMAE: {t_mae:5.2f}s | ValMAE: {v_mae:5.2f}s | RMSE: {v_rmse:5.2f}s | Err>60s: {v_e60:.2f} {'*' if is_best else ''}")
        
        if no_imp >= CONFIG["patience"]: 
            print(f"Early stop. Best Val MAE: {best_mae:.2f}s")
            break

if __name__ == "__main__":
    main()
