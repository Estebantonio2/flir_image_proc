"""
DSTFS-adapted.py
Implementación de alta fidelidad de la rama temporal de DSTFS (Deep Soft Threshold Feature Separation)
adaptada para tarea única de estimación temporal, corrigiendo la resolución espacial,
el umbral suave dinámico y el optimizador original según el estudio.
"""
from __future__ import annotations
from typing import cast
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# --- CONFIGURACIÓN PRINCIPAL ---
CONFIG = {
    "epochs": 120,
    "patience": 15,
    "min_delta": 1.0,
    "batch_size": 16,
    "lr": 0.0005,          # DSTFS: SGD lr inicial de 0.0005
    "weight_decay": 0.0005, # DSTFS: Regularización L2 de 0.0005
    "momentum": 0.9,       # DSTFS: SGD Momentum de 0.9
    "min_time_s": 0.0,
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "time_scale": 30.0,    # DSTFS: Factor de preprocesamiento de escala para etiquetas de tiempo
}

# --- DATASET Y AUGMENTATION ---
def _build_train_transform() -> transforms.Compose:
    # DSTFS: Replicación de aumento de datos térmicos multiescala y rotaciones
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
    Dataset para imágenes térmicas brutas (sin deltaT) para forzar al modelo 
    a aprender la física de disipación directamente del rastro térmico absoluto.
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
        
        # Escalar a uint8 para compatibilidad con transformaciones PIL
        tmin, tmax = thermal.min(), thermal.max()
        th_u8 = ((thermal - tmin) / (tmax - tmin) * 255).astype(np.uint8) if tmax - tmin > 1e-6 else np.zeros_like(thermal, dtype=np.uint8)
        x = cast(torch.Tensor, self.transform(th_u8))
        
        # DSTFS: Normalización lineal [-1, 1] por muestra individual para mitigar variaciones del fondo térmico
        xmin, xmax = x.min(), x.max()
        if xmax - xmin > 1e-6: x = (x - xmin) / (xmax - xmin) * 2.0 - 1.0
        else: x = torch.zeros_like(x)

        # DSTFS: Escalar las etiquetas de tiempo (dividir por 30) para evitar fallas de convergencia por gradientes explosivos
        t_scaled = float(row["label_time_s"]) / CONFIG["time_scale"]
        return x, torch.tensor(t_scaled, dtype=torch.float32)

    def _resolve(self, p: str) -> Path:
        path = Path(str(p).replace("\\", "/"))
        return path if path.is_absolute() else self.root / path

# --- COMPONENTES DEL MODELO (DSTFS) ---

class SoftThresholdPReLU(nn.Module):
    """
    DSTFS: Umbral Suave PReLU (SPRelu) dinámico y adaptativo.
    Calcula un umbral t dependiente de la muestra mediante GAP y un MLP de 2 capas.
    Esto permite filtrar ruido débil e inestable acumulado en el rastro térmico.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.prelu = nn.PReLU(channels)
        # Dos capas FC implementadas mediante Conv2d 1x1 para conservar dimensiones espaciales fácilmente
        hidden = max(channels // 16, 4)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Se aplica la activación no lineal inicial PReLU
        x_act = self.prelu(x)
        # 1. Operación de valor absoluto
        abs_x = torch.abs(x_act)
        # 2. GAP (Global Average Pooling) para extraer el promedio espacial
        g = torch.mean(abs_x, dim=(2, 3), keepdim=True)
        # 3. Paso por dos capas FC con Sigmoid al final
        scale = self.sigmoid(self.fc2(self.relu(self.fc1(g))))
        # 4. Multiplicación de la escala (0,1) por el GAP original para obtener el umbral t
        t = scale * g
        # 5. Activación Soft-thresholding: sign(x_act) * max(0, |x_act| - t)
        return torch.sign(x_act) * torch.relu(abs_x - t)

class SPPChannelAttention(nn.Module):
    """
    DSTFS: Channel Attention (CA) con Spatial Pyramid Pooling (SPP) multiescala.
    Utiliza pooling promedio y máximo en paralelo en sub-regiones para capturar difusión térmica amplia.
    """
    def __init__(self, channels: int, pool_scales=(1, 2, 4)):
        super().__init__()
        self.scales = pool_scales
        hidden = max(channels // 16, 4)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size(2), x.size(3)
        spp_feats = []
        for scale in self.scales:
            # División espacial para pooling en rejillas
            stride = (h // scale, w // scale)
            kernel = (h // scale + (h % scale > 0), w // scale + (w % scale > 0))
            avg_p = nn.functional.adaptive_avg_pool2d(nn.functional.avg_pool2d(x, kernel_size=kernel, stride=stride), 1)
            max_p = nn.functional.adaptive_max_pool2d(nn.functional.max_pool2d(x, kernel_size=kernel, stride=stride), 1)
            spp_feats.append(avg_p + max_p)
        # Promedio y excitación de los canales
        spp_sum = sum(spp_feats) / len(self.scales)
        return x * torch.sigmoid(self.fc2(self.relu(self.fc1(spp_sum))))

class SpatialAttention(nn.Module):
    """
    DSTFS: Spatial Attention (SA) mediante convolución 7x7 sobre la concatenación
    de promedios y máximos espaciales para enfocar las regiones de disipación de calor.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class ResidualAttentionBlock(nn.Module):
    """
    DSTFS: Bloque residual que combina SPRelu (solo en stride=1 para preservar ruido espacial)
    con atención dual (SA + CA con SPP) fusionada por SUMA en paralelo sobre el residuo.
    """
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm2d(out_c), nn.BatchNorm2d(out_c)
        # DSTFS: Particularidad crítica - SPRelu solo se añade en stride=1
        self.act1 = SoftThresholdPReLU(out_c) if stride == 1 else nn.PReLU(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.att_c, self.att_s = SPPChannelAttention(out_c), SpatialAttention()
        self.act2 = nn.PReLU(out_c)
        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride, bias=False), 
            nn.BatchNorm2d(out_c)
        ) if stride != 1 or in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.act1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        # DSTFS: Fusión en paralelo por suma de atenciones en lugar de CBAM secuencial
        att_res = self.att_c(res) + self.att_s(res)
        return self.act2(att_res + self.skip(x))

class ThermalDepartureTimeNet(nn.Module):
    """
    DSTFS Temporal (Tarea única): Stem -> 4 Stages -> Head.
    Salida de Stage 3 exacta a 14x14x256 al remover el MaxPool de stem, preservando detalles.
    """
    def __init__(self):
        super().__init__()
        # DSTFS: Se elimina MaxPool del stem para preservar la resolución espacial en imágenes térmicas
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.PReLU(64)
        )
        self.stage1 = self._stage(64, 64, 2, 1)  # 56x56x64
        self.stage2 = self._stage(64, 128, 2, 2) # 28x28x128
        self.stage3 = self._stage(128, 256, 2, 2) # 14x14x256 (Resolución de salida del backbone)
        self.stage4 = self._stage(256, 512, 2, 2) # 7x7x512 (Rama Temporal)
        
        # DSTFS Clasificador A: Cabeza de regresión de dos capas lineales con 512 neuronas iniciales
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(), 
            nn.Dropout(0.25), 
            nn.Linear(512, 512), 
            nn.PReLU(512), 
            nn.Linear(512, 1), 
            nn.Softplus() # Softplus garantiza que el tiempo predicho sea estrictamente no negativo
        )

    def _stage(self, in_c, out_c, blocks, stride):
        layers = [ResidualAttentionBlock(in_c, out_c, stride)] + [ResidualAttentionBlock(out_c, out_c, 1) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x)))))).squeeze(1)

class SqrtScaledMSELoss(nn.Module):
    """
    DSTFS: Pérdida temporal L = MSE(sqrt(pred), sqrt(target)).
    La raíz cuadrada amplifica el error en tiempos pequeños (capturando mejor la fase inicial de enfriamiento rápido).
    """
    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(torch.sqrt(p.clamp_min(1e-6)), torch.sqrt(t.clamp_min(1e-6)))

# --- ENTRENAMIENTO Y EVALUACIÓN ---

def split_by_sequence(df: pd.DataFrame, fraction=0.8, seed=42):
    """Garantiza separación estricta por secuencias de rastro para evitar fuga de información biográfica/tiempo."""
    seqs = df["sequence_id"].drop_duplicates().to_numpy()
    np.random.default_rng(seed).shuffle(seqs)
    train_seqs = set(seqs[:max(1, int(len(seqs) * fraction))])
    return df.index[df["sequence_id"].isin(train_seqs)].tolist(), df.index[~df["sequence_id"].isin(train_seqs)].tolist()

def eval_metrics(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float, float, float]:
    """Calcula MAE y RMSE en segundos reales desescalando la predicción y etiquetas (*30.0),
    además de obtener el Accuracy dentro de márgenes de tolerancia de 60 y 120 segundos.
    """
    model.eval()
    abs_err, sq_err, n = 0.0, 0.0, 0
    acc60_count, acc120_count = 0, 0
    scale = CONFIG["time_scale"]
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # Desescalamos a escala real en segundos antes de calcular métricas de reporte
            p_s = model(x) * scale
            y_s = y * scale
            
            # Acumulación de errores absolutos y cuadráticos
            abs_err += (p_s - y_s).abs().sum().item()
            sq_err += ((p_s - y_s)**2).sum().item()
            
            # Calcular exactitud dentro del umbral de 60s y 120s
            errors = (p_s - y_s).abs()
            acc60_count += errors.le(60).sum().item()
            acc120_count += errors.le(120).sum().item()
            
            n += x.size(0)
    
    # Retorna: MAE, RMSE, Acc-60, Acc-120
    return abs_err / n, (sq_err / n)**0.5, acc60_count / n, acc120_count / n

def main():
    dev = CONFIG["device"]
    train_ds_full = ThermalTraceDataset(is_train=True)
    val_ds_full = ThermalTraceDataset(is_train=False)
    
    t_idx, v_idx = split_by_sequence(train_ds_full.df)
    train_loader = DataLoader(Subset(train_ds_full, t_idx), batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True)
    val_loader = DataLoader(Subset(val_ds_full, v_idx), batch_size=CONFIG["batch_size"], pin_memory=True)
    train_eval_loader = DataLoader(Subset(train_ds_full, t_idx), batch_size=CONFIG["batch_size"], pin_memory=True)

    model = ThermalDepartureTimeNet().to(dev)
    crit = SqrtScaledMSELoss()
    
    # DSTFS: Configuración del optimizador SGD con Momentum y regularización L2 exactos
    opt = torch.optim.SGD(model.parameters(), lr=CONFIG["lr"], momentum=CONFIG["momentum"], weight_decay=CONFIG["weight_decay"])
    # DSTFS: Reducción al 80% del LR en los epochs 30, 60 y 80
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 60, 80], gamma=0.8)

    print(f"Modelo DSTFS Temporal: {sum(p.numel() for p in model.parameters()):,} params | Dataset: {len(t_idx)} train / {len(v_idx)} val | Device: {dev}")
    
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

        # scheduler de decaimiento escalonado
        scheduler.step()

        # Las métricas son reportadas en segundos desescalados reales para evaluar el error humano
        v_mae, v_rmse, v_acc60, v_acc120 = eval_metrics(model, val_loader, dev)
        t_mae, _, _, _ = eval_metrics(model, train_eval_loader, dev)

        is_best = v_mae < best_mae - CONFIG["min_delta"]
        if is_best:
            best_mae, no_imp = v_mae, 0
            torch.save(model.state_dict(), "DSTFS_adapted_best.pt")
        else:
            no_imp += 1
        
        # Impresión limpia reportando MAE, RMSE, y los porcentajes de exactitud a 60s y 120s
        print(f"Ep {ep:03d} | Loss: {train_loss/n:.4f} | TrMAE: {t_mae:5.2f}s | ValMAE: {v_mae:5.2f}s | "
            f"RMSE: {v_rmse:5.2f}s | Acc60: {v_acc60:.2%} | Acc120: {v_acc120:.2%} {'*' if is_best else ''}")
        
        if no_imp >= CONFIG["patience"]: 
            print(f"Early stop. Best Val MAE: {best_mae:.2f}s")
            break

if __name__ == "__main__":
    main()
