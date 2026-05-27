"""
MTDE-Net.py
Multimodal Thermal Decay Estimation Network (MTDE-Net)
Implementación multimodal de alta eficiencia (~750K parámetros) para estimar el tiempo
transcurrido del rastro térmico, fusionando la imagen térmica con variables del entorno.
"""

# ==============================================================================
# 1. IMPORTACIONES
# ==============================================================================
from __future__ import annotations
from typing import cast
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# ==============================================================================
# 2. SEMILLA GLOBAL Y REPRODUCIBILIDAD
# ==============================================================================
def set_seed(seed: int = 42):
    """Establece la semilla para garantizar reproducibilidad en todas las corridas."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# 3. CONFIGURACIÓN GENERAL (HIPERPARÁMETROS)
# ==============================================================================
CONFIG = {
    "epochs": 120,
    "patience": 15,
    "min_delta": 1.0,
    "batch_size": 16,
    "lr": 0.0005,          # SGD lr inicial (usado por AdamW como lr de partida)
    "weight_decay": 0.0005, # Regularización L2
    "momentum": 0.9,       # SGD Momentum
    "min_time_s": 0.0,
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "time_scale": 30.0,    # Escalar etiquetas de tiempo
}

# ==============================================================================
# 4. PREPROCESAMIENTO Y DATASET
# ==============================================================================
def _build_train_transform() -> transforms.Compose:
    """Aumento de datos suavizado para el backbone visual de alta eficiencia."""
    return transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((112, 112)),
        transforms.RandomRotation(5),          # Rotación sutil para estabilizar la red ligera
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
    ])

def _build_val_transform() -> transforms.Compose:
    """Transformaciones estándar de validación sin distorsión geométrica."""
    return transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((112, 112)), 
        transforms.ToTensor()
    ])

class MultimodalThermalDataset(Dataset):
    """
    Dataset Multimodal que carga:
    1. Imagen térmica normalizada a [-1, 1].
    2. Vector tabular de variables ambientales (Temp, Humedad, Superficie One-Hot).
    3. Etiqueta de tiempo de partida escalada (/30.0).
    """
    def __init__(self, metadata_csv="processed_data/metadata.csv", is_train=True):
        self.root = Path("processed_data")
        self.transform = _build_train_transform() if is_train else _build_val_transform()
        df = pd.read_csv(metadata_csv).dropna(subset=["thermal_path", "sequence_id", "label_time_s", "ambient_temp_C", "ambient_rh_pct", "surface"])
        self.df = df[df["label_time_s"].astype(float) > CONFIG["min_time_s"]].reset_index(drop=True)
        self.df = self.df[[self._resolve(p).exists() for p in self.df["thermal_path"]]].reset_index(drop=True)

    def __len__(self) -> int: return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[i]
        thermal = np.load(self._resolve(row["thermal_path"])).astype(np.float32)
        
        # 1. Procesamiento visual
        tmin, tmax = thermal.min(), thermal.max()
        th_u8 = ((thermal - tmin) / (tmax - tmin) * 255).astype(np.uint8) if tmax - tmin > 1e-6 else np.zeros_like(thermal, dtype=np.uint8)
        x_img = cast(torch.Tensor, self.transform(th_u8))
        
        # Normalización lineal [-1, 1] de imagen
        xmin, xmax = x_img.min(), x_img.max()
        if xmax - xmin > 1e-6: x_img = (x_img - xmin) / (xmax - xmin) * 2.0 - 1.0
        else: x_img = torch.zeros_like(x_img)

        # 2. Procesamiento Tabular de variables ambientales (calibración física)
        # Normalización lineal de Temperatura (15-35C) y Humedad (40-90%)
        temp_norm = (float(row["ambient_temp_C"]) - 15.0) / 20.0
        rh_norm = (float(row["ambient_rh_pct"]) - 40.0) / 50.0
        
        # Codificación One-Hot de la superficie (wood vs. glass)
        is_wood = 1.0 if str(row["surface"]).lower() == "wood" else 0.0
        is_glass = 1.0 if str(row["surface"]).lower() == "glass" else 0.0
        
        x_tab = torch.tensor([temp_norm, rh_norm, is_wood, is_glass], dtype=torch.float32)

        # 3. Escalado de etiqueta de tiempo
        t_scaled = float(row["label_time_s"]) / CONFIG["time_scale"]
        
        return x_img, x_tab, torch.tensor(t_scaled, dtype=torch.float32)

    def _resolve(self, p: str) -> Path:
        path = Path(str(p).replace("\\", "/"))
        return path if path.is_absolute() else self.root / path

# ==============================================================================
# 5. MÓDULOS DE ATENCIÓN Y ACTIVACIONES DE UMBRAL SUAVE
# ==============================================================================
class SoftThresholdPReLU(nn.Module):
    """
    MTDE-Net: Umbral Suave PReLU (SPRelu) dinámico y adaptativo.
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
    MTDE-Net: Channel Attention (CA) con Spatial Pyramid Pooling (SPP) multiescala.
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
    MTDE-Net: Spatial Attention (SA) mediante convolución 7x7 sobre la concatenación
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
    MTDE-Net: Bloque residual que combina SPRelu (solo en stride=1 para preservar ruido espacial)
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

# ==============================================================================
# 6. DEFINICIÓN DE LA RED PRINCIPAL
# ==============================================================================
class MTDE_Net(nn.Module):
    """
    Multimodal Thermal Decay Estimation Network (MTDE-Net).
    Integra una rama visual Lite-DSTFS de alta eficiencia y una rama tabular ambiental
    mediante Fusión Tardía (Late Fusion).
    Parámetros totales: ~1.32 millones.
    """
    def __init__(self, tabular_dim=4):
        super().__init__()
        # 1. Rama Visual Lite (Canales reducidos y 1 bloque por etapa para prevenir sobreajuste)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, 2, 3, bias=False), 
            nn.BatchNorm2d(32), 
            nn.PReLU(32)
        )
        self.stage1 = ResidualAttentionBlock(32, 32, stride=1)  # 56x56x32
        self.stage2 = ResidualAttentionBlock(32, 64, stride=2)  # 28x28x64
        self.stage3 = ResidualAttentionBlock(64, 128, stride=2) # 14x14x128
        self.stage4 = ResidualAttentionBlock(128, 256, stride=2) # 7x7x256
        self.visual_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten() # Salida: B x 256
        )
        
        # 2. Rama Tabular de Contexto Climático/Superficie (MLP compacto)
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.Linear(32, 64),
            nn.PReLU(64) # Salida: B x 64
        )
        
        # 3. Cabezal de Regresión Multimodal (Late Fusion)
        # Vector concatenado: 256 (visual) + 64 (tabular) = 320 dimensiones
        self.multimodal_head = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Softplus() # Garantiza estimación de tiempo >= 0
        )

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        feat_vis = self.visual_gap(self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x_img))))))
        feat_tab = self.tabular_branch(x_tab)
        
        # Fusión tardía por concatenación
        feat_fused = torch.cat([feat_vis, feat_tab], dim=1) # B x 320
        return self.multimodal_head(feat_fused).squeeze(1)

# ==============================================================================
# 7. PÉRDIDA Y MÉTRICAS DE EVALUACIÓN
# ==============================================================================
class SqrtScaledMSELoss(nn.Module):
    """Pérdida L = MSE(sqrt(pred), sqrt(target)) para amplificar error en la fase inicial de enfriamiento rápido."""
    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(torch.sqrt(p.clamp_min(1e-6)), torch.sqrt(t.clamp_min(1e-6)))

def eval_metrics(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float, float, float, float]:
    """Calcula MAE, RMSE y porcentaje de errores bajo 60s/120s desescalando predicciones a segundos reales."""
    model.eval()
    abs_err, sq_err, n, err60, err120 = 0.0, 0.0, 0, 0, 0
    scale = CONFIG["time_scale"]
    with torch.no_grad():
        for x_img, x_tab, y in loader:
            x_img, x_tab, y = x_img.to(device), x_tab.to(device), y.to(device)
            p_s = model(x_img, x_tab) * scale
            y_s = y * scale
            abs_err += (p_s - y_s).abs().sum().item()
            sq_err += ((p_s - y_s)**2).sum().item()
            err60 += (p_s - y_s).abs().gt(60).sum().item()
            err120 += (p_s - y_s).abs().gt(120).sum().item()
            n += x_img.size(0)
    return abs_err / n, (sq_err / n)**0.5, (1.0 - err60 / n) * 100, (1.0 - err120 / n) * 100, n

# ==============================================================================
# 8. EJECUCIÓN PRINCIPAL (ENTRENAMIENTO)
# ==============================================================================
def split_by_sequence(df: pd.DataFrame, fraction=0.8, seed=42):
    """Evita data leakage separando estrictamente por secuencias físicas de rastro térmico."""
    seqs = df["sequence_id"].drop_duplicates().to_numpy()
    np.random.default_rng(seed).shuffle(seqs)
    train_seqs = set(seqs[:max(1, int(len(seqs) * fraction))])
    return df.index[df["sequence_id"].isin(train_seqs)].tolist(), df.index[~df["sequence_id"].isin(train_seqs)].tolist()

def main():
    # Establecer la semilla global para garantizar reproducibilidad absoluta
    set_seed(42)
    
    dev = CONFIG["device"]
    train_ds_full = MultimodalThermalDataset(is_train=True)
    val_ds_full = MultimodalThermalDataset(is_train=False)
    
    t_idx, v_idx = split_by_sequence(train_ds_full.df)
    train_loader = DataLoader(Subset(train_ds_full, t_idx), batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(Subset(val_ds_full, v_idx), batch_size=CONFIG["batch_size"])
    train_eval_loader = DataLoader(Subset(train_ds_full, t_idx), batch_size=CONFIG["batch_size"])

    model = MTDE_Net().to(dev)
    crit = SqrtScaledMSELoss()
    
    # Optimizador AdamW adaptativo para fusionar ramas de forma balanceada
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 60, 80], gamma=0.8)

    print(f"Modelo MTDE-Net Multimodal: {sum(p.numel() for p in model.parameters()):,} params | Dataset: {len(t_idx)} train / {len(v_idx)} val | Device: {dev}")
    
    best_mae, no_imp = float("inf"), 0
    for ep in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_loss, n = 0.0, 0
        for x_img, x_tab, y in train_loader:
            x_img, x_tab, y = x_img.to(dev), x_tab.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x_img, x_tab), y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x_img.size(0)
            n += x_img.size(0)

        scheduler.step()

        # Evaluación en segundos reales desescalados
        v_mae, v_rmse, v_a60, v_a120, _ = eval_metrics(model, val_loader, dev)
        t_mae, _, _, _, _ = eval_metrics(model, train_eval_loader, dev)
        
        is_best = v_mae < best_mae - CONFIG["min_delta"]
        if is_best: 
            best_mae, no_imp = v_mae, 0
            torch.save(model.state_dict(), "MTDE_Net_best.pt")
        else: 
            no_imp += 1

        print(f"Ep {ep:03d} | Loss: {train_loss/n:.4f} | TrMAE: {t_mae:5.2f}s | ValMAE: {v_mae:5.2f}s | RMSE: {v_rmse:5.2f}s | Acc60: {v_a60:.2f}% | Acc120: {v_a120:.2f}% {'*' if is_best else ''}")
        
        if no_imp >= CONFIG["patience"]: 
            print(f"Early stop. Best Val MAE: {best_mae:.2f}s")
            break

if __name__ == "__main__":
    main()
