from __future__ import annotations
from typing import cast
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def _build_train_transform() -> transforms.Compose:
    """Aumento de datos térmicos multiescala y rotaciones robustas."""
    return transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10), 
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((112, 112)), 
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
    2. Vector tabular con 8 variables continuas normalizadas y 2 categóricas de superficie codificadas (one-hot).
    3. Etiqueta de tiempo de partida escalada.
    """
    means: np.ndarray
    stds: np.ndarray

    def __init__(
        self, 
        metadata_csv="processed_data/metadata_train.csv", 
        is_train=True, 
        min_time_s=0.0, 
        time_scale=30.0,
        indices=None,
        means=None,
        stds=None
    ):
        self.root = Path(metadata_csv).parent
        self.transform = _build_train_transform() if is_train else _build_val_transform()
        self.time_scale = time_scale
        
        # Cargar y pre-filtrar
        df = pd.read_csv(metadata_csv).dropna(
            subset=["thermal_path", "sequence_id", "t_seconds", "ambient_temp_C", "ambient_rh_pct", "surface"]
        )
        df = df[df["t_seconds"].astype(float) > min_time_s].reset_index(drop=True)
        df = df[[self._resolve(p).exists() for p in df["thermal_path"]]].reset_index(drop=True)
        
        # Filtrar por índices de fold si es necesario
        if indices is not None:
            self.df = df.iloc[indices].reset_index(drop=True)
        else:
            self.df = df
            
        self.continuous_cols = [
            "ambient_temp_C", "ambient_rh_pct", "img_tmax_C", "img_tstd_C",
            "delta_tmean_C", "delta_tstd_C", "hot_area_px_p95", "hot_delta_tmean_C_p95"
        ]
        
        # Calcular medias y stds en entrenamiento o heredar en validación/test
        if means is None or stds is None:
            self.means = np.asarray(self.df[self.continuous_cols].mean().to_numpy(), dtype=np.float32)
            stds_arr = np.asarray(self.df[self.continuous_cols].std().to_numpy(), dtype=np.float32)
            # Reemplazar ceros o std muy pequeños para evitar divisiones por cero
            self.stds = np.where(stds_arr < 1e-6, np.float32(1.0), stds_arr).astype(np.float32)
        else:
            self.means = np.asarray(means, dtype=np.float32)
            self.stds = np.asarray(stds, dtype=np.float32)

    def __len__(self) -> int: 
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[i]
        thermal = np.load(self._resolve(row["thermal_path"])).astype(np.float32)
        
        # 1. Procesamiento visual
        tmin, tmax = float(thermal.min()), float(thermal.max())
        th_u8 = ((thermal - tmin) / (tmax - tmin) * 255).astype(np.uint8) if tmax - tmin > 1e-6 else np.zeros_like(thermal, dtype=np.uint8)
        x_img = cast(torch.Tensor, self.transform(th_u8))
        
        # Normalización lineal [-1, 1] de la imagen
        xmin, xmax = float(x_img.min()), float(x_img.max())
        if xmax - xmin > 1e-6: 
            x_img = (x_img - xmin) / (xmax - xmin) * 2.0 - 1.0
        else: 
            x_img = torch.zeros_like(x_img)

        # 2. Procesamiento Tabular
        raw_cont = np.asarray(row[self.continuous_cols].to_numpy(), dtype=np.float32)
        cont_vals = (raw_cont - self.means) / self.stds
        
        is_wood = 1.0 if str(row["surface"]).lower() == "wood" else 0.0
        is_glass = 1.0 if str(row["surface"]).lower() == "glass" else 0.0
        
        # Vector final tabular: 8 continuas normalizadas + 2 binarias de superficie = 10 dimensiones
        x_tab = torch.cat([
            torch.tensor(cont_vals, dtype=torch.float32),
            torch.tensor([is_wood, is_glass], dtype=torch.float32)
        ], dim=0)

        # 3. Escalado de etiqueta de tiempo
        t_scaled = float(row["t_seconds"]) / self.time_scale
        
        return x_img, x_tab, torch.tensor(t_scaled, dtype=torch.float32)

    def _resolve(self, p: str) -> Path:
        path = Path(str(p).replace("\\", "/"))
        return path if path.is_absolute() else self.root / path
