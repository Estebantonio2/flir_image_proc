from __future__ import annotations
from typing import cast
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def _build_train_transform() -> transforms.Compose:
    """Aumento de datos suavizado para el backbone visual de alta eficiencia."""
    return transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((112, 112)),
        transforms.RandomRotation(5),
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
    3. Etiqueta de tiempo de partida escalada.
    """
    def __init__(self, metadata_csv="processed_data/metadata.csv", is_train=True, min_time_s=0.0, time_scale=30.0):
        self.root = Path(metadata_csv).parent
        self.transform = _build_train_transform() if is_train else _build_val_transform()
        self.time_scale = time_scale
        df = pd.read_csv(metadata_csv).dropna(subset=["thermal_path", "sequence_id", "label_time_s", "ambient_temp_C", "ambient_rh_pct", "surface"])
        self.df = df[df["label_time_s"].astype(float) > min_time_s].reset_index(drop=True)
        self.df = self.df[[self._resolve(p).exists() for p in self.df["thermal_path"]]].reset_index(drop=True)

    def __len__(self) -> int: 
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[i]
        thermal = np.load(self._resolve(row["thermal_path"])).astype(np.float32)
        
        # 1. Procesamiento visual
        tmin, tmax = thermal.min(), thermal.max()
        th_u8 = ((thermal - tmin) / (tmax - tmin) * 255).astype(np.uint8) if tmax - tmin > 1e-6 else np.zeros_like(thermal, dtype=np.uint8)
        x_img = cast(torch.Tensor, self.transform(th_u8))
        
        # Normalización lineal [-1, 1] de la imagen
        xmin, xmax = x_img.min(), x_img.max()
        if xmax - xmin > 1e-6: 
            x_img = (x_img - xmin) / (xmax - xmin) * 2.0 - 1.0
        else: 
            x_img = torch.zeros_like(x_img)

        # 2. Procesamiento Tabular de variables ambientales
        temp_norm = (float(row["ambient_temp_C"]) - 15.0) / 20.0
        rh_norm = (float(row["ambient_rh_pct"]) - 40.0) / 50.0
        
        is_wood = 1.0 if str(row["surface"]).lower() == "wood" else 0.0
        is_glass = 1.0 if str(row["surface"]).lower() == "glass" else 0.0
        
        x_tab = torch.tensor([temp_norm, rh_norm, is_wood, is_glass], dtype=torch.float32)

        # 3. Escalado de etiqueta de tiempo
        t_scaled = float(row["label_time_s"]) / self.time_scale
        
        return x_img, x_tab, torch.tensor(t_scaled, dtype=torch.float32)

    def _resolve(self, p: str) -> Path:
        path = Path(str(p).replace("\\", "/"))
        return path if path.is_absolute() else self.root / path
