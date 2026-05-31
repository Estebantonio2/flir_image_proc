from __future__ import annotations
from typing import cast
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def _build_train_transform() -> transforms.Compose:
    """Aumento de datos térmicos multiescala y rotaciones."""
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
    """Transformaciones estándar de validación sin distorsión geométrica."""
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
    def __init__(self, metadata_csv="processed_data/metadata.csv", is_train=True, min_time_s=0.0):
        self.root = Path(metadata_csv).parent
        self.transform = _build_train_transform() if is_train else _build_val_transform()
        df = pd.read_csv(metadata_csv).dropna(subset=["thermal_path", "sequence_id", "label_time_s"])
        self.df = df[df["label_time_s"].astype(float) > min_time_s].reset_index(drop=True)
        self.df = self.df[[self._resolve(p).exists() for p in self.df["thermal_path"]]].reset_index(drop=True)

    def __len__(self) -> int: 
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[i]
        thermal = np.load(self._resolve(row["thermal_path"])).astype(np.float32)
        
        # Escalar a uint8 para compatibilidad con transformaciones PIL
        tmin, tmax = thermal.min(), thermal.max()
        th_u8 = ((thermal - tmin) / (tmax - tmin) * 255).astype(np.uint8) if tmax - tmin > 1e-6 else np.zeros_like(thermal, dtype=np.uint8)
        x = cast(torch.Tensor, self.transform(th_u8))
        
        # DSTFS: Normalización lineal [-1, 1] por muestra individual
        xmin, xmax = x.min(), x.max()
        if xmax - xmin > 1e-6: 
            x = (x - xmin) / (xmax - xmin) * 2.0 - 1.0
        else: 
            x = torch.zeros_like(x)

        return x, torch.tensor(float(row["label_time_s"]), dtype=torch.float32)

    def _resolve(self, p: str) -> Path:
        path = Path(str(p).replace("\\", "/"))
        return path if path.is_absolute() else self.root / path
