from __future__ import annotations
from typing import cast
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def _build_train_transform() -> transforms.Compose:
    """Aumento de datos secuencial leve para estabilizar el entrenamiento temporal."""
    return transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((112, 112)),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
    ])

def _build_val_transform() -> transforms.Compose:
    """Transformación de validación estándar."""
    return transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((112, 112)), 
        transforms.ToTensor()
    ])

class ThermalSequenceDataset(Dataset):
    """
    Dataset para generar secuencias de imágenes térmicas (ventanas móviles de tamaño seq_len)
    garantizando que todas las imágenes en una ventana pertenezcan a la misma secuencia física.
    """
    def __init__(
        self, 
        metadata_csv="processed_data/metadata_train.csv", 
        is_train=True, 
        min_time_s=0.0, 
        seq_len=5,
        sequence_ids=None
    ):
        self.root = Path(metadata_csv).parent
        self.transform = _build_train_transform() if is_train else _build_val_transform()
        self.seq_len = seq_len
        
        # Cargar y pre-filtrar metadatos básicos
        df = pd.read_csv(metadata_csv).dropna(subset=["thermal_path", "sequence_id", "t_seconds"])
        df = df[df["t_seconds"].astype(float) > min_time_s].reset_index(drop=True)
        
        # Validar la existencia de archivos npy
        df = df[[self._resolve(p).exists() for p in df["thermal_path"]]].reset_index(drop=True)
        
        # Si se especifica una partición por sequence_ids, filtramos aquí para evitar fugas de información
        if sequence_ids is not None:
            df = df[df["sequence_id"].isin(sequence_ids)].reset_index(drop=True)
            
        self.df = df
        self.windows = []
        self._build_windows()

    def _build_windows(self):
        """Agrupa por secuencia y crea ventanas deslizantes dentro de cada secuencia."""
        grouped = self.df.groupby("sequence_id")
        for seq_id, group in grouped:
            # Ordenar cronológicamente dentro de cada secuencia
            sorted_group = group.sort_values("t_seconds")
            indices = sorted_group.index.tolist()
            
            # Construir ventanas móviles si hay suficientes snapshots
            if len(indices) >= self.seq_len:
                for i in range(len(indices) - self.seq_len + 1):
                    self.windows.append(indices[i:i + self.seq_len])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        window_indices = self.windows[i]
        
        frames = []
        for idx in window_indices:
            row = self.df.iloc[idx]
            thermal = np.load(self._resolve(row["thermal_path"])).astype(np.float32)
            
            # Escalar a uint8 para compatibilidad con transformaciones de PIL
            tmin, tmax = thermal.min(), thermal.max()
            th_u8 = ((thermal - tmin) / (tmax - tmin) * 255).astype(np.uint8) if tmax - tmin > 1e-6 else np.zeros_like(thermal, dtype=np.uint8)
            x = cast(torch.Tensor, self.transform(th_u8))
            
            # Normalización lineal [-1, 1] individual por frame para estabilidad térmica
            xmin, xmax = x.min(), x.max()
            if xmax - xmin > 1e-6:
                x = (x - xmin) / (xmax - xmin) * 2.0 - 1.0
            else:
                x = torch.zeros_like(x)
            frames.append(x)
            
        # Apilar a lo largo de la dimensión temporal -> (seq_len, 1, H, W)
        x_seq = torch.stack(frames, dim=0)
        
        # El tiempo objetivo es el del último fotograma de la ventana
        last_row = self.df.iloc[window_indices[-1]]
        y = torch.tensor(float(last_row["t_seconds"]), dtype=torch.float32)
        
        return x_seq, y

    def _resolve(self, p: str) -> Path:
        path = Path(str(p).replace("\\", "/"))
        return path if path.is_absolute() else self.root / path
