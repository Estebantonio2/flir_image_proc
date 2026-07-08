from __future__ import annotations
from typing import cast
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def _build_train_transform() -> transforms.Compose:
    """Transformación de entrenamiento con aumentos geométricos estándar robustos."""
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
    """Transformación de validación estándar."""
    return transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((112, 112)), 
        transforms.ToTensor()
    ])

class MultimodalThermalSequenceDataset(Dataset):
    """
    Dataset específico para el modelo Multimodal CNN-LSTM.
    Carga secuencias de imágenes térmicas (rama visual) y variables contextuales tabulares (rama tabular)
    filtrando rigurosamente los tiempos negativos (antes de la disipación).
    """
    def __init__(
        self, 
        metadata_csv="processed_data/metadata_train.csv", 
        is_train=True, 
        min_time_s=0.0, 
        seq_len=5,
        sequence_ids=None,
        means=None,
        stds=None
    ):
        self.root = Path(metadata_csv).parent
        self.transform = _build_train_transform() if is_train else _build_val_transform()
        self.seq_len = seq_len
        
        # Cargar y pre-filtrar metadatos
        df = pd.read_csv(metadata_csv).dropna(
            subset=["thermal_path", "sequence_id", "t_seconds", "ambient_temp_C", "ambient_rh_pct", "surface"]
        )
        
        # Filtrar explícitamente valores negativos y menores a min_time_s
        df = df[df["t_seconds"].astype(float) > min_time_s].reset_index(drop=True)
        
        # Validar la existencia de archivos npy
        df = df[[self._resolve(p).exists() for p in df["thermal_path"]]].reset_index(drop=True)
        
        if sequence_ids is not None:
            df = df[df["sequence_id"].isin(sequence_ids)].reset_index(drop=True)
            
        self.df = df
        self.continuous_cols = [
            "ambient_temp_C", "ambient_rh_pct"
        ]
        
        # Calcular estadísticas de normalización o heredarlas
        if means is None or stds is None:
            self.means = np.asarray(self.df[self.continuous_cols].mean().to_numpy(), dtype=np.float32)
            stds_arr = np.asarray(self.df[self.continuous_cols].std().to_numpy(), dtype=np.float32)
            self.stds = np.where(stds_arr < 1e-6, np.float32(1.0), stds_arr).astype(np.float32)
        else:
            self.means = np.asarray(means, dtype=np.float32)
            self.stds = np.asarray(stds, dtype=np.float32)
            
        self.windows = []
        self._build_windows()
        
    def _build_windows(self):
        """Agrupa por secuencia y crea ventanas deslizantes dentro de cada secuencia."""
        grouped = self.df.groupby("sequence_id")
        for seq_id, group in grouped:
            sorted_group = group.sort_values("t_seconds")
            indices = sorted_group.index.tolist()
            if len(indices) >= self.seq_len:
                for i in range(len(indices) - self.seq_len + 1):
                    self.windows.append(indices[i:i + self.seq_len])
                    
    def __len__(self) -> int:
        return len(self.windows)
        
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window_indices = self.windows[i]
        
        # Rama 1: Secuencia de Imágenes Térmicas
        frames = []
        for idx in window_indices:
            row = self.df.iloc[idx]
            thermal = np.load(self._resolve(row["thermal_path"])).astype(np.float32)
            tmin, tmax = thermal.min(), thermal.max()
            th_u8 = ((thermal - tmin) / (tmax - tmin) * 255).astype(np.uint8) if tmax - tmin > 1e-6 else np.zeros_like(thermal, dtype=np.uint8)
            x = cast(torch.Tensor, self.transform(th_u8))
            
            xmin, xmax = x.min(), x.max()
            if xmax - xmin > 1e-6:
                x = (x - xmin) / (xmax - xmin) * 2.0 - 1.0
            else:
                x = torch.zeros_like(x)
            frames.append(x)
            
        x_seq = torch.stack(frames, dim=0)
        
        # Rama 2: Características Tabulares del Contexto (del último frame)
        last_row = self.df.iloc[window_indices[-1]]
        raw_cont = np.asarray(last_row[self.continuous_cols].to_numpy(), dtype=np.float32)
        cont_vals = (raw_cont - self.means) / self.stds
        
        is_wood = 1.0 if str(last_row["surface"]).lower() == "wood" else 0.0
        is_glass = 1.0 if str(last_row["surface"]).lower() == "glass" else 0.0
        
        x_tab = torch.cat([
            torch.tensor(cont_vals, dtype=torch.float32),
            torch.tensor([is_wood, is_glass], dtype=torch.float32)
        ], dim=0)
        
        # Target (Tiempo en segundos reales)
        y = torch.tensor(float(last_row["t_seconds"]), dtype=torch.float32)
        
        return x_seq, x_tab, y
        
    def _resolve(self, p: str) -> Path:
        path = Path(str(p).replace("\\", "/"))
        return path if path.is_absolute() else self.root / path
