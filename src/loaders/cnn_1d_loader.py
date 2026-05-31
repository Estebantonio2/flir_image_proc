from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Thermal1DDataset(Dataset):
    """
    Dataset para el Modelo 4 (CNN 1D).
    Carga de forma directa y ultra-veloz las curvas térmicas 1D precalculadas en metadata.csv,
    generando secuencias por ventanas deslizantes y aplicando normalización estándar robusta.
    """
    def __init__(
        self,
        metadata_csv="processed_data/metadata.csv",
        is_train=True,
        min_time_s=0.0,
        seq_len=5,
        sequence_ids=None,
        means: np.ndarray | None = None,
        stds: np.ndarray | None = None
    ):
        self.root = Path(metadata_csv).parent
        self.seq_len = seq_len
        self.is_train = is_train
        
        # 1. Cargar y filtrar metadatos de las 4 variables físicas seleccionadas
        self.feature_cols = [
            "hot_delta_tmean_C_p95", 
            "hot_area_px_p95", 
            "delta_tmean_C", 
            "img_tmax_C"
        ]
        
        df = pd.read_csv(metadata_csv).dropna(subset=["sequence_id", "label_time_s"] + self.feature_cols)
        df = df[df["label_time_s"].astype(float) > min_time_s].reset_index(drop=True)
        
        # Filtrar por sequence_ids de la partición (previene fugas)
        if sequence_ids is not None:
            df = df[df["sequence_id"].isin(sequence_ids)].reset_index(drop=True)
            
        self.df = df
        
        # 2. Normalización Estándar (Evita data leakage pasando medias y desviaciones del train al val)
        features_raw = self.df[self.feature_cols].to_numpy(dtype=np.float32)
        if is_train:
            self.means = features_raw.mean(axis=0)
            self.stds = features_raw.std(axis=0)
            # Prevenir división por cero en columnas estáticas
            self.stds[self.stds < 1e-6] = 1.0
        else:
            self.means = np.zeros(len(self.feature_cols), dtype=np.float32) if means is None else means
            self.stds = np.ones(len(self.feature_cols), dtype=np.float32) if stds is None else stds
            
        self.normalized_features = (features_raw - self.means) / self.stds
        
        # 3. Construcción de ventanas móviles deslizantes
        self.windows = []
        self._build_windows()

    def _build_windows(self):
        """Agrupa por secuencia y crea ventanas deslizantes 1D dentro de cada secuencia."""
        grouped = self.df.groupby("sequence_id")
        for seq_id, group in grouped:
            sorted_group = group.sort_values("label_time_s")
            indices = sorted_group.index.tolist()
            
            if len(indices) >= self.seq_len:
                for i in range(len(indices) - self.seq_len + 1):
                    self.windows.append(indices[i:i + self.seq_len])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        window_indices = self.windows[i]
        
        # Extraer bloque de características normalizadas: (seq_len, 4)
        x = self.normalized_features[window_indices]
        x_tensor = torch.tensor(x, dtype=torch.float32)
        
        # Transponer para pasar de (seq_len, C) a (C, seq_len) requerido por Conv1d
        x_tensor = x_tensor.transpose(0, 1)
        
        # Target temporal del último elemento de la ventana
        last_row = self.df.iloc[window_indices[-1]]
        y = torch.tensor(float(last_row["label_time_s"]), dtype=torch.float32)
        
        return x_tensor, y
