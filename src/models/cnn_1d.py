import torch
import torch.nn as nn

class Thermal1DCNN(nn.Module):
    """
    Arquitectura CNN 1D clásica y ligera para estimación de decaimiento térmico.
    Procesa secuencias 1D de métricas físicas y estima el tiempo transcurrido del último punto de la ventana.
    """
    def __init__(self, in_channels: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        
        # Convolución 1D multicanal
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding="same"),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.Dropout(dropout),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding="same"),
            nn.BatchNorm1d(64),
            nn.PReLU(64),
            nn.Dropout(dropout)
        )
        
        # Reducción espacial temporal (Global Average Pooling 1D)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Cabezal de regresión con Softplus
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.PReLU(32),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Softplus() # Predicciones estrictamente no negativas
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x esperado: (B, C, W) donde C = in_channels y W = seq_len
        # Si la entrada nos llega como (B, W, C), le hacemos transpose para conv1d
        if x.dim() == 3 and x.size(1) != self.in_channels and x.size(2) == self.in_channels:
            x = x.transpose(1, 2)
            
        feats = self.conv(x)      # Salida: (B, 64, W)
        pooled = self.gap(feats)  # Salida: (B, 64, 1)
        return self.regressor(pooled).squeeze(1)
