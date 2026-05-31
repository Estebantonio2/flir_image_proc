import torch
import torch.nn as nn
from src.models.mtde_net import ResidualAttentionBlock

class LiteDSTFSBackbone(nn.Module):
    """
    Extractor visual ligero basado en Lite-DSTFS.
    Extrae un vector de 256 características de cada imagen térmica individual.
    """
    def __init__(self):
        super().__init__()
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
            nn.Flatten()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual_gap(self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x))))))

class ThermalCNNLSTM(nn.Module):
    """
    Arquitectura combinada CNN-LSTM para regresión temporal.
    Procesa secuencias de imágenes térmicas y predice el tiempo transcurrido del último frame.
    """
    def __init__(self, lstm_hidden_dim: int = 128, lstm_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.backbone = LiteDSTFSBackbone()
        
        # LSTM para modelar la evolución temporal de las características visuales
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Cabeza de regresión sobre el último estado oculto
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, 64),
            nn.PReLU(64),
            nn.Linear(64, 1),
            nn.Softplus() # Garantiza que el tiempo predicho sea strictly positivo
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq tiene forma: (B, L, 1, H, W)
        batch_size, seq_len, channels, height, width = x_seq.size()
        
        # 1. Aplanar lote y dimensiones de secuencia para procesar con la CNN 2D
        x_flat = x_seq.view(batch_size * seq_len, channels, height, width)
        feats_flat = self.backbone(x_flat) # Salida: (B * L, 256)
        
        # 2. Re-estructurar a formato secuencial para el LSTM: (B, L, 256)
        feats_seq = feats_flat.view(batch_size, seq_len, -1)
        
        # 3. Paso por la capa recurrente LSTM
        lstm_out, _ = self.lstm(feats_seq) # Salida: (B, L, lstm_hidden_dim)
        
        # 4. Extraer el estado del último paso de tiempo para la regresión
        last_step_feat = lstm_out[:, -1, :] # Salida: (B, lstm_hidden_dim)
        
        return self.regressor(last_step_feat).squeeze(1)
